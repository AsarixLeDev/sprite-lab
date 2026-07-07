"""Live training dashboard: tail ``train_metrics.jsonl`` files and show progress.

This is a read-only watcher. It never imports torch, never touches training, and
is safe to run in a second terminal while an audit or a single training run is in
flight. It renders either:

* a live terminal dashboard (``rich`` when available), with per-run progress bars
  and unicode loss sparklines, or
* a self-contained, auto-refreshing HTML page with inline SVG loss curves
  (``--html PATH``), so a bored human can watch the curves move in a browser.

The pure helpers (:func:`collect_run_states`, :func:`render_html`,
:func:`render_text_dashboard`) have no side effects and are unit tested.
"""

from __future__ import annotations

import html
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.training.progress import format_duration, sparkline

STALE_AFTER_SECONDS = 20.0
_CHART_POINTS = 240


@dataclass
class RunState:
    name: str
    metrics_path: Path
    step: int = 0
    max_steps: int | None = None
    last_loss: float | None = None
    best_loss: float | None = None
    first_loss: float | None = None
    ema_loss: float | None = None
    val_loss: float | None = None
    elapsed: float | None = None
    rate: float | None = None
    eta: float | None = None
    updated_ago: float | None = None
    status: str = "pending"
    loss_curve: list[float] = field(default_factory=list)
    step_curve: list[int] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        if not self.max_steps:
            return 0.0
        return max(0.0, min(1.0, self.step / self.max_steps))

    @property
    def improvement(self) -> float | None:
        if self.first_loss is None or self.last_loss is None or self.first_loss == 0:
            return None
        return (self.first_loss - self.last_loss) / abs(self.first_loss)


def _downsample(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return list(values)
    stride = len(values) / float(limit)
    return [values[int(i * stride)] for i in range(limit)]


def _read_max_steps(run_dir: Path) -> int | None:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    value = payload.get("max_steps")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_val_loss(run_dir: Path) -> float | None:
    for name in ("train_report.json", "report.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        value = payload.get("val_loss")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _run_state_from_metrics(metrics_path: Path, *, now: float, ema_alpha: float = 0.05) -> RunState:
    run_dir = metrics_path.parent
    state = RunState(name=run_dir.name, metrics_path=metrics_path)
    state.max_steps = _read_max_steps(run_dir)
    state.val_loss = _read_val_loss(run_dir)

    losses: list[float] = []
    steps: list[int] = []
    last_elapsed: float | None = None
    try:
        raw = metrics_path.read_text(encoding="utf-8").splitlines()
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return state
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        loss = record.get("loss")
        step = record.get("step")
        if isinstance(loss, (int, float)) and isinstance(step, (int, float)):
            losses.append(float(loss))
            steps.append(int(step))
        if isinstance(record.get("elapsed_seconds"), (int, float)):
            last_elapsed = float(record["elapsed_seconds"])

    state.updated_ago = max(0.0, now - mtime)
    if not losses:
        state.status = _classify(state, has_data=False)
        return state

    state.step = steps[-1]
    state.last_loss = losses[-1]
    state.first_loss = losses[0]
    state.best_loss = min(losses)
    ema = losses[0]
    for value in losses:
        ema = (1 - ema_alpha) * ema + ema_alpha * value
    state.ema_loss = ema
    state.elapsed = last_elapsed
    if last_elapsed and state.step > 0:
        state.rate = state.step / last_elapsed
        if state.max_steps and state.rate > 0:
            state.eta = max(0.0, (state.max_steps - state.step) / state.rate)
    state.loss_curve = _downsample(losses, _CHART_POINTS)
    state.step_curve = _downsample(steps, _CHART_POINTS)
    state.status = _classify(state, has_data=True)
    return state


def _classify(state: RunState, *, has_data: bool) -> str:
    checkpoint_done = (state.metrics_path.parent / "checkpoint_last.pt").is_file()
    if state.max_steps and state.step >= state.max_steps:
        return "done"
    if checkpoint_done:
        return "done"
    if not has_data:
        return "pending"
    if state.updated_ago is not None and state.updated_ago > STALE_AFTER_SECONDS:
        return "stalled"
    return "running"


def collect_run_states(root: Path, *, now: float | None = None) -> list[RunState]:
    """Find every ``train_metrics.jsonl`` beneath ``root`` and summarise it."""
    root = Path(root)
    now = time.time() if now is None else now
    if (root / "train_metrics.jsonl").is_file():
        paths = [root / "train_metrics.jsonl"]
    else:
        paths = sorted(root.glob("**/train_metrics.jsonl"))
    states = [_run_state_from_metrics(path, now=now) for path in paths]
    states.sort(key=lambda s: (_STATUS_ORDER.get(s.status, 9), s.name))
    return states


_STATUS_ORDER = {"running": 0, "stalled": 1, "pending": 2, "done": 3}
_STATUS_GLYPH = {"running": "▶", "stalled": "■", "pending": "·", "done": "✓"}
_STATUS_GLYPH_ASCII = {"running": ">", "stalled": "!", "pending": ".", "done": "*"}


def _stdout_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "✓█▶".encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def aggregate(states: list[RunState]) -> dict[str, Any]:
    total_steps = sum(s.step for s in states)
    total_target = sum(s.max_steps or 0 for s in states)
    return {
        "runs": len(states),
        "running": sum(1 for s in states if s.status == "running"),
        "done": sum(1 for s in states if s.status == "done"),
        "stalled": sum(1 for s in states if s.status == "stalled"),
        "total_steps": total_steps,
        "total_target": total_target,
        "fraction": (total_steps / total_target) if total_target else 0.0,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_text_dashboard(states: list[RunState], *, ascii_only: bool | None = None) -> str:
    """Plain-text snapshot (used for ``--once`` and non-rich terminals)."""
    if ascii_only is None:
        ascii_only = not _stdout_supports_unicode()
    glyphs = _STATUS_GLYPH_ASCII if ascii_only else _STATUS_GLYPH
    fill_char, empty_char = ("#", "-") if ascii_only else ("█", "░")
    lines = ["Sprite-Lab live training monitor", ""]
    if not states:
        lines.append("(no train_metrics.jsonl found yet)")
        return "\n".join(lines)
    agg = aggregate(states)
    lines.append(
        f"runs={agg['runs']} running={agg['running']} done={agg['done']} "
        f"stalled={agg['stalled']} overall={agg['fraction'] * 100:.1f}%"
    )
    lines.append("")
    for state in states:
        glyph = glyphs.get(state.status, "?")
        pct = int(state.fraction * 100)
        bar_w = 20
        filled = int(state.fraction * bar_w)
        bar = fill_char * filled + empty_char * (bar_w - filled)
        spark = sparkline(state.loss_curve, width=24, ascii_only=ascii_only)
        head = (
            f"{glyph} {state.name:<28} [{bar}] {pct:3d}%  "
            f"{state.step}/{state.max_steps or '?'}"
        )
        stats = (
            f"    loss {_fmt(state.last_loss)}  best {_fmt(state.best_loss)}  "
            f"ema {_fmt(state.ema_loss)}  {(state.rate or 0):.1f} it/s  "
            f"eta {format_duration(state.eta)}  {spark}"
        )
        if state.val_loss is not None:
            stats += f"  val {_fmt(state.val_loss)}"
        lines.append(head)
        lines.append(stats)
    return "\n".join(lines)


def _svg_loss_curve(state: RunState, *, width: int = 320, height: int = 90) -> str:
    values = state.loss_curve
    if len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    lo = min(values)
    hi = max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = []
    for i, value in enumerate(values):
        x = i / (n - 1) * (width - 6) + 3
        y = height - 6 - (value - lo) / span * (height - 12)
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" class="spark">'
        f'<polyline fill="none" stroke="var(--accent)" stroke-width="1.6" points="{polyline}"/>'
        f"</svg>"
    )


def render_html(states: list[RunState], *, refresh_seconds: float = 3.0, title: str = "Sprite-Lab Training") -> str:
    agg = aggregate(states)
    cards = []
    for state in states:
        pct = state.fraction * 100
        improvement = state.improvement
        imp_txt = "n/a" if improvement is None else f"{improvement * 100:.1f}%"
        val_row = "" if state.val_loss is None else f"<div><span>val loss</span><b>{_fmt(state.val_loss)}</b></div>"
        cards.append(
            f"""
      <div class="card status-{state.status}">
        <div class="card-head">
          <span class="dot"></span>
          <h2>{html.escape(state.name)}</h2>
          <span class="badge">{html.escape(state.status)}</span>
        </div>
        {_svg_loss_curve(state)}
        <div class="progress"><div class="fill" style="width:{pct:.1f}%"></div></div>
        <div class="meta">
          <div><span>step</span><b>{state.step} / {state.max_steps or '?'}</b></div>
          <div><span>loss</span><b>{_fmt(state.last_loss)}</b></div>
          <div><span>best</span><b>{_fmt(state.best_loss)}</b></div>
          <div><span>ema</span><b>{_fmt(state.ema_loss)}</b></div>
          <div><span>it/s</span><b>{(state.rate or 0):.1f}</b></div>
          <div><span>eta</span><b>{format_duration(state.eta)}</b></div>
          <div><span>elapsed</span><b>{format_duration(state.elapsed)}</b></div>
          <div><span>improve</span><b>{imp_txt}</b></div>
          {val_row}
        </div>
      </div>"""
        )
    body = "\n".join(cards) if cards else '<p class="empty">Waiting for train_metrics.jsonl…</p>'
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}"/>
<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#0f1220; --panel:#171a2b; --ink:#e7e9f3; --muted:#8b90a8; --accent:#6ea8fe; --ok:#4ade80; --warn:#fbbf24; --idle:#5b6079; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:ui-monospace,Menlo,Consolas,monospace; background:var(--bg); color:var(--ink); }}
  header {{ padding:18px 24px; border-bottom:1px solid #262a44; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
  header h1 {{ font-size:18px; margin:0; }}
  header .sub {{ color:var(--muted); font-size:13px; }}
  .overall {{ margin-left:auto; color:var(--muted); font-size:13px; }}
  main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; padding:20px 24px; }}
  .card {{ background:var(--panel); border:1px solid #262a44; border-radius:12px; padding:14px 16px; }}
  .card-head {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }}
  .card-head h2 {{ font-size:14px; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .badge {{ margin-left:auto; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }}
  .dot {{ width:9px; height:9px; border-radius:50%; background:var(--idle); flex:0 0 auto; }}
  .status-running .dot {{ background:var(--ok); box-shadow:0 0 8px var(--ok); animation:pulse 1.4s infinite; }}
  .status-stalled .dot {{ background:var(--warn); }}
  .status-done .dot {{ background:var(--accent); }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
  .spark {{ width:100%; height:90px; display:block; background:#11142250; border-radius:8px; }}
  .progress {{ height:7px; background:#0c0e18; border-radius:6px; overflow:hidden; margin:10px 0; }}
  .fill {{ height:100%; background:linear-gradient(90deg,var(--accent),#a78bfa); }}
  .meta {{ display:grid; grid-template-columns:repeat(2,1fr); gap:4px 14px; font-size:12px; }}
  .meta div {{ display:flex; justify-content:space-between; }}
  .meta span {{ color:var(--muted); }}
  .empty {{ color:var(--muted); padding:40px; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <span class="sub">auto-refresh {max(1, int(refresh_seconds))}s · generated {generated}</span>
  <span class="overall">{agg['done']}/{agg['runs']} done · {agg['running']} running · overall {agg['fraction'] * 100:.1f}%</span>
</header>
<main>
{body}
</main>
</body>
</html>
"""


def _render_rich(states: list[RunState]) -> Any:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    agg = aggregate(states)
    table = Table(expand=True, header_style="bold cyan")
    table.add_column("run", ratio=3, no_wrap=True)
    table.add_column("progress", ratio=3)
    table.add_column("step", justify="right")
    table.add_column("loss", justify="right")
    table.add_column("best", justify="right")
    table.add_column("it/s", justify="right")
    table.add_column("eta", justify="right")
    table.add_column("loss curve", ratio=2)
    colors = {"running": "green", "stalled": "yellow", "done": "cyan", "pending": "grey58"}
    for state in states:
        color = colors.get(state.status, "white")
        bar_w = 18
        filled = int(state.fraction * bar_w)
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (bar_w - filled), style="grey30")
        bar.append(f" {int(state.fraction * 100):3d}%")
        table.add_row(
            Text(f"{_STATUS_GLYPH.get(state.status, '?')} {state.name}", style=color),
            bar,
            f"{state.step}/{state.max_steps or '?'}",
            _fmt(state.last_loss),
            _fmt(state.best_loss),
            f"{(state.rate or 0):.1f}",
            format_duration(state.eta),
            Text(sparkline(state.loss_curve, width=22), style=color),
        )
    subtitle = (
        f"{agg['done']}/{agg['runs']} done · {agg['running']} running · "
        f"{agg['stalled']} stalled · overall {agg['fraction'] * 100:.1f}%"
    )
    return Panel(table, title="Sprite-Lab live training monitor", subtitle=subtitle, border_style="cyan")


def run_live_monitor(
    root: Path,
    *,
    interval: float = 2.0,
    html_path: Path | None = None,
    once: bool = False,
    use_rich: bool = True,
) -> dict[str, Any]:
    """Watch ``root`` until every run is done (or once), rendering the dashboard.

    Returns the final aggregate summary. Ctrl-C exits cleanly.
    """
    root = Path(root)

    def tick() -> list[RunState]:
        states = collect_run_states(root)
        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(render_html(states, refresh_seconds=interval), encoding="utf-8")
        return states

    if once:
        states = tick()
        print(render_text_dashboard(states))
        return aggregate(states)

    rich_live = None
    if use_rich:
        try:
            from rich.console import Console
            from rich.live import Live

            rich_live = (Console(), Live)
        except ImportError:
            rich_live = None

    def all_done(states: list[RunState]) -> bool:
        return bool(states) and all(s.status in {"done", "stalled"} for s in states)

    try:
        if rich_live is not None:
            console, live_cls = rich_live
            with live_cls(console=console, refresh_per_second=4, screen=False) as live:
                while True:
                    states = tick()
                    live.update(_render_rich(states))
                    if all_done(states):
                        break
                    time.sleep(max(0.2, interval))
        else:
            while True:
                states = tick()
                print("\n" + render_text_dashboard(states), flush=True)
                if all_done(states):
                    break
                time.sleep(max(0.2, interval))
    except KeyboardInterrupt:
        states = collect_run_states(root)
        print("\nmonitor stopped.")
        return aggregate(states)

    final = collect_run_states(root)
    if not rich_live:
        print(render_text_dashboard(final))
    return aggregate(final)

"""Static, offline v3 project reports."""

from __future__ import annotations

import html
import json
import webbrowser
from pathlib import Path
from typing import Any

from spritelab.v3.model import ProjectState, StageStatus
from spritelab.v3.run_state import atomic_write_json, list_runs

_COLORS = {
    StageStatus.COMPLETE: "#2f855a",
    StageStatus.READY: "#2b6cb0",
    StageStatus.NEEDS_REVIEW: "#b7791f",
    StageStatus.BLOCKED: "#c53030",
    StageStatus.FAILED: "#9b2c2c",
    StageStatus.STALE: "#805ad5",
    StageStatus.INCONCLUSIVE: "#718096",
    StageStatus.RUNNING: "#2c7a7b",
    StageStatus.PAUSED: "#b7791f",
    StageStatus.NOT_STARTED: "#718096",
}


def _stage_card(stage: Any) -> str:
    evidence = (
        "".join(
            f"<li><code>{html.escape(item.path)}</code><br><small>SHA-256 {html.escape(item.sha256 or 'unavailable')}</small></li>"
            for item in stage.evidence
        )
        or "<li>No data yet</li>"
    )
    blockers = "".join(f"<li>{html.escape(item)}</li>" for item in stage.blockers) or "<li>None</li>"
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in stage.warnings) or "<li>None</li>"
    metrics = (
        html.escape(json.dumps(stage.metrics, indent=2, sort_keys=True, ensure_ascii=False))
        if stage.metrics
        else "No data yet"
    )
    return f"""
    <article class="card">
      <header><span class="dot" style="background:{_COLORS[stage.status]}"></span>
      <h3>{html.escape(stage.title)}</h3><strong>{stage.status.value}</strong></header>
      <p>{html.escape(stage.explanation)}</p>
      <p><b>Independent audit:</b> {html.escape(stage.audit.value)} &nbsp; <b>Production authorized:</b> {str(stage.production_authorized).lower()}</p>
      <details><summary>Blockers</summary><ul>{blockers}</ul></details>
      <details><summary>Warnings</summary><ul>{warnings}</ul></details>
      <details><summary>Evidence</summary><ul>{evidence}</ul></details>
      <details><summary>Available data</summary><pre>{metrics}</pre></details>
      <p><b>Next:</b> {html.escape(stage.next_action)}<br><code>{html.escape(stage.next_command)}</code></p>
    </article>"""


def generate_report(project: ProjectState, output: Path, *, run: dict[str, Any] | None = None) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    report_json = output / "report.json"
    payload = {
        "schema_version": "spritelab.v3.report.v1",
        "project_state": project.to_dict(),
        "run": run,
    }
    atomic_write_json(report_json, payload)
    complete = sum(stage.status == StageStatus.COMPLETE for stage in project.stages)
    blocked = sum(stage.status in {StageStatus.BLOCKED, StageStatus.FAILED} for stage in project.stages)
    total = len(project.stages)
    percent = round(100 * complete / total) if total else 0
    cards = "\n".join(_stage_card(stage) for stage in project.stages)
    index = output / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(project.project_name)} — Sprite Lab v3 report</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 2rem; line-height: 1.45; }}
.summary,.card {{ border: 1px solid #71809666; border-radius: .7rem; padding: 1rem; margin: 1rem 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:1rem; }}
.card {{ margin:0; overflow-wrap:anywhere; }} .card header {{ display:flex; align-items:center; gap:.6rem; }}
.card h3 {{ flex:1; margin:.2rem 0; }} .dot {{ width:.8rem; height:.8rem; border-radius:50%; }}
.bar {{ background:#71809644; height:1rem; border-radius:1rem; overflow:hidden; }}
.bar span {{ display:block; height:100%; background:#2f855a; width:{percent}%; }}
code,pre {{ font-family:ui-monospace,Consolas,monospace; }} pre {{ white-space:pre-wrap; }}
small {{ opacity:.75; }} @media print {{ details {{ display:block }} }}
</style></head><body>
<h1>Sprite Lab v3 — {html.escape(project.project_name)}</h1>
<p><small>Generated {html.escape(project.generated_at or "unknown")} · Source {html.escape(project.source_commit or "unknown")}</small></p>
<section class="summary"><h2>Project overview</h2><div class="bar"><span></span></div>
<p>{complete} of {total} stages complete · {blocked} blocked or failed.</p>
<p><b>Production authorization is independent of implementation readiness and audit history.</b></p></section>
<h2>Pipeline stages</h2><div class="grid">{cards}</div>
<h2>Dataset, training, and evaluation detail</h2>
<p>The cards above display every metric found in authoritative artifacts. Missing charts and galleries are shown as “No data yet”; this report never fabricates them.</p>
<footer><small>Offline report: no external scripts, fonts, styles, or network resources.</small></footer>
</body></html>""",
        encoding="utf-8",
        newline="\n",
    )
    return index, report_json


def latest_report(runs_dir: Path) -> Path | None:
    for run in list_runs(runs_dir):
        candidate = Path(run["directory"]) / "report" / "index.html"
        if candidate.is_file():
            return candidate
    return None


def open_report(path: Path) -> bool:
    return webbrowser.open(path.resolve().as_uri())

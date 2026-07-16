"""TTY-aware rendering for structured v3 progress snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProgressSnapshot:
    stage: str
    status: str
    current: int
    total: int | None
    elapsed_seconds: float
    message: str
    observations: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _duration(seconds: float) -> str:
    whole = max(0, round(seconds))
    minutes, seconds = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


def render_progress(snapshot: ProgressSnapshot, *, tty: bool, no_color: bool = False, width: int = 20) -> str:
    """Render a snapshot without cursor controls; output stays safe for logs."""
    if not tty:
        total = "?" if snapshot.total is None else str(snapshot.total)
        return (
            f"stage={snapshot.stage} status={snapshot.status} current={snapshot.current} total={total} "
            f"elapsed_seconds={snapshot.elapsed_seconds:.1f} message={snapshot.message}"
        )
    lines = [snapshot.stage, ""]
    if snapshot.total and snapshot.total > 0:
        fraction = min(1.0, max(0.0, snapshot.current / snapshot.total))
        completed = round(width * fraction)
        full, empty = ("#", "-") if no_color else ("█", "░")
        lines.append(
            f"[{full * completed}{empty * (width - completed)}] {fraction:>4.0%}  {snapshot.current:,} / {snapshot.total:,}"
        )
    else:
        lines.append(f"Progress: {snapshot.current:,} completed (total unknown)")
    lines.extend(["", f"Current stage: {snapshot.message}", f"Elapsed: {_duration(snapshot.elapsed_seconds)}"])
    if snapshot.total and snapshot.current > 0 and snapshot.observations >= 3 and snapshot.elapsed_seconds > 0:
        remaining = max(0, snapshot.total - snapshot.current)
        eta = remaining / (snapshot.current / snapshot.elapsed_seconds)
        lines.append(f"ETA:     {_duration(eta)}")
    return "\n".join(lines)

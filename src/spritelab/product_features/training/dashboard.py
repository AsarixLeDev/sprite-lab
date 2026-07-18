"""Backend-neutral live training dashboard derived only from ProductEvents."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from spritelab.product_core import ProductEvent, ProductStatus, validate_finite_json

CHART_METRICS = ("loss", "validation_loss", "learning_rate")
LEARNING_RATE_KEYS = ("learning_rate", "lr")


@dataclass
class SeedProgress:
    seed: int
    status: ProductStatus = ProductStatus.NOT_STARTED
    optimizer_step: int = 0
    total_steps: int | None = None
    loss: float | None = None
    validation_loss: float | None = None
    learning_rate: float | None = None
    gradient_norm: float | None = None
    gpu_utilization: float | None = None
    vram_bytes: int | None = None


@dataclass
class CheckpointState:
    checkpoint: str
    seed: int | None
    optimizer_step: int
    sha256: str | None
    backend_id: str
    remote: bool
    downloaded: bool
    hash_verified: bool
    remote_identity_verified: bool
    safe_resume: bool
    synchronization: str
    verification: str


@dataclass
class DashboardState:
    run_id: str
    backend_id: str
    event_cursor: int = 0
    status: ProductStatus = ProductStatus.NOT_STARTED
    campaign_current: int = 0
    campaign_total: int | None = None
    seeds: dict[int, SeedProgress] = field(default_factory=dict)
    loss_curve: list[dict[str, Any]] = field(default_factory=list)
    validation_loss_curve: list[dict[str, Any]] = field(default_factory=list)
    learning_rate_curve: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[CheckpointState] = field(default_factory=list)
    checkpoint_schedule: list[int] = field(default_factory=list)
    estimated_completion: str | None = None
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    previews: list[dict[str, Any]] = field(default_factory=list)
    terminal_status: str | None = None
    seed_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    job_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    unknown_backend_operation_count: int = 0
    remote_resource_uncertain: bool = False
    may_accrue_cost: bool = False
    shutdown_guidance: str | None = None
    pause_available: bool = False
    resume_available: bool = False
    cancel_available: bool = False

    def apply(self, event: ProductEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError("ProductEvent belongs to a different dashboard run.")
        self.status = event.status
        if event.stage == "campaign":
            self.campaign_current, self.campaign_total = event.current, event.total
        seed_value = event.metrics.get("seed")
        seed = int(seed_value) if isinstance(seed_value, (int, str)) and str(seed_value).lstrip("-").isdigit() else None
        if seed is not None:
            current = self.seeds.setdefault(seed, SeedProgress(seed))
            current.status = event.status
            if event.metrics.get("optimizer_step") is not None:
                current.optimizer_step = int(event.metrics["optimizer_step"])
            elif event.stage == "seed":
                current.optimizer_step = event.current
            if event.metrics.get("total_steps") is not None:
                current.total_steps = int(event.metrics["total_steps"])
            elif event.stage == "seed":
                current.total_steps = event.total
            metric_values = {
                "loss": _finite(event.metrics.get("loss")),
                "validation_loss": _finite(event.metrics.get("validation_loss")),
                "learning_rate": _first_finite(event.metrics, LEARNING_RATE_KEYS),
            }
            for name, value in metric_values.items():
                if value is not None:
                    setattr(current, name, value)
            for name in ("gradient_norm", "gpu_utilization"):
                value = _finite(event.metrics.get(name))
                if value is not None:
                    setattr(current, name, value)
            if isinstance(event.metrics.get("vram_bytes"), int):
                current.vram_bytes = int(event.metrics["vram_bytes"])
            if _is_chart_metric_event(event.metrics):
                for name, curve in (
                    ("loss", self.loss_curve),
                    ("validation_loss", self.validation_loss_curve),
                    ("learning_rate", self.learning_rate_curve),
                ):
                    value = metric_values[name]
                    curve.append({"seed": seed, "step": current.optimizer_step, "value": value})
        if event.event_type == "checkpoint":
            self._checkpoint(event, seed)
        elif event.event_type == "log":
            self.logs.append(event.message)
        elif event.event_type in {"warning", "preview_failed"}:
            self.warnings.append(event.message)
        elif event.event_type == "exploratory_preview":
            self.previews.append(dict(event.metrics))
        if event.metrics.get("resource_state_uncertain") is True:
            self.remote_resource_uncertain = True
            self.shutdown_guidance = str(
                event.metrics.get("shutdown_guidance")
                or "Provider state is uncertain. Open the provider console and stop or terminate the retained resource."
            )
        if event.metrics.get("may_accrue_cost") is True:
            self.may_accrue_cost = True
        seed_outcomes = event.metrics.get("seed_outcomes")
        if isinstance(seed_outcomes, (list, tuple)):
            self.seed_outcomes = {
                str(item.get("run_id")): dict(item)
                for item in seed_outcomes
                if isinstance(item, dict) and isinstance(item.get("run_id"), str)
            }
        job_outcomes = event.metrics.get("job_outcomes")
        if isinstance(job_outcomes, (list, tuple)):
            self.job_outcomes = {
                str(item.get("job_id")): dict(item)
                for item in job_outcomes
                if isinstance(item, dict) and isinstance(item.get("job_id"), str)
            }
        terminal_status = event.metrics.get("terminal_status")
        if isinstance(terminal_status, str) and terminal_status:
            self.terminal_status = terminal_status
        unknown_count = event.metrics.get("unknown_backend_operation_count")
        if type(unknown_count) is int and unknown_count >= 0:
            self.unknown_backend_operation_count = unknown_count
        if event.metrics.get("resource_shutdown_verified") is True:
            self.remote_resource_uncertain = False
            self.may_accrue_cost = False
            self.shutdown_guidance = None
        elif event.metrics.get("resource_state_verified") is True:
            self.remote_resource_uncertain = False
            if event.metrics.get("may_accrue_cost") is not True:
                self.shutdown_guidance = None
        elif event.metrics.get("may_accrue_cost") is True:
            self.may_accrue_cost = True
        if event.metrics.get("estimated_completion"):
            self.estimated_completion = str(event.metrics["estimated_completion"])
        if event.metrics.get("checkpoint_schedule"):
            self.checkpoint_schedule = [int(item) for item in event.metrics["checkpoint_schedule"]]
        self.pause_available = self.status == ProductStatus.RUNNING
        self.resume_available = self.status == ProductStatus.PAUSED and any(
            item.safe_resume is True for item in self.checkpoints
        )
        retained_risk = bool(
            self.job_outcomes
            or self.unknown_backend_operation_count
            or self.remote_resource_uncertain
            or self.may_accrue_cost
        )
        self.cancel_available = self.status in {ProductStatus.RUNNING, ProductStatus.PAUSED} or (
            self.status in {ProductStatus.BLOCKED, ProductStatus.FAILED} and retained_risk
        )

    def _checkpoint(self, event: ProductEvent, seed: int | None) -> None:
        remote = self.backend_id != "local"
        downloaded = event.metrics.get("downloaded") is True
        hash_verified = event.metrics.get("hash_verified") is True
        remote_identity_verified = event.metrics.get("remote_identity_verified") is True
        local_identity_verified = event.metrics.get("identity_verified") is True
        safe = (
            downloaded and hash_verified and remote_identity_verified
            if remote
            else hash_verified and local_identity_verified
        )
        synchronization = (
            "downloaded and verified" if downloaded and hash_verified else ("remote only" if remote else "local")
        )
        verification = "verified" if safe else "not yet safe for resume"
        self.checkpoints.append(
            CheckpointState(
                checkpoint=str(event.metrics.get("checkpoint") or ""),
                seed=seed,
                optimizer_step=int(event.metrics.get("optimizer_step", event.current)),
                sha256=str(event.metrics["sha256"]) if event.metrics.get("sha256") else None,
                backend_id=self.backend_id,
                remote=remote,
                downloaded=downloaded,
                hash_verified=hash_verified,
                remote_identity_verified=remote_identity_verified,
                safe_resume=safe,
                synchronization=synchronization,
                verification=verification,
            )
        )

    @property
    def latest_verified_checkpoint(self) -> CheckpointState | None:
        verified = [item for item in self.checkpoints if item.verification == "verified"]
        return max(verified, key=lambda item: item.optimizer_step, default=None)

    @property
    def last_safe_resume_point(self) -> CheckpointState | None:
        safe = [item for item in self.checkpoints if item.safe_resume is True]
        return max(safe, key=lambda item: item.optimizer_step, default=None)

    def to_dict(self) -> dict[str, Any]:
        latest = self.latest_verified_checkpoint
        safe = self.last_safe_resume_point
        payload = {
            "run_id": self.run_id,
            "backend_id": self.backend_id,
            "event_cursor": self.event_cursor,
            "status": self.status.value,
            "terminal_status": self.terminal_status,
            "campaign_progress": {"current": self.campaign_current, "total": self.campaign_total},
            "seeds": [
                {**asdict(item), "status": item.status.value}
                for item in sorted(self.seeds.values(), key=lambda row: row.seed)
            ],
            "loss_curve": self.loss_curve,
            "validation_loss_curve": self.validation_loss_curve,
            "learning_rate_curve": self.learning_rate_curve,
            "checkpoints": [asdict(item) for item in self.checkpoints],
            "latest_verified_checkpoint": asdict(latest) if latest else None,
            "last_safe_resume_point": asdict(safe) if safe else None,
            "checkpoint_schedule": self.checkpoint_schedule,
            "estimated_completion": self.estimated_completion,
            "pause_available": self.pause_available,
            "resume_available": self.resume_available,
            "cancel_available": self.cancel_available,
            "unsafe_resume_available": False,
            "logs": self.logs,
            "warnings": self.warnings,
            "previews": self.previews,
            "seed_outcomes": [dict(value) for _key, value in sorted(self.seed_outcomes.items())],
            "job_outcomes": [dict(value) for _key, value in sorted(self.job_outcomes.items())],
            "unknown_backend_operation_count": self.unknown_backend_operation_count,
            "remote_resource_uncertain": self.remote_resource_uncertain,
            "may_accrue_cost": self.may_accrue_cost,
            "shutdown_guidance": self.shutdown_guidance,
        }
        validate_finite_json(payload)
        return payload


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_finite(metrics: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metrics:
            value = _finite(metrics.get(key))
            if value is not None:
                return value
    return None


def _is_chart_metric_event(metrics: Any) -> bool:
    return any(key in metrics for key in (*CHART_METRICS, *LEARNING_RATE_KEYS))

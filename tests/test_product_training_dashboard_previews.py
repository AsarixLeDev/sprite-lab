from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spritelab.product_core import ProductEvent, ProductStatus
from spritelab.product_core.events import ProductEventValidationError
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.previews import PreviewConfiguration, PreviewScheduler


def _event(event_type: str = "progress", **metrics) -> ProductEvent:
    return ProductEvent(
        run_id="campaign",
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature="training",
        stage="seed",
        event_type=event_type,
        status=ProductStatus.RUNNING,
        current=int(metrics.get("optimizer_step", 0)),
        total=1_000,
        metrics=metrics,
    )


def test_loss_curves_seed_progress_and_optional_gpu_metrics() -> None:
    dashboard = DashboardState("campaign", "local")
    dashboard.apply(
        _event(
            seed=731001,
            optimizer_step=100,
            total_steps=1_000,
            loss=0.42,
            validation_loss=0.51,
            learning_rate=0.0002,
            gradient_norm=1.3,
            gpu_utilization=88,
            vram_bytes=8_000_000_000,
        )
    )
    result = dashboard.to_dict()
    assert result["seeds"][0]["optimizer_step"] == 100
    assert result["loss_curve"] == [{"seed": 731001, "step": 100, "value": 0.42}]
    assert result["validation_loss_curve"][0]["value"] == 0.51
    assert result["seeds"][0]["gradient_norm"] == 1.3
    assert result["seeds"][0]["gpu_utilization"] == 88.0


def test_learning_rate_curve_has_no_data_one_point_history_and_explicit_gaps() -> None:
    dashboard = DashboardState("campaign", "local")
    assert dashboard.to_dict()["learning_rate_curve"] == []
    dashboard.apply(_event(seed=731001, optimizer_step=10, learning_rate=0.001))
    dashboard.apply(_event(seed=731001, optimizer_step=20, loss=0.9))
    dashboard.apply(_event(seed=731001, optimizer_step=30, lr=0.0005))
    assert dashboard.to_dict()["learning_rate_curve"] == [
        {"seed": 731001, "step": 10, "value": 0.001},
        {"seed": 731001, "step": 20, "value": None},
        {"seed": 731001, "step": 30, "value": 0.0005},
    ]


def test_non_finite_learning_rate_is_rejected_and_replay_is_stable() -> None:
    with pytest.raises(ProductEventValidationError, match="Non-finite numeric value"):
        _event(seed=731001, optimizer_step=20, learning_rate=math.inf)

    events = [
        _event(seed=731001, optimizer_step=10, learning_rate=0.001),
        _event(seed=731001, optimizer_step=20, loss=0.9),
    ]
    first = DashboardState("campaign", "local")
    replay = DashboardState("campaign", "local")
    for event in events:
        first.apply(event)
        replay.apply(event)
    assert first.to_dict()["learning_rate_curve"] == replay.to_dict()["learning_rate_curve"]
    assert first.to_dict()["learning_rate_curve"][-1]["value"] is None
    assert first.to_dict()["seeds"][0]["learning_rate"] == 0.001


def test_learning_rate_chart_is_accessible_gap_aware_and_narrow_responsive() -> None:
    root = Path(__file__).resolve().parents[1]
    template = (root / "src/spritelab/product_features/training/templates/training.html").read_text(encoding="utf-8")
    javascript = (root / "src/spritelab/product_features/training/static/training.js").read_text(encoding="utf-8")
    css = (root / "src/spritelab/product_features/training/static/training.css").read_text(encoding="utf-8")
    assert "Learning rate by optimizer step" in template
    assert 'id="learning-rate-summary"' in template
    assert 'id="learning-rate-table"' in template
    assert "No learning-rate data yet." in template
    assert "point.value==null?null" in javascript
    assert "if(!Number.isFinite(point.numericValue))" in javascript
    assert "@media(max-width:430px)" in css
    assert "@media(forced-colors:active)" in css


def test_remote_checkpoint_is_not_safe_until_download_hash_and_identity_verify() -> None:
    dashboard = DashboardState("campaign", "ssh")
    dashboard.apply(
        _event(
            "checkpoint",
            seed=731001,
            optimizer_step=500,
            checkpoint="seed/checkpoint.pt",
            sha256="a" * 64,
            remote_identity_verified=True,
            downloaded=False,
            hash_verified=False,
        )
    )
    assert dashboard.last_safe_resume_point is None
    dashboard.apply(
        _event(
            "checkpoint",
            seed=731001,
            optimizer_step=500,
            checkpoint="seed/checkpoint.pt",
            sha256="a" * 64,
            remote_identity_verified=True,
            downloaded=True,
            hash_verified=True,
        )
    )
    assert dashboard.last_safe_resume_point is not None
    assert dashboard.to_dict()["unsafe_resume_available"] is False


def test_remote_uncertainty_warns_about_continuing_cost() -> None:
    dashboard = DashboardState("campaign", "ssh")
    dashboard.apply(
        _event(
            "remote_failure",
            resource_state_uncertain=True,
            may_accrue_cost=True,
            shutdown_guidance="Open provider console and terminate resource pod-123.",
        )
    )
    result = dashboard.to_dict()
    assert result["remote_resource_uncertain"] is True
    assert result["may_accrue_cost"] is True
    assert "terminate" in result["shutdown_guidance"]


def test_scheduled_previews_record_prompt_seed_parameters_and_exploratory_label(tmp_path: Path) -> None:
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"png")
        return kwargs["output_path"]

    scheduler = PreviewScheduler(
        PreviewConfiguration(
            interval_steps=500,
            prompts=("small blue potion",),
            generation_seeds=(42,),
            parameters=(("cfg", 3.0),),
        ),
        generate,
    )
    events = scheduler.generate(
        run_id="campaign",
        run_root=tmp_path,
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_step=500,
        training_seed=731001,
        checkpoint_schedule=[500, 1000],
    )
    assert len(calls) == 1 and len(events) == 1
    assert events[0].metrics["exploratory"] is True
    assert events[0].metrics["benchmark_evidence"] is False
    assert events[0].metrics["promotion_evidence"] is False
    assert events[0].metrics["generation_seed"] == 42


def test_preview_failure_never_fails_training(tmp_path: Path) -> None:
    def fail(**kwargs):
        raise RuntimeError("preview GPU unavailable")

    scheduler = PreviewScheduler(
        PreviewConfiguration(interval_steps=500, prompts=("fixed prompt",)),
        fail,
    )
    events = scheduler.generate(
        run_id="campaign",
        run_root=tmp_path,
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_step=500,
        training_seed=731001,
        checkpoint_schedule=[500],
    )
    assert events[0].event_type == "preview_failed"
    assert events[0].status == ProductStatus.RUNNING


def test_disabled_previews_call_no_generator(tmp_path: Path) -> None:
    scheduler = PreviewScheduler(
        PreviewConfiguration(enabled=False),
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("generator called")),
    )
    assert (
        scheduler.generate(
            run_id="campaign",
            run_root=tmp_path,
            checkpoint=tmp_path / "checkpoint.pt",
            checkpoint_step=500,
            training_seed=1,
            checkpoint_schedule=[500],
        )
        == ()
    )

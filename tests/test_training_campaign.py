from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.training import campaign as campaign_module
from spritelab.training.campaign import (
    CAMPAIGN_ARTIFACTS,
    DEFAULT_SEEDS,
    PER_RUN_ARTIFACTS,
    CampaignResumeError,
    CampaignValidationError,
    aggregate_cross_seed_metrics,
    audit_artifact_completeness,
    audit_resume,
    checkpoint_steps,
    effective_pass_report,
    evaluation_steps,
    execute_campaign,
    plan_campaign,
    validate_campaign,
    validate_fixed_step_fairness,
)
from spritelab.training.cli import main as train_cli

HASHES = {
    name: f"{index:064x}"
    for index, name in enumerate(
        (
            "dataset_view_manifest_hash",
            "split_manifest_hash",
            "model_config_hash",
            "conditioning_vocabulary_hash",
            "optimizer_config_hash",
            "schedule_config_hash",
            "loss_config_hash",
            "determinism_config_hash",
        ),
        start=1,
    )
}
BENCHMARK_HASH = f"{99:064x}"


def _spec(tmp_path: Path, *, cells: list[dict] | None = None) -> dict:
    return {
        "campaign_id": "architecture_fairness_v1",
        "purpose": "Fixed-step comparison of auxiliary architecture heads.",
        "architecture_cells": cells
        or [
            {"cell_id": "base", "comparison_values": {"auxiliary_heads_mode": "off"}},
            {"cell_id": "heads", "comparison_values": {"auxiliary_heads_mode": "on"}},
        ],
        "experimental_variables": ["auxiliary_heads_mode"],
        "identities": deepcopy(HASHES),
        "seeds": list(DEFAULT_SEEDS),
        "training": {
            "max_optimizer_steps": 25_000,
            "micro_batch_size": 8,
            "gradient_accumulation": 4,
            "effective_batch_size": 32,
            "precision": "bf16",
            "sampler_policy": "weighted_replacement_v1",
            "positive_sampling_mass_records": 800.0,
            "nominal_record_count": 1000,
            "positive_weight_record_count": 900,
            "positive_weight_sum": 800.0,
        },
        "optimizer": {"name": "adamw", "learning_rate": 0.0002},
        "schedule": {"name": "cosine", "warmup_steps": 500},
        "loss": {"name": "uniform_velocity"},
        "determinism": {"mode": "strict", "loader": "seeded"},
        "evaluation": {
            "cadence": 1000,
            "include_step_zero": False,
            "benchmark_manifest_hash": BENCHMARK_HASH,
            "evaluation_config_hash": f"{100:064x}",
            "cfg_value": 3.0,
            "sampling_steps": 30,
            "ema_policy": "both",
            "live_weight_evaluation_policy": "required",
        },
        "checkpoint": {"cadence": 5000, "require_resumability_metadata": True},
        "output_root": str(tmp_path / "runs"),
        "abort_conditions": ["non-finite metric"],
        "promotion_restrictions": ["all three seeds and independent approval required"],
        "executable": True,
        "baseline_launch_authorized": False,
    }


def _plan(tmp_path: Path) -> dict:
    plan = plan_campaign(_spec(tmp_path))
    assert plan["plan_status"] == "ready"
    assert plan["executable"] is True
    return plan


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_campaign_has_exactly_three_unique_default_seeds(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan["seeds"] == [731001, 731002, 731003]
    assert len(set(plan["seeds"])) == 3
    assert len(plan["expected_runs"]) == 6


@pytest.mark.parametrize(
    ("seeds", "message"),
    [([731001, 731001, 731003], "unique"), ([731001, 731002], "exactly 3")],
)
def test_duplicate_or_missing_seed_is_rejected(tmp_path: Path, seeds: list[int], message: str) -> None:
    spec = _spec(tmp_path)
    spec["seeds"] = seeds
    validation = validate_campaign(plan_campaign(spec))
    assert any(message in error for error in validation["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training.max_optimizer_steps", 24_999),
        ("training.effective_batch_size", 16),
        ("identities.dataset_view_manifest_hash", f"{123:064x}"),
        ("evaluation.cadence", 500),
        ("checkpoint.cadence", 2500),
        ("evaluation.cfg_value", 2.5),
        ("evaluation.sampling_steps", 20),
        ("evaluation.ema_policy", "ema"),
    ],
)
def test_protected_fairness_mismatches_are_all_reported(tmp_path: Path, field: str, value: object) -> None:
    spec = _spec(tmp_path)
    spec["architecture_cells"][1]["overrides"] = {field: value}
    report = validate_fixed_step_fairness(plan_campaign(spec))
    assert not report["fair"]
    assert any(row["field"] == field for row in report["mismatches"])
    assert any(field in error for error in report["errors"])


def test_one_declared_architecture_variable_may_differ(tmp_path: Path) -> None:
    report = validate_fixed_step_fairness(_plan(tmp_path))
    assert report["fair"]
    assert report["errors"] == []


def test_undeclared_architecture_variable_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["experimental_variables"] = []
    report = validate_fixed_step_fairness(plan_campaign(spec))
    assert any("auxiliary_heads_mode" in error for error in report["errors"])


def test_placeholder_hash_blocks_plan_and_execution(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["identities"]["model_config_hash"] = "UNRESOLVED_HEADLESS_MODEL_HASH"
    plan = plan_campaign(spec)
    assert plan["plan_status"] == "blocked"
    assert plan["executable"] is False
    assert any("model_config_hash" in blocker for blocker in plan["blockers"])
    with pytest.raises(CampaignValidationError, match="blocked"):
        execute_campaign(plan, execute=True, confirm_execute=True, runner=lambda *args, **kwargs: None)


def test_effective_pass_formulas_and_weighted_warning() -> None:
    report = effective_pass_report(
        optimizer_steps=25_000,
        effective_batch_size=32,
        positive_sampling_mass_records=800,
        nominal_record_count=1000,
        positive_weight_record_count=900,
        positive_weight_sum=750,
    )
    assert report["effective_dataset_passes"] == 1000
    assert report["nominal_record_passes"] == 800
    assert report["positive_weight_record_passes"] == pytest.approx(888.8888888889)
    assert report["expected_weighted_exposure_mass"] == pytest.approx(1066.6666666667)
    assert "do not mean every record" in report["interpretation_warning"]


def test_evaluation_and_checkpoint_schedules_include_final_without_disappearing() -> None:
    assert evaluation_steps(25_000, 1000) == list(range(1000, 25_001, 1000))
    assert evaluation_steps(2500, 1000, include_step_zero=True) == [0, 1000, 2000, 2500]
    assert checkpoint_steps(2500, 1000) == [1000, 2000, 2500]
    with pytest.raises(ValueError, match="fixed-epoch"):
        evaluation_steps(0, 1000)


def test_deterministic_repeated_planning(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert plan_campaign(spec) == plan_campaign(deepcopy(spec))


def test_existing_foreign_output_root_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan["expected_runs"][0]["output_root"]).mkdir(parents=True)
    report = audit_resume(plan)
    assert not report["safe"]
    assert any("foreign or unowned" in error for error in report["errors"])


def test_resume_with_mismatched_checkpoint_identity_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    _write_json(
        root / "run_identity.json",
        {
            "campaign_id": plan["campaign_id"],
            "campaign_identity": plan["campaign_identity"],
            "run_id": run["run_id"],
            "run_identity": run["run_identity"],
        },
    )
    _write_json(
        root / "checkpoint_step_005000.json",
        {
            "optimizer_step": 5000,
            "campaign_identity": plan["campaign_identity"],
            "run_identity": "wrong",
            "resumability_metadata": {"optimizer": "present", "rng": "present"},
        },
    )
    report = audit_resume(plan)
    assert not report["safe"]
    assert any("checkpoint run identity" in error for error in report["errors"])


def test_unsafe_resume_request_is_always_disallowed(tmp_path: Path) -> None:
    report = audit_resume(_plan(tmp_path), unsafe_resume=True)
    assert not report["safe"]
    assert all("unsafe resume" in error for error in report["errors"])


def test_valid_resume_uses_checkpoint_and_never_restarts_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)
    for run in plan["expected_runs"]:
        config = Path(run["resolved_config_path"])
        _write_json(config, {"synthetic": True})
    first = plan["expected_runs"][0]
    root = Path(first["output_root"])
    _write_json(
        root / "run_identity.json",
        {
            "campaign_id": plan["campaign_id"],
            "campaign_identity": plan["campaign_identity"],
            "run_id": first["run_id"],
            "run_identity": first["run_identity"],
        },
    )
    checkpoint = root / "checkpoint_step_005000.json"
    _write_json(
        checkpoint,
        {
            "optimizer_step": 5000,
            "campaign_identity": plan["campaign_identity"],
            "run_identity": first["run_identity"],
            "resumability_metadata": {"optimizer": True, "rng": True},
        },
    )
    report = execute_campaign(plan, execute=True, confirm_execute=True, resume=True)
    assert len(report["launched"]) == 6
    assert ["--resume", str(checkpoint)] == calls[0][-2:]
    assert all(command for command in calls)


def test_partial_completed_campaign_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    completed = plan["expected_runs"][0]
    root = Path(completed["output_root"])
    _write_json(
        root / "run_identity.json",
        {
            "campaign_id": plan["campaign_id"],
            "campaign_identity": plan["campaign_identity"],
            "run_id": completed["run_id"],
            "run_identity": completed["run_identity"],
        },
    )
    marker = root / "run_completion_marker.json"
    _write_json(marker, {"complete": True})
    for run in plan["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), {"synthetic": True})
    calls: list[list[str]] = []
    monkeypatch.setattr(
        campaign_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    report = execute_campaign(plan, execute=True, confirm_execute=True)
    assert report["preserved_completed_runs"] == [completed["run_id"]]
    assert len(calls) == 5
    assert json.loads(marker.read_text(encoding="utf-8")) == {"complete": True}


def test_resume_requires_explicit_resume_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    _write_json(
        root / "run_identity.json",
        {
            "campaign_id": plan["campaign_id"],
            "campaign_identity": plan["campaign_identity"],
            "run_id": run["run_id"],
            "run_identity": run["run_identity"],
        },
    )
    _write_json(
        root / "checkpoint_step_005000.json",
        {
            "optimizer_step": 5000,
            "campaign_identity": plan["campaign_identity"],
            "run_identity": run["run_identity"],
            "resumability_metadata": {"rng": True},
        },
    )
    for expected in plan["expected_runs"]:
        _write_json(Path(expected["resolved_config_path"]), {"synthetic": True})
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(CampaignResumeError, match="explicit resume"):
        execute_campaign(plan, execute=True, confirm_execute=True)


def test_missing_evaluation_artifact_keeps_campaign_incomplete(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    for name in PER_RUN_ARTIFACTS:
        if name != "evaluation_reports":
            _write_json(root / f"{name}.json", {})
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert "evaluation_reports" in report["runs"][0]["missing"]


def _synthetic_reports(plan: dict, *, omit_last: bool = False) -> dict[str, dict]:
    reports = {}
    selected = plan["expected_runs"][:-1] if omit_last else plan["expected_runs"]
    for run in selected:
        reports[run["run_id"]] = {
            "metrics": {"validation_loss": float(run["seed"] - 731000)},
            "metric_definitions": {"validation_loss": {"unit": "mean velocity MSE", "split": "validation"}},
        }
    return reports


def test_cross_seed_aggregation_reports_missing_seed_without_averaging(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    summary = aggregate_cross_seed_metrics(plan, _synthetic_reports(plan, omit_last=True))
    assert not summary["complete"]
    assert len(summary["missing_runs"]) == 1
    assert summary["promotion_eligible"] is False
    heads = next(row for row in summary["per_seed_metrics"] if row["cell_id"] == "heads")
    assert heads["mean"] is None


def test_all_three_synthetic_runs_per_cell_aggregate_deterministically(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    reports = _synthetic_reports(plan)
    first = aggregate_cross_seed_metrics(plan, reports)
    second = aggregate_cross_seed_metrics(plan, deepcopy(reports))
    assert first == second
    assert first["complete"]
    for row in first["per_seed_metrics"]:
        assert row["mean"] == 2.0
        assert row["standard_deviation"] == pytest.approx(0.816496580927726)
        assert row["minimum"] == 1.0
        assert row["maximum"] == 3.0
    assert first["promotion_eligible"] is False


def test_incompatible_metric_definitions_are_never_averaged(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    reports = _synthetic_reports(plan)
    reports[plan["expected_runs"][0]["run_id"]]["metric_definitions"]["validation_loss"]["unit"] = "sum"
    summary = aggregate_cross_seed_metrics(plan, reports)
    row = next(item for item in summary["per_seed_metrics"] if item["cell_id"] == "base")
    assert not row["compatible"]
    assert row["mean"] is None


def test_all_required_artifact_names_are_bound(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan["expected_artifact_contract"]["per_run"] == list(PER_RUN_ARTIFACTS)
    assert plan["expected_artifact_contract"]["campaign"] == list(CAMPAIGN_ARTIFACTS)


@pytest.mark.parametrize("command", ["campaign-plan", "campaign-validate", "campaign-run", "campaign-status"])
def test_campaign_cli_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        train_cli([command, "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert f"python -m spritelab train {command}" in output


def test_execution_requires_both_explicit_noninteractive_flags(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(CampaignValidationError, match="execute=True"):
        execute_campaign(plan, execute=False, confirm_execute=True, runner=lambda *args, **kwargs: None)
    with pytest.raises(CampaignValidationError, match="confirm-execute"):
        execute_campaign(plan, execute=True, confirm_execute=False, runner=lambda *args, **kwargs: None)

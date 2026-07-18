from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_NATIVE,
    record_event_history_origin,
)
from spritelab.training import campaign as campaign_module
from spritelab.training.campaign import (
    CAMPAIGN_ARTIFACTS,
    DEFAULT_SEEDS,
    PER_RUN_ARTIFACTS,
    RESUME_CHECKPOINT_SCHEMA_VERSION,
    CampaignResumeError,
    CampaignValidationError,
    aggregate_cross_seed_metrics,
    audit_artifact_completeness,
    audit_resume,
    checkpoint_steps,
    effective_pass_report,
    evaluation_steps,
    execute_campaign,
    file_sha256,
    plan_campaign,
    stable_hash,
    validate_campaign,
    validate_fixed_step_fairness,
)
from spritelab.training.cli import main as train_cli
from spritelab.training.cli.experiment_cmds import _authoritative_lr_schedule

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
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    dataset = inputs / "dataset.json"
    split = inputs / "split.json"
    vocabulary = inputs / "vocabulary.json"
    benchmark = inputs / "benchmark.json"
    for path, payload in (
        (dataset, {"view": "synthetic"}),
        (vocabulary, {"tokens": ["<pad>", "sprite"]}),
        (benchmark, {"prompts": ["sprite"]}),
    ):
        _write_json(path, payload)
    split.write_text(
        json.dumps({"split": "train", "sprite_id": "a", "npz_file": "train.npz", "npz_row": 0}) + "\n",
        encoding="utf-8",
    )
    (inputs / "train.npz").write_bytes(b"synthetic retained dataset artifact")
    optimizer = {"name": "adamw", "learning_rate": 0.0002}
    schedule = {"name": "cosine", "warmup_steps": 500}
    loss = {"name": "uniform_velocity"}
    determinism = {"mode": "strict", "loader": "seeded"}
    evaluation = {
        "cadence": 1000,
        "include_step_zero": False,
        "benchmark_manifest_hash": file_sha256(benchmark),
        "benchmark_manifest_path": str(benchmark),
        "cfg_value": 3.0,
        "sampling_steps": 30,
        "ema_policy": "both",
        "live_weight_evaluation_policy": "required",
    }
    evaluation["evaluation_config_hash"] = stable_hash(
        {key: value for key, value in evaluation.items() if not key.startswith("benchmark_manifest_")}
    )
    identities = {
        "dataset_view_manifest_hash": file_sha256(dataset),
        "dataset_view_manifest_path": str(dataset),
        "split_manifest_hash": file_sha256(split),
        "split_manifest_path": str(split),
        "conditioning_vocabulary_hash": file_sha256(vocabulary),
        "conditioning_vocabulary_path": str(vocabulary),
        "model_config_hash": stable_hash({}),
        "optimizer_config_hash": stable_hash(optimizer),
        "schedule_config_hash": stable_hash(schedule),
        "loss_config_hash": stable_hash(loss),
        "determinism_config_hash": stable_hash(determinism),
    }
    return {
        "campaign_id": "architecture_fairness_v1",
        "purpose": "Fixed-step comparison of auxiliary architecture heads.",
        "architecture_cells": cells
        or [
            {"cell_id": "base", "comparison_values": {"auxiliary_heads_mode": "off"}},
            {"cell_id": "heads", "comparison_values": {"auxiliary_heads_mode": "on"}},
        ],
        "experimental_variables": ["auxiliary_heads_mode"],
        "identities": identities,
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
        "optimizer": optimizer,
        "schedule": schedule,
        "loss": loss,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": {"cadence": 5000, "require_resumability_metadata": True},
        "output_root": str(tmp_path / "runs"),
        "campaign_artifact_root": str(tmp_path / "campaign-artifacts"),
        "abort_conditions": ["non-finite metric"],
        "promotion_restrictions": ["all three seeds and independent approval required"],
        "executable": True,
        "launch_authorized": True,
    }


def _plan(tmp_path: Path) -> dict:
    plan = plan_campaign(_spec(tmp_path), execution_root=tmp_path)
    assert plan["plan_status"] == "ready"
    assert plan["executable"] is True
    return plan


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_campaign_config(plan: dict, tmp_path: Path) -> Path:
    path = tmp_path / "campaign-config.json"
    _write_json(path, plan)
    return path


def _materialize_campaign_for_execution(plan: dict, tmp_path: Path) -> Path:
    for run in plan["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), run["resolved_config"])
    return _write_campaign_config(plan, tmp_path)


def _write_partial_child_outputs(plan: dict, run: dict) -> None:
    root = Path(run["output_root"])
    _write_json(root / "run_identity.json", _run_identity_payload(plan, run))
    _write_json(
        root / "training_metrics.json",
        {
            "campaign_identity": plan["campaign_identity"],
            "run_identity": run["run_identity"],
            "seed": run["seed"],
            "optimizer_step": 1,
        },
    )


def _run_identity_payload(plan: dict, run: dict) -> dict:
    return {
        "campaign_id": plan["campaign_id"],
        "campaign_identity": plan["campaign_identity"],
        "run_id": run["run_id"],
        "run_identity": run["run_identity"],
        "output_root": run["output_root"],
        "resolved_config_sha256": run["resolved_config_sha256"],
        "execution_contract_sha256": run["execution_contract_sha256"],
    }


def _write_run_identity(plan: dict, run: dict) -> None:
    root = Path(run["output_root"])
    _write_json(root / "run_identity.json", _run_identity_payload(plan, run))
    (root / EVENT_FILENAME).write_bytes(b"")
    record_event_history_origin(
        str(run["run_id"]),
        root,
        expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
        allow_binding_population=True,
    )


def _write_resume_sidecar(plan: dict, run: dict, *, step: int = 5000) -> tuple[Path, Path]:
    root = Path(run["output_root"])
    checkpoint = root / f"checkpoint_step_{step:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"synthetic exact-resume checkpoint {step}".encode())
    sidecar = root / f"checkpoint_step_{step:06d}.json"
    _write_json(
        sidecar,
        {
            "optimizer_step": step,
            "campaign_identity": plan["campaign_identity"],
            "run_identity": run["run_identity"],
            "resumability_metadata": {
                "schema_version": RESUME_CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_relative_path": checkpoint.name,
                "checkpoint_content_sha256": file_sha256(checkpoint),
                "source_checkpoint_identity": file_sha256(checkpoint),
                "target_runtime_identity": run["run_identity"],
                "experiment_manifest_identity": stable_hash(run["resolved_config"]),
                "exact_replay_eligible": True,
                "unsafe_resume": False,
                "max_optimizer_steps": plan["training"]["max_optimizer_steps"],
                "gradient_accumulation_position": 0,
                "state_presence": {
                    "model_state_dict": True,
                    "optimizer_state_dict": True,
                    "scheduler_state_dict": True,
                    "ema_state_dict": True,
                    "rng_states": True,
                    "sampler_state": True,
                    "dataloader_generator_state": True,
                },
            },
        },
    )
    return sidecar, checkpoint


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


def test_campaign_cli_uses_the_authoritative_top_level_learning_rate_schedule(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    resolved = plan["expected_runs"][0]["resolved_config"]

    assert _authoritative_lr_schedule(resolved) == ("cosine", 500)

    drifted = deepcopy(resolved)
    drifted["optimizer"]["schedule"] = "none"
    with pytest.raises(ValueError, match="differs from the authoritative campaign schedule"):
        _authoritative_lr_schedule(drifted)


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


def test_empty_output_root_is_fresh(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan["expected_runs"][0]["output_root"]).mkdir(parents=True)
    report = audit_resume(plan)
    assert report["safe"]
    assert report["root_state"] == "fresh"


def test_resume_with_mismatched_checkpoint_identity_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    _write_run_identity(plan, run)
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
        _write_json(config, run["resolved_config"])
    first = plan["expected_runs"][0]
    _write_run_identity(plan, first)
    checkpoint, actual_checkpoint = _write_resume_sidecar(plan, first)
    report = execute_campaign(
        plan,
        execute=True,
        confirm_execute=True,
        campaign_config_path=_write_campaign_config(plan, tmp_path),
        project_root=tmp_path,
        resume=True,
    )
    assert len(report["launched"]) == 6
    assert checkpoint.is_file()
    assert ("--resume", str(actual_checkpoint)) == calls[0][-2:]
    assert all(command for command in calls)


def test_sequential_seed_launches_retain_one_capability_while_children_publish_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    campaign_config = _materialize_campaign_for_execution(plan, tmp_path)
    calls: list[str] = []
    retained_identities: dict[str, tuple[int, int]] = {}

    def child_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        run = plan["expected_runs"][len(calls)]
        root = Path(run["output_root"])
        before = root.stat()
        retained_identities[run["run_id"]] = (before.st_dev, before.st_ino)
        _write_partial_child_outputs(plan, run)
        after = root.stat()
        assert (after.st_dev, after.st_ino) == retained_identities[run["run_id"]]
        calls.append(run["run_id"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", child_run)
    report = execute_campaign(
        plan,
        execute=True,
        confirm_execute=True,
        campaign_config_path=campaign_config,
        project_root=tmp_path,
    )

    expected_ids = [str(run["run_id"]) for run in plan["expected_runs"]]
    assert calls == expected_ids
    assert [row["run_id"] for row in report["launched"]] == expected_ids
    assert [plan["expected_runs"][index]["seed"] for index in range(3)] == list(DEFAULT_SEEDS)
    for run in plan["expected_runs"]:
        root = Path(run["output_root"])
        current = root.stat()
        assert (current.st_dev, current.st_ino) == retained_identities[run["run_id"]]
        assert (root / "training_metrics.json").is_file()


def test_pending_seed_namespace_rejects_a_foreign_entry_before_second_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    campaign_config = _materialize_campaign_for_execution(plan, tmp_path)
    calls: list[str] = []

    def first_child(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        run = plan["expected_runs"][len(calls)]
        _write_partial_child_outputs(plan, run)
        calls.append(str(run["run_id"]))
        if len(calls) == 1:
            foreign_root = Path(plan["expected_runs"][1]["output_root"])
            (foreign_root / "foreign-entry.bin").write_bytes(b"not issued by the selected child capability")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", first_child)
    with pytest.raises(CampaignValidationError, match="training output changed after the retained resume audit"):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            campaign_config_path=campaign_config,
            project_root=tmp_path,
        )

    assert calls == [plan["expected_runs"][0]["run_id"]]


def test_pending_seed_namespace_rejects_an_outside_hardlink_without_mutating_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    campaign_config = _materialize_campaign_for_execution(plan, tmp_path)
    sentinel = tmp_path / "outside-run-roots-sentinel.bin"
    sentinel.write_bytes(b"outside sentinel remains byte-identical")
    before = sentinel.read_bytes()
    calls: list[str] = []

    def first_child(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        run = plan["expected_runs"][len(calls)]
        _write_partial_child_outputs(plan, run)
        calls.append(str(run["run_id"]))
        if len(calls) == 1:
            pending_root = Path(plan["expected_runs"][1]["output_root"])
            try:
                os.link(sentinel, pending_root / "foreign-hardlink.bin")
            except (NotImplementedError, OSError):
                pytest.skip("hard links are unavailable in this test session")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign_module.subprocess, "run", first_child)
    with pytest.raises(CampaignValidationError, match="aliased"):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            campaign_config_path=campaign_config,
            project_root=tmp_path,
        )

    assert calls == [plan["expected_runs"][0]["run_id"]]
    assert sentinel.read_bytes() == before


def test_partial_completion_state_blocks_all_launches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    completed = plan["expected_runs"][0]
    root = Path(completed["output_root"])
    _write_run_identity(plan, completed)
    marker = root / "run_completion_marker.json"
    _write_json(marker, {"complete": True})
    for run in plan["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), run["resolved_config"])
    calls: list[list[str]] = []
    monkeypatch.setattr(
        campaign_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    with pytest.raises(CampaignResumeError):
        execute_campaign(plan, execute=True, confirm_execute=True)
    assert calls == []
    assert json.loads(marker.read_text(encoding="utf-8")) == {"complete": True}


def test_resume_requires_explicit_resume_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    run = plan["expected_runs"][0]
    _write_run_identity(plan, run)
    _write_resume_sidecar(plan, run)
    for expected in plan["expected_runs"]:
        _write_json(Path(expected["resolved_config_path"]), expected["resolved_config"])
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
            "campaign_identity": plan["campaign_identity"],
            "run_identity": run["run_identity"],
            "seed": run["seed"],
            "metrics": {"validation_loss": float(run["seed"] - 731000)},
            "metric_definitions": {"validation_loss": {"unit": "mean velocity MSE", "split": "validation"}},
        }
    return reports


def _write_complete_run_artifacts(plan: dict, runs: list[dict] | None = None) -> None:
    for run in runs or plan["expected_runs"]:
        root = Path(run["output_root"])
        checkpoints = []
        checkpoint_entries = []
        for step in run["expected_checkpoint_steps"]:
            checkpoint = root / f"checkpoint_{step}.bin"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"synthetic checkpoint {step}".encode())
            checkpoints.append(step)
            checkpoint_entries.append(
                {
                    "artifact_type": "checkpoint",
                    "relative_path": checkpoint.name,
                    "content_sha256": file_sha256(checkpoint),
                    "producing_run_identity": run["run_identity"],
                    "seed": run["seed"],
                    "scheduled_step": step,
                }
            )
        identity = {
            "campaign_identity": plan["campaign_identity"],
            "run_identity": run["run_identity"],
            "seed": run["seed"],
        }
        values = {name: dict(identity) for name in PER_RUN_ARTIFACTS if name != "artifact_manifest"}
        values["run_identity"].update(
            {
                "campaign_id": plan["campaign_id"],
                "run_id": run["run_id"],
                "output_root": run["output_root"],
                "resolved_config_sha256": run["resolved_config_sha256"],
                "execution_contract_sha256": run["execution_contract_sha256"],
            }
        )
        values["resolved_config"]["resolved_config"] = run["resolved_config"]
        values["checkpoint_series"]["checkpoint_steps"] = checkpoints
        values["training_metrics"]["definition"] = {"unit": "mean loss", "split": "train"}
        values["validation_metrics"]["definition"] = {"unit": "mean loss", "split": "validation"}
        values["ema_metrics"]["definition"] = {"weights": "ema"}
        values["live_metrics"]["definition"] = {"weights": "live"}
        values["evaluation_reports"].update(
            {
                "evaluation_steps": run["expected_evaluation_steps"],
                "evaluated_weights": ["ema", "live"],
                "metric_definitions": {"validation_loss": {"unit": "mean velocity MSE", "split": "validation"}},
            }
        )
        values["run_completion_marker"].update(
            {
                "complete": True,
                "failed": False,
                "partial": False,
                "final_optimizer_step": run["expected_checkpoint_steps"][-1],
            }
        )
        artifact_entries = []
        for name, value in values.items():
            path = root / f"{name}.json"
            _write_json(path, value)
            if name == "run_identity":
                (root / EVENT_FILENAME).write_bytes(b"")
                record_event_history_origin(
                    str(run["run_id"]),
                    root,
                    expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
                    allow_binding_population=True,
                )
            entry = {
                "artifact_type": name,
                "relative_path": path.name,
                "content_sha256": file_sha256(path),
                "producing_run_identity": run["run_identity"],
                "seed": run["seed"],
                "final_role": "required_run_artifact",
            }
            if name in {"training_metrics", "validation_metrics", "ema_metrics", "live_metrics"}:
                entry["metric_definition_identity"] = stable_hash(value["definition"])
            elif name == "evaluation_reports":
                entry["metric_definition_identity"] = stable_hash(value["metric_definitions"])
            artifact_entries.append(entry)
        _write_json(
            root / "artifact_manifest.json",
            {
                "schema_version": "spritelab_required_artifact_manifest_v1",
                **identity,
                "artifacts": artifact_entries + checkpoint_entries,
            },
        )


def _write_complete_campaign_artifacts(plan: dict) -> Path:
    root = Path(plan["campaign_artifact_root"])
    run_matrix = [
        {
            "run_id": run["run_id"],
            "run_identity": run["run_identity"],
            "cell_id": run["cell_id"],
            "seed": run["seed"],
        }
        for run in plan["expected_runs"]
    ]
    run_matrix_sha256 = stable_hash(run_matrix)
    entries = []
    for artifact_type in CAMPAIGN_ARTIFACTS:
        schema_version = f"spritelab_campaign_{artifact_type}_v1"
        payload = {
            "schema_version": schema_version,
            "campaign_identity_sha256": plan["campaign_identity"],
            "training_code_identity_sha256": plan["code_identity"]["sha256"],
            "expected_run_ids": plan["expected_run_ids"],
            "seeds": plan["seeds"],
            "run_matrix_sha256": run_matrix_sha256,
        }
        path = root / f"{artifact_type}.json"
        _write_json(path, payload)
        entries.append(
            {
                "artifact_type": artifact_type,
                "relative_path": path.name,
                "content_sha256": file_sha256(path),
                "schema_version": schema_version,
                "campaign_identity_sha256": plan["campaign_identity"],
                "training_code_identity_sha256": plan["code_identity"]["sha256"],
                "producing_stage": "synthetic_test",
                "required": True,
                "expected_role": artifact_type,
                "run_matrix_sha256": run_matrix_sha256,
            }
        )
    manifest_path = root / "campaign_artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "spritelab_campaign_artifact_manifest_v1",
            "campaign_identity_sha256": plan["campaign_identity"],
            "training_code_identity_sha256": plan["code_identity"]["sha256"],
            "expected_run_ids": plan["expected_run_ids"],
            "seeds": plan["seeds"],
            "run_matrix_sha256": run_matrix_sha256,
            "artifacts": entries,
        },
    )
    _write_json(
        root / "campaign_completion_report.json",
        {
            "schema_version": "spritelab_campaign_completion_report_v1",
            "campaign_identity_sha256": plan["campaign_identity"],
            "training_code_identity_sha256": plan["code_identity"]["sha256"],
            "campaign_artifact_manifest_sha256": file_sha256(manifest_path),
            "expected_run_ids": plan["expected_run_ids"],
            "seeds": plan["seeds"],
            "run_matrix_sha256": run_matrix_sha256,
            "complete": True,
        },
    )
    return root


def test_cross_seed_aggregation_reports_missing_seed_without_averaging(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    summary = aggregate_cross_seed_metrics(plan, _synthetic_reports(plan, omit_last=True))
    assert not summary["complete"]
    assert len(summary["missing_runs"]) == 1
    assert summary["promotion_eligible"] is False
    assert summary["per_seed_metrics"] == []
    assert summary["metrics_aggregated"] is False


def test_all_three_synthetic_runs_per_cell_aggregate_deterministically(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    _write_complete_campaign_artifacts(plan)
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
    _write_complete_run_artifacts(plan)
    _write_complete_campaign_artifacts(plan)
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


def test_campaign_completion_requires_an_authoritative_artifact_root(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.pop("campaign_artifact_root")
    plan = plan_campaign(spec)
    _write_complete_run_artifacts(plan)
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert report["status"] == "incomplete"
    assert "missing_campaign_artifact_root" in report["campaign_missing"]
    assert report["campaign_artifacts"]["comparability"] == "legacy_incomplete"


def test_empty_campaign_artifact_root_is_incomplete(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    Path(plan["campaign_artifact_root"]).mkdir(parents=True)
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert "campaign_artifact_manifest.json" in report["campaign_missing"]


@pytest.mark.parametrize("artifact_type", CAMPAIGN_ARTIFACTS)
def test_each_mandatory_campaign_artifact_is_required(tmp_path: Path, artifact_type: str) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    root = _write_complete_campaign_artifacts(plan)
    (root / f"{artifact_type}.json").unlink()
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert artifact_type in report["campaign_missing"]


@pytest.mark.parametrize(
    "mutation",
    [
        "directory",
        "path_traversal",
        "missing_hash",
        "malformed_hash",
        "changed_bytes",
        "wrong_campaign",
        "wrong_code",
        "wrong_schema",
        "duplicate_path",
        "optional_replacement",
    ],
)
def test_campaign_artifact_manifest_rejects_adversarial_replacements(tmp_path: Path, mutation: str) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    root = _write_complete_campaign_artifacts(plan)
    manifest_path = root / "campaign_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["artifacts"][0]
    artifact = root / entry["relative_path"]
    if mutation == "directory":
        artifact.unlink()
        artifact.mkdir()
    elif mutation == "path_traversal":
        entry["relative_path"] = "../escape.json"
    elif mutation == "missing_hash":
        entry.pop("content_sha256")
    elif mutation == "malformed_hash":
        entry["content_sha256"] = "not-a-hash"
    elif mutation == "changed_bytes":
        artifact.write_text('{"changed":true}', encoding="utf-8")
    elif mutation == "wrong_campaign":
        entry["campaign_identity_sha256"] = "f" * 64
    elif mutation == "wrong_code":
        entry["training_code_identity_sha256"] = "f" * 64
    elif mutation == "wrong_schema":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["schema_version"] = "wrong"
        _write_json(artifact, payload)
        entry["content_sha256"] = file_sha256(artifact)
    elif mutation == "duplicate_path":
        manifest["artifacts"][1]["relative_path"] = entry["relative_path"]
    elif mutation == "optional_replacement":
        entry["artifact_type"] = "optional_diagnostic"
    _write_json(manifest_path, manifest)
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert report["status"] == "not_comparable" or mutation == "optional_replacement"
    assert report["campaign_artifacts"]["errors"] or report["campaign_artifacts"]["missing"]


def test_campaign_artifact_symlink_escape_is_not_comparable(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    root = _write_complete_campaign_artifacts(plan)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    artifact = root / "campaign_manifest.json"
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert report["status"] == "not_comparable"


def test_campaign_artifact_root_rejects_undeclared_extra_entries(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    root = _write_complete_campaign_artifacts(plan)
    (root / "uncontracted_diagnostic.json").write_text("{}\n", encoding="utf-8")
    report = audit_artifact_completeness(plan)
    assert report["complete"] is False
    assert report["status"] == "not_comparable"
    assert any("unexpected campaign artifact root entry" in reason for reason in report["reasons"])


def test_exact_campaign_and_run_contracts_are_required_before_aggregation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    reports = _synthetic_reports(plan)
    blocked = aggregate_cross_seed_metrics(plan, reports)
    assert blocked["complete"] is False
    assert blocked["metrics_aggregated"] is False
    root = _write_complete_campaign_artifacts(plan)
    audit = audit_artifact_completeness(plan, campaign_artifact_root=root)
    assert audit["complete"] is True
    complete = aggregate_cross_seed_metrics(plan, reports, campaign_artifact_root=root)
    assert complete["complete"] is True
    assert complete["metrics_aggregated"] is True


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


@pytest.mark.parametrize("field", ["execute", "confirm_execute", "resume", "unsafe_resume"])
@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_non_boolean_runtime_flags_fail_before_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    plan = _plan(tmp_path)
    downstream_calls: list[str] = []

    def unexpected_downstream(*args: object, **kwargs: object) -> dict[str, object]:
        downstream_calls.append("called")
        pytest.fail("runtime flag validation must precede validation and path inspection")

    def unexpected_runner(*args: object, **kwargs: object) -> None:
        downstream_calls.append("runner")
        pytest.fail("runtime flag validation must precede runner selection")

    monkeypatch.setattr(campaign_module, "validate_campaign", unexpected_downstream)
    monkeypatch.setattr(campaign_module, "audit_resume", unexpected_downstream)
    flags: dict[str, object] = {
        "execute": True,
        "confirm_execute": True,
        "resume": False,
        "unsafe_resume": False,
    }
    flags[field] = value

    with pytest.raises(CampaignValidationError, match=rf"{field} must be a boolean; coercion is forbidden"):
        execute_campaign(
            plan,
            execute=flags["execute"],
            confirm_execute=flags["confirm_execute"],
            resume=flags["resume"],
            unsafe_resume=flags["unsafe_resume"],
            runner=unexpected_runner,
        )
    assert downstream_calls == []


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_audit_resume_rejects_non_boolean_unsafe_resume_before_path_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    plan = _plan(tmp_path)
    path_calls: list[str] = []

    def unexpected_path_work(*args: object, **kwargs: object) -> dict[str, object]:
        path_calls.append("called")
        pytest.fail("unsafe_resume validation must precede path inspection")

    monkeypatch.setattr(campaign_module, "_classify_run_root", unexpected_path_work)
    with pytest.raises(CampaignValidationError, match="unsafe_resume must be a boolean; coercion is forbidden"):
        audit_resume(plan, unsafe_resume=value)
    assert path_calls == []


def test_campaign_identity_is_mandatory_and_content_bound(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    missing = deepcopy(plan)
    missing.pop("campaign_identity")
    assert any("mandatory" in error for error in validate_campaign(missing)["errors"])
    forged = deepcopy(plan)
    forged["campaign_identity"] = "f" * 64
    assert any("manifest content" in error for error in validate_campaign(forged)["errors"])


@pytest.mark.parametrize("variable", ["*", "training.max_optimizer_steps", "evaluation.*"])
def test_experimental_variables_cannot_waive_safety_fields(tmp_path: Path, variable: str) -> None:
    spec = _spec(tmp_path)
    spec["experimental_variables"] = [variable]
    report = validate_fixed_step_fairness(plan_campaign(spec))
    assert not report["fair"]
    assert report["errors"]


def test_syntactic_fake_file_hash_is_blocked(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["identities"]["dataset_view_manifest_hash"] = "a" * 64
    plan = plan_campaign(spec)
    assert plan["plan_status"] == "blocked"
    assert any("actual file content" in blocker for blocker in plan["blockers"])


def test_launch_authorization_blocks_before_runner_selection(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["launch_authorized"] = False
    plan = plan_campaign(spec)
    calls = []
    with pytest.raises(CampaignValidationError, match="launch_authorized"):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            runner=lambda *args, **kwargs: calls.append(args),
        )
    assert calls == []


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_non_boolean_campaign_gate_values_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    reference = _plan(tmp_path)
    code_identity = deepcopy(reference["code_identity"])
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: deepcopy(code_identity))

    for field in ("executable", "launch_authorized"):
        spec = _spec(tmp_path)
        spec[field] = value
        planned = plan_campaign(spec)

        assert planned[field] is False
        assert planned["executable"] is False
        assert planned["plan_status"] == "blocked"
        assert any(f"{field} must be a boolean; coercion is forbidden" in item for item in planned["blockers"])
        assert validate_campaign(planned)["launch_ready"] is False

        materialized = deepcopy(reference)
        materialized[field] = value
        materialized["campaign_identity"] = stable_hash(campaign_module._campaign_identity_payload(materialized))
        validation = validate_campaign(materialized)
        assert any(f"{field} must be a boolean; coercion is forbidden" == item for item in validation["errors"])
        assert validation["launch_ready"] is False
        runner_calls: list[object] = []
        message = "blocked" if field == "executable" else "launch_authorized"
        with pytest.raises(CampaignValidationError, match=message):
            execute_campaign(
                materialized,
                execute=True,
                confirm_execute=True,
                runner=lambda *args, _calls=runner_calls, **kwargs: _calls.append((args, kwargs)),
            )
        assert runner_calls == []


@pytest.mark.parametrize(
    ("executable", "launch_authorized", "expected_executable", "expected_launch_ready"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, False, False),
        (False, False, False, False),
    ],
)
def test_real_boolean_campaign_gates_preserve_exact_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executable: bool,
    launch_authorized: bool,
    expected_executable: bool,
    expected_launch_ready: bool,
) -> None:
    reference = _plan(tmp_path)
    code_identity = deepcopy(reference["code_identity"])
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: deepcopy(code_identity))
    spec = _spec(tmp_path)
    spec["executable"] = executable
    spec["launch_authorized"] = launch_authorized

    planned = plan_campaign(spec)
    validation = validate_campaign(planned)

    assert planned["executable"] is expected_executable
    assert planned["launch_authorized"] is launch_authorized
    assert planned["plan_status"] == ("ready" if expected_executable else "blocked")
    assert not validation["errors"]
    assert not validation["blockers"]
    assert validation["launch_ready"] is expected_launch_ready


def test_missing_raw_campaign_gates_keep_safe_false_defaults(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.pop("executable")
    spec.pop("launch_authorized")

    planned = plan_campaign(spec)

    assert planned["executable"] is False
    assert planned["launch_authorized"] is False
    assert planned["plan_status"] == "blocked"
    assert not any("must be a boolean" in item for item in planned["blockers"])
    assert validate_campaign(planned)["launch_ready"] is False


@pytest.mark.parametrize("changed", ["config", "command"])
def test_changed_run_bindings_block_without_launch(tmp_path: Path, changed: str) -> None:
    plan = _plan(tmp_path)
    campaign_path = _write_campaign_config(plan, tmp_path)
    for run in plan["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), run["resolved_config"])
    if changed == "config":
        _write_json(Path(plan["expected_runs"][0]["resolved_config_path"]).with_suffix(".changed"), {})
        path = Path(plan["expected_runs"][0]["resolved_config_path"])
        path.write_text('{"changed":true}\n', encoding="utf-8")
    else:
        plan["expected_runs"][0]["experiment_command"].append("--forged")
    calls = []
    with pytest.raises(CampaignValidationError):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            campaign_config_path=campaign_path,
            runner=lambda *args, **kwargs: calls.append(args),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("artifact", "payload", "message"),
    [
        ("checkpoint_series", {"checkpoint_steps": [5000]}, "scheduled steps"),
        ("evaluation_reports", {"evaluation_steps": [1000], "evaluated_weights": ["ema", "live"]}, "scheduled steps"),
        (
            "evaluation_reports",
            {"evaluation_steps": list(range(1000, 25001, 1000)), "evaluated_weights": ["ema"]},
            "EMA/live",
        ),
    ],
)
def test_incomplete_schedule_or_weight_policy_is_not_complete(
    tmp_path: Path, artifact: str, payload: dict, message: str
) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    run = plan["expected_runs"][0]
    Path(run["output_root"], f"{artifact}.json").write_text(json.dumps(payload), encoding="utf-8")
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any(message in error for error in report["runs"][0]["errors"])


def _changed_code_identity(identity: dict, relative_path: str) -> dict:
    changed = deepcopy(identity)
    record = next(item for item in changed["files"] if item["path"] == relative_path)
    record["sha256"] = "f" * 64 if record["sha256"] != "f" * 64 else "e" * 64
    changed["sha256"] = stable_hash({key: value for key, value in changed.items() if key != "sha256"})
    return changed


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/spritelab/training/generator_challenger.py",
        "src/spritelab/training/cli/experiment_cmds.py",
        "src/spritelab/product_web/events.py",
    ],
)
def test_behavior_affecting_trainer_or_cli_change_invalidates_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    plan = _plan(tmp_path)
    changed = _changed_code_identity(plan["code_identity"], relative_path)
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: changed)
    report = validate_campaign(plan)
    assert not report["launch_ready"]
    assert any("code identity is stale" in blocker for blocker in report["blockers"])


def test_code_identity_covers_training_tree_and_excludes_unrelated_docs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    paths = {item["path"] for item in plan["code_identity"]["files"]}
    assert {
        "src/spritelab/training/campaign.py",
        "src/spritelab/training/experiment_system.py",
        "src/spritelab/training/generator_challenger.py",
        "src/spritelab/training/data.py",
        "src/spritelab/training/optim_utils.py",
        "src/spritelab/training/cli/experiment_cmds.py",
        "src/spritelab/__main__.py",
    }.issubset(paths)
    assert not any(path.startswith("docs/") for path in paths)
    (tmp_path / "unrelated.md").write_text("does not affect production code\n", encoding="utf-8")
    assert validate_campaign(plan)["launch_ready"]


def test_missing_code_identity_source_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)

    def missing_source() -> dict:
        raise CampaignValidationError("bound code-identity source is missing: trainer.py")

    monkeypatch.setattr(campaign_module, "_code_identity", missing_source)
    report = validate_campaign(plan)
    assert not report["launch_ready"]
    assert any("cannot be verified" in blocker for blocker in report["blockers"])


def test_unresolved_blocker_prevents_every_launch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["blockers"] = ["synthetic unresolved blocker"]
    plan["campaign_identity"] = stable_hash(campaign_module._campaign_identity_payload(plan))
    calls: list[list[str]] = []
    with pytest.raises(CampaignValidationError):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            runner=lambda command, **kwargs: calls.append(command),
        )
    assert calls == []


def test_partial_invalid_root_prevents_every_launch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    for run in plan["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), run["resolved_config"])
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    _write_run_identity(plan, run)
    _write_json(root / "run_completion_marker.json", {"complete": False, "partial": True})
    calls: list[list[str]] = []
    with pytest.raises(CampaignResumeError):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            runner=lambda command, **kwargs: calls.append(command),
        )
    assert calls == []
    assert audit_resume(plan)["root_state"] == "partial_invalid"


def test_changed_current_code_identity_prevents_every_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(tmp_path)
    changed = _changed_code_identity(plan["code_identity"], "src/spritelab/training/campaign.py")
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: changed)
    calls: list[list[str]] = []
    with pytest.raises(CampaignValidationError):
        execute_campaign(
            plan,
            execute=True,
            confirm_execute=True,
            runner=lambda command, **kwargs: calls.append(command),
        )
    assert calls == []


def test_output_root_classification_is_closed_and_complete(tmp_path: Path) -> None:
    fresh = _plan(tmp_path / "fresh")
    assert audit_resume(fresh)["root_state"] == "fresh"

    resumable = _plan(tmp_path / "resumable")
    run = resumable["expected_runs"][0]
    _write_run_identity(resumable, run)
    _write_resume_sidecar(resumable, run)
    assert audit_resume(resumable)["root_state"] == "valid_resumable"

    complete = _plan(tmp_path / "complete")
    _write_complete_run_artifacts(complete)
    assert audit_resume(complete)["root_state"] == "complete"

    partial = _plan(tmp_path / "partial")
    _write_complete_run_artifacts(partial, [partial["expected_runs"][0]])
    assert audit_resume(partial)["root_state"] == "partial_valid"

    foreign = _plan(tmp_path / "foreign")
    foreign_root = Path(foreign["expected_runs"][0]["output_root"])
    _write_json(foreign_root / "unknown.json", {"foreign": True})
    assert audit_resume(foreign)["root_state"] == "foreign"

    corrupt = _plan(tmp_path / "corrupt")
    corrupt_root = Path(corrupt["expected_runs"][0]["output_root"])
    corrupt_root.mkdir(parents=True)
    (corrupt_root / "run_identity.json").write_text("{", encoding="utf-8")
    assert audit_resume(corrupt)["root_state"] == "corrupt"


def _artifact_manifest(plan: dict, run: dict) -> tuple[Path, dict]:
    path = Path(run["output_root"]) / "artifact_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_artifact_manifest_rejects_missing_hash_changed_content_and_path_traversal(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    run = plan["expected_runs"][0]
    path, manifest = _artifact_manifest(plan, run)
    target = next(item for item in manifest["artifacts"] if item["artifact_type"] == "training_metrics")
    target_path = Path(run["output_root"], target["relative_path"])
    target_bytes = target_path.read_bytes()

    missing_hash = deepcopy(manifest)
    next(item for item in missing_hash["artifacts"] if item["artifact_type"] == "training_metrics").pop(
        "content_sha256"
    )
    _write_json(path, missing_hash)
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any("missing a concrete content hash" in error for error in report["runs"][0]["errors"])

    _write_json(path, manifest)
    target_path.write_text('{"changed":true}', encoding="utf-8")
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any("content hash mismatch" in error for error in report["runs"][0]["errors"])

    target_path.write_bytes(target_bytes)
    _write_json(path, manifest)
    next(item for item in manifest["artifacts"] if item["artifact_type"] == "training_metrics")["relative_path"] = (
        "../escape.json"
    )
    _write_json(path, manifest)
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any("unsafe relative path" in error for error in report["runs"][0]["errors"])


def test_foreign_artifact_identity_and_off_schedule_checkpoint_prevent_completion(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    run = plan["expected_runs"][0]
    root = Path(run["output_root"])
    experiment_manifest = root / "experiment_manifest.json"
    original_experiment_manifest = experiment_manifest.read_bytes()
    payload = json.loads(experiment_manifest.read_text(encoding="utf-8"))
    payload["run_identity"] = "f" * 64
    _write_json(experiment_manifest, payload)
    manifest_path, manifest = _artifact_manifest(plan, run)
    original_manifest = deepcopy(manifest)
    next(item for item in manifest["artifacts"] if item["relative_path"] == experiment_manifest.name)[
        "content_sha256"
    ] = file_sha256(experiment_manifest)
    _write_json(manifest_path, manifest)
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any("foreign run_identity" in error for error in report["runs"][0]["errors"])

    experiment_manifest.write_bytes(original_experiment_manifest)
    _write_json(manifest_path, original_manifest)
    (root / "checkpoint_4000.bin").write_bytes(b"off-schedule")
    report = audit_artifact_completeness(plan)
    assert not report["complete"]
    assert any("off-schedule checkpoint" in error for error in report["runs"][0]["errors"])


def test_duplicate_seed_and_foreign_report_prevent_aggregation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_complete_run_artifacts(plan)
    reports = _synthetic_reports(plan)
    for report in reports.values():
        report["seed"] = DEFAULT_SEEDS[0]
    duplicate = aggregate_cross_seed_metrics(plan, reports)
    assert not duplicate["complete"]
    assert duplicate["duplicate_seeds"]
    assert all(row["mean"] is None for row in duplicate["per_seed_metrics"])

    reports = _synthetic_reports(plan)
    reports["foreign-run"] = {
        "campaign_identity": plan["campaign_identity"],
        "run_identity": "f" * 64,
        "seed": 999,
        "metrics": {"validation_loss": 0.0},
        "metric_definitions": {"validation_loss": {"unit": "mean velocity MSE"}},
    }
    foreign = aggregate_cross_seed_metrics(plan, reports)
    assert not foreign["complete"]
    assert foreign["foreign_runs"] == ["foreign-run"]
    assert all(row["mean"] is None for row in foreign["per_seed_metrics"])

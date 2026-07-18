from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_features.training.plans import synthetic_training_path_contract_for_tests
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState, StageStatus
from spritelab.v3.orchestration import ExecutionOptions, _run_backend, dataset_build, evaluate, train
from spritelab.v3.run_state import RunState
from training_launch_test_utils import validated_launch


def _config(tmp_path: Path) -> ProjectConfig:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    values["paths"]["runs"] = "runs/v3"
    values["evaluation"].update(
        {
            "dataset_identity": "synthetic-training-dataset-v1",
            "training_view_identity": "synthetic-training-view-v1",
        }
    )
    values.update(synthetic_training_path_contract_for_tests(tmp_path))
    return ProjectConfig(root=tmp_path, path=tmp_path / "spritelab.yaml", values=values)


def _configure_valid_campaign(config: ProjectConfig, tmp_path: Path) -> SimpleNamespace:
    launch = validated_launch(tmp_path, "local")
    config.values["training"]["campaign_config"] = str(launch.validator_context.campaign_config_path)
    config.values["execution"]["allow_training"] = True
    return SimpleNamespace(
        campaign=launch.campaign,
        manifest={"image_count": 2_500},
        audit_status=AuditStatus.PASS,
        config=config,
    )


def _stage(
    key: str, status: StageStatus = StageStatus.COMPLETE, *, audit: AuditStatus = AuditStatus.NOT_AUDITED, blockers=None
) -> StageState:
    return StageState(
        key=key,
        title=key,
        status=status,
        explanation=f"{key} {status.value}",
        blockers=list(blockers or []),
        audit=audit,
        production_authorized=status == StageStatus.COMPLETE,
    )


def _state(tmp_path: Path, overrides: dict[str, StageState] | None = None) -> ProjectState:
    keys = [
        "raw-source-provenance",
        "extraction",
        "suitability",
        "semantic-labeling",
        "semantic-calibration",
        "dataset-v5-view-construction",
        "dataset-freeze",
        "training-infrastructure-audit",
        "training-campaign",
        "evaluation-generation",
        "evaluation-metrics",
        "memorization-review",
        "promotion-decision",
    ]
    stages = {key: _stage(key) for key in keys}
    stages["training-infrastructure-audit"].audit = AuditStatus.PASS
    stages["memorization-review"].audit = AuditStatus.PASS
    if overrides:
        stages.update(overrides)
    return ProjectState("synthetic", tmp_path, tmp_path / "spritelab.yaml", "abc123", list(stages.values()))


def test_dataset_dry_run_succeeds_through_synthetic_pipeline(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    result = dataset_build(config, ["dataset", "build", "--dry-run"], ExecutionOptions(dry_run=True))
    assert result.exit_code == 0
    assert result.data["production_freezes"] == 0
    assert Path(result.report_path).is_file()


@pytest.mark.parametrize(
    "key,status,code",
    [
        ("raw-source-provenance", StageStatus.BLOCKED, 3),
        ("extraction", StageStatus.FAILED, 3),
        ("semantic-labeling", StageStatus.NEEDS_REVIEW, 4),
        ("dataset-freeze", StageStatus.BLOCKED, 3),
    ],
)
def test_dataset_stops_at_first_mandatory_gate(
    monkeypatch, tmp_path: Path, key: str, status: StageStatus, code: int
) -> None:
    config = _config(tmp_path)
    failed = _stage(key, status, blockers=[f"{key} closed"])
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path, {key: failed}))
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    result = dataset_build(config, [], ExecutionOptions(dry_run=True))
    assert result.exit_code == code
    assert failed.blockers == result.blockers


def test_training_missing_freeze_never_launches(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    campaign = _stage("training-campaign", StageStatus.BLOCKED, blockers=["Dataset-v5 is not frozen."])
    monkeypatch.setattr(
        "spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path, {"training-campaign": campaign})
    )
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    assert train(config, [], ExecutionOptions(dry_run=False)).exit_code == 3


def test_training_stale_audit_uses_stale_exit(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    audit = _stage("training-infrastructure-audit", StageStatus.STALE, audit=AuditStatus.STALE, blockers=["stale"])
    monkeypatch.setattr(
        "spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path, {audit.key: audit})
    )
    assert train(config, [], ExecutionOptions(dry_run=True)).exit_code == 6


def test_valid_training_dry_run_never_initializes_or_launches(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    activation = _configure_valid_campaign(config, tmp_path)
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    monkeypatch.setattr(
        "spritelab.product_features.training.service.load_conditioned_training_activation",
        lambda *_args, **_kwargs: activation,
    )
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    result = train(config, [], ExecutionOptions(dry_run=True))
    assert result.exit_code == 0
    assert result.data["training_runs"] == 0
    assert result.data["cuda_initialized"] is False


def test_interactive_confirmation_defaults_to_no(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    activation = _configure_valid_campaign(config, tmp_path)
    config.values["execution"]["training_command"] = ["never-run"]
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    monkeypatch.setattr(
        "spritelab.product_features.training.service.load_conditioned_training_activation",
        lambda *_args, **_kwargs: activation,
    )
    monkeypatch.setattr("spritelab.v3.orchestration.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    result = train(config, [], ExecutionOptions())
    assert result.exit_code == 5 and result.status == "PAUSED"


def test_noninteractive_confirmation_requires_two_flags(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.values["execution"]["training_command"] = ["never-run"]
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    monkeypatch.setattr("spritelab.v3.orchestration.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    assert train(config, [], ExecutionOptions(yes=True)).exit_code == 3


def test_backend_uses_argument_array_without_shell(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = RunState.create(config, command="test", argv=[], source_commit="abc", dry_run=False)
    captured = {}

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return Completed()

    monkeypatch.setattr("spritelab.v3.orchestration.subprocess.run", fake_run)
    assert _run_backend(["tool", "path with spaces", "$(unsafe)"], root=tmp_path, run=run) == 0
    assert captured["command"] == ["tool", "path with spaces", "$(unsafe)"]
    assert captured["shell"] is False


@pytest.mark.parametrize(
    "key,audit,expected",
    [
        ("evaluation-generation", AuditStatus.PASS, 3),
        ("memorization-review", AuditStatus.FAIL, 3),
        ("memorization-review", AuditStatus.STALE, 6),
    ],
)
def test_evaluation_blocks_missing_or_failed_gates(
    monkeypatch, tmp_path: Path, key: str, audit: AuditStatus, expected: int
) -> None:
    config = _config(tmp_path)
    if key == "evaluation-generation":
        stage = _stage(key, StageStatus.BLOCKED, blockers=["checkpoint missing", "benchmark missing"])
    else:
        stage = _stage(
            key,
            StageStatus.FAILED if audit == AuditStatus.FAIL else StageStatus.STALE,
            audit=audit,
            blockers=[audit.value],
        )
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path, {key: stage}))
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )
    assert evaluate(config, [], ExecutionOptions(dry_run=True)).exit_code == expected


def test_evaluation_dry_run_generates_or_promotes_nothing(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    benchmark = tmp_path / "benchmark.json"
    checkpoint.touch()
    benchmark.touch()
    config.values["evaluation"].update({"checkpoint": str(checkpoint), "benchmark": str(benchmark)})
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    result = evaluate(config, [], ExecutionOptions(dry_run=True))
    assert result.exit_code == 0
    assert result.data["generation_runs"] == 0
    assert result.data["promotion_actions"] == 0
    assert result.data["plan"]["training_dataset_identity"] == "synthetic-training-dataset-v1"
    assert result.data["plan"]["training_view_identity"] == "synthetic-training-view-v1"
    assert result.run_id is not None
    state = json.loads((config.runs_dir / result.run_id / "state.json").read_text(encoding="utf-8"))
    identity = state["backend_identity"]
    assert identity["training_dataset_identity"] == "synthetic-training-dataset-v1"
    assert identity["training_view_identity"] == "synthetic-training-view-v1"
    assert identity["checkpoint_path"] == str(checkpoint.resolve())
    assert identity["checkpoint_sha256"] == sha256(checkpoint.read_bytes()).hexdigest()
    assert identity["benchmark_path"] == str(benchmark.resolve())
    assert identity["benchmark_sha256"] == sha256(benchmark.read_bytes()).hexdigest()


def test_evaluation_backend_result_persists_exact_input_identity(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    benchmark = tmp_path / "benchmark.json"
    checkpoint.write_bytes(b"synthetic checkpoint")
    benchmark.write_bytes(b'{"prompts": []}\n')
    config.values["evaluation"].update({"checkpoint": str(checkpoint), "benchmark": str(benchmark)})
    config.values["execution"]["evaluation_command"] = ["synthetic-evaluator"]
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    monkeypatch.setattr("spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: 0)

    result = evaluate(
        config,
        [],
        ExecutionOptions(yes=True, non_interactive_confirm=True),
    )

    assert result.exit_code == 0
    assert result.run_id is not None
    state = json.loads((config.runs_dir / result.run_id / "state.json").read_text(encoding="utf-8"))
    identity = state["backend_identity"]
    assert identity["dataset_identity"] == "synthetic-training-dataset-v1"
    assert identity["view_identity"] == "synthetic-training-view-v1"
    assert identity["checkpoint_sha256"] == sha256(checkpoint.read_bytes()).hexdigest()
    assert identity["benchmark_sha256"] == sha256(benchmark.read_bytes()).hexdigest()


def test_evaluation_missing_training_identity_never_reaches_backend(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.values["evaluation"]["training_view_identity"] = ""
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _state(tmp_path))
    monkeypatch.setattr(
        "spritelab.v3.orchestration._run_backend", lambda *_args, **_kwargs: pytest.fail("backend launched")
    )

    result = evaluate(config, [], ExecutionOptions(dry_run=True))

    assert result.exit_code == 3
    assert result.data.get("generation_runs", 0) == 0
    assert any("training view identity" in blocker.casefold() for blocker in result.blockers)

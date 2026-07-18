from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from spritelab.utils.safe_fs import AnchoredDirectory
from spritelab.v3 import config as config_module
from spritelab.v3.config import DEFAULT_CONFIG, ConfigError, ProjectConfig, discover_config
from spritelab.v3.model import AuditStatus, StageStatus
from spritelab.v3.status import build_project_state


def _write_json(root: Path, relative: str, value: object) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _project(tmp_path: Path) -> ProjectConfig:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    values["paths"]["runs"] = "runs/v3"
    (tmp_path / "spritelab.yaml").write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _authoritative_artifacts(root: Path, *, count: int = 31) -> None:
    _write_json(
        root,
        DEFAULT_CONFIG["dataset"]["raw_provenance_report"],
        {
            "source_gate_passed": True,
            "remaining_unresolved_sources": 0,
            "sources_verified": count,
            "sources_excluded": 50,
            "newly_eligible_record_count": count * 10,
            "missing_downloads_still_requiring_external_retrieval": 1,
        },
    )
    _write_json(
        root,
        DEFAULT_CONFIG["dataset"]["extraction_report"],
        {"determinism": {"byte_identical_rgba_outputs": True}, "remaining_ambiguity": {"explicit": 2}},
    )
    _write_json(
        root,
        DEFAULT_CONFIG["dataset"]["suitability_report"],
        {"unique_extraction_suitability_status_counts": {"accept": 2, "reject": 1}},
    )
    _write_json(
        root,
        DEFAULT_CONFIG["dataset"]["view_manifest"],
        {"status": "blocked", "candidate_dataset_created": False, "view": "synthetic"},
    )
    _write_json(
        root,
        DEFAULT_CONFIG["labeling"]["campaign_report"],
        {
            "campaign_status": "stopped_health_gate",
            "labels_are_calibrated_truth": False,
            "pass_a_completed": 10,
            "pass_b_completed": 0,
            "health_checks": [
                {"critical_field_comparisons": 10, "critical_field_disagreement_rate": 0.3, "passed": False}
            ],
            "resume": {"next_pass": "A"},
        },
    )
    _write_json(
        root,
        DEFAULT_CONFIG["labeling"]["audit_report"],
        {"core_stable_enough_to_continue": False, "resume_authorized": False},
    )
    audited = root / "src/spritelab/training/synthetic.py"
    audited.parent.mkdir(parents=True, exist_ok=True)
    audited.write_text("SAFE = True\n", encoding="utf-8")
    digest = hashlib.sha256(audited.read_bytes()).hexdigest()
    _write_json(root, DEFAULT_CONFIG["training"]["audit_report"], {"gates": {"1": "FAIL"}, "training_runs": 0})
    _write_json(
        root,
        DEFAULT_CONFIG["training"]["audit_hashes"],
        {"files": [{"path": "src/spritelab/training/synthetic.py", "sha256_before": digest}]},
    )
    _write_json(
        root,
        DEFAULT_CONFIG["evaluation"]["memorization_audit"],
        {
            "commit": "deadbeef",
            "overall_verdict": "FAIL",
            "findings": [{"id": "SYN-1", "severity": "critical", "summary": "Synthetic failure"}],
            "authorization": {"checkpoint_promotion": False},
        },
    )
    raw_inventory = root / DEFAULT_CONFIG["dataset"]["raw_inventory"]
    raw_inventory.parent.mkdir(parents=True, exist_ok=True)
    raw_inventory.write_text("{}\n", encoding="utf-8")


def test_config_discovery_from_child_directory(tmp_path: Path) -> None:
    _project(tmp_path)
    child = tmp_path / "a/b/c"
    child.mkdir(parents=True)
    assert discover_config(child) == (tmp_path / "spritelab.yaml").resolve()


def test_config_discovery_does_not_escape_repository_boundary(tmp_path: Path) -> None:
    (tmp_path / "spritelab.yaml").write_text(yaml.safe_dump(DEFAULT_CONFIG), encoding="utf-8")
    repository = tmp_path / "checkout"
    child = repository / "a/b/c"
    child.mkdir(parents=True)
    (repository / ".git").mkdir()

    assert discover_config(child) is None


def test_missing_config_is_error_when_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("spritelab.v3.config.discover_config", lambda _start: None)
    with pytest.raises(ConfigError, match="v3 init"):
        ProjectConfig.load(tmp_path)


def test_missing_config_can_load_inspection_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("spritelab.v3.config.discover_config", lambda _start: None)
    config = ProjectConfig.load(tmp_path, required=False)
    assert config.path is None
    assert config.values["execution"]["allow_training"] is False


def test_activation_commit_reader_accepts_only_one_exact_retained_stage(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    payload = b'{"committed":true}\n'
    target.write_bytes(payload)
    alias = tmp_path / f".{target.name}.staging-{'a' * 32}"
    alias.hardlink_to(target)

    with AnchoredDirectory(tmp_path, tmp_path) as anchor:
        assert (
            config_module._safe_child_bytes(
                anchor,
                target.name,
                max_bytes=1024,
                allow_retained_stage=True,
            )
            == payload
        )
        with pytest.raises(ConfigError):
            config_module._safe_child_bytes(
                anchor,
                target.name,
                max_bytes=1024,
                allow_retained_stage=False,
            )

    (tmp_path / f".{target.name}.staging-{'b' * 32}").write_bytes(b"wrong inode")
    with AnchoredDirectory(tmp_path, tmp_path) as anchor, pytest.raises(ConfigError):
        config_module._safe_child_bytes(
            anchor,
            target.name,
            max_bytes=1024,
            allow_retained_stage=True,
        )


@pytest.mark.parametrize(
    "value,match",
    [
        ("- not-a-mapping\n", "mapping"),
        ("unknown:\n  value: 1\n", "Unknown configuration section"),
        ("project:\n  name: x\n  mystery: 1\n", "Unknown key"),
        ("project:\n  schema_version: 2\n", "schema_version"),
        ("execution:\n  training_command: echo unsafe\n", "argument strings"),
        ("evaluation:\n  dataset_identity: [not, a, string]\n", "dataset_identity must be a string"),
        ("evaluation:\n  training_view_identity: ' padded '\n", "must not contain leading or trailing whitespace"),
    ],
)
def test_malformed_or_unknown_config_fails(tmp_path: Path, value: str, match: str) -> None:
    (tmp_path / "spritelab.yaml").write_text(value, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        ProjectConfig.load(tmp_path)


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "elsewhere/config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(DEFAULT_CONFIG), encoding="utf-8")
    root = tmp_path / "root override"
    runs = tmp_path / "run override"
    monkeypatch.setenv("SPRITELAB_CONFIG", str(config_path))
    monkeypatch.setenv("SPRITELAB_PROJECT_ROOT", str(root))
    monkeypatch.setenv("SPRITELAB_RUNS_DIR", str(runs))
    assert not root.exists()
    config = ProjectConfig.load(tmp_path)
    assert config.root == root.resolve()
    assert config.runs_dir == runs.resolve()
    assert config.values["execution"] == DEFAULT_CONFIG["execution"]
    assert not root.exists()


def test_environment_override_missing_file_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPRITELAB_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError, match="does not name a file"):
        discover_config(tmp_path)


def test_state_parses_counts_instead_of_hardcoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _project(tmp_path)
    _authoritative_artifacts(tmp_path, count=47)
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.FAIL)
    state = build_project_state(config)
    raw = state.stage("provenance")
    assert raw.status == StageStatus.COMPLETE
    assert raw.metrics["sources_verified"] == 47
    assert raw.metrics["newly_eligible_record_count"] == 470


def test_label_health_failure_is_review_not_truth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _project(tmp_path)
    _authoritative_artifacts(tmp_path)
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.FAIL)
    state = build_project_state(config)
    stage = state.stage("labeling")
    assert stage.status == StageStatus.NEEDS_REVIEW
    assert stage.audit == AuditStatus.NOT_COMPARABLE
    assert stage.metrics["critical_disagreements"] == 3
    assert stage.resume_available is False


def test_training_and_memorization_failures_remain_blocking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _project(tmp_path)
    _authoritative_artifacts(tmp_path)
    monkeypatch.setattr("spritelab.v3.status._training_audit_status", lambda *_: AuditStatus.FAIL)
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.FAIL)
    state = build_project_state(config)
    assert state.stage("training-audit").audit == AuditStatus.FAIL
    assert state.stage("training").status == StageStatus.BLOCKED
    assert state.stage("memorization").audit == AuditStatus.FAIL
    assert state.stage("promotion").production_authorized is False


def test_changed_training_identity_marks_audit_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _project(tmp_path)
    _authoritative_artifacts(tmp_path)
    (tmp_path / "src/spritelab/training/synthetic.py").write_text("SAFE = False\n", encoding="utf-8")
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.FAIL)
    state = build_project_state(config)
    assert state.stage("training-audit").audit == AuditStatus.STALE
    assert state.stage("training-audit").status == StageStatus.STALE


def test_missing_audit_is_not_audited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _project(tmp_path)
    _authoritative_artifacts(tmp_path)
    (tmp_path / DEFAULT_CONFIG["training"]["audit_report"]).unlink()
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.NOT_AUDITED)
    state = build_project_state(config)
    assert state.stage("training-audit").audit == AuditStatus.NOT_AUDITED
    assert state.stage("memorization").audit == AuditStatus.NOT_AUDITED


def test_directory_existence_does_not_imply_freeze(tmp_path: Path) -> None:
    config = _project(tmp_path)
    (tmp_path / "dataset").mkdir()
    state = build_project_state(config)
    assert state.stage("freeze").status == StageStatus.BLOCKED


def test_evaluation_and_promotion_readiness_require_training_identity_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _project(tmp_path)
    run_directory = tmp_path / "runs" / "v3" / "train-identity"
    checkpoint = run_directory / "checkpoints" / "checkpoint_step_000100_ema.pt"
    benchmark = tmp_path / "benchmark.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    benchmark.write_text('{"prompts": ["synthetic sprite"]}\n', encoding="utf-8")
    checkpoint_claim = {
        "path": "checkpoints/checkpoint_step_000100_ema.pt",
        "dataset_identity": "synthetic-training-dataset-v1",
        "view_identity": "synthetic-training-view-v1",
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "verified": True,
    }
    training_state = {
        "schema_version": "spritelab.v3.run-state.v1",
        "run_id": "train-identity",
        "command": "train",
        "status": "COMPLETE",
        "backend_identity": {
            "dataset_identity": "synthetic-training-dataset-v1",
            "view_identity": "synthetic-training-view-v1",
        },
        "checkpoints": [checkpoint_claim],
    }
    _write_json(tmp_path, "runs/v3/train-identity/state.json", training_state)
    _write_json(
        tmp_path,
        "runs/v3/train-identity/run_completion_marker.json",
        {"complete": True, "checkpoints": [checkpoint_claim]},
    )
    _write_json(
        tmp_path,
        "runs/v3/train-identity/command.json",
        {"command": "train", "project_root": str(tmp_path)},
    )
    config.values["evaluation"].update(
        {
            "checkpoint": str(checkpoint),
            "benchmark": str(benchmark),
        }
    )
    config.values["execution"].update(
        {
            "allow_generation": True,
            "allow_promotion": True,
        }
    )
    _write_json(
        tmp_path,
        DEFAULT_CONFIG["evaluation"]["memorization_audit"],
        {"authorization": {"checkpoint_promotion": True}},
    )
    monkeypatch.setattr("spritelab.v3.status._memorization_audit_status", lambda *_: AuditStatus.PASS)

    missing = build_project_state(config)

    assert missing.stage("evaluation").status == StageStatus.BLOCKED
    assert any("dataset identity" in blocker.casefold() for blocker in missing.stage("evaluation").blockers)
    assert any("view identity" in blocker.casefold() for blocker in missing.stage("evaluation").blockers)
    assert missing.stage("promotion").production_authorized is False

    config.values["evaluation"].update(
        {
            "dataset_identity": "synthetic-training-dataset-v1",
            "training_view_identity": "synthetic-training-view-v1",
        }
    )
    bound = build_project_state(config)

    assert bound.stage("evaluation").status == StageStatus.READY
    assert bound.stage("evaluation").metrics["identity_binding_complete"] is True
    assert bound.stage("promotion").status == StageStatus.READY
    assert bound.stage("promotion").production_authorized is True
    assert bound.stage("promotion").metrics["training_dataset_identity"] == "synthetic-training-dataset-v1"
    assert bound.stage("promotion").metrics["training_view_identity"] == "synthetic-training-view-v1"

    training_state["backend_identity"]["view_identity"] = "different-training-view"
    _write_json(tmp_path, "runs/v3/train-identity/state.json", training_state)
    mismatched = build_project_state(config)

    assert mismatched.stage("evaluation").status == StageStatus.BLOCKED
    assert mismatched.stage("evaluation").metrics["identity_binding_complete"] is False
    assert any("not an eligible checkpoint bound" in blocker for blocker in mismatched.stage("evaluation").blockers)
    assert mismatched.stage("promotion").production_authorized is False

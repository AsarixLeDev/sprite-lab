from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from spritelab.dev_features.audits import collect_audits
from spritelab.dev_features.cli import DeveloperCommandEnvironment, main
from spritelab.dev_features.projection import project_user_status
from spritelab.dev_features.state import build_developer_state
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, Evidence, ProjectState, StageState, StageStatus


def _config(root: Path) -> ProjectConfig:
    values = copy.deepcopy(DEFAULT_CONFIG)
    for section in ("dataset", "labeling", "training", "evaluation"):
        for key in values[section]:
            values[section][key] = [] if key == "review_queues" else ""
    values["paths"]["runs"] = "runs/v3"
    return ProjectConfig(root, None, values)


def _stage(
    key: str,
    *,
    audit: AuditStatus = AuditStatus.NOT_AUDITED,
    authorized: bool = False,
    evidence: list[Evidence] | None = None,
) -> StageState:
    return StageState(
        key=key,
        title=key.replace("-", " ").title(),
        status=StageStatus.COMPLETE if authorized else StageStatus.BLOCKED,
        explanation="Synthetic developer state.",
        blockers=[] if authorized else ["Synthetic blocker."],
        evidence=evidence or [],
        source_commit="a" * 40,
        audit=audit,
        implementation="IMPLEMENTED",
        production_authorized=authorized,
        metrics={"failed_gates": ["1", "5"]} if audit == AuditStatus.FAIL else {},
    )


def _state(root: Path, *, training_audit: AuditStatus = AuditStatus.FAIL, evidence: list[Evidence] | None = None):
    return ProjectState(
        project_name="synthetic",
        project_root=root,
        config_path=None,
        source_commit="a" * 40,
        stages=[
            _stage("semantic-labeling"),
            _stage("dataset-freeze"),
            _stage("training-infrastructure-audit", audit=training_audit, evidence=evidence),
            _stage("training-campaign"),
            _stage("memorization-review"),
            _stage("promotion-decision"),
        ],
    )


def _write_training_report(config: ProjectConfig, payload: object) -> Path:
    path = config.root / "training-audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config.values["training"]["audit_report"] = path.name
    return path


def test_status_exposes_detailed_developer_evidence(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "audit.json"
    artifact.write_text("{}\n", encoding="utf-8")
    evidence = Evidence(str(artifact), "f" * 64, "deadbeef")
    config = _config(tmp_path)
    state = _state(tmp_path, evidence=[evidence])
    environment = DeveloperCommandEnvironment(lambda: config, lambda _config: state)
    with pytest.raises(SystemExit) as caught:
        main(["status", "--json"], environment=environment)
    output = capsys.readouterr().out
    assert caught.value.code == 0
    assert "deadbeef" in output
    assert '"sha256"' in output
    assert '"branch"' in output
    assert '"audit": "FAIL"' in output


def test_stale_audit_is_not_current_certification(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_training_report(config, {"gates": {"1": "PASS"}, "commit": "b" * 40})
    audit = next(
        item
        for item in collect_audits(config, _state(tmp_path, training_audit=AuditStatus.STALE))
        if item["subsystem"] == "training-infrastructure"
    )
    assert audit["verdict"] == "STALE"
    assert audit["freshness"] == "STALE"
    assert audit["applicable"] is False
    assert audit["current_certification"] is False
    assert audit["authorization_consequence"] == "NO_CURRENT_CERTIFICATION"


def test_applicable_fail_lists_gates_and_blocks_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_training_report(config, {"gates": {"1": "FAIL", "2": "PASS"}})
    audit = next(
        item for item in collect_audits(config, _state(tmp_path)) if item["subsystem"] == "training-infrastructure"
    )
    assert audit["applicability"] == "APPLICABLE"
    assert audit["failed_gates"] == ["1", "5"]
    assert audit["authorization_consequence"] == "BLOCKS_TRAINING"


def test_applicable_pass_is_eligible_but_does_not_authorize_by_itself(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_training_report(config, {"gates": {"1": "PASS"}, "commit": "a" * 40})
    state = _state(tmp_path, training_audit=AuditStatus.PASS)
    audit = next(item for item in collect_audits(config, state) if item["subsystem"] == "training-infrastructure")
    assert audit["freshness"] == "FRESH"
    assert audit["current_certification"] is True
    assert audit["authorization_consequence"] == "ELIGIBLE_FOR_DEPENDENT_AUTHORIZATION"
    assert state.stage("training-campaign").production_authorized is False


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema_version":"old","schema_version":"spritelab.memorization.independent-audit-report.v1"}',
        '{"schema_version":"spritelab.memorization.independent-audit-report.v1","score":NaN}',
    ),
)
def test_developer_memorization_projection_uses_strict_shared_loader(tmp_path: Path, payload: str) -> None:
    config = _config(tmp_path)
    report = tmp_path / "memorization-audit.json"
    report.write_text(payload, encoding="utf-8")
    config.values["evaluation"]["memorization_audit"] = report.name

    audit = next(item for item in collect_audits(config, _state(tmp_path)) if item["subsystem"] == "memorization")

    assert audit["verdict"] == "NOT_COMPARABLE"
    assert audit["applicable"] is False
    assert audit["current_certification"] is False
    assert audit["staleness_reasons"] == ["audit_report_json_invalid"]


def test_user_projection_hides_hashes_branches_commits_and_audit_matrices(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = _state(tmp_path, evidence=[Evidence("audit.json", "f" * 64, "deadbeef")])
    details = build_developer_state(config, state)
    details["repository"] = {"branch": "feat/internal", "commit": "a" * 40}
    payload = json.dumps(project_user_status(details), sort_keys=True)
    assert "feat/internal" not in payload
    assert "deadbeef" not in payload
    assert "f" * 64 not in payload
    assert "audit" not in payload.lower()
    training = next(item for item in project_user_status(details)["areas"] if item["key"] == "training")
    assert training["message"] == "Not available yet"

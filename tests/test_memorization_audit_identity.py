from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from spritelab.evaluation.audit_identity import (
    MEMORIZATION_AUDIT_BOUND_FILES,
    MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION,
    MEMORIZATION_AUDIT_RECEIPT_SCHEMA,
    MEMORIZATION_AUDIT_REPORT_SCHEMA,
    MEMORIZATION_AUDIT_SEMANTIC_FILES,
    MEMORIZATION_AUDIT_SUBJECT_SCHEMA,
    MemorizationAuditIdentityError,
    load_memorization_audit_report,
    memorization_audit_code_identity,
)
from spritelab.product_features.evaluation.memorization_display import promotion_integrity_display
from spritelab.v3.model import AuditStatus
from spritelab.v3.status import (
    verify_memorization_audit_applicability,
    verify_memorization_audit_path,
)


def _materialize(root: Path) -> None:
    for relative in MEMORIZATION_AUDIT_BOUND_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"semantic line one\nsemantic line two\n")


def _report(root: Path, *, verdict: str = "PASS") -> dict[str, object]:
    code_identity = memorization_audit_code_identity(root)
    subject: dict[str, object] = {
        "schema_version": MEMORIZATION_AUDIT_SUBJECT_SCHEMA,
        "dataset_identity": "dataset-v5-synthetic",
        "training_view_identity": "training-view-synthetic",
        "freeze_manifest_sha256": "1" * 64,
        "campaign_identity_sha256": "2" * 64,
        "checkpoint_id": "checkpoint-synthetic",
        "checkpoint_weights": "ema",
        "checkpoint_sha256": "3" * 64,
        "benchmark_manifest_sha256": "4" * 64,
        "metric_definition_sha256": "5" * 64,
        "policy_identity_sha256": "6" * 64,
        "candidate_evidence_sha256": "7" * 64,
        "review_log_identity_sha256": "8" * 64,
        "code_identity_sha256": code_identity["code_identity_sha256"],
    }
    subject["audit_subject_sha256"] = hashlib.sha256(
        json.dumps(subject, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    auditor = {
        "auditor_id": "synthetic-independent-auditor",
        "implementation_identity_sha256": "9" * 64,
        "review_identity_sha256": "a" * 64,
    }
    report: dict[str, object] = {
        "schema_version": MEMORIZATION_AUDIT_REPORT_SCHEMA,
        "subsystem": "memorization",
        "audit_kind": "independent_memorization_integration",
        "evidence_role": "production_authority",
        "independent_audit": True,
        "overall_verdict": verdict,
        "authorization": {"checkpoint_promotion": verdict == "PASS"},
        "audit_subject": subject,
        "code_identity": code_identity,
        "auditor": auditor,
        "operation_identity_sha256": "b" * 64,
    }
    report_payload_sha256 = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    receipt: dict[str, object] = {
        "schema_version": MEMORIZATION_AUDIT_RECEIPT_SCHEMA,
        "audit_subject_sha256": subject["audit_subject_sha256"],
        "report_payload_sha256": report_payload_sha256,
        "operation_identity_sha256": report["operation_identity_sha256"],
        "auditor_id": auditor["auditor_id"],
        "server_managed": True,
        "terminal_status": "COMPLETE",
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    report["receipt"] = receipt
    report["audit_report_identity_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return report


def _rehash_identity(identity: dict[str, object]) -> None:
    payload = {key: value for key, value in identity.items() if key != "code_identity_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    identity["code_identity_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_v4_inventory_is_complete_deterministic_and_descriptive(tmp_path: Path) -> None:
    _materialize(tmp_path)
    identity = memorization_audit_code_identity(tmp_path)
    paths = [item["path"] for item in identity["bound_files"]]
    assert identity["contract_version"] == MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION
    assert identity["contract_version"] == "sprite_lab_memorization_audit_code_identity_v4"
    assert paths == sorted(paths) == list(MEMORIZATION_AUDIT_BOUND_FILES)
    assert len(paths) == len(set(paths))
    assert all(item["semantic_role"] and item["decision_effect"] for item in identity["bound_files"])
    assert len(identity["bound_files"]) == len(MEMORIZATION_AUDIT_SEMANTIC_FILES)
    assert {
        "src/spritelab/__main__.py",
        "src/spritelab/evaluation/conditional.py",
        "src/spritelab/evaluation/metric_definitions.py",
        "src/spritelab/evaluation/metrics.py",
        "src/spritelab/evaluation/strict_json.py",
        "src/spritelab/harvest/label_v4/risk.py",
        "src/spritelab/harvest/label_v4/training_quality.py",
        "src/spritelab/product_core/__init__.py",
        "src/spritelab/product_core/audit_evidence.py",
        "src/spritelab/product_core/contracts.py",
        "src/spritelab/product_core/events.py",
        "src/spritelab/product_core/plugins.py",
        "src/spritelab/product_core/web.py",
        "src/spritelab/product_features/dataset/certification.py",
        "src/spritelab/product_features/dataset/plugin.py",
        "src/spritelab/product_features/evaluation/checkpoints.py",
        "src/spritelab/product_features/evaluation/dashboard.py",
        "src/spritelab/product_features/evaluation/playground.py",
        "src/spritelab/product_features/training/dashboard.py",
        "src/spritelab/product_features/training/plans.py",
        "src/spritelab/product_features/training/service.py",
        "src/spritelab/product_runtime.py",
        "src/spritelab/product_web/events.py",
        "src/spritelab/remote_compute/contracts.py",
        "src/spritelab/remote_compute/local.py",
        "src/spritelab/remote_compute/ssh.py",
        "src/spritelab/training/campaign.py",
        "src/spritelab/training/launch.py",
        "src/spritelab/v3/model.py",
        "src/spritelab/v3/orchestration.py",
        "src/spritelab/v3/report.py",
        "src/spritelab/v3/run_state.py",
    } <= set(paths)
    assert len(paths) == 67


MUTATION_MATRIX = (
    ("product_event_validation", "src/spritelab/product_core/contracts.py"),
    ("strict_json_parser", "src/spritelab/product_core/events.py"),
    ("product_contract_facade", "src/spritelab/product_core/__init__.py"),
    ("controlled_api_projection", "src/spritelab/product_core/api.py"),
    ("audit_evidence_authorization_contract", "src/spritelab/product_core/audit_evidence.py"),
    ("product_cli_binding", "src/spritelab/product_core/cli.py"),
    ("evaluation_metrics", "src/spritelab/evaluation/metrics.py"),
    ("conditional_metrics", "src/spritelab/evaluation/conditional.py"),
    ("training_quality_projection", "src/spritelab/harvest/label_v4/training_quality.py"),
    ("training_quality_risk", "src/spritelab/harvest/label_v4/risk.py"),
    ("product_checkpoints", "src/spritelab/product_features/evaluation/checkpoints.py"),
    ("evaluation_dashboard", "src/spritelab/product_features/evaluation/dashboard.py"),
    ("detector_implementation", "src/spritelab/evaluation/memorization.py"),
    ("bundle_writer", "src/spritelab/evaluation/candidate_bundle.py"),
    ("machine_candidate_writer", "src/spritelab/evaluation/suite.py"),
    ("source_verifier", "src/spritelab/evaluation/promotion_decision.py"),
    ("review_loader", "src/spritelab/evaluation/memorization_review.py"),
    ("review_authoring", "src/spritelab/evaluation/memorization_review.py"),
    ("review_replay", "src/spritelab/evaluation/memorization_review.py"),
    ("dataset_training_authorization", "src/spritelab/product_features/dataset/certification.py"),
    ("product_review_discovery", "src/spritelab/product_features/dataset/web.py"),
    ("product_authority_display", "src/spritelab/product_features/evaluation/memorization_display.py"),
    ("action_adapter", "src/spritelab/product_features/evaluation/web.py"),
    ("promotion_recomputation", "src/spritelab/evaluation/promotion_decision.py"),
    ("promotion_projection", "src/spritelab/product_core/backend_contracts.py"),
    ("promotion_stage_model", "src/spritelab/v3/model.py"),
    ("active_checkpoint_projection", "src/spritelab/product_features/evaluation/checkpoints.py"),
    ("dataset_view_projection", "src/spritelab/product_features/evaluation/service.py"),
    ("benchmark_projection", "src/spritelab/product_features/evaluation/service.py"),
    ("exploratory_benchmark_separation", "src/spritelab/product_features/evaluation/playground.py"),
    ("training_campaign_dataset_view_identity", "src/spritelab/training/campaign.py"),
    ("training_launch_receipt_identity", "src/spritelab/training/launch.py"),
    ("training_checkpoint_provenance_projection", "src/spritelab/product_features/training/service.py"),
    ("v3_evaluation_promotion_projection", "src/spritelab/v3/orchestration.py"),
    ("offline_product_authority_report", "src/spritelab/v3/report.py"),
    ("training_checkpoint_event_projection", "src/spritelab/product_features/training/dashboard.py"),
    ("product_training_plan_projection", "src/spritelab/product_features/training/plans.py"),
    ("normal_product_plugin_composition", "src/spritelab/product_runtime.py"),
    ("remote_checkpoint_event_projection", "src/spritelab/remote_compute/local.py"),
    ("v3_durable_identity_projection", "src/spritelab/v3/run_state.py"),
    ("normal_v3_action_dispatch", "src/spritelab/v3/cli.py"),
)


@pytest.mark.parametrize(("case", "relative"), MUTATION_MATRIX, ids=[item[0] for item in MUTATION_MATRIX])
def test_each_required_semantic_mutation_stales_prior_audit(
    tmp_path: Path,
    case: str,
    relative: str,
) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    with (tmp_path / relative).open("ab") as handle:
        handle.write(f"# semantic mutation: {case}\n".encode())
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.STALE
    assert verification.reasons == ("code_identity_changed",)


@pytest.mark.parametrize("relative", MEMORIZATION_AUDIT_BOUND_FILES)
def test_every_inventory_file_mutation_changes_identity(tmp_path: Path, relative: str) -> None:
    _materialize(tmp_path)
    before = memorization_audit_code_identity(tmp_path)["code_identity_sha256"]
    with (tmp_path / relative).open("ab") as handle:
        handle.write(b"# changed\n")
    assert memorization_audit_code_identity(tmp_path)["code_identity_sha256"] != before


@pytest.mark.parametrize(
    "relative",
    (
        "docs/memorization.md",
        "experiments/evidence/remediation_report.md",
        "tests/test_unrelated.py",
        "src/spritelab/product_web/static/unrelated.css",
        "src/spritelab/product_features/providers/web.py",
    ),
)
def test_nonsemantic_change_keeps_identity_current_but_cannot_trust_embedded_receipt(
    tmp_path: Path, relative: str
) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unrelated\n", encoding="utf-8")
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.NOT_COMPARABLE
    assert verification.reasons == ("trusted_audit_receipt_unavailable",)
    assert verification.identity_current is True


@pytest.mark.parametrize(
    "relative",
    (
        "src/spritelab/evaluation/metrics.py",
        "src/spritelab/product_core/audit_evidence.py",
        "src/spritelab/product_features/dataset/certification.py",
        "src/spritelab/v3/report.py",
    ),
)
def test_missing_bound_file_fails_closed(tmp_path: Path, relative: str) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    (tmp_path / relative).unlink()
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.NOT_COMPARABLE
    assert verification.reasons[0].startswith("current_code_identity_unavailable:bound_source_missing:")
    with pytest.raises(MemorizationAuditIdentityError, match="bound_source_missing"):
        memorization_audit_code_identity(tmp_path)


def test_malformed_recorded_file_list_fails_closed(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    report["code_identity"]["bound_files"] = {"not": "a list"}
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.STALE
    assert "recorded_file_list_malformed" in verification.reasons


def test_changed_path_order_cannot_preserve_old_identity(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    identity = copy.deepcopy(report["code_identity"])
    identity["bound_files"].reverse()
    _rehash_identity(identity)
    report["code_identity"] = identity
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.STALE
    assert "recorded_file_order_or_uniqueness_invalid" in verification.reasons


def test_duplicate_recorded_path_fails_closed_even_with_recomputed_identity_hash(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    identity = copy.deepcopy(report["code_identity"])
    identity["bound_files"].insert(1, copy.deepcopy(identity["bound_files"][0]))
    _rehash_identity(identity)
    report["code_identity"] = identity

    verification = verify_memorization_audit_applicability(tmp_path, report)

    assert verification.status is AuditStatus.STALE
    assert "recorded_file_order_or_uniqueness_invalid" in verification.reasons
    assert "recorded_file_inventory_incomplete" in verification.reasons


def test_wrong_recorded_source_hash_fails_closed_even_with_recomputed_identity_hash(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    identity = copy.deepcopy(report["code_identity"])
    identity["bound_files"][0]["sha256"] = "f" * 64
    _rehash_identity(identity)
    report["code_identity"] = identity

    verification = verify_memorization_audit_applicability(tmp_path, report)

    assert verification.status is AuditStatus.STALE
    assert verification.reasons == ("code_identity_changed",)


def test_identity_level_subsystem_mismatch_fails_closed(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    identity = copy.deepcopy(report["code_identity"])
    identity["subsystem"] = "training"
    _rehash_identity(identity)
    report["code_identity"] = identity

    verification = verify_memorization_audit_applicability(tmp_path, report)

    assert verification.status is AuditStatus.STALE
    assert "recorded_identity_subsystem_mismatch" in verification.reasons


def test_line_endings_are_canonicalized_deterministically(tmp_path: Path) -> None:
    _materialize(tmp_path)
    relative = "src/spritelab/evaluation/metrics.py"
    path = tmp_path / relative
    path.write_bytes(b"first\nsecond\n")
    lf_identity = memorization_audit_code_identity(tmp_path)
    path.write_bytes(b"first\r\nsecond\r\n")
    crlf_identity = memorization_audit_code_identity(tmp_path)
    assert crlf_identity == lf_identity


@pytest.mark.parametrize(
    "legacy_version",
    [
        "sprite_lab_memorization_audit_code_identity_v2",
        "sprite_lab_memorization_audit_code_identity_v3",
    ],
)
def test_legacy_identity_is_stale_and_never_upgraded(tmp_path: Path, legacy_version: str) -> None:
    _materialize(tmp_path)
    report = {
        "overall_verdict": "PASS",
        "code_identity": {
            "contract_version": legacy_version,
            "bound_files": [],
        },
    }
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.STALE
    assert verification.reasons == ("legacy_or_incomplete_v3_code_identity",)
    assert not report["code_identity"]["contract_version"].endswith("v4")


def test_unsupported_future_identity_version_fails_closed(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    identity = copy.deepcopy(report["code_identity"])
    identity["contract_version"] = "sprite_lab_memorization_audit_code_identity_v999"
    _rehash_identity(identity)
    report["code_identity"] = identity

    verification = verify_memorization_audit_applicability(tmp_path, report)

    assert verification.status is AuditStatus.STALE
    assert verification.reasons == ("legacy_or_incomplete_v3_code_identity",)


def test_one_subsystem_audit_cannot_satisfy_memorization(tmp_path: Path) -> None:
    _materialize(tmp_path)
    report = _report(tmp_path)
    report["subsystem"] = "training"
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.NOT_COMPARABLE
    assert verification.reasons == ("audit_subsystem_mismatch",)


def test_product_projection_uses_shared_verifier_and_blocks_stale_wording(tmp_path: Path) -> None:
    _materialize(tmp_path)
    audit_path = tmp_path / "experiments/audit_report.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(_report(tmp_path)), encoding="utf-8")
    current = promotion_integrity_display(audit_path, repository_root=tmp_path)
    assert current["audit_applicability"] == "NOT_COMPARABLE"
    assert current["audit_applicability_reasons"] == ["trusted_audit_receipt_unavailable"]
    assert current["audit_code_identity_current"] is True
    assert current["integrity_certified"] is False
    with (tmp_path / "src/spritelab/evaluation/metrics.py").open("ab") as handle:
        handle.write(b"# semantic change\n")
    stale = promotion_integrity_display(audit_path, repository_root=tmp_path)
    assert stale["audit_applicability"] == "STALE"
    assert stale["audit_code_identity_current"] is False
    assert stale["integrity_certified"] is False
    assert stale["promotion_authorized"] is False
    assert "not currently certified" in stale["message"].lower()


def test_minimal_or_non_exact_pass_never_becomes_applicable_or_certified(tmp_path: Path) -> None:
    _materialize(tmp_path)
    minimal = {
        "subsystem": "memorization",
        "overall_verdict": "PASS",
        "authorization": {"checkpoint_promotion": True},
        "code_identity": memorization_audit_code_identity(tmp_path),
    }
    verification = verify_memorization_audit_applicability(tmp_path, minimal)
    assert verification.status is AuditStatus.STALE
    assert verification.reasons == ("legacy_or_incomplete_audit_report_contract",)
    report = _report(tmp_path)
    report["authorization"] = {"checkpoint_promotion": "true"}
    verification = verify_memorization_audit_applicability(tmp_path, report)
    assert verification.status is AuditStatus.NOT_COMPARABLE
    assert "audit_authorization_boolean_invalid" in verification.reasons


def test_duplicate_or_nonfinite_audit_json_is_not_certified(tmp_path: Path) -> None:
    _materialize(tmp_path)
    audit_path = tmp_path / "experiments/audit_report.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"overall_verdict":"FAIL","overall_verdict":"PASS"}', encoding="utf-8")
    duplicate = promotion_integrity_display(audit_path, repository_root=tmp_path)
    assert duplicate["integrity_certified"] is False
    assert duplicate["audit_applicability"] == "NOT_COMPARABLE"
    assert duplicate["audit_applicability_reasons"] == ["audit_report_json_invalid"]
    audit_path.write_text('{"overall_verdict":NaN}', encoding="utf-8")
    nonfinite = promotion_integrity_display(audit_path, repository_root=tmp_path)
    assert nonfinite["integrity_certified"] is False
    assert nonfinite["audit_applicability"] == "NOT_COMPARABLE"
    assert nonfinite["audit_applicability_reasons"] == ["audit_report_json_invalid"]


def test_shared_path_loader_rejects_non_object_and_bounds_input(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("[]", encoding="utf-8")
    root = load_memorization_audit_report(audit_path)
    assert root.errors == ("audit_report_root_invalid",)
    audit_path.write_bytes(b" " * 1_000_001)
    bounded = verify_memorization_audit_path(tmp_path, audit_path)
    assert bounded.status is AuditStatus.NOT_COMPARABLE
    assert bounded.reasons == ("audit_report_too_large",)


def test_product_and_developer_status_share_the_memorization_applicability_verifier() -> None:
    import inspect

    import spritelab.dev_features.audits as developer_audits
    import spritelab.v3.status as status_module

    assert developer_audits.verify_memorization_audit_path is status_module.verify_memorization_audit_path
    product_source = inspect.getsource(promotion_integrity_display)
    assert "from spritelab.v3.status import verify_memorization_audit_path" in product_source

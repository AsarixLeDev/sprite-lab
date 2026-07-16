from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests._labeling_audit import audit_context, copy_labeling_identity_root, write_labeling_audit

from spritelab.dev_features.audits import collect_audits
from spritelab.product_core import (
    AUDIT_HASH_MANIFEST_SCHEMA,
    CALIBRATION_READINESS,
    CONSERVATIVE_PROPOSAL_GENERATION,
    HUMAN_TRUTH,
    LABELING_BOUND_FILES,
    PRODUCTION_CONDITIONED_DATASET_FREEZE,
    TRAINING_INFRASTRUCTURE,
    ApplicabilityStatus,
    ArtifactVerificationStatus,
    BackendCapabilitySnapshot,
    CapabilityState,
    compute_labeling_audit_identity,
    verify_labeling_audit,
)
from spritelab.product_core.audit_evidence import LABELING_IDENTITY_SCHEMA
from spritelab.product_features.dataset.certification import (
    authorize_labeling_scope,
    labeling_capability,
    project_labeling_status,
)
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState, StageStatus

SOURCE_ROOT = Path(__file__).resolve().parents[1]
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
INTAKE_LABELING_BOUND_FILES = (
    "src/spritelab/dataset_maker/model.py",
    "src/spritelab/dataset_v5/identity.py",
    "src/spritelab/dataset_v5/raw_inventory.py",
    "src/spritelab/harvest/sources.py",
    "src/spritelab/harvest/suitability.py",
    "src/spritelab/product_core/__init__.py",
    "src/spritelab/product_features/dataset/evidence.py",
    "src/spritelab/product_features/dataset/packs.py",
    "src/spritelab/product_features/dataset/sheets.py",
    "src/spritelab/product_features/dataset/sidecar.py",
    "src/spritelab/product_features/dataset/static/metadata.js",
    "src/spritelab/product_features/dataset/templates/metadata.html",
)


def _verification(root: Path, report: Path | None, manifest: Path | None, *, commit: str = COMMIT_A):
    return verify_labeling_audit(
        root,
        report,
        manifest,
        authoritative_stage_source_commit=commit,
    )


def _copied_root(tmp_path: Path) -> Path:
    return copy_labeling_identity_root(SOURCE_ROOT, tmp_path / "repo")


def _mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _rehash(report: Path, manifest: Path) -> None:
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_HASH_MANIFEST_SCHEMA,
                "artifacts": [{"path": str(report.resolve()), "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )


def _current_pass(tmp_path: Path, *, scopes: tuple[str, ...] = (CONSERVATIVE_PROPOSAL_GENERATION,)):
    report, manifest = write_labeling_audit(SOURCE_ROOT, tmp_path / "audit", scopes=scopes)
    verification = _verification(SOURCE_ROOT, report, manifest)
    assert verification.is_current_pass
    assert verification.evidence is not None
    return verification


def test_ready_labeling_capability_without_audit_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="verified audit evidence"):
        BackendCapabilitySnapshot(
            backend_id="labeling",
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.READY,
            production_state=CapabilityState.READY,
        )


def test_ready_labeling_capability_with_malformed_audit_hash_is_rejected(tmp_path: Path) -> None:
    evidence = _current_pass(tmp_path).evidence
    assert evidence is not None
    with pytest.raises(ValueError, match="audit report sha256"):
        replace(evidence, audit_report_sha256="not-a-sha256")


def test_ready_labeling_capability_with_missing_code_identity_is_rejected(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(SOURCE_ROOT, tmp_path / "audit")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload.pop("bound_code_identity_sha256")
    report.write_text(json.dumps(payload), encoding="utf-8")
    _rehash(report, manifest)
    context = audit_context(SOURCE_ROOT, report, manifest)
    capability = labeling_capability(context)
    assert capability.independent_certification_state is CapabilityState.CERTIFICATION_PENDING
    assert capability.production_authorized is False


def test_audit_fail_never_produces_ready(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(SOURCE_ROOT, tmp_path / "audit", verdict="FAIL")
    capability = labeling_capability(audit_context(SOURCE_ROOT, report, manifest))
    assert capability.independent_certification_state is CapabilityState.BLOCKED
    assert capability.production_state is CapabilityState.BLOCKED


@pytest.mark.parametrize(
    "relative,old,new,reason",
    [
        (
            "src/spritelab/dataset_v5/conservative_labeling.py",
            "field_output_missing",
            "field_output_missing_changed",
            "code_identity_changed",
        ),
        (
            "src/spritelab/dataset_v5/conservative_labeling.py",
            "sprite_lab_conservative_reconciliation_v3",
            "sprite_lab_conservative_reconciliation_v4",
            "component_identity_changed:reconciliation",
        ),
        (
            "src/spritelab/dataset_v5/conservative_labeling.py",
            "sprite_lab_visual_taxonomy_hierarchy_v2",
            "sprite_lab_visual_taxonomy_hierarchy_v3",
            "taxonomy_identity_changed",
        ),
        (
            "src/spritelab/dataset_v5/conservative_labeling.py",
            "sprite_lab_field_health_gate_v3",
            "sprite_lab_field_health_gate_v4",
            "component_identity_changed:health_gate",
        ),
        (
            "src/spritelab/product_features/dataset/certification.py",
            '"""Identity-bound labeling capability adapter and plain-language projection."""',
            '"""Changed labeling capability adapter."""',
            "component_identity_changed:product_capability_adapter",
        ),
        (
            "src/spritelab/product_features/dataset/plugin.py",
            '"""Dataset intake ProductPlugin export."""',
            '"""Changed dataset status authorization projection."""',
            "component_identity_changed:product_status_projection",
        ),
    ],
)
def test_behavior_affecting_labeling_changes_make_pass_stale(
    tmp_path: Path, relative: str, old: str, new: str, reason: str
) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    _mutate(root / relative, old, new)
    verification = _verification(root, report, manifest)
    assert verification.applicability_status is ApplicabilityStatus.STALE
    assert verification.is_current_pass is False
    assert reason in verification.reasons


def test_applicable_limited_pass_grants_only_declared_scope(tmp_path: Path) -> None:
    verification = _current_pass(tmp_path)
    assert verification.authorizes(CONSERVATIVE_PROPOSAL_GENERATION)
    assert verification.authorized_scopes == (CONSERVATIVE_PROPOSAL_GENERATION,)


@pytest.mark.parametrize(
    "forbidden_scope",
    (HUMAN_TRUTH, CALIBRATION_READINESS, PRODUCTION_CONDITIONED_DATASET_FREEZE),
)
def test_limited_pass_does_not_grant_downstream_authority(tmp_path: Path, forbidden_scope: str) -> None:
    verification = _current_pass(tmp_path)
    assert verification.authorizes(forbidden_scope) is False


def test_legacy_bound_commit_exact_match_is_conservatively_applicable(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(
        SOURCE_ROOT,
        tmp_path / "audit",
        legacy=True,
        bound_commit=COMMIT_A,
        scopes=(CONSERVATIVE_PROPOSAL_GENERATION,),
    )
    verification = _verification(SOURCE_ROOT, report, manifest, commit=COMMIT_A)
    assert verification.applicability_status is ApplicabilityStatus.LEGACY_APPLICABLE
    assert verification.is_current_pass
    assert verification.reasons == ("legacy_bound_commit_exact_match_only",)


def test_legacy_bound_commit_mismatch_is_stale(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(
        SOURCE_ROOT,
        tmp_path / "audit",
        legacy=True,
        bound_commit=COMMIT_A,
    )
    verification = _verification(SOURCE_ROOT, report, manifest, commit=COMMIT_B)
    assert verification.applicability_status is ApplicabilityStatus.STALE
    assert verification.display_verdict == "STALE"


def test_new_identity_bound_audit_does_not_require_whole_repo_commit_equality(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(
        SOURCE_ROOT,
        tmp_path / "audit",
        bound_commit=COMMIT_A,
        scopes=(CONSERVATIVE_PROPOSAL_GENERATION,),
    )
    verification = _verification(SOURCE_ROOT, report, manifest, commit=COMMIT_B)
    assert verification.applicability_status is ApplicabilityStatus.APPLICABLE


def test_developer_projection_uses_same_labeling_verifier(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    values = copy.deepcopy(DEFAULT_CONFIG)
    values["labeling"].update(
        {
            "audit_report": str(report),
            "audit_hashes": str(manifest),
            "audit_stage_source_commit": COMMIT_A,
        }
    )
    config = ProjectConfig(root, None, values)
    stage = StageState(
        key="semantic-labeling",
        title="Semantic labeling",
        status=StageStatus.READY,
        explanation="Synthetic.",
        source_commit=COMMIT_A,
        audit=AuditStatus.PASS,
    )
    state = ProjectState("synthetic", root, None, COMMIT_A, [stage])
    first = collect_audits(config, state)[0]
    assert first["applicability"] == "APPLICABLE"
    assert first["verified_artifact_status"] == "VERIFIED"
    assert first["authorized_scopes"] == [CONSERVATIVE_PROPOSAL_GENERATION]
    _mutate(
        root / "src/spritelab/product_features/dataset/certification.py",
        '"""Identity-bound labeling capability adapter and plain-language projection."""',
        '"""Mutated adapter."""',
    )
    second = collect_audits(config, state)[0]
    assert second["verdict"] == "STALE"
    assert second["current_certification"] is False
    assert "code_identity_changed" in second["staleness_reasons"]


def test_product_projection_hides_hashes_but_preserves_consequences(tmp_path: Path) -> None:
    verification = _current_pass(tmp_path)
    payload = json.dumps(project_labeling_status(verification).to_public_dict(), sort_keys=True)
    assert verification.current_identity is not None
    assert verification.current_identity.code_identity_sha256 not in payload
    assert COMMIT_A not in payload
    assert "Available for broad suggestions" in payload
    assert "insufficient reviewed truth" in payload


def test_cached_ready_is_downgraded_after_code_mutation_and_restart(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    context = audit_context(root, report, manifest)
    cached = {"production_state": "READY", "audit_passed": True}
    before = labeling_capability(context, persisted_capability=cached)
    assert before.production_state is CapabilityState.READY
    _mutate(
        root / "src/spritelab/product_features/dataset/certification.py",
        '"""Identity-bound labeling capability adapter and plain-language projection."""',
        '"""Restart mutation."""',
    )
    after_restart = labeling_capability(context, persisted_capability=cached)
    assert after_restart.production_state is CapabilityState.STALE
    assert after_restart.production_authorized is False


def test_action_time_revalidation_blocks_evidence_that_became_stale(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    context = audit_context(root, report, manifest)
    assert authorize_labeling_scope(context, CONSERVATIVE_PROPOSAL_GENERATION).authorized
    _mutate(
        root / "src/spritelab/product_features/dataset/semantics.py",
        '"""Optional semantic proposals through the shared VisionProvider contract."""',
        '"""Changed action adapter."""',
    )
    authorization = authorize_labeling_scope(context, CONSERVATIVE_PROPOSAL_GENERATION)
    assert authorization.authorized is False


def test_missing_audit_artifact_is_not_comparable(tmp_path: Path) -> None:
    manifest = tmp_path / "artifact_hashes.json"
    manifest.write_text(json.dumps({"schema_version": AUDIT_HASH_MANIFEST_SCHEMA, "artifacts": []}), encoding="utf-8")
    verification = _verification(SOURCE_ROOT, tmp_path / "missing.json", manifest)
    assert verification.applicability_status is ApplicabilityStatus.NOT_COMPARABLE
    assert verification.artifact_status is ArtifactVerificationStatus.MISSING


def test_changed_audit_artifact_bytes_are_not_comparable(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(SOURCE_ROOT, tmp_path / "audit")
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    verification = _verification(SOURCE_ROOT, report, manifest)
    assert verification.applicability_status is ApplicabilityStatus.NOT_COMPARABLE
    assert verification.artifact_status is ArtifactVerificationStatus.HASH_MISMATCH


@pytest.mark.parametrize("relative", ("docs/unrelated.md", "src/spritelab/product_web/static/cosmetic.css"))
def test_unrelated_documentation_and_css_do_not_stale_semantic_certification(tmp_path: Path, relative: str) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("purely unrelated presentation text\n", encoding="utf-8")
    assert _verification(root, report, manifest).applicability_status is ApplicabilityStatus.APPLICABLE


def test_unrelated_training_slice_in_shared_status_module_does_not_stale_labeling(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    _mutate(
        root / "src/spritelab/v3/status.py",
        "def _training_audit_status(",
        "def _training_audit_status_changed(",
    )
    verification = _verification(root, report, manifest)
    assert verification.applicability_status is ApplicabilityStatus.APPLICABLE


def test_source_checkout_line_endings_do_not_stale_labeling(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    path = root / "src/spritelab/product_features/dataset/certification.py"
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
    verification = _verification(root, report, manifest)
    assert verification.applicability_status is ApplicabilityStatus.APPLICABLE


def test_product_capability_probing_is_side_effect_free(tmp_path: Path) -> None:
    report, manifest = write_labeling_audit(
        SOURCE_ROOT,
        tmp_path / "audit",
        scopes=(CONSERVATIVE_PROPOSAL_GENERATION,),
    )
    context = audit_context(SOURCE_ROOT, report, manifest)
    before = {path.name: path.stat().st_mtime_ns for path in (report, manifest)}
    capability = labeling_capability(context)
    after = {path.name: path.stat().st_mtime_ns for path in (report, manifest)}
    assert capability.production_authorized
    assert before == after


def test_no_provider_call_occurs_during_evidence_verification(tmp_path: Path) -> None:
    calls = {"probe": 0, "execute": 0}

    class Provider:
        def probe(self, _context):
            calls["probe"] += 1

        def execute(self, *_args):
            calls["execute"] += 1

    provider = Provider()
    report, manifest = write_labeling_audit(SOURCE_ROOT, tmp_path / "audit")
    assert _verification(SOURCE_ROOT, report, manifest).artifact_status is ArtifactVerificationStatus.VERIFIED
    assert provider is not None
    assert calls == {"probe": 0, "execute": 0}


def test_current_identity_records_complete_files_and_required_components() -> None:
    identity = compute_labeling_audit_identity(SOURCE_ROOT)
    components = dict(identity.component_identities)
    assert components.keys() >= {
        "response_normalization",
        "taxonomy",
        "reconciliation",
        "health_gate",
        "product_capability_adapter",
        "product_status_projection",
        "review_routing",
    }
    assert len(identity.bound_files) >= 14


def test_labeling_identity_binds_universal_intake_helpers() -> None:
    assert LABELING_IDENTITY_SCHEMA == "spritelab.labeling-audit-identity.v3"
    assert set(INTAKE_LABELING_BOUND_FILES) <= set(LABELING_BOUND_FILES)
    assert LABELING_BOUND_FILES == tuple(sorted(set(LABELING_BOUND_FILES)))


@pytest.mark.parametrize(
    "relative,old,new",
    [
        (
            "src/spritelab/dataset_maker/model.py",
            're.compile(r"[^a-z0-9_.-]+")',
            're.compile(r"[^a-z0-9_-]+")',
        ),
        (
            "src/spritelab/dataset_v5/identity.py",
            'BLOB_ID_VERSION = "decoded_rgba_v1"',
            'BLOB_ID_VERSION = "decoded_rgba_v2"',
        ),
        (
            "src/spritelab/dataset_v5/raw_inventory.py",
            "digest = hashlib.sha256()",
            "digest = hashlib.sha512()",
        ),
        (
            "src/spritelab/harvest/sources.py",
            "return normalize_license_name(license_name) in TRAINING_ALLOWED_LICENSES",
            "return True",
        ),
        (
            "src/spritelab/harvest/suitability.py",
            "target_height=None, max_dimension=512",
            "target_height=None, max_dimension=513",
        ),
        (
            "src/spritelab/product_core/__init__.py",
            "    VisionProvider,",
            "    VisionProvider as VisionProviderV2,",
        ),
        (
            "src/spritelab/product_features/dataset/evidence.py",
            '"source.yaml",',
            '"source-v2.yaml",',
        ),
        (
            "src/spritelab/product_features/dataset/packs.py",
            "spritelab.dataset.pack_detection.v1",
            "spritelab.dataset.pack_detection.v2",
        ),
        (
            "src/spritelab/product_features/dataset/sheets.py",
            "spritelab.dataset.sheet_extraction_policy.v1",
            "spritelab.dataset.sheet_extraction_policy.v2",
        ),
        (
            "src/spritelab/product_features/dataset/sidecar.py",
            "spritelab.dataset.pack_metadata.v2",
            "spritelab.dataset.pack_metadata.v3",
        ),
        (
            "src/spritelab/product_features/dataset/static/metadata.js",
            "data.original_work_declaration=form.elements.original_work_declaration.checked;",
            "data.original_work_declaration=true;",
        ),
        (
            "src/spritelab/product_features/dataset/templates/metadata.html",
            '<option value="" {% if not license %}selected{% endif %}>',
            '<option value="cc0" selected>',
        ),
    ],
)
def test_universal_intake_helper_changes_revoke_labeling_audit(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    _mutate(root / relative, old, new)

    verification = _verification(root, report, manifest)

    assert verification.applicability_status is ApplicabilityStatus.STALE
    assert verification.authorized_scopes == ()
    assert "code_identity_changed" in verification.reasons


@pytest.mark.parametrize("relative", INTAKE_LABELING_BOUND_FILES)
def test_missing_universal_intake_dependency_revokes_labeling_audit(tmp_path: Path, relative: str) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    (root / relative).unlink()

    verification = _verification(root, report, manifest)

    assert verification.applicability_status is ApplicabilityStatus.NOT_COMPARABLE
    assert verification.authorized_scopes == ()
    assert verification.reasons == (f"current_identity_unavailable:bound_file_missing:{relative}",)


def test_action_time_revalidation_blocks_changed_intake_sidecar_contract(tmp_path: Path) -> None:
    root = _copied_root(tmp_path)
    report, manifest = write_labeling_audit(root, tmp_path / "audit", scopes=(CONSERVATIVE_PROPOSAL_GENERATION,))
    context = audit_context(root, report, manifest)
    assert authorize_labeling_scope(context, CONSERVATIVE_PROPOSAL_GENERATION).authorized
    _mutate(
        root / "src/spritelab/product_features/dataset/sidecar.py",
        "spritelab.dataset.pack_metadata.v2",
        "spritelab.dataset.pack_metadata.v3",
    )

    authorization = authorize_labeling_scope(context, CONSERVATIVE_PROPOSAL_GENERATION)

    assert authorization.authorized is False


def test_shared_evidence_pattern_preserves_subsystem_and_scope_boundaries(tmp_path: Path) -> None:
    labeling = _current_pass(tmp_path).evidence
    assert labeling is not None
    training = replace(
        labeling,
        subsystem="training",
        audit_kind="training_infrastructure",
        authorized_scopes=(TRAINING_INFRASTRUCTURE,),
    )
    memorization = replace(
        labeling,
        subsystem="memorization",
        audit_kind="memorization_integration",
        authorized_scopes=(),
    )
    assert training.authorizes(TRAINING_INFRASTRUCTURE, subsystem="training")
    assert training.authorizes(PRODUCTION_CONDITIONED_DATASET_FREEZE, subsystem="training") is False
    assert memorization.is_current_pass
    assert memorization.authorized_scopes == ()
    with pytest.raises(ValueError, match="one subsystem"):
        BackendCapabilitySnapshot(
            backend_id="training",
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.READY,
            production_state=CapabilityState.READY,
            audit_evidence=labeling,
        )

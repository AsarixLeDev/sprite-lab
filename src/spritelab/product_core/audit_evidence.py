"""Fail-closed, identity-bound independent audit evidence.

The verifier in this module is deliberately read-only.  It hashes local files,
validates audit metadata, and recomputes applicability without importing a
provider, starting a backend, or trusting a persisted readiness value.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

VERIFIED_AUDIT_EVIDENCE_SCHEMA = "spritelab.verified-audit-evidence.v1"
AUDIT_REPORT_SCHEMA = "spritelab.independent-audit-report.v1"
AUDIT_HASH_MANIFEST_SCHEMA = "spritelab.audit-artifact-hashes.v1"
LABELING_IDENTITY_SCHEMA = "spritelab.labeling-audit-identity.v3"

CONSERVATIVE_PROPOSAL_GENERATION = "conservative_proposal_generation"
CONDITIONED_VIEW_CANDIDATES = "conditioned_view_candidates"
HUMAN_TRUTH = "human_truth"
CALIBRATION_READINESS = "calibration_readiness"
STRONG_SUPERVISION = "strong_supervision"
PRODUCTION_CONDITIONED_DATASET_FREEZE = "production_conditioned_dataset_freeze"
BULK_UNREVIEWED_EXACT_OBJECT_LABELS = "bulk_unreviewed_exact_object_labels"
TRAINING_INFRASTRUCTURE = "training_infrastructure"
CHECKPOINT_PROMOTION = "checkpoint_promotion"

KNOWN_AUTHORIZATION_SCOPES = frozenset(
    {
        CONSERVATIVE_PROPOSAL_GENERATION,
        CONDITIONED_VIEW_CANDIDATES,
        HUMAN_TRUTH,
        CALIBRATION_READINESS,
        STRONG_SUPERVISION,
        PRODUCTION_CONDITIONED_DATASET_FREEZE,
        BULK_UNREVIEWED_EXACT_OBJECT_LABELS,
        TRAINING_INFRASTRUCTURE,
        CHECKPOINT_PROMOTION,
    }
)

# This is the complete semantic applicability boundary.  Documentation and
# cosmetic assets are intentionally absent.  A change to any listed file
# invalidates a new identity-bound audit.
LABELING_BOUND_FILES = (
    "src/spritelab/dataset_maker/model.py",
    "src/spritelab/dataset_v5/conservative_labeling.py",
    "src/spritelab/dataset_v5/identity.py",
    "src/spritelab/dataset_v5/raw_inventory.py",
    "src/spritelab/harvest/sources.py",
    "src/spritelab/harvest/suitability.py",
    "src/spritelab/product_core/__init__.py",
    "src/spritelab/product_core/audit_evidence.py",
    "src/spritelab/product_core/backend_contracts.py",
    "src/spritelab/product_core/contracts.py",
    "src/spritelab/product_features/dataset/certification.py",
    "src/spritelab/product_features/dataset/cli.py",
    "src/spritelab/product_features/dataset/evidence.py",
    "src/spritelab/product_features/dataset/intake.py",
    "src/spritelab/product_features/dataset/packs.py",
    "src/spritelab/product_features/dataset/plugin.py",
    "src/spritelab/product_features/dataset/review.py",
    "src/spritelab/product_features/dataset/semantics.py",
    "src/spritelab/product_features/dataset/sheets.py",
    "src/spritelab/product_features/dataset/sidecar.py",
    "src/spritelab/product_features/dataset/static/metadata.js",
    "src/spritelab/product_features/dataset/templates/metadata.html",
    "src/spritelab/product_features/dataset/web.py",
    "src/spritelab/v3/config.py",
    "src/spritelab/v3/model.py",
    "src/spritelab/v3/orchestration.py",
    "src/spritelab/v3/status.py",
)

_CONTRACT_ASSIGNMENTS = (
    "CONTRACT_VERSION",
    "TAXONOMY_VERSION",
    "FIELD_STATE_VERSION",
    "COMPARISON_VERSION",
    "HEALTH_GATE_VERSION",
    "RECONCILIATION_VERSION",
    "REVIEW_QUEUE_VERSION",
    "REVIEW_EVENT_VERSION",
    "CALIBRATION_INPUT_VERSION",
    "PROMPT_VERSION",
    "CORE_FIELDS",
    "CONDITIONAL_FIELDS",
    "AUXILIARY_FIELDS",
    "FIELD_STATES",
    "ABSTENTION_REASONS",
    "COMPARISON_CLASSIFICATIONS",
    "CALIBRATION_STATES",
    "REVIEW_ACTIONS",
)
_CONTRACT_FUNCTIONS = (
    "taxonomy_hierarchy",
    "hierarchical_label_contract",
    "field_state_contract",
    "disagreement_classification_contract",
    "field_health_gate_contract",
    "reconciliation_contract",
    "review_queue_contract",
    "calibration_input_contract",
    "conservative_prompt_v2",
    "conservative_output_schema",
)
_TAXONOMY_NODES = ("TAXONOMY_VERSION", "_HIERARCHIES", "_SAFE_SIBLING_PARENTS")
_COMPONENT_SELECTORS: dict[str, tuple[str, ...]] = {
    "provider_neutral_labeling_contract": (
        "ProviderAdapter",
        "provider_neutral_request",
        "hierarchical_label_contract",
        "conservative_output_schema",
    ),
    "response_normalization": ("validate_field", "adapt_historical_label"),
    "taxonomy": ("TAXONOMY_VERSION", "_HIERARCHIES", "_SAFE_SIBLING_PARENTS", "taxonomy_hierarchy"),
    "hierarchy_compatibility": ("_parents", "_ancestors", "safest_shared_parent"),
    "field_comparison": ("COMPARISON_VERSION", "compare_field"),
    "reconciliation": ("RECONCILIATION_VERSION", "reconciliation_contract", "reconcile_proposals"),
    "health_gate": ("HEALTH_GATE_VERSION", "field_health_gate_contract", "field_health_report"),
    "salvage_classification": ("reconcile_proposals", "stage_independent_audit"),
    "calibration_input_generation": (
        "CALIBRATION_INPUT_VERSION",
        "calibration_input_contract",
        "build_calibration_inputs",
        "calibration_readiness",
    ),
}

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class AuditVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ApplicabilityStatus(str, Enum):
    APPLICABLE = "APPLICABLE"
    LEGACY_APPLICABLE = "LEGACY_APPLICABLE"
    STALE = "STALE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ArtifactVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MISSING = "MISSING"
    HASH_MISMATCH = "HASH_MISMATCH"
    MALFORMED = "MALFORMED"


@dataclass(frozen=True)
class BoundFileIdentity:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("bound file paths must be non-empty repository-relative paths")
        _require_sha256(self.sha256, "bound file sha256")


@dataclass(frozen=True)
class LabelingAuditIdentity:
    schema_version: str
    code_identity_sha256: str
    contract_identity_sha256: str
    data_identity_sha256: str
    bound_files: tuple[BoundFileIdentity, ...]
    component_identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version != LABELING_IDENTITY_SCHEMA:
            raise ValueError("unsupported labeling identity schema")
        _require_sha256(self.code_identity_sha256, "code identity")
        _require_sha256(self.contract_identity_sha256, "contract identity")
        _require_sha256(self.data_identity_sha256, "data identity")
        if tuple(item.path for item in self.bound_files) != LABELING_BOUND_FILES:
            raise ValueError("labeling identity must record the complete ordered bound-file list")
        names = tuple(name for name, _digest in self.component_identities)
        if len(names) != len(set(names)) or not names:
            raise ValueError("component identities must be non-empty and unique")
        for name, digest in self.component_identities:
            if not name:
                raise ValueError("component identity name cannot be empty")
            _require_sha256(digest, f"component identity {name}")

    def component(self, name: str) -> str:
        try:
            return dict(self.component_identities)[name]
        except KeyError as exc:
            raise KeyError(f"unknown labeling identity component: {name}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_identity_sha256": self.code_identity_sha256,
            "contract_identity_sha256": self.contract_identity_sha256,
            "data_identity_sha256": self.data_identity_sha256,
            "bound_files": [{"path": item.path, "sha256": item.sha256} for item in self.bound_files],
            "component_identities": dict(self.component_identities),
        }


@dataclass(frozen=True)
class VerifiedAuditEvidence:
    """Immutable evidence created only by successful artifact/schema verification."""

    schema_version: str
    subsystem: str
    audit_kind: str
    verdict: AuditVerdict
    audit_report_path: str
    audit_report_sha256: str
    artifact_hash_manifest_path: str
    artifact_hash_manifest_sha256: str
    bound_commit: str
    bound_code_identity_sha256: str
    bound_contract_identity_sha256: str
    bound_data_identity_sha256: str | None
    auditor_identity: str
    created_at_utc: str
    applicability_status: ApplicabilityStatus
    applicability_reasons: tuple[str, ...]
    authorized_scopes: tuple[str, ...] = ()
    bound_component_identities: tuple[tuple[str, str], ...] = ()
    evidence_role: str = "independent_certification"

    def __post_init__(self) -> None:
        if self.schema_version != VERIFIED_AUDIT_EVIDENCE_SCHEMA:
            raise ValueError("unsupported verified audit evidence schema")
        if not self.subsystem.strip() or not self.audit_kind.strip():
            raise ValueError("verified audit evidence requires subsystem and audit kind")
        if self.evidence_role != "independent_certification":
            raise ValueError("remediation evidence is not independent certification")
        if not self.audit_report_path or not self.artifact_hash_manifest_path:
            raise ValueError("verified evidence requires report and artifact manifest paths")
        _require_sha256(self.audit_report_sha256, "audit report sha256")
        _require_sha256(self.artifact_hash_manifest_sha256, "artifact manifest sha256")
        if not _HEX_40.fullmatch(self.bound_commit):
            raise ValueError("bound_commit must be a full lowercase git commit identity")
        _require_sha256(self.bound_code_identity_sha256, "bound code identity")
        _require_sha256(self.bound_contract_identity_sha256, "bound contract identity")
        if self.bound_data_identity_sha256 is not None:
            _require_sha256(self.bound_data_identity_sha256, "bound data identity")
        if not self.auditor_identity.strip():
            raise ValueError("verified evidence requires an auditor or audit-run identity")
        _require_utc_timestamp(self.created_at_utc)
        if len(self.authorized_scopes) != len(set(self.authorized_scopes)):
            raise ValueError("authorized scopes cannot contain duplicates")
        unknown = set(self.authorized_scopes) - KNOWN_AUTHORIZATION_SCOPES
        if unknown:
            raise ValueError(f"unknown authorization scope(s): {', '.join(sorted(unknown))}")
        if self.verdict is not AuditVerdict.PASS and self.authorized_scopes:
            raise ValueError("only an independent PASS may carry authorization scopes")
        if (
            self.applicability_status
            not in {
                ApplicabilityStatus.APPLICABLE,
                ApplicabilityStatus.LEGACY_APPLICABLE,
            }
            and not self.applicability_reasons
        ):
            raise ValueError("non-applicable evidence requires applicability reasons")
        component_names = [name for name, _digest in self.bound_component_identities]
        if len(component_names) != len(set(component_names)):
            raise ValueError("bound component identities cannot contain duplicates")
        for name, digest in self.bound_component_identities:
            if not name:
                raise ValueError("bound component identity name cannot be empty")
            _require_sha256(digest, f"bound component identity {name}")

    @property
    def is_current_pass(self) -> bool:
        return self.verdict is AuditVerdict.PASS and self.applicability_status in {
            ApplicabilityStatus.APPLICABLE,
            ApplicabilityStatus.LEGACY_APPLICABLE,
        }

    def authorizes(self, scope: str, *, subsystem: str | None = None) -> bool:
        return (
            self.is_current_pass
            and (subsystem is None or subsystem == self.subsystem)
            and scope in self.authorized_scopes
        )

    def to_internal_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subsystem": self.subsystem,
            "audit_kind": self.audit_kind,
            "verdict": self.verdict.value,
            "audit_report_path": self.audit_report_path,
            "audit_report_sha256": self.audit_report_sha256,
            "artifact_hash_manifest_path": self.artifact_hash_manifest_path,
            "artifact_hash_manifest_sha256": self.artifact_hash_manifest_sha256,
            "bound_commit": self.bound_commit,
            "bound_code_identity_sha256": self.bound_code_identity_sha256,
            "bound_contract_identity_sha256": self.bound_contract_identity_sha256,
            "bound_data_identity_sha256": self.bound_data_identity_sha256,
            "bound_component_identities": dict(self.bound_component_identities),
            "auditor_identity": self.auditor_identity,
            "created_at_utc": self.created_at_utc,
            "applicability_status": self.applicability_status.value,
            "applicability_reasons": list(self.applicability_reasons),
            "authorized_scopes": list(self.authorized_scopes),
            "evidence_role": self.evidence_role,
        }


@dataclass(frozen=True)
class AuditVerification:
    subsystem: str
    applicability_status: ApplicabilityStatus
    reasons: tuple[str, ...]
    artifact_status: ArtifactVerificationStatus
    current_identity: LabelingAuditIdentity | None
    evidence: VerifiedAuditEvidence | None = None
    report_verdict: AuditVerdict | None = None
    bound_commit: str | None = None

    @property
    def authorized_scopes(self) -> tuple[str, ...]:
        return self.evidence.authorized_scopes if self.evidence and self.evidence.is_current_pass else ()

    @property
    def is_current_pass(self) -> bool:
        return bool(self.evidence and self.evidence.is_current_pass)

    @property
    def display_verdict(self) -> str:
        if self.applicability_status is ApplicabilityStatus.STALE:
            return "STALE"
        if self.applicability_status is ApplicabilityStatus.NOT_COMPARABLE:
            return "NOT_COMPARABLE"
        return self.report_verdict.value if self.report_verdict else "NOT_AUDITED"

    def authorizes(self, scope: str) -> bool:
        return bool(self.evidence and self.evidence.authorizes(scope, subsystem=self.subsystem))


class IdentityComputationError(ValueError):
    """The current applicability identity could not be computed safely."""


def compute_labeling_audit_identity(root: Path) -> LabelingAuditIdentity:
    """Hash the complete labeling semantic boundary without executing it."""

    resolved_root = root.resolve()
    bound_files: list[BoundFileIdentity] = []
    for relative in LABELING_BOUND_FILES:
        path = resolved_root / relative
        if not path.is_file():
            raise IdentityComputationError(f"bound_file_missing:{relative}")
        bound_files.append(BoundFileIdentity(relative, _bound_file_sha256(path, relative)))

    conservative_path = resolved_root / "src/spritelab/dataset_v5/conservative_labeling.py"
    component_nodes = {selector for selectors in _COMPONENT_SELECTORS.values() for selector in selectors}
    semantic_nodes = _selected_ast_nodes(
        conservative_path,
        set(_CONTRACT_ASSIGNMENTS) | set(_CONTRACT_FUNCTIONS) | set(_TAXONOMY_NODES) | component_nodes,
    )
    missing_contracts = sorted((set(_CONTRACT_ASSIGNMENTS) | set(_CONTRACT_FUNCTIONS)) - semantic_nodes.keys())
    if missing_contracts:
        raise IdentityComputationError(f"contract_nodes_missing:{','.join(missing_contracts)}")
    missing_taxonomy = sorted(set(_TAXONOMY_NODES) - semantic_nodes.keys())
    if missing_taxonomy:
        raise IdentityComputationError(f"taxonomy_nodes_missing:{','.join(missing_taxonomy)}")

    contract_payload = {name: semantic_nodes[name] for name in (*_CONTRACT_ASSIGNMENTS, *_CONTRACT_FUNCTIONS)}
    taxonomy_payload = {name: semantic_nodes[name] for name in _TAXONOMY_NODES}
    component_identities = _component_identities(resolved_root, semantic_nodes)
    code_payload = {
        "schema_version": LABELING_IDENTITY_SCHEMA,
        "bound_files": [{"path": item.path, "sha256": item.sha256} for item in bound_files],
    }
    return LabelingAuditIdentity(
        schema_version=LABELING_IDENTITY_SCHEMA,
        code_identity_sha256=_canonical_sha256(code_payload),
        contract_identity_sha256=_canonical_sha256(contract_payload),
        data_identity_sha256=_canonical_sha256(taxonomy_payload),
        bound_files=tuple(bound_files),
        component_identities=tuple(sorted(component_identities.items())),
    )


def verify_labeling_audit(
    root: Path,
    report_path: Path | None,
    manifest_path: Path | None,
    *,
    authoritative_stage_source_commit: str | None,
) -> AuditVerification:
    """Verify artifacts, schema, identity, and applicability in one fail-closed path."""

    try:
        current_identity = compute_labeling_audit_identity(root)
    except (OSError, UnicodeError, SyntaxError, IdentityComputationError, ValueError) as exc:
        return AuditVerification(
            subsystem="labeling",
            applicability_status=ApplicabilityStatus.NOT_COMPARABLE,
            reasons=(f"current_identity_unavailable:{exc}",),
            artifact_status=ArtifactVerificationStatus.MALFORMED,
            current_identity=None,
        )
    if report_path is None or not report_path.is_file():
        return _unverified(current_identity, ArtifactVerificationStatus.MISSING, "audit_report_missing")

    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _unverified(current_identity, ArtifactVerificationStatus.MALFORMED, f"artifact_malformed:{exc}")
    if not isinstance(report, dict):
        return _unverified(current_identity, ArtifactVerificationStatus.MALFORMED, "artifact_root_not_object")
    declared_verdict = _declared_verdict(report)
    declared_commit = report.get("bound_commit")
    declared_commit = (
        declared_commit if isinstance(declared_commit, str) and _HEX_40.fullmatch(declared_commit) else None
    )
    if manifest_path is None or not manifest_path.is_file():
        return _unverified(
            current_identity,
            ArtifactVerificationStatus.MISSING,
            "artifact_hash_manifest_missing",
            verdict=declared_verdict,
            bound_commit=declared_commit,
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _unverified(
            current_identity,
            ArtifactVerificationStatus.MALFORMED,
            f"artifact_malformed:{exc}",
            verdict=declared_verdict,
            bound_commit=declared_commit,
        )
    if not isinstance(manifest, dict):
        return _unverified(
            current_identity,
            ArtifactVerificationStatus.MALFORMED,
            "artifact_root_not_object",
            verdict=declared_verdict,
            bound_commit=declared_commit,
        )
    artifact_reason = _verify_report_hash(root, report_path, manifest, _sha256_bytes(report_bytes))
    if artifact_reason:
        status = (
            ArtifactVerificationStatus.HASH_MISMATCH
            if artifact_reason == "audit_report_hash_mismatch"
            else ArtifactVerificationStatus.MALFORMED
        )
        return _unverified(
            current_identity,
            status,
            artifact_reason,
            verdict=declared_verdict,
            bound_commit=declared_commit,
        )

    common, schema_reason = _validated_common_report(report)
    if schema_reason:
        return _unverified(current_identity, ArtifactVerificationStatus.VERIFIED, schema_reason)
    assert common is not None
    verdict: AuditVerdict = common["verdict"]
    bound_commit: str = common["bound_commit"]
    report_sha256 = _sha256_bytes(report_bytes)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    identity_keys = {
        "bound_code_identity_sha256",
        "bound_contract_identity_sha256",
        "bound_data_identity_sha256",
        "bound_component_identities",
    }
    present_identity_keys = identity_keys & report.keys()
    legacy = not present_identity_keys
    reasons: list[str] = []
    status: ApplicabilityStatus
    if legacy:
        if authoritative_stage_source_commit is None or not _HEX_40.fullmatch(authoritative_stage_source_commit):
            status = ApplicabilityStatus.NOT_COMPARABLE
            reasons.append("authoritative_stage_source_commit_missing")
        elif bound_commit != authoritative_stage_source_commit:
            status = ApplicabilityStatus.STALE
            reasons.append("legacy_bound_commit_mismatch")
        else:
            status = ApplicabilityStatus.LEGACY_APPLICABLE
            reasons.append("legacy_bound_commit_exact_match_only")
        bound_code = current_identity.code_identity_sha256
        bound_contract = current_identity.contract_identity_sha256
        bound_data = current_identity.data_identity_sha256
        bound_components = current_identity.component_identities
    else:
        if present_identity_keys != identity_keys:
            return _unverified(
                current_identity,
                ArtifactVerificationStatus.VERIFIED,
                "incomplete_versioned_identity",
                verdict=verdict,
                bound_commit=bound_commit,
            )
        identity_reason = _validate_report_identities(report, current_identity)
        if identity_reason:
            return _unverified(
                current_identity,
                ArtifactVerificationStatus.VERIFIED,
                identity_reason,
                verdict=verdict,
                bound_commit=bound_commit,
            )
        bound_code = str(report["bound_code_identity_sha256"])
        bound_contract = str(report["bound_contract_identity_sha256"])
        bound_data = str(report["bound_data_identity_sha256"])
        bound_components = tuple(
            sorted((str(key), str(value)) for key, value in report["bound_component_identities"].items())
        )
        if bound_code != current_identity.code_identity_sha256:
            reasons.append("code_identity_changed")
        if bound_contract != current_identity.contract_identity_sha256:
            reasons.append("contract_identity_changed")
        if bound_data != current_identity.data_identity_sha256:
            reasons.append("taxonomy_identity_changed")
        current_components = dict(current_identity.component_identities)
        for name, digest in bound_components:
            if current_components.get(name) != digest:
                reasons.append(f"component_identity_changed:{name}")
        status = ApplicabilityStatus.STALE if reasons else ApplicabilityStatus.APPLICABLE

    declared_scopes: tuple[str, ...] = common["authorized_scopes"]
    scopes = declared_scopes if verdict is AuditVerdict.PASS else ()
    if verdict is not AuditVerdict.PASS and declared_scopes:
        reasons.append("non_pass_scope_declarations_ignored")
    evidence = VerifiedAuditEvidence(
        schema_version=VERIFIED_AUDIT_EVIDENCE_SCHEMA,
        subsystem="labeling",
        audit_kind=common["audit_kind"],
        verdict=verdict,
        audit_report_path=str(report_path.resolve()),
        audit_report_sha256=report_sha256,
        artifact_hash_manifest_path=str(manifest_path.resolve()),
        artifact_hash_manifest_sha256=manifest_sha256,
        bound_commit=bound_commit,
        bound_code_identity_sha256=bound_code,
        bound_contract_identity_sha256=bound_contract,
        bound_data_identity_sha256=bound_data,
        bound_component_identities=bound_components,
        auditor_identity=common["auditor_identity"],
        created_at_utc=common["created_at_utc"],
        applicability_status=status,
        applicability_reasons=tuple(reasons),
        authorized_scopes=scopes,
    )
    return AuditVerification(
        subsystem="labeling",
        applicability_status=status,
        reasons=tuple(reasons),
        artifact_status=ArtifactVerificationStatus.VERIFIED,
        current_identity=current_identity,
        evidence=evidence,
        report_verdict=verdict,
        bound_commit=bound_commit,
    )


def _component_identities(
    root: Path,
    semantic_nodes: Mapping[str, str],
) -> dict[str, str]:
    components: dict[str, str] = {}
    for name, selectors in _COMPONENT_SELECTORS.items():
        missing = sorted(set(selectors) - semantic_nodes.keys())
        if missing:
            raise IdentityComputationError(f"component_nodes_missing:{name}:{','.join(missing)}")
        components[name] = _canonical_sha256({selector: semantic_nodes[selector] for selector in selectors})
    file_components = {
        "product_capability_adapter": (
            "src/spritelab/product_core/backend_contracts.py",
            "src/spritelab/product_features/dataset/certification.py",
            "src/spritelab/product_features/dataset/semantics.py",
        ),
        "product_status_projection": (
            "src/spritelab/product_features/dataset/plugin.py",
            "src/spritelab/v3/model.py",
            "src/spritelab/v3/status.py",
        ),
        "review_routing": (
            "src/spritelab/product_features/dataset/review.py",
            "src/spritelab/product_features/dataset/web.py",
        ),
        "applicability_verifier": ("src/spritelab/product_core/audit_evidence.py",),
        "action_time_revalidation": (
            "src/spritelab/product_features/dataset/certification.py",
            "src/spritelab/product_features/dataset/review.py",
            "src/spritelab/product_features/dataset/semantics.py",
            "src/spritelab/v3/orchestration.py",
        ),
    }
    for name, paths in file_components.items():
        components[name] = _canonical_sha256(
            {relative: _bound_file_sha256(root / relative, relative) for relative in paths}
        )
    return components


def _bound_file_sha256(path: Path, relative: str) -> str:
    """Hash only the labeling slice of shared mixed-subsystem modules."""

    if relative == "src/spritelab/v3/orchestration.py":
        selected = _selected_ast_nodes(path, {"dataset_build"})
        if "dataset_build" not in selected:
            raise IdentityComputationError("component_nodes_missing:v3_orchestration:dataset_build")
        return _canonical_sha256(selected)
    if relative == "src/spritelab/v3/status.py":
        return _canonical_sha256(_v3_labeling_status_nodes(path))
    # Git may materialize the same source blob with CRLF on Windows and LF on
    # other platforms. Line-ending conversion is not a semantic code change.
    return _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _v3_labeling_status_nodes(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise IdentityComputationError(f"v3_status_parse_failed:{exc}") from exc
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    helper = functions.get("_labeling_audit_status")
    builder = functions.get("build_project_state")
    if helper is None or builder is None:
        raise IdentityComputationError("component_nodes_missing:v3_status:labeling_status")
    selected: list[str] = []
    in_labeling_block = False
    for statement in builder.body:
        assigned = _assigned_names(statement)
        if assigned & {"labeling_verification", "labeling_audit_status"}:
            selected.append(ast.dump(statement, annotate_fields=True, include_attributes=False))
        if "label_stopped" in assigned:
            in_labeling_block = True
        if "training_audit" in assigned:
            in_labeling_block = False
            break
        if in_labeling_block:
            selected.append(ast.dump(statement, annotate_fields=True, include_attributes=False))
    if not selected:
        raise IdentityComputationError("component_nodes_missing:v3_status:labeling_block")
    return {
        "_labeling_audit_status": ast.dump(helper, annotate_fields=True, include_attributes=False),
        "build_project_state_labeling_block": selected,
    }


def _assigned_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    names: set[str] = set()
    for target in targets:
        names.update(node.id for node in ast.walk(target) if isinstance(node, ast.Name))
    return names


def _selected_ast_nodes(path: Path, wanted: set[str]) -> dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise IdentityComputationError(f"semantic_contract_parse_failed:{exc}") from exc
    selected: dict[str, str] = {}
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers = [target.id for target in targets if isinstance(target, ast.Name)]
            name = identifiers[0] if len(identifiers) == 1 else None
        if name in wanted:
            selected[name] = ast.dump(node, annotate_fields=True, include_attributes=False)
    return selected


def _verify_report_hash(root: Path, report_path: Path, manifest: Mapping[str, Any], actual: str) -> str | None:
    if manifest.get("schema_version") != AUDIT_HASH_MANIFEST_SCHEMA:
        return "artifact_hash_manifest_schema_invalid"
    artifacts = manifest.get("artifacts")
    entries: list[tuple[str, str]] = []
    if isinstance(artifacts, Mapping):
        entries = [(str(path), str(digest)) for path, digest in artifacts.items()]
    elif isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        for item in artifacts:
            if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
                return "artifact_hash_manifest_entry_invalid"
            entries.append((str(item["path"]), str(item["sha256"])))
    else:
        return "artifact_hash_manifest_entries_missing"
    report_resolved = report_path.resolve()
    matched: list[str] = []
    for raw_path, digest in entries:
        candidate = Path(raw_path)
        candidates = (
            [candidate.resolve()]
            if candidate.is_absolute()
            else [
                (root / candidate).resolve(),
                (report_path.parent / candidate).resolve(),
            ]
        )
        if report_resolved in candidates:
            matched.append(digest)
    if len(matched) != 1 or not _HEX_64.fullmatch(matched[0]):
        return "audit_report_hash_entry_missing_or_ambiguous"
    return None if matched[0] == actual else "audit_report_hash_mismatch"


def _validated_common_report(report: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    schema = report.get("schema_version")
    if not isinstance(schema, str) or not schema.strip():
        return None, "report_schema_version_missing"
    subsystem = report.get("subsystem")
    if subsystem != "labeling":
        return None, "report_subsystem_mismatch"
    audit_kind = report.get("audit_kind")
    if not isinstance(audit_kind, str) or not audit_kind.strip() or "remediation" in audit_kind.casefold():
        return None, "audit_kind_invalid"
    try:
        verdict = AuditVerdict(str(report.get("verdict", "")))
    except ValueError:
        return None, "audit_verdict_invalid"
    bound_commit = report.get("bound_commit")
    if not isinstance(bound_commit, str) or not _HEX_40.fullmatch(bound_commit):
        return None, "bound_commit_invalid"
    auditor = report.get("auditor_identity") or report.get("audit_run_identity")
    if not isinstance(auditor, str) or not auditor.strip():
        return None, "auditor_identity_missing"
    created = report.get("created_at_utc")
    try:
        _require_utc_timestamp(created)
    except ValueError:
        return None, "created_at_utc_invalid"
    role = report.get("evidence_role")
    independent = role == "independent_certification" or report.get("independent_audit") is True
    if not independent:
        return None, "independent_certification_role_missing"
    scopes = report.get("authorized_scopes", [])
    if not isinstance(scopes, list) or not all(isinstance(item, str) for item in scopes):
        return None, "authorized_scopes_invalid"
    if len(scopes) != len(set(scopes)) or set(scopes) - KNOWN_AUTHORIZATION_SCOPES:
        return None, "authorized_scopes_invalid"
    if schema == AUDIT_REPORT_SCHEMA and not isinstance(report.get("bound_code_identity_sha256"), str):
        return None, "versioned_report_identity_missing"
    return (
        {
            "schema_version": schema,
            "audit_kind": audit_kind,
            "verdict": verdict,
            "bound_commit": bound_commit,
            "auditor_identity": auditor,
            "created_at_utc": created,
            "authorized_scopes": tuple(scopes),
        },
        None,
    )


def _declared_verdict(report: Mapping[str, Any]) -> AuditVerdict | None:
    try:
        return AuditVerdict(str(report.get("verdict", "")))
    except ValueError:
        return None


def _validate_report_identities(report: Mapping[str, Any], current: LabelingAuditIdentity) -> str | None:
    for key in (
        "bound_code_identity_sha256",
        "bound_contract_identity_sha256",
        "bound_data_identity_sha256",
    ):
        value = report.get(key)
        if not isinstance(value, str) or not _HEX_64.fullmatch(value):
            return f"{key}_invalid"
    components = report.get("bound_component_identities")
    if not isinstance(components, Mapping):
        return "bound_component_identities_invalid"
    expected_names = {name for name, _digest in current.component_identities}
    if set(components) != expected_names:
        return "bound_component_identity_set_incomplete"
    if any(not isinstance(value, str) or not _HEX_64.fullmatch(value) for value in components.values()):
        return "bound_component_identity_invalid"
    return None


def _unverified(
    current: LabelingAuditIdentity,
    artifact_status: ArtifactVerificationStatus,
    reason: str,
    *,
    verdict: AuditVerdict | None = None,
    bound_commit: str | None = None,
) -> AuditVerification:
    return AuditVerification(
        subsystem="labeling",
        applicability_status=ApplicabilityStatus.NOT_COMPARABLE,
        reasons=(reason,),
        artifact_status=artifact_status,
        current_identity=current,
        report_verdict=verdict,
        bound_commit=bound_commit,
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase sha256 identity")


def _require_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("created_at_utc must be UTC")

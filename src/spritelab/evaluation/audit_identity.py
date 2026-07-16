"""Versioned semantic source identity for memorization audit applicability.

The inventory is intentionally explicit.  It binds production behavior that can
change the machine evidence, signed review authority, active evaluation inputs,
or promotion authorization without making documentation and unrelated product
surfaces part of the audit freshness boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MEMORIZATION_AUDIT_SUBSYSTEM = "memorization"
MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION = "sprite_lab_memorization_audit_code_identity_v4"
MEMORIZATION_AUDIT_SOURCE_CANONICALIZATION = "source_bytes_crlf_to_lf_v1"
MEMORIZATION_AUDIT_INVENTORY_ORDER = "relative_posix_path_ascending_v1"

# (relative path, semantic role, reason the file can affect the decision)
MEMORIZATION_AUDIT_SEMANTIC_FILES: tuple[tuple[str, str, str], ...] = (
    (
        "src/spritelab/__main__.py",
        "package command dispatcher",
        "Selects the normal v3, product, evaluation, review, and legacy command paths.",
    ),
    (
        "src/spritelab/dev_features/audits.py",
        "developer audit applicability projection",
        "Projects memorization freshness and current certification on developer surfaces.",
    ),
    (
        "src/spritelab/evaluation/audit_identity.py",
        "audit identity contract and inventory",
        "Defines, hashes, and validates the complete semantic freshness boundary.",
    ),
    (
        "src/spritelab/evaluation/candidate_bundle.py",
        "candidate bundle writer and strict source verifier",
        "Constructs candidate evidence and validates machine, generated, training, and policy bindings.",
    ),
    (
        "src/spritelab/evaluation/cli.py",
        "evaluation review and promotion action adapter",
        "Projects normal CLI inputs into evaluation, signed review authoring, and promotion recomputation.",
    ),
    (
        "src/spritelab/evaluation/conditional.py",
        "conditional comparison scorer",
        "Computes per-sample adherence values and the conditional-not-worse promotion gate.",
    ),
    (
        "src/spritelab/evaluation/memorization.py",
        "detector policy and evidence classifier",
        "Controls detector identity, diagnostics, translations, thresholds, and evidence classes.",
    ),
    (
        "src/spritelab/evaluation/memorization_review.py",
        "signed review loading, authoring, and replay",
        "Controls bound review identities, append-only events, chain validation, and authoritative replay.",
    ),
    (
        "src/spritelab/evaluation/metrics.py",
        "comparison hashes, pixel distances, alpha metrics, and IoU",
        "Implements decoded RGBA hashes, normalized alpha, pixel/perceptual distance, and mask IoU.",
    ),
    (
        "src/spritelab/evaluation/promotion_decision.py",
        "promotion evidence recomputation and source verification",
        "Revalidates all bound artifacts, candidate classes, review events, and promotion blockers.",
    ),
    (
        "src/spritelab/evaluation/suite.py",
        "machine candidate-set and bundle orchestration",
        "Selects candidates, projects checkpoint/dataset/view/benchmark identities, and writes machine evidence.",
    ),
    (
        "src/spritelab/harvest/label_v4/risk.py",
        "evaluation quality-risk vocabulary and scoring",
        "Defines semantic fields, calibrated bands, and risk projections consumed by evaluation evidence.",
    ),
    (
        "src/spritelab/harvest/label_v4/training_quality.py",
        "evaluation uncertainty and training-quality projection",
        "Computes quality strata, uncertainty summaries, and correlations persisted in machine reports.",
    ),
    (
        "src/spritelab/product_core/__init__.py",
        "product contract facade",
        "Selects the strict JSON, ProductEvent, API, CLI, and result implementations used by bound product paths.",
    ),
    (
        "src/spritelab/product_core/api.py",
        "controlled product API error projection",
        "Controls endpoint classification and the recoverable error envelope for invalid evaluation inputs.",
    ),
    (
        "src/spritelab/product_core/audit_evidence.py",
        "independent audit evidence and authorization contract",
        "Validates audit applicability, subsystem identity, authorized scopes, and evidence integrity consumed by product readiness gates.",
    ),
    (
        "src/spritelab/product_core/backend_contracts.py",
        "product promotion authorization projection",
        "Controls whether backend promotion evidence can become an authoritative product authorization.",
    ),
    (
        "src/spritelab/product_core/cli.py",
        "product evaluation CLI registry",
        "Controls whether the certified evaluation handler owns and receives the normal product eval action.",
    ),
    (
        "src/spritelab/product_core/contracts.py",
        "finite ProductEvent construction and deserialization contract",
        "Validates every persisted and replayed ProductEvent field consumed as evaluation evidence.",
    ),
    (
        "src/spritelab/product_core/events.py",
        "strict event JSON parsing, validation, and serialization",
        "Controls strict benchmark/report parsing, non-finite rejection, and durable event byte semantics.",
    ),
    (
        "src/spritelab/product_core/plugins.py",
        "product plugin registry and route composition",
        "Controls which evaluation, review, training, CLI, and web adapters are active in the normal product.",
    ),
    (
        "src/spritelab/product_core/web.py",
        "product web runtime and authorization contract",
        "Controls plugin mounting, server settings, and request protections around review authoring.",
    ),
    (
        "src/spritelab/product_features/dataset/certification.py",
        "dataset and labeling authorization adapter",
        "Recomputes labeling applicability and controls conditioned-view, dataset-freeze, and downstream checkpoint eligibility.",
    ),
    (
        "src/spritelab/product_features/dataset/plugin.py",
        "dataset and memorization-review plugin adapter",
        "Registers the normal product route that discovers evidence and authors bound review decisions.",
    ),
    (
        "src/spritelab/product_features/dataset/static/review.js",
        "memorization review browser action adapter",
        "Offers and submits controlled signed-review outcomes from actionable product rows.",
    ),
    (
        "src/spritelab/product_features/dataset/templates/review_entry.html",
        "memorization review action template",
        "Controls whether authoritative clearing actions and outcome values are exposed to reviewers.",
    ),
    (
        "src/spritelab/product_features/dataset/web.py",
        "strict product candidate discovery and review authoring adapter",
        "Discovers current evidence, recomputes action availability, and submits signed review events.",
    ),
    (
        "src/spritelab/product_features/evaluation/checkpoints.py",
        "active checkpoint, dataset/view, completion, verification, and weight projection",
        "Determines eligible checkpoints, active dataset/view compatibility, completion, and live/EMA identity.",
    ),
    (
        "src/spritelab/product_features/evaluation/dashboard.py",
        "evaluation comparison and metric-definition projection",
        "Computes finite dashboard metrics, definition identities, category deltas, and paired comparisons.",
    ),
    (
        "src/spritelab/product_features/evaluation/memorization_display.py",
        "product review and audit authority display",
        "Replays strict evidence and projects current certification and promotion-integrity wording.",
    ),
    (
        "src/spritelab/product_features/evaluation/models.py",
        "checkpoint eligibility and active variant model",
        "Controls eligible catalog membership, default checkpoint selection, and live/EMA lookup.",
    ),
    (
        "src/spritelab/product_features/evaluation/playground.py",
        "exploratory generation non-promotion boundary",
        "Binds Playground outputs to explicit non-benchmark, non-promotional state and rejects altered authority flags.",
    ),
    (
        "src/spritelab/product_features/evaluation/plugin.py",
        "product evaluation status and CLI projection",
        "Projects checkpoint, benchmark, audit integrity, and evaluation actions into product capabilities.",
    ),
    (
        "src/spritelab/product_features/evaluation/service.py",
        "evaluation input selection and durable authority projection",
        "Recomputes checkpoint, dataset, benchmark, candidate evidence, and audit integrity for product runs.",
    ),
    (
        "src/spritelab/product_features/evaluation/static/evaluation.js",
        "product evaluation selection action adapter",
        "Submits the selected checkpoint and live/EMA variant and exposes memorization review actions.",
    ),
    (
        "src/spritelab/product_features/evaluation/templates/evaluation.html",
        "product evaluation action and authority template",
        "Exposes checkpoint/weight selection, review navigation, and promotion authority wording.",
    ),
    (
        "src/spritelab/product_features/evaluation/templates/evaluation_standalone.html",
        "standalone product authority template",
        "Provides the fallback evaluation action and promotion-integrity wording surface.",
    ),
    (
        "src/spritelab/product_features/evaluation/web.py",
        "product evaluation HTTP action adapter and scope separation",
        "Separates exploratory requests from benchmark evaluation and submits server-validated active inputs.",
    ),
    (
        "src/spritelab/product_features/training/__init__.py",
        "training plugin facade",
        "Selects the normal training plugin, service, and backend adapters that produce checkpoint provenance.",
    ),
    (
        "src/spritelab/product_features/training/config.py",
        "training compute configuration projection",
        "Selects and validates the compute configuration used by checkpoint-producing adapters.",
    ),
    (
        "src/spritelab/product_features/training/dashboard.py",
        "training checkpoint event projection",
        "Converts validated training events into checkpoint path, hash, verification, and safe-resume state.",
    ),
    (
        "src/spritelab/product_features/training/models.py",
        "training plan and readiness model",
        "Controls the resolved plan and gate values consumed before checkpoint-producing execution.",
    ),
    (
        "src/spritelab/product_features/training/plans.py",
        "product training campaign resolver",
        "Projects active dataset, campaign, audit, and compute inputs into the validated launch plan.",
    ),
    (
        "src/spritelab/product_features/training/service.py",
        "training receipt and checkpoint provenance projection",
        "Persists launch-bound dataset/view identities into durable training state consumed by checkpoint selection.",
    ),
    (
        "src/spritelab/product_features/training/web.py",
        "product training action adapter",
        "Drives start, refresh, pause, and resume paths that create durable checkpoint provenance.",
    ),
    (
        "src/spritelab/product_runtime.py",
        "normal product plugin composition",
        "Selects the active dataset-review, training, and evaluation plugins for CLI and web execution.",
    ),
    (
        "src/spritelab/product_web/app.py",
        "product web application composition",
        "Mounts review and evaluation routers and enforces the application request boundary.",
    ),
    (
        "src/spritelab/product_web/cli.py",
        "product web and CLI runtime composition",
        "Builds the normal application and dispatches the registered product command surfaces.",
    ),
    (
        "src/spritelab/product_web/events.py",
        "durable evaluation event and state repository",
        "Persists and reconstructs evaluation state and artifact identities consumed by product authority projections.",
    ),
    (
        "src/spritelab/remote_compute/__init__.py",
        "compute backend facade",
        "Selects the backend contracts used to produce and retain checkpoint evidence.",
    ),
    (
        "src/spritelab/remote_compute/contracts.py",
        "compute request and event contracts",
        "Defines validated launch receipts, backend requests, and checkpoint-producing event identities.",
    ),
    (
        "src/spritelab/remote_compute/hosted.py",
        "hosted plugin compute delegation",
        "Controls hosted checkpoint handoffs and the adapter identity projected into durable training state.",
    ),
    (
        "src/spritelab/remote_compute/local.py",
        "local compute event adapter",
        "Parses local training events and checkpoint evidence into the product event contract.",
    ),
    (
        "src/spritelab/remote_compute/runpod.py",
        "RunPod compute safety scaffold",
        "Controls whether the RunPod path can produce any checkpoint or remain unavailable.",
    ),
    (
        "src/spritelab/remote_compute/ssh.py",
        "SSH compute event adapter",
        "Parses remote training events and checkpoint evidence into the product event contract.",
    ),
    (
        "src/spritelab/remote_compute/utils.py",
        "compute artifact and remote identity helpers",
        "Computes hashes and remote identities used by local and SSH checkpoint verification.",
    ),
    (
        "src/spritelab/training/campaign.py",
        "training campaign dataset/view identity construction",
        "Constructs the dataset and view identities bound into validated training campaigns and checkpoints.",
    ),
    (
        "src/spritelab/training/launch.py",
        "validated training launch identity receipt",
        "Binds campaign, dataset, view, run, and checkpoint provenance before any training handoff.",
    ),
    (
        "src/spritelab/v3/cli.py",
        "v3 product command registry and dispatch",
        "Installs and dispatches the normal evaluation and review actions through the product plugin registry.",
    ),
    (
        "src/spritelab/v3/config.py",
        "checkpoint, dataset/view, benchmark, audit, and promotion configuration projection",
        "Defines and resolves the active configuration fields consumed by evaluation and promotion status.",
    ),
    (
        "src/spritelab/v3/model.py",
        "project audit and promotion-stage model",
        "Defines audit states, production authorization fields, stage lookup, and status serialization.",
    ),
    (
        "src/spritelab/v3/orchestration.py",
        "v3 evaluation and promotion action projection",
        "Carries active dataset/view identities into evaluation plans and blocks actions when identity gates fail.",
    ),
    (
        "src/spritelab/v3/report.py",
        "offline product authority report projection",
        "Persists and renders stage blockers, audit state, evidence identities, and production-authorization decisions.",
    ),
    (
        "src/spritelab/v3/run_state.py",
        "v3 durable command-state writer",
        "Persists dataset/view and backend identities projected by v3 training and evaluation orchestration.",
    ),
    (
        "src/spritelab/v3/status.py",
        "shared memorization applicability and promotion status verifier",
        "Recomputes audit freshness and blocks stale certification and promotion authorization.",
    ),
)

MEMORIZATION_AUDIT_BOUND_FILES = tuple(record[0] for record in MEMORIZATION_AUDIT_SEMANTIC_FILES)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = frozenset(
    {
        "contract_version",
        "subsystem",
        "source_canonicalization",
        "inventory_order",
        "bound_files",
        "code_identity_sha256",
    }
)
_FILE_KEYS = frozenset({"path", "semantic_role", "decision_effect", "sha256"})


class MemorizationAuditIdentityError(RuntimeError):
    """The current semantic code identity could not be computed safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_source_sha256(path: Path) -> str:
    """Hash source bytes with the repository's cross-platform CRLF normalization."""

    try:
        source = path.read_bytes()
    except OSError as exc:
        raise MemorizationAuditIdentityError(f"bound_source_unreadable:{path.as_posix()}:{exc}") from exc
    return hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()


def _validate_contract_inventory() -> None:
    paths = list(MEMORIZATION_AUDIT_BOUND_FILES)
    if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise MemorizationAuditIdentityError("semantic_inventory_not_unique_sorted")
    for path, role, reason in MEMORIZATION_AUDIT_SEMANTIC_FILES:
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise MemorizationAuditIdentityError(f"semantic_inventory_path_invalid:{path}")
        if not role.strip() or not reason.strip():
            raise MemorizationAuditIdentityError(f"semantic_inventory_metadata_missing:{path}")


def memorization_audit_code_identity(root: Path) -> dict[str, Any]:
    """Return the deterministic, complete v4 semantic identity for ``root``."""

    _validate_contract_inventory()
    resolved_root = root.resolve()
    files: list[dict[str, str]] = []
    for relative, role, reason in MEMORIZATION_AUDIT_SEMANTIC_FILES:
        path = resolved_root / relative
        if not path.is_file():
            raise MemorizationAuditIdentityError(f"bound_source_missing:{relative}")
        files.append(
            {
                "path": relative,
                "semantic_role": role,
                "decision_effect": reason,
                "sha256": _canonical_source_sha256(path),
            }
        )
    identity: dict[str, Any] = {
        "contract_version": MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION,
        "subsystem": MEMORIZATION_AUDIT_SUBSYSTEM,
        "source_canonicalization": MEMORIZATION_AUDIT_SOURCE_CANONICALIZATION,
        "inventory_order": MEMORIZATION_AUDIT_INVENTORY_ORDER,
        "bound_files": files,
    }
    identity["code_identity_sha256"] = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return identity


def recorded_identity_errors(identity: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate a recorded v4 identity without silently repairing or reordering it."""

    errors: list[str] = []
    if set(identity) != _IDENTITY_KEYS:
        errors.append("recorded_identity_fields_invalid")
    if identity.get("contract_version") != MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION:
        errors.append("recorded_identity_contract_not_v4")
    if identity.get("subsystem") != MEMORIZATION_AUDIT_SUBSYSTEM:
        errors.append("recorded_identity_subsystem_mismatch")
    if identity.get("source_canonicalization") != MEMORIZATION_AUDIT_SOURCE_CANONICALIZATION:
        errors.append("recorded_identity_canonicalization_invalid")
    if identity.get("inventory_order") != MEMORIZATION_AUDIT_INVENTORY_ORDER:
        errors.append("recorded_identity_order_contract_invalid")
    raw_files = identity.get("bound_files")
    if not isinstance(raw_files, list):
        return tuple(dict.fromkeys((*errors, "recorded_file_list_malformed")))
    paths: list[str] = []
    expected_metadata = {path: (role, reason) for path, role, reason in MEMORIZATION_AUDIT_SEMANTIC_FILES}
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != _FILE_KEYS:
            errors.append("recorded_file_entry_malformed")
            continue
        path = item.get("path")
        if not isinstance(path, str):
            errors.append("recorded_file_path_malformed")
            continue
        paths.append(path)
        expected = expected_metadata.get(path)
        if expected is None or (item.get("semantic_role"), item.get("decision_effect")) != expected:
            errors.append("recorded_file_semantics_mismatch")
        if not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(str(item.get("sha256"))):
            errors.append("recorded_file_hash_malformed")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append("recorded_file_order_or_uniqueness_invalid")
    if tuple(paths) != MEMORIZATION_AUDIT_BOUND_FILES:
        errors.append("recorded_file_inventory_incomplete")
    recorded_hash = identity.get("code_identity_sha256")
    payload = {key: value for key, value in identity.items() if key != "code_identity_sha256"}
    try:
        expected_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    except (TypeError, ValueError):
        errors.append("recorded_identity_json_malformed")
        return tuple(dict.fromkeys(errors))
    if not isinstance(recorded_hash, str) or not _SHA256.fullmatch(recorded_hash) or recorded_hash != expected_hash:
        errors.append("recorded_code_identity_hash_invalid")
    return tuple(dict.fromkeys(errors))

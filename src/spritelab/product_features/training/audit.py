"""Server-managed independent training-infrastructure audit execution.

The public service in this module deliberately accepts no caller-supplied gate
verdicts.  It derives the eighteen infrastructure verdicts from the exact
prospective conditioned activation, the tracked production-code inventory, a
server-produced CPU/CUDA smoke bundle, and fixed curated/full test commands.
The report and hash inventory are useful only when the final immutable
execution receipt is also present and still applicable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from spritelab.product_core import ProjectContext
from spritelab.product_features.training.activation import (
    MANDATORY_TRAINING_AUDIT_GATES,
    TRAINING_AUDIT_HASHES_SCHEMA,
    TRAINING_AUDIT_REPORT_SCHEMA,
    ConditionedActivationError,
    ConditionedTrainingActivation,
    load_conditioned_training_activation,
)
from spritelab.product_features.training.models import TrainingProfile
from spritelab.training.campaign import (
    CampaignValidationError,
    stable_hash,
    training_code_identity_source_paths,
)
from spritelab.training.smoke_bundle import (
    SmokeBundleError,
    artifact_bundle_directory,
    load_plan,
    run_bundle_directory,
    verify_complete_bundle,
)
from spritelab.utils.pinned_executable import PinnedExecutableError, pinned_git_ls_files
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus

TRAINING_AUDIT_RECEIPT_SCHEMA: Final = "spritelab.training.infrastructure-audit-receipt.v1"
TRAINING_AUDIT_ACTION_RECORD_SCHEMA: Final = "spritelab.training.infrastructure-audit-action-record.v1"
TRAINING_AUDIT_OPERATION_SCHEMA: Final = "spritelab.training.infrastructure-audit-operation.v1"
TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA: Final = "spritelab.training.infrastructure-audit-runner-inventory.v1"
TRAINING_AUDIT_TEST_RESULT_SCHEMA: Final = "spritelab.training.infrastructure-audit-test-result.v1"
TRAINING_AUDIT_TEST_HARNESS_SCHEMA: Final = "spritelab.training.infrastructure-audit-test-harness.v1"
TRAINING_AUDIT_LAUNCH_AUTHORIZATION_SCHEMA: Final = "spritelab.training.launch-authorization-evidence.v1"
TRAINING_AUDITOR_ID: Final = "spritelab.training.infrastructure-auditor.server.v1"

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_NONCE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,79}$")
_REFERENCE_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_MAPPING_BYTES: Final = 128 * 1024 * 1024
_MAX_INVENTORY_FILE_BYTES: Final = 2 * 1024**3
_AUDIT_OUTPUT_ROOT: Final = PurePosixPath("artifacts/training")
_AUDIT_ATTEMPT_ROOT: Final = PurePosixPath("artifacts/training/audits")
_TEST_ENVIRONMENT_ALLOWLIST: Final = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
_PYTEST_BOOTSTRAP: Final = (
    "import sys;"
    "n=int(sys.argv.pop(1));"
    "p=[sys.argv.pop(1) for _ in range(n)];"
    "sys.path[:0]=p;"
    "import pytest;"
    "r=sys.argv.pop(1);"
    "sys.path[:0]=[r+'/tests',r+'/src'];"
    "raise SystemExit(pytest.main(sys.argv[1:]))"
)

_RUNNER_SOURCE_PATHS: Final = (
    "src/spritelab/product_features/training/audit.py",
    "src/spritelab/product_features/training/activation.py",
    "src/spritelab/training/campaign.py",
    "src/spritelab/training/smoke_bundle.py",
)

_CURATED_TEST_TARGETS: Final = (
    "tests/test_product_training_activation.py",
    "tests/test_training_campaign.py",
    "tests/test_training_conditioned_campaign_launch.py",
    "tests/test_training_resume_binding.py",
    "tests/test_training_event_history_origin.py",
    "tests/test_training_migration_resume.py",
    "tests/test_training_plan_path_confinement.py",
    "tests/test_training_product_integration_remediation.py",
    "tests/test_training_web_readiness_hardening.py",
    "tests/test_safe_filesystem.py",
    "tests/test_smoke_artifact_schema_strictness.py",
    "tests/test_smoke_execution_state_cas.py",
    "tests/test_smoke_worker_protocol_strictness.py",
)

_TEST_PLAN: Final = (
    {
        "name": "curated",
        "targets": list(_CURATED_TEST_TARGETS),
        "timeout_seconds": 3_600,
    },
    {
        "name": "full",
        "targets": ["tests"],
        "timeout_seconds": 10_800,
    },
)
_TEST_PLAN_IDENTITY: Final = stable_hash(
    {
        "schema_version": "spritelab.training.infrastructure-audit-test-plan.v1",
        "commands": _TEST_PLAN,
        "python_options": ["-I", "-B", "-S", "-X", "pycache_prefix=<managed-basetemp>/pycache", "-c"],
        "bootstrap_sha256": hashlib.sha256(_PYTEST_BOOTSTRAP.encode("utf-8")).hexdigest(),
        "pytest_options": [
            "-c",
            "pyproject.toml",
            "--rootdir=.",
            "--confcutdir=.",
            "--import-mode=importlib",
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        "environment_policy": {
            "allowlist": sorted(_TEST_ENVIRONMENT_ALLOWLIST),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SPRITELAB_PROGRESS": "0",
        },
    }
)

_REPORT_KEYS: Final = frozenset(
    {
        "schema_version",
        "verdict",
        "independent",
        "generated_by",
        "operation_identity",
        "runner",
        "test_harness",
        "bindings",
        "gates",
        "gate_evidence",
        "test_results",
        "paths_exposed",
        "report_identity",
    }
)
_HASHES_KEYS: Final = frozenset(
    {
        "schema_version",
        "operation_identity",
        "audit_report_sha256",
        "audit_report_byte_count",
        "audit_receipt_path",
        "runner_inventory_sha256",
        "test_harness_inventory_sha256",
        "bindings",
        "files",
        "artifacts",
        "smoke_artifacts",
        "inventory_identity",
    }
)
_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "operation",
        "operation_identity",
        "runner",
        "test_harness",
        "bindings",
        "verdict",
        "gates_identity",
        "test_results",
        "report",
        "hash_inventory",
        "artifact_inventory_identity",
        "smoke_artifact_inventory_identity",
        "completed_at",
        "paths_exposed",
        "receipt_identity",
    }
)
_ACTION_RECORD_KEYS: Final = frozenset(
    {
        "schema_version",
        "source_job_id",
        "operation_identity",
        "prospective_configuration_identity_sha256",
        "base_config_sha256",
        "verdict",
        "report_sha256",
        "hash_inventory_sha256",
        "receipt_sha256",
        "receipt_identity",
        "config_unchanged",
        "configuration_activated",
        "training_started",
        "paths_exposed",
        "record_identity",
    }
)


class TrainingAuditExecutionError(ValueError):
    """A server-managed infrastructure audit could not be executed safely."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass(frozen=True)
class TrainingAuditExecution:
    """Immutable paths and verdict returned by one completed audit action."""

    operation_identity: str
    verdict: AuditStatus
    report_path: Path
    hashes_path: Path
    receipt_path: Path


@dataclass
class _HeldAuditFile:
    path: Path
    parent: AnchoredDirectory
    descriptor: int
    identity: OwnedFileIdentity
    metadata: os.stat_result
    payload: bytes

    def verify_unchanged(self) -> None:
        self.parent.verify()
        held = os.fstat(self.descriptor)
        visible = self.parent.lstat(self.path.name)
        _require_safe_bound_regular(
            held,
            boundary_device=self.parent.directory_metadata().st_dev,
            max_bytes=_MAX_MAPPING_BYTES,
        )
        if not _same_exact_file(self.metadata, held) or not _same_exact_file(held, visible):
            raise UnsafeFilesystemOperation("held audit evidence changed identity")
        current = _read_held_audit_descriptor(self.descriptor, maximum_bytes=_MAX_MAPPING_BYTES)
        after = os.fstat(self.descriptor)
        final = self.parent.lstat(self.path.name)
        if current != self.payload or not _same_exact_file(held, after) or not _same_exact_file(after, final):
            raise UnsafeFilesystemOperation("held audit evidence changed bytes")
        self.parent.verify()


@dataclass
class TrainingAuditExecutionSnapshot:
    """Coherent retained audit evidence for one launch-authorization action.

    The context returned by :func:`open_training_audit_execution_snapshot`
    owns this value.  Callers must keep that context open through their final
    authorization action and call :meth:`verify_unchanged` immediately before
    dispatch.  Using the snapshot after context exit fails closed.
    """

    status: AuditStatus
    operation_identity: str
    report: Mapping[str, Any]
    report_sha256: str
    hash_inventory_sha256: str
    receipt_sha256: str
    receipt_identity: str
    action_record_sha256: str
    launch_authorization_evidence_sha256: str
    _held_files: tuple[_HeldAuditFile, ...]
    _closed: bool = False

    def verify_unchanged(self) -> None:
        if self._closed:
            raise UnsafeFilesystemOperation("training audit snapshot is closed")
        for held in self._held_files:
            held.verify_unchanged()


def run_training_infrastructure_audit(
    context: ProjectContext | ProjectConfig,
    *,
    profile: TrainingProfile | str = TrainingProfile.RECOMMENDED,
    operation_nonce: str,
    smoke_id: str | None,
    source_job_id: str | None = None,
) -> TrainingAuditExecution:
    """Execute and immutably publish one server-managed infrastructure audit.

    This is an explicit, potentially long-running service action.  It never
    changes project configuration and never launches a production campaign.
    The configured report and hash paths, plus the derived receipt path, must
    all be absent; publication refuses replacement.
    """

    if not isinstance(operation_nonce, str) or _OPERATION_NONCE.fullmatch(operation_nonce) is None:
        raise TrainingAuditExecutionError(
            "audit_operation_nonce",
            "A unique 8-80 character audit operation nonce is required.",
        )
    if smoke_id is not None and (
        not isinstance(smoke_id, str)
        or not smoke_id
        or smoke_id != smoke_id.strip()
        or len(smoke_id) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", smoke_id) is None
    ):
        raise TrainingAuditExecutionError("audit_smoke_id", "The selected smoke-bundle identity is invalid.")
    if source_job_id is not None and (
        not isinstance(source_job_id, str)
        or not source_job_id
        or source_job_id != source_job_id.strip()
        or len(source_job_id) > 160
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_job_id) is None
    ):
        raise TrainingAuditExecutionError("audit_source_job", "The selected conditioned job identity is invalid.")

    config = _config(context)
    selected_profile = profile if isinstance(profile, TrainingProfile) else TrainingProfile(str(profile))
    report_path = _configured_target(config, "audit_report")
    hashes_path = _configured_target(config, "audit_hashes")
    receipt_path = training_audit_receipt_path(config)
    targets = (report_path, hashes_path, receipt_path)
    if len(set(targets)) != len(targets) or len({path.parent for path in targets}) != 1:
        raise TrainingAuditExecutionError(
            "audit_output_collision",
            "The audit report, hashes, and receipt must be distinct files in one managed attempt directory.",
        )
    for parent in sorted({path.parent for path in targets}, key=lambda value: (len(value.parts), str(value))):
        _ensure_project_directory(config.root, parent)
    _require_absent_targets(config.root, targets)

    try:
        activation = load_conditioned_training_activation(
            config,
            selected_profile,
            require_audit=False,
            require_activation_commit=False,
        )
    except (ConditionedActivationError, CampaignValidationError, OSError, ValueError) as exc:
        raise TrainingAuditExecutionError(
            "audit_activation_invalid",
            "The exact prospective conditioned activation is unavailable or invalid.",
        ) from exc

    runner = training_audit_runner_inventory(config.root)
    test_harness = training_audit_test_harness_inventory(config.root)
    bindings = _activation_bindings(activation)
    code_files = _training_code_inventory(config.root, activation)
    activation_artifacts = _activation_artifact_inventory(config.root, activation)
    operation = {
        "schema_version": TRAINING_AUDIT_OPERATION_SCHEMA,
        "operation_nonce": operation_nonce,
        "profile": selected_profile.value,
        "source_job_id": source_job_id,
        "smoke_id": smoke_id,
        "test_plan_identity": _TEST_PLAN_IDENTITY,
        "test_harness_inventory_sha256": test_harness["inventory_sha256"],
        "report_path": report_path.relative_to(config.root).as_posix(),
        "hashes_path": hashes_path.relative_to(config.root).as_posix(),
        "receipt_path": receipt_path.relative_to(config.root).as_posix(),
        "bindings": bindings,
        "runner_inventory_sha256": runner["inventory_sha256"],
    }
    operation_identity = stable_hash(operation)

    smoke_status, smoke_sources, smoke_artifacts = _verify_smoke_evidence(
        config.root,
        activation,
        smoke_id,
    )
    test_results = tuple(_execute_fixed_test_plan(config.root, operation_identity))
    test_statuses = {str(result["name"]): str(result["verdict"]) for result in test_results}
    curated_status = _audit_status(test_statuses.get("curated"))
    full_status = _audit_status(test_statuses.get("full"))
    tests_status = _combine(curated_status, full_status)

    validation_source = _artifact_source(activation, "validation_report")
    campaign_source = {
        "kind": "campaign_validation",
        "campaign_identity_sha256": bindings["campaign_identity_sha256"],
    }
    code_source = {
        "kind": "tracked_production_code_inventory",
        "inventory_sha256": stable_hash(code_files),
        "file_count": len(code_files),
    }
    test_sources = [
        {
            "kind": "server_test_execution",
            "name": result["name"],
            "result_identity": result["result_identity"],
        }
        for result in test_results
    ]

    gate_inputs: dict[str, tuple[AuditStatus, list[dict[str, Any]]]] = {
        "tracked_code_identity_inventory": (AuditStatus.PASS, [code_source]),
        "no_untracked_production_python": (AuditStatus.PASS, [code_source]),
        "dataset_view_freeze_campaign_vocabulary_identity": (
            AuditStatus.PASS,
            [_activation_source(activation), campaign_source],
        ),
        "dataset_and_training_manifest_qa": (AuditStatus.PASS, [validation_source]),
        "production_loader_coverage": (smoke_status, [validation_source, *smoke_sources]),
        "campaign_experiment_compatibility": (AuditStatus.PASS, [campaign_source]),
        "cpu_cuda_smoke_evidence": (smoke_status, smoke_sources),
        "cuda_driver_torch_device_compatibility": (smoke_status, smoke_sources),
        "determinism_environment_qualification": (smoke_status, smoke_sources),
        "launch_receipt_execution_contract_binding": (
            _combine(smoke_status, tests_status),
            [*smoke_sources, *test_sources],
        ),
        "backend_command_safety": (tests_status, test_sources),
        "idempotency_concurrency_refusal": (tests_status, test_sources),
        "output_root_resume_safety": (tests_status, test_sources),
        "event_history_migration_identity": (tests_status, test_sources),
        "publication_config_atomicity_restart": (tests_status, test_sources),
        "filesystem_containment_link_defenses": (tests_status, test_sources),
        "api_ui_privacy": (tests_status, test_sources),
        "curated_full_test_results": (tests_status, test_sources),
    }
    if set(gate_inputs) != set(MANDATORY_TRAINING_AUDIT_GATES):
        raise TrainingAuditExecutionError(
            "audit_gate_implementation", "The server audit implementation has an incomplete gate map."
        )

    gates: dict[str, str] = {}
    gate_evidence: dict[str, dict[str, Any]] = {}
    for gate in MANDATORY_TRAINING_AUDIT_GATES:
        status, sources = gate_inputs[gate]
        verdict = _verdict(status)
        evidence_payload = {
            "gate": gate,
            "verdict": verdict,
            "sources": sources,
        }
        gates[gate] = verdict
        gate_evidence[gate] = {
            **evidence_payload,
            "evidence_identity": stable_hash(evidence_payload),
        }
    if gates != _derived_gate_verdicts(smoke_status, test_results):
        raise TrainingAuditExecutionError(
            "audit_gate_implementation", "The server audit gate evidence diverged from its fixed inputs."
        )
    overall = _overall(gates)
    report_payload = {
        "schema_version": TRAINING_AUDIT_REPORT_SCHEMA,
        "verdict": overall.value,
        "independent": True,
        "generated_by": "server_managed_training_audit_service",
        "operation_identity": operation_identity,
        "runner": runner,
        "test_harness": test_harness,
        "bindings": bindings,
        "gates": gates,
        "gate_evidence": gate_evidence,
        "test_results": list(test_results),
        "paths_exposed": False,
    }
    report = {**report_payload, "report_identity": stable_hash(report_payload)}
    report_bytes = _canonical_bytes(report)

    hashes_payload = {
        "schema_version": TRAINING_AUDIT_HASHES_SCHEMA,
        "operation_identity": operation_identity,
        "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "audit_report_byte_count": len(report_bytes),
        "audit_receipt_path": receipt_path.relative_to(config.root).as_posix(),
        "runner_inventory_sha256": runner["inventory_sha256"],
        "test_harness_inventory_sha256": test_harness["inventory_sha256"],
        "bindings": bindings,
        "files": code_files,
        "artifacts": activation_artifacts,
        "smoke_artifacts": smoke_artifacts,
    }
    hashes = {**hashes_payload, "inventory_identity": stable_hash(hashes_payload)}
    hashes_bytes = _canonical_bytes(hashes)

    # Test execution may be long.  Re-evaluate every exact prospective input
    # before publishing evidence for it.
    _revalidate_before_publication(
        config,
        selected_profile,
        activation,
        code_files,
        activation_artifacts,
        smoke_artifacts,
        test_harness,
        smoke_id,
    )
    _require_absent_targets(config.root, targets)

    completed_at = datetime.now(timezone.utc).isoformat()
    receipt_payload = {
        "schema_version": TRAINING_AUDIT_RECEIPT_SCHEMA,
        "operation": operation,
        "operation_identity": operation_identity,
        "runner": runner,
        "test_harness": test_harness,
        "bindings": bindings,
        "verdict": overall.value,
        "gates_identity": stable_hash(gates),
        "test_results": [
            {"name": result["name"], "result_identity": result["result_identity"]} for result in test_results
        ],
        "report": {
            "path": report_path.relative_to(config.root).as_posix(),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "byte_count": len(report_bytes),
            "report_identity": report["report_identity"],
        },
        "hash_inventory": {
            "path": hashes_path.relative_to(config.root).as_posix(),
            "sha256": hashlib.sha256(hashes_bytes).hexdigest(),
            "byte_count": len(hashes_bytes),
            "inventory_identity": hashes["inventory_identity"],
        },
        "artifact_inventory_identity": stable_hash(activation_artifacts),
        "smoke_artifact_inventory_identity": stable_hash(smoke_artifacts),
        "completed_at": completed_at,
        "paths_exposed": False,
    }
    receipt = {**receipt_payload, "receipt_identity": stable_hash(receipt_payload)}
    receipt_bytes = _canonical_bytes(receipt)

    # The receipt is intentionally last: a crash or race can leave inert
    # residue, but can never leave an applicable audit without its receipt.
    _write_exclusive(config.root, report_path, report_bytes)
    _write_exclusive(config.root, hashes_path, hashes_bytes)
    _write_exclusive(config.root, receipt_path, receipt_bytes)

    return TrainingAuditExecution(
        operation_identity=operation_identity,
        verdict=overall,
        report_path=report_path,
        hashes_path=hashes_path,
        receipt_path=receipt_path,
    )


def verify_training_audit_execution(
    config: ProjectConfig,
    report: Mapping[str, Any] | None,
    activation: ConditionedTrainingActivation,
) -> AuditStatus:
    """Reverify one immutable server-managed audit receipt without execution."""

    if report is None:
        return AuditStatus.NOT_AUDITED
    try:
        with open_training_audit_execution_snapshot(config, report, activation) as snapshot:
            return snapshot.status
    except (
        CampaignValidationError,
        ConditionedActivationError,
        FileNotFoundError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
        UnsafeFilesystemOperation,
    ):
        return AuditStatus.STALE


@contextmanager
def open_training_audit_execution_snapshot(
    config: ProjectConfig,
    report: Mapping[str, Any] | None,
    activation: ConditionedTrainingActivation,
):
    """Retain one coherent report/hash/receipt/action set until dispatch.

    ``report=None`` deliberately derives the public report mapping from the
    exact retained report descriptor; it never performs a preliminary path
    read.  On normal context exit all four descriptors, visible names, and
    bytes are verified once more before their handles are closed.
    """

    stack = ExitStack()
    snapshot: TrainingAuditExecutionSnapshot | None = None
    try:
        report_path = _configured_existing_file(config, "audit_report")
        hashes_path = _configured_existing_file(config, "audit_hashes")
        receipt_path = training_audit_receipt_path(config)
        report_file = _open_held_audit_file(stack, report_path, config.root)
        hashes_file = _open_held_audit_file(stack, hashes_path, config.root)
        receipt_file = _open_held_audit_file(stack, receipt_path, config.root)
        report_bytes = report_file.payload
        hashes_bytes = hashes_file.payload
        receipt_bytes = receipt_file.payload
        stored_report = _mapping_from_bytes(report_bytes)
        receipt = _mapping_from_bytes(receipt_bytes)
        operation_identity = stored_report.get("operation_identity")
        operation = receipt.get("operation")
        source_job_id = operation.get("source_job_id") if isinstance(operation, Mapping) else None
        if not _is_sha(operation_identity) or not isinstance(source_job_id, str):
            raise ValueError("audit action binding is malformed")
        action_path = training_audit_action_record_path(config.root, source_job_id, operation_identity)
        action_file = _open_held_audit_file(stack, action_path, config.root)
        action_record_bytes = action_file.payload
        held = (report_file, hashes_file, receipt_file, action_file)
        for item in held:
            item.verify_unchanged()
        effective_report = stored_report if report is None else dict(report)
        status = _verify_training_audit_execution_bytes(
            config,
            effective_report,
            activation,
            report_bytes=report_bytes,
            hashes_bytes=hashes_bytes,
            receipt_bytes=receipt_bytes,
            action_record_bytes=action_record_bytes,
        )
        receipt_identity = receipt.get("receipt_identity")
        receipt_identity_value = receipt_identity if isinstance(receipt_identity, str) else ""
        evidence = {
            "schema_version": TRAINING_AUDIT_LAUNCH_AUTHORIZATION_SCHEMA,
            "status": status.value,
            "operation_identity": operation_identity,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "hash_inventory_sha256": hashlib.sha256(hashes_bytes).hexdigest(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "receipt_identity": receipt_identity_value,
            "action_record_sha256": hashlib.sha256(action_record_bytes).hexdigest(),
        }
        snapshot = TrainingAuditExecutionSnapshot(
            status=status,
            operation_identity=operation_identity,
            report=stored_report,
            report_sha256=evidence["report_sha256"],
            hash_inventory_sha256=evidence["hash_inventory_sha256"],
            receipt_sha256=evidence["receipt_sha256"],
            receipt_identity=receipt_identity_value,
            action_record_sha256=evidence["action_record_sha256"],
            launch_authorization_evidence_sha256=stable_hash(evidence),
            _held_files=held,
        )
        snapshot.verify_unchanged()
        yield snapshot
        snapshot.verify_unchanged()
    finally:
        if snapshot is not None:
            snapshot._closed = True
        stack.close()


def _verify_training_audit_execution_bytes(
    config: ProjectConfig,
    report: Mapping[str, Any],
    activation: ConditionedTrainingActivation,
    *,
    report_bytes: bytes,
    hashes_bytes: bytes,
    receipt_bytes: bytes,
    action_record_bytes: bytes,
) -> AuditStatus:
    """Validate semantic bindings using only one retained byte snapshot."""

    try:
        report_path = _configured_existing_file(config, "audit_report")
        hashes_path = _configured_existing_file(config, "audit_hashes")
        receipt_path = training_audit_receipt_path(config)
        stored_report = _mapping_from_bytes(report_bytes)
        hashes = _mapping_from_bytes(hashes_bytes)
        receipt = _mapping_from_bytes(receipt_bytes)
        if stored_report != dict(report):
            return AuditStatus.STALE
        if not _validate_report(stored_report):
            return AuditStatus.STALE
        if not _validate_hash_inventory(hashes):
            return AuditStatus.STALE
        if not _validate_receipt(receipt):
            return AuditStatus.STALE

        bindings = _activation_bindings(activation)
        operation_identity = stored_report["operation_identity"]
        runner = training_audit_runner_inventory(config.root)
        test_harness = training_audit_test_harness_inventory(config.root)
        if (
            hashes["operation_identity"] != operation_identity
            or receipt["operation_identity"] != operation_identity
            or stored_report["bindings"] != bindings
            or hashes["bindings"] != bindings
            or receipt["bindings"] != bindings
            or stored_report["runner"] != runner
            or receipt["runner"] != runner
            or hashes["runner_inventory_sha256"] != runner["inventory_sha256"]
            or stored_report["test_harness"] != test_harness
            or receipt["test_harness"] != test_harness
            or hashes["test_harness_inventory_sha256"] != test_harness["inventory_sha256"]
        ):
            return AuditStatus.STALE
        if receipt["operation_identity"] != stable_hash(receipt["operation"]):
            return AuditStatus.STALE
        operation = receipt["operation"]
        expected_operation_paths = {
            "report_path": report_path.relative_to(config.root).as_posix(),
            "hashes_path": hashes_path.relative_to(config.root).as_posix(),
            "receipt_path": receipt_path.relative_to(config.root).as_posix(),
        }
        if (
            operation.get("bindings") != bindings
            or operation.get("runner_inventory_sha256") != runner["inventory_sha256"]
            or operation.get("test_plan_identity") != _TEST_PLAN_IDENTITY
            or operation.get("test_harness_inventory_sha256") != test_harness["inventory_sha256"]
            or operation.get("profile") != activation.profile.value
            or any(operation.get(key) != value for key, value in expected_operation_paths.items())
        ):
            return AuditStatus.STALE

        if (
            hashes["audit_report_sha256"] != hashlib.sha256(report_bytes).hexdigest()
            or hashes["audit_report_byte_count"] != len(report_bytes)
            or hashes["audit_receipt_path"] != receipt_path.relative_to(config.root).as_posix()
            or receipt["report"]
            != {
                "path": report_path.relative_to(config.root).as_posix(),
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "byte_count": len(report_bytes),
                "report_identity": stored_report["report_identity"],
            }
            or receipt["hash_inventory"]
            != {
                "path": hashes_path.relative_to(config.root).as_posix(),
                "sha256": hashlib.sha256(hashes_bytes).hexdigest(),
                "byte_count": len(hashes_bytes),
                "inventory_identity": hashes["inventory_identity"],
            }
        ):
            return AuditStatus.STALE

        code_files = _training_code_inventory(config.root, activation)
        if hashes["files"] != code_files:
            return AuditStatus.STALE
        if not _verify_artifact_inventory(config.root, hashes["artifacts"]):
            return AuditStatus.STALE
        if receipt["artifact_inventory_identity"] != stable_hash(hashes["artifacts"]):
            return AuditStatus.STALE
        if receipt["smoke_artifact_inventory_identity"] != stable_hash(hashes["smoke_artifacts"]):
            return AuditStatus.STALE

        gates = stored_report["gates"]
        smoke_status = _audit_status(gates["cpu_cuda_smoke_evidence"])
        if smoke_status is AuditStatus.PASS:
            smoke_id = operation.get("smoke_id")
            if (
                not isinstance(smoke_id, str)
                or _smoke_bundle_inventory(config.root, smoke_id) != hashes["smoke_artifacts"]
            ):
                return AuditStatus.STALE
        elif not _verify_artifact_inventory(config.root, hashes["smoke_artifacts"]):
            return AuditStatus.STALE
        expected_gates = _derived_gate_verdicts(smoke_status, stored_report["test_results"])
        if gates != expected_gates:
            return AuditStatus.STALE
        if receipt["gates_identity"] != stable_hash(gates):
            return AuditStatus.STALE
        expected_results = [
            {"name": result["name"], "result_identity": result["result_identity"]}
            for result in stored_report["test_results"]
        ]
        if receipt["test_results"] != expected_results:
            return AuditStatus.STALE
        verdict = _overall(gates)
        if stored_report["verdict"] != verdict.value or receipt["verdict"] != verdict.value:
            return AuditStatus.STALE
        source_job_id = operation.get("source_job_id")
        if not isinstance(source_job_id, str):
            return AuditStatus.STALE
        action_record = _mapping_from_bytes(action_record_bytes)
        if not _validate_action_record(action_record):
            return AuditStatus.STALE
        expected_action_record = {
            "schema_version": TRAINING_AUDIT_ACTION_RECORD_SCHEMA,
            "source_job_id": source_job_id,
            "operation_identity": operation_identity,
            "prospective_configuration_identity_sha256": bindings["prospective_configuration_identity_sha256"],
            "base_config_sha256": action_record["base_config_sha256"],
            "verdict": verdict.value,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "hash_inventory_sha256": hashlib.sha256(hashes_bytes).hexdigest(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "receipt_identity": receipt["receipt_identity"],
            "config_unchanged": True,
            "configuration_activated": False,
            "training_started": False,
            "paths_exposed": False,
        }
        if action_record != {
            **expected_action_record,
            "record_identity": stable_hash(expected_action_record),
        }:
            return AuditStatus.STALE
        return verdict
    except (
        CampaignValidationError,
        ConditionedActivationError,
        FileNotFoundError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
        UnsafeFilesystemOperation,
    ):
        return AuditStatus.STALE


def training_audit_runner_inventory(project_root: str | Path) -> dict[str, Any]:
    """Return the exact current implementation identity of the audit service."""

    root = Path(project_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in _RUNNER_SOURCE_PATHS:
        files.append(_file_record(root, _relative_target(root, relative)))
    payload = {
        "schema_version": TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA,
        "auditor_id": TRAINING_AUDITOR_ID,
        "files": files,
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}


def training_audit_test_harness_inventory(project_root: str | Path) -> dict[str, Any]:
    """Bind the exact tracked pytest harness and reject executable additions."""

    root = Path(project_root).resolve()
    tests_root = require_confined_path(root / "tests", root)
    mandatory = {"pyproject.toml", "tests/conftest.py"}
    forbidden = {
        "conftest.py",
        "pytest.py",
        "sitecustomize.py",
        "usercustomize.py",
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
        "src/pytest.py",
        "src/sitecustomize.py",
        "src/usercustomize.py",
    }
    try:
        if any(_anchored_entry_exists(root, relative) for relative in forbidden):
            raise TrainingAuditExecutionError(
                "audit_test_bootstrap",
                "An unexpected Python or pytest bootstrap file can influence the fixed audit plan.",
            )
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise TrainingAuditExecutionError(
            "audit_test_bootstrap", "The fixed Python and pytest bootstrap namespace is unsafe."
        ) from exc
    try:
        stdout = pinned_git_ls_files(
            root,
            ("tests", "pyproject.toml"),
            timeout_seconds=15.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise TrainingAuditExecutionError(
            "audit_test_inventory", "The tracked test-harness inventory timed out."
        ) from exc
    except (OSError, PinnedExecutableError, subprocess.SubprocessError) as exc:
        raise TrainingAuditExecutionError(
            "audit_test_inventory", "The tracked test-harness inventory is unavailable."
        ) from exc
    tracked_relatives = {os.fsdecode(raw).replace("\\", "/") for raw in stdout.split(b"\0") if raw}
    if not mandatory.issubset(tracked_relatives):
        raise TrainingAuditExecutionError("audit_test_inventory", "Mandatory pytest harness files are not tracked.")

    tracked_paths = {root.joinpath(*PurePosixPath(relative).parts) for relative in tracked_relatives}
    try:
        discovered_python = _anchored_test_executables(root, tests_root)
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise TrainingAuditExecutionError(
            "audit_test_inventory", "The pytest harness could not be scanned safely."
        ) from exc
    if discovered_python - tracked_relatives:
        raise TrainingAuditExecutionError(
            "audit_test_inventory", "Untracked executable pytest harness files are forbidden."
        )

    try:
        files = [
            _file_record(root, path)
            for path in sorted(tracked_paths, key=lambda item: item.relative_to(root).as_posix())
        ]
    except (FileNotFoundError, OSError, TrainingAuditExecutionError, UnsafeFilesystemOperation) as exc:
        raise TrainingAuditExecutionError(
            "audit_test_inventory", "A tracked pytest harness file is missing, unsafe, or changed while hashed."
        ) from exc
    interpreter = Path(sys.executable).resolve()
    try:
        interpreter_record = _external_file_record(interpreter)
        pytest_record = _external_file_record(
            Path(importlib.metadata.distribution("pytest").locate_file("pytest/__init__.py")).resolve()
        )
    except (FileNotFoundError, OSError, UnsafeFilesystemOperation) as exc:
        raise TrainingAuditExecutionError(
            "audit_test_runtime", "The bound audit Python or pytest runtime is unavailable or unsafe."
        ) from exc
    payload = {
        "schema_version": TRAINING_AUDIT_TEST_HARNESS_SCHEMA,
        "files": files,
        "interpreter": {
            "sha256": interpreter_record["sha256"],
            "byte_count": interpreter_record["byte_count"],
            "python_version": platform.python_version(),
        },
        "pytest": {
            "version": importlib.metadata.version("pytest"),
            "entrypoint_sha256": pytest_record["sha256"],
        },
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}


def training_audit_receipt_path(config: ProjectConfig) -> Path:
    """Derive the fixed receipt sidecar for the configured report path."""

    report = _configured_target(config, "audit_report")
    if report.name == "audit_report.json":
        name = "audit_receipt.json"
    else:
        name = f"{report.stem}.receipt.json"
    return _relative_target(config.root, (report.parent / name).relative_to(config.root).as_posix())


def _anchored_entry_exists(root: Path, relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafeFilesystemOperation("audit bootstrap path is invalid")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in parts[:-1]:
            if not anchor.lexists(part):
                return False
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        return anchor.lexists(parts[-1])


def _anchored_test_executables(root: Path, tests_root: Path) -> set[str]:
    executable_suffixes = {".py", ".pyw", ".pyc", ".pyo", ".pyd", ".so"}
    base = PurePosixPath(tests_root.relative_to(root).as_posix())
    discovered: set[str] = set()

    def walk(anchor: AnchoredDirectory, relative: PurePosixPath) -> None:
        boundary_device = anchor.directory_metadata().st_dev
        for name in anchor.names():
            item_relative = relative / name
            metadata = anchor.lstat(name)
            reparse = bool(int(getattr(metadata, "st_file_attributes", 0)) & 0x400)
            if stat.S_ISLNK(metadata.st_mode) or reparse or metadata.st_dev != boundary_device:
                raise UnsafeFilesystemOperation("pytest harness crosses a link, reparse, or filesystem boundary")
            if stat.S_ISDIR(metadata.st_mode):
                if name == "__pycache__":
                    continue
                with anchor.open_directory_immovable(name) as child:
                    walk(child, item_relative)
            elif stat.S_ISREG(metadata.st_mode):
                if PurePosixPath(name).suffix.casefold() in executable_suffixes:
                    discovered.add(item_relative.as_posix())
            else:
                raise UnsafeFilesystemOperation("pytest harness contains a special filesystem entry")

    with open_anchored_directory(tests_root, root) as anchor:
        walk(anchor, base)
    return discovered


def training_audit_action_record_path(
    project_root: str | Path,
    source_job_id: str,
    operation_identity: str,
) -> Path:
    """Return the fixed immutable Conditioned-service action record path."""

    root = Path(project_root).resolve()
    if not _valid_reference_id(source_job_id, maximum=160, optional=False) or not _is_sha(operation_identity):
        raise TrainingAuditExecutionError("audit_action_record", "The server-managed audit action identity is invalid.")
    relative = (
        PurePosixPath("runs/v3/conditioned-dataset-v5")
        / source_job_id
        / "training_audits"
        / f"{operation_identity}.json"
    )
    return _relative_target(root, relative.as_posix())


def _config(context: ProjectContext | ProjectConfig) -> ProjectConfig:
    if isinstance(context, ProjectConfig):
        return context
    if context.config_path is not None and context.config_path.is_file():
        return ProjectConfig.load(context.project_root)
    return ProjectConfig(context.project_root.resolve(), context.config_path, dict(context.config))


def _activation_bindings(activation: ConditionedTrainingActivation) -> dict[str, Any]:
    return {
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign.get("campaign_identity"),
        "training_code_identity_sha256": dict(activation.campaign.get("code_identity") or {}).get("sha256"),
        "prospective_configuration_identity_sha256": stable_hash(activation.config.values),
    }


def _training_code_inventory(
    root: Path,
    activation: ConditionedTrainingActivation,
) -> list[dict[str, str]]:
    paths = training_code_identity_source_paths(root)
    records = []
    for path in paths:
        record = _file_record(root, path)
        records.append({"path": record["path"], "sha256_before": record["sha256"]})
    declared = activation.campaign.get("code_identity")
    declared_files = declared.get("files") if isinstance(declared, Mapping) else None
    if not isinstance(declared_files, list):
        raise TrainingAuditExecutionError(
            "audit_code_identity", "The selected campaign lacks a complete training-code identity."
        )
    expected = {str(item.get("path")): str(item.get("sha256")) for item in declared_files if isinstance(item, Mapping)}
    observed = {item["path"]: item["sha256_before"] for item in records}
    if expected != observed:
        raise TrainingAuditExecutionError(
            "audit_code_identity", "The selected campaign training-code identity is stale."
        )
    return records


def _activation_artifact_inventory(
    root: Path,
    activation: ConditionedTrainingActivation,
) -> list[dict[str, Any]]:
    paths = [activation.freeze_path, activation.campaign_config_path, *activation.artifacts.values()]
    return _merge_artifact_inventory([_file_record(root, path) for path in paths])


def _activation_source(activation: ConditionedTrainingActivation) -> dict[str, Any]:
    return {
        "kind": "prospective_conditioned_activation",
        "freeze_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign["campaign_identity"],
    }


def _artifact_source(activation: ConditionedTrainingActivation, name: str) -> dict[str, Any]:
    path = activation.artifacts[name]
    record = _file_record(activation.config.root, path)
    return {
        "kind": name,
        "sha256": record["sha256"],
        "byte_count": record["byte_count"],
    }


def _verify_smoke_evidence(
    root: Path,
    activation: ConditionedTrainingActivation,
    smoke_id: str | None,
) -> tuple[AuditStatus, list[dict[str, Any]], list[dict[str, Any]]]:
    if smoke_id is None:
        return (
            AuditStatus.INCONCLUSIVE,
            [{"kind": "cpu_cuda_smoke_bundle", "status": "MISSING"}],
            [],
        )
    try:
        plan = load_plan(root, smoke_id)
        bundle = verify_complete_bundle(root, plan)
        bindings = bundle.evidence.get("bindings")
        artifact_records = {
            name: _file_record(root, activation.artifacts[name])
            for name in (
                "view_manifest",
                "split_manifest",
                "conditioning_vocabulary",
                "benchmark_manifest",
            )
        }
        expected = {
            "activation_manifest_sha256": activation.freeze_sha256,
            "campaign_config_sha256": activation.campaign_config_sha256,
            "campaign_identity_sha256": activation.campaign["campaign_identity"],
            "training_code_identity_sha256": activation.campaign["code_identity"]["sha256"],
            "dataset_view_manifest_sha256": artifact_records["view_manifest"]["sha256"],
            "split_manifest_sha256": artifact_records["split_manifest"]["sha256"],
            "conditioning_vocabulary_sha256": artifact_records["conditioning_vocabulary"]["sha256"],
            "benchmark_manifest_sha256": artifact_records["benchmark_manifest"]["sha256"],
        }
        if not isinstance(bindings, Mapping) or any(bindings.get(key) != value for key, value in expected.items()):
            return (
                AuditStatus.FAIL,
                [{"kind": "cpu_cuda_smoke_bundle", "status": "BINDING_MISMATCH"}],
                [],
            )
        runs = bundle.evidence.get("runs")
        if not isinstance(runs, Mapping) or set(runs) != {"cpu", "cuda"}:
            return AuditStatus.FAIL, [{"kind": "cpu_cuda_smoke_bundle", "status": "INCOMPLETE"}], []
        published_evidence = _registered_smoke_evidence(root, smoke_id, bundle.evidence)
        sources = [
            {
                "kind": "cpu_cuda_smoke_bundle",
                "smoke_id": smoke_id,
                "evidence_identity": published_evidence["evidence_identity"],
                "plan_identity": bundle.evidence["plan_identity"],
                "cpu_receipt_identity": runs["cpu"]["receipt_identity"],
                "cuda_receipt_identity": runs["cuda"]["receipt_identity"],
            }
        ]
        artifacts = _smoke_bundle_inventory(root, smoke_id)
        by_path = {item["path"]: item for item in artifacts}
        for device in ("cpu", "cuda"):
            output_prefix = run_bundle_directory(root, smoke_id).joinpath(device).relative_to(root).as_posix()
            output_inventory = runs[device].get("output_inventory")
            if not isinstance(output_inventory, Mapping):
                raise SmokeBundleError("smoke_receipt_invalid", "Smoke output inventory is missing.")
            for name, record in output_inventory.items():
                expected_record = {"path": f"{output_prefix}/{name}", **dict(record)}
                if by_path.get(expected_record["path"]) != expected_record:
                    raise SmokeBundleError("smoke_receipt_stale", "Smoke output inventory changed after verification.")
            receipt_relative = f"{output_prefix}/smoke_run_receipt.json"
            if by_path.get(receipt_relative, {}).get("sha256") != runs[device].get("receipt_sha256"):
                raise SmokeBundleError("smoke_receipt_stale", "Smoke receipt bytes changed after verification.")
        return AuditStatus.PASS, sources, artifacts
    except FileNotFoundError:
        return AuditStatus.INCONCLUSIVE, [{"kind": "cpu_cuda_smoke_bundle", "status": "MISSING"}], []
    except (
        KeyError,
        OSError,
        RecursionError,
        SmokeBundleError,
        TypeError,
        ValueError,
        UnsafeFilesystemOperation,
    ) as exc:
        return (
            AuditStatus.FAIL,
            [{"kind": "cpu_cuda_smoke_bundle", "status": "INVALID", "reason": type(exc).__name__}],
            [],
        )


def _smoke_bundle_inventory(root: Path, smoke_id: str) -> list[dict[str, Any]]:
    artifact_root = artifact_bundle_directory(root, smoke_id)
    run_root = run_bundle_directory(root, smoke_id)
    records = _anchored_tree_inventory(root, artifact_root)
    records.extend(_anchored_tree_inventory(root, run_root))
    return _merge_artifact_inventory(records)


def _registered_smoke_evidence(
    root: Path,
    smoke_id: str,
    recomputed: Mapping[str, Any],
) -> dict[str, Any]:
    published = _read_mapping(artifact_bundle_directory(root, smoke_id) / "smoke_evidence.json", root)
    executions = published.get("server_execution_identities")
    if (
        not isinstance(executions, Mapping)
        or set(executions) != {"cpu", "cuda"}
        or any(not _is_sha(value) for value in executions.values())
        or not _valid_identity(published, "evidence_identity")
    ):
        raise SmokeBundleError("smoke_evidence_invalid", "Registered smoke evidence is invalid.")
    published_body = {
        key: value
        for key, value in published.items()
        if key not in {"evidence_identity", "server_execution_identities"}
    }
    recomputed_body = {key: value for key, value in recomputed.items() if key != "evidence_identity"}
    if published_body != recomputed_body:
        raise SmokeBundleError("smoke_evidence_stale", "Registered smoke evidence differs from the verified bundle.")
    return published


def _anchored_tree_inventory(
    root: Path,
    tree: Path,
    *,
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    confined = require_confined_path(tree, root)
    base = PurePosixPath(confined.relative_to(root).as_posix())
    skipped = excluded or set()
    records: list[dict[str, Any]] = []

    def walk(anchor: AnchoredDirectory, relative: PurePosixPath) -> None:
        for name in sorted(anchor.names(), key=lambda value: (value.casefold(), value)):
            item_relative = relative / name
            local_relative = item_relative.relative_to(base).as_posix()
            if local_relative in skipped:
                continue
            metadata = anchor.lstat(name)
            reparse = bool(int(getattr(metadata, "st_file_attributes", 0)) & 0x400)
            if stat.S_ISLNK(metadata.st_mode) or reparse:
                raise UnsafeFilesystemOperation("smoke inventory crosses a linked or reparse seam")
            if stat.S_ISDIR(metadata.st_mode):
                with anchor.open_directory_immovable(name) as child:
                    walk(child, item_relative)
            elif stat.S_ISREG(metadata.st_mode):
                records.append(_anchored_inventory_record(anchor, name, item_relative.as_posix()))
            else:
                raise UnsafeFilesystemOperation("smoke inventory contains a non-file entry")

    with AnchoredDirectory(confined, root) as anchor:
        walk(anchor, base)
    return records


def _anchored_inventory_record(anchor: AnchoredDirectory, name: str, relative: str) -> dict[str, Any]:
    digest, byte_count, _payload = _stream_anchored_regular(
        anchor,
        name,
        max_bytes=_MAX_INVENTORY_FILE_BYTES,
        capture=False,
    )
    return {"path": relative, "sha256": digest, "byte_count": byte_count}


def _stream_anchored_regular(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
    capture: bool,
) -> tuple[str, int, bytes | None]:
    """Read one exact direct-child inode and prove its path never changed."""

    if type(max_bytes) is not int or max_bytes < 1:
        raise UnsafeFilesystemOperation("anchored file bound is invalid")
    anchor.verify()
    boundary_device = anchor.directory_metadata().st_dev
    path_before = anchor.lstat(name)
    _require_safe_bound_regular(path_before, boundary_device=boundary_device, max_bytes=max_bytes)
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        opened = os.fstat(descriptor)
        _require_safe_bound_regular(opened, boundary_device=boundary_device, max_bytes=max_bytes)
        if not _same_exact_file(path_before, opened):
            raise UnsafeFilesystemOperation("anchored file changed while being opened")

        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture else None
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes or total > opened.st_size:
                raise UnsafeFilesystemOperation("anchored file grew beyond its bound while being read")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)

        after = os.fstat(descriptor)
        path_after = anchor.lstat(name)
        _require_safe_bound_regular(after, boundary_device=boundary_device, max_bytes=max_bytes)
        _require_safe_bound_regular(path_after, boundary_device=boundary_device, max_bytes=max_bytes)
        if total != opened.st_size or not _same_exact_file(opened, after) or not _same_exact_file(after, path_after):
            raise UnsafeFilesystemOperation("anchored file changed while being read")
        anchor.verify()
        return digest.hexdigest(), total, b"".join(chunks) if chunks is not None else None
    finally:
        os.close(descriptor)


def _require_safe_bound_regular(metadata: os.stat_result, *, boundary_device: int, max_bytes: int) -> None:
    reparse = bool(int(getattr(metadata, "st_file_attributes", 0)) & 0x400)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or reparse
        or metadata.st_nlink != 1
        or metadata.st_dev != boundary_device
        or metadata.st_size < 0
        or metadata.st_size > max_bytes
    ):
        raise UnsafeFilesystemOperation("anchored evidence is linked, special, oversized, or crosses a boundary")


def _same_exact_file(first: os.stat_result, second: os.stat_result) -> bool:
    return OwnedFileIdentity.from_stat(first).matches(second) and _stat_identity(first) == _stat_identity(second)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
    )


def _execute_fixed_test_plan(root: Path, operation_identity: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    site_roots = _pytest_site_roots()
    for command in _TEST_PLAN:
        name = str(command["name"])
        base = f".pytest_tmp_training_audit_{operation_identity[:16]}_{name}"
        base_path = _relative_target(root, base)
        if os.path.lexists(base_path):
            results.append(_test_result(name, "INCONCLUSIVE", None, b"", b"temporary root exists"))
            continue
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-S",
            "-X",
            f"pycache_prefix={base}/pycache",
            "-c",
            _PYTEST_BOOTSTRAP,
            str(len(site_roots)),
            *site_roots,
            root.as_posix(),
            *list(command["targets"]),
            "-c",
            "pyproject.toml",
            "--rootdir=.",
            "--confcutdir=.",
            "--import-mode=importlib",
            "-o",
            "addopts=",
            "-q",
            f"--basetemp={base}",
            "-p",
            "no:cacheprovider",
        ]
        environment = {key: value for key, value in os.environ.items() if key.upper() in _TEST_ENVIRONMENT_ALLOWLIST}
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["SPRITELAB_PROGRESS"] = "0"
        try:
            completed = subprocess.run(
                argv,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=int(command["timeout_seconds"]),
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = bytes(exc.stdout or b"")
            stderr = bytes(exc.stderr or b"")
            results.append(_test_result(name, "INCONCLUSIVE", None, stdout, stderr))
        except OSError as exc:
            results.append(_test_result(name, "INCONCLUSIVE", None, b"", type(exc).__name__.encode("ascii")))
        else:
            verdict = "PASS" if completed.returncode == 0 else "FAIL"
            results.append(_test_result(name, verdict, int(completed.returncode), completed.stdout, completed.stderr))
    return results


def _pytest_site_roots() -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for raw in sys.path:
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw).resolve()
        if path.name.casefold() not in {"site-packages", "dist-packages"} or not path.is_dir():
            continue
        rendered = str(path)
        identity = rendered.casefold()
        if identity not in seen:
            seen.add(identity)
            roots.append(rendered)
    if not roots:
        raise TrainingAuditExecutionError("audit_test_runtime", "The isolated pytest runtime is unavailable.")
    return roots


def _test_result(
    name: str,
    verdict: str,
    return_code: int | None,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    payload = {
        "schema_version": TRAINING_AUDIT_TEST_RESULT_SCHEMA,
        "name": name,
        "plan_identity": _TEST_PLAN_IDENTITY,
        "verdict": verdict,
        "return_code": return_code,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_byte_count": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_byte_count": len(stderr),
        "paths_exposed": False,
    }
    return {**payload, "result_identity": stable_hash(payload)}


def _revalidate_before_publication(
    config: ProjectConfig,
    profile: TrainingProfile,
    original: ConditionedTrainingActivation,
    code_files: list[dict[str, str]],
    activation_artifacts: list[dict[str, Any]],
    smoke_artifacts: list[dict[str, Any]],
    test_harness: Mapping[str, Any],
    smoke_id: str | None,
) -> None:
    try:
        current = load_conditioned_training_activation(
            config,
            profile,
            expected_campaign=original.campaign,
            require_audit=False,
            require_activation_commit=False,
        )
        if _activation_bindings(current) != _activation_bindings(original):
            raise TrainingAuditExecutionError("audit_inputs_changed", "Audit inputs changed during execution.")
        if _training_code_inventory(config.root, current) != code_files:
            raise TrainingAuditExecutionError("audit_inputs_changed", "Training code changed during execution.")
        if training_audit_test_harness_inventory(config.root) != test_harness:
            raise TrainingAuditExecutionError(
                "audit_inputs_changed", "The fixed test harness changed during execution."
            )
        smoke_status, _sources, current_smoke_artifacts = _verify_smoke_evidence(config.root, current, smoke_id)
        if smoke_artifacts and smoke_status is not AuditStatus.PASS:
            raise TrainingAuditExecutionError("audit_inputs_changed", "Smoke evidence changed during execution.")
        if current_smoke_artifacts != smoke_artifacts:
            raise TrainingAuditExecutionError("audit_inputs_changed", "Smoke artifacts changed during execution.")
        if _activation_artifact_inventory(config.root, current) != activation_artifacts:
            raise TrainingAuditExecutionError("audit_inputs_changed", "Audit artifacts changed during execution.")
    except TrainingAuditExecutionError:
        raise
    except (CampaignValidationError, ConditionedActivationError, OSError, RecursionError, ValueError) as exc:
        raise TrainingAuditExecutionError(
            "audit_inputs_changed", "Audit inputs changed or became unavailable during execution."
        ) from exc


def _validate_report(value: Mapping[str, Any]) -> bool:
    if set(value) != _REPORT_KEYS or value.get("schema_version") != TRAINING_AUDIT_REPORT_SCHEMA:
        return False
    if (
        value.get("independent") is not True
        or value.get("generated_by") != "server_managed_training_audit_service"
        or value.get("paths_exposed") is not False
        or not _is_sha(value.get("operation_identity"))
        or not _valid_identity(value, "report_identity")
    ):
        return False
    runner = value.get("runner")
    test_harness = value.get("test_harness")
    gates = value.get("gates")
    evidence = value.get("gate_evidence")
    results = value.get("test_results")
    if (
        not _valid_runner_inventory(runner)
        or not _valid_test_harness_inventory(test_harness)
        or not isinstance(gates, Mapping)
        or set(gates) != set(MANDATORY_TRAINING_AUDIT_GATES)
        or not isinstance(evidence, Mapping)
        or set(evidence) != set(MANDATORY_TRAINING_AUDIT_GATES)
        or not isinstance(results, list)
        or len(results) != 2
    ):
        return False
    for gate in MANDATORY_TRAINING_AUDIT_GATES:
        verdict = gates.get(gate)
        item = evidence.get(gate)
        if verdict not in {"PASS", "FAIL", "INCONCLUSIVE"} or not isinstance(item, Mapping):
            return False
        if set(item) != {"gate", "verdict", "sources", "evidence_identity"}:
            return False
        if item.get("gate") != gate or item.get("verdict") != verdict or not isinstance(item.get("sources"), list):
            return False
        if not _valid_identity(item, "evidence_identity"):
            return False
    if [result.get("name") for result in results if isinstance(result, Mapping)] != ["curated", "full"]:
        return False
    if any(not _valid_test_result(result) for result in results):
        return False
    return value.get("verdict") == _overall(gates).value


def _validate_hash_inventory(value: Mapping[str, Any]) -> bool:
    return (
        set(value) == _HASHES_KEYS
        and value.get("schema_version") == TRAINING_AUDIT_HASHES_SCHEMA
        and _is_sha(value.get("operation_identity"))
        and _is_sha(value.get("audit_report_sha256"))
        and type(value.get("audit_report_byte_count")) is int
        and value["audit_report_byte_count"] > 0
        and _is_sha(value.get("runner_inventory_sha256"))
        and _is_sha(value.get("test_harness_inventory_sha256"))
        and isinstance(value.get("files"), list)
        and isinstance(value.get("artifacts"), list)
        and isinstance(value.get("smoke_artifacts"), list)
        and _valid_identity(value, "inventory_identity")
    )


def _validate_receipt(value: Mapping[str, Any]) -> bool:
    if (
        set(value) != _RECEIPT_KEYS
        or value.get("schema_version") != TRAINING_AUDIT_RECEIPT_SCHEMA
        or value.get("paths_exposed") is not False
        or not _valid_identity(value, "receipt_identity")
        or not _is_sha(value.get("operation_identity"))
        or not _is_sha(value.get("gates_identity"))
        or not _is_sha(value.get("artifact_inventory_identity"))
        or not _is_sha(value.get("smoke_artifact_inventory_identity"))
        or not _valid_runner_inventory(value.get("runner"))
        or not _valid_test_harness_inventory(value.get("test_harness"))
        or not _valid_timestamp(value.get("completed_at"))
    ):
        return False
    operation = value.get("operation")
    if not isinstance(operation, Mapping) or set(operation) != {
        "schema_version",
        "operation_nonce",
        "profile",
        "source_job_id",
        "smoke_id",
        "test_plan_identity",
        "test_harness_inventory_sha256",
        "report_path",
        "hashes_path",
        "receipt_path",
        "bindings",
        "runner_inventory_sha256",
    }:
        return False
    if (
        operation.get("schema_version") != TRAINING_AUDIT_OPERATION_SCHEMA
        or not isinstance(operation.get("operation_nonce"), str)
        or _OPERATION_NONCE.fullmatch(operation["operation_nonce"]) is None
        or operation.get("profile") not in {profile.value for profile in TrainingProfile}
        or not _valid_reference_id(operation.get("source_job_id"), maximum=160, optional=True)
        or not _valid_reference_id(operation.get("smoke_id"), maximum=128, optional=True)
        or not _is_sha(operation.get("test_plan_identity"))
        or not _is_sha(operation.get("test_harness_inventory_sha256"))
        or not _is_sha(operation.get("runner_inventory_sha256"))
    ):
        return False
    report = value.get("report")
    if (
        not isinstance(report, Mapping)
        or set(report) != {"path", "sha256", "byte_count", "report_identity"}
        or not isinstance(report.get("path"), str)
        or not _is_sha(report.get("sha256"))
        or type(report.get("byte_count")) is not int
        or report["byte_count"] <= 0
        or not _is_sha(report.get("report_identity"))
    ):
        return False
    inventory = value.get("hash_inventory")
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != {"path", "sha256", "byte_count", "inventory_identity"}
        or not isinstance(inventory.get("path"), str)
        or not _is_sha(inventory.get("sha256"))
        or type(inventory.get("byte_count")) is not int
        or inventory["byte_count"] <= 0
        or not _is_sha(inventory.get("inventory_identity"))
    ):
        return False
    results = value.get("test_results")
    return (
        isinstance(results, list)
        and [item.get("name") for item in results if isinstance(item, Mapping)] == ["curated", "full"]
        and all(
            isinstance(item, Mapping)
            and set(item) == {"name", "result_identity"}
            and _is_sha(item.get("result_identity"))
            for item in results
        )
    )


def _validate_action_record(value: Mapping[str, Any]) -> bool:
    return (
        set(value) == _ACTION_RECORD_KEYS
        and value.get("schema_version") == TRAINING_AUDIT_ACTION_RECORD_SCHEMA
        and _valid_reference_id(value.get("source_job_id"), maximum=160, optional=False)
        and all(
            _is_sha(value.get(field))
            for field in (
                "operation_identity",
                "prospective_configuration_identity_sha256",
                "base_config_sha256",
                "report_sha256",
                "hash_inventory_sha256",
                "receipt_sha256",
                "receipt_identity",
            )
        )
        and value.get("verdict") in {"PASS", "FAIL", "INCONCLUSIVE"}
        and value.get("config_unchanged") is True
        and value.get("configuration_activated") is False
        and value.get("training_started") is False
        and value.get("paths_exposed") is False
        and _valid_identity(value, "record_identity")
    )


def _valid_runner_inventory(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "auditor_id",
        "files",
        "inventory_sha256",
    }:
        return False
    return (
        value.get("schema_version") == TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA
        and value.get("auditor_id") == TRAINING_AUDITOR_ID
        and isinstance(value.get("files"), list)
        and _valid_identity(value, "inventory_sha256")
    )


def _valid_test_harness_inventory(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "files",
        "interpreter",
        "pytest",
        "inventory_sha256",
    }:
        return False
    interpreter = value.get("interpreter")
    pytest_record = value.get("pytest")
    return (
        value.get("schema_version") == TRAINING_AUDIT_TEST_HARNESS_SCHEMA
        and isinstance(value.get("files"), list)
        and isinstance(interpreter, Mapping)
        and set(interpreter) == {"sha256", "byte_count", "python_version"}
        and _is_sha(interpreter.get("sha256"))
        and type(interpreter.get("byte_count")) is int
        and interpreter["byte_count"] > 0
        and isinstance(interpreter.get("python_version"), str)
        and isinstance(pytest_record, Mapping)
        and set(pytest_record) == {"version", "entrypoint_sha256"}
        and isinstance(pytest_record.get("version"), str)
        and _is_sha(pytest_record.get("entrypoint_sha256"))
        and _valid_identity(value, "inventory_sha256")
    )


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 64:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _valid_test_result(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "name",
        "plan_identity",
        "verdict",
        "return_code",
        "stdout_sha256",
        "stdout_byte_count",
        "stderr_sha256",
        "stderr_byte_count",
        "paths_exposed",
        "result_identity",
    }:
        return False
    verdict = value.get("verdict")
    return (
        value.get("schema_version") == TRAINING_AUDIT_TEST_RESULT_SCHEMA
        and value.get("name") in {"curated", "full"}
        and value.get("plan_identity") == _TEST_PLAN_IDENTITY
        and verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
        and (value.get("return_code") is None or type(value.get("return_code")) is int)
        and (
            (verdict == "PASS" and value.get("return_code") == 0)
            or (verdict == "FAIL" and type(value.get("return_code")) is int and value["return_code"] != 0)
            or (verdict == "INCONCLUSIVE" and value.get("return_code") is None)
        )
        and _is_sha(value.get("stdout_sha256"))
        and type(value.get("stdout_byte_count")) is int
        and value["stdout_byte_count"] >= 0
        and _is_sha(value.get("stderr_sha256"))
        and type(value.get("stderr_byte_count")) is int
        and value["stderr_byte_count"] >= 0
        and value.get("paths_exposed") is False
        and _valid_identity(value, "result_identity")
    )


def _valid_identity(value: Mapping[str, Any], field: str) -> bool:
    identity = value.get(field)
    if not _is_sha(identity):
        return False
    payload = {key: item for key, item in value.items() if key != field}
    return stable_hash(payload) == identity


def _overall(gates: Mapping[str, Any]) -> AuditStatus:
    verdicts = tuple(gates.values())
    if any(type(value) is str and value == "FAIL" for value in verdicts):
        return AuditStatus.FAIL
    if len(verdicts) == len(MANDATORY_TRAINING_AUDIT_GATES) and all(
        type(value) is str and value == "PASS" for value in verdicts
    ):
        return AuditStatus.PASS
    return AuditStatus.INCONCLUSIVE


def _derived_gate_verdicts(
    smoke_status: AuditStatus,
    test_results: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    statuses = {str(result.get("name")): _audit_status(result.get("verdict")) for result in test_results}
    tests_status = _combine(
        statuses.get("curated", AuditStatus.INCONCLUSIVE), statuses.get("full", AuditStatus.INCONCLUSIVE)
    )
    direct = {
        "tracked_code_identity_inventory",
        "no_untracked_production_python",
        "dataset_view_freeze_campaign_vocabulary_identity",
        "dataset_and_training_manifest_qa",
        "campaign_experiment_compatibility",
    }
    smoke = {
        "production_loader_coverage",
        "cpu_cuda_smoke_evidence",
        "cuda_driver_torch_device_compatibility",
        "determinism_environment_qualification",
    }
    tests = {
        "backend_command_safety",
        "idempotency_concurrency_refusal",
        "output_root_resume_safety",
        "event_history_migration_identity",
        "publication_config_atomicity_restart",
        "filesystem_containment_link_defenses",
        "api_ui_privacy",
        "curated_full_test_results",
    }
    result: dict[str, str] = {}
    for gate in MANDATORY_TRAINING_AUDIT_GATES:
        if gate in direct:
            status = AuditStatus.PASS
        elif gate in smoke:
            status = smoke_status
        elif gate in tests:
            status = tests_status
        elif gate == "launch_receipt_execution_contract_binding":
            status = _combine(smoke_status, tests_status)
        else:
            raise TrainingAuditExecutionError(
                "audit_gate_implementation", "The server audit implementation has an incomplete gate map."
            )
        result[gate] = _verdict(status)
    return result


def _combine(*statuses: AuditStatus) -> AuditStatus:
    if AuditStatus.FAIL in statuses:
        return AuditStatus.FAIL
    if statuses and all(status is AuditStatus.PASS for status in statuses):
        return AuditStatus.PASS
    return AuditStatus.INCONCLUSIVE


def _audit_status(value: Any) -> AuditStatus:
    try:
        return AuditStatus(str(value))
    except ValueError:
        return AuditStatus.INCONCLUSIVE


def _verdict(status: AuditStatus) -> str:
    return status.value if status in {AuditStatus.PASS, AuditStatus.FAIL} else "INCONCLUSIVE"


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    target = require_confined_path(path, root)
    with open_anchored_directory(target.parent, root) as parent:
        try:
            return _anchored_inventory_record(parent, target.name, target.relative_to(root).as_posix())
        except (FileNotFoundError, OSError, UnsafeFilesystemOperation) as exc:
            raise TrainingAuditExecutionError(
                "audit_artifact_unsafe",
                "An audit evidence artifact is missing, linked, unsafe, or changed while hashed.",
            ) from exc


def _external_file_record(path: Path) -> dict[str, Any]:
    """Hash a runtime dependency through an anchor rooted at its resolved parent."""

    target = path.resolve(strict=True)
    with AnchoredDirectory(target.parent, target.parent) as parent:
        record = _anchored_inventory_record(parent, target.name, target.name)
    return {"sha256": record["sha256"], "byte_count": record["byte_count"]}


def _merge_artifact_inventory(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for item in items:
        value = dict(item)
        path = value.get("path")
        if not isinstance(path, str) or set(value) != {"path", "sha256", "byte_count"}:
            raise TrainingAuditExecutionError("audit_artifact_inventory", "An audit artifact identity is invalid.")
        if path in observed and observed[path] != value:
            raise TrainingAuditExecutionError(
                "audit_artifact_inventory", "An audit artifact path has conflicting identities."
            )
        observed[path] = value
    return [observed[path] for path in sorted(observed)]


def _verify_artifact_inventory(root: Path, items: Any) -> bool:
    if not isinstance(items, list):
        return False
    try:
        normalized = _merge_artifact_inventory(items)
    except TrainingAuditExecutionError:
        return False
    if normalized != items:
        return False
    for item in normalized:
        try:
            target = _relative_target(root, item["path"])
            observed = _file_record(root, target)
        except (OSError, TrainingAuditExecutionError, ValueError, UnsafeFilesystemOperation):
            return False
        if observed != item:
            return False
    return True


def _configured_target(config: ProjectConfig, key: str) -> Path:
    training = config.values.get("training")
    raw = training.get(key) if isinstance(training, Mapping) else None
    if not isinstance(raw, str):
        raise TrainingAuditExecutionError("audit_output_path", "A configured audit output path is invalid.")
    target = _relative_target(config.root, raw)
    relative = PurePosixPath(target.relative_to(config.root).as_posix())
    expected_name = {"audit_report": "audit_report.json", "audit_hashes": "audit_hashes.json"}.get(key)
    parent = relative.parent
    managed_parent = parent == _AUDIT_OUTPUT_ROOT
    managed_attempt = parent.parent == _AUDIT_ATTEMPT_ROOT and _valid_reference_id(
        parent.name, maximum=80, optional=False
    )
    if relative.name != expected_name or not (managed_parent or managed_attempt):
        raise TrainingAuditExecutionError(
            "audit_output_path",
            "Training-audit outputs must use fixed names in the managed audit namespace.",
        )
    return target


def _configured_existing_file(config: ProjectConfig, key: str) -> Path:
    return _configured_target(config, key)


def _relative_target(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\x00" in raw:
        raise TrainingAuditExecutionError("audit_output_path", "Audit paths must be canonical relative paths.")
    pure = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TrainingAuditExecutionError("audit_output_path", "Audit paths must be canonical relative paths.")
    if pure.as_posix() != raw.replace("\\", "/") or "\\" in raw:
        raise TrainingAuditExecutionError("audit_output_path", "Audit paths must use portable separators.")
    return require_confined_path(root / Path(*pure.parts), root)


def _ensure_project_directory(root: Path, target: Path) -> None:
    confined = require_confined_path(target, root, allow_root=True)
    relative = confined.relative_to(root)
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts:
            if not anchor.lexists(part):
                anchor.mkdir(part)
            anchor = stack.enter_context(anchor.open_directory_immovable(part))


def _require_absent_targets(root: Path, targets: Sequence[Path]) -> None:
    for target in targets:
        confined = require_confined_path(target, root)
        with open_anchored_directory(confined.parent, root) as parent:
            if parent.lexists(confined.name):
                raise TrainingAuditExecutionError(
                    "audit_output_exists", "Audit outputs are immutable and must not replace an existing path."
                )


def _write_exclusive(root: Path, target: Path, content: bytes) -> None:
    confined = require_confined_path(target, root)
    if len(content) > _MAX_MAPPING_BYTES:
        raise TrainingAuditExecutionError("audit_output_size", "Audit output exceeds its bounded canonical size.")
    with open_anchored_directory(confined.parent, root) as parent:
        if parent.lexists(confined.name):
            raise TrainingAuditExecutionError(
                "audit_output_exists", "Audit outputs are immutable and must not replace an existing path."
            )
        descriptor = -1
        identity: OwnedFileIdentity | None = None
        temporary: str | None = None
        direct_final = False
        try:
            if os.name == "nt":
                for _attempt in range(16):
                    temporary = f".{confined.name}.audit-staging-{uuid.uuid4().hex}"
                    try:
                        descriptor = parent.open_file(
                            temporary,
                            os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                            0o600,
                        )
                    except FileExistsError:
                        continue
                    break
                else:
                    raise UnsafeFilesystemOperation("could not allocate unpredictable audit staging")
            else:
                try:
                    descriptor = parent.open_anonymous_file(0o600)
                except (OSError, UnsafeFilesystemOperation):
                    # Some POSIX filesystems (notably macOS/APFS) have no
                    # anonymous-file publication primitive.  Creating the
                    # immutable canonical name O_EXCL from birth has no
                    # pathname publication seam; a failure leaves, at worst,
                    # inert audit residue because the receipt is published
                    # last and every reader revalidates exact canonical bytes.
                    descriptor = parent.open_file(
                        confined.name,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                        0o600,
                    )
                    direct_final = True

            boundary_device = parent.directory_metadata().st_dev
            created = os.fstat(descriptor)
            created_link_count = 1 if temporary is not None or direct_final else 0
            if (
                not stat.S_ISREG(created.st_mode)
                or stat.S_ISLNK(created.st_mode)
                or bool(int(getattr(created, "st_file_attributes", 0)) & 0x400)
                or created.st_dev != boundary_device
                or created.st_nlink != created_link_count
                or created.st_size != 0
            ):
                raise UnsafeFilesystemOperation("exclusive audit staging was not created empty")
            identity = OwnedFileIdentity.from_stat(created)
            offset = 0
            while offset < len(content):
                written_count = os.write(descriptor, content[offset:])
                if written_count <= 0:
                    raise OSError("exclusive audit staging write was incomplete")
                offset += written_count
            os.fsync(descriptor)
            written = os.fstat(descriptor)
            if (
                not stat.S_ISREG(written.st_mode)
                or stat.S_ISLNK(written.st_mode)
                or bool(int(getattr(written, "st_file_attributes", 0)) & 0x400)
                or written.st_dev != boundary_device
                or written.st_nlink != created_link_count
                or written.st_size != len(content)
                or not identity.matches(written)
            ):
                raise UnsafeFilesystemOperation("exclusive audit staging changed while written")
            if temporary is not None:
                staged = parent.lstat(temporary)
                if not _same_exact_file(written, staged):
                    raise UnsafeFilesystemOperation("exclusive audit staging name changed while written")
            elif direct_final:
                staged = parent.lstat(confined.name)
                if not _same_exact_file(written, staged):
                    raise UnsafeFilesystemOperation("exclusive audit final name changed while written")

            if not direct_final:
                try:
                    parent.publish_held_file_no_replace(
                        descriptor,
                        temporary,
                        confined.name,
                        identity=identity,
                    )
                except FileExistsError as exc:
                    raise TrainingAuditExecutionError(
                        "audit_output_exists", "Audit outputs are immutable and must not replace an existing path."
                    ) from exc

            held_published = os.fstat(descriptor)
            published = parent.lstat(confined.name)
            _require_safe_bound_regular(
                held_published,
                boundary_device=boundary_device,
                max_bytes=max(1, len(content)),
            )
            if (
                held_published.st_size != len(content)
                or not identity.matches(held_published)
                or not _same_exact_file(held_published, published)
            ):
                raise UnsafeFilesystemOperation("exclusive audit publication changed identity")
            _digest, byte_count, reread = _stream_anchored_regular(
                parent,
                confined.name,
                max_bytes=max(1, len(content)),
                capture=True,
            )
            held_after = os.fstat(descriptor)
            final_path = parent.lstat(confined.name)
            if reread is None:
                raise UnsafeFilesystemOperation("exclusive audit publication could not be reread")
            if (
                byte_count != len(content)
                or reread != content
                or _canonical_bytes(_mapping_from_bytes(reread)) != content
                or not _same_exact_file(held_published, held_after)
                or not _same_exact_file(held_after, final_path)
            ):
                raise UnsafeFilesystemOperation("exclusive audit publication changed during canonical reread")
            parent.verify()
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            if identity is not None:
                _quarantine_owned_audit_file(parent, confined.name, temporary or "", identity)
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _quarantine_owned_audit_file(
    parent: AnchoredDirectory,
    target_name: str,
    staging_name: str,
    identity: OwnedFileIdentity,
) -> None:
    """Retire only the exact inode created by this audit attempt."""

    for name in (target_name, staging_name):
        if not name:
            continue
        try:
            metadata = parent.lstat(name)
        except FileNotFoundError:
            continue
        if identity.matches(metadata):
            parent.quarantine_if_owned(
                name,
                identity,
                prefix=f".training-audit-residue-{uuid.uuid4().hex}-",
            )


def _read_mapping(path: Path, root: Path) -> dict[str, Any]:
    return _mapping_from_bytes(_read_regular_bytes(path, root))


def _mapping_from_bytes(payload: bytes) -> dict[str, Any]:
    if len(payload) > _MAX_MAPPING_BYTES:
        raise ValueError("audit mapping exceeds its bound")
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_mapping,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("audit mapping is not an object")
    return value


def _read_regular_bytes(path: Path, root: Path) -> bytes:
    target = require_confined_path(path, root)
    with open_anchored_directory(target.parent, root) as parent:
        _digest, _byte_count, payload = _stream_anchored_regular(
            parent,
            target.name,
            max_bytes=_MAX_MAPPING_BYTES,
            capture=True,
        )
        if payload is None:
            raise UnsafeFilesystemOperation("audit evidence bytes were not retained")
        return payload


def _read_held_audit_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_bytes:
            raise UnsafeFilesystemOperation("held audit evidence exceeds its size bound")
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _open_held_audit_file(
    stack: ExitStack,
    path: Path,
    root: Path,
) -> _HeldAuditFile:
    target = require_confined_path(path, root)
    parent = stack.enter_context(open_anchored_directory(target.parent, root))
    boundary_device = parent.directory_metadata().st_dev
    visible_before = parent.lstat(target.name)
    _require_safe_bound_regular(
        visible_before,
        boundary_device=boundary_device,
        max_bytes=_MAX_MAPPING_BYTES,
    )
    descriptor = parent.open_file_immovable(
        target.name,
        os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
    )
    stack.callback(os.close, descriptor)
    opened = os.fstat(descriptor)
    _require_safe_bound_regular(opened, boundary_device=boundary_device, max_bytes=_MAX_MAPPING_BYTES)
    if not _same_exact_file(visible_before, opened):
        raise UnsafeFilesystemOperation("audit evidence changed while its descriptor was retained")
    payload = _read_held_audit_descriptor(descriptor, maximum_bytes=_MAX_MAPPING_BYTES)
    after = os.fstat(descriptor)
    visible_after = parent.lstat(target.name)
    if (
        len(payload) != opened.st_size
        or not _same_exact_file(opened, after)
        or not _same_exact_file(after, visible_after)
    ):
        raise UnsafeFilesystemOperation("audit evidence changed while its bytes were retained")
    parent.verify()
    return _HeldAuditFile(
        path=target,
        parent=parent,
        descriptor=descriptor,
        identity=OwnedFileIdentity.from_stat(opened),
        metadata=after,
        payload=payload,
    )


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate audit mapping key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _valid_reference_id(value: Any, *, maximum: int, optional: bool) -> bool:
    if value is None:
        return optional
    return (
        isinstance(value, str)
        and len(value) <= maximum
        and value == value.strip()
        and _REFERENCE_ID.fullmatch(value) is not None
    )


__all__ = [
    "TRAINING_AUDITOR_ID",
    "TRAINING_AUDIT_ACTION_RECORD_SCHEMA",
    "TRAINING_AUDIT_LAUNCH_AUTHORIZATION_SCHEMA",
    "TRAINING_AUDIT_OPERATION_SCHEMA",
    "TRAINING_AUDIT_RECEIPT_SCHEMA",
    "TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA",
    "TRAINING_AUDIT_TEST_HARNESS_SCHEMA",
    "TrainingAuditExecution",
    "TrainingAuditExecutionError",
    "TrainingAuditExecutionSnapshot",
    "open_training_audit_execution_snapshot",
    "run_training_infrastructure_audit",
    "training_audit_action_record_path",
    "training_audit_receipt_path",
    "training_audit_runner_inventory",
    "training_audit_test_harness_inventory",
    "verify_training_audit_execution",
]

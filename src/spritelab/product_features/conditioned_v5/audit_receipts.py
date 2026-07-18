"""Strict receipts for server-managed conditioned Dataset-v5 audits.

The conditioned audit report remains the detailed technical evidence.  This
module binds that report to the server operation which actually invoked the
trusted auditor.  Receipts are deliberately path-free and use an exact schema
so consumers cannot silently accept weaker, caller-authored substitutes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final

from spritelab.training.campaign import stable_hash

AUDIT_RECEIPT_SCHEMA: Final = "spritelab.audit.conditioned-run-receipt.v1"
AUDIT_OPERATION_SCHEMA: Final = "spritelab.audit.conditioned-run-operation.v1"
AUDIT_ACTION_RECORD_SCHEMA: Final = "spritelab.audit.conditioned-run-action-record.v1"
AUDIT_KINDS: Final = frozenset({"label_audit", "dataset_validation"})
AUDIT_RECEIPT_ARTIFACTS: Final = {
    "label_audit": ("labeling_audit_receipt", "evidence/label_audit_receipt.json"),
    "dataset_validation": ("validation_receipt", "evidence/dataset_validation_receipt.json"),
}
AUDIT_ACTION_RECORD_ARTIFACTS: Final = {
    "label_audit": ("labeling_audit_action_record", "evidence/label_audit_action.json"),
    "dataset_validation": ("validation_action_record", "evidence/dataset_validation_action.json"),
}

AUDIT_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "audit_kind",
        "job_id",
        "operation_id",
        "operation_identity",
        "terminal_status",
        "server_managed",
        "report_sha256",
        "report_byte_count",
        "audit_run_identity",
        "candidate_identity",
        "payload_inventory_sha256",
        "image_count",
        "auditor_id",
        "auditor_code_identity_sha256",
        "auditor_inventory_sha256",
        "started_at",
        "completed_at",
        "paths_exposed",
        "receipt_identity",
    }
)

AUDIT_ACTION_RECORD_KEYS: Final = frozenset(
    {
        "schema_version",
        "audit_kind",
        "job_id",
        "operation_id",
        "operation_identity",
        "terminal_status",
        "server_managed",
        "report_sha256",
        "report_byte_count",
        "audit_run_identity",
        "receipt_sha256",
        "receipt_byte_count",
        "receipt_identity",
        "candidate_identity",
        "payload_inventory_sha256",
        "image_count",
        "auditor_id",
        "auditor_inventory_sha256",
        "started_at",
        "completed_at",
        "committed_at",
        "paths_exposed",
        "record_identity",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^conditioned-[0-9a-f]{20}$")
_OPERATION_ID = re.compile(r"^audit-[0-9a-f]{32}$")


class ConditionedAuditReceiptError(ValueError):
    """A server-managed audit receipt failed its exact trust contract."""


def audit_operation_identity(
    *,
    kind: str,
    job_id: str,
    operation_id: str,
    candidate_identity: str,
    payload_inventory_sha256: str,
    image_count: int,
    auditor_id: str,
    auditor_code_identity_sha256: str,
    auditor_inventory_sha256: str,
    started_at: str,
) -> str:
    """Bind one audit attempt before any report bytes exist."""

    payload = {
        "schema_version": AUDIT_OPERATION_SCHEMA,
        "audit_kind": kind,
        "job_id": job_id,
        "operation_id": operation_id,
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": payload_inventory_sha256,
        "image_count": image_count,
        "auditor_id": auditor_id,
        "auditor_code_identity_sha256": auditor_code_identity_sha256,
        "auditor_inventory_sha256": auditor_inventory_sha256,
        "started_at": started_at,
        "paths_exposed": False,
    }
    _validate_operation_payload(payload)
    return stable_hash(payload)


def build_audit_receipt(
    *,
    kind: str,
    job_id: str,
    operation_id: str,
    report_sha256: str,
    report_byte_count: int,
    report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    current_auditor_inventory: Mapping[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    """Build and self-validate one terminal PASS receipt."""

    auditor = report.get("auditor")
    if not isinstance(auditor, Mapping):
        raise ConditionedAuditReceiptError("The audit report lacks its trusted auditor binding.")
    candidate_identity = _digest(candidate.get("candidate_identity"), "candidate identity")
    payload_inventory_sha256 = _digest(
        candidate.get("payload_inventory_sha256"),
        "payload inventory identity",
    )
    image_count = _positive_integer(candidate.get("image_count"), "image count")
    auditor_id = _nonempty_string(auditor.get("auditor_id"), "auditor ID")
    auditor_code_identity = _digest(auditor.get("code_identity_sha256"), "auditor code identity")
    auditor_inventory_identity = _digest(
        current_auditor_inventory.get("inventory_sha256"),
        "auditor inventory identity",
    )
    operation_identity = audit_operation_identity(
        kind=kind,
        job_id=job_id,
        operation_id=operation_id,
        candidate_identity=candidate_identity,
        payload_inventory_sha256=payload_inventory_sha256,
        image_count=image_count,
        auditor_id=auditor_id,
        auditor_code_identity_sha256=auditor_code_identity,
        auditor_inventory_sha256=auditor_inventory_identity,
        started_at=started_at,
    )
    base = {
        "schema_version": AUDIT_RECEIPT_SCHEMA,
        "audit_kind": kind,
        "job_id": job_id,
        "operation_id": operation_id,
        "operation_identity": operation_identity,
        "terminal_status": "PASS",
        "server_managed": True,
        "report_sha256": report_sha256,
        "report_byte_count": report_byte_count,
        "audit_run_identity": report.get("audit_run_identity"),
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": payload_inventory_sha256,
        "image_count": image_count,
        "auditor_id": auditor_id,
        "auditor_code_identity_sha256": auditor_code_identity,
        "auditor_inventory_sha256": auditor_inventory_identity,
        "started_at": started_at,
        "completed_at": completed_at,
        "paths_exposed": False,
    }
    receipt = {**base, "receipt_identity": stable_hash(base)}
    return validate_audit_receipt(
        receipt,
        kind=kind,
        expected_job_id=job_id,
        expected_report_sha256=report_sha256,
        expected_report_byte_count=report_byte_count,
        report=report,
        candidate=candidate,
        current_auditor_inventory=current_auditor_inventory,
    )


def validate_audit_receipt(
    receipt: Mapping[str, Any],
    *,
    kind: str,
    expected_job_id: str | None,
    expected_report_sha256: str,
    expected_report_byte_count: int,
    report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    current_auditor_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one receipt against exact report, candidate, and current auditor context."""

    if not isinstance(receipt, Mapping) or set(receipt) != AUDIT_RECEIPT_KEYS:
        raise ConditionedAuditReceiptError("The server-managed audit receipt schema is invalid.")
    value = dict(receipt)
    if value.get("schema_version") != AUDIT_RECEIPT_SCHEMA or kind not in AUDIT_KINDS:
        raise ConditionedAuditReceiptError("The server-managed audit receipt schema is unsupported.")
    if value.get("audit_kind") != kind:
        raise ConditionedAuditReceiptError("The audit receipt kind differs from its report.")
    job_id = _nonempty_string(value.get("job_id"), "job ID")
    if _JOB_ID.fullmatch(job_id) is None or (expected_job_id is not None and job_id != expected_job_id):
        raise ConditionedAuditReceiptError("The audit receipt job binding is invalid.")
    operation_id = _nonempty_string(value.get("operation_id"), "operation ID")
    if _OPERATION_ID.fullmatch(operation_id) is None:
        raise ConditionedAuditReceiptError("The audit receipt operation binding is invalid.")
    if (
        value.get("terminal_status") != "PASS"
        or value.get("server_managed") is not True
        or value.get("paths_exposed") is not False
    ):
        raise ConditionedAuditReceiptError("The audit receipt is not a path-free server-managed PASS.")

    report_digest = _digest(expected_report_sha256, "report SHA-256")
    report_bytes = _positive_integer(expected_report_byte_count, "report byte count")
    if value.get("report_sha256") != report_digest or value.get("report_byte_count") != report_bytes:
        raise ConditionedAuditReceiptError("The audit receipt differs from the exact report bytes.")
    if not isinstance(report, Mapping):
        raise ConditionedAuditReceiptError("The audit receipt report is invalid.")
    run_identity = _digest(report.get("audit_run_identity"), "audit-run identity")
    run_payload = dict(report)
    run_payload.pop("audit_run_identity", None)
    if stable_hash(run_payload) != run_identity or value.get("audit_run_identity") != run_identity:
        raise ConditionedAuditReceiptError("The audit receipt audit-run identity is invalid.")
    if (
        report.get("verdict") != "PASS"
        or report.get("independent") is not True
        or report.get("generated_by_conditioned_workflow") is not False
    ):
        raise ConditionedAuditReceiptError("The audit receipt report is not a literal independent PASS.")

    candidate_identity = _digest(candidate.get("candidate_identity"), "candidate identity")
    payload_identity = _digest(candidate.get("payload_inventory_sha256"), "payload inventory identity")
    image_count = _positive_integer(candidate.get("image_count"), "image count")
    if (
        value.get("candidate_identity") != candidate_identity
        or value.get("payload_inventory_sha256") != payload_identity
        or value.get("image_count") != image_count
    ):
        raise ConditionedAuditReceiptError("The audit receipt differs from the exact candidate.")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping) or any(
        bindings.get(field) != expected
        for field, expected in (
            ("candidate_identity", candidate_identity),
            ("payload_inventory_sha256", payload_identity),
            ("image_count", image_count),
        )
    ):
        raise ConditionedAuditReceiptError("The audit report and receipt candidate bindings differ.")

    inventory_identity = _digest(
        current_auditor_inventory.get("inventory_sha256"),
        "auditor inventory identity",
    )
    inventory_auditor_id = _nonempty_string(current_auditor_inventory.get("auditor_id"), "auditor ID")
    auditor = report.get("auditor")
    if not isinstance(auditor, Mapping) or set(auditor) != {
        "auditor_id",
        "code_identity_sha256",
        "implementation_inventory",
    }:
        raise ConditionedAuditReceiptError("The audit report auditor binding is invalid.")
    if (
        auditor.get("auditor_id") != inventory_auditor_id
        or auditor.get("code_identity_sha256") != inventory_identity
        or auditor.get("implementation_inventory") != current_auditor_inventory
        or value.get("auditor_id") != inventory_auditor_id
        or value.get("auditor_code_identity_sha256") != inventory_identity
        or value.get("auditor_inventory_sha256") != inventory_identity
    ):
        raise ConditionedAuditReceiptError("The audit receipt is stale or from an untrusted auditor.")

    started = _timestamp(value.get("started_at"), "start timestamp")
    completed = _timestamp(value.get("completed_at"), "completion timestamp")
    if completed < started:
        raise ConditionedAuditReceiptError("The audit receipt timestamp order is invalid.")
    expected_operation_identity = audit_operation_identity(
        kind=kind,
        job_id=job_id,
        operation_id=operation_id,
        candidate_identity=candidate_identity,
        payload_inventory_sha256=payload_identity,
        image_count=image_count,
        auditor_id=inventory_auditor_id,
        auditor_code_identity_sha256=inventory_identity,
        auditor_inventory_sha256=inventory_identity,
        started_at=str(value["started_at"]),
    )
    if value.get("operation_identity") != expected_operation_identity:
        raise ConditionedAuditReceiptError("The audit receipt operation identity is invalid.")
    identity = _digest(value.get("receipt_identity"), "receipt identity")
    payload = dict(value)
    payload.pop("receipt_identity", None)
    if stable_hash(payload) != identity:
        raise ConditionedAuditReceiptError("The audit receipt self-identity is invalid.")
    return value


def build_audit_action_record(
    *,
    kind: str,
    job_id: str,
    report_sha256: str,
    report_byte_count: int,
    report: Mapping[str, Any],
    receipt_sha256: str,
    receipt_byte_count: int,
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    current_auditor_inventory: Mapping[str, Any],
    committed_at: str,
) -> dict[str, Any]:
    """Build the durable server action record selected by publication.

    The receipt proves what ran.  This distinct record proves that the service
    completed the exact job/operation and durably selected those report and
    receipt bytes.  It is intentionally unkeyed: the accepted local trust
    boundary is the server action plus a no-replace job-owned record.
    """

    validated_receipt = validate_audit_receipt(
        receipt,
        kind=kind,
        expected_job_id=job_id,
        expected_report_sha256=report_sha256,
        expected_report_byte_count=report_byte_count,
        report=report,
        candidate=candidate,
        current_auditor_inventory=current_auditor_inventory,
    )
    receipt_digest = _digest(receipt_sha256, "receipt SHA-256")
    receipt_bytes = _positive_integer(receipt_byte_count, "receipt byte count")
    base = {
        "schema_version": AUDIT_ACTION_RECORD_SCHEMA,
        "audit_kind": kind,
        "job_id": job_id,
        "operation_id": validated_receipt["operation_id"],
        "operation_identity": validated_receipt["operation_identity"],
        "terminal_status": "PASS",
        "server_managed": True,
        "report_sha256": report_sha256,
        "report_byte_count": report_byte_count,
        "audit_run_identity": validated_receipt["audit_run_identity"],
        "receipt_sha256": receipt_digest,
        "receipt_byte_count": receipt_bytes,
        "receipt_identity": validated_receipt["receipt_identity"],
        "candidate_identity": validated_receipt["candidate_identity"],
        "payload_inventory_sha256": validated_receipt["payload_inventory_sha256"],
        "image_count": validated_receipt["image_count"],
        "auditor_id": validated_receipt["auditor_id"],
        "auditor_inventory_sha256": validated_receipt["auditor_inventory_sha256"],
        "started_at": validated_receipt["started_at"],
        "completed_at": validated_receipt["completed_at"],
        "committed_at": committed_at,
        "paths_exposed": False,
    }
    record = {**base, "record_identity": stable_hash(base)}
    return validate_audit_action_record(
        record,
        kind=kind,
        expected_job_id=job_id,
        expected_report_sha256=report_sha256,
        expected_report_byte_count=report_byte_count,
        report=report,
        expected_receipt_sha256=receipt_digest,
        expected_receipt_byte_count=receipt_bytes,
        receipt=validated_receipt,
        candidate=candidate,
        current_auditor_inventory=current_auditor_inventory,
    )


def validate_audit_action_record(
    record: Mapping[str, Any],
    *,
    kind: str,
    expected_job_id: str | None,
    expected_report_sha256: str,
    expected_report_byte_count: int,
    report: Mapping[str, Any],
    expected_receipt_sha256: str,
    expected_receipt_byte_count: int,
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    current_auditor_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact report -> receipt -> durable-action chain."""

    if not isinstance(record, Mapping) or set(record) != AUDIT_ACTION_RECORD_KEYS:
        raise ConditionedAuditReceiptError("The server-managed audit action-record schema is invalid.")
    value = dict(record)
    if value.get("schema_version") != AUDIT_ACTION_RECORD_SCHEMA or value.get("audit_kind") != kind:
        raise ConditionedAuditReceiptError("The server-managed audit action-record schema is unsupported.")
    validated_receipt = validate_audit_receipt(
        receipt,
        kind=kind,
        expected_job_id=expected_job_id,
        expected_report_sha256=expected_report_sha256,
        expected_report_byte_count=expected_report_byte_count,
        report=report,
        candidate=candidate,
        current_auditor_inventory=current_auditor_inventory,
    )
    job_id = str(validated_receipt["job_id"])
    if value.get("job_id") != job_id or (expected_job_id is not None and job_id != expected_job_id):
        raise ConditionedAuditReceiptError("The audit action record belongs to another job.")
    expected = {
        "operation_id": validated_receipt["operation_id"],
        "operation_identity": validated_receipt["operation_identity"],
        "terminal_status": "PASS",
        "server_managed": True,
        "report_sha256": _digest(expected_report_sha256, "report SHA-256"),
        "report_byte_count": _positive_integer(expected_report_byte_count, "report byte count"),
        "audit_run_identity": validated_receipt["audit_run_identity"],
        "receipt_sha256": _digest(expected_receipt_sha256, "receipt SHA-256"),
        "receipt_byte_count": _positive_integer(expected_receipt_byte_count, "receipt byte count"),
        "receipt_identity": validated_receipt["receipt_identity"],
        "candidate_identity": validated_receipt["candidate_identity"],
        "payload_inventory_sha256": validated_receipt["payload_inventory_sha256"],
        "image_count": validated_receipt["image_count"],
        "auditor_id": validated_receipt["auditor_id"],
        "auditor_inventory_sha256": validated_receipt["auditor_inventory_sha256"],
        "started_at": validated_receipt["started_at"],
        "completed_at": validated_receipt["completed_at"],
        "paths_exposed": False,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ConditionedAuditReceiptError("The audit action record differs from its report or receipt chain.")
    committed = _timestamp(value.get("committed_at"), "commit timestamp")
    completed = _timestamp(value.get("completed_at"), "completion timestamp")
    if committed < completed:
        raise ConditionedAuditReceiptError("The audit action record predates audit completion.")
    identity = _digest(value.get("record_identity"), "action-record identity")
    payload = dict(value)
    payload.pop("record_identity", None)
    if stable_hash(payload) != identity:
        raise ConditionedAuditReceiptError("The audit action-record self-identity is invalid.")
    return value


def _validate_operation_payload(value: Mapping[str, Any]) -> None:
    kind = value.get("audit_kind")
    job_id = value.get("job_id")
    operation_id = value.get("operation_id")
    if kind not in AUDIT_KINDS:
        raise ConditionedAuditReceiptError("The audit operation kind is unsupported.")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ConditionedAuditReceiptError("The audit operation job binding is invalid.")
    if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
        raise ConditionedAuditReceiptError("The audit operation ID is invalid.")
    _digest(value.get("candidate_identity"), "candidate identity")
    _digest(value.get("payload_inventory_sha256"), "payload inventory identity")
    _positive_integer(value.get("image_count"), "image count")
    _nonempty_string(value.get("auditor_id"), "auditor ID")
    _digest(value.get("auditor_code_identity_sha256"), "auditor code identity")
    _digest(value.get("auditor_inventory_sha256"), "auditor inventory identity")
    _timestamp(value.get("started_at"), "start timestamp")
    if value.get("paths_exposed") is not False:
        raise ConditionedAuditReceiptError("The audit operation exposed a private path.")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConditionedAuditReceiptError(f"The audit receipt {label} is invalid.")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConditionedAuditReceiptError(f"The audit receipt {label} is invalid.")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConditionedAuditReceiptError(f"The audit receipt {label} is invalid.")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConditionedAuditReceiptError(f"The audit receipt {label} is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConditionedAuditReceiptError(f"The audit receipt {label} must include a timezone.")
    return parsed


__all__ = [
    "AUDIT_ACTION_RECORD_ARTIFACTS",
    "AUDIT_ACTION_RECORD_KEYS",
    "AUDIT_ACTION_RECORD_SCHEMA",
    "AUDIT_KINDS",
    "AUDIT_OPERATION_SCHEMA",
    "AUDIT_RECEIPT_ARTIFACTS",
    "AUDIT_RECEIPT_KEYS",
    "AUDIT_RECEIPT_SCHEMA",
    "ConditionedAuditReceiptError",
    "audit_operation_identity",
    "build_audit_action_record",
    "build_audit_receipt",
    "validate_audit_action_record",
    "validate_audit_receipt",
]

"""Durable PREPARED evidence and the authoritative project activation marker."""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final

from spritelab.product_features.training.action_lock import ACTION_LOCK_PROTOCOL_IDENTITY
from spritelab.training.campaign import stable_hash

ACTIVATION_RECEIPT_SCHEMA: Final = "spritelab.dataset.conditioned-activation-receipt.v2"
ACTIVATION_JOURNAL_SCHEMA: Final = "spritelab.dataset.conditioned-activation-journal.v1"
ACTIVATION_COMMIT_RECORD_SCHEMA: Final = "spritelab.dataset.conditioned-activation-commit-record.v1"
ACTIVATION_COMMIT_SEMANTICS: Final = "prepared-evidence-bound-to-project-marker"
ACTIVATION_PROJECT_COMMIT_SCHEMA: Final = "spritelab.dataset.conditioned-project-activation-commit.v1"
ACTIVATION_PROJECT_COMMIT_NAME: Final = ".spritelab-conditioned-activation-commit.json"
ACTIVATION_PROJECT_COMMIT_SEMANTICS: Final = "immutable-project-record-is-authoritative"
ACTIVATION_CONFIGURATION_KEYS: Final = (
    "dataset.view_manifest",
    "dataset.freeze_manifest",
    "training.dataset_freeze",
    "training.campaign_config",
    "execution.allow_dataset_production_freeze",
    "execution.allow_training",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^conditioned-[0-9a-f]{20}$")
_OPERATION_ID = re.compile(r"^activation-[0-9a-f]{32}$")


class ActivationCommitError(ValueError):
    """Activation PREPARED/commit evidence is malformed or inconsistent."""


def build_activation_commit_documents(
    *,
    job_id: str,
    operation_id: str,
    candidate_identity: str,
    publication_identity_sha256: str,
    activation_manifest_sha256: str,
    campaign_config_sha256: str,
    campaign_identity_sha256: str,
    authorization_id_sha256: str,
    config_before_sha256: str,
    config_after_sha256: str,
    prepared_at: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build receipt, commit record, then PREPARED journal in dependency order."""

    receipt_base = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA,
        "job_id": job_id,
        "operation_id": operation_id,
        "candidate_identity": candidate_identity,
        "publication_identity_sha256": publication_identity_sha256,
        "activation_manifest_sha256": activation_manifest_sha256,
        "campaign_config_sha256": campaign_config_sha256,
        "campaign_identity_sha256": campaign_identity_sha256,
        "authorization_id_sha256": authorization_id_sha256,
        "config_before_sha256": config_before_sha256,
        "config_after_sha256": config_after_sha256,
        "configuration_keys": list(ACTIVATION_CONFIGURATION_KEYS),
        "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
        "commit_semantics": ACTIVATION_COMMIT_SEMANTICS,
        "training_started": False,
        "paths_exposed": False,
        "prepared_at": prepared_at,
    }
    receipt = {**receipt_base, "receipt_identity": stable_hash(receipt_base)}
    receipt_bytes = _canonical_bytes(receipt)
    receipt_sha256 = _bytes_sha256(receipt_bytes)
    record_base = {
        "schema_version": ACTIVATION_COMMIT_RECORD_SCHEMA,
        "terminal_status": "PREPARED_PROJECT_COMMIT_BINDING",
        "job_id": job_id,
        "operation_id": operation_id,
        "receipt_sha256": receipt_sha256,
        "receipt_byte_count": len(receipt_bytes),
        "receipt_identity": receipt["receipt_identity"],
        "config_before_sha256": config_before_sha256,
        "config_after_sha256": config_after_sha256,
        "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
        "commit_semantics": ACTIVATION_COMMIT_SEMANTICS,
        "training_started": False,
        "paths_exposed": False,
        "prepared_at": prepared_at,
    }
    record = {**record_base, "record_identity": stable_hash(record_base)}
    record_bytes = _canonical_bytes(record)
    journal_base = {
        "schema_version": ACTIVATION_JOURNAL_SCHEMA,
        "status": "PREPARED",
        "job_id": job_id,
        "operation_id": operation_id,
        "receipt_sha256": receipt_sha256,
        "receipt_byte_count": len(receipt_bytes),
        "receipt_identity": receipt["receipt_identity"],
        "record_sha256": _bytes_sha256(record_bytes),
        "record_byte_count": len(record_bytes),
        "record_identity": record["record_identity"],
        "config_before_sha256": config_before_sha256,
        "config_after_sha256": config_after_sha256,
        "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
        "commit_semantics": ACTIVATION_COMMIT_SEMANTICS,
        "training_started": False,
        "paths_exposed": False,
        "prepared_at": prepared_at,
    }
    journal = {**journal_base, "journal_identity": stable_hash(journal_base)}
    validate_activation_commit_documents(
        receipt=receipt,
        journal=journal,
        record=record,
        expected_job_id=job_id,
        current_config_sha256=config_before_sha256,
        require_committed=False,
    )
    return receipt, journal, record


def validate_activation_commit_documents(
    *,
    receipt: Mapping[str, Any],
    journal: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_job_id: str,
    current_config_sha256: str,
    require_committed: bool,
) -> dict[str, Any]:
    """Validate PREPARED evidence and classify the current config boundary."""

    receipt_keys = {
        "schema_version",
        "job_id",
        "operation_id",
        "candidate_identity",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "authorization_id_sha256",
        "config_before_sha256",
        "config_after_sha256",
        "configuration_keys",
        "action_lock_protocol_identity",
        "commit_semantics",
        "training_started",
        "paths_exposed",
        "prepared_at",
        "receipt_identity",
    }
    record_keys = {
        "schema_version",
        "terminal_status",
        "job_id",
        "operation_id",
        "receipt_sha256",
        "receipt_byte_count",
        "receipt_identity",
        "config_before_sha256",
        "config_after_sha256",
        "action_lock_protocol_identity",
        "commit_semantics",
        "training_started",
        "paths_exposed",
        "prepared_at",
        "record_identity",
    }
    journal_keys = {
        "schema_version",
        "status",
        "job_id",
        "operation_id",
        "receipt_sha256",
        "receipt_byte_count",
        "receipt_identity",
        "record_sha256",
        "record_byte_count",
        "record_identity",
        "config_before_sha256",
        "config_after_sha256",
        "action_lock_protocol_identity",
        "commit_semantics",
        "training_started",
        "paths_exposed",
        "prepared_at",
        "journal_identity",
    }
    if not all(isinstance(value, Mapping) for value in (receipt, journal, record)):
        raise ActivationCommitError("Activation commit evidence is not a mapping.")
    receipt_value, journal_value, record_value = dict(receipt), dict(journal), dict(record)
    if set(receipt_value) != receipt_keys or set(record_value) != record_keys or set(journal_value) != journal_keys:
        raise ActivationCommitError("Activation commit evidence has an invalid exact schema.")
    job_id = str(receipt_value.get("job_id") or "")
    operation_id = str(receipt_value.get("operation_id") or "")
    if (
        receipt_value.get("schema_version") != ACTIVATION_RECEIPT_SCHEMA
        or record_value.get("schema_version") != ACTIVATION_COMMIT_RECORD_SCHEMA
        or journal_value.get("schema_version") != ACTIVATION_JOURNAL_SCHEMA
        or job_id != expected_job_id
        or _JOB_ID.fullmatch(job_id) is None
        or _OPERATION_ID.fullmatch(operation_id) is None
        or any(
            value.get("job_id") != job_id or value.get("operation_id") != operation_id
            for value in (record_value, journal_value)
        )
    ):
        raise ActivationCommitError("Activation commit evidence has an invalid job or operation binding.")
    digest_fields = (
        "candidate_identity",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "authorization_id_sha256",
        "config_before_sha256",
        "config_after_sha256",
    )
    if any(_SHA256.fullmatch(str(receipt_value.get(name) or "")) is None for name in digest_fields):
        raise ActivationCommitError("Activation receipt identities are invalid.")
    if receipt_value["config_before_sha256"] == receipt_value["config_after_sha256"]:
        raise ActivationCommitError("Activation must change the exact project configuration bytes.")
    shared = {
        "config_before_sha256": receipt_value["config_before_sha256"],
        "config_after_sha256": receipt_value["config_after_sha256"],
        "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
        "commit_semantics": ACTIVATION_COMMIT_SEMANTICS,
        "training_started": False,
        "paths_exposed": False,
        "prepared_at": receipt_value["prepared_at"],
    }
    if (
        receipt_value.get("configuration_keys") != list(ACTIVATION_CONFIGURATION_KEYS)
        or any(receipt_value.get(name) != expected for name, expected in shared.items())
        or any(record_value.get(name) != expected for name, expected in shared.items())
        or any(journal_value.get(name) != expected for name, expected in shared.items())
        or record_value.get("terminal_status") != "PREPARED_PROJECT_COMMIT_BINDING"
        or journal_value.get("status") != "PREPARED"
    ):
        raise ActivationCommitError("Activation commit protocol fields are inconsistent.")
    _timestamp(receipt_value.get("prepared_at"))
    receipt_identity = _identity(receipt_value, "receipt_identity")
    record_identity = _identity(record_value, "record_identity")
    journal_identity = _identity(journal_value, "journal_identity")
    del journal_identity
    receipt_bytes = _canonical_bytes(receipt_value)
    record_bytes = _canonical_bytes(record_value)
    if (
        record_value.get("receipt_sha256") != _bytes_sha256(receipt_bytes)
        or record_value.get("receipt_byte_count") != len(receipt_bytes)
        or record_value.get("receipt_identity") != receipt_identity
        or journal_value.get("receipt_sha256") != _bytes_sha256(receipt_bytes)
        or journal_value.get("receipt_byte_count") != len(receipt_bytes)
        or journal_value.get("receipt_identity") != receipt_identity
        or journal_value.get("record_sha256") != _bytes_sha256(record_bytes)
        or journal_value.get("record_byte_count") != len(record_bytes)
        or journal_value.get("record_identity") != record_identity
    ):
        raise ActivationCommitError("Activation journal bindings differ from its exact durable bytes.")
    current = str(current_config_sha256)
    if _SHA256.fullmatch(current) is None:
        raise ActivationCommitError("The current project configuration identity is invalid.")
    if current == receipt_value["config_after_sha256"]:
        committed = True
    elif current == receipt_value["config_before_sha256"]:
        committed = False
    else:
        raise ActivationCommitError("Project configuration differs from both activation CAS boundaries.")
    if require_committed and not committed:
        raise ActivationCommitError("Activation is only PREPARED; its exact configuration CAS is not committed.")
    return {
        "schema_version": ACTIVATION_COMMIT_RECORD_SCHEMA,
        "job_id": job_id,
        "operation_id": operation_id,
        "receipt_identity": receipt_identity,
        "record_identity": record_identity,
        "journal_identity": str(journal_value["journal_identity"]),
        "config_before_sha256": receipt_value["config_before_sha256"],
        "config_after_sha256": receipt_value["config_after_sha256"],
        "candidate_identity": receipt_value["candidate_identity"],
        "publication_identity_sha256": receipt_value["publication_identity_sha256"],
        "activation_manifest_sha256": receipt_value["activation_manifest_sha256"],
        "campaign_config_sha256": receipt_value["campaign_config_sha256"],
        "campaign_identity_sha256": receipt_value["campaign_identity_sha256"],
        "committed": committed,
        "training_started": False,
        "paths_exposed": False,
    }


def canonical_activation_commit_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value)


def build_activation_project_commit(
    *,
    receipt: Mapping[str, Any],
    journal: Mapping[str, Any],
    record: Mapping[str, Any],
    config_after_bytes: bytes,
) -> dict[str, Any]:
    """Build the fixed project-level immutable activation commit marker."""

    job_id = str(receipt.get("job_id") or "")
    summary = validate_activation_commit_documents(
        receipt=receipt,
        journal=journal,
        record=record,
        expected_job_id=job_id,
        current_config_sha256=str(receipt.get("config_before_sha256") or ""),
        require_committed=False,
    )
    try:
        config_after_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ActivationCommitError("Activation configuration bytes are not UTF-8.") from exc
    if hashlib.sha256(config_after_bytes).hexdigest() != summary["config_after_sha256"]:
        raise ActivationCommitError("Activation configuration bytes differ from PREPARED evidence.")
    receipt_bytes = _canonical_bytes(receipt)
    journal_bytes = _canonical_bytes(journal)
    record_bytes = _canonical_bytes(record)
    marker_base = {
        "schema_version": ACTIVATION_PROJECT_COMMIT_SCHEMA,
        "terminal_status": "COMMITTED_BY_IMMUTABLE_PROJECT_RECORD",
        "job_id": job_id,
        "operation_id": summary["operation_id"],
        "receipt_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/receipt.json",
        "receipt_sha256": _bytes_sha256(receipt_bytes),
        "receipt_byte_count": len(receipt_bytes),
        "receipt_identity": summary["receipt_identity"],
        "journal_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/journal.json",
        "journal_sha256": _bytes_sha256(journal_bytes),
        "journal_byte_count": len(journal_bytes),
        "journal_identity": summary["journal_identity"],
        "record_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/record.json",
        "record_sha256": _bytes_sha256(record_bytes),
        "record_byte_count": len(record_bytes),
        "record_identity": summary["record_identity"],
        "config_before_sha256": summary["config_before_sha256"],
        "config_after_sha256": summary["config_after_sha256"],
        "config_after_byte_count": len(config_after_bytes),
        "config_after_base64": base64.b64encode(config_after_bytes).decode("ascii"),
        "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
        "commit_semantics": ACTIVATION_PROJECT_COMMIT_SEMANTICS,
        "training_started": False,
        "paths_exposed": False,
        "prepared_at": receipt["prepared_at"],
    }
    return {**marker_base, "marker_identity": stable_hash(marker_base)}


def validate_activation_project_commit(
    marker: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    journal: Mapping[str, Any],
    record: Mapping[str, Any],
    current_config_sha256: str,
    expected_job_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Validate the authoritative marker and return its exact effective config."""

    keys = {
        "schema_version",
        "terminal_status",
        "job_id",
        "operation_id",
        "receipt_relative_path",
        "receipt_sha256",
        "receipt_byte_count",
        "receipt_identity",
        "journal_relative_path",
        "journal_sha256",
        "journal_byte_count",
        "journal_identity",
        "record_relative_path",
        "record_sha256",
        "record_byte_count",
        "record_identity",
        "config_before_sha256",
        "config_after_sha256",
        "config_after_byte_count",
        "config_after_base64",
        "action_lock_protocol_identity",
        "commit_semantics",
        "training_started",
        "paths_exposed",
        "prepared_at",
        "marker_identity",
    }
    value = dict(marker)
    if set(value) != keys:
        raise ActivationCommitError("Project activation commit marker has an invalid exact schema.")
    job_id = str(value.get("job_id") or "")
    if expected_job_id is not None and job_id != expected_job_id:
        raise ActivationCommitError("Project activation commit marker selects another job.")
    if (
        value.get("schema_version") != ACTIVATION_PROJECT_COMMIT_SCHEMA
        or value.get("terminal_status") != "COMMITTED_BY_IMMUTABLE_PROJECT_RECORD"
        or value.get("commit_semantics") != ACTIVATION_PROJECT_COMMIT_SEMANTICS
        or value.get("action_lock_protocol_identity") != ACTION_LOCK_PROTOCOL_IDENTITY
        or value.get("training_started") is not False
        or value.get("paths_exposed") is not False
        or _identity(value, "marker_identity") != value["marker_identity"]
    ):
        raise ActivationCommitError("Project activation commit protocol fields are invalid.")
    summary = validate_activation_commit_documents(
        receipt=receipt,
        journal=journal,
        record=record,
        expected_job_id=job_id,
        current_config_sha256=current_config_sha256,
        require_committed=False,
    )
    expected_paths = {
        "receipt_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/receipt.json",
        "journal_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/journal.json",
        "record_relative_path": f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt/record.json",
    }
    document_bindings = (
        ("receipt", receipt, summary["receipt_identity"]),
        ("journal", journal, summary["journal_identity"]),
        ("record", record, summary["record_identity"]),
    )
    if any(value.get(name) != expected for name, expected in expected_paths.items()):
        raise ActivationCommitError("Project activation commit paths are not fixed to its selected job.")
    for prefix, document, identity in document_bindings:
        content = _canonical_bytes(document)
        if (
            value.get(f"{prefix}_sha256") != _bytes_sha256(content)
            or value.get(f"{prefix}_byte_count") != len(content)
            or value.get(f"{prefix}_identity") != identity
        ):
            raise ActivationCommitError("Project activation commit document bindings changed.")
    try:
        encoded = value.get("config_after_base64")
        if not isinstance(encoded, str):
            raise TypeError
        config_after = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ActivationCommitError("Project activation commit configuration encoding is invalid.") from exc
    if (
        isinstance(value.get("config_after_byte_count"), bool)
        or not isinstance(value.get("config_after_byte_count"), int)
        or value["config_after_byte_count"] != len(config_after)
        or hashlib.sha256(config_after).hexdigest() != value.get("config_after_sha256")
        or value.get("config_before_sha256") != summary["config_before_sha256"]
        or value.get("config_after_sha256") != summary["config_after_sha256"]
        or value.get("operation_id") != summary["operation_id"]
        or value.get("prepared_at") != receipt.get("prepared_at")
    ):
        raise ActivationCommitError("Project activation commit configuration binding changed.")
    committed = {
        **summary,
        "committed": True,
        "reconciliation_required": current_config_sha256 == summary["config_before_sha256"],
        "marker_identity": value["marker_identity"],
    }
    return committed, config_after


def _identity(value: Mapping[str, Any], field: str) -> str:
    identity = str(value.get(field) or "")
    payload = dict(value)
    payload.pop(field, None)
    if _SHA256.fullmatch(identity) is None or stable_hash(payload) != identity:
        raise ActivationCommitError("An activation commit self-identity is invalid.")
    return identity


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json_dumps(value) + "\n").encode("utf-8")


def json_dumps(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True)


def _bytes_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ActivationCommitError("The activation PREPARED timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ActivationCommitError("The activation PREPARED timestamp is invalid.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ActivationCommitError("The activation PREPARED timestamp must include a timezone.")
    return parsed


__all__ = [
    "ACTIVATION_COMMIT_RECORD_SCHEMA",
    "ACTIVATION_COMMIT_SEMANTICS",
    "ACTIVATION_CONFIGURATION_KEYS",
    "ACTIVATION_JOURNAL_SCHEMA",
    "ACTIVATION_PROJECT_COMMIT_NAME",
    "ACTIVATION_PROJECT_COMMIT_SCHEMA",
    "ACTIVATION_PROJECT_COMMIT_SEMANTICS",
    "ACTIVATION_RECEIPT_SCHEMA",
    "ActivationCommitError",
    "build_activation_commit_documents",
    "build_activation_project_commit",
    "canonical_activation_commit_bytes",
    "validate_activation_commit_documents",
    "validate_activation_project_commit",
]

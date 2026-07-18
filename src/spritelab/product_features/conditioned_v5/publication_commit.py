"""Pure commit evidence for a conditioned dataset/campaign publication pair.

The filesystem publisher owns placement and no-replace semantics.  This module
only builds and validates the canonical, path-free evidence that makes the
campaign marker authoritative for both direct-final directories.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Final

from spritelab.training.campaign import stable_hash
from spritelab.utils.portable_paths import canonical_portable_relative_path, portable_path_collision_key

PUBLICATION_JOURNAL_SCHEMA: Final = "spritelab.dataset.conditioned-publication-journal.v1"
DATASET_COMMIT_SCHEMA: Final = "spritelab.dataset.conditioned-publication-dataset-commit.v1"
CAMPAIGN_COMMIT_SCHEMA: Final = "spritelab.dataset.conditioned-publication-campaign-commit.v1"
PUBLICATION_INVENTORY_SCHEMA: Final = "spritelab.dataset.freeze.inventory.v1"

PUBLICATION_JOURNAL_NAME: Final = "publication-journal.json"
PUBLICATION_COMMIT_SEMANTICS: Final = "campaign-marker-authorizes-exact-dataset-campaign-pair"

PUBLICATION_JOURNAL_STATUS: Final = "PREPARED"
DATASET_COMMIT_STATUS: Final = "COMPONENT_COMMITTED"
CAMPAIGN_COMMIT_STATUS: Final = "PAIR_COMMITTED"
PUBLICATION_JOURNAL_KIND: Final = "CONDITIONED_DATASET_CAMPAIGN_PAIR"
DATASET_COMMIT_KIND: Final = "DATASET_DIRECTORY"
CAMPAIGN_COMMIT_KIND: Final = "CAMPAIGN_DIRECTORY_PAIR_AUTHORITY"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY_KEYS = {
    "schema_version",
    "files",
    "file_count",
    "total_bytes",
    "inventory_sha256",
}
_JOURNAL_KEYS = {
    "schema_version",
    "status",
    "kind",
    "publication_identity_sha256",
    "journal_name",
    "dataset_relative_path",
    "dataset_commit_relative_path",
    "dataset_inventory",
    "campaign_relative_path",
    "campaign_commit_relative_path",
    "campaign_inventory",
    "commit_semantics",
    "pair_authority",
    "paths_exposed",
    "journal_identity",
}
_DATASET_COMMIT_KEYS = {
    "schema_version",
    "status",
    "kind",
    "publication_identity_sha256",
    "directory_relative_path",
    "commit_relative_path",
    "inventory",
    "journal_name",
    "journal_sha256",
    "journal_byte_count",
    "journal_identity",
    "commit_semantics",
    "pair_authority",
    "paths_exposed",
    "marker_identity",
}
_CAMPAIGN_COMMIT_KEYS = {
    "schema_version",
    "status",
    "kind",
    "publication_identity_sha256",
    "directory_relative_path",
    "commit_relative_path",
    "inventory",
    "dataset_relative_path",
    "dataset_inventory",
    "dataset_commit_relative_path",
    "dataset_marker_sha256",
    "dataset_marker_byte_count",
    "dataset_marker_identity",
    "journal_name",
    "journal_sha256",
    "journal_byte_count",
    "journal_identity",
    "commit_semantics",
    "pair_authority",
    "paths_exposed",
    "marker_identity",
}


class PublicationCommitError(ValueError):
    """Publication journal or commit evidence is malformed or inconsistent."""


def dataset_commit_name(publication_identity: str) -> str:
    """Return the dataset-parent marker name for one publication identity."""

    identity = _publication_identity(publication_identity)
    return f"conditioned-v5-{identity}.commit.json"


def campaign_commit_name(publication_identity: str) -> str:
    """Return the campaign-parent marker name for one publication identity."""

    identity = _publication_identity(publication_identity)
    return f"conditioned-v5-{identity}.commit.json"


def build_publication_journal(
    *,
    publication_identity: str,
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build PREPARED evidence for two exact direct-final directories."""

    identity = _publication_identity(publication_identity)
    base = {
        "schema_version": PUBLICATION_JOURNAL_SCHEMA,
        "status": PUBLICATION_JOURNAL_STATUS,
        "kind": PUBLICATION_JOURNAL_KIND,
        "publication_identity_sha256": identity,
        "journal_name": PUBLICATION_JOURNAL_NAME,
        "dataset_relative_path": _dataset_relative_path(identity),
        "dataset_commit_relative_path": _dataset_commit_relative_path(identity),
        "dataset_inventory": _inventory_envelope(dataset_inventory),
        "campaign_relative_path": _campaign_relative_path(identity),
        "campaign_commit_relative_path": _campaign_commit_relative_path(identity),
        "campaign_inventory": _inventory_envelope(campaign_inventory),
        "commit_semantics": PUBLICATION_COMMIT_SEMANTICS,
        "pair_authority": False,
        "paths_exposed": False,
    }
    journal = {**base, "journal_identity": stable_hash(base)}
    return validate_publication_journal(
        journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )


def validate_publication_journal(
    journal: Mapping[str, Any],
    *,
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one journal against both complete observed inventories."""

    value = _exact_mapping(journal, _JOURNAL_KEYS, "publication journal")
    identity = _publication_identity(value.get("publication_identity_sha256"))
    if (
        value.get("schema_version") != PUBLICATION_JOURNAL_SCHEMA
        or value.get("status") != PUBLICATION_JOURNAL_STATUS
        or value.get("kind") != PUBLICATION_JOURNAL_KIND
        or value.get("journal_name") != PUBLICATION_JOURNAL_NAME
        or value.get("dataset_relative_path") != _dataset_relative_path(identity)
        or value.get("dataset_commit_relative_path") != _dataset_commit_relative_path(identity)
        or value.get("campaign_relative_path") != _campaign_relative_path(identity)
        or value.get("campaign_commit_relative_path") != _campaign_commit_relative_path(identity)
        or value.get("commit_semantics") != PUBLICATION_COMMIT_SEMANTICS
        or value.get("pair_authority") is not False
        or value.get("paths_exposed") is not False
    ):
        raise PublicationCommitError("Publication journal protocol or fixed paths are invalid.")
    _validate_inventory_envelope(value.get("dataset_inventory"), dataset_inventory, "dataset")
    _validate_inventory_envelope(value.get("campaign_inventory"), campaign_inventory, "campaign")
    _self_identity(value, "journal_identity", "publication journal")
    return value


def build_dataset_commit(
    *,
    journal: Mapping[str, Any],
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the non-authoritative component marker for the dataset directory."""

    journal_value = validate_publication_journal(
        journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    identity = str(journal_value["publication_identity_sha256"])
    journal_bytes = canonical_publication_commit_bytes(journal_value)
    base = {
        "schema_version": DATASET_COMMIT_SCHEMA,
        "status": DATASET_COMMIT_STATUS,
        "kind": DATASET_COMMIT_KIND,
        "publication_identity_sha256": identity,
        "directory_relative_path": _dataset_relative_path(identity),
        "commit_relative_path": _dataset_commit_relative_path(identity),
        "inventory": _inventory_envelope(dataset_inventory),
        "journal_name": PUBLICATION_JOURNAL_NAME,
        "journal_sha256": _bytes_sha256(journal_bytes),
        "journal_byte_count": len(journal_bytes),
        "journal_identity": journal_value["journal_identity"],
        "commit_semantics": PUBLICATION_COMMIT_SEMANTICS,
        "pair_authority": False,
        "paths_exposed": False,
    }
    marker = {**base, "marker_identity": stable_hash(base)}
    return validate_dataset_commit(
        marker,
        journal=journal_value,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )


def validate_dataset_commit(
    marker: Mapping[str, Any],
    *,
    journal: Mapping[str, Any],
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the dataset marker and its exact journal/inventory binding."""

    journal_value = validate_publication_journal(
        journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    value = _exact_mapping(marker, _DATASET_COMMIT_KEYS, "dataset commit marker")
    identity = str(journal_value["publication_identity_sha256"])
    journal_bytes = canonical_publication_commit_bytes(journal_value)
    if (
        value.get("schema_version") != DATASET_COMMIT_SCHEMA
        or value.get("status") != DATASET_COMMIT_STATUS
        or value.get("kind") != DATASET_COMMIT_KIND
        or value.get("publication_identity_sha256") != identity
        or value.get("directory_relative_path") != _dataset_relative_path(identity)
        or value.get("commit_relative_path") != _dataset_commit_relative_path(identity)
        or value.get("journal_name") != PUBLICATION_JOURNAL_NAME
        or value.get("journal_sha256") != _bytes_sha256(journal_bytes)
        or type(value.get("journal_byte_count")) is not int
        or value.get("journal_byte_count") != len(journal_bytes)
        or value.get("journal_identity") != journal_value["journal_identity"]
        or value.get("commit_semantics") != PUBLICATION_COMMIT_SEMANTICS
        or value.get("pair_authority") is not False
        or value.get("paths_exposed") is not False
    ):
        raise PublicationCommitError("Dataset commit marker bindings are invalid.")
    _validate_inventory_envelope(value.get("inventory"), dataset_inventory, "dataset")
    _self_identity(value, "marker_identity", "dataset commit marker")
    return value


def build_campaign_commit(
    *,
    journal: Mapping[str, Any],
    dataset_commit: Mapping[str, Any],
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the final marker that authorizes the exact publication pair."""

    journal_value = validate_publication_journal(
        journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    dataset_value = validate_dataset_commit(
        dataset_commit,
        journal=journal_value,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    identity = str(journal_value["publication_identity_sha256"])
    journal_bytes = canonical_publication_commit_bytes(journal_value)
    dataset_bytes = canonical_publication_commit_bytes(dataset_value)
    base = {
        "schema_version": CAMPAIGN_COMMIT_SCHEMA,
        "status": CAMPAIGN_COMMIT_STATUS,
        "kind": CAMPAIGN_COMMIT_KIND,
        "publication_identity_sha256": identity,
        "directory_relative_path": _campaign_relative_path(identity),
        "commit_relative_path": _campaign_commit_relative_path(identity),
        "inventory": _inventory_envelope(campaign_inventory),
        "dataset_relative_path": _dataset_relative_path(identity),
        "dataset_inventory": _inventory_envelope(dataset_inventory),
        "dataset_commit_relative_path": _dataset_commit_relative_path(identity),
        "dataset_marker_sha256": _bytes_sha256(dataset_bytes),
        "dataset_marker_byte_count": len(dataset_bytes),
        "dataset_marker_identity": dataset_value["marker_identity"],
        "journal_name": PUBLICATION_JOURNAL_NAME,
        "journal_sha256": _bytes_sha256(journal_bytes),
        "journal_byte_count": len(journal_bytes),
        "journal_identity": journal_value["journal_identity"],
        "commit_semantics": PUBLICATION_COMMIT_SEMANTICS,
        "pair_authority": True,
        "paths_exposed": False,
    }
    marker = {**base, "marker_identity": stable_hash(base)}
    return validate_campaign_commit(
        marker,
        journal=journal_value,
        dataset_commit=dataset_value,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )


def validate_campaign_commit(
    marker: Mapping[str, Any],
    *,
    journal: Mapping[str, Any],
    dataset_commit: Mapping[str, Any],
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the sole authority marker for both exact directories."""

    journal_value = validate_publication_journal(
        journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    dataset_value = validate_dataset_commit(
        dataset_commit,
        journal=journal_value,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    value = _exact_mapping(marker, _CAMPAIGN_COMMIT_KEYS, "campaign commit marker")
    identity = str(journal_value["publication_identity_sha256"])
    journal_bytes = canonical_publication_commit_bytes(journal_value)
    dataset_bytes = canonical_publication_commit_bytes(dataset_value)
    if (
        value.get("schema_version") != CAMPAIGN_COMMIT_SCHEMA
        or value.get("status") != CAMPAIGN_COMMIT_STATUS
        or value.get("kind") != CAMPAIGN_COMMIT_KIND
        or value.get("publication_identity_sha256") != identity
        or value.get("directory_relative_path") != _campaign_relative_path(identity)
        or value.get("commit_relative_path") != _campaign_commit_relative_path(identity)
        or value.get("dataset_relative_path") != _dataset_relative_path(identity)
        or value.get("dataset_commit_relative_path") != _dataset_commit_relative_path(identity)
        or value.get("dataset_marker_sha256") != _bytes_sha256(dataset_bytes)
        or type(value.get("dataset_marker_byte_count")) is not int
        or value.get("dataset_marker_byte_count") != len(dataset_bytes)
        or value.get("dataset_marker_identity") != dataset_value["marker_identity"]
        or value.get("journal_name") != PUBLICATION_JOURNAL_NAME
        or value.get("journal_sha256") != _bytes_sha256(journal_bytes)
        or type(value.get("journal_byte_count")) is not int
        or value.get("journal_byte_count") != len(journal_bytes)
        or value.get("journal_identity") != journal_value["journal_identity"]
        or value.get("commit_semantics") != PUBLICATION_COMMIT_SEMANTICS
        or value.get("pair_authority") is not True
        or value.get("paths_exposed") is not False
    ):
        raise PublicationCommitError("Campaign pair-authority marker bindings are invalid.")
    _validate_inventory_envelope(value.get("inventory"), campaign_inventory, "campaign")
    _validate_inventory_envelope(value.get("dataset_inventory"), dataset_inventory, "dataset")
    _self_identity(value, "marker_identity", "campaign commit marker")
    return value


def canonical_publication_commit_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 document bytes for a journal or marker."""

    if not isinstance(value, Mapping):
        raise PublicationCommitError("Publication commit evidence is not a mapping.")
    try:
        return (json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationCommitError("Publication commit evidence is not canonical JSON.") from exc


def _inventory_envelope(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    files = _normalize_inventory(inventory)
    base = {
        "schema_version": PUBLICATION_INVENTORY_SCHEMA,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(record["byte_count"] for record in files.values()),
    }
    return {**base, "inventory_sha256": stable_hash(base)}


def _validate_inventory_envelope(
    value: Any,
    exact_inventory: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    envelope = _exact_mapping(value, _INVENTORY_KEYS, f"{label} inventory")
    expected = _inventory_envelope(exact_inventory)
    files = _normalize_inventory(envelope.get("files"))
    if (
        envelope.get("schema_version") != PUBLICATION_INVENTORY_SCHEMA
        or files != expected["files"]
        or type(envelope.get("file_count")) is not int
        or envelope.get("file_count") != expected["file_count"]
        or type(envelope.get("total_bytes")) is not int
        or envelope.get("total_bytes") != expected["total_bytes"]
        or _digest(envelope.get("inventory_sha256"), f"{label} inventory identity") != expected["inventory_sha256"]
    ):
        raise PublicationCommitError(f"Publication {label} inventory differs from the exact directory bytes.")
    return expected


def _normalize_inventory(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise PublicationCommitError("Publication inventory is empty or malformed.")
    normalized: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    for raw_path, raw_record in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_path, str):
            raise PublicationCommitError("Publication inventory path is malformed.")
        path = raw_path
        try:
            canonical_portable_relative_path(path)
            collision = portable_path_collision_key(path)
        except ValueError as exc:
            raise PublicationCommitError("Publication inventory path is not canonical and relative.") from exc
        if collision in collision_keys:
            raise PublicationCommitError("Publication inventory has a case or Unicode path collision.")
        collision_keys.add(collision)
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"sha256", "byte_count"}:
            raise PublicationCommitError("Publication inventory record has an invalid exact schema.")
        digest = _digest(raw_record.get("sha256"), "inventory file SHA-256")
        byte_count = raw_record.get("byte_count")
        if type(byte_count) is not int or byte_count < 0:
            raise PublicationCommitError("Publication inventory byte count is invalid.")
        normalized[path] = {"sha256": digest, "byte_count": byte_count}
    return normalized


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PublicationCommitError(f"{label.capitalize()} has an invalid exact schema.")
    return dict(value)


def _self_identity(value: Mapping[str, Any], field: str, label: str) -> str:
    identity = _digest(value.get(field), f"{label} identity")
    payload = dict(value)
    payload.pop(field, None)
    if stable_hash(payload) != identity:
        raise PublicationCommitError(f"{label.capitalize()} self-identity is invalid.")
    return identity


def _publication_identity(value: Any) -> str:
    return _digest(value, "publication identity")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicationCommitError(f"Publication {label} is not a lowercase SHA-256.")
    return value


def _dataset_relative_path(identity: str) -> str:
    return f"datasets/conditioned-v5-{identity}"


def _campaign_relative_path(identity: str) -> str:
    return f"campaigns/conditioned-v5-{identity}"


def _dataset_commit_relative_path(identity: str) -> str:
    return f"datasets/{dataset_commit_name(identity)}"


def _campaign_commit_relative_path(identity: str) -> str:
    return f"campaigns/{campaign_commit_name(identity)}"


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "CAMPAIGN_COMMIT_KIND",
    "CAMPAIGN_COMMIT_SCHEMA",
    "CAMPAIGN_COMMIT_STATUS",
    "DATASET_COMMIT_KIND",
    "DATASET_COMMIT_SCHEMA",
    "DATASET_COMMIT_STATUS",
    "PUBLICATION_COMMIT_SEMANTICS",
    "PUBLICATION_INVENTORY_SCHEMA",
    "PUBLICATION_JOURNAL_KIND",
    "PUBLICATION_JOURNAL_NAME",
    "PUBLICATION_JOURNAL_SCHEMA",
    "PUBLICATION_JOURNAL_STATUS",
    "PublicationCommitError",
    "build_campaign_commit",
    "build_dataset_commit",
    "build_publication_journal",
    "campaign_commit_name",
    "canonical_publication_commit_bytes",
    "dataset_commit_name",
    "validate_campaign_commit",
    "validate_dataset_commit",
    "validate_publication_journal",
]

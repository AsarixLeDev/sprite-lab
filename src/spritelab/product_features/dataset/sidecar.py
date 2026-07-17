"""Project-side pack metadata sidecars; the imported folder is never written."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import yaml

from spritelab.dataset_v5.identity import canonical_json_bytes
from spritelab.dataset_v5.raw_inventory import file_sha256
from spritelab.harvest.sources import is_license_allowed_for_training
from spritelab.product_features.dataset.evidence import _recognize_license
from spritelab.product_features.dataset.packs import (
    SOURCE_PRESETS,
    SourcePack,
    detect_packs,
    pack_id_for_relative_root,
)
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    remove_confined_tree,
)

PACK_METADATA_SCHEMA = "spritelab.dataset.pack_metadata.v2"
PACK_METADATA_BATCH_SCHEMA = "spritelab.dataset.pack_metadata_batch.v2"
PACK_GROUPING_SCHEMA = "spritelab.dataset.pack_grouping.v1"
_METADATA_TRANSACTION_SCHEMA = "spritelab.dataset.metadata_transaction.v1"
_METADATA_TRANSACTION_MARKER_SCHEMA = "spritelab.dataset.metadata_transaction_marker.v1"
_METADATA_LOCK_TIMEOUT_SECONDS = 10.0
LICENSE_CHOICES = (
    "cc0",
    "public_domain",
    "cc_by",
    "cc_by_sa",
    "mit",
    "apache_2",
    "bsd",
    "wtfpl",
    "custom",
    "private_permission",
    "unknown",
)
_ATTRIBUTION_LICENSES = frozenset({"cc_by", "cc_by_sa"})
_DECLARED_PERMISSION_LICENSES = frozenset({"custom", "private_permission"})
_URL_FIELDS = ("source_page_url", "license_url", "direct_download_url")
_TEXT_FIELDS = (
    "creator_or_rights_holder",
    "pack_title",
    "source_type",
    "source_page_url",
    "license_identifier",
    "license_url",
    "license_evidence_file",
    "attribution_text",
    "direct_download_url",
    "version",
    "acquisition_date",
    "notes",
)
OWNERSHIP_NOTE = (
    "Sprite Lab records your declaration exactly as entered. It cannot verify ownership or give legal advice."
)
_METADATA_TRANSACTION_LOCK = threading.RLock()


class PackMetadataError(ValueError):
    """A pack metadata record is incomplete or does not bind to the folder."""


def metadata_store_root(project_root: Path) -> Path:
    return project_root / "datasets" / "source_metadata"


def grouping_path(project_root: Path, input_root: Path) -> Path:
    digest = hashlib.sha256(os.path.normcase(str(input_root.resolve())).encode("utf-8")).hexdigest()[:24]
    return metadata_store_root(project_root) / f"grouping_{digest}.json"


def load_grouping(project_root: Path, input_root: Path) -> dict[str, Any]:
    store = metadata_store_root(project_root)
    try:
        store.lstat()
    except FileNotFoundError:
        return {}
    with _metadata_store_guard(store, create=False):
        return _load_grouping_unlocked(project_root, input_root)


def _load_grouping_unlocked(project_root: Path, input_root: Path) -> dict[str, Any]:
    store = metadata_store_root(project_root)
    path = grouping_path(project_root, input_root)
    try:
        path.lstat()
    except FileNotFoundError:
        return {}
    value = _read_confined_json_file(path, store, label="Pack grouping")
    expected_input = str(input_root.resolve())
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != PACK_GROUPING_SCHEMA
        or not isinstance(value.get("input_root"), str)
        or os.path.normcase(str(value["input_root"])) != os.path.normcase(expected_input)
    ):
        raise PackMetadataError("Pack grouping metadata is corrupt or bound to another source folder.")
    roots = value.get("confirmed_pack_roots")
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise PackMetadataError("Pack grouping metadata contains invalid confirmed pack roots.")
    if roots != _normalize_grouping_roots(roots):
        raise PackMetadataError("Pack grouping metadata contains noncanonical or duplicate confirmed roots.")
    return dict(value)


def save_grouping(project_root: Path, input_root: Path, confirmed_pack_roots: Sequence[str]) -> Path:
    _ensure_project_store_outside_input(project_root, input_root)
    path = grouping_path(project_root, input_root)
    _write_json_transaction(((path, _grouping_record(input_root, confirmed_pack_roots)),))
    return path


def merge_grouping_roots(project_root: Path, input_root: Path, added_pack_roots: Sequence[str]) -> Path:
    """Atomically union confirmed roots so concurrent wizard actions cannot lose an update."""

    _ensure_project_store_outside_input(project_root, input_root)
    store = Path(os.path.abspath(metadata_store_root(project_root)))
    path = grouping_path(project_root, input_root)
    with _metadata_store_guard(store, create=True):
        current = _load_grouping_unlocked(project_root, input_root)
        roots = {str(value) for value in current.get("confirmed_pack_roots", ())}
        roots.update(str(value) for value in added_pack_roots)
        _write_json_transaction_locked(store, ((path, _grouping_record(input_root, sorted(roots))),))
    return path


def _grouping_record(input_root: Path, confirmed_pack_roots: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": PACK_GROUPING_SCHEMA,
        "input_root": str(input_root.resolve()),
        "confirmed_pack_roots": _normalize_grouping_roots(confirmed_pack_roots),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_grouping_roots(confirmed_pack_roots: Sequence[str]) -> list[str]:
    return sorted({str(value).replace("\\", "/").strip("/") or "." for value in confirmed_pack_roots})


def sidecar_path(project_root: Path, pack_id: str) -> Path:
    if not pack_id.startswith("pack_") or not pack_id[5:].isalnum():
        raise PackMetadataError(f"Invalid pack identity: {pack_id!r}")
    return metadata_store_root(project_root) / f"{pack_id}.json"


def load_pack_metadata(project_root: Path) -> dict[str, dict[str, Any]]:
    store = metadata_store_root(project_root)
    try:
        store.lstat()
    except FileNotFoundError:
        return {}
    with _metadata_store_guard(store, create=False):
        return _load_pack_metadata_unlocked(project_root)


def _load_pack_metadata_unlocked(project_root: Path) -> dict[str, dict[str, Any]]:
    store = metadata_store_root(project_root)
    records: dict[str, dict[str, Any]] = {}
    candidates = sorted(
        path for path in store.iterdir() if path.name.startswith("pack_") and path.name.endswith(".json")
    )
    for path in candidates:
        value = _read_confined_json_file(path, store, label="Pack metadata")
        if not isinstance(value, Mapping) or value.get("schema_version") != PACK_METADATA_SCHEMA:
            raise PackMetadataError(f"Pack metadata file {path.name!r} has an unsupported schema.")
        binding = value.get("binding")
        identity = binding.get("pack_identity") if isinstance(binding, Mapping) else None
        if not isinstance(identity, str):
            raise PackMetadataError(f"Pack metadata file {path.name!r} has no valid pack identity.")
        expected_path = sidecar_path(project_root, identity)
        if path.name != expected_path.name:
            raise PackMetadataError(
                f"Pack metadata file {path.name!r} does not match its bound pack identity {identity!r}."
            )
        if identity in records:
            raise PackMetadataError(f"Duplicate metadata records exist for pack identity {identity!r}.")
        records[identity] = dict(value)
    return records


def load_metadata_snapshot(
    project_root: Path,
    input_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read grouping and sidecars from one recovered transaction generation."""

    store = metadata_store_root(project_root)
    try:
        store.lstat()
    except FileNotFoundError:
        return {}, {}
    with _metadata_store_guard(store, create=False):
        return _load_grouping_unlocked(project_root, input_root), _load_pack_metadata_unlocked(project_root)


def validate_pack_metadata(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the user's declaration; unknown licenses stay quarantined."""

    record = {str(key): value for key, value in fields.items()}
    problems: list[str] = []
    normalized_text: dict[str, str] = {}

    def text(name: str) -> str:
        if name in normalized_text:
            return normalized_text[name]
        value = record.get(name)
        if value is None:
            normalized_text[name] = ""
        elif not isinstance(value, str):
            problems.append(f"{name} must be a JSON string when provided.")
            normalized_text[name] = ""
        else:
            normalized_text[name] = value.strip()
        return normalized_text[name]

    def boolean(name: str) -> bool:
        value = record.get(name)
        if value is None:
            return False
        if not isinstance(value, bool):
            problems.append(f"{name} must be a JSON boolean (true or false).")
            return False
        return value

    for name in _TEXT_FIELDS:
        text(name)

    if not text("creator_or_rights_holder"):
        problems.append("creator_or_rights_holder is required.")
    if not text("pack_title"):
        problems.append("pack_title is required.")
    source_type = text("source_type")
    if source_type not in SOURCE_PRESETS:
        problems.append(f"source_type must be one of: {', '.join(SOURCE_PRESETS)}.")
    original_declaration = boolean("original_work_declaration")
    if source_type == "my_original_work":
        if not original_declaration:
            problems.append(
                "My original work requires an explicit original_work_declaration confirming you hold the rights."
            )
    else:
        if original_declaration:
            problems.append("original_work_declaration is valid only when source_type is my_original_work.")
        if not text("source_page_url"):
            problems.append("source_page_url is required for downloaded or external sources.")
    license_identifier = text("license_identifier").casefold()
    if license_identifier not in LICENSE_CHOICES:
        problems.append(f"license_identifier must be one of: {', '.join(LICENSE_CHOICES)}.")
    if license_identifier not in {"unknown"} and not (
        text("license_url")
        or text("license_evidence_file")
        or (source_type == "my_original_work" and original_declaration)
    ):
        problems.append("Provide license_url or license_evidence_file for the declared license.")
    if license_identifier in _ATTRIBUTION_LICENSES and not text("attribution_text"):
        problems.append("attribution_text is required for attribution licenses (CC BY / CC BY-SA).")
    permission_confirmed = boolean("permission_confirmed")
    if license_identifier in _DECLARED_PERMISSION_LICENSES and not permission_confirmed:
        problems.append(
            "Custom or private licenses require permission_confirmed acknowledging you have usage permission."
        )
    if source_type == "my_original_work" and license_identifier == "unknown":
        problems.append("Choose the license or usage policy to record for your original work.")
    for name in _URL_FIELDS:
        value = text(name)
        if value and not _valid_http_url(value):
            problems.append(f"{name} must be a valid HTTP(S) URL when provided.")
    if problems:
        raise PackMetadataError(" ".join(problems))
    normalized = {
        "creator_or_rights_holder": text("creator_or_rights_holder"),
        "pack_title": text("pack_title"),
        "source_type": source_type,
        "source_page_url": text("source_page_url") or None,
        "original_work_declaration": original_declaration,
        "license_identifier": license_identifier,
        "license_url": text("license_url") or None,
        "license_evidence_file": text("license_evidence_file") or None,
        "attribution_text": text("attribution_text") or None,
        "permission_confirmed": permission_confirmed,
        "direct_download_url": text("direct_download_url") or None,
        "version": text("version") or None,
        "acquisition_date": text("acquisition_date") or None,
        "notes": text("notes") or None,
        "ownership_note": OWNERSHIP_NOTE,
    }
    return normalized


def save_pack_metadata(
    project_root: Path,
    input_root: Path,
    pack: SourcePack,
    fields: Mapping[str, Any],
    *,
    covered_byte_hashes: Sequence[str] = (),
) -> dict[str, Any]:
    _ensure_project_store_outside_input(project_root, input_root)
    current_packs = detect_packs(
        input_root,
        _discover_source_pngs(input_root),
        user_grouping=load_grouping(project_root, input_root),
    )
    current = next((candidate for candidate in current_packs if candidate.pack_id == pack.pack_id), None)
    if current is None or canonical_json_bytes(current.to_dict()) != canonical_json_bytes(pack.to_dict()):
        raise PackMetadataError("Source pack membership changed; inspect the selected source again.")
    pack = current
    record = _prepare_pack_metadata_record(input_root, pack, fields, covered_byte_hashes=covered_byte_hashes)
    store = Path(os.path.abspath(metadata_store_root(project_root)))
    with _metadata_store_guard(store, create=True):
        locked_packs = detect_packs(
            input_root,
            _discover_source_pngs(input_root),
            user_grouping=_load_grouping_unlocked(project_root, input_root),
        )
        locked_pack = next((candidate for candidate in locked_packs if candidate.pack_id == pack.pack_id), None)
        if locked_pack is None or canonical_json_bytes(locked_pack.to_dict()) != canonical_json_bytes(pack.to_dict()):
            raise PackMetadataError("Source pack grouping or membership changed; inspect the selected source again.")
        pack = locked_pack
        record = _prepare_pack_metadata_record(
            input_root,
            pack,
            fields,
            covered_byte_hashes=covered_byte_hashes,
        )
        expected_binding = pack_source_binding(input_root, pack)

        def validate_current_source() -> None:
            latest_packs = detect_packs(
                input_root,
                _discover_source_pngs(input_root),
                user_grouping=_load_grouping_unlocked(project_root, input_root),
            )
            latest = next((candidate for candidate in latest_packs if candidate.pack_id == pack.pack_id), None)
            if latest is None or canonical_json_bytes(pack_source_binding(input_root, latest)) != canonical_json_bytes(
                expected_binding
            ):
                raise PackMetadataError(
                    "Source pack grouping, membership, or identities changed during metadata validation."
                )

        _write_json_transaction_locked(
            store,
            ((sidecar_path(project_root, pack.pack_id), record),),
            validate_before_commit=validate_current_source,
        )
    return record


def _prepare_pack_metadata_record(
    input_root: Path,
    pack: SourcePack,
    fields: Mapping[str, Any],
    *,
    covered_byte_hashes: Sequence[str] = (),
) -> dict[str, Any]:
    normalized = validate_pack_metadata(fields)
    pack_root = (input_root / pack.relative_root).resolve() if pack.relative_root != "." else input_root.resolve()
    evidence_file = normalized.get("license_evidence_file")
    if evidence_file:
        if not _license_evidence_file_allowed(input_root, pack_root, pack, str(evidence_file)):
            raise PackMetadataError(
                "license_evidence_file must name an existing file inside this pack "
                "or inherited evidence detected for it."
            )
    binding = pack_source_binding(input_root, pack)
    covered_identities = dict(binding["covered_file_identities"])
    if covered_byte_hashes and sorted(covered_byte_hashes) != sorted(covered_identities.values()):
        raise PackMetadataError("Covered file identities changed while metadata was being saved; inspect again.")
    binding["declaration_identity_sha256"] = _declaration_identity(normalized)
    record = {
        "schema_version": PACK_METADATA_SCHEMA,
        **normalized,
        "binding": binding,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "input_folder_written": False,
    }
    return record


def sidecar_is_applicable(record: Mapping[str, Any], pack: SourcePack, input_root: Path) -> bool:
    """Changed source/license evidence invalidates only the affected pack."""

    binding = record.get("binding")
    if not isinstance(binding, Mapping):
        return False
    try:
        normalized = validate_pack_metadata(record)
        current_binding = pack_source_binding(input_root, pack)
    except (OSError, PackMetadataError):
        return False
    if binding.get("declaration_identity_sha256") != _declaration_identity(normalized):
        return False
    bound_source = {str(key): value for key, value in binding.items() if key != "declaration_identity_sha256"}
    return canonical_json_bytes(bound_source) == canonical_json_bytes(current_binding)


def effective_pack_evidence(
    record: Mapping[str, Any],
    pack: SourcePack,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate an applicable sidecar declaration into backend evidence records."""

    record_hash = hashlib.sha256(canonical_json_bytes(dict(record))).hexdigest()
    source = {
        "present": True,
        "path": f"sidecar:{pack.pack_id}",
        "evidence_sha256": record_hash,
        "source_name": record.get("pack_title"),
        "creator": record.get("creator_or_rights_holder"),
        "source_url": record.get("source_page_url"),
        "source_type": record.get("source_type"),
        "notes": str(record.get("notes") or "")[:4000],
        "interpretation": "user_declaration_sidecar",
    }
    identifier = str(record.get("license_identifier") or "unknown").casefold()
    normalized = _recognize_license(identifier, public_domain=identifier == "public_domain")
    if normalized == "unknown" and identifier in LICENSE_CHOICES:
        normalized = identifier
    if identifier == "unknown":
        training_allowed = False
    elif identifier in _DECLARED_PERMISSION_LICENSES:
        training_allowed = bool(record.get("permission_confirmed"))
    elif record.get("source_type") == "my_original_work" and bool(record.get("original_work_declaration")):
        training_allowed = True
    else:
        training_allowed = is_license_allowed_for_training(normalized)
    license_record = {
        "present": True,
        "path": f"sidecar:{pack.pack_id}",
        "evidence_sha256": record_hash,
        "license": normalized,
        "license_url": record.get("license_url"),
        "training_allowed": training_allowed,
        "attribution_text": record.get("attribution_text"),
        "interpretation": "user_declaration_sidecar",
    }
    return source, license_record


def export_metadata_files(record: Mapping[str, Any], pack_root: Path) -> dict[str, Any]:
    """Explicit user action: write metadata files without overwriting anything."""

    written: list[str] = []
    skipped: list[str] = []
    source_target = pack_root / "source.yaml"
    license_target = pack_root / "LICENSE.txt"
    source_payload = {
        "schema_version": PACK_METADATA_SCHEMA,
        "creator": record.get("creator_or_rights_holder"),
        "name": record.get("pack_title"),
        "source_type": record.get("source_type"),
        "source_url": record.get("source_page_url"),
        "download_url": record.get("direct_download_url"),
        "version": record.get("version"),
        "acquisition_date": record.get("acquisition_date"),
        "notes": record.get("notes"),
    }
    if source_target.exists():
        skipped.append(source_target.name)
    else:
        source_target.write_text(
            yaml.safe_dump({k: v for k, v in source_payload.items() if v is not None}, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        written.append(source_target.name)
    if license_target.exists():
        skipped.append(license_target.name)
    else:
        lines = [f"license: {record.get('license_identifier')}"]
        if record.get("license_url"):
            lines.append(f"license_url: {record.get('license_url')}")
        if record.get("attribution_text"):
            lines.append(f"attribution: {record.get('attribution_text')}")
        lines.append(str(OWNERSHIP_NOTE))
        license_target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        written.append(license_target.name)
    return {"written": written, "skipped_existing": skipped, "explicit_user_action": True}


def apply_metadata_file(
    project_root: Path,
    input_root: Path,
    packs: Sequence[SourcePack],
    metadata_file: Path,
) -> dict[str, Any]:
    """Apply a documented automation batch; every record must validate and bind."""

    _ensure_project_store_outside_input(project_root, input_root)
    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackMetadataError(f"Metadata file is unreadable: {metadata_file} ({exc})") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != PACK_METADATA_BATCH_SCHEMA:
        raise PackMetadataError(f"Metadata file must declare schema_version {PACK_METADATA_BATCH_SCHEMA!r}.")
    expected_root = str(input_root.resolve())
    supplied_root = str(payload.get("canonical_input_root") or "")
    if os.path.normcase(supplied_root) != os.path.normcase(expected_root):
        raise PackMetadataError("Metadata file source binding does not match the selected input root.")
    confirmed_roots = payload.get("confirmed_pack_roots", ())
    if not isinstance(confirmed_roots, Sequence) or isinstance(confirmed_roots, (str, bytes)):
        raise PackMetadataError("confirmed_pack_roots must be an array when provided.")
    if any(not isinstance(value, str) for value in confirmed_roots):
        raise PackMetadataError("confirmed_pack_roots entries must be JSON strings.")
    normalized_roots = _normalize_grouping_roots([str(value) for value in confirmed_roots])
    image_paths = _discover_source_pngs(input_root)
    current_packs = detect_packs(
        input_root,
        image_paths,
        user_grouping={"confirmed_pack_roots": normalized_roots},
    )
    if not current_packs:
        raise PackMetadataError("Metadata batches require at least one current PNG source pack.")
    supplied_pack_ids = {pack.pack_id for pack in packs}
    current_pack_ids = {pack.pack_id for pack in current_packs}
    if supplied_pack_ids != current_pack_ids:
        raise PackMetadataError("Source pack membership changed; inspect the selected source again.")
    packs = current_packs
    expected_pack_bindings = {pack.pack_id: pack_source_binding(input_root, pack) for pack in packs}
    rows = payload.get("packs")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PackMetadataError("Metadata file must contain a packs array.")
    by_identity = {pack.pack_id: pack for pack in packs}
    applied: list[str] = []
    seen_ids: set[str] = set()
    seen_roots: set[str] = set()
    pending: list[tuple[SourcePack, dict[str, Any]]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PackMetadataError(f"packs[{index}] must be an object.")
        key = str(raw.get("pack_relative_root") or "").replace("\\", "/").strip("/") or "."
        pack_id = str(raw.get("pack_id") or "")
        pack = by_identity.get(pack_id)
        if pack is None:
            raise PackMetadataError(f"packs[{index}] references an unknown or changed pack identity: {pack_id!r}")
        if pack.relative_root != key:
            raise PackMetadataError(f"packs[{index}] pack root no longer matches its bound pack identity.")
        if key in seen_roots or pack_id in seen_ids:
            raise PackMetadataError(f"packs[{index}] repeats a pack identity or root: {pack_id!r}")
        seen_roots.add(key)
        seen_ids.add(pack_id)
        supplied_binding = raw.get("source_binding")
        if not isinstance(supplied_binding, Mapping) or canonical_json_bytes(
            dict(supplied_binding)
        ) != canonical_json_bytes(pack_source_binding(input_root, pack)):
            raise PackMetadataError(f"packs[{index}] source binding changed; inspect the selected source again.")
        fields = {
            name: value
            for name, value in raw.items()
            if name not in {"pack_relative_root", "pack_id", "source_binding"}
        }
        validate_pack_metadata(fields)
        evidence_file = str(fields.get("license_evidence_file") or "").strip()
        if evidence_file:
            pack_root = input_root if pack.relative_root == "." else input_root / pack.relative_root
            if not _license_evidence_file_allowed(input_root, pack_root.resolve(), pack, evidence_file):
                raise PackMetadataError(
                    f"packs[{index}].license_evidence_file must name an existing file inside its pack "
                    "or inherited evidence detected for it."
                )
        pending.append((pack, fields))
    if seen_ids != set(by_identity):
        missing = sorted(set(by_identity) - seen_ids)
        raise PackMetadataError(
            "Metadata batch must contain exactly one record for every current source pack; "
            f"missing pack identities: {', '.join(missing)}."
        )
    prepared: list[tuple[SourcePack, dict[str, Any]]] = []
    for pack, fields in pending:
        covered = [file_sha256(input_root / relative) for relative in pack.image_relative_paths]
        prepared.append(
            (
                pack,
                _prepare_pack_metadata_record(input_root, pack, fields, covered_byte_hashes=covered),
            )
        )
    for pack, record in prepared:
        binding = record["binding"]
        bound_source = {key: value for key, value in binding.items() if key != "declaration_identity_sha256"}
        if canonical_json_bytes(bound_source) != canonical_json_bytes(pack_source_binding(input_root, pack)):
            raise PackMetadataError("Source identities changed while the metadata batch was being prepared.")
    writes: list[tuple[Path, Mapping[str, Any]]] = []
    if normalized_roots:
        writes.append(
            (
                grouping_path(project_root, input_root),
                _grouping_record(input_root, normalized_roots),
            )
        )
    writes.extend((sidecar_path(project_root, pack.pack_id), record) for pack, record in prepared)

    def validate_current_source() -> None:
        _validate_current_batch_source(input_root, normalized_roots, expected_pack_bindings)
        current_grouping = _load_grouping_unlocked(project_root, input_root)
        current_roots = _normalize_grouping_roots(current_grouping.get("confirmed_pack_roots", ()))
        if current_roots != normalized_roots:
            raise PackMetadataError("Persisted pack grouping changed while the metadata batch was being applied.")

    _validate_current_batch_source(input_root, normalized_roots, expected_pack_bindings)
    store = Path(os.path.abspath(metadata_store_root(project_root)))
    try:
        with _metadata_store_guard(store, create=True):
            current_grouping = _load_grouping_unlocked(project_root, input_root)
            current_roots = _normalize_grouping_roots(current_grouping.get("confirmed_pack_roots", ()))
            if current_roots != normalized_roots:
                raise PackMetadataError(
                    "Persisted pack grouping changed after this metadata batch was generated; regenerate it."
                )
            _write_json_transaction_locked(
                store,
                writes,
                validate_before_commit=validate_current_source,
            )
    except OSError as exc:
        raise PackMetadataError(f"Metadata batch could not be committed atomically: {exc}") from exc
    for pack, _record in prepared:
        applied.append(pack.pack_id)
    return {"schema_version": PACK_METADATA_BATCH_SCHEMA, "applied_pack_ids": applied}


def _validate_current_batch_source(
    input_root: Path,
    confirmed_roots: Sequence[str],
    expected_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    current_packs = detect_packs(
        input_root,
        _discover_source_pngs(input_root),
        user_grouping={"confirmed_pack_roots": list(confirmed_roots)},
    )
    current_bindings = {pack.pack_id: pack_source_binding(input_root, pack) for pack in current_packs}
    if canonical_json_bytes(current_bindings) != canonical_json_bytes(dict(expected_bindings)):
        raise PackMetadataError("Source pack membership or identities changed during metadata batch validation.")


def metadata_file_template(input_root: Path, packs: Sequence[SourcePack]) -> dict[str, Any]:
    """Return a documented automation skeleton without guessing missing declarations."""

    rows = []
    for pack in packs:
        prefill = dict(pack.prefill)
        rows.append(
            {
                "pack_relative_root": pack.relative_root,
                "pack_id": pack.pack_id,
                "source_binding": pack_source_binding(input_root, pack),
                "creator_or_rights_holder": prefill.get("creator_or_rights_holder", ""),
                "pack_title": prefill.get("pack_title", ""),
                "source_type": prefill.get("source_type", ""),
                "source_page_url": prefill.get("source_page_url", ""),
                "original_work_declaration": False,
                "license_identifier": prefill.get("license_identifier", "unknown"),
                "license_url": prefill.get("license_url", ""),
                "license_evidence_file": prefill.get("license_evidence_file", ""),
                "attribution_text": prefill.get("attribution_text", ""),
                "permission_confirmed": False,
                "direct_download_url": "",
                "version": "",
                "acquisition_date": "",
                "notes": "",
            }
        )
    return {
        "schema_version": PACK_METADATA_BATCH_SCHEMA,
        "canonical_input_root": str(input_root.resolve()),
        "description": "Complete one record per pack. Unknown licenses remain quarantined.",
        "confirmed_pack_roots": sorted(
            pack.relative_root for pack in packs if pack.boundary_evidence == "explicit_user_grouping"
        ),
        "packs": rows,
    }


def _valid_http_url(value: str) -> bool:
    if any(character.isspace() or ord(character) < 32 for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.hostname)


def _declaration_identity(fields: Mapping[str, Any]) -> str:
    normalized = validate_pack_metadata(fields)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _discover_source_pngs(input_root: Path) -> list[Path]:
    root = input_root.resolve()
    if not root.is_dir():
        raise PackMetadataError("The selected source folder is no longer available.")

    def walk_error(exc: OSError) -> None:
        name = Path(str(exc.filename or "source entry")).name
        raise PackMetadataError(f'The selected source contains an unreadable entry: "{name}".') from exc

    discovered: list[Path] = []
    for directory, names, filenames in os.walk(root, onerror=walk_error, followlinks=False):
        directory_path = Path(directory)
        _require_confined_source_entry(directory_path, root, expected="directory")
        names.sort(key=str.casefold)
        for name in names:
            _require_confined_source_entry(directory_path / name, root, expected="directory")
        for filename in sorted(filenames, key=str.casefold):
            candidate = directory_path / filename
            _require_confined_source_entry(candidate, root, expected="file")
            if candidate.suffix.casefold() == ".png":
                discovered.append(candidate)
    return discovered


def _confined_relative_source_path(
    root: Path,
    relative: object,
    *,
    expected: str,
    allow_root: bool = False,
) -> Path:
    if not isinstance(relative, str):
        raise PackMetadataError("Source pack paths must be strings.")
    normalized = relative.replace("\\", "/")
    value = Path(normalized)
    if (
        not normalized
        or value.is_absolute()
        or bool(value.drive)
        or ".." in value.parts
        or (normalized == "." and not allow_root)
    ):
        raise PackMetadataError("Source pack paths must be confined relative paths beneath the selected source.")
    candidate = root if normalized == "." else root.joinpath(*value.parts)
    return _require_confined_source_entry(candidate, root, expected=expected)


def _require_confined_source_entry(path: Path, root: Path, *, expected: str) -> Path:
    try:
        current = path
        target_status = None
        while True:
            status_value = current.lstat()
            if current == path:
                target_status = status_value
            is_reparse = bool(
                getattr(status_value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            is_junction = bool(getattr(current, "is_junction", lambda: False)())
            if current != root and (current.is_symlink() or is_junction or is_reparse):
                raise ValueError("linked source entries are not accepted")
            if current == root:
                break
            current = current.parent
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if target_status is None:
            raise ValueError("missing source entry status")
        if expected == "file" and not stat.S_ISREG(target_status.st_mode):
            raise ValueError("source entry is not a regular file")
        if expected == "directory" and not stat.S_ISDIR(target_status.st_mode):
            raise ValueError("source entry is not a directory")
    except (OSError, RuntimeError, ValueError) as exc:
        try:
            display = path.relative_to(root).as_posix()
        except ValueError:
            display = path.name
        raise PackMetadataError(
            f'The selected source contains an unreadable, linked, or non-regular entry: "{display}".'
        ) from exc
    return resolved


def pack_source_binding(input_root: Path, pack: SourcePack) -> dict[str, Any]:
    root = input_root.resolve()
    pack_root = _confined_relative_source_path(root, pack.relative_root, expected="directory", allow_root=True)
    if pack.pack_id != pack_id_for_relative_root(root, pack.relative_root):
        raise PackMetadataError("Source pack identity does not match its confined relative root.")
    if not pack.image_relative_paths or len(set(pack.image_relative_paths)) != len(pack.image_relative_paths):
        raise PackMetadataError("Source pack image membership must be non-empty and unique.")
    covered_identities: dict[str, str] = {}
    for relative in sorted(pack.image_relative_paths):
        path = _confined_relative_source_path(root, relative, expected="file")
        if path.suffix.casefold() != ".png" or not (path == pack_root or _is_relative_to(path, pack_root)):
            raise PackMetadataError("Source pack image paths must be confined PNG files beneath their pack root.")
        covered_identities[relative] = file_sha256(path)
    evidence_identities: dict[str, dict[str, Any]] = {}
    evidence_rows: list[tuple[str, Mapping[str, Any]]] = []
    for row in pack.evidence_files:
        if not isinstance(row, Mapping):
            raise PackMetadataError("Source pack evidence records must be objects.")
        relative_value = row.get("relative_path") or row.get("name")
        if not isinstance(relative_value, str):
            raise PackMetadataError("Source pack evidence paths must be strings.")
        evidence_rows.append((relative_value, row))
    for relative, row in sorted(evidence_rows, key=lambda value: value[0]):
        path = _confined_relative_source_path(root, relative, expected="file")
        evidence_identities[relative] = {
            "sha256": file_sha256(path),
            "byte_length": path.stat().st_size,
            "role": str(row.get("role") or "supporting"),
        }
    archive_identity = None
    if pack.archive:
        relative_value = pack.archive.get("relative_path")
        if not isinstance(relative_value, str):
            raise PackMetadataError("Source pack archive paths must be strings.")
        archive_path = _confined_relative_source_path(root, relative_value, expected="file")
        archive_identity = {
            "relative_path": relative_value,
            "sha256": file_sha256(archive_path),
            "byte_length": archive_path.stat().st_size,
        }
    return {
        "source_binding_schema": "spritelab.dataset.pack_source_binding.v2",
        "canonical_source_path": str(pack_root),
        "input_root": str(root),
        "pack_relative_root": pack.relative_root,
        "pack_identity": pack.pack_id,
        "pack_boundary_evidence": pack.boundary_evidence,
        "pack_boundary_status": pack.boundary_status,
        "evidence_file_hashes": {relative: identity["sha256"] for relative, identity in evidence_identities.items()},
        "evidence_file_identities": evidence_identities,
        "archive_sha256": archive_identity["sha256"] if archive_identity else None,
        "archive_identity": archive_identity,
        "covered_file_count": len(covered_identities),
        "covered_file_identities": covered_identities,
        "covered_files_digest": hashlib.sha256(canonical_json_bytes(covered_identities)).hexdigest(),
    }


def ensure_dataset_writes_outside_input(
    project_root: Path,
    input_root: Path,
    *,
    output_root: Path | None = None,
    runs_directory: Path | None = None,
) -> None:
    """Reject every automatic project write target that falls inside approved source input."""

    root = input_root.resolve()
    targets = {
        "project-side metadata": metadata_store_root(project_root).resolve(),
        "dataset output": output_root.expanduser().resolve() if output_root is not None else None,
        "durable run state": runs_directory.expanduser().resolve() if runs_directory is not None else None,
    }
    overlaps = [
        name
        for name, path in targets.items()
        if path is not None and (path == root or _is_relative_to(path, root) or _is_relative_to(root, path))
    ]
    if overlaps:
        raise PackMetadataError(
            f"Automatic write targets must stay outside the selected source/input folder ({', '.join(overlaps)}). "
            "Choose an external project, run, and output location before continuing."
        )


def _ensure_project_store_outside_input(project_root: Path, input_root: Path) -> None:
    ensure_dataset_writes_outside_input(project_root, input_root)


def _write_json_transaction(
    entries: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    validate_before_commit: Callable[[], None] | None = None,
) -> None:
    """Commit one recovered generation with a durable transaction-wide decision marker."""

    if not entries:
        return
    store = Path(os.path.abspath(entries[0][0].parent))
    with _metadata_store_guard(store, create=True):
        _write_json_transaction_locked(store, entries, validate_before_commit=validate_before_commit)


def _write_json_transaction_locked(
    store: Path,
    entries: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    validate_before_commit: Callable[[], None] | None = None,
) -> None:
    if not entries:
        return
    normalized: list[tuple[Path, bytes]] = []
    seen: set[str] = set()
    for path, value in entries:
        target = _safe_metadata_target(path, store)
        identity = os.path.normcase(str(target))
        if identity in seen:
            raise OSError("Metadata transaction contains duplicate output paths.")
        seen.add(identity)
        normalized.append((target, _json_document_bytes(value)))

    transaction_id = uuid.uuid4().hex
    transactions = _ensure_transactions_directory(store)
    transaction = transactions / f"txn_{transaction_id}"
    manifest_entries: list[dict[str, Any]] = []
    transaction_created = False
    try:
        with AnchoredDirectory(transactions, transactions) as transactions_anchor:
            owned_transaction = transactions_anchor.mkdir(transaction.name, exist_ok=False)
            transaction_created = True
            with AnchoredDirectory(transaction, transaction) as transaction_anchor:
                if not owned_transaction.matches(transaction_anchor.directory_metadata()):
                    raise OSError("Metadata transaction directory changed while it was being opened.")
                owned_new = transaction_anchor.mkdir("new", exist_ok=False)
                owned_old = transaction_anchor.mkdir("old", exist_ok=False)
                with (
                    AnchoredDirectory(transaction / "new", transaction / "new") as new_anchor,
                    AnchoredDirectory(transaction / "old", transaction / "old") as old_anchor,
                ):
                    if not owned_new.matches(new_anchor.directory_metadata()) or not owned_old.matches(
                        old_anchor.directory_metadata()
                    ):
                        raise OSError("Metadata transaction payload directory changed while it was being opened.")
                    for index, (target, new_bytes) in enumerate(normalized):
                        new_path = transaction / "new" / f"{index:04d}.json"
                        _write_durable_file(new_path, new_bytes)
                        try:
                            target.lstat()
                        except FileNotFoundError:
                            old_bytes = None
                        else:
                            old_bytes = _read_confined_bytes(target, store, label="Metadata transaction target")
                        old_path = transaction / "old" / f"{index:04d}.json"
                        if old_bytes is not None:
                            _write_durable_file(old_path, old_bytes)
                        manifest_entries.append(
                            {
                                "index": index,
                                "target": target.name,
                                "had_original": old_bytes is not None,
                                "old_sha256": hashlib.sha256(old_bytes).hexdigest() if old_bytes is not None else None,
                                "new_sha256": hashlib.sha256(new_bytes).hexdigest(),
                                "new_payload": f"new/{index:04d}.json",
                                "old_payload": f"old/{index:04d}.json" if old_bytes is not None else None,
                            }
                        )
                    manifest = {
                        "schema_version": _METADATA_TRANSACTION_SCHEMA,
                        "transaction_id": transaction_id,
                        "entries": manifest_entries,
                    }
                    manifest_bytes = _json_document_bytes(manifest)
                    _write_durable_file(transaction / "manifest.json", manifest_bytes)
                    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                    _write_transaction_marker(transaction / "PREPARED", transaction_id, manifest_sha256)
                    for row in manifest_entries:
                        expected_old = row["old_sha256"] if row["had_original"] else None
                        if _path_sha256(store / str(row["target"])) != expected_old:
                            raise OSError("Metadata transaction target changed after preparation.")
                        _install_transaction_payload(
                            transaction / str(row["new_payload"]),
                            store / str(row["target"]),
                            transaction_id=transaction_id,
                            index=int(row["index"]),
                        )
                    for row in manifest_entries:
                        if _path_sha256(store / str(row["target"])) != row["new_sha256"]:
                            raise OSError("Metadata transaction target verification failed before commit.")
                    if validate_before_commit is not None:
                        validate_before_commit()
                    for row in manifest_entries:
                        if _path_sha256(store / str(row["target"])) != row["new_sha256"]:
                            raise OSError("Metadata transaction target changed during final source validation.")
                    _write_transaction_marker(transaction / "COMMITTED", transaction_id, manifest_sha256)
    except Exception as exc:
        if not transaction_created:
            raise
        try:
            _recover_transaction_locked(store, transaction)
        except OSError as recovery_exc:
            raise OSError(
                f"Metadata transaction failed and transaction-wide recovery was incomplete: {recovery_exc}"
            ) from exc
        raise
    try:
        _cleanup_transaction_directory(transaction)
    except OSError:
        # COMMITTED is durable. A later reader/writer rolls forward and cleans up.
        pass


@contextmanager
def _metadata_store_guard(store: Path, *, create: bool) -> Iterator[None]:
    with _METADATA_TRANSACTION_LOCK:
        store = Path(os.path.abspath(store))
        if create:
            _ensure_metadata_store_directory(store)
        else:
            try:
                store.lstat()
            except FileNotFoundError:
                yield
                return
        _require_safe_directory(store.parent, parent=store.parent.parent)
        _require_safe_directory(store, parent=store.parent)
        with AnchoredDirectory(store, store):
            lock_path = store / ".metadata.lock"
            with _interprocess_metadata_lock(lock_path):
                _recover_transactions_locked(store)
                yield


@contextmanager
def _interprocess_metadata_lock(lock_path: Path) -> Iterator[None]:
    _require_safe_directory(lock_path.parent)
    try:
        lock_path.lstat()
    except FileNotFoundError:
        pass
    else:
        _require_safe_regular_file(lock_path, parent=lock_path.parent, label="Metadata lock")
    handle = lock_path.open("a+b")
    try:
        opened_status = os.fstat(handle.fileno())
        _require_safe_regular_file(lock_path, parent=lock_path.parent, label="Metadata lock")
        current_status = lock_path.lstat()
        if (opened_status.st_dev, opened_status.st_ino) != (current_status.st_dev, current_status.st_ino):
            raise OSError("Metadata lock changed while it was being opened.")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + _METADATA_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                _lock_file(handle)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise OSError("The metadata store is busy in another Sprite Lab process; try again.") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            _unlock_file(handle)
    finally:
        handle.close()


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_transactions_directory(store: Path) -> Path:
    transactions = store / ".transactions"
    if transactions.exists() or transactions.is_symlink():
        _require_safe_directory(transactions, parent=store)
        return transactions
    try:
        transactions.mkdir()
    except FileExistsError:
        pass
    _require_safe_directory(transactions, parent=store)
    _fsync_directory(store)
    return transactions


def _recover_transactions_locked(store: Path) -> None:
    _require_safe_directory(store)
    for abandoned in sorted(path for path in store.iterdir() if path.name.startswith(".transactions.creating.")):
        _require_safe_directory(abandoned, parent=store)
        _cleanup_transaction_directory(abandoned)
    transactions = store / ".transactions"
    try:
        transactions.lstat()
    except FileNotFoundError:
        return
    _require_safe_directory(transactions, parent=store)
    with AnchoredDirectory(transactions, transactions):
        children = sorted(transactions.iterdir())
        for preparing in (path for path in children if path.name.startswith("preparing_")):
            _require_safe_directory(preparing, parent=transactions)
            _cleanup_transaction_directory(preparing)
        children = sorted(transactions.iterdir())
        for stale in (path for path in children if path.name.startswith("cleanup_")):
            _require_safe_directory(stale, parent=transactions)
            try:
                _cleanup_transaction_directory(stale)
            except OSError:
                pass
        children = sorted(transactions.iterdir())
        for transaction in (path for path in children if path.name.startswith("txn_")):
            _require_safe_directory(transaction, parent=transactions)
            _recover_transaction_locked(store, transaction)


def _recover_transaction_locked(store: Path, transaction: Path) -> None:
    _require_safe_directory(store)
    _require_safe_directory(store / ".transactions", parent=store)
    _require_safe_directory(transaction, parent=store / ".transactions")
    identity = OwnedFileIdentity.from_stat(transaction.lstat())
    with AnchoredDirectory(transaction, transaction) as transaction_anchor:
        if not identity.matches(transaction_anchor.directory_metadata()):
            raise OSError("Metadata transaction directory changed while it was being opened.")
        _recover_transaction_contents_locked(store, transaction)
    _cleanup_transaction_directory(transaction)


def _recover_transaction_contents_locked(store: Path, transaction: Path) -> None:
    prepared_path = transaction / "PREPARED"
    try:
        prepared_path.lstat()
    except FileNotFoundError:
        return
    _require_safe_regular_file(prepared_path, parent=transaction, label="PREPARED marker")
    manifest_path = transaction / "manifest.json"
    try:
        manifest_bytes = _read_confined_bytes(manifest_path, transaction, label="Transaction manifest")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSError("Prepared metadata transaction manifest is unreadable.") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _METADATA_TRANSACTION_SCHEMA:
        raise OSError("Prepared metadata transaction manifest has an unsupported schema.")
    transaction_id = str(manifest.get("transaction_id") or "")
    if transaction.name != f"txn_{transaction_id}":
        raise OSError("Prepared metadata transaction identity does not match its directory.")
    rows = manifest.get("entries")
    if not isinstance(rows, list) or not rows:
        raise OSError("Prepared metadata transaction has no bound entries.")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    _validate_transaction_marker(prepared_path, transaction_id, manifest_sha256)
    committed_path = transaction / "COMMITTED"
    try:
        committed_path.lstat()
    except FileNotFoundError:
        committed = False
    else:
        _require_safe_regular_file(committed_path, parent=transaction, label="COMMITTED marker")
        committed = True
    if committed:
        _validate_transaction_marker(committed_path, transaction_id, manifest_sha256)
    seen_indexes: set[int] = set()
    seen_targets: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise OSError("Prepared metadata transaction contains an invalid entry.")
        index = raw.get("index")
        target = raw.get("target")
        if not isinstance(index, int) or index in seen_indexes or not isinstance(target, str) or target in seen_targets:
            raise OSError("Prepared metadata transaction contains duplicate or invalid entry bindings.")
        seen_indexes.add(index)
        seen_targets.add(target)
        _recover_transaction_entry(store, transaction, raw, committed=committed, transaction_id=transaction_id)
    _require_directory_durable(store)
    _require_directory_durable(transaction)


def _recover_transaction_entry(
    store: Path,
    transaction: Path,
    row: Mapping[str, Any],
    *,
    committed: bool,
    transaction_id: str,
) -> None:
    index = row.get("index")
    target_name = row.get("target")
    if not isinstance(index, int) or not isinstance(target_name, str) or Path(target_name).name != target_name:
        raise OSError("Prepared metadata transaction target binding is invalid.")
    target = _safe_metadata_target(store / target_name, store)
    expected_new_payload = f"new/{index:04d}.json"
    if row.get("new_payload") != expected_new_payload:
        raise OSError("Prepared metadata transaction new-payload binding is invalid.")
    new_path = transaction / expected_new_payload
    _require_safe_directory(new_path.parent, parent=transaction)
    new_sha256 = str(row.get("new_sha256") or "")
    if _path_sha256(new_path) != new_sha256:
        raise OSError("Prepared metadata transaction new payload failed identity verification.")
    had_original = row.get("had_original") is True
    old_sha256 = str(row.get("old_sha256") or "") if had_original else None
    temporary = target.with_name(f".{target.name}.{transaction_id}.{index:04d}.installing")
    temporary_sha256 = _path_sha256(temporary)
    if temporary_sha256 is not None:
        _move_recovery_entry_into_transaction(
            temporary,
            transaction,
            destination_name=f"discarded_installing_{index:04d}",
        )
    current_sha256 = _path_sha256(target)
    if committed:
        if current_sha256 == new_sha256:
            return
        if current_sha256 not in {None, old_sha256}:
            raise OSError("Committed metadata transaction conflicts with a later target modification.")
        _install_transaction_payload(
            new_path,
            target,
            transaction_id=transaction_id,
            index=index,
        )
        return
    if had_original:
        expected_old_payload = f"old/{index:04d}.json"
        if row.get("old_payload") != expected_old_payload:
            raise OSError("Prepared metadata transaction old-payload binding is invalid.")
        old_path = transaction / expected_old_payload
        _require_safe_directory(old_path.parent, parent=transaction)
        if _path_sha256(old_path) != old_sha256:
            raise OSError("Prepared metadata transaction old payload failed identity verification.")
        if current_sha256 == old_sha256:
            return
        if current_sha256 not in {None, new_sha256}:
            raise OSError("Uncommitted metadata transaction conflicts with a later target modification.")
        _install_transaction_payload(
            old_path,
            target,
            transaction_id=transaction_id,
            index=index,
        )
        return
    if current_sha256 is None:
        return
    if current_sha256 != new_sha256:
        raise OSError("Uncommitted metadata transaction conflicts with a later target modification.")
    _move_recovery_entry_into_transaction(
        target,
        transaction,
        destination_name=f"discarded_target_{index:04d}",
    )


def _write_transaction_marker(path: Path, transaction_id: str, manifest_sha256: str) -> None:
    temporary = path.with_name(f".{path.name}.{transaction_id}.installing")
    _write_durable_file(
        temporary,
        _json_document_bytes(
            {
                "schema_version": _METADATA_TRANSACTION_MARKER_SCHEMA,
                "transaction_id": transaction_id,
                "manifest_sha256": manifest_sha256,
            }
        ),
    )
    _durable_replace(temporary, path)


def _validate_transaction_marker(path: Path, transaction_id: str, manifest_sha256: str) -> None:
    try:
        marker = json.loads(_read_confined_bytes(path, path.parent, label="Transaction marker").decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSError("Metadata transaction decision marker is unreadable.") from exc
    if not isinstance(marker, Mapping) or marker != {
        "schema_version": _METADATA_TRANSACTION_MARKER_SCHEMA,
        "transaction_id": transaction_id,
        "manifest_sha256": manifest_sha256,
    }:
        raise OSError("Metadata transaction decision marker failed identity verification.")


def _install_transaction_payload(
    payload: Path,
    target: Path,
    *,
    transaction_id: str,
    index: int,
) -> None:
    _require_safe_directory(payload.parent)
    payload_bytes = _read_confined_bytes(payload, payload.parent, label="Transaction payload")
    _require_safe_directory(target.parent)
    temporary = target.with_name(f".{target.name}.{transaction_id}.{index:04d}.installing")
    _write_durable_file(temporary, payload_bytes)
    _durable_replace(temporary, target)


def _durable_replace(source: Path, target: Path) -> None:
    source_parent = source.parent
    target_parent = target.parent
    if os.name == "nt":
        import ctypes

        move_file_ex = ctypes.windll.kernel32.MoveFileExW
        move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(str(source), str(target), 0x1 | 0x8):
            raise ctypes.WinError()
    else:
        os.replace(source, target)
    _fsync_directory(target_parent)
    if source_parent != target_parent:
        _fsync_directory(source_parent)


def _write_durable_file(path: Path, payload: bytes) -> None:
    _require_safe_directory(path.parent)
    with AnchoredDirectory(path.parent, path.parent) as parent_anchor:
        descriptor = parent_anchor.open_file(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                opened = os.fstat(descriptor)
                current = parent_anchor.lstat(path.name)
                if not OwnedFileIdentity.from_stat(opened).matches(current) or opened.st_nlink != 1:
                    raise OSError("Durable metadata file changed while it was being created.")
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                if os.fstat(handle.fileno()).st_nlink != 1:
                    raise OSError("Durable metadata file acquired multiple hard links while it was being written.")
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_directory_durable(path: Path) -> None:
    """Retryable recovery barrier before its only journal is removed."""

    lexical = Path(os.path.abspath(path))
    _require_safe_directory(lexical)
    if os.name == "nt":
        # Every recovery rename uses MoveFileExW with MOVEFILE_WRITE_THROUGH.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        current = lexical.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise OSError("Metadata directory changed while establishing a durability barrier.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_transaction_directory(transaction: Path) -> None:
    parent = Path(os.path.abspath(transaction.parent))
    _require_safe_directory(parent)
    _require_safe_directory(transaction, parent=parent)
    try:
        remove_confined_tree(transaction, parent)
    except UnsafeFilesystemOperation as exc:
        raise OSError("Metadata transaction cleanup lost its exact owned directory.") from exc
    _fsync_directory(parent)


def _move_recovery_entry_into_transaction(path: Path, transaction: Path, *, destination_name: str) -> None:
    _require_safe_directory(transaction, parent=transaction.parent)
    _require_safe_regular_file(path, parent=path.parent, label="Recovery entry")
    destination = transaction / destination_name
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise OSError(f"Metadata recovery discard entry already exists: {destination.name}")
    _durable_replace(path, destination)
    _require_safe_regular_file(destination, parent=transaction, label="Recovery discard entry")


def _path_sha256(path: Path) -> str | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    return hashlib.sha256(_read_confined_bytes(path, path.parent, label="Metadata transaction file")).hexdigest()


def _safe_metadata_target(path: Path, store: Path) -> Path:
    target = Path(os.path.abspath(path))
    lexical_store = Path(os.path.abspath(store))
    if target.parent != lexical_store or target.name in {"", ".", ".."}:
        raise OSError("Metadata transaction targets must be direct children of one metadata store.")
    _path_sha256(target)
    return target


def _require_safe_directory(path: Path, *, parent: Path | None = None) -> None:
    lexical = Path(os.path.abspath(path))
    if parent is not None and lexical.parent != Path(os.path.abspath(parent)):
        raise OSError(f"Metadata directory is outside its expected parent: {lexical.name}")
    try:
        status_value = lexical.lstat()
    except FileNotFoundError as exc:
        raise OSError(f"Required metadata directory is missing: {lexical.name}") from exc
    if _is_linked_or_reparse(lexical, status_value):
        raise OSError(f"Metadata directory is a linked or reparse entry: {lexical.name}")
    if not stat.S_ISDIR(status_value.st_mode):
        raise OSError(f"Metadata path is not a directory: {lexical.name}")


def _ensure_metadata_store_directory(store: Path) -> None:
    lexical = Path(os.path.abspath(store))
    parent = lexical.parent
    project_root = parent.parent
    project_root.mkdir(parents=True, exist_ok=True)
    if not project_root.is_dir():
        raise OSError("The selected project root is not a directory.")
    try:
        parent.mkdir()
    except FileExistsError:
        pass
    _require_safe_directory(parent, parent=project_root)
    try:
        lexical.mkdir()
    except FileExistsError:
        pass
    _require_safe_directory(lexical, parent=parent)


def _require_safe_regular_file(path: Path, *, parent: Path | None = None, label: str) -> os.stat_result:
    lexical = Path(os.path.abspath(path))
    if parent is not None and lexical.parent != Path(os.path.abspath(parent)):
        raise OSError(f"{label} is outside its expected directory.")
    try:
        status_value = lexical.lstat()
    except FileNotFoundError as exc:
        raise OSError(f"{label} is missing: {lexical.name}") from exc
    if _is_linked_or_reparse(lexical, status_value):
        raise OSError(f"{label} is a linked or reparse entry: {lexical.name}")
    if not stat.S_ISREG(status_value.st_mode):
        raise OSError(f"{label} is not a regular file: {lexical.name}")
    if status_value.st_nlink != 1:
        raise OSError(f"{label} has multiple hard links: {lexical.name}")
    return status_value


def _is_linked_or_reparse(path: Path, status_value: os.stat_result) -> bool:
    is_reparse = bool(getattr(status_value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    return path.is_symlink() or is_junction or is_reparse


def _read_confined_bytes(path: Path, parent: Path, *, label: str) -> bytes:
    lexical = Path(os.path.abspath(path))
    lexical_parent = Path(os.path.abspath(parent))
    _require_safe_directory(lexical_parent)
    before = _require_safe_regular_file(lexical, parent=lexical_parent, label=label)
    with lexical.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise OSError(f"{label} changed while it was being opened.")
        payload = handle.read()
    after = _require_safe_regular_file(lexical, parent=lexical_parent, label=label)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise OSError(f"{label} changed while it was being read.")
    return payload


def _read_confined_json_file(path: Path, store: Path, *, label: str) -> Any:
    try:
        return json.loads(_read_confined_bytes(path, store, label=label).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackMetadataError(f"{label} file {path.name!r} is unreadable or corrupt.") from exc


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _license_evidence_file_allowed(
    input_root: Path,
    pack_root: Path,
    pack: SourcePack,
    evidence_file: str,
) -> bool:
    relative = Path(evidence_file)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    candidate = (input_root / relative).resolve()
    if not candidate.is_file() or not _is_relative_to(candidate, input_root.resolve()):
        return False
    if _is_relative_to(candidate, pack_root.resolve()):
        return True
    inherited = {
        (input_root / str(row.get("relative_path") or "")).resolve()
        for row in pack.evidence_files
        if row.get("inherited")
    }
    return candidate in inherited

"""Export Dataset Maker imports to Phase 7-ready fixed arrays."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import stat
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.codec.bundle import INDEX_MASK, INDEX_PAD, SpriteBundle
from spritelab.codec.palette import visible_palette_size
from spritelab.codec.roles import ROLE_NAMES, ROLE_TRANSPARENT, ROLE_UNKNOWN
from spritelab.codec.validate import validate_bundle
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import (
    ALLOWED_SPLITS,
    DatasetMakerItem,
    normalize_sprite_id,
    validate_dataset_maker_item,
)
from spritelab.harvest.config_loader import labeling_config_metadata
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    remove_confined_tree,
    require_confined_path,
)

ANCHORED_EXPORT_COMMIT_SCHEMA = "spritelab.dataset-maker.anchored-export-commit.v1"


@dataclass(frozen=True)
class DatasetMakerExportConfig:
    dataset_name: str
    output_root: Path
    max_palette_slots: int = 32
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 1337
    overwrite: bool = False


@dataclass(frozen=True)
class DatasetMakerExportResult:
    output_dir: Path
    train_count: int
    val_count: int
    test_count: int
    accepted_count: int
    excluded_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnchoredDatasetMakerExport:
    result: DatasetMakerExportResult
    parent_identity: OwnedFileIdentity
    directory_identity: OwnedFileIdentity


@dataclass(frozen=True)
class _PreparedSprite:
    item: DatasetMakerItem
    bundle: SpriteBundle
    category_id: int
    split: str
    auto_metadata: Mapping[str, Any]


def export_dataset_from_imported_sprites(
    imported: Sequence[ImportedSprite],
    config: DatasetMakerExportConfig,
) -> DatasetMakerExportResult:
    """Export accepted Dataset Maker sprites into npz arrays and manifests."""

    _validate_config(config)
    dataset_name = normalize_sprite_id(config.dataset_name)
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty and filesystem-safe.")

    output_root = Path(config.output_root).expanduser()
    output_dir = require_confined_path(output_root / dataset_name, output_root)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not config.overwrite:
            raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")

    prepared, excluded, category_to_id, result = _prepare_export(
        imported,
        config,
        output_dir=output_dir,
    )
    from spritelab.dataset_maker.report import build_dataset_maker_report

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_name}.staging-", dir=output_root))
    try:
        for split_name in ("train", "val", "test"):
            split_sprites = [sprite for sprite in prepared if sprite.split == split_name]
            _write_split_npz(
                staging / f"{split_name}.npz",
                split_sprites,
                max_palette_slots=config.max_palette_slots,
            )
            _write_manifest_jsonl(staging / f"manifest_{split_name}.jsonl", _manifest_records(split_sprites))

        _write_json(staging / "vocab.json", _vocab(category_to_id, prepared))
        _write_json(staging / "dataset_config.json", _dataset_config(dataset_name, config.max_palette_slots))
        _write_rejected_jsonl(staging / "rejected.jsonl", excluded)
        (staging / "dataset_report.md").write_text(
            build_dataset_maker_report(imported, result),
            encoding="utf-8",
        )
        _publish_export(staging, output_dir, output_root, overwrite=config.overwrite)
    except BaseException:
        remove_confined_tree(staging, output_root, missing_ok=True)
        raise
    return result


def export_dataset_from_imported_sprites_anchored(
    imported: Sequence[ImportedSprite],
    config: DatasetMakerExportConfig,
    *,
    output_parent: AnchoredDirectory,
) -> AnchoredDatasetMakerExport:
    """Export into a fresh held child without pathname-oriented mutations.

    This API is for managed workflows that already hold the exact output
    parent. Input sprites are validated and converted to in-memory payloads;
    their source paths are never used as export destinations. The fresh output
    directory remains held from exclusive creation through population. It is
    non-authoritative until :func:`commit_anchored_dataset_maker_export`
    publishes the fixed sibling completion marker.
    """

    _validate_config(config)
    dataset_name = normalize_sprite_id(config.dataset_name)
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty and filesystem-safe.")
    if config.overwrite:
        raise ValueError("anchored Dataset Maker export only supports fresh immutable outputs.")

    output_parent.verify()
    parent_identity = OwnedFileIdentity.from_stat(output_parent.directory_metadata())
    configured_root = Path(os.path.abspath(os.path.expanduser(os.fspath(config.output_root))))
    held_root = Path(os.path.abspath(output_parent.directory))
    if os.path.normcase(os.fspath(configured_root)) != os.path.normcase(os.fspath(held_root)):
        raise UnsafeFilesystemOperation("anchored Dataset Maker output parent does not match its configured root")
    commit_name = _anchored_export_commit_name(dataset_name)
    if output_parent.lexists(dataset_name) or output_parent.lexists(commit_name):
        raise FileExistsError(f"anchored output already exists: {dataset_name}")

    output_dir = held_root / dataset_name
    prepared, excluded, category_to_id, result = _prepare_export(
        imported,
        config,
        output_dir=output_dir,
    )
    output_identity = output_parent.mkdir(dataset_name)
    with output_parent.open_directory_immovable(dataset_name) as output:
        if not output_identity.matches(output.directory_metadata()):
            raise UnsafeFilesystemOperation("anchored Dataset Maker output identity changed after creation")
        expected: dict[str, tuple[int, str]] = {}
        for name, content in _iter_export_payloads(
            imported,
            result,
            prepared,
            excluded,
            category_to_id,
            dataset_name=dataset_name,
            max_palette_slots=config.max_palette_slots,
        ):
            _write_anchored_file_exclusive(output, name, content)
            expected[name] = (len(content), hashlib.sha256(content).hexdigest())
        _verify_anchored_export(output, expected)
        _verify_anchored_export(output, expected)
        output_parent.verify()
    return AnchoredDatasetMakerExport(
        result=result,
        parent_identity=parent_identity,
        directory_identity=output_identity,
    )


def commit_anchored_dataset_maker_export(
    output_parent: AnchoredDirectory,
    dataset_name: str,
    *,
    expected_parent_identity: OwnedFileIdentity,
    expected_directory_identity: OwnedFileIdentity,
    expected_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish the immutable sibling marker after a complete managed build."""

    normalized_name = normalize_sprite_id(dataset_name)
    if not normalized_name or normalized_name != dataset_name:
        raise ValueError("anchored dataset name must already be normalized")
    inventory = _normalize_anchored_inventory(expected_inventory)
    marker_name = _anchored_export_commit_name(normalized_name)
    output_parent.verify()
    if not expected_parent_identity.matches(output_parent.directory_metadata()):
        raise UnsafeFilesystemOperation("anchored Dataset Maker output parent identity changed before commit")
    if output_parent.lexists(marker_name):
        raise FileExistsError(f"anchored export completion marker already exists: {marker_name}")
    if not expected_directory_identity.matches(output_parent.lstat(normalized_name)):
        raise UnsafeFilesystemOperation("anchored Dataset Maker output identity changed before commit")
    with output_parent.open_directory_immovable(normalized_name) as output:
        if not expected_directory_identity.matches(output.directory_metadata()):
            raise UnsafeFilesystemOperation("anchored Dataset Maker output identity changed before commit")
        _verify_anchored_export(output, _expected_inventory_tuples(inventory))
        _verify_anchored_export(output, _expected_inventory_tuples(inventory))
    marker = {
        "schema_version": ANCHORED_EXPORT_COMMIT_SCHEMA,
        "dataset_name": normalized_name,
        "parent_directory_identity": _owned_directory_identity_payload(expected_parent_identity),
        "directory_identity": _owned_directory_identity_payload(expected_directory_identity),
        "inventory": inventory,
        "inventory_sha256": _anchored_inventory_identity(inventory),
    }
    _write_anchored_file_exclusive(output_parent, marker_name, _json_payload(marker))
    with output_parent.open_directory_immovable(normalized_name) as output:
        expected = _expected_inventory_tuples(inventory)
        _verify_anchored_export(output, expected)
        _verify_anchored_export(output, expected)
    verify_anchored_dataset_maker_export(
        output_parent,
        normalized_name,
        expected_inventory=inventory,
    )
    return marker


def verify_anchored_dataset_maker_export(
    output_parent: AnchoredDirectory,
    dataset_name: str,
    *,
    expected_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the fixed commit marker and its exact held-directory inventory."""

    normalized_name = normalize_sprite_id(dataset_name)
    if not normalized_name or normalized_name != dataset_name:
        raise ValueError("anchored dataset name must already be normalized")
    inventory = _normalize_anchored_inventory(expected_inventory)
    marker_name = _anchored_export_commit_name(normalized_name)
    output_parent.verify()
    marker_content = _read_anchored_file_bytes(output_parent, marker_name, maximum_bytes=16 * 1024 * 1024)
    try:
        marker = json.loads(
            marker_content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise UnsafeFilesystemOperation("anchored Dataset Maker completion marker is invalid") from exc
    if not isinstance(marker, dict) or set(marker) != {
        "schema_version",
        "dataset_name",
        "parent_directory_identity",
        "directory_identity",
        "inventory",
        "inventory_sha256",
    }:
        raise UnsafeFilesystemOperation("anchored Dataset Maker completion marker schema is invalid")
    if marker["schema_version"] != ANCHORED_EXPORT_COMMIT_SCHEMA or marker["dataset_name"] != normalized_name:
        raise UnsafeFilesystemOperation("anchored Dataset Maker completion marker binding changed")
    marker_inventory = _normalize_anchored_inventory(marker["inventory"])
    if marker_inventory != inventory or marker["inventory_sha256"] != _anchored_inventory_identity(inventory):
        raise UnsafeFilesystemOperation("anchored Dataset Maker completion inventory changed")
    parent_identity = _owned_directory_identity_from_payload(marker["parent_directory_identity"])
    if not parent_identity.matches(output_parent.directory_metadata()):
        raise UnsafeFilesystemOperation("anchored Dataset Maker committed parent identity changed")
    directory_identity = _owned_directory_identity_from_payload(marker["directory_identity"])
    if not directory_identity.matches(output_parent.lstat(normalized_name)):
        raise UnsafeFilesystemOperation("anchored Dataset Maker committed directory identity changed")
    with output_parent.open_directory_immovable(normalized_name) as output:
        if not directory_identity.matches(output.directory_metadata()):
            raise UnsafeFilesystemOperation("anchored Dataset Maker committed directory identity changed")
        _verify_anchored_export(output, _expected_inventory_tuples(inventory))
    output_parent.verify()
    return marker


def _anchored_export_commit_name(dataset_name: str) -> str:
    return f"{dataset_name}.commit.json"


def _normalize_anchored_inventory(value: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise UnsafeFilesystemOperation("anchored Dataset Maker inventory is empty or malformed")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str):
            raise UnsafeFilesystemOperation("anchored Dataset Maker inventory contains an unsafe file name")
        name = raw_name
        if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise UnsafeFilesystemOperation("anchored Dataset Maker inventory contains an unsafe file name")
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"sha256", "byte_count"}:
            raise UnsafeFilesystemOperation("anchored Dataset Maker inventory record is malformed")
        digest = raw_record.get("sha256")
        byte_count = raw_record.get("byte_count")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise UnsafeFilesystemOperation("anchored Dataset Maker inventory record is malformed")
        normalized[name] = {"sha256": digest, "byte_count": byte_count}
    return normalized


def _expected_inventory_tuples(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[int, str]]:
    return {name: (int(record["byte_count"]), str(record["sha256"])) for name, record in sorted(inventory.items())}


def _anchored_inventory_identity(inventory: Mapping[str, Mapping[str, Any]]) -> str:
    canonical = json.dumps(dict(inventory), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _owned_directory_identity_payload(identity: OwnedFileIdentity) -> dict[str, int]:
    if identity.file_type != stat.S_IFDIR:
        raise UnsafeFilesystemOperation("anchored Dataset Maker output identity is not a directory")
    return {"device": identity.device, "inode": identity.inode, "file_type": identity.file_type}


def _owned_directory_identity_from_payload(value: Any) -> OwnedFileIdentity:
    if not isinstance(value, Mapping) or set(value) != {"device", "inode", "file_type"}:
        raise UnsafeFilesystemOperation("anchored Dataset Maker directory identity is malformed")
    fields = tuple(value.get(name) for name in ("device", "inode", "file_type"))
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in fields):
        raise UnsafeFilesystemOperation("anchored Dataset Maker directory identity is malformed")
    identity = OwnedFileIdentity(device=fields[0], inode=fields[1], file_type=fields[2])
    if identity.file_type != stat.S_IFDIR:
        raise UnsafeFilesystemOperation("anchored Dataset Maker directory identity is malformed")
    return identity


def _read_anchored_file_bytes(anchor: AnchoredDirectory, name: str, *, maximum_bytes: int) -> bytes:
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        identity = OwnedFileIdentity.from_stat(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise UnsafeFilesystemOperation("anchored Dataset Maker file is irregular or oversized")
        content = handle.read(maximum_bytes + 1)
        after = os.fstat(handle.fileno())
    current = anchor.lstat(name)
    if (
        len(content) != before.st_size
        or len(content) > maximum_bytes
        or OwnedFileIdentity.from_stat(after) != identity
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or not identity.matches(current)
        or current.st_nlink != 1
        or current.st_size != before.st_size
        or current.st_mtime_ns != after.st_mtime_ns
        or current.st_ctime_ns != after.st_ctime_ns
    ):
        raise UnsafeFilesystemOperation("anchored Dataset Maker file changed while reading")
    return content


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _prepare_export(
    imported: Sequence[ImportedSprite],
    config: DatasetMakerExportConfig,
    *,
    output_dir: Path,
) -> tuple[list[_PreparedSprite], list[ImportedSprite], dict[str, int], DatasetMakerExportResult]:
    accepted = [sprite for sprite in imported if sprite.item.status == "accepted"]
    excluded = [sprite for sprite in imported if sprite.item.status != "accepted"]
    if not accepted:
        raise ValueError("no accepted sprites to export.")

    _validate_unique_sprite_ids([sprite.item for sprite in accepted])
    category_to_id = _category_vocab([sprite.item for sprite in accepted])
    split_lookup = make_dataset_maker_split(
        [sprite.item for sprite in accepted],
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    if not any(split == "train" for split in split_lookup.values()):
        raise ValueError("train split would be empty; adjust split overrides or fractions.")

    prepared: list[_PreparedSprite] = []
    for sprite in accepted:
        _validate_imported_for_export(sprite, max_palette_slots=config.max_palette_slots)
        prepared.append(
            _PreparedSprite(
                item=sprite.item,
                bundle=sprite.bundle,
                category_id=category_to_id[sprite.item.category],
                split=split_lookup[sprite.item.sprite_id],
                auto_metadata=dict(sprite.auto_metadata) if isinstance(sprite.auto_metadata, Mapping) else {},
            )
        )

    result = DatasetMakerExportResult(
        output_dir=output_dir,
        train_count=sum(1 for sprite in prepared if sprite.split == "train"),
        val_count=sum(1 for sprite in prepared if sprite.split == "val"),
        test_count=sum(1 for sprite in prepared if sprite.split == "test"),
        accepted_count=len(prepared),
        excluded_count=len(excluded),
        warnings=tuple(_split_warnings(prepared)),
    )
    return prepared, excluded, category_to_id, result


def _iter_export_payloads(
    imported: Sequence[ImportedSprite],
    result: DatasetMakerExportResult,
    prepared: Sequence[_PreparedSprite],
    excluded: Sequence[ImportedSprite],
    category_to_id: Mapping[str, int],
    *,
    dataset_name: str,
    max_palette_slots: int,
) -> Iterator[tuple[str, bytes]]:
    from spritelab.dataset_maker.report import build_dataset_maker_report

    for split_name in ("train", "val", "test"):
        split_sprites = [sprite for sprite in prepared if sprite.split == split_name]
        yield f"{split_name}.npz", _split_npz_bytes(split_sprites, max_palette_slots=max_palette_slots)
        yield f"manifest_{split_name}.jsonl", _manifest_jsonl_bytes(_manifest_records(split_sprites))
    yield "vocab.json", _json_payload(_vocab(category_to_id, prepared))
    yield "dataset_config.json", _json_payload(_dataset_config(dataset_name, max_palette_slots))
    yield "rejected.jsonl", _rejected_jsonl_bytes(excluded)
    yield "dataset_report.md", build_dataset_maker_report(imported, result).encode("utf-8")


def _write_anchored_file_exclusive(anchor: AnchoredDirectory, name: str, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    descriptor = anchor.open_file(name, flags, 0o600)
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            for offset in range(0, len(content), 1024 * 1024):
                before = os.fstat(handle.fileno())
                if OwnedFileIdentity.from_stat(before) != identity or before.st_nlink != 1:
                    raise UnsafeFilesystemOperation("anchored Dataset Maker output file changed while writing")
                handle.write(content[offset : offset + 1024 * 1024])
            handle.flush()
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
            if (
                OwnedFileIdentity.from_stat(after) != identity
                or not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or after.st_size != len(content)
            ):
                raise UnsafeFilesystemOperation("anchored Dataset Maker output file changed after writing")
        current = anchor.lstat(name)
        if (
            not identity.matches(current)
            or current.st_nlink != 1
            or current.st_size != len(content)
            or current.st_mtime_ns != after.st_mtime_ns
            or current.st_ctime_ns != after.st_ctime_ns
        ):
            raise UnsafeFilesystemOperation("anchored Dataset Maker output path changed after writing")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _verify_anchored_export(
    anchor: AnchoredDirectory,
    expected: Mapping[str, tuple[int, str]],
) -> None:
    if set(anchor.names()) != set(expected):
        raise UnsafeFilesystemOperation("anchored Dataset Maker export inventory changed")
    for name in sorted(expected):
        expected_size, expected_sha256 = expected[name]
        descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            identity = OwnedFileIdentity.from_stat(before)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_size:
                raise UnsafeFilesystemOperation("anchored Dataset Maker export contains an irregular file")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            stable_metadata = (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            ) == (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            if OwnedFileIdentity.from_stat(after) != identity or not stable_metadata:
                raise UnsafeFilesystemOperation("anchored Dataset Maker export changed while verifying")
        current = anchor.lstat(name)
        if (
            not identity.matches(current)
            or current.st_nlink != 1
            or current.st_size != expected_size
            or current.st_mtime_ns != after.st_mtime_ns
            or current.st_ctime_ns != after.st_ctime_ns
            or digest.hexdigest() != expected_sha256
        ):
            raise UnsafeFilesystemOperation("anchored Dataset Maker export bytes changed while verifying")


def _publish_export(staging: Path, output: Path, root: Path, *, overwrite: bool) -> None:
    """Publish a complete export while preserving the old tree on failure."""

    require_confined_path(staging, root)
    require_confined_path(output, root)
    if not output.exists():
        staging.replace(output)
        return
    if not output.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output}")
    non_empty = any(output.iterdir())
    if non_empty and not overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {output}")
    if not non_empty:
        output.rmdir()
        staging.replace(output)
        return

    backup = root / f".{output.name}.previous-{uuid.uuid4().hex}"
    require_confined_path(backup, root)
    output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        backup.replace(output)
        raise
    remove_confined_tree(backup, root)


def make_dataset_maker_split(
    items: Sequence[DatasetMakerItem],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Return a deterministic ``sprite_id -> split`` mapping."""

    _validate_fractions(train_fraction, val_fraction, test_fraction)
    ordered_items = sorted(items, key=lambda item: item.sprite_id)
    split_by_id: dict[str, str] = {}
    auto_items: list[DatasetMakerItem] = []
    for item in ordered_items:
        if item.split is None:
            auto_items.append(item)
        elif item.split in ALLOWED_SPLITS:
            split_by_id[item.sprite_id] = item.split
        else:
            raise ValueError(f"{item.sprite_id}: invalid split override {item.split!r}.")

    total_count = len(ordered_items)
    if total_count == 0:
        return {}
    if total_count < 3:
        for item in auto_items:
            split_by_id[item.sprite_id] = "train"
        return split_by_id

    target_counts = _target_split_counts(
        total_count,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    current_counts = Counter(split_by_id.values())
    shuffled = list(auto_items)
    random.Random(seed).shuffle(shuffled)

    for split_name in ("train", "val", "test"):
        if target_counts[split_name] <= 0 or current_counts[split_name] > 0:
            continue
        if not shuffled:
            break
        item = shuffled.pop(0)
        split_by_id[item.sprite_id] = split_name
        current_counts[split_name] += 1

    for item in shuffled:
        split_name = _best_split(current_counts, target_counts)
        split_by_id[item.sprite_id] = split_name
        current_counts[split_name] += 1

    return split_by_id


def _validate_imported_for_export(imported: ImportedSprite, *, max_palette_slots: int) -> None:
    item = imported.item
    item_errors = validate_dataset_maker_item(item)
    if item_errors:
        raise ValueError(f"{item.sprite_id}: invalid metadata: {'; '.join(item_errors)}")
    if imported.errors:
        raise ValueError(f"{item.sprite_id}: accepted sprite has import errors: {'; '.join(imported.errors)}")
    if imported.bundle is None:
        raise ValueError(f"{item.sprite_id}: accepted sprite has no SpriteBundle.")

    bundle = imported.bundle
    bundle_errors = validate_bundle(bundle)
    if bundle_errors:
        raise ValueError(f"{item.sprite_id}: invalid SpriteBundle: {'; '.join(bundle_errors)}")
    palette_size = visible_palette_size(np.asarray(bundle.palette))
    if palette_size > max_palette_slots:
        raise ValueError(
            f"{item.sprite_id}: palette has {palette_size} visible slots, above max_palette_slots={max_palette_slots}."
        )
    sample_errors = _validate_sample_contract(bundle, max_palette_slots=max_palette_slots)
    if sample_errors:
        raise ValueError(f"{item.sprite_id}: {'; '.join(sample_errors)}")


def _validate_sample_contract(bundle: SpriteBundle, *, max_palette_slots: int) -> list[str]:
    errors: list[str] = []
    alpha = np.asarray(bundle.alpha)
    index_map = np.asarray(bundle.index_map)
    palette = np.asarray(bundle.palette)
    visible_rows = int(palette.shape[0] - 1)

    if alpha.shape != (32, 32):
        errors.append("alpha shape must be [32, 32].")
    if index_map.shape != (32, 32):
        errors.append("index_map shape must be [32, 32].")
    if not np.all(np.isin(alpha, [0, 1])):
        errors.append("alpha values must be 0 or 1.")
    if bool(np.any((alpha == 0) & (index_map != 0))):
        errors.append("transparent pixels must have index_map == 0.")
    if bool(np.any((alpha == 1) & (index_map < 1))):
        errors.append("opaque pixels must have index_map >= 1.")
    if palette.shape[0] and not np.array_equal(palette[0], np.array([0, 0, 0], dtype=np.uint8)):
        errors.append("palette row 0 must be [0, 0, 0].")
    if visible_rows > max_palette_slots:
        errors.append("palette has more visible rows than max_palette_slots.")
    if index_map.size:
        maximum_index = index_map.max().item()
        if int(maximum_index) > visible_rows:
            errors.append("index_map points outside the true palette rows.")
    return errors


def _write_split_npz(path: Path, sprites: Sequence[_PreparedSprite], *, max_palette_slots: int) -> None:
    path.write_bytes(_split_npz_bytes(sprites, max_palette_slots=max_palette_slots))


def _split_npz_bytes(sprites: Sequence[_PreparedSprite], *, max_palette_slots: int) -> bytes:
    count = len(sprites)
    palette_rows = max_palette_slots + 1
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, palette_rows, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, palette_rows), dtype=bool)
    category_id = np.zeros((count,), dtype=np.int64)
    sprite_id = np.array([sprite.item.sprite_id for sprite in sprites], dtype=np.str_)

    for row, sprite in enumerate(sprites):
        bundle = sprite.bundle
        bundle_palette = np.asarray(bundle.palette, dtype=np.uint8)
        actual_rows = int(bundle_palette.shape[0])
        alpha[row] = np.asarray(bundle.alpha, dtype=np.uint8)
        index_map[row] = np.asarray(bundle.index_map, dtype=np.int16)
        role_map[row] = _role_map_for_export(bundle)
        palette[row, :actual_rows] = bundle_palette
        palette_mask[row, :actual_rows] = True
        category_id[row] = int(sprite.category_id)

    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        alpha=alpha,
        index_map=index_map,
        role_map=role_map,
        palette=palette,
        palette_mask=palette_mask,
        category_id=category_id,
        sprite_id=sprite_id,
    )
    return payload.getvalue()


def _role_map_for_export(bundle: SpriteBundle) -> np.ndarray:
    if bundle.role_map is not None:
        return np.asarray(bundle.role_map, dtype=np.uint8)
    fallback = np.zeros((32, 32), dtype=np.uint8)
    alpha = np.asarray(bundle.alpha)
    fallback[alpha == 0] = ROLE_TRANSPARENT
    fallback[alpha == 1] = ROLE_UNKNOWN
    return fallback


def _manifest_records(sprites: Sequence[_PreparedSprite]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sprite in sorted(sprites, key=lambda value: value.item.sprite_id):
        item = sprite.item
        label_v2 = _label_v2_manifest_metadata(sprite.auto_metadata)
        safe_prefill = (
            _mapping(sprite.auto_metadata.get("label_v2_safe_prefill"))
            if isinstance(sprite.auto_metadata, Mapping)
            else {}
        )
        semantic_v3 = (
            _mapping(sprite.auto_metadata.get("semantic_v3")) if isinstance(sprite.auto_metadata, Mapping) else {}
        )
        records.append(
            {
                "sprite_id": item.sprite_id,
                "split": sprite.split,
                "category": item.category,
                "category_id": sprite.category_id,
                "object_name": str(safe_prefill.get("object_name", "")),
                "label_confidence_tier": str(sprite.auto_metadata.get("label_v2_label_confidence_tier", "")),
                "tags": list(item.tags),
                "short_description": str(safe_prefill.get("short_description") or item.notes),
                "materials": [str(value) for value in safe_prefill.get("materials") or ()],
                "mood": [str(value) for value in safe_prefill.get("mood") or ()],
                "palette_size": int(visible_palette_size(np.asarray(sprite.bundle.palette))),
                "has_role_map": sprite.bundle.role_map is not None,
                "source_path": str(item.source_path),
                "source_name": item.source_name,
                "license": item.license,
                "author": item.author,
                "notes": item.notes,
                "label_v2": label_v2,
                **({"semantic_v3": semantic_v3} if semantic_v3 else {}),
            }
        )
    return records


def _label_v2_manifest_metadata(auto_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(auto_metadata, Mapping) or not auto_metadata.get("label_v2_applied"):
        return {}
    result: dict[str, Any] = {
        "applied": bool(auto_metadata.get("label_v2_applied", False)),
        "prediction_file": str(auto_metadata.get("label_v2_prediction_file", "")),
        "bucket": str(auto_metadata.get("label_v2_bucket", "")),
        "flags": [str(value) for value in auto_metadata.get("label_v2_flags") or ()],
        "candidate_object_names": [str(value) for value in auto_metadata.get("label_v2_candidate_object_names") or ()],
    }
    label_quality = _mapping(auto_metadata.get("label_v2_label_quality"))
    tier = str(
        auto_metadata.get("label_v2_label_confidence_tier", "") or label_quality.get("label_confidence_tier", "")
    ).strip()
    if tier:
        result["label_confidence_tier"] = tier
    result["fusion_bucket"] = result["bucket"]
    if auto_metadata.get("label_v2_applied_at_build_id"):
        result["applied_at_build_id"] = str(auto_metadata.get("label_v2_applied_at_build_id", ""))
    safe_prefill = _mapping(auto_metadata.get("label_v2_safe_prefill"))
    if safe_prefill:
        result["safe_prefill"] = safe_prefill
    vlm_descriptor = _mapping(auto_metadata.get("label_v2_vlm_descriptor"))
    if vlm_descriptor:
        result["vlm_descriptor"] = vlm_descriptor
        result["vlm_object_name"] = str(vlm_descriptor.get("object_name", ""))
        result["vlm_alternative_object_names"] = [
            str(value) for value in vlm_descriptor.get("alternative_object_names") or ()
        ]
        result["vlm_source_consistency"] = str(vlm_descriptor.get("source_consistency", ""))
    if label_quality:
        result["label_quality"] = label_quality
    conflict_reasons = [str(value) for value in auto_metadata.get("label_v2_conflict_reasons") or ()]
    if conflict_reasons:
        result["conflict_reasons"] = conflict_reasons
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_manifest_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_bytes(_manifest_jsonl_bytes(records))


def _manifest_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _write_rejected_jsonl(path: Path, sprites: Sequence[ImportedSprite]) -> None:
    path.write_bytes(_rejected_jsonl_bytes(sprites))


def _rejected_jsonl_bytes(sprites: Sequence[ImportedSprite]) -> bytes:
    lines: list[str] = []
    for sprite in sorted(sprites, key=lambda value: value.item.sprite_id):
        item = sprite.item
        lines.append(
            json.dumps(
                {
                    "sprite_id": item.sprite_id,
                    "status": item.status,
                    "category": item.category,
                    "tags": list(item.tags),
                    "source_path": str(item.source_path),
                    "source_name": item.source_name,
                    "license": item.license,
                    "author": item.author,
                    "notes": item.notes,
                    "errors": list(sprite.errors),
                    "warnings": list(sprite.warnings),
                },
                sort_keys=True,
            )
        )
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _vocab(category_to_id: Mapping[str, int], sprites: Sequence[_PreparedSprite]) -> dict[str, Any]:
    tags = sorted({tag for sprite in sprites for tag in sprite.item.tags})
    role_order = sorted(ROLE_NAMES)
    return {
        "category_to_id": dict(sorted(category_to_id.items(), key=lambda item: item[1])),
        "tag_to_id": {tag: index for index, tag in enumerate(tags, start=1)},
        "role_names": {str(role_id): ROLE_NAMES[role_id] for role_id in role_order},
    }


def _dataset_config(dataset_name: str, max_palette_slots: int) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "sprite_size": 32,
        "max_palette_slots": max_palette_slots,
        "index_transparent": 0,
        "index_pad": INDEX_PAD,
        "index_mask": INDEX_MASK,
        "role_transparent": ROLE_TRANSPARENT,
        "role_unknown": ROLE_UNKNOWN,
        "alpha_shape": [32, 32],
        "index_map_shape": [32, 32],
        "created_by": "spritelab.dataset_maker",
        "format_version": "1.0",
        "labeling_v2_config": labeling_config_metadata(),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_bytes(_json_payload(data))


def _json_payload(data: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(data), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _category_vocab(items: Sequence[DatasetMakerItem]) -> dict[str, int]:
    categories = {item.category or "unknown" for item in items}
    ordered = ["unknown", *sorted(category for category in categories if category != "unknown")]
    return {category: index for index, category in enumerate(ordered)}


def _validate_unique_sprite_ids(items: Sequence[DatasetMakerItem]) -> None:
    counts = Counter(item.sprite_id for item in items)
    duplicates = sorted(sprite_id for sprite_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sprite_id values among accepted sprites: {', '.join(duplicates)}")


def _validate_config(config: DatasetMakerExportConfig) -> None:
    if config.max_palette_slots < 1:
        raise ValueError("max_palette_slots must be at least 1.")
    _validate_fractions(config.train_fraction, config.val_fraction, config.test_fraction)


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(value < 0.0 for value in fractions):
        raise ValueError("split fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")


def _target_split_counts(
    total_count: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    names = ("train", "val", "test")
    fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    raw = {name: total_count * fractions[name] for name in names}
    counts = {name: int(np.floor(raw[name])) for name in names}
    remainder = total_count - sum(counts.values())
    for name in sorted(names, key=lambda split: (raw[split] - counts[split], split), reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    positive = [name for name in names if fractions[name] > 0.0]
    if total_count >= len(positive):
        for name in positive:
            if counts[name] == 0:
                donor = max((other for other in names if counts[other] > 1), key=lambda other: counts[other])
                counts[donor] -= 1
                counts[name] += 1
    return counts


def _best_split(current_counts: Mapping[str, int], target_counts: Mapping[str, int]) -> str:
    names = ("train", "val", "test")

    def score(name: str) -> tuple[float, int]:
        target = max(1, int(target_counts[name]))
        count = int(current_counts.get(name, 0))
        return ((target - count) / target, -count)

    return max(names, key=score)


def _split_warnings(sprites: Sequence[_PreparedSprite]) -> list[str]:
    counts = Counter(sprite.split for sprite in sprites)
    warnings: list[str] = []
    if len(sprites) < 3:
        warnings.append("Tiny dataset: val/test splits may be empty.")
    elif counts["val"] == 0 or counts["test"] == 0:
        warnings.append("One or more validation/test splits are empty.")
    return warnings

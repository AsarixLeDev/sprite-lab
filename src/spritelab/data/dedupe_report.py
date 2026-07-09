"""Duplicate and near-duplicate reports for SpriteBundle datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteBundle
from spritelab.codec.io import load_bundle
from spritelab.codec.palette import visible_palette_size
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, load_manifest
from spritelab.data.preview_grid import (
    PreviewGridRecord,
    filter_preview_records,
    load_preview_records,
)

SOURCE_SHA256 = "SOURCE_SHA256"
DECODED_RGBA_SHA256 = "DECODED_RGBA_SHA256"
BUNDLE_CONTENT_SHA256 = "BUNDLE_CONTENT_SHA256"
PERCEPTUAL_HASH = "PERCEPTUAL_HASH"


@dataclass(frozen=True)
class DedupeReportOptions:
    dataset_path: Path
    output_dir: Path | None = None
    max_items: int | None = None
    filter_category: str | None = None
    filter_split: str | None = None

    include_markdown: bool = True
    include_json: bool = True
    write_group_files: bool = True

    near_duplicate: bool = True
    near_duplicate_threshold: int = 8

    fail_on_load_error: bool = False


@dataclass(frozen=True)
class DedupeSpriteRecord:
    id: str
    bundle_dir: str

    source_path: str | None
    category: str | None
    split: str | None
    palette_size: int | None

    source_sha256: str | None
    decoded_rgba_sha256: str
    bundle_content_sha256: str

    average_hash: str | None
    difference_hash: str | None


@dataclass(frozen=True)
class DuplicateGroup:
    kind: str
    key: str
    ids: list[str]
    bundle_dirs: list[str]
    splits: list[str]
    crosses_splits: bool


@dataclass(frozen=True)
class NearDuplicateGroup:
    kind: str
    ids: list[str]
    bundle_dirs: list[str]
    splits: list[str]
    crosses_splits: bool
    max_distance: int
    pairs: list[dict[str, Any]]


@dataclass(frozen=True)
class FailedDedupeRecord:
    id: str | None
    bundle_dir: str
    reason: str


@dataclass(frozen=True)
class DedupeReportSummary:
    total_records: int
    analyzed_records: int
    failed_records: int

    exact_source_duplicate_groups: int
    exact_decoded_duplicate_groups: int
    exact_bundle_duplicate_groups: int
    near_duplicate_groups: int

    cross_split_exact_groups: int
    cross_split_near_groups: int

    duplicate_id_count: int
    duplicate_source_path_count: int


@dataclass(frozen=True)
class DedupeReport:
    dataset_name: str
    summary: DedupeReportSummary
    records: list[DedupeSpriteRecord]
    exact_groups: list[DuplicateGroup]
    near_groups: list[NearDuplicateGroup]
    duplicate_ids: dict[str, list[str]]
    duplicate_source_paths: dict[str, list[str]]
    failed: list[FailedDedupeRecord]
    options: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_rgba_sha256(bundle: SpriteBundle) -> str:
    """Hash the exact reconstructed 32x32 RGBA bytes for a bundle."""

    image = reconstruct_rgba(bundle).convert("RGBA")
    return sha256_bytes(image.tobytes())


def bundle_content_sha256(bundle: SpriteBundle) -> str:
    """Hash deterministic structural bundle arrays, excluding metadata."""

    digest = hashlib.sha256()
    _update_array_hash(digest, "alpha", np.asarray(bundle.alpha))
    _update_array_hash(digest, "palette", np.asarray(bundle.palette))
    _update_array_hash(digest, "index_map", np.asarray(bundle.index_map))
    if bundle.role_map is None:
        _update_bytes_hash(digest, b"role_map:null")
    else:
        _update_array_hash(digest, "role_map", np.asarray(bundle.role_map))
    return digest.hexdigest()


def average_hash_image(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a lightweight average hash as a fixed-length hex string."""

    if hash_size < 1:
        raise ValueError("hash_size must be at least 1.")

    grayscale = _fingerprint_grayscale(image)
    small = grayscale.resize((hash_size, hash_size), resample=Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.float32)
    mean = float(values.mean())
    bits = [1 if float(value) >= mean else 0 for value in values.flatten()]
    return _bits_to_hex(bits)


def difference_hash_image(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a lightweight difference hash as a fixed-length hex string."""

    if hash_size < 1:
        raise ValueError("hash_size must be at least 1.")

    grayscale = _fingerprint_grayscale(image)
    small = grayscale.resize((hash_size + 1, hash_size), resample=Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.int16)
    bits: list[int] = []
    for y in range(hash_size):
        for x in range(hash_size):
            bits.append(1 if int(values[y, x]) > int(values[y, x + 1]) else 0)
    return _bits_to_hex(bits)


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between equal-length hex bit hashes."""

    if len(hash_a) != len(hash_b):
        raise ValueError("hashes must have equal hex length.")
    value_a = int(hash_a, 16)
    value_b = int(hash_b, 16)
    return int((value_a ^ value_b).bit_count())


def create_dedupe_report(options: DedupeReportOptions) -> DedupeReport:
    """Load a dataset, compute duplicate groups, write reports, and return them."""

    preview_records = load_preview_records(options.dataset_path)
    preview_records = filter_preview_records(
        preview_records,
        filter_category=options.filter_category,
        filter_split=options.filter_split,
    )
    if options.max_items is not None:
        preview_records = preview_records[: options.max_items]

    manifest_lookup = _manifest_lookup(options.dataset_path)
    records: list[DedupeSpriteRecord] = []
    failed: list[FailedDedupeRecord] = []

    for preview_record in preview_records:
        try:
            bundle = load_bundle(preview_record.bundle_dir)
            reconstructed = reconstruct_rgba(bundle).convert("RGBA")
            source_sha = _source_sha256_for_record(preview_record, manifest_lookup)
            records.append(
                DedupeSpriteRecord(
                    id=preview_record.id,
                    bundle_dir=str(preview_record.bundle_dir),
                    source_path=preview_record.source_path,
                    category=preview_record.category,
                    split=preview_record.split,
                    palette_size=_palette_size_for_record(preview_record, bundle),
                    source_sha256=source_sha,
                    decoded_rgba_sha256=sha256_bytes(reconstructed.tobytes()),
                    bundle_content_sha256=bundle_content_sha256(bundle),
                    average_hash=average_hash_image(reconstructed) if options.near_duplicate else None,
                    difference_hash=difference_hash_image(reconstructed) if options.near_duplicate else None,
                )
            )
        except Exception as exc:
            if options.fail_on_load_error:
                raise
            failed.append(
                FailedDedupeRecord(
                    id=preview_record.id,
                    bundle_dir=str(preview_record.bundle_dir),
                    reason=str(exc),
                )
            )

    exact_groups = _exact_duplicate_groups(records)
    near_groups = (
        _near_duplicate_groups(records, threshold=options.near_duplicate_threshold) if options.near_duplicate else []
    )
    duplicate_ids = _duplicate_value_map(records, value_name="id")
    duplicate_source_paths = _duplicate_value_map(records, value_name="source_path")
    summary = _build_summary(
        total_records=len(preview_records),
        records=records,
        failed=failed,
        exact_groups=exact_groups,
        near_groups=near_groups,
        duplicate_ids=duplicate_ids,
        duplicate_source_paths=duplicate_source_paths,
    )
    report = DedupeReport(
        dataset_name=_dataset_name(options.dataset_path),
        summary=summary,
        records=records,
        exact_groups=exact_groups,
        near_groups=near_groups,
        duplicate_ids=duplicate_ids,
        duplicate_source_paths=duplicate_source_paths,
        failed=failed,
        options=_options_dict(options),
    )

    output_dir = _output_dir(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    if options.include_json:
        save_dedupe_report_json(report, output_dir / "dedupe_report.json")
    if options.include_markdown:
        save_dedupe_report_markdown(report, output_dir / "dedupe_report.md")
    if options.write_group_files:
        _write_group_files(report, output_dir / "duplicate_groups")

    return report


def save_dedupe_report_json(report: DedupeReport, path: str | Path) -> None:
    """Write a dedupe report as stable, readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dedupe_report_json(path: str | Path) -> DedupeReport:
    """Load a dedupe report from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DedupeReport(
        dataset_name=data["dataset_name"],
        summary=DedupeReportSummary(**data["summary"]),
        records=[DedupeSpriteRecord(**record) for record in data["records"]],
        exact_groups=[DuplicateGroup(**group) for group in data["exact_groups"]],
        near_groups=[NearDuplicateGroup(**group) for group in data["near_groups"]],
        duplicate_ids={str(key): list(value) for key, value in data["duplicate_ids"].items()},
        duplicate_source_paths={str(key): list(value) for key, value in data["duplicate_source_paths"].items()},
        failed=[FailedDedupeRecord(**record) for record in data["failed"]],
        options=dict(data["options"]),
    )


def render_dedupe_report_markdown(report: DedupeReport) -> str:
    """Render a dedupe report as simple Markdown."""

    summary = report.summary
    lines = [
        "# Dataset Dedupe Report",
        "",
        f"Dataset: {report.dataset_name}",
        "",
        "## Summary",
        "",
        f"- Total records: {summary.total_records}",
        f"- Analyzed records: {summary.analyzed_records}",
        f"- Failed records: {summary.failed_records}",
        f"- Exact source duplicate groups: {summary.exact_source_duplicate_groups}",
        f"- Exact decoded sprite duplicate groups: {summary.exact_decoded_duplicate_groups}",
        f"- Exact bundle duplicate groups: {summary.exact_bundle_duplicate_groups}",
        f"- Near-duplicate groups: {summary.near_duplicate_groups}",
        f"- Cross-split exact duplicate groups: {summary.cross_split_exact_groups}",
        f"- Cross-split near-duplicate groups: {summary.cross_split_near_groups}",
        f"- Duplicate ID count: {summary.duplicate_id_count}",
        f"- Duplicate source path count: {summary.duplicate_source_path_count}",
        "",
        "## Critical split leakage",
        "",
        "| Kind | IDs | Splits |",
        "|---|---|---|",
    ]
    critical_rows = _critical_split_rows(report)
    lines.extend(critical_rows or ["| None |  |  |"])

    lines.extend(
        [
            "",
            "## Exact duplicate groups",
            "",
            "| Kind | Count | Crosses splits | IDs |",
            "|---|---:|---|---|",
        ]
    )
    if report.exact_groups:
        for group in report.exact_groups:
            lines.append(
                f"| {group.kind} | {len(group.ids)} | {_yes_no(group.crosses_splits)} | {', '.join(group.ids)} |"
            )
    else:
        lines.append("| None | 0 | no |  |")

    lines.extend(
        [
            "",
            "## Near-duplicate groups",
            "",
            "| Count | Max distance | Crosses splits | IDs |",
            "|---:|---:|---|---|",
        ]
    )
    if report.near_groups:
        for group in report.near_groups:
            lines.append(
                f"| {len(group.ids)} | {group.max_distance} | {_yes_no(group.crosses_splits)} | "
                f"{', '.join(group.ids)} |"
            )
    else:
        lines.append("| 0 | 0 | no |  |")

    lines.extend(
        [
            "",
            "## Duplicate IDs",
            "",
            "| ID | Bundle dirs |",
            "|---|---|",
        ]
    )
    lines.extend(_duplicate_map_rows(report.duplicate_ids, empty_label="None"))

    lines.extend(
        [
            "",
            "## Duplicate source paths",
            "",
            "| Source path | Bundle dirs |",
            "|---|---|",
        ]
    )
    lines.extend(_duplicate_map_rows(report.duplicate_source_paths, empty_label="None"))

    lines.extend(
        [
            "",
            "## Failed records",
            "",
            "| ID | Bundle dir | Reason |",
            "|---|---|---|",
        ]
    )
    if report.failed:
        for record in report.failed:
            lines.append(f"| {record.id or ''} | {record.bundle_dir} | {record.reason} |")
    else:
        lines.append("| None |  |  |")

    lines.extend(
        [
            "",
            "## Hash meanings",
            "",
            "- `source_sha256`: SHA256 of the original source PNG when available.",
            "- `decoded_rgba_sha256`: SHA256 of reconstructed 32x32 RGBA bytes; this is the main exact visual duplicate signal.",
            "- `bundle_content_sha256`: SHA256 of structural bundle arrays, excluding metadata.",
            "- `average_hash` and `difference_hash`: lightweight perceptual fingerprints for heuristic near-duplicate grouping.",
        ]
    )

    return "\n".join(lines) + "\n"


def save_dedupe_report_markdown(report: DedupeReport, path: str | Path) -> None:
    """Write a dedupe report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dedupe_report_markdown(report), encoding="utf-8")


def _update_array_hash(digest: hashlib._Hash, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    header = {
        "name": name,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "nbytes": int(contiguous.nbytes),
    }
    _update_bytes_hash(digest, json.dumps(header, sort_keys=True).encode("utf-8"))
    _update_bytes_hash(digest, contiguous.tobytes())


def _update_bytes_hash(digest: hashlib._Hash, data: bytes) -> None:
    digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
    digest.update(data)


def _fingerprint_grayscale(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("L")


def _bits_to_hex(bits: list[int]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    hex_digits = (len(bits) + 3) // 4
    return f"{value:0{hex_digits}x}"


def _manifest_lookup(dataset_path: Path) -> dict[str, IngestedSpriteRecord]:
    manifest = _load_manifest_for_dataset(dataset_path)
    if manifest is None:
        return {}

    lookup: dict[str, IngestedSpriteRecord] = {}
    manifest_path = _manifest_path_for_dataset(dataset_path)
    if manifest_path is None:
        return lookup

    for record in manifest.records:
        resolved = _resolve_manifest_bundle_dir(record, manifest_path=manifest_path)
        lookup[_normalize_path(resolved)] = record
    return lookup


def _load_manifest_for_dataset(dataset_path: Path) -> DatasetManifest | None:
    manifest_path = _manifest_path_for_dataset(dataset_path)
    return load_manifest(manifest_path) if manifest_path is not None else None


def _manifest_path_for_dataset(dataset_path: Path) -> Path | None:
    path = Path(dataset_path)
    if path.is_file() and path.name == "manifest.json":
        return path
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    return None


def _resolve_manifest_bundle_dir(record: IngestedSpriteRecord, *, manifest_path: Path) -> Path:
    raw = Path(record.bundle_dir)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw

    manifest_dir = manifest_path.parent
    candidates = [
        manifest_dir / raw,
        manifest_dir / "bundles" / record.id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _source_sha256_for_record(
    record: PreviewGridRecord,
    manifest_lookup: dict[str, IngestedSpriteRecord],
) -> str | None:
    manifest_record = manifest_lookup.get(_normalize_path(record.bundle_dir))
    if manifest_record is not None and manifest_record.sha256:
        return manifest_record.sha256

    if record.source_path is None:
        return None

    source_path = Path(record.source_path)
    if source_path.exists() and source_path.is_file():
        return sha256_file(source_path)
    return None


def _palette_size_for_record(record: PreviewGridRecord, bundle: SpriteBundle) -> int | None:
    if record.palette_size is not None:
        return int(record.palette_size)
    if bundle.metadata.palette_size is not None:
        return int(bundle.metadata.palette_size)
    return visible_palette_size(np.asarray(bundle.palette))


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False)).lower()


def _exact_duplicate_groups(records: list[DedupeSpriteRecord]) -> list[DuplicateGroup]:
    groups: list[DuplicateGroup] = []
    groups.extend(_groups_by_hash(records, kind=SOURCE_SHA256, attr="source_sha256"))
    groups.extend(_groups_by_hash(records, kind=DECODED_RGBA_SHA256, attr="decoded_rgba_sha256"))
    groups.extend(_groups_by_hash(records, kind=BUNDLE_CONTENT_SHA256, attr="bundle_content_sha256"))
    return sorted(groups, key=lambda group: (not group.crosses_splits, group.kind, group.ids, group.key))


def _groups_by_hash(records: list[DedupeSpriteRecord], *, kind: str, attr: str) -> list[DuplicateGroup]:
    grouped: dict[str, list[DedupeSpriteRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, attr)
        if value is not None:
            grouped[str(value)].append(record)

    groups: list[DuplicateGroup] = []
    for key, group_records in grouped.items():
        if len(group_records) < 2:
            continue
        groups.append(_duplicate_group(kind=kind, key=key, records=group_records))
    return groups


def _duplicate_group(
    *,
    kind: str,
    key: str,
    records: list[DedupeSpriteRecord],
) -> DuplicateGroup:
    ordered = sorted(records, key=lambda record: (record.id, record.bundle_dir))
    splits = _unique_splits(ordered)
    return DuplicateGroup(
        kind=kind,
        key=key,
        ids=[record.id for record in ordered],
        bundle_dirs=[record.bundle_dir for record in ordered],
        splits=splits,
        crosses_splits=len(splits) > 1,
    )


def _near_duplicate_groups(
    records: list[DedupeSpriteRecord],
    *,
    threshold: int,
) -> list[NearDuplicateGroup]:
    if threshold < 0:
        raise ValueError("near_duplicate_threshold must be non-negative.")

    pairs: list[dict[str, Any]] = []
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left = records[left_index]
            right = records[right_index]
            if left.average_hash is None or right.average_hash is None:
                continue
            if left.difference_hash is None or right.difference_hash is None:
                continue

            average_distance = hamming_distance_hex(left.average_hash, right.average_hash)
            difference_distance = hamming_distance_hex(left.difference_hash, right.difference_hash)
            distance = min(average_distance, difference_distance)
            if distance <= threshold:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                pairs.append(
                    {
                        "a": left.id,
                        "b": right.id,
                        "average_hash_distance": average_distance,
                        "difference_hash_distance": difference_distance,
                        "distance": distance,
                    }
                )

    components = _connected_index_components(adjacency)
    near_groups: list[NearDuplicateGroup] = []
    for component in components:
        component_records = [records[index] for index in sorted(component, key=lambda i: records[i].id)]
        component_ids = {record.id for record in component_records}
        component_pairs = [pair for pair in pairs if pair["a"] in component_ids and pair["b"] in component_ids]
        if len(component_records) < 2 or not component_pairs:
            continue

        ordered = sorted(component_records, key=lambda record: (record.id, record.bundle_dir))
        splits = _unique_splits(ordered)
        near_groups.append(
            NearDuplicateGroup(
                kind=PERCEPTUAL_HASH,
                ids=[record.id for record in ordered],
                bundle_dirs=[record.bundle_dir for record in ordered],
                splits=splits,
                crosses_splits=len(splits) > 1,
                max_distance=max(int(pair["distance"]) for pair in component_pairs),
                pairs=sorted(component_pairs, key=lambda pair: (pair["distance"], pair["a"], pair["b"])),
            )
        )

    return sorted(near_groups, key=lambda group: (not group.crosses_splits, group.max_distance, group.ids))


def _connected_index_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    visited: set[int] = set()
    components: list[set[int]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            index = queue.popleft()
            component.add(index)
            for neighbor in sorted(adjacency[index]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _duplicate_value_map(records: list[DedupeSpriteRecord], *, value_name: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = getattr(record, value_name)
        if value is None:
            continue
        grouped[str(value)].append(record.bundle_dir)
    return {key: sorted(bundle_dirs) for key, bundle_dirs in sorted(grouped.items()) if len(bundle_dirs) >= 2}


def _unique_splits(records: list[DedupeSpriteRecord]) -> list[str]:
    return sorted({record.split for record in records if record.split is not None})


def _build_summary(
    *,
    total_records: int,
    records: list[DedupeSpriteRecord],
    failed: list[FailedDedupeRecord],
    exact_groups: list[DuplicateGroup],
    near_groups: list[NearDuplicateGroup],
    duplicate_ids: dict[str, list[str]],
    duplicate_source_paths: dict[str, list[str]],
) -> DedupeReportSummary:
    return DedupeReportSummary(
        total_records=total_records,
        analyzed_records=len(records),
        failed_records=len(failed),
        exact_source_duplicate_groups=sum(1 for group in exact_groups if group.kind == SOURCE_SHA256),
        exact_decoded_duplicate_groups=sum(1 for group in exact_groups if group.kind == DECODED_RGBA_SHA256),
        exact_bundle_duplicate_groups=sum(1 for group in exact_groups if group.kind == BUNDLE_CONTENT_SHA256),
        near_duplicate_groups=len(near_groups),
        cross_split_exact_groups=sum(1 for group in exact_groups if group.crosses_splits),
        cross_split_near_groups=sum(1 for group in near_groups if group.crosses_splits),
        duplicate_id_count=len(duplicate_ids),
        duplicate_source_path_count=len(duplicate_source_paths),
    )


def _critical_split_rows(report: DedupeReport) -> list[str]:
    rows: list[str] = []
    for group in report.exact_groups:
        if group.crosses_splits:
            rows.append(f"| {group.kind} | {', '.join(group.ids)} | {', '.join(group.splits)} |")
    for group in report.near_groups:
        if group.crosses_splits:
            rows.append(f"| {group.kind} | {', '.join(group.ids)} | {', '.join(group.splits)} |")
    return rows


def _duplicate_map_rows(values: dict[str, list[str]], *, empty_label: str) -> list[str]:
    if not values:
        return [f"| {empty_label} |  |"]
    return [f"| {key} | {', '.join(bundle_dirs)} |" for key, bundle_dirs in sorted(values.items())]


def _write_group_files(report: DedupeReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_duplicate_groups_file(
        directory / "exact_source_duplicates.txt",
        [group for group in report.exact_groups if group.kind == SOURCE_SHA256],
    )
    _write_duplicate_groups_file(
        directory / "exact_decoded_sprite_duplicates.txt",
        [group for group in report.exact_groups if group.kind == DECODED_RGBA_SHA256],
    )
    _write_duplicate_groups_file(
        directory / "exact_bundle_duplicates.txt",
        [group for group in report.exact_groups if group.kind == BUNDLE_CONTENT_SHA256],
    )
    _write_near_groups_file(directory / "near_duplicates.txt", report.near_groups)
    _write_duplicate_groups_file(
        directory / "cross_split_exact_duplicates.txt",
        [group for group in report.exact_groups if group.crosses_splits],
    )
    _write_near_groups_file(
        directory / "cross_split_near_duplicates.txt",
        [group for group in report.near_groups if group.crosses_splits],
    )
    _write_duplicate_map_file(directory / "duplicate_ids.txt", report.duplicate_ids, title="Duplicate IDs")
    _write_duplicate_map_file(
        directory / "duplicate_source_paths.txt",
        report.duplicate_source_paths,
        title="Duplicate source paths",
    )


def _write_duplicate_groups_file(path: Path, groups: list[DuplicateGroup]) -> None:
    lines = [f"# {path.stem}", ""]
    if not groups:
        lines.append("No groups.")
    for group in groups:
        lines.append(f"# {group.kind} {group.key}")
        lines.append(f"splits: {', '.join(group.splits) if group.splits else 'none'}")
        for sprite_id, bundle_dir in zip(group.ids, group.bundle_dirs):
            lines.append(f"{sprite_id}\t{bundle_dir}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_near_groups_file(path: Path, groups: list[NearDuplicateGroup]) -> None:
    lines = [f"# {path.stem}", ""]
    if not groups:
        lines.append("No groups.")
    for group in groups:
        lines.append(f"# {group.kind} max_distance={group.max_distance}")
        lines.append(f"splits: {', '.join(group.splits) if group.splits else 'none'}")
        for sprite_id, bundle_dir in zip(group.ids, group.bundle_dirs):
            lines.append(f"{sprite_id}\t{bundle_dir}")
        for pair in group.pairs:
            lines.append(f"pair\t{pair['a']}\t{pair['b']}\tdistance={pair['distance']}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_duplicate_map_file(path: Path, values: dict[str, list[str]], *, title: str) -> None:
    lines = [f"# {title}", ""]
    if not values:
        lines.append("No duplicates.")
    for key, bundle_dirs in sorted(values.items()):
        lines.append(f"# {key}")
        lines.extend(bundle_dirs)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _dataset_name(dataset_path: Path) -> str:
    path = Path(dataset_path)
    if path.is_file():
        return path.parent.name
    return path.name


def _output_dir(options: DedupeReportOptions) -> Path:
    if options.output_dir is not None:
        return Path(options.output_dir)
    dataset_path = Path(options.dataset_path)
    if dataset_path.is_file():
        return dataset_path.parent / "dedupe_report"
    return dataset_path / "dedupe_report"


def _options_dict(options: DedupeReportOptions) -> dict[str, Any]:
    data = asdict(options)
    data["dataset_path"] = str(options.dataset_path)
    data["output_dir"] = None if options.output_dir is None else str(options.output_dir)
    return data


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _parse_args() -> DedupeReportOptions:
    parser = argparse.ArgumentParser(description="Generate duplicate diagnostics for a SpriteBundle dataset.")
    parser.add_argument("--dataset", required=True, dest="dataset_path", type=Path)
    parser.add_argument("--output", dest="output_dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--category", dest="filter_category")
    parser.add_argument("--split", dest="filter_split")
    parser.add_argument("--no-markdown", action="store_false", dest="include_markdown")
    parser.add_argument("--no-json", action="store_false", dest="include_json")
    parser.add_argument("--no-group-files", action="store_false", dest="write_group_files")
    parser.add_argument("--no-near-duplicate", action="store_false", dest="near_duplicate")
    parser.add_argument("--near-threshold", type=int, default=8, dest="near_duplicate_threshold")
    parser.add_argument("--fail-on-load-error", action="store_true")
    args = parser.parse_args()
    return DedupeReportOptions(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_items=args.max_items,
        filter_category=args.filter_category,
        filter_split=args.filter_split,
        include_markdown=args.include_markdown,
        include_json=args.include_json,
        write_group_files=args.write_group_files,
        near_duplicate=args.near_duplicate,
        near_duplicate_threshold=args.near_duplicate_threshold,
        fail_on_load_error=args.fail_on_load_error,
    )


def main() -> None:
    options = _parse_args()
    report = create_dedupe_report(options)
    output_dir = _output_dir(options)
    print(f"Dataset: {report.dataset_name}")
    print(f"Analyzed: {report.summary.analyzed_records} / {report.summary.total_records}")
    print(f"Failed: {report.summary.failed_records}")
    print(f"Exact decoded duplicate groups: {report.summary.exact_decoded_duplicate_groups}")
    print(f"Near-duplicate groups: {report.summary.near_duplicate_groups}")
    print(f"Cross-split exact groups: {report.summary.cross_split_exact_groups}")
    print(f"Cross-split near groups: {report.summary.cross_split_near_groups}")
    if options.include_json:
        print(f"JSON: {output_dir / 'dedupe_report.json'}")
    if options.include_markdown:
        print(f"Markdown: {output_dir / 'dedupe_report.md'}")


if __name__ == "__main__":
    main()

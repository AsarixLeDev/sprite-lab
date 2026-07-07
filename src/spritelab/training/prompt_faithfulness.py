"""Dataset-grounded prompt/object faithfulness diagnostics for generated sprites."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.color_names import color_name
from spritelab.training.data import read_jsonl
from spritelab.training.prompt_sensitivity import COLOR_WORDS, discover_prompt_pairs, pairwise_image_metrics
from spritelab.training.source_match_review import compute_source_match_metrics, load_source_sprite_index

SCHEMA_VERSION = "prompt_faithfulness_v1.0"
SPRITE_SIZE = 32


@dataclass(frozen=True)
class PromptFaithfulnessConfig:
    generated: Path
    prompts: Path | None
    dataset: Path
    out: Path
    out_json: Path
    max_sources: int | None = None
    source_selection: str = "auto"


SOURCE_SELECTION_MODES = ("all", "deterministic_first_n", "deterministic_balanced")


def run_prompt_faithfulness(config: PromptFaithfulnessConfig) -> dict[str, Any]:
    generated_dir = Path(config.generated)
    generated_records = _read_jsonl(generated_dir / "generated_manifest.jsonl")
    prompt_records = _read_jsonl(config.prompts) if config.prompts is not None and Path(config.prompts).is_file() else []
    prompts_by_id = {str(record.get("prompt_id", "")): record for record in prompt_records}
    manifest_path = Path(config.dataset) / "training_manifest.jsonl"
    source_index = load_source_sprite_index(config.dataset, manifest_path)
    mode, requested_n = _resolve_source_selection_mode(config.max_sources, config.source_selection)
    selected_ids = _select_source_ids(source_index, mode=mode, max_sources=requested_n)
    source_candidates = _source_candidates(source_index, selected_ids)
    used_ids = [str(candidate["sprite_id"]) for candidate in source_candidates]
    source_selection = _source_selection_metadata(
        source_index,
        selected_ids=used_ids,
        mode=mode,
        max_sources=config.max_sources,
    )
    source_categories_by_object = _source_categories_by_object(read_jsonl(manifest_path))

    samples: list[dict[str, Any]] = []
    images: dict[str, Image.Image] = {}
    for record in generated_records:
        sample_id = str(record.get("sample_id", ""))
        image = _open_generated_image(generated_dir, record)
        images[sample_id] = image
        prompt_record = prompts_by_id.get(str(record.get("prompt_id", "")), {})
        samples.append(
            _review_sample(
                record=record,
                prompt_record=prompt_record,
                image=image,
                source_candidates=source_candidates,
                source_categories_by_object=source_categories_by_object,
            )
        )

    silhouette_clusters = _clusters(samples, "alpha_silhouette_hash")
    structural_clusters = _clusters(samples, "structural_hash")
    color_clusters = _clusters(samples, "color_histogram_signature")
    pair_separability = _prompt_pair_separability(prompt_records, samples, images)
    color_prompt_summary = _color_prompt_summary(samples)
    object_summary = _object_family_summary(samples)
    collapse_summary = _collapse_summary(samples)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated": str(generated_dir),
        "prompts": None if config.prompts is None else str(config.prompts),
        "dataset": str(config.dataset),
        "sample_count": len(samples),
        "source_selection": source_selection,
        "nearest_source_summary": _nearest_source_summary(samples),
        # Retrieval-based: category of the nearest source sprite, not a trained classifier.
        "nearest_source_category_consistency_rate": _rate(samples, "category_consistent"),
        # Kept as a backwards-compatible alias of the nearest-source metric above.
        "category_consistency_rate": _rate(samples, "category_consistent"),
        "color_consistency_rate": _rate(samples, "color_consistent"),
        "shape_bbox_consistency_rate": _rate(samples, "shape_bbox_consistent"),
        "repeated_silhouette_rate": _cluster_member_rate(silhouette_clusters),
        "nearest_neighbor_duplicate_rate": _nearest_neighbor_duplicate_rate(samples),
        "generic_potion_collapse_rate": collapse_summary["generic_potion_collapse_rate"],
        "generic_flame_collapse_rate": collapse_summary["generic_flame_collapse_rate"],
        "generic_blob_collapse_rate": collapse_summary["generic_blob_collapse_rate"],
        "top_repeated_generated_silhouettes": silhouette_clusters[:10],
        "top_repeated_structural_hashes": structural_clusters[:10],
        "top_color_histogram_clusters": color_clusters[:10],
        "collapsed_prompt_families": _collapsed_prompt_families(silhouette_clusters),
        "prompt_pair_separability": pair_separability,
        "color_prompts_worked": color_prompt_summary["worked"],
        "color_prompts_failed": color_prompt_summary["failed"],
        "object_families_worst_faithfulness": object_summary[:15],
        "samples": samples,
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }

    out_json = Path(config.out_json)
    out_md = Path(config.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(format_prompt_faithfulness_markdown(report), encoding="utf-8")
    mapping_path = out_json.with_name("prompt_faithfulness_mapping.jsonl")
    mapping_path.write_text(
        "".join(json.dumps(_sample_mapping(sample), sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    report["mapping"] = str(mapping_path)
    candidate_ids_path = out_json.with_name("source_candidate_ids.jsonl")
    candidate_ids_path.write_text(
        "".join(
            json.dumps(_source_candidate_row(source_index, sprite_id), sort_keys=True) + "\n"
            for sprite_id in used_ids
        ),
        encoding="utf-8",
    )
    report["source_candidate_ids"] = str(candidate_ids_path)
    out_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def format_prompt_faithfulness_markdown(report: Mapping[str, Any]) -> str:
    nearest = report.get("nearest_source_summary") if isinstance(report.get("nearest_source_summary"), Mapping) else {}
    selection = report.get("source_selection") if isinstance(report.get("source_selection"), Mapping) else {}
    lines = [
        "# Prompt Faithfulness Report",
        "",
        f"Generated: `{report.get('generated', '')}`",
        f"Samples: {int(report.get('sample_count') or 0)}",
        "",
        "## Source Selection",
        "",
        f"- Mode: `{selection.get('mode', 'unknown')}`",
        f"- Sources used / total: {int(selection.get('source_count_used') or 0)} / {int(selection.get('source_count_total') or 0)}",
        f"- Source candidate hash: `{selection.get('source_candidate_hash', '')}`",
        "",
        "## Summary",
        "",
        f"- Mean nearest-source distance: {_fmt(nearest.get('mean_distance'))}",
        f"- Nearest-source category consistency: {_fmt(report.get('nearest_source_category_consistency_rate', report.get('category_consistency_rate')))}",
        f"- Color consistency heuristic: {_fmt(report.get('color_consistency_rate'))}",
        f"- Shape/bbox consistency heuristic: {_fmt(report.get('shape_bbox_consistency_rate'))}",
        f"- Repeated silhouette rate: {_fmt(report.get('repeated_silhouette_rate'))}",
        f"- Nearest-neighbor duplicate rate: {_fmt(report.get('nearest_neighbor_duplicate_rate'))}",
        f"- Generic potion collapse rate: {_fmt(report.get('generic_potion_collapse_rate'))}",
        f"- Generic flame collapse rate: {_fmt(report.get('generic_flame_collapse_rate'))}",
        f"- Generic blob collapse rate: {_fmt(report.get('generic_blob_collapse_rate'))}",
        "",
        "## Top Repeated Generated Silhouettes",
        "",
        "| Count | Prompts | Nearest sources |",
        "|---:|---|---|",
    ]
    for cluster in report.get("top_repeated_generated_silhouettes") or []:
        if not isinstance(cluster, Mapping) or int(cluster.get("count") or 0) <= 1:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(cluster.get("count") or 0)),
                    _md_escape(", ".join(str(v) for v in cluster.get("prompts", [])[:5])),
                    _md_escape(", ".join(str(v) for v in cluster.get("nearest_source_objects", [])[:5])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prompt Pairs",
            "",
            "| Pair | Difference | Near duplicate |",
            "|---|---:|---|",
        ]
    )
    for pair in report.get("prompt_pair_separability", {}).get("pairs", []) if isinstance(report.get("prompt_pair_separability"), Mapping) else []:
        metrics = pair.get("metrics") if isinstance(pair.get("metrics"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(str(pair.get("pair_id", ""))),
                    _fmt(metrics.get("combined_difference_score")),
                    "yes" if metrics.get("near_duplicate") else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Color Prompts That Failed",
            "",
            "| Sample | Prompt | Expected | Generated | Nearest source |",
            "|---|---|---|---|---|",
        ]
    )
    for sample in report.get("color_prompts_failed") or []:
        if not isinstance(sample, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(sample.get('sample_id', '')))}`",
                    _md_escape(str(sample.get("prompt", ""))),
                    _md_escape(", ".join(sample.get("prompt_colors", []) or [])),
                    _md_escape(", ".join(sample.get("generated_colors", []) or [])),
                    _md_escape(str(sample.get("nearest_source_object_name", ""))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Worst Object Families", "", "| Object | Count | Mean distance | Consistency |", "|---|---:|---:|---:|"])
    for item in report.get("object_families_worst_faithfulness") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(str(item.get("object", ""))),
                    str(int(item.get("count") or 0)),
                    _fmt(item.get("mean_nearest_source_distance")),
                    _fmt(item.get("faithfulness_rate")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _review_sample(
    *,
    record: Mapping[str, Any],
    prompt_record: Mapping[str, Any],
    image: Image.Image,
    source_candidates: Sequence[Mapping[str, Any]],
    source_categories_by_object: Mapping[str, set[str]],
) -> dict[str, Any]:
    prompt = str(record.get("prompt") or prompt_record.get("prompt") or "")
    prompt_object = _prompt_object(record, prompt_record)
    prompt_colors = _prompt_colors(record, prompt_record, prompt)
    nearest = _nearest_source(image, source_candidates)
    nearest_meta = nearest.get("metadata") if isinstance(nearest.get("metadata"), Mapping) else {}
    generated_colors = _dominant_color_names(image)
    structural = _structural_stats(image)
    expected_categories = source_categories_by_object.get(prompt_object, set()) if prompt_object else set()
    category_consistent = None if not expected_categories else str(nearest_meta.get("category") or "") in expected_categories
    color_consistent = None if not prompt_colors else bool(set(prompt_colors) & set(generated_colors))
    shape_bbox_consistent = _shape_bbox_consistent(prompt_object, nearest.get("metrics", {}))
    sample = {
        "sample_id": str(record.get("sample_id") or ""),
        "sample_filename": _sample_filename(record),
        "prompt_id": str(record.get("prompt_id") or prompt_record.get("prompt_id") or ""),
        "prompt": prompt,
        "prompt_category": str(record.get("category") or prompt_record.get("category") or ""),
        "prompt_object": prompt_object,
        "prompt_colors": prompt_colors,
        "generated_colors": generated_colors,
        "nearest_source_sprite_id": str(nearest.get("sprite_id") or ""),
        "nearest_source_object_name": str(nearest_meta.get("object_name") or ""),
        "nearest_source_category": str(nearest_meta.get("category") or ""),
        "nearest_source_distance": nearest.get("distance"),
        "nearest_source_metrics": nearest.get("metrics", {}),
        "category_consistent": category_consistent,
        "color_consistent": color_consistent,
        "shape_bbox_consistent": shape_bbox_consistent,
        "alpha_silhouette_hash": _alpha_silhouette_hash(image),
        "structural_hash": _structural_hash(image),
        "color_histogram_signature": _color_histogram_signature(image),
        "generic_blob_like": structural["generic_blob_like"],
        "structural_stats": structural,
        "seed": record.get("seed"),
        "noise_seed": record.get("noise_seed"),
        "conditioning_mode": str(record.get("conditioning_mode") or ""),
    }
    sample["faithfulness_ok"] = _sample_faithfulness_ok(sample)
    return sample


def _source_candidates(
    source_index: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sprite_id in selected_ids:
        source = source_index.get(sprite_id)
        if not isinstance(source, Mapping):
            continue
        source_image = source.get("image")
        if not isinstance(source_image, Image.Image):
            continue
        candidates.append(
            {
                "sprite_id": sprite_id,
                "metadata": source.get("metadata", {}),
                "image": source_image,
                "feature": _image_prefilter_feature(source_image),
            }
        )
    return candidates


def _resolve_source_selection_mode(max_sources: int | None, source_selection: str) -> tuple[str, int | None]:
    """Return the (resolved_mode, requested_count) for source-candidate selection.

    ``max_sources`` of ``None`` or ``<= 0`` always means "all sources". Selection is
    deterministic in every mode: candidates are ordered by stable sprite ID before any
    truncation, so repeated runs on the same dataset use an identical candidate set.
    """

    mode = str(source_selection or "auto").strip().lower()
    requested = None if max_sources is None else int(max_sources)
    if mode in ("", "auto"):
        mode = "all" if (requested is None or requested <= 0) else "deterministic_first_n"
    if mode not in SOURCE_SELECTION_MODES:
        raise ValueError(f"Unknown source_selection mode: {source_selection!r} (expected one of {SOURCE_SELECTION_MODES} or 'auto')")
    if requested is None or requested <= 0:
        mode = "all"
    return mode, requested


def _select_source_ids(
    source_index: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    max_sources: int | None,
) -> list[str]:
    ordered = sorted(source_index.keys())
    if mode == "all" or max_sources is None or int(max_sources) <= 0:
        return ordered
    limit = max(0, int(max_sources))
    if mode == "deterministic_balanced":
        return _balanced_source_ids(source_index, ordered, limit)
    return ordered[:limit]


def _balanced_source_ids(
    source_index: Mapping[str, Mapping[str, Any]],
    ordered_ids: Sequence[str],
    limit: int,
) -> list[str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for sprite_id in ordered_ids:
        metadata = source_index[sprite_id].get("metadata") if isinstance(source_index[sprite_id], Mapping) else {}
        category = str((metadata or {}).get("category") or "unknown")
        by_category[category].append(sprite_id)
    categories = sorted(by_category)
    selected: list[str] = []
    while len(selected) < limit and any(by_category[category] for category in categories):
        for category in categories:
            if len(selected) >= limit:
                break
            if by_category[category]:
                selected.append(by_category[category].pop(0))
    return selected


def _source_selection_metadata(
    source_index: Mapping[str, Mapping[str, Any]],
    *,
    selected_ids: Sequence[str],
    mode: str,
    max_sources: int | None,
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    for sprite_id in selected_ids:
        source = source_index.get(sprite_id)
        metadata = source.get("metadata") if isinstance(source, Mapping) else {}
        category_counts[str((metadata or {}).get("category") or "unknown")] += 1
    # Hash the sorted candidate set so the hash changes iff the candidate set changes,
    # independent of the order candidates are visited in.
    hash_input = "\n".join(sorted(str(sprite_id) for sprite_id in selected_ids))
    candidate_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    return {
        "mode": mode,
        "max_sources": None if max_sources is None else int(max_sources),
        "source_count_total": len(source_index),
        "source_count_used": len(selected_ids),
        "source_candidate_hash": candidate_hash,
        "source_category_counts": dict(sorted(category_counts.items())),
    }


def _source_candidate_row(source_index: Mapping[str, Mapping[str, Any]], sprite_id: str) -> dict[str, Any]:
    source = source_index.get(sprite_id)
    metadata = source.get("metadata") if isinstance(source, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "sprite_id": str(sprite_id),
        "category": str(metadata.get("category") or ""),
        "object_name": str(metadata.get("object_name") or ""),
        "base_object": str(metadata.get("base_object") or ""),
        "split": str(metadata.get("split") or ""),
    }


def _nearest_source(image: Image.Image, source_candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    if not source_candidates:
        return {"sprite_id": "", "metadata": {}, "metrics": {}, "distance": None}
    generated_feature = _image_prefilter_feature(image)
    features = np.stack([np.asarray(candidate["feature"], dtype=np.float32) for candidate in source_candidates])
    distances = np.mean(np.abs(features - generated_feature[None, :]), axis=1)
    limit = min(32, len(source_candidates))
    candidate_indices = np.argsort(distances)[:limit]
    for index in candidate_indices:
        source = source_candidates[int(index)]
        source_image = source.get("image")
        if not isinstance(source_image, Image.Image):
            continue
        metrics = compute_source_match_metrics(image, source_image)
        distance = float(metrics.get("combined_difference_score") or 0.0)
        if best is None or distance < float(best["distance"]):
            best = {
                "sprite_id": str(source.get("sprite_id") or ""),
                "metadata": source.get("metadata", {}),
                "metrics": metrics,
                "distance": distance,
            }
    return best or {"sprite_id": "", "metadata": {}, "metrics": {}, "distance": None}


def _image_prefilter_feature(image: Image.Image) -> np.ndarray:
    small = image.convert("RGBA").resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    alpha_weight = arr[..., 3:4]
    rgb = arr[..., :3] * alpha_weight
    return np.concatenate([rgb, alpha_weight], axis=-1).reshape(-1).astype(np.float32, copy=False)


def _prompt_object(record: Mapping[str, Any], prompt_record: Mapping[str, Any]) -> str:
    for source in (record, prompt_record):
        for key in ("base_object", "object_name"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
        target = source.get("target_semantics") if isinstance(source.get("target_semantics"), Mapping) else {}
        for key in ("base_object", "object_name", "open_name"):
            value = str(target.get(key) or "").strip()
            if value:
                return value
    return ""


def _prompt_colors(record: Mapping[str, Any], prompt_record: Mapping[str, Any], prompt: str) -> list[str]:
    colors: list[str] = []
    for source in (record, prompt_record):
        value = source.get("colors")
        if isinstance(value, str):
            colors.append(value)
        elif isinstance(value, Sequence):
            colors.extend(str(item) for item in value)
        target = source.get("target_semantics") if isinstance(source.get("target_semantics"), Mapping) else {}
        attrs = target.get("attributes") if isinstance(target.get("attributes"), Mapping) else {}
        colors.extend(str(item) for item in attrs.get("colors") or [])
    tokens = [token.strip(".,:;!?()[]{}").lower() for token in prompt.replace("_", " ").split()]
    colors.extend(token for token in tokens if token in COLOR_WORDS)
    result: list[str] = []
    seen: set[str] = set()
    for color in colors:
        normalized = str(color).lower().replace("grey", "gray")
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _dominant_color_names(image: Image.Image, *, max_colors: int = 4) -> list[str]:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    visible = arr[..., 3] > 0
    if not bool(np.any(visible)):
        return []
    colors, counts = np.unique(arr[..., :3][visible], axis=0, return_counts=True)
    names: Counter[str] = Counter()
    for rgb, count in zip(colors, counts, strict=False):
        names[color_name(rgb)] += int(count)
    return [name for name, _ in names.most_common(max_colors)]


def _structural_stats(image: Image.Image) -> dict[str, Any]:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = arr[..., 3] > 0
    area = int(np.count_nonzero(mask))
    bbox = _bbox(mask)
    bbox_area = _bbox_area(bbox)
    fill = area / float(bbox_area) if bbox_area else 0.0
    edge = _edge_map(mask)
    edge_density = int(np.count_nonzero(edge)) / float(area) if area else 0.0
    return {
        "alpha_area": area,
        "bbox_area": bbox_area,
        "bbox_fill_ratio": fill,
        "alpha_edge_density": edge_density,
        "generic_blob_like": bool(area > 24 and fill >= 0.72 and edge_density <= 0.32),
    }


def _shape_bbox_consistent(prompt_object: str, metrics: Mapping[str, Any]) -> bool | None:
    if not prompt_object:
        return None
    center = metrics.get("bbox_center_distance")
    area_diff = float(metrics.get("bbox_area_difference") or 0.0)
    if center is None:
        return area_diff <= 0.20
    return float(center) <= 6.0 and area_diff <= 0.30


def _sample_faithfulness_ok(sample: Mapping[str, Any]) -> bool:
    checks = [
        sample.get("category_consistent"),
        sample.get("color_consistent"),
        sample.get("shape_bbox_consistent"),
    ]
    known = [bool(value) for value in checks if value is not None]
    if not known:
        return False
    return sum(1 for value in known if value) >= max(1, math.ceil(len(known) / 2))


def _clusters(samples: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get(key) or "")].append(sample)
    clusters: list[dict[str, Any]] = []
    for value, rows in groups.items():
        if not value:
            continue
        clusters.append(
            {
                "hash": value,
                "count": len(rows),
                "sample_ids": [str(row.get("sample_id") or "") for row in rows],
                "prompt_ids": [str(row.get("prompt_id") or "") for row in rows],
                "prompts": [str(row.get("prompt") or "") for row in rows],
                "prompt_objects": sorted({str(row.get("prompt_object") or "") for row in rows if row.get("prompt_object")}),
                "nearest_source_objects": sorted(
                    {str(row.get("nearest_source_object_name") or "") for row in rows if row.get("nearest_source_object_name")}
                ),
            }
        )
    clusters.sort(key=lambda item: (-int(item["count"]), str(item["hash"])))
    return clusters


def _prompt_pair_separability(
    prompt_records: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    images: Mapping[str, Image.Image],
) -> dict[str, Any]:
    if not prompt_records:
        return {"pair_count": 0, "pairs": [], "near_duplicate_rate": 0.0}
    by_prompt_id = {str(sample.get("prompt_id") or ""): sample for sample in samples}
    pairs = discover_prompt_pairs(prompt_records, max_pairs=12)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        left = pair.get("a") if isinstance(pair.get("a"), Mapping) else {}
        right = pair.get("b") if isinstance(pair.get("b"), Mapping) else {}
        sample_a = by_prompt_id.get(str(left.get("prompt_id") or ""))
        sample_b = by_prompt_id.get(str(right.get("prompt_id") or ""))
        if sample_a is None or sample_b is None:
            continue
        image_a = images.get(str(sample_a.get("sample_id") or ""))
        image_b = images.get(str(sample_b.get("sample_id") or ""))
        if image_a is None or image_b is None:
            continue
        metrics = pairwise_image_metrics(image_a, image_b)
        rows.append(
            {
                "pair_id": str(pair.get("pair_id") or ""),
                "prompt_a": str(left.get("prompt") or ""),
                "prompt_b": str(right.get("prompt") or ""),
                "sample_id_a": str(sample_a.get("sample_id") or ""),
                "sample_id_b": str(sample_b.get("sample_id") or ""),
                "metrics": metrics,
            }
        )
    near = sum(1 for row in rows if row.get("metrics", {}).get("near_duplicate"))
    return {"pair_count": len(rows), "pairs": rows, "near_duplicate_rate": near / float(len(rows)) if rows else 0.0}


def _color_prompt_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    color_samples = [sample for sample in samples if sample.get("prompt_colors")]
    worked = [dict(sample) for sample in color_samples if sample.get("color_consistent") is True][:20]
    failed = [dict(sample) for sample in color_samples if sample.get("color_consistent") is False][:20]
    return {"worked": worked, "failed": failed}


def _object_family_summary(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get("prompt_object") or "unknown")].append(sample)
    rows: list[dict[str, Any]] = []
    for object_name, group in groups.items():
        distances = [
            float(sample["nearest_source_distance"])
            for sample in group
            if isinstance(sample.get("nearest_source_distance"), (int, float))
        ]
        rows.append(
            {
                "object": object_name,
                "count": len(group),
                "mean_nearest_source_distance": float(statistics.fmean(distances)) if distances else None,
                "faithfulness_rate": sum(1 for sample in group if sample.get("faithfulness_ok")) / float(len(group)),
            }
        )
    rows.sort(
        key=lambda item: (
            float(item["faithfulness_rate"]),
            -(float(item["mean_nearest_source_distance"]) if item["mean_nearest_source_distance"] is not None else 0.0),
            str(item["object"]),
        )
    )
    return rows


def _collapse_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = len(samples)
    if not count:
        return {
            "generic_potion_collapse_rate": 0.0,
            "generic_flame_collapse_rate": 0.0,
            "generic_blob_collapse_rate": 0.0,
        }

    def object_has(sample: Mapping[str, Any], needles: tuple[str, ...], *, nearest: bool) -> bool:
        key = "nearest_source_object_name" if nearest else "prompt_object"
        value = str(sample.get(key) or "").lower()
        return any(needle in value for needle in needles)

    potion = sum(1 for sample in samples if object_has(sample, ("potion", "bottle", "vial"), nearest=True) and not object_has(sample, ("potion", "bottle", "vial"), nearest=False))
    flame = sum(1 for sample in samples if object_has(sample, ("flame", "fire", "fireball"), nearest=True) and not object_has(sample, ("flame", "fire", "fireball"), nearest=False))
    blob = sum(1 for sample in samples if sample.get("generic_blob_like") and not object_has(sample, ("potion", "flame", "fire"), nearest=False))
    return {
        "generic_potion_collapse_rate": potion / float(count),
        "generic_flame_collapse_rate": flame / float(count),
        "generic_blob_collapse_rate": blob / float(count),
    }


def _collapsed_prompt_families(clusters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        objects = cluster.get("prompt_objects") if isinstance(cluster.get("prompt_objects"), list) else []
        if int(cluster.get("count") or 0) >= 2 and len(set(objects)) >= 2:
            rows.append(dict(cluster))
    return rows[:20]


def _nearest_source_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances = [
        float(sample["nearest_source_distance"])
        for sample in samples
        if isinstance(sample.get("nearest_source_distance"), (int, float))
    ]
    nearest_objects = Counter(str(sample.get("nearest_source_object_name") or "") for sample in samples)
    return {
        "mean_distance": float(statistics.fmean(distances)) if distances else None,
        "median_distance": float(statistics.median(distances)) if distances else None,
        "top_nearest_source_objects": dict(nearest_objects.most_common(20)),
    }


def _nearest_neighbor_duplicate_rate(samples: Sequence[Mapping[str, Any]]) -> float:
    if len(samples) < 2:
        return 0.0
    duplicate_members: set[str] = set()
    for a, b in itertools.combinations(samples, 2):
        if a.get("alpha_silhouette_hash") == b.get("alpha_silhouette_hash") and a.get("color_histogram_signature") == b.get("color_histogram_signature"):
            duplicate_members.add(str(a.get("sample_id") or ""))
            duplicate_members.add(str(b.get("sample_id") or ""))
    return len(duplicate_members) / float(len(samples))


def _source_categories_by_object(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key in ("object_name", "base_object"):
            value = str(row.get(key) or "").strip()
            if value:
                result[value].add(str(row.get("category") or "unknown"))
    return result


def _cluster_member_rate(clusters: Sequence[Mapping[str, Any]]) -> float:
    total = sum(int(cluster.get("count") or 0) for cluster in clusters)
    repeated = sum(int(cluster.get("count") or 0) for cluster in clusters if int(cluster.get("count") or 0) > 1)
    return repeated / float(total) if total else 0.0


def _rate(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [bool(sample.get(key)) for sample in samples if sample.get(key) is not None]
    return sum(1 for value in values if value) / float(len(values)) if values else None


def _sample_mapping(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "sample_filename": sample.get("sample_filename"),
        "prompt": sample.get("prompt"),
        "prompt_id": sample.get("prompt_id"),
        "seed": sample.get("seed"),
        "noise_seed": sample.get("noise_seed"),
        "conditioning": sample.get("conditioning_mode"),
        "nearest_source_sprite_id": sample.get("nearest_source_sprite_id"),
        "nearest_source_object_name": sample.get("nearest_source_object_name"),
        "nearest_source_category": sample.get("nearest_source_category"),
        "nearest_source_distance": sample.get("nearest_source_distance"),
    }


def _alpha_silhouette_hash(image: Image.Image) -> str:
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3] > 0
    return hashlib.sha1(np.packbits(alpha.astype(np.uint8)).tobytes()).hexdigest()[:16]


def _structural_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    bits = arr >= float(arr.mean())
    return hashlib.sha1(np.packbits(bits.astype(np.uint8)).tobytes()).hexdigest()[:16]


def _color_histogram_signature(image: Image.Image) -> str:
    colors = _dominant_color_names(image, max_colors=5)
    return "|".join(colors) or "transparent"


def _sample_filename(record: Mapping[str, Any]) -> str:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    return str(paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba") or "")


def _open_generated_image(generated_dir: Path, record: Mapping[str, Any]) -> Image.Image:
    rel = _sample_filename(record)
    if not rel:
        raise ValueError(f"{record.get('sample_id', '')}: generated record has no image path")
    image = Image.open(generated_dir / rel).convert("RGBA")
    image.load()
    return image.copy()


def _read_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            rows.append({"prompt": value, "prompt_id": f"prompt_{line_no:04d}"})
        elif isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("prompt", str(row.get("caption") or ""))
            row.setdefault("prompt_id", f"prompt_{line_no:04d}")
            rows.append(row)
    return rows


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))


def _edge_map(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    full = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return np.asarray(mask, dtype=bool) & ~full


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dataset-grounded prompt faithfulness diagnostics.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-sources", type=int, help="0 or negative means use all sources.")
    parser.add_argument(
        "--source-selection",
        default="auto",
        choices=["auto", *SOURCE_SELECTION_MODES],
        help="Deterministic source-candidate selection strategy.",
    )
    parsed = parser.parse_args(argv)
    report = run_prompt_faithfulness(PromptFaithfulnessConfig(**vars(parsed)))
    print(f"Prompt faithfulness samples: {report['sample_count']}")
    print(f"Repeated silhouette rate: {_fmt(report.get('repeated_silhouette_rate'))}")


if __name__ == "__main__":
    main()

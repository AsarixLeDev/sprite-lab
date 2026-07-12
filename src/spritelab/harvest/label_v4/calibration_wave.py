"""Deterministic frozen 100-representative calibration-wave construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.filename_parser import parse_filename_semantics
from spritelab.harvest.label_v4.pipeline import infer_pack_context

CALIBRATION_WAVE_SCHEMA = "label_v4_audit_selection_v1"
AUDIT_FIELDS = (
    "canonical_object",
    "category",
    "domain",
    "role",
    "explicit_material",
    "surface_alias",
    "colors",
    "description",
)
SCORE_BUCKETS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20))


def build_calibration_wave1(
    candidate_manifest: str | Path,
    output_root: str | Path,
    *,
    target_size: int = 100,
    seed: int = 41,
) -> dict[str, Any]:
    source = Path(candidate_manifest).resolve()
    output = Path(output_root).resolve()
    before = _file_hash(source)
    candidates = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    representatives = [row for row in candidates if row.get("annotation_representative")]
    image_index = _imported_image_index(Path.cwd() / "harvest_runs")
    enriched = [_enrich(row, index, image_index) for index, row in enumerate(representatives)]
    selected = _select_grouped(enriched, target_size=target_size, seed=seed)
    if len(selected) != target_size:
        raise ValueError(f"could not construct exact {target_size}-record grouped audit; selected {len(selected)}")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "audit_manifest.jsonl", selected)
    distribution = _distribution(selected)
    manifest = {
        "schema_version": CALIBRATION_WAVE_SCHEMA,
        "audit_size": len(selected),
        "seed": seed,
        "source_manifest": str(source),
        "source_manifest_hash": before,
        "gold_labels_present": False,
        "calibrator_fitted": False,
        "records": selected,
        "distribution": distribution,
    }
    _write_json(output / "audit_manifest.json", manifest)
    report = {
        "schema_version": CALIBRATION_WAVE_SCHEMA,
        "candidate_representatives": len(representatives),
        "selected": len(selected),
        "distribution": distribution,
        "geometry_groups_preserved": _groups_preserved(selected, enriched, "variant_geometry_group"),
        "declared_variant_groups_preserved": _groups_preserved(selected, enriched, "declared_variant_group"),
        "selection_policy": "stratified_group_greedy_v1",
        "provisional_uncertainty_is_calibrated": False,
    }
    _write_json(output / "sampling_report.json", report)
    (output / "sampling_report.md").write_text(
        "# Calibration wave 1 sampling\n\n"
        f"Selected {len(selected)} of {len(representatives)} representatives. Geometry and declared-variant groups are atomic. "
        "Provisional uncertainty is sampling metadata only, never calibrated strength. No gold labels were created.\n\n"
        "```json\n" + json.dumps(distribution, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
        newline="\n",
    )
    readiness = _readiness(selected)
    _write_json(output / "calibration_readiness_report.json", readiness)
    (output / "calibration_readiness_report.md").write_text(
        "# Calibration readiness\n\nNo audited human truth exists yet, so no field is calibrator-ready and uncertainty monotonicity is unknown. "
        "The report lists the minimum additional reviewed field judgments needed before fitting.\n\n"
        "```json\n" + json.dumps(readiness, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "human_truth.jsonl").touch(exist_ok=True)
    freeze = {
        "schema_version": "label_v4_calibration_freeze_v1",
        "source_manifest_hash": before,
        "files": {
            path.name: _file_hash(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name not in {"freeze_manifest.json", "human_truth.jsonl"}
        },
        "audit_set_hash": _stable_hash([row["sprite_id"] for row in selected]),
        "frozen": True,
        "gold_labels_present": False,
        "append_only_truth_path": "human_truth.jsonl",
        "append_only_truth_excluded_from_frozen_hashes": True,
    }
    _write_json(output / "freeze_manifest.json", freeze)
    if _file_hash(source) != before:
        raise RuntimeError("frozen candidate manifest changed during calibration sampling")
    return {
        "selected": len(selected),
        "audit_set_hash": freeze["audit_set_hash"],
        "distribution": distribution,
        "source_unchanged": True,
        "output": str(output / "audit_manifest.jsonl"),
    }


def _enrich(row: Mapping[str, Any], index: int, images: Mapping[str, str]) -> dict[str, Any]:
    value = dict(row)
    deterministic = parse_filename_semantics(value, pack_context=infer_pack_context(value))
    generic = not bool(deterministic.canonical_object)
    score = _provisional_score(value, generic, index)
    bucket = next(f"{low}-{high}" for low, high in SCORE_BUCKETS if low <= score <= high)
    field_name = AUDIT_FIELDS[index % len(AUDIT_FIELDS)]
    path = "vlm_rich" if generic or deterministic.open_set_tokens else "deterministic_only"
    proposal = {
        "canonical_object": deterministic.canonical_object,
        "category": deterministic.category,
        "domain": deterministic.domain,
        "role": deterministic.role,
        "explicit_material": deterministic.explicit_material,
        "surface_alias": deterministic.surface_alias,
        "colors": {"filename_color_hints": list(deterministic.filename_color_hints)},
        "description": None,
    }
    field_proposals = {
        name: {"value": value_, "alternatives": [], "support": ["deterministic_filename_or_pack"]}
        for name, value_ in proposal.items()
    }
    label_quality = {
        "fields": {
            name: {
                "uncertainty_1_20": score,
                "calibration_state": "uncalibrated",
                "training_state": "pending_human_audit",
            }
            for name in field_proposals
        }
    }
    return {
        **value,
        "schema_version": CALIBRATION_WAVE_SCHEMA,
        "audit_id": f"wave1-{index:04d}-{value.get('sprite_id', '')}",
        "audit_focus_field": field_name,
        "field": field_name,
        "image_path": images.get(str(value.get("sprite_id", "")), ""),
        "normalized_proposal": proposal,
        "field_proposals": field_proposals,
        "label_quality": label_quality,
        "uncertainty_1_20": score,
        "uncertainty_bucket": bucket,
        "uncertainty_state": "provisional_uncalibrated",
        "inference_path": path,
        "conflict_disposition": "none" if not generic else "open_set_requires_review",
        "open_set_state": "open_set" if generic else "closed_named",
        "propagation_relation": "geometry_representative",
        "quality_status": value.get("suitability_status", "unknown"),
        "source_generation": "new_acquisition" if str(value.get("source_id", "")).startswith("acq_") else "legacy",
        "human_truth": None,
        "review_status": "not_started",
    }


def _provisional_score(row: Mapping[str, Any], generic: bool, index: int) -> int:
    # Deliberately spans all audit buckets; it is sampling metadata, not confidence.
    base = (index * 7 + int(float(row.get("quality_confidence", 0) or 0))) % 20 + 1
    if generic:
        return max(9, base)
    # Named deterministic candidates remain represented even at conservative highs.
    return base


def _select_grouped(rows: Sequence[Mapping[str, Any]], *, target_size: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        declared = str(row.get("declared_variant_group") or "")
        geometry = str(row.get("variant_geometry_group") or row.get("sprite_id"))
        key = f"declared:{declared}" if declared else f"geometry:{geometry}"
        groups.setdefault(key, []).append(dict(row))
    wanted = {f"{low}-{high}": 20 for low, high in SCORE_BUCKETS}
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    def rank(item: tuple[str, list[dict[str, Any]]]) -> tuple[float, str]:
        key, members = item
        score = 0.0
        for member in members:
            bucket = str(member["uncertainty_bucket"])
            score += 8.0 if wanted.get(bucket, 0) > 0 else 0.0
            score += 3.0 / (1 + source_counts[str(member.get("source_id", ""))])
            score += 2.0 / (1 + category_counts[str(member.get("broad_pack_type", ""))])
            score += 2.0 if member["inference_path"] == "deterministic_only" else 2.5
            score += (
                int(hashlib.sha256(f"{seed}:{member['sprite_id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            ) * 0.01
        return (-score / len(members), key)

    remaining = dict(groups)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < target_size:
        ordered = sorted(remaining.items(), key=rank)
        chosen_key = next((key for key, members in ordered if len(selected) + len(members) <= target_size), None)
        if chosen_key is None:
            break
        members = remaining.pop(chosen_key)
        selected.extend(members)
        for member in members:
            bucket = str(member["uncertainty_bucket"])
            wanted[bucket] = max(0, wanted.get(bucket, 0) - 1)
            source_counts[str(member.get("source_id", ""))] += 1
            category_counts[str(member.get("broad_pack_type", ""))] += 1
    return sorted(selected, key=lambda row: str(row["audit_id"]))


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = (
        "uncertainty_bucket",
        "audit_focus_field",
        "source_id",
        "pack_id",
        "broad_pack_type",
        "inference_path",
        "conflict_disposition",
        "open_set_state",
        "propagation_relation",
        "quality_status",
        "source_generation",
    )
    return {name: dict(sorted(Counter(str(row.get(name, "<missing>")) for row in rows).items())) for name in dimensions}


def _groups_preserved(selected: Sequence[Mapping[str, Any]], all_rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    chosen = {str(row.get("sprite_id")) for row in selected}
    groups: dict[str, set[str]] = {}
    for row in all_rows:
        key = str(row.get(field) or "")
        if key:
            groups.setdefault(key, set()).add(str(row.get("sprite_id")))
    return all(not (members & chosen) or members <= chosen for members in groups.values())


def _readiness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support = Counter(str(row.get("audit_focus_field")) for row in rows)
    return {
        "schema_version": "label_v4_calibration_readiness_v1",
        "audited_truth_rows": 0,
        "field_manifest_support": dict(sorted(support.items())),
        "fields_with_audit_support": [],
        "unsupported_strata": ["all strata pending human review"],
        "minimum_additional_samples_needed": {field: max(0, 30 - 0) for field in AUDIT_FIELDS},
        "uncertainty_ordering_monotonic": None,
        "fit_final_calibrator": False,
        "strong_scores_allowed": False,
        "reason": "No human truth has been recorded.",
    }


def _imported_image_index(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*/imported.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image = Path(str(row.get("final_png_path") or ""))
            if image and not image.is_absolute():
                image = Path.cwd() / image
            if image.is_file():
                result.setdefault(str(row.get("sprite_id", "")), str(image.resolve()))
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

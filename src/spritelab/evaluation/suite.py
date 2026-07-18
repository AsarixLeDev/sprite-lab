"""Benchmark orchestration, reports, promotion gates, and paired comparisons."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import random
import shutil
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.evaluation.candidate_bundle import (
    canonical_sha256,
    decoded_rgba_sha256,
    file_sha256,
    write_candidate_bundle,
)
from spritelab.evaluation.conditional import FIELDS, score_conditions
from spritelab.evaluation.memorization import (
    COMPARISON_METHOD,
    COMPARISON_PARAMETERS,
    COMPARISON_PARAMETERS_SHA256,
    DETECTOR_POLICY_SHA256,
    DETECTOR_POLICY_VERSION,
    MemorizationMachineStatus,
    TrainingImage,
    detector_policy_record,
    evaluate_memorization_outcome,
    load_training_images,
    resolve_training_context_identities,
    retrieve_neighbors,
    suspicious_kind,
)
from spritelab.evaluation.metric_definitions import IncompatibleMetricDefinitions, metric_definition_identity
from spritelab.evaluation.metrics import batch_metrics, duplicate_groups, score_image
from spritelab.evaluation.strict_json import strict_json_loads
from spritelab.harvest.label_v4.training_quality import (
    evaluation_uncertainty_report,
    extract_training_quality,
    uncertainty_correlation_report,
)

DEFAULT_GATES: dict[str, float] = {
    "max_malformed": 0,
    "max_exact_train_duplicates": 0,
    "max_near_train_duplicate_rate": 0.01,
    "max_semi_transparent_ratio": 0.01,
    "max_palette_size_mean": 32,
    "max_exact_duplicate_rate": 0.02,
    "max_repeated_template_rate": 0.25,
    "max_conditional_regression": 0.03,
}

_MAX_COMPARISON_REPORT_BYTES = 4 * 1024 * 1024
_MAX_COMPARISON_JSONL_BYTES = 64 * 1024 * 1024
_MAX_COMPARISON_ROWS = 100_000


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        int(getattr(metadata, "st_nlink", 1)),
        metadata.st_size,
        getattr(metadata, "st_mtime_ns", None),
    )


def _load_comparison_report(directory: Path) -> dict[str, Any]:
    """Read one stable, bounded benchmark summary before authoring output."""

    path = directory / "summary.json"
    descriptor = -1
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or int(getattr(info, "st_nlink", 1)) != 1
            or info.st_size > _MAX_COMPARISON_REPORT_BYTES
        ):
            raise ValueError("Evaluation summary is not one bounded regular file.")
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_COMPARISON_REPORT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_COMPARISON_REPORT_BYTES:
                raise ValueError("Evaluation summary exceeds its byte limit.")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identities = {_stable_file_identity(entry) for entry in (info, before, after, current)}
        if len(identities) != 1:
            raise ValueError("Evaluation summary changed while it was read.")
        value = strict_json_loads(b"".join(chunks))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation summary could not be read safely.") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise ValueError("Evaluation summary root must be an object.")
    if value.get("schema_version") != "generation_benchmark_v1.0":
        raise ValueError("Evaluation summary schema is unsupported.")
    return value


def _comparison_key(item: Mapping[str, Any]) -> tuple[str, int]:
    identity: str | None = None
    for field in ("prompt_id", "prompt", "sample_id"):
        if field not in item:
            continue
        value = item[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"Per-image metric field {field} is malformed.")
        identity = value
        break
    if identity is None:
        raise ValueError("Per-image metric row has no comparison identity.")
    seed_field = "noise_seed" if "noise_seed" in item else "seed"
    seed = item.get(seed_field)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Per-image metric field {seed_field} is malformed.")
    return identity, seed


def _load_comparison_rows(directory: Path) -> list[dict[str, Any]]:
    """Load a complete stable JSONL comparison input before output exists."""

    path = directory / "per_image_metrics.jsonl"
    descriptor = -1
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or int(getattr(info, "st_nlink", 1)) != 1
            or info.st_size > _MAX_COMPARISON_JSONL_BYTES
        ):
            raise ValueError("Per-image metrics are not one bounded regular file.")
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_COMPARISON_JSONL_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_COMPARISON_JSONL_BYTES:
                raise ValueError("Per-image metrics exceed their byte limit.")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if len({_stable_file_identity(entry) for entry in (info, before, after, current)}) != 1:
            raise ValueError("Per-image metrics changed while they were read.")
        text = b"".join(chunks).decode("utf-8", errors="strict")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Per-image metrics could not be read safely.") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    lines = text.splitlines()
    if not lines or len(lines) > _MAX_COMPARISON_ROWS or any(not line.strip() for line in lines):
        raise ValueError("Per-image metrics must contain bounded nonblank JSONL rows.")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    for line in lines:
        value = strict_json_loads(line)
        if not isinstance(value, dict) or not value:
            raise ValueError("Every per-image metric row must be a nonempty object.")
        key = _comparison_key(value)
        metrics = value.get("metrics")
        pixel_art = metrics.get("pixel_art") if isinstance(metrics, Mapping) else None
        if not isinstance(pixel_art, Mapping):
            raise ValueError("Per-image metric row is missing its typed comparison metrics.")
        for field in (
            "unique_palette_size",
            "silhouette_occupancy",
            "foreground_fragmentation",
            "high_frequency_pixel_noise",
        ):
            number = pixel_art.get(field)
            if field not in pixel_art or (
                number is not None
                and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number))
            ):
                raise ValueError(f"Per-image metric field {field} is missing or malformed.")
        if key in keys:
            raise ValueError("Per-image metrics contain a duplicate comparison identity.")
        keys.add(key)
        rows.append(value)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def discover_runs(root: Path) -> list[Path]:
    if (root / "generated_manifest.jsonl").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("generated_manifest.jsonl")})


def resolve_image(run: Path, record: Mapping[str, Any]) -> Path:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    for key in ("indexed_png", "hard_rgba", "raw_rgba"):
        if paths.get(key):
            return run / str(paths[key])
    return run / str(record.get("image") or "")


def score_suite(
    generated: Path,
    out: Path,
    *,
    training_manifests: list[Path] | None = None,
    limit: int = 0,
    gates_path: Path | None = None,
    checkpoint: Path | None = None,
    benchmark_manifest: Path | None = None,
    training_dataset_identity: str | None = None,
    training_view_identity: str | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    comparison_input_reasons: list[str] = []
    try:
        training = load_training_images(training_manifests or []) if training_manifests else []
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        training = []
        comparison_input_reasons.append(f"training comparison input is malformed: {error}")
    items: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    for run in discover_runs(generated):
        for record in read_jsonl(run / "generated_manifest.jsonl"):
            if limit and len(items) >= limit:
                break
            image_path = resolve_image(run, record)
            metrics = score_image(image_path, record)
            arr = _load_rgba(image_path)
            neighbors = retrieve_neighbors(arr, training) if arr is not None and training else []
            conditional = (
                score_conditions(record, arr, metrics["pixel_art"].get("palette_adherence")) if arr is not None else {}
            )
            nearest = neighbors[0] if neighbors else None
            item = {
                "sample_id": str(record.get("sample_id") or image_path.stem),
                "prompt_id": str(record.get("prompt_id") or ""),
                "prompt": str(record.get("prompt") or record.get("caption") or ""),
                "category": str(record.get("category") or "unknown"),
                "seed": record.get("seed"),
                "noise_seed": record.get("noise_seed"),
                "checkpoint": str(record.get("checkpoint") or ""),
                "run": str(run),
                "image": str(image_path),
                "generated_manifest_path": str((run / "generated_manifest.jsonl").resolve()),
                "metrics": metrics,
                "conditional": conditional,
                "training_neighbors": neighbors,
                "detector_policy_version": DETECTOR_POLICY_VERSION,
                "comparison_method": COMPARISON_METHOD,
                "comparison_parameters": COMPARISON_PARAMETERS,
                "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
                "detector_policy_sha256": DETECTOR_POLICY_SHA256,
                "memorization_evidence_class": nearest.get("evidence_class") if nearest else "no_material_match",
                "evidence_strength": nearest.get("evidence_strength") if nearest else "none",
                "requires_human_review": bool(nearest and nearest.get("requires_human_review")),
                "machine_hard_block_candidate": bool(nearest and nearest.get("machine_hard_block_candidate")),
                "warning_only": bool(nearest and nearest.get("warning_only")),
                "low_evidence_reason": nearest.get("low_evidence_reason") if nearest else None,
                "suspicious_memorization": suspicious_kind(nearest),
                "label_quality": record.get("label_quality"),
                "split": str(record.get("split") or ""),
                "source_id": str(record.get("source_id") or ""),
                "source_pack": str(record.get("source_pack") or record.get("pack") or ""),
                "artist": str(record.get("artist") or record.get("author") or ""),
                "inference_path": str(record.get("inference_path") or ""),
                "propagation_relation": str(record.get("propagation_relation") or ""),
                "open_set": bool(record.get("open_set")),
                "unseen_pack": bool(record.get("unseen_pack")),
            }
            item["conditional_adherence"] = _conditional_adherence(conditional)
            # Legacy field: now means machine-hard evidence only.  Candidate,
            # review, and warning states remain separate in the fields above.
            item["memorization_indicator"] = bool(item["machine_hard_block_candidate"])
            item["generation_failed"] = not bool(metrics["hard_validity"].get("pass"))
            items.append(item)
            if arr is not None:
                arrays.append(arr)
        if limit and len(items) >= limit:
            break
    valid_items = [item for item in items if item["metrics"]["pixel_art"]]
    valid_arrays = [_load_rgba(Path(item["image"])) for item in valid_items]
    batch = batch_metrics(valid_items, [array for array in valid_arrays if array is not None])
    summary = _summarize(items, batch)
    comparison_reasons: list[str] = list(comparison_input_reasons)
    candidate_pairs: list[dict[str, Any]] = []
    candidate_context = _candidate_context(
        items,
        training,
        training_manifests or [],
        checkpoint=checkpoint,
        benchmark_manifest=benchmark_manifest,
        training_dataset_identity=training_dataset_identity,
        training_view_identity=training_view_identity,
        reasons=comparison_reasons,
    )
    if candidate_context is not None:
        candidate_pairs = _candidate_pairs(
            items,
            training,
            out / "memorization_training_images",
            training_manifest=candidate_context["training_manifest"],
            training_dataset_identity=candidate_context["training_dataset_identity"],
            training_view_identity=candidate_context["training_view_identity"],
        )
        if len(candidate_pairs) != len(items):
            comparison_reasons.append("not every generated sample has one complete training comparison")
            candidate_pairs = []
            candidate_context = None
    memo = summary["memorization"]
    if candidate_context is not None:
        memo["candidate_pair_ids"] = [pair["pair_id"] for pair in candidate_pairs]
        memo["candidate_count"] = len(candidate_pairs)
        memo["evidence_contract_state"] = "complete"
        memo["evidence_contract_reasons"] = []
    else:
        memo["evidence_contract_state"] = "incomplete"
        memo["evidence_contract_reasons"] = sorted(set(comparison_reasons))
    gates = dict(DEFAULT_GATES)
    if gates_path:
        gates.update(json.loads(gates_path.read_text(encoding="utf-8")))
    promotion = evaluate_gates(summary, gates)
    _apply_production_evidence_contract(promotion, memo)
    training_label_rows = [row for path in training_manifests or [] for row in read_jsonl(path)]
    label_quality = {
        "generated_conditions": evaluation_uncertainty_report(items),
        "training_labels": evaluation_uncertainty_report(training_label_rows),
        "metrics_by_label_quality_stratum": _label_quality_metric_strata(items),
        "correlation_analysis": uncertainty_correlation_report(items),
    }
    report = {
        "schema_version": "generation_benchmark_v1.0",
        "detector_policy_version": DETECTOR_POLICY_VERSION,
        "comparison_method": COMPARISON_METHOD,
        "comparison_parameters": COMPARISON_PARAMETERS,
        "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
        "detector_policy_sha256": DETECTOR_POLICY_SHA256,
        "detector_policy": detector_policy_record(),
        "generated": str(generated),
        "training_manifests": [str(path) for path in training_manifests or []],
        "thresholds": {
            "perceptual_near_duplicate": 0.035,
            "geometry_duplicate_iou": 0.96,
            "near_train_pixel_distance": 0.025,
            "suspicious_geometry_iou": 0.98,
            "palette_adherence_rgb_tolerance": "12/255",
            "memorization_detector": COMPARISON_PARAMETERS["thresholds"],
        },
        "summary": summary,
        "label_quality": label_quality,
        "promotion": promotion,
        "artifacts": {
            "candidate_evidence": str((out / "candidate_evidence.json").resolve())
            if candidate_context is not None
            else None,
            "detector_policy": str((out / "detector_policy.json").resolve()),
        },
    }
    write_jsonl(out / "per_image_metrics.jsonl", items)
    (out / "detector_policy.json").write_text(
        json.dumps(detector_policy_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_markdown(report), encoding="utf-8")
    (out / "promotion_gates.json").write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if candidate_context is not None:
        try:
            write_candidate_bundle(
                out / "candidate_evidence.json",
                pairs=candidate_pairs,
                checkpoint=candidate_context["checkpoint"],
                benchmark_manifest=candidate_context["benchmark_manifest"],
                machine_report=out / "summary.json",
                generated_report=out / "per_image_metrics.jsonl",
                generated_manifest=candidate_context["generated_manifest"],
                training_manifest=candidate_context["training_manifest"],
                detector_policy_artifact=out / "detector_policy.json",
                training_dataset_identity=candidate_context["training_dataset_identity"],
                training_view_identity=candidate_context["training_view_identity"],
                generated_images_root=candidate_context["generated_manifest"].parent,
                training_images_root=out / "memorization_training_images",
            )
        except (OSError, ValueError) as error:
            memo.pop("candidate_pair_ids", None)
            memo.pop("candidate_count", None)
            memo["evidence_contract_state"] = "incomplete"
            memo["evidence_contract_reasons"] = [f"candidate bundle emission failed: {error}"]
            report["artifacts"]["candidate_evidence"] = None
            promotion = evaluate_gates(summary, gates)
            _apply_production_evidence_contract(promotion, memo)
            report["promotion"] = promotion
            (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (out / "promotion_gates.json").write_text(
                json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    _write_suspicious_sheet(items, training, out / "suspicious_training_pairs.html", out / "contact_assets")
    _write_overview_sheet(items, out / "contact_sheet.html")
    return report


def _label_quality_metric_strata(items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    def belongs(name: str, item: Mapping[str, Any]) -> bool:
        quality = extract_training_quality(item)
        band = str(quality.get("record_uncertainty_band")) if quality else "not_scorable"
        split = str(item.get("split") or "").lower()
        relation = str(item.get("propagation_relation") or "")
        checks = {
            "strong_labels": band == "strong",
            "usable_weak_labels": band == "usable_weak",
            "all_labels": True,
            "source_ood": split in {"source_ood", "source_ood_test"},
            "open_set": bool(item.get("open_set")) or split == "open_set_test",
            "unseen_pack": bool(item.get("unseen_pack")),
            "propagated_labels": bool(relation),
            "non_propagated_labels": not relation,
        }
        return checks[name]

    result: dict[str, dict[str, Any]] = {}
    for name in (
        "strong_labels",
        "usable_weak_labels",
        "all_labels",
        "source_ood",
        "open_set",
        "unseen_pack",
        "propagated_labels",
        "non_propagated_labels",
    ):
        rows = [item for item in items if belongs(name, item)]
        result[name] = {
            "sample_count": len(rows),
            "hard_validity_pass_rate": _fraction(rows, lambda row: not bool(row.get("generation_failed"))),
            "conditional_adherence": _mean_scalar(rows, "conditional_adherence"),
            "memorization_indicator_rate": _fraction(rows, lambda row: bool(row.get("memorization_indicator"))),
            "generation_failure_rate": _fraction(rows, lambda row: bool(row.get("generation_failed"))),
        }
    return result


def _conditional_adherence(conditional: Mapping[str, Any]) -> float | None:
    values = [str(value) for value in conditional.values()]
    scorable = [value for value in values if value in {"represented", "omitted", "contradicted"}]
    return scorable.count("represented") / len(scorable) if scorable else None


def _fraction(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float | None:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def _mean_scalar(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _load_rgba(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image.load()
            return np.asarray(image.convert("RGBA"))
    except Exception:
        return None


def _candidate_context(
    items: Sequence[Mapping[str, Any]],
    training: Sequence[TrainingImage],
    training_manifests: Sequence[Path],
    *,
    checkpoint: Path | None,
    benchmark_manifest: Path | None,
    training_dataset_identity: str | None,
    training_view_identity: str | None,
    reasons: list[str],
) -> dict[str, Any] | None:
    """Resolve every immutable production input before candidate authoring."""
    if len(training_manifests) != 1:
        reasons.append("exactly one training manifest is required by the bound-v2 production contract")
    if not training:
        reasons.append("training comparison evidence is missing")
    if not items:
        reasons.append("generated comparison evidence is empty")
    if any(not item.get("training_neighbors") for item in items):
        reasons.append("one or more generated samples have no training comparison")
    generated_manifests = {
        Path(str(item.get("generated_manifest_path"))).resolve()
        for item in items
        if item.get("generated_manifest_path")
    }
    if len(generated_manifests) != 1:
        reasons.append("exactly one generated manifest is required by the bound-v2 production contract")
    resolved_checkpoint = checkpoint.resolve() if checkpoint is not None else None
    if resolved_checkpoint is None:
        recorded = {str(item.get("checkpoint") or "") for item in items}
        recorded.discard("")
        if len(recorded) == 1:
            candidate = Path(next(iter(recorded))).expanduser()
            if candidate.is_file():
                resolved_checkpoint = candidate.resolve()
    if resolved_checkpoint is None or not resolved_checkpoint.is_file():
        reasons.append("checkpoint identity is missing or unreadable")
    resolved_benchmark = benchmark_manifest.resolve() if benchmark_manifest is not None else None
    if resolved_benchmark is None or not resolved_benchmark.is_file():
        reasons.append("benchmark manifest identity is missing or unreadable")
    if reasons:
        return None
    training_manifest = training_manifests[0].resolve()
    manifest_hash = file_sha256(training_manifest)
    dataset_identity, view_identity = resolve_training_context_identities(
        dataset_identities=(item.dataset_identity for item in training),
        view_identities=(item.view_identity for item in training),
        manifest_sha256=manifest_hash,
        explicit_dataset_identity=training_dataset_identity,
        explicit_view_identity=training_view_identity,
    )
    return {
        "checkpoint": resolved_checkpoint,
        "benchmark_manifest": resolved_benchmark,
        "generated_manifest": next(iter(generated_manifests)),
        "training_manifest": training_manifest,
        "training_dataset_identity": dataset_identity,
        "training_view_identity": view_identity,
    }


def _candidate_pairs(
    items: Sequence[Mapping[str, Any]],
    training: Sequence[TrainingImage],
    training_image_root: Path,
    *,
    training_manifest: Path,
    training_dataset_identity: str,
    training_view_identity: str,
) -> list[dict[str, Any]]:
    """Project the exact ordered comparison rows used by the machine report."""
    target_by_identity = {
        (str(Path(item.dataset).resolve()), item.npz_file, item.npz_row, item.sprite_id): item for item in training
    }
    training_image_root.mkdir(parents=True, exist_ok=True)
    manifest_hash = file_sha256(training_manifest)
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        neighbors = item.get("training_neighbors")
        if not isinstance(neighbors, list) or not neighbors or not isinstance(neighbors[0], Mapping):
            continue
        nearest = dict(neighbors[0])
        key = (
            str(Path(str(nearest.get("dataset"))).resolve()),
            str(nearest.get("npz_file")),
            int(nearest.get("npz_row", -1)),
            str(nearest.get("sprite_id")),
        )
        target = target_by_identity.get(key)
        if target is None:
            continue
        identity_seed = {
            "generated_sample_id": item.get("sample_id"),
            "generated_png_path": str(Path(str(item.get("image"))).resolve()),
            "training_source_sprite_id": target.sprite_id,
            "training_source_blob_path": str((Path(target.dataset) / target.npz_file).resolve()),
            "training_row_or_index": target.npz_row,
        }
        pair_id = f"{item.get('sample_id')}__{target.sprite_id}"
        if pair_id in seen:
            pair_id = f"{pair_id}__{canonical_sha256(identity_seed)[:12]}"
        seen.add(pair_id)
        training_png = training_image_root / f"{canonical_sha256(identity_seed)[:24]}.png"
        Image.fromarray(np.clip(target.rgba, 0, 255).astype(np.uint8), "RGBA").save(training_png)
        generated_png = Path(str(item.get("image"))).resolve()
        source_blob = (Path(target.dataset) / target.npz_file).resolve()
        evidence_metrics = {
            field: nearest.get(field)
            for field in (
                "exact_rgba",
                "exact_alpha",
                "rgb_values_differ",
                "translated_duplicate",
                "pixel_distance",
                "union_rgba_distance",
                "compared_foreground_pixel_count",
                "alpha_iou",
                "perceptual_distance",
                "geometry_iou",
                "optional_embedding_similarity",
                "low_evidence_reason",
            )
        }
        pair: dict[str, Any] = {
            "pair_id": pair_id,
            "generated_sample_id": str(item.get("sample_id")),
            "prompt_id": str(item.get("prompt_id")),
            "seed": item.get("seed"),
            "noise_seed": item.get("noise_seed"),
            "generated_png_path": str(generated_png),
            "generated_png_sha256": file_sha256(generated_png),
            "generated_decoded_rgba_sha256": decoded_rgba_sha256(generated_png),
            "training_dataset_identity": training_dataset_identity,
            "training_view_identity": training_view_identity,
            "training_source_sprite_id": target.sprite_id,
            "training_row_or_index": target.npz_row,
            "training_image_path": str(training_png.resolve()),
            "training_source_blob_path": str(source_blob),
            "training_source_blob_sha256": file_sha256(source_blob),
            "training_decoded_rgba_sha256": decoded_rgba_sha256(training_png),
            "training_manifest_sha256": manifest_hash,
            "evidence_class": nearest.get("evidence_class"),
            "exact_rgba": nearest.get("exact_rgba") is True,
            "evidence_metrics": evidence_metrics,
            "evidence_diagnostics": {
                "generated": nearest.get("generated_diagnostics"),
                "training": nearest.get("training_diagnostics"),
            },
        }
        if isinstance(nearest.get("evidence_reasons"), list):
            pair["evidence_reasons"] = list(nearest["evidence_reasons"])
        pairs.append(pair)
    return sorted(pairs, key=lambda pair: str(pair["pair_id"]))


def _summarize(items: list[dict[str, Any]], batch: dict[str, Any]) -> dict[str, Any]:
    malformed = sum(not bool(item["metrics"]["hard_validity"].get("pass")) for item in items)
    pixels = [item["metrics"]["pixel_art"] for item in items if item["metrics"]["pixel_art"]]
    suspicious = [item for item in items if item.get("suspicious_memorization")]
    hard_evidence = [item for item in items if item.get("machine_hard_block_candidate")]
    review_required = [item for item in items if item.get("requires_human_review")]
    low_evidence = [item for item in items if item.get("warning_only")]
    conditional_counts = {
        field: Counter(item.get("conditional", {}).get(field, "unscorable") for item in items) for field in FIELDS
    }
    represented = sum(counts["represented"] for counts in conditional_counts.values())
    scorable = sum(
        sum(counts[key] for key in ("represented", "omitted", "contradicted")) for counts in conditional_counts.values()
    )
    return {
        "sample_count": len(items),
        "hard_validity": {"malformed_count": malformed, "pass_rate": 1.0 - malformed / max(1, len(items))},
        "pixel_art": {
            "palette_size_mean": _mean(pixels, "unique_palette_size"),
            "palette_concentration_mean": _mean(pixels, "palette_concentration"),
            "semi_transparent_ratio_mean": _mean(pixels, "semi_transparent_pixel_ratio"),
            "antialiased_edge_ratio_mean": _mean(pixels, "antialiased_edge_ratio"),
            "silhouette_occupancy_mean": _mean(pixels, "silhouette_occupancy"),
            "border_clipping_rate": _bool_mean(pixels, "border_clipping"),
            "fragmentation_mean": _mean(pixels, "foreground_fragmentation"),
            "high_frequency_noise_mean": _mean(pixels, "high_frequency_pixel_noise"),
            "palette_adherence_mean": _mean(pixels, "palette_adherence"),
        },
        "diversity": batch,
        "duplicate_groups": {
            "exact": duplicate_groups(items, "rgba_sha256"),
            "alpha": duplicate_groups(items, "alpha_sha256"),
            "translated_alpha": duplicate_groups(items, "normalized_alpha_sha256"),
        },
        "memorization": {
            "detector_policy_version": DETECTOR_POLICY_VERSION,
            "comparison_method": COMPARISON_METHOD,
            "comparison_parameters": COMPARISON_PARAMETERS,
            "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
            "detector_policy_sha256": DETECTOR_POLICY_SHA256,
            "hard_evidence_count": len(hard_evidence),
            "review_required_count": len(review_required),
            "low_evidence_collision_count": len(low_evidence),
            "warning_count": len(low_evidence),
            "unresolved_candidate_count": len(hard_evidence) + len(review_required),
            "evidence_class_counts": dict(Counter(item.get("memorization_evidence_class") for item in items)),
            # Legacy compatibility: suspicious includes every retained hard,
            # review, or warning candidate.  Gates do not use it as proof.
            "suspicious_count": len(suspicious),
            "suspicious_rate": len(suspicious) / max(1, len(items)),
            # Legacy compatibility: exact_rgba_count is restricted to
            # nontrivial exact RGBA machine-hard evidence.
            "exact_rgba_count": len(hard_evidence),
            "examples": [
                {
                    "sample_id": item["sample_id"],
                    "kind": item["suspicious_memorization"],
                    "nearest": item["training_neighbors"][0],
                }
                for item in suspicious[:20]
            ],
        },
        "conditional": {
            "counts": {field: dict(counts) for field, counts in conditional_counts.items()},
            "represented_rate": represented / max(1, scorable),
            "scorable_decisions": scorable,
        },
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _bool_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def evaluate_gates(
    summary: Mapping[str, Any], gates: Mapping[str, float], baseline: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    hard = summary["hard_validity"]
    pixel = summary["pixel_art"]
    diversity = summary["diversity"]
    memo = summary.get("memorization")
    memo_mapping = memo if isinstance(memo, Mapping) else {}
    detector_policy_supported = memo_mapping.get("detector_policy_version") == DETECTOR_POLICY_VERSION
    outcome = evaluate_memorization_outcome(memo, expected_total=summary.get("sample_count"))
    outcome_reasons = list(outcome.reasons)
    machine_status = outcome.status
    if not detector_policy_supported:
        machine_status = MemorizationMachineStatus.NOT_COMPARABLE
        outcome_reasons.append("detector policy version is missing or unsupported")
    hard_evidence_count = outcome.counts.get("hard_evidence_count", 0)
    review_required_count = outcome.counts.get("review_required_count", 0)
    unresolved_count = outcome.counts.get("unresolved_candidate_count", 0)
    sample_count = summary.get("sample_count")
    valid_sample_count = sample_count if isinstance(sample_count, int) and not isinstance(sample_count, bool) else 0
    unresolved_rate = unresolved_count / max(1, valid_sample_count)
    evidence_valid = machine_status not in {
        MemorizationMachineStatus.INCOMPLETE,
        MemorizationMachineStatus.NOT_COMPARABLE,
    }
    checks = {
        "detector_policy_supported": detector_policy_supported,
        "memorization_evidence_complete": evidence_valid,
        "malformed": int(hard["malformed_count"]) <= gates["max_malformed"],
        "memorization_hard_evidence": evidence_valid and hard_evidence_count == 0,
        "memorization_reviews_resolved": evidence_valid and review_required_count == 0,
        "exact_train_duplicates": evidence_valid and hard_evidence_count <= gates["max_exact_train_duplicates"],
        # Legacy gate name, now derived only from unresolved hard/review
        # candidates.  Warning-only collisions cannot fail this check.
        "near_train_duplicates": evidence_valid and unresolved_rate <= gates["max_near_train_duplicate_rate"],
        "alpha_quality": float(pixel["semi_transparent_ratio_mean"] or 0.0) <= gates["max_semi_transparent_ratio"],
        "palette": float(pixel["palette_size_mean"] or 0.0) <= gates["max_palette_size_mean"],
        "exact_duplicates": float(diversity.get("exact_duplicate_rate") or 0.0) <= gates["max_exact_duplicate_rate"],
        "template_collapse": float(diversity.get("repeated_template_rate") or 0.0)
        <= gates["max_repeated_template_rate"],
    }
    if baseline:
        candidate_rate = float(summary["conditional"]["represented_rate"])
        baseline_rate = float(baseline["conditional"]["represented_rate"])
        checks["conditional_not_worse"] = candidate_rate + gates["max_conditional_regression"] >= baseline_rate
    return {
        "policy_version": "generation_benchmark_v1.0",
        "detector_policy_version": str(memo_mapping.get("detector_policy_version") or "unsupported_or_missing"),
        "comparison_method": memo_mapping.get("comparison_method"),
        "comparison_parameters": memo_mapping.get("comparison_parameters"),
        "comparison_parameters_sha256": memo_mapping.get("comparison_parameters_sha256"),
        "detector_policy_sha256": memo_mapping.get("detector_policy_sha256"),
        "thresholds": dict(gates),
        "checks": checks,
        "pass": all(checks.values()),
        "memorization_machine_status": machine_status.value,
        "memorization_outcome_reasons": sorted(set(outcome_reasons)),
        "memorization_warnings": list(outcome.warnings),
        "manual_review_required": machine_status == MemorizationMachineStatus.MANUAL_REVIEW_REQUIRED,
        "hard_evidence_count": hard_evidence_count,
        "review_required_count": review_required_count,
        "low_evidence_collision_count": outcome.counts.get("warning_count", 0),
        "unresolved_candidate_count": unresolved_count,
    }


def _apply_production_evidence_contract(promotion: dict[str, Any], memorization: Mapping[str, Any]) -> None:
    """Fail closed when normal scoring could not author its bound-v2 artifacts."""
    if memorization.get("evidence_contract_state") == "complete":
        return
    raw_reasons = memorization.get("evidence_contract_reasons")
    reasons = (
        [str(reason) for reason in raw_reasons]
        if isinstance(raw_reasons, list)
        else ["bound-v2 memorization evidence is incomplete"]
    )
    promotion["memorization_machine_status"] = MemorizationMachineStatus.INCOMPLETE.value
    promotion["memorization_outcome_reasons"] = sorted({*promotion.get("memorization_outcome_reasons", []), *reasons})
    promotion["manual_review_required"] = False
    checks = promotion.get("checks")
    if isinstance(checks, dict):
        for name in (
            "memorization_evidence_complete",
            "memorization_hard_evidence",
            "memorization_reviews_resolved",
            "exact_train_duplicates",
            "near_train_duplicates",
        ):
            if name in checks:
                checks[name] = False
    promotion["pass"] = False


def compare_reports(baseline: Path, candidate: Path, out: Path, *, architecture_change: bool = False) -> dict[str, Any]:
    base_report = _load_comparison_report(baseline)
    cand_report = _load_comparison_report(candidate)
    definition_identity = metric_definition_identity(base_report)
    if definition_identity != metric_definition_identity(cand_report):
        raise IncompatibleMetricDefinitions(
            "Evaluation reports use incompatible metric definitions; no averages or deltas were computed."
        )
    base_items = _load_comparison_rows(baseline)
    cand_items = _load_comparison_rows(candidate)
    base_index = {_pair_key(item): item for item in base_items}
    cand_index = {_pair_key(item): item for item in cand_items}
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(base_index.keys() & cand_index.keys())
    deltas: list[dict[str, Any]] = []
    for key in keys:
        a = base_index[key]["metrics"]["pixel_art"]
        b = cand_index[key]["metrics"]["pixel_art"]
        deltas.append(
            {
                "key": list(key),
                "palette_size_delta": _delta(a, b, "unique_palette_size"),
                "occupancy_delta": _delta(a, b, "silhouette_occupancy"),
                "fragmentation_delta": _delta(a, b, "foreground_fragmentation"),
                "noise_delta": _delta(a, b, "high_frequency_pixel_noise"),
            }
        )
    gates = evaluate_gates(cand_report["summary"], DEFAULT_GATES, base_report["summary"])
    gates["manual_review_required"] = bool(gates["manual_review_required"] or architecture_change)
    report = {
        "schema_version": "generation_benchmark_compare_v1.0",
        "paired_count": len(keys),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "metric_definition_identity": definition_identity,
        "paired_deltas": deltas,
        "promotion": gates,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Paired checkpoint comparison",
        "",
        f"Paired outputs: {len(keys)}",
        f"Promotion pass: {gates['pass']}",
        "",
    ]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    _write_paired_sheet(base_index, cand_index, keys, out / "paired_contact_sheet.html")
    return report


def _pair_key(item: Mapping[str, Any]) -> tuple[str, int]:
    return _comparison_key(item)


def _delta(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> float | None:
    return None if a.get(key) is None or b.get(key) is None else float(b[key]) - float(a[key])


def human_package(
    a_root: Path, b_root: Path, out: Path, *, seed: int = 731001, mode: str = "side-by-side"
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "images"
    assets.mkdir(exist_ok=True)
    a_items = _raw_index(a_root)
    b_items = _raw_index(b_root)
    keys = sorted(a_items.keys() & b_items.keys())
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        blind_id = f"eval_{index:05d}"
        swap = bool(rng.getrandbits(1))
        first, second = (b_items[key], a_items[key]) if swap else (a_items[key], b_items[key])
        left = assets / f"{blind_id}_left.png"
        right = assets / f"{blind_id}_right.png"
        shutil.copyfile(first["image"], left)
        shutil.copyfile(second["image"], right)
        rows.append(
            {
                "image_id": blind_id,
                "prompt": first["prompt"],
                "condition": json.dumps(first.get("conditions", {}), sort_keys=True),
                "left": str(left.relative_to(out)).replace("\\", "/"),
                "right": str(right.relative_to(out)).replace("\\", "/"),
                "left_rating": "",
                "right_rating": "",
                "preference": "",
                "notes": "",
                "hidden_left_source": "B" if swap else "A",
                "hidden_right_source": "A" if swap else "B",
            }
        )
    visible_fields = [
        "image_id",
        "prompt",
        "condition",
        "left",
        "right",
        "left_rating",
        "right_rating",
        "preference",
        "notes",
    ]
    with (out / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=visible_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in visible_fields} for row in rows)
    write_jsonl(out / "blind_key.jsonl", rows)
    schema = {"ratings": "integer 1..5", "preference": "left|right|tie|unscorable", "notes": "optional string"}
    (out / "result_import_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    _write_human_html(rows, out / "index.html", mode)
    return {"pair_count": len(rows), "seed": seed, "mode": mode}


def _raw_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in discover_runs(root):
        for row in read_jsonl(run / "generated_manifest.jsonl"):
            result[_pair_key(row)] = {
                **row,
                "image": resolve_image(run, row),
                "prompt": str(row.get("prompt") or row.get("caption") or ""),
            }
    return result


def _write_overview_sheet(items: Sequence[Mapping[str, Any]], path: Path) -> None:
    cards = "".join(
        f'<figure><img src="{html.escape(Path(str(item["image"])).resolve().as_uri())}"><figcaption>{html.escape(str(item["sample_id"]))}</figcaption></figure>'
        for item in items
    )
    path.write_text(_html_page("Benchmark contact sheet", cards), encoding="utf-8")


def _write_suspicious_sheet(
    items: Sequence[Mapping[str, Any]], training: Sequence[TrainingImage], path: Path, assets: Path
) -> None:
    assets.mkdir(exist_ok=True)
    train_index = {(item.dataset, item.npz_file, item.npz_row): item for item in training}
    cards: list[str] = []
    for index, item in enumerate(item for item in items if item.get("suspicious_memorization")):
        neighbor = item["training_neighbors"][0]
        target = train_index.get((neighbor["dataset"], neighbor["npz_file"], neighbor["npz_row"]))
        if target is None:
            continue
        train_path = assets / f"train_{index:04d}.png"
        Image.fromarray(target.rgba, "RGBA").save(train_path)
        cards.append(
            f'<figure><img src="{html.escape(Path(str(item["image"])).resolve().as_uri())}"><img src="contact_assets/{train_path.name}">'
            f"<figcaption>{html.escape(str(item['sample_id']))} / {html.escape(target.sprite_id)} / {item['suspicious_memorization']}</figcaption></figure>"
        )
    path.write_text(
        _html_page("Suspicious generation/training pairs", "".join(cards) or "<p>No suspicious pairs.</p>"),
        encoding="utf-8",
    )


def _write_paired_sheet(
    a: Mapping[Any, Mapping[str, Any]], b: Mapping[Any, Mapping[str, Any]], keys: Sequence[Any], path: Path
) -> None:
    cards = "".join(
        f'<figure><img src="{Path(str(a[key]["image"])).resolve().as_uri()}"><img src="{Path(str(b[key]["image"])).resolve().as_uri()}"><figcaption>{html.escape(str(key))}</figcaption></figure>'
        for key in keys
    )
    path.write_text(_html_page("Paired checkpoint outputs", cards), encoding="utf-8")


def _write_human_html(rows: Sequence[Mapping[str, Any]], path: Path, mode: str) -> None:
    cards = "".join(
        f'<section><h3>{row["image_id"]}</h3><p>{html.escape(str(row["prompt"]))}</p><img src="{row["left"]}"><img src="{row["right"]}"></section>'
        for row in rows
    )
    path.write_text(_html_page(f"Blind A/B evaluation ({mode})", cards), encoding="utf-8")


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html><meta charset=\"utf-8\"><title>{html.escape(title)}</title>
<style>body{{font:14px sans-serif}}figure,section{{display:inline-block;vertical-align:top;margin:12px;padding:8px;border:1px solid #ccc}}img{{width:128px;height:128px;image-rendering:pixelated;background:#ddd;margin:4px}}figcaption{{max-width:280px}}</style><h1>{html.escape(title)}</h1>{body}"""


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Generation benchmark v1 summary",
            "",
            f"- Samples: {summary['sample_count']}",
            f"- Malformed: {summary['hard_validity']['malformed_count']}",
            f"- Exact duplicate rate: {summary['diversity'].get('exact_duplicate_rate')}",
            f"- Suspicious training neighbors: {summary['memorization']['suspicious_count']}",
            f"- Hard memorization evidence: {summary['memorization']['hard_evidence_count']}",
            f"- Review-required candidates: {summary['memorization']['review_required_count']}",
            f"- Low-evidence collision warnings: {summary['memorization']['low_evidence_collision_count']}",
            f"- Promotion pass: {report['promotion']['pass']}",
            "",
            "Raw per-image metrics remain in `per_image_metrics.jsonl`.",
            "",
        ]
    )

"""Benchmark orchestration, reports, promotion gates, and paired comparisons."""

from __future__ import annotations

import csv
import html
import json
import random
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.evaluation.conditional import FIELDS, score_conditions
from spritelab.evaluation.memorization import TrainingImage, load_training_images, retrieve_neighbors, suspicious_kind
from spritelab.evaluation.metrics import batch_metrics, duplicate_groups, score_image
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
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    training = load_training_images(training_manifests or []) if training_manifests else []
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
                "metrics": metrics,
                "conditional": conditional,
                "training_neighbors": neighbors,
                "suspicious_memorization": suspicious_kind(neighbors[0] if neighbors else None),
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
            item["memorization_indicator"] = bool(item["suspicious_memorization"])
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
    gates = dict(DEFAULT_GATES)
    if gates_path:
        gates.update(json.loads(gates_path.read_text(encoding="utf-8")))
    promotion = evaluate_gates(summary, gates)
    training_label_rows = [row for path in training_manifests or [] for row in read_jsonl(path)]
    label_quality = {
        "generated_conditions": evaluation_uncertainty_report(items),
        "training_labels": evaluation_uncertainty_report(training_label_rows),
        "metrics_by_label_quality_stratum": _label_quality_metric_strata(items),
        "correlation_analysis": uncertainty_correlation_report(items),
    }
    report = {
        "schema_version": "generation_benchmark_v1.0",
        "generated": str(generated),
        "training_manifests": [str(path) for path in training_manifests or []],
        "thresholds": {
            "perceptual_near_duplicate": 0.035,
            "geometry_duplicate_iou": 0.96,
            "near_train_pixel_distance": 0.025,
            "suspicious_geometry_iou": 0.98,
            "palette_adherence_rgb_tolerance": "12/255",
        },
        "summary": summary,
        "label_quality": label_quality,
        "promotion": promotion,
    }
    write_jsonl(out / "per_image_metrics.jsonl", items)
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_markdown(report), encoding="utf-8")
    (out / "promotion_gates.json").write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _summarize(items: list[dict[str, Any]], batch: dict[str, Any]) -> dict[str, Any]:
    malformed = sum(not bool(item["metrics"]["hard_validity"].get("pass")) for item in items)
    pixels = [item["metrics"]["pixel_art"] for item in items if item["metrics"]["pixel_art"]]
    suspicious = [item for item in items if item.get("suspicious_memorization")]
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
            "suspicious_count": len(suspicious),
            "suspicious_rate": len(suspicious) / max(1, len(items)),
            "exact_rgba_count": sum(item.get("suspicious_memorization") == "exact_rgba" for item in items),
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
    memo = summary["memorization"]
    checks = {
        "malformed": int(hard["malformed_count"]) <= gates["max_malformed"],
        "exact_train_duplicates": int(memo["exact_rgba_count"]) <= gates["max_exact_train_duplicates"],
        "near_train_duplicates": float(memo["suspicious_rate"]) <= gates["max_near_train_duplicate_rate"],
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
        "thresholds": dict(gates),
        "checks": checks,
        "pass": all(checks.values()),
        "manual_review_required": False,
    }


def compare_reports(baseline: Path, candidate: Path, out: Path, *, architecture_change: bool = False) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    base_report = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    cand_report = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
    base_items = read_jsonl(baseline / "per_image_metrics.jsonl")
    cand_items = read_jsonl(candidate / "per_image_metrics.jsonl")
    base_index = {_pair_key(item): item for item in base_items}
    cand_index = {_pair_key(item): item for item in cand_items}
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
    gates["manual_review_required"] = architecture_change
    report = {
        "schema_version": "generation_benchmark_compare_v1.0",
        "paired_count": len(keys),
        "baseline": str(baseline),
        "candidate": str(candidate),
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
    return str(item.get("prompt_id") or item.get("prompt") or item.get("sample_id")), int(
        item.get("noise_seed") or item.get("seed") or 0
    )


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
            f"- Promotion pass: {report['promotion']['pass']}",
            "",
            "Raw per-image metrics remain in `per_image_metrics.jsonl`.",
            "",
        ]
    )

"""Deterministic v1 release/demo gallery: prompts -> v1 preset sampling -> QA/review -> contact sheets -> report.

This module never trains. It only builds a representative prompt set (or
reads a user-supplied one), samples it with the official v1 export preset
(Phase 1 EMA checkpoint, CFG 3.0, 30 steps, k16 deterministic palette
projection), and packages the results for visual QA / release validation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.training.generated_qa import qa_generated_sprites
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.generated_canonicalizer import build_generation_contact_sheet
from spritelab.training.generator_challenger import ChallengerSampleConfig, run_sample_generator_challenger
from spritelab.training.ood_prompts import OodCompositionalPromptConfig, build_ood_compositional_prompts
from spritelab.training.sample_generator import read_prompt_records

SCHEMA_VERSION = "v1_gallery_v1.0"

DEFAULT_V1_CHECKPOINT = Path("experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt")
DEFAULT_OOD_PROMPT_LIMIT = 16

OFFICIAL_V1_STATEMENT = (
    "Official v1 default: Phase 1 EMA + CFG 3.0 + k16 deterministic palette projection.\n"
    "Palette-swap branches are experimental and not used for v1."
)

# Validated v1 OOD result (96 samples), recorded separately from any single gallery run.
# See docs/v1_default.md for provenance.
VALIDATED_V1_OOD_REFERENCE: dict[str, Any] = {
    "description": "Historical validated v1 OOD result (96 samples); not recomputed by this gallery run.",
    "qa_errors": 0,
    "median_visible_colors_before": 32,
    "median_visible_colors_after": 12,
    "rare_color_warning_rate_before": 0.3229,
    "rare_color_warning_rate_after": 0.0,
    "category_consistency": 0.8068,
    "color_consistency": 0.8438,
    "repeated_silhouette_rate": 0.0,
    "blob_collapse_rate": 0.3021,
    "potion_collapse_rate": 0.0521,
    "border_touch_rate": 0.5104,
    "mean_rgb_mae_visible": 0.0206,
    "destructive_rate": 0.0,
    "source_count_used": 928,
    "source_hash": "083d55be9803",
}

V1_GALLERY_CATEGORY_OBJECTS: dict[str, tuple[str, ...]] = {
    "weapon": ("sword", "axe", "bow", "dagger"),
    "armor": ("helm", "shield", "chestplate", "boots"),
    "item_icon": ("potion", "scroll", "ring", "lantern"),
    "tool": ("pickaxe", "wrench", "fishing_rod", "hoe"),
    "material": ("gem", "coin", "ingot", "crystal"),
    "effect_icon": ("flame", "spark", "star", "snowflake"),
    "plant": ("mushroom", "flower", "leaf", "cactus"),
}

V1_GALLERY_CATEGORY_COLORS: dict[str, tuple[str, ...]] = {
    "weapon": ("red", "gold"),
    "armor": ("blue", "metallic"),
    "item_icon": ("red", "purple"),
    "tool": ("brown", "gray"),
    "material": ("gray", "gold"),
    "effect_icon": ("red", "blue"),
    "plant": ("green", "yellow"),
}

V1_GALLERY_COMPOSITIONAL_PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "category": "weapon",
        "object_name": "dagger",
        "text": "rusty iron dagger",
        "colors": ["gray"],
        "materials": ["iron"],
        "style": ["rusty"],
    },
    {
        "category": "armor",
        "object_name": "shield",
        "text": "mossy stone shield",
        "colors": ["green"],
        "materials": ["stone"],
        "style": ["mossy"],
    },
    {
        "category": "item_icon",
        "object_name": "potion",
        "text": "glowing red potion",
        "colors": ["red"],
        "materials": [],
        "style": ["glowing"],
    },
    {
        "category": "tool",
        "object_name": "fishing_rod",
        "text": "golden fishing rod",
        "colors": ["gold"],
        "materials": ["gold"],
        "style": [],
    },
    {
        "category": "material",
        "object_name": "crystal",
        "text": "frosted blue crystal",
        "colors": ["blue"],
        "materials": ["ice"],
        "style": ["frosted"],
    },
    {
        "category": "effect_icon",
        "object_name": "star",
        "text": "sparkling purple star",
        "colors": ["purple"],
        "materials": [],
        "style": ["sparkling"],
    },
    {
        "category": "plant",
        "object_name": "cactus",
        "text": "spiked green cactus",
        "colors": ["green"],
        "materials": [],
        "style": ["spiked"],
    },
)

_STRESS_MODIFIERS: tuple[str, ...] = (
    "tiny minimalist",
    "oversized detailed",
    "flat silhouette",
    "high contrast",
)


@dataclass(frozen=True)
class BuildV1GalleryConfig:
    out_dir: Path
    checkpoint: Path = DEFAULT_V1_CHECKPOINT
    prompts: Path | None = None
    device: str = "cpu"
    seed: int = 20260723
    batch_size: int = 32
    num_samples: int | None = None
    categories: tuple[str, ...] | None = None
    contact_sheet_columns: int = 8
    include_ood: bool = True
    include_grounded: bool = True
    include_stress_prompts: bool = True


def build_v1_gallery_demo(config: BuildV1GalleryConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    contact_sheets_dir = out_dir / "contact_sheets"
    prompts_path = out_dir / "v1_gallery_prompts.jsonl"

    rows = _resolve_prompt_rows(config, out_dir)
    if not rows:
        raise ValueError("v1 gallery prompt set is empty")
    _write_prompt_rows(prompts_path, rows)

    sample_config = ChallengerSampleConfig(
        checkpoint=Path(config.checkpoint),
        prompts=prompts_path,
        out_dir=samples_dir,
        export_preset="v1",
        max_samples=len(rows),
        steps=30,
        cfg_scale=3.0,
        max_colors=32,
        alpha_threshold=0.5,
        device=config.device,
        seed=config.seed,
        batch_size=config.batch_size,
        dither=False,
        write_raw_rgba=True,
        write_hard_rgba=True,
        contact_sheet_labels="prompt",
        project_palette=True,
        project_palette_target_colors=16,
        project_palette_min_pixel_share=0.01,
        project_palette_method="deterministic_kmeans",
    )
    generation_report = run_sample_generator_challenger(sample_config)

    qa_result = qa_generated_sprites(samples_dir)
    review_result = review_generated_sprites(
        GeneratedReviewConfig(generated_dir=samples_dir, out_dir=samples_dir / "review", group_by="category")
    )

    manifest_records = _read_manifest(samples_dir / "generated_manifest.jsonl")
    contact_sheets = _write_contact_sheets(
        samples_dir, manifest_records, contact_sheets_dir, columns=config.contact_sheet_columns
    )
    projection_report = _read_json_if_exists(samples_dir / "palette_projection_report.json")

    category_counts = Counter(str(row.get("category") or "unknown") for row in rows)
    color_counts = Counter(
        str(row["colors"][0]) for row in rows if isinstance(row.get("colors"), list) and row["colors"]
    )

    report = _assemble_report(
        out_dir=out_dir,
        samples_dir=samples_dir,
        contact_sheets_dir=contact_sheets_dir,
        prompts_path=prompts_path,
        sample_config=sample_config,
        prompt_count=len(rows),
        category_counts=category_counts,
        color_counts=color_counts,
        generation_report=generation_report,
        projection_report=projection_report,
        qa_result=qa_result,
        review_result=review_result,
        contact_sheets=contact_sheets,
    )

    (out_dir / "v1_gallery_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "v1_gallery_report.md").write_text(_format_markdown(report), encoding="utf-8")
    return report


def build_default_v1_gallery_prompts(
    *,
    categories: Sequence[str] | None = None,
    include_grounded: bool = True,
    include_stress_prompts: bool = True,
) -> list[dict[str, Any]]:
    """Build the deterministic built-in v1 gallery prompt set (excluding OOD rows)."""

    selected = set(categories) if categories else set(V1_GALLERY_CATEGORY_OBJECTS)
    rows: list[dict[str, Any]] = []
    if include_grounded:
        rows.extend(_build_grounded_rows(selected))
        rows.extend(_build_compositional_rows(selected))
    if include_stress_prompts:
        rows.extend(_build_stress_rows(selected))
    return rows


def _resolve_prompt_rows(config: BuildV1GalleryConfig, out_dir: Path) -> list[dict[str, Any]]:
    if config.prompts is not None:
        rows = read_prompt_records(config.prompts)
    else:
        rows = build_default_v1_gallery_prompts(
            categories=config.categories,
            include_grounded=config.include_grounded,
            include_stress_prompts=config.include_stress_prompts,
        )
        if config.include_ood:
            rows = rows + _build_ood_rows(out_dir)
    if config.num_samples is not None:
        rows = rows[: max(0, int(config.num_samples))]
    return rows


def _build_ood_rows(out_dir: Path) -> list[dict[str, Any]]:
    ood_path = out_dir / "ood_compositional_prompts.jsonl"
    build_ood_compositional_prompts(OodCompositionalPromptConfig(out=ood_path, max_prompts=DEFAULT_OOD_PROMPT_LIMIT))
    rows = read_prompt_records(ood_path)
    for row in rows:
        row.setdefault("prompt_family", "ood_compositional")
    return rows


def _row(
    *,
    prompt_id: str,
    prompt: str,
    category: str,
    object_name: str,
    base_object: str,
    colors: list[str],
    materials: list[str] | None = None,
    style: list[str] | None = None,
    prompt_family: str,
) -> dict[str, Any]:
    materials = list(materials or [])
    style = list(style or ["pixel_art", "icon"])
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "target_sprite_id": prompt_id,
        "category": category,
        "object_name": object_name,
        "base_object": base_object,
        "colors": list(colors),
        "prompt_family": prompt_family,
        "conditioning": {
            "semantic_v3": {
                "category": category,
                "object_name": object_name,
                "open_name": object_name,
                "base_object": base_object,
                "attributes": {
                    "colors": list(colors),
                    "materials": materials,
                    "shapes": [],
                    "function": [],
                    "effects": [],
                    "state": [],
                    "style": style,
                },
            }
        },
    }


def _build_grounded_rows(selected: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, objects in V1_GALLERY_CATEGORY_OBJECTS.items():
        if category not in selected:
            continue
        for object_name in objects:
            for color in V1_GALLERY_CATEGORY_COLORS[category]:
                prompt_id = f"v1_gallery_{category}_{object_name}_{color}"
                prompt = f"{color} {object_name.replace('_', ' ')} 32x32 pixel art icon"
                rows.append(
                    _row(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        category=category,
                        object_name=object_name,
                        base_object=object_name,
                        colors=[color],
                        prompt_family="grounded",
                    )
                )
    return rows


def _build_compositional_rows(selected: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in V1_GALLERY_COMPOSITIONAL_PROMPTS:
        if spec["category"] not in selected:
            continue
        prompt_id = f"v1_gallery_comp_{spec['category']}_{spec['object_name']}"
        prompt = f"{spec['text']} 32x32 pixel art icon"
        rows.append(
            _row(
                prompt_id=prompt_id,
                prompt=prompt,
                category=str(spec["category"]),
                object_name=str(spec["object_name"]),
                base_object=str(spec["object_name"]),
                colors=list(spec["colors"]),
                materials=list(spec["materials"]),
                style=list(spec["style"]),
                prompt_family="unseen_composition",
            )
        )
    return rows


def _build_stress_rows(selected: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = [category for category in V1_GALLERY_CATEGORY_OBJECTS if category in selected]
    for index, category in enumerate(categories):
        object_name = V1_GALLERY_CATEGORY_OBJECTS[category][0]
        color = V1_GALLERY_CATEGORY_COLORS[category][0]
        modifier = _STRESS_MODIFIERS[index % len(_STRESS_MODIFIERS)]
        prompt_id = f"v1_gallery_stress_{category}_{object_name}"
        prompt = f"{modifier} {color} {object_name.replace('_', ' ')} 32x32 pixel art icon"
        rows.append(
            _row(
                prompt_id=prompt_id,
                prompt=prompt,
                category=category,
                object_name=object_name,
                base_object=object_name,
                colors=[color],
                style=[modifier.replace(" ", "_")],
                prompt_family="style_stress",
            )
        )
    return rows


def _write_prompt_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_contact_sheets(
    samples_dir: Path,
    manifest_records: list[dict[str, Any]],
    out_dir: Path,
    *,
    columns: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, str] = {}

    overall = build_generation_contact_sheet(
        samples_dir, manifest_records, out_dir / "overall.png", include_raw=False, columns=columns
    )
    if overall is not None:
        sheets["overall"] = overall.name

    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in manifest_records:
        by_category.setdefault(str(record.get("category") or "unknown"), []).append(record)
    for category in sorted(by_category):
        path = build_generation_contact_sheet(
            samples_dir,
            by_category[category],
            out_dir / f"category_{_safe_name(category)}.png",
            include_raw=False,
            columns=columns,
        )
        if path is not None:
            sheets[f"category_{category}"] = path.name

    by_color: dict[str, list[dict[str, Any]]] = {}
    for record in manifest_records:
        colors = record.get("colors")
        color = str(colors[0]) if isinstance(colors, list) and colors else None
        if not color:
            continue
        by_color.setdefault(color, []).append(record)
    for color in sorted(by_color):
        path = build_generation_contact_sheet(
            samples_dir,
            by_color[color],
            out_dir / f"color_{_safe_name(color)}.png",
            include_raw=False,
            columns=columns,
        )
        if path is not None:
            sheets[f"color_{color}"] = path.name

    return sheets


def _assemble_report(
    *,
    out_dir: Path,
    samples_dir: Path,
    contact_sheets_dir: Path,
    prompts_path: Path,
    sample_config: ChallengerSampleConfig,
    prompt_count: int,
    category_counts: Counter[str],
    color_counts: Counter[str],
    generation_report: Mapping[str, Any],
    projection_report: Mapping[str, Any] | None,
    qa_result: Any,
    review_result: Any,
    contact_sheets: Mapping[str, str],
) -> dict[str, Any]:
    generation_config = generation_report.get("config") if isinstance(generation_report.get("config"), Mapping) else {}
    checkpoint_resolved = generation_config.get("checkpoint_resolved")

    review_overall = review_result.report.get("overall") if isinstance(review_result.report, Mapping) else {}
    warning_counts = review_overall.get("warning_counts") if isinstance(review_overall.get("warning_counts"), Mapping) else {}
    review_sample_count = int(review_overall.get("count") or prompt_count)
    rare_color_warning_rate = (
        int(warning_counts.get("too_many_rare_colors", 0)) / review_sample_count if review_sample_count else None
    )
    border_touch_rate = (
        int(warning_counts.get("touches_border", 0)) / review_sample_count if review_sample_count else None
    )

    projection_summary = None
    if projection_report:
        projection_summary = {
            "median_visible_colors_before": projection_report.get("median_visible_color_count_before"),
            "median_visible_colors_after": projection_report.get("median_visible_color_count_after"),
            "mean_visible_colors_before": projection_report.get("mean_visible_color_count_before"),
            "mean_visible_colors_after": projection_report.get("mean_visible_color_count_after"),
            "mean_rgb_mae_visible": projection_report.get("mean_rgb_mae_visible"),
            "destructive_rate": projection_report.get("destructive_rate"),
            "safe_count": projection_report.get("safe_count"),
            "moderate_count": projection_report.get("moderate_count"),
            "destructive_count": projection_report.get("destructive_count"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "checkpoint_requested": str(sample_config.checkpoint),
            "checkpoint_resolved": checkpoint_resolved,
        },
        "preset": {
            "name": "v1",
            "cfg_scale": sample_config.cfg_scale,
            "steps": sample_config.steps,
            "projection_method": sample_config.project_palette_method,
            "projection_target_colors": sample_config.project_palette_target_colors,
            "projection_min_pixel_share": sample_config.project_palette_min_pixel_share,
        },
        "seed": sample_config.seed,
        "prompt_set": {
            "prompt_count": prompt_count,
            "category_counts": dict(sorted(category_counts.items())),
            "color_counts": dict(sorted(color_counts.items())),
        },
        "sample_count": int(generation_report.get("sample_count") or prompt_count),
        "output_paths": {
            "out_dir": str(out_dir),
            "samples_dir": str(samples_dir),
            "contact_sheets_dir": str(contact_sheets_dir),
            "prompts_file": str(prompts_path),
            "report_json": str(out_dir / "v1_gallery_report.json"),
            "report_markdown": str(out_dir / "v1_gallery_report.md"),
            "contact_sheets": dict(sorted(contact_sheets.items())),
        },
        "projection_summary": projection_summary,
        "generated_qa": {
            "sample_count": int(getattr(qa_result, "sample_count", 0)),
            "errors": len(getattr(qa_result, "errors", [])),
            "warnings": len(getattr(qa_result, "warnings", [])),
            "ok": bool(getattr(qa_result, "ok", True)),
        },
        "generated_review": {
            "sample_count": review_sample_count,
            "median_visible_colors": review_overall.get("median_visible_color_count"),
            "rare_color_warning_rate": rare_color_warning_rate,
            "border_touch_rate": border_touch_rate,
            "warning_counts": dict(warning_counts),
        },
        "validated_v1_ood_reference": dict(VALIDATED_V1_OOD_REFERENCE),
        "official_statement": OFFICIAL_V1_STATEMENT,
    }


def _format_markdown(report: Mapping[str, Any]) -> str:
    model = report.get("model", {})
    preset = report.get("preset", {})
    prompt_set = report.get("prompt_set", {})
    projection = report.get("projection_summary") or {}
    qa = report.get("generated_qa", {})
    review = report.get("generated_review", {})
    output_paths = report.get("output_paths", {})

    lines = [
        "# v1 Gallery Report",
        "",
        f"- Checkpoint requested: `{model.get('checkpoint_requested')}`",
        f"- Checkpoint resolved: `{model.get('checkpoint_resolved')}`",
        f"- Preset: `{preset.get('name')}` (CFG {preset.get('cfg_scale')}, {preset.get('steps')} steps)",
        f"- Projection: `{preset.get('projection_method')}`, "
        f"target colors {preset.get('projection_target_colors')}, "
        f"min pixel share {preset.get('projection_min_pixel_share')}",
        f"- Seed: {report.get('seed')}",
        f"- Prompt count: {prompt_set.get('prompt_count')}",
        f"- Sample count: {report.get('sample_count')}",
        "",
        "## Prompt Set",
        "",
        f"- Category counts: `{json.dumps(prompt_set.get('category_counts', {}), sort_keys=True)}`",
        f"- Color counts: `{json.dumps(prompt_set.get('color_counts', {}), sort_keys=True)}`",
        "",
        "## Palette Projection",
        "",
    ]
    if projection:
        lines.extend(
            [
                f"- Median visible colors: {projection.get('median_visible_colors_before')} -> "
                f"{projection.get('median_visible_colors_after')}",
                f"- Mean visible colors: {projection.get('mean_visible_colors_before')} -> "
                f"{projection.get('mean_visible_colors_after')}",
                f"- Mean RGB MAE visible: {projection.get('mean_rgb_mae_visible')}",
                f"- Destructive rate: {projection.get('destructive_rate')}",
                f"- Safe / moderate / destructive: {projection.get('safe_count')} / "
                f"{projection.get('moderate_count')} / {projection.get('destructive_count')}",
            ]
        )
    else:
        lines.append("- Palette projection report was not available.")

    lines.extend(
        [
            "",
            "## Generated QA",
            "",
            f"- Samples: {qa.get('sample_count')}",
            f"- Errors: {qa.get('errors')}",
            f"- Warnings: {qa.get('warnings')}",
            f"- OK: {qa.get('ok')}",
            "",
            "## Generated Review",
            "",
            f"- Median visible colors: {review.get('median_visible_colors')}",
            f"- Rare-color warning rate: {review.get('rare_color_warning_rate')}",
            f"- Border-touch rate: {review.get('border_touch_rate')}",
            "",
            "## Output Paths",
            "",
            f"- Samples: `{output_paths.get('samples_dir')}`",
            f"- Contact sheets: `{output_paths.get('contact_sheets_dir')}`",
            f"- Prompts file: `{output_paths.get('prompts_file')}`",
            "",
            "## Official v1 Default",
            "",
            str(report.get("official_statement", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value).strip())
    return cleaned or "unknown"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the deterministic v1 demo/release gallery.")
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--categories")
    parser.add_argument("--contact-sheet-columns", type=int, default=8)
    parser.add_argument("--include-ood", action="store_true", default=True)
    parser.add_argument("--no-include-ood", action="store_false", dest="include_ood")
    parser.add_argument("--include-grounded", action="store_true", default=True)
    parser.add_argument("--no-include-grounded", action="store_false", dest="include_grounded")
    parser.add_argument("--include-stress-prompts", action="store_true", default=True)
    parser.add_argument("--no-include-stress-prompts", action="store_false", dest="include_stress_prompts")
    parsed = parser.parse_args(argv)

    categories = None
    if parsed.categories:
        categories = tuple(token.strip() for token in str(parsed.categories).split(",") if token.strip())

    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=parsed.out_dir,
            checkpoint=parsed.checkpoint,
            prompts=parsed.prompts,
            device=parsed.device,
            seed=parsed.seed,
            batch_size=parsed.batch_size,
            num_samples=parsed.num_samples,
            categories=categories,
            contact_sheet_columns=parsed.contact_sheet_columns,
            include_ood=parsed.include_ood,
            include_grounded=parsed.include_grounded,
            include_stress_prompts=parsed.include_stress_prompts,
        )
    )
    print(f"Prompt count: {report['prompt_set']['prompt_count']}")
    print(f"Outputs written to {parsed.out_dir}")


if __name__ == "__main__":
    main()

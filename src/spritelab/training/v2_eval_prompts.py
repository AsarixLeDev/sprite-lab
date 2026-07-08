"""Deterministic larger OOD/compositional eval prompt builder for v2 Phase 0.

Invoke via:
    python -m spritelab train build-v2-eval-prompts [...args]

Generates a 256-384 row JSONL prompt file from the training manifest vocab,
covering category-color grids, object-color pairs, rare combos, style stress,
and in-distribution anchors. All prompts are deterministic given the same
dataset/manifest/seed/target-count.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.training.data import read_jsonl


# ── Constants ───────────────────────────────────────────────────────────────

SCHEMA_VERSION = "v2_eval_prompts_v1.0"

DEFAULT_COLORS: tuple[str, ...] = (
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "black", "white", "gray", "brown", "gold", "silver", "cyan",
)

MATERIAL_COLORS: set[str] = {"gold", "silver", "metallic", "wooden", "stone", "iron"}

# Simple style stress modifiers
STYLE_MODIFIERS: tuple[str, ...] = (
    "tiny minimalist",
    "oversized detailed",
    "flat silhouette",
    "high contrast",
    "cartoon chibi",
    "dark moody",
)

# Object names reserved for OOD-style combos (should be fairly universal)
UNIVERSAL_OBJECTS: tuple[str, ...] = (
    "sword", "axe", "bow", "dagger", "shield", "helm",
    "potion", "bottle", "scroll", "ring", "gem", "coin",
    "mushroom", "flower", "star", "flame", "crystal",
)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class V2EvalPromptsConfig:
    dataset: Path
    training_manifest: Path
    out: Path
    target_count: int = 384
    seed: int = 20260706
    include_grounded_grid: bool = True
    include_compositional: bool = True
    include_rare_combos: bool = True
    include_style_stress: bool = True
    out_report: bool = True
    category_balance: bool = True
    color_balance: bool = True


# ── Manifest vocab extraction ───────────────────────────────────────────────

def _extract_vocab(manifest: Path) -> dict[str, Any]:
    """Extract vocab (categories, objects, colors, etc.) from a training manifest."""
    rows = read_jsonl(manifest)

    categories: set[str] = set()
    objects: set[str] = set()
    base_objects: set[str] = set()
    colors: set[str] = set()
    materials: set[str] = set()
    shapes: set[str] = set()
    functions: set[str] = set()
    styles: set[str] = set()

    # track known (object, color) pairs for rare-combo detection
    seen_pairs: set[tuple[str, str]] = set()
    # track known (category, object) mappings
    category_objects: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        category = str(row.get("category") or "").strip().lower()
        obj = str(row.get("object_name") or row.get("base_object") or "").strip().lower()
        base = str(row.get("base_object") or "").strip().lower()

        if category:
            categories.add(category)
        if obj:
            objects.add(obj)
            if category:
                category_objects[category].add(obj)
        if base:
            base_objects.add(base)

        # Extract attributes from conditioning.semantic_v3.attributes
        attrs = _extract_attributes(row)
        for c in attrs.get("colors", []):
            c_norm = str(c).strip().lower()
            if c_norm:
                colors.add(c_norm)
            if obj and c_norm:
                seen_pairs.add((obj, c_norm))
        for m in attrs.get("materials", []):
            m_norm = str(m).strip().lower()
            if m_norm:
                materials.add(m_norm)
        for s in attrs.get("shapes", []):
            s_norm = str(s).strip().lower()
            if s_norm:
                shapes.add(s_norm)
        for f in attrs.get("function", []):
            f_norm = str(f).strip().lower()
            if f_norm:
                functions.add(f_norm)
        for s in attrs.get("style", []):
            s_norm = str(s).strip().lower()
            if s_norm:
                styles.add(s_norm)

    return {
        "categories": sorted(categories),
        "objects": sorted(objects),
        "base_objects": sorted(base_objects),
        "colors": sorted(colors),
        "materials": sorted(materials),
        "shapes": sorted(shapes),
        "functions": sorted(functions),
        "styles": sorted(styles),
        "seen_object_color_pairs": seen_pairs,
        "category_objects": {k: sorted(v) for k, v in category_objects.items()},
    }


def _extract_attributes(row: Mapping[str, Any]) -> dict[str, Any]:
    cond = row.get("conditioning")
    if isinstance(cond, Mapping):
        sem = cond.get("semantic_v3")
        if isinstance(sem, Mapping):
            attrs = sem.get("attributes")
            if isinstance(attrs, Mapping):
                return dict(attrs)
    return {}


# ── Prompt builders ─────────────────────────────────────────────────────────

def _color_for_category(category: str, colors: Sequence[str], seed: int, count: int = 6) -> list[str]:
    """Select `count` colors for a category, deterministic by seed + category."""
    rng = _category_rng(category, seed)
    available = [c for c in colors if c.lower() not in MATERIAL_COLORS]
    if len(available) < count:
        available = list(colors)
    indices = _sample_indices(len(available), count, rng)
    return [available[i] for i in indices]


def _sample_indices(population_size: int, count: int, rng: Any) -> list[int]:
    """Sample `count` indices without replacement, deterministically."""
    pool = list(range(population_size))
    rng.shuffle(pool)
    return pool[:min(count, population_size)]


def _category_rng(category: str, seed: int) -> Any:
    """Deterministic RNG keyed by seed + category."""
    import random
    h = hashlib.sha256(f"{seed}:{category}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _global_rng(seed: int) -> Any:
    import random
    return random.Random(seed)


def _safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("-", "_").replace(".", "").lower()


def _make_record(
    prompt_id: str,
    prompt: str,
    *,
    category: str,
    object_name: str,
    colors: list[str],
    prompt_family: str,
    materials: list[str] | None = None,
    style: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "target_sprite_id": prompt_id,
        "category": category,
        "object_name": object_name,
        "base_object": object_name,
        "colors": list(colors),
        "prompt_family": prompt_family,
        "conditioning": {
            "semantic_v3": {
                "category": category,
                "object_name": object_name,
                "open_name": object_name.replace("_", " "),
                "base_object": object_name,
                "attributes": {
                    "colors": list(colors),
                    "materials": list(materials or []),
                    "shapes": [],
                    "function": [],
                    "effects": [],
                    "state": [],
                    "style": list(style or []) or ["pixel_art", "icon"],
                },
            }
        },
    }


# ── Family builders ─────────────────────────────────────────────────────────

def _build_category_color_grid(
    vocab: dict[str, Any],
    rng: Any,
    target: int,
) -> list[dict[str, Any]]:
    """Build category x color grid prompts."""
    rows: list[dict[str, Any]] = []
    categories = vocab["categories"]
    colors = vocab["colors"] or list(DEFAULT_COLORS)

    if not categories:
        return rows

    rows_per_cat = max(1, target // max(1, len(categories)))
    for cat in categories:
        cat_colors = _color_for_category(cat, colors, rng.randint(0, 2**31 - 1), count=6)
        for color in cat_colors[:rows_per_cat]:
            obj = cat  # use category as object placeholder
            prompt_id = f"eval_catcolor_{_safe_name(cat)}_{_safe_name(color)}"
            prompt = f"{color} {cat.replace('_', ' ')} 32x32 pixel art icon"
            rows.append(
                _make_record(
                    prompt_id, prompt,
                    category=cat, object_name=cat, colors=[color],
                    prompt_family="category_color_grid",
                )
            )
    return rows


def _build_object_color_pairs(
    vocab: dict[str, Any],
    rng: Any,
    target: int,
) -> list[dict[str, Any]]:
    """Build object x color pair prompts that are plausible."""
    rows: list[dict[str, Any]] = []
    objects = vocab["objects"]
    colors = vocab["colors"] or list(DEFAULT_COLORS)
    seen_pairs = vocab.get("seen_object_color_pairs", set())
    category_objects = vocab.get("category_objects", {})

    # Prefer objects that come from diverse categories
    selectable = objects if objects else list(UNIVERSAL_OBJECTS)
    hue_colors = [c for c in colors if c.lower() not in MATERIAL_COLORS]

    # Select objects round-robin across categories to maintain balance
    ordered_objects: list[str] = []
    if category_objects:
        cat_list = sorted(category_objects)
        max_obj = max((len(category_objects[c]) for c in cat_list), default=0)
        for i in range(max_obj):
            for cat in cat_list:
                objs = category_objects[cat]
                if i < len(objs):
                    ordered_objects.append(objs[i])

    if not ordered_objects:
        ordered_objects = sorted(selectable)

    rng.shuffle(hue_colors)
    rng.shuffle(ordered_objects)

    for obj in ordered_objects[:target]:
        for color in hue_colors:
            if len(rows) >= target:
                break
            # Check if this combo exists in training (still include it for eval coverage)
            prompt_id = f"eval_objcolor_{_safe_name(obj)}_{_safe_name(color)}"
            prompt = f"{color} {obj.replace('_', ' ')} 32x32 pixel art icon"
            # Find the category for this object
            cat = "unknown"
            for c, objs in category_objects.items():
                if obj in objs:
                    cat = c
                    break
            rows.append(
                _make_record(
                    prompt_id, prompt,
                    category=cat, object_name=obj, colors=[color],
                    prompt_family="object_color_pairs",
                )
            )

    return rows[:target]


def _build_rare_combos(
    vocab: dict[str, Any],
    rng: Any,
    target: int,
) -> list[dict[str, Any]]:
    """Build prompts for object-color combos not seen in training."""
    rows: list[dict[str, Any]] = []
    objects = vocab["objects"]
    colors = vocab["colors"] or list(DEFAULT_COLORS)
    seen_pairs = vocab.get("seen_object_color_pairs", set())

    if not objects or not colors:
        return rows

    hue_colors = [c for c in colors if c.lower() not in MATERIAL_COLORS]
    if not hue_colors:
        hue_colors = list(colors)

    # Find novel (object, color) pairs
    novel: list[tuple[str, str]] = []
    for obj in sorted(objects):
        for color in hue_colors:
            if (obj, color) not in seen_pairs:
                novel.append((obj, color))

    rng.shuffle(novel)
    for obj, color in novel[:target]:
        prompt_id = f"eval_rare_{_safe_name(obj)}_{_safe_name(color)}"
        prompt = f"{color} {obj.replace('_', ' ')} 32x32 pixel art icon"
        cat = "unknown"
        for c, objs in (vocab.get("category_objects") or {}).items():
            if obj in objs:
                cat = c
                break
        rows.append(
            _make_record(
                prompt_id, prompt,
                category=cat, object_name=obj, colors=[color],
                prompt_family="rare_combos",
            )
        )

    return rows[:target]


def _build_style_stress(
    vocab: dict[str, Any],
    rng: Any,
    target: int,
) -> list[dict[str, Any]]:
    """Build style stress prompts with modifiers applied to various objects."""
    rows: list[dict[str, Any]] = []
    objects = vocab["objects"] or list(UNIVERSAL_OBJECTS)
    colors = vocab["colors"] or list(DEFAULT_COLORS)

    if not objects:
        return rows

    hue_colors = [c for c in colors if c.lower() not in MATERIAL_COLORS] or list(colors)
    modifiers = list(STYLE_MODIFIERS)

    pairs: list[tuple[str, str, str]] = []  # (modifier, object, color)
    for mod in modifiers:
        rng.shuffle(list(objects))  # re-shuffle for each modifier
        for obj in sorted(objects)[:4]:  # 4 objects per modifier
            color = rng.choice(hue_colors) if hue_colors else "red"
            pairs.append((mod, obj, color))

    rng.shuffle(pairs)
    for mod, obj, color in pairs[:target]:
        prompt_id = f"eval_style_{_safe_name(mod)}_{_safe_name(obj)}"
        prompt = f"{mod} {color} {obj.replace('_', ' ')} 32x32 pixel art icon"
        cat = "unknown"
        for c, objs in (vocab.get("category_objects") or {}).items():
            if obj in objs:
                cat = c
                break
        rows.append(
            _make_record(
                prompt_id, prompt,
                category=cat, object_name=obj, colors=[color],
                prompt_family="style_stress",
                style=[mod] if mod else [],
            )
        )

    return rows[:target]


def _build_anchors(
    vocab: dict[str, Any],
    rng: Any,
    target: int,
    manifest: Path,
) -> list[dict[str, Any]]:
    """Build in-distribution anchor prompts from the manifest."""
    rows_list = read_jsonl(manifest)
    # Filter to train split
    train_rows = [r for r in rows_list if r.get("split") == "train"]
    if not train_rows:
        return []

    # Deduplicate by (object_name, color)
    rng.shuffle(train_rows)
    seen: set[tuple[str, str]] = set()
    anchors: list[dict[str, Any]] = []

    for row in train_rows:
        if len(anchors) >= target:
            break
        obj = str(row.get("object_name") or "").strip().lower()
        attrs = _extract_attributes(row)
        color_list = [str(c).strip().lower() for c in attrs.get("colors", [])]
        color_key = color_list[0] if color_list else ""
        pair = (obj, color_key)
        if pair in seen:
            continue
        seen.add(pair)

        cat = str(row.get("category") or "unknown").strip().lower()
        prompt_text = str(row.get("caption") or f"{color_key} {obj}").strip()
        prompt_id = f"eval_anchor_{_safe_name(obj)}_{_safe_name(color_key)}_{len(anchors):04d}"
        anchors.append(
            _make_record(
                prompt_id, prompt_text,
                category=cat, object_name=obj,
                colors=color_list if color_list else ["unknown"],
                prompt_family="in_distribution_anchors",
                materials=[str(m).strip().lower() for m in attrs.get("materials", [])],
                style=[str(s).strip().lower() for s in attrs.get("style", [])],
            )
        )

    return anchors


# ── Main builder ────────────────────────────────────────────────────────────

def build_v2_eval_prompts(config: V2EvalPromptsConfig) -> dict[str, Any]:
    rng = _global_rng(config.seed)

    vocab = _extract_vocab(config.training_manifest)
    dataset_hash = _dataset_hash(config.dataset)

    # Allocate prompt budget across families
    families_enabled = []
    if config.include_grounded_grid:
        families_enabled.append("category_color_grid")
    if config.include_compositional:
        families_enabled.append("object_color_pairs")
    if config.include_rare_combos:
        families_enabled.append("rare_combos")
    if config.include_style_stress:
        families_enabled.append("style_stress")
    families_enabled.append("in_distribution_anchors")

    per_family = max(1, config.target_count // len(families_enabled))
    remainder = config.target_count - per_family * len(families_enabled)

    all_rows: list[dict[str, Any]] = []
    family_builders = {
        "category_color_grid": lambda t: _build_category_color_grid(vocab, rng, t),
        "object_color_pairs": lambda t: _build_object_color_pairs(vocab, rng, t),
        "rare_combos": lambda t: _build_rare_combos(vocab, rng, t),
        "style_stress": lambda t: _build_style_stress(vocab, rng, t),
        "in_distribution_anchors": lambda t: _build_anchors(vocab, rng, t, config.training_manifest),
    }

    for i, family in enumerate(families_enabled):
        budget = per_family + (1 if i < remainder else 0)
        builder = family_builders.get(family)
        if builder:
            rows = builder(budget)
            all_rows.extend(rows)

    # Deduplicate by prompt_id
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicates_removed = 0
    for row in all_rows:
        pid = str(row.get("prompt_id", ""))
        if pid not in seen_ids:
            seen_ids.add(pid)
            deduped.append(row)
        else:
            duplicates_removed += 1

    # Index sequentially
    for i, row in enumerate(deduped):
        row["eval_prompt_index"] = i

    # Write JSONL
    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Build coverage report
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    for row in deduped:
        family_counts[str(row.get("prompt_family", ""))] += 1
        category_counts[str(row.get("category", ""))] += 1
        for c in (row.get("colors") or []):
            color_counts[str(c)] += 1
        object_counts[str(row.get("object_name", ""))] += 1

    # Known vs novel composition estimate
    seen_pairs = vocab.get("seen_object_color_pairs", set())
    known_count = 0
    novel_count = 0
    for row in deduped:
        obj = str(row.get("object_name", "")).strip().lower()
        colors = row.get("colors") or []
        if colors and (obj, colors[0]) in seen_pairs:
            known_count += 1
        else:
            novel_count += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "prompt_file": str(out_path),
        "prompt_count": len(deduped),
        "target_count": config.target_count,
        "seed": config.seed,
        "dataset": str(config.dataset),
        "dataset_hash": dataset_hash,
        "duplicates_removed": duplicates_removed,
        "families": dict(sorted(family_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "color_counts": dict(sorted(color_counts.items())),
        "object_counts_top20": dict(sorted(object_counts.items(), key=lambda x: -x[1])[:20]),
        "known_train_composition_count": known_count,
        "novel_composition_count": novel_count,
        "vocab_summary": {
            "categories": len(vocab["categories"]),
            "objects": len(vocab["objects"]),
            "base_objects": len(vocab["base_objects"]),
            "colors": len(vocab["colors"]),
            "materials": len(vocab["materials"]),
            "shapes": len(vocab["shapes"]),
            "functions": len(vocab["functions"]),
            "styles": len(vocab["styles"]),
        },
        "config": {
            "include_grounded_grid": config.include_grounded_grid,
            "include_compositional": config.include_compositional,
            "include_rare_combos": config.include_rare_combos,
            "include_style_stress": config.include_style_stress,
            "category_balance": config.category_balance,
            "color_balance": config.color_balance,
        },
    }

    if config.out_report:
        json_path = out_path.with_suffix(".report.json")
        md_path = Path(str(out_path).replace(".jsonl", "_report.md"))
        if md_path == out_path:
            md_path = out_path.with_name(out_path.stem + "_report.md")
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_prompt_report_md(report, md_path)
        report["report_json"] = str(json_path)
        report["report_md"] = str(md_path)

    return report


def _dataset_hash(dataset: Path) -> str:
    manifest = dataset / "training_manifest.jsonl"
    if not manifest.is_file():
        return ""
    h = hashlib.sha256()
    with manifest.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_prompt_report_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v2 Eval Prompts Coverage Report",
        "",
        f"- **Prompt count**: {report['prompt_count']} (target: {report['target_count']})",
        f"- **Seed**: {report['seed']}",
        f"- **Dataset hash**: `{report['dataset_hash'][:16]}...`",
        f"- **Duplicates removed**: {report['duplicates_removed']}",
        f"- **Known train compositions**: {report['known_train_composition_count']}",
        f"- **Novel compositions**: {report['novel_composition_count']}",
        "",
        "## Prompt Families",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    for family, count in sorted((report.get("families") or {}).items()):
        lines.append(f"| {family} | {count} |")

    lines.extend([
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ])
    for cat, count in sorted((report.get("category_counts") or {}).items()):
        lines.append(f"| {cat} | {count} |")

    lines.extend([
        "",
        "## Color Counts",
        "",
        "| Color | Count |",
        "|---|---:|",
    ])
    for color, count in sorted((report.get("color_counts") or {}).items()):
        lines.append(f"| {color} | {count} |")

    lines.extend([
        "",
        "## Top 20 Objects",
        "",
        "| Object | Count |",
        "|---|---:|",
    ])
    for obj, count in sorted((report.get("object_counts_top20") or {}).items(), key=lambda x: -x[1])[:20]:
        lines.append(f"| {obj} | {count} |")

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a larger deterministic OOD/eval prompt suite for v2 Phase 0.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-count", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--include-grounded-grid", action="store_true", default=True)
    parser.add_argument("--no-include-grounded-grid", action="store_false", dest="include_grounded_grid")
    parser.add_argument("--include-compositional", action="store_true", default=True)
    parser.add_argument("--no-include-compositional", action="store_false", dest="include_compositional")
    parser.add_argument("--include-rare-combos", action="store_true", default=True)
    parser.add_argument("--no-include-rare-combos", action="store_false", dest="include_rare_combos")
    parser.add_argument("--include-style-stress", action="store_true", default=True)
    parser.add_argument("--no-include-style-stress", action="store_false", dest="include_style_stress")
    parser.add_argument("--out-report", action="store_true", default=True)
    parser.add_argument("--no-out-report", action="store_false", dest="out_report")
    parsed = parser.parse_args(argv)

    config = V2EvalPromptsConfig(
        dataset=parsed.dataset,
        training_manifest=parsed.training_manifest,
        out=parsed.out,
        target_count=parsed.target_count,
        seed=parsed.seed,
        include_grounded_grid=parsed.include_grounded_grid,
        include_compositional=parsed.include_compositional,
        include_rare_combos=parsed.include_rare_combos,
        include_style_stress=parsed.include_style_stress,
        out_report=parsed.out_report,
    )
    report = build_v2_eval_prompts(config)
    print(f"Prompts written: {report['prompt_count']} (target: {config.target_count})")
    print(f"Families: {report['families']}")
    print(f"Output: {config.out}")
    if config.out_report:
        print(f"Report JSON: {report.get('report_json')}")
        print(f"Report MD:   {report.get('report_md')}")

"""Training manifest / caption sampling layer for semantic-v3 datasets.

Turns an *exported* semantic-v3 dataset (the layout produced by
:mod:`spritelab.dataset_maker.exporter`) into training-ready conditioning
examples. Each accepted sprite is expanded into ``variants_per_sprite``
JSONL rows, each carrying one grounded caption sampled under a caption
policy plus a record of which attributes were kept vs dropped (semantic
dropout). The goal is to teach a future generator *composition* -- a golden
sword is ``sword + gold + metal + protection + rpg icon style`` -- instead of
memorising exact object names.

Deterministic, offline: no VLM/LLM calls, no network, no GPU, no torch. Every
caption word is traceable to a semantic-v3 attribute token or the fixed style
vocabulary; nothing is invented.

See the semantic-v3 layer in :mod:`spritelab.harvest.semantic_v3`.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.harvest.semantic_v3 import (
    _CATEGORY_WORD as CATEGORY_WORD,
)
from spritelab.harvest.semantic_v3 import (
    _NEUTRAL_COLORS as NEUTRAL_COLORS,
)
from spritelab.harvest.semantic_v3 import (
    DEFAULT_NEGATIVE_TAGS,
    SemanticAttributes,
    SemanticV3Record,
    semantic_v3_from_json,
)

SCHEMA_VERSION = "training_manifest_v1.0"

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

CAPTION_POLICIES: tuple[str, ...] = ("object_only", "style_aware", "attribute", "minimal", "mixed")

# Which caption types each policy draws from (in preference order for mixed).
POLICY_TYPES: dict[str, tuple[str, ...]] = {
    "object_only": ("object", "minimal"),
    "style_aware": ("style_aware",),
    "attribute": ("attribute",),
    "minimal": ("minimal", "object"),
    "mixed": ("object", "style_aware", "attribute", "minimal"),
}

# Positive-prompt content that must never appear in a training caption.
FORBIDDEN_CAPTION_CONTENT: tuple[str, ...] = (
    "photorealistic",
    "realistic photo",
    "watermark",
    "text overlay",
)

MAX_CAPTION_LENGTH = 220

# Attribute groups tracked for dropout, mapped to their dropout op suffix.
_DROPOUT_GROUPS: dict[str, str] = {
    "colors": "color",
    "materials": "material",
    "effects": "effect",
    "shapes": "shape",
    "state": "state",
    "function": "function",
}

# Materials that do not read as a solid substance in a "made of X" clause.
_NON_SOLID_MATERIALS = frozenset({"liquid", "organic"})


@dataclass(frozen=True)
class CaptionVariant:
    """One sampled caption plus its provenance and dropout accounting."""

    caption: str
    caption_type: str  # object | style_aware | attribute | minimal
    caption_source: str
    kept: dict[str, list[str]]
    dropped: dict[str, list[str]]
    dropout_ops: tuple[str, ...]


@dataclass(frozen=True)
class TrainingManifestResult:
    dataset_dir: Path
    rows: list[dict[str, Any]]
    caption_policy: str
    variants_per_sprite: int
    seed: int
    source_records: int
    unique_sprites: int
    split_rows: dict[str, int]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_manifest(
    dataset_dir: Path,
    *,
    variants_per_sprite: int,
    caption_policy: str,
    seed: int,
) -> TrainingManifestResult:
    """Expand a semantic-v3 dataset into training conditioning rows."""

    dataset_dir = Path(dataset_dir)
    if caption_policy not in POLICY_TYPES:
        raise ValueError(f"unknown caption policy {caption_policy!r}; expected one of {sorted(POLICY_TYPES)}")
    variants_per_sprite = max(1, int(variants_per_sprite))

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    source_records = 0
    split_rows: dict[str, int] = dict.fromkeys(SPLIT_NAMES, 0)

    for split in SPLIT_NAMES:
        manifest_path = dataset_dir / f"manifest_{split}.jsonl"
        if not manifest_path.is_file():
            warnings.append(f"missing split manifest: {manifest_path.name}")
            continue
        records = _read_jsonl(manifest_path)
        npz_rows = _npz_sprite_rows(dataset_dir / f"{split}.npz")
        if npz_rows is None:
            warnings.append(f"missing or unreadable split npz: {split}.npz")
            npz_rows = {}

        for manifest_row, record in enumerate(records):
            source_records += 1
            sprite_id = str(record.get("sprite_id", "")).strip()
            if not sprite_id:
                warnings.append(f"{manifest_path.name}:{manifest_row}: record has empty sprite_id")
                continue
            npz_row = npz_rows.get(sprite_id)
            if npz_row is None:
                warnings.append(f"{sprite_id}: not present in {split}.npz raster array")

            semantic = _resolve_semantic_record(record)
            rng = random.Random(f"{seed}:{caption_policy}:{sprite_id}")
            variants = sample_caption_variants(semantic, policy=caption_policy, rng=rng, count=variants_per_sprite)
            for variant_index, variant in enumerate(variants):
                rows.append(
                    _build_row(
                        dataset_dir=dataset_dir,
                        manifest_file=manifest_path.name,
                        manifest_row=manifest_row,
                        split=split,
                        npz_row=npz_row,
                        record=record,
                        semantic=semantic,
                        variant=variant,
                        variant_index=variant_index,
                        caption_policy=caption_policy,
                        seed=seed,
                    )
                )
                split_rows[split] += 1

    unique_sprites = len({row["sprite_id"] for row in rows})
    return TrainingManifestResult(
        dataset_dir=dataset_dir,
        rows=rows,
        caption_policy=caption_policy,
        variants_per_sprite=variants_per_sprite,
        seed=seed,
        source_records=source_records,
        unique_sprites=unique_sprites,
        split_rows=split_rows,
        warnings=warnings,
    )


def sample_caption_variants(
    semantic: SemanticV3Record,
    *,
    policy: str,
    rng: random.Random,
    count: int,
) -> list[CaptionVariant]:
    """Sample ``count`` caption variants under ``policy``, deterministic in ``rng``.

    Prefers distinct captions and balances across the policy's caption types;
    falls back to allowing repeats only when the candidate pool is too small.
    """

    count = max(1, int(count))
    candidates = _candidate_variants(semantic)
    if not candidates:
        candidates = [_fallback_variant(semantic)]

    wanted_types = POLICY_TYPES.get(policy, POLICY_TYPES["mixed"])
    pool = [variant for variant in candidates if variant.caption_type in wanted_types]
    if not pool:
        pool = list(candidates)

    by_type: dict[str, list[CaptionVariant]] = {}
    for variant in pool:
        by_type.setdefault(variant.caption_type, []).append(variant)
    for variants in by_type.values():
        rng.shuffle(variants)

    order_types = list(by_type.keys())
    rng.shuffle(order_types)

    selected: list[CaptionVariant] = []
    seen: set[str] = set()
    cursor = 0
    while len(selected) < count and any(by_type.values()):
        caption_type = order_types[cursor % len(order_types)]
        cursor += 1
        bucket = by_type.get(caption_type)
        if not bucket:
            continue
        variant = bucket.pop(0)
        if variant.caption.lower() in seen:
            continue
        seen.add(variant.caption.lower())
        selected.append(variant)

    if len(selected) < count:
        filler = list(pool) or list(candidates)
        rng.shuffle(filler)
        index = 0
        while len(selected) < count:
            selected.append(filler[index % len(filler)])
            index += 1

    return selected[:count]


# ---------------------------------------------------------------------------
# Candidate caption generation
# ---------------------------------------------------------------------------


def _candidate_variants(semantic: SemanticV3Record) -> list[CaptionVariant]:
    attributes = semantic.attributes
    base_noun = _open(semantic.base_object)
    open_name = semantic.open_name or base_noun
    if not base_noun:
        base_noun = open_name

    colors = list(attributes.colors)
    color = _lead_color(colors)
    materials = list(attributes.materials)
    material = materials[0] if materials else ""
    solid = _first_solid(materials)
    effect = _first_grounded(attributes.effects, base_noun)
    shape = attributes.shapes[0] if attributes.shapes else ""
    state = attributes.state[0] if attributes.state else ""
    is_icon = "rpg_icon" in attributes.style
    fantasy = "fantasy" in attributes.mood
    category_word = CATEGORY_WORD.get(semantic.category, "")

    groups = _attribute_groups(attributes)

    seeds: list[tuple[str, str, str]] = []  # (text, caption_type, caption_source)

    def add(text: str, caption_type: str, source: str) -> None:
        seeds.append((_clean(text), caption_type, source))

    # object / minimal ------------------------------------------------------
    add(open_name, "object", "semantic_v3.open_name")
    add(base_noun, "minimal", "semantic_v3.base_object")
    add(_join(open_name, "icon") if is_icon else open_name, "object", "synthesized.object_icon")
    if category_word:
        add(_join(open_name, category_word), "object", "synthesized.object_category")
    for alias in semantic.aliases:
        add(_open(alias), "object", "semantic_v3.aliases")
    if color:
        add(_join(color, base_noun), "minimal", "synthesized.color_object")

    # style-aware -----------------------------------------------------------
    style_color = "" if _color_in(color, open_name) else color
    if is_icon:
        add(
            _join("32x32 pixel art", "fantasy RPG" if fantasy else "", style_color, open_name, "icon"),
            "style_aware",
            "synthesized.style",
        )
        add(_join("pixel art", open_name, "icon"), "style_aware", "synthesized.style")
        add(_join("32x32", open_name, "icon"), "style_aware", "synthesized.style")
        if fantasy:
            add(_join("fantasy", open_name, "icon"), "style_aware", "synthesized.style")
    else:
        add(_join("32x32 pixel art", style_color, open_name), "style_aware", "synthesized.style")
    add(_join("32x32 pixel art", base_noun), "style_aware", "synthesized.style")
    if category_word and is_icon:
        add(_join("32x32 pixel art", category_word, "icon"), "style_aware", "synthesized.style")
    for phrase in semantic.prompt_phrases:
        add(phrase, "style_aware", "semantic_v3.prompt_phrases")

    # attribute-decomposed --------------------------------------------------
    decomposed = _join(effect, color, state, base_noun)
    if solid and _open(solid) != color:
        decomposed = _join(decomposed, "made of", _open(solid))
    add(decomposed, "attribute", "synthesized.attribute")
    if color:
        add(_join(color, base_noun), "attribute", "synthesized.attribute")
    if material:
        add(_join(_open(material), base_noun), "attribute", "synthesized.attribute")
    if shape:
        add(_join(_open(shape), base_noun), "attribute", "synthesized.attribute")
    if effect:
        add(_join(effect, base_noun), "attribute", "synthesized.attribute")
    if color and material and _open(material) != color:
        add(_join(color, _open(material), base_noun), "attribute", "synthesized.attribute")
    if category_word:
        add(_join(category_word, "icon") if is_icon else category_word, "attribute", "synthesized.category")

    # semantic-v3 captions (classified) ------------------------------------
    for caption in semantic.captions:
        add(caption, _classify_caption(caption, base_noun, open_name), "semantic_v3.captions")

    return _finalize_candidates(seeds, groups, base_noun)


def _finalize_candidates(
    seeds: Sequence[tuple[str, str, str]],
    groups: Mapping[str, tuple[str, ...]],
    base_noun: str,
) -> list[CaptionVariant]:
    variants: list[CaptionVariant] = []
    seen: set[str] = set()
    for text, caption_type, source in seeds:
        if not text or len(text) > MAX_CAPTION_LENGTH:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kept, dropped, ops = _dropout_accounting(text, groups, base_noun)
        variants.append(
            CaptionVariant(
                caption=text,
                caption_type=caption_type,
                caption_source=source,
                kept=kept,
                dropped=dropped,
                dropout_ops=ops,
            )
        )
    return variants


def _fallback_variant(semantic: SemanticV3Record) -> CaptionVariant:
    text = _clean(semantic.open_name or _open(semantic.base_object) or semantic.object_name or "icon")
    return CaptionVariant(
        caption=text or "icon",
        caption_type="minimal",
        caption_source="fallback",
        kept={},
        dropped={},
        dropout_ops=("minimal_base_object",),
    )


# ---------------------------------------------------------------------------
# Dropout accounting
# ---------------------------------------------------------------------------


def _attribute_groups(attributes: SemanticAttributes) -> dict[str, tuple[str, ...]]:
    return {
        "colors": attributes.colors,
        "materials": attributes.materials,
        "effects": attributes.effects,
        "shapes": attributes.shapes,
        "state": attributes.state,
        "function": attributes.function,
    }


def _dropout_accounting(
    caption: str,
    groups: Mapping[str, tuple[str, ...]],
    base_noun: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]], tuple[str, ...]]:
    lowered = caption.lower()
    kept: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    ops: list[str] = []
    for group, tokens in groups.items():
        present = [token for token in tokens if _token_in(token, lowered)]
        absent = [token for token in tokens if token not in present]
        kept[group] = present
        dropped[group] = absent
        if tokens and not present:
            ops.append(f"drop_{_DROPOUT_GROUPS[group]}")
    if lowered == base_noun.lower():
        ops.append("minimal_base_object")
    return kept, dropped, tuple(ops)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _build_row(
    *,
    dataset_dir: Path,
    manifest_file: str,
    manifest_row: int,
    split: str,
    npz_row: int | None,
    record: Mapping[str, Any],
    semantic: SemanticV3Record,
    variant: CaptionVariant,
    variant_index: int,
    caption_policy: str,
    seed: int,
) -> dict[str, Any]:
    attributes = semantic.attributes
    label_v2 = record.get("label_v2") if isinstance(record.get("label_v2"), Mapping) else {}
    bucket = str(label_v2.get("bucket", "")) if isinstance(label_v2, Mapping) else ""

    return {
        "schema_version": SCHEMA_VERSION,
        "sprite_id": str(record.get("sprite_id", "")),
        "split": split,
        "npz_file": f"{split}.npz",
        "npz_row": int(npz_row) if npz_row is not None else -1,
        "category": str(record.get("category", "")) or semantic.category,
        "object_name": str(record.get("object_name", "")) or semantic.object_name,
        "base_object": semantic.base_object,
        "caption": variant.caption,
        "caption_type": variant.caption_type,
        "caption_source": variant.caption_source,
        "conditioning": {
            "semantic_v3": {
                "base_object": semantic.base_object,
                "open_name": semantic.open_name,
                "attributes": {
                    "colors": list(attributes.colors),
                    "materials": list(attributes.materials),
                    "shapes": list(attributes.shapes),
                    "effects": list(attributes.effects),
                    "state": list(attributes.state),
                    "function": list(attributes.function),
                },
            },
            "kept_attributes": {group: list(values) for group, values in variant.kept.items()},
            "dropped_attributes": {group: list(values) for group, values in variant.dropped.items()},
            "dropout_policy": "balanced",
            "dropout_ops": list(variant.dropout_ops),
        },
        "dropout_mask": _dropout_mask(variant),
        "negative_tags": list(semantic.negative_tags or DEFAULT_NEGATIVE_TAGS),
        "source": {
            "dataset_dir": str(dataset_dir).replace("\\", "/"),
            "manifest_file": manifest_file,
            "manifest_row": manifest_row,
        },
        "audit": {
            "label_v2_bucket": bucket,
            "semantic_schema_version": semantic.schema_version,
            "caption_policy": caption_policy,
            "variant_index": variant_index,
            "seed": seed,
        },
    }


def _dropout_mask(variant: CaptionVariant) -> dict[str, list[str]]:
    mask: dict[str, list[str]] = {}
    for group in variant.kept:
        mask[f"kept_{group}"] = list(variant.kept.get(group, []))
        mask[f"dropped_{group}"] = list(variant.dropped.get(group, []))
    return mask


# ---------------------------------------------------------------------------
# Loading / IO helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _npz_sprite_rows(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "sprite_id" not in data.files:
                return {}
            sprite_ids = [str(value) for value in np.asarray(data["sprite_id"])]
    except Exception:  # pragma: no cover - defensive
        return None
    return {sprite_id: row for row, sprite_id in enumerate(sprite_ids)}


def _resolve_semantic_record(record: Mapping[str, Any]) -> SemanticV3Record:
    semantic = record.get("semantic_v3")
    parsed = semantic_v3_from_json(semantic) if isinstance(semantic, Mapping) else None
    if parsed is not None and (parsed.base_object or parsed.open_name):
        return parsed
    object_name = str(record.get("object_name", ""))
    return SemanticV3Record(
        schema_version="semantic_v3.0",
        category=str(record.get("category", "")),
        object_name=object_name,
        base_object=object_name,
        open_name=object_name.replace("_", " ").strip(),
        attributes=SemanticAttributes(),
        negative_tags=DEFAULT_NEGATIVE_TAGS,
    )


def write_training_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_per_split_manifests(out_path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    """Write ``training_manifest_{split}.jsonl`` beside ``out_path``."""

    stem = out_path.stem
    written: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        split_rows = [row for row in rows if row.get("split") == split]
        split_path = out_path.with_name(f"{stem}_{split}.jsonl")
        write_training_manifest(split_path, split_rows)
        written[split] = split_path
    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_training_manifest(result: TrainingManifestResult) -> dict[str, Any]:
    rows = result.rows
    caption_types: Counter[str] = Counter()
    caption_sources: Counter[str] = Counter()
    base_objects: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    materials: Counter[str] = Counter()
    dropout_ops: Counter[str] = Counter()
    per_sprite: Counter[str] = Counter()
    caption_length_total = 0

    for row in rows:
        caption_types[str(row.get("caption_type", ""))] += 1
        caption_sources[str(row.get("caption_source", ""))] += 1
        base_objects[str(row.get("base_object", ""))] += 1
        per_sprite[str(row.get("sprite_id", ""))] += 1
        caption_length_total += len(str(row.get("caption", "")))
        conditioning = row.get("conditioning") if isinstance(row.get("conditioning"), Mapping) else {}
        semantic = conditioning.get("semantic_v3") if isinstance(conditioning, Mapping) else {}
        attributes = semantic.get("attributes") if isinstance(semantic, Mapping) else {}
        if isinstance(attributes, Mapping):
            for value in attributes.get("colors") or ():
                colors[str(value)] += 1
            for value in attributes.get("materials") or ():
                materials[str(value)] += 1
        for op in conditioning.get("dropout_ops") or () if isinstance(conditioning, Mapping) else ():
            dropout_ops[str(op)] += 1

    variant_counts = list(per_sprite.values())
    return {
        "dataset_dir": str(result.dataset_dir).replace("\\", "/"),
        "caption_policy": result.caption_policy,
        "variants_per_sprite": result.variants_per_sprite,
        "seed": result.seed,
        "source_records": result.source_records,
        "total_rows": len(rows),
        "unique_sprites": result.unique_sprites,
        "split_rows": dict(result.split_rows),
        "variants_per_sprite_min": min(variant_counts) if variant_counts else 0,
        "variants_per_sprite_max": max(variant_counts) if variant_counts else 0,
        "variants_per_sprite_avg": (sum(variant_counts) / len(variant_counts)) if variant_counts else 0.0,
        "average_caption_length": (caption_length_total / len(rows)) if rows else 0.0,
        "caption_type_counts": dict(caption_types.most_common()),
        "caption_source_counts": dict(caption_sources.most_common()),
        "dropout_op_counts": dict(dropout_ops.most_common()),
        "top_base_objects": dict(base_objects.most_common(25)),
        "top_colors": dict(colors.most_common(15)),
        "top_materials": dict(materials.most_common(15)),
        "warnings": list(result.warnings),
    }


def format_training_manifest_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Training Manifest Report",
        "",
        f"Dataset: `{summary.get('dataset_dir', '')}`",
        f"Caption policy: {summary.get('caption_policy', '')}",
        f"Seed: {summary.get('seed', '')}",
        f"Source records: {int(summary.get('source_records', 0))}",
        f"Total rows: {int(summary.get('total_rows', 0))}",
        f"Unique sprites: {int(summary.get('unique_sprites', 0))}",
        "Variants per sprite: "
        f"min={int(summary.get('variants_per_sprite_min', 0))} "
        f"max={int(summary.get('variants_per_sprite_max', 0))} "
        f"avg={float(summary.get('variants_per_sprite_avg', 0.0)):.1f}",
        f"Average caption length: {float(summary.get('average_caption_length', 0.0)):.1f}",
        "",
        "## Split rows",
    ]
    for split in SPLIT_NAMES:
        lines.append(f"- {split}: {int(dict(summary.get('split_rows') or {}).get(split, 0))}")
    lines.extend(["", "## Caption types"])
    for name, count in dict(summary.get("caption_type_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Caption sources"])
    for name, count in dict(summary.get("caption_source_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Dropout ops"])
    dropout = dict(summary.get("dropout_op_counts") or {})
    if dropout:
        for name, count in dropout.items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Top base objects"])
    for name, count in dict(summary.get("top_base_objects") or {}).items():
        lines.append(f"- {name}: {count}")
    for title, key in (("Top colors", "top_colors"), ("Top materials", "top_materials")):
        lines.extend(["", f"## {title}"])
        for name, count in dict(summary.get(key) or {}).items():
            lines.append(f"- {name}: {count}")
    lines.extend(["", "## Warnings"])
    warnings = list(summary.get("warnings") or [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def write_training_manifest_reports(summary: Mapping[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_training_manifest_report(summary), encoding="utf-8")


# ---------------------------------------------------------------------------
# Small string helpers
# ---------------------------------------------------------------------------


def _open(value: str) -> str:
    return str(value).replace("_", " ").strip()


def _clean(text: str) -> str:
    """Normalise whitespace and collapse consecutive duplicate words.

    Composed captions can repeat a token when the same value surfaces as two
    attributes (e.g. gold as both colour and material -> "gold gold axe");
    collapsing keeps every caption grounded and free of "red red potion"
    style repeats.
    """

    words = str(text).split()
    collapsed: list[str] = []
    for word in words:
        if collapsed and _word_core(collapsed[-1]) == _word_core(word) and _word_core(word):
            # Keep the later token so trailing punctuation (commas) survives.
            collapsed[-1] = word
            continue
        collapsed.append(word)
    return " ".join(collapsed)


def _word_core(word: str) -> str:
    return word.strip(",.;:").lower()


def _join(*values: str) -> str:
    return " ".join(word for word in (str(value).strip() for value in values) if word)


def _lead_color(colors: Sequence[str]) -> str:
    for value in colors:
        if value not in NEUTRAL_COLORS:
            return value
    return colors[0] if colors else ""


def _first_solid(materials: Sequence[str]) -> str:
    for material in materials:
        if material not in _NON_SOLID_MATERIALS:
            return material
    return ""


def _first_grounded(effects: Sequence[str], base_noun: str) -> str:
    base_tokens = set(base_noun.lower().split())
    for effect in effects:
        if effect.lower() not in base_tokens:
            return effect
    return ""


def _color_in(color: str, open_name: str) -> bool:
    if not color:
        return True
    return color in open_name.lower()


def _token_in(token: str, lowered_caption: str) -> bool:
    phrase = token.replace("_", " ").lower()
    return bool(phrase) and phrase in lowered_caption


def _classify_caption(caption: str, base_noun: str, open_name: str) -> str:
    lowered = caption.lower()
    if any(marker in lowered for marker in ("pixel art", "32x32", "transparent background", "centered", "icon")):
        return "style_aware"
    if "made of" in lowered:
        return "attribute"
    if lowered in {base_noun.lower(), open_name.lower()}:
        return "object"
    return "attribute"

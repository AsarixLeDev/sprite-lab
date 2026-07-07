"""Fixed evaluation prompt-set generation for a future 32x32 generator.

Builds a deterministic JSONL of prompts *before* any model training, grouped
into categories that probe different generalisation abilities:

* ``seen_object`` -- objects that actually appear in the dataset;
* ``unseen_composition`` -- attribute recombinations whose individual factors
  are all present in the dataset but whose exact combination is not;
* ``creative_concept`` -- ``modifier + substance`` names decomposed through the
  shared :mod:`spritelab.harvest.creative_concepts` grammar;
* ``style_stress`` -- pure style / format constraints;
* ``negative_control`` -- out-of-scope prompts a 32x32 icon model should reject.

Deterministic, offline, no torch. This only writes a prompt file; it never runs
a model.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from spritelab.harvest.creative_concepts import parse_creative_concept
from spritelab.harvest.semantic_extractors import COLOR_TOKENS, MATERIAL_TOKENS, SHAPE_TOKENS
from spritelab.harvest.semantic_v3 import semantic_v3_from_json

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# Creative concept names decomposed via the shared grammar.
CREATIVE_CONCEPT_NAMES: tuple[str, ...] = (
    "charged_sinew",
    "calming_spores",
    "frozen_resin",
    "glowing_dreamroot",
    "moonlit_resin",
    "bitter_dreamroot",
    "cursed_bone",
    "blessed_crystal",
    "mossy_root",
    "burning_ash",
)

STYLE_STRESS_PROMPTS: tuple[str, ...] = (
    "32x32 pixel art icon, centered, transparent background, black outline",
    "32x32 pixel art item icon, single object, no background",
    "centered 32x32 fantasy RPG icon, crisp pixel edges",
)

NEGATIVE_CONTROL_PROMPTS: tuple[str, ...] = (
    "photorealistic dolphin photo",
    "large cinematic landscape",
    "text logo",
    "high resolution portrait photograph",
)


@dataclass(frozen=True)
class EvalPromptsResult:
    dataset_dir: Path
    prompts: list[dict[str, Any]]
    seed: int
    warnings: list[str] = field(default_factory=list)


def build_eval_prompts(
    dataset_dir: Path,
    *,
    seed: int,
    seen_object_count: int = 40,
    unseen_composition_count: int = 40,
) -> EvalPromptsResult:
    """Generate a fixed evaluation prompt set from a semantic-v3 dataset."""

    dataset_dir = Path(dataset_dir)
    warnings: list[str] = []
    records = _load_semantic_records(dataset_dir, warnings)

    prompts: list[dict[str, Any]] = []
    prompts.extend(_seen_object_prompts(records, count=seen_object_count, seed=seed))
    prompts.extend(_unseen_composition_prompts(records, count=unseen_composition_count, seed=seed))
    prompts.extend(_creative_concept_prompts())
    prompts.extend(_style_stress_prompts())
    prompts.extend(_negative_control_prompts())

    return EvalPromptsResult(dataset_dir=dataset_dir, prompts=prompts, seed=seed, warnings=warnings)


# ---------------------------------------------------------------------------
# Prompt category builders
# ---------------------------------------------------------------------------


def _seen_object_prompts(
    records: Sequence[_Semantic], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda item: item.sprite_id)
    rng = Random(f"{seed}:seen_object")
    rng.shuffle(ordered)
    prompts: list[dict[str, Any]] = []
    for index, record in enumerate(ordered[: max(0, count)]):
        prompt_text = record.open_name or record.base_object
        if not prompt_text:
            continue
        prompts.append(
            {
                "prompt_id": f"seen_object_{index:04d}",
                "category": "seen_object",
                "prompt": prompt_text,
                "target_semantics": {
                    "base_object": record.base_object,
                    "attributes": _target_attributes(
                        colors=record.colors, shapes=record.shapes, materials=record.materials
                    ),
                },
                "seen_factors": {
                    "base_object": True,
                    "color": bool(record.colors),
                    "shape": bool(record.shapes),
                    "exact_combination_seen": True,
                },
            }
        )
    return prompts


def _unseen_composition_prompts(
    records: Sequence[_Semantic], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if not records:
        return []
    base_objects = _sorted_seen(record.base_object for record in records)
    colors = _sorted_seen(
        token for record in records for token in record.colors if token in COLOR_TOKENS
    )
    shapes = _sorted_seen(
        token for record in records for token in record.shapes if token in SHAPE_TOKENS
    )
    materials = _sorted_seen(
        token for record in records for token in record.materials if token in MATERIAL_TOKENS
    )
    if not base_objects or not colors:
        return []

    combos_by_base = _combos_by_base(records)
    rng = Random(f"{seed}:unseen_composition")

    prompts: list[dict[str, Any]] = []
    seen_prompt_text: set[str] = set()
    attempts = 0
    max_attempts = max(count * 40, 400)
    while len(prompts) < max(0, count) and attempts < max_attempts:
        attempts += 1
        base = rng.choice(base_objects)
        color = rng.choice(colors)
        factors = {"color": color}
        parts = [color]
        if shapes and rng.random() < 0.5:
            shape = rng.choice(shapes)
            factors["shape"] = shape
            parts.insert(0, shape)
        if materials and rng.random() < 0.35:
            material = rng.choice(materials)
            factors["material"] = material
            parts.insert(0, material)
        prompt_text = " ".join([*parts, base.replace("_", " ")]).strip()
        if not prompt_text or prompt_text in seen_prompt_text:
            continue

        combo_tokens = set(factors.values())
        exact_seen = any(combo_tokens <= attrs for attrs in combos_by_base.get(base, []))
        if exact_seen:
            continue  # keep only genuinely novel compositions
        seen_prompt_text.add(prompt_text)
        prompts.append(
            {
                "prompt_id": f"unseen_composition_{len(prompts):04d}",
                "category": "unseen_composition",
                "prompt": prompt_text,
                "target_semantics": {
                    "base_object": base,
                    "attributes": _target_attributes(
                        colors=[color],
                        shapes=[factors["shape"]] if "shape" in factors else [],
                        materials=[factors["material"]] if "material" in factors else [],
                    ),
                },
                "seen_factors": {
                    "base_object": True,
                    "color": True,
                    "shape": "shape" in factors,
                    "material": "material" in factors,
                    "exact_combination_seen": False,
                },
            }
        )
    return prompts


def _creative_concept_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for index, name in enumerate(CREATIVE_CONCEPT_NAMES):
        concept = parse_creative_concept(name)
        prompts.append(
            {
                "prompt_id": f"creative_concept_{index:04d}",
                "category": "creative_concept",
                "prompt": name.replace("_", " "),
                "target_semantics": {
                    "base_object": concept.base_object,
                    "attributes": _target_attributes(
                        colors=list(concept.attributes.colors),
                        shapes=list(concept.attributes.shapes),
                        materials=list(concept.attributes.materials),
                        effects=list(concept.attributes.effects),
                        mood=list(concept.attributes.mood),
                    ),
                },
                "seen_factors": {
                    "base_object": bool(concept.base_object),
                    "recognized": bool(concept.recognized),
                    "exact_combination_seen": False,
                },
            }
        )
    return prompts


def _style_stress_prompts() -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": f"style_stress_{index:04d}",
            "category": "style_stress",
            "prompt": prompt,
            "target_semantics": {"base_object": "", "attributes": {}},
            "seen_factors": {"exact_combination_seen": False},
        }
        for index, prompt in enumerate(STYLE_STRESS_PROMPTS)
    ]


def _negative_control_prompts() -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": f"negative_control_{index:04d}",
            "category": "negative_control",
            "prompt": prompt,
            "target_semantics": {"base_object": "", "attributes": {}},
            "seen_factors": {"in_scope": False, "exact_combination_seen": False},
        }
        for index, prompt in enumerate(NEGATIVE_CONTROL_PROMPTS)
    ]


# ---------------------------------------------------------------------------
# Semantic record loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Semantic:
    sprite_id: str
    base_object: str
    open_name: str
    colors: tuple[str, ...]
    shapes: tuple[str, ...]
    materials: tuple[str, ...]


def _load_semantic_records(dataset_dir: Path, warnings: list[str]) -> list[_Semantic]:
    records: list[_Semantic] = []
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if not path.is_file():
            warnings.append(f"missing split manifest: {path.name}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            semantic = record.get("semantic_v3")
            parsed = semantic_v3_from_json(semantic) if isinstance(semantic, Mapping) else None
            base_object = parsed.base_object if parsed else str(record.get("object_name", ""))
            if not base_object:
                continue
            attributes = parsed.attributes if parsed else None
            records.append(
                _Semantic(
                    sprite_id=str(record.get("sprite_id", "")),
                    base_object=base_object,
                    open_name=(parsed.open_name if parsed else "") or base_object.replace("_", " "),
                    colors=attributes.colors if attributes else (),
                    shapes=attributes.shapes if attributes else (),
                    materials=attributes.materials if attributes else (),
                )
            )
    return records


def _combos_by_base(records: Sequence[_Semantic]) -> dict[str, list[set[str]]]:
    combos: dict[str, list[set[str]]] = {}
    for record in records:
        tokens = {*record.colors, *record.shapes, *record.materials}
        combos.setdefault(record.base_object, []).append(tokens)
    return combos


def _sorted_seen(values: Any) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _target_attributes(**groups: Sequence[str]) -> dict[str, list[str]]:
    return {name: [str(value) for value in values] for name, values in groups.items() if values}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_eval_prompts(result: EvalPromptsResult) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    unseen_true = 0
    unseen_total = 0
    for prompt in result.prompts:
        category = str(prompt.get("category", ""))
        category_counts[category] += 1
        if category == "unseen_composition":
            unseen_total += 1
            if not prompt.get("seen_factors", {}).get("exact_combination_seen", True):
                unseen_true += 1
    return {
        "dataset_dir": str(result.dataset_dir).replace("\\", "/"),
        "seed": result.seed,
        "total_prompts": len(result.prompts),
        "category_counts": dict(sorted(category_counts.items())),
        "unseen_composition_novel": unseen_true,
        "unseen_composition_total": unseen_total,
        "warnings": list(result.warnings),
    }


def format_eval_prompts_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Eval Prompts Report",
        "",
        f"Dataset: `{summary.get('dataset_dir', '')}`",
        f"Seed: {summary.get('seed', '')}",
        f"Total prompts: {int(summary.get('total_prompts', 0))}",
        "",
        "## Category counts",
    ]
    for name, count in dict(summary.get("category_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## Unseen compositions",
            f"- novel combinations: {int(summary.get('unseen_composition_novel', 0))}"
            f" / {int(summary.get('unseen_composition_total', 0))}",
            "",
            "## Warnings",
        ]
    )
    warnings = list(summary.get("warnings") or [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def write_eval_prompts(path: Path, prompts: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(prompt), sort_keys=True) for prompt in prompts]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_eval_prompts_reports(summary: Mapping[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_eval_prompts_report(summary), encoding="utf-8")

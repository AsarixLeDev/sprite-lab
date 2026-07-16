"""Generic pack-level context for v3 prefill ranking."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.harvest.label_v3.field_prefill import filename_semantics


@dataclass(frozen=True)
class PackContext:
    pack_homogeneity_score: float
    pack_candidate_vocabulary: tuple[str, ...]
    pack_prior_strength: float
    category_distribution: dict[str, float]
    filename_token_distribution: dict[str, int]
    repeated_prefixes: tuple[str, ...] = ()
    repeated_suffixes: tuple[str, ...] = ()
    variant_groups: dict[str, tuple[str, ...]] | None = None
    image_size_consistency: float = 0.0
    recolor_groups: tuple[tuple[str, ...], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "pack_homogeneity_score": round(self.pack_homogeneity_score, 4),
            "pack_candidate_vocabulary": list(self.pack_candidate_vocabulary),
            "pack_prior_strength": round(self.pack_prior_strength, 4),
            "category_distribution": dict(self.category_distribution),
            "filename_token_distribution": dict(self.filename_token_distribution),
            "repeated_prefixes": list(self.repeated_prefixes),
            "repeated_suffixes": list(self.repeated_suffixes),
            "variant_groups": {k: list(v) for k, v in (self.variant_groups or {}).items()},
            "image_size_consistency": round(self.image_size_consistency, 4),
            "recolor_groups": [list(v) for v in self.recolor_groups],
        }


def analyze_pack_context(records: Sequence[Mapping[str, Any]]) -> PackContext:
    aliases: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    variants: dict[str, list[str]] = {}
    stems: list[str] = []
    for record in records:
        semantics = filename_semantics(record)
        alias = str(semantics.get("normalized_alias", ""))
        if alias:
            aliases[alias] += 1
        category = str(semantics.get("category", "")) or _source_category(record)
        if category:
            categories[category] += 1
        path = str(record.get("relative_path") or record.get("filename") or "")
        stem = re.sub(r"\.[^.]+$", "", path.replace("\\", "/").rsplit("/", 1)[-1].lower())
        stems.append(stem)
        for token in re.findall(r"[a-z]+", stem):
            tokens[token] += 1
        base = re.sub(r"(?:[_-]?\d+|[_-]?[a-z])$", "", stem)
        variants.setdefault(base, []).append(str(record.get("sprite_id", stem)))
        try:
            size = (int(record.get("width", 0)), int(record.get("height", 0)))
            if all(size):
                sizes[size] += 1
        except (TypeError, ValueError):
            pass
    total = max(1, len(records))
    category_ratio = max(categories.values(), default=0) / total
    size_ratio = max(sizes.values(), default=0) / total
    # A pack is homogeneous when semantic category and sprite geometry agree;
    # a single repeated archive/path token is intentionally not another vote.
    homogeneity = min(1.0, 0.75 * category_ratio + 0.25 * size_ratio) if categories else 0.25 * size_ratio
    strength = max(0.0, min(0.9, (homogeneity - 0.45) / 0.55 * 0.9))
    category_dist = {k: round(v / total, 4) for k, v in sorted(categories.items())}
    repeated_prefixes = tuple(
        sorted(
            {s.split("_", 1)[0] for s in stems if "_" in s and sum(x.startswith(s.split("_", 1)[0]) for x in stems) > 1}
        )
    )
    repeated_suffixes = tuple(
        sorted(
            {
                s.rsplit("_", 1)[-1]
                for s in stems
                if "_" in s and sum(x.endswith(s.rsplit("_", 1)[-1]) for x in stems) > 1
            }
        )
    )
    return PackContext(
        pack_homogeneity_score=homogeneity,
        pack_candidate_vocabulary=tuple(name for name, _ in aliases.most_common(32)),
        pack_prior_strength=strength,
        category_distribution=category_dist,
        filename_token_distribution=dict(tokens.most_common(64)),
        repeated_prefixes=repeated_prefixes,
        repeated_suffixes=repeated_suffixes,
        variant_groups={k: tuple(v) for k, v in sorted(variants.items()) if len(v) > 1},
        image_size_consistency=size_ratio,
    )


def _source_category(record: Mapping[str, Any]) -> str:
    """Infer only a broad pack subject from source/profile tokens."""
    text = " ".join(str(record.get(key, "")) for key in ("source_id", "source_name", "pack_name")).lower()
    auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
    safe = auto.get("label_v2_safe_prefill") if isinstance(auto.get("label_v2_safe_prefill"), Mapping) else {}
    text += " " + " ".join(str(v) for v in safe.get("evidence") or ())
    rules = (
        ("gem", ("gem", "crystal")),
        ("weapon", ("weapon", "sword")),
        ("armor", ("armor", "armour", "shield", "headgear")),
        ("tool", ("tool", "key", "farming")),
        ("food", ("food", "potion")),
        ("plant", ("plant", "mushroom")),
    )
    matches = [category for category, needles in rules if any(re.search(rf"\b{re.escape(n)}\b", text) for n in needles)]
    return matches[0] if len(set(matches)) == 1 else ""


def pack_outlier_score(record: Mapping[str, Any], context: PackContext) -> float:
    semantics = filename_semantics(record)
    alias = str(semantics.get("normalized_alias", ""))
    vocabulary = set(context.pack_candidate_vocabulary)
    if not alias or not vocabulary:
        return 0.0
    return 0.0 if alias in vocabulary else context.pack_prior_strength

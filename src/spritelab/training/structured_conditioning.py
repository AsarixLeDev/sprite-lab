"""Structured semantic conditioning extraction for challenger generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNKNOWN_TOKEN = "<unk>"

ID_FIELDS: tuple[str, ...] = (
    "category_id",
    "object_id",
    "base_object_id",
    "primary_color_id",
)
MULTI_HOT_FIELDS: tuple[str, ...] = (
    "color_multi_hot",
    "material_multi_hot",
    "shape_multi_hot",
    "function_multi_hot",
    "style_multi_hot",
)
STRUCTURED_BATCH_KEYS: tuple[str, ...] = tuple(f"structured_{field}" for field in (*ID_FIELDS, *MULTI_HOT_FIELDS))

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class StructuredConditioningVocab:
    categories: tuple[str, ...]
    objects: tuple[str, ...]
    base_objects: tuple[str, ...]
    colors: tuple[str, ...]
    materials: tuple[str, ...]
    shapes: tuple[str, ...]
    functions: tuple[str, ...]
    styles: tuple[str, ...]

    def sizes(self) -> dict[str, int]:
        return {
            "category_vocab_size": len(self.categories),
            "object_vocab_size": len(self.objects),
            "base_object_vocab_size": len(self.base_objects),
            "color_vocab_size": len(self.colors),
            "material_vocab_size": len(self.materials),
            "shape_vocab_size": len(self.shapes),
            "function_vocab_size": len(self.functions),
            "style_vocab_size": len(self.styles),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "structured_conditioning_vocab_v1",
            "categories": list(self.categories),
            "objects": list(self.objects),
            "base_objects": list(self.base_objects),
            "colors": list(self.colors),
            "materials": list(self.materials),
            "shapes": list(self.shapes),
            "functions": list(self.functions),
            "styles": list(self.styles),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any] | None) -> "StructuredConditioningVocab | None":
        if not isinstance(data, Mapping):
            return None
        return cls(
            categories=_tokens_from_json(data.get("categories")),
            objects=_tokens_from_json(data.get("objects")),
            base_objects=_tokens_from_json(data.get("base_objects")),
            colors=_tokens_from_json(data.get("colors")),
            materials=_tokens_from_json(data.get("materials")),
            shapes=_tokens_from_json(data.get("shapes")),
            functions=_tokens_from_json(data.get("functions")),
            styles=_tokens_from_json(data.get("styles")),
        )

    @classmethod
    def empty(cls) -> "StructuredConditioningVocab":
        return cls(
            categories=(UNKNOWN_TOKEN,),
            objects=(UNKNOWN_TOKEN,),
            base_objects=(UNKNOWN_TOKEN,),
            colors=(UNKNOWN_TOKEN,),
            materials=(UNKNOWN_TOKEN,),
            shapes=(UNKNOWN_TOKEN,),
            functions=(UNKNOWN_TOKEN,),
            styles=(UNKNOWN_TOKEN,),
        )


def build_structured_conditioning_vocab(records: Iterable[Mapping[str, Any]]) -> StructuredConditioningVocab:
    categories: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    base_objects: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    materials: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    functions: Counter[str] = Counter()
    styles: Counter[str] = Counter()

    for record in records:
        fields = extract_structured_fields(record)
        _count(categories, fields["category"])
        _count(objects, fields["object_name"])
        _count(base_objects, fields["base_object"])
        for token in fields["colors"]:
            _count(colors, token)
        for token in fields["materials"]:
            _count(materials, token)
        for token in fields["shapes"]:
            _count(shapes, token)
        for token in fields["functions"]:
            _count(functions, token)
        for token in fields["styles"]:
            _count(styles, token)

    return StructuredConditioningVocab(
        categories=_ordered_vocab(categories),
        objects=_ordered_vocab(objects),
        base_objects=_ordered_vocab(base_objects),
        colors=_ordered_vocab(colors),
        materials=_ordered_vocab(materials),
        shapes=_ordered_vocab(shapes),
        functions=_ordered_vocab(functions),
        styles=_ordered_vocab(styles),
    )


def extract_structured_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    semantic = _semantic_v3(record)
    attributes = _semantic_attributes(record, semantic)
    colors = _token_list(
        attributes.get("colors")
        or record.get("colors")
        or record.get("color")
        or record.get("colour")
    )
    materials = _token_list(attributes.get("materials") or record.get("materials") or record.get("material"))
    shapes = _token_list(attributes.get("shapes") or record.get("shapes") or record.get("shape"))
    functions = _token_list(attributes.get("function") or record.get("function") or record.get("functions"))
    styles = _token_list(attributes.get("style") or record.get("style") or record.get("styles"))
    if not styles:
        styles = _token_list(record.get("caption_type"))
    object_name = _first_token(
        record.get("object_name"),
        record.get("prompt_object"),
        semantic.get("open_name") if isinstance(semantic, Mapping) else None,
    )
    base_object = _first_token(
        record.get("base_object"),
        semantic.get("base_object") if isinstance(semantic, Mapping) else None,
        object_name,
    )
    category = _first_token(
        record.get("category"),
        record.get("prompt_category"),
        semantic.get("category") if isinstance(semantic, Mapping) else None,
    )
    return {
        "category": category,
        "object_name": object_name,
        "base_object": base_object,
        "primary_color": colors[0] if colors else "",
        "colors": colors,
        "materials": materials,
        "shapes": shapes,
        "functions": functions,
        "styles": styles,
    }


def encode_structured_conditioning(
    record: Mapping[str, Any],
    vocab: StructuredConditioningVocab,
) -> dict[str, Any]:
    fields = extract_structured_fields(record)
    return {
        "category_id": _index(vocab.categories, fields["category"]),
        "object_id": _index(vocab.objects, fields["object_name"]),
        "base_object_id": _index(vocab.base_objects, fields["base_object"]),
        "primary_color_id": _index(vocab.colors, fields["primary_color"]),
        "color_multi_hot": _multi_hot(vocab.colors, fields["colors"]),
        "material_multi_hot": _multi_hot(vocab.materials, fields["materials"]),
        "shape_multi_hot": _multi_hot(vocab.shapes, fields["shapes"]),
        "function_multi_hot": _multi_hot(vocab.functions, fields["functions"]),
        "style_multi_hot": _multi_hot(vocab.styles, fields["styles"]),
        "structured_present": structured_fields_present(record),
    }


def structured_fields_present(record: Mapping[str, Any]) -> bool:
    fields = extract_structured_fields(record)
    return any(
        [
            fields["category"],
            fields["object_name"],
            fields["base_object"],
            fields["colors"],
            fields["materials"],
            fields["shapes"],
            fields["functions"],
            fields["styles"],
        ]
    )


def structured_prompt_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "category": 0,
        "object_name": 0,
        "base_object": 0,
        "colors": 0,
        "materials": 0,
        "shapes": 0,
        "function": 0,
        "style": 0,
    }
    present = 0
    for record in records:
        fields = extract_structured_fields(record)
        if structured_fields_present(record):
            present += 1
        for key in ("category", "object_name", "base_object"):
            if fields[key]:
                counts[key] += 1
        if fields["colors"]:
            counts["colors"] += 1
        if fields["materials"]:
            counts["materials"] += 1
        if fields["shapes"]:
            counts["shapes"] += 1
        if fields["functions"]:
            counts["function"] += 1
        if fields["styles"]:
            counts["style"] += 1
    return {
        "structured_fields_present": present > 0,
        "structured_present_count": present,
        "structured_field_counts": counts,
    }


def structured_vocab_from_checkpoint(checkpoint: Mapping[str, Any]) -> StructuredConditioningVocab | None:
    direct = StructuredConditioningVocab.from_json_dict(checkpoint.get("structured_conditioning_vocab"))
    if direct is not None:
        return direct
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, Mapping):
        return StructuredConditioningVocab.from_json_dict(train_config.get("structured_conditioning_vocab"))
    return None


def save_structured_conditioning_vocab(vocab: StructuredConditioningVocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _semantic_v3(record: Mapping[str, Any]) -> Mapping[str, Any]:
    conditioning = record.get("conditioning") if isinstance(record.get("conditioning"), Mapping) else {}
    nested = conditioning.get("semantic_v3") if isinstance(conditioning, Mapping) else {}
    if isinstance(nested, Mapping):
        return nested
    top = record.get("semantic_v3")
    return top if isinstance(top, Mapping) else {}


def _semantic_attributes(record: Mapping[str, Any], semantic: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = semantic.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes
    target = record.get("target_semantics") if isinstance(record.get("target_semantics"), Mapping) else {}
    target_attributes = target.get("attributes") if isinstance(target, Mapping) else {}
    return target_attributes if isinstance(target_attributes, Mapping) else {}


def _ordered_vocab(counter: Counter[str]) -> tuple[str, ...]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return (UNKNOWN_TOKEN, *(token for token, _count in ordered if token and token != UNKNOWN_TOKEN))


def _tokens_from_json(value: Any) -> tuple[str, ...]:
    tokens = tuple(str(item) for item in value or () if str(item))
    if not tokens or tokens[0] != UNKNOWN_TOKEN:
        return (UNKNOWN_TOKEN, *tuple(token for token in tokens if token != UNKNOWN_TOKEN))
    return tokens


def _count(counter: Counter[str], value: str) -> None:
    if value:
        counter[value] += 1


def _index(vocab: Sequence[str], token: str) -> int:
    if not token:
        return 0
    try:
        return list(vocab).index(token)
    except ValueError:
        return 0


def _multi_hot(vocab: Sequence[str], tokens: Sequence[str]) -> list[float]:
    result = [0.0] * len(vocab)
    if not tokens:
        return result
    index_by_token = {token: index for index, token in enumerate(vocab)}
    unknown = False
    for token in tokens:
        index = index_by_token.get(token)
        if index is None:
            unknown = True
        else:
            result[index] = 1.0
    if unknown and result:
        result[0] = 1.0
    return result


def _first_token(*values: Any) -> str:
    for value in values:
        tokens = _token_list(value)
        if tokens:
            return tokens[0]
    return ""


def _token_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        items: list[str] = []
        for nested in value.values():
            items.extend(_token_list(nested))
        return _dedupe(items)
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = []
        for item in value:
            items.extend(_token_list(item))
        return _dedupe(items)
    normalized = _normalize_token(str(value))
    return [normalized] if normalized else []


def _normalize_token(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _TOKEN_RE.sub("_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _dedupe(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result

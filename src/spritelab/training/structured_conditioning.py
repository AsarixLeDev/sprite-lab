"""Structured semantic conditioning extraction for challenger generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.training_quality import FIELD_VALUE_STATE_IDS, extract_training_quality

MISSING_TOKEN = "<missing>"
UNKNOWN_TOKEN = "<unk>"
STRUCTURED_VOCAB_SCHEMA_V1 = "structured_conditioning_vocab_v1"
STRUCTURED_VOCAB_SCHEMA_V2 = "structured_conditioning_vocab_v2"

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
STATUS_FIELDS: tuple[str, ...] = (
    "category_status_id",
    "object_status_id",
    "base_object_status_id",
    "primary_color_status_id",
    "color_status_id",
    "material_status_id",
    "shape_status_id",
    "function_status_id",
    "style_status_id",
)
STRUCTURED_BATCH_KEYS: tuple[str, ...] = tuple(
    f"structured_{field}" for field in (*ID_FIELDS, *MULTI_HOT_FIELDS, *STATUS_FIELDS)
)

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
    schema_version: str = STRUCTURED_VOCAB_SCHEMA_V2

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
            "field_status_vocab_size": len(FIELD_VALUE_STATE_IDS),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field_status_schema_version": "label_field_value_states_v1",
            "field_value_states": list(FIELD_VALUE_STATE_IDS),
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
    def from_json_dict(cls, data: Mapping[str, Any] | None) -> StructuredConditioningVocab | None:
        if not isinstance(data, Mapping):
            return None
        version = data.get("schema_version")
        if version != STRUCTURED_VOCAB_SCHEMA_V2:
            if version in {None, STRUCTURED_VOCAB_SCHEMA_V1}:
                raise ValueError(
                    "structured conditioning schema v1 requires the explicit adapt_schema_v1_vocab compatibility path"
                )
            raise ValueError(f"unsupported structured conditioning vocabulary schema: {version!r}")
        states = data.get("field_value_states")
        if states is not None and tuple(str(value) for value in states) != tuple(FIELD_VALUE_STATE_IDS):
            raise ValueError("unsupported structured field value-state mapping")
        return cls(
            categories=_tokens_from_json(data.get("categories"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            objects=_tokens_from_json(data.get("objects"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            base_objects=_tokens_from_json(data.get("base_objects"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            colors=_tokens_from_json(data.get("colors"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            materials=_tokens_from_json(data.get("materials"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            shapes=_tokens_from_json(data.get("shapes"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            functions=_tokens_from_json(data.get("functions"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            styles=_tokens_from_json(data.get("styles"), version=STRUCTURED_VOCAB_SCHEMA_V2),
            schema_version=STRUCTURED_VOCAB_SCHEMA_V2,
        )

    @classmethod
    def empty(cls) -> StructuredConditioningVocab:
        return cls(
            categories=(MISSING_TOKEN, UNKNOWN_TOKEN),
            objects=(MISSING_TOKEN, UNKNOWN_TOKEN),
            base_objects=(MISSING_TOKEN, UNKNOWN_TOKEN),
            colors=(MISSING_TOKEN, UNKNOWN_TOKEN),
            materials=(MISSING_TOKEN, UNKNOWN_TOKEN),
            shapes=(MISSING_TOKEN, UNKNOWN_TOKEN),
            functions=(MISSING_TOKEN, UNKNOWN_TOKEN),
            styles=(MISSING_TOKEN, UNKNOWN_TOKEN),
        )


def adapt_schema_v1_vocab(data: Mapping[str, Any] | None) -> StructuredConditioningVocab | None:
    """Load a v1 vocabulary without changing its IDs.

    This adapter deliberately preserves the legacy conflation at ID 0.  It is a
    loading compatibility path, not a migration to v2.
    """
    if not isinstance(data, Mapping):
        return None
    version = data.get("schema_version")
    if version not in {None, STRUCTURED_VOCAB_SCHEMA_V1}:
        raise ValueError(f"schema-v1 adapter cannot load {version!r}")
    values = {
        field: _tokens_from_json(data.get(field), version=STRUCTURED_VOCAB_SCHEMA_V1)
        for field in (
            "categories",
            "objects",
            "base_objects",
            "colors",
            "materials",
            "shapes",
            "functions",
            "styles",
        )
    }
    return StructuredConditioningVocab(**values, schema_version=STRUCTURED_VOCAB_SCHEMA_V1)


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
        attributes.get("primary_colors")
        or attributes.get("colors")
        or _nested_value(attributes, "color_roles", "primary")
        or record.get("colors")
        or record.get("color")
        or record.get("colour")
    )
    materials = _token_list(
        attributes.get("explicit_material")
        or attributes.get("materials")
        or record.get("explicit_material")
        or record.get("materials")
        or record.get("material")
    )
    shapes = _token_list(
        attributes.get("shapes") or attributes.get("shape") or record.get("shapes") or record.get("shape")
    )
    functions = _token_list(
        attributes.get("role")
        or attributes.get("function")
        or record.get("role")
        or record.get("function")
        or record.get("functions")
    )
    styles = _token_list(attributes.get("style") or record.get("style") or record.get("styles"))
    if not styles:
        styles = _token_list(record.get("caption_type"))
    object_name = _first_token(
        record.get("object_name"),
        record.get("canonical_object"),
        record.get("prompt_object"),
        semantic.get("canonical_object") if isinstance(semantic, Mapping) else None,
        semantic.get("open_name") if isinstance(semantic, Mapping) else None,
    )
    base_object = _first_token(
        record.get("base_object"),
        semantic.get("base_object") if isinstance(semantic, Mapping) else None,
        semantic.get("canonical_object") if isinstance(semantic, Mapping) else None,
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
    statuses = {
        "category": _structured_field_state(record, "category", fields["category"], vocab.categories),
        "object": _structured_field_state(record, "canonical_object", fields["object_name"], vocab.objects),
        "base_object": _structured_field_state(record, "canonical_object", fields["base_object"], vocab.base_objects),
        "primary_color": _structured_field_state(record, "primary_colors", fields["primary_color"], vocab.colors),
        "colors": _structured_field_state(record, "primary_colors", fields["colors"], vocab.colors),
        "materials": _structured_field_state(record, "explicit_material", fields["materials"], vocab.materials),
        "shapes": _structured_field_state(
            record,
            ("silhouette", "aspect", "orientation", "structure", "edge_profile", "parts"),
            fields["shapes"],
            vocab.shapes,
        ),
        "functions": _structured_field_state(record, "role", fields["functions"], vocab.functions),
        "styles": _structured_field_state(record, "style", fields["styles"], vocab.styles),
    }
    category = _conditioned_value(fields["category"], statuses["category"])
    object_name = _conditioned_value(fields["object_name"], statuses["object"])
    base_object = _conditioned_value(fields["base_object"], statuses["base_object"])
    primary_color = _conditioned_value(fields["primary_color"], statuses["primary_color"])
    colors = _conditioned_values(fields["colors"], statuses["colors"])
    materials = _conditioned_values(fields["materials"], statuses["materials"])
    shapes = _conditioned_values(fields["shapes"], statuses["shapes"])
    functions = _conditioned_values(fields["functions"], statuses["functions"])
    styles = _conditioned_values(fields["styles"], statuses["styles"])
    return {
        "category_id": _index(vocab.categories, category),
        "object_id": _index(vocab.objects, object_name),
        "base_object_id": _index(vocab.base_objects, base_object),
        "primary_color_id": _index(vocab.colors, primary_color),
        "color_multi_hot": _multi_hot(vocab.colors, colors),
        "material_multi_hot": _multi_hot(vocab.materials, materials),
        "shape_multi_hot": _multi_hot(vocab.shapes, shapes),
        "function_multi_hot": _multi_hot(vocab.functions, functions),
        "style_multi_hot": _multi_hot(vocab.styles, styles),
        "category_status_id": FIELD_VALUE_STATE_IDS[statuses["category"]],
        "object_status_id": FIELD_VALUE_STATE_IDS[statuses["object"]],
        "base_object_status_id": FIELD_VALUE_STATE_IDS[statuses["base_object"]],
        "primary_color_status_id": FIELD_VALUE_STATE_IDS[statuses["primary_color"]],
        "color_status_id": FIELD_VALUE_STATE_IDS[statuses["colors"]],
        "material_status_id": FIELD_VALUE_STATE_IDS[statuses["materials"]],
        "shape_status_id": FIELD_VALUE_STATE_IDS[statuses["shapes"]],
        "function_status_id": FIELD_VALUE_STATE_IDS[statuses["functions"]],
        "style_status_id": FIELD_VALUE_STATE_IDS[statuses["styles"]],
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


def structured_vocab_from_checkpoint(
    checkpoint: Mapping[str, Any], *, allow_schema_v1_adapter: bool = False
) -> StructuredConditioningVocab | None:
    raw = checkpoint.get("structured_conditioning_vocab")
    if raw is None and isinstance(checkpoint.get("train_config"), Mapping):
        raw = checkpoint["train_config"].get("structured_conditioning_vocab")
    if raw is None:
        return None
    version = raw.get("schema_version") if isinstance(raw, Mapping) else None
    if version in {None, STRUCTURED_VOCAB_SCHEMA_V1}:
        if not allow_schema_v1_adapter:
            raise ValueError(
                "checkpoint uses structured conditioning schema v1; request the explicit schema-v1 adapter"
            )
        return adapt_schema_v1_vocab(raw)
    direct = StructuredConditioningVocab.from_json_dict(raw)
    if direct is not None:
        return direct
    return None


def save_structured_conditioning_vocab(vocab: StructuredConditioningVocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _semantic_v3(record: Mapping[str, Any]) -> Mapping[str, Any]:
    conditioning = record.get("conditioning") if isinstance(record.get("conditioning"), Mapping) else {}
    nested = (
        (conditioning.get("semantic_v4") or conditioning.get("semantic_v3"))
        if isinstance(conditioning, Mapping)
        else {}
    )
    if isinstance(nested, Mapping):
        return nested
    top = record.get("semantic_v4") or record.get("semantics") or record.get("semantic_v3")
    return top if isinstance(top, Mapping) else {}


def _semantic_attributes(record: Mapping[str, Any], semantic: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = semantic.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes
    if any(
        key in semantic
        for key in ("canonical_object", "explicit_material", "shape", "color_roles", "primary_colors", "role")
    ):
        return semantic
    target = record.get("target_semantics") if isinstance(record.get("target_semantics"), Mapping) else {}
    target_attributes = target.get("attributes") if isinstance(target, Mapping) else {}
    return target_attributes if isinstance(target_attributes, Mapping) else {}


def _nested_value(value: Mapping[str, Any], parent: str, child: str) -> Any:
    nested = value.get(parent)
    return nested.get(child) if isinstance(nested, Mapping) else None


def _ordered_vocab(counter: Counter[str]) -> tuple[str, ...]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    reserved = {MISSING_TOKEN, UNKNOWN_TOKEN}
    return (MISSING_TOKEN, UNKNOWN_TOKEN, *(token for token, _count in ordered if token and token not in reserved))


def _tokens_from_json(value: Any, *, version: str) -> tuple[str, ...]:
    tokens = tuple(str(item) for item in value or () if str(item))
    if version == STRUCTURED_VOCAB_SCHEMA_V1:
        if not tokens:
            return (UNKNOWN_TOKEN,)
        if tokens[0] != UNKNOWN_TOKEN or len(set(tokens)) != len(tokens):
            raise ValueError("schema-v1 vocabulary must record <unk> exactly once at ID 0")
        return tokens
    if len(tokens) < 2 or tokens[:2] != (MISSING_TOKEN, UNKNOWN_TOKEN) or len(set(tokens)) != len(tokens):
        raise ValueError("schema-v2 vocabulary must record <missing> at ID 0 and <unk> at ID 1 without duplicates")
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
        return 1 if len(vocab) > 1 and vocab[1] == UNKNOWN_TOKEN else 0


def _multi_hot(vocab: Sequence[str], tokens: Sequence[str]) -> list[float]:
    result = [0.0] * len(vocab)
    if not tokens:
        if result:
            result[0] = 1.0
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
        unknown_index = 1 if len(vocab) > 1 and vocab[1] == UNKNOWN_TOKEN else 0
        result[unknown_index] = 1.0
    return result


def _structured_field_state(
    record: Mapping[str, Any],
    quality_field: str | Sequence[str],
    value: Any,
    vocab: Sequence[str],
) -> str:
    quality = extract_training_quality(record)
    if quality is not None:
        names = (quality_field,) if isinstance(quality_field, str) else tuple(quality_field)
        quality_fields = quality.get("fields", {})
        selected = [quality_fields.get(name, {}) for name in names if isinstance(quality_fields, Mapping)]
        selected = [field for field in selected if isinstance(field, Mapping)]
        if any(str(field.get("value_state")) == "known" and not field.get("conditioning_mask") for field in selected):
            return "abstained"
        states = [str(field.get("value_state") or "missing") for field in selected]
        for state in ("abstained", "out_of_vocabulary", "unknown", "known", "missing"):
            if state in states:
                return state
    tokens = _token_list(value)
    if not tokens:
        return "missing"
    if all(token == "unknown" for token in tokens):
        return "unknown"
    known = set(vocab)
    if any(token not in known for token in tokens):
        return "out_of_vocabulary"
    return "known"


def _conditioned_value(value: str, state: str) -> str:
    if state in {"missing", "unknown", "abstained"}:
        return ""
    return value


def _conditioned_values(values: Sequence[str], state: str) -> list[str]:
    if state in {"missing", "unknown", "abstained"}:
        return []
    return list(values)


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

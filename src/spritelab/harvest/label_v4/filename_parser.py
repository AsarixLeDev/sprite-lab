"""Compositional deterministic semantics for Labeling v4 source names.

This parser recognizes token classes rather than maintaining a table of whole
filenames.  Every input token is retained with its source, classification, and
normalization so a later decision can be reproduced or challenged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Literal

from spritelab.harvest.label_v4.semantic_axes import (
    CATEGORY_VALUES,
    DOMAIN_VALUES,
    ROLE_VALUES,
    MaterialEvidence,
    normalize_semantic_term,
)

FILENAME_PARSER_VERSION = "filename_parser_v4.2"

# Lower numbers are stronger. Sprite and archive-member basenames are one
# evidence group; a per-cell mapping cannot override an explicit identity in
# either filename. Sheet and pack text remain useful context, but are never
# eligible to synthesize a surface alias.
EVIDENCE_SOURCE_PRECEDENCE: tuple[str, ...] = (
    "sprite_filename",
    "member_filename",
    "explicit_cell_mapping",
    "reviewed_variant_metadata",
    "source_record_metadata",
    "sheet_name",
    "pack_name",
)
EVIDENCE_SOURCE_PRIORITY: dict[str, int] = {source: index for index, source in enumerate(EVIDENCE_SOURCE_PRECEDENCE)}

TokenClass = Literal[
    "object",
    "material",
    "color",
    "size",
    "condition",
    "style",
    "orientation",
    "variant",
    "sequence",
    "generic",
    "open_set",
]


@dataclass(frozen=True)
class ObjectSemantics:
    canonical_object: str
    category: str
    role: str


OBJECT_TOKENS: dict[str, ObjectSemantics] = {
    "sword": ObjectSemantics("sword", "weapon", "weapon"),
    "dagger": ObjectSemantics("dagger", "weapon", "weapon"),
    "knife": ObjectSemantics("dagger", "weapon", "weapon"),
    "axe": ObjectSemantics("axe", "weapon", "weapon"),
    "mace": ObjectSemantics("mace", "weapon", "weapon"),
    "hammer": ObjectSemantics("hammer", "weapon", "weapon"),
    "bow": ObjectSemantics("bow", "weapon", "weapon"),
    "arrow": ObjectSemantics("arrow", "weapon", "weapon"),
    "spear": ObjectSemantics("spear", "weapon", "weapon"),
    "buckler": ObjectSemantics("buckler", "armor", "defensive_equipment"),
    "shield": ObjectSemantics("shield", "armor", "defensive_equipment"),
    "helmet": ObjectSemantics("helmet", "armor", "wearable_equipment"),
    "helm": ObjectSemantics("helmet", "armor", "wearable_equipment"),
    "armor": ObjectSemantics("armor", "armor", "wearable_equipment"),
    "armour": ObjectSemantics("armor", "armor", "wearable_equipment"),
    "chestplate": ObjectSemantics("chestplate", "armor", "wearable_equipment"),
    "greaves": ObjectSemantics("greaves", "armor", "wearable_equipment"),
    "boots": ObjectSemantics("boots", "clothing", "wearable_equipment"),
    "cap": ObjectSemantics("cap", "clothing", "wearable_equipment"),
    "pants": ObjectSemantics("pants", "clothing", "wearable_equipment"),
    "trousers": ObjectSemantics("pants", "clothing", "wearable_equipment"),
    "shirt": ObjectSemantics("shirt", "clothing", "wearable_equipment"),
    "jacket": ObjectSemantics("jacket", "clothing", "wearable_equipment"),
    "ring": ObjectSemantics("ring", "jewelry", "wearable_equipment"),
    "amulet": ObjectSemantics("amulet", "jewelry", "wearable_equipment"),
    "necklace": ObjectSemantics("necklace", "jewelry", "wearable_equipment"),
    "key": ObjectSemantics("key", "key", "quest_item"),
    "pickaxe": ObjectSemantics("pickaxe", "tool", "tool"),
    "shovel": ObjectSemantics("shovel", "tool", "tool"),
    "hoe": ObjectSemantics("hoe", "tool", "tool"),
    "scissors": ObjectSemantics("scissors", "tool", "tool"),
    "gem": ObjectSemantics("gem", "gem", "resource"),
    "gemstone": ObjectSemantics("gem", "gem", "resource"),
    "jewel": ObjectSemantics("gem", "gem", "resource"),
    "diamond": ObjectSemantics("diamond", "gem", "resource"),
    "agate": ObjectSemantics("agate", "gem", "resource"),
    "amethyst": ObjectSemantics("amethyst", "gem", "resource"),
    "emerald": ObjectSemantics("emerald", "gem", "resource"),
    "opal": ObjectSemantics("opal", "gem", "resource"),
    "ruby": ObjectSemantics("ruby", "gem", "resource"),
    "sapphire": ObjectSemantics("sapphire", "gem", "resource"),
    "crystal": ObjectSemantics("crystal", "gem", "resource"),
    "potion": ObjectSemantics("potion", "potion", "consumable"),
    "bottle": ObjectSemantics("bottle", "container", "container"),
    "chest": ObjectSemantics("chest", "container", "container"),
    "apple": ObjectSemantics("apple", "food", "consumable"),
    "bread": ObjectSemantics("bread", "food", "consumable"),
    "eggplant": ObjectSemantics("eggplant", "food", "consumable"),
    "flower": ObjectSemantics("flower", "plant", "resource"),
    "mushroom": ObjectSemantics("mushroom", "plant", "resource"),
    "scroll": ObjectSemantics("scroll", "spell", "consumable"),
}

COMPOUND_OBJECTS: dict[tuple[str, ...], ObjectSemantics] = {
    ("crystal", "cluster"): ObjectSemantics("crystal_cluster", "gem", "resource"),
    ("mace", "head"): ObjectSemantics("mace_head", "material", "crafting_material"),
}

MATERIAL_TOKENS: dict[str, str] = {
    "iron": "iron",
    "steel": "steel",
    "copper": "copper",
    "bronze": "bronze",
    "silver": "silver",
    "gold": "gold",
    "golden": "gold",
    "platemail": "plate_metal",
    "plate": "plate_metal",
    "chainmail": "chainmail",
    "chain": "chainmail",
    "leather": "leather",
    "cloth": "cloth",
    "wood": "wood",
    "wooden": "wood",
    "stone": "stone",
    "crystal": "crystal",
    "glass": "glass",
    "bone": "bone",
    "paper": "paper",
}

MATERIAL_COMPOUNDS: dict[tuple[str, ...], str] = {
    ("plate", "mail"): "plate_metal",
    ("chain", "mail"): "chainmail",
}

COLOR_TOKENS: dict[str, str] = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "teal": "teal",
    "cyan": "cyan",
    "blue": "blue",
    "purple": "purple",
    "violet": "purple",
    "pink": "pink",
    "brown": "brown",
    "tan": "tan",
    "white": "white",
    "grey": "gray",
    "gray": "gray",
    "black": "black",
}

SIZE_TOKENS = {"tiny", "small", "medium", "large", "huge", "short", "long", "wide", "narrow"}
CONDITION_TOKENS = {
    "ancient",
    "broken",
    "chipped",
    "cracked",
    "damaged",
    "polished",
    "quilted",
    "raw",
    "rusted",
    "rusty",
    "tattered",
    "worn",
}
STYLE_TOKEN_MAP = {
    "pixelart": "pixel_art",
    "outlined": "outlined",
    "isometric": "isometric",
    "ornate": "ornate",
    "minimal": "minimal",
    "fantasy": "fantasy",
}
ORIENTATION_TOKEN_MAP = {
    "front": "front_facing",
    "frontfacing": "front_facing",
    "side": "side_facing",
    "left": "left_facing",
    "right": "right_facing",
    "topdown": "top_down",
    "horizontal": "horizontal",
    "vertical": "vertical",
    "diagonal": "diagonal",
}
VARIANT_TOKENS = {"alt", "alternate", "copy", "recolor", "variant"}
GENERIC_TOKENS = {
    "a",
    "asset",
    "cell",
    "frame",
    "icon",
    "image",
    "img",
    "item",
    "object",
    "sheet",
    "sprite",
    "tile",
    "32x32",
    "cc0",
    "icons",
    "png",
}


@dataclass(frozen=True)
class TokenProvenance:
    source: str
    source_text: str
    index: int
    raw_token: str
    normalized_token: str
    classification: TokenClass
    value: Any
    transformation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_text": self.source_text,
            "index": self.index,
            "raw_token": self.raw_token,
            "normalized_token": self.normalized_token,
            "classification": self.classification,
            "value": self.value,
            "transformation": self.transformation,
        }


@dataclass(frozen=True)
class FilenameParseResult:
    schema_version: str = FILENAME_PARSER_VERSION
    domain: str = "unknown"
    canonical_object: str | None = None
    surface_alias: str | None = None
    category: str = "unknown"
    role: str = "unknown"
    explicit_material: str | None = None
    explicit_material_candidates: tuple[str, ...] = ()
    filename_color_hints: tuple[str, ...] = ()
    size_hint: str | None = None
    size_hints: tuple[str, ...] = ()
    condition_hints: tuple[str, ...] = ()
    style_modifiers: tuple[str, ...] = ()
    orientation_hints: tuple[str, ...] = ()
    variant_suffixes: tuple[str, ...] = ()
    sequence_numbers: tuple[str, ...] = ()
    open_set_tokens: tuple[str, ...] = ()
    generic: bool = True
    object_source: str = ""
    surface_alias_source: str = ""
    field_sources: dict[str, str] = field(default_factory=dict)
    token_provenance: tuple[TokenProvenance, ...] = ()
    transformations: tuple[str, ...] = ()
    source_values: dict[str, Any] = field(default_factory=dict)

    @property
    def material(self) -> MaterialEvidence:
        support = tuple(sorted({token.source for token in self.token_provenance if token.classification == "material"}))
        return MaterialEvidence(explicit_material=self.explicit_material, explicit_support=support)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "canonical_object": self.canonical_object,
            "surface_alias": self.surface_alias,
            "category": self.category,
            "role": self.role,
            "explicit_material": self.explicit_material,
            "explicit_material_candidates": list(self.explicit_material_candidates),
            "filename_color_hints": list(self.filename_color_hints),
            "size_hint": self.size_hint,
            "size_hints": list(self.size_hints),
            "condition_hints": list(self.condition_hints),
            "style_modifiers": list(self.style_modifiers),
            "orientation_hints": list(self.orientation_hints),
            "variant_suffixes": list(self.variant_suffixes),
            "sequence_numbers": list(self.sequence_numbers),
            "open_set_tokens": list(self.open_set_tokens),
            "generic": self.generic,
            "object_source": self.object_source,
            "surface_alias_source": self.surface_alias_source,
            "field_sources": dict(self.field_sources),
            "token_provenance": [token.to_dict() for token in self.token_provenance],
            "transformations": list(self.transformations),
            "source_values": dict(self.source_values),
        }


@dataclass(frozen=True)
class _Lexeme:
    raw: str
    normalized: str
    transformation: str


def _dedupe(values: Sequence[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _lexemes(text: Any, *, path_like: bool) -> tuple[_Lexeme, ...]:
    raw_text = str(text or "").strip().lower().replace("\\", "/")
    if not raw_text:
        return ()
    if path_like:
        # Identity parsing uses the basename only. Directory names are source,
        # sheet, or pack context and must never leak into a sprite alias.
        path = PurePath(raw_text)
        raw_text = path.stem if path.parts else raw_text
    pieces = [piece for piece in re.split(r"[^a-z0-9]+", raw_text) if piece]
    result: list[_Lexeme] = []
    for piece in pieces:
        if piece in GENERIC_TOKENS:
            result.append(_Lexeme(piece, piece, "identity"))
            continue
        if re.fullmatch(r"[rc]\d+", piece):
            result.append(_Lexeme(piece, piece, "preserve_sheet_coordinate"))
            continue
        if re.fullmatch(r"v\d+", piece):
            result.append(_Lexeme(piece, piece, "preserve_variant_version"))
            continue
        match = re.fullmatch(r"([a-z]+)(\d+)", piece)
        if match:
            result.append(_Lexeme(piece, match.group(1), "split_alpha_numeric:alpha"))
            result.append(_Lexeme(piece, match.group(2), "split_alpha_numeric:sequence"))
            continue
        result.append(_Lexeme(piece, piece, "identity"))
    return tuple(result)


def _singular(token: str) -> str:
    if token.endswith("ies") and token[:-3] + "y" in OBJECT_TOKENS:
        return token[:-3] + "y"
    if token.endswith("s") and token[:-1] in OBJECT_TOKENS:
        return token[:-1]
    if token == "clusters":
        return "cluster"
    return token


def _event(
    source: str,
    source_text: str,
    index: int,
    lexeme: _Lexeme,
    classification: TokenClass,
    value: Any,
    transformation: str,
) -> TokenProvenance:
    steps = [lexeme.transformation] if lexeme.transformation != "identity" else []
    if transformation and transformation != "identity":
        steps.append(transformation)
    return TokenProvenance(
        source=source,
        source_text=source_text,
        index=index,
        raw_token=lexeme.raw,
        normalized_token=lexeme.normalized,
        classification=classification,
        value=value,
        transformation=";".join(steps) or "identity",
    )


def _classify(source: str, source_text: str, lexemes: tuple[_Lexeme, ...]) -> list[TokenProvenance]:
    events: list[TokenProvenance] = []
    consumed: set[int] = set()

    normalized = [_singular(lexeme.normalized) for lexeme in lexemes]
    for compound, semantics in COMPOUND_OBJECTS.items():
        size = len(compound)
        for start in range(len(normalized) - size + 1):
            if tuple(normalized[start : start + size]) != compound or any(
                i in consumed for i in range(start, start + size)
            ):
                continue
            for index in range(start, start + size):
                events.append(
                    _event(
                        source,
                        source_text,
                        index,
                        lexemes[index],
                        "object",
                        semantics.canonical_object,
                        f"compound:{'+'.join(compound)}->canonical_object:{semantics.canonical_object}",
                    )
                )
                consumed.add(index)

    for compound, material in MATERIAL_COMPOUNDS.items():
        size = len(compound)
        for start in range(len(normalized) - size + 1):
            if tuple(normalized[start : start + size]) != compound:
                continue
            for index in range(start, start + size):
                if index in consumed:
                    continue
                events.append(
                    _event(
                        source,
                        source_text,
                        index,
                        lexemes[index],
                        "material",
                        material,
                        f"compound:{'+'.join(compound)}->explicit_material:{material}",
                    )
                )
                consumed.add(index)

    # Two-word style and orientation phrases are compositional too.
    phrase_maps: tuple[tuple[tuple[str, str], TokenClass, str], ...] = (
        (("pixel", "art"), "style", "pixel_art"),
        (("top", "down"), "orientation", "top_down"),
        (("front", "facing"), "orientation", "front_facing"),
        (("side", "facing"), "orientation", "side_facing"),
    )
    for phrase, classification, value in phrase_maps:
        for start in range(len(normalized) - 1):
            if tuple(normalized[start : start + 2]) != phrase or any(i in consumed for i in (start, start + 1)):
                continue
            for index in (start, start + 1):
                events.append(
                    _event(
                        source,
                        source_text,
                        index,
                        lexemes[index],
                        classification,
                        value,
                        f"compound:{'+'.join(phrase)}->{classification}:{value}",
                    )
                )
                consumed.add(index)

    for index, lexeme in enumerate(lexemes):
        if index in consumed:
            continue
        token = _singular(lexeme.normalized)
        singular_step = f"singularize:{lexeme.normalized}->{token}" if token != lexeme.normalized else ""
        if token in OBJECT_TOKENS:
            semantics = OBJECT_TOKENS[token]
            events.append(
                _event(
                    source,
                    source_text,
                    index,
                    lexeme,
                    "object",
                    semantics.canonical_object,
                    singular_step or f"{token}->canonical_object:{semantics.canonical_object}",
                )
            )
        elif token in MATERIAL_TOKENS:
            material = MATERIAL_TOKENS[token]
            events.append(
                _event(
                    source, source_text, index, lexeme, "material", material, f"{token}->explicit_material:{material}"
                )
            )
        elif token in COLOR_TOKENS:
            color = COLOR_TOKENS[token]
            events.append(_event(source, source_text, index, lexeme, "color", color, f"{token}->color:{color}"))
        elif token in SIZE_TOKENS:
            events.append(_event(source, source_text, index, lexeme, "size", token, "identity"))
        elif token in CONDITION_TOKENS:
            condition = "rusty" if token == "rusted" else token
            events.append(_event(source, source_text, index, lexeme, "condition", condition, f"{token}->{condition}"))
        elif token in STYLE_TOKEN_MAP:
            style = STYLE_TOKEN_MAP[token]
            events.append(_event(source, source_text, index, lexeme, "style", style, f"{token}->{style}"))
        elif token in ORIENTATION_TOKEN_MAP:
            orientation = ORIENTATION_TOKEN_MAP[token]
            events.append(
                _event(source, source_text, index, lexeme, "orientation", orientation, f"{token}->{orientation}")
            )
        elif token in VARIANT_TOKENS or re.fullmatch(r"v\d+", token):
            events.append(_event(source, source_text, index, lexeme, "variant", token, "identity"))
        elif token.isdigit() or re.fullmatch(r"[rc]\d+", token):
            events.append(_event(source, source_text, index, lexeme, "sequence", token, "identity"))
        elif token in GENERIC_TOKENS or len(token) <= 1:
            events.append(_event(source, source_text, index, lexeme, "generic", token, "identity"))
        else:
            events.append(_event(source, source_text, index, lexeme, "open_set", token, "preserve_open_set_token"))
    return sorted(events, key=lambda event: (event.index, event.classification, str(event.value)))


def _mapping_value(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = data.get(name)
        if value not in (None, "", [], ()):
            return value
    return None


def _controlled_mapping_value(data: Mapping[str, Any], vocabulary: Sequence[str], *names: str) -> str:
    value = normalize_semantic_term(_mapping_value(data, *names))
    return value if value in vocabulary else ""


def _source_events(events: Sequence[TokenProvenance], source: str) -> list[TokenProvenance]:
    return [event for event in events if event.source == source]


def _alias_from_events(events: Sequence[TokenProvenance]) -> str | None:
    # Modifiers alone (for example ``small_purple``) are not an object alias.
    # An explicit object or open-set identity token must be present.
    if not any(event.classification in {"object", "open_set"} for event in events):
        return None
    tokens: list[str] = []
    for event in events:
        if event.classification in {"generic", "sequence", "variant", "style", "orientation"}:
            continue
        token = normalize_semantic_term(event.normalized_token).replace("_", " ")
        if token and token not in tokens:
            tokens.append(token)
    return " ".join(tokens).strip() or None


def _is_sheet_cell_mapping(mapping: Mapping[str, Any]) -> bool:
    return any(
        _mapping_value(mapping, name) not in (None, "", [], ())
        for name in ("sheet_coordinate", "cell_coordinate", "cell_index", "row", "column")
    )


def _ranked_first(candidates: Sequence[tuple[int, int, str, str]]) -> tuple[str, str]:
    if not candidates:
        return "", ""
    _priority, _index, value, source = sorted(candidates)[0]
    return value, source


def _context_axis_candidates(
    events: Sequence[TokenProvenance],
    *,
    axis: str,
    priority: int,
    source: str,
) -> list[tuple[int, int, str, str]]:
    result: list[tuple[int, int, str, str]] = []
    vocabulary = CATEGORY_VALUES if axis == "category" else ROLE_VALUES
    for event in events:
        value = ""
        if event.classification == "object":
            semantics = OBJECT_TOKENS.get(str(event.value))
            if semantics is None:
                semantics = next(
                    (item for item in COMPOUND_OBJECTS.values() if item.canonical_object == str(event.value)),
                    None,
                )
            if semantics is not None:
                value = semantics.category if axis == "category" else semantics.role
        token = normalize_semantic_term(event.normalized_token)
        singular = token[:-1] if token.endswith("s") and token[:-1] in vocabulary else token
        if not value and singular in vocabulary:
            value = singular
        if value in vocabulary:
            result.append((priority, event.index, value, source))
    return result


def parse_filename_semantics(
    filename: str | Mapping[str, Any],
    *,
    member_path: str = "",
    sheet_name: str = "",
    pack_name: str = "",
    declarative_mapping: Mapping[str, Any] | None = None,
    reviewed_variant_metadata: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    pack_context: Mapping[str, Any] | None = None,
) -> FilenameParseResult:
    """Parse all source-facing semantic text while preserving its lineage.

    ``filename`` may be a record mapping for convenient integration with
    harvest manifests. Scheduler broad-type metadata is intentionally ignored.
    """

    record: Mapping[str, Any] = filename if isinstance(filename, Mapping) else {}
    if record:
        filename_text = str(_mapping_value(record, "relative_path", "filename", "final_png_path") or "")
        member_path = member_path or str(_mapping_value(record, "archive_member", "member_path") or "")
        sheet_name = sheet_name or str(_mapping_value(record, "source_sheet", "sheet_name") or "")
        pack_name = pack_name or str(_mapping_value(record, "pack_name", "source_name") or "")
        auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
        declarative_mapping = declarative_mapping or (
            auto.get("sheet_mapping") if isinstance(auto.get("sheet_mapping"), Mapping) else {}
        )
        reviewed_variant_metadata = reviewed_variant_metadata or next(
            (
                value
                for value in (
                    record.get("reviewed_variant_metadata"),
                    record.get("reviewed_family_mapping"),
                    auto.get("reviewed_variant_metadata"),
                )
                if isinstance(value, Mapping)
            ),
            {},
        )
        # Existing ``tags`` may be legacy model output. They remain in the
        # source record but are never upgraded into fresh deterministic facts.
        if source_metadata is None:
            embedded = record.get("source_metadata") if isinstance(record.get("source_metadata"), Mapping) else {}
            source_metadata = {
                **dict(embedded),
                **{
                    key: record[key]
                    for key in (
                        "declared_canonical_object",
                        "declared_surface_alias",
                        "declared_material",
                        "declared_category",
                        "declared_domain",
                        "declared_role",
                    )
                    if record.get(key) not in (None, "", [], ())
                },
            }
    else:
        filename_text = str(filename)

    declarative = dict(declarative_mapping or {})
    reviewed = dict(reviewed_variant_metadata or {})
    source_meta = dict(source_metadata or {})
    context = dict(pack_context or {})
    sheet_cell = _is_sheet_cell_mapping(declarative)

    source_values: dict[str, Any] = {
        "filename": filename_text,
        "member_path": member_path,
        "sheet_name": sheet_name,
        "pack_name": pack_name,
        "declarative_mapping": declarative,
        "reviewed_variant_metadata": reviewed,
        "source_metadata": source_meta,
        "pack_context": context,
        "sheet_cell_record": sheet_cell,
    }

    texts: list[tuple[str, str, bool]] = [
        ("filename", filename_text, True),
        ("member_path", member_path, True),
        ("sheet_name", sheet_name, True),
        ("pack_name", pack_name, False),
    ]
    for key in sorted(source_meta):
        value = source_meta[key]
        if isinstance(value, (str, int, float)):
            texts.append((f"source_metadata.{key}", str(value), False))
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            texts.append((f"source_metadata.{key}", " ".join(str(v) for v in value), False))
    for key in sorted(declarative):
        value = declarative[key]
        if isinstance(value, (str, int, float)):
            texts.append((f"declarative_mapping.{key}", str(value), False))
    for key in sorted(reviewed):
        value = reviewed[key]
        if isinstance(value, (str, int, float)):
            texts.append((f"reviewed_variant_metadata.{key}", str(value), False))

    events: list[TokenProvenance] = []
    for source, text, path_like in texts:
        events.extend(_classify(source, text, _lexemes(text, path_like=path_like)))

    transformations: list[str] = [
        f"{event.source}[{event.index}]:{event.transformation}"
        for event in events
        if event.transformation != "identity"
    ]

    filename_events = _source_events(events, "filename")
    member_events = _source_events(events, "member_path")
    sheet_events = _source_events(events, "sheet_name")
    pack_events = _source_events(events, "pack_name")

    # Structured evidence is read directly. Tokenized copies remain in the
    # provenance ledger but cannot accidentally override the declared field.
    mapped_object = normalize_semantic_term(_mapping_value(declarative, "canonical_object", "object_name"))
    reviewed_object = normalize_semantic_term(_mapping_value(reviewed, "canonical_object", "object_name"))
    source_object = normalize_semantic_term(
        _mapping_value(source_meta, "declared_canonical_object", "canonical_object")
    )
    mapped_domain = _controlled_mapping_value(declarative, DOMAIN_VALUES, "domain")
    reviewed_domain = _controlled_mapping_value(reviewed, DOMAIN_VALUES, "domain")
    source_domain = _controlled_mapping_value(source_meta, DOMAIN_VALUES, "declared_domain", "domain")
    contextual_domain = _controlled_mapping_value(context, DOMAIN_VALUES, "domain")
    mapped_category = _controlled_mapping_value(declarative, CATEGORY_VALUES, "category")
    reviewed_category = _controlled_mapping_value(reviewed, CATEGORY_VALUES, "category")
    source_category = _controlled_mapping_value(source_meta, CATEGORY_VALUES, "declared_category", "category")
    contextual_category = _controlled_mapping_value(context, CATEGORY_VALUES, "category")
    mapped_role = _controlled_mapping_value(declarative, ROLE_VALUES, "role")
    reviewed_role = _controlled_mapping_value(reviewed, ROLE_VALUES, "role")
    source_role = _controlled_mapping_value(source_meta, ROLE_VALUES, "declared_role", "role")
    contextual_role = _controlled_mapping_value(context, ROLE_VALUES, "role")
    mapped_material = normalize_semantic_term(_mapping_value(declarative, "explicit_material", "material"))
    reviewed_material = normalize_semantic_term(_mapping_value(reviewed, "explicit_material", "material"))
    source_material = normalize_semantic_term(
        _mapping_value(source_meta, "declared_material", "explicit_material", "material")
    )
    contextual_material = normalize_semantic_term(_mapping_value(context, "explicit_material", "material"))

    object_candidates: list[tuple[int, int, str, str]] = []
    if not sheet_cell:
        object_candidates.extend(
            (
                EVIDENCE_SOURCE_PRIORITY["sprite_filename"],
                event.index,
                str(event.value),
                "sprite_filename",
            )
            for event in filename_events
            if event.classification == "object"
        )
        object_candidates.extend(
            (
                EVIDENCE_SOURCE_PRIORITY["member_filename"],
                event.index,
                str(event.value),
                "member_filename",
            )
            for event in member_events
            if event.classification == "object"
        )
    if mapped_object:
        object_candidates.append(
            (EVIDENCE_SOURCE_PRIORITY["explicit_cell_mapping"], -1, mapped_object, "explicit_cell_mapping")
        )
    if reviewed_object:
        object_candidates.append(
            (
                EVIDENCE_SOURCE_PRIORITY["reviewed_variant_metadata"],
                -1,
                reviewed_object,
                "reviewed_variant_metadata",
            )
        )
    if source_object:
        object_candidates.append(
            (
                EVIDENCE_SOURCE_PRIORITY["source_record_metadata"],
                -1,
                source_object,
                "source_record_metadata",
            )
        )

    canonical_value, object_source = _ranked_first(object_candidates)
    canonical_object = canonical_value or None

    semantics = OBJECT_TOKENS.get(canonical_object or "")
    if semantics is None and canonical_object:
        semantics = next(
            (item for item in COMPOUND_OBJECTS.values() if item.canonical_object == canonical_object),
            None,
        )
    category_candidates: list[tuple[int, int, str, str]] = []
    role_candidates: list[tuple[int, int, str, str]] = []
    if semantics is not None and object_source:
        priority = EVIDENCE_SOURCE_PRIORITY[object_source]
        category_candidates.append((priority, -1, semantics.category, object_source))
        role_candidates.append((priority, -1, semantics.role, object_source))
    for priority_name, category_value, role_value in (
        ("explicit_cell_mapping", mapped_category, mapped_role),
        ("reviewed_variant_metadata", reviewed_category, reviewed_role),
        ("source_record_metadata", source_category, source_role),
    ):
        priority = EVIDENCE_SOURCE_PRIORITY[priority_name]
        if category_value:
            category_candidates.append((priority, -1, category_value, priority_name))
        if role_value:
            role_candidates.append((priority, -1, role_value, priority_name))
    category_candidates.extend(
        _context_axis_candidates(
            sheet_events,
            axis="category",
            priority=EVIDENCE_SOURCE_PRIORITY["sheet_name"],
            source="sheet_name",
        )
    )
    role_candidates.extend(
        _context_axis_candidates(
            sheet_events,
            axis="role",
            priority=EVIDENCE_SOURCE_PRIORITY["sheet_name"],
            source="sheet_name",
        )
    )
    category_candidates.extend(
        _context_axis_candidates(
            pack_events,
            axis="category",
            priority=EVIDENCE_SOURCE_PRIORITY["pack_name"],
            source="pack_name",
        )
    )
    role_candidates.extend(
        _context_axis_candidates(
            pack_events,
            axis="role",
            priority=EVIDENCE_SOURCE_PRIORITY["pack_name"],
            source="pack_name",
        )
    )
    if contextual_category:
        category_candidates.append((EVIDENCE_SOURCE_PRIORITY["pack_name"], 10_000, contextual_category, "pack_context"))
    if contextual_role:
        role_candidates.append((EVIDENCE_SOURCE_PRIORITY["pack_name"], 10_000, contextual_role, "pack_context"))
    category, category_source = _ranked_first(category_candidates)
    role, role_source = _ranked_first(role_candidates)

    # Construction can refine the broad policy without changing object
    # identity. A chainmail jacket is armor under the v4 default policy.
    filename_materials = [str(event.value) for event in filename_events if event.classification == "material"]
    if canonical_object == "jacket" and "chainmail" in filename_materials:
        category = "armor"
        role = "wearable_equipment"
        transformations.append("policy:chainmail+jacket->category:armor")

    material_candidates: list[tuple[int, int, str, str]] = []
    filename_material_priority = (
        EVIDENCE_SOURCE_PRIORITY["sheet_name"] if sheet_cell else EVIDENCE_SOURCE_PRIORITY["sprite_filename"]
    )
    member_material_priority = (
        EVIDENCE_SOURCE_PRIORITY["sheet_name"] if sheet_cell else EVIDENCE_SOURCE_PRIORITY["member_filename"]
    )
    material_candidates.extend(
        (filename_material_priority, event.index, str(event.value), "sheet_name" if sheet_cell else "sprite_filename")
        for event in filename_events
        if event.classification == "material"
    )
    material_candidates.extend(
        (member_material_priority, event.index, str(event.value), "sheet_name" if sheet_cell else "member_filename")
        for event in member_events
        if event.classification == "material"
    )
    for source_name, value in (
        ("explicit_cell_mapping", mapped_material),
        ("reviewed_variant_metadata", reviewed_material),
        ("source_record_metadata", source_material),
    ):
        if value:
            material_candidates.append((EVIDENCE_SOURCE_PRIORITY[source_name], -1, value, source_name))
    material_candidates.extend(
        (EVIDENCE_SOURCE_PRIORITY["sheet_name"], event.index, str(event.value), "sheet_name")
        for event in sheet_events
        if event.classification == "material"
    )
    material_candidates.extend(
        (EVIDENCE_SOURCE_PRIORITY["pack_name"], event.index, str(event.value), "pack_name")
        for event in pack_events
        if event.classification == "material"
    )
    if contextual_material:
        material_candidates.append((EVIDENCE_SOURCE_PRIORITY["pack_name"], 10_000, contextual_material, "pack_context"))
    material_candidates.sort()
    materials = list(_dedupe([value for _priority, _index, value, _source in material_candidates if value]))
    explicit_material, material_source = _ranked_first(material_candidates)

    identity_events = [] if sheet_cell else [*filename_events, *member_events]
    colors = _dedupe([str(event.value) for event in identity_events if event.classification == "color"])
    sizes = _dedupe([str(event.value) for event in identity_events if event.classification == "size"])
    conditions = _dedupe([str(event.value) for event in identity_events if event.classification == "condition"])
    styles = _dedupe([str(event.value) for event in events if event.classification == "style"])
    orientations = _dedupe([str(event.value) for event in events if event.classification == "orientation"])
    variants = _dedupe([str(event.value) for event in identity_events if event.classification == "variant"])
    sequences = _dedupe([str(event.value) for event in identity_events if event.classification == "sequence"])
    open_terms = _dedupe([str(event.value) for event in identity_events if event.classification == "open_set"])

    useful = {
        "object",
        "material",
        "color",
        "size",
        "condition",
        "style",
        "orientation",
        "open_set",
    }
    generic = not any(event.classification in useful for event in identity_events)

    alias_candidates: list[tuple[int, int, str, str]] = []
    if not sheet_cell:
        filename_alias = _alias_from_events(filename_events)
        member_alias = _alias_from_events(member_events)
        if filename_alias:
            alias_candidates.append(
                (EVIDENCE_SOURCE_PRIORITY["sprite_filename"], -1, filename_alias, "sprite_filename")
            )
        if member_alias:
            alias_candidates.append((EVIDENCE_SOURCE_PRIORITY["member_filename"], -1, member_alias, "member_filename"))
    for source_name, mapping_value, object_value in (
        ("explicit_cell_mapping", declarative, mapped_object),
        ("reviewed_variant_metadata", reviewed, reviewed_object),
        ("source_record_metadata", source_meta, source_object),
    ):
        explicit_alias = str(
            _mapping_value(mapping_value, "surface_alias", "declared_surface_alias", "alias") or ""
        ).strip()
        if not explicit_alias and object_value:
            explicit_alias = object_value.replace("_", " ")
        if explicit_alias:
            alias_candidates.append((EVIDENCE_SOURCE_PRIORITY[source_name], -1, explicit_alias, source_name))
    alias_value, surface_alias_source = _ranked_first(alias_candidates)
    surface_alias = alias_value or None

    if canonical_object:
        transformations.append(f"{object_source}->{canonical_object}:canonical_object")
    if surface_alias:
        transformations.append(f"{surface_alias_source}->{surface_alias}:surface_alias")

    domain_candidates: list[tuple[int, int, str, str]] = []
    for source_name, value in (
        ("explicit_cell_mapping", mapped_domain),
        ("reviewed_variant_metadata", reviewed_domain),
        ("source_record_metadata", source_domain),
    ):
        if value:
            domain_candidates.append((EVIDENCE_SOURCE_PRIORITY[source_name], -1, value, source_name))
    if contextual_domain:
        domain_candidates.append((EVIDENCE_SOURCE_PRIORITY["pack_name"], 10_000, contextual_domain, "pack_context"))
    domain, domain_source = _ranked_first(domain_candidates)

    field_sources = {
        key: value
        for key, value in {
            "canonical_object": object_source,
            "surface_alias": surface_alias_source,
            "category": category_source,
            "domain": domain_source,
            "role": role_source,
            "explicit_material": material_source,
            "filename_color_hints": "sprite_or_member_filename" if colors else "",
            "size_hint": "sprite_or_member_filename" if sizes else "",
        }.items()
        if value
    }

    return FilenameParseResult(
        domain=domain or "unknown",
        canonical_object=canonical_object,
        surface_alias=surface_alias,
        category=category or "unknown",
        role=role or "unknown",
        explicit_material=explicit_material or None,
        explicit_material_candidates=tuple(materials),
        filename_color_hints=tuple(colors),
        size_hint=sizes[0] if sizes else None,
        size_hints=tuple(sizes),
        condition_hints=tuple(conditions),
        style_modifiers=tuple(styles),
        orientation_hints=tuple(orientations),
        variant_suffixes=tuple(variants),
        sequence_numbers=tuple(sequences),
        open_set_tokens=tuple(open_terms),
        generic=generic,
        object_source=object_source,
        surface_alias_source=surface_alias_source,
        field_sources=field_sources,
        token_provenance=tuple(events),
        transformations=tuple(_dedupe(transformations)),
        source_values=source_values,
    )


# Short aliases for interactive callers and compatibility with likely v4 CLI
# naming. They intentionally share the exact same compositional implementation.
parse_filename = parse_filename_semantics
parse_compositional_filename = parse_filename_semantics

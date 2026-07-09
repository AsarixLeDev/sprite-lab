"""Source/profile detection for harvest label v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from spritelab.harvest.label_taxonomy import normalize_tag


@dataclass(frozen=True)
class SourceProfile:
    name: str
    domain: str
    filename_trust: Literal["exact", "prefix_family", "none"]
    expected_category_bias: tuple[str, ...]
    known_path_tokens: tuple[str, ...]
    notes: str = ""
    sheet_specialization: str | None = None
    fusion_threshold_override: float | None = None

    @property
    def trusted_filename(self) -> bool:
        """Backward-compatible exact filename trust accessor."""

        return self.filename_trust == "exact"


_PROFILES: dict[str, SourceProfile] = {
    "cc0_food": SourceProfile(
        name="cc0_food",
        domain="food",
        filename_trust="exact",
        expected_category_bias=("item_icon",),
        known_path_tokens=("food", "ocal", "cc0"),
        notes="Clean OCAL food object filenames.",
    ),
    "cc0_tool": SourceProfile(
        name="cc0_tool",
        domain="tool",
        filename_trust="exact",
        expected_category_bias=("tool",),
        known_path_tokens=("tool", "ocal", "cc0"),
        notes="Clean OCAL tool object filenames.",
    ),
    "cc0_gem": SourceProfile(
        name="cc0_gem",
        domain="gem",
        filename_trust="exact",
        expected_category_bias=("material",),
        known_path_tokens=("gem", "crystal", "7soul1", "cc0"),
        notes="Clean gem/material filenames.",
    ),
    "cc0_jewelry": SourceProfile(
        name="cc0_jewelry",
        domain="jewelry",
        filename_trust="exact",
        expected_category_bias=("item_icon",),
        known_path_tokens=("jewelry", "accessory", "cc0"),
        notes="Clean jewelry/accessory filenames.",
    ),
    "cc0_key": SourceProfile(
        name="cc0_key",
        domain="key",
        filename_trust="exact",
        expected_category_bias=("item_icon",),
        known_path_tokens=("key", "keys", "cc0"),
        notes="Clean key object filenames.",
    ),
    "cc0_potion": SourceProfile(
        name="cc0_potion",
        domain="potion",
        filename_trust="exact",
        expected_category_bias=("item_icon",),
        known_path_tokens=("potion", "vial", "bottle", "cc0"),
        notes="Potion/container icon filenames.",
    ),
    "mushroom": SourceProfile(
        name="mushroom",
        domain="plant",
        filename_trust="exact",
        expected_category_bias=("plant",),
        known_path_tokens=("mushroom", "fungus"),
        notes="Mushroom-focused source pack.",
    ),
    "kenney_tiny_dungeon": SourceProfile(
        name="kenney_tiny_dungeon",
        domain="tileset",
        filename_trust="none",
        expected_category_bias=("block", "environment_prop", "item_icon"),
        known_path_tokens=("kenney", "tiny", "dungeon"),
        notes="Sheet/tile names are often generic.",
    ),
    "kenney_micro_roguelike": SourceProfile(
        name="kenney_micro_roguelike",
        domain="tileset",
        filename_trust="none",
        expected_category_bias=("block", "environment_prop", "item_icon"),
        known_path_tokens=("kenney", "micro", "roguelike"),
        notes="Many assets are generic sheet tiles.",
    ),
    "oga_496_rpg_icons": SourceProfile(
        name="oga_496_rpg_icons",
        domain="rpg_icons",
        filename_trust="prefix_family",
        expected_category_bias=("item_icon", "weapon", "armor", "material", "effect_icon"),
        known_path_tokens=("oga", "496", "rpg", "icons", "32fix"),
        notes="Structured RPG icon prefixes such as W_, I_C_, S_.",
        sheet_specialization="rpg_496",
        fusion_threshold_override=0.65,
    ),
    "generic_unknown": SourceProfile(
        name="generic_unknown",
        domain="unknown",
        filename_trust="none",
        expected_category_bias=("unknown",),
        known_path_tokens=(),
        notes="No source-specific filename guarantees.",
    ),
}


def is_exact_filename_trusted(profile: SourceProfile) -> bool:
    return profile.filename_trust == "exact"


def is_prefix_family_trusted(profile: SourceProfile) -> bool:
    return profile.filename_trust == "prefix_family"


def detect_source_profile(record: Mapping[str, Any]) -> SourceProfile:
    """Infer a source profile from source id/name/path fields."""

    haystack = " ".join(
        normalize_tag(str(record.get(key, "")))
        for key in ("source_id", "source_name", "relative_path", "final_png_path")
    )
    tokens = {token for token in haystack.replace("\\", "_").replace("/", "_").split("_") if token}

    source_id = normalize_tag(str(record.get("source_id", "")))
    source_name = normalize_tag(str(record.get("source_name", "")))
    text = f"{source_id}_{source_name}_{haystack}"

    if "oga_cc0_food" in text or {"cc0", "food"} <= tokens:
        return _PROFILES["cc0_food"]
    if "oga_cc0_tool" in text or {"cc0", "tool"} <= tokens:
        return _PROFILES["cc0_tool"]
    if "oga_cc0_gem" in text or {"cc0", "gem"} <= tokens:
        return _PROFILES["cc0_gem"]
    if "jewelry" in tokens or "jewellery" in tokens:
        return _PROFILES["cc0_jewelry"]
    if "oga_cc0_key" in text or ({"cc0", "key"} <= tokens):
        return _PROFILES["cc0_key"]
    if "kenney_tiny_dungeon" in text or {"kenney", "tiny", "dungeon"} <= tokens:
        return _PROFILES["kenney_tiny_dungeon"]
    if "kenney_micro_roguelike" in text or {"kenney", "micro", "roguelike"} <= tokens:
        return _PROFILES["kenney_micro_roguelike"]
    if "oga_496_rpg_icons" in text or {"496", "rpg", "icons"} <= tokens:
        return _PROFILES["oga_496_rpg_icons"]
    if "potion" in tokens or "vial" in tokens:
        if "cc0" in tokens or "oga_potion" in text:
            return _PROFILES["cc0_potion"]
    if "mushroom" in tokens or "fungus" in tokens:
        return _PROFILES["mushroom"]
    return _PROFILES["generic_unknown"]


def source_profile_to_json(profile: SourceProfile) -> dict[str, Any]:
    result = {
        "name": profile.name,
        "domain": profile.domain,
        "filename_trust": profile.filename_trust,
        "expected_category_bias": list(profile.expected_category_bias),
        "known_path_tokens": list(profile.known_path_tokens),
        "notes": profile.notes,
    }
    if profile.sheet_specialization is not None:
        result["sheet_specialization"] = profile.sheet_specialization
    if profile.fusion_threshold_override is not None:
        result["fusion_threshold_override"] = profile.fusion_threshold_override
    return result

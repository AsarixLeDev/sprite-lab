"""Source/profile detection for harvest label v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from spritelab.harvest.config_loader import load_source_profiles_config
from spritelab.harvest.label_taxonomy import CATEGORY_VALUES, normalize_tag


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


_FALLBACK_PROFILES: dict[str, SourceProfile] = {
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
    "shade_weapons": SourceProfile(
        name="shade_weapons",
        domain="weapon",
        filename_trust="exact",
        expected_category_bias=("weapon",),
        known_path_tokens=("shade", "weapons"),
        notes="Trusted only with declarative sheet metadata.",
    ),
    "flare_armor": SourceProfile(
        name="flare_armor",
        domain="armor",
        filename_trust="exact",
        expected_category_bias=("armor",),
        known_path_tokens=("flare", "armor"),
        notes="Trusted only with declarative sheet metadata.",
    ),
    "farming_tools": SourceProfile(
        name="farming_tools",
        domain="tool",
        filename_trust="exact",
        expected_category_bias=("tool",),
        known_path_tokens=("farming", "calciumtrice", "tools"),
        notes="Trusted only with declarative sheet metadata.",
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


def hardcoded_source_profiles() -> dict[str, SourceProfile]:
    """Return the immutable built-in profiles used if config is absent."""

    return dict(_FALLBACK_PROFILES)


def _profile_data(profile: SourceProfile) -> dict[str, Any]:
    result: dict[str, Any] = {
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


def _load_profiles() -> dict[str, SourceProfile]:
    fallback = {"profiles": {name: _profile_data(profile) for name, profile in _FALLBACK_PROFILES.items()}}
    config = load_source_profiles_config(fallback)
    if set(config) != {"schema_version", "profiles"}:
        raise ValueError("invalid label-v2 source_profiles config: unknown or missing top-level keys")
    raw_profiles = config.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("invalid label-v2 source_profiles config: 'profiles' must be an object")
    if set(raw_profiles) != set(_FALLBACK_PROFILES):
        raise ValueError("invalid label-v2 source_profiles config: profile ids must match built-in profiles")

    profiles: dict[str, SourceProfile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("invalid label-v2 source_profiles config: every profile must be an object")
        allowed_keys = {
            "domain",
            "filename_trust",
            "expected_category_bias",
            "known_path_tokens",
            "notes",
            "sheet_specialization",
            "fusion_threshold_override",
        }
        if set(raw) - allowed_keys:
            raise ValueError(f"invalid source profile '{name}': unknown keys {sorted(set(raw) - allowed_keys)}")
        trust = raw.get("filename_trust")
        bias = raw.get("expected_category_bias")
        tokens = raw.get("known_path_tokens")
        if trust not in {"exact", "prefix_family", "none"}:
            raise ValueError(f"invalid source profile '{name}': filename_trust must be exact, prefix_family, or none")
        if not isinstance(raw.get("domain"), str) or not isinstance(bias, list) or not isinstance(tokens, list):
            raise ValueError(
                f"invalid source profile '{name}': domain, expected_category_bias, and known_path_tokens required"
            )
        if not all(isinstance(value, str) for value in [*bias, *tokens]):
            raise ValueError(f"invalid source profile '{name}': bias and path tokens must be strings")
        if any(value not in CATEGORY_VALUES for value in bias):
            raise ValueError(f"invalid source profile '{name}': expected_category_bias contains unknown category")
        override = raw.get("fusion_threshold_override")
        if override is not None and (not isinstance(override, (int, float)) or not 0.0 <= float(override) <= 1.0):
            raise ValueError(f"invalid source profile '{name}': fusion_threshold_override must be 0..1")
        profiles[name] = SourceProfile(
            name=name,
            domain=str(raw["domain"]),
            filename_trust=trust,
            expected_category_bias=tuple(bias),
            known_path_tokens=tuple(tokens),
            notes=str(raw.get("notes", "")),
            sheet_specialization=str(raw["sheet_specialization"]) if raw.get("sheet_specialization") else None,
            fusion_threshold_override=float(override) if override is not None else None,
        )
    return profiles


_PROFILES = _load_profiles()


def loaded_source_profiles() -> dict[str, SourceProfile]:
    """Return the validated active source-profile configuration."""

    return dict(_PROFILES)


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
    tokens.update(token for token in text.replace("\\", "_").replace("/", "_").split("_") if token)

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
    for profile in _PROFILES.values():
        if profile.name == "generic_unknown" or not profile.known_path_tokens:
            continue
        if set(profile.known_path_tokens) <= tokens:
            return profile
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

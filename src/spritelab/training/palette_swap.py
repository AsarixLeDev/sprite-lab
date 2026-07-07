"""Deterministic palette-swap augmentation for indexed sprite training data.

Operates on the exported indexed representation (index map + per-sprite palette +
role map) *before* RGBA target construction. Only palette RGB values are changed:
the index map, alpha, and palette mask are preserved, so the augmentation cannot
corrupt sprite geometry or produce invalid palette indices.

The augmentation is a pure function of a per-sample seed derived from the training
seed and the sprite id, so it is byte-for-byte reproducible, independent of
DataLoader worker order, and safe to memoize.

Beyond the raw recolor, this module evaluates *eligibility* (whether the sprite
has a real, confidently-detected fill color family and passes the configured
conservative filters). Augmentation is only applied to sprites that are both
triggered (by the per-sample probability gate) and eligible.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spritelab.codec.oklab import oklab_array_to_rgb_u8, rgb_u8_array_to_oklab
from spritelab.codec.roles import (
    ROLE_ACCENT,
    ROLE_DEEP_SHADOW,
    ROLE_HIGHLIGHT,
    ROLE_LIGHT,
    ROLE_MIDTONE,
    ROLE_OUTLINE,
    ROLE_SHADOW,
    ROLE_TEXTURE_DETAIL,
    ROLE_TRANSPARENT,
    role_name,
)

# Stable color families. The augmentation default set is a safe subset that
# excludes near-black/near-white families so recoloring stays visible.
FAMILY_ANCHORS: dict[str, tuple[int, int, int]] = {
    "red": (206, 52, 52),
    "blue": (56, 100, 208),
    "green": (54, 158, 74),
    "yellow": (232, 200, 66),
    "purple": (142, 70, 194),
    "brown": (128, 84, 48),
    "gold": (200, 158, 52),
    "gray": (128, 128, 128),
    "black": (24, 24, 24),
    "white": (236, 236, 236),
}
ALL_FAMILIES: tuple[str, ...] = tuple(FAMILY_ANCHORS.keys())
DEFAULT_SWAP_FAMILIES: tuple[str, ...] = ("red", "blue", "green", "yellow", "purple", "brown", "gold", "gray")
DEFAULT_SWAP_FAMILIES_TEXT = ",".join(DEFAULT_SWAP_FAMILIES)

# Families where a target hue can safely tint a near-white highlight.
TINT_HIGHLIGHT_FAMILIES: frozenset[str] = frozenset({"yellow", "gold", "gray", "white"})
# Families that desaturate rather than push a hue.
DESATURATED_FAMILIES: frozenset[str] = frozenset({"gray", "black", "white"})
# Achromatic families that are rarely a genuine "object color".
ACHROMATIC_FAMILIES: frozenset[str] = frozenset({"gray", "black", "white"})
# Families that double as materials, so their color is often ambiguous/material-like.
MATERIAL_COLOR_FAMILIES: frozenset[str] = frozenset({"gold", "brown"})

# Roles that are recolored (fill / highlight / accent surfaces).
_RECOLOR_ROLES: frozenset[int] = frozenset(
    {ROLE_MIDTONE, ROLE_LIGHT, ROLE_HIGHLIGHT, ROLE_ACCENT, ROLE_TEXTURE_DETAIL}
)
# Roles that must never be recolored regardless of options.
_ALWAYS_PRESERVE_ROLES: frozenset[int] = frozenset({ROLE_TRANSPARENT, ROLE_OUTLINE})
# Extra roles preserved only when preserve_outline is set.
_OUTLINE_PROTECTED_ROLES: frozenset[int] = frozenset({ROLE_DEEP_SHADOW, ROLE_SHADOW})
_KNOWN_VISIBLE_ROLES: frozenset[int] = frozenset(range(ROLE_OUTLINE, ROLE_TEXTURE_DETAIL + 1))

_COLOR_WORD_RE = {
    name: re.compile(rf"\b{name}\b", re.IGNORECASE) for name in (*ALL_FAMILIES, "grey")
}

# Precomputed anchor OKLab and hue direction/chroma.
_ANCHOR_OKLAB = {name: rgb_u8_array_to_oklab(np.asarray(rgb, dtype=np.uint8)) for name, rgb in FAMILY_ANCHORS.items()}
_ANCHOR_CHROMA = {name: float(np.hypot(lab[1], lab[2])) for name, lab in _ANCHOR_OKLAB.items()}
_ANCHOR_DIR = {
    name: (np.asarray([lab[1], lab[2]], dtype=np.float64) / (_ANCHOR_CHROMA[name] or 1.0))
    for name, lab in _ANCHOR_OKLAB.items()
}
_ANCHOR_STACK_NAMES = ALL_FAMILIES
_ANCHOR_STACK = np.stack([_ANCHOR_OKLAB[name] for name in _ANCHOR_STACK_NAMES])

# Fraction of visible pixels that should be fill for full color confidence.
_FILL_PRESENCE_TARGET = 0.35


@dataclass(frozen=True)
class PaletteSwapConfig:
    """Resolved palette-swap augmentation settings."""

    enabled: bool = False
    prob: float = 0.0
    families: tuple[str, ...] = DEFAULT_SWAP_FAMILIES  # target families
    preserve_outline: bool = True
    update_prompts: bool = True
    seed: int = 0
    # Conservative controls (defaults preserve original permissive behavior).
    source_families: tuple[str, ...] | None = None
    category_filter: tuple[str, ...] | None = None
    min_color_confidence: float = 0.0
    require_role_map: bool = False
    require_explicit_color: bool = False
    require_explicit_caption_color: bool = False
    require_explicit_semantic_color: bool = False
    no_caption_prepend: bool = False
    allow_material_colors: bool = True

    @classmethod
    def from_training_config(cls, config: Any) -> "PaletteSwapConfig":
        target = getattr(config, "palette_swap_target_families", None)
        if target:
            families = parse_families(target)
        else:
            families = parse_families(getattr(config, "palette_swap_families", DEFAULT_SWAP_FAMILIES_TEXT))
        return cls(
            enabled=bool(getattr(config, "palette_swap_augmentation", False)),
            prob=float(getattr(config, "palette_swap_prob", 0.0)),
            families=families,
            preserve_outline=bool(getattr(config, "palette_swap_preserve_outline", True)),
            update_prompts=bool(getattr(config, "palette_swap_update_prompts", True)),
            seed=int(getattr(config, "seed", 0)),
            source_families=parse_family_filter(getattr(config, "palette_swap_source_families", None)),
            category_filter=parse_category_filter(getattr(config, "palette_swap_category_filter", None)),
            min_color_confidence=float(getattr(config, "palette_swap_min_color_confidence", 0.0)),
            require_role_map=bool(getattr(config, "palette_swap_require_role_map", False)),
            require_explicit_color=bool(getattr(config, "palette_swap_require_explicit_color", False)),
            require_explicit_caption_color=bool(
                getattr(config, "palette_swap_require_explicit_caption_color", False)
            ),
            require_explicit_semantic_color=bool(
                getattr(config, "palette_swap_require_explicit_semantic_color", False)
            ),
            no_caption_prepend=bool(getattr(config, "palette_swap_no_caption_prepend", False)),
            allow_material_colors=bool(getattr(config, "palette_swap_allow_material_colors", True)),
        )

    def active(self) -> bool:
        return bool(self.enabled) and float(self.prob) > 0.0 and bool(self.families)

    def require_caption_color(self) -> bool:
        """Resolved caption-color gate; legacy alias enables this gate too."""

        return bool(self.require_explicit_caption_color or self.require_explicit_color)

    def require_semantic_color(self) -> bool:
        """Resolved semantic-color gate; legacy alias enables this gate too."""

        return bool(self.require_explicit_semantic_color or self.require_explicit_color)

    def report_dict(self) -> dict[str, Any]:
        return {
            "palette_swap_augmentation": bool(self.enabled),
            "palette_swap_prob": float(self.prob),
            "palette_swap_families": list(self.families),
            "palette_swap_target_families": list(self.families),
            "palette_swap_source_families": None if self.source_families is None else list(self.source_families),
            "palette_swap_category_filter": None if self.category_filter is None else list(self.category_filter),
            "palette_swap_min_color_confidence": float(self.min_color_confidence),
            "palette_swap_require_role_map": bool(self.require_role_map),
            "palette_swap_require_explicit_color": bool(self.require_explicit_color),
            "palette_swap_require_explicit_color_maps_to": "caption_and_semantic",
            "palette_swap_require_explicit_caption_color": self.require_caption_color(),
            "palette_swap_require_explicit_semantic_color": self.require_semantic_color(),
            "palette_swap_no_caption_prepend": bool(self.no_caption_prepend),
            "palette_swap_allow_material_colors": bool(self.allow_material_colors),
            "palette_swap_preserve_outline": bool(self.preserve_outline),
            "palette_swap_update_prompts": bool(self.update_prompts),
        }


@dataclass
class PaletteSwapResult:
    applied: bool
    palette_rgb: np.ndarray
    record: Mapping[str, Any]
    caption: str
    source_color_family: str = ""
    target_color_family: str = ""
    seed: int = 0
    roles_recolored: list[str] = field(default_factory=list)
    triggered: bool = False
    eligible: bool = False
    ineligibility_reason: str | None = None
    source_color_confidence: float = 0.0
    role_map_trusted: bool = False
    fallback_heuristic_used: bool = False
    recolored_palette_indices: list[int] = field(default_factory=list)
    caption_change_kind: str = "none"  # replaced | prepended | none
    materials_dropped: list[str] = field(default_factory=list)
    target_resampled_from_same_family: bool = False
    same_family_skip: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "palette_swap_applied": bool(self.applied),
            "palette_swap_triggered": bool(self.triggered),
            "eligible_for_palette_swap": bool(self.eligible),
            "ineligibility_reason": self.ineligibility_reason,
            "source_color_family": self.source_color_family,
            "source_color_confidence": round(float(self.source_color_confidence), 4),
            "target_color_family": self.target_color_family,
            "role_map_trusted": bool(self.role_map_trusted),
            "fallback_heuristic_used": bool(self.fallback_heuristic_used),
            "roles_recolored": list(self.roles_recolored),
            "recolored_palette_indices": list(self.recolored_palette_indices),
            "caption_change_kind": self.caption_change_kind,
            "material_conflict_drop_count": len(self.materials_dropped),
            "target_resampled_from_same_family": bool(self.target_resampled_from_same_family),
            "same_family_skip": bool(self.same_family_skip),
            "palette_swap_seed": int(self.seed),
        }


def parse_families(text: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated (or sequence) family spec into known families."""

    families = _parse_family_tokens(text)
    return families or DEFAULT_SWAP_FAMILIES


def parse_family_filter(text: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Parse an optional family filter; ``None``/empty means "no restriction"."""

    if text is None:
        return None
    families = _parse_family_tokens(text)
    return families or None


def parse_category_filter(text: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Parse an optional category filter; ``None``/empty means "no restriction"."""

    if text is None:
        return None
    if isinstance(text, str):
        tokens = [part.strip().lower() for part in text.split(",")]
    else:
        tokens = [str(part).strip().lower() for part in text]
    tokens = tuple(token for token in tokens if token)
    return tokens or None


def _parse_family_tokens(text: str | Sequence[str] | None) -> tuple[str, ...]:
    if text is None:
        return ()
    if isinstance(text, str):
        tokens = [part.strip().lower() for part in text.split(",")]
    else:
        tokens = [str(part).strip().lower() for part in text]
    return tuple(token for token in tokens if token in FAMILY_ANCHORS)


def sample_seed(base_seed: int, sprite_id: str) -> int:
    """Stable per-sample seed independent of process/worker/hash randomization."""

    digest = hashlib.sha256(f"{int(base_seed)}:{sprite_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def nearest_family(rgb_float: Sequence[float]) -> str:
    """Return the nearest color family name for one RGB triple in [0, 1]."""

    rgb_u8 = np.clip(np.rint(np.asarray(rgb_float, dtype=np.float64)[:3] * 255.0), 0, 255).astype(np.uint8)
    lab = rgb_u8_array_to_oklab(rgb_u8)
    distances = np.linalg.norm(_ANCHOR_STACK - lab, axis=-1)
    return _ANCHOR_STACK_NAMES[int(np.argmin(distances))]


def apply_palette_swap(
    *,
    index_map: np.ndarray,
    alpha: np.ndarray,
    role_map: np.ndarray | None,
    palette_rgb: np.ndarray,
    palette_mask: np.ndarray,
    record: Mapping[str, Any],
    caption: str,
    sprite_id: str,
    config: PaletteSwapConfig,
) -> PaletteSwapResult:
    """Evaluate eligibility and recolor visible fill entries into a target family.

    The returned result always carries full diagnostics (source family,
    confidence, eligibility, reason, role-map trust, fallback usage) so it can be
    used both by the DataLoader and by the offline review tool. The palette is
    only actually recolored when the sample is both triggered and eligible.
    """

    palette = np.array(palette_rgb, dtype=np.float32, copy=True)
    seed = sample_seed(config.seed, sprite_id)
    result = PaletteSwapResult(applied=False, palette_rgb=palette, record=record, caption=caption, seed=seed)
    if not config.active():
        return result

    rng = np.random.default_rng(seed)
    triggered = bool(rng.random() < float(config.prob))
    targets = tuple(config.families)
    target_index = int(rng.integers(0, len(targets)))
    target_family = targets[target_index]
    result.triggered = triggered

    index = np.asarray(index_map)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim == 3 and alpha_arr.shape[0] == 1:
        alpha_arr = alpha_arr[0]
    visible = alpha_arr > 0.0
    mask = np.asarray(palette_mask, dtype=bool)

    entry_role, entry_coverage, role_reliable = _entry_roles(index, visible, role_map, mask.shape[0])
    result.role_map_trusted = bool(role_reliable)
    result.fallback_heuristic_used = not bool(role_reliable)

    recolor_indices, roles_recolored = _select_recolor_entries(
        entry_role=entry_role,
        entry_coverage=entry_coverage,
        role_reliable=role_reliable,
        palette=palette,
        mask=mask,
        preserve_outline=config.preserve_outline,
        target_family=target_family,
    )

    total_visible = int(sum(entry_coverage.values()))
    source_family, confidence = _source_family_and_confidence(palette, recolor_indices, entry_coverage, total_visible)
    result.source_color_family = source_family
    result.source_color_confidence = confidence

    eligible, reason = _evaluate_eligibility(
        config,
        recolor_indices=recolor_indices,
        role_reliable=role_reliable,
        source_family=source_family,
        confidence=confidence,
        record=record,
        caption=caption,
        total_visible=total_visible,
    )
    different_targets = tuple(family for family in targets if family != source_family)
    if eligible and source_family and not different_targets:
        eligible = False
        reason = "same_family_target_unavailable"
        result.same_family_skip = True
    result.eligible = eligible
    result.ineligibility_reason = reason

    if not (triggered and eligible):
        return result

    # Guarantee the augmentation actually changes the dominant hue.
    if target_family == source_family:
        target_family = different_targets[int(rng.integers(0, len(different_targets)))]
        result.target_resampled_from_same_family = True

    palette[recolor_indices, :3] = _recolor_entries(palette[recolor_indices, :3], target_family)
    result.applied = True
    result.target_color_family = target_family
    result.roles_recolored = roles_recolored
    result.recolored_palette_indices = sorted(int(idx) for idx in recolor_indices)

    if config.update_prompts:
        updated_record, updated_caption, caption_kind, dropped = _update_prompt_fields(
            record,
            caption,
            target_family,
            no_caption_prepend=config.no_caption_prepend,
        )
        result.record = updated_record
        result.caption = updated_caption
        result.caption_change_kind = caption_kind
        result.materials_dropped = dropped
    return result


def estimate_applied(records: Sequence[Mapping[str, Any]], config: PaletteSwapConfig) -> dict[str, Any]:
    """Approximate applied count/rate from the deterministic per-sample trigger.

    This ignores eligibility (it does not load images); use
    :func:`spritelab.training.palette_swap_review.summarize_dataset_palette_swap`
    for eligibility-aware statistics.
    """

    if not config.active():
        return {"applied_count": 0, "applied_rate": 0.0, "sample_count": len(records)}
    applied = 0
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        rng = np.random.default_rng(sample_seed(config.seed, sprite_id))
        if bool(rng.random() < float(config.prob)):
            applied += 1
    total = len(records)
    return {
        "applied_count": applied,
        "applied_rate": (applied / float(total)) if total else 0.0,
        "sample_count": total,
    }


def _evaluate_eligibility(
    config: PaletteSwapConfig,
    *,
    recolor_indices: Sequence[int],
    role_reliable: bool,
    source_family: str,
    confidence: float,
    record: Mapping[str, Any],
    caption: str,
    total_visible: int,
) -> tuple[bool, str | None]:
    if total_visible <= 0:
        return False, "no_visible_pixels"
    if not recolor_indices:
        return False, "no_recolorable_fill"
    if config.require_role_map and not role_reliable:
        return False, "role_map_unreliable"
    if config.category_filter is not None:
        category = _record_category(record)
        if category not in config.category_filter:
            return False, "category_filtered"
    if not source_family:
        return False, "unknown_source_family"
    if config.source_families is not None and source_family not in config.source_families:
        return False, "source_family_filtered"
    if not config.allow_material_colors and source_family in MATERIAL_COLOR_FAMILIES:
        return False, "material_color_source"
    if config.min_color_confidence > 0.0 and confidence < float(config.min_color_confidence):
        return False, "low_color_confidence"
    caption_colors = _explicit_caption_colors(record, caption)
    semantic_colors = _explicit_semantic_colors(record)
    if config.require_caption_color() and not caption_colors:
        return False, "no_explicit_caption_color"
    if config.require_semantic_color() and not semantic_colors:
        return False, "no_explicit_semantic_color"
    if config.no_caption_prepend and not caption_colors and not (config.require_semantic_color() and semantic_colors):
        return False, "no_caption_color_no_prepend"
    return True, None


def _source_family_and_confidence(
    palette: np.ndarray,
    recolor_indices: Sequence[int],
    entry_coverage: Mapping[int, int],
    total_visible: int,
) -> tuple[str, float]:
    if not recolor_indices or total_visible <= 0:
        return "", 0.0
    families = {int(idx): nearest_family(palette[int(idx), :3]) for idx in recolor_indices}
    family_pixels: dict[str, int] = {}
    fill_visible = 0
    for idx in recolor_indices:
        coverage = int(entry_coverage.get(int(idx), 0))
        fill_visible += coverage
        family_pixels[families[int(idx)]] = family_pixels.get(families[int(idx)], 0) + coverage
    if fill_visible <= 0:
        return "", 0.0
    source_family = max(family_pixels, key=lambda name: (family_pixels[name], name))
    dominant_fraction = family_pixels[source_family] / float(fill_visible)
    fill_presence = min(1.0, (fill_visible / float(total_visible)) / _FILL_PRESENCE_TARGET)
    confidence = float(np.clip(dominant_fraction * fill_presence, 0.0, 1.0))
    return source_family, confidence


def _record_category(record: Mapping[str, Any]) -> str:
    for value in (record.get("category"), record.get("prompt_category")):
        token = str(value or "").strip().lower()
        if token:
            return token
    conditioning = record.get("conditioning") if isinstance(record.get("conditioning"), Mapping) else {}
    semantic = conditioning.get("semantic_v3") if isinstance(conditioning, Mapping) else {}
    if isinstance(semantic, Mapping):
        token = str(semantic.get("category") or "").strip().lower()
        if token:
            return token
    return ""


def _explicit_prompt_colors(record: Mapping[str, Any], caption: str, *, allow_material: bool) -> set[str]:
    prompt_colors: set[str] = set(_explicit_caption_colors(record, caption))
    prompt_colors |= _explicit_semantic_colors(record)
    if allow_material:
        prompt_colors |= _families_in_value(record.get("materials"))
        prompt_colors |= _families_in_value(record.get("material"))
    return prompt_colors


def _explicit_caption_colors(record: Mapping[str, Any], caption: str) -> set[str]:
    prompt_colors: set[str] = set(_families_in_text(caption))
    for key in ("caption", "prompt"):
        prompt_colors |= _families_in_text(record.get(key))
    return prompt_colors


def _explicit_semantic_colors(record: Mapping[str, Any]) -> set[str]:
    prompt_colors: set[str] = set()
    prompt_colors |= _families_in_value(record.get("colors"))
    prompt_colors |= _families_in_value(record.get("color"))
    prompt_colors |= _families_in_value(record.get("primary_color"))
    conditioning = record.get("conditioning") if isinstance(record.get("conditioning"), Mapping) else {}
    semantic = conditioning.get("semantic_v3") if isinstance(conditioning, Mapping) else {}
    if isinstance(semantic, Mapping):
        prompt_colors |= _families_in_value(semantic.get("colors"))
        attributes = semantic.get("attributes") if isinstance(semantic.get("attributes"), Mapping) else {}
        prompt_colors |= _families_in_value(attributes.get("colors"))
    target_semantics = record.get("target_semantics") if isinstance(record.get("target_semantics"), Mapping) else {}
    if isinstance(target_semantics, Mapping):
        prompt_colors |= _families_in_value(target_semantics.get("colors"))
        attributes = target_semantics.get("attributes") if isinstance(target_semantics.get("attributes"), Mapping) else {}
        prompt_colors |= _families_in_value(attributes.get("colors"))
    return prompt_colors


def _families_in_text(text: Any) -> set[str]:
    value = str(text or "")
    found: set[str] = set()
    for name, pattern in _COLOR_WORD_RE.items():
        if pattern.search(value):
            found.add("gray" if name == "grey" else name)
    return found


def _families_in_value(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        token = value.strip().lower().replace("grey", "gray")
        return {token} if token in FAMILY_ANCHORS else set()
    if isinstance(value, Mapping):
        found: set[str] = set()
        for nested in value.values():
            found |= _families_in_value(nested)
        return found
    if isinstance(value, Sequence):
        found = set()
        for item in value:
            found |= _families_in_value(item)
        return found
    return set()


def _entry_roles(
    index: np.ndarray,
    visible: np.ndarray,
    role_map: np.ndarray | None,
    palette_size: int,
) -> tuple[dict[int, int], dict[int, int], bool]:
    """Return per-palette-index majority role, coverage, and role reliability."""

    entry_role: dict[int, int] = {}
    entry_coverage: dict[int, int] = {}
    visible_index = index[visible].astype(np.int64, copy=False)
    if visible_index.size == 0:
        return entry_role, entry_coverage, False

    roles_flat = None
    if role_map is not None:
        role_arr = np.asarray(role_map)
        if role_arr.shape == index.shape:
            roles_flat = role_arr[visible].astype(np.int64, copy=False)

    known_visible = 0
    for palette_index in np.unique(visible_index):
        palette_index = int(palette_index)
        selection = visible_index == palette_index
        entry_coverage[palette_index] = int(np.count_nonzero(selection))
        if roles_flat is not None:
            entry_role_values = roles_flat[selection]
            if entry_role_values.size:
                values, counts = np.unique(entry_role_values, return_counts=True)
                majority = int(values[int(np.argmax(counts))])
                entry_role[palette_index] = majority
                known_visible += int(np.count_nonzero(np.isin(entry_role_values, list(_KNOWN_VISIBLE_ROLES))))
    reliable = roles_flat is not None and known_visible >= 0.5 * float(visible_index.size)
    return entry_role, entry_coverage, reliable


def _select_recolor_entries(
    *,
    entry_role: Mapping[int, int],
    entry_coverage: Mapping[int, int],
    role_reliable: bool,
    palette: np.ndarray,
    mask: np.ndarray,
    preserve_outline: bool,
    target_family: str,
) -> tuple[list[int], list[str]]:
    recolor_indices: list[int] = []
    roles_recolored: set[str] = set()
    for palette_index in sorted(entry_coverage):
        if palette_index < 0 or palette_index >= mask.shape[0] or not bool(mask[palette_index]):
            continue
        role = entry_role.get(palette_index)
        if role_reliable and role is not None:
            if role in _ALWAYS_PRESERVE_ROLES:
                continue
            if preserve_outline and role in _OUTLINE_PROTECTED_ROLES:
                continue
            if role in _RECOLOR_ROLES or role not in _KNOWN_VISIBLE_ROLES:
                if not _luminance_recolorable(palette[palette_index, :3], target_family):
                    continue
                recolor_indices.append(palette_index)
                roles_recolored.add(role_name(int(role)) if role is not None else "unknown")
            continue
        # Fallback: no reliable role map. Use luminance/chroma heuristics.
        if _luminance_recolorable(palette[palette_index, :3], target_family):
            recolor_indices.append(palette_index)
            roles_recolored.add(_luminance_bucket(palette[palette_index, :3]))
    return recolor_indices, sorted(roles_recolored)


def _luminance_recolorable(rgb_float: np.ndarray, target_family: str) -> bool:
    lab = rgb_u8_array_to_oklab(_to_u8(rgb_float))
    lum = float(lab[0])
    if lum < 0.20:  # near-black outline / deep shadow
        return False
    if lum > 0.90 and target_family not in TINT_HIGHLIGHT_FAMILIES:
        return False
    return True


def _luminance_bucket(rgb_float: np.ndarray) -> str:
    lab = rgb_u8_array_to_oklab(_to_u8(rgb_float))
    lum = float(lab[0])
    if lum < 0.45:
        return "midtone"
    if lum < 0.75:
        return "light"
    return "highlight"


def _recolor_entries(rgb_float: np.ndarray, target_family: str) -> np.ndarray:
    rgb_u8 = _to_u8(rgb_float)
    lab = rgb_u8_array_to_oklab(rgb_u8).reshape(-1, 3)
    lum = lab[:, 0]
    if target_family in DESATURATED_FAMILIES:
        new_lab = np.stack([lum, np.zeros_like(lum), np.zeros_like(lum)], axis=-1)
        return oklab_array_to_rgb_u8(new_lab).astype(np.float32) / 255.0

    source_chroma = np.hypot(lab[:, 1], lab[:, 2])
    anchor_chroma = _ANCHOR_CHROMA[target_family]
    direction = _ANCHOR_DIR[target_family]
    # Preserve relative luminance; reduce chroma near luminance extremes so the
    # recolored ramp keeps dark/mid/light structure without clipping.
    lum_factor = np.clip(4.0 * lum * (1.0 - lum), 0.4, 1.0)
    target_chroma = np.clip(np.maximum(source_chroma, anchor_chroma * 0.6), 0.0, anchor_chroma * 1.15) * lum_factor
    new_a = direction[0] * target_chroma
    new_b = direction[1] * target_chroma
    new_lab = np.stack([lum, new_a, new_b], axis=-1)
    return oklab_array_to_rgb_u8(new_lab).astype(np.float32) / 255.0


def _to_u8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(rgb_float, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _update_prompt_fields(
    record: Mapping[str, Any],
    caption: str,
    target_family: str,
    *,
    no_caption_prepend: bool = False,
) -> tuple[dict[str, Any], str, str, list[str]]:
    updated = copy.deepcopy(dict(record))
    updated_caption, caption_kind = _rewrite_caption(
        caption,
        target_family,
        no_caption_prepend=no_caption_prepend,
    )
    updated["caption"] = updated_caption
    updated["colors"] = [target_family]
    updated["color"] = target_family
    updated["primary_color"] = target_family
    dropped: list[str] = []
    _filter_materials(updated, target_family, key="materials", dropped=dropped)
    _filter_materials(updated, target_family, key="material", dropped=dropped)

    conditioning = updated.get("conditioning")
    if isinstance(conditioning, dict):
        semantic = conditioning.get("semantic_v3")
        if isinstance(semantic, dict):
            attributes = semantic.get("attributes")
            if isinstance(attributes, dict):
                _overwrite_attribute_colors(attributes, target_family, dropped=dropped)
    target_semantics = updated.get("target_semantics")
    if isinstance(target_semantics, dict):
        attributes = target_semantics.get("attributes")
        if isinstance(attributes, dict):
            _overwrite_attribute_colors(attributes, target_family, dropped=dropped)
    return updated, updated_caption, caption_kind, dropped


def _overwrite_attribute_colors(attributes: dict[str, Any], target_family: str, *, dropped: list[str]) -> None:
    attributes["colors"] = [target_family]
    materials = attributes.get("materials")
    if materials is not None:
        attributes["materials"] = _drop_color_materials(materials, target_family, dropped=dropped)


def _filter_materials(record: dict[str, Any], target_family: str, *, key: str, dropped: list[str]) -> None:
    if key not in record:
        return
    record[key] = _drop_color_materials(record[key], target_family, dropped=dropped)


def _drop_color_materials(value: Any, target_family: str, *, dropped: list[str]) -> Any:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in FAMILY_ANCHORS and token != target_family:
            dropped.append(token)
            return ""
        return value
    if isinstance(value, Sequence):
        kept = []
        for item in value:
            token = str(item).strip().lower()
            if token in FAMILY_ANCHORS and token != target_family:
                dropped.append(token)
                continue
            kept.append(item)
        return kept
    return value


def _rewrite_caption(caption: str, target_family: str, *, no_caption_prepend: bool = False) -> tuple[str, str]:
    text = str(caption or "")
    replaced = False
    for name, pattern in _COLOR_WORD_RE.items():
        if pattern.search(text):
            text = pattern.sub(target_family, text)
            replaced = True
    if replaced:
        # Collapse any repeated target family words introduced by replacement.
        text = re.sub(rf"(?:\b{target_family}\b)(?:\s+\b{target_family}\b)+", target_family, text)
        return text.strip(), "replaced"
    if no_caption_prepend:
        return text.strip(), "none"
    if not text.strip():
        return target_family, "prepended"
    return f"{target_family} {text.strip()}", "prepended"

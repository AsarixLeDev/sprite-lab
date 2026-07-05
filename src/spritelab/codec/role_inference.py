"""Deterministic role-map inference for palette-index sprite bundles."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_SIZE, SPRITE_WIDTH, SpriteBundle, SpriteMetadata
from spritelab.codec.oklab import rgb_u8_array_to_oklab
from spritelab.codec.roles import (
    ROLE_ACCENT,
    ROLE_DEEP_SHADOW,
    ROLE_EMISSIVE,
    ROLE_HIGHLIGHT,
    ROLE_LIGHT,
    ROLE_MIDTONE,
    ROLE_NAMES,
    ROLE_OUTLINE,
    ROLE_PREVIEW_COLORS,
    ROLE_SHADOW,
    ROLE_TEXTURE_DETAIL,
    ROLE_TRANSPARENT,
    ROLE_UNKNOWN,
    role_name,
)
from spritelab.codec.validate import assert_valid_bundle

ROLE_INFERENCE_VERSION = "v2_heuristic"
KNOWN_ROLE_IDS = set(ROLE_NAMES)


@dataclass(frozen=True)
class RoleInferenceOptions:
    use_spatial_bias: bool = True
    use_edge_contact: bool = True
    use_chroma: bool = True
    use_frequency: bool = True
    rare_color_frequency_threshold: float = 0.08
    emissive_luminance_threshold: float = 0.68
    low_contrast_luminance_range: float = 0.12
    preserve_existing_role_map: bool = False


@dataclass(frozen=True)
class PaletteSlotRoleFeatures:
    slot: int
    rgb: tuple[int, int, int]

    oklab_l: float
    oklab_a: float
    oklab_b: float
    chroma: float
    hue_degrees: float

    pixel_count: int
    frequency: float

    edge_contact_count: int
    edge_contact_ratio: float

    transparent_neighbor_ratio: float
    outline_neighbor_ratio: float | None

    mean_x: float
    mean_y: float
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    local_contrast_mean: float
    local_same_color_ratio: float

    is_rare: bool
    is_dark: bool
    is_light: bool
    is_high_chroma: bool


@dataclass(frozen=True)
class RoleInferenceResult:
    slot_roles: dict[int, int]
    role_map: np.ndarray
    slot_features: dict[int, PaletteSlotRoleFeatures]
    confidence: dict[int, float]
    debug_scores: dict[int, dict[str, float]] = field(default_factory=dict)


def compute_palette_slot_role_features(
    palette: np.ndarray,
    index_map: np.ndarray,
    alpha: np.ndarray,
) -> dict[int, PaletteSlotRoleFeatures]:
    """Compute deterministic role features for every visible palette slot."""

    _validate_role_inputs(palette, index_map, alpha)
    opaque_pixel_count = int(np.count_nonzero(alpha == 1))
    if opaque_pixel_count == 0:
        raise ValueError("Cannot infer roles for an empty sprite with no opaque pixels.")

    palette_u8 = np.asarray(palette, dtype=np.uint8)
    palette_oklab = rgb_u8_array_to_oklab(palette_u8)
    raw: list[dict[str, Any]] = []

    for slot in range(1, int(palette_u8.shape[0])):
        slot_mask = index_map == slot
        pixel_count = int(np.count_nonzero(slot_mask))
        rgb = tuple(int(channel) for channel in palette_u8[slot])
        lab = palette_oklab[slot]
        chroma = float(math.sqrt(float(lab[1]) ** 2 + float(lab[2]) ** 2))
        hue = _hue_degrees(float(lab[1]), float(lab[2]), chroma)

        if pixel_count == 0:
            raw.append(
                {
                    "slot": slot,
                    "rgb": rgb,
                    "oklab_l": float(lab[0]),
                    "oklab_a": float(lab[1]),
                    "oklab_b": float(lab[2]),
                    "chroma": chroma,
                    "hue_degrees": hue,
                    "pixel_count": 0,
                    "frequency": 0.0,
                    "edge_contact_count": 0,
                    "edge_contact_ratio": 0.0,
                    "transparent_neighbor_ratio": 0.0,
                    "outline_neighbor_ratio": None,
                    "mean_x": -1.0,
                    "mean_y": -1.0,
                    "min_x": -1,
                    "min_y": -1,
                    "max_x": -1,
                    "max_y": -1,
                    "local_contrast_mean": 0.0,
                    "local_same_color_ratio": 0.0,
                }
            )
            continue

        coords = np.argwhere(slot_mask)
        min_y = int(np.min(coords[:, 0]))
        max_y = int(np.max(coords[:, 0]))
        min_x = int(np.min(coords[:, 1]))
        max_x = int(np.max(coords[:, 1]))
        edge_contact_count = 0
        transparent_neighbor_sides = 0
        local_contrast_sum = 0.0
        opaque_neighbor_sides = 0
        same_color_neighbor_sides = 0

        for y_raw, x_raw in coords:
            y = int(y_raw)
            x = int(x_raw)
            touches_transparency = False
            for ny, nx in _neighbors4_with_outside(y, x):
                if ny < 0 or nx < 0 or ny >= SPRITE_HEIGHT or nx >= SPRITE_WIDTH:
                    transparent_neighbor_sides += 1
                    touches_transparency = True
                    continue
                if int(alpha[ny, nx]) == 0:
                    transparent_neighbor_sides += 1
                    touches_transparency = True
                    continue

                opaque_neighbor_sides += 1
                neighbor_slot = int(index_map[ny, nx])
                if neighbor_slot == slot:
                    same_color_neighbor_sides += 1
                if 0 <= neighbor_slot < palette_oklab.shape[0]:
                    local_contrast_sum += abs(float(lab[0]) - float(palette_oklab[neighbor_slot, 0]))

            if touches_transparency:
                edge_contact_count += 1

        local_contrast_mean = local_contrast_sum / opaque_neighbor_sides if opaque_neighbor_sides else 0.0
        local_same_color_ratio = same_color_neighbor_sides / opaque_neighbor_sides if opaque_neighbor_sides else 0.0

        raw.append(
            {
                "slot": slot,
                "rgb": rgb,
                "oklab_l": float(lab[0]),
                "oklab_a": float(lab[1]),
                "oklab_b": float(lab[2]),
                "chroma": chroma,
                "hue_degrees": hue,
                "pixel_count": pixel_count,
                "frequency": pixel_count / opaque_pixel_count,
                "edge_contact_count": edge_contact_count,
                "edge_contact_ratio": edge_contact_count / pixel_count,
                "transparent_neighbor_ratio": transparent_neighbor_sides / (pixel_count * 4),
                "outline_neighbor_ratio": None,
                "mean_x": float(np.mean(coords[:, 1])),
                "mean_y": float(np.mean(coords[:, 0])),
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "local_contrast_mean": local_contrast_mean,
                "local_same_color_ratio": local_same_color_ratio,
            }
        )

    return _with_relative_flags(raw, options=RoleInferenceOptions())


def infer_palette_slot_roles_v2(
    palette: np.ndarray,
    index_map: np.ndarray,
    alpha: np.ndarray,
    *,
    options: RoleInferenceOptions | None = None,
) -> RoleInferenceResult:
    """Infer semantic roles for visible palette slots and a per-pixel role map."""

    opts = options or RoleInferenceOptions()
    features = compute_palette_slot_role_features(palette, index_map, alpha)
    used_features = {slot: feature for slot, feature in features.items() if feature.pixel_count > 0}
    if not used_features:
        raise ValueError("Cannot infer roles for an empty sprite with no used visible palette slots.")

    if len(used_features) <= 3:
        slot_roles, debug_scores = _tiny_palette_roles(used_features, options=opts)
    else:
        slot_roles, debug_scores = _scored_roles(used_features, options=opts)

    for slot, feature in features.items():
        if feature.pixel_count == 0:
            slot_roles[slot] = ROLE_UNKNOWN
            debug_scores[slot] = {"unknown": 1.0}

    role_map = build_role_map_from_slot_roles(index_map, alpha, slot_roles)
    role_map = _refine_role_map(role_map, index_map, alpha, slot_roles, features)
    confidence = {
        slot: _confidence_for_scores(debug_scores.get(slot, {}), slot_roles.get(slot, ROLE_UNKNOWN))
        for slot in features
    }
    return RoleInferenceResult(
        slot_roles=dict(sorted(slot_roles.items())),
        role_map=role_map,
        slot_features=features,
        confidence=confidence,
        debug_scores={slot: dict(sorted(scores.items())) for slot, scores in sorted(debug_scores.items())},
    )


def build_role_map_from_slot_roles(
    index_map: np.ndarray,
    alpha: np.ndarray,
    slot_roles: dict[int, int],
) -> np.ndarray:
    """Build a 32x32 role map from palette-slot roles."""

    if index_map.shape != SPRITE_SIZE:
        raise ValueError("index_map shape must be exactly 32x32.")
    if alpha.shape != SPRITE_SIZE:
        raise ValueError("alpha shape must be exactly 32x32.")

    role_map = np.full(SPRITE_SIZE, ROLE_UNKNOWN, dtype=np.uint8)
    role_map[alpha == 0] = ROLE_TRANSPARENT
    for slot in np.unique(index_map[alpha == 1]):
        slot_int = int(slot)
        role = int(slot_roles.get(slot_int, ROLE_UNKNOWN))
        role_map[(alpha == 1) & (index_map == slot_int)] = role
    return role_map


def apply_role_inference_to_bundle(
    bundle: SpriteBundle,
    *,
    options: RoleInferenceOptions | None = None,
    preserve_existing: bool = False,
) -> SpriteBundle:
    """Return a copy of ``bundle`` with an inferred role map."""

    assert_valid_bundle(bundle)
    opts = options or RoleInferenceOptions()
    if (preserve_existing or opts.preserve_existing_role_map) and bundle.role_map is not None:
        return _copy_bundle_with_metadata(bundle, _metadata_with_role_inference(bundle.metadata, preserved=True))

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha, options=opts)
    metadata = _metadata_with_role_inference(bundle.metadata, preserved=False)
    output = SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=np.asarray(bundle.palette).copy(),
        index_map=np.asarray(bundle.index_map).copy(),
        role_map=np.asarray(result.role_map, dtype=np.uint8).copy(),
        metadata=metadata,
    )
    assert_valid_bundle(output)
    return output


def describe_role_inference(result: RoleInferenceResult) -> list[str]:
    """Return human-readable debug lines for inferred palette roles."""

    lines: list[str] = []
    for slot in sorted(result.slot_features):
        feature = result.slot_features[slot]
        role = result.slot_roles.get(slot, ROLE_UNKNOWN)
        rgb_hex = f"#{feature.rgb[0]:02x}{feature.rgb[1]:02x}{feature.rgb[2]:02x}"
        lines.append(
            f"slot {slot} {rgb_hex} {role_name(role)} "
            f"confidence={result.confidence.get(slot, 0.0):.2f} "
            f"freq={feature.frequency:.2f} edge={feature.edge_contact_ratio:.2f} "
            f"L={feature.oklab_l:.2f} C={feature.chroma:.2f}"
        )
    return lines


def validate_role_map(role_map: np.ndarray, alpha: np.ndarray) -> list[str]:
    """Validate role-map shape, known IDs, and basic alpha consistency."""

    errors: list[str] = []
    if not isinstance(role_map, np.ndarray):
        return ["role_map must be a numpy array."]
    if not isinstance(alpha, np.ndarray):
        return ["alpha must be a numpy array."]
    if role_map.shape != SPRITE_SIZE:
        errors.append("role_map shape must be exactly 32x32.")
        return errors
    if alpha.shape != SPRITE_SIZE:
        errors.append("alpha shape must be exactly 32x32.")
        return errors

    unknown_values = sorted(int(value) for value in np.unique(role_map) if int(value) not in KNOWN_ROLE_IDS)
    if unknown_values:
        errors.append(f"role_map contains unknown role IDs: {unknown_values}.")
    if bool(np.any((alpha == 0) & (role_map != ROLE_TRANSPARENT))):
        errors.append("transparent pixels should have ROLE_TRANSPARENT.")
    if bool(np.any((alpha == 1) & (role_map == ROLE_TRANSPARENT))):
        errors.append("opaque pixels should not have ROLE_TRANSPARENT.")
    return errors


def role_map_to_preview_image(
    role_map: np.ndarray,
    *,
    scale: int = 8,
) -> Image.Image:
    """Convert a 32x32 role map to a nearest-neighbor RGBA debug preview."""

    if scale < 1:
        raise ValueError("scale must be at least 1.")
    if not isinstance(role_map, np.ndarray) or role_map.shape != SPRITE_SIZE:
        raise ValueError("role_map shape must be exactly 32x32.")

    pixels = np.zeros((SPRITE_HEIGHT, SPRITE_WIDTH, 4), dtype=np.uint8)
    for role_id in np.unique(role_map):
        role_int = int(role_id)
        color = ROLE_PREVIEW_COLORS.get(role_int, ROLE_PREVIEW_COLORS[ROLE_UNKNOWN])
        pixels[role_map == role_int] = color
    image = Image.fromarray(pixels, mode="RGBA")
    return image.resize((SPRITE_WIDTH * scale, SPRITE_HEIGHT * scale), resample=Image.Resampling.NEAREST)


def save_role_map_preview(
    role_map: np.ndarray,
    path: str | Path,
    *,
    scale: int = 8,
) -> None:
    """Save a role-map debug preview image."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    role_map_to_preview_image(role_map, scale=scale).save(output_path)


def _validate_role_inputs(palette: np.ndarray, index_map: np.ndarray, alpha: np.ndarray) -> None:
    if not isinstance(palette, np.ndarray) or palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette shape must be Kx3 RGB.")
    if palette.shape[0] < 2:
        raise ValueError("palette must include dummy transparent row plus one visible row.")
    if not isinstance(index_map, np.ndarray) or index_map.shape != SPRITE_SIZE:
        raise ValueError("index_map shape must be exactly 32x32.")
    if not isinstance(alpha, np.ndarray) or alpha.shape != SPRITE_SIZE:
        raise ValueError("alpha shape must be exactly 32x32.")


def _with_relative_flags(
    raw: list[dict[str, Any]],
    *,
    options: RoleInferenceOptions,
) -> dict[int, PaletteSlotRoleFeatures]:
    used = [entry for entry in raw if int(entry["pixel_count"]) > 0]
    l_values = [float(entry["oklab_l"]) for entry in used]
    chroma_values = [float(entry["chroma"]) for entry in used]
    min_l = min(l_values) if l_values else 0.0
    max_l = max(l_values) if l_values else 0.0
    l_range = max_l - min_l
    chroma_threshold = _quantile(chroma_values, 0.70) if chroma_values else 1.0

    features: dict[int, PaletteSlotRoleFeatures] = {}
    for entry in raw:
        pixel_count = int(entry["pixel_count"])
        frequency = float(entry["frequency"])
        oklab_l = float(entry["oklab_l"])
        chroma = float(entry["chroma"])
        if pixel_count == 0 or l_range < options.low_contrast_luminance_range:
            is_dark = False
            is_light = False
        else:
            is_dark = oklab_l <= min_l + 0.30 * l_range or oklab_l <= min_l + 0.25
            is_light = oklab_l >= min_l + 0.70 * l_range or oklab_l >= min_l + 0.75 * l_range

        features[int(entry["slot"])] = PaletteSlotRoleFeatures(
            slot=int(entry["slot"]),
            rgb=entry["rgb"],
            oklab_l=oklab_l,
            oklab_a=float(entry["oklab_a"]),
            oklab_b=float(entry["oklab_b"]),
            chroma=chroma,
            hue_degrees=float(entry["hue_degrees"]),
            pixel_count=pixel_count,
            frequency=frequency,
            edge_contact_count=int(entry["edge_contact_count"]),
            edge_contact_ratio=float(entry["edge_contact_ratio"]),
            transparent_neighbor_ratio=float(entry["transparent_neighbor_ratio"]),
            outline_neighbor_ratio=entry["outline_neighbor_ratio"],
            mean_x=float(entry["mean_x"]),
            mean_y=float(entry["mean_y"]),
            min_x=int(entry["min_x"]),
            min_y=int(entry["min_y"]),
            max_x=int(entry["max_x"]),
            max_y=int(entry["max_y"]),
            local_contrast_mean=float(entry["local_contrast_mean"]),
            local_same_color_ratio=float(entry["local_same_color_ratio"]),
            is_rare=frequency <= options.rare_color_frequency_threshold,
            is_dark=is_dark,
            is_light=is_light,
            is_high_chroma=bool(pixel_count > 0 and chroma >= chroma_threshold and chroma > 0.03),
        )
    return dict(sorted(features.items()))


def _tiny_palette_roles(
    features: dict[int, PaletteSlotRoleFeatures],
    *,
    options: RoleInferenceOptions,
) -> tuple[dict[int, int], dict[int, dict[str, float]]]:
    ordered = sorted(features.values(), key=lambda feature: (feature.oklab_l, feature.slot))
    roles: dict[int, int] = {}
    scores: dict[int, dict[str, float]] = {}

    if len(ordered) == 1:
        feature = ordered[0]
        if feature.is_dark and feature.edge_contact_ratio >= 0.45:
            role = ROLE_OUTLINE
        else:
            role = ROLE_MIDTONE
        roles[feature.slot] = role
        scores[feature.slot] = {role_name(role): 1.0}
        return roles, scores

    darkest = ordered[0]
    lightest = ordered[-1]
    if darkest.edge_contact_ratio >= 0.35 and (darkest.is_dark or darkest.oklab_l < 0.45):
        roles[darkest.slot] = ROLE_OUTLINE
    else:
        roles[darkest.slot] = ROLE_SHADOW
    scores[darkest.slot] = {role_name(roles[darkest.slot]): 1.0}

    if len(ordered) == 2:
        roles[lightest.slot] = ROLE_LIGHT if lightest.is_light else ROLE_MIDTONE
        scores[lightest.slot] = {role_name(roles[lightest.slot]): 1.0}
        return roles, scores

    middle = ordered[1]
    roles[middle.slot] = ROLE_MIDTONE
    scores[middle.slot] = {role_name(ROLE_MIDTONE): 1.0}
    if lightest.is_high_chroma and lightest.oklab_l >= options.emissive_luminance_threshold:
        roles[lightest.slot] = ROLE_EMISSIVE
    elif lightest.is_light and lightest.is_rare:
        roles[lightest.slot] = ROLE_HIGHLIGHT
    else:
        roles[lightest.slot] = ROLE_LIGHT
    scores[lightest.slot] = {role_name(roles[lightest.slot]): 1.0}
    return roles, scores


def _scored_roles(
    features: dict[int, PaletteSlotRoleFeatures],
    *,
    options: RoleInferenceOptions,
) -> tuple[dict[int, int], dict[int, dict[str, float]]]:
    l_values = [feature.oklab_l for feature in features.values()]
    chroma_values = [feature.chroma for feature in features.values()]
    contrast_values = [feature.local_contrast_mean for feature in features.values()]
    min_l = min(l_values)
    max_l = max(l_values)
    l_range = max(max_l - min_l, 1e-9)
    max_chroma = max(max(chroma_values), 1e-9)
    max_contrast = max(max(contrast_values), 1e-9)

    roles: dict[int, int] = {}
    debug_scores: dict[int, dict[str, float]] = {}
    for slot, feature in features.items():
        norm_l = 0.5 if l_range < options.low_contrast_luminance_range else (feature.oklab_l - min_l) / l_range
        darkness = 1.0 - norm_l
        lightness = norm_l
        frequency_strength = min(feature.frequency * 4.0, 1.0) if options.use_frequency else 0.5
        rarity = 1.0 - min(feature.frequency / max(options.rare_color_frequency_threshold, 1e-9), 1.0)
        edge = feature.edge_contact_ratio if options.use_edge_contact else 0.0
        chroma = feature.chroma / max_chroma if options.use_chroma else 0.0
        contrast = feature.local_contrast_mean / max_contrast
        same_color_cluster = feature.local_same_color_ratio
        not_edge = 1.0 - edge
        mid_luminance_score = max(0.0, 1.0 - abs(norm_l - 0.5) * 2.0)
        if options.use_spatial_bias and feature.mean_y >= 0:
            upper_spatial_bias = 1.0 - feature.mean_y / 31.0
            lower_spatial_bias = feature.mean_y / 31.0
        else:
            upper_spatial_bias = 0.5
            lower_spatial_bias = 0.5
        local_isolated_detail = 1.0 - same_color_cluster

        scores = {
            "outline": 0.45 * darkness + 0.35 * edge + 0.10 * frequency_strength + 0.10 * contrast,
            "deep_shadow": 0.55 * darkness + 0.20 * frequency_strength + 0.15 * same_color_cluster + 0.10 * lower_spatial_bias,
            "shadow": 0.45 * darkness + 0.25 * frequency_strength + 0.20 * same_color_cluster + 0.10 * lower_spatial_bias,
            "midtone": 0.45 * frequency_strength + 0.25 * mid_luminance_score + 0.20 * same_color_cluster + 0.10 * not_edge,
            "light": 0.50 * lightness + 0.20 * frequency_strength + 0.15 * same_color_cluster + 0.15 * upper_spatial_bias,
            "highlight": 0.45 * lightness + 0.25 * rarity + 0.15 * contrast + 0.15 * upper_spatial_bias,
            "accent": 0.45 * chroma + 0.30 * rarity + 0.15 * contrast + 0.10 * not_edge,
            "emissive": 0.40 * chroma + 0.35 * lightness + 0.15 * rarity + 0.10 * contrast,
            "texture_detail": 0.35 * rarity + 0.25 * contrast + 0.20 * not_edge + 0.20 * local_isolated_detail,
        }

        if not (feature.is_high_chroma and feature.oklab_l >= options.emissive_luminance_threshold and (feature.is_rare or feature.frequency <= 0.20)):
            scores["emissive"] = 0.0
        if not (feature.is_high_chroma and feature.is_rare):
            scores["accent"] *= 0.35
        if not (feature.is_light or feature.is_rare):
            scores["highlight"] *= 0.55
        if not feature.is_rare:
            scores["texture_detail"] *= 0.50

        best_name = max(scores, key=lambda name: (scores[name], feature.pixel_count, -feature.slot))
        roles[slot] = _role_id_for_name(best_name)
        debug_scores[slot] = scores

    outline_candidates = [
        feature
        for feature in features.values()
        if (feature.is_dark or feature.oklab_l <= min_l + 0.28) and feature.edge_contact_ratio >= 0.35
    ]
    if outline_candidates:
        outline = max(
            outline_candidates,
            key=lambda feature: (
                debug_scores[feature.slot]["outline"],
                feature.edge_contact_ratio,
                feature.pixel_count,
                -feature.slot,
            ),
        )
        roles[outline.slot] = ROLE_OUTLINE

    return roles, debug_scores


def _refine_role_map(
    role_map: np.ndarray,
    index_map: np.ndarray,
    alpha: np.ndarray,
    slot_roles: dict[int, int],
    features: dict[int, PaletteSlotRoleFeatures],
) -> np.ndarray:
    refined = np.asarray(role_map, dtype=np.uint8).copy()
    for y in range(SPRITE_HEIGHT):
        for x in range(SPRITE_WIDTH):
            if int(alpha[y, x]) == 0:
                continue
            slot = int(index_map[y, x])
            feature = features.get(slot)
            if feature is None:
                continue
            if (
                feature.is_dark
                and slot_roles.get(slot) in {ROLE_SHADOW, ROLE_DEEP_SHADOW, ROLE_MIDTONE}
                and _touches_transparency4(alpha, y, x)
            ):
                refined[y, x] = ROLE_OUTLINE
    return refined


def _hue_degrees(a_value: float, b_value: float, chroma: float) -> float:
    if chroma <= 1e-9:
        return 0.0
    return (math.degrees(math.atan2(b_value, a_value)) + 360.0) % 360.0


def _neighbors4_with_outside(y: int, x: int) -> list[tuple[int, int]]:
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def _touches_transparency4(alpha: np.ndarray, y: int, x: int) -> bool:
    for ny, nx in _neighbors4_with_outside(y, x):
        if ny < 0 or nx < 0 or ny >= SPRITE_HEIGHT or nx >= SPRITE_WIDTH:
            return True
        if int(alpha[ny, nx]) == 0:
            return True
    return False


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def _role_id_for_name(name: str) -> int:
    mapping = {
        "outline": ROLE_OUTLINE,
        "deep_shadow": ROLE_DEEP_SHADOW,
        "shadow": ROLE_SHADOW,
        "midtone": ROLE_MIDTONE,
        "light": ROLE_LIGHT,
        "highlight": ROLE_HIGHLIGHT,
        "accent": ROLE_ACCENT,
        "emissive": ROLE_EMISSIVE,
        "texture_detail": ROLE_TEXTURE_DETAIL,
    }
    return mapping.get(name, ROLE_UNKNOWN)


def _confidence_for_scores(scores: dict[str, float], role: int) -> float:
    if not scores:
        return 0.0
    role_key = role_name(role)
    best = float(scores.get(role_key, max(scores.values())))
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, best / total * 2.0))


def _metadata_with_role_inference(metadata: SpriteMetadata, *, preserved: bool) -> SpriteMetadata:
    metadata_data = copy.deepcopy(metadata.to_dict())
    extra = dict(metadata_data.get("extra") or {})
    extra["role_inference"] = {
        "version": ROLE_INFERENCE_VERSION,
        "preserved_existing_role_map": preserved,
    }
    metadata_data["extra"] = extra
    return SpriteMetadata.from_dict(metadata_data)


def _copy_bundle_with_metadata(bundle: SpriteBundle, metadata: SpriteMetadata) -> SpriteBundle:
    return SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=np.asarray(bundle.palette).copy(),
        index_map=np.asarray(bundle.index_map).copy(),
        role_map=None if bundle.role_map is None else np.asarray(bundle.role_map).copy(),
        metadata=metadata,
    )

"""Conservative conditional-adherence adapters for trusted benchmark fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

FIELDS = ("category", "canonical_object", "color", "shape", "material", "role", "palette", "transparent_background")

_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (210, 55, 50),
    "orange": (225, 125, 40),
    "yellow": (225, 205, 55),
    "green": (65, 165, 75),
    "blue": (55, 105, 205),
    "purple": (135, 75, 175),
    "pink": (220, 125, 170),
    "brown": (125, 80, 45),
}


def score_conditions(record: Mapping[str, Any], rgba: np.ndarray, palette_adherence: float | None) -> dict[str, str]:
    """Return represented/omitted/contradicted/unscorable per field.

    Semantic fields require reviewed predictions in ``condition_predictions``. Color,
    palette and transparency have deterministic visual adapters.
    """
    predictions = (
        record.get("condition_predictions") if isinstance(record.get("condition_predictions"), Mapping) else {}
    )
    nested = record.get("conditions") if isinstance(record.get("conditions"), Mapping) else {}
    conditions = {**record, **nested}
    result: dict[str, str] = {}
    for field in FIELDS:
        expected = _expected(conditions, field)
        if expected in (None, "", [], {}):
            result[field] = "unscorable"
            continue
        if field == "transparent_background":
            has_transparency = bool(np.any(rgba[..., 3] == 0))
            result[field] = "represented" if bool(expected) == has_transparency else "contradicted"
        elif field == "palette":
            if palette_adherence is None:
                result[field] = "unscorable"
            elif palette_adherence >= 0.9:
                result[field] = "represented"
            elif palette_adherence <= 0.25:
                result[field] = "contradicted"
            else:
                result[field] = "omitted"
        elif field == "color":
            result[field] = _score_color(expected, rgba)
        elif field in predictions:
            result[field] = _compare(expected, predictions[field])
        else:
            result[field] = "unscorable"
    return result


def _expected(values: Mapping[str, Any], field: str) -> Any:
    aliases = {
        "canonical_object": ("canonical_object", "base_object", "object_name"),
        "color": ("color", "colors"),
        "shape": ("shape", "shapes"),
        "material": ("material", "materials"),
        "palette": ("palette", "target_palette", "palette_condition"),
    }
    for key in aliases.get(field, (field,)):
        if key in values:
            return values[key]
    return None


def _compare(expected: Any, predicted: Any) -> str:
    exp = {str(v).lower() for v in (expected if isinstance(expected, list) else [expected])}
    got = {str(v).lower() for v in (predicted if isinstance(predicted, list) else [predicted]) if v not in (None, "")}
    if not got:
        return "omitted"
    return "represented" if exp & got else "contradicted"


def _score_color(expected: Any, rgba: np.ndarray) -> str:
    names = [str(value).lower() for value in (expected if isinstance(expected, list) else [expected])]
    targets = np.asarray([_RGB[name] for name in names if name in _RGB], dtype=np.float32)
    visible = rgba[..., :3][rgba[..., 3] > 0].astype(np.float32)
    if not len(targets) or not len(visible):
        return "unscorable"
    distances = np.sqrt(np.sum((visible[:, None, :] - targets[None, :, :]) ** 2, axis=2)).min(axis=1)
    share = float(np.mean(distances <= 85.0))
    if share >= 0.20:
        return "represented"
    if share <= 0.02:
        return "contradicted"
    return "omitted"

"""Metadata-free visual evidence and request construction for Dataset-v5."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from spritelab.dataset_v5.identity import (
    assert_opaque_id,
    canonical_json_bytes,
    decoded_rgba_sha256,
    make_geometry_family_id,
)

BLIND_REQUEST_SCHEMA_VERSION = "blind_semantic_request_v1"
BLIND_OUTPUT_SCHEMA_VERSION = "blind_semantic_output_v1"
BLIND_PROMPT_VERSION = "blind_visual_taxonomy_v1"
SOL_ADJUDICATION_PROMPT_VERSION = "sol_blind_adjudication_v1"
SOL_CONSISTENCY_PROMPT_VERSION = "sol_blind_consistency_v1"
PIXEL_FACTS_VERSION = "deterministic_pixel_facts_v1"

PASS_FIELD_ORDER = {
    "adjudication": (
        "category",
        "canonical_object",
        "domain",
        "role",
        "visual_form",
        "material_applicability",
        "explicit_material",
        "color_roles",
        "description",
    ),
    "consistency": (
        "visual_form",
        "role",
        "description",
        "domain",
        "color_roles",
        "canonical_object",
        "material_applicability",
        "category",
        "explicit_material",
    ),
}

DEFAULT_TAXONOMY: dict[str, list[str]] = {
    "categories": [
        "armor",
        "clothing",
        "container",
        "food",
        "gem",
        "key",
        "mineral",
        "plant",
        "potion",
        "shield",
        "tool",
        "weapon",
        "unknown",
        "oov",
    ],
    "domains": [
        "equipment_icon",
        "food_icon",
        "inventory_icon",
        "plant_icon",
        "resource_icon",
        "unknown",
        "oov",
    ],
    "roles": [
        "consumable",
        "crafting_resource",
        "decorative_item",
        "functional_tool",
        "key_item",
        "protective_equipment",
        "unknown",
        "oov",
        "weapon",
        "wearable_equipment",
    ],
    "material_applicability": ["applicable", "not_applicable", "unknown"],
}

FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "artist",
        "author",
        "creator",
        "current_canonical_object",
        "current_category",
        "current_description",
        "directory_path",
        "download_url",
        "existing_semantic_proposals",
        "filename",
        "local_path",
        "member_path",
        "old_labels",
        "original_archive_member",
        "original_filename",
        "original_source_filename",
        "pack",
        "pack_name",
        "path",
        "renamed_filename",
        "source_creator",
        "source_description",
        "source_metadata",
        "source_pack",
        "source_page_title",
        "source_url",
        "sprite_id",
        "sprite_name",
    }
)


class BlindPayloadLeakageError(ValueError):
    """Raised when provenance or semantic naming enters a blind request."""


@dataclass(frozen=True)
class BlindInput:
    """The only record data allowed to cross the blind-provider boundary."""

    record_id: str
    rgba: np.ndarray
    pixel_facts: Mapping[str, Any]
    taxonomy: Mapping[str, list[str]]

    @classmethod
    def from_rgba(
        cls,
        record_id: str,
        rgba: np.ndarray,
        *,
        taxonomy: Mapping[str, list[str]] | None = None,
    ) -> BlindInput:
        assert_opaque_id(record_id)
        value = _checked_rgba(rgba)
        return cls(
            record_id=record_id,
            rgba=value,
            pixel_facts=deterministic_pixel_facts(value),
            taxonomy=json.loads(json.dumps(taxonomy or DEFAULT_TAXONOMY)),
        )


def deterministic_pixel_facts(rgba: np.ndarray) -> dict[str, Any]:
    """Compute authoritative facts that make no semantic object claims."""

    value = _checked_rgba(rgba)
    height, width = value.shape[:2]
    alpha = value[:, :, 3]
    visible = alpha > 0
    ys, xs = np.nonzero(visible)
    bbox = None
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    colors = Counter(map(tuple, value[visible].tolist()))
    palette = [
        {"count": count, "rgba": list(color)}
        for color, count in sorted(colors.items(), key=lambda item: (-item[1], item[0]))
    ]
    horizontal = bool(np.array_equal(visible, np.fliplr(visible)))
    vertical = bool(np.array_equal(visible, np.flipud(visible)))
    alpha_values = sorted(int(item) for item in np.unique(alpha))
    tight_dimensions = [0, 0] if bbox is None else [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return {
        "alpha_bounds": bbox,
        "alpha_levels": alpha_values,
        "blob_id": decoded_rgba_sha256(value),
        "connected_components_4": _connected_components(visible),
        "decoded_dimensions": [width, height],
        "geometry_family_id": make_geometry_family_id(alpha),
        "occupancy": round(float(visible.mean()), 12),
        "opaque_pixels": int(np.count_nonzero(alpha == 255)),
        "palette_rgba": palette,
        "palette_size": len(palette),
        "pixel_art_structure": {
            "fully_binary_alpha": set(alpha_values).issubset({0, 255}),
            "integer_pixel_grid": True,
            "stored_without_upscale": True,
        },
        "schema_version": PIXEL_FACTS_VERSION,
        "symmetry": {"horizontal": horizontal, "vertical": vertical},
        "tight_dimensions": tight_dimensions,
        "visible_pixels": int(visible.sum()),
    }


def blind_output_schema(field_order: Iterable[str]) -> dict[str, Any]:
    """Build the strict semantic output schema in the requested field order."""

    field_definitions = {
        "category": {"type": ["string", "null"]},
        "canonical_object": {"type": ["string", "null"]},
        "domain": {"type": ["string", "null"]},
        "role": {"type": ["string", "null"]},
        "visual_form": {"type": ["array", "null"], "items": {"type": "string"}},
        "material_applicability": {
            "type": "string",
            "enum": ["applicable", "not_applicable", "unknown"],
        },
        "explicit_material": {"type": ["string", "null"]},
        "color_roles": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "highlight": {"type": "array", "items": {"type": "string"}},
                "outline": {"type": "array", "items": {"type": "string"}},
                "primary": {"type": "array", "items": {"type": "string"}},
                "secondary": {"type": "array", "items": {"type": "string"}},
                "shadow": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["primary", "secondary", "outline", "shadow", "highlight"],
        },
        "description": {"type": ["string", "null"]},
    }
    ordered = list(field_order)
    if set(ordered) != set(field_definitions):
        raise ValueError("field_order must contain every semantic field exactly once")
    properties: dict[str, Any] = {name: field_definitions[name] for name in ordered}
    properties.update(
        {
            "abstentions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "field_rationales": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "field_risk_signals": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "field_confidence": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "schema_version": {"const": BLIND_OUTPUT_SCHEMA_VERSION},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": [
            *ordered,
            "abstentions",
            "field_rationales",
            "field_risk_signals",
            "field_confidence",
            "schema_version",
        ],
        "type": "object",
    }


def build_blind_request(
    blind_input: BlindInput,
    *,
    model: str,
    request_id: str,
    pass_kind: str,
) -> dict[str, Any]:
    """Construct one blind request without accepting provenance parameters."""

    if pass_kind not in PASS_FIELD_ORDER:
        raise ValueError(f"unknown pass_kind: {pass_kind}")
    assert_opaque_id(blind_input.record_id)
    rgba = _checked_rgba(blind_input.rgba)
    facts = json.loads(canonical_json_bytes(dict(blind_input.pixel_facts)).decode("utf-8"))
    if facts.get("blob_id") != decoded_rgba_sha256(rgba):
        raise ValueError("pixel_facts blob_id does not bind the supplied image")
    image_bytes = deterministic_png_bytes(rgba)
    schema = blind_output_schema(PASS_FIELD_ORDER[pass_kind])
    prompt_version = SOL_ADJUDICATION_PROMPT_VERSION if pass_kind == "adjudication" else SOL_CONSISTENCY_PROMPT_VERSION
    role_text = (
        "You are the final blind visual taxonomy adjudicator. Judge only visible pixels and supplied deterministic facts."
        if pass_kind == "adjudication"
        else "You are a blind consistency examiner using a different field order. Re-evaluate the image from scratch."
    )
    instructions = {
        "abstention": (
            "Do not guess. Use null for unsupported nullable fields and explain each abstention. "
            "Use unknown and OOV only as distinct taxonomy states."
        ),
        "field_definitions": _field_definitions(),
        "record_id": blind_input.record_id,
        "taxonomy": blind_input.taxonomy,
    }
    payload = {
        "model": model,
        "request_id": request_id,
        "request_schema_version": BLIND_REQUEST_SCHEMA_VERSION,
        "messages": [
            {"role": "system", "content": role_text},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
                    },
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"deterministic_pixel_facts": facts},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
        ],
        "metadata": {
            "pass_kind": pass_kind,
            "prompt_version": prompt_version,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"sprite_lab_{pass_kind}",
                "schema": schema,
                "strict": True,
            },
        },
    }
    audit_blind_payload(payload)
    return payload


def blind_cache_key(payload: Mapping[str, Any], *, endpoint_identity: str, provider: str) -> str:
    """Hash semantic inputs while deliberately excluding opaque request IDs."""

    identity = json.loads(canonical_json_bytes(dict(payload)).decode("utf-8"))
    identity.pop("request_id", None)
    envelope = {
        "endpoint_identity": endpoint_identity,
        "payload": identity,
        "provider": provider,
        "version": "blind_label_cache_v1",
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def audit_blind_payload(
    payload: Mapping[str, Any],
    *,
    forbidden_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed if forbidden keys or exact provenance values enter a request."""

    findings: list[dict[str, str]] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).casefold()
                if normalized in FORBIDDEN_BLIND_KEYS:
                    findings.append({"location": f"{location}.{key}", "reason": "forbidden_key"})
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")

    walk(payload, "$")
    serialized = canonical_json_bytes(payload).decode("utf-8").casefold()
    if forbidden_metadata:
        for key, value in _scalar_metadata(forbidden_metadata):
            text = str(value).strip()
            if len(text) >= 4 and text.casefold() in serialized:
                findings.append({"location": key, "reason": "forbidden_metadata_value"})
    if findings:
        raise BlindPayloadLeakageError(json.dumps(findings, sort_keys=True))
    return {"ok": True, "filename_leakage": 0, "findings": []}


def requests_equal_except_id(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = json.loads(canonical_json_bytes(left).decode("utf-8"))
    b = json.loads(canonical_json_bytes(right).decode("utf-8"))
    a.pop("request_id", None)
    b.pop("request_id", None)
    return a == b


def deterministic_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode a transport PNG deterministically without resizing pixels."""

    value = _checked_rgba(rgba)
    output = io.BytesIO()
    Image.fromarray(value, mode="RGBA").save(
        output,
        format="PNG",
        compress_level=9,
        optimize=False,
        pnginfo=None,
    )
    return output.getvalue()


def _checked_rgba(rgba: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(rgba, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 4:
        raise ValueError(f"expected RGBA array [height,width,4], got {value.shape}")
    return value


def _connected_components(mask: np.ndarray) -> int:
    pending = np.asarray(mask, dtype=bool).copy()
    height, width = pending.shape
    components = 0
    for y in range(height):
        for x in range(width):
            if not pending[y, x]:
                continue
            components += 1
            pending[y, x] = False
            queue: deque[tuple[int, int]] = deque([(x, y)])
            while queue:
                px, py = queue.popleft()
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < width and 0 <= ny < height and pending[ny, nx]:
                        pending[ny, nx] = False
                        queue.append((nx, ny))
    return components


def _field_definitions() -> dict[str, str]:
    return {
        "canonical_object": "Most specific visually supported object identity; null when ambiguous.",
        "category": "Controlled broad visual category.",
        "color_roles": "Visible color-role descriptions grounded in supplied palette facts.",
        "description": "Short visual description containing no unsupported identity or material claim.",
        "domain": "Controlled icon-use domain inferred only when visually supportable.",
        "explicit_material": "Exact material only when distinctive visual evidence supports it; otherwise null.",
        "material_applicability": "Whether an explicit material label is meaningful for this object.",
        "role": "Controlled functional role; null/unknown when function is not visually justified.",
        "visual_form": "Non-semantic visible geometry and parts.",
    }


def _scalar_metadata(value: Mapping[str, Any], prefix: str = "$") -> Iterable[tuple[str, Any]]:
    for key, child in value.items():
        location = f"{prefix}.{key}"
        if isinstance(child, Mapping):
            yield from _scalar_metadata(child, location)
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if not isinstance(item, (Mapping, list)) and item is not None:
                    yield f"{location}[{index}]", item
        elif child is not None:
            yield location, child

"""Codex-native, metadata-blind visual labeling for Dataset-v5.

The module deliberately separates three trust zones:

* source manifests, which may contain tainted provenance metadata;
* blind staging, which contains only content-bound identifiers and pixels;
* frozen label outputs, which may be reconciled with provenance only later.

No provider client or network API is used here.  The generated contact sheets
are intended for local inspection by a Codex session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spritelab.dataset_v5.blind import deterministic_pixel_facts
from spritelab.dataset_v5.identity import (
    BLOB_ID_VERSION,
    assert_opaque_id,
    canonical_json_bytes,
    decoded_rgba_sha256,
)

SCHEMA_VERSION = "sprite_lab_codex_blind_label_v1"
PROMPT_VERSION = "codex_blind_prompt_v1"
CAMPAIGN_VERSION = "sprite_lab_codex_blind_campaign_v1"
EVIDENCE_ORIGIN = "codex_visual_inspection"
SHARD_SIZE = 25
CONTACT_SHEET_SIDE = 4
DISPLAY_SIZE = 128
CRITICAL_FIELDS = ("canonical_object", "category", "domain", "role")
EXACT_MATERIAL_TERMS = frozenset(
    {
        "bronze",
        "copper",
        "diamond",
        "emerald",
        "gold",
        "iron",
        "ruby",
        "sapphire",
        "silver",
        "steel",
    }
)

FIELD_NAMES = (
    "domain",
    "category",
    "canonical_object",
    "surface_alias",
    "role",
    "visual_form",
    "explicit_material",
    "visual_material_cue",
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "description",
)
PASS_FIELD_ORDER = {
    "A": FIELD_NAMES,
    "B": (
        "visual_form",
        "outline_colors",
        "role",
        "surface_alias",
        "description",
        "domain",
        "visual_material_cue",
        "canonical_object",
        "secondary_colors",
        "category",
        "explicit_material",
        "primary_colors",
    ),
}
FIELD_STATES = ("known", "model_abstained", "not_applicable", "unsupported")
CONFIDENCE_LEVELS = ("low", "medium", "high")

DOMAINS = (
    "inventory_icon",
    "equipment_icon",
    "resource_icon",
    "food_icon",
    "plant_icon",
    "spell_icon",
    "unknown",
)
CATEGORIES = (
    "weapon",
    "armor",
    "tool",
    "key",
    "gem",
    "material",
    "plant",
    "food",
    "potion",
    "jewelry",
    "clothing",
    "container",
    "spell",
    "misc_item",
    "unknown",
)
CANONICAL_OBJECTS = (
    "amulet",
    "apple",
    "arrow",
    "axe",
    "bag",
    "belt",
    "berry",
    "bone",
    "book",
    "boots",
    "bottle",
    "bow",
    "bracelet",
    "bread",
    "bucket",
    "carrot",
    "cheese",
    "chest",
    "chest armor",
    "cloak",
    "club",
    "coin",
    "crystal",
    "cut gemstone",
    "dagger",
    "egg",
    "feather",
    "fish",
    "flame",
    "flower",
    "gloves",
    "hammer",
    "hat",
    "helmet",
    "herb",
    "hoe",
    "ingot",
    "key",
    "lantern",
    "leaf",
    "lightning bolt",
    "magic orb",
    "map",
    "meat",
    "mushroom",
    "necklace",
    "ore chunk",
    "pants",
    "pickaxe",
    "plant sprig",
    "potion bottle",
    "pouch",
    "ring",
    "robe",
    "rope",
    "rune",
    "scroll",
    "seed",
    "shell",
    "shield",
    "shirt",
    "shoes",
    "shovel",
    "sickle",
    "spear",
    "spell book",
    "staff",
    "sword",
    "torch",
    "tree branch",
    "wand",
)
ROLES = (
    "access_token",
    "combat_weapon",
    "consumable_food",
    "consumable_potion",
    "crafting_resource",
    "crafting_tool",
    "cutting_tool",
    "decorative_item",
    "mining_tool",
    "misc_item",
    "protective_equipment",
    "spell_effect",
    "storage_container",
    "wearable_accessory",
    "wearable_clothing",
    "unknown",
)
VISUAL_MATERIAL_CUES = (
    "metallic",
    "wooden",
    "stone-like",
    "crystalline",
    "fabric-like",
    "organic",
    "liquid",
    "unknown",
)
COLOR_TERMS = (
    "black",
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "magenta",
    "orange",
    "pink",
    "purple",
    "red",
    "tan",
    "teal",
    "white",
    "yellow",
)

FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "archive_member_path",
        "artist",
        "author",
        "creator",
        "creator_or_publishers",
        "description_old",
        "directory_path",
        "distribution_platform",
        "distribution_platforms",
        "download_url",
        "download_urls",
        "filename",
        "local_path",
        "member_path",
        "normalized_filename",
        "old_description",
        "old_labels",
        "original_archive_path",
        "original_archive_paths",
        "original_filename",
        "pack",
        "packs",
        "path",
        "platform",
        "source_bindings",
        "source_creator",
        "source_description",
        "source_metadata",
        "source_pack",
        "source_page_title",
        "source_url",
        "source_urls",
        "sprite_id",
        "sprite_name",
        "vlm_proposals",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{3,}")


class CodexBlindError(RuntimeError):
    """Base error for a fail-closed campaign operation."""


class ForbiddenMetadataError(CodexBlindError):
    """Raised before inspection when a blind artifact contains tainted metadata."""


def taxonomy() -> dict[str, list[str]]:
    return {
        "canonical_object": list(CANONICAL_OBJECTS),
        "category": list(CATEGORIES),
        "domain": list(DOMAINS),
        "role": list(ROLES),
        "visual_material_cue": list(VISUAL_MATERIAL_CUES),
    }


def output_schema() -> dict[str, Any]:
    field_value_schemas: dict[str, dict[str, Any]] = {
        "domain": {"type": ["string", "null"], "enum": [*DOMAINS, None]},
        "category": {"type": ["string", "null"], "enum": [*CATEGORIES, None]},
        "canonical_object": {"type": ["string", "null"], "enum": [*CANONICAL_OBJECTS, None]},
        "surface_alias": {"type": ["string", "null"]},
        "role": {"type": ["string", "null"], "enum": [*ROLES, None]},
        "visual_form": {"type": ["string", "null"]},
        "explicit_material": {"type": ["string", "null"]},
        "visual_material_cue": {
            "type": ["string", "null"],
            "enum": [*VISUAL_MATERIAL_CUES, None],
        },
        "primary_colors": {"type": ["array", "null"], "items": {"enum": list(COLOR_TERMS)}},
        "secondary_colors": {"type": ["array", "null"], "items": {"enum": list(COLOR_TERMS)}},
        "outline_colors": {"type": ["array", "null"], "items": {"enum": list(COLOR_TERMS)}},
        "description": {"type": ["string", "null"]},
    }
    fields: dict[str, Any] = {}
    for name in FIELD_NAMES:
        fields[name] = {
            "additionalProperties": False,
            "properties": {
                "value": field_value_schemas[name],
                "state": {"enum": list(FIELD_STATES)},
                "confidence": {"enum": list(CONFIDENCE_LEVELS)},
                "visual_evidence": {"type": "string", "minLength": 1},
            },
            "required": ["value", "state", "confidence", "visual_evidence"],
            "type": "object",
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "record_id": {"pattern": r"^rec_[0-9a-f]{64}$", "type": "string"},
            "image_sha256": {"pattern": r"^[0-9a-f]{64}$", "type": "string"},
            "pass_id": {"enum": ["A", "B"]},
            "fields": {
                "additionalProperties": False,
                "properties": fields,
                "required": list(FIELD_NAMES),
                "type": "object",
            },
            "overall_visual_certainty": {"enum": list(CONFIDENCE_LEVELS)},
            "needs_individual_inspection": {"type": "boolean"},
            "quality_flags": {"type": "array", "items": {"type": "string"}},
            "evidence_origin": {"const": EVIDENCE_ORIGIN},
            "labeler": {
                "additionalProperties": False,
                "properties": {
                    "surface": {"const": "codex"},
                    "model_display": {"type": "string", "minLength": 1},
                    "session_id": {"type": "string", "minLength": 1},
                    "prompt_version": {"const": PROMPT_VERSION},
                },
                "required": ["surface", "model_display", "session_id", "prompt_version"],
                "type": "object",
            },
        },
        "required": [
            "schema_version",
            "record_id",
            "image_sha256",
            "pass_id",
            "fields",
            "overall_visual_certainty",
            "needs_individual_inspection",
            "quality_flags",
            "evidence_origin",
            "labeler",
        ],
        "type": "object",
    }


def blind_prompt() -> str:
    return """# Codex blind visual labeling prompt v1

Inspect only the opaque record ID, image, dimensions, deterministic pixel facts,
controlled taxonomy, and these field definitions. Never seek or use filenames,
paths, creators, packs, platforms, source pages, prior labels, descriptions, or
other semantic proposals. Source eligibility is out of scope and cannot be
changed by a semantic label.

Return one schema-valid JSON object per record. Ground every field in visible
pixels. Abstain rather than guessing. `canonical_object` and `role` must use the
controlled taxonomy. `surface_alias` is a concise visual identity, or null.
`visual_form` is a literal shape/form description. `explicit_material` is null
unless an exact material is visually or deterministically justified. Color
alone never justifies iron, bronze, gold, silver, steel, ruby, emerald, or any
other exact material. Use `visual_material_cue` for metallic, wooden, stone-like,
crystalline, fabric-like, organic, liquid, or unknown appearance.

Color fields contain broad visible color terms. The description is one grounded
sentence with no story, power, game mechanic, source attribution, or use beyond
visible evidence. Low-confidence critical fields, ambiguous objects, uncertain
categories, and suspected multipart content require individual inspection.

Field states are `known`, `model_abstained`, `not_applicable`, and `unsupported`.
Confidence values are `low`, `medium`, and `high`. Every field needs concise
visual evidence. These labels are weak Codex proposals, never human ground truth.
"""


def forbidden_metadata_policy() -> dict[str, Any]:
    return {
        "allowed_input_classes": [
            "opaque_content_bound_record_id",
            "image_pixels",
            "dimensions",
            "deterministic_pixel_facts",
            "controlled_taxonomy",
            "field_definitions",
            "abstention_instructions",
        ],
        "forbidden_keys": sorted(FORBIDDEN_METADATA_KEYS),
        "image_filename_pattern": r"^rec_[0-9a-f]{64}\.png$",
        "leakage_action": "stop_before_visual_inspection",
        "prompt_version": PROMPT_VERSION,
        "schema_version": "sprite_lab_codex_forbidden_metadata_policy_v1",
    }


def stage_campaign(
    raw_experiment: str | Path,
    output_root: str | Path,
    *,
    model_display: str,
    session_id: str,
) -> dict[str, Any]:
    """Build blind images and non-semantic manifests from verified raw evidence."""

    raw_root = Path(raw_experiment).resolve()
    output = Path(output_root).resolve()
    extraction_path = raw_root / "extraction_manifest.jsonl"
    blob_manifest_path = raw_root / "blob_manifest.jsonl"
    provenance_path = raw_root / "provenance_manifest.jsonl"
    suitability_path = raw_root / "suitability_manifest.jsonl"
    for required in (extraction_path, blob_manifest_path, provenance_path, suitability_path):
        if not required.is_file():
            raise CodexBlindError(f"required verified raw artifact is missing: {required}")

    if output.exists() and any(output.iterdir()):
        raise CodexBlindError(f"refusing to overwrite non-empty campaign root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "blind_images",
        "original_blobs",
        "contact_sheets_pass_a",
        "contact_sheets_pass_b",
        "pass_a",
        "pass_b",
        "health_checks",
    ):
        (output / directory).mkdir()

    blob_rows = {str(row["blob_id"]): row for row in _read_jsonl(blob_manifest_path)}
    extraction_rows = _read_jsonl(extraction_path)
    provenance_rows = {str(row.get("record_id")): row for row in _read_jsonl(provenance_path)}
    suitability_rows = {str(row.get("record_id")): row for row in _read_jsonl(suitability_path)}

    blind_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    forbidden_fingerprints: set[str] = set()
    for row in sorted(extraction_rows, key=lambda item: str(item.get("record_id") or "")):
        record_id = row.get("record_id")
        if row.get("decode_status") != "verified_from_original":
            failures.append(
                {
                    "failure_reason": "not_readable_or_unresolved",
                    "record_id": record_id,
                    "schema_version": "sprite_lab_codex_blind_failure_v1",
                }
            )
            continue
        if not isinstance(record_id, str):
            raise CodexBlindError("decoded extraction row has no opaque record_id")
        assert_opaque_id(record_id)
        if row.get("crop_coordinates") is None or row.get("interpolation_policy") != "none":
            failures.append(
                {
                    "failure_reason": "ambiguous_or_interpolated_operation",
                    "record_id": record_id,
                    "schema_version": "sprite_lab_codex_blind_failure_v1",
                }
            )
            continue
        blob_id = str(row.get("blob_id") or "")
        blob_row = blob_rows.get(blob_id)
        if blob_row is None:
            failures.append(
                {
                    "failure_reason": "missing_image_blob",
                    "record_id": record_id,
                    "schema_version": "sprite_lab_codex_blind_failure_v1",
                }
            )
            continue
        source_blob = raw_root / str(blob_row["blob_path"])
        payload = source_blob.read_bytes()
        if hashlib.sha256(payload).hexdigest() != blob_row["blob_file_sha256"]:
            raise CodexBlindError(f"blob file hash mismatch for opaque record {record_id}")
        width = int(row["width"])
        height = int(row["height"])
        rgba = _decode_canonical_rgba(payload, width=width, height=height)
        observed_image_hash = decoded_rgba_sha256(rgba)
        if observed_image_hash != blob_id:
            raise CodexBlindError(f"decoded image hash mismatch for opaque record {record_id}")

        original_blob_path = output / "original_blobs" / f"{record_id}.rgba"
        original_blob_path.write_bytes(payload)
        rendered = render_blind_image(rgba)
        image_path = output / "blind_images" / f"{record_id}.png"
        rendered.save(image_path, format="PNG", compress_level=9, optimize=False, pnginfo=None)
        facts = row.get("deterministic_pixel_facts")
        if not isinstance(facts, Mapping) or facts.get("blob_id") != observed_image_hash:
            facts = deterministic_pixel_facts(rgba)
        blind_rows.append(
            {
                "deterministic_pixel_facts": facts,
                "dimensions": [width, height],
                "image_sha256": observed_image_hash,
                "record_id": record_id,
                "schema_version": "sprite_lab_codex_blind_input_v1",
            }
        )
        provenance = provenance_rows.get(record_id, {})
        suitability = suitability_rows.get(record_id, {})
        eligibility_rows.append(
            {
                "forensic_inclusion_decision": row.get("forensic_inclusion_decision"),
                "provenance_status": provenance.get("provenance_status"),
                "record_id": record_id,
                "schema_version": "sprite_lab_codex_blind_eligibility_v1",
                "suitability_status": suitability.get("audit_status"),
            }
        )
        forbidden_fingerprints.update(_metadata_token_fingerprints(provenance))

    if not blind_rows:
        raise CodexBlindError("no readable records were staged")
    _assert_unique(str(row["record_id"]) for row in blind_rows)
    _write_jsonl(output / "blind_manifest.jsonl", blind_rows)
    _write_jsonl(output / "eligibility_manifest.jsonl", eligibility_rows)
    _write_jsonl(output / "failures.jsonl", failures)
    _write_json(output / "output_schema.json", output_schema())
    _write_json(output / "forbidden_metadata_policy.json", forbidden_metadata_policy())
    (output / "blind_prompt.md").write_text(blind_prompt(), encoding="utf-8", newline="\n")
    _write_json(
        output / "forbidden_metadata_fingerprints.json",
        {
            "algorithm": "sha256(normalized_metadata_token)",
            "fingerprints": sorted(forbidden_fingerprints),
            "schema_version": "sprite_lab_forbidden_metadata_fingerprints_v1",
        },
    )
    prompt_audit = audit_blind_value(
        {
            "field_order": list(PASS_FIELD_ORDER["A"]),
            "instructions": blind_prompt(),
            "output_schema": output_schema(),
            "taxonomy": taxonomy(),
        },
        forbidden_fingerprints=forbidden_fingerprints,
    )
    source_hashes = {
        path.name: _file_sha256(path)
        for path in (extraction_path, blob_manifest_path, provenance_path, suitability_path)
    }
    progress = {
        "campaign_status": "staged",
        "campaign_version": CAMPAIGN_VERSION,
        "health_gate_status": "not_run",
        "labeler": {
            "model_display": model_display,
            "prompt_version": PROMPT_VERSION,
            "session_id": session_id,
            "surface": "codex",
        },
        "next_unprocessed": {"A": blind_rows[0]["record_id"], "B": None},
        "pass_a_completed": 0,
        "pass_b_completed": 0,
        "readable_records": len(blind_rows),
        "source_artifact_hashes": source_hashes,
        "staging_prompt_audit": prompt_audit,
        "usage_limit_stop": False,
    }
    _write_json(output / "progress.json", progress)
    for name in (
        "reconciled_labels.jsonl",
        "supervision_candidates.jsonl",
        "conflicts.jsonl",
        "abstentions.jsonl",
        "source_reconciliation.jsonl",
    ):
        (output / name).write_text("", encoding="utf-8", newline="\n")
    (output / "command_log.txt").write_text(
        "stage: python -m spritelab.dataset_v5.codex_blind stage --raw-experiment <verified_raw> "
        "--output <campaign> --model-display <displayed_model> --session-id <codex_thread_id>\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / ".gitignore").write_text(
        "blind_images/\nblind_manifest.jsonl\noriginal_blobs/\n"
        "contact_sheets_pass_a/\ncontact_sheets_pass_b/\n"
        "pass_a/**/inputs/\npass_b/**/inputs/\n"
        "pass_a/**/batch_payload.json\npass_b/**/batch_payload.json\n"
        "health_checks/**/inputs/\nhealth_checks/**/sheet_*.png\n"
        "health_checks/**/batch_payload.json\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "decode_failures": len(failures),
        "prompt_audit": prompt_audit,
        "readable_records": len(blind_rows),
        "status": "staged",
    }


def deterministic_shards(rows: Sequence[Mapping[str, Any]], pass_id: str) -> list[list[dict[str, Any]]]:
    """Return stable disjoint shards, with a separate deterministic Pass-B shuffle."""

    if pass_id not in PASS_FIELD_ORDER:
        raise CodexBlindError(f"invalid pass_id: {pass_id!r}")
    normalized = [dict(row) for row in rows]
    _assert_unique(str(row["record_id"]) for row in normalized)
    if pass_id == "A":
        normalized.sort(key=lambda row: str(row["record_id"]))
    else:
        normalized.sort(
            key=lambda row: hashlib.sha256(
                b"codex_blind_pass_b_shuffle_v1\0" + str(row["record_id"]).encode("ascii")
            ).digest()
        )
    return [normalized[index : index + SHARD_SIZE] for index in range(0, len(normalized), SHARD_SIZE)]


def prepare_pass(output_root: str | Path, pass_id: str) -> dict[str, Any]:
    """Create audited shard inputs and 4x4 sheets without reading another pass."""

    output = Path(output_root).resolve()
    rows = _read_jsonl(output / "blind_manifest.jsonl")
    fingerprints = set(_read_json(output / "forbidden_metadata_fingerprints.json")["fingerprints"])
    shards = deterministic_shards(rows, pass_id)
    pass_dir = output / f"pass_{pass_id.casefold()}"
    sheets_dir = output / f"contact_sheets_pass_{pass_id.casefold()}"
    pass_dir.mkdir(exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)
    sheet_index: list[dict[str, Any]] = []
    for shard_index, records in enumerate(shards):
        shard_id = f"shard_{shard_index:04d}"
        shard_dir = pass_dir / shard_id
        shard_dir.mkdir(exist_ok=True)
        inputs_dir = shard_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        safe_records = [
            {
                "deterministic_pixel_facts": record["deterministic_pixel_facts"],
                "dimensions": record["dimensions"],
                "image_sha256": record["image_sha256"],
                "record_id": record["record_id"],
            }
            for record in records
        ]
        batch = {
            "field_order": list(PASS_FIELD_ORDER[pass_id]),
            "instructions": blind_prompt(),
            "output_schema_sha256": hashlib.sha256(canonical_json_bytes(output_schema())).hexdigest(),
            "pass_id": pass_id,
            "prompt_version": PROMPT_VERSION,
            "records": safe_records,
            "schema_version": "sprite_lab_codex_blind_batch_v1",
            "shard_id": shard_id,
            "taxonomy": taxonomy(),
        }
        audit = audit_blind_value(batch, forbidden_fingerprints=fingerprints)
        batch["pre_inspection_audit"] = audit
        _write_new_or_equal_json(shard_dir / "batch_payload.json", batch)
        _write_new_or_equal_json(
            shard_dir / "shard_manifest.json",
            {
                "pass_id": pass_id,
                "record_ids": [record["record_id"] for record in safe_records],
                "schema_version": "sprite_lab_codex_blind_shard_v1",
                "shard_id": shard_id,
            },
        )
        for record in safe_records:
            source = output / "blind_images" / f"{record['record_id']}.png"
            destination = inputs_dir / source.name
            if not source.is_file():
                raise CodexBlindError(f"blind image missing for {record['record_id']}")
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
        for sheet_number, start in enumerate(range(0, len(safe_records), CONTACT_SHEET_SIDE**2)):
            sheet_records = safe_records[start : start + CONTACT_SHEET_SIDE**2]
            sheet_name = f"{shard_id}_sheet_{sheet_number:02d}.png"
            sheet_path = sheets_dir / sheet_name
            rendered = render_contact_sheet(output, sheet_records, pass_id=pass_id)
            if sheet_path.exists():
                with Image.open(sheet_path) as existing:
                    if existing.convert("RGB").tobytes() != rendered.tobytes():
                        raise CodexBlindError(f"refusing to overwrite changed contact sheet: {sheet_path}")
            else:
                rendered.save(sheet_path, format="PNG", compress_level=9, optimize=False, pnginfo=None)
            sheet_index.append(
                {
                    "pass_id": pass_id,
                    "record_ids": [record["record_id"] for record in sheet_records],
                    "sheet": sheet_name,
                    "shard_id": shard_id,
                }
            )
    _write_json(sheets_dir / "index.json", {"sheets": sheet_index, "total": len(sheet_index)})
    ownership_path = output / "shard_ownership.json"
    ownership = (
        _read_json(ownership_path)
        if ownership_path.exists()
        else {
            "passes": {"A": {}, "B": {}},
            "schema_version": "sprite_lab_codex_shard_ownership_v1",
        }
    )
    for shard_index in range(len(shards)):
        ownership["passes"][pass_id].setdefault(f"shard_{shard_index:04d}", None)
    _write_json(ownership_path, ownership)
    progress = _read_json(output / "progress.json")
    progress[f"pass_{pass_id.casefold()}_shard_count"] = len(shards)
    progress[f"pass_{pass_id.casefold()}_prepared"] = True
    _write_json(output / "progress.json", progress)
    return {
        "pass_id": pass_id,
        "record_count": len(rows),
        "shard_count": len(shards),
        "sheet_count": len(sheet_index),
        "status": "prepared",
    }


def render_contact_sheet(
    output_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    pass_id: str,
) -> Image.Image:
    """Render a 4x4 sheet containing only full opaque IDs and blind images."""

    if len(records) > CONTACT_SHEET_SIDE**2:
        raise ValueError("a contact sheet accepts at most 16 records")
    output = Path(output_root)
    tile_width = 160
    label_height = 44
    tile_height = DISPLAY_SIZE + label_height + 8
    canvas = Image.new(
        "RGB",
        (CONTACT_SHEET_SIDE * tile_width, CONTACT_SHEET_SIDE * tile_height),
        (128, 128, 128),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        if pass_id == "A":
            row, column = divmod(index, CONTACT_SHEET_SIDE)
        elif pass_id == "B":
            column, row = divmod(index, CONTACT_SHEET_SIDE)
        else:
            raise CodexBlindError(f"invalid pass_id: {pass_id!r}")
        left = column * tile_width
        top = row * tile_height
        record_id = str(record["record_id"])
        assert_opaque_id(record_id)
        with Image.open(output / "blind_images" / f"{record_id}.png") as source:
            image = source.convert("RGB")
        image_left = left + (tile_width - DISPLAY_SIZE) // 2
        if pass_id == "A":
            image_top = top + 4
            label_top = top + DISPLAY_SIZE + 8
        else:
            label_top = top + 3
            image_top = top + label_height + 4
        canvas.paste(image, (image_left, image_top))
        for line_index, text in enumerate(_wrapped_record_id(record_id)):
            draw.text((left + 8, label_top + line_index * 10), text, fill=(245, 245, 245), font=font)
    return canvas


def validate_batch(output_root: str | Path, pass_id: str, shard_id: str) -> dict[str, Any]:
    """Re-run the leakage and binding audit immediately before inspection."""

    output = Path(output_root).resolve()
    batch_path = output / f"pass_{pass_id.casefold()}" / shard_id / "batch_payload.json"
    batch = _read_json(batch_path)
    if batch.get("pass_id") != pass_id or batch.get("shard_id") != shard_id:
        raise CodexBlindError("batch pass/shard binding mismatch")
    fingerprints = _read_json(output / "forbidden_metadata_fingerprints.json")["fingerprints"]
    audit = audit_blind_value(batch, forbidden_fingerprints=fingerprints)
    expected = {row["record_id"]: row for row in _read_jsonl(output / "blind_manifest.jsonl")}
    for record in batch.get("records", []):
        bound = expected.get(record.get("record_id"))
        if bound is None or record.get("image_sha256") != bound.get("image_sha256"):
            raise CodexBlindError("batch record/image hash binding mismatch")
        image_path = output / "blind_images" / f"{record['record_id']}.png"
        if not image_path.is_file():
            raise CodexBlindError("batch image is missing")
    return {**audit, "record_count": len(batch["records"]), "shard_id": shard_id}


def claim_shards(output_root: str | Path, pass_id: str, owner: str, shard_ids: Sequence[str]) -> dict[str, Any]:
    output = Path(output_root).resolve()
    ownership_path = output / "shard_ownership.json"
    ownership = _read_json(ownership_path)
    pass_owners = ownership["passes"][pass_id]
    for shard_id in shard_ids:
        if shard_id not in pass_owners:
            raise CodexBlindError(f"unknown shard: {pass_id}/{shard_id}")
        previous = pass_owners[shard_id]
        if previous not in (None, owner):
            raise CodexBlindError(f"duplicate shard ownership: {pass_id}/{shard_id} belongs to {previous}")
    for shard_id in shard_ids:
        pass_owners[shard_id] = owner
    _write_json(ownership_path, ownership)
    return {"claimed": list(shard_ids), "owner": owner, "pass_id": pass_id}


def ingest_compact_labels(
    output_root: str | Path,
    pass_id: str,
    shard_id: str,
    *,
    model_display: str,
    session_id: str,
) -> dict[str, Any]:
    """Expand visually authored compact rows into the required field envelopes."""

    output = Path(output_root).resolve()
    shard_dir = output / f"pass_{pass_id.casefold()}" / shard_id
    labels_path = shard_dir / "labels.jsonl"
    checkpoint_path = shard_dir / "checkpoint.json"
    if checkpoint_path.exists() or labels_path.exists():
        raise CodexBlindError(f"refusing to overwrite completed or existing shard output: {pass_id}/{shard_id}")
    validate_batch(output, pass_id, shard_id)
    compact_path = shard_dir / "compact_labels.jsonl"
    compact_rows = _read_jsonl(compact_path)
    batch = _read_json(shard_dir / "batch_payload.json")
    expected = {str(row["record_id"]): row for row in batch["records"]}
    _assert_unique(str(row.get("record_id")) for row in compact_rows)
    if set(expected) != {str(row.get("record_id")) for row in compact_rows}:
        raise CodexBlindError("compact labels have missing or extra records")
    labels = [
        _expand_compact_label(
            row,
            expected[str(row["record_id"])],
            pass_id=pass_id,
            model_display=model_display,
            session_id=session_id,
        )
        for row in compact_rows
    ]
    for label in labels:
        validate_label(label, expected[label["record_id"]], pass_id=pass_id)
    label_bytes = b"".join(canonical_json_bytes(label) + b"\n" for label in labels)
    labels_path.write_bytes(label_bytes)
    labels_sha256 = hashlib.sha256(label_bytes).hexdigest()
    _write_json(
        checkpoint_path,
        {
            "completed_record_ids": [label["record_id"] for label in labels],
            "jsonl_sha256": labels_sha256,
            "label_count": len(labels),
            "pass_id": pass_id,
            "prompt_version": PROMPT_VERSION,
            "schema_version": "sprite_lab_codex_blind_shard_checkpoint_v1",
            "session_id": session_id,
            "shard_id": shard_id,
        },
    )
    _refresh_progress(output)
    return {
        "jsonl_sha256": labels_sha256,
        "label_count": len(labels),
        "pass_id": pass_id,
        "shard_id": shard_id,
        "status": "completed",
    }


def validate_label(label: Mapping[str, Any], expected: Mapping[str, Any], *, pass_id: str) -> None:
    required = {
        "schema_version",
        "record_id",
        "image_sha256",
        "pass_id",
        "fields",
        "overall_visual_certainty",
        "needs_individual_inspection",
        "quality_flags",
        "evidence_origin",
        "labeler",
    }
    if set(label) != required:
        raise CodexBlindError("label top-level schema keys are invalid")
    if label["schema_version"] != SCHEMA_VERSION or label["pass_id"] != pass_id:
        raise CodexBlindError("label schema/pass binding is invalid")
    if label["record_id"] != expected["record_id"] or label["image_sha256"] != expected["image_sha256"]:
        raise CodexBlindError("label record/image binding is invalid")
    assert_opaque_id(str(label["record_id"]))
    if label["overall_visual_certainty"] not in CONFIDENCE_LEVELS:
        raise CodexBlindError("invalid overall_visual_certainty")
    if not isinstance(label["needs_individual_inspection"], bool) or not isinstance(label["quality_flags"], list):
        raise CodexBlindError("invalid inspection/quality fields")
    if label["evidence_origin"] != EVIDENCE_ORIGIN:
        raise CodexBlindError("invalid evidence_origin")
    labeler = label["labeler"]
    if not isinstance(labeler, Mapping) or labeler.get("surface") != "codex":
        raise CodexBlindError("invalid labeler surface")
    if labeler.get("prompt_version") != PROMPT_VERSION or not labeler.get("session_id"):
        raise CodexBlindError("invalid labeler binding")
    fields = label["fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(FIELD_NAMES):
        raise CodexBlindError("label fields are incomplete")
    controlled = {
        "domain": set(DOMAINS),
        "category": set(CATEGORIES),
        "canonical_object": set(CANONICAL_OBJECTS),
        "role": set(ROLES),
        "visual_material_cue": set(VISUAL_MATERIAL_CUES),
    }
    for name, field in fields.items():
        if not isinstance(field, Mapping) or set(field) != {"value", "state", "confidence", "visual_evidence"}:
            raise CodexBlindError(f"invalid field envelope: {name}")
        if field["state"] not in FIELD_STATES or field["confidence"] not in CONFIDENCE_LEVELS:
            raise CodexBlindError(f"invalid field state/confidence: {name}")
        if not isinstance(field["visual_evidence"], str) or not field["visual_evidence"].strip():
            raise CodexBlindError(f"missing visual evidence: {name}")
        value = field["value"]
        if field["state"] != "known" and value is not None:
            raise CodexBlindError(f"non-known field must be null: {name}")
        if field["state"] == "known" and value is None:
            raise CodexBlindError(f"known field must have a value: {name}")
        if name in controlled and value is not None and value not in controlled[name]:
            raise CodexBlindError(f"taxonomy-invalid field value: {name}={value!r}")
        if name.endswith("_colors") and value is not None:
            if not isinstance(value, list) or any(item not in COLOR_TERMS for item in value):
                raise CodexBlindError(f"invalid controlled colors: {name}")


def _expand_compact_label(
    compact: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    pass_id: str,
    model_display: str,
    session_id: str,
) -> dict[str, Any]:
    certainty = str(compact.get("certainty") or "low")
    if certainty not in CONFIDENCE_LEVELS:
        raise CodexBlindError("compact certainty is invalid")
    common_evidence = str(compact.get("evidence") or "Visible pixels do not support a more specific label.")
    color_evidence = str(compact.get("color_evidence") or common_evidence)
    material_evidence = str(compact.get("material_evidence") or common_evidence)
    fields: dict[str, Any] = {}
    color_names = {"primary_colors", "secondary_colors", "outline_colors"}
    for name in PASS_FIELD_ORDER[pass_id]:
        if name == "explicit_material":
            state = "not_applicable" if compact.get("material_not_applicable") is True else "unsupported"
            value = None
            evidence = material_evidence
            confidence = "high" if state == "not_applicable" else certainty
        else:
            value = compact.get(name)
            state = "known" if value is not None else "model_abstained"
            evidence = (
                color_evidence
                if name in color_names
                else material_evidence
                if name == "visual_material_cue"
                else common_evidence
            )
            confidence = certainty if state == "known" else "low"
        fields[name] = {
            "value": value,
            "state": state,
            "confidence": confidence,
            "visual_evidence": evidence,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": compact["record_id"],
        "image_sha256": expected["image_sha256"],
        "pass_id": pass_id,
        "fields": fields,
        "overall_visual_certainty": certainty,
        "needs_individual_inspection": bool(compact.get("needs_individual_inspection", certainty == "low")),
        "quality_flags": sorted({str(item) for item in compact.get("quality_flags", [])}),
        "evidence_origin": EVIDENCE_ORIGIN,
        "labeler": {
            "surface": "codex",
            "model_display": model_display,
            "session_id": session_id,
            "prompt_version": PROMPT_VERSION,
        },
    }


def _refresh_progress(output: Path) -> None:
    progress = _read_json(output / "progress.json")
    manifest = _read_jsonl(output / "blind_manifest.jsonl")
    all_ids = {str(row["record_id"]) for row in manifest}
    for pass_id in ("A", "B"):
        completed: set[str] = set()
        pass_dir = output / f"pass_{pass_id.casefold()}"
        for checkpoint_path in sorted(pass_dir.glob("shard_*/checkpoint.json")):
            checkpoint = _read_json(checkpoint_path)
            labels_path = checkpoint_path.with_name("labels.jsonl")
            if _file_sha256(labels_path) != checkpoint.get("jsonl_sha256"):
                raise CodexBlindError(f"checkpoint JSONL hash mismatch: {labels_path}")
            completed.update(str(value) for value in checkpoint["completed_record_ids"])
        progress[f"pass_{pass_id.casefold()}_completed"] = len(completed)
        order = [str(row["record_id"]) for shard in deterministic_shards(manifest, pass_id) for row in shard]
        progress["next_unprocessed"][pass_id] = next(
            (record_id for record_id in order if record_id not in completed), None
        )
        if not completed.issubset(all_ids):
            raise CodexBlindError(f"Pass {pass_id} checkpoint contains an unknown record")
    _write_json(output / "progress.json", progress)


def _wrapped_record_id(record_id: str) -> list[str]:
    return [record_id[index : index + 17] for index in range(0, len(record_id), 17)]


def _write_new_or_equal_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise CodexBlindError(f"refusing to overwrite changed artifact: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexBlindError(f"invalid JSON object at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CodexBlindError(f"expected JSON object: {path}")
    return value


def freeze_pass(output_root: str | Path, pass_id: str, *, allow_partial: bool = False) -> dict[str, Any]:
    """Freeze immutable completed shard hashes before cross-pass comparison."""

    output = Path(output_root).resolve()
    pass_dir = output / f"pass_{pass_id.casefold()}"
    freeze_path = pass_dir / "freeze.json"
    if freeze_path.exists():
        return _read_json(freeze_path)
    manifest_count = len(_read_jsonl(output / "blind_manifest.jsonl"))
    checkpoints: list[dict[str, Any]] = []
    completed: set[str] = set()
    for checkpoint_path in sorted(pass_dir.glob("shard_*/checkpoint.json")):
        checkpoint = _read_json(checkpoint_path)
        labels_path = checkpoint_path.with_name("labels.jsonl")
        observed = _file_sha256(labels_path)
        if observed != checkpoint.get("jsonl_sha256"):
            raise CodexBlindError(f"cannot freeze changed shard JSONL: {labels_path}")
        ids = [str(value) for value in checkpoint["completed_record_ids"]]
        overlap = completed.intersection(ids)
        if overlap:
            raise CodexBlindError(f"duplicate completed IDs across shards: {sorted(overlap)}")
        completed.update(ids)
        checkpoints.append(
            {
                "jsonl_sha256": observed,
                "label_count": checkpoint["label_count"],
                "shard_id": checkpoint["shard_id"],
            }
        )
    if len(completed) != manifest_count and not allow_partial:
        raise CodexBlindError(
            f"Pass {pass_id} is incomplete: {len(completed)} of {manifest_count}; use allow_partial only at a stop boundary"
        )
    frozen = {
        "allow_partial": allow_partial,
        "completed_record_count": len(completed),
        "pass_id": pass_id,
        "record_ids_sha256": hashlib.sha256("\n".join(sorted(completed)).encode("ascii")).hexdigest(),
        "schema_version": "sprite_lab_codex_blind_pass_freeze_v1",
        "shards": checkpoints,
        "status": "frozen_complete" if len(completed) == manifest_count else "frozen_partial_stop_boundary",
    }
    _write_json(freeze_path, frozen)
    return frozen


def reconcile_campaign(output_root: str | Path) -> dict[str, Any]:
    """Reconcile frozen Pass A/B labels conservatively, field by field."""

    output = Path(output_root).resolve()
    for pass_id in ("A", "B"):
        if not (output / f"pass_{pass_id.casefold()}" / "freeze.json").is_file():
            raise CodexBlindError(f"Pass {pass_id} must be frozen before reconciliation")
    labels_a, sources_a = _load_frozen_labels(output, "A")
    labels_b, sources_b = _load_frozen_labels(output, "B")
    shared_ids = sorted(set(labels_a).intersection(labels_b))
    reconciled: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    mandatory: list[dict[str, Any]] = []
    status_counts = dict.fromkeys(_resolution_statuses(), 0)
    for record_id in shared_ids:
        left = labels_a[record_id]
        right = labels_b[record_id]
        if left["image_sha256"] != right["image_sha256"]:
            raise CodexBlindError(f"Pass A/B image hash mismatch: {record_id}")
        field_resolutions: dict[str, Any] = {}
        for field_name in FIELD_NAMES:
            resolution = _reconcile_field(field_name, left["fields"][field_name], right["fields"][field_name])
            field_resolutions[field_name] = resolution
            if resolution["status"] == "codex_conflicted":
                conflicts.append(
                    {
                        "field": field_name,
                        "image_sha256": left["image_sha256"],
                        "pass_a": left["fields"][field_name],
                        "pass_b": right["fields"][field_name],
                        "record_id": record_id,
                        "schema_version": "sprite_lab_codex_blind_conflict_v1",
                    }
                )
            if resolution["status"] in {"codex_abstained", "codex_unsupported"}:
                abstentions.append(
                    {
                        "field": field_name,
                        "image_sha256": left["image_sha256"],
                        "reason": resolution["rationale"],
                        "record_id": record_id,
                        "schema_version": "sprite_lab_codex_blind_abstention_v1",
                        "status": resolution["status"],
                    }
                )
            if field_name in CRITICAL_FIELDS and (
                resolution["status"] == "codex_conflicted" or resolution["confidence"] == "low"
            ):
                mandatory.append(
                    {
                        "field": field_name,
                        "reason": "critical_conflict"
                        if resolution["status"] == "codex_conflicted"
                        else "low_confidence_critical_field",
                        "record_id": record_id,
                    }
                )
            if field_name == "explicit_material" and resolution["value"] is not None:
                mandatory.append({"field": field_name, "reason": "exact_material_claim", "record_id": record_id})
        critical_statuses = [field_resolutions[name]["status"] for name in CRITICAL_FIELDS]
        overall_status = _overall_resolution_status(critical_statuses)
        status_counts[overall_status] += 1
        reconciled.append(
            {
                "critical_status": overall_status,
                "fields": field_resolutions,
                "image_sha256": left["image_sha256"],
                "pass_a_jsonl_sha256": sources_a[record_id],
                "pass_b_jsonl_sha256": sources_b[record_id],
                "prompt_version": PROMPT_VERSION,
                "record_id": record_id,
                "schema_version": "sprite_lab_codex_blind_reconciled_v1",
            }
        )
    _replace_empty_or_equal_jsonl(output / "reconciled_labels.jsonl", reconciled)
    _replace_empty_or_equal_jsonl(output / "conflicts.jsonl", conflicts)
    _replace_empty_or_equal_jsonl(output / "abstentions.jsonl", abstentions)
    _replace_empty_or_equal_jsonl(output / "mandatory_inspection.jsonl", mandatory)
    summary = {
        "abstention_count": len(abstentions),
        "conflict_count": len(conflicts),
        "reconciled_record_count": len(reconciled),
        "status_counts": status_counts,
    }
    _write_json(output / "reconciliation_report.json", summary)
    return summary


def prepare_health_check(output_root: str | Path, milestone: int) -> dict[str, Any]:
    """Select 20 completed records and build a fresh blind audit payload."""

    if milestone <= 0 or milestone % 100:
        raise CodexBlindError("health-check milestone must be a positive multiple of 100")
    output = Path(output_root).resolve()
    labels, _ = _load_checkpointed_labels(output, "A")
    if len(labels) < milestone:
        raise CodexBlindError(f"health check {milestone} requires at least {milestone} completed Pass-A labels")
    ranked = sorted(
        labels,
        key=lambda record_id: hashlib.sha256(
            f"codex_blind_health_v1\0{milestone}\0{record_id}".encode("ascii")
        ).digest(),
    )
    selected_ids = ranked[:20]
    blind = {str(row["record_id"]): row for row in _read_jsonl(output / "blind_manifest.jsonl")}
    records = [blind[record_id] for record_id in selected_ids]
    audit_dir = output / "health_checks" / f"check_{milestone:06d}"
    if audit_dir.exists() and any(audit_dir.iterdir()):
        raise CodexBlindError(f"refusing to overwrite health-check directory: {audit_dir}")
    audit_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = audit_dir / "inputs"
    inputs_dir.mkdir()
    safe_records = [
        {
            "deterministic_pixel_facts": record["deterministic_pixel_facts"],
            "dimensions": record["dimensions"],
            "image_sha256": record["image_sha256"],
            "record_id": record["record_id"],
        }
        for record in records
    ]
    batch = {
        "audit_kind": "fresh_codex_blind_health_relabel",
        "field_order": list(PASS_FIELD_ORDER["B"]),
        "instructions": blind_prompt(),
        "milestone": milestone,
        "prompt_version": PROMPT_VERSION,
        "records": safe_records,
        "schema_version": "sprite_lab_codex_blind_health_batch_v1",
        "taxonomy": taxonomy(),
    }
    fingerprints = _read_json(output / "forbidden_metadata_fingerprints.json")["fingerprints"]
    batch["pre_inspection_audit"] = audit_blind_value(batch, forbidden_fingerprints=fingerprints)
    _write_json(audit_dir / "batch_payload.json", batch)
    for record in safe_records:
        source = output / "blind_images" / f"{record['record_id']}.png"
        destination = inputs_dir / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    for sheet_number, start in enumerate(range(0, len(safe_records), CONTACT_SHEET_SIDE**2)):
        sheet = render_contact_sheet(output, safe_records[start : start + CONTACT_SHEET_SIDE**2], pass_id="B")
        sheet.save(audit_dir / f"sheet_{sheet_number:02d}.png", format="PNG", compress_level=9, optimize=False)
    return {"milestone": milestone, "record_count": len(safe_records), "status": "prepared"}


def validate_health_batch(output_root: str | Path, milestone: int) -> dict[str, Any]:
    """Re-run blindness and image binding checks before a fresh audit inspects pixels."""

    output = Path(output_root).resolve()
    audit_dir = output / "health_checks" / f"check_{milestone:06d}"
    batch = _read_json(audit_dir / "batch_payload.json")
    fingerprints = _read_json(output / "forbidden_metadata_fingerprints.json")["fingerprints"]
    audit = audit_blind_value(batch, forbidden_fingerprints=fingerprints)
    blind = {str(row["record_id"]): row for row in _read_jsonl(output / "blind_manifest.jsonl")}
    for record in batch.get("records", []):
        expected = blind.get(str(record.get("record_id")))
        if expected is None or record.get("image_sha256") != expected.get("image_sha256"):
            raise CodexBlindError("health batch record/image hash binding mismatch")
    return {**audit, "milestone": milestone, "record_count": len(batch["records"])}


def ingest_health_compact(
    output_root: str | Path,
    milestone: int,
    *,
    model_display: str,
    session_id: str,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    audit_dir = output / "health_checks" / f"check_{milestone:06d}"
    labels_path = audit_dir / "audit_labels.jsonl"
    if labels_path.exists():
        raise CodexBlindError("refusing to overwrite health audit labels")
    batch = _read_json(audit_dir / "batch_payload.json")
    fingerprints = _read_json(output / "forbidden_metadata_fingerprints.json")["fingerprints"]
    audit_blind_value(batch, forbidden_fingerprints=fingerprints)
    compact_rows = _read_jsonl(audit_dir / "compact_labels.jsonl")
    expected = {str(row["record_id"]): row for row in batch["records"]}
    _assert_unique(str(row.get("record_id")) for row in compact_rows)
    if set(expected) != {str(row.get("record_id")) for row in compact_rows}:
        raise CodexBlindError("health audit has missing or duplicate records")
    labels = [
        _expand_compact_label(
            row,
            expected[str(row["record_id"])],
            pass_id="B",
            model_display=model_display,
            session_id=session_id,
        )
        for row in compact_rows
    ]
    for label in labels:
        validate_label(label, expected[label["record_id"]], pass_id="B")
    payload = b"".join(canonical_json_bytes(label) + b"\n" for label in labels)
    labels_path.write_bytes(payload)
    report = evaluate_health_check(output, milestone)
    return report


def evaluate_health_check(output_root: str | Path, milestone: int) -> dict[str, Any]:
    """Calculate hard health gates and stop the campaign on any violation."""

    output = Path(output_root).resolve()
    audit_dir = output / "health_checks" / f"check_{milestone:06d}"
    batch = _read_json(audit_dir / "batch_payload.json")
    expected = {str(row["record_id"]): row for row in batch["records"]}
    raw_audit = _read_jsonl(audit_dir / "audit_labels.jsonl")
    ids = [str(row.get("record_id")) for row in raw_audit]
    duplicate_count = len(ids) - len(set(ids))
    missing_count = len(set(expected) - set(ids))
    image_hash_mismatch = 0
    taxonomy_invalid = 0
    exact_material_overclaim = 0
    audit: dict[str, dict[str, Any]] = {}
    for row in raw_audit:
        record_id = str(row.get("record_id"))
        if record_id in audit:
            continue
        audit[record_id] = row
        bound = expected.get(record_id)
        if bound is None or row.get("image_sha256") != bound.get("image_sha256"):
            image_hash_mismatch += 1
            continue
        try:
            validate_label(row, bound, pass_id="B")
        except CodexBlindError:
            taxonomy_invalid += 1
            continue
        material = row["fields"]["explicit_material"]
        if material["value"] is not None and "deterministically_justified_exact_material" not in row["quality_flags"]:
            exact_material_overclaim += 1
    reference, _ = _load_checkpointed_labels(output, "A")
    critical_compared = 0
    critical_disagreements = 0
    reference_sessions: set[str] = set()
    audit_sessions: set[str] = set()
    for record_id in set(expected).intersection(audit).intersection(reference):
        left = reference[record_id]
        right = audit[record_id]
        reference_sessions.add(str(left["labeler"]["session_id"]))
        audit_sessions.add(str(right["labeler"]["session_id"]))
        for field_name in CRITICAL_FIELDS:
            left_field = left["fields"][field_name]
            right_field = right["fields"][field_name]
            if left_field["state"] == "known" or right_field["state"] == "known":
                critical_compared += 1
                if left_field["state"] != right_field["state"] or left_field["value"] != right_field["value"]:
                    critical_disagreements += 1
    disagreement_rate = critical_disagreements / critical_compared if critical_compared else 0.0
    exact_material_rate = exact_material_overclaim / len(expected) if expected else 0.0
    forbidden_leakage = int(batch.get("pre_inspection_audit", {}).get("forbidden_metadata_leakage", 1))
    fresh_session_violation = bool(reference_sessions.intersection(audit_sessions))
    gate_failures = []
    if forbidden_leakage:
        gate_failures.append("forbidden_metadata_leakage")
    if image_hash_mismatch:
        gate_failures.append("image_hash_mismatch")
    if taxonomy_invalid:
        gate_failures.append("taxonomy_invalid_output")
    if disagreement_rate > 0.05:
        gate_failures.append("critical_field_disagreement")
    if exact_material_rate > 0.01:
        gate_failures.append("exact_material_overclaim")
    if missing_count:
        gate_failures.append("missing_record")
    if duplicate_count:
        gate_failures.append("duplicate_record")
    if fresh_session_violation:
        gate_failures.append("fresh_session_violation")
    report = {
        "critical_field_comparisons": critical_compared,
        "critical_field_disagreement_rate": round(disagreement_rate, 6),
        "duplicate_record_count": duplicate_count,
        "exact_material_overclaim_rate": round(exact_material_rate, 6),
        "forbidden_metadata_leakage": forbidden_leakage,
        "fresh_session_violation": fresh_session_violation,
        "gate_failures": gate_failures,
        "image_hash_mismatch_count": image_hash_mismatch,
        "milestone": milestone,
        "missing_record_count": missing_count,
        "passed": not gate_failures,
        "schema_version": "sprite_lab_codex_blind_health_report_v1",
        "taxonomy_invalid_count": taxonomy_invalid,
    }
    _write_json(audit_dir / "health_report.json", report)
    progress = _read_json(output / "progress.json")
    progress["health_gate_status"] = "passed" if report["passed"] else "failed_stopped"
    if not report["passed"]:
        progress["campaign_status"] = "stopped_health_gate"
    _write_json(output / "progress.json", progress)
    return report


def _reconcile_field(name: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_known = left["state"] == "known"
    right_known = right["state"] == "known"
    if left_known and right_known and left["value"] == right["value"]:
        return {
            "confidence": _minimum_confidence(str(left["confidence"]), str(right["confidence"])),
            "rationale": "Both blind passes independently returned the same value.",
            "status": "codex_consistent",
            "value": left["value"],
        }
    if not left_known and not right_known:
        unsupported = left["state"] in {"unsupported", "not_applicable"} and right["state"] in {
            "unsupported",
            "not_applicable",
        }
        return {
            "confidence": "low",
            "rationale": "Neither blind pass supplied a supported value.",
            "status": "codex_unsupported" if unsupported else "codex_abstained",
            "value": None,
        }
    if left_known != right_known:
        known = left if left_known else right
        if known["confidence"] == "low":
            return {
                "confidence": "low",
                "rationale": "One pass abstained and the other value was low confidence.",
                "status": "codex_abstained",
                "value": None,
            }
        return {
            "confidence": str(known["confidence"]),
            "rationale": "Only one blind pass supplied a non-low-confidence value.",
            "status": "codex_single_pass",
            "value": known["value"],
        }
    if name == "description":
        shared = _shared_visible_description(str(left["value"]), str(right["value"]))
        if shared is not None:
            return {
                "confidence": "low",
                "rationale": "The conservative description retains only visible terms shared by both passes.",
                "status": "codex_single_pass",
                "value": shared,
            }
    if name == "explicit_material":
        return {
            "confidence": "low",
            "rationale": "The blind passes disagreed on exact material, so exact material is nulled.",
            "status": "codex_conflicted",
            "value": None,
        }
    if name == "visual_material_cue" and "unknown" in {left["value"], right["value"]}:
        known = right if left["value"] == "unknown" else left
        return {
            "confidence": "low",
            "rationale": "One pass supplied a broad compatible cue while the other remained unknown.",
            "status": "codex_single_pass",
            "value": known["value"],
        }
    return {
        "confidence": "low",
        "rationale": "The blind passes returned conflicting values.",
        "status": "codex_conflicted",
        "value": None,
    }


def _overall_resolution_status(statuses: Sequence[str]) -> str:
    if "codex_conflicted" in statuses:
        return "codex_conflicted"
    if all(status == "codex_consistent" for status in statuses):
        return "codex_consistent"
    if "codex_single_pass" in statuses:
        return "codex_single_pass"
    if all(status == "codex_unsupported" for status in statuses):
        return "codex_unsupported"
    return "codex_abstained"


def _resolution_statuses() -> tuple[str, ...]:
    return (
        "codex_consistent",
        "codex_single_pass",
        "codex_conflicted",
        "codex_abstained",
        "codex_unsupported",
    )


def _minimum_confidence(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order[left] <= order[right] else right


def _shared_visible_description(left: str, right: str) -> str | None:
    stop = {
        "with",
        "from",
        "that",
        "this",
        "visible",
        "pixels",
        "shows",
        "showing",
        "form",
        "object",
    }
    right_words = {word for word in re.findall(r"[a-z]+", right.casefold()) if len(word) >= 3 and word not in stop}
    shared: list[str] = []
    for word in re.findall(r"[a-z]+", left.casefold()):
        if len(word) >= 3 and word not in stop and word in right_words and word not in shared:
            shared.append(word)
    if len(shared) < 2:
        return None
    return "Visible pixels show " + " ".join(shared[:8]) + "."


def _load_frozen_labels(output: Path, pass_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    freeze = _read_json(output / f"pass_{pass_id.casefold()}" / "freeze.json")
    labels, sources = _load_checkpointed_labels(output, pass_id)
    if len(labels) != freeze["completed_record_count"]:
        raise CodexBlindError(f"Pass {pass_id} changed after freeze")
    record_hash = hashlib.sha256("\n".join(sorted(labels)).encode("ascii")).hexdigest()
    if record_hash != freeze["record_ids_sha256"]:
        raise CodexBlindError(f"Pass {pass_id} membership changed after freeze")
    return labels, sources


def _load_checkpointed_labels(output: Path, pass_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifest = {str(row["record_id"]): row for row in _read_jsonl(output / "blind_manifest.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for checkpoint_path in sorted((output / f"pass_{pass_id.casefold()}").glob("shard_*/checkpoint.json")):
        checkpoint = _read_json(checkpoint_path)
        labels_path = checkpoint_path.with_name("labels.jsonl")
        jsonl_hash = _file_sha256(labels_path)
        if jsonl_hash != checkpoint.get("jsonl_sha256"):
            raise CodexBlindError(f"checkpoint JSONL hash mismatch: {labels_path}")
        for label in _read_jsonl(labels_path):
            record_id = str(label.get("record_id"))
            if record_id in labels:
                raise CodexBlindError(f"duplicate label record across shards: {record_id}")
            expected = manifest.get(record_id)
            if expected is None:
                raise CodexBlindError(f"label has unknown record_id: {record_id}")
            validate_label(label, expected, pass_id=pass_id)
            labels[record_id] = label
            sources[record_id] = jsonl_hash
    return labels, sources


def _replace_empty_or_equal_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current and current != encoded:
            raise CodexBlindError(f"refusing to replace frozen JSONL: {path}")
        if current == encoded:
            return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def export_supervision_candidates(output_root: str | Path) -> dict[str, Any]:
    """Map Codex evidence to weak/auxiliary/masked supervision only."""

    output = Path(output_root).resolve()
    reconciled = _read_jsonl(output / "reconciled_labels.jsonl")
    blind = {str(row["record_id"]): row for row in _read_jsonl(output / "blind_manifest.jsonl")}
    candidates: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {"auxiliary_only": 0, "masked": 0, "supervised_weak": 0}
    for row in reconciled:
        field_candidates: dict[str, Any] = {}
        for field_name, resolution in row["fields"].items():
            status = resolution["status"]
            if status == "codex_consistent" and field_name in CRITICAL_FIELDS:
                supervision_class = "supervised_weak"
                masked = False
            elif status in {"codex_abstained", "codex_unsupported"}:
                supervision_class = "masked"
                masked = True
            else:
                supervision_class = "auxiliary_only"
                masked = status == "codex_conflicted"
            class_counts[supervision_class] += 1
            field_candidates[field_name] = {
                "masked": masked,
                "source_status": status,
                "supervision_class": supervision_class,
                "value": resolution["value"] if not masked else None,
            }
        facts = blind[str(row["record_id"])]["deterministic_pixel_facts"]
        candidates.append(
            {
                "codex_is_human_ground_truth": False,
                "deterministic_pixel_supervision": {
                    "palette_rgba": facts.get("palette_rgba", []),
                    "supervision_class": "deterministic_field_supervision_only",
                },
                "fields": field_candidates,
                "image_sha256": row["image_sha256"],
                "record_id": row["record_id"],
                "schema_version": "sprite_lab_codex_supervision_candidate_v1",
            }
        )
    serialized = json.dumps(candidates, sort_keys=True)
    if "supervised_strong" in serialized:
        raise CodexBlindError("Codex-only evidence must never create supervised_strong")
    _replace_empty_or_equal_jsonl(output / "supervision_candidates.jsonl", candidates)
    return {"candidate_count": len(candidates), "field_supervision_class_counts": class_counts}


def reconcile_source_metadata(output_root: str | Path, raw_experiment: str | Path) -> dict[str, Any]:
    """Load source metadata only after both blind passes and reconciliation freeze."""

    output = Path(output_root).resolve()
    raw_root = Path(raw_experiment).resolve()
    for pass_id in ("A", "B"):
        if not (output / f"pass_{pass_id.casefold()}" / "freeze.json").is_file():
            raise CodexBlindError("source metadata cannot be loaded before Pass A and Pass B are frozen")
    reconciled_path = output / "reconciled_labels.jsonl"
    if not reconciled_path.is_file():
        raise CodexBlindError("source metadata cannot be loaded before blind reconciliation")
    provenance_path = raw_root / "provenance_manifest.jsonl"
    progress = _read_json(output / "progress.json")
    expected_hash = progress["source_artifact_hashes"].get("provenance_manifest.jsonl")
    if expected_hash != _file_sha256(provenance_path):
        raise CodexBlindError("source provenance artifact changed since blind staging")
    provenance = {str(row.get("record_id")): row for row in _read_jsonl(provenance_path)}
    reconciled = _read_jsonl(reconciled_path)
    rows: list[dict[str, Any]] = []
    state_counts = {
        "source_metadata_unverifiable": 0,
        "visual_source_agreement": 0,
        "visual_source_conflict": 0,
    }
    for visual in reconciled:
        record_id = str(visual["record_id"])
        source = provenance.get(record_id, {})
        declared = _declared_semantic_source_fields(source)
        agreements: list[str] = []
        conflicts: list[str] = []
        for field_name, source_value in declared.items():
            resolution = visual["fields"].get(field_name, {})
            visual_value = resolution.get("value")
            if visual_value is None:
                continue
            if str(visual_value).casefold() == str(source_value).casefold():
                agreements.append(field_name)
            else:
                conflicts.append(field_name)
        if conflicts:
            state = "visual_source_conflict"
        elif agreements:
            state = "visual_source_agreement"
        else:
            state = "source_metadata_unverifiable"
        state_counts[state] += 1
        rows.append(
            {
                "agreement_fields": agreements,
                "conflict_fields": conflicts,
                "declared_variant_count": len(source.get("declared_sheet_ids", []))
                if isinstance(source.get("declared_sheet_ids"), list)
                else 0,
                "documented_materials": _documented_materials(source),
                "image_sha256": visual["image_sha256"],
                "license_status": "present"
                if isinstance(source.get("licenses"), list) and source.get("licenses")
                else "missing_or_unverifiable",
                "provenance_status": source.get("provenance_status", "unverifiable"),
                "record_id": record_id,
                "schema_version": "sprite_lab_codex_source_reconciliation_v1",
                "semantic_filename_taint": bool(source.get("original_filename")),
                "source_metadata_state": state,
                "visual_label_replaced": False,
            }
        )
    _replace_empty_or_equal_jsonl(output / "source_reconciliation.jsonl", rows)
    progress["source_metadata_loaded_after_blind_freeze"] = True
    progress["source_reconciliation_state_counts"] = state_counts
    _write_json(output / "progress.json", progress)
    return {"record_count": len(rows), "state_counts": state_counts}


def build_label_distribution_report(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    reconciled = _read_jsonl(output / "reconciled_labels.jsonl")
    eligibility = _read_jsonl(output / "eligibility_manifest.jsonl")
    fields: dict[str, Any] = {}
    for field_name in FIELD_NAMES:
        status_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}
        for row in reconciled:
            resolution = row["fields"][field_name]
            status = str(resolution["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            value = resolution["value"]
            key = "<null>" if value is None else json.dumps(value, sort_keys=True)
            value_counts[key] = value_counts.get(key, 0) + 1
        fields[field_name] = {
            "status_counts": dict(sorted(status_counts.items())),
            "value_counts": dict(sorted(value_counts.items())),
        }
    eligibility_counts: dict[str, dict[str, int]] = {}
    for key in ("forensic_inclusion_decision", "provenance_status", "suitability_status"):
        counts: dict[str, int] = {}
        for row in eligibility:
            value = str(row.get(key, "unavailable"))
            counts[value] = counts.get(value, 0) + 1
        eligibility_counts[key] = dict(sorted(counts.items()))
    report = {
        "eligibility_status_counts": eligibility_counts,
        "fields": fields,
        "reconciled_record_count": len(reconciled),
        "schema_version": "sprite_lab_codex_label_distribution_v1",
    }
    _write_json(output / "label_distribution_report.json", report)
    return report


def checkpoint_stop(output_root: str | Path, reason: str) -> dict[str, Any]:
    if reason not in {"active_context_unreliable", "health_gate", "usage_limit"}:
        raise CodexBlindError("invalid campaign stop reason")
    output = Path(output_root).resolve()
    _refresh_progress(output)
    progress = _read_json(output / "progress.json")
    progress["campaign_status"] = f"stopped_{reason}"
    progress["stop_reason"] = reason
    progress["usage_limit_stop"] = reason == "usage_limit"
    _write_json(output / "progress.json", progress)
    return progress


def resume_status(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _refresh_progress(output)
    progress = _read_json(output / "progress.json")
    pass_id = "A" if progress["next_unprocessed"]["A"] is not None else "B"
    record_id = progress["next_unprocessed"][pass_id]
    shard_id = None
    if record_id is not None:
        manifest = _read_jsonl(output / "blind_manifest.jsonl")
        for index, shard in enumerate(deterministic_shards(manifest, pass_id)):
            if record_id in {row["record_id"] for row in shard}:
                shard_id = f"shard_{index:04d}"
                break
    return {
        "next_pass": pass_id if record_id is not None else None,
        "next_record_id": record_id,
        "next_shard_id": shard_id,
        "resume_command": f'python -m spritelab.dataset_v5.codex_blind resume --output "{output}"',
    }


def finalize_campaign(output_root: str | Path, raw_experiment: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    raw_root = Path(raw_experiment).resolve()
    supervision = export_supervision_candidates(output)
    distribution = build_label_distribution_report(output)
    progress = _read_json(output / "progress.json")
    reconciliation = _read_json(output / "reconciliation_report.json")
    failures = _read_jsonl(output / "failures.jsonl")
    health_reports = [
        _read_json(path) for path in sorted((output / "health_checks").glob("check_*/health_report.json"))
    ]
    source_report = _read_json(raw_root / "rebuild_report.json")
    provenance_blocked = distribution["eligibility_status_counts"]["provenance_status"].get("blocked", 0)
    provenance_blocked_total = sum(
        row.get("provenance_status") == "blocked" for row in _read_jsonl(raw_root / "provenance_manifest.jsonl")
    )
    pass_a_labels, _ = _load_checkpointed_labels(output, "A")
    pass_a_abstention_fields = sum(
        field["state"] == "model_abstained" for label in pass_a_labels.values() for field in label["fields"].values()
    )
    pass_a_abstention_records = sum(
        any(field["state"] == "model_abstained" for field in label["fields"].values())
        for label in pass_a_labels.values()
    )
    resume = resume_status(output)
    report = {
        "abstention_field_count": reconciliation["abstention_count"],
        "campaign_status": progress["campaign_status"],
        "conflict_field_count": reconciliation["conflict_count"],
        "consistent_critical_record_count": reconciliation["status_counts"]["codex_consistent"],
        "failed_record_count": len(failures),
        "health_checks": health_reports,
        "labels_are_calibrated_truth": False,
        "pass_a_completed": progress["pass_a_completed"],
        "pass_a_field_abstention_count": pass_a_abstention_fields,
        "pass_a_record_abstention_count": pass_a_abstention_records,
        "pass_b_completed": progress["pass_b_completed"],
        "readable_records": progress["readable_records"],
        "remaining_blockers": {
            "production_freeze_blocked": not source_report["freeze"]["production_frozen"],
            "provenance_blocked_readable_records": provenance_blocked,
            "provenance_blocked_total_records": provenance_blocked_total,
            "raw_source_gate_passed": source_report["raw_source_gate_passed"],
            "source_excluded_or_unreadable_records": len(failures),
        },
        "resume": resume,
        "schema_version": "sprite_lab_codex_blind_campaign_report_v1",
        "supervision": supervision,
        "usage_limit_stop": progress.get("usage_limit_stop", False),
    }
    _write_json(output / "campaign_report.json", report)
    health_lines = (
        "none completed"
        if not health_reports
        else ", ".join(
            f"{item['milestone']}: {item['critical_field_disagreement_rate']:.2%}" for item in health_reports
        )
    )
    markdown = f"""# Dataset-v5 Codex blind labeling campaign

This campaign contains conservative Codex visual proposals. They are not human
ground truth and are not claimed to be calibrated truth.

- Readable records: {report["readable_records"]}
- Pass A completed: {report["pass_a_completed"]}
- Pass B completed: {report["pass_b_completed"]}
- Codex-consistent critical records: {report["consistent_critical_record_count"]}
- Conflict fields: {report["conflict_field_count"]}
- Reconciled abstained/unsupported fields: {report["abstention_field_count"]}
- Pass-A records with a model abstention: {report["pass_a_record_abstention_count"]}
- Pass-A field abstentions: {report["pass_a_field_abstention_count"]}
- Source-excluded or failed records: {report["failed_record_count"]}
- Health-check critical disagreement: {health_lines}
- Usage-limit stop: {str(report["usage_limit_stop"]).lower()}

Resume with:

```powershell
{resume["resume_command"]}
```

The raw source gate and production freeze remain blocked. Provenance remains
blocked for {provenance_blocked} readable records ({provenance_blocked_total}
records across the full forensic inventory). A semantic label never
changes source eligibility.
"""
    (output / "campaign_report.md").write_text(markdown, encoding="utf-8", newline="\n")
    command_log = output / "command_log.txt"
    with command_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("prepare: python -m spritelab.dataset_v5.codex_blind prepare-pass --pass-id A|B ...\n")
        handle.write("validate: python -m spritelab.dataset_v5.codex_blind validate-batch ...\n")
        handle.write("reconcile: python -m spritelab.dataset_v5.codex_blind reconcile ...\n")
        handle.write("finalize: python -m spritelab.dataset_v5.codex_blind finalize ...\n")
    hashes = _artifact_hash_manifest(output)
    _write_json(output / "artifact_hashes.json", hashes)
    return report


def _artifact_hash_manifest(output: Path) -> dict[str, Any]:
    excluded_roots = {
        "blind_images",
        "contact_sheets_pass_a",
        "contact_sheets_pass_b",
        "original_blobs",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "artifact_hashes.json":
            continue
        relative = path.relative_to(output).as_posix()
        if relative.split("/", 1)[0] in excluded_roots or "/inputs/" in f"/{relative}/":
            continue
        artifacts[relative] = {"byte_length": path.stat().st_size, "sha256": _file_sha256(path)}
    root_hash = hashlib.sha256(canonical_json_bytes(artifacts)).hexdigest()
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "root_sha256": root_hash,
        "schema_version": "sprite_lab_codex_blind_artifact_hashes_v1",
    }


def _declared_semantic_source_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "declared_canonical_object": "canonical_object",
        "declared_category": "category",
        "declared_domain": "domain",
        "declared_role": "role",
    }
    result: dict[str, Any] = {}
    for source_key, field_name in allowed.items():
        value = source.get(source_key)
        if isinstance(value, str) and value.strip():
            result[field_name] = value.strip()
    return result


def _documented_materials(source: Mapping[str, Any]) -> list[str]:
    value = source.get("documented_materials")
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if isinstance(item, str) and item.strip()})


def render_blind_image(rgba: np.ndarray, *, display_size: int = DISPLAY_SIZE) -> Image.Image:
    """Nearest-neighbor fit on a neutral checkerboard, always display_size square."""

    source = Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), mode="RGBA")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("cannot render an empty image")
    scale = min(display_size / width, display_size / height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = source.resize(target, resample=Image.Resampling.NEAREST)
    checker = Image.new("RGBA", (display_size, display_size))
    draw = ImageDraw.Draw(checker)
    cell = 8
    colors = ((176, 176, 176, 255), (208, 208, 208, 255))
    for y in range(0, display_size, cell):
        for x in range(0, display_size, cell):
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=colors[(x // cell + y // cell) % 2])
    offset = ((display_size - target[0]) // 2, (display_size - target[1]) // 2)
    checker.alpha_composite(resized, dest=offset)
    return checker.convert("RGB")


def audit_blind_value(
    value: Any,
    *,
    forbidden_fingerprints: Iterable[str] = (),
) -> dict[str, Any]:
    """Inspect one generated prompt/input and fail before visual inspection."""

    findings: list[dict[str, str]] = []

    def walk(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).casefold()
                if normalized in FORBIDDEN_METADATA_KEYS:
                    findings.append({"location": f"{location}.{key}", "reason": "forbidden_key"})
                walk(child, f"{location}.{key}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")

    walk(value, "$")
    allowed = _approved_prompt_tokens()
    fingerprints = set(forbidden_fingerprints)
    serialized = canonical_json_bytes(value).decode("utf-8").casefold()
    for token in _TOKEN_RE.findall(serialized):
        if token in allowed or token.startswith(("rec_", "geo_")):
            continue
        if hashlib.sha256(token.encode("utf-8")).hexdigest() in fingerprints:
            findings.append({"location": "$", "reason": "forbidden_metadata_token"})
    if findings:
        raise ForbiddenMetadataError(json.dumps(findings, sort_keys=True))
    return {"findings": [], "forbidden_metadata_leakage": 0, "ok": True}


def _metadata_token_fingerprints(value: Any) -> set[str]:
    allowed = {token.casefold() for values in taxonomy().values() for token in values}
    result: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            for token in _TOKEN_RE.findall(item.casefold()):
                if token not in allowed:
                    result.add(hashlib.sha256(token.encode("utf-8")).hexdigest())

    walk(value)
    return result


def _approved_prompt_tokens() -> set[str]:
    static_value = {
        "confidence_levels": CONFIDENCE_LEVELS,
        "field_names": FIELD_NAMES,
        "field_states": FIELD_STATES,
        "instructions": blind_prompt(),
        "output_schema": output_schema(),
        "pixel_fact_vocabulary": (
            "alpha_bounds",
            "alpha_levels",
            "blob_id",
            "connected_components_4",
            "count",
            "decoded_dimensions",
            "fully_binary_alpha",
            "geometry_family_id",
            "horizontal",
            "integer_pixel_grid",
            "occupancy",
            "opaque_pixels",
            "palette_rgba",
            "palette_size",
            "pixel_art_structure",
            "rgba",
            "schema_version",
            "stored_without_upscale",
            "symmetry",
            "tight_dimensions",
            "vertical",
            "visible_pixels",
        ),
        "taxonomy": taxonomy(),
    }
    serialized = canonical_json_bytes(static_value).decode("utf-8").casefold()
    return set(_TOKEN_RE.findall(serialized))


def _decode_canonical_rgba(payload: bytes, *, width: int, height: int) -> np.ndarray:
    prefix = BLOB_ID_VERSION.encode("ascii") + b"\0" + struct.pack(">II", width, height)
    expected_length = len(prefix) + width * height * 4
    if len(payload) != expected_length or not payload.startswith(prefix):
        raise CodexBlindError("canonical RGBA blob encoding or dimensions are invalid")
    return np.frombuffer(payload[len(prefix) :], dtype=np.uint8).reshape((height, width, 4)).copy()


def _assert_unique(values: Iterable[str]) -> None:
    observed: set[str] = set()
    for value in values:
        if value in observed:
            raise CodexBlindError(f"duplicate opaque record_id: {value}")
        observed.add(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexBlindError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise CodexBlindError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_command(args: argparse.Namespace) -> dict[str, Any]:
    return stage_campaign(
        args.raw_experiment,
        args.output,
        model_display=args.model_display,
        session_id=args.session_id,
    )


def _prepare_command(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_pass(args.output, args.pass_id)


def _validate_batch_command(args: argparse.Namespace) -> dict[str, Any]:
    return validate_batch(args.output, args.pass_id, args.shard_id)


def _claim_command(args: argparse.Namespace) -> dict[str, Any]:
    return claim_shards(args.output, args.pass_id, args.owner, args.shard)


def _ingest_command(args: argparse.Namespace) -> dict[str, Any]:
    return ingest_compact_labels(
        args.output,
        args.pass_id,
        args.shard_id,
        model_display=args.model_display,
        session_id=args.session_id,
    )


def _freeze_command(args: argparse.Namespace) -> dict[str, Any]:
    return freeze_pass(args.output, args.pass_id, allow_partial=args.allow_partial)


def _reconcile_command(args: argparse.Namespace) -> dict[str, Any]:
    return reconcile_campaign(args.output)


def _prepare_health_command(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_health_check(args.output, args.milestone)


def _validate_health_command(args: argparse.Namespace) -> dict[str, Any]:
    return validate_health_batch(args.output, args.milestone)


def _ingest_health_command(args: argparse.Namespace) -> dict[str, Any]:
    return ingest_health_compact(
        args.output,
        args.milestone,
        model_display=args.model_display,
        session_id=args.session_id,
    )


def _source_reconcile_command(args: argparse.Namespace) -> dict[str, Any]:
    return reconcile_source_metadata(args.output, args.raw_experiment)


def _stop_command(args: argparse.Namespace) -> dict[str, Any]:
    return checkpoint_stop(args.output, args.reason)


def _resume_command(args: argparse.Namespace) -> dict[str, Any]:
    return resume_status(args.output)


def _finalize_command(args: argparse.Namespace) -> dict[str, Any]:
    return finalize_campaign(args.output, args.raw_experiment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.dataset_v5.codex_blind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage", help="Build a fail-closed opaque blind staging campaign.")
    stage.add_argument("--raw-experiment", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--model-display", required=True)
    stage.add_argument("--session-id", default=os.environ.get("CODEX_THREAD_ID", "unavailable"))
    stage.set_defaults(handler=_stage_command)
    prepare = subparsers.add_parser("prepare-pass", help="Create deterministic audited shards and contact sheets.")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--pass-id", choices=("A", "B"), required=True)
    prepare.set_defaults(handler=_prepare_command)
    validate = subparsers.add_parser("validate-batch", help="Fail closed immediately before visual inspection.")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--pass-id", choices=("A", "B"), required=True)
    validate.add_argument("--shard-id", required=True)
    validate.set_defaults(handler=_validate_batch_command)
    claim = subparsers.add_parser("claim-shards", help="Assign disjoint shard ownership to one Codex session.")
    claim.add_argument("--output", type=Path, required=True)
    claim.add_argument("--pass-id", choices=("A", "B"), required=True)
    claim.add_argument("--owner", required=True)
    claim.add_argument("--shard", action="append", required=True)
    claim.set_defaults(handler=_claim_command)
    ingest = subparsers.add_parser("ingest-compact", help="Expand and checkpoint one visually labeled shard.")
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--pass-id", choices=("A", "B"), required=True)
    ingest.add_argument("--shard-id", required=True)
    ingest.add_argument("--model-display", required=True)
    ingest.add_argument("--session-id", default=os.environ.get("CODEX_THREAD_ID", "unavailable"))
    ingest.set_defaults(handler=_ingest_command)
    freeze = subparsers.add_parser("freeze-pass", help="Freeze completed shard membership and JSONL hashes.")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--pass-id", choices=("A", "B"), required=True)
    freeze.add_argument("--allow-partial", action="store_true")
    freeze.set_defaults(handler=_freeze_command)
    reconcile = subparsers.add_parser("reconcile", help="Conservatively reconcile frozen Pass A and Pass B.")
    reconcile.add_argument("--output", type=Path, required=True)
    reconcile.set_defaults(handler=_reconcile_command)
    health = subparsers.add_parser("prepare-health-check", help="Prepare a fresh 20-record blind audit.")
    health.add_argument("--output", type=Path, required=True)
    health.add_argument("--milestone", type=int, required=True)
    health.set_defaults(handler=_prepare_health_command)
    health_validate = subparsers.add_parser(
        "validate-health-batch", help="Fail closed immediately before a fresh health audit."
    )
    health_validate.add_argument("--output", type=Path, required=True)
    health_validate.add_argument("--milestone", type=int, required=True)
    health_validate.set_defaults(handler=_validate_health_command)
    health_ingest = subparsers.add_parser("ingest-health-compact", help="Checkpoint and evaluate a fresh blind audit.")
    health_ingest.add_argument("--output", type=Path, required=True)
    health_ingest.add_argument("--milestone", type=int, required=True)
    health_ingest.add_argument("--model-display", required=True)
    health_ingest.add_argument("--session-id", default=os.environ.get("CODEX_THREAD_ID", "unavailable"))
    health_ingest.set_defaults(handler=_ingest_health_command)
    source = subparsers.add_parser("source-reconcile", help="Load provenance only after blind freeze.")
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--raw-experiment", type=Path, required=True)
    source.set_defaults(handler=_source_reconcile_command)
    stop = subparsers.add_parser("checkpoint-stop", help="Record a resumable terminal campaign boundary.")
    stop.add_argument("--output", type=Path, required=True)
    stop.add_argument("--reason", choices=("active_context_unreliable", "health_gate", "usage_limit"), required=True)
    stop.set_defaults(handler=_stop_command)
    resume = subparsers.add_parser("resume", help="Report the exact next blind record and shard.")
    resume.add_argument("--output", type=Path, required=True)
    resume.set_defaults(handler=_resume_command)
    finalize = subparsers.add_parser("finalize", help="Write supervision, distributions, reports, and hashes.")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--raw-experiment", type=Path, required=True)
    finalize.set_defaults(handler=_finalize_command)
    args = parser.parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

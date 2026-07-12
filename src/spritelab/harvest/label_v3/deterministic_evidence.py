"""Auto-Labeling v3: deterministic evidence producers.

Extracts immutable evidence from file provenance, filenames, source profiles,
sheet mappings, visual facts, and pack structure. All evidence is versioned
and carries a stage hash for cache identity.

No VLM/LLM calls. Deterministic: same inputs = same outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v3.evidence import (
    SCHEMA_VERSION,
    CalibrationStratum,
    EvidenceFamily,
    EvidenceItem,
    TargetField,
)
from spritelab.harvest.label_v3.sha256_utils import stable_evidence_id

logger = logging.getLogger(__name__)

STAGE_NAME = "deterministic_evidence"
STAGE_HASH = "d3t3rm_evid_v3_1"


@dataclass(frozen=True)
class DeterministicEvidenceBatch:
    sprite_id: str
    provenance_evidence: EvidenceItem | None = None
    filename_evidence: EvidenceItem | None = None
    filename_value_evidence: EvidenceItem | None = None
    source_profile_evidence: EvidenceItem | None = None
    sheet_mapping_evidence: EvidenceItem | None = None
    visual_facts_evidence: EvidenceItem | None = None
    palette_evidence: EvidenceItem | None = None
    alpha_integrity_evidence: EvidenceItem | None = None
    blank_fragment_evidence: EvidenceItem | None = None
    coordinate_evidence: EvidenceItem | None = None
    source_archetype_evidence: EvidenceItem | None = None

    def all_evidence(self) -> tuple[EvidenceItem, ...]:
        return tuple(
            e
            for e in [
                self.provenance_evidence,
                self.filename_evidence,
                self.filename_value_evidence,
                self.source_profile_evidence,
                self.sheet_mapping_evidence,
                self.visual_facts_evidence,
                self.palette_evidence,
                self.alpha_integrity_evidence,
                self.blank_fragment_evidence,
                self.coordinate_evidence,
                self.source_archetype_evidence,
            ]
            if e is not None
        )


def extract_deterministic_evidence(
    record: Mapping[str, Any],
    *,
    run_dir: str | Path = "",
) -> DeterministicEvidenceBatch:
    sprite_id = str(record.get("sprite_id", ""))
    source_id = str(record.get("source_id", ""))
    pack_id = str(record.get("source_name", "") or record.get("pack_name", ""))
    relative_path = str(record.get("relative_path", "") or record.get("final_png_path", ""))
    filename = Path(relative_path).name if relative_path else ""

    # 1. File and archive provenance
    provenance = _build_provenance_evidence(
        sprite_id=sprite_id,
        source_id=source_id,
        pack_id=pack_id,
        relative_path=relative_path,
        filename=filename,
        record=record,
    )

    # 2. Filename and path evidence (tokens only — provenance).
    filename_evidence = _build_filename_evidence(
        sprite_id=sprite_id,
        source_id=source_id,
        pack_id=pack_id,
        filename=filename,
        relative_path=relative_path,
        record=record,
    )

    # 2b. Filename *value* evidence: category/object proposed by the v2
    # filename rules. Shares the "filename" dependency group with (2) so a VLM
    # that later sees the filename cannot be double-counted against it.
    filename_value_evidence = _build_filename_value_evidence(
        sprite_id=sprite_id,
        source_id=source_id,
        pack_id=pack_id,
        record=record,
    )

    # 3. Source profile evidence
    source_profile = _build_source_profile_evidence(
        sprite_id=sprite_id,
        source_id=source_id,
        pack_id=pack_id,
        record=record,
    )

    # 4. Sheet mapping evidence
    sheet_mapping = _build_sheet_mapping_evidence(
        sprite_id=sprite_id,
        pack_id=pack_id,
        record=record,
    )

    # 5. Visual facts (loaded if available)
    visual_facts = _build_visual_facts_evidence(
        sprite_id=sprite_id,
        record=record,
        run_dir=run_dir,
    )

    # 6. Palette and color evidence
    palette = _build_palette_evidence(
        sprite_id=sprite_id,
        record=record,
        run_dir=run_dir,
    )

    # 7. Alpha integrity
    alpha = _build_alpha_integrity_evidence(
        sprite_id=sprite_id,
        record=record,
        run_dir=run_dir,
    )

    # 8. Blank and fragment detection
    blank = _build_blank_fragment_evidence(
        sprite_id=sprite_id,
        visual_facts=visual_facts,
        record=record,
    )

    # 9. Coordinate evidence
    coordinate = _build_coordinate_evidence(
        sprite_id=sprite_id,
        record=record,
    )

    # 10. Source structural archetype
    archetype = _build_source_archetype_evidence(
        sprite_id=sprite_id,
        pack_id=pack_id,
        record=record,
    )

    return DeterministicEvidenceBatch(
        sprite_id=sprite_id,
        provenance_evidence=provenance,
        filename_evidence=filename_evidence,
        filename_value_evidence=filename_value_evidence,
        source_profile_evidence=source_profile,
        sheet_mapping_evidence=sheet_mapping,
        visual_facts_evidence=visual_facts,
        palette_evidence=palette,
        alpha_integrity_evidence=alpha,
        blank_fragment_evidence=blank,
        coordinate_evidence=coordinate,
        source_archetype_evidence=archetype,
    )


def _base_evidence(
    sprite_id: str,
    source_id: str,
    pack_id: str,
    family: EvidenceFamily,
    target_fields: tuple[TargetField, ...],
    *,
    deterministic: bool = True,
    calibration_stratum: CalibrationStratum = "uncalibrated",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sprite_id": sprite_id,
        "source_id": source_id,
        "pack_id": pack_id,
        "producer_stage": "deterministic",
        "deterministic": deterministic,
        "stochastic": False,
        "source_hints_exposed": False,
        "candidate_hints_exposed": False,
        "stage_hash": STAGE_HASH,
    }


def _eid(sprite_id: str, suffix: str) -> str:
    return stable_evidence_id(sprite_id, f"deterministic_{suffix}", STAGE_HASH)


def _build_provenance_evidence(
    sprite_id: str,
    source_id: str,
    pack_id: str,
    relative_path: str,
    filename: str,
    record: Mapping[str, Any],
) -> EvidenceItem:
    parts = _base_evidence(sprite_id, source_id, pack_id, "archive_path", ("category", "canonical_object"))
    auto_meta = record.get("auto_metadata") or {}
    sheet_mapping = auto_meta.get("sheet_mapping") or {}

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "provenance"),
        evidence_family="archive_path",
        proposed_value={
            "relative_path": relative_path,
            "filename": filename,
            "source_id": source_id,
            "pack_id": pack_id,
            "source_name": str(record.get("source_name", "")),
            "sheet_mapping_name": str(sheet_mapping.get("mapping_name", "")),
        },
        input_artifact_refs=(relative_path,),
        warnings=() if relative_path else ("missing_relative_path",),
    )


def _build_filename_evidence(
    sprite_id: str,
    source_id: str,
    pack_id: str,
    filename: str,
    relative_path: str,
    record: Mapping[str, Any],
) -> EvidenceItem:
    parts = _base_evidence(sprite_id, source_id, pack_id, "filename", ("category", "canonical_object"))

    tokens = _path_tokens(filename)
    path_tokens = _path_tokens(relative_path)

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "filename"),
        evidence_family="filename",
        proposed_value={
            "filename": filename,
            "filename_tokens": list(tokens),
            "path_tokens": list(path_tokens),
            "relative_path": relative_path,
        },
        input_artifact_refs=(relative_path,),
        raw_score=0.5 if tokens else 0.0,
    )


def _build_filename_value_evidence(
    sprite_id: str,
    source_id: str,
    pack_id: str,
    record: Mapping[str, Any],
) -> EvidenceItem | None:
    """Propose category/object VALUES from the deterministic v2 filename rules.

    Trust flows from the rule's own confidence, which already drops for
    unknown/echoed tokens. An unknown filename in an exact-trust profile
    therefore does NOT inherit exact trust — its raw_score reflects the low
    filename-rule confidence, and calibration gates any acceptance downstream.
    """
    try:
        from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
    except Exception:
        return None

    try:
        result = suggest_from_filename_v2(record)
    except Exception:
        return None

    suggestion = result.suggestion
    object_name = str(suggestion.object_name or "").strip()
    category = str(suggestion.category or "").strip()
    if not object_name and not category:
        return None

    # Distinguish a genuinely-known object from an echoed filename token so the
    # provenance is honest. A "known" object is one the v2 rules recognised
    # (reflected in the confidence_reason), not a raw token echoed back.
    reason = str(result.confidence_reason or "")
    reason_lower = reason.lower()
    object_known = bool(object_name) and "not recognized" not in reason_lower and "unknown" not in reason_lower

    profile = result.profile
    trust = str(profile.filename_trust)
    calibration_stratum: CalibrationStratum = "source_profile"
    raw_score = float(result.confidence)

    warnings: list[str] = []

    # Exact-trust profiles are sheet-based: the ONLY trustworthy object identity
    # is the exact sheet-cell map, not an arbitrary filename token. An unmapped
    # filename in such a profile must NOT inherit the profile's exact trust —
    # downgrade it to a low-trust candidate.
    sheet_cell_backed = "sheet-cell" in reason_lower
    if trust == "exact" and not sheet_cell_backed:
        object_known = False
        raw_score = min(raw_score, 0.35)
        warnings.append("exact_profile_filename_not_sheet_mapped")

    if not object_known:
        warnings.append("filename_object_not_recognized")

    parts = _base_evidence(
        sprite_id,
        source_id,
        pack_id,
        "filename",
        ("category", "canonical_object", "tags"),
        calibration_stratum=calibration_stratum,
    )

    proposed: dict[str, Any] = {
        "category": category,
        # Only propose a canonical_object value when the rules recognised it;
        # echoed tokens are still surfaced as a candidate but flagged.
        "canonical_object": object_name if object_known else "",
        "object_name": object_name if object_known else "",
        "candidate_object_names": [object_name] if object_name else [],
        "tags": list(suggestion.tags or ()),
        "filename_trust": trust,
        "object_known": object_known,
        "confidence_reason": reason,
    }

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "filename_value"),
        evidence_family="filename",
        target_fields=("category", "canonical_object", "tags"),
        proposed_value=proposed,
        raw_score=raw_score,
        calibration_stratum=calibration_stratum,
        dependency_group="filename",
        warnings=tuple(warnings),
    )


def _build_source_profile_evidence(
    sprite_id: str,
    source_id: str,
    pack_id: str,
    record: Mapping[str, Any],
) -> EvidenceItem:
    from spritelab.harvest.source_profiles import detect_source_profile

    profile = detect_source_profile(record)
    parts = _base_evidence(
        sprite_id,
        source_id,
        pack_id,
        "source_profile",
        ("category", "canonical_object", "domain"),
        calibration_stratum="source_profile",
    )

    profile_trust = profile.filename_trust
    trusted = profile_trust == "exact"
    is_generic = profile.name == "generic_unknown"

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "source_profile"),
        evidence_family="source_profile",
        proposed_value={
            "profile_name": profile.name,
            "domain": profile.domain,
            "filename_trust": str(profile_trust),
            "expected_category_bias": list(profile.expected_category_bias),
            "is_trusted": trusted,
            "is_generic": is_generic,
        },
        raw_score=0.9 if trusted else 0.4 if profile_trust == "prefix_family" else 0.1,
        calibration_stratum="source_profile",
        dependency_group="source_profile",
        warnings=("generic_unknown_profile",) if is_generic else (),
    )


def _build_sheet_mapping_evidence(
    sprite_id: str,
    pack_id: str,
    record: Mapping[str, Any],
) -> EvidenceItem:
    from spritelab.harvest.source_profiles import detect_source_profile

    parts = _base_evidence(sprite_id, "", pack_id, "declarative_sheet_mapping", ("category", "canonical_object"))

    auto_meta = record.get("auto_metadata") or {}
    sheet_mapping = auto_meta.get("sheet_mapping") or {}
    profile = detect_source_profile(record)

    has_mapping = bool(sheet_mapping)
    mapping_valid = has_mapping and sheet_mapping.get("mapping_excluded") != "true"
    mapped_object = str(sheet_mapping.get("object_name", ""))
    mapped_category = str(sheet_mapping.get("category", ""))
    mapped_material = str(sheet_mapping.get("material", ""))

    requires_mapping = profile.name in {"shade_weapons", "flare_armor", "farming_tools"}

    warnings: list[str] = []
    if requires_mapping and not has_mapping:
        warnings.append("required_sheet_mapping_missing")
    if requires_mapping and has_mapping and not mapping_valid:
        warnings.append("sheet_mapping_excluded")

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "sheet_mapping"),
        evidence_family="declarative_sheet_mapping",
        proposed_value={
            "object_name": mapped_object,
            "category": mapped_category,
            "material": mapped_material,
            "mapping_name": str(sheet_mapping.get("mapping_name", "")),
            "mapping_valid": mapping_valid,
            "requires_mapping": requires_mapping,
        },
        raw_score=0.96 if mapping_valid else 0.0,
        dependency_group="sheet_mapping",
        warnings=tuple(warnings),
    )


def _build_visual_facts_evidence(
    sprite_id: str,
    record: Mapping[str, Any],
    *,
    run_dir: str | Path = "",
) -> EvidenceItem | None:
    parts = _base_evidence(
        sprite_id, "", "", "deterministic_visual", ("category", "canonical_object", "color", "shape", "material")
    )

    # Try to load existing visual_facts from record
    visual_facts = record.get("visual_facts") or {}
    if not isinstance(visual_facts, dict):
        visual_facts = {}

    # Try to extract from PNG if available
    image_path = _resolve_image_path(record, run_dir=run_dir)
    if image_path and Path(image_path).exists():
        try:
            from spritelab.harvest.visual_facts import extract_visual_facts_from_png, visual_facts_to_json

            facts = extract_visual_facts_from_png(Path(image_path))
            visual_facts = visual_facts_to_json(facts) or {}
        except Exception:
            pass

    if not visual_facts:
        return None

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "visual_facts"),
        evidence_family="deterministic_visual",
        target_fields=("category", "canonical_object", "color", "shape", "material"),
        proposed_value=visual_facts,
        raw_score=0.8,
    )


def _build_palette_evidence(
    sprite_id: str,
    record: Mapping[str, Any],
    *,
    run_dir: str | Path = "",
) -> EvidenceItem | None:
    parts = _base_evidence(sprite_id, "", "", "color_palette", ("color", "material"))

    visual_facts = record.get("visual_facts") or {}
    if not isinstance(visual_facts, dict):
        visual_facts = {}

    dominant_colors = visual_facts.get("dominant_colors") or ()
    palette_size = visual_facts.get("palette_size") or 0

    if not dominant_colors and not palette_size:
        return None

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "palette"),
        evidence_family="color_palette",
        target_fields=("color", "material"),
        proposed_value={
            "dominant_colors": list(dominant_colors),
            "palette_size": palette_size,
        },
        raw_score=0.7 if dominant_colors else 0.0,
    )


def _build_alpha_integrity_evidence(
    sprite_id: str,
    record: Mapping[str, Any],
    *,
    run_dir: str | Path = "",
) -> EvidenceItem | None:
    parts = _base_evidence(sprite_id, "", "", "deterministic_visual", ("canonical_object",))

    image_path = _resolve_image_path(record, run_dir=run_dir)
    if not image_path or not Path(image_path).exists():
        return None

    try:
        from PIL import Image

        with Image.open(image_path) as img:
            rgba = img.convert("RGBA")
        import numpy as np

        alpha = np.asarray(rgba)[:, :, 3]
        alpha_values = {int(v) for v in np.unique(alpha)}
        alpha_hard = alpha_values <= {0, 255}
        opaque_count = int(np.count_nonzero(alpha))
        has_soft_alpha = not alpha_hard and alpha_values != {0, 255}
    except Exception:
        return None

    warnings: list[str] = []
    if has_soft_alpha:
        warnings.append("soft_alpha_detected")
    if opaque_count == 0:
        warnings.append("fully_transparent")

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "alpha_integrity"),
        evidence_family="deterministic_visual",
        target_fields=("canonical_object",),
        proposed_value={
            "alpha_hard": alpha_hard,
            "has_soft_alpha": has_soft_alpha,
            "opaque_pixels": opaque_count,
        },
        raw_score=1.0 if alpha_hard else 0.3,
        warnings=tuple(warnings),
    )


def _build_blank_fragment_evidence(
    sprite_id: str,
    visual_facts: EvidenceItem | None,
    record: Mapping[str, Any],
) -> EvidenceItem | None:
    parts = _base_evidence(sprite_id, "", "", "deterministic_visual", ("canonical_object",))

    content_width = 0
    content_height = 0
    shape_hints: tuple[str, ...] = ()
    if visual_facts is not None and isinstance(visual_facts.proposed_value, dict):
        vf = visual_facts.proposed_value
        content_width = int(vf.get("content_width", 0))
        content_height = int(vf.get("content_height", 0))
        shape_hints = tuple(str(v) for v in vf.get("shape_hints", ()))

    is_empty = content_width == 0 or content_height == 0
    is_small = content_width <= 3 or content_height <= 3
    is_fragment = "small_content" in shape_hints
    is_blank = "empty" in shape_hints

    reasons: list[str] = []
    if is_blank or is_empty:
        reasons.append("blank_or_empty_sprite")
    if is_small or is_fragment:
        reasons.append("below_minimum_information")

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "blank_fragment"),
        evidence_family="deterministic_visual",
        target_fields=("canonical_object",),
        proposed_value={
            "is_empty": is_empty or is_blank,
            "is_small_fragment": is_small or is_fragment,
            "content_width": content_width,
            "content_height": content_height,
        },
        raw_score=1.0 if not (is_empty or is_blank or is_small or is_fragment) else 0.0,
        warnings=tuple(reasons),
    )


def _build_coordinate_evidence(
    sprite_id: str,
    record: Mapping[str, Any],
) -> EvidenceItem | None:
    parts = _base_evidence(sprite_id, "", "", "sheet_coordinate", ("canonical_object",))

    coord = record.get("sheet_coordinate") or record.get("grid_position")
    if coord is None:
        return None

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "coordinate"),
        evidence_family="sheet_coordinate",
        target_fields=("canonical_object",),
        proposed_value=coord if isinstance(coord, dict) else {"position": str(coord)},
        raw_score=0.3,
    )


def _build_source_archetype_evidence(
    sprite_id: str,
    pack_id: str,
    record: Mapping[str, Any],
) -> EvidenceItem | None:
    parts = _base_evidence(sprite_id, "", pack_id, "source_profile", ("domain",))

    struct_type = str(record.get("source_structure_type", "unknown"))
    if struct_type == "unknown":
        return None

    return EvidenceItem(
        **parts,
        evidence_id=_eid(sprite_id, "archetype"),
        evidence_family="source_profile",
        target_fields=("domain",),
        proposed_value={"source_archetype": struct_type},
        raw_score=0.5,
    )


def _path_tokens(path_str: str) -> list[str]:
    cleaned = str(path_str).lower()
    for sep in ("/", "\\", "-", ".", " "):
        cleaned = cleaned.replace(sep, "_")
    return [t for t in cleaned.split("_") if t and len(t) > 0]


def _resolve_image_path(record: Mapping[str, Any], *, run_dir: str | Path = "") -> str | None:
    raw = str(record.get("final_png_path", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return raw
    if run_dir:
        candidate = Path(run_dir) / path
        if candidate.exists():
            return str(candidate)
    return raw if Path(raw).exists() else None

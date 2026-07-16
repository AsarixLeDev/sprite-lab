"""Resumable, zero-configuration folder intake built on Dataset-v5 backends."""

from __future__ import annotations

import copy
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.data.dedupe_report import average_hash_image, hamming_distance_hex
from spritelab.dataset_v5.identity import canonical_json_bytes, decoded_rgba_sha256
from spritelab.dataset_v5.raw_extraction import ExtractionTransform, RawExtractionSpec, build_raw_extraction
from spritelab.dataset_v5.raw_inventory import RawSourceRecord, file_sha256
from spritelab.harvest.suitability import SuitabilityInput, audit_sprite, load_config
from spritelab.hierarchical_labeling.product import prepare_configured_labeling
from spritelab.product_core import (
    ProductBlocker,
    ProductCapability,
    ProductResult,
    ProductStatus,
    ProductWarning,
    ProjectContext,
    VisionProvider,
)
from spritelab.product_features.dataset.evidence import evidence_digest_payload, evidence_for_image
from spritelab.product_features.dataset.packs import SourcePack, detect_packs
from spritelab.product_features.dataset.semantics import propose_semantics
from spritelab.product_features.dataset.sheets import (
    EXTRACTION_POLICY_VERSION,
    plan_is_usable,
    propose_sheet_plan,
    uniform_grid_plan,
)
from spritelab.product_features.dataset.sidecar import (
    PackMetadataError,
    effective_pack_evidence,
    ensure_dataset_writes_outside_input,
    load_grouping,
    load_metadata_snapshot,
    pack_source_binding,
    sidecar_is_applicable,
)

INTAKE_SCHEMA = "spritelab.dataset.intake.v1"
STATE_SCHEMA = "spritelab.dataset.intake_state.v1"
REPORT_SCHEMA = "spritelab.dataset.report_data.v1"
SHEET_DECISIONS_SCHEMA = "spritelab.dataset.sheet_decisions.v2"
DISPOSITIONS = (
    "accepted",
    "rejected",
    "uncertain",
    "quarantined",
    "requires_special_extraction",
    "sheet_split",
)
NEAR_DUPLICATE_THRESHOLD = 8
_NON_RESCUABLE = frozenset(
    {
        "missing_source",
        "conflicting_source_evidence",
        "conflicting_license_evidence",
        "missing_creator",
        "missing_pack_title",
        "ambiguous_pack_boundary",
        "missing_license",
        "unverified_license",
        "unreadable",
        "blank",
        "unsupported_animation",
        "unresolved_sheet",
        "duplicate",
    }
)


class DatasetInputError(ValueError):
    """A user input can be corrected without a traceback."""


class DatasetImportInterrupted(InterruptedError):
    """Synthetic or real interruption after resumable state was persisted."""


@dataclass(frozen=True)
class DatasetBuildLocation:
    input_root: Path
    output_root: Path


class DatasetIntakeService:
    """The feature service; optional vision is injected through its public protocol."""

    def __init__(self, vision_provider: VisionProvider | None = None) -> None:
        self.vision_provider = vision_provider

    def build(
        self,
        folder: str | Path,
        *,
        output_root: str | Path | None = None,
        context: ProjectContext | None = None,
        interrupt_after: int | None = None,
    ) -> ProductResult:
        try:
            location = _resolve_locations(folder, output_root)
            return self._build(location, context=context, interrupt_after=interrupt_after)
        except DatasetInputError as exc:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="dataset",
                message=f"DATASET INPUT NEEDS INFORMATION\n\n{exc}",
                blockers=(ProductBlocker("invalid_dataset_input", str(exc)),),
                data={"schema_version": INTAKE_SCHEMA, "processed": 0},
            )

    def _build(
        self,
        location: DatasetBuildLocation,
        *,
        context: ProjectContext | None,
        interrupt_after: int | None,
    ) -> ProductResult:
        root, output = location.input_root, location.output_root
        paths = _discover_pngs(root)
        if not paths:
            raise DatasetInputError(f'No PNG files were found under "{root}".')
        effective_context = context or ProjectContext(project_root=Path.cwd(), config={})
        try:
            ensure_dataset_writes_outside_input(
                effective_context.project_root,
                root,
                output_root=output,
                runs_directory=effective_context.runs_directory,
            )
        except PackMetadataError as exc:
            raise DatasetInputError(str(exc)) from exc
        output.mkdir(parents=True, exist_ok=True)
        grouping, sidecars = load_metadata_snapshot(effective_context.project_root, root)
        packs = detect_packs(root, paths, user_grouping=grouping)
        initial_pack_bindings = _source_pack_bindings(root, packs)
        pack_by_image = {relative: pack for pack in packs for relative in pack.image_relative_paths}
        applicable_sidecars = {
            pack.pack_id: sidecars[pack.pack_id]
            for pack in packs
            if pack.pack_id in sidecars and sidecar_is_applicable(sidecars[pack.pack_id], pack, root)
        }
        state_path = output / "preprocessing_state.json"
        state = _load_state(state_path, root)
        journal_path = output / "preprocessing_journal.jsonl"
        prior_items = dict(state.get("items", {})) | _load_journal(journal_path)
        sheet_decisions = _load_sheet_decisions(output)
        patterns = _load_patterns(root)
        labels = _load_labels(root)
        groups = _load_groups(root)
        current: dict[str, dict[str, Any]] = {}
        reused = 0
        processed_now = 0
        state.update(
            {
                "schema_version": STATE_SCHEMA,
                "input_root": str(root),
                "output_root": str(output),
                "completed": False,
            }
        )
        _write_json_atomic(state_path, state)
        for path in paths:
            relative = path.relative_to(root).as_posix()
            byte_hash = file_sha256(path)
            pack = pack_by_image.get(relative)
            source, license_record = _effective_evidence(
                path, root, pack, applicable_sidecars.get(pack.pack_id) if pack else None
            )
            stored_decision = sheet_decisions.get(_item_id_for(relative))
            signature = _item_signature(relative, byte_hash, source, license_record, patterns, stored_decision)
            cached = prior_items.get(relative)
            reused_item = isinstance(cached, Mapping) and cached.get("preprocessing_signature") == signature
            if reused_item:
                item = copy.deepcopy(dict(cached))
                item["source_path"] = str(path)
                reused += 1
            else:
                item = _inspect_item(path, root, byte_hash)
                item["source"] = source
                item["license"] = license_record
                item["preprocessing_signature"] = signature
                processed_now += 1
            item["source"] = source
            item["license"] = license_record
            item["policy_excluded"] = _policy_excluded(relative, patterns)
            item["_human_labels"] = _labels_for_item(relative, labels)
            item["groups"] = _groups_for_item(relative, groups)
            if pack is not None:
                item["pack_id"] = pack.pack_id
                item["pack_relative_root"] = pack.relative_root
                item["pack_boundary_status"] = pack.boundary_status
            decision = stored_decision if _sheet_decision_is_applicable(stored_decision, item) else None
            applicable_signature = _item_signature(relative, byte_hash, source, license_record, patterns, decision)
            if applicable_signature != signature:
                if reused_item:
                    item = _inspect_item(path, root, byte_hash)
                    item["source"] = source
                    item["license"] = license_record
                    item["policy_excluded"] = _policy_excluded(relative, patterns)
                    item["_human_labels"] = _labels_for_item(relative, labels)
                    item["groups"] = _groups_for_item(relative, groups)
                    if pack is not None:
                        item["pack_id"] = pack.pack_id
                        item["pack_relative_root"] = pack.relative_root
                        item["pack_boundary_status"] = pack.boundary_status
                    reused -= 1
                    processed_now += 1
                    reused_item = False
                item["preprocessing_signature"] = applicable_signature
            current[relative] = item
            for child_relative, child in _sheet_children(item, path, decision, prior_items, output):
                child["source"] = source
                child["license"] = license_record
                child["policy_excluded"] = item["policy_excluded"]
                child["_human_labels"] = None
                child["groups"] = {}
                if pack is not None:
                    child["pack_id"] = pack.pack_id
                    child["pack_relative_root"] = pack.relative_root
                    child["pack_boundary_status"] = pack.boundary_status
                current[child_relative] = child
                if child.get("_fresh"):
                    child.pop("_fresh", None)
                    _append_journal(journal_path, child_relative, child)
            if not reused_item:
                _append_journal(journal_path, relative, item)
            if interrupt_after is not None and processed_now >= interrupt_after:
                raise DatasetImportInterrupted(
                    f"Import interrupted after {processed_now} newly processed item(s); resumable state was preserved."
                )
        deleted = sorted(set(prior_items) - set(current))
        items = [current[key] for key in sorted(current)]
        _finalize_dispositions(items, prior_items)
        _apply_near_duplicate_flags(items)

        def revalidate_source() -> None:
            _revalidate_source_snapshot(
                items,
                root,
                effective_context.project_root,
                grouping,
                initial_pack_bindings,
            )

        revalidate_source()
        semantic = propose_semantics(items, self.vision_provider, effective_context)
        revalidate_source()
        hierarchical = prepare_configured_labeling(
            items,
            config=effective_context.config,
            output_root=output,
        )
        revalidate_source()
        rebuild_raw_extraction(output, items, validate_before_publish=revalidate_source)
        pack_summary = _pack_summary(packs, applicable_sidecars, items)
        summary = _summary(items, semantic, hierarchical=hierarchical, pack_summary=pack_summary)
        result = _product_result(root, output, items, summary, reused=reused, deleted=deleted)
        _write_outputs(output, root, items, summary, result, reused=reused, deleted=deleted)
        state.update(
            {
                "completed": True,
                "items": {item["relative_path"]: item for item in items},
                "deleted_since_previous_run": deleted,
                "last_summary": summary,
                "reused_preprocessing_count": reused,
            }
        )
        _write_json_atomic(state_path, state)
        journal_path.unlink(missing_ok=True)
        return result


def build_dataset(
    folder: str | Path,
    *,
    output_root: str | Path | None = None,
    vision_provider: VisionProvider | None = None,
    context: ProjectContext | None = None,
    interrupt_after: int | None = None,
) -> ProductResult:
    """Convenience API used by the plugin and tests."""

    return DatasetIntakeService(vision_provider).build(
        folder,
        output_root=output_root,
        context=context,
        interrupt_after=interrupt_after,
    )


def inspect_dataset_folder(
    folder: str | Path,
    *,
    context: ProjectContext | None = None,
) -> dict[str, Any]:
    """Inspect pack boundaries and metadata needs without preprocessing or provider use."""

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise DatasetInputError(f'Dataset folder does not exist or is not a directory: "{root}"')
    paths = _discover_pngs(root)
    effective_context = context or ProjectContext(project_root=Path.cwd(), config={})
    grouping, sidecars = load_metadata_snapshot(effective_context.project_root, root)
    packs = detect_packs(root, paths, user_grouping=grouping)
    rows: list[dict[str, Any]] = []
    for pack in packs:
        record = sidecars.get(pack.pack_id)
        applicable = bool(record and sidecar_is_applicable(record, pack, root))
        first = root / pack.image_relative_paths[0]
        source, license_record = _effective_evidence(first, root, pack, record if applicable else None)
        missing = _pack_missing_fields(source, license_record, pack, sidecar_applied=applicable)
        rows.append(
            {
                **pack.to_dict(),
                "sidecar_applied": applicable,
                "sidecar_stale": bool(record) and not applicable,
                "missing_fields": missing,
                "wizard_complete": not missing,
                "training_eligible_evidence": not missing and bool(license_record.get("training_allowed")),
                "quarantined_by_declaration": not bool(license_record.get("training_allowed")) and not missing,
                "source": source,
                "license": license_record,
            }
        )
    source_ready = bool(rows) and all(bool(row["source"].get("present")) for row in rows)
    license_ready = bool(rows) and all(bool(row["license"].get("present")) for row in rows)
    return {
        "schema_version": "spritelab.dataset.intake_inspection.v1",
        "input_root": str(root),
        "image_count": len(paths),
        "pack_count": len(rows),
        "packs": rows,
        # Preserve the original folder-inspection contract for existing web clients.
        # Pack-level readiness and missing fields remain authoritative for the wizard.
        "source_ready": source_ready,
        "license_ready": license_ready,
        "next_action": "Build dataset" if not any(row["missing_fields"] for row in rows) else "Complete information",
        "wizard_required": any(row["missing_fields"] for row in rows),
        "grouping_confirmation_required": any(row["boundary_status"] == "needs_confirmation" for row in rows),
        "provider_contacted": False,
        "input_mutated": False,
    }


def discover_source_packs(
    folder: str | Path,
    *,
    context: ProjectContext | None = None,
) -> tuple[Path, list[Path], list[SourcePack]]:
    """Return recursive PNG discovery and conservative pack boundaries."""

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise DatasetInputError(f'Dataset folder does not exist or is not a directory: "{root}"')
    paths = _discover_pngs(root)
    effective_context = context or ProjectContext(project_root=Path.cwd(), config={})
    packs = detect_packs(root, paths, user_grouping=load_grouping(effective_context.project_root, root))
    return root, paths, packs


def rebuild_raw_extraction(
    output: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    validate_before_publish: Callable[[], None] | None = None,
) -> None:
    """Materialize current accepted records through the existing raw extraction backend."""

    accepted = [item for item in items if item.get("current_disposition") == "accepted"]
    destination = output / "raw_extraction"
    candidate = output / ".raw_extraction.next"
    backup = output / ".raw_extraction.previous"
    for stale in (candidate, backup):
        if stale.exists():
            shutil.rmtree(stale)
    if not accepted:
        if validate_before_publish is not None:
            validate_before_publish()
        if destination.exists():
            shutil.rmtree(destination)
        return
    specs = [_raw_spec(item) for item in accepted]
    try:
        build_raw_extraction(specs, candidate)
        if validate_before_publish is not None:
            validate_before_publish()
    except BaseException:
        if candidate.exists():
            shutil.rmtree(candidate)
        raise
    try:
        if destination.exists():
            destination.replace(backup)
        candidate.replace(destination)
        if validate_before_publish is not None:
            validate_before_publish()
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            backup.replace(destination)
        if candidate.exists():
            shutil.rmtree(candidate)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def recompute_summary_from_items(items: Sequence[Mapping[str, Any]], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Update disposition counts after review while preserving semantic evidence."""

    semantic = dict(previous.get("semantic", {}))
    hierarchical = dict(previous.get("hierarchical_labeling", {}))
    return _summary(
        list(items),
        semantic,
        hierarchical=hierarchical,
        pack_summary=previous.get("packs"),
    )


def terminal_message(root: Path, summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    missing_source = int(counts["missing_source"])
    missing_license = int(counts["missing_license"])
    missing_information = int(counts.get("missing_information", 0))
    if missing_information:
        lines = ["DATASET INPUT NEEDS INFORMATION", ""]
        if missing_source:
            lines.append(f"{missing_source:,} images have no source information.")
        if missing_license:
            lines.append(f"{missing_license:,} images have no license information.")
        lines.extend(
            [
                f"{missing_information:,} images are waiting for source or license information.",
                "[Complete information]",
                "",
                "Only affected packs were quarantined; valid independent packs remain usable.",
            ]
        )
    elif int(counts["accepted"]):
        lines = ["DATASET READY", ""]
    else:
        lines = ["DATASET NEEDS REVIEW", ""]
    lines.extend(
        [
            f"{counts['processed']:,} PNG files processed",
            "",
            f"Accepted automatically:       {counts['accepted_automatically']:,}",
            f"Extracted from sheets:         {counts['extracted_from_sheets']:,}",
            f"Exact duplicates removed:      {counts['exact_duplicates_removed']:,}",
            f"Rejected technically:          {counts['rejected_technically']:,}",
            f"Needs sheet review:            {counts['needs_sheet_review']:,}",
            f"Missing source/license:        {counts['missing_information']:,}",
            "",
            f"{counts['image_only_eligible']:,} images are ready for an image-only candidate dataset.",
            "",
            "Review excluded images later with:",
            "  python -m spritelab v3 review",
            "",
            "No source images were modified.",
        ]
    )
    return "\n".join(lines)


def preprocessing_prompt(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    return "\n".join(
        [
            f"{counts['processed']:,} PNG files processed",
            "",
            f"Accepted automatically:       {counts['accepted_automatically']:,}",
            f"Extracted from sheets:         {counts['extracted_from_sheets']:,}",
            f"Exact duplicates removed:      {counts['exact_duplicates_removed']:,}",
            f"Rejected technically:          {counts['rejected_technically']:,}",
            f"Needs sheet review:            {counts['needs_sheet_review']:,}",
            f"Missing source/license:        {counts['missing_information']:,}",
            "",
            f"{counts['image_only_eligible']:,} images are ready for an image-only candidate dataset.",
        ]
    )


def _resolve_locations(folder: str | Path, output_root: str | Path | None) -> DatasetBuildLocation:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise DatasetInputError(f'Dataset folder does not exist or is not a directory: "{root}"')
    if output_root is None:
        candidate = (Path.cwd() / "datasets" / f"{_safe_name(root.name)}-dataset").resolve()
        output = root.parent / f"{root.name}-spritelab-dataset" if _is_relative_to(candidate, root) else candidate
    else:
        output = Path(output_root).expanduser().resolve()
    if output == root or _is_relative_to(output, root) or _is_relative_to(root, output):
        raise DatasetInputError(
            "The output directory and input folder must not contain one another so source images stay unchanged."
        )
    return DatasetBuildLocation(root, output)


def _discover_pngs(root: Path) -> list[Path]:
    discovered: list[Path] = []
    boundary = root.resolve()

    def walk_error(exc: OSError) -> None:
        name = Path(str(exc.filename or "source entry")).name
        raise DatasetInputError(f'The approved source contains an unreadable subtree: "{name}".') from exc

    for directory, names, filenames in os.walk(root, onerror=walk_error, followlinks=False):
        directory_path = Path(directory)
        _require_source_entry_confined(directory_path, boundary, expected="directory")
        names.sort(key=str.casefold)
        for name in names:
            _require_source_entry_confined(directory_path / name, boundary, expected="directory")
        for filename in sorted(filenames, key=str.casefold):
            candidate = directory_path / filename
            _require_source_entry_confined(candidate, boundary, expected="file")
            if Path(filename).suffix.casefold() == ".png":
                discovered.append(candidate)
    return discovered


def _require_source_entry_confined(path: Path, boundary: Path, *, expected: str | None = None) -> None:
    try:
        status = path.lstat()
        is_reparse = bool(getattr(status, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        is_junction = bool(getattr(path, "is_junction", lambda: False)())
        if path != boundary and (path.is_symlink() or is_junction or is_reparse):
            raise ValueError("symbolic links and reparse points are not accepted")
        resolved = path.resolve(strict=True)
        resolved.relative_to(boundary)
        if expected == "file" and not stat.S_ISREG(status.st_mode):
            raise ValueError("source entry is not a regular file")
        if expected == "directory" and not stat.S_ISDIR(status.st_mode):
            raise ValueError("source entry is not a directory")
    except (OSError, RuntimeError, ValueError) as exc:
        try:
            display = path.relative_to(boundary).as_posix()
        except ValueError:
            display = path.name
        raise DatasetInputError(
            f'The approved source contains an unreadable, symbolic-link, junction, or reparse entry: "{display}". '
            "Linked entries are not accepted because they can cross source-pack provenance boundaries."
        ) from exc


def _item_id_for(relative: str) -> str:
    return "item_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def _inspect_item(path: Path, root: Path, byte_hash: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    item_id = _item_id_for(relative)
    base: dict[str, Any] = {
        "schema_version": INTAKE_SCHEMA,
        "item_id": item_id,
        "relative_path": relative,
        "filename": path.name,
        "source_path": str(path),
        "byte_sha256": byte_hash,
        "decoded_rgba_sha256": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "suitability": None,
        "technical_disposition": "rejected",
        "technical_reasons": [],
    }
    if path.name.startswith("._"):
        base.update(technical_disposition="rejected", technical_reasons=["policy_excluded"], appledouble=True)
        return base
    try:
        with Image.open(path) as opened:
            base["frame_count"] = int(getattr(opened, "n_frames", 1))
            base["format"] = str(opened.format or "unknown")
            if base["frame_count"] != 1:
                base.update(
                    technical_disposition="requires_special_extraction",
                    technical_reasons=["unsupported_animation"],
                )
                return base
            opened.load()
            rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        base.update(technical_disposition="rejected", technical_reasons=["unreadable"], decode_error=str(exc))
        return base
    height, width = rgba.shape[:2]
    base.update(width=width, height=height, decoded_rgba_sha256=decoded_rgba_sha256(rgba))
    if np.unique(rgba.reshape(-1, 4), axis=0).shape[0] == 1 or not np.any(rgba[:, :, 3]):
        base.update(technical_disposition="rejected", technical_reasons=["blank"])
        return base
    base["average_hash"] = average_hash_image(Image.fromarray(rgba, "RGBA"))
    suitability = audit_sprite(
        SuitabilityInput(sprite_id=item_id, image_path=path),
        load_config("single_object_source_resolution"),
    ).to_dict()
    base["suitability"] = suitability
    reasons = [_controlled_reason(code) for code in suitability["reason_codes"]]
    if "unresolved_sheet" in reasons:
        disposition = "requires_special_extraction"
        base["sheet_plan"] = propose_sheet_plan(rgba)
    elif suitability["status"] == "reject":
        disposition = "rejected"
    elif suitability["status"] == "quarantine":
        disposition = "uncertain"
    else:
        disposition = "accepted"
    base.update(technical_disposition=disposition, technical_reasons=sorted(set(reasons)))
    return base


def _effective_evidence(
    path: Path,
    root: Path,
    pack: SourcePack | None,
    sidecar_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Folder evidence stays authoritative; an applicable sidecar only fills gaps."""

    source, license_record = evidence_for_image(path, root)
    if pack is None or sidecar_record is None:
        return source, license_record
    sidecar_source, sidecar_license = effective_pack_evidence(sidecar_record, pack)
    if source.get("conflict"):
        sidecar_source["resolved_folder_conflict"] = {
            "evidence_records": source.get("evidence_records", []),
            "conflict_details": source.get("conflict_details", []),
        }
        source = sidecar_source
    elif not source.get("present"):
        source = sidecar_source
    else:
        merged_source = dict(source)
        for field in ("source_name", "creator", "source_url", "source_type"):
            if not merged_source.get(field) and sidecar_source.get(field):
                merged_source[field] = sidecar_source[field]
        if any(merged_source.get(field) != source.get(field) for field in merged_source):
            merged_source["interpretation"] = "user_evidence_with_sidecar_completion"
            merged_source["sidecar_path"] = sidecar_source["path"]
            merged_source["sidecar_evidence_sha256"] = sidecar_source["evidence_sha256"]
        source = merged_source
    if license_record.get("conflict"):
        sidecar_license["resolved_folder_conflict"] = {
            "evidence_records": license_record.get("evidence_records", []),
            "conflict_details": license_record.get("conflict_details", []),
        }
        license_record = sidecar_license
    elif not license_record.get("present") or (
        not license_record.get("training_allowed") and sidecar_license.get("training_allowed")
    ):
        license_record = sidecar_license
    return source, license_record


def _pack_missing_fields(
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    pack: SourcePack,
    *,
    sidecar_applied: bool,
) -> list[str]:
    """Identify declaration fields that require one pack-level wizard response."""

    missing: list[str] = []
    if source.get("conflict"):
        missing.append("conflicting_source_evidence")
    if license_record.get("conflict"):
        missing.append("conflicting_license_evidence")
    if pack.boundary_status == "needs_confirmation":
        missing.append("pack_boundary_confirmation")
    if not source.get("present"):
        missing.extend(
            (
                "creator_or_rights_holder",
                "pack_title",
                "source_type",
                "source_page_url_or_original_work_declaration",
            )
        )
    else:
        if not source.get("creator"):
            missing.append("creator_or_rights_holder")
        if not source.get("source_name"):
            missing.append("pack_title")
        if pack.prefill.get("source_type") in {"opengameart", "kenney"} and not source.get("source_url"):
            missing.append("source_page_url")
    if not license_record.get("present"):
        missing.append("license_identifier")
    elif not license_record.get("training_allowed") and not sidecar_applied:
        missing.append("license_identifier")
    return list(dict.fromkeys(missing))


def _load_sheet_decisions(output: Path) -> dict[str, dict[str, Any]]:
    path = output / "sheet_decisions.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, Mapping) or value.get("schema_version") != SHEET_DECISIONS_SCHEMA:
        return {}
    decisions = value.get("decisions")
    if not isinstance(decisions, Mapping):
        return {}
    return {str(key): dict(row) for key, row in decisions.items() if isinstance(row, Mapping)}


def save_sheet_decision(output: Path, item: Mapping[str, Any], decision: Mapping[str, Any]) -> Path:
    path = output / "sheet_decisions.json"
    decisions = _load_sheet_decisions(output)
    item_id = str(item.get("item_id") or "")
    if not item_id:
        raise DatasetInputError("A sheet decision requires a current item identity.")
    decisions[item_id] = {**dict(decision), "binding": _sheet_decision_binding(item)}
    _write_json_atomic(path, {"schema_version": SHEET_DECISIONS_SCHEMA, "decisions": decisions})
    return path


def _sheet_decision_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    proposal = item.get("sheet_plan") if isinstance(item.get("sheet_plan"), Mapping) else None
    return {
        "schema_version": "spritelab.dataset.sheet_decision_binding.v1",
        "item_id": str(item.get("item_id") or ""),
        "relative_path": str(item.get("relative_path") or ""),
        "source_byte_sha256": str(item.get("byte_sha256") or ""),
        "decoded_rgba_sha256": str(item.get("decoded_rgba_sha256") or ""),
        "pack_id": str(item.get("pack_id") or ""),
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "reviewed_proposal_sha256": hashlib.sha256(canonical_json_bytes(proposal)).hexdigest(),
    }


def _sheet_decision_is_applicable(decision: Mapping[str, Any] | None, item: Mapping[str, Any]) -> bool:
    if not isinstance(decision, Mapping) or not isinstance(decision.get("binding"), Mapping):
        return False
    return canonical_json_bytes(dict(decision["binding"])) == canonical_json_bytes(_sheet_decision_binding(item))


def _resolved_sheet_plan(
    item: Mapping[str, Any],
    rgba: np.ndarray,
    decision: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Choose the automatic plan, a user-kept proposal, or a user-adjusted grid."""

    proposal = item.get("sheet_plan") if isinstance(item.get("sheet_plan"), Mapping) else None
    action = str(decision.get("action")) if isinstance(decision, Mapping) else ""
    if action == "exclude_sheet":
        return None
    if action == "adjust_grid" and isinstance(decision, Mapping):
        grid = decision.get("grid") if isinstance(decision.get("grid"), Mapping) else {}
        try:
            return uniform_grid_plan(rgba, columns=int(grid.get("columns", 0)), rows=int(grid.get("rows", 0)))
        except (TypeError, ValueError):
            return None
    if action == "keep_proposal" and proposal is not None and len(proposal.get("proposed_crops") or ()) >= 2:
        confirmed = dict(proposal)
        confirmed["crops"] = [list(crop) for crop in proposal["proposed_crops"]]
        confirmed["unambiguous"] = True
        confirmed["ambiguity_reasons"] = []
        confirmed["separator_policy"] = "user_confirmed_proposal"
        return confirmed
    if proposal is not None and plan_is_usable(proposal):
        return dict(proposal)
    return None


def _sheet_children(
    item: dict[str, Any],
    path: Path,
    decision: Mapping[str, Any] | None,
    prior: Mapping[str, Any],
    output: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Deterministically extract frames from a resolvable sheet; never touch the source."""

    if "unresolved_sheet" not in item.get("technical_reasons", ()):
        return []
    action = str(decision.get("action")) if isinstance(decision, Mapping) else ""
    if action == "exclude_sheet":
        item["sheet_decision"] = "exclude_sheet"
        item["technical_disposition"] = "rejected"
        item["technical_reasons"] = sorted(
            (set(item.get("technical_reasons", ())) - {"unresolved_sheet"}) | {"sheet_excluded_by_user"}
        )
        item["review_confirmed"] = True
        return []
    proposal = item.get("sheet_plan")
    needs_pixels = action in {"adjust_grid", "keep_proposal"} or plan_is_usable(
        proposal if isinstance(proposal, Mapping) else None
    )
    if not needs_pixels:
        return []
    try:
        with Image.open(path) as opened:
            opened.load()
            rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return []
    plan = _resolved_sheet_plan(item, rgba, decision)
    if plan is None or not plan.get("crops"):
        return []
    item["sheet_plan_applied"] = {key: plan[key] for key in plan if key != "proposed_crops"}
    if action:
        item["sheet_decision"] = action
    plan_digest = hashlib.sha256(canonical_json_bytes(item["sheet_plan_applied"])).hexdigest()
    children: list[tuple[str, dict[str, Any]]] = []
    for index, crop in enumerate(plan["crops"]):
        left, top, right, bottom = (int(value) for value in crop)
        child_relative = f"{item['relative_path']}#frame{index:04d}"
        child_signature = hashlib.sha256(
            f"{item['preprocessing_signature']}\0{plan_digest}\0{index}".encode()
        ).hexdigest()
        cached = prior.get(child_relative)
        if isinstance(cached, Mapping) and cached.get("preprocessing_signature") == child_signature:
            child = copy.deepcopy(dict(cached))
            child["source_path"] = str(path)
            children.append((child_relative, child))
            continue
        cell = np.ascontiguousarray(rgba[top:bottom, left:right], dtype=np.uint8)
        blank = not np.any(cell[:, :, 3])
        output_identity = decoded_rgba_sha256(cell)
        suitability, disposition, reasons = _audit_extracted_cell(
            cell,
            item_id=f"{item['item_id']}-{plan_digest[:12]}-{index:04d}",
            output=output,
        )
        if blank:
            disposition, reasons = "rejected", ["blank"]
        child_item_id = (
            "item_"
            + hashlib.sha256(
                f"{item['item_id']}\0{plan_digest}\0{index}\0{left},{top},{right},{bottom}".encode()
            ).hexdigest()[:24]
        )
        child = {
            "schema_version": INTAKE_SCHEMA,
            "item_id": child_item_id,
            "relative_path": child_relative,
            "filename": f"{path.name}#frame{index:04d}",
            "source_path": str(path),
            "byte_sha256": hashlib.sha256(f"{item['byte_sha256']}\0{left},{top},{right},{bottom}".encode()).hexdigest(),
            "decoded_rgba_sha256": output_identity,
            "width": right - left,
            "height": bottom - top,
            "frame_count": 1,
            "suitability": suitability,
            "technical_disposition": disposition,
            "technical_reasons": reasons,
            "average_hash": None if blank else average_hash_image(Image.fromarray(cell, "RGBA")),
            "preprocessing_signature": child_signature,
            "sheet_extraction": {
                "source_item_id": item["item_id"],
                "source_relative_path": item["relative_path"],
                "source_byte_sha256": item["byte_sha256"],
                "source_decoded_rgba_sha256": item.get("decoded_rgba_sha256"),
                "crop_rectangle": [left, top, right, bottom],
                "frame_index": index,
                "output_decoded_rgba_sha256": output_identity,
                "extraction_policy_version": EXTRACTION_POLICY_VERSION,
                "source_sheet_modified": False,
            },
            "_fresh": True,
        }
        children.append((child_relative, child))
    if children:
        item["technical_disposition"] = "sheet_split"
        item["technical_reasons"] = sorted(
            (set(item.get("technical_reasons", ())) - {"unresolved_sheet"}) | {"sheet_extracted"}
        )
    return children


def _audit_extracted_cell(
    cell: np.ndarray,
    *,
    item_id: str,
    output: Path,
) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Run the existing suitability backend against a temporary derived crop."""

    audit_root = output / ".sheet-suitability-audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    path = audit_root / f"{item_id}.png"
    try:
        Image.fromarray(cell, "RGBA").save(path, format="PNG")
        suitability = audit_sprite(
            SuitabilityInput(sprite_id=item_id, image_path=path),
            load_config("single_object_source_resolution"),
        ).to_dict()
    finally:
        path.unlink(missing_ok=True)
        try:
            audit_root.rmdir()
        except OSError:
            pass
    reasons = sorted({_controlled_reason(code) for code in suitability["reason_codes"]})
    if "unresolved_sheet" in reasons:
        disposition = "requires_special_extraction"
    elif suitability["status"] == "reject":
        disposition = "rejected"
    elif suitability["status"] == "quarantine":
        disposition = "uncertain"
    else:
        disposition = "accepted"
    return suitability, disposition, reasons


def _apply_near_duplicate_flags(items: list[dict[str, Any]]) -> None:
    """Report possible near duplicates without silently deleting anything."""

    candidates = [
        item
        for item in items
        if item.get("current_disposition") == "accepted" and item.get("average_hash") and not item.get("duplicate_of")
    ]
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Exact Hamming-radius lookup through a BK tree avoids a quadratic all-pairs
    # pass for ordinary large imports while retaining deterministic results.
    tree: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        fingerprint = str(candidate["average_hash"])
        if not tree:
            tree.append({"fingerprint": fingerprint, "members": [index], "children": {}})
            continue
        pending = [0]
        while pending:
            node_index = pending.pop()
            node = tree[node_index]
            distance = hamming_distance_hex(fingerprint, str(node["fingerprint"]))
            if distance <= NEAR_DUPLICATE_THRESHOLD:
                for member in node["members"]:
                    parents[find(index)] = find(int(member))
            lower = max(0, distance - NEAR_DUPLICATE_THRESHOLD)
            upper = distance + NEAR_DUPLICATE_THRESHOLD
            pending.extend(int(child) for edge, child in node["children"].items() if lower <= int(edge) <= upper)
        node_index = 0
        while True:
            node = tree[node_index]
            distance = hamming_distance_hex(fingerprint, str(node["fingerprint"]))
            if distance == 0:
                node["members"].append(index)
                break
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = len(tree)
                tree.append({"fingerprint": fingerprint, "members": [index], "children": {}})
                break
            node_index = int(child)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    for members in groups.values():
        if len(members) < 2:
            continue
        group_id = (
            "near_"
            + hashlib.sha256(canonical_json_bytes(sorted(str(candidates[i]["item_id"]) for i in members))).hexdigest()[
                :16
            ]
        )
        for member in members:
            candidates[member]["possible_near_duplicate"] = True
            candidates[member]["near_duplicate_group"] = group_id


def _pack_summary(
    packs: Sequence[SourcePack],
    applicable_sidecars: Mapping[str, Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_pack: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        pack_id = str(item.get("pack_id") or "")
        if pack_id:
            by_pack.setdefault(pack_id, []).append(item)
    rows = []
    for pack in packs:
        members = [item for item in by_pack.get(pack.pack_id, []) if not item.get("sheet_extraction")]
        missing_reasons = {
            "ambiguous_pack_boundary",
            "conflicting_license_evidence",
            "conflicting_source_evidence",
            "missing_creator",
            "missing_license",
            "missing_pack_title",
            "missing_source",
        }
        if pack.pack_id not in applicable_sidecars:
            missing_reasons.add("unverified_license")
        missing = [item for item in members if missing_reasons & set(item.get("reasons", ()))]
        quarantined = [item for item in members if item.get("current_disposition") == "quarantined"]
        rows.append(
            {
                **pack.to_dict(),
                "sidecar_applied": pack.pack_id in applicable_sidecars,
                "images_missing_information": len(missing),
                "information_complete": not missing,
                "eligibility_state": ("quarantined" if quarantined else "eligible"),
            }
        )
    waiting = sum(row["images_missing_information"] for row in rows)
    return {
        "schema_version": "spritelab.dataset.pack_summary.v1",
        "packs": rows,
        "pack_count": len(rows),
        "packs_missing_information": sum(1 for row in rows if not row["information_complete"]),
        "images_missing_information": waiting,
        "packs_needing_grouping_confirmation": sum(1 for row in rows if row["boundary_status"] == "needs_confirmation"),
        "asked_once_per_pack": True,
    }


def _finalize_dispositions(items: list[dict[str, Any]], prior: Mapping[str, Any]) -> None:
    legal_reasons = {
        "ambiguous_pack_boundary",
        "conflicting_license_evidence",
        "conflicting_source_evidence",
        "missing_creator",
        "missing_license",
        "missing_pack_title",
        "missing_source",
        "unverified_license",
    }
    # Establish provenance and technical state before selecting exact-duplicate
    # canonicals. This ensures an incomplete pack cannot suppress an otherwise
    # valid independent pack merely because its path sorts first.
    for item in items:
        item.pop("duplicate_of", None)
        item.pop("duplicate_kind", None)
        item.pop("duplicate_associations", None)
        reasons = list(item.get("technical_reasons", ()))
        disposition = str(item["technical_disposition"])
        source, license_record = item["source"], item["license"]
        if source.get("conflict"):
            reasons.append("conflicting_source_evidence")
        if license_record.get("conflict"):
            reasons.append("conflicting_license_evidence")
        if not source.get("present"):
            reasons.append("missing_source")
        else:
            if not source.get("creator"):
                reasons.append("missing_creator")
            if not source.get("source_name"):
                reasons.append("missing_pack_title")
        if not license_record.get("present"):
            reasons.append("missing_license")
        elif not license_record.get("training_allowed"):
            reasons.append("unverified_license")
        if item.get("pack_boundary_status") == "needs_confirmation":
            reasons.append("ambiguous_pack_boundary")
        if item.get("policy_excluded"):
            reasons.append("policy_excluded")
            disposition = "rejected"
        item["_base_reasons"] = sorted(set(reasons))
        item["_base_disposition"] = disposition

    exact_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        decoded = item.get("decoded_rgba_sha256")
        key = ("decoded_rgba", str(decoded)) if decoded else ("bytes", str(item["byte_sha256"]))
        exact_groups.setdefault(key, []).append(item)

    for members in exact_groups.values():
        if len(members) < 2:
            continue

        def canonical_rank(candidate: Mapping[str, Any]) -> tuple[int, int, int, str]:
            reasons = set(candidate.get("_base_reasons", ()))
            return (
                int(bool(reasons & legal_reasons)),
                int("policy_excluded" in reasons),
                int(candidate.get("_base_disposition") != "accepted"),
                str(candidate["relative_path"]).casefold(),
            )

        canonical = min(members, key=canonical_rank)
        canonical_id = str(canonical["item_id"])
        seen_byte_hashes = {str(canonical["byte_sha256"])}
        for duplicate in sorted(
            (member for member in members if member is not canonical),
            key=lambda member: str(member["relative_path"]).casefold(),
        ):
            byte_hash = str(duplicate["byte_sha256"])
            duplicate["duplicate_of"] = canonical_id
            duplicate["duplicate_kind"] = (
                "duplicate_bytes" if byte_hash in seen_byte_hashes else "duplicate_decoded_rgba"
            )
            seen_byte_hashes.add(byte_hash)
            canonical.setdefault("duplicate_associations", []).append(
                {
                    "item_id": duplicate["item_id"],
                    "relative_path": duplicate["relative_path"],
                    "pack_id": duplicate.get("pack_id"),
                    "source": dict(duplicate["source"]),
                    "license": dict(duplicate["license"]),
                }
            )

    for item in items:
        relative = item["relative_path"]
        reasons = list(item.pop("_base_reasons"))
        disposition = str(item.pop("_base_disposition"))
        duplicate_of = item.get("duplicate_of")
        if duplicate_of:
            reasons.append("duplicate")
        if set(reasons) & legal_reasons:
            disposition = "quarantined"
        elif duplicate_of:
            disposition = "rejected"
        reasons = sorted(set(reasons))
        item["automatic_disposition"] = disposition
        item["current_disposition"] = disposition
        item["reasons"] = reasons
        item["reason_categories"] = sorted({"legal" if reason in legal_reasons else "technical" for reason in reasons})
        item["current_decision"] = "keep" if disposition == "accepted" else "exclude"
        item["review_rescuable"] = not bool(set(reasons) & _NON_RESCUABLE)
        previous = prior.get(relative)
        if (
            isinstance(previous, Mapping)
            and previous.get("preprocessing_signature") == item.get("preprocessing_signature")
            and previous.get("current_decision") == "keep"
            and item["review_rescuable"]
        ):
            item["current_disposition"] = "accepted"
            item["current_decision"] = "keep"
            item["human_override"] = "rescued"


def _revalidate_source_snapshot(
    items: Sequence[Mapping[str, Any]],
    root: Path,
    project_root: Path,
    initial_grouping: Mapping[str, Any],
    initial_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    _revalidate_source_identities(items)
    _revalidate_source_pack_bindings(root, project_root, initial_grouping, initial_bindings)


def _revalidate_source_identities(items: Sequence[Mapping[str, Any]]) -> None:
    changed = []
    for item in items:
        path = Path(str(item["source_path"]))
        extraction = item.get("sheet_extraction")
        expected = extraction.get("source_byte_sha256") if isinstance(extraction, Mapping) else item["byte_sha256"]
        if not path.is_file() or file_sha256(path) != expected:
            changed.append(str(item["relative_path"]))
    if changed:
        preview = ", ".join(changed[:5])
        raise DatasetInputError(f"Source files changed during import ({preview}). Run the same build command again.")


def _source_pack_bindings(root: Path, packs: Sequence[SourcePack]) -> dict[str, dict[str, Any]]:
    try:
        return {pack.pack_id: pack_source_binding(root, pack) for pack in packs}
    except OSError as exc:
        raise DatasetInputError("Source pack evidence became unreadable during import; run the build again.") from exc


def _revalidate_source_pack_bindings(
    root: Path,
    project_root: Path,
    initial_grouping: Mapping[str, Any],
    initial_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    current_grouping = load_grouping(project_root, root)
    if canonical_json_bytes(dict(current_grouping)) != canonical_json_bytes(dict(initial_grouping)):
        raise DatasetInputError("Source pack grouping changed during import; run the build again.")
    current_paths = _discover_pngs(root)
    current_packs = detect_packs(root, current_paths, user_grouping=current_grouping)
    current_bindings = _source_pack_bindings(root, current_packs)
    if canonical_json_bytes(current_bindings) != canonical_json_bytes(dict(initial_bindings)):
        raise DatasetInputError(
            "Source images, evidence files, archive identities, or pack boundaries changed during import; "
            "run the build again."
        )


def _raw_spec(item: Mapping[str, Any]) -> RawExtractionSpec:
    path = Path(str(item["source_path"]))
    source = dict(item["source"])
    license_record = dict(item["license"])
    evidence_hash = hashlib.sha256(canonical_json_bytes(evidence_digest_payload(source, license_record))).hexdigest()
    row = {
        "item_id": item["item_id"],
        "source": source,
        "license": license_record,
        "relative_path": item["relative_path"],
    }
    extraction = item.get("sheet_extraction")
    source_byte_sha256 = (
        str(extraction["source_byte_sha256"]) if isinstance(extraction, Mapping) else str(item["byte_sha256"])
    )
    transform = (
        ExtractionTransform(crop_coordinates=tuple(extraction["crop_rectangle"]), padding=None)
        if isinstance(extraction, Mapping)
        else ExtractionTransform.whole_image()
    )
    raw_source = RawSourceRecord(
        acquisition_run="product_dataset_intake",
        source_id=str(item["item_id"]),
        source_name=str(source.get("source_name") or source.get("path") or ""),
        source_type="local_directory",
        source_url=str(source.get("source_url") or ""),
        download_url="",
        distribution_platform="local_folder",
        creator_or_publisher=str(source.get("creator") or ""),
        original_filename=str(item["relative_path"]),
        manifest_path=str(source.get("path") or "source.txt"),
        manifest_sha256=evidence_hash,
        source_row_sha256=hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        archive_path=str(path),
        archive_sha256=source_byte_sha256,
        archive_size_bytes=path.stat().st_size,
        expected_archive_sha256=source_byte_sha256,
        resolution_method="direct_user_file",
        provenance_status="verified_user_supplied_evidence",
        license=license_record,
        source_record=row,
        resolved_archive_path=path,
    )
    return RawExtractionSpec(raw_source, None, transform)


def _summary(
    items: Sequence[Mapping[str, Any]],
    semantic: Mapping[str, Any],
    *,
    hierarchical: Mapping[str, Any] | None = None,
    pack_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dispositions = Counter(str(item["current_disposition"]) for item in items)
    automatic = Counter(str(item["automatic_disposition"]) for item in items)
    reason_counts = Counter(reason for item in items for reason in item.get("reasons", ()))
    source_items = [item for item in items if not item.get("sheet_extraction")]
    extracted_items = [item for item in items if item.get("sheet_extraction")]
    accepted = dispositions["accepted"]
    excluded = sum(
        dispositions[name] for name in ("rejected", "uncertain", "quarantined", "requires_special_extraction")
    )
    byte_duplicates = sum(1 for item in items if item.get("duplicate_kind") == "duplicate_bytes")
    decoded_duplicates = sum(1 for item in items if item.get("duplicate_kind") == "duplicate_decoded_rgba")
    legal_missing_reasons = {
        "ambiguous_pack_boundary",
        "conflicting_license_evidence",
        "conflicting_source_evidence",
        "missing_creator",
        "missing_license",
        "missing_pack_title",
        "missing_source",
        "unverified_license",
    }
    missing_information = sum(1 for item in source_items if set(item.get("reasons", ())) & legal_missing_reasons)
    rejected_technically = sum(
        1 for item in source_items if item.get("current_disposition") == "rejected" and not item.get("duplicate_kind")
    )
    near_groups = {str(item["near_duplicate_group"]) for item in items if item.get("near_duplicate_group")}
    counts = {
        "processed": len(source_items),
        "derived_items_processed": len(extracted_items),
        "accepted": accepted,
        "rejected": dispositions["rejected"],
        "uncertain": dispositions["uncertain"],
        "quarantined": dispositions["quarantined"],
        "duplicates": reason_counts["duplicate"],
        "byte_duplicates": byte_duplicates,
        "decoded_pixel_duplicates": decoded_duplicates,
        "exact_duplicates_removed": byte_duplicates + decoded_duplicates,
        "possible_near_duplicates": sum(1 for item in items if item.get("possible_near_duplicate")),
        "possible_near_duplicate_groups": len(near_groups),
        "special_extraction": dispositions["requires_special_extraction"],
        "needs_sheet_review": sum(
            1
            for item in source_items
            if item.get("current_disposition") == "requires_special_extraction"
            and "unresolved_sheet" in item.get("reasons", ())
        ),
        "sheets_split": dispositions["sheet_split"],
        "extracted_from_sheets": len(extracted_items),
        "rejected_technically": rejected_technically,
        "semantically_labeled": int(semantic.get("semantically_labeled", 0)),
        "semantically_abstained": int(semantic.get("semantically_abstained", 0)),
        "training_eligible": accepted,
        "image_only_eligible": accepted,
        "excluded": excluded,
        "accepted_automatically": automatic["accepted"],
        "rejected_automatically": automatic["rejected"],
        "missing_source": sum(1 for item in source_items if "missing_source" in item.get("reasons", ())),
        "missing_license": sum(1 for item in source_items if "missing_license" in item.get("reasons", ())),
        "missing_creator": sum(1 for item in source_items if "missing_creator" in item.get("reasons", ())),
        "missing_information": missing_information,
    }
    return {
        "schema_version": INTAKE_SCHEMA,
        "counts": counts,
        "disposition_counts": {name: dispositions[name] for name in DISPOSITIONS},
        "reason_counts": dict(sorted(reason_counts.items())),
        "semantic": dict(semantic),
        "packs": dict(pack_summary or {}),
        "hierarchical_labeling": dict(hierarchical or {}),
        "image_only_dataset": {"status": "READY" if accepted else "NOT_READY", "eligible": accepted},
        "conditioned_dataset": {
            "status": "READY" if semantic.get("conditioned_dataset_ready") else "NOT_READY",
            "reason": None if semantic.get("conditioned_dataset_ready") else "semantic_labels_incomplete",
        },
        "input_mutated": False,
        "training_launched": False,
    }


def _product_result(
    root: Path,
    output: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    reused: int,
    deleted: Sequence[str],
) -> ProductResult:
    counts = summary["counts"]
    accepted = int(counts["accepted"])
    status = ProductStatus.COMPLETE if accepted else ProductStatus.NEEDS_REVIEW
    blockers = ()
    if not accepted:
        blockers = (
            ProductBlocker(
                "no_training_eligible_images",
                "No image currently passes technical, provenance, license, and suitability gates.",
                f'python -m spritelab v3 dataset build "{root}"',
            ),
        )
    warnings: list[ProductWarning] = []
    if summary["semantic"].get("provider_status") == "health_failed":
        warnings.append(
            ProductWarning(
                "vision_provider_health_failed",
                str(summary["semantic"].get("health_failure")),
                "The accepted image-only dataset remains usable.",
            )
        )
    return ProductResult(
        status=status,
        feature="dataset",
        message=terminal_message(root, summary),
        capabilities=(
            ProductCapability(
                "dataset.image_only",
                "Image-only dataset",
                ProductStatus.READY if accepted else ProductStatus.BLOCKED,
                details={"eligible": accepted},
            ),
            ProductCapability(
                "dataset.conditioned",
                "Conditioned dataset",
                ProductStatus.READY
                if summary["conditioned_dataset"]["status"] == "READY"
                else ProductStatus.UNAVAILABLE,
                details=dict(summary["conditioned_dataset"]),
            ),
            ProductCapability(
                "dataset.review",
                "Exception review",
                ProductStatus.NEEDS_REVIEW if int(counts["excluded"]) else ProductStatus.COMPLETE,
                details={"items": int(counts["excluded"])},
            ),
        ),
        blockers=blockers,
        warnings=tuple(warnings),
        data={
            "schema_version": INTAKE_SCHEMA,
            "input_root": str(root),
            "output_root": str(output),
            "counts": dict(counts),
            "dispositions": dict(summary["disposition_counts"]),
            "semantic": dict(summary["semantic"]),
            "packs": dict(summary.get("packs", {})),
            "hierarchical_labeling": dict(summary.get("hierarchical_labeling", {})),
            "status_cards": _status_cards(summary),
            "review_queue": str(output / "review_queue.json"),
            "machine_result": str(output / "result.json"),
            "static_report_data": str(output / "report_data.json"),
            "next_command": "python -m spritelab v3 review",
            "resumability": {"reused": reused, "deleted": list(deleted)},
            "input_mutated": False,
            "training_launched": False,
        },
    )


def _write_outputs(
    output: Path,
    root: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    result: ProductResult,
    *,
    reused: int,
    deleted: Sequence[str],
) -> None:
    public_items = [_public_item(item) for item in items]
    _write_jsonl_atomic(output / "items.jsonl", public_items)
    queue_items = [
        _review_item(item)
        for item in public_items
        if item["current_disposition"] not in {"accepted", "sheet_split"}
        and item.get("sheet_decision") != "exclude_sheet"
    ]
    near_duplicate_items = [
        _near_duplicate_review_item(item) for item in public_items if item.get("possible_near_duplicate")
    ]
    semantic_items = [
        _semantic_review_item(item)
        for item in public_items
        if item["current_disposition"] == "accepted" and item.get("semantic", {}).get("needs_review")
    ]
    _write_json_atomic(
        output / "review_queue.json",
        {
            "schema_version": "spritelab.dataset.review_queue.v1",
            "input_root": str(root),
            "output_root": str(output),
            "items": queue_items + near_duplicate_items + semantic_items,
            "append_only_log": str(output / "review_log.jsonl"),
        },
    )
    (output / "review_log.jsonl").touch(exist_ok=True)
    report = {
        "schema_version": REPORT_SCHEMA,
        "summary": dict(summary),
        "status_cards": _status_cards(summary),
        "review_reason_filters": sorted(summary["reason_counts"]),
        "next_command": "python -m spritelab v3 review",
        "resumability": {"reused": reused, "deleted": list(deleted)},
        "input_mutated": False,
    }
    _write_json_atomic(output / "report_data.json", report)
    _write_json_atomic(output / "status_cards.json", {"cards": report["status_cards"]})
    _write_json_atomic(output / "pack_detection.json", dict(summary.get("packs", {})))
    _write_json_atomic(output / "result.json", result.to_dict())


def _review_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(item),
        "queue_kind": "intake_exception",
        "automatic_decision": "exclude",
        "default_visible": item["current_disposition"] in {"rejected", "uncertain", "requires_special_extraction"}
        and item.get("sheet_decision") != "exclude_sheet",
        "thumbnail_url": f"/dataset/review/thumb/{item['item_id']}",
    }


def _near_duplicate_review_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(item),
        "queue_kind": "near_duplicate_exception",
        "reasons": ["possible_near_duplicate"],
        "automatic_decision": "keep",
        "default_visible": False,
        "thumbnail_url": f"/dataset/review/thumb/{item['item_id']}",
    }


def _semantic_review_item(item: Mapping[str, Any]) -> dict[str, Any]:
    semantic = dict(item.get("semantic", {}))
    reasons = []
    if semantic.get("state") in {"abstained", "pending"}:
        reasons.append("semantic_abstention")
    if semantic.get("confidence") is not None and float(semantic["confidence"]) < 0.8:
        reasons.append("semantic_low_confidence")
    if semantic.get("conflicts"):
        reasons.append("semantic_conflict")
    if semantic.get("health_failure") or semantic.get("health_ok") is False:
        reasons.append("semantic_health_failure")
    return {
        **dict(item),
        "queue_kind": "semantic_exception",
        "reasons": reasons or ["semantic_review_required"],
        "automatic_decision": "keep_image_only",
        "default_visible": False,
        "thumbnail_url": f"/dataset/review/thumb/{item['item_id']}",
    }


def _public_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in item.items() if not str(key).startswith("_")}


def _status_cards(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = summary["counts"]
    return [
        {"id": "processed", "title": "Processed", "value": counts["processed"], "status": "COMPLETE"},
        {"id": "accepted", "title": "Accepted", "value": counts["accepted"], "status": "READY"},
        {
            "id": "extracted",
            "title": "Extracted from sheets",
            "value": counts["extracted_from_sheets"],
            "status": "COMPLETE",
        },
        {
            "id": "duplicates",
            "title": "Exact duplicates",
            "value": counts["exact_duplicates_removed"],
            "status": "COMPLETE",
        },
        {
            "id": "rejected",
            "title": "Rejected",
            "value": counts["rejected"],
            "status": "NEEDS_REVIEW" if counts["rejected"] else "COMPLETE",
        },
        {
            "id": "uncertain",
            "title": "Uncertain",
            "value": counts["uncertain"],
            "status": "NEEDS_REVIEW" if counts["uncertain"] else "COMPLETE",
        },
        {
            "id": "quarantined",
            "title": "Quarantined",
            "value": counts["quarantined"],
            "status": "NEEDS_REVIEW" if counts["quarantined"] else "COMPLETE",
        },
        {
            "id": "excluded",
            "title": "Excluded",
            "value": counts["excluded"],
            "status": "NEEDS_REVIEW" if counts["excluded"] else "COMPLETE",
        },
        {
            "id": "image-only",
            "title": "Image-only",
            "value": counts["image_only_eligible"],
            "status": summary["image_only_dataset"]["status"],
        },
        {
            "id": "conditioned",
            "title": "Conditioned",
            "value": counts["semantically_labeled"],
            "status": summary["conditioned_dataset"]["status"],
        },
    ]


def _load_state(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": STATE_SCHEMA, "input_root": str(root), "items": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetInputError(f"Saved import state is unreadable: {path} ({exc})") from exc
    if value.get("schema_version") != STATE_SCHEMA:
        raise DatasetInputError(f"Saved import state has an unsupported schema: {path}")
    if Path(str(value.get("input_root", ""))).resolve() != root:
        raise DatasetInputError(f"The output folder already belongs to a different dataset input: {path.parent}")
    return dict(value)


def _load_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    items: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            relative = str(record["relative_path"])
            item = record["item"]
            if not isinstance(item, Mapping):
                raise TypeError("journal item must be an object")
            items[relative] = dict(item)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetInputError(f"Saved preprocessing journal is unreadable: {path} ({exc})") from exc
    return items


def _append_journal(path: Path, relative: str, item: Mapping[str, Any]) -> None:
    record = {"relative_path": relative, "item": dict(item)}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _load_patterns(root: Path) -> dict[str, tuple[str, ...]]:
    return {name: _pattern_lines(root / f"{name}.txt") for name in ("include", "exclude")}


def _pattern_lines(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _policy_excluded(relative: str, patterns: Mapping[str, Sequence[str]]) -> bool:
    includes = patterns.get("include", ())
    excludes = patterns.get("exclude", ())
    return bool(includes and not any(_matches_pattern(relative, pattern) for pattern in includes)) or any(
        _matches_pattern(relative, pattern) for pattern in excludes
    )


def _matches_pattern(relative: str, pattern: str) -> bool:
    normalized = relative.replace("\\", "/")
    return fnmatch.fnmatchcase(normalized, pattern) or fnmatch.fnmatchcase(Path(normalized).name, pattern)


def _load_labels(root: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = root / "labels.csv"
    jsonl_path = root / "labels.jsonl"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    return _index_optional_rows(rows)


def _load_groups(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "groups.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return _index_optional_rows([dict(row) for row in csv.DictReader(handle)])


def _index_optional_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = dict(row)
        key = next(
            (
                str(value.get(name)).replace("\\", "/")
                for name in ("path", "relative_path", "filename", "image", "file")
                if value.get(name)
            ),
            None,
        )
        if key:
            indexed[key.casefold()] = value
    return indexed


def _labels_for_item(relative: str, labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    row = labels.get(relative.casefold()) or labels.get(Path(relative).name.casefold())
    if not row:
        return None
    identifiers = {"path", "relative_path", "filename", "image", "file"}
    return {str(key): value for key, value in row.items() if key not in identifiers and value not in (None, "")}


def _groups_for_item(relative: str, groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row = groups.get(relative.casefold()) or groups.get(Path(relative).name.casefold())
    return dict(row or {})


def _item_signature(
    relative: str,
    byte_hash: str,
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    patterns: Mapping[str, Sequence[str]],
    sheet_decision: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": INTAKE_SCHEMA,
        "relative_path": relative,
        "byte_sha256": byte_hash,
        "evidence": evidence_digest_payload(source, license_record),
        "patterns": {key: list(value) for key, value in patterns.items()},
        "sheet_decision": dict(sheet_decision) if sheet_decision else None,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _controlled_reason(code: str) -> str:
    mapping = {
        "FILE_MISSING": "unreadable",
        "FILE_UNREADABLE": "unreadable",
        "FILE_EMPTY": "unreadable",
        "IMAGE_DECODE_FAILED": "unreadable",
        "DECOMPRESSION_BOMB": "unreadable",
        "FULLY_TRANSPARENT": "blank",
        "EFFECTIVELY_EMPTY": "blank",
        "INVALID_PARTIAL_ALPHA": "invalid_alpha",
        "SPRITE_SHEET_OR_ANIMATION_STRIP": "unresolved_sheet",
        "POSSIBLE_SPRITE_STRIP": "unresolved_sheet",
        "INVALID_DIMENSIONS": "unusual_dimensions",
    }
    return mapping.get(code, code.casefold())


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for value in values
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    normalized = "".join(character.casefold() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "dataset"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

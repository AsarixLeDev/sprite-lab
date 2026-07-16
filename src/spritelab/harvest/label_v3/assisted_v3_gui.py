"""Auto-Labeling v3: prefilled correction Gradio GUI.

Two modes:
  calibration — appends ``V3CorrectionEvent`` to ``v3_corrections.jsonl``
  evaluation  — appends ``GoldenLabel`` to ``golden_labels.jsonl``

Evaluation mode accepts a frozen-suite manifest + partition, runs leakage
checks before the GUI opens, and writes records compatible with
``label-v3-eval``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.golden import GoldenLabel, append_golden_label, load_golden_labels
from spritelab.harvest.label_v3.assisted_golden_v3 import (
    V3_CORRECTIONS_FILENAME,
    V3CorrectionEvent,
    append_v3_correction,
    load_all_v3_records,
    load_v3_corrections,
    select_v3_calibration_sample,
    v3_candidate_summary_for_gui,
)
from spritelab.harvest.label_v3.field_prefill import COLOR_VALUES
from spritelab.harvest.label_v3.frozen_suites_v3 import FrozenSuiteManifest, check_suite_leakage
from spritelab.harvest.sources import utc_timestamp

logger = logging.getLogger(__name__)

V3_ASSISTED_STATE_FILENAME = "v3_assisted_state.json"
GOLDEN_LABELS_FILENAME = "golden_labels.jsonl"

EDITABLE_FIELDS = (
    "domain",
    "category",
    "canonical_object",
    "surface_alias",
    "color",
    "material",
    "shape",
    "role",
    "description",
)

FIELD_STATE_CHOICES = [
    "accepted",
    "abstained",
    "unknown",
    "novel",
    "ambiguous",
    "rejected",
    "quarantined",
    "not_applicable",
]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _reviewed_sprite_ids(state: V3GUIState) -> set[str]:
    return {str(event.get("sprite_id", "")) for event in state.corrections if event.get("sprite_id")}


def _display_candidate(state: V3GUIState) -> dict[str, Any] | None:
    """Overlay persisted corrections so navigation never resurrects stale values."""
    candidate = _current_candidate(state)
    if candidate is None:
        return None
    result = {**candidate, "fields": {name: dict(value) for name, value in (candidate.get("fields") or {}).items()}}
    latest = {
        str(event.get("field_name", "")): event
        for event in state.corrections
        if event.get("sprite_id") == candidate.get("sprite_id") and event.get("field_name") != "__review__"
    }
    for field_name, event in latest.items():
        if field_name == "tags":
            result["prefill_tags"] = _as_list(event.get("corrected_value"))
            continue
        field = result["fields"].get(field_name)
        if field is None:
            continue
        field["value"] = event.get("corrected_value")
        field["state"] = event.get("corrected_state", field.get("state", "unlabeled"))
        field["value_source"] = "reviewed_correction"
    changed = {name for name in ("canonical_object", "color", "material", "shape") if name in latest}
    if changed and "tags" not in latest:
        stale: set[str] = set()
        for name in ("canonical_object", "color", "shape", "role"):
            stale.update(_as_list(candidate.get("fields", {}).get(name, {}).get("value")))
        tags = [tag for tag in candidate.get("prefill_tags", ()) if tag not in stale]
        for name in ("canonical_object", "color", "shape", "role"):
            for value in _as_list(result["fields"].get(name, {}).get("value")):
                if value not in tags:
                    tags.append(value)
        result["prefill_tags"] = tags
    if changed and "description" not in latest:
        from spritelab.harvest.label_v3.description_enrichment import canonical_description_from_facts

        facts = {
            name: result["fields"].get(name, {}).get("value")
            for name in ("canonical_object", "color", "material", "shape")
        }
        color_roles = (result.get("prefill_metadata") or {}).get("color_roles") or {}
        facts.update(
            {
                name: color_roles[name]
                for name in ("primary_colors", "highlight_colors", "outline_color")
                if color_roles.get(name)
            }
        )
        canonical = canonical_description_from_facts(facts)
        result.setdefault("description_artifact", {}).update(
            {"canonical_description": canonical, "enriched_description": canonical, "regenerated_from_review": True}
        )
        result["fields"].setdefault("description", {})["value"] = canonical
        result["fields"]["description"]["value_source"] = "regenerated_from_review"
    return result


def prefill_conflict_explanation(candidate: dict[str, Any]) -> str:
    """Render the distinction between a high ranking prefill and auto-accept."""
    messages: list[str] = []
    for name, field in (candidate.get("fields") or {}).items():
        score = float(field.get("prefill_confidence") or 0.0)
        state = str(field.get("state", "unlabeled"))
        conflicts = list(field.get("conflicting_sources") or ())
        codes = list(field.get("contradiction_codes") or ())
        if score < 0.65 or (state == "accepted" and not conflicts and not codes):
            continue
        reason = str(field.get("decision_reason") or "not auto-accepted")
        source_bits = list(conflicts or field.get("supporting_sources") or ())
        refs = list(field.get("evidence_refs") or ())
        sources = ", ".join(map(str, source_bits)) or "calibration support unavailable"
        if refs:
            sources += f"; evidence refs: {', '.join(map(str, refs))}"
        messages.append(
            f"- **{name}** — prefill score: `{score:.2f}` ({field.get('confidence_kind', 'ranking score')}); "
            f"auto-accept: **no** (`{state}`, `{reason}`); conflicting evidence/source: "
            f"`{', '.join(map(str, codes)) if codes else sources}`."
        )
    return "\n".join(messages) if messages else "Prefill and calibrated decisions have no high-score conflict."


@dataclass(frozen=True)
class V3GUIState:
    run_dir: str = ""
    candidates: tuple[dict[str, Any], ...] = ()
    corrections: tuple[dict[str, Any], ...] = ()
    golden_labels: tuple[dict[str, Any], ...] = ()
    index: int = 0
    skipped: tuple[str, ...] = ()
    labeler: str = "operator"
    session_id: str = ""
    mode: str = "calibration"
    v3_records_path: str = ""
    correction_path: str = ""
    golden_path: str = ""
    suite_name: str = ""
    partition: str = ""
    scheduler_cohort: str = ""
    cohort_hash: str = ""
    cohort_mode: str = ""
    completed_ids_path: str = ""
    quality_decision_path: str = ""
    estimated_propagated_variants: int = 0


def launch_assisted_v3_gui(
    run_dir: str | Path,
    *,
    v3_records_path: str | Path | None = None,
    correction_path: str | Path | None = None,
    golden_path: str | Path | None = None,
    n: int | None = None,
    seed: int = 496,
    host: str = "127.0.0.1",
    port: int | None = None,
    labeler: str = "operator",
    mode: str = "calibration",
    suite_path: str | Path | None = None,
    partition: str = "",
    calibration_run: str | Path | None = None,
    scheduler_cohort: str | Path | None = None,
    pool_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    harvest_root: str | Path = "harvest_runs",
    completed_ids_path: str | Path | None = None,
) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The assisted-v3 GUI requires gradio. Install with: pip install gradio") from exc

    run_path = Path(work_dir or run_dir)
    scheduler_ids: list[str] | None = None
    cohort_hash = ""
    cohort_mode = ""
    estimated_propagated_variants = 0
    if scheduler_cohort:
        if pool_path is None:
            raise SystemExit("--scheduler-cohort requires --pool")
        from spritelab.harvest.label_v3.scheduler_input import cohort_sha256, prepare_scheduler_v3

        preparation = prepare_scheduler_v3(scheduler_cohort, pool_path, run_path, harvest_root=harvest_root)
        v3_path = preparation.records_path
        scheduler_ids = list(preparation.candidate_ids)
        cohort_hash = cohort_sha256(scheduler_cohort)
        estimated_propagated_variants = preparation.estimated_propagated_variants
        cohort_rows = read_jsonl(Path(scheduler_cohort))
        cohort_mode = str((cohort_rows[0].get("cohort_context") or {}).get("mode", "semantic_accept_only"))
    else:
        v3_path = Path(v3_records_path) if v3_records_path else _find_v3_records(run_path)
    if not v3_path.is_file():
        raise SystemExit(f"v3 records file not found: {v3_path}")

    session_id = utc_timestamp()

    is_eval = mode == "evaluation"
    corr_path = Path(correction_path) if correction_path else run_path / V3_CORRECTIONS_FILENAME
    gold_path = Path(golden_path) if golden_path else run_path / GOLDEN_LABELS_FILENAME
    state_path = run_path / V3_ASSISTED_STATE_FILENAME

    # Load v3 records
    v3_record_map = load_all_v3_records(v3_path)
    v3_records = list(v3_record_map.values())
    if not v3_records:
        raise SystemExit(f"No v3 records found in: {v3_path}")
    print(f"Loaded {len(v3_records)} v3 records from: {v3_path}")

    # --- Evaluation mode: frozen suite + leakage check ---
    partition_ids: set[str] | None = None
    manifest_suite_name = ""
    if is_eval and suite_path:
        manifest = FrozenSuiteManifest.load(Path(suite_path))
        manifest_suite_name = manifest.suite_name
        if not partition:
            raise SystemExit("--partition is required when --suite is provided in evaluation mode")
        partition_ids = set(manifest.partition_ids(partition))
        if not partition_ids:
            raise SystemExit(f"Partition '{partition}' is empty in suite '{manifest.suite_name}'")

        # Collect calibration IDs for leakage check
        calib_ids: set[str] = set()
        if calibration_run:
            calib_corr = Path(calibration_run) / V3_CORRECTIONS_FILENAME
            if calib_corr.is_file():
                calib_ids = {c.sprite_id for c in load_v3_corrections(calib_corr)}
        report = check_suite_leakage(manifest, tuning_ids=calib_ids)
        if not report.ok:
            details = []
            if report.cross_partition_overlaps:
                details.append(
                    f"cross-partition overlaps: { {k: len(v) for k, v in report.cross_partition_overlaps.items()} }"
                )
            if report.tuning_overlap:
                details.append(f"tuning overlap ({len(report.tuning_overlap)} IDs)")
            raise SystemExit(f"Frozen suite leakage detected: {'; '.join(details)}. Refusing to open GUI.")

        # Filter v3 records to partition
        filtered = {sid: v3_record_map[sid] for sid in partition_ids if sid in v3_record_map}
        if not filtered:
            raise SystemExit(
                f"No v3 records match partition '{partition}' (suite has {len(partition_ids)} IDs, "
                f"v3 file has {len(v3_record_map)} IDs)"
            )
        v3_record_map = filtered
        print(
            f"Suite: {manifest.suite_name}  partition: {partition}  "
            f"candidates: {len(v3_record_map)}/{len(partition_ids)}  leakage OK"
        )

    # Load corrections before rendering candidates: a persisted correction is
    # the first binding source and must be visible on initial load and resume.
    existing_corrections = load_v3_corrections(corr_path)
    correction_dicts = tuple(e.to_dict() for e in existing_corrections)

    # Select candidates
    effective_n = n if n is not None else min(len(v3_record_map), 50)
    all_v3 = list(v3_record_map.values())
    selected_ids = (
        scheduler_ids[:effective_n]
        if scheduler_ids is not None
        else select_v3_calibration_sample(all_v3, effective_n, seed=seed)
    )
    selected_records = [v3_record_map[sid] for sid in selected_ids if sid in v3_record_map]
    candidates = [v3_candidate_summary_for_gui(rec, existing_corrections) for rec in selected_records]

    # Load existing data
    existing_golden = load_golden_labels(gold_path)
    golden_dicts = tuple(
        {"sprite_id": sid, "category": gl.category, "object_name": gl.object_name, "tags": list(gl.tags)}
        for sid, gl in existing_golden.items()
    )

    # Determine starting index
    start_index = 0
    if state_path.is_file():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            start_index = _resume_index(saved, [candidate["sprite_id"] for candidate in candidates], cohort_hash)
            if not (0 <= start_index < len(candidates)):
                start_index = 0
        except Exception:
            start_index = 0

    state = V3GUIState(
        run_dir=str(run_path),
        candidates=tuple(candidates),
        corrections=correction_dicts,
        golden_labels=golden_dicts,
        index=start_index,
        labeler=labeler,
        session_id=session_id,
        mode=mode,
        v3_records_path=str(v3_path),
        correction_path=str(corr_path),
        golden_path=str(gold_path),
        suite_name=manifest_suite_name,
        partition=partition,
        scheduler_cohort=str(scheduler_cohort or ""),
        cohort_hash=cohort_hash,
        cohort_mode=cohort_mode,
        completed_ids_path=(
            str(completed_ids_path or (run_path / "completed_representative_ids.jsonl")) if scheduler_cohort else ""
        ),
        quality_decision_path=str(run_path / "quality_decisions.jsonl") if cohort_mode == "quality_quarantine" else "",
        estimated_propagated_variants=estimated_propagated_variants,
    )
    _write_state_json(state_path, state)

    preview_cache: dict[str, Image.Image | None] = {}

    def _preview(rec: dict[str, Any]) -> Image.Image | None:
        sid = rec["sprite_id"]
        if sid in preview_cache:
            return preview_cache[sid]
        png_path = Path(str((rec.get("prefill_metadata") or {}).get("resolved_png_path", "")))
        if not png_path.is_file():
            png_path = _find_png_path(run_path, sid)
        img = _load_preview(png_path)
        preview_cache[sid] = img
        return img

    def current_view(s: V3GUIState) -> tuple[Any, ...]:
        return _view_outputs(s, _preview)

    def previous(s: V3GUIState) -> tuple[Any, ...]:
        if not s.candidates:
            return (s, *_view_outputs(s, _preview))
        ns = replace(s, index=max(0, s.index - 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def next_item(s: V3GUIState) -> tuple[Any, ...]:
        if not s.candidates:
            return (s, *_view_outputs(s, _preview))
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def _write_output(s: V3GUIState, sprite_id: str, category: str, object_name: str, tags_value: Any) -> V3GUIState:
        if s.mode == "evaluation":
            label = GoldenLabel(
                sprite_id=sprite_id,
                category=category if category else "unknown",
                object_name=object_name,
                tags=tuple(_as_list(tags_value)),
                labeler=s.labeler,
                labeled_at=utc_timestamp(),
            )
            append_golden_label(Path(s.golden_path), label)
            gd = {
                "sprite_id": sprite_id,
                "category": label.category,
                "object_name": label.object_name,
                "tags": list(label.tags),
            }
            return replace(s, golden_labels=(*s.golden_labels, gd))
        return s

    def _mark_reviewed(s: V3GUIState, sprite_id: str, action: str) -> V3GUIState:
        """Persist an action marker so Accept/Save count even for blank fields."""
        if s.mode == "evaluation" or sprite_id in _reviewed_sprite_ids(s):
            return s
        event = V3CorrectionEvent(
            sprite_id=sprite_id,
            field_name="__review__",
            original_value=None,
            corrected_value=None,
            original_state="unlabeled",
            corrected_state="reviewed",
            selection_reason=action,
            review_action=action,
            reviewer_id=s.labeler,
            session_id=s.session_id,
        )
        append_v3_correction(Path(s.correction_path), event)
        if s.completed_ids_path:
            from spritelab.harvest.label_v3.scheduler_input import append_completed_id

            append_completed_id(s.completed_ids_path, sprite_id, cohort_hash=s.cohort_hash)
        return replace(s, corrections=(*s.corrections, event.to_dict()))

    def _complete_scheduler(s: V3GUIState, sprite_id: str) -> None:
        if s.completed_ids_path:
            from spritelab.harvest.label_v3.scheduler_input import append_completed_id

            append_completed_id(s.completed_ids_path, sprite_id, cohort_hash=s.cohort_hash)

    def quality_decide(s: V3GUIState, decision: str) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None or s.cohort_mode != "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        from spritelab.harvest.label_v3.scheduler_input import append_completed_id, append_quality_decision

        context = (candidate.get("prefill_metadata") or {}).get("scheduler_context") or {}
        reason_codes = (context.get("suitability") or {}).get("reason_codes") or ()
        append_quality_decision(
            s.quality_decision_path,
            candidate["sprite_id"],
            decision,
            suitability_reason_codes=reason_codes,
            reviewer_id=s.labeler,
        )
        if s.completed_ids_path:
            append_completed_id(s.completed_ids_path, candidate["sprite_id"], cohort_hash=s.cohort_hash)
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def accept_as_is(s: V3GUIState) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        if s.cohort_mode == "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        sid = candidate["sprite_id"]
        fields = candidate.get("fields", {})
        cat = str(fields.get("category", {}).get("value", "") or "")
        obj = str(fields.get("canonical_object", {}).get("value", "") or "")
        tags = list(candidate.get("prefill_tags", ()) or candidate.get("accepted_tags", ()))

        if s.mode == "evaluation":
            s = _write_output(s, sid, cat, obj, tags)
        else:
            for field_name in EDITABLE_FIELDS:
                field_data = fields.get(field_name, {})
                original_state = field_data.get("state", "unlabeled")
                original_value = field_data.get("value")
                if original_value not in (None, "", []):
                    event = V3CorrectionEvent(
                        sprite_id=sid,
                        field_name=field_name,
                        original_value=original_value,
                        corrected_value=original_value,
                        original_state=original_state,
                        corrected_state="accepted",
                        selection_reason="accept_as_is",
                        review_action="accepted_as_prefilled",
                        reviewer_id=s.labeler,
                        session_id=s.session_id,
                    )
                    append_v3_correction(Path(s.correction_path), event)
                    s = replace(s, corrections=(*s.corrections, event.to_dict()))
            if candidate.get("prefill_tags"):
                event = V3CorrectionEvent(
                    sprite_id=sid,
                    field_name="tags",
                    original_value=list(candidate["prefill_tags"]),
                    corrected_value=list(candidate["prefill_tags"]),
                    original_state="abstained",
                    corrected_state="accepted",
                    selection_reason="accept_as_is",
                    review_action="accepted_as_prefilled",
                    reviewer_id=s.labeler,
                    session_id=s.session_id,
                )
                append_v3_correction(Path(s.correction_path), event)
                s = replace(s, corrections=(*s.corrections, event.to_dict()))
            s = _mark_reviewed(s, sid, "accept_as_is")
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def correct(
        s: V3GUIState,
        domain_val: str,
        category_val: str,
        object_val: str,
        alias_val: str,
        color_val: list[str],
        material_val: str,
        shape_val: list[str],
        role_val: str,
        tags_val: list[str],
        description_val: str,
        domain_state: str,
        category_state: str,
        object_state: str,
        alias_state: str,
        color_state: str,
        material_state: str,
        shape_state: str,
        role_state: str,
        description_state: str,
    ) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        if s.cohort_mode == "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        sid = candidate["sprite_id"]

        if s.mode == "evaluation":
            cat = category_val.strip() or "unknown"
            obj = object_val.strip()
            s = _write_output(s, sid, cat, obj, tags_val)
        else:
            field_map = {
                "domain": (domain_val, domain_state),
                "category": (category_val, category_state),
                "canonical_object": (object_val, object_state),
                "surface_alias": (alias_val, alias_state),
                "color": (_as_list(color_val), color_state),
                "material": (material_val, material_state),
                "shape": (_as_list(shape_val), shape_state),
                "role": (role_val, role_state),
                "description": (description_val, description_state),
            }
            fields = candidate.get("fields", {})
            for field_name, (raw_value, raw_state) in field_map.items():
                field_data = fields.get(field_name, {})
                original_state = field_data.get("state", "unlabeled")
                original_value = field_data.get("value")
                value = (
                    _as_list(raw_value)
                    if isinstance(raw_value, (list, tuple))
                    else raw_value.strip()
                    if raw_value and raw_value.strip()
                    else original_value
                )
                new_state = raw_state.strip() if raw_state and raw_state.strip() else original_state
                if new_state != original_state or (new_state == "accepted" and value != original_value):
                    event = V3CorrectionEvent(
                        sprite_id=sid,
                        field_name=field_name,
                        original_value=original_value,
                        corrected_value=value,
                        original_state=original_state,
                        corrected_state=new_state,
                        selection_reason="manual_correction",
                        reviewer_id=s.labeler,
                        session_id=s.session_id,
                    )
                    append_v3_correction(Path(s.correction_path), event)
                    s = replace(s, corrections=(*s.corrections, event.to_dict()))
            normalized_tags = _as_list(tags_val)
            original_tags = list(candidate.get("prefill_tags", ()))
            if normalized_tags != original_tags:
                event = V3CorrectionEvent(
                    sprite_id=sid,
                    field_name="tags",
                    original_value=original_tags,
                    corrected_value=normalized_tags,
                    original_state="abstained",
                    corrected_state="accepted",
                    selection_reason="manual_correction",
                    review_action="corrected",
                    reviewer_id=s.labeler,
                    session_id=s.session_id,
                )
                append_v3_correction(Path(s.correction_path), event)
                s = replace(s, corrections=(*s.corrections, event.to_dict()))
            s = _mark_reviewed(s, sid, "save_and_next")
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def mark_unknown(s: V3GUIState) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        if s.cohort_mode == "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        sid = candidate["sprite_id"]
        if s.mode == "evaluation":
            s = _write_output(s, sid, "unknown", "", "")
        else:
            for field_name in ("category", "canonical_object"):
                field_data = candidate.get("fields", {}).get(field_name, {})
                event = V3CorrectionEvent(
                    sprite_id=sid,
                    field_name=field_name,
                    original_value=field_data.get("value"),
                    corrected_value=field_data.get("value"),
                    original_state=field_data.get("state", "unlabeled"),
                    corrected_state="unknown",
                    selection_reason="human_unknown",
                    reviewer_id=s.labeler,
                    session_id=s.session_id,
                )
                append_v3_correction(Path(s.correction_path), event)
                s = replace(s, corrections=(*s.corrections, event.to_dict()))
        _complete_scheduler(s, sid)
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def abstain_field(s: V3GUIState, field_name: str) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        if s.cohort_mode == "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        if s.mode == "evaluation":
            return (s, *_view_outputs(s, _preview))
        field_data = candidate.get("fields", {}).get(field_name, {})
        event = V3CorrectionEvent(
            sprite_id=candidate["sprite_id"],
            field_name=field_name,
            original_value=field_data.get("value"),
            corrected_value=field_data.get("value"),
            original_state=field_data.get("state", "unlabeled"),
            corrected_state="abstained",
            selection_reason="human_abstention",
            reviewer_id=s.labeler,
            session_id=s.session_id,
        )
        append_v3_correction(Path(s.correction_path), event)
        s = replace(s, corrections=(*s.corrections, event.to_dict()))
        _write_state_json(state_path, s)
        return (s, *_view_outputs(s, _preview))

    def reject_record(s: V3GUIState) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        if s.cohort_mode == "quality_quarantine":
            return (s, *_view_outputs(s, _preview))
        sid = candidate["sprite_id"]
        if s.mode == "evaluation":
            s = _write_output(s, sid, "unknown", "", "malformed,out_of_domain")
        else:
            for field_name in EDITABLE_FIELDS:
                field_data = candidate.get("fields", {}).get(field_name, {})
                event = V3CorrectionEvent(
                    sprite_id=sid,
                    field_name=field_name,
                    original_value=field_data.get("value"),
                    corrected_value=field_data.get("value"),
                    original_state=field_data.get("state", "unlabeled"),
                    corrected_state="rejected",
                    selection_reason="human_reject_malformed_ood",
                    reviewer_id=s.labeler,
                    session_id=s.session_id,
                )
                append_v3_correction(Path(s.correction_path), event)
                s = replace(s, corrections=(*s.corrections, event.to_dict()))
        _complete_scheduler(s, sid)
        ns = replace(s, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def skip(s: V3GUIState) -> tuple[Any, ...]:
        candidate = _display_candidate(s)
        if candidate is None:
            return (s, *_view_outputs(s, _preview))
        skipped = tuple(sorted({*s.skipped, candidate["sprite_id"]}))
        ns = replace(s, skipped=skipped, index=min(len(s.candidates) - 1, s.index + 1))
        _write_state_json(state_path, ns)
        return (ns, *_view_outputs(ns, _preview))

    def filtered_table(
        s: V3GUIState,
        unlabeled_only: bool,
        corrected_only: bool,
        category: str,
        source: str,
        text: str,
    ) -> dict[str, Any]:
        if s.mode == "evaluation":
            labeled_ids = {g.get("sprite_id") for g in s.golden_labels}
        else:
            labeled_ids = {
                c.get("sprite_id")
                for c in s.corrections
                if c.get("corrected_state", c.get("original_state")) != c.get("original_state")
            }
        rows = []
        for i, cand in enumerate(s.candidates):
            sid = cand.get("sprite_id", "")
            cfields = cand.get("fields", {})
            ccat = cfields.get("category", {}).get("value", "")
            ccat_state = cfields.get("category", {}).get("state", "unlabeled")
            if unlabeled_only and sid in labeled_ids:
                continue
            if corrected_only and sid not in labeled_ids:
                continue
            if category and str(ccat) != category:
                continue
            if source and source not in sid:
                continue
            if text and text.lower() not in str(cand).lower():
                continue
            rows.append([i + 1, ccat_state, ccat, sid])
        return {"headers": ["#", "state", "category", "sprite_id"], "data": rows[:200]}

    with gr.Blocks(title="SpriteLab V3 Correction") as demo:
        title = (
            "# Sprite quality review"
            if cohort_mode == "quality_quarantine"
            else "# Auto-Labeling v3 — Evaluation"
            if is_eval
            else "# Auto-Labeling v3 — Calibration"
        )
        gr.Markdown(title)
        state_box = gr.State(state)
        with gr.Row():
            progress_md = gr.Markdown()
            record_info = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(label="Sprite preview", type="pil", height=384)
                sprite_info = gr.Markdown()
            with gr.Column(scale=5):
                with gr.Row():
                    domain_state_dd = gr.Dropdown(label="Domain state", choices=FIELD_STATE_CHOICES, value="accepted")
                    domain_val = gr.Textbox(label="Domain")
                with gr.Row():
                    category_state_dd = gr.Dropdown(
                        label="Category state", choices=FIELD_STATE_CHOICES, value="accepted"
                    )
                    category_val = gr.Textbox(label="Category")
                with gr.Row():
                    object_state_dd = gr.Dropdown(label="Object state", choices=FIELD_STATE_CHOICES, value="accepted")
                    object_val = gr.Textbox(label="Object")
                with gr.Row():
                    alias_state_dd = gr.Dropdown(
                        label="Surface alias state", choices=FIELD_STATE_CHOICES, value="accepted"
                    )
                    alias_val = gr.Textbox(label="Surface alias")
                with gr.Row():
                    color_state_dd = gr.Dropdown(label="Color state", choices=FIELD_STATE_CHOICES, value="accepted")
                    color_val = gr.Dropdown(
                        label="Colors", choices=sorted(COLOR_VALUES), multiselect=True, allow_custom_value=True
                    )
                with gr.Row():
                    material_state_dd = gr.Dropdown(
                        label="Material state", choices=FIELD_STATE_CHOICES, value="accepted"
                    )
                    material_val = gr.Textbox(label="Material")
                with gr.Row():
                    shape_state_dd = gr.Dropdown(label="Shape state", choices=FIELD_STATE_CHOICES, value="accepted")
                    shape_val = gr.Dropdown(label="Shapes", choices=[], multiselect=True, allow_custom_value=True)
                with gr.Row():
                    role_state_dd = gr.Dropdown(label="Role state", choices=FIELD_STATE_CHOICES, value="accepted")
                    role_val = gr.Textbox(label="Role")
                with gr.Row():
                    description_state_dd = gr.Dropdown(
                        label="Description state", choices=FIELD_STATE_CHOICES, value="accepted"
                    )
                    description_val = gr.Textbox(label="Enriched description")
                tags_val = gr.Dropdown(label="Tags", choices=[], multiselect=True, allow_custom_value=True)
                canonical_description_val = gr.Textbox(label="Canonical description", interactive=False)
        with gr.Accordion("Why these prefills", open=False):
            prefill_explanation = gr.Markdown(label="Prefill / calibration explanation")
        with gr.Accordion("Decision diagnostics", open=False):
            v3_json = gr.JSON(label="V3 decision details")
        with gr.Row():
            accept_btn = gr.Button("Accept as-is", variant="primary")
            save_btn = gr.Button("Save + next", variant="primary")
            skip_btn = gr.Button("Skip")
        with gr.Row():
            unknown_btn = gr.Button("Mark unknown")
            reject_btn = gr.Button("Reject (malformed/OOD)", variant="stop")
        with gr.Row(visible=cohort_mode == "quality_quarantine"):
            quality_accept_btn = gr.Button("Quality accept", variant="primary")
            quality_reject_btn = gr.Button("Quality reject", variant="stop")
            quality_uncertain_btn = gr.Button("Quality uncertain")
        with gr.Row():
            with gr.Column():
                abstain_cat_btn = gr.Button("Abstain → category")
                abstain_obj_btn = gr.Button("Abstain → object")
            with gr.Column():
                previous_btn = gr.Button("Previous")
                next_btn = gr.Button("Next")
        with gr.Row():
            unlabeled_cb = gr.Checkbox(label="Unlabeled only", value=False)
            corrected_cb = gr.Checkbox(label="Corrected only", value=False)
            cat_filter = gr.Dropdown(label="Category filter", choices=[], value="", allow_custom_value=True)
            source_filter = gr.Textbox(label="Source filter")
            search_filter = gr.Textbox(label="Text search")
        with gr.Accordion("Candidate status", open=False):
            table = gr.JSON(label="Candidate status")

        view_outputs = [
            progress_md,
            record_info,
            image,
            sprite_info,
            domain_state_dd,
            domain_val,
            category_state_dd,
            category_val,
            object_state_dd,
            object_val,
            alias_state_dd,
            alias_val,
            color_state_dd,
            color_val,
            material_state_dd,
            material_val,
            shape_state_dd,
            shape_val,
            role_state_dd,
            role_val,
            description_state_dd,
            description_val,
            tags_val,
            canonical_description_val,
            v3_json,
            prefill_explanation,
            table,
        ]

        demo.load(current_view, inputs=state_box, outputs=view_outputs)
        previous_btn.click(previous, inputs=state_box, outputs=[state_box, *view_outputs])
        next_btn.click(next_item, inputs=state_box, outputs=[state_box, *view_outputs])
        accept_btn.click(accept_as_is, inputs=state_box, outputs=[state_box, *view_outputs])
        save_btn.click(
            correct,
            inputs=[
                state_box,
                domain_val,
                category_val,
                object_val,
                alias_val,
                color_val,
                material_val,
                shape_val,
                role_val,
                tags_val,
                description_val,
                domain_state_dd,
                category_state_dd,
                object_state_dd,
                alias_state_dd,
                color_state_dd,
                material_state_dd,
                shape_state_dd,
                role_state_dd,
                description_state_dd,
            ],
            outputs=[state_box, *view_outputs],
        )
        skip_btn.click(skip, inputs=state_box, outputs=[state_box, *view_outputs])
        unknown_btn.click(mark_unknown, inputs=state_box, outputs=[state_box, *view_outputs])
        reject_btn.click(reject_record, inputs=state_box, outputs=[state_box, *view_outputs])
        quality_accept_btn.click(
            lambda s: quality_decide(s, "quality_accept"), inputs=state_box, outputs=[state_box, *view_outputs]
        )
        quality_reject_btn.click(
            lambda s: quality_decide(s, "quality_reject"), inputs=state_box, outputs=[state_box, *view_outputs]
        )
        quality_uncertain_btn.click(
            lambda s: quality_decide(s, "quality_uncertain"), inputs=state_box, outputs=[state_box, *view_outputs]
        )
        abstain_cat_btn.click(
            lambda s: abstain_field(s, "category"), inputs=state_box, outputs=[state_box, *view_outputs]
        )
        abstain_obj_btn.click(
            lambda s: abstain_field(s, "canonical_object"), inputs=state_box, outputs=[state_box, *view_outputs]
        )
        for ctrl in (unlabeled_cb, corrected_cb, cat_filter, source_filter, search_filter):
            ctrl.change(
                filtered_table,
                inputs=[state_box, unlabeled_cb, corrected_cb, cat_filter, source_filter, search_filter],
                outputs=table,
            )

    demo.launch(server_name=host, server_port=port)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_v3_records(run_dir: Path) -> Path:
    from spritelab.harvest.label_v3.pipeline_stages_v3 import CANONICAL_RECORDS_NAME
    from spritelab.harvest.label_v3.pipeline_v3 import RECORD_OUTPUT_SUFFIX

    # non-sharded label-v3 output: v3_ + _v3_records.jsonl = v3_v3_records.jsonl
    pipeline_output_name = f"v3{RECORD_OUTPUT_SUFFIX}"
    candidates = [
        run_dir / "v3_output" / pipeline_output_name,
        run_dir / "v3_output" / CANONICAL_RECORDS_NAME,
        run_dir / CANONICAL_RECORDS_NAME,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return run_dir / "v3_output" / pipeline_output_name


def _find_png_path(run_dir: Path, sprite_id: str) -> Path | None:
    imported_path = run_dir / "imported.jsonl"
    if not imported_path.is_file():
        return None
    for record in read_jsonl(imported_path):
        if record.get("sprite_id") == sprite_id:
            png_path = record.get("final_png_path", "")
            return Path(png_path) if png_path else None
    return None


def _load_preview(path: Path | None, scale: int = 10) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
    except Exception:
        return None
    preview = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
    checker = _checkerboard(preview.size)
    return Image.alpha_composite(checker, preview).convert("RGB")


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (238, 238, 238, 255))
    pixels = image.load()
    tile = max(8, width // 16)
    for y in range(height):
        for x in range(width):
            value = 238 if ((x // tile) + (y // tile)) % 2 == 0 else 188
            pixels[x, y] = (value, value, value, 255)
    return image


def _current_candidate(state: V3GUIState) -> dict[str, Any] | None:
    if not state.candidates:
        return None
    return state.candidates[min(max(0, int(state.index)), len(state.candidates) - 1)]


def _view_outputs(
    state: V3GUIState,
    preview_fn,
) -> tuple[Any, ...]:
    candidate = _display_candidate(state)
    total = len(state.candidates)
    if state.mode == "evaluation":
        labeled_ids = {g.get("sprite_id") for g in state.golden_labels}
    else:
        labeled_ids = _reviewed_sprite_ids(state)
    scheduler_completed = _completed_scheduler_ids(state)
    if scheduler_completed:
        labeled_ids |= scheduler_completed
    labeled = len(labeled_ids)
    skipped = len(state.skipped)
    suite_info = f"\n\nSuite: {state.suite_name}  partition: {state.partition}" if state.suite_name else ""
    completed_propagation = sum(
        int(((candidate.get("prefill_metadata") or {}).get("scheduler_context") or {}).get("propagation_count", 0))
        for candidate in state.candidates
        if candidate.get("sprite_id") in labeled_ids
    )
    geometry_count = len(
        {
            ((candidate.get("prefill_metadata") or {}).get("scheduler_context") or {}).get("geometry_group")
            for candidate in state.candidates
            if ((candidate.get("prefill_metadata") or {}).get("scheduler_context") or {}).get("geometry_group")
        }
    )
    scheduler_progress = (
        f"\n\nGeometry representatives: {geometry_count}"
        f"\n\nEstimated propagated variants: {state.estimated_propagated_variants}"
        f"\n\nCompleted propagated variants: {completed_propagation}"
        f"\n\nRemaining propagation value: {state.estimated_propagated_variants - completed_propagation}"
        if state.scheduler_cohort
        else ""
    )
    progress = (
        f"**Progress**: {state.index + 1} / {total}\n\n"
        f"Labeled: {labeled}\n\n"
        f"Skipped: {skipped}\n\n"
        f"Mode: {state.cohort_mode or state.mode}{suite_info}{scheduler_progress}"
    )

    if candidate is None:
        defaults = (
            progress,
            "",
            None,
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "accepted",
            "",
            "",
            "",
            {},
            "",
            {"data": []},
        )
        return defaults

    fields = candidate.get("fields", {})
    scheduler_context = (candidate.get("prefill_metadata") or {}).get("scheduler_context") or {}
    scheduling_info = (
        f"Scheduling type (context only): `{scheduler_context.get('broad_type', 'unknown')}`\n\n"
        f"Suitability: `{(scheduler_context.get('suitability') or {}).get('status', 'unknown')}`; "
        f"reason codes: {', '.join((scheduler_context.get('suitability') or {}).get('reason_codes') or ()) or 'none'}\n\n"
        if scheduler_context
        else ""
    )
    record_info = (
        f"Sprite ID: `{candidate['sprite_id']}`\n\n"
        f"Record state: `{candidate['record_state']}`\n\n"
        f"Reason codes: {', '.join(candidate.get('reason_codes', ())) or 'none'}\n\n"
        f"{scheduling_info}"
    )
    sprite_info = (
        f"Accepted fields: {', '.join(candidate.get('accepted_fields', ())) or 'none'}\n\n"
        f"Abstained fields: {', '.join(candidate.get('abstained_fields', ())) or 'none'}\n\n"
        f"Prefill tags: {', '.join(candidate.get('prefill_tags', ()) or candidate.get('accepted_tags', ())) or 'none'}"
    )

    def _fd(fn: str) -> dict[str, Any]:
        return fields.get(fn, {"state": "unlabeled", "value": ""})

    preview = preview_fn(candidate)

    table_rows = []
    for i, cand in enumerate(state.candidates):
        cid = cand.get("sprite_id", "")
        cfields = cand.get("fields", {})
        ccat = cfields.get("category", {}).get("value", "")
        ccat_state = cfields.get("category", {}).get("state", "unlabeled")
        in_labeled = cid in labeled_ids
        status = "labeled" if in_labeled else "unlabeled"
        table_rows.append([i + 1, status, ccat_state, ccat, cid])

    return (
        progress,
        record_info,
        preview,
        sprite_info,
        _fd("domain").get("state", "unlabeled"),
        str(_fd("domain").get("value", "") or ""),
        _fd("category").get("state", "unlabeled"),
        str(_fd("category").get("value", "") or ""),
        _fd("canonical_object").get("state", "unlabeled"),
        str(_fd("canonical_object").get("value", "") or ""),
        _fd("surface_alias").get("state", "unlabeled"),
        str(_fd("surface_alias").get("value", "") or ""),
        _fd("color").get("state", "unlabeled"),
        _as_list(_fd("color").get("value", "")),
        _fd("material").get("state", "unlabeled"),
        str(_fd("material").get("value", "") or ""),
        _fd("shape").get("state", "unlabeled"),
        _as_list(_fd("shape").get("value", "")),
        _fd("role").get("state", "unlabeled"),
        str(_fd("role").get("value", "") or ""),
        _fd("description").get("state", "unlabeled"),
        str(_fd("description").get("value", "") or ""),
        _as_list(candidate.get("prefill_tags", ())),
        str(candidate.get("description_artifact", {}).get("canonical_description", "") or ""),
        {
            "fields": {
                k: {
                    kk: str(vv) if not isinstance(vv, (list, dict)) else vv
                    for kk, vv in v.items()
                    if kk
                    in (
                        "state",
                        "value",
                        "decision_reason",
                        "contradiction_codes",
                        "evidence_count",
                        "ci_lower",
                        "ci_upper",
                        "prefill_confidence",
                        "confidence_kind",
                        "alternatives",
                        "score_components",
                        "evidence_refs",
                        "supporting_sources",
                        "conflicting_sources",
                        "normalization_actions",
                        "prefill_warnings",
                    )
                }
                for k, v in fields.items()
            },
            "record_state": candidate["record_state"],
            "reason_codes": candidate.get("reason_codes", []),
            "description_artifact": candidate.get("description_artifact", {}),
            "prefill_metadata": candidate.get("prefill_metadata", {}),
        },
        prefill_conflict_explanation(candidate),
        {"headers": ["#", "status", "cat_state", "category", "sprite_id"], "data": table_rows[:200]},
    )


def _write_state_json(path: str | Path, state: V3GUIState) -> None:
    output = {
        "current_index": state.index,
        "labeler": state.labeler,
        "session_id": state.session_id,
        "mode": state.mode,
        "suite_name": state.suite_name,
        "partition": state.partition,
        "last_opened": utc_timestamp(),
        "total": len(state.candidates),
        "labeled_count": len({g.get("sprite_id") for g in state.golden_labels})
        if state.mode == "evaluation"
        else len(_reviewed_sprite_ids(state)),
        "skipped": list(state.skipped),
        "candidate_ids": [candidate.get("sprite_id", "") for candidate in state.candidates],
        "scheduler_cohort": state.scheduler_cohort,
        "cohort_hash": state.cohort_hash,
        "cohort_mode": state.cohort_mode,
        "estimated_propagated_variants": state.estimated_propagated_variants,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _completed_scheduler_ids(state: V3GUIState) -> set[str]:
    if not state.completed_ids_path:
        return set()
    from spritelab.annotation_scheduler.scheduler import read_completed_ids

    return read_completed_ids(state.completed_ids_path)


def _resume_index(saved: Mapping[str, Any], candidate_ids: Sequence[str], cohort_hash: str = "") -> int:
    """Resume only when the persisted ordered cohort identity matches exactly."""

    saved_ids = [str(value) for value in saved.get("candidate_ids") or ()]
    if saved_ids and saved_ids != list(candidate_ids):
        return 0
    if cohort_hash and saved.get("cohort_hash") not in (None, "", cohort_hash):
        return 0
    return int(saved.get("current_index", 0))

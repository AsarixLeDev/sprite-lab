"""Strict quality-only and semantic-only Labeling-v4 review GUIs."""

from __future__ import annotations

import base64
import inspect
import io
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v4.audit_prefill import (
    AUDIT_SELECTION_SCHEMA,
    PREFILL_FIELDS,
    detect_audit_schema,
    require_prefilled_records,
)
from spritelab.harvest.label_v4.review import (
    abstain_field,
    accept_model_abstention,
    accept_proposal,
    compact_review_presenter,
    field_value_shape,
    load_review_events,
    mark_not_applicable,
    mark_unsupported,
    record_quality_decision,
    save_field_correction,
    select_alternative,
)
from spritelab.harvest.label_v4.two_pass import (
    QUALITY_ELIGIBLE,
    QualityResolution,
    calibration_denominator_report,
    has_real_semantic_proposal,
    quality_resume_state,
    require_semantic_ready_records,
    resolve_quality_decisions,
    semantic_completion,
    semantic_field_progress,
    semantic_readiness,
    validate_semantic_field,
)

REVIEW_MODES = frozenset({"quality_only", "semantic_assisted", "manual_truth_diagnostic"})
ZOOM_CHOICES = (1, 8, 12, 16)
DEFAULT_ZOOM = 12
CHECKERBOARD_CSS = (
    "background-color:#d8d8d8;"
    "background-image:linear-gradient(45deg,#b8b8b8 25%,transparent 25%),"
    "linear-gradient(-45deg,#b8b8b8 25%,transparent 25%),"
    "linear-gradient(45deg,transparent 75%,#b8b8b8 75%),"
    "linear-gradient(-45deg,transparent 75%,#b8b8b8 75%);"
    "background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0;"
)
GUI_CSS = ".sprite-pixel-viewer img{image-rendering:pixelated!important}.field-header{font-size:1.2rem}"


def _blocks(gr: Any, title: str) -> Any:
    """Pass CSS at the supported location across Gradio 4-6."""

    if "css" in inspect.signature(gr.Blocks.launch).parameters:
        return gr.Blocks(title=title)
    return gr.Blocks(title=title, css=GUI_CSS)


def _launch_gradio(demo: Any, host: str, port: int, share: bool) -> Any:
    kwargs: dict[str, Any] = {"server_name": host, "server_port": int(port), "share": bool(share)}
    if "css" in inspect.signature(demo.launch).parameters:
        kwargs["css"] = GUI_CSS
    return demo.launch(**kwargs)


def normalize_review_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in REVIEW_MODES:
        raise ValueError(f"invalid assisted-v4 mode: {value}")
    return mode


def gui_mode_contract(mode: str) -> dict[str, Any]:
    normalized = normalize_review_mode(mode)
    return {
        "mode": normalized,
        "semantic_controls_present": normalized != "quality_only",
        "banner": {
            "quality_only": "QUALITY REVIEW ONLY\nNo semantic judgment is requested in this pass.",
            "semantic_assisted": "SEMANTIC CALIBRATION\nJudge the model proposal, not the source metadata.",
            "manual_truth_diagnostic": (
                "MANUAL TRUTH DIAGNOSTIC\nNot suitable for assisted-model accuracy calibration."
            ),
        }[normalized],
    }


def load_assisted_records(
    path: str | Path,
    *,
    mode: str = "quality_only",
    diagnostic_allow_selection: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(Path(path))
    normalized = normalize_review_mode(mode)
    if diagnostic_allow_selection and rows and detect_audit_schema(rows[0]) == AUDIT_SELECTION_SCHEMA:
        raise ValueError(
            "Diagnostic mode confirmed a raw audit selection manifest. It is intentionally non-reviewable; "
            "run label-v4-prepare-audit first."
        )
    require_prefilled_records(rows)
    if normalized == "semantic_assisted":
        require_semantic_ready_records(rows)
        for record in rows:
            if record.get("review_mode") == "blind" and not has_real_semantic_proposal(record):
                raise ValueError(f"Record {record.get('sprite_id')} cannot use blind review without a real proposal")
    return rows


def review_resume_state(
    records: list[dict[str, Any]], corrections_path: str | Path, *, mode: str = "quality_only"
) -> dict[str, Any]:
    events = load_review_events(corrections_path, strict=False)
    normalized = normalize_review_mode(mode)
    if normalized == "quality_only":
        return quality_resume_state(records, events)
    quality = resolve_quality_decisions(records, events)
    remaining_indices: list[int] = []
    for index, record in enumerate(records):
        sprite_id = str(record.get("sprite_id", ""))
        decision = quality[sprite_id]
        if decision.effective_state == "quality_unreviewed" and record.get("quality_state") in QUALITY_ELIGIBLE:
            decision = QualityResolution(sprite_id, str(record["quality_state"]), None, 0, 0)
        if not semantic_completion(record, events, decision)["complete"]:
            remaining_indices.append(index)
    return {
        "next_index": remaining_indices[0] if remaining_indices else None,
        "review_complete": not remaining_indices,
        "remaining": len(remaining_indices),
        "completed": len(records) - len(remaining_indices),
        "total": len(records),
    }


def review_resume_index(
    records: list[dict[str, Any]], corrections_path: str | Path, *, mode: str = "quality_only"
) -> int | None:
    return review_resume_state(records, corrections_path, mode=mode)["next_index"]


def require_quality_eligible_for_semantic(records: list[dict[str, Any]], events: Any) -> None:
    resolved = resolve_quality_decisions(records, events)
    ineligible = [
        record["sprite_id"]
        for record in records
        if resolved[record["sprite_id"]].effective_state not in QUALITY_ELIGIBLE
        and record.get("quality_state") not in QUALITY_ELIGIBLE
    ]
    if ineligible:
        raise ValueError(
            f"semantic-assisted requires eligible quality decisions; {len(ineligible)} records are not eligible"
        )


def pixel_preview_html(image_path: str | Path, zoom: int = DEFAULT_ZOOM, *, include_crop: bool = True) -> str:
    if int(zoom) not in ZOOM_CHOICES:
        raise ValueError(f"zoom must be one of {ZOOM_CHOICES}")
    with Image.open(Path(image_path)) as source:
        rgba = source.convert("RGBA")
        width, height = rgba.size
        bbox = rgba.getchannel("A").getbbox()
        crop = rgba.crop(bbox) if bbox and include_crop and bbox != (0, 0, width, height) else None
        full_uri = _image_data_uri(rgba)
        crop_uri = _image_data_uri(crop) if crop is not None else ""
    display_width = max(384, width * int(zoom)) if zoom != 1 else width
    display_height = max(384, height * int(zoom)) if zoom != 1 else height
    style = "image-rendering:pixelated;image-rendering:crisp-edges;object-fit:contain;display:block"
    cells = [
        f'<figure><div style="{CHECKERBOARD_CSS}display:inline-block">'
        f'<img alt="full sprite canvas" src="{full_uri}" width="{display_width}" height="{display_height}" '
        f'style="{style}"></div><figcaption>Decoded exported canvas — {width}\u00d7{height}, '
        f"zoom {zoom}\u00d7</figcaption></figure>"
    ]
    if crop is not None:
        crop_width, crop_height = crop.size
        cells.append(
            f'<figure><div style="{CHECKERBOARD_CSS}display:inline-block">'
            f'<img alt="tight foreground crop" src="{crop_uri}" width="{max(192, crop_width * int(zoom))}" '
            f'height="{max(192, crop_height * int(zoom))}" style="{style}"></div>'
            f"<figcaption>Tight foreground crop — {crop_width}\u00d7{crop_height}</figcaption></figure>"
        )
    return (
        '<div class="sprite-pixel-viewer" style="display:flex;gap:20px;align-items:flex-start;overflow:auto">'
        + "".join(cells)
        + "</div>"
    )


def quality_record_summary(record: dict[str, Any]) -> str:
    source = record.get("source_metadata", {})
    suitability = record.get("source_suitability", {})
    source_native = record.get("native_dimensions") or {}
    decoded_exported = record.get("decoded_exported_dimensions") or {}
    return (
        f"**Source suitability:** `{suitability.get('status', 'unknown')}`  \n"
        f"**Reason codes:** `{json.dumps(suitability.get('reason_codes', []))}`  \n"
        f"**Source:** `{source.get('source_id')}`  \n"
        f"**Pack:** `{source.get('pack_name') or source.get('pack_id')}`  \n"
        f"**Sheet/image:** `{source.get('source_sheet') or source.get('source_image')}`  \n"
        f"**Source-native size:** `{source_native.get('width')}x{source_native.get('height')}`  \n"
        f"**Decoded export size:** `{decoded_exported.get('width')}x{decoded_exported.get('height')}`  \n"
        f"**Resize policy:** `{record.get('resize_policy') or 'unspecified'}`"
    )


def gui_record_summary(record: dict[str, Any]) -> str:
    fields = record["fields"]
    values = {
        "Canonical object": fields["canonical_object"]["value"],
        "Category": fields["category"]["value"],
        "Domain": fields["domain"]["value"],
        "Role": fields["role"]["value"],
        "Material": fields["explicit_material"]["value"],
        "Main colors": fields["primary_colors"]["value"] or fields["palette_colors"]["value"],
        "Description": fields["description"]["value"],
        "Record uncertainty": record.get("label_quality", {}).get("record_uncertainty_1_20"),
    }
    return "\n".join(f"**{label}:** `{json.dumps(value, ensure_ascii=False)}`" for label, value in values.items())


def gui_field_view(record: dict[str, Any], field_name: str, corrections_path: str | Path) -> dict[str, Any]:
    events = load_review_events(corrections_path, strict=False)
    view = compact_review_presenter(record, events)
    rendered = dict(view.get("fields", {}).get(field_name, {}))
    source = dict(record.get("fields", {}).get(field_name, {}))
    blind_locked = (
        record.get("review_mode") == "blind"
        and has_real_semantic_proposal(record)
        and not _critical_judgment_exists(record, events)
    )
    return {
        "sprite_id": view.get("sprite_id", ""),
        "field": field_name,
        "proposed_value": None if blind_locked else rendered.get("proposed_value"),
        "value_state": source.get("value_state", "unsupported"),
        "reason": source.get("reason", "missing_field_reason"),
        "alternatives": [] if blind_locked else list(rendered.get("alternatives") or ()),
        "uncertainty": None if blind_locked else rendered.get("uncertainty_1_20"),
        "risk_band": "hidden_blind_first_judgment" if blind_locked else rendered.get("risk_band", "not_scorable"),
        "evidence_summary": [] if blind_locked else list(rendered.get("evidence_summary") or ()),
        "conflict_disposition": source.get("conflict_disposition", "none"),
        "propagation_scope": rendered.get("propagation_scope", "none"),
        "training_consequence": source.get("training_consequence") or rendered.get("training_consequence", "excluded"),
        "review_state": rendered.get("review_state", "unreviewed"),
        "blind_locked": blind_locked,
        "details": {
            "field_quality": record.get("label_quality", {}).get("fields", {}).get(field_name, {}),
            "prediction_state": record.get("prediction_state"),
            "missing_stages": record.get("missing_stages", []),
            "raw_proposal_hash": view.get("raw_proposal_hash", ""),
        },
    }


def launch_assisted_v4(
    records_path: str | Path,
    corrections_path: str | Path,
    *,
    mode: str = "quality_only",
    host: str = "127.0.0.1",
    port: int = 7862,
    share: bool = False,
    diagnostic_allow_selection: bool = False,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("assisted-v4 requires gradio; install the harvest-ui extra") from exc
    normalized = normalize_review_mode(mode)
    records = load_assisted_records(
        records_path, mode=normalized, diagnostic_allow_selection=diagnostic_allow_selection
    )
    corrections = Path(corrections_path)
    if normalized == "semantic_assisted":
        events = load_review_events(corrections, strict=False)
        require_quality_eligible_for_semantic(records, events)
    resume = review_resume_index(records, corrections, mode=normalized)
    opened_at: dict[int, tuple[float, str]] = {}
    contract = gui_mode_contract(normalized)
    if normalized == "quality_only":
        return _launch_quality_gui(gr, records, corrections, resume, opened_at, contract, host, port, share)
    return _launch_semantic_gui(gr, records, corrections, resume, opened_at, contract, normalized, host, port, share)


def _launch_quality_gui(
    gr: Any,
    records: list[dict[str, Any]],
    corrections: Path,
    resume: int | None,
    opened_at: dict[int, tuple[float, str]],
    contract: dict[str, Any],
    host: str,
    port: int,
    share: bool,
) -> Any:
    def render(index: int | None, zoom: int = DEFAULT_ZOOM) -> tuple[Any, ...]:
        if index is None:
            return (
                None,
                "",
                "",
                "**Review complete.** All quality records have terminal decisions.",
                "review_complete",
            )
        index = max(0, min(len(records) - 1, int(index)))
        record = records[index]
        opened_at.setdefault(index, (time.monotonic(), _utc_now()))
        resolved = resolve_quality_decisions(records, load_review_events(corrections, strict=False))[
            record["sprite_id"]
        ]
        return (
            index,
            record["sprite_id"],
            pixel_preview_html(record["image_path"], int(zoom)),
            quality_record_summary(record),
            resolved.effective_state,
        )

    def decide(index: int | None, outcome: str, zoom: int) -> tuple[Any, ...]:
        if index is None:
            raise ValueError("quality review is already complete")
        record = records[int(index)]
        event = record_quality_decision(
            corrections,
            record,
            outcome,
            reviewer_id="assisted_v4_quality_gui",
            session_id=str(record.get("audit_id", "")),
            metadata={
                **_timing_metadata(record, int(index), opened_at),
                "audit_id": record.get("audit_id"),
                "review_mode": "quality_only",
            },
        )
        next_index = review_resume_index(records, corrections, mode="quality_only")
        return f"Appended immutable {event.human_outcome}; advanced.", *render(next_index, zoom)

    def navigate(index: int | None, delta: int, zoom: int) -> tuple[Any, ...]:
        if index is None:
            return render(len(records) - 1 if delta < 0 else None, zoom)
        return render(int(index) + delta, zoom)

    with _blocks(gr, "Sprite Lab Labeling v4 Quality Review") as demo:
        index_state = gr.State(resume)
        gr.Markdown(f"# {contract['banner']}")
        with gr.Row():
            previous = gr.Button("Previous")
            next_ = gr.Button("Next")
            sprite_id = gr.Textbox(label="Sprite", interactive=False)
            zoom = gr.Dropdown(label="Zoom", choices=list(ZOOM_CHOICES), value=DEFAULT_ZOOM)
        preview = gr.HTML(label="Nearest-neighbor sprite preview")
        metadata = gr.Markdown()
        state = gr.Textbox(label="Effective quality state", interactive=False)
        with gr.Row():
            suitable = gr.Button("Suitable")
            uncertain_usable = gr.Button("Uncertain but usable")
            unsuitable = gr.Button("Unsuitable")
            uncertain_not_usable = gr.Button("Uncertain not usable")
        status = gr.Textbox(label="Append-only quality event", interactive=False)
        outputs = [index_state, sprite_id, preview, metadata, state]
        demo.load(lambda: render(resume), outputs=outputs)
        zoom.change(render, inputs=[index_state, zoom], outputs=outputs)
        previous.click(lambda i, z: navigate(i, -1, z), inputs=[index_state, zoom], outputs=outputs)
        next_.click(lambda i, z: navigate(i, 1, z), inputs=[index_state, zoom], outputs=outputs)
        for button, outcome in (
            (suitable, "quality_suitable"),
            (uncertain_usable, "quality_uncertain_usable"),
            (unsuitable, "quality_unsuitable"),
            (uncertain_not_usable, "quality_uncertain_not_usable"),
        ):
            button.click(
                lambda i, z, value=outcome: decide(i, value, z), inputs=[index_state, zoom], outputs=[status, *outputs]
            )
    return _launch_gradio(demo, host, port, share)


def correction_format_hint(record: dict[str, Any], field_name: str) -> str:
    shape = field_value_shape(record, field_name)
    examples = {
        "string": "Enter plain text, e.g. `bracers`.",
        "list": 'Enter JSON, e.g. `["gray", "brown"]`. Comma-separated text is also accepted.',
        "object": 'Enter a JSON object, e.g. `{"silhouette": ["compact"]}`.',
        "integer": "Enter a JSON integer, including `0`.",
        "number": "Enter a JSON number, including `0`.",
        "boolean": "Enter JSON `true` or `false`.",
    }
    return examples[shape]


def preview_field_correction(record: dict[str, Any], field_name: str, raw_value: Any) -> str:
    from spritelab.harvest.label_v4.review import parse_field_correction

    try:
        parsed = parse_field_correction(record, field_name, raw_value)
    except ValueError as exc:
        return f"**Nothing will be saved.** {exc}"
    return f"**Will save:** `{field_name} = {json.dumps(parsed, ensure_ascii=False)}`"


def field_action_availability(record: dict[str, Any], field_name: str, *, blind_locked: bool = False) -> dict[str, Any]:
    validation = validate_semantic_field(record, field_name)
    state = validation.value_state
    known = validation.valid and state == "known" and not blind_locked
    proven_abstention = validation.valid and state == "model_abstained" and not blind_locked
    correction = state not in {"missing_prediction", "provider_failed", "not_scorable", "not_scorable_due_to_image"}
    reasons = {
        "accept": "Available only for a valid, non-null known proposal.",
        "model_abstention": "Available only for a model abstention with completed, identity-bound stage proof.",
        "alternative": "Available only when a visible alternative is selected.",
        "correction": "Unavailable for missing or failed predictions; prepare semantic inference first.",
    }
    return {
        "accept": known,
        "model_abstention": proven_abstention,
        "alternative": correction and not blind_locked,
        "correction": correction,
        "human_abstention": correction,
        "unsupported": correction,
        "not_applicable": correction,
        "reasons": reasons,
    }


def _progress_markdown(progress: dict[str, Any]) -> tuple[str, str]:
    lines = ["| Field | Model state | Human state | Status |", "|---|---|---|---|"]
    for row in progress["fields"]:
        lines.append(f"| `{row['field']}` | {row['model_state']} | {row['human_state']} | {row['status']} |")
    remaining = ", ".join(progress["remaining_required_fields"]) or "none"
    complete = "yes" if progress["record_complete"] else "no"
    record = (
        f"**Reviewed required fields:** {progress['required_reviewed']} / {progress['required_total']}  \n"
        f"**Remaining required fields:** {remaining}  \n**Record complete:** {complete}"
    )
    if progress["record_complete"]:
        record += "  \nAll required fields have terminal judgments."
    return "\n".join(lines), record


def build_semantic_gui(
    gr: Any,
    records: list[dict[str, Any]],
    corrections: Path,
    resume: int | None,
    opened_at: dict[int, tuple[float, str]],
    contract: dict[str, Any],
    mode: str,
) -> Any:
    """Build, but do not launch, the semantic GUI so callbacks are integration-testable."""

    def render(index: int | None, field_name: str, zoom: int = DEFAULT_ZOOM) -> tuple[Any, ...]:
        if index is None:
            disabled = gr.update(interactive=False)
            return (
                None,
                "",
                "",
                "**Review complete.**",
                "REVIEW COMPLETE",
                None,
                uuid.uuid4().hex,
                gr.update(choices=list(PREFILL_FIELDS), value=None, interactive=False),
                "",
                "",
                {},
                gr.update(label="Correction", value="", interactive=False),
                "",
                "",
                gr.update(choices=[]),
                "",
                disabled,
                disabled,
                disabled,
                disabled,
                disabled,
                disabled,
                disabled,
                "",
                "**Record complete:** yes",
                "No fields are available for bulk acceptance.",
                disabled,
            )
        index = max(0, min(len(records) - 1, int(index)))
        record = records[index]
        events = load_review_events(corrections, strict=False)
        quality = _quality_for_record(record, records, events)
        progress = semantic_field_progress(record, events, quality)
        field_name = (
            field_name if field_name in record["fields"] else (progress["next_unresolved_field"] or "canonical_object")
        )
        panel = gui_field_view(record, field_name, corrections)
        opened_at.setdefault(index, (time.monotonic(), _utc_now()))
        ready, readiness_reasons = semantic_readiness(record)
        available = field_action_availability(record, field_name, blind_locked=panel["blind_locked"])
        value = panel["proposed_value"]
        proposal_text = "No value proposed" if value is None else json.dumps(value, ensure_ascii=False)
        state_labels = {
            "model_abstained": "Model abstained",
            "known": "Known proposal",
            "missing_prediction": "Missing prediction",
        }
        evidence = ", ".join(panel["evidence_summary"]) or "None"
        proposal_card = (
            f"### Model proposal\n{proposal_text}\n\n**Model state:** {state_labels.get(panel['value_state'], panel['value_state'])}  \n"
            f"**Reason:** `{panel['reason']}`  \n**Uncertainty:** {panel['uncertainty'] or 'not scored'}/20  \n"
            f"**Evidence:** {evidence}"
        )
        required_order = progress["completion"]["required_fields"]
        position = required_order.index(field_name) + 1 if field_name in required_order else len(required_order) + 1
        header = (
            f"## Editing field: `{field_name}`\nField {position} of {len(record['fields'])} required/reviewable fields"
        )
        checklist, record_progress = _progress_markdown(progress)
        unresolved = set(progress["remaining_required_fields"])
        bulk_accept = [
            name
            for name in progress["completion"]["required_fields"]
            if name in unresolved
            and validate_semantic_field(record, name).valid
            and validate_semantic_field(record, name).value_state == "known"
        ]
        bulk_remaining = [name for name in progress["remaining_required_fields"] if name not in bulk_accept]
        bulk_preview = (
            "**Will accept:** " + (", ".join(bulk_accept) or "none") + "  \n"
            "**Will remain unresolved:** " + (", ".join(bulk_remaining) or "none")
        )
        disabled = [name for name in ("accept", "model_abstention", "correction") if not available[name]]
        explanations = "  \n".join(f"**{name} disabled:** {available['reasons'][name]}" for name in disabled)
        details = {**panel["details"], "readiness_reasons": readiness_reasons, "exact_field": panel}

        def button(label: str, enabled: bool) -> Any:
            return gr.update(value=label, interactive=bool(ready and enabled))

        unavailable = "" if ready else "**PREDICTION NOT AVAILABLE.** " + ", ".join(readiness_reasons)
        return (
            index,
            record["sprite_id"],
            pixel_preview_html(record["image_path"], int(zoom)),
            gui_record_summary(record),
            unavailable,
            field_name,
            uuid.uuid4().hex,
            gr.update(choices=list(record["fields"]), value=field_name, interactive=ready),
            header,
            proposal_card,
            details,
            gr.update(
                label=f"Your corrected value for {field_name}", value="", interactive=ready and available["correction"]
            ),
            correction_format_hint(record, field_name),
            "",
            gr.update(
                choices=[json.dumps(item, ensure_ascii=False) for item in panel["alternatives"]],
                value=None,
                interactive=ready and available["alternative"] and bool(panel["alternatives"]),
            ),
            explanations,
            button(f"Accept model value for {field_name}", available["accept"]),
            button(f"Confirm model abstention for {field_name}", available["model_abstention"]),
            button(f"Select alternative for {field_name}", available["alternative"] and bool(panel["alternatives"])),
            button(f"Save correction for {field_name}", available["correction"]),
            button(f"Mark {field_name} as human abstention", available["human_abstention"]),
            button(f"Mark {field_name} unsupported", available["unsupported"]),
            button(f"Mark {field_name} not applicable", available["not_applicable"]),
            checklist,
            record_progress,
            bulk_preview,
            gr.update(value="Next record", interactive=bool(progress["record_complete"] and index < len(records) - 1)),
        )

    def perform(
        index: int | None,
        field_name: str,
        alternative: str,
        edited: str,
        zoom: int,
        submission_token: str,
        displayed_field: str,
        action: str,
    ) -> tuple[Any, ...]:
        if index is None:
            return "Nothing was saved. Semantic review is complete.", *render(None, field_name, zoom)
        record = records[int(index)]
        if field_name != displayed_field:
            return "Nothing was saved. Field selection changed; review the newly displayed field.", *render(
                index, field_name, zoom
            )
        events = load_review_events(corrections, strict=False)
        quality = _quality_for_record(record, records, events)
        kwargs = {
            "reviewer_id": "assisted_v4_semantic_gui",
            "session_id": str(record.get("audit_id", "")),
            "metadata": {
                **_timing_metadata(record, int(index), opened_at),
                "audit_id": record.get("audit_id"),
                "review_mode": mode,
                "submission_token": submission_token,
            },
        }
        try:
            if action == "correction":
                result = save_field_correction(
                    record,
                    field_name,
                    edited,
                    corrections,
                    {
                        "displayed_field": displayed_field,
                        "sprite_id": record["sprite_id"],
                        "audit_record_id": record.get("audit_id"),
                        "proposal_hash": panel_hash(record),
                        "review_mode": mode,
                        "submission_token": submission_token,
                        "reviewer_id": kwargs["reviewer_id"],
                        "quality": quality,
                        "metadata": kwargs["metadata"],
                    },
                )
                message = result.message
            else:
                panel = gui_field_view(record, field_name, corrections)
                available = field_action_availability(record, field_name, blind_locked=panel["blind_locked"])
                if not available[action]:
                    raise ValueError(available["reasons"].get(action, f"{action} is unavailable for this field"))
                if action == "accept":
                    event = accept_proposal(corrections, record, field_name, **kwargs)
                elif action == "model_abstention":
                    event = accept_model_abstention(corrections, record, field_name, **kwargs)
                elif action == "alternative":
                    if not alternative:
                        raise ValueError("select an alternative first")
                    event = select_alternative(corrections, record, field_name, json.loads(alternative), **kwargs)
                elif action == "human_abstention":
                    event = abstain_field(corrections, record, field_name, **kwargs)
                elif action == "unsupported":
                    event = mark_unsupported(corrections, record, field_name, **kwargs)
                elif action == "not_applicable":
                    event = mark_not_applicable(corrections, record, field_name, **kwargs)
                else:
                    raise ValueError(action)
                message = f"Saved {field_name}: {event.human_outcome}\nEvent: {event.event_id}"
            refreshed = semantic_field_progress(record, load_review_events(corrections, strict=False), quality)
            next_field = refreshed["next_unresolved_field"] or field_name
            return message, *render(index, next_field, zoom)
        except Exception as exc:  # every callback failure must be visible and non-silent
            return f"Nothing was saved.\n{exc}", *render(index, field_name, zoom)

    def panel_hash(record: dict[str, Any]) -> str:
        return str(compact_review_presenter(record)["raw_proposal_hash"])

    def accept_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "accept")

    def abstention_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "model_abstention")

    def alternative_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "alternative")

    def correction_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "correction")

    def human_abstention_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "human_abstention")

    def unsupported_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "unsupported")

    def not_applicable_callback(i: Any, f: str, a: str, e: str, z: int, t: str, d: str) -> tuple[Any, ...]:
        return perform(i, f, a, e, z, t, d, "not_applicable")

    def preview_callback(index: int | None, field_name: str, edited: str) -> str:
        if index is None or not edited.strip():
            return ""
        return preview_field_correction(records[int(index)], field_name, edited)

    def navigate_callback(index: int | None, field_name: str, zoom: int, delta: int) -> tuple[Any, ...]:
        if index is None:
            return "No record to navigate from.", *render(None, field_name, zoom)
        record = records[int(index)]
        events = load_review_events(corrections, strict=False)
        progress = semantic_field_progress(record, events, _quality_for_record(record, records, events))
        if not progress["record_complete"]:
            return "Navigation blocked. Resolve all required fields on this record first.", *render(
                index, field_name, zoom
            )
        target = int(index) + delta
        return "", *render(target if 0 <= target < len(records) else index, field_name, zoom)

    def previous_callback(index: int | None, field_name: str, zoom: int) -> tuple[Any, ...]:
        return navigate_callback(index, field_name, zoom, -1)

    def next_callback(index: int | None, field_name: str, zoom: int) -> tuple[Any, ...]:
        return navigate_callback(index, field_name, zoom, 1)

    def bulk_callback(index: int | None, field_name: str, zoom: int) -> tuple[Any, ...]:
        if index is None:
            return "Nothing was saved. Semantic review is complete.", *render(None, field_name, zoom)
        record = records[int(index)]
        events = load_review_events(corrections, strict=False)
        quality = _quality_for_record(record, records, events)
        progress = semantic_field_progress(record, events, quality)
        targets = [
            name
            for name in progress["remaining_required_fields"]
            if validate_semantic_field(record, name).valid
            and validate_semantic_field(record, name).value_state == "known"
        ]
        if not targets:
            return "Nothing was saved. No currently valid proposed critical fields can be accepted.", *render(
                index, field_name, zoom
            )
        try:
            for name in targets:
                accept_proposal(
                    corrections,
                    record,
                    name,
                    reviewer_id="assisted_v4_semantic_gui",
                    session_id=str(record.get("audit_id", "")),
                    metadata={
                        **_timing_metadata(record, int(index), opened_at),
                        "audit_id": record.get("audit_id"),
                        "review_mode": mode,
                        "bulk_action": True,
                    },
                )
            refreshed = semantic_field_progress(record, load_review_events(corrections, strict=False), quality)
            remaining = ", ".join(refreshed["remaining_required_fields"]) or "none"
            return f"Accepted: {', '.join(targets)}. Remaining required fields: {remaining}", *render(
                index, refreshed["next_unresolved_field"] or field_name, zoom
            )
        except Exception as exc:
            return f"Bulk action stopped: {exc}", *render(index, field_name, zoom)

    with _blocks(gr, "Sprite Lab Labeling v4 Semantic Review") as demo:
        index_state = gr.State(resume)
        displayed_field = gr.State("canonical_object")
        submission_token = gr.State(uuid.uuid4().hex)
        gr.Markdown(f"# {contract['banner']}")
        if mode == "manual_truth_diagnostic":
            gr.Markdown("**Diagnostics are excluded from assisted-model accuracy denominators.**")
        with gr.Row():
            previous = gr.Button("Previous record")
            next_ = gr.Button("Next record", interactive=False)
            sprite_id = gr.Textbox(label="Sprite", interactive=False)
            zoom = gr.Dropdown(label="Zoom", choices=list(ZOOM_CHOICES), value=DEFAULT_ZOOM)
        preview = gr.HTML(label="Nearest-neighbor sprite preview")
        summary = gr.Markdown()
        unavailable = gr.Markdown()
        field = gr.Dropdown(label="Field to review", choices=list(PREFILL_FIELDS), value="canonical_object")
        field_header = gr.Markdown(elem_classes=["field-header"])
        proposal_card = gr.Markdown()
        with gr.Accordion("Technical diagnostics", open=False):
            details = gr.JSON(label="Exact proposal and identity values")
        edited = gr.Textbox(label="Your corrected value")
        format_hint = gr.Markdown()
        parsed_preview = gr.Markdown()
        alternatives = gr.Dropdown(label="Visible model alternatives", choices=[])
        disabled_reasons = gr.Markdown()
        with gr.Row():
            accept = gr.Button("Accept model value")
            accept_abstention = gr.Button("Confirm model abstention")
            choose = gr.Button("Select alternative")
            correction = gr.Button("Save correction")
        with gr.Row():
            human_abstention = gr.Button("Mark as human abstention")
            unsupported = gr.Button("Mark unsupported")
            not_applicable = gr.Button("Mark not applicable")
        gr.Markdown("### Field progress")
        checklist = gr.Markdown()
        record_progress = gr.Markdown()
        with gr.Accordion("Advanced / Bulk actions", open=False):
            bulk_preview = gr.Markdown()
            accept_critical = gr.Button("Accept all currently valid proposed critical fields")
        status = gr.Textbox(label="Save result", interactive=False, lines=3)
        outputs = [
            index_state,
            sprite_id,
            preview,
            summary,
            unavailable,
            displayed_field,
            submission_token,
            field,
            field_header,
            proposal_card,
            details,
            edited,
            format_hint,
            parsed_preview,
            alternatives,
            disabled_reasons,
            accept,
            accept_abstention,
            choose,
            correction,
            human_abstention,
            unsupported,
            not_applicable,
            checklist,
            record_progress,
            bulk_preview,
            next_,
        ]
        action_inputs = [index_state, field, alternatives, edited, zoom, submission_token, displayed_field]
        demo.load(lambda: render(resume, "canonical_object"), outputs=outputs)
        field.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        zoom.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        edited.change(preview_callback, inputs=[index_state, field, edited], outputs=[parsed_preview], queue=False)
        previous.click(previous_callback, inputs=[index_state, field, zoom], outputs=[status, *outputs])
        next_.click(next_callback, inputs=[index_state, field, zoom], outputs=[status, *outputs])
        accept.click(accept_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="accept_field")
        accept_abstention.click(
            abstention_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="confirm_field_abstention"
        )
        choose.click(
            alternative_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="select_alternative"
        )
        correction.click(
            correction_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="save_field_correction"
        )
        human_abstention.click(
            human_abstention_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="human_abstention"
        )
        unsupported.click(
            unsupported_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="unsupported"
        )
        not_applicable.click(
            not_applicable_callback, inputs=action_inputs, outputs=[status, *outputs], api_name="not_applicable"
        )
        accept_critical.click(bulk_callback, inputs=[index_state, field, zoom], outputs=[status, *outputs])
    return demo


def _launch_semantic_gui(
    gr: Any,
    records: list[dict[str, Any]],
    corrections: Path,
    resume: int | None,
    opened_at: dict[int, tuple[float, str]],
    contract: dict[str, Any],
    mode: str,
    host: str,
    port: int,
    share: bool,
) -> Any:
    demo = build_semantic_gui(gr, records, corrections, resume, opened_at, contract, mode)
    return _launch_gradio(demo, host, port, share)


def audit_review_metrics(records_path: str | Path, corrections_path: str | Path) -> dict[str, Any]:
    records = _read_jsonl(Path(records_path))
    events = load_review_events(corrections_path, strict=False)
    actions = Counter(event.action for event in events)
    durations = [event.review_duration_seconds for event in events if event.review_duration_seconds is not None]
    return {
        **calibration_denominator_report(records, events),
        "field_correctness": None,
        "field_edit_count": actions.get("edit", 0),
        "review_time_seconds_total": sum(durations),
        "review_time_seconds_mean": sum(durations) / len(durations) if durations else None,
        "events": len(events),
    }


def _quality_for_record(record: dict[str, Any], records: list[dict[str, Any]], events: Any) -> QualityResolution:
    decision = resolve_quality_decisions(records, events)[record["sprite_id"]]
    if decision.effective_state == "quality_unreviewed" and record.get("quality_state") in QUALITY_ELIGIBLE:
        return QualityResolution(record["sprite_id"], str(record["quality_state"]), None, 0, 0)
    return decision


def _critical_judgment_exists(record: dict[str, Any], events: Any) -> bool:
    return any(
        event.sprite_id == record["sprite_id"]
        and event.field_name in {"canonical_object", "category", "domain", "role"}
        for event in events
    )


def _timing_metadata(record: dict[str, Any], index: int, opened_at: dict[int, tuple[float, str]]) -> dict[str, Any]:
    monotonic, started = opened_at.get(index, (time.monotonic(), _utc_now()))
    return {
        "review_mode": record.get("review_mode", "assisted"),
        "proposal_visible_before_judgment": record.get("proposal_visible_before_judgment", True),
        "review_started_at": started,
        "review_completed_at": _utc_now(),
        "review_duration_seconds": round(max(0.0, time.monotonic() - monotonic), 6),
        "prediction_state": record.get("prediction_state"),
    }


def _image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

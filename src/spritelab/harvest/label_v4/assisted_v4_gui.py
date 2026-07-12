"""Strict quality-only and semantic-only Labeling-v4 review GUIs."""

from __future__ import annotations

import base64
import io
import json
import time
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
    edit_field,
    load_review_events,
    mark_not_applicable,
    mark_unsupported,
    record_quality_decision,
    select_alternative,
)
from spritelab.harvest.label_v4.two_pass import (
    QUALITY_ELIGIBLE,
    QualityResolution,
    calibration_denominator_report,
    has_real_semantic_proposal,
    quality_resume_index,
    require_semantic_ready_records,
    resolve_quality_decisions,
    semantic_completion,
    semantic_readiness,
    validate_accept_all,
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


def review_resume_index(
    records: list[dict[str, Any]], corrections_path: str | Path, *, mode: str = "quality_only"
) -> int:
    events = load_review_events(corrections_path, strict=False)
    normalized = normalize_review_mode(mode)
    if normalized == "quality_only":
        return quality_resume_index(records, events)
    quality = resolve_quality_decisions(records, events)
    for index, record in enumerate(records):
        sprite_id = str(record.get("sprite_id", ""))
        decision = quality[sprite_id]
        if decision.effective_state == "quality_unreviewed" and record.get("quality_state") in QUALITY_ELIGIBLE:
            decision = QualityResolution(sprite_id, str(record["quality_state"]), None, 0, 0)
        if not semantic_completion(record, events, decision)["complete"]:
            return index
    return 0


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
        f'style="{style}"></div><figcaption>Full canvas — native {width}\u00d7{height}, zoom {zoom}\u00d7</figcaption></figure>'
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
    return (
        f"**Source suitability:** `{suitability.get('status', 'unknown')}`  \n"
        f"**Reason codes:** `{json.dumps(suitability.get('reason_codes', []))}`  \n"
        f"**Source:** `{source.get('source_id')}`  \n"
        f"**Pack:** `{source.get('pack_name') or source.get('pack_id')}`  \n"
        f"**Sheet/image:** `{source.get('source_sheet') or source.get('source_image')}`"
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
    resume: int,
    opened_at: dict[int, tuple[float, str]],
    contract: dict[str, Any],
    host: str,
    port: int,
    share: bool,
) -> Any:
    def render(index: int, zoom: int = DEFAULT_ZOOM) -> tuple[Any, ...]:
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

    def decide(index: int, outcome: str, zoom: int) -> tuple[Any, ...]:
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
        return f"Appended immutable {event.human_outcome}; advanced.", *render(int(index) + 1, zoom)

    with gr.Blocks(
        title="Sprite Lab Labeling v4 Quality Review",
        css=".sprite-pixel-viewer img{image-rendering:pixelated!important}",
    ) as demo:
        index_state = gr.State(0)
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
        previous.click(lambda i, z: render(int(i) - 1, z), inputs=[index_state, zoom], outputs=outputs)
        next_.click(lambda i, z: render(int(i) + 1, z), inputs=[index_state, zoom], outputs=outputs)
        for button, outcome in (
            (suitable, "quality_suitable"),
            (uncertain_usable, "quality_uncertain_usable"),
            (unsuitable, "quality_unsuitable"),
            (uncertain_not_usable, "quality_uncertain_not_usable"),
        ):
            button.click(
                lambda i, z, value=outcome: decide(i, value, z), inputs=[index_state, zoom], outputs=[status, *outputs]
            )
    return demo.launch(server_name=host, server_port=int(port), share=bool(share))


def _launch_semantic_gui(
    gr: Any,
    records: list[dict[str, Any]],
    corrections: Path,
    resume: int,
    opened_at: dict[int, tuple[float, str]],
    contract: dict[str, Any],
    mode: str,
    host: str,
    port: int,
    share: bool,
) -> Any:
    semantic_buttons_enabled = all(semantic_readiness(record)[0] for record in records)

    def render(index: int, field_name: str, zoom: int = DEFAULT_ZOOM) -> tuple[Any, ...]:
        index = max(0, min(len(records) - 1, int(index)))
        record = records[index]
        field_name = field_name if field_name in PREFILL_FIELDS else "canonical_object"
        panel = gui_field_view(record, field_name, corrections)
        opened_at.setdefault(index, (time.monotonic(), _utc_now()))
        ready, reasons = semantic_readiness(record)
        banner = "" if ready else "PREDICTION NOT AVAILABLE\nThis record cannot be semantically reviewed yet."
        proposed = (
            "(hidden until independent blind judgment)"
            if panel["blind_locked"]
            else json.dumps(panel["proposed_value"], ensure_ascii=False)
        )
        return (
            index,
            record["sprite_id"],
            pixel_preview_html(record["image_path"], int(zoom)),
            gui_record_summary(record),
            banner,
            gr.update(choices=list(PREFILL_FIELDS), value=field_name, interactive=ready),
            proposed,
            f"{panel['value_state']}: {panel['reason']}",
            gr.update(
                choices=[json.dumps(value, ensure_ascii=False) for value in panel["alternatives"]],
                value=None,
                interactive=ready,
            ),
            "hidden" if panel["blind_locked"] else str(panel["uncertainty"]),
            ", ".join(panel["evidence_summary"]) or "(none or hidden)",
            panel["review_state"],
            {**panel["details"], "readiness_reasons": reasons},
        )

    def act(index: int, field_name: str, action: str, alternative: str, edited: str, zoom: int) -> tuple[Any, ...]:
        record = records[int(index)]
        ready, reasons = semantic_readiness(record)
        if not ready:
            raise ValueError("PREDICTION NOT AVAILABLE: " + ", ".join(reasons))
        panel = gui_field_view(record, field_name, corrections)
        if panel["blind_locked"] and action in {"accept", "model_abstention", "alternative"}:
            raise ValueError("blind first judgment must be independent of the hidden proposal")
        kwargs = {
            "reviewer_id": "assisted_v4_semantic_gui",
            "session_id": str(record.get("audit_id", "")),
            "metadata": {**_timing_metadata(record, int(index), opened_at), "audit_id": record.get("audit_id")},
        }
        if action == "accept":
            event = accept_proposal(corrections, record, field_name, **kwargs)
        elif action == "model_abstention":
            event = accept_model_abstention(corrections, record, field_name, **kwargs)
        elif action == "alternative":
            if not alternative:
                raise ValueError("select an alternative first")
            event = select_alternative(corrections, record, field_name, json.loads(alternative), **kwargs)
        elif action == "edit":
            if not edited.strip():
                raise ValueError("enter an independent or corrected value first")
            try:
                value = json.loads(edited)
            except json.JSONDecodeError:
                value = edited.strip()
            event = edit_field(corrections, record, field_name, value, **kwargs)
        elif action == "human_abstention":
            event = abstain_field(corrections, record, field_name, **kwargs)
        elif action == "unsupported":
            event = mark_unsupported(corrections, record, field_name, **kwargs)
        elif action == "not_applicable":
            event = mark_not_applicable(corrections, record, field_name, **kwargs)
        else:  # pragma: no cover
            raise ValueError(action)
        events = load_review_events(corrections, strict=False)
        quality = _quality_for_record(record, records, events)
        complete = semantic_completion(record, events, quality)["complete"]
        next_index = int(index) + 1 if complete else int(index)
        return f"Appended {event.human_outcome}; semantic_complete={complete}.", *render(next_index, field_name, zoom)

    def accept_all(index: int, field_name: str, zoom: int) -> tuple[Any, ...]:
        record = records[int(index)]
        required = list(validate_accept_all(record))
        events = load_review_events(corrections, strict=False)
        quality = _quality_for_record(record, records, events)
        kwargs = {
            "reviewer_id": "assisted_v4_semantic_gui",
            "session_id": str(record.get("audit_id", "")),
            "metadata": {**_timing_metadata(record, int(index), opened_at), "audit_id": record.get("audit_id")},
        }
        for name in required:
            accept_proposal(corrections, record, name, **kwargs)
        completion = semantic_completion(record, load_review_events(corrections, strict=False), quality)
        if not completion["complete"]:
            raise RuntimeError("accept all invariant failed: " + ", ".join(completion["reasons"]))
        return f"Accepted {len(required)} critical fields; semantic complete.", *render(
            int(index) + 1, field_name, zoom
        )

    with gr.Blocks(
        title="Sprite Lab Labeling v4 Semantic Review",
        css=".sprite-pixel-viewer img{image-rendering:pixelated!important}",
    ) as demo:
        index_state = gr.State(0)
        gr.Markdown(f"# {contract['banner']}")
        if mode == "manual_truth_diagnostic":
            gr.Markdown("**Diagnostics are excluded from assisted-model accuracy denominators.**")
        with gr.Row():
            previous = gr.Button("Previous")
            next_ = gr.Button("Next")
            sprite_id = gr.Textbox(label="Sprite", interactive=False)
            zoom = gr.Dropdown(label="Zoom", choices=list(ZOOM_CHOICES), value=DEFAULT_ZOOM)
        preview = gr.HTML(label="Nearest-neighbor sprite preview")
        summary = gr.Markdown()
        unavailable = gr.Markdown()
        field = gr.Dropdown(label="Semantic field", choices=list(PREFILL_FIELDS), value="canonical_object")
        proposed = gr.Textbox(label="Model proposal", interactive=False)
        value_state = gr.Textbox(label="Value state / reason", interactive=False)
        alternatives = gr.Dropdown(label="Alternatives", choices=[])
        edited = gr.Textbox(label="Independent/corrected value (JSON or text)", interactive=semantic_buttons_enabled)
        uncertainty = gr.Textbox(label="Uncertainty", interactive=False)
        evidence = gr.Textbox(label="Evidence", interactive=False)
        review_state = gr.Textbox(label="Review state", interactive=False)
        details = gr.JSON(label="Readiness and provenance")
        with gr.Row():
            accept = gr.Button("Accept proposed value", interactive=semantic_buttons_enabled)
            accept_abstention = gr.Button("Accept model abstention", interactive=semantic_buttons_enabled)
            choose = gr.Button("Select alternative", interactive=semantic_buttons_enabled)
            edit = gr.Button("Save independent/edit judgment", interactive=semantic_buttons_enabled)
        with gr.Row():
            human_abstention = gr.Button("Mark human abstention", interactive=semantic_buttons_enabled)
            unsupported = gr.Button("Mark unsupported", interactive=semantic_buttons_enabled)
            not_applicable = gr.Button("Mark not applicable", interactive=semantic_buttons_enabled)
            accept_critical = gr.Button("Accept all required critical fields", interactive=semantic_buttons_enabled)
        status = gr.Textbox(label="Append-only semantic event", interactive=False)
        outputs = [
            index_state,
            sprite_id,
            preview,
            summary,
            unavailable,
            field,
            proposed,
            value_state,
            alternatives,
            uncertainty,
            evidence,
            review_state,
            details,
        ]
        demo.load(lambda: render(resume, "canonical_object"), outputs=outputs)
        field.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        zoom.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        previous.click(lambda i, f, z: render(int(i) - 1, f, z), inputs=[index_state, field, zoom], outputs=outputs)
        next_.click(lambda i, f, z: render(int(i) + 1, f, z), inputs=[index_state, field, zoom], outputs=outputs)
        for button, action in (
            (accept, "accept"),
            (accept_abstention, "model_abstention"),
            (choose, "alternative"),
            (edit, "edit"),
            (human_abstention, "human_abstention"),
            (unsupported, "unsupported"),
            (not_applicable, "not_applicable"),
        ):
            button.click(
                lambda i, f, a, e, z, selected=action: act(i, f, selected, a, e, z),
                inputs=[index_state, field, alternatives, edited, zoom],
                outputs=[status, *outputs],
            )
        accept_critical.click(accept_all, inputs=[index_state, field, zoom], outputs=[status, *outputs])
    return demo.launch(server_name=host, server_port=int(port), share=bool(share))


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

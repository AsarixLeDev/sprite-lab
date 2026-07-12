"""Record-oriented assisted review GUI for prepared Labeling-v4 audits."""

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
    CRITICAL_FIELDS,
    PREFILL_FIELDS,
    PREFILLED_AUDIT_SCHEMA,
    detect_audit_schema,
    require_prefilled_records,
)
from spritelab.harvest.label_v4.review import (
    abstain_field,
    accept_proposal,
    compact_review_presenter,
    edit_field,
    load_review_events,
    mark_not_applicable,
    mark_suitable_image,
    mark_uncertain_quality,
    mark_unsuitable_image,
    mark_unsupported,
    mark_wrong_taxonomy,
    select_alternative,
)

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


def load_assisted_records(path: str | Path, *, diagnostic_allow_selection: bool = False) -> list[dict[str, Any]]:
    rows = _read_jsonl(Path(path))
    if diagnostic_allow_selection and rows and detect_audit_schema(rows[0]) == AUDIT_SELECTION_SCHEMA:
        raise ValueError(
            "Diagnostic mode confirmed a raw audit selection manifest. It is intentionally non-reviewable; "
            "run label-v4-prepare-audit first."
        )
    require_prefilled_records(rows)
    return rows


def review_resume_index(records: list[dict[str, Any]], corrections_path: str | Path) -> int:
    """Resume after records explicitly saved complete, never after a partial field event."""

    events = load_review_events(corrections_path, strict=False)
    complete_ids = {event.sprite_id for event in events if event.metadata.get("record_completed")}
    return next((index for index, row in enumerate(records) if row["sprite_id"] not in complete_ids), 0)


def pixel_preview_html(image_path: str | Path, zoom: int = DEFAULT_ZOOM, *, include_crop: bool = True) -> str:
    """Return lossless PNG previews enlarged by browser nearest-neighbor rendering."""

    if int(zoom) not in ZOOM_CHOICES:
        raise ValueError(f"zoom must be one of {ZOOM_CHOICES}")
    path = Path(image_path)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        width, height = rgba.size
        alpha = rgba.getchannel("A")
        bbox = alpha.getbbox()
        crop = rgba.crop(bbox) if bbox and include_crop and bbox != (0, 0, width, height) else None
        full_uri = _image_data_uri(rgba)
        crop_uri = _image_data_uri(crop) if crop is not None else ""
    display_width = max(384, width * int(zoom)) if zoom != 1 else width
    display_height = max(384, height * int(zoom)) if zoom != 1 else height
    image_style = "image-rendering:pixelated;image-rendering:crisp-edges;object-fit:contain;display:block"
    cells = [
        f'<figure><div style="{CHECKERBOARD_CSS}display:inline-block">'
        f'<img alt="full sprite canvas" src="{full_uri}" width="{display_width}" height="{display_height}" '
        f'style="{image_style}"></div><figcaption>Full canvas — native {width}\u00d7{height}, zoom {zoom}\u00d7</figcaption></figure>'
    ]
    if crop is not None:
        crop_width, crop_height = crop.size
        cells.append(
            f'<figure><div style="{CHECKERBOARD_CSS}display:inline-block">'
            f'<img alt="tight foreground crop" src="{crop_uri}" width="{max(192, crop_width * int(zoom))}" '
            f'height="{max(192, crop_height * int(zoom))}" style="{image_style}"></div>'
            f"<figcaption>Tight foreground crop — {crop_width}\u00d7{crop_height}</figcaption></figure>"
        )
    return (
        '<div class="sprite-pixel-viewer" style="display:flex;gap:20px;align-items:flex-start;overflow:auto">'
        + "".join(cells)
        + "</div>"
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
    blind_locked = record.get("review_mode") == "blind" and not _critical_judgment_exists(record, events)
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
        "training_consequence": source.get("training_consequence")
        or rendered.get("training_consequence", "excluded_not_scorable"),
        "review_state": rendered.get("review_state", "unreviewed"),
        "blind_locked": blind_locked,
        "details": {
            "field_quality": record.get("label_quality", {}).get("fields", {}).get(field_name, {}),
            "model_provenance_preserved": True,
            "prediction_state": record.get("prediction_state"),
            "missing_stages": record.get("missing_stages", []),
            "raw_proposal_hash": view.get("raw_proposal_hash", ""),
        },
    }


def launch_assisted_v4(
    records_path: str | Path,
    corrections_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7862,
    share: bool = False,
    diagnostic_allow_selection: bool = False,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("assisted-v4 requires gradio; install the harvest-ui extra") from exc

    records = load_assisted_records(records_path, diagnostic_allow_selection=diagnostic_allow_selection)
    corrections = Path(corrections_path)
    resume_index = review_resume_index(records, corrections)
    opened_at: dict[int, tuple[float, str]] = {}

    def render(index: int, field_name: str, zoom: int = DEFAULT_ZOOM) -> tuple[Any, ...]:
        index = max(0, min(len(records) - 1, int(index)))
        record = records[index]
        field_name = field_name if field_name in PREFILL_FIELDS else PREFILL_FIELDS[0]
        panel = gui_field_view(record, field_name, corrections)
        opened_at.setdefault(index, (time.monotonic(), _utc_now()))
        events = load_review_events(corrections, strict=False)
        suitability = _suitability_decision(record, events)
        semantic_enabled = suitability == "suitable"
        proposed = (
            "(hidden until first blind judgment)"
            if panel["blind_locked"]
            else json.dumps(panel["proposed_value"], ensure_ascii=False)
        )
        state = f"{panel['value_state']}: {panel['reason']}"
        uncertainty = (
            "hidden for blind first judgment"
            if panel["blind_locked"]
            else (
                "not scorable" if panel["uncertainty"] is None else f"{panel['uncertainty']}/20 — {panel['risk_band']}"
            )
        )
        return (
            index,
            record["sprite_id"],
            pixel_preview_html(record["image_path"], int(zoom)),
            gui_record_summary(record),
            f"Source quality: {record['source_suitability']['status']} | reason codes: {', '.join(record['suitability_reason_codes']) or '(none)'} | human decision: {suitability}",
            gr.update(choices=list(PREFILL_FIELDS), value=field_name, interactive=semantic_enabled),
            proposed,
            state,
            gr.update(
                choices=[json.dumps(value, ensure_ascii=False) for value in panel["alternatives"]],
                value=None,
                interactive=semantic_enabled,
            ),
            uncertainty,
            ", ".join(panel["evidence_summary"]) or "(none or hidden)",
            panel["conflict_disposition"],
            panel["training_consequence"],
            panel["review_state"],
            panel["details"],
        )

    def quality_action(index: int, decision: str, field_name: str, zoom: int) -> tuple[Any, ...]:
        record = records[int(index)]
        kwargs = _review_kwargs(record, int(index), opened_at, completed=decision != "suitable")
        if decision == "suitable":
            event = mark_suitable_image(corrections, record, **kwargs)
            next_index = int(index)
        elif decision == "unsuitable":
            event = mark_unsuitable_image(corrections, record, **kwargs)
            next_index = min(len(records) - 1, int(index) + 1)
        else:
            event = mark_uncertain_quality(corrections, record, **kwargs)
            next_index = min(len(records) - 1, int(index) + 1)
        return (
            f"Appended {event.action}; semantic scoring {'enabled' if decision == 'suitable' else 'skipped'}.",
            *render(next_index, field_name, zoom),
        )

    def act(index: int, field_name: str, action: str, alternative: str, edited: str, zoom: int) -> tuple[Any, ...]:
        record = records[int(index)]
        events = load_review_events(corrections, strict=False)
        if _suitability_decision(record, events) != "suitable":
            raise ValueError("record suitability must be marked suitable before semantic review")
        kwargs = _review_kwargs(record, int(index), opened_at, completed=False)
        if action == "accept":
            event = accept_proposal(corrections, record, field_name, **kwargs)
        elif action == "alternative":
            if not alternative:
                raise ValueError("select an alternative first")
            event = select_alternative(corrections, record, field_name, json.loads(alternative), **kwargs)
        elif action == "edit":
            if not edited.strip():
                raise ValueError("enter an edited value first")
            try:
                value = json.loads(edited)
            except json.JSONDecodeError:
                value = edited.strip()
            event = edit_field(corrections, record, field_name, value, **kwargs)
        elif action == "abstain":
            event = abstain_field(corrections, record, field_name, **kwargs)
        elif action == "unsupported":
            event = mark_unsupported(corrections, record, field_name, **kwargs)
        elif action == "wrong_taxonomy":
            event = mark_wrong_taxonomy(corrections, record, field_name, **kwargs)
        elif action == "not_applicable":
            event = mark_not_applicable(corrections, record, field_name, **kwargs)
        else:  # pragma: no cover
            raise ValueError(action)
        return (
            f"Appended {event.human_outcome} for {event.sprite_id}:{field_name}",
            *render(int(index), field_name, zoom),
        )

    def accept_all(index: int, field_name: str, zoom: int) -> tuple[Any, ...]:
        record = records[int(index)]
        events = load_review_events(corrections, strict=False)
        if _suitability_decision(record, events) != "suitable":
            raise ValueError("record suitability must be marked suitable first")
        kwargs = _review_kwargs(record, int(index), opened_at, completed=True)
        accepted = 0
        for name in CRITICAL_FIELDS:
            field = record["fields"][name]
            if field["value_state"] == "known":
                accept_proposal(corrections, record, name, **kwargs)
                accepted += 1
        return (
            f"Accepted {accepted} displayed critical fields; record saved.",
            *render(min(len(records) - 1, int(index) + 1), field_name, zoom),
        )

    with gr.Blocks(
        title="Sprite Lab Labeling v4 Review", css=".sprite-pixel-viewer img{image-rendering:pixelated!important}"
    ) as demo:
        index_state = gr.State(0)
        gr.Markdown(f"# Labeling-v4 calibration review\nInput contract: `{PREFILLED_AUDIT_SCHEMA}`")
        with gr.Row():
            previous = gr.Button("Previous")
            next_ = gr.Button("Next")
            sprite_id = gr.Textbox(label="Sprite", interactive=False)
            zoom = gr.Dropdown(label="Zoom", choices=list(ZOOM_CHOICES), value=DEFAULT_ZOOM)
        preview = gr.HTML(label="Nearest-neighbor sprite preview")
        summary = gr.Markdown()
        suitability_status = gr.Textbox(label="Suitability first", interactive=False)
        with gr.Row():
            suitable = gr.Button("Suitable")
            unsuitable = gr.Button("Unsuitable — save and next")
            uncertain = gr.Button("Uncertain quality — save and next")
        with gr.Group():
            field = gr.Dropdown(label="Semantic field", choices=list(PREFILL_FIELDS), value=PREFILL_FIELDS[0])
            proposed = gr.Textbox(label="Normalized proposed value", interactive=False)
            value_state = gr.Textbox(label="Value state / reason", interactive=False)
            alternatives = gr.Dropdown(label="Alternatives", choices=[])
            edited = gr.Textbox(label="Edit only when needed (JSON or text)")
            uncertainty = gr.Textbox(label="Uncertainty / risk band", interactive=False)
            evidence = gr.Textbox(label="Evidence summary", interactive=False)
            conflicts = gr.Textbox(label="Conflict disposition", interactive=False)
            training = gr.Textbox(label="Training consequence", interactive=False)
            review_state = gr.Textbox(label="Review state", interactive=False)
            with gr.Accordion("Expanded provenance details", open=False):
                details = gr.JSON(label="Details")
            with gr.Row():
                accept = gr.Button("Accept field")
                choose = gr.Button("Select alternative")
                edit = gr.Button("Save edit")
                abstain = gr.Button("Accept abstention")
            with gr.Row():
                unsupported = gr.Button("Mark unsupported")
                wrong_taxonomy = gr.Button("Mark wrong taxonomy")
                not_applicable = gr.Button("Mark not applicable")
                accept_critical = gr.Button("Accept all displayed critical fields; save + next")
        status = gr.Textbox(label="Append-only audit status", interactive=False)
        outputs = [
            index_state,
            sprite_id,
            preview,
            summary,
            suitability_status,
            field,
            proposed,
            value_state,
            alternatives,
            uncertainty,
            evidence,
            conflicts,
            training,
            review_state,
            details,
        ]
        demo.load(lambda: render(resume_index, PREFILL_FIELDS[0], DEFAULT_ZOOM), outputs=outputs)
        field.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        zoom.change(render, inputs=[index_state, field, zoom], outputs=outputs)
        previous.click(lambda i, f, z: render(int(i) - 1, f, z), inputs=[index_state, field, zoom], outputs=outputs)
        next_.click(lambda i, f, z: render(int(i) + 1, f, z), inputs=[index_state, field, zoom], outputs=outputs)
        for button, decision in ((suitable, "suitable"), (unsuitable, "unsuitable"), (uncertain, "uncertain_quality")):
            button.click(
                lambda i, f, z, d=decision: quality_action(i, d, f, z),
                inputs=[index_state, field, zoom],
                outputs=[status, *outputs],
            )
        for button, action in (
            (accept, "accept"),
            (choose, "alternative"),
            (edit, "edit"),
            (abstain, "abstain"),
            (unsupported, "unsupported"),
            (wrong_taxonomy, "wrong_taxonomy"),
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
    records = {str(row.get("sprite_id", "")): row for row in _read_jsonl(Path(records_path))}
    events = load_review_events(corrections_path, strict=False)
    actions = Counter(event.action for event in events)
    outcomes = Counter(event.human_outcome for event in events)
    durations = [event.review_duration_seconds for event in events if event.review_duration_seconds is not None]
    return {
        "field_correctness": None,
        "field_edit_count": actions.get("edit", 0),
        "accept_as_is_rate": actions.get("accept_proposal", 0) / len(events) if events else None,
        "review_time_seconds_total": sum(durations),
        "review_time_seconds_mean": sum(durations) / len(durations) if durations else None,
        "human_outcomes": dict(outcomes),
        "events": len(events),
        "records_in_scope": len(records),
        "note": "field correctness remains null until reviewed values are compared with adjudicated truth",
    }


def _critical_judgment_exists(record: dict[str, Any], events: Any) -> bool:
    return any(event.sprite_id == record["sprite_id"] and event.field_name in CRITICAL_FIELDS for event in events)


def _suitability_decision(record: dict[str, Any], events: Any) -> str:
    for event in reversed(events):
        if event.sprite_id != record["sprite_id"] or event.field_name:
            continue
        return {
            "mark_suitable_image": "suitable",
            "mark_unsuitable_image": "unsuitable",
            "mark_uncertain_quality": "uncertain_quality",
        }.get(event.action, "pending")
    return "pending"


def _review_kwargs(
    record: dict[str, Any], index: int, opened_at: dict[int, tuple[float, str]], *, completed: bool
) -> dict[str, Any]:
    monotonic, started = opened_at.get(index, (time.monotonic(), _utc_now()))
    return {
        "reviewer_id": "assisted_v4_gui",
        "session_id": record.get("audit_id", ""),
        "metadata": {
            "review_mode": record.get("review_mode", "assisted"),
            "proposal_visible_before_judgment": record.get("proposal_visible_before_judgment", True),
            "review_started_at": started,
            "review_completed_at": _utc_now(),
            "review_duration_seconds": round(max(0.0, time.monotonic() - monotonic), 6),
            "record_completed": bool(completed),
            "prediction_state": record.get("prediction_state"),
        },
    }


def _image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

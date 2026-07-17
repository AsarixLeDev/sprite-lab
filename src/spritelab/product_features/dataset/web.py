"""Feature-owned FastAPI review router and no-build interface."""

from __future__ import annotations

import io
import json
import re
import secrets
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from starlette.requests import Request

from spritelab.dataset_v5.raw_inventory import file_sha256
from spritelab.evaluation.memorization import (
    HARD_EVIDENCE_CLASSES,
    REVIEW_REQUIRED_EVIDENCE_CLASSES,
    parse_evidence_class,
)
from spritelab.evaluation.memorization_review import append_bound_review_event
from spritelab.hierarchical_labeling.contracts import LabelEvidenceBundle
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.product import product_status
from spritelab.hierarchical_labeling.review import append_review_action, latest_events_by_record, load_review_events
from spritelab.hierarchical_labeling.taxonomy import load_default_taxonomy
from spritelab.hierarchical_labeling.technical import extract_technical_evidence
from spritelab.product_core import (
    ApprovedFolderError,
    ApprovedFolderStore,
    ProductEvent,
    ProductSettingsError,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
    VisionProvider,
    api_error,
    choose_native_folder,
    interactive_desktop_available,
    product_api,
)
from spritelab.product_features.dataset.intake import (
    DatasetInputError,
    DatasetIntakeService,
    discover_source_packs,
    inspect_dataset_folder,
    prepare_existing_dataset_labeling,
    save_sheet_decision,
)
from spritelab.product_features.dataset.managed import ManagedDatasetError, validate_managed_dataset_output
from spritelab.product_features.dataset.review import DatasetReviewStore, ReviewDecisionError
from spritelab.product_features.dataset.sheets import uniform_grid_plan
from spritelab.product_features.dataset.sidecar import (
    PackMetadataError,
    ensure_dataset_writes_outside_input,
    export_metadata_files,
    load_pack_metadata,
    merge_grouping_roots,
    save_pack_metadata,
    sidecar_is_applicable,
)
from spritelab.product_features.evaluation.checkpoints import (
    expected_dataset_identity,
    expected_training_view_identity,
)
from spritelab.product_features.evaluation.memorization_display import memorization_display
from spritelab.product_features.providers.state import passive_provider_projection
from spritelab.product_web.events import EventRepository
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, require_confined_path

ProviderFactory = Callable[[ProjectContext, Callable[[str], bool] | None], VisionProvider | None]


def _job_log(message: str) -> dict[str, str]:
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "message": message}


def _public_inspection(
    inspection: Mapping[str, Any],
    *,
    approval_id: str | None,
    folder_name: str,
) -> dict[str, Any]:
    public = {str(key): value for key, value in inspection.items() if key != "input_root"}
    public.update(
        {
            "approval_id": approval_id,
            "folder_name": folder_name or "Selected folder",
            "paths_exposed": False,
        }
    )
    return public


def _public_dataset_result(
    value: Mapping[str, Any],
    *,
    folder: Path,
    output: Path,
    project_root: Path,
    approval_id: str,
    folder_name: str,
    dataset_id: str,
) -> dict[str, Any]:
    public = _redact_local_paths(
        dict(value),
        (
            (folder.resolve(), folder_name or "Selected folder"),
            (output.resolve(), "[local dataset artifacts]"),
            (project_root.resolve(), "[local project]"),
        ),
    )
    data = public.get("data")
    if isinstance(data, dict):
        for key in (
            "input_root",
            "output_root",
            "review_queue",
            "machine_result",
            "static_report_data",
        ):
            data.pop(key, None)
        data.update(
            {
                "approval_id": approval_id,
                "folder_name": folder_name or "Selected folder",
                "dataset_id": dataset_id,
                "review_url": "/dataset/review",
                "paths_exposed": False,
            }
        )
    return public


def _public_review_queue(queue: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    replacements: list[tuple[Path, str]] = [(project_root.resolve(), "[local project]")]
    for key, replacement in (
        ("input_root", "Selected folder"),
        ("output_root", "[local dataset artifacts]"),
    ):
        raw = queue.get(key)
        if isinstance(raw, str) and raw:
            replacements.append((Path(raw).expanduser().resolve(), replacement))
    redacted = _redact_local_paths(dict(queue), tuple(replacements))
    public = {
        str(key): value
        for key, value in redacted.items()
        if key not in {"input_root", "output_root", "append_only_log"}
    }
    items = redacted.get("items")
    if isinstance(items, list):
        public["items"] = [
            {str(key): value for key, value in item.items() if key != "source_path"}
            for item in items
            if isinstance(item, Mapping)
        ]
    public["paths_exposed"] = False
    return public


def _full_review_queue(output: Path, *, context: ProjectContext) -> dict[str, Any]:
    """Join the exception queue with the complete manifest for an auditable full-dataset review."""

    queue = DatasetReviewStore(output, context=context).queue()
    queued = {
        str(item.get("item_id")): dict(item)
        for item in queue.get("items", ())
        if isinstance(item, Mapping) and item.get("item_id")
    }
    full: list[dict[str, Any]] = []
    for manifest_item in _read_labeling_items(output):
        item_id = str(manifest_item.get("item_id") or "")
        item = {**manifest_item, **queued.get(item_id, {})}
        item.setdefault("queue_kind", "dataset_audit")
        item.setdefault("default_visible", False)
        item["thumbnail_url"] = f"/dataset/review/thumb/{item_id}"
        suitability = item.get("suitability") if isinstance(item.get("suitability"), Mapping) else {}
        potential_bad = item.get("current_disposition") == "accepted" and bool(
            item.get("automatic_disposition") != "accepted"
            or item.get("technical_reasons")
            or item.get("reasons")
            or suitability.get("status") not in {None, "accept"}
            or item.get("queue_kind") == "near_duplicate_exception"
        )
        item["potential_bad"] = potential_bad
        full.append(item)
    combined = dict(queue)
    combined["items"] = full
    combined["complete_item_count"] = len(full)
    combined["potential_bad_count"] = sum(bool(item["potential_bad"]) for item in full)
    return combined


def _public_memorization_review(review: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    replacements: list[tuple[Path, str]] = [(project_root.resolve(), "[local project]")]
    for item in review.get("items", ()):
        if not isinstance(item, Mapping):
            continue
        for field in ("generated_image", "training_comparison_image", "candidate_bundle_path"):
            value = item.get(field)
            if isinstance(value, str) and value:
                path = Path(value).expanduser()
                if path.is_absolute():
                    replacements.append((path.resolve(), "[local review artifact]"))
    redacted = _redact_local_paths(dict(review), tuple(replacements))
    public_items = []
    for item in redacted.get("items", ()):
        if not isinstance(item, Mapping):
            continue
        pair_id = str(item.get("pair_id") or "")
        public_item = {
            str(key): value
            for key, value in item.items()
            if key not in {"generated_image", "training_comparison_image", "candidate_bundle_path"}
        }
        public_item["generated_image_url"] = f"/review/memorization/{pair_id}/image/generated"
        public_item["training_comparison_image_url"] = f"/review/memorization/{pair_id}/image/training"
        public_items.append(public_item)
    redacted["items"] = public_items
    redacted["paths_exposed"] = False
    return redacted


def _redact_local_paths(value: Any, replacements: tuple[tuple[Path, str], ...]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_local_paths(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_local_paths(child, replacements) for child in value]
    if isinstance(value, tuple):
        return [_redact_local_paths(child, replacements) for child in value]
    if isinstance(value, str):
        redacted = value
        ordered = sorted(replacements, key=lambda item: len(str(item[0])), reverse=True)
        for path, replacement in ordered:
            for spelling in {str(path), path.as_posix()}:
                redacted = redacted.replace(spelling, replacement)
        return redacted
    return value


def _public_exception_message(
    exc: BaseException,
    replacements: tuple[tuple[Path, str], ...],
) -> str:
    """Preserve actionable validation text without returning a local absolute path."""

    message = str(_redact_local_paths(str(exc), replacements)).strip()
    has_absolute_path = bool(
        re.search(r"(?i)[a-z]:[\\/]", message)
        or re.search(r"\\\\[^\\/\s]+[\\/]", message)
        or re.search(r"(?<![:/\w.])/(?!/)[^\s]+", message)
    )
    if has_absolute_path or not message:
        return "A local file or folder could not be processed safely. Reinspect the selected folder and try again."
    return message


class MetadataWizardOutcome(str, Enum):
    """Terminal outcome that authorizes, or prevents, CLI build continuation."""

    PENDING = "pending"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class MetadataWizardSession:
    """Thread-safe, server-owned lifecycle state for one CLI metadata wizard."""

    def __init__(self) -> None:
        self._outcome = MetadataWizardOutcome.PENDING
        self._lock = Lock()

    @property
    def outcome(self) -> MetadataWizardOutcome:
        with self._lock:
            return self._outcome

    def finish(self, outcome: MetadataWizardOutcome) -> bool:
        if outcome is MetadataWizardOutcome.PENDING:
            raise ValueError("A wizard session cannot finish as pending.")
        with self._lock:
            if self._outcome is outcome:
                return True
            if self._outcome is not MetadataWizardOutcome.PENDING:
                return False
            self._outcome = outcome
            return True


def build_review_router(
    context: ProjectContext,
    *,
    provider_factory: ProviderFactory | None = None,
    folder_chooser: Callable[[], str | Path | None] | None = None,
    approved_folders: ApprovedFolderStore | None = None,
) -> object:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

    router = APIRouter()
    dataset_config = context.config.get("dataset", {}) if isinstance(context.config, Mapping) else {}
    configured_roots = dataset_config.get("import_roots", ()) if isinstance(dataset_config, Mapping) else ()
    if isinstance(configured_roots, (str, Path)):
        configured_roots = (configured_roots,)
    import_roots = tuple(
        path if path.is_absolute() else context.project_root / path
        for value in configured_roots
        if str(value).strip()
        for path in (Path(str(value)).expanduser(),)
    )
    store = approved_folders or ApprovedFolderStore(context, import_roots=import_roots)
    settings_repository = ProductSettingsRepository(context)
    events = EventRepository(context.runs_directory, private_roots=(context.project_root,))
    build_jobs: dict[str, dict[str, Any]] = {}
    build_jobs_lock = Lock()
    existing_dataset_jobs: dict[str, dict[str, Any]] = {}
    existing_dataset_jobs_lock = Lock()
    labeling_jobs: dict[str, dict[str, Any]] = {}
    labeling_jobs_lock = Lock()
    active_labeling_job: dict[str, str] = {}
    active_selection_lock = Lock()
    try:
        saved_dataset_settings, _saved_dataset_version, _dataset_settings_saved = (
            settings_repository.effective_settings("dataset")
        )
    except ProductSettingsError:
        saved_dataset_settings = {}
    active_selection: dict[str, str] = {"mode": "existing"} if saved_dataset_settings.get("selected_root") else {}
    router.spritelab_approved_folders = store
    pending_input = dataset_config.get("pending_input_root") if isinstance(dataset_config, Mapping) else None
    pending_input_root = Path(str(pending_input)).expanduser().resolve() if pending_input else None

    def effective_context() -> ProjectContext:
        """Return project configuration overlaid with current web settings."""

        try:
            return settings_repository.effective_context()
        except ProductSettingsError:
            return context

    def public_error(
        exc: BaseException,
        *,
        folder: Path | None = None,
        output: Path | None = None,
    ) -> str:
        replacements = [(context.project_root.resolve(), "[local project]")]
        if context.runs_directory is not None:
            replacements.append((context.runs_directory.resolve(), "[local run state]"))
        if folder is not None:
            replacements.append((folder.resolve(), "Selected folder"))
        if output is not None:
            replacements.append((output.resolve(), "[local dataset artifacts]"))
        return _public_exception_message(exc, tuple(replacements))

    def update_build_job(
        job: dict[str, Any],
        *,
        status: str,
        message: str,
        log: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        with build_jobs_lock:
            job.update(status=status, message=message)
            job["logs"].append(_job_log(log))
            if result is not None:
                job["result"] = result

    def update_existing_dataset_job(
        job: dict[str, Any],
        *,
        status: str,
        message: str,
        log: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        with existing_dataset_jobs_lock:
            job.update(status=status, message=message)
            job["logs"].append(_job_log(log))
            if result is not None:
                job["result"] = result

    def update_labeling_job(
        job: dict[str, Any],
        *,
        status: str,
        stage: str,
        progress: int,
        message: str,
        log: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        with labeling_jobs_lock:
            job.update(
                status=status,
                stage=stage,
                progress=max(0, min(100, int(progress))),
                message=message,
            )
            if log:
                job["logs"].append(_job_log(log))
            if result is not None:
                job["result"] = dict(result)
            if status in {"complete", "failed"}:
                job["finished_at"] = datetime.now(timezone.utc).isoformat()

    def public_labeling_job(job: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key
            in {
                "job_id",
                "status",
                "stage",
                "progress",
                "message",
                "logs",
                "result",
                "started_at",
                "finished_at",
            }
        }

    def metadata_wizard_session(request: Request) -> MetadataWizardSession | None:
        session = getattr(request.app.state, "spritelab_metadata_wizard_session", None)
        return session if isinstance(session, MetadataWizardSession) else None

    def wizard_active(request: Request) -> bool:
        session = metadata_wizard_session(request)
        return session is not None and session.outcome is MetadataWizardOutcome.PENDING

    def stop_wizard_server(request: Request) -> None:
        request_shutdown = getattr(request.app.state, "spritelab_request_shutdown", None)
        if callable(request_shutdown):
            request_shutdown()

    def metadata_folder(approval_id: str | None = None) -> Path:
        if approval_id:
            return store.resolve(approval_id)
        if pending_input_root is not None and pending_input_root.is_dir():
            return pending_input_root
        raise ApprovedFolderError("Choose the folder again before editing pack information.")

    def pack_record(folder: Path, pack_id: str) -> tuple[Any, list[Path]]:
        _root, paths, packs = discover_source_packs(folder, context=context)
        pack = next((value for value in packs if value.pack_id == pack_id), None)
        if pack is None:
            raise PackMetadataError("The pack boundary changed; inspect the folder again.")
        return pack, paths

    def review_item_record(item_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        output = find_dataset_output(context)
        if output is None:
            raise HTTPException(status_code=404, detail="No dataset review queue is available.")
        queue = DatasetReviewStore(output, context=context).queue()
        item = next((value for value in queue["items"] if value.get("item_id") == item_id), None)
        if item is None:
            item = next((value for value in _read_labeling_items(output) if value.get("item_id") == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Unknown review item.")
        return output, queue, item

    def existing_dataset_summary(output: Path) -> dict[str, Any]:
        DatasetReviewStore(output, context=context).queue()
        items = _read_labeling_items(output)
        counts: dict[str, int] = {}
        for item in items:
            disposition = str(item.get("current_disposition") or item.get("automatic_disposition") or "unknown")
            counts[disposition] = counts.get(disposition, 0) + 1
        return {
            "status": "selected",
            "folder_name": output.name,
            "item_count": len(items),
            "counts": counts,
            "review_url": "/dataset/review",
            "labeling_url": "/labeling",
            "paths_exposed": False,
            "message": "The existing imported dataset is active. No rebuild is required.",
        }

    def render(
        request: Request, template: str, values: Mapping[str, Any] | None = None, *, status_code: int = 200
    ) -> Any:
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(request, "dataset.intake", template, values, status_code=status_code)
        return None

    @router.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request) -> Any:
        settings_error = None
        try:
            labeling, version, saved = settings_repository.effective_settings("labeling")
        except ProductSettingsError as exc:
            labeling, version, saved = {}, 0, False
            settings_error = public_error(exc)
        return render(
            request,
            "settings.html",
            {
                "labeling_settings": {
                    "hierarchical_enabled": labeling.get("hierarchical_enabled") is True,
                    "hierarchical_profile": str(labeling.get("hierarchical_profile", "fast_local")),
                    "reference_cohort_size": labeling.get("reference_cohort_size", 400),
                },
                "configuration_version": version,
                "settings_saved": saved,
                "settings_error": settings_error,
            },
        )

    @router.post("/settings/api/labeling")
    @product_api
    def save_labeling_settings(payload: dict[str, Any]) -> JSONResponse:
        expected = {"hierarchical_enabled", "hierarchical_profile", "reference_cohort_size"}
        if set(payload) != expected:
            return api_error(
                422,
                "labeling_settings_invalid",
                "Hierarchical labeling settings are incomplete or contain an unknown field.",
            )
        enabled = payload.get("hierarchical_enabled")
        profile = payload.get("hierarchical_profile")
        cohort_size = payload.get("reference_cohort_size")
        if type(enabled) is not bool:
            return api_error(422, "labeling_settings_invalid", "Enabled must be true or false.")
        if profile not in {"fast_local", "balanced", "high_quality"}:
            return api_error(422, "labeling_settings_invalid", "Choose a supported labeling profile.")
        if type(cohort_size) is not int or not 300 <= cohort_size <= 500:
            return api_error(
                422,
                "labeling_settings_invalid",
                "Reference cohort size must be a whole number from 300 through 500.",
            )
        try:
            saved = settings_repository.save(
                "labeling",
                {
                    "hierarchical_enabled": enabled,
                    "hierarchical_profile": profile,
                    "reference_cohort_size": cohort_size,
                },
            )
        except ProductSettingsError as exc:
            return api_error(409, "labeling_settings_not_saved", public_error(exc))
        return JSONResponse(
            {
                "schema_version": "spritelab.labeling.product-settings-result.v1",
                "saved": True,
                "enabled": enabled,
                "configuration_version": saved["configuration_version"],
                "message": (
                    "Hierarchical labeling is enabled. Rebuild the dataset to generate hierarchical suggestions."
                    if enabled
                    else "Hierarchical labeling is disabled."
                ),
            }
        )

    @router.delete("/settings/api/labeling")
    @product_api
    def clear_labeling_settings() -> JSONResponse:
        try:
            settings_repository.clear("labeling")
        except ProductSettingsError as exc:
            return api_error(409, "labeling_settings_not_cleared", public_error(exc))
        enabled = context.config.get("labeling", {}).get("hierarchical_enabled") is True
        return JSONResponse(
            {
                "schema_version": "spritelab.labeling.product-settings-result.v1",
                "saved": False,
                "enabled": enabled,
                "message": "Project-file labeling defaults were restored.",
            }
        )

    @router.get("/dataset", response_class=HTMLResponse)
    def dataset_page(request: Request) -> Any:
        response = render(
            request,
            "dataset.html",
            {
                "folder_policy_message": "Sprite Lab can only read folders you explicitly choose.",
                "native_picker_available": interactive_desktop_available(),
                "configured_import_roots": len(store.import_roots),
            },
        )
        return response or HTMLResponse(_dataset_page())

    @router.get("/dataset/metadata", response_class=HTMLResponse)
    def metadata_page(request: Request, approval_id: str | None = None) -> Any:
        folder: Path | None = None
        try:
            folder = metadata_folder(approval_id)
            inspection = inspect_dataset_folder(folder, context=context)
        except (ApprovedFolderError, DatasetInputError, PackMetadataError) as exc:
            return HTMLResponse(
                _metadata_page(
                    {"packs": [], "error": public_error(exc, folder=folder)},
                    approval_id=None,
                    wizard_session_active=wizard_active(request),
                ),
                status_code=409,
            )
        public_inspection = _public_inspection(inspection, approval_id=approval_id, folder_name=folder.name)
        values = {
            "inspection": public_inspection,
            "approval_id": approval_id,
            "wizard_session_active": wizard_active(request),
        }
        response = render(request, "metadata.html", values)
        return response or HTMLResponse(
            _metadata_page(
                public_inspection,
                approval_id=approval_id,
                wizard_session_active=wizard_active(request),
            )
        )

    @router.get("/labeling", response_class=HTMLResponse)
    def labeling_page(request: Request) -> Any:
        current = effective_context()
        status = product_status(current.config, current.project_root)
        try:
            provider_projection = passive_provider_projection(current)
        except (ValueError, ProductSettingsError):
            provider_projection = {"state": "not_configured", "configured": False}
        response = render(
            request,
            "labeling.html",
            {
                "labeling_status": status,
                "reference_cohort_size": current.config.get("labeling", {}).get("reference_cohort_size", 400),
                "provider_projection": provider_projection,
            },
        )
        return response or HTMLResponse(_labeling_page(status))

    @router.get("/labeling/api/status")
    @product_api
    def hierarchical_status() -> JSONResponse:
        current = effective_context()
        return JSONResponse(product_status(current.config, current.project_root))

    @router.get("/labeling/api/taxonomy")
    @product_api
    def hierarchical_taxonomy() -> JSONResponse:
        graph = load_default_taxonomy()
        return JSONResponse(
            {
                "schema_version": "spritelab.labeling.product-taxonomy.v1",
                "version": graph.version,
                "unknown_policy": "abstain",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "display_name": node.display_name,
                        "parent_id": node.parent_id,
                        "depth": graph.depth(node.node_id),
                        "definition": node.definition,
                    }
                    for node in graph.nodes
                ],
            }
        )

    @router.get("/labeling/api/queue")
    @product_api
    def hierarchical_queue(limit: int = 50) -> JSONResponse:
        output = find_dataset_output(context)
        if output is None:
            return JSONResponse(
                {
                    "schema_version": "spritelab.labeling.product-queue.v1",
                    "items": [],
                    "message": "Build a dataset before opening semantic review.",
                }
            )
        try:
            items = _read_labeling_items(output)
            reviewed = latest_events_by_record(
                load_review_events(output / "hierarchical_labeling" / "review_events.jsonl")
            )
        except (OSError, ValueError, json.JSONDecodeError, ContractValidationError):
            return api_error(
                409,
                "labeling_queue_unavailable",
                "The semantic review queue could not be verified.",
                next_action="Rebuild the dataset, then reopen Labeling.",
            )
        graph = load_default_taxonomy()
        eligible = [
            item
            for item in items
            if item.get("current_disposition") == "accepted" and item.get("hierarchical_labeling")
        ]
        pending = []
        for item in eligible:
            semantic = item.get("semantic") if isinstance(item.get("semantic"), Mapping) else {}
            item_id = str(item.get("item_id") or "")
            if semantic.get("needs_review") is not True or item_id in reviewed:
                continue
            pending.append(item)
        rows = [
            _public_labeling_item(output, item, graph)
            for item in pending[: max(1, min(limit, 250))]
        ]
        auto_prefilled = sum(
            _semantic_triage_state(item) == "auto_prefilled" for item in eligible
        )
        provider_pending = sum(
            _semantic_triage_state(item) in {"provider_unavailable", "certification_blocked"}
            for item in eligible
        )
        if pending:
            queue_message = (
                f"{auto_prefilled} high-certainty labels were prefilled automatically. "
                f"{len(pending)} low-certainty labels are excluded from semantic supervision; add human truth only "
                "for an image you want to keep semantically."
            )
        elif provider_pending:
            queue_message = (
                f"No human review is requested. {provider_pending} images still need provider prefill; "
                "configure and test the vision provider, then prepare labeling again."
            )
        else:
            queue_message = f"All {auto_prefilled} eligible labels were prefilled automatically."
        return JSONResponse(
            {
                "schema_version": "spritelab.labeling.product-queue.v1",
                "items": rows,
                "message": queue_message,
                "auto_prefilled": auto_prefilled,
                "provider_prefill_pending": provider_pending,
                "low_certainty_remaining": len(pending),
                "human_truth_required_for_all": False,
                "low_certainty_default": "exclude_semantic_supervision",
                "internal_ids_required": False,
            }
        )

    @router.get("/labeling/api/thumb/{item_id}")
    @product_api
    def hierarchical_thumbnail(item_id: str) -> FileResponse | JSONResponse:
        output = find_dataset_output(context)
        if output is None:
            return api_error(404, "labeling_item_unavailable", "No semantic review item is available.")
        item = next((row for row in _read_labeling_items(output) if row.get("item_id") == item_id), None)
        if item is None:
            return api_error(404, "labeling_item_unavailable", "The semantic review item is not current.")
        try:
            path = _verified_dataset_image(output, item)
        except (OSError, ValueError) as exc:
            return api_error(409, "labeling_image_changed", str(exc), next_action="Rebuild the dataset.")
        return FileResponse(path, media_type="image/png")

    @router.get("/labeling/api/view/{item_id}/{render_type}")
    @product_api
    def hierarchical_render(item_id: str, render_type: str) -> FileResponse | JSONResponse:
        output = find_dataset_output(context)
        if output is None:
            return api_error(404, "labeling_render_unavailable", "No semantic render is available.")
        artifact = _labeling_artifact(output, item_id)
        views = artifact.get("render_bundle", {}).get("views", ()) if artifact else ()
        view = next((row for row in views if row.get("render_type") == render_type), None)
        if not isinstance(view, Mapping):
            return api_error(404, "labeling_render_unavailable", "The selected semantic render is unavailable.")
        path = Path(str(view.get("artifact_path", ""))).resolve()
        render_root = (output / "hierarchical_labeling" / "renders").resolve()
        try:
            path.relative_to(render_root)
        except ValueError:
            return api_error(404, "labeling_render_invalid", "The selected semantic render is invalid.")
        if not path.is_file() or file_sha256(path) != view.get("render_sha256"):
            return api_error(409, "labeling_render_changed", "The semantic render changed; rebuild it before review.")
        return FileResponse(path, media_type="image/png")

    @router.post("/labeling/api/review/{item_id}")
    @product_api
    def hierarchical_review(item_id: str, payload: dict[str, Any]) -> JSONResponse:
        output = find_dataset_output(context)
        if output is None:
            return api_error(404, "labeling_queue_unavailable", "No semantic review queue is available.")
        item = next((row for row in _read_labeling_items(output) if row.get("item_id") == item_id), None)
        if item is None or item.get("current_disposition") != "accepted":
            return api_error(404, "labeling_item_unavailable", "The semantic review item is not current.")
        legal_reasons = {"missing_source", "missing_license", "unverified_license"} & set(item.get("reasons", ()))
        if legal_reasons:
            return api_error(
                409,
                "labeling_legal_ineligible",
                "Semantic review cannot override source or license restrictions.",
            )
        action = str(payload.get("action", ""))
        if action not in {
            "accept_suggested_path",
            "choose_parent",
            "choose_alternative",
            "abstain",
            "mark_unusable",
            "flag_taxonomy_gap",
        }:
            return api_error(422, "labeling_action_invalid", "Choose one of the visible semantic review actions.")
        reviewer = str(payload.get("reviewer_identity", "")).strip()
        if not reviewer:
            return api_error(422, "labeling_reviewer_required", "Enter a reviewer identity before saving review.")
        try:
            path = _verified_dataset_image(output, item)
            graph = load_default_taxonomy()
            technical = extract_technical_evidence(path, record_identity=str(item_id))
            bundle = LabelEvidenceBundle(str(item_id), technical.image_identity, graph.identity, technical)
            selected_node = str(payload.get("selected_node", "")).strip() or None
            if action in {"abstain", "mark_unusable", "flag_taxonomy_gap"}:
                selected_node = None
            event = append_review_action(
                output / "hierarchical_labeling" / "review_events.jsonl",
                bundle,
                graph,
                action=action,
                reviewer_identity=reviewer,
                partition=str(payload.get("partition", "reference")),
                selected_node=selected_node,
                explicit_abstentions=tuple(str(value) for value in payload.get("explicit_abstentions", ())),
                render_identities=(str(item["hierarchical_labeling"]["render_bundle_identity"]),),
                review_notes=str(payload.get("review_notes", "")).strip() or None,
                review_confidence=payload.get("review_confidence"),
                exclude_semantic_supervision=payload.get("exclude_semantic_supervision") is True,
                legal_and_provenance_eligible=True,
                submission_token=str(payload.get("submission_token", "")).strip() or None,
            )
        except (OSError, KeyError, TypeError, ValueError, ContractValidationError) as exc:
            return api_error(
                422,
                "labeling_review_invalid",
                str(exc),
                next_action="Check the selected taxonomy node and review fields.",
            )
        return JSONResponse(
            {
                "schema_version": "spritelab.labeling.product-review-result.v1",
                "saved": True,
                "record_identity": item_id,
                "event_id": event.event_id,
                "event_hash": event.event_hash,
                "truth_status": "verified_append_only_human_review",
            }
        )

    @router.post("/labeling/api/actions/{action}")
    @product_api
    def hierarchical_action(action: str) -> JSONResponse:
        if action not in {"open_semantic_review", "run_automatic_labeling"}:
            return api_error(404, "labeling_action_unknown", "The requested labeling action is not available.")
        current = effective_context()
        status = product_status(current.config, current.project_root)
        if action == "run_automatic_labeling" and not status["enabled"]:
            return api_error(
                409,
                "hierarchical_labeling_disabled",
                "Enable hierarchical labeling in project settings before running automatic suggestions.",
            )
        if action == "open_semantic_review":
            return JSONResponse(
                {
                    "schema_version": "spritelab.labeling.product-action.v2",
                    "action": action,
                    "message": "Semantic review is ready below.",
                    "review_url": "/labeling#semantic-review",
                    "started": False,
                }
            )
        output = find_dataset_output(context)
        if output is None:
            return api_error(
                409,
                "labeling_dataset_required",
                "Build or select a dataset before preparing hierarchical labeling.",
                next_action="Open Dataset and choose the active dataset.",
            )
        try:
            queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
            input_root = Path(str(queue.get("input_root") or "")).resolve(strict=True)
            ensure_dataset_writes_outside_input(
                context.project_root,
                input_root,
                output_root=output,
                runs_directory=context.runs_directory,
            )
        except (OSError, ValueError, json.JSONDecodeError, PackMetadataError) as exc:
            return api_error(
                409,
                "labeling_dataset_invalid",
                public_error(exc, output=output),
                next_action="Rebuild or reselect the dataset before labeling.",
            )
        with labeling_jobs_lock:
            current_job = labeling_jobs.get(active_labeling_job.get("job_id", ""))
            if current_job and current_job.get("status") in {"queued", "running"}:
                return JSONResponse(public_labeling_job(current_job), status_code=202)
            job_id = secrets.token_urlsafe(18)
            job = {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "message": "Hierarchical labeling is queued.",
                "logs": [_job_log("Queued hierarchical labeling in the background.")],
                "result": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
            }
            labeling_jobs[job_id] = job
            active_labeling_job["job_id"] = job_id

        def run_labeling() -> None:
            last_log = {"stage": "", "bucket": -1}

            def progress(stage: str, current_count: int, total: int, message: str) -> None:
                if (
                    stage == "vision_provider"
                    and provider_state.get("state") != "not_configured"
                    and "not configured" in message.casefold()
                ):
                    state_label = str(provider_state.get("state", "unavailable")).replace("_", " ")
                    message = f"Vision provider stage skipped: {state_label}. Test the provider to enable descriptions."
                if stage == "inventory":
                    percent = 5 + round((current_count / max(total, 1)) * 15)
                    public_stage = "Verifying dataset"
                elif stage == "vision_provider":
                    percent = 25 + round((current_count / max(total, 1)) * 25)
                    public_stage = "Vision provider"
                else:
                    percent = 55 + round((current_count / max(total, 1)) * 40)
                    public_stage = "Preparing labeling"
                bucket = percent // 10
                should_log = stage != last_log["stage"] or bucket > last_log["bucket"] or current_count == total
                update_labeling_job(
                    job,
                    status="running",
                    stage=public_stage,
                    progress=percent,
                    message=message,
                    log=message if should_log else None,
                )
                if should_log:
                    last_log.update(stage=stage, bucket=bucket)

            try:
                update_labeling_job(
                    job,
                    status="running",
                    stage="Starting",
                    progress=4,
                    message="Loading the active dataset.",
                    log="Loaded the active dataset and verified its managed output boundary.",
                )
                runtime_context = effective_context()
                provider_state = passive_provider_projection(runtime_context)
                provider = (
                    provider_factory(runtime_context, None)
                    if provider_factory and provider_state.get("state") == "previously_verified"
                    else None
                )
                result = prepare_existing_dataset_labeling(
                    output,
                    context=runtime_context,
                    vision_provider=provider,
                    progress=progress,
                )
                DatasetReviewStore(output, context=runtime_context).refresh_projection()
                processed = int(result.get("processed", 0))
                provider_status = str(result.get("provider_status", "not_configured"))
                if provider is None and provider_state.get("state") != "not_configured":
                    provider_status = str(provider_state.get("state"))
                provider_message = {
                    "not_configured": (
                        "No vision provider was configured; technical evidence is ready without automatic descriptions."
                    ),
                    "available": "Automatic vision descriptions and hierarchical evidence are ready for review.",
                    "health_failed": (
                        "Technical evidence is ready, but the configured vision provider was unavailable."
                    ),
                    "configured_unverified": (
                        "Technical evidence is ready; test the saved vision provider before automatic descriptions."
                    ),
                    "stale_cached": (
                        "Technical evidence is ready; re-test the vision provider before automatic descriptions."
                    ),
                    "unavailable_cached": (
                        "Technical evidence is ready; the saved vision provider or selected model is unavailable."
                    ),
                    "authentication_required_cached": (
                        "Technical evidence is ready; the saved vision provider needs authentication."
                    ),
                    "certification_blocked": (
                        "Technical evidence is ready; provider use was blocked by the current labeling scope."
                    ),
                }.get(provider_status, "Hierarchical evidence is ready for review.")
                update_labeling_job(
                    job,
                    status="complete",
                    stage="Ready for review",
                    progress=100,
                    message=f"Prepared {processed} images for semantic review. {provider_message}",
                    log=(
                        f"Hierarchical labeling finished. {processed} images are ready for semantic review. "
                        f"{provider_message}"
                    ),
                    result={
                        "processed": processed,
                        "failures": int(result.get("failures", 0)),
                        "provider_status": provider_status,
                        "semantically_labeled": int(result.get("semantically_labeled", 0)),
                        "review_url": "/labeling#semantic-review",
                    },
                )
            except (OSError, ValueError, DatasetInputError, ProductSettingsError) as exc:
                update_labeling_job(
                    job,
                    status="failed",
                    stage="Needs attention",
                    progress=int(job.get("progress", 0)),
                    message=public_error(exc, folder=input_root, output=output),
                    log="Labeling stopped safely. Existing dataset files were preserved when possible.",
                )

        Thread(target=run_labeling, name=f"hierarchical-labeling-{job_id[:8]}", daemon=True).start()
        return JSONResponse(
            {
                **public_labeling_job(job),
                "schema_version": "spritelab.labeling.product-action.v2",
                "action": action,
                "started": True,
                "status_url": f"/labeling/api/jobs/{job_id}",
            },
            status_code=202,
        )

    @router.get("/labeling/api/jobs/current")
    @product_api
    def current_labeling_job() -> JSONResponse:
        with labeling_jobs_lock:
            job = labeling_jobs.get(active_labeling_job.get("job_id", ""))
            return JSONResponse({"job": public_labeling_job(job) if job else None})

    @router.get("/labeling/api/jobs/{job_id}")
    @product_api
    def labeling_job(job_id: str) -> JSONResponse:
        with labeling_jobs_lock:
            job = labeling_jobs.get(job_id)
            if job is None:
                return api_error(404, "labeling_job_not_found", "The labeling job is no longer available.")
            return JSONResponse(public_labeling_job(job))

    @router.post("/dataset/api/folders/choose")
    @product_api
    async def choose_folder(request: Request) -> JSONResponse:
        settings = getattr(request.app.state, "spritelab_settings", None)
        if settings is not None and not settings.is_loopback:
            return api_error(
                403,
                "native_picker_loopback_only",
                "Native folder selection is available only from the loopback desktop session.",
                recoverable=False,
                next_action="Use a configured import root or restart Sprite Lab on loopback.",
            )
        try:
            body = (
                await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
            )
        except (ValueError, json.JSONDecodeError):
            body = {}
        if isinstance(body, Mapping) and any(key in body for key in ("folder", "path", "directory")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Browser-supplied filesystem paths are not accepted.",
                recoverable=True,
                next_action="Use Choose image folder to open the native folder picker.",
            )
        try:
            selected = (folder_chooser or choose_native_folder)()
            if not selected:
                return api_error(
                    409,
                    "folder_selection_cancelled",
                    "No folder was selected.",
                    recoverable=True,
                    next_action="Choose image folder when you are ready.",
                )
            approval = store.approve(selected, source="native_picker")
            with active_selection_lock:
                active_selection.clear()
                active_selection.update(mode="source", approval_id=approval.approval_id)
        except ApprovedFolderError as exc:
            return api_error(409, "folder_selection_unavailable", public_error(exc))
        return JSONResponse(
            {
                "approval": approval.public_dict(),
                "message": "Sprite Lab can only read folders you explicitly choose.",
            }
        )

    @router.post("/dataset/api/existing/choose")
    @product_api
    async def choose_existing_dataset(request: Request) -> JSONResponse:
        settings = getattr(request.app.state, "spritelab_settings", None)
        if settings is not None and not settings.is_loopback:
            return api_error(
                403,
                "native_picker_loopback_only",
                "Existing dataset selection is available only from the loopback desktop session.",
                recoverable=False,
                next_action="Restart Sprite Lab on loopback before selecting a local dataset.",
            )
        try:
            body = (
                await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
            )
        except (ValueError, json.JSONDecodeError):
            body = {}
        if isinstance(body, Mapping) and any(key in body for key in ("folder", "path", "directory")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Browser-supplied filesystem paths are not accepted.",
                next_action="Use Select existing dataset to open the native folder picker.",
            )
        job_id = secrets.token_urlsafe(18)
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "Preparing the existing dataset selector.",
            "logs": [_job_log("Selection job queued; the web application remains available.")],
            "result": None,
        }
        with existing_dataset_jobs_lock:
            existing_dataset_jobs[job_id] = job

        def run_selection() -> None:
            output: Path | None = None
            try:
                update_existing_dataset_job(
                    job,
                    status="running",
                    message="Waiting for folder selection.",
                    log="Opening the native folder picker.",
                )
                selected = (folder_chooser or choose_native_folder)()
                if not selected:
                    raise ValueError("No dataset folder was selected.")
                update_existing_dataset_job(
                    job,
                    status="running",
                    message="Validating the selected dataset.",
                    log="Folder selected. Checking the imported dataset structure.",
                )
                output = validate_managed_dataset_output(
                    selected,
                    context=context,
                    require_datasets_root=True,
                )
                update_existing_dataset_job(
                    job,
                    status="running",
                    message="Reading the imported dataset manifests.",
                    log="Required files found. Loading and validating the review queue and item manifest.",
                )
                summary = existing_dataset_summary(output)
                items_count = int(summary["item_count"])
                update_existing_dataset_job(
                    job,
                    status="running",
                    message="Saving the project selection.",
                    log=f"Validated {items_count} imported item(s). Saving the active dataset selection.",
                )
                current, _version, _saved = settings_repository.effective_settings("dataset")
                settings_repository.save("dataset", {**current, "selected_root": str(output)})
                with active_selection_lock:
                    active_selection.clear()
                    active_selection.update(mode="existing")
                update_existing_dataset_job(
                    job,
                    status="complete",
                    message="Existing dataset ready.",
                    log="Selection complete. Review and labeling actions are now available.",
                    result=summary,
                )
            except (OSError, ValueError, json.JSONDecodeError, ManagedDatasetError, ProductSettingsError) as exc:
                update_existing_dataset_job(
                    job,
                    status="failed",
                    message=public_error(exc, output=output),
                    log="Selection failed safely. Choose a completed imported Sprite Lab dataset and try again.",
                )

        Thread(target=run_selection, name=f"dataset-select-{job_id[:8]}", daemon=True).start()
        return JSONResponse(
            {
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/dataset/api/existing/{job_id}",
                "message": "Dataset selection started in the background.",
            },
            status_code=202,
        )

    @router.get("/dataset/api/selection")
    @product_api
    def current_dataset_selection() -> JSONResponse:
        with active_selection_lock:
            selection = dict(active_selection)
        if selection.get("mode") == "source":
            try:
                approval_id = str(selection.get("approval_id") or "")
                folder = store.resolve(approval_id)
                inspection = inspect_dataset_folder(folder, context=context)
            except (ApprovedFolderError, DatasetInputError):
                return JSONResponse({"mode": "none", "paths_exposed": False})
            public = _public_inspection(inspection, approval_id=approval_id, folder_name=folder.name)
            public.update(mode="source", status="ready")
            return JSONResponse(public)
        if selection.get("mode") == "existing":
            output = find_dataset_output(context)
            if output is not None:
                try:
                    return JSONResponse({"mode": "existing", **existing_dataset_summary(output)})
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        return JSONResponse({"mode": "none", "paths_exposed": False})

    @router.get("/dataset/api/existing/{job_id}")
    @product_api
    def existing_dataset_job(job_id: str) -> JSONResponse:
        with existing_dataset_jobs_lock:
            job = existing_dataset_jobs.get(job_id)
            public = dict(job) if job is not None else None
            if public is not None:
                public["logs"] = [dict(entry) for entry in job["logs"]]
                public["result"] = dict(job["result"]) if isinstance(job.get("result"), Mapping) else None
        if public is None:
            return api_error(404, "existing_dataset_job_unknown", "The dataset selection job is unavailable.")
        return JSONResponse(public)

    @router.get("/dataset/api/folders/approved")
    @product_api
    def approved_folder_records() -> JSONResponse:
        return JSONResponse(
            {
                "approvals": store.public_records(),
                "configured_import_roots": len(store.import_roots),
                "paths_exposed": False,
            }
        )

    @router.post("/dataset/api/folders/import-root")
    @product_api
    def approve_import_root_folder(payload: dict[str, Any]) -> JSONResponse:
        if any(key in payload for key in ("folder", "path", "directory", "absolute_path")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Import-root selection accepts only a configured root number and safe relative folder name.",
                next_action="Use a configured project import root or the native folder picker.",
            )
        try:
            root_index = int(payload.get("root_index", -1))
            relative = str(payload.get("relative") or "")
            approval = store.approve_import_root_child(root_index, relative)
        except (TypeError, ValueError, ApprovedFolderError) as exc:
            return api_error(
                422,
                "import_root_selection_invalid",
                public_error(exc),
                next_action="Choose a folder beneath a configured import root.",
            )
        return JSONResponse(
            {
                "approval": approval.public_dict(),
                "message": "The configured import-root folder was approved for read-only access.",
            }
        )

    @router.post("/dataset/api/inspect")
    @product_api
    def inspect_folder(payload: dict[str, Any]) -> JSONResponse:
        if any(key in payload for key in ("folder", "path", "directory")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Raw filesystem paths are not accepted by the dataset web API.",
                recoverable=True,
                next_action="Choose the folder with the native folder picker.",
            )
        try:
            folder = store.resolve(str(payload.get("approval_id") or ""))
        except ApprovedFolderError as exc:
            return api_error(422, "folder_approval_invalid", public_error(exc), next_action="Choose the folder again.")
        try:
            inspection = inspect_dataset_folder(folder, context=context)
        except DatasetInputError as exc:
            return api_error(
                422,
                "approved_folder_boundary_invalid",
                public_error(exc, folder=folder),
                next_action="Choose another folder.",
            )
        inspection = _public_inspection(
            inspection,
            approval_id=str(payload.get("approval_id") or ""),
            folder_name=folder.name,
        )
        inspection["status"] = "NEEDS_INFORMATION" if inspection["wizard_required"] else "READY"
        inspection["next_action"] = "Complete pack information" if inspection["wizard_required"] else "Build dataset"
        inspection["approval_id"] = str(payload.get("approval_id"))
        return JSONResponse(inspection)

    @router.post("/dataset/api/metadata/inspect")
    @product_api
    def inspect_metadata(payload: dict[str, Any]) -> JSONResponse:
        folder: Path | None = None
        try:
            approval_id = str(payload.get("approval_id") or "") or None
            folder = metadata_folder(approval_id)
            inspection = inspect_dataset_folder(folder, context=context)
        except (ApprovedFolderError, DatasetInputError) as exc:
            return api_error(
                422,
                "folder_approval_invalid",
                public_error(exc, folder=folder),
                next_action="Choose the folder again.",
            )
        return JSONResponse(_public_inspection(inspection, approval_id=approval_id, folder_name=folder.name))

    @router.post("/dataset/api/metadata/save")
    @product_api
    def save_metadata(payload: dict[str, Any]) -> JSONResponse:
        folder: Path | None = None
        try:
            folder = metadata_folder(str(payload.get("approval_id") or "") or None)
            pack, _paths = pack_record(folder, str(payload.get("pack_id") or ""))
            fields = payload.get("metadata")
            if not isinstance(fields, Mapping):
                raise PackMetadataError("metadata must be an object.")
            covered = [file_sha256(folder / relative) for relative in pack.image_relative_paths]
            record = save_pack_metadata(
                context.project_root,
                folder,
                pack,
                fields,
                covered_byte_hashes=covered,
            )
        except (ApprovedFolderError, DatasetInputError, PackMetadataError, OSError) as exc:
            return api_error(
                422,
                "pack_metadata_invalid",
                public_error(exc, folder=folder),
                next_action="Correct this pack declaration and save it again.",
            )
        return JSONResponse(
            {
                "saved": True,
                "pack_id": pack.pack_id,
                "input_folder_written": False,
                "sidecar_schema": record["schema_version"],
                "inspection": _public_inspection(
                    inspect_dataset_folder(folder, context=context),
                    approval_id=str(payload.get("approval_id") or "") or None,
                    folder_name=folder.name,
                ),
            }
        )

    @router.post("/dataset/api/metadata/complete")
    @product_api
    def complete_metadata_wizard(request: Request, payload: dict[str, Any]) -> JSONResponse:
        wizard_session = metadata_wizard_session(request)
        if wizard_session is None:
            return api_error(
                409,
                "metadata_wizard_session_unavailable",
                "No command is waiting for this metadata wizard.",
                recoverable=False,
            )
        try:
            folder = metadata_folder(str(payload.get("approval_id") or "") or None)
            inspection = inspect_dataset_folder(folder, context=context)
        except (ApprovedFolderError, PackMetadataError) as exc:
            return api_error(409, "metadata_wizard_incomplete", str(exc))
        if inspection["wizard_required"]:
            return api_error(
                409,
                "metadata_wizard_incomplete",
                "Complete every required pack declaration before continuing the command.",
                next_action="Save every pack that still needs information.",
            )
        if not wizard_session.finish(MetadataWizardOutcome.COMPLETE):
            return api_error(
                409,
                "metadata_wizard_already_finished",
                f"This metadata wizard already finished as {wizard_session.outcome.value}.",
                recoverable=False,
            )
        stop_wizard_server(request)
        return JSONResponse(
            {
                "outcome": MetadataWizardOutcome.COMPLETE.value,
                "message": "Pack information is complete. The dataset command will continue.",
            }
        )

    @router.post("/dataset/api/metadata/cancel")
    @product_api
    def cancel_metadata_wizard(request: Request, _payload: dict[str, Any]) -> JSONResponse:
        wizard_session = metadata_wizard_session(request)
        if wizard_session is None:
            return api_error(
                409,
                "metadata_wizard_session_unavailable",
                "No command is waiting for this metadata wizard.",
                recoverable=False,
            )
        if not wizard_session.finish(MetadataWizardOutcome.CANCELLED):
            return api_error(
                409,
                "metadata_wizard_already_finished",
                f"This metadata wizard already finished as {wizard_session.outcome.value}.",
                recoverable=False,
            )
        stop_wizard_server(request)
        return JSONResponse(
            {
                "outcome": MetadataWizardOutcome.CANCELLED.value,
                "message": "Dataset build cancelled. No preprocessing was started.",
            }
        )

    @router.post("/dataset/api/metadata/grouping")
    @product_api
    def confirm_grouping(payload: dict[str, Any]) -> JSONResponse:
        folder: Path | None = None
        try:
            folder = metadata_folder(str(payload.get("approval_id") or "") or None)
            pack, _paths = pack_record(folder, str(payload.get("pack_id") or ""))
            action = str(payload.get("action") or "")
            if action == "keep_proposal":
                additions = [pack.relative_root]
            elif action == "split_children" and pack.proposed_children:
                additions = list(pack.proposed_children)
            else:
                raise PackMetadataError("Choose Keep proposed pack or Split into proposed child packs.")
            merge_grouping_roots(context.project_root, folder, additions)
        except (ApprovedFolderError, DatasetInputError, PackMetadataError) as exc:
            return api_error(422, "pack_grouping_invalid", public_error(exc, folder=folder))
        return JSONResponse(
            {
                "saved": True,
                "input_folder_written": False,
                "inspection": _public_inspection(
                    inspect_dataset_folder(folder, context=context),
                    approval_id=str(payload.get("approval_id") or "") or None,
                    folder_name=folder.name,
                ),
            }
        )

    @router.post("/dataset/api/metadata/export")
    @product_api
    def export_metadata(payload: dict[str, Any]) -> JSONResponse:
        folder: Path | None = None
        try:
            folder = metadata_folder(str(payload.get("approval_id") or "") or None)
            pack, _paths = pack_record(folder, str(payload.get("pack_id") or ""))
            records = load_pack_metadata(context.project_root)
            record = records.get(pack.pack_id)
            if record is None or not sidecar_is_applicable(record, pack, folder):
                raise PackMetadataError("Save current pack information before exporting it.")
            pack_root = folder if pack.relative_root == "." else folder / pack.relative_root
            result = export_metadata_files(record, pack_root)
        except (ApprovedFolderError, DatasetInputError, PackMetadataError, OSError) as exc:
            return api_error(422, "pack_metadata_export_failed", public_error(exc, folder=folder))
        return JSONResponse(result)

    @router.post("/dataset/api/build")
    @product_api
    def build_folder(payload: dict[str, Any]) -> JSONResponse:
        if any(key in payload for key in ("folder", "path", "directory")):
            return api_error(
                422,
                "browser_path_not_allowed",
                "Raw filesystem paths are not accepted by the dataset web API.",
                next_action="Choose the folder with the native folder picker.",
            )
        try:
            folder = store.resolve(str(payload.get("approval_id") or ""))
        except ApprovedFolderError as exc:
            return api_error(422, "folder_approval_invalid", public_error(exc), next_action="Choose the folder again.")
        approval = store.record(str(payload.get("approval_id") or ""))
        if approval is None:
            return api_error(422, "folder_approval_invalid", "The folder approval is no longer available.")
        confirmed = payload.get("confirm_hosted") is True
        try:
            output = _product_output_root(context, folder)
            ensure_dataset_writes_outside_input(
                context.project_root,
                folder,
                output_root=output,
                runs_directory=context.runs_directory,
            )
        except (DatasetInputError, PackMetadataError) as exc:
            return api_error(
                422,
                "dataset_write_boundary_overlap",
                public_error(exc, folder=folder, output=output),
                recoverable=True,
                next_action="Choose a project and output location outside the selected source folder.",
            )
        job_id = secrets.token_urlsafe(18)
        run_id = f"dataset-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        started = datetime.now(timezone.utc).isoformat()
        events.create_run(
            run_id,
            feature="dataset",
            command="dataset.build",
            status=ProductStatus.RUNNING.value,
            stage="intake",
            started_at=started,
            resumable=True,
            extra={"approval_source": approval.source},
        )
        events.append(
            ProductEvent(
                run_id,
                started,
                "dataset",
                "intake",
                "dataset_started",
                ProductStatus.RUNNING,
                message="Dataset intake started from an explicitly approved read-only folder.",
            )
        )
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "Dataset build queued in the background.",
            "logs": [_job_log("Queued dataset build.")],
            "result": None,
            "run": {"run_id": run_id, "feature": "dataset", "status": ProductStatus.RUNNING.value},
        }
        with build_jobs_lock:
            build_jobs[job_id] = job

        def run_build() -> None:
            update_build_job(
                job,
                status="running",
                message="Processing the approved image folder.",
                log="Started dataset build.",
            )
            try:
                current = effective_context()
                provider = provider_factory(current, lambda _prompt: confirmed) if provider_factory else None
                result = DatasetIntakeService(provider).build(
                    folder,
                    output_root=output,
                    context=current,
                )
                counts = result.data.get("counts", {}) if isinstance(result.data, Mapping) else {}
                events.append(
                    ProductEvent(
                        run_id,
                        datetime.now(timezone.utc).isoformat(),
                        "dataset",
                        "review",
                        "dataset_completed",
                        result.status,
                        current=int(counts.get("processed", 0)),
                        total=int(counts.get("processed", 0)),
                        message=result.message,
                        metrics={
                            "accepted": int(counts.get("accepted", 0)),
                            "excluded": int(counts.get("excluded", counts.get("rejected", 0))),
                        },
                    )
                )
                body = result.to_dict()
                body["run"] = {"run_id": run_id, "feature": "dataset", "status": result.status.value}
                public_result = _public_dataset_result(
                    body,
                    folder=folder,
                    output=output,
                    project_root=context.project_root,
                    approval_id=approval.approval_id,
                    folder_name=folder.name,
                    dataset_id=run_id,
                )
                update_build_job(
                    job,
                    status="complete",
                    message=str(public_result.get("message") or "Dataset build finished."),
                    log="Dataset build finished.",
                    result=public_result,
                )
            except Exception as exc:  # Keep worker failures visible without killing the web process.
                events.append(
                    ProductEvent(
                        run_id,
                        datetime.now(timezone.utc).isoformat(),
                        "dataset",
                        "intake",
                        "dataset_failed",
                        ProductStatus.FAILED,
                        message="Dataset intake failed safely. The selected source folder was not modified.",
                    )
                )
                safe_error = public_error(exc, folder=folder, output=output)
                update_build_job(
                    job,
                    status="failed",
                    message="Dataset build failed. Your source files were not changed.",
                    log=f"Build failed: {type(exc).__name__}: {safe_error}",
                )

        Thread(target=run_build, name=f"dataset-build-{job_id[:8]}", daemon=True).start()
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/dataset/api/build/{job_id}",
                "run": job["run"],
            },
        )

    @router.get("/dataset/api/build/{job_id}")
    @product_api
    def build_status(job_id: str) -> JSONResponse:
        with build_jobs_lock:
            job = build_jobs.get(job_id)
            snapshot = dict(job) if job else None
            if snapshot is not None:
                snapshot["logs"] = list(job["logs"])
        if snapshot is None:
            return api_error(404, "dataset_build_not_found", "Dataset build not found.", recoverable=False)
        return JSONResponse(snapshot)

    @router.get("/review", response_class=HTMLResponse)
    def shared_review_page(request: Request, queue: str | None = None) -> Any:
        summary = discover_review_queues(context)
        response = render(request, "review_entry.html", {"review_summary": summary, "selected_queue": queue})
        return response or HTMLResponse(_review_entry_page(summary, selected=queue))

    @router.get("/dataset/review", response_class=HTMLResponse)
    def review_page(request: Request) -> Any:
        output = find_dataset_output(context)
        if output is None:
            response = render(request, "review_empty.html")
            return response or HTMLResponse(_empty_page())
        queue = _public_review_queue(_full_review_queue(output, context=context), project_root=context.project_root)
        response = render(request, "review.html", {"review_queue": queue})
        return response or HTMLResponse(_review_page(queue))

    @router.get("/dataset/review/data")
    @router.get("/dataset/api/review/data")
    @product_api
    def review_data() -> dict[str, Any]:
        output = find_dataset_output(context)
        if output is None:
            raise HTTPException(status_code=404, detail="No dataset review queue is available.")
        return _public_review_queue(_full_review_queue(output, context=context), project_root=context.project_root)

    @router.get("/dataset/review/thumb/{item_id}")
    @router.get("/dataset/api/review/thumb/{item_id}")
    @product_api
    def thumbnail(item_id: str) -> Any:
        _output, queue, item = review_item_record(item_id)
        root = Path(str(queue.get("input_root", ""))).resolve()
        extraction = item.get("sheet_extraction")
        relative = Path(
            str(extraction.get("source_relative_path"))
            if isinstance(extraction, Mapping)
            else str(item.get("relative_path", ""))
        )
        path = Path(str(item.get("source_path", ""))).resolve()
        expected = (root / relative).resolve()
        try:
            expected.relative_to(root)
        except ValueError as exc:
            raise HTTPException(
                status_code=404, detail="Review image reference is outside the approved dataset."
            ) from exc
        if relative.is_absolute() or ".." in relative.parts or path != expected:
            raise HTTPException(status_code=404, detail="Review image reference is invalid.")
        if not path.is_file() or path.suffix.casefold() != ".png":
            raise HTTPException(status_code=404, detail="Source image is no longer available.")
        expected_hash = (
            extraction.get("source_byte_sha256") if isinstance(extraction, Mapping) else item.get("byte_sha256")
        )
        if file_sha256(path) != expected_hash:
            raise HTTPException(
                status_code=409, detail="Source image changed after preprocessing; rebuild the dataset."
            )
        if isinstance(extraction, Mapping):
            with Image.open(path) as opened:
                opened.load()
                preview = opened.convert("RGBA").crop(tuple(int(value) for value in extraction["crop_rectangle"]))
            return StreamingResponse(io.BytesIO(_png_bytes(preview)), media_type="image/png")
        return FileResponse(path, media_type="image/png")

    @router.get("/dataset/review/sheets/{item_id}/preview/{kind}")
    @router.get("/dataset/api/review/sheets/{item_id}/preview/{kind}")
    @product_api
    def sheet_preview(item_id: str, kind: str) -> StreamingResponse:
        _output, queue, item = review_item_record(item_id)
        root = Path(str(queue.get("input_root", ""))).resolve()
        relative = Path(str(item.get("relative_path", "")))
        path = Path(str(item.get("source_path", ""))).resolve()
        if relative.is_absolute() or ".." in relative.parts or path != (root / relative).resolve():
            raise HTTPException(status_code=404, detail="Review sheet reference is invalid.")
        if not path.is_file() or file_sha256(path) != item.get("byte_sha256"):
            raise HTTPException(status_code=409, detail="Source sheet changed; rebuild before review.")
        plan = item.get("sheet_plan")
        if not isinstance(plan, Mapping):
            raise HTTPException(status_code=404, detail="No sheet proposal is available.")
        crops = plan.get("proposed_crops")
        if not isinstance(crops, list):
            crops = []
        with Image.open(path) as opened:
            opened.load()
            image = opened.convert("RGBA")
        if kind == "proposal":
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for index, crop in enumerate(crops):
                rectangle = tuple(int(value) for value in crop)
                draw.rectangle(rectangle, outline=(255, 80, 50, 255), width=1)
                draw.text((rectangle[0] + 1, rectangle[1] + 1), str(index + 1), fill=(255, 255, 255, 255))
            preview = overlay
        else:
            try:
                index = int(kind)
                crop = crops[index]
            except (ValueError, IndexError, TypeError) as exc:
                raise HTTPException(status_code=404, detail="Unknown proposed crop.") from exc
            preview = image.crop(tuple(int(value) for value in crop))
        return StreamingResponse(io.BytesIO(_png_bytes(preview)), media_type="image/png")

    @router.post("/dataset/review/sheets/{item_id}/decision")
    @router.post("/dataset/api/review/sheets/{item_id}/decision")
    @product_api
    def decide_sheet(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        output, queue, item = review_item_record(item_id)
        action = str(payload.get("action") or "")
        if action not in {"keep_proposal", "adjust_grid", "exclude_sheet"}:
            raise HTTPException(status_code=422, detail="Unknown sheet review action.")
        plan = item.get("sheet_plan")
        if action == "keep_proposal" and (not isinstance(plan, Mapping) or len(plan.get("proposed_crops") or ()) < 2):
            raise HTTPException(status_code=409, detail="This sheet has no usable crop proposal to keep.")
        decision: dict[str, Any] = {"action": action}
        if action == "adjust_grid":
            grid = payload.get("grid") if isinstance(payload.get("grid"), Mapping) else {}
            try:
                columns, rows = int(grid.get("columns", 0)), int(grid.get("rows", 0))
                with Image.open(Path(str(item["source_path"]))) as opened:
                    opened.load()
                    rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
                uniform_grid_plan(rgba, columns=columns, rows=rows)
            except (OSError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=public_error(
                        exc,
                        folder=Path(str(queue["input_root"])),
                        output=output,
                    ),
                ) from exc
            decision["grid"] = {"columns": columns, "rows": rows}
        save_sheet_decision(output, item, decision)
        result = DatasetIntakeService().build(
            Path(str(queue["input_root"])),
            output_root=output,
            context=effective_context(),
        )
        return {
            "saved": True,
            "action": action,
            "counts": dict(result.data.get("counts", {})),
            "input_folder_written": False,
        }

    @router.post("/dataset/review/items/{item_id}/decision")
    @router.post("/dataset/api/review/items/{item_id}/decision")
    @product_api
    def decide(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        output = find_dataset_output(context)
        if output is None:
            raise HTTPException(status_code=404, detail="No dataset review queue is available.")
        try:
            result = DatasetReviewStore(output, context=context).apply(item_id, str(payload.get("decision", "")))
        except ReviewDecisionError as exc:
            raise HTTPException(status_code=409, detail=public_error(exc, output=output)) from exc
        result.pop("review_log", None)
        result["paths_exposed"] = False
        return result

    @router.post("/dataset/review/confirm-exclusions")
    @router.post("/dataset/api/review/confirm-exclusions")
    @product_api
    def confirm_exclusions(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        output = find_dataset_output(context)
        if output is None:
            raise HTTPException(status_code=404, detail="No dataset review queue is available.")
        reason = str(payload.get("reason")) if payload and payload.get("reason") else None
        result = DatasetReviewStore(output, context=context).confirm_all_current_exclusions(reason=reason)
        result.pop("review_log", None)
        result["paths_exposed"] = False
        return result

    @router.get("/review/memorization/data")
    @product_api
    def memorization_review_data() -> dict[str, Any]:
        return _public_memorization_review(discover_memorization_review(context), project_root=context.project_root)

    @router.get("/review/memorization/{pair_id}/image/{kind}")
    @product_api
    def memorization_image(pair_id: str, kind: str) -> FileResponse:
        review = discover_memorization_review(context)
        item = next((row for row in review.get("items", ()) if row.get("pair_id") == pair_id), None)
        field = {"generated": "generated_image", "training": "training_comparison_image"}.get(kind)
        if item is None or field is None:
            raise HTTPException(status_code=404, detail="Unknown memorization review image.")
        path = Path(str(item.get(field) or ""))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Memorization review image is unavailable.")
        return FileResponse(path, media_type="image/png")

    @router.post("/review/memorization/{pair_id}/decision")
    @product_api
    def decide_memorization(pair_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        review = discover_memorization_review(context)
        item = next((row for row in review.get("items", ()) if row.get("pair_id") == pair_id), None)
        if item is None or item.get("review_action_available") is not True:
            reason = item.get("action_unavailable_reason") if item else review.get("review_message")
            raise HTTPException(status_code=409, detail=str(reason or "Review action is unavailable."))
        reviewer_id = str(payload.get("reviewer_id") or "").strip()
        if not reviewer_id:
            raise HTTPException(status_code=422, detail="reviewer_id is required.")
        candidate = Path(str(item["candidate_bundle_path"]))
        review_log = _memorization_review_log(context, candidate)
        try:
            event = append_bound_review_event(
                candidate,
                review_log,
                pair_id=pair_id,
                review_outcome=str(payload.get("review_outcome") or ""),
                reviewer_id=reviewer_id,
                human_note=str(payload.get("human_note") or ""),
                expected_context=_memorization_expected_context(context),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=public_error(exc, output=candidate.parent)) from exc
        return {
            "schema_version": "spritelab.product.memorization-review-write.v2",
            "pair_id": pair_id,
            "event_sha256": event["event_sha256"],
            "review_event_schema": event["schema_version"],
            "legacy_rows_written": 0,
        }

    return router


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _read_labeling_items(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (output / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("The current dataset item manifest is invalid.")
        rows.append(dict(value))
    return rows


def _verified_dataset_image(output: Path, item: Mapping[str, Any]) -> Path:
    queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
    root = Path(str(queue.get("input_root", ""))).resolve()
    relative = Path(str(item.get("relative_path", "")))
    path = Path(str(item.get("source_path", ""))).resolve()
    expected = (root / relative).resolve()
    try:
        expected.relative_to(root)
    except ValueError as exc:
        raise ValueError("The semantic review image is outside the approved dataset.") from exc
    if relative.is_absolute() or ".." in relative.parts or path != expected:
        raise ValueError("The semantic review image reference is invalid.")
    if not path.is_file() or path.suffix.casefold() != ".png":
        raise ValueError("The semantic review image is no longer available.")
    if file_sha256(path) != item.get("byte_sha256"):
        raise ValueError("The semantic review image changed after preprocessing.")
    return path


def _labeling_artifact(output: Path, item_id: str) -> Mapping[str, Any] | None:
    artifacts = output / "hierarchical_labeling" / "artifacts"
    if not artifacts.is_dir():
        return None
    for path in artifacts.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and value.get("record_identity") == item_id:
            return value
    return None


def _public_labeling_item(output: Path, item: Mapping[str, Any], graph: Any) -> dict[str, Any]:
    item_id = str(item.get("item_id", ""))
    semantic = item.get("semantic") if isinstance(item.get("semantic"), Mapping) else {}
    labels = semantic.get("labels") if isinstance(semantic.get("labels"), Mapping) else {}
    candidate_values = [labels.get(name) for name in ("canonical_object", "category", "domain")]
    resolved = next(
        (graph.resolve(str(value)) for value in candidate_values if value and graph.resolve(str(value))), None
    )
    artifact = _labeling_artifact(output, item_id)
    views = artifact.get("render_bundle", {}).get("views", ()) if artifact else ()
    return {
        "record_identity": item_id,
        "image_url": f"/labeling/api/thumb/{item_id}",
        "render_views": [
            {
                "render_type": str(view.get("render_type")),
                "url": f"/labeling/api/view/{item_id}/{view.get('render_type')}",
            }
            for view in views
            if isinstance(view, Mapping) and view.get("render_type")
        ],
        "suggested_path": list(graph.path(resolved)) if resolved else [],
        "top_k_alternatives": [],
        "retrieved_verified_neighbors": [],
        "visual_description": labels.get("description"),
        "metadata_evidence": {"filename": Path(str(item.get("relative_path", ""))).name},
        "metadata_is_separate": True,
        "conflicts": list(semantic.get("conflicts", ())),
        "confidence": semantic.get("confidence"),
        "triage_state": semantic.get("triage_state"),
        "keep_requires_human_truth": semantic.get("keep_requires_human_truth") is True,
        "abstention_available": True,
        "exclude_semantic_supervision_available": True,
    }


def _semantic_triage_state(item: Mapping[str, Any]) -> str:
    semantic = item.get("semantic") if isinstance(item.get("semantic"), Mapping) else {}
    explicit = str(semantic.get("triage_state") or "")
    if explicit:
        return explicit
    state = str(semantic.get("state") or "pending")
    if state == "proposed":
        return "low_certainty_rejected" if semantic.get("needs_review") is True else "auto_prefilled"
    if state == "abstained":
        return "low_certainty_rejected"
    return "provider_unavailable"


def _labeling_page(status: Mapping[str, Any]) -> str:
    cards = "".join(
        f"<li><strong>{value['title']}</strong>: {value['status']}</li>" for value in status.get("cards", ())
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Labeling · Sprite Lab</title></head>"
        f"<body><main><h1>Hierarchical labeling</h1><ul>{cards}</ul>"
        "<p>Choose the deepest visually defensible node; abstention is always available.</p></main></body></html>"
    )


def find_dataset_output(context: ProjectContext) -> Path | None:
    try:
        dataset_config, _version, _saved = ProductSettingsRepository(context).effective_settings("dataset")
    except ProductSettingsError:
        dataset_config = context.config.get("dataset", {}) if isinstance(context.config, Mapping) else {}
    if isinstance(dataset_config, Mapping):
        configured = (
            dataset_config.get("selected_root")
            or dataset_config.get("output_root")
            or dataset_config.get("result_path")
        )
        if configured:
            path = Path(str(configured)).expanduser()
            if path.name == "result.json":
                path = path.parent
            try:
                return validate_managed_dataset_output(
                    path,
                    context=context,
                    require_datasets_root=bool(dataset_config.get("selected_root")),
                )
            except ManagedDatasetError:
                pass
    datasets = context.project_root / "datasets"
    if not datasets.is_dir():
        return None
    candidates: list[Path] = []
    for result_path in datasets.glob("*/result.json"):
        try:
            candidates.append(
                validate_managed_dataset_output(result_path.parent, context=context, require_datasets_root=True)
            )
        except ManagedDatasetError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "result.json").stat().st_mtime_ns).resolve()


def _product_output_root(context: ProjectContext, folder: Path) -> Path:
    dataset_config = context.config.get("dataset", {}) if isinstance(context.config, Mapping) else {}
    datasets_root = context.project_root / "datasets"
    if isinstance(dataset_config, Mapping):
        configured = dataset_config.get("output_root") or dataset_config.get("result_path")
        if configured:
            candidate = Path(str(configured)).expanduser()
            if candidate.name == "result.json":
                candidate = candidate.parent
            try:
                return require_confined_path(candidate, datasets_root)
            except UnsafeFilesystemOperation as exc:
                raise DatasetInputError(
                    "The dataset output must be a dedicated directory below this project's datasets directory."
                ) from exc
    name = re.sub(r"[^a-z0-9]+", "-", folder.name.casefold()).strip("-") or "images"
    return require_confined_path(datasets_root / f"{name}-dataset", datasets_root)


def discover_review_queues(context: ProjectContext) -> dict[str, Any]:
    """Discover product review work without translating authoritative rows."""

    dataset_items: list[Mapping[str, Any]] = []
    output = find_dataset_output(context)
    if output is not None:
        try:
            queue = DatasetReviewStore(output, context=context).queue()
            dataset_items = [item for item in queue.get("items", ()) if isinstance(item, Mapping)]
        except (OSError, ValueError, json.JSONDecodeError):
            dataset_items = []
    intake = [item for item in dataset_items if item.get("queue_kind") == "intake_exception"]
    semantic = [item for item in dataset_items if item.get("queue_kind") == "semantic_exception"]
    near_duplicates = [item for item in dataset_items if item.get("queue_kind") == "near_duplicate_exception"]
    extraction = [
        item
        for item in intake
        if item.get("current_disposition") == "requires_special_extraction"
        or "special_extraction" in item.get("reasons", ())
    ]
    memorization_review = discover_memorization_review(context)
    memorization = _memorization_candidates(context, discovered=memorization_review)
    result = {
        "schema_version": "spritelab.product.review-routing.v1",
        "available": bool(
            dataset_items or memorization or memorization_review.get("evidence_state") not in {None, "incomplete"}
        ),
        "queues": [
            {
                "queue_id": "dataset",
                "title": "Rescue images",
                "count": len(intake),
                "route": "/dataset/review",
                "authoritative_format": "spritelab.dataset.review_queue.v1",
            },
            {
                "queue_id": "extraction",
                "title": "Extraction exceptions",
                "count": len(extraction),
                "route": "/dataset/review?reason=special_extraction",
                "authoritative_format": "spritelab.dataset.review_queue.v1",
            },
            {
                "queue_id": "near-duplicates",
                "title": "Possible near duplicates",
                "count": len(near_duplicates),
                "route": "/dataset/review?reason=possible_near_duplicate",
                "authoritative_format": "spritelab.dataset.review_queue.v1",
            },
            {
                "queue_id": "semantic",
                "title": "Description exceptions",
                "count": len(semantic),
                "route": "/dataset/review?reason=semantic",
                "authoritative_format": "spritelab.dataset.review_queue.v1",
            },
            {
                "queue_id": "memorization",
                "title": "Memorization candidates",
                "count": len(memorization),
                "route": "/review?queue=memorization",
                "authoritative_format": "sprite_lab_memorization_candidate_evidence_v2",
            },
        ],
        "memorization_candidates": memorization,
        "memorization_state": memorization_review,
        "formats_preserved": True,
    }
    return result


def _config_path(context: ProjectContext, section: str, key: str) -> Path | None:
    values = context.config.get(section) if isinstance(context.config, Mapping) else None
    raw = values.get(key) if isinstance(values, Mapping) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()


def _memorization_review_log(context: ProjectContext, candidate: Path) -> Path:
    return _config_path(context, "evaluation", "review_log") or candidate.with_name("review_events.jsonl")


def _memorization_expected_context(context: ProjectContext) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for key, field in (
        ("checkpoint", "checkpoint_path"),
        ("benchmark", "benchmark_manifest_path"),
    ):
        value = _config_path(context, "evaluation", key)
        if value is not None:
            expected[field] = value
    dataset_identity = expected_dataset_identity(context.config)
    view_identity = expected_training_view_identity(context.config)
    if dataset_identity is None or view_identity is None:
        raise ValueError("Active training dataset and view identities must both be configured before review.")
    expected["training_dataset_identity"] = dataset_identity
    expected["training_view_identity"] = view_identity
    return expected


def discover_memorization_review(context: ProjectContext) -> dict[str, Any]:
    """Discover the one current bundle, then delegate all parsing to strict-v2."""
    configured = _config_path(context, "evaluation", "candidate_evidence")
    roots = [context.runs_directory, context.project_root / "runs", context.project_root / "evaluation"]
    paths: list[Path] = [] if configured is None else [configured]
    if configured is None:
        for root in roots:
            if root is not None and root.is_dir():
                paths.extend(root.rglob("candidate_evidence.json"))
    existing = [path.resolve() for path in set(paths) if path.is_file()]
    candidate = configured or (max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None)
    try:
        expected_context = _memorization_expected_context(context)
    except ValueError as error:
        message = str(error)
        display = memorization_display(None)
        display.update(
            {
                "review_message": message,
                "action_unavailable_reason": message,
                "validation_reasons": [message],
            }
        )
        return display
    return memorization_display(
        candidate,
        review_log=_memorization_review_log(context, candidate) if candidate is not None else None,
        expected_context=expected_context,
    )


def _memorization_candidates(
    context: ProjectContext,
    *,
    discovered: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    review = dict(discovered or discover_memorization_review(context))
    candidates: list[dict[str, Any]] = []
    for row in review.get("items", ()):
        if not isinstance(row, Mapping):
            continue
        try:
            evidence_class = parse_evidence_class(row.get("evidence_class"))
        except ValueError:
            continue
        if evidence_class not in HARD_EVIDENCE_CLASSES | REVIEW_REQUIRED_EVIDENCE_CLASSES:
            continue
        candidates.append({**dict(row), "hard_evidence": evidence_class in HARD_EVIDENCE_CLASSES})
    return candidates


def _dataset_page() -> str:
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dataset · Sprite Lab</title>
<style>body{font:16px/1.5 system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}ol{display:grid;gap:.6rem}
form{display:grid;gap:.8rem;padding:1rem;border:1px solid #9996;border-radius:12px}input{padding:.7rem}button{padding:.75rem 1rem}
pre{white-space:pre-wrap}.primary{font-weight:700}</style></head><body><a href="/">Home</a><h1>Build a dataset</h1>
<ol><li>Choose image folder.</li><li>Check source and license.</li><li>Build dataset.</li><li>Train.</li><li>Evaluate and try the model.</li></ol>
<form id="dataset-form"><label>Image folder <input id="dataset-folder" name="folder" required autocomplete="off"></label>
<div><button class="primary" id="inspect-folder" type="button">Choose image folder</button>
<button id="build-dataset" type="button" disabled>Build dataset</button></div>
<label><input id="confirm-hosted" type="checkbox"> Allow the configured hosted description provider for this build</label></form>
<pre id="dataset-result" role="status" aria-live="polite">Next action: Choose image folder</pre>
<p><a href="/review">Open Review</a></p><script src="/plugins/dataset.intake/static/dataset.js" defer></script></body></html>"""


def _metadata_page(
    inspection: Mapping[str, Any],
    *,
    approval_id: str | None,
    wizard_session_active: bool = False,
) -> str:
    import html

    cards = []
    for pack in inspection.get("packs", ()):
        prefill = pack.get("prefill", {})
        license_value = str(prefill.get("license_identifier") or "")
        license_options = "".join(
            f'<option value="{value}"{" selected" if license_value == value else ""}>{title}</option>'
            for value, title in (
                ("", "Choose…"),
                ("cc0", "CC0"),
                ("public_domain", "Public domain"),
                ("cc_by", "CC BY"),
                ("cc_by_sa", "CC BY-SA"),
                ("mit", "MIT"),
                ("apache_2", "Apache-2.0"),
                ("bsd", "BSD"),
                ("wtfpl", "WTFPL"),
                ("custom", "Custom"),
                ("private_permission", "Private permission"),
                ("unknown", "Unknown"),
            )
        )
        cards.append(
            f'<section class="metadata-pack" data-pack-id="{html.escape(str(pack.get("pack_id")))}">'
            f"<h2>{html.escape(str(pack.get('relative_root')))}</h2>"
            f"<p>{int(pack.get('image_count', 0))} image(s); missing: "
            f"{html.escape(', '.join(str(value) for value in pack.get('missing_fields', ())) or 'none')}</p>"
            '<form class="metadata-form">'
            f'<label>Creator or rights holder <input name="creator_or_rights_holder" value="{html.escape(str(prefill.get("creator_or_rights_holder", "")))}"></label>'
            f'<label>Pack title <input name="pack_title" value="{html.escape(str(prefill.get("pack_title", prefill.get("folder_name", ""))))}"></label>'
            '<label>Source type <select name="source_type"><option value="opengameart">OpenGameArt</option><option value="kenney">Kenney</option><option value="other_downloaded">Other downloaded source</option><option value="my_original_work">My original work</option><option value="custom_private">Custom/private agreement</option></select></label>'
            f'<label>Source page URL <input name="source_page_url" value="{html.escape(str(prefill.get("source_page_url", "")))}"></label>'
            f'<label>License <select name="license_identifier" required>{license_options}</select></label>'
            f'<label>License URL <input name="license_url" value="{html.escape(str(prefill.get("license_url", "")))}"></label>'
            f'<label>License evidence file <input name="license_evidence_file" value="{html.escape(str(prefill.get("license_evidence_file", "")))}"></label>'
            '<label>Attribution <textarea name="attribution_text"></textarea></label>'
            '<label><input type="checkbox" name="original_work_declaration"> I declare this is my original work.</label>'
            '<label><input type="checkbox" name="permission_confirmed"> I confirm private/custom permission.</label>'
            '<button type="submit">Save pack information</button></form></section>'
        )
    approval = html.escape(approval_id or "")
    dataset_href = f"/dataset?approval_id={approval}" if approval else "/dataset"
    wizard_controls = (
        '<button id="metadata-complete" type="button"'
        + (" disabled" if inspection.get("wizard_required") else "")
        + '>Complete and continue</button><button id="metadata-cancel" type="button">Cancel</button>'
        if wizard_session_active
        else f'<a id="metadata-return" href="{dataset_href}">Return to dataset</a>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Pack information · Sprite Lab</title></head>
<body><h1>Complete source and license information</h1><p>Sprite Lab records your declaration but cannot verify ownership or give legal advice.</p>
<div id="metadata-status" role="status">{int(inspection.get("image_count", 0))} PNG file(s)</div>
<main class="metadata-pack-list" data-approval-id="{approval}">{"".join(cards) or html.escape(str(inspection.get("error") or "No packs found."))}</main>
<p id="metadata-next">Unknown licenses remain quarantined.</p>{wizard_controls}
<script src="/plugins/dataset.intake/static/metadata.js" defer></script></body></html>"""


def _review_entry_page(summary: Mapping[str, Any], *, selected: str | None) -> str:
    import html

    cards = [
        f'<li><a href="{html.escape(str(queue["route"]))}"><strong>{html.escape(str(queue["title"]))}</strong></a> '
        f"— {int(queue['count'])} item(s)</li>"
        for queue in summary.get("queues", ())
    ]
    memo = ""
    if selected == "memorization":
        rows: list[str] = []
        for item in summary.get("memorization_candidates", ()):
            pair_id = html.escape(str(item.get("pair_id")))
            diagnostics = html.escape(json.dumps(item.get("diagnostics"), sort_keys=True, default=str))
            action = (
                "Controlled signed-v2 outcomes are available."
                if item.get("review_action_available")
                else html.escape(str(item.get("action_unavailable_reason") or "Review action is unavailable."))
            )
            controls = "".join(
                f'<button type="button" data-pair-id="{pair_id}" '
                f'data-review-outcome="{html.escape(str(outcome))}">{html.escape(str(outcome))}</button>'
                for outcome in item.get("controlled_review_outcomes", ())
            )
            rows.append(
                f"<li><article><h3><code>{pair_id}</code></h3>"
                f'<p><img src="/review/memorization/{pair_id}/image/generated" alt="Generated image"> '
                f'<img src="/review/memorization/{pair_id}/image/training" alt="Training comparison image"></p>'
                f"<p>Evidence: {html.escape(str(item.get('evidence_class')))}<br>"
                f"State: {html.escape(str(item.get('current_review_state')))}<br>"
                f"Event chain: {html.escape(str(item.get('event_chain_status')))} "
                f"(valid={str(bool(item.get('event_chain_valid'))).casefold()})</p>"
                f"<details><summary>Diagnostics</summary><pre>{diagnostics}</pre></details>"
                f"<p>{action}</p><div>{controls}</div></article></li>"
            )
        state = summary.get("memorization_state", {})
        legacy_rows = [
            f"<li><code>{html.escape(str(row.get('pair_id') or 'unknown'))}</code> — "
            "legacy, read-only, non-authoritative</li>"
            for row in state.get("legacy_reviews", ())
            if isinstance(row, Mapping)
        ]
        message = html.escape(str(state.get("review_message") or "No candidates found."))
        memo = (
            f"<section><h2>Memorization candidates</h2><p>{message}</p>"
            f"<ul>{''.join(rows) or '<li>No actionable candidates found.</li>'}</ul>"
            f"<h3>Legacy reviews</h3><ul>{''.join(legacy_rows) or '<li>No legacy reviews.</li>'}</ul></section>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review · Sprite Lab</title><style>body{{font:16px/1.5 system-ui;max-width:850px;margin:2rem auto;padding:0 1rem}}li{{margin:.8rem 0}}article{{border:1px solid #9996;border-radius:10px;padding:1rem}}article img{{width:160px;height:160px;object-fit:contain;image-rendering:pixelated;background:#222}}pre{{white-space:pre-wrap}}</style></head>
<body><a href="/">Home</a><h1>Review</h1><p>Use one entry point; each review type keeps its authoritative append-only format.</p>
<ul>{"".join(cards)}</ul>{memo}</body></html>"""


def _legacy_review_entry_page(summary: Mapping[str, Any], *, selected: str | None) -> str:
    import html

    cards = []
    for queue in summary.get("queues", ()):
        cards.append(
            f'<li><a href="{html.escape(str(queue["route"]))}"><strong>{html.escape(str(queue["title"]))}</strong></a> '
            f"— {int(queue['count'])} item(s)</li>"
        )
    memo = ""
    if selected == "memorization":
        rows = []
        for item in summary.get("memorization_candidates", ()):
            action = (
                "Signed-v2 review action available."
                if item.get("review_action_available")
                else str(item.get("action_unavailable_reason") or "No review action is available.")
            )
            pair_id = html.escape(str(item.get("pair_id")))
            diagnostics = html.escape(json.dumps(item.get("diagnostics"), indent=2, sort_keys=True))
            rows.append(
                f"<li><code>{pair_id}</code> — {html.escape(str(item.get('evidence_class')))} — "
                f"state={html.escape(str(item.get('current_review_state')))} — "
                f"chain={html.escape(str(item.get('event_chain_status')))} — {html.escape(action)}"
                f'<div><img width="160" height="160" alt="Generated image" '
                f'src="/review/memorization/{pair_id}/image/generated"> '
                f'<img width="160" height="160" alt="Training comparison image" '
                f'src="/review/memorization/{pair_id}/image/training"></div>'
                f"<pre>{diagnostics}</pre></li>"
            )
        state = summary.get("memorization_state", {})
        legacy = "".join(
            f"<li>{html.escape(str(item.get('pair_id', 'unknown')))} — "
            "Legacy review, read-only and non-authoritative.</li>"
            for item in summary.get("legacy_memorization_reviews", ())
        )
        memo = (
            "<section><h2>Memorization candidates</h2>"
            f"<p>{html.escape(str(state.get('message', '')))}</p>"
            f"<ul>{''.join(rows) or '<li>No actionable candidates found.</li>'}</ul>"
            f"<h3>Legacy reviews</h3><ul>{legacy or '<li>No legacy reviews.</li>'}</ul></section>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review · Sprite Lab</title><style>body{{font:16px/1.5 system-ui;max-width:850px;margin:2rem auto;padding:0 1rem}}li{{margin:.8rem 0}}</style></head>
<body><a href="/">Home</a><h1>Review</h1><p>Use one entry point; each review type keeps its own authoritative append-only format.</p>
<ul>{"".join(cards)}</ul>{memo}</body></html>"""


def _review_page(queue: Mapping[str, Any]) -> str:
    import html
    import json

    items = list(queue.get("items", ()))
    reasons = sorted({str(reason) for item in items for reason in item.get("reasons", ())})
    cards = []
    for index, item in enumerate(items):
        reason_text = ", ".join(str(value) for value in item.get("reasons", ())) or "review required"
        category = "Legal evidence" if "legal" in item.get("reason_categories", ()) else "Technical check"
        current_decision = str(item.get("current_decision", "exclude"))
        hidden = "" if item.get("default_visible") else " hidden"
        keep_class = "keep selected" if current_decision == "keep" else "keep"
        exclude_class = "exclude selected" if current_decision == "exclude" else "exclude"
        cards.append(
            f'''<article class="review-card" data-index="{index}" data-item-id="{html.escape(str(item["item_id"]))}"
                data-reasons="{html.escape(reason_text)}" data-default-visible="{str(bool(item.get("default_visible"))).casefold()}"
                data-current-decision="{html.escape(current_decision)}"{hidden}>
              <img src="{html.escape(str(item["thumbnail_url"]))}" alt="Thumbnail for {html.escape(str(item["relative_path"]))}">
              <div class="details"><p class="category">{category}</p><h2>{html.escape(str(item["relative_path"]))}</h2>
              <p class="reasons">{html.escape(reason_text)}</p>
              <p class="evidence">Source: {html.escape(str(item.get("source", {}).get("path") or "missing"))}<br>
              License: {html.escape(str(item.get("license", {}).get("license") or "missing"))}</p>
              <div class="actions"><button class="{keep_class}" data-decision="keep" aria-label="Keep — Rescue image">Rescue image <kbd>K</kbd></button>
              <button class="{exclude_class}" data-decision="exclude">Exclude <kbd>E</kbd></button></div></div></article>'''
        )
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(f'<option value="{html.escape(reason)}">{html.escape(reason)}</option>' for reason in reasons)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprite Lab — exception review</title><style>
:root{{--bg:#10141b;--panel:#19212c;--text:#eef4ff;--muted:#9db0c7;--keep:#4fd18b;--exclude:#ff7b72}}
body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif}}header{{position:sticky;top:0;background:#0d1118;padding:1rem 2rem;z-index:2}}
.toolbar{{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}}main{{padding:1rem 2rem;display:grid;gap:1rem}}main.contact-sheet{{grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}}
.review-card{{display:flex;gap:1rem;background:var(--panel);padding:1rem;border-radius:12px}}.review-card.active{{outline:3px solid #64a8ff}}img{{width:160px;height:160px;object-fit:contain;image-rendering:pixelated;background:#273241}}
.details{{flex:1}}.category{{color:var(--muted);text-transform:uppercase;font-size:.75rem;letter-spacing:.08em}}.actions{{display:flex;gap:.6rem}}button,select{{font:inherit;padding:.65rem 1rem;border:0;border-radius:8px}}
.keep{{background:var(--keep)}}.exclude{{background:var(--exclude)}}kbd{{background:#0003;padding:.1rem .3rem;border-radius:3px}}.legal{{color:#ffd580}}
</style></head><body><header><h1>Review exceptions</h1><p>Rescuing a false rejection is the main action. Missing legal evidence cannot be overridden.</p>
<div class="toolbar" data-keyboard-shortcuts="ArrowLeft ArrowRight K E"><button id="previous">← Previous</button><button id="next">Next →</button>
<label>Reason <select id="reason-filter"><option value="">All default exceptions</option>{options}</select></label>
<button id="contact-sheet">Contact sheet</button><button id="confirm-exclusions">Confirm all current exclusions</button></div></header>
<main id="queue">{"".join(cards) or "<p>No rejected, uncertain, or special-extraction items need review.</p>"}</main>
<script id="review-data" type="application/json">{payload}</script>
<script src="/plugins/dataset.intake/static/review.js" defer></script></body></html>"""


def _empty_page() -> str:
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Dataset review</title></head>
<body><h1>No dataset review is available</h1><p>Build a dataset first with <code>python -m spritelab v3 dataset build &lt;folder&gt;</code>.</p></body></html>"""

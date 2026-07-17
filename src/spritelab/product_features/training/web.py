"""Shared-shell Training settings, durable dashboard, and action APIs."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from importlib.resources import files
from typing import Any
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from spritelab.product_core import (
    ProductResult,
    ProductSettingsError,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
    api_error,
    product_api,
)
from spritelab.product_features.training.config import ComputeSettings
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.preparation_jobs import (
    PreparationJobRepository,
    PreparationJobStateError,
)
from spritelab.product_features.training.service import TrainingService

PLUGIN_ID = "training"


def create_router(
    context: ProjectContext,
    service: TrainingService | None = None,
    *,
    service_factory: Callable[[], TrainingService] | None = None,
) -> object:
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, JSONResponse

    if service is None and service_factory is None:
        raise ValueError("Training router requires a service or service factory.")
    router = APIRouter()
    repository = ProductSettingsRepository(context)
    cached_service: TrainingService | None = service
    cached_version: int | None = None
    cached_config_stamp: int | None = None
    preparation_jobs = PreparationJobRepository(context)
    preparation_jobs.reconstruct()

    def current_service() -> TrainingService:
        nonlocal cached_config_stamp, cached_service, cached_version
        if service is not None:
            return service
        _raw, version, _saved = repository.effective_settings("compute")
        try:
            config_stamp = context.config_path.stat().st_mtime_ns if context.config_path is not None else None
        except OSError:
            config_stamp = None
        if cached_service is None or cached_version != version or cached_config_stamp != config_stamp:
            assert service_factory is not None
            cached_service = service_factory()
            cached_version = version
            cached_config_stamp = config_stamp
        return cached_service

    @router.get("/training/api/preparation")
    @product_api
    def preparation_state() -> JSONResponse:
        return JSONResponse(_preparation_projection(preparation_jobs.load()))

    @router.get("/training/api/preparation/error-image")
    @product_api
    def preparation_error_image(item_id: str) -> Response:
        state = preparation_jobs.load()
        error = state.get("error")
        if (
            state.get("status") != "failed"
            or not isinstance(error, Mapping)
            or error.get("code") != "canonical_encoding_failed"
            or error.get("item_id") != item_id
        ):
            return api_error(404, "preparation_error_image_unavailable", "No current preparation error image exists.")
        from spritelab.product_features.training.preparation import (
            TrainingPreparationError,
            load_accepted_source_image,
        )

        try:
            content = load_accepted_source_image(context, item_id)
        except TrainingPreparationError as exc:
            return api_error(409, exc.code, exc.public_message, next_action="Rebuild or review the current dataset.")
        return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.post("/training/api/preparation")
    @product_api
    async def start_preparation(request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_training_preparation", "Preparation settings must be a JSON object.")
        if payload.get("authorize_baseline") is not True:
            return api_error(422, "baseline_authorization_required", "Confirm the immutable image-only baseline first.")
        if payload.get("authorize_training") is True or payload.get("authorize_freeze") is True:
            return api_error(
                422,
                "baseline_cannot_authorize_training",
                "Image-only baseline preparation cannot authorize a production freeze or training.",
                recoverable=False,
                next_action="Publish and audit a conditioned Dataset-v5 freeze through the dataset workflow.",
            )
        try:
            from spritelab.product_features.training.preparation import (
                TrainingPreparationError,
                preparation_job_identities,
                prepare_active_dataset,
            )

            identities = preparation_job_identities(context)
            preparation = preparation_jobs.begin(identities)
        except TrainingPreparationError as exc:
            return api_error(409, exc.code, exc.public_message)
        except PreparationJobStateError:
            state = preparation_jobs.load()
            return JSONResponse(_preparation_projection(state), status_code=409)
        job_id = str(preparation["job_id"])
        owner_token = str(preparation["worker_owner"])

        def update(current: int, total: int, message: str) -> None:
            preparation_jobs.progress(job_id, owner_token, current, total, message)

        def run() -> None:
            try:
                result = prepare_active_dataset(
                    context,
                    authorize_baseline=True,
                    progress=update,
                )
            except TrainingPreparationError as exc:
                preparation_jobs.transition(
                    job_id,
                    owner_token,
                    status="failed",
                    error=exc.to_public_dict(),
                    message="Preparation stopped safely; no local paths were exposed.",
                )
            except Exception:
                preparation_jobs.transition(
                    job_id,
                    owner_token,
                    status="failed",
                    error={
                        "code": "training_preparation_failed",
                        "message": "Training preparation failed safely before launch.",
                    },
                    message="Preparation stopped safely; review server logs for details.",
                )
            else:
                preparation_jobs.transition(
                    job_id,
                    owner_token,
                    status="complete",
                    result=result,
                    message="Immutable baseline artifacts are ready. No production or training authorization changed.",
                )

        threading.Thread(target=run, name="spritelab-training-preparation", daemon=True).start()
        return JSONResponse(_preparation_projection(preparation), status_code=202)

    def settings() -> tuple[ComputeSettings, int, bool]:
        raw, version, saved = repository.effective_settings("compute")
        return ComputeSettings.from_mapping(raw, allow_unavailable=True), version, saved

    @router.get("/training", response_class=HTMLResponse)
    def training_page(request: Request) -> Any:
        training_service = current_service()
        run_id = training_service.latest_run_id()
        dashboard = training_service.dashboard(run_id).data if run_id else None
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            configured, version, saved = settings()
            return renderer(
                request,
                PLUGIN_ID,
                "training.html",
                {
                    "compute_settings": configured,
                    "compute_configuration_version": version,
                    "compute_settings_saved": saved,
                    "training_run_id": run_id,
                    "training_dashboard": dashboard,
                },
            )
        template = files("spritelab.product_features.training").joinpath("templates/training.html")
        return HTMLResponse(template.read_text(encoding="utf-8"))

    @router.get("/training/api/state")
    @product_api
    def training_state(profile: str = "recommended") -> JSONResponse:
        try:
            selected_profile = TrainingProfile(profile)
            result = current_service().status(selected_profile)
        except (ValueError, LookupError) as exc:
            return api_error(422, "training_profile_invalid", str(exc))
        payload = result.to_dict()
        contract = _conditioned_contract(context, selected_profile)
        data = dict(payload.get("data") or {})
        data["conditioned_dataset_contract"] = contract
        if not contract.get("ready"):
            data["ready"] = False
            data["availability_state"] = "Training unavailable"
            if payload.get("status") == ProductStatus.READY.value:
                payload["status"] = ProductStatus.BLOCKED.value
                payload["message"] = "Training requires an audited conditioned Dataset-v5 freeze and bound campaign."
            payload["blockers"] = [
                *list(payload.get("blockers") or []),
                *[
                    {
                        "code": str(item.get("code") or "conditioned_dataset_contract"),
                        "message": str(item.get("message") or "The conditioned dataset contract is incomplete."),
                        "resolution": "Complete the conditioned Dataset-v5 freeze, audits, campaign binding, and execution policy.",
                    }
                    for item in contract.get("blockers", [])
                    if isinstance(item, Mapping)
                ],
            ]
        payload["data"] = data
        return JSONResponse(payload)

    @router.get("/training/api/settings")
    @product_api
    def training_settings() -> JSONResponse:
        try:
            configured, version, saved = settings()
        except (ValueError, ProductSettingsError) as exc:
            return api_error(409, "compute_configuration_invalid", str(exc))
        return JSONResponse(
            {
                "options": ["local", "ssh", "runpod", "other"],
                "display_options": [
                    {"id": "local", "title": "Local computer", "available": True},
                    {"id": "ssh", "title": "Remote SSH machine", "available": True},
                    {
                        "id": "runpod",
                        "title": "RunPod",
                        "available": False,
                        "message": "Not available in this build.",
                    },
                    {"id": "other", "title": "Other provider plugin", "available": True},
                ],
                "selected": configured.backend_type,
                "configuration": configured.to_persisted_dict(),
                "configuration_version": version,
                "saved": saved,
                "credentials_persisted": False,
                "connection_test_is_explicit": True,
                "compute_probes": 0,
                "remote_calls": 0,
            }
        )

    @router.post("/training/api/settings")
    @product_api
    async def save_compute_settings(request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_compute_settings", "Compute settings must be a JSON object.")
        try:
            configured = ComputeSettings.from_mapping(payload, allow_unavailable=False)
            saved = repository.save("compute", configured.to_persisted_dict())
        except (ValueError, ProductSettingsError) as exc:
            return api_error(
                422,
                "invalid_compute_settings",
                str(exc),
                next_action="Correct the compute setting and save again.",
            )
        return JSONResponse(
            {
                "status": "saved",
                "configuration_version": saved["configuration_version"],
                "message": "Compute settings were saved. No remote connection was made.",
                "compute_probes": 0,
                "remote_calls": 0,
            }
        )

    @router.delete("/training/api/settings")
    @product_api
    def clear_compute_settings() -> JSONResponse:
        try:
            cleared = repository.clear("compute")
        except ProductSettingsError as exc:
            return api_error(409, "compute_settings_clear_failed", str(exc))
        return JSONResponse(
            {
                "status": "cleared" if cleared else "already_clear",
                "message": "Saved compute settings were cleared.",
                "remote_calls": 0,
            }
        )

    @router.post("/training/api/connection-test")
    @product_api
    def connection_test() -> JSONResponse:
        try:
            capabilities = [
                {
                    "capability_id": item.capability_id,
                    "title": item.title,
                    "status": item.status.value,
                    "message": item.message,
                    "details": dict(item.details),
                }
                for item in current_service().backend.probe(current_service().context)
            ]
        except Exception as exc:
            return api_error(
                503,
                "compute_connection_test_failed",
                f"The explicit compute connection test failed safely ({type(exc).__name__}).",
                next_action="Review the host and credential reference, then test again.",
            )
        return JSONResponse({"capabilities": capabilities, "probe_operations": 1})

    @router.post("/training/api/estimate")
    @product_api
    def estimate_resources() -> JSONResponse:
        try:
            plan = current_service().plan(before_launch=False)
        except (ValueError, LookupError) as exc:
            return api_error(409, "resource_estimate_unavailable", str(exc))
        return JSONResponse(
            {
                "estimate": plan.estimate.to_dict(),
                "backend_id": plan.backend_id,
                "probe_operations": 0,
            }
        )

    @router.post("/training/api/start")
    @product_api
    async def start_training(request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_training_request", "Training request must be a JSON object.")
        configured, _version, _saved = settings()
        if configured.backend_type == "runpod":
            return api_error(
                409,
                "runpod_unavailable",
                "RunPod is not available in this build and cannot launch.",
                recoverable=False,
                next_action="Choose Local computer, Remote SSH machine, or a registered provider plugin.",
            )
        try:
            profile = TrainingProfile(str(payload.get("profile") or configured.run_profile))
        except ValueError:
            return api_error(422, "training_profile_invalid", "The selected training profile is invalid.")
        custom = payload.get("custom") if isinstance(payload.get("custom"), dict) else None
        contract = _conditioned_contract(context, profile, custom_spec=custom)
        if not contract.get("ready"):
            return api_error(
                409,
                "conditioned_dataset_contract_required",
                "Training requires an audited conditioned Dataset-v5 freeze and an exactly bound campaign.",
                recoverable=True,
                next_action="Complete the conditioned dataset, independent audits, campaign binding, and execution authorizations.",
            )
        result = current_service().start(
            profile,
            custom_spec=custom,
            cloud_confirmation=payload.get("confirm_cloud") is True,
        )
        if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
            return api_error(
                409,
                "training_launch_blocked",
                result.message,
                recoverable=True,
                next_action=_next_action(result, "Resolve the displayed safety gate, then start again."),
            )
        return JSONResponse(result.to_dict())

    @router.get("/training/api/runs/{run_id}")
    @product_api
    def dashboard(run_id: str) -> JSONResponse:
        result = current_service().dashboard(run_id)
        if result.status == ProductStatus.UNAVAILABLE:
            return api_error(404, "training_run_not_found", result.message)
        return JSONResponse(result.to_dict())

    @router.post("/training/api/runs/{run_id}/refresh")
    @product_api
    def refresh_dashboard(run_id: str) -> JSONResponse:
        result = current_service().refresh(run_id)
        if result.status == ProductStatus.UNAVAILABLE:
            return api_error(404, "training_run_not_found", result.message)
        return JSONResponse(result.to_dict())

    @router.post("/training/api/runs/{run_id}/pause")
    @product_api
    def pause(run_id: str) -> JSONResponse:
        return _action_response(current_service().pause(run_id), "pause")

    @router.post("/training/api/runs/{run_id}/cancel")
    @product_api
    def cancel(run_id: str) -> JSONResponse:
        return _action_response(current_service().cancel(run_id), "cancel")

    @router.post("/training/api/runs/{run_id}/resume")
    @product_api
    async def resume(run_id: str, request: Request) -> JSONResponse:
        payload = await _json_mapping(request) or {}
        return _action_response(
            current_service().resume(run_id, cloud_confirmation=payload.get("confirm_cloud") is True),
            "resume",
        )

    def run_action(action: str, run_id: str, payload: Mapping[str, Any]) -> ProductResult:
        if action == "pause":
            return current_service().pause(run_id)
        if action == "cancel":
            return current_service().cancel(run_id)
        if action == "resume":
            return current_service().resume(run_id, cloud_confirmation=payload.get("confirm_cloud") is True)
        return ProductResult(ProductStatus.UNAVAILABLE, "Run action is not supported.", feature="training")

    router.spritelab_run_action_handler = run_action
    router.spritelab_run_action_feature = "training"
    return router


async def _json_mapping(request: Any) -> dict[str, Any] | None:
    try:
        value = await request.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _preparation_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: value for key, value in state.items() if key not in {"worker_owner", "worker_pid"}}
    error = projected.get("error")
    if isinstance(error, Mapping) and error.get("code") == "canonical_encoding_failed":
        item_id = error.get("item_id")
        if isinstance(item_id, str) and item_id:
            public_error = dict(error)
            public_error["image_url"] = f"/training/api/preparation/error-image?item_id={quote(item_id, safe='')}"
            public_error["review_url"] = "/dataset/review"
            projected["error"] = public_error
    return projected


def _conditioned_contract(
    context: ProjectContext,
    profile: TrainingProfile = TrainingProfile.RECOMMENDED,
    *,
    custom_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from spritelab.product_features.training.preparation import conditioned_training_contract

        return conditioned_training_contract(context, profile, custom_spec=custom_spec)
    except Exception:
        return {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": False,
            "profile": profile.value,
            "blockers": [
                {
                    "code": "conditioned_dataset_contract",
                    "message": "The conditioned dataset contract could not be verified safely.",
                }
            ],
            "paths_exposed": False,
        }


def _next_action(result: ProductResult, fallback: str) -> str:
    for blocker in result.blockers:
        if blocker.resolution:
            return blocker.resolution
    return fallback


def _action_response(result: ProductResult, action: str) -> Any:
    if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
        return api_error(
            409,
            f"training_{action}_unsupported"
            if result.status == ProductStatus.UNAVAILABLE
            else f"training_{action}_blocked",
            result.message,
            recoverable=result.status != ProductStatus.UNAVAILABLE,
            next_action=_next_action(result, "Review the durable run status and backend capability."),
        )
    return JSONResponse(result.to_dict())


__all__ = ["create_router"]

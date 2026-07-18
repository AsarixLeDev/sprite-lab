"""Shared-shell Training settings, durable dashboard, and action APIs."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from spritelab.product_core import (
    ProductAction,
    ProductResult,
    ProductRun,
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
from spritelab.product_web.events import (
    is_sensitive_public_key,
    normalize_public_field_name,
    sanitize_public_text,
)

PLUGIN_ID = "training"
_PUBLIC_DROP = object()
_PUBLIC_BOOLEAN_FIELDS = frozenset(
    {
        "activated",
        "advanced_collapsed",
        "benchmark_evidence",
        "cancel_available",
        "cancelled",
        "cloud",
        "config_unchanged",
        "configuration_activated",
        "credential_reference_configured",
        "downloaded",
        "eligible",
        "exploratory",
        "hash_verified",
        "immutable",
        "may_accrue_cost",
        "passed",
        "paths_exposed",
        "pause_available",
        "production_authorized",
        "promotion_evidence",
        "ready",
        "remote",
        "remote_identity_verified",
        "remote_resource_uncertain",
        "requires_confirmation",
        "resumable",
        "resource_shutdown_verified",
        "resource_state_uncertain",
        "resume_available",
        "safe",
        "safe_resume",
        "training_authorized",
        "training_started",
        "trustworthy",
        "unsafe_resume_available",
        "validated",
    }
)


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
        return JSONResponse(_preparation_projection(preparation_jobs.load(), context.project_root))

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
            return JSONResponse(_preparation_projection(state, context.project_root), status_code=409)
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
        return JSONResponse(_preparation_projection(preparation, context.project_root), status_code=202)

    def settings() -> tuple[ComputeSettings, int, bool]:
        raw, version, saved = repository.effective_settings("compute")
        return ComputeSettings.from_mapping(raw, allow_unavailable=True), version, saved

    @router.get("/training", response_class=HTMLResponse)
    def training_page(request: Request) -> Any:
        training_service = current_service()
        service_context = _service_context(training_service, context)
        run_id = training_service.latest_run_id()
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            configured, version, saved = settings()
            return renderer(
                request,
                PLUGIN_ID,
                "training.html",
                {
                    "compute_settings": _public_compute_settings(configured, service_context.project_root),
                    "compute_configuration_version": version,
                    "compute_backend_identity": _public_text(
                        _backend_identity(configured), service_context.project_root
                    ),
                    "compute_settings_saved": saved,
                    "training_run_id": _public_text(run_id, service_context.project_root),
                    # Durable events are reconstructed once by the lazy dashboard API.
                    "training_dashboard": None,
                },
            )
        template = files("spritelab.product_features.training").joinpath("templates/training.html")
        return HTMLResponse(template.read_text(encoding="utf-8"))

    @router.get("/training/api/state")
    @product_api
    def training_state(profile: str = "recommended") -> JSONResponse:
        try:
            selected_profile = TrainingProfile(profile)
            training_service = current_service()
            service_context = _service_context(training_service, context)
            result = training_service.status(selected_profile)
        except (ValueError, LookupError) as exc:
            return api_error(422, "training_profile_invalid", str(exc))
        payload = result.to_dict()
        contract = _conditioned_contract(service_context, selected_profile)
        data = dict(payload.get("data") or {})
        data["conditioned_dataset_contract"] = contract
        if contract.get("ready") is not True:
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
        return JSONResponse(_public_state_projection(payload, service_context.project_root))

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
                "backend_identity": _public_text(_backend_identity(configured), context.project_root),
                "configuration": _public_compute_settings(configured, context.project_root),
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
                "backend_identity": _public_text(_backend_identity(configured), context.project_root),
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
            configured, version, _saved = settings()
        except ProductSettingsError as exc:
            return api_error(409, "compute_settings_clear_failed", str(exc))
        return JSONResponse(
            {
                "status": "cleared" if cleared else "already_clear",
                "configuration_version": version,
                "backend_identity": _public_text(_backend_identity(configured), context.project_root),
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
                    "message": _public_text(item.message, context.project_root),
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
                "estimate": _public_estimate_projection(plan.estimate.to_dict(), context.project_root),
                "backend_id": _public_text(plan.backend_id, context.project_root),
                "probe_operations": 0,
            }
        )

    @router.post("/training/api/cloud-challenge")
    @product_api
    async def issue_cloud_challenge(request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_training_request", "Cloud authorization must be a JSON object.")
        confirmation, confirmation_error = _strict_cloud_confirmation(payload)
        if confirmation_error is not None:
            return confirmation_error
        if confirmation is not True:
            return api_error(
                422,
                "cloud_confirmation_required",
                "A fresh cloud-cost confirmation is required for this exact action.",
                next_action="Review the saved backend and cost notice, then confirm immediately before launch.",
            )
        action = payload.get("action")
        if not isinstance(action, str) or action not in {"start", "resume"}:
            return api_error(
                422,
                "cloud_challenge_action_invalid",
                "Cloud authorization is available only for Start or Resume.",
            )
        if (
            payload.get("profile", TrainingProfile.RECOMMENDED.value) != TrainingProfile.RECOMMENDED.value
            or payload.get("custom") is not None
        ):
            return api_error(
                422,
                "conditioned_profile_ineligible",
                "Production conditioned training accepts only the recommended profile.",
            )
        supplied_run_id = payload.get("run_id")
        if action == "resume":
            if (
                not isinstance(supplied_run_id, str)
                or not supplied_run_id
                or supplied_run_id.strip() != supplied_run_id
            ):
                return api_error(
                    422,
                    "cloud_challenge_run_invalid",
                    "Resume cloud authorization requires the exact retained run.",
                )
            run_id: str | None = supplied_run_id
        else:
            if supplied_run_id is not None:
                return api_error(
                    422,
                    "cloud_challenge_run_invalid",
                    "Start cloud authorization cannot target a retained run.",
                )
            run_id = None
        try:
            configured, version, _saved = settings()
        except (ValueError, ProductSettingsError):
            return api_error(
                409,
                "compute_configuration_invalid",
                "The saved compute configuration could not be verified.",
                next_action="Review and save the current compute settings before authorizing cloud use.",
            )
        training_service = current_service()
        if configured.backend_type == "runpod":
            return api_error(
                409,
                "runpod_unavailable",
                "RunPod is not available in this build and cannot launch.",
                recoverable=False,
            )
        binding_error = _fresh_launch_binding_error(
            payload,
            configured=configured,
            configuration_version=version,
            training_service=training_service,
            run_id=run_id,
        )
        if binding_error is not None:
            return binding_error
        if not _is_cloud_backend(configured, training_service):
            return api_error(
                409,
                "cloud_challenge_not_required",
                "The authoritative backend does not require cloud authorization.",
            )
        result = training_service.issue_cloud_challenge(
            action=action,
            run_id=run_id,
            profile=TrainingProfile.RECOMMENDED,
        )
        if result.status != ProductStatus.READY or not isinstance(result.data, Mapping):
            return api_error(
                409,
                "cloud_challenge_unavailable",
                "A fresh cloud authorization could not be issued safely.",
                recoverable=True,
                next_action="Reload the current training state, review all gates, and confirm the action again.",
            )
        challenge = result.data.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            challenge = result.data.get("challenge_token")
        if not isinstance(challenge, str) or not challenge:
            return api_error(
                409,
                "cloud_challenge_unavailable",
                "A fresh cloud authorization could not be issued safely.",
                recoverable=True,
                next_action="Reload the current training state and confirm the action again.",
            )
        return JSONResponse(
            {
                "status": ProductStatus.READY.value,
                "message": "Fresh cloud authorization is ready for this exact action.",
                "data": {"challenge": challenge},
            },
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/training/api/start")
    @product_api
    async def start_training(request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_training_request", "Training request must be a JSON object.")
        _confirmation, confirmation_error = _strict_cloud_confirmation(payload)
        if confirmation_error is not None:
            return confirmation_error
        configured, version, _saved = settings()
        training_service = current_service()
        service_context = _service_context(training_service, context)
        if configured.backend_type == "runpod":
            return api_error(
                409,
                "runpod_unavailable",
                "RunPod is not available in this build and cannot launch.",
                recoverable=False,
                next_action="Choose Local computer, Remote SSH machine, or a registered provider plugin.",
            )
        binding_error = _fresh_launch_binding_error(
            payload,
            configured=configured,
            configuration_version=version,
            training_service=training_service,
        )
        if binding_error is not None:
            return binding_error
        try:
            profile = TrainingProfile(str(payload.get("profile") or configured.run_profile))
        except ValueError:
            return api_error(422, "training_profile_invalid", "The selected training profile is invalid.")
        if profile is not TrainingProfile.RECOMMENDED or payload.get("custom") is not None:
            return api_error(
                422,
                "conditioned_profile_ineligible",
                "Production conditioned training accepts only the recommended profile.",
            )
        challenge, challenge_error = _launch_challenge(payload, configured, training_service)
        if challenge_error is not None:
            return challenge_error
        custom = None
        contract = _conditioned_contract(service_context, profile, custom_spec=custom)
        if contract.get("ready") is not True:
            return api_error(
                409,
                "conditioned_dataset_contract_required",
                "Training requires an audited conditioned Dataset-v5 freeze and an exactly bound campaign.",
                recoverable=True,
                next_action="Complete the conditioned dataset, independent audits, campaign binding, and execution authorizations.",
            )
        if challenge is None:
            result = training_service.start(profile, custom_spec=custom)
        else:
            result = training_service.start(
                profile,
                custom_spec=custom,
                cloud_challenge=challenge,
            )
        if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
            public = _public_start_projection(result, service_context.project_root)
            return api_error(
                409,
                "training_launch_blocked",
                _public_text(result.message, service_context.project_root),
                recoverable=True,
                next_action=_public_text(
                    _next_action(result, "Resolve the displayed safety gate, then start again."),
                    service_context.project_root,
                ),
                details=public["data"],
                include_details=True,
            )
        return JSONResponse(_public_start_projection(result, service_context.project_root))

    @router.get("/training/api/runs/{run_id}")
    @product_api
    def dashboard(run_id: str) -> JSONResponse:
        training_service = current_service()
        service_context = _service_context(training_service, context)
        result = training_service.dashboard(run_id)
        if result.status == ProductStatus.UNAVAILABLE:
            return api_error(404, "training_run_not_found", _public_text(result.message, service_context.project_root))
        return JSONResponse(_public_dashboard_result(result, service_context.project_root))

    @router.post("/training/api/runs/{run_id}/refresh")
    @product_api
    def refresh_dashboard(run_id: str) -> JSONResponse:
        training_service = current_service()
        service_context = _service_context(training_service, context)
        result = training_service.refresh(run_id)
        if result.status == ProductStatus.UNAVAILABLE:
            return api_error(404, "training_run_not_found", _public_text(result.message, service_context.project_root))
        return JSONResponse(_public_dashboard_result(result, service_context.project_root))

    @router.post("/training/api/runs/{run_id}/pause")
    @product_api
    def pause(run_id: str) -> JSONResponse:
        training_service = current_service()
        service_context = _service_context(training_service, context)
        return _action_response(training_service.pause(run_id), "pause", service_context.project_root)

    @router.post("/training/api/runs/{run_id}/cancel")
    @product_api
    def cancel(run_id: str) -> JSONResponse:
        training_service = current_service()
        service_context = _service_context(training_service, context)
        return _action_response(training_service.cancel(run_id), "cancel", service_context.project_root)

    @router.post("/training/api/runs/{run_id}/resume")
    @product_api
    async def resume(run_id: str, request: Request) -> JSONResponse:
        payload = await _json_mapping(request)
        if payload is None:
            return api_error(400, "invalid_training_request", "Resume settings must be a JSON object.")
        if (
            payload.get("profile", TrainingProfile.RECOMMENDED.value) != TrainingProfile.RECOMMENDED.value
            or payload.get("custom") is not None
        ):
            return api_error(
                422,
                "conditioned_profile_ineligible",
                "Production conditioned training accepts only the recommended profile.",
            )
        _confirmation, confirmation_error = _strict_cloud_confirmation(payload)
        if confirmation_error is not None:
            return confirmation_error
        configured, version, _saved = settings()
        training_service = current_service()
        service_context = _service_context(training_service, context)
        binding_error = _fresh_launch_binding_error(
            payload,
            configured=configured,
            configuration_version=version,
            training_service=training_service,
            run_id=run_id,
        )
        if binding_error is not None:
            return binding_error
        challenge, challenge_error = _launch_challenge(payload, configured, training_service)
        if challenge_error is not None:
            return challenge_error
        result = (
            training_service.resume(run_id)
            if challenge is None
            else training_service.resume(run_id, cloud_challenge=challenge)
        )
        return _action_response(result, "resume", service_context.project_root)

    def run_action(action: str, run_id: str, payload: Mapping[str, Any]) -> ProductResult:
        if action == "pause":
            training_service = current_service()
            service_context = _service_context(training_service, context)
            return _public_action_result(training_service.pause(run_id), service_context.project_root)
        if action == "cancel":
            training_service = current_service()
            service_context = _service_context(training_service, context)
            return _public_action_result(training_service.cancel(run_id), service_context.project_root)
        if action == "resume":
            if (
                payload.get("profile", TrainingProfile.RECOMMENDED.value) != TrainingProfile.RECOMMENDED.value
                or payload.get("custom") is not None
            ):
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Production conditioned training accepts only the recommended profile.",
                    feature="training",
                )
            _confirmation, confirmation_error = _strict_cloud_confirmation(payload)
            if confirmation_error is not None:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Resume requires an exact boolean cloud confirmation.",
                    feature="training",
                )
            configured, version, _saved = settings()
            training_service = current_service()
            service_context = _service_context(training_service, context)
            binding_error = _fresh_launch_binding_error(
                payload,
                configured=configured,
                configuration_version=version,
                training_service=training_service,
                run_id=run_id,
            )
            if binding_error is not None:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Resume authorization is stale or does not match the authoritative compute backend.",
                    feature="training",
                )
            challenge, challenge_error = _launch_challenge(payload, configured, training_service)
            if challenge_error is not None:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Resume requires a fresh one-use cloud authorization for the exact retained run.",
                    feature="training",
                )
            result = (
                training_service.resume(run_id)
                if challenge is None
                else training_service.resume(run_id, cloud_challenge=challenge)
            )
            return _public_action_result(result, service_context.project_root)
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


def _preparation_projection(state: Mapping[str, Any], project_root: Any) -> dict[str, Any]:
    projected: dict[str, Any] = {
        key: state.get(key)
        for key in (
            "schema_version",
            "job_id",
            "status",
            "current",
            "total",
            "input_identity",
            "source_identity",
            "config_identity",
            "code_identity",
            "started_at",
            "updated_at",
            "result_identity",
        )
    }
    projected["logs"] = _public_text_rows(state.get("logs"), project_root)
    error = state.get("error")
    if isinstance(error, Mapping):
        projected["error"] = {
            key: _public_text(error.get(key), project_root)
            for key in ("code", "message", "item_id", "next_action")
            if isinstance(error.get(key), str)
        }
        if isinstance(error.get("reasons"), (list, tuple)):
            projected["error"]["reasons"] = _public_text_rows(error["reasons"], project_root)
    else:
        projected["error"] = None
    result = state.get("result")
    if isinstance(result, Mapping):
        public_result: dict[str, Any] = {}
        image_count = _public_number(result.get("image_count"))
        if image_count is not None:
            public_result["image_count"] = image_count
        for key in ("artifact_kind", "required_contract", "remaining_gate"):
            if isinstance(result.get(key), str):
                public_result[key] = _public_text(result[key], project_root)
        for key in ("immutable", "production_authorized", "training_authorized", "activated", "paths_exposed"):
            if key in result:
                public_result[key] = _public_bool(result.get(key))
        projected["result"] = public_result
    else:
        projected["result"] = None
    if isinstance(error, Mapping) and error.get("code") == "canonical_encoding_failed":
        item_id = error.get("item_id")
        if isinstance(item_id, str) and item_id:
            public_error = dict(projected["error"])
            public_error["image_url"] = f"/training/api/preparation/error-image?item_id={quote(item_id, safe='')}"
            public_error["review_url"] = "/dataset/review"
            projected["error"] = public_error
    return projected


def _backend_identity(configured: ComputeSettings) -> str:
    if configured.backend_type == "other":
        return str(configured.backend_id or "")
    return configured.backend_type


def _service_context(training_service: Any, fallback: ProjectContext) -> ProjectContext:
    service_context = getattr(training_service, "context", None)
    return service_context if isinstance(service_context, ProjectContext) else fallback


def _strict_cloud_confirmation(payload: Mapping[str, Any]) -> tuple[bool, JSONResponse | None]:
    if "confirm_cloud" not in payload:
        return False, None
    confirmation = payload["confirm_cloud"]
    if type(confirmation) is not bool:
        return False, api_error(
            422,
            "cloud_confirmation_invalid",
            "Cloud confirmation must be the JSON boolean true or false.",
            next_action="Review the selected backend and confirm again from the current Training page.",
        )
    return confirmation, None


def _is_cloud_backend(configured: ComputeSettings, training_service: Any) -> bool:
    configured_cloud = configured.backend_type in {"other", "runpod"} or (
        configured.backend_type == "ssh" and configured.cloud
    )
    backend = getattr(training_service, "backend", None)
    try:
        service_cloud = getattr(backend, "is_cloud", _PUBLIC_DROP)
    except Exception:
        return True
    if service_cloud is _PUBLIC_DROP:
        return configured_cloud
    if type(service_cloud) is not bool:
        # An untyped or unreadable backend claim can never suppress the
        # authorization challenge at the public launch boundary.
        return True
    return configured_cloud or service_cloud


def _launch_challenge(
    payload: Mapping[str, Any],
    configured: ComputeSettings,
    training_service: Any,
) -> tuple[str | None, JSONResponse | None]:
    if not _is_cloud_backend(configured, training_service):
        return None, None
    challenge = payload.get("cloud_challenge")
    if not isinstance(challenge, str) or not challenge or challenge.strip() != challenge:
        return None, api_error(
            422,
            "cloud_challenge_required",
            "A fresh one-use cloud authorization is required for this exact action.",
            next_action="Review the saved backend and cost notice, then confirm immediately before launch.",
        )
    return challenge, None


def _fresh_launch_binding_error(
    payload: Mapping[str, Any],
    *,
    configured: ComputeSettings,
    configuration_version: int,
    training_service: Any,
    run_id: str | None = None,
) -> JSONResponse | None:
    backend = getattr(training_service, "backend", None)
    configured_identity = _backend_identity(configured)
    service_identity = str(getattr(backend, "backend_id", "") or "")
    service_context = getattr(training_service, "context", None)
    service_configuration_matches = False
    if isinstance(service_context, ProjectContext):
        compute = service_context.config.get("compute") if isinstance(service_context.config, Mapping) else None
        training = compute.get("training") if isinstance(compute, Mapping) else None
        if isinstance(training, Mapping):
            try:
                service_settings = ComputeSettings.from_mapping(training, allow_unavailable=True)
            except (TypeError, ValueError):
                pass
            else:
                service_configuration_matches = service_settings.to_persisted_dict() == configured.to_persisted_dict()
    is_cloud = _is_cloud_backend(configured, training_service)

    supplied_version = payload.get("compute_configuration_version")
    if type(supplied_version) is not int or supplied_version != configuration_version:
        return api_error(
            409,
            "compute_authorization_stale",
            "The request does not match the current saved compute configuration.",
            next_action="Reload the current compute settings before starting or resuming training.",
        )
    supplied_identity = payload.get("backend_identity")
    if (
        not isinstance(supplied_identity, str)
        or not supplied_identity
        or supplied_identity != configured_identity
        or service_identity != configured_identity
        or not service_configuration_matches
    ):
        return api_error(
            409,
            "compute_backend_mismatch",
            "The request does not match the authoritative compute backend.",
            next_action="Reload the Training page and verify the saved backend before trying again.",
        )
    if run_id is not None:
        repository = getattr(training_service, "repository", None)
        state = repository.state(run_id) if repository is not None and hasattr(repository, "state") else {}
        durable_identity = str(state.get("backend_id") or "") if isinstance(state, Mapping) else ""
        if durable_identity and (durable_identity != configured_identity or durable_identity != service_identity):
            return api_error(
                409,
                "training_run_backend_mismatch",
                "The retained run belongs to a different compute backend and cannot be resumed here.",
                recoverable=False,
                next_action="Restore the run's exact registered backend before attempting safe Resume.",
            )
    if not is_cloud:
        return None
    if payload.get("confirm_cloud") is not True:
        return api_error(
            422,
            "cloud_confirmation_required",
            "A fresh cloud-cost confirmation is required for this Start or Resume request.",
            next_action="Review the current saved compute configuration, then confirm immediately before launch.",
        )
    return None


def _public_compute_settings(configured: ComputeSettings, project_root: Any) -> dict[str, Any]:
    credential_reference = configured.credential_reference
    if credential_reference and credential_reference.startswith("file:"):
        credential_reference = None
    return {
        "backend_type": _public_text(configured.backend_type, project_root),
        "device_policy": _public_text(configured.device_policy, project_root),
        "memory_limit_gb": configured.memory_limit_gb,
        "cpu_threads": configured.cpu_threads,
        "preview_interval": configured.preview_interval,
        "run_profile": _public_text(configured.run_profile, project_root),
        "host": _public_text(configured.host, project_root),
        "port": configured.port,
        "username": _public_text(configured.username, project_root),
        # This is a validated path on the selected remote host, not a local
        # filesystem disclosure.  Preserve it exactly so the editable SSH form
        # can round-trip its required absolute workspace.
        "remote_workspace": configured.remote_workspace,
        "credential_reference": _public_text(credential_reference, project_root),
        "credential_reference_configured": configured.credential_reference is not None,
        "environment_profile": _public_text(configured.environment_profile, project_root),
        "artifact_sync_policy": _public_text(configured.artifact_sync_policy, project_root),
        "cloud": configured.cloud,
        "backend_id": _public_text(configured.backend_id, project_root),
    }


def _public_estimate_projection(value: Any, project_root: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in ("duration_seconds", "disk_required_bytes", "gpu_memory_required_bytes"):
        number = _public_number(source.get(key))
        if number is not None:
            result[key] = number
    if "trustworthy" in source:
        result["trustworthy"] = _public_bool(source.get("trustworthy"))
    if isinstance(source.get("message"), str):
        result["message"] = _public_text(source["message"], project_root)
    return result


def _public_state_projection(payload: Mapping[str, Any], project_root: Any) -> dict[str, Any]:
    source = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    dataset = source.get("dataset") if isinstance(source.get("dataset"), Mapping) else {}
    estimate = source.get("estimate") if isinstance(source.get("estimate"), Mapping) else {}
    data: dict[str, Any] = {
        "profile": _public_string(source.get("profile"), project_root),
        "model_label": _public_string(source.get("model_label"), project_root),
        "dataset": {
            "images": _public_number(dataset.get("images")),
            "status": _public_string(dataset.get("status"), project_root),
        },
        "compute": _public_string(source.get("compute"), project_root),
        "estimate": _public_estimate_projection(estimate, project_root),
        "ready": _public_bool(source.get("ready")),
        "advanced_collapsed": _public_bool(source.get("advanced_collapsed"), default=True),
        "campaign_identity": _public_string(source.get("campaign_identity"), project_root),
        "seeds": [item for item in source.get("seeds", ()) if type(item) is int],
        "checkpoint_schedule": _public_checkpoint_schedule(source.get("checkpoint_schedule"), project_root),
        "resume": _public_resume_projection(source.get("resume"), project_root),
        "implementation_state": _public_string(source.get("implementation_state"), project_root),
        "certification_state": _public_string(source.get("certification_state"), project_root),
        "availability_state": _public_string(source.get("availability_state"), project_root),
        "gates": _public_gate_rows(source.get("gates"), project_root),
        "blockers": _public_gate_rows(source.get("blockers"), project_root),
        "conditioned_dataset_contract": _public_conditioned_contract(
            source.get("conditioned_dataset_contract"), project_root
        ),
    }
    return _public_result_envelope(payload, data=data, project_root=project_root)


def _public_start_projection(result: ProductResult, project_root: Any) -> dict[str, Any]:
    source = result.data if isinstance(result.data, Mapping) else {}
    data: dict[str, Any] = {}
    if isinstance(source.get("run_id"), str):
        data["run_id"] = _public_text(source["run_id"], project_root)
    if isinstance(source.get("dashboard"), Mapping):
        data["dashboard"] = _public_dashboard_projection(source["dashboard"], project_root)
    if isinstance(source.get("training_identity"), Mapping):
        data["training_identity"] = {
            key: _public_text(value, project_root)
            for key, value in source["training_identity"].items()
            if key in {"dataset_identity", "view_identity", "training_view_identity"} and isinstance(value, str)
        }
    if type(source.get("backend_launches")) is int:
        data["backend_launches"] = source["backend_launches"]
    data["seed_outcomes"] = _public_outcome_rows(source.get("seed_outcomes"), kind="seed", project_root=project_root)
    data["job_outcomes"] = _public_outcome_rows(source.get("job_outcomes"), kind="job", project_root=project_root)
    terminal_status = source.get("terminal_status")
    if terminal_status == "CANCELLED":
        data["terminal_status"] = "CANCELLED"
    for key in (
        "cancelled",
        "cancel_available",
        "resource_state_uncertain",
        "may_accrue_cost",
    ):
        if type(source.get(key)) is bool:
            data[key] = source[key]
    for key in (
        "cancel_attempt_count",
        "cancel_unverified_count",
        "prepared_cleanup_attempt_count",
        "prepared_cleanup_unverified_count",
        "unknown_backend_operation_count",
        "pause_attempt_count",
        "pause_unverified_count",
    ):
        if type(source.get(key)) is int and source[key] >= 0:
            data[key] = source[key]
    execution = source.get("execution")
    if isinstance(execution, Mapping):
        launched = execution.get("launched") if isinstance(execution.get("launched"), list) else []
        data["launch"] = {
            "schema_version": _public_string(execution.get("schema_version"), project_root),
            "campaign_id": _public_string(execution.get("campaign_id"), project_root),
            "launched_count": len(launched),
            "run_ids": [
                _public_text(item.get("run_id"), project_root)
                for item in launched
                if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
            ],
        }
    return _public_result_envelope(result.to_dict(), data=data, project_root=project_root)


def _public_dashboard_result(result: ProductResult, project_root: Any) -> dict[str, Any]:
    data = _public_dashboard_projection(result.data, project_root)
    return _public_result_envelope(result.to_dict(), data=data, project_root=project_root)


def _public_dashboard_projection(value: Any, project_root: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    campaign = source.get("campaign_progress") if isinstance(source.get("campaign_progress"), Mapping) else {}
    result: dict[str, Any] = {
        "run_id": _public_string(source.get("run_id"), project_root),
        "backend_id": _public_string(source.get("backend_id"), project_root),
        "status": _public_string(source.get("status"), project_root),
        "terminal_status": "CANCELLED" if source.get("terminal_status") == "CANCELLED" else None,
        "campaign_progress": {
            "current": _public_number(campaign.get("current")),
            "total": _public_number(campaign.get("total")),
        },
        "seeds": _public_numeric_rows(
            source.get("seeds"),
            (
                "seed",
                "status",
                "optimizer_step",
                "total_steps",
                "loss",
                "validation_loss",
                "learning_rate",
                "gradient_norm",
                "gpu_utilization",
                "vram_bytes",
            ),
            project_root,
        ),
        "loss_curve": _public_numeric_rows(source.get("loss_curve"), ("seed", "step", "value"), project_root),
        "validation_loss_curve": _public_numeric_rows(
            source.get("validation_loss_curve"), ("seed", "step", "value"), project_root
        ),
        "learning_rate_curve": _public_numeric_rows(
            source.get("learning_rate_curve"), ("seed", "step", "value"), project_root
        ),
        "checkpoints": _public_checkpoint_rows(source.get("checkpoints"), project_root),
        "latest_verified_checkpoint": _public_checkpoint_row(source.get("latest_verified_checkpoint"), project_root),
        "last_safe_resume_point": _public_checkpoint_row(source.get("last_safe_resume_point"), project_root),
        "checkpoint_schedule": _public_checkpoint_schedule(source.get("checkpoint_schedule"), project_root),
        "estimated_completion": _public_string(source.get("estimated_completion"), project_root),
        "pause_available": _public_bool(source.get("pause_available")),
        "resume_available": _public_bool(source.get("resume_available")),
        "cancel_available": _public_bool(source.get("cancel_available")),
        "unsafe_resume_available": False,
        "logs": _public_text_rows(source.get("logs"), project_root),
        "warnings": _public_text_rows(source.get("warnings"), project_root),
        "previews": _public_preview_rows(source.get("previews"), project_root),
        "seed_outcomes": _public_outcome_rows(source.get("seed_outcomes"), kind="seed", project_root=project_root),
        "job_outcomes": _public_outcome_rows(source.get("job_outcomes"), kind="job", project_root=project_root),
        "unknown_backend_operation_count": (
            source["unknown_backend_operation_count"]
            if type(source.get("unknown_backend_operation_count")) is int
            and source["unknown_backend_operation_count"] >= 0
            else 0
        ),
        "remote_resource_uncertain": _public_bool(source.get("remote_resource_uncertain")),
        "may_accrue_cost": _public_bool(source.get("may_accrue_cost")),
        "shutdown_guidance": _public_string(source.get("shutdown_guidance"), project_root),
    }
    for key in ("elapsed_seconds", "eta_seconds"):
        number = _public_number(source.get(key))
        if number is not None:
            result[key] = number
    if type(source.get("event_cursor")) is int and source["event_cursor"] >= 0:
        result["event_cursor"] = source["event_cursor"]
    return result


def _public_outcome_rows(value: Any, *, kind: str, project_root: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or kind not in {"seed", "job"}:
        return []
    rows: list[dict[str, Any]] = []
    text_keys = ("run_id", "status", "stage") if kind == "seed" else ("job_id", "run_id", "status", "stage")
    boolean_keys = ("may_accrue_cost", "resource_shutdown_verified") if kind == "job" else ()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row = {key: _public_text(item.get(key), project_root) for key in text_keys if isinstance(item.get(key), str)}
        if kind == "seed" and type(item.get("seed")) is int:
            row["seed"] = item["seed"]
        for key in boolean_keys:
            if type(item.get(key)) is bool:
                row[key] = item[key]
        if (kind == "seed" and "run_id" in row) or (kind == "job" and "job_id" in row):
            rows.append(row)
    return rows


def _public_checkpoint_row(value: Any, project_root: Any = None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    step = value.get("optimizer_step") if type(value.get("optimizer_step")) is int else 0
    sha256 = value.get("sha256") if isinstance(value.get("sha256"), str) else None
    result: dict[str, Any] = {}
    for key in ("seed", "optimizer_step"):
        if type(value.get(key)) is int:
            result[key] = value[key]
    for key in ("sha256", "backend_id"):
        if isinstance(value.get(key), str):
            result[key] = _public_text(value[key], project_root)
    for key in ("remote", "downloaded", "hash_verified", "remote_identity_verified", "safe_resume"):
        if key in value:
            result[key] = _public_bool(value.get(key))
    for key in ("synchronization", "verification"):
        if key not in value:
            continue
        projected = _public_nested_value(value.get(key), project_root, field_name=key)
        if projected is not _PUBLIC_DROP:
            result[key] = projected
    public_sha256 = _public_text(sha256, project_root) if sha256 else None
    result["checkpoint_id"] = public_sha256[:12] if public_sha256 else f"step-{step}"
    result["checkpoint_label"] = f"Checkpoint at step {step}"
    return result


def _public_checkpoint_rows(value: Any, project_root: Any = None) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [projected for item in value if (projected := _public_checkpoint_row(item, project_root)) is not None]


def _public_checkpoint_schedule(value: Any, project_root: Any = None) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, list[int]] = {}
        for key, child in value.items():
            if not isinstance(key, (str, int)) or not isinstance(child, (list, tuple)):
                continue
            raw_key = str(key)
            if _is_private_public_key(raw_key):
                continue
            public_key = _public_text(raw_key, project_root)
            if not isinstance(public_key, str) or not public_key or public_key in result:
                continue
            result[public_key] = [item for item in child if type(item) is int]
        return result
    if isinstance(value, (list, tuple)):
        return [item for item in value if type(item) is int]
    return []


def _public_numeric_rows(value: Any, keys: tuple[str, ...], project_root: Any = None) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in keys:
            child = item.get(key)
            if key == "status" and isinstance(child, str):
                row[key] = _public_text(child, project_root)
                continue
            number = _public_number(child)
            if number is not None:
                row[key] = number
        rows.append(row)
    return rows


def _public_preview_rows(value: Any, project_root: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    allowed = {
        "training_seed",
        "generation_seed",
        "optimizer_step",
        "prompt",
        "parameters",
        "exploratory",
        "benchmark_evidence",
        "promotion_evidence",
    }
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {}
        for key, child in item.items():
            if key not in allowed:
                continue
            projected = _public_nested_value(child, project_root, field_name=str(key))
            if projected is not _PUBLIC_DROP:
                row[str(key)] = projected
        rows.append(row)
    return rows


def _public_gate_rows(value: Any, project_root: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "gate_id": _public_string(item.get("gate_id"), project_root),
                "passed": _public_bool(item.get("passed")),
                "message": _public_string(item.get("message"), project_root),
                "resolution": _public_string(item.get("resolution"), project_root),
            }
        )
    return rows


def _public_resume_row(value: Mapping[str, Any], project_root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("run_id", "status", "next_action"):
        if isinstance(value.get(key), str):
            result[key] = _public_text(value[key], project_root)
    resume_step = _public_number(value.get("resume_step"))
    if resume_step is not None:
        result["resume_step"] = resume_step
    return result


def _public_resume_projection(value: Any, project_root: Any = None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    runs = value.get("runs") if isinstance(value.get("runs"), (list, tuple)) else []
    return {
        "schema_version": _public_string(value.get("schema_version"), project_root),
        "safe": _public_bool(value.get("safe")),
        "run_count": len(runs),
        "error_count": len(value.get("errors")) if isinstance(value.get("errors"), (list, tuple)) else 0,
        "foreign_root_count": len(value.get("foreign_run_roots"))
        if isinstance(value.get("foreign_run_roots"), (list, tuple))
        else 0,
        "runs": [_public_resume_row(item, project_root) for item in runs if isinstance(item, Mapping)],
    }


def _public_conditioned_contract(value: Any, project_root: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in (
        "schema_version",
        "profile",
        "required_freeze_schema",
        "freeze_sha256",
        "campaign_identity_sha256",
        "training_code_identity_sha256",
    ):
        if isinstance(source.get(key), str):
            result[key] = _public_text(source[key], project_root)
    if "image_count" in source:
        image_count = _public_number(source.get("image_count"))
        if image_count is not None:
            result["image_count"] = image_count
    for key in ("ready", "paths_exposed"):
        if key in source:
            result[key] = _public_bool(source.get(key))
    if "audit_status" in source:
        audit_status = _public_nested_value(source.get("audit_status"), project_root, field_name="audit_status")
        if audit_status is not _PUBLIC_DROP:
            result["audit_status"] = audit_status
    blockers = source.get("blockers") if isinstance(source.get("blockers"), (list, tuple)) else []
    result["blockers"] = [
        {
            "code": _public_string(item.get("code"), project_root),
            "message": _public_string(item.get("message"), project_root),
        }
        for item in blockers
        if isinstance(item, Mapping)
    ]
    return result


def _public_action_projection(value: Mapping[str, Any], project_root: Any) -> dict[str, Any]:
    result = {
        key: _public_text(value.get(key), project_root)
        for key in ("action_id", "feature", "title")
        if isinstance(value.get(key), str)
    }
    if "requires_confirmation" in value:
        result["requires_confirmation"] = _public_bool(value.get("requires_confirmation"))
    return result


def _public_result_envelope(
    payload: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
    project_root: Any,
) -> dict[str, Any]:
    action = payload.get("action") if isinstance(payload.get("action"), Mapping) else None
    run = payload.get("run") if isinstance(payload.get("run"), Mapping) else None
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), (list, tuple)) else []
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), (list, tuple)) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), (list, tuple)) else []
    return {
        "schema_version": _public_string(payload.get("schema_version"), project_root),
        "status": _public_string(payload.get("status"), project_root),
        "message": _public_string(payload.get("message"), project_root),
        "feature": _public_string(payload.get("feature"), project_root),
        "action": _public_action_projection(action, project_root) if action else None,
        "run": {
            key: _public_text(run.get(key), project_root)
            for key in ("run_id", "feature", "action_id", "status", "backend_id", "started_at", "ended_at")
            if isinstance(run.get(key), str)
        }
        if run
        else None,
        "capabilities": [
            {
                "capability_id": _public_string(item.get("capability_id"), project_root),
                "title": _public_string(item.get("title"), project_root),
                "status": _public_string(item.get("status"), project_root),
                "message": _public_string(item.get("message"), project_root),
            }
            for item in capabilities
            if isinstance(item, Mapping)
        ],
        "blockers": [
            {
                "code": _public_string(item.get("code"), project_root),
                "message": _public_string(item.get("message"), project_root),
                "resolution": _public_string(item.get("resolution"), project_root),
            }
            for item in blockers
            if isinstance(item, Mapping)
        ],
        "warnings": [
            {
                "code": _public_string(item.get("code"), project_root),
                "message": _public_string(item.get("message"), project_root),
                "resolution": _public_string(item.get("resolution"), project_root),
            }
            for item in warnings
            if isinstance(item, Mapping)
        ],
        "data": dict(data),
    }


def _public_action_result(result: ProductResult, project_root: Any) -> ProductResult:
    public = _public_start_projection(result, project_root)
    action: ProductAction | None = None
    action_data = public.get("action")
    if (
        isinstance(action_data, Mapping)
        and isinstance(action_data.get("action_id"), str)
        and isinstance(action_data.get("feature"), str)
        and isinstance(action_data.get("title"), str)
    ):
        action = ProductAction(
            action_data["action_id"],
            action_data["feature"],
            action_data["title"],
            requires_confirmation=_public_bool(action_data.get("requires_confirmation")),
        )
    run: ProductRun | None = None
    run_data = public.get("run")
    if (
        isinstance(run_data, Mapping)
        and isinstance(run_data.get("run_id"), str)
        and isinstance(run_data.get("feature"), str)
        and isinstance(run_data.get("action_id"), str)
        and isinstance(run_data.get("status"), str)
    ):
        try:
            run_status = ProductStatus(run_data["status"])
        except ValueError:
            pass
        else:
            run = ProductRun(
                run_data["run_id"],
                run_data["feature"],
                run_data["action_id"],
                run_status,
                backend_id=run_data.get("backend_id") if isinstance(run_data.get("backend_id"), str) else None,
                started_at=run_data.get("started_at") if isinstance(run_data.get("started_at"), str) else None,
                ended_at=run_data.get("ended_at") if isinstance(run_data.get("ended_at"), str) else None,
            )
    return ProductResult(
        result.status,
        str(public["message"] or ""),
        feature=public["feature"] if isinstance(public.get("feature"), str) else None,
        action=action,
        run=run,
        data=dict(public["data"]),
    )


def _public_bool(value: Any, *, default: bool = False) -> bool:
    return value if type(value) is bool else default


def _public_number(value: Any) -> int | float | None:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return None


def _is_private_public_key(value: str) -> bool:
    return is_sensitive_public_key(value)


def _is_public_boolean_field(value: str) -> bool:
    normalized = normalize_public_field_name(value)
    return normalized in _PUBLIC_BOOLEAN_FIELDS or normalized.endswith(
        (
            "_available",
            "_authorized",
            "_eligible",
            "_enabled",
            "_passed",
            "_ready",
            "_safe",
            "_uncertain",
            "_validated",
            "_verified",
        )
    )


def _public_nested_value(
    value: Any,
    project_root: Any,
    *,
    field_name: str,
    depth: int = 0,
) -> Any:
    if _is_public_boolean_field(field_name):
        return _public_bool(value)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else _PUBLIC_DROP
    if isinstance(value, str):
        return _public_text(value, project_root)
    if depth >= 8:
        return _PUBLIC_DROP
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or _is_private_public_key(raw_key):
                continue
            public_key = _public_text(raw_key, project_root)
            if not isinstance(public_key, str) or not public_key or public_key in result:
                continue
            projected = _public_nested_value(
                child,
                project_root,
                field_name=raw_key,
                depth=depth + 1,
            )
            if projected is not _PUBLIC_DROP:
                result[public_key] = projected
        return result
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        for child in value:
            projected = _public_nested_value(
                child,
                project_root,
                field_name=field_name,
                depth=depth + 1,
            )
            if projected is not _PUBLIC_DROP:
                result_list.append(projected)
        return result_list
    return _PUBLIC_DROP


def _public_text_rows(value: Any, project_root: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(_public_text(item, project_root) or "") for item in value if isinstance(item, str)]


def _public_string(value: Any, project_root: Any) -> str | None:
    return _public_text(value, project_root) if isinstance(value, str) else None


def _public_text(value: Any, project_root: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        private_roots = (Path(project_root),) if project_root is not None else ()
    except TypeError:
        private_roots = ()
    return sanitize_public_text(value, private_roots)


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


def _action_response(result: ProductResult, action: str, project_root: Any) -> Any:
    if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
        public = _public_start_projection(result, project_root)
        return api_error(
            409,
            f"training_{action}_unsupported"
            if result.status == ProductStatus.UNAVAILABLE
            else f"training_{action}_blocked",
            _public_text(result.message, project_root),
            recoverable=result.status != ProductStatus.UNAVAILABLE,
            next_action=_public_text(
                _next_action(result, "Review the durable run status and backend capability."),
                project_root,
            ),
            details=public["data"],
            include_details=True,
        )
    return JSONResponse(_public_start_projection(result, project_root))


__all__ = ["create_router"]

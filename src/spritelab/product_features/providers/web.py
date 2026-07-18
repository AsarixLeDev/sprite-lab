"""Side-effect-free provider settings page and explicit provider actions."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from spritelab.product_core import (
    ProductSettingsError,
    ProductSettingsRepository,
    ProjectContext,
    api_error,
    product_api,
)
from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.contracts import DiscoveryState, PrivacyClass, PrivacyPolicy, ProviderMode
from spritelab.product_features.providers.discovery import VisionProviderRegistry
from spritelab.product_features.providers.errors import ProviderError
from spritelab.product_features.providers.state import passive_provider_projection

PLUGIN_ID = "vision.providers"


def create_settings_router(
    context: ProjectContext,
    *,
    registry_factory: Callable[[ProviderSettings], VisionProviderRegistry] | None = None,
) -> object:
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, JSONResponse

    router = APIRouter()
    repository = ProductSettingsRepository(context)

    def registry(settings: ProviderSettings) -> VisionProviderRegistry:
        return registry_factory(settings) if registry_factory else VisionProviderRegistry(settings)

    def effective() -> tuple[ProviderSettings, int, bool]:
        raw, version, saved = repository.effective_settings("provider")
        return ProviderSettings.from_mapping(raw), version, saved

    @router.get("/settings/vision", response_class=HTMLResponse)
    def vision_settings_page(request: Request) -> Any:
        try:
            settings, version, saved = effective()
            projection = passive_provider_projection(repository.effective_context())
            error = ""
        except (ValueError, ProductSettingsError) as exc:
            settings, version, saved = ProviderSettings(), 0, False
            projection = {
                "state": "not_configured",
                "observation_timestamp": None,
                "configuration_version": 0,
            }
            error = str(exc)
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(
                request,
                PLUGIN_ID,
                "providers.html",
                {
                    "settings": settings,
                    "settings_payload": settings.to_persisted_dict(),
                    "configuration_version": version,
                    "settings_saved": saved,
                    "provider_projection": projection,
                    "provider_actions_available": projection.get("state") != "not_configured",
                    "provider_models_available": projection.get("state") == "previously_verified",
                    "provider_clear_available": saved,
                    "settings_error": error,
                },
            )
        return HTMLResponse(_settings_html(settings, error=error, projection=projection))

    @router.get("/api/vision-providers")
    @product_api
    def configured_providers() -> JSONResponse:
        """Compatibility listing: construction only, with no probe or request."""

        try:
            settings, _version, _saved = effective()
            providers = registry(settings).providers()
        except (ValueError, ProductSettingsError) as exc:
            return api_error(400, "provider_configuration_invalid", str(exc))
        return JSONResponse(
            {
                "providers": [
                    {
                        "provider_id": provider.provider_id,
                        "display_name": provider.display_name,
                        "provider_kind": provider.provider_kind.value,
                        "privacy_class": provider.privacy_class.value,
                        "state": "not_checked",
                    }
                    for provider in providers
                ],
                "provider_requests": 0,
            }
        )

    @router.post("/settings/vision/api/settings")
    @product_api
    async def save_settings(request: Request) -> JSONResponse:
        body = await _json_mapping(request)
        if body is None:
            return api_error(400, "invalid_provider_settings", "Settings must be a JSON object.")
        if _contains_image_payload(body):
            return api_error(400, "image_payload_not_allowed", "Provider settings never accept image data.")
        try:
            settings = ProviderSettings.from_mapping(body)
            saved = repository.save("provider", settings.to_persisted_dict())
        except (ValueError, ProductSettingsError) as exc:
            return api_error(
                422,
                "invalid_provider_settings",
                str(exc),
                recoverable=True,
                next_action="Correct the provider setting and save again.",
            )
        return JSONResponse(
            {
                "status": "saved",
                "configuration_version": saved["configuration_version"],
                "provider_requests": 0,
                "message": "Provider settings were saved. No connection was made.",
            }
        )

    @router.delete("/settings/vision/api/settings")
    @product_api
    def clear_settings() -> JSONResponse:
        try:
            cleared = repository.clear("provider")
        except ProductSettingsError as exc:
            return api_error(409, "provider_settings_clear_failed", str(exc))
        return JSONResponse(
            {
                "status": "cleared" if cleared else "already_clear",
                "provider_requests": 0,
                "message": "Saved provider configuration was cleared.",
            }
        )

    @router.post("/settings/vision/api/detect")
    @router.post("/api/vision-providers/detect")
    @product_api
    def detect_providers() -> JSONResponse:
        try:
            settings, version, _saved = effective()
            # One explicit registry discovery operation. Individual candidates
            # may perform their documented health-only probe; no image is sent.
            discovered = registry(settings).discover()
            state = _observation_state([item.probe.state.value for item in discovered])
            observation = repository.record_observation(
                "provider",
                {
                    "configuration_version": version,
                    "action": "detect",
                    "state": state,
                    "providers": [
                        {
                            "provider_id": item.provider.provider_id,
                            "state": item.probe.state.value,
                            "privacy_class": item.provider.privacy_class.value,
                        }
                        for item in discovered
                    ],
                },
            )
        except (ValueError, ProductSettingsError, ProviderError) as exc:
            return api_error(
                503,
                "provider_detection_failed",
                str(exc),
                recoverable=True,
                next_action="Check the endpoint and credential reference, then detect again.",
            )
        return JSONResponse(
            {
                "providers": [item.to_dict() for item in discovered],
                "observation_timestamp": observation["observed_at"],
                "discovery_operations": 1,
                "image_inference_requests": 0,
            }
        )

    @router.post("/settings/vision/api/test")
    @router.post("/api/vision-providers/test")
    @product_api
    async def test_connection(request: Request) -> JSONResponse:
        body = await _json_mapping(request)
        if body is None:
            return api_error(400, "invalid_provider_settings", "Settings must be a JSON object.")
        if _contains_image_payload(body):
            return api_error(
                400,
                "image_payload_not_allowed",
                "The connection-test action does not accept or transmit images.",
            )
        try:
            current, version, _saved = effective()
            settings = ProviderSettings.from_mapping(body) if body else current
            selected = _select_unprobed(registry(settings), settings, provider_id=body.get("provider_id"))
            if selected is None:
                return api_error(
                    503,
                    "provider_not_configured",
                    "No provider is configured for a connection test.",
                    next_action="Save provider settings or choose Automatic, then test again.",
                )
            probe = await run_in_threadpool(selected.probe)
            validation = await run_in_threadpool(selected.validate_model, settings.model) if probe.available else None
            observed_state = validation.state if validation is not None else probe.state
            observation = repository.record_observation(
                "provider",
                {
                    "configuration_version": version,
                    "action": "test",
                    "state": observed_state.value,
                    "provider_id": selected.provider_id,
                },
            )
        except (ValueError, ProductSettingsError, ProviderError) as exc:
            return api_error(400, "provider_test_failed", str(exc))
        available = observed_state == DiscoveryState.AVAILABLE
        payload = {
            **probe.to_dict(),
            "state": observed_state.value,
            "message": validation.message if validation is not None else probe.message,
            "available": available,
            "model_validation": {
                "model_id": validation.model_id,
                "state": validation.state.value,
                "message": validation.message,
                "capabilities": list(validation.capabilities),
            }
            if validation is not None
            else None,
            "display_name": selected.display_name,
            "privacy_class": selected.privacy_class.value,
            "test_kind": "one model-list health check; no image inference",
            "probe_operations": 1,
            "image_inference_requests": 0,
            "observation_timestamp": observation["observed_at"],
        }
        return JSONResponse(status_code=200, content=payload)

    @router.post("/settings/vision/api/models/refresh")
    @product_api
    async def refresh_models(request: Request) -> JSONResponse:
        body = await _json_mapping(request)
        if body is None or _contains_image_payload(body):
            return api_error(400, "invalid_model_refresh", "Model refresh accepts settings, never image data.")
        try:
            current, _version, _saved = effective()
            settings = ProviderSettings.from_mapping(body) if body else current
            selected = _select_unprobed(registry(settings), settings, provider_id=body.get("provider_id"))
            if selected is None:
                return api_error(503, "provider_not_configured", "No provider is configured for model refresh.")
            models = await run_in_threadpool(selected.list_models)
        except (ValueError, ProductSettingsError, ProviderError) as exc:
            return api_error(503, "model_refresh_failed", str(exc))
        return JSONResponse(
            {
                "provider_id": selected.provider_id,
                "models": [
                    {
                        "model_id": model.model_id,
                        "display_name": model.display_name,
                        "capabilities": list(model.capabilities),
                        "metadata": dict(model.metadata),
                    }
                    for model in models
                ],
                "model_list_requests": 1,
                "image_inference_requests": 0,
            }
        )

    return router


async def _json_mapping(request: Any) -> dict[str, Any] | None:
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return dict(body) if isinstance(body, Mapping) else None


def _contains_image_payload(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("image", "images", "image_test", "files", "file"))


def _select_unprobed(
    registry: VisionProviderRegistry,
    settings: ProviderSettings,
    *,
    provider_id: Any = None,
) -> Any | None:
    candidates = list(registry.providers())
    selected_id = str(provider_id or settings.provider_id or "")
    if selected_id:
        return next((item for item in candidates if item.provider_id == selected_id), None)
    if settings.mode == ProviderMode.LOCAL or settings.privacy_policy == PrivacyPolicy.LOCAL_ONLY:
        candidates = [item for item in candidates if item.privacy_class == PrivacyClass.LOCAL]
    elif settings.mode == ProviderMode.HOSTED or settings.privacy_policy == PrivacyPolicy.HOSTED_ONLY:
        candidates = [item for item in candidates if item.privacy_class == PrivacyClass.HOSTED]
    return candidates[0] if candidates else None


def _observation_state(states: Sequence[str]) -> str:
    if "available" in states:
        return "available"
    if "authentication_required" in states:
        return "authentication_required"
    return "unavailable"


def _settings_html(
    settings: ProviderSettings,
    *,
    error: str = "",
    projection: Mapping[str, Any] | None = None,
) -> str:
    """Standalone fallback retained for isolated router and contract tests."""

    selected_mode = settings.mode.value

    def checked(value: str) -> str:
        return " checked" if selected_mode == value else ""

    endpoint = html.escape(settings.endpoint or "")
    model = html.escape(settings.model or "")
    credential_env = html.escape(settings.credential_env or "SPRITELAB_VISION_API_KEY")
    error_html = f'<p role="alert">{html.escape(error)}</p>' if error else ""
    state = html.escape(str((projection or {}).get("state") or "configured_unverified"))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Vision labeling settings</title></head>
<body><main><h1>Vision labeling</h1>{error_html}<p role="status">Provider state: {state}. Provider has not been checked yet.</p>
<form id="vision-provider-settings"><fieldset><legend>Provider mode</legend>
<label><input type="radio" name="type" value="auto"{checked("automatic")}> Automatic</label>
<label><input type="radio" name="type" value="ollama"{checked("specific") if settings.adapter == "ollama" else ""}> Local Ollama</label>
<label><input type="radio" name="type" value="openai_compatible"> vLLM / OpenAI-compatible</label>
<label><input type="radio" name="type" value="hosted"{checked("hosted")}> Hosted endpoint</label>
<label><input type="radio" name="type" value="plugin"{checked("custom_plugin")}> Custom plugin</label></fieldset>
<label>Model <input name="model" value="{model}" placeholder="Auto-detect"></label>
<label>Endpoint <input name="endpoint" value="{endpoint}" placeholder="Explicit http(s) base URL"></label>
<label>Credential environment-variable name <input name="credential_env" value="{credential_env}"></label>
<label>Privacy policy <select name="privacy_policy">{_privacy_options(settings)}</select></label>
<label>Timeout (seconds) <input name="timeout" type="number" value="{settings.timeout_seconds:g}"></label>
<button type="button" id="save">Save settings</button><button type="button" id="detect">Detect providers</button>
<button type="button" id="test">Test connection</button><button type="button" id="refresh-models">Refresh models</button>
<button type="button" id="clear">Clear provider configuration</button>
<div id="summary" role="status" aria-live="polite">Capability summary and batch-size recommendation will appear here. Cost: unknown.</div>
</form></main></body></html>"""


def _privacy_options(settings: ProviderSettings) -> str:
    values = ("local_only", "allow_hosted", "hosted_only", "ask_before_hosted")
    return "".join(
        f'<option value="{value}"{(" selected" if value == settings.privacy_policy.value else "")}>{value}</option>'
        for value in values
    )


__all__ = ["_settings_html", "create_settings_router"]

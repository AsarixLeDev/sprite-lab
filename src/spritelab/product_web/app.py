"""FastAPI application factory for the unified Sprite Lab product shell."""

from __future__ import annotations

import asyncio
import hmac
import importlib.util
import inspect
import logging
import math
import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, FileSystemLoader, PrefixLoader
from starlette.exceptions import HTTPException as StarletteHTTPException

from spritelab.product_core import (
    DEFAULT_API_PREFIXES,
    ProductCapability,
    ProductPlugin,
    ProductPluginRegistry,
    ProductResult,
    ProductStatus,
    ProjectContext,
    WebNavigationItem,
    WebServerSettings,
    api_error,
    product_api,
    request_is_api,
    strict_json_dumps,
)
from spritelab.product_web.components import (
    bar_chart,
    distribution,
    image_gallery,
    line_chart,
    metric_card,
    run_timeline,
)
from spritelab.product_web.events import RUN_ID_PATTERN, EventRepository

LOGGER = logging.getLogger(__name__)
PACKAGE_DIRECTORY = Path(__file__).resolve().parent
CORE_NAVIGATION = (
    WebNavigationItem("home", "Home", "/", 0),
    WebNavigationItem("dataset", "Dataset", "/dataset", 10),
    WebNavigationItem("training", "Training", "/training", 20),
    WebNavigationItem("evaluation", "Evaluation", "/evaluation", 30),
    WebNavigationItem("playground", "Playground", "/playground", 40),
    WebNavigationItem("runs", "Runs", "/runs", 50),
    WebNavigationItem("settings", "Settings", "/settings", 60),
)
AREA_TITLES = {
    "dataset": "Dataset",
    "training": "Training",
    "evaluation": "Evaluation",
    "playground": "Playground",
    "settings": "Settings",
}
HEX_DETAIL_PATTERN = re.compile(r"\b[0-9a-fA-F]{12,64}\b")


@dataclass
class PluginSurface:
    plugin: ProductPlugin
    capabilities: tuple[ProductCapability, ...]
    missing_capabilities: tuple[str, ...]
    mounted: bool


def _package_child(package: str, child: str) -> Path | None:
    """Resolve only package-owned asset directories, including on Windows."""

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    roots = list(spec.submodule_search_locations or ())
    if not roots and spec.origin:
        roots = [str(Path(spec.origin).parent)]
    for root in roots:
        base = Path(root).resolve()
        candidate = (base / child).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        if candidate.is_dir():
            return candidate
    return None


def _probe(plugin: ProductPlugin, context: ProjectContext) -> tuple[ProductCapability, ...]:
    try:
        return tuple(item for item in plugin.capability_probe(context) if isinstance(item, ProductCapability))
    except Exception:
        LOGGER.exception("Capability probe failed for product plugin %s", plugin.plugin_id)
        return ()


def _status(plugin: ProductPlugin, context: ProjectContext) -> ProductResult:
    try:
        result = plugin.status_provider(context)
        if isinstance(result, ProductResult):
            return result
    except Exception:
        LOGGER.exception("Status provider failed for product plugin %s", plugin.plugin_id)
    return ProductResult(
        status=ProductStatus.FAILED,
        feature=plugin.title,
        message="Status is temporarily unavailable.",
    )


def _is_developer_navigation(item: WebNavigationItem) -> bool:
    navigation_id = item.navigation_id.lower()
    path = item.path.lower()
    return navigation_id.startswith(("dev", "developer", "audit", "internal")) or path.startswith(
        ("/dev", "/developer", "/audit", "/internal")
    )


def _navigation(surfaces: list[PluginSurface]) -> list[WebNavigationItem]:
    items = {item.path: item for item in CORE_NAVIGATION}
    for surface in surfaces:
        for item in surface.plugin.navigation:
            if not _is_developer_navigation(item):
                items[item.path] = item
    return sorted(items.values(), key=lambda item: (item.order, item.title.lower()))


def _surface_for_area(surfaces: list[PluginSurface], area: str) -> PluginSurface | None:
    expected = f"/{area}"
    for surface in surfaces:
        if any(item.path == expected or item.navigation_id.lower() == area for item in surface.plugin.navigation):
            return surface
    for surface in surfaces:
        haystack = f"{surface.plugin.plugin_id} {surface.plugin.title}".lower()
        if area in haystack:
            return surface
    return None


def _clean_text(value: Any, context: ProjectContext, *, fallback: str = "") -> str:
    text = str(value or fallback)
    for private in (context.project_root, context.config_path, context.runs_directory):
        if private:
            raw = str(private)
            text = text.replace(raw, "<project>").replace(raw.replace("\\", "/"), "<project>")
    return HEX_DETAIL_PATTERN.sub("[technical detail]", text)


def _project_name(context: ProjectContext) -> str:
    project = context.config.get("project", {})
    if isinstance(project, Mapping):
        value = project.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Sprite Lab project"


def _status_label(value: str) -> str:
    return value.replace("_", " ").title()


def _duration(value: float | None) -> str:
    if value is None:
        return "Not available"
    seconds = max(0, int(value))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _area_card(area: str, surface: PluginSurface | None, context: ProjectContext) -> dict[str, Any]:
    title = AREA_TITLES[area]
    if surface is None:
        reason = {
            "dataset": "No dataset backend is registered.",
            "training": "No training backend is registered.",
            "evaluation": "No evaluation backend is registered.",
            "playground": "No generation backend is registered.",
            "settings": "No settings provider is registered.",
        }[area]
        return {
            "title": title,
            "path": f"/{area}",
            "status": ProductStatus.UNAVAILABLE.value,
            "status_label": "Not available yet",
            "message": reason,
            "value": None,
            "action": None,
        }
    result = _status(surface.plugin, context)
    missing = surface.missing_capabilities
    status = ProductStatus.UNAVAILABLE if missing else result.status
    reason = (
        f"Required capability is not registered: {missing[0]}."
        if len(missing) == 1
        else f"Required capabilities are not registered: {', '.join(missing)}."
        if missing
        else _clean_text(result.message, context, fallback=f"{title} status is available.")
    )
    data = result.data if isinstance(result.data, Mapping) else {}
    value = data.get("usable_images")
    if value is not None:
        try:
            value = f"{int(value):,} usable images"
        except (TypeError, ValueError):
            value = None
    return {
        "title": title,
        "path": f"/{area}",
        "status": status.value,
        "status_label": _status_label(status.value),
        "message": reason,
        "value": value,
        "action": result.action,
    }


def _recommendation(cards: dict[str, dict[str, Any]], current_run: Any) -> dict[str, str]:
    dataset = cards["dataset"]["status"]
    training = cards["training"]["status"]
    evaluation = cards["evaluation"]["status"]
    if current_run and not current_run.terminal:
        return {"title": "Monitor the current run", "path": f"/runs/{current_run.run_id}", "action": "View run"}
    if dataset not in {ProductStatus.READY.value, ProductStatus.COMPLETE.value}:
        return {"title": "Choose an image folder", "path": "/dataset", "action": "Choose image folder"}
    if cards["dataset"].get("value") is None:
        return {"title": "Choose an image folder", "path": "/dataset", "action": "Choose image folder"}
    if training not in {ProductStatus.RUNNING.value, ProductStatus.COMPLETE.value}:
        return {"title": "Start training", "path": "/training", "action": "Start training"}
    if evaluation not in {ProductStatus.READY.value, ProductStatus.COMPLETE.value}:
        return {"title": "Set up evaluation", "path": "/evaluation", "action": "Open evaluation"}
    return {"title": "Try your model", "path": "/playground", "action": "Open playground"}


def _plugin_cards(surfaces: list[PluginSurface], context: ProjectContext) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for surface in surfaces:
        if not surface.plugin.navigation and surface.plugin.web_router_factory is None:
            continue
        result = _status(surface.plugin, context)
        path = surface.plugin.navigation[0].path if surface.plugin.navigation else "/"
        supplied = result.data.get("status_cards", ()) if isinstance(result.data, Mapping) else ()
        if isinstance(supplied, (list, tuple)):
            for item in supplied:
                if not isinstance(item, Mapping):
                    continue
                cards.append(
                    {
                        "title": _clean_text(item.get("title"), context, fallback=surface.plugin.title),
                        "status": _clean_text(item.get("status"), context, fallback=result.status.value),
                        "message": _clean_text(item.get("message"), context, fallback=result.message),
                        "path": str(item.get("path")) if str(item.get("path", "")).startswith("/") else path,
                    }
                )
        if not supplied:
            cards.append(
                {
                    "title": surface.plugin.title,
                    "status": ProductStatus.UNAVAILABLE.value if surface.missing_capabilities else result.status.value,
                    "message": _clean_text(result.message, context),
                    "path": path,
                }
            )
    return cards


def _plugin_actions(surfaces: list[PluginSurface], context: ProjectContext) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for surface in surfaces:
        result = _status(surface.plugin, context)
        path = surface.plugin.navigation[0].path if surface.plugin.navigation else "/"
        if result.action:
            actions.append({"title": result.action.title, "path": path})
        supplied = result.data.get("actions", ()) if isinstance(result.data, Mapping) else ()
        if isinstance(supplied, (list, tuple)):
            for item in supplied:
                if not isinstance(item, Mapping):
                    continue
                item_path = str(item.get("path", ""))
                actions.append(
                    {
                        "title": _clean_text(item.get("title"), context, fallback="Open feature"),
                        "path": item_path if item_path.startswith("/") else path,
                    }
                )
    return actions


def _sse(event_id: int | None, event: str, data: Mapping[str, Any]) -> str:
    payload = strict_json_dumps(dict(data), ensure_ascii=False, separators=(",", ":"))
    identifier = f"id: {event_id}\n" if event_id is not None else ""
    return f"{identifier}event: {event}\ndata: {payload}\n\n"


def create_app(
    context: ProjectContext,
    *,
    plugins: Iterable[ProductPlugin] = (),
    settings: WebServerSettings | None = None,
    event_poll_interval: float = 1.0,
) -> FastAPI:
    """Create the complete shell without starting a browser, backend, or provider."""

    effective = settings or WebServerSettings()
    effective.validate()
    registry = ProductPluginRegistry(plugins)
    plugin_list = list(registry)
    probes = {plugin.plugin_id: _probe(plugin, context) for plugin in plugin_list}
    available_capabilities = {
        item.capability_id for capabilities in probes.values() for item in capabilities if item.available
    }
    surfaces = [
        PluginSurface(
            plugin=plugin,
            capabilities=probes[plugin.plugin_id],
            missing_capabilities=tuple(
                capability
                for capability in plugin.required_backend_capabilities
                if capability not in available_capabilities
            ),
            mounted=False,
        )
        for plugin in plugin_list
    ]
    navigation = _navigation(surfaces)
    repository = EventRepository(context.runs_directory, private_roots=(context.project_root,))
    core_templates = PACKAGE_DIRECTORY / "templates"
    templates = Jinja2Templates(directory=str(core_templates))
    plugin_template_loaders: dict[str, FileSystemLoader] = {}
    for plugin in plugin_list:
        for bundle in plugin.web_assets:
            template_directory = _package_child(bundle.package, bundle.templates)
            if template_directory:
                plugin_template_loaders[plugin.plugin_id] = FileSystemLoader(str(template_directory))
                break
    if plugin_template_loaders:
        templates.env.loader = ChoiceLoader(
            [FileSystemLoader(str(core_templates)), PrefixLoader(plugin_template_loaders, delimiter="/")]
        )
    templates.env.globals.update(
        line_chart=line_chart,
        bar_chart=bar_chart,
        distribution=distribution,
        metric_card=metric_card,
        image_gallery=image_gallery,
        run_timeline=run_timeline,
        duration=_duration,
        status_label=_status_label,
    )
    app = FastAPI(title="Sprite Lab", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIRECTORY / "static")), name="product-static")
    csrf_token = secrets.token_urlsafe(32)
    app.state.spritelab_context = context
    app.state.spritelab_plugins = tuple(plugin_list)
    app.state.spritelab_surfaces = surfaces
    app.state.spritelab_events = repository
    app.state.spritelab_settings = effective
    app.state.spritelab_csrf_token = csrf_token
    app.state.spritelab_api_prefixes = tuple(
        sorted(
            {
                *DEFAULT_API_PREFIXES,
                *(prefix for plugin in plugin_list for prefix in plugin.api_prefixes),
            }
        )
    )
    app.state.spritelab_run_action_handlers = {}

    @app.middleware("http")
    async def safe_errors(request: Request, call_next: Any) -> Any:
        try:
            return await call_next(request)
        except Exception:
            reference = f"ERR-{secrets.token_hex(5).upper()}"
            LOGGER.exception("Unexpected web error %s", reference)
            if request_is_api(request):
                return api_error(
                    500,
                    "unexpected_error",
                    "The request could not be completed. Your completed work was preserved when possible.",
                    recoverable=True,
                    next_action="Try again. If the problem continues, use the error reference in the logs.",
                    error_reference=reference,
                )
            return templates.TemplateResponse(
                request=request,
                name="error.html",
                context={
                    **base_context(request),
                    "error_reference": reference,
                    "safe_details": "The request could not be completed. Try again or return home.",
                },
                status_code=500,
            )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(
            "spritelab_csrf",
            csrf_token,
            httponly=False,
            samesite="strict",
            secure=False,
        )
        return response

    if not effective.is_loopback:
        authentication_token = effective.authentication_token or ""

        @app.middleware("http")
        async def runtime_authentication(request: Request, call_next: Any) -> Any:
            if request.url.path == "/auth":
                return await call_next(request)
            authorization = request.headers.get("authorization", "")
            header_token = authorization[7:] if authorization.startswith("Bearer ") else ""
            cookie_token = request.cookies.get("spritelab_auth", "")
            if not (
                hmac.compare_digest(header_token, authentication_token)
                or hmac.compare_digest(cookie_token, authentication_token)
            ):
                if request_is_api(request):
                    return api_error(
                        401,
                        "authentication_required",
                        "Authentication is required for this Sprite Lab session.",
                        recoverable=True,
                        next_action="Authenticate, then retry the request.",
                    )
                return RedirectResponse("/auth", status_code=303)
            return await call_next(request)

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next: Any) -> Any:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path != "/auth":
            supplied = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(supplied, csrf_token):
                if request_is_api(request):
                    return api_error(
                        403,
                        "csrf_validation_failed",
                        "This action expired or did not originate from Sprite Lab.",
                        recoverable=True,
                        next_action="Reload the page and try again.",
                        error_reference="SEC-CSRF",
                    )
                return templates.TemplateResponse(
                    request=request,
                    name="error.html",
                    context={
                        **base_context(request),
                        "error_reference": "SEC-CSRF",
                        "safe_details": "This action expired or did not originate from Sprite Lab. Reload and try again.",
                    },
                    status_code=403,
                )
        return await call_next(request)

    def status_results() -> dict[str, ProductResult]:
        return {surface.plugin.plugin_id: _status(surface.plugin, context) for surface in surfaces}

    def base_context(request: Request) -> dict[str, Any]:
        results = status_results()
        blocker = next(
            (
                _clean_text(item.message, context)
                for result in results.values()
                for item in result.blockers
                if item.message
            ),
            None,
        )
        return {
            "request": request,
            "navigation": navigation,
            "current_path": request.url.path,
            "project_name": _project_name(context),
            "current_run": repository.current_run(),
            "global_blocker": blocker,
            "csrf_token": csrf_token,
            "non_loopback": not effective.is_loopback,
            "plugin_results": results,
        }

    def render_plugin_template(
        request: Request,
        plugin_id: str,
        template_name: str,
        context_values: Mapping[str, Any] | None = None,
        *,
        status_code: int = 200,
    ) -> Any:
        return templates.TemplateResponse(
            request=request,
            name=f"{plugin_id}/{template_name}",
            context={**base_context(request), **dict(context_values or {})},
            status_code=status_code,
        )

    app.state.spritelab_render_plugin_template = render_plugin_template

    @app.exception_handler(StarletteHTTPException)
    async def product_http_exception(request: Request, exc: StarletteHTTPException) -> Any:
        if request_is_api(request):
            message = str(exc.detail) if exc.status_code < 500 else "The request could not be completed."
            return api_error(
                exc.status_code,
                f"http_{exc.status_code}",
                message,
                recoverable=exc.status_code not in {401, 403, 404},
                next_action="Review the request and try again." if exc.status_code < 500 else "Try again later.",
            )
        return templates.TemplateResponse(
            request=request,
            name="not_found.html" if exc.status_code == 404 else "error.html",
            context={
                **base_context(request),
                "error_reference": f"HTTP-{exc.status_code}",
                "safe_details": str(exc.detail),
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def product_validation_exception(request: Request, _exc: RequestValidationError) -> Any:
        if request_is_api(request):
            return api_error(
                422,
                "invalid_request",
                "The request contains an invalid or missing value.",
                recoverable=True,
                next_action="Correct the highlighted value and try again.",
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                **base_context(request),
                "error_reference": "HTTP-422",
                "safe_details": "The submitted form contains an invalid value.",
            },
            status_code=422,
        )

    @app.get("/auth", response_class=HTMLResponse, include_in_schema=False)
    async def authentication_page(request: Request) -> Any:
        if effective.is_loopback:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"request": request, "csrf_token": csrf_token, "project_name": _project_name(context)},
        )

    @app.post("/auth", response_class=HTMLResponse, include_in_schema=False)
    async def authenticate(request: Request) -> Any:
        if effective.is_loopback:
            return RedirectResponse("/", status_code=303)
        body = (await request.body()).decode("utf-8", errors="replace")
        values = parse_qs(body, keep_blank_values=True)
        supplied_csrf = values.get("csrf_token", [""])[0]
        supplied_token = values.get("token", [""])[0]
        expected_token = effective.authentication_token or ""
        if not hmac.compare_digest(supplied_csrf, csrf_token) or not hmac.compare_digest(
            supplied_token, expected_token
        ):
            return templates.TemplateResponse(
                request=request,
                name="auth.html",
                context={
                    "request": request,
                    "csrf_token": csrf_token,
                    "project_name": _project_name(context),
                    "auth_error": "The authentication token was not accepted.",
                },
                status_code=401,
            )
        response = RedirectResponse("/", status_code=303)
        response.set_cookie("spritelab_auth", expected_token, httponly=True, samesite="strict", secure=False)
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def home(request: Request) -> Any:
        cards = {
            area: _area_card(area, _surface_for_area(surfaces, area), context)
            for area in ("dataset", "training", "evaluation")
        }
        current_run = repository.current_run()
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                **base_context(request),
                "cards": cards,
                "recommendation": _recommendation(cards, current_run),
                "plugin_cards": _plugin_cards(surfaces, context),
                "plugin_actions": _plugin_actions(surfaces, context),
            },
        )

    @app.get("/runs", response_class=HTMLResponse, include_in_schema=False)
    async def runs(request: Request) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="runs.html",
            context={**base_context(request), "runs": repository.recent_runs()},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    async def run_detail(request: Request, run_id: str) -> Any:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            return templates.TemplateResponse(
                request=request,
                name="not_found.html",
                context=base_context(request),
                status_code=404,
            )
        snapshot = repository.snapshot(run_id)
        events = repository.events(run_id)
        metric_series: dict[str, list[tuple[str, Any]]] = {}
        for indexed in events:
            for key, value in indexed.public_dict()["metrics"].items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (not isinstance(value, float) or math.isfinite(value))
                ):
                    metric_series.setdefault(key, []).append((str(indexed.event_id), value))
        return templates.TemplateResponse(
            request=request,
            name="run_detail.html",
            context={
                **base_context(request),
                "run": snapshot,
                "metric_series": metric_series,
                "log_text": repository.log_text(run_id),
            },
        )

    @app.get("/runs/{run_id}/logs", response_class=HTMLResponse, include_in_schema=False)
    async def run_logs(request: Request, run_id: str) -> Any:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            return templates.TemplateResponse(
                request=request, name="not_found.html", context=base_context(request), status_code=404
            )
        return templates.TemplateResponse(
            request=request,
            name="logs.html",
            context={
                **base_context(request),
                "run": repository.snapshot(run_id),
                "log_text": repository.log_text(run_id),
            },
        )

    @app.get("/runs/{run_id}/report", include_in_schema=False)
    async def run_report(request: Request, run_id: str) -> Any:
        path = repository.report_path(run_id)
        if path is None:
            return templates.TemplateResponse(
                request=request, name="not_found.html", context=base_context(request), status_code=404
            )
        return FileResponse(path, media_type="text/html", filename=f"{run_id}-report.html")

    @app.post("/runs/{run_id}/actions/{action}", include_in_schema=False)
    @product_api
    async def run_action(request: Request, run_id: str, action: str) -> Any:
        if not RUN_ID_PATTERN.fullmatch(run_id) or action not in {"pause", "cancel", "resume"}:
            return api_error(404, "run_action_not_found", "The requested run action is not available.")
        state = repository.state(run_id)
        feature = str(state.get("feature") or state.get("command") or "")
        handler = app.state.spritelab_run_action_handlers.get(feature)
        if handler is None:
            return api_error(
                409,
                "run_action_unsupported",
                f"{action.title()} is not supported by the registered backend for this run.",
                recoverable=False,
                next_action="Open the run report or logs for the available next steps.",
            )
        payload: dict[str, Any] = {}
        if request.headers.get("content-type", "").split(";", 1)[0].strip().casefold() == "application/json":
            raw = await request.json()
            payload = raw if isinstance(raw, dict) else {}
        result = handler(action, run_id, payload)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ProductResult):
            return api_error(500, "invalid_backend_result", "The backend returned an invalid action result.")
        if result.status in {ProductStatus.BLOCKED, ProductStatus.UNAVAILABLE, ProductStatus.FAILED}:
            return api_error(
                409,
                "run_action_unsupported" if result.status == ProductStatus.UNAVAILABLE else "run_action_blocked",
                result.message,
                recoverable=result.status != ProductStatus.UNAVAILABLE,
                next_action="Review the run status and backend capability, then try again if it becomes safe.",
            )
        return JSONResponse(result.to_dict())

    @app.get("/api/current-run", include_in_schema=False)
    @product_api
    async def current_run_api() -> Any:
        current = repository.current_run()
        return {"run": current.to_dict() if current else None}

    @app.get("/api/runs/{run_id}/events", include_in_schema=False)
    @product_api
    async def run_events(request: Request, run_id: str, after: int = 0, once: bool = False) -> Any:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            return api_error(404, "run_not_found", "Run not found.", recoverable=False, next_action=None)
        header_id = request.headers.get("last-event-id", "")
        try:
            cursor = max(after, int(header_id)) if header_id else max(0, after)
        except ValueError:
            cursor = max(0, after)
        reconnecting = cursor > 0

        async def generate() -> Any:
            nonlocal cursor
            yield "retry: 1500\n\n"
            idle = 0
            warning_sent = False
            while True:
                replay = repository.replay(run_id, after_id=cursor)
                items = list(replay.events)
                for item in items:
                    cursor = item.event_id
                    yield _sse(item.event_id, "product", item.public_dict())
                if replay.invalid_event_count and not warning_sent:
                    warning_sent = True
                    yield _sse(
                        None,
                        "warning",
                        {
                            "code": "invalid_product_events",
                            "invalid_event_count": replay.invalid_event_count,
                            "message": replay.warnings[0],
                        },
                    )
                snapshot = repository.snapshot(run_id)
                if once or (snapshot.terminal and not items):
                    snapshot_payload = snapshot.to_dict()
                    if reconnecting:
                        snapshot_payload["recent_messages"] = []
                        snapshot_payload["timeline"] = []
                    yield _sse(cursor, "snapshot", snapshot_payload)
                    break
                if await request.is_disconnected():
                    break
                idle += 1
                if idle % 15 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(max(0.05, event_poll_interval))

        return StreamingResponse(generate(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/logs", include_in_schema=False)
    @product_api
    async def run_log_events(request: Request, run_id: str, after: int = 0, once: bool = False) -> Any:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            return api_error(404, "run_not_found", "Run not found.", recoverable=False, next_action=None)

        async def generate() -> Any:
            cursor = max(0, after)
            while True:
                lines = repository.log_text(run_id).splitlines()
                for line_number, line in enumerate(lines[cursor:], start=cursor + 1):
                    yield _sse(line_number, "log", {"line": line})
                    cursor = line_number
                if once or repository.snapshot(run_id).terminal or await request.is_disconnected():
                    break
                await asyncio.sleep(max(0.05, event_poll_interval))

        return StreamingResponse(generate(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/technical-details", response_class=HTMLResponse, include_in_schema=False)
    async def technical_details(request: Request) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="technical_details.html",
            context={
                **base_context(request),
                "stack": ("FastAPI", "Jinja templates", "Server-sent events", "Local static assets"),
                "plugin_titles": [surface.plugin.title for surface in surfaces],
            },
        )

    for surface in surfaces:
        plugin = surface.plugin
        if surface.missing_capabilities or plugin.web_router_factory is None:
            continue
        router = plugin.web_router_factory(context)
        action_handler = getattr(router, "spritelab_run_action_handler", None)
        action_feature = str(getattr(router, "spritelab_run_action_feature", "") or "")
        if callable(action_handler) and action_feature:
            app.state.spritelab_run_action_handlers[action_feature] = action_handler
        app.include_router(router)
        surface.mounted = True
        for bundle_index, bundle in enumerate(plugin.web_assets):
            static_directory = _package_child(bundle.package, bundle.static)
            template_directory = _package_child(bundle.package, bundle.templates)
            if static_directory:
                route = f"/plugins/{plugin.plugin_id}/static"
                if not any(getattr(item, "path", None) == route for item in app.routes):
                    app.mount(
                        route,
                        StaticFiles(directory=str(static_directory)),
                        name=f"plugin-static-{plugin.plugin_id}-{bundle_index}",
                    )
            if template_directory:
                app.state.__dict__.setdefault("spritelab_plugin_templates", {})[plugin.plugin_id] = template_directory

    async def feature_page(request: Request, area: str) -> Any:
        surface = _surface_for_area(surfaces, area)
        card = _area_card(area, surface, context)
        return templates.TemplateResponse(
            request=request,
            name="feature.html",
            context={**base_context(request), "feature": card, "surface": surface},
            status_code=200,
        )

    for area in AREA_TITLES:

        async def fallback(request: Request, area_name: str = area) -> Any:
            return await feature_page(request, area_name)

        app.add_api_route(f"/{area}", fallback, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)

    core_paths = {item.path for item in CORE_NAVIGATION} | {"/technical-details"}
    for item in navigation:
        if item.path in core_paths:
            continue

        async def plugin_fallback(request: Request, navigation_item: WebNavigationItem = item) -> Any:
            return templates.TemplateResponse(
                request=request,
                name="feature.html",
                context={
                    **base_context(request),
                    "feature": {
                        "title": navigation_item.title,
                        "status": ProductStatus.UNAVAILABLE.value,
                        "status_label": "Not available yet",
                        "message": "The feature route is not registered yet.",
                        "path": navigation_item.path,
                        "value": None,
                    },
                    "surface": None,
                },
            )

        app.add_api_route(
            item.path, plugin_fallback, methods=["GET"], response_class=HTMLResponse, include_in_schema=False
        )

    return app


__all__ = ["create_app"]

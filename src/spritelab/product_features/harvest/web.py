"""FastAPI surface for controlled, durable Harvest acquisition."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from starlette.requests import Request

from spritelab.product_core import ProjectContext, api_error, product_api, strict_json_dumps
from spritelab.product_features.harvest.service import HarvestError, HarvestService

_START_KEYS = frozenset(
    {
        "source_id",
        "idempotency_key",
        "explicit_action",
        "authorize_zero_cost",
        "authorize_permissive_license",
        "authorize_existing_inventory_reviewed",
        "reuse_evidence",
    }
)
_RETRY_KEYS = frozenset(
    {
        "idempotency_key",
        "explicit_action",
        "authorize_zero_cost",
        "authorize_permissive_license",
        "authorize_existing_inventory_reviewed",
        "reuse_evidence",
    }
)
_CANCEL_KEYS = frozenset({"explicit_action"})
_IMPORT_KEYS = frozenset({"explicit_action", "idempotency_key"})
_PROBE_KEYS = frozenset(
    {
        "source_id",
        "title",
        "creator",
        "source_page",
        "license_id",
        "license_evidence_url",
        "terms_evidence_url",
        "direct_download_url",
        "attribution_text",
        "taxonomy_hints",
        "inventory_identity",
        "backend_capability_evidence_identity",
        "idempotency_key",
        "explicit_action",
        "authorize_network",
        "authorize_hash_probe",
        "authorize_zero_cost",
        "authorize_permissive_license",
    }
)
_PROBE_URL_KEYS = frozenset({"source_page", "license_evidence_url", "terms_evidence_url", "direct_download_url"})
_PROMOTE_KEYS = frozenset({"explicit_action", "authorize_catalog_promotion"})
_PATH_KEY_FRAGMENTS = ("path", "directory", "folder", "output", "destination", "uri", "argv", "command")


def _resource_text(name: str) -> str:
    return files("spritelab.product_features.harvest").joinpath(name).read_text(encoding="utf-8")


def create_harvest_router(
    context: ProjectContext,
    *,
    service: HarvestService | None = None,
) -> Any:
    """Create a router whose GET endpoints remain passive and network-free."""

    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    router = APIRouter()
    harvest = service or HarvestService(context.project_root)

    @router.get("/harvest", response_class=HTMLResponse)
    def harvest_page(request: Request) -> Any:
        initial = {
            "inventory": harvest.inventory(),
            "catalog": harvest.sources(),
        }
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(
                request,
                "harvest.acquisition",
                "harvest.html",
                {"harvest_initial_state": initial},
            )
        standalone = _resource_text("templates/harvest_standalone.html")
        return standalone.replace(
            "__HARVEST_INITIAL_STATE__",
            strict_json_dumps(initial, ensure_ascii=False).replace("<", "\\u003c"),
        )

    @router.get("/harvest/static/harvest.css")
    def harvest_css() -> Response:
        return Response(_resource_text("static/harvest.css"), media_type="text/css")

    @router.get("/harvest/static/harvest.js")
    def harvest_js() -> Response:
        return Response(_resource_text("static/harvest.js"), media_type="application/javascript")

    @router.get("/harvest/api/inventory")
    @product_api
    def inventory() -> Any:
        return _call(harvest.inventory)

    @router.get("/harvest/api/sources")
    @product_api
    def sources() -> dict[str, Any]:
        return harvest.sources()

    @router.post("/harvest/api/jobs")
    @product_api
    def start_job(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _START_KEYS)
        if rejected is not None:
            return rejected
        try:
            job, created = harvest.start(
                _required_string(payload, "source_id"),
                idempotency_key=_required_string(payload, "idempotency_key"),
                explicit_action=payload.get("explicit_action") is True,
                authorize_zero_cost=payload.get("authorize_zero_cost") is True,
                authorize_permissive_license=payload.get("authorize_permissive_license") is True,
                authorize_existing_inventory_reviewed=payload.get("authorize_existing_inventory_reviewed") is True,
                reuse_evidence=payload.get("reuse_evidence"),
            )
        except HarvestError as exc:
            return _harvest_error(exc)
        except ValueError:
            return _invalid_payload()
        return JSONResponse(
            status_code=202 if created else 200,
            content={"created": created, "job": job},
        )

    @router.get("/harvest/api/jobs/{run_id}")
    @product_api
    def job_status(run_id: str) -> Any:
        return _call(lambda: harvest.job(run_id))

    @router.post("/harvest/api/jobs/{run_id}/retry")
    @product_api
    def retry_job(run_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _RETRY_KEYS)
        if rejected is not None:
            return rejected
        try:
            job, created = harvest.retry(
                run_id,
                idempotency_key=_required_string(payload, "idempotency_key"),
                explicit_action=payload.get("explicit_action") is True,
                authorize_zero_cost=payload.get("authorize_zero_cost") is True,
                authorize_permissive_license=payload.get("authorize_permissive_license") is True,
                authorize_existing_inventory_reviewed=payload.get("authorize_existing_inventory_reviewed") is True,
                reuse_evidence=payload.get("reuse_evidence"),
            )
        except HarvestError as exc:
            return _harvest_error(exc)
        except ValueError:
            return _invalid_payload()
        return JSONResponse(
            status_code=202 if created else 200,
            content={"created": created, "job": job},
        )

    @router.post("/harvest/api/jobs/{run_id}/cancel")
    @product_api
    def cancel_job(run_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _CANCEL_KEYS)
        if rejected is not None:
            return rejected
        return _call(lambda: harvest.cancel(run_id, explicit_action=payload.get("explicit_action") is True))

    @router.get("/harvest/api/jobs/{run_id}/handoff")
    @product_api
    def dataset_handoff(run_id: str) -> Any:
        return _call(lambda: harvest.handoff(run_id))

    @router.get("/harvest/api/jobs/{run_id}/evidence")
    @product_api
    def durable_evidence(run_id: str) -> Any:
        return _call(lambda: harvest.evidence(run_id))

    @router.post("/harvest/api/jobs/{run_id}/import")
    @product_api
    def import_to_dataset(run_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _IMPORT_KEYS)
        if rejected is not None:
            return rejected
        try:
            return harvest.import_to_dataset(
                run_id,
                explicit_action=payload.get("explicit_action") is True,
                idempotency_key=_required_string(payload, "idempotency_key"),
            )
        except HarvestError as exc:
            return _harvest_error(exc)
        except ValueError:
            return _invalid_payload()

    @router.post("/harvest/api/probes")
    @product_api
    def start_probe(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PROBE_KEYS, allowed_url_keys=_PROBE_URL_KEYS)
        if rejected is not None:
            return rejected
        try:
            probe, created = harvest.start_probe(payload)
        except HarvestError as exc:
            return _harvest_error(exc)
        except ValueError:
            return _invalid_payload()
        return JSONResponse(status_code=202 if created else 200, content={"created": created, "probe": probe})

    @router.get("/harvest/api/probes/{probe_id}")
    @product_api
    def probe_status(probe_id: str) -> Any:
        return _call(lambda: harvest.probe(probe_id))

    @router.get("/harvest/api/probes/{probe_id}/evidence")
    @product_api
    def probe_evidence(probe_id: str) -> Any:
        return _call(lambda: harvest.probe_evidence(probe_id))

    @router.post("/harvest/api/probes/{probe_id}/cancel")
    @product_api
    def cancel_probe(probe_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _CANCEL_KEYS)
        if rejected is not None:
            return rejected
        return _call(lambda: harvest.cancel_probe(probe_id, explicit_action=payload.get("explicit_action") is True))

    @router.post("/harvest/api/probes/{probe_id}/retry")
    @product_api
    def retry_probe(probe_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PROBE_KEYS, allowed_url_keys=_PROBE_URL_KEYS)
        if rejected is not None:
            return rejected
        try:
            probe, created = harvest.retry_probe(probe_id, payload)
        except HarvestError as exc:
            return _harvest_error(exc)
        except ValueError:
            return _invalid_payload()
        return JSONResponse(status_code=202 if created else 200, content={"created": created, "probe": probe})

    @router.post("/harvest/api/probes/{probe_id}/promote")
    @product_api
    def promote_probe(probe_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PROMOTE_KEYS)
        if rejected is not None:
            return rejected
        return _call(
            lambda: harvest.promote_probe(
                probe_id,
                explicit_action=payload.get("explicit_action") is True,
                authorize_catalog_promotion=payload.get("authorize_catalog_promotion") is True,
            )
        )

    return router


def _call(function: Any) -> Any:
    try:
        return function()
    except HarvestError as exc:
        return _harvest_error(exc)


def _harvest_error(exc: HarvestError) -> Any:
    return api_error(
        exc.status_code,
        exc.code,
        str(exc),
        recoverable=exc.status_code < 500,
        next_action="Review the Harvest source and authorization, then try again.",
    )


def _validate_payload(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    allowed_url_keys: frozenset[str] = frozenset(),
) -> Any | None:
    keys = set(payload)
    forbidden_path_key = next(
        (
            key
            for key in keys
            if key not in allowed_url_keys
            and ("url" in key.casefold() or any(fragment in key.casefold() for fragment in _PATH_KEY_FRAGMENTS))
        ),
        None,
    )
    if forbidden_path_key is not None:
        return api_error(
            422,
            "browser_path_not_allowed",
            "Harvest uses managed repository-local output; filesystem paths, commands, and unrecognized URLs are forbidden.",
            next_action="Use only the explicit evidence URL fields shown by catalog onboarding.",
        )
    if keys - allowed:
        return _invalid_payload()
    return None


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _invalid_payload() -> Any:
    return api_error(
        422,
        "invalid_harvest_payload",
        "Harvest request fields are missing or not recognized.",
        next_action="Use only the fields shown by the Harvest form.",
    )


__all__ = ["create_harvest_router"]

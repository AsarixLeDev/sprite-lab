"""Web surface for the conditioned Dataset-v5 workflow."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from starlette.requests import Request

from spritelab.product_core import ProjectContext, api_error, product_api, strict_json_dumps
from spritelab.product_features.conditioned_v5.service import (
    ConditionedDatasetError,
    ConditionedDatasetService,
)

_PREVIEW_KEYS = frozenset({"dataset_references"})
_BUILD_KEYS = frozenset({"dataset_references", "idempotency_key", "explicit_action"})
_CANCEL_KEYS = frozenset({"explicit_action"})
_EVIDENCE_KEYS = frozenset({"kind", "document"})
_PUBLISH_KEYS = frozenset(
    {
        "candidate_identity",
        "label_audit_sha256",
        "dataset_validation_sha256",
        "authorization_id",
        "explicit_action",
        "authorize_one_time_freeze",
    }
)
_PATH_FRAGMENTS = ("path", "directory", "folder", "output", "destination", "url", "uri")


def _resource_text(name: str) -> str:
    return files("spritelab.product_features.conditioned_v5").joinpath(name).read_text(encoding="utf-8")


def create_router(
    context: ProjectContext,
    *,
    service: ConditionedDatasetService | None = None,
) -> Any:
    """Create the local, CSRF-protected JSON and page routes."""

    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    router = APIRouter()
    conditioned = service or ConditionedDatasetService(context.project_root)

    @router.get("/dataset-v5", response_class=HTMLResponse)
    def page(request: Request) -> Any:
        initial = conditioned.inventory()
        renderer = getattr(request.app.state, "spritelab_render_plugin_template", None)
        if callable(renderer):
            return renderer(
                request,
                "dataset.conditioned_v5",
                "conditioned_v5.html",
                {"conditioned_initial_state": initial},
            )
        standalone = _resource_text("templates/conditioned_v5_standalone.html")
        return standalone.replace(
            "__CONDITIONED_INITIAL_STATE__",
            strict_json_dumps(initial, ensure_ascii=False).replace("<", "\\u003c"),
        )

    @router.get("/dataset-v5/static/conditioned-v5.css")
    def css() -> Response:
        return Response(_resource_text("static/conditioned-v5.css"), media_type="text/css")

    @router.get("/dataset-v5/static/conditioned-v5.js")
    def javascript() -> Response:
        return Response(_resource_text("static/conditioned-v5.js"), media_type="application/javascript")

    @router.get("/dataset-v5/api/inventory")
    @product_api
    def inventory() -> dict[str, Any]:
        return conditioned.inventory()

    @router.post("/dataset-v5/api/preview")
    @product_api
    def preview(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PREVIEW_KEYS)
        if rejected is not None:
            return rejected
        return _call(lambda: conditioned.preview(_dataset_references(payload)))

    @router.post("/dataset-v5/api/jobs")
    @product_api
    def start(payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _BUILD_KEYS)
        if rejected is not None:
            return rejected
        try:
            job, created = conditioned.start_build(
                _dataset_references(payload),
                idempotency_key=_required_string(payload, "idempotency_key"),
                explicit_action=payload.get("explicit_action") is True,
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()
        return JSONResponse(status_code=202 if created else 200, content={"created": created, "job": job})

    @router.get("/dataset-v5/api/jobs/{job_id}")
    @product_api
    def job(job_id: str) -> Any:
        return _call(lambda: conditioned.job(job_id))

    @router.post("/dataset-v5/api/jobs/{job_id}/cancel")
    @product_api
    def cancel(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _CANCEL_KEYS)
        if rejected is not None:
            return rejected
        return _call(lambda: conditioned.cancel(job_id, explicit_action=payload.get("explicit_action") is True))

    @router.post("/dataset-v5/api/jobs/{job_id}/evidence")
    @product_api
    def evidence(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _EVIDENCE_KEYS, reject_path_keys=False)
        if rejected is not None:
            return rejected
        kind = payload.get("kind")
        document = payload.get("document")
        if not isinstance(kind, str) or not isinstance(document, dict):
            return _invalid()
        return _call(lambda: conditioned.attach_evidence(job_id, kind=kind, document=document))

    @router.post("/dataset-v5/api/jobs/{job_id}/publish")
    @product_api
    def publish(job_id: str, payload: dict[str, Any]) -> Any:
        rejected = _validate_payload(payload, _PUBLISH_KEYS)
        if rejected is not None:
            return rejected
        try:
            return conditioned.publish(
                job_id,
                candidate_identity=_required_string(payload, "candidate_identity"),
                label_audit_sha256=_required_string(payload, "label_audit_sha256"),
                dataset_validation_sha256=_required_string(payload, "dataset_validation_sha256"),
                authorization_id=_required_string(payload, "authorization_id"),
                explicit_action=payload.get("explicit_action") is True,
                authorize_one_time_freeze=payload.get("authorize_one_time_freeze") is True,
            )
        except ConditionedDatasetError as exc:
            return _error(exc)
        except (TypeError, ValueError):
            return _invalid()

    return router


def _call(function: Any) -> Any:
    try:
        return function()
    except ConditionedDatasetError as exc:
        return _error(exc)


def _error(exc: ConditionedDatasetError) -> Any:
    return api_error(
        exc.status_code,
        exc.code,
        exc.public_message,
        recoverable=exc.status_code < 500,
        next_action="Review the managed handoffs, candidate evidence, and exact authorization before retrying.",
    )


def _validate_payload(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    reject_path_keys: bool = True,
) -> Any | None:
    if set(payload) - allowed:
        return _invalid()
    if reject_path_keys and any(fragment in str(key).casefold() for key in payload for fragment in _PATH_FRAGMENTS):
        return api_error(
            422,
            "browser_path_not_allowed",
            "Dataset-v5 uses managed repository-local artifacts; browser paths and URLs are not accepted.",
        )
    return None


def _dataset_references(payload: dict[str, Any]) -> list[str]:
    value = payload.get("dataset_references")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ConditionedDatasetError(
            "managed_intake_selection", "Select completed managed Dataset imports.", status_code=422
        )
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{key} is required")
    return value


def _invalid() -> Any:
    return api_error(
        422,
        "invalid_conditioned_v5_payload",
        "The Dataset-v5 request fields are missing or not recognized.",
    )


__all__ = ["create_router"]

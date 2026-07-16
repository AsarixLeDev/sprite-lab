"""Shared classification and safe error responses for product APIs."""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from spritelab.product_core.events import StrictJSONError, strict_json_dumps

API_ERROR_SCHEMA = "spritelab.product.api-error.v1"
DEFAULT_API_PREFIXES = (
    "/api",
    "/dataset/api",
    "/training/api",
    "/evaluation/api",
    "/settings/api",
    "/settings/vision/api",
    "/providers/api",
)
RUN_ACTION_PATTERN = re.compile(r"^/runs/[A-Za-z0-9][A-Za-z0-9._-]{0,159}/actions/[a-z-]+$")

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def product_api(function: _Callable) -> _Callable:
    """Mark an endpoint as a product API even when its path is unconventional."""

    function.__spritelab_product_api__ = True
    return function


def request_is_api(request: Any) -> bool:
    """Classify API requests using route metadata and registered feature prefixes."""

    route = request.scope.get("route") if hasattr(request, "scope") else None
    endpoint = getattr(route, "endpoint", None)
    if bool(getattr(endpoint, "__spritelab_product_api__", False)):
        return True
    path = str(request.url.path)
    prefixes = getattr(getattr(request.app, "state", None), "spritelab_api_prefixes", DEFAULT_API_PREFIXES)
    for prefix in prefixes:
        normalized = str(prefix).rstrip("/") or "/"
        if path == normalized or path.startswith(normalized + "/"):
            return True
    # Plugin APIs conventionally contain an /api/ segment. Prefix metadata is
    # still preferred because it also covers unconventional plugin routes.
    if "/api/" in path or path.endswith("/api"):
        return True
    if RUN_ACTION_PATTERN.fullmatch(path):
        return True
    accepted = request.headers.get("accept", "") if hasattr(request, "headers") else ""
    return "application/json" in accepted.casefold()


def api_error_payload(
    status_code: int,
    error_code: str,
    message: str,
    *,
    recoverable: bool,
    next_action: str | None,
    error_reference: str | None = None,
    details: Mapping[str, Any] | None = None,
    include_details: bool = False,
) -> dict[str, Any]:
    """Build the single allowlisted product API error envelope."""

    payload: dict[str, Any] = {
        "schema_version": API_ERROR_SCHEMA,
        "status": int(status_code),
        "error_code": str(error_code),
        "message": str(message),
        "error_reference": error_reference or f"ERR-{secrets.token_hex(5).upper()}",
        "recoverable": bool(recoverable),
        "next_action": str(next_action) if next_action else None,
    }
    if include_details and details:
        candidate = dict(details)
        try:
            strict_json_dumps(candidate)
        except StrictJSONError:
            pass
        else:
            payload["details"] = candidate
    return payload


def api_error(
    status_code: int,
    error_code: str,
    message: str,
    *,
    recoverable: bool = True,
    next_action: str | None = "Try again",
    error_reference: str | None = None,
    details: Mapping[str, Any] | None = None,
    include_details: bool = False,
) -> Any:
    """Return a JSONResponse without importing FastAPI in non-web callers."""

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content=api_error_payload(
            status_code,
            error_code,
            message,
            recoverable=recoverable,
            next_action=next_action,
            error_reference=error_reference,
            details=details,
            include_details=include_details,
        ),
    )


__all__ = [
    "API_ERROR_SCHEMA",
    "DEFAULT_API_PREFIXES",
    "api_error",
    "api_error_payload",
    "product_api",
    "request_is_api",
]

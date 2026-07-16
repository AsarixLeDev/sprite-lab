"""Small injectable HTTP transport used by all built-in network adapters."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol

from spritelab.product_features.providers.errors import (
    ProviderCancelledError,
    ProviderError,
    ProviderTimeoutError,
)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def json_text(self) -> str:
        return self.body.decode("utf-8")


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout: float,
        cancel_event: Event | None = None,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    """Stdlib transport with bounded timeouts and cooperative cancellation."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout: float,
        cancel_event: Event | None = None,
    ) -> HttpResponse:
        if cancel_event and cancel_event.is_set():
            raise ProviderCancelledError()
        request = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                result = HttpResponse(
                    status=int(getattr(response, "status", response.getcode())),
                    body=response.read(),
                    headers={str(key): str(value) for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            result = HttpResponse(
                status=int(exc.code),
                body=exc.read() if exc.fp is not None else b"",
                headers={str(key): str(value) for key, value in (exc.headers.items() if exc.headers else ())},
            )
        except TimeoutError as exc:
            raise ProviderTimeoutError() from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeoutError() from exc
            raise ProviderError(
                "provider_unavailable", "The provider endpoint is unavailable.", retryable=True
            ) from exc
        except OSError as exc:
            raise ProviderError(
                "provider_unavailable", "The provider endpoint is unavailable.", retryable=True
            ) from exc
        if cancel_event and cancel_event.is_set():
            raise ProviderCancelledError()
        return result


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def normalized_http_error(response: HttpResponse) -> ProviderError:
    status = int(response.status)
    if status in {401, 403}:
        return ProviderError(
            "provider_authentication_required", "Provider authentication is required.", status_code=status
        )
    if status == 404:
        return ProviderError(
            "provider_endpoint_not_found", "The configured provider endpoint was not found.", status_code=404
        )
    if status == 408:
        return ProviderTimeoutError()
    if status == 429:
        return ProviderError(
            "provider_rate_limited",
            "The provider rate limit was reached.",
            retryable=True,
            status_code=429,
            retry_after=retry_after_seconds(response.headers),
        )
    if status >= 500:
        return ProviderError(
            "provider_server_error", "The provider reported a server error.", retryable=True, status_code=status
        )
    return ProviderError("provider_request_rejected", "The provider rejected the request.", status_code=status)

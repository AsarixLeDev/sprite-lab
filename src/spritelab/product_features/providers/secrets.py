"""Environment-only credential resolution and log-safe redaction."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie"})
_SENSITIVE_QUERY_MARKERS = ("key", "token", "secret", "password", "signature", "credential")


def resolve_credential(reference: str | None, *, environ: Mapping[str, str] | None = None) -> str | None:
    if not reference:
        return None
    if not _ENV_NAME.fullmatch(reference):
        raise ValueError("credential_env must be an environment-variable name")
    source = environ if environ is not None else os.environ
    return source.get(reference)


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name): "<redacted>" if str(name).lower() in _SENSITIVE_HEADERS else str(value)
        for name, value in headers.items()
    }


def redact_url(url: str) -> str:
    parts = urlsplit(str(url))
    safe_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        safe_query.append(
            (key, "<redacted>" if any(marker in lowered for marker in _SENSITIVE_QUERY_MARKERS) else value)
        )
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, urlencode(safe_query), parts.fragment))


def redact_text(value: str, credentials: tuple[str, ...] = ()) -> str:
    redacted = str(value)
    for credential in credentials:
        if credential:
            redacted = redacted.replace(credential, "<redacted>")
    redacted = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1<redacted>", redacted)
    return redacted

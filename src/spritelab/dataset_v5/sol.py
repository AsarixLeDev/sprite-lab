"""Strict GPT-5.6 Sol transport for the raw Dataset-v5 rebuild.

There are intentionally no model defaults, fallbacks, response repairs, or
legacy label-v4 provider adapters in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from spritelab.dataset_v5.blind import (
    BLIND_OUTPUT_SCHEMA_VERSION,
    BlindInput,
    blind_cache_key,
    build_blind_request,
)
from spritelab.dataset_v5.identity import canonical_json_bytes

SOL_MODEL_UNAVAILABLE = "SOL_MODEL_UNAVAILABLE"
SOL_PROVIDER_SCHEMA_VERSION = "sol_provider_transport_v1"
SOL_MODEL_FAMILY = "GPT-5.6 Sol"
SOL_MODEL_ID_PATTERN = re.compile(r"^gpt-5\.6-sol(?:-[0-9]{4}-[0-9]{2}-[0-9]{2})?$")
SUPPORTED_BACKENDS = frozenset({"openai_responses_v1", "openai_compatible_responses_v1"})


class SolModelUnavailable(RuntimeError):
    def __init__(self, details: str = "") -> None:
        self.details = details
        super().__init__(SOL_MODEL_UNAVAILABLE)


class SolResponseInvalid(ValueError):
    """The provider answered, but the response is not authoritative."""


@dataclass(frozen=True)
class SolConfig:
    backend: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SolConfig:
        values = os.environ if environ is None else environ
        required = {
            "backend": values.get("SPRITELAB_SOL_BACKEND", "").strip(),
            "model": values.get("SPRITELAB_SOL_MODEL", "").strip(),
            "base_url": values.get("SPRITELAB_SOL_BASE_URL", "").strip(),
            "api_key": values.get("SPRITELAB_SOL_API_KEY", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SolModelUnavailable("missing explicit Sol settings: " + ", ".join(sorted(missing)))
        config = cls(**required)
        config.validate()
        return config

    def validate(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise SolModelUnavailable(f"unsupported Sol backend {self.backend!r}")
        if not SOL_MODEL_ID_PATTERN.fullmatch(self.model):
            raise SolModelUnavailable(f"configured model is not an exact GPT-5.6 Sol identifier: {self.model!r}")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password:
            raise SolModelUnavailable("invalid Sol base URL")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise SolModelUnavailable("non-local Sol endpoint must use HTTPS")
        if not self.api_key:
            raise SolModelUnavailable("missing Sol API key")

    @property
    def endpoint_identity(self) -> str:
        parsed = urllib.parse.urlsplit(self.base_url)
        path = parsed.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))

    def public_identity(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "configured_model_identifier": self.model,
            "endpoint_identity": self.endpoint_identity,
            "provider_schema_version": SOL_PROVIDER_SCHEMA_VERSION,
        }


@dataclass(frozen=True)
class SolEndpointIdentity:
    backend: str
    endpoint_identity: str
    model_identifier: str
    model_family: str
    model_version: str
    provider: str
    request_schema_version: str

    def canonical(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "endpoint_identity": self.endpoint_identity,
            "model_family": self.model_family,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
            "provider": self.provider,
            "request_schema_version": self.request_schema_version,
        }


Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], Mapping[str, Any]]


class SolProvider:
    def __init__(self, config: SolConfig, *, transport: Transport | None = None) -> None:
        config.validate()
        self.config = config
        self._transport = transport or _urllib_transport
        self._identity: SolEndpointIdentity | None = None

    def preflight(self) -> SolEndpointIdentity:
        """Verify product identity; a model-name string alone is insufficient."""

        url = self.config.base_url.rstrip("/") + "/models/" + urllib.parse.quote(self.config.model, safe="")
        try:
            response = self._transport("GET", url, self._headers(), None, self.config.timeout_seconds)
        except Exception as exc:
            raise SolModelUnavailable(f"Sol model preflight failed: {type(exc).__name__}") from exc
        if int(response.get("status", 0)) != 200:
            raise SolModelUnavailable(f"Sol model preflight status {response.get('status')}")
        body = response.get("json")
        if not isinstance(body, Mapping):
            raise SolModelUnavailable("Sol model preflight returned no identity object")
        model_identifier = str(body.get("id") or "")
        family = str(body.get("family") or body.get("model_family") or "")
        version = str(body.get("version") or body.get("model_version") or "")
        provider = str(body.get("provider") or body.get("owned_by") or "")
        schema_version = str(body.get("request_schema_version") or "")
        if model_identifier != self.config.model:
            raise SolModelUnavailable("provider model identifier differs from configured exact identifier")
        if family.casefold() != SOL_MODEL_FAMILY.casefold():
            raise SolModelUnavailable("provider did not attest the GPT-5.6 Sol model family")
        if not version or not provider or not schema_version:
            raise SolModelUnavailable("provider model version/provider/request schema identity is incomplete")
        if provider.casefold() != "openai":
            raise SolModelUnavailable("provider identity is not exactly OpenAI")
        _assert_final_url(response, url)
        identity = SolEndpointIdentity(
            backend=self.config.backend,
            endpoint_identity=self.config.endpoint_identity,
            model_identifier=model_identifier,
            model_family=family,
            model_version=version,
            provider=provider,
            request_schema_version=schema_version,
        )
        self._identity = identity
        return identity

    @property
    def identity(self) -> SolEndpointIdentity:
        return self._identity or self.preflight()

    def call_blind_pass(
        self,
        blind_input: BlindInput,
        *,
        request_id: str,
        pass_kind: str,
    ) -> dict[str, Any]:
        identity = self.identity
        payload = build_blind_request(
            blind_input,
            model=self.config.model,
            request_id=request_id,
            pass_kind=pass_kind,
        )
        request_bytes = canonical_json_bytes(payload)
        url = self.config.base_url.rstrip("/") + "/responses"
        started = time.perf_counter()
        try:
            response = self._transport(
                "POST",
                url,
                self._headers(),
                request_bytes,
                self.config.timeout_seconds,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._failure_artifact(
                identity,
                payload,
                request_bytes,
                pass_kind,
                latency_ms,
                "transport_error",
                type(exc).__name__,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        _assert_final_url(response, url)
        status = int(response.get("status", 0))
        body = response.get("json")
        raw_response = response.get("body")
        raw_response_bytes = (
            bytes(raw_response)
            if isinstance(raw_response, (bytes, bytearray))
            else canonical_json_bytes(body)
            if isinstance(body, Mapping)
            else str(raw_response or "").encode("utf-8", errors="replace")
        )
        base = self._artifact_base(identity, payload, request_bytes, pass_kind, latency_ms)
        base.update(
            {
                "http_status": status,
                "raw_provider_response_sha256": hashlib.sha256(raw_response_bytes).hexdigest(),
            }
        )
        if status != 200 or not isinstance(body, Mapping):
            return {
                **base,
                "authoritative": False,
                "error_code": "provider_http_or_body_invalid",
                "raw_provider_response": raw_response_bytes.decode("utf-8", errors="replace"),
                "status": "invalid",
            }
        try:
            parsed, raw_output = _parse_provider_body(body, identity)
            validate_sol_output(parsed, payload)
        except (SolModelUnavailable, SolResponseInvalid) as exc:
            if isinstance(exc, SolModelUnavailable):
                raise
            return {
                **base,
                "authoritative": False,
                "error_code": "strict_response_rejected",
                "error_detail": str(exc),
                "raw_provider_response": raw_response_bytes.decode("utf-8", errors="replace"),
                "status": "invalid",
            }
        usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
        return {
            **base,
            "authoritative": True,
            "error_code": None,
            "output": parsed,
            "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            "status": "success",
            "token_usage": {
                "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
                "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + self.config.api_key,
            "Content-Type": "application/json",
            "User-Agent": "sprite-lab-raw-v5/1",
        }

    def _artifact_base(
        self,
        identity: SolEndpointIdentity,
        payload: Mapping[str, Any],
        request_bytes: bytes,
        pass_kind: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        return {
            "backend": identity.backend,
            "cache_key": blind_cache_key(
                payload,
                endpoint_identity=identity.endpoint_identity,
                provider=identity.provider,
            ),
            "endpoint_identity": identity.endpoint_identity,
            "http_attempts": 1,
            "latency_ms": round(latency_ms, 3),
            "model_family": identity.model_family,
            "model_identifier": identity.model_identifier,
            "model_version": identity.model_version,
            "pass_kind": pass_kind,
            "prompt_version": str(payload.get("metadata", {}).get("prompt_version") or ""),
            "provider": identity.provider,
            "provider_schema_version": SOL_PROVIDER_SCHEMA_VERSION,
            "request_schema_version": str(payload.get("request_schema_version") or ""),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "response_schema_version": BLIND_OUTPUT_SCHEMA_VERSION,
        }

    def _failure_artifact(
        self,
        identity: SolEndpointIdentity,
        payload: Mapping[str, Any],
        request_bytes: bytes,
        pass_kind: str,
        latency_ms: float,
        code: str,
        detail: str,
    ) -> dict[str, Any]:
        return {
            **self._artifact_base(identity, payload, request_bytes, pass_kind, latency_ms),
            "authoritative": False,
            "error_code": code,
            "error_detail": detail,
            "http_attempts": 1,
            "status": "invalid",
        }


def validate_sol_output(value: Mapping[str, Any], request_payload: Mapping[str, Any]) -> None:
    """Validate exact shape and controlled values without altering the output."""

    schema = request_payload.get("response_format", {}).get("json_schema", {}).get("schema", {})
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    required = schema.get("required", []) if isinstance(schema, Mapping) else []
    if not isinstance(value, Mapping):
        raise SolResponseInvalid("output is not one JSON object")
    if set(value) != set(properties) or any(name not in value for name in required):
        raise SolResponseInvalid("output keys do not exactly match the strict schema")
    if value.get("schema_version") != BLIND_OUTPUT_SCHEMA_VERSION:
        raise SolResponseInvalid("output schema version mismatch")
    taxonomy = _request_taxonomy(request_payload)
    controlled = {
        "category": set(taxonomy.get("categories", [])),
        "domain": set(taxonomy.get("domains", [])),
        "role": set(taxonomy.get("roles", [])),
        "material_applicability": set(taxonomy.get("material_applicability", [])),
    }
    for field, allowed in controlled.items():
        field_value = value.get(field)
        if field_value is not None and field_value not in allowed:
            raise SolResponseInvalid(f"taxonomy violation for {field}: {field_value!r}")
    for field in ("visual_form",):
        if value.get(field) is not None and not _string_list(value[field]):
            raise SolResponseInvalid(f"{field} must be a list of strings or null")
    color_roles = value.get("color_roles")
    if not isinstance(color_roles, Mapping) or set(color_roles) != {
        "primary",
        "secondary",
        "outline",
        "shadow",
        "highlight",
    }:
        raise SolResponseInvalid("color_roles shape is invalid")
    if any(not _string_list(item) for item in color_roles.values()):
        raise SolResponseInvalid("color role values must be string lists")
    for name in ("abstentions", "field_rationales", "field_risk_signals", "field_confidence"):
        if not isinstance(value.get(name), Mapping):
            raise SolResponseInvalid(f"{name} must be an object")
    if any(not isinstance(item, str) for item in value["abstentions"].values()):
        raise SolResponseInvalid("abstention reasons must be strings")
    if any(not isinstance(item, str) for item in value["field_rationales"].values()):
        raise SolResponseInvalid("field rationales must be strings")
    if any(not _string_list(item) for item in value["field_risk_signals"].values()):
        raise SolResponseInvalid("field risk signals must be string lists")
    for confidence in value["field_confidence"].values():
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise SolResponseInvalid("field confidence must be in [0,1]")
    nullable = {
        "category",
        "canonical_object",
        "domain",
        "role",
        "visual_form",
        "explicit_material",
        "description",
    }
    for field in nullable:
        if value.get(field) is None and field not in value["abstentions"]:
            raise SolResponseInvalid(f"null field {field} has no explicit abstention")
    if value.get("material_applicability") != "applicable" and value.get("explicit_material") is not None:
        raise SolResponseInvalid("explicit material provided when material is not applicable")


def _parse_provider_body(body: Mapping[str, Any], identity: SolEndpointIdentity) -> tuple[Mapping[str, Any], str]:
    model = str(body.get("model") or "")
    version = str(body.get("model_version") or "")
    provider = str(body.get("provider") or "")
    if model != identity.model_identifier or version != identity.model_version or provider != identity.provider:
        raise SolModelUnavailable("provider response model identity changed after preflight")
    raw_output = body.get("output_text")
    if not isinstance(raw_output, str):
        try:
            raw_output = body["output"][0]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SolResponseInvalid("provider response has no exact output text") from exc
    if raw_output.lstrip().startswith("```"):
        raise SolResponseInvalid("JSON code fences are forbidden; response was not repaired")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise SolResponseInvalid(f"invalid JSON at byte {exc.pos}; response was not repaired") from exc
    if not isinstance(parsed, Mapping):
        raise SolResponseInvalid("output JSON must be one object")
    return parsed, raw_output


def _request_taxonomy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        text = payload["messages"][1]["content"][0]["text"]
        decoded = json.loads(text)
        taxonomy = decoded["taxonomy"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SolResponseInvalid("request taxonomy is missing") from exc
    if not isinstance(taxonomy, Mapping):
        raise SolResponseInvalid("request taxonomy is invalid")
    return taxonomy


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _assert_final_url(response: Mapping[str, Any], expected_url: str) -> None:
    final_url = response.get("final_url")
    if not isinstance(final_url, str) or not final_url:
        raise SolModelUnavailable("provider transport did not attest its final endpoint")
    expected = urllib.parse.urlsplit(expected_url)
    observed = urllib.parse.urlsplit(final_url)
    expected_identity = urllib.parse.urlunsplit(
        (expected.scheme.lower(), expected.netloc.lower(), expected.path, expected.query, "")
    )
    observed_identity = urllib.parse.urlunsplit(
        (observed.scheme.lower(), observed.netloc.lower(), observed.path, observed.query, "")
    )
    if observed_identity != expected_identity:
        raise SolModelUnavailable("provider transport final endpoint changed")


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
            final_url = response.geturl()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
        final_url = exc.geturl()
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    return {"body": raw, "final_url": final_url, "json": decoded, "status": status}

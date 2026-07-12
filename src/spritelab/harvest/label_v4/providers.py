"""Structured JSON providers for blind proposal, reconciliation, and verification.

Real network use is opt-in through the CLI canary command.  Tests and cohort
experiments use deterministic mock responses and never spend provider credit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.pixel_evidence import exact_rgba_content_hash

PROVIDER_ARTIFACT_SCHEMA_VERSION = "label_provider_artifact_v1.1"
MOCK_JSON_REQUEST_POLICY_VERSION = "mock_json_request_v1.0"
OPENAI_COMPATIBLE_REQUEST_POLICY_VERSION = "openai_compatible_json_request_v1.1"


@dataclass(frozen=True)
class ProviderArtifact:
    stage: str
    raw_output: str
    parsed_output: dict[str, Any] | None
    model_identity: str
    request_hash: str
    image_hash: str
    prompt_version: str
    prompt_hash: str
    latency_ms: float
    http_attempts: int = 0
    request_policy_version: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    cache_namespace: str = ""
    failure_diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PROVIDER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.http_attempts) < 0:
            raise ValueError("http_attempts must be non-negative")

    @property
    def ok(self) -> bool:
        return self.parsed_output is not None and not self.failure_diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "model_identity": self.model_identity,
            "request_hash": self.request_hash,
            "image_hash": self.image_hash,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "latency_ms": self.latency_ms,
            "http_attempts": int(self.http_attempts),
            "request_policy_version": self.request_policy_version,
            "token_usage": dict(self.token_usage),
            "cache_namespace": self.cache_namespace,
            "failure_diagnostics": dict(self.failure_diagnostics),
        }


class MockJSONProvider:
    """Exact-image-hash mock provider used by all no-cost tests."""

    def __init__(
        self,
        responses: Mapping[Any, Mapping[str, Any] | str] | None = None,
        *,
        responder: Callable[[str, str, str], Mapping[str, Any] | str] | None = None,
        model_identity: str = "mock-label-v4",
        namespace: str = "mock",
    ) -> None:
        self.responses = dict(responses or {})
        self.responder = responder
        self.model_identity = model_identity
        self.namespace = namespace
        self.request_policy_version = MOCK_JSON_REQUEST_POLICY_VERSION
        self.call_count = 0

    def call_json(
        self,
        *,
        stage: str,
        prompt: str,
        prompt_version: str,
        image_path: str | Path | None = None,
        payload: Mapping[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> ProviderArtifact:
        image_hash = exact_rgba_content_hash(image_path) if image_path else ""
        request_hash, prompt_hash = request_identity(
            stage=stage,
            model_identity=self.model_identity,
            prompt=prompt,
            prompt_version=prompt_version,
            image_hash=image_hash,
            payload=payload,
            request_policy_version=self.request_policy_version,
            request_options={"max_tokens": int(max_tokens)},
        )
        self.call_count += 1
        key_options = ((stage, image_hash), image_hash, stage, request_hash)
        response: Mapping[str, Any] | str | None = None
        for key in key_options:
            if key in self.responses:
                response = self.responses[key]
                break
        if response is None and self.responder is not None:
            response = self.responder(stage, image_hash, prompt)
        if response is None:
            diagnostic = {"error_type": "mock_response_missing", "retryable": False}
            return ProviderArtifact(
                stage=stage,
                raw_output="",
                parsed_output=None,
                model_identity=self.model_identity,
                request_hash=request_hash,
                image_hash=image_hash,
                prompt_version=prompt_version,
                prompt_hash=prompt_hash,
                latency_ms=0.0,
                http_attempts=0,
                request_policy_version=self.request_policy_version,
                token_usage={},
                cache_namespace=self.namespace,
                failure_diagnostics=diagnostic,
            )
        raw = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False, sort_keys=True)
        parsed, parse_failure = parse_json_object(raw)
        return ProviderArtifact(
            stage=stage,
            raw_output=raw,
            parsed_output=parsed,
            model_identity=self.model_identity,
            request_hash=request_hash,
            image_hash=image_hash,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            latency_ms=0.0,
            http_attempts=0,
            request_policy_version=self.request_policy_version,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cache_namespace=self.namespace,
            failure_diagnostics=parse_failure,
        )


class OpenAICompatibleJSONProvider:
    """Minimal OpenAI-compatible structured provider for bounded canaries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        namespace: str,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.model = str(model)
        self.namespace = str(namespace)
        self.request_policy_version = OPENAI_COMPATIBLE_REQUEST_POLICY_VERSION
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.call_count = 0

    @property
    def model_identity(self) -> str:
        # Endpoint host is part of identity; credentials never are.
        return f"openai-compatible:{self.base_url}:{self.model}"

    def call_json(
        self,
        *,
        stage: str,
        prompt: str,
        prompt_version: str,
        image_path: str | Path | None = None,
        payload: Mapping[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> ProviderArtifact:
        image_hash = exact_rgba_content_hash(image_path) if image_path else ""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if payload:
            content.append(
                {
                    "type": "text",
                    "text": "\nStructured evidence payload:\n"
                    + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }
            )
        if image_path:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_options = {
            "temperature": body["temperature"],
            "max_tokens": body["max_tokens"],
            "response_format": body["response_format"],
            "chat_template_kwargs": body["chat_template_kwargs"],
        }
        request_hash, prompt_hash = request_identity(
            stage=stage,
            model_identity=self.model_identity,
            prompt=prompt,
            prompt_version=prompt_version,
            image_hash=image_hash,
            payload=payload,
            request_policy_version=self.request_policy_version,
            request_options=request_options,
        )
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        started = time.perf_counter()
        self.call_count += 1
        raw_output = ""
        parsed_output: dict[str, Any] | None = None
        usage: dict[str, int] = {}
        failure: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            raw_output = _response_content(envelope)
            parsed_output, failure = parse_json_object(raw_output)
            usage = {
                str(key): int(value)
                for key, value in dict(envelope.get("usage") or {}).items()
                if isinstance(value, (int, float))
            }
        except urllib.error.HTTPError as exc:
            failure = {
                "error_type": "http_error",
                "status": int(exc.code),
                "retryable": int(exc.code) in {408, 409, 429} or int(exc.code) >= 500,
            }
        except urllib.error.URLError as exc:
            failure = {"error_type": "network_error", "reason_type": type(exc.reason).__name__, "retryable": True}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            failure = {
                "error_type": "malformed_provider_response",
                "exception_type": type(exc).__name__,
                "retryable": True,
            }
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ProviderArtifact(
            stage=stage,
            raw_output=raw_output,
            parsed_output=parsed_output,
            model_identity=self.model_identity,
            request_hash=request_hash,
            image_hash=image_hash,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
            http_attempts=1,
            request_policy_version=self.request_policy_version,
            token_usage=usage,
            cache_namespace=self.namespace,
            failure_diagnostics=failure,
        )


def request_identity(
    *,
    stage: str,
    model_identity: str,
    prompt: str,
    prompt_version: str,
    image_hash: str,
    payload: Mapping[str, Any] | None,
    request_policy_version: str = "generic_json_request_v1.0",
    request_options: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    canonical = json.dumps(
        {
            "stage": stage,
            "model_identity": model_identity,
            "prompt_hash": prompt_hash,
            "prompt_version": prompt_version,
            "image_hash": image_hash,
            "payload": payload,
            "request_policy_version": request_policy_version,
            "request_options": request_options,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), prompt_hash


def parse_json_object(raw_output: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    text = str(raw_output).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, {
            "error_type": "invalid_json",
            "line": int(exc.lineno),
            "column": int(exc.colno),
            "retryable": True,
        }
    if not isinstance(parsed, dict):
        return None, {"error_type": "json_root_not_object", "retryable": True}
    return parsed, {}


def image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _response_content(envelope: Mapping[str, Any]) -> str:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("provider response has no message")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return str(content)

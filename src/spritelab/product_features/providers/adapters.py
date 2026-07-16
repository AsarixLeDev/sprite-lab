"""Ollama, OpenAI-compatible, deterministic mock, plugin, and unavailable adapters."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from spritelab.product_features.providers.base import ProviderBase, capability_set, probe_from_error
from spritelab.product_features.providers.contracts import (
    BatchResponse,
    DiscoveryState,
    ImageLabelResult,
    LabelRequest,
    ModelValidation,
    PrivacyClass,
    ProbeResult,
    ProviderHealth,
    ProviderKind,
    ProviderModel,
    RequestEstimate,
    VisionCapabilities,
    validate_provider_contract,
)
from spritelab.product_features.providers.errors import ProviderError, ProviderInvalidOutputError
from spritelab.product_features.providers.identity import response_identity
from spritelab.product_features.providers.schema import (
    BATCH_LABEL_JSON_SCHEMA,
    LABEL_FIELDS,
    parse_json_object,
    validate_batch_items,
    validate_label,
)
from spritelab.product_features.providers.secrets import resolve_credential
from spritelab.product_features.providers.transport import (
    HttpResponse,
    HttpTransport,
    UrllibHttpTransport,
    normalized_http_error,
)

OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


def join_endpoint(base_url: str, path: str) -> str:
    """Join URL paths without treating Windows drive letters or URL ports as filesystem syntax."""

    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


class OllamaVisionProvider(ProviderBase):
    provider_id = "ollama.local"
    display_name = "Local Ollama"
    provider_kind = ProviderKind.OLLAMA
    privacy_class = PrivacyClass.LOCAL

    def __init__(
        self,
        *,
        endpoint: str = OLLAMA_DEFAULT_ENDPOINT,
        model: str | None = None,
        timeout_seconds: float = 10.0,
        transport: HttpTransport | None = None,
        maximum_payload_size: int = 20 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.transport = transport or UrllibHttpTransport()
        self.maximum_payload_size = maximum_payload_size

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities:
        return capability_set(
            self.privacy_class,
            maximum_image_count=1,
            maximum_payload_size=self.maximum_payload_size,
        )

    def list_models(self) -> tuple[ProviderModel, ...]:
        response = self.transport.request("GET", join_endpoint(self.endpoint, "api/tags"), timeout=self.timeout_seconds)
        _raise_for_status(response)
        try:
            payload = json.loads(response.json_text())
            models = payload["models"]
            if not isinstance(models, list):
                raise TypeError
            return tuple(
                ProviderModel(
                    model_id=str(item.get("model") or item.get("name")),
                    display_name=str(item.get("name") or item.get("model")),
                    metadata={"digest": item.get("digest"), "size": item.get("size")},
                )
                for item in models
                if isinstance(item, Mapping) and (item.get("model") or item.get("name"))
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError("provider_invalid_response", "Ollama returned an invalid model list.") from exc

    def probe(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            models = self.list_models()
        except ProviderError as exc:
            return probe_from_error(self.provider_id, exc, _elapsed_ms(started))
        return ProbeResult(
            DiscoveryState.AVAILABLE,
            "Ollama is available; no inference request was made.",
            self.provider_id,
            _elapsed_ms(started),
            models,
            {"probe": "GET /api/tags"},
        )

    def validate_model(self, model_id: str | None) -> ModelValidation:
        selected = model_id or self.model
        try:
            models = self.list_models()
        except ProviderError as exc:
            return ModelValidation(probe_from_error(self.provider_id, exc).state, selected, str(exc))
        if not selected:
            if len(models) != 1:
                return ModelValidation(
                    DiscoveryState.UNSUPPORTED_MODEL,
                    None,
                    "Select a model because Ollama did not expose exactly one installed model.",
                )
            selected = models[0].model_id
        if selected not in {model.model_id for model in models}:
            return ModelValidation(
                DiscoveryState.UNSUPPORTED_MODEL, selected, "The selected Ollama model is not installed."
            )
        body = json.dumps({"model": selected, "verbose": False}, separators=(",", ":")).encode()
        try:
            response = self.transport.request(
                "POST",
                join_endpoint(self.endpoint, "api/show"),
                headers={"Content-Type": "application/json"},
                body=body,
                timeout=self.timeout_seconds,
            )
            _raise_for_status(response)
            payload = json.loads(response.json_text())
            capabilities = payload.get("capabilities", [])
            if not isinstance(capabilities, list):
                raise TypeError
        except (json.JSONDecodeError, TypeError):
            return ModelValidation(DiscoveryState.MISCONFIGURED, selected, "Ollama returned invalid model details.")
        except ProviderError as exc:
            return ModelValidation(probe_from_error(self.provider_id, exc).state, selected, str(exc))
        normalized = tuple(str(value) for value in capabilities)
        if "vision" not in normalized:
            return ModelValidation(
                DiscoveryState.UNSUPPORTED_MODEL,
                selected,
                "The selected Ollama model does not declare vision capability.",
                normalized,
            )
        return ModelValidation(DiscoveryState.AVAILABLE, selected, "The Ollama vision model is available.", normalized)

    def label_batch(self, request: LabelRequest) -> BatchResponse:
        event = self._begin_request(request)
        try:
            self._check_cancelled(event)
            image = request.images[0]
            prompt = _batch_prompt(request)
            payload = {
                "model": request.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(image.data).decode("ascii")],
                    }
                ],
                "stream": False,
                "format": BATCH_LABEL_JSON_SCHEMA,
                "options": {"temperature": 0},
            }
            response = self.transport.request(
                "POST",
                join_endpoint(self.endpoint, "api/chat"),
                headers={"Content-Type": "application/json"},
                body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
                timeout=request.timeout_seconds,
                cancel_event=event,
            )
            _raise_for_status(response)
            try:
                envelope = json.loads(response.json_text())
                content = envelope["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ProviderInvalidOutputError("Ollama returned a malformed response envelope.") from exc
            return _normalized_batch_response(request, parse_json_object(content), envelope)
        finally:
            self._end_request(request.request_id)


class OpenAICompatibleVisionProvider(ProviderBase):
    provider_kind = ProviderKind.OPENAI_COMPATIBLE

    def __init__(
        self,
        *,
        endpoint: str,
        privacy_class: PrivacyClass,
        model: str | None = None,
        provider_id: str = "openai-compatible.configured",
        display_name: str = "vLLM / OpenAI-compatible",
        credential_env: str | None = None,
        timeout_seconds: float = 10.0,
        transport: HttpTransport | None = None,
        vision: bool = True,
        structured_output: bool = True,
        multiple_images: bool = True,
        batching: bool = True,
        maximum_image_count: int = 8,
        maximum_payload_size: int = 20 * 1024 * 1024,
        pricing_per_request: float | None = None,
        pricing_per_image: float | None = None,
        pricing_currency: str | None = None,
    ) -> None:
        super().__init__(
            pricing_per_request=pricing_per_request,
            pricing_per_image=pricing_per_image,
            pricing_currency=pricing_currency,
        )
        self.endpoint = endpoint.rstrip("/")
        self.privacy_class = privacy_class
        self.model = model
        self.provider_id = provider_id
        self.display_name = display_name
        self.credential_env = credential_env
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.transport = transport or UrllibHttpTransport()
        self._capabilities = capability_set(
            privacy_class,
            vision=vision,
            structured_output=structured_output,
            multiple_images=multiple_images,
            batching=batching,
            maximum_image_count=maximum_image_count,
            maximum_payload_size=maximum_payload_size,
        )

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities:
        return self._capabilities

    def _headers(self) -> dict[str, str]:
        credential = resolve_credential(self.credential_env)
        if self.credential_env and not credential:
            raise ProviderError(
                "provider_credential_missing",
                f"Credential environment variable {self.credential_env} is not set.",
            )
        headers = {"Content-Type": "application/json"}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        return headers

    def list_models(self) -> tuple[ProviderModel, ...]:
        response = self.transport.request(
            "GET",
            join_endpoint(self.endpoint, "models"),
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        _raise_for_status(response)
        try:
            payload = json.loads(response.json_text())
            data = payload["data"]
            if not isinstance(data, list):
                raise TypeError
            return tuple(
                ProviderModel(
                    model_id=str(item["id"]),
                    display_name=str(item["id"]),
                    metadata={key: value for key, value in item.items() if key in {"created", "owned_by", "object"}},
                )
                for item in data
                if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError("provider_invalid_response", "The endpoint returned an invalid model list.") from exc

    def probe(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            models = self.list_models()
        except (ProviderError, ValueError) as exc:
            error = exc if isinstance(exc, ProviderError) else ProviderError("provider_invalid_configuration", str(exc))
            return probe_from_error(self.provider_id, error, _elapsed_ms(started))
        return ProbeResult(
            DiscoveryState.AVAILABLE,
            "The OpenAI-compatible endpoint is available; no inference request was made.",
            self.provider_id,
            _elapsed_ms(started),
            models,
            {"probe": "GET /models"},
        )

    def validate_model(self, model_id: str | None) -> ModelValidation:
        selected = model_id or self.model
        try:
            models = self.list_models()
        except (ProviderError, ValueError) as exc:
            error = exc if isinstance(exc, ProviderError) else ProviderError("provider_invalid_configuration", str(exc))
            return ModelValidation(probe_from_error(self.provider_id, error).state, selected, str(error))
        if not selected:
            if len(models) != 1:
                return ModelValidation(
                    DiscoveryState.UNSUPPORTED_MODEL,
                    None,
                    "Select a model because the endpoint did not expose exactly one model.",
                )
            selected = models[0].model_id
        if selected not in {model.model_id for model in models}:
            return ModelValidation(DiscoveryState.UNSUPPORTED_MODEL, selected, "The selected model is not available.")
        if not self._capabilities.vision:
            return ModelValidation(
                DiscoveryState.UNSUPPORTED_MODEL, selected, "This endpoint does not declare vision capability."
            )
        if not self._capabilities.structured_output:
            return ModelValidation(
                DiscoveryState.UNSUPPORTED_MODEL,
                selected,
                "This endpoint does not declare the required structured-output capability.",
            )
        return ModelValidation(DiscoveryState.AVAILABLE, selected, "The configured model is available.")

    def label_batch(self, request: LabelRequest) -> BatchResponse:
        event = self._begin_request(request)
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": _batch_prompt(request)}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
                    },
                }
                for image in request.images
            )
            payload = {
                "model": request.model_id,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "spritelab_vision_labels",
                        "strict": True,
                        "schema": BATCH_LABEL_JSON_SCHEMA,
                    },
                },
            }
            response = self.transport.request(
                "POST",
                join_endpoint(self.endpoint, "chat/completions"),
                headers=self._headers(),
                body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
                timeout=request.timeout_seconds,
                cancel_event=event,
            )
            _raise_for_status(response)
            try:
                envelope = json.loads(response.json_text())
                content_value = envelope["choices"][0]["message"]["content"]
                if isinstance(content_value, list):
                    content_value = "\n".join(
                        str(part.get("text", "")) for part in content_value if isinstance(part, Mapping)
                    )
                if not isinstance(content_value, str):
                    raise TypeError
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise ProviderInvalidOutputError("The endpoint returned a malformed response envelope.") from exc
            return _normalized_batch_response(request, parse_json_object(content_value), envelope)
        finally:
            self._end_request(request.request_id)


class DeterministicMockVisionProvider(ProviderBase):
    provider_id = "mock.deterministic"
    display_name = "Deterministic mock provider"
    provider_kind = ProviderKind.MOCK
    privacy_class = PrivacyClass.LOCAL

    def __init__(
        self,
        responses: Mapping[str, Mapping[str, Any] | ProviderError] | None = None,
        *,
        failures: Sequence[ProviderError] = (),
        responder: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.responses = dict(responses or {})
        self.failures = list(failures)
        self.responder = responder
        self.call_count = 0

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities:
        return capability_set(
            self.privacy_class,
            multiple_images=True,
            batching=True,
            maximum_image_count=32,
            maximum_payload_size=100 * 1024 * 1024,
        )

    def list_models(self) -> tuple[ProviderModel, ...]:
        return (ProviderModel("mock-vision-v1", "Mock vision v1", ("vision", "structured_output")),)

    def probe(self) -> ProbeResult:
        return ProbeResult(
            DiscoveryState.AVAILABLE,
            "The deterministic mock provider is available.",
            self.provider_id,
            0.0,
            self.list_models(),
            {"network_calls": 0},
        )

    def validate_model(self, model_id: str | None) -> ModelValidation:
        selected = model_id or "mock-vision-v1"
        if selected != "mock-vision-v1":
            return ModelValidation(DiscoveryState.UNSUPPORTED_MODEL, selected, "Unknown deterministic mock model.")
        return ModelValidation(DiscoveryState.AVAILABLE, selected, "The deterministic mock model is available.")

    def label_batch(self, request: LabelRequest) -> BatchResponse:
        event = self._begin_request(request)
        self.call_count += 1
        try:
            self._check_cancelled(event)
            if self.failures:
                raise self.failures.pop(0)
            payload: list[dict[str, Any]] = []
            results: list[ImageLabelResult] = []
            for image in request.images:
                value = self.responses.get(image.image_id)
                if value is None and self.responder is not None:
                    value = self.responder(image.image_id)
                if value is None:
                    value = {
                        "state": "abstained",
                        "domain": None,
                        "category": None,
                        "canonical_object": None,
                        "role": None,
                        "description": None,
                        "confidence": 0.0,
                        "abstention_reasons": ["deterministic_mock_has_no_fixture"],
                        "provider_metadata": {"mock": True},
                    }
                if isinstance(value, ProviderError):
                    payload.append({"image_id": image.image_id, "error": value.code})
                    continue
                payload.append({"image_id": image.image_id, **dict(value)})
            response_id = response_identity(request.request_id, payload)
            by_id = {str(item["image_id"]): item for item in payload}
            for image in request.images:
                item = by_id[image.image_id]
                if "error" in item:
                    error = self.responses[image.image_id]
                    assert isinstance(error, ProviderError)
                    results.append(_error_result(image.image_id, request.request_id, response_id, error))
                    continue
                try:
                    label = validate_label({name: item[name] for name in LABEL_FIELDS})
                    results.append(ImageLabelResult(image.image_id, request.request_id, response_id, label=label))
                except (KeyError, ProviderInvalidOutputError) as exc:
                    results.append(
                        ImageLabelResult(
                            image.image_id,
                            request.request_id,
                            response_id,
                            error_code="provider_invalid_output",
                            error_message=str(exc),
                        )
                    )
            return BatchResponse(request.request_id, response_id, tuple(results), {"mock": True})
        finally:
            self._end_request(request.request_id)


class PluginVisionProviderAdapter:
    provider_kind = ProviderKind.PLUGIN

    def __init__(self, provider: object) -> None:
        validate_provider_contract(provider)
        self.provider: Any = provider
        self.provider_id = str(self.provider.provider_id)
        self.display_name = str(self.provider.display_name)
        self.privacy_class = PrivacyClass(self.provider.privacy_class)

    def probe(self) -> ProbeResult:
        return self.provider.probe()  # type: ignore[no-any-return,attr-defined]

    def list_models(self) -> tuple[ProviderModel, ...]:
        return tuple(self.provider.list_models())  # type: ignore[attr-defined]

    def validate_model(self, model_id: str | None) -> ModelValidation:
        return self.provider.validate_model(model_id)  # type: ignore[no-any-return,attr-defined]

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities:
        return self.provider.capabilities(model_id)  # type: ignore[no-any-return,attr-defined]

    def estimate_request(self, request: LabelRequest) -> RequestEstimate:
        return self.provider.estimate_request(request)  # type: ignore[no-any-return,attr-defined]

    def label_batch(self, request: LabelRequest) -> BatchResponse:
        return self.provider.label_batch(request)  # type: ignore[no-any-return,attr-defined]

    def cancel(self, request_id: str) -> bool:
        return bool(self.provider.cancel(request_id))  # type: ignore[attr-defined]

    def health(self) -> ProviderHealth:
        return self.provider.health()  # type: ignore[no-any-return,attr-defined]


class UnavailableVisionProvider(ProviderBase):
    provider_kind = ProviderKind.UNAVAILABLE

    def __init__(
        self,
        provider_id: str,
        display_name: str,
        message: str,
        *,
        state: DiscoveryState = DiscoveryState.MISCONFIGURED,
        privacy_class: PrivacyClass = PrivacyClass.HOSTED,
    ) -> None:
        super().__init__()
        self.provider_id = provider_id
        self.display_name = display_name
        self.message = message
        self.state = state
        self.privacy_class = privacy_class

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities:
        return capability_set(self.privacy_class, vision=False, structured_output=False)

    def probe(self) -> ProbeResult:
        return ProbeResult(self.state, self.message, self.provider_id, details={"network_calls": 0})

    def list_models(self) -> tuple[ProviderModel, ...]:
        return ()

    def validate_model(self, model_id: str | None) -> ModelValidation:
        return ModelValidation(self.state, model_id, self.message)

    def label_batch(self, request: LabelRequest) -> BatchResponse:
        raise ProviderError("provider_unavailable", self.message)

    def health(self) -> ProviderHealth:
        return ProviderHealth(self.state, self.message, datetime.now(UTC).isoformat())


def _batch_prompt(request: LabelRequest) -> str:
    identities = ", ".join(image.image_id for image in request.images)
    return (
        f"{request.prompt}\n\nImage identities in order: {identities}. "
        "Return exactly the supplied JSON schema. Preserve uncertainty by using state=abstained and explicit "
        "abstention_reasons; do not invent an identity."
    )


def _normalized_batch_response(
    request: LabelRequest, payload: Mapping[str, Any], envelope: Mapping[str, Any]
) -> BatchResponse:
    response_id = response_identity(request.request_id, payload)
    labels, failures = validate_batch_items(payload, [image.image_id for image in request.images])
    results: list[ImageLabelResult] = []
    for image in request.images:
        if image.image_id in labels:
            results.append(
                ImageLabelResult(image.image_id, request.request_id, response_id, label=labels[image.image_id])
            )
        else:
            results.append(
                ImageLabelResult(
                    image.image_id,
                    request.request_id,
                    response_id,
                    error_code="provider_invalid_output",
                    error_message=failures.get(image.image_id, "The provider omitted this image result."),
                )
            )
    metadata = {
        "provider_response_id": envelope.get("id"),
        "model": envelope.get("model"),
        "usage": envelope.get("usage") if isinstance(envelope.get("usage"), Mapping) else {},
    }
    return BatchResponse(request.request_id, response_id, tuple(results), metadata)


def _raise_for_status(response: HttpResponse) -> None:
    if not 200 <= response.status < 300:
        raise normalized_http_error(response)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _error_result(image_id: str, request_id: str, response_id: str, error: ProviderError) -> ImageLabelResult:
    return ImageLabelResult(
        image_id=image_id,
        request_id=request_id,
        response_id=response_id,
        error_code=error.code,
        error_message=str(error),
        retryable=error.retryable,
    )

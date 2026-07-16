from __future__ import annotations

import json
from collections import defaultdict
from threading import Event
from urllib.parse import urlsplit

from spritelab.product_features.providers import (
    DiscoveryState,
    ImageInput,
    LabelRequest,
    OllamaVisionProvider,
    OpenAICompatibleVisionProvider,
    PrivacyClass,
    ProviderError,
    ProviderSettings,
    VisionProviderRegistry,
)
from spritelab.product_features.providers.transport import HttpResponse


class FakeTransport:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[HttpResponse | Exception]] = defaultdict(list)
        self.calls: list[dict[str, object]] = []

    def add(self, method: str, path: str, *responses: HttpResponse | Exception) -> FakeTransport:
        self.routes[(method, path)].extend(responses)
        return self

    def request(
        self,
        method: str,
        url: str,
        *,
        headers=None,
        body=None,
        timeout: float,
        cancel_event: Event | None = None,
    ) -> HttpResponse:
        path = urlsplit(url).path
        self.calls.append({"method": method, "url": url, "path": path, "headers": dict(headers or {}), "body": body})
        values = self.routes[(method, path)]
        if not values:
            raise AssertionError(f"Unexpected real-like HTTP request: {method} {url}")
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, Exception):
            raise value
        return value


def response(status: int, value: object, headers: dict[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status, json.dumps(value).encode(), headers or {})


def normalized(image_id: str = "sprite-1", *, state: str = "labeled") -> dict[str, object]:
    return {
        "image_id": image_id,
        "state": state,
        "domain": "fantasy" if state == "labeled" else None,
        "category": "weapon" if state == "labeled" else None,
        "canonical_object": "sword" if state == "labeled" else None,
        "role": "equipment" if state == "labeled" else None,
        "description": "A small sword." if state == "labeled" else "Ambiguous sprite.",
        "confidence": 0.9 if state == "labeled" else 0.1,
        "abstention_reasons": [] if state == "labeled" else ["ambiguous_pixels"],
        "provider_metadata": {"fixture": True},
    }


def openai_envelope(*items: dict[str, object]) -> dict[str, object]:
    return {
        "id": "response-fixture",
        "model": "vision-model",
        "choices": [{"message": {"content": json.dumps({"results": list(items)})}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_automatic_local_discovery_uses_only_ollama_health_probe() -> None:
    transport = FakeTransport().add("GET", "/api/tags", response(200, {"models": [{"name": "vision-local"}]}))
    registry = VisionProviderRegistry(ProviderSettings(), transport=transport, plugin_entry_points=())
    discovered = registry.discover()
    assert [(item.provider.provider_id, item.probe.state) for item in discovered] == [
        ("ollama.local", DiscoveryState.AVAILABLE)
    ]
    assert [call["path"] for call in transport.calls] == ["/api/tags"]


def test_ollama_unavailable_is_normalized() -> None:
    transport = FakeTransport().add(
        "GET", "/api/tags", ProviderError("provider_unavailable", "Endpoint unavailable", retryable=True)
    )
    result = OllamaVisionProvider(transport=transport).probe()
    assert result.state == DiscoveryState.UNAVAILABLE


def test_ollama_available_and_vision_model_validation() -> None:
    transport = FakeTransport()
    transport.add("GET", "/api/tags", response(200, {"models": [{"name": "vision-local"}]}))
    transport.add("POST", "/api/show", response(200, {"capabilities": ["completion", "vision"]}))
    provider = OllamaVisionProvider(model="vision-local", transport=transport)
    validation = provider.validate_model(None)
    assert validation.valid
    assert validation.capabilities == ("completion", "vision")


def test_ollama_rejects_installed_nonvision_model_without_inference() -> None:
    transport = FakeTransport()
    transport.add("GET", "/api/tags", response(200, {"models": [{"name": "text-only"}]}))
    transport.add("POST", "/api/show", response(200, {"capabilities": ["completion"]}))
    validation = OllamaVisionProvider(model="text-only", transport=transport).validate_model(None)
    assert validation.state == DiscoveryState.UNSUPPORTED_MODEL
    assert [call["path"] for call in transport.calls] == ["/api/tags", "/api/show"]


def test_openai_compatible_vllm_endpoint_probe_and_model_validation() -> None:
    transport = FakeTransport().add(
        "GET", "/v1/models", response(200, {"data": [{"id": "vision-model", "owned_by": "local"}]})
    )
    provider = OpenAICompatibleVisionProvider(
        endpoint="http://127.0.0.1:8000/v1",
        privacy_class=PrivacyClass.LOCAL,
        model="vision-model",
        transport=transport,
    )
    assert provider.probe().state == DiscoveryState.AVAILABLE
    assert provider.validate_model("missing").state == DiscoveryState.UNSUPPORTED_MODEL


def test_hosted_endpoint_and_authentication_required() -> None:
    transport = FakeTransport().add("GET", "/v1/models", response(401, {"error": "never surfaced"}))
    provider = OpenAICompatibleVisionProvider(
        endpoint="https://vision.example/v1",
        privacy_class=PrivacyClass.HOSTED,
        transport=transport,
    )
    result = provider.probe()
    assert provider.privacy_class == PrivacyClass.HOSTED
    assert result.state == DiscoveryState.AUTHENTICATION_REQUIRED
    assert "never surfaced" not in result.message


def test_missing_credential_reference_is_misconfigured_without_http(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VISION_KEY", raising=False)
    transport = FakeTransport()
    provider = OpenAICompatibleVisionProvider(
        endpoint="https://vision.example/v1",
        privacy_class=PrivacyClass.HOSTED,
        credential_env="MISSING_VISION_KEY",
        transport=transport,
    )
    assert provider.probe().state == DiscoveryState.MISCONFIGURED
    assert transport.calls == []


def test_openai_request_uses_official_multimodal_and_json_schema_shapes(monkeypatch) -> None:
    monkeypatch.setenv("VISION_TEST_KEY", "runtime-only-secret")
    transport = FakeTransport().add("POST", "/v1/chat/completions", response(200, openai_envelope(normalized())))
    provider = OpenAICompatibleVisionProvider(
        endpoint="http://localhost:8000/v1",
        privacy_class=PrivacyClass.LOCAL,
        credential_env="VISION_TEST_KEY",
        transport=transport,
    )
    request = LabelRequest("request-1", "vision-model", "Label conservatively.", (ImageInput("sprite-1", b"png"),), 5)
    result = provider.label_batch(request)
    assert result.results[0].label.canonical_object == "sword"
    call = transport.calls[0]
    payload = json.loads(call["body"])
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert call["headers"]["Authorization"] == "Bearer runtime-only-secret"


def test_ollama_label_request_uses_native_chat_images_and_schema() -> None:
    envelope = {"model": "vision-local", "message": {"content": json.dumps({"results": [normalized()]})}}
    transport = FakeTransport().add("POST", "/api/chat", response(200, envelope))
    provider = OllamaVisionProvider(model="vision-local", transport=transport)
    request = LabelRequest("request-1", "vision-local", "Label.", (ImageInput("sprite-1", b"png"),), 5)
    assert provider.label_batch(request).results[0].ok
    payload = json.loads(transport.calls[0]["body"])
    assert payload["stream"] is False
    assert payload["format"]["required"] == ["results"]
    assert payload["messages"][0]["images"]


def test_response_item_validation_preserves_valid_siblings() -> None:
    malformed = normalized("bad")
    malformed["confidence"] = "high"
    transport = FakeTransport().add(
        "POST", "/v1/chat/completions", response(200, openai_envelope(normalized("good"), malformed))
    )
    provider = OpenAICompatibleVisionProvider(
        endpoint="http://localhost:8000/v1", privacy_class=PrivacyClass.LOCAL, transport=transport
    )
    request = LabelRequest(
        "request-1", "vision-model", "Label.", (ImageInput("good", b"one"), ImageInput("bad", b"two")), 5
    )
    result = provider.label_batch(request)
    assert result.results[0].ok
    assert result.results[1].error_code == "provider_invalid_output"


def test_no_provider_test_uses_the_real_network(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    provider = OllamaVisionProvider(transport=FakeTransport().add("GET", "/api/tags", response(200, {"models": []})))
    assert provider.probe().state == DiscoveryState.AVAILABLE

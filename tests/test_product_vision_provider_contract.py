from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_core import ProductPlugin, ProjectContext
from spritelab.product_features.providers import (
    DeterministicMockVisionProvider,
    DiscoveryState,
    LabelState,
    PrivacyClass,
    PrivacyPolicy,
    ProviderMode,
    ProviderSettings,
    VisionProviderRegistry,
)
from spritelab.product_features.providers.adapters import PluginVisionProviderAdapter, join_endpoint
from spritelab.product_features.providers.plugin import SETTINGS_SCHEMA, build_plugin
from spritelab.product_features.providers.schema import validate_label
from spritelab.product_features.providers.secrets import redact_headers, redact_text, redact_url, resolve_credential
from spritelab.product_features.providers.web import _settings_html, create_settings_router


def test_provider_settings_support_normal_user_modes() -> None:
    assert ProviderSettings.from_mapping({"type": "auto"}).mode == ProviderMode.AUTOMATIC
    assert ProviderSettings.from_mapping({"type": "local"}).mode == ProviderMode.LOCAL
    hosted = ProviderSettings.from_mapping({"type": "hosted", "endpoint": "https://vision.example/v1"})
    assert hosted.mode == ProviderMode.HOSTED
    assert hosted.location == PrivacyClass.HOSTED


def test_windows_endpoint_configuration_is_not_treated_as_a_path() -> None:
    settings = ProviderSettings.from_mapping(
        {"type": "vllm", "endpoint": "http://127.0.0.1:8000/v1", "model": "vision-model"}
    )
    assert settings.endpoint == "http://127.0.0.1:8000/v1"
    assert join_endpoint(settings.endpoint, "chat/completions") == "http://127.0.0.1:8000/v1/chat/completions"
    with pytest.raises(ValueError, match="http"):
        ProviderSettings.from_mapping({"type": "vllm", "endpoint": r"C:\models\server"})


def test_non_loopback_endpoint_requires_explicit_hosted_location() -> None:
    with pytest.raises(ValueError, match="location"):
        ProviderSettings.from_mapping({"type": "openai_compatible", "endpoint": "https://vision.example/v1"})


def test_environment_configuration_contains_references_not_credentials() -> None:
    settings = ProviderSettings.from_mapping(
        {"type": "auto"},
        environ={
            "SPRITELAB_VISION_ENDPOINT": "http://localhost:8000/v1",
            "SPRITELAB_VISION_MODEL": "local-vlm",
            "SPRITELAB_VISION_CREDENTIAL_ENV": "LOCAL_VLM_KEY",
        },
    )
    assert settings.source == "environment"
    assert settings.credential_env == "LOCAL_VLM_KEY"
    assert resolve_credential(settings.credential_env, environ={"LOCAL_VLM_KEY": "runtime-secret"}) == "runtime-secret"
    assert "runtime-secret" not in repr(settings)


def test_credential_redaction_covers_headers_urls_and_messages() -> None:
    assert redact_headers({"Authorization": "Bearer secret", "Accept": "json"}) == {
        "Authorization": "<redacted>",
        "Accept": "json",
    }
    redacted_url = redact_url("https://example.test/v1?api_key=secret&model=safe")
    assert "secret" not in redacted_url
    assert "model=safe" in redacted_url
    assert "secret" not in redact_text("failed Bearer secret", ("secret",))


def test_result_normalization_is_strict_and_preserves_abstention() -> None:
    label = validate_label(
        {
            "state": "abstained",
            "domain": None,
            "category": None,
            "canonical_object": None,
            "role": None,
            "description": "Ambiguous 16px silhouette.",
            "confidence": 0.2,
            "abstention_reasons": ["silhouette_matches_multiple_objects", "insufficient_detail"],
            "provider_metadata": {"source": "fixture"},
        }
    )
    assert label.state == LabelState.ABSTAINED
    assert label.abstention_reasons == ("silhouette_matches_multiple_objects", "insufficient_detail")
    with pytest.raises(Exception, match="confidence"):
        validate_label({**label.to_dict(), "confidence": "0.2"})


def test_product_plugin_exports_settings_page_contract() -> None:
    plugin = build_plugin()
    assert isinstance(plugin, ProductPlugin)
    assert plugin.plugin_id == "vision.providers"
    assert plugin.navigation[0].path == "/settings/vision"
    assert plugin.settings_schema == SETTINGS_SCHEMA
    router = plugin.web_router_factory(ProjectContext(project_root=Path.cwd(), config={}))
    assert "/settings/vision" in {route.path for route in router.routes}


def test_settings_page_has_required_nonexpert_controls() -> None:
    page = _settings_html(ProviderSettings())
    for text in (
        "Automatic",
        "Local Ollama",
        "vLLM / OpenAI-compatible",
        "Hosted endpoint",
        "Custom plugin",
        "Auto-detect",
        "Test connection",
        "Privacy policy",
        "Capability summary",
    ):
        assert text in page


def test_connection_action_is_health_only_and_rejects_image_payloads() -> None:
    provider = DeterministicMockVisionProvider()

    class Request:
        async def json(self):
            return {"type": "plugin", "image_test": True}

    router = create_settings_router(
        ProjectContext(Path.cwd(), config={}),
        registry_factory=lambda settings: VisionProviderRegistry(
            settings, providers=(provider,), plugin_entry_points=()
        ),
    )
    endpoint = next(route.endpoint for route in router.routes if route.path == "/api/vision-providers/test")
    response = asyncio.run(endpoint(Request()))
    assert response.status_code == 400
    assert b"does not accept or transmit images" in response.body
    assert provider.call_count == 0


def test_plugin_adapter_loading_and_discovery() -> None:
    provider = DeterministicMockVisionProvider()
    entry_point = SimpleNamespace(name="fixture", load=lambda: lambda: provider)
    settings = ProviderSettings.from_mapping({"type": "plugin"})
    discovered = VisionProviderRegistry(settings, plugin_entry_points=(entry_point,)).discover()
    assert len(discovered) == 1
    assert discovered[0].probe.state == DiscoveryState.AVAILABLE
    assert discovered[0].source == "plugin"
    assert isinstance(discovered[0].provider, PluginVisionProviderAdapter)


def test_broken_plugin_becomes_tested_unavailable_state() -> None:
    def broken() -> object:
        raise RuntimeError("contains-sensitive-plugin-detail")

    entry_point = SimpleNamespace(name="broken", load=broken)
    settings = ProviderSettings.from_mapping({"type": "plugin"})
    discovered = VisionProviderRegistry(settings, plugin_entry_points=(entry_point,)).discover()
    assert discovered[0].probe.state == DiscoveryState.MISCONFIGURED
    assert "contains-sensitive" not in discovered[0].probe.message


def test_runpod_native_is_scaffold_only_because_openai_compatibility_is_official() -> None:
    settings = ProviderSettings.from_mapping({"type": "runpod", "endpoint": "https://example.test/v1"})
    discovered = VisionProviderRegistry(settings, plugin_entry_points=()).discover()
    assert discovered[0].provider.provider_id == "runpod.native.unavailable"
    assert discovered[0].probe.state == DiscoveryState.MISCONFIGURED
    assert "OpenAI-compatible" in discovered[0].probe.message


def test_all_privacy_policies_are_parseable() -> None:
    for policy in PrivacyPolicy:
        settings = ProviderSettings.from_mapping({"type": "auto", "privacy_policy": policy.value})
        assert settings.privacy_policy == policy


def test_settings_schema_serializes_without_secrets() -> None:
    payload = json.dumps(SETTINGS_SCHEMA, sort_keys=True)
    assert "credential_env" in payload
    assert "api_key" not in payload

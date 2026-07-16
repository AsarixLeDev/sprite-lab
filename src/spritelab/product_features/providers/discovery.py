"""Safe ordered discovery for configured, local, and plugin vision providers."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.product_features.providers.adapters import (
    OllamaVisionProvider,
    OpenAICompatibleVisionProvider,
    PluginVisionProviderAdapter,
    UnavailableVisionProvider,
)
from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.contracts import (
    PrivacyClass,
    PrivacyPolicy,
    ProbeResult,
    ProviderMode,
    VisionProvider,
)
from spritelab.product_features.providers.transport import HttpTransport

PLUGIN_ENTRY_POINT_GROUP = "spritelab.vision_providers"


@dataclass(frozen=True)
class DiscoveredProvider:
    provider: VisionProvider
    probe: ProbeResult
    source: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.probe.to_dict(),
            "display_name": self.provider.display_name,
            "provider_kind": self.provider.provider_kind.value,
            "privacy_class": self.provider.privacy_class.value,
            "source": self.source,
            "capabilities": self.provider.capabilities().to_dict(),
        }


class VisionProviderRegistry:
    """Instance-scoped provider registry; discovery never scans network ranges."""

    def __init__(
        self,
        settings: ProviderSettings | None = None,
        *,
        transport: HttpTransport | None = None,
        providers: Iterable[VisionProvider] = (),
        plugin_entry_points: Sequence[object] | None = None,
    ) -> None:
        self.settings = settings or ProviderSettings()
        self.transport = transport
        self._providers: list[tuple[VisionProvider, str]] = [(provider, "injected") for provider in providers]
        self._plugin_entry_points = plugin_entry_points
        self._built = False

    def register(self, provider: VisionProvider, *, source: str = "injected") -> None:
        if any(existing.provider_id == provider.provider_id for existing, _ in self._providers):
            raise ValueError(f"Duplicate vision provider ID: {provider.provider_id}")
        self._providers.append((provider, source))

    def providers(self) -> tuple[VisionProvider, ...]:
        self._ensure_built()
        return tuple(provider for provider, _ in self._providers)

    def discover(self) -> tuple[DiscoveredProvider, ...]:
        self._ensure_built()
        ordered = sorted(self._providers, key=lambda item: _discovery_order(item[0], item[1]))
        return tuple(
            DiscoveredProvider(provider, provider.probe(), source, index)
            for index, (provider, source) in enumerate(ordered, start=1)
        )

    def select(
        self, discovered: Sequence[DiscoveredProvider] | None = None, *, provider_id: str | None = None
    ) -> DiscoveredProvider | None:
        candidates = list(discovered or self.discover())
        available = [item for item in candidates if item.probe.available]
        settings = self.settings
        if provider_id or settings.provider_id:
            selected_id = provider_id or settings.provider_id
            return next((item for item in available if item.provider.provider_id == selected_id), None)
        if settings.mode == ProviderMode.CUSTOM_PLUGIN:
            return next((item for item in available if item.source == "plugin"), None)
        if settings.mode == ProviderMode.LOCAL:
            available = [item for item in available if item.provider.privacy_class == PrivacyClass.LOCAL]
        elif settings.mode == ProviderMode.HOSTED:
            available = [item for item in available if item.provider.privacy_class == PrivacyClass.HOSTED]
        if settings.privacy_policy == PrivacyPolicy.LOCAL_ONLY:
            available = [item for item in available if item.provider.privacy_class == PrivacyClass.LOCAL]
        elif settings.privacy_policy == PrivacyPolicy.HOSTED_ONLY:
            available = [item for item in available if item.provider.privacy_class == PrivacyClass.HOSTED]
        return available[0] if available else None

    def _ensure_built(self) -> None:
        if self._built:
            return
        self._built = True
        settings = self.settings
        if settings.adapter == "runpod_native":
            self.register(
                UnavailableVisionProvider(
                    "runpod.native.unavailable",
                    "RunPod-specific adapter",
                    "Current RunPod vLLM endpoints use the OpenAI-compatible API; configure that adapter instead.",
                ),
                source="configured",
            )
        if settings.endpoint and settings.adapter in {None, "openai_compatible"}:
            assert settings.location is not None
            self.register(
                OpenAICompatibleVisionProvider(
                    endpoint=settings.endpoint,
                    privacy_class=settings.location,
                    model=settings.model,
                    provider_id=settings.provider_id or "openai-compatible.configured",
                    display_name="Hosted endpoint"
                    if settings.location == PrivacyClass.HOSTED
                    else "vLLM / OpenAI-compatible",
                    credential_env=settings.credential_env,
                    timeout_seconds=settings.timeout_seconds,
                    transport=self.transport,
                    vision=settings.vision,
                    structured_output=settings.structured_output,
                    multiple_images=settings.multiple_images,
                    batching=settings.batching,
                    maximum_image_count=settings.maximum_image_count,
                    maximum_payload_size=settings.maximum_payload_size,
                    pricing_per_request=settings.pricing_per_request,
                    pricing_per_image=settings.pricing_per_image,
                    pricing_currency=settings.pricing_currency,
                ),
                source="configured",
            )
        if settings.mode != ProviderMode.HOSTED and settings.adapter in {None, "ollama"}:
            self.register(
                OllamaVisionProvider(
                    endpoint=settings.endpoint
                    if settings.adapter == "ollama" and settings.endpoint
                    else None or "http://127.0.0.1:11434",
                    model=settings.model,
                    timeout_seconds=min(settings.timeout_seconds, 10.0),
                    transport=self.transport,
                    maximum_payload_size=settings.maximum_payload_size,
                ),
                source="local_default",
            )
        self._load_plugins()

    def _load_plugins(self) -> None:
        entry_points = self._plugin_entry_points
        if entry_points is None:
            discovered = importlib.metadata.entry_points()
            entry_points = (
                tuple(discovered.select(group=PLUGIN_ENTRY_POINT_GROUP)) if hasattr(discovered, "select") else ()
            )
        for entry_point in entry_points:
            name = str(getattr(entry_point, "name", "unknown"))
            try:
                loaded = entry_point.load()  # type: ignore[attr-defined]
                provider = loaded() if callable(loaded) else loaded
                self.register(PluginVisionProviderAdapter(provider), source="plugin")
            except Exception as exc:
                self.register(
                    UnavailableVisionProvider(
                        f"plugin.{name}.unavailable",
                        f"Plugin provider {name}",
                        f"Provider plugin could not be loaded ({type(exc).__name__}).",
                    ),
                    source="plugin",
                )


def _discovery_order(provider: VisionProvider, source: str) -> tuple[int, int, str]:
    # Local candidates always precede hosted candidates in automatic mode.
    privacy = 0 if provider.privacy_class == PrivacyClass.LOCAL else 1
    source_order = {"configured": 0, "local_default": 1, "plugin": 2, "injected": 3}.get(source, 9)
    return privacy, source_order, provider.provider_id

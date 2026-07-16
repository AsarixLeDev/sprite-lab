"""ProductPlugin export for the Sprite Lab vision-provider settings feature."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from spritelab.product_core import (
    ProductCapability,
    ProductCliRegistry,
    ProductPlugin,
    ProductResult,
    ProductSettingsError,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
    WebAssetBundle,
    WebNavigationItem,
)
from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.discovery import VisionProviderRegistry
from spritelab.product_features.providers.state import passive_provider_projection
from spritelab.product_features.providers.web import create_settings_router

PLUGIN_ID = "vision.providers"

SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "type": {
            "type": "string",
            "enum": ["auto", "local", "ollama", "vllm", "openai_compatible", "hosted", "plugin"],
            "default": "auto",
        },
        "provider_id": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "endpoint": {"type": ["string", "null"], "format": "uri"},
        "credential_env": {"type": ["string", "null"], "description": "Environment-variable name only."},
        "privacy_policy": {
            "type": "string",
            "enum": ["local_only", "allow_hosted", "hosted_only", "ask_before_hosted"],
            "default": "ask_before_hosted",
        },
        "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600, "default": 30},
        "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 4},
        "capabilities": {"type": "object"},
        "pricing": {
            "type": "object",
            "description": "Optional explicit prices; no prices are scraped or built in.",
        },
    },
}


def build_plugin() -> ProductPlugin:
    return ProductPlugin(
        plugin_id=PLUGIN_ID,
        title="Vision providers",
        cli_registration=_register_cli,
        status_provider=_status,
        capability_probe=_capabilities,
        web_router_factory=create_settings_router,
        navigation=(WebNavigationItem("vision-provider-settings", "Vision labeling", "/settings/vision", order=30),),
        required_backend_capabilities=(),
        settings_schema=SETTINGS_SCHEMA,
        web_assets=(WebAssetBundle("spritelab.product_features.providers"),),
        api_prefixes=("/settings/vision/api", "/api/vision-providers"),
    )


def _register_cli(registry: ProductCliRegistry) -> None:
    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("provider_action", nargs="?", choices=("list", "detect", "test"), default="list")

    registry.command(
        "providers",
        owner=PLUGIN_ID,
        help="List, detect, or health-test vision providers without image inference.",
        handler=_handle_cli,
        configure=configure,
    )


def _handle_cli(args: argparse.Namespace, raw: list[str]) -> ProductResult:
    context = _load_context()
    try:
        settings = ProviderSettings.from_context_config(context.config)
        registry = VisionProviderRegistry(settings)
        if args.provider_action == "list":
            providers = registry.providers()
            data = {
                "providers": [
                    {
                        "provider_id": provider.provider_id,
                        "display_name": provider.display_name,
                        "provider_kind": provider.provider_kind.value,
                        "privacy_class": provider.privacy_class.value,
                        "capabilities": provider.capabilities().to_dict(),
                        "state": "not_probed",
                    }
                    for provider in providers
                ],
                "network_calls": 0,
            }
            return ProductResult(
                ProductStatus.READY, "Vision providers are configured.", feature="providers", data=data
            )
        discovered = registry.discover()
    except ValueError as exc:
        return ProductResult(ProductStatus.BLOCKED, str(exc), feature="providers")
    available = [item for item in discovered if item.probe.available]
    status = ProductStatus.READY if available else ProductStatus.UNAVAILABLE
    message = (
        "Provider health check completed without image inference."
        if args.provider_action == "test"
        else "Safe provider discovery completed without image inference."
    )
    return ProductResult(
        status,
        message,
        feature="providers",
        data={
            "providers": [item.to_dict() for item in discovered],
            "selected_provider": registry.select(discovered).provider.provider_id
            if registry.select(discovered)
            else None,
            "image_inference_requests": 0,
        },
    )


def _status(context: ProjectContext) -> ProductResult:
    try:
        projection = passive_provider_projection(context)
    except (ValueError, ProductSettingsError) as exc:
        return ProductResult(ProductStatus.BLOCKED, str(exc), feature="providers")
    state = str(projection["state"])
    status = {
        "not_configured": ProductStatus.NOT_STARTED,
        "configured_unverified": ProductStatus.NOT_STARTED,
        "previously_verified": ProductStatus.READY,
        "unavailable_cached": ProductStatus.UNAVAILABLE,
        "authentication_required_cached": ProductStatus.BLOCKED,
        "stale_cached": ProductStatus.NOT_STARTED,
    }[state]
    messages = {
        "not_configured": "Provider has not been configured yet.",
        "configured_unverified": "Provider has not been checked yet.",
        "previously_verified": "Provider was verified by an explicit action.",
        "unavailable_cached": "The last explicit provider check reported unavailable.",
        "authentication_required_cached": "The last explicit provider check requires authentication.",
        "stale_cached": "The last provider check is stale. Refresh it explicitly.",
    }
    return ProductResult(
        status,
        messages[state],
        feature="providers",
        data=projection,
    )


def _capabilities(context: ProjectContext) -> tuple[ProductCapability, ...]:
    try:
        raw, _version, _saved = ProductSettingsRepository(context).effective_settings("provider")
        settings = ProviderSettings.from_mapping(raw)
    except (ValueError, ProductSettingsError) as exc:
        return (ProductCapability("vision.providers", "Vision provider hub", ProductStatus.BLOCKED, str(exc)),)
    return (
        ProductCapability(
            "vision.providers",
            "Vision provider hub",
            ProductStatus.READY,
            "Automatic, local, hosted, and plugin provider configuration is available.",
            details={"privacy_policy": settings.privacy_policy.value, "settings_path": "/settings/vision"},
        ),
    )


def _load_context() -> ProjectContext:
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    return ProjectContext(config.root, config=config.values, config_path=config.path, runs_directory=config.runs_dir)

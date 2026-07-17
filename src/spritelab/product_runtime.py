"""Instance-scoped composition of Sprite Lab's built-in product plugins."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from spritelab.product_core import (
    ProductCapability,
    ProductPlugin,
    ProductPluginRegistry,
    ProductResult,
    ProductStatus,
    ProjectContext,
)
from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_features.conditioned_v5 import build_plugin as build_conditioned_v5_plugin
from spritelab.product_features.conditioned_v5.intake import ConditionedDatasetImportAdapter
from spritelab.product_features.dataset.plugin import create_plugin as create_dataset_plugin
from spritelab.product_features.evaluation.plugin import build_plugin as build_evaluation_plugin
from spritelab.product_features.harvest import create_plugin as create_harvest_plugin
from spritelab.product_features.providers.plugin import build_plugin as build_provider_plugin
from spritelab.product_features.providers.product_adapter import HubProductVisionProvider
from spritelab.product_features.training import build_plugin as build_training_plugin
from spritelab.product_ux.copy_catalog import copy_for


@dataclass(frozen=True)
class ProductRuntime:
    """One disposable registry used for a single CLI or web application."""

    registry: ProductPluginRegistry

    @property
    def plugins(self) -> tuple[ProductPlugin, ...]:
        return tuple(self.registry)


def build_product_runtime() -> ProductRuntime:
    """Build the complete product without creating mutable process-global state."""

    registry = ProductPluginRegistry()
    dataset_import_adapter = ConditionedDatasetImportAdapter()
    for plugin in (
        _web_shell_plugin(),
        build_provider_plugin(),
        create_harvest_plugin(dataset_import_callback=dataset_import_adapter),
        build_conditioned_v5_plugin(),
        create_dataset_plugin(provider_factory=_dataset_provider),
        build_training_plugin(),
        build_evaluation_plugin(),
        _developer_commands_plugin(),
        _novice_ux_plugin(),
    ):
        registry.register(plugin)
    return ProductRuntime(registry)


def _dataset_provider(
    context: ProjectContext,
    confirm_hosted: Callable[[str], bool] | None,
) -> HubProductVisionProvider | None:
    try:
        return HubProductVisionProvider(context, confirm_hosted=confirm_hosted)
    except ValueError:
        return None


def _web_shell_plugin() -> ProductPlugin:
    def register_cli(registry: ProductCliRegistry) -> None:
        def configure(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--host", help="Explicit bind host; the default is loopback-only.")
            parser.add_argument("--port", help="Port number or 'auto'.")
            parser.add_argument("--no-open", action="store_true", help="Do not open a browser page.")
            parser.add_argument("--auth-token", help="Runtime-only token for explicit non-loopback binding.")

        registry.command(
            "app",
            owner="web.shell",
            help="Open the local Sprite Lab product.",
            configure=configure,
            handler=lambda _args, _argv: ProductResult(
                ProductStatus.READY,
                "The local Sprite Lab application is registered.",
                feature="web",
                data={"launch_command": "python -m spritelab v3 app"},
            ),
        )

    def capabilities(_context: ProjectContext) -> tuple[ProductCapability, ...]:
        # These capabilities describe mounted product contracts. Individual
        # services still enforce their own readiness and certification gates.
        return (
            ProductCapability("web.shell", "Product web shell", ProductStatus.READY),
            ProductCapability("compute.training", "Training compute contract", ProductStatus.READY),
            ProductCapability("evaluation.score_suite", "Evaluation backend contract", ProductStatus.READY),
            ProductCapability("generation.typed", "Typed generation contract", ProductStatus.READY),
        )

    return ProductPlugin(
        plugin_id="web.shell",
        title="Sprite Lab application",
        cli_registration=register_cli,
        status_provider=lambda _context: ProductResult(
            ProductStatus.READY,
            "The local product application is available.",
            feature="web",
        ),
        capability_probe=capabilities,
    )


def _developer_commands_plugin() -> ProductPlugin:
    return ProductPlugin(
        plugin_id="developer.commands",
        title="Developer commands",
        cli_registration=lambda _registry: None,
        status_provider=lambda _context: ProductResult(
            ProductStatus.READY,
            "Detailed engineering evidence is available separately.",
            feature="developer",
            data={"command": "python -m spritelab dev status"},
        ),
        capability_probe=lambda _context: (
            ProductCapability("developer.commands", "Developer command namespace", ProductStatus.READY),
        ),
    )


def _novice_ux_plugin() -> ProductPlugin:
    welcome = copy_for("welcome")

    def register_cli(registry: ProductCliRegistry) -> None:
        registry.command(
            "status",
            owner="novice.ux",
            help="Show a simple end-user project status.",
            handler=_product_status,
            replace=True,
        )

    return ProductPlugin(
        plugin_id="novice.ux",
        title="Guided product journey",
        cli_registration=register_cli,
        status_provider=lambda _context: ProductResult(
            ProductStatus.READY,
            welcome.body[0],
            feature="onboarding",
            data={
                "actions": ({"title": welcome.primary_action, "path": "/dataset"},),
                "next_action": welcome.next_action,
            },
        ),
        capability_probe=lambda _context: (
            ProductCapability("novice.copy", "Plain-language copy catalog", ProductStatus.READY),
            ProductCapability("novice.launchers", "Project launchers", ProductStatus.READY),
        ),
    )


def _product_status(_args: argparse.Namespace, _argv: list[str]) -> ProductResult:
    areas = (
        {
            "key": "dataset",
            "title": "Dataset",
            "status": "READY",
            "message": "Image preparation is available. Automatic descriptions still require an independent reliability check.",
        },
        {
            "key": "training",
            "title": "Training",
            "status": "UNAVAILABLE",
            "message": "Temporarily unavailable while final safety checks are verified.",
        },
        {
            "key": "evaluation",
            "title": "Evaluation",
            "status": "BLOCKED",
            "message": "Waiting for a certified trained model.",
        },
    )
    message = "\n\n".join(f"{area['title']}\n  {area['message']}" for area in areas)
    return ProductResult(
        ProductStatus.READY,
        message,
        feature="status",
        data={
            "areas": areas,
            "next_command": "python -m spritelab v3 dataset build <folder>",
            "developer_command": "python -m spritelab dev status",
        },
    )


__all__ = ["ProductRuntime", "build_product_runtime"]

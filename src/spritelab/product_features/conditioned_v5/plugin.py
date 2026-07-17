"""ProductPlugin registration for conditioned Dataset-v5."""

from __future__ import annotations

from collections.abc import Callable

from spritelab.product_core import (
    ProductCapability,
    ProductPlugin,
    ProductResult,
    ProductStatus,
    ProjectContext,
    WebAssetBundle,
    WebNavigationItem,
)
from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_features.conditioned_v5.service import ConditionedDatasetService
from spritelab.product_features.conditioned_v5.web import create_router

PLUGIN_ID = "dataset.conditioned_v5"
ServiceFactory = Callable[[ProjectContext], ConditionedDatasetService]


def register_cli(registry: ProductCliRegistry) -> None:
    """The production workflow is intentionally web-operated."""

    del registry


def create_plugin(*, service_factory: ServiceFactory | None = None) -> ProductPlugin:
    def service(context: ProjectContext) -> ConditionedDatasetService:
        return (
            service_factory(context) if service_factory is not None else ConditionedDatasetService(context.project_root)
        )

    def status(context: ProjectContext) -> ProductResult:
        inventory = service(context).inventory()
        intakes = inventory["managed_intakes"]
        jobs = inventory["jobs"]
        if any(job.get("status") == "COMPLETE" for job in jobs):
            state = ProductStatus.COMPLETE
            message = "A conditioned Dataset-v5 publication is recorded; activation and training remain separate."
        elif intakes:
            state = ProductStatus.READY
            message = "Managed Dataset imports are ready for conditioned preview."
        else:
            state = ProductStatus.NOT_STARTED
            message = "Import a trusted CC0/public-domain Harvest handoff into Dataset before conditioning."
        return ProductResult(state, message, feature=PLUGIN_ID, data=inventory)

    def capabilities(context: ProjectContext) -> tuple[ProductCapability, ...]:
        inventory = service(context).inventory()
        available = bool(inventory["managed_intakes"])
        return (
            ProductCapability(
                "dataset.conditioned_v5",
                "Conditioned Dataset-v5",
                ProductStatus.READY if available else ProductStatus.NOT_STARTED,
                "Offline candidate build with independent audit and one-time publication gates.",
                details={"managed_intake_count": len(inventory["managed_intakes"]), "network_actions": 0},
            ),
        )

    return ProductPlugin(
        plugin_id=PLUGIN_ID,
        title="Conditioned Dataset-v5",
        cli_registration=register_cli,
        status_provider=status,
        capability_probe=capabilities,
        web_router_factory=lambda context: create_router(context, service=service(context)),
        navigation=(WebNavigationItem("dataset-v5", "Dataset v5", "/dataset-v5", order=27),),
        web_assets=(WebAssetBundle("spritelab.product_features.conditioned_v5"),),
        api_prefixes=("/dataset-v5/api",),
    )


def build_plugin() -> ProductPlugin:
    return create_plugin()


__all__ = ["PLUGIN_ID", "build_plugin", "create_plugin", "register_cli"]

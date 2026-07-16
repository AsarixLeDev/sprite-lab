"""Training ProductPlugin registration."""

from __future__ import annotations

from collections.abc import Iterable

from spritelab.product_core import (
    ProductBlocker,
    ProductCapability,
    ProductPlugin,
    ProductResult,
    ProductStatus,
    ProjectContext,
    WebAssetBundle,
    WebNavigationItem,
)
from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_features.training.config import (
    effective_compute_context,
    passive_compute_projection,
)
from spritelab.product_features.training.service import TrainingService, backend_from_context
from spritelab.product_features.training.web import create_router
from spritelab.remote_compute import ComputeBackend, HostedBackendRegistry
from spritelab.v3.config import ProjectConfig

PLUGIN_ID = "training"


def register_training_cli(registry: ProductCliRegistry) -> None:
    """Retain the foundation-owned ``python -m spritelab v3 train`` command."""

    del registry


def create_plugin(*, hosted_backends: Iterable[ComputeBackend] = ()) -> ProductPlugin:
    hosted = HostedBackendRegistry(hosted_backends)

    def service(context: ProjectContext) -> TrainingService:
        loaded = ProjectConfig.load(context.project_root)
        fresh = ProjectContext(loaded.root, loaded.values, loaded.path, loaded.runs_dir)
        effective, _settings, _version, _saved = effective_compute_context(fresh)
        return TrainingService(effective, backend_from_context(effective, hosted_backends=hosted))

    def status_provider(context: ProjectContext) -> ProductResult:
        projection = passive_compute_projection(context)
        if projection["state"] == "invalid":
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Training compute configuration is unavailable.",
                feature="training",
                blockers=(ProductBlocker("compute_configuration", str(projection["message"])),),
                data=projection,
            )
        if projection["backend_type"] == "runpod":
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "RunPod is not available in this build.",
                feature="training",
                data=projection,
            )
        return ProductResult(
            ProductStatus.NOT_STARTED,
            "Compute is configured. Training safety gates are checked only after an explicit Start action.",
            feature="training",
            data=projection,
        )

    def capability_probe(context: ProjectContext) -> tuple[ProductCapability, ...]:
        projection = passive_compute_projection(context)
        available = projection["state"] != "invalid" and projection.get("backend_type") != "runpod"
        return (
            ProductCapability(
                "training.compute",
                "Training compute",
                ProductStatus.READY if available else ProductStatus.UNAVAILABLE,
                str(projection["message"]),
                details={"backend_type": projection.get("backend_type"), "compute_probes": 0},
            ),
        )

    def router_factory(context: ProjectContext) -> object:
        return create_router(context, service_factory=lambda: service(context))

    return ProductPlugin(
        plugin_id=PLUGIN_ID,
        title="Training",
        cli_registration=register_training_cli,
        status_provider=status_provider,
        capability_probe=capability_probe,
        web_router_factory=router_factory,
        navigation=(WebNavigationItem("training", "Training", "/training", order=30),),
        required_backend_capabilities=("compute.training",),
        settings_schema={
            "type": "object",
            "properties": {
                "type": {"enum": ["local", "ssh", "runpod", "other"]},
                "gpu_type_ids": {"type": "array", "items": {"type": "string"}},
                "container_disk_gb": {"type": "integer", "minimum": 20},
                "volume_gb": {"type": "integer", "minimum": 20},
                "shutdown_policy": {"enum": ["terminate_after_artifact_verification", "manual"]},
            },
            "additionalProperties": True,
            "secrets_persisted": False,
        },
        web_assets=(WebAssetBundle("spritelab.product_features.training"),),
        api_prefixes=("/training/api",),
    )


def build_plugin() -> ProductPlugin:
    """Required plugin registration function used by the integration layer."""

    return create_plugin()


__all__ = ["PLUGIN_ID", "build_plugin", "create_plugin", "register_training_cli"]

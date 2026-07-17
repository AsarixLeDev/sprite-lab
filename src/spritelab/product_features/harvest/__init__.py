"""Product plugin registration for controlled Harvest acquisition."""

from __future__ import annotations

from collections.abc import Callable, Iterable

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
from spritelab.product_features.harvest.catalog import (
    CatalogEvidenceBinding,
    HarvestSource,
    TrustedCatalogError,
    load_trusted_catalog,
)
from spritelab.product_features.harvest.certification import (
    BackendCapabilityCertificateError,
    BackendCapabilityEvidence,
    load_backend_capability_certificate,
    load_backend_capability_evidence,
)
from spritelab.product_features.harvest.service import HarvestError, HarvestService
from spritelab.product_features.harvest.trusted_backend import (
    BackendFactory,
    CertifiedBackendCapabilities,
    DatasetImportCallback,
    HardenedArchiveAcquisitionBackend,
    HarvestLimits,
)
from spritelab.product_features.harvest.web import create_harvest_router

PLUGIN_ID = "harvest.acquisition"
DatasetImportCallbackFactory = Callable[[ProjectContext], DatasetImportCallback]


def register_harvest_cli(registry: ProductCliRegistry) -> None:
    """Web-only foundation; legacy Harvest CLI ownership remains unchanged."""

    del registry


def create_plugin(
    *,
    sources: Iterable[HarvestSource] | None = None,
    backend_factory: BackendFactory | None = None,
    backend_capabilities: CertifiedBackendCapabilities | None = None,
    backend_capability_evidence: BackendCapabilityEvidence | None = None,
    limits: HarvestLimits | None = None,
    dataset_import_callback: DatasetImportCallback | None = None,
    dataset_import_callback_factory: DatasetImportCallbackFactory | None = None,
    load_repository_capabilities: bool = False,
    probe_resolver: object | None = None,
    probe_transport: object | None = None,
    probe_downloader: object | None = None,
) -> ProductPlugin:
    """Create an injectable plugin without probing or constructing its backend.

    When sources are not injected, each passive service construction reads the
    strict repository-local trusted catalog. Loading never constructs the
    backend or contacts the network.
    """

    injected_catalog = None if sources is None else tuple(sources)
    if load_repository_capabilities and (
        backend_factory is not None or backend_capabilities is not None or backend_capability_evidence is not None
    ):
        raise ValueError("Repository capability loading cannot be combined with injected backend configuration.")
    if dataset_import_callback is not None and dataset_import_callback_factory is not None:
        raise ValueError("Dataset import callback and context-bound factory are mutually exclusive.")

    def repository_configuration(
        context: ProjectContext,
    ) -> tuple[tuple[HarvestSource, ...], BackendCapabilityEvidence | None]:
        return (
            load_trusted_catalog(context.project_root),
            load_backend_capability_evidence(context.project_root),
        )

    def service_with_configuration(
        context: ProjectContext,
    ) -> tuple[
        HarvestService,
        TrustedCatalogError | BackendCapabilityCertificateError | None,
        CertifiedBackendCapabilities | None,
    ]:
        configuration_error: TrustedCatalogError | BackendCapabilityCertificateError | None = None
        if injected_catalog is None:
            try:
                catalog = load_trusted_catalog(context.project_root)
            except TrustedCatalogError as exc:
                catalog = ()
                configuration_error = exc
        else:
            catalog = injected_catalog
        active_factory = backend_factory
        active_capabilities = backend_capabilities
        active_evidence: BackendCapabilityEvidence | None = backend_capability_evidence
        if load_repository_capabilities and configuration_error is None:
            try:
                active_evidence = load_backend_capability_evidence(context.project_root)
                active_capabilities = active_evidence.capabilities if active_evidence is not None else None
            except BackendCapabilityCertificateError as exc:
                active_capabilities = None
                configuration_error = exc
            if active_capabilities is not None and catalog:
                certified = active_capabilities

                def repository_backend_factory() -> HardenedArchiveAcquisitionBackend:
                    return HardenedArchiveAcquisitionBackend(certified)

                active_factory = repository_backend_factory
            else:
                active_capabilities = None
                active_evidence = None
        active_callback = (
            dataset_import_callback_factory(context)
            if dataset_import_callback_factory is not None
            else dataset_import_callback
        )
        harvest_service = HarvestService(
            context.project_root,
            sources=catalog,
            backend_factory=active_factory,
            backend_capabilities=active_capabilities,
            backend_capability_evidence=active_evidence,
            live_configuration_loader=(
                (lambda: repository_configuration(context)) if load_repository_capabilities else None
            ),
            limits=limits,
            dataset_import_callback=active_callback,
            probe_resolver=probe_resolver,
            probe_transport=probe_transport,
            probe_downloader=probe_downloader,
        )
        return harvest_service, configuration_error, active_capabilities

    def service(context: ProjectContext) -> HarvestService:
        return service_with_configuration(context)[0]

    def status_provider(context: ProjectContext) -> ProductResult:
        harvest_service, configuration_error, _active_capabilities = service_with_configuration(context)
        try:
            inventory = harvest_service.inventory()
        except HarvestError as exc:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Harvest inventory is unsafe or unavailable.",
                feature=PLUGIN_ID,
                blockers=(ProductBlocker(exc.code, str(exc)),),
            )
        if configuration_error is not None or not harvest_service.acquisition_configured:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Harvest inventory is available, but no current independently certified acquisition configuration is active.",
                feature=PLUGIN_ID,
                data=inventory,
            )
        return ProductResult(
            ProductStatus.READY,
            "Harvest is ready. Acquisition requires reuse review plus explicit zero-cost and CC0/public-domain authorization.",
            feature=PLUGIN_ID,
            data=inventory,
        )

    def capability_probe(context: ProjectContext) -> tuple[ProductCapability, ...]:
        harvest_service, configuration_error, active_capabilities = service_with_configuration(context)
        try:
            inventory = harvest_service.inventory()
        except HarvestError as exc:
            return (
                ProductCapability(
                    "harvest.inventory",
                    "Harvest inventory",
                    ProductStatus.BLOCKED,
                    str(exc),
                    details={"network_actions": 0, "paths_exposed": False},
                ),
            )
        return (
            ProductCapability(
                "harvest.inventory",
                "Harvest inventory",
                ProductStatus.READY,
                "Immediate repository-local Harvest runs can be inventoried passively.",
                details={
                    "run_count": inventory["run_count"],
                    "legacy_run_count": inventory["legacy_run_count"],
                    "network_actions": 0,
                    "paths_exposed": False,
                },
            ),
            ProductCapability(
                "harvest.acquisition",
                "Controlled acquisition",
                (
                    ProductStatus.READY
                    if configuration_error is None and harvest_service.acquisition_configured
                    else ProductStatus.UNAVAILABLE
                ),
                (
                    "A separately certified adapter is configured; every run remains explicitly authorized."
                    if configuration_error is None and harvest_service.acquisition_configured
                    else "No current independently certified source adapter is configured."
                ),
                details={
                    "explicit_authorization_required": True,
                    "reuse_review_required": True,
                    "backend_capability_identity": (
                        active_capabilities.identity if active_capabilities is not None else None
                    ),
                    "configuration_valid": configuration_error is None,
                    "network_probes": 0,
                },
            ),
        )

    def router_factory(context: ProjectContext) -> object:
        return create_harvest_router(context, service=service(context))

    return ProductPlugin(
        plugin_id=PLUGIN_ID,
        title="Harvest",
        cli_registration=register_harvest_cli,
        status_provider=status_provider,
        capability_probe=capability_probe,
        web_router_factory=router_factory,
        navigation=(WebNavigationItem("harvest", "Harvest", "/harvest", order=15),),
        web_assets=(WebAssetBundle("spritelab.product_features.harvest"),),
        api_prefixes=("/harvest/api",),
    )


def build_plugin() -> ProductPlugin:
    """Load passive repo evidence and activate only an independently certified adapter."""

    return create_plugin(load_repository_capabilities=True)


__all__ = [
    "PLUGIN_ID",
    "BackendCapabilityCertificateError",
    "CatalogEvidenceBinding",
    "CertifiedBackendCapabilities",
    "DatasetImportCallbackFactory",
    "HarvestLimits",
    "HarvestService",
    "HarvestSource",
    "TrustedCatalogError",
    "build_plugin",
    "create_plugin",
    "load_backend_capability_certificate",
    "load_backend_capability_evidence",
    "load_trusted_catalog",
    "register_harvest_cli",
]

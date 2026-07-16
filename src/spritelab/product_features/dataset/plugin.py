"""Dataset intake ProductPlugin export."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from spritelab.product_core import (
    ProductCapability,
    ProductPlugin,
    ProductResult,
    ProductStatus,
    ProjectContext,
    VisionProvider,
    WebAssetBundle,
    WebNavigationItem,
)
from spritelab.product_features.dataset.certification import (
    labeling_audit_verification,
    labeling_capability,
    project_labeling_status,
)
from spritelab.product_features.dataset.cli import register_cli
from spritelab.product_features.dataset.web import (
    build_review_router,
    find_dataset_output,
)

ProviderFactory = Callable[[ProjectContext, Callable[[str], bool] | None], VisionProvider | None]


def create_plugin(
    *,
    provider_factory: ProviderFactory | None = None,
    folder_chooser: Callable[[], str | Path | None] | None = None,
) -> ProductPlugin:
    """Return the self-contained folder intake and exception review feature."""

    return ProductPlugin(
        plugin_id="dataset.intake",
        title="Dataset intake and review",
        cli_registration=lambda registry: register_cli(registry, provider_factory=provider_factory),
        status_provider=_status,
        capability_probe=_capabilities,
        web_router_factory=lambda context: build_review_router(
            context,
            provider_factory=provider_factory,
            folder_chooser=folder_chooser,
        ),
        navigation=(
            WebNavigationItem("dataset", "Dataset", "/dataset", order=20),
            WebNavigationItem("labeling", "Labeling", "/labeling", order=25),
        ),
        required_backend_capabilities=(),
        settings_schema={
            "type": "object",
            "properties": {
                "output_root": {"type": "string", "description": "Latest dataset build output (normally automatic)."},
                "vision_provider": {
                    "type": ["string", "null"],
                    "description": "Optional shared VisionProvider ID; intake remains image-only without it.",
                },
                "hierarchical_labeling": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": False},
                        "profile": {
                            "type": "string",
                            "enum": ["fast_local", "balanced", "high_quality"],
                            "default": "fast_local",
                        },
                        "reference_cohort_size": {
                            "type": "integer",
                            "minimum": 300,
                            "maximum": 500,
                            "default": 400,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        web_assets=(WebAssetBundle("spritelab.product_features.dataset"),),
        api_prefixes=("/dataset/api", "/labeling/api"),
    )


def build_plugin() -> ProductPlugin:
    return create_plugin()


def _status(context: ProjectContext) -> ProductResult:
    projection = project_labeling_status(labeling_audit_verification(context))
    public_labeling = projection.to_public_dict()
    output = find_dataset_output(context)
    if output is None:
        return ProductResult(
            ProductStatus.READY,
            f"Image preparation is available. Automatic image descriptions: {projection.automatic_descriptions_message}",
            feature="dataset",
            data={
                "next_command": "python -m spritelab v3 dataset build <folder>",
                "status_cards": [],
                "labeling": public_labeling,
            },
        )
    try:
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ProductResult(
            ProductStatus.FAILED,
            "The latest dataset result cannot be read.",
            feature="dataset",
            data={},
        )
    status = ProductStatus(str(result.get("status", "FAILED")))
    data = dict(result.get("data", {}))
    return ProductResult(
        status,
        f"{result.get('message', 'Dataset status is available.')}\n\n"
        f"Automatic image descriptions: {projection.automatic_descriptions_message}\n"
        f"Exact object labels: {projection.exact_object_labels_message}",
        feature="dataset",
        data={**data, "output_root": str(output), "labeling": public_labeling},
    )


def _capabilities(context: ProjectContext) -> tuple[ProductCapability, ...]:
    verification = labeling_audit_verification(context)
    backend_capability = labeling_capability(context)
    projection = project_labeling_status(verification)
    semantic_status = (
        ProductStatus.READY if projection.automatic_descriptions_status == "READY" else ProductStatus.UNAVAILABLE
    )
    output = find_dataset_output(context)
    if output is None:
        return (
            ProductCapability(
                "dataset.intake",
                "Folder dataset intake",
                ProductStatus.READY,
                "PNG folders can be imported without a manifest.",
            ),
            ProductCapability(
                "dataset.review",
                "Exception review",
                ProductStatus.NOT_STARTED,
                "Build a dataset to create a prefilled review queue.",
            ),
            ProductCapability(
                "dataset.semantic",
                "Automatic image descriptions",
                semantic_status,
                projection.automatic_descriptions_message,
                details={"authorized_scopes": list(backend_capability.authorized_scopes)},
            ),
            ProductCapability(
                "dataset.exact_labels",
                "Exact object labels",
                ProductStatus.READY if projection.exact_object_labels_status == "READY" else ProductStatus.UNAVAILABLE,
                projection.exact_object_labels_message,
            ),
        )
    result_path = output / "result.json"
    counts = {}
    if result_path.is_file():
        try:
            counts = dict(json.loads(result_path.read_text(encoding="utf-8")).get("data", {}).get("counts", {}))
        except (OSError, json.JSONDecodeError):
            counts = {}
    return (
        ProductCapability("dataset.intake", "Folder dataset intake", ProductStatus.READY),
        ProductCapability(
            "dataset.review",
            "Exception review",
            ProductStatus.NEEDS_REVIEW if int(counts.get("excluded", 0)) else ProductStatus.COMPLETE,
            details={"items": int(counts.get("excluded", 0))},
        ),
        ProductCapability(
            "dataset.semantic",
            "Automatic image descriptions",
            semantic_status,
            projection.automatic_descriptions_message,
            details={
                "labeled": int(counts.get("semantically_labeled", 0)),
                "authorized_scopes": list(backend_capability.authorized_scopes),
            },
        ),
        ProductCapability(
            "dataset.exact_labels",
            "Exact object labels",
            ProductStatus.READY if projection.exact_object_labels_status == "READY" else ProductStatus.UNAVAILABLE,
            projection.exact_object_labels_message,
        ),
        ProductCapability(
            "dataset.calibration",
            "Calibration readiness",
            ProductStatus.READY if projection.calibration_status == "READY" else ProductStatus.UNAVAILABLE,
            "Ready" if projection.calibration_status == "READY" else "More reviewed truth is required.",
        ),
        ProductCapability(
            "dataset.conditioned_freeze",
            "Conditioned dataset freeze",
            ProductStatus.READY if projection.dataset_freeze_status == "AUTHORIZED" else ProductStatus.UNAVAILABLE,
            "Authorized" if projection.dataset_freeze_status == "AUTHORIZED" else "Not authorized by this audit.",
        ),
    )

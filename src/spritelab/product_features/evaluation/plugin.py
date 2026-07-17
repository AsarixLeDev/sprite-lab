"""ProductPlugin registration for evaluation and exploratory generation."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from spritelab.product_features.evaluation.checkpoints import (
    expected_dataset_identity,
    expected_training_view_identity,
)
from spritelab.product_features.evaluation.memorization_display import promotion_integrity_display
from spritelab.product_features.evaluation.service import EvaluationRequest, EvaluationService
from spritelab.product_features.evaluation.web import create_evaluation_router

PLUGIN_ID = "evaluation.playground"


def _service(context: ProjectContext) -> EvaluationService:
    return EvaluationService(
        project_root=context.project_root,
        config=context.config,
        runs_directory=context.runs_directory,
    )


def _project_context() -> ProjectContext:
    from spritelab.v3.config import ProjectConfig

    config = ProjectConfig.load(Path.cwd(), required=False)
    return ProjectContext(
        project_root=config.root,
        config=config.values,
        config_path=config.path,
        runs_directory=config.runs_dir,
    )


def _configure_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", help="Eligible checkpoint identity; defaults to latest complete checkpoint.")
    parser.add_argument("--benchmark", type=Path, help="Benchmark manifest; defaults to project configuration.")
    parser.add_argument("--weights", choices=("live", "ema"), default="ema", help="Checkpoint weights (default: EMA).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display the plan without generation.")
    parser.add_argument("--yes", action="store_true", help="Confirm remote or billable generation if configured.")
    parser.add_argument(
        "--allow-source-results",
        action="store_true",
        help="Include permitted source-level aggregates; private source paths remain hidden.",
    )


def _cli_handler(args: argparse.Namespace, _argv: list[str]) -> ProductResult:
    service = _service(_project_context())
    return service.run(
        EvaluationRequest(
            checkpoint_id=args.checkpoint,
            benchmark=args.benchmark,
            weights=args.weights,
            dry_run=bool(args.dry_run),
            explicit_action=not bool(args.dry_run),
            confirm_billable=bool(args.yes),
            allow_source_results=bool(args.allow_source_results),
        )
    )


def register_cli(registry: ProductCliRegistry) -> None:
    """Replace the reserved v3 eval command through the feature-owned plugin contract."""

    registry.command(
        "eval",
        owner=PLUGIN_ID,
        handler=_cli_handler,
        help="Evaluate an eligible checkpoint with the Standard Sprite Lab benchmark.",
        configure=_configure_cli,
        replace=True,
    )


def status_provider(context: ProjectContext) -> ProductResult:
    service = _service(context)
    catalog = service.catalog
    benchmark_valid, benchmark_message = service._validate_benchmark(service.configured_benchmark)
    integrity = promotion_integrity_display(service.memorization_audit, repository_root=context.project_root)
    dataset_identity = expected_dataset_identity(context.config)
    view_identity = expected_training_view_identity(context.config)
    identities_configured = bool(dataset_identity and view_identity)
    checkpoint_bound = bool(catalog.eligible) and identities_configured
    ready = checkpoint_bound and benchmark_valid
    if not identities_configured:
        message = "Configure the active training dataset and view identities before evaluation."
    elif not catalog.eligible:
        message = "Waiting for a certified trained model bound to the active dataset and view."
    elif not benchmark_valid:
        message = benchmark_message
    else:
        message = "Evaluation is ready for an explicit start action."
    return ProductResult(
        status=ProductStatus.READY if ready else ProductStatus.BLOCKED,
        feature="evaluation",
        message=message,
        data={
            "eligible_checkpoint_count": len(catalog.eligible),
            "benchmark": benchmark_message,
            "training_identity": {
                "dataset_identity": dataset_identity,
                "view_identity": view_identity,
                "bound": checkpoint_bound,
            },
            "promotion": integrity,
            "certification_state": "Promotion certification pending",
            "generation_on_page_open": False,
        },
    )


def capability_probe(context: ProjectContext) -> tuple[ProductCapability, ...]:
    service = _service(context)
    catalog = service.catalog
    benchmark_valid, benchmark_message = service._validate_benchmark(service.configured_benchmark)
    dataset_identity = expected_dataset_identity(context.config)
    view_identity = expected_training_view_identity(context.config)
    identities_configured = bool(dataset_identity and view_identity)
    checkpoint_ready = bool(catalog.eligible) and identities_configured
    checkpoint_message = (
        f"{len(catalog.eligible)} eligible checkpoint(s) bound to the active dataset and view."
        if checkpoint_ready
        else "Active training dataset and view identities must be configured."
        if not identities_configured
        else "No eligible checkpoint is bound to the active dataset and view."
    )
    return (
        ProductCapability(
            "evaluation.checkpoint_selection",
            "Checkpoint selection",
            ProductStatus.READY if checkpoint_ready else ProductStatus.BLOCKED,
            checkpoint_message,
        ),
        ProductCapability(
            "evaluation.benchmark",
            "Standard Sprite Lab benchmark",
            ProductStatus.READY if benchmark_valid else ProductStatus.BLOCKED,
            benchmark_message,
        ),
        ProductCapability(
            "evaluation.playground",
            "Exploratory prompt playground",
            ProductStatus.READY if checkpoint_ready else ProductStatus.BLOCKED,
            (
                "The local typed generator is ready for an explicit exploratory action."
                if checkpoint_ready
                else "The local generator is installed; an eligible active-dataset checkpoint is required."
            ),
        ),
        ProductCapability(
            "evaluation.promotion_display",
            "Promotion gate display",
            ProductStatus.BLOCKED,
            str(
                promotion_integrity_display(
                    service.memorization_audit,
                    repository_root=context.project_root,
                )["message"]
            ),
        ),
    )


def build_plugin() -> ProductPlugin:
    """Return the feature-owned ProductPlugin without editing a central registry."""

    return ProductPlugin(
        plugin_id=PLUGIN_ID,
        title="Evaluation",
        cli_registration=register_cli,
        status_provider=status_provider,
        capability_probe=capability_probe,
        web_router_factory=create_evaluation_router,
        navigation=(WebNavigationItem("evaluation", "Evaluation", "/evaluation", order=40),),
        required_backend_capabilities=("evaluation.score_suite", "generation.typed"),
        settings_schema={
            "type": "object",
            "properties": {
                "default_weights": {"enum": ["live", "ema"], "default": "ema"},
                "playground_image_count": {"type": "integer", "minimum": 1, "maximum": 16, "default": 4},
            },
            "additionalProperties": False,
        },
        web_assets=(
            WebAssetBundle(
                package="spritelab.product_features.evaluation",
                templates="templates",
                static="static",
            ),
        ),
        api_prefixes=("/evaluation/api",),
    )

"""User-facing evaluation dashboard and exploratory generation playground."""

from spritelab.product_features.evaluation.checkpoints import discover_checkpoint_candidates
from spritelab.product_features.evaluation.dashboard import (
    IncompatibleMetricDefinitions,
    build_dashboard,
    compare_evaluations,
    filter_gallery,
)
from spritelab.product_features.evaluation.local_generator import (
    LocalCheckpointPlaygroundGenerator,
    LocalPlaygroundGenerationError,
)
from spritelab.product_features.evaluation.memorization_display import (
    PROMOTION_INTEGRITY_MESSAGE,
    MemorizationDisplayState,
    memorization_display,
    promotion_integrity_display,
)
from spritelab.product_features.evaluation.models import (
    CheckpointAvailability,
    CheckpointCandidate,
    CheckpointCatalog,
)
from spritelab.product_features.evaluation.playground import (
    EXPLORATORY_SCOPE,
    GeneratedAsset,
    GenerationCancelledError,
    GenerationRequest,
    GenerationSafetyError,
    PlaygroundService,
)
from spritelab.product_features.evaluation.plugin import build_plugin, register_cli
from spritelab.product_features.evaluation.service import EvaluationRequest, EvaluationService
from spritelab.product_features.evaluation.web import create_evaluation_router

__all__ = [
    "EXPLORATORY_SCOPE",
    "PROMOTION_INTEGRITY_MESSAGE",
    "CheckpointAvailability",
    "CheckpointCandidate",
    "CheckpointCatalog",
    "EvaluationRequest",
    "EvaluationService",
    "GeneratedAsset",
    "GenerationCancelledError",
    "GenerationRequest",
    "GenerationSafetyError",
    "IncompatibleMetricDefinitions",
    "LocalCheckpointPlaygroundGenerator",
    "LocalPlaygroundGenerationError",
    "MemorizationDisplayState",
    "PlaygroundService",
    "build_dashboard",
    "build_plugin",
    "compare_evaluations",
    "create_evaluation_router",
    "discover_checkpoint_candidates",
    "filter_gallery",
    "memorization_display",
    "promotion_integrity_display",
    "register_cli",
]

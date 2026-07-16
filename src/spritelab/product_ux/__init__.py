"""Novice-facing copy, onboarding, accessibility, and acceptance contracts."""

from spritelab.product_ux.acceptance import USER_JOURNEYS, run_synthetic_journey, user_journey_results_payload
from spritelab.product_ux.accessibility import (
    ACCESSIBILITY_REQUIREMENTS,
    MINIMUM_TARGET_SIZE_CSS_PIXELS,
    accessibility_checklist_payload,
)
from spritelab.product_ux.copy_catalog import COPY_CATALOG, ProductCopy, copy_catalog_payload, copy_for
from spritelab.product_ux.launchers import LAUNCH_COMMAND, LauncherResult, generate_project_launchers
from spritelab.product_ux.onboarding import ONBOARDING_STEPS, first_launch_contract, onboarding_contract_payload

__all__ = [
    "ACCESSIBILITY_REQUIREMENTS",
    "COPY_CATALOG",
    "LAUNCH_COMMAND",
    "MINIMUM_TARGET_SIZE_CSS_PIXELS",
    "ONBOARDING_STEPS",
    "USER_JOURNEYS",
    "LauncherResult",
    "ProductCopy",
    "accessibility_checklist_payload",
    "copy_catalog_payload",
    "copy_for",
    "first_launch_contract",
    "generate_project_launchers",
    "onboarding_contract_payload",
    "run_synthetic_journey",
    "user_journey_results_payload",
]

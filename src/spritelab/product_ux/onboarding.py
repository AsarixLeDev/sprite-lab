"""Stable onboarding presentation contracts for product surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from spritelab.product_ux.copy_catalog import copy_for


@dataclass(frozen=True)
class OnboardingAction:
    action_id: str
    label: str
    primary: bool = True


@dataclass(frozen=True)
class OnboardingStep:
    number: int
    key: str
    title: str
    instruction: str
    actions: tuple[OnboardingAction, ...]
    phase: str
    asks_for_internal_path: bool = False
    asks_for_hash: bool = False

    @property
    def primary_action(self) -> OnboardingAction:
        primary = [action for action in self.actions if action.primary]
        if len(primary) != 1:
            raise ValueError(f"Onboarding step {self.key!r} must have exactly one primary action.")
        return primary[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ONBOARDING_STEPS = (
    OnboardingStep(
        number=1,
        key="choose_folder",
        title="Choose an image folder",
        instruction="Pick the folder that contains the images you want Sprite Lab to learn from.",
        actions=(OnboardingAction("choose_image_folder", "Choose image folder"),),
        phase="Build a dataset",
    ),
    OnboardingStep(
        number=2,
        key="check_source_license",
        title="Check source and license",
        instruction="Confirm where the images came from and that you are allowed to use them.",
        actions=(OnboardingAction("check_source_license", "Check source and license"),),
        phase="Build a dataset",
    ),
    OnboardingStep(
        number=3,
        key="build_dataset",
        title="Build the dataset",
        instruction="Let Sprite Lab check and prepare copies of the usable images.",
        actions=(OnboardingAction("build_dataset", "Build dataset"),),
        phase="Build a dataset",
    ),
    OnboardingStep(
        number=4,
        key="train",
        title="Train",
        instruction="Review the time and computer-use estimate before training starts.",
        actions=(OnboardingAction("review_training_plan", "Review training plan"),),
        phase="Train",
    ),
    OnboardingStep(
        number=5,
        key="evaluate_generate",
        title="Evaluate and try the model",
        instruction="Check the model with examples, then try your own prompt.",
        actions=(OnboardingAction("evaluate_model", "Evaluate model"),),
        phase="Evaluate and try",
    ),
)


def first_launch_contract() -> dict[str, Any]:
    """Return the single-action page shown before a project exists."""

    welcome = copy_for("welcome")
    step = ONBOARDING_STEPS[0]
    return {
        "schema_version": "spritelab.first-launch.v1",
        "title": welcome.title,
        "body": list(welcome.body),
        "primary_action": asdict(step.primary_action),
        "actions": [asdict(action) for action in step.actions],
        "next_action": welcome.next_action,
        "starts_with_project_configuration": False,
        "asks_for_internal_path": False,
        "asks_for_hash": False,
    }


def onboarding_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": "spritelab.onboarding.v1",
        "first_launch": first_launch_contract(),
        "dataset_setup_steps": [step.to_dict() for step in ONBOARDING_STEPS[:3]],
        "next_steps": [step.to_dict() for step in ONBOARDING_STEPS[3:]],
    }

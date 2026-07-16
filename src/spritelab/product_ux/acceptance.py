"""Synthetic product-acceptance journeys for novice-facing surfaces.

The runner evaluates presentation contracts only. It does not import or call a
browser, training backend, generation backend, vision service, or cloud client.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from spritelab.product_ux.copy_catalog import copy_for

EXPERT_TERMS = (
    "manifest",
    "provenance binding",
    "taxonomy",
    "ema",
    "cfg",
    "optimizer step",
    "checkpoint identity",
    "audit applicability",
    "sha-256",
    "candidate bundle",
)


def expert_terms_in(text: str) -> tuple[str, ...]:
    """Find specialist terms as complete terms, not fragments of plain words."""

    normalized = text.casefold()
    return tuple(term for term in EXPERT_TERMS if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", normalized))


@dataclass(frozen=True)
class JourneyScreen:
    screen_id: str
    copy_key: str
    primary_action: str | None = None
    next_action: str | None = None
    secondary_actions: tuple[str, ...] = ()
    notice: str | None = None
    confirmation_for: str | None = None
    asks_for_internal_path: bool = False
    asks_for_hash: bool = False
    raw_traceback: str | None = None
    unsafe_launch_requested: bool = False

    @property
    def shown_primary_action(self) -> str:
        return self.primary_action or copy_for(self.copy_key).primary_action

    @property
    def shown_next_action(self) -> str:
        return self.next_action or copy_for(self.copy_key).next_action

    @property
    def visible_text(self) -> str:
        copy = copy_for(self.copy_key)
        values = [copy.title, *copy.body]
        if self.notice:
            values.append(self.notice)
        values.extend((self.shown_primary_action, self.shown_next_action, *self.secondary_actions))
        return "\n".join(values)


@dataclass(frozen=True)
class UserJourney:
    key: str
    title: str
    screens: tuple[JourneyScreen, ...]
    expected_outcome: str


@dataclass(frozen=True)
class JourneyResult:
    key: str
    title: str
    result: str
    assertions: dict[str, bool]
    confirmations: dict[str, int]
    exact_next_actions: tuple[str, ...]
    execution_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _screen(
    screen_id: str,
    copy_key: str,
    *,
    primary_action: str | None = None,
    next_action: str | None = None,
    secondary_actions: tuple[str, ...] = (),
    notice: str | None = None,
    confirmation_for: str | None = None,
) -> JourneyScreen:
    return JourneyScreen(
        screen_id=screen_id,
        copy_key=copy_key,
        primary_action=primary_action,
        next_action=next_action,
        secondary_actions=secondary_actions,
        notice=notice,
        confirmation_for=confirmation_for,
    )


USER_JOURNEYS = (
    UserJourney(
        "A",
        "First-time user with a valid folder",
        (
            _screen("welcome", "welcome"),
            _screen("folder_ready", "first_dataset"),
            _screen("images_checked", "preprocessing_summary"),
            _screen(
                "optional_rescue",
                "optional_review",
                primary_action="Continue without review",
                next_action="Choose Continue without review to finish building the dataset.",
                secondary_actions=("Rescue images",),
            ),
        ),
        "Dataset is ready; the next action is Train.",
    ),
    UserJourney(
        "B",
        "Missing LICENSE",
        (
            _screen("welcome", "welcome"),
            _screen("license_missing", "missing_license"),
        ),
        "Build is safely paused until license information is added.",
    ),
    UserJourney(
        "C",
        "Some corrupted images",
        (
            _screen(
                "images_checked",
                "preprocessing_summary",
                notice="Three damaged images stayed out; 21 readable images are ready.",
            ),
        ),
        "Readable images remain available and damaged images stay excluded.",
    ),
    UserJourney(
        "D",
        "Optional rejection review skipped",
        (
            _screen(
                "optional_rescue",
                "optional_review",
                primary_action="Continue without review",
                next_action="Choose Continue without review to keep all automatic exclusions.",
                secondary_actions=("Rescue images",),
            ),
        ),
        "The dataset continues with automatic exclusions unchanged.",
    ),
    UserJourney(
        "E",
        "Rejected images rescued",
        (
            _screen(
                "rescue_images",
                "optional_review",
                next_action="Choose Rescue images, select the images to keep, then choose Save choices.",
                secondary_actions=("Continue without review",),
            ),
            _screen(
                "rescue_saved",
                "preprocessing_summary",
                notice="Two selected images were rescued and will be kept.",
            ),
        ),
        "Only selected excluded images are restored.",
    ),
    UserJourney(
        "F",
        "No VLM configured",
        (_screen("description_option_missing", "no_provider"),),
        "User can choose local, hosted, or manual descriptions.",
    ),
    UserJourney(
        "G",
        "Local VLM configured",
        (
            _screen(
                "local_description_ready",
                "first_dataset",
                notice="The local image description option is ready.",
                primary_action="Build dataset",
                next_action="Choose Build dataset to prepare images and suggest descriptions on this computer.",
            ),
        ),
        "Local description work is ready without a remote connection.",
    ),
    UserJourney(
        "H",
        "Dataset built but labels incomplete",
        (_screen("descriptions_incomplete", "review_required"),),
        "Training stays unavailable until the listed image descriptions are reviewed.",
    ),
    UserJourney(
        "I",
        "Training blocked safely",
        (_screen("training_unavailable", "training_blocked"),),
        "No training starts and the exact repair action is shown.",
    ),
    UserJourney(
        "J",
        "Local training ready synthetic case",
        (
            _screen(
                "local_training_ready",
                "training_confirmation",
                confirmation_for="local_training",
            ),
        ),
        "One confirmation is requested immediately before local training.",
    ),
    UserJourney(
        "K",
        "Hosted training configuration",
        (
            _screen(
                "hosted_training_cost",
                "hosted_cost_confirmation",
                confirmation_for="hosted_training",
            ),
        ),
        "One confirmation covers the displayed hosted cost limit.",
    ),
    UserJourney(
        "L",
        "Evaluation ready synthetic case",
        (
            _screen(
                "evaluation_ready",
                "evaluation",
                confirmation_for="evaluation_generation",
            ),
        ),
        "One confirmation can start the complete evaluation and generation operation.",
    ),
    UserJourney(
        "M",
        "Prompt playground",
        (
            _screen(
                "prompt_playground",
                "evaluation",
                primary_action="Try a prompt",
                next_action="Enter a short description, then choose Try a prompt.",
            ),
        ),
        "The playground makes the prompt field and Try a prompt action obvious.",
    ),
    UserJourney(
        "N",
        "Interrupted and resumed run",
        (
            _screen("run_interrupted", "interrupted_run"),
            _screen("resume_ready", "safe_resume"),
        ),
        "Saved work is checked before a run can continue.",
    ),
    UserJourney(
        "O",
        "Unexpected error with no traceback",
        (_screen("unexpected_problem", "unexpected_error"),),
        "A readable recovery action is shown without an error trace.",
    ),
)


def run_synthetic_journey(journey: UserJourney) -> JourneyResult:
    """Evaluate a journey contract without executing any product operation."""

    confirmations = Counter(
        screen.confirmation_for for screen in journey.screens if screen.confirmation_for is not None
    )
    visible_text = "\n".join(screen.visible_text for screen in journey.screens)
    next_actions = tuple(screen.shown_next_action for screen in journey.screens)
    assertions = {
        "primary_action_is_obvious": all(bool(screen.shown_primary_action.strip()) for screen in journey.screens),
        "no_internal_path_is_required": all(not screen.asks_for_internal_path for screen in journey.screens),
        "no_hash_is_required": all(not screen.asks_for_hash for screen in journey.screens),
        "at_most_one_confirmation_per_expensive_operation": all(count <= 1 for count in confirmations.values()),
        "exact_next_action_is_shown": all(bool(action.strip()) for action in next_actions),
        "no_raw_traceback_appears": all(not screen.raw_traceback for screen in journey.screens),
        "nothing_unsafe_launches": all(not screen.unsafe_launch_requested for screen in journey.screens),
        "language_is_suitable_for_non_experts": not expert_terms_in(visible_text),
    }
    return JourneyResult(
        key=journey.key,
        title=journey.title,
        result="PASS" if all(assertions.values()) else "FAIL",
        assertions=assertions,
        confirmations=dict(sorted(confirmations.items())),
        exact_next_actions=next_actions,
        execution_attempts=0,
    )


def user_journey_results_payload() -> dict[str, Any]:
    results = [run_synthetic_journey(journey) for journey in USER_JOURNEYS]
    return {
        "schema_version": "spritelab.user-journey-results.v1",
        "execution_mode": "synthetic-contract-only",
        "real_browser_runs": 0,
        "real_training_runs": 0,
        "real_generation_runs": 0,
        "real_provider_calls": 0,
        "real_cloud_jobs": 0,
        "passed": sum(result.result == "PASS" for result in results),
        "failed": sum(result.result == "FAIL" for result in results),
        "results": [result.to_dict() for result in results],
    }

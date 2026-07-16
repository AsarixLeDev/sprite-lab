from __future__ import annotations

import inspect
import re

import pytest

from spritelab.product_ux.acceptance import (
    USER_JOURNEYS,
    run_synthetic_journey,
    user_journey_results_payload,
)


@pytest.mark.parametrize("journey", USER_JOURNEYS, ids=lambda journey: f"journey-{journey.key}")
def test_synthetic_user_journey_acceptance(journey) -> None:
    result = run_synthetic_journey(journey)
    assert result.result == "PASS"
    assert result.assertions == {
        "primary_action_is_obvious": True,
        "no_internal_path_is_required": True,
        "no_hash_is_required": True,
        "at_most_one_confirmation_per_expensive_operation": True,
        "exact_next_action_is_shown": True,
        "no_raw_traceback_appears": True,
        "nothing_unsafe_launches": True,
        "language_is_suitable_for_non_experts": True,
    }
    assert result.execution_attempts == 0
    assert all(count <= 1 for count in result.confirmations.values())
    assert all(action.strip() for action in result.exact_next_actions)


def test_all_fifteen_required_journeys_are_present_in_order() -> None:
    assert [journey.key for journey in USER_JOURNEYS] == list("ABCDEFGHIJKLMNO")
    assert [journey.title for journey in USER_JOURNEYS] == [
        "First-time user with a valid folder",
        "Missing LICENSE",
        "Some corrupted images",
        "Optional rejection review skipped",
        "Rejected images rescued",
        "No VLM configured",
        "Local VLM configured",
        "Dataset built but labels incomplete",
        "Training blocked safely",
        "Local training ready synthetic case",
        "Hosted training configuration",
        "Evaluation ready synthetic case",
        "Prompt playground",
        "Interrupted and resumed run",
        "Unexpected error with no traceback",
    ]


def test_first_user_action_is_choose_image_folder() -> None:
    first_screen = USER_JOURNEYS[0].screens[0]
    assert first_screen.shown_primary_action == "Choose image folder"


def test_no_journey_requires_a_literal_internal_location_or_fingerprint() -> None:
    text = "\n".join(screen.visible_text for journey in USER_JOURNEYS for screen in journey.screens)
    assert not re.search(r"[A-Za-z]:\\", text)
    assert "/tmp/" not in text
    assert "sha-256" not in text.casefold()


def test_only_expensive_synthetic_operations_request_confirmation() -> None:
    results = {result.key: result for result in map(run_synthetic_journey, USER_JOURNEYS)}
    assert results["J"].confirmations == {"local_training": 1}
    assert results["K"].confirmations == {"hosted_training": 1}
    assert results["L"].confirmations == {"evaluation_generation": 1}
    assert all(not result.confirmations for key, result in results.items() if key not in {"J", "K", "L"})


def test_optional_review_can_be_skipped_or_used_to_rescue_images() -> None:
    skipped = USER_JOURNEYS[3].screens[0]
    rescued = USER_JOURNEYS[4].screens[0]
    assert skipped.shown_primary_action == "Continue without review"
    assert "Rescue images" in skipped.secondary_actions
    assert rescued.shown_primary_action == "Rescue images"
    assert "select the images to keep" in rescued.shown_next_action


def test_interrupted_run_is_checked_before_resume() -> None:
    interrupted = USER_JOURNEYS[13]
    assert [screen.copy_key for screen in interrupted.screens] == ["interrupted_run", "safe_resume"]
    assert interrupted.screens[-1].shown_primary_action == "Continue run"


def test_unexpected_error_never_contains_a_raw_traceback() -> None:
    error_screen = USER_JOURNEYS[14].screens[0]
    assert error_screen.raw_traceback is None
    assert "Traceback (most recent call last)" not in error_screen.visible_text
    assert error_screen.shown_primary_action == "Try again"


def test_acceptance_runner_has_no_real_execution_dependencies() -> None:
    source = inspect.getsource(run_synthetic_journey)
    assert "subprocess" not in source
    assert "webbrowser" not in source
    assert "requests" not in source
    assert "torch" not in source


def test_results_payload_records_zero_real_operations() -> None:
    payload = user_journey_results_payload()
    assert payload["passed"] == 15
    assert payload["failed"] == 0
    assert payload["execution_mode"] == "synthetic-contract-only"
    assert payload["real_browser_runs"] == 0
    assert payload["real_training_runs"] == 0
    assert payload["real_generation_runs"] == 0
    assert payload["real_provider_calls"] == 0
    assert payload["real_cloud_jobs"] == 0

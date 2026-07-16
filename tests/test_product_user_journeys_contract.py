from __future__ import annotations

import inspect
import json
import os
import stat
from pathlib import Path

from spritelab.product_ux import (
    ACCESSIBILITY_REQUIREMENTS,
    COPY_CATALOG,
    LAUNCH_COMMAND,
    MINIMUM_TARGET_SIZE_CSS_PIXELS,
    ONBOARDING_STEPS,
    acceptance,
    accessibility_checklist_payload,
    copy_catalog,
    copy_catalog_payload,
    first_launch_contract,
    generate_project_launchers,
    onboarding_contract_payload,
    user_journey_results_payload,
)
from spritelab.product_ux.accessibility import ERROR_REGION, FOCUS_ORDER, PROGRESS_LIVE_REGION

REQUIRED_COPY_KEYS = {
    "welcome",
    "first_dataset",
    "missing_source",
    "missing_license",
    "preprocessing_summary",
    "optional_review",
    "training_blocked",
    "training_confirmation",
    "hosted_cost_confirmation",
    "evaluation",
    "memorization_block",
    "review_required",
    "unexpected_error",
    "interrupted_run",
    "safe_resume",
    "no_data",
    "no_provider",
    "provider_authentication",
    "remote_connection_loss",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_copy_catalog_covers_every_required_product_state() -> None:
    assert set(COPY_CATALOG) == REQUIRED_COPY_KEYS
    assert all(entry.next_action and entry.primary_action for entry in COPY_CATALOG.values())


def test_visible_copy_uses_plain_language_first() -> None:
    visible = "\n".join(entry.plain_text() for entry in COPY_CATALOG.values())
    assert acceptance.expert_terms_in(visible) == ()


def test_technical_details_are_optional_and_expandable() -> None:
    detailed_entries = [entry for entry in COPY_CATALOG.values() if entry.technical_details]
    assert detailed_entries
    assert all(entry.details_label == "Technical details" for entry in detailed_entries)
    assert all(entry.technical_details not in entry.plain_text() for entry in detailed_entries)


def test_required_plain_language_examples_are_used() -> None:
    assert (
        COPY_CATALOG["training_blocked"].body[0]
        == "Training is temporarily unavailable because its safety checks need repair."
    )
    assert "too close to images used for training" in COPY_CATALOG["memorization_block"].body[0]
    assert "no error trace is shown" in str(COPY_CATALOG["unexpected_error"].technical_details)


def test_rejection_review_is_rescue_images() -> None:
    review = COPY_CATALOG["optional_review"]
    assert review.title == "Rescue images"
    assert review.primary_action == "Rescue images"
    assert review.body == (
        "Sprite Lab excluded these images automatically.",
        "You only need to select images that should be kept. Everything else can remain excluded.",
    )
    assert "classifier" not in review.plain_text().casefold()
    assert "negative" not in review.plain_text().casefold()


def test_first_launch_has_one_primary_action_and_no_configuration_gate() -> None:
    page = first_launch_contract()
    assert page["primary_action"]["label"] == "Choose image folder"
    assert page["actions"] == [page["primary_action"]]
    assert page["starts_with_project_configuration"] is False
    assert page["asks_for_internal_path"] is False
    assert page["asks_for_hash"] is False


def test_onboarding_follows_the_five_required_steps() -> None:
    assert [step.number for step in ONBOARDING_STEPS] == [1, 2, 3, 4, 5]
    assert [step.title for step in ONBOARDING_STEPS] == [
        "Choose an image folder",
        "Check source and license",
        "Build the dataset",
        "Train",
        "Evaluate and try the model",
    ]
    assert all(step.primary_action for step in ONBOARDING_STEPS)
    assert all(not step.asks_for_internal_path and not step.asks_for_hash for step in ONBOARDING_STEPS)


def test_copy_and_onboarding_payloads_are_json_ready() -> None:
    copy_payload = json.loads(json.dumps(copy_catalog_payload()))
    onboarding_payload = json.loads(json.dumps(onboarding_contract_payload()))
    assert copy_payload["schema_version"] == "spritelab.product-copy.v1"
    assert onboarding_payload["first_launch"]["primary_action"]["label"] == "Choose image folder"


def test_committed_contract_artifacts_match_the_reusable_payloads() -> None:
    experiment = REPO_ROOT / "experiments" / "v3_novice_ux_v1"
    expected = {
        "copy_catalog.json": copy_catalog_payload(),
        "onboarding_contract.json": onboarding_contract_payload(),
        "accessibility_checklist.json": accessibility_checklist_payload(),
        "user_journey_results.json": user_journey_results_payload(),
    }
    for filename, payload in expected.items():
        normalized_payload = json.loads(json.dumps(payload))
        assert json.loads((experiment / filename).read_text(encoding="utf-8")) == normalized_payload


def test_copy_package_does_not_reimplement_product_state_logic() -> None:
    source = inspect.getsource(copy_catalog) + inspect.getsource(acceptance)
    assert "spritelab.v3.status" not in source
    assert "build_project_state" not in source
    assert "StageStatus" not in source


def test_launchers_are_project_local_and_invoke_only_the_v3_product(tmp_path) -> None:
    results = generate_project_launchers(tmp_path)
    assert {result.status for result in results} == {"CREATED"}
    assert {path.name for path in tmp_path.iterdir()} == {"Start Sprite Lab.cmd", "start-sprite-lab.sh"}
    for result in results:
        assert result.path.parent == tmp_path.resolve()
        assert LAUNCH_COMMAND in result.path.read_text(encoding="utf-8")
    assert 'cd /d "%~dp0"' in (tmp_path / "Start Sprite Lab.cmd").read_text(encoding="utf-8")
    shell_launcher = tmp_path / "start-sprite-lab.sh"
    assert 'cd "$SCRIPT_DIR"' in shell_launcher.read_text(encoding="utf-8")
    if os.name != "nt":
        assert shell_launcher.stat().st_mode & stat.S_IXUSR


def test_existing_launchers_are_preserved_with_an_explicit_result(tmp_path) -> None:
    existing = tmp_path / "Start Sprite Lab.cmd"
    existing.write_text("user-owned launcher\n", encoding="utf-8")
    results = {result.path.name: result for result in generate_project_launchers(tmp_path)}
    assert results[existing.name].status == "PRESERVED"
    assert "preserved" in results[existing.name].message.casefold()
    assert existing.read_text(encoding="utf-8") == "user-owned launcher\n"
    assert results["start-sprite-lab.sh"].status == "CREATED"


def test_launcher_generation_rejects_a_missing_project_folder(tmp_path) -> None:
    missing = tmp_path / "not-created"
    try:
        generate_project_launchers(missing)
    except NotADirectoryError as exc:
        assert str(missing.resolve()) in str(exc)
    else:
        raise AssertionError("A missing project folder must be rejected.")


def test_accessibility_contract_covers_every_required_area() -> None:
    assert {requirement.key for requirement in ACCESSIBILITY_REQUIREMENTS} == {
        "keyboard_only",
        "focus_order",
        "visible_focus",
        "screen_reader_labels",
        "progress_announcements",
        "text_plus_color",
        "reduced_motion",
        "readable_errors",
        "minimum_target_size",
        "no_hover_only",
        "non_expert_language",
    }
    assert all(requirement.acceptance for requirement in ACCESSIBILITY_REQUIREMENTS)


def test_accessibility_semantics_are_testable_and_unambiguous() -> None:
    assert FOCUS_ORDER == ("page_title", "status_message", "primary_action", "secondary_actions", "help")
    assert PROGRESS_LIVE_REGION == {"role": "status", "aria-live": "polite", "aria-atomic": "true"}
    assert ERROR_REGION["role"] == "alert"
    assert MINIMUM_TARGET_SIZE_CSS_PIXELS == 44
    payload = accessibility_checklist_payload()
    assert payload["result"] == "PASS"
    assert all(item["result"] == "PASS" for item in payload["requirements"])

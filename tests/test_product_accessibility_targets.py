import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_CSS = (ROOT / "src/spritelab/product_web/static/app.css").read_text(encoding="utf-8")
BASE_TEMPLATE = (ROOT / "src/spritelab/product_web/templates/base.html").read_text(encoding="utf-8")
TRAINING_TEMPLATE = (ROOT / "src/spritelab/product_features/training/templates/training.html").read_text(
    encoding="utf-8"
)
DATASET_TEMPLATE = (ROOT / "src/spritelab/product_features/dataset/templates/dataset.html").read_text(encoding="utf-8")
PROVIDERS_TEMPLATE = (ROOT / "src/spritelab/product_features/providers/templates/providers.html").read_text(
    encoding="utf-8"
)
EVALUATION_TEMPLATE = (ROOT / "src/spritelab/product_features/evaluation/templates/evaluation.html").read_text(
    encoding="utf-8"
)
EVALUATION_A11Y_CSS = (ROOT / "src/spritelab/product_features/evaluation/static/evaluation-a11y.css").read_text(
    encoding="utf-8"
)


def _rule(css: str, selector: str) -> str:
    match = re.search(rf"(?:^|\n){re.escape(selector)}\s*\{{([^}}]+)}}", css)
    assert match is not None, f"missing CSS rule for {selector}"
    return match.group(1)


def test_shared_primary_and_disabled_button_targets_are_at_least_44px() -> None:
    shared_button = _rule(APP_CSS, ".button")
    assert "min-width: 44px" in shared_button
    assert "min-height: 44px" in shared_button
    assert 'class="button primary" id="start"' in TRAINING_TEMPLATE
    assert 'id="start" type="button" disabled' in TRAINING_TEMPLATE

    disabled_button = _rule(APP_CSS, ".button:disabled")
    assert "width" not in disabled_button
    assert "height" not in disabled_button


def test_primary_dataset_provider_compute_and_evaluation_actions_have_target_contract() -> None:
    assert 'class="button primary" id="choose-folder"' in DATASET_TEMPLATE
    assert 'class="button primary" id="build-dataset"' in DATASET_TEMPLATE
    for control_id in ("detect", "save", "test"):
        assert f'id="{control_id}"' in PROVIDERS_TEMPLATE
        assert (
            'class="button '
            in PROVIDERS_TEMPLATE.split(f'id="{control_id}"', maxsplit=1)[0].rsplit("<button", maxsplit=1)[1]
        )
    for control_id in ("save-compute", "test-compute"):
        assert (
            'class="button '
            in TRAINING_TEMPLATE.split(f'id="{control_id}"', maxsplit=1)[0].rsplit("<button", maxsplit=1)[1]
        )

    assert 'id="start-evaluation" class="primary"' in EVALUATION_TEMPLATE
    assert "button,select,input,textarea,a.button-link{min-height:44px}" in EVALUATION_A11Y_CSS


def test_icon_controls_and_narrow_navigation_targets_are_at_least_44px() -> None:
    icon_button = _rule(APP_CSS, ".icon-button")
    for declaration in ("width: 44px", "min-width: 44px", "height: 44px", "min-height: 44px"):
        assert declaration in icon_button

    primary_navigation = _rule(APP_CSS, ".primary-nav a")
    assert "min-height: 44px" in primary_navigation
    assert 'aria-controls="primary-sidebar"' in BASE_TEMPLATE
    assert 'aria-label="Switch color theme"' in BASE_TEMPLATE
    assert 'aria-label="Notifications"' in BASE_TEMPLATE
    assert 'aria-label="Close notifications"' in BASE_TEMPLATE
    assert ".dialog-close{min-width:44px;min-height:44px}" in EVALUATION_A11Y_CSS


def test_zoom_focus_and_responsive_no_overflow_contracts_are_preserved() -> None:
    assert 'name="viewport" content="width=device-width, initial-scale=1"' in BASE_TEMPLATE
    assert "html { min-width: 300px" in APP_CSS
    assert "@media (max-width: 760px)" in APP_CSS
    assert "@media (max-width: 430px)" in APP_CSS
    assert ".main-content { margin-left: 0;" in APP_CSS
    assert ".recommendation .button { grid-column: 1 / -1; width: 100%; }" in APP_CSS
    assert ":focus-visible { outline: 3px solid var(--brand); outline-offset: 3px;" in APP_CSS
    assert "@media (forced-colors: active)" in APP_CSS
    assert "@media (prefers-reduced-motion: reduce)" in APP_CSS

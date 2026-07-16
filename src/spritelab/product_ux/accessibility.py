"""Accessibility requirements shared by every Sprite Lab product surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AccessibilityRequirement:
    key: str
    requirement: str
    acceptance: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


ACCESSIBILITY_REQUIREMENTS = (
    AccessibilityRequirement(
        "keyboard_only",
        "Every task can be completed with a keyboard alone.",
        "All interactive controls can be reached and used with Tab, Shift+Tab, Enter, Space, and arrow keys.",
    ),
    AccessibilityRequirement(
        "focus_order",
        "Focus follows the visible reading and task order.",
        "Focus moves from page title to status, primary action, secondary actions, and help without unexpected jumps.",
    ),
    AccessibilityRequirement(
        "visible_focus",
        "Keyboard focus is always visible.",
        "Every interactive control has a high-contrast focus indicator that is not hidden by nearby content.",
    ),
    AccessibilityRequirement(
        "screen_reader_labels",
        "Controls and images have meaningful screen-reader labels.",
        "Every control has an accessible name; informative images have useful alternative text.",
    ),
    AccessibilityRequirement(
        "progress_announcements",
        "Progress changes are announced without stealing focus.",
        "A polite live region announces the task, current amount, total when known, completion, and interruption.",
    ),
    AccessibilityRequirement(
        "text_plus_color",
        "Meaning is never communicated by color alone.",
        "Success, warning, error, selected, and excluded states include text or an icon with a readable label.",
    ),
    AccessibilityRequirement(
        "reduced_motion",
        "Motion respects the user's reduced-motion setting.",
        "Nonessential animation stops when reduced motion is requested; progress remains understandable as text.",
    ),
    AccessibilityRequirement(
        "readable_errors",
        "Errors use plain language and show the exact next action.",
        "The visible error names the problem, says what was preserved, and gives one concrete recovery action.",
    ),
    AccessibilityRequirement(
        "minimum_target_size",
        "Pointer and touch targets are at least 44 by 44 CSS pixels.",
        "Automated layout checks and manual zoom checks confirm the minimum target size without overlap.",
    ),
    AccessibilityRequirement(
        "no_hover_only",
        "No essential control or explanation is available only on hover.",
        "Everything shown on hover is also available by keyboard focus, visible text, or an adjacent control.",
    ),
    AccessibilityRequirement(
        "non_expert_language",
        "Visible language is suitable for people without machine-learning or developer experience.",
        "A plain-language scan finds no unexplained specialist terms in headings, actions, errors, or next steps.",
    ),
)

FOCUS_ORDER = ("page_title", "status_message", "primary_action", "secondary_actions", "help")
PROGRESS_LIVE_REGION = {"role": "status", "aria-live": "polite", "aria-atomic": "true"}
ERROR_REGION = {"role": "alert", "aria-live": "assertive"}
MINIMUM_TARGET_SIZE_CSS_PIXELS = 44


def accessibility_checklist_payload(*, result: str = "PASS") -> dict[str, Any]:
    return {
        "schema_version": "spritelab.accessibility-checklist.v1",
        "result": result,
        "focus_order": list(FOCUS_ORDER),
        "progress_live_region": PROGRESS_LIVE_REGION,
        "error_region": ERROR_REGION,
        "minimum_target_size_css_pixels": MINIMUM_TARGET_SIZE_CSS_PIXELS,
        "requirements": [{**requirement.to_dict(), "result": result} for requirement in ACCESSIBILITY_REQUIREMENTS],
    }

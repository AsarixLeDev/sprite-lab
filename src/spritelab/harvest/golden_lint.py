"""Non-destructive consistency linting for golden label JSONL files."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.label_candidates import FOOD_CANDIDATES, GEM_CANDIDATES, GENERIC_OBJECT_NAMES, TOOL_CANDIDATES
from spritelab.harvest.label_taxonomy import normalize_category, normalize_object_name, normalize_tags

FOOD_OBJECTS = frozenset(FOOD_CANDIDATES)
GEM_OBJECTS = frozenset((*GEM_CANDIDATES, "gem", "ruby", "sapphire", "emerald", "diamond", "amethyst", "crystal"))
TOOL_OBJECTS = frozenset(TOOL_CANDIDATES)
TYPO_OBJECTS = ("saphire", "amethist", "ovale")


def lint_golden_file(path: str | Path, *, fix: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lint a golden JSONL file and optionally return fixed suggestion rows."""

    rows = _read_rows(path)
    issues: list[dict[str, Any]] = []
    by_sprite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sprite_id = str(row.get("sprite_id", "")).strip()
        if sprite_id:
            by_sprite[sprite_id].append(row)
        issues.extend(_lint_row(row))
    for sprite_id, sprite_rows in sorted(by_sprite.items()):
        labels = {_label_key(row) for row in sprite_rows}
        if len(labels) > 1:
            issues.append(
                {
                    "sprite_id": sprite_id,
                    "code": "duplicate_conflicting_labels",
                    "message": "duplicate sprite_id has conflicting golden labels",
                    "severity": "error",
                }
            )
    suggestions = _fix_suggestions(rows, issues) if fix else []
    return issues, suggestions


def format_golden_lint_report(issues: Sequence[Mapping[str, Any]]) -> str:
    if not issues:
        return "Golden lint: no issues found.\n"
    lines = [f"Golden lint issues: {len(issues)}"]
    for issue in issues:
        sprite_id = issue.get("sprite_id", "")
        code = issue.get("code", "issue")
        message = issue.get("message", "")
        lines.append(f"- {sprite_id}: {code}: {message}")
    return "\n".join(lines) + "\n"


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _lint_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    sprite_id = str(row.get("sprite_id", "")).strip()
    category = normalize_category(str(row.get("category", "unknown")))
    object_name = normalize_object_name(str(row.get("object_name", "")))
    tags = normalize_tags(row.get("tags") or ())
    issues: list[dict[str, Any]] = []
    raw_object = str(row.get("object_name", "")).strip().lower()
    for typo in TYPO_OBJECTS:
        if typo in raw_object:
            issues.append(
                _issue(sprite_id, "typo_object_name", f"object name contains likely typo {typo!r}", "warning")
            )
            break
    expected = _expected_category(object_name)
    if expected and category != expected:
        issues.append(
            _issue(
                sprite_id,
                "category_object_mismatch",
                f"{object_name} usually belongs to {expected}, not {category}",
                "warning",
            )
        )
    if object_name == "ice_cream_sandwich" and category == "effect_icon":
        issues.append(
            _issue(
                sprite_id,
                "ice_cream_sandwich_effect_icon",
                "ice_cream_sandwich should probably be item_icon",
                "warning",
            )
        )
    if object_name in GENERIC_OBJECT_NAMES:
        issues.append(_issue(sprite_id, "generic_object_name", f"generic object name {object_name!r}", "warning"))
    if len(tags) <= 1 and (not tags or tags == (object_name,)):
        issues.append(_issue(sprite_id, "sparse_tags", "tags are sparse or only repeat object_name", "info"))
    return issues


def _expected_category(object_name: str) -> str:
    if object_name in FOOD_OBJECTS:
        return "item_icon"
    if object_name in GEM_OBJECTS or object_name.endswith("_gem"):
        return "material"
    if object_name in TOOL_OBJECTS:
        return "tool"
    return ""


def _fix_suggestions(rows: Sequence[Mapping[str, Any]], issues: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    codes_by_id: dict[str, set[str]] = defaultdict(set)
    for issue in issues:
        codes_by_id[str(issue.get("sprite_id", ""))].add(str(issue.get("code", "")))
    suggestions: list[dict[str, Any]] = []
    for row in rows:
        sprite_id = str(row.get("sprite_id", ""))
        object_name = normalize_object_name(str(row.get("object_name", "")))
        category = normalize_category(str(row.get("category", "unknown")))
        expected = _expected_category(object_name)
        suggested_tags = list(normalize_tags(row.get("tags") or ()))
        if "sparse_tags" in codes_by_id.get(sprite_id, set()) and object_name:
            suggested_tags = list(normalize_tags((object_name, *_domain_tags(object_name))))
        suggestions.append(
            {
                **dict(row),
                "lint_issue_codes": sorted(codes_by_id.get(sprite_id, set())),
                "suggested_category": expected or category,
                "suggested_object_name": object_name,
                "suggested_tags": suggested_tags,
                "fix_mode": "suggestion_only",
            }
        )
    return suggestions


def _domain_tags(object_name: str) -> tuple[str, ...]:
    if object_name in FOOD_OBJECTS:
        return ("food", "consumable")
    if object_name in GEM_OBJECTS or object_name.endswith("_gem"):
        return ("gem", "material")
    if object_name in TOOL_OBJECTS:
        return ("tool",)
    return ()


def _label_key(row: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        normalize_category(str(row.get("category", "unknown"))),
        normalize_object_name(str(row.get("object_name", ""))),
        normalize_tags(row.get("tags") or ()),
    )


def _issue(sprite_id: str, code: str, message: str, severity: str) -> dict[str, Any]:
    return {"sprite_id": sprite_id, "code": code, "message": message, "severity": severity}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows

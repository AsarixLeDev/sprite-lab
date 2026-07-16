"""User-editable metadata model for Dataset Maker imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

ALLOWED_STATUSES = {"accepted", "rejected", "needs_fix", "quarantine"}
ALLOWED_SPLITS = {"train", "val", "test"}

_TOKEN_SEPARATORS_RE = re.compile(r"[\s/\\]+")
_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9_.-]+")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def normalize_tag(value: str) -> str:
    """Normalize a free-form tag to a stable lowercase token."""

    return _normalize_token(value, default="")


def normalize_category(value: str) -> str:
    """Normalize a category, falling back to ``unknown`` when empty."""

    return _normalize_token(value, default="unknown")


def normalize_sprite_id(value: str) -> str:
    """Normalize a sprite ID into a filesystem-safe lowercase identifier."""

    return _normalize_token(value, default="")


@dataclass(frozen=True)
class DatasetMakerItem:
    """Metadata edited by the local Dataset Maker GUI."""

    sprite_id: str
    source_path: Path
    status: str
    category: str = "unknown"
    tags: tuple[str, ...] = ()
    notes: str = ""
    source_name: str = ""
    license: str = "unknown"
    author: str = ""
    split: str | None = None
    quality_issues: tuple[str, ...] = ()
    palette_size: int | None = None
    has_role_map: bool = False

    allowed_statuses: ClassVar[set[str]] = ALLOWED_STATUSES
    allowed_splits: ClassVar[set[str]] = ALLOWED_SPLITS

    def __post_init__(self) -> None:
        object.__setattr__(self, "sprite_id", normalize_sprite_id(str(self.sprite_id)))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "status", str(self.status).strip().lower())
        object.__setattr__(self, "category", normalize_category(str(self.category)))
        object.__setattr__(self, "tags", _dedupe_tags(self.tags))
        object.__setattr__(self, "notes", str(self.notes))
        object.__setattr__(self, "source_name", str(self.source_name).strip())
        object.__setattr__(self, "license", normalize_category(str(self.license)))
        object.__setattr__(self, "author", str(self.author).strip())
        object.__setattr__(
            self, "quality_issues", tuple(str(issue).strip() for issue in self.quality_issues if str(issue).strip())
        )

        split = self.split
        if split is not None:
            split_text = str(split).strip().lower()
            object.__setattr__(self, "split", split_text if split_text and split_text != "auto" else None)


def validate_dataset_maker_item(item: DatasetMakerItem) -> list[str]:
    """Return non-throwing validation errors for a Dataset Maker item."""

    errors: list[str] = []
    if not item.sprite_id:
        errors.append("sprite_id must be non-empty.")
    elif not _SAFE_ID_RE.fullmatch(item.sprite_id):
        errors.append(
            "sprite_id must start with a letter or digit and contain only lowercase letters, digits, _, ., or -."
        )

    if item.status not in ALLOWED_STATUSES:
        errors.append(f"status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}.")

    if item.split is not None and item.split not in ALLOWED_SPLITS:
        errors.append(f"split must be one of: {', '.join(sorted(ALLOWED_SPLITS))}, or None.")

    if not item.category:
        errors.append("category must be non-empty.")

    if item.palette_size is not None:
        if not isinstance(item.palette_size, int) or isinstance(item.palette_size, bool) or item.palette_size < 0:
            errors.append("palette_size must be a non-negative integer when provided.")

    return errors


def _dedupe_tags(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        tag = normalize_tag(str(value))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return tuple(result)


def _normalize_token(value: str, *, default: str) -> str:
    text = str(value).strip().lower()
    text = _TOKEN_SEPARATORS_RE.sub("_", text)
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or default

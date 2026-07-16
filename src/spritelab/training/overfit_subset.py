"""Deterministic tiny-subset helpers for generator overfit experiments."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.training.data import read_jsonl


@dataclass(frozen=True)
class OverfitSubsetSelection:
    """Selected sprite IDs plus auditable metadata about the filtered rows."""

    sprite_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    seed: int
    split: str | None
    stratify: str | None
    requested_count: int | None
    missing_sprite_ids: tuple[str, ...] = ()

    @property
    def sprite_id_set(self) -> set[str]:
        return set(self.sprite_ids)

    def to_report(self) -> dict[str, Any]:
        first_by_id = _first_rows_by_sprite_id(self.rows)
        return {
            "seed": int(self.seed),
            "split": self.split,
            "stratify": self.stratify,
            "requested_count": self.requested_count,
            "selected_sprite_count": len(self.sprite_ids),
            "selected_row_count": len(self.rows),
            "sprite_ids": list(self.sprite_ids),
            "missing_sprite_ids": list(self.missing_sprite_ids),
            "categories": dict(sorted(Counter(_row_category(row) for row in first_by_id.values()).items())),
            "object_names": {
                sprite_id: str(first_by_id.get(sprite_id, {}).get("object_name") or "") for sprite_id in self.sprite_ids
            },
        }


def read_sprite_id_list(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if path.suffix.lower() == ".json" or stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid sprite ID JSON: {exc}") from exc
        raw_ids = payload.get("sprite_ids") if isinstance(payload, Mapping) else payload
        if not isinstance(raw_ids, list):
            raise ValueError(f"{path}: sprite ID JSON must be a list or an object with sprite_ids")
        return [str(value).strip() for value in raw_ids if str(value).strip()]

    ids: list[str] = []
    for line in text.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(value)
    return ids


def select_overfit_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int | None = None,
    sprite_ids: Sequence[str] | None = None,
    split: str | None = "train",
    seed: int = 0,
    stratify: str | None = None,
) -> OverfitSubsetSelection:
    """Select a deterministic set of sprite IDs and matching manifest rows."""

    split_rows = [dict(row) for row in records if split is None or str(row.get("split", "")) == str(split)]
    first_by_id = _first_rows_by_sprite_id(split_rows)

    requested_count = None if count is None else max(0, int(count))
    missing: list[str] = []
    if sprite_ids is not None:
        ordered_ids = []
        seen: set[str] = set()
        for raw in sprite_ids:
            sprite_id = str(raw).strip()
            if not sprite_id or sprite_id in seen:
                continue
            seen.add(sprite_id)
            if sprite_id in first_by_id:
                ordered_ids.append(sprite_id)
            else:
                missing.append(sprite_id)
        if requested_count is not None:
            ordered_ids = ordered_ids[:requested_count]
    elif requested_count is not None:
        ordered_ids = _select_ids(first_by_id, requested_count, seed=seed, stratify=stratify)
    else:
        ordered_ids = tuple(first_by_id.keys())

    selected = set(ordered_ids)
    rows = tuple(row for row in split_rows if str(row.get("sprite_id", "")) in selected)
    return OverfitSubsetSelection(
        sprite_ids=tuple(ordered_ids),
        rows=rows,
        seed=int(seed),
        split=split,
        stratify=stratify,
        requested_count=requested_count,
        missing_sprite_ids=tuple(missing),
    )


def make_overfit_subset(
    *,
    dataset: str | Path,
    training_manifest: str | Path,
    out: str | Path,
    count: int,
    seed: int,
    split: str = "train",
    stratify: str | None = None,
) -> dict[str, Any]:
    """Write a sprite-id list and companion JSON/Markdown subset report."""

    del dataset  # The dataset path is recorded by the caller-facing report only.
    manifest_path = Path(training_manifest)
    out_path = Path(out)
    rows = read_jsonl(manifest_path)
    selection = select_overfit_subset(rows, count=count, split=split, seed=seed, stratify=stratify)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{sprite_id}\n" for sprite_id in selection.sprite_ids), encoding="utf-8")
    report = {
        **selection.to_report(),
        "training_manifest": str(manifest_path),
        "out": str(out_path),
    }
    json_path = out_path.with_suffix(out_path.suffix + ".json") if out_path.suffix else out_path.with_suffix(".json")
    md_path = out_path.with_suffix(out_path.suffix + ".md") if out_path.suffix else out_path.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_overfit_subset_markdown(report), encoding="utf-8")
    return report


def format_overfit_subset_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Overfit Subset",
        "",
        f"Seed: `{report.get('seed')}`",
        f"Split: `{report.get('split')}`",
        f"Selected sprites: {int(report.get('selected_sprite_count') or 0)}",
        f"Selected rows: {int(report.get('selected_row_count') or 0)}",
        "",
        "## Categories",
        "",
    ]
    categories = report.get("categories") if isinstance(report.get("categories"), Mapping) else {}
    if categories:
        lines.extend(f"- {key}: {value}" for key, value in categories.items())
    else:
        lines.append("- (none)")
    lines.extend(["", "## Sprite IDs", ""])
    object_names = report.get("object_names") if isinstance(report.get("object_names"), Mapping) else {}
    for sprite_id in report.get("sprite_ids") or []:
        lines.append(f"- `{sprite_id}` {object_names.get(sprite_id, '')}")
    lines.append("")
    return "\n".join(lines)


def filter_records_to_sprite_ids(
    records: Iterable[Mapping[str, Any]],
    sprite_ids: Iterable[str] | None,
) -> list[dict[str, Any]]:
    if sprite_ids is None:
        return [dict(row) for row in records]
    selected = {str(sprite_id) for sprite_id in sprite_ids}
    return [dict(row) for row in records if str(row.get("sprite_id", "")) in selected]


def _select_ids(
    first_by_id: Mapping[str, Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    stratify: str | None,
) -> tuple[str, ...]:
    available = sorted(first_by_id)
    limit = min(max(0, int(count)), len(available))
    if limit <= 0:
        return ()
    rng = random.Random(int(seed))
    if not stratify:
        shuffled = list(available)
        rng.shuffle(shuffled)
        return tuple(shuffled[:limit])

    key = str(stratify)
    groups: dict[str, list[str]] = defaultdict(list)
    for sprite_id in available:
        groups[str(first_by_id[sprite_id].get(key) or "unknown")].append(sprite_id)
    for ids in groups.values():
        rng.shuffle(ids)
    selected: list[str] = []
    group_names = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for group_name in group_names:
            ids = groups[group_name]
            if ids:
                selected.append(ids.pop(0))
                if len(selected) >= limit:
                    break
    return tuple(selected)


def _first_rows_by_sprite_id(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in records:
        sprite_id = str(row.get("sprite_id", "")).strip()
        if sprite_id and sprite_id not in first:
            first[sprite_id] = dict(row)
    return first


def _row_category(row: Mapping[str, Any]) -> str:
    return str(row.get("category") or "unknown")

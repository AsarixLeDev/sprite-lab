"""Golden-set sampling and label storage for prefill evaluation."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.dataset_maker.model import normalize_category, normalize_sprite_id, normalize_tag
from spritelab.harvest.sources import utc_timestamp

GOLDEN_SAMPLE_FILENAME = "golden_sample.jsonl"
GOLDEN_LABELS_FILENAME = "golden_labels.jsonl"


@dataclass(frozen=True)
class GoldenLabel:
    """One human-verified label used only for measurement, never training."""

    sprite_id: str
    category: str
    object_name: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    labeler: str = ""
    labeled_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sprite_id", normalize_sprite_id(self.sprite_id))
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "object_name", normalize_tag(self.object_name))
        object.__setattr__(self, "tags", _dedupe_tags(self.tags))
        object.__setattr__(self, "notes", str(self.notes).strip())
        object.__setattr__(self, "labeler", str(self.labeler).strip())
        object.__setattr__(self, "labeled_at", str(self.labeled_at).strip())


def sample_golden_candidates(
    records: Sequence[Mapping[str, Any]],
    n: int,
    *,
    stratify_by: Sequence[str] = ("source_name",),
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Deterministically sample imported.jsonl records, stratified by keys.

    Uses proportional allocation (largest remainder) across strata so every
    source is represented, then a seeded shuffle within each stratum. The
    same records/n/seed always produce the same sample.
    """

    if n <= 0:
        return []

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        if not sprite_id:
            continue
        key = tuple(str(record.get(field, "")) for field in stratify_by)
        groups.setdefault(key, []).append(dict(record))

    total = sum(len(members) for members in groups.values())
    if total == 0:
        return []
    n = min(n, total)

    sorted_keys = sorted(groups)
    for key in sorted_keys:
        groups[key].sort(key=lambda record: str(record.get("sprite_id", "")))
        rng = random.Random(f"{seed}:{'/'.join(key)}")
        rng.shuffle(groups[key])

    quotas = _largest_remainder_quotas(
        [(key, len(groups[key])) for key in sorted_keys],
        n=n,
        total=total,
    )

    sample: list[dict[str, Any]] = []
    for key in sorted_keys:
        for record in groups[key][: quotas[key]]:
            sample.append(
                {
                    "sprite_id": str(record.get("sprite_id", "")),
                    "filename": Path(str(record.get("relative_path") or record.get("final_png_path", ""))).name,
                    "final_png_path": str(record.get("final_png_path", "")),
                    "strata": list(key),
                }
            )
    sample.sort(key=lambda record: record["sprite_id"])
    return sample


def append_golden_label(path: str | Path, label: GoldenLabel) -> Path:
    """Append one golden label line; the file is append-only."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = golden_label_to_dict(label)
    if not record["labeled_at"]:
        record["labeled_at"] = utc_timestamp()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_golden_labels(path: str | Path) -> dict[str, GoldenLabel]:
    """Load golden labels keyed by sprite_id; the last write per sprite wins."""

    path = Path(path)
    if not path.exists():
        return {}
    labels: dict[str, GoldenLabel] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            label = golden_label_from_dict(data)
            if label.sprite_id:
                labels[label.sprite_id] = label
    return labels


def golden_label_to_dict(label: GoldenLabel) -> dict[str, Any]:
    return {
        "sprite_id": label.sprite_id,
        "category": label.category,
        "object_name": label.object_name,
        "tags": list(label.tags),
        "notes": label.notes,
        "labeler": label.labeler,
        "labeled_at": label.labeled_at,
    }


def golden_label_from_dict(data: Mapping[str, Any]) -> GoldenLabel:
    tags = data.get("tags")
    if isinstance(tags, str):
        tags = (tags,)
    return GoldenLabel(
        sprite_id=str(data.get("sprite_id", "")),
        category=str(data.get("category", "unknown")),
        object_name=str(data.get("object_name", "")),
        tags=tuple(str(tag) for tag in tags or ()),
        notes=str(data.get("notes", "")),
        labeler=str(data.get("labeler", "")),
        labeled_at=str(data.get("labeled_at", "")),
    )


def _largest_remainder_quotas(
    sized_keys: Sequence[tuple[tuple[str, ...], int]],
    *,
    n: int,
    total: int,
) -> dict[tuple[str, ...], int]:
    exact = {key: n * size / total for key, size in sized_keys}
    sizes = dict(sized_keys)
    # Floor of 1 per stratum (when n allows) so tiny sources stay represented.
    floor = 1 if n >= len(sized_keys) else 0
    quotas = {key: min(sizes[key], max(floor, int(exact[key]))) for key, _ in sized_keys}
    if sum(quotas.values()) > n:
        # Floors overshot the budget; shrink the largest quotas first, deterministically.
        for key in sorted(quotas, key=lambda key: (-quotas[key], key)):
            if sum(quotas.values()) <= n:
                break
            if quotas[key] > 0:
                quotas[key] -= 1
    remaining = n - sum(quotas.values())
    # Hand out leftovers by largest fractional remainder, deterministic tie-break on key.
    by_remainder = sorted(
        (key for key, _ in sized_keys),
        key=lambda key: (-(exact[key] - int(exact[key])), key),
    )
    while remaining > 0:
        progressed = False
        for key in by_remainder:
            if remaining <= 0:
                break
            if quotas[key] < sizes[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _dedupe_tags(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        tag = normalize_tag(str(value))
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return tuple(result)

"""Dataset manifest records and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IngestedSpriteRecord:
    id: str
    source_path: str
    bundle_dir: str
    width: int
    height: int
    category: str | None
    subtype: str | None
    license: str | None
    palette_size: int
    sha256: str
    split: str | None = None


@dataclass(frozen=True)
class RejectedSpriteRecord:
    source_path: str
    reason: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    records: list[IngestedSpriteRecord]
    rejected_count: int
    total_seen: int
    options: dict[str, Any]


def save_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Write a dataset manifest as stable, readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load a dataset manifest from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest(
        dataset_name=data["dataset_name"],
        records=[IngestedSpriteRecord(**record) for record in data["records"]],
        rejected_count=int(data["rejected_count"]),
        total_seen=int(data["total_seen"]),
        options=dict(data["options"]),
    )


def save_rejected_report(rejected: list[RejectedSpriteRecord], path: str | Path) -> None:
    """Write rejected records as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rejected": [asdict(record) for record in rejected]}
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rejected_report(path: str | Path) -> list[RejectedSpriteRecord]:
    """Load rejected records from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RejectedSpriteRecord(**record) for record in data.get("rejected", [])]

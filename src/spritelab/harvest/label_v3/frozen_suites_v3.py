"""Auto-Labeling v3: frozen evaluation suites and leakage control.

A frozen suite is an immutable, versioned partitioning of golden sprite ids into
evaluation strata (``in_domain`` / ``unseen_pack`` / ``source_ood`` / …), pinned
to a taxonomy version and annotation guidance. Promotion gates must be measured
on these frozen partitions, and the same ids must never be used for threshold
tuning or calibration — the leakage checks here enforce that.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUITE_SCHEMA_VERSION = "frozen_suite_v3.1"

# Canonical partitions the promotion policy evaluates independently.
CORE_PARTITIONS: tuple[str, ...] = ("in_domain", "unseen_pack", "source_ood")


@dataclass(frozen=True)
class FrozenSuiteManifest:
    """An immutable partitioning of golden ids into evaluation strata."""

    schema_version: str = SUITE_SCHEMA_VERSION
    suite_name: str = "unnamed"
    taxonomy_version: str = ""
    annotation_guidance: str = ""
    created_at: str = ""
    partitions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def all_ids(self) -> set[str]:
        ids: set[str] = set()
        for members in self.partitions.values():
            ids.update(members)
        return ids

    def partition_ids(self, name: str) -> tuple[str, ...]:
        return tuple(self.partitions.get(name, ()))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "taxonomy_version": self.taxonomy_version,
            "annotation_guidance": self.annotation_guidance,
            "created_at": self.created_at,
            "partitions": {k: list(v) for k, v in self.partitions.items()},
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> FrozenSuiteManifest:
        raw = data.get("partitions") or {}
        partitions = {str(k): tuple(str(x) for x in (v or ())) for k, v in raw.items()}
        return cls(
            schema_version=str(data.get("schema_version", SUITE_SCHEMA_VERSION)),
            suite_name=str(data.get("suite_name", "unnamed")),
            taxonomy_version=str(data.get("taxonomy_version", "")),
            annotation_guidance=str(data.get("annotation_guidance", "")),
            created_at=str(data.get("created_at", "")),
            partitions=partitions,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FrozenSuiteManifest:
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LeakageReport:
    suite_name: str
    ok: bool
    cross_partition_overlaps: dict[str, tuple[str, ...]]
    tuning_overlap: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "ok": self.ok,
            "cross_partition_overlaps": {k: list(v) for k, v in self.cross_partition_overlaps.items()},
            "tuning_overlap": list(self.tuning_overlap),
        }


def check_suite_leakage(
    manifest: FrozenSuiteManifest,
    *,
    tuning_ids: Iterable[str] = (),
) -> LeakageReport:
    """Verify a frozen suite is internally disjoint and does not overlap tuning.

    Two failures are reported:
      * a sprite id appearing in more than one partition (partitions must be
        mutually exclusive);
      * any suite id also present in ``tuning_ids`` (calibration/threshold data),
        which would make the evaluation self-referential.
    """
    names = list(manifest.partitions.keys())
    cross: dict[str, tuple[str, ...]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sorted(set(manifest.partitions[a]) & set(manifest.partitions[b]))
            if overlap:
                cross[f"{a}&{b}"] = tuple(overlap)

    tuning_set = {str(x) for x in tuning_ids}
    tuning_overlap = tuple(sorted(manifest.all_ids() & tuning_set))

    ok = not cross and not tuning_overlap
    return LeakageReport(
        suite_name=manifest.suite_name,
        ok=ok,
        cross_partition_overlaps=cross,
        tuning_overlap=tuning_overlap,
    )

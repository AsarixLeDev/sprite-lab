"""Assisted golden-set labeling from existing harvest prefill outputs."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.dataset_maker.model import normalize_category as _normalize_category_token
from spritelab.dataset_maker.model import normalize_tag
from spritelab.harvest.catalog import read_jsonl, write_jsonl
from spritelab.harvest.filename_rules import filename_suggestion_to_dict, parse_filename_metadata
from spritelab.harvest.sources import utc_timestamp

GOLDEN_CATEGORY_VALUES = (
    "unknown",
    "item_icon",
    "block",
    "plant",
    "ui_icon",
    "entity",
    "character",
    "weapon",
    "tool",
    "armor",
    "material",
    "effect_icon",
    "environment_prop",
)

GOLDEN_CANDIDATES_FILENAME = "golden_candidates.jsonl"
GOLDEN_CANDIDATES_PREFILLED_FILENAME = "golden_candidates_prefilled.jsonl"
GOLDEN_LABELS_FILENAME = "golden_labels.jsonl"
GOLDEN_ASSISTED_STATE_FILENAME = "golden_assisted_state.json"

_BAD_FUSION_FLAGS = {"degenerate", "invalid_json", "request_failure", "warning_only"}


@dataclass(frozen=True)
class AssistedGoldenCandidate:
    sprite_id: str
    final_png_path: Path
    source_id: str = ""
    source_name: str = ""
    relative_path: str = ""
    license: str = ""
    author: str = ""
    existing_category: str = "unknown"
    existing_tags: tuple[str, ...] = ()
    rule_category: str = "unknown"
    rule_object_name: str = ""
    rule_tags: tuple[str, ...] = ()
    qwen_category: str = "unknown"
    qwen_object_name: str = ""
    qwen_tags: tuple[str, ...] = ()
    qwen_description: str = ""
    qwen_confidence: float | None = None
    qwen_warnings: tuple[str, ...] = ()
    fused_category: str = "unknown"
    fused_object_name: str = ""
    fused_tags: tuple[str, ...] = ()
    fused_description: str = ""
    fused_quality_flags: tuple[str, ...] = ()
    suggested_category: str = "unknown"
    suggested_object_name: str = ""
    suggested_tags: tuple[str, ...] = ()
    suggested_description: str = ""
    suggested_source: str = "none"
    needs_review_reason: str = ""
    quality_bucket: str = ""
    review_priority: float = 0.0
    prefill_source: str = ""
    prefill_category: str = "unknown"
    prefill_object_name: str = ""
    prefill_tags: tuple[str, ...] = ()
    prefill_short_description: str = ""
    prefill_materials: tuple[str, ...] = ()
    prefill_mood: tuple[str, ...] = ()
    prefill_bucket: str = ""
    prefill_flags: tuple[str, ...] = ()
    prefill_confidence: float = 0.0
    candidate_object_names: tuple[str, ...] = ()
    alternative_object_names: tuple[str, ...] = ()
    vlm_object_name: str = ""
    vlm_short_description: str = ""
    vlm_source_consistency: str = ""
    visual_facts: Mapping[str, Any] | None = None
    gold_category: str = "unknown"
    gold_object_name: str = ""
    gold_tags: tuple[str, ...] = ()
    gold_short_description: str = ""
    gold_materials: tuple[str, ...] = ()
    gold_mood: tuple[str, ...] = ()
    prefill_was_corrected: bool = False
    correction_fields: tuple[str, ...] = ()
    label_status: str = "needs_review"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sprite_id", normalize_object_name(self.sprite_id))
        object.__setattr__(self, "final_png_path", Path(self.final_png_path))
        object.__setattr__(self, "existing_category", normalize_category(self.existing_category))
        object.__setattr__(self, "existing_tags", normalize_tags(self.existing_tags))
        object.__setattr__(self, "rule_category", normalize_category(self.rule_category))
        object.__setattr__(self, "rule_object_name", normalize_object_name(self.rule_object_name))
        object.__setattr__(self, "rule_tags", normalize_tags(self.rule_tags))
        object.__setattr__(self, "qwen_category", normalize_category(self.qwen_category))
        object.__setattr__(self, "qwen_object_name", normalize_object_name(self.qwen_object_name))
        object.__setattr__(self, "qwen_tags", normalize_tags(self.qwen_tags))
        object.__setattr__(
            self, "qwen_warnings", tuple(str(value).strip() for value in self.qwen_warnings if str(value).strip())
        )
        object.__setattr__(self, "fused_category", normalize_category(self.fused_category))
        object.__setattr__(self, "fused_object_name", normalize_object_name(self.fused_object_name))
        object.__setattr__(self, "fused_tags", normalize_tags(self.fused_tags))
        object.__setattr__(self, "fused_quality_flags", normalize_tags(self.fused_quality_flags))
        object.__setattr__(self, "suggested_category", normalize_category(self.suggested_category))
        object.__setattr__(self, "suggested_object_name", normalize_object_name(self.suggested_object_name))
        object.__setattr__(self, "suggested_tags", normalize_tags(self.suggested_tags))
        object.__setattr__(self, "quality_bucket", str(self.quality_bucket).strip())
        object.__setattr__(self, "review_priority", max(0.0, min(1.0, float(self.review_priority))))
        object.__setattr__(self, "prefill_category", normalize_category(self.prefill_category))
        object.__setattr__(self, "prefill_object_name", normalize_object_name(self.prefill_object_name))
        object.__setattr__(self, "prefill_tags", normalize_tags(self.prefill_tags))
        object.__setattr__(self, "prefill_materials", normalize_tags(self.prefill_materials))
        object.__setattr__(self, "prefill_mood", normalize_tags(self.prefill_mood))
        object.__setattr__(self, "prefill_bucket", str(self.prefill_bucket).strip())
        object.__setattr__(self, "prefill_flags", normalize_tags(self.prefill_flags))
        object.__setattr__(self, "prefill_confidence", max(0.0, min(1.0, float(self.prefill_confidence or 0.0))))
        object.__setattr__(self, "candidate_object_names", normalize_tags(self.candidate_object_names))
        object.__setattr__(self, "alternative_object_names", normalize_tags(self.alternative_object_names))
        object.__setattr__(self, "vlm_object_name", normalize_object_name(self.vlm_object_name))
        object.__setattr__(self, "vlm_source_consistency", normalize_object_name(self.vlm_source_consistency))
        object.__setattr__(self, "visual_facts", dict(self.visual_facts or {}))
        object.__setattr__(self, "gold_category", normalize_category(self.gold_category))
        object.__setattr__(self, "gold_object_name", normalize_object_name(self.gold_object_name))
        object.__setattr__(self, "gold_tags", normalize_tags(self.gold_tags))
        object.__setattr__(self, "gold_materials", normalize_tags(self.gold_materials))
        object.__setattr__(self, "gold_mood", normalize_tags(self.gold_mood))
        object.__setattr__(self, "correction_fields", normalize_tags(self.correction_fields))
        object.__setattr__(self, "label_status", normalize_object_name(self.label_status) or "needs_review")


@dataclass(frozen=True)
class AssistedGoldenLabel:
    sprite_id: str
    category: str
    object_name: str
    tags: tuple[str, ...]
    short_description: str = ""
    materials: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    notes: str = ""
    labeler: str = "mathieu"
    labeled_at: str = ""
    source_id: str = ""
    source_name: str = ""
    relative_path: str = ""
    prefill_source: str = ""
    prefill_category: str = "unknown"
    prefill_object_name: str = ""
    prefill_tags: tuple[str, ...] = ()
    prefill_short_description: str = ""
    prefill_materials: tuple[str, ...] = ()
    prefill_mood: tuple[str, ...] = ()
    prefill_bucket: str = ""
    prefill_flags: tuple[str, ...] = ()
    prefill_confidence: float = 0.0
    candidate_object_names: tuple[str, ...] = ()
    alternative_object_names: tuple[str, ...] = ()
    vlm_object_name: str = ""
    vlm_short_description: str = ""
    vlm_source_consistency: str = ""
    visual_facts: Mapping[str, Any] | None = None
    prefill_was_corrected: bool = False
    correction_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sprite_id", normalize_object_name(self.sprite_id))
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "object_name", normalize_object_name(self.object_name))
        object.__setattr__(self, "tags", normalize_tags(self.tags))
        object.__setattr__(self, "short_description", str(self.short_description).strip())
        object.__setattr__(self, "materials", normalize_tags(self.materials))
        object.__setattr__(self, "mood", normalize_tags(self.mood))
        object.__setattr__(self, "notes", str(self.notes).strip())
        object.__setattr__(self, "labeler", str(self.labeler).strip() or "mathieu")
        object.__setattr__(self, "labeled_at", str(self.labeled_at).strip())
        object.__setattr__(self, "prefill_category", normalize_category(self.prefill_category))
        object.__setattr__(self, "prefill_object_name", normalize_object_name(self.prefill_object_name))
        object.__setattr__(self, "prefill_tags", normalize_tags(self.prefill_tags))
        object.__setattr__(self, "prefill_materials", normalize_tags(self.prefill_materials))
        object.__setattr__(self, "prefill_mood", normalize_tags(self.prefill_mood))
        object.__setattr__(self, "prefill_flags", normalize_tags(self.prefill_flags))
        object.__setattr__(self, "prefill_confidence", max(0.0, min(1.0, float(self.prefill_confidence or 0.0))))
        object.__setattr__(self, "candidate_object_names", normalize_tags(self.candidate_object_names))
        object.__setattr__(self, "alternative_object_names", normalize_tags(self.alternative_object_names))
        object.__setattr__(self, "vlm_object_name", normalize_object_name(self.vlm_object_name))
        object.__setattr__(self, "vlm_source_consistency", normalize_object_name(self.vlm_source_consistency))
        object.__setattr__(self, "visual_facts", dict(self.visual_facts or {}))
        object.__setattr__(self, "correction_fields", normalize_tags(self.correction_fields))


def load_assisted_candidates(
    run_dir: str | Path,
    *,
    n: int | None = None,
    seed: int = 1337,
    include_statuses: tuple[str, ...] = ("accepted",),
    prefer_needs_review: bool = True,
) -> list[AssistedGoldenCandidate]:
    """Load assisted golden candidates from a harvest run.

    Existing run metadata is combined with filename-rule, Qwen, and fused
    suggestions. If ``n`` is provided, a deterministic high-value sample is
    returned; otherwise all matching records are returned.
    """

    run_path = Path(run_dir)
    statuses = {str(status).strip().lower() for status in include_statuses if str(status).strip()}
    records = [*read_jsonl(run_path / "imported.jsonl"), *read_jsonl(run_path / "rejected.jsonl")]
    qwen_by_id = _qwen_by_id(read_jsonl(run_path / "qwen_suggestions.jsonl"))
    fused_by_id = _fused_by_id(read_jsonl(run_path / "fused_suggestions.jsonl"))
    label_v2_by_id = _fused_by_id(read_jsonl(run_path / "label_v2_suggestions.jsonl"))
    candidates = [
        _candidate_from_record(
            run_path, record, qwen_by_id=qwen_by_id, fused_by_id=fused_by_id, label_v2_by_id=label_v2_by_id
        )
        for record in records
        if _record_status(record) in statuses
    ]
    candidates = [candidate for candidate in candidates if candidate.sprite_id]

    if n is not None:
        n = max(0, int(n))
        if prefer_needs_review:
            rng = random.Random(seed)
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    -candidate_review_priority(candidate),
                    rng.random(),
                    candidate.sprite_id,
                ),
            )[:n]
        else:
            rng = random.Random(seed)
            candidates = list(candidates)
            rng.shuffle(candidates)
            candidates = sorted(candidates[:n], key=lambda candidate: candidate.sprite_id)
    else:
        candidates = sorted(
            candidates, key=lambda candidate: (-candidate_review_priority(candidate), candidate.sprite_id)
        )
    return candidates


def build_label_v2_prefilled_candidates(
    run_dir: str | Path,
    *,
    prediction_file: str | Path = "label_v2_suggestions.jsonl",
    n: int | None = None,
    seed: int = 496,
    stratify_by: Sequence[str] = ("source_profile.name", "bucket", "safe_prefill.object_name"),
    out_golden_file: str | Path = GOLDEN_LABELS_FILENAME,
    overwrite: bool = False,
) -> list[AssistedGoldenCandidate]:
    """Build assisted golden candidates initialized from label-v2 safe prefill records."""

    run_path = Path(run_dir)
    prediction_path = Path(prediction_file)
    if not prediction_path.is_absolute():
        prediction_path = run_path / prediction_path
    label_v2_records = [record for record in read_jsonl(prediction_path) if str(record.get("sprite_id", ""))]
    sampled = _sample_label_v2_records(label_v2_records, n=n, stratify_by=stratify_by, seed=seed)

    golden_path = Path(out_golden_file)
    if not golden_path.is_absolute():
        golden_path = run_path / golden_path
    existing_labels = load_existing_golden_labels(golden_path)
    return [
        _candidate_from_label_v2_record(run_path, record, existing_labels=existing_labels, overwrite=overwrite)
        for record in sampled
    ]


def summarize_golden_prefill_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize correction metadata in prefilled golden labels or candidates."""

    total = len(records)
    prefilled = 0
    corrected = 0
    fields: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    for record in records:
        source = str(record.get("prefill_source", ""))
        if not source:
            continue
        prefilled += 1
        correction_fields = tuple(str(field) for field in record.get("correction_fields") or ())
        if not correction_fields:
            correction_fields = tuple(_computed_correction_fields(record))
        is_corrected = bool(record.get("prefill_was_corrected", False)) or bool(correction_fields)
        if is_corrected:
            corrected += 1
            fields.update(correction_fields)
            bucket = str(record.get("prefill_bucket", "") or "missing")
            buckets[bucket] += 1
            object_name = str(record.get("prefill_object_name", "") or "missing")
            objects[object_name] += 1
    unchanged = prefilled - corrected
    return {
        "total": total,
        "prefilled_from_label_v2": sum(1 for record in records if str(record.get("prefill_source", "")) == "label_v2"),
        "prefilled_total": prefilled,
        "unchanged": unchanged,
        "corrected": corrected,
        "correction_rate": corrected / prefilled if prefilled else 0.0,
        "corrections_by_field": dict(fields.most_common()),
        "corrections_by_bucket": dict(buckets.most_common()),
        "most_corrected_objects": dict(objects.most_common(20)),
    }


def format_golden_prefill_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Total golden labels: {int(summary.get('total', 0))}",
        f"Prefilled from label-v2: {int(summary.get('prefilled_from_label_v2', 0))}",
        f"Unchanged: {int(summary.get('unchanged', 0))}",
        f"Corrected: {int(summary.get('corrected', 0))}",
        f"Correction rate: {float(summary.get('correction_rate', 0.0)) * 100:.1f}%",
        "",
        "Corrections by field:",
    ]
    for field, count in dict(summary.get("corrections_by_field") or {}).items():
        lines.append(f"- {field}: {count}")
    lines.extend(["", "Corrections by bucket:"])
    for bucket, count in dict(summary.get("corrections_by_bucket") or {}).items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "Most corrected objects:"])
    for object_name, count in dict(summary.get("most_corrected_objects") or {}).items():
        lines.append(f"- {object_name}: {count}")
    return "\n".join(lines) + "\n"


def choose_best_prefill(candidate: AssistedGoldenCandidate) -> tuple[str, str, tuple[str, ...], str, str]:
    """Choose category/object/tags/description/source for editable prefill."""

    if _usable(candidate.fused_category, candidate.fused_object_name, candidate.fused_tags) and not (
        set(candidate.fused_quality_flags) & _BAD_FUSION_FLAGS
    ):
        return (
            candidate.fused_category,
            candidate.fused_object_name,
            candidate.fused_tags,
            candidate.fused_description,
            "fusion",
        )
    if _filename_confident(candidate):
        return (
            candidate.rule_category,
            candidate.rule_object_name,
            candidate.rule_tags,
            _description_for(candidate.rule_object_name, candidate.rule_category),
            "filename_rules",
        )
    if candidate.qwen_category != "unknown" and candidate.qwen_object_name:
        return (
            candidate.qwen_category,
            candidate.qwen_object_name,
            candidate.qwen_tags,
            candidate.qwen_description,
            "qwen",
        )
    if candidate.existing_category != "unknown" or candidate.existing_tags:
        return (
            candidate.existing_category,
            "",
            candidate.existing_tags,
            "",
            "existing",
        )
    return ("unknown", "", (), "", "none")


def candidate_review_priority(candidate: AssistedGoldenCandidate) -> float:
    """Return a higher score for candidates worth reviewing early."""

    score = 0.0
    flags = set(candidate.fused_quality_flags)
    if flags:
        score += 30.0
    if _qwen_filename_conflict(candidate):
        score += 35.0
    if candidate.qwen_category == "unknown" and _filename_confident(candidate):
        score += 25.0
    if candidate.qwen_category == "unknown" and (candidate.qwen_confidence or 0.0) >= 0.85:
        score += 20.0
    if any(flag in flags for flag in ("needs_review", "filename_qwen_conflict", "low_confidence", "degenerate")):
        score += 15.0
    if candidate.suggested_source == "filename_rules" and _filename_confident(candidate):
        score -= 10.0
    if candidate.needs_review_reason:
        score += 10.0
    return score


def load_existing_golden_labels(path: str | Path) -> dict[str, AssistedGoldenLabel]:
    """Load assisted golden labels; last write per sprite_id wins."""

    labels: dict[str, AssistedGoldenLabel] = {}
    for record in read_jsonl(path):
        label = assisted_golden_label_from_dict(record)
        if label.sprite_id:
            labels[label.sprite_id] = label
    return labels


def append_golden_label(path: str | Path, label: AssistedGoldenLabel) -> None:
    """Append one assisted golden label to the append-only JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = assisted_golden_label_to_dict(label)
    if not record["labeled_at"]:
        record["labeled_at"] = utc_timestamp()
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_golden_candidates_jsonl(path: str | Path, candidates: Sequence[AssistedGoldenCandidate]) -> None:
    """Write candidate context and suggestions for assisted labeling."""

    write_jsonl(path, [assisted_candidate_to_dict(candidate) for candidate in candidates])


def load_golden_candidates_jsonl(path: str | Path) -> list[AssistedGoldenCandidate]:
    """Load assisted golden candidates previously written as JSONL."""

    return [assisted_candidate_from_dict(record) for record in read_jsonl(path)]


def build_assisted_golden_label(
    candidate: AssistedGoldenCandidate,
    *,
    category: str,
    object_name: str,
    tags: str | Sequence[str],
    short_description: str = "",
    materials: str | Sequence[str] = (),
    mood: str | Sequence[str] = (),
    notes: str = "",
    labeler: str = "mathieu",
) -> AssistedGoldenLabel:
    """Build a label and compute correction tracking against the prefill."""

    normalized_category = normalize_category(category)
    normalized_object = normalize_object_name(object_name)
    normalized_tags = normalize_tags(tags)
    normalized_description = str(short_description).strip()
    normalized_materials = normalize_tags(materials)
    normalized_mood = normalize_tags(mood)
    normalized_notes = str(notes).strip()
    prefill = _candidate_prefill_fields(candidate)
    corrections = _changed_fields(
        category=normalized_category,
        object_name=normalized_object,
        tags=normalized_tags,
        short_description=normalized_description,
        materials=normalized_materials,
        mood=normalized_mood,
        prefill=prefill,
    )
    if normalized_notes:
        corrections.append("notes")
    return AssistedGoldenLabel(
        sprite_id=candidate.sprite_id,
        category=normalized_category,
        object_name=normalized_object,
        tags=normalized_tags,
        short_description=normalized_description,
        materials=normalized_materials,
        mood=normalized_mood,
        notes=normalized_notes,
        labeler=labeler,
        source_id=candidate.source_id,
        source_name=candidate.source_name,
        relative_path=candidate.relative_path,
        prefill_source=prefill["source"],
        prefill_category=prefill["category"],
        prefill_object_name=prefill["object_name"],
        prefill_tags=tuple(prefill["tags"]),
        prefill_short_description=prefill["short_description"],
        prefill_materials=tuple(prefill["materials"]),
        prefill_mood=tuple(prefill["mood"]),
        prefill_bucket=candidate.prefill_bucket or candidate.quality_bucket,
        prefill_flags=candidate.prefill_flags or candidate.fused_quality_flags,
        prefill_confidence=candidate.prefill_confidence,
        candidate_object_names=candidate.candidate_object_names,
        alternative_object_names=candidate.alternative_object_names,
        vlm_object_name=candidate.vlm_object_name,
        vlm_short_description=candidate.vlm_short_description,
        vlm_source_consistency=candidate.vlm_source_consistency,
        visual_facts=candidate.visual_facts,
        prefill_was_corrected=bool(corrections),
        correction_fields=tuple(corrections),
    )


def _candidate_prefill_fields(candidate: AssistedGoldenCandidate) -> dict[str, Any]:
    category = candidate.prefill_category if candidate.prefill_source else candidate.suggested_category
    object_name = candidate.prefill_object_name if candidate.prefill_source else candidate.suggested_object_name
    tags = candidate.prefill_tags if candidate.prefill_source else candidate.suggested_tags
    description = candidate.prefill_short_description if candidate.prefill_source else candidate.suggested_description
    return {
        "source": candidate.prefill_source or candidate.suggested_source,
        "category": category,
        "object_name": object_name,
        "tags": tuple(tags),
        "short_description": description,
        "materials": tuple(candidate.prefill_materials),
        "mood": tuple(candidate.prefill_mood),
    }


def _changed_fields(
    *,
    category: str,
    object_name: str,
    tags: tuple[str, ...],
    short_description: str,
    materials: tuple[str, ...],
    mood: tuple[str, ...],
    prefill: Mapping[str, Any],
) -> list[str]:
    corrections: list[str] = []
    if category != str(prefill.get("category", "unknown")):
        corrections.append("category")
    if object_name != str(prefill.get("object_name", "")):
        corrections.append("object_name")
    if set(tags) != set(prefill.get("tags") or ()):
        corrections.append("tags")
    if short_description != str(prefill.get("short_description", "")):
        corrections.append("short_description")
    if set(materials) != set(prefill.get("materials") or ()):
        corrections.append("materials")
    if set(mood) != set(prefill.get("mood") or ()):
        corrections.append("mood")
    return corrections


def normalize_object_name(value: str) -> str:
    return normalize_tag(str(value))


def normalize_tags(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values = re.split(r"[,\s]+", value)
    else:
        raw_values = [str(item) for item in value]
    seen: set[str] = set()
    tags: list[str] = []
    for raw in raw_values:
        tag = normalize_tag(str(raw))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tuple(tags)


def normalize_category(value: str) -> str:
    category = _normalize_category_token(str(value))
    return category if category in GOLDEN_CATEGORY_VALUES else "unknown"


def assisted_candidate_to_dict(candidate: AssistedGoldenCandidate) -> dict[str, Any]:
    data = asdict(candidate)
    data["final_png_path"] = str(candidate.final_png_path)
    data["existing_tags"] = list(candidate.existing_tags)
    data["rule_tags"] = list(candidate.rule_tags)
    data["qwen_tags"] = list(candidate.qwen_tags)
    data["qwen_warnings"] = list(candidate.qwen_warnings)
    data["fused_tags"] = list(candidate.fused_tags)
    data["fused_quality_flags"] = list(candidate.fused_quality_flags)
    data["suggested_tags"] = list(candidate.suggested_tags)
    data["prefill_tags"] = list(candidate.prefill_tags)
    data["prefill_materials"] = list(candidate.prefill_materials)
    data["prefill_mood"] = list(candidate.prefill_mood)
    data["prefill_flags"] = list(candidate.prefill_flags)
    data["candidate_object_names"] = list(candidate.candidate_object_names)
    data["alternative_object_names"] = list(candidate.alternative_object_names)
    data["gold_tags"] = list(candidate.gold_tags)
    data["gold_materials"] = list(candidate.gold_materials)
    data["gold_mood"] = list(candidate.gold_mood)
    data["correction_fields"] = list(candidate.correction_fields)
    data["visual_facts"] = dict(candidate.visual_facts or {})
    return data


def assisted_candidate_from_dict(data: Mapping[str, Any]) -> AssistedGoldenCandidate:
    return AssistedGoldenCandidate(
        sprite_id=str(data.get("sprite_id", "")),
        final_png_path=Path(str(data.get("final_png_path", ""))),
        source_id=str(data.get("source_id", "")),
        source_name=str(data.get("source_name", "")),
        relative_path=str(data.get("relative_path", "")),
        license=str(data.get("license", "")),
        author=str(data.get("author", "")),
        existing_category=str(data.get("existing_category", "unknown")),
        existing_tags=tuple(str(tag) for tag in data.get("existing_tags") or ()),
        rule_category=str(data.get("rule_category", "unknown")),
        rule_object_name=str(data.get("rule_object_name", "")),
        rule_tags=tuple(str(tag) for tag in data.get("rule_tags") or ()),
        qwen_category=str(data.get("qwen_category", "unknown")),
        qwen_object_name=str(data.get("qwen_object_name", "")),
        qwen_tags=tuple(str(tag) for tag in data.get("qwen_tags") or ()),
        qwen_description=str(data.get("qwen_description", "")),
        qwen_confidence=_float_or_none(data.get("qwen_confidence")),
        qwen_warnings=tuple(str(warning) for warning in data.get("qwen_warnings") or ()),
        fused_category=str(data.get("fused_category", "unknown")),
        fused_object_name=str(data.get("fused_object_name", "")),
        fused_tags=tuple(str(tag) for tag in data.get("fused_tags") or ()),
        fused_description=str(data.get("fused_description", "")),
        fused_quality_flags=tuple(str(flag) for flag in data.get("fused_quality_flags") or ()),
        suggested_category=str(data.get("suggested_category", "unknown")),
        suggested_object_name=str(data.get("suggested_object_name", "")),
        suggested_tags=tuple(str(tag) for tag in data.get("suggested_tags") or ()),
        suggested_description=str(data.get("suggested_description", "")),
        suggested_source=str(data.get("suggested_source", "none")),
        needs_review_reason=str(data.get("needs_review_reason", "")),
        quality_bucket=str(data.get("quality_bucket", "")),
        review_priority=_float_or_none(data.get("review_priority")) or 0.0,
        prefill_source=str(data.get("prefill_source", "")),
        prefill_category=str(data.get("prefill_category", "unknown")),
        prefill_object_name=str(data.get("prefill_object_name", "")),
        prefill_tags=tuple(str(tag) for tag in data.get("prefill_tags") or ()),
        prefill_short_description=str(data.get("prefill_short_description", "")),
        prefill_materials=tuple(str(value) for value in data.get("prefill_materials") or ()),
        prefill_mood=tuple(str(value) for value in data.get("prefill_mood") or ()),
        prefill_bucket=str(data.get("prefill_bucket", "")),
        prefill_flags=tuple(str(flag) for flag in data.get("prefill_flags") or ()),
        prefill_confidence=_float_or_none(data.get("prefill_confidence")) or 0.0,
        candidate_object_names=tuple(str(value) for value in data.get("candidate_object_names") or ()),
        alternative_object_names=tuple(str(value) for value in data.get("alternative_object_names") or ()),
        vlm_object_name=str(data.get("vlm_object_name", "")),
        vlm_short_description=str(data.get("vlm_short_description", "")),
        vlm_source_consistency=str(data.get("vlm_source_consistency", "")),
        visual_facts=dict(data.get("visual_facts") or {}) if isinstance(data.get("visual_facts"), Mapping) else {},
        gold_category=str(data.get("gold_category", data.get("suggested_category", "unknown"))),
        gold_object_name=str(data.get("gold_object_name", data.get("suggested_object_name", ""))),
        gold_tags=tuple(str(tag) for tag in data.get("gold_tags", data.get("suggested_tags", ())) or ()),
        gold_short_description=str(data.get("gold_short_description", data.get("suggested_description", ""))),
        gold_materials=tuple(str(value) for value in data.get("gold_materials") or ()),
        gold_mood=tuple(str(value) for value in data.get("gold_mood") or ()),
        prefill_was_corrected=bool(data.get("prefill_was_corrected", False)),
        correction_fields=tuple(str(field) for field in data.get("correction_fields") or ()),
        label_status=str(data.get("label_status", "needs_review")),
    )


def assisted_golden_label_to_dict(label: AssistedGoldenLabel) -> dict[str, Any]:
    return {
        "sprite_id": label.sprite_id,
        "category": label.category,
        "object_name": label.object_name,
        "tags": list(label.tags),
        "short_description": label.short_description,
        "materials": list(label.materials),
        "mood": list(label.mood),
        "notes": label.notes,
        "labeler": label.labeler,
        "labeled_at": label.labeled_at,
        "source_id": label.source_id,
        "source_name": label.source_name,
        "relative_path": label.relative_path,
        "prefill_source": label.prefill_source,
        "prefill_category": label.prefill_category,
        "prefill_object_name": label.prefill_object_name,
        "prefill_tags": list(label.prefill_tags),
        "prefill_short_description": label.prefill_short_description,
        "prefill_materials": list(label.prefill_materials),
        "prefill_mood": list(label.prefill_mood),
        "prefill_bucket": label.prefill_bucket,
        "prefill_flags": list(label.prefill_flags),
        "prefill_confidence": label.prefill_confidence,
        "candidate_object_names": list(label.candidate_object_names),
        "alternative_object_names": list(label.alternative_object_names),
        "vlm_object_name": label.vlm_object_name,
        "vlm_short_description": label.vlm_short_description,
        "vlm_source_consistency": label.vlm_source_consistency,
        "visual_facts": dict(label.visual_facts or {}),
        "prefill_was_corrected": bool(label.prefill_was_corrected),
        "correction_fields": list(label.correction_fields),
    }


def assisted_golden_label_from_dict(data: Mapping[str, Any]) -> AssistedGoldenLabel:
    return AssistedGoldenLabel(
        sprite_id=str(data.get("sprite_id", "")),
        category=str(data.get("category", "unknown")),
        object_name=str(data.get("object_name", "")),
        tags=tuple(str(tag) for tag in data.get("tags") or ()),
        short_description=str(data.get("short_description", "")),
        materials=tuple(str(value) for value in data.get("materials") or ()),
        mood=tuple(str(value) for value in data.get("mood") or ()),
        notes=str(data.get("notes", "")),
        labeler=str(data.get("labeler", "mathieu")),
        labeled_at=str(data.get("labeled_at", "")),
        source_id=str(data.get("source_id", "")),
        source_name=str(data.get("source_name", "")),
        relative_path=str(data.get("relative_path", "")),
        prefill_source=str(data.get("prefill_source", "")),
        prefill_category=str(data.get("prefill_category", "unknown")),
        prefill_object_name=str(data.get("prefill_object_name", "")),
        prefill_tags=tuple(str(tag) for tag in data.get("prefill_tags") or ()),
        prefill_short_description=str(data.get("prefill_short_description", "")),
        prefill_materials=tuple(str(value) for value in data.get("prefill_materials") or ()),
        prefill_mood=tuple(str(value) for value in data.get("prefill_mood") or ()),
        prefill_bucket=str(data.get("prefill_bucket", "")),
        prefill_flags=tuple(str(flag) for flag in data.get("prefill_flags") or ()),
        prefill_confidence=_float_or_none(data.get("prefill_confidence")) or 0.0,
        candidate_object_names=tuple(str(value) for value in data.get("candidate_object_names") or ()),
        alternative_object_names=tuple(str(value) for value in data.get("alternative_object_names") or ()),
        vlm_object_name=str(data.get("vlm_object_name", "")),
        vlm_short_description=str(data.get("vlm_short_description", "")),
        vlm_source_consistency=str(data.get("vlm_source_consistency", "")),
        visual_facts=dict(data.get("visual_facts") or {}) if isinstance(data.get("visual_facts"), Mapping) else {},
        prefill_was_corrected=bool(data.get("prefill_was_corrected", False)),
        correction_fields=tuple(str(field) for field in data.get("correction_fields") or ()),
    )


def _candidate_from_label_v2_record(
    run_dir: Path,
    record: Mapping[str, Any],
    *,
    existing_labels: Mapping[str, AssistedGoldenLabel],
    overwrite: bool,
) -> AssistedGoldenCandidate:
    sprite_id = str(record.get("sprite_id", ""))
    safe = _mapping_or_none(record.get("safe_prefill")) or {}
    vlm = _mapping_or_none(record.get("vlm_descriptor")) or _mapping_or_none(record.get("vlm_suggestion")) or {}
    quality = _mapping_or_none(record.get("label_quality")) or {}
    visual_facts = _mapping_or_none(record.get("visual_facts")) or {}
    candidate_names = tuple(str(value) for value in record.get("candidate_object_names") or ())
    alternatives = tuple(str(value) for value in vlm.get("alternative_object_names") or ())

    prefill_category = str(safe.get("category", "unknown"))
    prefill_object = str(safe.get("object_name", ""))
    prefill_tags = tuple(str(tag) for tag in safe.get("tags") or ())
    prefill_description = str(safe.get("short_description", ""))
    prefill_materials = tuple(str(value) for value in safe.get("materials") or ())
    prefill_mood = tuple(str(value) for value in safe.get("mood") or ())
    existing = existing_labels.get(normalize_object_name(sprite_id))
    if existing is not None and not overwrite:
        gold_category = existing.category
        gold_object = existing.object_name
        gold_tags = existing.tags
        gold_description = existing.short_description
        gold_materials = existing.materials
        gold_mood = existing.mood
        label_status = "labeled"
    else:
        gold_category = prefill_category
        gold_object = prefill_object
        gold_tags = prefill_tags
        gold_description = prefill_description
        gold_materials = prefill_materials
        gold_mood = prefill_mood
        label_status = "needs_review"
    prefill = {
        "category": normalize_category(prefill_category),
        "object_name": normalize_object_name(prefill_object),
        "tags": normalize_tags(prefill_tags),
        "short_description": prefill_description,
        "materials": normalize_tags(prefill_materials),
        "mood": normalize_tags(prefill_mood),
    }
    correction_fields = _changed_fields(
        category=normalize_category(gold_category),
        object_name=normalize_object_name(gold_object),
        tags=normalize_tags(gold_tags),
        short_description=str(gold_description).strip(),
        materials=normalize_tags(gold_materials),
        mood=normalize_tags(gold_mood),
        prefill=prefill,
    )
    final_png_path = str(record.get("final_png_path", ""))
    return AssistedGoldenCandidate(
        sprite_id=sprite_id,
        final_png_path=_resolve_path(run_dir, final_png_path),
        source_id=str(record.get("source_id", "")),
        source_name=str(record.get("source_name", "")),
        relative_path=str(record.get("relative_path", "")),
        existing_category=str(record.get("category", "unknown")),
        existing_tags=tuple(str(tag) for tag in record.get("tags") or ()),
        qwen_category=str(vlm.get("category", "unknown")),
        qwen_object_name=str(vlm.get("object_name", "")),
        qwen_tags=tuple(str(tag) for tag in vlm.get("tags") or ()),
        qwen_description=str(vlm.get("short_description", "")),
        qwen_confidence=_float_or_none(vlm.get("confidence")),
        qwen_warnings=tuple(str(warning) for warning in vlm.get("warnings") or ()),
        fused_category=prefill_category,
        fused_object_name=prefill_object,
        fused_tags=prefill_tags,
        fused_description=prefill_description,
        fused_quality_flags=tuple(str(flag) for flag in quality.get("flags") or ()),
        suggested_category=gold_category,
        suggested_object_name=gold_object,
        suggested_tags=gold_tags,
        suggested_description=gold_description,
        suggested_source="label_v2",
        needs_review_reason=_needs_review_reason(quality),
        quality_bucket=str(quality.get("bucket", record.get("bucket", ""))),
        review_priority=_float_or_none(quality.get("review_priority")) or 0.0,
        prefill_source="label_v2",
        prefill_category=prefill_category,
        prefill_object_name=prefill_object,
        prefill_tags=prefill_tags,
        prefill_short_description=prefill_description,
        prefill_materials=prefill_materials,
        prefill_mood=prefill_mood,
        prefill_bucket=str(quality.get("bucket", record.get("bucket", ""))),
        prefill_flags=tuple(str(flag) for flag in quality.get("flags") or ()),
        prefill_confidence=_float_or_none(safe.get("confidence")) or 0.0,
        candidate_object_names=candidate_names,
        alternative_object_names=alternatives,
        vlm_object_name=str(vlm.get("object_name", "")),
        vlm_short_description=str(vlm.get("short_description", "")),
        vlm_source_consistency=str(vlm.get("source_consistency", "")),
        visual_facts=visual_facts,
        gold_category=gold_category,
        gold_object_name=gold_object,
        gold_tags=gold_tags,
        gold_short_description=gold_description,
        gold_materials=gold_materials,
        gold_mood=gold_mood,
        prefill_was_corrected=bool(correction_fields),
        correction_fields=tuple(correction_fields),
        label_status=label_status,
    )


def _candidate_from_record(
    run_dir: Path,
    record: Mapping[str, Any],
    *,
    qwen_by_id: Mapping[str, Mapping[str, Any]],
    fused_by_id: Mapping[str, Mapping[str, Any]],
    label_v2_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> AssistedGoldenCandidate:
    sprite_id = str(record.get("sprite_id", ""))
    auto_metadata = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
    image_path = _resolve_path(run_dir, str(record.get("final_png_path", "")))
    relative_path = str(record.get("relative_path") or image_path.name)
    filename = Path(relative_path).name

    filename_suggestion = _mapping_or_none(auto_metadata.get("filename_suggestion"))
    if filename_suggestion is None:
        filename_suggestion = filename_suggestion_to_dict(parse_filename_metadata(sprite_id, filename=filename))

    qwen = _mapping_or_none(auto_metadata.get("qwen_suggestion")) or dict(qwen_by_id.get(sprite_id, {}))
    fused_record = _mapping_or_none(fused_by_id.get(sprite_id)) or {}
    fused = (
        _mapping_or_none(auto_metadata.get("fused_suggestion"))
        or _mapping_or_none(fused_record.get("fused_suggestion"))
        or {}
    )
    quality = (
        _mapping_or_none(auto_metadata.get("prefill_quality"))
        or _mapping_or_none(fused_record.get("prefill_quality"))
        or {}
    )
    label_v2_record = _mapping_or_none((label_v2_by_id or {}).get(sprite_id)) or {}
    candidate_object_names: tuple[str, ...] = ()
    alternative_object_names: tuple[str, ...] = ()
    vlm_source_consistency = ""
    vlm_short_description = ""
    visual_facts: dict[str, Any] = {}
    if label_v2_record:
        filename_suggestion = _mapping_or_none(label_v2_record.get("filename_suggestion")) or filename_suggestion
        qwen = (
            _mapping_or_none(label_v2_record.get("vlm_descriptor"))
            or _mapping_or_none(label_v2_record.get("vlm_suggestion"))
            or qwen
        )
        fused = _mapping_or_none(label_v2_record.get("safe_prefill")) or fused
        quality = _mapping_or_none(label_v2_record.get("label_quality")) or quality
        candidate_object_names = tuple(str(value) for value in label_v2_record.get("candidate_object_names") or ())
        alternative_object_names = tuple(str(value) for value in qwen.get("alternative_object_names") or ())
        vlm_source_consistency = str(qwen.get("source_consistency", ""))
        vlm_short_description = str(qwen.get("short_description", ""))
        visual_facts = _mapping_or_none(label_v2_record.get("visual_facts")) or {}

    base = AssistedGoldenCandidate(
        sprite_id=sprite_id,
        final_png_path=image_path,
        source_id=str(record.get("source_id", "")),
        source_name=str(record.get("source_name", "")),
        relative_path=relative_path,
        license=str(record.get("license", "")),
        author=str(record.get("author", "")),
        existing_category=str(record.get("category", "unknown")),
        existing_tags=tuple(str(tag) for tag in record.get("tags") or ()),
        rule_category=str(filename_suggestion.get("category", "unknown")),
        rule_object_name=str(filename_suggestion.get("object_name", "")),
        rule_tags=tuple(str(tag) for tag in filename_suggestion.get("tags") or ()),
        qwen_category=str(qwen.get("category", "unknown")),
        qwen_object_name=str(qwen.get("object_name", "")),
        qwen_tags=tuple(str(tag) for tag in qwen.get("tags") or ()),
        qwen_description=str(qwen.get("short_description", "")),
        qwen_confidence=_float_or_none(qwen.get("confidence")),
        qwen_warnings=tuple(str(warning) for warning in qwen.get("warnings") or ()),
        fused_category=str(fused.get("category", "unknown")),
        fused_object_name=str(fused.get("object_name", "")),
        fused_tags=tuple(str(tag) for tag in fused.get("tags") or ()),
        fused_description=str(fused.get("short_description", "")),
        fused_quality_flags=tuple(str(flag) for flag in quality.get("flags") or ()),
        needs_review_reason=_needs_review_reason(quality),
        quality_bucket=str(quality.get("bucket", "")),
        review_priority=_float_or_none(quality.get("review_priority")) or 0.0,
        candidate_object_names=candidate_object_names,
        alternative_object_names=alternative_object_names,
        vlm_object_name=str(qwen.get("object_name", "")),
        vlm_short_description=vlm_short_description,
        vlm_source_consistency=vlm_source_consistency,
        visual_facts=visual_facts,
    )
    category, object_name, tags, description, source = choose_best_prefill(base)
    if label_v2_record and fused:
        source = "safe_prefill"
    return AssistedGoldenCandidate(
        **{
            **asdict(base),
            "final_png_path": base.final_png_path,
            "suggested_category": category,
            "suggested_object_name": object_name,
            "suggested_tags": tags,
            "suggested_description": description,
            "suggested_source": source,
            "prefill_source": "label_v2" if label_v2_record and fused else source,
            "prefill_category": category,
            "prefill_object_name": object_name,
            "prefill_tags": tags,
            "prefill_short_description": description,
            "prefill_materials": tuple(str(value) for value in fused.get("materials") or ()),
            "prefill_mood": tuple(str(value) for value in fused.get("mood") or ()),
            "prefill_bucket": str(quality.get("bucket", "")),
            "prefill_flags": tuple(str(flag) for flag in quality.get("flags") or ()),
            "prefill_confidence": _float_or_none(fused.get("confidence")) or 0.0,
            "gold_category": category,
            "gold_object_name": object_name,
            "gold_tags": tags,
            "gold_short_description": description,
            "gold_materials": tuple(str(value) for value in fused.get("materials") or ()),
            "gold_mood": tuple(str(value) for value in fused.get("mood") or ()),
        }
    )


def _qwen_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("sprite_id", "")): {key: value for key, value in record.items() if key != "sprite_id"}
        for record in records
        if record.get("sprite_id")
    }


def _fused_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("sprite_id", "")): dict(record) for record in records if record.get("sprite_id")}


def _sample_label_v2_records(
    records: Sequence[Mapping[str, Any]],
    *,
    n: int | None,
    stratify_by: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    values = [dict(record) for record in records if str(record.get("sprite_id", ""))]
    if n is None:
        return sorted(values, key=lambda record: str(record.get("sprite_id", "")))
    n = max(0, min(int(n), len(values)))
    if n == 0:
        return []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    fields = tuple(str(field).strip() for field in stratify_by if str(field).strip())
    for record in values:
        key = tuple(_nested_value(record, field) for field in fields) if fields else ("all",)
        groups.setdefault(key, []).append(record)
    sorted_keys = sorted(groups)
    for key in sorted_keys:
        groups[key].sort(key=lambda record: str(record.get("sprite_id", "")))
        random.Random(f"{seed}:{'/'.join(key)}").shuffle(groups[key])
    total = sum(len(members) for members in groups.values())
    quotas = _largest_remainder_quotas([(key, len(groups[key])) for key in sorted_keys], n=n, total=total)
    sampled: list[dict[str, Any]] = []
    for key in sorted_keys:
        sampled.extend(groups[key][: quotas[key]])
    return sorted(sampled, key=lambda record: str(record.get("sprite_id", "")))


def _largest_remainder_quotas(
    sized_keys: Sequence[tuple[tuple[str, ...], int]],
    *,
    n: int,
    total: int,
) -> dict[tuple[str, ...], int]:
    if total <= 0:
        return {key: 0 for key, _ in sized_keys}
    exact = {key: n * size / total for key, size in sized_keys}
    sizes = dict(sized_keys)
    floor = 1 if n >= len(sized_keys) else 0
    quotas = {key: min(sizes[key], max(floor, int(exact[key]))) for key, _ in sized_keys}
    if sum(quotas.values()) > n:
        for key in sorted(quotas, key=lambda value: (-quotas[value], value)):
            if sum(quotas.values()) <= n:
                break
            if quotas[key] > 0:
                quotas[key] -= 1
    remaining = n - sum(quotas.values())
    by_remainder = sorted((key for key, _ in sized_keys), key=lambda key: (-(exact[key] - int(exact[key])), key))
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


def _nested_value(record: Mapping[str, Any], field: str) -> str:
    value: Any = record
    for part in str(field).split("."):
        if isinstance(value, Mapping):
            value = value.get(part, "")
        else:
            return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _computed_correction_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    prefill = {
        "category": normalize_category(str(record.get("prefill_category", "unknown"))),
        "object_name": normalize_object_name(str(record.get("prefill_object_name", ""))),
        "tags": normalize_tags(tuple(str(value) for value in record.get("prefill_tags") or ())),
        "short_description": str(record.get("prefill_short_description", "")),
        "materials": normalize_tags(tuple(str(value) for value in record.get("prefill_materials") or ())),
        "mood": normalize_tags(tuple(str(value) for value in record.get("prefill_mood") or ())),
    }
    return tuple(
        _changed_fields(
            category=normalize_category(str(record.get("gold_category", record.get("category", "unknown")))),
            object_name=normalize_object_name(str(record.get("gold_object_name", record.get("object_name", "")))),
            tags=normalize_tags(tuple(str(value) for value in record.get("gold_tags", record.get("tags", ())) or ())),
            short_description=str(record.get("gold_short_description", record.get("short_description", ""))),
            materials=normalize_tags(
                tuple(str(value) for value in record.get("gold_materials", record.get("materials", ())) or ())
            ),
            mood=normalize_tags(tuple(str(value) for value in record.get("gold_mood", record.get("mood", ())) or ())),
            prefill=prefill,
        )
    )


def _record_status(record: Mapping[str, Any]) -> str:
    return str(record.get("status", "")).strip().lower()


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _resolve_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (path, Path.cwd() / path, run_dir / path, run_dir.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _needs_review_reason(quality: Mapping[str, Any]) -> str:
    reasons = []
    bucket = str(quality.get("bucket", ""))
    if bucket and bucket not in {"fused_automatically", "missing"}:
        reasons.append(bucket)
    reasons.extend(str(reason) for reason in quality.get("conflict_reasons") or () if str(reason))
    return "; ".join(reasons)


def _usable(category: str, object_name: str, tags: Sequence[str]) -> bool:
    return normalize_category(category) != "unknown" or bool(normalize_object_name(object_name)) or bool(tags)


def _filename_confident(candidate: AssistedGoldenCandidate) -> bool:
    return candidate.rule_category != "unknown" and bool(candidate.rule_object_name)


def _qwen_filename_conflict(candidate: AssistedGoldenCandidate) -> bool:
    if not _filename_confident(candidate):
        return False
    if candidate.qwen_category == "unknown" and not candidate.qwen_object_name:
        return False
    if candidate.qwen_category != "unknown" and candidate.qwen_category != candidate.rule_category:
        return True
    return bool(candidate.qwen_object_name and candidate.qwen_object_name != candidate.rule_object_name)


def _description_for(object_name: str, category: str) -> str:
    if not object_name:
        return ""
    readable = object_name.replace("_", " ")
    return f"A 32x32 pixel-art {readable} {category.replace('_', ' ')}."


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None

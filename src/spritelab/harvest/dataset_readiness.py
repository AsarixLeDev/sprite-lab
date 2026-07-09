"""Dataset readiness scanner for the multi-pack expansion layer.

This module inspects ``harvest_runs/*`` and ``datasets/*`` and reports, per
candidate pack, how close it is to being safely mergeable into a multi-source
semantic-v3 training dataset. It **never mutates** any run or dataset -- it only
reads state that earlier pipeline stages already wrote.

The output drives which packs are safe to onboard. A pack is only
``ready_for_merge`` when it has an exported dataset whose dataset QA and
training-manifest QA both pass with zero errors.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.build_semantic_dataset import (
    RAW_LABEL_V2_PREDICTION_PREFERENCE,
    is_raw_label_v2_prediction_file,
)
from spritelab.harvest.merge_datasets import derive_source_pack

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# A run whose review rate exceeds this is flagged rather than auto-onboarded.
DEFAULT_REVIEW_RATE_CEILING = 0.25

RECOMMENDED_ACTIONS: tuple[str, ...] = (
    "ready_for_merge",
    "needs_label_v2",
    "needs_semantic_v3",
    "needs_apply_label_v2",
    "needs_export",
    "needs_dataset_qa",
    "needs_training_manifest",
    "zero_apply_acceptance",
    "needs_source_family_profile",
    "too_many_review_items",
    "empty_or_import_broken",
    "manual_attention_required",
    "aggregate_dataset_not_merge_input",
    "legacy_nonsemantic_dataset",
)


@dataclass
class PackReadiness:
    run_name: str = ""
    run_path: str = ""
    has_label_v2_predictions: bool = False
    has_semantic_v3_predictions: bool = False
    has_apply_report: bool = False
    has_review_queue: bool = False
    has_exported_dataset: bool = False
    exported_dataset_path: str = ""
    dataset_qa_status: str = "missing"  # pass | fail | missing
    training_manifest_status: str = "missing"  # pass | fail | missing
    total_records: int = 0
    accepted_records: int = 0
    quarantined_records: int = 0
    raw_prediction_records: int = 0
    raw_prediction_auto_count: int = 0
    raw_prediction_review_count: int = 0
    apply_report_applied_auto_labels: int = 0
    apply_report_accepted_auto_labels: int = 0
    exported_dataset_records: int = 0
    dataset_qa_errors: int = 0
    training_manifest_qa_errors: int = 0
    review_rate: float = 0.0
    auto_rate: float = 0.0
    semantic_v3_coverage: float = 0.0
    base_object_coverage: float = 0.0
    caption_count_average: float = 0.0
    category_distribution: dict[str, int] = field(default_factory=dict)
    top_reasons: list[str] = field(default_factory=list)
    is_atomic_dataset: bool = False
    is_aggregate_dataset: bool = False
    is_merge_input_candidate: bool = False
    recommended_action: str = "manual_attention_required"
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessReport:
    runs_root: str
    datasets_root: str
    packs: list[PackReadiness] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        actions: Counter[str] = Counter(pack.recommended_action for pack in self.packs)
        return {
            "runs_root": self.runs_root,
            "datasets_root": self.datasets_root,
            "pack_count": len(self.packs),
            "ready_for_merge": [p.run_name for p in self.packs if p.recommended_action == "ready_for_merge"],
            "action_counts": dict(sorted(actions.items())),
            "packs": [pack.to_json_dict() for pack in self.packs],
        }


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_readiness(
    runs_root: str | Path,
    datasets_root: str | Path,
    *,
    review_rate_ceiling: float = DEFAULT_REVIEW_RATE_CEILING,
) -> ReadinessReport:
    """Scan harvest runs and exported datasets, returning per-pack readiness."""

    runs_root = Path(runs_root)
    datasets_root = Path(datasets_root)
    report = ReadinessReport(runs_root=str(runs_root), datasets_root=str(datasets_root))

    dataset_dirs = _list_exported_datasets(datasets_root)
    dataset_by_pack: dict[str, Path] = {}
    for dataset_dir in dataset_dirs:
        pack = derive_source_pack(dataset_dir.name)
        existing = dataset_by_pack.get(pack)
        # Prefer the semantic-v3 export when several datasets map to one pack.
        if existing is None or (
            dataset_dir.name.endswith("_semantic_v3") and not existing.name.endswith("_semantic_v3")
        ):
            dataset_by_pack[pack] = dataset_dir

    consumed_datasets: set[Path] = set()

    run_dirs = _list_subdirs(runs_root)
    for run_dir in run_dirs:
        exported = dataset_by_pack.get(run_dir.name)
        if exported is not None:
            consumed_datasets.add(exported)
        report.packs.append(_scan_run(run_dir, exported_dataset=exported, review_rate_ceiling=review_rate_ceiling))

    # Exported datasets with no matching run (e.g. hand-built or merged datasets)
    # still deserve an entry so they can be considered for merging.
    for dataset_dir in dataset_dirs:
        if dataset_dir in consumed_datasets:
            continue
        report.packs.append(_scan_standalone_dataset(dataset_dir, review_rate_ceiling=review_rate_ceiling))

    report.packs.sort(key=lambda pack: pack.run_name)
    return report


def _scan_run(run_dir: Path, *, exported_dataset: Path | None, review_rate_ceiling: float) -> PackReadiness:
    pack = PackReadiness(run_name=run_dir.name, run_path=str(run_dir))

    pack.has_label_v2_predictions = bool(_label_v2_prediction_files(run_dir))
    pack.has_semantic_v3_predictions = bool(_semantic_v3_prediction_files(run_dir))
    pack.has_apply_report = (run_dir / "label_v2_apply_report.json").is_file()
    pack.has_review_queue = (run_dir / "label_v2_review_queue.jsonl").is_file()

    raw_predictions = _read_raw_predictions(run_dir)
    if raw_predictions:
        _fill_raw_prediction_counts(pack, raw_predictions)
    summary = _load_json(run_dir / "label_v2_summary.json")
    if summary:
        pack.total_records = int(summary.get("total", 0) or 0)
        pack.review_rate = float(summary.get("review_rate", 0.0) or 0.0)
        auto_count = int(summary.get("auto_count", 0) or 0)
        pack.auto_rate = (auto_count / pack.total_records) if pack.total_records else 0.0
        categories = summary.get("top_categories")
        if isinstance(categories, Mapping):
            pack.category_distribution = {str(k): int(v) for k, v in categories.items()}
    if raw_predictions:
        pack.total_records = max(pack.total_records, pack.raw_prediction_records)
        pack.review_rate = (
            (pack.raw_prediction_review_count / pack.raw_prediction_records) if pack.raw_prediction_records else 0.0
        )
        pack.auto_rate = (
            (pack.raw_prediction_auto_count / pack.raw_prediction_records) if pack.raw_prediction_records else 0.0
        )
    if pack.total_records == 0:
        pack.total_records = _count_imported_records(run_dir)

    apply_report = _load_json(run_dir / "label_v2_apply_report.json")
    if apply_report:
        pack.apply_report_applied_auto_labels = int(apply_report.get("applied_auto_labels", 0) or 0)
        pack.apply_report_accepted_auto_labels = int(apply_report.get("accepted_auto_labels", 0) or 0)

    if exported_dataset is not None:
        _fill_from_dataset(pack, exported_dataset)
    elif pack.has_semantic_v3_predictions:
        _fill_semantic_from_predictions(pack, run_dir)

    pack.recommended_action = _recommend_action(pack, review_rate_ceiling=review_rate_ceiling)
    return pack


def _scan_standalone_dataset(dataset_dir: Path, *, review_rate_ceiling: float) -> PackReadiness:
    pack = PackReadiness(run_name=dataset_dir.name, run_path="")
    _fill_from_dataset(pack, dataset_dir)
    pack.recommended_action = _recommend_action(pack, review_rate_ceiling=review_rate_ceiling)
    pack.notes.append("standalone exported dataset (no matching harvest run)")
    return pack


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------


def _fill_from_dataset(pack: PackReadiness, dataset_dir: Path) -> None:
    pack.has_exported_dataset = True
    pack.exported_dataset_path = str(dataset_dir)
    _fill_dataset_kind(pack, dataset_dir)

    records = _load_all_manifest_records(dataset_dir)
    pack.total_records = max(pack.total_records, len(records))
    pack.accepted_records = len(records)
    pack.exported_dataset_records = len(records)

    if records:
        with_semantic = [r for r in records if isinstance(r.get("semantic_v3"), Mapping) and r.get("semantic_v3")]
        pack.semantic_v3_coverage = len(with_semantic) / len(records)
        base_objects = sum(1 for r in with_semantic if str((r.get("semantic_v3") or {}).get("base_object", "")).strip())
        pack.base_object_coverage = base_objects / len(records)
        caption_counts = [
            len((r.get("semantic_v3") or {}).get("captions") or [])
            for r in with_semantic
            if isinstance((r.get("semantic_v3") or {}).get("captions"), list)
        ]
        pack.caption_count_average = (sum(caption_counts) / len(caption_counts)) if caption_counts else 0.0
        categories: Counter[str] = Counter(str(r.get("category", "")) or "unknown" for r in records)
        pack.category_distribution = dict(sorted(categories.items()))

    qa = _load_json(dataset_dir / "dataset_qa_report.json")
    if qa:
        pack.dataset_qa_errors = len(qa.get("errors") or [])
        pack.dataset_qa_status = "pass" if pack.dataset_qa_errors == 0 else "fail"
        for error in list(qa.get("errors") or [])[:5]:
            pack.top_reasons.append(f"dataset_qa: {error}")
    tm_qa = _load_json(dataset_dir / "training_manifest_qa_report.json")
    if tm_qa:
        pack.training_manifest_qa_errors = len(tm_qa.get("errors") or [])
        pack.training_manifest_status = "pass" if pack.training_manifest_qa_errors == 0 else "fail"
        for error in list(tm_qa.get("errors") or [])[:5]:
            pack.top_reasons.append(f"training_manifest_qa: {error}")
    elif (dataset_dir / "training_manifest.jsonl").is_file():
        pack.training_manifest_status = "missing"  # built but not QA'd


def _fill_semantic_from_predictions(pack: PackReadiness, run_dir: Path) -> None:
    files = _semantic_v3_prediction_files(run_dir)
    if not files:
        return
    records = _read_jsonl(files[0])
    if not records:
        return
    with_semantic = [r for r in records if isinstance(r.get("semantic_v3"), Mapping) and r.get("semantic_v3")]
    pack.semantic_v3_coverage = len(with_semantic) / len(records)
    base_objects = sum(1 for r in with_semantic if str((r.get("semantic_v3") or {}).get("base_object", "")).strip())
    pack.base_object_coverage = base_objects / len(records)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _recommend_action(pack: PackReadiness, *, review_rate_ceiling: float) -> str:
    if pack.has_exported_dataset:
        if pack.is_aggregate_dataset:
            return "aggregate_dataset_not_merge_input"
        if (not pack.is_atomic_dataset or pack.semantic_v3_coverage < 1.0) and pack.dataset_qa_status != "fail":
            return "legacy_nonsemantic_dataset"
        if pack.dataset_qa_status == "missing":
            return "needs_dataset_qa"
        if pack.dataset_qa_status == "fail":
            return "manual_attention_required"
        # dataset QA passes.
        if pack.training_manifest_status == "pass":
            return "ready_for_merge" if pack.is_merge_input_candidate else "legacy_nonsemantic_dataset"
        if pack.training_manifest_status == "fail":
            return "manual_attention_required"
        return "needs_training_manifest"

    # No exported dataset yet.
    if pack.total_records == 0:
        return "empty_or_import_broken"
    if pack.has_apply_report and pack.apply_report_accepted_auto_labels == 0:
        return "zero_apply_acceptance"
    if pack.raw_prediction_records and pack.raw_prediction_auto_count == 0:
        return "needs_source_family_profile"
    if pack.review_rate > review_rate_ceiling:
        return "too_many_review_items"
    if pack.has_semantic_v3_predictions:
        return "needs_export" if _has_exportable_auto_records(pack) else "needs_apply_label_v2"
    if pack.has_label_v2_predictions:
        return "needs_semantic_v3"
    return "needs_label_v2"


# ---------------------------------------------------------------------------
# Filesystem helpers (robust to missing files)
# ---------------------------------------------------------------------------


def _list_subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


def _list_exported_datasets(datasets_root: Path) -> list[Path]:
    result: list[Path] = []
    for dataset_dir in _list_subdirs(datasets_root):
        if _is_exported_dataset(dataset_dir):
            result.append(dataset_dir)
    return result


def _is_exported_dataset(dataset_dir: Path) -> bool:
    return (dataset_dir / "manifest_train.jsonl").is_file() or (dataset_dir / "train.npz").is_file()


def _label_v2_prediction_files(run_dir: Path) -> list[Path]:
    candidates = [path for path in run_dir.glob("label_v2_suggestions*.jsonl") if is_raw_label_v2_prediction_file(path)]
    by_name = {path.name: path for path in candidates}
    ordered = [by_name[name] for name in RAW_LABEL_V2_PREDICTION_PREFERENCE if name in by_name]
    ordered.extend(path for path in sorted(candidates) if path.name not in set(RAW_LABEL_V2_PREDICTION_PREFERENCE))
    return ordered


def _semantic_v3_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*_semantic_v3.jsonl"))


def _count_imported_records(run_dir: Path) -> int:
    path = run_dir / "imported.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_all_manifest_records(dataset_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if path.is_file():
            records.extend(_read_jsonl(path))
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_raw_predictions(run_dir: Path) -> list[dict[str, Any]]:
    files = _label_v2_prediction_files(run_dir)
    return _read_jsonl(files[0]) if files else []


def _fill_raw_prediction_counts(pack: PackReadiness, records: Sequence[Mapping[str, Any]]) -> None:
    pack.raw_prediction_records = len(records)
    review = 0
    for record in records:
        bucket = _prediction_bucket(record)
        if bool(record.get("needs_review")) or bucket.startswith("needs_review"):
            review += 1
    pack.raw_prediction_review_count = review
    pack.raw_prediction_auto_count = max(0, len(records) - review)


def _prediction_bucket(record: Mapping[str, Any]) -> str:
    quality = record.get("label_quality")
    if not isinstance(quality, Mapping):
        quality = {}
    return str(record.get("bucket") or quality.get("bucket") or "missing")


def _fill_dataset_kind(pack: PackReadiness, dataset_dir: Path) -> None:
    config = _load_json(dataset_dir / "dataset_config.json")
    created_by = str(config.get("created_by", ""))
    name = dataset_dir.name
    pack.is_aggregate_dataset = (
        name.startswith("sprite_lab_multisource")
        or created_by == "spritelab.harvest.merge_datasets"
        or (dataset_dir / "merge_report.json").is_file()
    )
    pack.is_atomic_dataset = (
        not pack.is_aggregate_dataset
        and name.endswith("_label_v2_semantic_v3")
        and created_by != "spritelab.harvest.merge_datasets"
    )
    pack.is_merge_input_candidate = pack.is_atomic_dataset


def _has_exportable_auto_records(pack: PackReadiness) -> bool:
    return pack.apply_report_accepted_auto_labels > 0


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_readiness_reports(report: ReadinessReport, *, out_md: Path | None, out_json: Path | None) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(format_readiness_report(report), encoding="utf-8")


def format_readiness_report(report: ReadinessReport) -> str:
    lines: list[str] = []
    lines.append("# Dataset Readiness Report")
    lines.append("")
    lines.append(f"Runs root: `{report.runs_root}`")
    lines.append(f"Datasets root: `{report.datasets_root}`")
    lines.append(f"Packs scanned: {len(report.packs)}")
    lines.append("")

    actions: Counter[str] = Counter(pack.recommended_action for pack in report.packs)
    lines.append("## Recommended actions")
    lines.append("")
    for action in RECOMMENDED_ACTIONS:
        if actions.get(action):
            lines.append(f"- {action}: {actions[action]}")
    lines.append("")

    ready = [pack for pack in report.packs if pack.recommended_action == "ready_for_merge"]
    lines.append("## Ready for merge")
    lines.append("")
    if ready:
        for pack in ready:
            lines.append(f"- `{pack.exported_dataset_path or pack.run_name}` ({pack.accepted_records} records)")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Packs")
    lines.append("")
    for pack in report.packs:
        lines.append(f"### {pack.run_name}")
        lines.append("")
        lines.append(f"- recommended action: **{pack.recommended_action}**")
        lines.append(f"- exported dataset: {pack.exported_dataset_path or '(none)'}")
        lines.append(
            "- dataset kind: "
            f"atomic={pack.is_atomic_dataset} aggregate={pack.is_aggregate_dataset} "
            f"merge_input_candidate={pack.is_merge_input_candidate}"
        )
        lines.append(f"- dataset QA: {pack.dataset_qa_status}")
        lines.append(f"- training-manifest QA: {pack.training_manifest_status}")
        lines.append(
            f"- records: total={pack.total_records} accepted={pack.accepted_records} "
            f"exported={pack.exported_dataset_records}"
        )
        lines.append(
            "- raw predictions: "
            f"records={pack.raw_prediction_records} auto={pack.raw_prediction_auto_count} "
            f"review={pack.raw_prediction_review_count}"
        )
        lines.append(
            "- apply report: "
            f"applied_auto={pack.apply_report_applied_auto_labels} "
            f"accepted_auto={pack.apply_report_accepted_auto_labels}"
        )
        lines.append(
            f"- QA errors: dataset={pack.dataset_qa_errors} training_manifest={pack.training_manifest_qa_errors}"
        )
        lines.append(f"- review rate: {pack.review_rate:.3f}  auto rate: {pack.auto_rate:.3f}")
        lines.append(
            f"- semantic_v3 coverage: {pack.semantic_v3_coverage:.3f}  "
            f"base_object coverage: {pack.base_object_coverage:.3f}  "
            f"avg captions: {pack.caption_count_average:.2f}"
        )
        lines.append(
            "- predictions: "
            f"label_v2={pack.has_label_v2_predictions} semantic_v3={pack.has_semantic_v3_predictions} "
            f"apply_report={pack.has_apply_report} review_queue={pack.has_review_queue}"
        )
        if pack.category_distribution:
            dist = ", ".join(f"{k}={v}" for k, v in list(pack.category_distribution.items())[:8])
            lines.append(f"- categories: {dist}")
        for reason in pack.top_reasons[:5]:
            lines.append(f"- reason: {reason}")
        for note in pack.notes:
            lines.append(f"- note: {note}")
        lines.append("")
    return "\n".join(lines)

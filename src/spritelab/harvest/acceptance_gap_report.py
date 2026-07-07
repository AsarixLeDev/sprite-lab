"""Read-only report for raw-auto/apply/export acceptance gaps."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.apply_label_v2 import is_raw_auto_prediction, is_review_prediction, is_safe_auto_prediction, prediction_bucket
from spritelab.harvest.build_semantic_dataset import RAW_LABEL_V2_PREDICTION_PREFERENCE, is_raw_label_v2_prediction_file
from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.dataset_readiness import scan_readiness

SPLITS = ("train", "val", "test")
SOURCE_AUTHOR_TOKENS = ("arlantr", "arlan_tr", "rcorre", "buch", "dcss", "bizmasterstudios", "kotnaszynce", "melle")


def build_acceptance_gap_report(runs_root: str | Path, datasets_root: str | Path) -> dict[str, Any]:
    readiness = scan_readiness(runs_root, datasets_root)
    packs: list[dict[str, Any]] = []
    for pack in readiness.packs:
        run_path = Path(pack.run_path) if pack.run_path else None
        raw_predictions = _read_raw_predictions(run_path) if run_path is not None else []
        auto_predictions = [record for record in raw_predictions if is_raw_auto_prediction(record)]
        safe_auto_predictions = [record for record in raw_predictions if is_safe_auto_prediction(record)]
        review_predictions = [record for record in raw_predictions if is_review_prediction(record)]
        apply_report = _load_json(run_path / "label_v2_apply_report.json") if run_path is not None else {}
        dataset_dir = Path(pack.exported_dataset_path) if pack.exported_dataset_path else None
        exported_records = _read_dataset_records(dataset_dir) if dataset_dir is not None else []
        dataset_qa = _load_json(dataset_dir / "dataset_qa_report.json") if dataset_dir is not None else {}
        tm_qa = _load_json(dataset_dir / "training_manifest_qa_report.json") if dataset_dir is not None else {}

        raw_auto_count = len(auto_predictions)
        accepted_auto_count = int(apply_report.get("accepted_auto_labels", pack.apply_report_accepted_auto_labels) or 0)
        exported_count = len(exported_records) if exported_records else int(pack.exported_dataset_records or 0)
        gap_raw_auto_to_accepted = max(0, raw_auto_count - accepted_auto_count)
        gap_accepted_to_exported = max(0, accepted_auto_count - exported_count)
        missing_object_examples = _missing_object_examples(dataset_qa, exported_records)
        review_reasons = _top_review_reasons(review_predictions)
        fix_class = _recommended_fix_class(
            pack_action=pack.recommended_action,
            raw_predictions=raw_predictions,
            auto_predictions=auto_predictions,
            accepted_auto_count=accepted_auto_count,
            exported_count=exported_count,
            missing_object_examples=missing_object_examples,
            apply_report=apply_report,
        )
        estimated = _estimate_recoverable(
            fix_class=fix_class,
            raw_auto_count=raw_auto_count,
            accepted_auto_count=accepted_auto_count,
            exported_count=exported_count,
            apply_report=apply_report,
        )
        packs.append(
            {
                "pack": pack.run_name,
                "run_path": pack.run_path,
                "dataset_path": pack.exported_dataset_path,
                "recommended_action": pack.recommended_action,
                "total_records": int(pack.total_records),
                "raw_prediction_records": len(raw_predictions),
                "raw_auto_count": raw_auto_count,
                "raw_safe_auto_count": len(safe_auto_predictions),
                "raw_review_count": len(review_predictions),
                "apply_applied_auto_count": int(apply_report.get("applied_auto_labels", pack.apply_report_applied_auto_labels) or 0),
                "apply_accepted_auto_count": accepted_auto_count,
                "exported_count": exported_count,
                "dataset_qa_errors": len(dataset_qa.get("errors") or []) if dataset_qa else int(pack.dataset_qa_errors),
                "training_manifest_qa_errors": len(tm_qa.get("errors") or []) if tm_qa else int(pack.training_manifest_qa_errors),
                "gap_raw_auto_to_accepted": gap_raw_auto_to_accepted,
                "gap_accepted_to_exported": gap_accepted_to_exported,
                "top_object_names_in_raw_auto": _top_safe_objects(auto_predictions),
                "top_object_names_in_accepted": dict(Counter(str(record.get("object_name") or "(empty)") for record in exported_records).most_common(20)),
                "top_missing_object_examples": missing_object_examples[:20],
                "top_review_reasons": review_reasons,
                "apply_auto_skip_reasons": dict(apply_report.get("auto_skip_reasons") or {}),
                "apply_auto_validation_counts": dict(apply_report.get("auto_validation_counts") or {}),
                "recommended_fix_class": fix_class,
                "estimated_safe_recoverable_records": estimated,
            }
        )
    packs.sort(key=lambda row: (-int(row["estimated_safe_recoverable_records"]), row["pack"]))
    return {
        "runs_root": str(runs_root),
        "datasets_root": str(datasets_root),
        "pack_count": len(packs),
        "packs": packs,
    }


def write_acceptance_gap_reports(report: Mapping[str, Any], *, out_md: Path | None, out_json: Path | None) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(format_acceptance_gap_report(report), encoding="utf-8")


def format_acceptance_gap_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Acceptance Gap Report",
        "",
        f"Runs root: `{report.get('runs_root', '')}`",
        f"Datasets root: `{report.get('datasets_root', '')}`",
        f"Packs scanned: {int(report.get('pack_count', 0))}",
        "",
        "## Ranked Packs",
        "",
    ]
    for pack in report.get("packs") or ():
        row = dict(pack)
        lines.append(f"### {row.get('pack', '')}")
        lines.append("")
        lines.append(f"- recommended action: {row.get('recommended_action', '')}")
        lines.append(f"- fix class: {row.get('recommended_fix_class', '')}")
        lines.append(f"- estimated safe recoverable records: {int(row.get('estimated_safe_recoverable_records', 0))}")
        lines.append(
            "- counts: "
            f"total={int(row.get('total_records', 0))} "
            f"raw={int(row.get('raw_prediction_records', 0))} "
            f"raw_auto={int(row.get('raw_auto_count', 0))} "
            f"raw_safe_auto={int(row.get('raw_safe_auto_count', 0))} "
            f"raw_review={int(row.get('raw_review_count', 0))} "
            f"applied_auto={int(row.get('apply_applied_auto_count', 0))} "
            f"accepted_auto={int(row.get('apply_accepted_auto_count', 0))} "
            f"exported={int(row.get('exported_count', 0))}"
        )
        lines.append(
            "- gaps: "
            f"raw_auto_to_accepted={int(row.get('gap_raw_auto_to_accepted', 0))} "
            f"accepted_to_exported={int(row.get('gap_accepted_to_exported', 0))}"
        )
        lines.append(
            "- QA errors: "
            f"dataset={int(row.get('dataset_qa_errors', 0))} "
            f"training_manifest={int(row.get('training_manifest_qa_errors', 0))}"
        )
        for title, key in (
            ("raw-auto objects", "top_object_names_in_raw_auto"),
            ("accepted objects", "top_object_names_in_accepted"),
            ("review reasons", "top_review_reasons"),
            ("apply auto skips", "apply_auto_skip_reasons"),
            ("apply auto validation", "apply_auto_validation_counts"),
        ):
            values = dict(row.get(key) or {})
            if values:
                rendered = ", ".join(f"{name}={count}" for name, count in list(values.items())[:8])
                lines.append(f"- {title}: {rendered}")
        examples = list(row.get("top_missing_object_examples") or ())
        for example in examples[:5]:
            lines.append(f"- missing-object example: {example}")
        lines.append("")
    return "\n".join(lines)


def _read_raw_predictions(run_path: Path | None) -> list[dict[str, Any]]:
    if run_path is None or not run_path.is_dir():
        return []
    candidates = [path for path in run_path.glob("label_v2_suggestions*.jsonl") if is_raw_label_v2_prediction_file(path)]
    by_name = {path.name: path for path in candidates}
    ordered = [by_name[name] for name in RAW_LABEL_V2_PREDICTION_PREFERENCE if name in by_name]
    ordered.extend(path for path in sorted(candidates) if path.name not in set(RAW_LABEL_V2_PREDICTION_PREFERENCE))
    return read_jsonl(ordered[0]) if ordered else []


def _read_dataset_records(dataset_dir: Path | None) -> list[dict[str, Any]]:
    if dataset_dir is None:
        return []
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        records.extend(read_jsonl(dataset_dir / f"manifest_{split}.jsonl"))
    return records


def _top_safe_objects(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(str(_safe(record).get("object_name") or "(empty)") for record in records).most_common(20)
    )


def _top_review_reasons(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[prediction_bucket(record)] += 1
        for reason in record.get("conflict_reasons") or ():
            counts[str(reason)] += 1
        quality = record.get("label_quality") if isinstance(record.get("label_quality"), Mapping) else {}
        for reason in quality.get("conflict_reasons") or ():
            counts[str(reason)] += 1
        for flag in record.get("flags") or quality.get("flags") or ():
            counts[str(flag)] += 1
    return dict(counts.most_common(20))


def _missing_object_examples(dataset_qa: Mapping[str, Any], exported_records: Sequence[Mapping[str, Any]]) -> list[str]:
    examples = [
        str(value)
        for value in (dict(dataset_qa.get("manifest_checks") or {}).get("missing_object_name") or ())
        if str(value)
    ]
    for record in exported_records:
        if str(record.get("object_name", "")).strip():
            continue
        sprite_id = str(record.get("sprite_id", ""))
        if sprite_id and sprite_id not in examples:
            examples.append(sprite_id)
    return examples


def _recommended_fix_class(
    *,
    pack_action: str,
    raw_predictions: Sequence[Mapping[str, Any]],
    auto_predictions: Sequence[Mapping[str, Any]],
    accepted_auto_count: int,
    exported_count: int,
    missing_object_examples: Sequence[str],
    apply_report: Mapping[str, Any],
) -> str:
    if not raw_predictions:
        return "empty_import"
    if _has_source_author_pollution(auto_predictions):
        return "source_author_token_pollution"
    if missing_object_examples:
        return "missing_object_name_after_export"
    if auto_predictions and accepted_auto_count == 0:
        skips = dict(apply_report.get("auto_skip_reasons") or {})
        if skips.get("missing_object_name"):
            return "base_object_extractor_gap"
        if skips.get("missing_category"):
            return "category_mapping_gap"
        return "zero_apply_acceptance"
    if len(auto_predictions) > accepted_auto_count:
        return "raw_auto_not_applied"
    if accepted_auto_count > exported_count:
        return "missing_object_name_after_export"
    if pack_action == "empty_or_import_broken":
        return "spritesheet_slice_issue"
    if raw_predictions and not auto_predictions:
        return "base_object_extractor_gap"
    return "needs_manual_golden_seed"


def _estimate_recoverable(
    *,
    fix_class: str,
    raw_auto_count: int,
    accepted_auto_count: int,
    exported_count: int,
    apply_report: Mapping[str, Any],
) -> int:
    invalid = sum(
        int(dict(apply_report.get("auto_validation_counts") or {}).get(reason, 0) or 0)
        for reason in ("missing_object_name", "missing_category", "invalid_safe_prefill")
    )
    if fix_class in {"raw_auto_not_applied", "missing_object_name_after_export", "category_mapping_gap"}:
        return max(0, raw_auto_count - min(accepted_auto_count, exported_count) - invalid)
    if fix_class == "zero_apply_acceptance":
        return max(0, raw_auto_count - invalid)
    return 0


def _has_source_author_pollution(records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        object_name = str(_safe(record).get("object_name") or "").lower()
        if any(token in object_name for token in SOURCE_AUTHOR_TOKENS):
            return True
    return False


def _safe(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("safe_prefill")
    return dict(value) if isinstance(value, Mapping) else {}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}

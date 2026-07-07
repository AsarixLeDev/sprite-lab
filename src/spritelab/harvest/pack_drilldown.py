"""Read-only drilldown report for one harvest pack onboarding state."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.apply_label_v2 import is_safe_auto_prediction, prediction_bucket
from spritelab.harvest.build_semantic_dataset import is_raw_label_v2_prediction_file
from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.semantic_v3 import summarize_semantic_v3_records
from spritelab.harvest.source_profiles import detect_source_profile, source_profile_to_json

SOURCE_AUTHOR_TOKENS = {"arlantr", "arlan_tr", "rcorre", "buch", "dcss", "potion_buch", "potion_rcorre"}
GRID_ONLY_RE = re.compile(r"^(?:r\d+_c\d+|tile_\d+|sprite_\d+)$")


def build_pack_drilldown(
    run_dir: str | Path,
    *,
    prediction_file: str | Path = "label_v2_suggestions.jsonl",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    prediction_path = _resolve_in_run(run_path, prediction_file)
    imported = read_jsonl(run_path / "imported.jsonl")
    predictions = read_jsonl(prediction_path) if prediction_path.is_file() else []
    apply_report = _load_json(run_path / "label_v2_apply_report.json")

    buckets: Counter[str] = Counter(prediction_bucket(record) for record in predictions)
    needs_review = sum(1 for record in predictions if _is_review(record))
    safe_object = sum(1 for record in predictions if _safe(record).get("object_name"))
    safe_category = sum(1 for record in predictions if _safe(record).get("category"))
    candidates = sum(1 for record in predictions if record.get("candidate_object_names"))

    profile = _profile_for(imported, predictions)
    semantic_summary = _semantic_summary(run_path, prediction_path, predictions)
    objects = Counter(str(_safe(record).get("object_name") or "(empty)") for record in predictions)
    categories = Counter(str(_safe(record).get("category") or "unknown") for record in predictions)
    author_examples = _examples(predictions, lambda record: _contains_source_author_token(str(_safe(record).get("object_name", ""))))
    sheet_examples = _examples(predictions, lambda record: _is_sheet_coordinate(record))
    review_examples = _examples(predictions, _is_review)
    auto_examples = _examples(predictions, is_safe_auto_prediction)

    fix_classes = _fix_classes(
        imported=imported,
        predictions=predictions,
        prediction_path=prediction_path,
        needs_review=needs_review,
        author_examples=author_examples,
        sheet_examples=sheet_examples,
        apply_report=apply_report,
    )

    return {
        "run_dir": str(run_path),
        "record_count": len(imported),
        "prediction_file": str(prediction_path),
        "prediction_file_is_raw_label_v2": is_raw_label_v2_prediction_file(prediction_path),
        "raw_prediction_records": len(predictions),
        "raw_prediction_buckets": dict(sorted(buckets.items())),
        "needs_review_count": needs_review,
        "safe_prefill_object_coverage": safe_object,
        "safe_prefill_category_coverage": safe_category,
        "apply_acceptance_count": int(apply_report.get("accepted_auto_labels", 0) or 0),
        "apply_applied_auto_count": int(apply_report.get("applied_auto_labels", 0) or 0),
        "candidate_coverage": candidates,
        "source_profile": source_profile_to_json(profile),
        "filename_trust_mode": profile.filename_trust,
        "semantic_v3_records": int(semantic_summary.get("records", 0) or 0),
        "semantic_v3_base_object_coverage": float(semantic_summary.get("base_object_coverage", 0.0) or 0.0),
        "semantic_v3_warnings": dict(semantic_summary.get("warnings") or {}),
        "top_object_names": dict(objects.most_common(20)),
        "top_categories": dict(categories.most_common(20)),
        "top_source_author_token_objects": dict(
            Counter(str(_safe(record).get("object_name") or "") for record in predictions if _contains_source_author_token(str(_safe(record).get("object_name", "")))).most_common(20)
        ),
        "examples_that_should_stay_review": review_examples[:10],
        "examples_that_could_be_auto_trusted": auto_examples[:10],
        "source_author_token_examples": author_examples[:10],
        "sheet_coordinate_only_examples": sheet_examples[:10],
        "recommended_fix_class": fix_classes[0] if fix_classes else "ready_for_export",
        "recommended_fix_classes": fix_classes or ["ready_for_export"],
    }


def write_pack_drilldown_reports(report: Mapping[str, Any], *, out_md: Path | None, out_json: Path | None) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(format_pack_drilldown(report), encoding="utf-8")


def format_pack_drilldown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Pack Drilldown",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Prediction file: `{report.get('prediction_file', '')}`",
        f"Record count: {int(report.get('record_count', 0))}",
        f"Raw prediction rows: {int(report.get('raw_prediction_records', 0))}",
        f"Needs review: {int(report.get('needs_review_count', 0))}",
        f"Apply accepted auto labels: {int(report.get('apply_acceptance_count', 0))}",
        f"Source profile: `{dict(report.get('source_profile') or {}).get('name', '')}`",
        f"Filename trust mode: `{report.get('filename_trust_mode', '')}`",
        f"Semantic-v3 base object coverage: {float(report.get('semantic_v3_base_object_coverage', 0.0)):.3f}",
        f"Recommended fix class: `{report.get('recommended_fix_class', '')}`",
        "",
        "## Buckets",
    ]
    for name, count in dict(report.get("raw_prediction_buckets") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Coverage"])
    lines.append(f"- safe_prefill object: {int(report.get('safe_prefill_object_coverage', 0))}")
    lines.append(f"- safe_prefill category: {int(report.get('safe_prefill_category_coverage', 0))}")
    lines.append(f"- candidates: {int(report.get('candidate_coverage', 0))}")
    for title, key in (
        ("Top Objects", "top_object_names"),
        ("Top Categories", "top_categories"),
        ("Semantic Warnings", "semantic_v3_warnings"),
        ("Source Author Token Objects", "top_source_author_token_objects"),
    ):
        lines.extend(["", f"## {title}"])
        values = dict(report.get(key) or {})
        if values:
            for name, count in values.items():
                lines.append(f"- {name}: {count}")
        else:
            lines.append("- (none)")
    for title, key in (
        ("Examples That Should Stay Review", "examples_that_should_stay_review"),
        ("Examples That Could Be Auto Trusted", "examples_that_could_be_auto_trusted"),
    ):
        lines.extend(["", f"## {title}"])
        examples = list(report.get(key) or ())
        if examples:
            for example in examples[:10]:
                lines.append(f"- {example}")
        else:
            lines.append("- (none)")
    lines.extend(["", "## Fix Classes"])
    for value in report.get("recommended_fix_classes") or ():
        lines.append(f"- {value}")
    return "\n".join(lines) + "\n"


def _fix_classes(
    *,
    imported: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    prediction_path: Path,
    needs_review: int,
    author_examples: Sequence[str],
    sheet_examples: Sequence[str],
    apply_report: Mapping[str, Any],
) -> list[str]:
    fixes: list[str] = []
    if not is_raw_label_v2_prediction_file(prediction_path):
        fixes.append("semantic_prediction_file_used_as_raw")
    if imported and not predictions:
        fixes.append("label_v2_zero_rows_bug")
    if predictions and needs_review == len(predictions):
        fixes.append("all_predictions_need_review")
    if author_examples:
        fixes.append("source_author_token_misparsed")
    if sheet_examples:
        fixes.append("sheet_coordinate_only")
    if predictions and needs_review and not author_examples and not sheet_examples:
        fixes.append("family_allowlist_missing")
    if int(apply_report.get("accepted_auto_labels", 0) or 0) == 0 and predictions:
        fixes.append("all_predictions_need_review" if needs_review == len(predictions) else "stale_accepted_leakage")
    profile = _profile_for(imported, predictions)
    if profile.name.startswith("kenney"):
        semantic_summary = summarize_semantic_v3_records(predictions)
        if needs_review == len(predictions) or float(semantic_summary.get("base_object_coverage", 0.0) or 0.0) < 0.25:
            fixes.append("base_object_extractor_gap")
            fixes.append("needs_manual_golden_seed")
    return _dedupe(fixes)


def _semantic_summary(run_path: Path, prediction_path: Path, predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if predictions and any(isinstance(record.get("semantic_v3"), Mapping) for record in predictions):
        return summarize_semantic_v3_records(predictions)
    semantic_path = prediction_path.with_name(f"{prediction_path.stem}_semantic_v3.jsonl")
    if semantic_path.is_file():
        return summarize_semantic_v3_records(read_jsonl(semantic_path))
    latest = sorted(run_path.glob("*_semantic_v3.jsonl"))
    if latest:
        return summarize_semantic_v3_records(read_jsonl(latest[0]))
    return {}


def _profile_for(imported: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]):
    for record in predictions:
        profile = record.get("source_profile")
        if isinstance(profile, Mapping) and profile.get("name"):
            return detect_source_profile(
                {
                    "source_id": record.get("source_id", profile.get("name", "")),
                    "source_name": record.get("source_name", profile.get("name", "")),
                    "relative_path": record.get("relative_path", ""),
                    "final_png_path": record.get("final_png_path", ""),
                }
            )
    return detect_source_profile(imported[0] if imported else {})


def _examples(records: Sequence[Mapping[str, Any]], predicate: Any) -> list[str]:
    result: list[str] = []
    for record in records:
        if not predicate(record):
            continue
        safe = _safe(record)
        result.append(
            f"{record.get('sprite_id', '')}: {Path(str(record.get('relative_path') or record.get('filename') or '')).name} -> "
            f"{safe.get('object_name', '')} [{prediction_bucket(record)}]"
        )
    return result


def _safe(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("safe_prefill")
    return dict(value) if isinstance(value, Mapping) else {}


def _is_review(record: Mapping[str, Any]) -> bool:
    return bool(record.get("needs_review")) or prediction_bucket(record).startswith("needs_review")


def _contains_source_author_token(value: str) -> bool:
    normalized = str(value).strip().lower()
    return any(token in normalized for token in SOURCE_AUTHOR_TOKENS)


def _is_sheet_coordinate(record: Mapping[str, Any]) -> bool:
    for key in ("filename", "relative_path", "final_png_path", "sprite_id"):
        stem = Path(str(record.get(key, ""))).stem.lower()
        if GRID_ONLY_RE.fullmatch(stem):
            return True
    return False


def _resolve_in_run(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

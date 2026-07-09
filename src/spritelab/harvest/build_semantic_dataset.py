"""Safe one-pack semantic-v3 dataset build orchestration.

This module chains the *already existing* pipeline steps for a single harvest
run into one safe convenience build:

    semantic-v3
      -> apply-label-v2 (auto-only)
      -> export
      -> dataset-qa (--require-semantic-v3)
      -> build-training-manifest
      -> training-manifest-qa
      -> build-eval-prompts

It reuses the existing functions rather than reimplementing them, and it never
applies review/quarantine records: only safe-auto labels are accepted. Human
labels already present on a record are preserved (``overwrite_human_labels`` is
never set here).

The build fails clearly (``BuildError``) if the prediction file is missing or if
a structural step cannot run. QA failures are recorded on the returned report
and surfaced by the CLI, without corrupting the dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BuildError(RuntimeError):
    """Raised when a one-pack build cannot proceed."""


RAW_LABEL_V2_PREDICTION_PREFERENCE: tuple[str, ...] = (
    "label_v2_suggestions_fresh_qwen.jsonl",
    "label_v2_suggestions_qwen.jsonl",
    "label_v2_suggestions_fresh_novlm.jsonl",
    "label_v2_suggestions.jsonl",
    "label_v2_suggestions_triage.jsonl",
)

SEMANTIC_V3_INPUT_ERROR = (
    "build-semantic-dataset expects raw label-v2 predictions, not "
    "semantic-v3-enriched predictions. Use label_v2_suggestions*.jsonl "
    "without _semantic_v3 in the name."
)


@dataclass
class BuildReport:
    run_dir: str
    dataset_name: str
    prediction_file: str
    semantic_prediction_file: str = ""
    output_dir: str = ""
    accept_auto_only: bool = True
    steps: list[dict[str, Any]] = field(default_factory=list)
    accepted_records: int = 0
    review_queue_size: int = 0
    raw_auto_rows_seen: int = 0
    applied_auto_labels: int = 0
    accepted_auto_labels: int = 0
    auto_rows_skipped: int = 0
    auto_skip_reasons: dict[str, int] = field(default_factory=dict)
    auto_validation_counts: dict[str, int] = field(default_factory=dict)
    dataset_qa_errors: int = 0
    dataset_qa_warnings: int = 0
    training_manifest_rows: int = 0
    training_manifest_qa_errors: int = 0
    eval_prompt_count: int = 0
    ok: bool = True

    def add_step(self, name: str, **details: Any) -> None:
        self.steps.append({"step": name, **details})

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "dataset_name": self.dataset_name,
            "prediction_file": self.prediction_file,
            "semantic_prediction_file": self.semantic_prediction_file,
            "output_dir": self.output_dir,
            "accept_auto_only": self.accept_auto_only,
            "accepted_records": self.accepted_records,
            "review_queue_size": self.review_queue_size,
            "raw_auto_rows_seen": self.raw_auto_rows_seen,
            "applied_auto_labels": self.applied_auto_labels,
            "accepted_auto_labels": self.accepted_auto_labels,
            "auto_rows_skipped": self.auto_rows_skipped,
            "auto_skip_reasons": dict(self.auto_skip_reasons),
            "auto_validation_counts": dict(self.auto_validation_counts),
            "dataset_qa_errors": self.dataset_qa_errors,
            "dataset_qa_warnings": self.dataset_qa_warnings,
            "training_manifest_rows": self.training_manifest_rows,
            "training_manifest_qa_errors": self.training_manifest_qa_errors,
            "eval_prompt_count": self.eval_prompt_count,
            "ok": self.ok,
            "steps": list(self.steps),
        }


def build_semantic_dataset(
    run_dir: str | Path,
    *,
    dataset_name: str,
    output_root: str | Path = "datasets",
    prediction_file: str = "label_v2_suggestions.jsonl",
    max_palette_slots: int = 32,
    accept_auto_only: bool = True,
    caption_policy: str = "mixed",
    variants_per_sprite: int = 8,
    seed: int = 20260706,
    max_captions: int = 8,
    overwrite: bool = False,
) -> BuildReport:
    """Build one exported + QA'd semantic-v3 dataset from a harvest run."""

    from spritelab.harvest.catalog import read_jsonl, write_jsonl
    from spritelab.harvest.semantic_v3 import convert_label_v2_predictions, summarize_semantic_v3_records

    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise BuildError(f"harvest run directory not found: {run_path}")

    prediction_path = resolve_raw_label_v2_prediction_file(run_path, prediction_file)
    if not prediction_path.exists():
        raise BuildError(f"prediction file not found: {prediction_path}")
    if not is_raw_label_v2_prediction_file(prediction_path):
        raise BuildError(SEMANTIC_V3_INPUT_ERROR)
    if not (run_path / "imported.jsonl").is_file():
        raise BuildError(f"imported.jsonl not found in run: {run_path}")

    report = BuildReport(
        run_dir=str(run_path),
        dataset_name=dataset_name,
        prediction_file=prediction_path.name,
        accept_auto_only=bool(accept_auto_only),
    )

    # 1. semantic-v3
    predictions = read_jsonl(prediction_path)
    if not predictions:
        raise BuildError(f"prediction file is empty: {prediction_path}")
    converted = convert_label_v2_predictions(predictions, max_captions=max(1, int(max_captions)))
    build_id = _semantic_build_id(prediction_path, converted)
    semantic_path = prediction_path.with_name(f"{prediction_path.stem}_semantic_v3.jsonl")
    write_jsonl(semantic_path, converted)
    semantic_summary = summarize_semantic_v3_records(converted)
    report.semantic_prediction_file = semantic_path.name
    report.add_step(
        "semantic_v3",
        out=semantic_path.name,
        records=int(semantic_summary.get("records", 0)),
        records_with_semantic_v3=int(semantic_summary.get("records_with_semantic_v3", 0)),
        base_object_coverage=float(semantic_summary.get("base_object_coverage", 0.0)),
        build_id=build_id,
    )

    # 2. apply-label-v2 (auto-only, accept auto). Never touches review records.
    from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions

    apply_report = apply_label_v2_predictions(
        run_path,
        prediction_file=semantic_path.name,
        mode="auto-only",
        accept_auto=bool(accept_auto_only),
        overwrite_human_labels=False,
        build_id=build_id,
        require_semantic_v3_for_auto=True,
    )
    report.raw_auto_rows_seen = int(apply_report.get("raw_auto_rows_seen", 0))
    report.applied_auto_labels = int(apply_report.get("applied_auto_labels", 0))
    report.accepted_auto_labels = int(apply_report.get("accepted_auto_labels", 0))
    report.auto_rows_skipped = int(apply_report.get("auto_rows_skipped", 0))
    report.auto_skip_reasons = {str(k): int(v) for k, v in dict(apply_report.get("auto_skip_reasons") or {}).items()}
    report.auto_validation_counts = {
        str(k): int(v) for k, v in dict(apply_report.get("auto_validation_counts") or {}).items()
    }
    report.review_queue_size = int(apply_report.get("review_queue_size", 0))
    report.add_step(
        "apply_label_v2",
        mode="auto-only",
        accept_auto=bool(accept_auto_only),
        raw_auto_rows_seen=report.raw_auto_rows_seen,
        applied_auto_labels=report.applied_auto_labels,
        accepted_auto_labels=report.accepted_auto_labels,
        auto_rows_skipped=report.auto_rows_skipped,
        auto_skip_reasons=dict(report.auto_skip_reasons),
        auto_validation_counts=dict(report.auto_validation_counts),
        review_queue_size=report.review_queue_size,
        human_labels_preserved=int(apply_report.get("human_labels_preserved", 0)),
    )

    # 3. export accepted, license-checked sprites.
    from spritelab.harvest.cli import _rehydrate_run  # reuse the canonical rehydrator
    from spritelab.harvest.pipeline import export_harvested_dataset

    _, harvested = _rehydrate_run(run_path)
    accepted = [s for s in harvested if _is_current_build_exportable(s, build_id=build_id)]
    if not accepted:
        raise BuildError(
            "no current auto-only accepted sprites to export after apply-label-v2; "
            "the run may need source-family auto labels, semantic-v3 metadata, or manual review"
        )
    export_result = export_harvested_dataset(
        accepted,
        dataset_name=dataset_name,
        output_root=output_root,
        max_palette_slots=max_palette_slots,
        overwrite=overwrite,
    )
    dataset_dir = export_result.output_dir
    report.output_dir = str(dataset_dir)
    report.accepted_records = export_result.accepted_count
    report.add_step(
        "export",
        output_dir=str(dataset_dir),
        train=export_result.train_count,
        val=export_result.val_count,
        test=export_result.test_count,
        warnings=list(export_result.warnings),
    )

    # 4. dataset-qa (--require-semantic-v3)
    from spritelab.dataset_maker.qa import qa_dataset, write_reports

    qa_result = qa_dataset(dataset_dir, require_semantic_v3=True)
    write_reports(
        qa_result,
        out_json=dataset_dir / "dataset_qa_report.json",
        out_md=dataset_dir / "dataset_qa_report.md",
    )
    report.dataset_qa_errors = len(qa_result.errors)
    report.dataset_qa_warnings = len(qa_result.warnings)
    report.add_step("dataset_qa", errors=len(qa_result.errors), warnings=len(qa_result.warnings))
    if qa_result.errors:
        report.ok = False
        _write_build_reports(report, dataset_dir)
        return report

    # 5. build-training-manifest
    from spritelab.dataset_maker.training_manifest import (
        build_training_manifest,
        summarize_training_manifest,
        write_training_manifest,
        write_training_manifest_reports,
    )

    manifest_result = build_training_manifest(
        dataset_dir,
        variants_per_sprite=variants_per_sprite,
        caption_policy=caption_policy,
        seed=seed,
    )
    manifest_path = dataset_dir / "training_manifest.jsonl"
    write_training_manifest(manifest_path, manifest_result.rows)
    manifest_summary = summarize_training_manifest(manifest_result)
    write_training_manifest_reports(
        manifest_summary,
        out_json=dataset_dir / "training_manifest_report.json",
        out_md=dataset_dir / "training_manifest_report.md",
    )
    report.training_manifest_rows = len(manifest_result.rows)
    report.add_step("build_training_manifest", rows=len(manifest_result.rows))

    # 6. training-manifest-qa
    from spritelab.dataset_maker.training_manifest_qa import (
        qa_training_manifest,
        write_training_manifest_qa_reports,
    )

    tm_qa = qa_training_manifest(dataset_dir, manifest_path)
    write_training_manifest_qa_reports(
        tm_qa,
        out_json=dataset_dir / "training_manifest_qa_report.json",
        out_md=dataset_dir / "training_manifest_qa_report.md",
    )
    report.training_manifest_qa_errors = len(tm_qa.errors)
    report.add_step("training_manifest_qa", errors=len(tm_qa.errors), warnings=len(tm_qa.warnings))
    if tm_qa.errors:
        report.ok = False
        _write_build_reports(report, dataset_dir)
        return report

    # 7. build-eval-prompts
    from spritelab.dataset_maker.eval_prompts import (
        build_eval_prompts,
        summarize_eval_prompts,
        write_eval_prompts,
        write_eval_prompts_reports,
    )

    eval_result = build_eval_prompts(dataset_dir, seed=seed)
    eval_path = dataset_dir / "eval_prompts.jsonl"
    write_eval_prompts(eval_path, eval_result.prompts)
    eval_summary = summarize_eval_prompts(eval_result)
    write_eval_prompts_reports(
        eval_summary,
        out_json=dataset_dir / "eval_prompts_report.json",
        out_md=dataset_dir / "eval_prompts_report.md",
    )
    report.eval_prompt_count = len(eval_result.prompts)
    report.add_step("build_eval_prompts", prompts=len(eval_result.prompts))

    _write_build_reports(report, dataset_dir)
    return report


def _write_build_reports(report: BuildReport, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "semantic_dataset_build_report.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_dir / "semantic_dataset_build_report.md").write_text(format_build_report(report), encoding="utf-8")


def format_build_report(report: BuildReport) -> str:
    lines: list[str] = []
    lines.append("# Semantic Dataset Build Report")
    lines.append("")
    lines.append(f"Run: `{report.run_dir}`")
    lines.append(f"Dataset: `{report.dataset_name}`")
    lines.append(f"Output: `{report.output_dir}`")
    lines.append(f"Status: **{'PASS' if report.ok else 'FAIL'}**")
    lines.append(f"Prediction file: `{report.prediction_file}`")
    lines.append(f"Semantic prediction file: `{report.semantic_prediction_file}`")
    lines.append(f"Accept auto only: {report.accept_auto_only}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- accepted records: {report.accepted_records}")
    lines.append(f"- raw auto rows seen: {report.raw_auto_rows_seen}")
    lines.append(f"- applied auto labels: {report.applied_auto_labels}")
    lines.append(f"- accepted auto labels: {report.accepted_auto_labels}")
    lines.append(f"- auto rows skipped: {report.auto_rows_skipped}")
    lines.append(f"- review queue size: {report.review_queue_size}")
    lines.append(f"- dataset QA errors: {report.dataset_qa_errors}")
    lines.append(f"- dataset QA warnings: {report.dataset_qa_warnings}")
    lines.append(f"- training manifest rows: {report.training_manifest_rows}")
    lines.append(f"- training manifest QA errors: {report.training_manifest_qa_errors}")
    lines.append(f"- eval prompts: {report.eval_prompt_count}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    for step in report.steps:
        detail = ", ".join(f"{k}={v}" for k, v in step.items() if k != "step")
        lines.append(f"- {step['step']}: {detail}")
    lines.append("")
    lines.append("## Auto Skip Reasons")
    lines.append("")
    if report.auto_skip_reasons:
        for reason, count in report.auto_skip_reasons.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Auto Validation Counts")
    lines.append("")
    if report.auto_validation_counts:
        for reason, count in report.auto_validation_counts.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _resolve_in_run(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def is_raw_label_v2_prediction_file(path: Path) -> bool:
    """Return true for raw label-v2 prediction JSONL files.

    Semantic-v3-enriched files are outputs of the next pipeline stage and must
    never be fed back into build-semantic-dataset as raw predictions.
    """

    name = Path(path).name
    if not name.endswith(".jsonl"):
        return False
    if "_semantic_v3" in name:
        return False
    return True


def resolve_raw_label_v2_prediction_file(run_dir: Path, value: str | Path | None = None) -> Path:
    """Resolve an explicit or preferred raw label-v2 prediction file in a run."""

    if value is not None:
        path = _resolve_in_run(run_dir, value)
        if not is_raw_label_v2_prediction_file(path):
            raise BuildError(SEMANTIC_V3_INPUT_ERROR)
        return path

    for name in RAW_LABEL_V2_PREDICTION_PREFERENCE:
        path = run_dir / name
        if path.is_file() and is_raw_label_v2_prediction_file(path):
            return path
    candidates = sorted(
        path for path in run_dir.glob("label_v2_suggestions*.jsonl") if is_raw_label_v2_prediction_file(path)
    )
    if candidates:
        return candidates[0]
    return run_dir / RAW_LABEL_V2_PREDICTION_PREFERENCE[-2]


def _semantic_build_id(prediction_path: Path, records: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"semantic_v3:{prediction_path.name}:{digest}"


def _is_current_build_exportable(sprite: Any, *, build_id: str) -> bool:
    if getattr(sprite.final_item, "status", "") != "accepted":
        return False
    metadata = sprite.auto_metadata if isinstance(sprite.auto_metadata, Mapping) else {}
    label_v2 = metadata.get("label_v2")
    if not isinstance(label_v2, Mapping):
        label_v2 = {}
    applied = bool(metadata.get("label_v2_applied")) or label_v2.get("applied") is True
    marker = str(metadata.get("label_v2_applied_at_build_id") or label_v2.get("applied_at_build_id") or "")
    if not applied or marker != build_id:
        return False
    safe = metadata.get("label_v2_safe_prefill")
    if not isinstance(safe, Mapping):
        safe = {}
    semantic = metadata.get("semantic_v3")
    if not isinstance(semantic, Mapping) or not semantic:
        return False
    category = str(safe.get("category", "")).strip()
    return bool(str(safe.get("object_name", "")).strip() and category and category != "unknown")

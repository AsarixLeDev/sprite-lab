"""Safe helper for building a multisource dataset from atomic ready datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.merge_datasets import MergeError, merge_datasets


class BuildMultisourceError(RuntimeError):
    """Raised when the multisource helper cannot proceed."""


@dataclass
class BuildMultisourceReport:
    datasets_root: str
    output_dir: str
    selected_datasets: list[str] = field(default_factory=list)
    excluded_datasets: dict[str, str] = field(default_factory=dict)
    total_records: int = 0
    dataset_qa_errors: int = 0
    training_manifest_rows: int = 0
    training_manifest_qa_errors: int = 0
    eval_prompt_count: int = 0
    ok: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "datasets_root": self.datasets_root,
            "output_dir": self.output_dir,
            "selected_datasets": list(self.selected_datasets),
            "excluded_datasets": dict(self.excluded_datasets),
            "total_records": self.total_records,
            "dataset_qa_errors": self.dataset_qa_errors,
            "training_manifest_rows": self.training_manifest_rows,
            "training_manifest_qa_errors": self.training_manifest_qa_errors,
            "eval_prompt_count": self.eval_prompt_count,
            "ok": self.ok,
        }


def build_multisource_dataset(
    datasets_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260706,
    split_policy: str = "preserve",
    caption_policy: str = "mixed",
    variants_per_sprite: int = 8,
    max_palette_slots: int = 32,
    only_atomic_ready: bool = True,
    overwrite: bool = False,
) -> BuildMultisourceReport:
    datasets_root = Path(datasets_root)
    output_dir = Path(output_dir)
    selected, excluded = select_atomic_ready_datasets(datasets_root, only_atomic_ready=only_atomic_ready)
    if not selected:
        raise BuildMultisourceError("no atomic ready datasets selected")

    report = BuildMultisourceReport(
        datasets_root=str(datasets_root),
        output_dir=str(output_dir),
        selected_datasets=[str(path) for path in selected],
        excluded_datasets=excluded,
    )

    try:
        merge_result = merge_datasets(
            selected,
            output_dir,
            seed=seed,
            split_policy=split_policy,
            max_palette_slots=max_palette_slots,
            overwrite=overwrite,
        )
    except MergeError as exc:
        raise BuildMultisourceError(str(exc)) from exc
    report.total_records = int(merge_result.total_records)
    if merge_result.errors:
        report.ok = False
        _write_reports(report, output_dir)
        return report

    from spritelab.dataset_maker.qa import qa_dataset, write_reports

    qa_result = qa_dataset(output_dir, require_semantic_v3=True)
    write_reports(
        qa_result,
        out_json=output_dir / "dataset_qa_report.json",
        out_md=output_dir / "dataset_qa_report.md",
    )
    report.dataset_qa_errors = len(qa_result.errors)
    if qa_result.errors:
        report.ok = False
        _write_reports(report, output_dir)
        return report

    from spritelab.dataset_maker.training_manifest import (
        build_training_manifest,
        summarize_training_manifest,
        write_training_manifest,
        write_training_manifest_reports,
    )

    manifest = build_training_manifest(
        output_dir,
        variants_per_sprite=variants_per_sprite,
        caption_policy=caption_policy,
        seed=seed,
    )
    manifest_path = output_dir / "training_manifest.jsonl"
    write_training_manifest(manifest_path, manifest.rows)
    manifest_summary = summarize_training_manifest(manifest)
    write_training_manifest_reports(
        manifest_summary,
        out_json=output_dir / "training_manifest_report.json",
        out_md=output_dir / "training_manifest_report.md",
    )
    report.training_manifest_rows = len(manifest.rows)

    from spritelab.dataset_maker.training_manifest_qa import (
        qa_training_manifest,
        write_training_manifest_qa_reports,
    )

    tm_qa = qa_training_manifest(output_dir, manifest_path)
    write_training_manifest_qa_reports(
        tm_qa,
        out_json=output_dir / "training_manifest_qa_report.json",
        out_md=output_dir / "training_manifest_qa_report.md",
    )
    report.training_manifest_qa_errors = len(tm_qa.errors)
    if tm_qa.errors:
        report.ok = False
        _write_reports(report, output_dir)
        return report

    from spritelab.dataset_maker.eval_prompts import (
        build_eval_prompts,
        summarize_eval_prompts,
        write_eval_prompts,
        write_eval_prompts_reports,
    )

    eval_result = build_eval_prompts(output_dir, seed=seed)
    write_eval_prompts(output_dir / "eval_prompts.jsonl", eval_result.prompts)
    write_eval_prompts_reports(
        summarize_eval_prompts(eval_result),
        out_json=output_dir / "eval_prompts_report.json",
        out_md=output_dir / "eval_prompts_report.md",
    )
    report.eval_prompt_count = len(eval_result.prompts)
    _write_reports(report, output_dir)
    return report


def select_atomic_ready_datasets(
    datasets_root: str | Path,
    *,
    only_atomic_ready: bool = True,
) -> tuple[list[Path], dict[str, str]]:
    datasets_root = Path(datasets_root)
    selected: list[Path] = []
    excluded: dict[str, str] = {}
    if not datasets_root.is_dir():
        return selected, excluded
    for dataset_dir in sorted((path for path in datasets_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        reason = _dataset_exclusion_reason(dataset_dir, only_atomic_ready=only_atomic_ready)
        if reason:
            excluded[dataset_dir.name] = reason
        else:
            selected.append(dataset_dir)
    return selected, excluded


def format_build_multisource_report(report: BuildMultisourceReport) -> str:
    lines = [
        "# Build Multisource Report",
        "",
        f"Output: `{report.output_dir}`",
        f"Status: **{'PASS' if report.ok else 'FAIL'}**",
        f"Total records: {report.total_records}",
        f"Dataset QA errors: {report.dataset_qa_errors}",
        f"Training manifest rows: {report.training_manifest_rows}",
        f"Training manifest QA errors: {report.training_manifest_qa_errors}",
        f"Eval prompts: {report.eval_prompt_count}",
        "",
        "## Selected Datasets",
    ]
    for dataset in report.selected_datasets:
        lines.append(f"- {dataset}")
    lines.extend(["", "## Excluded Datasets"])
    if report.excluded_datasets:
        for dataset, reason in sorted(report.excluded_datasets.items()):
            lines.append(f"- {dataset}: {reason}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def _dataset_exclusion_reason(dataset_dir: Path, *, only_atomic_ready: bool) -> str:
    if not _is_exported_dataset(dataset_dir):
        return "not_exported_dataset"
    config = _load_json(dataset_dir / "dataset_config.json")
    if dataset_dir.name.startswith("sprite_lab_multisource") or (dataset_dir / "merge_report.json").is_file():
        return "aggregate_dataset"
    if str(config.get("created_by", "")) == "spritelab.harvest.merge_datasets":
        return "aggregate_dataset"
    if only_atomic_ready and not dataset_dir.name.endswith("_label_v2_semantic_v3"):
        return "legacy_nonsemantic_dataset"
    dataset_qa = _load_json(dataset_dir / "dataset_qa_report.json")
    if not dataset_qa:
        return "missing_dataset_qa"
    if dataset_qa.get("errors"):
        return "dataset_qa_errors"
    training_qa = _load_json(dataset_dir / "training_manifest_qa_report.json")
    if not training_qa:
        return "missing_training_manifest_qa"
    if training_qa.get("errors"):
        return "training_manifest_qa_errors"
    return ""


def _write_reports(report: BuildMultisourceReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_multisource_report.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "build_multisource_report.md").write_text(
        format_build_multisource_report(report), encoding="utf-8"
    )


def _is_exported_dataset(dataset_dir: Path) -> bool:
    return (dataset_dir / "manifest_train.jsonl").is_file() or (dataset_dir / "train.npz").is_file()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, dict) else {}

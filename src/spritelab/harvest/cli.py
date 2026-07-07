"""CLI for the harvest package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from spritelab.harvest.sources import SourceLicense, SourceRecord, normalize_license_name


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m spritelab harvest",
        description="License-aware dataset harvester.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    gui = subparsers.add_parser("gui", help="Launch the harvest GUI.")
    gui.add_argument("--output-root", default="datasets")
    gui.add_argument("--run-root", default="harvest_runs")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int)

    import_zip = subparsers.add_parser("import-zip", help="Import a manually downloaded ZIP.")
    _add_source_args(import_zip)
    _add_import_args(import_zip)
    import_zip.add_argument("--zip", required=True, type=Path, dest="zip_path")

    import_dir = subparsers.add_parser("import-dir", help="Import a local PNG directory.")
    _add_source_args(import_dir)
    _add_import_args(import_dir)
    import_dir.add_argument("--dir", required=True, type=Path, dest="dir_path")

    download_zip = subparsers.add_parser("download-zip", help="Download and import a direct ZIP URL.")
    _add_source_args(download_zip)
    _add_import_args(download_zip)
    download_zip.add_argument("--url", required=True)

    prefill = subparsers.add_parser("qwen-prefill", aliases=["qwen_prefill"], help="Batch Qwen metadata prefill for a run.")
    _add_qwen_prefill_args(prefill)

    filename_prefill = subparsers.add_parser("filename-prefill", help="Suggest metadata from sprite filenames.")
    filename_source = filename_prefill.add_mutually_exclusive_group(required=True)
    filename_source.add_argument("--run", type=Path)
    filename_source.add_argument("--sprite-id")
    filename_prefill.add_argument("--filename", default="")
    filename_prefill.add_argument("--out", type=Path)

    prefill_review = subparsers.add_parser("prefill-review", help="Launch a tiny GUI to compare Qwen and filename suggestions.")
    prefill_review.add_argument("--run", required=True, type=Path)
    prefill_review.add_argument("--host", default="127.0.0.1")
    prefill_review.add_argument("--port", type=int)

    fuse_prefill = subparsers.add_parser("fuse-prefill", help="Fuse filename-rule and Qwen prefill suggestions for a run.")
    fuse_prefill.add_argument("--run", required=True, type=Path)
    fuse_prefill.add_argument("--out", type=Path)
    fuse_prefill.add_argument("--min-qwen-confidence", type=float, default=0.55)
    fuse_prefill.add_argument("--fusion-policy", default="weighted")

    label_v2 = subparsers.add_parser("label-v2", help="Run filename/source-first safe label v2 suggestions.")
    _add_label_v2_args(label_v2, include_vlm_args=True)

    fuse_prefill_v2 = subparsers.add_parser("fuse-prefill-v2", help="Safely fuse existing Qwen suggestions through label v2.")
    _add_label_v2_args(fuse_prefill_v2, include_vlm_args=False)

    label_v2_report = subparsers.add_parser("label-v2-report", help="Print/write a label-v2 run summary.")
    label_v2_report.add_argument("--run", required=True, type=Path)
    label_v2_report.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")

    apply_label_v2 = subparsers.add_parser("apply-label-v2", help="Apply label-v2 suggestions to harvest metadata.")
    apply_label_v2.add_argument("--run", required=True, type=Path)
    apply_label_v2.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    apply_label_v2.add_argument("--mode", default="auto-only", choices=["auto-only", "all", "review-only"])
    accept_auto = apply_label_v2.add_mutually_exclusive_group()
    accept_auto.add_argument("--accept-auto", action="store_true", dest="accept_auto", default=False)
    accept_auto.add_argument("--no-accept-auto", action="store_false", dest="accept_auto")
    apply_label_v2.add_argument("--out-imported", type=Path)
    apply_label_v2.add_argument("--out-review", type=Path)
    apply_label_v2.add_argument("--dry-run", action="store_true")
    apply_label_v2.add_argument("--overwrite-human-labels", action="store_true")

    semantic_v3 = subparsers.add_parser("semantic-v3", help="Add semantic-v3 compositional metadata to label-v2 predictions.")
    semantic_v3.add_argument("--run", required=True, type=Path)
    semantic_v3.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    semantic_v3.add_argument("--out", type=Path, help="Defaults to <prediction-file stem>_semantic_v3.jsonl in the run directory.")
    semantic_v3.add_argument("--max-captions", type=int, default=8)

    semantic_v3_report = subparsers.add_parser("semantic-v3-report", help="Summarize semantic-v3 coverage for a prediction file.")
    semantic_v3_report.add_argument("--run", required=True, type=Path)
    semantic_v3_report.add_argument("--prediction-file", default="label_v2_suggestions_semantic_v3.jsonl")
    semantic_v3_report.add_argument("--out-json", type=Path)
    semantic_v3_report.add_argument("--out-md", type=Path)

    prefill_eval_v2 = subparsers.add_parser("prefill-eval-v2", help="Evaluate label-v2 safe prefill against cross-run golden labels.")
    prefill_eval_v2.add_argument("--golden", required=True, type=Path)
    prefill_eval_v2.add_argument("--runs", required=True, help="Comma-separated harvest run directories.")
    prefill_eval_v2.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    prefill_eval_v2.add_argument("--out", required=True, type=Path)
    prefill_eval_v2.add_argument("--errors-out", type=Path)
    prefill_eval_v2.add_argument("--errors-mode", default="all", choices=["all", "hard", "object", "tag"])

    golden_lint = subparsers.add_parser("golden-lint", help="Report likely golden label inconsistencies.")
    golden_lint.add_argument("--golden", required=True, type=Path)
    golden_lint.add_argument("--fix", action="store_true")
    golden_lint.add_argument("--out", type=Path)

    label_v2_sweep = subparsers.add_parser("label-v2-sweep", help="Sweep label-v2 thresholds against golden labels.")
    label_v2_sweep.add_argument("--golden", required=True, type=Path)
    label_v2_sweep.add_argument("--runs", required=True, help="Comma-separated harvest run directories.")
    label_v2_sweep.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    label_v2_sweep.add_argument("--out", required=True, type=Path)

    golden_sample = subparsers.add_parser("golden-sample", help="Sample sprites for the golden evaluation set.")
    golden_sample.add_argument("--run", required=True, type=Path)
    golden_sample.add_argument("--n", type=int, default=400)
    golden_sample.add_argument("--seed", type=int, default=0)
    golden_sample.add_argument("--stratify-by", default="source_name", help="Comma-separated record fields to stratify on.")
    golden_sample.add_argument("--out", type=Path)

    golden_label = subparsers.add_parser("golden-label", help="Launch the golden-set labeling GUI.")
    golden_label.add_argument("--run", required=True, type=Path)
    golden_label.add_argument("--host", default="127.0.0.1")
    golden_label.add_argument("--port", type=int)
    golden_label.add_argument("--labeler", default="")

    golden_prefill_v2 = subparsers.add_parser("golden-prefill-v2", help="Create assisted golden candidates prefilled from label-v2 suggestions.")
    golden_prefill_v2.add_argument("--run", required=True, type=Path)
    golden_prefill_v2.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    golden_prefill_v2.add_argument("--n", type=int, default=160)
    golden_prefill_v2.add_argument("--seed", type=int, default=496)
    golden_prefill_v2.add_argument("--stratify-by", default="source_profile.name,bucket,safe_prefill.object_name")
    golden_prefill_v2.add_argument("--out", type=Path)
    golden_prefill_v2.add_argument("--overwrite", action="store_true", help="Initialize gold_* fields from label-v2 even when a human golden label already exists.")

    golden_prefill_report = subparsers.add_parser("golden-prefill-report", help="Summarize correction rates for prefilled golden labels.")
    golden_prefill_report.add_argument("--golden", required=True, type=Path)

    assisted_sample = subparsers.add_parser("assisted-golden-sample", help="Write assisted golden correction candidates.")
    assisted_sample.add_argument("--run", required=True, type=Path)
    assisted_sample.add_argument("--n", type=int)
    assisted_sample.add_argument("--seed", type=int, default=1337)
    assisted_sample.add_argument("--include-status", action="append")

    assisted = subparsers.add_parser("assisted-golden", help="Launch assisted golden correction GUI.")
    assisted.add_argument("--run", required=True, type=Path)
    assisted.add_argument("--n", type=int)
    assisted.add_argument("--seed", type=int, default=1337)
    assisted.add_argument("--host", default="127.0.0.1")
    assisted.add_argument("--port", type=int)
    assisted.add_argument("--labeler", default="mathieu")
    assisted.add_argument("--include-status", action="append")
    assisted.add_argument(
        "--order",
        default="needs_review_first",
        choices=["needs_review_first", "random", "source_order", "uncertain_first", "unlabeled_first"],
    )

    prefill_eval = subparsers.add_parser("prefill-eval", help="Evaluate prefill suggestions against golden labels.")
    prefill_eval.add_argument("--run", required=True, type=Path)
    prefill_eval.add_argument("--fused", type=Path, help="Defaults to <run>/fused_suggestions.jsonl.")
    prefill_eval.add_argument("--golden", type=Path, help="Defaults to <run>/golden_labels.jsonl.")
    prefill_eval.add_argument("--out", type=Path, help="Defaults to <run>/prefill_eval.json.")

    policy = subparsers.add_parser("apply-policy", help="Apply a bulk accept/quarantine/reject policy.")
    policy.add_argument("--run", required=True, type=Path)
    policy.add_argument("--auto-accept-valid-cc0", action="store_true")
    policy.add_argument("--auto-accept-own-work", action="store_true")
    policy.add_argument("--auto-accept-allowlisted", action="store_true")
    policy.add_argument("--quarantine-unknown-license", action="store_true")
    policy.add_argument("--quarantine-low-qwen-confidence", action="store_true")
    policy.add_argument("--qwen-confidence-threshold", type=float, default=0.3)
    policy.add_argument("--reject-invalid", action="store_true")

    dataset_qa = subparsers.add_parser("dataset-qa", help="Validate an exported dataset directory (QA gate).")
    dataset_qa.add_argument("--dataset", required=True, type=Path)
    dataset_qa.add_argument("--review-queue", type=Path)
    dataset_qa.add_argument("--out-json", type=Path)
    dataset_qa.add_argument("--out-md", type=Path)
    dataset_qa.add_argument("--sample-contact-sheet", type=Path)
    dataset_qa.add_argument("--no-contact-sheet", action="store_true", help="Skip contact-sheet generation.")
    dataset_qa.add_argument("--sample-limit", type=int, default=64)
    dataset_qa.add_argument("--strict", action="store_true", help="Escalate soft raster/tag warnings to errors.")
    dataset_qa.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero if any warnings exist.")
    dataset_qa.add_argument("--require-semantic-v3", action="store_true", help="Fail records that lack semantic_v3 metadata.")

    build_training_manifest = subparsers.add_parser(
        "build-training-manifest", help="Expand a semantic-v3 dataset into training conditioning rows."
    )
    build_training_manifest.add_argument("--dataset", required=True, type=Path)
    build_training_manifest.add_argument("--out", type=Path, help="Defaults to <dataset>/training_manifest.jsonl.")
    build_training_manifest.add_argument(
        "--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"]
    )
    build_training_manifest.add_argument("--variants-per-sprite", type=int, default=8)
    build_training_manifest.add_argument("--seed", type=int, default=4962026)
    build_training_manifest.add_argument("--per-split", action="store_true", help="Also write training_manifest_{split}.jsonl.")

    training_manifest_qa = subparsers.add_parser(
        "training-manifest-qa", help="Validate a generated training manifest against its source dataset."
    )
    training_manifest_qa.add_argument("--dataset", required=True, type=Path)
    training_manifest_qa.add_argument("--manifest", type=Path, help="Defaults to <dataset>/training_manifest.jsonl.")
    training_manifest_qa.add_argument("--allow-duplicate-captions", action="store_true")
    training_manifest_qa.add_argument("--out-json", type=Path)
    training_manifest_qa.add_argument("--out-md", type=Path)
    training_manifest_qa.add_argument("--fail-on-warning", action="store_true")

    training_manifest_report = subparsers.add_parser(
        "training-manifest-report", help="Summarize an existing training manifest."
    )
    training_manifest_report.add_argument("--manifest", required=True, type=Path)
    training_manifest_report.add_argument("--out-json", type=Path)
    training_manifest_report.add_argument("--out-md", type=Path)

    build_eval_prompts = subparsers.add_parser(
        "build-eval-prompts", help="Generate a fixed evaluation prompt set from a semantic-v3 dataset."
    )
    build_eval_prompts.add_argument("--dataset", required=True, type=Path)
    build_eval_prompts.add_argument("--out", type=Path, help="Defaults to <dataset>/eval_prompts.jsonl.")
    build_eval_prompts.add_argument("--seed", type=int, default=4962026)
    build_eval_prompts.add_argument("--seen-object-count", type=int, default=40)
    build_eval_prompts.add_argument("--unseen-composition-count", type=int, default=40)

    dataset_readiness = subparsers.add_parser(
        "dataset-readiness", help="Scan harvest runs and datasets and report merge readiness."
    )
    dataset_readiness.add_argument("--runs-root", default="harvest_runs", type=Path)
    dataset_readiness.add_argument("--datasets-root", default="datasets", type=Path)
    dataset_readiness.add_argument("--out", type=Path, help="Markdown report path.")
    dataset_readiness.add_argument("--out-json", type=Path, help="JSON report path.")
    dataset_readiness.add_argument("--review-rate-ceiling", type=float, default=0.25)

    acceptance_gap_report = subparsers.add_parser(
        "acceptance-gap-report", help="Rank packs by raw-auto/apply/export acceptance gaps."
    )
    acceptance_gap_report.add_argument("--runs-root", default="harvest_runs", type=Path)
    acceptance_gap_report.add_argument("--datasets-root", default="datasets", type=Path)
    acceptance_gap_report.add_argument("--out", type=Path, help="Markdown report path.")
    acceptance_gap_report.add_argument("--out-json", type=Path, help="JSON report path.")

    pack_drilldown = subparsers.add_parser(
        "pack-drilldown", help="Write a read-only drilldown report for one harvest run."
    )
    pack_drilldown.add_argument("--run", required=True, type=Path)
    pack_drilldown.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    pack_drilldown.add_argument("--out", type=Path, help="Markdown report path.")
    pack_drilldown.add_argument("--out-json", type=Path, help="JSON report path.")

    build_semantic_dataset = subparsers.add_parser(
        "build-semantic-dataset",
        help="Safely build one exported+QA'd semantic-v3 dataset from a harvest run.",
    )
    build_semantic_dataset.add_argument("--run", required=True, type=Path)
    build_semantic_dataset.add_argument("--dataset-name", required=True)
    build_semantic_dataset.add_argument("--output-root", default="datasets", type=Path)
    build_semantic_dataset.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    build_semantic_dataset.add_argument("--max-palette-slots", type=int, default=32)
    build_semantic_dataset.add_argument("--max-captions", type=int, default=8)
    build_semantic_dataset.add_argument("--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"])
    build_semantic_dataset.add_argument("--variants-per-sprite", type=int, default=8)
    build_semantic_dataset.add_argument("--seed", type=int, default=20260706)
    accept_auto = build_semantic_dataset.add_mutually_exclusive_group()
    accept_auto.add_argument("--accept-auto-only", action="store_true", dest="accept_auto_only", default=True)
    accept_auto.add_argument("--no-accept-auto-only", action="store_false", dest="accept_auto_only")
    build_semantic_dataset.add_argument("--overwrite", action="store_true")

    merge_datasets = subparsers.add_parser(
        "merge-datasets", help="Merge exported semantic-v3 datasets into one multi-source dataset."
    )
    merge_datasets.add_argument("--datasets", required=True, nargs="+", type=Path)
    merge_datasets.add_argument("--out", required=True, type=Path)
    merge_datasets.add_argument("--seed", type=int, default=20260706)
    merge_datasets.add_argument("--split-policy", default="preserve", choices=["preserve", "reshuffle"])
    merge_datasets.add_argument("--max-palette-slots", type=int, default=32)
    merge_datasets.add_argument("--overwrite", action="store_true")

    build_multisource = subparsers.add_parser(
        "build-multisource", help="Build a safe multisource dataset from atomic ready datasets."
    )
    build_multisource.add_argument("--datasets-root", default="datasets", type=Path)
    build_multisource.add_argument("--out", required=True, type=Path)
    build_multisource.add_argument("--seed", type=int, default=20260706)
    build_multisource.add_argument("--split-policy", default="preserve", choices=["preserve", "reshuffle"])
    build_multisource.add_argument("--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"])
    build_multisource.add_argument("--variants-per-sprite", type=int, default=8)
    build_multisource.add_argument("--max-palette-slots", type=int, default=32)
    build_multisource.add_argument("--only-atomic-ready", action="store_true", default=False)
    build_multisource.add_argument("--overwrite", action="store_true")

    import_diagnostics = subparsers.add_parser(
        "import-diagnostics", help="Diagnose empty/import-broken harvest run state without mutating it."
    )
    import_diagnostics.add_argument("--run", required=True, type=Path)
    import_diagnostics.add_argument("--out", type=Path, help="Markdown report path.")
    import_diagnostics.add_argument("--out-json", type=Path, help="JSON report path.")

    export = subparsers.add_parser("export", help="Export accepted sprites to a dataset.")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--dataset-name", required=True)
    export.add_argument("--output-root", default="datasets", type=Path)
    export.add_argument("--max-palette-slots", type=int, default=32)
    export.add_argument("--train", type=float, default=0.8)
    export.add_argument("--val", type=float, default=0.1)
    export.add_argument("--test", type=float, default=0.1)
    export.add_argument("--seed", type=int, default=1337)
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--allow-unknown-license", action="store_true")

    parsed = parser.parse_args(argv)
    if parsed.subcommand == "gui":
        _run_gui(parsed)
    elif parsed.subcommand == "import-zip":
        _run_import(parsed, kind="zip")
    elif parsed.subcommand == "import-dir":
        _run_import(parsed, kind="dir")
    elif parsed.subcommand == "download-zip":
        _run_import(parsed, kind="url")
    elif parsed.subcommand in {"qwen-prefill", "qwen_prefill"}:
        _run_qwen_prefill(parsed)
    elif parsed.subcommand == "filename-prefill":
        _run_filename_prefill(parsed)
    elif parsed.subcommand == "prefill-review":
        _run_prefill_review(parsed)
    elif parsed.subcommand == "fuse-prefill":
        _run_fuse_prefill(parsed)
    elif parsed.subcommand == "label-v2":
        _run_label_v2(parsed)
    elif parsed.subcommand == "fuse-prefill-v2":
        _run_fuse_prefill_v2(parsed)
    elif parsed.subcommand == "label-v2-report":
        _run_label_v2_report(parsed)
    elif parsed.subcommand == "apply-label-v2":
        _run_apply_label_v2(parsed)
    elif parsed.subcommand == "semantic-v3":
        _run_semantic_v3(parsed)
    elif parsed.subcommand == "semantic-v3-report":
        _run_semantic_v3_report(parsed)
    elif parsed.subcommand == "prefill-eval-v2":
        _run_prefill_eval_v2(parsed)
    elif parsed.subcommand == "golden-lint":
        _run_golden_lint(parsed)
    elif parsed.subcommand == "label-v2-sweep":
        _run_label_v2_sweep(parsed)
    elif parsed.subcommand == "golden-sample":
        _run_golden_sample(parsed)
    elif parsed.subcommand == "golden-label":
        _run_golden_label(parsed)
    elif parsed.subcommand == "golden-prefill-v2":
        _run_golden_prefill_v2(parsed)
    elif parsed.subcommand == "golden-prefill-report":
        _run_golden_prefill_report(parsed)
    elif parsed.subcommand == "assisted-golden-sample":
        _run_assisted_golden_sample(parsed)
    elif parsed.subcommand == "assisted-golden":
        _run_assisted_golden(parsed)
    elif parsed.subcommand == "prefill-eval":
        _run_prefill_eval(parsed)
    elif parsed.subcommand == "apply-policy":
        _run_apply_policy(parsed)
    elif parsed.subcommand == "dataset-qa":
        _run_dataset_qa(parsed)
    elif parsed.subcommand == "build-training-manifest":
        _run_build_training_manifest(parsed)
    elif parsed.subcommand == "training-manifest-qa":
        _run_training_manifest_qa(parsed)
    elif parsed.subcommand == "training-manifest-report":
        _run_training_manifest_report(parsed)
    elif parsed.subcommand == "build-eval-prompts":
        _run_build_eval_prompts(parsed)
    elif parsed.subcommand == "dataset-readiness":
        _run_dataset_readiness(parsed)
    elif parsed.subcommand == "acceptance-gap-report":
        _run_acceptance_gap_report(parsed)
    elif parsed.subcommand == "pack-drilldown":
        _run_pack_drilldown(parsed)
    elif parsed.subcommand == "build-semantic-dataset":
        _run_build_semantic_dataset(parsed)
    elif parsed.subcommand == "merge-datasets":
        _run_merge_datasets(parsed)
    elif parsed.subcommand == "build-multisource":
        _run_build_multisource(parsed)
    elif parsed.subcommand == "import-diagnostics":
        _run_import_diagnostics(parsed)
    elif parsed.subcommand == "export":
        _run_export(parsed)


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", default="harvest_runs", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-type", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--license-url", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--attribution-required", action="store_true")
    parser.add_argument("--share-alike", action="store_true")
    parser.add_argument("--user-confirmed-license", action="store_true")
    parser.add_argument("--notes", default="")


def _add_import_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-palette-slots", type=int, default=32)
    parser.add_argument("--no-quantize-overcolor", action="store_false", dest="quantize_overcolor")
    parser.add_argument("--no-infer-role-map", action="store_false", dest="infer_role_map")
    parser.add_argument("--no-canonicalize-palette", action="store_false", dest="canonicalize_palette")
    parser.add_argument("--allow-nearest-resize", action="store_true")
    parser.add_argument("--no-center-pad", action="store_false", dest="center_pad")
    parser.add_argument("--slice-sheets", action="store_true", default=True)
    parser.add_argument("--no-slice-sheets", action="store_false", dest="slice_sheets")
    parser.add_argument("--tile-size", type=int, default=32)


def _add_qwen_prefill_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--runpod-token", default="")
    parser.add_argument("--cache-dir", default=".prefill_cache", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent Qwen/Ollama prefill requests.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--backend", default="openai_compatible", choices=["openai_compatible", "ollama", "rule_based"])
    hint = parser.add_mutually_exclusive_group()
    hint.add_argument(
        "--filename-hint",
        action="store_true",
        dest="include_filename_hint",
        default=False,
        help="Embed the filename-rule hint in the labeling prompt (off by default: blind-first).",
    )
    hint.add_argument("--no-filename-hint", action="store_false", dest="include_filename_hint")
    parser.add_argument("--no-adjudicate", action="store_false", dest="adjudicate", default=True)
    parser.add_argument(
        "--adjudication-threshold",
        type=float,
        default=0.6,
        help="Minimum filename-rule confidence for a conflict to trigger the forced-choice call.",
    )
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--no-retry-warning-only", action="store_false", dest="retry_on_warning_only")
    parser.add_argument("--min-qwen-confidence", type=float, default=0.55)
    parser.add_argument("--fusion-policy", default="weighted")
    parser.add_argument(
        "--structured-output",
        default="auto",
        choices=["auto", "on", "off"],
        help="Enforce the JSON schema at decode time (vLLM response_format / Ollama format).",
    )
    parser.add_argument("--votes", type=int, default=3, help="Self-consistency samples when voting triggers.")
    parser.add_argument(
        "--vote-mode",
        default="adaptive",
        choices=["adaptive", "always", "off"],
        help="adaptive: vote only when the first answer looks weak; always: vote on every sprite.",
    )
    parser.add_argument("--vote-temperature", type=float, default=0.5)
    parser.add_argument("--vlm-role", default="labeler", choices=["labeler", "descriptor", "verifier"])
    parser.add_argument(
        "--no-propagate-dups",
        action="store_false",
        dest="propagate_dups",
        default=True,
        help="Disable labeling exact-duplicate images once and copying the result.",
    )
    parser.add_argument("--propagate-near-dups", action="store_true", default=False)
    parser.add_argument("--near-dup-threshold", type=int, default=2)


def _add_label_v2_args(parser: argparse.ArgumentParser, *, include_vlm_args: bool) -> None:
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--include-status",
        action="append",
        help="Imported sprite status to include (repeatable/comma-separated). Defaults to accepted,quarantine,needs_fix; use all for every status.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--trusted-filename-threshold", type=float, default=0.85)
    parser.add_argument("--auto-vlm-threshold", type=float, default=0.80)
    conflict = parser.add_mutually_exclusive_group()
    conflict.add_argument("--review-conflicts", action="store_true", dest="review_conflicts", default=False)
    conflict.add_argument("--auto-trusted-filename-conflicts", action="store_false", dest="review_conflicts")
    dup = parser.add_mutually_exclusive_group()
    dup.add_argument("--propagate-dups", action="store_true", dest="propagate_dups", default=True)
    dup.add_argument("--no-propagate-dups", action="store_false", dest="propagate_dups")
    parser.add_argument("--propagate-near-dups", action="store_true", default=False)
    parser.add_argument("--near-dup-threshold", type=float, default=2.0)
    if include_vlm_args:
        vlm = parser.add_mutually_exclusive_group()
        vlm.add_argument("--use-vlm", action="store_true", dest="use_vlm", default=True)
        vlm.add_argument("--no-vlm", action="store_false", dest="use_vlm")
        parser.add_argument("--refresh-vlm", action="store_true", help="Ignore existing qwen_suggestions.jsonl and call the configured VLM backend when enabled.")
        parser.add_argument("--ignore-existing-vlm", action="store_true", help="Ignore qwen_suggestions.jsonl for this label-v2 run.")
        parser.add_argument("--vlm-only-when-needed", action="store_true")
        parser.add_argument("--vlm-role", default="descriptor", choices=["labeler", "descriptor", "verifier"])
        parser.add_argument("--backend", default="none", choices=["none", "openai_compatible", "ollama", "rule_based"])
        parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
        parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        parser.add_argument("--api-key", default="not-needed")
        parser.add_argument("--runpod-token", default="")
        parser.add_argument("--cache-dir", default=".prefill_cache_label_v2", type=Path)
        parser.add_argument("--timeout-seconds", type=float, default=60.0)
        parser.add_argument("--structured-output", default="auto", choices=["auto", "on", "off"])
        parser.add_argument("--vlm-image-view", default="both", choices=["full", "crop", "both"])


def _build_source(parsed: argparse.Namespace, *, kind: str) -> SourceRecord:
    source_type = parsed.source_type or {"zip": "manual_zip", "dir": "local_directory", "url": "direct_zip_url"}[kind]
    return SourceRecord(
        source_id=parsed.source_id,
        source_name=parsed.source_name,
        source_type=source_type,
        source_url=parsed.source_url,
        download_url=parsed.url if kind == "url" else "",
        local_archive_path=str(parsed.zip_path) if kind == "zip" else "",
        local_root_path=str(parsed.dir_path) if kind == "dir" else "",
        author=parsed.author,
        license=SourceLicense(
            license=normalize_license_name(parsed.license),
            license_url=parsed.license_url,
            attribution_required=parsed.attribution_required,
            share_alike=parsed.share_alike,
            user_confirmed=parsed.user_confirmed_license,
        ),
        notes=parsed.notes,
    )


def _run_import(parsed: argparse.Namespace, *, kind: str) -> None:
    from spritelab.harvest.catalog import (
        append_harvest_event,
        write_candidates_jsonl,
        write_imported_jsonl,
        write_sources_jsonl,
    )
    from spritelab.harvest.pipeline import HarvestImportOptions, harvest_source_to_imported_sprites
    from spritelab.harvest.report import write_harvest_reports
    from spritelab.harvest.sheets import SheetSliceConfig

    source = _build_source(parsed, kind=kind)
    run_dir = Path(parsed.run_root) / parsed.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if kind == "zip" and not source.local_archive_path:
        raise SystemExit("--zip is required for import-zip")

    options = HarvestImportOptions(
        max_palette_slots=parsed.max_palette_slots,
        quantize_overcolor=parsed.quantize_overcolor,
        allow_nearest_resize=parsed.allow_nearest_resize,
        allow_center_pad_to_32=parsed.center_pad,
        infer_role_map=parsed.infer_role_map,
        canonicalize_palette=parsed.canonicalize_palette,
        slice_sheets=parsed.slice_sheets,
        sheet_config=SheetSliceConfig(tile_width=parsed.tile_size, tile_height=parsed.tile_size),
    )
    harvested = harvest_source_to_imported_sprites(source, options=options, work_dir=run_dir)
    candidates = _unique_candidates(harvested)

    write_sources_jsonl(run_dir, [source])
    write_candidates_jsonl(run_dir, candidates)
    write_imported_jsonl(run_dir, harvested)
    write_harvest_reports(run_dir, [source], harvested)
    append_harvest_event(run_dir, "import", {"source_id": source.source_id, "count": len(harvested)})

    valid = sum(1 for sprite in harvested if sprite.imported.bundle is not None)
    print(f"Run: {run_dir}")
    print(f"Candidates: {len(candidates)}")
    print(f"Imported: {len(harvested)}")
    print(f"Valid: {valid}")
    print(f"Invalid: {len(harvested) - valid}")


def _run_qwen_prefill(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.autolabel import QwenBatchPrefillConfig, batch_prefill_with_qwen
    from spritelab.harvest.catalog import append_harvest_event, write_imported_jsonl, write_jsonl

    run_dir = Path(parsed.run)
    sources, harvested = _rehydrate_run(run_dir)
    config = QwenBatchPrefillConfig(
        enabled=True,
        model=parsed.model,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        runpod_token=parsed.runpod_token,
        timeout_seconds=parsed.timeout_seconds,
        cache_dir=parsed.cache_dir,
        max_items=parsed.max_items,
        workers=parsed.workers,
        backend=parsed.backend,
        include_filename_hint=parsed.include_filename_hint,
        adjudicate=parsed.adjudicate,
        adjudication_threshold=parsed.adjudication_threshold,
        retry_attempts=parsed.retry_attempts,
        retry_on_warning_only=parsed.retry_on_warning_only,
        min_qwen_confidence=parsed.min_qwen_confidence,
        fusion_policy=parsed.fusion_policy,
        structured_output=parsed.structured_output,
        votes=parsed.votes,
        vote_mode=parsed.vote_mode,
        vote_temperature=parsed.vote_temperature,
        vlm_role=parsed.vlm_role,
        propagate_dups=parsed.propagate_dups,
        propagate_near_dups=parsed.propagate_near_dups,
        near_dup_threshold=parsed.near_dup_threshold,
    )
    updated = batch_prefill_with_qwen(harvested, config)
    write_imported_jsonl(run_dir, updated)
    write_jsonl(
        run_dir / "qwen_suggestions.jsonl",
        [
            {
                "sprite_id": sprite.final_item.sprite_id,
                **sprite.auto_metadata["qwen_suggestion"],
                **_prefill_propagation_metadata(sprite),
            }
            for sprite in updated
            if "qwen_suggestion" in sprite.auto_metadata
        ],
    )
    write_jsonl(
        run_dir / "fused_suggestions.jsonl",
        [
            {
                "sprite_id": sprite.final_item.sprite_id,
                "filename_suggestion": sprite.auto_metadata.get("filename_suggestion", {}),
                "qwen_suggestion": sprite.auto_metadata.get("qwen_suggestion", {}),
                "fused_suggestion": sprite.auto_metadata.get("fused_suggestion", {}),
                "prefill_quality": sprite.auto_metadata.get("prefill_quality", {}),
                **_prefill_propagation_metadata(sprite),
            }
            for sprite in updated
            if "fused_suggestion" in sprite.auto_metadata
        ],
    )
    propagation_counts = _prefill_propagation_counts(updated)
    append_harvest_event(
        run_dir,
        "qwen_prefill",
        {
            "count": len(updated),
            "workers": max(1, int(parsed.workers or 1)),
            **propagation_counts,
        },
    )
    suggested = sum(1 for sprite in updated if "qwen_suggestion" in sprite.auto_metadata)
    failed = sum(1 for sprite in updated if "qwen_error" in sprite.auto_metadata)
    quality_counts = _quality_counts_from_harvested(updated)
    print(f"Prefilled: {suggested}")
    print(f"Failed: {failed}")
    print(f"Workers: {max(1, int(parsed.workers or 1))}")
    print(f"Propagated exact duplicates: {propagation_counts['propagated_exact_duplicates']}")
    print(f"Propagated near duplicates: {propagation_counts['propagated_near_duplicates']}")
    for bucket, count in sorted(quality_counts.items()):
        print(f"{bucket}: {count}")


def _run_filename_prefill(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl, write_jsonl
    from spritelab.harvest.filename_rules import filename_suggestion_to_dict, parse_filename_metadata

    if parsed.run is None:
        suggestion = parse_filename_metadata(parsed.sprite_id, filename=parsed.filename or None)
        print(json.dumps(filename_suggestion_to_dict(suggestion), sort_keys=True))
        return

    run_dir = Path(parsed.run)
    records = []
    for record in read_jsonl(run_dir / "imported.jsonl"):
        filename = Path(str(record.get("relative_path") or record.get("final_png_path", ""))).name
        suggestion = parse_filename_metadata(str(record.get("sprite_id", "")), filename=filename)
        records.append(
            {
                "sprite_id": record.get("sprite_id", ""),
                "filename": filename,
                **filename_suggestion_to_dict(suggestion),
            }
        )
    output_path = parsed.out or (run_dir / "filename_suggestions.jsonl")
    write_jsonl(output_path, records)
    print(f"Wrote: {output_path}")
    print(f"Suggestions: {len(records)}")


def _run_prefill_review(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.prefill_review_gui import launch_prefill_review_gui

    try:
        launch_prefill_review_gui(parsed.run, host=parsed.host, port=parsed.port)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)


def _run_fuse_prefill(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import read_jsonl, write_jsonl
    from spritelab.harvest.filename_rules import filename_suggestion_to_dict, parse_filename_metadata
    from spritelab.harvest.prefill_fusion import fuse_prefill_suggestions, summarize_prefill_quality

    run_dir = Path(parsed.run)
    qwen_by_id = {
        str(record.get("sprite_id", "")): {key: value for key, value in record.items() if key != "sprite_id"}
        for record in read_jsonl(run_dir / "qwen_suggestions.jsonl")
    }
    records = []
    for record in read_jsonl(run_dir / "imported.jsonl"):
        sprite_id = str(record.get("sprite_id", ""))
        auto_metadata = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), dict) else {}
        filename = Path(str(record.get("relative_path") or record.get("final_png_path", ""))).name
        filename_suggestion = parse_filename_metadata(sprite_id, filename=filename)
        filename_dict = dict(auto_metadata.get("filename_suggestion") or filename_suggestion_to_dict(filename_suggestion))
        qwen_suggestion = dict(auto_metadata.get("qwen_suggestion") or qwen_by_id.get(sprite_id, {}))
        adjudication = auto_metadata.get("adjudication") if isinstance(auto_metadata.get("adjudication"), dict) else None
        fused = fuse_prefill_suggestions(
            filename_suggestion,
            qwen_suggestion,
            min_qwen_confidence=parsed.min_qwen_confidence,
            fusion_policy=parsed.fusion_policy,
            adjudication=adjudication,
        )
        records.append(
            {
                "sprite_id": sprite_id,
                "filename": filename,
                "filename_suggestion": filename_dict,
                "qwen_suggestion": qwen_suggestion,
                "fused_suggestion": fused.fused_suggestion,
                "prefill_quality": fused.prefill_quality,
            }
        )
    output_path = parsed.out or (run_dir / "fused_suggestions.jsonl")
    write_jsonl(output_path, records)
    print(f"Wrote: {output_path}")
    print(f"Suggestions: {len(records)}")
    for bucket, count in summarize_prefill_quality(records).items():
        print(f"{bucket}: {count}")


def _run_label_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v2_pipeline import (
        build_label_v2_records,
        create_vlm_backend_from_args,
        select_label_v2_input_records,
        summarize_label_v2_records,
        write_label_v2_outputs,
    )

    backend = create_vlm_backend_from_args(parsed) if parsed.use_vlm else None
    input_selection = select_label_v2_input_records(parsed.run, include_statuses=parsed.include_status)
    records = build_label_v2_records(
        parsed.run,
        use_vlm=bool(parsed.use_vlm),
        vlm_only_when_needed=bool(parsed.vlm_only_when_needed),
        max_items=parsed.max_items,
        propagate_dups=bool(parsed.propagate_dups),
        trusted_filename_threshold=parsed.trusted_filename_threshold,
        auto_vlm_threshold=parsed.auto_vlm_threshold,
        review_conflicts=bool(parsed.review_conflicts),
        backend=backend,
        refresh_vlm=bool(parsed.refresh_vlm),
        ignore_existing_vlm=bool(parsed.ignore_existing_vlm),
        workers=parsed.workers,
        include_statuses=parsed.include_status,
    )
    paths = write_label_v2_outputs(parsed.run, records, out=parsed.out, input_selection=input_selection)
    summary = summarize_label_v2_records(records)
    summary["input_selection"] = input_selection.to_summary()
    print(f"Wrote: {paths['suggestions']}")
    print(f"Suggestions: {summary['total']}")
    print(f"Needs review: {summary['needs_review_count']}")
    for reason, count in summary.get("input_selection", {}).get("skipped_by_reason", {}).items():
        print(f"{reason}: {count}")
    print(f"Workers: {max(1, int(parsed.workers or 1))}")
    for key, count in summary.get("vlm_stats", {}).items():
        print(f"{key}: {count}")
    for bucket, count in summary["buckets"].items():
        print(f"{bucket}: {count}")


def _run_fuse_prefill_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v2_pipeline import (
        build_label_v2_records,
        select_label_v2_input_records,
        summarize_label_v2_records,
        write_label_v2_outputs,
    )

    input_selection = select_label_v2_input_records(parsed.run, include_statuses=parsed.include_status)
    records = build_label_v2_records(
        parsed.run,
        use_vlm=True,
        vlm_only_when_needed=False,
        max_items=parsed.max_items,
        propagate_dups=bool(parsed.propagate_dups),
        trusted_filename_threshold=parsed.trusted_filename_threshold,
        auto_vlm_threshold=parsed.auto_vlm_threshold,
        review_conflicts=bool(parsed.review_conflicts),
        backend=None,
        workers=parsed.workers,
        include_statuses=parsed.include_status,
    )
    paths = write_label_v2_outputs(parsed.run, records, out=parsed.out, input_selection=input_selection)
    summary = summarize_label_v2_records(records)
    summary["input_selection"] = input_selection.to_summary()
    print(f"Wrote: {paths['suggestions']}")
    print(f"Suggestions: {summary['total']}")
    print(f"Needs review: {summary['needs_review_count']}")
    for reason, count in summary.get("input_selection", {}).get("skipped_by_reason", {}).items():
        print(f"{reason}: {count}")
    print(f"Workers: {max(1, int(parsed.workers or 1))}")
    for bucket, count in summary["buckets"].items():
        print(f"{bucket}: {count}")


def _run_label_v2_report(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.label_v2_pipeline import format_label_v2_run_report, summarize_label_v2_records

    run_dir = Path(parsed.run)
    records = read_jsonl(run_dir / parsed.prediction_file)
    summary = summarize_label_v2_records(records)
    summary_path = run_dir / "label_v2_summary.json"
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if isinstance(previous, dict) and isinstance(previous.get("input_selection"), dict):
            summary["input_selection"] = previous["input_selection"]
    report = format_label_v2_run_report(summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "label_v2_report.md").write_text(report, encoding="utf-8")
    print(report, end="")


def _run_apply_label_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions, format_apply_summary

    report = apply_label_v2_predictions(
        parsed.run,
        prediction_file=parsed.prediction_file,
        mode=parsed.mode,
        accept_auto=bool(parsed.accept_auto),
        out_imported=parsed.out_imported,
        out_review=parsed.out_review,
        dry_run=bool(parsed.dry_run),
        overwrite_human_labels=bool(parsed.overwrite_human_labels),
    )
    print(format_apply_summary(report), end="")


def _run_semantic_v3(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event, read_jsonl, write_jsonl
    from spritelab.harvest.semantic_v3 import convert_label_v2_predictions, summarize_semantic_v3_records

    run_dir = Path(parsed.run)
    prediction_path = _resolve_in_run(run_dir, parsed.prediction_file)
    if not prediction_path.exists():
        raise SystemExit(f"prediction file not found: {prediction_path}")
    if parsed.out is not None:
        output_path = _resolve_in_run(run_dir, parsed.out)
    else:
        output_path = prediction_path.with_name(f"{prediction_path.stem}_semantic_v3.jsonl")

    predictions = read_jsonl(prediction_path)
    converted = convert_label_v2_predictions(predictions, max_captions=max(1, int(parsed.max_captions)))
    write_jsonl(output_path, converted)
    summary = summarize_semantic_v3_records(converted)
    append_harvest_event(
        run_dir,
        "semantic_v3",
        {
            "prediction_file": prediction_path.name,
            "out": output_path.name,
            "records": summary["records"],
            "records_with_semantic_v3": summary["records_with_semantic_v3"],
        },
    )
    print(f"Wrote: {output_path}")
    print(f"Records: {summary['records']}")
    print(f"Records with semantic_v3: {summary['records_with_semantic_v3']}")
    print(f"Average captions: {summary['average_captions']:.2f}")
    print(f"Base object coverage: {summary['base_object_coverage']:.3f}")
    for warning, count in dict(summary.get("warnings") or {}).items():
        print(f"warning {warning}: {count}")


def _run_semantic_v3_report(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.semantic_v3 import format_semantic_v3_report, summarize_semantic_v3_records

    run_dir = Path(parsed.run)
    prediction_path = _resolve_in_run(run_dir, parsed.prediction_file)
    if not prediction_path.exists():
        raise SystemExit(f"prediction file not found: {prediction_path}")
    records = read_jsonl(prediction_path)
    summary = summarize_semantic_v3_records(records)
    report = format_semantic_v3_report(summary)
    if parsed.out_json is not None:
        out_json = _resolve_in_run(run_dir, parsed.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {out_json}")
    if parsed.out_md is not None:
        out_md = _resolve_in_run(run_dir, parsed.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        print(f"Wrote: {out_md}")
    print(report, end="")


def _resolve_in_run(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _run_prefill_eval_v2(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.golden import load_golden_labels
    from spritelab.harvest.label_v2_eval import (
        evaluate_label_v2,
        format_label_v2_report,
        label_v2_error_records,
        load_label_v2_predictions,
    )

    runs = _parse_runs_arg(parsed.runs)
    golden = load_golden_labels(parsed.golden)
    records = load_label_v2_predictions(runs, prediction_file=parsed.prediction_file)
    result = evaluate_label_v2(golden, records)
    result["golden_path"] = str(parsed.golden)
    result["runs"] = [str(run) for run in runs]
    result["prediction_file"] = parsed.prediction_file
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if parsed.errors_out:
        from spritelab.harvest.catalog import write_jsonl

        errors = label_v2_error_records(golden, records, errors_mode=parsed.errors_mode)
        parsed.errors_out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(parsed.errors_out, errors)
    print(format_label_v2_report(result))
    print(f"\nWrote: {parsed.out}")
    if parsed.errors_out:
        print(f"Wrote errors: {parsed.errors_out}")


def _run_golden_lint(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.golden_lint import format_golden_lint_report, lint_golden_file, write_jsonl

    issues, suggestions = lint_golden_file(parsed.golden, fix=bool(parsed.fix))
    print(format_golden_lint_report(issues), end="")
    if parsed.fix:
        if parsed.out is None:
            raise SystemExit("--fix requires --out so the golden labels are not mutated in place.")
        write_jsonl(parsed.out, suggestions)
        print(f"Wrote suggestions: {parsed.out}")


def _run_label_v2_sweep(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.golden import load_golden_labels
    from spritelab.harvest.label_v2_eval import load_label_v2_predictions, sweep_label_v2_operating_points

    runs = _parse_runs_arg(parsed.runs)
    golden = load_golden_labels(parsed.golden)
    records = load_label_v2_predictions(runs, prediction_file=parsed.prediction_file)
    result = sweep_label_v2_operating_points(golden, records)
    result["golden_path"] = str(parsed.golden)
    result["runs"] = [str(run) for run in runs]
    result["prediction_file"] = parsed.prediction_file
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = result.get("best") or {}
    print("Best operating point:")
    for key in ("trusted_filename_threshold", "vlm_threshold", "conflict_policy", "auto_coverage", "auto_precision", "object_token_f1"):
        if key in best:
            print(f"{key}: {best[key]}")
    print(f"\nWrote: {parsed.out}")


def _run_golden_sample(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event, read_jsonl, write_jsonl
    from spritelab.harvest.golden import sample_golden_candidates

    run_dir = Path(parsed.run)
    records = read_jsonl(run_dir / "imported.jsonl")
    stratify_by = tuple(field.strip() for field in str(parsed.stratify_by).split(",") if field.strip())
    sample = sample_golden_candidates(records, parsed.n, stratify_by=stratify_by or ("source_name",), seed=parsed.seed)
    output_path = parsed.out or (run_dir / "golden_sample.jsonl")
    write_jsonl(output_path, sample)
    append_harvest_event(run_dir, "golden_sample", {"count": len(sample), "seed": parsed.seed})
    print(f"Wrote: {output_path}")
    print(f"Sampled: {len(sample)} of {len(records)}")


def _run_golden_label(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.prefill_review_gui import launch_golden_label_gui

    try:
        launch_golden_label_gui(parsed.run, host=parsed.host, port=parsed.port, labeler=parsed.labeler)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)


def _run_golden_prefill_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import build_label_v2_prefilled_candidates, write_golden_candidates_jsonl
    from spritelab.harvest.catalog import append_harvest_event

    run_dir = Path(parsed.run)
    stratify_by = tuple(field.strip() for field in str(parsed.stratify_by).split(",") if field.strip())
    candidates = build_label_v2_prefilled_candidates(
        run_dir,
        prediction_file=parsed.prediction_file,
        n=parsed.n,
        seed=parsed.seed,
        stratify_by=stratify_by or ("source_profile.name", "bucket", "safe_prefill.object_name"),
        overwrite=bool(parsed.overwrite),
    )
    output_path = parsed.out or (run_dir / "golden_candidates_prefilled.jsonl")
    write_golden_candidates_jsonl(output_path, candidates)
    append_harvest_event(
        run_dir,
        "golden_prefill_v2",
        {
            "count": len(candidates),
            "seed": parsed.seed,
            "prediction_file": str(parsed.prediction_file),
            "stratify_by": list(stratify_by),
            "overwrite": bool(parsed.overwrite),
        },
    )
    print(f"Wrote: {output_path}")
    print(f"Candidates: {len(candidates)}")


def _run_golden_prefill_report(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import format_golden_prefill_report, summarize_golden_prefill_records
    from spritelab.harvest.catalog import read_jsonl

    records = read_jsonl(parsed.golden)
    print(format_golden_prefill_report(summarize_golden_prefill_records(records)), end="")


def _run_assisted_golden_sample(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import (
        GOLDEN_CANDIDATES_FILENAME,
        load_assisted_candidates,
        write_golden_candidates_jsonl,
    )
    from spritelab.harvest.catalog import append_harvest_event

    run_dir = Path(parsed.run)
    include_statuses = _parse_include_statuses(parsed.include_status)
    candidates = load_assisted_candidates(
        run_dir,
        n=parsed.n,
        seed=parsed.seed,
        include_statuses=include_statuses,
    )
    output_path = run_dir / GOLDEN_CANDIDATES_FILENAME
    write_golden_candidates_jsonl(output_path, candidates)
    append_harvest_event(
        run_dir,
        "assisted_golden_sample",
        {"count": len(candidates), "seed": parsed.seed, "include_statuses": list(include_statuses)},
    )
    print(f"Wrote: {output_path}")
    print(f"Candidates: {len(candidates)}")


def _run_assisted_golden(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden_gui import launch_assisted_golden_gui

    include_statuses = _parse_include_statuses(parsed.include_status)
    try:
        launch_assisted_golden_gui(
            parsed.run,
            n=parsed.n,
            seed=parsed.seed,
            host=parsed.host,
            port=parsed.port,
            labeler=parsed.labeler,
            include_statuses=include_statuses,
            order=parsed.order,
        )
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)


def _run_prefill_eval(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.golden import load_golden_labels
    from spritelab.harvest.prefill_eval import evaluate_prefill, format_eval_report

    run_dir = Path(parsed.run)
    golden_path = parsed.golden or (run_dir / "golden_labels.jsonl")
    fused_path = parsed.fused or (run_dir / "fused_suggestions.jsonl")
    golden = load_golden_labels(golden_path)
    if not golden:
        print(f"No golden labels found at {golden_path}. Run `harvest golden-sample` then `harvest golden-label` first.")
        raise SystemExit(1)
    fused_records = read_jsonl(fused_path)
    if not fused_records:
        print(f"No fused suggestions found at {fused_path}. Run `harvest qwen-prefill` or `harvest fuse-prefill` first.")
        raise SystemExit(1)

    result = evaluate_prefill(golden, fused_records)
    result["golden_path"] = str(golden_path)
    result["fused_path"] = str(fused_path)
    from spritelab.dataset_maker.prefill import PROMPT_VERSION

    result["prompt_version"] = PROMPT_VERSION
    output_path = parsed.out or (run_dir / "prefill_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_eval_report(result))
    print(f"\nWrote: {output_path}")


def _run_apply_policy(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event, write_imported_jsonl
    from spritelab.harvest.pipeline import HarvestPolicy, apply_harvest_policy
    from spritelab.harvest.report import write_harvest_reports

    run_dir = Path(parsed.run)
    sources, harvested = _rehydrate_run(run_dir)
    policy = HarvestPolicy(
        auto_accept_valid_cc0=parsed.auto_accept_valid_cc0,
        auto_accept_own_work=parsed.auto_accept_own_work,
        auto_accept_allowlisted=parsed.auto_accept_allowlisted,
        quarantine_unknown_license=parsed.quarantine_unknown_license,
        quarantine_low_qwen_confidence=parsed.quarantine_low_qwen_confidence,
        qwen_confidence_threshold=parsed.qwen_confidence_threshold,
        reject_invalid=parsed.reject_invalid,
    )
    updated = apply_harvest_policy(harvested, policy)
    write_imported_jsonl(run_dir, updated)
    write_harvest_reports(run_dir, sources, updated)
    append_harvest_event(run_dir, "apply_policy", {"count": len(updated)})
    from collections import Counter

    counts = Counter(sprite.final_item.status for sprite in updated)
    for status in ("accepted", "quarantine", "needs_fix", "rejected"):
        print(f"{status}: {counts.get(status, 0)}")


def _run_dataset_qa(parsed: argparse.Namespace) -> None:
    from spritelab.dataset_maker.qa import build_contact_sheet, qa_dataset, write_reports

    dataset_dir = Path(parsed.dataset)
    result = qa_dataset(
        dataset_dir,
        sample_limit=parsed.sample_limit,
        review_queue=parsed.review_queue,
        strict=bool(parsed.strict),
        require_semantic_v3=bool(parsed.require_semantic_v3),
    )

    out_json = parsed.out_json or (dataset_dir / "dataset_qa_report.json")
    out_md = parsed.out_md or (dataset_dir / "dataset_qa_report.md")
    write_reports(result, out_json=out_json, out_md=out_md)

    contact_sheet_path: Path | None = None
    if not parsed.no_contact_sheet:
        target = parsed.sample_contact_sheet or (dataset_dir / "dataset_qa_contact_sheet.png")
        try:
            contact_sheet_path = build_contact_sheet(dataset_dir, target, sample_limit=parsed.sample_limit)
        except Exception as exc:  # pragma: no cover - contact sheet is best-effort
            print(f"Warning: contact sheet failed: {exc}")

    print(f"Dataset: {dataset_dir}")
    print(f"Records: {result.total_records}")
    print(f"Images: {result.total_images}")
    print(
        "Splits: "
        + " ".join(f"{split}={result.splits.get(split, 0)}" for split in ("train", "val", "test"))
    )
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    if contact_sheet_path is not None:
        print(f"Wrote: {contact_sheet_path}")

    if result.errors:
        for error in result.errors[:20]:
            print(f"  error: {error}")
        raise SystemExit(1)
    if parsed.fail_on_warning and result.warnings:
        raise SystemExit(1)


def _run_build_training_manifest(parsed: argparse.Namespace) -> None:
    from spritelab.dataset_maker.training_manifest import (
        build_training_manifest,
        format_training_manifest_report,
        summarize_training_manifest,
        write_per_split_manifests,
        write_training_manifest,
        write_training_manifest_reports,
    )

    dataset_dir = Path(parsed.dataset)
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {dataset_dir}")
    out_path = parsed.out or (dataset_dir / "training_manifest.jsonl")

    result = build_training_manifest(
        dataset_dir,
        variants_per_sprite=parsed.variants_per_sprite,
        caption_policy=parsed.caption_policy,
        seed=parsed.seed,
    )
    write_training_manifest(out_path, result.rows)
    if parsed.per_split:
        write_per_split_manifests(out_path, result.rows)

    summary = summarize_training_manifest(result)
    write_training_manifest_reports(
        summary,
        out_json=dataset_dir / "training_manifest_report.json",
        out_md=dataset_dir / "training_manifest_report.md",
    )
    print(format_training_manifest_report(summary), end="")
    print(f"Wrote: {out_path}")


def _run_training_manifest_qa(parsed: argparse.Namespace) -> None:
    from spritelab.dataset_maker.training_manifest_qa import (
        qa_training_manifest,
        write_training_manifest_qa_reports,
    )

    dataset_dir = Path(parsed.dataset)
    manifest_path = parsed.manifest or (dataset_dir / "training_manifest.jsonl")
    result = qa_training_manifest(
        dataset_dir, manifest_path, allow_duplicate_captions=bool(parsed.allow_duplicate_captions)
    )
    out_json = parsed.out_json or (dataset_dir / "training_manifest_qa_report.json")
    out_md = parsed.out_md or (dataset_dir / "training_manifest_qa_report.md")
    write_training_manifest_qa_reports(result, out_json=out_json, out_md=out_md)

    variant_values = [count for sid, count in result.variants_per_sprite.items() if sid]
    low = min(variant_values) if variant_values else 0
    high = max(variant_values) if variant_values else 0
    avg = (sum(variant_values) / len(variant_values)) if variant_values else 0.0
    print(f"Training manifest: {manifest_path}")
    print(f"Rows: {result.total_rows}")
    print(f"Unique sprites: {result.unique_sprites}")
    print(f"Variants per sprite: min={low} max={high} avg={avg:.1f}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    if result.errors:
        for error in result.errors[:20]:
            print(f"  error: {error}")
        raise SystemExit(1)
    if parsed.fail_on_warning and result.warnings:
        raise SystemExit(1)


def _run_training_manifest_report(parsed: argparse.Namespace) -> None:
    from spritelab.dataset_maker.training_manifest import (
        TrainingManifestResult,
        _read_jsonl,
        format_training_manifest_report,
        summarize_training_manifest,
        write_training_manifest_reports,
    )

    manifest_path = Path(parsed.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"training manifest not found: {manifest_path}")
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise SystemExit("training manifest is empty")

    from collections import Counter

    split_rows = Counter(str(row.get("split", "")) for row in rows)
    policies = {str(row.get("audit", {}).get("caption_policy", "")) for row in rows if isinstance(row.get("audit"), dict)}
    seeds = {int(row.get("audit", {}).get("seed", 0)) for row in rows if isinstance(row.get("audit"), dict)}
    variants = Counter(str(row.get("sprite_id", "")) for row in rows)
    result = TrainingManifestResult(
        dataset_dir=Path(str(rows[0].get("source", {}).get("dataset_dir", ""))),
        rows=rows,
        caption_policy=next(iter(sorted(policies)), "mixed") if policies else "mixed",
        variants_per_sprite=max(variants.values()) if variants else 0,
        seed=next(iter(sorted(seeds)), 0) if seeds else 0,
        source_records=len(variants),
        unique_sprites=len([sid for sid in variants if sid]),
        split_rows={split: split_rows.get(split, 0) for split in ("train", "val", "test")},
        warnings=[],
    )
    summary = summarize_training_manifest(result)
    if parsed.out_json is not None or parsed.out_md is not None:
        write_training_manifest_reports(
            summary,
            out_json=parsed.out_json or manifest_path.with_name("training_manifest_report.json"),
            out_md=parsed.out_md or manifest_path.with_name("training_manifest_report.md"),
        )
    print(format_training_manifest_report(summary), end="")


def _run_build_eval_prompts(parsed: argparse.Namespace) -> None:
    from spritelab.dataset_maker.eval_prompts import (
        build_eval_prompts,
        format_eval_prompts_report,
        summarize_eval_prompts,
        write_eval_prompts,
        write_eval_prompts_reports,
    )

    dataset_dir = Path(parsed.dataset)
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {dataset_dir}")
    out_path = parsed.out or (dataset_dir / "eval_prompts.jsonl")

    result = build_eval_prompts(
        dataset_dir,
        seed=parsed.seed,
        seen_object_count=parsed.seen_object_count,
        unseen_composition_count=parsed.unseen_composition_count,
    )
    write_eval_prompts(out_path, result.prompts)
    summary = summarize_eval_prompts(result)
    write_eval_prompts_reports(
        summary,
        out_json=dataset_dir / "eval_prompts_report.json",
        out_md=dataset_dir / "eval_prompts_report.md",
    )
    print(format_eval_prompts_report(summary), end="")
    print(f"Wrote: {out_path}")


def _run_dataset_readiness(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.dataset_readiness import (
        format_readiness_report,
        scan_readiness,
        write_readiness_reports,
    )

    report = scan_readiness(
        parsed.runs_root,
        parsed.datasets_root,
        review_rate_ceiling=parsed.review_rate_ceiling,
    )
    write_readiness_reports(report, out_md=parsed.out, out_json=parsed.out_json)
    print(format_readiness_report(report), end="")
    ready = [pack.run_name for pack in report.packs if pack.recommended_action == "ready_for_merge"]
    print(f"\nPacks scanned: {len(report.packs)}")
    print(f"Ready for merge: {len(ready)}")
    if parsed.out is not None:
        print(f"Wrote: {parsed.out}")
    if parsed.out_json is not None:
        print(f"Wrote: {parsed.out_json}")


def _run_acceptance_gap_report(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.acceptance_gap_report import (
        build_acceptance_gap_report,
        format_acceptance_gap_report,
        write_acceptance_gap_reports,
    )

    report = build_acceptance_gap_report(parsed.runs_root, parsed.datasets_root)
    write_acceptance_gap_reports(report, out_md=parsed.out, out_json=parsed.out_json)
    print(format_acceptance_gap_report(report), end="")
    if parsed.out is not None:
        print(f"Wrote: {parsed.out}")
    if parsed.out_json is not None:
        print(f"Wrote: {parsed.out_json}")


def _run_pack_drilldown(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.pack_drilldown import (
        build_pack_drilldown,
        format_pack_drilldown,
        write_pack_drilldown_reports,
    )

    report = build_pack_drilldown(parsed.run, prediction_file=parsed.prediction_file)
    write_pack_drilldown_reports(report, out_md=parsed.out, out_json=parsed.out_json)
    print(format_pack_drilldown(report), end="")
    if parsed.out is not None:
        print(f"Wrote: {parsed.out}")
    if parsed.out_json is not None:
        print(f"Wrote: {parsed.out_json}")


def _run_import_diagnostics(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.import_diagnostics import (
        build_import_diagnostics,
        format_import_diagnostics,
        write_import_diagnostics_reports,
    )

    report = build_import_diagnostics(parsed.run)
    write_import_diagnostics_reports(report, out_md=parsed.out, out_json=parsed.out_json)
    print(format_import_diagnostics(report), end="")
    if parsed.out is not None:
        print(f"Wrote: {parsed.out}")
    if parsed.out_json is not None:
        print(f"Wrote: {parsed.out_json}")


def _run_build_semantic_dataset(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.build_semantic_dataset import (
        BuildError,
        build_semantic_dataset,
        format_build_report,
    )

    try:
        report = build_semantic_dataset(
            parsed.run,
            dataset_name=parsed.dataset_name,
            output_root=parsed.output_root,
            prediction_file=parsed.prediction_file,
            max_palette_slots=parsed.max_palette_slots,
            accept_auto_only=bool(parsed.accept_auto_only),
            caption_policy=parsed.caption_policy,
            variants_per_sprite=parsed.variants_per_sprite,
            seed=parsed.seed,
            max_captions=parsed.max_captions,
            overwrite=bool(parsed.overwrite),
        )
    except BuildError as exc:
        print(f"Build failed: {exc}")
        raise SystemExit(1)

    print(format_build_report(report), end="")
    print(f"\nOutput: {report.output_dir}")
    if not report.ok:
        raise SystemExit(1)


def _run_merge_datasets(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.merge_datasets import MergeError, format_merge_report, merge_datasets

    try:
        result = merge_datasets(
            parsed.datasets,
            parsed.out,
            seed=parsed.seed,
            split_policy=parsed.split_policy,
            max_palette_slots=parsed.max_palette_slots,
            overwrite=bool(parsed.overwrite),
        )
    except MergeError as exc:
        print(f"Merge failed: {exc}")
        raise SystemExit(1)

    print(format_merge_report(result), end="")
    print(f"\nOutput: {result.output_dir}")
    print(f"Total records: {result.total_records}")
    print(f"Splits: " + " ".join(f"{s}={result.split_counts.get(s, 0)}" for s in ("train", "val", "test")))
    if result.errors:
        raise SystemExit(1)


def _run_build_multisource(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.build_multisource import (
        BuildMultisourceError,
        build_multisource_dataset,
        format_build_multisource_report,
    )

    try:
        report = build_multisource_dataset(
            parsed.datasets_root,
            parsed.out,
            seed=parsed.seed,
            split_policy=parsed.split_policy,
            caption_policy=parsed.caption_policy,
            variants_per_sprite=parsed.variants_per_sprite,
            max_palette_slots=parsed.max_palette_slots,
            only_atomic_ready=bool(parsed.only_atomic_ready),
            overwrite=bool(parsed.overwrite),
        )
    except BuildMultisourceError as exc:
        print(f"Build multisource failed: {exc}")
        raise SystemExit(1)

    print(format_build_multisource_report(report), end="")
    print(f"\nOutput: {report.output_dir}")
    print(f"Total records: {report.total_records}")
    if not report.ok:
        raise SystemExit(1)


def _run_export(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event
    from spritelab.harvest.pipeline import export_harvested_dataset

    run_dir = Path(parsed.run)
    _, harvested = _rehydrate_run(run_dir)
    try:
        result = export_harvested_dataset(
            harvested,
            dataset_name=parsed.dataset_name,
            output_root=parsed.output_root,
            max_palette_slots=parsed.max_palette_slots,
            train_fraction=parsed.train,
            val_fraction=parsed.val,
            test_fraction=parsed.test,
            seed=parsed.seed,
            overwrite=parsed.overwrite,
            allow_unknown_license=parsed.allow_unknown_license,
        )
    except ValueError as exc:
        print(f"Export blocked: {exc}")
        raise SystemExit(1)
    append_harvest_event(run_dir, "export", {"dataset": str(result.output_dir)})
    print(f"Output: {result.output_dir}")
    print(f"Train: {result.train_count}")
    print(f"Val: {result.val_count}")
    print(f"Test: {result.test_count}")
    for warning in result.warnings:
        print(f"Warning: {warning}")


def _run_gui(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.gui import launch_harvest_gui

    try:
        launch_harvest_gui(
            output_root=parsed.output_root,
            run_root=parsed.run_root,
            host=parsed.host,
            port=parsed.port,
        )
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)


def _rehydrate_run(run_dir: Path):
    """Rebuild HarvestedSprite objects from a run's JSONL state.

    Re-imports each final PNG (deterministic) and restores stored metadata,
    including any status set by earlier policy/GUI passes.
    """

    from spritelab.dataset_maker.importer import ImportOptions, import_png_as_dataset_item
    from spritelab.dataset_maker.model import DatasetMakerItem
    from spritelab.harvest.catalog import load_harvest_run
    from spritelab.harvest.pipeline import HarvestedSprite

    run = load_harvest_run(run_dir)
    candidates_by_id = {candidate.candidate_id: candidate for candidate in run["candidates"]}
    sources_by_id = {source.source_id: source for source in run["sources"]}

    harvested: list[HarvestedSprite] = []
    for record in [*run["imported"], *run["rejected"]]:
        source = sources_by_id.get(record["source_id"])
        candidate = candidates_by_id.get(record["candidate_id"])
        if source is None or candidate is None:
            continue
        imported = import_png_as_dataset_item(
            record["final_png_path"],
            options=ImportOptions(),
        )
        auto_metadata = dict(record.get("auto_metadata", {}))
        item = DatasetMakerItem(
            sprite_id=record["sprite_id"],
            source_path=Path(record["final_png_path"]),
            status=record["status"],
            category=record["category"],
            tags=tuple(record.get("tags", ())),
            notes=record.get("notes", ""),
            source_name=record.get("source_name", ""),
            license=record.get("license", "unknown"),
            author=record.get("author", ""),
            palette_size=record.get("palette_size"),
            has_role_map=bool(record.get("has_role_map", False)),
        )
        harvested.append(
            HarvestedSprite(
                source=source,
                candidate=candidate,
                imported=replace(imported, item=item, auto_metadata=auto_metadata),
                auto_metadata=auto_metadata,
                final_item=item,
            )
        )
    return list(sources_by_id.values()), harvested


def _unique_candidates(harvested):
    seen: set[str] = set()
    result = []
    for sprite in harvested:
        if sprite.candidate.candidate_id in seen:
            continue
        seen.add(sprite.candidate.candidate_id)
        result.append(sprite.candidate)
    return result


def _quality_counts_from_harvested(harvested) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for sprite in harvested:
        quality = sprite.auto_metadata.get("prefill_quality")
        if isinstance(quality, dict):
            counts[str(quality.get("bucket") or "unknown")] += 1
            for flag in quality.get("flags") or ():
                counts[str(flag)] += 1
    return dict(counts)


def _prefill_propagation_metadata(sprite) -> dict[str, object]:
    metadata: dict[str, object] = {}
    auto_metadata = sprite.auto_metadata
    if "prefill_propagated_from" in auto_metadata:
        metadata["prefill_propagated_from"] = auto_metadata["prefill_propagated_from"]
    if auto_metadata.get("prefill_propagated_exact_dup"):
        metadata["prefill_propagated_exact_dup"] = True
    if auto_metadata.get("prefill_propagated_near_dup"):
        metadata["prefill_propagated_near_dup"] = True
    return metadata


def _prefill_propagation_counts(harvested) -> dict[str, int]:
    exact = sum(1 for sprite in harvested if sprite.auto_metadata.get("prefill_propagated_exact_dup"))
    near = sum(1 for sprite in harvested if sprite.auto_metadata.get("prefill_propagated_near_dup"))
    return {
        "propagated_exact_duplicates": exact,
        "propagated_near_duplicates": near,
    }


def _parse_include_statuses(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values: list[str] = []
    for value in values or ("accepted",):
        raw_values.extend(str(value).split(","))
    statuses = tuple(
        status.strip().lower()
        for status in raw_values
        if status.strip()
    )
    return statuses or ("accepted",)


def _parse_runs_arg(value: str) -> tuple[Path, ...]:
    runs = tuple(Path(part.strip()) for part in str(value).split(",") if part.strip())
    if not runs:
        raise SystemExit("--runs must include at least one run directory")
    return runs

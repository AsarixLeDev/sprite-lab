"""Dataset commands: apply-policy, dataset-qa, build-training-manifest, training-manifest-qa, training-manifest-report, build-eval-prompts, dataset-readiness, acceptance-gap-report, pack-drilldown, build-semantic-dataset, merge-datasets, build-multisource, export."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.harvest.cli._args import _rehydrate_run


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_apply_policy(subparsers)
    _register_dataset_qa(subparsers)
    _register_build_training_manifest(subparsers)
    _register_training_manifest_qa(subparsers)
    _register_training_manifest_report(subparsers)
    _register_build_eval_prompts(subparsers)
    _register_dataset_readiness(subparsers)
    _register_acceptance_gap_report(subparsers)
    _register_pack_drilldown(subparsers)
    _register_build_semantic_dataset(subparsers)
    _register_merge_datasets(subparsers)
    _register_build_multisource(subparsers)
    _register_export(subparsers)


def _register_apply_policy(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("apply-policy", help="Apply a bulk accept/quarantine/reject policy.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--auto-accept-valid-cc0", action="store_true")
    p.add_argument("--auto-accept-own-work", action="store_true")
    p.add_argument("--auto-accept-allowlisted", action="store_true")
    p.add_argument("--quarantine-unknown-license", action="store_true")
    p.add_argument("--quarantine-low-qwen-confidence", action="store_true")
    p.add_argument("--qwen-confidence-threshold", type=float, default=0.3)
    p.add_argument("--reject-invalid", action="store_true")
    p.set_defaults(func=_run_apply_policy)


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


def _register_dataset_qa(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("dataset-qa", help="Validate an exported dataset directory (QA gate).")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--review-queue", type=Path)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    p.add_argument("--sample-contact-sheet", type=Path)
    p.add_argument("--no-contact-sheet", action="store_true", help="Skip contact-sheet generation.")
    p.add_argument("--sample-limit", type=int, default=64)
    p.add_argument("--strict", action="store_true", help="Escalate soft raster/tag warnings to errors.")
    p.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero if any warnings exist.")
    p.add_argument("--require-semantic-v3", action="store_true", help="Fail records that lack semantic_v3 metadata.")
    p.set_defaults(func=_run_dataset_qa)


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
        except Exception as exc:
            print(f"Warning: contact sheet failed: {exc}")

    print(f"Dataset: {dataset_dir}")
    print(f"Records: {result.total_records}")
    print(f"Images: {result.total_images}")
    print("Splits: " + " ".join(f"{split}={result.splits.get(split, 0)}" for split in ("train", "val", "test")))
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


def _register_build_training_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build-training-manifest", help="Expand a semantic-v3 dataset into training conditioning rows."
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Defaults to <dataset>/training_manifest.jsonl.")
    p.add_argument(
        "--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"]
    )
    p.add_argument("--variants-per-sprite", type=int, default=8)
    p.add_argument("--seed", type=int, default=4962026)
    p.add_argument("--per-split", action="store_true", help="Also write training_manifest_{split}.jsonl.")
    p.set_defaults(func=_run_build_training_manifest)


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


def _register_training_manifest_qa(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "training-manifest-qa", help="Validate a generated training manifest against its source dataset."
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--manifest", type=Path, help="Defaults to <dataset>/training_manifest.jsonl.")
    p.add_argument("--allow-duplicate-captions", action="store_true")
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    p.add_argument("--fail-on-warning", action="store_true")
    p.set_defaults(func=_run_training_manifest_qa)


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


def _register_training_manifest_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("training-manifest-report", help="Summarize an existing training manifest.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    p.set_defaults(func=_run_training_manifest_report)


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
    policies = {
        str(row.get("audit", {}).get("caption_policy", "")) for row in rows if isinstance(row.get("audit"), dict)
    }
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


def _register_build_eval_prompts(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build-eval-prompts", help="Generate a fixed evaluation prompt set from a semantic-v3 dataset."
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Defaults to <dataset>/eval_prompts.jsonl.")
    p.add_argument("--seed", type=int, default=4962026)
    p.add_argument("--seen-object-count", type=int, default=40)
    p.add_argument("--unseen-composition-count", type=int, default=40)
    p.set_defaults(func=_run_build_eval_prompts)


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


def _register_dataset_readiness(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("dataset-readiness", help="Scan harvest runs and datasets and report merge readiness.")
    p.add_argument("--runs-root", default="harvest_runs", type=Path)
    p.add_argument("--datasets-root", default="datasets", type=Path)
    p.add_argument("--out", type=Path, help="Markdown report path.")
    p.add_argument("--out-json", type=Path, help="JSON report path.")
    p.add_argument("--review-rate-ceiling", type=float, default=0.25)
    p.set_defaults(func=_run_dataset_readiness)


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


def _register_acceptance_gap_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("acceptance-gap-report", help="Rank packs by raw-auto/apply/export acceptance gaps.")
    p.add_argument("--runs-root", default="harvest_runs", type=Path)
    p.add_argument("--datasets-root", default="datasets", type=Path)
    p.add_argument("--out", type=Path, help="Markdown report path.")
    p.add_argument("--out-json", type=Path, help="JSON report path.")
    p.set_defaults(func=_run_acceptance_gap_report)


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


def _register_pack_drilldown(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("pack-drilldown", help="Write a read-only drilldown report for one harvest run.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--out", type=Path, help="Markdown report path.")
    p.add_argument("--out-json", type=Path, help="JSON report path.")
    p.set_defaults(func=_run_pack_drilldown)


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


def _register_build_semantic_dataset(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build-semantic-dataset",
        help="Safely build one exported+QA'd semantic-v3 dataset from a harvest run.",
    )
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--output-root", default="datasets", type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--max-palette-slots", type=int, default=32)
    p.add_argument("--max-captions", type=int, default=8)
    p.add_argument(
        "--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"]
    )
    p.add_argument("--variants-per-sprite", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260706)
    accept_auto = p.add_mutually_exclusive_group()
    accept_auto.add_argument("--accept-auto-only", action="store_true", dest="accept_auto_only", default=True)
    accept_auto.add_argument("--no-accept-auto-only", action="store_false", dest="accept_auto_only")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_run_build_semantic_dataset)


def _run_build_semantic_dataset(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.build_semantic_dataset import (
        BuildError,
        build_semantic_dataset,
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
        raise SystemExit(1) from exc
    print(f"\nOutput: {report.output_dir}")
    if not report.ok:
        raise SystemExit(1)


def _register_merge_datasets(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "merge-datasets", help="Merge exported semantic-v3 datasets into one multi-source dataset."
    )
    p.add_argument("--datasets", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260706)
    p.add_argument("--split-policy", default="preserve", choices=["preserve", "reshuffle"])
    p.add_argument("--max-palette-slots", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_run_merge_datasets)


def _run_merge_datasets(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.merge_datasets import MergeError, merge_datasets

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
        raise SystemExit(1) from exc
    print(f"\nOutput: {result.output_dir}")
    print(f"Total records: {result.total_records}")
    print("Splits: " + " ".join(f"{s}={result.split_counts.get(s, 0)}" for s in ("train", "val", "test")))
    if result.errors:
        raise SystemExit(1)


def _register_build_multisource(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("build-multisource", help="Build a safe multisource dataset from atomic ready datasets.")
    p.add_argument("--datasets-root", default="datasets", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260706)
    p.add_argument("--split-policy", default="preserve", choices=["preserve", "reshuffle"])
    p.add_argument(
        "--caption-policy", default="mixed", choices=["object_only", "style_aware", "attribute", "minimal", "mixed"]
    )
    p.add_argument("--variants-per-sprite", type=int, default=8)
    p.add_argument("--max-palette-slots", type=int, default=32)
    p.add_argument("--only-atomic-ready", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_run_build_multisource)


def _run_build_multisource(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.build_multisource import (
        BuildMultisourceError,
        build_multisource_dataset,
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
        raise SystemExit(1) from exc
    print(f"\nOutput: {report.output_dir}")
    print(f"Total records: {report.total_records}")
    if not report.ok:
        raise SystemExit(1)


def _register_export(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("export", help="Export accepted sprites to a dataset.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--output-root", default="datasets", type=Path)
    p.add_argument("--max-palette-slots", type=int, default=32)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-unknown-license", action="store_true")
    p.set_defaults(func=_run_export)


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
        raise SystemExit(1) from exc
    append_harvest_event(run_dir, "export", {"dataset": str(result.output_dir)})
    print(f"Output: {result.output_dir}")
    print(f"Train: {result.train_count}")
    print(f"Val: {result.val_count}")
    print(f"Test: {result.test_count}")
    for warning in result.warnings:
        print(f"Warning: {warning}")

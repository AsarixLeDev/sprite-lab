"""Auto-Labeling v3 CLI commands.

Commands:
  label-v3           — Run the v3 labeling pipeline (defaults to dry-run)
  label-v3-eval      — Evaluate v3 records against golden labels
  label-v3-report    — Generate per-pack and global v3 reports
  label-v3-promote   — Check promotion gates and recommend rollout stage
  calibrate-v3       — Build or rebuild calibration artifacts from corrections
  calibrate-v3-all   — Build calibration artifacts for all fields from one correction file
  assisted-v3        — Launch the v3 prefilled correction Gradio GUI
  freeze-v3-suite    — Create disjoint frozen evaluation suites from harvest runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_label_v3(subparsers)
    _register_label_v3_shard(subparsers)
    _register_label_v3_merge(subparsers)
    _register_label_v3_retry(subparsers)
    _register_label_v3_eval(subparsers)
    _register_label_v3_report(subparsers)
    _register_label_v3_promote(subparsers)
    _register_label_v3_apply(subparsers)
    _register_calibrate_v3(subparsers)
    _register_calibrate_v3_all(subparsers)
    _register_assisted_v3(subparsers)
    _register_freeze_v3_suite(subparsers)


def _register_label_v3_apply(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "label-v3-apply",
        help="Apply v3 accepted fields to a NEW output root (dry-run by default).",
    )
    p.add_argument("--v3-records", required=True, type=Path, help="Path to canonical v3_records.jsonl")
    p.add_argument("--output-root", required=True, type=Path, help="NEW output directory (never a historical run)")
    p.add_argument("--apply", action="store_true", default=False, help="Actually write (default is dry-run)")
    p.add_argument("--no-partial", action="store_true", default=False, help="Exclude partial_accept records")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite an existing v3 sidecar")
    p.set_defaults(func=_run_label_v3_apply)


def _run_label_v3_apply(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v3.apply_v3 import apply_v3_records, format_migration_md

    report = apply_v3_records(
        parsed.v3_records,
        parsed.output_root,
        dry_run=not parsed.apply,
        include_partial=not parsed.no_partial,
        force=parsed.force,
    )
    print(format_migration_md(report))
    if report.dry_run:
        print("(DRY RUN — pass --apply to write the v3 sidecar + migration report.)")


def _load_calibration(path) -> Any:
    if not path:
        return None
    from spritelab.harvest.label_v3.calibration import CalibrationArtifact

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationArtifact.from_json_dict(data)


def _pipeline_config(parsed: argparse.Namespace):
    from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy, V3PipelineConfig

    policy = V3LabelingPolicy(
        precision_target_category=getattr(parsed, "precision_target_category", 0.99),
        precision_target_canonical_object=getattr(parsed, "precision_target_object", 0.99),
    )
    return V3PipelineConfig(policy=policy)


# ---------------------------------------------------------------------------
# label-v3-shard / merge / retry  (Phase 6: scale, resume, failures)
# ---------------------------------------------------------------------------


def _register_label_v3_shard(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-shard", help="Run one deterministic, resumable shard of the v3 pipeline.")
    p.add_argument("--run", required=True, type=Path, help="Harvest run directory")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--shard", default="0/1", help="Shard as I/N, e.g. 0/8")
    p.add_argument("--no-resume", action="store_true", default=False, help="Recompute this shard from scratch")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--calibration-artifact", type=Path, default=None)
    p.add_argument("--use-vlm", action="store_true", default=False)
    p.add_argument("--precision-target-category", type=float, default=0.99)
    p.add_argument("--precision-target-object", type=float, default=0.99)
    p.set_defaults(func=_run_label_v3_shard)


def _parse_shard(spec: str) -> tuple[int, int]:
    try:
        i_str, n_str = str(spec).split("/", 1)
        return int(i_str), int(n_str)
    except Exception as exc:
        raise SystemExit(f"invalid --shard {spec!r}; expected I/N") from exc


def _run_label_v3_shard(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v3.pipeline_stages_v3 import run_v3_shard

    shard_index, shard_count = _parse_shard(parsed.shard)
    result = run_v3_shard(
        parsed.run,
        parsed.output_root,
        _pipeline_config(parsed),
        calibration=_load_calibration(parsed.calibration_artifact),
        use_vlm=parsed.use_vlm,
        shard_index=shard_index,
        shard_count=shard_count,
        resume=not parsed.no_resume,
        max_records=parsed.max_records,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _register_label_v3_merge(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-merge", help="Deterministically merge v3 shards into a canonical file.")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--out-file", type=Path, default=None)
    p.add_argument("--allow-duplicates", action="store_true", default=False)
    p.set_defaults(func=_run_label_v3_merge)


def _run_label_v3_merge(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v3.pipeline_stages_v3 import merge_v3_shards

    info = merge_v3_shards(parsed.output_root, out_path=parsed.out_file, allow_duplicates=parsed.allow_duplicates)
    print(json.dumps(info, indent=2, sort_keys=True))


def _register_label_v3_retry(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-retry", help="Retry the retryable failures for one v3 shard.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--shard", default="0/1", help="Shard as I/N")
    p.add_argument("--calibration-artifact", type=Path, default=None)
    p.add_argument("--use-vlm", action="store_true", default=False)
    p.add_argument("--precision-target-category", type=float, default=0.99)
    p.add_argument("--precision-target-object", type=float, default=0.99)
    p.set_defaults(func=_run_label_v3_retry)


def _run_label_v3_retry(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v3.pipeline_stages_v3 import retry_v3_failures

    shard_index, shard_count = _parse_shard(parsed.shard)
    info = retry_v3_failures(
        parsed.run,
        parsed.output_root,
        _pipeline_config(parsed),
        calibration=_load_calibration(parsed.calibration_artifact),
        use_vlm=parsed.use_vlm,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# label-v3
# ---------------------------------------------------------------------------


def _register_label_v3(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3", help="Run Auto-Labeling v3 pipeline (dry-run by default).")
    p.add_argument("--run", required=True, type=Path, help="Harvest run directory")
    p.add_argument("--output-root", default=None, type=Path, help="Output directory (defaults to <run>/v3_output)")
    p.add_argument("--max-records", type=int, default=None, help="Cap the number of records processed")
    p.add_argument("--use-vlm", action="store_true", default=False, help="Enable VLM stages (disabled by default)")
    p.add_argument("--vlm-model", default=os.environ.get("SPRITELAB_V3_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    p.add_argument("--vlm-backend", default=os.environ.get("SPRITELAB_V3_VLM_BACKEND", "openai_compatible"))
    p.add_argument("--vlm-base-url", default=os.environ.get("SPRITELAB_V3_VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--vlm-api-key", default=os.environ.get("SPRITELAB_V3_VLM_API_KEY", "not-needed"))
    p.add_argument("--vlm-structured-output", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--vlm-prompt-version", default="vlm_prefill_v3_2")
    p.add_argument(
        "--vlm-cascade-profile",
        default="fast",
        choices=["fast", "balanced", "full"],
        help="fast=2 calls/sprite, balanced=4, full=5 (default: fast)",
    )
    p.add_argument(
        "--vlm-thinking",
        action="store_true",
        default=False,
        help="Allow model reasoning tokens (disabled by default for low-latency structured prefill)",
    )
    p.add_argument("--vlm-timeout-seconds", type=float, default=60.0)
    p.add_argument("--vlm-retries", type=int, default=1)
    p.add_argument("--vlm-concurrency", type=int, default=1)
    p.add_argument("--vlm-retry-backoff-seconds", type=float, default=1.0)
    p.add_argument("--vlm-cache-dir", type=Path, default=None)
    p.add_argument("--vlm-failure-diagnostics-dir", type=Path, default=None)
    p.add_argument("--no-vlm-failure-diagnostics", action="store_true", default=False)
    p.add_argument("--enable-text-enrichment", action="store_true", default=False)
    p.add_argument("--text-enrichment-backend", default=os.environ.get("SPRITELAB_V3_TEXT_BACKEND", "none"))
    p.add_argument("--text-enrichment-model", default=os.environ.get("SPRITELAB_V3_TEXT_MODEL", ""))
    p.add_argument("--text-enrichment-base-url", default=os.environ.get("SPRITELAB_V3_TEXT_BASE_URL", ""))
    p.add_argument("--text-enrichment-api-key", default=os.environ.get("SPRITELAB_V3_TEXT_API_KEY", ""))
    p.add_argument("--text-enrichment-timeout-seconds", type=float, default=60.0)
    p.add_argument("--text-enrichment-retries", type=int, default=1)
    p.add_argument(
        "--vlm-image-view",
        default="magenta_matte",
        choices=["magenta_matte", "native_alpha", "checkerboard_experimental"],
    )
    p.add_argument(
        "--no-dry-run", action="store_true", default=False, help="Actually write outputs (default is dry-run)"
    )
    p.add_argument("--precision-target-category", type=float, default=0.99)
    p.add_argument("--precision-target-object", type=float, default=0.99)
    p.add_argument("--precision-target-color", type=float, default=0.95)
    p.add_argument("--precision-target-material", type=float, default=0.95)
    p.add_argument("--precision-target-shape", type=float, default=0.90)
    p.add_argument("--calibration-artifact", type=Path, default=None, help="Path to calibration artifact JSON")
    p.add_argument("--policy-version", default="v3.1.0")
    p.set_defaults(func=_run_label_v3)


def _run_label_v3(parsed: argparse.Namespace) -> None:
    # v3 inference is operationally expensive; show concise lifecycle logs by
    # default so a user never has to guess whether a remote/local backend ran.
    import logging

    logging.getLogger().setLevel(logging.INFO)
    from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy, V3PipelineConfig
    from spritelab.harvest.label_v3.pipeline_v3 import format_pipeline_dry_run_report, run_v3_pipeline

    run_dir = Path(parsed.run)
    output_root = Path(parsed.output_root) if parsed.output_root else run_dir / "v3_output"
    dry_run = not parsed.no_dry_run

    policy = V3LabelingPolicy(
        policy_version=parsed.policy_version,
        precision_target_category=parsed.precision_target_category,
        precision_target_canonical_object=parsed.precision_target_object,
        precision_target_color=parsed.precision_target_color,
        precision_target_material=parsed.precision_target_material,
        precision_target_shape=parsed.precision_target_shape,
        shadow_mode=dry_run,
        dry_run_apply=dry_run,
    )

    config = V3PipelineConfig(
        policy=policy,
        vlm_backend=parsed.vlm_backend if parsed.use_vlm else "none",
        vlm_model=parsed.vlm_model,
        vlm_base_url=parsed.vlm_base_url,
        vlm_api_key=parsed.vlm_api_key,
        vlm_structured_output=parsed.vlm_structured_output,
        vlm_prompt_version=parsed.vlm_prompt_version,
        vlm_cascade_profile=parsed.vlm_cascade_profile,
        vlm_disable_thinking=not parsed.vlm_thinking,
        vlm_timeout_seconds=parsed.vlm_timeout_seconds,
        vlm_retries=parsed.vlm_retries,
        vlm_concurrency=parsed.vlm_concurrency,
        vlm_retry_backoff_seconds=parsed.vlm_retry_backoff_seconds,
        vlm_cache_dir=str(parsed.vlm_cache_dir) if parsed.vlm_cache_dir else "",
        vlm_failure_diagnostics_enabled=not parsed.no_vlm_failure_diagnostics,
        vlm_failure_diagnostics_dir=(
            str(parsed.vlm_failure_diagnostics_dir) if parsed.vlm_failure_diagnostics_dir else ""
        ),
        text_enrichment_enabled=parsed.enable_text_enrichment,
        text_enrichment_model=parsed.text_enrichment_model,
        text_enrichment_backend=parsed.text_enrichment_backend,
        text_enrichment_base_url=parsed.text_enrichment_base_url,
        text_enrichment_api_key=parsed.text_enrichment_api_key,
        text_enrichment_timeout_seconds=parsed.text_enrichment_timeout_seconds,
        text_enrichment_retries=parsed.text_enrichment_retries,
        vlm_image_view=parsed.vlm_image_view,
        vlm_include_filename_hint=False,
        run_dir=str(run_dir),
        max_records=parsed.max_records,
    )

    result = run_v3_pipeline(
        run_dir=run_dir,
        output_root=output_root,
        config=config,
        use_vlm=parsed.use_vlm,
        max_records=parsed.max_records,
        dry_run=dry_run,
    )

    if dry_run:
        print(format_pipeline_dry_run_report(result))
    else:
        print(json.dumps(result.summary(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# label-v3-eval
# ---------------------------------------------------------------------------


def _register_label_v3_eval(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-eval", help="Evaluate v3 records against golden labels.")
    p.add_argument("--v3-records", required=True, type=Path, help="Path to v3_records.jsonl")
    p.add_argument("--golden-labels", required=True, type=Path, help="Path to golden_labels.jsonl")
    p.add_argument("--suite-name", default="unnamed", help="Name for this evaluation suite")
    p.add_argument("--precision-target-category", type=float, default=0.99)
    p.add_argument("--precision-target-object", type=float, default=0.99)
    p.add_argument("--out-json", type=Path, help="Write results JSON to this path")
    p.add_argument("--out-md", type=Path, help="Write report markdown to this path")
    p.set_defaults(func=_run_label_v3_eval)


def _run_label_v3_eval(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.golden import load_existing_golden_labels
    from spritelab.harvest.label_v3.label_v3_eval import (
        evaluate_v3_against_golden,
        format_v3_evaluation_report,
        load_v3_records_from_run,
        promotion_recommendation,
    )

    v3_path = Path(parsed.v3_records)
    golden_path = Path(parsed.golden_labels)

    if not v3_path.is_file():
        raise SystemExit(f"v3 records file not found: {v3_path}")
    if not golden_path.is_file():
        raise SystemExit(f"golden labels file not found: {golden_path}")

    v3_records = load_v3_records_from_run(v3_path.parent, v3_file=v3_path.name)
    golden_labels = load_existing_golden_labels(golden_path)

    result = evaluate_v3_against_golden(
        golden=golden_labels,
        v3_records=v3_records,
        suite_name=parsed.suite_name,
        precision_target_category=parsed.precision_target_category,
        precision_target_canonical_object=parsed.precision_target_object,
    )

    report = format_v3_evaluation_report(result)
    recommendation = promotion_recommendation(result)

    print(report)
    print(f"\nPromotion recommendation: {recommendation}")

    if parsed.out_json:
        out_json = Path(parsed.out_json)
        if not out_json.is_absolute():
            out_json = v3_path.parent / out_json
        out_json.write_text(
            json.dumps(
                {
                    "suite_name": result.suite_name,
                    "matched": result.matched,
                    "per_field": result.per_field,
                    "promotion_gates": result.promotion_gates,
                    "recommendation": recommendation,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"JSON written: {out_json}")

    if parsed.out_md:
        out_md = Path(parsed.out_md)
        if not out_md.is_absolute():
            out_md = v3_path.parent / out_md
        out_md.write_text(report + f"\n\n## Recommendation\n\n{recommendation}\n", encoding="utf-8")
        print(f"Report written: {out_md}")


# ---------------------------------------------------------------------------
# label-v3-report
# ---------------------------------------------------------------------------


def _register_label_v3_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-report", help="Generate per-pack and global v3 labeling reports (streaming).")
    p.add_argument("--run", required=True, type=Path, help="Run directory containing v3_records.jsonl")
    p.add_argument("--v3-file", default="v3_records.jsonl")
    p.add_argument("--out-md", type=Path)
    p.add_argument("--out-json", type=Path)
    p.set_defaults(func=_run_label_v3_report)


def _run_label_v3_report(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v3.pipeline_stages_v3 import format_stream_report_md, stream_v3_report

    run_dir = Path(parsed.run)
    v3_path = run_dir / parsed.v3_file
    if not v3_path.is_file():
        raise SystemExit(f"v3 records file not found: {v3_path}")

    # Single streaming pass — bounded memory, safe for 100k+ record files.
    report = stream_v3_report(v3_path)
    report_md = format_stream_report_md(report)
    print(report_md)

    if parsed.out_md:
        out_md = Path(parsed.out_md)
        if not out_md.is_absolute():
            out_md = run_dir / out_md
        out_md.write_text(report_md, encoding="utf-8")
        print(f"Report written: {out_md}")
    if getattr(parsed, "out_json", None):
        out_json = Path(parsed.out_json)
        if not out_json.is_absolute():
            out_json = run_dir / out_json
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON written: {out_json}")


# ---------------------------------------------------------------------------
# label-v3-promote
# ---------------------------------------------------------------------------


def _register_label_v3_promote(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v3-promote", help="Check all promotion gates across evaluation suites.")
    p.add_argument("--eval-jsons", nargs="+", required=True, type=Path, help="One or more evaluation JSON result files")
    p.add_argument("--out-md", type=Path)
    p.set_defaults(func=_run_label_v3_promote)


def _run_label_v3_promote(parsed: argparse.Namespace) -> None:
    import json

    all_gates: dict[str, dict[str, bool]] = {}
    all_recommendations: list[str] = []

    for path in parsed.eval_jsons:
        p = Path(path)
        if not p.is_file():
            print(f"Warning: {p} not found, skipping")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        suite = data.get("suite_name", p.stem)
        gates = data.get("promotion_gates", {})
        all_gates[suite] = gates

        from spritelab.harvest.label_v3.label_v3_eval import V3EvalResult, promotion_recommendation

        res = V3EvalResult(suite_name=suite)
        res.matched = int(data.get("matched", 0))
        res.per_field = data.get("per_field", {})
        res.promotion_gates = gates
        all_recommendations.append(promotion_recommendation(res))

    lines = [
        "# Auto-Labeling v3 Promotion Report",
        "",
        "## Per-Suite Gates",
    ]

    all_suite_gates = set()
    for gates in all_gates.values():
        all_suite_gates.update(gates.keys())

    for suite, gates in all_gates.items():
        lines.append(f"\n### {suite}")
        for gate in sorted(all_suite_gates):
            status = gates.get(gate, False)
            icon = "PASS" if status else "FAIL"
            lines.append(f"- {icon} {gate}")

    all_pass = all(all_suite_gates.issubset(gates.keys()) and all(gates.values()) for gates in all_gates.values())

    lines.append("\n## Overall Recommendation")
    if "blocked" in all_recommendations:
        lines.append("**BLOCKED** — one or more suites do not meet core gates.")
    elif all_pass:
        lines.append("**ELIGIBLE FOR LARGE BATCH** — all gates pass on all suites.")
    elif all(all_gates):
        lines.append("**LIMITED OPT-IN** — core gates pass, some optional gates need attention.")
    else:
        lines.append("**SHADOW ONLY** — gates do not meet promotion thresholds.")

    report = "\n".join(lines) + "\n"
    print(report)

    if parsed.out_md:
        Path(parsed.out_md).write_text(report, encoding="utf-8")
        print(f"Report written: {parsed.out_md}")


# ---------------------------------------------------------------------------
# calibrate-v3
# ---------------------------------------------------------------------------


def _register_calibrate_v3(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("calibrate-v3", help="Build or rebuild calibration artifacts from corrections.")
    p.add_argument("--run", required=True, type=Path, help="Run directory containing v3_corrections.jsonl")
    p.add_argument("--correction-file", default="v3_corrections.jsonl")
    p.add_argument(
        "--field",
        default="category",
        choices=["domain", "category", "canonical_object", "color", "material", "shape", "role"],
    )
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--out", type=Path, help="Output path for calibration artifact JSON")
    p.set_defaults(func=_run_calibrate_v3)


def _run_calibrate_v3(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.label_v3.assisted_golden_v3 import load_v3_corrections
    from spritelab.harvest.label_v3.calibration import (
        CalibrationArtifact,
        CalibrationStratumData,
        build_empty_calibration_artifact,
        compute_ece,
        compute_lower_confidence_bound,
    )

    run_dir = Path(parsed.run)
    corrections_path = run_dir / parsed.correction_file
    if not corrections_path.is_file():
        raise SystemExit(f"No correction file found: {corrections_path}")

    corrections = load_v3_corrections(corrections_path)
    field_corrections = [c for c in corrections if c.field_name == parsed.field]

    if not field_corrections:
        print(f"No corrections for field '{parsed.field}'. Writing empty artifact.")
        artifact = build_empty_calibration_artifact(field=parsed.field)
        out = Path(parsed.out) if parsed.out else run_dir / f"calibration_{parsed.field}.json"
        out.write_text(
            json.dumps(artifact.to_json_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(f"Written: {out}")
        return

    correct_count = sum(1 for c in field_corrections if c.corrected_state == c.original_state)
    total = len(field_corrections)
    error_count = total - correct_count

    precision = correct_count / max(1, total)
    ci_lower = compute_lower_confidence_bound(correct_count, total)
    ci_upper = precision + (1.96 * (precision * (1 - precision) / max(1, total)) ** 0.5) if total > 0 else 1.0

    probs = [1.0 if c.corrected_state == c.original_state else 0.0 for c in field_corrections]
    labels = [1 if c.corrected_state == c.original_state else 0 for c in field_corrections]
    ece = compute_ece(probs, labels)

    sufficient = total >= parsed.min_samples

    stratum_data = CalibrationStratumData(
        field=parsed.field,
        stratum="global",
        sample_count=total,
        error_count=error_count,
        observed_precision=precision,
        calibrated_probability=precision,
        ci_lower=ci_lower,
        ci_upper=min(1.0, ci_upper),
        ece=ece,
        sufficient=sufficient,
    )

    artifact = CalibrationArtifact(
        field_name=parsed.field,
        evidence_policy="v3_deterministic+vlm_staged",
        calibration_split_identity=run_dir.name,
        strata_data=(stratum_data,),
        observed_errors={c.field_name: error_count for c in field_corrections if c.corrected_state != c.original_state},
    )

    out = Path(parsed.out) if parsed.out else run_dir / f"calibration_{parsed.field}.json"
    out.write_text(json.dumps(artifact.to_json_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"Field: {parsed.field}")
    print(f"Samples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Precision: {precision:.4f}")
    print(f"CI Lower: {ci_lower:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Sufficient: {sufficient}")
    print(f"Written: {out}")


# ---------------------------------------------------------------------------
# calibrate-v3-all
# ---------------------------------------------------------------------------


def _register_calibrate_v3_all(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("calibrate-v3-all", help="Build calibration artifacts for all supported fields.")
    p.add_argument("--run", required=True, type=Path, help="Run directory containing v3_corrections.jsonl")
    p.add_argument("--correction-file", default="v3_corrections.jsonl")
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for calibration JSONs")
    p.set_defaults(func=_run_calibrate_v3_all)


def _run_calibrate_v3_all(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.label_v3.assisted_golden_v3 import load_v3_corrections
    from spritelab.harvest.label_v3.calibration import (
        CalibrationArtifact,
        CalibrationStratumData,
        build_empty_calibration_artifact,
        compute_ece,
        compute_lower_confidence_bound,
    )

    run_dir = Path(parsed.run)
    corrections_path = run_dir / parsed.correction_file
    if not corrections_path.is_file():
        raise SystemExit(f"No correction file found: {corrections_path}")

    corrections = load_v3_corrections(corrections_path)
    if not corrections:
        raise SystemExit("No corrections found in file.")

    fields = ("domain", "category", "canonical_object", "color", "material", "shape", "role")
    out_dir = Path(parsed.out_dir) if parsed.out_dir else run_dir

    for field in fields:
        field_corrections = [c for c in corrections if c.field_name == field]
        if not field_corrections:
            artifact = build_empty_calibration_artifact(field=field)
            out_path = out_dir / f"calibration_{field}.json"
            out_path.write_text(
                json.dumps(artifact.to_json_dict(), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"  {field}: no corrections → empty artifact at {out_path}")
            continue

        correct_count = sum(1 for c in field_corrections if c.corrected_state == c.original_state)
        total = len(field_corrections)
        error_count = total - correct_count
        precision = correct_count / max(1, total)
        ci_lower = compute_lower_confidence_bound(correct_count, total)
        ci_upper = precision + (1.96 * (precision * (1 - precision) / max(1, total)) ** 0.5) if total > 0 else 1.0
        probs = [1.0 if c.corrected_state == c.original_state else 0.0 for c in field_corrections]
        labels = [1 if c.corrected_state == c.original_state else 0 for c in field_corrections]
        ece = compute_ece(probs, labels)
        sufficient = total >= parsed.min_samples

        stratum_data = CalibrationStratumData(
            field=field,
            stratum="global",
            sample_count=total,
            error_count=error_count,
            observed_precision=precision,
            calibrated_probability=precision,
            ci_lower=ci_lower,
            ci_upper=min(1.0, ci_upper),
            ece=ece,
            sufficient=sufficient,
        )

        artifact = CalibrationArtifact(
            field_name=field,
            evidence_policy="v3_deterministic+vlm_staged",
            calibration_split_identity=run_dir.name,
            strata_data=(stratum_data,),
            observed_errors={field: error_count} if error_count > 0 else {},
        )

        out_path = out_dir / f"calibration_{field}.json"
        out_path.write_text(
            json.dumps(artifact.to_json_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"  {field}: {total} corrections, precision={precision:.4f}, ECE={ece:.4f} → {out_path}")

    print("Done. All calibration artifacts written.")


# ---------------------------------------------------------------------------
# assisted-v3 (prefilled v3 correction GUI)
# ---------------------------------------------------------------------------


def _register_assisted_v3(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("assisted-v3", help="Launch the prefilled v3 correction Gradio GUI.")
    p.add_argument("--run", type=Path, default=None, help="Harvest run directory")
    p.add_argument("--scheduler-cohort", type=Path, default=None, help="Explicit scheduler cohort JSONL")
    p.add_argument("--pool", type=Path, default=None, help="Frozen pool directory for scheduler resolution")
    p.add_argument("--work-dir", type=Path, default=None, help="Writable scheduler review state directory")
    p.add_argument("--harvest-root", type=Path, default=Path("harvest_runs"))
    p.add_argument("--completed-ids", type=Path, default=None, help="Append-only scheduler completion events")
    p.add_argument("--prepare-only", action="store_true", help="Generate deterministic prefills and exit")
    p.add_argument("--v3-records", type=Path, default=None, help="Path to v3_records.jsonl (auto-detected)")
    p.add_argument("--correction-file", type=Path, default=None, help="Path to v3_corrections.jsonl (auto-detected)")
    p.add_argument("--n", type=int, default=None, help="Number of calibration candidates")
    p.add_argument("--seed", type=int, default=496, help="Deterministic sample seed")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--labeler", default="mathieu")
    p.add_argument("--mode", default="calibration", choices=["calibration", "evaluation"])
    p.add_argument("--suite", type=Path, default=None, help="Frozen suite manifest JSON (evaluation mode)")
    p.add_argument("--partition", default="", help="Suite partition to load, e.g. frozen_in_domain_test")
    p.add_argument("--calibration-run", type=Path, default=None, help="Calibration run dir for leakage check")
    p.add_argument("--golden-path", type=Path, default=None, help="golden_labels.jsonl path (evaluation mode)")
    p.set_defaults(func=_run_assisted_v3)


def _run_assisted_v3(parsed: argparse.Namespace) -> None:
    if parsed.scheduler_cohort:
        if not parsed.pool or not parsed.work_dir:
            raise SystemExit("--scheduler-cohort requires --pool and --work-dir")
        if parsed.prepare_only:
            from spritelab.harvest.label_v3.scheduler_input import prepare_scheduler_v3

            preparation = prepare_scheduler_v3(
                parsed.scheduler_cohort,
                parsed.pool,
                parsed.work_dir,
                harvest_root=parsed.harvest_root,
            )
            print(json.dumps(preparation.to_dict(), indent=2, sort_keys=True))
            return
    elif parsed.run is None:
        raise SystemExit("--run is required unless --scheduler-cohort is supplied")

    from spritelab.harvest.label_v3.assisted_v3_gui import launch_assisted_v3_gui

    try:
        launch_assisted_v3_gui(
            parsed.run or parsed.work_dir,
            v3_records_path=parsed.v3_records,
            correction_path=parsed.correction_file,
            golden_path=parsed.golden_path,
            n=parsed.n,
            seed=parsed.seed,
            host=parsed.host,
            port=parsed.port,
            labeler=parsed.labeler,
            mode=parsed.mode,
            suite_path=parsed.suite,
            partition=parsed.partition,
            calibration_run=parsed.calibration_run,
            scheduler_cohort=parsed.scheduler_cohort,
            pool_path=parsed.pool,
            work_dir=parsed.work_dir,
            harvest_root=parsed.harvest_root,
            completed_ids_path=parsed.completed_ids,
        )
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# freeze-v3-suite
# ---------------------------------------------------------------------------


def _register_freeze_v3_suite(subparsers: argparse._SubParsersAction) -> None:
    from spritelab.harvest.label_v3.freeze_v3_suites_cli import register as _freeze_register

    _freeze_register(subparsers)

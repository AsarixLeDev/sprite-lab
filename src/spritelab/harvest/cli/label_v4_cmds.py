"""CLI commands for risk-aware Labeling v4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def register(subparsers: argparse._SubParsersAction) -> None:
    smoke = subparsers.add_parser("label-v4-smoke", help="Run a no-cost mocked Labeling-v4 smoke comparison.")
    smoke.add_argument(
        "--cohort", type=Path, default=Path("out/r2_annotation_batch_0001_semantic_accept_only_25.jsonl")
    )
    smoke.add_argument(
        "--resolved",
        type=Path,
        default=Path("out/r2_assisted_v3_batch_0001/scheduler_resolved_candidates.jsonl"),
    )
    smoke.add_argument("--output-root", type=Path, default=Path("experiments/label_v4_mock_smoke"))
    smoke.add_argument("--max-records", type=int, default=3)
    _add_record_targeting_arguments(smoke)
    smoke.set_defaults(func=_run_smoke)

    cohort = subparsers.add_parser("label-v4-cohort", help="Rebuild the fixed 25-record mocked A/B/C regression.")
    cohort.add_argument(
        "--cohort", type=Path, default=Path("out/r2_annotation_batch_0001_semantic_accept_only_25.jsonl")
    )
    cohort.add_argument(
        "--resolved",
        type=Path,
        default=Path("out/r2_assisted_v3_batch_0001/scheduler_resolved_candidates.jsonl"),
    )
    cohort.add_argument("--output-root", type=Path, default=Path("experiments/label_v4_same_cohort_comparison"))
    cohort.set_defaults(func=_run_cohort)

    canary = subparsers.add_parser(
        "label-v4-canary",
        help="Run a bounded real OpenAI-compatible provider canary (network/credit use).",
    )
    canary.add_argument(
        "--cohort", type=Path, default=Path("out/r2_annotation_batch_0001_semantic_accept_only_25.jsonl")
    )
    canary.add_argument(
        "--resolved",
        type=Path,
        default=Path("out/r2_assisted_v3_batch_0001/scheduler_resolved_candidates.jsonl"),
    )
    canary.add_argument("--output-root", type=Path, required=True)
    canary.add_argument("--max-records", type=int, default=1, help="Hard-capped at 5 records.")
    canary.add_argument("--mode", choices=("adaptive", "B", "C"), default="C")
    canary.add_argument("--runpod-endpoint-id", default=os.environ.get("SPRITELAB_RUNPOD_ENDPOINT_ID", ""))
    canary.add_argument("--base-url", default="", help="Override with a local OpenAI-compatible URL if desired.")
    canary.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    canary.add_argument("--vlm-model", required=True)
    canary.add_argument("--text-model", default="")
    canary.add_argument("--verifier-model", default="")
    canary.add_argument("--timeout-seconds", type=float, default=90.0)
    canary.add_argument("--input-cost-per-million", type=float, default=None)
    canary.add_argument("--output-cost-per-million", type=float, default=None)
    _add_record_targeting_arguments(canary)
    canary.add_argument(
        "--shared-cache-root",
        type=Path,
        default=None,
        help="Append-only cache shared by matching B/C comparison runs.",
    )
    canary.add_argument(
        "--require-shared-bc-cache",
        action="store_true",
        default=False,
        help="In mode C, fail closed unless the matching B proposal and reconciliation are in the shared cache.",
    )
    canary.set_defaults(func=_run_canary)

    assisted = subparsers.add_parser("assisted-v4", help="Launch the compact append-only Labeling-v4 review GUI.")
    assisted.add_argument("--records", type=Path, required=True)
    assisted.add_argument("--corrections", type=Path, required=True)
    assisted.add_argument(
        "--mode",
        choices=("quality-only", "semantic-assisted", "manual-truth-diagnostic"),
        default="quality-only",
    )
    assisted.add_argument("--host", default="127.0.0.1")
    assisted.add_argument("--port", type=int, default=7862)
    assisted.add_argument("--share", action="store_true", default=False)
    assisted.add_argument(
        "--diagnostic-allow-selection",
        action="store_true",
        default=False,
        help="Permit raw selection rows only for contract diagnostics; they remain non-reviewable.",
    )
    assisted.set_defaults(func=_run_assisted)

    prepare = subparsers.add_parser(
        "label-v4-prepare-audit",
        help="Prepare schema-checked, provider-safe Labeling-v4 calibration review records.",
    )
    prepare_input = prepare.add_mutually_exclusive_group(required=True)
    prepare_input.add_argument("--audit-selection", type=Path)
    prepare_input.add_argument("--inference-queue", type=Path)
    prepare.add_argument(
        "--bound-audit-selection",
        type=Path,
        help="Actual audit-selection source bound by --inference-queue (required for queue verification).",
    )
    prepare.add_argument(
        "--bound-prefilled-records",
        type=Path,
        help="Actual prefilled-record source bound by --inference-queue (required for queue verification).",
    )
    prepare.add_argument(
        "--bound-human-truth",
        type=Path,
        help="Actual human-truth source bound by --inference-queue (required for queue verification).",
    )
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument(
        "--inference-policy",
        choices=("deterministic-only", "cached-only", "semantic-minimal"),
        default="cached-only",
    )
    prepare.add_argument("--allow-provider-calls", action="store_true", default=False)
    prepare.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=None,
        help="Search provider-produced Labeling-v4 JSONL artifacts for compatible B/C results; repeatable.",
    )
    prepare.add_argument("--base-url", default="")
    prepare.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    prepare.add_argument("--vlm-model", default="")
    prepare.add_argument("--text-model", default="")
    prepare.add_argument("--timeout-seconds", type=float, default=90.0)
    prepare.set_defaults(func=_run_prepare_audit)

    freeze_queue = subparsers.add_parser(
        "label-v4-freeze-inference-queue",
        help="Freeze a deterministic quality-eligible Labeling-v4 inference queue.",
    )
    freeze_queue.add_argument("--audit-selection", type=Path, required=True)
    freeze_queue.add_argument("--prefilled-records", type=Path, required=True)
    freeze_queue.add_argument("--human-truth", type=Path, required=True)
    freeze_queue.add_argument("--output-root", type=Path, required=True)
    freeze_queue.add_argument(
        "--include-quality-state",
        action="append",
        choices=("quality_suitable", "quality_uncertain_usable"),
        default=None,
    )
    freeze_queue.add_argument("--allow-partial", action="store_true", default=False)
    freeze_queue.set_defaults(func=_run_freeze_inference_queue)

    shard = subparsers.add_parser("label-v4-shard", help="Run one deterministic, resumable Labeling-v4 shard.")
    shard.add_argument("--input", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard", default="0/1", help="Stable shard as I/N")
    shard.add_argument("--mode", choices=("A", "B", "C", "adaptive"), default="A")
    shard.add_argument("--workers", type=int, default=1)
    shard.add_argument("--max-records", type=int, default=None)
    shard.add_argument("--no-resume", action="store_true", default=False)
    shard.set_defaults(func=_run_shard)

    merge = subparsers.add_parser("label-v4-merge", help="Canonically merge a complete Labeling-v4 shard set.")
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.add_argument("--output", type=Path, default=None)
    merge.set_defaults(func=_run_merge)

    replay = subparsers.add_parser(
        "label-v4-replay", help="Replay saved Labeling-v4 artifacts with provider access disabled."
    )
    replay.add_argument("--input-pilot", type=Path, required=True)
    replay.add_argument("--output-root", type=Path, required=True)
    replay.add_argument("--shared-cache-root", type=Path, default=None)
    replay.add_argument("--require-complete-cache", action="store_true", default=False)
    replay.add_argument("--allow-deterministic-fallback", action="store_true", default=False)
    replay.set_defaults(func=_run_replay)

    calibration = subparsers.add_parser(
        "label-v4-calibration-wave1", help="Freeze the deterministic 100-representative human calibration audit."
    )
    calibration.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("datasets/sprite_lab_unlabeled_pool_v1_r2/candidate_manifest.jsonl"),
    )
    calibration.add_argument("--output-root", type=Path, required=True)
    calibration.add_argument("--target-size", type=int, default=100)
    calibration.add_argument("--seed", type=int, default=41)
    calibration.set_defaults(func=_run_calibration_wave1)


def _run_smoke(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.cohort import run_same_cohort_comparison

    result = run_same_cohort_comparison(
        cohort_path=parsed.cohort,
        resolved_path=parsed.resolved,
        output_dir=parsed.output_root,
        max_records=max(1, min(25, int(parsed.max_records))),
        sprite_ids=parsed.sprite_id,
        record_manifest=parsed.record_manifest,
    )
    _print_json(result)


def _run_cohort(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.cohort import run_same_cohort_comparison

    result = run_same_cohort_comparison(
        cohort_path=parsed.cohort,
        resolved_path=parsed.resolved,
        output_dir=parsed.output_root,
    )
    _print_json(result)


def _run_canary(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.cohort import select_cohort_rows
    from spritelab.harvest.label_v4.pipeline import LabelV4PipelineConfig, label_record_v4
    from spritelab.harvest.label_v4.providers import OpenAICompatibleJSONProvider

    maximum = max(1, min(5, int(parsed.max_records)))
    if parsed.require_shared_bc_cache and parsed.mode != "C":
        raise SystemExit("--require-shared-bc-cache requires --mode C")
    if parsed.require_shared_bc_cache and parsed.shared_cache_root is None:
        raise SystemExit("--require-shared-bc-cache requires --shared-cache-root")
    if (parsed.input_cost_per_million is None) != (parsed.output_cost_per_million is None):
        raise SystemExit("configure both --input-cost-per-million and --output-cost-per-million, or neither")
    api_key = os.environ.get(str(parsed.api_key_env), "").strip()
    if not api_key:
        raise SystemExit(f"provider key missing: set environment variable {parsed.api_key_env}")
    if parsed.base_url:
        base_url = str(parsed.base_url).rstrip("/")
    elif parsed.runpod_endpoint_id:
        base_url = f"https://api.runpod.ai/v2/{parsed.runpod_endpoint_id}/openai/v1"
    else:
        raise SystemExit("provide --runpod-endpoint-id or --base-url")

    cohort = select_cohort_rows(
        _read_jsonl(parsed.cohort),
        sprite_ids=parsed.sprite_id,
        record_manifest=parsed.record_manifest,
        max_records=maximum,
    )
    resolved = {str(row.get("sprite_id", "")): row for row in _read_jsonl(parsed.resolved)}
    ordered_ids = [str(row.get("sprite_id", "")) for row in cohort]
    missing = [sprite_id for sprite_id in ordered_ids if sprite_id not in resolved]
    if missing:
        raise SystemExit(f"targeted records missing from resolved inputs: {missing}")
    records = [resolved[sprite_id] for sprite_id in ordered_ids]
    text_model = str(parsed.text_model or parsed.vlm_model)
    verifier_model = str(parsed.verifier_model or parsed.vlm_model)
    vlm = OpenAICompatibleJSONProvider(
        base_url=base_url,
        api_key=api_key,
        model=parsed.vlm_model,
        namespace="blind_vlm_proposal_v4",
        timeout_seconds=parsed.timeout_seconds,
    )
    text = OpenAICompatibleJSONProvider(
        base_url=base_url,
        api_key=api_key,
        model=text_model,
        namespace="text_reconciliation_v4",
        timeout_seconds=parsed.timeout_seconds,
    )
    verifier = OpenAICompatibleJSONProvider(
        base_url=base_url,
        api_key=api_key,
        model=verifier_model,
        namespace="independent_verifier_v4",
        timeout_seconds=parsed.timeout_seconds,
    )
    config = LabelV4PipelineConfig(
        mode=parsed.mode,
        cache_dir=parsed.shared_cache_root or parsed.output_root / "cache_v1",
        shared_cache=parsed.shared_cache_root is not None,
        require_shared_bc_cache=bool(parsed.require_shared_bc_cache),
        input_cost_per_million=parsed.input_cost_per_million,
        output_cost_per_million=parsed.output_cost_per_million,
        use_cache=True,
        force_vlm_for_comparison=parsed.mode in {"B", "C"},
    )
    outputs = [
        label_record_v4(
            record,
            config=config,
            vlm_provider=vlm,
            text_provider=text,
            verifier_provider=verifier,
        )
        for record in records
    ]
    parsed.output_root.mkdir(parents=True, exist_ok=True)
    output_path = parsed.output_root / "canary_records_v1.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )
    provider_failures = [
        {
            "sprite_id": str(row.get("sprite_id", "")),
            "stage": str(stage.get("stage", "")),
            "failure_diagnostics": dict(stage.get("failure_diagnostics") or {}),
        }
        for row in outputs
        for stage in row.get("stage_ledger", [])
        if stage.get("failure_diagnostics")
    ]
    accounting = _summarize_provider_accounting(outputs)
    summary: dict[str, Any] = {
        "records": len(outputs),
        "ordered_sprite_ids": ordered_ids,
        "mode": parsed.mode,
        "base_url": base_url,
        "models": {"vlm": parsed.vlm_model, "text": text_model, "verifier": verifier_model},
        "provider_calls": accounting["new_provider_calls"],
        "cache_hits": accounting["cache_hits"],
        **accounting,
        "shared_cache_root": str(parsed.shared_cache_root) if parsed.shared_cache_root is not None else None,
        "require_shared_bc_cache": bool(parsed.require_shared_bc_cache),
        "provider_failure_count": len(provider_failures),
        "provider_failures": provider_failures,
        "status": "completed_with_provider_failures" if provider_failures else "completed",
        "credentials_persisted": False,
        "bulk_run": False,
        "warning": "Canary output is uncalibrated model evidence, not verified truth.",
        "output": str(output_path),
    }
    (parsed.output_root / "canary_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_json(summary)


def _run_assisted(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.assisted_v4_gui import launch_assisted_v4

    launch_assisted_v4(
        parsed.records,
        parsed.corrections,
        mode=parsed.mode,
        host=parsed.host,
        port=parsed.port,
        share=parsed.share,
        diagnostic_allow_selection=parsed.diagnostic_allow_selection,
    )


def _run_prepare_audit(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.audit_prefill import prepare_audit

    vlm = text = None
    if parsed.allow_provider_calls:
        if parsed.inference_policy != "semantic-minimal":
            raise SystemExit("--allow-provider-calls requires --inference-policy semantic-minimal")
        api_key = os.environ.get(str(parsed.api_key_env), "").strip()
        if not parsed.base_url or not parsed.vlm_model or not api_key:
            raise SystemExit(
                "provider calls require --base-url, --vlm-model, and the configured --api-key-env; "
                "omit --allow-provider-calls for a no-provider prefill"
            )
        from spritelab.harvest.label_v4.providers import OpenAICompatibleJSONProvider

        vlm = OpenAICompatibleJSONProvider(
            base_url=str(parsed.base_url).rstrip("/"),
            api_key=api_key,
            model=str(parsed.vlm_model),
            namespace="blind_vlm_proposal_v4",
            timeout_seconds=float(parsed.timeout_seconds),
        )
        text = OpenAICompatibleJSONProvider(
            base_url=str(parsed.base_url).rstrip("/"),
            api_key=api_key,
            model=str(parsed.text_model or parsed.vlm_model),
            namespace="text_reconciliation_v4",
            timeout_seconds=float(parsed.timeout_seconds),
        )
    roots = parsed.artifact_root or [
        Path("experiments/label_v4_real_pilot_15_v1"),
        Path("experiments/label_v4_pilot_replay_v2"),
    ]
    result = prepare_audit(
        parsed.audit_selection or parsed.inference_queue,
        parsed.output_root,
        inference_policy=parsed.inference_policy,
        allow_provider_calls=bool(parsed.allow_provider_calls),
        artifact_roots=roots,
        vlm_provider=vlm,
        text_provider=text,
        bound_audit_selection=parsed.bound_audit_selection,
        bound_prefilled_records=parsed.bound_prefilled_records,
        bound_human_truth=parsed.bound_human_truth,
    )
    _print_json(result)


def _run_freeze_inference_queue(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.two_pass import freeze_inference_queue

    result = freeze_inference_queue(
        parsed.audit_selection,
        parsed.prefilled_records,
        parsed.human_truth,
        parsed.output_root,
        inclusion_policy=parsed.include_quality_state or ("quality_suitable", "quality_uncertain_usable"),
        allow_partial=bool(parsed.allow_partial),
    )
    _print_json(result)


def _run_shard(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.batch import run_label_v4_shard
    from spritelab.harvest.label_v4.pipeline import LabelV4PipelineConfig

    try:
        index_text, count_text = str(parsed.shard).split("/", 1)
        index, count = int(index_text), int(count_text)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid --shard {parsed.shard!r}; expected I/N") from exc
    config = LabelV4PipelineConfig(
        mode=parsed.mode,
        cache_dir=parsed.output_root / "cache_v1",
        use_cache=True,
        force_vlm_for_comparison=parsed.mode in {"B", "C"},
    )
    result = run_label_v4_shard(
        parsed.input,
        parsed.output_root,
        shard_index=index,
        shard_count=count,
        config=config,
        workers=max(1, int(parsed.workers)),
        resume=not parsed.no_resume,
        max_records=parsed.max_records,
    )
    _print_json(result.to_dict())


def _run_merge(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.batch import merge_label_v4_shards

    result = merge_label_v4_shards(
        parsed.output_root,
        shard_count=parsed.shard_count,
        output_path=parsed.output,
    )
    _print_json(result)


def _run_replay(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.replay import replay_pilot

    result = replay_pilot(
        parsed.input_pilot,
        parsed.output_root,
        shared_cache_root=parsed.shared_cache_root,
        require_complete_cache=bool(parsed.require_complete_cache),
        allow_deterministic_fallback=bool(parsed.allow_deterministic_fallback),
    )
    _print_json(result)


def _run_calibration_wave1(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v4.calibration_wave import build_calibration_wave1

    if int(parsed.target_size) != 100:
        raise SystemExit("calibration wave 1 is frozen at exactly 100 representatives")
    result = build_calibration_wave1(
        parsed.candidate_manifest,
        parsed.output_root,
        target_size=100,
        seed=int(parsed.seed),
    )
    _print_json(result)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _add_record_targeting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sprite-id",
        action="append",
        default=[],
        help="Target one sprite id; repeat to preserve an explicit execution order.",
    )
    parser.add_argument(
        "--record-manifest",
        type=Path,
        default=None,
        help="Target JSONL rows containing sprite_id, preserving manifest order after explicit --sprite-id values.",
    )


def _summarize_provider_accounting(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "logical_stage_count": 0,
        "actual_http_attempts": 0,
        "new_provider_calls": 0,
        "shared_cache_hits": 0,
        "cache_hits": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "per_stage": {},
    }
    estimated_cost = 0.0
    pricing_configured = bool(outputs)
    for row in outputs:
        accounting = row.get("provider_accounting")
        if not isinstance(accounting, dict):
            accounting = _legacy_provider_accounting(row)
        for field_name in (
            "logical_stage_count",
            "actual_http_attempts",
            "new_provider_calls",
            "shared_cache_hits",
            "cache_hits",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            totals[field_name] += int(accounting.get(field_name, 0) or 0)
        configured = bool(accounting.get("pricing_configured"))
        cost = accounting.get("estimated_provider_cost")
        pricing_configured = pricing_configured and configured and cost is not None
        if configured and cost is not None:
            estimated_cost += float(cost)
        for stage, stage_values in dict(accounting.get("per_stage") or {}).items():
            if not isinstance(stage_values, dict):
                continue
            target = totals["per_stage"].setdefault(
                str(stage),
                {
                    "logical_executions": 0,
                    "new_provider_calls": 0,
                    "actual_http_attempts": 0,
                    "cache_hits": 0,
                    "shared_cache_hits": 0,
                    "execution_latency_ms": 0.0,
                    "provider_latency_ms": 0.0,
                    "new_token_usage": {},
                },
            )
            target["logical_executions"] += 1
            target["new_provider_calls"] += int(stage_values.get("new_provider_calls", 0) or 0)
            target["actual_http_attempts"] += int(stage_values.get("actual_http_attempts", 0) or 0)
            target["cache_hits"] += int(bool(stage_values.get("cache_hit")))
            target["shared_cache_hits"] += int(bool(stage_values.get("shared_cache_hit")))
            target["execution_latency_ms"] += float(stage_values.get("execution_latency_ms", 0.0) or 0.0)
            target["provider_latency_ms"] += float(stage_values.get("provider_latency_ms", 0.0) or 0.0)
            usage_target = target["new_token_usage"]
            for name, value in dict(stage_values.get("new_token_usage") or {}).items():
                usage_target[str(name)] = int(usage_target.get(str(name), 0)) + int(value or 0)
    totals["pricing_configured"] = pricing_configured
    totals["estimated_provider_cost"] = estimated_cost if pricing_configured else None
    return totals


def _legacy_provider_accounting(row: dict[str, Any]) -> dict[str, Any]:
    stages = [
        stage
        for stage in row.get("stage_ledger", [])
        if isinstance(stage, dict) and str(stage.get("stage", "")).startswith(("B_", "C_", "D_"))
    ]
    return {
        "logical_stage_count": len(stages),
        "actual_http_attempts": sum(int(stage.get("actual_http_attempts", 0) or 0) for stage in stages),
        "new_provider_calls": sum(bool(stage.get("provider_call")) for stage in stages),
        "shared_cache_hits": sum(bool(stage.get("shared_cache_hit")) for stage in stages),
        "cache_hits": sum(bool(stage.get("cache_hit")) for stage in stages),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "pricing_configured": False,
        "estimated_provider_cost": None,
        "per_stage": {},
    }


def _print_json(value: Any) -> None:
    """Write human-readable JSON through an explicitly UTF-8 stdout when supported."""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure) and str(getattr(sys.stdout, "encoding", "")).lower().replace("_", "-") != "utf-8":
        try:
            reconfigure(encoding="utf-8", errors="strict")
        except (OSError, ValueError):
            pass
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n")

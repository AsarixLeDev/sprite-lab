"""Qwen prefill commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.harvest.cli._args import (
    _add_qwen_prefill_args,
    _prefill_propagation_counts,
    _prefill_propagation_metadata,
    _quality_counts_from_harvested,
    _rehydrate_run,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_qwen_prefill(subparsers)


def _register_qwen_prefill(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("qwen-prefill", aliases=["qwen_prefill"], help="Batch Qwen metadata prefill for a run.")
    _add_qwen_prefill_args(p)
    p.set_defaults(func=_run_qwen_prefill)


def _run_qwen_prefill(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.autolabel import QwenBatchPrefillConfig, batch_prefill_with_qwen
    from spritelab.harvest.catalog import append_harvest_event, write_imported_jsonl, write_jsonl

    run_dir = Path(parsed.run)
    _, harvested = _rehydrate_run(run_dir)
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

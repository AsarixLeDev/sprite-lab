"""Shared argument helpers and utility functions for harvest CLI commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from spritelab.harvest.sources import SourceLicense, SourceRecord, normalize_license_name


def _parsed_config_kwargs(parsed: argparse.Namespace) -> dict[str, object]:
    values = vars(parsed).copy()
    values.pop("subcommand", None)
    values.pop("func", None)
    return values


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
        parser.add_argument(
            "--refresh-vlm",
            action="store_true",
            help="Ignore existing qwen_suggestions.jsonl and call the configured VLM backend when enabled.",
        )
        parser.add_argument(
            "--ignore-existing-vlm", action="store_true", help="Ignore qwen_suggestions.jsonl for this label-v2 run."
        )
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
    statuses = tuple(status.strip().lower() for status in raw_values if status.strip())
    return statuses or ("accepted",)


def _parse_runs_arg(value: str) -> tuple[Path, ...]:
    runs = tuple(Path(part.strip()) for part in str(value).split(",") if part.strip())
    if not runs:
        raise SystemExit("--runs must include at least one run directory")
    return runs


def _resolve_in_run(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path

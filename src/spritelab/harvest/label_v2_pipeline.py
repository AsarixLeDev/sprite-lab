"""Pipeline helpers for label-v2 CLI commands."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.dataset_maker.model import normalize_sprite_id
from spritelab.harvest.catalog import read_jsonl, write_jsonl
from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.label_candidates import candidate_objects_for_record, specialize_496_rpg_object
from spritelab.harvest.label_dedupe import duplicate_metadata_for_member, group_label_records_by_exact_rgba
from spritelab.harvest.label_fusion_v2 import FusionThresholds, fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion, label_suggestion_from_json, label_suggestion_to_json
from spritelab.harvest.label_taxonomy import normalize_tags
from spritelab.harvest.source_profiles import source_profile_to_json
from spritelab.harvest.visual_facts import extract_visual_facts_from_png, visual_facts_to_json

LABEL_V2_SUGGESTIONS = "label_v2_suggestions.jsonl"
LABEL_V2_REPORT = "label_v2_report.md"
LABEL_V2_SUMMARY = "label_v2_summary.json"
VLM_STAT_KEYS = (
    "vlm_reused_existing",
    "vlm_backend_called",
    "vlm_skipped_not_needed",
    "vlm_skipped_no_backend",
    "vlm_failed",
    "vlm_propagated_duplicate",
)


@dataclass(frozen=True)
class _VlmSelection:
    label: LabelSuggestion | None
    status: str
    stats: tuple[str, ...]


def build_label_v2_records(
    run_dir: str | Path,
    *,
    use_vlm: bool = False,
    vlm_only_when_needed: bool = False,
    max_items: int | None = None,
    propagate_dups: bool = True,
    trusted_filename_threshold: float = 0.85,
    auto_vlm_threshold: float = 0.8,
    review_conflicts: bool = False,
    existing_vlm_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    backend: Any | None = None,
    refresh_vlm: bool = False,
    ignore_existing_vlm: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Build label-v2 suggestion records for accepted sprites in a harvest run."""

    run_path = Path(run_dir)
    imported = [record for record in read_jsonl(run_path / "imported.jsonl") if str(record.get("status", "")).lower() == "accepted"]
    if max_items is not None:
        imported = imported[: max(0, int(max_items))]

    reuse_existing_vlm = not (refresh_vlm or ignore_existing_vlm)
    qwen_by_id = dict(existing_vlm_by_id or _qwen_by_id(read_jsonl(run_path / "qwen_suggestions.jsonl"))) if reuse_existing_vlm else {}
    thresholds = FusionThresholds(
        trusted_filename_threshold=trusted_filename_threshold,
        auto_vlm_threshold=auto_vlm_threshold,
        review_trusted_filename_conflicts=review_conflicts,
    )

    groups = group_label_records_by_exact_rgba(imported, run_dir=run_path) if propagate_dups else []
    if not groups:
        groups = [
            type("_SingleGroup", (), {"representative_index": index, "member_indices": (index,), "kind": "single", "representative_sprite_id": str(record.get("sprite_id", ""))})()
            for index, record in enumerate(imported)
        ]

    worker_count = max(1, int(workers or 1))
    representative_selections = _select_vlm_for_groups(
        groups,
        imported=imported,
        run_path=run_path,
        use_vlm=use_vlm,
        vlm_only_when_needed=vlm_only_when_needed,
        qwen_by_id=qwen_by_id,
        backend=backend,
        workers=worker_count,
    )

    output_by_index: dict[int, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        rep_index = int(group.representative_index)
        rep_selection = representative_selections[group_index]
        for index in group.member_indices:
            record = imported[index]
            sprite_id = str(record.get("sprite_id", ""))
            duplicate_metadata = duplicate_metadata_for_member(group, sprite_id) if propagate_dups and hasattr(group, "member_sprite_ids") else {}
            if duplicate_metadata:
                selection = _VlmSelection(rep_selection.label, "propagated_duplicate", ("vlm_propagated_duplicate",))
            elif index == rep_index:
                selection = rep_selection
            else:
                selection = _vlm_for_record(
                    record,
                    run_dir=run_path,
                    use_vlm=use_vlm,
                    vlm_only_when_needed=vlm_only_when_needed,
                    qwen_by_id=qwen_by_id,
                    backend=backend,
                )
            output_by_index[index] = build_label_v2_record(
                record,
                run_dir=run_path,
                vlm=selection.label,
                vlm_status=selection.status,
                vlm_stats=selection.stats,
                thresholds=thresholds,
                duplicate_metadata=duplicate_metadata,
            )

    return [output_by_index[index] for index in sorted(output_by_index)]


def _select_vlm_for_groups(
    groups: Sequence[Any],
    *,
    imported: Sequence[Mapping[str, Any]],
    run_path: Path,
    use_vlm: bool,
    vlm_only_when_needed: bool,
    qwen_by_id: Mapping[str, Mapping[str, Any]],
    backend: Any | None,
    workers: int,
) -> dict[int, _VlmSelection]:
    """Run VLM selection once per duplicate group representative."""

    if not groups:
        return {}

    def run_one(group_index: int) -> tuple[int, _VlmSelection]:
        group = groups[group_index]
        rep_record = imported[int(group.representative_index)]
        selection = _vlm_for_record(
            rep_record,
            run_dir=run_path,
            use_vlm=use_vlm,
            vlm_only_when_needed=vlm_only_when_needed,
            qwen_by_id=qwen_by_id,
            backend=backend,
        )
        return group_index, selection

    indices = list(range(len(groups)))
    selections: dict[int, _VlmSelection] = {}
    if workers <= 1 or len(indices) <= 1:
        for group_index in _progress(indices, "label-v2 VLM", total=len(indices)):
            key, selection = run_one(group_index)
            selections[key] = selection
        return selections

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, group_index) for group_index in indices]
        for future in _progress(as_completed(futures), "label-v2 VLM", total=len(futures)):
            key, selection = future.result()
            selections[key] = selection
    return selections


def build_label_v2_record(
    record: Mapping[str, Any],
    *,
    run_dir: str | Path,
    vlm: LabelSuggestion | None,
    thresholds: FusionThresholds,
    vlm_status: str = "",
    vlm_stats: Sequence[str] = (),
    duplicate_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    filename_result = suggest_from_filename_v2(record)
    candidate_object_names = candidate_objects_for_record(
        record,
        filename_result.profile,
        label_suggestion_to_json(filename_result.suggestion),
    )
    image_path = _resolve_path(record, run_dir=run_dir)
    visual_facts = None
    if image_path.exists():
        try:
            visual_facts = extract_visual_facts_from_png(image_path)
        except Exception:
            visual_facts = None
    specialization = specialize_496_rpg_object(
        record,
        filename_result.profile,
        filename_result.suggestion,
        parsed_tokens=filename_result.parsed_tokens,
        candidate_object_names=candidate_object_names,
        vlm=vlm,
        visual_facts=visual_facts,
    )
    if specialization.candidate_object_names:
        candidate_object_names = specialization.candidate_object_names
    filename_suggestion = (
        replace(filename_result.suggestion, candidate_object_names=candidate_object_names)
        if candidate_object_names
        else filename_result.suggestion
    )
    if specialization.object_name or specialization.category:
        filename_suggestion = replace(
            filename_suggestion,
            category=specialization.category or filename_suggestion.category,
            object_name=specialization.object_name or filename_suggestion.object_name,
            tags=_specialized_tags(filename_suggestion.tags, specialization.object_name or filename_suggestion.object_name, specialization.category or filename_suggestion.category),
            confidence=max(filename_suggestion.confidence, 0.78 if specialization.object_name else filename_suggestion.confidence),
            confidence_reason=_specialized_confidence_reason(filename_suggestion.confidence_reason, specialization.flags),
            short_description=_specialized_description(specialization.object_name or filename_suggestion.object_name, specialization.category or filename_suggestion.category),
            evidence=(*filename_suggestion.evidence, *(f"rpg_496_specialization:{flag}" for flag in specialization.flags)),
            candidate_object_names=candidate_object_names,
        )
    if vlm is not None and candidate_object_names:
        vlm = replace(vlm, candidate_object_names=candidate_object_names)
    fused = fuse_label_v2(
        filename_suggestion,
        vlm,
        visual_facts,
        profile=filename_result.profile,
        thresholds=thresholds,
    )
    if specialization.flags:
        fused = replace(
            fused,
            flags=(*fused.flags, *specialization.flags),
            safe_prefill=replace(fused.safe_prefill, candidate_object_names=candidate_object_names),
            fused_suggestion=replace(fused.fused_suggestion, candidate_object_names=candidate_object_names),
        )
    safe = label_suggestion_to_json(fused.safe_prefill) or {}
    filename_json = label_suggestion_to_json(filename_suggestion) or {}
    vlm_json = label_suggestion_to_json(vlm) if vlm is not None else None
    label_quality = {
        "bucket": fused.bucket,
        "needs_review": fused.needs_review,
        "flags": list(fused.flags),
        "conflict_reasons": list(fused.conflict_reasons),
        "provenance": fused.provenance,
        "review_priority": fused.review_priority,
    }
    output = {
        "sprite_id": str(record.get("sprite_id", "")),
        "filename": Path(str(record.get("relative_path") or record.get("final_png_path") or "")).name,
        "source_id": str(record.get("source_id", "")),
        "source_name": str(record.get("source_name", "")),
        "relative_path": str(record.get("relative_path", "")),
        "final_png_path": str(record.get("final_png_path", "")),
        "source_profile": source_profile_to_json(filename_result.profile),
        "candidate_object_names": list(candidate_object_names),
        "filename_suggestion": filename_json,
        "filename_rule_result": {
            "parsed_tokens": list(filename_result.parsed_tokens),
            "raw_tokens": list(filename_result.raw_tokens),
            "confidence": filename_result.confidence,
            "confidence_reason": filename_result.confidence_reason,
        },
        "vlm_descriptor": vlm_json,
        "vlm_suggestion": vlm_json,
        "vlm_status": vlm_status,
        "vlm_stats": list(vlm_stats),
        "visual_facts": visual_facts_to_json(visual_facts),
        "safe_prefill": safe,
        "fused_suggestion": safe,
        "label_quality": label_quality,
        "bucket": fused.bucket,
        "needs_review": fused.needs_review,
        "flags": list(fused.flags),
        "conflict_reasons": list(fused.conflict_reasons),
        "provenance": fused.provenance,
        "review_priority": fused.review_priority,
    }
    output.update(dict(duplicate_metadata or {}))
    if duplicate_metadata:
        output["label_quality"]["flags"] = sorted({*output["label_quality"].get("flags", ()), "duplicate_propagated"})
        output["flags"] = sorted({*output.get("flags", ()), "duplicate_propagated"})
    return output


def write_label_v2_outputs(run_dir: str | Path, records: Sequence[Mapping[str, Any]], *, out: str | Path | None = None) -> dict[str, Path]:
    run_path = Path(run_dir)
    suggestions_path = Path(out) if out is not None else run_path / LABEL_V2_SUGGESTIONS
    write_jsonl(suggestions_path, records)
    summary = summarize_label_v2_records(records)
    summary_path = run_path / LABEL_V2_SUMMARY
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = run_path / LABEL_V2_REPORT
    report_path.write_text(format_label_v2_run_report(summary), encoding="utf-8")
    return {"suggestions": suggestions_path, "summary": summary_path, "report": report_path}


def summarize_label_v2_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    candidate_families: Counter[str] = Counter()
    vlm_stats: Counter[str] = Counter({key: 0 for key in VLM_STAT_KEYS})
    duplicate_count = 0
    needs_review = 0
    records_with_candidates = 0
    for record in records:
        quality = record.get("label_quality") if isinstance(record.get("label_quality"), Mapping) else {}
        bucket = str(record.get("bucket") or quality.get("bucket") or "missing")
        buckets[bucket] += 1
        if bool(record.get("needs_review", quality.get("needs_review", False))) or bucket.startswith("needs_review"):
            needs_review += 1
        for flag in record.get("flags") or quality.get("flags") or ():
            flags[str(flag)] += 1
        safe = record.get("safe_prefill") if isinstance(record.get("safe_prefill"), Mapping) else {}
        categories[str(safe.get("category", "unknown"))] += 1
        for tag in safe.get("tags") or ():
            tags[str(tag)] += 1
        filename = _object_from_mapping(record.get("filename_suggestion"))
        vlm = _object_from_mapping(record.get("vlm_descriptor") or record.get("vlm_suggestion"))
        if filename and vlm and filename != vlm:
            conflicts[f"{filename}->{vlm}"] += 1
        candidates = tuple(str(candidate) for candidate in record.get("candidate_object_names") or () if str(candidate))
        if candidates:
            records_with_candidates += 1
            candidate_families[candidates[0]] += 1
        for stat in record.get("vlm_stats") or ():
            stat_key = str(stat)
            if stat_key in VLM_STAT_KEYS:
                vlm_stats[stat_key] += 1
        if record.get("duplicate_propagation") == "exact":
            duplicate_count += 1
    total = len(records)
    auto = total - needs_review
    return {
        "total": total,
        "auto_count": auto,
        "needs_review_count": needs_review,
        "review_rate": needs_review / total if total else 0.0,
        "buckets": dict(sorted(buckets.items())),
        "flags": dict(flags.most_common()),
        "top_categories": dict(categories.most_common(20)),
        "top_tags": dict(tags.most_common(30)),
        "top_conflict_pairs": dict(conflicts.most_common(20)),
        "duplicate_propagation_count": duplicate_count,
        "vlm_stats": {key: int(vlm_stats.get(key, 0)) for key in VLM_STAT_KEYS},
        "known_hallucinations": flags.get("vlm_known_hallucination", 0),
        "records_with_candidates": records_with_candidates,
        "records_without_candidates": max(0, total - records_with_candidates),
        "top_candidate_families": dict(candidate_families.most_common(20)),
    }


def format_label_v2_run_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Label v2 Report",
        "",
        f"Total: {summary.get('total', 0)}",
        f"Auto: {summary.get('auto_count', 0)}",
        f"Needs review: {summary.get('needs_review_count', 0)}",
        f"Review rate: {float(summary.get('review_rate', 0.0)):.3f}",
        f"Known hallucinations: {summary.get('known_hallucinations', 0)}",
        f"Duplicate propagations: {summary.get('duplicate_propagation_count', 0)}",
        "",
        "## VLM",
    ]
    vlm_stats = dict(summary.get("vlm_stats") or {})
    for key in VLM_STAT_KEYS:
        lines.append(f"- {key}: {int(vlm_stats.get(key, 0))}")
    lines.extend(
        [
            "",
            "## Candidates",
            f"- records_with_candidates: {int(summary.get('records_with_candidates', 0))}",
            f"- records_without_candidates: {int(summary.get('records_without_candidates', 0))}",
        ]
    )
    candidate_families = dict(summary.get("top_candidate_families") or {})
    if candidate_families:
        lines.append("- top_candidate_families:")
        for family, count in candidate_families.items():
            lines.append(f"  - {family}: {count}")
    lines.extend([
        "",
        "## Buckets",
    ])
    for bucket, count in dict(summary.get("buckets") or {}).items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## Flags"])
    for flag, count in dict(summary.get("flags") or {}).items():
        lines.append(f"- {flag}: {count}")
    conflicts = dict(summary.get("top_conflict_pairs") or {})
    if conflicts:
        lines.extend(["", "## Top Conflict Pairs"])
        for pair, count in conflicts.items():
            lines.append(f"- {pair}: {count}")
    return "\n".join(lines) + "\n"


def create_vlm_backend_from_args(parsed: Any) -> Any | None:
    """Create an existing prefill backend from CLI args, or None."""

    backend_name = str(getattr(parsed, "backend", "none")).strip().lower()
    if backend_name in {"", "none"}:
        return None
    from spritelab.dataset_maker.prefill import PrefillConfig, create_prefill_backend

    return create_prefill_backend(
        PrefillConfig(
            enabled=True,
            backend=backend_name,
            model=getattr(parsed, "model", "Qwen/Qwen3-VL-8B-Instruct"),
            base_url=getattr(parsed, "base_url", "http://127.0.0.1:8000/v1"),
            api_key=getattr(parsed, "api_key", "not-needed"),
            runpod_token=getattr(parsed, "runpod_token", ""),
            timeout_seconds=float(getattr(parsed, "timeout_seconds", 60.0)),
            cache_dir=getattr(parsed, "cache_dir", None),
            include_filename_hint=True,
            vlm_role=getattr(parsed, "vlm_role", "descriptor"),
            structured_output=getattr(parsed, "structured_output", "auto"),
            vlm_image_view=getattr(parsed, "vlm_image_view", "both"),
            votes=1,
            vote_mode="off",
        )
    )


def _vlm_for_record(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    use_vlm: bool,
    vlm_only_when_needed: bool,
    qwen_by_id: Mapping[str, Mapping[str, Any]],
    backend: Any | None,
) -> _VlmSelection:
    filename_result = suggest_from_filename_v2(record)
    candidate_object_names = candidate_objects_for_record(
        record,
        filename_result.profile,
        label_suggestion_to_json(filename_result.suggestion),
    )
    if vlm_only_when_needed and filename_result.profile.trusted_filename and filename_result.confidence >= 0.9:
        return _VlmSelection(None, "skipped_not_needed", ("vlm_skipped_not_needed",))
    sprite_id = str(record.get("sprite_id", ""))
    if sprite_id in qwen_by_id:
        return _VlmSelection(_label_from_vlm_dict(qwen_by_id[sprite_id]), "reused_existing", ("vlm_reused_existing",))
    if not use_vlm or backend is None:
        return _VlmSelection(None, "skipped_no_backend", ("vlm_skipped_no_backend",))
    label = _call_vlm_backend(
        record,
        run_dir=run_dir,
        backend=backend,
        filename_suggestion=filename_result.suggestion,
        profile=filename_result.profile,
        candidate_object_names=candidate_object_names,
    )
    if label is None or _vlm_call_failed(label):
        return _VlmSelection(label, "failed", ("vlm_backend_called", "vlm_failed"))
    return _VlmSelection(label, "backend_called", ("vlm_backend_called",))


def _call_vlm_backend(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    backend: Any,
    filename_suggestion: LabelSuggestion,
    profile: SourceProfile,
    candidate_object_names: Sequence[str] = (),
) -> LabelSuggestion | None:
    from spritelab.dataset_maker.prefill import PrefillRequest, suggestion_to_json_dict

    image_path = _resolve_path(record, run_dir=run_dir)
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
        facts = extract_visual_facts_from_png(image_path)
        filename_json = label_suggestion_to_json(filename_suggestion) or {}
        filename_json["source_profile_name"] = profile.name
        filename_json["filename_trust"] = profile.filename_trust
        request = PrefillRequest(
            sprite_id=normalize_sprite_id(str(record.get("sprite_id", ""))),
            image=rgba,
            existing_category=str(record.get("category", "unknown")),
            existing_tags=tuple(str(tag) for tag in record.get("tags") or ()),
            source_path=str(image_path),
            filename_suggestion=filename_json,
            image_facts=visual_facts_to_json(facts),
            candidate_object_names=tuple(candidate_object_names),
        )
        suggestion = backend.suggest(request)
    except Exception as exc:
        return LabelSuggestion(
            category="unknown",
            object_name="",
            warnings=(f"vlm_descriptor_failed: {exc}",),
            source="vlm_descriptor",
        )
    return _label_from_vlm_dict(suggestion_to_json_dict(suggestion))


def _label_from_vlm_dict(data: Mapping[str, Any]) -> LabelSuggestion | None:
    label = label_suggestion_from_json({**dict(data), "source": str(data.get("source") or "vlm_descriptor")})
    if label is None:
        return None
    return label


def _vlm_call_failed(label: LabelSuggestion) -> bool:
    return any(str(warning).startswith("vlm_descriptor_failed") for warning in label.warnings)


def _qwen_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("sprite_id", "")): {key: value for key, value in record.items() if key != "sprite_id"}
        for record in records
        if record.get("sprite_id")
    }


def _resolve_path(record: Mapping[str, Any], *, run_dir: str | Path) -> Path:
    raw = str(record.get("final_png_path", "")).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    run_path = Path(run_dir)
    for candidate in (Path.cwd() / path, run_path / path, run_path.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _object_from_mapping(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("object_name") or value.get("possible_object_name") or "")


def _specialized_tags(existing: Sequence[str], object_name: str, category: str) -> tuple[str, ...]:
    tags = [object_name, *existing]
    if category == "material":
        tags.extend(["material", "crafting_material"])
    elif category == "armor":
        tags.extend(["armor", "defense"])
        if "shield" in object_name:
            tags.append("shield")
        if "chestplate" in object_name:
            tags.extend(["chestplate", "wearable"])
    elif category == "weapon":
        tags.append("weapon")
        if "sword" in object_name or "dagger" in object_name:
            tags.append("blade")
        if "arrow" in object_name or "bow" in object_name:
            tags.append("ranged")
    elif category == "effect_icon":
        tags.extend(["effect", "status_effect"])
        if "arrow" in object_name:
            tags.extend(["arrow", "ranged"])
        if "buff" in object_name:
            tags.append("buff")
    else:
        if any(part in object_name for part in ("potion", "vial", "bottle", "flask")):
            tags.extend(["potion", "liquid", "container"])
        if any(part in object_name for part in ("meat", "fish", "pie", "watermelon")):
            tags.extend(["food", "consumable"])
    return normalize_tags(tags)


def _specialized_confidence_reason(existing: str, flags: Sequence[str]) -> str:
    if not flags:
        return existing
    return f"{existing}; 496 RPG specialization: {', '.join(flags)}" if existing else f"496 RPG specialization: {', '.join(flags)}"


def _specialized_description(object_name: str, category: str) -> str:
    if not object_name:
        return ""
    return f"A 32x32 pixel-art {object_name.replace('_', ' ')} {category.replace('_', ' ')}."


def _progress(items: Any, description: str, *, total: int | None = None) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(items, desc=description, total=total)
    except ImportError:
        return items

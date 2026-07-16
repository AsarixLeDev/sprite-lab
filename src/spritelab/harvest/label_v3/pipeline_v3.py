"""Auto-Labeling v3: pipeline orchestration.

Independently resumable stages with deterministic sharding, content-addressed
caching, and atomic per-record persistence. Each stage has:
  - immutable input identity
  - config/stage hash
  - content-addressed output
  - append-safe persistence
  - completion ledger
  - failure queue
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v3.calibration import (
    CalibrationArtifact,
    calibration_support_for_field,
)
from spritelab.harvest.label_v3.config_v3 import V3LabelingPolicy, V3PipelineConfig
from spritelab.harvest.label_v3.deterministic_evidence import (
    DeterministicEvidenceBatch,
    extract_deterministic_evidence,
)
from spritelab.harvest.label_v3.evidence import evidence_item_to_json
from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, FieldDecision, TagDecision
from spritelab.harvest.label_v3.field_prefill import FieldPrefill, build_prefills
from spritelab.harvest.label_v3.fusion_v3 import (
    build_field_fusion_input,
    fuse_field,
    fuse_hierarchical_object,
    validate_combinations,
)
from spritelab.harvest.label_v3.pack_context import analyze_pack_context, pack_outlier_score
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    RecordState,
    derive_record_state,
    record_decision_to_json,
)
from spritelab.harvest.label_v3.vlm_orchestration import (
    VlmCascadeResult,
    create_unavailable_cascade,
    run_vlm_cascade,
)
from spritelab.utils.jsonl import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v3.1.0"

RECORD_OUTPUT_SUFFIX = "_v3_records.jsonl"
EVIDENCE_OUTPUT_SUFFIX = "_v3_evidence.jsonl"
LEDGER_SUFFIX = "_v3_ledger.json"
FAILURES_SUFFIX = "_v3_failures.jsonl"
REPORT_SUFFIX = "_v3_report.md"
SUMMARY_SUFFIX = "_v3_summary.json"


def _progress(total: int, description: str):
    """Use a terminal progress bar when tqdm is available, else log progress."""
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=description, unit="sprite", leave=True)
    except ImportError:
        return None


def _safe_provider_label(backend: str, base_url: str, model: str) -> str:
    """Human-readable config line; deliberately excludes API credentials."""
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(base_url)
    safe_url = urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    return f"provider={backend or 'none'} endpoint={safe_url or '<default>'} model={model or '<default>'}"


@dataclass
class PipelineStageOutput:
    stage: str
    records: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    completed_count: int
    failed_count: int
    cache_hits: int
    duration_seconds: float


@dataclass
class PipelineRunResult:
    run_dir: Path
    output_dir: Path
    policy_hash: str
    stages: dict[str, PipelineStageOutput]
    total_records: int
    auto_accept: int
    partial_accept: int
    quarantine: int
    hard_reject: int
    unknown: int
    vlm_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "policy_hash": self.policy_hash,
            "stages": list(self.stages.keys()),
            "total_records": self.total_records,
            "auto_accept": self.auto_accept,
            "partial_accept": self.partial_accept,
            "quarantine": self.quarantine,
            "hard_reject": self.hard_reject,
            "unknown": self.unknown,
            "acceptance_rate": (self.auto_accept + self.partial_accept) / max(1, self.total_records),
            "vlm_calls": sum(
                max(0, s.completed_count - s.cache_hits) for s in self.stages.values() if "vlm" in s.stage
            ),
            "vlm_cache_hits": sum(s.cache_hits for s in self.stages.values() if "vlm" in s.stage),
            "vlm_metrics": dict(self.vlm_metrics),
        }


def run_v3_pipeline(
    run_dir: str | Path,
    output_root: str | Path,
    config: V3PipelineConfig,
    *,
    calibration: CalibrationArtifact | None = None,
    use_vlm: bool = False,
    max_records: int | None = None,
    dry_run: bool = True,
) -> PipelineRunResult:
    """Run the complete v3 labeling pipeline.

    Stages:
      1. input_identity — load records, verify integrity
      2. deterministic_evidence — extract all deterministic evidence
      3. vlm_stages — run staged VLM cascade (if enabled)
      4. fusion — fuse per-field and produce record decisions
      5. report — generate summary and per-pack reports
    """

    run_path = Path(run_dir)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    policy_hash = config.policy.policy_hash()

    # Stage 1: Input identity
    records = _load_records(run_path, max_records=max_records)
    if not records:
        return PipelineRunResult(
            run_dir=run_path,
            output_dir=output_path,
            policy_hash=policy_hash,
            stages={},
            total_records=0,
            auto_accept=0,
            partial_accept=0,
            quarantine=0,
            hard_reject=0,
            unknown=0,
        )

    # Derive one generic, non-duplicated pack dependency group.  Source path,
    # archive name, and profile are inputs to this artifact, not separate votes.
    pack_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("source_id") or record.get("pack_name") or record.get("source_name") or "unknown")
        pack_groups.setdefault(key, []).append(record)
    for grouped_records in pack_groups.values():
        context = analyze_pack_context(grouped_records)
        for record in grouped_records:
            record["_v3_pack_context"] = {
                **context.to_json(),
                "pack_outlier_score": pack_outlier_score(record, context),
                "dependency_group": "pack_context",
            }

    # Stage 2: Deterministic evidence
    evidence_batches: list[tuple[dict[str, Any], DeterministicEvidenceBatch]] = []
    for record in records:
        batch = extract_deterministic_evidence(record, run_dir=run_path)
        evidence_batches.append((record, batch))

    # Stage 3: VLM cascade.  This remains shadow-only: its output is written
    # beside deterministic evidence and is still subject to field calibration.
    vlm_results: dict[str, VlmCascadeResult] = {}
    metric_names = (
        "logical_stage_requests",
        "successful_stage_outputs",
        "cache_hits",
        "http_attempts",
        "retries",
        "timeouts",
        "transport_failures",
        "json_parse_failures",
        "schema_validation_failures",
        "fallbacks",
        "abstentions_caused_by_backend_failure",
    )
    vlm_stats: dict[str, Any] = dict.fromkeys(metric_names, 0)
    vlm_stats["by_stage"] = {}
    # Compatibility aliases retained for existing report and GUI consumers.
    vlm_stats.update({"calls": 0, "invalid_outputs": 0, "abstentions": 0, "variant_calls_saved": 0})
    if use_vlm and config.vlm_backend not in {"", "none"}:
        from spritelab.harvest.label_v3.stage_cache_v3 import StageCache
        from spritelab.harvest.label_v3.taxonomy_v3 import all_hierarchy_nodes, taxonomy_version_hash
        from spritelab.harvest.label_v3.vlm_runtime import VlmRuntimeConfig, create_v3_backend, prepare_v3_views

        cache = StageCache(config.vlm_cache_dir or (output_path / "vlm_stage_cache"))
        vocabulary = tuple(sorted(set(all_hierarchy_nodes()) | {"unknown", "none_of_the_above"}))
        workers = min(5, max(1, int(config.vlm_concurrency)))
        logger.info(
            "VLM starting: %s workers=%d timeout=%.1fs retries=%d cache=%s",
            _safe_provider_label(config.vlm_backend, config.vlm_base_url, config.vlm_model),
            workers,
            config.vlm_timeout_seconds,
            config.vlm_retries,
            cache.root,
        )

        def run_one(record: Mapping[str, Any]) -> tuple[str, VlmCascadeResult]:
            sprite_id = str(record.get("sprite_id", ""))
            png = _record_png_path(record, run_path)
            if png is None:
                return sprite_id, create_unavailable_cascade(sprite_id, reason="missing_sprite_image")
            try:
                views, preprocessing_hash, image_hash, geometry_hash = prepare_v3_views(png)
                backend = create_v3_backend(
                    VlmRuntimeConfig(
                        backend=config.vlm_backend,
                        model=config.vlm_model or "Qwen/Qwen3-VL-8B-Instruct",
                        base_url=config.vlm_base_url,
                        api_key=config.vlm_api_key,
                        structured_output=config.vlm_structured_output,
                        prompt_version=config.vlm_prompt_version,
                        disable_thinking=config.vlm_disable_thinking,
                        timeout_seconds=config.vlm_timeout_seconds,
                        retries=config.vlm_retries,
                        concurrency=config.vlm_concurrency,
                        retry_backoff_seconds=config.vlm_retry_backoff_seconds,
                        cache_dir=config.vlm_cache_dir,
                        failure_diagnostics_enabled=config.vlm_failure_diagnostics_enabled,
                        failure_diagnostics_dir=(
                            config.vlm_failure_diagnostics_dir or str(output_path / "vlm_failure_diagnostics")
                        ),
                    ),
                    views,
                )
                cascade = run_vlm_cascade(
                    sprite_id,
                    backend=backend,
                    image_ref=views["checkerboard"],
                    image_hash=image_hash,
                    image_view="checkerboard+nearest_neighbor+tight_crop",
                    candidates=vocabulary,
                    cache=cache,
                    preprocessing_hash=preprocessing_hash,
                    taxonomy_hash=taxonomy_version_hash(),
                    geometry_hash=geometry_hash,
                    prompt_version=config.vlm_prompt_version,
                    profile=config.vlm_cascade_profile,
                )
                return sprite_id, cascade
            except Exception as exc:
                logger.warning("VLM prefill unavailable for %s: %s", sprite_id, exc)
                return sprite_id, create_unavailable_cascade(sprite_id, reason=f"runtime_error:{type(exc).__name__}")

        # The runtime configuration clamps workers to 5.  This permits the
        # requested RunPod concurrency without turning a small review run into
        # unbounded paid inference.
        vlm_connected = False
        failures_by_reason: Counter[str] = Counter()
        progress = _progress(len(records), "VLM prefill")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, record) for record in records]
            for future in as_completed(futures):
                sprite_id, cascade = future.result()
                vlm_results[sprite_id] = cascade
                if progress is not None:
                    progress.update(1)
                else:
                    logger.info("VLM progress: %d/%d sprites", len(vlm_results), len(records))
                stages = (cascade.stage_a, cascade.stage_b, cascade.stage_c, cascade.stage_d, cascade.stage_e)
                for stage in stages:
                    stage_metrics = stage.metrics.as_dict()
                    bucket = vlm_stats["by_stage"].setdefault(stage.stage_id, dict.fromkeys(metric_names, 0))
                    for name in metric_names:
                        bucket[name] += stage_metrics[name]
                        vlm_stats[name] += stage_metrics[name]
                vlm_stats["variant_calls_saved"] += sum(
                    1
                    for stage in cascade.available_stages()
                    if stage.cache_hit and stage.stage_id != "stage_a_blind_descriptor"
                )
                vlm_stats["calls"] = vlm_stats["http_attempts"]
                vlm_stats["invalid_outputs"] = (
                    vlm_stats["json_parse_failures"] + vlm_stats["schema_validation_failures"]
                )
                vlm_stats["abstentions"] = vlm_stats["abstentions_caused_by_backend_failure"]
                if cascade.available_stages() and not vlm_connected:
                    vlm_connected = True
                    logger.info("VLM connected: received valid structured output (first sprite=%s)", sprite_id)
                if cascade.all_failed:
                    reason = cascade.stage_a.failure_reason or "unavailable"
                    failures_by_reason[reason] += 1
                    logger.warning(
                        "VLM fallback: sprite=%s reason=%s; continuing with deterministic evidence", sprite_id, reason
                    )
        if progress is not None:
            progress.close()
        if not vlm_connected:
            logger.warning("VLM fallback active: no valid VLM responses; all records use deterministic evidence only")
        logger.info(
            "VLM complete: logical=%d successful=%d cache_hits=%d HTTP_attempts=%d retries=%d timeouts=%d transport=%d JSON=%d schema=%d fallbacks=%d backend_abstentions=%d",
            vlm_stats["logical_stage_requests"],
            vlm_stats["successful_stage_outputs"],
            vlm_stats["cache_hits"],
            vlm_stats["http_attempts"],
            vlm_stats["retries"],
            vlm_stats["timeouts"],
            vlm_stats["transport_failures"],
            vlm_stats["json_parse_failures"],
            vlm_stats["schema_validation_failures"],
            vlm_stats["fallbacks"],
            vlm_stats["abstentions_caused_by_backend_failure"],
        )
    else:
        vlm_results = {
            str(r.get("sprite_id", "")): create_unavailable_cascade(
                str(r.get("sprite_id", "")), reason="no_vlm_backend"
            )
            for r in records
        }
        logger.info("VLM disabled: deterministic prefill only")

    text_generator = None
    if config.text_enrichment_enabled and config.text_enrichment_model:
        from spritelab.harvest.label_v3.vlm_runtime import VlmRuntimeConfig, make_text_enricher

        text_generator = make_text_enricher(
            VlmRuntimeConfig(
                backend=config.text_enrichment_backend,
                model=config.vlm_model or "Qwen/Qwen3-VL-8B-Instruct",
                base_url=config.text_enrichment_base_url or config.vlm_base_url,
                api_key=config.text_enrichment_api_key or config.vlm_api_key,
                timeout_seconds=config.text_enrichment_timeout_seconds,
                retries=config.text_enrichment_retries,
                retry_backoff_seconds=config.vlm_retry_backoff_seconds,
                enrichment_enabled=True,
                enrichment_model=config.text_enrichment_model,
            )
        )
        if text_generator is None:
            logger.warning("LLM enrichment fallback: backend is unavailable; canonical descriptions only")
        else:
            logger.info(
                "LLM enrichment starting: %s timeout=%.1fs retries=%d",
                _safe_provider_label(
                    config.text_enrichment_backend,
                    config.text_enrichment_base_url or config.vlm_base_url,
                    config.text_enrichment_model,
                ),
                config.text_enrichment_timeout_seconds,
                config.text_enrichment_retries,
            )

    # Stage 4: Fusion
    record_decisions: list[RecordDecision] = []
    text_progress = _progress(len(evidence_batches), "LLM enrichment") if text_generator is not None else None
    for record, batch in evidence_batches:
        sprite_id = str(record.get("sprite_id", ""))
        decision = _fuse_record(
            sprite_id=sprite_id,
            batch=batch,
            vlm=vlm_results.get(sprite_id),
            policy=config.policy,
            policy_hash=policy_hash,
            calibration=calibration,
            record=record,
            enrichment_generator=text_generator,
        )
        record_decisions.append(decision)
        if text_progress is not None:
            text_progress.update(1)
    if text_progress is not None:
        text_progress.close()
        enriched = sum(1 for record in record_decisions if record.description_artifact.get("enriched_description"))
        invalid = sum(1 for record in record_decisions if not record.description_artifact.get("valid", True))
        logger.info("LLM enrichment complete: descriptions=%d invalid_fallbacks=%d", enriched, invalid)

    # Count record states
    state_counts: Counter[str] = Counter()
    for rd in record_decisions:
        state_counts[rd.record_state] += 1

    # Write outputs
    result = PipelineRunResult(
        run_dir=run_path,
        output_dir=output_path,
        policy_hash=policy_hash,
        stages={
            "deterministic": PipelineStageOutput(
                stage="deterministic",
                records=[],
                evidence=_serialize_evidence_batches(evidence_batches),
                failures=[],
                completed_count=len(evidence_batches),
                failed_count=0,
                cache_hits=0,
                duration_seconds=0,
            ),
            "vlm": PipelineStageOutput(
                stage="vlm",
                records=[],
                evidence=[evidence_item_to_json(ev) for v in vlm_results.values() for ev in v.all_evidence()],
                failures=[],
                completed_count=sum(len(v.available_stages()) for v in vlm_results.values()),
                failed_count=sum(1 for v in vlm_results.values() if v.all_failed),
                cache_hits=vlm_stats["cache_hits"],
                duration_seconds=0,
            ),
        },
        total_records=len(records),
        auto_accept=state_counts.get("auto_accept", 0),
        partial_accept=state_counts.get("partial_accept", 0),
        quarantine=state_counts.get("quarantine", 0),
        hard_reject=state_counts.get("hard_reject", 0),
        unknown=state_counts.get("unknown", 0),
        vlm_metrics=vlm_stats,
    )

    if not dry_run:
        _write_outputs(output_path, record_decisions, evidence_batches, result, vlm_results)

    return result


def _load_records(run_path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    imported_path = run_path / "imported.jsonl"
    if not imported_path.exists():
        return []
    records = read_jsonl(imported_path)
    if max_records is not None:
        records = records[: max(0, int(max_records))]
    return records


def _record_png_path(record: Mapping[str, Any], run_path: Path) -> Path | None:
    raw = str(record.get("final_png_path", "") or record.get("png_path", "") or "")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.is_file() else None
    # Harvest records may store either a run-relative path (``extracted/...``)
    # or a workspace-relative path (``harvest_runs/<run>/extracted/...``).
    # Prefer the stored path exactly before prefixing ``run_path``; otherwise a
    # valid workspace-relative path is accidentally doubled.
    if path.is_file():
        return path
    run_relative = run_path / path
    return run_relative if run_relative.is_file() else None


def compute_record_decision(
    record: Mapping[str, Any],
    config: V3PipelineConfig,
    *,
    calibration: CalibrationArtifact | None = None,
    use_vlm: bool = False,
    run_dir: str | Path = "",
) -> RecordDecision:
    """Compute one v3 RecordDecision from an imported record.

    This is the single, deterministic per-record unit of work shared by the
    in-memory pipeline and the sharded/resumable runner. Given identical inputs
    and config it always produces an identical decision (no VLM in the default
    path).
    """
    sprite_id = str(record.get("sprite_id", ""))
    batch = extract_deterministic_evidence(record, run_dir=run_dir)
    vlm = create_unavailable_cascade(sprite_id, reason="no_vlm_backend" if not use_vlm else "vlm_not_implemented")
    return _fuse_record(
        sprite_id=sprite_id,
        batch=batch,
        vlm=vlm,
        policy=config.policy,
        policy_hash=config.policy.policy_hash(),
        calibration=calibration,
        record=record,
    )


def _fuse_record(
    sprite_id: str,
    batch: DeterministicEvidenceBatch,
    vlm: VlmCascadeResult | None,
    policy: V3LabelingPolicy,
    policy_hash: str,
    calibration: CalibrationArtifact | None,
    record: Mapping[str, Any],
    enrichment_generator: Any = None,
) -> RecordDecision:
    """Fuse all evidence for one sprite into a RecordDecision.

    Per-field calibration support is resolved from the supplied calibration
    artifact using the record's source/profile/domain strata. Without a
    calibration artifact that meets a field's precision target, that field
    abstains — no field is ever auto-accepted on an uncalibrated stratum.
    """

    all_evidence = list(batch.all_evidence())

    # Add VLM evidence if available
    if vlm is not None:
        all_evidence.extend(vlm.all_evidence())

    # Resolve the calibration stratum context for this record once.
    source_id = str(record.get("source_id", ""))
    profile_name, domain_name = _profile_domain(record)

    def support(field: str) -> dict[str, Any] | None:
        if calibration is None:
            return None
        return calibration_support_for_field(
            field,
            source_id,
            profile_name,
            domain_name,
            calibration,
            min_samples=policy.min_calibration_samples,
        )

    def accepts(field: str, support_data: dict[str, Any] | None, target: float) -> bool:
        if not policy.auto_accept_enabled or support_data is None:
            return False
        ci_lower = support_data.get("ci_lower")
        return ci_lower is not None and float(ci_lower) >= target

    # Fuse domain
    domain_input = build_field_fusion_input(
        "domain",
        tuple(all_evidence),
        precision_target=policy.precision_target_category,
        policy_hash=policy_hash,
    )
    domain_result = fuse_field(domain_input, calibration_support=support("domain"))

    # Fuse category. build_field_fusion_input already selects evidence whose
    # target_fields include "category" (or that declare no target). This
    # intentionally includes declarative sheet mappings — the highest-trust
    # deterministic category source — which an earlier hand-rolled filter
    # wrongly excluded. Only evidence that actually proposes a category value
    # contributes to consensus.
    cat_input = build_field_fusion_input(
        "category",
        tuple(all_evidence),
        precision_target=policy.precision_target_category,
        policy_hash=policy_hash,
    )
    cat_result = fuse_field(cat_input, calibration_support=support("category"))

    # Fuse canonical object with hierarchy backoff. Object auto-accept is gated
    # on a calibration lower bound meeting the object precision target.
    obj_support = support("canonical_object")
    obj_result = fuse_hierarchical_object(
        tuple(all_evidence),
        sprite_id=sprite_id,
        policy_hash=policy_hash,
        precision_target=policy.precision_target_canonical_object,
        auto_accept_enabled=accepts("canonical_object", obj_support, policy.precision_target_canonical_object),
        calibration_support=obj_support,
    )

    # Fuse color
    color_input = build_field_fusion_input(
        "color",
        tuple(all_evidence),
        precision_target=policy.precision_target_color,
        policy_hash=policy_hash,
    )
    color_result = fuse_field(color_input, calibration_support=support("color"))

    # Fuse material
    mat_input = build_field_fusion_input(
        "material",
        tuple(all_evidence),
        precision_target=policy.precision_target_material,
        policy_hash=policy_hash,
    )
    mat_result = fuse_field(mat_input, calibration_support=support("material"))

    # Fuse shape
    shape_input = build_field_fusion_input(
        "shape",
        tuple(all_evidence),
        precision_target=policy.precision_target_shape,
        policy_hash=policy_hash,
    )
    shape_result = fuse_field(shape_input, calibration_support=support("shape"))

    # Role is visual/context-derived prefill only.  It remains independently
    # calibrated and therefore abstains by default rather than being promoted
    # from a VLM score.
    role_input = build_field_fusion_input(
        "role", tuple(all_evidence), precision_target=policy.precision_target_tags, policy_hash=policy_hash
    )
    role_result = fuse_field(role_input, calibration_support=support("role"))

    # Build calibrated decisions.  An unaccepted decision must never carry an
    # accepted value; its best guess lives in the independent prefill below.
    field_decisions = {
        "domain": domain_result.decision,
        "category": cat_result.decision,
        "canonical_object": obj_result.decision,
        "color": color_result.decision,
        "material": mat_result.decision,
        "shape": shape_result.decision,
        "role": role_result.decision,
    }
    field_decisions = {
        name: decision if decision.state == "accepted" else replace(decision, accepted_value=None, accepted_values=())
        for name, decision in field_decisions.items()
    }

    prefills, prefill_tags, prefill_metadata = build_prefills(sprite_id, record, all_evidence)
    alias_prefill = prefills["surface_alias"]
    field_decisions["surface_alias"] = FieldDecision(
        sprite_id=sprite_id,
        field_name="surface_alias",
        candidates=(str(alias_prefill.value),) if alias_prefill.value else (),
        n_best_alternatives=tuple((str(a.value), a.score) for a in alias_prefill.alternatives),
        state="abstained",
        evidence_refs=alias_prefill.evidence_refs,
        decision_reason="calibration_insufficient" if alias_prefill.value else "insufficient_evidence",
        policy_hash=policy_hash,
    )

    # Validate impossible combinations only across *accepted* field values, so a
    # clean record whose fields merely abstained can never be hard-rejected on
    # speculative consensus values.
    violation_codes, violation_descs = validate_combinations(
        category=_accepted_value(cat_result.decision),
        object_name=_accepted_value(obj_result.decision),
        material=_accepted_value(mat_result.decision),
        shape=_accepted_value(shape_result.decision),
    )

    if violation_codes:
        field_decisions["canonical_object"] = FieldDecision(
            sprite_id=sprite_id,
            field_name="canonical_object",
            state="rejected",
            contradiction_codes=violation_codes,
            decision_reason="impossible_combination",
            policy_hash=policy_hash,
        )

    # Derive record state
    state = derive_record_state(field_decisions)

    # Collect reason codes
    reason_codes = _collect_reason_codes(field_decisions, state, violation_codes)

    # Descriptions are a strictly downstream, derived artifact.  They do not
    # appear in ``all_evidence`` and cannot change category/object fusion.
    from spritelab.harvest.label_v3.description_enrichment import enrich_description

    description_fields = {
        name: prefill.value
        for name, prefill in prefills.items()
        if prefill.confidence >= 0.55
        and name in {"domain", "category", "canonical_object", "color", "material", "shape"}
    }
    description_fields["style"] = list(prefill_metadata.get("style_tags") or ())
    color_roles = prefill_metadata.get("color_roles") or {}
    for name in ("primary_colors", "secondary_colors", "highlight_colors", "shadow_colors", "outline_color"):
        if color_roles.get(name):
            description_fields[name] = color_roles[name]
    literal = ""
    for item in all_evidence:
        raw = item.proposed_value if isinstance(item.proposed_value, Mapping) else {}
        stage_output = raw.get("stage_output", {}) if isinstance(raw, Mapping) else {}
        if isinstance(stage_output, Mapping) and stage_output.get("literal_description"):
            literal = str(stage_output["literal_description"])
            break
    description_artifact = enrich_description(
        description_fields, literal_description=literal, generator=enrichment_generator
    ).to_dict()
    description_artifact["source"] = "normalized_prefill"
    description_artifact["excluded_from_fusion"] = True
    description_value = description_artifact.get("enriched_description") or description_artifact.get(
        "canonical_description"
    )
    prefills["description"] = FieldPrefill(
        sprite_id=sprite_id,
        field_name="description",
        value=description_value or None,
        normalized_value=description_artifact.get("canonical_description") or None,
        confidence=min((p.confidence for name, p in prefills.items() if name in description_fields), default=0.0),
        reason="deterministic_from_normalized_prefills",
        normalization_actions=("semantic_claim_validation",),
        warnings=tuple(description_artifact.get("unsupported_claims_detected") or ()),
    )
    description_decision = FieldDecision(
        sprite_id=sprite_id,
        field_name="description",
        accepted_value=None,
        candidates=(description_artifact["canonical_description"],)
        if description_artifact["canonical_description"]
        else (),
        state="abstained",
        evidence_refs=(),
        decision_reason="field_not_applicable",
        policy_hash=policy_hash,
    )

    return RecordDecision(
        sprite_id=sprite_id,
        record_state=state,
        reason_codes=tuple(reason_codes),
        reason_details=tuple(violation_descs),
        domain=field_decisions.get("domain", FieldDecision(field_name="domain", sprite_id=sprite_id)),
        category=field_decisions.get("category", FieldDecision(field_name="category", sprite_id=sprite_id)),
        canonical_object=field_decisions.get(
            "canonical_object", FieldDecision(field_name="canonical_object", sprite_id=sprite_id)
        ),
        surface_alias=field_decisions.get(
            "surface_alias", FieldDecision(field_name="surface_alias", sprite_id=sprite_id)
        ),
        color=field_decisions.get("color", FieldDecision(field_name="color", sprite_id=sprite_id)),
        material=field_decisions.get("material", FieldDecision(field_name="material", sprite_id=sprite_id)),
        shape=field_decisions.get("shape", FieldDecision(field_name="shape", sprite_id=sprite_id)),
        role=field_decisions.get("role", FieldDecision(field_name="role", sprite_id=sprite_id)),
        tags=AcceptedTagSet(
            decisions=tuple(
                TagDecision(tag=tag, state="abstained", provenance={"kind": "prefill", "calibrated": False})
                for tag in prefill_tags
            ),
            provenance={"prefill_tags": list(prefill_tags)},
        ),
        description=description_decision,
        description_artifact=description_artifact,
        prefills=prefills,
        prefill_tags=prefill_tags,
        prefill_metadata=prefill_metadata,
        policy_hash=policy_hash,
        lineage={
            "pipeline_version": PIPELINE_VERSION,
            "policy_hash": policy_hash,
            "source_id": source_id,
            "pack_id": str(record.get("source_name", "") or record.get("pack_name", "")),
            "profile": profile_name,
            "domain": domain_name,
        },
    )


def _accepted_value(decision: FieldDecision) -> str:
    """Return the accepted string value of a field, or '' when not accepted."""
    if decision.state != "accepted":
        return ""
    value = decision.accepted_value
    return str(value) if isinstance(value, str) else ""


def _profile_domain(record: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve (profile_name, domain) for calibration-stratum lookup."""
    try:
        from spritelab.harvest.source_profiles import detect_source_profile

        profile = detect_source_profile(record)
        return profile.name, profile.domain
    except Exception:
        return "unknown", "unknown"


def _collect_reason_codes(
    field_decisions: dict[str, FieldDecision],
    state: RecordState,
    violation_codes: tuple[str, ...],
) -> list[str]:
    codes: list[str] = []
    if state == "hard_reject":
        codes.append("irreconcilable_contradiction")
    if state == "quarantine":
        codes.append("unresolved_high_severity_contradiction")
    for fd in field_decisions.values():
        if fd.decision_reason == "insufficient_evidence":
            codes.append("insufficient_evidence")
        if fd.decision_reason == "calibration_insufficient":
            codes.append("calibration_insufficient")
        if fd.contradiction_codes:
            codes.extend(fd.contradiction_codes)
    codes.extend(violation_codes)
    return sorted(set(codes))


def _serialize_evidence_batches(
    batches: list[tuple[dict[str, Any], DeterministicEvidenceBatch]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _record, batch in batches:
        for evidence in batch.all_evidence():
            output.append(
                {
                    "sprite_id": evidence.sprite_id,
                    "evidence": evidence_item_to_json(evidence),
                }
            )
    return output


def _write_outputs(
    output_path: Path,
    decisions: list[RecordDecision],
    evidence_batches: list[tuple[dict[str, Any], DeterministicEvidenceBatch]],
    result: PipelineRunResult,
    vlm_results: Mapping[str, VlmCascadeResult] | None = None,
) -> None:
    records_path = output_path / f"v3{RECORD_OUTPUT_SUFFIX}"
    evidence_path = output_path / f"v3{EVIDENCE_OUTPUT_SUFFIX}"

    records_data = [record_decision_to_json(d) for d in decisions]
    write_jsonl(records_path, records_data)

    evidence_data = _serialize_evidence_batches(evidence_batches)
    for cascade in (vlm_results or {}).values():
        evidence_data.extend(
            {"sprite_id": ev.sprite_id, "evidence": evidence_item_to_json(ev)} for ev in cascade.all_evidence()
        )
    write_jsonl(evidence_path, evidence_data)

    summary_path = output_path / f"v3{SUMMARY_SUFFIX}"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result.summary(), f, indent=2, sort_keys=True)


def format_pipeline_dry_run_report(result: PipelineRunResult) -> str:
    s = result.summary()
    lines = [
        "# Auto-Labeling v3 Dry-Run Report",
        "",
        f"Run dir: {s['run_dir']}",
        f"Output dir: {s['output_dir']}",
        f"Policy hash: {s['policy_hash']}",
        "",
        "## Record States",
        f"- Total records: {s['total_records']}",
        f"- Auto accept: {s['auto_accept']}",
        f"- Partial accept: {s['partial_accept']}",
        f"- Quarantine: {s['quarantine']}",
        f"- Hard reject: {s['hard_reject']}",
        f"- Unknown: {s['unknown']}",
        f"- Acceptance rate: {s['acceptance_rate']:.3f}",
        "",
        f"## Stages executed: {', '.join(s['stages'])}",
        "",
        "**This is a DRY RUN. No records were mutated.**",
    ]
    return "\n".join(lines) + "\n"

"""Fully synthetic, CPU-only end-to-end architecture demonstration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spritelab.hierarchical_labeling.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningPolicy,
    generate_review_round,
)
from spritelab.hierarchical_labeling.calibration import (
    SYNTHETIC_ORACLE_SCOPE,
    CalibrationExample,
    evaluate_holdout,
    fit_calibration,
)
from spritelab.hierarchical_labeling.cohort import (
    CohortCandidate,
    CohortSelectionPolicy,
    select_reference_cohort,
)
from spritelab.hierarchical_labeling.contracts import (
    LabelEvidenceBundle,
    MetadataEvidence,
    RetrievalEvidence,
    SyntheticOracleLabel,
)
from spritelab.hierarchical_labeling.decision import decide_hierarchical_label
from spritelab.hierarchical_labeling.json_utils import content_identity
from spritelab.hierarchical_labeling.renders import FAST_LOCAL_POLICY, build_render_bundle
from spritelab.hierarchical_labeling.reporting import build_report_data, write_offline_report
from spritelab.hierarchical_labeling.retrieval import (
    EmbeddingSample,
    ExactRetrievalIndex,
    RetrievalIndexRecord,
    StructuralEmbeddingBackend,
)
from spritelab.hierarchical_labeling.review import (
    GENESIS_EVENT_HASH,
    append_review_event,
    create_review_event,
    synthetic_oracle_reference_label,
)
from spritelab.hierarchical_labeling.semantic import (
    DESCRIPTION_SCHEMA_VERSION,
    HYPOTHESIS_SCHEMA_VERSION,
    parse_semantic_hypotheses,
    parse_visual_description,
    prompt_identity,
    structured_attributes,
    visual_description_prompt,
)
from spritelab.hierarchical_labeling.supervision import export_supervision
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph, load_default_taxonomy
from spritelab.hierarchical_labeling.technical import extract_technical_evidence
from spritelab.v3.run_state import atomic_write_json

SYNTHETIC_DEMO_SCHEMA = "spritelab.labeling.synthetic-end-to-end.v1"


@dataclass
class _SyntheticRunRecorder:
    events: list[dict[str, Any]]

    def record(self, event_type: str, count: int = 1, **properties: Any) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "count": count,
                **dict(sorted(properties.items())),
            }
        )

    def summary(self) -> dict[str, int]:
        def total(event_type: str, **matches: Any) -> int:
            return sum(
                int(event["count"])
                for event in self.events
                if event["event_type"] == event_type
                and all(event.get(name) == value for name, value in matches.items())
            )

        return {
            "real_provider_calls": total("provider_call", real=True),
            "fake_provider_calls": total("provider_call", real=False),
            "hosted_calls": total("provider_call", hosted=True),
            "network_calls": total("network_call"),
            "gpu_initializations": total("gpu_initialization"),
            "training_runs": total("training_run"),
            "production_freezes": total("production_freeze"),
            "synthetic_oracle_labels": total("oracle_label"),
            "human_review_events": total("human_review_event"),
            "human_review_truth_events": total("human_review_event", used_as_truth=True),
            "human_labels_auto_created": total("human_label_auto_created"),
        }

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "spritelab.labeling.synthetic-runtime-events.v1",
            "events": self.events,
            "summary": self.summary(),
        }
        payload["ledger_identity"] = content_identity(payload["schema_version"], payload)
        return payload


@dataclass(frozen=True)
class _Spec:
    record_identity: str
    kind: str
    taxonomy_path: tuple[str, ...]
    partition: str | None
    cluster_identity: str
    duplicate_cluster_identity: str
    near_duplicate_cluster_identity: str | None
    source_identity: str
    style_identity: str
    ambiguous_leaf: bool = False
    metadata_conflict: bool = False
    sheet_derived: bool = False
    animation_frame: bool = False
    novel: bool = False
    variant: int = 0


def run_synthetic_demo(output_root: str | Path) -> dict[str, Any]:
    """Exercise the architecture without network, hosted provider, GPU, or training."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    recorder = _SyntheticRunRecorder([])
    graph = load_default_taxonomy()
    specs = _specs(graph)
    image_paths = {spec.record_identity: _write_sprite(root / "corpus", spec) for spec in specs}
    technical = {
        spec.record_identity: extract_technical_evidence(
            image_paths[spec.record_identity],
            record_identity=spec.record_identity,
            duplicate_cluster_identity=spec.duplicate_cluster_identity,
            near_duplicate_cluster_identity=spec.near_duplicate_cluster_identity,
        )
        for spec in specs
    }
    renders = {
        spec.record_identity: build_render_bundle(
            image_paths[spec.record_identity],
            technical[spec.record_identity],
            root / "renders" / spec.record_identity,
            policy=FAST_LOCAL_POLICY,
            scale=4,
        )
        for spec in specs
    }
    description_prompt_identity = prompt_identity("synthetic-description-prompt-v1", visual_description_prompt())
    descriptions = {
        spec.record_identity: parse_visual_description(
            _description_value(spec),
            record_identity=spec.record_identity,
            image_identity=technical[spec.record_identity].image_identity,
            render_bundle_identity=renders[spec.record_identity].identity,
            provider_identity="synthetic-fixture-provider",
            model_identity="synthetic-fixture-model-v1",
            prompt_identity_value=description_prompt_identity,
        )
        for spec in specs
    }
    recorder.record("provider_call", len(specs), real=False, hosted=False, stage="description")
    hypotheses = {
        spec.record_identity: parse_semantic_hypotheses(
            _hypothesis_value(spec, graph),
            record_identity=spec.record_identity,
            graph=graph,
            description=descriptions[spec.record_identity],
            provider_identity="synthetic-fixture-provider",
            model_identity="synthetic-fixture-model-v1",
            prompt_identity_value=content_identity("synthetic-hypothesis-prompt-v1", {"taxonomy": graph.identity}),
            render_bundle_identity=renders[spec.record_identity].identity,
        )
        for spec in specs
    }
    recorder.record("provider_call", len(specs), real=False, hosted=False, stage="hypothesis")
    base_bundles = {
        spec.record_identity: LabelEvidenceBundle(
            spec.record_identity,
            technical[spec.record_identity].image_identity,
            graph.identity,
            technical[spec.record_identity],
            descriptions[spec.record_identity],
            structured_attributes(descriptions[spec.record_identity]),
            (hypotheses[spec.record_identity],),
            metadata=(
                MetadataEvidence(
                    spec.record_identity,
                    content_identity("synthetic-metadata-v1", {"record": spec.record_identity}),
                    (("category", "resource"),),
                    False,
                )
                if spec.metadata_conflict
                else None
            ),
        )
        for spec in specs
    }
    samples = tuple(
        EmbeddingSample(
            spec.record_identity,
            technical[spec.record_identity].image_identity,
            technical[spec.record_identity],
            renders[spec.record_identity].views,
        )
        for spec in specs
    )
    backend = StructuralEmbeddingBackend()
    vectors = (*backend.embed_images(samples), *backend.embed_views(samples))
    by_record: dict[str, list[Any]] = defaultdict(list)
    for vector in vectors:
        by_record[vector.record_identity].append(vector)
    index_records = []
    for spec in specs:
        index_records.append(
            RetrievalIndexRecord.from_vectors(
                by_record[spec.record_identity],
                taxonomy_identity=graph.identity,
                proposal_taxonomy_path=spec.taxonomy_path,
                metadata={"synthetic_source_identity": spec.source_identity},
            )
        )
    index = ExactRetrievalIndex(
        index_records,
        backend_identity=backend.cache_identity,
        fusion_weights={"technical_feature": 0.4, "alpha_silhouette": 0.4, "palette_composition": 0.2},
    )
    bundles: dict[str, LabelEvidenceBundle] = {}
    for spec in specs:
        query = {vector.representation: vector.vector for vector in by_record[spec.record_identity]}
        neighbors = index.nearest_neighbors(query, k=8, exclude_record_identity=spec.record_identity)
        novelty = 0.95 if spec.novel else index.novelty_score(query, exclude_record_identity=spec.record_identity)
        retrieval = RetrievalEvidence(
            spec.record_identity,
            technical[spec.record_identity].image_identity,
            backend.cache_identity,
            index.identity,
            graph.identity,
            None,
            None,
            neighbors,
            tuple(index.fusion_weights.items()),
            novelty,
        )
        base = base_bundles[spec.record_identity]
        bundles[spec.record_identity] = LabelEvidenceBundle(
            base.record_identity,
            base.image_identity,
            base.taxonomy_identity,
            base.technical,
            base.visual_description,
            base.visual_attributes,
            base.taxonomy_hypotheses,
            retrieval,
            base.metadata,
        )
    oracle_set_identity = content_identity(
        "spritelab-synthetic-oracle-set-v1",
        [spec.record_identity for spec in specs if spec.partition is not None],
    )
    oracle_cohort_identity = content_identity(
        "spritelab-synthetic-oracle-cohort-v1",
        {
            "partitions": {
                partition: [spec.record_identity for spec in specs if spec.partition == partition]
                for partition in ("reference", "calibration", "holdout")
            }
        },
    )
    oracle_labels = {
        spec.record_identity: _oracle_label(
            spec,
            graph,
            bundles[spec.record_identity],
            oracle_set_identity=oracle_set_identity,
            oracle_cohort_identity=oracle_cohort_identity,
        )
        for spec in specs
        if spec.partition is not None
    }
    recorder.record("oracle_label", len(oracle_labels), used_as_truth=True, scope="synthetic_fixture")
    calibration_examples = [
        CalibrationExample(
            spec.record_identity,
            "object",
            0.55 + 0.02 * (index % 10),
            bundles[spec.record_identity].identity,
            spec.source_identity,
            spec.cluster_identity,
            oracle_labels[spec.record_identity],
            spec.duplicate_cluster_identity,
            spec.near_duplicate_cluster_identity,
            f"{spec.partition}-group-{spec.record_identity}",
        )
        for index, spec in enumerate(item for item in specs if item.partition == "calibration")
    ]
    calibration = fit_calibration(
        calibration_examples,
        graph,
        target_precision=0.95,
        minimum_global_samples=20,
        minimum_class_samples=10,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    holdout_examples = [
        CalibrationExample(
            spec.record_identity,
            "object",
            0.8,
            bundles[spec.record_identity].identity,
            spec.source_identity,
            spec.cluster_identity,
            oracle_labels[spec.record_identity],
            spec.duplicate_cluster_identity,
            spec.near_duplicate_cluster_identity,
            f"{spec.partition}-group-{spec.record_identity}",
        )
        for spec in specs
        if spec.partition == "holdout"
    ]
    calibration, holdout_result = evaluate_holdout(
        calibration,
        holdout_examples,
        graph,
        minimum_holdout_samples=5,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    decisions = {
        spec.record_identity: decide_hierarchical_label(bundles[spec.record_identity], graph, calibration)
        for spec in specs
    }
    exports = {
        spec.record_identity: export_supervision(bundles[spec.record_identity], decisions[spec.record_identity], graph)
        for spec in specs
    }
    cohort = select_reference_cohort(
        [_cohort_candidate(spec, technical[spec.record_identity].image_identity) for spec in specs],
        dataset_identity=content_identity(
            "synthetic-dataset-v1", [value.image_identity for value in technical.values()]
        ),
        embedding_identity=backend.cache_identity,
        clustering_identity=index.identity,
        policy=CohortSelectionPolicy(
            target_size=12,
            seed=20260715,
            reference_fraction=1.0,
            calibration_fraction=0.0,
            holdout_fraction=0.0,
        ),
    )
    reference_spec = next(spec for spec in specs if spec.partition == "reference")
    review_event = create_review_event(
        bundles[reference_spec.record_identity],
        graph,
        action="accept_suggested_path",
        reviewer_identity="synthetic-human-fixture",
        partition="reference",
        previous_event_hash=GENESIS_EVENT_HASH,
        selected_node=reference_spec.taxonomy_path[-1],
        render_identities=(renders[reference_spec.record_identity].identity,),
        review_confidence=1.0,
        timestamp="2026-07-15T00:00:00+00:00",
        submission_token="synthetic-review-fixture-1",
    )
    append_review_event(root / "human_review_events.jsonl", review_event)
    recorder.record("human_review_event", 1, used_as_truth=False, fixture=True)
    active_round = generate_review_round(
        [_active_candidate(spec, decisions[spec.record_identity]) for spec in specs],
        dataset_identity=cohort["dataset_identity"],
        reference_set_identity=cohort["cohort_identity"],
        embedding_identity=backend.cache_identity,
        calibration_identity=calibration.identity,
        round_number=1,
        policy=ActiveLearningPolicy(review_budget=8, seed=20260715),
    )
    report_records = [_report_record(spec, decisions[spec.record_identity]) for spec in specs]
    oracle_rows = [
        {
            "record_identity": spec.record_identity,
            "evidence_bundle_identity": bundles[spec.record_identity].identity,
            "predicted_path": list(decisions[spec.record_identity].taxonomy_path),
            "calibrated_probability": 0.8,
            "source_identity": spec.source_identity,
            "cluster_identity": spec.cluster_identity,
            "leakage_group_identity": f"{spec.partition}-group-{spec.record_identity}",
            "human_reference": oracle_labels[spec.record_identity],
        }
        for spec in specs
        if spec.partition == "holdout"
    ]
    report = build_report_data(
        report_records,
        graph,
        synthetic_oracle_rows=oracle_rows,
        calibration_state=calibration.state.value,
        calibration_truth_scope=SYNTHETIC_ORACLE_SCOPE,
        operational={
            "provider_usage": {"synthetic_fixture_provider": len(specs)},
            "provider_failures": {},
            "cache": {"hits": 0, "lookups": 0, "rate": None},
            "hosted_call_count": 0,
            "cost_estimate": "unknown",
            "human_review": {"completed": 1, "pending": len(active_round["selected"]), "throughput_per_hour": None},
            "taxonomy_gaps": [spec.record_identity for spec in specs if spec.novel],
            "retrieval_neighbor_examples": [],
            "cluster_medoids": list(index.cluster_medoids(index.cluster_assignments()).values()),
            "sample_gallery": [spec.record_identity for spec in specs[:8]],
        },
    )
    report_json, report_html = write_offline_report(report, root / "report")
    runtime_ledger = recorder.payload()
    atomic_write_json(root / "synthetic_runtime_events.json", runtime_ledger)
    proposed_leaf = sum(bool(spec.taxonomy_path) for spec in specs)
    accepted_leaf = sum(
        decision.deepest_accepted_node == spec.taxonomy_path[-1]
        for spec in specs
        for decision in (decisions[spec.record_identity],)
        if spec.taxonomy_path
    )
    accepted_parent = sum(
        bool(decision.taxonomy_path) and decision.deepest_accepted_node != spec.taxonomy_path[-1]
        for spec in specs
        for decision in (decisions[spec.record_identity],)
        if spec.taxonomy_path
    )
    conflicts = sum(bool(decision.conflicts) for decision in decisions.values())
    result: dict[str, Any] = {
        "schema_version": SYNTHETIC_DEMO_SCHEMA,
        "corpus": {
            "records": len(specs),
            "clear_broad_categories": True,
            "ambiguous_leaves": sum(spec.ambiguous_leaf for spec in specs),
            "visually_similar_categories": True,
            "metadata_conflicts": sum(spec.metadata_conflict for spec in specs),
            "duplicate_clusters": len(specs) - len({spec.duplicate_cluster_identity for spec in specs}),
            "near_duplicates": sum(spec.near_duplicate_cluster_identity is not None for spec in specs),
            "sheet_derived": sum(spec.sheet_derived for spec in specs),
            "rare_clusters": sum(spec.novel for spec in specs),
            "novel_outliers": sum(spec.novel for spec in specs),
            "animation_frames": sum(spec.animation_frame for spec in specs),
        },
        "stages_exercised": [
            "technical evidence",
            "multi-view renders",
            "visual descriptions",
            "taxonomy hypotheses",
            "embeddings",
            "retrieval",
            "reference cohort",
            "human review fixture",
            "calibration",
            "hierarchical decisions",
            "active-learning queue",
            "supervision export",
            "offline report",
        ],
        "demonstrations": {
            "proposed_leaf_records": proposed_leaf,
            "accepted_leaf_records": accepted_leaf,
            "accepted_safer_parent_records": accepted_parent,
            "hierarchy_increases_broad_accepted_coverage": accepted_parent > 0,
            "uncertain_leaves_abstain": all(
                decisions[spec.record_identity].deepest_accepted_node != spec.taxonomy_path[-1]
                for spec in specs
                if spec.ambiguous_leaf
            ),
            "metadata_conflicts_visible": conflicts > 0,
            "retrieval_reviewed_neighbors_authoritative_only": all(
                not neighbor.verified_taxonomy_path or neighbor.review_status == "reviewed"
                for bundle in bundles.values()
                for neighbor in bundle.retrieval.neighbors
            ),
            "calibration_truth_source": holdout_result.truth_source,
            "calibration_fit_records": len(calibration.fit_record_identities),
            "holdout_sample_size": holdout_result.sample_size,
            "held_out_precision": holdout_result.precision,
            "held_out_coverage": holdout_result.coverage,
            "model_model_agreement_is_truth": False,
            "image_only_eligibility_preserved": True,
            "report_precision_graph_available": report["claims"]["precision_graph_available"],
            "synthetic_oracle_precision_available": report["synthetic_oracle_metrics"] is not None,
        },
        "artifacts": {
            "review_log": "human_review_events.jsonl",
            "report_json": str(report_json.relative_to(root).as_posix()),
            "report_html": str(report_html.relative_to(root).as_posix()),
            "cohort_identity": cohort["cohort_identity"],
            "active_learning_round_identity": active_round["round_identity"],
            "calibration_identity": calibration.identity,
            "retrieval_index_identity": index.identity,
            "supervision_exports": len(exports),
            "runtime_event_ledger": "synthetic_runtime_events.json",
            "runtime_event_ledger_identity": runtime_ledger["ledger_identity"],
        },
        "safety": runtime_ledger["summary"],
        "limitations": [
            "All images, descriptions, hypotheses, and oracle labels are synthetic fixtures.",
            "The persisted human review fixture is unrelated to oracle truth and is never used for metrics.",
            "Synthetic precision and coverage do not generalize to real sprite data.",
            "No conditioned production authorization is created by this demonstration.",
        ],
        "production_authorization": False,
    }
    result["result_identity"] = content_identity(SYNTHETIC_DEMO_SCHEMA, result)
    atomic_write_json(root / "synthetic_end_to_end_results.json", result)
    return result


def _specs(graph: TaxonomyGraph) -> tuple[_Spec, ...]:
    kinds = (
        ("bottle", "bottle"),
        ("chest", "chest"),
        ("sword", "sword"),
        ("axe", "axe"),
        ("character", "character"),
    )
    values: list[_Spec] = []
    partitions = ["reference"] * 5 + ["calibration"] * 20 + ["holdout"] * 5
    for index, partition in enumerate(partitions):
        kind, node = kinds[index % len(kinds)]
        values.append(
            _Spec(
                f"synthetic-{index:03d}",
                kind,
                graph.path(node),
                partition,
                f"{partition}-cluster-{kind}",
                f"duplicate-{index:03d}",
                None,
                f"{partition}-source-{index % 4}",
                f"style-{index % 3}",
                variant=index,
            )
        )
    extras = (
        _Spec(
            "synthetic-ambiguous-bottle",
            "bottle",
            graph.path("bottle"),
            None,
            "cluster-bottle",
            "duplicate-ambiguous-bottle",
            "near-bottle",
            "source-extra",
            "style-muted",
            ambiguous_leaf=True,
            variant=41,
        ),
        _Spec(
            "synthetic-metadata-conflict",
            "sword",
            graph.path("sword"),
            None,
            "cluster-sword",
            "duplicate-metadata-conflict",
            None,
            "source-extra",
            "style-bright",
            metadata_conflict=True,
            variant=42,
        ),
        _Spec(
            "synthetic-duplicate-a",
            "chest",
            graph.path("chest"),
            None,
            "cluster-chest",
            "duplicate-shared-chest",
            None,
            "source-duplicate",
            "style-flat",
            variant=0,
        ),
        _Spec(
            "synthetic-duplicate-b",
            "chest",
            graph.path("chest"),
            None,
            "cluster-chest",
            "duplicate-shared-chest",
            None,
            "source-duplicate",
            "style-flat",
            variant=0,
        ),
        _Spec(
            "synthetic-sheet-tile",
            "tile",
            graph.path("tile"),
            None,
            "cluster-tile",
            "duplicate-sheet-tile",
            None,
            "source-sheet",
            "style-grid",
            sheet_derived=True,
            variant=44,
        ),
        _Spec(
            "synthetic-animation-frame-a",
            "effect",
            graph.path("effect"),
            None,
            "cluster-animation",
            "duplicate-animation-a",
            "near-animation",
            "source-animation",
            "style-effect",
            animation_frame=True,
            variant=45,
        ),
        _Spec(
            "synthetic-animation-frame-b",
            "effect",
            graph.path("effect"),
            None,
            "cluster-animation",
            "duplicate-animation-b",
            "near-animation",
            "source-animation",
            "style-effect",
            animation_frame=True,
            variant=46,
        ),
        _Spec(
            "synthetic-novel-outlier",
            "novel",
            graph.path("object"),
            None,
            "cluster-rare-novel",
            "duplicate-novel",
            None,
            "source-rare",
            "style-noisy",
            novel=True,
            variant=47,
        ),
    )
    return (*values, *extras)


def _write_sprite(directory: Path, spec: _Spec) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    accent = (40 + (spec.variant * 17) % 120, 80 + (spec.variant * 11) % 120, 180, 255)
    outline = (28, 34, 48, 255)
    if spec.kind == "bottle":
        draw.rectangle((6, 2, 9, 5), fill=outline)
        draw.rectangle((4, 5, 11, 13), fill=outline)
        draw.rectangle((5, 6, 10, 12), fill=accent)
    elif spec.kind == "chest":
        draw.rectangle((2, 5, 13, 12), fill=outline)
        draw.rectangle((3, 6, 12, 8), fill=accent)
        draw.rectangle((3, 10, 12, 11), fill=accent)
        draw.point((7, 9), fill=(240, 190, 40, 255))
    elif spec.kind in {"sword", "axe"}:
        draw.rectangle((7, 2, 8, 12), fill=(190, 205, 220, 255))
        draw.rectangle((5, 11, 10, 12), fill=outline)
        draw.rectangle((7, 13, 8, 15), fill=(120, 70, 35, 255))
        if spec.kind == "axe":
            draw.rectangle((4, 3, 8, 6), fill=accent)
    elif spec.kind == "character":
        draw.ellipse((5, 1, 10, 6), fill=accent)
        draw.rectangle((4, 6, 11, 13), fill=outline)
        draw.rectangle((2, 7, 13, 9), fill=accent)
    elif spec.kind == "tile":
        draw.rectangle((1, 1, 14, 14), fill=outline)
        for y in range(2, 14, 4):
            for x in range(2, 14, 4):
                draw.rectangle((x, y, x + 2, y + 2), fill=accent)
    elif spec.kind == "effect":
        for offset in range(1, 7):
            draw.point((8 + (offset % 3) - 1, offset * 2), fill=accent)
            draw.point((offset * 2, 8 + (offset % 3) - 1), fill=(230, 210, 70, 255))
    else:
        for index in range(28):
            x = (index * 7 + spec.variant) % 16
            y = (index * 11 + spec.variant * 3) % 16
            draw.point((x, y), fill=(accent[0], (accent[1] + index * 9) % 255, accent[2], 255))
    # Preserve intentional exact duplicates (same kind + variant), while making
    # partitioned oracle fixtures visually distinct at the decoded-image boundary.
    draw.point((15, 15), fill=accent)
    path = directory / f"{spec.record_identity}.png"
    image.save(path)
    return path


def _description_value(spec: _Spec) -> dict[str, Any]:
    observation = {
        "bottle": "upright blue container-like form with a narrow neck",
        "chest": "rectangular lidded box-like form with a central latch",
        "sword": "long narrow blade-like form with a cross guard",
        "axe": "long handled form with a broad side head",
        "character": "single upright figure-like silhouette with head and limbs",
        "tile": "square repeating patterned surface",
        "effect": "small radiating cluster of bright pixels",
        "novel": "irregular disconnected multicolor pixel marks",
    }[spec.kind]
    interpretation = {
        "bottle": ["bottle", "decorative container"],
        "chest": ["chest", "storage box"],
        "sword": ["sword", "narrow tool"],
        "axe": ["axe", "handled tool"],
        "character": ["character", "humanoid figure"],
        "tile": ["tile", "pattern swatch"],
        "effect": ["effect", "spark"],
        "novel": ["abstract effect", "unresolved marks"],
    }[spec.kind]
    ambiguities = ["leaf identity is visually ambiguous"] if spec.ambiguous_leaf or spec.novel else []
    return {
        "schema_version": DESCRIPTION_SCHEMA_VERSION,
        "visible_observations": [observation],
        "visible_entities": ["one visible form"],
        "entity_count": 1,
        "shape_terms": ["compact" if spec.kind not in {"sword", "axe"} else "elongated"],
        "visual_forms": [f"{spec.kind}-like" if spec.kind != "novel" else "irregular"],
        "dominant_colors": ["blue"],
        "secondary_colors": ["dark outline"],
        "material_like_cues": ["metal-like"] if spec.kind in {"sword", "axe"} else [],
        "orientation": ["upright"],
        "symmetry": ["approximately vertical"],
        "visible_parts": ["outline", "interior color region"],
        "possible_interpretations": interpretation,
        "ambiguities": ambiguities,
        "resolution_limitations": ["low pixel resolution"],
        "scene_or_icon_context": "isolated sprite icon",
        "caption_short": observation,
        "caption_detailed": f"An isolated low-resolution sprite showing {observation}.",
    }


def _hypothesis_value(spec: _Spec, graph: TaxonomyGraph) -> dict[str, Any]:
    if spec.novel:
        return {
            "schema_version": HYPOTHESIS_SCHEMA_VERSION,
            "no_safe_hypothesis": True,
            "reason": "synthetic novel outlier has no safe specific hypothesis",
            "hypotheses": [],
        }
    items = []
    for node_id in spec.taxonomy_path:
        deepest = node_id == spec.taxonomy_path[-1]
        items.append(
            {
                "node_id": node_id,
                "depth": graph.depth(node_id),
                "rank": 1,
                "raw_model_confidence": 0.25 if deepest and spec.ambiguous_leaf else 0.96 if not deepest else 0.9,
                "evidence_citations": ["visible observation fixture"],
                "contradicting_observations": ["leaf ambiguity"] if deepest and spec.ambiguous_leaf else [],
                "abstention_recommended": deepest and spec.ambiguous_leaf,
            }
        )
    alternative = {
        "bottle": "chest",
        "chest": "bottle",
        "sword": "axe",
        "axe": "sword",
        "character": "creature",
        "tile": "effect",
        "effect": "tile",
    }.get(spec.kind)
    if alternative and graph.depth(alternative) == graph.depth(spec.taxonomy_path[-1]):
        items.append(
            {
                "node_id": alternative,
                "depth": graph.depth(alternative),
                "rank": 2,
                "raw_model_confidence": 0.35,
                "evidence_citations": ["secondary visual interpretation"],
                "contradicting_observations": ["weaker silhouette match"],
                "abstention_recommended": True,
            }
        )
    return {
        "schema_version": HYPOTHESIS_SCHEMA_VERSION,
        "no_safe_hypothesis": False,
        "reason": "synthetic strict fixture",
        "hypotheses": items,
    }


def _oracle_label(
    spec: _Spec,
    graph: TaxonomyGraph,
    bundle: LabelEvidenceBundle,
    *,
    oracle_set_identity: str,
    oracle_cohort_identity: str,
) -> SyntheticOracleLabel:
    assert spec.partition is not None
    return synthetic_oracle_reference_label(
        record_identity=spec.record_identity,
        taxonomy_identity=graph.identity,
        taxonomy_path=spec.taxonomy_path,
        deepest_accepted_node=spec.taxonomy_path[-1],
        explicit_abstentions=(),
        partition=spec.partition,
        oracle_set_identity=oracle_set_identity,
        image_identity=bundle.image_identity,
        evidence_bundle_identity=bundle.identity,
        cohort_identity=oracle_cohort_identity,
        source_identity=spec.source_identity,
        cluster_identity=spec.cluster_identity,
        leakage_group_identity=f"{spec.partition}-group-{spec.record_identity}",
        duplicate_cluster_identity=spec.duplicate_cluster_identity,
        near_duplicate_cluster_identity=spec.near_duplicate_cluster_identity,
    )


def _cohort_candidate(spec: _Spec, image_identity: str) -> CohortCandidate:
    return CohortCandidate(
        spec.record_identity,
        image_identity,
        spec.cluster_identity,
        spec.duplicate_cluster_identity,
        spec.near_duplicate_cluster_identity,
        spec.source_identity,
        spec.style_identity,
        "16x16",
        8 if spec.cluster_identity in {"cluster-bottle", "cluster-chest", "cluster-sword"} else 2,
        spec.record_identity.endswith("000") or spec.record_identity.endswith("005"),
        0.95 if spec.novel else 0.2,
        0.9 if spec.ambiguous_leaf else 0.25,
        spec.metadata_conflict,
        0.9 if spec.metadata_conflict or spec.ambiguous_leaf else 0.2,
        spec.sheet_derived,
        spec.animation_frame,
        spec.novel,
        True,
        True,
    )


def _active_candidate(spec: _Spec, decision: Any) -> ActiveLearningCandidate:
    retrieval_novelty = 0.95 if spec.novel else 0.25
    return ActiveLearningCandidate(
        spec.record_identity,
        content_identity("synthetic-active-image-v1", {"record": spec.record_identity}),
        spec.cluster_identity,
        spec.duplicate_cluster_identity,
        spec.near_duplicate_cluster_identity,
        0.9 if not decision.taxonomy_path else 0.3,
        0.8,
        retrieval_novelty,
        0.8 if decision.conflicts else 0.2,
        spec.metadata_conflict,
        0.8 if spec.ambiguous_leaf else 0.3,
        spec.novel,
        8 if spec.cluster_identity in {"cluster-bottle", "cluster-chest", "cluster-sword"} else 2,
        spec.novel,
        0.2,
        0.1,
        spec.partition is not None,
        True,
        True,
    )


def _report_record(spec: _Spec, decision: Any) -> dict[str, Any]:
    return {
        "record_identity": spec.record_identity,
        "accepted_path": list(decision.taxonomy_path),
        "abstained": not bool(decision.taxonomy_path),
        "source_identity": spec.source_identity,
        "cluster_identity": spec.cluster_identity,
        "novelty": 0.95 if spec.novel else 0.2,
        "visual_metadata_conflict": spec.metadata_conflict,
        "description_complete": True,
        "embedding_complete": True,
        "retrieval_complete": True,
        "decision_complete": True,
        "review_pending": spec.partition is None,
    }

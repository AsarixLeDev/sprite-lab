from __future__ import annotations

import copy
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from hierarchical_labeling_support import write_sprite
from spritelab.hierarchical_labeling.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningPolicy,
    generate_review_round,
    stopping_reason,
)
from spritelab.hierarchical_labeling.cache import (
    NO_CONTEXT_IDENTITY,
    TECHNICAL_SCHEMA_IDENTITY,
    HierarchicalLabelingCache,
    LabelingCacheIdentity,
    LabelingRunStore,
)
from spritelab.hierarchical_labeling.cascade import (
    CascadeBudget,
    CascadePolicy,
    CascadeProfile,
    HostedCostRate,
    LabelingCascade,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.pilot import PilotCandidate, plan_pilot
from spritelab.hierarchical_labeling.product import prepare_configured_labeling, product_status
from spritelab.hierarchical_labeling.reporting import build_report_data, write_offline_report
from spritelab.hierarchical_labeling.taxonomy import load_default_taxonomy
from spritelab.product_core import ProjectContext
from spritelab.product_features.dataset.plugin import create_plugin
from spritelab.product_features.providers.adapters import DeterministicMockVisionProvider
from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.contracts import ImageInput, PrivacyClass, PrivacyPolicy
from spritelab.product_features.providers.discovery import VisionProviderRegistry
from spritelab.product_features.providers.errors import ProviderCancelledError, ProviderError
from spritelab.product_features.providers.hub import VisionProviderHub
from spritelab.product_web.app import create_app
from spritelab.v3.config import DEFAULT_CONFIG, ConfigError, ProjectConfig


class HostedMockProvider(DeterministicMockVisionProvider):
    provider_id = "mock.hosted"
    display_name = "Hosted mock"
    privacy_class = PrivacyClass.HOSTED


def _cascade(provider, *, privacy=PrivacyPolicy.LOCAL_ONLY, clock=None, batch_size=4):
    selected_clock = clock or time.monotonic
    settings = ProviderSettings(
        privacy_policy=privacy,
        maximum_retries=2,
        backoff_seconds=0.0,
        batch_size=batch_size,
    )
    hub = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,)),
        settings=settings,
        sleep=lambda _seconds: None,
        clock=selected_clock,
    )
    return LabelingCascade(
        hub,
        {"local": provider, "primary": provider, "verifier": provider},
        clock=selected_clock,
    )


def _images():
    return (ImageInput("r1", b"image-one"), ImageInput("r2", b"image-two"))


def _prompts():
    return {"description": "describe", "hypotheses": "hypothesize", "verification": "verify"}


def test_cascade_fast_local_balanced_escalation_and_high_quality() -> None:
    fast_provider = DeterministicMockVisionProvider()
    fast = _cascade(fast_provider).run(
        _images(),
        policy=CascadePolicy(CascadeProfile.FAST_LOCAL, CascadeBudget(maximum_requests=10)),
        prompts=_prompts(),
    )
    assert fast.status == "completed" and [stage.stage for stage in fast.stages] == ["description", "hypotheses"]
    balanced_provider = DeterministicMockVisionProvider()
    balanced = _cascade(balanced_provider).run(
        _images(),
        policy=CascadePolicy(CascadeProfile.BALANCED, CascadeBudget(maximum_requests=10)),
        prompts=_prompts(),
        uncertain_record_ids=("r2",),
    )
    assert [stage.requested_record_ids for stage in balanced.stages][-1] == ("r2",)
    quality_provider = DeterministicMockVisionProvider()
    quality = _cascade(quality_provider).run(
        _images(),
        policy=CascadePolicy(CascadeProfile.HIGH_QUALITY, CascadeBudget(maximum_requests=10)),
        prompts=_prompts(),
    )
    assert len(quality.stages) == 3


def test_cascade_local_only_and_hosted_confirmation_policy() -> None:
    hosted = HostedMockProvider()
    local_only = _cascade(hosted, privacy=PrivacyPolicy.LOCAL_ONLY).run(
        _images(),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(maximum_hosted_records=2, maximum_requests=10),
        ),
        prompts=_prompts(),
    )
    assert local_only.status == "partial" and local_only.stop_reason == "provider_policy_blocked"
    ask_provider = HostedMockProvider()
    ask = _cascade(ask_provider, privacy=PrivacyPolicy.ASK_BEFORE_HOSTED).run(
        _images(),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(maximum_hosted_records=2, maximum_requests=10),
        ),
        prompts=_prompts(),
        confirm_hosted=lambda _message: False,
    )
    assert ask.stop_reason == "provider_hosted_confirmation_required"
    assert hosted.call_count == local_only.hosted_request_count == 0
    assert ask_provider.call_count == ask.hosted_request_count == 0


def test_cascade_budget_exhaustion_unknown_cost_and_partial_success() -> None:
    exhausted = _cascade(DeterministicMockVisionProvider()).run(
        _images(),
        policy=CascadePolicy(
            CascadeProfile.FAST_LOCAL,
            CascadeBudget(maximum_requests=1, maximum_retries=0),
        ),
        prompts=_prompts(),
    )
    assert exhausted.status == "partial" and exhausted.stop_reason == "labeling_request_budget_exhausted"
    unknown_cost = _cascade(HostedMockProvider(), privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        _images(),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(maximum_hosted_records=2, maximum_requests=10, maximum_estimated_cost=1.0),
        ),
        prompts=_prompts(),
    )
    assert unknown_cost.stop_reason == "labeling_cost_unknown_budget_unenforceable"
    failure = ProviderError("synthetic_item_failure", "fake", retryable=False)
    partial_provider = DeterministicMockVisionProvider(responses={"r2": failure})
    partial = _cascade(partial_provider).run(
        _images(),
        policy=CascadePolicy(CascadeProfile.FAST_LOCAL, CascadeBudget(maximum_requests=10)),
        prompts=_prompts(),
    )
    assert partial.status == "partial" and partial.exception_queue == ("r2",)
    assert any(result.image_id == "r1" and result.ok for stage in partial.stages for result in stage.results)


def test_cascade_retry_and_cancellation_no_real_calls() -> None:
    retryable = ProviderError("synthetic_retry", "fake", retryable=True)
    provider = DeterministicMockVisionProvider(failures=(retryable,))
    retried = _cascade(provider).run(
        _images(),
        policy=CascadePolicy(
            CascadeProfile.FAST_LOCAL,
            CascadeBudget(maximum_requests=10, maximum_retries=1),
        ),
        prompts=_prompts(),
    )
    assert retried.status == "completed" and retried.request_count == 3
    cancelled_provider = DeterministicMockVisionProvider(failures=(ProviderCancelledError(),))
    cancelled = _cascade(cancelled_provider).run(
        _images(),
        policy=CascadePolicy(CascadeProfile.FAST_LOCAL, CascadeBudget(maximum_requests=10)),
        prompts=_prompts(),
    )
    assert cancelled.status == "cancelled"
    assert retried.hosted_record_count == cancelled.hosted_record_count == 0


def test_cascade_charges_every_hosted_stage_request_not_unique_records() -> None:
    blocked_provider = HostedMockProvider()
    blocked = _cascade(blocked_provider, privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=3,
                maximum_retries=0,
                maximum_estimated_cost=1.5,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert blocked.status == "partial"
    assert blocked.stop_reason == "labeling_cost_budget_exhausted"
    assert blocked.request_count == blocked.hosted_request_count == blocked_provider.call_count == 1
    assert blocked.hosted_record_count == 1
    assert blocked.estimated_cost == 1.0
    assert [(charge.stage, charge.estimated_cost) for charge in blocked.hosted_charges] == [("description", 1.0)]

    allowed_provider = HostedMockProvider()
    allowed = _cascade(allowed_provider, privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=3,
                maximum_retries=0,
                maximum_estimated_cost=3.0,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert allowed.status == "completed"
    assert allowed.request_count == allowed.hosted_request_count == allowed_provider.call_count == 3
    assert allowed.hosted_record_count == 1
    assert allowed.estimated_cost == 3.0
    assert [charge.stage for charge in allowed.hosted_charges] == [
        "description",
        "hypotheses",
        "verification",
    ]


def test_cascade_route_rates_and_retry_reservation_are_fail_closed() -> None:
    routes = tuple(
        HostedCostRate("mock.hosted", "mock-vision-v1", stage, cost_per_request=cost)
        for stage, cost in (("description", 0.25), ("hypotheses", 0.5), ("verification", 0.75))
    )
    routed_provider = HostedMockProvider()
    routed = _cascade(routed_provider, privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=3,
                maximum_retries=0,
                maximum_estimated_cost=1.5,
                trusted_hosted_cost_rates=routes,
            ),
        ),
        prompts=_prompts(),
    )
    assert routed.status == "completed" and routed.estimated_cost == 1.5
    assert [charge.estimated_cost for charge in routed.hosted_charges] == [0.25, 0.5, 0.75]

    retrying_provider = HostedMockProvider(failures=(ProviderError("synthetic_retry", "fake", retryable=True),))
    retry_blocked = _cascade(retrying_provider, privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=6,
                maximum_retries=1,
                maximum_estimated_cost=1.5,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert retry_blocked.stop_reason == "labeling_cost_budget_exhausted"
    assert retry_blocked.request_count == retry_blocked.hosted_request_count == retrying_provider.call_count == 0

    retried_provider = HostedMockProvider(failures=(ProviderError("synthetic_retry", "fake", retryable=True),))
    retried = _cascade(retried_provider, privacy=PrivacyPolicy.ALLOW_HOSTED).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=6,
                maximum_retries=1,
                maximum_estimated_cost=6.0,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert retried.status == "completed"
    assert retried.request_count == retried.hosted_request_count == retried_provider.call_count == 4
    assert retried.estimated_cost == 4.0
    assert [charge.stage for charge in retried.hosted_charges] == [
        "description",
        "description",
        "hypotheses",
        "verification",
    ]


def test_cascade_charges_request_and_image_rates_for_every_batch() -> None:
    stages = ("description", "hypotheses", "verification")
    rates = tuple(
        HostedCostRate(
            "mock.hosted",
            "mock-vision-v1",
            stage,
            cost_per_request=0.5,
            cost_per_image=0.25,
        )
        for stage in stages
    )
    provider = HostedMockProvider()
    result = _cascade(provider, privacy=PrivacyPolicy.ALLOW_HOSTED, batch_size=2).run(
        (
            ImageInput("r1", b"one"),
            ImageInput("r2", b"two"),
            ImageInput("r3", b"three"),
        ),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=3,
                maximum_requests=6,
                maximum_retries=0,
                maximum_estimated_cost=5.25,
                trusted_hosted_cost_rates=rates,
            ),
        ),
        prompts=_prompts(),
    )
    assert result.status == "completed"
    assert result.request_count == result.hosted_request_count == provider.call_count == 6
    assert result.estimated_cost == 5.25
    assert [charge.estimated_cost for charge in result.hosted_charges] == [1.0, 0.75] * 3


class _FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _LateHostedProvider(HostedMockProvider):
    def __init__(self, clock: _FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.seen_timeout: float | None = None

    def label_batch(self, request):
        self.seen_timeout = request.timeout_seconds
        self.clock.advance(2.0)
        return super().label_batch(request)


def test_cascade_propagates_deadline_discards_late_result_and_keeps_charge() -> None:
    clock = _FakeClock()
    provider = _LateHostedProvider(clock)
    result = _cascade(provider, privacy=PrivacyPolicy.ALLOW_HOSTED, clock=clock).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=3,
                maximum_retries=0,
                maximum_elapsed_seconds=1.0,
                maximum_estimated_cost=3.0,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert provider.seen_timeout == 1.0
    assert provider.call_count == result.request_count == result.hosted_request_count == 1
    assert result.estimated_cost == 1.0
    assert result.status == "partial" and result.stop_reason == "labeling_elapsed_budget_exhausted"
    assert not result.stages and result.exception_queue == ("r1",)


def test_cascade_deadline_stops_retry_before_second_hosted_request() -> None:
    clock = _FakeClock()
    provider = HostedMockProvider(failures=(ProviderError("synthetic_retry", "fake", retryable=True),))
    result = _cascade(provider, privacy=PrivacyPolicy.ALLOW_HOSTED, clock=clock).run(
        (ImageInput("r1", b"one"),),
        policy=CascadePolicy(
            CascadeProfile.HIGH_QUALITY,
            CascadeBudget(
                maximum_hosted_records=1,
                maximum_requests=6,
                maximum_retries=1,
                maximum_elapsed_seconds=0.1,
                maximum_estimated_cost=6.0,
                trusted_hosted_cost_per_record=1.0,
            ),
        ),
        prompts=_prompts(),
    )
    assert result.stop_reason == "labeling_elapsed_budget_exhausted"
    assert provider.call_count == result.request_count == result.hosted_request_count == 1
    assert result.estimated_cost == 1.0


def _cache_identity() -> LabelingCacheIdentity:
    return LabelingCacheIdentity(
        "decision",
        "source",
        "rgba",
        "renders",
        "provider",
        "model",
        "prompt",
        "taxonomy",
        "description-schema",
        "hypothesis-schema",
        "embedding",
        "retrieval-index",
        "reference-set",
        "decision-policy",
        "calibration",
        "metadata",
        "provider-configuration",
        "reviewed-truth",
        NO_CONTEXT_IDENTITY,
        "technical-evidence",
        "technical-extraction",
        TECHNICAL_SCHEMA_IDENTITY,
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_image_identity",
        "decoded_rgba_identity",
        "render_bundle_identity",
        "provider_identity",
        "model_identity",
        "prompt_identity",
        "taxonomy_identity",
        "description_schema",
        "hypothesis_schema",
        "embedding_identity",
        "retrieval_index_identity",
        "reference_set_identity",
        "decision_policy_identity",
        "calibration_identity",
        "metadata_identity",
        "provider_configuration_identity",
        "reviewed_truth_identity",
        "context_identity",
        "technical_evidence_identity",
        "technical_extraction_identity",
        "technical_schema_identity",
    ],
)
def test_cache_every_identity_dimension_rejects_stale_result(tmp_path, field) -> None:
    identity = _cache_identity()
    changed = replace(identity, **{field: f"changed-{field}"})
    assert changed.identity != identity.identity
    cache = HierarchicalLabelingCache(tmp_path / field)
    cache.put(identity, {"ok": True})
    assert cache.get(identity) == {"ok": True}
    assert cache.get(changed) is None


def test_cache_interrupted_batch_safe_resume_preserves_success(tmp_path) -> None:
    run = LabelingRunStore.open_or_create(
        tmp_path / "run", run_identity="run-1", command=("label", "run"), architecture_identity="arch-1"
    )
    run.mark_item(stage="description", record_identity="r1", status="succeeded", artifact_identity="a1")
    run.mark_item(stage="description", record_identity="r2", status="failed", error_code="fake_failure")
    resumed = LabelingRunStore.open_or_create(
        tmp_path / "run", run_identity="run-1", command=("label", "run"), architecture_identity="arch-1"
    )
    assert resumed.successful_items("description") == {"r1": "a1"}
    assert resumed.pending_items("description", ("r1", "r2", "r3")) == ("r2", "r3")
    with pytest.raises(ContractValidationError, match="cannot be downgraded"):
        resumed.mark_item(stage="description", record_identity="r1", status="failed", error_code="late")


def _active(index, **changes):
    values = {
        "record_identity": f"r{index}",
        "image_identity": f"i{index}",
        "cluster_identity": f"c{index % 3}",
        "duplicate_cluster_identity": "duplicate-shared" if index in {0, 1} else f"d{index}",
        "near_duplicate_cluster_identity": "near-shared" if index in {2, 3, 4} else None,
        "expected_information_gain": 0.9 - index * 0.03,
        "cluster_representativeness": 0.7,
        "novelty": 0.4,
        "high_confidence_disagreement": 0.3,
        "visual_metadata_conflict": False,
        "low_calibrated_margin": 0.5,
        "rare_class": False,
        "affected_cluster_size": 10,
        "taxonomy_gap": False,
        "source_drift": 0.2,
        "provider_disagreement": 0.2,
        "already_reviewed": False,
        "legally_eligible": True,
        "technically_usable": True,
    }
    values.update(changes)
    return ActiveLearningCandidate(**values)


def test_active_learning_deterministic_priority_duplicate_and_eligibility_controls() -> None:
    candidates = [
        *(_active(index) for index in range(6)),
        _active(7, already_reviewed=True),
        _active(8, legally_eligible=False),
        _active(9, technically_usable=False),
    ]
    policy = ActiveLearningPolicy(review_budget=5, seed=99, maximum_per_near_duplicate_cluster=2)
    kwargs = {
        "dataset_identity": "dataset",
        "reference_set_identity": "reference",
        "embedding_identity": "embedding",
        "calibration_identity": "calibration",
        "round_number": 1,
        "policy": policy,
    }
    first = generate_review_round(candidates, **kwargs)
    second = generate_review_round(candidates, **kwargs)
    assert first == second
    assert not {"r0", "r1"}.issubset(first["selected_record_identities"])
    assert not {"r7", "r8", "r9"} & set(first["selected_record_identities"])
    assert first["round_identity"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"coverage_target_reached": True}, "coverage_target_reached"),
        ({"precision_target_maintained": False}, "precision_target_cannot_be_maintained"),
        ({"taxonomy_work_required": True}, "new_taxonomy_work_required"),
        ({"reviews_used": 10}, "review_budget_exhausted"),
        ({"marginal_gain": 0.01}, "marginal_gain_below_threshold"),
    ],
)
def test_active_learning_stopping_conditions(changes, expected) -> None:
    values = {
        "coverage_target_reached": False,
        "precision_target_maintained": True,
        "marginal_gain": 0.5,
        "reviews_used": 1,
        "review_budget": 10,
        "taxonomy_work_required": False,
        "minimum_marginal_gain": 0.05,
    }
    values.update(changes)
    assert stopping_reason(**values) == expected


def test_reporting_no_data_precision_coverage_confusion_galleries_denominators_and_unknown_cost(tmp_path) -> None:
    graph = load_default_taxonomy()
    records = [
        {"record_identity": "r1", "accepted_path": list(graph.path("bottle")), "abstained": False},
        {"record_identity": "r2", "accepted_path": [], "abstained": True},
    ]
    no_truth = build_report_data(
        records,
        graph,
        operational={"sample_gallery": ["r1"], "cache": {"hits": 1, "lookups": 2, "rate": 0.5}},
    )
    assert no_truth["truth_metrics"]["precision_coverage_curve"] is None
    assert no_truth["operational"]["cost_estimate"] == "unknown"
    with pytest.raises(ContractValidationError, match="raw truth mappings"):
        build_report_data(
            records,
            graph,
            truth_rows=(
                {
                    "record_identity": "r1",
                    "predicted_path": list(graph.path("bottle")),
                    "truth_path": list(graph.path("bottle")),
                    "calibrated_probability": 0.9,
                    "truth_source": "human_review",
                    "source_identity": "source",
                },
            ),
        )
    assert no_truth["claims"]["precision_graph_available"] is False
    assert no_truth["galleries"]["samples"] == ["r1"]
    json_path, html_path = write_offline_report(no_truth, tmp_path / "report")
    assert json_path.is_file() and "Precision and coverage" in html_path.read_text(encoding="utf-8")


def _pilot_candidates():
    return [
        PilotCandidate(
            f"r{index}",
            f"i{index}",
            f"source-{index % 3}",
            f"pack-{index % 4}",
            f"cluster-{index % 5}",
            f"style-{index % 2}",
            f"size-{index % 3}",
            index % 7 == 0,
            f"technical-{index % 4}",
            "duplicate-shared" if index in {0, 1} else f"duplicate-{index}",
            (index % 10) / 10,
            True,
            True,
        )
        for index in range(25)
    ]


def test_pilot_planner_representative_deterministic_non_authorizing() -> None:
    first = plan_pilot(
        _pilot_candidates(), dataset_identity="dataset", target_size=20, reference_cohort_size=300, seed=7
    )
    second = plan_pilot(
        _pilot_candidates(), dataset_identity="dataset", target_size=20, reference_cohort_size=300, seed=7
    )
    assert first == second and first["selected_records"] == 20
    assert not first["production_authorization"] and not first["pilot_runs_automatically"]
    assert first["maximum_hosted_calls"] == 0 and first["time_estimate_hours"].startswith("unknown")


def test_product_configuration_page_status_and_no_precision_claim_without_truth(tmp_path) -> None:
    values = copy.deepcopy(DEFAULT_CONFIG)
    context = ProjectContext(tmp_path, values, None, tmp_path / "runs")
    app = create_app(context, plugins=(create_plugin(),))
    client = TestClient(app)
    page = client.get("/labeling")
    assert page.status_code == 200 and "Hierarchical labeling" in page.text
    status = client.get("/labeling/api/status").json()
    reliability = next(card for card in status["cards"] if card["key"] == "automatic_reliability")
    assert reliability["held_out_precision"] is None
    assert status["precision_claim_suppressed_without_truth"] is True
    assert not status["production_authorized"]


def test_product_deterministic_preparation_preserves_image_only_and_has_zero_provider_calls(tmp_path) -> None:
    image = write_sprite(tmp_path / "input" / "sprite.png")
    item = {
        "item_id": "r1",
        "relative_path": "sprite.png",
        "source_path": str(image),
        "current_disposition": "accepted",
    }
    config = {
        "labeling": {"hierarchical_enabled": True, "hierarchical_profile": "fast_local", "reference_cohort_size": 400}
    }
    result = prepare_configured_labeling([item], config=config, output_root=tmp_path / "output")
    assert result["status"] == "prepared" and result["provider_calls"] == 0
    assert item["hierarchical_labeling"]["truth_status"] == "not_human_truth"
    assert item["hierarchical_labeling"]["conditioned_dataset_ready"] is False


def test_product_config_validation_profiles_and_reference_size(tmp_path) -> None:
    values = copy.deepcopy(DEFAULT_CONFIG)
    values["labeling"]["hierarchical_profile"] = "invalid"
    config = tmp_path / "spritelab.yaml"
    import yaml

    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ConfigError, match="hierarchical_profile"):
        ProjectConfig.load(tmp_path)


def test_product_status_never_authorizes_conditioned_dataset(tmp_path) -> None:
    status = product_status({"labeling": {"hierarchical_enabled": True}}, tmp_path)
    conditioned = next(card for card in status["cards"] if card["key"] == "conditioned_readiness")
    assert conditioned["status"] == "NOT_AUTHORIZED"

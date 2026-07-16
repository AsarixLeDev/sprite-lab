# The multiplication sign is an intentional UTF-8 round-trip fixture.
# ruff: noqa: RUF001

from __future__ import annotations

import copy
import io
import json
import urllib.error
from typing import Any

import pytest

from spritelab.harvest.label_v4.providers import (
    MOCK_JSON_REQUEST_POLICY_VERSION,
    OPENAI_COMPATIBLE_REQUEST_POLICY_VERSION,
    MockJSONProvider,
    OpenAICompatibleJSONProvider,
    request_identity,
)
from spritelab.harvest.label_v4.routing import AdaptiveRoutingSignals, decide_adaptive_routing
from spritelab.harvest.label_v4.verifier import (
    classify_verifier_independence,
    derive_verifier_effects,
    parse_verifier_response,
)


class _Response:
    def __init__(self, value: dict[str, Any]) -> None:
        self.payload = json.dumps(value, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _provider() -> OpenAICompatibleJSONProvider:
    return OpenAICompatibleJSONProvider(
        base_url="https://unit.invalid/v1",
        api_key="not-a-real-key",
        model="mock/model",
        namespace="test",
    )


def test_mock_artifact_has_zero_http_attempts_and_utf8_raw_output() -> None:
    provider = MockJSONProvider({"stage": {"text": "— é ×"}})
    artifact = provider.call_json(stage="stage", prompt="é", prompt_version="v1")

    assert artifact.http_attempts == 0
    assert artifact.request_policy_version == MOCK_JSON_REQUEST_POLICY_VERSION
    assert "— é ×" in artifact.raw_output
    assert "\\u2014" not in artifact.raw_output
    assert artifact.to_dict()["http_attempts"] == 0


def test_openai_request_and_response_are_utf8_and_count_one_http_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        content = json.dumps({"text": "— é ×"}, ensure_ascii=False)
        return _Response(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    artifact = _provider().call_json(
        stage="D",
        prompt="Vérify — size × color",
        prompt_version="v2",
        payload={"claim": "café — 2×"},
    )

    request = captured["request"]
    assert request.get_header("Content-type") == "application/json; charset=utf-8"
    body = request.data.decode("utf-8")
    assert "Vérify — size × color" in body
    assert "café — 2×" in body
    assert artifact.parsed_output == {"text": "— é ×"}
    assert artifact.http_attempts == 1
    assert artifact.request_policy_version == OPENAI_COMPATIBLE_REQUEST_POLICY_VERSION


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        urllib.error.HTTPError("https://unit.invalid", 503, "unavailable", {}, io.BytesIO()),
    ],
)
def test_openai_failures_still_count_exact_http_attempt(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    artifact = _provider().call_json(stage="D", prompt="verify", prompt_version="v2")

    assert artifact.http_attempts == 1
    assert artifact.parsed_output is None
    assert artifact.failure_diagnostics


def test_request_policy_and_options_are_part_of_request_identity() -> None:
    common = {
        "stage": "D",
        "model_identity": "m",
        "prompt": "p",
        "prompt_version": "v",
        "image_hash": "h",
        "payload": {"text": "— é ×"},
    }
    first, _ = request_identity(
        **common,
        request_policy_version="policy-a",
        request_options={"max_tokens": 128},
    )
    changed_policy, _ = request_identity(
        **common,
        request_policy_version="policy-b",
        request_options={"max_tokens": 128},
    )
    changed_options, _ = request_identity(
        **common,
        request_policy_version="policy-a",
        request_options={"max_tokens": 256},
    )

    assert len({first, changed_policy, changed_options}) == 3


def test_strict_verifier_parser_rejects_unknown_duplicate_and_invalid_results_without_mutation() -> None:
    raw = {
        "claim_results": [
            {"claim_id": "known", "verdict": "maybe", "visible_support": "ambiguous"},
            {"claim_id": "unknown", "verdict": "supported"},
            {"claim_id": "duplicate", "verdict": "supported"},
            {"claim_id": "duplicate", "verdict": "unresolved"},
        ]
    }
    before = copy.deepcopy(raw)
    parsed = parse_verifier_response(raw, known_claims=["known", "duplicate"])

    assert not parsed.claim_results
    assert {result.reason for result in parsed.rejected_results} == {
        "invalid_verdict",
        "unknown_claim_id",
        "duplicate_claim_id",
    }
    assert parsed.unanswered_claim_ids == ("known", "duplicate")
    assert raw == before
    assert parsed.rejected_results[0].raw_result


def test_verifier_independence_labels_same_and_different_models() -> None:
    assert classify_verifier_independence("model-a", ["model-a", "model-b"]) == "same_model_independent_prompt"
    assert classify_verifier_independence("model-c", ["model-a", "model-b"]) == "different_model_independent_prompt"


def test_verifier_effects_use_exact_conflict_ids_and_record_no_decision_change() -> None:
    claims = [
        {
            "claim_id": "first",
            "field": "category",
            "claimed_value": "weapon",
            "conflict_id": "conflict-1",
            "dispute_code": "same-code",
            "resolve_on_supported": True,
        },
        {
            "claim_id": "second",
            "field": "canonical_object",
            "claimed_value": "rod",
            "conflict_id": "conflict-2",
            "dispute_code": "same-code",
            "resolve_on_supported": True,
        },
    ]
    parsed = parse_verifier_response(
        {
            "claim_results": [
                {"claim_id": "first", "verdict": "supported"},
                {"claim_id": "second", "verdict": "unresolved"},
            ]
        },
        known_claims=claims,
    )
    effects = derive_verifier_effects(parsed, known_claims=claims)

    assert effects.resolved_conflict_ids == ("conflict-1",)
    assert effects.retained_conflict_ids == ("conflict-2",)
    assert effects.claim_effects[0].effects == ("conflict_resolved",)
    assert effects.claim_effects[1].effects == ("claim_abstained", "conflict_retained")
    assert effects.claim_effects[1].decision_change == "no_decision_change"

    unresolved_only = parse_verifier_response(
        {"claim_results": [{"claim_id": "second", "verdict": "unresolved"}]},
        known_claims=[claims[1]],
    )
    no_change = derive_verifier_effects(unresolved_only, known_claims=[claims[1]])
    assert no_change.decision_change == "no_decision_change"
    assert "no_decision_change" in no_change.overall_effects


def test_routing_never_uses_global_risk_or_open_set_novelty_alone() -> None:
    high_risk = decide_adaptive_routing(AdaptiveRoutingSignals(critical_field_risk_upper=0.9))
    novel = decide_adaptive_routing(AdaptiveRoutingSignals(open_set_novelty=1.0))

    assert high_risk.run_stage_d is False
    assert high_risk.stage_d_skipped_reasons == ("risk_without_concrete_dispute",)
    assert novel.run_stage_d is False
    assert novel.stage_d_skipped_reasons == ("open_set_novelty_without_concrete_dispute",)


def test_routing_skips_source_category_visual_form_dispute_when_policy_abstains() -> None:
    decision = decide_adaptive_routing(
        AdaptiveRoutingSignals(
            vlm_deterministic_disagreement=True,
            critical_field_risk_upper=0.9,
            open_set_novelty=1.0,
            policy_abstains_object_identity=True,
            source_category_authoritative=True,
        )
    )

    assert decision.run_stage_d is False
    assert "policy_abstains_source_category_visual_form_disagreement" in decision.stage_d_skipped_reasons


def test_routing_runs_for_concrete_dispute_but_respects_zero_eligible_claims() -> None:
    concrete = decide_adaptive_routing(
        AdaptiveRoutingSignals(
            filename_visual_color_conflict=True,
            critical_field_risk_upper=0.9,
            open_set_novelty=1.0,
            verifier_eligible_claim_count=1,
        )
    )
    none_eligible = decide_adaptive_routing(
        AdaptiveRoutingSignals(
            filename_visual_color_conflict=True,
            verifier_eligible_claim_count=0,
        )
    )

    assert concrete.run_stage_d is True
    assert concrete.stage_d_reasons == (
        "critical_field_risk_above_threshold",
        "filename_visual_color_conflict",
        "open_set_novelty_high",
    )
    assert none_eligible.run_stage_d is False
    assert "no_verifier_eligible_claims" in none_eligible.stage_d_skipped_reasons

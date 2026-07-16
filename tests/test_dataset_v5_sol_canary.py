from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from spritelab.dataset_v5.blind import BlindInput
from spritelab.dataset_v5.canary import (
    REQUIRED_CANARY_TAGS,
    SolCanaryModelUnavailable,
    SolPricing,
    run_sol_canary,
)
from spritelab.dataset_v5.raw_cli import main
from spritelab.dataset_v5.sol import SolModelUnavailable


class _Identity:
    def canonical(self) -> dict[str, str]:
        return {
            "backend": "openai_responses_v1",
            "endpoint_identity": "https://sol.example.test/v1",
            "model_family": "GPT-5.6 Sol",
            "model_identifier": "gpt-5.6-sol-2026-07-01",
            "model_version": "2026-07-01.1",
            "provider": "Fixture Sol",
            "request_schema_version": "responses-v1",
        }


class _Provider:
    def __init__(self, *, fail_at: int | None = None, invalid_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.invalid_at = invalid_at
        self.identity = _Identity()

    def preflight(self) -> _Identity:
        return self.identity

    def call_blind_pass(
        self,
        blind_input: BlindInput,
        *,
        request_id: str,
        pass_kind: str,
    ) -> dict[str, Any]:
        del blind_input, request_id
        self.calls += 1
        if self.calls == self.fail_at:
            raise SolModelUnavailable("response identity changed")
        base: dict[str, Any] = {
            "authoritative": self.calls != self.invalid_at,
            "latency_ms": 5.0,
            "pass_kind": pass_kind,
            "status": "success" if self.calls != self.invalid_at else "invalid",
            "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        if base["authoritative"]:
            base["output"] = _valid_output()
        else:
            base["error_code"] = "strict_response_rejected"
        return base


def _valid_output() -> dict[str, Any]:
    return {
        "abstentions": {},
        "canonical_object": "hammer",
        "category": "tool",
        "color_roles": {
            "highlight": [],
            "outline": ["dark"],
            "primary": ["blue"],
            "secondary": [],
            "shadow": [],
        },
        "description": "A compact object with a straight handle.",
        "domain": "inventory_icon",
        "explicit_material": None,
        "field_rationales": {},
        "field_risk_signals": {},
        "material_applicability": "unknown",
        "role": "functional_tool",
        "schema_version": "blind_semantic_output_v1",
        "visual_form": ["compact head", "straight handle"],
    }


def _cohort() -> list[tuple[BlindInput, Mapping[str, Any]]]:
    records: list[tuple[BlindInput, Mapping[str, Any]]] = []
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba[1, 1] = [20, 80, 160, 255]
    for index in range(20):
        digest = hashlib.sha256(f"canary-{index}".encode()).hexdigest()
        tags = sorted(REQUIRED_CANARY_TAGS) if index == 0 else []
        records.append((BlindInput.from_rgba("rec_" + digest, rgba), {"canary_tags": tags}))
    return records


def _pricing() -> SolPricing:
    return SolPricing(
        input_per_million=1.0,
        output_per_million=2.0,
        pricing_identity="fixture-pricing-2026-07-13",
    )


def test_canary_strict_schema_rejection_is_a_blocking_quality_gate() -> None:
    report = run_sol_canary(
        _cohort(),
        provider=_Provider(invalid_at=4),  # type: ignore[arg-type]
        projected_record_count=200,
        pricing=None,
        metered=False,
    )

    assert report["invalid_json_count"] == 1
    assert report["quality_gates"]["strict_response_schema"]["passed"] is False
    assert report["quality_gates_passed"] is False
    assert report["stop_before_bulk"] is True
    assert report["status"] == "blocked_non_authoritative"
    assert report["authoritative"] is False


def test_canary_metered_and_unmetered_cost_authorization_rules() -> None:
    missing_pricing_provider = _Provider()
    with pytest.raises(ValueError, match="requires explicit pricing"):
        run_sol_canary(
            _cohort(),
            provider=missing_pricing_provider,  # type: ignore[arg-type]
            projected_record_count=200,
            pricing=None,
            metered=True,
        )
    assert missing_pricing_provider.calls == 0

    unmetered = run_sol_canary(
        _cohort(),
        provider=_Provider(),  # type: ignore[arg-type]
        projected_record_count=200,
        pricing=None,
        metered=False,
    )
    assert unmetered["observed_cost"] == 0.0
    assert unmetered["projected_total_cost"] == 0.0
    assert unmetered["stop_before_bulk"] is False
    assert unmetered["status"] == "pass"

    paid_without_authorization = run_sol_canary(
        _cohort(),
        provider=_Provider(),  # type: ignore[arg-type]
        projected_record_count=200,
        pricing=_pricing(),
        metered=True,
    )
    assert paid_without_authorization["projected_total_cost"] > 0
    assert paid_without_authorization["stop_before_bulk"] is True
    assert paid_without_authorization["status"] == "canary_passed_bulk_blocked"

    paid_with_authorization = run_sol_canary(
        _cohort(),
        provider=_Provider(),  # type: ignore[arg-type]
        projected_record_count=200,
        pricing=_pricing(),
        metered=True,
        explicit_bulk_cost_authorization=True,
    )
    assert paid_with_authorization["stop_before_bulk"] is False
    assert paid_with_authorization["status"] == "pass"


def test_canary_requires_a_positive_projected_record_count_before_calls() -> None:
    provider = _Provider()
    with pytest.raises(ValueError, match="positive integer"):
        run_sol_canary(
            _cohort(),
            provider=provider,  # type: ignore[arg-type]
            projected_record_count=0,
            pricing=None,
            metered=False,
        )
    assert provider.calls == 0


def test_mid_canary_identity_failure_preserves_partial_call_evidence() -> None:
    provider = _Provider(fail_at=4)
    with pytest.raises(SolCanaryModelUnavailable) as caught:
        run_sol_canary(
            _cohort(),
            provider=provider,  # type: ignore[arg-type]
            projected_record_count=200,
            pricing=None,
            metered=False,
        )

    report = caught.value.partial_report
    assert provider.calls == 4
    assert report["provider_calls"] == 4
    assert report["records_started"] == 2
    assert report["records_completed"] == 1
    assert report["canary_record_count"] == 1
    assert report["token_usage"] == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
    assert report["token_usage_reported_calls"] == 3
    assert report["latency_ms"]["max"] >= 5.0
    assert report["artifacts"][-1]["error_code"] == "exact_sol_identity_changed"
    assert report["artifacts"][-1]["token_usage_available"] is False
    assert report["reason"] == "SOL_MODEL_UNAVAILABLE"
    assert report["status"] == "blocked_non_authoritative"


def test_cli_writes_partial_report_instead_of_zero_call_unavailable_report(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.dataset_v5 import raw_cli

    for key, value in {
        "SPRITELAB_SOL_API_KEY": "fixture-secret",
        "SPRITELAB_SOL_BACKEND": "openai_responses_v1",
        "SPRITELAB_SOL_BASE_URL": "https://sol.example.test/v1",
        "SPRITELAB_SOL_MODEL": "gpt-5.6-sol-2026-07-01",
    }.items():
        monkeypatch.setenv(key, value)
    provider = _Provider(fail_at=4)
    monkeypatch.setattr(raw_cli, "SolProvider", lambda config: provider)
    monkeypatch.setattr(raw_cli, "_load_canary_cohort", lambda *args: _cohort())
    output = tmp_path / "partial_canary.json"

    exit_code = main(
        [
            "sol-canary",
            "--cohort",
            str(tmp_path / "not-read.jsonl"),
            "--image-root",
            str(tmp_path),
            "--projected-record-count",
            "200",
            "--unmetered",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 78
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["provider_calls"] == 4
    assert report["token_usage"]["total_tokens"] == 45
    assert report["reason"] == "SOL_MODEL_UNAVAILABLE"


def test_cli_rejects_non_positive_projection_and_unmetered_price_flags(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        main(
            [
                "sol-canary",
                "--cohort",
                str(tmp_path / "missing.jsonl"),
                "--image-root",
                str(tmp_path),
                "--projected-record-count",
                "0",
                "--unmetered",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        main(
            [
                "sol-canary",
                "--cohort",
                str(tmp_path / "missing.jsonl"),
                "--image-root",
                str(tmp_path),
                "--projected-record-count",
                "20",
                "--unmetered",
                "--input-cost-per-million",
                "1",
                "--output-cost-per-million",
                "1",
                "--pricing-identity",
                "contradictory",
                "--output",
                str(tmp_path / "other-report.json"),
            ]
        )

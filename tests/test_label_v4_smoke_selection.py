from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from spritelab.harvest.cli.label_v4_cmds import _summarize_provider_accounting, register
from spritelab.harvest.label_v4.cohort import (
    _atomic_write_json,
    _atomic_write_jsonl,
    run_same_cohort_comparison,
    same_cohort_acceptance_checks,
    select_cohort_rows,
)

MULTIPLICATION_SIGN = "\N{MULTIPLICATION SIGN}"
UTF8_SENTINEL = f"— é {MULTIPLICATION_SIGN}"


def test_repeatable_sprite_ids_preserve_order_and_max_records_applies_after_targeting() -> None:
    rows = [{"sprite_id": value, "ordinal": index} for index, value in enumerate(("a", "b", "c"))]

    selected = select_cohort_rows(rows, sprite_ids=("c", "a"), max_records=1)

    assert [row["sprite_id"] for row in selected] == ["c"]
    assert selected[0]["ordinal"] == 2


def test_record_manifest_preserves_jsonl_order_after_explicit_targets(tmp_path: Path) -> None:
    rows = [{"sprite_id": value} for value in ("first", "é", MULTIPLICATION_SIGN, "—")]
    manifest = tmp_path / "targets.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"sprite_id": value}, ensure_ascii=False) + "\n" for value in ("—", "é", MULTIPLICATION_SIGN)
        ),
        encoding="utf-8",
    )

    selected = select_cohort_rows(rows, sprite_ids=("first",), record_manifest=manifest)

    assert [row["sprite_id"] for row in selected] == ["first", "—", "é", MULTIPLICATION_SIGN]


def test_record_manifest_rejects_duplicate_and_unknown_targets(tmp_path: Path) -> None:
    rows = [{"sprite_id": "a"}, {"sprite_id": "b"}]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"sprite_id":"a"}\n{"sprite_id":"a"}\n', encoding="utf-8")
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text('{"sprite_id":"missing"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate targeted sprite ids"):
        select_cohort_rows(rows, record_manifest=duplicate)
    with pytest.raises(ValueError, match="targeted records missing from cohort"):
        select_cohort_rows(rows, record_manifest=unknown)


def test_absent_acceptance_record_is_not_evaluated_but_present_mismatch_fails() -> None:
    matching = {
        "sprite_id": "acq_idylwild_armory_iron_buckler",
        "canonical_object": "buckler",
        "category": "armor",
        "explicit_material": "iron",
        "role": "defensive_equipment",
    }
    mismatch = {**matching, "canonical_object": "shield"}

    absent = same_cohort_acceptance_checks({"A": []})["A"]["named_deterministic_recovery"]
    passed = same_cohort_acceptance_checks({"A": [matching]})["A"]["named_deterministic_recovery"]
    failed = same_cohort_acceptance_checks({"A": [mismatch]})["A"]["named_deterministic_recovery"]

    assert absent[matching["sprite_id"]]["status"] == "not_evaluated"
    assert absent[matching["sprite_id"]]["pass"] is None
    assert absent[matching["sprite_id"]]["observed"] is None
    assert passed[matching["sprite_id"]]["status"] == "passed"
    assert passed[matching["sprite_id"]]["pass"] is True
    assert failed[matching["sprite_id"]]["status"] == "failed"
    assert failed[matching["sprite_id"]]["pass"] is False


def test_three_record_smoke_marks_unexecuted_named_recoveries_not_evaluated(tmp_path: Path) -> None:
    result = run_same_cohort_comparison(output_dir=tmp_path / "smoke", max_records=3)

    for variant in ("A", "B", "C"):
        named = result["acceptance_checks"][variant]["named_deterministic_recovery"]
        assert named
        assert {check["status"] for check in named.values()} == {"not_evaluated"}
        assert all(check["pass"] is None for check in named.values())


def test_smoke_and_canary_parsers_expose_target_and_shared_cache_flags(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    manifest = tmp_path / "targets.jsonl"

    smoke = parser.parse_args(
        [
            "label-v4-smoke",
            "--sprite-id",
            "second",
            "--sprite-id",
            "first",
            "--record-manifest",
            str(manifest),
        ]
    )
    canary = parser.parse_args(
        [
            "label-v4-canary",
            "--output-root",
            str(tmp_path / "out"),
            "--vlm-model",
            "mock-model",
            "--mode",
            "C",
            "--sprite-id",
            "one",
            "--record-manifest",
            str(manifest),
            "--shared-cache-root",
            str(tmp_path / "shared"),
            "--require-shared-bc-cache",
            "--input-cost-per-million",
            "0.25",
            "--output-cost-per-million",
            "0.75",
        ]
    )

    assert smoke.sprite_id == ["second", "first"]
    assert smoke.record_manifest == manifest
    assert canary.sprite_id == ["one"]
    assert canary.record_manifest == manifest
    assert canary.shared_cache_root == tmp_path / "shared"
    assert canary.require_shared_bc_cache is True
    assert canary.input_cost_per_million == 0.25
    assert canary.output_cost_per_million == 0.75


def test_canary_accounting_summary_aggregates_current_run_usage_only() -> None:
    outputs = [
        {
            "provider_accounting": {
                "logical_stage_count": 2,
                "actual_http_attempts": 0,
                "new_provider_calls": 0,
                "shared_cache_hits": 2,
                "cache_hits": 2,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "pricing_configured": True,
                "estimated_provider_cost": 0.0,
                "per_stage": {
                    "B_blind_vlm_proposal": {
                        "new_provider_calls": 0,
                        "actual_http_attempts": 0,
                        "cache_hit": True,
                        "shared_cache_hit": True,
                        "execution_latency_ms": 1.0,
                        "provider_latency_ms": 0.0,
                        "new_token_usage": {},
                    },
                    "C_text_reconciliation": {
                        "new_provider_calls": 0,
                        "actual_http_attempts": 0,
                        "cache_hit": True,
                        "shared_cache_hit": True,
                        "execution_latency_ms": 2.0,
                        "provider_latency_ms": 0.0,
                        "new_token_usage": {},
                    },
                },
            }
        },
        {
            "provider_accounting": {
                "logical_stage_count": 1,
                "actual_http_attempts": 1,
                "new_provider_calls": 1,
                "shared_cache_hits": 0,
                "cache_hits": 0,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "pricing_configured": True,
                "estimated_provider_cost": 0.00000625,
                "per_stage": {
                    "D_independent_verifier": {
                        "new_provider_calls": 1,
                        "actual_http_attempts": 1,
                        "cache_hit": False,
                        "shared_cache_hit": False,
                        "execution_latency_ms": 8.0,
                        "provider_latency_ms": 7.0,
                        "new_token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                },
            }
        },
    ]

    summary = _summarize_provider_accounting(outputs)

    assert summary["logical_stage_count"] == 3
    assert summary["actual_http_attempts"] == 1
    assert summary["new_provider_calls"] == 1
    assert summary["shared_cache_hits"] == 2
    assert summary["total_tokens"] == 15
    assert summary["estimated_provider_cost"] == pytest.approx(0.00000625)
    assert summary["per_stage"]["D_independent_verifier"]["new_token_usage"]["total_tokens"] == 15


def test_json_and_jsonl_writes_round_trip_utf8_without_ascii_escaping(tmp_path: Path) -> None:
    value = {"text": UTF8_SENTINEL}
    json_path = tmp_path / "value.json"
    jsonl_path = tmp_path / "value.jsonl"

    _atomic_write_json(json_path, value)
    _atomic_write_jsonl(jsonl_path, [value])

    for path in (json_path, jsonl_path):
        payload = path.read_bytes()
        assert UTF8_SENTINEL.encode() in payload
        assert json.loads(payload.decode("utf-8"))["text"] == UTF8_SENTINEL

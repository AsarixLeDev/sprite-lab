from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.evaluation.memorization_review import (
    BOUND_REVIEW_SCHEMA_VERSION,
    canonical_sha256,
    replay_review_events,
    review_event_sha256,
)
from spritelab.evaluation.promotion_decision import (
    CANDIDATE_SCHEMA_VERSION,
    decide_promotion,
    decoded_rgba_sha256,
    file_sha256,
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _bundle(tmp_path: Path, *, outcome: str = "different_sprite", machine_pass: bool = True) -> dict[str, Any]:
    paths = {
        name: tmp_path / name
        for name in (
            "checkpoint.pt",
            "benchmark.json",
            "machine.json",
            "generated_report.json",
            "generated_manifest.json",
            "training_manifest.json",
            "candidate.json",
            "reviews.jsonl",
            "generated.png",
            "training.png",
        )
    }
    paths["checkpoint.pt"].write_bytes(b"checkpoint")
    _json(paths["benchmark.json"], {"schema_version": "benchmark_fixture_v1"})
    _json(paths["machine.json"], {"promotion": {"pass": machine_pass}})
    _json(paths["generated_report.json"], {"samples": ["sample-1"]})
    _json(paths["generated_manifest.json"], {"sample_ids": ["sample-1"]})
    _json(paths["training_manifest.json"], {"sprite_ids": ["train-1"]})
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(paths["generated.png"])
    Image.new("RGBA", (2, 2), (30, 20, 10, 255)).save(paths["training.png"])
    pair = {
        "pair_id": "sample-1__train-1",
        "generated_sample_id": "sample-1",
        "prompt_id": "prompt-1",
        "seed": 7,
        "noise_seed": 70,
        "generated_png_path": str(paths["generated.png"].resolve()),
        "generated_png_sha256": file_sha256(paths["generated.png"]),
        "generated_decoded_rgba_sha256": decoded_rgba_sha256(paths["generated.png"]),
        "training_image_path": str(paths["training.png"].resolve()),
        "training_source_sprite_id": "train-1",
        "training_row_or_index": 3,
        "training_decoded_rgba_sha256": decoded_rgba_sha256(paths["training.png"]),
        "evidence_class": "near_duplicate",
        "exact_rgba": False,
    }
    evidence = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "checkpoint_path": str(paths["checkpoint.pt"].resolve()),
        "checkpoint_sha256": file_sha256(paths["checkpoint.pt"]),
        "benchmark_manifest_path": str(paths["benchmark.json"].resolve()),
        "benchmark_manifest_sha256": file_sha256(paths["benchmark.json"]),
        "machine_report_path": str(paths["machine.json"].resolve()),
        "machine_report_sha256": file_sha256(paths["machine.json"]),
        "generated_report_path": str(paths["generated_report.json"].resolve()),
        "generated_report_sha256": file_sha256(paths["generated_report.json"]),
        "generated_manifest_path": str(paths["generated_manifest.json"].resolve()),
        "generated_manifest_sha256": file_sha256(paths["generated_manifest.json"]),
        "training_dataset_identity": "synthetic-training-v1",
        "training_manifest_path": str(paths["training_manifest.json"].resolve()),
        "training_manifest_sha256": file_sha256(paths["training_manifest.json"]),
        "detector_policy_version": "detector-v2",
        "comparison_method": "decoded_rgba_and_perceptual_v1",
        "comparison_parameters_sha256": "a" * 64,
        "pairs": [pair],
    }
    _json(paths["candidate.json"], evidence)
    event = _event(evidence, pair, outcome=outcome)
    paths["reviews.jsonl"].write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    return {"paths": paths, "evidence": evidence, "pair": pair, "event": event}


def _event(
    evidence: dict[str, Any],
    pair: dict[str, Any],
    *,
    outcome: str,
    revision: int = 1,
    previous: str | None = None,
    event_id: str = "event-1",
    legacy_toggle: bool | None = None,
) -> dict[str, Any]:
    event = {
        "schema_version": BOUND_REVIEW_SCHEMA_VERSION,
        "event_id": event_id,
        "pair_id": pair["pair_id"],
        "revision": revision,
        "previous_event_sha256": previous,
        "reviewer_id": "reviewer-fixture",
        "created_at_utc": "2026-07-12T10:00:00+00:00",
        "review_outcome": outcome,
        "human_note": "synthetic review",
        "checkpoint_path": evidence["checkpoint_path"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "benchmark_manifest_path": evidence["benchmark_manifest_path"],
        "benchmark_manifest_sha256": evidence["benchmark_manifest_sha256"],
        "generated_report_path": evidence["generated_report_path"],
        "generated_report_sha256": evidence["generated_report_sha256"],
        "generated_manifest_sha256": evidence["generated_manifest_sha256"],
        "generated_sample_id": pair["generated_sample_id"],
        "prompt_id": pair["prompt_id"],
        "seed": pair["seed"],
        "noise_seed": pair["noise_seed"],
        "generated_png_sha256": pair["generated_png_sha256"],
        "generated_decoded_rgba_sha256": pair["generated_decoded_rgba_sha256"],
        "training_dataset_identity": evidence["training_dataset_identity"],
        "training_manifest_path": evidence["training_manifest_path"],
        "training_manifest_sha256": evidence["training_manifest_sha256"],
        "training_source_sprite_id": pair["training_source_sprite_id"],
        "training_row_or_index": pair["training_row_or_index"],
        "training_decoded_rgba_sha256": pair["training_decoded_rgba_sha256"],
        "detector_policy_version": evidence["detector_policy_version"],
        "comparison_method": evidence["comparison_method"],
        "comparison_parameters_sha256": evidence["comparison_parameters_sha256"],
        "candidate_evidence_sha256": canonical_sha256(pair),
    }
    if legacy_toggle is not None:
        event["block_promotion"] = legacy_toggle
    event["event_sha256"] = review_event_sha256(event)
    return event


def _decide(bundle: dict[str, Any], *, expected_policy: str = "detector-v2") -> dict[str, Any]:
    paths = bundle["paths"]
    return decide_promotion(
        checkpoint=paths["checkpoint.pt"],
        benchmark_manifest=paths["benchmark.json"],
        machine_report=paths["machine.json"],
        generated_report=paths["generated_report.json"],
        generated_manifest=paths["generated_manifest.json"],
        training_dataset_identity="synthetic-training-v1",
        training_manifest=paths["training_manifest.json"],
        candidate_evidence=paths["candidate.json"],
        review_event_log=paths["reviews.jsonl"],
        detector_policy_version=expected_policy,
    )


def _rewrite_event(bundle: dict[str, Any], event: dict[str, Any]) -> None:
    event.pop("event_sha256", None)
    event["event_sha256"] = review_event_sha256(event)
    bundle["paths"]["reviews.jsonl"].write_text(json.dumps(event) + "\n", encoding="utf-8")


def _rewrite_evidence(bundle: dict[str, Any]) -> None:
    _json(bundle["paths"]["candidate.json"], bundle["evidence"])


def test_outcome_is_authoritative_over_legacy_toggle(tmp_path: Path) -> None:
    for index, (outcome, toggle, expected) in enumerate(
        (
            ("same_sprite_or_memorized", None, "blocked"),
            ("same_sprite_or_memorized", False, "blocked"),
            ("uncertain", True, "manual_review_required"),
            ("uncertain", None, "manual_review_required"),
            ("different_sprite", False, "eligible"),
        )
    ):
        root = tmp_path / str(index)
        root.mkdir()
        bundle = _bundle(root, outcome=outcome)
        event = _event(bundle["evidence"], bundle["pair"], outcome=outcome, legacy_toggle=toggle)
        _rewrite_event(bundle, event)
        assert _decide(bundle)["decision"] == expected


def test_missing_review_fails_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["paths"]["reviews.jsonl"].write_text("", encoding="utf-8")
    decision = _decide(bundle)
    assert decision["decision"] == "manual_review_required"
    assert decision["review_set_complete"] is False


def test_malformed_wrong_schema_and_unknown_outcome_are_not_comparable(tmp_path: Path) -> None:
    for index, content in enumerate(("{bad", json.dumps({"schema_version": "wrong", "pair_id": "x"}))):
        root = tmp_path / str(index)
        root.mkdir()
        bundle = _bundle(root)
        bundle["paths"]["reviews.jsonl"].write_text(content + "\n", encoding="utf-8")
        decision = _decide(bundle)
        assert decision["classification"] == "not_comparable"
        assert decision["decision"] == "blocked"
    root = tmp_path / "unknown"
    root.mkdir()
    bundle = _bundle(root)
    event = dict(bundle["event"], review_outcome="invented")
    _rewrite_event(bundle, event)
    assert _decide(bundle)["classification"] == "not_comparable"


def test_duplicate_revision_gap_and_invalid_chain_hash_block(tmp_path: Path) -> None:
    for name in ("duplicate", "gap", "hash"):
        root = tmp_path / name
        root.mkdir()
        bundle = _bundle(root)
        first = bundle["event"]
        if name == "duplicate":
            second = _event(bundle["evidence"], bundle["pair"], outcome="uncertain", event_id="event-2")
        elif name == "gap":
            second = _event(
                bundle["evidence"],
                bundle["pair"],
                outcome="different_sprite",
                revision=3,
                previous=review_event_sha256(first),
                event_id="event-3",
            )
        else:
            second = _event(
                bundle["evidence"],
                bundle["pair"],
                outcome="different_sprite",
                revision=2,
                previous="f" * 64,
                event_id="event-2",
            )
        bundle["paths"]["reviews.jsonl"].write_text(
            json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
        )
        decision = _decide(bundle)
        assert decision["decision"] == "blocked"
        assert decision["classification"] == "not_comparable"


def test_bound_input_file_hash_mismatches_are_not_comparable(tmp_path: Path) -> None:
    cases = (
        "checkpoint.pt",
        "benchmark.json",
        "generated_report.json",
        "generated.png",
        "training.png",
    )
    for index, name in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        bundle = _bundle(root)
        if name.endswith(".png"):
            Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(bundle["paths"][name])
        else:
            bundle["paths"][name].write_bytes(b"changed")
        decision = _decide(bundle)
        assert decision["decision"] == "blocked", name
        assert decision["identity_valid"] is False, name


def test_review_identity_hash_mismatches_are_not_comparable(tmp_path: Path) -> None:
    fields = (
        "checkpoint_sha256",
        "benchmark_manifest_sha256",
        "generated_report_sha256",
        "generated_png_sha256",
        "training_decoded_rgba_sha256",
    )
    for index, field in enumerate(fields):
        root = tmp_path / str(index)
        root.mkdir()
        bundle = _bundle(root)
        event = dict(bundle["event"])
        event[field] = "0" * 64
        _rewrite_event(bundle, event)
        assert _decide(bundle)["classification"] == "not_comparable", field


def test_detector_policy_mismatch_is_not_comparable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    decision = _decide(bundle, expected_policy="detector-v3")
    assert decision["decision"] == "blocked"
    assert "detector policy version is incompatible" in decision["not_comparable_reasons"]


def test_candidate_and_review_sets_must_match_exactly(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    bundle = _bundle(missing)
    bundle["paths"]["reviews.jsonl"].write_text("", encoding="utf-8")
    assert _decide(bundle)["pending_pairs"] == ["sample-1__train-1"]

    extra = tmp_path / "extra"
    extra.mkdir()
    bundle = _bundle(extra)
    extra_pair = dict(bundle["pair"], pair_id="sample-x__train-x")
    extra_event = _event(bundle["evidence"], extra_pair, outcome="different_sprite", event_id="event-x")
    bundle["paths"]["reviews.jsonl"].write_text(
        json.dumps(bundle["event"]) + "\n" + json.dumps(extra_event) + "\n", encoding="utf-8"
    )
    decision = _decide(bundle)
    assert decision["classification"] == "not_comparable"
    assert any("absent from current candidate set" in reason for reason in decision["not_comparable_reasons"])


def test_machine_gate_failure_cannot_be_cleared_by_human(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, outcome="different_sprite", machine_pass=False)
    decision = _decide(bundle)
    assert decision["decision"] == "blocked"
    assert decision["machine_gates_passed"] is False
    assert "existing machine-gate failure" in decision["hard_block_reasons"]


def test_same_or_memorized_blocks_when_machine_gates_pass(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, outcome="same_sprite_or_memorized", machine_pass=True)
    decision = _decide(bundle)
    assert decision["decision"] == "blocked"
    assert decision["blocked_pairs"] == ["sample-1__train-1"]


def test_all_valid_and_conclusively_cleared_is_eligible_and_deterministic(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    first = _decide(bundle)
    second = _decide(bundle)
    assert first == second
    assert first["decision"] == "eligible"
    assert first["eligible_for_promotion"] is True
    assert first["review_set_complete"] is True


def test_nontrivial_exact_rgba_is_an_automatic_hard_block(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    pair = bundle["pair"]
    pair["exact_rgba"] = True
    pair["evidence_class"] = "exact_decoded_rgba"
    bundle["evidence"]["pairs"] = [pair]
    _rewrite_evidence(bundle)
    _rewrite_event(bundle, _event(bundle["evidence"], pair, outcome="different_sprite"))
    decision = _decide(bundle)
    assert decision["decision"] == "blocked"
    assert any("exact decoded-RGBA" in reason for reason in decision["hard_block_reasons"])


def test_historical_v1_is_display_only_and_cannot_authorize(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    legacy = {
        "schema_version": "memorization_review_v1.0",
        "pair_id": bundle["pair"]["pair_id"],
        "revision": 1,
        "classification": "likely_false_positive",
        "block_promotion": False,
    }
    bundle["paths"]["reviews.jsonl"].write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    replay = replay_review_events(bundle["paths"]["reviews.jsonl"])
    assert replay.legacy_events[0]["promotion_authority"] is False
    assert replay.legacy_events[0]["identity_status"] == "unbound_legacy"
    decision = _decide(bundle)
    assert decision["decision"] == "blocked"
    assert decision["classification"] == "not_comparable"

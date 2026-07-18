from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.evaluation.candidate_bundle import CANDIDATE_CONTRACT_VERSION
from spritelab.evaluation.memorization import (
    COMPARISON_METHOD,
    COMPARISON_PARAMETERS_SHA256,
    DETECTOR_POLICY_SHA256,
    DETECTOR_POLICY_VERSION,
    REVIEW_REQUIRED_EVIDENCE_CLASSES,
    detector_policy_record,
    recompute_memorization_status,
)
from spritelab.evaluation.memorization_review import (
    BOUND_REVIEW_GENESIS_SHA256,
    BOUND_REVIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    append_bound_review_event,
    bound_event_identity,
    load_bound_review_tasks,
    replay_review_events,
    review_event_sha256,
)
from spritelab.evaluation.promotion_decision import (
    CANDIDATE_SCHEMA_VERSION,
    candidate_bundle_sha256,
    decide_promotion,
    decoded_rgba_sha256,
    file_sha256,
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, evidence_class: str = "exact_alpha_review_required") -> dict[str, Any]:
    generated_root = tmp_path / "generated_images"
    training_root = tmp_path / "training_images"
    generated_root.mkdir()
    training_root.mkdir()
    paths = {
        name: tmp_path / name
        for name in (
            "checkpoint.pt",
            "benchmark.json",
            "machine.json",
            "generated_report.json",
            "generated_manifest.json",
            "training_manifest.json",
            "policy.json",
            "candidate.json",
            "reviews.jsonl",
        )
    }
    paths["generated.png"] = generated_root / "generated.png"
    paths["training.png"] = training_root / "training.png"
    paths["checkpoint.pt"].write_bytes(b"checkpoint")
    _json(paths["benchmark.json"], {"cases": [{"id": "prompt-1"}], "seeds": [7]})
    _json(paths["generated_report.json"], {"samples": ["sample-1"]})
    _json(paths["policy.json"], detector_policy_record())
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(paths["generated.png"])
    Image.new("RGBA", (8, 8), (30, 20, 10, 255)).save(paths["training.png"])
    pair = {
        "pair_id": "sample-1__train-1",
        "generated_sample_id": "sample-1",
        "prompt_id": "prompt-1",
        "seed": 7,
        "noise_seed": 70,
        "generated_png_path": str(paths["generated.png"].resolve()),
        "training_dataset_identity": "synthetic-training-v1",
        "training_view_identity": "training-view-v1",
        "training_source_sprite_id": "train-1",
        "training_row_or_index": 3,
        "training_image_path": str(paths["training.png"].resolve()),
        "evidence_class": evidence_class,
        "exact_rgba": evidence_class == "exact_rgba_nontrivial",
        "evidence_metrics": {"alpha_iou": 1.0, "union_rgba_distance": 0.1},
        "evidence_diagnostics": {"generated_foreground_pixels": 64, "training_foreground_pixels": 64},
    }
    bundle = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_images_root": str(generated_root.resolve()),
        "training_images_root": str(training_root.resolve()),
        "training_dataset_identity": "synthetic-training-v1",
        "training_view_identity": "training-view-v1",
        "detector_policy_version": DETECTOR_POLICY_VERSION,
        "detector_policy_sha256": DETECTOR_POLICY_SHA256,
        "comparison_method": COMPARISON_METHOD,
        "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
        "candidate_count": 1,
        "candidate_order": [pair["pair_id"]],
        "pairs": [pair],
    }
    state = {"paths": paths, "bundle": bundle, "pair": pair, "outcome": "different_sprite"}
    _finalize(state)
    return state


def _finalize(state: dict[str, Any]) -> None:
    paths = state["paths"]
    bundle = state["bundle"]
    pairs = bundle["pairs"]
    for pair in pairs:
        pair["generated_png_sha256"] = file_sha256(Path(pair["generated_png_path"]))
        pair["generated_decoded_rgba_sha256"] = decoded_rgba_sha256(Path(pair["generated_png_path"]))
        pair["training_source_blob_path"] = pair["training_image_path"]
        pair["training_source_blob_sha256"] = file_sha256(Path(pair["training_image_path"]))
        pair["training_decoded_rgba_sha256"] = decoded_rgba_sha256(Path(pair["training_image_path"]))
    generated_records = list(
        {
            pair["generated_sample_id"]: {
                "sample_id": pair["generated_sample_id"],
                "image_path": pair["generated_png_path"],
                "png_sha256": pair["generated_png_sha256"],
                "decoded_rgba_sha256": pair["generated_decoded_rgba_sha256"],
            }
            for pair in pairs
        }.values()
    )
    training_records = [
        {
            "dataset_identity": pair["training_dataset_identity"],
            "view_identity": pair["training_view_identity"],
            "source_sprite_id": pair["training_source_sprite_id"],
            "row_or_index": pair["training_row_or_index"],
            "image_path": pair["training_image_path"],
            "source_blob_sha256": pair["training_source_blob_sha256"],
            "decoded_rgba_sha256": pair["training_decoded_rgba_sha256"],
        }
        for pair in pairs
    ]
    _json(paths["generated_manifest.json"], {"samples": generated_records})
    _json(paths["training_manifest.json"], {"records": training_records})
    for pair in pairs:
        pair["training_manifest_sha256"] = file_sha256(paths["training_manifest.json"])
    classes = [pair["evidence_class"] for pair in pairs]
    status = recompute_memorization_status(classes).value
    review_classes = {item.value for item in REVIEW_REQUIRED_EVIDENCE_CLASSES}
    memo = {
        "candidate_pair_ids": [pair["pair_id"] for pair in pairs],
        "candidate_count": len(pairs),
        "hard_evidence_count": sum(pair["evidence_class"] == "exact_rgba_nontrivial" for pair in pairs),
        "review_required_count": sum(pair["evidence_class"] in review_classes for pair in pairs),
        "warning_count": sum(
            pair["evidence_class"]
            in {"exact_rgba_low_evidence_collision", "generic_sparse_collision", "blank_collision"}
            for pair in pairs
        ),
        "evidence_class_counts": dict(Counter(classes)),
        "unresolved_candidate_count": sum(
            pair["evidence_class"] == "exact_rgba_nontrivial" or pair["evidence_class"] in review_classes
            for pair in pairs
        ),
        "machine_status": status,
        "detector_policy_sha256": DETECTOR_POLICY_SHA256,
        "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
    }
    machine = {
        "summary": {"sample_count": len(pairs), "memorization": memo},
        "promotion": {
            "pass": status == "pass",
            "memorization_machine_status": status,
            "checks": {
                "memorization_hard_evidence": memo["hard_evidence_count"] == 0,
                "memorization_reviews_resolved": memo["review_required_count"] == 0,
                "malformed": True,
                "palette": True,
            },
        },
    }
    _json(paths["machine.json"], machine)
    bindings = {
        "checkpoint": "checkpoint.pt",
        "benchmark_manifest": "benchmark.json",
        "machine_report": "machine.json",
        "generated_report": "generated_report.json",
        "generated_manifest": "generated_manifest.json",
        "training_manifest": "training_manifest.json",
        "detector_policy_artifact": "policy.json",
    }
    for prefix, name in bindings.items():
        bundle[f"{prefix}_path"] = str(paths[name].resolve())
        bundle[f"{prefix}_sha256"] = file_sha256(paths[name])
    bundle["contract_version"] = CANDIDATE_CONTRACT_VERSION
    bundle["review_log_contract"] = {
        "schema_version": "sprite_lab_review_log_contract_v2",
        "absence_allowed_before_first_signed_review": True,
        "absence_allowed_when_no_review_candidates": True,
    }
    identity_inputs = {
        field: bundle[field]
        for field in (
            "training_dataset_identity",
            "training_view_identity",
            "checkpoint_sha256",
            "benchmark_manifest_sha256",
            "machine_report_sha256",
            "generated_manifest_sha256",
            "training_manifest_sha256",
            "detector_policy_sha256",
            "comparison_parameters_sha256",
        )
    }
    pair_bindings = (
        "training_dataset_identity",
        "training_view_identity",
        "checkpoint_path",
        "checkpoint_sha256",
        "benchmark_manifest_path",
        "benchmark_manifest_sha256",
        "generated_manifest_path",
        "generated_manifest_sha256",
        "training_manifest_path",
        "training_manifest_sha256",
        "detector_policy_version",
        "detector_policy_sha256",
        "comparison_method",
        "comparison_parameters_sha256",
    )
    for pair in pairs:
        for field in pair_bindings:
            pair[field] = bundle[field]
        pair["candidate_bundle_identity_inputs"] = dict(identity_inputs)
    bundle["candidate_count"] = len(pairs)
    bundle["candidate_order"] = [pair["pair_id"] for pair in pairs]
    bundle["candidate_evidence_sha256"] = candidate_bundle_sha256(bundle)
    _json(paths["candidate.json"], bundle)
    events = []
    if state.get("outcome") is not None:
        for pair in pairs:
            if pair["evidence_class"] in review_classes:
                events.append(_event(bundle, pair, state["outcome"]))
    paths["reviews.jsonl"].write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8"
    )


def _event(bundle: dict[str, Any], pair: dict[str, Any], outcome: str) -> dict[str, Any]:
    event = {
        "schema_version": BOUND_REVIEW_SCHEMA_VERSION,
        "event_id": f"event-{pair['pair_id']}",
        "pair_id": pair["pair_id"],
        "revision": 1,
        "previous_event_sha256": BOUND_REVIEW_GENESIS_SHA256,
        "reviewer_id": "reviewer-fixture",
        "created_at_utc": "2026-07-12T10:00:00+00:00",
        "review_outcome": outcome,
        "human_note": "synthetic review",
        **bound_event_identity(bundle, pair),
    }
    event["event_sha256"] = review_event_sha256(event)
    return event


def _decide(state: dict[str, Any]) -> dict[str, Any]:
    paths = state["paths"]
    return decide_promotion(
        checkpoint=paths["checkpoint.pt"],
        benchmark_manifest=paths["benchmark.json"],
        machine_report=paths["machine.json"],
        generated_report=paths["generated_report.json"],
        generated_manifest=paths["generated_manifest.json"],
        generated_images=paths["generated.png"].parent,
        training_dataset_identity="synthetic-training-v1",
        training_view_identity="training-view-v1",
        training_manifest=paths["training_manifest.json"],
        training_images=paths["training.png"].parent,
        candidate_evidence=paths["candidate.json"],
        review_event_log=paths["reviews.jsonl"],
        detector_policy=paths["policy.json"],
        detector_policy_version=DETECTOR_POLICY_VERSION,
    )


def test_review_resolution_matrix(tmp_path: Path) -> None:
    for index, (outcome, expected) in enumerate(
        (
            (None, "manual_review_required"),
            ("different_sprite", "eligible"),
            ("common_generic_shape", "eligible"),
            ("likely_false_positive", "eligible"),
            ("same_sprite_or_memorized", "blocked"),
            ("uncertain", "manual_review_required"),
        )
    ):
        root = tmp_path / str(index)
        root.mkdir()
        state = _fixture(root)
        state["outcome"] = outcome
        _finalize(state)
        assert _decide(state)["decision"] == expected


def test_hard_and_warning_evidence_semantics(tmp_path: Path) -> None:
    hard_root = tmp_path / "hard"
    hard_root.mkdir()
    hard = _fixture(hard_root, "exact_rgba_nontrivial")
    assert _decide(hard)["decision"] == "blocked"
    warning_root = tmp_path / "warning"
    warning_root.mkdir()
    warning = _fixture(warning_root, "generic_sparse_collision")
    decision = _decide(warning)
    assert decision["decision"] == "eligible"
    assert decision["review_set_complete"] is True


def test_unknown_class_policy_and_bundle_tampering_fail_closed(tmp_path: Path) -> None:
    for name in ("unknown", "policy", "removed", "metrics"):
        root = tmp_path / name
        root.mkdir()
        state = _fixture(root)
        if name == "unknown":
            state["pair"]["evidence_class"] = "invented"
            _json(state["paths"]["candidate.json"], state["bundle"])
        elif name == "policy":
            policy = detector_policy_record()
            policy["thresholds"]["minimum_foreground_pixels"] = 17
            _json(state["paths"]["policy.json"], policy)
        elif name == "removed":
            state["bundle"]["pairs"] = []
            _json(state["paths"]["candidate.json"], state["bundle"])
        else:
            state["pair"]["evidence_metrics"]["alpha_iou"] = 0.5
            _json(state["paths"]["candidate.json"], state["bundle"])
        decision = _decide(state)
        assert decision["classification"] == "not_comparable", name


def test_machine_recomputation_and_source_image_tampering(tmp_path: Path) -> None:
    for name in ("stored_pass", "training_blob", "generated_png"):
        root = tmp_path / name
        root.mkdir()
        state = _fixture(root, "exact_rgba_nontrivial" if name == "stored_pass" else "exact_alpha_review_required")
        if name == "stored_pass":
            machine = json.loads(state["paths"]["machine.json"].read_text(encoding="utf-8"))
            machine["promotion"]["memorization_machine_status"] = "pass"
            machine["summary"]["memorization"]["machine_status"] = "pass"
            _json(state["paths"]["machine.json"], machine)
            state["bundle"]["machine_report_sha256"] = file_sha256(state["paths"]["machine.json"])
            state["bundle"]["candidate_evidence_sha256"] = candidate_bundle_sha256(state["bundle"])
            _json(state["paths"]["candidate.json"], state["bundle"])
        elif name == "training_blob":
            Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(state["paths"]["training.png"])
        else:
            Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(state["paths"]["generated.png"])
        decision = _decide(state)
        assert decision["classification"] == "not_comparable", name


def test_reordering_is_order_sensitive_and_deterministic(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    second = dict(state["pair"], pair_id="sample-1__train-2", training_source_sprite_id="train-2")
    state["bundle"]["pairs"] = [state["pair"], second]
    _finalize(state)
    first_hash = state["bundle"]["candidate_evidence_sha256"]
    state["bundle"]["pairs"].reverse()
    _finalize(state)
    second_hash = state["bundle"]["candidate_evidence_sha256"]
    assert first_hash != second_hash
    assert second_hash == candidate_bundle_sha256(state["bundle"])


def test_legacy_v1_review_cannot_authorize(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    legacy = {"schema_version": SCHEMA_VERSION, "pair_id": state["pair"]["pair_id"], "revision": 1}
    state["paths"]["reviews.jsonl"].write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    decision = _decide(state)
    assert decision["decision"] == "blocked"
    assert decision["classification"] == "not_comparable"


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _second_event(first: dict[str, Any], *, revision: int = 2, outcome: str = "different_sprite") -> dict[str, Any]:
    event = {
        **first,
        "event_id": f"{first['event_id']}-revision-{revision}",
        "revision": revision,
        "previous_event_sha256": first["event_sha256"],
        "review_outcome": outcome,
        "human_note": f"revision {revision}",
    }
    event["event_sha256"] = review_event_sha256(event)
    return event


def test_removed_or_stale_hash_cannot_change_blocking_event_to_clearing(tmp_path: Path) -> None:
    for mode in ("removed", "stale"):
        root = tmp_path / mode
        root.mkdir()
        state = _fixture(root)
        state["outcome"] = "same_sprite_or_memorized"
        _finalize(state)
        event = json.loads(state["paths"]["reviews.jsonl"].read_text(encoding="utf-8"))
        event["review_outcome"] = "different_sprite"
        if mode == "removed":
            event.pop("event_sha256")
        _write_events(state["paths"]["reviews.jsonl"], [event])
        decision = _decide(state)
        assert decision["classification"] == "not_comparable"
        assert decision["eligible_for_promotion"] is False


def test_invalid_later_event_preserves_but_cannot_use_latest_valid_revision(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    first = _event(state["bundle"], state["pair"], "same_sprite_or_memorized")
    second = _second_event(first)
    second.pop("event_sha256")
    _write_events(state["paths"]["reviews.jsonl"], [first, second])
    replay = replay_review_events(state["paths"]["reviews.jsonl"])
    chain = replay.chains[state["pair"]["pair_id"]]
    assert chain.chain_status == "invalid"
    assert chain.latest_valid_revision == 1
    assert chain.authoritative_event == first
    assert state["pair"]["pair_id"] not in replay.current


def test_signed_clearing_and_blocking_events_are_authoritative(tmp_path: Path) -> None:
    for index, (outcome, expected) in enumerate(
        (("different_sprite", "eligible"), ("same_sprite_or_memorized", "blocked"))
    ):
        root = tmp_path / str(index)
        root.mkdir()
        state = _fixture(root)
        state["outcome"] = outcome
        _finalize(state)
        decision = _decide(state)
        assert decision["decision"] == expected
        assert decision["review_chain_statuses"] == {state["pair"]["pair_id"]: "valid"}


def test_revision_gap_and_changed_previous_hash_fail_closed(tmp_path: Path) -> None:
    for mode, expected_status in (("gap", "incomplete"), ("previous", "invalid")):
        root = tmp_path / mode
        root.mkdir()
        state = _fixture(root)
        first = _event(state["bundle"], state["pair"], "uncertain")
        second = _second_event(first, revision=3 if mode == "gap" else 2)
        if mode == "previous":
            second["previous_event_sha256"] = "f" * 64
            second["event_sha256"] = review_event_sha256(second)
        _write_events(state["paths"]["reviews.jsonl"], [first, second])
        decision = _decide(state)
        assert decision["classification"] == "not_comparable"
        assert decision["review_chain_statuses"][state["pair"]["pair_id"]] == expected_status


def test_duplicate_or_competing_same_revision_is_contradictory(tmp_path: Path) -> None:
    for mode in ("duplicate", "competing"):
        root = tmp_path / mode
        root.mkdir()
        state = _fixture(root)
        first = _event(state["bundle"], state["pair"], "uncertain")
        competing = {**first, "event_id": f"{first['event_id']}-{mode}"}
        if mode == "competing":
            competing["review_outcome"] = "different_sprite"
            competing["human_note"] = "competing result"
        competing["event_sha256"] = review_event_sha256(competing)
        _write_events(state["paths"]["reviews.jsonl"], [first, competing])
        replay = replay_review_events(
            state["paths"]["reviews.jsonl"],
            expected_identities={state["pair"]["pair_id"]: bound_event_identity(state["bundle"], state["pair"])},
        )
        chain = replay.chains[state["pair"]["pair_id"]]
        assert chain.chain_status == "contradictory"
        assert chain.pending_review is True


def test_mutating_any_signed_event_body_identity_invalidates_hash(tmp_path: Path) -> None:
    mutations = {
        "human_note": "changed note",
        "review_outcome": "same_sprite_or_memorized",
        "pair_id": "changed-pair",
        "generated_decoded_rgba_sha256": "1" * 64,
        "training_decoded_rgba_sha256": "2" * 64,
    }
    for field, value in mutations.items():
        root = tmp_path / field
        root.mkdir()
        state = _fixture(root)
        event = json.loads(state["paths"]["reviews.jsonl"].read_text(encoding="utf-8"))
        event[field] = value
        _write_events(state["paths"]["reviews.jsonl"], [event])
        decision = _decide(state)
        assert decision["classification"] == "not_comparable", field


def test_all_review_required_classes_are_in_v2_queue_and_reasons_coalesce(tmp_path: Path) -> None:
    observed: set[str] = set()
    for index, evidence_class in enumerate(
        (
            "exact_alpha_review_required",
            "translation_alpha_review_required",
            "near_pixel_review_required",
        )
    ):
        root = tmp_path / str(index)
        root.mkdir()
        state = _fixture(root, evidence_class)
        state["pair"]["evidence_reasons"] = [evidence_class, evidence_class]
        _finalize(state)
        _, tasks = load_bound_review_tasks(state["paths"]["candidate.json"])
        assert len(tasks) == 1
        assert len(tasks[0].evidence_reasons) == 1
        observed.update(tasks[0].evidence_classes)
    assert observed == {
        "exact_alpha_review_required",
        "translation_alpha_review_required",
        "near_pixel_review_required",
    }


def test_v2_authoring_appends_a_signed_event(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    state["outcome"] = None
    _finalize(state)
    event = append_bound_review_event(
        state["paths"]["candidate.json"],
        state["paths"]["reviews.jsonl"],
        pair_id=state["pair"]["pair_id"],
        review_outcome="different_sprite",
        reviewer_id="reviewer-test",
        human_note="reviewed through v2 backend",
        created_at_utc="2026-07-13T10:00:00+00:00",
    )
    assert event["event_sha256"] == review_event_sha256(event)
    assert event["previous_event_sha256"] == BOUND_REVIEW_GENESIS_SHA256
    replay = replay_review_events(state["paths"]["reviews.jsonl"])
    assert replay.chains[state["pair"]["pair_id"]].chain_status == "valid"


def test_duplicate_review_outcome_is_never_authoritative(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    state["outcome"] = None
    _finalize(state)
    append_bound_review_event(
        state["paths"]["candidate.json"],
        state["paths"]["reviews.jsonl"],
        pair_id=state["pair"]["pair_id"],
        review_outcome="different_sprite",
        reviewer_id="reviewer-test",
        created_at_utc="2026-07-13T10:00:00+00:00",
    )
    review_path = state["paths"]["reviews.jsonl"]
    payload = review_path.read_text(encoding="utf-8")
    marker = '"review_outcome":"different_sprite"'
    if marker not in payload:
        marker = '"review_outcome": "different_sprite"'
    review_path.write_text(payload.replace(marker, '"review_outcome":"blocked",' + marker, 1), encoding="utf-8")

    replay = replay_review_events(review_path)

    assert replay.global_invalid_events
    assert state["pair"]["pair_id"] not in replay.chains


def test_missing_empty_malformed_and_disappeared_review_logs_are_controlled(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = _fixture(missing_root)
    missing["paths"]["reviews.jsonl"].unlink()
    missing_decision = _decide(missing)
    assert missing_decision["decision"] == "manual_review_required"
    assert missing_decision["review_set_complete"] is False
    assert any("missing_review_log" in reason for reason in missing_decision["pending_review_reasons"])

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = _fixture(empty_root)
    empty["outcome"] = None
    _finalize(empty)
    assert _decide(empty)["decision"] == "manual_review_required"

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    malformed = _fixture(malformed_root)
    malformed["paths"]["reviews.jsonl"].write_text("{not-json}\n", encoding="utf-8")
    assert _decide(malformed)["classification"] == "not_comparable"

    disappeared_root = tmp_path / "disappeared"
    disappeared_root.mkdir()
    disappeared = _fixture(disappeared_root)
    disappeared["outcome"] = None
    _finalize(disappeared)
    disappeared["bundle"]["review_event_log_path"] = str(disappeared["paths"]["reviews.jsonl"].resolve())
    disappeared["bundle"]["review_event_log_sha256"] = file_sha256(disappeared["paths"]["reviews.jsonl"])
    disappeared["bundle"]["candidate_evidence_sha256"] = candidate_bundle_sha256(disappeared["bundle"])
    _json(disappeared["paths"]["candidate.json"], disappeared["bundle"])
    disappeared["paths"]["reviews.jsonl"].unlink()
    disappeared_decision = _decide(disappeared)
    assert disappeared_decision["classification"] == "not_comparable"
    assert "bound review log disappeared" in disappeared_decision["not_comparable_reasons"]


def test_hard_evidence_cannot_be_review_cleared(tmp_path: Path) -> None:
    state = _fixture(tmp_path, "exact_rgba_nontrivial")
    event = _event(state["bundle"], state["pair"], "different_sprite")
    _write_events(state["paths"]["reviews.jsonl"], [event])
    decision = _decide(state)
    assert decision["decision"] == "blocked"
    assert decision["eligible_for_promotion"] is False


def test_all_cleared_pairs_require_every_unrelated_gate(tmp_path: Path) -> None:
    state = _fixture(tmp_path)
    second = deepcopy(state["pair"])
    second["pair_id"] = "sample-1__train-2"
    second["training_source_sprite_id"] = "train-2"
    state["bundle"]["pairs"] = [state["pair"], second]
    _finalize(state)
    assert _decide(state)["decision"] == "eligible"

    machine = json.loads(state["paths"]["machine.json"].read_text(encoding="utf-8"))
    machine["promotion"]["checks"]["palette"] = False
    _json(state["paths"]["machine.json"], machine)
    state["bundle"]["machine_report_sha256"] = file_sha256(state["paths"]["machine.json"])
    state["bundle"]["candidate_evidence_sha256"] = candidate_bundle_sha256(state["bundle"])
    _json(state["paths"]["candidate.json"], state["bundle"])
    _write_events(
        state["paths"]["reviews.jsonl"],
        [_event(state["bundle"], pair, "different_sprite") for pair in state["bundle"]["pairs"]],
    )
    decision = _decide(state)
    assert decision["decision"] == "blocked"
    assert decision["eligible_for_promotion"] is False

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from spritelab.evaluation.candidate_bundle import (
    CANDIDATE_SCHEMA_VERSION,
    candidate_bundle_sha256,
    file_sha256,
    load_candidate_bundle,
)
from spritelab.evaluation.cli import main as evaluation_cli
from spritelab.evaluation.memorization import resolve_training_context_identities
from spritelab.evaluation.memorization_review import (
    BOUND_REVIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    append_bound_review_event,
    review_event_sha256,
)
from spritelab.evaluation.promotion_decision import decide_promotion
from spritelab.evaluation.suite import score_suite
from spritelab.product_core import ProjectContext
from spritelab.product_features.dataset.plugin import create_plugin as create_dataset_plugin
from spritelab.product_features.dataset.web import build_review_router, discover_memorization_review
from spritelab.product_features.evaluation.memorization_display import (
    INCOMPLETE_EVIDENCE_MESSAGE,
    INVALID_EVIDENCE_MESSAGE,
    memorization_display,
)
from spritelab.product_web.app import create_app
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus
from spritelab.v3.status import (
    MEMORIZATION_AUDIT_BOUND_FILES,
    _memorization_audit_status,
    memorization_audit_code_identity,
)


def _rgba(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    value = np.zeros((32, 32, 4), dtype=np.uint8)
    value[mask, :3] = color
    value[mask, 3] = 255
    return value


def _base_mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[7:24, 8:23] = True
    mask[7:10, 8:12] = False
    mask[19:24, 19:23] = False
    mask[13:16, 14:17] = False
    return mask


def _production_run(root: Path, kind: str = "exact_alpha") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.pt"
    benchmark = root / "benchmark.json"
    checkpoint.write_bytes(b"synthetic-checkpoint-identity-only")
    benchmark.write_text(json.dumps({"cases": [{"id": "prompt-1"}], "seeds": [7]}), encoding="utf-8")

    dataset = root / "training"
    dataset.mkdir()
    training_mask = _base_mask()
    if kind == "warning":
        training_mask = np.zeros((32, 32), dtype=bool)
        training_mask[12:14, 12:14] = True
    training_rgba = _rgba(training_mask, (20, 40, 60))
    alpha = (training_rgba[..., 3] > 0)[None, ...].astype(np.uint8)
    index_map = np.zeros((1, 32, 32), dtype=np.uint8)
    palette = np.array([[[20, 40, 60]]], dtype=np.uint8)
    palette_mask = np.ones((1, 1), dtype=bool)
    npz_path = dataset / "train.npz"
    np.savez(npz_path, alpha=alpha, index_map=index_map, palette=palette, palette_mask=palette_mask)
    training_manifest = dataset / "training_manifest.jsonl"
    training_manifest.write_text(
        json.dumps(
            {
                "sprite_id": "training-sprite-1",
                "npz_file": "train.npz",
                "npz_row": 0,
                "dataset_identity": "synthetic-training-v1",
                "view_identity": "synthetic-training-view-v1",
                "source": {"dataset_dir": str(dataset.resolve())},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    generated_mask = training_mask.copy()
    generated_color = (22, 42, 62)
    if kind == "translation_alpha":
        generated_mask = np.roll(training_mask, 1, axis=1)
    elif kind == "near_pixel":
        generated_mask[12, 13] = False
    elif kind == "hard":
        generated_color = (20, 40, 60)
    elif kind == "warning":
        generated_color = (20, 40, 60)
    elif kind == "none":
        generated_mask = np.zeros((32, 32), dtype=bool)
        generated_mask[5:12, 5:25] = True
        generated_mask[20:27, 5:25] = True
        generated_color = (180, 30, 120)
    generated_rgba = _rgba(generated_mask, generated_color)
    generated = root / "generated"
    generated.mkdir()
    image = generated / "sample-1.png"
    Image.fromarray(generated_rgba, "RGBA").save(image)
    generated_manifest = generated / "generated_manifest.jsonl"
    generated_manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "prompt_id": "prompt-1",
                "prompt": "synthetic sprite",
                "checkpoint": str(checkpoint.resolve()),
                "seed": 7,
                "noise_seed": 70,
                "steps": 20,
                "cfg_scale": 3.0,
                "model_output_finite": True,
                "image": str(image.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = root / "metrics"
    report = score_suite(
        generated,
        metrics,
        training_manifests=[training_manifest],
        checkpoint=checkpoint,
        benchmark_manifest=benchmark,
        training_dataset_identity="synthetic-training-v1",
        training_view_identity="synthetic-training-view-v1",
    )
    candidate = metrics / "candidate_evidence.json"
    bundle = json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else {}
    return {
        "root": root,
        "checkpoint": checkpoint,
        "benchmark": benchmark,
        "generated": generated,
        "generated_manifest": generated_manifest,
        "training_manifest": training_manifest,
        "metrics": metrics,
        "machine_report": metrics / "summary.json",
        "generated_report": metrics / "per_image_metrics.jsonl",
        "policy": metrics / "detector_policy.json",
        "candidate": candidate,
        "reviews": metrics / "review_events.jsonl",
        "report": report,
        "bundle": bundle,
    }


def _decision(state: dict[str, Any]) -> dict[str, Any]:
    bundle = json.loads(state["candidate"].read_text(encoding="utf-8"))
    return decide_promotion(
        checkpoint=state["checkpoint"],
        benchmark_manifest=state["benchmark"],
        machine_report=state["machine_report"],
        generated_report=state["generated_report"],
        generated_manifest=state["generated_manifest"],
        generated_images=Path(bundle["generated_images_root"]),
        training_dataset_identity="synthetic-training-v1",
        training_view_identity="synthetic-training-view-v1",
        training_manifest=state["training_manifest"],
        training_images=Path(bundle["training_images_root"]),
        candidate_evidence=state["candidate"],
        review_event_log=state["reviews"],
        detector_policy=state["policy"],
    )


def _rewrite_bundle(state: dict[str, Any], mutate: Any, *, rehash: bool = True) -> None:
    bundle = json.loads(state["candidate"].read_text(encoding="utf-8"))
    mutate(bundle)
    if rehash:
        bundle["candidate_evidence_sha256"] = candidate_bundle_sha256(bundle)
    state["candidate"].write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")


def _context(state: dict[str, Any], *, include_view_identity: bool = True) -> ProjectContext:
    evaluation = {
        "candidate_evidence": str(state["candidate"]),
        "review_log": str(state["reviews"]),
        "checkpoint": str(state["checkpoint"]),
        "benchmark": str(state["benchmark"]),
        "dataset_identity": "synthetic-training-v1",
    }
    if include_view_identity:
        evaluation["training_view_identity"] = "synthetic-training-view-v1"
    config = {"evaluation": evaluation}
    return ProjectContext(state["root"], config=config, runs_directory=state["root"] / "runs")


@pytest.mark.parametrize(
    ("kind", "expected_class"),
    (
        ("exact_alpha", "exact_alpha_review_required"),
        ("translation_alpha", "translation_alpha_review_required"),
        ("near_pixel", "near_pixel_review_required"),
    ),
)
def test_normal_evaluation_writes_all_review_classes(tmp_path: Path, kind: str, expected_class: str) -> None:
    state = _production_run(tmp_path / kind, kind)
    assert state["bundle"]["schema_version"] == CANDIDATE_SCHEMA_VERSION
    assert state["bundle"]["pairs"][0]["evidence_class"] == expected_class
    assert load_candidate_bundle(state["candidate"]).valid is True


def test_bundle_hash_covers_complete_ordered_bundle(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    assert state["bundle"]["candidate_evidence_sha256"] == candidate_bundle_sha256(state["bundle"])
    assert state["bundle"]["candidate_order"] == [state["bundle"]["pairs"][0]["pair_id"]]


def test_self_hashed_training_view_tamper_is_not_comparable(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    _rewrite_bundle(
        state,
        lambda bundle: bundle["pairs"][0].update(training_view_identity="wrong-training-view"),
    )

    validation = load_candidate_bundle(state["candidate"])
    assert validation.valid is False
    assert any("training_view_identity binding mismatch" in reason for reason in validation.reasons)

    decision = _decision(state)
    assert decision["classification"] == "not_comparable"
    assert decision["identity_valid"] is False
    assert decision["eligible_for_promotion"] is False
    assert decision["checkpoint_copies"] == 0
    assert decision["promotion_actions"] == 0


@pytest.mark.parametrize("malformed", (" padded ", 7, True))
def test_malformed_training_context_identities_fail_closed(malformed: Any) -> None:
    with pytest.raises(ValueError, match="malformed identity"):
        resolve_training_context_identities(
            dataset_identities=[malformed],
            view_identities=["view-v1"],
            manifest_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="explicit training view identity is malformed"):
        resolve_training_context_identities(
            dataset_identities=["dataset-v1"],
            view_identities=["view-v1"],
            manifest_sha256="a" * 64,
            explicit_view_identity=malformed,
        )


def test_self_hashed_bundle_view_tamper_is_rejected_by_independent_context(tmp_path: Path) -> None:
    state = _production_run(tmp_path)

    def change_view(bundle: dict[str, Any]) -> None:
        bundle["training_view_identity"] = "wrong-training-view"
        pair = bundle["pairs"][0]
        pair["training_view_identity"] = "wrong-training-view"
        pair["candidate_bundle_identity_inputs"]["training_view_identity"] = "wrong-training-view"

    _rewrite_bundle(state, change_view)
    validation = load_candidate_bundle(state["candidate"])
    assert validation.valid is False
    assert any("training view identity" in reason for reason in validation.reasons)
    assert discover_memorization_review(_context(state))["review_action_available"] is False
    assert (
        discover_memorization_review(_context(state, include_view_identity=False))["review_action_available"] is False
    )

    decision = _decision(state)
    assert decision["classification"] == "not_comparable"
    assert decision["identity_valid"] is False
    assert decision["eligible_for_promotion"] is False
    assert decision["checkpoint_copies"] == 0
    assert decision["promotion_actions"] == 0

    assert any("training_view_identity" in reason for reason in decision["not_comparable_reasons"])


@pytest.mark.parametrize("mode", ("explicit", "nested", "missing", "mixed", "alias"))
def test_training_identity_authoring_and_validation_share_one_contract(tmp_path: Path, mode: str) -> None:
    state = _production_run(tmp_path / mode)
    manifest = state["training_manifest"]
    row = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [row]
    declared_dataset: str | None = None
    declared_view: str | None = None

    if mode == "explicit":
        declared_dataset = "explicit-dataset-identity"
        declared_view = "explicit-view-identity"
    elif mode == "nested":
        row["source"]["dataset_identity"] = row.pop("dataset_identity")
        row["source"]["view_identity"] = row.pop("view_identity")
    elif mode == "missing":
        row.pop("dataset_identity")
        row.pop("view_identity")
    elif mode == "mixed":
        second_archive = manifest.parent / "train-other.npz"
        second_archive.write_bytes((manifest.parent / "train.npz").read_bytes())
        rows.append(
            {
                **row,
                "sprite_id": "training-sprite-2",
                "npz_file": second_archive.name,
                "dataset_identity": "other-training-dataset",
                "view_identity": "other-training-view",
            }
        )
    else:
        row["training_dataset_identity"] = row.pop("dataset_identity")
        row["training_view_identity"] = row.pop("view_identity")

    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8", newline="\n")
    manifest_hash = file_sha256(manifest)
    if mode == "explicit":
        expected_dataset = declared_dataset
        expected_view = declared_view
    elif mode == "nested":
        expected_dataset = "synthetic-training-v1"
        expected_view = "synthetic-training-view-v1"
    else:
        expected_dataset = f"training-manifest-sha256:{manifest_hash}"
        expected_view = f"training-view-sha256:{manifest_hash}"

    output = state["root"] / "identity-rescore"
    score_suite(
        state["generated"],
        output,
        training_manifests=[manifest],
        checkpoint=state["checkpoint"],
        benchmark_manifest=state["benchmark"],
        training_dataset_identity=declared_dataset,
        training_view_identity=declared_view,
    )
    candidate = output / "candidate_evidence.json"
    expected_context = (
        {
            "training_dataset_identity": expected_dataset,
            "training_view_identity": expected_view,
        }
        if mode == "explicit"
        else None
    )
    validation = load_candidate_bundle(candidate, expected_context=expected_context)
    assert validation.valid is True, validation.reasons
    if mode == "explicit":
        assert load_candidate_bundle(candidate).valid is False
    bundle = validation.bundle
    assert bundle["training_dataset_identity"] == expected_dataset
    assert bundle["training_view_identity"] == expected_view
    assert bundle["pairs"][0]["training_dataset_identity"] == expected_dataset
    assert bundle["pairs"][0]["training_view_identity"] == expected_view
    assert bundle["pairs"][0]["candidate_bundle_identity_inputs"]["training_dataset_identity"] == expected_dataset
    assert bundle["pairs"][0]["candidate_bundle_identity_inputs"]["training_view_identity"] == expected_view

    decision = decide_promotion(
        checkpoint=state["checkpoint"],
        benchmark_manifest=state["benchmark"],
        machine_report=output / "summary.json",
        generated_report=output / "per_image_metrics.jsonl",
        generated_manifest=state["generated_manifest"],
        generated_images=Path(bundle["generated_images_root"]),
        training_dataset_identity=str(expected_dataset),
        training_view_identity=str(expected_view),
        training_manifest=manifest,
        training_images=Path(bundle["training_images_root"]),
        candidate_evidence=candidate,
        review_event_log=output / "review_events.jsonl",
        detector_policy=output / "detector_policy.json",
    )
    assert decision["identity_valid"] is True, decision["not_comparable_reasons"]
    assert decision["classification"] == "promotion_hold"
    assert decision["eligible_for_promotion"] is False
    assert decision["checkpoint_copies"] == 0
    assert decision["promotion_actions"] == 0

    if mode == "explicit":
        with pytest.raises(ValueError, match="invalid or outdated"):
            append_bound_review_event(
                candidate,
                output / "review_events.jsonl",
                pair_id=bundle["pairs"][0]["pair_id"],
                review_outcome="different_sprite",
                reviewer_id="reviewer-explicit",
            )
    event = append_bound_review_event(
        candidate,
        output / "review_events.jsonl",
        pair_id=bundle["pairs"][0]["pair_id"],
        review_outcome="different_sprite",
        reviewer_id=f"reviewer-{mode}",
        expected_context=expected_context,
    )
    assert event["training_dataset_identity"] == expected_dataset
    assert event["training_view_identity"] == expected_view


@pytest.mark.parametrize("field", ("dataset_identity", "view_identity"))
def test_training_manifest_top_and_source_identity_conflict_fails_closed(tmp_path: Path, field: str) -> None:
    state = _production_run(tmp_path)
    manifest = state["training_manifest"]
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["source"][field] = f"conflicting-{field}"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = state["root"] / "conflict-rescore"
    report = score_suite(
        state["generated"],
        output,
        training_manifests=[manifest],
        checkpoint=state["checkpoint"],
        benchmark_manifest=state["benchmark"],
    )

    assert not (output / "candidate_evidence.json").exists()
    memo = report["summary"]["memorization"]
    assert memo["evidence_contract_state"] == "incomplete"
    assert any("aliases disagree" in reason for reason in memo["evidence_contract_reasons"])

    bundle = json.loads(state["candidate"].read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest)
    bundle["training_manifest_sha256"] = manifest_hash
    for pair in bundle["pairs"]:
        pair["training_manifest_sha256"] = manifest_hash
        pair["candidate_bundle_identity_inputs"]["training_manifest_sha256"] = manifest_hash
    bundle["candidate_evidence_sha256"] = candidate_bundle_sha256(bundle)
    state["candidate"].write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
    validation = load_candidate_bundle(
        state["candidate"],
        expected_context={
            "training_dataset_identity": "synthetic-training-v1",
            "training_view_identity": "synthetic-training-view-v1",
        },
    )
    assert validation.valid is False
    assert any("aliases disagree" in reason for reason in validation.reasons)


def test_machine_report_uses_exact_duplicate_free_candidate_set(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    memo = state["report"]["summary"]["memorization"]
    assert memo["candidate_pair_ids"] == state["bundle"]["candidate_order"]
    assert len(memo["candidate_pair_ids"]) == len(set(memo["candidate_pair_ids"]))
    assert memo["candidate_count"] == state["bundle"]["candidate_count"]


def test_multiple_evidence_reasons_are_preserved(tmp_path: Path) -> None:
    state = _production_run(tmp_path, "exact_alpha")
    pair = state["bundle"]["pairs"][0]
    assert pair["evidence_class"] == "exact_alpha_review_required"
    assert "near_pixel_review_required" in pair["evidence_reasons"]


@pytest.mark.parametrize(
    ("kind", "decision"),
    (("hard", "blocked"), ("warning", "eligible"), ("none", "eligible")),
)
def test_hard_warning_and_no_candidate_outcomes(tmp_path: Path, kind: str, decision: str) -> None:
    state = _production_run(tmp_path / kind, kind)
    result = _decision(state)
    assert result["decision"] == decision, json.dumps(result, indent=2)
    assert result["checkpoint_copies"] == 0
    assert result["promotion_actions"] == 0


def test_strict_product_discovery_includes_all_review_classes(tmp_path: Path) -> None:
    for kind in ("exact_alpha", "translation_alpha", "near_pixel"):
        state = _production_run(tmp_path / kind, kind)
        discovered = discover_memorization_review(_context(state))
        assert discovered["items"][0]["evidence_class"].startswith(kind.replace("_alpha", "_alpha"))
        assert discovered["items"][0]["review_action_available"] is True


@pytest.mark.parametrize("missing", ("view", "dataset_and_view"))
def test_product_review_requires_active_dataset_and_view_identity(tmp_path: Path, missing: str) -> None:
    state = _production_run(tmp_path)
    context = _context(state, include_view_identity=False)
    if missing == "dataset_and_view":
        context.config["evaluation"].pop("dataset_identity")

    discovered = discover_memorization_review(context)

    assert discovered["items"] == []
    assert discovered["review_action_available"] is False
    assert "dataset and view identities" in discovered["review_message"]

    app = FastAPI()
    app.include_router(build_review_router(context))
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    response = TestClient(app).post(
        f"/review/memorization/{pair_id}/decision",
        json={"review_outcome": "different_sprite", "reviewer_id": "product-reviewer"},
    )
    assert response.status_code == 409
    assert not state["reviews"].exists()


def test_product_review_rejects_bad_bundle_self_hash(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    _rewrite_bundle(state, lambda bundle: bundle.update({"candidate_count": 99}), rehash=False)
    discovered = discover_memorization_review(_context(state))
    assert discovered["items"] == []
    assert discovered["review_message"] == INVALID_EVIDENCE_MESSAGE


def test_product_review_rejects_unknown_class(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    _rewrite_bundle(state, lambda bundle: bundle["pairs"][0].update({"evidence_class": "invented"}))
    assert discover_memorization_review(_context(state))["review_action_available"] is False


def test_product_review_rejects_inconsistent_candidate_counts(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    _rewrite_bundle(state, lambda bundle: bundle.update({"candidate_count": 2}))
    assert discover_memorization_review(_context(state))["items"] == []


def test_product_review_rejects_stale_checkpoint_binding(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    stale = tmp_path / "stale.pt"
    stale.write_bytes(b"stale")
    display = memorization_display(
        state["candidate"],
        expected_context={"checkpoint_path": stale},
    )
    assert display["items"] == []
    assert any("expected checkpoint_path" in reason for reason in display["validation_reasons"])


@pytest.mark.parametrize(
    "identity_field",
    (
        "checkpoint_sha256",
        "candidate_evidence_sha256",
        "benchmark_manifest_sha256",
        "detector_policy_sha256",
        "comparison_parameters_sha256",
        "generated_png_sha256",
        "training_decoded_rgba_sha256",
        "generated_manifest_sha256",
        "training_manifest_sha256",
    ),
)
def test_recomputed_self_hash_never_authorizes_wrong_binding(tmp_path: Path, identity_field: str) -> None:
    state = _production_run(tmp_path)
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    event = append_bound_review_event(
        state["candidate"],
        state["reviews"],
        pair_id=pair_id,
        review_outcome="different_sprite",
        reviewer_id="reviewer-1",
    )
    event[identity_field] = "f" * 64 if event[identity_field] != "f" * 64 else "e" * 64
    event["event_sha256"] = review_event_sha256(event)
    state["reviews"].write_text(json.dumps(event) + "\n", encoding="utf-8")
    display = memorization_display(state["candidate"], review_log=state["reviews"])
    item = display["items"][0]
    assert item["current_review_state"] == "not_comparable"
    assert item["review_authoritative"] is False
    assert item["event_chain_status"] == "identity_mismatch"


@pytest.mark.parametrize(
    ("outcome", "state_name", "authoritative", "decision"),
    (
        ("different_sprite", "cleared_by_valid_bound_review", True, "eligible"),
        ("same_sprite_or_memorized", "blocked", True, "blocked"),
    ),
)
def test_valid_signed_review_controls_display_and_promotion(
    tmp_path: Path,
    outcome: str,
    state_name: str,
    authoritative: bool,
    decision: str,
) -> None:
    state = _production_run(tmp_path)
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    event = append_bound_review_event(
        state["candidate"],
        state["reviews"],
        pair_id=pair_id,
        review_outcome=outcome,
        reviewer_id="reviewer-1",
    )
    assert event["schema_version"] == BOUND_REVIEW_SCHEMA_VERSION
    item = memorization_display(state["candidate"], review_log=state["reviews"])["items"][0]
    assert item["current_review_state"] == state_name
    assert item["review_authoritative"] is authoritative
    result = _decision(state)
    assert result["decision"] == decision, json.dumps(result, indent=2)


def test_wrongly_bound_self_hashed_review_is_not_comparable_for_promotion(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    event = append_bound_review_event(
        state["candidate"],
        state["reviews"],
        pair_id=pair_id,
        review_outcome="different_sprite",
        reviewer_id="reviewer-1",
    )
    event["checkpoint_sha256"] = "f" * 64
    event["event_sha256"] = review_event_sha256(event)
    state["reviews"].write_text(json.dumps(event) + "\n", encoding="utf-8")
    result = _decision(state)
    assert result["classification"] == "not_comparable"
    assert result["eligible_for_promotion"] is False


def test_legacy_v1_is_display_only_and_never_actionable(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    legacy = {"schema_version": SCHEMA_VERSION, "pair_id": "legacy-pair", "classification": "uncertain"}
    state["reviews"].write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    display = memorization_display(state["candidate"], review_log=state["reviews"])
    assert display["legacy_reviews"][0]["promotion_authority"] is False
    assert display["review_action_available"] is False


def test_legacy_v1_remains_readable_when_current_bundle_is_missing(tmp_path: Path) -> None:
    legacy_log = tmp_path / "legacy.jsonl"
    legacy_log.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "pair_id": "legacy-pair"}) + "\n",
        encoding="utf-8",
    )
    display = memorization_display(tmp_path / "missing.json", review_log=legacy_log)
    assert display["items"] == []
    assert display["legacy_reviews"][0]["promotion_authority"] is False
    assert display["review_action_available"] is False


def test_legacy_cli_command_cannot_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "legacy"
    evaluation_cli(["review-memorization", "--out", str(output)])
    assert "read-only" in capsys.readouterr().out
    assert not output.exists()


def test_product_review_endpoint_writes_only_bound_v2(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    context = _context(state)
    app = FastAPI()
    app.include_router(build_review_router(context))
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    response = TestClient(app).post(
        f"/review/memorization/{pair_id}/decision",
        json={"review_outcome": "different_sprite", "reviewer_id": "product-reviewer"},
    )
    assert response.status_code == 200
    assert response.json()["review_event_schema"] == BOUND_REVIEW_SCHEMA_VERSION
    row = json.loads(state["reviews"].read_text(encoding="utf-8"))
    assert row["schema_version"] == BOUND_REVIEW_SCHEMA_VERSION


def test_rendered_product_review_uses_strict_cards_and_signed_v2_controls(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    client = TestClient(create_app(_context(state), plugins=[create_dataset_plugin()]))
    response = client.get("/review?queue=memorization")
    assert response.status_code == 200
    assert "Generated image for" in response.text
    assert "Training comparison image for" in response.text
    assert "exact_alpha_review_required" in response.text
    assert 'data-review-outcome="different_sprite"' in response.text
    assert "Event chain: missing (valid=true)" in response.text


def test_rendered_hard_or_invalid_evidence_has_no_review_controls(tmp_path: Path) -> None:
    hard = _production_run(tmp_path / "hard", "hard")
    hard_client = TestClient(create_app(_context(hard), plugins=[create_dataset_plugin()]))
    hard_response = hard_client.get("/review?queue=memorization")
    assert hard_response.status_code == 200
    assert "Hard memorization evidence cannot be cleared by review." in hard_response.text
    assert "data-review-outcome=" not in hard_response.text

    invalid = _production_run(tmp_path / "invalid")
    _rewrite_bundle(invalid, lambda bundle: bundle.update(checkpoint_sha256="0" * 64), rehash=False)
    invalid_client = TestClient(create_app(_context(invalid), plugins=[create_dataset_plugin()]))
    invalid_response = invalid_client.get("/review?queue=memorization")
    assert invalid_response.status_code == 200
    assert INVALID_EVIDENCE_MESSAGE in invalid_response.text
    assert "data-review-outcome=" not in invalid_response.text


def test_hard_evidence_has_no_clear_action(tmp_path: Path) -> None:
    state = _production_run(tmp_path, "hard")
    item = memorization_display(state["candidate"], review_log=state["reviews"])["items"][0]
    assert item["current_review_state"] == "blocked"
    assert item["clear_action_available"] is False
    assert item["review_action_available"] is False
    assert item["controlled_review_outcomes"] == []


@pytest.mark.parametrize("change", ("missing", "extra", "omitted", "duplicate"))
def test_candidate_pair_set_mismatch_is_not_comparable(tmp_path: Path, change: str) -> None:
    state = _production_run(tmp_path)
    machine = json.loads(state["machine_report"].read_text(encoding="utf-8"))
    ids = list(machine["summary"]["memorization"]["candidate_pair_ids"])
    if change == "missing":
        machine["summary"]["memorization"].pop("candidate_pair_ids")
    elif change == "extra":
        machine["summary"]["memorization"]["candidate_pair_ids"] = [*ids, "extra-pair"]
    elif change == "omitted":
        machine["summary"]["memorization"]["candidate_pair_ids"] = []
    else:
        machine["summary"]["memorization"]["candidate_pair_ids"] = [*ids, *ids]
    state["machine_report"].write_text(json.dumps(machine), encoding="utf-8")
    result = _decision(state)
    assert result["classification"] == "not_comparable"
    assert result["eligible_for_promotion"] is False


def test_missing_candidate_bundle_is_controlled_incomplete(tmp_path: Path) -> None:
    display = memorization_display(tmp_path / "missing.json")
    assert display["evidence_state"] == "incomplete"
    assert display["review_message"] == INCOMPLETE_EVIDENCE_MESSAGE
    assert display["items"] == []


def test_malformed_bundle_has_no_actionable_product_review(tmp_path: Path) -> None:
    path = tmp_path / "candidate_evidence.json"
    path.write_text("{bad-json", encoding="utf-8")
    display = memorization_display(path)
    assert display["items"] == []
    assert display["review_action_available"] is False


def test_code_identity_binds_every_semantic_product_adapter(tmp_path: Path) -> None:
    for relative in MEMORIZATION_AUDIT_BOUND_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"semantic:{relative}\n", encoding="utf-8")
    report = {
        "commit": "synthetic",
        "subsystem": "memorization",
        "overall_verdict": "PASS",
        "code_identity": memorization_audit_code_identity(tmp_path),
    }
    config = ProjectConfig(tmp_path, None, {})
    assert _memorization_audit_status(config, report) is AuditStatus.PASS
    required = {
        "src/spritelab/evaluation/candidate_bundle.py",
        "src/spritelab/product_features/dataset/web.py",
        "src/spritelab/product_features/dataset/static/review.js",
        "src/spritelab/product_features/dataset/templates/review_entry.html",
        "src/spritelab/product_features/evaluation/memorization_display.py",
        "src/spritelab/product_features/evaluation/service.py",
    }
    assert required <= set(MEMORIZATION_AUDIT_BOUND_FILES)


def test_pre_product_audit_without_versioned_code_identity_is_stale(tmp_path: Path) -> None:
    report = {"commit": "synthetic", "overall_verdict": "PASS"}
    assert _memorization_audit_status(ProjectConfig(tmp_path, None, {}), report) is AuditStatus.STALE


@pytest.mark.parametrize(
    "relative",
    (
        "src/spritelab/evaluation/suite.py",
        "src/spritelab/evaluation/candidate_bundle.py",
        "src/spritelab/product_features/dataset/web.py",
        "src/spritelab/product_features/dataset/static/review.js",
        "src/spritelab/product_features/dataset/templates/review_entry.html",
        "src/spritelab/product_features/evaluation/memorization_display.py",
        "src/spritelab/product_features/evaluation/service.py",
    ),
)
def test_semantic_adapter_change_makes_audit_stale(tmp_path: Path, relative: str) -> None:
    for bound in MEMORIZATION_AUDIT_BOUND_FILES:
        path = tmp_path / bound
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before\n", encoding="utf-8")
    report = {
        "commit": "synthetic",
        "subsystem": "memorization",
        "overall_verdict": "PASS",
        "code_identity": memorization_audit_code_identity(tmp_path),
    }
    (tmp_path / relative).write_text("after\n", encoding="utf-8")
    assert _memorization_audit_status(ProjectConfig(tmp_path, None, {}), report) is AuditStatus.STALE


@pytest.mark.parametrize("relative", ("docs/guide.md", "src/spritelab/product_web/static/cosmetic.css"))
def test_unrelated_documentation_and_css_do_not_stale_audit(tmp_path: Path, relative: str) -> None:
    for bound in MEMORIZATION_AUDIT_BOUND_FILES:
        path = tmp_path / bound
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before\n", encoding="utf-8")
    report = {
        "commit": "synthetic",
        "subsystem": "memorization",
        "overall_verdict": "PASS",
        "code_identity": memorization_audit_code_identity(tmp_path),
    }
    cosmetic = tmp_path / relative
    cosmetic.parent.mkdir(parents=True, exist_ok=True)
    cosmetic.write_text("cosmetic\n", encoding="utf-8")
    assert _memorization_audit_status(ProjectConfig(tmp_path, None, {}), report) is AuditStatus.PASS


def test_synthetic_eligible_decision_performs_no_mutation(tmp_path: Path) -> None:
    state = _production_run(tmp_path)
    pair_id = state["bundle"]["pairs"][0]["pair_id"]
    append_bound_review_event(
        state["candidate"],
        state["reviews"],
        pair_id=pair_id,
        review_outcome="different_sprite",
        reviewer_id="reviewer-1",
    )
    before = state["checkpoint"].read_bytes()
    result = _decision(state)
    assert result["eligible_for_promotion"] is True, result
    assert result["checkpoint_copies"] == 0
    assert result["promotion_actions"] == 0
    assert state["checkpoint"].read_bytes() == before

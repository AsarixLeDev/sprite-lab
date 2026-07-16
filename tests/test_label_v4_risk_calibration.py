from __future__ import annotations

from pathlib import Path

from PIL import Image

from spritelab.harvest.label_v4.calibration import (
    allocate_sparse_audit,
    build_calibration_bundle,
    calibration_metrics,
    fit_isotonic,
    wilson_upper_bound,
)
from spritelab.harvest.label_v4.pixel_evidence import analyze_pixels, exact_rgba_content_hash
from spritelab.harvest.label_v4.providers import MockJSONProvider
from spritelab.harvest.label_v4.risk import (
    SEMANTIC_FIELDS,
    ensure_all_field_risks,
    estimate_field_risk,
    risk_score_from_upper,
    summarize_record_risk,
)


def _png(path: Path, color: tuple[int, int, int, int] = (120, 80, 200, 255)) -> None:
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    for x in range(1, 3):
        for y in range(1, 3):
            image.putpixel((x, y), color)
    image.save(path)


def test_uncertainty_contract_uses_upper_bound_and_clamps() -> None:
    assert risk_score_from_upper(0.0) == 1
    assert risk_score_from_upper(0.08) == 2
    assert risk_score_from_upper(0.14) == 3
    assert risk_score_from_upper(1.0) == 20


def test_uncalibrated_critical_field_is_conservative() -> None:
    risk = estimate_field_risk(
        "canonical_object",
        value_present=True,
        risk_features={
            "deterministic_evidence_strong": True,
            "filename_vlm_agreement": True,
            "independent_dependency_groups": 2,
        },
    )
    assert risk.calibration_state == "uncalibrated"
    assert risk.uncertainty_1_20 is not None and risk.uncertainty_1_20 >= 13
    assert risk.risk_upper_95 is not None and risk.risk_upper_95 >= 0.65


def test_supported_calibration_and_conflict_floor() -> None:
    calibrated = estimate_field_risk(
        "category",
        value_present=True,
        calibration={
            "calibration_state": "calibrated",
            "calibration_support_n": 84,
            "calibration_stratum": "category/gem/unseen_pack",
            "p_error_estimate": 0.08,
            "risk_upper_95": 0.14,
        },
    )
    assert calibrated.calibration_state == "calibrated"
    assert calibrated.uncertainty_1_20 == 3

    conflict = estimate_field_risk(
        "category",
        value_present=True,
        risk_features={"unresolved_conflict": True, "contradiction_count": 1},
        calibration={
            "calibration_state": "calibrated",
            "calibration_support_n": 84,
            "p_error_estimate": 0.01,
            "risk_upper_95": 0.03,
        },
    )
    assert conflict.uncertainty_1_20 is not None and conflict.uncertainty_1_20 >= 9
    assert conflict.uncertainty_band != "strong"


def test_every_field_has_numeric_risk_or_not_scorable() -> None:
    risks = ensure_all_field_risks({"canonical_object": "buckler"}, {})
    assert set(risks) == set(SEMANTIC_FIELDS)
    assert risks["canonical_object"].uncertainty_1_20 is not None
    assert risks["description"].calibration_state == "not_scorable"


def test_record_risk_preserves_critical_max() -> None:
    fields = {
        "canonical_object": estimate_field_risk(
            "canonical_object",
            value_present=True,
            calibration={
                "calibration_state": "calibrated",
                "calibration_support_n": 80,
                "p_error_estimate": 0.7,
                "risk_upper_95": 0.82,
            },
        ),
        "category": estimate_field_risk(
            "category",
            value_present=True,
            calibration={
                "calibration_state": "calibrated",
                "calibration_support_n": 80,
                "p_error_estimate": 0.02,
                "risk_upper_95": 0.04,
            },
        ),
        "domain": estimate_field_risk(
            "domain",
            value_present=True,
            calibration={
                "calibration_state": "calibrated",
                "calibration_support_n": 80,
                "p_error_estimate": 0.02,
                "risk_upper_95": 0.04,
            },
        ),
        "role": estimate_field_risk(
            "role",
            value_present=True,
            calibration={
                "calibration_state": "calibrated",
                "calibration_support_n": 80,
                "p_error_estimate": 0.02,
                "risk_upper_95": 0.04,
            },
        ),
    }
    summary = summarize_record_risk(fields)
    assert summary.critical_field_max_uncertainty == 17
    assert summary.record_uncertainty_1_20 == 17
    assert summary.mean_field_uncertainty is not None and summary.mean_field_uncertainty < 10


def test_isotonic_calibration_and_sparse_audit() -> None:
    bins = fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert [row.p_error for row in bins] == sorted(row.p_error for row in bins)
    assert wilson_upper_bound(0, 4) > 0.0

    reviewed = [
        {
            "field": "category",
            "stratum": "category/gem/seen_pack",
            "raw_risk": index / 40,
            "is_error": index >= 32,
            "review_id": str(index),
        }
        for index in range(40)
    ]
    bundle = build_calibration_bundle(reviewed, min_support=30)
    model = bundle.strata[0]
    assert model.calibration_state == "calibrated"
    assert model.predict(0.9)["risk_upper_95"] >= model.predict(0.1)["p_error_estimate"]

    candidates = [
        {
            "sprite_id": f"s{index}",
            "field": "category" if index % 2 else "canonical_object",
            "uncertainty_1_20": 20 - index % 20,
            "pack": f"pack{index % 7}",
            "calibration_state": "uncalibrated" if index % 3 else "calibrated",
        }
        for index in range(50)
    ]
    first = allocate_sparse_audit(candidates, target_size=20, seed=3)
    second = allocate_sparse_audit(candidates, target_size=20, seed=3)
    assert first["selected_ids"] == second["selected_ids"]
    assert first["audit_set_hash"] == second["audit_set_hash"]


def test_calibration_metrics_include_required_curves() -> None:
    metrics = calibration_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert metrics["brier_score"] < 0.1
    assert metrics["expected_calibration_error"] >= 0.0
    assert metrics["reliability_diagram"]
    assert metrics["selective_risk_curve"][-1]["coverage"] == 1.0


def test_exact_image_identity_pixel_palette_and_mock_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sprite.png"
    _png(path)
    first_hash = exact_rgba_content_hash(path)
    evidence = analyze_pixels(path)
    assert evidence["image_hash"] == first_hash
    assert "purple" in evidence["palette_colors"]
    assert set(evidence["shape"]) == {
        "silhouette",
        "aspect",
        "orientation",
        "structure",
        "edge_profile",
        "parts",
    }

    provider = MockJSONProvider({("blind_vlm", first_hash): {"object_candidates": []}})
    artifact = provider.call_json(
        stage="blind_vlm",
        prompt="visible evidence only",
        prompt_version="p1",
        image_path=path,
    )
    assert artifact.ok
    assert artifact.image_hash == first_hash
    assert artifact.model_identity == "mock-label-v4"
    assert artifact.request_hash and artifact.prompt_hash

    _png(path, (20, 200, 80, 255))
    assert exact_rgba_content_hash(path) != first_hash

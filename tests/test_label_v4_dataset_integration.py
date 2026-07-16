from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.dataset_v5.builder import _strict_quality, _verify_loader_contract, _write_loader_adapter
from spritelab.dataset_v5.policy_v2 import WeightingPolicy, compute_sampling_weights
from spritelab.evaluation.suite import _label_quality_metric_strata
from spritelab.harvest.label_v4.risk import CRITICAL_FIELDS, SEMANTIC_FIELDS
from spritelab.harvest.label_v4.training_quality import uncertainty_correlation_report


def _quality(score: int, state: str = "calibrated") -> dict:
    return {
        "schema_version": "label_training_quality_v1",
        "record_uncertainty_1_20": score,
        "record_uncertainty_band": "strong" if score <= 4 else "auxiliary_only",
        "record_loss_weight": 1.0 - score / 20.0,
        "unresolved_conflict_count": 0,
        "fields": {
            name: {
                "uncertainty_1_20": score,
                "risk_upper_95": score / 20.0,
                "calibration_state": state,
                "supervision_mask": int(score <= 8),
                "auxiliary_mask": int(score <= 12),
                "conditioning_mask": int(score <= 12),
                "loss_weight": max(0.05, 1.0 - score / 20.0),
                "value_state": "known",
            }
            for name in SEMANTIC_FIELDS
        },
    }


def _arrays(sprite_id: str) -> dict[str, np.ndarray]:
    return {
        "alpha": np.zeros((32, 32), dtype=np.uint8),
        "index_map": np.zeros((32, 32), dtype=np.uint8),
        "role_map": np.zeros((32, 32), dtype=np.uint8),
        "palette": np.zeros((32, 3), dtype=np.uint8),
        "palette_mask": np.zeros((32,), dtype=bool),
        "category_id": np.asarray(0, dtype=np.int64),
        "sprite_id": np.asarray(sprite_id),
    }


def test_dataset_v5_loader_adapter_carries_quality_and_uses_val_split(tmp_path: Path) -> None:
    quality = _quality(3)
    record = {
        "sprite_id": "sprite",
        "category": "weapon",
        "object_name": "sword",
        "source_id": "source",
        "source_pack": "pack",
        "author": "artist",
        "suitability_status": "accept",
        "label_provenance": {"adapter": "label_v4"},
        "label_quality": quality,
        "training_record": {},
        "_arrays": _arrays("sprite"),
        "_rgba": np.zeros((32, 32, 4), dtype=np.uint8),
    }
    _write_loader_adapter(tmp_path, [record], {"sprite": "validation"})
    row = json.loads((tmp_path / "training_manifest.jsonl").read_text().strip())
    assert row["split"] == "val"
    assert row["npz_file"] == "val.npz"
    assert row["label_quality"]["record_uncertainty_1_20"] == 3
    assert row["source_pack"] == "pack"
    assert _verify_loader_contract(tmp_path)["ok"] is True


def test_dataset_v5_strict_quality_requires_calibrated_strong_critical_fields() -> None:
    assert _strict_quality({"label_quality": _quality(3)}) is True
    uncalibrated = _quality(3, "uncalibrated")
    assert _strict_quality({"label_quality": uncalibrated}) is False
    weak = _quality(7)
    assert _strict_quality({"label_quality": weak}) is False
    assert set(CRITICAL_FIELDS) <= set(weak["fields"])


def test_quality_aware_sampling_is_bounded_and_reports_kish_ess() -> None:
    rows = [
        {
            "sprite_id": "strong",
            "source_pack": "pack",
            "sub_artist": "artist",
            "source_family": "family",
            "object_name": "sword",
            "is_supervised": True,
            "strict_quality": False,
            "label_quality": _quality(3),
        },
        {
            "sprite_id": "aux",
            "source_pack": "pack",
            "sub_artist": "artist",
            "source_family": "family",
            "object_name": "sword",
            "is_supervised": True,
            "strict_quality": False,
            "label_quality": _quality(11),
        },
    ]
    policy = WeightingPolicy(
        pack_exponent=0,
        artist_exponent=0,
        source_family_exponent=0,
        canonical_object_exponent=0,
        geometry_family_exponent=0,
        minimum_weight=0.05,
        maximum_weight=2.0,
    )
    weights, report = compute_sampling_weights(rows, {"strong": "a", "aux": "b"}, policy)
    assert set(weights) == {"strong", "aux"}
    assert weights["strong"] > weights["aux"] >= 0.05
    assert 1.0 <= report["effective_sample_size"] <= 2.0


def test_evaluation_report_strata_and_noncausal_correlations() -> None:
    strong = {
        "label_quality": _quality(3),
        "conditional_adherence": 0.9,
        "memorization_indicator": False,
        "generation_failed": False,
        "split": "test",
    }
    weak = {
        "label_quality": _quality(7),
        "conditional_adherence": 0.4,
        "memorization_indicator": True,
        "generation_failed": True,
        "split": "source_ood_test",
        "unseen_pack": True,
        "propagation_relation": "recolor",
    }
    strata = _label_quality_metric_strata([strong, weak])
    assert strata["all_labels"]["sample_count"] == 2
    assert strata["strong_labels"]["sample_count"] == 1
    assert strata["source_ood"]["sample_count"] == 1
    assert strata["propagated_labels"]["generation_failure_rate"] == 1.0
    correlation = uncertainty_correlation_report([strong, weak])
    assert correlation["causal_claim"] is False
    assert correlation["relationships"]["conditional_adherence"]["paired_support_n"] == 2

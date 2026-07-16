from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import spritelab.dataset_v5.named_views as named_views
from spritelab.dataset_v5.builder import alpha_mask_sha256, canonical_rgba_sha256, source_file_sha256
from spritelab.dataset_v5.named_views import (
    DatasetV5ViewError,
    build_view,
    freeze_view,
    validate_contract,
    verify_freeze,
    verify_view,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "experiments" / "v5_view_contract_v1"
SOURCE_SCHEMA = "dataset_v5_source_record_v1.0.0"
POLICY_SCHEMA = "dataset_v5_named_view_policy_v1.0.0"
APPROVAL_SCHEMA = "dataset_v5_approved_decisions_v1.0.0"
EXPECTED_BUILD_ARTIFACTS = {
    "record_manifest.jsonl",
    "excluded_record_manifest.jsonl",
    "split_manifest.json",
    "weight_manifest.jsonl",
    "evaluation_manifest.jsonl",
    "license_provenance.jsonl",
    "validation_report.json",
    "view_manifest.json",
}
SEMANTIC_FIELDS = (
    "category",
    "canonical_object",
    "domain",
    "role",
    "explicit_material",
    "shape",
    "palette_colors",
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "shadow_colors",
    "highlight_colors",
)
ALL_APPROVALS = {
    "representative_selector": True,
    "explicit_recolor_identity": True,
    "size_envelope_exception": True,
    "quarantine_policy": True,
    "weighting_policy": True,
    "raking_policy": True,
    "evaluation_minimum": True,
    "human_truth": True,
    "calibrated_strata": True,
    "source_ood_scope": True,
    "open_set_concept": True,
    "frozen_r2_binding": True,
}


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row.get("record_id") or row.get("sprite_id") or ""))
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
            for row in ordered
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tree_snapshot(path: Path) -> dict[str, bytes]:
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}


def _rgba(seed: int, *, mask_seed: int | None = None, color: tuple[int, int, int] | None = None) -> np.ndarray:
    geometry = seed if mask_seed is None else mask_seed
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    x = 1 + (geometry * 5) % 24
    y = 1 + (geometry * 7) % 24
    width = 2 + geometry % 5
    height = 2 + (geometry // 3) % 5
    rgb = color or ((31 * seed) % 251 + 1, (67 * seed) % 251 + 1, (97 * seed) % 251 + 1)
    rgba[y : y + height, x : x + width, :3] = rgb
    rgba[y : y + height, x : x + width, 3] = 255
    return rgba


def _field_values(object_name: str) -> dict[str, Any]:
    return {
        "category": "weapon",
        "canonical_object": object_name,
        "domain": "item",
        "role": "equipment",
        "explicit_material": "steel",
        "shape": "long",
        "palette_colors": ["gray", "blue"],
        "primary_colors": ["gray"],
        "secondary_colors": ["blue"],
        "outline_colors": ["black"],
        "shadow_colors": ["dark_gray"],
        "highlight_colors": ["white"],
    }


def _supervision(
    supervision_class: str,
    values: dict[str, Any],
    *,
    target_state: str | None = None,
) -> dict[str, Any]:
    if supervision_class == "supervised_strong":
        default_state = "known"
        mask = 1
        weight = 1.0
        uncertainty = {"state": "calibrated", "score_1_20": 2}
        calibration: str | None = "synthetic-human-truth-v1"
    elif supervision_class == "supervised_weak":
        default_state = "known"
        mask = 1
        weight = 0.5
        uncertainty = {"state": "not_scorable", "score_1_20": None}
        calibration = None
    elif supervision_class == "auxiliary_only":
        default_state = "known"
        mask = 0
        weight = 0.0
        uncertainty = {"state": "provisional_uncalibrated", "score_1_20": 12}
        calibration = None
    else:
        default_state = "missing"
        mask = 0
        weight = 0.0
        uncertainty = {"state": "not_scorable", "score_1_20": None}
        calibration = None
    state = target_state or default_state
    contributing = state == "known" and supervision_class in {"supervised_strong", "supervised_weak"}
    return {
        "supervision_class": supervision_class,
        "targets": {
            field: {"state": state, "value": values[field] if state == "known" else None} for field in SEMANTIC_FIELDS
        },
        "field_masks": dict.fromkeys(SEMANTIC_FIELDS, mask if contributing else 0),
        "field_weights": dict.fromkeys(SEMANTIC_FIELDS, weight if contributing else 0.0),
        "field_uncertainty": {field: copy.deepcopy(uncertainty) for field in SEMANTIC_FIELDS},
        "field_calibration_identity": dict.fromkeys(SEMANTIC_FIELDS, calibration if contributing else None),
    }


def _source_record(
    source_root: Path,
    sprite_id: str,
    seed: int,
    *,
    supervision_class: str = "supervised_strong",
    requested_split: str = "train",
    review_quality: str = "strict",
    suitability_status: str = "accept",
    quality_eligibility: str = "eligible",
    source_pack: str | None = None,
    source_family: str | None = None,
    artist: str | None = None,
    creator_lineage: str | None = None,
    geometry_family_id: str | None = None,
    recolor_family_id: str | None = None,
    declared_variant_group_id: str | None = None,
    sheet_group_id: str | None = None,
    hard_relation_group_ids: list[str] | None = None,
    known_translation_group_id: str | None = None,
    known_flip_group_id: str | None = None,
    mask_seed: int | None = None,
    color: tuple[int, int, int] | None = None,
    object_name: str = "sword",
    evaluation_candidate: bool = False,
    evaluation_stratum: str | None = None,
    source_ood: bool = False,
    source_ood_scope: str | None = None,
    source_ood_rationale: str | None = None,
    open_set: bool = False,
    open_set_rationale: str | None = None,
    semantic_origin: str = "human_truth",
) -> dict[str, Any]:
    rgba = _rgba(seed, mask_seed=mask_seed, color=color)
    payload = rgba.tobytes()
    rgba_hash = canonical_rgba_sha256(rgba)
    blob_hash = hashlib.sha256(payload).hexdigest()
    blob_path = source_root / "blobs" / f"{rgba_hash}.rgba"
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    if blob_path.exists():
        assert blob_path.read_bytes() == payload
    else:
        blob_path.write_bytes(payload)
    values = _field_values(object_name)
    supervision = _supervision(supervision_class, values)
    pack = source_pack or f"pack_{seed}"
    family = source_family or f"source_family_{seed}"
    sub_artist = artist or f"artist_{seed}"
    alpha = rgba[..., 3]
    return {
        "schema_version": SOURCE_SCHEMA,
        "record_id": f"record:{sprite_id}",
        "sprite_id": sprite_id,
        "blob_path": blob_path.relative_to(source_root).as_posix(),
        "blob_sha256": blob_hash,
        "exported_rgba_hash": rgba_hash,
        "exported_width": 32,
        "exported_height": 32,
        "alpha_mask_hash": alpha_mask_sha256(alpha),
        "normalized_alpha_hash": alpha_mask_sha256(alpha),
        "source_artifact": "source_manifest.jsonl",
        "source_record_id": f"source:{sprite_id}",
        "source_pack": pack,
        "sub_artist": sub_artist,
        "source_family": family,
        "license": "cc0",
        "provenance_status": "verified",
        "geometry_family_id": geometry_family_id or f"geometry:{sprite_id}",
        "recolor_family_id": recolor_family_id,
        "declared_variant_group_id": declared_variant_group_id,
        "sheet_group_id": sheet_group_id,
        "hard_relation_group_ids": hard_relation_group_ids or [],
        "suitability_status": suitability_status,
        "suitability_reason_codes": ["SYNTHETIC_QUARANTINE"] if suitability_status == "quarantine" else [],
        "quality_eligibility": quality_eligibility,
        "membership": "included",
        "split": requested_split,
        "sampling_weight": 1.0,
        "evaluation_weight": 1.0 if evaluation_candidate else 0.0,
        "view_inclusion_reason": "synthetic_fixture",
        "view_exclusion_reason": None,
        **supervision,
        **values,
        "source_ood": source_ood,
        "source_ood_scope": source_ood_scope,
        "source_ood_rationale": source_ood_rationale,
        "open_set": open_set,
        "open_set_rationale": open_set_rationale,
        "evaluation_stratum": evaluation_stratum,
        "requested_split": requested_split,
        "review_quality": review_quality,
        "base_sampling_weight": 1.0,
        "base_evaluation_weight": 1.0 if evaluation_candidate else 0.0,
        "creator_lineage": creator_lineage or f"creator:{sub_artist}",
        "acquisition_run_identity": f"acquisition-run:{seed}",
        "distribution_platform": "synthetic_fixture",
        "provenance": {
            "semantic_origin": semantic_origin,
            "source_url": f"https://example.invalid/{sprite_id}",
            "license_confirmed": True,
            "attribution": sub_artist,
        },
        "known_translation_group_id": known_translation_group_id,
        "known_flip_group_id": known_flip_group_id,
        "variant_inclusion_reason": "synthetic_variant" if declared_variant_group_id else None,
        "evaluation_candidate": evaluation_candidate,
    }


def _write_source_manifest(source_root: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_jsonl(source_root / "source_manifest.jsonl", rows)


def _policy(
    path: Path,
    view_name: str,
    *,
    target_size: int,
    view_status: str | None = None,
    frozen_r2_binding: dict[str, Any] | None = None,
) -> Path:
    status = view_status or "diagnostic"
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA,
        "view_name": view_name,
        "view_status": status,
        "synthetic_fixture": True,
        "target_size": target_size,
        "quality_multipliers": {"strict": 1.0, "standard": 0.8, "unreviewed": 0.0},
        "approvals": dict(ALL_APPROVALS),
        "code_identity": {"git_commit": "a" * 40, "dirty": False},
    }
    if frozen_r2_binding is not None:
        value["frozen_r2_binding"] = frozen_r2_binding
    return _write_json(path, value)


def _approved_decisions(path: Path) -> Path:
    return _write_json(path, {"schema_version": APPROVAL_SCHEMA, "approvals": dict(ALL_APPROVALS)})


def _assert_error(error: DatasetV5ViewError, exit_code: int, token: str) -> None:
    assert error.exit_code == exit_code
    assert token.lower() in str(error).lower()


def _view_rows(source_root: Path, view_name: str) -> list[dict[str, Any]]:
    if view_name in {"v5_debug", "v5_architecture", "v5_scale_check"}:
        rows = [
            _source_record(source_root, f"{view_name}_strong", 1, supervision_class="supervised_strong"),
            _source_record(source_root, f"{view_name}_weak", 2, supervision_class="supervised_weak"),
            _source_record(
                source_root,
                f"{view_name}_auxiliary",
                3,
                supervision_class="auxiliary_only",
                semantic_origin="model_proposal",
            ),
            _source_record(source_root, f"{view_name}_unlabeled", 4, supervision_class="unlabeled"),
        ]
        if view_name == "v5_debug":
            rows.append(
                _source_record(
                    source_root,
                    "v5_debug_quarantine",
                    5,
                    supervision_class="unlabeled",
                    suitability_status="quarantine",
                    quality_eligibility="diagnostic_only",
                )
            )
        return rows
    if view_name == "v5_eval_balanced":
        return [
            _source_record(
                source_root,
                f"balanced_{index}",
                10 + index,
                requested_split="test",
                evaluation_candidate=True,
                evaluation_stratum=f"stratum_{index}",
            )
            for index in range(2)
        ]
    if view_name == "v5_source_ood":
        return [
            _source_record(
                source_root,
                f"source_ood_{index}",
                20 + index,
                source_pack=f"held_pack_{index}",
                requested_split="source_ood_test",
                evaluation_candidate=True,
                evaluation_stratum=f"ood_{index}",
                source_ood=True,
                source_ood_scope="held_out_pack",
                source_ood_rationale="synthetic held-out pack",
            )
            for index in range(2)
        ]
    if view_name == "v5_open_set":
        return [
            _source_record(
                source_root,
                f"open_set_{index}",
                30 + index,
                object_name="relic",
                requested_split="open_set_test",
                evaluation_candidate=True,
                evaluation_stratum=f"open_{index}",
                open_set=True,
                open_set_rationale="synthetic approved concept",
            )
            for index in range(2)
        ]
    return [
        _source_record(source_root, f"unlabeled_{index}", 40 + index, supervision_class="unlabeled")
        for index in range(2)
    ]


def test_contract_root_is_supported() -> None:
    result = validate_contract(CONTRACT_ROOT)
    assert result["ok"] is True
    assert result["contract_version"] == "dataset_v5_view_contract_v1.0.0"


@pytest.mark.parametrize(
    ("view_name", "expected_status", "eligible_classes"),
    [
        (
            "v5_debug",
            "diagnostic",
            {"supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled"},
        ),
        (
            "v5_architecture",
            "diagnostic",
            {"supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled"},
        ),
        (
            "v5_scale_check",
            "diagnostic",
            {"supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled"},
        ),
        ("v5_eval_balanced", "diagnostic", {"supervised_strong"}),
        ("v5_source_ood", "diagnostic", {"supervised_strong"}),
        ("v5_open_set", "diagnostic", {"supervised_strong"}),
        ("v5_unlabeled", "diagnostic", {"unlabeled", "auxiliary_only"}),
    ],
)
def test_all_seven_views_emit_complete_canonical_artifacts(
    tmp_path: Path,
    view_name: str,
    expected_status: str,
    eligible_classes: set[str],
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, view_name)
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", view_name, target_size=len(rows), view_status=expected_status)
    output = tmp_path / "view"

    result = build_view(CONTRACT_ROOT, view_name, policy, [source_manifest], output)

    assert result["ok"] is True
    assert EXPECTED_BUILD_ARTIFACTS <= {path.name for path in output.iterdir() if path.is_file()}
    assert not (output / "FREEZE.json").exists()
    manifest = _read_json(output / "view_manifest.json")
    assert manifest["view_name"] == view_name
    assert manifest["view_status"] == expected_status
    assert manifest["example_only"] is True
    assert manifest["production_frozen"] is False
    assert manifest["promotion_forbidden"] is True
    assert manifest["hard_relation_validation"]["passed"] is True
    assert manifest["hard_relation_validation"]["crossing_count"] == 0
    output_rows = _read_jsonl(output / "record_manifest.jsonl")
    assert [row["record_id"] for row in output_rows] == sorted(row["record_id"] for row in output_rows)
    assert output_rows
    assert {row["supervision_class"] for row in output_rows} <= eligible_classes
    verification = verify_view(CONTRACT_ROOT, output)
    assert verification["ok"] is True, verification
    if view_name == "v5_debug":
        quarantine = next(row for row in output_rows if row["sprite_id"] == "v5_debug_quarantine")
        assert quarantine["quality_eligibility"] == "diagnostic_only"
    if view_name in {"v5_eval_balanced", "v5_source_ood", "v5_open_set"}:
        assert all(row["sampling_weight"] == 0 for row in output_rows)
        assert all(row["evaluation_weight"] > 0 for row in output_rows)
        assert all(row["evaluation_stratum"] for row in output_rows)
    if view_name == "v5_unlabeled":
        assert all(value == 0 for row in output_rows for value in row["field_masks"].values())
        assert all(value == 0 for row in output_rows for value in row["field_weights"].values())
        assert all(row["evaluation_weight"] == 0 for row in output_rows)


def test_exact_quality_multipliers_do_not_change_membership(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = [
        _source_record(source_root, "strict", 50, review_quality="strict"),
        _source_record(source_root, "standard", 51, review_quality="standard"),
        _source_record(
            source_root,
            "unreviewed",
            52,
            supervision_class="auxiliary_only",
            review_quality="unreviewed",
            semantic_origin="model_proposal",
        ),
    ]
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=3)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    by_id = {row["sprite_id"]: row for row in _read_jsonl(output / "record_manifest.jsonl")}
    assert set(by_id) == {"strict", "standard", "unreviewed"}
    assert by_id["strict"]["sampling_weight"] == pytest.approx(1.0)
    assert by_id["standard"]["sampling_weight"] == pytest.approx(0.8)
    assert by_id["unreviewed"]["sampling_weight"] == 0.0
    assert all(row["membership"] == "included" for row in by_id.values())


def test_architecture_priority_prefers_core_then_material_variant(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    shared_geometry = "geometry:architecture-priority"
    rows = [
        _source_record(
            source_root,
            "a_core",
            53,
            geometry_family_id=shared_geometry,
        ),
        _source_record(
            source_root,
            "b_material",
            54,
            geometry_family_id=shared_geometry,
            declared_variant_group_id="material:steel",
        ),
        _source_record(
            source_root,
            "c_recolor",
            55,
            geometry_family_id=shared_geometry,
            recolor_family_id="recolor:blue",
        ),
        _source_record(
            source_root,
            "d_plain_variant",
            56,
            geometry_family_id=shared_geometry,
        ),
    ]
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=2)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    assert {row["sprite_id"] for row in _read_jsonl(output / "record_manifest.jsonl")} == {
        "a_core",
        "b_material",
    }


def test_exact_duplicate_is_excluded_with_zero_weight(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = [
        _source_record(source_root, "duplicate_a", 57),
        _source_record(source_root, "duplicate_b", 57),
    ]
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=2)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    included = _read_jsonl(output / "record_manifest.jsonl")
    excluded = _read_jsonl(output / "excluded_record_manifest.jsonl")
    assert [row["sprite_id"] for row in included] == ["duplicate_a"]
    assert [row["sprite_id"] for row in excluded] == ["duplicate_b"]
    assert excluded[0]["membership"] == "excluded"
    assert excluded[0]["sampling_weight"] == 0
    assert excluded[0]["evaluation_weight"] == 0
    assert "exact rgba duplicate" in excluded[0]["view_exclusion_reason"].lower()


def test_build_and_freeze_are_byte_identical(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, list(reversed(rows)))
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], first)
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], second)
    assert _tree_snapshot(first) == _tree_snapshot(second)

    approvals = _approved_decisions(tmp_path / "approved.json")
    first_freeze = freeze_view(CONTRACT_ROOT, first, approvals, "dataset-v5 freeze-view --view-root $VIEW_ROOT")
    second_freeze = freeze_view(CONTRACT_ROOT, second, approvals, "dataset-v5 freeze-view --view-root $VIEW_ROOT")
    assert first_freeze["ok"] is True
    assert second_freeze["ok"] is True
    assert _tree_snapshot(first) == _tree_snapshot(second)
    assert verify_freeze(first)["ok"] is True
    assert verify_freeze(second)["ok"] is True


@pytest.mark.parametrize("view_name", list(named_views.VIEW_NAMES))
def test_all_seven_view_rebuilds_and_freezes_are_byte_identical(tmp_path: Path, view_name: str) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, view_name)
    source_manifest = _write_source_manifest(source_root, list(reversed(rows)))
    policy = _policy(tmp_path / "policy.json", view_name, target_size=len(rows))
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_view(CONTRACT_ROOT, view_name, policy, [source_manifest], first)
    build_view(CONTRACT_ROOT, view_name, policy, [source_manifest], second)
    assert _tree_snapshot(first) == _tree_snapshot(second)

    approvals = _approved_decisions(tmp_path / "approved.json")
    freeze_view(CONTRACT_ROOT, first, approvals, "dataset-v5 freeze-view --view-root $VIEW_ROOT")
    freeze_view(CONTRACT_ROOT, second, approvals, "dataset-v5 freeze-view --view-root $VIEW_ROOT")
    assert _tree_snapshot(first) == _tree_snapshot(second)
    assert verify_freeze(first)["ok"] is True
    assert verify_freeze(second)["ok"] is True


def test_existing_output_root_fails_without_mutation(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_debug")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=len(rows))
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8", newline="\n")

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_debug", policy, [source_manifest], output)

    _assert_error(captured.value, 20, "exist")
    assert _tree_snapshot(output) == {"sentinel.txt": b"unchanged\n"}


def test_missing_source_hash_binding_fails_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_debug")
    source_manifest = _write_source_manifest(source_root, rows)
    policy_path = _policy(tmp_path / "policy.json", "v5_debug", target_size=len(rows))
    policy = _read_json(policy_path)
    policy["source_manifest_sha256"] = {}
    _write_json(policy_path, policy)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_debug", policy_path, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 21, "source hash binding")
    assert not (tmp_path / "view" / "view_manifest.json").exists()


def test_unsupported_contract_fails_closed(tmp_path: Path) -> None:
    contract = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, contract)
    contracts_path = contract / "view_contracts.json"
    contracts = _read_json(contracts_path)
    contracts["contract_version"] = "dataset_v5_view_contract_v999.0.0"
    _write_json(contracts_path, contracts)
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_debug")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=len(rows))

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(contract, "v5_debug", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 23, "contract")
    assert not (tmp_path / "view" / "view_manifest.json").exists()


def test_same_version_contract_clause_tampering_is_rejected(tmp_path: Path) -> None:
    contract = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, contract)
    contracts_path = contract / "view_contracts.json"
    contracts = _read_json(contracts_path)
    assert contracts["contract_version"] == "dataset_v5_view_contract_v1.0.0"
    contracts["views"]["v5_debug"]["promotion_restrictions"][0] = "production promotion permitted"
    _write_json(contracts_path, contracts)

    with pytest.raises(DatasetV5ViewError) as captured:
        validate_contract(contract)

    _assert_error(captured.value, 23, "contract artifacts")


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        (lambda row: row.pop("field_masks"), "field_masks"),
        (
            lambda row: (
                row.update({"supervision_class": "supervised_strong"}),
                row.update({"provenance": {**row["provenance"], "semantic_origin": "model_proposal"}}),
                row.update(
                    {
                        "field_uncertainty": {
                            field: {"state": "provisional_uncalibrated", "score_1_20": 12} for field in SEMANTIC_FIELDS
                        },
                        "field_calibration_identity": dict.fromkeys(SEMANTIC_FIELDS),
                    }
                ),
            ),
            "calibrat",
        ),
        (
            lambda row: (
                row["field_masks"].update({"category": 1}),
                row["field_weights"].update({"category": 1.0}),
            ),
            "unlabeled",
        ),
        (
            lambda row: (
                row["targets"].update({"category": {"state": "missing", "value": None}}),
                row["field_masks"].update({"category": 1}),
                row["field_weights"].update({"category": 1.0}),
            ),
            "known",
        ),
    ],
)
def test_supervision_attacks_fail_closed(
    tmp_path: Path,
    attack: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    source_root = tmp_path / "source"
    supervision_class = "unlabeled" if message == "unlabeled" else "supervised_strong"
    row = _source_record(source_root, "attacked", 60, supervision_class=supervision_class)
    attack(row)
    source_manifest = _write_source_manifest(source_root, [row])
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_debug", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 23, message)
    assert not (tmp_path / "view" / "view_manifest.json").exists()


def test_uncalibrated_model_derived_supervised_weak_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    row = _source_record(
        source_root,
        "uncalibrated_model_weak",
        62,
        supervision_class="supervised_weak",
        semantic_origin="model_derived",
    )
    assert all(row["field_masks"][field] == 1 for field in SEMANTIC_FIELDS)
    assert all(row["field_uncertainty"][field]["state"] != "calibrated" for field in SEMANTIC_FIELDS)
    assert all(row["field_calibration_identity"][field] is None for field in SEMANTIC_FIELDS)
    source_manifest = _write_source_manifest(source_root, [row])
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_debug", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 23, "uncalibrated model-derived")
    assert not (tmp_path / "view" / "view_manifest.json").exists()


def test_oov_remains_distinct_and_noncontributing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    row = _source_record(source_root, "oov", 61, supervision_class="supervised_weak")
    row["targets"]["category"] = {"state": "oov", "value": None}
    row["field_masks"]["category"] = 0
    row["field_weights"]["category"] = 0.0
    row["provenance"]["raw_oov_category"] = "novel_widget"
    source_manifest = _write_source_manifest(source_root, [row])
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=1)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_debug", policy, [source_manifest], output)

    result = _read_jsonl(output / "record_manifest.jsonl")[0]
    assert result["targets"]["category"] == {"state": "oov", "value": None}
    assert result["field_masks"]["category"] == 0
    assert result["field_weights"]["category"] == 0


@pytest.mark.parametrize(
    "relation",
    [
        "exact_rgba",
        "recolor",
        "geometry",
        "declared_variant",
        "sheet",
        "hard_relation",
        "translation",
        "flip",
    ],
)
def test_verify_view_detects_leakage_tampering(tmp_path: Path, relation: str) -> None:
    source_root = tmp_path / "source"
    first = _source_record(source_root, "left", 70, requested_split="train")
    second = _source_record(source_root, "right", 71, requested_split="train")
    if relation == "recolor":
        original_second_blob = source_root / second["blob_path"]
        second = _source_record(
            source_root,
            "right",
            71,
            requested_split="train",
            mask_seed=70,
            color=(0, 0, 255),
            recolor_family_id="recolor:shared",
        )
        if original_second_blob != source_root / second["blob_path"]:
            original_second_blob.unlink()
        first["recolor_family_id"] = "recolor:shared"
    elif relation == "geometry":
        first["geometry_family_id"] = "geometry:shared"
        second["geometry_family_id"] = "geometry:shared"
    elif relation == "declared_variant":
        first["declared_variant_group_id"] = "variant:shared"
        second["declared_variant_group_id"] = "variant:shared"
    elif relation == "sheet":
        first["sheet_group_id"] = "sheet:shared"
        second["sheet_group_id"] = "sheet:shared"
    elif relation == "hard_relation":
        first["hard_relation_group_ids"] = ["hard:shared"]
        second["hard_relation_group_ids"] = ["hard:shared"]
    elif relation == "translation":
        first["known_translation_group_id"] = "shared"
        second["known_translation_group_id"] = "shared"
    elif relation == "flip":
        first["known_flip_group_id"] = "shared"
        second["known_flip_group_id"] = "shared"
    source_manifest = _write_source_manifest(source_root, [first, second])
    policy = _policy(tmp_path / "policy.json", "v5_debug", target_size=2)
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_debug", policy, [source_manifest], output)
    records_path = output / "record_manifest.jsonl"
    records = _read_jsonl(records_path)
    assert len(records) == 2
    records[1]["split"] = "test"
    if relation == "exact_rgba":
        records[1]["exported_rgba_hash"] = records[0]["exported_rgba_hash"]
    _write_jsonl(records_path, records)

    verification = verify_view(CONTRACT_ROOT, output)

    assert verification["ok"] is False
    diagnostic = json.dumps(verification, sort_keys=True).lower()
    assert "leak" in diagnostic or relation.split("_")[0] in diagnostic


def test_ood_identity_leakage_fails_build(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    train = _source_record(
        source_root,
        "train_reference",
        80,
        requested_split="train",
        geometry_family_id="geometry:shared",
    )
    ood = _source_record(
        source_root,
        "ood_candidate",
        81,
        requested_split="source_ood_test",
        geometry_family_id="geometry:shared",
        source_pack="held_pack",
        evaluation_candidate=True,
        evaluation_stratum="held_pack",
        source_ood=True,
        source_ood_scope="held_out_pack",
        source_ood_rationale="synthetic held-out pack",
    )
    source_manifest = _write_source_manifest(source_root, [train, ood])
    policy = _policy(tmp_path / "policy.json", "v5_source_ood", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_source_ood", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 25, "leak")


def test_source_manifest_and_blob_tampering_are_detected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    original_manifest = source_manifest.read_bytes()
    source_manifest.write_bytes(original_manifest + b"\n")
    source_verification = verify_view(CONTRACT_ROOT, output)
    assert source_verification["ok"] is False
    assert "source" in json.dumps(source_verification, sort_keys=True).lower()
    source_manifest.write_bytes(original_manifest)

    blob = source_root / rows[0]["blob_path"]
    original_blob = blob.read_bytes()
    blob.write_bytes(bytes([original_blob[0] ^ 1]) + original_blob[1:])
    blob_verification = verify_view(CONTRACT_ROOT, output)
    assert blob_verification["ok"] is False
    assert "blob" in json.dumps(blob_verification, sort_keys=True).lower()


def test_verify_freeze_detects_record_manifest_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    freeze_view(
        CONTRACT_ROOT,
        output,
        _approved_decisions(tmp_path / "approved.json"),
        "dataset-v5 freeze-view --view-root $VIEW_ROOT",
    )
    manifest_path = output / "record_manifest.jsonl"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    verification = verify_freeze(output)

    assert verification["ok"] is False
    assert "hash" in json.dumps(verification, sort_keys=True).lower()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exact_command_line", ""),
        ("resolved_policy_sha256", "0" * 64),
        ("source_binding_sha256", "0" * 64),
        ("freeze_kind", "unsupported_kind"),
    ],
)
def test_verify_freeze_rejects_tampered_freeze_metadata(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    freeze_view(
        CONTRACT_ROOT,
        output,
        _approved_decisions(tmp_path / "approved.json"),
        "dataset-v5 freeze-view --view-root $VIEW_ROOT",
    )
    freeze_path = output / "FREEZE.json"
    freeze = _read_json(freeze_path)
    freeze[field] = value
    _write_json(freeze_path, freeze)

    verification = verify_freeze(output)

    assert verification["ok"] is False


@pytest.mark.parametrize("artifact_name", ["split_manifest.json", "weight_manifest.jsonl", "validation_report.json"])
def test_verify_view_rejects_self_consistent_auxiliary_artifact_tampering(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    artifact = output / artifact_name
    if artifact_name == "split_manifest.json":
        value = _read_json(artifact)
        value["assignments"][0]["split"] = "test"
        _write_json(artifact, value)
    elif artifact_name == "weight_manifest.jsonl":
        values = _read_jsonl(artifact)
        values[0]["sampling_weight"] = 99.0
        _write_jsonl(artifact, values)
    else:
        value = _read_json(artifact)
        value["ok"] = False
        _write_json(artifact, value)
    manifest_path = output / "view_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["freeze_boundary"]["artifact_sha256"][artifact_name] = source_file_sha256(artifact)
    _write_json(manifest_path, manifest)

    verification = verify_view(CONTRACT_ROOT, output)

    assert verification["ok"] is False


def test_verify_view_rejects_self_consistent_record_rewrite_by_exact_source_replay(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    record_path = output / "record_manifest.jsonl"
    output_rows = _read_jsonl(record_path)
    rewritten = next(row for row in output_rows if row["sprite_id"] == "v5_architecture_strong")
    rewritten["canonical_object"] = "forged_sword"
    rewritten["targets"]["canonical_object"]["value"] = "forged_sword"
    _write_jsonl(record_path, output_rows)
    validation_path = output / "validation_report.json"
    validation = _read_json(validation_path)
    validation["artifact_sha256"]["record_manifest.jsonl"] = source_file_sha256(record_path)
    _write_json(validation_path, validation)
    manifest_path = output / "view_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["record_manifest_sha256"] = source_file_sha256(record_path)
    manifest["freeze_boundary"]["artifact_sha256"]["record_manifest.jsonl"] = source_file_sha256(record_path)
    manifest["freeze_boundary"]["artifact_sha256"]["validation_report.json"] = source_file_sha256(validation_path)
    _write_json(manifest_path, manifest)

    verification = verify_view(CONTRACT_ROOT, output)

    assert verification["ok"] is False
    assert any("deterministic replay of exact bound sources" in error for error in verification["errors"])


@pytest.mark.parametrize("tampered_blob_path", ["../outside.rgba", "/outside.rgba"])
def test_verify_view_rejects_escaping_or_absolute_output_blob_path(
    tmp_path: Path,
    tampered_blob_path: str,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_architecture")
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=len(rows))
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    record_path = output / "record_manifest.jsonl"
    output_rows = _read_jsonl(record_path)
    output_rows[0]["blob_path"] = tampered_blob_path
    _write_jsonl(record_path, output_rows)
    validation_path = output / "validation_report.json"
    validation = _read_json(validation_path)
    validation["artifact_sha256"]["record_manifest.jsonl"] = source_file_sha256(record_path)
    _write_json(validation_path, validation)
    manifest_path = output / "view_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["record_manifest_sha256"] = source_file_sha256(record_path)
    manifest["freeze_boundary"]["artifact_sha256"]["record_manifest.jsonl"] = source_file_sha256(record_path)
    manifest["freeze_boundary"]["artifact_sha256"]["validation_report.json"] = source_file_sha256(validation_path)
    _write_json(manifest_path, manifest)

    verification = verify_view(CONTRACT_ROOT, output)

    assert verification["ok"] is False
    assert "blob_path" in json.dumps(verification, sort_keys=True)


def test_failed_production_freeze_does_not_mutate_view_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_eval_balanced")
    source_manifest = _write_source_manifest(source_root, rows)
    policy_path = _policy(tmp_path / "policy.json", "v5_eval_balanced", target_size=len(rows))
    policy = _read_json(policy_path)
    policy["synthetic_fixture"] = False
    policy["production_freeze_authorized"] = True
    policy.pop("code_identity")
    _write_json(policy_path, policy)
    monkeypatch.setattr(
        named_views,
        "_git_identity",
        lambda policy=None: {"git_commit": "a" * 40, "dirty": False},
    )
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_eval_balanced", policy_path, [source_manifest], output)
    approvals_path = _approved_decisions(tmp_path / "approved.json")
    approvals = _read_json(approvals_path)
    approvals["approvals"]["production_freeze"] = True
    _write_json(approvals_path, approvals)
    manifest_path = output / "view_manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(DatasetV5ViewError) as captured:
        freeze_view(CONTRACT_ROOT, output, approvals_path, "   ")

    _assert_error(captured.value, 23, "command line")
    assert manifest_path.read_bytes() == before
    assert not (output / "FREEZE.json").exists()


def test_non_synthetic_production_cannot_spoof_clean_code_identity(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_eval_balanced")
    source_manifest = _write_source_manifest(source_root, rows)
    policy_path = _policy(tmp_path / "policy.json", "v5_eval_balanced", target_size=len(rows))
    policy = _read_json(policy_path)
    policy["synthetic_fixture"] = False
    policy["production_freeze_authorized"] = True
    policy["code_identity"] = {"git_commit": "a" * 40, "dirty": False}
    _write_json(policy_path, policy)
    output = tmp_path / "view"

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_eval_balanced", policy_path, [source_manifest], output)

    _assert_error(captured.value, 23, "code identity")
    assert not (output / "view_manifest.json").exists()


def test_production_freeze_requires_contract_approvals_even_if_policy_omits_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    rows = _view_rows(source_root, "v5_eval_balanced")
    source_manifest = _write_source_manifest(source_root, rows)
    policy_path = _policy(tmp_path / "policy.json", "v5_eval_balanced", target_size=len(rows))
    policy = _read_json(policy_path)
    policy["synthetic_fixture"] = False
    policy["production_freeze_authorized"] = True
    policy["approvals"] = {}
    policy.pop("code_identity")
    _write_json(policy_path, policy)
    monkeypatch.setattr(
        named_views,
        "_git_identity",
        lambda policy=None: {"git_commit": "a" * 40, "dirty": False},
    )
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_eval_balanced", policy_path, [source_manifest], output)
    decisions_path = _write_json(
        tmp_path / "approved.json",
        {"schema_version": APPROVAL_SCHEMA, "approvals": {"production_freeze": True}},
    )

    with pytest.raises(DatasetV5ViewError) as captured:
        freeze_view(
            CONTRACT_ROOT,
            output,
            decisions_path,
            "dataset-v5 freeze-view --view-root $VIEW_ROOT",
        )

    _assert_error(captured.value, 24, "approval")
    assert not (output / "FREEZE.json").exists()


def test_frozen_r2_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "r2"
    rows = _view_rows(source_root, "v5_unlabeled")
    source_manifest = _write_source_manifest(source_root, rows)
    freeze_manifest = _write_json(
        source_root / "freeze_manifest.json",
        {
            "schema_version": "unlabeled_pool_freeze_v1",
            "artifact_hashes": {source_manifest.name: source_file_sha256(source_manifest)},
            "blob_hashes": {row["blob_path"]: source_file_sha256(source_root / row["blob_path"]) for row in rows},
        },
    )
    binding = {
        "candidate_manifest": source_manifest.as_posix(),
        "candidate_manifest_sha256": "0" * 64,
        "freeze_manifest": freeze_manifest.as_posix(),
        "freeze_manifest_sha256": source_file_sha256(freeze_manifest),
        "expected_record_count": len(rows),
        "expected_geometry_family_count": len({row["geometry_family_id"] for row in rows}),
    }
    policy = _policy(
        tmp_path / "policy.json",
        "v5_unlabeled",
        target_size=len(rows),
        frozen_r2_binding=binding,
    )

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_unlabeled", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 22, "r2")
    assert not (tmp_path / "view" / "view_manifest.json").exists()


@pytest.mark.parametrize("missing_field", ["distribution_platform", "creator_lineage"])
def test_required_canonical_lineage_fields_fail_closed(tmp_path: Path, missing_field: str) -> None:
    source_root = tmp_path / "source"
    row = _source_record(source_root, "missing_lineage", 120)
    row.pop(missing_field)
    source_manifest = _write_source_manifest(source_root, [row])
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], tmp_path / "view")

    assert captured.value.reason_code == "provenance_failure"
    assert missing_field in str(captured.value)


@pytest.mark.parametrize("identity_field", ["sprite_id", "source_record_id"])
def test_identity_alias_collisions_report_both_record_ids(tmp_path: Path, identity_field: str) -> None:
    source_root = tmp_path / "source"
    first = _source_record(source_root, "identity_a", 121)
    second = _source_record(source_root, "identity_b", 122)
    second[identity_field] = first[identity_field]
    source_manifest = _write_source_manifest(source_root, [first, second])
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=2)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], tmp_path / "view")

    diagnostic = str(captured.value)
    assert identity_field in diagnostic
    assert first["record_id"] in diagnostic
    assert second["record_id"] in diagnostic


def test_source_blob_store_rejects_extra_blob(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    row = _source_record(source_root, "extra_blob", 123)
    source_manifest = _write_source_manifest(source_root, [row])
    (source_root / "blobs" / f"{'f' * 64}.rgba").write_bytes(bytes(32 * 32 * 4))
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], tmp_path / "view")

    assert "inventory" in str(captured.value).lower()
    assert captured.value.details["extra_or_unexpected"]


def test_source_blob_store_rejects_missing_blob(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    row = _source_record(source_root, "missing_blob", 124)
    source_manifest = _write_source_manifest(source_root, [row])
    (source_root / row["blob_path"]).unlink()
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 21, "missing source blob")


def test_creator_lineage_is_grouped_across_ordinary_requested_splits(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = [
        _source_record(
            source_root,
            "lineage_train",
            125,
            requested_split="train",
            creator_lineage="creator:shared",
        ),
        _source_record(
            source_root,
            "lineage_test",
            126,
            requested_split="test",
            creator_lineage="creator:shared",
        ),
    ]
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=2)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    result = _read_jsonl(output / "record_manifest.jsonl")
    assert {row["split"] for row in result} == {"train"}


def test_source_ood_creator_lineage_crossing_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    regular = _source_record(source_root, "regular_lineage", 127, creator_lineage="creator:shared")
    held_out = _source_record(
        source_root,
        "held_lineage",
        128,
        creator_lineage="creator:shared",
        requested_split="source_ood_test",
        source_pack="held_pack",
        evaluation_candidate=True,
        evaluation_stratum="held_pack",
        source_ood=True,
        source_ood_scope="held_out_pack",
        source_ood_rationale="synthetic held-out pack",
    )
    source_manifest = _write_source_manifest(source_root, [regular, held_out])
    policy = _policy(tmp_path / "policy.json", "v5_source_ood", target_size=1)

    with pytest.raises(DatasetV5ViewError) as captured:
        build_view(CONTRACT_ROOT, "v5_source_ood", policy, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 25, "leak")


def test_geometry_closure_groups_different_creator_lineages(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    rows = [
        _source_record(
            source_root,
            "geometry_train",
            129,
            requested_split="train",
            creator_lineage="creator:left",
            geometry_family_id="geometry:shared",
        ),
        _source_record(
            source_root,
            "geometry_test",
            130,
            requested_split="test",
            creator_lineage="creator:right",
            geometry_family_id="geometry:shared",
        ),
    ]
    source_manifest = _write_source_manifest(source_root, rows)
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=2)
    output = tmp_path / "view"

    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)

    assert {row["split"] for row in _read_jsonl(output / "record_manifest.jsonl")} == {"train"}


def test_side_provenance_must_exactly_match_canonical_record(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    row = _source_record(source_root, "provenance_mismatch", 131)
    source_manifest = _write_source_manifest(source_root, [row])
    policy = _policy(tmp_path / "policy.json", "v5_architecture", target_size=1)
    output = tmp_path / "view"
    build_view(CONTRACT_ROOT, "v5_architecture", policy, [source_manifest], output)
    side_path = output / "license_provenance.jsonl"
    side = _read_jsonl(side_path)
    side[0]["distribution_platform"] = "contradictory-platform"
    _write_jsonl(side_path, side)

    verification = verify_view(CONTRACT_ROOT, output)

    assert verification["ok"] is False
    assert "license/provenance manifest mismatch" in "\n".join(verification["errors"])


def test_explicit_synthetic_os_temp_fixture_is_accepted(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-v5-synthetic-") as directory:
        source_root = Path(directory) / "source"
        row = _source_record(source_root, "os_temp_synthetic", 132)
        source_manifest = _write_source_manifest(source_root, [row])
        policy = _policy(Path(directory) / "policy.json", "v5_architecture", target_size=1)

        result = build_view(
            CONTRACT_ROOT,
            "v5_architecture",
            policy,
            [source_manifest],
            tmp_path / "view",
        )

    assert result["ok"] is True


def test_unmarked_os_temp_source_is_rejected(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-v5-unmarked-") as directory:
        source_root = Path(directory) / "source"
        row = _source_record(source_root, "os_temp_unmarked", 133)
        source_manifest = _write_source_manifest(source_root, [row])
        policy_path = _policy(Path(directory) / "policy.json", "v5_architecture", target_size=1)
        policy = _read_json(policy_path)
        policy.pop("synthetic_fixture")
        policy.pop("code_identity")
        _write_json(policy_path, policy)

        with pytest.raises(DatasetV5ViewError) as captured:
            build_view(CONTRACT_ROOT, "v5_architecture", policy_path, [source_manifest], tmp_path / "view")

    _assert_error(captured.value, 23, "repository-relative")


def test_production_or_candidate_build_marked_synthetic_is_rejected(tmp_path: Path) -> None:
    policy_path = _policy(tmp_path / "policy.json", "v5_architecture", target_size=1)
    policy = _read_json(policy_path)
    policy["view_status"] = "candidate"
    policy["production_freeze_authorized"] = True
    _write_json(policy_path, policy)

    with pytest.raises(DatasetV5ViewError) as captured:
        named_views._load_policy(policy_path, "v5_architecture")

    assert "synthetic fixture status" in str(captured.value)

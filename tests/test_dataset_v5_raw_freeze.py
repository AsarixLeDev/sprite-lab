from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from spritelab.dataset_v5.raw_freeze import (
    ArtifactVerificationError,
    FreezeGateError,
    FreezeGateEvidence,
    freeze_candidate_dataset,
    verify_candidate_dataset,
    verify_deterministic_rebuild,
    verify_frozen_rebuild,
    write_candidate_dataset,
)
from spritelab.dataset_v5.raw_relations import (
    HARD_RELATION_KINDS,
    HardRelationLeakageError,
    assign_component_splits,
    build_relation_manifest,
    validate_relation_manifest,
)
from spritelab.dataset_v5.raw_views import VIEW_NAMES, RawViewError, build_candidate_views


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(index: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "audit_status": "passed",
        "blob_id": _digest(f"blob:{index}"),
        "category": "item",
        "creator_lineage": f"creator:{index}",
        "field_masks": {},
        "geometry_family_id": f"geo_{_digest(f'geometry:{index}')}",
        "license": {"license": "cc0"},
        "original_filename": f"misleading_{index}.png",
        "output_height": 16 + index,
        "output_width": 16 + index,
        "provenance_status": "expected_archive_hash_verified",
        "record_id": f"rec_{_digest(f'record:{index}')}",
        "source_binding_valid": True,
        "source_pack": f"pack:{index}",
        "suitability_status": "accept",
        "supervision_class": "supervised_weak",
    }
    row.update(updates)
    return row


def _candidate_fixture(tmp_path: Path, name: str = "candidate") -> tuple[Path, list[dict], dict, dict]:
    records = [
        _record(1, category="tool"),
        _record(2, category="gem", supervision_class="unlabeled"),
        _record(3, category="tool"),
    ]
    relations = build_relation_manifest(records)
    views = build_candidate_views(records, relations)
    root = tmp_path / name
    write_candidate_dataset(
        root,
        records=records,
        relation_manifest=relations,
        view_bundle=views,
        gate_evidence=FreezeGateEvidence.passing(),
    )
    return root, records, relations, views


def test_discovers_all_required_hard_relations_with_opaque_content_bound_ids() -> None:
    first = _record(
        1,
        alpha_mask_sha256="alpha-shared",
        blob_id="a" * 64,
        creator_lineage="creator shared",
        declared_variant_ids=["declared shared"],
        flip_family_id="flip shared",
        geometry_family_id=f"geo_{'b' * 64}",
        hard_relation_group_ids=["other shared"],
        sheet_id="sheet shared",
        source_pack="pack shared",
        translation_family_id="translation shared",
    )
    second = _record(
        2,
        alpha_mask_sha256="alpha-shared",
        blob_id="a" * 64,
        creator_lineage="creator shared",
        declared_variant_ids=["declared shared"],
        flip_family_id="flip shared",
        geometry_family_id=f"geo_{'b' * 64}",
        hard_relation_group_ids=["other shared"],
        sheet_id="sheet shared",
        source_pack="pack shared",
        translation_family_id="translation shared",
    )

    manifest = build_relation_manifest([second, first])

    assert {relation["kind"] for relation in manifest["relations"]} == set(HARD_RELATION_KINDS)
    assert len(manifest["hard_relation_components"]) == 1
    serialized = json.dumps(manifest, sort_keys=True)
    for tainted_value in (
        "creator shared",
        "declared shared",
        "flip shared",
        "other shared",
        "pack shared",
        "sheet shared",
        "translation shared",
    ):
        assert tainted_value not in serialized
    for relation in manifest["relations"]:
        assert re.fullmatch(r"rel_[0-9a-f]{64}", relation["relation_id"])
    assert re.fullmatch(r"grp_[0-9a-f]{64}", manifest["hard_relation_components"][0]["component_id"])
    assert validate_relation_manifest([first, second], manifest)["ok"] is True


def test_semantic_pack_and_filename_renaming_does_not_change_relation_ids() -> None:
    records = [_record(1, source_pack="helmet pack"), _record(2, source_pack="helmet pack")]
    renamed = [
        {**records[0], "original_filename": "mineral.png", "source_pack": "mineral pack"},
        {**records[1], "original_filename": "sword.png", "source_pack": "mineral pack"},
    ]

    assert build_relation_manifest(records) == build_relation_manifest(renamed)


def test_dsu_closure_prevents_hard_relation_split_crossing() -> None:
    first = _record(1, blob_id="f" * 64, requested_split="train")
    second = _record(2, blob_id="f" * 64, requested_split="test")
    manifest = build_relation_manifest([first, second])

    with pytest.raises(HardRelationLeakageError, match="conflicting split requests"):
        assign_component_splits([first, second], manifest)

    first.pop("requested_split")
    second.pop("requested_split")
    manifest = build_relation_manifest([first, second])
    assignments = assign_component_splits([first, second], manifest)
    assert len(set(assignments.values())) == 1


def test_geometry_facts_infer_recolors_translations_and_cropped_sheet_membership() -> None:
    geometry = f"geo_{'c' * 64}"
    archive = "e" * 64
    first = _record(
        1,
        archive_member_path="sheet/items.png",
        crop_coordinates=[0, 0, 16, 16],
        geometry_family_id=geometry,
        output_height=16,
        output_width=16,
        source_archive_sha256=archive,
        tight_foreground_bbox=[2, 2, 10, 10],
    )
    recolor = _record(
        2,
        archive_member_path="sheet/items.png",
        crop_coordinates=[16, 0, 32, 16],
        geometry_family_id=geometry,
        output_height=16,
        output_width=16,
        source_archive_sha256=archive,
        tight_foreground_bbox=[2, 2, 10, 10],
    )
    translated = _record(
        3,
        geometry_family_id=geometry,
        output_height=16,
        output_width=16,
        tight_foreground_bbox=[3, 2, 11, 10],
    )

    manifest = build_relation_manifest([first, recolor, translated])
    by_kind = {relation["kind"]: relation for relation in manifest["relations"]}

    assert {first["record_id"], recolor["record_id"]}.issubset(by_kind["alpha_recolor"]["members"])
    assert set(by_kind["translation"]["members"]) == {
        first["record_id"],
        recolor["record_id"],
        translated["record_id"],
    }
    assert set(by_kind["sheet"]["members"]) == {first["record_id"], recolor["record_id"]}


def test_inconsistent_decoded_rgba_aliases_fail_closed() -> None:
    row = _record(1, output_decoded_rgba_sha256="0" * 64)

    with pytest.raises(ValueError, match="inconsistent decoded RGBA"):
        build_relation_manifest([row])


def test_exact_duplicates_add_zero_architecture_value_and_all_views_exist() -> None:
    duplicate_blob = "d" * 64
    records = [
        _record(1, blob_id=duplicate_blob, category="tool"),
        _record(2, blob_id=duplicate_blob, category="tool"),
        _record(3, category="gem", supervision_class="unlabeled"),
        _record(4, category="plant", source_ood=True),
        _record(5, category="unknown", open_set=True, target_state="unknown"),
        _record(6, suitability_status="quarantine"),
        _record(7, suitability_status="reject"),
    ]
    manifest = build_relation_manifest(records)

    bundle = build_candidate_views(records, manifest)

    assert tuple(bundle["views"]) == VIEW_NAMES
    architecture_ids = {row["record_id"] for row in bundle["views"]["v5_architecture"]["records"]}
    assert len(architecture_ids & {records[0]["record_id"], records[1]["record_id"]}) == 1
    assert bundle["views"]["v5_source_ood"]["record_count"] == 1
    assert bundle["views"]["v5_open_set"]["record_count"] == 1
    assert bundle["views"]["v5_unlabeled"]["record_count"] == 1
    assert records[5]["record_id"] not in {row["record_id"] for row in bundle["views"]["v5_debug"]["records"]}
    serialized = json.dumps(bundle, sort_keys=True)
    assert "misleading_" not in serialized
    assert bundle["candidate_only"] is True
    assert bundle["production_frozen"] is False
    assert bundle["promotion_forbidden"] is True
    assert bundle["training_authorized"] is False


def test_view_membership_requires_explicit_suitability_not_semantic_candidate() -> None:
    row = _record(1)
    row.pop("suitability_status")
    row["inclusion_decision"] = "candidate"
    relations = build_relation_manifest([row])

    with pytest.raises(RawViewError, match="missing explicit suitability"):
        build_candidate_views([row], relations)


def test_balanced_eval_uses_observed_minimum_without_forcing_target_size() -> None:
    records = [
        _record(1, category="tool", evaluation_candidate=True),
        _record(2, category="tool", evaluation_candidate=True),
        _record(3, category="gem", evaluation_candidate=True),
    ]
    relations = build_relation_manifest(records)

    bundle = build_candidate_views(records, relations)
    evaluation = bundle["views"]["v5_eval_balanced"]["records"]

    assert len(evaluation) == 2
    assert {row["category"] for row in evaluation} == {"tool", "gem"}
    assert {row["split"] for row in evaluation} == {"test"}


def test_candidate_is_fresh_and_has_recursive_immutable_manifest(tmp_path: Path) -> None:
    candidate, records, relations, views = _candidate_fixture(tmp_path)

    result = verify_candidate_dataset(candidate)

    assert result["ok"] is True
    assert result["candidate_only"] is True
    assert result["production_frozen"] is False
    artifact = json.loads((candidate / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "created_at" not in json.dumps(artifact)
    assert "timestamp" not in json.dumps(artifact)
    with pytest.raises(FileExistsError):
        write_candidate_dataset(
            candidate,
            records=records,
            relation_manifest=relations,
            view_bundle=views,
            gate_evidence=FreezeGateEvidence.passing(),
        )


def test_candidate_creation_requires_every_automated_gate(tmp_path: Path) -> None:
    records = [_record(1)]
    relations = build_relation_manifest(records)
    views = build_candidate_views(records, relations)
    candidate = tmp_path / "candidate"
    evidence = replace(FreezeGateEvidence.passing(), sol_audit_completed=False)

    with pytest.raises(FreezeGateError, match="sol_audit_completed"):
        write_candidate_dataset(
            candidate,
            records=records,
            relation_manifest=relations,
            view_bundle=views,
            gate_evidence=evidence,
        )

    assert not candidate.exists()


@pytest.mark.parametrize(
    "failed_gate",
    [
        "critical_conflicts_resolved",
        "filename_leakage_free",
        "provenance_complete",
        "licenses_valid",
        "masks_valid",
        "audit_batches_passed",
    ],
)
def test_freeze_is_blocked_by_each_required_safety_gate(tmp_path: Path, failed_gate: str) -> None:
    candidate, _, _, _ = _candidate_fixture(tmp_path)
    evidence = replace(FreezeGateEvidence.passing(), **{failed_gate: False})
    frozen = tmp_path / f"frozen-{failed_gate}"

    with pytest.raises(FreezeGateError, match=failed_gate):
        freeze_candidate_dataset(candidate, frozen, evidence)

    assert not frozen.exists()


def test_production_freeze_keeps_training_unauthorized_and_candidate_unchanged(tmp_path: Path) -> None:
    candidate, _, _, _ = _candidate_fixture(tmp_path)
    before = {path.relative_to(candidate): path.read_bytes() for path in candidate.rglob("*") if path.is_file()}
    frozen = tmp_path / "frozen"

    result = freeze_candidate_dataset(candidate, frozen, FreezeGateEvidence.passing())

    assert result["ok"] is True
    assert result["production_frozen"] is True
    assert result["training_authorized"] is False
    assert result["candidate_only"] is False
    assert verify_frozen_rebuild(frozen)["ok"] is True
    after = {path.relative_to(candidate): path.read_bytes() for path in candidate.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("tamper", ["changed", "missing", "extra"])
def test_frozen_verification_detects_changed_missing_and_extra_artifacts(tmp_path: Path, tamper: str) -> None:
    candidate, _, _, _ = _candidate_fixture(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_candidate_dataset(candidate, frozen, FreezeGateEvidence.passing())

    if tamper == "changed":
        path = frozen / "record_manifest.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "missing":
        (frozen / "record_manifest.jsonl").unlink()
    else:
        (frozen / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(ArtifactVerificationError):
        verify_frozen_rebuild(frozen)


def test_two_complete_candidate_and_frozen_rebuilds_are_byte_identical(tmp_path: Path) -> None:
    first, records, relations, views = _candidate_fixture(tmp_path, "candidate-a")
    second = tmp_path / "candidate-b"
    write_candidate_dataset(
        second,
        records=records,
        relation_manifest=relations,
        view_bundle=views,
        gate_evidence=FreezeGateEvidence.passing(),
    )

    assert verify_deterministic_rebuild(first, second)["byte_identical"] is True

    frozen_a = tmp_path / "frozen-a"
    frozen_b = tmp_path / "frozen-b"
    freeze_candidate_dataset(first, frozen_a, FreezeGateEvidence.passing())
    freeze_candidate_dataset(second, frozen_b, FreezeGateEvidence.passing())
    comparison = verify_frozen_rebuild(frozen_a, frozen_b)
    assert comparison["byte_identical"] is True


def test_candidate_creation_rejects_direct_leakage_without_leaving_output(tmp_path: Path) -> None:
    records = [_record(1, filename_leakage_detected=True)]
    relations = build_relation_manifest(records)
    views = build_candidate_views(records, relations)
    candidate = tmp_path / "candidate"

    with pytest.raises(FreezeGateError, match="filename leakage"):
        write_candidate_dataset(
            candidate,
            records=records,
            relation_manifest=relations,
            view_bundle=views,
            gate_evidence=FreezeGateEvidence.passing(),
        )

    assert not candidate.exists()

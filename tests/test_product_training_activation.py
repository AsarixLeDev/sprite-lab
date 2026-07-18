from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

import spritelab.training.campaign as campaign_module
from spritelab.product_features.conditioned_v5.audit_receipts import (
    build_audit_action_record,
    build_audit_receipt,
)
from spritelab.product_features.conditioned_v5.identity import (
    TRUSTED_AUDITOR_IDS,
    trusted_auditor_inventory,
)
from spritelab.product_features.conditioned_v5.publication_commit import (
    PUBLICATION_JOURNAL_NAME,
    build_campaign_commit,
    build_dataset_commit,
    build_publication_journal,
    campaign_commit_name,
    canonical_publication_commit_bytes,
    dataset_commit_name,
)
from spritelab.product_features.training import activation as activation_module
from spritelab.product_features.training import audit as training_audit_module
from spritelab.product_features.training.activation import (
    CONDITIONED_DATASET_FREEZE_SCHEMA,
    CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
    MANDATORY_TRAINING_AUDIT_GATES,
    TRAINING_AUDIT_HASHES_SCHEMA,
    TRAINING_AUDIT_REPORT_SCHEMA,
    ConditionedActivationError,
    ConditionedTrainingActivation,
    build_conditioned_three_seed_campaign,
    load_conditioned_training_activation,
    training_audit_status,
)
from spritelab.product_features.training.activation_commit import (
    ACTIVATION_PROJECT_COMMIT_NAME,
    build_activation_commit_documents,
    build_activation_project_commit,
    canonical_activation_commit_bytes,
)
from spritelab.product_features.training.models import TrainingProfile
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, stable_hash
from spritelab.v3.config import DEFAULT_CONFIG, ConfigError, ProjectConfig
from spritelab.v3.model import AuditStatus


@dataclass(frozen=True)
class ActivationProject:
    config: ProjectConfig
    freeze: Path
    publication: Path
    campaign_path: Path
    portable_campaign: dict[str, Any]


def test_project_config_load_uses_immutable_activation_marker_without_mutating_canonical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    before_values = deepcopy(DEFAULT_CONFIG)
    before_values["project"]["name"] = "marker-overlay"
    after_values = deepcopy(before_values)
    after_values["dataset"]["view_manifest"] = "datasets/freeze/view_manifest.json"
    after_values["dataset"]["freeze_manifest"] = "datasets/freeze/activation.json"
    after_values["training"]["dataset_freeze"] = "datasets/freeze/activation.json"
    after_values["training"]["campaign_config"] = "campaigns/freeze/campaign.json"
    after_values["execution"]["allow_dataset_production_freeze"] = True
    after_values["execution"]["allow_training"] = True
    before = yaml.safe_dump(before_values, sort_keys=False, allow_unicode=True).encode("utf-8")
    after = yaml.safe_dump(after_values, sort_keys=False, allow_unicode=True).encode("utf-8")
    config_path = root / "spritelab.yaml"
    config_path.write_bytes(before)
    outside = tmp_path / "outside-marker-sentinel.bin"
    outside.write_bytes(b"preserve")
    job_id = "conditioned-0123456789abcdefabcd"
    receipt, journal, record = build_activation_commit_documents(
        job_id=job_id,
        operation_id="activation-0123456789abcdef0123456789abcdef",
        candidate_identity="1" * 64,
        publication_identity_sha256="2" * 64,
        activation_manifest_sha256="3" * 64,
        campaign_config_sha256="4" * 64,
        campaign_identity_sha256="5" * 64,
        authorization_id_sha256="6" * 64,
        config_before_sha256=hashlib.sha256(before).hexdigest(),
        config_after_sha256=hashlib.sha256(after).hexdigest(),
        prepared_at="2026-07-18T00:03:00+00:00",
    )
    receipt_root = root / "runs" / "v3" / "conditioned-dataset-v5" / job_id / "activation_receipt"
    receipt_root.mkdir(parents=True)
    for name, value in (("receipt.json", receipt), ("journal.json", journal), ("record.json", record)):
        (receipt_root / name).write_bytes(canonical_activation_commit_bytes(value))
    marker = build_activation_project_commit(
        receipt=receipt,
        journal=journal,
        record=record,
        config_after_bytes=after,
    )
    (root / ACTIVATION_PROJECT_COMMIT_NAME).write_bytes(canonical_activation_commit_bytes(marker))

    loaded = ProjectConfig.load(config_path)

    assert loaded.values["project"]["name"] == "marker-overlay"
    assert loaded.values["training"]["campaign_config"] == "campaigns/freeze/campaign.json"
    assert loaded.values["execution"]["allow_training"] is True
    assert config_path.read_bytes() == before
    assert outside.read_bytes() == b"preserve"

    third_values = deepcopy(before_values)
    third_values["project"]["name"] = "foreign-third-config"
    third = yaml.safe_dump(third_values, sort_keys=False, allow_unicode=True).encode("utf-8")
    config_path.write_bytes(third)
    with pytest.raises(ConfigError, match="immutable project activation commit"):
        ProjectConfig.load(config_path)
    assert config_path.read_bytes() == third
    assert outside.read_bytes() == b"preserve"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_mapping_with_shadowed_key(
    path: Path,
    value: dict[str, Any],
    *,
    key: str,
    shadow_value: Any,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_yaml = path.suffix.casefold() in {".yaml", ".yml"}
    text = yaml.safe_dump(value, sort_keys=True) if is_yaml else json.dumps(value, indent=2, sort_keys=True) + "\n"
    marker = f"{key}:" if is_yaml else f'"{key}":'
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.lstrip().startswith(marker)]
    assert len(matches) == 1
    index = matches[0]
    indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    encoded_shadow = json.dumps(shadow_value)
    separator = "" if is_yaml else ","
    lines.insert(index, f"{indentation}{marker} {encoded_shadow}{separator}\n")
    path.write_text("".join(lines), encoding="utf-8")
    parsed = (
        yaml.safe_load(path.read_text(encoding="utf-8")) if is_yaml else json.loads(path.read_text(encoding="utf-8"))
    )
    assert isinstance(parsed, dict)
    return parsed


def _assert_sanitized_mapping_error(error: ConditionedActivationError) -> None:
    assert error.code == "activation_mapping"
    assert error.public_message == "An activation mapping is unreadable."
    assert str(error) == error.public_message


def _stabilize_campaign_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep parser tests independent of repository-wide code-identity churn."""

    monkeypatch.setattr(
        activation_module,
        "validate_campaign",
        lambda _campaign: {"errors": [], "blockers": [], "launch_ready": True},
    )


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _inventory(root: Path, *, exclude: set[str] = frozenset()) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _inventory_payload(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "total_bytes": sum(item["byte_count"] for item in files.values()),
    }


def _fixture_audit_subjects(
    image_count: int, category_counts: dict[str, int], source_counts: dict[str, int]
) -> dict[str, Any]:
    sprite_ids = [f"sprite-{index:04d}" for index in range(image_count)]
    stratified = sprite_ids[:40]
    base = {
        "schema_version": activation_module.CONDITIONED_AUDIT_SUBJECTS_SCHEMA,
        "stratified_sample_ids": stratified,
        "low_confidence_ids": [],
        "disagreement_ids": [],
        "high_impact_ids": [],
        "generic_label_ids": [],
        "required_label_audit_ids": stratified,
        "visual_descriptor_bindings": [
            {
                "sprite_id": sprite_id,
                "descriptor_identity": hashlib.sha256(f"descriptor:{sprite_id}".encode()).hexdigest(),
                "decoded_rgba_sha256": hashlib.sha256(f"rgba:{sprite_id}".encode()).hexdigest(),
            }
            for sprite_id in sprite_ids
        ],
        "local_pixel_vision_algorithm": "local_pixel_vision_v1",
        "local_pixel_vision_config_identity": "1" * 64,
        "distributions": {
            "category": category_counts,
            "source": source_counts,
            "confidence": {"source_grounded_high": image_count},
            "confidence_reason": {"source_filename": image_count},
            "disagreement": {"no_disagreement": image_count},
            "generic_label": {"specific": image_count},
        },
        "all_low_confidence_required": True,
        "all_disagreements_required": True,
        "all_high_impact_required": True,
        "all_generic_labels_required": True,
        "all_visual_descriptors_recompute_required": True,
        "quality_rates_basis_points": {
            "unknown_category": 0,
            "generic_object": 0,
            "disagreement": 0,
            "useful_label": 10_000,
        },
        "human_truth_claim": False,
    }
    return {**base, "subjects_identity": stable_hash(base)}


def _fixture_evidence(
    kind: str,
    *,
    candidate_identity: str,
    payload_inventory_sha256: str,
    image_count: int,
    production_code_identity: str,
    subjects: dict[str, Any],
    subject_files: dict[str, dict[str, Any]],
    category_counts: dict[str, int],
    source_counts: dict[str, int],
    split_counts: dict[str, int],
    benchmark_category_counts: dict[str, int],
    near_duplicate_config_identity: str,
    retained_gate: dict[str, Any],
) -> dict[str, Any]:
    is_label = kind == "label_audit"
    inventory = trusted_auditor_inventory(kind)
    metrics = (
        {
            "audited_record_ids": subjects["required_label_audit_ids"],
            "stratified_sample_ids": subjects["stratified_sample_ids"],
            "low_confidence_ids": subjects["low_confidence_ids"],
            "disagreement_ids": subjects["disagreement_ids"],
            "high_impact_ids": subjects["high_impact_ids"],
            "generic_label_ids": subjects["generic_label_ids"],
            "distributions": subjects["distributions"],
            "quality_rates_basis_points": subjects["quality_rates_basis_points"],
            "recomputed_visual_descriptor_bindings": subjects["visual_descriptor_bindings"],
            "local_pixel_vision_config_identity": subjects["local_pixel_vision_config_identity"],
        }
        if is_label
        else {
            "split_counts": split_counts,
            "category_counts": category_counts,
            "source_counts": source_counts,
            "benchmark_category_counts": benchmark_category_counts,
            "payload_inventory_sha256": payload_inventory_sha256,
            "verified_file_count": len(subject_files),
            "near_duplicate_recomputation": {
                "algorithm_id": activation_module.CONDITIONED_NEAR_DUPLICATE_ALGORITHM,
                "config_identity": near_duplicate_config_identity,
                "retained_count": image_count,
                "checked_same_category_pairs": sum(count * (count - 1) // 2 for count in category_counts.values()),
                "violation_count": 0,
                "gate_identity": retained_gate["gate_identity"],
            },
        }
    )
    report = {
        "schema_version": (
            activation_module.CONDITIONED_LABEL_AUDIT_SCHEMA
            if is_label
            else activation_module.CONDITIONED_VALIDATION_SCHEMA
        ),
        "verdict": "PASS",
        "independent": True,
        "generated_by_conditioned_workflow": False,
        "auditor": {
            "auditor_id": TRUSTED_AUDITOR_IDS[kind],
            "code_identity_sha256": inventory["inventory_sha256"],
            "implementation_inventory": inventory,
        },
        "bindings": {
            "candidate_identity": candidate_identity,
            "payload_inventory_sha256": payload_inventory_sha256,
            "image_count": image_count,
            "production_code_identity": production_code_identity,
            "label_audit_subjects_identity": subjects["subjects_identity"],
        },
        "subject_files": subject_files,
        "checks": dict.fromkeys(
            activation_module.CONDITIONED_LABEL_AUDIT_GATES
            if is_label
            else activation_module.CONDITIONED_VALIDATION_GATES,
            "PASS",
        ),
        "audit_subjects": subjects,
        "metrics": metrics,
    }
    return {**report, "audit_run_identity": stable_hash(report)}


def _activation_project(tmp_path: Path) -> ActivationProject:
    root = tmp_path / "project"
    publication = root / "artifacts" / "dataset" / "conditioned-v5"
    campaign_directory = root / "artifacts" / "training"
    output_root = root / "runs" / "training"
    publication.mkdir(parents=True)
    campaign_directory.mkdir(parents=True)
    image_count = 2_417
    category_counts = {"food": 605, "plant": 604, "potion": 604, "terrain": 604}
    source_counts = {"source.one": 1_209, "source.two": 1_208}
    split_counts = {"test": 242, "train": 1_933, "val": 242}
    benchmark_category_counts = {"food": 1, "plant": 1, "potion": 1, "terrain": 1}
    source_bindings = [
        {
            "source_id": "source.one",
            "license_id": "cc0-1.0",
            "managed_intake_receipt_identity": "2" * 64,
        },
        {
            "source_id": "source.two",
            "license_id": "cc0-1.0",
            "managed_intake_receipt_identity": "3" * 64,
        },
    ]
    production_code_identity = "4" * 64
    near_duplicate_config_identity = "5" * 64
    retained_gate_base = {
        "algorithm_id": activation_module.CONDITIONED_NEAR_DUPLICATE_ALGORITHM,
        "config": {"fixture": "exact-retained-pair-policy"},
        "config_identity": near_duplicate_config_identity,
        "retained_count": image_count,
        "violation_count": 0,
        "violations": [],
        "ok": True,
    }
    retained_gate = {**retained_gate_base, "gate_identity": stable_hash(retained_gate_base)}
    subjects = _fixture_audit_subjects(image_count, category_counts, source_counts)
    _write_json(publication / "conditioned_records.jsonl", {"fixture": "conditioned records"})
    _write_json(publication / "training_manifest.jsonl", {"fixture": "training manifest"})
    _write_json(publication / "conditioning_vocabulary.json", {"fixture": "conditioning vocabulary"})
    _write_json(
        publication / "benchmark_manifest.json",
        {"schema_version": "spritelab.dataset.conditioned-benchmark.v1", "category_counts": benchmark_category_counts},
    )
    _write_json(
        publication / "coverage_report.json",
        {
            "schema_version": "spritelab.dataset.conditioned-coverage.v1",
            "category_counts": category_counts,
            "source_counts": source_counts,
            "split_counts": split_counts,
        },
    )
    _write_json(publication / "split_integrity_report.json", {"ok": True})
    _write_json(
        publication / "provenance_manifest.json",
        {
            "schema_version": activation_module.CONDITIONED_PROVENANCE_SCHEMA,
            "sources": source_bindings,
            "all_source_files_rehashed": True,
            "license_policy": ["cc0-1.0", "public-domain"],
            "paths_are_portable": True,
        },
    )
    _write_json(
        publication / "duplicate_report.json",
        {
            "schema_version": "spritelab.dataset.conditioned-duplicates.v2",
            "near_duplicate_algorithm": activation_module.CONDITIONED_NEAR_DUPLICATE_ALGORITHM,
            "near_duplicate_config_identity": near_duplicate_config_identity,
            "near_duplicate_implementation_code_inventory_sha256": production_code_identity,
            "retained_near_duplicate_gate": retained_gate,
        },
    )
    _write_json(publication / "label_audit_subjects.json", subjects)
    records_hash = file_sha256(publication / "conditioned_records.jsonl")
    view_base = {
        "schema_version": activation_module.CONDITIONED_VIEW_SCHEMA,
        "view_identity": stable_hash(
            {
                "managed_intake_receipt_identities": [
                    source["managed_intake_receipt_identity"] for source in source_bindings
                ],
                "image_count": image_count,
                "records_sha256": records_hash,
            }
        ),
        "image_count": image_count,
        "records_path": "conditioned_records.jsonl",
        "records_sha256": records_hash,
        "training_manifest_path": "training_manifest.jsonl",
        "training_manifest_sha256": file_sha256(publication / "training_manifest.jsonl"),
        "split_integrity_sha256": file_sha256(publication / "split_integrity_report.json"),
        "coverage_report_sha256": file_sha256(publication / "coverage_report.json"),
        "requires_semantic_labels": True,
        "human_truth_claim": False,
        "paths_are_portable": True,
    }
    _write_json(publication / "view_manifest.json", view_base)

    candidate_files = _inventory(publication)
    payload_inventory_sha256 = stable_hash(_inventory_payload(candidate_files))
    candidate_identity = stable_hash(
        {
            "schema_version": activation_module.CONDITIONED_CANDIDATE_SCHEMA,
            "input_bindings": source_bindings,
            "production_code_identity": production_code_identity,
            "payload_inventory_sha256": payload_inventory_sha256,
            "image_count": image_count,
            "recipe": activation_module.CONDITIONED_RECIPE,
        }
    )
    evidence_args = {
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": payload_inventory_sha256,
        "image_count": image_count,
        "production_code_identity": production_code_identity,
        "subjects": subjects,
        "subject_files": candidate_files,
        "category_counts": category_counts,
        "source_counts": source_counts,
        "split_counts": split_counts,
        "benchmark_category_counts": benchmark_category_counts,
        "near_duplicate_config_identity": near_duplicate_config_identity,
        "retained_gate": retained_gate,
    }
    artifacts = {
        "view_manifest": publication / "view_manifest.json",
        "split_manifest": publication / "training_manifest.jsonl",
        "conditioning_vocabulary": publication / "conditioning_vocabulary.json",
        "benchmark_manifest": publication / "benchmark_manifest.json",
        "labeling_audit": publication / "evidence" / "label_audit.json",
        "labeling_audit_receipt": publication / "evidence" / "label_audit_receipt.json",
        "labeling_audit_action_record": publication / "evidence" / "label_audit_action.json",
        "validation_report": publication / "evidence" / "dataset_validation.json",
        "validation_receipt": publication / "evidence" / "dataset_validation_receipt.json",
        "validation_action_record": publication / "evidence" / "dataset_validation_action.json",
    }
    _write_json(artifacts["labeling_audit"], _fixture_evidence("label_audit", **evidence_args))
    _write_json(artifacts["validation_report"], _fixture_evidence("dataset_validation", **evidence_args))
    candidate_receipt_context = {
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": payload_inventory_sha256,
        "image_count": image_count,
    }
    for index, (kind, report_name, receipt_name, action_name) in enumerate(
        (
            (
                "label_audit",
                "labeling_audit",
                "labeling_audit_receipt",
                "labeling_audit_action_record",
            ),
            (
                "dataset_validation",
                "validation_report",
                "validation_receipt",
                "validation_action_record",
            ),
        ),
        start=1,
    ):
        report_path = artifacts[report_name]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        receipt = build_audit_receipt(
            kind=kind,
            job_id="conditioned-0123456789abcdefabcd",
            operation_id=f"audit-{index:032x}",
            report_sha256=file_sha256(report_path),
            report_byte_count=report_path.stat().st_size,
            report=report,
            candidate=candidate_receipt_context,
            current_auditor_inventory=report["auditor"]["implementation_inventory"],
            started_at="2026-07-18T00:00:00+00:00",
            completed_at="2026-07-18T00:01:00+00:00",
        )
        _write_json(artifacts[receipt_name], receipt)
        action = build_audit_action_record(
            kind=kind,
            job_id="conditioned-0123456789abcdefabcd",
            report_sha256=file_sha256(report_path),
            report_byte_count=report_path.stat().st_size,
            report=report,
            receipt_sha256=file_sha256(artifacts[receipt_name]),
            receipt_byte_count=artifacts[receipt_name].stat().st_size,
            receipt=receipt,
            candidate=candidate_receipt_context,
            current_auditor_inventory=report["auditor"]["implementation_inventory"],
            committed_at="2026-07-18T00:02:00+00:00",
        )
        _write_json(artifacts[action_name], action)
        _write_json(
            root
            / "runs"
            / "v3"
            / "conditioned-dataset-v5"
            / "conditioned-0123456789abcdefabcd"
            / "audit_actions"
            / f"{kind}-{action['operation_id']}.json",
            action,
        )

    inventory_files = _inventory(publication)
    inventory_payload = _inventory_payload(inventory_files)
    publication_identity = stable_hash(inventory_payload)
    previous_publication = publication
    publication = root / "datasets" / f"conditioned-v5-{publication_identity}"
    publication.parent.mkdir(parents=True, exist_ok=True)
    previous_publication.rename(publication)
    artifacts = {name: publication / path.relative_to(previous_publication) for name, path in artifacts.items()}
    campaign_directory = root / "campaigns" / f"conditioned-v5-{publication_identity}"
    campaign_directory.parent.mkdir(parents=True, exist_ok=True)
    freeze = publication / "activation.json"
    _write_json(
        freeze,
        {
            "schema_version": CONDITIONED_DATASET_FREEZE_SCHEMA,
            "dataset_version": 5,
            "dataset_kind": "conditioned",
            "requires_semantic_labels": True,
            "status": "complete",
            "production_authorized": True,
            "immutable": True,
            "image_count": image_count,
            "dataset_identity": candidate_identity,
            "publication_identity_sha256": publication_identity,
            "labeling_audit_sha256": file_sha256(artifacts["labeling_audit"]),
            "validation_report_sha256": file_sha256(artifacts["validation_report"]),
            "artifacts": {
                name: {
                    "path": _relative(publication, path),
                    "sha256": file_sha256(path),
                    "byte_count": path.stat().st_size,
                }
                for name, path in artifacts.items()
            },
            "publication_inventory": {
                **inventory_payload,
                "inventory_sha256": stable_hash(inventory_payload),
            },
            "licenses": ["cc0-1.0"],
            "paths_are_relative": True,
            "paths_exposed": False,
        },
    )
    built = build_conditioned_three_seed_campaign(
        root,
        campaign_directory=_relative(root, campaign_directory),
        activation_manifest=_relative(root, freeze),
        activation_manifest_sha256=file_sha256(freeze),
        view_manifest=_relative(root, artifacts["view_manifest"]),
        split_manifest=_relative(root, artifacts["split_manifest"]),
        conditioning_vocabulary=_relative(root, artifacts["conditioning_vocabulary"]),
        benchmark_manifest=_relative(root, artifacts["benchmark_manifest"]),
        output_root=_relative(root, output_root),
        campaign_id="conditioned-v5-production",
    )
    portable = deepcopy(dict(built.portable_campaign))
    campaign_directory.mkdir(parents=True, exist_ok=False)
    campaign_path = campaign_directory / "campaigns.json"
    _write_json(
        campaign_path,
        {
            "product_profiles": {
                profile.value: {"campaign": portable}
                for profile in (TrainingProfile.RECOMMENDED, TrainingProfile.QUALITY, TrainingProfile.CUSTOM)
            }
        },
    )
    dataset_commit_inventory = _inventory(publication)
    campaign_commit_inventory = _inventory(campaign_directory)
    publication_journal = build_publication_journal(
        publication_identity=publication_identity,
        dataset_inventory=dataset_commit_inventory,
        campaign_inventory=campaign_commit_inventory,
    )
    dataset_commit = build_dataset_commit(
        journal=publication_journal,
        dataset_inventory=dataset_commit_inventory,
        campaign_inventory=campaign_commit_inventory,
    )
    campaign_commit = build_campaign_commit(
        journal=publication_journal,
        dataset_commit=dataset_commit,
        dataset_inventory=dataset_commit_inventory,
        campaign_inventory=campaign_commit_inventory,
    )
    job_root = root / "runs" / "v3" / "conditioned-dataset-v5" / "conditioned-0123456789abcdefabcd"
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / PUBLICATION_JOURNAL_NAME).write_bytes(canonical_publication_commit_bytes(publication_journal))
    (publication.parent / dataset_commit_name(publication_identity)).write_bytes(
        canonical_publication_commit_bytes(dataset_commit)
    )
    (campaign_directory.parent / campaign_commit_name(publication_identity)).write_bytes(
        canonical_publication_commit_bytes(campaign_commit)
    )
    values = deepcopy(DEFAULT_CONFIG)
    values["dataset"]["freeze_manifest"] = _relative(root, freeze)
    values["training"]["dataset_freeze"] = _relative(root, freeze)
    values["training"]["campaign_config"] = _relative(root, campaign_path)
    values["execution"]["allow_dataset_production_freeze"] = True
    values["execution"]["allow_training"] = True
    config_path = root / "spritelab.yaml"
    config_path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
    activation_receipt = (
        root / "runs" / "v3" / "conditioned-dataset-v5" / "conditioned-0123456789abcdefabcd" / "activation_receipt"
    )
    activation_receipt.mkdir(parents=True)
    manifest = json.loads(freeze.read_text(encoding="utf-8"))
    receipt, journal, record = build_activation_commit_documents(
        job_id="conditioned-0123456789abcdefabcd",
        operation_id="activation-0123456789abcdef0123456789abcdef",
        candidate_identity=candidate_identity,
        publication_identity_sha256=manifest["publication_identity_sha256"],
        activation_manifest_sha256=file_sha256(freeze),
        campaign_config_sha256=file_sha256(campaign_path),
        campaign_identity_sha256=built.campaign["campaign_identity"],
        authorization_id_sha256="a" * 64,
        config_before_sha256="b" * 64,
        config_after_sha256=file_sha256(config_path),
        prepared_at="2026-07-18T00:03:00+00:00",
    )
    for name, value in (("receipt.json", receipt), ("journal.json", journal), ("record.json", record)):
        (activation_receipt / name).write_bytes(canonical_activation_commit_bytes(value))
    marker = build_activation_project_commit(
        receipt=receipt,
        journal=journal,
        record=record,
        config_after_bytes=config_path.read_bytes(),
    )
    (root / ACTIVATION_PROJECT_COMMIT_NAME).write_bytes(canonical_activation_commit_bytes(marker))
    return ActivationProject(ProjectConfig.load(config_path), freeze, publication, campaign_path, portable)


def _write_training_audit(
    root: Path,
    gates: dict[str, Any],
) -> tuple[ProjectConfig, dict[str, Any], ConditionedTrainingActivation]:
    freeze_path = root / "artifacts" / "dataset" / "activation.json"
    campaign_path = root / "artifacts" / "training" / "campaign.json"
    _write_json(freeze_path, {"fixture": "freeze"})
    _write_json(campaign_path, {"fixture": "campaign"})
    values = deepcopy(DEFAULT_CONFIG)
    report_path = root / "artifacts" / "training" / "audit_report.json"
    hashes_path = root / "artifacts" / "training" / "audit_hashes.json"
    values["training"]["audit_report"] = _relative(root, report_path)
    values["training"]["audit_hashes"] = _relative(root, hashes_path)
    config = ProjectConfig(root, None, values)
    activation = ConditionedTrainingActivation(
        config=config,
        profile=TrainingProfile.RECOMMENDED,
        freeze_path=freeze_path,
        freeze_sha256=file_sha256(freeze_path),
        campaign_config_path=campaign_path,
        campaign_config_sha256=file_sha256(campaign_path),
        manifest={},
        artifacts={},
        selected_spec={},
        campaign={"campaign_identity": "c" * 64, "code_identity": {"sha256": "d" * 64}},
        audit_status=AuditStatus.NOT_AUDITED,
    )
    bindings = {
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign["campaign_identity"],
        "training_code_identity_sha256": activation.campaign["code_identity"]["sha256"],
    }
    report = {
        "schema_version": TRAINING_AUDIT_REPORT_SCHEMA,
        "bindings": bindings,
        "gates": gates,
    }
    _write_json(report_path, report)
    _write_json(
        hashes_path,
        {
            "schema_version": TRAINING_AUDIT_HASHES_SCHEMA,
            "audit_report_sha256": file_sha256(report_path),
            "bindings": bindings,
            "files": [],
        },
    )
    return config, report, activation


def _pin_fixture_auditor_inventories(
    project: ActivationProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {
        "label_audit": json.loads((project.publication / "evidence" / "label_audit.json").read_text(encoding="utf-8")),
        "dataset_validation": json.loads(
            (project.publication / "evidence" / "dataset_validation.json").read_text(encoding="utf-8")
        ),
    }
    inventories = {kind: report["auditor"]["implementation_inventory"] for kind, report in reports.items()}
    monkeypatch.setattr(activation_module, "trusted_auditor_inventory", lambda kind: deepcopy(inventories[kind]))


def _resign_activation_outer_bindings(
    project: ActivationProject,
    *,
    artifact_name: str,
    manifest_hash_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    evidence_binding = manifest["artifacts"][artifact_name]
    evidence_path = project.publication / evidence_binding["path"]
    evidence_digest = file_sha256(evidence_path)
    evidence_binding["sha256"] = evidence_digest
    evidence_binding["byte_count"] = evidence_path.stat().st_size
    manifest[manifest_hash_name] = evidence_digest
    inventory_files = _inventory(project.publication, exclude={"activation.json"})
    inventory_payload = _inventory_payload(inventory_files)
    inventory_identity = stable_hash(inventory_payload)
    manifest["publication_inventory"] = {
        **inventory_payload,
        "inventory_sha256": inventory_identity,
    }
    manifest["publication_identity_sha256"] = inventory_identity
    _write_json(project.freeze, manifest)

    campaign = json.loads(project.campaign_path.read_text(encoding="utf-8"))
    freeze_digest = file_sha256(project.freeze)
    for profile in campaign["product_profiles"].values():
        profile["campaign"]["identities"]["dataset_freeze_hash"] = freeze_digest
    _write_json(project.campaign_path, campaign)
    return manifest, campaign


def _fully_resign_embedded_evidence_attack(
    project: ActivationProject,
    *,
    artifact_name: str,
    manifest_hash_name: str,
    gate: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    evidence_binding = manifest["artifacts"][artifact_name]
    evidence_path = project.publication / evidence_binding["path"]
    report = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert report["checks"][gate] == "PASS"

    # Model a local author who changes the embedded result and recomputes every
    # unkeyed identity from the report through the selected campaign.
    report["checks"][gate] = "pass"
    run_payload = {key: value for key, value in report.items() if key != "audit_run_identity"}
    report["audit_run_identity"] = stable_hash(run_payload)
    _write_json(evidence_path, report)
    manifest, campaign = _resign_activation_outer_bindings(
        project,
        artifact_name=artifact_name,
        manifest_hash_name=manifest_hash_name,
    )
    return report, manifest, campaign


def test_builder_and_all_selected_profiles_bind_the_exact_activation_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _activation_project(tmp_path)
    _pin_fixture_auditor_inventories(project, monkeypatch)
    code_identity = campaign_module._code_identity()
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: deepcopy(code_identity))
    recommended = load_conditioned_training_activation(
        project.config,
        TrainingProfile.RECOMMENDED,
        require_audit=False,
    )
    quality = load_conditioned_training_activation(
        project.config,
        TrainingProfile.QUALITY,
        require_audit=False,
    )
    custom = load_conditioned_training_activation(
        project.config,
        TrainingProfile.CUSTOM,
        custom_spec=project.portable_campaign,
        require_audit=False,
    )

    assert tuple(recommended.campaign["seeds"]) == DEFAULT_SEEDS
    assert recommended.campaign["training"]["max_optimizer_steps"] == 5_000
    assert recommended.campaign["training"]["positive_sampling_mass_records"] == 2_417.0
    assert recommended.campaign["executable"] is True
    assert recommended.campaign["launch_authorized"] is True
    assert {
        recommended.campaign["campaign_identity"],
        quality.campaign["campaign_identity"],
        custom.campaign["campaign_identity"],
    } == {recommended.campaign["campaign_identity"]}
    assert {recommended.freeze_sha256, quality.freeze_sha256, custom.freeze_sha256} == {file_sha256(project.freeze)}
    contract = recommended.to_contract_dict()
    assert contract["schema_version"] == recommended.schema_version
    assert contract["ready"] is recommended.ready
    assert contract["profile"] == TrainingProfile.RECOMMENDED.value
    assert contract["image_count"] == recommended.manifest["image_count"]
    assert contract["campaign_identity_sha256"] == recommended.campaign["campaign_identity"]
    assert contract["paths_exposed"] is False

    with pytest.raises(ConditionedActivationError, match="exactly match"):
        load_conditioned_training_activation(
            project.config,
            TrainingProfile.CUSTOM,
            custom_spec={**project.portable_campaign, "campaign_id": "changed"},
            require_audit=False,
        )


def test_activation_rejects_foreign_publication_pair_copied_under_selected_marker_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _activation_project(tmp_path)
    _pin_fixture_auditor_inventories(project, monkeypatch)
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    selected_identity = manifest["publication_identity_sha256"]
    foreign_identity = "b" * 64 if selected_identity != "b" * 64 else "c" * 64
    dataset_inventory = _inventory(project.publication)
    campaign_inventory = _inventory(project.campaign_path.parent)
    foreign_journal = build_publication_journal(
        publication_identity=foreign_identity,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    foreign_dataset_marker = build_dataset_commit(
        journal=foreign_journal,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    foreign_campaign_marker = build_campaign_commit(
        journal=foreign_journal,
        dataset_commit=foreign_dataset_marker,
        dataset_inventory=dataset_inventory,
        campaign_inventory=campaign_inventory,
    )
    job_root = project.config.root / "runs" / "v3" / "conditioned-dataset-v5" / "conditioned-0123456789abcdefabcd"
    (job_root / PUBLICATION_JOURNAL_NAME).write_bytes(canonical_publication_commit_bytes(foreign_journal))
    selected_dataset_marker = project.publication.parent / dataset_commit_name(selected_identity)
    selected_campaign_marker = project.campaign_path.parent.parent / campaign_commit_name(selected_identity)
    selected_dataset_marker.write_bytes(canonical_publication_commit_bytes(foreign_dataset_marker))
    selected_campaign_marker.write_bytes(canonical_publication_commit_bytes(foreign_campaign_marker))

    assert foreign_journal["publication_identity_sha256"] == foreign_identity
    assert selected_dataset_marker.name == dataset_commit_name(selected_identity)
    assert foreign_dataset_marker["commit_relative_path"].endswith(dataset_commit_name(foreign_identity))
    assert selected_campaign_marker.name == campaign_commit_name(selected_identity)
    assert foreign_campaign_marker["commit_relative_path"].endswith(campaign_commit_name(foreign_identity))

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(project.config, require_audit=False)

    assert captured.value.code == "publication_commit_invalid"
    assert captured.value.public_message == "The conditioned publication pair marker is invalid or stale."


@pytest.mark.parametrize(
    ("artifact_name", "manifest_hash_name", "gate"),
    (
        ("labeling_audit", "labeling_audit_sha256", "semantic_coverage"),
        ("validation_report", "validation_report_sha256", "count_range"),
    ),
)
def test_activation_rejects_fully_resigned_nonliteral_embedded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    manifest_hash_name: str,
    gate: str,
) -> None:
    project = _activation_project(tmp_path)
    _pin_fixture_auditor_inventories(project, monkeypatch)

    report, manifest, campaign = _fully_resign_embedded_evidence_attack(
        project,
        artifact_name=artifact_name,
        manifest_hash_name=manifest_hash_name,
        gate=gate,
    )

    evidence_binding = manifest["artifacts"][artifact_name]
    evidence_path = project.publication / evidence_binding["path"]
    inventory = manifest["publication_inventory"]
    inventory_payload = {key: inventory[key] for key in ("schema_version", "files", "file_count", "total_bytes")}
    freeze_digest = file_sha256(project.freeze)
    assert report["audit_run_identity"] == stable_hash(
        {key: value for key, value in report.items() if key != "audit_run_identity"}
    )
    assert evidence_binding == {
        "path": evidence_binding["path"],
        "sha256": file_sha256(evidence_path),
        "byte_count": evidence_path.stat().st_size,
    }
    assert manifest[manifest_hash_name] == file_sha256(evidence_path)
    assert inventory["files"][evidence_binding["path"]]["sha256"] == file_sha256(evidence_path)
    assert inventory["inventory_sha256"] == stable_hash(inventory_payload)
    assert manifest["publication_identity_sha256"] == inventory["inventory_sha256"]
    assert {
        profile["campaign"]["identities"]["dataset_freeze_hash"] for profile in campaign["product_profiles"].values()
    } == {freeze_digest}

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(project.config, require_audit=False)

    assert captured.value.code == "conditioned_evidence_checks"


@pytest.mark.parametrize(
    ("artifact_name", "manifest_hash_name"),
    (
        ("labeling_audit", "labeling_audit_sha256"),
        ("validation_report", "validation_report_sha256"),
    ),
)
def test_activation_rejects_resigned_report_bytes_without_a_matching_server_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    manifest_hash_name: str,
) -> None:
    project = _activation_project(tmp_path)
    _pin_fixture_auditor_inventories(project, monkeypatch)
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    evidence_path = project.publication / manifest["artifacts"][artifact_name]["path"]
    original_digest = file_sha256(evidence_path)
    report = json.loads(evidence_path.read_text(encoding="utf-8"))

    # The report means exactly the same thing and retains its valid unkeyed run
    # identity, but its exact bytes were not produced by the receipted server run.
    evidence_path.write_text(
        json.dumps(report, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    assert file_sha256(evidence_path) != original_digest
    rebound_manifest, campaign = _resign_activation_outer_bindings(
        project,
        artifact_name=artifact_name,
        manifest_hash_name=manifest_hash_name,
    )
    freeze_digest = file_sha256(project.freeze)
    assert rebound_manifest[manifest_hash_name] == file_sha256(evidence_path)
    assert {
        profile["campaign"]["identities"]["dataset_freeze_hash"] for profile in campaign["product_profiles"].values()
    } == {freeze_digest}

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(project.config, require_audit=False)

    assert captured.value.code == "conditioned_evidence_receipt"


def test_fast_preview_is_ineligible_for_conditioned_production_activation(tmp_path: Path) -> None:
    config = ProjectConfig(tmp_path, None, deepcopy(DEFAULT_CONFIG))

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(
            config,
            TrainingProfile.FAST_PREVIEW,
            require_audit=False,
        )

    assert captured.value.code == "conditioned_profile_ineligible"
    assert "fast_preview" in captured.value.public_message


def test_activation_rejects_fully_rehashed_cross_job_action_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _activation_project(tmp_path)
    _pin_fixture_auditor_inventories(project, monkeypatch)
    report_path = project.publication / "evidence" / "label_audit.json"
    receipt_path = project.publication / "evidence" / "label_audit_receipt.json"
    action_path = project.publication / "evidence" / "label_audit_action.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    candidate = {
        "candidate_identity": manifest["dataset_identity"],
        "payload_inventory_sha256": report["bindings"]["payload_inventory_sha256"],
        "image_count": manifest["image_count"],
    }
    forged_job = "conditioned-fedcba9876543210abcd"
    forged_receipt = build_audit_receipt(
        kind="label_audit",
        job_id=forged_job,
        operation_id="audit-99999999999999999999999999999999",
        report_sha256=file_sha256(report_path),
        report_byte_count=report_path.stat().st_size,
        report=report,
        candidate=candidate,
        current_auditor_inventory=report["auditor"]["implementation_inventory"],
        started_at="2026-07-18T01:00:00+00:00",
        completed_at="2026-07-18T01:01:00+00:00",
    )
    _write_json(receipt_path, forged_receipt)
    forged_action = build_audit_action_record(
        kind="label_audit",
        job_id=forged_job,
        report_sha256=file_sha256(report_path),
        report_byte_count=report_path.stat().st_size,
        report=report,
        receipt_sha256=file_sha256(receipt_path),
        receipt_byte_count=receipt_path.stat().st_size,
        receipt=forged_receipt,
        candidate=candidate,
        current_auditor_inventory=report["auditor"]["implementation_inventory"],
        committed_at="2026-07-18T01:02:00+00:00",
    )
    _write_json(action_path, forged_action)
    for artifact_name, path in (
        ("labeling_audit_receipt", receipt_path),
        ("labeling_audit_action_record", action_path),
    ):
        manifest["artifacts"][artifact_name].update({"sha256": file_sha256(path), "byte_count": path.stat().st_size})
    inventory_files = _inventory(project.publication, exclude={"activation.json"})
    inventory_payload = _inventory_payload(inventory_files)
    manifest["publication_inventory"] = {
        **inventory_payload,
        "inventory_sha256": stable_hash(inventory_payload),
    }
    manifest["publication_identity_sha256"] = stable_hash(inventory_payload)
    _write_json(project.freeze, manifest)
    campaign = json.loads(project.campaign_path.read_text(encoding="utf-8"))
    for entry in campaign["product_profiles"].values():
        entry["campaign"]["identities"]["dataset_freeze_hash"] = file_sha256(project.freeze)
    _write_json(project.campaign_path, campaign)

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(project.config, require_audit=False)
    assert captured.value.code == "conditioned_evidence_action"


@pytest.mark.parametrize("extension", (".json", ".yaml"))
def test_activation_rejects_nested_duplicate_inventory_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
) -> None:
    _stabilize_campaign_fixture(monkeypatch)
    project = _activation_project(tmp_path)
    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    expected_identity = manifest["publication_inventory"]["inventory_sha256"]
    hostile_path = project.freeze.with_suffix(extension)
    permissive = _write_mapping_with_shadowed_key(
        hostile_path,
        manifest,
        key="inventory_sha256",
        shadow_value="0" * 64,
    )
    assert permissive["publication_inventory"]["inventory_sha256"] == expected_identity

    values = deepcopy(project.config.values)
    configured_path = _relative(project.config.root, hostile_path)
    values["dataset"]["freeze_manifest"] = configured_path
    values["training"]["dataset_freeze"] = configured_path
    plan_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(activation_module, "plan_campaign", lambda spec, **_kwargs: plan_calls.append(dict(spec)))

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(
            ProjectConfig(project.config.root, None, values),
            require_audit=False,
        )

    _assert_sanitized_mapping_error(captured.value)
    assert plan_calls == []


def test_campaign_selection_rejects_duplicate_authorization_and_binding_in_json_and_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stabilize_campaign_fixture(monkeypatch)
    project = _activation_project(tmp_path)
    plan_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(activation_module, "plan_campaign", lambda spec, **_kwargs: plan_calls.append(dict(spec)))
    attacks = (
        ("authorization", "launch_authorized", False, True),
        (
            "binding",
            "dataset_freeze_hash",
            "0" * 64,
            project.portable_campaign["identities"]["dataset_freeze_hash"],
        ),
    )

    for extension in (".json", ".yaml"):
        for attack_name, key, shadow_value, expected_value in attacks:
            document = {
                "product_profiles": {
                    TrainingProfile.RECOMMENDED.value: {"campaign": deepcopy(project.portable_campaign)}
                }
            }
            hostile_path = project.campaign_path.with_name(f"campaign-{attack_name}{extension}")
            permissive = _write_mapping_with_shadowed_key(
                hostile_path,
                document,
                key=key,
                shadow_value=shadow_value,
            )
            selected = permissive["product_profiles"][TrainingProfile.RECOMMENDED.value]["campaign"]
            observed = selected[key] if attack_name == "authorization" else selected["identities"][key]
            assert observed == expected_value
            values = deepcopy(project.config.values)
            values["training"]["campaign_config"] = _relative(project.config.root, hostile_path)

            with pytest.raises(ConditionedActivationError) as captured:
                load_conditioned_training_activation(
                    ProjectConfig(project.config.root, None, values),
                    require_audit=False,
                )

            _assert_sanitized_mapping_error(captured.value)
            assert plan_calls == []


def test_activation_rejects_absolute_configuration_and_inventory_byte_drift(tmp_path: Path) -> None:
    project = _activation_project(tmp_path)
    absolute_values = deepcopy(project.config.values)
    absolute_values["dataset"]["freeze_manifest"] = str(project.freeze)
    absolute_values["training"]["dataset_freeze"] = str(project.freeze)
    with pytest.raises(ConditionedActivationError, match="Absolute"):
        load_conditioned_training_activation(
            ProjectConfig(project.config.root, None, absolute_values),
            require_audit=False,
        )

    manifest = json.loads(project.freeze.read_text(encoding="utf-8"))
    record = manifest["publication_inventory"]["files"]["view_manifest.json"]
    record["byte_count"] += 1
    payload = {
        key: manifest["publication_inventory"][key] for key in ("schema_version", "files", "file_count", "total_bytes")
    }
    manifest["publication_inventory"]["inventory_sha256"] = stable_hash(payload)
    _write_json(project.freeze, manifest)
    with pytest.raises(ConditionedActivationError, match="byte count"):
        load_conditioned_training_activation(project.config, require_audit=False)


def test_activation_inventory_rejects_linked_directory_without_touching_outside(tmp_path: Path) -> None:
    project = _activation_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"preserve")
    try:
        os.symlink(outside, project.publication / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this test session")

    with pytest.raises(ConditionedActivationError, match="linked directory"):
        load_conditioned_training_activation(project.config, require_audit=False)
    assert sentinel.read_bytes() == b"preserve"


def test_pass_audit_becomes_stale_when_untracked_python_appears_under_bound_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _activation_project(tmp_path)
    code_root = project.config.root / "src" / "spritelab" / "product_features" / "training"
    code_root.mkdir(parents=True)
    tracked = code_root / "tracked.py"
    tracked.write_text("BOUND = True\n", encoding="utf-8")
    mandatory_outside_root = project.config.root / "src" / "spritelab" / "product_runtime.py"
    mandatory_outside_root.parent.mkdir(parents=True, exist_ok=True)
    mandatory_outside_root.write_text("BOUND_OUTSIDE_ROOT = True\n", encoding="utf-8")
    relative = _relative(project.config.root, tracked)
    monkeypatch.setattr(
        activation_module,
        "TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS",
        (str(code_root.relative_to(project.config.root)).replace("\\", "/"),),
    )
    monkeypatch.setattr(
        activation_module,
        "training_code_identity_source_paths",
        lambda _root: (tracked, mandatory_outside_root),
    )

    activation = load_conditioned_training_activation(project.config, require_audit=False)
    bindings = {
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign["campaign_identity"],
        "training_code_identity_sha256": activation.campaign["code_identity"]["sha256"],
    }
    report_path = project.config.root / "artifacts" / "training" / "audit_report.json"
    hashes_path = project.config.root / "artifacts" / "training" / "audit_hashes.json"
    report = {
        "schema_version": TRAINING_AUDIT_REPORT_SCHEMA,
        "bindings": bindings,
        "gates": dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS"),
    }
    _write_json(report_path, report)
    _write_json(
        hashes_path,
        {
            "schema_version": TRAINING_AUDIT_HASHES_SCHEMA,
            "audit_report_sha256": file_sha256(report_path),
            "bindings": bindings,
            "files": [
                {"path": relative, "sha256_before": file_sha256(tracked)},
                {
                    "path": _relative(project.config.root, mandatory_outside_root),
                    "sha256_before": file_sha256(mandatory_outside_root),
                },
            ],
        },
    )
    project.config.values["training"]["audit_report"] = _relative(project.config.root, report_path)
    project.config.values["training"]["audit_hashes"] = _relative(project.config.root, hashes_path)

    assert (
        activation_module._verify_audited_code_inventory(
            project.config.root,
            json.loads(hashes_path.read_text(encoding="utf-8"))["files"],
        )
        is True
    )
    # A caller-authored report and fully rehashed inventory are never trusted
    # without the server-managed execution receipt.
    assert training_audit_status(project.config, report, activation) is AuditStatus.STALE
    from spritelab.v3.status import _training_audit_status

    assert _training_audit_status(project.config, report) is AuditStatus.STALE
    (code_root / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    assert (
        activation_module._verify_audited_code_inventory(
            project.config.root,
            json.loads(hashes_path.read_text(encoding="utf-8"))["files"],
        )
        is False
    )
    assert training_audit_status(project.config, report, activation) is AuditStatus.STALE
    assert _training_audit_status(project.config, report) is AuditStatus.STALE


def test_training_audit_requires_literal_uppercase_pass_verdicts() -> None:
    for gate_value, expected in (
        ("PASS", AuditStatus.PASS),
        ("pass", AuditStatus.INCONCLUSIVE),
        ("Pass", AuditStatus.INCONCLUSIVE),
        ("fail", AuditStatus.INCONCLUSIVE),
        (True, AuditStatus.INCONCLUSIVE),
        (1, AuditStatus.INCONCLUSIVE),
        (None, AuditStatus.INCONCLUSIVE),
    ):
        gates = dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")
        gates["api_ui_privacy"] = gate_value
        assert training_audit_module._overall(gates) is expected, gate_value


def test_literal_fail_dominates_other_inconclusive_audit_verdicts() -> None:
    gates = dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")
    gates["api_ui_privacy"] = "pass"
    gates["backend_command_safety"] = "FAIL"

    assert training_audit_module._overall(gates) is AuditStatus.FAIL


def test_training_audit_rejects_duplicate_gates_in_json_and_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(activation_module, "_verify_audited_code_inventory", lambda *_args: True)

    for extension in (".json", ".yaml"):
        root = tmp_path / extension.removeprefix(".")
        gates = dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")
        config, report, activation = _write_training_audit(root, gates)
        hostile_path = root / "artifacts" / "training" / f"audit_report{extension}"
        permissive = _write_mapping_with_shadowed_key(
            hostile_path,
            report,
            key="api_ui_privacy",
            shadow_value="FAIL",
        )
        assert permissive["gates"]["api_ui_privacy"] == "PASS"
        config.values["training"]["audit_report"] = _relative(root, hostile_path)
        hashes_path = root / config.values["training"]["audit_hashes"]
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        hashes["audit_report_sha256"] = file_sha256(hostile_path)
        _write_json(hashes_path, hashes)

        with pytest.raises(ConditionedActivationError) as captured:
            activation_module._read_mapping(hostile_path)
        _assert_sanitized_mapping_error(captured.value)
        assert training_audit_status(config, permissive, activation) is AuditStatus.STALE


def test_training_audit_hashes_reject_duplicate_bindings_in_json_and_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(activation_module, "_verify_audited_code_inventory", lambda *_args: True)

    for extension in (".json", ".yaml"):
        root = tmp_path / extension.removeprefix(".")
        gates = dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")
        config, report, activation = _write_training_audit(root, gates)
        original_hashes_path = root / config.values["training"]["audit_hashes"]
        hashes = json.loads(original_hashes_path.read_text(encoding="utf-8"))
        expected_identity = hashes["bindings"]["campaign_identity_sha256"]
        hostile_path = original_hashes_path.with_suffix(extension)
        permissive = _write_mapping_with_shadowed_key(
            hostile_path,
            hashes,
            key="campaign_identity_sha256",
            shadow_value="0" * 64,
        )
        assert permissive["bindings"]["campaign_identity_sha256"] == expected_identity
        config.values["training"]["audit_hashes"] = _relative(root, hostile_path)

        with pytest.raises(ConditionedActivationError) as captured:
            activation_module._read_mapping(hostile_path)
        _assert_sanitized_mapping_error(captured.value)
        assert training_audit_status(config, report, activation) is AuditStatus.STALE


def test_activation_publication_inventory_refuses_lstat_link_open_restore_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    publication = project / "datasets" / "conditioned-v5-test"
    publication.mkdir(parents=True)
    activation = publication / "activation.json"
    activation.write_bytes(b"{}\n")
    victim = publication / "view_manifest.json"
    inside_content = b'{"inside":true}\n'
    victim.write_bytes(inside_content)
    outside = tmp_path / "outside-sentinel.json"
    outside_content = b'{"outside":"preserve"}\n'
    outside.write_bytes(outside_content)
    inventory = activation_module._publication_inventory(
        publication,
        exclude=activation,
        boundary=project,
    )
    assert inventory == {
        victim.name: {
            "sha256": hashlib.sha256(inside_content).hexdigest(),
            "byte_size": len(inside_content),
        }
    }

    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(outside, probe)
        probe.rename(tmp_path / "symlink-probe-residue")
    except OSError:
        pytest.skip("file symlinks are unavailable in this test session")

    real_path_open = Path.open
    real_anchored_open = activation_module.AnchoredDirectory.open_file
    raced = False

    def race(open_file: Any) -> Any:
        nonlocal raced
        if raced:
            return open_file()
        raced = True
        parked = publication / "view-manifest-parked.json"
        residue = publication / "view-manifest-link-residue.json"
        victim.rename(parked)
        os.symlink(outside, victim)
        try:
            return open_file()
        finally:
            victim.rename(residue)
            parked.rename(victim)

    def raced_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == victim:
            return race(lambda: real_path_open(path, *args, **kwargs))
        return real_path_open(path, *args, **kwargs)

    def raced_anchored_open(anchor: Any, name: str, flags: int, mode: int = 0o600) -> int:
        if anchor.directory == publication and name == victim.name:
            return race(lambda: real_anchored_open(anchor, name, flags, mode))
        return real_anchored_open(anchor, name, flags, mode)

    monkeypatch.setattr(Path, "open", raced_path_open)
    monkeypatch.setattr(activation_module.AnchoredDirectory, "open_file", raced_anchored_open)

    with pytest.raises(ConditionedActivationError) as captured:
        activation_module._publication_inventory(
            publication,
            exclude=activation,
            boundary=project,
        )

    assert captured.value.code == "publication_entry_changed"
    assert raced is True
    assert outside.read_bytes() == outside_content
    assert victim.read_bytes() == inside_content


@pytest.mark.parametrize("target_kind", ("freeze", "campaign"))
def test_activation_mapping_snapshot_refuses_same_parent_open_restore_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    project = _activation_project(tmp_path / "project")
    _pin_fixture_auditor_inventories(project, monkeypatch)
    target = project.freeze if target_kind == "freeze" else project.campaign_path
    original = target.read_bytes()
    alternate = target.with_name(f".{target.name}.alternate")
    alternate.write_bytes(original + b" \n")
    outside = tmp_path / f"outside-{target_kind}.sentinel"
    outside_content = b"preserve-outside"
    outside.write_bytes(outside_content)
    real_open = activation_module.AnchoredDirectory.open_file
    real_immovable = activation_module.AnchoredDirectory.open_file_immovable
    raced = False

    def raced_open(
        anchor: Any,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal raced
        if raced or anchor.directory != target.parent or name != target.name:
            return real_immovable(anchor, name, flags, mode)
        raced = True
        parked = target.with_name(f".{target.name}.parked")
        residue = target.with_name(f".{target.name}.alternate-residue")
        target.rename(parked)
        alternate.rename(target)
        try:
            descriptor = real_open(anchor, name, flags, mode)
        finally:
            target.rename(residue)
            parked.rename(target)
        return descriptor

    monkeypatch.setattr(activation_module.AnchoredDirectory, "open_file_immovable", raced_open)

    with pytest.raises(ConditionedActivationError) as captured:
        load_conditioned_training_activation(
            project.config,
            require_audit=False,
            require_activation_commit=False,
        )

    assert captured.value.code == "activation_snapshot_changed"
    assert raced is True
    assert target.read_bytes() == original
    assert outside.read_bytes() == outside_content


def test_activation_publication_inventory_refuses_early_in_place_mutation_during_later_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    publication = project / "datasets" / "conditioned-v5-test"
    publication.mkdir(parents=True)
    activation = publication / "activation.json"
    activation.write_bytes(b"{}\n")
    early = publication / "a-early.json"
    early.write_bytes(b"AAAA")
    later = publication / "z-later.json"
    later.write_bytes(b"ZZZZ")
    outside = tmp_path / "outside-inventory.sentinel"
    outside_content = b"preserve-outside"
    outside.write_bytes(outside_content)
    real_identity = activation_module._publication_file_identity
    mutated = False

    def mutate_early_after_later(
        anchor: Any,
        name: str,
        root_device: int,
    ) -> dict[str, Any]:
        nonlocal mutated
        identity = real_identity(anchor, name, root_device)
        if name == later.name and not mutated:
            early.write_bytes(b"BBBB")
            mutated = True
        return identity

    monkeypatch.setattr(activation_module, "_publication_file_identity", mutate_early_after_later)

    with pytest.raises(ConditionedActivationError) as captured:
        activation_module._publication_inventory(
            publication,
            exclude=activation,
            boundary=project,
        )

    assert captured.value.code == "publication_entry_changed"
    assert mutated is True
    assert early.read_bytes() == b"BBBB"
    assert outside.read_bytes() == outside_content


def test_activation_publication_readers_accept_only_one_exact_retained_stage(tmp_path: Path) -> None:
    root = tmp_path / "project"
    publication = root / "datasets" / "published"
    publication.mkdir(parents=True)
    target = publication / "artifact.json"
    payload = b'{"ready":true}\n'
    target.write_bytes(payload)
    alias = publication / f".{target.name}.staging-{'a' * 32}"
    os.link(target, alias)

    assert (
        activation_module._project_relative_file(
            root,
            target.relative_to(root).as_posix(),
            allow_retained_stage=True,
        )
        == target
    )
    assert activation_module._publication_file(publication, target.name) == target
    assert activation_module._campaign_relative_file(publication, target.name, root) == target
    with activation_module._held_mapping_snapshot(target, root, allow_retained_stage=True) as snapshot:
        assert snapshot.value == {"ready": True}
        assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert activation_module._read_bound_evidence_mapping(
        target,
        {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)},
        boundary=root,
    ) == {"ready": True}
    assert activation_module._publication_inventory(publication, exclude=None, boundary=root) == {
        target.name: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_size": len(payload)}
    }
    with pytest.raises(ConditionedActivationError):
        activation_module._project_relative_file(root, target.relative_to(root).as_posix())


@pytest.mark.parametrize("attack", ("malformed", "wrong_inode", "extra"))
def test_activation_publication_readers_reject_hostile_retained_stage_topologies(
    tmp_path: Path,
    attack: str,
) -> None:
    root = tmp_path / attack
    publication = root / "publication"
    publication.mkdir(parents=True)
    target = publication / "artifact.json"
    target.write_bytes(b'{"ready":true}\n')
    prefix = f".{target.name}.staging-"
    if attack == "malformed":
        os.link(target, publication / f"{prefix}{'A' * 32}")
    elif attack == "wrong_inode":
        os.link(target, publication / "unreserved-hard-link")
        (publication / f"{prefix}{'a' * 32}").write_bytes(b"wrong inode")
    else:
        os.link(target, publication / f"{prefix}{'a' * 32}")
        (publication / f"{prefix}{'b' * 32}").write_bytes(b"extra wrong alias")

    with pytest.raises(ConditionedActivationError):
        with activation_module._held_mapping_snapshot(target, root, allow_retained_stage=True):
            pass
    with pytest.raises(ConditionedActivationError):
        activation_module._publication_inventory(publication, exclude=None, boundary=root)

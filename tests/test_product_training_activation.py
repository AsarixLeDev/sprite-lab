from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from spritelab.product_features.training import activation as activation_module
from spritelab.product_features.training.activation import (
    CONDITIONED_DATASET_FREEZE_SCHEMA,
    CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
    MANDATORY_TRAINING_AUDIT_GATES,
    TRAINING_AUDIT_HASHES_SCHEMA,
    TRAINING_AUDIT_REPORT_SCHEMA,
    ConditionedActivationError,
    build_conditioned_three_seed_campaign,
    load_conditioned_training_activation,
    training_audit_status,
)
from spritelab.product_features.training.models import TrainingProfile
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, stable_hash
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus


@dataclass(frozen=True)
class ActivationProject:
    config: ProjectConfig
    freeze: Path
    publication: Path
    campaign_path: Path
    portable_campaign: dict[str, Any]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _activation_project(tmp_path: Path) -> ActivationProject:
    root = tmp_path / "project"
    publication = root / "artifacts" / "dataset" / "conditioned-v5"
    campaign_directory = root / "artifacts" / "training"
    output_root = root / "runs" / "training"
    publication.mkdir(parents=True)
    campaign_directory.mkdir(parents=True)
    artifacts: dict[str, Path] = {}
    for name in (
        "view_manifest",
        "split_manifest",
        "conditioning_vocabulary",
        "benchmark_manifest",
        "labeling_audit",
        "validation_report",
    ):
        path = publication / f"{name}.json"
        _write_json(path, {"artifact": name, "status": "PASS"})
        artifacts[name] = path

    inventory_files = {
        _relative(publication, path): {
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for path in artifacts.values()
    }
    inventory_payload = {
        "schema_version": CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
        "files": dict(sorted(inventory_files.items())),
        "file_count": len(inventory_files),
        "total_bytes": sum(item["byte_count"] for item in inventory_files.values()),
    }
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
            "image_count": 2_417,
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
    values = deepcopy(DEFAULT_CONFIG)
    values["dataset"]["freeze_manifest"] = _relative(root, freeze)
    values["training"]["dataset_freeze"] = _relative(root, freeze)
    values["training"]["campaign_config"] = _relative(root, campaign_path)
    values["execution"]["allow_dataset_production_freeze"] = True
    values["execution"]["allow_training"] = True
    return ActivationProject(ProjectConfig(root, None, values), freeze, publication, campaign_path, portable)


def test_builder_and_all_selected_profiles_bind_the_exact_activation_campaign(tmp_path: Path) -> None:
    project = _activation_project(tmp_path)
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

    with pytest.raises(ConditionedActivationError, match="exactly match"):
        load_conditioned_training_activation(
            project.config,
            TrainingProfile.CUSTOM,
            custom_spec={**project.portable_campaign, "campaign_id": "changed"},
            require_audit=False,
        )


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

    assert training_audit_status(project.config, report, activation) is AuditStatus.PASS
    from spritelab.v3.status import _training_audit_status

    assert _training_audit_status(project.config, report) is AuditStatus.PASS
    (code_root / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    assert training_audit_status(project.config, report, activation) is AuditStatus.STALE
    assert _training_audit_status(project.config, report) is AuditStatus.STALE

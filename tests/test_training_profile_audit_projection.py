from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import spritelab.product_features.training.activation as activation_module
import spritelab.training.campaign as campaign_module
import spritelab.v3.status as status_module
from spritelab.product_core.audit_evidence import (
    ApplicabilityStatus,
    ArtifactVerificationStatus,
    AuditVerification,
)
from spritelab.product_features.training.activation import (
    CONDITIONED_DATASET_FREEZE_SCHEMA,
    ConditionedCampaignBuild,
    build_conditioned_three_seed_campaign,
)
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.plans import (
    TrainingProfileError,
    _resolve_selected_campaign_paths,
    _select_campaign_spec_with_origin,
    select_campaign_spec,
)
from spritelab.training.campaign import file_sha256, plan_campaign, stable_hash
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, StageStatus


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _conditioned_freeze_config(
    root: Path,
    *,
    schema_version: str,
    allow_training: bool,
) -> ProjectConfig:
    values = deepcopy(DEFAULT_CONFIG)
    freeze_relative = "artifacts/dataset/conditioned-v5/activation.json"
    values["dataset"]["freeze_manifest"] = freeze_relative
    values["training"]["dataset_freeze"] = freeze_relative
    values["execution"]["allow_training"] = allow_training
    _write_json(
        root / freeze_relative,
        {
            "schema_version": schema_version,
            "dataset_version": 5,
            "dataset_kind": "conditioned",
            "requires_semantic_labels": True,
            "status": "complete",
            "production_authorized": True,
            "image_count": 2_400,
        },
    )
    return ProjectConfig(root, None, values)


def _labeling_verification_without_freeze_scope() -> AuditVerification:
    return AuditVerification(
        subsystem="labeling",
        applicability_status=ApplicabilityStatus.NOT_COMPARABLE,
        reasons=("independent_labeling_audit_not_configured",),
        artifact_status=ArtifactVerificationStatus.MISSING,
        current_identity=None,
    )


def _portable_conditioned_campaign(
    root: Path,
    campaign_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    campaign_id: str,
) -> tuple[ConditionedCampaignBuild, Path]:
    publication = root / "artifacts" / "dataset" / "conditioned-v5"
    campaign_directory.mkdir(parents=True)
    activation = publication / "activation.json"
    view = publication / "view.json"
    split = publication / "split.json"
    vocabulary = publication / "vocabulary.json"
    benchmark = publication / "benchmark.json"
    _write_json(
        activation,
        {
            "schema_version": CONDITIONED_DATASET_FREEZE_SCHEMA,
            "dataset_version": 5,
            "dataset_kind": "conditioned",
            "image_count": 2_400,
        },
    )
    for path in (view, split, vocabulary, benchmark):
        _write_json(path, {"name": path.stem})

    code_identity = {
        "schema_version": campaign_module.CODE_IDENTITY_SCHEMA_VERSION,
        "contract": "stable-test-identity",
        "files": [],
    }
    code_identity["sha256"] = stable_hash(code_identity)
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: deepcopy(code_identity))

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return (
        build_conditioned_three_seed_campaign(
            root,
            campaign_directory=relative(campaign_directory),
            activation_manifest=relative(activation),
            activation_manifest_sha256=file_sha256(activation),
            view_manifest=relative(view),
            split_manifest=relative(split),
            conditioning_vocabulary=relative(vocabulary),
            benchmark_manifest=relative(benchmark),
            output_root="runs/training",
            campaign_id=campaign_id,
        ),
        activation,
    )


@pytest.mark.parametrize("profile", [TrainingProfile.QUALITY, TrainingProfile.CUSTOM])
def test_training_audit_status_loads_the_exact_selected_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: TrainingProfile,
) -> None:
    config = ProjectConfig(tmp_path, None, deepcopy(DEFAULT_CONFIG))
    activation = object()
    selected: list[TrainingProfile] = []

    def load_selected(_config, selected_profile, *, require_audit):
        assert require_audit is False
        selected.append(selected_profile)
        return activation

    monkeypatch.setattr(activation_module, "load_conditioned_training_activation", load_selected)
    monkeypatch.setattr(
        activation_module,
        "training_audit_status",
        lambda observed_config, report, observed_activation: (
            AuditStatus.PASS
            if observed_config is config and report == {} and observed_activation is activation
            else AuditStatus.STALE
        ),
    )

    assert status_module._training_audit_status(config, {}, profile) is AuditStatus.PASS
    assert selected == [profile]


@pytest.mark.parametrize(
    ("audit", "expected_stage", "has_blocker"),
    [
        (AuditStatus.PASS, StageStatus.COMPLETE, False),
        (AuditStatus.STALE, StageStatus.STALE, True),
        (AuditStatus.FAIL, StageStatus.FAILED, True),
        (AuditStatus.NOT_AUDITED, StageStatus.INCONCLUSIVE, True),
    ],
)
def test_training_audit_stage_projects_authoritative_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit: AuditStatus,
    expected_stage: StageStatus,
    has_blocker: bool,
) -> None:
    config = ProjectConfig(tmp_path, None, deepcopy(DEFAULT_CONFIG))
    observed_profiles: list[TrainingProfile | str] = []

    def selected_audit(_config, _report, profile, activation):
        assert activation is status_module._ACTIVATION_NOT_LOADED
        observed_profiles.append(profile)
        return audit

    monkeypatch.setattr(status_module, "_source_commit", lambda _root: None)
    monkeypatch.setattr(status_module, "_training_audit_status", selected_audit)

    state = status_module.build_project_state(config, training_profile=TrainingProfile.QUALITY)
    stage = state.stage("training-infrastructure-audit")

    assert observed_profiles == [TrainingProfile.QUALITY]
    assert stage.status is expected_stage
    assert bool(stage.blockers) is has_blocker
    assert stage.audit is audit
    assert stage.metrics["selected_profile"] == "quality"
    if audit is AuditStatus.PASS:
        assert "applicable and passed" in stage.explanation
        assert "Proceed" in stage.next_action
        assert "Remediate" not in stage.next_action


def test_strict_conditioned_v5_freeze_uses_exact_activation_without_legacy_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _conditioned_freeze_config(
        tmp_path,
        schema_version=CONDITIONED_DATASET_FREEZE_SCHEMA,
        allow_training=True,
    )
    report = {}
    _write_json(tmp_path / DEFAULT_CONFIG["training"]["audit_report"], report)
    activation = object()
    selected: list[TrainingProfile] = []

    def load_selected(observed_config, profile, *, require_audit):
        assert observed_config is config
        assert require_audit is False
        selected.append(profile)
        return activation

    monkeypatch.setattr(status_module, "_source_commit", lambda _root: None)
    monkeypatch.setattr(
        status_module,
        "labeling_audit_verification",
        lambda _context: _labeling_verification_without_freeze_scope(),
    )
    monkeypatch.setattr(activation_module, "load_conditioned_training_activation", load_selected)
    monkeypatch.setattr(
        activation_module,
        "training_audit_status",
        lambda observed_config, observed_report, observed_activation: (
            AuditStatus.PASS
            if observed_config is config and observed_report == report and observed_activation is activation
            else AuditStatus.STALE
        ),
    )

    state = status_module.build_project_state(config, training_profile=TrainingProfile.QUALITY)

    assert selected == [TrainingProfile.QUALITY]
    assert state.stage("freeze").status is StageStatus.COMPLETE
    assert state.stage("freeze").production_authorized is True
    assert state.stage("training-audit").audit is AuditStatus.PASS
    assert state.stage("training").status is StageStatus.READY


def test_legacy_conditioned_freeze_still_requires_legacy_labeling_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _conditioned_freeze_config(
        tmp_path,
        schema_version="spritelab.dataset.freeze.conditioned.v4",
        allow_training=True,
    )
    monkeypatch.setattr(status_module, "_source_commit", lambda _root: None)
    monkeypatch.setattr(
        status_module,
        "labeling_audit_verification",
        lambda _context: _labeling_verification_without_freeze_scope(),
    )

    state = status_module.build_project_state(config)

    assert state.stage("freeze").status is StageStatus.BLOCKED
    assert state.stage("freeze").production_authorized is False
    assert state.stage("training").status is StageStatus.BLOCKED


def test_strict_conditioned_v5_activation_failure_marks_audit_stale_and_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _conditioned_freeze_config(
        tmp_path,
        schema_version=CONDITIONED_DATASET_FREEZE_SCHEMA,
        allow_training=True,
    )
    _write_json(tmp_path / DEFAULT_CONFIG["training"]["audit_report"], {})
    selected: list[TrainingProfile] = []

    def reject_stale_activation(_config, profile, *, require_audit):
        assert require_audit is False
        selected.append(profile)
        raise activation_module.ConditionedActivationError(
            "selected_campaign_changed",
            "The selected campaign identity is stale.",
        )

    monkeypatch.setattr(status_module, "_source_commit", lambda _root: None)
    monkeypatch.setattr(
        status_module,
        "labeling_audit_verification",
        lambda _context: _labeling_verification_without_freeze_scope(),
    )
    monkeypatch.setattr(activation_module, "load_conditioned_training_activation", reject_stale_activation)

    state = status_module.build_project_state(config, training_profile=TrainingProfile.RECOMMENDED)

    assert selected == [TrainingProfile.RECOMMENDED]
    assert state.stage("freeze").status is StageStatus.BLOCKED
    assert state.stage("training-audit").status is StageStatus.STALE
    assert state.stage("training-audit").audit is AuditStatus.STALE
    assert state.stage("training").status is StageStatus.BLOCKED


@pytest.mark.parametrize(
    ("audit", "allow_training", "expected_blocker"),
    [
        (AuditStatus.FAIL, True, "Independent training-infrastructure audit: FAIL."),
        (AuditStatus.PASS, False, "Project execution policy does not authorize training."),
    ],
)
def test_strict_conditioned_v5_freeze_does_not_bypass_remaining_training_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit: AuditStatus,
    allow_training: bool,
    expected_blocker: str,
) -> None:
    config = _conditioned_freeze_config(
        tmp_path,
        schema_version=CONDITIONED_DATASET_FREEZE_SCHEMA,
        allow_training=allow_training,
    )
    _write_json(tmp_path / DEFAULT_CONFIG["training"]["audit_report"], {})
    activation = object()

    monkeypatch.setattr(status_module, "_source_commit", lambda _root: None)
    monkeypatch.setattr(
        status_module,
        "labeling_audit_verification",
        lambda _context: _labeling_verification_without_freeze_scope(),
    )
    monkeypatch.setattr(
        activation_module,
        "load_conditioned_training_activation",
        lambda *_args, **_kwargs: activation,
    )
    monkeypatch.setattr(
        activation_module,
        "training_audit_status",
        lambda _config, _report, observed_activation: audit if observed_activation is activation else AuditStatus.STALE,
    )

    state = status_module.build_project_state(config)

    assert state.stage("freeze").status is StageStatus.COMPLETE
    assert state.stage("training").status is StageStatus.BLOCKED
    assert expected_blocker in state.stage("training").blockers


def test_custom_profile_uses_only_the_exact_configured_campaign(tmp_path: Path) -> None:
    configured = {"campaign_id": "configured-custom", "nested": {"enabled": True}}
    document = {
        "product_profiles": {
            "custom": {
                "display": {"display_name": "Configured custom"},
                "campaign": configured,
            }
        }
    }

    selected, display = select_campaign_spec(
        document,
        TrainingProfile.CUSTOM,
        config_directory=tmp_path,
    )
    assert selected == configured
    assert display == {"display_name": "Configured custom"}

    matching, _ = select_campaign_spec(
        document,
        TrainingProfile.CUSTOM,
        config_directory=tmp_path,
        custom_spec=deepcopy(configured),
    )
    assert matching == configured

    with pytest.raises(TrainingProfileError, match="exactly match"):
        select_campaign_spec(
            document,
            TrainingProfile.CUSTOM,
            config_directory=tmp_path,
            custom_spec={**configured, "campaign_id": "arbitrary-override"},
        )

    with pytest.raises(TrainingProfileError, match="not configured"):
        select_campaign_spec(
            {"product_profiles": {"recommended": {"campaign": configured}}},
            TrainingProfile.CUSTOM,
            config_directory=tmp_path,
        )


def test_quality_and_direct_recommended_profile_selection_are_preserved(tmp_path: Path) -> None:
    quality = {"campaign_id": "quality"}
    document = {"product_profiles": {"quality": {"campaign": quality}}}

    assert (
        select_campaign_spec(
            document,
            TrainingProfile.QUALITY,
            config_directory=tmp_path,
        )[0]
        == quality
    )
    assert (
        select_campaign_spec(
            quality,
            TrainingProfile.RECOMMENDED,
            config_directory=tmp_path,
        )[0]
        == quality
    )


def test_portable_conditioned_freeze_path_preserves_campaign_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    campaign_directory = root / "artifacts" / "training"
    built, activation = _portable_conditioned_campaign(
        root,
        campaign_directory,
        monkeypatch,
        campaign_id="conditioned-portable-path-test",
    )

    assert str(built.portable_campaign["identities"]["dataset_freeze_path"]).startswith("../dataset/")
    resolved = _resolve_selected_campaign_paths(built.portable_campaign, campaign_directory)
    assert resolved["identities"]["dataset_freeze_path"] == str(activation.resolve())
    assert plan_campaign(resolved, execution_root=root)["campaign_identity"] == built.campaign["campaign_identity"]
    assert not (root / "runs" / "training").exists()


def test_nested_profile_campaign_resolves_bindings_from_its_own_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    config_directory = root / "artifacts" / "training"
    nested_directory = config_directory / "profiles"
    built, activation = _portable_conditioned_campaign(
        root,
        nested_directory,
        monkeypatch,
        campaign_id="nested-conditioned-profile-test",
    )
    nested_campaign = nested_directory / "quality.json"
    _write_json(nested_campaign, built.portable_campaign)
    document = {
        "product_profiles": {
            "quality": {
                "campaign_path": "profiles/quality.json",
            }
        }
    }

    selected, _display, source_directory = _select_campaign_spec_with_origin(
        document,
        TrainingProfile.QUALITY,
        config_directory=config_directory,
    )
    assert source_directory == nested_directory.resolve()
    resolved = _resolve_selected_campaign_paths(selected, source_directory)
    assert resolved["identities"]["dataset_freeze_path"] == str(activation.resolve())
    assert plan_campaign(resolved, execution_root=root)["campaign_identity"] == built.campaign["campaign_identity"]
    assert not (root / "runs" / "training").exists()

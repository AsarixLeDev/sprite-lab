"""Translate simple product profiles into existing authoritative campaign plans."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingGate, TrainingProfile
from spritelab.remote_compute import ComputeBackend
from spritelab.training.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    audit_resume,
    plan_campaign,
    validate_campaign,
)
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus, StageStatus
from spritelab.v3.status import build_project_state


class TrainingProfileError(ValueError):
    """A product profile cannot select an existing backend campaign spec."""


def project_config_from_context(context: ProjectContext) -> ProjectConfig:
    if context.config:
        return ProjectConfig(
            root=context.project_root.resolve(),
            path=context.config_path,
            values=dict(context.config),
        )
    return ProjectConfig.load(context.project_root)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TrainingProfileError(f"Unable to read training campaign configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingProfileError("Training campaign configuration must be a mapping.")
    return value


def select_campaign_spec(
    document: Mapping[str, Any],
    profile: TrainingProfile,
    *,
    config_directory: Path,
    custom_spec: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select, never reinterpret, a campaign spec registered under a product profile."""

    if profile == TrainingProfile.CUSTOM:
        if custom_spec is None:
            raise TrainingProfileError("Custom profile requires an advanced campaign specification.")
        return dict(custom_spec), {"display_name": "Custom configuration"}
    profiles = document.get("product_profiles")
    if isinstance(profiles, Mapping):
        selected = profiles.get(profile.value)
        if not isinstance(selected, Mapping):
            raise TrainingProfileError(f"Training profile {profile.value!r} is not configured by the backend.")
        display = selected.get("display") if isinstance(selected.get("display"), Mapping) else {}
        if isinstance(selected.get("campaign"), Mapping):
            return dict(selected["campaign"]), dict(display)
        if selected.get("campaign_path"):
            campaign_path = (config_directory / str(selected["campaign_path"])).resolve()
            try:
                campaign_path.relative_to(config_directory.resolve())
            except ValueError as exc:
                raise TrainingProfileError(
                    "Profile campaign_path must remain under the campaign config directory."
                ) from exc
            return _read_mapping(campaign_path), dict(display)
        raise TrainingProfileError(f"Training profile {profile.value!r} has no campaign or campaign_path.")
    if profile != TrainingProfile.RECOMMENDED:
        raise TrainingProfileError("Only the recommended profile is available in this campaign configuration.")
    return dict(document), {"display_name": "Recommended baseline"}


def _dataset_count(config: ProjectConfig) -> int | None:
    path = config.path_for("training", "dataset_freeze") or config.path_for("dataset", "freeze_manifest")
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    for key in ("image_count", "record_count", "dataset_size", "records"):
        count = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(count, int) and count >= 0:
            return count
    return None


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


class TrainingPlanResolver:
    def resolve(
        self,
        context: ProjectContext,
        profile: TrainingProfile,
        backend: ComputeBackend,
        *,
        custom_spec: Mapping[str, Any] | None = None,
        probe_backend: bool = True,
    ) -> ResolvedTrainingPlan:
        config = project_config_from_context(context)
        state = build_project_state(config)
        freeze = state.stage("dataset-freeze")
        audit = state.stage("training-infrastructure-audit")
        authorization = state.stage("training-campaign")
        gates: list[TrainingGate] = [
            TrainingGate(
                "dataset_freeze",
                freeze.status == StageStatus.COMPLETE and freeze.production_authorized,
                freeze.explanation,
                freeze.next_action,
            ),
            TrainingGate(
                "training_audit_applicability",
                audit.audit == AuditStatus.PASS,
                audit.explanation,
                audit.next_action,
                {"audit_status": audit.audit.value},
            ),
            TrainingGate(
                "authorization",
                not authorization.blockers and authorization.production_authorized,
                authorization.explanation,
                authorization.next_action,
            ),
        ]
        campaign: dict[str, Any] | None = None
        resume_report: dict[str, Any] | None = None
        display: dict[str, Any] = {"display_name": "Recommended baseline"}
        campaign_path = config.path_for("training", "campaign_config")
        if campaign_path is None or not campaign_path.is_file():
            gates.append(
                TrainingGate(
                    "campaign_configuration",
                    False,
                    "No authoritative training campaign configuration is configured.",
                    "Configure training.campaign_config after the backend training audit passes.",
                )
            )
        else:
            try:
                document = _read_mapping(campaign_path)
                spec, display = select_campaign_spec(
                    document,
                    profile,
                    config_directory=campaign_path.parent,
                    custom_spec=custom_spec,
                )
                campaign = dict(spec) if spec.get("schema_version") == CAMPAIGN_SCHEMA_VERSION else plan_campaign(spec)
            except TrainingProfileError as exc:
                gates.append(TrainingGate("profile_translation", False, str(exc)))
            except (TypeError, ValueError, OSError) as exc:
                gates.append(TrainingGate("campaign_resolution", False, f"Campaign resolution failed: {exc}"))
        if campaign is not None:
            validation = validate_campaign(campaign)
            validation_messages = [*validation["errors"], *validation["blockers"]]
            gates.extend(
                [
                    TrainingGate(
                        "dataset_identity",
                        not any(
                            "dataset_view_manifest" in item or "split_manifest" in item for item in validation_messages
                        ),
                        "Dataset and split identities are bound."
                        if not any(
                            "dataset_view_manifest" in item or "split_manifest" in item for item in validation_messages
                        )
                        else "Dataset identity validation failed.",
                        "Regenerate the authoritative campaign from the frozen dataset.",
                    ),
                    TrainingGate(
                        "campaign_identity",
                        not any("campaign identity" in item for item in validation_messages),
                        "Campaign identity is valid."
                        if not any("campaign identity" in item for item in validation_messages)
                        else "Campaign identity validation failed.",
                    ),
                    TrainingGate(
                        "completion_contract",
                        not any("artifact contract" in item for item in validation_messages),
                        "Completion artifact contract is complete."
                        if not any("artifact contract" in item for item in validation_messages)
                        else "Completion artifact contract is incomplete.",
                    ),
                    TrainingGate(
                        "campaign_validation",
                        bool(validation["launch_ready"] and not validation_messages),
                        "Resolved backend campaign is launch-ready."
                        if validation["launch_ready"] and not validation_messages
                        else "; ".join(validation_messages) or "Campaign launch authorization is absent.",
                    ),
                ]
            )
            resume_report = audit_resume(campaign, unsafe_resume=False)
            gates.append(
                TrainingGate(
                    "safe_resume",
                    bool(resume_report["safe"]),
                    "Every existing output root is fresh, complete, or safely resumable."
                    if resume_report["safe"]
                    else "; ".join(resume_report["errors"]),
                    "Move foreign outputs aside or restore the exact verified checkpoint identity.",
                )
            )
            gates.append(
                TrainingGate(
                    "output_root",
                    bool(resume_report["safe"]),
                    "Output roots are owned by this campaign."
                    if resume_report["safe"]
                    else "An output root is foreign, stale, or unsafe.",
                )
            )
        estimate = backend.estimate(context, campaign or {})
        if campaign is not None and estimate.disk_required_bytes > 0:
            roots = [Path(str(run["output_root"])) for run in campaign.get("expected_runs", ())]
            free = min((shutil.disk_usage(_nearest_existing_parent(root)).free for root in roots), default=0)
            gates.append(
                TrainingGate(
                    "disk_space",
                    free >= estimate.disk_required_bytes,
                    f"{free} bytes free; {estimate.disk_required_bytes} bytes required.",
                    "Free disk space or select a backend with sufficient persistent storage.",
                )
            )
        else:
            gates.append(
                TrainingGate(
                    "disk_space", True, "No additional product disk estimate is declared; backend checks still apply."
                )
            )
        if probe_backend and not any(not gate.passed for gate in gates):
            capabilities = tuple(backend.probe(context))
            device_ready = bool(capabilities) and all(item.status == ProductStatus.READY for item in capabilities)
            gates.append(
                TrainingGate(
                    "device",
                    device_ready,
                    "; ".join(item.message for item in capabilities)
                    or "Compute backend did not report a device capability.",
                    "Use connection test or resolve the backend environment check.",
                )
            )
        elif probe_backend:
            gates.append(
                TrainingGate("device", False, "Device check was not run because an earlier mandatory gate is closed.")
            )
        else:
            gates.append(
                TrainingGate(
                    "device",
                    True,
                    "Device check is deferred until Start training; page load initializes no device runtime.",
                )
            )
        return ResolvedTrainingPlan(
            profile=profile,
            model_label=str(display.get("display_name") or "Recommended baseline"),
            dataset_count=_dataset_count(config),
            dataset_ready=freeze.status == StageStatus.COMPLETE and freeze.production_authorized,
            backend_id=backend.backend_id,
            campaign=campaign,
            gates=tuple(gates),
            estimate=estimate,
            resume_report=resume_report,
        )

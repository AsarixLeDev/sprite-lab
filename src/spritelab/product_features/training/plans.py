"""Translate simple product profiles into existing authoritative campaign plans."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.training.activation import (
    ConditionedActivationError,
    load_conditioned_training_activation,
)
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingGate, TrainingProfile
from spritelab.remote_compute import ComputeBackend
from spritelab.training.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    audit_resume,
    stable_hash,
    validate_campaign,
)
from spritelab.utils.portable_paths import canonical_portable_relative_path
from spritelab.utils.safe_fs import (
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus


class TrainingProfileError(ValueError):
    """A product profile cannot select an existing backend campaign spec."""


class TrainingPathConfinementError(TrainingProfileError):
    """A passive training read cannot prove that its path is private and confined."""


class _DuplicateMappingKeyError(ValueError):
    """An untrusted campaign mapping contains an ambiguous key."""


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys in every mapping."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            keys: set[Any] = set()
            for key_node, _value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    if key in keys:
                        raise _DuplicateMappingKeyError
                    keys.add(key)
                except TypeError:
                    # The base safe constructor rejects unsupported mapping keys.
                    pass
        return super().construct_mapping(node, deep=deep)


_SYNTHETIC_PATH_CONTRACT_KEY = "_spritelab_training_synthetic_path_contract"
_MAX_CONFINED_MAPPING_BYTES = 4 * 1024 * 1024
_PROFILE_LABELS = {
    TrainingProfile.RECOMMENDED: "Recommended baseline",
    TrainingProfile.FAST_PREVIEW: "Fast preview",
    TrainingProfile.QUALITY: "Quality training",
    TrainingProfile.CUSTOM: "Custom configuration",
}


@dataclass(frozen=True)
class _SyntheticTrainingPathContract:
    project_root: Path


@dataclass(frozen=True)
class _TrainingPathPolicy:
    root: Path
    allow_absolute_fixture_paths: bool


def synthetic_training_path_contract_for_tests(project_root: Path) -> dict[str, object]:
    """Return a non-serializable, project-bound compatibility contract for synthetic tests.

    The contract never permits external or linked paths and is ignored whenever a
    real config entry exists. Production configuration cannot express this value.
    """

    root = Path(os.path.abspath(os.fspath(project_root)))
    return {_SYNTHETIC_PATH_CONTRACT_KEY: _SyntheticTrainingPathContract(root)}


def _path_policy(context: ProjectContext) -> _TrainingPathPolicy:
    root = Path(os.path.abspath(os.fspath(context.project_root)))
    contract = context.config.get(_SYNTHETIC_PATH_CONTRACT_KEY)
    config_exists = context.config_path is not None and os.path.lexists(context.config_path)
    trusted_fixture = (
        not config_exists and type(contract) is _SyntheticTrainingPathContract and contract.project_root == root
    )
    return _TrainingPathPolicy(root, trusted_fixture)


def _path_refusal() -> TrainingPathConfinementError:
    return TrainingPathConfinementError(
        "A configured training file or output folder is not a safe project-contained path."
    )


def _confined_path(
    raw: Any,
    *,
    origin: Path,
    policy: _TrainingPathPolicy,
    allow_parent_parts: bool,
    allow_absolute: bool,
) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\x00" in raw:
        raise _path_refusal()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    absolute_shape = posix.is_absolute() or windows.is_absolute() or bool(windows.drive)
    if absolute_shape:
        candidate = Path(raw)
        if not allow_absolute or not candidate.is_absolute():
            raise _path_refusal()
    else:
        parts = raw.split("/")
        parent_count = 0
        while parent_count < len(parts) and parts[parent_count] == "..":
            parent_count += 1
        suffix = "/".join(parts[parent_count:])
        if (not allow_parent_parts and parent_count) or not suffix:
            raise _path_refusal()
        try:
            canonical_portable_relative_path(suffix)
        except ValueError as exc:
            raise _path_refusal() from exc
        candidate = origin.joinpath(*posix.parts)
    try:
        return require_confined_path(candidate, policy.root)
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise _path_refusal() from exc


def _verify_single_link_file(path: Path, policy: _TrainingPathPolicy) -> None:
    try:
        with open_anchored_directory(path.parent, policy.root) as parent:
            metadata = parent.lstat(path.name)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise _path_refusal()
    except TrainingPathConfinementError:
        raise
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise _path_refusal() from exc


def _confined_existing_file(
    raw: Any,
    *,
    origin: Path,
    policy: _TrainingPathPolicy,
    allow_parent_parts: bool,
    allow_absolute: bool,
) -> Path:
    path = _confined_path(
        raw,
        origin=origin,
        policy=policy,
        allow_parent_parts=allow_parent_parts,
        allow_absolute=allow_absolute,
    )
    _verify_single_link_file(path, policy)
    return path


def _confined_optional_project_file(
    config: ProjectConfig,
    section: str,
    key: str,
    policy: _TrainingPathPolicy,
) -> Path | None:
    values = config.values.get(section)
    raw = values.get(key) if isinstance(values, Mapping) else None
    if not raw:
        return None
    path = _confined_path(
        raw,
        origin=policy.root,
        policy=policy,
        allow_parent_parts=False,
        allow_absolute=policy.allow_absolute_fixture_paths,
    )
    if os.path.lexists(path):
        _verify_single_link_file(path, policy)
    return path


def _verify_output_path(path: Path, policy: _TrainingPathPolicy) -> None:
    current = path
    while not os.path.lexists(current) and current.parent != current:
        current = current.parent
    try:
        current = require_confined_path(current, policy.root, allow_root=True)
        if not current.is_dir() or (current != policy.root and current.is_mount()):
            raise _path_refusal()
        with open_anchored_directory(current, policy.root):
            pass
    except TrainingPathConfinementError:
        raise
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise _path_refusal() from exc


def _confined_output_path(
    raw: Any,
    *,
    origin: Path,
    policy: _TrainingPathPolicy,
    allow_absolute: bool,
) -> Path:
    path = _confined_path(
        raw,
        origin=origin,
        policy=policy,
        allow_parent_parts=True,
        allow_absolute=allow_absolute,
    )
    _verify_output_path(path, policy)
    return path


def _read_confined_text(path: Path, policy: _TrainingPathPolicy) -> str:
    def unchanged(metadata: os.stat_result, identity: OwnedFileIdentity, before: os.stat_result) -> bool:
        return (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and identity.matches(metadata)
            and metadata.st_size == before.st_size
            and metadata.st_mtime_ns == before.st_mtime_ns
        )

    try:
        with open_anchored_directory(path.parent, policy.root) as parent:
            before = parent.lstat(path.name)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > _MAX_CONFINED_MAPPING_BYTES
            ):
                raise _path_refusal()
            identity = OwnedFileIdentity.from_stat(before)
            descriptor = parent.open_file(path.name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
            try:
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    if not unchanged(os.fstat(handle.fileno()), identity, before):
                        raise _path_refusal()
                    payload = handle.read(_MAX_CONFINED_MAPPING_BYTES + 1)
                    after = os.fstat(handle.fileno())
                    if len(payload) != before.st_size or not unchanged(after, identity, before):
                        raise _path_refusal()
                visible = parent.lstat(path.name)
                if not unchanged(visible, identity, before):
                    raise _path_refusal()
                parent.verify()
                return payload.decode("utf-8")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except TrainingPathConfinementError:
        raise
    except (OSError, UnicodeDecodeError, UnsafeFilesystemOperation, ValueError) as exc:
        raise _path_refusal() from exc


def _public_training_value(value: Any, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_training_value(child, project_root) for key, child in value.items()}
    if isinstance(value, list):
        return [_public_training_value(child, project_root) for child in value]
    if isinstance(value, tuple):
        return [_public_training_value(child, project_root) for child in value]
    if not isinstance(value, str):
        return value
    public = value
    for spelling in {str(project_root), project_root.as_posix()}:
        public = public.replace(spelling, "<project>")
    public = re.sub(r"(?i)(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/])[^\s;,]*", "<local-path>", public)
    public = re.sub(r"(?<![:/\w.])/(?!/)[^\s;,]+", "<local-path>", public)
    return public


def project_config_from_context(context: ProjectContext) -> ProjectConfig:
    if context.config:
        values = dict(context.config)
        values.pop(_SYNTHETIC_PATH_CONTRACT_KEY, None)
        return ProjectConfig(
            root=context.project_root.resolve(),
            path=context.config_path,
            values=values,
        )
    return ProjectConfig.load(context.project_root)


def _read_mapping(path: Path, *, path_policy: _TrainingPathPolicy | None = None) -> dict[str, Any]:
    try:
        if path_policy is not None:
            content = _read_confined_text(path, path_policy)
        else:
            with path.open("rb") as handle:
                payload = handle.read(_MAX_CONFINED_MAPPING_BYTES + 1)
            if len(payload) > _MAX_CONFINED_MAPPING_BYTES:
                raise TrainingProfileError("Unable to read training campaign configuration.")
            content = payload.decode("utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(content, Loader=_StrictSafeLoader)
        else:
            value = json.loads(content, object_pairs_hook=_strict_json_mapping)
    except TrainingPathConfinementError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        _DuplicateMappingKeyError,
    ) as exc:
        raise TrainingProfileError("Unable to read training campaign configuration.") from exc
    if not isinstance(value, dict):
        raise TrainingProfileError("Training campaign configuration must be a mapping.")
    return value


def _strict_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise _DuplicateMappingKeyError
        mapping[key] = value
    return mapping


def _resolve_selected_campaign_paths(spec: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Resolve portable campaign bindings relative to the selected document."""

    result = deepcopy(dict(spec))

    def resolve(value: Any) -> str:
        candidate = Path(str(value)).expanduser()
        return str(candidate if candidate.is_absolute() else (directory / candidate).resolve())

    identities = result.get("identities")
    if isinstance(identities, dict):
        for field in (
            "dataset_freeze_path",
            "dataset_view_manifest_path",
            "split_manifest_path",
            "conditioning_vocabulary_path",
        ):
            if identities.get(field):
                identities[field] = resolve(identities[field])
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("benchmark_manifest_path"):
        evaluation["benchmark_manifest_path"] = resolve(evaluation["benchmark_manifest_path"])
    if result.get("output_root"):
        result["output_root"] = resolve(result["output_root"])
    return result


def _resolve_confined_campaign_paths(
    spec: Mapping[str, Any],
    directory: Path,
    policy: _TrainingPathPolicy,
) -> dict[str, Any]:
    """Resolve every passively consumed campaign path after confinement checks."""

    result = deepcopy(dict(spec))
    identities = result.get("identities")
    if isinstance(identities, dict):
        for field in (
            "dataset_freeze_path",
            "dataset_view_manifest_path",
            "split_manifest_path",
            "conditioning_vocabulary_path",
        ):
            if identities.get(field):
                identities[field] = str(
                    _confined_existing_file(
                        identities[field],
                        origin=directory,
                        policy=policy,
                        allow_parent_parts=True,
                        allow_absolute=policy.allow_absolute_fixture_paths,
                    )
                )
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("benchmark_manifest_path"):
        evaluation["benchmark_manifest_path"] = str(
            _confined_existing_file(
                evaluation["benchmark_manifest_path"],
                origin=directory,
                policy=policy,
                allow_parent_parts=True,
                allow_absolute=policy.allow_absolute_fixture_paths,
            )
        )
    if result.get("output_root"):
        result["output_root"] = str(
            _confined_output_path(
                result["output_root"],
                origin=directory,
                policy=policy,
                allow_absolute=policy.allow_absolute_fixture_paths,
            )
        )
    if result.get("campaign_artifact_root"):
        result["campaign_artifact_root"] = str(
            _confined_output_path(
                result["campaign_artifact_root"],
                origin=directory,
                policy=policy,
                allow_absolute=policy.allow_absolute_fixture_paths,
            )
        )
    if result.get("schema_version") == CAMPAIGN_SCHEMA_VERSION:
        expected = result.get("expected_output_roots")
        if isinstance(expected, list):
            result["expected_output_roots"] = [
                str(
                    _confined_output_path(
                        raw,
                        origin=directory,
                        policy=policy,
                        allow_absolute=policy.allow_absolute_fixture_paths,
                    )
                )
                for raw in expected
            ]
        runs = result.get("expected_runs")
        if isinstance(runs, list):
            for run in runs:
                if isinstance(run, dict) and run.get("output_root"):
                    run["output_root"] = str(
                        _confined_output_path(
                            run["output_root"],
                            origin=directory,
                            policy=policy,
                            allow_absolute=policy.allow_absolute_fixture_paths,
                        )
                    )
    return result


def _select_campaign_spec_with_origin(
    document: Mapping[str, Any],
    profile: TrainingProfile,
    *,
    config_directory: Path,
    custom_spec: Mapping[str, Any] | None = None,
    path_policy: _TrainingPathPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Select a registered campaign together with its path-resolution origin."""

    profiles = document.get("product_profiles")
    if isinstance(profiles, Mapping):
        selected = profiles.get(profile.value)
        if not isinstance(selected, Mapping):
            raise TrainingProfileError(f"Training profile {profile.value!r} is not configured by the backend.")
        display = selected.get("display") if isinstance(selected.get("display"), Mapping) else {}
        source: dict[str, Any]
        source_directory = config_directory
        if isinstance(selected.get("campaign"), Mapping):
            source = dict(selected["campaign"])
        elif selected.get("campaign_path"):
            if path_policy is not None:
                campaign_path = _confined_existing_file(
                    selected["campaign_path"],
                    origin=config_directory,
                    policy=path_policy,
                    allow_parent_parts=False,
                    allow_absolute=False,
                )
            else:
                campaign_path = (config_directory / str(selected["campaign_path"])).resolve()
                try:
                    campaign_path.relative_to(config_directory.resolve())
                except ValueError as exc:
                    raise TrainingProfileError(
                        "Profile campaign_path must remain under the campaign config directory."
                    ) from exc
            source = _read_mapping(campaign_path, path_policy=path_policy)
            source_directory = campaign_path.parent
        else:
            raise TrainingProfileError(f"Training profile {profile.value!r} has no campaign or campaign_path.")
        if profile is TrainingProfile.CUSTOM and custom_spec is not None:
            try:
                matches_configured = isinstance(custom_spec, Mapping) and stable_hash(dict(custom_spec)) == stable_hash(
                    source
                )
            except (TypeError, ValueError):
                matches_configured = False
            if not matches_configured:
                raise TrainingProfileError("The custom campaign must exactly match the configured custom profile.")
        result_display = dict(display)
        if profile is TrainingProfile.CUSTOM:
            result_display.setdefault("display_name", "Custom configuration")
        return source, result_display, source_directory
    if profile != TrainingProfile.RECOMMENDED:
        raise TrainingProfileError("Only the recommended profile is available in this campaign configuration.")
    return dict(document), {"display_name": "Recommended baseline"}, config_directory


def select_campaign_spec(
    document: Mapping[str, Any],
    profile: TrainingProfile,
    *,
    config_directory: Path,
    custom_spec: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select, never reinterpret, a campaign spec registered under a product profile."""

    spec, display, _source_directory = _select_campaign_spec_with_origin(
        document,
        profile,
        config_directory=config_directory,
        custom_spec=custom_spec,
        path_policy=None,
    )
    return spec, display


def _verify_planned_output_roots(campaign: Mapping[str, Any], policy: _TrainingPathPolicy) -> None:
    for run in campaign.get("expected_runs", ()):
        if not isinstance(run, Mapping) or not run.get("output_root"):
            continue
        _confined_output_path(
            run["output_root"],
            origin=policy.root,
            policy=policy,
            allow_absolute=True,
        )


def _execution_authorized(config: ProjectConfig) -> bool:
    execution = config.values.get("execution")
    return isinstance(execution, Mapping) and execution.get("allow_training") is True


def _public_resume_report(report: Mapping[str, Any]) -> dict[str, Any]:
    allowed_statuses = {
        "complete",
        "corrupt",
        "foreign",
        "fresh",
        "partial_invalid",
        "partial_valid",
        "valid_resumable",
    }
    counts: dict[str, int] = {}
    runs = report.get("runs")
    if isinstance(runs, list):
        for run in runs:
            status_value = run.get("status") if isinstance(run, Mapping) else None
            status = status_value if isinstance(status_value, str) and status_value in allowed_statuses else "unknown"
            counts[status] = counts.get(status, 0) + 1
    errors = report.get("errors")
    foreign = report.get("foreign_run_roots")
    return {
        "safe": report.get("safe") is True,
        "run_count": len(runs) if isinstance(runs, list) else 0,
        "run_status_counts": dict(sorted(counts.items())),
        "error_count": len(errors) if isinstance(errors, list) else 0,
        "foreign_run_root_count": len(foreign) if isinstance(foreign, list) else 0,
        "paths_exposed": False,
    }


def _path_blocked_plan(
    context: ProjectContext,
    profile: TrainingProfile,
    backend: ComputeBackend,
    *,
    probe_backend: bool,
) -> ResolvedTrainingPlan:
    gates = [
        TrainingGate(
            "path_confinement",
            False,
            "A configured training file or output folder could not be verified safely inside this project.",
            "Use project-contained regular files with portable paths and no linked filesystem seams.",
            {"paths_exposed": False},
        ),
        TrainingGate(
            "device",
            not probe_backend,
            "Device check was not run because training path confinement failed."
            if probe_backend
            else "Device check is deferred until Start training; page load initializes no device runtime.",
        ),
    ]
    return ResolvedTrainingPlan(
        profile=profile,
        model_label="Training configuration",
        dataset_count=None,
        dataset_ready=False,
        backend_id=backend.backend_id,
        campaign=None,
        gates=tuple(gates),
        estimate=backend.estimate(context, {}),
        resume_report=None,
    )


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


class TrainingPlanResolver:
    def __init__(self, *, activation_loader: Callable[..., Any] | None = None) -> None:
        self.activation_loader = activation_loader or load_conditioned_training_activation

    def resolve(
        self,
        context: ProjectContext,
        profile: TrainingProfile,
        backend: ComputeBackend,
        *,
        custom_spec: Mapping[str, Any] | None = None,
        probe_backend: bool = True,
    ) -> ResolvedTrainingPlan:
        policy = _path_policy(context)
        config = project_config_from_context(context)
        try:
            _confined_optional_project_file(config, "dataset", "view_manifest", policy)
            _confined_optional_project_file(config, "dataset", "freeze_manifest", policy)
            _confined_optional_project_file(config, "training", "dataset_freeze", policy)
            campaign_path = _confined_optional_project_file(config, "training", "campaign_config", policy)
        except TrainingPathConfinementError:
            return _path_blocked_plan(context, profile, backend, probe_backend=probe_backend)

        campaign: dict[str, Any] | None = None
        resume_report: dict[str, Any] | None = None
        activation: Any | None = None
        selection_error = False
        if campaign_path is not None and campaign_path.is_file():
            try:
                document = _read_mapping(campaign_path, path_policy=policy)
                preflight_spec, _display, source_directory = _select_campaign_spec_with_origin(
                    document,
                    profile,
                    config_directory=campaign_path.parent,
                    custom_spec=custom_spec,
                    path_policy=policy,
                )
                _resolve_confined_campaign_paths(preflight_spec, source_directory, policy)
            except TrainingPathConfinementError:
                return _path_blocked_plan(context, profile, backend, probe_backend=probe_backend)
            except TrainingProfileError:
                selection_error = True

        if campaign_path is not None and campaign_path.is_file() and not selection_error:
            try:
                activation = self.activation_loader(
                    context,
                    profile,
                    custom_spec=custom_spec,
                    require_audit=False,
                )
                activated_campaign = getattr(activation, "campaign", None)
                manifest = getattr(activation, "manifest", None)
                activated_audit = getattr(activation, "audit_status", None)
                image_count = manifest.get("image_count") if isinstance(manifest, Mapping) else None
                if (
                    not isinstance(activated_campaign, Mapping)
                    or not isinstance(manifest, Mapping)
                    or not isinstance(activated_audit, AuditStatus)
                    or isinstance(image_count, bool)
                    or not isinstance(image_count, int)
                    or image_count < 0
                ):
                    raise TrainingProfileError("Conditioned training activation is incomplete.")
                campaign = deepcopy(dict(activated_campaign))
                _verify_planned_output_roots(campaign, policy)
            except TrainingPathConfinementError:
                return _path_blocked_plan(context, profile, backend, probe_backend=probe_backend)
            except (ConditionedActivationError, OSError, TypeError, ValueError):
                activation = None
                campaign = None

        audit_status = getattr(activation, "audit_status", AuditStatus.NOT_AUDITED)
        if not isinstance(audit_status, AuditStatus):
            audit_status = AuditStatus.NOT_AUDITED
        activation_verified = activation is not None
        activated_config = getattr(activation, "config", None)
        authorized = _execution_authorized(activated_config if isinstance(activated_config, ProjectConfig) else config)
        gates: list[TrainingGate] = [
            TrainingGate(
                "dataset_freeze",
                activation_verified,
                "The exact conditioned dataset freeze and publication are verified."
                if activation_verified
                else "The conditioned dataset freeze could not be verified.",
                "Prepare one project-contained conditioned Dataset-v5 activation." if not activation_verified else None,
            ),
            TrainingGate(
                "training_audit_applicability",
                audit_status is AuditStatus.PASS,
                "The exact selected activation has an applicable PASS training audit."
                if audit_status is AuditStatus.PASS
                else "The exact selected activation has no applicable PASS training audit.",
                "Run or refresh the independent training audit for this exact activation."
                if audit_status is not AuditStatus.PASS
                else None,
                {"audit_status": audit_status.value, "paths_exposed": False},
            ),
            TrainingGate(
                "authorization",
                authorized,
                "Project execution policy authorizes training."
                if authorized
                else "Project execution policy does not authorize training.",
                "Set execution.allow_training to true after reviewing the exact activation."
                if not authorized
                else None,
            ),
        ]
        if campaign_path is None or not campaign_path.is_file():
            gates.append(
                TrainingGate(
                    "campaign_configuration",
                    False,
                    "No authoritative training campaign configuration is configured.",
                    "Configure training.campaign_config after the backend training audit passes.",
                )
            )
        elif selection_error:
            gates.append(
                TrainingGate(
                    "profile_translation",
                    False,
                    "The configured training profile could not be selected safely.",
                )
            )
        elif not activation_verified:
            gates.append(
                TrainingGate(
                    "conditioned_activation",
                    False,
                    "The exact conditioned training activation could not be verified.",
                    "Repair the configured freeze, campaign, and audit bindings inside this project.",
                    {"paths_exposed": False},
                )
            )
        if campaign is not None:
            try:
                validation = validate_campaign(campaign)
                validation_ready = bool(
                    validation.get("launch_ready") and not validation.get("errors") and not validation.get("blockers")
                )
            except (OSError, TypeError, ValueError):
                validation_ready = False
            gates.extend(
                [
                    TrainingGate(
                        "dataset_identity",
                        validation_ready,
                        "Dataset and split identities are bound."
                        if validation_ready
                        else "Dataset identity validation failed.",
                        "Regenerate the authoritative campaign from the frozen dataset.",
                    ),
                    TrainingGate(
                        "campaign_identity",
                        validation_ready,
                        "Campaign identity is valid." if validation_ready else "Campaign identity validation failed.",
                    ),
                    TrainingGate(
                        "completion_contract",
                        validation_ready,
                        "Completion artifact contract is complete."
                        if validation_ready
                        else "Completion artifact contract is incomplete.",
                    ),
                    TrainingGate(
                        "campaign_validation",
                        validation_ready,
                        "Resolved backend campaign is launch-ready."
                        if validation_ready
                        else "Campaign validation failed without exposing local details.",
                    ),
                ]
            )
            try:
                private_resume_report = audit_resume(campaign, unsafe_resume=False)
            except (OSError, TypeError, ValueError):
                private_resume_report = {"safe": False, "errors": ["blocked"], "runs": []}
            resume_report = _public_resume_report(private_resume_report)
            gates.append(
                TrainingGate(
                    "safe_resume",
                    private_resume_report.get("safe") is True,
                    "Every existing output root is fresh, complete, or safely resumable."
                    if private_resume_report.get("safe") is True
                    else "Existing output roots could not be verified as safely resumable.",
                    "Move foreign outputs aside or restore the exact verified checkpoint identity.",
                )
            )
            gates.append(
                TrainingGate(
                    "output_root",
                    private_resume_report.get("safe") is True,
                    "Output roots are owned by this campaign."
                    if private_resume_report.get("safe") is True
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
                    "; ".join(str(_public_training_value(item.message, policy.root)) for item in capabilities)
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
            model_label=_PROFILE_LABELS[profile],
            dataset_count=(
                getattr(activation, "manifest", {}).get("image_count")
                if isinstance(getattr(activation, "manifest", None), Mapping)
                and isinstance(getattr(activation, "manifest", {}).get("image_count"), int)
                and not isinstance(getattr(activation, "manifest", {}).get("image_count"), bool)
                else None
            ),
            dataset_ready=activation_verified,
            backend_id=backend.backend_id,
            campaign=campaign,
            gates=tuple(gates),
            estimate=estimate,
            resume_report=resume_report,
        )

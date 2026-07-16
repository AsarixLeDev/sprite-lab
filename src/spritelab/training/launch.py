"""Authoritative, CPU-only preparation and verification for training launches.

Every training process or remote transport boundary consumes a receipt produced
here.  Product status projections and adapter-supplied identity strings are not
validation evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from spritelab.product_web.events import verify_event_migration
from spritelab.training.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    CampaignResumeError,
    CampaignValidationError,
    audit_resume,
    canonical_json,
    file_sha256,
    is_concrete_hash,
    plan_campaign,
    stable_hash,
    validate_campaign,
)

TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION = "spritelab_training_launch_receipt_v3"
EVENT_HISTORY_ORIGIN_RECEIPT_STATES = frozenset({"native", "migrated_legacy"})
TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION = "spritelab_training_launch_context_v1"
TRAINING_LAUNCH_RECEIPT_TTL_SECONDS = 300
_VALIDATOR_ISSUER_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class TrainingLaunchReceipt:
    schema_version: str
    receipt_id: str
    campaign_identity_sha256: str
    campaign_manifest_sha256: str
    campaign_validation_report_sha256: str
    training_code_identity_sha256: str
    resolved_configuration_sha256: str
    dataset_identity: str
    view_identity: str
    split_identity: str
    architecture_identity: str
    optimizer_identity: str
    schedule_identity: str
    loss_identity: str
    maximum_optimizer_steps: int
    run_identity: str
    cell_identity: str
    seed: int
    output_root_identity: str
    compute_backend_id: str
    execution_spec_sha256: str
    argv_sha256: str
    resume_validation_sha256: str
    event_migration_state: str
    event_migration_identity_sha256: str
    event_history_origin: str
    event_migration_required: bool
    event_migration_record_sha256: str | None
    event_canonical_prefix_sha256: str | None
    event_canonical_identity_sha256: str | None
    source_checkpoint_identity: str | None
    unsafe_resume: bool
    launch_authorized: bool
    execute_confirmed: bool
    created_at_utc: str
    expires_at_utc: str
    validator_proof_sha256: str
    receipt_sha256: str

    def body(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("receipt_sha256", None)
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingLaunchContext:
    """Authoritative inputs retained so an adapter can revalidate a receipt."""

    schema_version: str
    campaign_config_path: Path
    campaign_profile: str
    project_root: Path
    run_id: str
    resume: bool
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ValidatedTrainingLaunch:
    receipt: TrainingLaunchReceipt
    validator_context: TrainingLaunchContext
    campaign: Mapping[str, Any]
    run: Mapping[str, Any]
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    output_root: Path


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignValidationError(f"campaign configuration is missing or not a regular file: {path}")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignValidationError(f"campaign configuration cannot be loaded: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignValidationError("campaign configuration must contain a mapping")
    return value


def load_exact_campaign_configuration(path: str | Path, *, profile: str = "recommended") -> dict[str, Any]:
    """Load and fully resolve the exact configured product or low-level campaign."""

    source = Path(path).resolve()
    document = _read_mapping(source)
    selected: Mapping[str, Any] = document
    profiles = document.get("product_profiles")
    if isinstance(profiles, Mapping):
        profile_entry = profiles.get(profile)
        if not isinstance(profile_entry, Mapping):
            raise CampaignValidationError(f"campaign profile {profile!r} is not configured")
        if isinstance(profile_entry.get("campaign"), Mapping):
            selected = profile_entry["campaign"]
        elif profile_entry.get("campaign_path"):
            nested = (source.parent / str(profile_entry["campaign_path"])).resolve()
            try:
                nested.relative_to(source.parent.resolve())
            except ValueError as exc:
                raise CampaignValidationError("campaign_path escapes its configuration directory") from exc
            selected = _read_mapping(nested)
        else:
            raise CampaignValidationError(f"campaign profile {profile!r} has no campaign configuration")
    campaign = dict(selected)
    return campaign if campaign.get("schema_version") == CAMPAIGN_SCHEMA_VERSION else plan_campaign(campaign)


def _normalise_environment(environment: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for raw_key, raw_value in dict(environment or {}).items():
        key, value = str(raw_key), str(raw_value)
        if not key or "\x00" in key or "=" in key or "\x00" in value:
            raise CampaignValidationError("execution environment contains an unsafe key or NUL byte")
        values.append((key, value))
    return tuple(sorted(values))


def _output_root_identity(campaign: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "campaign_identity": campaign.get("campaign_identity"),
            "run_identity": run.get("run_identity"),
            "output_root": str(Path(str(run.get("output_root"))).resolve()),
        }
    )


def _execution_spec(
    *,
    campaign: Mapping[str, Any],
    run: Mapping[str, Any],
    backend_id: str,
    project_root: Path,
    argv: Sequence[str],
    environment: tuple[tuple[str, str], ...],
    resume: bool,
    source_checkpoint_identity: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "spritelab_training_execution_spec_v1",
        "campaign_identity_sha256": campaign.get("campaign_identity"),
        "run_identity": run.get("run_identity"),
        "cell_id": run.get("cell_id"),
        "seed": run.get("seed"),
        "backend_id": backend_id,
        "project_root": str(project_root.resolve()),
        "output_root": str(Path(str(run.get("output_root"))).resolve()),
        "argv": list(argv),
        "environment": dict(environment),
        "resume": resume,
        "source_checkpoint_identity": source_checkpoint_identity,
    }


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignValidationError(f"receipt {label} is malformed") from exc
    if parsed.tzinfo is None:
        raise CampaignValidationError(f"receipt {label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validator_proof(body: Mapping[str, Any]) -> str:
    protected = {key: value for key, value in body.items() if key not in {"validator_proof_sha256", "receipt_sha256"}}
    return hmac.new(_VALIDATOR_ISSUER_KEY, canonical_json(protected).encode("utf-8"), hashlib.sha256).hexdigest()


def _authoritative_snapshot(
    context: TrainingLaunchContext,
    *,
    backend_id: str,
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    output_root: str | Path | None = None,
    require_materialized_configs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[tuple[str, str], ...], dict[str, Any]]:
    if context.schema_version != TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION:
        raise CampaignValidationError("training launch validator context schema is unsupported")
    if not backend_id.strip():
        raise CampaignValidationError("compute backend identity is required")
    campaign = load_exact_campaign_configuration(context.campaign_config_path, profile=context.campaign_profile)
    validation = validate_campaign(campaign)
    problems = [*validation["errors"], *validation["blockers"]]
    if problems or not validation["launch_ready"]:
        raise CampaignValidationError("campaign validation failed: " + "; ".join(problems or ["not launch-ready"]))
    if not campaign.get("executable") or campaign.get("plan_status") != "ready":
        raise CampaignValidationError("campaign is blocked or executable=false")
    if campaign.get("launch_authorized") is not True:
        raise CampaignValidationError("campaign launch_authorized must be true")
    run = next((item for item in campaign.get("expected_runs", ()) if item.get("run_id") == context.run_id), None)
    if not isinstance(run, Mapping):
        raise CampaignValidationError(f"run {context.run_id!r} is not an exact campaign cell")
    resume_report = audit_resume(campaign, unsafe_resume=False)
    if not resume_report["safe"]:
        raise CampaignResumeError("unsafe campaign state: " + "; ".join(resume_report["errors"]))
    state = next(item for item in resume_report["runs"] if item["run_id"] == context.run_id)
    must_resume = state["status"] == "valid_resumable"
    if context.resume != must_resume:
        raise CampaignResumeError("launch resume mode does not match the current authoritative output-root state")
    expected_command = tuple(str(item) for item in run.get("experiment_command") or ())
    source_checkpoint_identity: str | None = None
    if must_resume:
        checkpoint = Path(str(state["checkpoint"]))
        source_checkpoint_identity = file_sha256(checkpoint)
        expected_command = (*expected_command, "--resume", str(checkpoint))
    command = expected_command if argv is None else tuple(str(item) for item in argv)
    if command != expected_command:
        raise CampaignValidationError("requested argv does not match the exact campaign execution contract")
    normalised_environment = _normalise_environment(environment)
    if normalised_environment != context.environment:
        raise CampaignValidationError("requested environment changed after launch validation")
    expected_root = Path(str(run["output_root"])).resolve()
    if output_root is not None and Path(output_root).resolve() != expected_root:
        raise CampaignValidationError("requested output root does not match the exact campaign run")
    migration_verification = verify_event_migration(
        str(run["run_id"]),
        expected_root,
        origin_required=must_resume,
    )
    if not migration_verification.resume_compatible:
        raise CampaignResumeError(
            "event migration evidence is not safe for continuation: "
            f"{migration_verification.state.value}: {migration_verification.message}"
        )
    if must_resume and migration_verification.migration_required and not migration_verification.migration_verified:
        raise CampaignResumeError(
            "a recorded migrated event history must fully verify before continuation: "
            f"{migration_verification.state.value}: {migration_verification.message}"
        )
    for candidate in campaign.get("expected_runs", ()):
        path = Path(str(candidate["resolved_config_path"]))
        if not require_materialized_configs:
            if stable_hash(candidate.get("resolved_config")) != candidate.get("resolved_config_sha256"):
                raise CampaignValidationError(f"embedded resolved run config identity changed: {path}")
            continue
        if not path.is_file():
            raise CampaignValidationError(f"resolved run config is missing: {path}")
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignValidationError(f"resolved run config cannot be read: {path}: {exc}") from exc
        if actual != candidate.get("resolved_config") or stable_hash(actual) != candidate.get("resolved_config_sha256"):
            raise CampaignValidationError(f"resolved run config identity changed: {path}")
    execution_spec = _execution_spec(
        campaign=campaign,
        run=run,
        backend_id=backend_id,
        project_root=context.project_root,
        argv=command,
        environment=normalised_environment,
        resume=must_resume,
        source_checkpoint_identity=source_checkpoint_identity,
    )
    snapshot = {
        "campaign": campaign,
        "run": dict(run),
        "validation": validation,
        "resume_report": resume_report,
        "event_migration_verification": migration_verification,
        "source_checkpoint_identity": source_checkpoint_identity,
        "execution_spec": execution_spec,
    }
    return campaign, dict(run), command, normalised_environment, snapshot


def validate_training_launch_plan(
    campaign_config_path: str | Path,
    *,
    compute_backend_id: str,
    project_root: str | Path,
    campaign_profile: str = "recommended",
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Perform the full CPU-only dry-run validation without issuing a receipt."""

    config_path = Path(campaign_config_path).resolve()
    campaign = load_exact_campaign_configuration(config_path, profile=campaign_profile)
    resume_report = audit_resume(campaign, unsafe_resume=False)
    launches: list[dict[str, Any]] = []
    states = {item["run_id"]: item for item in resume_report.get("runs", ())}
    for run in campaign.get("expected_runs", ()):
        context = TrainingLaunchContext(
            TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION,
            config_path,
            campaign_profile,
            Path(project_root).resolve(),
            str(run["run_id"]),
            states.get(run["run_id"], {}).get("status") == "valid_resumable",
            _normalise_environment(environment),
        )
        _, current_run, argv, _, snapshot = _authoritative_snapshot(
            context,
            backend_id=compute_backend_id,
            environment=dict(context.environment),
            require_materialized_configs=False,
        )
        launches.append(
            {
                "run_id": current_run["run_id"],
                "run_identity": current_run["run_identity"],
                "seed": current_run["seed"],
                "argv_sha256": stable_hash(list(argv)),
                "execution_spec_sha256": stable_hash(snapshot["execution_spec"]),
                "resume": context.resume,
            }
        )
    return {
        "schema_version": "spritelab_training_launch_dry_run_v1",
        "campaign_id": campaign.get("campaign_id"),
        "campaign_identity_sha256": campaign.get("campaign_identity"),
        "campaign_manifest_sha256": file_sha256(config_path),
        "compute_backend_id": compute_backend_id,
        "validation": validate_campaign(campaign),
        "resume_validation": resume_report,
        "launches": launches,
        "valid": bool(launches) and resume_report.get("safe") is True,
        "receipts_issued": 0,
        "processes_started": 0,
    }


def prepare_validated_training_launch(
    campaign_config_path: str | Path,
    *,
    run_id: str,
    compute_backend_id: str,
    project_root: str | Path,
    execute_confirmed: bool,
    campaign_profile: str = "recommended",
    environment: Mapping[str, str] | None = None,
    resume: bool = False,
    now: datetime | None = None,
) -> ValidatedTrainingLaunch:
    """Run every launch gate in order and issue one short-lived receipt."""

    if not execute_confirmed:
        raise CampaignValidationError("training launch requires explicit execution confirmation")
    config_path = Path(campaign_config_path).resolve()
    context = TrainingLaunchContext(
        TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION,
        config_path,
        str(campaign_profile),
        Path(project_root).resolve(),
        str(run_id),
        bool(resume),
        _normalise_environment(environment),
    )
    campaign, run, argv, normalised_environment, snapshot = _authoritative_snapshot(
        context, backend_id=compute_backend_id, environment=dict(context.environment)
    )
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = created + timedelta(seconds=TRAINING_LAUNCH_RECEIPT_TTL_SECONDS)
    identities = dict(campaign.get("identities") or {})
    code_identity = dict(campaign.get("code_identity") or {})
    cell = next(item for item in campaign.get("architecture_cells", ()) if item.get("cell_id") == run.get("cell_id"))
    body: dict[str, Any] = {
        "schema_version": TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
        "receipt_id": secrets.token_hex(16),
        "campaign_identity_sha256": campaign["campaign_identity"],
        "campaign_manifest_sha256": file_sha256(config_path),
        "campaign_validation_report_sha256": stable_hash(snapshot["validation"]),
        "training_code_identity_sha256": code_identity.get("sha256"),
        "resolved_configuration_sha256": run.get("resolved_config_sha256"),
        "dataset_identity": identities.get("dataset_identity_hash", identities.get("dataset_view_manifest_hash")),
        "view_identity": identities.get("dataset_view_manifest_hash"),
        "split_identity": identities.get("split_manifest_hash"),
        "architecture_identity": stable_hash(cell),
        "optimizer_identity": identities.get("optimizer_config_hash"),
        "schedule_identity": identities.get("schedule_config_hash"),
        "loss_identity": identities.get("loss_config_hash"),
        "maximum_optimizer_steps": dict(campaign.get("training") or {}).get("max_optimizer_steps"),
        "run_identity": run.get("run_identity"),
        "cell_identity": stable_hash({"cell_id": run.get("cell_id"), "run_identity": run.get("run_identity")}),
        "seed": run.get("seed"),
        "output_root_identity": _output_root_identity(campaign, run),
        "compute_backend_id": compute_backend_id,
        "execution_spec_sha256": stable_hash(snapshot["execution_spec"]),
        "argv_sha256": stable_hash(list(argv)),
        "resume_validation_sha256": stable_hash(snapshot["resume_report"]),
        "event_migration_state": snapshot["event_migration_verification"].state.value,
        "event_migration_identity_sha256": snapshot["event_migration_verification"].evidence_sha256,
        "event_history_origin": snapshot["event_migration_verification"].event_history_origin,
        "event_migration_required": snapshot["event_migration_verification"].migration_required,
        "event_migration_record_sha256": snapshot["event_migration_verification"].migration_record_sha256,
        "event_canonical_prefix_sha256": snapshot["event_migration_verification"].canonical_prefix_sha256,
        "event_canonical_identity_sha256": snapshot["event_migration_verification"].canonical_event_identity_sha256,
        "source_checkpoint_identity": snapshot["source_checkpoint_identity"],
        "unsafe_resume": False,
        "launch_authorized": True,
        "execute_confirmed": True,
        "created_at_utc": created.isoformat(),
        "expires_at_utc": expires.isoformat(),
    }
    hash_fields = (
        "campaign_identity_sha256",
        "campaign_manifest_sha256",
        "campaign_validation_report_sha256",
        "training_code_identity_sha256",
        "resolved_configuration_sha256",
        "dataset_identity",
        "view_identity",
        "split_identity",
        "architecture_identity",
        "optimizer_identity",
        "schedule_identity",
        "loss_identity",
        "run_identity",
        "cell_identity",
        "output_root_identity",
        "execution_spec_sha256",
        "argv_sha256",
        "resume_validation_sha256",
        "event_migration_identity_sha256",
        "event_canonical_prefix_sha256",
        "event_canonical_identity_sha256",
    )
    invalid = [field for field in hash_fields if not is_concrete_hash(body.get(field))]
    if invalid:
        raise CampaignValidationError("launch receipt has non-concrete protected identities: " + ", ".join(invalid))
    if body["event_history_origin"] not in EVENT_HISTORY_ORIGIN_RECEIPT_STATES:
        raise CampaignValidationError("launch receipt event-history origin is not a controlled origin state")
    if type(body["event_migration_required"]) is not bool:
        raise CampaignValidationError("launch receipt event-migration-required flag must be a strict boolean")
    optional_event_hash_fields = ("event_migration_record_sha256",)
    malformed_event_hashes = [
        field
        for field in optional_event_hash_fields
        if body.get(field) is not None and not is_concrete_hash(body.get(field))
    ]
    if malformed_event_hashes:
        raise CampaignValidationError(
            "launch receipt has malformed event evidence identities: " + ", ".join(malformed_event_hashes)
        )
    if body["event_migration_required"] and (
        body["event_migration_record_sha256"] is None or body["event_canonical_prefix_sha256"] is None
    ):
        raise CampaignValidationError(
            "launch receipt requires concrete migration-record and canonical-prefix identities for migrated runs"
        )
    if (body["event_history_origin"] == "migrated_legacy") is not body["event_migration_required"]:
        raise CampaignValidationError("launch receipt origin and migration-required classification disagree")
    if not body["event_migration_required"] and body["event_migration_record_sha256"] is not None:
        raise CampaignValidationError("native launch receipt cannot carry a migration-record identity")
    body["validator_proof_sha256"] = _validator_proof(body)
    body["receipt_sha256"] = stable_hash(body)
    receipt = TrainingLaunchReceipt(**body)
    return ValidatedTrainingLaunch(
        receipt,
        context,
        campaign,
        run,
        argv,
        dict(normalised_environment),
        Path(str(run["output_root"])).resolve(),
    )


def verify_validated_training_launch(
    receipt: TrainingLaunchReceipt,
    context: TrainingLaunchContext,
    *,
    compute_backend_id: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    output_root: str | Path,
    campaign_identity: str,
    run_identity: str,
    now: datetime | None = None,
) -> ValidatedTrainingLaunch:
    """Recompute authoritative state immediately before the process/transport seam."""

    if not isinstance(receipt, TrainingLaunchReceipt):
        raise CampaignValidationError("a typed validator-issued training launch receipt is required")
    if receipt.schema_version != TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION:
        raise CampaignValidationError("training launch receipt schema is unsupported")
    if stable_hash(receipt.body()) != receipt.receipt_sha256:
        raise CampaignValidationError("training launch receipt self-hash is invalid")
    if not hmac.compare_digest(_validator_proof(receipt.body()), receipt.validator_proof_sha256):
        raise CampaignValidationError("training launch receipt was not issued by the active validator")
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created = _parse_utc(receipt.created_at_utc, label="created_at_utc")
    expires = _parse_utc(receipt.expires_at_utc, label="expires_at_utc")
    if expires <= created or expires - created > timedelta(seconds=TRAINING_LAUNCH_RECEIPT_TTL_SECONDS):
        raise CampaignValidationError("training launch receipt lifetime is invalid")
    if checked_at < created - timedelta(seconds=5) or checked_at > expires:
        raise CampaignValidationError("training launch receipt is expired or not yet valid")
    if (
        receipt.unsafe_resume is not False
        or receipt.launch_authorized is not True
        or receipt.execute_confirmed is not True
    ):
        raise CampaignValidationError("training launch receipt does not prove safe authorization and confirmation")
    if not is_concrete_hash(campaign_identity) or not is_concrete_hash(run_identity):
        raise CampaignValidationError("compute request campaign and run identities must be concrete SHA-256 values")
    if campaign_identity != receipt.campaign_identity_sha256 or run_identity != receipt.run_identity:
        raise CampaignValidationError("compute request identity does not match its launch receipt")
    if compute_backend_id != receipt.compute_backend_id:
        raise CampaignValidationError("compute backend does not match the launch receipt")
    campaign, run, command, normalised_environment, _snapshot = _authoritative_snapshot(
        context,
        backend_id=compute_backend_id,
        argv=argv,
        environment=environment,
        output_root=output_root,
    )
    expected = prepare_validated_training_launch(
        context.campaign_config_path,
        run_id=context.run_id,
        compute_backend_id=compute_backend_id,
        project_root=context.project_root,
        execute_confirmed=True,
        campaign_profile=context.campaign_profile,
        environment=dict(normalised_environment),
        resume=context.resume,
        now=created,
    ).receipt
    ignored = {"receipt_id", "validator_proof_sha256", "receipt_sha256"}
    stale = [key for key, value in expected.body().items() if key not in ignored and receipt.body().get(key) != value]
    if stale:
        raise CampaignValidationError("training launch receipt is stale or forged: " + ", ".join(sorted(stale)))
    return ValidatedTrainingLaunch(
        receipt,
        context,
        campaign,
        run,
        command,
        dict(normalised_environment),
        Path(str(run["output_root"])).resolve(),
    )


def receipt_with_recomputed_hash(receipt: TrainingLaunchReceipt, **changes: Any) -> TrainingLaunchReceipt:
    """Test/evidence helper: a self-consistent body still needs authoritative verification."""

    changed = replace(receipt, **changes, receipt_sha256="")
    return replace(changed, receipt_sha256=stable_hash(changed.body()))

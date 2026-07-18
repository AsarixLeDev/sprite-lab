"""Product orchestration that delegates every launch gate to existing backends."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from spritelab.product_core import (
    ProductBlocker,
    ProductEvent,
    ProductResult,
    ProductRun,
    ProductStatus,
    ProductWarning,
    ProjectContext,
    strict_json_dumps,
    strict_json_loads,
)
from spritelab.product_features.training.action_lock import TrainingActionLock, TrainingActionLockError
from spritelab.product_features.training.activation import (
    ConditionedActivationError,
    load_conditioned_training_activation,
)
from spritelab.product_features.training.cloud_challenge import CloudChallengeError, CloudChallengeStore
from spritelab.product_features.training.config import ComputeSettings
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import (
    ResolvedTrainingPlan,
    TrainingGate,
    TrainingProfile,
)
from spritelab.product_features.training.plans import TrainingPlanResolver, project_config_from_context
from spritelab.product_web.events import (
    EVENT_FILENAME,
    LEGACY_EVENT_FILENAME,
    EventMigrationState,
    EventRepository,
    verify_event_migration,
)
from spritelab.remote_compute import (
    ArtifactReference,
    ComputeBackend,
    ComputeBackendError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputePoll,
    ComputeStatus,
    HostedBackendRegistry,
    LocalComputeBackend,
    PreparedCompute,
    ResumeRequest,
    RunPodComputeBackend,
    RunPodSettings,
    SSHComputeBackend,
    SSHSettings,
    select_hosted_backend,
)
from spritelab.remote_compute.utils import validate_identifier
from spritelab.training.campaign import (
    CampaignValidationError,
    audit_artifact_completeness,
    execute_campaign,
    stable_hash,
)
from spritelab.training.launch import ValidatedTrainingLaunch
from spritelab.v3.model import AuditStatus

EVALUATION_CHECKPOINT_BINDING_SCHEMA = "spritelab.training.evaluation-checkpoint-binding.v1"


class BackendSelectionError(ValueError):
    pass


def backend_from_context(
    context: ProjectContext, *, hosted_backends: HostedBackendRegistry | None = None
) -> ComputeBackend:
    settings = context.config.get("compute", {}).get("training", {"type": "local"})
    if not isinstance(settings, Mapping):
        raise BackendSelectionError("compute.training must be a mapping.")
    backend_type = str(settings.get("type") or "local").lower()
    if backend_type == "local":
        return LocalComputeBackend()
    if backend_type in {"ssh", "remote_ssh"}:
        return SSHComputeBackend(SSHSettings.from_mapping(settings))
    if backend_type == "runpod":
        return RunPodComputeBackend(RunPodSettings.from_mapping(settings))
    if backend_type in {"other", "plugin", "hosted"}:
        backend_id = str(settings.get("backend_id") or "")
        if not backend_id:
            raise BackendSelectionError("Other provider requires compute.training.backend_id.")
        return select_hosted_backend(hosted_backends or HostedBackendRegistry(), backend_id)
    if hosted_backends and hosted_backends.get(backend_type):
        return select_hosted_backend(hosted_backends, backend_type)
    raise BackendSelectionError(f"Unknown training compute backend: {backend_type}")


@dataclass
class TrainingSession:
    run_id: str
    backend: ComputeBackend
    plan: ResolvedTrainingPlan
    jobs: list[ComputeJob] = field(default_factory=list)
    dashboard: DashboardState | None = None
    cursors: dict[str, int] = field(default_factory=dict)
    prepared: dict[str, PreparedCompute] = field(default_factory=dict)
    prepared_stages: dict[str, str] = field(default_factory=dict)
    requests: dict[str, ComputeJobRequest] = field(default_factory=dict)
    dataset_identity: str | None = None
    view_identity: str | None = None
    launch_authorization_evidence_sha256: str | None = None
    operation_nonce: str | None = None
    operation_action: str | None = None
    seed_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    job_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    unknown_backend_operation_count: int = 0
    reconstructed: bool = False


class TrainingService:
    def __init__(
        self,
        context: ProjectContext,
        backend: ComputeBackend,
        *,
        resolver: TrainingPlanResolver | None = None,
        activation_loader: Callable[..., Any] | None = None,
        audit_snapshot_opener: Callable[..., Any] | None = None,
    ) -> None:
        self.context = context
        self.backend = backend
        self.activation_loader = activation_loader or load_conditioned_training_activation
        self.audit_snapshot_opener = audit_snapshot_opener or _default_training_audit_snapshot_opener
        self.resolver = resolver or TrainingPlanResolver(activation_loader=self.activation_loader)
        self.sessions: dict[str, TrainingSession] = {}
        self.repository = EventRepository(context.runs_directory, private_roots=(context.project_root,))
        self.cloud_challenges = CloudChallengeStore(context.project_root)

    def plan(
        self,
        profile: TrainingProfile = TrainingProfile.RECOMMENDED,
        *,
        custom_spec: Mapping[str, Any] | None = None,
        before_launch: bool = False,
    ) -> ResolvedTrainingPlan:
        return self.resolver.resolve(
            self.context,
            profile,
            self.backend,
            custom_spec=custom_spec,
            probe_backend=before_launch,
        )

    def status(self, profile: TrainingProfile = TrainingProfile.RECOMMENDED) -> ProductResult:
        if profile is not TrainingProfile.RECOMMENDED:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Production conditioned training accepts only the recommended profile.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "conditioned_profile_ineligible",
                        "Quality, fast-preview, and custom profiles are not production-authorized.",
                    ),
                ),
                data={
                    "profile": profile.value,
                    "ready": False,
                    "implementation_state": "Implementation repaired",
                    "certification_state": "Profile not production-certified",
                    "availability_state": "Training unavailable",
                },
            )
        plan = self.plan(profile, before_launch=False)
        blockers = tuple(ProductBlocker(gate.gate_id, gate.message, gate.resolution) for gate in plan.blockers)
        return ProductResult(
            status=ProductStatus.READY if plan.ready else ProductStatus.BLOCKED,
            feature="training",
            message="Training is ready."
            if plan.ready
            else "Temporarily unavailable while final safety checks are verified.",
            blockers=blockers,
            data={
                **plan.to_dict(),
                "implementation_state": "Implementation repaired",
                "certification_state": "Independent certification pending"
                if not plan.ready
                else "Certified inputs ready",
                "availability_state": "Training available" if plan.ready else "Training unavailable",
            },
        )

    def start(
        self,
        profile: TrainingProfile = TrainingProfile.RECOMMENDED,
        *,
        custom_spec: Mapping[str, Any] | None = None,
        cloud_confirmation: bool = False,
        cloud_challenge: str | None = None,
        resume: bool = False,
    ) -> ProductResult:
        if profile is not TrainingProfile.RECOMMENDED or custom_spec is not None:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Production conditioned training accepts only the recommended profile.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "conditioned_profile_ineligible",
                        "Quality, fast-preview, and custom profiles are not production-authorized.",
                    ),
                ),
                data={"backend_launches": 0},
            )
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                return self._start_locked(
                    profile,
                    custom_spec=custom_spec,
                    cloud_confirmation=cloud_confirmation,
                    cloud_challenge=cloud_challenge,
                    resume=resume,
                )
        except TrainingActionLockError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Training Start is temporarily unavailable while activation or another launch action commits.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "training_action_conflict",
                        "Wait for the current activation, Start, or Resume action to finish, then retry.",
                    ),
                ),
                data={"backend_launches": 0},
            )

    def _start_locked(
        self,
        profile: TrainingProfile,
        *,
        custom_spec: Mapping[str, Any] | None,
        cloud_confirmation: bool,
        cloud_challenge: str | None,
        resume: bool,
    ) -> ProductResult:
        del cloud_confirmation
        if self.backend.is_cloud and not cloud_challenge:
            plan = self.plan(profile, custom_spec=custom_spec, before_launch=False)
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Cloud training requires a fresh one-use server challenge immediately before launch.",
                blockers=(
                    ProductBlocker(
                        "cloud_challenge_required",
                        "Confirm that the selected hosted resource may incur cost and obtain a fresh challenge.",
                        "Review GPU, disk, shutdown policy, and credential status, then confirm Start training.",
                    ),
                ),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        plan = self.plan(profile, custom_spec=custom_spec, before_launch=True)
        if plan.blockers:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Training was not launched because a mandatory gate is closed.",
                blockers=tuple(ProductBlocker(gate.gate_id, gate.message, gate.resolution) for gate in plan.blockers),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        campaign = plan.campaign
        assert campaign is not None
        try:
            activation = self.activation_loader(
                self.context,
                profile,
                custom_spec=custom_spec,
                expected_campaign=campaign,
                require_audit=True,
            )
        except (ConditionedActivationError, OSError, TypeError, KeyError) as exc:
            code = exc.code if isinstance(exc, ConditionedActivationError) else "conditioned_activation_invalid"
            message = (
                exc.public_message
                if isinstance(exc, ConditionedActivationError)
                else "The exact conditioned training activation could not be verified safely."
            )
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Training was not launched because its conditioned activation is not applicable.",
                blockers=(ProductBlocker(code, message),),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        try:
            activation_record_identity, project_config_sha256 = _committed_activation_binding(
                activation,
                campaign,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Training was not launched because activation is not durably committed.",
                blockers=(ProductBlocker("activation_commit_required", str(exc)),),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        config = project_config_from_context(self.context)
        session_id = str(campaign["campaign_id"])
        try:
            with self.audit_snapshot_opener(config, None, activation) as audit_snapshot:
                launch_authorization_evidence_sha256 = _require_applicable_training_audit_snapshot(audit_snapshot)
                return self._start_authorized_locked(
                    profile,
                    custom_spec=custom_spec,
                    cloud_challenge=cloud_challenge,
                    resume=resume,
                    plan=plan,
                    campaign=campaign,
                    activation=activation,
                    activation_record_identity=activation_record_identity,
                    project_config_sha256=project_config_sha256,
                    config=config,
                    audit_snapshot=audit_snapshot,
                    launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                )
        except (OSError, RuntimeError, TypeError, ValueError):
            session = self.sessions.get(session_id)
            return ProductResult(
                ProductStatus.BLOCKED,
                "Training was not launched because its retained infrastructure-audit evidence changed.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "training_audit_snapshot_stale",
                        "The exact report, hash inventory, receipt, or action record changed during authorization.",
                    ),
                ),
                data={
                    **plan.to_dict(),
                    "backend_launches": len(session.jobs) if session is not None else 0,
                    "cancel_available": bool(session and (session.jobs or session.prepared)),
                },
            )

    def _start_authorized_locked(
        self,
        profile: TrainingProfile,
        *,
        custom_spec: Mapping[str, Any] | None,
        cloud_challenge: str | None,
        resume: bool,
        plan: ResolvedTrainingPlan,
        campaign: Mapping[str, Any],
        activation: Any,
        activation_record_identity: str,
        project_config_sha256: str,
        config: Any,
        audit_snapshot: Any,
        launch_authorization_evidence_sha256: str,
    ) -> ProductResult:
        campaign_config_path = config.path_for("training", "campaign_config")
        if campaign_config_path is None or not campaign_config_path.is_file():
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="A validated training campaign could not be prepared. No process was started.",
                blockers=(ProductBlocker("campaign_configuration", "training.campaign_config is required."),),
                data={"backend_launches": 0},
            )
        session_id = str(campaign["campaign_id"])
        if self.repository.state(session_id):
            return ProductResult(
                ProductStatus.BLOCKED,
                "This campaign already has durable product run state.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "existing_run",
                        "Open the existing run and use safe Resume only when a verified resume point is available.",
                    ),
                ),
                data={"backend_launches": 0},
            )
        try:
            self._materialize_resolved_configs(campaign)
        except (OSError, ValueError) as exc:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Resolved backend configuration could not be bound.",
                blockers=(ProductBlocker("resolved_configuration", str(exc)),),
                data={"backend_launches": 0},
            )
        operation_nonce = f"operation-{uuid.uuid4().hex}"
        dashboard = DashboardState(session_id, self.backend.backend_id)
        session = TrainingSession(
            session_id,
            self.backend,
            plan,
            dashboard=dashboard,
            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
            operation_nonce=operation_nonce,
            operation_action="start",
        )
        started_at: str | None = None
        durable_start_claimed = False
        configured_inputs = self._configured_inputs(campaign)
        challenge_bindings = self._cloud_action_bindings(
            action="start",
            run_id=session_id,
            campaign=campaign,
            activation_record_identity=activation_record_identity,
            project_config_sha256=project_config_sha256,
            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
        )
        cloud_authorized = False
        if self.backend.is_cloud:
            try:
                audit_snapshot.verify_unchanged()
                self.cloud_challenges.consume_locked(
                    str(cloud_challenge or ""),
                    expected_bindings=challenge_bindings,
                    operation_nonce=operation_nonce,
                )
            except CloudChallengeError as exc:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Cloud training was not launched because its one-use confirmation is invalid.",
                    feature="training",
                    blockers=(ProductBlocker(exc.code, exc.public_message),),
                    data={"backend_launches": 0},
                )
            cloud_authorized = True

        def claim_durable_start() -> None:
            """Claim the campaign only at the first validated backend-operation seam."""

            nonlocal durable_start_claimed, started_at
            if durable_start_claimed:
                return
            claimed_at = datetime.now(timezone.utc).isoformat()
            self.repository.create_run(
                session_id,
                feature="training",
                command="training.start",
                status=ProductStatus.RUNNING.value,
                stage="campaign",
                started_at=claimed_at,
                resumable=False,
                backend_id=self.backend.backend_id,
                backend_run_reference=session_id,
                backend_identity={
                    "backend_id": self.backend.backend_id,
                    "configuration_identity_sha256": self._backend_configuration_identity(),
                },
                extra={
                    "plan": _plan_to_state(plan),
                    "jobs": [],
                    "cursors": {},
                    "conditioned_activation": activation.to_contract_dict(),
                    "active_operation": _operation_record(
                        action="start",
                        run_id=session_id,
                        operation_nonce=operation_nonce,
                        campaign_identity=str(campaign["campaign_identity"]),
                        backend_id=self.backend.backend_id,
                        backend_configuration_identity=self._backend_configuration_identity(),
                        activation_commit_record_identity=activation_record_identity,
                        project_config_sha256=project_config_sha256,
                        launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                        status="RUNNING",
                    ),
                    "operation_history": [],
                    "seed_outcomes": {},
                    "job_outcomes": {},
                },
            )
            self.sessions[session_id] = session
            durable_start_claimed = True
            started_at = claimed_at
            self._apply(
                session,
                ProductEvent(
                    run_id=session_id,
                    timestamp=claimed_at,
                    feature="training",
                    stage="campaign",
                    event_type="training_started",
                    status=ProductStatus.RUNNING,
                    current=0,
                    total=len(campaign["expected_runs"]),
                    message="Validated training campaign started.",
                ),
            )

        def runner(command: list[str], **kwargs: Any) -> Any:
            validated = kwargs.get("validated_launch")
            if not isinstance(validated, ValidatedTrainingLaunch):
                raise CampaignValidationError("Campaign runner requires a validator-issued launch receipt.")
            _require_validated_launch_authorization(validated, launch_authorization_evidence_sha256)
            audit_snapshot.verify_unchanged()
            self._bind_training_identity(session, validated)
            run = validated.run
            seed_key = str(run["run_id"])
            seed = int(run["seed"])
            request = ComputeJobRequest(
                run_id=str(run["run_id"]),
                command=tuple(command),
                idempotency_key=str(run["run_id"]),
                campaign_identity=validated.receipt.campaign_identity_sha256,
                run_identity=str(run["run_identity"]),
                local_project_root=self.context.project_root,
                output_root=validated.output_root,
                event_path=validated.output_root / EVENT_FILENAME,
                environment=validated.environment,
                execution_spec_identity=validated.receipt.execution_spec_sha256,
                output_root_identity=validated.receipt.output_root_identity,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                compute_backend_id=self.backend.backend_id,
                launch_receipt=validated.receipt,
                validator_context=validated.validator_context,
                launch_authorization_verifier=audit_snapshot,
            )
            audit_snapshot.verify_unchanged()
            claim_durable_start()
            session.seed_outcomes[seed_key] = _seed_outcome(run, status="RUNNING", stage="prepare")
            self._persist_session(session)
            stage = "prepare"
            try:
                prepared = self._idempotent_backend_call(
                    session,
                    lambda: _verify_audit_then(
                        audit_snapshot,
                        lambda: self.backend.prepare(self.context, request),
                    ),
                )
                session.prepared[request.idempotency_key] = prepared
                session.prepared_stages[request.idempotency_key] = "prepared"
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="RUNNING", stage="upload")
                self._persist_session(session)
                stage = "upload"
                self._idempotent_backend_call(
                    session,
                    lambda: _verify_audit_then(
                        audit_snapshot,
                        lambda: self.backend.upload(
                            prepared,
                            [Path(str(run["resolved_config_path"])), *configured_inputs],
                        ),
                    ),
                )
                session.prepared_stages[request.idempotency_key] = "uploaded"
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="RUNNING", stage="launch")
                session.prepared_stages[request.idempotency_key] = "possibly_launched"
                self._persist_session(session)
                stage = "launch"
                job = self._idempotent_backend_call(
                    session,
                    lambda: _verify_audit_then(
                        audit_snapshot,
                        lambda: self.backend.launch(
                            prepared,
                            request,
                            cloud_confirmation=cloud_authorized,
                        ),
                    ),
                )
                session.jobs.append(job)
                session.requests[job.job_id] = request
                session.prepared.pop(request.idempotency_key, None)
                session.prepared_stages.pop(request.idempotency_key, None)
                session.prepared[job.job_id] = prepared
                session.prepared_stages[job.job_id] = "launched"
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="LAUNCHED", stage="launched")
                session.job_outcomes[job.job_id] = _job_outcome(job, status="LAUNCHED")
                self._persist_session(session)
            except (CampaignValidationError, ComputeBackendError, OSError, ValueError):
                uncertain = stage in {"prepare", "launch"}
                if stage == "launch" and request.idempotency_key in session.prepared:
                    session.prepared_stages[request.idempotency_key] = "possibly_launched"
                session.seed_outcomes[seed_key] = _seed_outcome(
                    run,
                    status="UNCERTAIN" if uncertain else "BLOCKED",
                    stage=stage,
                )
                self._persist_session(session)
                raise
            self._apply(
                session,
                ProductEvent(
                    run_id=session_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    feature="training",
                    stage="campaign",
                    event_type="seed_launched",
                    status=ProductStatus.RUNNING,
                    current=len(session.jobs),
                    total=len(campaign["expected_runs"]),
                    message=f"Seed {run['seed']} launched.",
                    metrics={"seed": seed, "checkpoint_schedule": run["expected_checkpoint_steps"]},
                ),
            )
            self._persist_session(session)
            return SimpleNamespace(returncode=0)

        try:
            audit_snapshot.verify_unchanged()
            execution = execute_campaign(
                campaign,
                execute=True,
                confirm_execute=True,
                campaign_config_path=campaign_config_path,
                campaign_profile=profile.value,
                compute_backend_id=self.backend.backend_id,
                execution_environment=self._execution_environment(),
                project_root=self.context.project_root,
                resume=resume,
                unsafe_resume=False,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                runner=runner,
            )
        except (CampaignValidationError, ComputeBackendError, OSError, ValueError) as exc:
            del exc
            if durable_start_claimed:
                _complete_seed_outcomes(session, campaign)
                uncertain = any(item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values())
                self._apply(
                    session,
                    ProductEvent(
                        run_id=session_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="campaign",
                        event_type="training_blocked",
                        status=ProductStatus.BLOCKED,
                        message="The authoritative campaign refused launch.",
                        metrics={
                            "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
                            "resource_state_uncertain": uncertain,
                            "may_accrue_cost": bool(uncertain and session.backend.is_cloud),
                            "shutdown_guidance": (
                                "Open the configured backend and stop any resource associated with this campaign."
                                if uncertain
                                else None
                            ),
                        },
                    ),
                )
                self._persist_session(session)
                self._finish_operation(session, "UNCERTAIN" if uncertain else "BLOCKED")
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="The authoritative campaign refused launch.",
                blockers=(
                    ProductBlocker(
                        "authoritative_launch",
                        "A validated backend seam failed or could not be reconciled safely.",
                    ),
                ),
                data={
                    "backend_launches": len(session.jobs),
                    "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
                    "job_outcomes": _sorted_outcomes(session.job_outcomes),
                    "cancel_available": bool(
                        session.jobs
                        or session.prepared
                        or any(item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values())
                    ),
                    **(
                        {
                            "run_id": session_id,
                            "dashboard": session.dashboard.to_dict(),
                        }
                        if durable_start_claimed and session.dashboard is not None
                        else {}
                    ),
                },
            )
        if not durable_start_claimed or started_at is None:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="The authoritative campaign completed without requesting a backend operation.",
                blockers=(
                    ProductBlocker(
                        "authoritative_launch",
                        "No validated backend operation was requested; the campaign remains safely retryable.",
                    ),
                ),
                data={"backend_launches": 0},
            )
        _complete_seed_outcomes(session, campaign)
        self._persist_session(session)
        self._finish_operation(session, "LAUNCHED")
        return ProductResult(
            status=ProductStatus.RUNNING,
            feature="training",
            message="Training launched through the validated campaign contract.",
            run=ProductRun(
                run_id=session_id,
                feature="training",
                action_id="start",
                status=ProductStatus.RUNNING,
                backend_id=self.backend.backend_id,
                started_at=started_at,
            ),
            data={
                "execution": execution,
                "dashboard": dashboard.to_dict(),
                "training_identity": self._training_identity_projection(session),
                "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
            },
        )

    def issue_cloud_challenge(
        self,
        *,
        action: str,
        run_id: str | None = None,
        profile: TrainingProfile = TrainingProfile.RECOMMENDED,
    ) -> ProductResult:
        """Issue one short-lived confirmation without touching a backend seam."""

        if not self.backend.is_cloud:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "A cloud confirmation challenge is unnecessary for the selected backend.",
                feature="training",
            )
        if action not in {"start", "resume"} or profile is not TrainingProfile.RECOMMENDED:
            return ProductResult(
                ProductStatus.BLOCKED,
                "The cloud confirmation request is not production-authorized.",
                feature="training",
                blockers=(ProductBlocker("cloud_challenge_action_invalid", "Use recommended Start or Resume."),),
            )
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                if action == "start":
                    plan = self.plan(profile, before_launch=False)
                    if plan.blockers or plan.campaign is None:
                        return ProductResult(
                            ProductStatus.BLOCKED,
                            "A cloud challenge cannot be issued while mandatory training gates are closed.",
                            feature="training",
                            blockers=tuple(
                                ProductBlocker(item.gate_id, item.message, item.resolution) for item in plan.blockers
                            ),
                        )
                    campaign = plan.campaign
                    bound_run_id = str(campaign["campaign_id"])
                    if run_id not in {None, bound_run_id} or self.repository.state(bound_run_id):
                        return ProductResult(
                            ProductStatus.BLOCKED,
                            "The requested Start challenge does not match a fresh campaign.",
                            feature="training",
                            blockers=(ProductBlocker("cloud_challenge_run_mismatch", "Reload Training and retry."),),
                        )
                    activation = self.activation_loader(
                        self.context,
                        profile,
                        expected_campaign=campaign,
                        require_audit=True,
                    )
                else:
                    if not run_id:
                        return ProductResult(
                            ProductStatus.BLOCKED,
                            "Resume requires an exact retained run ID.",
                            feature="training",
                            blockers=(ProductBlocker("cloud_challenge_run_mismatch", "Reload the retained run."),),
                        )
                    session = self._session(run_id)
                    if session is None or session.plan.profile is not TrainingProfile.RECOMMENDED:
                        return ProductResult(
                            ProductStatus.BLOCKED,
                            "The requested Resume challenge does not match an eligible retained run.",
                            feature="training",
                            blockers=(ProductBlocker("cloud_challenge_run_mismatch", "Reload the retained run."),),
                        )
                    campaign = session.plan.campaign
                    if not isinstance(campaign, Mapping):
                        raise ValueError("The retained campaign is unavailable.")
                    bound_run_id = run_id
                    activation = self.activation_loader(
                        self.context,
                        TrainingProfile.RECOMMENDED,
                        expected_campaign=campaign,
                        require_audit=True,
                    )
                record_identity, config_identity = _committed_activation_binding(activation, campaign)
                config = project_config_from_context(self.context)
                with self.audit_snapshot_opener(config, None, activation) as audit_snapshot:
                    launch_authorization_evidence_sha256 = _require_applicable_training_audit_snapshot(audit_snapshot)
                    issued = self.cloud_challenges.issue(
                        self._cloud_action_bindings(
                            action=action,
                            run_id=bound_run_id,
                            campaign=campaign,
                            activation_record_identity=record_identity,
                            project_config_sha256=config_identity,
                            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                        )
                    )
        except TrainingActionLockError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "A cloud challenge cannot be issued while activation or another launch action is committing.",
                feature="training",
                blockers=(ProductBlocker("training_action_conflict", "Wait for the current action, then retry."),),
            )
        except (
            CloudChallengeError,
            ConditionedActivationError,
            OSError,
            RuntimeError,
            TypeError,
            KeyError,
            ValueError,
        ) as exc:
            code = getattr(exc, "code", "cloud_challenge_unavailable")
            message = getattr(exc, "public_message", "The exact cloud action could not be verified safely.")
            return ProductResult(
                ProductStatus.BLOCKED,
                "A fresh cloud confirmation challenge could not be issued.",
                feature="training",
                blockers=(ProductBlocker(str(code), str(message)),),
            )
        return ProductResult(
            ProductStatus.READY,
            "Fresh cloud confirmation challenge issued for this exact action.",
            feature="training",
            data=issued,
        )

    def _cloud_action_bindings(
        self,
        *,
        action: str,
        run_id: str,
        campaign: Mapping[str, Any],
        activation_record_identity: str,
        project_config_sha256: str,
        launch_authorization_evidence_sha256: str,
    ) -> dict[str, str]:
        return {
            "action": action,
            "run_id": run_id,
            "campaign_identity_sha256": str(campaign["campaign_identity"]),
            "backend_id": self.backend.backend_id,
            "backend_configuration_identity_sha256": self._backend_configuration_identity(),
            "project_config_sha256": project_config_sha256,
            "activation_commit_record_identity": activation_record_identity,
            "launch_authorization_evidence_sha256": launch_authorization_evidence_sha256,
        }

    def _backend_configuration_identity(self) -> str:
        compute = self.context.config.get("compute") if isinstance(self.context.config, Mapping) else None
        training = compute.get("training") if isinstance(compute, Mapping) else None
        configured = ComputeSettings.from_mapping(training, allow_unavailable=True)
        return stable_hash(configured.to_persisted_dict())

    def _assert_operation_owned(self, session: TrainingSession) -> None:
        if not session.operation_nonce or not session.operation_action:
            raise CampaignValidationError("Durable Training operation ownership is unavailable.")
        state = self.repository.state(session.run_id)
        operation = _validate_operation_record(state.get("active_operation"), expected_run_id=session.run_id)
        if (
            operation.get("operation_nonce") != session.operation_nonce
            or operation.get("action") != session.operation_action
            or operation.get("status") != "RUNNING"
            or operation.get("backend_id") != session.backend.backend_id
            or operation.get("backend_configuration_identity_sha256") != self._backend_configuration_identity()
            or operation.get("launch_authorization_evidence_sha256") != session.launch_authorization_evidence_sha256
        ):
            raise CampaignValidationError("Durable Training operation ownership changed before a backend seam.")

    def _idempotent_backend_call(self, session: TrainingSession, operation: Callable[[], Any]) -> Any:
        """Retry one identity-bound seam once to reconcile side-effect-then-throw adapters."""

        self._assert_operation_owned(session)
        try:
            return operation()
        except (ComputeBackendError, OSError):
            self._assert_operation_owned(session)
            return operation()

    def _claim_operation(
        self,
        session: TrainingSession,
        *,
        action: str,
        operation_nonce: str,
        campaign: Mapping[str, Any],
        activation_commit_record_identity: str,
        project_config_sha256: str,
        launch_authorization_evidence_sha256: str,
        recover_stale: bool = False,
    ) -> None:
        state = self.repository.state(session.run_id)
        active = state.get("active_operation")
        history = [
            _validate_operation_record(item, expected_run_id=session.run_id)
            for item in state.get("operation_history", ())
            if isinstance(item, Mapping)
        ]
        if isinstance(active, Mapping):
            active = _validate_operation_record(active, expected_run_id=session.run_id)
            if (
                active.get("campaign_identity_sha256") != campaign.get("campaign_identity")
                or active.get("backend_id") != session.backend.backend_id
                or active.get("activation_commit_record_identity") != activation_commit_record_identity
                or active.get("project_config_sha256") != project_config_sha256
                or active.get("launch_authorization_evidence_sha256") != launch_authorization_evidence_sha256
            ):
                raise CampaignValidationError("Durable Training operation bindings changed.")
            if active.get("status") == "RUNNING":
                if not recover_stale:
                    raise CampaignValidationError("Another durable Training operation already owns this campaign.")
                history.append(_operation_transition(active, "UNCERTAIN"))
        durable_backend = state.get("backend_identity")
        if not isinstance(durable_backend, Mapping) or (
            durable_backend.get("configuration_identity_sha256") != self._backend_configuration_identity()
        ):
            raise CampaignValidationError("The configured compute backend no longer matches durable Training state.")
        operation = _operation_record(
            action=action,
            run_id=session.run_id,
            operation_nonce=operation_nonce,
            campaign_identity=str(campaign["campaign_identity"]),
            backend_id=session.backend.backend_id,
            backend_configuration_identity=self._backend_configuration_identity(),
            activation_commit_record_identity=activation_commit_record_identity,
            project_config_sha256=project_config_sha256,
            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
            status="RUNNING",
        )
        self.repository.update_state(
            session.run_id,
            active_operation=operation,
            operation_history=history[-64:],
        )
        session.operation_nonce = operation_nonce
        session.operation_action = action
        session.launch_authorization_evidence_sha256 = launch_authorization_evidence_sha256

    def _finish_operation(self, session: TrainingSession, status: str) -> None:
        state = self.repository.state(session.run_id)
        operation = _validate_operation_record(state.get("active_operation"), expected_run_id=session.run_id)
        if (
            operation.get("operation_nonce") != session.operation_nonce
            or operation.get("action") != session.operation_action
            or operation.get("status") != "RUNNING"
        ):
            raise CampaignValidationError("Durable Training operation ownership changed before terminal commit.")
        terminal = _operation_transition(operation, status)
        history = [
            _validate_operation_record(item, expected_run_id=session.run_id)
            for item in state.get("operation_history", ())
            if isinstance(item, Mapping)
        ]
        history.append(terminal)
        self.repository.update_state(
            session.run_id,
            active_operation=terminal,
            operation_history=history[-64:],
        )

    def _claim_control_operation(self, session: TrainingSession, action: str) -> None:
        state = self.repository.state(session.run_id)
        active = _validate_operation_record(state.get("active_operation"), expected_run_id=session.run_id)
        campaign = session.plan.campaign
        if not isinstance(campaign, Mapping):
            raise CampaignValidationError("The retained campaign is unavailable.")
        self._claim_operation(
            session,
            action=action,
            operation_nonce=f"operation-{uuid.uuid4().hex}",
            campaign=campaign,
            activation_commit_record_identity=str(active["activation_commit_record_identity"]),
            project_config_sha256=str(active["project_config_sha256"]),
            launch_authorization_evidence_sha256=str(active["launch_authorization_evidence_sha256"]),
            recover_stale=True,
        )

    def _verify_job_control_identity(self, session: TrainingSession, job: ComputeJob) -> None:
        request = session.requests.get(job.job_id)
        prepared = session.prepared.get(job.job_id)
        if (
            job.backend_id != session.backend.backend_id
            or request is None
            or prepared is None
            or request.run_id != job.run_id
            or request.compute_backend_id != job.backend_id
            or prepared.backend_id != job.backend_id
            or prepared.remote_identity != job.remote_identity
        ):
            raise CampaignValidationError("The retained backend job identity is incomplete or stale.")
        state = self.repository.state(session.run_id)
        matching = []
        for row in state.get("jobs", ()):
            if not isinstance(row, Mapping):
                continue
            durable_job = _job_from_state(row.get("job"))
            if durable_job is not None and durable_job.job_id == job.job_id:
                matching.append(row)
        if len(matching) != 1:
            raise CampaignValidationError("The durable backend job identity is missing or ambiguous.")
        row = matching[0]
        if (
            _job_to_state(job) != row.get("job")
            or _request_to_state(request) != row.get("request")
            or _prepared_to_state(prepared) != row.get("prepared")
        ):
            raise CampaignValidationError("The durable backend job identity changed before a control seam.")

    @staticmethod
    def _action_conflict(action: str) -> ProductResult:
        return ProductResult(
            ProductStatus.BLOCKED,
            f"Training {action} is temporarily unavailable while another action commits.",
            feature="training",
            blockers=(ProductBlocker("training_action_conflict", "Wait for the current action, then retry."),),
        )

    def refresh(self, run_id: str) -> ProductResult:
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                return self._refresh_locked(run_id)
        except TrainingActionLockError:
            return self._action_conflict("Refresh")

    def _refresh_locked(self, run_id: str) -> ProductResult:
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            dashboard = self._dashboard_from_events(run_id)
            if dashboard is None:
                return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
            self._disable_unsafe_dashboard_actions(dashboard)
            return ProductResult(
                dashboard.status,
                "Training progress was reconstructed, but safe actions are disabled by incomplete durable state.",
                feature="training",
                data=dashboard.to_dict(),
            )
        if session.dashboard.terminal_status == "CANCELLED":
            return ProductResult(
                ProductStatus.COMPLETE,
                "Training is durably cancelled.",
                feature="training",
                data=session.dashboard.to_dict(),
            )

        polls: list[ComputePoll] = []
        poll_uncertain = False
        possible_cost = False
        shutdown_verified: list[bool] = []
        for job in session.jobs:
            try:
                self._verify_job_control_identity(session, job)
            except CampaignValidationError:
                poll = ComputePoll(
                    ComputeStatus.UNCERTAIN,
                    "Durable backend identity verification failed.",
                    may_accrue_cost=bool(job.may_accrue_cost or session.backend.is_cloud),
                    resource_state_uncertain=True,
                )
                polls.append(poll)
                poll_uncertain = True
                possible_cost |= poll.may_accrue_cost
                session.job_outcomes[job.job_id] = _job_outcome(job, status="UNCERTAIN", stage="identity")
                self._apply(
                    session,
                    ProductEvent(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="remote",
                        event_type="remote_failure",
                        status=ProductStatus.BLOCKED,
                        message="Durable backend identity verification failed before polling.",
                        metrics={
                            "resource_state_uncertain": True,
                            "may_accrue_cost": poll.may_accrue_cost,
                            "shutdown_guidance": (
                                "Open the configured backend and explicitly stop or terminate the retained resource."
                            ),
                            "job_outcomes": _sorted_outcomes(session.job_outcomes),
                        },
                    ),
                )
                continue

            cursor = session.cursors.get(job.job_id, 0)
            try:
                events, next_cursor = session.backend.stream_events(job, cursor=cursor)
                session.cursors[job.job_id] = next_cursor
                for event in events:
                    normalized = ProductEvent(
                        run_id=run_id,
                        timestamp=event.timestamp,
                        feature=event.feature,
                        stage=event.stage,
                        event_type=event.event_type,
                        status=event.status,
                        current=event.current,
                        total=event.total,
                        message=event.message,
                        metrics=event.metrics,
                        artifact_references=event.artifact_references,
                    )
                    self._apply(session, normalized)
            except (ComputeBackendError, NotImplementedError, OSError, TypeError, ValueError):
                self._apply(
                    session,
                    ProductEvent(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="events",
                        event_type="warning",
                        status=session.dashboard.status,
                        message="Backend events could not be read; terminal state polling continued.",
                    ),
                )
            try:
                poll = session.backend.poll(job)
            except (ComputeBackendError, NotImplementedError, OSError, TypeError, ValueError):
                poll = ComputePoll(
                    ComputeStatus.UNCERTAIN,
                    "Backend state could not be reached.",
                    may_accrue_cost=bool(job.may_accrue_cost or session.backend.is_cloud),
                    resource_state_uncertain=True,
                )
            polls.append(poll)
            verified_shutdown = poll.metadata.get("resource_shutdown_verified") is True
            shutdown_verified.append(verified_shutdown)
            job_uncertain = poll.resource_state_uncertain or poll.status == ComputeStatus.UNCERTAIN
            terminal_cloud_without_shutdown = (
                session.backend.is_cloud
                and poll.status
                in {
                    ComputeStatus.COMPLETE,
                    ComputeStatus.FAILED,
                    ComputeStatus.CANCELLED,
                }
                and not verified_shutdown
            )
            job_uncertain |= terminal_cloud_without_shutdown
            job_cost = (
                False
                if verified_shutdown
                else bool(session.backend.is_cloud or poll.may_accrue_cost or job.may_accrue_cost)
            )
            poll_uncertain |= job_uncertain
            possible_cost |= job_cost
            session.job_outcomes[job.job_id] = _job_outcome(
                job,
                status="UNCERTAIN" if job_uncertain else poll.status.value,
                stage="poll",
                may_accrue_cost=job_cost,
                resource_shutdown_verified=verified_shutdown,
            )
            if job_uncertain:
                self._apply(
                    session,
                    ProductEvent(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="remote",
                        event_type="remote_failure",
                        status=ProductStatus.BLOCKED,
                        message="Backend state or resource shutdown could not be verified.",
                        metrics={
                            "resource_state_uncertain": True,
                            "may_accrue_cost": job_cost,
                            "shutdown_guidance": (
                                "Open the configured backend and explicitly stop or terminate the retained resource."
                            ),
                            "job_outcomes": _sorted_outcomes(session.job_outcomes),
                        },
                    ),
                )

        status = session.dashboard.status
        completion_validated = False
        statuses = {item.status for item in polls}
        active_statuses = {
            ComputeStatus.RUNNING,
            ComputeStatus.PAUSING,
            ComputeStatus.PREPARED,
            ComputeStatus.UPLOADING,
        }
        if polls:
            if poll_uncertain:
                status = ProductStatus.BLOCKED
            elif all(item.status == ComputeStatus.COMPLETE for item in polls):
                campaign = session.plan.campaign
                completion = (
                    audit_artifact_completeness(campaign)
                    if campaign is not None
                    else {"complete": False, "reasons": ["missing campaign"]}
                )
                completion_validated = bool(completion["complete"])
                status = ProductStatus.COMPLETE if completion_validated else ProductStatus.BLOCKED
                if not completion_validated:
                    session.dashboard.warnings.append(
                        "Backend jobs stopped, but exact campaign completion artifacts are incomplete."
                    )
            elif all(item.status == ComputeStatus.PAUSED for item in polls):
                status = ProductStatus.PAUSED
            elif ComputeStatus.FAILED in statuses and statuses & active_statuses:
                status = ProductStatus.BLOCKED
            elif any(item.status == ComputeStatus.FAILED for item in polls):
                status = ProductStatus.FAILED
            elif all(item.status == ComputeStatus.CANCELLED for item in polls):
                status = ProductStatus.BLOCKED
            else:
                status = ProductStatus.RUNNING
            self._apply(
                session,
                ProductEvent(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    feature="training",
                    stage="campaign",
                    event_type="backend_state",
                    status=status,
                    current=session.dashboard.campaign_current,
                    total=session.dashboard.campaign_total,
                    message=f"Backend jobs are {status.value.lower()}.",
                    metrics={
                        "completion_validated": completion_validated,
                        "resource_state_uncertain": poll_uncertain,
                        "resource_state_verified": not poll_uncertain,
                        "may_accrue_cost": possible_cost,
                        "resource_shutdown_verified": bool(
                            session.backend.is_cloud and shutdown_verified and all(shutdown_verified)
                        ),
                        "job_outcomes": _sorted_outcomes(session.job_outcomes),
                        **(
                            {
                                "evaluation_checkpoint_binding": _evaluation_checkpoint_binding(
                                    session,
                                    self.repository.runs_directory,
                                )
                            }
                            if status is ProductStatus.COMPLETE
                            else {}
                        ),
                    },
                ),
            )
        self._persist_session(session)
        return ProductResult(
            status=session.dashboard.status,
            message="Training dashboard updated.",
            feature="training",
            data=session.dashboard.to_dict(),
        )

    def pause(self, run_id: str) -> ProductResult:
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                return self._pause_locked(run_id)
        except TrainingActionLockError:
            return self._action_conflict("Pause")

    def _pause_locked(self, run_id: str) -> ProductResult:
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        if session.dashboard.status != ProductStatus.RUNNING or not session.jobs:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Pause is unavailable because no retained backend job is currently running.",
                feature="training",
            )
        try:
            self._claim_control_operation(session, "pause")
        except CampaignValidationError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Pause could not claim the exact durable backend operation.",
                feature="training",
                blockers=(ProductBlocker("training_operation_conflict", "Refresh the run and retry."),),
            )

        unverified = 0
        paused = 0
        for job in session.jobs:
            try:
                self._verify_job_control_identity(session, job)
                result = session.backend.pause(job)
                if result.changed is not True:
                    raise ComputeBackendError("Pause was not acknowledged.")
                poll = session.backend.poll(job)
                if poll.resource_state_uncertain or poll.status == ComputeStatus.UNCERTAIN:
                    raise ComputeBackendError("Pause state is uncertain.")
                if poll.status == ComputeStatus.PAUSED:
                    paused += 1
                    session.job_outcomes[job.job_id] = _job_outcome(job, status="PAUSED", stage="pause")
                else:
                    session.job_outcomes[job.job_id] = _job_outcome(
                        job,
                        status=poll.status.value,
                        stage="pause_pending",
                    )
            except (CampaignValidationError, ComputeBackendError, NotImplementedError, OSError, ValueError):
                unverified += 1
                session.job_outcomes[job.job_id] = _job_outcome(job, status="UNCERTAIN", stage="pause")

        status = (
            ProductStatus.BLOCKED
            if unverified
            else (ProductStatus.PAUSED if paused == len(session.jobs) else ProductStatus.RUNNING)
        )
        self._apply(
            session,
            ProductEvent(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                feature="training",
                stage="campaign",
                event_type="training_pause_state",
                status=status,
                current=session.dashboard.campaign_current,
                total=session.dashboard.campaign_total,
                message=(
                    "Training pause is verified for every retained job."
                    if status == ProductStatus.PAUSED
                    else "Pause was attempted for every retained job; at least one state is not yet verified."
                ),
                metrics={
                    "pause_attempt_count": len(session.jobs),
                    "pause_unverified_count": unverified,
                    "resource_state_uncertain": bool(unverified),
                    "may_accrue_cost": bool(session.backend.is_cloud),
                    "job_outcomes": _sorted_outcomes(session.job_outcomes),
                },
            ),
        )
        self._persist_session(session)
        self._finish_operation(
            session,
            "UNCERTAIN" if unverified else ("PAUSED" if status == ProductStatus.PAUSED else "COMPLETE"),
        )
        return ProductResult(
            status,
            "Pause control finished for every retained backend job.",
            feature="training",
            warnings=(
                (ProductWarning("pause_unverified", "At least one backend pause state remains unverified."),)
                if unverified
                else ()
            ),
            data={
                "pause_attempt_count": len(session.jobs),
                "pause_unverified_count": unverified,
                "unsafe_resume_available": False,
                "job_outcomes": _sorted_outcomes(session.job_outcomes),
            },
        )

    def cancel(self, run_id: str) -> ProductResult:
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                return self._cancel_locked(run_id)
        except TrainingActionLockError:
            return self._action_conflict("Cancel")

    def _cancel_locked(self, run_id: str) -> ProductResult:
        durable = self.repository.state(run_id)
        if (
            str(durable.get("feature") or durable.get("command")) == "training"
            and str(durable.get("status") or "").upper() == "CANCELLED"
        ):
            dashboard = self._dashboard_from_events(run_id)
            return ProductResult(
                ProductStatus.COMPLETE,
                "Training is durably cancelled.",
                feature="training",
                data={"cancelled": True, "dashboard": dashboard.to_dict() if dashboard else None},
            )
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        try:
            active_operation = _validate_operation_record(durable.get("active_operation"), expected_run_id=run_id)
        except CampaignValidationError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Cancel could not verify the exact durable backend operation.",
                feature="training",
                blockers=(ProductBlocker("training_operation_conflict", "Refresh the run and retry."),),
                data={"cancelled": False},
            )
        stale_active_operation = active_operation.get("status") == "RUNNING"
        pending_stages = {"prepared", "uploaded", "possibly_launched", "cancel_uncertain", "cleanup_uncertain"}
        pending = [
            (prepared_run_id, prepared)
            for prepared_run_id, prepared in session.prepared.items()
            if session.prepared_stages.get(prepared_run_id) in pending_stages
        ]
        has_unknown_seed = any(
            item.get("status") in {"RUNNING", "UNCERTAIN"} for item in session.seed_outcomes.values()
        )
        if stale_active_operation and not session.jobs and not pending and not has_unknown_seed:
            session.unknown_backend_operation_count = max(1, session.unknown_backend_operation_count)
        cancelable = bool(session.jobs or pending) and session.dashboard.status in {
            ProductStatus.RUNNING,
            ProductStatus.PAUSED,
            ProductStatus.BLOCKED,
            ProductStatus.FAILED,
        }
        cancelable |= session.dashboard.status == ProductStatus.BLOCKED and any(
            item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values()
        )
        cancelable |= session.dashboard.status in {
            ProductStatus.RUNNING,
            ProductStatus.PAUSED,
            ProductStatus.BLOCKED,
            ProductStatus.FAILED,
        } and bool(
            session.unknown_backend_operation_count
            or any(item.get("status") in {"RUNNING", "UNCERTAIN"} for item in session.seed_outcomes.values())
        )
        if not cancelable:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Cancel is unavailable because no retained backend resource is cancelable.",
                feature="training",
            )
        try:
            self._claim_control_operation(session, "cancel")
        except CampaignValidationError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Cancel could not claim the exact durable backend operation.",
                feature="training",
                blockers=(ProductBlocker("training_operation_conflict", "Refresh the run and retry."),),
                data={"cancelled": False},
            )

        for seed_key, outcome in tuple(session.seed_outcomes.items()):
            if outcome.get("status") == "RUNNING":
                session.seed_outcomes[seed_key] = {**outcome, "status": "UNCERTAIN", "stage": "cancel"}

        cancel_unverified = 0
        cleanup_unverified = 0
        may_accrue_cost = False
        for job in session.jobs:
            verified = False
            shutdown = not session.backend.is_cloud
            try:
                self._verify_job_control_identity(session, job)
                result = session.backend.cancel(job)
                if result.changed is not True:
                    raise ComputeBackendError("Cancellation was not acknowledged.")
                for attempt in range(3):
                    poll = session.backend.poll(job)
                    shutdown = poll.metadata.get("resource_shutdown_verified") is True or not session.backend.is_cloud
                    if poll.status == ComputeStatus.CANCELLED and not poll.resource_state_uncertain and shutdown:
                        verified = True
                        break
                    if attempt < 2:
                        time.sleep(0.05)
            except (CampaignValidationError, ComputeBackendError, NotImplementedError, OSError, ValueError):
                verified = False
            if verified:
                session.prepared_stages[job.job_id] = "cancelled"
                session.job_outcomes[job.job_id] = _job_outcome(
                    job,
                    status="CANCELLED",
                    stage="cancel",
                    resource_shutdown_verified=shutdown,
                )
                if job.run_id in session.seed_outcomes:
                    session.seed_outcomes[job.run_id] = {
                        **session.seed_outcomes[job.run_id],
                        "status": "CANCELLED",
                        "stage": "cancelled",
                    }
            else:
                cancel_unverified += 1
                may_accrue_cost |= bool(job.may_accrue_cost or session.backend.is_cloud)
                session.job_outcomes[job.job_id] = _job_outcome(
                    job,
                    status="UNCERTAIN",
                    stage="cancel",
                    resource_shutdown_verified=False,
                )
                if job.run_id in session.seed_outcomes:
                    session.seed_outcomes[job.run_id] = {
                        **session.seed_outcomes[job.run_id],
                        "status": "UNCERTAIN",
                        "stage": "cancel",
                    }

        for prepared_run_id, prepared in pending:
            verified = False
            prepared_stage = session.prepared_stages.get(prepared_run_id)
            try:
                result = session.backend.cleanup(prepared)
                shutdown = result.metadata.get("resource_shutdown_verified") is True or not session.backend.is_cloud
                verified = result.changed is True and shutdown and prepared_stage != "possibly_launched"
            except (ComputeBackendError, NotImplementedError, OSError, ValueError):
                verified = False
            if verified:
                session.prepared_stages[prepared_run_id] = "cleaned"
            else:
                cleanup_unverified += 1
                session.prepared_stages[prepared_run_id] = "cleanup_uncertain"
                may_accrue_cost |= bool(session.backend.is_cloud)

        unknown_unverified = max(
            session.unknown_backend_operation_count,
            sum(1 for item in session.seed_outcomes.values() if item.get("status") == "UNCERTAIN"),
        )
        session.unknown_backend_operation_count = unknown_unverified
        may_accrue_cost |= bool(unknown_unverified and session.backend.is_cloud)

        attempt_metrics = {
            "launched_job_count": len(session.jobs),
            "cancel_attempt_count": len(session.jobs),
            "cancel_unverified_count": cancel_unverified,
            "prepared_cleanup_attempt_count": len(pending),
            "prepared_cleanup_unverified_count": cleanup_unverified,
            "unknown_backend_operation_count": unknown_unverified,
            "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
            "job_outcomes": _sorted_outcomes(session.job_outcomes),
        }
        if cancel_unverified or cleanup_unverified or unknown_unverified:
            guidance = "Open the configured backend and explicitly stop or terminate every retained resource."
            self._apply(
                session,
                ProductEvent(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    feature="training",
                    stage="remote",
                    event_type="remote_failure",
                    status=ProductStatus.BLOCKED,
                    current=session.dashboard.campaign_current,
                    total=session.dashboard.campaign_total,
                    message="Backend shutdown could not be verified for every retained resource.",
                    metrics={
                        **attempt_metrics,
                        "resource_state_uncertain": True,
                        "may_accrue_cost": may_accrue_cost,
                        "resource_shutdown_verified": False,
                        "shutdown_guidance": guidance,
                    },
                ),
            )
            self._persist_session(session)
            self._finish_operation(session, "UNCERTAIN")
            return ProductResult(
                ProductStatus.BLOCKED,
                "Backend shutdown could not be verified for every retained resource.",
                feature="training",
                blockers=(ProductBlocker("backend_cleanup", guidance),),
                warnings=(ProductWarning("cancel_unverified", "Cancellation remains unverified and retryable."),),
                data={
                    **attempt_metrics,
                    "cancelled": False,
                    "resource_state_uncertain": True,
                    "may_accrue_cost": may_accrue_cost,
                    "cancel_available": True,
                },
            )

        event = ProductEvent(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            feature="training",
            stage="campaign",
            event_type="training_cancelled",
            status=ProductStatus.COMPLETE,
            current=session.dashboard.campaign_current,
            total=session.dashboard.campaign_total,
            message="Training cancellation and backend shutdown are verified.",
            metrics={
                **attempt_metrics,
                "terminal_status": "CANCELLED",
                "resource_state_uncertain": False,
                "may_accrue_cost": False,
                "resource_shutdown_verified": True,
            },
        )
        self._apply(session, event)
        session.unknown_backend_operation_count = 0
        self._persist_session(session)
        self._finish_operation(session, "CANCELLED")
        self.repository.update_state(run_id, status="CANCELLED", ended_at=event.timestamp, resumable=False)
        return ProductResult(
            ProductStatus.COMPLETE,
            "Training was cancelled and shutdown was verified.",
            feature="training",
            data={**attempt_metrics, "cancelled": True, "terminal_status": "CANCELLED"},
        )

    def resume(
        self,
        run_id: str,
        *,
        cloud_confirmation: bool = False,
        cloud_challenge: str | None = None,
    ) -> ProductResult:
        try:
            with TrainingActionLock(self.context.project_root, timeout_seconds=0.0):
                return self._resume_locked(
                    run_id,
                    cloud_confirmation=cloud_confirmation,
                    cloud_challenge=cloud_challenge,
                )
        except TrainingActionLockError:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Training Resume is temporarily unavailable while activation or another launch action commits.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "training_action_conflict",
                        "Wait for the current activation, Start, or Resume action to finish, then retry.",
                    ),
                ),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )

    def _resume_locked(
        self,
        run_id: str,
        *,
        cloud_confirmation: bool,
        cloud_challenge: str | None,
    ) -> ProductResult:
        del cloud_confirmation
        had_cached_session = run_id in self.sessions
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            if had_cached_session:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Safe resume was refused because the cached run no longer matches durable state.",
                    feature="training",
                    blockers=(
                        ProductBlocker(
                            "durable_training_state",
                            "Durable training identity or event evidence is missing, malformed, or stale.",
                        ),
                    ),
                    data={"backend_launches": 0, "unsafe_resume_available": False},
                )
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        if session.plan.profile is not TrainingProfile.RECOMMENDED:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Production conditioned Resume accepts only the recommended profile.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "conditioned_profile_ineligible",
                        "Quality, fast-preview, and custom retained campaigns are not production-authorized.",
                    ),
                ),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        if not session.dashboard.resume_available:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume is unavailable.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "safe_resume",
                        "No downloaded, hash-verified checkpoint with a verified backend identity is available.",
                    ),
                ),
                data={"unsafe_resume_available": False, "backend_launches": 0},
            )
        try:
            self._verify_session_migrations(session)
        except CampaignValidationError as exc:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume event evidence is not comparable.",
                feature="training",
                blockers=(ProductBlocker("event_migration", str(exc)),),
                data={"unsafe_resume_available": False, "backend_launches": 0},
            )
        if session.backend.is_cloud and not cloud_challenge:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Cloud Resume requires a fresh one-use server challenge.",
                feature="training",
                blockers=(ProductBlocker("cloud_challenge_required", "Confirm hosted cost before Resume."),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        try:
            activation = self.activation_loader(
                self.context,
                session.plan.profile,
                expected_campaign=session.plan.campaign,
                require_audit=True,
            )
        except (ConditionedActivationError, OSError, TypeError, KeyError) as exc:
            code = exc.code if isinstance(exc, ConditionedActivationError) else "conditioned_activation_invalid"
            message = (
                exc.public_message
                if isinstance(exc, ConditionedActivationError)
                else "The exact conditioned training activation could not be verified safely."
            )
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume was refused because the retained conditioned activation is no longer applicable.",
                feature="training",
                blockers=(ProductBlocker(code, message),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        try:
            activation_record_identity, project_config_sha256 = _committed_activation_binding(
                activation,
                session.plan.campaign or {},
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe Resume requires the exact committed activation receipt.",
                feature="training",
                blockers=(ProductBlocker("activation_commit_required", str(exc)),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        custom_spec = activation.selected_spec if session.plan.profile is TrainingProfile.CUSTOM else None
        plan = self.plan(session.plan.profile, custom_spec=custom_spec, before_launch=True)
        if plan.blockers or plan.campaign is None:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume gates are no longer satisfied.",
                feature="training",
                blockers=tuple(ProductBlocker(item.gate_id, item.message, item.resolution) for item in plan.blockers),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        if plan.campaign.get("campaign_identity") != activation.campaign.get("campaign_identity"):
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume was refused because the selected campaign changed.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "selected_campaign_changed",
                        "The retained run, current profile, and conditioned activation do not bind the same campaign.",
                    ),
                ),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        campaign = plan.campaign
        config = project_config_from_context(self.context)
        prior_job_count = len(session.jobs)
        try:
            with self.audit_snapshot_opener(config, None, activation) as audit_snapshot:
                launch_authorization_evidence_sha256 = _require_applicable_training_audit_snapshot(audit_snapshot)
                return self._resume_authorized_locked(
                    run_id,
                    session=session,
                    cloud_challenge=cloud_challenge,
                    campaign=campaign,
                    activation_record_identity=activation_record_identity,
                    project_config_sha256=project_config_sha256,
                    config=config,
                    audit_snapshot=audit_snapshot,
                    launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                )
        except (OSError, RuntimeError, TypeError, ValueError):
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe Resume was refused because its retained infrastructure-audit evidence changed.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "training_audit_snapshot_stale",
                        "The exact report, hash inventory, receipt, or action record changed during authorization.",
                    ),
                ),
                data={
                    "backend_launches": max(0, len(session.jobs) - prior_job_count),
                    "unsafe_resume_available": False,
                    "cancel_available": bool(session.jobs or session.prepared),
                },
            )

    def _resume_authorized_locked(
        self,
        run_id: str,
        *,
        session: TrainingSession,
        cloud_challenge: str | None,
        campaign: Mapping[str, Any],
        activation_record_identity: str,
        project_config_sha256: str,
        config: Any,
        audit_snapshot: Any,
        launch_authorization_evidence_sha256: str,
    ) -> ProductResult:
        campaign_config_path = config.path_for("training", "campaign_config")
        if campaign_config_path is None or not campaign_config_path.is_file():
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume requires the exact authoritative campaign configuration.",
                feature="training",
                blockers=(ProductBlocker("campaign_configuration", "training.campaign_config is required."),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        operation_nonce = f"operation-{uuid.uuid4().hex}"
        challenge_bindings = self._cloud_action_bindings(
            action="resume",
            run_id=run_id,
            campaign=campaign,
            activation_record_identity=activation_record_identity,
            project_config_sha256=project_config_sha256,
            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
        )
        cloud_authorized = False
        if session.backend.is_cloud:
            try:
                audit_snapshot.verify_unchanged()
                self.cloud_challenges.consume_locked(
                    str(cloud_challenge or ""),
                    expected_bindings=challenge_bindings,
                    operation_nonce=operation_nonce,
                )
            except CloudChallengeError as exc:
                return ProductResult(
                    ProductStatus.BLOCKED,
                    "Cloud Resume was refused because its one-use confirmation is invalid.",
                    feature="training",
                    blockers=(ProductBlocker(exc.code, exc.public_message),),
                    data={"backend_launches": 0, "unsafe_resume_available": False},
                )
            cloud_authorized = True
        try:
            audit_snapshot.verify_unchanged()
            self._claim_operation(
                session,
                action="resume",
                operation_nonce=operation_nonce,
                campaign=campaign,
                activation_commit_record_identity=activation_record_identity,
                project_config_sha256=project_config_sha256,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
            )
        except CampaignValidationError as exc:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe Resume could not claim durable launch ownership.",
                feature="training",
                blockers=(ProductBlocker("training_operation_conflict", str(exc)),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        session.seed_outcomes = {}
        launched: list[ComputeJob] = []

        def runner(command: list[str], **kwargs: Any) -> Any:
            validated = kwargs.get("validated_launch")
            if not isinstance(validated, ValidatedTrainingLaunch):
                raise CampaignValidationError("Resume runner requires a validator-issued continuation receipt.")
            _require_validated_launch_authorization(validated, launch_authorization_evidence_sha256)
            audit_snapshot.verify_unchanged()
            self._bind_training_identity(session, validated)
            self._verify_session_migrations(session)
            run = validated.run
            seed = int(run["seed"])
            seed_key = str(run["run_id"])
            session.seed_outcomes[seed_key] = _seed_outcome(run, status="RUNNING", stage="checkpoint")
            self._persist_session(session)
            stage = "checkpoint"
            try:
                checkpoint_state = max(
                    (item for item in session.dashboard.checkpoints if item.seed == seed and item.safe_resume),
                    key=lambda item: item.optimizer_step,
                    default=None,
                )
                if checkpoint_state is None:
                    raise CampaignValidationError(f"Seed {seed} has no product-verified safe checkpoint.")
                previous = _request_for_run(session, str(run["run_id"]))
                if previous is None:
                    raise CampaignValidationError(f"Seed {seed} has no retained launch request.")
            except (CampaignValidationError, KeyError, ValueError):
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="BLOCKED", stage=stage)
                self._persist_session(session)
                raise
            migration = verify_event_migration(
                str(run["run_id"]),
                validated.output_root,
                expected_evidence_sha256=validated.receipt.event_migration_identity_sha256,
                migration_required=(
                    validated.receipt.event_migration_required
                    or validated.receipt.event_migration_state != EventMigrationState.NO_MIGRATION.value
                ),
            )
            if not migration.resume_compatible or migration.state.value != validated.receipt.event_migration_state:
                raise CampaignValidationError(
                    "Continuation event migration evidence changed or is invalid: "
                    f"{migration.state.value}: {migration.message}"
                )
            if (
                migration.event_history_origin != validated.receipt.event_history_origin
                or migration.migration_required != validated.receipt.event_migration_required
                or migration.migration_record_sha256 != validated.receipt.event_migration_record_sha256
                or migration.canonical_prefix_sha256 != validated.receipt.event_canonical_prefix_sha256
                or migration.canonical_event_identity_sha256 != validated.receipt.event_canonical_identity_sha256
            ):
                raise CampaignValidationError(
                    "Continuation event-history origin bindings changed after receipt validation."
                )
            if checkpoint_state.sha256 != validated.receipt.source_checkpoint_identity:
                raise CampaignValidationError("Product checkpoint identity does not match the continuation receipt.")
            request = ComputeJobRequest(
                run_id=previous.run_id,
                command=tuple(command),
                idempotency_key=f"{previous.run_id[:150]}-resume-{checkpoint_state.optimizer_step}",
                campaign_identity=validated.receipt.campaign_identity_sha256,
                run_identity=str(run["run_identity"]),
                local_project_root=previous.local_project_root,
                output_root=validated.output_root,
                event_path=previous.event_path,
                environment=validated.environment,
                execution_spec_identity=validated.receipt.execution_spec_sha256,
                output_root_identity=validated.receipt.output_root_identity,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                compute_backend_id=session.backend.backend_id,
                launch_receipt=validated.receipt,
                validator_context=validated.validator_context,
                launch_authorization_verifier=audit_snapshot,
            )
            audit_snapshot.verify_unchanged()
            stage = "prepare"
            try:
                prepared = self._idempotent_backend_call(
                    session,
                    lambda: _verify_audit_then(
                        audit_snapshot,
                        lambda: session.backend.prepare(self.context, request),
                    ),
                )
                session.prepared[request.idempotency_key] = prepared
                session.prepared_stages[request.idempotency_key] = "prepared"
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="RUNNING", stage="resume")
                self._persist_session(session)
                artifact = ArtifactReference(
                    checkpoint_state.checkpoint,
                    checkpoint_state.sha256 or "",
                    prepared.remote_identity,
                    Path(checkpoint_state.checkpoint),
                    downloaded=checkpoint_state.downloaded,
                    hash_verified=checkpoint_state.hash_verified,
                    remote_identity_verified=checkpoint_state.remote_identity_verified,
                )
                stage = "resume"
                session.prepared_stages[request.idempotency_key] = "possibly_launched"
                self._persist_session(session)
                job = self._idempotent_backend_call(
                    session,
                    lambda: _verify_audit_then(
                        audit_snapshot,
                        lambda: session.backend.resume(
                            prepared,
                            ResumeRequest(request, artifact, safe_resume=True),
                            cloud_confirmation=cloud_authorized,
                        ),
                    ),
                )
                session.requests[job.job_id] = request
                session.prepared.pop(request.idempotency_key, None)
                session.prepared_stages.pop(request.idempotency_key, None)
                session.prepared[job.job_id] = prepared
                session.prepared_stages[job.job_id] = "launched"
                session.jobs.append(job)
                session.seed_outcomes[seed_key] = _seed_outcome(run, status="LAUNCHED", stage="resumed")
                session.job_outcomes[job.job_id] = _job_outcome(job, status="LAUNCHED")
                self._persist_session(session)
            except (CampaignValidationError, ComputeBackendError, OSError, ValueError):
                uncertain = stage in {"prepare", "resume"}
                if stage == "resume" and request.idempotency_key in session.prepared:
                    session.prepared_stages[request.idempotency_key] = "possibly_launched"
                session.seed_outcomes[seed_key] = _seed_outcome(
                    run,
                    status="UNCERTAIN" if uncertain else "BLOCKED",
                    stage=stage,
                )
                self._persist_session(session)
                raise
            launched.append(job)
            return SimpleNamespace(returncode=0)

        try:
            audit_snapshot.verify_unchanged()
            execution = execute_campaign(
                campaign,
                execute=True,
                confirm_execute=True,
                campaign_config_path=campaign_config_path,
                campaign_profile=session.plan.profile.value,
                compute_backend_id=session.backend.backend_id,
                execution_environment=self._execution_environment(),
                project_root=self.context.project_root,
                resume=True,
                unsafe_resume=False,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                runner=runner,
            )
        except (CampaignValidationError, ComputeBackendError, OSError, ValueError) as exc:
            del exc
            _block_incomplete_seed_outcomes(session)
            _complete_seed_outcomes(session, campaign)
            uncertain = any(item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values())
            self._apply(
                session,
                ProductEvent(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    feature="training",
                    stage="campaign",
                    event_type="training_blocked",
                    status=ProductStatus.BLOCKED,
                    message="The authoritative safe-resume contract refused continuation.",
                    metrics={
                        "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
                        "resource_state_uncertain": uncertain,
                        "may_accrue_cost": bool(uncertain and session.backend.is_cloud),
                        "shutdown_guidance": (
                            "Open the configured backend and stop any resource associated with this campaign."
                            if uncertain
                            else None
                        ),
                    },
                ),
            )
            self._persist_session(session)
            self._finish_operation(session, "UNCERTAIN" if uncertain else "BLOCKED")
            return ProductResult(
                ProductStatus.BLOCKED,
                "The authoritative safe-resume contract refused continuation.",
                feature="training",
                blockers=(
                    ProductBlocker(
                        "safe_resume",
                        "A validated resume seam failed or could not be reconciled safely.",
                    ),
                ),
                data={
                    "backend_launches": len(launched),
                    "unsafe_resume_available": False,
                    "run_id": run_id,
                    "dashboard": session.dashboard.to_dict(),
                    "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
                    "job_outcomes": _sorted_outcomes(session.job_outcomes),
                    "cancel_available": bool(
                        session.jobs
                        or session.prepared
                        or any(item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values())
                    ),
                },
            )
        resumed_event = ProductEvent(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            feature="training",
            stage="campaign",
            event_type="training_resumed",
            status=ProductStatus.RUNNING,
            current=session.dashboard.campaign_current,
            total=session.dashboard.campaign_total,
            message="Safe resume launched after backend identity revalidation.",
            metrics={"seed_outcomes": _sorted_outcomes(session.seed_outcomes)},
        )
        self._apply(session, resumed_event)
        _complete_seed_outcomes(session, campaign)
        self._persist_session(session)
        self._finish_operation(session, "LAUNCHED")
        return ProductResult(
            ProductStatus.RUNNING,
            "Safe resume launched through the validated campaign and backend contracts.",
            feature="training",
            data={
                "execution": execution,
                "unsafe_resume_available": False,
                "seed_outcomes": _sorted_outcomes(session.seed_outcomes),
            },
        )

    def dashboard(self, run_id: str) -> ProductResult:
        dashboard = self._dashboard_from_events(run_id)
        if dashboard is None:
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        if not self._dashboard_identity_is_valid(run_id, dashboard):
            self._disable_unsafe_dashboard_actions(dashboard)
        return ProductResult(
            dashboard.status,
            "Training progress reconstructed from durable state.",
            feature="training",
            data=dashboard.to_dict(),
        )

    @staticmethod
    def _disable_unsafe_dashboard_actions(dashboard: DashboardState) -> None:
        dashboard.pause_available = False
        dashboard.resume_available = False
        warning = "Safe training actions are disabled because the durable run identity is incomplete or invalid."
        if warning not in dashboard.warnings:
            dashboard.warnings.append(warning)

    def _dashboard_identity_is_valid(self, run_id: str, dashboard: DashboardState) -> bool:
        state = self.repository.state(run_id)
        if str(state.get("feature") or state.get("command")) != "training":
            return False
        if not self.repository.replay(run_id).safe_for_resume:
            return False
        try:
            durable_backend = state.get("backend_identity")
            if (
                not isinstance(durable_backend, Mapping)
                or durable_backend.get("backend_id") != self.backend.backend_id
                or durable_backend.get("configuration_identity_sha256") != self._backend_configuration_identity()
            ):
                return False
            plan = _plan_from_state(state.get("plan"))
            if plan is None:
                return False
            dataset_identity = _durable_identity(state, "dataset_identity")
            view_identity = _durable_identity(
                state,
                "view_identity",
                "training_view_identity",
                "dataset_view_manifest_hash",
            )
            identity_required = bool(
                state.get("resumable") or dashboard.resume_available or dataset_identity or view_identity
            )
            if not identity_required:
                return True
            if not dataset_identity or not view_identity:
                return False
            expected_dataset, expected_view = _plan_training_identities(plan)
            return dataset_identity == expected_dataset and view_identity == expected_view
        except (KeyError, TypeError, ValueError, OSError):
            return False

    def latest_run_id(self) -> str | None:
        run_ids = self.repository.recent_run_ids(feature="training", limit=1)
        return run_ids[0] if run_ids else None

    def _apply(self, session: TrainingSession, event: ProductEvent) -> None:
        if session.dashboard is None:
            raise ValueError("Training session dashboard is unavailable.")
        session.dashboard.apply(event)
        session.dashboard.event_cursor = self.repository.append(event)

    def _dashboard_from_events(self, run_id: str) -> DashboardState | None:
        state = self.repository.state(run_id)
        if str(state.get("feature") or state.get("command")) != "training":
            return None
        backend_id = str(state.get("backend_id") or "unknown")
        dashboard = DashboardState(run_id, backend_id)
        replay = self.repository.replay(run_id)
        for indexed in replay.events:
            try:
                dashboard.apply(indexed.event)
                dashboard.event_cursor = indexed.event_id
            except (TypeError, ValueError):
                continue
        dashboard.warnings.extend(replay.warnings)
        if replay.integrity_status != "VALID":
            dashboard.status = ProductStatus.BLOCKED
        if not dashboard.status or dashboard.status == ProductStatus.NOT_STARTED:
            try:
                dashboard.status = ProductStatus(str(state.get("status") or ProductStatus.NOT_STARTED.value))
            except ValueError:
                dashboard.status = ProductStatus.FAILED
        self._restore_cancel_availability(dashboard, state)
        return dashboard

    @staticmethod
    def _restore_cancel_availability(dashboard: DashboardState, state: Mapping[str, Any]) -> None:
        if str(state.get("status") or "").upper() == "CANCELLED":
            dashboard.terminal_status = "CANCELLED"
            dashboard.cancel_available = False
        dashboard.seed_outcomes = {
            str(key): dict(value)
            for key, value in dict(state.get("seed_outcomes") or {}).items()
            if isinstance(value, Mapping)
        }
        dashboard.job_outcomes = {
            str(key): dict(value)
            for key, value in dict(state.get("job_outcomes") or {}).items()
            if isinstance(value, Mapping)
        }
        if dashboard.terminal_status == "CANCELLED":
            return
        active = dashboard.status in {ProductStatus.RUNNING, ProductStatus.PAUSED}
        jobs = state.get("jobs") if isinstance(state.get("jobs"), (list, tuple)) else ()
        prepared = state.get("prepared_resources") if isinstance(state.get("prepared_resources"), (list, tuple)) else ()
        pending = any(
            isinstance(row, Mapping)
            and row.get("stage")
            in {"prepared", "uploaded", "possibly_launched", "cancel_uncertain", "cleanup_uncertain"}
            for row in prepared
        )
        uncertain = any(
            isinstance(value, Mapping) and value.get("status") == "UNCERTAIN"
            for value in dict(state.get("seed_outcomes") or {}).values()
        )
        unknown_count = state.get("unknown_backend_operation_count", 0)
        if type(unknown_count) is not int or unknown_count < 0:
            unknown_count = 0
        dashboard.unknown_backend_operation_count = unknown_count
        retained_resource = bool(jobs) or pending or uncertain or bool(unknown_count)
        dashboard.cancel_available = active or (
            dashboard.status in {ProductStatus.BLOCKED, ProductStatus.FAILED} and retained_resource
        )

    def _session(self, run_id: str) -> TrainingSession | None:
        existing = self.sessions.get(run_id)
        state = self.repository.state(run_id)
        if str(state.get("feature") or state.get("command")) != "training":
            self.sessions.pop(run_id, None)
            return None
        if str(state.get("backend_id") or "") != self.backend.backend_id:
            self.sessions.pop(run_id, None)
            return None
        durable_backend = state.get("backend_identity")
        if (
            not isinstance(durable_backend, Mapping)
            or durable_backend.get("backend_id") != self.backend.backend_id
            or durable_backend.get("configuration_identity_sha256") != self._backend_configuration_identity()
        ):
            self.sessions.pop(run_id, None)
            return None
        if not self.repository.replay(run_id).safe_for_resume:
            self.sessions.pop(run_id, None)
            return None
        try:
            plan = _plan_from_state(state.get("plan"))
            if plan is None:
                return None
            active_operation = state.get("active_operation")
            validated_operation = (
                _validate_operation_record(active_operation, expected_run_id=run_id)
                if isinstance(active_operation, Mapping)
                else None
            )
            launch_authorization_evidence_sha256 = (
                str(validated_operation["launch_authorization_evidence_sha256"])
                if validated_operation is not None
                else None
            )
            dashboard = existing.dashboard if existing is not None else self._dashboard_from_events(run_id)
            if dashboard is None:
                return None
            dataset_identity = _durable_identity(state, "dataset_identity")
            view_identity = _durable_identity(
                state,
                "view_identity",
                "training_view_identity",
                "dataset_view_manifest_hash",
            )
            identity_required = bool(
                state.get("resumable")
                or dashboard.resume_available
                or dataset_identity
                or view_identity
                or (existing is not None and (existing.dataset_identity or existing.view_identity))
            )
            if identity_required:
                if not dataset_identity or not view_identity:
                    raise ValueError("Durable training dataset and view identities are both required.")
                expected_dataset, expected_view = _plan_training_identities(plan)
                if dataset_identity != expected_dataset or view_identity != expected_view:
                    raise ValueError("Durable training identities disagree with the retained campaign plan.")
            if existing is not None:
                if existing.dataset_identity not in (None, dataset_identity):
                    raise ValueError("Cached and durable training dataset identities disagree.")
                if existing.view_identity not in (None, view_identity):
                    raise ValueError("Cached and durable training-view identities disagree.")
                if existing.launch_authorization_evidence_sha256 not in (
                    None,
                    launch_authorization_evidence_sha256,
                ):
                    raise ValueError("Cached and durable launch-authorization identities disagree.")
                existing.dataset_identity = dataset_identity
                existing.view_identity = view_identity
                existing.launch_authorization_evidence_sha256 = launch_authorization_evidence_sha256
                existing.seed_outcomes = {
                    str(key): dict(value)
                    for key, value in dict(state.get("seed_outcomes") or {}).items()
                    if isinstance(value, Mapping)
                }
                existing.job_outcomes = {
                    str(key): dict(value)
                    for key, value in dict(state.get("job_outcomes") or {}).items()
                    if isinstance(value, Mapping)
                }
                existing.unknown_backend_operation_count = _nonnegative_int(
                    state.get("unknown_backend_operation_count", 0),
                    label="unknown backend operation count",
                )
                existing.operation_nonce = (
                    str(validated_operation["operation_nonce"]) if validated_operation is not None else None
                )
                existing.operation_action = (
                    str(validated_operation["action"]) if validated_operation is not None else None
                )
                self._restore_cancel_availability(existing.dashboard, state)
                return existing
            session = TrainingSession(
                run_id,
                self.backend,
                plan,
                dashboard=dashboard,
                cursors={str(key): int(value) for key, value in dict(state.get("cursors") or {}).items()},
                dataset_identity=dataset_identity,
                view_identity=view_identity,
                launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
                operation_nonce=(
                    str(validated_operation["operation_nonce"]) if validated_operation is not None else None
                ),
                operation_action=(str(validated_operation["action"]) if validated_operation is not None else None),
                seed_outcomes={
                    str(key): dict(value)
                    for key, value in dict(state.get("seed_outcomes") or {}).items()
                    if isinstance(value, Mapping)
                },
                job_outcomes={
                    str(key): dict(value)
                    for key, value in dict(state.get("job_outcomes") or {}).items()
                    if isinstance(value, Mapping)
                },
                unknown_backend_operation_count=_nonnegative_int(
                    state.get("unknown_backend_operation_count", 0),
                    label="unknown backend operation count",
                ),
                reconstructed=True,
            )
            for row in state.get("jobs", ()):
                if not isinstance(row, Mapping):
                    continue
                job = _job_from_state(row.get("job"))
                prepared = _prepared_from_state(row.get("prepared"))
                request = _request_from_state(row.get("request"))
                if job is None or prepared is None or request is None:
                    continue
                migration = verify_event_migration(request.run_id, request.output_root)
                if not migration.resume_compatible:
                    return None
                session.jobs.append(job)
                session.prepared[job.job_id] = prepared
                session.requests[job.job_id] = request
                session.prepared_stages[job.job_id] = "launched"
            for row in state.get("prepared_resources", ()):
                if not isinstance(row, Mapping):
                    raise ValueError("Durable prepared-resource state is malformed.")
                prepared_run_id = validate_identifier(str(row["run_id"]), label="prepared run id")
                stage = str(row["stage"])
                if stage not in {
                    "prepared",
                    "uploaded",
                    "possibly_launched",
                    "cancel_uncertain",
                    "cleanup_uncertain",
                    "launched",
                    "cleaned",
                    "cancelled",
                }:
                    raise ValueError("Durable prepared-resource stage is invalid.")
                prepared = _prepared_reference_from_state(row.get("reference"), session.backend)
                retained = session.prepared.get(prepared_run_id)
                if retained is not None and (
                    retained.backend_id != prepared.backend_id
                    or retained.operation_id != prepared.operation_id
                    or retained.remote_identity != prepared.remote_identity
                ):
                    raise ValueError("Durable prepared-resource references disagree.")
                session.prepared[prepared_run_id] = retained or prepared
                session.prepared_stages[prepared_run_id] = stage
        except (KeyError, TypeError, ValueError, OSError):
            self.sessions.pop(run_id, None)
            return None
        self.sessions[run_id] = session
        return session

    def _verify_session_migrations(self, session: TrainingSession) -> None:
        """Revalidate every retained per-run event stream before resume planning."""

        replay = self.repository.replay(session.run_id)
        if not replay.safe_for_resume:
            detail = replay.warnings[0] if replay.warnings else "durable training events are not comparable"
            raise CampaignValidationError(f"{session.run_id}: {replay.migration_state}: {detail}")
        for request in session.requests.values():
            migration = verify_event_migration(request.run_id, request.output_root)
            if not migration.resume_compatible:
                raise CampaignValidationError(f"{request.run_id}: {migration.state.value}: {migration.message}")

    def _persist_session(self, session: TrainingSession) -> None:
        rows = []
        for job in session.jobs:
            request = session.requests.get(job.job_id)
            prepared = session.prepared.get(job.job_id)
            if request is None or prepared is None:
                continue
            rows.append(
                {
                    "job": _job_to_state(job),
                    "prepared": _prepared_to_state(prepared),
                    "request": _request_to_state(request),
                }
            )
        dashboard = session.dashboard
        if dashboard is not None:
            dashboard.seed_outcomes = {key: dict(value) for key, value in session.seed_outcomes.items()}
            dashboard.job_outcomes = {key: dict(value) for key, value in session.job_outcomes.items()}
            pending = any(
                stage in {"prepared", "uploaded", "possibly_launched", "cancel_uncertain", "cleanup_uncertain"}
                for stage in session.prepared_stages.values()
            )
            uncertain = any(item.get("status") == "UNCERTAIN" for item in session.seed_outcomes.values())
            retained_resource = bool(session.jobs or pending or uncertain or session.unknown_backend_operation_count)
            dashboard.cancel_available = dashboard.status in {
                ProductStatus.RUNNING,
                ProductStatus.PAUSED,
            } or (dashboard.status in {ProductStatus.BLOCKED, ProductStatus.FAILED} and retained_resource)
        backend_identity: dict[str, Any] = {
            "backend_id": session.backend.backend_id,
            "configuration_identity_sha256": self._backend_configuration_identity(),
            "remote_identities": sorted({item.remote_identity for item in session.prepared.values()}),
        }
        updates: dict[str, Any] = {
            "jobs": rows,
            "prepared_resources": [
                {
                    "run_id": validate_identifier(run_id, label="prepared run id"),
                    "stage": session.prepared_stages.get(run_id, "launched"),
                    "reference": _prepared_reference_to_state(prepared),
                }
                for run_id, prepared in sorted(session.prepared.items())
            ],
            "cursors": dict(session.cursors),
            "seed_outcomes": {key: dict(value) for key, value in sorted(session.seed_outcomes.items())},
            "job_outcomes": {key: dict(value) for key, value in sorted(session.job_outcomes.items())},
            "unknown_backend_operation_count": session.unknown_backend_operation_count,
            "status": dashboard.status.value if dashboard else ProductStatus.NOT_STARTED.value,
            "resumable": bool(dashboard and dashboard.resume_available),
        }
        checkpoint_rows: list[dict[str, Any]] = []
        for checkpoint in dashboard.checkpoints if dashboard else ():
            row = asdict(checkpoint)
            if session.dataset_identity:
                row["dataset_identity"] = session.dataset_identity
            if session.view_identity:
                row["view_identity"] = session.view_identity
                row["training_view_identity"] = session.view_identity
            checkpoint_rows.append(row)
        updates["checkpoints"] = checkpoint_rows
        if session.dataset_identity:
            updates["dataset_identity"] = session.dataset_identity
            backend_identity["dataset_identity"] = session.dataset_identity
        if session.view_identity:
            updates["view_identity"] = session.view_identity
            updates["training_view_identity"] = session.view_identity
            backend_identity["view_identity"] = session.view_identity
            backend_identity["training_view_identity"] = session.view_identity
            backend_identity["dataset_view_manifest_hash"] = session.view_identity
        updates["backend_identity"] = backend_identity
        self.repository.update_state(
            session.run_id,
            **updates,
        )

    @staticmethod
    def _bind_training_identity(session: TrainingSession, validated: ValidatedTrainingLaunch) -> None:
        dataset_identity = validated.receipt.dataset_identity
        view_identity = validated.receipt.view_identity
        if session.dataset_identity not in (None, dataset_identity):
            raise CampaignValidationError("Campaign launch receipts disagree on the training dataset identity.")
        if session.view_identity not in (None, view_identity):
            raise CampaignValidationError("Campaign launch receipts disagree on the training-view identity.")
        session.dataset_identity = dataset_identity
        session.view_identity = view_identity

    @staticmethod
    def _training_identity_projection(session: TrainingSession) -> dict[str, str]:
        if not session.dataset_identity or not session.view_identity:
            raise CampaignValidationError("Validated training identity was not retained in product run state.")
        return {
            "dataset_identity": session.dataset_identity,
            "view_identity": session.view_identity,
            "training_view_identity": session.view_identity,
        }

    def _execution_environment(self) -> dict[str, str]:
        compute = self.context.config.get("compute", {}) if isinstance(self.context.config, Mapping) else {}
        training = compute.get("training", {}) if isinstance(compute, Mapping) else {}
        try:
            return ComputeSettings.from_mapping(training, allow_unavailable=True).execution_environment()
        except ValueError:
            return {}

    def _materialize_resolved_configs(self, campaign: Mapping[str, Any]) -> None:
        for run in campaign.get("expected_runs", ()):
            target = Path(str(run["resolved_config_path"]))
            payload = strict_json_dumps(run["resolved_config"], indent=2, sort_keys=True) + "\n"
            if target.exists():
                current = strict_json_loads(target.read_bytes())
                if current != run["resolved_config"] or stable_hash(current) != run["resolved_config_sha256"]:
                    raise ValueError(f"Resolved configuration already exists with different bytes: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            partial.write_text(payload, encoding="utf-8")
            partial.replace(target)

    def _configured_inputs(self, campaign: Mapping[str, Any]) -> list[Path]:
        config = project_config_from_context(self.context)
        candidates = [config.path_for("training", "dataset_freeze") or config.path_for("dataset", "freeze_manifest")]
        identities = campaign.get("identities") if isinstance(campaign.get("identities"), Mapping) else {}
        for identity_field in ("dataset_view_manifest_path", "split_manifest_path", "conditioning_vocabulary_path"):
            if identities.get(identity_field):
                candidates.append(Path(str(identities[identity_field])))
        unique = []
        for path in candidates:
            if path is not None and path.is_file() and path not in unique:
                unique.append(path)
        return unique


def _default_training_audit_snapshot_opener(config: Any, report: Any, activation: Any) -> Any:
    from spritelab.product_features.training.audit import open_training_audit_execution_snapshot

    return open_training_audit_execution_snapshot(config, report, activation)


def _require_applicable_training_audit_snapshot(snapshot: Any) -> str:
    if getattr(snapshot, "status", None) is not AuditStatus.PASS:
        raise ValueError("The retained training infrastructure audit is not an applicable PASS.")
    evidence = getattr(snapshot, "launch_authorization_evidence_sha256", None)
    if not isinstance(evidence, str) or not _is_sha256(evidence):
        raise ValueError("The retained training infrastructure audit has no concrete authorization identity.")
    snapshot.verify_unchanged()
    return evidence


def _require_validated_launch_authorization(validated: ValidatedTrainingLaunch, expected: str) -> None:
    if not _is_sha256(expected):
        raise CampaignValidationError("Launch authorization evidence identity is malformed.")
    receipt_identity = validated.receipt.launch_authorization_evidence_sha256
    context_identity = validated.validator_context.launch_authorization_evidence_sha256
    if receipt_identity != expected or context_identity != expected:
        raise CampaignValidationError("Validated launch authorization evidence changed before backend dispatch.")


def _verify_audit_then(snapshot: Any, operation: Callable[[], Any]) -> Any:
    snapshot.verify_unchanged()
    return operation()


def _committed_activation_binding(activation: Any, campaign: Mapping[str, Any]) -> tuple[str, str]:
    commit = getattr(activation, "activation_commit", None)
    if getattr(activation, "ready", None) is not True or not isinstance(commit, Mapping):
        raise ValueError("A committed conditioned activation receipt is required.")
    record_identity = str(commit.get("record_identity") or "")
    config_identity = str(commit.get("config_after_sha256") or "")
    if (
        commit.get("committed") is not True
        or not _is_sha256(record_identity)
        or not _is_sha256(config_identity)
        or commit.get("campaign_identity_sha256") != campaign.get("campaign_identity")
    ):
        raise ValueError("The committed activation receipt does not bind this exact campaign and configuration.")
    return record_identity, config_identity


def _operation_record(
    *,
    action: str,
    run_id: str,
    operation_nonce: str,
    campaign_identity: str,
    backend_id: str,
    backend_configuration_identity: str,
    activation_commit_record_identity: str,
    project_config_sha256: str,
    launch_authorization_evidence_sha256: str,
    status: str,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    base = {
        "schema_version": "spritelab.training.backend-operation.v2",
        "action": action,
        "run_id": run_id,
        "operation_nonce": operation_nonce,
        "campaign_identity_sha256": campaign_identity,
        "backend_id": backend_id,
        "backend_configuration_identity_sha256": backend_configuration_identity,
        "activation_commit_record_identity": activation_commit_record_identity,
        "project_config_sha256": project_config_sha256,
        "launch_authorization_evidence_sha256": launch_authorization_evidence_sha256,
        "status": status,
        "started_at": started_at,
        "completed_at": None,
        "paths_exposed": False,
    }
    return _validate_operation_record({**base, "operation_identity": stable_hash(base)}, expected_run_id=run_id)


def _validate_operation_record(value: Any, *, expected_run_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignValidationError("Durable Training operation state is missing or malformed.")
    record = dict(value)
    required = {
        "schema_version",
        "action",
        "run_id",
        "operation_nonce",
        "campaign_identity_sha256",
        "backend_id",
        "backend_configuration_identity_sha256",
        "activation_commit_record_identity",
        "project_config_sha256",
        "launch_authorization_evidence_sha256",
        "status",
        "started_at",
        "completed_at",
        "paths_exposed",
        "operation_identity",
    }
    if set(record) != required:
        raise CampaignValidationError("Durable Training operation fields are not the exact supported schema.")
    action = record.get("action")
    status = record.get("status")
    run_id = record.get("run_id")
    nonce = record.get("operation_nonce")
    if record.get("schema_version") != "spritelab.training.backend-operation.v2":
        raise CampaignValidationError("Durable Training operation schema is unsupported.")
    if action not in {"start", "resume", "pause", "cancel"}:
        raise CampaignValidationError("Durable Training operation action is invalid.")
    if status not in {"RUNNING", "LAUNCHED", "COMPLETE", "BLOCKED", "PAUSED", "CANCELLED", "UNCERTAIN"}:
        raise CampaignValidationError("Durable Training operation status is invalid.")
    if not isinstance(run_id, str) or not run_id or (expected_run_id is not None and run_id != expected_run_id):
        raise CampaignValidationError("Durable Training operation run identity is invalid.")
    if (
        not isinstance(nonce, str)
        or not nonce.startswith("operation-")
        or len(nonce) != 42
        or not all(character in "0123456789abcdef" for character in nonce[10:])
    ):
        raise CampaignValidationError("Durable Training operation nonce is invalid.")
    for key in (
        "campaign_identity_sha256",
        "backend_configuration_identity_sha256",
        "activation_commit_record_identity",
        "project_config_sha256",
        "launch_authorization_evidence_sha256",
        "operation_identity",
    ):
        if not isinstance(record.get(key), str) or not _is_sha256(str(record[key])):
            raise CampaignValidationError("Durable Training operation identity is invalid.")
    if not isinstance(record.get("backend_id"), str) or not str(record["backend_id"]).strip():
        raise CampaignValidationError("Durable Training operation backend is invalid.")
    if not isinstance(record.get("started_at"), str) or not record["started_at"]:
        raise CampaignValidationError("Durable Training operation start timestamp is invalid.")
    completed_at = record.get("completed_at")
    if (status == "RUNNING") != (completed_at is None):
        raise CampaignValidationError("Durable Training operation terminal timestamp is inconsistent.")
    if completed_at is not None and (not isinstance(completed_at, str) or not completed_at):
        raise CampaignValidationError("Durable Training operation completion timestamp is invalid.")
    if record.get("paths_exposed") is not False:
        raise CampaignValidationError("Durable Training operation privacy marker is invalid.")
    identity = record.pop("operation_identity")
    if stable_hash(record) != identity:
        raise CampaignValidationError("Durable Training operation identity verification failed.")
    return {**record, "operation_identity": identity}


def _operation_transition(operation: Mapping[str, Any], status: str) -> dict[str, Any]:
    if status not in {"LAUNCHED", "COMPLETE", "BLOCKED", "PAUSED", "CANCELLED", "UNCERTAIN"}:
        raise CampaignValidationError("Durable Training operation terminal status is invalid.")
    validated = _validate_operation_record(operation)
    base = {
        **validated,
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    base.pop("operation_identity", None)
    return _validate_operation_record({**base, "operation_identity": stable_hash(base)})


def _seed_outcome(run: Mapping[str, Any], *, status: str, stage: str) -> dict[str, Any]:
    return {
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "status": status,
        "stage": stage,
        "paths_exposed": False,
    }


def _job_outcome(job: ComputeJob, *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "run_id": job.run_id,
        "remote_identity_sha256": stable_hash({"remote_identity": job.remote_identity}),
        "status": status,
        "paths_exposed": False,
        **extra,
    }


def _complete_seed_outcomes(session: TrainingSession, campaign: Mapping[str, Any]) -> None:
    for run in campaign.get("expected_runs", ()):
        if not isinstance(run, Mapping):
            continue
        key = str(run.get("run_id") or "")
        if key not in session.seed_outcomes:
            session.seed_outcomes[key] = _seed_outcome(run, status="NOT_ATTEMPTED", stage="not_attempted")


def _block_incomplete_seed_outcomes(session: TrainingSession) -> None:
    for key, outcome in tuple(session.seed_outcomes.items()):
        if outcome.get("status") == "RUNNING":
            session.seed_outcomes[key] = {**outcome, "status": "BLOCKED"}


def _request_for_run(session: TrainingSession, run_id: str) -> ComputeJobRequest | None:
    candidates = [request for request in session.requests.values() if request.run_id == run_id]
    return min(
        candidates, key=lambda request: ("-resume-" in request.idempotency_key, request.idempotency_key), default=None
    )


def _sorted_outcomes(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(value)
        for _key, value in sorted(
            values.items(),
            key=lambda item: (int(item[1].get("seed", -1)), str(item[0])),
        )
    ]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Durable Training {label} is invalid.")
    return value


def _durable_identity(state: Mapping[str, Any], *keys: str) -> str | None:
    backend = state.get("backend_identity")
    backend = backend if isinstance(backend, Mapping) else {}
    values: set[str] = set()
    for container in (state, backend):
        for key in keys:
            if key not in container:
                continue
            value = container[key]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"Durable training identity alias {key} must be a non-empty, unpadded string.")
            values.add(value)
    if len(values) > 1:
        raise ValueError(f"Durable training identity aliases disagree: {', '.join(keys)}")
    return next(iter(values), None)


def _plan_training_identities(plan: ResolvedTrainingPlan) -> tuple[str, str]:
    campaign = plan.campaign
    if not isinstance(campaign, Mapping):
        raise ValueError("Durable training identity requires a retained campaign plan.")
    identities = campaign.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("Retained campaign training identities are missing.")
    dataset_identity = identities.get("dataset_identity_hash", identities.get("dataset_view_manifest_hash"))
    view_identity = identities.get("dataset_view_manifest_hash")
    for label, value in (
        ("dataset", dataset_identity),
        ("view", view_identity),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"Retained campaign {label} identity is malformed.")
    return dataset_identity, view_identity


def _evaluation_checkpoint_binding(
    session: TrainingSession,
    runs_directory: Path | None,
) -> dict[str, Any]:
    """Project event-derived checkpoint authority into the terminal event."""

    run_directory = None if runs_directory is None else (runs_directory / session.run_id).resolve()
    rows: list[dict[str, Any]] = []
    for checkpoint in session.dashboard.checkpoints if session.dashboard is not None else ():
        raw_path = Path(checkpoint.checkpoint)
        candidate = raw_path if raw_path.is_absolute() or run_directory is None else run_directory / raw_path
        try:
            relative_path = candidate.resolve().relative_to(run_directory).as_posix() if run_directory else ""
        except (OSError, RuntimeError, ValueError):
            continue
        if not relative_path:
            continue
        rows.append(
            {
                "path": relative_path,
                "sha256": checkpoint.sha256.lower() if isinstance(checkpoint.sha256, str) else None,
                "seed": checkpoint.seed,
                "optimizer_step": checkpoint.optimizer_step,
                "backend_id": checkpoint.backend_id,
                "remote": checkpoint.remote,
                "downloaded": checkpoint.downloaded,
                "hash_verified": checkpoint.hash_verified,
                "remote_identity_verified": checkpoint.remote_identity_verified,
                "safe_resume": checkpoint.safe_resume,
                "verification": checkpoint.verification,
            }
        )
    rows.sort(key=lambda row: (row["path"], row["optimizer_step"], row["seed"] is None, row["seed"] or 0))
    return {
        "schema_version": EVALUATION_CHECKPOINT_BINDING_SCHEMA,
        "run_id": session.run_id,
        "dataset_identity": session.dataset_identity,
        "view_identity": session.view_identity,
        "checkpoints": rows,
    }


def _plan_to_state(plan: ResolvedTrainingPlan) -> dict[str, Any]:
    return {
        "profile": plan.profile.value,
        "model_label": plan.model_label,
        "dataset_count": plan.dataset_count,
        "dataset_ready": plan.dataset_ready,
        "backend_id": plan.backend_id,
        "campaign": plan.campaign,
        "gates": [asdict(item) for item in plan.gates],
        "estimate": plan.estimate.to_dict(),
        "resume_report": plan.resume_report,
        "advanced_collapsed": plan.advanced_collapsed,
    }


def _plan_from_state(value: Any) -> ResolvedTrainingPlan | None:
    if not isinstance(value, Mapping):
        return None
    gates = tuple(TrainingGate(**dict(item)) for item in value.get("gates", ()) if isinstance(item, Mapping))
    estimate_raw = value.get("estimate")
    if not isinstance(estimate_raw, Mapping):
        return None
    estimate = ComputeEstimate(**dict(estimate_raw))
    campaign = value.get("campaign")
    if campaign is not None and not isinstance(campaign, Mapping):
        return None
    return ResolvedTrainingPlan(
        profile=TrainingProfile(str(value["profile"])),
        model_label=str(value.get("model_label") or "Training"),
        dataset_count=int(value["dataset_count"]) if value.get("dataset_count") is not None else None,
        dataset_ready=bool(value.get("dataset_ready")),
        backend_id=str(value.get("backend_id") or "unknown"),
        campaign=dict(campaign) if isinstance(campaign, Mapping) else None,
        gates=gates,
        estimate=estimate,
        resume_report=dict(value["resume_report"]) if isinstance(value.get("resume_report"), Mapping) else None,
        advanced_collapsed=bool(value.get("advanced_collapsed", True)),
    )


def _job_to_state(job: ComputeJob) -> dict[str, Any]:
    return {
        "backend_id": job.backend_id,
        "job_id": job.job_id,
        "run_id": job.run_id,
        "status": job.status.value,
        "remote_identity": job.remote_identity,
        "may_accrue_cost": job.may_accrue_cost,
        "metadata": dict(job.metadata),
    }


def _job_from_state(value: Any) -> ComputeJob | None:
    if not isinstance(value, Mapping):
        return None
    return ComputeJob(
        backend_id=str(value["backend_id"]),
        job_id=str(value["job_id"]),
        run_id=str(value["run_id"]),
        status=ComputeStatus(str(value["status"])),
        remote_identity=str(value["remote_identity"]),
        may_accrue_cost=bool(value.get("may_accrue_cost", False)),
        metadata=dict(value.get("metadata") or {}),
    )


def _prepared_reference_to_state(prepared: PreparedCompute) -> dict[str, str]:
    """Serialize only opaque backend-owned identifiers needed for cleanup recovery."""

    return {
        "backend_id": validate_identifier(prepared.backend_id, label="prepared backend id"),
        "operation_id": validate_identifier(prepared.operation_id, label="prepared operation id"),
        "remote_identity": validate_identifier(prepared.remote_identity, label="prepared remote identity"),
    }


def _prepared_reference_from_state(value: Any, backend: ComputeBackend) -> PreparedCompute:
    if not isinstance(value, Mapping):
        raise ValueError("Durable prepared-resource reference is malformed.")
    backend_id = validate_identifier(str(value["backend_id"]), label="prepared backend id")
    operation_id = validate_identifier(str(value["operation_id"]), label="prepared operation id")
    remote_identity = validate_identifier(str(value["remote_identity"]), label="prepared remote identity")
    if backend_id != backend.backend_id:
        raise ValueError("Durable prepared-resource backend identity changed.")

    # Prefer the adapter's live prepared registry when available. Stateless SSH
    # adapters can safely recover their configured remote workspace; local and
    # fake cleanup use only the opaque operation identity.
    registry = getattr(backend, "_prepared", None)
    retained = registry.get(operation_id) if isinstance(registry, Mapping) else None
    if isinstance(retained, PreparedCompute):
        if retained.remote_identity != remote_identity or retained.backend_id != backend_id:
            raise ValueError("Backend prepared-resource identity changed.")
        return retained
    settings = getattr(backend, "settings", None)
    configured_workspace = getattr(settings, "workspace", "")
    workspace = configured_workspace if isinstance(configured_workspace, str) else ""
    return PreparedCompute(backend_id, operation_id, workspace, remote_identity)


def _prepared_to_state(prepared: PreparedCompute) -> dict[str, Any]:
    return {
        "backend_id": prepared.backend_id,
        "operation_id": prepared.operation_id,
        "workspace": prepared.workspace,
        "remote_identity": prepared.remote_identity,
        "metadata": dict(prepared.metadata),
    }


def _prepared_from_state(value: Any) -> PreparedCompute | None:
    if not isinstance(value, Mapping):
        return None
    return PreparedCompute(
        backend_id=str(value["backend_id"]),
        operation_id=str(value["operation_id"]),
        workspace=str(value["workspace"]),
        remote_identity=str(value["remote_identity"]),
        metadata=dict(value.get("metadata") or {}),
    )


def _request_to_state(request: ComputeJobRequest) -> dict[str, Any]:
    return {
        "run_id": request.run_id,
        "command": list(request.command),
        "idempotency_key": request.idempotency_key,
        "campaign_identity": request.campaign_identity,
        "run_identity": request.run_identity,
        "local_project_root": str(request.local_project_root),
        "output_root": str(request.output_root),
        "event_path": str(request.event_path) if request.event_path else None,
        "environment": dict(request.environment),
        "execution_spec_identity": request.execution_spec_identity,
        "output_root_identity": request.output_root_identity,
        "launch_authorization_evidence_sha256": request.launch_authorization_evidence_sha256,
        "compute_backend_id": request.compute_backend_id,
    }


def _request_from_state(value: Any) -> ComputeJobRequest | None:
    if not isinstance(value, Mapping):
        return None
    event_path = value.get("event_path")
    normalized_event_path = Path(str(event_path)) if event_path else None
    if normalized_event_path is not None and normalized_event_path.name == LEGACY_EVENT_FILENAME:
        normalized_event_path = normalized_event_path.with_name(EVENT_FILENAME)
    return ComputeJobRequest(
        run_id=str(value["run_id"]),
        command=tuple(str(item) for item in value["command"]),
        idempotency_key=str(value["idempotency_key"]),
        campaign_identity=str(value["campaign_identity"]),
        run_identity=str(value["run_identity"]),
        local_project_root=Path(str(value["local_project_root"])),
        output_root=Path(str(value["output_root"])),
        event_path=normalized_event_path,
        environment={str(key): str(item) for key, item in dict(value.get("environment") or {}).items()},
        execution_spec_identity=str(value.get("execution_spec_identity") or ""),
        output_root_identity=str(value.get("output_root_identity") or ""),
        launch_authorization_evidence_sha256=str(value["launch_authorization_evidence_sha256"]),
        compute_backend_id=str(value.get("compute_backend_id") or ""),
    )

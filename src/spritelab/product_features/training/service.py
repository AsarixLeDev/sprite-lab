"""Product orchestration that delegates every launch gate to existing backends."""

from __future__ import annotations

from collections.abc import Mapping
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
from spritelab.training.campaign import (
    CampaignValidationError,
    audit_artifact_completeness,
    execute_campaign,
    stable_hash,
)
from spritelab.training.launch import ValidatedTrainingLaunch


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
    requests: dict[str, ComputeJobRequest] = field(default_factory=dict)
    dataset_identity: str | None = None
    view_identity: str | None = None
    reconstructed: bool = False


class TrainingService:
    def __init__(
        self,
        context: ProjectContext,
        backend: ComputeBackend,
        *,
        resolver: TrainingPlanResolver | None = None,
    ) -> None:
        self.context = context
        self.backend = backend
        self.resolver = resolver or TrainingPlanResolver()
        self.sessions: dict[str, TrainingSession] = {}
        self.repository = EventRepository(context.runs_directory, private_roots=(context.project_root,))

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
        resume: bool = False,
    ) -> ProductResult:
        plan = self.plan(profile, custom_spec=custom_spec, before_launch=True)
        if plan.blockers:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Training was not launched because a mandatory gate is closed.",
                blockers=tuple(ProductBlocker(gate.gate_id, gate.message, gate.resolution) for gate in plan.blockers),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        if self.backend.is_cloud and not cloud_confirmation:
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="Cloud training requires explicit confirmation immediately before launch.",
                blockers=(
                    ProductBlocker(
                        "cloud_confirmation",
                        "Confirm that the selected hosted resource may incur cost.",
                        "Review GPU, disk, shutdown policy, and credential status, then confirm Start training.",
                    ),
                ),
                data={**plan.to_dict(), "backend_launches": 0},
            )
        campaign = plan.campaign
        assert campaign is not None
        config = project_config_from_context(self.context)
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
        dashboard = DashboardState(session_id, self.backend.backend_id)
        session = TrainingSession(session_id, self.backend, plan, dashboard=dashboard)
        self.sessions[session_id] = session
        started_at = datetime.now(timezone.utc).isoformat()
        self.repository.create_run(
            session_id,
            feature="training",
            command="training.start",
            status=ProductStatus.RUNNING.value,
            stage="campaign",
            started_at=started_at,
            resumable=False,
            backend_id=self.backend.backend_id,
            backend_run_reference=session_id,
            backend_identity={"backend_id": self.backend.backend_id},
            extra={"plan": _plan_to_state(plan), "jobs": [], "cursors": {}},
        )
        self._apply(
            session,
            ProductEvent(
                run_id=session_id,
                timestamp=started_at,
                feature="training",
                stage="campaign",
                event_type="training_started",
                status=ProductStatus.RUNNING,
                current=0,
                total=len(campaign["expected_runs"]),
                message="Validated training campaign started.",
            ),
        )
        configured_inputs = self._configured_inputs(campaign)

        def runner(command: list[str], **kwargs: Any) -> Any:
            validated = kwargs.get("validated_launch")
            if not isinstance(validated, ValidatedTrainingLaunch):
                raise CampaignValidationError("Campaign runner requires a validator-issued launch receipt.")
            self._bind_training_identity(session, validated)
            run = validated.run
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
                compute_backend_id=self.backend.backend_id,
                launch_receipt=validated.receipt,
                validator_context=validated.validator_context,
            )
            prepared = self.backend.prepare(self.context, request)
            self.backend.upload(prepared, [Path(str(run["resolved_config_path"])), *configured_inputs])
            job = self.backend.launch(
                prepared,
                request,
                cloud_confirmation=cloud_confirmation,
            )
            session.jobs.append(job)
            session.prepared[request.run_id] = prepared
            session.requests[request.run_id] = request
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
                    metrics={"seed": run["seed"], "checkpoint_schedule": run["expected_checkpoint_steps"]},
                ),
            )
            self._persist_session(session)
            return SimpleNamespace(returncode=0)

        try:
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
                runner=runner,
            )
        except (CampaignValidationError, ComputeBackendError, OSError, ValueError) as exc:
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
                ),
            )
            return ProductResult(
                status=ProductStatus.BLOCKED,
                feature="training",
                message="The authoritative campaign refused launch.",
                blockers=(ProductBlocker("authoritative_launch", str(exc)),),
                data={"backend_launches": len(session.jobs)},
            )
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
                started_at=datetime.now(timezone.utc).isoformat(),
            ),
            data={
                "execution": execution,
                "dashboard": dashboard.to_dict(),
                "training_identity": self._training_identity_projection(session),
            },
        )

    def refresh(self, run_id: str) -> ProductResult:
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
        polls = []
        for job in session.jobs:
            cursor = session.cursors.get(job.job_id, 0)
            try:
                events, next_cursor = session.backend.stream_events(job, cursor=cursor)
                session.cursors[job.job_id] = next_cursor
                for event in events:
                    # Backend seed events have per-seed run IDs; normalize only the
                    # aggregate dashboard identity while preserving all metrics.
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
                poll = session.backend.poll(job)
                polls.append(poll)
                if poll.resource_state_uncertain:
                    self._apply(
                        session,
                        ProductEvent(
                            run_id=run_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            feature="training",
                            stage="remote",
                            event_type="remote_failure",
                            status=ProductStatus.FAILED,
                            message=poll.message,
                            metrics={
                                "resource_state_uncertain": True,
                                "may_accrue_cost": poll.may_accrue_cost,
                                "shutdown_guidance": "Open the provider console and explicitly stop or terminate the resource by ID.",
                            },
                        ),
                    )
            except ComputeBackendError as exc:
                session.dashboard.warnings.append(str(exc))
        if polls and not any(item.resource_state_uncertain for item in polls):
            if all(item.status == ComputeStatus.COMPLETE for item in polls):
                campaign = session.plan.campaign
                completion = (
                    audit_artifact_completeness(campaign)
                    if campaign is not None
                    else {"complete": False, "reasons": ["missing campaign"]}
                )
                status = ProductStatus.COMPLETE if completion["complete"] else ProductStatus.BLOCKED
                if not completion["complete"]:
                    session.dashboard.warnings.append(
                        "Backend jobs stopped successfully, but exact campaign completion artifacts are incomplete."
                    )
            elif all(item.status == ComputeStatus.PAUSED for item in polls):
                status = ProductStatus.PAUSED
            elif any(item.status == ComputeStatus.FAILED for item in polls):
                status = ProductStatus.FAILED
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
                    metrics={"completion_validated": status == ProductStatus.COMPLETE},
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
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        if session.dashboard.status != ProductStatus.RUNNING:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Pause is unavailable because this run is not currently running.",
                feature="training",
            )
        try:
            messages = [session.backend.pause(job).message for job in session.jobs]
        except (ComputeBackendError, NotImplementedError) as exc:
            return ProductResult(ProductStatus.UNAVAILABLE, f"Pause is not supported: {exc}", feature="training")
        refreshed = self.refresh(run_id)
        paused = refreshed.status == ProductStatus.PAUSED
        return ProductResult(
            ProductStatus.PAUSED if paused else ProductStatus.RUNNING,
            (
                "Training paused gracefully. Resume remains unavailable until a checkpoint passes the safe-resume contract."
                if paused
                else "Graceful pause requested; waiting for the backend to preserve safe state."
            ),
            feature="training",
            warnings=tuple(ProductWarning("pause", message) for message in messages),
            data={"unsafe_resume_available": False},
        )

    def cancel(self, run_id: str) -> ProductResult:
        session = self._session(run_id)
        if session is None or session.dashboard is None:
            return ProductResult(ProductStatus.UNAVAILABLE, "Training run is not available.", feature="training")
        if session.dashboard.status not in {ProductStatus.RUNNING, ProductStatus.PAUSED}:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "Cancel is unavailable because this run is no longer active.",
                feature="training",
            )
        try:
            messages = [session.backend.cancel(job).message for job in session.jobs]
        except (ComputeBackendError, NotImplementedError) as exc:
            return ProductResult(ProductStatus.UNAVAILABLE, f"Cancel is not supported: {exc}", feature="training")
        event = ProductEvent(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            feature="training",
            stage="campaign",
            event_type="training_cancelled",
            status=ProductStatus.FAILED,
            current=session.dashboard.campaign_current,
            total=session.dashboard.campaign_total,
            message="Training was cancelled through the registered compute backend.",
        )
        self._apply(session, event)
        self.repository.update_state(run_id, status="CANCELLED", ended_at=event.timestamp, resumable=False)
        return ProductResult(
            ProductStatus.FAILED,
            "Training was cancelled.",
            feature="training",
            warnings=tuple(ProductWarning("cancel", message) for message in messages),
            data={"cancelled": True},
        )

    def resume(self, run_id: str, *, cloud_confirmation: bool = False) -> ProductResult:
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
        if session.backend.is_cloud and not cloud_confirmation:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Cloud resume requires explicit confirmation.",
                feature="training",
                blockers=(ProductBlocker("cloud_confirmation", "Confirm hosted cost before resume."),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        plan = self.plan(session.plan.profile, before_launch=True)
        if plan.blockers or plan.campaign is None:
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume gates are no longer satisfied.",
                feature="training",
                blockers=tuple(ProductBlocker(item.gate_id, item.message, item.resolution) for item in plan.blockers),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        campaign = plan.campaign
        config = project_config_from_context(self.context)
        campaign_config_path = config.path_for("training", "campaign_config")
        if campaign_config_path is None or not campaign_config_path.is_file():
            return ProductResult(
                ProductStatus.BLOCKED,
                "Safe resume requires the exact authoritative campaign configuration.",
                feature="training",
                blockers=(ProductBlocker("campaign_configuration", "training.campaign_config is required."),),
                data={"backend_launches": 0, "unsafe_resume_available": False},
            )
        launched: list[ComputeJob] = []

        def runner(command: list[str], **kwargs: Any) -> Any:
            validated = kwargs.get("validated_launch")
            if not isinstance(validated, ValidatedTrainingLaunch):
                raise CampaignValidationError("Resume runner requires a validator-issued continuation receipt.")
            self._bind_training_identity(session, validated)
            self._verify_session_migrations(session)
            run = validated.run
            seed = int(run["seed"])
            checkpoint_state = max(
                (item for item in session.dashboard.checkpoints if item.seed == seed and item.safe_resume),
                key=lambda item: item.optimizer_step,
                default=None,
            )
            if checkpoint_state is None:
                raise CampaignValidationError(f"Seed {seed} has no product-verified safe checkpoint.")
            previous = session.requests[str(run["run_id"])]
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
                compute_backend_id=session.backend.backend_id,
                launch_receipt=validated.receipt,
                validator_context=validated.validator_context,
            )
            prepared = session.backend.prepare(self.context, request)
            artifact = ArtifactReference(
                checkpoint_state.checkpoint,
                checkpoint_state.sha256 or "",
                prepared.remote_identity,
                Path(checkpoint_state.checkpoint),
                downloaded=checkpoint_state.downloaded,
                hash_verified=checkpoint_state.hash_verified,
                remote_identity_verified=checkpoint_state.remote_identity_verified,
            )
            job = session.backend.resume(
                prepared,
                ResumeRequest(request, artifact, safe_resume=True),
                cloud_confirmation=cloud_confirmation,
            )
            session.prepared[request.run_id] = prepared
            session.requests[request.run_id] = request
            launched.append(job)
            return SimpleNamespace(returncode=0)

        try:
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
                runner=runner,
            )
        except (CampaignValidationError, ComputeBackendError, OSError, ValueError) as exc:
            return ProductResult(
                ProductStatus.BLOCKED,
                "The authoritative safe-resume contract refused continuation.",
                feature="training",
                blockers=(ProductBlocker("safe_resume", str(exc)),),
                data={"backend_launches": len(launched), "unsafe_resume_available": False},
            )
        session.jobs.extend(launched)
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
        )
        self._apply(session, resumed_event)
        self._persist_session(session)
        return ProductResult(
            ProductStatus.RUNNING,
            "Safe resume launched through the validated campaign and backend contracts.",
            feature="training",
            data={"execution": execution, "unsafe_resume_available": False},
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
        self.repository.append(event)

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
        return dashboard

    def _session(self, run_id: str) -> TrainingSession | None:
        existing = self.sessions.get(run_id)
        state = self.repository.state(run_id)
        if str(state.get("feature") or state.get("command")) != "training":
            self.sessions.pop(run_id, None)
            return None
        if not self.repository.replay(run_id).safe_for_resume:
            self.sessions.pop(run_id, None)
            return None
        try:
            plan = _plan_from_state(state.get("plan"))
            if plan is None:
                return None
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
                existing.dataset_identity = dataset_identity
                existing.view_identity = view_identity
                return existing
            session = TrainingSession(
                run_id,
                self.backend,
                plan,
                dashboard=dashboard,
                cursors={str(key): int(value) for key, value in dict(state.get("cursors") or {}).items()},
                dataset_identity=dataset_identity,
                view_identity=view_identity,
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
                session.prepared[request.run_id] = prepared
                session.requests[request.run_id] = request
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
            request = session.requests.get(job.run_id)
            prepared = session.prepared.get(job.run_id)
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
        backend_identity: dict[str, Any] = {
            "backend_id": session.backend.backend_id,
            "remote_identities": sorted({item.remote_identity for item in session.prepared.values()}),
        }
        updates: dict[str, Any] = {
            "jobs": rows,
            "cursors": dict(session.cursors),
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
    )

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spritelab.product_core import ProductEvent, ProductStatus, ProjectContext, strict_json_dumps
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.service import (
    TrainingService,
    TrainingSession,
    _job_to_state,
    _plan_to_state,
    _prepared_to_state,
    _request_to_state,
)
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    EventMigrationState,
    EventRepository,
    record_event_history_origin,
    verify_event_migration,
)
from spritelab.remote_compute import (
    ArtifactReference,
    ComputeBackendError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputeStatus,
    FakeComputeBackend,
    LocalComputeBackend,
    PreparedCompute,
    ResumeRequest,
    SSHComputeBackend,
    SSHSettings,
)
from spritelab.remote_compute.ssh import RemoteResult
from spritelab.training.campaign import RESUME_CHECKPOINT_SCHEMA_VERSION, file_sha256, stable_hash
from spritelab.training.launch import ValidatedTrainingLaunch, prepare_validated_training_launch
from training_launch_test_utils import validated_launch


def _event_bytes(run_id: str, *, current: int = 1, terminal_newline: bool = True) -> bytes:
    event = ProductEvent(
        run_id=run_id,
        timestamp="2026-07-14T10:00:00+00:00",
        feature="training",
        stage="seed",
        event_type="progress",
        status=ProductStatus.RUNNING,
        current=current,
        total=10,
        message="Synthetic migration evidence.",
        metrics={"optimizer_step": current, "loss": 0.5},
    )
    value = strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return value + (b"\n" if terminal_newline else b"")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_pre_origin_legacy_fixture(directory: Path) -> None:
    """Convert a newly created synthetic run into a pre-contract legacy run."""

    (directory / EVENT_FILENAME).unlink()
    (directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in tuple(state):
        if (
            key.startswith("event_history_origin")
            or key.startswith("event_migration_")
            or key.startswith("event_canonical_")
        ):
            state.pop(key)
    _write_json(state_path, state)


def _migrated_directory(
    tmp_path: Path,
    run_id: str,
    *,
    source_removed: bool = False,
    terminal_newline: bool = True,
) -> tuple[Path, bytes]:
    directory = tmp_path / run_id
    directory.mkdir(parents=True)
    source = _event_bytes(run_id, terminal_newline=terminal_newline)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    EventRepository(tmp_path).migrate_legacy_events(run_id)
    if source_removed:
        (directory / LEGACY_EVENT_FILENAME).unlink()
    return directory, source


def _copy_migration_evidence(target: Path, run_id: str, tmp_path: Path, *, source_removed: bool) -> None:
    staging_root = tmp_path / "event-migration-staging"
    staging, _source = _migrated_directory(staging_root, run_id, source_removed=source_removed)
    target.mkdir(parents=True, exist_ok=True)
    for name in (EVENT_FILENAME, LEGACY_MIGRATION_FILENAME, EVENT_HISTORY_ORIGIN_FILENAME):
        (target / name).write_bytes((staging / name).read_bytes())
    if not source_removed:
        (target / LEGACY_EVENT_FILENAME).write_bytes((staging / LEGACY_EVENT_FILENAME).read_bytes())
    record_event_history_origin(
        run_id,
        target,
        expected_origin=EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
        allow_binding_population=True,
    )


def test_migration_classifier_accepts_source_present_and_explicitly_removed(tmp_path: Path) -> None:
    present, _ = _migrated_directory(tmp_path / "present", "present")
    removed, _ = _migrated_directory(tmp_path / "removed", "removed", source_removed=True)

    present_result = verify_event_migration("present", present)
    removed_result = verify_event_migration("removed", removed)

    assert present_result.state == EventMigrationState.VERIFIED_SOURCE_PRESENT
    assert removed_result.state == EventMigrationState.VERIFIED_SOURCE_REMOVED
    assert present_result.safe_for_migrated_resume
    assert removed_result.safe_for_migrated_resume


def test_source_removal_without_explicit_versioned_permission_is_invalid(tmp_path: Path) -> None:
    directory, _ = _migrated_directory(tmp_path, "no-removal-policy", source_removed=True)
    record_path = directory / LEGACY_MIGRATION_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["legacy_source_policy"] = "must_be_retained"
    _write_json(record_path, record)

    result = verify_event_migration("no-removal-policy", directory)

    assert result.state == EventMigrationState.INVALID_RECORD
    assert not result.resume_compatible


@pytest.mark.parametrize("mutation", ["changed", "truncated", "replaced"])
def test_source_removed_canonical_tamper_is_not_comparable(tmp_path: Path, mutation: str) -> None:
    run_id = f"canonical-{mutation}"
    directory, source = _migrated_directory(tmp_path, run_id, source_removed=True)
    canonical = directory / EVENT_FILENAME
    if mutation == "changed":
        canonical.write_bytes(source.replace(b'"feature":"training"', b'"feature":"draining"', 1))
    elif mutation == "truncated":
        canonical.write_bytes(source[:-1])
    else:
        canonical.write_bytes(_event_bytes(run_id, current=9))

    result = verify_event_migration(run_id, directory)

    assert result.state == EventMigrationState.NOT_COMPARABLE
    assert not result.resume_compatible


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("legacy_size_bytes", 999),
        ("legacy_sha256", "a" * 64),
        ("canonical_prefix_sha256", "b" * 64),
        ("canonical_relative_path", "other-events.jsonl"),
        ("run_id", "other-run"),
    ],
)
def test_source_removed_record_metadata_tamper_is_invalid(tmp_path: Path, field: str, value: object) -> None:
    run_id = f"record-{field.replace('_', '-')}"
    directory, _ = _migrated_directory(tmp_path, run_id, source_removed=True)
    record_path = directory / LEGACY_MIGRATION_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = value
    _write_json(record_path, record)

    result = verify_event_migration(run_id, directory)

    assert result.state == EventMigrationState.INVALID_RECORD
    assert not result.resume_compatible


def test_malformed_and_required_missing_records_fail_closed(tmp_path: Path) -> None:
    malformed, _ = _migrated_directory(tmp_path / "bad", "bad-record", source_removed=True)
    (malformed / LEGACY_MIGRATION_FILENAME).write_text("{", encoding="utf-8")
    assert verify_event_migration("bad-record", malformed).state == EventMigrationState.INVALID_RECORD

    missing, _ = _migrated_directory(tmp_path / "missing", "missing-record", source_removed=True)
    (missing / LEGACY_MIGRATION_FILENAME).unlink()
    result = verify_event_migration("missing-record", missing, migration_required=True)
    assert result.state == EventMigrationState.INVALID_RECORD
    assert not result.resume_compatible

    unbound = verify_event_migration("missing-record", missing)
    assert unbound.state == EventMigrationState.INVALID_RECORD
    assert unbound.state != EventMigrationState.NO_MIGRATION
    assert not unbound.resume_compatible


def test_missing_required_record_blocks_append_without_mutating_canonical(tmp_path: Path) -> None:
    run_id = "missing-record-append"
    repository = EventRepository(tmp_path)
    repository.create_run(run_id, feature="training", command="training")
    directory = tmp_path / run_id
    _make_pre_origin_legacy_fixture(directory)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(_event_bytes(run_id))
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-14T10:01:00+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=2,
            total=10,
            message="Bound migration marker.",
        )
    )
    (directory / LEGACY_EVENT_FILENAME).unlink()
    (directory / LEGACY_MIGRATION_FILENAME).unlink()
    canonical = directory / EVENT_FILENAME
    before = canonical.read_bytes()

    with pytest.raises(ValueError, match="record is missing"):
        repository.append(
            ProductEvent(
                run_id=run_id,
                timestamp="2026-07-14T10:02:00+00:00",
                feature="training",
                stage="seed",
                event_type="progress",
                status=ProductStatus.RUNNING,
                current=3,
                total=10,
                message="Must not append.",
            )
        )

    assert canonical.read_bytes() == before


def test_conflicting_streams_and_changed_retained_source_are_closed_states(tmp_path: Path) -> None:
    conflicting = tmp_path / "conflicting"
    conflicting.mkdir()
    (conflicting / LEGACY_EVENT_FILENAME).write_bytes(_event_bytes("conflicting", current=1))
    (conflicting / EVENT_FILENAME).write_bytes(_event_bytes("conflicting", current=2))
    assert verify_event_migration("conflicting", conflicting).state == EventMigrationState.CONFLICTING_STREAMS

    changed, source = _migrated_directory(tmp_path / "changed", "changed-source")
    (changed / LEGACY_EVENT_FILENAME).write_bytes(source + b" ")
    assert verify_event_migration("changed-source", changed).state == EventMigrationState.STALE_SOURCE_CHANGED


def test_valid_append_after_unterminated_migrated_prefix_remains_verified(tmp_path: Path) -> None:
    run_id = "unterminated-prefix"
    directory, source = _migrated_directory(tmp_path, run_id, terminal_newline=False)
    repository = EventRepository(tmp_path)

    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-14T10:01:00+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=2,
            total=10,
            message="Valid append.",
        )
    )

    canonical = (directory / EVENT_FILENAME).read_bytes()
    assert canonical.startswith(source + b"\n")
    assert verify_event_migration(run_id, directory).state == EventMigrationState.VERIFIED_SOURCE_PRESENT


def test_product_snapshot_is_not_comparable_and_not_resumable_after_prefix_tamper(tmp_path: Path) -> None:
    run_id = "product-status"
    repository = EventRepository(tmp_path)
    repository.create_run(run_id, feature="training", command="training", resumable=True)
    directory = tmp_path / run_id
    _make_pre_origin_legacy_fixture(directory)
    source = _event_bytes(run_id)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    repository.migrate_legacy_events(run_id)
    (directory / LEGACY_EVENT_FILENAME).unlink()
    (directory / EVENT_FILENAME).write_bytes(source.replace(b'"stage":"seed"', b'"stage":"reed"', 1))

    snapshot = repository.snapshot(run_id)

    assert snapshot.status == "NOT_COMPARABLE"
    assert snapshot.resumable is False


def _resume_launch(
    tmp_path: Path,
    backend_id: str,
    *,
    source_removed: bool = True,
) -> tuple[ValidatedTrainingLaunch, Path]:
    initial = validated_launch(tmp_path, backend_id)
    campaign = initial.campaign
    run = initial.run
    root = initial.output_root
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "run_identity.json",
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_identity": campaign["campaign_identity"],
            "run_id": run["run_id"],
            "run_identity": run["run_identity"],
            "output_root": run["output_root"],
            "resolved_config_sha256": run["resolved_config_sha256"],
            "execution_contract_sha256": run["execution_contract_sha256"],
        },
    )
    checkpoint = root / "checkpoint_step_000005.pt"
    checkpoint.write_bytes(b"synthetic exact-resume checkpoint")
    _write_json(
        root / "checkpoint_step_000005.json",
        {
            "optimizer_step": 5,
            "campaign_identity": campaign["campaign_identity"],
            "run_identity": run["run_identity"],
            "resumability_metadata": {
                "schema_version": RESUME_CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_relative_path": checkpoint.name,
                "checkpoint_content_sha256": file_sha256(checkpoint),
                "source_checkpoint_identity": file_sha256(checkpoint),
                "target_runtime_identity": run["run_identity"],
                "experiment_manifest_identity": stable_hash(run["resolved_config"]),
                "exact_replay_eligible": True,
                "unsafe_resume": False,
                "max_optimizer_steps": campaign["training"]["max_optimizer_steps"],
                "gradient_accumulation_position": 0,
                "state_presence": {
                    "model_state_dict": True,
                    "optimizer_state_dict": True,
                    "scheduler_state_dict": True,
                    "ema_state_dict": True,
                    "rng_states": True,
                    "sampler_state": True,
                    "dataloader_generator_state": True,
                },
            },
        },
    )
    _copy_migration_evidence(root, str(run["run_id"]), tmp_path, source_removed=source_removed)
    continuation = prepare_validated_training_launch(
        initial.validator_context.campaign_config_path,
        run_id=str(run["run_id"]),
        compute_backend_id=backend_id,
        project_root=tmp_path,
        execute_confirmed=True,
        resume=True,
    )
    assert continuation.receipt.event_migration_state == (
        EventMigrationState.VERIFIED_SOURCE_REMOVED.value
        if source_removed
        else EventMigrationState.VERIFIED_SOURCE_PRESENT.value
    )
    return continuation, checkpoint


def _request(launch: ValidatedTrainingLaunch) -> ComputeJobRequest:
    return ComputeJobRequest(
        run_id=str(launch.run["run_id"]),
        command=launch.argv,
        idempotency_key=f"{launch.run['run_id']}-continuation",
        campaign_identity=launch.receipt.campaign_identity_sha256,
        run_identity=launch.receipt.run_identity,
        local_project_root=launch.validator_context.project_root,
        output_root=launch.output_root,
        event_path=launch.output_root / EVENT_FILENAME,
        environment=launch.environment,
        execution_spec_identity=launch.receipt.execution_spec_sha256,
        output_root_identity=launch.receipt.output_root_identity,
        compute_backend_id=launch.receipt.compute_backend_id,
        launch_receipt=launch.receipt,
        validator_context=launch.validator_context,
    )


def _durable_product_resume(
    tmp_path: Path,
) -> tuple[TrainingService, FakeComputeBackend, Path, ValidatedTrainingLaunch]:
    launch, checkpoint = _resume_launch(tmp_path, "fake")
    request = _request(launch)
    backend = FakeComputeBackend()
    context = ProjectContext(
        tmp_path,
        {"training": {"campaign_config": str(launch.validator_context.campaign_config_path)}},
        runs_directory=tmp_path / "product-runs",
    )
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Synthetic durable continuation",
        1,
        True,
        backend.backend_id,
        dict(launch.campaign),
        (),
        ComputeEstimate(1, 0, trustworthy=True),
        {"safe": True, "runs": []},
    )

    class Resolver:
        def resolve(self, *args: Any, **kwargs: Any) -> ResolvedTrainingPlan:
            return plan

    session_id = "durable-resume-identity-session"
    prepared = PreparedCompute(
        backend.backend_id,
        request.idempotency_key,
        str(launch.output_root),
        "synthetic-remote-identity",
    )
    job = ComputeJob(
        backend.backend_id,
        request.idempotency_key,
        request.run_id,
        ComputeStatus.PAUSED,
        prepared.remote_identity,
    )
    repository = EventRepository(context.runs_directory)
    repository.create_run(
        session_id,
        feature="training",
        command="training.start",
        status=ProductStatus.PAUSED.value,
        stage="seed",
        resumable=True,
        backend_id=backend.backend_id,
        backend_identity={
            "backend_id": backend.backend_id,
            "dataset_identity": launch.receipt.dataset_identity,
            "view_identity": launch.receipt.view_identity,
            "training_view_identity": launch.receipt.view_identity,
            "dataset_view_manifest_hash": launch.receipt.view_identity,
        },
        extra={
            "plan": _plan_to_state(plan),
            "jobs": [
                {
                    "job": _job_to_state(job),
                    "prepared": _prepared_to_state(prepared),
                    "request": _request_to_state(request),
                }
            ],
            "cursors": {},
            "dataset_identity": launch.receipt.dataset_identity,
            "view_identity": launch.receipt.view_identity,
            "training_view_identity": launch.receipt.view_identity,
        },
    )
    repository.append(
        ProductEvent(
            run_id=session_id,
            timestamp="2026-07-15T10:00:00+00:00",
            feature="training",
            stage="seed",
            event_type="checkpoint",
            status=ProductStatus.PAUSED,
            current=5,
            total=10,
            message="Synthetic resumable checkpoint.",
            metrics={
                "seed": launch.run["seed"],
                "checkpoint": str(checkpoint),
                "optimizer_step": 5,
                "sha256": file_sha256(checkpoint),
                "downloaded": True,
                "hash_verified": True,
                "remote_identity_verified": True,
            },
        )
    )
    state_path = context.runs_directory / session_id / "state.json"  # type: ignore[operator]
    return TrainingService(context, backend, resolver=Resolver()), backend, state_path, launch


@pytest.mark.parametrize("identity", ["dataset", "view"])
def test_reconstructed_resume_rejects_deleted_durable_training_identity(
    tmp_path: Path,
    identity: str,
) -> None:
    service, backend, state_path, _launch = _durable_product_resume(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if identity == "dataset":
        state.pop("dataset_identity")
        state["backend_identity"].pop("dataset_identity")
    else:
        for key in ("view_identity", "training_view_identity"):
            state.pop(key)
        for key in ("view_identity", "training_view_identity", "dataset_view_manifest_hash"):
            state["backend_identity"].pop(key)
    _write_json(state_path, state)

    result = service.resume(str(state["run_id"]))

    assert result.status == ProductStatus.UNAVAILABLE
    assert backend.calls == []
    assert service.sessions == {}


@pytest.mark.parametrize("identity", ["dataset", "view"])
def test_reconstructed_resume_rejects_consistently_forged_training_identity_before_backend(
    tmp_path: Path,
    identity: str,
) -> None:
    service, backend, state_path, launch = _durable_product_resume(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    forged = "f" * 64
    if identity == "dataset":
        assert forged != launch.receipt.dataset_identity
        state["dataset_identity"] = forged
        state["backend_identity"]["dataset_identity"] = forged
    else:
        assert forged != launch.receipt.view_identity
        state["view_identity"] = forged
        state["training_view_identity"] = forged
        state["backend_identity"]["view_identity"] = forged
        state["backend_identity"]["training_view_identity"] = forged
        state["backend_identity"]["dataset_view_manifest_hash"] = forged
    _write_json(state_path, state)

    result = service.resume(str(state["run_id"]))

    assert result.status == ProductStatus.UNAVAILABLE
    assert backend.calls == []
    assert service.sessions == {}


@pytest.mark.parametrize("identity", ["dataset", "view"])
def test_cached_resume_revalidates_deleted_durable_training_identity_before_backend(
    tmp_path: Path,
    identity: str,
) -> None:
    service, backend, state_path, _launch = _durable_product_resume(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run_id = str(state["run_id"])
    assert service._session(run_id) is not None
    if identity == "dataset":
        state.pop("dataset_identity")
        state["backend_identity"].pop("dataset_identity")
    else:
        for key in ("view_identity", "training_view_identity"):
            state.pop(key)
        for key in ("view_identity", "training_view_identity", "dataset_view_manifest_hash"):
            state["backend_identity"].pop(key)
    _write_json(state_path, state)

    result = service.resume(run_id)

    assert result.status == ProductStatus.BLOCKED
    assert result.data["backend_launches"] == 0
    assert any(blocker.code == "durable_training_state" for blocker in result.blockers)
    assert backend.calls == []
    assert service.sessions == {}


@pytest.mark.parametrize("mutation", ["deleted", "forged", "padded"])
def test_dashboard_never_reports_resume_ready_with_invalid_durable_training_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    service, backend, state_path, _launch = _durable_product_resume(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run_id = str(state["run_id"])
    if mutation == "deleted":
        state.pop("dataset_identity")
        state["backend_identity"].pop("dataset_identity")
    else:
        value = "f" * 64 if mutation == "forged" else f" {state['dataset_identity']} "
        state["dataset_identity"] = value
        state["backend_identity"]["dataset_identity"] = value
    _write_json(state_path, state)

    dashboard = service.dashboard(run_id)
    refreshed = service.refresh(run_id)

    assert dashboard.data["resume_available"] is False
    assert refreshed.data["resume_available"] is False
    assert backend.calls == []
    assert service.sessions == {}


@pytest.mark.parametrize(
    ("container_name", "alias"),
    [
        ("state", "dataset_identity"),
        ("backend_identity", "dataset_identity"),
        ("state", "view_identity"),
        ("state", "training_view_identity"),
        ("state", "dataset_view_manifest_hash"),
        ("backend_identity", "view_identity"),
        ("backend_identity", "training_view_identity"),
        ("backend_identity", "dataset_view_manifest_hash"),
    ],
)
@pytest.mark.parametrize("malformed_kind", ["numeric", "boolean", "list", "padded"])
def test_reconstructed_resume_rejects_malformed_training_identity_alias_before_backend(
    tmp_path: Path,
    container_name: str,
    alias: str,
    malformed_kind: str,
) -> None:
    service, backend, state_path, launch = _durable_product_resume(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    container = state if container_name == "state" else state["backend_identity"]
    reference = launch.receipt.dataset_identity if alias == "dataset_identity" else launch.receipt.view_identity
    malformed: object
    if malformed_kind == "numeric":
        malformed = 7
    elif malformed_kind == "boolean":
        malformed = True
    elif malformed_kind == "list":
        malformed = [reference]
    else:
        malformed = f" {reference} "
    container[alias] = malformed
    _write_json(state_path, state)

    result = service.resume(str(state["run_id"]))

    assert result.status == ProductStatus.UNAVAILABLE
    assert backend.calls == []
    assert service.sessions == {}


class _SyntheticProcess:
    pid = 101

    def poll(self) -> None:
        return None


class _RecordingSSHTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, script: str, payload: dict[str, Any]) -> RemoteResult:
        self.calls.append((script, dict(payload)))
        return RemoteResult(0, json.dumps({"status": "RUNNING", "changed": True, **payload}))

    def upload(self, local_path: Path, remote_path: str) -> RemoteResult:
        self.calls.append(("upload", {"local": str(local_path), "remote": remote_path}))
        return RemoteResult(0)

    def download(self, remote_path: str, local_path: Path) -> RemoteResult:
        self.calls.append(("download", {"remote": remote_path, "local": str(local_path)}))
        return RemoteResult(0)


def _mutate_resume_evidence(root: Path, mutation: str) -> None:
    canonical = root / EVENT_FILENAME
    record_path = root / LEGACY_MIGRATION_FILENAME
    if mutation == "changed_prefix":
        canonical.write_bytes(canonical.read_bytes().replace(b'"stage":"seed"', b'"stage":"reed"', 1))
    elif mutation == "truncated_prefix":
        canonical.write_bytes(canonical.read_bytes()[:-1])
    elif mutation == "replaced_canonical":
        canonical.write_bytes(b"{}\n")
    elif mutation == "malformed_record":
        record_path.write_text("{", encoding="utf-8")
    elif mutation == "missing_record":
        record_path.unlink()
    elif mutation == "conflicting_source":
        (root / LEGACY_EVENT_FILENAME).write_bytes(b"{}\n")
    else:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        field, value = {
            "source_removal_policy": ("legacy_source_policy", "must_be_retained"),
            "source_size": ("legacy_size_bytes", 999),
            "source_hash": ("legacy_sha256", "a" * 64),
            "prefix_hash": ("canonical_prefix_sha256", "b" * 64),
            "canonical_filename": ("canonical_relative_path", "other.jsonl"),
            "run_binding": ("run_id", "other-run"),
        }[mutation]
        record[field] = value
        _write_json(record_path, record)


_LOWEST_SEAM_INVALID_MUTATIONS = (
    "changed_prefix",
    "truncated_prefix",
    "replaced_canonical",
    "malformed_record",
    "missing_record",
    "source_removal_policy",
    "source_size",
    "source_hash",
    "prefix_hash",
    "canonical_filename",
    "run_binding",
    "conflicting_source",
)


@pytest.mark.parametrize("backend_id", ["local", "ssh"])
@pytest.mark.parametrize("mutation", _LOWEST_SEAM_INVALID_MUTATIONS)
def test_receipt_then_event_tamper_reaches_zero_lowest_resume_seams(
    tmp_path: Path, backend_id: str, mutation: str
) -> None:
    launch, checkpoint = _resume_launch(tmp_path, backend_id)
    request = _request(launch)
    context = ProjectContext(tmp_path, {})
    processes: list[list[str]] = []
    transport = _RecordingSSHTransport()
    if backend_id == "local":
        backend: Any = LocalComputeBackend(
            process_factory=lambda command, **_kwargs: processes.append(command) or _SyntheticProcess()
        )
    else:
        backend = SSHComputeBackend(
            SSHSettings("example.test", "trainer", "/workspace/sprite-lab", cloud=False),
            transport=transport,
        )
    prepared = backend.prepare(context, request)
    transport.calls.clear()
    _mutate_resume_evidence(launch.output_root, mutation)
    artifact = ArtifactReference(
        checkpoint.name,
        file_sha256(checkpoint),
        prepared.remote_identity,
        checkpoint,
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )

    with pytest.raises(ComputeBackendError):
        backend.resume(prepared, ResumeRequest(request, artifact, safe_resume=True))

    assert processes == []
    assert transport.calls == []


@pytest.mark.parametrize("backend_id", ["local", "ssh"])
def test_valid_synthetic_continuation_reaches_one_intercepted_lowest_seam(tmp_path: Path, backend_id: str) -> None:
    launch, checkpoint = _resume_launch(tmp_path, backend_id)
    request = _request(launch)
    context = ProjectContext(tmp_path, {})
    processes: list[list[str]] = []
    transport = _RecordingSSHTransport()
    if backend_id == "local":
        backend: Any = LocalComputeBackend(
            process_factory=lambda command, **_kwargs: processes.append(command) or _SyntheticProcess()
        )
    else:
        backend = SSHComputeBackend(
            SSHSettings("example.test", "trainer", "/workspace/sprite-lab", cloud=False),
            transport=transport,
        )
    prepared = backend.prepare(context, request)
    transport.calls.clear()
    artifact = ArtifactReference(
        checkpoint.name,
        file_sha256(checkpoint),
        prepared.remote_identity,
        checkpoint,
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )

    job = backend.resume(prepared, ResumeRequest(request, artifact, safe_resume=True))

    assert job.status == ComputeStatus.RUNNING
    assert len(processes) == (1 if backend_id == "local" else 0)
    assert len(transport.calls) == (1 if backend_id == "ssh" else 0)


def test_receipt_binds_migration_state_and_evidence_identity(tmp_path: Path) -> None:
    launch, _checkpoint = _resume_launch(tmp_path, "local")

    assert launch.receipt.event_migration_state == EventMigrationState.VERIFIED_SOURCE_REMOVED.value
    assert len(launch.receipt.event_migration_identity_sha256) == 64
    assert (
        launch.receipt.event_migration_identity_sha256
        == verify_event_migration(str(launch.run["run_id"]), launch.output_root).evidence_sha256
    )


@pytest.mark.parametrize("mutation", ["changed_prefix", "malformed_record"])
def test_training_service_resume_rejects_removed_source_tamper_before_backend_handoff(
    tmp_path: Path, mutation: str
) -> None:
    launch, _checkpoint = _resume_launch(tmp_path, "fake")
    request = _request(launch)
    context = ProjectContext(tmp_path, {}, runs_directory=tmp_path / "product-runs")
    backend = FakeComputeBackend()
    service = TrainingService(context, backend)
    session_id = "product-training-session"
    dashboard = DashboardState(
        session_id,
        backend.backend_id,
        status=ProductStatus.PAUSED,
        resume_available=True,
    )
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Synthetic continuation",
        1,
        True,
        backend.backend_id,
        dict(launch.campaign),
        (),
        ComputeEstimate(1, 0, trustworthy=True),
    )
    session = TrainingSession(session_id, backend, plan, dashboard=dashboard)
    session.requests[request.run_id] = request
    service.repository.create_run(
        session_id,
        feature="training",
        command="training.start",
        status=ProductStatus.PAUSED.value,
        stage="seed",
        resumable=True,
        backend_id=backend.backend_id,
        backend_identity={
            "dataset_identity": launch.receipt.dataset_identity,
            "view_identity": launch.receipt.view_identity,
            "training_view_identity": launch.receipt.view_identity,
            "dataset_view_manifest_hash": launch.receipt.view_identity,
        },
        extra={
            "plan": _plan_to_state(plan),
            "jobs": [],
            "cursors": {},
            "dataset_identity": launch.receipt.dataset_identity,
            "view_identity": launch.receipt.view_identity,
            "training_view_identity": launch.receipt.view_identity,
        },
    )
    service.sessions[session_id] = session
    _mutate_resume_evidence(launch.output_root, mutation)

    result = service.resume(session_id)

    assert result.status == ProductStatus.BLOCKED
    assert result.data["backend_launches"] == 0
    assert "prepare" not in backend.calls and "resume" not in backend.calls and "launch" not in backend.calls

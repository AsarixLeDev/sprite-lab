"""Authoritative event-history origin: classification, planning, receipts, and seams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spritelab.product_core import ProductEvent, ProductStatus, ProjectContext, strict_json_dumps
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.service import TrainingService, TrainingSession
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
    EVENT_HISTORY_ORIGIN_NATIVE,
    EVENT_HISTORY_ORIGIN_SCHEMA,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    EventMigrationState,
    EventRepository,
    LegacyEventMigrationError,
    record_event_history_origin,
    verify_event_migration,
)
from spritelab.remote_compute import (
    ArtifactReference,
    ComputeBackendError,
    ComputeEstimate,
    ComputeJobRequest,
    ComputeStatus,
    FakeComputeBackend,
    HostedBackendRegistry,
    LocalComputeBackend,
    ResumeRequest,
    RunPodComputeBackend,
    RunPodSettings,
    SSHComputeBackend,
    SSHSettings,
)
from spritelab.remote_compute.contracts import CapabilityUnavailableError, TrainingLaunchRejected
from spritelab.remote_compute.ssh import RemoteResult
from spritelab.training.campaign import (
    RESUME_CHECKPOINT_SCHEMA_VERSION,
    CampaignResumeError,
    CampaignValidationError,
    audit_resume,
    file_sha256,
    stable_hash,
)
from spritelab.training.launch import (
    ValidatedTrainingLaunch,
    load_exact_campaign_configuration,
    prepare_validated_training_launch,
)
from training_launch_test_utils import launch_authorization_verifier, validated_launch

ORIGIN_HASH_EXCLUDED = "record_sha256"

REQUIRED_ORIGIN_FIELDS = {
    "schema_version",
    "run_id",
    "event_history_origin",
    "canonical_event_path",
    "canonical_event_identity_sha256",
    "migration_required",
    "migration_record_path",
    "migration_record_sha256",
    "legacy_source_path",
    "legacy_source_size_bytes",
    "legacy_source_sha256",
    "canonical_prefix_size_bytes",
    "canonical_prefix_sha256",
    "legacy_source_removal_permitted",
    "created_at_utc",
    "record_sha256",
}


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
        message="Synthetic origin evidence.",
        metrics={"optimizer_step": current, "loss": 0.5},
    )
    value = strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return value + (b"\n" if terminal_newline else b"")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rehash_origin(origin: dict[str, Any]) -> dict[str, Any]:
    import hashlib

    payload = {key: value for key, value in origin.items() if key != ORIGIN_HASH_EXCLUDED}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    origin[ORIGIN_HASH_EXCLUDED] = hashlib.sha256(encoded).hexdigest()
    return origin


def _migrated_directory(tmp_path: Path, run_id: str, *, source_removed: bool = False) -> tuple[Path, bytes]:
    directory = tmp_path / run_id
    directory.mkdir(parents=True)
    _write_json(
        directory / "state.json",
        {"schema_version": "spritelab.product.run-state.v1", "run_id": run_id},
    )
    source = _event_bytes(run_id)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    EventRepository(tmp_path).migrate_legacy_events(run_id)
    if source_removed:
        (directory / LEGACY_EVENT_FILENAME).unlink()
    return directory, source


def _resume_root(tmp_path: Path, backend_id: str, *, source_removed: bool = True) -> ValidatedTrainingLaunch:
    """Build a valid migrated resumable output root and return the fresh launch context."""

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
    staging_root = tmp_path / "origin-staging"
    staging, _source = _migrated_directory(staging_root, str(run["run_id"]), source_removed=source_removed)
    for name in (EVENT_FILENAME, LEGACY_MIGRATION_FILENAME, EVENT_HISTORY_ORIGIN_FILENAME):
        (root / name).write_bytes((staging / name).read_bytes())
    if not source_removed:
        (root / LEGACY_EVENT_FILENAME).write_bytes((staging / LEGACY_EVENT_FILENAME).read_bytes())
    record_event_history_origin(
        str(run["run_id"]),
        root,
        expected_origin=EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
        allow_binding_population=True,
    )
    return initial


def _continuation(initial: ValidatedTrainingLaunch, backend_id: str) -> ValidatedTrainingLaunch:
    return prepare_validated_training_launch(
        initial.validator_context.campaign_config_path,
        run_id=str(initial.run["run_id"]),
        compute_backend_id=backend_id,
        project_root=initial.validator_context.project_root,
        execute_confirmed=True,
        resume=True,
    )


def _request(launch: ValidatedTrainingLaunch, *, backend_id: str | None = None) -> ComputeJobRequest:
    return ComputeJobRequest(
        run_id=str(launch.run["run_id"]),
        command=launch.argv,
        idempotency_key=f"{launch.run['run_id']}-origin-continuation",
        campaign_identity=launch.receipt.campaign_identity_sha256,
        run_identity=launch.receipt.run_identity,
        local_project_root=launch.validator_context.project_root,
        output_root=launch.output_root,
        event_path=launch.output_root / EVENT_FILENAME,
        environment=launch.environment,
        execution_spec_identity=launch.receipt.execution_spec_sha256,
        output_root_identity=launch.receipt.output_root_identity,
        launch_authorization_evidence_sha256=launch.receipt.launch_authorization_evidence_sha256,
        compute_backend_id=backend_id or launch.receipt.compute_backend_id,
        launch_receipt=launch.receipt,
        validator_context=launch.validator_context,
        launch_authorization_verifier=launch_authorization_verifier(launch),
    )


def _checkpoint_artifact(launch: ValidatedTrainingLaunch, remote_identity: str) -> ArtifactReference:
    checkpoint = launch.output_root / "checkpoint_step_000005.pt"
    return ArtifactReference(
        checkpoint.name,
        file_sha256(checkpoint),
        remote_identity,
        checkpoint,
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )


class _SyntheticProcess:
    pid = 4242

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


def _mutate_origin_evidence(root: Path, run_id: str, mutation: str) -> None:
    origin_path = root / EVENT_HISTORY_ORIGIN_FILENAME
    record_path = root / LEGACY_MIGRATION_FILENAME
    canonical_path = root / EVENT_FILENAME
    legacy_path = root / LEGACY_EVENT_FILENAME
    state_path = root / "state.json"
    if not state_path.exists():
        state_path = root / "run_identity.json"
    if mutation.startswith("state_"):
        if mutation == "state_malformed":
            state_path.write_text("{", encoding="utf-8")
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if mutation == "state_missing_migration_required":
            state.pop("event_migration_required")
        elif mutation == "state_missing_current_identity":
            state.pop("event_canonical_current_identity_sha256")
        elif mutation == "state_wrong_run_identity":
            state["run_id"] = "other-run"
        elif mutation == "state_changed_canonical_after_current_binding_removed":
            state.pop("event_canonical_current_identity_sha256")
            canonical_path.write_bytes(canonical_path.read_bytes() + _event_bytes(run_id, current=9))
        else:
            raise AssertionError(f"unknown state mutation: {mutation}")
        _write_json(state_path, state)
        return
    if mutation == "changed_canonical_prefix":
        canonical_path.write_bytes(canonical_path.read_bytes().replace(b'"stage":"seed"', b'"stage":"reed"', 1))
        return
    if mutation == "truncated_canonical_prefix":
        canonical_path.write_bytes(canonical_path.read_bytes()[:-1])
        return
    if mutation == "changed_retained_source":
        legacy_path.write_bytes(legacy_path.read_bytes() + b" ")
        return
    if mutation == "malformed_migration_record":
        record_path.write_text("{", encoding="utf-8")
        return
    if mutation == "deleted_migration_record":
        record_path.unlink()
        return
    if mutation == "deleted_origin_record":
        origin_path.unlink()
        return
    if mutation == "downgrade_after_record_deletion":
        record_path.unlink()
    if mutation == "origin_malformed":
        origin_path.write_text("{", encoding="utf-8")
        return
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    if mutation == "origin_swapped_to_native":
        origin.update(
            {
                "event_history_origin": EVENT_HISTORY_ORIGIN_NATIVE,
                "migration_required": False,
                "migration_record_path": None,
                "migration_record_sha256": None,
                "legacy_source_path": None,
                "legacy_source_size_bytes": None,
                "legacy_source_sha256": None,
                "legacy_source_removal_permitted": False,
            }
        )
    elif mutation == "origin_missing_migration_required":
        origin["migration_required"] = False
    elif mutation == "origin_wrong_run_identity":
        origin["run_id"] = "other-run"
    elif mutation == "origin_wrong_prefix_hash":
        origin["canonical_prefix_sha256"] = "b" * 64
        origin["legacy_source_sha256"] = "b" * 64
    elif mutation == "origin_wrong_prefix_size":
        origin["canonical_prefix_size_bytes"] = 7
        origin["legacy_source_size_bytes"] = 7
    elif mutation == "origin_wrong_record_hash":
        origin["migration_record_sha256"] = "c" * 64
    elif mutation == "origin_wrong_canonical_identity":
        origin["canonical_event_identity_sha256"] = "e" * 64
    elif mutation == "downgrade_after_record_deletion":
        origin.update(
            {
                "event_history_origin": EVENT_HISTORY_ORIGIN_NATIVE,
                "migration_required": False,
                "migration_record_path": None,
                "migration_record_sha256": None,
                "legacy_source_path": None,
                "legacy_source_size_bytes": None,
                "legacy_source_sha256": None,
                "legacy_source_removal_permitted": False,
            }
        )
    elif mutation == "origin_stale_self_hash":
        origin["canonical_prefix_sha256"] = "d" * 64
        _write_json(origin_path, origin)
        return
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    _rehash_origin(origin)
    _write_json(origin_path, origin)


ORIGIN_TAMPER_EXPECTATIONS = {
    "deleted_migration_record": EventMigrationState.INVALID_RECORD,
    "deleted_origin_record": EventMigrationState.INVALID_RECORD,
    "origin_malformed": EventMigrationState.INVALID_RECORD,
    "origin_swapped_to_native": EventMigrationState.INVALID_RECORD,
    "origin_missing_migration_required": EventMigrationState.INVALID_RECORD,
    "origin_wrong_run_identity": EventMigrationState.INVALID_RECORD,
    "origin_wrong_prefix_hash": EventMigrationState.INVALID_RECORD,
    "origin_wrong_prefix_size": EventMigrationState.INVALID_RECORD,
    "origin_wrong_record_hash": EventMigrationState.INVALID_RECORD,
    "origin_wrong_canonical_identity": EventMigrationState.INVALID_RECORD,
    "origin_stale_self_hash": EventMigrationState.INVALID_RECORD,
    "downgrade_after_record_deletion": EventMigrationState.INVALID_RECORD,
    "state_malformed": EventMigrationState.INVALID_RECORD,
    "state_missing_current_identity": EventMigrationState.INVALID_RECORD,
    "state_missing_migration_required": EventMigrationState.INVALID_RECORD,
    "state_wrong_run_identity": EventMigrationState.INVALID_RECORD,
    "state_changed_canonical_after_current_binding_removed": EventMigrationState.INVALID_RECORD,
}

POST_RECEIPT_TAMPERS = (
    *sorted(ORIGIN_TAMPER_EXPECTATIONS),
    "changed_canonical_prefix",
    "truncated_canonical_prefix",
    "changed_retained_source",
    "malformed_migration_record",
)


def test_native_and_migrated_origin_records_carry_the_exact_versioned_contract(tmp_path: Path) -> None:
    run_id = "origin-contract"
    repository = EventRepository(tmp_path)
    created = repository.create_run(run_id, feature="training", command="training")
    initial_origin = json.loads((tmp_path / run_id / EVENT_HISTORY_ORIGIN_FILENAME).read_text(encoding="utf-8"))
    assert (tmp_path / run_id / EVENT_FILENAME).read_bytes() == b""
    assert initial_origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_NATIVE
    assert initial_origin["canonical_prefix_size_bytes"] == 0
    assert created["event_history_origin_record_sha256"] == file_sha256(
        tmp_path / run_id / EVENT_HISTORY_ORIGIN_FILENAME
    )
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-14T10:01:00+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=1,
            total=10,
            message="Native origin.",
        )
    )
    native = json.loads((tmp_path / run_id / EVENT_HISTORY_ORIGIN_FILENAME).read_text(encoding="utf-8"))
    assert set(native) == REQUIRED_ORIGIN_FIELDS
    assert native["schema_version"] == EVENT_HISTORY_ORIGIN_SCHEMA
    assert native["event_history_origin"] == EVENT_HISTORY_ORIGIN_NATIVE
    assert native["migration_required"] is False
    assert native["migration_record_path"] is None
    assert native["legacy_source_removal_permitted"] is False
    state = repository.state(run_id)
    assert state["event_history_origin"] == EVENT_HISTORY_ORIGIN_NATIVE
    assert state["event_history_origin_record_sha256"] == file_sha256(tmp_path / run_id / EVENT_HISTORY_ORIGIN_FILENAME)
    assert state["event_migration_required"] is False
    assert state["event_canonical_prefix_sha256"] == native["canonical_prefix_sha256"]

    migrated_dir, source = _migrated_directory(tmp_path / "migrated", "origin-migrated")
    migrated = json.loads((migrated_dir / EVENT_HISTORY_ORIGIN_FILENAME).read_text(encoding="utf-8"))
    assert set(migrated) == REQUIRED_ORIGIN_FIELDS
    assert migrated["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    assert migrated["migration_required"] is True
    assert migrated["migration_record_path"] == LEGACY_MIGRATION_FILENAME
    assert migrated["legacy_source_path"] == LEGACY_EVENT_FILENAME
    assert migrated["legacy_source_size_bytes"] == len(source)
    assert migrated["canonical_prefix_size_bytes"] == len(source)
    assert migrated["legacy_source_removal_permitted"] is True
    assert migrated["migration_record_sha256"] == file_sha256(migrated_dir / LEGACY_MIGRATION_FILENAME)


def test_verification_exposes_receipt_bindings_for_native_and_migrated(tmp_path: Path) -> None:
    present, _source = _migrated_directory(tmp_path / "present", "bind-present")
    removed, _ = _migrated_directory(tmp_path / "removed", "bind-removed", source_removed=True)

    present_result = verify_event_migration("bind-present", present)
    removed_result = verify_event_migration("bind-removed", removed)

    assert present_result.state == EventMigrationState.VERIFIED_SOURCE_PRESENT
    assert removed_result.state == EventMigrationState.VERIFIED_SOURCE_REMOVED
    for result, directory in ((present_result, present), (removed_result, removed)):
        assert result.event_history_origin == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
        assert result.migration_required is True
        assert result.migration_record_sha256 == file_sha256(directory / LEGACY_MIGRATION_FILENAME)
        assert result.canonical_prefix_sha256 is not None
        assert result.canonical_event_identity_sha256 == file_sha256(directory / EVENT_FILENAME)

    repository = EventRepository(tmp_path)
    repository.create_run("bind-native", feature="training", command="training")
    repository.append(
        ProductEvent(
            run_id="bind-native",
            timestamp="2026-07-14T10:02:00+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=1,
            total=10,
            message="Native binding.",
        )
    )
    native_result = verify_event_migration("bind-native", tmp_path / "bind-native")
    assert native_result.state == EventMigrationState.NO_MIGRATION
    assert native_result.event_history_origin == EVENT_HISTORY_ORIGIN_NATIVE
    assert native_result.migration_required is False
    assert native_result.migration_record_sha256 is None
    assert native_result.canonical_event_identity_sha256 == file_sha256(tmp_path / "bind-native" / EVENT_FILENAME)


@pytest.mark.parametrize("mutation", sorted(ORIGIN_TAMPER_EXPECTATIONS))
def test_every_origin_tamper_state_fails_closed_and_is_never_no_migration(tmp_path: Path, mutation: str) -> None:
    run_id = f"tamper-{mutation.replace('_', '-')}"
    directory, _ = _migrated_directory(tmp_path, run_id, source_removed=True)

    _mutate_origin_evidence(directory, run_id, mutation)
    result = verify_event_migration(run_id, directory)

    assert result.state == ORIGIN_TAMPER_EXPECTATIONS[mutation]
    assert result.state != EventMigrationState.NO_MIGRATION
    assert not result.resume_compatible
    replay = EventRepository(tmp_path).replay(run_id)
    assert replay.integrity_status in {"NOT_COMPARABLE", "STALE"}
    assert not replay.safe_for_resume
    if mutation == "deleted_origin_record":
        assert not (directory / EVENT_HISTORY_ORIGIN_FILENAME).exists()
        assert verify_event_migration(run_id, directory).state == EventMigrationState.INVALID_RECORD


def test_changed_and_truncated_canonical_prefix_fail_with_origin_present(tmp_path: Path) -> None:
    changed, source = _migrated_directory(tmp_path / "changed", "prefix-changed", source_removed=True)
    (changed / EVENT_FILENAME).write_bytes(source.replace(b'"stage":"seed"', b'"stage":"reed"', 1))
    assert verify_event_migration("prefix-changed", changed).state == EventMigrationState.INVALID_RECORD

    truncated, source_two = _migrated_directory(tmp_path / "truncated", "prefix-truncated", source_removed=True)
    (truncated / EVENT_FILENAME).write_bytes(source_two[:-1])
    assert verify_event_migration("prefix-truncated", truncated).state == EventMigrationState.INVALID_RECORD


def test_changed_retained_source_remains_stale_with_origin_present(tmp_path: Path) -> None:
    directory, source = _migrated_directory(tmp_path, "retained-changed")
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source + b" ")
    assert verify_event_migration("retained-changed", directory).state == EventMigrationState.STALE_SOURCE_CHANGED


def test_native_origin_prefix_tamper_blocks_native_stream(tmp_path: Path) -> None:
    run_id = "native-prefix"
    repository = EventRepository(tmp_path)
    repository.create_run(run_id, feature="training", command="training")
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-14T10:03:00+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=1,
            total=10,
            message="Native prefix.",
        )
    )
    directory = tmp_path / run_id
    canonical = directory / EVENT_FILENAME
    canonical.write_bytes(_event_bytes(run_id, current=9))

    result = verify_event_migration(run_id, directory)

    assert result.state == EventMigrationState.INVALID_RECORD
    assert not result.resume_compatible


def test_deleted_native_origin_cannot_fall_back_to_no_migration(tmp_path: Path) -> None:
    run_id = "native-origin-deleted"
    repository = EventRepository(tmp_path)
    repository.create_run(run_id, feature="training", command="training")
    repository.append(
        ProductEvent(
            run_id=run_id,
            timestamp="2026-07-14T10:03:30+00:00",
            feature="training",
            stage="seed",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=1,
            total=10,
            message="Native origin deletion probe.",
        )
    )
    directory = tmp_path / run_id
    (directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()

    result = verify_event_migration(run_id, directory)
    replay = repository.replay(run_id)

    assert result.state == EventMigrationState.INVALID_RECORD
    assert result.state != EventMigrationState.NO_MIGRATION
    assert not result.resume_compatible
    assert not replay.safe_for_resume
    assert not (directory / EVENT_HISTORY_ORIGIN_FILENAME).exists()


@pytest.mark.parametrize("mutation", POST_RECEIPT_TAMPERS)
def test_pre_receipt_invalid_origin_evidence_blocks_planning_and_receipt_creation(
    tmp_path: Path, mutation: str
) -> None:
    initial = _resume_root(tmp_path, "local", source_removed=mutation != "changed_retained_source")
    root = initial.output_root
    run_id = str(initial.run["run_id"])
    _mutate_origin_evidence(root, run_id, mutation)

    direct = verify_event_migration(run_id, root)
    assert direct.state != EventMigrationState.NO_MIGRATION
    assert not direct.resume_compatible

    campaign = load_exact_campaign_configuration(initial.validator_context.campaign_config_path)
    resume_report = audit_resume(campaign)
    state = next(item for item in resume_report["runs"] if item["run_id"] == run_id)
    if "event_migration_state" in state:
        assert state["event_migration_state"] == direct.state.value
    assert state["status"] in {"partial_invalid", "corrupt", "foreign"}
    assert state["next_action"] == "refuse"
    assert resume_report["safe"] is False

    with pytest.raises((CampaignResumeError, CampaignValidationError)):
        _continuation(initial, "local")


@pytest.mark.parametrize("backend_id", ["local", "ssh", "plugin-fake"])
def test_deleted_migration_record_before_receipt_blocks_every_continuation_adapter(
    tmp_path: Path, backend_id: str
) -> None:
    initial = _resume_root(tmp_path, backend_id, source_removed=True)
    root = initial.output_root
    run_id = str(initial.run["run_id"])
    (root / LEGACY_MIGRATION_FILENAME).unlink()

    verification = verify_event_migration(run_id, root, migration_required=True, origin_required=True)
    campaign = load_exact_campaign_configuration(initial.validator_context.campaign_config_path)
    resume_report = audit_resume(campaign)

    assert verification.state is EventMigrationState.INVALID_RECORD
    assert verification.state is not EventMigrationState.NO_MIGRATION
    assert not verification.resume_compatible
    assert resume_report["safe"] is False
    with pytest.raises((CampaignResumeError, CampaignValidationError)):
        _continuation(initial, backend_id)


@pytest.mark.parametrize("backend_id", ["local", "ssh"])
@pytest.mark.parametrize("mutation", POST_RECEIPT_TAMPERS)
def test_receipt_then_origin_tamper_reaches_zero_local_and_ssh_seams(
    tmp_path: Path, backend_id: str, mutation: str
) -> None:
    initial = _resume_root(tmp_path, backend_id, source_removed=mutation != "changed_retained_source")
    launch = _continuation(initial, backend_id)
    assert launch.receipt.event_history_origin == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    assert launch.receipt.event_migration_required is True
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
    _mutate_origin_evidence(launch.output_root, str(launch.run["run_id"]), mutation)
    artifact = _checkpoint_artifact(launch, prepared.remote_identity)

    with pytest.raises(ComputeBackendError):
        backend.resume(prepared, ResumeRequest(request, artifact, safe_resume=True))

    assert processes == []
    assert transport.calls == []


@pytest.mark.parametrize("mutation", POST_RECEIPT_TAMPERS)
def test_receipt_then_origin_tamper_reaches_zero_plugin_seams(tmp_path: Path, mutation: str) -> None:
    initial = _resume_root(tmp_path, "plugin-fake", source_removed=mutation != "changed_retained_source")
    launch = _continuation(initial, "plugin-fake")
    inner = FakeComputeBackend(is_cloud=True)
    inner.backend_id = "plugin-fake"
    registry = HostedBackendRegistry([inner])
    backend = registry.get("plugin-fake")
    assert backend is not None
    request = _request(launch)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    inner.calls.clear()
    _mutate_origin_evidence(launch.output_root, str(launch.run["run_id"]), mutation)
    artifact = _checkpoint_artifact(launch, prepared.remote_identity)

    with pytest.raises(TrainingLaunchRejected):
        backend.resume(prepared, ResumeRequest(request, artifact, safe_resume=True), cloud_confirmation=True)

    assert inner.calls == []


def test_valid_plugin_continuation_reaches_one_intercepted_seam(tmp_path: Path) -> None:
    initial = _resume_root(tmp_path, "plugin-fake", source_removed=True)
    launch = _continuation(initial, "plugin-fake")
    inner = FakeComputeBackend(is_cloud=True)
    inner.backend_id = "plugin-fake"
    registry = HostedBackendRegistry([inner])
    backend = registry.get("plugin-fake")
    assert backend is not None
    request = _request(launch)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    inner.calls.clear()
    artifact = _checkpoint_artifact(launch, prepared.remote_identity)

    job = backend.resume(prepared, ResumeRequest(request, artifact, safe_resume=True), cloud_confirmation=True)

    assert job.status == ComputeStatus.RUNNING
    assert inner.calls == ["resume", "launch"]


@pytest.mark.parametrize("mutation", POST_RECEIPT_TAMPERS)
def test_runpod_scaffold_verifies_receipts_before_reporting_unavailability(tmp_path: Path, mutation: str) -> None:
    initial = _resume_root(tmp_path, "runpod", source_removed=mutation != "changed_retained_source")
    launch = _continuation(initial, "runpod")
    backend = RunPodComputeBackend(RunPodSettings(gpu_type_ids=("GPU",), image_name="image:tag"))
    request = _request(launch)

    _mutate_origin_evidence(launch.output_root, str(launch.run["run_id"]), mutation)
    with pytest.raises(TrainingLaunchRejected):
        backend.prepare(ProjectContext(tmp_path, {}), request)


def test_runpod_scaffold_accepts_valid_receipt_then_remains_unavailable(tmp_path: Path) -> None:
    initial = _resume_root(tmp_path, "runpod", source_removed=True)
    launch = _continuation(initial, "runpod")
    backend = RunPodComputeBackend(RunPodSettings(gpu_type_ids=("GPU",), image_name="image:tag"))

    with pytest.raises(CapabilityUnavailableError):
        backend.prepare(ProjectContext(tmp_path, {}), _request(launch))


def test_receipt_binds_origin_migration_and_canonical_identities(tmp_path: Path) -> None:
    initial = _resume_root(tmp_path, "local", source_removed=True)
    launch = _continuation(initial, "local")
    root = launch.output_root

    assert launch.receipt.schema_version == "spritelab_training_launch_receipt_v4"
    assert launch.receipt.event_migration_state == EventMigrationState.VERIFIED_SOURCE_REMOVED.value
    assert launch.receipt.event_history_origin == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    assert launch.receipt.event_migration_required is True
    assert launch.receipt.event_migration_record_sha256 == file_sha256(root / LEGACY_MIGRATION_FILENAME)
    origin = json.loads((root / EVENT_HISTORY_ORIGIN_FILENAME).read_text(encoding="utf-8"))
    assert launch.receipt.event_canonical_prefix_sha256 == origin["canonical_prefix_sha256"]
    assert launch.receipt.event_canonical_identity_sha256 == file_sha256(root / EVENT_FILENAME)


def test_fresh_native_launch_receipt_binds_native_origin(tmp_path: Path) -> None:
    launch = validated_launch(tmp_path, "local")

    assert launch.receipt.event_migration_state == EventMigrationState.NO_MIGRATION.value
    assert launch.receipt.event_history_origin == EVENT_HISTORY_ORIGIN_NATIVE
    assert launch.receipt.event_migration_required is False
    assert launch.receipt.event_migration_record_sha256 is None


@pytest.mark.parametrize(
    "mutation",
    POST_RECEIPT_TAMPERS,
)
def test_product_service_resume_rejects_origin_tamper_before_backend_handoff(tmp_path: Path, mutation: str) -> None:
    initial = _resume_root(tmp_path, "fake", source_removed=mutation != "changed_retained_source")
    launch = _continuation(initial, "fake")
    request = _request(launch)
    context = ProjectContext(tmp_path, {}, runs_directory=tmp_path / "product-runs")
    backend = FakeComputeBackend()
    service = TrainingService(context, backend)
    session_id = "product-origin-session"
    dashboard = DashboardState(
        session_id,
        backend.backend_id,
        status=ProductStatus.PAUSED,
        resume_available=True,
    )
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Synthetic origin continuation",
        1,
        True,
        backend.backend_id,
        dict(launch.campaign),
        (),
        ComputeEstimate(1, 0, trustworthy=True),
    )
    session = TrainingSession(session_id, backend, plan, dashboard=dashboard)
    session.requests[request.run_id] = request
    service.sessions[session_id] = session
    _mutate_origin_evidence(launch.output_root, str(launch.run["run_id"]), mutation)

    result = service.resume(session_id)

    assert result.status == ProductStatus.BLOCKED
    assert result.data["backend_launches"] == 0
    assert "prepare" not in backend.calls and "resume" not in backend.calls and "launch" not in backend.calls


def test_record_event_history_origin_cannot_be_inferred_and_is_idempotent(tmp_path: Path) -> None:
    run_id = "origin-guard"
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / LEGACY_EVENT_FILENAME).write_bytes(_event_bytes(run_id))
    with pytest.raises(LegacyEventMigrationError, match="cannot be reconstructed"):
        record_event_history_origin(run_id, directory)

    EventRepository(tmp_path).migrate_legacy_events(run_id)
    first = json.loads((directory / EVENT_HISTORY_ORIGIN_FILENAME).read_text(encoding="utf-8"))
    second = record_event_history_origin(run_id, directory)
    assert second == first


def test_default_origin_recorder_cannot_heal_a_fully_stripped_downgrade(tmp_path: Path) -> None:
    run_id = "origin-stripped-downgrade"
    directory, _source = _migrated_directory(tmp_path, run_id, source_removed=True)
    _mutate_origin_evidence(directory, run_id, "downgrade_after_record_deletion")
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in (
        "event_history_origin",
        "event_history_origin_record_sha256",
        "event_migration_required",
        "event_migration_record_sha256",
        "event_canonical_prefix_sha256",
        "event_canonical_origin_identity_sha256",
        "event_canonical_current_identity_sha256",
    ):
        state.pop(field)
    _write_json(state_path, state)

    with pytest.raises(LegacyEventMigrationError, match="missing all event-history origin bindings"):
        record_event_history_origin(run_id, directory)

    verification = verify_event_migration(run_id, directory)
    assert verification.state == EventMigrationState.INVALID_RECORD
    assert verification.state != EventMigrationState.NO_MIGRATION


def test_append_is_blocked_for_every_origin_tamper_state(tmp_path: Path) -> None:
    run_id = "append-blocked"
    repository = EventRepository(tmp_path)
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / LEGACY_EVENT_FILENAME).write_bytes(_event_bytes(run_id))
    follow_up = ProductEvent(
        run_id=run_id,
        timestamp="2026-07-14T10:04:00+00:00",
        feature="training",
        stage="seed",
        event_type="progress",
        status=ProductStatus.RUNNING,
        current=2,
        total=10,
        message="Bound origin.",
    )
    repository.append(follow_up)
    (directory / LEGACY_EVENT_FILENAME).unlink()
    (directory / LEGACY_MIGRATION_FILENAME).unlink()
    canonical_before = (directory / EVENT_FILENAME).read_bytes()

    with pytest.raises(ValueError, match="record is missing"):
        repository.append(follow_up)

    assert (directory / EVENT_FILENAME).read_bytes() == canonical_before

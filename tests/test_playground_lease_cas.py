from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from spritelab.product_features.evaluation import local_generator as module
from spritelab.product_features.evaluation.local_generator import (
    LocalCheckpointPlaygroundGenerator,
    LocalPlaygroundGenerationError,
)
from spritelab.utils.safe_fs import AnchoredDirectory


def _generator(root: Path) -> LocalCheckpointPlaygroundGenerator:
    return LocalCheckpointPlaygroundGenerator(
        project_root=root,
        work_root=root / "runs" / "playground-sampler-work",
        sampler=lambda _config: {},
    )


def _acquire(generator: LocalCheckpointPlaygroundGenerator) -> str:
    return generator._acquire_lease(operation_check=lambda: None)


def _read(generator: LocalCheckpointPlaygroundGenerator) -> dict[str, object]:
    return json.loads(generator._lease_path.read_text(encoding="utf-8"))


def _publish(path: Path, value: dict[str, object]) -> None:
    path.write_bytes((json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def test_lease_v2_has_exact_identity_owner_and_transition_chain(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    lease_id = _acquire(generator)
    acquired = _read(generator)

    assert set(acquired) == {
        "schema_version",
        "lease_id",
        "lease_identity",
        "transition_sequence",
        "prior_lease_identity",
        "status",
        "owner",
        "acquired_at",
        "heartbeat_at",
        "ended_at",
        "retryable",
        "invocation_id",
        "recovered_orphan",
    }
    assert acquired["schema_version"] == module._PLAYGROUND_LEASE_SCHEMA
    assert acquired["lease_id"] == lease_id
    assert acquired["transition_sequence"] == 0
    assert acquired["prior_lease_identity"] is None
    assert acquired["status"] == "ACTIVE"
    assert acquired["retryable"] is False
    assert acquired["ended_at"] is None
    assert set(acquired["owner"]) == {"pid", "process_birth_identity"}
    assert acquired["owner"]["pid"] == os.getpid()
    assert acquired["owner"]["process_birth_identity"] == module._process_birth_identity(os.getpid())
    assert acquired["lease_identity"] == module._record_identity(acquired, "lease_identity")

    generator._update_lease(lease_id, invocation_id="invocation-01")
    active = _read(generator)
    assert active["transition_sequence"] == 1
    assert active["prior_lease_identity"] == acquired["lease_identity"]
    assert active["invocation_id"] == "invocation-01"
    assert active["lease_identity"] == module._record_identity(active, "lease_identity")

    assert generator._release_lease(lease_id, status="COMPLETE", retryable=False) is True
    complete = _read(generator)
    assert complete["transition_sequence"] == 2
    assert complete["prior_lease_identity"] == active["lease_identity"]
    assert complete["status"] == "COMPLETE"
    assert complete["retryable"] is False
    assert complete["ended_at"] == complete["heartbeat_at"]


def test_repeated_lease_heartbeats_remain_single_link_and_leave_no_writer_residue(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    lease_id = _acquire(generator)

    for _index in range(25):
        generator._update_lease(lease_id)
    assert generator._release_lease(lease_id, status="COMPLETE", retryable=False) is True

    lease_metadata = generator._lease_path.stat(follow_symlinks=False)
    assert lease_metadata.st_nlink == 1
    assert {path.name for path in generator.work_root.iterdir()} == {
        generator._lease_path.name,
        generator._lease_lock_path.name,
    }
    state = _read(generator)
    assert state["transition_sequence"] == 26
    assert state["status"] == "COMPLETE"


@pytest.mark.skipif(os.name != "nt", reason="Windows no-delete directory anchors are platform-specific.")
def test_lease_publication_retains_the_exact_work_root_against_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator(tmp_path)
    moved = tmp_path / "work-root-moved"
    outside = tmp_path / "outside-lease-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original = AnchoredDirectory.atomic_write_bytes
    rename_denied = False

    def race(anchor: AnchoredDirectory, name: str, content: bytes) -> Path:
        nonlocal rename_denied
        if anchor.directory == generator.work_root and name == generator._lease_path.name:
            with pytest.raises(OSError):
                os.replace(generator.work_root, moved)
            rename_denied = True
        return original(anchor, name, content)

    monkeypatch.setattr(AnchoredDirectory, "atomic_write_bytes", race)

    lease_id = _acquire(generator)

    assert lease_id
    assert rename_denied is True
    assert generator.work_root.is_dir()
    assert not moved.exists()
    assert sentinel.read_bytes() == b"outside-byte-identical"


def test_lease_protects_immutable_fields_and_terminal_state(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    lease_id = _acquire(generator)

    with pytest.raises(LocalPlaygroundGenerationError, match="Protected"):
        generator._update_lease(lease_id, owner={"pid": os.getpid()})
    generator._update_lease(lease_id, invocation_id="invocation-01")
    with pytest.raises(LocalPlaygroundGenerationError, match="immutable"):
        generator._update_lease(lease_id, invocation_id="invocation-02")

    assert generator._release_lease(lease_id, status="COMPLETE", retryable=False) is True
    terminal = generator._lease_path.read_bytes()
    assert generator._release_lease(lease_id, status="FAILED", retryable=True) is False
    with pytest.raises(LocalPlaygroundGenerationError, match="lost or stale"):
        generator._update_lease(lease_id)
    assert generator._lease_path.read_bytes() == terminal


def test_stale_heartbeat_refuses_sequence_rollback(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    lease_id = _acquire(generator)
    original = _read(generator)
    generator._update_lease(lease_id)

    _publish(generator._lease_path, original)
    with pytest.raises(LocalPlaygroundGenerationError, match="lost or stale"):
        generator._update_lease(lease_id)
    assert _read(generator) == original


def test_stale_heartbeat_refuses_same_sequence_foreign_identity_and_owner(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    lease_id = _acquire(generator)
    current = _read(generator)
    forged = dict(current)
    forged["owner"] = {
        "pid": os.getpid(),
        "process_birth_identity": "portable-instance:" + "f" * 64,
    }
    forged["lease_identity"] = ""
    forged["lease_identity"] = module._record_identity(forged, "lease_identity")
    _publish(generator._lease_path, forged)

    with pytest.raises(LocalPlaygroundGenerationError, match="lost or stale"):
        generator._update_lease(lease_id)
    assert _read(generator) == forged


@pytest.mark.parametrize("foreign_action", ("cancel", "heartbeat"))
def test_two_adapters_race_from_one_prior_identity_under_filesystem_cas(
    tmp_path: Path,
    foreign_action: str,
) -> None:
    owner = _generator(tmp_path)
    foreign = _generator(tmp_path)
    lease_id = _acquire(owner)
    initial = _read(owner)
    # A second adapter is not entitled to reconstruct an in-flight cursor. Give
    # it the same stale snapshot deliberately to exercise the file-lock CAS.
    foreign._lease_cursors[lease_id] = owner._lease_cursors[lease_id]
    barrier = threading.Barrier(2)
    outcomes: dict[str, bool] = {}

    def complete() -> None:
        barrier.wait()
        outcomes["complete"] = owner._release_lease(lease_id, status="COMPLETE", retryable=False)

    def foreign_write() -> None:
        barrier.wait()
        if foreign_action == "cancel":
            outcomes["foreign"] = foreign._release_lease(lease_id, status="FAILED", retryable=True)
            return
        try:
            foreign._update_lease(lease_id)
        except LocalPlaygroundGenerationError:
            outcomes["foreign"] = False
        else:
            outcomes["foreign"] = True

    threads = [threading.Thread(target=complete), threading.Thread(target=foreign_write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(outcomes.values()) == 1
    state = _read(owner)
    assert state["transition_sequence"] == initial["transition_sequence"] + 1
    assert state["prior_lease_identity"] == initial["lease_identity"]
    assert state["lease_identity"] == module._record_identity(state, "lease_identity")
    if outcomes["complete"]:
        assert state["status"] == "COMPLETE"
    elif foreign_action == "cancel":
        assert state["status"] == "FAILED"
    else:
        assert state["status"] == "ACTIVE"


def test_pid_reuse_recovers_v2_owner_and_chains_prior_identity(tmp_path: Path, monkeypatch) -> None:
    first = _generator(tmp_path)
    _acquire(first)
    abandoned = _read(first)
    simulated_birth = "windows-filetime:0000000000000001"
    monkeypatch.setattr(module, "_process_birth_identity", lambda pid: simulated_birth if pid == os.getpid() else None)

    second = _generator(tmp_path)
    lease_id = _acquire(second)
    recovered = _read(second)
    assert recovered["lease_id"] == lease_id
    assert recovered["transition_sequence"] == abandoned["transition_sequence"] + 1
    assert recovered["prior_lease_identity"] == abandoned["lease_identity"]
    assert recovered["owner"] == {"pid": os.getpid(), "process_birth_identity": simulated_birth}
    assert recovered["recovered_orphan"] == {
        "lease_id": abandoned["lease_id"],
        "retryable": True,
        "status": "ORPHANED",
    }


def test_legacy_v1_is_only_blocked_or_fail_closed_migrated(tmp_path: Path) -> None:
    generator = _generator(tmp_path)
    generator.work_root.mkdir(parents=True)
    legacy = {
        "schema_version": "spritelab.playground-sampler-lease.v1",
        "lease_id": "legacy-terminal",
        "status": "COMPLETE",
        "owner_pid": os.getpid(),
    }
    generator._lease_path.write_text(json.dumps(legacy), encoding="utf-8")

    lease_id = _acquire(generator)
    migrated = _read(generator)
    assert migrated["lease_id"] == lease_id
    assert migrated["schema_version"] == module._PLAYGROUND_LEASE_SCHEMA
    assert migrated["transition_sequence"] == 0
    assert migrated["prior_lease_identity"] is None
    assert "legacy-terminal" not in generator._lease_cursors

    live_root = tmp_path / "live"
    live = _generator(live_root)
    live.work_root.mkdir(parents=True)
    live._lease_path.write_text(
        json.dumps(
            {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": "legacy-live",
                "status": "ACTIVE",
                "owner_pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="already active"):
        _acquire(live)
    assert live._lease_cursors == {}


@pytest.mark.parametrize(
    ("field", "value"),
    (("transition_sequence", True), ("retryable", 1), ("unexpected", "field")),
)
def test_lease_v2_rejects_non_exact_or_bool_as_int_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    generator = _generator(tmp_path)
    _acquire(generator)
    malformed = _read(generator)
    malformed[field] = value
    malformed["lease_identity"] = ""
    malformed["lease_identity"] = module._record_identity(malformed, "lease_identity")
    _publish(generator._lease_path, malformed)

    with pytest.raises(LocalPlaygroundGenerationError, match="malformed"):
        with generator._anchored_work_root() as anchor:
            module._read_lease(anchor, generator._lease_path.name)

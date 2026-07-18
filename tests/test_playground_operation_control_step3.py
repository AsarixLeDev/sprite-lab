from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

import spritelab.product_features.evaluation.local_generator as local_generator
from spritelab.product_features.evaluation.local_generator import LocalCheckpointPlaygroundGenerator
from spritelab.product_features.evaluation.playground import (
    GenerationCancelledError,
    GenerationTimedOutError,
)
from spritelab.utils.safe_fs import AnchoredDirectory


def _checkpoint(project_root: Path) -> Path:
    import torch

    path = project_root / "runs" / "checkpoint.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_type": "generator_challenger",
            "ema_weights": False,
            "step": 7,
            "global_step": 7,
        },
        path,
    )
    return path


def _sampler(config):
    prompt = json.loads(config.prompts.read_text(encoding="utf-8"))
    relative = "indexed_png/sample_0000.png"
    (config.out_dir / "indexed_png").mkdir()
    Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(config.out_dir / relative)
    row = {
        **prompt,
        "sample_id": "sample_000000",
        "seed": config.seed,
        "noise_seed": config.noise_seed,
        "model_type": "generator_challenger",
        "steps": config.steps,
        "cfg_scale": config.cfg_scale,
        "paths": {"indexed_png": relative},
    }
    (config.out_dir / "generated_manifest.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"sample_count": 1}


def _generator(project_root: Path) -> LocalCheckpointPlaygroundGenerator:
    return LocalCheckpointPlaygroundGenerator(
        project_root=project_root,
        work_root=project_root / "runs" / "playground-work",
        sampler=_sampler,
    )


def _generate(generator: LocalCheckpointPlaygroundGenerator, checkpoint: Path):
    return generator.generate(
        checkpoint=checkpoint,
        prompt="small shield",
        seed=1,
        sampling_steps=2,
        guidance=2.0,
        image_count=1,
        weights="live",
        expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        expected_step=7,
        expected_variant="live",
    )


def _prepare(generator: LocalCheckpointPlaygroundGenerator, run_id: str) -> None:
    deadline = datetime.now(timezone.utc) + timedelta(minutes=2)
    generator.prepare_control(run_id, deadline.isoformat())


def test_prepared_code_inventory_observes_cancellation_inside_scan(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    run_id = "cancel-code-inventory"
    _prepare(generator, run_id)
    observed = False

    def cancelled_inventory(project_root: Path, *, operation_check):
        nonlocal observed
        assert project_root == tmp_path
        assert generator.cancel(run_id) is True
        observed = True
        operation_check()
        raise AssertionError("cancelled inventory continued")

    monkeypatch.setattr(
        local_generator,
        "_operation_checked_training_code_identity_source_paths",
        cancelled_inventory,
    )
    try:
        with pytest.raises(GenerationCancelledError, match="cancelled"):
            _ = generator.code_identity_sha256
    finally:
        generator.finish_control(run_id)

    assert observed is True
    assert not generator.work_root.exists()


def test_prepared_code_inventory_honors_monotonic_deadline_before_scan(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    run_id = "expired-code-inventory"
    _prepare(generator, run_id)
    prepared_run_id, deadline, _monotonic_deadline, cancel_event = generator._control_local.value
    generator._control_local.value = (prepared_run_id, deadline, time.monotonic() - 1.0, cancel_event)
    called = False

    def unexpected_inventory(project_root: Path, *, operation_check):
        nonlocal called
        called = True
        return ()

    monkeypatch.setattr(
        local_generator,
        "_operation_checked_training_code_identity_source_paths",
        unexpected_inventory,
    )
    try:
        with pytest.raises(GenerationTimedOutError, match="deadline"):
            _ = generator.code_identity_sha256
    finally:
        generator.finish_control(run_id)

    assert called is False


def test_code_inventory_rejects_scan_to_hash_inode_substitution_without_touching_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src/spritelab/product_features/evaluation/local_generator.py"
    worker = source.with_name("playground_worker.py")
    runtime = tmp_path / "src/spritelab/utils/runtime_closure.py"
    confinement = tmp_path / "src/spritelab/utils/write_confinement.py"
    for path, payload in (
        (source, b"source-A\n"),
        (worker, b"worker-A\n"),
        (runtime, b"runtime-A\n"),
        (confinement, b"confinement-A\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    outside = tmp_path / "outside-sentinel.bin"
    outside.write_bytes(b"outside-byte-identical")
    parked = source.with_suffix(".owned")
    replacement = source.with_suffix(".foreign")
    replacement.write_bytes(b"source-B\n")
    generator = _generator(tmp_path)
    monkeypatch.setattr(local_generator, "__file__", str(source))

    def substitute_after_scan(project_root: Path, *, operation_check):
        assert project_root == tmp_path
        if operation_check is not None:
            operation_check()
        os.replace(source, parked)
        os.replace(replacement, source)
        return (source, worker, runtime, confinement)

    monkeypatch.setattr(
        local_generator,
        "_operation_checked_training_code_identity_source_paths",
        substitute_after_scan,
    )

    with pytest.raises(local_generator.LocalPlaygroundGenerationError, match="changed after inventory"):
        generator._code_inventory(lambda: None)

    assert source.read_bytes() == b"source-B\n"
    assert parked.read_bytes() == b"source-A\n"
    assert outside.read_bytes() == b"outside-byte-identical"


def test_checkpoint_snapshot_checks_cancellation_between_hash_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "snapshot.pt"
    source_bytes = b"a" * (2 * 1024 * 1024 + 17)
    source.write_bytes(source_bytes)
    calls = 0

    def cancel_after_first_chunk() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise GenerationCancelledError("cancel checkpoint copy")

    with pytest.raises(GenerationCancelledError, match="checkpoint copy"):
        LocalCheckpointPlaygroundGenerator._snapshot_checkpoint(
            source,
            destination,
            expected_sha256=sha256(source_bytes).hexdigest(),
            operation_check=cancel_after_first_chunk,
        )

    assert calls == 4
    assert not destination.exists()
    assert source.read_bytes() == source_bytes


def test_anchored_read_checks_cancellation_between_chunks(tmp_path: Path) -> None:
    content = b"b" * (2 * 1024 * 1024 + 31)
    path = tmp_path / "result.json"
    path.write_bytes(content)
    calls = 0

    def cancel_after_first_chunk() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise GenerationCancelledError("cancel anchored read")

    with AnchoredDirectory(tmp_path, tmp_path) as anchor:
        with pytest.raises(GenerationCancelledError, match="anchored read"):
            local_generator._read_anchored_regular_bytes(
                anchor,
                path.name,
                maximum_bytes=len(content),
                label="hostile result",
                operation_check=cancel_after_first_chunk,
            )

    assert calls == 4
    assert path.read_bytes() == content


def test_cancellation_after_worker_result_read_preempts_report_scan(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    run_id = "cancel-after-worker-result-read"
    _prepare(generator, run_id)
    observed = False

    def racing_read(anchor, name, *, maximum_bytes, label, operation_check=None):
        nonlocal observed
        assert name == "sampler-result.json"
        assert maximum_bytes == 16 * 1024 * 1024
        assert label == "contained sampler result"
        assert operation_check is not None
        assert generator.cancel(run_id) is True
        observed = True
        return b"{}"

    monkeypatch.setattr(local_generator, "_read_anchored_regular_bytes", racing_read)
    try:
        operation_check = generator._prepared_operation_check()
        assert operation_check is not None
        with AnchoredDirectory(tmp_path, tmp_path) as anchor:
            with pytest.raises(GenerationCancelledError, match="cancelled"):
                generator._read_contained_sampler_result(
                    anchor,
                    "sampler-result.json",
                    operation_check=operation_check,
                )
    finally:
        generator.finish_control(run_id)

    assert observed is True


def test_deadline_while_waiting_for_process_activation_terminates_child() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 123
            self.terminated = threading.Event()

        def poll(self):
            return 0 if self.terminated.is_set() else None

        def terminate(self) -> None:
            self.terminated.set()

        def wait(self, timeout=None) -> int:
            assert timeout == 5
            return 0

        def kill(self) -> None:
            self.terminated.set()

    process = FakeProcess()
    checks = 0

    def activation() -> None:
        assert process.terminated.wait(timeout=2.0)

    def operation_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise GenerationTimedOutError("activation deadline")

    with pytest.raises(GenerationTimedOutError, match="activation deadline"):
        local_generator._wait_for_process_activation(
            process,
            activate=activation,
            operation_check=operation_check,
        )

    assert checks == 2
    assert process.terminated.is_set()


def test_interrupted_operation_wait_is_bounded_and_reaps_late_value(monkeypatch) -> None:
    release = threading.Event()
    cleaned = threading.Event()
    value = object()
    observed: dict[str, object] = {}
    checks = 0

    def action() -> object:
        release.wait()
        return value

    def operation_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise GenerationCancelledError("bounded operation cancellation")

    def cleanup(result: object, interruption: BaseException) -> None:
        observed["result"] = result
        observed["interruption"] = interruption
        cleaned.set()

    monkeypatch.setattr(local_generator, "_INTERRUPTED_CLEANUP_WAIT_SECONDS", 0.02)
    started_at = time.monotonic()
    with pytest.raises(GenerationCancelledError, match="bounded operation cancellation"):
        local_generator._wait_for_operation_call(
            action,
            operation_check=operation_check,
            interrupted_cleanup=cleanup,
        )
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    release.set()
    assert cleaned.wait(timeout=1.0)
    assert observed["result"] is value
    assert isinstance(observed["interruption"], GenerationCancelledError)


def test_interrupted_activation_bounds_cleanup_and_reaps_late_handle(monkeypatch) -> None:
    activation_release = threading.Event()
    termination_release = threading.Event()
    termination_started = threading.Event()
    terminated = threading.Event()
    handle_closed = threading.Event()
    closed_handles: list[int] = []
    checks = 0

    class FakeProcess:
        pass

    process = FakeProcess()

    def activation() -> int:
        activation_release.wait()
        return 8675309

    def operation_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise GenerationTimedOutError("bounded activation deadline")

    def terminate(_process) -> None:  # type: ignore[no-untyped-def]
        assert _process is process
        termination_started.set()
        termination_release.wait()
        terminated.set()

    def close_handle(handle: int) -> None:
        closed_handles.append(handle)
        handle_closed.set()

    monkeypatch.setattr(local_generator, "_INTERRUPTED_CLEANUP_WAIT_SECONDS", 0.02)
    monkeypatch.setattr(local_generator, "_terminate_contained_process", terminate)
    monkeypatch.setattr(local_generator, "close_windows_handle", close_handle)

    started_at = time.monotonic()
    with pytest.raises(GenerationTimedOutError, match="bounded activation deadline"):
        local_generator._wait_for_process_activation(
            process,  # type: ignore[arg-type]
            activate=activation,
            operation_check=operation_check,
        )
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert termination_started.wait(timeout=1.0)
    activation_release.set()
    assert handle_closed.wait(timeout=1.0)
    assert closed_handles == [8675309]
    termination_release.set()
    assert terminated.wait(timeout=1.0)


def test_cancellation_during_lease_wait_prevents_active_lease_publication(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    run_id = "cancel-lease-wait"
    _prepare(generator, run_id)

    @contextmanager
    def cancelled_lock(anchor, name: str, *, timeout: float = 5.0, operation_check=None):
        assert anchor.directory == generator.work_root
        assert name == generator._lease_lock_path.name
        assert timeout == 5.0
        assert operation_check is not None
        assert generator.cancel(run_id) is True
        operation_check()
        yield

    monkeypatch.setattr(local_generator, "_interprocess_lock", cancelled_lock)
    try:
        with pytest.raises(GenerationCancelledError, match="cancelled"):
            _generate(generator, checkpoint)
    finally:
        generator.finish_control(run_id)

    assert not generator._lease_path.exists()


def test_complete_publication_rechecks_cancellation_and_records_failed(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    run_id = "cancel-before-complete-write"
    _prepare(generator, run_id)
    original_write = local_generator._write_lease
    raced = False

    def racing_write(anchor, name: str, value, *, operation_check=None):
        nonlocal raced
        if value.get("status") == "COMPLETE" and not raced:
            raced = True
            assert generator.cancel(run_id) is True
        return original_write(anchor, name, value, operation_check=operation_check)

    monkeypatch.setattr(local_generator, "_write_lease", racing_write)
    try:
        with pytest.raises(GenerationCancelledError, match="cancelled"):
            _generate(generator, checkpoint)
    finally:
        generator.finish_control(run_id)

    lease = json.loads(generator._lease_path.read_text(encoding="utf-8"))
    assert raced is True
    assert lease["status"] == "FAILED"
    assert lease["retryable"] is True


def test_cancellation_after_complete_publication_is_checked_before_return(tmp_path: Path, monkeypatch) -> None:
    generator = _generator(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    run_id = "cancel-before-return"
    _prepare(generator, run_id)
    original_release = generator._release_lease
    raced = False

    def racing_release(lease_id: str, *, status: str, retryable: bool, **kwargs):
        nonlocal raced
        result = original_release(lease_id, status=status, retryable=retryable, **kwargs)
        if status == "COMPLETE" and result and not raced:
            raced = True
            assert generator.cancel(run_id) is True
        return result

    monkeypatch.setattr(generator, "_release_lease", racing_release)
    try:
        with pytest.raises(GenerationCancelledError, match="cancelled"):
            _generate(generator, checkpoint)
    finally:
        generator.finish_control(run_id)

    lease = json.loads(generator._lease_path.read_text(encoding="utf-8"))
    assert raced is True
    assert lease["status"] == "FAILED"
    assert lease["retryable"] is True

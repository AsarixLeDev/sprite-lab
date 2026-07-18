from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import spritelab.product_features.evaluation.local_generator as local_generator_module
from spritelab.product_features.evaluation.local_generator import (
    LocalCheckpointPlaygroundGenerator,
    LocalPlaygroundGenerationError,
)
from spritelab.utils.safe_fs import AnchoredDirectory


def _checkpoint(root: Path) -> Path:
    import torch

    path = root / "runs" / "train-smoke" / "checkpoint_last.pt"
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
    indexed = config.out_dir / "indexed_png"
    indexed.mkdir()
    rows = []
    prompt_rows = [json.loads(line) for line in config.prompts.read_text(encoding="utf-8").splitlines()]
    for index in range(config.max_samples):
        relative = f"indexed_png/sample_{index:04d}.png"
        Image.new("RGBA", (32, 32), (index, 20, 30, 255)).save(config.out_dir / relative)
        rows.append(
            {
                **prompt_rows[index],
                "sample_id": f"sample_{index:06d}",
                "seed": config.seed,
                "noise_seed": config.noise_seed + index,
                "model_type": "generator_challenger",
                "steps": config.steps,
                "cfg_scale": config.cfg_scale,
                "paths": {"indexed_png": relative},
            }
        )
    (config.out_dir / "generated_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"sample_count": len(rows)}


def _manifest_row(config, *, index: int, relative: str) -> dict[str, object]:
    prompt = json.loads(config.prompts.read_text(encoding="utf-8").splitlines()[index])
    return {
        **prompt,
        "sample_id": f"sample_{index:06d}",
        "seed": config.seed,
        "noise_seed": config.noise_seed + index,
        "model_type": "generator_challenger",
        "steps": config.steps,
        "cfg_scale": config.cfg_scale,
        "paths": {"indexed_png": relative},
    }


def _invoke(generator: LocalCheckpointPlaygroundGenerator, checkpoint: Path, **overrides):
    values = {
        "checkpoint": checkpoint,
        "prompt": "small shield",
        "seed": 1,
        "sampling_steps": 2,
        "guidance": 2.0,
        "image_count": 1,
        "weights": "live",
        "expected_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "expected_step": 7,
        "expected_variant": "live",
    }
    values.update(overrides)
    return generator.generate(**values)


def test_local_generator_is_passive_until_generate_and_returns_exact_pngs(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    assert generator.remote is False
    assert generator.billable is False
    assert not work_root.exists()

    checkpoint = _checkpoint(tmp_path)
    assets = generator.generate(
        checkpoint=checkpoint,
        prompt="small blue potion",
        seed=71,
        sampling_steps=2,
        guidance=2.5,
        image_count=2,
        weights="live",
        expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        expected_step=7,
        expected_variant="live",
    )

    assert len(assets) == 2
    assert all(asset.media_type == "image/png" for asset in assets)
    assert all(asset.content.startswith(b"\x89PNG\r\n\x1a\n") for asset in assets)
    invocations = [path for path in work_root.iterdir() if path.is_dir()]
    assert len(invocations) == 1
    prompt_rows = [
        json.loads(line) for line in (invocations[0] / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["prompt"] for row in prompt_rows] == ["small blue potion", "small blue potion"]
    assert {row["scope"] for row in prompt_rows} == {"EXPLORATORY"}


@pytest.mark.skipif(os.name != "nt" or sys.platform != "win32", reason="Windows mandatory labels are required.")
def test_windows_invocation_is_empty_and_untrusted_before_population(tmp_path: Path) -> None:
    import spritelab.utils.write_confinement as confinement_module

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=_sampler,
    )

    invocation = generator._new_invocation_directory()

    assert tuple(invocation.iterdir()) == ()
    assert confinement_module._windows_path_integrity_label(invocation) == (0, True)


@pytest.mark.skipif(os.name != "nt" or sys.platform != "win32", reason="Windows no-delete anchors are required.")
def test_windows_invocation_cannot_be_renamed_at_the_integrity_label_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=_sampler,
    )
    moved = tmp_path / "invocation-moved"
    outside = tmp_path / "outside-invocation-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original = local_generator_module.prepare_windows_untrusted_integrity_workspace
    rename_denied = False

    def race(invocation: Path):
        nonlocal rename_denied
        with pytest.raises(OSError):
            os.replace(invocation, moved)
        rename_denied = True
        return original(invocation)

    monkeypatch.setattr(local_generator_module, "prepare_windows_untrusted_integrity_workspace", race)

    with generator._new_anchored_invocation_directory() as (invocation, anchor):
        assert rename_denied is True
        assert anchor.names() == ()
        assert invocation.is_dir()
        assert not moved.exists()
        assert sentinel.read_bytes() == b"outside-byte-identical"


@pytest.mark.skipif(os.name != "nt" or sys.platform != "win32", reason="Windows exact-handle launch is required.")
def test_windows_contained_sampler_uses_safe_launcher_with_one_exact_worker_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.product_features.evaluation import local_generator as generator_module
    from spritelab.utils import runtime_closure

    project_root = tmp_path / "project"
    worker_path = project_root / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    worker_path.parent.mkdir(parents=True)
    worker_payload = b"# exact held worker fixture\n"
    worker_path.write_bytes(worker_payload)
    work_root = project_root / "runs" / "playground-sampler-work"
    calls: list[tuple[list[str], dict[str, Any]]] = []
    events: list[str] = []

    class CompletedProcess:
        pid = os.getpid()

        @staticmethod
        def poll() -> int:
            return 70

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            del timeout
            return 70

        @staticmethod
        def terminate() -> None:
            events.append("terminate")

        kill = terminate

    def process_factory(argv: list[str], **options: Any) -> CompletedProcess:
        calls.append((list(argv), dict(options)))
        return CompletedProcess()

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=project_root,
        work_root=work_root,
        windows_process_factory=process_factory,
    )
    worker_sha256 = sha256(worker_payload).hexdigest()
    monkeypatch.setattr(
        generator,
        "_code_inventory",
        lambda _operation_check: [
            {
                "path": "src/spritelab/product_features/evaluation/playground_worker.py",
                "sha256": worker_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        runtime_closure,
        "prepare_exact_python_runtime_closure",
        lambda _root, *, operation_check: {
            "runtime_closure_identity": "1" * 64,
            "operation_checked": operation_check is not None,
        },
    )
    monkeypatch.setattr(
        runtime_closure,
        "exact_python_runtime_environment_paths",
        lambda _root, *, operation_check: ((), ()),
    )

    def activate(process: Any, *, verifier: Any) -> int:
        events.append("activate")
        verifier(process)
        return 17

    monkeypatch.setattr(generator_module, "activate_windows_suspended_process", activate)
    monkeypatch.setattr(generator_module, "verify_process_image", lambda _process, _pin: events.append("verify"))
    monkeypatch.setattr(generator_module, "close_windows_handle", lambda handle: events.append(f"close:{handle}"))

    invocation = generator._new_invocation_directory()
    checkpoint = invocation / "checkpoint.snapshot.pt"
    prompts = invocation / "prompts.jsonl"
    output = invocation / "generated"
    checkpoint.write_bytes(b"checkpoint")
    prompts.write_bytes(b'{"prompt":"shield"}\n')
    output.mkdir()
    cancel_event = threading.Event()
    generator._active["run"] = {"cancel": cancel_event, "process": None}

    with AnchoredDirectory(invocation, invocation) as invocation_anchor:
        with pytest.raises(LocalPlaygroundGenerationError, match="failed safely"):
            generator._run_contained_sampler(
                invocation=invocation,
                checkpoint=checkpoint,
                prompts=prompts,
                prompts_sha256=sha256(prompts.read_bytes()).hexdigest(),
                output=output,
                seed=1,
                sampling_steps=2,
                guidance=2.0,
                image_count=1,
                expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                expected_step=7,
                expected_variant="live",
                run_id="run",
                deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
                cancel_event=cancel_event,
                operation_check=lambda: None,
                invocation_anchor=invocation_anchor,
            )

    assert len(calls) == 1
    argv, options = calls[0]
    worker_handle = int(argv[argv.index("--worker-handle") + 1])
    assert options == {
        "cwd": invocation,
        "env": generator_module._minimal_sampler_environment(
            invocation,
            project_root=project_root,
            import_paths=(),
            runtime_paths=(),
        ),
        "stdin_payload": b"",
        "writable_roots": (invocation,),
        "stdio_root": invocation / "tmp",
        "inherited_handles": (worker_handle,),
    }
    assert events == ["activate", "verify", "close:17"]


@pytest.mark.skipif(
    sys.platform.startswith("linux") or (sys.platform == "win32" and os.name == "nt"),
    reason="Linux and Windows have exact writable-root launchers.",
)
def test_production_local_generator_fails_closed_without_exact_platform_confinement(tmp_path: Path) -> None:
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs/playground-sampler-work",
    )
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(LocalPlaygroundGenerationError, match="exact writable-root launcher"):
        _invoke(generator, checkpoint)


def test_local_generator_rejects_checkpoint_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-checkpoint.pt"
    outside.write_bytes(b"outside")
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=_sampler,
    )
    with pytest.raises(ValueError, match="escapes"):
        generator.generate(
            checkpoint=outside,
            prompt="small shield",
            seed=1,
            sampling_steps=2,
            guidance=2.0,
            image_count=1,
            weights="ema",
            expected_sha256=sha256(outside.read_bytes()).hexdigest(),
            expected_step=7,
            expected_variant="ema",
        )
    assert outside.read_bytes() == b"outside"


def test_local_generator_rejects_manifest_traversal_without_reading_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.png"
    sentinel.write_bytes(b"outside-sentinel")

    def malicious_sampler(config):
        (config.out_dir / "generated_manifest.jsonl").write_text(
            json.dumps(_manifest_row(config, index=0, relative="../../sentinel.png")) + "\n",
            encoding="utf-8",
        )
        return {"sample_count": 1}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=malicious_sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="escapes"):
        generator.generate(
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
    assert sentinel.read_bytes() == b"outside-sentinel"


def test_local_generator_rejects_count_and_image_contract_mismatches(tmp_path: Path) -> None:
    def wrong_image_sampler(config):
        indexed = config.out_dir / "indexed_png"
        indexed.mkdir()
        Image.new("RGBA", (16, 16), (0, 0, 0, 255)).save(indexed / "wrong.png")
        (config.out_dir / "generated_manifest.jsonl").write_text(
            json.dumps(_manifest_row(config, index=0, relative="indexed_png/wrong.png")) + "\n",
            encoding="utf-8",
        )
        return {"sample_count": 1}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=wrong_image_sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="32x32"):
        generator.generate(
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


def test_local_generator_rejects_catalog_hash_mismatch_before_sampler(tmp_path: Path) -> None:
    called = False

    def sampler(config):
        nonlocal called
        called = True
        return _sampler(config)

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="durable catalog hash"):
        _invoke(generator, checkpoint, expected_sha256="0" * 64)
    assert called is False


def test_checkpoint_hash_mismatch_persists_no_snapshot_or_outside_secret(tmp_path: Path) -> None:
    outside_secret = tmp_path / "outside-secret.bin"
    secret_bytes = b"OUTSIDE-CHECKPOINT-SECRET-MUST-NOT-PERSIST"
    outside_secret.write_bytes(secret_bytes)
    work_root = tmp_path / "runs" / "playground-sampler-work"
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(LocalPlaygroundGenerationError, match="durable catalog hash"):
        _invoke(generator, checkpoint, expected_sha256=sha256(secret_bytes).hexdigest())

    assert outside_secret.read_bytes() == secret_bytes
    assert not any(path.name == "checkpoint.snapshot.pt" for path in work_root.rglob("*"))
    for path in work_root.rglob("*"):
        if path.is_file():
            assert secret_bytes not in path.read_bytes()


def test_post_preflight_checkpoint_mutation_publishes_no_snapshot_or_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bytes = b"A" * (2 * 1024 * 1024 + 17)
    secret_bytes = b"POST-PREFLIGHT-SECRET-" + b"S" * (len(original_bytes) - 22)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(original_bytes)
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel_bytes = b"OUTSIDE-SENTINEL-BYTE-IDENTICAL"
    sentinel.write_bytes(sentinel_bytes)
    source_descriptor = os.open(
        checkpoint,
        os.O_RDWR | int(getattr(os, "O_BINARY", 0)),
    )
    real_lseek = local_generator_module.os.lseek
    source_rewinds = 0

    def mutate_after_preflight(descriptor: int, offset: int, whence: int) -> int:
        nonlocal source_rewinds
        result = real_lseek(descriptor, offset, whence)
        if descriptor == source_descriptor and offset == 0 and whence == os.SEEK_SET:
            source_rewinds += 1
            if source_rewinds == 2:
                os.ftruncate(descriptor, 0)
                os.write(descriptor, secret_bytes)
                real_lseek(descriptor, 0, os.SEEK_SET)
        return result

    monkeypatch.setattr(local_generator_module.os, "lseek", mutate_after_preflight)
    try:
        with AnchoredDirectory(invocation, invocation) as anchor:
            with pytest.raises(LocalPlaygroundGenerationError, match=r"Checkpoint (?:snapshot|changed)"):
                LocalCheckpointPlaygroundGenerator._snapshot_checkpoint(
                    checkpoint,
                    invocation / "checkpoint.snapshot.pt",
                    expected_sha256=sha256(original_bytes).hexdigest(),
                    source_descriptor=source_descriptor,
                    destination_anchor=anchor,
                )
    finally:
        os.close(source_descriptor)

    assert source_rewinds >= 2
    assert sentinel.read_bytes() == sentinel_bytes
    assert not (invocation / "checkpoint.snapshot.pt").exists()
    for path in invocation.rglob("*"):
        if path.is_file():
            assert secret_bytes not in path.read_bytes()


def test_checkpoint_snapshot_publishes_held_inode_and_never_retires_foreign_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_bytes = b"held-checkpoint-bytes"
    checkpoint.write_bytes(checkpoint_bytes)
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    destination = invocation / "checkpoint.snapshot.pt"
    outside_sentinel = tmp_path / "outside-sentinel.bin"
    outside_sentinel.write_bytes(b"outside-byte-identical")
    original_publish = AnchoredDirectory.publish_held_file_no_replace
    observed: dict[str, Path | None] = {"owned": None, "foreign": None}

    def substitute_before_publish(
        anchor: AnchoredDirectory,
        source_descriptor: int,
        source_name: str | None,
        destination_name: str,
        *,
        identity,
    ) -> None:
        if anchor.directory == invocation and destination_name == destination.name:
            if source_name is None:
                foreign_fd = anchor.open_file(
                    destination_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                )
                try:
                    os.write(foreign_fd, b"foreign-final")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
                observed["foreign"] = destination
            else:
                owned = invocation / "attacker-displaced-owned.bin"
                anchor.rename(source_name, owned.name, replace=False)
                foreign_fd = anchor.open_file(
                    source_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                )
                try:
                    os.write(foreign_fd, b"foreign-staging")
                    os.fsync(foreign_fd)
                finally:
                    os.close(foreign_fd)
                observed["owned"] = owned
                observed["foreign"] = invocation / source_name
        return original_publish(
            anchor,
            source_descriptor,
            source_name,
            destination_name,
            identity=identity,
        )

    monkeypatch.setattr(AnchoredDirectory, "publish_held_file_no_replace", substitute_before_publish)
    with AnchoredDirectory(invocation, invocation) as anchor:
        with pytest.raises(
            (FileExistsError, LocalPlaygroundGenerationError, local_generator_module.UnsafeFilesystemOperation)
        ):
            LocalCheckpointPlaygroundGenerator._snapshot_checkpoint(
                checkpoint,
                destination,
                expected_sha256=sha256(checkpoint_bytes).hexdigest(),
                destination_anchor=anchor,
            )

    foreign = observed["foreign"]
    assert foreign is not None and foreign.is_file()
    assert foreign.read_bytes() in {b"foreign-final", b"foreign-staging"}
    owned = observed["owned"]
    if owned is not None:
        assert owned.read_bytes() == checkpoint_bytes
        assert not destination.exists()
    else:
        assert destination.read_bytes() == b"foreign-final"
    assert outside_sentinel.read_bytes() == b"outside-byte-identical"


@pytest.mark.skipif(os.name == "nt", reason="POSIX anonymous-file fallback is platform-specific.")
def test_checkpoint_snapshot_no_anonymous_file_falls_back_to_single_link_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    payload = b"portable-direct-final"
    checkpoint.write_bytes(payload)
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    destination = invocation / "checkpoint.snapshot.pt"

    def unsupported(_anchor: AnchoredDirectory, _mode: int = 0o600) -> int:
        raise local_generator_module.UnsafeFilesystemOperation("anonymous files unavailable")

    monkeypatch.setattr(AnchoredDirectory, "open_anonymous_file", unsupported)
    with AnchoredDirectory(invocation, invocation) as anchor:
        identity = LocalCheckpointPlaygroundGenerator._snapshot_checkpoint(
            checkpoint,
            destination,
            expected_sha256=sha256(payload).hexdigest(),
            destination_anchor=anchor,
        )
        metadata = anchor.lstat(destination.name)

    assert identity.matches(metadata)
    assert metadata.st_nlink == 1
    assert destination.read_bytes() == payload


def test_evaluation_template_matches_playground_numeric_contract() -> None:
    template = Path(local_generator_module.__file__).with_name("templates").joinpath("evaluation.html")
    content = template.read_text(encoding="utf-8")

    assert 'id="gallery-seed" type="number" min="0" max="9223372036854775807"' in content
    assert 'id="play-seed" type="number" min="0" max="9223372036854775807"' in content
    assert 'id="play-guidance" type="number" min="0.1" max="50" step="0.1"' in content


def test_local_generator_rejects_manifest_prompt_semantics(tmp_path: Path) -> None:
    def sampler(config):
        report = _sampler(config)
        manifest = config.out_dir / "generated_manifest.jsonl"
        row = json.loads(manifest.read_text(encoding="utf-8"))
        row["prompt"] = "different prompt"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return report

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="semantics"):
        _invoke(generator, _checkpoint(tmp_path))


def test_local_generator_serializes_active_process_lease(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    work_root.mkdir(parents=True)
    (work_root / "sampler-lease.json").write_text(
        json.dumps(
            {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": "active-owner",
                "status": "ACTIVE",
                "owner_pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="already active"):
        _invoke(generator, _checkpoint(tmp_path))


def test_local_generator_recovers_dead_owner_as_retryable_orphan(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    work_root.mkdir(parents=True)
    (work_root / "sampler-lease.json").write_text(
        json.dumps(
            {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": "dead-owner",
                "status": "ACTIVE",
                "owner_pid": 2_147_483_647,
            }
        ),
        encoding="utf-8",
    )
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    assert len(_invoke(generator, _checkpoint(tmp_path))) == 1
    lease = json.loads((work_root / "sampler-lease.json").read_text(encoding="utf-8"))
    assert lease["status"] == "COMPLETE"
    assert lease["recovered_orphan"] == {
        "lease_id": "dead-owner",
        "retryable": True,
        "status": "ORPHANED",
    }


def test_local_generator_rejects_unicode_normalized_output_collision(tmp_path: Path) -> None:
    def sampler(config):
        indexed = config.out_dir / "indexed_png"
        indexed.mkdir()
        relatives = ("indexed_png/A.png", "indexed_png/\uff21.png")
        rows = []
        for index, relative in enumerate(relatives):
            Image.new("RGBA", (32, 32), (index, 2, 3, 255)).save(config.out_dir / relative)
            rows.append(_manifest_row(config, index=index, relative=relative))
        (config.out_dir / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {"sample_count": 2}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="colliding"):
        _invoke(generator, _checkpoint(tmp_path), image_count=2)


def test_local_generator_binds_actual_sampler_load_to_original_snapshot_hash(tmp_path: Path) -> None:
    def swapping_sampler(config):
        import torch

        from spritelab.training.generator_challenger import run_sample_generator_challenger

        replacement = config.checkpoint.with_name("replacement-step-999.pt")
        torch.save(
            {
                "model_type": "generator_challenger",
                "ema_weights": False,
                "step": 999,
                "global_step": 999,
            },
            replacement,
        )
        os.replace(replacement, config.checkpoint)
        return run_sample_generator_challenger(config)

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=swapping_sampler,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        _invoke(generator, _checkpoint(tmp_path))

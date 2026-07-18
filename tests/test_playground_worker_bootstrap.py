from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_features.evaluation import local_generator
from spritelab.product_features.evaluation import playground_worker as worker


def _marker_source(marker: Path, value: bytes) -> bytes:
    return f"from pathlib import Path\nPath({str(marker)!r}).write_bytes({value!r})\n".encode()


def _run_bootstrap(
    root: Path,
    held_payload: bytes,
    *,
    expected_sha256: str | None = None,
    declared_size: int | None = None,
    bootstrap_sha256: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    source = root / "held-worker.py"
    source.write_bytes(held_payload)
    descriptor = os.open(source, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    environment = dict(os.environ)
    environment.update(
        {
            "SPRITELAB_BOUND_BOOTSTRAP_SHA256": local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256,
            "SPRITELAB_ISOLATED_PATHS": "",
            "SPRITELAB_RUNTIME_ROOTS": "",
        }
    )
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP,
        "--bootstrap-sha256",
        bootstrap_sha256 or local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256,
        "--worker-sha256",
        expected_sha256 or hashlib.sha256(held_payload).hexdigest(),
        "--worker-size",
        str(len(held_payload) if declared_size is None else declared_size),
    ]
    try:
        with local_generator._inherited_worker_transport(descriptor) as (arguments, options):
            run_options = dict(options)
            inherited_states: dict[int, bool] = {}
            inherited_handles = tuple(run_options.pop("inherited_handles", ()))
            if inherited_handles:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.lpAttributeList = {"handle_list": list(inherited_handles)}
                run_options.update({"close_fds": True, "startupinfo": startupinfo})
                inherited_states = {handle: os.get_handle_inheritable(handle) for handle in inherited_handles}
                for handle in inherited_handles:
                    os.set_handle_inheritable(handle, True)
            try:
                return subprocess.run(
                    [*argv, *arguments],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    shell=False,
                    timeout=15,
                    check=False,
                    **run_options,
                )
            finally:
                for handle, was_inheritable in inherited_states.items():
                    os.set_handle_inheritable(handle, was_inheritable)
    finally:
        os.close(descriptor)


def test_real_bound_bootstrap_executes_exact_held_bytes(tmp_path: Path) -> None:
    marker = tmp_path / "executed.marker"
    result = _run_bootstrap(tmp_path, _marker_source(marker, b"exact"))

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == b""
    assert result.stderr == b""
    assert marker.read_bytes() == b"exact"


@pytest.mark.parametrize("failure", ("worker_hash", "bootstrap_hash", "short_size"))
def test_bootstrap_mismatch_executes_no_marker(tmp_path: Path, failure: str) -> None:
    marker = tmp_path / f"{failure}.marker"
    payload = _marker_source(marker, b"must-not-run")
    arguments: dict[str, object] = {}
    if failure == "worker_hash":
        arguments["expected_sha256"] = "0" * 64
    elif failure == "bootstrap_hash":
        arguments["bootstrap_sha256"] = "0" * 64
    else:
        arguments["declared_size"] = len(payload) - 1

    result = _run_bootstrap(tmp_path, payload, **arguments)

    assert result.returncode == 70
    assert result.stdout == b""
    assert result.stderr == b""
    assert not marker.exists()


def test_bootstrap_foreign_descriptor_substitution_executes_no_marker(tmp_path: Path) -> None:
    trusted_marker = tmp_path / "trusted.marker"
    hostile_marker = tmp_path / "hostile.marker"
    trusted = _marker_source(trusted_marker, b"trusted")
    substituted = _marker_source(hostile_marker, b"substituted")

    result = _run_bootstrap(
        tmp_path,
        substituted,
        expected_sha256=hashlib.sha256(trusted).hexdigest(),
    )

    assert result.returncode == 70
    assert not trusted_marker.exists()
    assert not hostile_marker.exists()


def test_bootstrap_oversize_input_executes_no_marker(tmp_path: Path) -> None:
    marker = tmp_path / "oversize.marker"
    prefix = _marker_source(marker, b"must-not-run")
    payload = prefix + b"#" * (local_generator._MAX_BOUND_PLAYGROUND_WORKER_BYTES + 1 - len(prefix))

    result = _run_bootstrap(tmp_path, payload)

    assert len(payload) == local_generator._MAX_BOUND_PLAYGROUND_WORKER_BYTES + 1
    assert result.returncode == 70
    assert not marker.exists()


def test_worker_transport_inherits_only_the_intended_handle_and_restores_it(tmp_path: Path) -> None:
    source = tmp_path / "worker.py"
    source.write_bytes(b"pass\n")
    descriptor = os.open(source, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        if os.name == "nt":
            import msvcrt

            handle = int(msvcrt.get_osfhandle(descriptor))
            inherited_before = os.get_handle_inheritable(handle)
            with local_generator._inherited_worker_transport(descriptor) as (arguments, options):
                assert arguments == ("--worker-handle", str(handle))
                assert options == {"inherited_handles": (handle,)}
                assert os.get_handle_inheritable(handle) is inherited_before
            assert os.get_handle_inheritable(handle) is inherited_before
        else:
            with local_generator._inherited_worker_transport(descriptor) as (arguments, options):
                assert arguments == ("--worker-fd", str(descriptor))
                assert options == {"pass_fds": (descriptor,)}
    finally:
        os.close(descriptor)


def _control(root: Path) -> tuple[dict[str, object], SimpleNamespace]:
    worker_sha256 = "1" * 64
    inventory = [
        {
            "path": "src/spritelab/product_features/evaluation/playground_worker.py",
            "sha256": worker_sha256,
        }
    ]
    runtime_identity = "2" * 64
    bootstrap_identity = "3" * 64
    value: dict[str, object] = {
        "schema_version": worker._CONTROL_SCHEMA,
        "control_identity": "",
        "bootstrap_identity": bootstrap_identity,
        "worker_sha256": worker_sha256,
        "worker_size": 100,
        "checkpoint": "checkpoint.snapshot.pt",
        "checkpoint_sha256": "4" * 64,
        "prompts": "prompts.jsonl",
        "prompts_sha256": "5" * 64,
        "output": "generated",
        "report": "sampler-result.json",
        "seed": 7,
        "sampling_steps": 20,
        "guidance": 4.5,
        "image_count": 2,
        "expected_step": 11,
        "expected_variant": "ema",
        "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        "code_inventory": inventory,
        "code_inventory_identity": worker._canonical_sha256(inventory),
        "runtime_closure": {"runtime_closure_identity": runtime_identity},
        "runtime_closure_identity": runtime_identity,
        "workspace_identity": {"device": 1, "inode": 2},
    }
    value["control_identity"] = worker._record_identity(value, "control_identity")
    parsed = SimpleNamespace(
        bootstrap_sha256=bootstrap_identity,
        worker_sha256=worker_sha256,
        worker_size=100,
    )
    return value, parsed


def test_sampler_control_schema_is_exact_and_identity_bound(tmp_path: Path) -> None:
    value, parsed = _control(tmp_path)
    validated, deadline = worker._validate_control(value, workspace=tmp_path, parsed=parsed)
    assert validated == value
    assert deadline > datetime.now(timezone.utc)

    for mutation in ("extra", "bool_seed", "runtime_substitution", "worker_substitution"):
        hostile = dict(value)
        if mutation == "extra":
            hostile["extra"] = None
        elif mutation == "bool_seed":
            hostile["seed"] = True
        elif mutation == "runtime_substitution":
            hostile["runtime_closure_identity"] = "9" * 64
        else:
            hostile["worker_sha256"] = "9" * 64
        hostile["control_identity"] = ""
        hostile["control_identity"] = worker._record_identity(hostile, "control_identity")
        with pytest.raises(RuntimeError, match=r"malformed|bound"):
            worker._validate_control(hostile, workspace=tmp_path, parsed=parsed)


@pytest.mark.parametrize(
    ("key", "replacement"),
    (
        ("seed", 2**63),
        ("guidance", 0.0),
        ("guidance", 50.000_001),
        ("sampling_steps", 501),
        ("image_count", 17),
    ),
)
def test_sampler_control_bounds_match_the_public_request_contract(
    tmp_path: Path,
    key: str,
    replacement: object,
) -> None:
    value, parsed = _control(tmp_path)
    value[key] = replacement
    value["control_identity"] = ""
    value["control_identity"] = worker._record_identity(value, "control_identity")

    with pytest.raises(RuntimeError, match="parameters are malformed"):
        worker._validate_control(value, workspace=tmp_path, parsed=parsed)


def _workspace_digest(identity: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _confinement_evidence(*, workspace_identity: dict[str, int] | None = None) -> dict[str, object]:
    identity = workspace_identity or {"device": 1, "inode": 2}
    return {
        "schema_version": "spritelab.write-confinement-evidence.v3",
        "strategy": "linux-landlock-v1",
        "platform": "linux",
        "kernel_abi": 6,
        "root_identity_sha256": _workspace_digest(identity),
        "handled_access_fs": 65_522,
        "allowed_access_fs": 25_010,
        "no_new_privileges": True,
        "restricted_token": False,
        "integrity_level_rid": 0,
        "mandatory_no_write_up": False,
        "workspace_integrity_level_rid": 0,
        "startup_integrity_level_rid": 0,
        "bootstrap_lowered_before_worker_import": False,
        "new_thread_integrity_level_rid": 0,
        "raise_to_low_denied": False,
        "medium_probe_write_denied": False,
        "low_world_probe_write_denied": False,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": False,
        "job_active_process_limit": 0,
        "paths_exposed": False,
    }


def _windows_confinement_evidence(
    *,
    restricted_token: bool,
    workspace_identity: dict[str, int] | None = None,
) -> dict[str, object]:
    identity = workspace_identity or {"device": 1, "inode": 2}
    return {
        "schema_version": "spritelab.write-confinement-evidence.v3",
        "strategy": "windows-bootstrap-to-untrusted-integrity-v1",
        "platform": "windows",
        "kernel_abi": 0,
        "root_identity_sha256": _workspace_digest(identity),
        "handled_access_fs": 0,
        "allowed_access_fs": 0,
        "no_new_privileges": False,
        "restricted_token": restricted_token,
        "integrity_level_rid": 0,
        "mandatory_no_write_up": True,
        "workspace_integrity_level_rid": 0,
        "startup_integrity_level_rid": 4096,
        "bootstrap_lowered_before_worker_import": True,
        "new_thread_integrity_level_rid": 0,
        "raise_to_low_denied": True,
        "medium_probe_write_denied": True,
        "low_world_probe_write_denied": True,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": True,
        "job_active_process_limit": 1,
        "paths_exposed": False,
    }


@pytest.mark.parametrize("restricted_token", (False, True))
def test_sampler_result_accepts_only_exact_windows_confinement_evidence(restricted_token: bool) -> None:
    workspace_identity = {"device": 1, "inode": 2}
    evidence = _windows_confinement_evidence(restricted_token=restricted_token)
    assert (
        local_generator._valid_pathless_confinement_evidence(
            evidence,
            workspace_identity=workspace_identity,
        )
        is True
    )

    for key, replacement in (
        ("strategy", "windows-low-integrity-v1"),
        ("platform", "win32"),
        ("integrity_level_rid", 4096),
        ("mandatory_no_write_up", False),
        ("workspace_integrity_level_rid", 4096),
        ("startup_integrity_level_rid", 0),
        ("bootstrap_lowered_before_worker_import", False),
        ("raise_to_low_denied", False),
        ("medium_probe_write_denied", False),
        ("low_world_probe_write_denied", False),
        ("job_active_process_limit", 2),
        ("paths_exposed", True),
    ):
        hostile = dict(evidence)
        hostile[key] = replacement
        assert (
            local_generator._valid_pathless_confinement_evidence(
                hostile,
                workspace_identity=workspace_identity,
            )
            is False
        )


def test_windows_worker_proves_confinement_before_project_runtime_and_reads_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.utils import runtime_closure as _runtime_closure  # noqa: F401
    from spritelab.utils import write_confinement

    checkpoint_payload = b"checkpoint"
    prompts_payload = b'{"prompt":"sprite"}\n'
    checkpoint = tmp_path / "checkpoint.snapshot.pt"
    prompts = tmp_path / "prompts.jsonl"
    checkpoint.write_bytes(checkpoint_payload)
    prompts.write_bytes(prompts_payload)
    value = {
        "workspace_identity": {"device": 12, "inode": 34},
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "prompts": prompts.name,
        "prompts_sha256": hashlib.sha256(prompts_payload).hexdigest(),
        "output": "generated",
        "runtime_closure": {},
    }
    parsed = SimpleNamespace(workspace_fd=None, checkpoint_fd=None, prompts_fd=None)
    events: list[object] = []

    class _Evidence:
        def to_dict(self) -> dict[str, object]:
            return _windows_confinement_evidence(restricted_token=True)

    def prove_confinement(
        workspace: Path,
        *,
        expected_device: int,
        expected_inode: int,
    ) -> _Evidence:
        events.append(("confinement", workspace, expected_device, expected_inode))
        return _Evidence()

    real_import = builtins.__import__

    def tracked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "spritelab" or name.startswith("spritelab."):
            events.append(("project-import", name))
        return real_import(name, *args, **kwargs)

    def stable_bytes(
        path: Path,
        _maximum: int,
        operation_check: object = None,
    ) -> bytes:
        events.append(("path-read", path))
        if path == checkpoint:
            return checkpoint_payload
        if path == prompts:
            return prompts_payload
        raise AssertionError(f"unexpected path read: {path}")

    class _RuntimeBoundaryReached(RuntimeError):
        pass

    @contextmanager
    def stop_before_generator(*_args: object, **_kwargs: object):
        raise _RuntimeBoundaryReached
        yield  # pragma: no cover

    monkeypatch.setattr(worker.sys, "platform", "win32")
    monkeypatch.setattr(
        write_confinement,
        "windows_current_process_confinement_evidence",
        prove_confinement,
    )
    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.setattr(worker, "_stable_bytes", stable_bytes)
    monkeypatch.setattr(
        worker,
        "_stable_descriptor_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("descriptor path used on Windows")),
    )
    monkeypatch.setattr(worker, "_bound_exact_runtime_imports", stop_before_generator)

    with pytest.raises(_RuntimeBoundaryReached):
        worker._run_with_bound_project_sources(
            parsed,
            tmp_path,
            tmp_path,
            value,
            object(),  # type: ignore[arg-type]
            lambda: None,
        )

    project_imports = [event for event in events if event[0] == "project-import"]
    assert project_imports[:2] == [
        ("project-import", "spritelab.utils.write_confinement"),
        ("project-import", "spritelab.utils.runtime_closure"),
    ]
    assert events.index(("confinement", tmp_path, 12, 34)) < events.index(project_imports[1])
    assert ("path-read", checkpoint) in events
    assert ("path-read", prompts) in events


def test_sampler_result_schema_rejects_rehashed_binding_substitution(tmp_path: Path) -> None:
    control, _parsed = _control(tmp_path)
    result: dict[str, object] = {
        "schema_version": local_generator._PLAYGROUND_RESULT_SCHEMA,
        "result_identity": "",
        "control_identity": control["control_identity"],
        "bootstrap_identity": control["bootstrap_identity"],
        "worker_sha256": control["worker_sha256"],
        "checkpoint_sha256": control["checkpoint_sha256"],
        "prompts_sha256": control["prompts_sha256"],
        "code_inventory_identity": control["code_inventory_identity"],
        "runtime_closure_identity": control["runtime_closure_identity"],
        "workspace_identity": dict(control["workspace_identity"]),
        "deadline_at": control["deadline_at"],
        "report": {"sample_count": 2},
        "runtime_identity": {"runtime_closure_identity": control["runtime_closure_identity"]},
        "write_confinement": _confinement_evidence(),
    }
    result["result_identity"] = local_generator._record_identity(result, "result_identity")
    assert local_generator._valid_sampler_result(result, control) is True

    substituted = dict(result)
    substituted["checkpoint_sha256"] = "9" * 64
    substituted["result_identity"] = ""
    substituted["result_identity"] = local_generator._record_identity(substituted, "result_identity")
    assert local_generator._valid_sampler_result(substituted, control) is False

    extra = dict(result)
    extra["extra"] = None
    extra["result_identity"] = ""
    extra["result_identity"] = local_generator._record_identity(extra, "result_identity")
    assert local_generator._valid_sampler_result(extra, control) is False

    private_report = dict(result)
    private_report["report"] = {
        "sample_count": 2,
        "config": {"checkpoint": r"C:\\private\\checkpoint.pt"},
    }
    private_report["result_identity"] = ""
    private_report["result_identity"] = local_generator._record_identity(private_report, "result_identity")
    assert local_generator._valid_sampler_result(private_report, control) is False

    rebound_root = dict(result)
    evidence = dict(_confinement_evidence())
    evidence["root_identity_sha256"] = _workspace_digest({"device": 9, "inode": 9})
    rebound_root["write_confinement"] = evidence
    rebound_root["result_identity"] = ""
    rebound_root["result_identity"] = local_generator._record_identity(rebound_root, "result_identity")
    assert local_generator._valid_sampler_result(rebound_root, control) is False


@pytest.mark.parametrize(
    ("key", "replacement"),
    (
        ("kernel_abi", 2),
        ("kernel_abi", 11),
        ("handled_access_fs", 32_754),
        ("allowed_access_fs", 25_011),
        ("no_new_privileges", False),
        ("root_identity_sha256", "9" * 64),
    ),
)
def test_linux_confinement_evidence_rejects_every_weakened_or_rebound_field(
    key: str,
    replacement: object,
) -> None:
    identity = {"device": 1, "inode": 2}
    evidence = _confinement_evidence(workspace_identity=identity)
    evidence[key] = replacement

    assert (
        local_generator._valid_pathless_confinement_evidence(
            evidence,
            workspace_identity=identity,
        )
        is False
    )


def test_worker_projects_diagnostics_to_a_pathless_exact_allowlist(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    (output / "indexed_png").mkdir(parents=True)
    (output / "indexed_png" / "sample_0000.png").write_bytes(b"png")
    private_path = r"C:\\private\\checkpoint-secret.pt"
    row = {
        "cfg_scale": 4.5,
        "model_type": "generator_challenger",
        "noise_seed": 7,
        "paths": {
            "indexed_png": "indexed_png/sample_0000.png",
            "checkpoint": private_path,
        },
        "prompt": "tiny shield",
        "prompt_id": "playground_0000",
        "sample_id": "sample_000000",
        "scope": "EXPLORATORY",
        "seed": 7,
        "steps": 20,
        "requested_checkpoint": private_path,
        "config": {"out_dir": private_path, "nested": {"secret": private_path}},
    }
    (output / "generated_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (output / "generation_report.json").write_text(
        json.dumps({"sample_count": 1, "checkpoint": private_path, "config": row["config"]}),
        encoding="utf-8",
    )
    (output / "generation_report.md").write_text(f"checkpoint: {private_path}\n", encoding="utf-8")

    report = worker._project_playground_diagnostics(
        tmp_path,
        Path("generated"),
        image_count=1,
        operation_check=lambda: None,
    )

    assert report == {"sample_count": 1}
    projected = json.loads((output / "generated_manifest.jsonl").read_text(encoding="utf-8"))
    assert set(projected) == {
        "cfg_scale",
        "model_type",
        "noise_seed",
        "paths",
        "prompt",
        "prompt_id",
        "sample_id",
        "scope",
        "seed",
        "steps",
    }
    assert projected["paths"] == {"indexed_png": "indexed_png/sample_0000.png"}
    retained = b"".join(
        (output / name).read_bytes()
        for name in ("generated_manifest.jsonl", "generation_report.json", "generation_report.md")
    )
    assert private_path.encode() not in retained
    assert json.loads((output / "generation_report.json").read_text(encoding="utf-8")) == {
        "manifest": "generated_manifest.jsonl",
        "sample_count": 1,
    }


def test_worker_rejects_absolute_manifest_paths_before_projection(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    output.mkdir()
    sentinel = tmp_path / "outside.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    row = {
        "cfg_scale": 4.5,
        "model_type": "generator_challenger",
        "noise_seed": 7,
        "paths": {"indexed_png": os.fspath(sentinel)},
        "prompt": "tiny shield",
        "prompt_id": "playground_0000",
        "sample_id": "sample_000000",
        "scope": "EXPLORATORY",
        "seed": 7,
        "steps": 20,
    }
    manifest = output / "generated_manifest.jsonl"
    original = (json.dumps(row) + "\n").encode()
    manifest.write_bytes(original)

    with pytest.raises(RuntimeError, match="path is malformed"):
        worker._project_playground_diagnostics(
            tmp_path,
            Path("generated"),
            image_count=1,
            operation_check=lambda: None,
        )

    assert manifest.read_bytes() == original
    assert sentinel.read_bytes() == b"outside-byte-identical"


def test_bound_bootstrap_stays_well_below_windows_command_limit() -> None:
    assert len(local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP.encode("utf-8")) < 8_192
    assert (
        local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256
        == hashlib.sha256(local_generator._BOUND_PLAYGROUND_WORKER_BOOTSTRAP.encode("utf-8")).hexdigest()
    )

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from spritelab.utils.write_confinement import (
    LINUX_LANDLOCK_STRATEGY,
    WINDOWS_PARENT_ANCHORS_STRATEGY,
    WindowsLowIntegrityProcess,
    WriteConfinementError,
    WriteConfinementUnavailable,
    create_windows_bootstrap_untrusted_process,
    directory_identity,
    enforce_linux_landlock_write_confinement,
    enforce_linux_landlock_write_confinement_fds,
    prepare_windows_low_integrity_workspace,
    prepare_windows_untrusted_integrity_workspace,
    write_confinement_strategy,
)


class _FakeWindowsProcessKernel:
    def __init__(self, wait_results: list[int]) -> None:
        self.wait_results = list(wait_results)
        self.terminate_calls: list[tuple[int, int]] = []
        self.closed_handles: list[int] = []

    def WaitForSingleObject(self, handle: int, _timeout_ms: int) -> int:
        assert handle == 0x1234
        assert self.wait_results
        return self.wait_results.pop(0)

    def TerminateProcess(self, handle: int, exit_code: int) -> int:
        self.terminate_calls.append((handle, exit_code))
        return 1

    def CloseHandle(self, handle: int) -> int:
        self.closed_handles.append(handle)
        return 1


def _stdio_test_process(
    tmp_path: Path,
    *,
    stdout: bytes,
    stderr: bytes,
    maximum: int,
) -> WindowsLowIntegrityProcess:
    descriptors: list[int] = []
    for name, payload in (("stdin", b""), ("stdout", stdout), ("stderr", stderr)):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(payload)
        descriptors.append(os.open(path, os.O_RDWR | int(getattr(os, "O_BINARY", 0))))
    return WindowsLowIntegrityProcess(
        args=("python", "-I", "-B", "-S", "-c", "pass"),
        process_handle=0x1234,
        pid=4321,
        stdin_descriptor=descriptors[0],
        stdout_descriptor=descriptors[1],
        stderr_descriptor=descriptors[2],
        expected_input=b"",
        max_stdout_bytes=maximum,
        bootstrap_identity_sha256="a" * 64,
        desktop_handle=0,
        desktop_identity_sha256="b" * 64,
        confinement_probes={},
        confinement_anchors=ExitStack(),
        restricted_token=True,
        restricted_sid_hashes=(),
    )


def test_windows_process_poll_terminates_at_private_stdio_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    kernel = _FakeWindowsProcessKernel([0x102])
    monkeypatch.setattr(module, "_windows_kernel32", lambda: kernel)
    process = _stdio_test_process(tmp_path, stdout=b"12345", stderr=b"", maximum=4)
    try:
        with pytest.raises(OSError, match="stdout exceeded"):
            process.poll()
        assert kernel.terminate_calls == [(0x1234, 1)]
        assert process.returncode is None
    finally:
        process.close()


def test_windows_process_poll_then_wait_rechecks_private_stdio_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    kernel = _FakeWindowsProcessKernel([0])
    monkeypatch.setattr(module, "_windows_kernel32", lambda: kernel)
    monkeypatch.setattr(module, "_windows_process_exit_code", lambda _kernel, _handle: 0)
    process = _stdio_test_process(tmp_path, stdout=b"ok", stderr=b"", maximum=4)
    try:
        assert process.poll() == 0
        os.lseek(process._stdout_descriptor, 0, os.SEEK_END)
        assert os.write(process._stdout_descriptor, b"xxx") == 3
        with pytest.raises(OSError, match="stdout exceeded"):
            process.wait(timeout=1)
        assert kernel.terminate_calls == []
    finally:
        process.close()


def test_windows_process_communicate_rejects_oversized_private_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    kernel = _FakeWindowsProcessKernel([0])
    monkeypatch.setattr(module, "_windows_kernel32", lambda: kernel)
    monkeypatch.setattr(module, "_windows_process_exit_code", lambda _kernel, _handle: 0)
    process = _stdio_test_process(tmp_path, stdout=b"ok", stderr=b"12345", maximum=4)
    try:
        with pytest.raises(OSError, match="stdout exceeded"):
            process.communicate(timeout=1)
        assert process.returncode == 0
        assert kernel.terminate_calls == []
    finally:
        process.close()


def test_platform_strategy_has_no_weak_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform.startswith("linux"):
        assert write_confinement_strategy() == LINUX_LANDLOCK_STRATEGY
    elif sys.platform == "win32":
        assert write_confinement_strategy() == WINDOWS_PARENT_ANCHORS_STRATEGY
    else:
        with pytest.raises(WriteConfinementUnavailable):
            write_confinement_strategy()

    import spritelab.utils.write_confinement as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    with pytest.raises(WriteConfinementUnavailable):
        module.write_confinement_strategy()


def test_linux_uapi_structure_sizes_are_stable() -> None:
    import spritelab.utils.write_confinement as module

    assert ctypes.sizeof(module._RulesetAttr) == 8
    assert ctypes.sizeof(module._PathBeneathAttr) == 12


def test_directory_identity_binds_exact_unlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    metadata = root.stat()

    identity = directory_identity(root)

    assert identity.device == metadata.st_dev
    assert identity.inode == metadata.st_ino
    assert len(identity.identity_sha256) == 64
    with pytest.raises(WriteConfinementError):
        directory_identity("")


def test_directory_identity_rejects_link(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    alias = tmp_path / "alias"
    root.mkdir()
    try:
        alias.symlink_to(root, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(WriteConfinementError):
        directory_identity(alias)


def test_non_linux_child_api_fails_closed(tmp_path: Path) -> None:
    if sys.platform.startswith("linux"):
        pytest.skip("the Linux API is exercised in an isolated child below")
    root = tmp_path / "workspace"
    root.mkdir()
    identity = directory_identity(root)

    with pytest.raises(WriteConfinementUnavailable):
        enforce_linux_landlock_write_confinement(
            root,
            expected_device=identity.device,
            expected_inode=identity.inode,
        )

    with pytest.raises(WriteConfinementUnavailable):
        enforce_linux_landlock_write_confinement_fds(((0, identity.device, identity.inode),))


@pytest.mark.skipif(sys.platform != "win32", reason="mandatory-label inheritance is Windows-specific")
def test_windows_workspace_labeling_mutates_only_the_exact_empty_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    workspace = tmp_path / "exact-empty-root"
    workspace.mkdir()
    calls: list[tuple[int, bool, int]] = []
    original = module._set_windows_mandatory_integrity_label_handle

    def record(handle: int, *, directory: bool, integrity_rid: int) -> None:
        calls.append((handle, directory, integrity_rid))
        original(handle, directory=directory, integrity_rid=integrity_rid)

    monkeypatch.setattr(module, "_set_windows_mandatory_integrity_label_handle", record)
    monkeypatch.setattr(
        module,
        "_set_windows_mandatory_integrity_label",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pathname label mutation used")),
    )
    monkeypatch.setattr(
        module,
        "_snapshot_windows_private_tree",
        lambda _root: (_ for _ in ()).throw(AssertionError("root preparation must not traverse")),
    )

    prepared = prepare_windows_untrusted_integrity_workspace(workspace)

    assert prepared.entry_count == 1
    assert len(calls) == 1
    assert calls[0][0] > 0
    assert calls[0][1:] == (True, 0)
    child = workspace / "inherited-child"
    child.mkdir()
    assert module._windows_path_integrity_label(child) == (0, True)


@pytest.mark.skipif(sys.platform != "win32", reason="mandatory-label inheritance is Windows-specific")
def test_windows_workspace_labeling_refuses_a_populated_root_before_any_acl_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    workspace = tmp_path / "populated-root"
    workspace.mkdir()
    (workspace / "user-data.bin").write_bytes(b"preserve")
    calls: list[int] = []
    monkeypatch.setattr(
        module,
        "_set_windows_mandatory_integrity_label_handle",
        lambda handle, **_kwargs: calls.append(handle),
    )

    with pytest.raises(WriteConfinementError, match="before population"):
        prepare_windows_untrusted_integrity_workspace(workspace)

    assert calls == []
    assert (workspace / "user-data.bin").read_bytes() == b"preserve"


@pytest.mark.skipif(sys.platform != "win32", reason="no-delete Windows anchors are platform-specific")
def test_windows_workspace_cannot_be_renamed_at_the_handle_label_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    workspace = tmp_path / "workspace-label-race"
    moved = tmp_path / "workspace-label-race-moved"
    outside = tmp_path / "outside-label-race"
    workspace.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original = module._set_windows_mandatory_integrity_label_handle
    rename_denied = False

    def race(handle: int, *, directory: bool, integrity_rid: int) -> None:
        nonlocal rename_denied
        with pytest.raises(OSError):
            os.replace(workspace, moved)
        rename_denied = True
        original(handle, directory=directory, integrity_rid=integrity_rid)

    monkeypatch.setattr(module, "_set_windows_mandatory_integrity_label_handle", race)

    prepare_windows_untrusted_integrity_workspace(workspace)

    assert rename_denied is True
    assert workspace.is_dir()
    assert not moved.exists()
    assert sentinel.read_bytes() == b"outside-byte-identical"


@pytest.mark.skipif(sys.platform != "win32", reason="no-delete Windows stdio handles are platform-specific")
def test_windows_stdio_cannot_be_replaced_while_its_exact_handle_is_labeled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    stdio = tmp_path / "stdio"
    outside = tmp_path / "outside-stdio-race"
    stdio.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original = module._set_windows_mandatory_integrity_label_handle
    rename_denied = False

    def race(handle: int, *, directory: bool, integrity_rid: int) -> None:
        nonlocal rename_denied
        if not directory:
            candidates = tuple(stdio.glob(".restricted-stdin-*.bin"))
            assert len(candidates) == 1
            with pytest.raises(OSError):
                os.replace(candidates[0], sentinel)
            rename_denied = True
        original(handle, directory=directory, integrity_rid=integrity_rid)

    monkeypatch.setattr(module, "_set_windows_mandatory_integrity_label_handle", race)
    descriptor = module._exclusive_windows_stdio_file(
        stdio,
        "stdin",
        b"held-stdio",
        integrity_rid=0,
    )
    try:
        assert rename_denied is True
        assert module._read_windows_stdio_descriptor(descriptor, 64) == b"held-stdio"
        assert sentinel.read_bytes() == b"outside-byte-identical"
    finally:
        os.close(descriptor)


@pytest.mark.skipif(sys.platform != "win32", reason="no-delete Windows probe anchors are platform-specific")
def test_windows_probe_root_cannot_be_renamed_during_handle_labeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    moved = tmp_path / "probe-moved"
    outside = tmp_path / "outside-probe-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-byte-identical")
    original = module._set_windows_mandatory_integrity_label_handle
    rename_denied = False

    def race(handle: int, *, directory: bool, integrity_rid: int) -> None:
        nonlocal rename_denied
        if directory and not rename_denied:
            candidates = tuple(tmp_path.glob(".spritelab-confinement-probe-*"))
            assert len(candidates) == 1
            with pytest.raises(OSError):
                os.replace(candidates[0], moved)
            rename_denied = True
        original(handle, directory=directory, integrity_rid=integrity_rid)

    monkeypatch.setattr(module, "_set_windows_mandatory_integrity_label_handle", race)
    _probes, anchors = module._create_windows_untrusted_confinement_probes(tmp_path)
    try:
        assert rename_denied is True
        assert not moved.exists()
        assert sentinel.read_bytes() == b"outside-byte-identical"
    finally:
        anchors.close()


@pytest.mark.parametrize(
    ("is_restricted", "restricted_sid_hashes", "message"),
    [
        (True, (), "unbound inherited restricted-token state"),
        (False, ("a" * 64,), "restricting SIDs without a restricted token"),
    ],
)
def test_windows_startup_token_refuses_inconsistent_inherited_restriction_evidence(
    monkeypatch: pytest.MonkeyPatch,
    is_restricted: bool,
    restricted_sid_hashes: tuple[str, ...],
    message: str,
) -> None:
    import spritelab.utils.write_confinement as module

    closed: list[int] = []

    class FakeAdvapi32:
        create_restricted_token_called = False

        @staticmethod
        def OpenProcessToken(
            _process: object,
            _access: int,
            token: object,
        ) -> int:
            token._obj.value = 0x1234  # type: ignore[attr-defined]
            return 1

        def CreateRestrictedToken(self, *_args: object) -> int:
            self.create_restricted_token_called = True
            return 0

    class FakeKernel32:
        @staticmethod
        def GetCurrentProcess() -> int:
            return -1

        @staticmethod
        def CloseHandle(handle: ctypes.c_void_p) -> int:
            closed.append(int(handle.value))
            return 1

    advapi32 = FakeAdvapi32()
    monkeypatch.setattr(module, "_windows_advapi32", lambda: advapi32)
    monkeypatch.setattr(module, "_windows_kernel32", lambda: FakeKernel32())
    monkeypatch.setattr(
        module,
        "_windows_token_confinement",
        lambda _token: (is_restricted, 8192, True, restricted_sid_hashes),
    )

    with pytest.raises(WriteConfinementUnavailable, match=message):
        module._create_windows_low_startup_token()

    assert advapi32.create_restricted_token_called is False
    assert closed == [0x1234]


def test_windows_startup_token_refuses_changed_inherited_restriction_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as module

    closed: list[int] = []

    class FakeAdvapi32:
        create_restricted_token_called = False

        @staticmethod
        def OpenProcessToken(
            _process: object,
            _access: int,
            token: object,
        ) -> int:
            token._obj.value = 0x1234  # type: ignore[attr-defined]
            return 1

        @staticmethod
        def DuplicateTokenEx(
            _current: object,
            _access: int,
            _attributes: object,
            _impersonation_level: int,
            _token_type: int,
            token: object,
        ) -> int:
            token._obj.value = 0x5678  # type: ignore[attr-defined]
            return 1

        def CreateRestrictedToken(self, *_args: object) -> int:
            self.create_restricted_token_called = True
            return 0

    class FakeKernel32:
        @staticmethod
        def GetCurrentProcess() -> int:
            return -1

        @staticmethod
        def CloseHandle(handle: ctypes.c_void_p) -> int:
            closed.append(int(handle.value))
            return 1

    advapi32 = FakeAdvapi32()
    monkeypatch.setattr(module, "_windows_advapi32", lambda: advapi32)
    monkeypatch.setattr(module, "_windows_kernel32", lambda: FakeKernel32())

    def confinement(token: ctypes.c_void_p) -> tuple[bool, int, bool, tuple[str, ...]]:
        if int(token.value) == 0x1234:
            return True, 8192, True, ("a" * 64,)
        assert int(token.value) == 0x5678
        return True, 8192, True, ("b" * 64,)

    monkeypatch.setattr(module, "_windows_token_confinement", confinement)

    with pytest.raises(WriteConfinementError, match="changed inherited restrictions"):
        module._create_windows_low_startup_token()

    assert advapi32.create_restricted_token_called is False
    assert closed == [0x5678, 0x1234]


@pytest.mark.skipif(sys.platform != "win32", reason="restricted-token confinement is Windows-specific")
def test_windows_bootstrap_untrusted_denies_medium_and_low_outside_writes(tmp_path: Path) -> None:
    import msvcrt

    import spritelab.utils.write_confinement as module
    from spritelab.utils.pinned_executable import (
        activate_windows_suspended_process,
        close_windows_handle,
        pin_executable,
        read_executable_identity,
        verify_process_image,
    )

    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    prepare_windows_untrusted_integrity_workspace(workspace)
    (workspace / "tmp").mkdir()
    medium_outside = tmp_path / "medium-outside"
    low_outside = tmp_path / "low-outside"
    medium_outside.mkdir()
    low_outside.mkdir()
    medium_sentinel = medium_outside / "sentinel.bin"
    low_sentinel = low_outside / "sentinel.bin"
    medium_sentinel.write_bytes(b"medium-preserve")
    prepare_windows_low_integrity_workspace(low_outside)
    low_sentinel.write_bytes(b"low-preserve")
    script = """
import json
import importlib.util
import os
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("_bound_write_confinement", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("bound write-confinement helper unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
workspace, medium, low = map(Path, sys.argv[2:5])
import msvcrt
held_fd = msvcrt.open_osfhandle(int(sys.argv[7]), os.O_RDONLY | getattr(os, "O_BINARY", 0))
held_payload = os.read(held_fd, 1024)
evidence = module.windows_current_process_confinement_evidence(
    workspace,
    expected_device=int(sys.argv[5]),
    expected_inode=int(sys.argv[6]),
)
(workspace / "inside.bin").write_bytes(b"inside")
denied = []
for name, target in (("medium", medium), ("low", low)):
    try:
        target.write_bytes(b"compromised")
    except OSError:
        denied.append(name)
print(json.dumps({"denied": denied, "evidence": evidence.to_dict(), "held": held_payload.decode("ascii")}, sort_keys=True))
"""
    workspace_identity = directory_identity(workspace)
    executable = Path(sys.executable).resolve(strict=True)
    executable_identity = read_executable_identity(executable)
    environment = {
        "TEMP": str(workspace / "tmp"),
        "TMP": str(workspace / "tmp"),
        "SystemRoot": os.environ["SystemRoot"],
        "WINDIR": os.environ["WINDIR"],
    }
    job_handle = 0
    held_source = tmp_path / "held-source.bin"
    held_source.write_bytes(b"exact-held-source")
    held_descriptor = os.open(held_source, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        held_handle = int(msvcrt.get_osfhandle(held_descriptor))
        inherited_before = os.get_handle_inheritable(held_handle)
        with pin_executable(
            executable,
            expected_sha256=executable_identity.executable_sha256,
            expected_size=executable_identity.byte_count,
            expected_metadata_sha256=executable_identity.metadata_sha256,
        ) as pinned:
            process = create_windows_bootstrap_untrusted_process(
                [
                    pinned.launch_path,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    script,
                    str(Path(__file__).resolve().parents[1] / "src/spritelab/utils/write_confinement.py"),
                    str(workspace),
                    str(medium_sentinel),
                    str(low_sentinel),
                    str(workspace_identity.device),
                    str(workspace_identity.inode),
                    str(held_handle),
                ],
                cwd=workspace,
                env=environment,
                stdin_payload=b"",
                inherited_handles=(held_handle,),
            )
            assert os.get_handle_inheritable(held_handle) is inherited_before
            try:
                job_handle = activate_windows_suspended_process(
                    process,
                    verifier=lambda child: verify_process_image(child, pinned),
                )
                stdout, _stderr = process.communicate(timeout=30)
            finally:
                if job_handle:
                    close_windows_handle(job_handle)
                process.close()
    finally:
        os.close(held_descriptor)

    assert process.returncode == 0, (process.returncode, _stderr.decode("utf-8", "replace"))
    payload = json.loads(stdout)
    assert payload["denied"] == ["medium", "low"]
    assert payload["held"] == "exact-held-source"
    evidence = payload["evidence"]
    assert evidence["schema_version"] == "spritelab.write-confinement-evidence.v3"
    assert evidence["strategy"] == WINDOWS_PARENT_ANCHORS_STRATEGY
    assert evidence["startup_integrity_level_rid"] == 4096
    assert evidence["integrity_level_rid"] == 0
    assert evidence["workspace_integrity_level_rid"] == 0
    assert evidence["bootstrap_lowered_before_worker_import"] is True
    assert evidence["new_thread_integrity_level_rid"] == 0
    assert evidence["raise_to_low_denied"] is True
    assert evidence["medium_probe_write_denied"] is True
    assert evidence["low_world_probe_write_denied"] is True
    assert evidence["untrusted_world_outside_guaranteed"] is False
    assert evidence["job_kill_on_close"] is True
    assert evidence["job_active_process_limit"] == 1
    caller_restricted, _rid, _policy, caller_restricting_sids = module._windows_token_confinement(None)
    if caller_restricted:
        assert caller_restricting_sids
        assert evidence["restricted_token"] is True
    assert (workspace / "inside.bin").read_bytes() == b"inside"
    assert medium_sentinel.read_bytes() == b"medium-preserve"
    assert low_sentinel.read_bytes() == b"low-preserve"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-specific")
def test_linux_landlock_allows_owned_workspace_and_denies_outside_and_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"preserve")
    alias = workspace / "outside-alias"
    alias.symlink_to(outside, target_is_directory=True)
    identity = directory_identity(workspace)
    script = r"""
import json
import os
import sys
from pathlib import Path
from spritelab.utils.write_confinement import (
    WriteConfinementUnavailable,
    enforce_linux_landlock_write_confinement,
)
workspace = Path(sys.argv[1])
outside = Path(sys.argv[2])
try:
    evidence = enforce_linux_landlock_write_confinement(
        workspace,
        expected_device=int(sys.argv[3]),
        expected_inode=int(sys.argv[4]),
    )
except WriteConfinementUnavailable as exc:
    print(json.dumps({"supported": False, "error_type": type(exc).__name__}))
    raise SystemExit(0)
(workspace / "inside.txt").write_bytes(b"inside")
denied = []
for target in (outside / "new.txt", workspace / "outside-alias" / "sentinel.txt"):
    try:
        target.write_bytes(b"changed")
    except PermissionError:
        denied.append(target.name)
print(json.dumps({"supported": True, "denied": denied, "evidence": evidence.to_dict()}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(workspace),
            str(outside),
            str(identity.device),
            str(identity.inode),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        close_fds=True,
    )
    payload = json.loads(result.stdout)

    assert sentinel.read_bytes() == b"preserve"
    assert not (outside / "new.txt").exists()
    if payload["supported"]:
        assert (workspace / "inside.txt").read_bytes() == b"inside"
        assert sorted(payload["denied"]) == ["new.txt", "sentinel.txt"]
        assert payload["evidence"]["strategy"] == LINUX_LANDLOCK_STRATEGY
        assert payload["evidence"]["kernel_abi"] >= 3
        assert payload["evidence"]["no_new_privileges"] is True
        assert payload["evidence"]["paths_exposed"] is False
    else:
        # Unsupported kernels are an honest feature-unavailable result, never
        # a pathname-only fallback.
        assert payload["error_type"] == "WriteConfinementUnavailable"
        assert not (workspace / "inside.txt").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-specific")
def test_linux_landlock_binds_multiple_inherited_directory_handles(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    outside = tmp_path / "outside"
    first.mkdir()
    second.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"preserve")
    open_flags = (
        int(getattr(os, "O_PATH", os.O_RDONLY)) | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptors = (os.open(first, open_flags), os.open(second, open_flags))
    identities = tuple(os.fstat(descriptor) for descriptor in descriptors)
    script = r"""
import json
import sys
from pathlib import Path
from spritelab.utils.write_confinement import (
    WriteConfinementUnavailable,
    enforce_linux_landlock_write_confinement_fds,
)
first, second, outside = map(Path, sys.argv[1:4])
anchors = (
    (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])),
    (int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])),
)
try:
    evidence = enforce_linux_landlock_write_confinement_fds(anchors)
except WriteConfinementUnavailable as exc:
    print(json.dumps({"supported": False, "error_type": type(exc).__name__}))
    raise SystemExit(0)
(first / "one.txt").write_bytes(b"one")
(second / "two.txt").write_bytes(b"two")
denied = False
try:
    (outside / "sentinel.txt").write_bytes(b"changed")
except PermissionError:
    denied = True
print(json.dumps({"supported": True, "denied": denied, "evidence": evidence.to_dict()}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                str(first),
                str(second),
                str(outside),
                str(descriptors[0]),
                str(identities[0].st_dev),
                str(identities[0].st_ino),
                str(descriptors[1]),
                str(identities[1].st_dev),
                str(identities[1].st_ino),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            close_fds=True,
            pass_fds=descriptors,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    payload = json.loads(result.stdout)

    assert sentinel.read_bytes() == b"preserve"
    if payload["supported"]:
        assert (first / "one.txt").read_bytes() == b"one"
        assert (second / "two.txt").read_bytes() == b"two"
        assert payload["denied"] is True
        assert payload["evidence"]["strategy"] == LINUX_LANDLOCK_STRATEGY
        assert len(payload["evidence"]["root_identity_sha256"]) == 64
        assert payload["evidence"]["paths_exposed"] is False
    else:
        assert payload["error_type"] == "WriteConfinementUnavailable"
        assert not (first / "one.txt").exists()
        assert not (second / "two.txt").exists()

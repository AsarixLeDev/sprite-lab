"""Hold and verify an exact executable target across process creation.

This module is deliberately stdlib-only.  Callers remain responsible for
their own process containment and for terminating a child when verification
raises.  Windows callers that create a suspended child must keep the context
open through containment assignment, ``verify_process_image``, and only then
resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_EXECUTABLE_BYTES = 2 * 1024**3


class PinnedExecutableError(RuntimeError):
    """An executable target or launched process failed exact verification."""


@dataclass(frozen=True)
class ExecutableIdentity:
    resolved_path: Path
    executable_sha256: str
    byte_count: int
    metadata_sha256: str


@dataclass(frozen=True)
class PinnedExecutable(ExecutableIdentity):
    launch_path: str
    descriptor: int
    pass_fds: tuple[int, ...]


def read_executable_identity(path: str | Path) -> ExecutableIdentity:
    """Read a stable regular executable through a held descriptor."""

    resolved = Path(path).resolve(strict=True)
    descriptor = _open_pinned_file(resolved)
    try:
        digest, byte_count, metadata_sha256 = _read_descriptor(descriptor, resolved)
        return ExecutableIdentity(
            resolved_path=resolved,
            executable_sha256=digest,
            byte_count=byte_count,
            metadata_sha256=metadata_sha256,
        )
    finally:
        os.close(descriptor)


@contextmanager
def pin_executable(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_metadata_sha256: str | None = None,
) -> Iterator[PinnedExecutable]:
    """Pin expected executable bytes and hold them across a caller's launch."""

    if not _valid_digest(expected_sha256) or type(expected_size) is not int or expected_size <= 0:
        raise PinnedExecutableError("The expected executable identity is invalid.")
    if expected_metadata_sha256 is not None and not _valid_digest(expected_metadata_sha256):
        raise PinnedExecutableError("The expected executable metadata identity is invalid.")
    resolved = Path(path).resolve(strict=True)
    descriptor = _open_pinned_file(resolved)
    try:
        digest, byte_count, metadata_sha256 = _read_descriptor(descriptor, resolved)
        if (
            digest != expected_sha256
            or byte_count != expected_size
            or (expected_metadata_sha256 is not None and metadata_sha256 != expected_metadata_sha256)
        ):
            raise PinnedExecutableError("The executable target differs from its expected identity.")
        if sys.platform.startswith("linux"):
            launch_path = f"/proc/self/fd/{descriptor}"
            pass_fds = (descriptor,)
        elif os.name == "nt":
            launch_path = str(resolved)
            pass_fds = ()
        else:
            raise PinnedExecutableError("Pinned executable launch is unsupported on this platform.")
        pin = PinnedExecutable(
            resolved_path=resolved,
            executable_sha256=digest,
            byte_count=byte_count,
            metadata_sha256=metadata_sha256,
            launch_path=launch_path,
            descriptor=descriptor,
            pass_fds=pass_fds,
        )
        yield pin
        post_digest, post_count, post_metadata = _read_descriptor(descriptor, resolved)
        if post_digest != digest or post_count != byte_count or post_metadata != metadata_sha256:
            raise PinnedExecutableError("The executable target changed during process launch.")
    finally:
        os.close(descriptor)


def verify_process_image(process: Any, pin: PinnedExecutable) -> None:
    """Verify a live ``Popen``-like process against its held executable."""

    pid = int(getattr(process, "pid", 0))
    poll = getattr(process, "poll", None)
    if pid <= 0 or not callable(poll) or poll() is not None:
        raise PinnedExecutableError("The pinned executable process did not start safely.")
    if sys.platform.startswith("linux"):
        image_descriptor = os.open(
            f"/proc/{pid}/exe",
            os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
        )
        try:
            _verify_image_descriptors(image_descriptor, pin)
        finally:
            os.close(image_descriptor)
        return
    if os.name == "nt":
        image_path = _windows_process_image_path(pid)
        if os.path.normcase(os.path.abspath(image_path)) != os.path.normcase(str(pin.resolved_path)):
            raise PinnedExecutableError("The process image path differs from the pinned executable.")
        image_descriptor = _open_windows_pinned_file(Path(image_path))
        try:
            _verify_image_descriptors(image_descriptor, pin)
        finally:
            os.close(image_descriptor)
        return
    raise PinnedExecutableError("Pinned executable launch is unsupported on this platform.")


def activate_windows_suspended_process(
    process: Any,
    *,
    verifier: Any,
    assigner: Any = None,
    resumer: Any = None,
    closer: Any = None,
    terminator: Any = None,
) -> int:
    """Assign, verify, then resume a caller-created suspended Windows process."""

    if not callable(verifier):
        raise PinnedExecutableError("A suspended-process image verifier is required.")
    assign = assigner or _assign_windows_kill_job
    resume = resumer or _resume_windows_process
    close = closer or close_windows_handle
    terminate = terminator or _terminate_process
    handle: int | None = None
    try:
        handle = int(assign(process))
        verifier(process)
        resume(process)
        return handle
    except BaseException:
        terminate(process)
        if handle:
            close(handle)
        raise


def close_windows_handle(handle: int) -> None:
    """Close a native Windows handle retained for process containment."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(handle)


def linux_parent_death_signal(
    expected_parent_pid: int,
    *,
    libc_factory: Any = None,
    parent_pid: Any = None,
    exit_process: Any = None,
) -> Any:
    """Return a Linux pre-exec guard that dies with, or races, its parent."""

    def configure() -> None:
        if not sys.platform.startswith("linux") and libc_factory is None:
            raise PinnedExecutableError("Linux parent-death containment is unavailable.")
        if libc_factory is None:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
        else:
            libc = libc_factory()
        get_parent = parent_pid or os.getppid
        terminate = exit_process or os._exit
        death_signal = int(getattr(signal, "SIGKILL", 9))
        if libc.prctl(1, death_signal, 0, 0, 0) != 0 or get_parent() != expected_parent_pid:
            terminate(127)

    return configure


def _verify_image_descriptors(image_descriptor: int, pin: PinnedExecutable) -> None:
    image = os.fstat(image_descriptor)
    pinned = os.fstat(pin.descriptor)
    image_digest, image_count = _hash_descriptor(image_descriptor)
    pinned_digest, pinned_count = _hash_descriptor(pin.descriptor)
    if (
        (image.st_dev, image.st_ino) != (pinned.st_dev, pinned.st_ino)
        or image_digest != pin.executable_sha256
        or image_count != pin.byte_count
        or pinned_digest != pin.executable_sha256
        or pinned_count != pin.byte_count
    ):
        raise PinnedExecutableError("The process image differs from the pinned executable.")


def _assign_windows_kill_job(process: Any) -> int:
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError("unable to create contained process job")
    limits = ExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel32.CloseHandle(handle)
        raise OSError("unable to configure contained process job")
    process_handle = int(getattr(process, "_handle", 0))
    if not process_handle or not kernel32.AssignProcessToJobObject(handle, process_handle):
        kernel32.CloseHandle(handle)
        raise OSError("unable to assign contained process")
    return int(handle)


def _resume_windows_process(process: Any) -> None:
    import ctypes
    from ctypes import wintypes

    handle = int(getattr(process, "_handle", 0))
    if not handle:
        raise OSError("contained process handle is unavailable")
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(handle))
    if status != 0:
        raise OSError("unable to resume contained process")


def _terminate_process(process: Any) -> None:
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=5)
        except BaseException:
            return


def _open_pinned_file(path: Path) -> int:
    if os.name == "nt":
        return _open_windows_pinned_file(path)
    if sys.platform.startswith("linux"):
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        return os.open(path, flags)
    raise PinnedExecutableError("Pinned executable launch is unsupported on this platform.")


def _open_windows_pinned_file(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ: deny write and delete sharing
        None,
        3,  # OPEN_EXISTING
        0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if not handle or int(handle) == invalid:
        raise PinnedExecutableError("The executable target could not be pinned.")
    try:
        return int(msvcrt.open_osfhandle(int(handle), os.O_RDONLY | int(getattr(os, "O_BINARY", 0))))
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _read_descriptor(descriptor: int, path: Path) -> tuple[str, int, str]:
    before = os.fstat(descriptor)
    lexical_before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(lexical_before.st_mode)
        or int(getattr(lexical_before, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        or _metadata_identity(before) != _metadata_identity(lexical_before)
        or int(getattr(before, "st_nlink", 1)) != 1
        or before.st_size <= 0
        or before.st_size > _MAX_EXECUTABLE_BYTES
    ):
        raise PinnedExecutableError("The executable target is not a safe regular file.")
    digest, byte_count = _hash_descriptor(descriptor)
    after = os.fstat(descriptor)
    lexical_after = path.stat(follow_symlinks=False)
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(lexical_after)
        or byte_count != before.st_size
    ):
        raise PinnedExecutableError("The executable target changed while it was read.")
    return digest, byte_count, _stable_metadata_hash(after)


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), byte_count


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int | None]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(getattr(metadata, "st_nlink", 1)),
        getattr(metadata, "st_mtime_ns", None),
    )


def _stable_metadata_hash(metadata: os.stat_result) -> str:
    payload = {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "link_count": int(getattr(metadata, "st_nlink", 1)),
        "mtime_ns": getattr(metadata, "st_mtime_ns", None),
        "size": int(metadata.st_size),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _windows_process_image_path(pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        raise PinnedExecutableError("The process image is unavailable.")
    try:
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            raise PinnedExecutableError("The process image is unavailable.")
        return buffer.value
    finally:
        kernel32.CloseHandle(handle)


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ExecutableIdentity",
    "PinnedExecutable",
    "PinnedExecutableError",
    "activate_windows_suspended_process",
    "close_windows_handle",
    "linux_parent_death_signal",
    "pin_executable",
    "read_executable_identity",
    "verify_process_image",
]

"""Fail-closed infrastructure-smoke bundles for exploratory Playground use.

This module never discovers arbitrary paths.  Every artifact is derived from an
opaque smoke ID under one of the fixed project roots.  Ordinary ``--smoke``
runs are intentionally outside this contract and cannot be registered.
"""

from __future__ import annotations

import hashlib
import importlib.abc as importlib_abc
import importlib.machinery as importlib_machinery
import importlib.metadata as importlib_metadata
import importlib.util as importlib_util
import json
import math
import os
import platform
import re
import secrets
import stat
import sys
import sysconfig
import time
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from spritelab.utils.pinned_executable import (
    PinnedExecutable,
    PinnedExecutableError,
    pin_executable,
    read_executable_identity,
    verify_process_image,
)
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

SMOKE_PLAN_SCHEMA = "spritelab.training.smoke-plan.v1"
SMOKE_RUN_STATE_SCHEMA = "spritelab.training.smoke-run-state.v1"
SMOKE_DEVICE_RECEIPT_SCHEMA = "spritelab.training.smoke-device-receipt.v1"
SMOKE_EVIDENCE_SCHEMA = "spritelab.training.smoke-evidence.v1"
SMOKE_CHILD_ENVIRONMENT_SCHEMA = "spritelab.training.smoke-child-environment.v1"
SMOKE_INTERPRETER_SCHEMA = "spritelab.training.smoke-interpreter.v1"
SMOKE_ORCHESTRATION_CODE_SCHEMA = "spritelab.training.smoke-orchestration-code.v1"
SMOKE_RUNTIME_CLOSURE_SCHEMA = "spritelab.training.smoke-runtime-closure.v2"
SMOKE_SCOPE = "EXPLORATORY_INFRASTRUCTURE_SMOKE"
SMOKE_STATUS = "PROVISIONALLY_VERIFIED"
EXPLORATORY_REGISTRATION_SCHEMA = "spritelab.playground.exploratory-checkpoint-registration.v1"
_PUBLICATION_COMPLETION_SCHEMA = "spritelab.training.immutable-publication-completion.v1"
_PUBLICATION_COMPLETION_FILENAME = ".spritelab-publication-complete.json"
_MAX_PUBLICATION_ENTRIES = 4096
_MAX_PUBLICATION_DEPTH = 16
_PUBLICATION_CONVERGENCE_SECONDS = 5.0
SMOKE_ID_PATTERN = re.compile(r"^smoke-[0-9a-f]{20}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SMOKE_DEVICES = ("cpu", "cuda")
SMOKE_WALL_CLOCK_LIMIT_SECONDS = {"cpu": 10 * 60, "cuda": 15 * 60}
FALSE_ELIGIBILITY = {
    "production_eligible": False,
    "campaign_execution_evidence": False,
    "training_resume_eligible": False,
    "evaluation_eligible": False,
    "promotion_eligible": False,
}
_SANDBOXED_ENVIRONMENT_PATHS = (
    "APPDATA",
    "CUDA_CACHE_PATH",
    "HOME",
    "LOCALAPPDATA",
    "MPLCONFIGDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_HOME",
    "USERPROFILE",
    "XDG_CACHE_HOME",
)
_REQUIRED_RUNTIME_DISTRIBUTIONS = ("numpy", "pillow", "pyyaml", "torch")
_REQUIRED_RUNTIME_ROLES = ("destshared", "platstdlib", "runtime-libraries", "stdlib")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_MAX_RUNTIME_FILE_BYTES = 8 * 1024**3
_MAX_BOOTSTRAP_BYTES = 128 * 1024
_BOOTSTRAP_RELATIVE_PATH = "bootstrap/preflight.py"
_CUDA_QUALIFICATION_FILENAME = "cuda_determinism_qualification.json"
_CUDA_GUARANTEE_SCOPE = "same GPU model, driver, CUDA, cuDNN, Torch, code, and inputs only"
_CUDA_QUALIFICATION_KEYS = frozenset(
    {
        "qualified",
        "mode",
        "device",
        "steps",
        "interrupted_after",
        "repeated_forward_backward_bit_exact",
        "resume_bit_exact",
        "environment",
        "guarantee_scope",
        "cross_gpu_or_version_identity_claimed",
    }
)
_CUDA_ENVIRONMENT_KEYS = frozenset(
    {
        "platform",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cudnn_version",
        "gpus",
    }
)
_CUDA_GPU_KEYS = frozenset({"index", "name", "compute_capability", "total_memory_bytes"})
RUNTIME_EXECUTION_BYTE_POLICY = "trusted-installed-runtime-source-exact-native-resource-drift-detected-v1"
RUNTIME_BOUNDED_RESIDUALS = (
    "installed-runtime-bootstrap-is-a-trusted-baseline-before-loader-policy",
    "dependent-native-libraries-are-pre-post-hashed-but-not-fd-pinned",
    "runtime-resource-opens-are-prechecked-and-posthashed-but-not-fd-pinned",
)
_ACTIVE_OPERATION_CHECK: ContextVar[Callable[[], None] | None] = ContextVar(
    "spritelab_smoke_operation_check",
    default=None,
)
_OPERATION_CHECK_RUNNING: ContextVar[bool] = ContextVar(
    "spritelab_smoke_operation_check_running",
    default=False,
)


def _operation_checkpoint(check: Callable[[], None] | None) -> None:
    if _OPERATION_CHECK_RUNNING.get():
        return
    active = check if check is not None else _ACTIVE_OPERATION_CHECK.get()
    if active is not None:
        token = _OPERATION_CHECK_RUNNING.set(True)
        try:
            active()
        finally:
            _OPERATION_CHECK_RUNNING.reset(token)


def _operation_controlled(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("operation_check")
        active = requested if requested is not None else _ACTIVE_OPERATION_CHECK.get()
        token = _ACTIVE_OPERATION_CHECK.set(active)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE_OPERATION_CHECK.reset(token)

    return wrapped


_CHILD_PREFLIGHT_SOURCE = r"""
import sys as _y
if "_sha2" in _y.builtin_module_names:
    import _sha2 as _h
elif "_sha256" in _y.builtin_module_names:
    import _sha256 as _h
else:
    raise SystemExit(70)
import importlib.abc as _ia
import importlib.machinery as _im
import importlib.util as _iu
import datetime as _d
import json as _j
import os as _o
import re as _r
import stat as _s
import time as _t
import unicodedata as _u

_RP = int(getattr(_s, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
_CONTROL = None
_CONTROL_DEADLINE = None
_CONTROL_NEXT = 0.0
_CONTROL_ACTIVE = False


def _fail():
    raise SystemExit(70)


def _canonical(value):
    return _j.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_hash(value):
    return _h.sha256(_canonical(value)).hexdigest()


def _metadata(value):
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(getattr(value, "st_nlink", 1)),
        int(getattr(value, "st_mtime_ns", 0)),
    )


def _safe_directory(path):
    value = _o.lstat(path)
    if (
        not _s.S_ISDIR(value.st_mode)
        or _s.S_ISLNK(value.st_mode)
        or int(getattr(value, "st_file_attributes", 0)) & _RP
    ):
        _fail()
    return value


def _stable_file(path, maximum=8 * 1024 * 1024, controlled=True):
    if controlled:
        _checkpoint()
    lexical = _o.lstat(path)
    if (
        not _s.S_ISREG(lexical.st_mode)
        or _s.S_ISLNK(lexical.st_mode)
        or int(getattr(lexical, "st_file_attributes", 0)) & _RP
        or int(getattr(lexical, "st_nlink", 1)) != 1
        or lexical.st_size < 0
        or lexical.st_size > maximum
    ):
        _fail()
    flags = _o.O_RDONLY | int(getattr(_o, "O_BINARY", 0)) | int(getattr(_o, "O_NOFOLLOW", 0))
    descriptor = _o.open(path, flags)
    try:
        before = _o.fstat(descriptor)
        if _metadata(before) != _metadata(lexical):
            _fail()
        chunks = []
        total = 0
        while True:
            if controlled:
                _checkpoint()
            chunk = _o.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail()
        after = _o.fstat(descriptor)
        if _metadata(before) != _metadata(after) or _metadata(after) != _metadata(_o.lstat(path)):
            _fail()
        if controlled:
            _checkpoint()
        return b"".join(chunks)
    finally:
        _o.close(descriptor)


def _stable_digest(path, maximum=8 * 1024**3):
    _checkpoint()
    lexical = _o.lstat(path)
    if (
        not _s.S_ISREG(lexical.st_mode)
        or _s.S_ISLNK(lexical.st_mode)
        or int(getattr(lexical, "st_file_attributes", 0)) & _RP
        or int(getattr(lexical, "st_nlink", 1)) != 1
        or lexical.st_size < 0
        or lexical.st_size > maximum
    ):
        _fail()
    flags = _o.O_RDONLY | int(getattr(_o, "O_BINARY", 0)) | int(getattr(_o, "O_NOFOLLOW", 0))
    descriptor = _o.open(path, flags)
    try:
        before = _o.fstat(descriptor)
        if _metadata(before) != _metadata(lexical):
            _fail()
        digest = _h.sha256()
        total = 0
        while True:
            _checkpoint()
            chunk = _o.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > maximum:
                _fail()
        after = _o.fstat(descriptor)
        if _metadata(before) != _metadata(after) or _metadata(after) != _metadata(_o.lstat(path)):
            _fail()
        _checkpoint()
        return digest.hexdigest(), total
    finally:
        _o.close(descriptor)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail()
        value[key] = item
    return value


def _checkpoint(force=False):
    global _CONTROL_ACTIVE, _CONTROL_DEADLINE, _CONTROL_NEXT
    if _CONTROL is None or _CONTROL_ACTIVE:
        return
    now = _t.monotonic()
    if _CONTROL_DEADLINE is not None and now >= _CONTROL_DEADLINE:
        raise SystemExit(124)
    if not force and now < _CONTROL_NEXT:
        return
    state_path, smoke_id, device, plan_identity = _CONTROL
    _CONTROL_ACTIVE = True
    try:
        state = _j.loads(
            _stable_file(state_path, 4 * 1024 * 1024, controlled=False),
            object_pairs_hook=_unique_object,
        )
    except SystemExit:
        raise
    except BaseException:
        _fail()
    finally:
        _CONTROL_ACTIVE = False
    if (
        not isinstance(state, dict)
        or state.get("smoke_id") != smoke_id
        or state.get("device") != device
        or state.get("plan_identity") != plan_identity
    ):
        _fail()
    status = state.get("status")
    if status == "CANCELLED":
        raise SystemExit(130)
    if status == "TIMED_OUT":
        raise SystemExit(124)
    if status not in {"STARTING", "RUNNING"}:
        _fail()
    deadline_text = state.get("deadline_at")
    try:
        deadline = _d.datetime.fromisoformat(deadline_text)
        if deadline.tzinfo is None:
            _fail()
        remaining = (deadline.astimezone(_d.timezone.utc) - _d.datetime.now(_d.timezone.utc)).total_seconds()
    except SystemExit:
        raise
    except BaseException:
        _fail()
    if remaining <= 0:
        raise SystemExit(124)
    now = _t.monotonic()
    _CONTROL_DEADLINE = now + remaining
    _CONTROL_NEXT = now + 0.05


def _argument(name):
    indexes = [index for index, value in enumerate(_y.argv[:-1]) if value == name]
    if len(indexes) != 1:
        _fail()
    return _y.argv[indexes[0] + 1]


def _scan_sources(root):
    _checkpoint()
    source = _o.path.join(root, "src", "spritelab")
    _safe_directory(_o.path.join(root, "src"))
    inventory = {}

    def walk(directory, prefix):
        _checkpoint()
        before = _safe_directory(directory)
        with _o.scandir(directory) as stream:
            entries = sorted(stream, key=lambda item: item.name)
        _checkpoint()
        for entry in entries:
            _checkpoint()
            value = entry.stat(follow_symlinks=False)
            if _s.S_ISLNK(value.st_mode) or int(getattr(value, "st_file_attributes", 0)) & _RP:
                _fail()
            relative = f"{prefix}/{entry.name}"
            if _s.S_ISDIR(value.st_mode):
                walk(entry.path, relative)
            elif entry.name.endswith(".py"):
                payload = _stable_file(entry.path)
                inventory[relative] = (_h.sha256(payload).hexdigest(), len(payload))
        if _metadata(before) != _metadata(_safe_directory(directory)):
            _fail()
        _checkpoint()

    walk(source, "src/spritelab")
    _checkpoint(force=True)
    return inventory


def _portable_relative(value):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _u.normalize("NFC", value) != value
        or "\\" in value
        or "//" in value
        or value.startswith("/")
        or ":" in value
    ):
        _fail()
    parts = value.split("/")
    reserved = {"aux", "con", "nul", "prn"}
    reserved.update("com%d" % number for number in range(1, 10))
    reserved.update("lpt%d" % number for number in range(1, 10))
    for part in parts:
        folded = part.casefold()
        if (
            not part
            or part in {".", ".."}
            or part[-1:] in {".", " "}
            or folded.split(".", 1)[0] in reserved
            or any(character in '<>:"|?*' for character in part)
            or any(ord(character) < 32 for character in part)
        ):
            _fail()
    return parts


def _collision_key(value):
    return "/".join(_u.normalize("NFC", part).casefold() for part in _portable_relative(value))


def _runtime_scan(root, scan_roots):
    _checkpoint()
    actual = set()
    collisions = {}

    def add(relative):
        key = _collision_key(relative)
        previous = collisions.setdefault(key, relative)
        if previous != relative:
            _fail()
        actual.add(relative)

    def walk(directory, prefix):
        _checkpoint()
        before = _safe_directory(directory)
        with _o.scandir(directory) as stream:
            entries = sorted(stream, key=lambda item: (item.name.casefold(), item.name))
        _checkpoint()
        sibling_keys = {}
        for entry in entries:
            _checkpoint()
            relative = "%s/%s" % (prefix, entry.name)
            key = _collision_key(relative)
            previous = sibling_keys.setdefault(key, relative)
            if previous != relative:
                _fail()
            value = entry.stat(follow_symlinks=False)
            if _s.S_ISLNK(value.st_mode) or int(getattr(value, "st_file_attributes", 0)) & _RP:
                _fail()
            if _s.S_ISDIR(value.st_mode):
                walk(entry.path, relative)
            elif _s.S_ISREG(value.st_mode):
                add(relative)
            else:
                _fail()
        if _metadata(before) != _metadata(_safe_directory(directory)):
            _fail()
        _checkpoint()

    for name in scan_roots:
        _checkpoint()
        parts = _portable_relative(name)
        if len(parts) != 1:
            _fail()
        path = _o.path.join(root, name)
        value = _o.lstat(path)
        if _s.S_ISDIR(value.st_mode):
            walk(path, name)
        elif _s.S_ISREG(value.st_mode):
            add(name)
        else:
            _fail()
    _checkpoint(force=True)
    return actual


def _verify_runtime_closure(paths, closure):
    _checkpoint()
    if not isinstance(closure, dict):
        _fail()
    body = dict(closure)
    identity = body.pop("runtime_closure_identity", None)
    if (
        _stable_hash(body) != identity
        or closure.get("required_distributions") != ["numpy", "pillow", "pyyaml", "torch"]
        or closure.get("required_runtime_roles") != ["destshared", "platstdlib", "runtime-libraries", "stdlib"]
        or closure.get("inventory_policy")
        != "record-owned-plus-exact-third-party-and-standard-native-roots-v2"
        or closure.get("execution_byte_policy")
        != "trusted-installed-runtime-source-exact-native-resource-drift-detected-v1"
        or closure.get("bounded_residuals")
        != [
            "installed-runtime-bootstrap-is-a-trusted-baseline-before-loader-policy",
            "dependent-native-libraries-are-pre-post-hashed-but-not-fd-pinned",
            "runtime-resource-opens-are-prechecked-and-posthashed-but-not-fd-pinned",
        ]
        or not isinstance(closure.get("roots"), list)
        or not isinstance(closure.get("denied_scan_summary"), list)
    ):
        _fail()
    roots_by_hash = {
        _h.sha256(value.encode("utf-8", "surrogatepass")).hexdigest(): value
        for value in paths
    }
    expected_by_root = {}
    runtime_roles = set()
    denied_by_root = {}
    for root_record in closure["roots"]:
        _checkpoint()
        if not isinstance(root_record, dict):
            _fail()
        root_body = dict(root_record)
        root_identity = root_body.pop("root_identity", None)
        token = root_record.get("path_sha256")
        root = roots_by_hash.get(token)
        files = root_record.get("files")
        scan_roots = root_record.get("scan_roots")
        allowed_files = root_record.get("allowed_files")
        roles = root_record.get("roles")
        if (
            _stable_hash(root_body) != root_identity
            or root is None
            or not isinstance(files, list)
            or not isinstance(scan_roots, list)
            or not isinstance(allowed_files, list)
            or not allowed_files
            or any(not isinstance(value, str) for value in allowed_files)
            or allowed_files != sorted(set(allowed_files), key=lambda value: (value.casefold(), value))
            or not isinstance(roles, list)
            or roles != sorted(set(roles))
        ):
            _fail()
        runtime_roles.update(roles)
        root_stat = _safe_directory(root)
        directory_identity = _stable_hash(
            {"device": int(root_stat.st_dev), "inode": int(root_stat.st_ino), "mode": int(_s.S_IFMT(root_stat.st_mode))}
        )
        if directory_identity != root_record.get("directory_identity_sha256"):
            _fail()
        expected = {}
        collisions = {}
        for record in files:
            _checkpoint()
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                _fail()
            relative = record["path"]
            key = _collision_key(relative)
            previous = collisions.setdefault(key, relative)
            if previous != relative or relative in expected:
                _fail()
            digest, byte_count = _stable_digest(_o.path.join(root, *_portable_relative(relative)))
            if digest != record.get("sha256") or byte_count != record.get("byte_count"):
                _fail()
            expected[relative] = (digest, byte_count)
        if _runtime_scan(root, scan_roots) != set(expected):
            _fail()
        if any(relative not in expected for relative in allowed_files):
            _fail()
        denied_count = len(expected) - len(allowed_files)
        if denied_count:
            denied_by_root[token] = denied_count
        expected_by_root[token] = {relative: expected[relative] for relative in allowed_files}
    summary = closure.get("denied_scan_summary")
    if (
        any(
            not isinstance(row, dict)
            or set(row) != {"root_path_sha256", "file_count"}
            for row in summary
        )
        or {row["root_path_sha256"]: row["file_count"] for row in summary} != denied_by_root
    ):
        _fail()
    names = set()
    for distribution in closure.get("distributions", []):
        _checkpoint()
        if not isinstance(distribution, dict):
            _fail()
        distribution_body = dict(distribution)
        distribution_identity = distribution_body.pop("distribution_identity", None)
        name = distribution.get("name")
        files = distribution.get("files")
        expected = expected_by_root.get(distribution.get("root_path_sha256"))
        if (
            _stable_hash(distribution_body) != distribution_identity
            or not isinstance(name, str)
            or name in names
            or not isinstance(files, list)
            or expected is None
        ):
            _fail()
        names.add(name)
        for record in files:
            _checkpoint()
            if not isinstance(record, dict) or record.get("path") not in expected:
                _fail()
            if expected[record["path"]] != (record.get("sha256"), record.get("byte_count")):
                _fail()
    if (
        not {"numpy", "pillow", "pyyaml", "torch"}.issubset(names)
        or not {"destshared", "platstdlib", "runtime-libraries", "stdlib"}.issubset(runtime_roles)
    ):
        _fail()
    _checkpoint(force=True)
    return {roots_by_hash[token]: expected for token, expected in expected_by_root.items()}


def _install_linux_write_confinement(roots, descriptors):
    if not _y.platform.startswith("linux"):
        return
    if not isinstance(descriptors, list) or len(descriptors) != len(roots):
        _fail()
    import ctypes as _c
    import platform as _p

    if _p.machine().casefold() not in {"x86_64", "amd64", "aarch64", "arm64", "riscv64"}:
        _fail()
    libc = _c.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = _c.c_long
    abi = int(syscall(444, _c.c_void_p(), _c.c_size_t(0), _c.c_uint(1)))
    if abi < 3 or abi > 10:
        _fail()
    write_file = 1 << 1
    remove_dir = 1 << 4
    remove_file = 1 << 5
    make_char = 1 << 6
    make_dir = 1 << 7
    make_reg = 1 << 8
    make_sock = 1 << 9
    make_fifo = 1 << 10
    make_block = 1 << 11
    make_sym = 1 << 12
    refer = 1 << 13
    truncate = 1 << 14
    handled = (
        write_file
        | remove_dir
        | remove_file
        | make_char
        | make_dir
        | make_reg
        | make_sock
        | make_fifo
        | make_block
        | make_sym
        | refer
        | truncate
    )
    allowed = write_file | remove_dir | remove_file | make_dir | make_reg | refer | truncate

    class _Ruleset(_c.Structure):
        _fields_ = [("handled_access_fs", _c.c_uint64)]

    class _PathRule(_c.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", _c.c_uint64), ("parent_fd", _c.c_int32)]

    attribute = _Ruleset(handled_access_fs=handled)
    ruleset = int(syscall(444, _c.byref(attribute), _c.sizeof(attribute), _c.c_uint(0)))
    if ruleset < 0:
        _fail()
    try:
        for root, descriptor in zip(roots, descriptors, strict=True):
            opened = _o.fstat(descriptor)
            current = _safe_directory(root)
            if _metadata(opened) != _metadata(current):
                _fail()
            rule = _PathRule(allowed_access=allowed, parent_fd=descriptor)
            if int(syscall(445, ruleset, 1, _c.byref(rule), 0)) < 0:
                _fail()
        prctl = libc.prctl
        if int(prctl(38, 1, 0, 0, 0)) != 0 or int(syscall(446, ruleset, 0)) < 0:
            _fail()
    finally:
        for descriptor in descriptors:
            _o.close(descriptor)
        _o.close(ruleset)


class _BoundSourceLoader(_ia.Loader):
    def __init__(self, fullname, path, expected, package):
        self.fullname = fullname
        self.path = path
        self.expected = expected
        self.package = package

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        if fullname != self.fullname:
            _fail()
        return self.path

    def get_code(self, fullname):
        _checkpoint()
        if fullname != self.fullname:
            _fail()
        payload = _stable_file(self.path)
        if _h.sha256(payload).hexdigest() != self.expected:
            _fail()
        code = compile(payload, self.path, "exec", dont_inherit=True)
        _checkpoint()
        return code

    def is_package(self, fullname):
        if fullname != self.fullname:
            _fail()
        return self.package

    def exec_module(self, module):
        _checkpoint()
        exec(self.get_code(self.fullname), module.__dict__)
        _checkpoint(force=True)


class _BoundBytecodeLoader(_im.SourcelessFileLoader):
    def __init__(self, fullname, path, expected):
        super().__init__(fullname, path)
        self.expected = expected

    def get_data(self, path):
        _checkpoint()
        if _o.path.normcase(_o.path.abspath(path)) != _o.path.normcase(_o.path.abspath(self.path)):
            _fail()
        payload = _stable_file(self.path, 256 * 1024 * 1024)
        if _h.sha256(payload).hexdigest() != self.expected:
            _fail()
        _checkpoint()
        return payload


class _BoundExtensionLoader(_ia.Loader):
    def __init__(self, fullname, path, expected, byte_count):
        _checkpoint()
        self.fullname = fullname
        self.path = path
        self.expected = expected
        self.byte_count = byte_count
        self.posix_fd = None
        self.load_path = path
        if _y.platform.startswith("linux"):
            self.posix_fd = _o.open(
                path,
                _o.O_RDONLY | int(getattr(_o, "O_BINARY", 0)) | int(getattr(_o, "O_NOFOLLOW", 0)),
            )
            self.load_path = "/proc/self/fd/%d" % self.posix_fd
        self.loader = _im.ExtensionFileLoader(fullname, self.load_path)
        self.handle = None
        self._verify()
        _checkpoint()

    def _pin_windows(self):
        _checkpoint()
        if _o.name != "nt" or self.handle is not None:
            return
        import _winapi as _w

        try:
            handle = _w.CreateFile(self.path, 0x80000000, 0x00000001, 0, 3, 0x80, 0)
        except OSError:
            _fail()
        self.handle = (_w, handle)
        _checkpoint()

    def _close(self):
        if self.handle is not None:
            winapi, handle = self.handle
            self.handle = None
            winapi.CloseHandle(handle)
        if self.posix_fd is not None:
            descriptor = self.posix_fd
            self.posix_fd = None
            _o.close(descriptor)

    def _verify(self):
        _checkpoint()
        if self.posix_fd is not None:
            _o.lseek(self.posix_fd, 0, _o.SEEK_SET)
            digest = _h.sha256()
            byte_count = 0
            while True:
                _checkpoint()
                chunk = _o.read(self.posix_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                if byte_count > 8 * 1024**3:
                    _fail()
            _o.lseek(self.posix_fd, 0, _o.SEEK_SET)
            if digest.hexdigest() != self.expected or byte_count != self.byte_count:
                _fail()
        digest, byte_count = _stable_digest(self.path)
        if digest != self.expected or byte_count != self.byte_count:
            _fail()
        _checkpoint()

    def create_module(self, spec):
        _checkpoint()
        self._verify()
        self._pin_windows()
        try:
            module = self.loader.create_module(spec)
            self._verify()
            _checkpoint()
            return module
        except BaseException:
            self._close()
            raise

    def exec_module(self, module):
        _checkpoint()
        self._pin_windows()
        try:
            self._verify()
            self.loader.exec_module(module)
            _checkpoint()
            self._verify()
            _checkpoint(force=True)
        finally:
            self._close()

    def __del__(self):
        self._close()


class _BoundRuntimeFinder(_ia.MetaPathFinder):
    def __init__(self, roots):
        self.roots = {
            _o.path.normcase(_o.path.abspath(root)): expected
            for root, expected in roots.items()
        }
        self.audit_active = False

    def _owned(self, path):
        _checkpoint()
        absolute = _o.path.abspath(path)
        for root, expected in self.roots.items():
            _checkpoint()
            try:
                relative = _o.path.relpath(absolute, root).replace(_o.sep, "/")
            except ValueError:
                continue
            if relative != ".." and not relative.startswith("../"):
                return expected, relative
        return None

    def audit(self, event, arguments):
        if event != "open" or self.audit_active or not arguments or not isinstance(arguments[0], str):
            return
        _checkpoint()
        owned = self._owned(arguments[0])
        if owned is None:
            return
        expected, relative = owned
        record = expected.get(relative)
        if record is None:
            _fail()
        self.audit_active = True
        try:
            digest, byte_count = _stable_digest(arguments[0])
        finally:
            self.audit_active = False
        if (digest, byte_count) != record:
            _fail()
        _checkpoint()

    def find_spec(self, fullname, path=None, target=None):
        _checkpoint()
        if fullname == "spritelab" or fullname.startswith("spritelab."):
            return None
        spec = _im.PathFinder.find_spec(fullname, path, target)
        _checkpoint()
        if spec is None:
            return None
        locations = spec.submodule_search_locations
        if spec.origin in {None, "built-in", "frozen"}:
            if locations is not None:
                owned_locations = [self._owned(location) for location in locations]
                if all(owned is None for owned in owned_locations):
                    return None
                if any(owned is None for owned in owned_locations):
                    _fail()
                for owned in owned_locations:
                    if owned is None:
                        _fail()
                    expected, relative = owned
                    prefix = relative.rstrip("/") + "/"
                    if not any(name == relative or name.startswith(prefix) for name in expected):
                        _fail()
            return spec
        if not isinstance(spec.origin, str):
            _fail()
        owned = self._owned(spec.origin)
        if owned is None:
            _fail()
        expected, relative = owned
        record = expected.get(relative)
        if record is None:
            _fail()
        digest, byte_count = record
        package = locations is not None
        if relative.endswith(".py"):
            loader = _BoundSourceLoader(fullname, spec.origin, digest, package)
        elif relative.endswith(".pyc"):
            _fail()
        elif any(relative.endswith(suffix) for suffix in _im.EXTENSION_SUFFIXES):
            loader = _BoundExtensionLoader(fullname, spec.origin, digest, byte_count)
        else:
            _fail()
        return _iu.spec_from_file_location(
            fullname,
            loader.load_path if isinstance(loader, _BoundExtensionLoader) else spec.origin,
            loader=loader,
            submodule_search_locations=list(locations) if package else None,
        )


class _BoundSourceFinder(_ia.MetaPathFinder):
    def __init__(self, root, expected):
        self.root = _o.path.normcase(_o.path.abspath(root))
        self.expected = expected

    def find_spec(self, fullname, path=None, target=None):
        _checkpoint()
        if fullname != "spritelab" and not fullname.startswith("spritelab."):
            return None
        spec = _im.PathFinder.find_spec(fullname, path, target)
        _checkpoint()
        if spec is None or not isinstance(spec.origin, str) or not spec.origin.endswith(".py"):
            _fail()
        origin = _o.path.abspath(spec.origin)
        if _o.path.normcase(_o.path.realpath(origin)) != _o.path.normcase(origin):
            _fail()
        try:
            relative = _o.path.relpath(origin, self.root).replace(_o.sep, "/")
        except ValueError:
            _fail()
        if relative.startswith("../") or relative not in self.expected:
            _fail()
        package = spec.submodule_search_locations is not None
        loader = _BoundSourceLoader(fullname, origin, self.expected[relative], package)
        locations = [str(_o.path.dirname(origin))] if package else None
        return _iu.spec_from_file_location(
            fullname,
            origin,
            loader=loader,
            submodule_search_locations=locations,
        )


_BOUND_RUNTIME_STATE = None


def _spritelab_run_bound(module_name):
    _checkpoint(force=True)
    state = _BOUND_RUNTIME_STATE
    if not isinstance(state, tuple) or len(state) != 4:
        _fail()
    finder, runtime_paths, closure, run_name = state
    if run_name != module_name:
        _fail()
    try:
        import runpy as _runpy

        _checkpoint()
        _runpy.run_module(module_name, run_name="__main__")
        _checkpoint(force=True)
    finally:
        finder.audit_active = True
        try:
            _verify_runtime_closure(runtime_paths, closure)
            _checkpoint(force=True)
        finally:
            finder.audit_active = False


def _preflight(mode, source_sha256, source_byte_count):
    global _CONTROL
    if mode not in {"main", "worker"} or not _r.fullmatch(r"[0-9a-f]{64}", source_sha256):
        _fail()
    if not isinstance(source_byte_count, int) or isinstance(source_byte_count, bool) or source_byte_count <= 0:
        _fail()
    if any(name == "spritelab" or name.startswith("spritelab.") for name in _y.modules):
        _fail()
    smoke_id = _argument("--smoke-bundle-id" if mode == "main" else "--smoke-id")
    device = _argument("--smoke-device" if mode == "main" else "--device")
    plan_identity = _argument("--smoke-plan-identity" if mode == "main" else "--plan-identity")
    if not _r.fullmatch(r"smoke-[0-9a-f]{20}", smoke_id) or device not in {"cpu", "cuda"}:
        _fail()
    root = _o.path.realpath(_o.getcwd())
    cursor = root
    for part in ("artifacts", "training", "smokes", smoke_id):
        cursor = _o.path.join(cursor, part)
        _safe_directory(cursor)
    state_path = _o.path.join(cursor, "execution", device, "state.json")
    _CONTROL = (state_path, smoke_id, device, plan_identity)
    _checkpoint(force=True)
    completion_payload = _stable_file(
        _o.path.join(cursor, ".spritelab-publication-complete.json"),
        16 * 1024 * 1024,
    )
    completion = _j.loads(completion_payload, object_pairs_hook=_unique_object)
    if (
        not isinstance(completion, dict)
        or set(completion)
        != {
            "schema_version",
            "status",
            "publication_name",
            "inventory",
            "inventory_sha256",
            "completion_identity",
        }
        or completion.get("schema_version")
        != "spritelab.training.immutable-publication-completion.v1"
        or completion.get("status") != "COMPLETE"
        or completion.get("publication_name") != smoke_id
        or not isinstance(completion.get("inventory"), dict)
        or completion.get("inventory_sha256") != _stable_hash(completion["inventory"])
    ):
        _fail()
    completion_body = dict(completion)
    stored_completion_identity = completion_body.pop("completion_identity", None)
    if (
        not _r.fullmatch(r"[0-9a-f]{64}", str(stored_completion_identity or ""))
        or _stable_hash(completion_body) != stored_completion_identity
    ):
        _fail()
    plan_payload = _stable_file(_o.path.join(cursor, "plan.json"), 16 * 1024 * 1024)
    plan_record = completion["inventory"].get("plan.json")
    if (
        not isinstance(plan_record, dict)
        or set(plan_record) != {"kind", "sha256", "byte_count"}
        or plan_record.get("kind") != "file"
        or plan_record.get("sha256") != _h.sha256(plan_payload).hexdigest()
        or plan_record.get("byte_count") != len(plan_payload)
    ):
        _fail()
    plan = _j.loads(plan_payload, object_pairs_hook=_unique_object)
    if not isinstance(plan, dict) or plan.get("smoke_id") != smoke_id:
        _fail()
    identity_payload = dict(plan)
    stored_plan_identity = identity_payload.pop("plan_identity", None)
    if stored_plan_identity != plan_identity or _stable_hash(identity_payload) != plan_identity:
        _fail()
    configurations = plan.get("configurations")
    if not isinstance(configurations, dict) or not isinstance(configurations.get(device), dict):
        _fail()
    configuration = configurations[device]
    child = configuration.get("child_environment")
    environment = dict(_o.environ)
    writable_fd_value = environment.pop("SPRITELAB_WRITABLE_ROOT_FDS", None)
    if _y.platform.startswith("linux"):
        try:
            writable_fds = [int(value) for value in str(writable_fd_value).split(",")]
        except ValueError:
            _fail()
        if len(writable_fds) != 2 or len(set(writable_fds)) != 2 or any(value <= 2 for value in writable_fds):
            _fail()
        kept_fds = [_o.dup(value) for value in writable_fds]
    else:
        if writable_fd_value is not None:
            _fail()
        writable_fds = []
        kept_fds = []
    if not isinstance(child, dict) or _stable_hash(environment) != child.get("environment_sha256"):
        _fail()
    paths_value = environment.get("SPRITELAB_ISOLATED_PATHS")
    runtime_paths_value = environment.get("SPRITELAB_RUNTIME_ROOTS")
    if not isinstance(paths_value, str) or not isinstance(runtime_paths_value, str):
        _fail()
    paths = paths_value.split(_o.pathsep)
    runtime_paths = runtime_paths_value.split(_o.pathsep)
    path_hashes = [_h.sha256(value.encode("utf-8", "surrogatepass")).hexdigest() for value in paths]
    runtime_path_hashes = [
        _h.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
        for value in runtime_paths
    ]
    if (
        not paths
        or any(not value for value in paths)
        or len(paths) != child.get("isolated_import_path_count")
        or _stable_hash(path_hashes) != child.get("isolated_import_paths_sha256")
        or not runtime_paths
        or any(not value for value in runtime_paths)
        or len(runtime_paths) != child.get("runtime_root_count")
        or _stable_hash(runtime_path_hashes) != child.get("runtime_roots_sha256")
        or _o.path.normcase(_o.path.realpath(paths[0]))
        != _o.path.normcase(_o.path.join(root, "src"))
    ):
        _fail()
    writable_relative = configuration.get("writable_roots")
    if not isinstance(writable_relative, list) or len(writable_relative) != 2:
        _fail()
    state = _j.loads(_stable_file(state_path, 4 * 1024 * 1024), object_pairs_hook=_unique_object)
    writable_records = state.get("writable_roots") if isinstance(state, dict) else None
    if (
        not isinstance(writable_records, list)
        or len(writable_records) != 2
        or state.get("smoke_id") != smoke_id
        or state.get("device") != device
        or state.get("plan_identity") != plan_identity
    ):
        _fail()
    writable_roots = []
    collision_keys = set()
    for index, (relative, record) in enumerate(zip(writable_relative, writable_records, strict=True)):
        _checkpoint()
        if not isinstance(record, dict) or record.get("relative_path") != relative:
            _fail()
        key = _collision_key(relative)
        if key in collision_keys:
            _fail()
        collision_keys.add(key)
        path = _o.path.join(root, *_portable_relative(relative))
        metadata = _safe_directory(path)
        identity = _stable_hash(
            {"device": int(metadata.st_dev), "inode": int(metadata.st_ino), "mode": int(_s.S_IFMT(metadata.st_mode))}
        )
        if identity != record.get("identity_sha256"):
            _fail()
        if _y.platform.startswith("linux"):
            opened = _o.fstat(writable_fds[index])
            if _metadata(opened) != _metadata(metadata):
                _fail()
        writable_roots.append(path)
    _install_linux_write_confinement(writable_roots, writable_fds)
    if _y.platform.startswith("linux"):
        _o.environ["SPRITELAB_WRITABLE_ROOT_FDS"] = ",".join(str(value) for value in kept_fds)
    interpreter = plan.get("interpreter")
    if not isinstance(interpreter, dict):
        _fail()
    image_path = _o.path.realpath(f"/proc/{_o.getpid()}/exe") if _y.platform.startswith("linux") else _y.executable
    image = _stable_file(image_path, 2 * 1024**3)
    if (
        len(image) != interpreter.get("byte_count")
        or _h.sha256(image).hexdigest() != interpreter.get("executable_sha256")
    ):
        _fail()
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict):
        _fail()
    code = bindings.get("training_code_identity")
    if not isinstance(code, dict) or code.get("sha256") != bindings.get("training_code_identity_sha256"):
        _fail()
    code_payload = dict(code)
    code_identity = code_payload.pop("sha256", None)
    if _stable_hash(code_payload) != code_identity or not isinstance(code.get("files"), list):
        _fail()
    expected = {}
    for record in code["files"]:
        _checkpoint()
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"].startswith("src/spritelab/")
            or not record["path"].endswith(".py")
            or not _r.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""))
            or record["path"] in expected
        ):
            _fail()
        expected[record["path"]] = record["sha256"]
    actual = _scan_sources(root)
    if {path: value[0] for path, value in actual.items()} != expected:
        _fail()
    orchestration = plan.get("orchestration_code")
    bootstrap_payload = orchestration.get("bootstrap_payload") if isinstance(orchestration, dict) else None
    if (
        not isinstance(orchestration, dict)
        or orchestration.get("preflight_sha256") != source_sha256
        or not isinstance(bootstrap_payload, dict)
        or bootstrap_payload.get("relative_path") != "bootstrap/preflight.py"
        or bootstrap_payload.get("sha256") != source_sha256
        or bootstrap_payload.get("byte_count") != source_byte_count
        or bootstrap_payload.get("max_byte_count") != 128 * 1024
        or not isinstance(orchestration.get("inventory"), dict)
    ):
        _fail()
    for path, record in orchestration["inventory"].items():
        _checkpoint()
        if (
            path not in actual
            or not isinstance(record, dict)
            or actual[path] != (record.get("sha256"), record.get("byte_count"))
        ):
            _fail()
    runtime_roots = _verify_runtime_closure(runtime_paths, plan.get("runtime_closure"))
    importable_paths = []
    for value in [*paths, *runtime_paths]:
        _checkpoint()
        if value not in importable_paths:
            importable_paths.append(value)
    _y.dont_write_bytecode = True
    _y.path[:0] = importable_paths
    runtime_finder = _BoundRuntimeFinder(runtime_roots)
    _y.meta_path.insert(0, runtime_finder)
    _y.meta_path.insert(0, _BoundSourceFinder(root, expected))
    _y.addaudithook(runtime_finder.audit)
    global _BOUND_RUNTIME_STATE
    _BOUND_RUNTIME_STATE = (
        runtime_finder,
        runtime_paths,
        plan.get("runtime_closure"),
        "spritelab" if mode == "main" else "spritelab.training.smoke_worker",
    )
    _checkpoint(force=True)


def _spritelab_smoke_preflight(mode, source_sha256, source_byte_count):
    try:
        _preflight(mode, source_sha256, source_byte_count)
    except SystemExit:
        raise
    except BaseException:
        _fail()
"""
_CHILD_PREFLIGHT_SOURCE = "\n".join(line.rstrip() for line in _CHILD_PREFLIGHT_SOURCE.splitlines() if line.strip())
_CHILD_PREFLIGHT_BYTES = _CHILD_PREFLIGHT_SOURCE.encode("utf-8")
_CHILD_PREFLIGHT_SHA256 = hashlib.sha256(_CHILD_PREFLIGHT_BYTES).hexdigest()


def _compact_bootstrap_loader(mode: str) -> str:
    """Return a small, project-import-free loader for the bound preflight file."""

    if mode not in {"main", "worker"}:
        raise ValueError("unsupported bootstrap mode")
    flag = "--smoke-bundle-id" if mode == "main" else "--smoke-id"
    module = "spritelab" if mode == "main" else "spritelab.training.smoke_worker"
    source = f"""
import sys as _y
if any(name == "spritelab" or name.startswith("spritelab.") for name in _y.modules):
    raise SystemExit(70)
import os as _o
import stat as _s

def _fail():
    raise SystemExit(70)

def _meta(value):
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(getattr(value, "st_nlink", 1)), int(getattr(value, "st_mtime_ns", 0)),
    )

def _directory(path):
    value = _o.lstat(path)
    if (
        not _s.S_ISDIR(value.st_mode)
        or _s.S_ISLNK(value.st_mode)
        or int(getattr(value, "st_file_attributes", 0))
        & int(getattr(_s, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    ):
        _fail()
    return value

try:
    expected_sha256 = _y.argv[1]
    expected_byte_count = int(_y.argv[2])
except (IndexError, TypeError, ValueError):
    _fail()
del _y.argv[1:3]
if (
    len(expected_sha256) != 64
    or any(character not in "0123456789abcdef" for character in expected_sha256)
    or expected_byte_count <= 0
    or expected_byte_count > {_MAX_BOOTSTRAP_BYTES}
):
    _fail()
indexes = [index for index, value in enumerate(_y.argv[:-1]) if value == {flag!r}]
if len(indexes) != 1:
    _fail()
smoke_id = _y.argv[indexes[0] + 1]
if (
    not isinstance(smoke_id, str)
    or len(smoke_id) != 26
    or not smoke_id.startswith("smoke-")
    or any(character not in "0123456789abcdef" for character in smoke_id[6:])
):
    _fail()
private_cwd = _o.path.realpath(_o.getcwd())
root = private_cwd
if _o.name == "nt":
    bound_root = getattr(_y, "_spritelab_windows_project_root", None)
    device_flag = {("--smoke-device" if mode == "main" else "--device")!r}
    try:
        device_value = _y.argv[_y.argv.index(device_flag) + 1]
    except (ValueError, IndexError):
        _fail()
    if device_value not in ("cpu", "cuda") or not isinstance(bound_root, str) or not bound_root:
        _fail()
    root = _o.path.realpath(bound_root)
    expected_cwd = _o.path.join(
        root, "artifacts", "training", "smokes", smoke_id, "execution", device_value
    )
    if _o.path.normcase(private_cwd) != _o.path.normcase(_o.path.realpath(expected_cwd)):
        _fail()
    _o.chdir(root)
cursor = root
parents = []
for part in ("artifacts", "training", "smokes", smoke_id, "bootstrap"):
    cursor = _o.path.join(cursor, part)
    parents.append((cursor, _meta(_directory(cursor))))
path = _o.path.join(cursor, "preflight.py")
lexical = _o.lstat(path)
attributes = int(getattr(lexical, "st_file_attributes", 0))
if (
    not _s.S_ISREG(lexical.st_mode)
    or _s.S_ISLNK(lexical.st_mode)
    or attributes & int(getattr(_s, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    or int(getattr(lexical, "st_nlink", 1)) != 1
    or int(lexical.st_size) != expected_byte_count
):
    _fail()
flags = _o.O_RDONLY | int(getattr(_o, "O_BINARY", 0)) | int(getattr(_o, "O_NOFOLLOW", 0))
descriptor = _o.open(path, flags)
try:
    before = _o.fstat(descriptor)
    if _meta(before) != _meta(lexical):
        _fail()
    chunks = []
    total = 0
    digest = __import__(
        "_sha2" if "_sha2" in _y.builtin_module_names else "_sha256"
    ).sha256()
    while True:
        chunk = _o.read(descriptor, min(1024 * 1024, {_MAX_BOOTSTRAP_BYTES} - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        total += len(chunk)
        if total > {_MAX_BOOTSTRAP_BYTES}:
            _fail()
    after = _o.fstat(descriptor)
finally:
    _o.close(descriptor)
if (
    _meta(before) != _meta(after)
    or _meta(after) != _meta(_o.lstat(path))
    or total != expected_byte_count
    or digest.hexdigest() != expected_sha256
):
    _fail()
for parent_path, parent_identity in parents:
    if _meta(_directory(parent_path)) != parent_identity:
        _fail()
payload = b"".join(chunks)
exec(compile(payload, path, "exec", dont_inherit=True), globals())
_spritelab_smoke_preflight({mode!r}, expected_sha256, expected_byte_count)
_spritelab_run_bound({module!r})
"""
    return "\n".join(line.rstrip() for line in source.splitlines() if line.strip())


_ISOLATED_MAIN_BOOTSTRAP = _compact_bootstrap_loader("main")
_ISOLATED_WORKER_BOOTSTRAP = _compact_bootstrap_loader("worker")
_ORCHESTRATION_CODE_PATHS = (
    "src/spritelab/__main__.py",
    "src/spritelab/training/cli/experiment_cmds.py",
    "src/spritelab/training/smoke_bundle.py",
    "src/spritelab/training/smoke_runner.py",
    "src/spritelab/training/smoke_worker.py",
    "src/spritelab/utils/pinned_executable.py",
)


class SmokeBundleError(ValueError):
    """A public, path-free smoke-bundle contract failure."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass(frozen=True)
class VerifiedSmokeCheckpoint:
    weights: str
    path: Path
    sha256: str
    byte_count: int
    step: int
    variant: str


@dataclass(frozen=True)
class VerifiedSmokeBundle:
    evidence: Mapping[str, Any]
    checkpoints: tuple[VerifiedSmokeCheckpoint, ...]


PinnedSmokeInterpreter = PinnedExecutable


def smoke_id_for_campaign(campaign_identity: str, preparation_nonce: str) -> str:
    if not SHA256_PATTERN.fullmatch(str(campaign_identity)):
        raise SmokeBundleError("campaign_identity", "The campaign identity is malformed.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,79}", str(preparation_nonce)):
        raise SmokeBundleError("preparation_nonce", "A fresh smoke preparation nonce is required.")
    identity = hashlib.sha256(f"{campaign_identity}:{preparation_nonce}".encode()).hexdigest()
    return f"smoke-{identity[:20]}"


def artifact_bundle_directory(project_root: str | Path, smoke_id: str) -> Path:
    _require_smoke_id(smoke_id)
    return Path(project_root).resolve() / "artifacts" / "training" / "smokes" / smoke_id


def run_bundle_directory(project_root: str | Path, smoke_id: str) -> Path:
    _require_smoke_id(smoke_id)
    return Path(project_root).resolve() / "runs" / "v3" / "training-smokes" / smoke_id


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + ("\n" if pretty else "")).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], *, code: str) -> None:
    if set(value) != set(expected) or any(not isinstance(key, str) for key in value):
        raise SmokeBundleError(code, "A persisted smoke artifact has an inexact schema.")


def prepare_smoke_environment_binding(
    project_root: str | Path,
    smoke_id: str,
    device: str,
    *,
    inherited_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind a minimal child environment without persisting private path values."""

    root = Path(project_root).resolve()
    normalized = _require_device(device)
    _require_smoke_id(smoke_id)
    public = _public_smoke_environment(normalized)
    inherited = _allowed_inherited_environment(inherited_environment or os.environ)
    temporary_relative = f"artifacts/training/smokes/{smoke_id}/execution/{normalized}/temp"
    temporary = _fixed_relative(root, temporary_relative)
    import_paths = _isolated_import_paths(root)
    runtime_paths = _runtime_verification_paths(root)
    child = _compose_child_environment(
        public,
        inherited,
        temporary,
        import_paths,
        runtime_paths,
        _lexical_interpreter_path(),
    )
    binding = {
        "schema_version": SMOKE_CHILD_ENVIRONMENT_SCHEMA,
        "inherited_names": sorted(inherited),
        "temporary_root": temporary_relative,
        "sandboxed_path_variables": list(_SANDBOXED_ENVIRONMENT_PATHS),
        "isolated_import_path_count": len(import_paths),
        "isolated_import_paths_sha256": stable_hash(
            [hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in import_paths]
        ),
        "runtime_root_count": len(runtime_paths),
        "runtime_roots_sha256": stable_hash(
            [hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in runtime_paths]
        ),
        "environment_sha256": stable_hash(child),
    }
    return public, binding


def build_smoke_child_environment(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    inherited_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Reconstruct and verify the exact full environment bound by a plan."""

    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    record = dict(validated["configurations"])[normalized]
    public = _validate_environment_record(record.get("environment"), normalized)
    binding = _validate_child_environment_binding(record.get("child_environment"))
    inherited_source = inherited_environment or os.environ
    allowed = _allowed_inherited_environment(inherited_source)
    bound_interpreter = Path(str(inherited_source.get("SPRITELAB_BOUND_INTERPRETER") or _lexical_interpreter_path()))
    if sorted(allowed) != binding["inherited_names"]:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The allowlisted host environment changed after smoke preparation; prepare a fresh bundle.",
        )
    temporary = _fixed_relative(root, str(binding["temporary_root"]))
    expected_relative = f"artifacts/training/smokes/{validated['smoke_id']}/execution/{normalized}/temp"
    if str(binding["temporary_root"]) != expected_relative:
        raise SmokeBundleError("smoke_environment", "The smoke temporary environment root is invalid.")
    import_paths = _isolated_import_paths(root)
    runtime_paths = _runtime_verification_paths(root)
    path_hash = stable_hash(
        [hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in import_paths]
    )
    if (
        binding["isolated_import_path_count"] != len(import_paths)
        or binding["isolated_import_paths_sha256"] != path_hash
        or binding["runtime_root_count"] != len(runtime_paths)
        or binding["runtime_roots_sha256"]
        != stable_hash([hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in runtime_paths])
    ):
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The isolated dependency environment import path changed after preparation; prepare a fresh bundle.",
        )
    child = _compose_child_environment(
        public,
        allowed,
        temporary,
        import_paths,
        runtime_paths,
        bound_interpreter,
    )
    if stable_hash(child) != binding["environment_sha256"]:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The exact host environment bound by the smoke plan changed; prepare a fresh bundle.",
        )
    return child


def prepare_smoke_interpreter_binding(*, lexical_path: str | Path | None = None) -> dict[str, Any]:
    """Hash the exact Python binary while exposing no executable path."""

    lexical = _lexical_interpreter_path() if lexical_path is None else Path(lexical_path).absolute()
    lexical_metadata = _lexical_interpreter_metadata(lexical)
    executable = lexical.resolve(strict=True)
    try:
        executable_identity = read_executable_identity(executable)
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter",
            "The isolated Python interpreter is unsafe.",
        ) from exc
    digest = executable_identity.executable_sha256
    byte_count = executable_identity.byte_count
    path_identity = hashlib.sha256(
        os.path.normcase(str(executable_identity.resolved_path)).encode("utf-8", "surrogatepass")
    ).hexdigest()
    return finalize_identity(
        {
            "schema_version": SMOKE_INTERPRETER_SCHEMA,
            "implementation": sys.implementation.name,
            "implementation_version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "cache_tag": sys.implementation.cache_tag,
            "python_implementation": platform.python_implementation(),
            "executable_sha256": digest,
            "byte_count": byte_count,
            "lexical_path_sha256": hashlib.sha256(
                os.path.normcase(str(lexical)).encode("utf-8", "surrogatepass")
            ).hexdigest(),
            "lexical_metadata_sha256": stable_hash(lexical_metadata),
            "lexical_kind": lexical_metadata["kind"],
            "resolved_path_sha256": path_identity,
            "resolved_metadata_sha256": executable_identity.metadata_sha256,
            "isolated_startup": True,
            "isolated_flags": ["-I", "-B", "-S"],
        },
        "interpreter_identity",
    )


def validate_smoke_interpreter(
    plan: Mapping[str, Any],
    *,
    lexical_path: str | Path | None = None,
) -> dict[str, Any]:
    validated = validate_plan(plan)
    expected = _validate_interpreter_record(validated.get("interpreter"))
    actual = prepare_smoke_interpreter_binding(lexical_path=lexical_path)
    if actual != expected:
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    return actual


@contextmanager
def pinned_smoke_interpreter(
    plan: Mapping[str, Any],
    *,
    lexical_path: str | Path | None = None,
) -> Iterator[PinnedSmokeInterpreter]:
    """Hold the plan-bound executable target across process creation.

    Windows opens the target without write/delete sharing.  Linux executes the
    already-open descriptor through ``/proc/self/fd``.  Both paths are checked
    again while the descriptor is still held after ``Popen`` returns.
    """

    validated = validate_plan(plan)
    expected = _validate_interpreter_record(validated.get("interpreter"))
    lexical = _lexical_interpreter_path() if lexical_path is None else Path(lexical_path).absolute()
    lexical_metadata = _lexical_interpreter_metadata(lexical)
    if (
        hashlib.sha256(os.path.normcase(str(lexical)).encode("utf-8", "surrogatepass")).hexdigest()
        != expected["lexical_path_sha256"]
        or stable_hash(lexical_metadata) != expected["lexical_metadata_sha256"]
        or lexical_metadata["kind"] != expected["lexical_kind"]
    ):
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    resolved = lexical.resolve(strict=True)
    if (
        hashlib.sha256(os.path.normcase(str(resolved)).encode("utf-8", "surrogatepass")).hexdigest()
        != expected["resolved_path_sha256"]
    ):
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    try:
        with pin_executable(
            resolved,
            expected_sha256=str(expected["executable_sha256"]),
            expected_size=int(expected["byte_count"]),
            expected_metadata_sha256=str(expected["resolved_metadata_sha256"]),
        ) as pin:
            yield pin
            if stable_hash(_lexical_interpreter_metadata(lexical)) != expected["lexical_metadata_sha256"]:
                raise SmokeBundleError(
                    "smoke_interpreter_changed",
                    "The exact isolated Python interpreter changed during process launch.",
                )
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed during process launch.",
        ) from exc


def verify_pinned_process_image(process: Any, pin: PinnedSmokeInterpreter) -> None:
    """Prove that a newly created process is executing the held target."""

    try:
        verify_process_image(process, pin)
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter_launch",
            "The isolated Python process image differs from the pinned interpreter.",
        ) from exc


@_operation_controlled
def prepare_smoke_orchestration_code_identity(
    project_root: str | Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    inventory: dict[str, dict[str, Any]] = {}
    for relative in _ORCHESTRATION_CODE_PATHS:
        _operation_checkpoint(operation_check)
        payload = read_stable_single_link_bytes(
            _fixed_relative(root, relative),
            boundary=root,
            max_bytes=8 * 1024 * 1024,
            operation_check=operation_check,
        )
        _operation_checkpoint(operation_check)
        inventory[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
    _operation_checkpoint(operation_check)
    result = finalize_identity(
        {
            "schema_version": SMOKE_ORCHESTRATION_CODE_SCHEMA,
            "paths": list(_ORCHESTRATION_CODE_PATHS),
            "inventory": inventory,
            "preflight_sha256": _CHILD_PREFLIGHT_SHA256,
            "bootstrap_payload": {
                "relative_path": _BOOTSTRAP_RELATIVE_PATH,
                "sha256": _CHILD_PREFLIGHT_SHA256,
                "byte_count": len(_CHILD_PREFLIGHT_BYTES),
                "max_byte_count": _MAX_BOOTSTRAP_BYTES,
            },
            "bootstrap_sha256": stable_hash(
                {
                    "main": _ISOLATED_MAIN_BOOTSTRAP,
                    "worker": _ISOLATED_WORKER_BOOTSTRAP,
                }
            ),
        },
        "orchestration_code_identity",
    )
    _operation_checkpoint(operation_check)
    return result


@_operation_controlled
def validate_smoke_orchestration_code(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    validated = validate_plan(plan)
    expected = _validate_orchestration_code_record(validated.get("orchestration_code"))
    try:
        actual = prepare_smoke_orchestration_code_identity(
            project_root,
            operation_check=operation_check,
        )
    except (OSError, SmokeBundleError) as exc:
        raise SmokeBundleError(
            "smoke_orchestration_code_changed",
            "The smoke orchestration code is unavailable; prepare a fresh bundle.",
        ) from exc
    if actual != expected:
        raise SmokeBundleError(
            "smoke_orchestration_code_changed",
            "The smoke orchestration code changed after preparation; prepare a fresh bundle.",
        )
    _operation_checkpoint(operation_check)
    return actual


@_operation_controlled
def prepare_smoke_runtime_closure(
    project_root: str | Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Bind the installed runtime bytes available to a smoke child.

    Distribution RECORD inventories establish ownership while a recursive scan
    of every owned top-level import root also binds generated bytecode, native
    modules, vendored libraries, and other supplemental files that Python or a
    native loader could consume.  Absolute site-package paths never enter the
    plan; children match roots through SHA-256 path tokens from their already
    bound isolated import-path environment.
    """

    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    import_paths: list[Path] = []
    for value in _isolated_import_paths(root):
        _operation_checkpoint(operation_check)
        import_paths.append(Path(value).resolve())
    site_roots: list[Path] = []
    for value in import_paths[1:]:
        _operation_checkpoint(operation_check)
        if value.name.casefold() in {"site-packages", "dist-packages"}:
            site_roots.append(value)
    if not site_roots:
        raise SmokeBundleError("smoke_runtime_closure", "No isolated installed-distribution root is available.")
    for site_root in site_roots:
        _operation_checkpoint(operation_check)
        _require_directory_metadata(site_root.lstat())

    discovered: dict[str, importlib_metadata.Distribution] = {}
    _operation_checkpoint(operation_check)
    for distribution in importlib_metadata.distributions(path=[str(value) for value in site_roots]):
        _operation_checkpoint(operation_check)
        name = _distribution_name(distribution)
        if name in discovered:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "The isolated runtime contains an ambiguous installed distribution.",
            )
        discovered[name] = distribution
    _operation_checkpoint(operation_check)

    selected: dict[str, importlib_metadata.Distribution] = {}
    pending = list(_REQUIRED_RUNTIME_DISTRIBUTIONS)
    while pending:
        _operation_checkpoint(operation_check)
        name = _canonical_distribution_name(pending.pop(0))
        if name in selected:
            continue
        distribution = discovered.get(name)
        if distribution is None:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                f"The required isolated runtime distribution {name} is unavailable.",
            )
        selected[name] = distribution
        pending.extend(_required_distribution_dependencies(distribution))
        _operation_checkpoint(operation_check)

    distribution_specs: list[tuple[str, str, Path, list[str]]] = []
    root_owned_paths: dict[Path, set[str]] = {value: set() for value in site_roots}
    root_scan_roots: dict[Path, set[str]] = {value: set() for value in site_roots}
    root_roles: dict[Path, set[str]] = {value: {"site-packages"} for value in site_roots}
    for standard_root, roles, scan_roots in _standard_runtime_root_specs():
        _operation_checkpoint(operation_check)
        root_owned_paths.setdefault(standard_root, set())
        root_scan_roots.setdefault(standard_root, set()).update(scan_roots)
        root_roles.setdefault(standard_root, set()).update(roles)
    for name, distribution in sorted(selected.items()):
        _operation_checkpoint(operation_check)
        site_root = _distribution_site_root(distribution, site_roots)
        package_paths = distribution.files
        _operation_checkpoint(operation_check)
        if package_paths is None:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "An installed runtime distribution has no exact file inventory.",
            )
        owned_paths: list[str] = []
        for package_path in package_paths:
            _operation_checkpoint(operation_check)
            located = Path(os.path.abspath(distribution.locate_file(package_path)))
            _operation_checkpoint(operation_check)
            try:
                relative_path = located.relative_to(site_root)
            except ValueError:
                # Console scripts and headers outside site-packages are not in
                # the child's isolated import closure.
                continue
            relative = PurePosixPath(*relative_path.parts).as_posix()
            portable_relative_parts(relative)
            try:
                metadata = located.lstat()
            except OSError as exc:
                raise SmokeBundleError(
                    "smoke_runtime_closure",
                    "An installed runtime distribution file is missing.",
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                continue
            owned_paths.append(relative)
            root_owned_paths[site_root].add(relative)
            root_scan_roots[site_root].add(portable_relative_parts(relative)[0])
        _operation_checkpoint(operation_check)
        if not owned_paths:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "An installed runtime distribution has no importable files under site-packages.",
            )
        _reject_portable_collisions(owned_paths, code="smoke_runtime_closure")
        _operation_checkpoint(operation_check)
        distribution_specs.append((name, str(distribution.version), site_root, owned_paths))

    root_rows: list[dict[str, Any]] = []
    denied_scan_summary: list[dict[str, Any]] = []
    exact_files_by_root: dict[Path, dict[str, dict[str, Any]]] = {}
    for runtime_root in sorted(root_scan_roots, key=_runtime_root_path_sha256):
        _operation_checkpoint(operation_check)
        scan_roots = sorted(root_scan_roots[runtime_root], key=lambda value: (value.casefold(), value))
        if not scan_roots:
            continue
        exact_files, root_metadata = _scan_runtime_files_with_identity(
            runtime_root,
            scan_roots,
            operation_check=operation_check,
        )
        exact_files_by_root[runtime_root] = exact_files
        _operation_checkpoint(operation_check)
        missing = root_owned_paths[runtime_root] - set(exact_files)
        supplemental = set(exact_files) - root_owned_paths[runtime_root]
        if missing:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "An installed runtime distribution inventory changed during preparation.",
            )
        allowed_files = set(exact_files)
        if "site-packages" in root_roles[runtime_root]:
            allowed_files = set(root_owned_paths[runtime_root])
            allowed_files.update(
                relative
                for relative in supplemental
                if _supplemental_bytecode_is_owned(relative, root_owned_paths[runtime_root])
            )
        denied_count = len(set(exact_files) - allowed_files)
        if denied_count:
            denied_scan_summary.append(
                {
                    "root_path_sha256": _runtime_root_path_sha256(runtime_root),
                    "file_count": denied_count,
                }
            )
        root_rows.append(
            finalize_identity(
                {
                    "path_sha256": _runtime_root_path_sha256(runtime_root),
                    "directory_identity_sha256": stable_hash(
                        {
                            "device": int(root_metadata.st_dev),
                            "inode": int(root_metadata.st_ino),
                            "mode": int(stat.S_IFMT(root_metadata.st_mode)),
                        }
                    ),
                    "roles": sorted(root_roles[runtime_root]),
                    "scan_roots": scan_roots,
                    "allowed_files": sorted(allowed_files, key=lambda value: (value.casefold(), value)),
                    "files": [
                        {"path": relative, **exact_files[relative]}
                        for relative in sorted(exact_files, key=lambda value: (value.casefold(), value))
                    ],
                },
                "root_identity",
            )
        )
        _operation_checkpoint(operation_check)
    distribution_rows: list[dict[str, Any]] = []
    for name, version, site_root, owned_paths in distribution_specs:
        _operation_checkpoint(operation_check)
        exact_files = exact_files_by_root.get(site_root)
        if exact_files is None or any(relative not in exact_files for relative in owned_paths):
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "An installed runtime distribution inventory changed during preparation.",
            )
        body = {
            "name": name,
            "version": version,
            "root_path_sha256": _runtime_root_path_sha256(site_root),
            "files": [
                {"path": relative, **exact_files[relative]}
                for relative in sorted(owned_paths, key=lambda value: (value.casefold(), value))
            ],
        }
        distribution_rows.append(finalize_identity(body, "distribution_identity"))
        _operation_checkpoint(operation_check)
    if not root_rows:
        raise SmokeBundleError("smoke_runtime_closure", "The isolated runtime closure is empty.")
    _operation_checkpoint(operation_check)
    result = finalize_identity(
        {
            "schema_version": SMOKE_RUNTIME_CLOSURE_SCHEMA,
            "required_distributions": list(_REQUIRED_RUNTIME_DISTRIBUTIONS),
            "required_runtime_roles": list(_REQUIRED_RUNTIME_ROLES),
            "dependency_policy": "recursive-installed-requires-markers-no-extras-v1",
            "inventory_policy": "record-owned-plus-exact-third-party-and-standard-native-roots-v2",
            "execution_byte_policy": RUNTIME_EXECUTION_BYTE_POLICY,
            "bounded_residuals": list(RUNTIME_BOUNDED_RESIDUALS),
            "distributions": distribution_rows,
            "roots": root_rows,
            "denied_scan_summary": sorted(
                denied_scan_summary,
                key=lambda row: str(row["root_path_sha256"]),
            ),
            "paths_exposed": False,
        },
        "runtime_closure_identity",
    )
    _operation_checkpoint(operation_check)
    return result


@_operation_controlled
def validate_smoke_runtime_closure(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    validated = validate_plan(plan)
    expected = _validate_runtime_closure_record(validated.get("runtime_closure"))
    try:
        actual = prepare_smoke_runtime_closure(project_root, operation_check=operation_check)
    except (OSError, ValueError, SmokeBundleError) as exc:
        raise SmokeBundleError(
            "smoke_runtime_changed",
            "The exact isolated third-party runtime is unavailable; prepare a fresh bundle.",
        ) from exc
    if actual != expected:
        raise SmokeBundleError(
            "smoke_runtime_changed",
            "The exact isolated third-party runtime changed after preparation; prepare a fresh bundle.",
        )
    return actual


@_operation_controlled
def verify_prepared_runtime_closure(
    project_root: str | Path,
    value: Any,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Rehash a prepared closure without importing distribution tooling.

    This is the pre-third-party-import verifier used by contained Playground
    children.  Dependency resolution happened in the explicit parent action;
    the child only maps path tokens to its isolated site roots and proves the
    exact scanned file bytes and directory identities again.
    """

    _operation_checkpoint(operation_check)
    expected = _validate_runtime_closure_record(value)
    _operation_checkpoint(operation_check)
    project = Path(project_root).resolve()
    roots: dict[str, Path] = {}
    for item in _runtime_verification_paths(project):
        _operation_checkpoint(operation_check)
        path = Path(item).resolve()
        roots[_runtime_root_path_sha256(path)] = path
    exact_by_root: dict[str, dict[str, dict[str, Any]]] = {}
    for root_record in expected["roots"]:
        _operation_checkpoint(operation_check)
        token = str(root_record["path_sha256"])
        root = roots.get(token)
        if root is None:
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime root is unavailable.")
        metadata = root.lstat()
        _operation_checkpoint(operation_check)
        _require_directory_metadata(metadata)
        if stable_hash(
            {
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "mode": int(stat.S_IFMT(metadata.st_mode)),
            }
        ) != root_record.get("directory_identity_sha256"):
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime root identity changed.")
        actual, scanned_metadata = _scan_runtime_files_with_identity(
            root,
            list(root_record["scan_roots"]),
            operation_check=operation_check,
        )
        _operation_checkpoint(operation_check)
        if stable_hash(
            {
                "device": int(scanned_metadata.st_dev),
                "inode": int(scanned_metadata.st_ino),
                "mode": int(stat.S_IFMT(scanned_metadata.st_mode)),
            }
        ) != root_record.get("directory_identity_sha256"):
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime root changed during its scan.")
        expected_files: dict[str, dict[str, Any]] = {}
        for item in root_record["files"]:
            _operation_checkpoint(operation_check)
            expected_files[str(item["path"])] = {
                "sha256": str(item["sha256"]),
                "byte_count": int(item["byte_count"]),
            }
        if actual != expected_files:
            raise SmokeBundleError("smoke_runtime_changed", "The exact bound runtime files changed.")
        exact_by_root[token] = actual
    for distribution in expected["distributions"]:
        _operation_checkpoint(operation_check)
        files = exact_by_root.get(str(distribution["root_path_sha256"]))
        changed = files is None
        if files is not None:
            for item in distribution["files"]:
                _operation_checkpoint(operation_check)
                if files.get(str(item["path"])) != {
                    "sha256": str(item["sha256"]),
                    "byte_count": int(item["byte_count"]),
                }:
                    changed = True
                    break
        if changed:
            raise SmokeBundleError("smoke_runtime_changed", "A bound distribution file changed.")
    _operation_checkpoint(operation_check)
    return expected


class _ExactRuntimeSourceLoader(importlib_abc.Loader):
    def __init__(
        self,
        fullname: str,
        path: str,
        expected_sha256: str,
        package: bool,
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> None:
        self.fullname = fullname
        self.path = path
        self.expected_sha256 = expected_sha256
        self.package = package
        self.operation_check = operation_check

    def create_module(self, spec: Any) -> None:
        return None

    def get_filename(self, fullname: str) -> str:
        if fullname != self.fullname:
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime import changed.")
        return self.path

    def get_code(self, fullname: str) -> Any:
        _operation_checkpoint(self.operation_check)
        payload = read_stable_single_link_bytes(
            Path(self.get_filename(fullname)),
            boundary=Path(self.path).parent,
            max_bytes=256 * 1024 * 1024,
            operation_check=self.operation_check,
        )
        _operation_checkpoint(self.operation_check)
        if hashlib.sha256(payload).hexdigest() != self.expected_sha256:
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime source changed before import.")
        code = compile(payload, self.path, "exec", dont_inherit=True)
        _operation_checkpoint(self.operation_check)
        return code

    def is_package(self, fullname: str) -> bool:
        if fullname != self.fullname:
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime import changed.")
        return self.package

    def exec_module(self, module: Any) -> None:
        _operation_checkpoint(self.operation_check)
        code = self.get_code(self.fullname)
        _operation_checkpoint(self.operation_check)
        exec(code, module.__dict__)
        _operation_checkpoint(self.operation_check)


class _ExactRuntimeExtensionLoader(importlib_abc.Loader):
    def __init__(
        self,
        fullname: str,
        path: str,
        expected: tuple[str, int],
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> None:
        self.fullname = fullname
        self.path = path
        self.expected = expected
        self.operation_check = operation_check
        self._posix_fd: int | None = None
        self._windows_handle: tuple[Any, int] | None = None
        _operation_checkpoint(self.operation_check)
        load_path = path
        if sys.platform.startswith("linux"):
            try:
                self._posix_fd = os.open(
                    path,
                    os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0)),
                )
            except OSError as exc:
                raise SmokeBundleError("smoke_runtime_changed", "A bound native runtime file is unavailable.") from exc
            load_path = f"/proc/self/fd/{self._posix_fd}"
        self.load_path = load_path
        self.delegate = importlib_machinery.ExtensionFileLoader(fullname, load_path)
        self._verify()
        _operation_checkpoint(self.operation_check)

    def _verify(self) -> None:
        _operation_checkpoint(self.operation_check)
        if self._posix_fd is not None:
            digest = hashlib.sha256()
            os.lseek(self._posix_fd, 0, os.SEEK_SET)
            byte_count = 0
            while True:
                _operation_checkpoint(self.operation_check)
                chunk = os.read(self._posix_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                if byte_count > _MAX_RUNTIME_FILE_BYTES:
                    raise SmokeBundleError("smoke_runtime_changed", "A bound native runtime file is too large.")
            os.lseek(self._posix_fd, 0, os.SEEK_SET)
            if (digest.hexdigest(), byte_count) != self.expected:
                raise SmokeBundleError("smoke_runtime_changed", "A held native runtime file changed.")
        payload = read_stable_single_link_bytes(
            Path(self.path),
            boundary=Path(self.path).parent,
            max_bytes=_MAX_RUNTIME_FILE_BYTES,
            operation_check=self.operation_check,
        )
        if (hashlib.sha256(payload).hexdigest(), len(payload)) != self.expected:
            raise SmokeBundleError("smoke_runtime_changed", "A bound native runtime file changed.")
        _operation_checkpoint(self.operation_check)

    def _pin_windows(self) -> None:
        if os.name != "nt" or self._windows_handle is not None:
            return
        import _winapi

        try:
            handle = _winapi.CreateFile(self.path, 0x80000000, 0x00000001, 0, 3, 0x80, 0)
        except OSError as exc:
            raise SmokeBundleError("smoke_runtime_changed", "A bound native runtime file is unavailable.") from exc
        self._windows_handle = (_winapi, handle)

    def _close(self) -> None:
        if self._windows_handle is not None:
            winapi, handle = self._windows_handle
            self._windows_handle = None
            winapi.CloseHandle(handle)
        if self._posix_fd is not None:
            descriptor = self._posix_fd
            self._posix_fd = None
            os.close(descriptor)

    def create_module(self, spec: Any) -> Any:
        _operation_checkpoint(self.operation_check)
        self._verify()
        try:
            self._pin_windows()
            _operation_checkpoint(self.operation_check)
            module = self.delegate.create_module(spec)
            _operation_checkpoint(self.operation_check)
            self._verify()
            return module
        except BaseException:
            self._close()
            raise

    def exec_module(self, module: Any) -> None:
        _operation_checkpoint(self.operation_check)
        try:
            self._pin_windows()
            _operation_checkpoint(self.operation_check)
            self._verify()
            self.delegate.exec_module(module)
            _operation_checkpoint(self.operation_check)
            self._verify()
            _operation_checkpoint(self.operation_check)
        finally:
            self._close()

    def __del__(self) -> None:
        self._close()


class _ExactRuntimeFinder(importlib_abc.MetaPathFinder):
    def __init__(
        self,
        roots: Mapping[Path, Mapping[str, tuple[str, int]]],
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> None:
        self.operation_check = operation_check
        _operation_checkpoint(self.operation_check)
        self.roots = {os.path.normcase(str(root.resolve())): dict(files) for root, files in roots.items()}
        self.audit_active = False
        self.enabled = True

    def _owned(self, path: str | os.PathLike[str]) -> tuple[dict[str, tuple[str, int]], str] | None:
        _operation_checkpoint(self.operation_check)
        absolute = os.path.abspath(os.fsdecode(path))
        for root, expected in self.roots.items():
            _operation_checkpoint(self.operation_check)
            try:
                relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            except ValueError:
                continue
            if relative != ".." and not relative.startswith("../"):
                return expected, relative
        return None

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        _operation_checkpoint(self.operation_check)
        if not self.enabled or fullname == "spritelab" or fullname.startswith("spritelab."):
            return None
        spec = importlib_machinery.PathFinder.find_spec(fullname, path, target)
        _operation_checkpoint(self.operation_check)
        if spec is None:
            return None
        locations = spec.submodule_search_locations
        if spec.origin in {None, "built-in", "frozen"}:
            if locations is not None:
                owned_locations = [self._owned(location) for location in locations]
                if any(owned is None for owned in owned_locations):
                    raise SmokeBundleError("smoke_runtime_changed", "A runtime namespace escaped its closure.")
            return spec
        if not isinstance(spec.origin, str):
            raise SmokeBundleError("smoke_runtime_changed", "A runtime import origin is invalid.")
        owned = self._owned(spec.origin)
        if owned is None:
            raise SmokeBundleError("smoke_runtime_changed", "An import escaped the exact runtime closure.")
        expected, relative = owned
        record = expected.get(relative)
        if record is None:
            raise SmokeBundleError("smoke_runtime_changed", "An unbound runtime file was selected for import.")
        package = locations is not None
        if relative.endswith(".py"):
            loader: Any = _ExactRuntimeSourceLoader(
                fullname,
                spec.origin,
                record[0],
                package,
                operation_check=self.operation_check,
            )
        elif relative.endswith(".pyc"):
            raise SmokeBundleError("smoke_runtime_changed", "Sourceless runtime bytecode is forbidden.")
        elif any(relative.endswith(suffix) for suffix in importlib_machinery.EXTENSION_SUFFIXES):
            loader = _ExactRuntimeExtensionLoader(
                fullname,
                spec.origin,
                record,
                operation_check=self.operation_check,
            )
        else:
            raise SmokeBundleError("smoke_runtime_changed", "A non-executable runtime file was selected as code.")
        return importlib_util.spec_from_file_location(
            fullname,
            loader.load_path if isinstance(loader, _ExactRuntimeExtensionLoader) else spec.origin,
            loader=loader,
            submodule_search_locations=list(locations) if package else None,
        )

    def audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if (
            not self.enabled
            or event != "open"
            or self.audit_active
            or not arguments
            or not isinstance(arguments[0], (str, bytes, os.PathLike))
        ):
            return
        self.audit_active = True
        try:
            _operation_checkpoint(self.operation_check)
            owned = self._owned(arguments[0])
            if owned is None:
                return
            expected, relative = owned
            record = expected.get(relative)
            if record is None:
                raise SmokeBundleError("smoke_runtime_changed", "An unbound runtime resource was opened.")
            payload = read_stable_single_link_bytes(
                Path(arguments[0]),
                boundary=Path(arguments[0]).parent,
                max_bytes=_MAX_RUNTIME_FILE_BYTES,
                operation_check=self.operation_check,
            )
            if (hashlib.sha256(payload).hexdigest(), len(payload)) != record:
                raise SmokeBundleError("smoke_runtime_changed", "A bound runtime resource changed before use.")
            _operation_checkpoint(self.operation_check)
        finally:
            self.audit_active = False


@contextmanager
def bound_runtime_import_policy(
    project_root: str | Path,
    value: Any,
    *,
    operation_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Install exact runtime loaders and rehash the closure after execution."""

    _operation_checkpoint(operation_check)
    project = Path(project_root).resolve()
    expected = verify_prepared_runtime_closure(
        project,
        value,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    available: dict[str, Path] = {}
    for item in _runtime_verification_paths(project, operation_check=operation_check):
        _operation_checkpoint(operation_check)
        path = Path(item).resolve()
        available[_runtime_root_path_sha256(path)] = path
    roots: dict[Path, dict[str, tuple[str, int]]] = {}
    for root_record in expected["roots"]:
        _operation_checkpoint(operation_check)
        root = available.get(str(root_record["path_sha256"]))
        if root is None:
            raise SmokeBundleError("smoke_runtime_changed", "A bound runtime root is unavailable.")
        allowed = set(root_record["allowed_files"])
        root_files: dict[str, tuple[str, int]] = {}
        for row in root_record["files"]:
            _operation_checkpoint(operation_check)
            if row["path"] in allowed:
                root_files[str(row["path"])] = (str(row["sha256"]), int(row["byte_count"]))
        roots[root] = root_files
    _operation_checkpoint(operation_check)
    finder = _ExactRuntimeFinder(roots, operation_check=operation_check)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, finder)
    sys.addaudithook(finder.audit)
    try:
        _operation_checkpoint(operation_check)
        yield expected
        _operation_checkpoint(operation_check)
    finally:
        finder.audit_active = True
        try:
            _operation_checkpoint(operation_check)
            verify_prepared_runtime_closure(
                project,
                expected,
                operation_check=operation_check,
            )
            _operation_checkpoint(operation_check)
        finally:
            finder.audit_active = False
            finder.enabled = False
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
    _operation_checkpoint(operation_check)


def _canonical_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", str(value).strip()).casefold()
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized):
        raise SmokeBundleError("smoke_runtime_closure", "An installed distribution name is malformed.")
    return normalized


def _distribution_name(distribution: importlib_metadata.Distribution) -> str:
    value = distribution.metadata.get("Name")
    if not isinstance(value, str):
        raise SmokeBundleError("smoke_runtime_closure", "An installed distribution name is missing.")
    return _canonical_distribution_name(value)


def _required_distribution_dependencies(distribution: importlib_metadata.Distribution) -> list[str]:
    _operation_checkpoint(None)
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError as exc:
        raise SmokeBundleError(
            "smoke_runtime_closure",
            "Installed-distribution dependency markers cannot be evaluated safely.",
        ) from exc
    values: list[str] = []
    for raw in distribution.requires or ():
        _operation_checkpoint(None)
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            raise SmokeBundleError(
                "smoke_runtime_closure",
                "An installed runtime dependency declaration is malformed.",
            ) from exc
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        values.append(_canonical_distribution_name(requirement.name))
    result = sorted(set(values))
    _operation_checkpoint(None)
    return result


def _distribution_site_root(
    distribution: importlib_metadata.Distribution,
    roots: Sequence[Path],
) -> Path:
    located = Path(distribution.locate_file("")).resolve()
    for root in roots:
        if located == root:
            return root
    raise SmokeBundleError(
        "smoke_runtime_closure",
        "An installed distribution resolved outside the isolated runtime roots.",
    )


def _runtime_root_path_sha256(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8", "surrogatepass")).hexdigest()


def _hash_runtime_file(root: Path, relative: str) -> tuple[str, int]:
    parts = portable_relative_parts(relative)
    with anchored_directory(root, root) as root_anchor, ExitStack() as stack:
        anchor = root_anchor
        for part in parts[:-1]:
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        return _hash_runtime_anchored_file(anchor, parts[-1])


def _hash_runtime_anchored_file(
    anchor: AnchoredDirectory,
    name: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[str, int]:
    _operation_checkpoint(operation_check)
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        _operation_checkpoint(operation_check)
        before = os.fstat(descriptor)
        _require_file_metadata(before, max_bytes=_MAX_RUNTIME_FILE_BYTES)
        if _metadata_identity(before) != _metadata_identity(anchor.lstat(name)):
            raise SmokeBundleError("smoke_runtime_changed", "A runtime file changed while it was opened.")
        _operation_checkpoint(operation_check)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            _operation_checkpoint(operation_check)
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                _operation_checkpoint(operation_check)
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > _MAX_RUNTIME_FILE_BYTES:
                raise SmokeBundleError("smoke_runtime_changed", "A runtime file exceeds its safety bound.")
        after = os.fstat(descriptor)
        _operation_checkpoint(operation_check)
        if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(after) != _metadata_identity(
            anchor.lstat(name)
        ):
            raise SmokeBundleError("smoke_runtime_changed", "A runtime file changed while it was read.")
        _operation_checkpoint(operation_check)
        return digest.hexdigest(), byte_count
    finally:
        os.close(descriptor)


def _scan_runtime_files_with_identity(
    root: Path,
    scan_roots: Sequence[str],
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], os.stat_result]:
    _operation_checkpoint(operation_check)
    scan_root_names = list(scan_roots)
    if len(scan_root_names) != len(set(scan_root_names)):
        raise SmokeBundleError("smoke_runtime_closure", "A runtime scan root is duplicated.")
    _reject_portable_collisions(scan_root_names, code="smoke_runtime_closure")
    _operation_checkpoint(operation_check)
    inventory: dict[str, dict[str, Any]] = {}

    def walk(anchor: AnchoredDirectory, prefix: PurePosixPath) -> None:
        _operation_checkpoint(operation_check)
        names = anchor.names()
        _operation_checkpoint(operation_check)
        _reject_portable_collisions(
            [(prefix / name).as_posix() for name in names],
            code="smoke_runtime_closure",
        )
        _operation_checkpoint(operation_check)
        for name in names:
            _operation_checkpoint(operation_check)
            metadata = anchor.lstat(name)
            _operation_checkpoint(operation_check)
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            relative = prefix / name
            portable_relative_parts(relative.as_posix())
            if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
                raise SmokeBundleError("smoke_runtime_closure", "The runtime closure crosses an unsafe link.")
            if stat.S_ISDIR(metadata.st_mode):
                with anchor.open_directory_immovable(name) as child:
                    walk(child, relative)
                _operation_checkpoint(operation_check)
            elif stat.S_ISREG(metadata.st_mode):
                if operation_check is None:
                    digest, byte_count = _hash_runtime_anchored_file(anchor, name)
                else:
                    digest, byte_count = _hash_runtime_anchored_file(
                        anchor,
                        name,
                        operation_check=operation_check,
                    )
                inventory[relative.as_posix()] = {"sha256": digest, "byte_count": byte_count}
                _operation_checkpoint(operation_check)
            else:
                raise SmokeBundleError(
                    "smoke_runtime_closure",
                    "The runtime closure contains a non-regular filesystem entry.",
                )

    with anchored_directory(root, root) as anchor:
        root_metadata = anchor.directory_metadata()
        _operation_checkpoint(operation_check)
        for name in scan_roots:
            _operation_checkpoint(operation_check)
            parts = portable_relative_parts(name)
            if len(parts) != 1 or not anchor.lexists(name):
                raise SmokeBundleError("smoke_runtime_closure", "A runtime scan root is unavailable.")
            metadata = anchor.lstat(name)
            _operation_checkpoint(operation_check)
            if stat.S_ISDIR(metadata.st_mode):
                with anchor.open_directory_immovable(name) as child:
                    walk(child, PurePosixPath(name))
                _operation_checkpoint(operation_check)
            elif stat.S_ISREG(metadata.st_mode):
                if operation_check is None:
                    digest, byte_count = _hash_runtime_anchored_file(anchor, name)
                else:
                    digest, byte_count = _hash_runtime_anchored_file(
                        anchor,
                        name,
                        operation_check=operation_check,
                    )
                inventory[name] = {"sha256": digest, "byte_count": byte_count}
                _operation_checkpoint(operation_check)
            else:
                raise SmokeBundleError("smoke_runtime_closure", "A runtime scan root is unsafe.")
        if _metadata_identity(root_metadata) != _metadata_identity(anchor.directory_metadata()):
            raise SmokeBundleError("smoke_runtime_changed", "A runtime root changed while it was scanned.")
        _operation_checkpoint(operation_check)
    _reject_portable_collisions(list(inventory), code="smoke_runtime_closure")
    _operation_checkpoint(operation_check)
    if _metadata_identity(root_metadata) != _metadata_identity(root.lstat()):
        raise SmokeBundleError("smoke_runtime_changed", "A runtime root path changed during its scan.")
    _operation_checkpoint(operation_check)
    return inventory, root_metadata


def _scan_runtime_files(root: Path, scan_roots: Sequence[str]) -> dict[str, dict[str, Any]]:
    return _scan_runtime_files_with_identity(root, scan_roots)[0]


def _supplemental_bytecode_is_owned(relative: str, owned: set[str]) -> bool:
    pure = PurePosixPath(relative)
    if pure.suffix.casefold() != ".pyc" or len(pure.parts) < 2 or pure.parts[-2] != "__pycache__":
        return False
    source_name = f"{pure.name.split('.', 1)[0]}.py"
    source = PurePosixPath(*pure.parts[:-2], source_name).as_posix()
    return source in owned


def _validate_training_code_identity_record(value: Any) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    identity = dict(value)
    _require_exact_keys(
        identity,
        {"schema_version", "contract", "files", "sha256"},
        code="smoke_training_code_changed",
    )
    files = identity.get("files")
    if (
        identity.get("schema_version") != "spritelab_training_code_identity_v4"
        or identity.get("contract") != "all_tracked_production_python_v5_with_untracked_rejection"
        or not isinstance(files, list)
        or not files
        or not _is_sha256(identity.get("sha256"))
        or stable_hash({key: item for key, item in identity.items() if key != "sha256"}) != identity.get("sha256")
    ):
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    ordered_paths: list[str] = []
    for item in files:
        _operation_checkpoint(None)
        if not isinstance(item, Mapping):
            raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
        if set(item) != {"path", "binding", "semantic_role", "sha256"}:
            raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
        relative = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in ordered_paths
            or not relative.startswith("src/spritelab/")
            or not relative.endswith(".py")
            or item.get("binding") != "whole_file"
            or not isinstance(item.get("semantic_role"), str)
            or not item["semantic_role"].strip()
            or item["semantic_role"] != item["semantic_role"].strip()
            or not _is_sha256(digest)
        ):
            raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
        portable_relative_parts(relative)
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths, key=lambda path: (path.casefold(), path)):
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    _reject_portable_collisions(ordered_paths, code="smoke_training_code_changed")
    _operation_checkpoint(None)
    return identity


@_operation_controlled
def validate_bound_training_code_identity(
    project_root: str | Path,
    value: Any,
    *,
    expected_sha256: str,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Rehash every production Python file and reject inventory drift."""

    _operation_checkpoint(operation_check)
    identity = _validate_training_code_identity_record(value)
    if identity["sha256"] != expected_sha256:
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    expected: dict[str, str] = {}
    for item in identity["files"]:
        _operation_checkpoint(operation_check)
        relative = str(item["path"])
        _fixed_relative(Path(project_root).resolve(), relative)
        expected[relative] = str(item["sha256"])
    actual = _production_python_inventory(
        Path(project_root).resolve(),
        operation_check=operation_check,
    )
    if actual != expected:
        raise SmokeBundleError(
            "smoke_training_code_changed",
            "Tracked or untracked production Python changed after campaign preparation.",
        )
    _operation_checkpoint(operation_check)
    return identity


def smoke_launch_identity(plan: Mapping[str, Any], device: str) -> str:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    record = dict(validated["configurations"])[normalized]
    binding = _validate_child_environment_binding(record.get("child_environment"))
    return stable_hash(
        {
            "schema_version": "spritelab.training.smoke-launch.v1",
            "smoke_id": validated["smoke_id"],
            "plan_identity": validated["plan_identity"],
            "device": normalized,
            "training_argv": _base_smoke_training_argv(validated, normalized),
            "environment_sha256": binding["environment_sha256"],
            "interpreter_identity": dict(validated["interpreter"])["interpreter_identity"],
            "orchestration_code_identity": dict(validated["orchestration_code"])["orchestration_code_identity"],
            "runtime_closure_identity": dict(validated["runtime_closure"])["runtime_closure_identity"],
            "training_code_identity_sha256": dict(validated["bindings"])["training_code_identity_sha256"],
            "retry_policy": "NEW_BUNDLE_REQUIRED",
            "wall_clock_limit_seconds": int(record["wall_clock_limit_seconds"]),
        }
    )


def smoke_training_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    return [
        *_base_smoke_training_argv(validated, normalized),
        "--smoke-launch-identity",
        smoke_launch_identity(validated, normalized),
    ]


def smoke_worker_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    return [
        "python",
        "-I",
        "-B",
        "-S",
        "-c",
        _ISOLATED_WORKER_BOOTSTRAP,
        _CHILD_PREFLIGHT_SHA256,
        str(len(_CHILD_PREFLIGHT_BYTES)),
        "--smoke-id",
        str(validated["smoke_id"]),
        "--device",
        normalized,
        "--plan-identity",
        str(validated["plan_identity"]),
        "--launch-identity",
        smoke_launch_identity(validated, normalized),
    ]


@_operation_controlled
def file_sha256(
    path: Path,
    *,
    boundary: Path,
    max_bytes: int = 2 * 1024**3,
    operation_check: Callable[[], None] | None = None,
) -> str:
    _operation_checkpoint(operation_check)
    payload = read_stable_single_link_bytes(
        path,
        boundary=boundary,
        max_bytes=max_bytes,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    digest = hashlib.sha256(payload).hexdigest()
    _operation_checkpoint(operation_check)
    return digest


def finalize_identity(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise SmokeBundleError("identity_field", "The smoke identity payload is malformed.")
    result[field] = stable_hash(result)
    return result


def validate_identity(payload: Mapping[str, Any], field: str) -> None:
    expected = payload.get(field)
    if not _is_sha256(expected):
        raise SmokeBundleError("identity_missing", "The smoke identity is missing or malformed.")
    body = {key: value for key, value in payload.items() if key != field}
    if stable_hash(body) != expected:
        raise SmokeBundleError("identity_changed", "The smoke evidence identity no longer matches its content.")


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(plan, Mapping):
        raise SmokeBundleError("smoke_plan_schema", "The server-prepared smoke plan is unavailable.")
    value = dict(plan)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "smoke_id",
            "preparation_nonce",
            "status",
            "scope",
            "purpose",
            "interpreter",
            "orchestration_code",
            "runtime_closure",
            "bindings",
            "source",
            "derivation",
            "config_sha256_before",
            "full_campaign_output_roots",
            "configurations",
            "retry_policy",
            *FALSE_ELIGIBILITY,
            "plan_identity",
        },
        code="smoke_plan_schema",
    )
    if value.get("schema_version") != SMOKE_PLAN_SCHEMA or value.get("status") != "PREPARED":
        raise SmokeBundleError("smoke_plan_schema", "The server-prepared smoke plan is unavailable.")
    _require_smoke_id(value.get("smoke_id"))
    if (
        value.get("scope") != SMOKE_SCOPE
        or value.get("purpose") != "exploratory"
        or not isinstance(value.get("preparation_nonce"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,79}", value["preparation_nonce"])
        or value.get("retry_policy")
        != "A failed or interrupted device run is never resumed; prepare a fresh nonce/bundle."
        or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("smoke_plan_scope", "The smoke plan is not exclusively exploratory.")
    bindings = value.get("bindings")
    configurations = value.get("configurations")
    if not isinstance(bindings, Mapping) or not isinstance(configurations, Mapping):
        raise SmokeBundleError("smoke_plan_content", "The server-prepared smoke plan is incomplete.")
    bindings = dict(bindings)
    _require_exact_keys(
        bindings,
        {
            "conditioned_job_id",
            "candidate_identity_sha256",
            "publication_identity_sha256",
            "activation_manifest_sha256",
            "campaign_config_sha256",
            "campaign_identity_sha256",
            "training_code_identity_sha256",
            "training_code_identity",
            "dataset_view_manifest_sha256",
            "split_manifest_sha256",
            "conditioning_vocabulary_sha256",
            "benchmark_manifest_sha256",
        },
        code="smoke_plan_binding",
    )
    configurations = dict(configurations)
    if set(configurations) != set(SMOKE_DEVICES):
        raise SmokeBundleError("smoke_plan_config", "The smoke plan has an inexact device configuration set.")
    _validate_interpreter_record(value.get("interpreter"))
    _validate_orchestration_code_record(value.get("orchestration_code"))
    _validate_runtime_closure_record(value.get("runtime_closure"))
    for key in (
        "candidate_identity_sha256",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "training_code_identity_sha256",
        "dataset_view_manifest_sha256",
        "split_manifest_sha256",
        "conditioning_vocabulary_sha256",
        "benchmark_manifest_sha256",
    ):
        _operation_checkpoint(None)
        if not _is_sha256(bindings.get(key)):
            raise SmokeBundleError("smoke_plan_binding", "A required smoke-plan binding is malformed.")
    conditioned_job_id = bindings.get("conditioned_job_id")
    if (
        not isinstance(conditioned_job_id, str)
        or not conditioned_job_id
        or conditioned_job_id != conditioned_job_id.strip()
        or len(conditioned_job_id) > 256
    ):
        raise SmokeBundleError("smoke_plan_binding", "A required smoke-plan binding is malformed.")
    training_code = bindings.get("training_code_identity")
    validated_training_code = _validate_training_code_identity_record(training_code)
    if validated_training_code["sha256"] != bindings.get("training_code_identity_sha256"):
        raise SmokeBundleError("smoke_plan_binding", "The full training-code binding is malformed.")
    for device in SMOKE_DEVICES:
        _operation_checkpoint(None)
        record = configurations.get(device)
        if not isinstance(record, Mapping):
            raise SmokeBundleError("smoke_plan_config", "The smoke plan is missing a device configuration.")
        record = dict(record)
        _require_exact_keys(
            record,
            {
                "config_path",
                "config_sha256",
                "manifest_path",
                "manifest_sha256",
                "output_path",
                "environment",
                "child_environment",
                "wall_clock_limit_seconds",
                "writable_roots",
            },
            code="smoke_plan_config",
        )
        for key in ("config_sha256", "manifest_sha256"):
            _operation_checkpoint(None)
            if not _is_sha256(record.get(key)):
                raise SmokeBundleError("smoke_plan_config", "A smoke configuration identity is malformed.")
        _validate_environment_record(record.get("environment"), device)
        _validate_child_environment_binding(record.get("child_environment"))
        if record.get("wall_clock_limit_seconds") != SMOKE_WALL_CLOCK_LIMIT_SECONDS[device]:
            raise SmokeBundleError("smoke_plan_config", "The smoke wall-clock limit is malformed.")
        expected_writable_roots = [
            f"artifacts/training/smokes/{value['smoke_id']}/execution/{device}",
            f"runs/v3/training-smokes/{value['smoke_id']}/{device}",
        ]
        if record.get("writable_roots") != expected_writable_roots:
            raise SmokeBundleError("smoke_plan_config", "The smoke writable-root closure is malformed.")
        _reject_portable_collisions(expected_writable_roots, code="smoke_plan_config")
        expected_paths = {
            "config_path": f"artifacts/training/smokes/{value['smoke_id']}/configs/{device}.json",
            "manifest_path": f"artifacts/training/smokes/{value['smoke_id']}/configs/{device}.manifest.json",
            "output_path": f"runs/v3/training-smokes/{value['smoke_id']}/{device}",
        }
        if any(record.get(key) != expected for key, expected in expected_paths.items()):
            raise SmokeBundleError("smoke_plan_config", "A smoke output path is malformed.")
        child_binding = _validate_child_environment_binding(record["child_environment"])
        expected_temporary = f"artifacts/training/smokes/{value['smoke_id']}/execution/{device}/temp"
        if child_binding["temporary_root"] != expected_temporary:
            raise SmokeBundleError("smoke_plan_config", "The smoke child environment is malformed.")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise SmokeBundleError("smoke_plan_content", "The smoke source binding is malformed.")
    source = dict(source)
    _require_exact_keys(
        source,
        {"source_run_id", "source_run_identity_sha256", "source_resolved_config_sha256", "source_seed"},
        code="smoke_plan_content",
    )
    if (
        not isinstance(source.get("source_run_id"), str)
        or not source["source_run_id"].strip()
        or source["source_run_id"] != source["source_run_id"].strip()
        or not _is_sha256(source.get("source_run_identity_sha256"))
        or not _is_sha256(source.get("source_resolved_config_sha256"))
        or type(source.get("source_seed")) is not int
    ):
        raise SmokeBundleError("smoke_plan_content", "The smoke source binding is malformed.")
    derivation = value.get("derivation")
    if not isinstance(derivation, Mapping):
        raise SmokeBundleError("smoke_plan_content", "The smoke derivation binding is malformed.")
    derivation = dict(derivation)
    _require_exact_keys(
        derivation,
        {"base_config_sha256", "allowed_overrides", "smoke_cli_semantics"},
        code="smoke_plan_content",
    )
    overrides = derivation.get("allowed_overrides")
    semantics = derivation.get("smoke_cli_semantics")
    if (
        derivation.get("base_config_sha256") != source["source_resolved_config_sha256"]
        or not isinstance(overrides, Mapping)
        or set(overrides) != set(SMOKE_DEVICES)
        or not isinstance(semantics, Mapping)
        or dict(semantics)
        != {
            "steps": 2,
            "batch_size_max": 2,
            "sample_every": 0,
            "save_every": 1,
            "resume": False,
            "unsafe_resume": False,
        }
    ):
        raise SmokeBundleError("smoke_plan_content", "The smoke derivation binding is malformed.")
    for device in SMOKE_DEVICES:
        _operation_checkpoint(None)
        override = overrides.get(device)
        if not isinstance(override, Mapping) or set(override) != {"name", "runtime.device", "runtime.out_dir"}:
            raise SmokeBundleError("smoke_plan_content", "The smoke derivation binding is malformed.")
        if (
            not isinstance(override.get("name"), str)
            or not override["name"].strip()
            or override["name"] != override["name"].strip()
            or override.get("runtime.device") != device
            or override.get("runtime.out_dir") != configurations[device]["output_path"]
        ):
            raise SmokeBundleError("smoke_plan_content", "The smoke derivation binding is malformed.")
    if not _is_sha256(value.get("config_sha256_before")):
        raise SmokeBundleError("smoke_plan_content", "The smoke project configuration binding is malformed.")
    sentinels = value.get("full_campaign_output_roots")
    if not isinstance(sentinels, list):
        raise SmokeBundleError("smoke_plan_content", "The smoke campaign sentinels are malformed.")
    sentinel_paths: list[str] = []
    sentinel_run_ids: list[str] = []
    for sentinel in sentinels:
        _operation_checkpoint(None)
        if (
            not isinstance(sentinel, Mapping)
            or set(sentinel) != {"run_id", "relative_path", "state"}
            or sentinel.get("state") != "ABSENT"
        ):
            raise SmokeBundleError("smoke_plan_content", "A smoke campaign sentinel is malformed.")
        run_id = sentinel.get("run_id")
        relative = sentinel.get("relative_path")
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or run_id != run_id.strip()
            or not isinstance(relative, str)
        ):
            raise SmokeBundleError("smoke_plan_content", "A smoke campaign sentinel is malformed.")
        portable_relative_parts(relative)
        sentinel_run_ids.append(run_id)
        sentinel_paths.append(relative)
    if (
        not sentinels
        or sentinel_run_ids != sorted(set(sentinel_run_ids))
        or source["source_run_id"] not in sentinel_run_ids
        or len(sentinel_paths) != len(set(sentinel_paths))
    ):
        raise SmokeBundleError("smoke_plan_content", "A smoke campaign sentinel is malformed.")
    _reject_portable_collisions(sentinel_paths, code="smoke_plan_content")
    validate_identity(value, "plan_identity")
    _operation_checkpoint(None)
    return value


def load_plan(project_root: str | Path, smoke_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    bundle = artifact_bundle_directory(root, smoke_id)
    with anchored_directory(bundle, root) as anchor:
        _require_publication_complete_from_anchor(anchor, publication_name=smoke_id, exact=False)
        value = _read_json_from_anchor(anchor, "plan.json", max_bytes=16 * 1024 * 1024)
    plan = validate_plan(value)
    if plan["smoke_id"] != smoke_id:
        raise SmokeBundleError("smoke_plan_id", "The smoke plan belongs to a different bundle.")
    return plan


def publish_plan(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    configurations: Mapping[str, bytes],
    manifests: Mapping[str, bytes],
) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    smoke_id = str(validated["smoke_id"])
    parent = ensure_managed_directory(root, ("artifacts", "training", "smokes"))
    files: dict[str, bytes] = {
        "plan.json": canonical_json_bytes(validated, pretty=True),
        _BOOTSTRAP_RELATIVE_PATH: _CHILD_PREFLIGHT_BYTES,
    }
    for device in SMOKE_DEVICES:
        files[f"configs/{device}.json"] = configurations[device]
        files[f"configs/{device}.manifest.json"] = manifests[device]
    return publish_immutable_tree(parent, root=root, final_name=smoke_id, files=files)


def publish_run_container(project_root: str | Path, plan: Mapping[str, Any]) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    parent = ensure_managed_directory(root, ("runs", "v3", "training-smokes"))
    state = {
        "schema_version": SMOKE_RUN_STATE_SCHEMA,
        "smoke_id": validated["smoke_id"],
        "status": "PREPARED",
        "scope": SMOKE_SCOPE,
        "plan_identity": validated["plan_identity"],
        "resumable": False,
        "retry_policy": "NEW_BUNDLE_REQUIRED",
        **FALSE_ELIGIBILITY,
    }
    return publish_immutable_tree(
        parent,
        root=root,
        final_name=str(validated["smoke_id"]),
        files={"state.json": canonical_json_bytes(state, pretty=True)},
    )


def _device_run_state(plan: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_RUN_STATE_SCHEMA,
        "smoke_id": plan["smoke_id"],
        "device": device,
        "status": "RUNNING",
        "scope": SMOKE_SCOPE,
        "execution_mode": "windows-direct-trainer-v1" if os.name == "nt" else "linux-worker-trainer-v1",
        "plan_identity": plan["plan_identity"],
        "resumable": False,
        "retry_policy": "NEW_BUNDLE_REQUIRED",
        **FALSE_ELIGIBILITY,
    }


def begin_device_run_anchored(
    output_anchor: AnchoredDirectory,
    plan: Mapping[str, Any],
    device: str,
) -> Path:
    """Initialize an already-held pristine device directory and mark it complete."""

    validated = validate_plan(plan)
    normalized = _require_device(device)
    output_anchor.verify()
    if output_anchor.directory.name != normalized or output_anchor.names():
        raise SmokeBundleError("smoke_run_conflict", "The reserved smoke device output is not pristine.")
    state_bytes = canonical_json_bytes(_device_run_state(validated, normalized), pretty=True)
    _write_exclusive_to_anchor(output_anchor, "smoke_run_state.json", state_bytes)
    _write_publication_completion(
        output_anchor,
        publication_name=normalized,
        expected_inventory=_publication_inventory_from_files({"smoke_run_state.json": state_bytes}),
    )
    return output_anchor.directory


def begin_device_run(project_root: str | Path, plan: Mapping[str, Any], device: str) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    parent = run_bundle_directory(root, str(validated["smoke_id"]))
    _require_publication_complete(
        parent,
        boundary=root,
        publication_name=str(validated["smoke_id"]),
        exact=False,
    )
    state = _device_run_state(validated, normalized)
    output = parent / normalized
    if anchored_path_is_absent(output, root):
        return publish_immutable_tree(
            parent,
            root=root,
            final_name=normalized,
            files={"smoke_run_state.json": canonical_json_bytes(state, pretty=True)},
        )
    _require_publication_complete(
        output,
        boundary=root,
        publication_name=normalized,
        exact=False,
    )
    existing = _read_json(output / "smoke_run_state.json", boundary=output, max_bytes=1024 * 1024)
    if existing != state or set(flat_file_inventory(output, boundary=output)) != {"smoke_run_state.json"}:
        raise SmokeBundleError("smoke_run_conflict", "The reserved smoke device output is not pristine.")
    return output


def _validate_output_inventory(value: Any) -> dict[str, dict[str, Any]]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping) or not value:
        raise SmokeBundleError("smoke_receipt_invalid", "A smoke output inventory is invalid.")
    inventory = dict(value)
    if list(inventory) != sorted(inventory, key=lambda name: (name.casefold(), name)):
        raise SmokeBundleError("smoke_receipt_invalid", "A smoke output inventory is not canonical.")
    _reject_portable_collisions(list(inventory), code="smoke_receipt_invalid")
    for name, raw in inventory.items():
        _operation_checkpoint(None)
        if len(portable_relative_parts(name)) != 1 or not isinstance(raw, Mapping):
            raise SmokeBundleError("smoke_receipt_invalid", "A smoke output inventory is invalid.")
        record = dict(raw)
        if (
            set(record) != {"sha256", "byte_count"}
            or not _is_sha256(record.get("sha256"))
            or type(record.get("byte_count")) is not int
            or not 0 <= record["byte_count"] <= 2 * 1024**3
        ):
            raise SmokeBundleError("smoke_receipt_invalid", "A smoke output inventory is invalid.")
    _operation_checkpoint(None)
    return inventory


def _validate_cuda_qualification(value: Any, *, code: str) -> dict[str, Any]:
    """Validate the exact artifact emitted by ``qualify_determinism`` for a smoke."""

    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification is invalid.")
    qualification = dict(value)
    _require_exact_keys(qualification, _CUDA_QUALIFICATION_KEYS, code=code)
    if (
        qualification.get("qualified") is not True
        or qualification.get("mode") != "strict"
        or qualification.get("device") != "cuda"
        or type(qualification.get("steps")) is not int
        or qualification["steps"] != 2
        or type(qualification.get("interrupted_after")) is not int
        or qualification["interrupted_after"] != 1
        or qualification.get("repeated_forward_backward_bit_exact") is not True
        or qualification.get("resume_bit_exact") is not True
        or qualification.get("guarantee_scope") != _CUDA_GUARANTEE_SCOPE
        or qualification.get("cross_gpu_or_version_identity_claimed") is not False
    ):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification is invalid.")

    raw_environment = qualification.get("environment")
    if not isinstance(raw_environment, Mapping):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification environment is invalid.")
    environment = dict(raw_environment)
    _require_exact_keys(environment, _CUDA_ENVIRONMENT_KEYS, code=code)
    platform_name = environment.get("platform")
    torch_version = environment.get("torch_version")
    cuda_runtime_version = environment.get("cuda_runtime_version")
    cuda_driver_version = environment.get("cuda_driver_version")
    cudnn_version = environment.get("cudnn_version")
    if (
        not isinstance(platform_name, str)
        or not platform_name.strip()
        or platform_name != platform_name.strip()
        or not isinstance(torch_version, str)
        or not torch_version.strip()
        or torch_version != torch_version.strip()
        or (
            cuda_runtime_version is not None
            and (
                not isinstance(cuda_runtime_version, str)
                or not cuda_runtime_version.strip()
                or cuda_runtime_version != cuda_runtime_version.strip()
            )
        )
        or (cuda_driver_version is not None and (type(cuda_driver_version) is not int or cuda_driver_version <= 0))
        or (cudnn_version is not None and (type(cudnn_version) is not int or cudnn_version <= 0))
    ):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification environment is invalid.")

    gpus = environment.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise SmokeBundleError(code, "Strict CUDA determinism qualification GPU evidence is invalid.")
    for expected_index, raw_gpu in enumerate(gpus):
        _operation_checkpoint(None)
        if not isinstance(raw_gpu, Mapping):
            raise SmokeBundleError(code, "Strict CUDA determinism qualification GPU evidence is invalid.")
        gpu = dict(raw_gpu)
        _require_exact_keys(gpu, _CUDA_GPU_KEYS, code=code)
        name = gpu.get("name")
        capability = gpu.get("compute_capability")
        if (
            type(gpu.get("index")) is not int
            or gpu["index"] != expected_index
            or not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or not isinstance(capability, str)
            or re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None
            or type(gpu.get("total_memory_bytes")) is not int
            or gpu["total_memory_bytes"] <= 0
        ):
            raise SmokeBundleError(code, "Strict CUDA determinism qualification GPU evidence is invalid.")
    _operation_checkpoint(None)
    return qualification


def _cuda_qualification_summary(value: Any, *, code: str) -> dict[str, Any]:
    qualification = _validate_cuda_qualification(value, code=code)
    return {
        "qualified": qualification["qualified"],
        "mode": qualification["mode"],
        "device": qualification["device"],
        "steps": qualification["steps"],
        "repeated_forward_backward_bit_exact": qualification["repeated_forward_backward_bit_exact"],
        "resume_bit_exact": qualification["resume_bit_exact"],
    }


def _validate_cuda_qualification_summary(value: Any, *, code: str) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification summary is invalid.")
    summary = dict(value)
    expected_keys = {
        "qualified",
        "mode",
        "device",
        "steps",
        "repeated_forward_backward_bit_exact",
        "resume_bit_exact",
    }
    _require_exact_keys(summary, expected_keys, code=code)
    if (
        summary.get("qualified") is not True
        or summary.get("mode") != "strict"
        or summary.get("device") != "cuda"
        or type(summary.get("steps")) is not int
        or summary["steps"] != 2
        or summary.get("repeated_forward_backward_bit_exact") is not True
        or summary.get("resume_bit_exact") is not True
    ):
        raise SmokeBundleError(code, "Strict CUDA determinism qualification summary is invalid.")
    _operation_checkpoint(None)
    return summary


def _validate_verification_record(value: Any, device: str) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_receipt_invalid", "Smoke verification is missing.")
    verification = dict(value)
    _require_exact_keys(
        verification,
        {
            "status",
            "steps_completed",
            "device",
            "determinism",
            "determinism_qualified",
            "report_sha256",
            "metrics_sha256",
            "output_inventory",
            "output_inventory_sha256",
            "checkpoints",
            "qualification",
            *FALSE_ELIGIBILITY,
        },
        code="smoke_receipt_invalid",
    )
    inventory = _validate_output_inventory(verification.get("output_inventory"))
    checkpoints = verification.get("checkpoints")
    if (
        verification.get("status") != "COMPLETE"
        or verification.get("steps_completed") != 2
        or verification.get("device") != device
        or verification.get("determinism") != "strict"
        or verification.get("determinism_qualified") is not True
        or not _is_sha256(verification.get("report_sha256"))
        or not _is_sha256(verification.get("metrics_sha256"))
        or inventory.get("train_report.json", {}).get("sha256") != verification["report_sha256"]
        or inventory.get("train_metrics.jsonl", {}).get("sha256") != verification["metrics_sha256"]
        or verification.get("output_inventory_sha256") != stable_hash(inventory)
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 2
        or any(verification.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("smoke_receipt_invalid", "Smoke verification is invalid.")
    expected_checkpoints = (
        ("live", "checkpoint_step_000002.pt", "step"),
        ("ema", "checkpoint_step_000002_ema.pt", "step_ema"),
    )
    for raw, (weights, filename, variant) in zip(checkpoints, expected_checkpoints, strict=True):
        _operation_checkpoint(None)
        if not isinstance(raw, Mapping):
            raise SmokeBundleError("smoke_receipt_invalid", "Smoke checkpoint verification is invalid.")
        record = dict(raw)
        inventory_record = inventory.get(filename)
        if (
            set(record) != {"weights", "sha256", "byte_count", "step", "variant"}
            or record.get("weights") != weights
            or record.get("variant") != variant
            or record.get("step") != 2
            or not _is_sha256(record.get("sha256"))
            or type(record.get("byte_count")) is not int
            or record["byte_count"] <= 0
            or not isinstance(inventory_record, Mapping)
            or record["sha256"] != inventory_record.get("sha256")
            or record["byte_count"] != inventory_record.get("byte_count")
        ):
            raise SmokeBundleError("smoke_receipt_invalid", "Smoke checkpoint verification is invalid.")
    qualification = verification.get("qualification")
    if device == "cpu":
        if qualification is not None or _CUDA_QUALIFICATION_FILENAME in inventory:
            raise SmokeBundleError("smoke_receipt_invalid", "CPU smoke verification has a CUDA qualification.")
    else:
        qualification_record = inventory.get(_CUDA_QUALIFICATION_FILENAME)
        if not isinstance(qualification_record, Mapping) or qualification_record.get("byte_count", 0) <= 0:
            raise SmokeBundleError("smoke_receipt_invalid", "CUDA smoke qualification evidence is missing.")
        _validate_cuda_qualification_summary(qualification, code="smoke_receipt_invalid")
    _operation_checkpoint(None)
    return verification


def _validate_device_receipt_record(
    receipt: Any,
    plan: Mapping[str, Any],
    device: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    _operation_checkpoint(operation_check)
    if not isinstance(receipt, Mapping):
        raise SmokeBundleError("smoke_receipt_invalid", "A smoke-device completion receipt is invalid.")
    value = dict(receipt)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "smoke_id",
            "device",
            "status",
            "scope",
            "plan_identity",
            "launch_identity",
            "interpreter",
            "interpreter_identity",
            "orchestration_code",
            "orchestration_code_identity",
            "runtime_closure",
            "runtime_closure_identity",
            "config_sha256_before",
            "config_sha256_after",
            "configuration_unchanged",
            "environment",
            "environment_sha256",
            "environment_policy_sha256",
            "verification",
            *FALSE_ELIGIBILITY,
            "receipt_identity",
        },
        code="smoke_receipt_invalid",
    )
    validated = validate_plan(plan)
    normalized = _require_device(device)
    record = dict(validated["configurations"])[normalized]
    environment = _validate_environment_record(record.get("environment"), normalized)
    environment_binding = _validate_child_environment_binding(record.get("child_environment"))
    _validate_interpreter_record(value.get("interpreter"))
    _operation_checkpoint(operation_check)
    _validate_orchestration_code_record(value.get("orchestration_code"))
    _operation_checkpoint(operation_check)
    _validate_runtime_closure_record(value.get("runtime_closure"))
    _operation_checkpoint(operation_check)
    _validate_verification_record(value.get("verification"), normalized)
    if (
        value.get("schema_version") != SMOKE_DEVICE_RECEIPT_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("scope") != SMOKE_SCOPE
        or value.get("smoke_id") != validated["smoke_id"]
        or value.get("device") != normalized
        or value.get("plan_identity") != validated["plan_identity"]
        or value.get("launch_identity") != smoke_launch_identity(validated, normalized)
        or value.get("interpreter") != validated["interpreter"]
        or value.get("interpreter_identity") != dict(validated["interpreter"])["interpreter_identity"]
        or value.get("orchestration_code") != validated["orchestration_code"]
        or value.get("orchestration_code_identity")
        != dict(validated["orchestration_code"])["orchestration_code_identity"]
        or value.get("runtime_closure") != validated["runtime_closure"]
        or value.get("runtime_closure_identity") != dict(validated["runtime_closure"])["runtime_closure_identity"]
        or value.get("config_sha256_before") != record.get("config_sha256")
        or value.get("config_sha256_after") != record.get("config_sha256")
        or value.get("configuration_unchanged") is not True
        or value.get("environment") != environment
        or value.get("environment_sha256") != environment_binding["environment_sha256"]
        or value.get("environment_policy_sha256") != stable_hash(environment_binding)
        or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("smoke_receipt_invalid", "A smoke-device completion receipt is invalid.")
    validate_identity(value, "receipt_identity")
    _operation_checkpoint(operation_check)
    return value


@_operation_controlled
def write_device_receipt(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    config_sha256_before: str,
    config_sha256_after: str,
    environment: Mapping[str, str],
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    validate_smoke_interpreter(
        validated,
        lexical_path=environment.get("SPRITELAB_BOUND_INTERPRETER"),
    )
    validate_smoke_environment(root, validated, normalized, environment)
    record = dict(validated["configurations"])[normalized]
    public_environment = _validate_environment_record(record.get("environment"), normalized)
    environment_binding = _validate_child_environment_binding(record.get("child_environment"))
    verify_execution_guards(root, validated, operation_check=operation_check)
    verification, _checkpoints = _verify_device_output(
        root,
        validated,
        normalized,
        operation_check=operation_check,
    )
    verify_execution_guards(root, validated, operation_check=operation_check)
    _operation_checkpoint(operation_check)
    receipt = finalize_identity(
        {
            "schema_version": SMOKE_DEVICE_RECEIPT_SCHEMA,
            "smoke_id": validated["smoke_id"],
            "device": normalized,
            "status": "COMPLETE",
            "scope": SMOKE_SCOPE,
            "plan_identity": validated["plan_identity"],
            "launch_identity": smoke_launch_identity(validated, normalized),
            "interpreter": dict(validated["interpreter"]),
            "interpreter_identity": dict(validated["interpreter"])["interpreter_identity"],
            "orchestration_code": dict(validated["orchestration_code"]),
            "orchestration_code_identity": dict(validated["orchestration_code"])["orchestration_code_identity"],
            "runtime_closure": dict(validated["runtime_closure"]),
            "runtime_closure_identity": dict(validated["runtime_closure"])["runtime_closure_identity"],
            "config_sha256_before": config_sha256_before,
            "config_sha256_after": config_sha256_after,
            "configuration_unchanged": config_sha256_before == config_sha256_after,
            "environment": public_environment,
            "environment_sha256": environment_binding["environment_sha256"],
            "environment_policy_sha256": stable_hash(environment_binding),
            "verification": verification,
            **FALSE_ELIGIBILITY,
        },
        "receipt_identity",
    )
    if not receipt["configuration_unchanged"]:
        raise SmokeBundleError("smoke_config_changed", "The server-generated smoke configuration changed.")
    receipt = _validate_device_receipt_record(
        receipt,
        validated,
        normalized,
        operation_check=operation_check,
    )
    output = run_bundle_directory(root, str(validated["smoke_id"])) / normalized
    _operation_checkpoint(operation_check)
    _write_exclusive(
        output / "smoke_run_receipt.json",
        canonical_json_bytes(receipt, pretty=True),
        boundary=output,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    return receipt


@_operation_controlled
def load_device_receipt(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Load a final receipt without importing Torch or revalidating checkpoints."""

    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    output = run_bundle_directory(root, str(validated["smoke_id"])) / normalized
    with anchored_directory(output, root) as anchor:
        _require_publication_complete_from_anchor(
            anchor,
            publication_name=normalized,
            exact=False,
            operation_check=operation_check,
        )
        receipt = _read_json_from_anchor(
            anchor,
            "smoke_run_receipt.json",
            max_bytes=16 * 1024 * 1024,
            operation_check=operation_check,
        )
    result = _validate_device_receipt_record(
        receipt,
        validated,
        normalized,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    return result


@_operation_controlled
def verify_complete_bundle(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> VerifiedSmokeBundle:
    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    verify_execution_guards(root, validated, operation_check=operation_check)
    runs: dict[str, Any] = {}
    selected: list[VerifiedSmokeCheckpoint] = []
    for device in SMOKE_DEVICES:
        _operation_checkpoint(operation_check)
        output = run_bundle_directory(root, str(validated["smoke_id"])) / device
        try:
            receipt = load_device_receipt(root, validated, device, operation_check=operation_check)
        except OSError as exc:
            raise SmokeBundleError("smoke_output_incomplete", "The two-step smoke output is incomplete.") from exc
        verification, checkpoints = _verify_device_output(
            root,
            validated,
            device,
            operation_check=operation_check,
        )
        if receipt.get("verification") != verification:
            raise SmokeBundleError("smoke_receipt_stale", "Smoke output changed after its completion receipt.")
        receipt_path = output / "smoke_run_receipt.json"
        runs[device] = {
            **verification,
            "environment": dict(receipt["environment"]),
            "environment_sha256": receipt["environment_sha256"],
            "receipt_identity": receipt["receipt_identity"],
            "receipt_sha256": file_sha256(
                receipt_path,
                boundary=output,
                max_bytes=16 * 1024 * 1024,
                operation_check=operation_check,
            ),
        }
        if device == "cuda":
            selected.extend(checkpoints)
    verify_execution_guards(root, validated, operation_check=operation_check)
    _operation_checkpoint(operation_check)
    evidence = finalize_identity(
        {
            "schema_version": SMOKE_EVIDENCE_SCHEMA,
            "smoke_id": validated["smoke_id"],
            "scope": SMOKE_SCOPE,
            "status": SMOKE_STATUS,
            "plan_identity": validated["plan_identity"],
            "bindings": dict(validated["bindings"]),
            "source": dict(validated["source"]),
            "derivation": dict(validated["derivation"]),
            "orchestration_code": dict(validated["orchestration_code"]),
            "runtime_closure": dict(validated["runtime_closure"]),
            "config_sha256_before": validated["config_sha256_before"],
            "full_campaign_output_roots": list(validated["full_campaign_output_roots"]),
            "runs": runs,
            **FALSE_ELIGIBILITY,
        },
        "evidence_identity",
    )
    result = VerifiedSmokeBundle(evidence=evidence, checkpoints=tuple(selected))
    _operation_checkpoint(operation_check)
    return result


@_operation_controlled
def publish_evidence(
    project_root: str | Path,
    evidence: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> Path:
    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    value = dict(evidence)
    validate_identity(value, "evidence_identity")
    smoke_id = str(value.get("smoke_id") or "")
    _require_smoke_id(smoke_id)
    target = artifact_bundle_directory(root, smoke_id) / "smoke_evidence.json"
    _operation_checkpoint(operation_check)
    _write_exclusive(
        target,
        canonical_json_bytes(value, pretty=True),
        boundary=root,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    return target


@_operation_controlled
def publish_playground_snapshot(
    project_root: str | Path,
    bundle: VerifiedSmokeBundle,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Snapshot the verified CUDA live+EMA pair into a Playground-only root."""

    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    evidence = dict(bundle.evidence)
    validate_identity(evidence, "evidence_identity")
    content_id = f"exploratory-{evidence['evidence_identity'][:24]}"
    parent = ensure_managed_directory(
        root,
        ("runs", "v3", "playground", "exploratory-checkpoints"),
        operation_check=operation_check,
    )
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in sorted(bundle.checkpoints, key=lambda item: item.weights):
        _operation_checkpoint(operation_check)
        checkpoint_rows.append(
            {
                "weights": checkpoint.weights,
                "path": f"checkpoint_step_000002{'_ema' if checkpoint.weights == 'ema' else ''}.pt",
                "sha256": checkpoint.sha256,
                "byte_count": checkpoint.byte_count,
                "step": checkpoint.step,
                "variant": checkpoint.variant,
            }
        )
    evidence_path = artifact_bundle_directory(root, str(evidence["smoke_id"])) / "smoke_evidence.json"
    evidence_sha256 = file_sha256(
        evidence_path,
        boundary=root,
        max_bytes=64 * 1024 * 1024,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    registration = finalize_identity(
        {
            "schema_version": EXPLORATORY_REGISTRATION_SCHEMA,
            "content_id": content_id,
            "smoke_id": evidence["smoke_id"],
            "status": SMOKE_STATUS,
            "purpose": "exploratory",
            "scope": SMOKE_SCOPE,
            "evidence_identity": evidence["evidence_identity"],
            "smoke_evidence_sha256": evidence_sha256,
            "plan_identity": evidence["plan_identity"],
            "bindings": dict(evidence["bindings"]),
            "checkpoints": checkpoint_rows,
            **FALSE_ELIGIBILITY,
        },
        "registration_identity",
    )
    final = parent / content_id
    if os.path.lexists(final):
        _await_publication_complete(
            final,
            boundary=root,
            publication_name=content_id,
            exact=True,
            operation_check=operation_check,
        )
        existing = load_playground_registration(root, content_id, operation_check=operation_check)
        if existing != registration:
            raise SmokeBundleError(
                "registration_conflict", "An exploratory checkpoint registration already has different content."
            )
        _operation_checkpoint(operation_check)
        return final, existing
    registration_bytes = canonical_json_bytes(registration, pretty=True)
    file_records = {str(row["path"]): (str(row["sha256"]), int(row["byte_count"])) for row in checkpoint_rows}
    file_records["registration.json"] = (hashlib.sha256(registration_bytes).hexdigest(), len(registration_bytes))
    expected_inventory = _publication_inventory_from_records(file_records)
    with anchored_directory(parent, root) as anchor:
        try:
            publication_identity = anchor.mkdir(content_id)
        except FileExistsError:
            _await_publication_complete(
                final,
                boundary=root,
                publication_name=content_id,
                exact=True,
                operation_check=operation_check,
            )
            existing = load_playground_registration(root, content_id, operation_check=operation_check)
            if existing != registration:
                raise SmokeBundleError(
                    "registration_conflict", "Concurrent checkpoint registration conflicted."
                ) from None
            _operation_checkpoint(operation_check)
            return final, existing
        _operation_checkpoint(operation_check)
        with anchor.open_directory_immovable(content_id) as publication_anchor:
            _require_owned_publication_directory(
                anchor,
                content_id,
                publication_identity,
                code="registration_changed",
            )
            by_weight = {checkpoint.weights: checkpoint for checkpoint in bundle.checkpoints}
            for row in checkpoint_rows:
                _operation_checkpoint(operation_check)
                checkpoint = by_weight[str(row["weights"])]
                _copy_stable_single_link_file_to_anchor(
                    checkpoint.path,
                    publication_anchor,
                    str(row["path"]),
                    source_boundary=root,
                    expected_sha256=checkpoint.sha256,
                    expected_bytes=checkpoint.byte_count,
                    operation_check=operation_check,
                )
                _require_owned_publication_directory(
                    anchor,
                    content_id,
                    publication_identity,
                    code="registration_changed",
                )
            _write_exclusive_to_anchor(
                publication_anchor,
                "registration.json",
                registration_bytes,
                operation_check=operation_check,
            )
            _require_owned_publication_directory(
                anchor,
                content_id,
                publication_identity,
                code="registration_changed",
            )
            _write_publication_completion(
                publication_anchor,
                publication_name=content_id,
                expected_inventory=expected_inventory,
                operation_check=operation_check,
            )
            _require_owned_publication_directory(
                anchor,
                content_id,
                publication_identity,
                code="registration_changed",
            )
    _operation_checkpoint(operation_check)
    return final, registration


def load_playground_registration(
    project_root: str | Path,
    content_id: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    if not re.fullmatch(r"exploratory-[0-9a-f]{24}", str(content_id)):
        raise SmokeBundleError("registration_id", "The exploratory checkpoint registration ID is invalid.")
    publication = root / "runs" / "v3" / "playground" / "exploratory-checkpoints" / content_id
    with anchored_directory(publication, root) as anchor:
        _require_publication_complete_from_anchor(
            anchor,
            publication_name=content_id,
            exact=True,
            operation_check=operation_check,
        )
        value = _read_json_from_anchor(
            anchor,
            "registration.json",
            max_bytes=16 * 1024 * 1024,
            operation_check=operation_check,
        )
    if (
        value.get("schema_version") != EXPLORATORY_REGISTRATION_SCHEMA
        or value.get("content_id") != content_id
        or value.get("status") != SMOKE_STATUS
        or value.get("purpose") != "exploratory"
        or value.get("scope") != SMOKE_SCOPE
        or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("registration_invalid", "The exploratory checkpoint registration is invalid.")
    validate_identity(value, "registration_identity")
    _operation_checkpoint(operation_check)
    return value


def write_exclusive_bytes(
    path: Path,
    content: bytes,
    *,
    boundary: Path,
    operation_check: Callable[[], None] | None = None,
) -> None:
    """Publish one new direct-child file without replacement."""

    _write_exclusive(path, content, boundary=boundary, operation_check=operation_check)


def expected_config_path(project_root: str | Path, plan: Mapping[str, Any], device: str) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    relative = str(dict(validated["configurations"])[normalized].get("config_path") or "")
    return _fixed_relative(root, relative)


def expected_manifest(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    relative = str(dict(validated["configurations"])[normalized].get("manifest_path") or "")
    path = _fixed_relative(root, relative)
    value = _read_json(
        path,
        boundary=root,
        max_bytes=32 * 1024 * 1024,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    expected_hash = str(dict(validated["configurations"])[normalized]["manifest_sha256"])
    if hashlib.sha256(canonical_json_bytes(value, pretty=True)).hexdigest() != expected_hash:
        raise SmokeBundleError("smoke_manifest_changed", "The server-generated smoke manifest changed.")
    _operation_checkpoint(operation_check)
    return value


def validate_cli_configuration(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    config_path: Path,
) -> tuple[str, dict[str, Any]]:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    expected = expected_config_path(root, validated, normalized)
    try:
        actual = config_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeBundleError("smoke_config_missing", "The server-generated smoke configuration is missing.") from exc
    if actual != expected:
        raise SmokeBundleError("smoke_config_path", "Only the server-generated smoke configuration is accepted.")
    payload = read_stable_single_link_bytes(actual, boundary=root, max_bytes=16 * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    record = dict(validated["configurations"])[normalized]
    if digest != record["config_sha256"]:
        raise SmokeBundleError("smoke_config_changed", "The server-generated smoke configuration changed.")
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeBundleError("smoke_config_invalid", "The server-generated smoke configuration is invalid.") from exc
    if not isinstance(config, dict):
        raise SmokeBundleError("smoke_config_invalid", "The server-generated smoke configuration is invalid.")
    runtime = config.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or str(runtime.get("device")) != normalized
        or str(runtime.get("determinism")) != "strict"
        or str(runtime.get("out_dir")) != record.get("output_path")
    ):
        raise SmokeBundleError("smoke_config_semantics", "The smoke configuration has unsafe runtime semantics.")
    return digest, config


def validate_smoke_environment(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Verify the exact security-relevant environment bound by a smoke plan."""

    expected = build_smoke_child_environment(
        project_root,
        plan,
        device,
        inherited_environment=environment,
    )
    if dict(environment) != expected:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The smoke runtime environment differs from the server-prepared plan.",
        )
    return expected


@_operation_controlled
def verify_execution_guards(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> None:
    """Recheck the real config and absent production roots without mutation."""

    root = Path(project_root).resolve()
    _operation_checkpoint(operation_check)
    validated = validate_plan(plan)
    bindings = dict(validated["bindings"])
    validate_smoke_orchestration_code(root, validated, operation_check=operation_check)
    validate_smoke_runtime_closure(root, validated, operation_check=operation_check)
    _operation_checkpoint(operation_check)
    validate_bound_training_code_identity(
        root,
        bindings.get("training_code_identity"),
        expected_sha256=str(bindings["training_code_identity_sha256"]),
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    config_path = root / "spritelab.yaml"
    if file_sha256(
        config_path,
        boundary=root,
        max_bytes=16 * 1024 * 1024,
        operation_check=operation_check,
    ) != validated.get("config_sha256_before"):
        raise SmokeBundleError("project_config_changed", "The real project configuration changed during the smoke.")
    for sentinel in validated.get("full_campaign_output_roots") or ():
        _operation_checkpoint(operation_check)
        if not isinstance(sentinel, Mapping) or sentinel.get("state") != "ABSENT":
            raise SmokeBundleError("campaign_sentinel", "A campaign output-root sentinel is invalid.")
        relative = str(sentinel.get("relative_path") or "")
        path = _fixed_relative(root, relative)
        if not anchored_path_is_absent(path, root, operation_check=operation_check):
            raise SmokeBundleError("campaign_output_changed", "A full-campaign output root changed during the smoke.")
    _operation_checkpoint(operation_check)


def _verify_device_output(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], tuple[VerifiedSmokeCheckpoint, ...]]:
    _operation_checkpoint(operation_check)
    output = run_bundle_directory(root, str(plan["smoke_id"])) / device
    _require_publication_complete(
        output,
        boundary=root,
        publication_name=device,
        exact=False,
        operation_check=operation_check,
    )
    inventory = flat_file_inventory(
        output,
        boundary=output,
        exclude=("smoke_run_receipt.json",),
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    required = {
        "smoke_run_state.json",
        "config.json",
        "train_report.json",
        "train_metrics.jsonl",
        "checkpoint_step_000002.pt",
        "checkpoint_step_000002_ema.pt",
    }
    if device == "cuda":
        required.add(_CUDA_QUALIFICATION_FILENAME)
    if not required <= set(inventory):
        raise SmokeBundleError("smoke_output_incomplete", "The two-step smoke output is incomplete.")
    report = _read_json(
        output / "train_report.json",
        boundary=output,
        max_bytes=64 * 1024 * 1024,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    determinism = report.get("determinism")
    if (
        report.get("model_type") != "generator_challenger"
        or report.get("steps_completed") != 2
        or report.get("max_steps") != 2
        or str(report.get("device")) != device
        or not isinstance(determinism, Mapping)
        or determinism.get("mode") != "strict"
        or determinism.get("qualified") is not True
        or determinism.get("issues") not in ([], ())
    ):
        raise SmokeBundleError("smoke_report_invalid", "The smoke report is not a qualified strict two-step run.")
    _require_finite_tree(report)
    _operation_checkpoint(operation_check)
    metrics = _read_metrics(
        output / "train_metrics.jsonl",
        boundary=output,
        operation_check=operation_check,
    )
    if not metrics or metrics[-1].get("step") != 2:
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metrics do not contain a finite final step 2.")
    _operation_checkpoint(operation_check)
    manifest = expected_manifest(root, plan, device, operation_check=operation_check)
    checkpoint_values: dict[str, Mapping[str, Any]] = {}
    for name, record in inventory.items():
        _operation_checkpoint(operation_check)
        if not name.startswith("checkpoint") or not name.endswith(".pt"):
            continue
        from spritelab.training.checkpoint_io import load_checkpoint

        _operation_checkpoint(operation_check)
        checkpoint_values[name] = load_checkpoint(output / name, expected_sha256=str(record["sha256"]))
        _operation_checkpoint(operation_check)
        value = checkpoint_values[name]
        if (
            value.get("model_type") != "generator_challenger"
            or value.get("experiment_manifest") != manifest
            or value.get("step") not in (1, 2)
            or value.get("global_step") != value.get("step")
        ):
            raise SmokeBundleError("smoke_checkpoint_invalid", "A smoke checkpoint failed safe validation.")
    selected: list[VerifiedSmokeCheckpoint] = []
    for weights, name, variant, ema in (
        ("live", "checkpoint_step_000002.pt", "step", False),
        ("ema", "checkpoint_step_000002_ema.pt", "step_ema", True),
    ):
        _operation_checkpoint(operation_check)
        value = checkpoint_values.get(name)
        record = inventory[name]
        if (
            not isinstance(value, Mapping)
            or value.get("step") != 2
            or value.get("global_step") != 2
            or value.get("checkpoint_variant") != variant
            or value.get("ema_weights") is not ema
        ):
            raise SmokeBundleError("smoke_checkpoint_variant", "The step-2 live/EMA checkpoint pair is invalid.")
        selected.append(
            VerifiedSmokeCheckpoint(
                weights=weights,
                path=output / name,
                sha256=str(record["sha256"]),
                byte_count=int(record["byte_count"]),
                step=2,
                variant=variant,
            )
        )
    qualification = None
    if device == "cuda":
        raw_qualification = _read_cuda_qualification_artifact(
            output / _CUDA_QUALIFICATION_FILENAME,
            boundary=output,
            inventory_record=inventory[_CUDA_QUALIFICATION_FILENAME],
            operation_check=operation_check,
        )
        qualification = _cuda_qualification_summary(raw_qualification, code="cuda_qualification_invalid")
        _operation_checkpoint(operation_check)
    result = (
        {
            "status": "COMPLETE",
            "steps_completed": 2,
            "device": device,
            "determinism": "strict",
            "determinism_qualified": True,
            "report_sha256": inventory["train_report.json"]["sha256"],
            "metrics_sha256": inventory["train_metrics.jsonl"]["sha256"],
            "output_inventory": inventory,
            "output_inventory_sha256": stable_hash(inventory),
            "checkpoints": [
                {
                    "weights": item.weights,
                    "sha256": item.sha256,
                    "byte_count": item.byte_count,
                    "step": item.step,
                    "variant": item.variant,
                }
                for item in selected
            ],
            "qualification": qualification,
            **FALSE_ELIGIBILITY,
        },
        tuple(selected),
    )
    _operation_checkpoint(operation_check)
    return result


def ensure_managed_directory(
    start: Path,
    parts: Sequence[str],
    *,
    boundary: Path | None = None,
    operation_check: Callable[[], None] | None = None,
) -> Path:
    _operation_checkpoint(operation_check)
    approved_root = (boundary or start).resolve()
    current = start
    with ExitStack() as stack:
        anchor = stack.enter_context(anchored_directory(current, approved_root))
        for part in parts:
            _operation_checkpoint(operation_check)
            if not part or Path(part).name != part or part in {".", ".."}:
                raise SmokeBundleError("managed_path", "The fixed smoke storage path is invalid.")
            if anchor.lexists(part):
                _require_directory_metadata(anchor.lstat(part))
            else:
                try:
                    identity = anchor.mkdir(part)
                except FileExistsError:
                    _require_directory_metadata(anchor.lstat(part))
                else:
                    if not identity.matches(anchor.lstat(part)):
                        raise SmokeBundleError("managed_path_changed", "Fixed smoke storage changed during creation.")
            current = current / part
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
            _operation_checkpoint(operation_check)
    _operation_checkpoint(operation_check)
    return current


def _publication_inventory_from_files(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if not isinstance(files, Mapping) or not files:
        raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
    records: dict[str, tuple[str, int]] = {}
    for relative, content in files.items():
        if not isinstance(relative, str) or type(content) is not bytes:
            raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
        records[relative] = (hashlib.sha256(content).hexdigest(), len(content))
    return _publication_inventory_from_records(records)


def _publication_inventory_from_records(
    records: Mapping[str, tuple[str, int]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, Mapping) or not records:
        raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
    _reject_portable_collisions(list(records), code="publication_inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for relative, record in records.items():
        try:
            parts = portable_relative_parts(relative)
        except SmokeBundleError as exc:
            raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.") from exc
        if (
            len(parts) > _MAX_PUBLICATION_DEPTH
            or _PUBLICATION_COMPLETION_FILENAME in parts
            or not isinstance(record, tuple)
            or len(record) != 2
        ):
            raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
        digest, byte_count = record
        if not _is_sha256(digest) or type(byte_count) is not int or not 0 <= byte_count <= 2 * 1024**3:
            raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
        for index in range(1, len(parts)):
            directory = PurePosixPath(*parts[:index]).as_posix()
            previous = inventory.setdefault(directory, {"kind": "directory"})
            if previous != {"kind": "directory"}:
                raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
        if relative in inventory:
            raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
        inventory[relative] = {"kind": "file", "sha256": digest, "byte_count": byte_count}
    if len(inventory) > _MAX_PUBLICATION_ENTRIES:
        raise SmokeBundleError("publication_inventory", "The smoke publication inventory is too large.")
    _reject_portable_collisions(list(inventory), code="publication_inventory")
    return dict(sorted(inventory.items(), key=lambda item: (item[0].casefold(), item[0])))


def _validate_publication_inventory(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value or len(value) > _MAX_PUBLICATION_ENTRIES:
        raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
    inventory = dict(value)
    if list(inventory) != sorted(inventory, key=lambda name: (name.casefold(), name)):
        raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is not canonical.")
    _reject_portable_collisions(list(inventory), code="publication_incomplete")
    for relative, raw in inventory.items():
        try:
            parts = portable_relative_parts(relative)
        except SmokeBundleError as exc:
            raise SmokeBundleError(
                "publication_incomplete", "A smoke publication completion inventory is invalid."
            ) from exc
        if len(parts) > _MAX_PUBLICATION_DEPTH or _PUBLICATION_COMPLETION_FILENAME in parts:
            raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
        if not isinstance(raw, Mapping):
            raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
        record = dict(raw)
        kind = record.get("kind")
        if kind == "directory":
            if record != {"kind": "directory"}:
                raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
        elif kind == "file":
            if (
                set(record) != {"kind", "sha256", "byte_count"}
                or not _is_sha256(record.get("sha256"))
                or type(record.get("byte_count")) is not int
                or not 0 <= record["byte_count"] <= 2 * 1024**3
            ):
                raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
        else:
            raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
        for index in range(1, len(parts)):
            parent = PurePosixPath(*parts[:index]).as_posix()
            if inventory.get(parent) != {"kind": "directory"}:
                raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
    for relative, record in inventory.items():
        if record == {"kind": "directory"} and not any(
            candidate.startswith(f"{relative}/") for candidate in inventory if candidate != relative
        ):
            raise SmokeBundleError("publication_incomplete", "A smoke publication completion inventory is invalid.")
    return inventory


def _hash_publication_anchored_file(
    anchor: AnchoredDirectory,
    name: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[str, int]:
    _operation_checkpoint(operation_check)
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        before = os.fstat(descriptor)
        _require_file_metadata(before, max_bytes=2 * 1024**3)
        if _metadata_identity(before) != _metadata_identity(anchor.lstat(name)):
            raise SmokeBundleError("publication_changed", "A smoke publication file changed while opening.")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            _operation_checkpoint(operation_check)
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > 2 * 1024**3:
                raise SmokeBundleError("publication_changed", "A smoke publication file is too large.")
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(after) != _metadata_identity(
            anchor.lstat(name)
        ):
            raise SmokeBundleError("publication_changed", "A smoke publication file changed while reading.")
        _operation_checkpoint(operation_check)
        return digest.hexdigest(), byte_count
    finally:
        os.close(descriptor)


def _scan_publication_inventory(
    anchor: AnchoredDirectory,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}

    def walk(current: AnchoredDirectory, prefix: tuple[str, ...]) -> None:
        _operation_checkpoint(operation_check)
        if len(prefix) >= _MAX_PUBLICATION_DEPTH:
            raise SmokeBundleError("publication_changed", "A smoke publication tree is too deep.")
        for name in current.names():
            _operation_checkpoint(operation_check)
            if not prefix and name == _PUBLICATION_COMPLETION_FILENAME:
                continue
            relative = PurePosixPath(*prefix, name).as_posix()
            try:
                portable_relative_parts(relative)
            except SmokeBundleError as exc:
                raise SmokeBundleError("publication_changed", "A smoke publication path is invalid.") from exc
            metadata = current.lstat(name)
            if stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata):
                inventory[relative] = {"kind": "directory"}
                if len(inventory) > _MAX_PUBLICATION_ENTRIES:
                    raise SmokeBundleError("publication_changed", "A smoke publication tree is too large.")
                with current.open_directory_immovable(name) as child:
                    walk(child, (*prefix, name))
            else:
                digest, byte_count = _hash_publication_anchored_file(
                    current,
                    name,
                    operation_check=operation_check,
                )
                inventory[relative] = {"kind": "file", "sha256": digest, "byte_count": byte_count}
                if len(inventory) > _MAX_PUBLICATION_ENTRIES:
                    raise SmokeBundleError("publication_changed", "A smoke publication tree is too large.")

    walk(anchor, ())
    _reject_portable_collisions(list(inventory), code="publication_changed")
    return dict(sorted(inventory.items(), key=lambda item: (item[0].casefold(), item[0])))


def _verify_publication_inventory_subset(
    anchor: AnchoredDirectory,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    operation_check: Callable[[], None] | None = None,
) -> None:
    for relative, record in inventory.items():
        _operation_checkpoint(operation_check)
        parts = portable_relative_parts(relative)
        with ExitStack() as stack:
            current = anchor
            for part in parts[:-1]:
                current = stack.enter_context(current.open_directory_immovable(part))
            name = parts[-1]
            if record.get("kind") == "directory":
                with current.open_directory_immovable(name):
                    pass
            else:
                digest, byte_count = _hash_publication_anchored_file(
                    current,
                    name,
                    operation_check=operation_check,
                )
                if (digest, byte_count) != (record.get("sha256"), record.get("byte_count")):
                    raise SmokeBundleError(
                        "publication_changed", "A smoke publication no longer matches its completion inventory."
                    )


def _decode_publication_completion(payload: bytes, publication_name: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SmokeBundleError("publication_incomplete", "A smoke publication completion marker is invalid.") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value, pretty=True):
        raise SmokeBundleError("publication_incomplete", "A smoke publication completion marker is invalid.")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "status",
            "publication_name",
            "inventory",
            "inventory_sha256",
            "completion_identity",
        },
        code="publication_incomplete",
    )
    inventory = _validate_publication_inventory(value.get("inventory"))
    if (
        value.get("schema_version") != _PUBLICATION_COMPLETION_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("publication_name") != publication_name
        or value.get("inventory_sha256") != stable_hash(inventory)
    ):
        raise SmokeBundleError("publication_incomplete", "A smoke publication completion marker is invalid.")
    validate_identity(value, "completion_identity")
    return value


def _require_publication_complete_from_anchor(
    anchor: AnchoredDirectory,
    *,
    publication_name: str,
    exact: bool,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    _operation_checkpoint(operation_check)
    if not anchor.lexists(_PUBLICATION_COMPLETION_FILENAME):
        raise SmokeBundleError("publication_incomplete", "A smoke publication is missing its completion marker.")
    payload = _read_from_anchor(
        anchor,
        _PUBLICATION_COMPLETION_FILENAME,
        max_bytes=16 * 1024 * 1024,
        operation_check=operation_check,
    )
    marker = _decode_publication_completion(payload, publication_name)
    inventory = dict(marker["inventory"])
    if exact:
        actual = _scan_publication_inventory(anchor, operation_check=operation_check)
        if actual != inventory:
            raise SmokeBundleError(
                "publication_changed", "A smoke publication no longer matches its completion inventory."
            )
    else:
        _verify_publication_inventory_subset(anchor, inventory, operation_check=operation_check)
    _operation_checkpoint(operation_check)
    return marker


def _require_publication_complete(
    directory: Path,
    *,
    boundary: Path,
    publication_name: str,
    exact: bool,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    with anchored_directory(directory, boundary) as anchor:
        return _require_publication_complete_from_anchor(
            anchor,
            publication_name=publication_name,
            exact=exact,
            operation_check=operation_check,
        )


def _await_publication_complete(
    directory: Path,
    *,
    boundary: Path,
    publication_name: str,
    exact: bool,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Allow an already-active direct-final writer to expose its last marker."""

    deadline = time.monotonic() + _PUBLICATION_CONVERGENCE_SECONDS
    last_error: BaseException | None = None
    while True:
        _operation_checkpoint(operation_check)
        try:
            return _require_publication_complete(
                directory,
                boundary=boundary,
                publication_name=publication_name,
                exact=exact,
                operation_check=operation_check,
            )
        except (OSError, SmokeBundleError, UnsafeFilesystemOperation) as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            assert last_error is not None
            raise last_error
        time.sleep(0.01)


def _write_publication_completion(
    anchor: AnchoredDirectory,
    *,
    publication_name: str,
    expected_inventory: Mapping[str, Mapping[str, Any]],
    operation_check: Callable[[], None] | None = None,
) -> None:
    inventory = _validate_publication_inventory(expected_inventory)
    actual = _scan_publication_inventory(anchor, operation_check=operation_check)
    if actual != inventory:
        raise SmokeBundleError("publication_changed", "Smoke publication bytes changed before completion.")
    marker = finalize_identity(
        {
            "schema_version": _PUBLICATION_COMPLETION_SCHEMA,
            "status": "COMPLETE",
            "publication_name": publication_name,
            "inventory": inventory,
            "inventory_sha256": stable_hash(inventory),
        },
        "completion_identity",
    )
    _operation_checkpoint(operation_check)
    _write_exclusive_to_anchor(
        anchor,
        _PUBLICATION_COMPLETION_FILENAME,
        canonical_json_bytes(marker, pretty=True),
        operation_check=operation_check,
    )
    _require_publication_complete_from_anchor(
        anchor,
        publication_name=publication_name,
        exact=True,
        operation_check=operation_check,
    )


def _write_publication_relative_file(
    anchor: AnchoredDirectory,
    relative: str,
    content: bytes,
    *,
    operation_check: Callable[[], None] | None = None,
) -> None:
    try:
        parts = portable_relative_parts(relative)
    except SmokeBundleError as exc:
        raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.") from exc
    with ExitStack() as stack:
        current = anchor
        for part in parts[:-1]:
            _operation_checkpoint(operation_check)
            if current.lexists(part):
                _require_directory_metadata(current.lstat(part))
            else:
                try:
                    identity = current.mkdir(part)
                except FileExistsError:
                    _require_directory_metadata(current.lstat(part))
                else:
                    if not identity.matches(current.lstat(part)):
                        raise SmokeBundleError("publication_changed", "A smoke publication directory changed.")
            current = stack.enter_context(current.open_directory_immovable(part))
        _write_exclusive_to_anchor(
            current,
            parts[-1],
            content,
            operation_check=operation_check,
        )


def _require_owned_publication_directory(
    parent: AnchoredDirectory,
    name: str,
    identity: OwnedFileIdentity,
    *,
    code: str,
) -> None:
    parent.verify()
    try:
        metadata = parent.lstat(name)
    except FileNotFoundError as exc:
        raise SmokeBundleError(code, "A smoke publication directory disappeared.") from exc
    _require_directory_metadata(metadata)
    if not identity.matches(metadata):
        raise SmokeBundleError(code, "A smoke publication directory identity changed.")


def publish_immutable_tree(
    parent: Path,
    *,
    root: Path,
    final_name: str,
    files: Mapping[str, bytes],
    operation_check: Callable[[], None] | None = None,
) -> Path:
    _operation_checkpoint(operation_check)
    boundary = root.resolve()
    if not final_name or Path(final_name).name != final_name:
        raise SmokeBundleError("publication_name", "The fixed smoke publication name is invalid.")
    expected_inventory = _publication_inventory_from_files(files)
    with anchored_directory(parent, boundary) as anchor:
        if anchor.lexists(final_name):
            _await_publication_complete(
                parent / final_name,
                boundary=boundary,
                publication_name=final_name,
                exact=False,
                operation_check=operation_check,
            )
            raise SmokeBundleError("publication_exists", "That immutable smoke publication already exists.")
        try:
            publication_identity = anchor.mkdir(final_name)
        except FileExistsError as exc:
            _await_publication_complete(
                parent / final_name,
                boundary=boundary,
                publication_name=final_name,
                exact=False,
                operation_check=operation_check,
            )
            raise SmokeBundleError("publication_exists", "That immutable smoke publication already exists.") from exc
        _operation_checkpoint(operation_check)
        with anchor.open_directory_immovable(final_name) as publication_anchor:
            _require_owned_publication_directory(
                anchor,
                final_name,
                publication_identity,
                code="publication_changed",
            )
            for relative, content in sorted(files.items(), key=lambda item: (item[0].casefold(), item[0])):
                _operation_checkpoint(operation_check)
                _write_publication_relative_file(
                    publication_anchor,
                    relative,
                    content,
                    operation_check=operation_check,
                )
                _require_owned_publication_directory(
                    anchor,
                    final_name,
                    publication_identity,
                    code="publication_changed",
                )
            _write_publication_completion(
                publication_anchor,
                publication_name=final_name,
                expected_inventory=expected_inventory,
                operation_check=operation_check,
            )
            _require_owned_publication_directory(
                anchor,
                final_name,
                publication_identity,
                code="publication_changed",
            )
    _operation_checkpoint(operation_check)
    return parent / final_name


def read_stable_single_link_bytes(
    path: Path,
    *,
    boundary: Path,
    max_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    _operation_checkpoint(operation_check)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with anchored_directory(path.parent, boundary) as anchor:
        payload = _read_from_anchor(
            anchor,
            path.name,
            max_bytes=max_bytes,
            operation_check=operation_check,
        )
    _operation_checkpoint(operation_check)
    return payload


def flat_file_inventory(
    directory: Path,
    *,
    boundary: Path,
    exclude: Sequence[str] = (),
    operation_check: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    _operation_checkpoint(operation_check)
    excluded = {_PUBLICATION_COMPLETION_FILENAME, *exclude}
    inventory: dict[str, dict[str, Any]] = {}
    with anchored_directory(directory, boundary) as anchor:
        names = anchor.names()
        _operation_checkpoint(operation_check)
        for name in names:
            _operation_checkpoint(operation_check)
            if name in excluded:
                continue
            digest, byte_count = _hash_publication_anchored_file(
                anchor,
                name,
                operation_check=operation_check,
            )
            inventory[name] = {"sha256": digest, "byte_count": byte_count}
            _operation_checkpoint(operation_check)
    result = dict(sorted(inventory.items(), key=lambda item: (item[0].casefold(), item[0])))
    _operation_checkpoint(operation_check)
    return result


def copy_stable_single_link_file(
    source: Path,
    destination: Path,
    *,
    source_boundary: Path,
    destination_boundary: Path,
    expected_sha256: str,
    expected_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> None:
    _operation_checkpoint(operation_check)
    if not SHA256_PATTERN.fullmatch(expected_sha256) or expected_bytes < 0:
        raise SmokeBundleError("snapshot_identity", "The checkpoint snapshot identity is invalid.")
    with (
        anchored_directory(source.parent, source_boundary) as source_anchor,
        anchored_directory(destination.parent, destination_boundary) as destination_anchor,
    ):
        source_fd = source_anchor.open_file(source.name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        destination_fd = -1
        try:
            _operation_checkpoint(operation_check)
            source_before = os.fstat(source_fd)
            _require_file_metadata(source_before, max_bytes=2 * 1024**3)
            if _metadata_identity(source_before) != _metadata_identity(source_anchor.lstat(source.name)):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed before snapshotting.")
            _operation_checkpoint(operation_check)
            destination_fd = destination_anchor.open_file(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
            )
            destination_identity = OwnedFileIdentity.from_stat(os.fstat(destination_fd))
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                _operation_checkpoint(operation_check)
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    _operation_checkpoint(operation_check)
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                view = memoryview(chunk)
                while view:
                    _operation_checkpoint(operation_check)
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
            source_after = os.fstat(source_fd)
            _operation_checkpoint(operation_check)
            if (
                _metadata_identity(source_before) != _metadata_identity(source_after)
                or _metadata_identity(source_after) != _metadata_identity(source_anchor.lstat(source.name))
                or digest.hexdigest() != expected_sha256
                or byte_count != expected_bytes
            ):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed during snapshotting.")
            destination_after = os.fstat(destination_fd)
            _operation_checkpoint(operation_check)
            if (
                not destination_identity.matches(destination_after)
                or int(getattr(destination_after, "st_nlink", 1)) != 1
                or destination_after.st_size != expected_bytes
                or not destination_identity.matches(destination_anchor.lstat(destination.name))
            ):
                raise SmokeBundleError("snapshot_destination_changed", "The checkpoint snapshot is unsafe.")
            _operation_checkpoint(operation_check)
        finally:
            os.close(source_fd)
            if destination_fd >= 0:
                os.close(destination_fd)


def _copy_stable_single_link_file_to_anchor(
    source: Path,
    destination_anchor: AnchoredDirectory,
    destination_name: str,
    *,
    source_boundary: Path,
    expected_sha256: str,
    expected_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha256) or expected_bytes < 0:
        raise SmokeBundleError("snapshot_identity", "The checkpoint snapshot identity is invalid.")
    with anchored_directory(source.parent, source_boundary) as source_anchor:
        source_fd = source_anchor.open_file(source.name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        destination_fd = -1
        try:
            _operation_checkpoint(operation_check)
            destination_anchor.verify()
            if destination_anchor.lexists(destination_name):
                raise FileExistsError(destination_anchor.directory / destination_name)
            source_before = os.fstat(source_fd)
            _require_file_metadata(source_before, max_bytes=2 * 1024**3)
            if _metadata_identity(source_before) != _metadata_identity(source_anchor.lstat(source.name)):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed before snapshotting.")
            destination_fd = destination_anchor.open_file(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
            )
            destination_identity = OwnedFileIdentity.from_stat(os.fstat(destination_fd))
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                _operation_checkpoint(operation_check)
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                view = memoryview(chunk)
                while view:
                    _operation_checkpoint(operation_check)
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
            source_after = os.fstat(source_fd)
            destination_after = os.fstat(destination_fd)
            if (
                _metadata_identity(source_before) != _metadata_identity(source_after)
                or _metadata_identity(source_after) != _metadata_identity(source_anchor.lstat(source.name))
                or digest.hexdigest() != expected_sha256
                or byte_count != expected_bytes
            ):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed during snapshotting.")
            if (
                not destination_identity.matches(destination_after)
                or int(getattr(destination_after, "st_nlink", 1)) != 1
                or destination_after.st_size != expected_bytes
                or not destination_identity.matches(destination_anchor.lstat(destination_name))
            ):
                raise SmokeBundleError("snapshot_destination_changed", "The checkpoint snapshot is unsafe.")
            destination_anchor.verify()
            _operation_checkpoint(operation_check)
        finally:
            os.close(source_fd)
            if destination_fd >= 0:
                os.close(destination_fd)


def _write_exclusive(
    path: Path,
    content: bytes,
    *,
    boundary: Path,
    operation_check: Callable[[], None] | None = None,
) -> None:
    _operation_checkpoint(operation_check)
    with anchored_directory(path.parent, boundary) as anchor:
        _write_exclusive_to_anchor(anchor, path.name, content, operation_check=operation_check)
    _operation_checkpoint(operation_check)


def _write_exclusive_to_anchor(
    anchor: AnchoredDirectory,
    name: str,
    content: bytes,
    *,
    operation_check: Callable[[], None] | None = None,
) -> None:
    """Publish one exact held file beneath an already retained parent."""

    _operation_checkpoint(operation_check)
    anchor.verify()
    if anchor.lexists(name):
        raise FileExistsError(f"refusing to overwrite immutable smoke artifact: {name}")
    temporary: str | None = None
    current_name: str | None = None
    direct_final = False
    descriptor = -1
    identity: OwnedFileIdentity | None = None
    try:
        if os.name == "nt":
            temporary = f".{name}.partial-{secrets.token_hex(12)}"
            descriptor = anchor.open_file(
                temporary,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
            )
            current_name = temporary
        else:
            try:
                descriptor = anchor.open_anonymous_file()
            except (OSError, UnsafeFilesystemOperation):
                # Direct O_EXCL creation keeps macOS and filesystems without
                # O_TMPFILE operational without introducing a name-to-inode
                # publication seam.  Callers expose authority only after
                # canonical validation or a directory completion marker.
                descriptor = anchor.open_file(
                    name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                )
                current_name = name
                direct_final = True
        created = os.fstat(descriptor)
        identity = OwnedFileIdentity.from_stat(created)
        expected_links = 1 if temporary is not None or direct_final else 0
        if (
            not stat.S_ISREG(created.st_mode)
            or _metadata_is_link_or_reparse(created)
            or int(getattr(created, "st_nlink", 1)) != expected_links
            or created.st_size != 0
        ):
            raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke write started unsafely.")
        if current_name is not None and not identity.matches(anchor.lstat(current_name)):
            raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke write name changed.")
        _operation_checkpoint(operation_check)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _operation_checkpoint(operation_check)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or int(getattr(metadata, "st_nlink", 1)) != expected_links
            or metadata.st_size != len(content)
            or not identity.matches(metadata)
        ):
            raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke write changed unexpectedly.")
        if current_name is not None and not identity.matches(anchor.lstat(current_name)):
            raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke write name changed.")
        _operation_checkpoint(operation_check)
        if not direct_final:
            anchor.publish_held_file_no_replace(descriptor, temporary, name, identity=identity)
        current_name = name
        published = os.fstat(descriptor)
        final = anchor.lstat(name)
        if (
            not stat.S_ISREG(published.st_mode)
            or _metadata_is_link_or_reparse(published)
            or int(getattr(published, "st_nlink", 1)) != 1
            or published.st_size != len(content)
            or not identity.matches(published)
            or _metadata_identity(published) != _metadata_identity(final)
        ):
            raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke publication changed.")
        anchor.verify()
        _operation_checkpoint(operation_check)
    except BaseException:
        if identity is not None and current_name is not None:
            anchor.quarantine_if_owned(current_name, identity, prefix=f".{name}.residue-")
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json(
    path: Path,
    *,
    boundary: Path,
    max_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    with anchored_directory(path.parent, boundary) as anchor:
        return _read_json_from_anchor(
            anchor,
            path.name,
            max_bytes=max_bytes,
            operation_check=operation_check,
        )


def _read_json_from_anchor(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    payload = _read_from_anchor(
        anchor,
        name,
        max_bytes=max_bytes,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.") from exc
    if not isinstance(value, dict):
        raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.")
    _operation_checkpoint(operation_check)
    return value


def _read_cuda_qualification_artifact(
    path: Path,
    *,
    boundary: Path,
    inventory_record: Mapping[str, Any],
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    payload = read_stable_single_link_bytes(
        path,
        boundary=boundary,
        max_bytes=16 * 1024 * 1024,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    if (
        set(inventory_record) != {"sha256", "byte_count"}
        or inventory_record.get("sha256") != hashlib.sha256(payload).hexdigest()
        or type(inventory_record.get("byte_count")) is not int
        or inventory_record["byte_count"] != len(payload)
    ):
        raise SmokeBundleError(
            "cuda_qualification_invalid",
            "CUDA qualification bytes do not match the verified smoke output inventory.",
        )

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SmokeBundleError("cuda_qualification_invalid", "CUDA qualification JSON is invalid.") from exc
    if not isinstance(value, dict):
        raise SmokeBundleError("cuda_qualification_invalid", "CUDA qualification JSON is invalid.")
    _operation_checkpoint(operation_check)
    _require_finite_tree(value)
    if payload != canonical_json_bytes(value, pretty=True):
        raise SmokeBundleError("cuda_qualification_invalid", "CUDA qualification JSON is not canonical.")
    result = _validate_cuda_qualification(value, code="cuda_qualification_invalid")
    _operation_checkpoint(operation_check)
    return result


def _read_metrics(
    path: Path,
    *,
    boundary: Path,
    operation_check: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    payload = read_stable_single_link_bytes(
        path,
        boundary=boundary,
        max_bytes=64 * 1024 * 1024,
        operation_check=operation_check,
    )
    _operation_checkpoint(operation_check)
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
        for line in text.splitlines():
            _operation_checkpoint(operation_check)
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            _require_finite_tree(value)
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metrics are malformed or non-finite.") from exc
    steps: list[int] = []
    for row in rows:
        raw_step = row.get("step")
        if type(raw_step) is not int:
            raise SmokeBundleError("smoke_metrics_invalid", "Smoke metric steps are not strictly ordered.")
        steps.append(int(raw_step))
    if steps != sorted(set(steps)):
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metric steps are not strictly ordered.")
    _operation_checkpoint(operation_check)
    return rows


def _require_finite_tree(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SmokeBundleError("smoke_nonfinite", "Smoke evidence contains a non-finite metric.")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_tree(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_tree(child)


def portable_relative_parts(relative: str) -> tuple[str, ...]:
    """Return a canonical cross-platform relative path or fail closed.

    Trust-bound paths are interpreted with both POSIX and Windows grammar even
    when the server is running on Linux.  This prevents a plan prepared on one
    platform from becoming absolute, device-relative, or aliasing a different
    name when consumed on another platform.
    """

    if not isinstance(relative, str):
        raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.")
    pure = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        not relative
        or relative != relative.strip()
        or unicodedata.normalize("NFC", relative) != relative
        or "\\" in relative
        or "//" in relative
        or pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or relative.startswith(("//", "\\\\", "\\?\\", "\\.\\"))
        or pure.as_posix() != relative
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.")
    for part in pure.parts:
        folded = unicodedata.normalize("NFC", part).casefold()
        stem = folded.split(".", 1)[0]
        if (
            part[-1:] in {".", " "}
            or stem in _WINDOWS_RESERVED_NAMES
            or any(character in '<>:"|?*' for character in part)
            or any(ord(character) < 32 for character in part)
        ):
            raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.")
    return tuple(pure.parts)


def _portable_collision_key(relative: str) -> str:
    return "/".join(unicodedata.normalize("NFC", part).casefold() for part in portable_relative_parts(relative))


def _reject_portable_collisions(paths: Sequence[str], *, code: str = "smoke_relative_path") -> None:
    seen: dict[str, str] = {}
    for relative in paths:
        try:
            key = _portable_collision_key(relative)
        except SmokeBundleError as exc:
            raise SmokeBundleError(code, "A trust-bound path inventory is invalid.") from exc
        previous = seen.setdefault(key, relative)
        if previous != relative:
            raise SmokeBundleError(code, "A trust-bound path inventory has a case or Unicode collision.")


def _fixed_relative(root: Path, relative: str) -> Path:
    parts = portable_relative_parts(relative)
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.") from exc
    return candidate


@contextmanager
def anchored_directory(path: Path, boundary: Path) -> Any:
    """Walk to ``path`` from one held project-root anchor without pathname reopening."""

    root = boundary.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SmokeBundleError("anchored_path", "A fixed smoke path is outside this project.") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise SmokeBundleError("anchored_path", "A fixed smoke path is invalid.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts:
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        yield anchor


def anchored_path_is_absent(
    path: Path,
    boundary: Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> bool:
    _operation_checkpoint(operation_check)
    """Prove absence by walking from a held root handle; reject linked seams."""

    root = boundary.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SmokeBundleError("anchored_path", "A fixed smoke path is outside this project.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SmokeBundleError("anchored_path", "A fixed smoke path is invalid.")
    missing_parent = False
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts[:-1]:
            _operation_checkpoint(operation_check)
            if not anchor.lexists(part):
                missing_parent = True
                break
            _require_directory_metadata(anchor.lstat(part))
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        result = missing_parent or not anchor.lexists(relative.parts[-1])
    _operation_checkpoint(operation_check)
    return result


def _production_python_inventory(
    root: Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, str]:
    source = root / "src" / "spritelab"
    inventory: dict[str, str] = {}

    def walk(anchor: AnchoredDirectory, prefix: PurePosixPath) -> None:
        _operation_checkpoint(operation_check)
        names = anchor.names()
        _operation_checkpoint(operation_check)
        for name in names:
            _operation_checkpoint(operation_check)
            metadata = anchor.lstat(name)
            _operation_checkpoint(operation_check)
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
                raise SmokeBundleError(
                    "smoke_training_code_changed",
                    "Production Python inventory crosses an unsafe filesystem seam.",
                )
            relative = prefix / name
            if stat.S_ISDIR(metadata.st_mode):
                with anchor.open_directory_immovable(name) as child:
                    walk(child, relative)
                _operation_checkpoint(operation_check)
            elif name.endswith(".py"):
                payload = _read_from_anchor(
                    anchor,
                    name,
                    max_bytes=8 * 1024 * 1024,
                    operation_check=operation_check,
                )
                inventory[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
                _operation_checkpoint(operation_check)

    with anchored_directory(source, root) as anchor:
        walk(anchor, PurePosixPath("src/spritelab"))
    _operation_checkpoint(operation_check)
    result = dict(sorted(inventory.items()))
    _operation_checkpoint(operation_check)
    return result


def _read_from_anchor(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    _operation_checkpoint(operation_check)
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        _operation_checkpoint(operation_check)
        before = os.fstat(descriptor)
        _require_file_metadata(before, max_bytes=max_bytes)
        path_before = anchor.lstat(name)
        _operation_checkpoint(operation_check)
        if _metadata_identity(before) != _metadata_identity(path_before):
            raise SmokeBundleError("file_changed", "A smoke artifact changed while it was opened.")
        chunks: list[bytes] = []
        total = 0
        while True:
            _operation_checkpoint(operation_check)
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                _operation_checkpoint(operation_check)
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise SmokeBundleError("file_too_large", "A smoke artifact exceeds its size limit.")
        after = os.fstat(descriptor)
        path_after = anchor.lstat(name)
        _operation_checkpoint(operation_check)
        if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(after) != _metadata_identity(
            path_after
        ):
            raise SmokeBundleError("file_changed", "A smoke artifact changed while it was read.")
        payload = b"".join(chunks)
        _operation_checkpoint(operation_check)
        return payload
    finally:
        os.close(descriptor)


def _require_smoke_id(smoke_id: str) -> str:
    if not SMOKE_ID_PATTERN.fullmatch(str(smoke_id)):
        raise SmokeBundleError("smoke_id", "The exploratory smoke bundle ID is invalid.")
    return smoke_id


def _require_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SMOKE_DEVICES:
        raise SmokeBundleError("smoke_device", "The smoke device must be cpu or cuda.")
    return normalized


def _validate_environment_record(value: Any, device: str) -> dict[str, str]:
    expected = _public_smoke_environment(device)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise SmokeBundleError("smoke_environment", "The smoke environment binding is invalid.")
    return expected


def _public_smoke_environment(device: str) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "SPRITELAB_PROGRESS": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if device == "cuda":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return dict(sorted(environment.items()))


def _allowed_inherited_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if os.name != "nt":
        return {}
    normalized = {str(key).upper(): str(value) for key, value in environment.items()}
    return {key: normalized[key] for key in ("SYSTEMROOT", "WINDIR") if normalized.get(key)}


def _compose_child_environment(
    public: Mapping[str, str],
    inherited: Mapping[str, str],
    temporary: Path,
    import_paths: Sequence[str],
    runtime_paths: Sequence[str],
    interpreter: Path,
) -> dict[str, str]:
    result = {str(key): str(value) for key, value in inherited.items()}
    result.update({str(key): str(value) for key, value in public.items()})
    temporary_value = str(temporary)
    result.update(dict.fromkeys(_SANDBOXED_ENVIRONMENT_PATHS, temporary_value))
    result["PYTHONPYCACHEPREFIX"] = str(temporary / "pycache")
    result["SPRITELAB_ISOLATED_PATHS"] = os.pathsep.join(import_paths)
    result["SPRITELAB_RUNTIME_ROOTS"] = os.pathsep.join(runtime_paths)
    result["SPRITELAB_BOUND_INTERPRETER"] = str(interpreter)
    return dict(sorted(result.items()))


def _direct_runtime_scan_roots(directory: Path, *, omit_site_packages: bool) -> tuple[str, ...]:
    _operation_checkpoint(None)
    try:
        with anchored_directory(directory, directory) as anchor:
            listed_names = anchor.names()
            _operation_checkpoint(None)
            root_metadata = anchor.directory_metadata()
            rows: list[tuple[str, os.stat_result]] = []
            for name in listed_names:
                _operation_checkpoint(None)
                rows.append((name, anchor.lstat(name)))
            if _metadata_identity(root_metadata) != _metadata_identity(anchor.directory_metadata()):
                raise SmokeBundleError("smoke_runtime_closure", "A standard-runtime directory changed.")
    except OSError as exc:
        raise SmokeBundleError("smoke_runtime_closure", "A standard-runtime directory is unavailable.") from exc
    names: list[str] = []
    for name, metadata in rows:
        _operation_checkpoint(None)
        if omit_site_packages and name.casefold() in {"site-packages", "dist-packages"}:
            continue
        portable_relative_parts(name)
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
            raise SmokeBundleError("smoke_runtime_closure", "The standard runtime crosses an unsafe link.")
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise SmokeBundleError("smoke_runtime_closure", "The standard runtime contains an unsafe entry.")
        names.append(name)
    if not names:
        raise SmokeBundleError("smoke_runtime_closure", "A standard-runtime inventory is empty.")
    _reject_portable_collisions(names, code="smoke_runtime_closure")
    _operation_checkpoint(None)
    return tuple(names)


def _standard_runtime_root_specs() -> list[tuple[Path, tuple[str, ...], tuple[str, ...]]]:
    """Return exact standard/native runtime roots without exposing them in plans."""

    _operation_checkpoint(None)
    aggregate: dict[Path, dict[str, set[str]]] = {}

    def add_tree(role: str, raw: Any, *, omit_site_packages: bool = False) -> None:
        _operation_checkpoint(None)
        if not raw:
            return
        try:
            directory = Path(str(raw)).resolve(strict=True)
        except OSError:
            return
        if not directory.is_dir():
            return
        row = aggregate.setdefault(directory, {"roles": set(), "scan_roots": set()})
        row["roles"].add(role)
        row["scan_roots"].update(_direct_runtime_scan_roots(directory, omit_site_packages=omit_site_packages))
        _operation_checkpoint(None)

    def add_file(role: str, raw: Any) -> None:
        _operation_checkpoint(None)
        if not raw:
            return
        try:
            path = Path(str(raw)).resolve(strict=True)
        except OSError:
            return
        if not path.is_file():
            return
        portable_relative_parts(path.name)
        row = aggregate.setdefault(path.parent, {"roles": set(), "scan_roots": set()})
        row["roles"].add(role)
        row["scan_roots"].add(path.name)
        _operation_checkpoint(None)

    _operation_checkpoint(None)
    paths = sysconfig.get_paths()
    add_tree("stdlib", paths.get("stdlib"), omit_site_packages=True)
    add_tree("platstdlib", paths.get("platstdlib"), omit_site_packages=True)
    destination_shared = sysconfig.get_config_var("DESTSHARED")
    if not destination_shared and os.name == "nt":
        destination_shared = Path(sys.base_prefix) / "DLLs"
    add_tree("destshared", destination_shared)

    library_directory = sysconfig.get_config_var("LIBDIR")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if library_directory and library_name:
        add_file("runtime-libraries", Path(str(library_directory)) / str(library_name))
    add_file("runtime-libraries", Path(sys.executable).resolve())

    for directory in {
        Path(sys.executable).resolve().parent,
        Path(sys.base_prefix).resolve(),
        Path(str(destination_shared or Path(sys.executable).parent)).resolve(),
    }:
        _operation_checkpoint(None)
        try:
            candidates = tuple(directory.iterdir())
        except OSError:
            continue
        _operation_checkpoint(None)
        for candidate in candidates:
            _operation_checkpoint(None)
            folded = candidate.name.casefold()
            if folded.startswith(("python", "libpython", "libcrypto", "libssl", "vcruntime")) and any(
                token in folded for token in (".dll", ".so", ".dylib")
            ):
                add_file("runtime-libraries", candidate)

    if sys.platform.startswith("linux"):
        try:
            import ssl  # noqa: F401 - load OpenSSL so its mapped images can be bound.

            mappings = Path("/proc/self/maps").read_text(encoding="utf-8", errors="strict").splitlines()
        except (OSError, UnicodeError, ImportError):
            mappings = []
        for line in mappings:
            _operation_checkpoint(None)
            candidate_text = line.rsplit(maxsplit=1)[-1] if "/" in line else ""
            folded = Path(candidate_text).name.casefold()
            if folded.startswith(("libpython", "libcrypto", "libssl")):
                add_file("runtime-libraries", candidate_text)

    roles = {role for row in aggregate.values() for role in row["roles"]}
    if not set(_REQUIRED_RUNTIME_ROLES) <= roles:
        raise SmokeBundleError("smoke_runtime_closure", "The complete standard/native runtime is unavailable.")
    result: list[tuple[Path, tuple[str, ...], tuple[str, ...]]] = []
    for directory, row in sorted(
        aggregate.items(),
        key=lambda item: _runtime_root_path_sha256(item[0]),
    ):
        _operation_checkpoint(None)
        result.append(
            (
                directory,
                tuple(sorted(row["roles"])),
                tuple(sorted(row["scan_roots"], key=lambda value: (value.casefold(), value))),
            )
        )
    _operation_checkpoint(None)
    return result


def _isolated_import_paths(root: Path) -> list[str]:
    _operation_checkpoint(None)
    candidates = [root / "src"]
    for value in sys.path:
        _operation_checkpoint(None)
        if not value:
            continue
        try:
            candidate = Path(value).resolve()
        except OSError:
            continue
        if candidate.name.casefold() in {"site-packages", "dist-packages"}:
            candidates.append(candidate)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        _operation_checkpoint(None)
        value = str(candidate)
        key = os.path.normcase(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    _operation_checkpoint(None)
    return result


@_operation_controlled
def _runtime_verification_paths(
    root: Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> list[str]:
    _operation_checkpoint(operation_check)
    candidates: list[Path] = []
    for value in _isolated_import_paths(root)[1:]:
        _operation_checkpoint(None)
        candidates.append(Path(value).resolve())
    for record in _standard_runtime_root_specs():
        _operation_checkpoint(None)
        candidates.append(record[0])
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        _operation_checkpoint(None)
        value = str(candidate)
        key = os.path.normcase(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    _operation_checkpoint(operation_check)
    return result


@_operation_controlled
def smoke_runtime_environment_paths(
    project_root: str | Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return parent-only exact import and verification paths for a contained child."""

    _operation_checkpoint(operation_check)
    root = Path(project_root).resolve()
    import_paths = tuple(_isolated_import_paths(root))
    _operation_checkpoint(operation_check)
    runtime_paths = tuple(_runtime_verification_paths(root))
    _operation_checkpoint(operation_check)
    return import_paths, runtime_paths


def _validate_child_environment_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_environment", "The smoke child-environment binding is invalid.")
    result = dict(value)
    _require_exact_keys(
        result,
        {
            "schema_version",
            "inherited_names",
            "temporary_root",
            "sandboxed_path_variables",
            "isolated_import_path_count",
            "isolated_import_paths_sha256",
            "runtime_root_count",
            "runtime_roots_sha256",
            "environment_sha256",
        },
        code="smoke_environment",
    )
    inherited_names = result.get("inherited_names")
    sandboxed = result.get("sandboxed_path_variables")
    if (
        result.get("schema_version") != SMOKE_CHILD_ENVIRONMENT_SCHEMA
        or not isinstance(inherited_names, list)
        or any(not isinstance(name, str) for name in inherited_names)
        or inherited_names != sorted(set(inherited_names))
        or any(name not in {"SYSTEMROOT", "WINDIR"} for name in inherited_names)
        or not isinstance(result.get("temporary_root"), str)
        or sandboxed != list(_SANDBOXED_ENVIRONMENT_PATHS)
        or type(result.get("isolated_import_path_count")) is not int
        or int(result["isolated_import_path_count"]) < 1
        or not _is_sha256(result.get("isolated_import_paths_sha256"))
        or type(result.get("runtime_root_count")) is not int
        or int(result["runtime_root_count"]) < 1
        or not _is_sha256(result.get("runtime_roots_sha256"))
        or not _is_sha256(result.get("environment_sha256"))
    ):
        raise SmokeBundleError("smoke_environment", "The smoke child-environment binding is invalid.")
    portable_relative_parts(result["temporary_root"])
    return result


def _base_smoke_training_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    record = dict(plan["configurations"])[device]
    return [
        "python",
        "-I",
        "-B",
        "-S",
        "-c",
        _ISOLATED_MAIN_BOOTSTRAP,
        _CHILD_PREFLIGHT_SHA256,
        str(len(_CHILD_PREFLIGHT_BYTES)),
        "train",
        "experiment",
        "run",
        "--config",
        str(record["config_path"]),
        "--smoke",
        "--smoke-bundle-id",
        str(plan["smoke_id"]),
        "--smoke-device",
        device,
        "--smoke-plan-identity",
        str(plan["plan_identity"]),
    ]


def _validate_interpreter_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_interpreter", "The smoke interpreter binding is invalid.")
    result = dict(value)
    _require_exact_keys(
        result,
        {
            "schema_version",
            "implementation",
            "implementation_version",
            "cache_tag",
            "python_implementation",
            "executable_sha256",
            "byte_count",
            "lexical_path_sha256",
            "lexical_metadata_sha256",
            "lexical_kind",
            "resolved_path_sha256",
            "resolved_metadata_sha256",
            "isolated_startup",
            "isolated_flags",
            "interpreter_identity",
        },
        code="smoke_interpreter",
    )
    version = result.get("implementation_version")
    if (
        result.get("schema_version") != SMOKE_INTERPRETER_SCHEMA
        or not isinstance(result.get("implementation"), str)
        or not result["implementation"]
        or not isinstance(version, list)
        or len(version) != 3
        or any(type(part) is not int or part < 0 for part in version)
        or not isinstance(result.get("cache_tag"), (str, type(None)))
        or result.get("cache_tag") == ""
        or not isinstance(result.get("python_implementation"), str)
        or not result["python_implementation"]
        or not _is_sha256(result.get("executable_sha256"))
        or type(result.get("byte_count")) is not int
        or int(result["byte_count"]) <= 0
        or int(result["byte_count"]) > 2 * 1024**3
        or not _is_sha256(result.get("lexical_path_sha256"))
        or not _is_sha256(result.get("lexical_metadata_sha256"))
        or result.get("lexical_kind") not in {"regular", "symlink", "reparse"}
        or not _is_sha256(result.get("resolved_path_sha256"))
        or not _is_sha256(result.get("resolved_metadata_sha256"))
        or result.get("isolated_startup") is not True
        or result.get("isolated_flags") != ["-I", "-B", "-S"]
    ):
        raise SmokeBundleError("smoke_interpreter", "The smoke interpreter binding is invalid.")
    validate_identity(result, "interpreter_identity")
    return result


def _validate_orchestration_code_record(value: Any) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    result = dict(value)
    _require_exact_keys(
        result,
        {
            "schema_version",
            "paths",
            "inventory",
            "preflight_sha256",
            "bootstrap_payload",
            "bootstrap_sha256",
            "orchestration_code_identity",
        },
        code="smoke_orchestration_code",
    )
    inventory = result.get("inventory")
    bootstrap_payload = result.get("bootstrap_payload")
    if (
        result.get("schema_version") != SMOKE_ORCHESTRATION_CODE_SCHEMA
        or result.get("paths") != list(_ORCHESTRATION_CODE_PATHS)
        or not isinstance(inventory, Mapping)
        or set(inventory) != set(_ORCHESTRATION_CODE_PATHS)
        or not _is_sha256(result.get("preflight_sha256"))
        or not _is_sha256(result.get("bootstrap_sha256"))
        or not isinstance(bootstrap_payload, Mapping)
    ):
        raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    bootstrap_payload = dict(bootstrap_payload)
    _require_exact_keys(
        bootstrap_payload,
        {"relative_path", "sha256", "byte_count", "max_byte_count"},
        code="smoke_orchestration_code",
    )
    if (
        bootstrap_payload.get("relative_path") != _BOOTSTRAP_RELATIVE_PATH
        or bootstrap_payload.get("sha256") != result["preflight_sha256"]
        or not _is_sha256(bootstrap_payload.get("sha256"))
        or type(bootstrap_payload.get("byte_count")) is not int
        or not 0 < bootstrap_payload["byte_count"] <= _MAX_BOOTSTRAP_BYTES
        or bootstrap_payload.get("max_byte_count") != _MAX_BOOTSTRAP_BYTES
    ):
        raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    for relative in _ORCHESTRATION_CODE_PATHS:
        _operation_checkpoint(None)
        record = inventory.get(relative)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "byte_count"}
            or not _is_sha256(record.get("sha256"))
            or type(record.get("byte_count")) is not int
            or int(record["byte_count"]) <= 0
            or int(record["byte_count"]) > 8 * 1024 * 1024
        ):
            raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    validate_identity(result, "orchestration_code_identity")
    _operation_checkpoint(None)
    return result


def _validate_runtime_closure_record(value: Any) -> dict[str, Any]:
    _operation_checkpoint(None)
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    result = dict(value)
    _require_exact_keys(
        result,
        {
            "schema_version",
            "required_distributions",
            "required_runtime_roles",
            "dependency_policy",
            "inventory_policy",
            "execution_byte_policy",
            "bounded_residuals",
            "distributions",
            "roots",
            "denied_scan_summary",
            "paths_exposed",
            "runtime_closure_identity",
        },
        code="smoke_runtime_closure",
    )
    distributions = result.get("distributions")
    roots = result.get("roots")
    denied_scan_summary = result.get("denied_scan_summary")
    if (
        result.get("schema_version") != SMOKE_RUNTIME_CLOSURE_SCHEMA
        or result.get("required_distributions") != list(_REQUIRED_RUNTIME_DISTRIBUTIONS)
        or result.get("required_runtime_roles") != list(_REQUIRED_RUNTIME_ROLES)
        or result.get("dependency_policy") != "recursive-installed-requires-markers-no-extras-v1"
        or result.get("inventory_policy") != "record-owned-plus-exact-third-party-and-standard-native-roots-v2"
        or result.get("execution_byte_policy") != RUNTIME_EXECUTION_BYTE_POLICY
        or result.get("bounded_residuals") != list(RUNTIME_BOUNDED_RESIDUALS)
        or result.get("paths_exposed") is not False
        or not isinstance(distributions, list)
        or not distributions
        or not isinstance(roots, list)
        or not roots
        or not isinstance(denied_scan_summary, list)
    ):
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    names: set[str] = set()
    root_tokens: set[str] = set()
    runtime_roles: set[str] = set()
    root_files_by_token: dict[str, dict[str, tuple[str, int]]] = {}
    denied_expected: dict[str, int] = {}
    ordered_root_tokens: list[str] = []
    for raw_root in roots:
        _operation_checkpoint(None)
        if not isinstance(raw_root, Mapping):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        root = dict(raw_root)
        _require_exact_keys(
            root,
            {
                "path_sha256",
                "directory_identity_sha256",
                "roles",
                "scan_roots",
                "allowed_files",
                "files",
                "root_identity",
            },
            code="smoke_runtime_closure",
        )
        files = root.get("files")
        scan_roots = root.get("scan_roots")
        allowed_files = root.get("allowed_files")
        roles = root.get("roles")
        if (
            not _is_sha256(root.get("path_sha256"))
            or root["path_sha256"] in root_tokens
            or not _is_sha256(root.get("directory_identity_sha256"))
            or not isinstance(scan_roots, list)
            or not scan_roots
            or any(not isinstance(path, str) for path in scan_roots)
            or scan_roots != sorted(set(scan_roots), key=lambda path: (path.casefold(), path))
            or not isinstance(files, list)
            or not files
            or not isinstance(allowed_files, list)
            or not allowed_files
            or any(not isinstance(path, str) for path in allowed_files)
            or allowed_files != sorted(set(allowed_files), key=lambda path: (path.casefold(), path))
            or not isinstance(roles, list)
            or roles != sorted(set(roles))
            or any(role not in {*_REQUIRED_RUNTIME_ROLES, "site-packages"} for role in roles)
        ):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        root_tokens.add(str(root["path_sha256"]))
        ordered_root_tokens.append(str(root["path_sha256"]))
        runtime_roles.update(str(role) for role in roles)
        _reject_portable_collisions(scan_roots, code="smoke_runtime_closure")
        file_records = _validate_runtime_file_rows(files)
        file_paths = set(file_records)
        if any(not isinstance(path, str) or path not in file_paths for path in allowed_files):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        _reject_portable_collisions(allowed_files, code="smoke_runtime_closure")
        token = str(root["path_sha256"])
        root_files_by_token[token] = file_records
        denied_count = len(file_records) - len(allowed_files)
        if denied_count:
            denied_expected[token] = denied_count
        validate_identity(root, "root_identity")
    if ordered_root_tokens != sorted(ordered_root_tokens):
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    summary_tokens: list[str] = []
    for row in denied_scan_summary:
        _operation_checkpoint(None)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"root_path_sha256", "file_count"}
            or not SHA256_PATTERN.fullmatch(str(row.get("root_path_sha256") or ""))
            or row.get("root_path_sha256") not in root_tokens
            or type(row.get("file_count")) is not int
            or int(row["file_count"]) <= 0
        ):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        summary_tokens.append(str(row["root_path_sha256"]))
    if summary_tokens != sorted(set(summary_tokens)):
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    if {str(row["root_path_sha256"]): int(row["file_count"]) for row in denied_scan_summary} != denied_expected:
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    ordered_names: list[str] = []
    for raw_distribution in distributions:
        _operation_checkpoint(None)
        if not isinstance(raw_distribution, Mapping):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        distribution = dict(raw_distribution)
        _require_exact_keys(
            distribution,
            {"name", "version", "root_path_sha256", "files", "distribution_identity"},
            code="smoke_runtime_closure",
        )
        name = _canonical_distribution_name(str(distribution.get("name") or ""))
        if (
            name != distribution.get("name")
            or name in names
            or not isinstance(distribution.get("version"), str)
            or not distribution["version"]
            or distribution.get("root_path_sha256") not in root_tokens
            or not isinstance(distribution.get("files"), list)
            or not distribution["files"]
        ):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        names.add(name)
        ordered_names.append(name)
        distribution_files = _validate_runtime_file_rows(distribution["files"])
        root_files = root_files_by_token[str(distribution["root_path_sha256"])]
        if any(root_files.get(path) != record for path, record in distribution_files.items()):
            raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
        validate_identity(distribution, "distribution_identity")
    if ordered_names != sorted(ordered_names):
        raise SmokeBundleError("smoke_runtime_closure", "The smoke runtime closure is invalid.")
    if not set(_REQUIRED_RUNTIME_DISTRIBUTIONS) <= names:
        raise SmokeBundleError("smoke_runtime_closure", "A required runtime distribution is missing.")
    if not set(_REQUIRED_RUNTIME_ROLES) <= runtime_roles:
        raise SmokeBundleError("smoke_runtime_closure", "A required standard/native runtime role is missing.")
    validate_identity(result, "runtime_closure_identity")
    _operation_checkpoint(None)
    return result


def _validate_runtime_file_rows(rows: Sequence[Any]) -> dict[str, tuple[str, int]]:
    _operation_checkpoint(None)
    paths: list[str] = []
    records: dict[str, tuple[str, int]] = {}
    for raw in rows:
        _operation_checkpoint(None)
        if not isinstance(raw, Mapping):
            raise SmokeBundleError("smoke_runtime_closure", "A runtime file inventory is invalid.")
        if set(raw) != {"path", "sha256", "byte_count"}:
            raise SmokeBundleError("smoke_runtime_closure", "A runtime file inventory is invalid.")
        relative = raw.get("path")
        if (
            not isinstance(relative, str)
            or relative in records
            or not _is_sha256(raw.get("sha256"))
            or type(raw.get("byte_count")) is not int
            or int(raw["byte_count"]) < 0
            or int(raw["byte_count"]) > _MAX_RUNTIME_FILE_BYTES
        ):
            raise SmokeBundleError("smoke_runtime_closure", "A runtime file inventory is invalid.")
        portable_relative_parts(relative)
        paths.append(relative)
        records[relative] = (str(raw["sha256"]), int(raw["byte_count"]))
    if paths != sorted(paths, key=lambda value: (value.casefold(), value)):
        raise SmokeBundleError("smoke_runtime_closure", "A runtime file inventory is not canonical.")
    _reject_portable_collisions(paths, code="smoke_runtime_closure")
    _operation_checkpoint(None)
    return records


def _lexical_interpreter_path() -> Path:
    return Path(os.path.abspath(sys.executable))


def _lexical_interpreter_metadata(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        target = os.readlink(path)
        link_target_sha256: str | None = hashlib.sha256(target.encode("utf-8", "surrogatepass")).hexdigest()
    elif attributes & reparse:
        kind = "reparse"
        link_target_sha256 = hashlib.sha256(str(path.resolve(strict=True)).encode("utf-8")).hexdigest()
    elif stat.S_ISREG(metadata.st_mode):
        kind = "regular"
        link_target_sha256 = None
    else:
        raise SmokeBundleError("smoke_interpreter", "The isolated Python launcher is unsafe.")
    return {
        "kind": kind,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(stat.S_IFMT(metadata.st_mode)),
        "size": int(metadata.st_size),
        "mtime_ns": int(getattr(metadata, "st_mtime_ns", 0)),
        "reparse_attributes": attributes & reparse,
        "link_target_sha256": link_target_sha256,
    }


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        int(getattr(metadata, "st_nlink", 1)),
        getattr(metadata, "st_mtime_ns", None),
    )


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _require_file_metadata(metadata: os.stat_result, *, max_bytes: int) -> None:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or attributes & reparse
        or int(getattr(metadata, "st_nlink", 1)) != 1
        or metadata.st_size < 0
        or metadata.st_size > max_bytes
    ):
        raise SmokeBundleError("smoke_file_unsafe", "A smoke artifact is not a bounded single-link file.")


def _require_directory_metadata(metadata: os.stat_result) -> None:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
        raise SmokeBundleError("smoke_directory_unsafe", "A fixed smoke directory is unsafe.")


__all__ = [
    "EXPLORATORY_REGISTRATION_SCHEMA",
    "FALSE_ELIGIBILITY",
    "SMOKE_CHILD_ENVIRONMENT_SCHEMA",
    "SMOKE_DEVICE_RECEIPT_SCHEMA",
    "SMOKE_EVIDENCE_SCHEMA",
    "SMOKE_INTERPRETER_SCHEMA",
    "SMOKE_ORCHESTRATION_CODE_SCHEMA",
    "SMOKE_PLAN_SCHEMA",
    "SMOKE_RUNTIME_CLOSURE_SCHEMA",
    "SMOKE_SCOPE",
    "SMOKE_STATUS",
    "SMOKE_WALL_CLOCK_LIMIT_SECONDS",
    "PinnedSmokeInterpreter",
    "SmokeBundleError",
    "VerifiedSmokeBundle",
    "VerifiedSmokeCheckpoint",
    "anchored_directory",
    "anchored_path_is_absent",
    "artifact_bundle_directory",
    "begin_device_run",
    "begin_device_run_anchored",
    "bound_runtime_import_policy",
    "build_smoke_child_environment",
    "canonical_json_bytes",
    "copy_stable_single_link_file",
    "ensure_managed_directory",
    "expected_config_path",
    "expected_manifest",
    "file_sha256",
    "finalize_identity",
    "flat_file_inventory",
    "load_device_receipt",
    "load_plan",
    "load_playground_registration",
    "pinned_smoke_interpreter",
    "portable_relative_parts",
    "prepare_smoke_environment_binding",
    "prepare_smoke_interpreter_binding",
    "prepare_smoke_orchestration_code_identity",
    "prepare_smoke_runtime_closure",
    "publish_evidence",
    "publish_immutable_tree",
    "publish_plan",
    "publish_playground_snapshot",
    "publish_run_container",
    "read_stable_single_link_bytes",
    "run_bundle_directory",
    "smoke_id_for_campaign",
    "smoke_launch_identity",
    "smoke_runtime_environment_paths",
    "smoke_training_argv",
    "smoke_worker_argv",
    "stable_hash",
    "validate_bound_training_code_identity",
    "validate_cli_configuration",
    "validate_identity",
    "validate_plan",
    "validate_smoke_environment",
    "validate_smoke_interpreter",
    "validate_smoke_orchestration_code",
    "validate_smoke_runtime_closure",
    "verify_complete_bundle",
    "verify_execution_guards",
    "verify_pinned_process_image",
    "verify_prepared_runtime_closure",
    "write_device_receipt",
    "write_exclusive_bytes",
]

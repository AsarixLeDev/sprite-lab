"""OS-enforced write confinement for controlled legacy child processes.

The conditioned Dataset intake still calls a small amount of pathname-based
legacy code.  Holding directory descriptors makes the parent publication
safe, but on POSIX it does not stop another process from renaming a workspace
while that legacy code is running.  Linux children therefore install a
Landlock ruleset bound to the exact already-opened workspace inode before they
import or execute legacy code.

Landlock ABI 10 still does not mediate metadata-only operations such as
``chmod``, ``chown``, ``setxattr``, or ``utime``. This helper is therefore only
valid for the audited conditioned legacy closure, which does not call those
operations. It mediates every content, truncate, create, remove, link, and
rename operation used by that closure; callers must retain a static
no-unmediated-operation regression in the audit-bound test suite.

Windows starts only a pinned interpreter and fixed bootstrap at Low integrity.
The process is created suspended so the parent can assign a kill-on-close,
one-process Job and verify its image.  On resume, the bootstrap irreversibly
lowers the primary token to Untrusted, proves thread inheritance, raise denial,
and Medium/Low outside-write probes, then executes the exact worker.  The
private workspace and stdio are explicitly Untrusted writable. Objects that
are themselves Untrusted and World-writable remain outside this guarantee.
Other platforms fail closed.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import time
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


class WriteConfinementError(RuntimeError):
    """A controlled child could not prove its write boundary."""


class WriteConfinementUnavailable(WriteConfinementError):
    """The current platform cannot establish the required write boundary."""


LINUX_LANDLOCK_STRATEGY = "linux-landlock-v1"
WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY = "windows-bootstrap-to-untrusted-integrity-v1"
# Backward-compatible strategy names resolve to the only current Windows
# bootstrap-to-Untrusted boundary.
WINDOWS_LOW_INTEGRITY_STRATEGY = WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY
WINDOWS_PARENT_ANCHORS_STRATEGY = WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY
_LOW_INTEGRITY_RID = 4096
_UNTRUSTED_INTEGRITY_RID = 0
_MEDIUM_INTEGRITY_RID = 8192
_TOKEN_MANDATORY_POLICY_NO_WRITE_UP = 0x1
_SYSTEM_MANDATORY_LABEL_NO_WRITE_UP = 0x1
_MAX_WINDOWS_LABELED_ENTRIES = 100_000

# Linux UAPI values from <linux/landlock.h>.  Keep the complete handled set
# separate from the smaller allow set: the controlled child may create normal
# files/directories and atomically rename them, but it may not create device
# nodes, sockets, FIFOs, or symlinks.
_ACCESS_FS_EXECUTE = 1 << 0
_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_READ_FILE = 1 << 2
_ACCESS_FS_READ_DIR = 1 << 3
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12
_ACCESS_FS_REFER = 1 << 13
_ACCESS_FS_TRUNCATE = 1 << 14
_ACCESS_FS_IOCTL_DEV = 1 << 15
_ACCESS_FS_RESOLVE_UNIX = 1 << 16

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_MINIMUM_LANDLOCK_ABI = 3  # ABI 3 is the first ABI that can mediate truncate(2).
_MAXIMUM_AUDITED_LANDLOCK_ABI = 10

_BASE_HANDLED_ACCESS = (
    _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_CHAR
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SOCK
    | _ACCESS_FS_MAKE_FIFO
    | _ACCESS_FS_MAKE_BLOCK
    | _ACCESS_FS_MAKE_SYM
    | _ACCESS_FS_REFER
    | _ACCESS_FS_TRUNCATE
)
_ALLOWED_WORKSPACE_ACCESS = (
    _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_REFER
    | _ACCESS_FS_TRUNCATE
)


WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE: Final = r"""
import ctypes,sys

class _SL_SAA(ctypes.Structure):
    _fields_=[("Sid",ctypes.c_void_p),("Attributes",ctypes.c_uint32)]
class _SL_TML(ctypes.Structure):
    _fields_=[("Label",_SL_SAA)]
class _SL_TMP(ctypes.Structure):
    _fields_=[("Policy",ctypes.c_uint32)]
class _SL_LUID(ctypes.Structure):
    _fields_=[("LowPart",ctypes.c_uint32),("HighPart",ctypes.c_int32)]
class _SL_LAA(ctypes.Structure):
    _fields_=[("Luid",_SL_LUID),("Attributes",ctypes.c_uint32)]
class _SL_IO(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ("a","b","c","d","e","f")]
class _SL_BL(ctypes.Structure):
    _fields_=[
        ("process_time",ctypes.c_longlong),("job_time",ctypes.c_longlong),
        ("flags",ctypes.c_uint32),("minimum",ctypes.c_size_t),("maximum",ctypes.c_size_t),
        ("active_limit",ctypes.c_uint32),("affinity",ctypes.c_size_t),
        ("priority",ctypes.c_uint32),("scheduling",ctypes.c_uint32),
    ]
class _SL_EL(ctypes.Structure):
    _fields_=[
        ("basic",_SL_BL),("io",_SL_IO),("process_memory",ctypes.c_size_t),
        ("job_memory",ctypes.c_size_t),("peak_process",ctypes.c_size_t),("peak_job",ctypes.c_size_t),
    ]

_sl_advapi=ctypes.WinDLL("advapi32",use_last_error=True)
_sl_kernel=ctypes.WinDLL("kernel32",use_last_error=True)
_sl_advapi.OpenProcessToken.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_void_p)]
_sl_advapi.OpenProcessToken.restype=ctypes.c_int
_sl_advapi.SetTokenInformation.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
_sl_advapi.SetTokenInformation.restype=ctypes.c_int
_sl_advapi.GetTokenInformation.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
_sl_advapi.GetTokenInformation.restype=ctypes.c_int
_sl_advapi.LookupPrivilegeValueW.argtypes=[ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.POINTER(_SL_LUID)]
_sl_advapi.LookupPrivilegeValueW.restype=ctypes.c_int
_sl_advapi.IsTokenRestricted.argtypes=[ctypes.c_void_p]
_sl_advapi.IsTokenRestricted.restype=ctypes.c_int
_sl_advapi.ConvertStringSidToSidW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_void_p)]
_sl_advapi.ConvertStringSidToSidW.restype=ctypes.c_int
_sl_advapi.GetLengthSid.argtypes=[ctypes.c_void_p]
_sl_advapi.GetLengthSid.restype=ctypes.c_uint32
_sl_advapi.GetSidSubAuthorityCount.argtypes=[ctypes.c_void_p]
_sl_advapi.GetSidSubAuthorityCount.restype=ctypes.POINTER(ctypes.c_ubyte)
_sl_advapi.GetSidSubAuthority.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
_sl_advapi.GetSidSubAuthority.restype=ctypes.POINTER(ctypes.c_uint32)
_sl_kernel.QueryInformationJobObject.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
_sl_kernel.QueryInformationJobObject.restype=ctypes.c_int

def _sl_open_token(access):
    handle=ctypes.c_void_p()
    if not _sl_advapi.OpenProcessToken(_sl_kernel.GetCurrentProcess(),access,ctypes.byref(handle)):
        raise RuntimeError("bootstrap process-token access was denied")
    return handle
def _sl_token_info(token,kind):
    required=ctypes.c_uint32()
    _sl_advapi.GetTokenInformation(token,kind,None,0,ctypes.byref(required))
    if not required.value:
        raise RuntimeError("bootstrap token evidence is unavailable")
    buffer=ctypes.create_string_buffer(required.value)
    if not _sl_advapi.GetTokenInformation(token,kind,buffer,len(buffer),ctypes.byref(required)):
        raise RuntimeError("bootstrap token evidence could not be queried")
    return buffer
def _sl_query(token):
    label=_SL_TML.from_buffer(_sl_token_info(token,25))
    count=_sl_advapi.GetSidSubAuthorityCount(label.Label.Sid).contents.value
    rid=int(_sl_advapi.GetSidSubAuthority(label.Label.Sid,count-1).contents.value)
    policy=int(_SL_TMP.from_buffer(_sl_token_info(token,27)).Policy)
    return rid,bool(policy&1)
def _sl_restricting_sid_bytes(token):
    restricted=bool(_sl_advapi.IsTokenRestricted(token))
    buffer=_sl_token_info(token,11)
    count=int(ctypes.c_uint32.from_buffer(buffer).value)
    if count>4096:
        raise RuntimeError("bootstrap restricting-SID evidence is excessive")
    offset=ctypes.sizeof(ctypes.c_void_p)
    required=offset+ctypes.sizeof(_SL_SAA)*count
    if len(buffer)<required:
        raise RuntimeError("bootstrap restricting-SID evidence is truncated")
    groups=(
        ctypes.cast(
            ctypes.addressof(buffer)+offset,
            ctypes.POINTER(_SL_SAA*count),
        ).contents
        if count
        else ()
    )
    values=tuple(
        sorted(
            ctypes.string_at(group.Sid,int(_sl_advapi.GetLengthSid(group.Sid)))
            for group in groups
        )
    )
    return restricted,values
def _sl_privileges(token):
    buffer=_sl_token_info(token,3)
    if len(buffer)<ctypes.sizeof(ctypes.c_uint32):
        raise RuntimeError("bootstrap privilege evidence is malformed")
    count=int(ctypes.c_uint32.from_buffer(buffer).value)
    if count>4096:
        raise RuntimeError("bootstrap privilege evidence is excessive")
    offset=ctypes.sizeof(ctypes.c_uint32)
    required=offset+ctypes.sizeof(_SL_LAA)*count
    if len(buffer)<required:
        raise RuntimeError("bootstrap privilege evidence is truncated")
    values=(
        ctypes.cast(
            ctypes.addressof(buffer)+offset,
            ctypes.POINTER(_SL_LAA*count),
        ).contents
        if count
        else ()
    )
    return tuple(
        sorted((int(value.Luid.LowPart),int(value.Luid.HighPart),int(value.Attributes)) for value in values)
    )
def _sl_exact_traverse_privilege(token):
    expected=_SL_LUID()
    if not _sl_advapi.LookupPrivilegeValueW(None,"SeChangeNotifyPrivilege",ctypes.byref(expected)):
        raise RuntimeError("bootstrap traverse-privilege identity is unavailable")
    values=_sl_privileges(token)
    return (
        len(values)==1
        and values[0][:2]==(int(expected.LowPart),int(expected.HighPart))
        and bool(values[0][2]&0x2)
        and not values[0][2]&~0x3
    )

_sl_query_handle=_sl_open_token(0x0008)
_sl_startup=_sl_query(_sl_query_handle)
if _sl_startup!=(4096,True):
    raise RuntimeError("bootstrap did not start at exact Low integrity")
_sl_startup_restricted,_sl_startup_sid_bytes=_sl_restricting_sid_bytes(_sl_query_handle)
if not _sl_exact_traverse_privilege(_sl_query_handle):
    raise RuntimeError("bootstrap startup token lacks its exact traverse-only privilege")
_sl_adjust=_sl_open_token(0x0080|0x0008)
_sl_untrusted=ctypes.c_void_p()
try:
    if not _sl_advapi.ConvertStringSidToSidW("S-1-16-0",ctypes.byref(_sl_untrusted)):
        raise RuntimeError("bootstrap Untrusted SID is unavailable")
    _sl_label=_SL_TML(_SL_SAA(_sl_untrusted,0x20))
    _sl_label_size=ctypes.sizeof(_sl_label)+int(_sl_advapi.GetLengthSid(_sl_untrusted))
    if not _sl_advapi.SetTokenInformation(_sl_adjust,25,ctypes.byref(_sl_label),_sl_label_size):
        raise RuntimeError("bootstrap could not lower the primary token")
finally:
    if _sl_untrusted:
        _sl_kernel.LocalFree(_sl_untrusted)
    _sl_kernel.CloseHandle(_sl_adjust)
_sl_lowered=_sl_query(_sl_query_handle)
if _sl_lowered!=(0,True):
    raise RuntimeError("bootstrap primary-token evidence is not exact Untrusted")
if not _sl_exact_traverse_privilege(_sl_query_handle):
    raise RuntimeError("bootstrap Untrusted token changed its traverse-only privilege")

import hashlib,json,os,_thread
_sl_bound_project_root=os.environ.pop("SPRITELAB_CONFINEMENT_PROJECT_ROOT",None)
if _sl_bound_project_root is not None and (
    not _sl_bound_project_root or "\x00" in _sl_bound_project_root
):
    raise RuntimeError("bootstrap project-root binding is malformed")
_sl_expected_restricted=os.environ.pop("SPRITELAB_CONFINEMENT_RESTRICTED_TOKEN","")
_sl_expected_sid_hashes=os.environ.pop("SPRITELAB_CONFINEMENT_RESTRICTED_SID_HASHES","")
try:
    _sl_expected_sid_hashes=tuple(json.loads(_sl_expected_sid_hashes))
except BaseException:
    raise RuntimeError("bootstrap inherited-restriction binding is malformed")
_sl_actual_sid_hashes=tuple(
    sorted(hashlib.sha256(value).hexdigest() for value in _sl_startup_sid_bytes)
)
if (
    _sl_expected_restricted not in ("0","1")
    or _sl_startup_restricted!=(_sl_expected_restricted=="1")
    or _sl_actual_sid_hashes!=_sl_expected_sid_hashes
):
    raise RuntimeError("bootstrap inherited restrictions changed")
_sl_probe_results={}
for _sl_name,_sl_environment_name in (
    ("medium","SPRITELAB_CONFINEMENT_MEDIUM_PROBE"),
    ("low_world","SPRITELAB_CONFINEMENT_LOW_WORLD_PROBE"),
):
    _sl_path=os.environ.pop(_sl_environment_name,"")
    if not _sl_path:
        raise RuntimeError("bootstrap write probe is unavailable")
    try:
        with open(_sl_path,"wb") as _sl_handle:
            _sl_handle.write(b"confinement-compromised")
    except OSError:
        _sl_probe_results[_sl_name]=True
    else:
        _sl_probe_results[_sl_name]=False
if not all(_sl_probe_results.values()):
    raise RuntimeError("bootstrap write confinement probe failed")

_sl_thread_values=[]
_sl_thread_done=_thread.allocate_lock()
_sl_thread_done.acquire()
def _sl_thread_query():
    try:
        _sl_thread_values.append(_sl_query(_sl_query_handle))
    finally:
        _sl_thread_done.release()
_thread.start_new_thread(_sl_thread_query,())
_sl_thread_done.acquire()
_sl_thread_done.release()
if _sl_thread_values!=[(0,True)]:
    raise RuntimeError("bootstrap thread did not inherit exact Untrusted integrity")

_sl_raise_denied=False
try:
    _sl_raise_handle=_sl_open_token(0x0080|0x0008)
except RuntimeError:
    _sl_raise_denied=True
else:
    _sl_low=ctypes.c_void_p()
    try:
        if not _sl_advapi.ConvertStringSidToSidW("S-1-16-4096",ctypes.byref(_sl_low)):
            raise RuntimeError("bootstrap Low SID is unavailable")
        _sl_raise_label=_SL_TML(_SL_SAA(_sl_low,0x20))
        _sl_raise_size=ctypes.sizeof(_sl_raise_label)+int(_sl_advapi.GetLengthSid(_sl_low))
        _sl_raise_denied=not bool(
            _sl_advapi.SetTokenInformation(
                _sl_raise_handle,25,ctypes.byref(_sl_raise_label),_sl_raise_size
            )
        )
    finally:
        if _sl_low:
            _sl_kernel.LocalFree(_sl_low)
        _sl_kernel.CloseHandle(_sl_raise_handle)
if not _sl_raise_denied or _sl_query(_sl_query_handle)!=(0,True):
    raise RuntimeError("bootstrap token could be raised above Untrusted integrity")

_sl_job=_SL_EL()
_sl_job_returned=ctypes.c_uint32()
if not _sl_kernel.QueryInformationJobObject(
    None,9,ctypes.byref(_sl_job),ctypes.sizeof(_sl_job),ctypes.byref(_sl_job_returned)
):
    raise RuntimeError("bootstrap Job evidence is unavailable")
if not (_sl_job.basic.flags&0x2000) or not (_sl_job.basic.flags&0x8) or _sl_job.basic.active_limit!=1:
    raise RuntimeError("bootstrap Job boundary is not exact")
_sl_kernel.CloseHandle(_sl_query_handle)
sys._spritelab_windows_untrusted_evidence={
    "strategy":"windows-bootstrap-to-untrusted-integrity-v1",
    "startup_integrity_level_rid":4096,
    "integrity_level_rid":0,
    "mandatory_no_write_up":True,
    "bootstrap_lowered_before_worker_import":True,
    "new_thread_integrity_level_rid":0,
    "raise_to_low_denied":True,
    "medium_probe_write_denied":True,
    "low_world_probe_write_denied":True,
    "untrusted_world_outside_guaranteed":False,
    "job_kill_on_close":True,
    "job_active_process_limit":1,
    "restricted_token":_sl_startup_restricted,
    "restricted_sid_hashes":_sl_actual_sid_hashes,
}
if _sl_bound_project_root is not None:
    sys._spritelab_windows_project_root=_sl_bound_project_root
""".lstrip()
_WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256: Final = hashlib.sha256(
    WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE.encode("utf-8")
).hexdigest()
WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256: Final = _WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256


@dataclass(frozen=True)
class DirectoryIdentity:
    """Machine-local identity for one exact held directory."""

    device: int
    inode: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> DirectoryIdentity:
        if not stat.S_ISDIR(metadata.st_mode):
            raise WriteConfinementError("The write-confinement root is not a directory.")
        return cls(device=int(metadata.st_dev), inode=int(metadata.st_ino))

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(
            {"device": self.device, "inode": self.inode},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WriteConfinementEvidence:
    """Path-free evidence returned after the kernel accepts confinement."""

    strategy: str
    platform: str
    kernel_abi: int
    root_identity_sha256: str
    handled_access_fs: int
    allowed_access_fs: int
    no_new_privileges: bool
    restricted_token: bool = False
    integrity_level_rid: int = 0
    mandatory_no_write_up: bool = False
    workspace_integrity_level_rid: int = 0
    startup_integrity_level_rid: int = 0
    bootstrap_lowered_before_worker_import: bool = False
    new_thread_integrity_level_rid: int = 0
    raise_to_low_denied: bool = False
    medium_probe_write_denied: bool = False
    low_world_probe_write_denied: bool = False
    untrusted_world_outside_guaranteed: bool = False
    job_kill_on_close: bool = False
    job_active_process_limit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.write-confinement-evidence.v3",
            "strategy": self.strategy,
            "platform": self.platform,
            "kernel_abi": self.kernel_abi,
            "root_identity_sha256": self.root_identity_sha256,
            "handled_access_fs": self.handled_access_fs,
            "allowed_access_fs": self.allowed_access_fs,
            "no_new_privileges": self.no_new_privileges,
            "restricted_token": self.restricted_token,
            "integrity_level_rid": self.integrity_level_rid,
            "mandatory_no_write_up": self.mandatory_no_write_up,
            "workspace_integrity_level_rid": self.workspace_integrity_level_rid,
            "startup_integrity_level_rid": self.startup_integrity_level_rid,
            "bootstrap_lowered_before_worker_import": self.bootstrap_lowered_before_worker_import,
            "new_thread_integrity_level_rid": self.new_thread_integrity_level_rid,
            "raise_to_low_denied": self.raise_to_low_denied,
            "medium_probe_write_denied": self.medium_probe_write_denied,
            "low_world_probe_write_denied": self.low_world_probe_write_denied,
            "untrusted_world_outside_guaranteed": self.untrusted_world_outside_guaranteed,
            "job_kill_on_close": self.job_kill_on_close,
            "job_active_process_limit": self.job_active_process_limit,
            "paths_exposed": False,
        }


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def write_confinement_strategy() -> str:
    """Return the only accepted strategy for this platform.

    Windows callers must label the exact private workspace and create the child
    through :func:`create_windows_bootstrap_untrusted_process`. Linux callers call
    :func:`enforce_linux_landlock_write_confinement` in the child.
    """

    if sys.platform.startswith("linux"):
        return LINUX_LANDLOCK_STRATEGY
    if sys.platform == "win32":
        return WINDOWS_LOW_INTEGRITY_STRATEGY
    raise WriteConfinementUnavailable(
        "Conditioned legacy writes are unavailable because this platform has no approved write confinement."
    )


def directory_identity(path: str | Path) -> DirectoryIdentity:
    """Open and identify one exact no-follow directory without mutating it."""

    raw = os.fspath(path)
    if not raw or not raw.strip() or raw.strip() in {".", ".."}:
        raise WriteConfinementError("The write-confinement root must be an explicit directory.")
    absolute = Path(os.path.abspath(os.path.expanduser(raw)))
    before = absolute.lstat()
    if not stat.S_ISDIR(before.st_mode) or _metadata_is_link_or_reparse(before):
        raise WriteConfinementError("The write-confinement root is linked or not a directory.")
    if os.name == "nt":
        # ``os.open`` does not provide the required directory-handle contract
        # on Windows.  Reuse the repository's native non-share-delete anchor.
        from spritelab.utils.safe_fs import AnchoredDirectory

        with AnchoredDirectory(absolute, absolute) as anchor:
            return DirectoryIdentity.from_stat(anchor.directory_metadata())
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        after = absolute.lstat()
        _require_same_directory(before, opened)
        _require_same_directory(before, after)
        return DirectoryIdentity.from_stat(opened)
    finally:
        os.close(descriptor)


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", ctypes.c_uint32)]


class _TokenUser(ctypes.Structure):
    _fields_ = [("User", _SidAndAttributes)]


class _TokenMandatoryLabel(ctypes.Structure):
    _fields_ = [("Label", _SidAndAttributes)]


class _TokenMandatoryPolicy(ctypes.Structure):
    _fields_ = [("Policy", ctypes.c_uint32)]


class _TokenGroups(ctypes.Structure):
    _fields_ = [("GroupCount", ctypes.c_uint32), ("Groups", _SidAndAttributes * 1)]


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _LuidAndAttributes(ctypes.Structure):
    _fields_ = [("Luid", _Luid), ("Attributes", ctypes.c_uint32)]


class _TokenPrivileges(ctypes.Structure):
    _fields_ = [("PrivilegeCount", ctypes.c_uint32), ("Privileges", _LuidAndAttributes * 1)]


class _AclSizeInformation(ctypes.Structure):
    _fields_ = [
        ("AceCount", ctypes.c_uint32),
        ("AclBytesInUse", ctypes.c_uint32),
        ("AclBytesFree", ctypes.c_uint32),
    ]


class _TrusteeW(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),
    ]


class _ExplicitAccessW(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", ctypes.c_uint32),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", ctypes.c_uint32),
        ("Trustee", _TrusteeW),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", ctypes.c_uint32),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", ctypes.c_int),
    ]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32),
        ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32),
        ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32),
        ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_ushort),
        ("cbReserved2", ctypes.c_ushort),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_uint32),
        ("dwThreadId", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class WindowsLowIntegrityRoot:
    """One exact root labeled writable by a low-integrity child."""

    identity: DirectoryIdentity
    entry_count: int


class WindowsLowIntegrityProcess:
    """Small ``Popen``-compatible wrapper for a bootstrap-lowered process."""

    def __init__(
        self,
        *,
        args: Sequence[str],
        process_handle: int,
        pid: int,
        stdin_descriptor: int,
        stdout_descriptor: int,
        stderr_descriptor: int,
        expected_input: bytes,
        max_stdout_bytes: int,
        bootstrap_identity_sha256: str,
        desktop_handle: int,
        desktop_identity_sha256: str,
        confinement_probes: Mapping[str, tuple[Path, bytes]],
        confinement_anchors: ExitStack,
        restricted_token: bool,
        restricted_sid_hashes: Sequence[str],
    ) -> None:
        self.args = tuple(args)
        self._handle = process_handle
        self.pid = pid
        self.returncode: int | None = None
        self._stdin_descriptor = stdin_descriptor
        self._stdout_descriptor = stdout_descriptor
        self._stderr_descriptor = stderr_descriptor
        self._expected_input = expected_input
        self._max_stdout_bytes = max_stdout_bytes
        self.bootstrap_identity_sha256 = bootstrap_identity_sha256
        self._desktop_handle = desktop_handle
        self.private_desktop_identity_sha256 = desktop_identity_sha256
        self._confinement_probes = dict(confinement_probes)
        self._confinement_anchors = confinement_anchors
        self.restricted_token = restricted_token
        self.restricted_sid_hashes_identity_sha256 = hashlib.sha256(
            json.dumps(
                tuple(restricted_sid_hashes),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    def poll(self) -> int | None:
        self._enforce_stdio_limit()
        return self._poll_process_handle()

    def _poll_process_handle(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if not self._handle:
            return self.returncode
        kernel32 = _windows_kernel32()
        result = int(kernel32.WaitForSingleObject(self._handle, 0))
        if result == 0x102:
            return None
        if result != 0:
            raise OSError("unable to query the restricted child process")
        self.returncode = _windows_process_exit_code(kernel32, self._handle)
        self._verify_confinement_probes()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._enforce_stdio_limit()
        if self.returncode is not None:
            return self.returncode
        if not self._handle:
            raise OSError("restricted child process handle is unavailable")
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError("timeout must be a non-negative number")
        kernel32 = _windows_kernel32()
        started = _windows_monotonic()
        while True:
            self._enforce_stdio_limit()
            if timeout is None:
                interval = 50
            else:
                remaining = float(timeout) - (_windows_monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self.args, timeout)
                interval = max(1, min(50, int(remaining * 1000)))
            result = int(kernel32.WaitForSingleObject(self._handle, interval))
            if result == 0:
                self.returncode = _windows_process_exit_code(kernel32, self._handle)
                self._verify_confinement_probes()
                return self.returncode
            if result != 0x102:
                raise OSError("unable to wait for the restricted child process")

    def communicate(
        self,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        if input is not None and bytes(input) != self._expected_input:
            raise ValueError("restricted child input differs from its preloaded bytes")
        self.wait(timeout=timeout)
        output = _read_windows_stdio_descriptor(self._stdout_descriptor, self._max_stdout_bytes)
        error_output = _read_windows_stdio_descriptor(self._stderr_descriptor, self._max_stdout_bytes)
        self.close()
        if len(output) > self._max_stdout_bytes or len(error_output) > self._max_stdout_bytes:
            raise OSError("restricted child stdout exceeded its byte limit")
        return output, error_output

    def terminate(self) -> None:
        if self._poll_process_handle() is not None:
            return
        kernel32 = _windows_kernel32()
        if not kernel32.TerminateProcess(self._handle, 1):
            raise OSError("unable to terminate the restricted child process")

    kill = terminate

    def _enforce_stdio_limit(self) -> None:
        descriptors = (self._stdout_descriptor, self._stderr_descriptor)
        open_descriptors = tuple(descriptor for descriptor in descriptors if descriptor >= 0)
        if not open_descriptors:
            return
        if len(open_descriptors) != len(descriptors):
            raise OSError("restricted child stdio is unavailable")
        if any(os.fstat(descriptor).st_size > self._max_stdout_bytes for descriptor in open_descriptors):
            self.terminate()
            raise OSError("restricted child stdout exceeded its byte limit")

    def _verify_confinement_probes(self) -> None:
        for name, (path, expected) in sorted(self._confinement_probes.items()):
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise OSError(f"the {name} confinement probe is unavailable") from exc
            if payload != expected:
                raise OSError(f"the {name} confinement probe was modified")

    def close(self) -> None:
        for name in ("_stdin_descriptor", "_stdout_descriptor", "_stderr_descriptor"):
            descriptor = int(getattr(self, name, -1))
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, name, -1)
        if self._handle:
            _windows_kernel32().CloseHandle(self._handle)
            self._handle = 0
        if self._desktop_handle:
            _windows_user32().CloseDesktop(self._desktop_handle)
            self._desktop_handle = 0
        anchors = getattr(self, "_confinement_anchors", None)
        if anchors is not None:
            anchors.close()
            self._confinement_anchors = None

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def prepare_windows_low_integrity_workspace(root: str | Path) -> WindowsLowIntegrityRoot:
    """Label one exact empty root before any private-tree population."""

    return prepare_windows_low_integrity_roots((root,))[0]


def prepare_windows_low_integrity_roots(
    roots: Sequence[str | Path],
) -> tuple[WindowsLowIntegrityRoot, ...]:
    """Label exact empty roots once, with inheritable Low mandatory labels."""

    return _prepare_windows_integrity_roots(roots, integrity_rid=_LOW_INTEGRITY_RID)


def prepare_windows_untrusted_integrity_workspace(root: str | Path) -> WindowsLowIntegrityRoot:
    """Label one exact empty root before any private-tree population."""

    return prepare_windows_untrusted_integrity_roots((root,))[0]


def prepare_windows_untrusted_integrity_roots(
    roots: Sequence[str | Path],
) -> tuple[WindowsLowIntegrityRoot, ...]:
    """Label exact empty roots once, with inheritable Untrusted labels."""

    return _prepare_windows_integrity_roots(roots, integrity_rid=_UNTRUSTED_INTEGRITY_RID)


def _prepare_windows_integrity_roots(
    roots: Sequence[str | Path],
    *,
    integrity_rid: int,
) -> tuple[WindowsLowIntegrityRoot, ...]:
    from spritelab.utils.safe_fs import AnchoredDirectory

    if integrity_rid not in {_UNTRUSTED_INTEGRITY_RID, _LOW_INTEGRITY_RID}:
        raise WriteConfinementError("The Windows workspace integrity level is unsupported.")

    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows integrity labeling is unavailable on this platform.")
    if not roots:
        raise WriteConfinementError("At least one private low-integrity root is required.")
    prepared: list[WindowsLowIntegrityRoot] = []
    seen: set[str] = set()
    try:
        with ExitStack() as anchors:
            for raw in roots:
                absolute = _explicit_windows_private_root(raw)
                key = os.path.normcase(os.fspath(absolute))
                if key in seen:
                    raise WriteConfinementError("Duplicate low-integrity roots are not allowed.")
                seen.add(key)
                anchor = anchors.enter_context(AnchoredDirectory(absolute, absolute))
                before = anchor.directory_metadata()
                if anchor.names():
                    raise WriteConfinementError("A private Windows workspace must be labeled before population.")
                _set_windows_mandatory_integrity_label_handle(
                    anchor._required_windows_handle(),
                    directory=True,
                    integrity_rid=integrity_rid,
                )
                anchor.verify()
                after = anchor.directory_metadata()
                if _windows_metadata_binding(after) != _windows_metadata_binding(before):
                    raise WriteConfinementError("A private Windows workspace root changed while it was labeled.")
                if _windows_handle_integrity_label(anchor._required_windows_handle(), directory=True) != (
                    integrity_rid,
                    True,
                ):
                    raise WriteConfinementError("A private root lacks its exact mandatory-integrity write label.")
                if anchor.names():
                    raise WriteConfinementError("A private Windows workspace changed while it was labeled.")
                prepared.append(
                    WindowsLowIntegrityRoot(
                        identity=DirectoryIdentity.from_stat(after),
                        entry_count=1,
                    )
                )
    except WriteConfinementError:
        raise
    except (OSError, ValueError) as exc:
        raise WriteConfinementError("A private Windows workspace root could not be anchored safely.") from exc
    return tuple(prepared)


def _verify_windows_integrity_roots(
    roots: Sequence[str | Path],
    *,
    integrity_rid: int,
) -> tuple[WindowsLowIntegrityRoot, ...]:
    """Read-only verification of inherited labels across bounded private trees."""

    verified, anchors = _retain_verified_windows_integrity_roots(roots, integrity_rid=integrity_rid)
    anchors.close()
    return verified


def _retain_verified_windows_integrity_roots(
    roots: Sequence[str | Path],
    *,
    integrity_rid: int,
) -> tuple[tuple[WindowsLowIntegrityRoot, ...], ExitStack]:
    """Verify every entry through exact handles and retain all root anchors."""

    from spritelab.utils.safe_fs import AnchoredDirectory

    anchors = ExitStack()
    verified: list[WindowsLowIntegrityRoot] = []
    seen: set[str] = set()
    try:
        for raw in roots:
            absolute = _explicit_windows_private_root(raw)
            key = os.path.normcase(os.fspath(absolute))
            if key in seen:
                raise WriteConfinementError("Duplicate low-integrity roots are not allowed.")
            seen.add(key)
            anchor = anchors.enter_context(AnchoredDirectory(absolute, absolute))
            entry_count = _verify_windows_integrity_tree_anchor(
                anchor,
                integrity_rid=integrity_rid,
                device=int(anchor.directory_metadata().st_dev),
            )
            verified.append(
                WindowsLowIntegrityRoot(
                    identity=DirectoryIdentity.from_stat(anchor.directory_metadata()),
                    entry_count=entry_count,
                )
            )
        return tuple(verified), anchors
    except BaseException:
        anchors.close()
        raise


def _verify_windows_integrity_tree_anchor(
    anchor: Any,
    *,
    integrity_rid: int,
    device: int,
    depth: int = 0,
) -> int:
    from spritelab.utils.safe_fs import OwnedFileIdentity

    if depth > 128:
        raise WriteConfinementError("A private Windows workspace exceeds its depth limit.")
    anchor.verify()
    if _windows_handle_integrity_label(anchor._required_windows_handle(), directory=True) != (
        integrity_rid,
        True,
    ):
        raise WriteConfinementError("A private-tree entry lacks its inherited mandatory-integrity label.")
    before_names = anchor.names()
    collision_keys: set[str] = set()
    count = 1
    for name in before_names:
        collision_key = unicodedata.normalize("NFKC", name).casefold()
        if collision_key in collision_keys:
            raise WriteConfinementError("A private Windows workspace contains a name collision.")
        collision_keys.add(collision_key)
        metadata = anchor.lstat(name)
        if _metadata_is_link_or_reparse(metadata) or int(metadata.st_dev) != device:
            raise WriteConfinementError("A private Windows workspace crosses a link or device seam.")
        if stat.S_ISDIR(metadata.st_mode):
            with anchor.open_directory_immovable(name) as child:
                count += _verify_windows_integrity_tree_anchor(
                    child,
                    integrity_rid=integrity_rid,
                    device=device,
                    depth=depth + 1,
                )
        elif stat.S_ISREG(metadata.st_mode):
            descriptor = anchor.open_file_immovable(
                name,
                os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
            )
            try:
                held = os.fstat(descriptor)
                identity = OwnedFileIdentity.from_stat(held)
                if int(held.st_nlink) != 1 or not identity.matches(metadata):
                    raise WriteConfinementError("A private Windows workspace contains an unsafe file.")
                if _windows_handle_integrity_label(_windows_os_handle(descriptor), directory=False) != (
                    integrity_rid,
                    True,
                ):
                    raise WriteConfinementError("A private-tree entry lacks its inherited mandatory-integrity label.")
                if not identity.matches(anchor.lstat(name)):
                    raise WriteConfinementError("A private Windows workspace changed during label verification.")
            finally:
                os.close(descriptor)
            count += 1
        else:
            raise WriteConfinementError("A private Windows workspace contains a non-regular entry.")
        if count > _MAX_WINDOWS_LABELED_ENTRIES:
            raise WriteConfinementError("A private Windows workspace exceeds its entry limit.")
    if anchor.names() != before_names:
        raise WriteConfinementError("A private Windows workspace changed during label verification.")
    anchor.verify()
    return count


def _windows_untrusted_bootstrap_arguments(
    arguments: Sequence[str],
    *,
    bootstrap_source: str | None = None,
    _expected_bootstrap_sha256: str = _WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256,
) -> tuple[str, ...]:
    values = tuple(arguments)
    if (
        len(values) < 6
        or values[1:5] not in {("-I", "-S", "-B", "-c"), ("-I", "-B", "-S", "-c")}
        or values.count("-c") != 1
    ):
        raise WriteConfinementError("The Windows bootstrap requires the exact -I -S -B -c interpreter boundary.")
    worker_source = values[5]
    if not worker_source or "\x00" in worker_source:
        raise WriteConfinementError("The Windows worker bootstrap source is invalid.")
    outer_source = WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE if bootstrap_source is None else bootstrap_source
    if (
        not outer_source
        or "\x00" in outer_source
        or hashlib.sha256(outer_source.encode("utf-8")).hexdigest() != _expected_bootstrap_sha256
    ):
        raise WriteConfinementError("The Windows confinement bootstrap differs from its audited source.")
    wrapped = (
        outer_source
        + "\nexec(compile("
        + repr(worker_source)
        + ",'<spritelab-bound-worker>','exec',dont_inherit=True),globals(),globals())\n"
    )
    return (*values[:5], wrapped, *values[6:])


def _create_windows_untrusted_confinement_probes(
    boundary: Path,
) -> tuple[dict[str, tuple[Path, bytes]], ExitStack]:
    from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity

    parent = _explicit_windows_private_root(boundary)
    anchors = ExitStack()
    try:
        parent_anchor = anchors.enter_context(AnchoredDirectory(parent, parent))
        probe_name, probe_identity = parent_anchor.mkdir_unique(".spritelab-confinement-probe-")
        if not probe_identity.matches(parent_anchor.lstat(probe_name)):
            raise WriteConfinementError("A Windows confinement probe root changed during creation.")
        probe_anchor = anchors.enter_context(parent_anchor.open_directory_immovable(probe_name))
        _set_windows_mandatory_integrity_label_handle(
            probe_anchor._required_windows_handle(),
            directory=True,
            integrity_rid=_MEDIUM_INTEGRITY_RID,
        )
        if _windows_handle_integrity_label(probe_anchor._required_windows_handle(), directory=True) != (
            _MEDIUM_INTEGRITY_RID,
            True,
        ):
            raise WriteConfinementError("A Windows confinement probe root lacks its Medium label.")
        probe_anchor.mkdir("medium")
        probe_anchor.mkdir("low-world")
        medium_anchor = anchors.enter_context(probe_anchor.open_directory_immovable("medium"))
        low_anchor = anchors.enter_context(probe_anchor.open_directory_immovable("low-world"))
        _set_windows_mandatory_integrity_label_handle(
            low_anchor._required_windows_handle(),
            directory=True,
            integrity_rid=_LOW_INTEGRITY_RID,
        )
        world_sid = _windows_string_sid("S-1-1-0")
        try:
            _grant_windows_handle_sid(low_anchor._required_windows_handle(), world_sid, directory=True)
        finally:
            _windows_kernel32().LocalFree(world_sid)
        payloads = {
            "medium": b"spritelab-medium-confinement-probe-v1",
            "low_world": b"spritelab-low-world-confinement-probe-v1",
        }
        result: dict[str, tuple[Path, bytes]] = {}
        for name, directory_anchor, directory_name in (
            ("medium", medium_anchor, "medium"),
            ("low_world", low_anchor, "low-world"),
        ):
            descriptor = directory_anchor.open_file_immovable(
                "sentinel.bin",
                os.O_CREAT | os.O_EXCL | os.O_RDWR | int(getattr(os, "O_BINARY", 0)),
                0o600,
            )
            anchors.callback(os.close, descriptor)
            identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
            if os.write(descriptor, payloads[name]) != len(payloads[name]):
                raise OSError("unable to initialize a Windows confinement probe")
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not identity.matches(directory_anchor.lstat("sentinel.bin"))
            ):
                raise WriteConfinementError("A Windows confinement probe is unsafe.")
            expected_rid = _MEDIUM_INTEGRITY_RID if name == "medium" else _LOW_INTEGRITY_RID
            if _windows_handle_integrity_label(_windows_os_handle(descriptor), directory=False) != (
                expected_rid,
                True,
            ):
                raise WriteConfinementError("A Windows confinement probe lacks its exact protected identity.")
            if _read_windows_stdio_descriptor(descriptor, len(payloads[name])) != payloads[name]:
                raise WriteConfinementError("A Windows confinement probe changed during initialization.")
            result[name] = (parent / probe_name / directory_name / "sentinel.bin", payloads[name])
        parent_anchor.verify()
        probe_anchor.verify()
        return result, anchors
    except BaseException:
        anchors.close()
        raise


def create_windows_bootstrap_untrusted_process(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    stdin_payload: bytes,
    writable_roots: Sequence[str | Path] | None = None,
    stdio_root: str | Path | None = None,
    inherited_handles: Sequence[int] = (),
    max_stdout_bytes: int = 16 * 1024 * 1024,
) -> WindowsLowIntegrityProcess:
    """Create a suspended Low-startup process that bootstraps to Untrusted.

    The caller must assign the returned suspended process to the repository's
    one-process kill-on-close Job and verify its pinned image before resume.
    Only the fixed audited bootstrap executes at Low integrity.  It lowers the
    primary token to Untrusted, proves the Job/token boundary and two outside
    write probes, then executes the caller's exact ``-c`` source.
    """

    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows bootstrap confinement is unavailable on this platform.")
    raw_arguments = tuple(os.fspath(value) for value in argv)
    outer_bootstrap_source = WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE
    arguments = _windows_untrusted_bootstrap_arguments(
        raw_arguments,
        bootstrap_source=outer_bootstrap_source,
    )
    if not arguments or any(not value or "\x00" in value for value in arguments):
        raise WriteConfinementError("The bootstrap-confined child command is invalid.")
    if not isinstance(stdin_payload, bytes):
        raise WriteConfinementError("The restricted child input must be exact bytes.")
    if type(max_stdout_bytes) is not int or not 0 < max_stdout_bytes <= 128 * 1024 * 1024:
        raise WriteConfinementError("The restricted child stdout limit is invalid.")
    if (
        not isinstance(inherited_handles, Sequence)
        or any(type(handle) is not int or handle <= 0 for handle in inherited_handles)
        or len(set(inherited_handles)) != len(inherited_handles)
        or len(inherited_handles) > 16
    ):
        raise WriteConfinementError("The restricted child handle list is invalid.")
    working_directory = _explicit_windows_private_root(cwd)
    selected_roots = tuple(writable_roots or (working_directory,))
    verified_roots, retained_root_anchors = _retain_verified_windows_integrity_roots(
        selected_roots,
        integrity_rid=_UNTRUSTED_INTEGRITY_RID,
    )
    confinement_anchors = ExitStack()
    confinement_anchors.callback(retained_root_anchors.close)
    try:
        if not any(
            _path_is_same_or_descendant(working_directory, _explicit_windows_private_root(root))
            for root in selected_roots
        ):
            raise WriteConfinementError("The bootstrap-confined child cwd is outside its writable roots.")
        del verified_roots
        io_directory = working_directory / "tmp" if stdio_root is None else _explicit_windows_private_root(stdio_root)
        if not _path_is_same_or_descendant(io_directory, working_directory):
            raise WriteConfinementError("The bootstrap-confined child stdio root is outside its private workspace.")
        io_metadata = io_directory.lstat()
        if not stat.S_ISDIR(io_metadata.st_mode) or _metadata_is_link_or_reparse(io_metadata):
            raise WriteConfinementError("The bootstrap-confined child stdio root is unsafe.")
        confinement_probes, probe_anchors = _create_windows_untrusted_confinement_probes(working_directory)
        confinement_anchors.callback(probe_anchors.close)
    except BaseException:
        confinement_anchors.close()
        raise
    child_environment = dict(env)
    child_environment["SPRITELAB_CONFINEMENT_MEDIUM_PROBE"] = os.fspath(confinement_probes["medium"][0])
    child_environment["SPRITELAB_CONFINEMENT_LOW_WORLD_PROBE"] = os.fspath(confinement_probes["low_world"][0])

    stdin_descriptor = -1
    stdout_descriptor = -1
    stderr_descriptor = -1
    token = 0
    process_handle = 0
    desktop_handle = 0
    attribute_buffer: ctypes.Array[ctypes.c_char] | None = None
    inherited_handle_states: dict[int, bool] = {}
    kernel32 = _windows_kernel32()
    try:
        stdin_descriptor = _exclusive_windows_stdio_file(
            io_directory,
            "stdin",
            stdin_payload,
            integrity_rid=_UNTRUSTED_INTEGRITY_RID,
        )
        stdout_descriptor = _exclusive_windows_stdio_file(
            io_directory,
            "stdout",
            b"",
            integrity_rid=_UNTRUSTED_INTEGRITY_RID,
        )
        stderr_descriptor = _exclusive_windows_stdio_file(
            io_directory,
            "stderr",
            b"",
            integrity_rid=_UNTRUSTED_INTEGRITY_RID,
        )
        stdio_handles = tuple(_windows_os_handle(fd) for fd in (stdin_descriptor, stdout_descriptor, stderr_descriptor))
        if set(stdio_handles).intersection(inherited_handles):
            raise WriteConfinementError("The restricted child handle list overlaps its private stdio.")
        for handle in inherited_handles:
            try:
                inherited_handle_states[handle] = os.get_handle_inheritable(handle)
            except OSError as exc:
                raise WriteConfinementError("A restricted child inherited handle is unavailable.") from exc
        handles = (*stdio_handles, *inherited_handles)
        for handle in handles:
            os.set_handle_inheritable(handle, True)

        token = _create_windows_low_startup_token()
        restricted, integrity_rid, no_write_up, restricted_sid_hashes = _windows_token_confinement(token)
        if (
            integrity_rid != _LOW_INTEGRITY_RID
            or not no_write_up
            or restricted != bool(restricted or restricted_sid_hashes)
            or not _windows_token_has_only_enabled_traverse_privilege(token)
        ):
            raise WriteConfinementError("The bootstrap startup token failed exact Low-integrity verification.")
        child_environment["SPRITELAB_CONFINEMENT_RESTRICTED_TOKEN"] = "1" if restricted else "0"
        child_environment["SPRITELAB_CONFINEMENT_RESTRICTED_SID_HASHES"] = json.dumps(
            restricted_sid_hashes,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        desktop_handle, desktop_name, desktop_identity_sha256 = _create_windows_private_desktop()

        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(_StartupInfoEx)
        startup.StartupInfo.lpDesktop = desktop_name
        startup.StartupInfo.dwFlags = 0x00000100
        startup.StartupInfo.hStdInput = handles[0]
        startup.StartupInfo.hStdOutput = handles[1]
        startup.StartupInfo.hStdError = handles[2]
        attribute_size = ctypes.c_size_t()
        kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attribute_size))
        if not attribute_size.value:
            raise WriteConfinementUnavailable("Windows explicit child handle-list setup is unavailable.")
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        startup.lpAttributeList = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not kernel32.InitializeProcThreadAttributeList(
            startup.lpAttributeList,
            1,
            0,
            ctypes.byref(attribute_size),
        ):
            raise WriteConfinementUnavailable("Windows explicit child handle-list setup failed.")
        handle_array = (ctypes.c_void_p * len(handles))(*handles)
        if not kernel32.UpdateProcThreadAttribute(
            startup.lpAttributeList,
            0,
            ctypes.c_size_t(0x00020002),
            ctypes.cast(handle_array, ctypes.c_void_p),
            ctypes.sizeof(handle_array),
            None,
            None,
        ):
            raise WriteConfinementUnavailable("Windows explicit child handle-list binding failed.")

        environment = _windows_environment_block(child_environment)
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(arguments)))
        process_information = _ProcessInformation()
        advapi32 = _windows_advapi32()
        creation_flags = 0x00000004 | 0x08000000 | 0x00000400 | 0x00080000
        if not advapi32.CreateProcessAsUserW(
            token,
            arguments[0],
            command_line,
            None,
            None,
            True,
            creation_flags,
            ctypes.cast(environment, ctypes.c_void_p),
            os.fspath(working_directory),
            ctypes.byref(startup),
            ctypes.byref(process_information),
        ):
            code = ctypes.get_last_error()
            raise WriteConfinementUnavailable(f"Windows could not create the bootstrap-confined child (error {code}).")
        process_handle = int(process_information.hProcess)
        kernel32.CloseHandle(process_information.hThread)
        for handle in stdio_handles:
            os.set_handle_inheritable(handle, False)
        for handle, was_inheritable in inherited_handle_states.items():
            os.set_handle_inheritable(handle, was_inheritable)
        inherited_handle_states.clear()
        process = WindowsLowIntegrityProcess(
            args=raw_arguments,
            process_handle=process_handle,
            pid=int(process_information.dwProcessId),
            stdin_descriptor=stdin_descriptor,
            stdout_descriptor=stdout_descriptor,
            stderr_descriptor=stderr_descriptor,
            expected_input=stdin_payload,
            max_stdout_bytes=max_stdout_bytes,
            bootstrap_identity_sha256=hashlib.sha256(outer_bootstrap_source.encode("utf-8")).hexdigest(),
            desktop_handle=desktop_handle,
            desktop_identity_sha256=desktop_identity_sha256,
            confinement_probes=confinement_probes,
            confinement_anchors=confinement_anchors,
            restricted_token=restricted,
            restricted_sid_hashes=restricted_sid_hashes,
        )
        process_handle = 0
        desktop_handle = 0
        stdin_descriptor = stdout_descriptor = stderr_descriptor = -1
        confinement_anchors = ExitStack()
        return process
    finally:
        if attribute_buffer is not None:
            try:
                kernel32.DeleteProcThreadAttributeList(ctypes.cast(attribute_buffer, ctypes.c_void_p))
            except (AttributeError, OSError):
                pass
        if token:
            kernel32.CloseHandle(token)
        if process_handle:
            kernel32.TerminateProcess(process_handle, 1)
            kernel32.CloseHandle(process_handle)
        if desktop_handle:
            _windows_user32().CloseDesktop(desktop_handle)
        for handle, was_inheritable in inherited_handle_states.items():
            try:
                os.set_handle_inheritable(handle, was_inheritable)
            except OSError:
                pass
        for descriptor in (stdin_descriptor, stdout_descriptor, stderr_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        confinement_anchors.close()


def windows_current_process_confinement_evidence(
    workspace: str | Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> WriteConfinementEvidence:
    """Query exact bootstrap-to-Untrusted evidence inside the worker."""

    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows bootstrap confinement evidence is unavailable.")
    absolute = _explicit_windows_private_root(workspace, allow_unavailable_profile=True)
    workspace_metadata = absolute.lstat()
    if not stat.S_ISDIR(workspace_metadata.st_mode) or _metadata_is_link_or_reparse(workspace_metadata):
        raise WriteConfinementError("The write-confinement workspace is unsafe.")
    identity = DirectoryIdentity.from_stat(workspace_metadata)
    if identity.device != expected_device or identity.inode != expected_inode:
        raise WriteConfinementError("The write-confinement root identity changed before child verification.")
    restricted, integrity_rid, mandatory_no_write_up, restricted_sid_hashes = _windows_token_confinement(None)
    privileges = _windows_token_privileges(None)
    workspace_rid, workspace_no_write_up = _windows_path_integrity_label(absolute)
    bootstrap = getattr(sys, "_spritelab_windows_untrusted_evidence", None)
    expected_bootstrap = {
        "strategy": WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY,
        "startup_integrity_level_rid": _LOW_INTEGRITY_RID,
        "integrity_level_rid": _UNTRUSTED_INTEGRITY_RID,
        "mandatory_no_write_up": True,
        "bootstrap_lowered_before_worker_import": True,
        "new_thread_integrity_level_rid": _UNTRUSTED_INTEGRITY_RID,
        "raise_to_low_denied": True,
        "medium_probe_write_denied": True,
        "low_world_probe_write_denied": True,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": True,
        "job_active_process_limit": 1,
        "restricted_token": restricted,
        "restricted_sid_hashes": restricted_sid_hashes,
    }
    if (
        integrity_rid != _UNTRUSTED_INTEGRITY_RID
        or not mandatory_no_write_up
        or workspace_rid != _UNTRUSTED_INTEGRITY_RID
        or not workspace_no_write_up
        or not isinstance(bootstrap, Mapping)
        or dict(bootstrap) != expected_bootstrap
        or not _is_only_enabled_traverse_privilege(privileges)
    ):
        raise WriteConfinementError("The Windows worker lacks its exact bootstrap-to-Untrusted boundary.")
    return WriteConfinementEvidence(
        strategy=WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY,
        platform="windows",
        kernel_abi=0,
        root_identity_sha256=identity.identity_sha256,
        handled_access_fs=0,
        allowed_access_fs=0,
        no_new_privileges=False,
        restricted_token=restricted,
        integrity_level_rid=integrity_rid,
        mandatory_no_write_up=True,
        workspace_integrity_level_rid=workspace_rid,
        startup_integrity_level_rid=_LOW_INTEGRITY_RID,
        bootstrap_lowered_before_worker_import=True,
        new_thread_integrity_level_rid=_UNTRUSTED_INTEGRITY_RID,
        raise_to_low_denied=True,
        medium_probe_write_denied=True,
        low_world_probe_write_denied=True,
        untrusted_world_outside_guaranteed=False,
        job_kill_on_close=True,
        job_active_process_limit=1,
    )


def _explicit_windows_private_root(
    root: str | Path,
    *,
    allow_unavailable_profile: bool = False,
) -> Path:
    raw = os.fspath(root)
    if not raw or not raw.strip() or raw.strip() in {".", ".."} or "\x00" in raw:
        raise WriteConfinementError("A low-integrity root must be an explicit directory.")
    absolute = Path(os.path.abspath(os.path.expanduser(raw)))
    try:
        profile = Path.home()
    except RuntimeError as exc:
        if not allow_unavailable_profile:
            raise WriteConfinementError("The Windows profile boundary is unavailable.") from exc
        profile = None
    if not absolute.is_absolute() or absolute == Path(absolute.anchor) or (profile is not None and absolute == profile):
        raise WriteConfinementError("A filesystem/profile root cannot be a low-integrity workspace.")
    metadata = absolute.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
        raise WriteConfinementError("A low-integrity root is linked or not a directory.")
    return absolute


def _path_is_same_or_descendant(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath((os.fspath(path), os.fspath(root)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(os.fspath(root))


def _windows_metadata_binding(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_nlink),
    )


def _snapshot_windows_private_tree(root: Path) -> dict[str, tuple[int, int, int, int, int]]:
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or _metadata_is_link_or_reparse(root_metadata):
        raise WriteConfinementError("A private Windows workspace is unsafe.")
    device = int(root_metadata.st_dev)
    snapshot = {"": _windows_metadata_binding(root_metadata)}
    pending: list[tuple[Path, str, int]] = [(root, "", 0)]
    while pending:
        directory, prefix, depth = pending.pop()
        if depth > 64:
            raise WriteConfinementError("A private Windows workspace is too deeply nested.")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: unicodedata.normalize("NFC", item.name).casefold())
        except OSError as exc:
            raise WriteConfinementError("A private Windows workspace could not be scanned safely.") from exc
        collision_keys: set[str] = set()
        for entry in entries:
            collision_key = unicodedata.normalize("NFC", entry.name).casefold()
            if collision_key in collision_keys:
                raise WriteConfinementError("A private Windows workspace contains a name collision.")
            collision_keys.add(collision_key)
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            metadata = Path(entry.path).lstat()
            if _metadata_is_link_or_reparse(metadata) or int(metadata.st_dev) != device:
                raise WriteConfinementError("A private Windows workspace crosses a link or device seam.")
            if stat.S_ISREG(metadata.st_mode):
                if int(metadata.st_nlink) != 1:
                    raise WriteConfinementError("A private Windows workspace contains a hard-linked file.")
            elif stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(entry.path), relative, depth + 1))
            else:
                raise WriteConfinementError("A private Windows workspace contains a non-regular entry.")
            snapshot[relative.replace("\\", "/")] = _windows_metadata_binding(metadata)
            if len(snapshot) > _MAX_WINDOWS_LABELED_ENTRIES:
                raise WriteConfinementError("A private Windows workspace exceeds its entry limit.")
    return dict(sorted(snapshot.items()))


def _set_windows_low_integrity_label(path: Path, *, directory: bool) -> None:
    _set_windows_mandatory_integrity_label(
        path,
        directory=directory,
        integrity_rid=_LOW_INTEGRITY_RID,
    )


def _set_windows_mandatory_integrity_label(
    path: Path,
    *,
    directory: bool,
    integrity_rid: int,
) -> None:
    absolute = Path(path).absolute()
    with _pinned_windows_path_handle(absolute, directory=directory) as handle:
        _set_windows_mandatory_integrity_label_handle(
            handle,
            directory=directory,
            integrity_rid=integrity_rid,
        )


def _set_windows_mandatory_integrity_label_handle(
    handle: int,
    *,
    directory: bool,
    integrity_rid: int,
) -> None:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    if integrity_rid not in {
        _UNTRUSTED_INTEGRITY_RID,
        _LOW_INTEGRITY_RID,
        _MEDIUM_INTEGRITY_RID,
    }:
        raise WriteConfinementError("The Windows mandatory integrity level is unsupported.")
    sid = _windows_string_sid(f"S-1-16-{integrity_rid}")
    try:
        sid_size = int(advapi32.GetLengthSid(sid))
        acl_buffer = ctypes.create_string_buffer(8 + 8 + sid_size)
        acl = ctypes.cast(acl_buffer, ctypes.c_void_p)
        if not advapi32.InitializeAcl(acl, len(acl_buffer), 2):
            raise WriteConfinementUnavailable("Windows could not initialize a mandatory-label ACL.")
        ace_flags = 0x3 if directory else 0
        if not advapi32.AddMandatoryAce(
            acl,
            2,
            ace_flags,
            _SYSTEM_MANDATORY_LABEL_NO_WRITE_UP,
            sid,
        ):
            raise WriteConfinementUnavailable("Windows could not construct a mandatory-label ACE.")
        with _reopened_windows_security_handle(handle, directory=directory, write_owner=True) as security_handle:
            result = int(
                advapi32.SetSecurityInfo(
                    ctypes.c_void_p(security_handle),
                    1,
                    0x10,
                    None,
                    None,
                    None,
                    acl,
                )
            )
        if result:
            raise WriteConfinementUnavailable(
                f"Windows could not label the private workspace at low integrity (error {result})."
            )
    finally:
        kernel32.LocalFree(sid)


def _windows_path_integrity_label(path: Path) -> tuple[int, bool]:
    absolute = Path(path).absolute()
    metadata = absolute.lstat()
    directory = stat.S_ISDIR(metadata.st_mode)
    if (not directory and not stat.S_ISREG(metadata.st_mode)) or _metadata_is_link_or_reparse(metadata):
        raise WriteConfinementError("The Windows integrity-label target is unsafe.")
    with _pinned_windows_path_handle(absolute, directory=directory) as handle:
        return _windows_handle_integrity_label(handle, directory=directory)


def _windows_handle_integrity_label(handle: int, *, directory: bool) -> tuple[int, bool]:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    security_descriptor = ctypes.c_void_p()
    sacl = ctypes.c_void_p()
    with _reopened_windows_security_handle(handle, directory=directory, write_owner=False) as security_handle:
        result = int(
            advapi32.GetSecurityInfo(
                ctypes.c_void_p(security_handle),
                1,
                0x10,
                None,
                None,
                None,
                ctypes.byref(sacl),
                ctypes.byref(security_descriptor),
            )
        )
    if result:
        raise WriteConfinementError(f"Windows could not query the workspace integrity label (error {result}).")
    try:
        if not sacl:
            raise WriteConfinementError("The private Windows workspace has no mandatory label.")
        information = _AclSizeInformation()
        if not advapi32.GetAclInformation(sacl, ctypes.byref(information), ctypes.sizeof(information), 2):
            raise WriteConfinementError("Windows could not inspect the workspace integrity ACL.")
        labels: list[tuple[int, bool]] = []
        for index in range(int(information.AceCount)):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(sacl, index, ctypes.byref(ace)) or not ace.value:
                raise WriteConfinementError("Windows could not inspect a workspace integrity ACE.")
            if ctypes.c_ubyte.from_address(ace.value).value != 0x11:
                continue
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            rid = _windows_sid_integrity_rid(ctypes.c_void_p(ace.value + 8))
            labels.append((rid, bool(mask & _SYSTEM_MANDATORY_LABEL_NO_WRITE_UP)))
        if not labels or len(set(labels)) != 1:
            raise WriteConfinementError("The private Windows workspace has an ambiguous mandatory label.")
        return labels[0]
    finally:
        if security_descriptor:
            kernel32.LocalFree(security_descriptor)


def _grant_windows_path_sid(path: Path, sid: ctypes.c_void_p, *, directory: bool) -> None:
    absolute = Path(path).absolute()
    with _pinned_windows_path_handle(absolute, directory=directory) as handle:
        _grant_windows_handle_sid(handle, sid, directory=directory)


def _grant_windows_handle_sid(handle: int, sid: ctypes.c_void_p, *, directory: bool) -> None:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    descriptor = ctypes.c_void_p()
    old_dacl = ctypes.c_void_p()
    with _reopened_windows_security_handle(handle, directory=directory, write_dac=True) as security_handle:
        result = int(
            advapi32.GetSecurityInfo(
                ctypes.c_void_p(security_handle),
                1,
                0x4,
                None,
                None,
                ctypes.byref(old_dacl),
                None,
                ctypes.byref(descriptor),
            )
        )
    if result:
        raise WriteConfinementUnavailable(f"Windows could not query the private workspace DACL (error {result}).")
    new_dacl = ctypes.c_void_p()
    try:
        # Content/name mutation, traversal, delete, and synchronization only.
        # Deliberately omit WRITE_DAC and WRITE_OWNER.
        access = _ExplicitAccessW()
        access.grfAccessPermissions = 0x001301FF
        access.grfAccessMode = 1
        access.grfInheritance = 0x3 if directory else 0
        access.Trustee.TrusteeForm = 0
        access.Trustee.TrusteeType = 0
        access.Trustee.ptstrName = sid
        result = int(advapi32.SetEntriesInAclW(1, ctypes.byref(access), old_dacl, ctypes.byref(new_dacl)))
        if result or not new_dacl:
            raise WriteConfinementUnavailable(
                f"Windows could not construct the private workspace DACL (error {result})."
            )
        with _reopened_windows_security_handle(handle, directory=directory, write_dac=True) as security_handle:
            result = int(
                advapi32.SetSecurityInfo(
                    ctypes.c_void_p(security_handle),
                    1,
                    0x4,
                    None,
                    None,
                    new_dacl,
                    None,
                )
            )
        if result:
            raise WriteConfinementUnavailable(
                f"Windows could not bind the requested private-tree SID (error {result})."
            )
    finally:
        if new_dacl:
            kernel32.LocalFree(new_dacl)
        if descriptor:
            kernel32.LocalFree(descriptor)


@contextmanager
def _pinned_windows_path_handle(path: Path, *, directory: bool):
    """Pin one exact path without DELETE sharing before handle security APIs."""

    absolute = Path(path).absolute()
    before = absolute.lstat()
    if (
        _metadata_is_link_or_reparse(before)
        or directory is not stat.S_ISDIR(before.st_mode)
        or (not directory and not stat.S_ISREG(before.st_mode))
    ):
        raise WriteConfinementError("A Windows security target is unsafe.")
    kernel32 = _windows_kernel32()
    flags = 0x00200000 | (0x02000000 if directory else 0)
    opened = kernel32.CreateFileW(
        os.fspath(absolute),
        0x80,
        0x00000001 | 0x00000002,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    handle = int(opened or 0)
    if not handle or handle == invalid:
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not pin an exact security target (error {code}).")
    try:
        attributes, file_index = _windows_handle_file_binding(handle)
        if (
            bool(attributes & 0x10) is not directory
            or bool(attributes & 0x400)
            or file_index != int(before.st_ino)
            or _windows_metadata_binding(absolute.lstat()) != _windows_metadata_binding(before)
        ):
            raise WriteConfinementError("A Windows security target changed while it was pinned.")
        yield handle
        if _windows_handle_file_binding(handle) != (attributes, file_index) or _windows_metadata_binding(
            absolute.lstat()
        ) != _windows_metadata_binding(before):
            raise WriteConfinementError("A Windows security target changed while its ACL was accessed.")
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@contextmanager
def _reopened_windows_security_handle(
    handle: int,
    *,
    directory: bool,
    write_owner: bool = False,
    write_dac: bool = False,
):
    """Reopen one exact object with security rights and no DELETE sharing."""

    if type(handle) is not int or handle <= 0:
        raise WriteConfinementError("A Windows security target handle is unavailable.")
    kernel32 = _windows_kernel32()
    before = _windows_handle_file_binding(handle)
    desired_access = 0x00020000  # READ_CONTROL
    if write_owner:
        desired_access |= 0x00080000  # WRITE_OWNER
    if write_dac:
        desired_access |= 0x00040000  # WRITE_DAC
    flags = 0x00200000 | (0x02000000 if directory else 0)
    required = int(kernel32.GetFinalPathNameByHandleW(ctypes.c_void_p(handle), None, 0, 0))
    if required <= 0 or required > 32_768:
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not resolve an exact security target (error {code}).")
    path_buffer = ctypes.create_unicode_buffer(required + 1)
    copied = int(
        kernel32.GetFinalPathNameByHandleW(
            ctypes.c_void_p(handle),
            path_buffer,
            len(path_buffer),
            0,
        )
    )
    if copied <= 0 or copied >= len(path_buffer):
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not resolve an exact security target (error {code}).")
    reopened = kernel32.CreateFileW(
        path_buffer.value,
        desired_access,
        0x00000001 | 0x00000002,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    numeric = int(reopened or 0)
    if not numeric or numeric == invalid:
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not open an exact security target (error {code}).")
    try:
        if _windows_handle_file_binding(numeric) != before:
            raise WriteConfinementError("A Windows security target changed while it was reopened.")
        yield numeric
        if _windows_handle_file_binding(numeric) != before or _windows_handle_file_binding(handle) != before:
            raise WriteConfinementError("A Windows security target changed during ACL access.")
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(numeric))


def _windows_handle_file_binding(handle: int) -> tuple[int, int]:
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
    get_information.restype = wintypes.BOOL
    information = _ByHandleFileInformation()
    if not get_information(wintypes.HANDLE(handle), ctypes.byref(information)):
        raise OSError(ctypes.get_last_error(), "could not inspect an exact Windows security handle")
    file_index = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
    return int(information.dwFileAttributes), file_index


def _create_windows_low_startup_token() -> int:
    """Create the startup-only Low token used before the fixed downgrade."""

    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    current_token = ctypes.c_void_p()
    caller_access = 0x0001 | 0x0002 | 0x0008 | 0x0080 | 0x0100
    startup_access = caller_access | 0x0020  # TOKEN_ADJUST_PRIVILEGES
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), caller_access, ctypes.byref(current_token)):
        raise WriteConfinementUnavailable("Windows could not open the caller token for restriction.")
    low_sid = ctypes.c_void_p()
    startup_token = ctypes.c_void_p()
    try:
        inherited_restricted, _integrity_rid, _no_write_up, inherited_restricted_sids = _windows_token_confinement(
            current_token
        )
        if inherited_restricted:
            if not inherited_restricted_sids:
                raise WriteConfinementUnavailable("Windows returned an unbound inherited restricted-token state.")
            # A restricted token cannot be passed back through
            # CreateRestrictedToken reliably (Windows returns
            # ERROR_INVALID_PARAMETER for the Codex host token). Duplicate it
            # as a primary token instead. This is not a fallback to weaker
            # confinement: the complete restricting-SID set is compared before
            # and after duplication, passed to the fixed bootstrap, and checked
            # again inside the child before any worker source executes.
            if not advapi32.DuplicateTokenEx(
                current_token,
                startup_access,
                None,
                2,
                1,
                ctypes.byref(startup_token),
            ):
                code = ctypes.get_last_error()
                raise WriteConfinementUnavailable(
                    f"Windows could not duplicate the inherited restricted startup token (error {code})."
                )
            duplicated = _windows_token_confinement(startup_token)
            if not duplicated[0] or duplicated[3] != inherited_restricted_sids:
                raise WriteConfinementError("Windows changed inherited restrictions during token duplication.")
        else:
            if inherited_restricted_sids:
                raise WriteConfinementUnavailable("Windows returned restricting SIDs without a restricted token.")
            # Disable removable privileges but do not add synthetic restricting
            # SIDs. The fixed bootstrap lowers the primary token from Low to
            # Untrusted before any worker code.
            if not advapi32.CreateRestrictedToken(
                current_token,
                0x1,
                0,
                None,
                0,
                None,
                0,
                None,
                ctypes.byref(startup_token),
            ):
                code = ctypes.get_last_error()
                raise WriteConfinementUnavailable(f"Windows could not create the Low startup token (error {code}).")
        _remove_windows_token_privileges(startup_token)
        low_sid = _windows_string_sid("S-1-16-4096")
        label = _TokenMandatoryLabel(_SidAndAttributes(low_sid, 0x20))
        label_size = ctypes.sizeof(label) + int(advapi32.GetLengthSid(low_sid))
        if not advapi32.SetTokenInformation(startup_token, 25, ctypes.byref(label), label_size):
            raise WriteConfinementUnavailable("Windows could not set the startup token integrity level.")
        policy = _TokenMandatoryPolicy(_TOKEN_MANDATORY_POLICY_NO_WRITE_UP)
        if not advapi32.SetTokenInformation(
            startup_token,
            27,
            ctypes.byref(policy),
            ctypes.sizeof(policy),
        ):
            code = ctypes.get_last_error()
            # Normal interactive callers do not hold SeTcbPrivilege. Accept
            # only the exact inherited NO_WRITE_UP policy.
            if code != 1314 or not _windows_token_confinement(startup_token)[2]:
                raise WriteConfinementUnavailable(
                    f"Windows could not establish the startup token mandatory policy (error {code})."
                )
        final_restricted, final_rid, final_no_write_up, final_sids = _windows_token_confinement(startup_token)
        if (
            final_rid != _LOW_INTEGRITY_RID
            or not final_no_write_up
            or final_sids != inherited_restricted_sids
            or (inherited_restricted and not final_restricted)
            or not _windows_token_has_only_enabled_traverse_privilege(startup_token)
        ):
            raise WriteConfinementError("Windows changed the startup token restriction boundary.")
        result = int(startup_token.value)
        startup_token = ctypes.c_void_p()
        return result
    finally:
        if startup_token:
            kernel32.CloseHandle(startup_token)
        if low_sid:
            kernel32.LocalFree(low_sid)
        kernel32.CloseHandle(current_token)


def _remove_windows_token_privileges(token: int | ctypes.c_void_p) -> None:
    """Remove every startup-token privilege except enabled path traversal."""

    privileges = _windows_token_privileges(token)
    expected_luid = _windows_privilege_luid("SeChangeNotifyPrivilege")
    removable = tuple(value for value in privileges if value[:2] != expected_luid)
    if not removable:
        if not _is_only_enabled_traverse_privilege(privileges, expected_luid=expected_luid):
            raise WriteConfinementError("Windows startup token lacks its exact traverse-only privilege.")
        return
    count = len(removable)
    buffer_size = _TokenPrivileges.Privileges.offset + ctypes.sizeof(_LuidAndAttributes) * count
    buffer = ctypes.create_string_buffer(buffer_size)
    ctypes.c_uint32.from_buffer(buffer).value = count
    values_address = ctypes.addressof(buffer) + _TokenPrivileges.Privileges.offset
    values = ctypes.cast(values_address, ctypes.POINTER(_LuidAndAttributes * count)).contents
    for target, (low_part, high_part, _attributes) in zip(values, removable, strict=True):
        target.Luid.LowPart = low_part
        target.Luid.HighPart = high_part
        target.Attributes = 0x00000004  # SE_PRIVILEGE_REMOVED
    ctypes.set_last_error(0)
    if not _windows_advapi32().AdjustTokenPrivileges(token, False, buffer, 0, None, None):
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not remove startup-token privileges (error {code}).")
    code = ctypes.get_last_error()
    if code:
        raise WriteConfinementUnavailable(f"Windows did not remove every startup-token privilege (error {code}).")
    retained = _windows_token_privileges(token)
    if any(value[:2] != expected_luid for value in retained):
        raise WriteConfinementError("Windows startup token retained removable privileges.")
    if not _is_only_enabled_traverse_privilege(retained, expected_luid=expected_luid):
        raise WriteConfinementError("Windows startup token changed its exact traverse-only privilege.")


def _windows_token_has_only_enabled_traverse_privilege(token: int | ctypes.c_void_p | None) -> bool:
    return _is_only_enabled_traverse_privilege(_windows_token_privileges(token))


def _is_only_enabled_traverse_privilege(
    privileges: tuple[tuple[int, int, int], ...],
    *,
    expected_luid: tuple[int, int] | None = None,
) -> bool:
    expected = _windows_privilege_luid("SeChangeNotifyPrivilege") if expected_luid is None else expected_luid
    return (
        len(privileges) == 1
        and privileges[0][:2] == expected
        and bool(privileges[0][2] & 0x00000002)
        and not privileges[0][2] & ~0x00000003
    )


def _windows_privilege_luid(name: str) -> tuple[int, int]:
    luid = _Luid()
    if not _windows_advapi32().LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
        code = ctypes.get_last_error()
        raise WriteConfinementUnavailable(f"Windows could not resolve a required privilege identity (error {code}).")
    return int(luid.LowPart), int(luid.HighPart)


def _windows_token_confinement(token: int | ctypes.c_void_p | None) -> tuple[bool, int, bool, tuple[str, ...]]:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    opened = ctypes.c_void_p()
    handle: int | ctypes.c_void_p
    if token is None:
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(opened)):
            raise WriteConfinementError("Windows could not query the current process token.")
        handle = opened
    else:
        handle = token
    try:
        restricted = bool(advapi32.IsTokenRestricted(handle))
        integrity_buffer = _windows_token_information(handle, 25)
        integrity = _TokenMandatoryLabel.from_buffer(integrity_buffer)
        integrity_rid = _windows_sid_integrity_rid(integrity.Label.Sid)
        policy_buffer = _windows_token_information(handle, 27)
        policy = _TokenMandatoryPolicy.from_buffer(policy_buffer)
        restricted_buffer = _windows_token_information(handle, 11)
        if len(restricted_buffer) < ctypes.sizeof(ctypes.c_uint32):
            raise WriteConfinementError("Windows returned malformed restricted-SID evidence.")
        group_count = int(ctypes.c_uint32.from_buffer(restricted_buffer).value)
        if group_count > 4096:
            raise WriteConfinementError("Windows returned excessive restricted-SID evidence.")
        if group_count:
            required = _TokenGroups.Groups.offset + ctypes.sizeof(_SidAndAttributes) * group_count
            if len(restricted_buffer) < required:
                raise WriteConfinementError("Windows returned truncated restricted-SID evidence.")
            groups_address = ctypes.addressof(restricted_buffer) + _TokenGroups.Groups.offset
            groups = ctypes.cast(groups_address, ctypes.POINTER(_SidAndAttributes * group_count)).contents
            hashes = tuple(sorted({_windows_sid_sha256(group.Sid) for group in groups}))
        else:
            hashes = ()
        return (
            restricted,
            integrity_rid,
            bool(int(policy.Policy) & _TOKEN_MANDATORY_POLICY_NO_WRITE_UP),
            hashes,
        )
    finally:
        if opened:
            kernel32.CloseHandle(opened)


def _windows_token_privileges(token: int | ctypes.c_void_p | None) -> tuple[tuple[int, int, int], ...]:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    opened = ctypes.c_void_p()
    handle: int | ctypes.c_void_p
    if token is None:
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(opened)):
            raise WriteConfinementError("Windows could not query current process-token privileges.")
        handle = opened
    else:
        handle = token
    try:
        buffer = _windows_token_information(handle, 3)  # TokenPrivileges
        if len(buffer) < ctypes.sizeof(ctypes.c_uint32):
            raise WriteConfinementError("Windows returned malformed privilege evidence.")
        count = int(ctypes.c_uint32.from_buffer(buffer).value)
        if count > 4096:
            raise WriteConfinementError("Windows returned excessive privilege evidence.")
        required = _TokenPrivileges.Privileges.offset + ctypes.sizeof(_LuidAndAttributes) * count
        if len(buffer) < required:
            raise WriteConfinementError("Windows returned truncated privilege evidence.")
        if not count:
            return ()
        values_address = ctypes.addressof(buffer) + _TokenPrivileges.Privileges.offset
        values = ctypes.cast(values_address, ctypes.POINTER(_LuidAndAttributes * count)).contents
        privileges = tuple(
            sorted((int(value.Luid.LowPart), int(value.Luid.HighPart), int(value.Attributes)) for value in values)
        )
        if len({(low_part, high_part) for low_part, high_part, _attributes in privileges}) != count:
            raise WriteConfinementError("Windows returned duplicate privilege evidence.")
        return privileges
    finally:
        if opened:
            kernel32.CloseHandle(opened)


def _windows_token_information(token: int | ctypes.c_void_p, information_class: int) -> ctypes.Array[ctypes.c_char]:
    advapi32 = _windows_advapi32()
    required = ctypes.c_uint32()
    advapi32.GetTokenInformation(token, information_class, None, 0, ctypes.byref(required))
    if not required.value:
        raise WriteConfinementError("Windows token evidence is unavailable.")
    buffer = ctypes.create_string_buffer(required.value)
    if not advapi32.GetTokenInformation(
        token,
        information_class,
        buffer,
        len(buffer),
        ctypes.byref(required),
    ):
        raise WriteConfinementError("Windows token evidence could not be read.")
    return buffer


def _windows_string_sid(value: str) -> ctypes.c_void_p:
    sid = ctypes.c_void_p()
    if not _windows_advapi32().ConvertStringSidToSidW(value, ctypes.byref(sid)) or not sid:
        raise WriteConfinementUnavailable("Windows could not allocate a security identifier.")
    return sid


def _windows_sid_sha256(sid: int | ctypes.c_void_p) -> str:
    advapi32 = _windows_advapi32()
    if not sid or not advapi32.IsValidSid(sid):
        raise WriteConfinementError("Windows returned an invalid security identifier.")
    length = int(advapi32.GetLengthSid(sid))
    if not 8 <= length <= 68:
        raise WriteConfinementError("Windows returned an invalid security identifier length.")
    return hashlib.sha256(ctypes.string_at(sid, length)).hexdigest()


def _windows_sid_integrity_rid(sid: int | ctypes.c_void_p) -> int:
    advapi32 = _windows_advapi32()
    if not sid or not advapi32.IsValidSid(sid):
        raise WriteConfinementError("Windows returned an invalid integrity security identifier.")
    count_pointer = advapi32.GetSidSubAuthorityCount(sid)
    if not count_pointer or not count_pointer.contents.value:
        raise WriteConfinementError("Windows returned an invalid integrity security identifier.")
    rid_pointer = advapi32.GetSidSubAuthority(sid, int(count_pointer.contents.value) - 1)
    if not rid_pointer:
        raise WriteConfinementError("Windows returned an invalid integrity security identifier.")
    return int(rid_pointer.contents.value)


def _exclusive_windows_stdio_file(
    directory: Path,
    label: str,
    payload: bytes,
    *,
    integrity_rid: int,
) -> int:
    from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity

    name = f".restricted-{label}-{uuid.uuid4().hex}.bin"
    descriptor = -1
    try:
        with AnchoredDirectory(directory, directory) as anchor:
            descriptor = anchor.open_file_immovable(
                name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | int(getattr(os, "O_BINARY", 0)),
                0o600,
            )
            identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
            if not identity.matches(anchor.lstat(name)):
                raise WriteConfinementError("A restricted child stdio file changed during creation.")
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("unable to preload restricted child stdio")
                written += count
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or int(metadata.st_nlink) != 1
                or not identity.matches(metadata)
                or not identity.matches(anchor.lstat(name))
            ):
                raise WriteConfinementError("A restricted child stdio file is unsafe.")
            os_handle = _windows_os_handle(descriptor)
            _set_windows_mandatory_integrity_label_handle(
                os_handle,
                directory=False,
                integrity_rid=integrity_rid,
            )
            if _windows_handle_integrity_label(os_handle, directory=False) != (integrity_rid, True):
                raise WriteConfinementError("A confined child stdio file lacks its exact integrity label.")
            if not identity.matches(anchor.lstat(name)) or not identity.matches(os.fstat(descriptor)):
                raise WriteConfinementError("A restricted child stdio path changed during labeling.")
            anchor.verify()
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _read_windows_stdio_descriptor(descriptor: int, maximum: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _windows_os_handle(descriptor: int) -> int:
    import msvcrt

    handle = int(msvcrt.get_osfhandle(descriptor))
    if handle == -1:
        raise OSError("restricted child stdio handle is unavailable")
    return handle


def _windows_environment_block(environment: Mapping[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    rows: list[str] = []
    seen: set[str] = set()
    for raw_name, raw_value in sorted(environment.items(), key=lambda item: str(item[0]).casefold()):
        name = str(raw_name)
        value = str(raw_value)
        folded = name.casefold()
        if not name or "=" in name or "\x00" in name or "\x00" in value or folded in seen:
            raise WriteConfinementError("The restricted child environment is invalid.")
        seen.add(folded)
        rows.append(f"{name}={value}")
    return ctypes.create_unicode_buffer("\x00".join(rows) + "\x00\x00")


def _create_windows_private_desktop() -> tuple[int, str, str]:
    advapi32 = _windows_advapi32()
    kernel32 = _windows_kernel32()
    user32 = _windows_user32()
    current_token = ctypes.c_void_p()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(current_token)):
        raise WriteConfinementUnavailable("Windows could not query the caller SID for a private desktop.")
    user_sid_string = ctypes.c_void_p()
    security_descriptor = ctypes.c_void_p()
    desktop_handle = 0
    try:
        user_buffer = _windows_token_information(current_token, 1)
        user = _TokenUser.from_buffer(user_buffer)
        if not advapi32.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(user_sid_string)):
            raise WriteConfinementUnavailable("Windows could not bind the caller SID to a private desktop.")
        caller_sid = ctypes.wstring_at(user_sid_string)
        # The startup token retains the exact caller SID. The private desktop
        # is Low only for interpreter initialization; the worker receives no
        # desktop handle and begins after the primary token is Untrusted.
        sddl = f"D:P(A;;GA;;;{caller_sid})S:(ML;;NW;;;LW)"
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl,
            1,
            ctypes.byref(security_descriptor),
            None,
        ):
            raise WriteConfinementUnavailable("Windows could not construct a private desktop security descriptor.")
        attributes = _SecurityAttributes(
            nLength=ctypes.sizeof(_SecurityAttributes),
            lpSecurityDescriptor=security_descriptor,
            bInheritHandle=False,
        )
        desktop_component = f"spritelab-{uuid.uuid4().hex}"
        desktop_handle = int(
            user32.CreateDesktopW(
                desktop_component,
                None,
                None,
                0,
                0x000F01FF,
                ctypes.byref(attributes),
            )
            or 0
        )
        if not desktop_handle:
            code = ctypes.get_last_error()
            raise WriteConfinementUnavailable(f"Windows could not create a private child desktop (error {code}).")
        station = _windows_user_object_name(user32.GetProcessWindowStation())
        actual_desktop = _windows_user_object_name(desktop_handle)
        if actual_desktop != desktop_component or not station or "\\" in station:
            raise WriteConfinementError("The private child desktop identity is invalid.")
        desktop_name = f"{station}\\{desktop_component}"
        identity_payload = json.dumps(
            {"station": station.casefold(), "desktop": desktop_component},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        identity = hashlib.sha256(identity_payload).hexdigest()
        result = desktop_handle, desktop_name, identity
        desktop_handle = 0
        return result
    finally:
        if desktop_handle:
            user32.CloseDesktop(desktop_handle)
        if security_descriptor:
            kernel32.LocalFree(security_descriptor)
        if user_sid_string:
            kernel32.LocalFree(user_sid_string)
        kernel32.CloseHandle(current_token)


def _windows_user_object_name(handle: int | ctypes.c_void_p) -> str:
    user32 = _windows_user32()
    required = ctypes.c_uint32()
    user32.GetUserObjectInformationW(handle, 2, None, 0, ctypes.byref(required))
    if required.value < 2 or required.value > 1024:
        raise WriteConfinementError("A Windows user-object name is unavailable.")
    buffer = ctypes.create_unicode_buffer((required.value + 1) // 2)
    if not user32.GetUserObjectInformationW(
        handle,
        2,
        buffer,
        ctypes.sizeof(buffer),
        ctypes.byref(required),
    ):
        raise WriteConfinementError("A Windows user-object name could not be queried.")
    return buffer.value


def _windows_process_exit_code(kernel32: Any, handle: int) -> int:
    exit_code = ctypes.c_uint32()
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
        raise OSError("unable to read the restricted child exit code")
    return int(exit_code.value)


def _windows_monotonic() -> float:
    return time.monotonic()


def _windows_kernel32() -> Any:
    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows security APIs are unavailable on this platform.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.ReOpenFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    kernel32.ReOpenFile.restype = ctypes.c_void_p
    kernel32.GetFinalPathNameByHandleW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    kernel32.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.InitializeProcThreadAttributeList.restype = ctypes.c_int
    kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.UpdateProcThreadAttribute.restype = ctypes.c_int
    kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    kernel32.DeleteProcThreadAttributeList.restype = None
    return kernel32


def _windows_advapi32() -> Any:
    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows security APIs are unavailable on this platform.")
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.OpenProcessToken.restype = ctypes.c_int
    advapi32.GetTokenInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    advapi32.GetTokenInformation.restype = ctypes.c_int
    advapi32.CreateRestrictedToken.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.CreateRestrictedToken.restype = ctypes.c_int
    advapi32.DuplicateTokenEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.DuplicateTokenEx.restype = ctypes.c_int
    advapi32.AdjustTokenPrivileges.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.AdjustTokenPrivileges.restype = ctypes.c_int
    advapi32.LookupPrivilegeValueW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.POINTER(_Luid)]
    advapi32.LookupPrivilegeValueW.restype = ctypes.c_int
    advapi32.SetTokenInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    advapi32.SetTokenInformation.restype = ctypes.c_int
    advapi32.IsTokenRestricted.argtypes = [ctypes.c_void_p]
    advapi32.IsTokenRestricted.restype = ctypes.c_int
    advapi32.ConvertStringSidToSidW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.ConvertStringSidToSidW.restype = ctypes.c_int
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.ConvertSidToStringSidW.restype = ctypes.c_int
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = ctypes.c_int
    advapi32.IsValidSid.argtypes = [ctypes.c_void_p]
    advapi32.IsValidSid.restype = ctypes.c_int
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = ctypes.c_uint32
    advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
    advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    advapi32.GetSidSubAuthority.restype = ctypes.POINTER(ctypes.c_uint32)
    advapi32.InitializeAcl.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    advapi32.InitializeAcl.restype = ctypes.c_int
    advapi32.AddMandatoryAce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    advapi32.AddMandatoryAce.restype = ctypes.c_int
    advapi32.GetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = ctypes.c_uint32
    advapi32.SetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = ctypes.c_uint32
    advapi32.GetSecurityInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = ctypes.c_uint32
    advapi32.SetSecurityInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.SetSecurityInfo.restype = ctypes.c_uint32
    advapi32.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
    advapi32.GetAclInformation.restype = ctypes.c_int
    advapi32.GetAce.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetAce.restype = ctypes.c_int
    advapi32.SetEntriesInAclW.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_ExplicitAccessW),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.SetEntriesInAclW.restype = ctypes.c_uint32
    advapi32.CreateProcessAsUserW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessInformation),
    ]
    advapi32.CreateProcessAsUserW.restype = ctypes.c_int
    return advapi32


def _windows_user32() -> Any:
    if sys.platform != "win32" or os.name != "nt":
        raise WriteConfinementUnavailable("Windows desktop APIs are unavailable on this platform.")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateDesktopW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_SecurityAttributes),
    ]
    user32.CreateDesktopW.restype = ctypes.c_void_p
    user32.CloseDesktop.argtypes = [ctypes.c_void_p]
    user32.CloseDesktop.restype = ctypes.c_int
    user32.GetProcessWindowStation.argtypes = []
    user32.GetProcessWindowStation.restype = ctypes.c_void_p
    user32.GetUserObjectInformationW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    user32.GetUserObjectInformationW.restype = ctypes.c_int
    return user32


def enforce_linux_landlock_write_confinement(
    root: str | Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> WriteConfinementEvidence:
    """Restrict this process to writes below one exact workspace inode.

    The function is intentionally process-global and irreversible.  Call it in
    a freshly spawned controlled child before importing the legacy writer.
    Reads and execution remain available so Python and Pillow can import their
    already trusted code. Content/name writes outside ``root`` are denied; the
    caller must separately prove that its code closure contains none of
    Landlock's documented unmediated metadata operations.
    """

    if not sys.platform.startswith("linux"):
        raise WriteConfinementUnavailable("Linux Landlock write confinement is unavailable on this platform.")
    if type(expected_device) is not int or type(expected_inode) is not int:
        raise WriteConfinementError("The expected workspace identity is malformed.")
    if expected_device < 0 or expected_inode <= 0:
        raise WriteConfinementError("The expected workspace identity is invalid.")

    raw = os.fspath(root)
    if not raw or not raw.strip() or raw.strip() in {".", ".."}:
        raise WriteConfinementError("The write-confinement root must be an explicit directory.")
    absolute = Path(os.path.abspath(os.path.expanduser(raw)))
    before = absolute.lstat()
    if not stat.S_ISDIR(before.st_mode) or _metadata_is_link_or_reparse(before):
        raise WriteConfinementError("The write-confinement root is linked or not a directory.")
    open_flags = (
        int(getattr(os, "O_PATH", os.O_RDONLY))
        | int(getattr(os, "O_DIRECTORY", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
    )
    root_fd = os.open(absolute, open_flags)
    try:
        opened = os.fstat(root_fd)
        _require_same_directory(before, opened)
        _require_same_directory(before, absolute.lstat())
        identity = DirectoryIdentity.from_stat(opened)
        if identity.device != expected_device or identity.inode != expected_inode:
            raise WriteConfinementError("The write-confinement root identity changed before restriction.")
        return enforce_linux_landlock_write_confinement_fd(
            root_fd,
            expected_device=identity.device,
            expected_inode=identity.inode,
        )
    finally:
        os.close(root_fd)


def enforce_linux_landlock_write_confinement_fd(
    root_fd: int,
    *,
    expected_device: int,
    expected_inode: int,
) -> WriteConfinementEvidence:
    """Restrict writes below one already-opened inherited directory handle.

    This is the preferred child API when the parent opens the no-follow
    directory before spawning and transfers it with ``pass_fds``.  The child
    never resolves the pathname again, so a rename/replacement race cannot
    switch the inode to which the kernel rule is bound.  Ownership of
    ``root_fd`` remains with the caller.
    """

    return enforce_linux_landlock_write_confinement_fds(((root_fd, expected_device, expected_inode),))


def enforce_linux_landlock_write_confinement_fds(
    anchors: Sequence[tuple[int, int, int]],
) -> WriteConfinementEvidence:
    """Restrict writes below exact inherited directory handles.

    Each tuple is ``(fd, expected_device, expected_inode)``.  All descriptors
    are verified before the irreversible restriction, and no pathname is
    opened or trusted.  This multi-root form supports workers whose output and
    private temporary directories are intentionally distinct.
    """

    if not sys.platform.startswith("linux"):
        raise WriteConfinementUnavailable("Linux Landlock write confinement is unavailable on this platform.")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        raise WriteConfinementError("The inherited write-confinement anchors are malformed.")
    if not anchors:
        raise WriteConfinementError("At least one inherited write-confinement anchor is required.")

    verified: list[tuple[int, DirectoryIdentity]] = []
    seen_descriptors: set[int] = set()
    seen_identities: set[tuple[int, int]] = set()
    for anchor in anchors:
        if not isinstance(anchor, tuple) or len(anchor) != 3:
            raise WriteConfinementError("An inherited write-confinement anchor is malformed.")
        root_fd, expected_device, expected_inode = anchor
        if type(root_fd) is not int or root_fd < 0:
            raise WriteConfinementError("An inherited write-confinement descriptor is invalid.")
        if type(expected_device) is not int or type(expected_inode) is not int:
            raise WriteConfinementError("An expected inherited workspace identity is malformed.")
        if expected_device < 0 or expected_inode <= 0:
            raise WriteConfinementError("An expected inherited workspace identity is invalid.")
        if root_fd in seen_descriptors:
            raise WriteConfinementError("Inherited write-confinement descriptors must be unique.")
        try:
            identity = DirectoryIdentity.from_stat(os.fstat(root_fd))
        except OSError as exc:
            raise WriteConfinementError("An inherited write-confinement descriptor is unavailable.") from exc
        if identity.device != expected_device or identity.inode != expected_inode:
            raise WriteConfinementError("An inherited write-confinement directory identity does not match.")
        identity_key = (identity.device, identity.inode)
        if identity_key in seen_identities:
            raise WriteConfinementError("Inherited write-confinement directories must be unique.")
        seen_descriptors.add(root_fd)
        seen_identities.add(identity_key)
        verified.append((root_fd, identity))

    syscall_numbers = _landlock_syscall_numbers()
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    abi = int(
        syscall(
            syscall_numbers[0],
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if abi < 0:
        _raise_landlock_unavailable("query Landlock ABI")
    if abi < _MINIMUM_LANDLOCK_ABI:
        raise WriteConfinementUnavailable(
            f"Linux Landlock ABI {abi} cannot mediate file truncation; ABI {_MINIMUM_LANDLOCK_ABI}+ is required."
        )
    if abi > _MAXIMUM_AUDITED_LANDLOCK_ABI:
        raise WriteConfinementUnavailable(
            f"Linux Landlock ABI {abi} is newer than audited ABI {_MAXIMUM_AUDITED_LANDLOCK_ABI}."
        )

    handled_access = _BASE_HANDLED_ACCESS
    if abi >= 5:
        # Deny device ioctls everywhere.  The controlled workers do not need
        # them, and no workspace rule grants this handled right.
        handled_access |= _ACCESS_FS_IOCTL_DEV
    if abi >= 9:
        # Pathname UNIX socket resolution is outside controlled worker needs.
        handled_access |= _ACCESS_FS_RESOLVE_UNIX

    ruleset_fd = -1
    try:
        ruleset_attr = _RulesetAttr(handled_access_fs=handled_access)
        ruleset_fd = int(
            syscall(
                syscall_numbers[0],
                ctypes.byref(ruleset_attr),
                ctypes.c_size_t(ctypes.sizeof(ruleset_attr)),
                ctypes.c_uint(0),
            )
        )
        if ruleset_fd < 0:
            _raise_landlock_error("create a Landlock ruleset")
        os.set_inheritable(ruleset_fd, False)

        for root_fd, identity in verified:
            # Re-verify every held handle immediately before binding its rule.
            current = DirectoryIdentity.from_stat(os.fstat(root_fd))
            if current != identity:
                raise WriteConfinementError("An inherited write-confinement directory identity changed.")
            path_rule = _PathBeneathAttr(
                allowed_access=_ALLOWED_WORKSPACE_ACCESS,
                parent_fd=root_fd,
            )
            result = int(
                syscall(
                    syscall_numbers[1],
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(path_rule),
                    ctypes.c_uint(0),
                )
            )
            if result < 0:
                _raise_landlock_error("add an inherited workspace Landlock rule")

        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _raise_landlock_error("set no_new_privs")
        result = int(
            syscall(
                syscall_numbers[2],
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint(0),
            )
        )
        if result < 0:
            _raise_landlock_error("restrict the child process")
    finally:
        if ruleset_fd >= 0:
            os.close(ruleset_fd)

    identities = [identity for _descriptor, identity in verified]
    if len(identities) == 1:
        root_identity_sha256 = identities[0].identity_sha256
    else:
        identity_payload = [{"device": identity.device, "inode": identity.inode} for identity in identities]
        root_identity_sha256 = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    return WriteConfinementEvidence(
        strategy=LINUX_LANDLOCK_STRATEGY,
        platform="linux",
        kernel_abi=abi,
        root_identity_sha256=root_identity_sha256,
        handled_access_fs=handled_access,
        allowed_access_fs=_ALLOWED_WORKSPACE_ACCESS,
        no_new_privileges=True,
    )


def _landlock_syscall_numbers() -> tuple[int, int, int]:
    machine = platform.machine().casefold()
    if machine not in {"x86_64", "amd64", "aarch64", "arm64", "riscv64"}:
        raise WriteConfinementUnavailable(
            f"Linux Landlock syscall numbers are not audited for architecture {machine or 'unknown'}."
        )
    # Linux allocated the Landlock calls from the common syscall-number range
    # on every architecture supported above.
    return 444, 445, 446


def _raise_landlock_unavailable(action: str) -> None:
    code = ctypes.get_errno()
    if code in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        raise WriteConfinementUnavailable(f"The kernel could not {action} safely (errno {code}).")
    raise WriteConfinementError(f"The kernel failed to {action} safely (errno {code}).")


def _raise_landlock_error(action: str) -> None:
    code = ctypes.get_errno()
    if code in {errno.ENOSYS, errno.EOPNOTSUPP}:
        raise WriteConfinementUnavailable(f"The kernel could not {action} safely (errno {code}).")
    raise WriteConfinementError(f"The kernel failed to {action} safely (errno {code}).")


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag)


def _require_same_directory(before: os.stat_result, after: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise WriteConfinementError("The write-confinement root changed while it was opened.")


__all__ = [
    "LINUX_LANDLOCK_STRATEGY",
    "WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY",
    "WINDOWS_LOW_INTEGRITY_STRATEGY",
    "WINDOWS_PARENT_ANCHORS_STRATEGY",
    "WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256",
    "WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE",
    "DirectoryIdentity",
    "WriteConfinementError",
    "WriteConfinementEvidence",
    "WriteConfinementUnavailable",
    "create_windows_bootstrap_untrusted_process",
    "directory_identity",
    "enforce_linux_landlock_write_confinement",
    "enforce_linux_landlock_write_confinement_fd",
    "enforce_linux_landlock_write_confinement_fds",
    "prepare_windows_low_integrity_roots",
    "prepare_windows_low_integrity_workspace",
    "prepare_windows_untrusted_integrity_roots",
    "prepare_windows_untrusted_integrity_workspace",
    "windows_current_process_confinement_evidence",
    "write_confinement_strategy",
]

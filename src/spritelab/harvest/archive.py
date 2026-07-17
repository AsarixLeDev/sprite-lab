"""Bounded, transactional archive extraction for harvested packs."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
import time
import unicodedata
import uuid
import warnings
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"
APPLESINGLE_MAGIC = b"\x00\x05\x16\x00"

DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000
DEFAULT_MAX_MEMBER_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_BYTES = DEFAULT_MAX_TOTAL_BYTES + 64 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
_COPY_CHUNK_BYTES = 1 << 20
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class ArchiveSecurityError(ValueError):
    """Raised when an archive cannot be extracted without ambiguity or risk."""


class ArchiveCancelled(RuntimeError):
    """Raised when bounded extraction is cancelled or exceeds its deadline."""


class ArchiveRecoveryResidueWarning(RuntimeWarning):
    """A verified archive committed while an exact previous tree was retained."""


@dataclass(frozen=True)
class _Limits:
    max_archive_bytes: int
    max_members: int
    max_member_bytes: int
    max_total_bytes: int
    max_compression_ratio: float


@dataclass(frozen=True)
class _ValidatedMember:
    raw: Any
    name: str
    collision_key: str
    is_dir: bool
    size: int
    selected: bool
    resource_fork: bool


class ArchiveSnapshot(AbstractContextManager["ArchiveSnapshot"]):
    """One private immutable byte snapshot bound to a stable source descriptor."""

    def __init__(
        self,
        source_path: Path,
        handle: BinaryIO,
        *,
        source_anchor: AnchoredDirectory,
        source_descriptor: int,
        source_metadata: os.stat_result,
        snapshot_metadata: os.stat_result,
        snapshot_residue_path: Path | None,
        cancel_requested: Callable[[], bool] | None,
        deadline_monotonic: float | None,
        byte_count: int,
        sha256: str,
    ) -> None:
        self.source_path = source_path
        self._handle = handle
        self._source_anchor: AnchoredDirectory | None = source_anchor
        self._source_descriptor = source_descriptor
        self._source_metadata = source_metadata
        self._snapshot_metadata = snapshot_metadata
        self.snapshot_residue_path = snapshot_residue_path
        self._cancel_requested = cancel_requested
        self._deadline_monotonic = deadline_monotonic
        self.byte_count = byte_count
        self.sha256 = sha256

    @classmethod
    def open(
        cls,
        archive_path: str | Path,
        *,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        expected_sha256: str | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
        source_anchor: AnchoredDirectory | None = None,
    ) -> ArchiveSnapshot:
        """Copy one verified source descriptor into a private temporary file."""

        if source_anchor is None:
            source_path = _validated_archive_path(archive_path)
            with AnchoredDirectory(source_path.parent, source_path.parent) as trusted_parent:
                return cls.open(
                    source_path,
                    max_archive_bytes=max_archive_bytes,
                    expected_sha256=expected_sha256,
                    cancel_requested=cancel_requested,
                    deadline_monotonic=deadline_monotonic,
                    source_anchor=trusted_parent,
                )
        source_anchor.verify()
        source_path = Path(os.path.abspath(os.path.expanduser(os.fspath(archive_path))))
        if source_path.parent != source_anchor.directory or Path(source_path.name).name != source_path.name:
            raise ArchiveSecurityError("archive source does not belong to the supplied anchored parent")
        before = source_anchor.lstat(source_path.name)
        _require_same_regular_file(before, before, "archive source is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = source_anchor.open_file(source_path.name, flags)
        retained_source_anchor: AnchoredDirectory | None = source_anchor.detached_duplicate()
        snapshot: BinaryIO | None = None
        snapshot_name: str | None = None
        snapshot_identity: OwnedFileIdentity | None = None
        snapshot_residue_path: Path | None = None
        failure: BaseException | None = None
        try:
            opened = os.fstat(descriptor)
            _require_same_regular_file(before, opened, "archive changed while opening")
            try:
                snapshot_descriptor = source_anchor.open_anonymous_file()
            except (OSError, UnsafeFilesystemOperation):
                snapshot_name = f".spritelab-archive-snapshot-{uuid.uuid4().hex}.tmp"
                snapshot_descriptor = source_anchor.open_file(
                    snapshot_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                )
            snapshot_identity = OwnedFileIdentity.from_stat(os.fstat(snapshot_descriptor))
            snapshot = os.fdopen(snapshot_descriptor, "w+b")
            digest = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                while True:
                    _check_archive_abort(cancel_requested, deadline_monotonic)
                    chunk = source.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_archive_bytes:
                        raise ArchiveSecurityError(f"archive exceeds the {max_archive_bytes}-byte input limit")
                    snapshot.write(chunk)
                    digest.update(chunk)
            snapshot.flush()
            os.fsync(snapshot.fileno())
            source_after = os.fstat(descriptor)
            path_after = source_anchor.lstat(source_path.name)
            _require_same_regular_file(before, source_after, "archive changed while snapshotting")
            _require_same_regular_file(before, path_after, "archive path changed while snapshotting")
            if total != before.st_size:
                raise ArchiveSecurityError("archive size changed while snapshotting")
            actual_digest = digest.hexdigest()
            expected_digest = _normalized_sha256(expected_sha256)
            if expected_digest is not None and actual_digest != expected_digest:
                raise ArchiveSecurityError(f"archive SHA256 mismatch: expected {expected_digest}, got {actual_digest}")
            if snapshot_name is not None and os.name != "nt":
                os.fchmod(snapshot.fileno(), 0o400)
                if stat.S_IMODE(os.fstat(snapshot.fileno()).st_mode) != 0o400:
                    raise ArchiveSecurityError("archive snapshot evidence could not be made read-only")
                evidence_name = f".spritelab-archive-snapshot-evidence-{actual_digest}-{uuid.uuid4().hex}.bin"
                source_anchor.rename(snapshot_name, evidence_name, replace=False)
                snapshot_name = evidence_name
                snapshot_residue_path = source_anchor.directory / evidence_name
            snapshot.seek(0)
            writable_snapshot_metadata = os.fstat(snapshot.fileno())
            if snapshot_name is None:
                snapshot_read_descriptor = _reopen_anonymous_snapshot_read_only(snapshot.fileno())
            else:
                snapshot_read_descriptor = source_anchor.open_file(
                    snapshot_name,
                    os.O_RDONLY | getattr(os, "O_BINARY", 0),
                )
            snapshot_metadata = os.fstat(snapshot_read_descriptor)
            if (
                snapshot_identity is None
                or not snapshot_identity.matches(snapshot_metadata)
                or snapshot_metadata.st_size != writable_snapshot_metadata.st_size
            ):
                os.close(snapshot_read_descriptor)
                raise ArchiveSecurityError("archive snapshot changed while reopening read-only")
            snapshot.close()
            snapshot = os.fdopen(snapshot_read_descriptor, "rb")
            if not stat.S_ISREG(snapshot_metadata.st_mode) or snapshot_metadata.st_size != total:
                raise ArchiveSecurityError("archive snapshot is not a stable regular file")
            if snapshot_name is not None and os.name == "nt":
                if not source_anchor.unlink_if_owned(snapshot_name, snapshot_identity, missing_ok=False):
                    raise ArchiveSecurityError("archive snapshot temporary path changed before retirement")
                snapshot_name = None
            elif snapshot_residue_path is not None:
                snapshot_name = None
            if retained_source_anchor is None:
                raise ArchiveSecurityError("archive source parent anchor was lost during snapshotting")
            result = cls(
                source_path,
                snapshot,
                source_anchor=retained_source_anchor,
                source_descriptor=descriptor,
                source_metadata=before,
                snapshot_metadata=snapshot_metadata,
                snapshot_residue_path=snapshot_residue_path,
                cancel_requested=cancel_requested,
                deadline_monotonic=deadline_monotonic,
                byte_count=total,
                sha256=actual_digest,
            )
            descriptor = -1
            retained_source_anchor = None
            return result
        except BaseException as exc:
            failure = exc
            if snapshot is not None:
                snapshot.close()
            if snapshot_name is not None and snapshot_identity is not None:
                source_anchor.quarantine_if_owned(
                    snapshot_name,
                    snapshot_identity,
                    prefix=".spritelab-archive-snapshot-failed-",
                )
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if retained_source_anchor is not None:
                retained_source_anchor.__exit__(
                    type(failure) if failure is not None else None,
                    failure,
                    failure.__traceback__ if failure is not None else None,
                )

    @property
    def suffixes(self) -> str:
        return "".join(self.source_path.suffixes).lower()

    def rewind(self) -> BinaryIO:
        self._assert_snapshot_unchanged()
        self._handle.seek(0)
        return self._handle

    def verify_final(self) -> None:
        """Rehash the same private snapshot and validate the source path entry."""

        self._assert_snapshot_unchanged()
        digest = hashlib.sha256()
        self._handle.seek(0)
        _check_archive_abort(self._cancel_requested, self._deadline_monotonic)
        while chunk := self._handle.read(_COPY_CHUNK_BYTES):
            _check_archive_abort(self._cancel_requested, self._deadline_monotonic)
            digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ArchiveSecurityError("archive snapshot changed while it was in use")
        self._assert_snapshot_unchanged()
        source_before = os.fstat(self._source_descriptor)
        _require_same_regular_file(
            self._source_metadata,
            source_before,
            "archive raw source descriptor changed while snapshot was in use",
        )
        source_digest = hashlib.sha256()
        os.lseek(self._source_descriptor, 0, os.SEEK_SET)
        _check_archive_abort(self._cancel_requested, self._deadline_monotonic)
        while chunk := os.read(self._source_descriptor, _COPY_CHUNK_BYTES):
            _check_archive_abort(self._cancel_requested, self._deadline_monotonic)
            source_digest.update(chunk)
        source_after = os.fstat(self._source_descriptor)
        _require_same_regular_file(
            self._source_metadata,
            source_after,
            "archive raw source descriptor changed while being reverified",
        )
        if source_digest.hexdigest() != self.sha256:
            raise ArchiveSecurityError("archive raw source bytes changed while snapshot was in use")
        _check_archive_abort(self._cancel_requested, self._deadline_monotonic)
        source_anchor = self._source_anchor
        if source_anchor is None:
            raise ArchiveSecurityError("archive source parent anchor is no longer available")
        try:
            path_after = source_anchor.lstat(self.source_path.name)
        except FileNotFoundError as exc:
            raise ArchiveSecurityError("archive source path disappeared while snapshot was in use") from exc
        _require_same_regular_file(
            self._source_metadata,
            path_after,
            "archive source path changed while snapshot was in use",
        )

    def _assert_snapshot_unchanged(self) -> None:
        current = os.fstat(self._handle.fileno())
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != self._snapshot_metadata.st_dev
            or current.st_ino != self._snapshot_metadata.st_ino
            or current.st_size != self._snapshot_metadata.st_size
            or current.st_mtime_ns != self._snapshot_metadata.st_mtime_ns
        ):
            raise ArchiveSecurityError("archive snapshot descriptor changed while in use")

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            try:
                if self._source_descriptor >= 0:
                    os.close(self._source_descriptor)
                    self._source_descriptor = -1
            finally:
                if self._source_anchor is not None:
                    self._source_anchor.__exit__(None, None, None)
                    self._source_anchor = None


def extract_archive(
    archive_path: str | Path | ArchiveSnapshot,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    include_member_globs: Sequence[str] = (),
    exclude_member_globs: Sequence[str] = (),
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    expected_sha256: str | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    deadline_monotonic: float | None = None,
    destination_parent_anchor: AnchoredDirectory | None = None,
) -> Path:
    """Validate and extract an archive into a staged tree, then publish it.

    Unsafe, ambiguous, linked, special, encrypted, colliding, or over-limit
    members reject the entire archive. The destination is not changed until
    every selected byte has been extracted successfully.
    """

    _check_archive_abort(cancel_requested, deadline_monotonic)
    limits = _validated_limits(
        max_archive_bytes=max_archive_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    owns_snapshot = not isinstance(archive_path, ArchiveSnapshot)
    snapshot = (
        ArchiveSnapshot.open(
            archive_path,
            max_archive_bytes=limits.max_archive_bytes,
            expected_sha256=expected_sha256,
            cancel_requested=cancel_requested,
            deadline_monotonic=deadline_monotonic,
        )
        if owns_snapshot
        else archive_path
    )
    if snapshot.byte_count > limits.max_archive_bytes:
        if owns_snapshot:
            snapshot.close()
        raise ArchiveSecurityError(f"archive exceeds the {limits.max_archive_bytes}-byte input limit")
    expected_digest = _normalized_sha256(expected_sha256)
    if expected_digest is not None and snapshot.sha256 != expected_digest:
        if owns_snapshot:
            snapshot.close()
        raise ArchiveSecurityError(f"archive SHA256 mismatch: expected {expected_digest}, got {snapshot.sha256}")
    suffixes = snapshot.suffixes
    is_zip = snapshot.source_path.suffix.lower() == ".zip"
    is_tar = suffixes.endswith((".tar", ".tar.gz", ".tgz"))
    if not is_zip and not is_tar:
        if owns_snapshot:
            snapshot.close()
        raise ValueError(f"unsupported archive type: {snapshot.source_path.name}")
    requested_output_dir = Path(output_dir)
    try:
        parent_context = (
            _anchored_destination_parent(output_dir)
            if destination_parent_anchor is None
            else _supplied_destination_parent(output_dir, destination_parent_anchor)
        )
        with parent_context as (output_dir, output_parent):
            _preflight_destination(output_parent, output_dir.name, overwrite=overwrite)
            _reject_source_output_overlap(snapshot.source_path, output_dir)
            staging_name, _staging_identity = output_parent.mkdir_unique(f".{output_dir.name}.extract-")
            try:
                with output_parent.open_directory(staging_name) as staging:
                    if is_zip:
                        _extract_zip(
                            snapshot,
                            staging,
                            include_member_globs,
                            exclude_member_globs,
                            limits,
                            cancel_requested,
                            progress,
                            deadline_monotonic,
                        )
                    else:
                        _extract_tar(
                            snapshot,
                            staging,
                            include_member_globs,
                            exclude_member_globs,
                            limits,
                            cancel_requested,
                            progress,
                            deadline_monotonic,
                        )
                    _check_archive_abort(cancel_requested, deadline_monotonic)
                    snapshot.verify_final()
                _publish_staged_tree(
                    output_parent,
                    staging_name,
                    output_dir.name,
                    overwrite=overwrite,
                )
                staging_name = ""
            finally:
                if staging_name:
                    _quarantine_failed_staging(output_parent, staging_name, output_dir.name)
    except UnsafeFilesystemOperation as exc:
        raise ArchiveSecurityError("destination crosses a linked or non-directory ancestor") from exc
    finally:
        if owns_snapshot:
            snapshot.close()
    return requested_output_dir


@contextmanager
def _supplied_destination_parent(
    path: str | Path,
    parent: AnchoredDirectory,
) -> Iterator[tuple[Path, AnchoredDirectory]]:
    output_dir = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent.verify()
    if output_dir.parent != parent.directory or Path(output_dir.name).name != output_dir.name:
        raise ArchiveSecurityError("archive destination does not belong to the supplied anchored parent")
    yield output_dir, parent
    parent.verify()


@contextmanager
def _anchored_destination_parent(
    path: str | Path,
) -> Iterator[tuple[Path, AnchoredDirectory]]:
    raw_path = os.fspath(path)
    if not raw_path.strip() or raw_path.strip() in {".", ".."}:
        raise ArchiveSecurityError("archive destination must be a specific non-root path")
    output_dir = Path(os.path.abspath(os.path.expanduser(raw_path)))
    existing_ancestor = output_dir.parent
    while not os.path.lexists(existing_ancestor):
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            raise ArchiveSecurityError(f"could not find an existing ancestor for destination: {output_dir}")
        existing_ancestor = parent
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(existing_ancestor, existing_ancestor))
        for part in output_dir.parent.relative_to(existing_ancestor).parts:
            if not anchor.lexists(part):
                anchor.mkdir(part)
            anchor = stack.enter_context(anchor.open_directory(part))
        yield output_dir, anchor


def _validated_archive_path(path: str | Path) -> Path:
    archive_path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not os.path.lexists(archive_path):
        raise FileNotFoundError(f"archive not found: {archive_path}")
    metadata = archive_path.lstat()
    if _is_link_or_reparse(metadata):
        raise ArchiveSecurityError(f"archive path may not be a link or reparse point: {archive_path}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ArchiveSecurityError(f"archive path must be a single-link regular file: {archive_path}")
    return archive_path


def _require_same_regular_file(before: os.stat_result, after: os.stat_result, message: str) -> None:
    if (
        not stat.S_ISREG(after.st_mode)
        or _is_link_or_reparse(after)
        or after.st_nlink != 1
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ArchiveSecurityError(message)


def _reopen_anonymous_snapshot_read_only(descriptor: int) -> int:
    if os.name == "posix":
        proc_descriptor = Path(f"/proc/self/fd/{descriptor}")
        try:
            reopened = os.open(proc_descriptor, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except OSError as exc:
            raise ArchiveSecurityError("anonymous archive snapshot could not be reopened read-only") from exc
        before = os.fstat(descriptor)
        after = os.fstat(reopened)
        if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
            os.close(reopened)
            raise ArchiveSecurityError("anonymous archive snapshot changed while reopening read-only")
        return reopened
    raise ArchiveSecurityError("anonymous archive snapshots are unsupported on this platform")


def _normalized_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA256 digest")
    return normalized


def _preflight_destination(
    parent: AnchoredDirectory,
    output_name: str,
    *,
    overwrite: bool,
) -> None:
    if not parent.lexists(output_name):
        return
    metadata = parent.lstat(output_name)
    if _is_link_or_reparse(metadata):
        raise ArchiveSecurityError("destination may not be a link or reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise FileExistsError(f"output path exists and is not a directory: {parent.directory / output_name}")
    with parent.open_directory(output_name) as output:
        if output.names() and not overwrite:
            raise FileExistsError(f"output directory already exists and is not empty: {parent.directory / output_name}")
        if overwrite:
            _assert_anchored_tree_has_no_link_seams(output)


def _reject_source_output_overlap(archive_path: Path, output_dir: Path) -> None:
    resolved_archive = archive_path.resolve(strict=True)
    resolved_output = output_dir.resolve(strict=False)
    try:
        resolved_archive.relative_to(resolved_output)
    except ValueError:
        return
    raise ArchiveSecurityError("archive input may not be located inside its extraction destination")


def _is_safe_member_name(name: str) -> bool:
    try:
        _validated_member_name(name)
    except ArchiveSecurityError:
        return False
    return True


def normalize_member_name(name: str) -> str:
    """Return the canonical forward-slash archive member spelling."""

    return str(name).replace("\\", "/")


def archive_member_summary(
    archive_path: str | Path | ArchiveSnapshot,
    *,
    include_member_globs: Sequence[str] = (),
    exclude_member_globs: Sequence[str] = (),
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    cancel_requested: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Describe deterministic ZIP member selection after full validation."""

    _check_archive_abort(cancel_requested, deadline_monotonic)
    limits = _validated_limits(
        max_archive_bytes=max_archive_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    owns_snapshot = not isinstance(archive_path, ArchiveSnapshot)
    snapshot = (
        ArchiveSnapshot.open(
            archive_path,
            max_archive_bytes=limits.max_archive_bytes,
            cancel_requested=cancel_requested,
            deadline_monotonic=deadline_monotonic,
        )
        if owns_snapshot
        else archive_path
    )
    try:
        if snapshot.source_path.suffix.lower() != ".zip":
            return {}
        if snapshot.byte_count > limits.max_archive_bytes:
            raise ArchiveSecurityError(f"archive exceeds the {limits.max_archive_bytes}-byte input limit")
        with zipfile.ZipFile(snapshot.rewind()) as archive:
            members = _validated_zip_members(
                archive.infolist(),
                include_member_globs,
                exclude_member_globs,
                limits,
            )
        _check_archive_abort(cancel_requested, deadline_monotonic)
        snapshot.verify_final()
    finally:
        if owns_snapshot:
            snapshot.close()
    files = [member for member in members if not member.is_dir]
    resource_forks = [member for member in files if member.resource_fork]
    eligible_files = [member for member in files if not member.resource_fork]
    selected = [member for member in eligible_files if member.selected]
    selected_images = [member.name for member in selected if member.name.lower().endswith(".png")]
    _require_selected_images(include_member_globs, exclude_member_globs, selected_images)
    return {
        "total_archive_members": len(members),
        "total_uncompressed_bytes": sum(member.size for member in files),
        "included_members": [member.name for member in selected],
        "excluded_members": [member.name for member in files if member not in selected],
        "resource_fork_members": [member.name for member in resource_forks],
        "unsupported_members": [member.name for member in selected if not member.name.lower().endswith(".png")],
        "selected_image_members": selected_images,
        "selected_uncompressed_bytes": sum(member.size for member in selected),
        "unsafe_members": [],
        "include_member_globs": list(include_member_globs),
        "exclude_member_globs": list(exclude_member_globs),
    }


def _member_selected(name: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if includes and not any(fnmatchcase(name, pattern) for pattern in includes):
        return False
    return not any(fnmatchcase(name, pattern) for pattern in excludes)


def appledouble_detection_basis(member_path: str, payload_prefix: bytes = b"") -> tuple[str, ...]:
    """Return deterministic AppleDouble/resource-fork detection evidence."""

    normalized = normalize_member_name(member_path)
    path = PurePosixPath(normalized)
    folded_parts = {part.casefold() for part in path.parts}
    evidence: list[str] = []
    if path.name.startswith("._"):
        evidence.append("dot_underscore_name")
    if "__macosx" in folded_parts:
        evidence.append("macosx_metadata_directory")
    if ".appledouble" in folded_parts or "resource.frk" in folded_parts:
        evidence.append("resource_fork_directory")
    if len(path.parts) >= 2 and path.parts[-2].casefold() == "namedfork" and path.name.casefold() == "rsrc":
        evidence.append("named_resource_fork")
    magic = bytes(payload_prefix[:4])
    if magic == APPLEDOUBLE_MAGIC:
        evidence.append("appledouble_magic")
    elif magic == APPLESINGLE_MAGIC:
        evidence.append("applesingle_magic")
    return tuple(evidence)


def is_appledouble_path(member_path: str) -> bool:
    """Return whether path structure alone marks metadata/resource-fork data."""

    return bool(appledouble_detection_basis(member_path))


def is_appledouble_record(member_path: str, payload_prefix: bytes = b"") -> bool:
    """Return whether path or file structure marks an AppleDouble artifact."""

    return bool(appledouble_detection_basis(member_path, payload_prefix))


def _extract_zip(
    snapshot: ArchiveSnapshot,
    staging: AnchoredDirectory,
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
    cancel_requested: Callable[[], bool] | None,
    progress: Callable[[int, int], None] | None,
    deadline_monotonic: float | None,
) -> None:
    with zipfile.ZipFile(snapshot.rewind()) as archive:
        members = _validated_zip_members(archive.infolist(), includes, excludes, limits)
        selected_images = [
            member.name
            for member in members
            if member.selected and not member.resource_fork and member.name.lower().endswith(".png")
        ]
        _require_selected_images(includes, excludes, selected_images)
        extracted_total = 0
        extracted_members = 0
        for member in members:
            _check_archive_abort(cancel_requested, deadline_monotonic)
            if member.is_dir or member.resource_fork or not member.selected:
                continue
            with archive.open(member.raw, "r") as source, _open_staging_sink(staging, member.name) as sink:
                copied = _copy_member(
                    source,
                    sink,
                    expected_size=member.size,
                    max_total=limits.max_total_bytes - extracted_total,
                    cancel_requested=cancel_requested,
                    deadline_monotonic=deadline_monotonic,
                )
            extracted_total += copied
            extracted_members += 1
            if progress is not None:
                progress(extracted_members, len(selected_images))


def _extract_tar(
    snapshot: ArchiveSnapshot,
    staging: AnchoredDirectory,
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
    cancel_requested: Callable[[], bool] | None,
    progress: Callable[[int, int], None] | None,
    deadline_monotonic: float | None,
) -> None:
    with tarfile.open(fileobj=snapshot.rewind(), mode="r:*") as archive:
        infos: list[tarfile.TarInfo] = []
        for info in archive:
            _check_archive_abort(cancel_requested, deadline_monotonic)
            infos.append(info)
            if len(infos) > limits.max_members:
                raise ArchiveSecurityError(f"archive contains more than {limits.max_members} members")
        members = _validated_tar_members(
            infos,
            includes,
            excludes,
            limits,
            compressed_size=snapshot.byte_count,
        )
        selected_images = [
            member.name
            for member in members
            if member.selected and not member.resource_fork and member.name.lower().endswith(".png")
        ]
        _require_selected_images(includes, excludes, selected_images)
        extracted_total = 0
        extracted_members = 0
        for member in members:
            _check_archive_abort(cancel_requested, deadline_monotonic)
            if member.is_dir or member.resource_fork or not member.selected:
                continue
            source = archive.extractfile(member.raw)
            if source is None:
                raise ArchiveSecurityError(f"could not read archive member {member.name!r}")
            with source, _open_staging_sink(staging, member.name) as sink:
                copied = _copy_member(
                    source,
                    sink,
                    expected_size=member.size,
                    max_total=limits.max_total_bytes - extracted_total,
                    cancel_requested=cancel_requested,
                    deadline_monotonic=deadline_monotonic,
                )
            extracted_total += copied
            extracted_members += 1
            if progress is not None:
                progress(extracted_members, len(selected_images))


def _validated_zip_members(
    infos: Sequence[zipfile.ZipInfo],
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
) -> list[_ValidatedMember]:
    if len(infos) > limits.max_members:
        raise ArchiveSecurityError(f"archive contains more than {limits.max_members} members")
    members: list[_ValidatedMember] = []
    total_size = 0
    for info in infos:
        name, collision_key = _validated_member_name(info.filename)
        if info.flag_bits & 0x1:
            raise ArchiveSecurityError(f"encrypted archive member is not allowed: {name!r}")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        is_dir = info.is_dir()
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ArchiveSecurityError(f"linked or special ZIP member is not allowed: {name!r}")
        if is_dir and file_type == stat.S_IFREG:
            raise ArchiveSecurityError(f"ZIP member has conflicting directory metadata: {name!r}")
        if not is_dir and file_type == stat.S_IFDIR:
            raise ArchiveSecurityError(f"ZIP member has conflicting file metadata: {name!r}")
        size = int(info.file_size)
        compressed_size = int(info.compress_size)
        _validate_member_size(name, size, limits)
        if compressed_size < 0:
            raise ArchiveSecurityError(f"archive member has a negative compressed size: {name!r}")
        if size and compressed_size == 0:
            raise ArchiveSecurityError(f"archive member has an impossible compression size: {name!r}")
        if size / max(1, compressed_size) > limits.max_compression_ratio:
            raise ArchiveSecurityError(f"archive member exceeds the compression-ratio limit: {name!r}")
        if not is_dir:
            total_size += size
            if total_size > limits.max_total_bytes:
                raise ArchiveSecurityError(f"archive expands beyond the {limits.max_total_bytes}-byte limit")
        members.append(
            _ValidatedMember(
                raw=info,
                name=name,
                collision_key=collision_key,
                is_dir=is_dir,
                size=size,
                selected=_member_selected(name, includes, excludes),
                resource_fork=not is_dir and is_appledouble_path(name),
            )
        )
    _reject_member_collisions(members)
    return members


def _validated_tar_members(
    infos: Sequence[tarfile.TarInfo],
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
    *,
    compressed_size: int,
) -> list[_ValidatedMember]:
    if len(infos) > limits.max_members:
        raise ArchiveSecurityError(f"archive contains more than {limits.max_members} members")
    members: list[_ValidatedMember] = []
    total_size = 0
    for info in infos:
        name, collision_key = _validated_member_name(info.name)
        if not info.isdir() and not info.isfile():
            raise ArchiveSecurityError(f"linked or special TAR member is not allowed: {name!r}")
        size = int(info.size)
        _validate_member_size(name, size, limits)
        if info.isfile():
            total_size += size
            if total_size > limits.max_total_bytes:
                raise ArchiveSecurityError(f"archive expands beyond the {limits.max_total_bytes}-byte limit")
        members.append(
            _ValidatedMember(
                raw=info,
                name=name,
                collision_key=collision_key,
                is_dir=info.isdir(),
                size=size,
                selected=_member_selected(name, includes, excludes),
                resource_fork=info.isfile() and is_appledouble_path(name),
            )
        )
    if total_size and total_size / max(1, compressed_size) > limits.max_compression_ratio:
        raise ArchiveSecurityError("archive exceeds the compression-ratio limit")
    _reject_member_collisions(members)
    return members


def _validated_member_name(raw_name: str) -> tuple[str, str]:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ArchiveSecurityError(f"unsafe archive member name: {raw_name!r}")
    if any(ord(character) < 32 for character in raw_name):
        raise ArchiveSecurityError(f"archive member contains control characters: {raw_name!r}")
    if len(raw_name) > 4096:
        raise ArchiveSecurityError("archive member path is too long")
    is_directory_spelling = raw_name.endswith("/")
    name = raw_name[:-1] if is_directory_spelling else raw_name
    if not name or name.startswith("/") or "//" in name:
        raise ArchiveSecurityError(f"unsafe archive member name: {raw_name!r}")
    parts = name.split("/")
    for part in parts:
        if part in {"", ".", ".."} or len(part) > 255:
            raise ArchiveSecurityError(f"unsafe archive member name: {raw_name!r}")
        if part.endswith((".", " ")) or ":" in part:
            raise ArchiveSecurityError(f"platform-ambiguous archive member name: {raw_name!r}")
        reserved_stem = part.rstrip(" .").split(".", 1)[0].casefold()
        if reserved_stem in _WINDOWS_RESERVED_NAMES:
            raise ArchiveSecurityError(f"reserved archive member name: {raw_name!r}")
    canonical = "/".join(parts)
    collision_key = unicodedata.normalize("NFC", canonical).casefold()
    return canonical, collision_key


def _reject_member_collisions(members: Sequence[_ValidatedMember]) -> None:
    by_key: dict[str, _ValidatedMember] = {}
    for member in members:
        previous = by_key.get(member.collision_key)
        if previous is not None:
            raise ArchiveSecurityError(
                f"duplicate or platform-colliding archive members: {previous.name!r} and {member.name!r}"
            )
        by_key[member.collision_key] = member
    file_keys = {member.collision_key for member in members if not member.is_dir}
    for member in members:
        parts = member.collision_key.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            if parent in file_keys:
                raise ArchiveSecurityError(f"archive member {member.name!r} is nested below a file member")


def _validate_member_size(name: str, size: int, limits: _Limits) -> None:
    if size < 0:
        raise ArchiveSecurityError(f"archive member has a negative size: {name!r}")
    if size > limits.max_member_bytes:
        raise ArchiveSecurityError(f"archive member {name!r} exceeds the {limits.max_member_bytes}-byte member limit")


def _copy_member(
    source: BinaryIO,
    sink: BinaryIO,
    *,
    expected_size: int,
    max_total: int,
    cancel_requested: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> int:
    copied = 0
    while True:
        _check_archive_abort(cancel_requested, deadline_monotonic)
        chunk = source.read(_COPY_CHUNK_BYTES)
        if not chunk:
            break
        if copied + len(chunk) > expected_size or copied + len(chunk) > max_total:
            raise ArchiveSecurityError("archive member expanded beyond its declared or total size limit")
        sink.write(chunk)
        copied += len(chunk)
    if copied != expected_size:
        raise ArchiveSecurityError(f"archive member size mismatch: expected {expected_size} bytes, extracted {copied}")
    sink.flush()
    os.fsync(sink.fileno())
    return copied


def _check_archive_abort(
    cancel_requested: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ArchiveCancelled("archive extraction was cancelled")
    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        raise ArchiveCancelled("archive extraction exceeded its deadline")


@contextmanager
def _open_staging_sink(staging: AnchoredDirectory, member_name: str) -> Iterator[BinaryIO]:
    parts = PurePosixPath(member_name).parts
    if not parts:
        raise ArchiveSecurityError("archive member has no extraction path")
    with ExitStack() as stack:
        parent = staging
        for part in parts[:-1]:
            if not parent.lexists(part):
                parent.mkdir(part)
            parent = stack.enter_context(parent.open_directory(part))
        name = parts[-1]
        if parent.lexists(name):
            raise ArchiveSecurityError(f"archive member destination already exists: {member_name!r}")
        descriptor = parent.open_file(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        handle = os.fdopen(descriptor, "wb")
        try:
            yield handle
        finally:
            handle.close()
        if not identity.matches(parent.lstat(name)):
            raise ArchiveSecurityError(f"archive member path changed during extraction: {member_name!r}")


def _publish_staged_tree(
    parent: AnchoredDirectory,
    staging_name: str,
    output_name: str,
    *,
    overwrite: bool,
) -> None:
    _preflight_destination(parent, output_name, overwrite=overwrite)
    staging_before = parent.lstat(staging_name)
    if not stat.S_ISDIR(staging_before.st_mode) or _is_link_or_reparse(staging_before):
        raise ArchiveSecurityError("archive staging is unsafe")
    owned_staging = OwnedFileIdentity.from_stat(staging_before)
    if not parent.lexists(output_name):
        try:
            parent.rename(staging_name, output_name, replace=False)
            _verify_published_directory(parent, output_name, staging_before)
        except BaseException:
            if _anchored_entry_matches(parent, output_name, owned_staging):
                parent.quarantine_if_owned(
                    output_name,
                    owned_staging,
                    prefix=f".{output_name}.rollback-",
                )
            raise
        return

    old_before = parent.lstat(output_name)
    if not stat.S_ISDIR(old_before.st_mode) or _is_link_or_reparse(old_before):
        raise ArchiveSecurityError("archive overwrite destination is unsafe")
    owned_old = OwnedFileIdentity.from_stat(old_before)
    backup_name = parent.quarantine_if_owned(
        output_name,
        owned_old,
        prefix=f".{output_name}.backup-",
    )
    if backup_name is None:
        raise ArchiveSecurityError("archive overwrite destination changed before backup")
    try:
        parent.rename(staging_name, output_name, replace=False)
        _verify_published_directory(parent, output_name, staging_before)
    except BaseException:
        if _anchored_entry_matches(parent, output_name, owned_staging):
            parent.quarantine_if_owned(
                output_name,
                owned_staging,
                prefix=f".{output_name}.rollback-",
            )
        if not parent.lexists(output_name) and _anchored_entry_matches(parent, backup_name, owned_old):
            parent.rename(backup_name, output_name, replace=False)
            restored = parent.lstat(output_name)
            if not owned_old.matches(restored):
                raise ArchiveSecurityError("archive rollback did not restore the exact previous destination") from None
        raise
    _emit_recovery_residue_warning(
        "Verified archive committed; the exact previous destination was retained as a recovery residue."
    )


def _emit_recovery_residue_warning(message: str) -> None:
    try:
        warnings.warn(message, ArchiveRecoveryResidueWarning, stacklevel=3)
    except Exception:
        # Warning delivery is advisory and occurs after the durable commit.
        # A warnings-as-errors policy or custom showwarning hook must not turn
        # the committed publication into a reported transactional failure.
        return


def _verify_published_directory(
    parent: AnchoredDirectory,
    output_name: str,
    staging_before: os.stat_result,
) -> None:
    published = parent.lstat(output_name)
    if (
        not stat.S_ISDIR(published.st_mode)
        or _is_link_or_reparse(published)
        or published.st_dev != staging_before.st_dev
        or published.st_ino != staging_before.st_ino
    ):
        raise ArchiveSecurityError("archive staging identity changed during publication")
    parent.verify()


def _anchored_entry_matches(
    parent: AnchoredDirectory,
    name: str,
    identity: OwnedFileIdentity,
) -> bool:
    try:
        return identity.matches(parent.lstat(name))
    except FileNotFoundError:
        return False


def _quarantine_failed_staging(
    parent: AnchoredDirectory,
    staging_name: str,
    output_name: str,
) -> None:
    try:
        metadata = parent.lstat(staging_name)
        identity = OwnedFileIdentity.from_stat(metadata)
        parent.quarantine_if_owned(
            staging_name,
            identity,
            prefix=f".{output_name}.failed-",
        )
    except (FileNotFoundError, OSError, ValueError):
        # Failure cleanup must never widen into deletion or obscure the primary
        # extraction error. An unproven staging path is deliberately retained.
        return


def _assert_anchored_tree_has_no_link_seams(root: AnchoredDirectory) -> None:
    for name in root.names():
        metadata = root.lstat(name)
        if _is_link_or_reparse(metadata):
            raise ArchiveSecurityError(f"destination tree contains a link or reparse point: {name}")
        if stat.S_ISDIR(metadata.st_mode):
            with root.open_directory(name) as child:
                _assert_anchored_tree_has_no_link_seams(child)
        elif not stat.S_ISREG(metadata.st_mode):
            raise ArchiveSecurityError(f"destination tree contains a special entry: {name}")


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _validated_limits(
    *,
    max_archive_bytes: int,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
    max_compression_ratio: float,
) -> _Limits:
    if (
        max_archive_bytes <= 0
        or max_members <= 0
        or max_member_bytes <= 0
        or max_total_bytes <= 0
        or max_compression_ratio <= 0
    ):
        raise ValueError("archive extraction limits must all be positive")
    if max_member_bytes > max_total_bytes:
        raise ValueError("max_member_bytes may not exceed max_total_bytes")
    return _Limits(
        max_archive_bytes=max_archive_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_compression_ratio=max_compression_ratio,
    )


def _require_selected_images(includes: Sequence[str], excludes: Sequence[str], selected: Sequence[str]) -> None:
    if (includes or excludes) and not selected:
        raise ValueError("archive member filters selected zero PNG images")


def iter_archive_pngs(archive_path: str | Path) -> list[str]:
    """Return sorted PNG member names inside a fully validated ZIP."""

    archive_path = _validated_archive_path(archive_path)
    limits = _validated_limits(
        max_archive_bytes=DEFAULT_MAX_ARCHIVE_BYTES,
        max_members=DEFAULT_MAX_ARCHIVE_MEMBERS,
        max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
        max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
        max_compression_ratio=DEFAULT_MAX_COMPRESSION_RATIO,
    )
    if archive_path.stat().st_size > limits.max_archive_bytes:
        raise ArchiveSecurityError(f"archive exceeds the {limits.max_archive_bytes}-byte input limit")
    with zipfile.ZipFile(archive_path) as archive:
        members = _validated_zip_members(archive.infolist(), (), (), limits)
    return sorted(
        member.name
        for member in members
        if not member.is_dir and member.name.lower().endswith(".png") and not member.resource_fork
    )


__all__ = [
    "APPLEDOUBLE_MAGIC",
    "APPLESINGLE_MAGIC",
    "DEFAULT_MAX_ARCHIVE_BYTES",
    "DEFAULT_MAX_ARCHIVE_MEMBERS",
    "DEFAULT_MAX_COMPRESSION_RATIO",
    "DEFAULT_MAX_MEMBER_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "ArchiveCancelled",
    "ArchiveRecoveryResidueWarning",
    "ArchiveSecurityError",
    "ArchiveSnapshot",
    "appledouble_detection_basis",
    "archive_member_summary",
    "extract_archive",
    "is_appledouble_path",
    "is_appledouble_record",
    "iter_archive_pngs",
    "normalize_member_name",
]

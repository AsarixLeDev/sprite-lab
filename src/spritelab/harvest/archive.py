"""Bounded, transactional archive extraction for harvested packs."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
import tempfile
import time
import unicodedata
import uuid
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from spritelab.utils.safe_fs import remove_confined_tree, require_confined_path

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


def extract_archive(
    archive_path: str | Path,
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
    archive_path = _validated_archive_path(archive_path)
    suffixes = "".join(archive_path.suffixes).lower()
    is_zip = archive_path.suffix.lower() == ".zip"
    is_tar = suffixes.endswith((".tar", ".tar.gz", ".tgz"))
    if not is_zip and not is_tar:
        raise ValueError(f"unsupported archive type: {archive_path.name}")
    if archive_path.stat().st_size > limits.max_archive_bytes:
        raise ArchiveSecurityError(f"archive exceeds the {limits.max_archive_bytes}-byte input limit")
    expected_digest = _normalized_sha256(expected_sha256)
    initial_digest = _compute_sha256(
        archive_path,
        max_bytes=limits.max_archive_bytes,
        cancel_requested=cancel_requested,
        deadline_monotonic=deadline_monotonic,
    )
    if expected_digest is not None and initial_digest != expected_digest:
        raise ArchiveSecurityError(f"archive SHA256 mismatch: expected {expected_digest}, got {initial_digest}")
    requested_output_dir = Path(output_dir)
    output_dir = _prepare_destination_path(output_dir)
    _preflight_destination(output_dir, overwrite=overwrite)
    _reject_source_output_overlap(archive_path, output_dir)

    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.extract-",
            dir=output_dir.parent,
        )
    )
    staging = require_confined_path(staging, output_dir.parent)
    try:
        if is_zip:
            _extract_zip(
                archive_path,
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
                archive_path,
                staging,
                include_member_globs,
                exclude_member_globs,
                limits,
                cancel_requested,
                progress,
                deadline_monotonic,
            )
        final_digest = _compute_sha256(
            archive_path,
            max_bytes=limits.max_archive_bytes,
            cancel_requested=cancel_requested,
            deadline_monotonic=deadline_monotonic,
        )
        if final_digest != initial_digest:
            raise ArchiveSecurityError("archive changed while it was being extracted")
        _publish_staged_tree(staging, output_dir, overwrite=overwrite)
        staging = None
    finally:
        if staging is not None:
            remove_confined_tree(staging, output_dir.parent, missing_ok=True)
    return requested_output_dir


def _validated_archive_path(path: str | Path) -> Path:
    archive_path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not os.path.lexists(archive_path):
        raise FileNotFoundError(f"archive not found: {archive_path}")
    metadata = archive_path.lstat()
    if _is_link_or_reparse(metadata):
        raise ArchiveSecurityError(f"archive path may not be a link or reparse point: {archive_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ArchiveSecurityError(f"archive path must be a regular file: {archive_path}")
    return archive_path


def _prepare_destination_path(path: str | Path) -> Path:
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
    metadata = existing_ancestor.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveSecurityError(f"destination crosses a linked or non-directory ancestor: {existing_ancestor}")
    output_dir = require_confined_path(output_dir, existing_ancestor)
    _create_destination_parents(output_dir.parent, existing_ancestor)
    return require_confined_path(output_dir, existing_ancestor)


def _create_destination_parents(parent: Path, root: Path) -> None:
    current = root
    for part in parent.relative_to(root).parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        metadata = current.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode) or current.is_mount():
            raise ArchiveSecurityError(f"destination crosses an unsafe directory seam: {current}")
        require_confined_path(current, root)


def _compute_sha256(
    path: Path,
    *,
    max_bytes: int,
    cancel_requested: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            _check_archive_abort(cancel_requested, deadline_monotonic)
            total += len(chunk)
            if total > max_bytes:
                raise ArchiveSecurityError(f"archive exceeds the {max_bytes}-byte input limit")
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA256 digest")
    return normalized


def _preflight_destination(output_dir: Path, *, overwrite: bool) -> None:
    if not os.path.lexists(output_dir):
        return
    require_confined_path(output_dir, output_dir.parent)
    metadata = output_dir.lstat()
    if _is_link_or_reparse(metadata):
        raise ArchiveSecurityError(f"destination may not be a link or reparse point: {output_dir}")
    if output_dir.is_mount():
        raise ArchiveSecurityError(f"destination may not be a mount point: {output_dir}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
    if overwrite:
        _assert_tree_has_no_link_seams(output_dir)


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
    archive_path: str | Path,
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
    archive_path = _validated_archive_path(archive_path)
    if archive_path.suffix.lower() != ".zip":
        return {}
    limits = _validated_limits(
        max_archive_bytes=max_archive_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    if archive_path.stat().st_size > limits.max_archive_bytes:
        raise ArchiveSecurityError(f"archive exceeds the {limits.max_archive_bytes}-byte input limit")
    with zipfile.ZipFile(archive_path) as archive:
        members = _validated_zip_members(
            archive.infolist(),
            include_member_globs,
            exclude_member_globs,
            limits,
        )
    _check_archive_abort(cancel_requested, deadline_monotonic)
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
    archive_path: Path,
    staging: Path,
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
    cancel_requested: Callable[[], bool] | None,
    progress: Callable[[int, int], None] | None,
    deadline_monotonic: float | None,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
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
            target = _staging_target(staging, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member.raw, "r") as source, target.open("xb") as sink:
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
    archive_path: Path,
    staging: Path,
    includes: Sequence[str],
    excludes: Sequence[str],
    limits: _Limits,
    cancel_requested: Callable[[], bool] | None,
    progress: Callable[[int, int], None] | None,
    deadline_monotonic: float | None,
) -> None:
    with tarfile.open(archive_path, mode="r:*") as archive:
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
            compressed_size=archive_path.stat().st_size,
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
            target = _staging_target(staging, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("xb") as sink:
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


def _staging_target(staging: Path, member_name: str) -> Path:
    target = staging.joinpath(*PurePosixPath(member_name).parts)
    return require_confined_path(target, staging)


def _publish_staged_tree(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    _preflight_destination(output_dir, overwrite=overwrite)
    if not os.path.lexists(output_dir):
        os.replace(staging, output_dir)
        return

    backup = require_confined_path(
        output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}",
        output_dir.parent,
    )
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    try:
        remove_confined_tree(backup, output_dir.parent)
    except BaseException:
        os.replace(output_dir, staging)
        os.replace(backup, output_dir)
        raise


def _assert_tree_has_no_link_seams(root: Path) -> None:
    for current_root, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)
        for name in [*directory_names, *file_names]:
            candidate = current / name
            metadata = candidate.lstat()
            if _is_link_or_reparse(metadata) or (stat.S_ISDIR(metadata.st_mode) and candidate.is_mount()):
                raise ArchiveSecurityError(f"destination tree contains a link or reparse point: {candidate}")


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
    "ArchiveSecurityError",
    "appledouble_detection_basis",
    "archive_member_summary",
    "extract_archive",
    "is_appledouble_path",
    "is_appledouble_record",
    "iter_archive_pngs",
    "normalize_member_name",
]

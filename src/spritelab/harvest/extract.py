"""PNG candidate discovery from extracted pack directories."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from spritelab.harvest.archive import appledouble_detection_basis
from spritelab.harvest.sources import SourceRecord
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation, require_confined_path

_MAX_DISCOVERY_PIXELS = 16_777_216
_MAX_CANDIDATE_FILE_BYTES = 256 * 1024 * 1024


class UnsafeSourceTreeError(ValueError):
    """Raised when candidate discovery encounters a link or alias seam."""


@dataclass(frozen=True)
class HarvestCandidate:
    candidate_id: str
    source_id: str
    source_path: str
    extracted_path: Path
    relative_path: str
    image_sha256: str
    width: int
    height: int
    mode: str
    pixel_sha256: str | None = None
    visible_pixel_count: int | None = None
    pixel_variation: bool | None = None
    status: str = "candidate"
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_kind: str = "image"
    extraction_disposition: str = "candidate"
    forensic_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DecodedPngDetails:
    width: int
    height: int
    mode: str
    pixel_sha256: str
    visible_pixel_count: int
    pixel_variation: bool


def make_candidate_id(source_id: str, relative_path: str, image_sha256: str) -> str:
    """Stable candidate ID from source, relative path, and content hash."""

    digest = hashlib.sha256()
    digest.update(source_id.encode("utf-8"))
    digest.update(relative_path.encode("utf-8"))
    digest.update(image_sha256.encode("utf-8"))
    return f"{source_id}__{digest.hexdigest()[:16]}"


def discover_png_candidates(
    root: str | Path,
    source: SourceRecord,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    root_anchor: AnchoredDirectory | None = None,
) -> list[HarvestCandidate]:
    """Discover PNG candidates in deterministic relative-path order."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    if root_anchor is None:
        if not os.path.lexists(root):
            raise NotADirectoryError(f"candidate root is not a directory: {root}")
        root_metadata = root.lstat()
        if _is_link_or_reparse(root_metadata):
            raise UnsafeSourceTreeError(f"candidate root may not be a link or reparse point: {root}")
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise NotADirectoryError(f"candidate root is not a directory: {root}")
        paths = _collect_png_paths(root, recursive=recursive, include_hidden=include_hidden)
    else:
        root_anchor.verify()
        if root != root_anchor.directory:
            raise UnsafeSourceTreeError("candidate root does not match its supplied anchor")
        relative_paths = _collect_anchored_png_paths(
            root_anchor,
            recursive=recursive,
            include_hidden=include_hidden,
        )
        paths = [root.joinpath(*Path(relative).parts) for relative in relative_paths]

    candidates: list[HarvestCandidate] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if root_anchor is None:
            image_sha256, prefix, image_details, load_error = _inspect_candidate_file(path, root)
        else:
            image_sha256, prefix, image_details, load_error = _inspect_anchored_candidate_file(
                root_anchor,
                Path(relative),
                path,
            )
        detection_basis = appledouble_detection_basis(relative, prefix)
        if detection_basis:
            candidates.append(
                HarvestCandidate(
                    candidate_id=make_candidate_id(source.source_id, relative, image_sha256),
                    source_id=source.source_id,
                    source_path=source.source_url or source.local_archive_path or source.local_root_path,
                    extracted_path=path,
                    relative_path=relative,
                    image_sha256=image_sha256,
                    width=0,
                    height=0,
                    mode="",
                    status="rejected",
                    rejection_reasons=("resource-fork metadata is not a sprite",),
                    artifact_kind="metadata_resource_fork",
                    extraction_disposition="reject_resource_fork",
                    forensic_evidence=detection_basis,
                )
            )
            continue
        if load_error is not None or image_details is None:
            candidates.append(
                HarvestCandidate(
                    candidate_id=make_candidate_id(source.source_id, relative, image_sha256),
                    source_id=source.source_id,
                    source_path=source.source_url or source.local_archive_path or source.local_root_path,
                    extracted_path=path,
                    relative_path=relative,
                    image_sha256=image_sha256,
                    width=0,
                    height=0,
                    mode="",
                    status="rejected",
                    rejection_reasons=(f"could not load PNG: {load_error}",),
                )
            )
            continue
        candidates.append(
            HarvestCandidate(
                candidate_id=make_candidate_id(source.source_id, relative, image_sha256),
                source_id=source.source_id,
                source_path=source.source_url or source.local_archive_path or source.local_root_path,
                extracted_path=path,
                relative_path=relative,
                image_sha256=image_sha256,
                width=image_details.width,
                height=image_details.height,
                mode=image_details.mode,
                pixel_sha256=image_details.pixel_sha256,
                visible_pixel_count=image_details.visible_pixel_count,
                pixel_variation=image_details.pixel_variation,
            )
        )
    return candidates


def _collect_anchored_png_paths(
    root: AnchoredDirectory,
    *,
    recursive: bool,
    include_hidden: bool,
) -> list[str]:
    root_device = root.directory_metadata().st_dev
    candidates: list[str] = []

    def visit(directory: AnchoredDirectory, prefix: tuple[str, ...], *, descend: bool) -> None:
        for name in directory.names():
            metadata = directory.lstat(name)
            if _is_link_or_reparse(metadata):
                raise UnsafeSourceTreeError("source tree contains a link or reparse point")
            if metadata.st_dev != root_device:
                raise UnsafeSourceTreeError("source tree crosses a device boundary")
            relative_parts = (*prefix, name)
            if stat.S_ISDIR(metadata.st_mode):
                if descend:
                    with directory.open_directory(name) as child:
                        visit(child, relative_parts, descend=True)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafeSourceTreeError("source tree contains a special filesystem entry")
            if metadata.st_nlink != 1:
                raise UnsafeSourceTreeError("source tree contains a hard-linked file")
            relative = Path(*relative_parts).as_posix()
            hidden = any(part.startswith(".") for part in relative_parts)
            if name.lower().endswith(".png") and (include_hidden or not hidden):
                candidates.append(relative)

    visit(root, (), descend=recursive)
    root.verify()
    return sorted(candidates, key=lambda value: (value.casefold(), value))


def _inspect_anchored_candidate_file(
    root: AnchoredDirectory,
    relative: Path,
    display_path: Path,
) -> tuple[str, bytes, _DecodedPngDetails | None, str | None]:
    with ExitStack() as stack:
        parent = root
        for part in relative.parts[:-1]:
            parent = stack.enter_context(parent.open_directory(part))
        name = relative.parts[-1]
        before = parent.lstat(name)
        _validate_candidate_metadata(display_path, before)
        if before.st_size > _MAX_CANDIDATE_FILE_BYTES:
            return (
                "",
                b"",
                None,
                f"file exceeds the {_MAX_CANDIDATE_FILE_BYTES}-byte safe candidate limit",
            )
        descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _identity(before) != _identity(opened):
                raise UnsafeSourceTreeError(f"candidate changed while it was opened: {display_path}")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                image_sha256, prefix = _hash_open_file(handle)
                image_details, load_error = _load_single_frame_png(handle)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = parent.lstat(name)
        if _identity(before) != _identity(after) or _identity(before) != _identity(path_after):
            raise UnsafeSourceTreeError(f"candidate changed while it was read: {display_path}")
        return image_sha256, prefix, image_details, load_error


def filter_candidate_basic(
    candidate: HarvestCandidate,
    *,
    allow_non_32: bool = True,
    min_size: int = 8,
    max_size: int = 512,
) -> HarvestCandidate:
    """Apply cheap validity filters without touching the file."""

    if candidate.status == "rejected":
        return candidate
    reasons: list[str] = []
    warnings = list(candidate.warnings)
    if candidate.width < min_size or candidate.height < min_size:
        reasons.append(f"image too small ({candidate.width}x{candidate.height}, min {min_size}).")
    if candidate.width > max_size or candidate.height > max_size:
        reasons.append(f"image too large ({candidate.width}x{candidate.height}, max {max_size}).")
    if (candidate.width, candidate.height) != (32, 32):
        if allow_non_32:
            warnings.append(f"non-32x32 image ({candidate.width}x{candidate.height}); may need slicing or padding.")
        else:
            reasons.append(f"expected 32x32, got {candidate.width}x{candidate.height}.")
    if candidate.visible_pixel_count == 0:
        reasons.append("image is fully transparent.")
    elif candidate.pixel_variation is False:
        reasons.append("image contains only one constant RGBA value.")
    if reasons:
        return replace(
            candidate,
            status="rejected",
            rejection_reasons=(*candidate.rejection_reasons, *reasons),
            warnings=tuple(warnings),
        )
    return replace(candidate, warnings=tuple(warnings))


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _collect_png_paths(root: Path, *, recursive: bool, include_hidden: bool) -> list[Path]:
    candidates: list[Path] = []

    def visit(directory: Path, *, descend: bool) -> None:
        _validate_directory(directory, root)
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: (entry.name.casefold(), entry.name))
        except OSError as exc:
            raise UnsafeSourceTreeError(f"could not enumerate source directory safely: {directory}") from exc
        for entry in entries:
            path = _require_source_path(Path(entry.path), root)
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise UnsafeSourceTreeError(f"could not inspect source entry safely: {path}") from exc
            if _is_link_or_reparse(metadata):
                raise UnsafeSourceTreeError(f"source tree contains a link or reparse point: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                if path.is_mount():
                    raise UnsafeSourceTreeError(f"source tree contains a nested mount point: {path}")
                if descend:
                    visit(path, descend=True)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafeSourceTreeError(f"source tree contains a special filesystem entry: {path}")
            if metadata.st_nlink != 1:
                raise UnsafeSourceTreeError(f"source tree contains a hard-linked file: {path}")
            if path.suffix.lower() == ".png" and (include_hidden or not _is_hidden(path, root)):
                candidates.append(path)

    visit(root, descend=recursive)
    return sorted(
        candidates,
        key=lambda path: (path.relative_to(root).as_posix().casefold(), path.relative_to(root).as_posix()),
    )


def _validate_directory(directory: Path, root: Path) -> None:
    if directory != root:
        _require_source_path(directory, root)
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise UnsafeSourceTreeError(f"could not inspect source directory safely: {directory}") from exc
    if (
        _is_link_or_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or (directory != root and directory.is_mount())
    ):
        raise UnsafeSourceTreeError(f"source directory changed type during discovery: {directory}")


def _inspect_candidate_file(
    path: Path,
    root: Path,
) -> tuple[str, bytes, _DecodedPngDetails | None, str | None]:
    _require_source_path(path, root)
    try:
        before = path.lstat()
    except OSError as exc:
        raise UnsafeSourceTreeError(f"could not inspect candidate safely: {path}") from exc
    _validate_candidate_metadata(path, before)
    if before.st_size > _MAX_CANDIDATE_FILE_BYTES:
        return (
            "",
            b"",
            None,
            f"file exceeds the {_MAX_CANDIDATE_FILE_BYTES}-byte safe candidate limit",
        )
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _identity(before) != _identity(opened):
                raise UnsafeSourceTreeError(f"candidate changed while it was opened: {path}")
            image_sha256, prefix = _hash_open_file(handle)
            image_details, load_error = _load_single_frame_png(handle)
            after = os.fstat(handle.fileno())
    except UnsafeSourceTreeError:
        raise
    except OSError as exc:
        return "", b"", None, str(exc)
    if _identity(before) != _identity(after):
        raise UnsafeSourceTreeError(f"candidate changed while it was read: {path}")
    return image_sha256, prefix, image_details, load_error


def _hash_open_file(handle: BinaryIO) -> tuple[str, bytes]:
    digest = hashlib.sha256()
    prefix = b""
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        if not prefix:
            prefix = chunk[:4]
        digest.update(chunk)
    return digest.hexdigest(), prefix


def _load_single_frame_png(handle: BinaryIO) -> tuple[_DecodedPngDetails | None, str | None]:
    """Decode only an exact, single-frame PNG through an already-held file."""

    try:
        handle.seek(0)
        with Image.open(handle) as image:
            width, height = image.size
            if image.format != "PNG":
                return None, "file extension does not contain a PNG image"
            frame_count = getattr(image, "n_frames", 1)
            if type(frame_count) is not int or frame_count != 1 or bool(getattr(image, "is_animated", False)):
                return None, "animated or multi-frame PNG/APNG images are not supported"
            if width <= 0 or height <= 0 or width * height > _MAX_DISCOVERY_PIXELS:
                return (
                    None,
                    f"image dimensions {width}x{height} exceed the {_MAX_DISCOVERY_PIXELS}-pixel safe decode limit",
                )
            image.load()
            original_mode = image.mode
            rgba = image.convert("RGBA")
            pixels = rgba.tobytes()
            first_pixel = pixels[:4]
            return (
                _DecodedPngDetails(
                    width=width,
                    height=height,
                    mode=original_mode,
                    pixel_sha256=hashlib.sha256(pixels).hexdigest(),
                    visible_pixel_count=sum(1 for alpha in pixels[3::4] if alpha != 0),
                    pixel_variation=any(
                        pixels[offset : offset + 4] != first_pixel for offset in range(4, len(pixels), 4)
                    ),
                ),
                None,
            )
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return None, str(exc)


def _validate_candidate_metadata(path: Path, metadata: os.stat_result) -> None:
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise UnsafeSourceTreeError(f"candidate is not a confined regular file: {path}")
    if metadata.st_nlink != 1:
        raise UnsafeSourceTreeError(f"candidate is hard-linked outside its source identity: {path}")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_mode,
        metadata.st_nlink,
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _require_source_path(path: Path, root: Path) -> Path:
    try:
        return require_confined_path(path, root)
    except UnsafeFilesystemOperation as exc:
        raise UnsafeSourceTreeError(f"source path crosses a link or reparse boundary: {path}") from exc


__all__ = [
    "HarvestCandidate",
    "UnsafeSourceTreeError",
    "discover_png_candidates",
    "filter_candidate_basic",
    "make_candidate_id",
]

"""Safe archive extraction for harvested packs."""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"
APPLESINGLE_MAGIC = b"\x00\x05\x16\x00"


def extract_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    include_member_globs: Sequence[str] = (),
    exclude_member_globs: Sequence[str] = (),
) -> Path:
    """Extract a .zip (or .tar/.tar.gz) archive path-traversal-safely.

    Entries with absolute paths or ``..`` components are skipped.
    """

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive_path.suffixes).lower()
    if archive_path.suffix.lower() == ".zip":
        _extract_zip(archive_path, output_dir, include_member_globs, exclude_member_globs)
    elif suffixes.endswith((".tar", ".tar.gz", ".tgz")):
        _extract_tar(archive_path, output_dir)
    else:
        raise ValueError(f"unsupported archive type: {archive_path.name}")
    return output_dir


def _is_safe_member_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")):
        return False
    path = Path(name.replace("\\", "/"))
    if path.is_absolute() or path.drive:
        return False
    return ".." not in path.parts


def normalize_member_name(name: str) -> str:
    """Return the canonical forward-slash archive member spelling."""

    return str(name).replace("\\", "/")


def archive_member_summary(
    archive_path: str | Path,
    *,
    include_member_globs: Sequence[str] = (),
    exclude_member_globs: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe deterministic ZIP member selection before extraction."""

    archive_path = Path(archive_path)
    if archive_path.suffix.lower() != ".zip":
        return {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = tuple(archive.infolist())
    safe_files = [info for info in infos if not info.is_dir() and _is_safe_member_name(info.filename)]
    unsafe = [normalize_member_name(info.filename) for info in infos if not _is_safe_member_name(info.filename)]
    resource_forks = [info for info in safe_files if is_appledouble_path(info.filename)]
    eligible_files = [info for info in safe_files if info not in resource_forks]
    selected = [
        info for info in eligible_files if _member_selected(info.filename, include_member_globs, exclude_member_globs)
    ]
    selected_images = [
        normalize_member_name(info.filename) for info in selected if info.filename.lower().endswith(".png")
    ]
    if (include_member_globs or exclude_member_globs) and not selected_images:
        raise ValueError("archive member filters selected zero PNG images")
    return {
        "total_archive_members": len(infos),
        "included_members": [normalize_member_name(info.filename) for info in selected],
        "excluded_members": [normalize_member_name(info.filename) for info in safe_files if info not in selected],
        "resource_fork_members": [normalize_member_name(info.filename) for info in resource_forks],
        "unsupported_members": [
            normalize_member_name(info.filename) for info in selected if not info.filename.lower().endswith(".png")
        ],
        "selected_image_members": selected_images,
        "unsafe_members": unsafe,
        "include_member_globs": list(include_member_globs),
        "exclude_member_globs": list(exclude_member_globs),
    }


def _member_selected(name: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    normalized = normalize_member_name(name)
    if includes and not any(fnmatchcase(normalized, pattern) for pattern in includes):
        return False
    return not any(fnmatchcase(normalized, pattern) for pattern in excludes)


def appledouble_detection_basis(member_path: str, payload_prefix: bytes = b"") -> tuple[str, ...]:
    """Return deterministic AppleDouble/resource-fork detection evidence.

    Path evidence is evaluated without decoding the member.  The structural
    magic check is intentionally limited to the first four bytes so callers
    never have to parse a resource fork as an image.
    """

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


def _extract_zip(archive_path: Path, output_dir: Path, includes: Sequence[str], excludes: Sequence[str]) -> None:
    resolved_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not _is_safe_member_name(info.filename):
                continue
            if not info.is_dir() and is_appledouble_path(info.filename):
                continue
            if not info.is_dir() and not _member_selected(info.filename, includes, excludes):
                continue
            target = (output_dir / info.filename.replace("\\", "/")).resolve()
            if resolved_root != target and resolved_root not in target.parents:
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                sink.write(source.read())


def _extract_tar(archive_path: Path, output_dir: Path) -> None:
    resolved_root = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _is_safe_member_name(member.name):
                continue
            target = (output_dir / member.name.replace("\\", "/")).resolve()
            if resolved_root != target and resolved_root not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted, target.open("wb") as sink:
                sink.write(extracted.read())


def iter_archive_pngs(archive_path: str | Path) -> list[str]:
    """Return sorted safe PNG member names inside a ZIP without extracting."""

    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".png")
            and _is_safe_member_name(info.filename)
            and not is_appledouble_path(info.filename)
        ]
    return sorted(names)

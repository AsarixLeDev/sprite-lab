"""Safe archive extraction for harvested packs."""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any


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
    selected = [
        info for info in safe_files if _member_selected(info.filename, include_member_globs, exclude_member_globs)
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


def _extract_zip(archive_path: Path, output_dir: Path, includes: Sequence[str], excludes: Sequence[str]) -> None:
    resolved_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not _is_safe_member_name(info.filename):
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
            if not info.is_dir() and info.filename.lower().endswith(".png") and _is_safe_member_name(info.filename)
        ]
    return sorted(names)

"""Safe archive extraction for harvested packs."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def extract_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
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
        _extract_zip(archive_path, output_dir)
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


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    resolved_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not _is_safe_member_name(info.filename):
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
        ]
    return sorted(names)

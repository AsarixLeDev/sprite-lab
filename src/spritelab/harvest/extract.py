"""PNG candidate discovery from extracted pack directories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from spritelab.harvest.download import compute_sha256
from spritelab.harvest.sources import SourceRecord


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
    status: str = "candidate"
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


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
) -> list[HarvestCandidate]:
    """Discover PNG candidates in deterministic relative-path order."""

    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"candidate root is not a directory: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and path.suffix.lower() == ".png"
            and (include_hidden or not _is_hidden(path, root))
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )

    candidates: list[HarvestCandidate] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError) as exc:
            image_sha256 = _safe_sha256(path)
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
                    rejection_reasons=(f"could not load PNG: {exc}",),
                )
            )
            continue
        image_sha256 = compute_sha256(path)
        candidates.append(
            HarvestCandidate(
                candidate_id=make_candidate_id(source.source_id, relative, image_sha256),
                source_id=source.source_id,
                source_path=source.source_url or source.local_archive_path or source.local_root_path,
                extracted_path=path,
                relative_path=relative,
                image_sha256=image_sha256,
                width=width,
                height=height,
                mode=mode,
            )
        )
    return candidates


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


def _safe_sha256(path: Path) -> str:
    try:
        return compute_sha256(path)
    except OSError:
        return ""

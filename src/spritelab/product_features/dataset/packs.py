"""Group discovered images into independently licensed source packs."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.dataset_v5.raw_inventory import file_sha256
from spritelab.product_features.dataset.evidence import evidence_for_image

PACK_DETECTION_SCHEMA = "spritelab.dataset.pack_detection.v1"
STRUCTURED_EVIDENCE_FILENAMES = ("source.yaml", "source.yml", "license.yaml", "license.yml", "metadata.yaml")
TEXT_EVIDENCE_FILENAMES = (
    "source.txt",
    "license",
    "license.txt",
    "copying",
    "readme",
    "readme.txt",
    "credits.txt",
    "attribution.txt",
)
EVIDENCE_FILENAMES = STRUCTURED_EVIDENCE_FILENAMES + TEXT_EVIDENCE_FILENAMES
SOURCE_EVIDENCE_FILENAMES = ("source.yaml", "source.yml", "source.txt", "metadata.yaml")
LICENSE_EVIDENCE_FILENAMES = ("license.yaml", "license.yml", "license", "license.txt", "copying")
SOURCE_PRESETS = ("opengameart", "kenney", "other_downloaded", "my_original_work", "custom_private")
_STRUCTURAL_DIRECTORY_NAMES = frozenset({"images", "image", "png", "pngs", "sprites", "assets"})


@dataclass
class SourcePack:
    """One independently licensed group of images beneath the input root."""

    pack_id: str
    relative_root: str
    boundary_evidence: str
    boundary_status: str
    image_relative_paths: list[str] = field(default_factory=list)
    evidence_files: list[dict[str, Any]] = field(default_factory=list)
    archive: dict[str, Any] | None = None
    prefill: dict[str, Any] = field(default_factory=dict)
    proposed_children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PACK_DETECTION_SCHEMA,
            "pack_id": self.pack_id,
            "relative_root": self.relative_root,
            "boundary_evidence": self.boundary_evidence,
            "boundary_status": self.boundary_status,
            "image_count": len(self.image_relative_paths),
            "image_relative_paths": list(self.image_relative_paths),
            "evidence_files": list(self.evidence_files),
            "archive": dict(self.archive) if self.archive else None,
            "prefill": dict(self.prefill),
            "proposed_children": list(self.proposed_children),
        }


def pack_id_for_relative_root(input_root: Path, relative_root: str) -> str:
    """Return an identity that cannot collide with the same layout in another folder."""

    identity = f"{os.path.normcase(str(input_root.resolve()))}\0{relative_root}"
    return "pack_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def detect_packs(
    root: Path,
    image_paths: Sequence[Path],
    *,
    user_grouping: Mapping[str, Any] | None = None,
) -> list[SourcePack]:
    """Group images into packs using explicit evidence before conservative inference."""

    root = root.resolve()
    confirmed_roots = _confirmed_roots(user_grouping)
    assignments: dict[str, list[str]] = {}
    boundary_kind: dict[str, str] = {}
    for path in image_paths:
        relative = path.relative_to(root).as_posix()
        pack_root, kind = _pack_root_for_image(path.parent, root, confirmed_roots)
        key = pack_root.relative_to(root).as_posix() if pack_root != root else "."
        assignments.setdefault(key, []).append(relative)
        previous = boundary_kind.get(key)
        boundary_kind[key] = _strongest_boundary(previous, kind)
    packs: list[SourcePack] = []
    for key in sorted(assignments):
        pack_root = root if key == "." else root / key
        first_image = root / assignments[key][0]
        evidence_files = _evidence_files(pack_root, root, first_image)
        archive = _archive_boundary(pack_root, root)
        status, children = _boundary_status(pack_root, root, boundary_kind[key], assignments[key], confirmed_roots)
        packs.append(
            SourcePack(
                pack_id=pack_id_for_relative_root(root, key),
                relative_root=key,
                boundary_evidence=boundary_kind[key],
                boundary_status=status,
                image_relative_paths=sorted(assignments[key]),
                evidence_files=evidence_files,
                archive=archive,
                prefill=_prefill(pack_root, root, evidence_files, first_image),
                proposed_children=children,
            )
        )
    return packs


def _confirmed_roots(user_grouping: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(user_grouping, Mapping):
        return set()
    roots = user_grouping.get("confirmed_pack_roots")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        return set()
    cleaned = set()
    for value in roots:
        text = str(value).strip().replace("\\", "/").strip("/")
        if text and ".." not in text.split("/"):
            cleaned.add(text)
        elif text in ("", "."):
            cleaned.add(".")
    return cleaned


def _pack_root_for_image(directory: Path, root: Path, confirmed: set[str]) -> tuple[Path, str]:
    directory = directory.resolve()
    evidence = _nearest_evidence_boundary(directory, root)
    if evidence is not None:
        return evidence
    archive = _nearest_archive_boundary(directory, root)
    if archive is not None:
        return archive, "original_archive_boundary"
    confirmed_root = _nearest_confirmed(directory, root, confirmed)
    if confirmed_root is not None:
        return confirmed_root, "explicit_user_grouping"
    return _inferred_root(directory, root), "conservative_directory_root_inference"


def _nearest_evidence_boundary(directory: Path, root: Path) -> tuple[Path, str] | None:
    """Choose the nearest evidence-bearing directory, then classify its evidence strength.

    A structured file at a shared parent must never swallow a nearer independently
    licensed child. Within the same directory, structured evidence has priority.
    """

    structured = {name.casefold() for name in STRUCTURED_EVIDENCE_FILENAMES}
    textual = {name.casefold() for name in TEXT_EVIDENCE_FILENAMES}
    current = directory
    while True:
        try:
            names = {entry.name.casefold() for entry in current.iterdir() if entry.is_file()}
        except OSError:
            names = set()
        if names & structured:
            return current, "explicit_structured_evidence"
        if names & textual:
            return current, "nearest_source_license_evidence"
        if current == root or root not in current.parents:
            return None
        current = current.parent


def _nearest_with(directory: Path, root: Path, names: Sequence[str]) -> Path | None:
    wanted = {name.casefold() for name in names}
    current = directory
    while True:
        try:
            if any(entry.is_file() and entry.name.casefold() in wanted for entry in current.iterdir()):
                return current
        except OSError:
            pass
        if current == root:
            return None
        if root not in current.parents:
            return None
        current = current.parent


def _nearest_archive_boundary(directory: Path, root: Path) -> Path | None:
    current = directory
    while current != root and root in current.parents:
        if _sibling_archive(current) is not None:
            return current
        current = current.parent
    return None


def _sibling_archive(directory: Path) -> Path | None:
    for suffix in (".zip",):
        candidate = directory.parent / f"{directory.name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _nearest_confirmed(directory: Path, root: Path, confirmed: set[str]) -> Path | None:
    if not confirmed:
        return None
    current = directory
    while True:
        key = current.relative_to(root).as_posix() if current != root else "."
        if key in confirmed:
            return current
        if current == root:
            return None
        current = current.parent


def _inferred_root(directory: Path, root: Path) -> Path:
    if directory == root:
        return root
    relative = directory.relative_to(root)
    top = root / relative.parts[0]
    if top.name.casefold() in _STRUCTURAL_DIRECTORY_NAMES:
        return root
    return top


def _strongest_boundary(previous: str | None, new: str) -> str:
    order = (
        "explicit_structured_evidence",
        "nearest_source_license_evidence",
        "original_archive_boundary",
        "explicit_user_grouping",
        "conservative_directory_root_inference",
    )
    if previous is None:
        return new
    return previous if order.index(previous) <= order.index(new) else new


def _boundary_status(
    pack_root: Path,
    root: Path,
    boundary_evidence: str,
    image_relatives: Sequence[str],
    confirmed: set[str],
) -> tuple[str, list[str]]:
    """Inferred multi-subfolder groupings are ambiguous until the user confirms them."""

    if boundary_evidence != "conservative_directory_root_inference":
        return "confirmed", []
    key = pack_root.relative_to(root).as_posix() if pack_root != root else "."
    if key in confirmed:
        return "confirmed", []
    prefix = "" if key == "." else key + "/"
    subdirs = set()
    direct = 0
    for relative in image_relatives:
        remainder = relative[len(prefix) :]
        parts = remainder.split("/")
        if len(parts) == 1:
            direct += 1
        elif parts[0].casefold() not in _STRUCTURAL_DIRECTORY_NAMES:
            subdirs.add(parts[0])
    if direct == 0 and len(subdirs) >= 2:
        children = sorted(f"{prefix}{name}" for name in subdirs)
        return "needs_confirmation", children
    return "confirmed", []


def _evidence_files(pack_root: Path, root: Path, image_path: Path) -> list[dict[str, Any]]:
    wanted = {name.casefold() for name in EVIDENCE_FILENAMES}
    rows = []
    try:
        entries = sorted(pack_root.iterdir(), key=lambda entry: entry.name.casefold())
    except OSError:
        entries = []
    for entry in entries:
        if entry.is_file() and entry.name.casefold() in wanted:
            rows.append(
                {
                    "name": entry.name,
                    "relative_path": entry.relative_to(root).as_posix(),
                    "role": _evidence_role(entry.name),
                    "sha256": file_sha256(entry),
                    "byte_length": entry.stat().st_size,
                }
            )
    seen = {str(row["relative_path"]) for row in rows}
    source, license_record = evidence_for_image(image_path, root)
    for role, aggregate in (("source", source), ("license", license_record)):
        records = aggregate.get("evidence_records")
        candidates = records if isinstance(records, list) else [aggregate]
        for record in candidates:
            if not isinstance(record, Mapping):
                continue
            relative = str(record.get("path") or "")
            if not relative or relative in seen:
                continue
            path = root / relative
            if not path.is_file():
                continue
            rows.append(
                {
                    "name": path.name,
                    "relative_path": relative,
                    "role": role,
                    "sha256": file_sha256(path),
                    "byte_length": path.stat().st_size,
                    "inherited": True,
                }
            )
            seen.add(relative)
    return sorted(rows, key=lambda row: str(row["relative_path"]).casefold())


def _evidence_role(name: str) -> str:
    folded = name.casefold()
    if folded in {value.casefold() for value in SOURCE_EVIDENCE_FILENAMES}:
        return "source"
    if folded in {value.casefold() for value in LICENSE_EVIDENCE_FILENAMES}:
        return "license"
    return "supporting"


def _archive_boundary(pack_root: Path, root: Path) -> dict[str, Any] | None:
    if pack_root == root:
        return None
    archive = _sibling_archive(pack_root)
    if archive is None:
        return None
    return {
        "relative_path": archive.relative_to(root).as_posix(),
        "sha256": file_sha256(archive),
        "byte_length": archive.stat().st_size,
        "mutated": False,
    }


def _prefill(
    pack_root: Path,
    root: Path,
    evidence_files: Sequence[Mapping[str, Any]],
    image_path: Path,
) -> dict[str, Any]:
    """Prefill only from explicit evidence; never assume a license from a platform."""

    supporting = [root / str(row["relative_path"]) for row in evidence_files if row["role"] == "supporting"]
    source, license_record = evidence_for_image(image_path, root)
    source_path = root / str(source["path"]) if source.get("path") else None
    combined_text = "\n".join(_safe_read(path) for path in ([source_path] if source_path else []) + supporting)
    preset = _detect_preset(combined_text + "\n" + str(source.get("source_url") or ""))
    supporting_urls = _urls(combined_text)
    attribution = _supporting_text(supporting, ("attribution", "credits"))
    source_conflicts = set(source.get("conflicting_fields") or ())
    license_conflict = bool(license_record.get("conflict"))
    prefill: dict[str, Any] = {
        "source_type": preset,
        "platform_name": {"opengameart": "OpenGameArt", "kenney": "Kenney"}.get(preset or ""),
        "creator_or_rights_holder": None if "creator" in source_conflicts else source.get("creator"),
        "pack_title": None if "source_name" in source_conflicts else source.get("source_name"),
        "source_page_url": (
            None
            if "source_url" in source_conflicts
            else source.get("source_url") or (supporting_urls[0] if supporting_urls else None)
        ),
        "license_identifier": (
            license_record.get("license") if license_record.get("present") and not license_conflict else None
        ),
        "license_url": None if license_conflict else license_record.get("license_url"),
        "license_evidence_file": None if license_conflict else license_record.get("path"),
        "attribution_text": attribution,
        "license_assumed_from_platform": False,
        "folder_name": pack_root.name if pack_root != root else root.name,
    }
    return {key: value for key, value in prefill.items() if value not in (None, "")}


def _detect_preset(text: str) -> str | None:
    folded = text.casefold()
    if "opengameart.org" in folded:
        return "opengameart"
    if "kenney.nl" in folded or "kenney" in folded:
        return "kenney"
    return None


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")[:20000]
    except OSError:
        return ""


def _urls(text: str) -> list[str]:
    import re

    return [value.rstrip(".,;)") for value in re.findall(r"https?://[^\s<>\"]+", text, re.IGNORECASE)]


def _supporting_text(paths: Sequence[Path], stems: Sequence[str]) -> str | None:
    wanted = {stem.casefold() for stem in stems}
    for path in paths:
        if path.stem.casefold() in wanted:
            value = _safe_read(path).strip()
            if value:
                return value[:4000]
    return None

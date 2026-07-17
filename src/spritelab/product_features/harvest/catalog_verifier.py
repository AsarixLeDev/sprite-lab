"""Live code identity for trusted Harvest catalog evidence validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from spritelab.utils.safe_fs import AnchoredDirectory, open_anchored_directory

CATALOG_EVIDENCE_VERIFIER_ID = "spritelab.catalog-evidence-v2"
_MAX_VERIFIER_MODULE_BYTES = 1 << 20
_VERIFIER_MODULES = (
    ("spritelab.harvest.download", Path("harvest") / "download.py"),
    ("spritelab.product_core.events", Path("product_core") / "events.py"),
    ("spritelab.product_features.harvest.catalog", Path("product_features") / "harvest" / "catalog.py"),
    (
        "spritelab.product_features.harvest.catalog_verifier",
        Path("product_features") / "harvest" / "catalog_verifier.py",
    ),
    (
        "spritelab.product_features.harvest.catalog_writer",
        Path("product_features") / "harvest" / "catalog_writer.py",
    ),
    (
        "spritelab.product_features.harvest.evidence_fetch",
        Path("product_features") / "harvest" / "evidence_fetch.py",
    ),
    (
        "spritelab.product_features.harvest.onboarding",
        Path("product_features") / "harvest" / "onboarding.py",
    ),
    ("spritelab.product_features.harvest.storage", Path("product_features") / "harvest" / "storage.py"),
    ("spritelab.utils.safe_fs", Path("utils") / "safe_fs.py"),
)


def catalog_evidence_verifier_code_identity() -> str:
    """Hash the exact live modules that validate catalog attestations."""

    spritelab_root = Path(os.path.abspath(__file__)).parents[2]
    modules: list[dict[str, object]] = []
    for module_name, relative_path in _VERIFIER_MODULES:
        path = spritelab_root / relative_path
        with open_anchored_directory(path.parent, spritelab_root) as anchor:
            payload = _read_verifier_module(anchor, path.name)
            modules.append(
                {
                    "module": module_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                }
            )
    encoded = json.dumps(
        {
            "schema_version": "spritelab.harvest.catalog-evidence-verifier-code.v2",
            "modules": modules,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_verifier_module(anchor: AnchoredDirectory, name: str) -> bytes:
    before = anchor.lstat(name)
    _validate_module(before)
    descriptor = anchor.open_file(
        name,
        os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
    )
    try:
        opened = os.fstat(descriptor)
        _validate_module(opened)
        if _module_identity(before) != _module_identity(opened):
            raise ValueError("Harvest catalog verifier module changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_VERIFIER_MODULE_BYTES + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if (
        len(payload) != before.st_size
        or len(payload) > _MAX_VERIFIER_MODULE_BYTES
        or _module_identity(before) != _module_identity(opened_after)
        or _module_identity(before) != _module_identity(path_after)
    ):
        raise ValueError("Harvest catalog verifier module changed while reading.")
    return payload


def _validate_module(metadata: os.stat_result) -> None:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse_flag)
        or metadata.st_nlink != 1
        or not 1 <= metadata.st_size <= _MAX_VERIFIER_MODULE_BYTES
    ):
        raise ValueError("Harvest catalog verifier module is unsafe.")


def _module_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


__all__ = [
    "CATALOG_EVIDENCE_VERIFIER_ID",
    "catalog_evidence_verifier_code_identity",
]

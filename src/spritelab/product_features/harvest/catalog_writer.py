"""Transactional publication for promoted Harvest trusted-catalog sources."""

from __future__ import annotations

import os
import stat
import uuid
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import (
    MAX_TRUSTED_CATALOG_BYTES,
    TRUSTED_CATALOG_RELATIVE_PATH,
    HarvestSource,
    TrustedCatalogSnapshot,
    load_trusted_catalog,
    load_trusted_catalog_snapshot,
    trusted_catalog_record,
)
from spritelab.product_features.harvest.storage import RepositoryMutationLock
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, require_confined_path


class CatalogPromotionError(ValueError):
    """A promoted source conflicts with or cannot safely update the catalog."""


def publish_trusted_catalog_source(
    project_root: str | Path,
    lock_root: str | Path,
    source: HarvestSource,
    *,
    lock_held: bool = False,
) -> tuple[tuple[HarvestSource, ...], bool]:
    """Compare and atomically publish one source under the Harvest mutation lock.

    Identical replay is a no-op. A reused ``source_id`` with any different
    content fails closed. The first catalog publication uses an exclusive
    no-replace rename so a target that appears concurrently is never replaced.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    lock = require_confined_path(Path(lock_root), root, allow_root=True)
    lock_context: Any = nullcontext() if lock_held else RepositoryMutationLock(lock)
    with lock_context:
        existing, existing_snapshot = load_trusted_catalog_snapshot(root)
        by_id = {item.source_id: item for item in existing}
        prior = by_id.get(source.source_id)
        if prior is not None:
            if prior.catalog_identity == source.catalog_identity:
                return existing, False
            raise CatalogPromotionError("Harvest source_id already belongs to a different trusted catalog record.")
        sources = tuple(sorted((*existing, source), key=lambda item: item.source_id))
        payload = (
            strict_json_dumps(trusted_catalog_record(sources), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_TRUSTED_CATALOG_BYTES:
            raise CatalogPromotionError("Harvest trusted catalog would exceed its bounded size.")
        _publish_catalog_bytes(root, payload, expected_snapshot=existing_snapshot)
        reloaded = load_trusted_catalog(root)
        if tuple(item.catalog_identity for item in reloaded) != tuple(item.catalog_identity for item in sources):
            raise CatalogPromotionError("Harvest trusted catalog did not retain the promoted source exactly.")
        return reloaded, True


def _publish_catalog_bytes(
    root: Path,
    payload: bytes,
    *,
    expected_snapshot: TrustedCatalogSnapshot | None,
) -> None:
    parts = TRUSTED_CATALOG_RELATIVE_PATH.parts
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in parts[:-1]:
            anchor.mkdir(part, exist_ok=True)
            anchor = stack.enter_context(anchor.open_directory(part))
        target = parts[-1]
        before: os.stat_result | None = None
        if anchor.lexists(target):
            before = anchor.lstat(target)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > MAX_TRUSTED_CATALOG_BYTES
            ):
                raise CatalogPromotionError("Harvest trusted catalog target is unsafe.")
        elif expected_snapshot is not None:
            raise CatalogPromotionError("Harvest trusted catalog disappeared during promotion.")

        temporary = f".trusted-catalog-{uuid.uuid4().hex}.tmp"
        descriptor = anchor.open_file(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if not identity.matches(anchor.lstat(temporary)):
                raise CatalogPromotionError("Harvest catalog temporary file changed before publication.")
            if before is None:
                if expected_snapshot is not None:
                    raise CatalogPromotionError("Harvest trusted catalog disappeared during promotion.")
                if anchor.lexists(target):
                    raise CatalogPromotionError("Harvest catalog target appeared during first publication.")
                anchor.rename(temporary, target, replace=False)
            else:
                if expected_snapshot is None:
                    raise CatalogPromotionError("Harvest catalog target appeared during first publication.")
                current_payload, current = _read_bound_catalog(anchor, target)
                if not expected_snapshot.matches(current_payload, current):
                    raise CatalogPromotionError(
                        "Harvest trusted catalog bytes or inode changed after they were parsed."
                    )
                anchor.rename(temporary, target, replace=True)
            if not identity.matches(anchor.lstat(target)):
                raise CatalogPromotionError("Harvest trusted catalog publication changed inode identity.")
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            anchor.unlink_if_owned(temporary, identity)
            raise


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_bound_catalog(anchor: AnchoredDirectory, name: str) -> tuple[bytes, os.stat_result]:
    before = anchor.lstat(name)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= MAX_TRUSTED_CATALOG_BYTES
    ):
        raise CatalogPromotionError("Harvest trusted catalog target is unsafe.")
    descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise CatalogPromotionError("Harvest trusted catalog changed while reopening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_TRUSTED_CATALOG_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if (
        len(payload) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(path_after)
    ):
        raise CatalogPromotionError("Harvest trusted catalog changed during comparison.")
    return payload, after


__all__ = ["CatalogPromotionError", "publish_trusted_catalog_source"]

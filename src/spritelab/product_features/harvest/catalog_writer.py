"""Transactional publication for promoted Harvest trusted-catalog sources."""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import (
    MAX_TRUSTED_CATALOG_BYTES,
    MAX_TRUSTED_CATALOG_SOURCES,
    TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH,
    HarvestSource,
    load_trusted_catalog,
    trusted_catalog_source_document,
    trusted_catalog_source_filename,
)
from spritelab.product_features.harvest.storage import RepositoryMutationLock
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    require_confined_path,
)


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
    content fails closed. Every source is one immutable append-only record;
    promotion never replaces a prior catalog file.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    lock = require_confined_path(Path(lock_root), root, allow_root=True)
    lock_context: Any = nullcontext() if lock_held else RepositoryMutationLock(lock)
    with lock_context:
        existing = load_trusted_catalog(root)
        by_id = {item.source_id: item for item in existing}
        prior = by_id.get(source.source_id)
        if prior is not None:
            if prior.catalog_identity == source.catalog_identity:
                return existing, False
            raise CatalogPromotionError("Harvest source_id already belongs to a different trusted catalog record.")
        sources = tuple(sorted((*existing, source), key=lambda item: item.source_id))
        if len(sources) > MAX_TRUSTED_CATALOG_SOURCES:
            raise CatalogPromotionError("Harvest trusted catalog source count would exceed its bound.")
        payload = (
            strict_json_dumps(
                trusted_catalog_source_document(source),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_TRUSTED_CATALOG_BYTES:
            raise CatalogPromotionError("Harvest trusted catalog source record would exceed its bounded size.")
        aggregate_bytes = sum(
            len(
                (
                    strict_json_dumps(
                        trusted_catalog_source_document(item),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
            for item in sources
        )
        if aggregate_bytes > MAX_TRUSTED_CATALOG_BYTES:
            raise CatalogPromotionError("Harvest trusted catalog would exceed its aggregate byte bound.")
        try:
            _publish_catalog_record(root, source.source_id, payload)
        except (OSError, UnsafeFilesystemOperation) as exc:
            raise CatalogPromotionError("Harvest append-only catalog record could not be published exactly.") from exc
        reloaded = load_trusted_catalog(root)
        if tuple(item.catalog_identity for item in reloaded) != tuple(item.catalog_identity for item in sources):
            raise CatalogPromotionError("Harvest trusted catalog did not retain the promoted source exactly.")
        return reloaded, True


def _publish_catalog_record(
    root: Path,
    source_id: str,
    payload: bytes,
) -> None:
    parts = TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH.parts
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in parts:
            anchor.mkdir(part, exist_ok=True)
            anchor = stack.enter_context(anchor.open_directory(part))
        target = trusted_catalog_source_filename(source_id)
        if anchor.lexists(target):
            raise CatalogPromotionError("Harvest append-only catalog record appeared during promotion.")

        temporary = f".stage-{uuid.uuid4().hex}"
        descriptor = anchor.open_file(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if not identity.matches(anchor.lstat(temporary)):
                raise CatalogPromotionError("Harvest catalog temporary file changed before publication.")
            anchor.publish_held_file_no_replace(descriptor, temporary, target, identity=identity)
            if not identity.matches(anchor.lstat(target)):
                raise CatalogPromotionError("Harvest trusted catalog publication changed inode identity.")
        finally:
            os.close(descriptor)


__all__ = ["CatalogPromotionError", "publish_trusted_catalog_source"]

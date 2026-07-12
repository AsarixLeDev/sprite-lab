"""Auto-Labeling v3: per-stage content-addressed cache with scoped invalidation.

The whole point of this cache is that a change to one dependency invalidates
*only* the stages that consumed it. A cache key is built from an explicit,
minimal set of dependency hashes — so changing a VLM prompt invalidates VLM
stage outputs but not deterministic evidence, and changing the taxonomy
invalidates fusion but not the blind VLM descriptor.

Each stage declares which dependencies it consumes via ``STAGE_DEPENDENCIES``.
Building a key with an irrelevant dependency is a programming error and is
rejected, which makes "cache key omits a relevant dependency" a testable
property rather than a silent bug.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v3.sha256_utils import sha256_short

logger = logging.getLogger(__name__)


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    exception: BaseException | None = None


_FLIGHTS_LOCK = threading.Lock()
_FLIGHTS: dict[tuple[str, str], _Flight] = {}

# The full universe of dependency names a cache key may reference.
ALL_DEPENDENCIES: tuple[str, ...] = (
    "input_content_hash",
    "stage_version",
    "model_identity",
    "prompt_hash",
    "image_view",
    "preprocessing_hash",
    "context_hash",
    "taxonomy_hash",
    "source_profiles_hash",
    "sheet_mapping_hash",
    "policy_hash",
)

# What each stage legitimately depends on. Anything outside this set must NOT be
# part of that stage's cache key (else unrelated changes cause spurious misses).
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "deterministic_evidence": (
        "input_content_hash",
        "stage_version",
        "source_profiles_hash",
        "sheet_mapping_hash",
        "preprocessing_hash",
    ),
    "vlm_blind_descriptor": (
        "input_content_hash",
        "stage_version",
        "model_identity",
        "prompt_hash",
        "image_view",
        "preprocessing_hash",
    ),
    "vlm_constrained_classification": (
        "input_content_hash",
        "stage_version",
        "model_identity",
        "prompt_hash",
        "image_view",
        "taxonomy_hash",
        "preprocessing_hash",
    ),
    "vlm_morphology": (
        "input_content_hash",
        "stage_version",
        "model_identity",
        "prompt_hash",
        "image_view",
        "preprocessing_hash",
    ),
    "vlm_verification": (
        "input_content_hash",
        "stage_version",
        "model_identity",
        "prompt_hash",
        "image_view",
        "taxonomy_hash",
        "preprocessing_hash",
    ),
    "vlm_context": (
        "input_content_hash",
        "stage_version",
        "model_identity",
        "prompt_hash",
        "image_view",
        "taxonomy_hash",
        "preprocessing_hash",
        "context_hash",
    ),
    "description_enrichment": ("input_content_hash", "stage_version", "model_identity", "prompt_hash"),
    "fusion": (
        "input_content_hash",
        "stage_version",
        "taxonomy_hash",
        "policy_hash",
    ),
}


class UnknownStageError(KeyError):
    pass


class IrrelevantDependencyError(ValueError):
    pass


class MissingDependencyError(ValueError):
    pass


def stage_cache_key(stage: str, dependencies: Mapping[str, str]) -> str:
    """Build a content-addressed key for ``stage`` from its declared deps only.

    Raises if a dependency is supplied that the stage does not declare
    (irrelevant → would over-invalidate) or if a declared dependency is missing
    (would under-invalidate — a silent correctness hole).
    """
    if stage not in STAGE_DEPENDENCIES:
        raise UnknownStageError(stage)
    declared = STAGE_DEPENDENCIES[stage]

    extra = set(dependencies) - set(declared)
    if extra:
        raise IrrelevantDependencyError(
            f"stage {stage!r} does not depend on {sorted(extra)}; including it would over-invalidate"
        )
    missing = [d for d in declared if not dependencies.get(d)]
    if missing:
        raise MissingDependencyError(f"stage {stage!r} cache key missing dependencies: {missing}")

    payload = {"stage": stage, "deps": {d: str(dependencies[d]) for d in declared}}
    canonical = json.dumps(payload, sort_keys=True)
    return sha256_short(canonical, length=24)


def record_content_hash(record: Mapping[str, Any]) -> str:
    """Content hash of a record's inputs (identity + provenance + metadata)."""
    return sha256_short(json.dumps(record, sort_keys=True, default=str), length=24)


def record_decision_cache_key(
    *,
    input_content_hash: str,
    policy_hash: str,
    calibration_hash: str,
    stage_version: str,
) -> str:
    """Composite key for a fully-fused record decision.

    ``policy_hash`` already folds in taxonomy + impossible-combination hashes, so
    any config change that could alter the decision changes this key. Calibration
    is included separately because it gates acceptance.
    """
    payload = {
        "input_content_hash": input_content_hash,
        "policy_hash": policy_hash,
        "calibration_hash": calibration_hash,
        "stage_version": stage_version,
    }
    return sha256_short(json.dumps(payload, sort_keys=True), length=24)


class StageCache:
    """Filesystem cache with atomic writes and in-process single-flight.

    A small exclusive lock file also serializes writers across processes. The
    in-process flight shares both successful values and exceptions with callers
    that arrive while a key is being computed. Exceptions and ``None`` values
    are never written as successful cache entries.
    """

    def __init__(self, root: str | Path, *, lock_timeout_seconds: float = 120.0):
        self.root = Path(root)
        self.lock_timeout_seconds = max(1.0, float(lock_timeout_seconds))

    def _path(self, key: str) -> Path:
        # Deterministic 2-level fan-out avoids a single flat directory at scale.
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["value"]
        except Exception:
            return None

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump({"key": key, "value": value}, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, p)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _lock_path(self, key: str) -> Path:
        return self._path(key).with_suffix(".lock")

    def _acquire_process_lock(self, key: str) -> Path:
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock_path
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > self.lock_timeout_seconds
                    if stale:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for stage-cache lock: {key}") from None
                time.sleep(0.02)

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> tuple[Any, bool]:
        """Return ``(value, was_cache_hit)`` with one concurrent computation."""
        cached = self.get(key)
        if cached is not None:
            return cached, True

        flight_key = (str(self.root.resolve()), key)
        with _FLIGHTS_LOCK:
            flight = _FLIGHTS.get(flight_key)
            leader = flight is None
            if flight is None:
                flight = _Flight()
                _FLIGHTS[flight_key] = flight

        if not leader:
            flight.event.wait()
            if flight.exception is not None:
                raise flight.exception
            return flight.value, True

        lock_path: Path | None = None
        try:
            lock_path = self._acquire_process_lock(key)
            cached = self.get(key)
            if cached is not None:
                flight.value = cached
                return cached, True
            value = compute()
            if value is None:
                raise ValueError("stage cache refused to store a None result")
            self.put(key, value)
            flight.value = value
            return value, False
        except BaseException as exc:
            flight.exception = exc
            raise
        finally:
            if lock_path is not None:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
            flight.event.set()
            with _FLIGHTS_LOCK:
                _FLIGHTS.pop(flight_key, None)

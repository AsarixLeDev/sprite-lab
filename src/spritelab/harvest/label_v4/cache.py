"""Content-addressed, atomic, single-flight caches for Labeling v4."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

CACHE_SCHEMA_VERSION = "label_cache_v4.1"
BLIND_PROPOSAL_NAMESPACE = "blind_vlm_proposal_v4"
TEXT_RECONCILIATION_NAMESPACE = "text_reconciliation_v4"
INDEPENDENT_VERIFIER_NAMESPACE = "independent_verifier_v4"

_NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_FULL_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class CacheError(RuntimeError):
    pass


class CacheCorruptionError(CacheError):
    pass


class CacheIdentityMismatchError(CacheError):
    pass


class CacheCollisionError(CacheError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def canonical_request_hash(request: Mapping[str, Any] | str | bytes) -> str:
    """Hash the provider request body; authentication must not be included."""

    if isinstance(request, Mapping):
        request = _canonical_json(request)
    return sha256_hex(request)


def _rgba_digest(rgba: bytes, width: int, height: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"spritelab-exact-rgba-v1\0")
    digest.update(int(width).to_bytes(8, "big", signed=False))
    digest.update(int(height).to_bytes(8, "big", signed=False))
    digest.update(b"RGBA\0")
    digest.update(rgba)
    return digest.hexdigest()


def exact_image_content_hash(
    image: str | Path | bytes | bytearray | memoryview | Image.Image,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Hash exact decoded RGBA content, dimensions, and mode with full SHA-256.

    Encoded PNG metadata therefore cannot create a false cache miss, while any
    pixel or dimension change necessarily creates a new identity.  Raw RGBA
    bytes require explicit dimensions.
    """

    if isinstance(image, Image.Image):
        rgba = image.convert("RGBA")
        return _rgba_digest(rgba.tobytes(), rgba.width, rgba.height)
    if isinstance(image, (str, Path)):
        with Image.open(Path(image)) as opened:
            rgba = opened.convert("RGBA")
            return _rgba_digest(rgba.tobytes(), rgba.width, rgba.height)
    payload = bytes(image)
    if width is not None or height is not None:
        if width is None or height is None or width < 1 or height < 1:
            raise ValueError("raw RGBA hashing requires positive width and height")
        if len(payload) != width * height * 4:
            raise ValueError("raw RGBA byte length does not match width * height * 4")
        return _rgba_digest(payload, width, height)
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            rgba = opened.convert("RGBA")
            return _rgba_digest(rgba.tobytes(), rgba.width, rgba.height)
    except Exception as exc:
        raise ValueError("encoded image bytes are invalid; raw RGBA needs width and height") from exc


@dataclass(frozen=True)
class CacheIdentity:
    """Every dependency that can make an image/model output incompatible."""

    namespace: str
    stage: str
    image_hash: str
    model_identity: str
    prompt_version: str
    prompt_hash: str
    schema_version: str
    request_hash: str
    provider: str = ""
    cache_schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _NAMESPACE_PATTERN.fullmatch(self.namespace):
            raise ValueError(f"invalid cache namespace: {self.namespace!r}")
        if not self.stage or not self.model_identity or not self.prompt_version or not self.schema_version:
            raise ValueError("stage, model_identity, prompt_version, and schema_version are required")
        for name in ("image_hash", "prompt_hash", "request_hash"):
            if not _FULL_SHA256_PATTERN.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a full lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "cache_schema_version": self.cache_schema_version,
            "namespace": self.namespace,
            "stage": self.stage,
            "image_hash": self.image_hash,
            "model_identity": self.model_identity,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "schema_version": self.schema_version,
            "request_hash": self.request_hash,
        }

    @property
    def key(self) -> str:
        return sha256_hex(_canonical_json(self.to_dict()))


def make_cache_identity(
    *,
    namespace: str,
    stage: str,
    image: str | Path | bytes | bytearray | memoryview | Image.Image,
    model_identity: str,
    prompt_version: str,
    prompt: str,
    schema_version: str,
    request: Mapping[str, Any] | str | bytes,
    provider: str = "",
    width: int | None = None,
    height: int | None = None,
) -> CacheIdentity:
    return CacheIdentity(
        namespace=namespace,
        stage=stage,
        image_hash=exact_image_content_hash(image, width=width, height=height),
        model_identity=model_identity,
        provider=provider,
        prompt_version=prompt_version,
        prompt_hash=sha256_hex(prompt),
        schema_version=schema_version,
        request_hash=canonical_request_hash(request),
    )


def verifier_identity(**kwargs: Any) -> CacheIdentity:
    kwargs = dict(kwargs)
    kwargs["namespace"] = INDEPENDENT_VERIFIER_NAMESPACE
    return make_cache_identity(**kwargs)


def blind_proposal_identity(**kwargs: Any) -> CacheIdentity:
    kwargs = dict(kwargs)
    kwargs["namespace"] = BLIND_PROPOSAL_NAMESPACE
    return make_cache_identity(**kwargs)


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":  # pragma: no cover - branch depends on host OS
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class LabelV4Cache:
    """Immutable JSON cache with atomic writes and cross-process single-flight."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, identity: CacheIdentity) -> Path:
        return self.root / identity.namespace / identity.key[:2] / f"{identity.key}.json"

    def _lock_path(self, identity: CacheIdentity) -> Path:
        return self.root / ".locks" / identity.namespace / f"{identity.key}.lock"

    def _read(self, identity: CacheIdentity) -> Any | None:
        path = self.path_for(identity)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheCorruptionError(f"invalid cache entry: {path}") from exc
        if envelope.get("cache_key") != identity.key:
            raise CacheIdentityMismatchError(f"cache key mismatch: {path}")
        if envelope.get("identity") != identity.to_dict():
            raise CacheIdentityMismatchError(f"cache identity mismatch: {path}")
        if "value" not in envelope:
            raise CacheCorruptionError(f"cache entry has no value: {path}")
        return copy.deepcopy(envelope["value"])

    def get(self, identity: CacheIdentity) -> Any | None:
        return self._read(identity)

    def _write_locked(self, identity: CacheIdentity, value: Any) -> Any:
        path = self.path_for(identity)
        existing = self._read(identity)
        if existing is not None:
            if _canonical_json(existing) != _canonical_json(value):
                raise CacheCollisionError(f"immutable cache entry already has a different value: {path}")
            return existing
        envelope = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": identity.key,
            "identity": identity.to_dict(),
            "value": copy.deepcopy(value),
        }
        _atomic_write_json(path, envelope)
        return copy.deepcopy(value)

    def put(self, identity: CacheIdentity, value: Any) -> Any:
        if value is None:
            raise ValueError("None is not a successful cache value")
        path = self.path_for(identity)
        with _thread_lock(path):
            with _exclusive_process_lock(self._lock_path(identity)):
                return self._write_locked(identity, value)

    def get_or_compute(self, identity: CacheIdentity, compute: Callable[[], Any]) -> tuple[Any, bool]:
        """Return ``(value, cache_hit)``; only one caller computes a key."""

        cached = self._read(identity)
        if cached is not None:
            return cached, True
        path = self.path_for(identity)
        with _thread_lock(path):
            with _exclusive_process_lock(self._lock_path(identity)):
                cached = self._read(identity)
                if cached is not None:
                    return cached, True
                value = compute()
                if value is None:
                    raise ValueError("compute returned None; failure results are not cached as successes")
                return self._write_locked(identity, value), False


AtomicStageCache = LabelV4Cache

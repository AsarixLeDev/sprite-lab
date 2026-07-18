"""Durable one-use cloud action challenges for Training Start and Resume.

The browser receives the opaque token only once.  Durable state stores only its
SHA-256 identity and exact launch bindings.  Consumption is performed while
the shared activation/launch action lock is held, so concurrent or restarted
servers cannot reuse one confirmation at a different backend seam.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from spritelab.product_core import strict_json_dumps, strict_json_loads
from spritelab.training.campaign import stable_hash
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

CLOUD_CHALLENGE_SCHEMA: Final = "spritelab.training.cloud-action-challenge.v1"
CLOUD_CHALLENGE_CONSUMPTION_SCHEMA: Final = "spritelab.training.cloud-action-consumption.v1"
CLOUD_CHALLENGE_DIRECTORY: Final = ".spritelab-training-cloud-challenges"
CLOUD_CHALLENGE_TTL_SECONDS: Final = 120
CLOUD_CHALLENGE_BINDING_KEYS: Final = frozenset(
    {
        "action",
        "run_id",
        "campaign_identity_sha256",
        "backend_id",
        "backend_configuration_identity_sha256",
        "project_config_sha256",
        "activation_commit_record_identity",
        "launch_authorization_evidence_sha256",
    }
)

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "challenge_id",
        "token_sha256",
        "status",
        "bindings",
        "issued_at",
        "expires_at",
        "consumed_at",
        "operation_nonce",
        "paths_exposed",
        "record_authentication_sha256",
        "record_identity",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^cloud-[0-9a-f]{64}$")
_OPERATION_NONCE = re.compile(r"^operation-[0-9a-f]{32}$")


class CloudChallengeError(ValueError):
    """A cloud confirmation challenge failed its public trust contract."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


class CloudChallengeStore:
    """Issue and atomically consume path-free cloud confirmation records."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        ttl_seconds: int = CLOUD_CHALLENGE_TTL_SECONDS,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._clock = clock or (lambda: datetime.now(UTC))
        self.ttl_seconds = max(1, int(ttl_seconds))

    def issue(self, bindings: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one fresh challenge and return its otherwise unpersisted token."""

        normalized = _validate_bindings(bindings)
        issued = _utc(self._clock())
        expires = issued + timedelta(seconds=self.ttl_seconds)
        for _attempt in range(16):
            token = f"cloud-{secrets.token_hex(32)}"
            token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
            challenge_id = token_sha256
            base = {
                "schema_version": CLOUD_CHALLENGE_SCHEMA,
                "challenge_id": challenge_id,
                "token_sha256": token_sha256,
                "status": "ISSUED",
                "bindings": normalized,
                "issued_at": _format_timestamp(issued),
                "expires_at": _format_timestamp(expires),
                "consumed_at": None,
                "operation_nonce": None,
                "paths_exposed": False,
            }
            authenticated = {
                **base,
                "record_authentication_sha256": _authenticate_record(token, base),
            }
            record = {**authenticated, "record_identity": stable_hash(authenticated)}
            content = _canonical_bytes(record)
            try:
                with self._store(create=True) as store:
                    issued_filename = _issued_filename(challenge_id)
                    consumed_filename = _consumed_filename(challenge_id)
                    if store.lexists(issued_filename) or store.lexists(consumed_filename):
                        continue
                    _write_exclusive_immutable(store, issued_filename, content)
                    if store.lexists(consumed_filename):
                        continue
            except FileExistsError:
                continue
            return {
                "schema_version": CLOUD_CHALLENGE_SCHEMA,
                "challenge_token": token,
                "challenge_id": challenge_id,
                "expires_at": record["expires_at"],
                "bindings": normalized,
                "paths_exposed": False,
            }
        raise CloudChallengeError(
            "cloud_challenge_unavailable",
            "A fresh cloud confirmation challenge could not be allocated safely.",
        )

    def consume_locked(
        self,
        token: str,
        *,
        expected_bindings: Mapping[str, Any],
        operation_nonce: str,
    ) -> dict[str, Any]:
        """Consume one challenge while ``TrainingActionLock`` is held by the caller."""

        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            raise CloudChallengeError(
                "cloud_challenge_invalid",
                "The cloud confirmation challenge is missing or malformed.",
            )
        if not isinstance(operation_nonce, str) or _OPERATION_NONCE.fullmatch(operation_nonce) is None:
            raise CloudChallengeError(
                "cloud_challenge_operation_invalid",
                "The cloud confirmation operation binding is invalid.",
            )
        bindings = _validate_bindings(expected_bindings)
        challenge_id = hashlib.sha256(token.encode("ascii")).hexdigest()
        issued_filename = _issued_filename(challenge_id)
        consumed_filename = _consumed_filename(challenge_id)
        try:
            with self._store(create=False) as store:
                try:
                    consumed_marker_present = store.lexists(consumed_filename)
                except UnsafeFilesystemOperation:
                    consumed_marker_present = True
                if consumed_marker_present:
                    raise CloudChallengeError(
                        "cloud_challenge_replayed",
                        "The cloud confirmation challenge was already consumed.",
                    )
                content = _read_exact(store, issued_filename, max_bytes=128 * 1024)
                record = _validate_issued_record(strict_json_loads(content), token=token)
                if content != _canonical_bytes(record):
                    raise CloudChallengeError(
                        "cloud_challenge_invalid",
                        "The cloud confirmation challenge is not canonical.",
                    )
                if record["challenge_id"] != challenge_id or record["token_sha256"] != challenge_id:
                    raise CloudChallengeError(
                        "cloud_challenge_invalid",
                        "The cloud confirmation challenge identity is invalid.",
                    )
                if record["bindings"] != bindings:
                    raise CloudChallengeError(
                        "cloud_challenge_binding_mismatch",
                        "The cloud confirmation challenge does not match this exact action.",
                    )
                now = _utc(self._clock())
                issued_at = _parse_timestamp(record["issued_at"])
                if now < issued_at:
                    raise CloudChallengeError(
                        "cloud_challenge_invalid",
                        "The cloud confirmation challenge timestamp is invalid.",
                    )
                if now >= _parse_timestamp(record["expires_at"]):
                    raise CloudChallengeError(
                        "cloud_challenge_expired",
                        "The cloud confirmation challenge expired; confirm the current action again.",
                    )
                consumed_base = {
                    "schema_version": CLOUD_CHALLENGE_CONSUMPTION_SCHEMA,
                    "challenge_id": challenge_id,
                    "token_sha256": challenge_id,
                    "status": "CONSUMED",
                    "bindings": bindings,
                    "issued_at": record["issued_at"],
                    "expires_at": record["expires_at"],
                    "consumed_at": _format_timestamp(now),
                    "operation_nonce": operation_nonce,
                    "issued_record_sha256": hashlib.sha256(content).hexdigest(),
                    "issued_record_identity": record["record_identity"],
                    "issued_record_authentication_sha256": record["record_authentication_sha256"],
                    "paths_exposed": False,
                }
                consumed = {**consumed_base, "record_identity": stable_hash(consumed_base)}
                try:
                    _write_exclusive_immutable(store, consumed_filename, _canonical_bytes(consumed))
                except FileExistsError as exc:
                    raise CloudChallengeError(
                        "cloud_challenge_replayed",
                        "The cloud confirmation challenge was already consumed.",
                    ) from exc
                return consumed
        except FileNotFoundError as exc:
            raise CloudChallengeError(
                "cloud_challenge_invalid",
                "The cloud confirmation challenge is unavailable.",
            ) from exc
        except (OSError, UnsafeFilesystemOperation) as exc:
            raise CloudChallengeError(
                "cloud_challenge_unavailable",
                "The cloud confirmation challenge store is unavailable.",
            ) from exc

    def _store(self, *, create: bool) -> Any:
        return _ChallengeStoreContext(self.project_root, create=create)


class _ChallengeStoreContext:
    def __init__(self, project_root: Path, *, create: bool) -> None:
        self.project_root = project_root
        self.create = create
        self._root: AnchoredDirectory | None = None
        self._child_context: Any = None

    def __enter__(self) -> AnchoredDirectory:
        self._root = AnchoredDirectory(self.project_root, self.project_root)
        self._root.__enter__()
        try:
            if self.create:
                self._root.mkdir(CLOUD_CHALLENGE_DIRECTORY, exist_ok=True)
            self._child_context = self._root.open_directory(CLOUD_CHALLENGE_DIRECTORY)
            return self._child_context.__enter__()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self._child_context is not None:
                self._child_context.__exit__(exc_type, exc_value, traceback)
                self._child_context = None
        finally:
            if self._root is not None:
                self._root.__exit__(exc_type, exc_value, traceback)
                self._root = None


def _validate_bindings(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != CLOUD_CHALLENGE_BINDING_KEYS:
        raise CloudChallengeError(
            "cloud_challenge_binding_invalid",
            "The cloud confirmation challenge bindings are invalid.",
        )
    result: dict[str, str] = {}
    for key in sorted(CLOUD_CHALLENGE_BINDING_KEYS):
        item = value.get(key)
        if not isinstance(item, str) or not item or item != item.strip():
            raise CloudChallengeError(
                "cloud_challenge_binding_invalid",
                "The cloud confirmation challenge bindings are invalid.",
            )
        result[key] = item
    if result["action"] not in {"start", "resume"}:
        raise CloudChallengeError(
            "cloud_challenge_binding_invalid",
            "The cloud confirmation action is invalid.",
        )
    for key in (
        "campaign_identity_sha256",
        "backend_configuration_identity_sha256",
        "project_config_sha256",
        "activation_commit_record_identity",
        "launch_authorization_evidence_sha256",
    ):
        if _SHA256.fullmatch(result[key]) is None:
            raise CloudChallengeError(
                "cloud_challenge_binding_invalid",
                "The cloud confirmation challenge identities are invalid.",
            )
    return result


def _validate_issued_record(value: Any, *, token: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECORD_KEYS:
        raise CloudChallengeError("cloud_challenge_invalid", "The cloud confirmation challenge is invalid.")
    record = dict(value)
    identity = str(record.get("record_identity") or "")
    payload = dict(record)
    payload.pop("record_identity", None)
    if (
        record.get("schema_version") != CLOUD_CHALLENGE_SCHEMA
        or _SHA256.fullmatch(str(record.get("challenge_id") or "")) is None
        or record.get("token_sha256") != record.get("challenge_id")
        or record.get("status") != "ISSUED"
        or record.get("paths_exposed") is not False
        or _SHA256.fullmatch(str(record.get("record_authentication_sha256") or "")) is None
        or _SHA256.fullmatch(identity) is None
        or stable_hash(payload) != identity
    ):
        raise CloudChallengeError("cloud_challenge_invalid", "The cloud confirmation challenge is invalid.")
    _validate_bindings(record.get("bindings"))
    issued = _parse_timestamp(record.get("issued_at"))
    expires = _parse_timestamp(record.get("expires_at"))
    if expires <= issued:
        raise CloudChallengeError("cloud_challenge_invalid", "The cloud confirmation challenge expiry is invalid.")
    if record.get("consumed_at") is not None or record.get("operation_nonce") is not None:
        raise CloudChallengeError("cloud_challenge_invalid", "The cloud confirmation challenge is invalid.")
    authenticated = dict(record)
    authenticated.pop("record_identity", None)
    authentication = str(authenticated.pop("record_authentication_sha256", ""))
    if not hmac.compare_digest(authentication, _authenticate_record(token, authenticated)):
        raise CloudChallengeError("cloud_challenge_invalid", "The cloud confirmation challenge is invalid.")
    return record


def _write_exclusive_immutable(anchor: AnchoredDirectory, name: str, content: bytes) -> None:
    """Create one final-name record and retain it on every later failure.

    Once O_EXCL succeeds, even a short or otherwise invalid file is a durable
    deny marker.  It is deliberately never removed or replaced: a process
    failure can burn one challenge, but it cannot reopen that authority.
    """

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    descriptor = anchor.open_file_immovable(name, flags, 0o600)
    try:
        before = os.fstat(descriptor)
        identity = OwnedFileIdentity.from_stat(before)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != 0:
            raise UnsafeFilesystemOperation("cloud challenge publication descriptor is unsafe")
        _write_all(descriptor, content)
        os.fsync(descriptor)
        _verify_exclusive_publication(anchor, name, descriptor, identity, content)
        _sync_publication_directory(anchor)
        _verify_exclusive_publication(anchor, name, descriptor, identity, content)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    written = 0
    while written < len(content):
        count = os.write(descriptor, content[written:])
        if count <= 0:
            raise OSError("cloud challenge write made no progress")
        written += count


def _read_descriptor_exact(descriptor: int, *, max_bytes: int) -> bytes:
    content = b""
    while len(content) <= max_bytes:
        chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(content)))
        if not chunk:
            break
        content += chunk
    if len(content) > max_bytes:
        raise UnsafeFilesystemOperation("cloud challenge record is too large")
    return content


def _verify_exclusive_publication(
    anchor: AnchoredDirectory,
    name: str,
    descriptor: int,
    identity: OwnedFileIdentity,
    content: bytes,
) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    reread = _read_descriptor_exact(descriptor, max_bytes=len(content))
    after = os.fstat(descriptor)
    if (
        reread != content
        or not identity.matches(after)
        or after.st_nlink != 1
        or after.st_size != len(content)
        or not identity.matches(anchor.lstat(name))
    ):
        raise UnsafeFilesystemOperation("cloud challenge publication identity changed")


def _sync_publication_directory(anchor: AnchoredDirectory) -> None:
    if os.name == "nt":
        return
    # ``fixed_directory_path`` intentionally returns /proc/self/fd or /dev/fd,
    # whose final component is a kernel-owned descriptor symlink.
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(anchor.fixed_directory_path(), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exact(anchor: AnchoredDirectory, name: str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = anchor.open_file(name, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes:
            raise UnsafeFilesystemOperation("cloud challenge record is unsafe")
        content = b""
        while len(content) <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(content)))
            if not chunk:
                break
            content += chunk
        after = os.fstat(descriptor)
        if len(content) > max_bytes or not OwnedFileIdentity.from_stat(before).matches(after):
            raise UnsafeFilesystemOperation("cloud challenge record changed while read")
        if not OwnedFileIdentity.from_stat(before).matches(anchor.lstat(name)):
            raise UnsafeFilesystemOperation("cloud challenge path identity changed while read")
        return content
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (strict_json_dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _authenticate_record(token: str, value: Mapping[str, Any]) -> str:
    return hmac.new(token.encode("ascii"), _canonical_bytes(value), hashlib.sha256).hexdigest()


def _issued_filename(challenge_id: str) -> str:
    return f"{challenge_id}.json"


def _consumed_filename(challenge_id: str) -> str:
    return f"{challenge_id}.consumed.json"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CloudChallengeError("cloud_challenge_clock_invalid", "The cloud challenge clock is invalid.")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CloudChallengeError("cloud_challenge_invalid", "A cloud challenge timestamp is invalid.")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise CloudChallengeError("cloud_challenge_invalid", "A cloud challenge timestamp is invalid.") from exc


__all__ = [
    "CLOUD_CHALLENGE_BINDING_KEYS",
    "CLOUD_CHALLENGE_CONSUMPTION_SCHEMA",
    "CLOUD_CHALLENGE_DIRECTORY",
    "CLOUD_CHALLENGE_SCHEMA",
    "CLOUD_CHALLENGE_TTL_SECONDS",
    "CloudChallengeError",
    "CloudChallengeStore",
]

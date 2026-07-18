"""Safe reconstruction of ProductEvent-backed run views."""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from spritelab.product_core import (
    ProductEvent,
    ProductStatus,
    strict_json_dumps,
    strict_json_loads,
    validate_finite_json,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
EVENT_FILENAME = "events.jsonl"
LEGACY_EVENT_FILENAME = "product_events.jsonl"
LEGACY_MIGRATION_FILENAME = "event_stream_migration.json"
LEGACY_MIGRATION_SCHEMA_V2 = "spritelab.product.event-stream-migration.v2"
LEGACY_MIGRATION_SCHEMA = "spritelab.product.event-stream-migration.v3"
LEGACY_SOURCE_REMOVAL_POLICY = "may_be_removed_after_verified_migration"
MIGRATION_EVIDENCE_SCHEMA = "spritelab.product.event-migration-evidence.v1"
EVENT_HISTORY_ORIGIN_FILENAME = "event_history_origin.json"
EVENT_HISTORY_ORIGIN_SCHEMA = "spritelab.product.event-history-origin.v1"
EVENT_HISTORY_ORIGIN_NATIVE = "native"
EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY = "migrated_legacy"
EVENT_HISTORY_ORIGIN_STATES = frozenset({EVENT_HISTORY_ORIGIN_NATIVE, EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY})
EVENT_HISTORY_TRANSACTION_FILENAME = "event_history_transaction.json"
EVENT_HISTORY_TRANSACTION_SCHEMA = "spritelab.product.event-history-transaction.v1"
EVENT_HISTORY_TRANSACTION_MIGRATION = "migration_publish"
EVENT_HISTORY_TRANSACTION_APPEND = "append_event"
EVENT_HISTORY_TRANSACTION_OPERATIONS = frozenset(
    {EVENT_HISTORY_TRANSACTION_MIGRATION, EVENT_HISTORY_TRANSACTION_APPEND}
)
RUN_COMPLETION_MARKER_FILENAME = "run_completion_marker.json"
EVENT_APPEND_SURFACE_REPOSITORY = "event_repository"
EVENT_APPEND_SURFACE_RUN_STATE = "v3_run_state"
EVENT_APPEND_SURFACES = frozenset({EVENT_APPEND_SURFACE_REPOSITORY, EVENT_APPEND_SURFACE_RUN_STATE})
MAX_EVENT_ROW_BYTES = 1_000_000
MAX_EVENT_HISTORY_TRANSACTION_BYTES = (MAX_EVENT_ROW_BYTES * 2) + 65_536
RUN_STATE_SCHEMA = "spritelab.product.run-state.v1"
TERMINAL_STATUSES = {
    ProductStatus.COMPLETE.value,
    ProductStatus.FAILED.value,
    ProductStatus.BLOCKED.value,
    ProductStatus.NEEDS_REVIEW.value,
    "CANCELLED",
    "NOT_COMPARABLE",
    "STALE",
}
KEY_COMPONENT_SEPARATOR_PATTERN = r"(?:[ _-]+|(?-i:(?=[A-Z])))"
SENSITIVE_KEY_PATTERN = (
    rf"(?:[a-z0-9]+{KEY_COMPONENT_SEPARATOR_PATTERN})*"
    r"(?:api[ _-]*key|access[ _-]*(?:key|token)|auth(?:entication)?[ _-]*token|"
    r"authorization|proxy[ _-]*authorization|client[ _-]*secret|private[ _-]*key|password|passwd|"
    rf"passphrase|bearer|secret|token|credentials?|cookie|sig(?:nature)?)(?:{KEY_COMPONENT_SEPARATOR_PATTERN}[a-z0-9]+)*"
)
SECRET_PATTERN = re.compile(
    rf"(?i)\b({SENSITIVE_KEY_PATTERN})[\"']?(\s*[:=]\s*)"
    r"(?!\[redacted\])"
    r"(?:(?:[\"'])[^\"'\r\n]*(?:[\"'])|(?:(?:bearer|basic)\s+)?[^\s,;&}\]]+)"
)
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
BASIC_AUTH_PATTERN = re.compile(r"(?i)\bBasic\s+[^\s,;]+")
URL_CREDENTIAL_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^@\s/]+@")
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"(?i)\b(?:sk-[a-z0-9_-]{12,}|sk_(?:live|test)_[a-z0-9]{12,}|rpa_[a-z0-9]{8,}|"
        r"hf_[a-z0-9]{16,}|gh[pousr]_[a-z0-9_]{16,}|github_pat_[a-z0-9_]{16,}|"
        r"AIza[a-z0-9_-]{20,})\b"
    ),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b"),
)
PRIVATE_KEY_PEM_PATTERN = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----.*?"
    r"(?:-----END(?: [A-Z0-9]+)* PRIVATE KEY-----|\Z)",
    re.IGNORECASE | re.DOTALL,
)
WINDOWS_LOCAL_PATH_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/])[^\r\n,;]*")
FORWARD_UNC_LOCAL_PATH_PATTERN = re.compile(r"(?<![:/\w.])//[^/\s,;]+/[^\s,;]+")
POSIX_LOCAL_PATH_PATTERN = re.compile(r"(?<![/\w.])/(?!/)[^\s,;]+")
FILE_URI_PATTERN = re.compile(r"""(?i)\bfile://[^\s,;"'<>\[\]{}()]+""")
PUBLIC_BOOLEAN_METRIC_FIELDS = frozenset(
    {
        "activated",
        "benchmark_evidence",
        "cancel_available",
        "cancelled",
        "completion_validated",
        "downloaded",
        "eligible",
        "hash_verified",
        "immutable",
        "may_accrue_cost",
        "passed",
        "paths_exposed",
        "pause_available",
        "production_authorized",
        "promotion_evidence",
        "ready",
        "remote",
        "remote_identity_verified",
        "remote_resource_uncertain",
        "requires_confirmation",
        "resource_shutdown_verified",
        "resource_state_uncertain",
        "resumable",
        "resume_available",
        "safe",
        "safe_resume",
        "training_authorized",
        "trustworthy",
        "unsafe_resume_available",
        "validated",
    }
)
_PUBLIC_FIELD_ACRONYM_BOUNDARY_PATTERN = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_PUBLIC_FIELD_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PUBLIC_FIELD_SEPARATOR_PATTERN = re.compile(r"[^A-Za-z0-9]+")
_SENSITIVE_PUBLIC_KEY_TOKENS = frozenset(
    {
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "passwd",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
_SENSITIVE_PUBLIC_KEY_COMPOUNDS = (
    ("api", "key"),
    ("access", "key"),
    ("access", "token"),
    ("auth", "token"),
    ("authentication", "token"),
    ("proxy", "authorization"),
    ("client", "secret"),
    ("private", "key"),
)
_SENSITIVE_PUBLIC_KEY_COMPACT_ALIASES = frozenset(
    {
        "accesskey",
        "accesstoken",
        "authenticationtoken",
        "authtoken",
        "clientsecret",
        "privatekey",
        "proxyauthorization",
    }
)
_REPOSITORY_LOCKS: dict[str, threading.RLock] = {}
_REPOSITORY_LOCK_GUARD = threading.Lock()
_EVENT_FILE_LOCKS: dict[str, threading.RLock] = {}
_EVENT_FILE_LOCK_GUARD = threading.Lock()
_EVENT_FILE_LOCK_LOCAL = threading.local()


class LegacyEventMigrationError(ValueError):
    """A legacy event stream could not be promoted without losing identity."""

    def __init__(self, message: str, *, status: str = "NOT_COMPARABLE") -> None:
        super().__init__(message)
        self.status = status


def _require_mutable_event_history(directory: Path) -> None:
    """Freeze every event-history writer once any completion marker path exists."""

    marker = directory / RUN_COMPLETION_MARKER_FILENAME
    if os.path.lexists(marker):
        raise LegacyEventMigrationError(
            "Run completion marker freezes event history; event append and origin mutation are refused."
        )


class EventMigrationState(str, Enum):
    """Closed classification for canonical/legacy event-stream relationships."""

    NO_MIGRATION = "NO_MIGRATION"
    VERIFIED_SOURCE_PRESENT = "VERIFIED_SOURCE_PRESENT"
    VERIFIED_SOURCE_REMOVED = "VERIFIED_SOURCE_REMOVED"
    STALE_SOURCE_CHANGED = "STALE_SOURCE_CHANGED"
    INVALID_RECORD = "INVALID_RECORD"
    INVALID_CANONICAL_PREFIX = "INVALID_CANONICAL_PREFIX"
    CONFLICTING_STREAMS = "CONFLICTING_STREAMS"
    NOT_COMPARABLE = "NOT_COMPARABLE"


VERIFIED_MIGRATION_STATES = frozenset(
    {
        EventMigrationState.VERIFIED_SOURCE_PRESENT,
        EventMigrationState.VERIFIED_SOURCE_REMOVED,
    }
)
RESUME_COMPATIBLE_MIGRATION_STATES = frozenset({EventMigrationState.NO_MIGRATION, *VERIFIED_MIGRATION_STATES})


@dataclass(frozen=True)
class EventMigrationVerification:
    state: EventMigrationState
    run_id: str
    evidence_sha256: str
    message: str
    record: dict[str, Any] | None = None
    details: Mapping[str, Any] | None = None

    @property
    def migration_verified(self) -> bool:
        return self.state in VERIFIED_MIGRATION_STATES

    @property
    def safe_for_migrated_resume(self) -> bool:
        """Only a fully verified recorded migration is safe as migration evidence."""

        return self.migration_verified

    @property
    def resume_compatible(self) -> bool:
        """Native streams need no migration; recorded migrations must verify."""

        return self.state in RESUME_COMPATIBLE_MIGRATION_STATES

    @property
    def event_history_origin(self) -> str:
        return str(dict(self.details or {}).get("event_history_origin") or "unknown")

    @property
    def migration_required(self) -> bool:
        return bool(dict(self.details or {}).get("migration_required"))

    @property
    def migration_record_sha256(self) -> str | None:
        value = dict(self.details or {}).get("migration_record_sha256")
        return str(value) if isinstance(value, str) else None

    @property
    def canonical_prefix_sha256(self) -> str | None:
        value = dict(self.details or {}).get("canonical_prefix_sha256")
        return str(value) if isinstance(value, str) else None

    @property
    def canonical_event_identity_sha256(self) -> str | None:
        value = dict(self.details or {}).get("canonical_sha256")
        return str(value) if isinstance(value, str) else None


def _repository_lock(path: Path | None) -> threading.RLock:
    key = os.path.normcase(str(path.resolve())) if path else "<disabled>"
    with _REPOSITORY_LOCK_GUARD:
        return _REPOSITORY_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def event_history_transaction_lock(directory: str | Path, *, timeout: float = 10.0) -> Iterator[None]:
    """Serialize event-WAL mutation with a crash-releasing advisory lock."""

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise LegacyEventMigrationError("Event-history transaction lock requires an existing regular directory.")
    path = directory / ".events.lock"
    key = os.path.normcase(str(path.resolve()))
    held = getattr(_EVENT_FILE_LOCK_LOCAL, "held", None)
    if held is None:
        held = {}
        _EVENT_FILE_LOCK_LOCAL.held = held
    if key in held:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    with _EVENT_FILE_LOCK_GUARD:
        process_lock = _EVENT_FILE_LOCKS.setdefault(key, threading.RLock())
    with process_lock:
        deadline = time.monotonic() + timeout
        handle = path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
                _fsync_parent_directory(directory)
            acquired = False
            while not acquired:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for event-history transaction lock: {path}") from None
                    time.sleep(0.05)
            held[key] = 1
            try:
                yield
            finally:
                held.pop(key, None)
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _public_reference(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "artifact"


def normalize_public_field_name(value: str) -> str:
    """Normalize a public field name without conflating ordinary key-like words."""

    separated = _PUBLIC_FIELD_ACRONYM_BOUNDARY_PATTERN.sub("_", value)
    separated = _PUBLIC_FIELD_CAMEL_BOUNDARY_PATTERN.sub("_", separated)
    return _PUBLIC_FIELD_SEPARATOR_PATTERN.sub("_", separated).strip("_").casefold()


def is_sensitive_public_key(value: str) -> bool:
    """Return whether a mapping key denotes a credential or secret value."""

    normalized = normalize_public_field_name(value)
    if not normalized:
        return False
    tokens = tuple(part for part in normalized.split("_") if part)
    if any(token in _SENSITIVE_PUBLIC_KEY_TOKENS for token in tokens):
        return True
    if any(token in _SENSITIVE_PUBLIC_KEY_COMPACT_ALIASES for token in tokens):
        return True
    return any(
        tokens[index : index + len(compound)] == compound
        for compound in _SENSITIVE_PUBLIC_KEY_COMPOUNDS
        for index in range(len(tokens) - len(compound) + 1)
    )


def _redact_public_secrets(value: str) -> str:
    public = PRIVATE_KEY_PEM_PATTERN.sub("[redacted]", value)
    public = SECRET_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}[redacted]" if is_sensitive_public_key(match.group(1)) else match.group(0)
        ),
        public,
    )
    public = BEARER_PATTERN.sub("Bearer [redacted]", public)
    public = BASIC_AUTH_PATTERN.sub("Basic [redacted]", public)
    public = URL_CREDENTIAL_PATTERN.sub(r"\1[redacted]@", public)
    for pattern in SECRET_VALUE_PATTERNS:
        public = pattern.sub("[redacted]", public)
    return public


def sanitize_public_text(value: str, private_roots: tuple[Path, ...] = ()) -> str:
    """Remove credentials and private local paths from one public string."""

    public = _redact_public_secrets(value)
    public = FILE_URI_PATTERN.sub("file://<local-path>", public)
    spellings = {spelling for root in private_roots for spelling in (str(root), root.as_posix()) if spelling}
    for spelling in sorted(spellings, key=len, reverse=True):
        flags = re.IGNORECASE if PureWindowsPath(spelling).is_absolute() else 0
        public = re.sub(re.escape(spelling), "<project>", public, flags=flags)
    candidate = public.strip()
    if PureWindowsPath(candidate).is_absolute() or PurePosixPath(candidate).is_absolute():
        reference = _public_reference(candidate)
        public = "<local-path>" if re.fullmatch(r"(?i)[a-z]:", reference) else reference
        return _redact_public_secrets(public)
    public = WINDOWS_LOCAL_PATH_PATTERN.sub("<local-path>", public)
    public = FORWARD_UNC_LOCAL_PATH_PATTERN.sub("<local-path>", public)
    public = POSIX_LOCAL_PATH_PATTERN.sub("<local-path>", public)
    return _redact_public_secrets(public)


def _public_text(value: str, private_roots: tuple[Path, ...]) -> str:
    return sanitize_public_text(value, private_roots)


def _public_metrics(metrics: dict[str, Any], *, private_roots: tuple[Path, ...] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        raw_key = str(key)
        normalized = normalize_public_field_name(raw_key)
        if is_sensitive_public_key(raw_key):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            public_key = _public_text(raw_key, private_roots)
            if not public_key or public_key in result:
                continue
            if normalized in PUBLIC_BOOLEAN_METRIC_FIELDS or normalized.endswith(
                (
                    "_available",
                    "_authorized",
                    "_eligible",
                    "_enabled",
                    "_passed",
                    "_ready",
                    "_safe",
                    "_uncertain",
                    "_validated",
                    "_verified",
                )
            ):
                result[public_key] = value if type(value) is bool else False
            else:
                result[public_key] = _public_text(value, private_roots) if isinstance(value, str) else value
    return result


@dataclass(frozen=True)
class IndexedEvent:
    event_id: int
    event: ProductEvent
    private_roots: tuple[Path, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        value = self.event.to_dict()
        for key in ("schema_version", "run_id", "timestamp", "feature", "stage", "event_type", "status", "message"):
            child = value.get(key)
            if isinstance(child, str):
                value[key] = _public_text(child, self.private_roots)
        value["metrics"] = _public_metrics(dict(self.event.metrics), private_roots=self.private_roots)
        value["artifact_references"] = [
            _public_text(_public_reference(item), self.private_roots) for item in self.event.artifact_references
        ]
        return value


@dataclass(frozen=True)
class EventReplay:
    events: tuple[IndexedEvent, ...]
    invalid_event_count: int = 0
    warnings: tuple[str, ...] = ()
    integrity_status: str = "VALID"
    migration: dict[str, Any] | None = None
    migration_state: str = EventMigrationState.NO_MIGRATION.value

    @property
    def safe_for_resume(self) -> bool:
        return self.integrity_status == "VALID" and self.invalid_event_count == 0


@dataclass
class RunSnapshot:
    run_id: str
    feature: str = "run"
    stage: str = "Waiting"
    status: str = ProductStatus.NOT_STARTED.value
    current: int = 0
    total: int | None = None
    progress_percent: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    message: str = "No events have been recorded yet."
    metrics: dict[str, Any] = field(default_factory=dict)
    recent_messages: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    timeline: list[dict[str, str]] = field(default_factory=list)
    event_count: int = 0
    resumable: bool = False
    report_available: bool = False
    invalid_event_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "feature": self.feature,
            "stage": self.stage,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "message": self.message,
            "metrics": self.metrics,
            "recent_messages": self.recent_messages,
            "artifacts": self.artifacts,
            "timeline": self.timeline,
            "event_count": self.event_count,
            "resumable": self.resumable,
            "report_available": self.report_available,
            "invalid_event_count": self.invalid_event_count,
            "warnings": self.warnings,
            "terminal": self.terminal,
        }
        validate_finite_json(payload)
        return payload


class EventRepository:
    """Durable ProductEvent and state repository beneath one runs directory."""

    def __init__(self, runs_directory: Path | None, *, private_roots: tuple[Path, ...] = ()) -> None:
        self.runs_directory = runs_directory.resolve() if runs_directory else None
        self.private_roots = tuple(root.resolve() for root in private_roots)
        self._lock = _repository_lock(self.runs_directory)

    def _run_directory(self, run_id: str) -> Path | None:
        if self.runs_directory is None or not RUN_ID_PATTERN.fullmatch(run_id):
            return None
        candidate = (self.runs_directory / run_id).resolve()
        try:
            candidate.relative_to(self.runs_directory)
        except ValueError:
            return None
        return candidate

    def run_directory(self, run_id: str, *, create: bool = False) -> Path | None:
        directory = self._run_directory(run_id)
        if create and directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _event_path(self, run_id: str) -> Path | None:
        directory = self._run_directory(run_id)
        if directory is None:
            return None
        canonical = directory / EVENT_FILENAME
        if canonical.is_file():
            return canonical
        legacy = directory / LEGACY_EVENT_FILENAME
        return legacy if legacy.is_file() else canonical

    def events(self, run_id: str, *, after_id: int = 0) -> list[IndexedEvent]:
        return list(self.replay(run_id, after_id=after_id).events)

    def replay(self, run_id: str, *, after_id: int = 0) -> EventReplay:
        directory = self._run_directory(run_id)
        if directory is None:
            return EventReplay((), warnings=("Product run ID is invalid.",), integrity_status="NOT_COMPARABLE")
        if not directory.is_dir():
            return EventReplay(())
        integrity_status = "VALID"
        integrity_warnings: list[str] = []
        migration: dict[str, Any] | None = None
        migration_state = EventMigrationState.NO_MIGRATION
        snapshot_bytes: bytes | None = None
        try:
            with self._lock, event_history_transaction_lock(directory):
                try:
                    pending = pending_event_history_transaction(run_id, directory)
                    if pending is not None and pending["operation"] == EVENT_HISTORY_TRANSACTION_APPEND:
                        append_event_transactionally(
                            run_id,
                            directory,
                            line=None,
                            request_sha256=None,
                            expected_origin=None,
                            append_surface=None,
                        )
                    migration = self._migrate_legacy_events(run_id, directory, directory / EVENT_FILENAME)
                except LegacyEventMigrationError as exc:
                    integrity_status = exc.status
                    integrity_warnings.append(str(exc))
                verification = verify_event_migration(run_id, directory)
                migration_state = verification.state
                if not verification.resume_compatible:
                    integrity_status = _migration_integrity_status(verification.state)
                    if verification.message not in integrity_warnings:
                        integrity_warnings.append(verification.message)
                elif verification.migration_verified:
                    migration = verification.record
                canonical = directory / EVENT_FILENAME
                legacy = directory / LEGACY_EVENT_FILENAME
                snapshot_path = canonical if canonical.is_file() else legacy
                if snapshot_path.is_symlink():
                    integrity_status = "NOT_COMPARABLE"
                    integrity_warnings.append("The product event stream is not a regular file.")
                elif snapshot_path.is_file():
                    try:
                        snapshot_bytes = snapshot_path.read_bytes()
                    except OSError:
                        integrity_status = "NOT_COMPARABLE"
                        integrity_warnings.append("The product event stream could not be read.")
                    if snapshot_bytes is not None and snapshot_path == canonical and verification.resume_compatible:
                        expected_identity = dict(verification.details or {}).get("canonical_sha256")
                        if expected_identity != _bytes_sha256(snapshot_bytes):
                            integrity_status = "NOT_COMPARABLE"
                            integrity_warnings.append(
                                "The canonical event snapshot changed after its authority was verified."
                            )
                _event_transaction_checkpoint("replay_snapshot_captured", directory)
        except TimeoutError:
            return EventReplay(
                (),
                warnings=("Event history is busy; retry after the active durable write completes.",),
                integrity_status="NOT_COMPARABLE",
                migration_state=EventMigrationState.NOT_COMPARABLE.value,
            )
        if snapshot_bytes is None:
            return EventReplay(
                (),
                warnings=tuple(integrity_warnings),
                integrity_status=integrity_status,
                migration=migration,
                migration_state=migration_state.value,
            )
        events: list[IndexedEvent] = []
        invalid_event_count = 0
        for event_id, line in enumerate(snapshot_bytes.splitlines(keepends=True), start=1):
            if len(line) > MAX_EVENT_ROW_BYTES:
                invalid_event_count += 1
                continue
            try:
                value = strict_json_loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Product event row is not a JSON object.")
                event = ProductEvent.from_dict(value)
            except (KeyError, TypeError, ValueError, UnicodeError):
                invalid_event_count += 1
                continue
            if event.run_id != run_id:
                invalid_event_count += 1
            elif event_id > after_id:
                events.append(IndexedEvent(event_id, event, self.private_roots))
        if invalid_event_count:
            integrity_warnings.append(f"{invalid_event_count} invalid product event row(s) were ignored.")
        if integrity_status != "VALID":
            # The validation parse above preserves only safe diagnostics such as
            # invalid-row counts.  Once authority verification fails, none of
            # the parsed event contents may become presentation evidence.
            events = []
        return EventReplay(
            tuple(events),
            invalid_event_count,
            tuple(integrity_warnings),
            integrity_status,
            migration,
            migration_state.value,
        )

    def append(self, event: ProductEvent) -> int:
        """Append one canonical event and durably record its identity in state."""

        if not RUN_ID_PATTERN.fullmatch(event.run_id):
            raise ValueError("ProductEvent run ID is invalid.")
        line = strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        directory = self.run_directory(event.run_id, create=True)
        if directory is None:
            raise OSError("Product runs directory is not configured.")
        request_sha256 = event_append_request_sha256({"surface": "EventRepository", "event": event.to_dict()})

        try:
            with self._lock, event_history_transaction_lock(directory):
                _require_mutable_event_history(directory)
                pending = pending_event_history_transaction(event.run_id, directory)
                if pending is not None and pending["operation"] == EVENT_HISTORY_TRANSACTION_APPEND:
                    recovered_event_id = append_event_transactionally(
                        event.run_id,
                        directory,
                        line=None,
                        request_sha256=None,
                        expected_origin=None,
                        append_surface=None,
                    )
                    if pending["request_sha256"] == request_sha256:
                        return recovered_event_id
                migration = self._migrate_legacy_events(
                    event.run_id,
                    directory,
                    directory / EVENT_FILENAME,
                    recover_transaction=True,
                )
                return append_event_transactionally(
                    event.run_id,
                    directory,
                    line=line,
                    request_sha256=request_sha256,
                    expected_origin=(
                        EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY if migration is not None else EVENT_HISTORY_ORIGIN_NATIVE
                    ),
                    append_surface=EVENT_APPEND_SURFACE_REPOSITORY,
                )
        except TimeoutError as exc:
            raise LegacyEventMigrationError(
                "Event history is busy; retry after the active durable write completes."
            ) from exc

    def migrate_legacy_events(self, run_id: str) -> dict[str, Any] | None:
        """Atomically migrate one valid legacy stream; never merge competing streams."""

        directory = self._run_directory(run_id)
        if directory is None:
            raise ValueError("Product run ID is invalid or the runs directory is unavailable.")
        try:
            with self._lock, event_history_transaction_lock(directory):
                return self._migrate_legacy_events(
                    run_id,
                    directory,
                    directory / EVENT_FILENAME,
                    recover_transaction=True,
                )
        except TimeoutError as exc:
            raise LegacyEventMigrationError(
                "Event history is busy; retry after the active durable write completes."
            ) from exc

    def verify_migration(
        self,
        run_id: str,
        *,
        expected_evidence_sha256: str | None = None,
        migration_required: bool = False,
        origin_required: bool = False,
    ) -> EventMigrationVerification:
        """Classify migration evidence without repairing or mutating it."""

        directory = self._run_directory(run_id)
        if directory is None:
            return _migration_verification(
                EventMigrationState.NOT_COMPARABLE,
                run_id,
                "Product run ID or event directory is invalid.",
            )
        with self._lock:
            return verify_event_migration(
                run_id,
                directory,
                expected_evidence_sha256=expected_evidence_sha256,
                migration_required=migration_required,
                origin_required=origin_required,
            )

    def _migrate_legacy_events(
        self,
        run_id: str,
        directory: Path,
        canonical: Path,
        *,
        recover_transaction: bool = False,
    ) -> dict[str, Any] | None:
        """Copy and revalidate a legacy byte prefix without normalizing it."""

        legacy = directory / LEGACY_EVENT_FILENAME
        record_path = directory / LEGACY_MIGRATION_FILENAME
        origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
        pending = pending_event_history_transaction(run_id, directory)
        if pending is not None:
            if not recover_transaction or pending["operation"] != EVENT_HISTORY_TRANSACTION_MIGRATION:
                raise LegacyEventMigrationError(
                    "An incomplete event-history transaction requires explicit controlled recovery."
                )
            _complete_migration_transaction(run_id, directory, pending)
            verification = verify_event_migration(run_id, directory, migration_required=True)
            if not verification.migration_verified:
                raise _migration_error(verification)
            return verification.record
        record_exists = record_path.exists() or record_path.is_symlink()
        if origin_path.exists() or origin_path.is_symlink():
            existing_origin = _load_event_history_origin(origin_path, run_id)
            origin_state = existing_origin["event_history_origin"]
            if origin_state == EVENT_HISTORY_ORIGIN_NATIVE and (
                record_exists or legacy.exists() or legacy.is_symlink()
            ):
                raise _migration_error(verify_event_migration(run_id, directory))
            if origin_state == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY and not record_exists:
                raise _migration_error(verify_event_migration(run_id, directory))
        if record_exists:
            verification = verify_event_migration(run_id, directory, migration_required=True)
            if not verification.migration_verified:
                raise _migration_error(verification)
            return verification.record
        if legacy.is_symlink():
            raise _migration_error(verify_event_migration(run_id, directory))
        if not legacy.is_file():
            verification = verify_event_migration(run_id, directory)
            if not verification.resume_compatible:
                raise _migration_error(verification)
            return None

        legacy_bytes = _read_event_bytes(legacy, "Legacy event stream")
        metadata = _validate_event_stream_bytes(legacy_bytes, run_id, "Legacy event stream")
        if canonical.is_file():
            if canonical.is_symlink():
                raise _migration_error(verify_event_migration(run_id, directory))
            canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
            if not legacy_bytes and canonical_bytes:
                raise LegacyEventMigrationError(
                    "Canonical and legacy event streams both exist without a migration record; refusing to merge."
                )
            if not canonical_bytes.startswith(legacy_bytes):
                raise LegacyEventMigrationError(
                    "Canonical and legacy event streams conflict; the canonical stream does not preserve the legacy prefix."
                )
            if (
                len(canonical_bytes) > len(legacy_bytes)
                and legacy_bytes
                and not legacy_bytes.endswith(b"\n")
                and canonical_bytes[len(legacy_bytes) : len(legacy_bytes) + 1] != b"\n"
            ):
                raise LegacyEventMigrationError(
                    "Canonical bytes after an unterminated legacy prefix do not begin with the required separator newline."
                )
            _validate_event_stream_bytes(canonical_bytes, run_id, "Canonical event stream")
            migration_status = "reconciled"
        else:
            canonical_bytes = legacy_bytes
            migration_status = "migrated"

        record = self._migration_record(run_id, legacy_bytes, metadata, migration_status)
        intent = _build_migration_transaction(
            run_id,
            legacy_bytes=legacy_bytes,
            canonical_bytes=canonical_bytes,
            migration_record=record,
        )
        _write_event_history_transaction(directory, intent)
        _event_transaction_checkpoint("migration_intent_published", directory)
        _complete_migration_transaction(run_id, directory, intent)
        verification = verify_event_migration(run_id, directory, migration_required=True)
        if not verification.migration_verified:
            raise _migration_error(verification)
        return verification.record

    @staticmethod
    def _migration_record(
        run_id: str,
        legacy_bytes: bytes,
        metadata: dict[str, Any],
        migration_status: str,
    ) -> dict[str, Any]:
        prefix_hash = _bytes_sha256(legacy_bytes)
        record = {
            "schema_version": LEGACY_MIGRATION_SCHEMA,
            "run_id": run_id,
            "legacy_relative_path": LEGACY_EVENT_FILENAME,
            "canonical_relative_path": EVENT_FILENAME,
            "legacy_size_bytes": len(legacy_bytes),
            "legacy_sha256": prefix_hash,
            "canonical_prefix_size_bytes": len(legacy_bytes),
            "canonical_prefix_sha256": prefix_hash,
            "validated_event_count": metadata["validated_event_count"],
            "had_terminal_newline": metadata["had_terminal_newline"],
            "line_ending_summary": metadata["line_ending_summary"],
            "legacy_source_policy": LEGACY_SOURCE_REMOVAL_POLICY,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "migration_status": migration_status,
        }
        record["record_sha256"] = _migration_record_sha256(record)
        return record

    def create_run(
        self,
        run_id: str,
        *,
        feature: str,
        command: str,
        status: str = ProductStatus.NOT_STARTED.value,
        stage: str = "waiting",
        started_at: str | None = None,
        resumable: bool = False,
        backend_id: str | None = None,
        backend_run_reference: str | None = None,
        backend_identity: dict[str, Any] | None = None,
        report_reference: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("Product run ID is invalid.")
        state: dict[str, Any] = {
            "schema_version": RUN_STATE_SCHEMA,
            "run_id": run_id,
            "feature": feature,
            "command": command,
            "status": status,
            "stage": stage,
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "resumable": bool(resumable),
            "backend_id": backend_id,
            "backend_run_reference": backend_run_reference,
            "backend_identity": backend_identity or {},
            "last_durable_event": None,
            "report_reference": report_reference,
        }
        if extra:
            state.update(extra)
        with self._lock:
            existing = self._state(run_id)
            if existing:
                raise FileExistsError(f"Product run already exists: {run_id}")
            directory = self._run_directory(run_id)
            if directory is None or self.runs_directory is None:
                raise OSError("Product runs directory is not configured.")
            self.runs_directory.mkdir(parents=True, exist_ok=True)
            if directory.exists() or directory.is_symlink():
                if directory.is_symlink() or not directory.is_dir():
                    raise FileExistsError(f"Product run path is not an unclaimed directory: {run_id}")
                claimed = (
                    directory / EVENT_FILENAME,
                    directory / EVENT_HISTORY_ORIGIN_FILENAME,
                    directory / "state.json",
                )
                if any(path.exists() or path.is_symlink() for path in claimed):
                    raise FileExistsError(f"Product run directory already carries authoritative state: {run_id}")
            else:
                directory.mkdir(exist_ok=False)
            _atomic_bytes(directory / EVENT_FILENAME, b"")
            origin = record_event_history_origin(
                run_id,
                directory,
                expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
            )
            state.update(
                _event_history_origin_bindings(
                    origin,
                    _bytes_sha256((directory / EVENT_HISTORY_ORIGIN_FILENAME).read_bytes()),
                    _bytes_sha256(b""),
                )
            )
            self._write_state(run_id, state)
        return state

    def initialize_run(
        self,
        run_id: str,
        *,
        feature: str,
        command: str,
        command_payload: Mapping[str, Any],
        planned_event: ProductEvent,
        status: str = ProductStatus.NOT_STARTED.value,
        stage: str = "waiting",
        started_at: str | None = None,
        resumable: bool = False,
        backend_id: str | None = None,
        backend_run_reference: str | None = None,
        backend_identity: dict[str, Any] | None = None,
        report_reference: str | None = None,
        extra: dict[str, Any] | None = None,
        required_directories: tuple[str, ...] = ("logs", "artifacts", "report"),
        on_step: Callable[[str, Path], None] | None = None,
    ) -> dict[str, Any]:
        """Publish an authoritative run only after its complete planned skeleton exists."""

        if not RUN_ID_PATTERN.fullmatch(run_id) or planned_event.run_id != run_id:
            raise ValueError("Product run and planned-event identities must match.")
        for name in required_directories:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Required run directory names must be safe single path components.")
        command_bytes = (
            strict_json_dumps(dict(command_payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
        event_bytes = (
            strict_json_dumps(planned_event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        state: dict[str, Any] = {
            "schema_version": RUN_STATE_SCHEMA,
            "run_id": run_id,
            "feature": feature,
            "command": command,
            "status": status,
            "stage": stage,
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "resumable": bool(resumable),
            "backend_id": backend_id,
            "backend_run_reference": backend_run_reference,
            "backend_identity": backend_identity or {},
            "last_durable_event": None,
            "report_reference": report_reference,
        }
        if extra:
            state.update(extra)

        def completed(step: str, directory: Path) -> None:
            if on_step is not None:
                on_step(step, directory)

        with self._lock:
            directory = self._run_directory(run_id)
            if directory is None or self.runs_directory is None:
                raise OSError("Product runs directory is not configured.")
            self.runs_directory.mkdir(parents=True, exist_ok=True)
            directory.mkdir(exist_ok=False)
            completed("run_directory_created", directory)
            for name in required_directories:
                (directory / name).mkdir(exist_ok=False)
            completed("directories_created", directory)
            _atomic_bytes(directory / "command.json", command_bytes)
            completed("command_written", directory)
            event_path = directory / EVENT_FILENAME
            with event_path.open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            completed("events_created", directory)
            with event_path.open("ab") as handle:
                handle.write(event_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            origin = record_event_history_origin(
                run_id,
                directory,
                expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
            )
            completed("planned_event_appended", directory)
            state.update(
                {
                    "status": planned_event.status.value,
                    "stage": planned_event.stage,
                    "message": planned_event.message,
                    "last_durable_event": {
                        "event_id": 1,
                        "event_type": planned_event.event_type,
                        "timestamp": planned_event.timestamp,
                    },
                    "event_history_origin": EVENT_HISTORY_ORIGIN_NATIVE,
                    "event_history_origin_record_sha256": _bytes_sha256(
                        (directory / EVENT_HISTORY_ORIGIN_FILENAME).read_bytes()
                    ),
                    "event_migration_required": False,
                    "event_migration_record_sha256": None,
                    "event_canonical_prefix_sha256": origin["canonical_prefix_sha256"],
                    "event_canonical_origin_identity_sha256": origin["canonical_event_identity_sha256"],
                    "event_canonical_current_identity_sha256": _bytes_sha256(event_bytes),
                }
            )
            completed("before_state", directory)
            self._write_state(run_id, state)
            completed("state_written", directory)
        return dict(state)

    def update_state(self, run_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            state = self._state(run_id)
            if not state:
                raise FileNotFoundError(f"Product run state is unavailable: {run_id}")
            protected = {"schema_version", "run_id"}
            if protected.intersection(updates):
                raise ValueError("Product run identity fields cannot be updated.")
            state.update(updates)
            self._write_state(run_id, state)
            return state

    def _write_state(self, run_id: str, state: dict[str, Any]) -> None:
        directory = self.run_directory(run_id, create=True)
        if directory is None:
            raise OSError("Product runs directory is not configured.")
        state = dict(state)
        state.setdefault("schema_version", RUN_STATE_SCHEMA)
        state["run_id"] = run_id
        _atomic_json(directory / "state.json", state)

    def _state(self, run_id: str) -> dict[str, Any]:
        directory = self._run_directory(run_id)
        path = directory / "state.json" if directory else None
        if path is None or not path.is_file() or path.stat().st_size > 1_000_000:
            return {}
        try:
            value = strict_json_loads(path.read_bytes())
        except (OSError, ValueError, UnicodeError):
            return {}
        return value if isinstance(value, dict) and value.get("run_id") == run_id else {}

    def state(self, run_id: str) -> dict[str, Any]:
        return dict(self._state(run_id))

    def snapshot(self, run_id: str) -> RunSnapshot:
        replay = self.replay(run_id)
        indexed = list(replay.events)
        state = self._state(run_id)
        snapshot = RunSnapshot(run_id=_public_text(run_id, self.private_roots))
        snapshot.resumable = state.get("resumable") if type(state.get("resumable")) is bool else False
        snapshot.report_available = self.report_path(run_id) is not None
        snapshot.invalid_event_count = replay.invalid_event_count
        snapshot.warnings = [_public_text(item, self.private_roots) for item in replay.warnings]
        if not indexed:
            snapshot.feature = _public_text(str(state.get("command") or "run"), self.private_roots)
            snapshot.stage = _public_text(str(state.get("stage") or "Waiting"), self.private_roots)
            snapshot.status = _public_text(
                str(state.get("status") or ProductStatus.NOT_STARTED.value), self.private_roots
            )
            snapshot.message = _public_text(str(state.get("message") or snapshot.message), self.private_roots)
            started_at = _safe_optional_text(state.get("started_at"))
            ended_at = _safe_optional_text(state.get("ended_at"))
            snapshot.started_at = _public_text(started_at, self.private_roots) if started_at else None
            snapshot.ended_at = _public_text(ended_at, self.private_roots) if ended_at else None
            if replay.integrity_status != "VALID":
                snapshot.status = replay.integrity_status
                snapshot.message = (
                    snapshot.warnings[0] if snapshot.warnings else "Event stream integrity could not be verified."
                )
                snapshot.resumable = False
            return snapshot
        first, last = indexed[0].event, indexed[-1].event
        snapshot.feature = _public_text(last.feature, self.private_roots)
        snapshot.stage = _public_text(last.stage, self.private_roots)
        snapshot.status = _public_text(last.status.value, self.private_roots)
        snapshot.current = last.current
        snapshot.total = last.total
        snapshot.message = _public_text(last.message, self.private_roots)
        snapshot.metrics = _public_metrics(dict(last.metrics), private_roots=self.private_roots)
        snapshot.started_at = _public_text(first.timestamp, self.private_roots)
        snapshot.ended_at = _public_text(last.timestamp, self.private_roots) if snapshot.terminal else None
        snapshot.event_count = len(indexed)
        if last.total is not None and last.total > 0:
            snapshot.progress_percent = max(0.0, min(100.0, 100.0 * last.current / last.total))
        start, end = _timestamp(first.timestamp), _timestamp(last.timestamp)
        if start and end:
            snapshot.elapsed_seconds = max(0.0, (end - start).total_seconds())
        supplied_eta = _numeric(last.metrics.get("eta_seconds"))
        if supplied_eta is not None:
            snapshot.eta_seconds = max(0.0, supplied_eta)
        elif (
            not snapshot.terminal
            and snapshot.elapsed_seconds
            and last.total is not None
            and 0 < last.current < last.total
        ):
            snapshot.eta_seconds = max(0.0, (last.total - last.current) * snapshot.elapsed_seconds / last.current)
        snapshot.recent_messages = [
            _public_text(event.event.message, self.private_roots) for event in indexed if event.event.message
        ][-8:]
        snapshot.artifacts = list(
            dict.fromkeys(
                _public_text(_public_reference(reference), self.private_roots)
                for item in indexed
                for reference in item.event.artifact_references
            )
        )
        timeline: dict[str, dict[str, str]] = {}
        for item in indexed:
            event = item.event
            if event.stage not in timeline:
                timeline[event.stage] = {
                    "stage": _public_text(event.stage, self.private_roots),
                    "status": _public_text(event.status.value, self.private_roots),
                    "message": _public_text(event.message, self.private_roots),
                }
            else:
                timeline[event.stage].update(
                    status=_public_text(event.status.value, self.private_roots),
                    message=_public_text(event.message, self.private_roots),
                )
        snapshot.timeline = list(timeline.values())
        if state.get("status") == "STALE":
            snapshot.status = "STALE"
            snapshot.message = _public_text(
                str(state.get("message") or "A durable artifact changed after this run completed."),
                self.private_roots,
            )
        if replay.integrity_status != "VALID":
            snapshot.status = replay.integrity_status
            snapshot.message = (
                snapshot.warnings[0] if snapshot.warnings else "Event stream integrity could not be verified."
            )
            snapshot.resumable = False
        elif replay.invalid_event_count:
            snapshot.resumable = False
        return snapshot

    def recent_runs(self, *, limit: int = 20) -> list[RunSnapshot]:
        if self.runs_directory is None or not self.runs_directory.is_dir():
            return []
        candidates: list[tuple[str, str]] = []
        try:
            for child in self.runs_directory.iterdir():
                if not child.is_dir() or not RUN_ID_PATTERN.fullmatch(child.name):
                    continue
                state = self._state(child.name)
                if state:
                    candidates.append((str(state.get("started_at") or ""), child.name))
        except OSError:
            return []
        candidates.sort(reverse=True)
        return [self.snapshot(run_id) for _, run_id in candidates[: max(0, min(limit, 100))]]

    def current_run(self) -> RunSnapshot | None:
        runs = self.recent_runs(limit=100)
        return next((run for run in runs if not run.terminal), runs[0] if runs else None)

    def recent_run_ids(self, *, feature: str | None = None, limit: int = 100) -> list[str]:
        if self.runs_directory is None or not self.runs_directory.is_dir():
            return []
        rows: list[tuple[str, str]] = []
        try:
            for child in self.runs_directory.iterdir():
                if not child.is_dir() or not RUN_ID_PATTERN.fullmatch(child.name):
                    continue
                state = self._state(child.name)
                state_feature = str(state.get("feature") or state.get("command") or "")
                if state and (feature is None or state_feature == feature):
                    rows.append((str(state.get("started_at") or ""), child.name))
        except OSError:
            return []
        rows.sort(reverse=True)
        return [run_id for _started, run_id in rows[: max(0, min(limit, 1000))]]

    def log_text(self, run_id: str, *, max_bytes: int = 200_000) -> str:
        directory = self._run_directory(run_id)
        path = directory / "logs" / "run.log" if directory else None
        if path is None or not path.is_file():
            return ""
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > max_bytes:
                    handle.seek(-max_bytes, 2)
                value = handle.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            return ""
        roots = self.private_roots or ((self.runs_directory.parent,) if self.runs_directory else ())
        return _public_text(value, roots)

    def report_path(self, run_id: str) -> Path | None:
        directory = self._run_directory(run_id)
        if directory is None:
            return None
        state = self._state(run_id)
        raw = state.get("report_reference") or "report/index.html"
        reference = Path(str(raw))
        if reference.is_absolute() or ".." in reference.parts:
            return None
        path = (directory / reference).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            return None
        return path if path.is_file() else None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_event_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LegacyEventMigrationError(f"{label} could not be read.") from exc


def _validate_event_stream_bytes(data: bytes, run_id: str, label: str) -> dict[str, Any]:
    summary = {"lf": 0, "crlf": 0, "unterminated": 0}
    event_count = 0
    segments = data.split(b"\n")
    for index, segment in enumerate(segments):
        terminated = index < len(segments) - 1
        if not terminated and segment == b"" and data.endswith(b"\n"):
            continue
        row_size = len(segment) + (1 if terminated else 0)
        if row_size > MAX_EVENT_ROW_BYTES:
            raise LegacyEventMigrationError(f"{label} row {index + 1} is oversized.")
        if terminated and segment.endswith(b"\r"):
            content = segment[:-1]
            summary["crlf"] += 1
        elif terminated:
            content = segment
            summary["lf"] += 1
        else:
            content = segment
            summary["unterminated"] += 1
        if not content:
            raise LegacyEventMigrationError(f"{label} row {index + 1} is empty.")
        try:
            value = strict_json_loads(content)
            if not isinstance(value, dict):
                raise ValueError("event row is not an object")
            event = ProductEvent.from_dict(value)
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise LegacyEventMigrationError(
                f"{label} row {index + 1} contains malformed or invalid event JSON."
            ) from exc
        if event.run_id != run_id:
            raise LegacyEventMigrationError(f"{label} row {index + 1} belongs to a different run.")
        event_count += 1
    return {
        "validated_event_count": event_count,
        "had_terminal_newline": bool(data.endswith(b"\n")),
        "line_ending_summary": summary,
    }


def _validate_migration_record(record: dict[str, Any], run_id: str) -> None:
    base_fields = {
        "schema_version",
        "run_id",
        "legacy_relative_path",
        "canonical_relative_path",
        "legacy_size_bytes",
        "legacy_sha256",
        "canonical_prefix_size_bytes",
        "canonical_prefix_sha256",
        "validated_event_count",
        "had_terminal_newline",
        "line_ending_summary",
        "legacy_source_policy",
        "created_at_utc",
        "migration_status",
    }
    schema = record.get("schema_version")
    required = set(base_fields)
    if schema == LEGACY_MIGRATION_SCHEMA:
        required.add("record_sha256")
    if not required.issubset(record):
        raise LegacyEventMigrationError("Legacy event migration record is missing required fields.")
    if set(record) != required:
        raise LegacyEventMigrationError("Legacy event migration record contains unsupported fields.")
    if schema not in {LEGACY_MIGRATION_SCHEMA_V2, LEGACY_MIGRATION_SCHEMA} or record.get("run_id") != run_id:
        raise LegacyEventMigrationError("Legacy event migration record schema or run identity is invalid.")
    if (
        record.get("legacy_relative_path") != LEGACY_EVENT_FILENAME
        or record.get("canonical_relative_path") != EVENT_FILENAME
    ):
        raise LegacyEventMigrationError("Legacy event migration record paths are invalid.")
    for field_name in ("legacy_size_bytes", "canonical_prefix_size_bytes", "validated_event_count"):
        if type(record.get(field_name)) is not int or int(record[field_name]) < 0:
            raise LegacyEventMigrationError(f"Legacy event migration record field {field_name} is invalid.")
    for field_name in ("legacy_sha256", "canonical_prefix_sha256"):
        if not isinstance(record.get(field_name), str) or not re.fullmatch(r"[0-9a-f]{64}", str(record[field_name])):
            raise LegacyEventMigrationError(f"Legacy event migration record field {field_name} is invalid.")
    if (
        record["legacy_size_bytes"] != record["canonical_prefix_size_bytes"]
        or record["legacy_sha256"] != record["canonical_prefix_sha256"]
    ):
        raise LegacyEventMigrationError("Legacy and canonical-prefix bindings disagree in the migration record.")
    summary = record.get("line_ending_summary")
    if not isinstance(summary, dict) or set(summary) != {"lf", "crlf", "unterminated"}:
        raise LegacyEventMigrationError("Legacy event migration line-ending summary is invalid.")
    if any(type(summary[key]) is not int or summary[key] < 0 for key in summary):
        raise LegacyEventMigrationError("Legacy event migration line-ending counts are invalid.")
    if sum(summary.values()) != record["validated_event_count"] or summary["unterminated"] not in {0, 1}:
        raise LegacyEventMigrationError("Legacy event migration line-ending counts do not match the event count.")
    terminal = record.get("had_terminal_newline")
    if type(terminal) is not bool:
        raise LegacyEventMigrationError("Legacy event migration terminal-newline flag is invalid.")
    if (terminal and summary["unterminated"]) or (
        not terminal and record["validated_event_count"] and summary["unterminated"] != 1
    ):
        raise LegacyEventMigrationError("Legacy event migration terminal-newline metadata is inconsistent.")
    if record.get("migration_status") not in {"migrated", "reconciled"}:
        raise LegacyEventMigrationError("Legacy event migration status is invalid.")
    if record.get("legacy_source_policy") != LEGACY_SOURCE_REMOVAL_POLICY:
        raise LegacyEventMigrationError("Legacy event migration source-removal policy is invalid.")
    if not isinstance(record.get("created_at_utc"), str) or _timestamp(record["created_at_utc"]) is None:
        raise LegacyEventMigrationError("Legacy event migration creation timestamp is invalid.")
    if schema == LEGACY_MIGRATION_SCHEMA:
        recorded_hash = record.get("record_sha256")
        if not isinstance(recorded_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise LegacyEventMigrationError("Legacy event migration record self-hash is invalid.")
        if recorded_hash != _migration_record_sha256(record):
            raise LegacyEventMigrationError("Legacy event migration record self-hash changed.")


def _load_migration_record(path: Path, run_id: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVENT_ROW_BYTES:
        raise LegacyEventMigrationError("Legacy event migration record is missing or oversized.")
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegacyEventMigrationError("Legacy event migration record is malformed.") from exc
    if not isinstance(value, dict):
        raise LegacyEventMigrationError("Legacy event migration record is not a JSON object.")
    _validate_migration_record(value, run_id)
    return value


def _migration_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = strict_json_dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _bytes_sha256(encoded)


_EVENT_HISTORY_ORIGIN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "event_history_origin",
        "canonical_event_path",
        "canonical_event_identity_sha256",
        "migration_required",
        "migration_record_path",
        "migration_record_sha256",
        "legacy_source_path",
        "legacy_source_size_bytes",
        "legacy_source_sha256",
        "canonical_prefix_size_bytes",
        "canonical_prefix_sha256",
        "legacy_source_removal_permitted",
        "created_at_utc",
        "record_sha256",
    }
)


def _validate_event_history_origin_record(record: Mapping[str, Any], run_id: str) -> None:
    if set(record) != _EVENT_HISTORY_ORIGIN_FIELDS:
        raise LegacyEventMigrationError("Event-history origin record does not contain the exact required fields.")
    if record.get("schema_version") != EVENT_HISTORY_ORIGIN_SCHEMA or record.get("run_id") != run_id:
        raise LegacyEventMigrationError("Event-history origin record schema or run identity is invalid.")
    origin = record.get("event_history_origin")
    if origin not in EVENT_HISTORY_ORIGIN_STATES:
        raise LegacyEventMigrationError("Event-history origin state is not a controlled origin value.")
    if record.get("canonical_event_path") != EVENT_FILENAME:
        raise LegacyEventMigrationError("Event-history origin canonical event path is invalid.")
    migrated = origin == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    if record.get("migration_required") is not migrated:
        raise LegacyEventMigrationError("Event-history origin migration-required flag does not match its origin.")
    for field_name in ("canonical_event_identity_sha256", "canonical_prefix_sha256"):
        if not isinstance(record.get(field_name), str) or not re.fullmatch(r"[0-9a-f]{64}", str(record[field_name])):
            raise LegacyEventMigrationError(f"Event-history origin field {field_name} is invalid.")
    if record["canonical_event_identity_sha256"] != record["canonical_prefix_sha256"]:
        raise LegacyEventMigrationError(
            "Event-history origin canonical identity does not match its immutable canonical prefix."
        )
    if type(record.get("canonical_prefix_size_bytes")) is not int or record["canonical_prefix_size_bytes"] < 0:
        raise LegacyEventMigrationError("Event-history origin canonical prefix size is invalid.")
    if migrated:
        if record.get("migration_record_path") != LEGACY_MIGRATION_FILENAME:
            raise LegacyEventMigrationError("Event-history origin migration record path is invalid.")
        if record.get("legacy_source_path") != LEGACY_EVENT_FILENAME:
            raise LegacyEventMigrationError("Event-history origin legacy source path is invalid.")
        for field_name in ("migration_record_sha256", "legacy_source_sha256"):
            if not isinstance(record.get(field_name), str) or not re.fullmatch(
                r"[0-9a-f]{64}", str(record[field_name])
            ):
                raise LegacyEventMigrationError(f"Event-history origin field {field_name} is invalid.")
        if type(record.get("legacy_source_size_bytes")) is not int or record["legacy_source_size_bytes"] < 0:
            raise LegacyEventMigrationError("Event-history origin legacy source size is invalid.")
        if record.get("legacy_source_removal_permitted") is not True:
            raise LegacyEventMigrationError("Event-history origin source-removal permission is invalid.")
        if (
            record["legacy_source_size_bytes"] != record["canonical_prefix_size_bytes"]
            or record["legacy_source_sha256"] != record["canonical_prefix_sha256"]
        ):
            raise LegacyEventMigrationError("Event-history origin legacy and canonical-prefix bindings disagree.")
    else:
        for field_name in (
            "migration_record_path",
            "migration_record_sha256",
            "legacy_source_path",
            "legacy_source_size_bytes",
            "legacy_source_sha256",
        ):
            if record.get(field_name) is not None:
                raise LegacyEventMigrationError("Event-history origin native record carries migration bindings.")
        if record.get("legacy_source_removal_permitted") is not False:
            raise LegacyEventMigrationError("Event-history origin native source-removal flag is invalid.")
    if not isinstance(record.get("created_at_utc"), str) or _timestamp(record["created_at_utc"]) is None:
        raise LegacyEventMigrationError("Event-history origin creation timestamp is invalid.")
    recorded_hash = record.get("record_sha256")
    if not isinstance(recorded_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
        raise LegacyEventMigrationError("Event-history origin record self-hash is invalid.")
    if recorded_hash != _migration_record_sha256(record):
        raise LegacyEventMigrationError("Event-history origin record self-hash changed.")


def _load_event_history_origin(path: Path, run_id: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVENT_ROW_BYTES:
        raise LegacyEventMigrationError("Event-history origin record is missing, irregular, or oversized.")
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegacyEventMigrationError("Event-history origin record is malformed.") from exc
    if not isinstance(value, dict):
        raise LegacyEventMigrationError("Event-history origin record is not a JSON object.")
    _validate_event_history_origin_record(value, run_id)
    return value


def _build_event_history_origin(
    run_id: str,
    directory: Path,
    *,
    migration_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    canonical = directory / EVENT_FILENAME
    canonical_bytes = b""
    if canonical.exists() or canonical.is_symlink():
        if canonical.is_symlink() or not canonical.is_file():
            raise LegacyEventMigrationError("Canonical event stream is not a regular file for origin recording.")
        canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
    if migration_record is not None:
        record_path = directory / LEGACY_MIGRATION_FILENAME
        if record_path.is_symlink() or not record_path.is_file():
            raise LegacyEventMigrationError("Migration record is required to record a migrated event origin.")
        record_sha256 = _bytes_sha256(record_path.read_bytes())
        origin: dict[str, Any] = {
            "schema_version": EVENT_HISTORY_ORIGIN_SCHEMA,
            "run_id": run_id,
            "event_history_origin": EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
            "canonical_event_path": EVENT_FILENAME,
            "canonical_event_identity_sha256": str(migration_record["canonical_prefix_sha256"]),
            "migration_required": True,
            "migration_record_path": LEGACY_MIGRATION_FILENAME,
            "migration_record_sha256": record_sha256,
            "legacy_source_path": LEGACY_EVENT_FILENAME,
            "legacy_source_size_bytes": int(migration_record["legacy_size_bytes"]),
            "legacy_source_sha256": str(migration_record["legacy_sha256"]),
            "canonical_prefix_size_bytes": int(migration_record["canonical_prefix_size_bytes"]),
            "canonical_prefix_sha256": str(migration_record["canonical_prefix_sha256"]),
            "legacy_source_removal_permitted": migration_record.get("legacy_source_policy")
            == LEGACY_SOURCE_REMOVAL_POLICY,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        origin = {
            "schema_version": EVENT_HISTORY_ORIGIN_SCHEMA,
            "run_id": run_id,
            "event_history_origin": EVENT_HISTORY_ORIGIN_NATIVE,
            "canonical_event_path": EVENT_FILENAME,
            "canonical_event_identity_sha256": _bytes_sha256(canonical_bytes),
            "migration_required": False,
            "migration_record_path": None,
            "migration_record_sha256": None,
            "legacy_source_path": None,
            "legacy_source_size_bytes": None,
            "legacy_source_sha256": None,
            "canonical_prefix_size_bytes": len(canonical_bytes),
            "canonical_prefix_sha256": _bytes_sha256(canonical_bytes),
            "legacy_source_removal_permitted": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    origin["record_sha256"] = _migration_record_sha256(origin)
    _validate_event_history_origin_record(origin, run_id)
    return origin


_EVENT_HISTORY_ORIGIN_BINDING_FIELDS = (
    "event_history_origin",
    "event_history_origin_record_sha256",
    "event_migration_required",
    "event_migration_record_sha256",
    "event_canonical_prefix_sha256",
    "event_canonical_origin_identity_sha256",
    "event_canonical_current_identity_sha256",
)


def _event_history_origin_bindings(
    origin: Mapping[str, Any],
    origin_file_sha256: str,
    canonical_current_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "event_history_origin": origin["event_history_origin"],
        "event_history_origin_record_sha256": origin_file_sha256,
        "event_migration_required": origin["migration_required"],
        "event_migration_record_sha256": origin["migration_record_sha256"],
        "event_canonical_prefix_sha256": origin["canonical_prefix_sha256"],
        "event_canonical_origin_identity_sha256": origin["canonical_event_identity_sha256"],
        "event_canonical_current_identity_sha256": canonical_current_identity_sha256,
    }


def _persist_event_history_origin_bindings(directory: Path, run_id: str, origin: Mapping[str, Any]) -> None:
    """Restate immutable origin facts in durable run identity/state when present."""

    origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
    origin_file_sha256 = _bytes_sha256(origin_path.read_bytes())
    canonical_path = directory / EVENT_FILENAME
    if canonical_path.is_symlink() or not canonical_path.is_file():
        raise LegacyEventMigrationError("Canonical event stream is missing or irregular for origin binding.")
    canonical_current_identity_sha256 = _bytes_sha256(_read_event_bytes(canonical_path, "Canonical event stream"))
    bindings = _event_history_origin_bindings(
        origin,
        origin_file_sha256,
        canonical_current_identity_sha256,
    )
    for name in ("state.json", "run_identity.json"):
        path = directory / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVENT_ROW_BYTES:
            raise LegacyEventMigrationError(f"Authoritative {name} is irregular or oversized for origin binding.")
        try:
            value = strict_json_loads(path.read_bytes())
        except (OSError, UnicodeError, ValueError) as exc:
            raise LegacyEventMigrationError(f"Authoritative {name} is malformed for origin binding.") from exc
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise LegacyEventMigrationError(f"Authoritative {name} has the wrong run identity for origin binding.")
        value.update(bindings)
        _atomic_json(path, value)


def _declared_event_history_origin_bindings(
    directory: Path,
    run_id: str,
    *,
    allow_unbound: bool = False,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    declarations: list[tuple[str, dict[str, Any]]] = []
    for name in ("state.json", "run_identity.json"):
        path = directory / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVENT_ROW_BYTES:
            raise LegacyEventMigrationError(f"Authoritative {name} is irregular or oversized for origin binding.")
        try:
            value = strict_json_loads(path.read_bytes())
        except (OSError, UnicodeError, ValueError) as exc:
            raise LegacyEventMigrationError(f"Authoritative {name} is malformed for origin binding.") from exc
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise LegacyEventMigrationError(f"Authoritative {name} has the wrong run identity for origin binding.")
        present = {key: value[key] for key in _EVENT_HISTORY_ORIGIN_BINDING_FIELDS if key in value}
        if not present:
            if allow_unbound:
                continue
            raise LegacyEventMigrationError(f"Authoritative {name} is missing all event-history origin bindings.")
        missing = [key for key in _EVENT_HISTORY_ORIGIN_BINDING_FIELDS if key not in present]
        if missing:
            raise LegacyEventMigrationError(
                f"Authoritative {name} has incomplete event-history origin bindings: " + ", ".join(missing)
            )
        if "event_history_origin" not in present or present["event_history_origin"] not in EVENT_HISTORY_ORIGIN_STATES:
            raise LegacyEventMigrationError(f"Authoritative {name} has an invalid event-history origin binding.")
        for hash_field in (
            "event_history_origin_record_sha256",
            "event_migration_record_sha256",
            "event_canonical_prefix_sha256",
            "event_canonical_origin_identity_sha256",
            "event_canonical_current_identity_sha256",
        ):
            field_value = present.get(hash_field)
            if field_value is not None and (
                not isinstance(field_value, str) or not re.fullmatch(r"[0-9a-f]{64}", field_value)
            ):
                raise LegacyEventMigrationError(f"Authoritative {name} has a malformed {hash_field} binding.")
        if "event_migration_required" in present and type(present["event_migration_required"]) is not bool:
            raise LegacyEventMigrationError(f"Authoritative {name} has a malformed event-migration-required binding.")
        declarations.append((name, present))
    return tuple(declarations)


def _validate_current_canonical_bindings(
    declarations: tuple[tuple[str, dict[str, Any]], ...],
    canonical_identity_sha256: str,
) -> None:
    for source_name, declared in declarations:
        recorded = declared.get("event_canonical_current_identity_sha256")
        if recorded is not None and recorded != canonical_identity_sha256:
            raise LegacyEventMigrationError(
                f"Authoritative {source_name} binds a different current canonical event identity."
            )


def record_event_history_origin(
    run_id: str,
    directory: str | Path,
    *,
    expected_origin: str | None = None,
    update_current_binding: bool = False,
    allow_binding_population: bool = False,
    require_existing: bool = False,
) -> dict[str, Any]:
    """Persist the authoritative event-history origin fact for one run directory.

    The origin is derived exactly once, at controlled write time, by the same
    component that created or migrated the canonical stream.  Verification never
    re-derives the origin from later file presence.
    """

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise LegacyEventMigrationError("Product run ID is invalid for event-history origin recording.")
    if allow_binding_population and expected_origin not in EVENT_HISTORY_ORIGIN_STATES:
        raise LegacyEventMigrationError(
            "Initial authoritative-state binding requires an explicit controlled event-history origin."
        )
    directory = Path(directory)
    _require_mutable_event_history(directory)
    origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
    record_path = directory / LEGACY_MIGRATION_FILENAME
    legacy = directory / LEGACY_EVENT_FILENAME
    if require_existing and not origin_path.exists() and not origin_path.is_symlink():
        raise LegacyEventMigrationError(
            "The append transaction requires its exact preexisting event-history origin record."
        )
    if origin_path.exists() or origin_path.is_symlink():
        existing = _load_event_history_origin(origin_path, run_id)
        if expected_origin is not None and existing["event_history_origin"] != expected_origin:
            raise LegacyEventMigrationError("Recorded event-history origin conflicts with current migration state.")
        if existing["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
            if record_path.is_symlink() or not record_path.is_file():
                raise LegacyEventMigrationError("Recorded migrated event-history origin requires its migration record.")
            _load_migration_record(record_path, run_id)
            if existing["migration_record_sha256"] != _bytes_sha256(record_path.read_bytes()):
                raise LegacyEventMigrationError("Recorded event-history origin binds a different migration record.")
        elif record_path.exists() or record_path.is_symlink() or legacy.exists() or legacy.is_symlink():
            raise LegacyEventMigrationError("Recorded native event-history origin conflicts with migration evidence.")
        origin_file_sha256 = _bytes_sha256(origin_path.read_bytes())
        canonical_path = directory / EVENT_FILENAME
        if canonical_path.is_symlink() or not canonical_path.is_file():
            raise LegacyEventMigrationError("Canonical event stream is missing or irregular for origin binding.")
        canonical_identity_sha256 = _bytes_sha256(_read_event_bytes(canonical_path, "Canonical event stream"))
        declarations = _declared_event_history_origin_bindings(
            directory,
            run_id,
            allow_unbound=allow_binding_population,
        )
        expected = _event_history_origin_bindings(existing, origin_file_sha256, canonical_identity_sha256)
        for source_name, declared in declarations:
            mismatched = [
                key
                for key, value in declared.items()
                if key != "event_canonical_current_identity_sha256" and expected.get(key) != value
            ]
            if mismatched:
                raise LegacyEventMigrationError(
                    f"Authoritative {source_name} event-history bindings disagree: " + ", ".join(mismatched)
                )
        if not update_current_binding:
            _validate_current_canonical_bindings(declarations, canonical_identity_sha256)
        _persist_event_history_origin_bindings(directory, run_id, existing)
        return existing
    if expected_origin not in EVENT_HISTORY_ORIGIN_STATES:
        raise LegacyEventMigrationError(
            "Event-history origin is missing and cannot be reconstructed from current file presence."
        )
    migration_record: dict[str, Any] | None = None
    if expected_origin == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
        if record_path.is_symlink() or not record_path.is_file():
            raise LegacyEventMigrationError("A migrated event-history origin requires a valid migration record.")
        migration_record = _load_migration_record(record_path, run_id)
    elif record_path.exists() or record_path.is_symlink() or legacy.exists() or legacy.is_symlink():
        raise LegacyEventMigrationError(
            "A legacy event stream must be migrated before its event-history origin can be recorded."
        )
    origin = _build_event_history_origin(run_id, directory, migration_record=migration_record)
    _atomic_json(origin_path, origin)
    _persist_event_history_origin_bindings(directory, run_id, origin)
    return origin


def _migration_evidence_sha256(
    state: EventMigrationState,
    run_id: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": MIGRATION_EVIDENCE_SCHEMA,
        "state": state.value,
        "run_id": run_id,
        "details": dict(details or {}),
    }
    encoded = strict_json_dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _bytes_sha256(encoded)


def _migration_verification(
    state: EventMigrationState,
    run_id: str,
    message: str,
    *,
    record: dict[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> EventMigrationVerification:
    return EventMigrationVerification(
        state,
        run_id,
        _migration_evidence_sha256(state, run_id, details=details),
        message,
        dict(record) if record is not None else None,
        dict(details) if details is not None else None,
    )


def _migration_marker_requires_record(directory: Path, run_id: str) -> bool:
    state_path = directory / "state.json"
    if state_path.is_symlink() or not state_path.is_file():
        return False
    try:
        if state_path.stat().st_size > MAX_EVENT_ROW_BYTES:
            return False
        value = strict_json_loads(state_path.read_bytes())
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        return False
    return isinstance(value.get("event_stream_migration"), dict) or (
        value.get("event_history_origin") == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    )


def _origin_evidence_details(
    origin: Mapping[str, Any] | None,
    origin_record_sha256: str | None,
    *,
    derived_origin: str,
    migration_required: bool,
) -> dict[str, Any]:
    if origin is None:
        return {
            "event_history_origin": derived_origin,
            "origin_record_present": False,
            "migration_required": migration_required,
        }
    return {
        "event_history_origin": origin["event_history_origin"],
        "origin_record_present": True,
        "origin_record_sha256": origin_record_sha256,
        "origin_self_sha256": origin["record_sha256"],
        "migration_required": bool(origin["migration_required"]),
    }


def _migration_integrity_status(state: EventMigrationState) -> str:
    return "STALE" if state == EventMigrationState.STALE_SOURCE_CHANGED else "NOT_COMPARABLE"


def _migration_error(verification: EventMigrationVerification) -> LegacyEventMigrationError:
    return LegacyEventMigrationError(
        verification.message,
        status=_migration_integrity_status(verification.state),
    )


_EVENT_HISTORY_TRANSACTION_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "transaction_id",
        "run_id",
        "operation",
        "canonical_event_path",
        "created_at_utc",
        "record_sha256",
    }
)
_EVENT_HISTORY_TRANSACTION_MIGRATION_FIELDS = _EVENT_HISTORY_TRANSACTION_COMMON_FIELDS | {
    "legacy_source_path",
    "migration_record_path",
    "event_history_origin_path",
    "legacy_size_bytes",
    "legacy_sha256",
    "canonical_size_bytes",
    "canonical_sha256",
    "migration_record",
    "migration_record_file_sha256",
}
_EVENT_HISTORY_TRANSACTION_APPEND_FIELDS = _EVENT_HISTORY_TRANSACTION_COMMON_FIELDS | {
    "append_surface",
    "expected_event_history_origin",
    "event_history_origin_record_file_sha256",
    "event_history_origin_record_self_sha256",
    "canonical_origin_identity_sha256",
    "canonical_prefix_size_bytes",
    "canonical_prefix_sha256",
    "migration_required",
    "migration_record_sha256",
    "request_sha256",
    "canonical_pre_size_bytes",
    "canonical_pre_sha256",
    "append_payload_base64",
    "append_separator_size_bytes",
    "append_payload_size_bytes",
    "append_payload_sha256",
    "canonical_post_size_bytes",
    "canonical_post_sha256",
}


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (strict_json_dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _event_transaction_checkpoint(stage: str, directory: Path) -> None:
    """Fault-injection seam for durability tests; production execution is a no-op."""


def _validate_sha256_field(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LegacyEventMigrationError(f"Event-history transaction field {field_name} is invalid.")


def _validate_event_history_transaction(value: dict[str, Any], run_id: str) -> None:
    operation = value.get("operation")
    expected_fields = {
        EVENT_HISTORY_TRANSACTION_MIGRATION: _EVENT_HISTORY_TRANSACTION_MIGRATION_FIELDS,
        EVENT_HISTORY_TRANSACTION_APPEND: _EVENT_HISTORY_TRANSACTION_APPEND_FIELDS,
    }.get(operation)
    if expected_fields is None or set(value) != expected_fields:
        raise LegacyEventMigrationError("Event-history transaction schema or fields are invalid.")
    if (
        value.get("schema_version") != EVENT_HISTORY_TRANSACTION_SCHEMA
        or value.get("run_id") != run_id
        or value.get("canonical_event_path") != EVENT_FILENAME
        or not isinstance(value.get("transaction_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", value["transaction_id"]) is None
        or not isinstance(value.get("created_at_utc"), str)
        or _timestamp(value["created_at_utc"]) is None
    ):
        raise LegacyEventMigrationError("Event-history transaction identity is invalid.")
    _validate_sha256_field(value.get("record_sha256"), "record_sha256")
    if value["record_sha256"] != _migration_record_sha256(value):
        raise LegacyEventMigrationError("Event-history transaction self-hash is invalid.")

    if operation == EVENT_HISTORY_TRANSACTION_MIGRATION:
        if (
            value.get("legacy_source_path") != LEGACY_EVENT_FILENAME
            or value.get("migration_record_path") != LEGACY_MIGRATION_FILENAME
            or value.get("event_history_origin_path") != EVENT_HISTORY_ORIGIN_FILENAME
        ):
            raise LegacyEventMigrationError("Event-history migration transaction paths are invalid.")
        for field_name in ("legacy_size_bytes", "canonical_size_bytes"):
            if type(value.get(field_name)) is not int or value[field_name] < 0:
                raise LegacyEventMigrationError(f"Event-history transaction field {field_name} is invalid.")
        for field_name in (
            "legacy_sha256",
            "canonical_sha256",
            "migration_record_file_sha256",
        ):
            _validate_sha256_field(value.get(field_name), field_name)
        migration_record = value.get("migration_record")
        if not isinstance(migration_record, dict):
            raise LegacyEventMigrationError("Event-history transaction migration record is invalid.")
        _validate_migration_record(migration_record, run_id)
        if value["migration_record_file_sha256"] != _bytes_sha256(_json_file_bytes(migration_record)):
            raise LegacyEventMigrationError("Event-history transaction binds different migration-record bytes.")
        if (
            value["legacy_size_bytes"] != migration_record["legacy_size_bytes"]
            or value["legacy_sha256"] != migration_record["legacy_sha256"]
            or value["legacy_size_bytes"] != migration_record["canonical_prefix_size_bytes"]
            or value["legacy_sha256"] != migration_record["canonical_prefix_sha256"]
        ):
            raise LegacyEventMigrationError("Event-history transaction migration bindings disagree.")
        if migration_record["migration_status"] == "migrated" and (
            value["canonical_size_bytes"] != value["legacy_size_bytes"]
            or value["canonical_sha256"] != value["legacy_sha256"]
        ):
            raise LegacyEventMigrationError("Event-history transaction migrated canonical identity is invalid.")
        return

    if value.get("append_surface") not in EVENT_APPEND_SURFACES:
        raise LegacyEventMigrationError("Event-history append transaction surface is invalid.")
    if value.get("expected_event_history_origin") not in EVENT_HISTORY_ORIGIN_STATES:
        raise LegacyEventMigrationError("Event-history append transaction origin is invalid.")
    for field_name in (
        "request_sha256",
        "event_history_origin_record_file_sha256",
        "event_history_origin_record_self_sha256",
        "canonical_origin_identity_sha256",
        "canonical_prefix_sha256",
        "canonical_pre_sha256",
        "append_payload_sha256",
        "canonical_post_sha256",
    ):
        _validate_sha256_field(value.get(field_name), field_name)
    for field_name in (
        "canonical_prefix_size_bytes",
        "canonical_pre_size_bytes",
        "append_separator_size_bytes",
        "append_payload_size_bytes",
        "canonical_post_size_bytes",
    ):
        if type(value.get(field_name)) is not int or value[field_name] < 0:
            raise LegacyEventMigrationError(f"Event-history transaction field {field_name} is invalid.")
    if value["append_separator_size_bytes"] not in {0, 1}:
        raise LegacyEventMigrationError("Event-history transaction append separator size is invalid.")
    migrated = value["expected_event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY
    if value.get("migration_required") is not migrated:
        raise LegacyEventMigrationError("Event-history append transaction migration requirement is invalid.")
    migration_record_sha256 = value.get("migration_record_sha256")
    if migrated:
        _validate_sha256_field(migration_record_sha256, "migration_record_sha256")
    elif migration_record_sha256 is not None:
        raise LegacyEventMigrationError("A native append transaction carries a migration-record identity.")
    encoded = value.get("append_payload_base64")
    if not isinstance(encoded, str):
        raise LegacyEventMigrationError("Event-history transaction append payload is invalid.")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise LegacyEventMigrationError("Event-history transaction append payload is malformed.") from exc
    if (
        len(payload) != value["append_payload_size_bytes"]
        or _bytes_sha256(payload) != value["append_payload_sha256"]
        or value["canonical_post_size_bytes"] != value["canonical_pre_size_bytes"] + value["append_payload_size_bytes"]
    ):
        raise LegacyEventMigrationError("Event-history transaction append payload bindings disagree.")
    event = _product_event_from_append_transaction(value)
    request_event = event.to_dict()
    if value["append_surface"] == EVENT_APPEND_SURFACE_REPOSITORY:
        expected_request_sha256 = event_append_request_sha256({"surface": "EventRepository", "event": request_event})
    else:
        request_event.pop("timestamp", None)
        expected_request_sha256 = event_append_request_sha256({"surface": "RunState", "event": request_event})
    if value["request_sha256"] != expected_request_sha256:
        raise LegacyEventMigrationError("Event-history append transaction request and ProductEvent disagree.")


def _event_history_transaction_path(directory: Path) -> Path:
    return directory / EVENT_HISTORY_TRANSACTION_FILENAME


def pending_event_history_transaction(run_id: str, directory: str | Path) -> dict[str, Any] | None:
    """Load and authenticate a live transaction without recovering it."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise LegacyEventMigrationError("Product run ID is invalid for event-history transaction recovery.")
    path = _event_history_transaction_path(Path(directory))
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVENT_HISTORY_TRANSACTION_BYTES:
        raise LegacyEventMigrationError("Event-history transaction record is irregular or oversized.")
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegacyEventMigrationError("Event-history transaction record is malformed.") from exc
    if not isinstance(value, dict):
        raise LegacyEventMigrationError("Event-history transaction record is not a JSON object.")
    _validate_event_history_transaction(value, run_id)
    return value


def _write_event_history_transaction(directory: Path, intent: dict[str, Any]) -> None:
    path = _event_history_transaction_path(directory)
    if path.exists() or path.is_symlink():
        raise LegacyEventMigrationError("Another event-history transaction is already active.")
    _atomic_json(path, intent)


def _clear_event_history_transaction(directory: Path, intent: Mapping[str, Any]) -> None:
    current = pending_event_history_transaction(str(intent["run_id"]), directory)
    if current is None or current["record_sha256"] != intent["record_sha256"]:
        raise LegacyEventMigrationError("Event-history transaction changed before commit.")
    _event_history_transaction_path(directory).unlink()
    _fsync_parent_directory(directory)


def _build_migration_transaction(
    run_id: str,
    *,
    legacy_bytes: bytes,
    canonical_bytes: bytes,
    migration_record: dict[str, Any],
) -> dict[str, Any]:
    intent: dict[str, Any] = {
        "schema_version": EVENT_HISTORY_TRANSACTION_SCHEMA,
        "transaction_id": uuid.uuid4().hex,
        "run_id": run_id,
        "operation": EVENT_HISTORY_TRANSACTION_MIGRATION,
        "canonical_event_path": EVENT_FILENAME,
        "legacy_source_path": LEGACY_EVENT_FILENAME,
        "migration_record_path": LEGACY_MIGRATION_FILENAME,
        "event_history_origin_path": EVENT_HISTORY_ORIGIN_FILENAME,
        "legacy_size_bytes": len(legacy_bytes),
        "legacy_sha256": _bytes_sha256(legacy_bytes),
        "canonical_size_bytes": len(canonical_bytes),
        "canonical_sha256": _bytes_sha256(canonical_bytes),
        "migration_record": migration_record,
        "migration_record_file_sha256": _bytes_sha256(_json_file_bytes(migration_record)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    intent["record_sha256"] = _migration_record_sha256(intent)
    _validate_event_history_transaction(intent, run_id)
    return intent


def _complete_migration_transaction(run_id: str, directory: Path, intent: dict[str, Any]) -> None:
    _validate_event_history_transaction(intent, run_id)
    if intent["operation"] != EVENT_HISTORY_TRANSACTION_MIGRATION:
        raise LegacyEventMigrationError("The live event-history transaction is not a migration transaction.")
    legacy = directory / LEGACY_EVENT_FILENAME
    canonical = directory / EVENT_FILENAME
    record_path = directory / LEGACY_MIGRATION_FILENAME
    if legacy.is_symlink() or not legacy.is_file():
        raise LegacyEventMigrationError("An incomplete migration requires its exact retained legacy source.")
    legacy_bytes = _read_event_bytes(legacy, "Legacy event stream")
    if len(legacy_bytes) != intent["legacy_size_bytes"] or _bytes_sha256(legacy_bytes) != intent["legacy_sha256"]:
        raise LegacyEventMigrationError("Legacy event stream changed during migration recovery.")

    migration_record = dict(intent["migration_record"])
    if not canonical.exists() and not canonical.is_symlink():
        if migration_record["migration_status"] != "migrated":
            raise LegacyEventMigrationError("A reconciled canonical stream disappeared during migration recovery.")
        _atomic_bytes(canonical, legacy_bytes)
    if canonical.is_symlink() or not canonical.is_file():
        raise LegacyEventMigrationError("Canonical event stream is irregular during migration recovery.")
    canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
    if (
        len(canonical_bytes) != intent["canonical_size_bytes"]
        or _bytes_sha256(canonical_bytes) != intent["canonical_sha256"]
    ):
        raise LegacyEventMigrationError("Canonical event stream changed during migration recovery.")

    if record_path.exists() or record_path.is_symlink():
        if record_path.is_symlink() or not record_path.is_file():
            raise LegacyEventMigrationError("Migration record is irregular during transaction recovery.")
        if _bytes_sha256(record_path.read_bytes()) != intent["migration_record_file_sha256"]:
            raise LegacyEventMigrationError("Migration record changed during transaction recovery.")
        _load_migration_record(record_path, run_id)
    else:
        _atomic_json(record_path, migration_record)
    _event_transaction_checkpoint("migration_record_published", directory)
    record_event_history_origin(
        run_id,
        directory,
        expected_origin=EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
        allow_binding_population=True,
    )
    _event_transaction_checkpoint("migration_origin_persisted", directory)
    _clear_event_history_transaction(directory, intent)


def event_append_request_sha256(value: Mapping[str, Any]) -> str:
    encoded = strict_json_dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _bytes_sha256(encoded)


def _product_event_from_append_transaction(intent: Mapping[str, Any]) -> ProductEvent:
    try:
        payload = base64.b64decode(str(intent["append_payload_base64"]).encode("ascii"), validate=True)
        separator_size = int(intent["append_separator_size_bytes"])
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise LegacyEventMigrationError("Event-history append transaction payload cannot be reconstructed.") from exc
    if separator_size == 1:
        if not payload.startswith(b"\n"):
            raise LegacyEventMigrationError("Event-history append transaction separator is invalid.")
    elif separator_size != 0 or payload.startswith(b"\n"):
        raise LegacyEventMigrationError("Event-history append transaction separator is invalid.")
    event_row = payload[separator_size:]
    if not event_row.endswith(b"\n") or b"\n" in event_row[:-1]:
        raise LegacyEventMigrationError("Event-history append transaction does not contain one exact event row.")
    try:
        value = strict_json_loads(event_row)
        if not isinstance(value, dict):
            raise ValueError("ProductEvent row is not an object.")
        event = ProductEvent.from_dict(value)
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise LegacyEventMigrationError("Event-history append transaction ProductEvent is invalid.") from exc
    if event.run_id != intent.get("run_id"):
        raise LegacyEventMigrationError("Event-history append transaction ProductEvent has the wrong run identity.")
    canonical_row = (
        strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    if event_row != canonical_row:
        raise LegacyEventMigrationError("Event-history append transaction ProductEvent bytes are not canonical.")
    return event


def _validate_append_origin_binding(directory: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(intent["run_id"])
    origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
    if origin_path.is_symlink() or not origin_path.is_file():
        raise LegacyEventMigrationError(
            "The append transaction requires its exact preexisting event-history origin record."
        )
    origin_bytes = origin_path.read_bytes()
    if _bytes_sha256(origin_bytes) != intent["event_history_origin_record_file_sha256"]:
        raise LegacyEventMigrationError("The preexisting event-history origin file changed during append recovery.")
    origin = _load_event_history_origin(origin_path, run_id)
    comparisons = {
        "event_history_origin": "expected_event_history_origin",
        "record_sha256": "event_history_origin_record_self_sha256",
        "canonical_event_identity_sha256": "canonical_origin_identity_sha256",
        "canonical_prefix_size_bytes": "canonical_prefix_size_bytes",
        "canonical_prefix_sha256": "canonical_prefix_sha256",
        "migration_required": "migration_required",
        "migration_record_sha256": "migration_record_sha256",
    }
    mismatched = [
        origin_field
        for origin_field, intent_field in comparisons.items()
        if origin.get(origin_field) != intent.get(intent_field)
    ]
    if mismatched:
        raise LegacyEventMigrationError(
            "The preexisting event-history origin identities changed during append recovery: " + ", ".join(mismatched)
        )
    if origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
        record_path = directory / LEGACY_MIGRATION_FILENAME
        if record_path.is_symlink() or not record_path.is_file():
            raise LegacyEventMigrationError("The append transaction requires its exact migration record.")
        _load_migration_record(record_path, run_id)
        if _bytes_sha256(record_path.read_bytes()) != intent["migration_record_sha256"]:
            raise LegacyEventMigrationError("The migration record changed during append recovery.")
    return origin


def _validate_append_authoritative_bindings(
    directory: Path,
    intent: Mapping[str, Any],
    origin: Mapping[str, Any],
) -> None:
    declarations = _declared_event_history_origin_bindings(directory, str(intent["run_id"]))
    expected = _event_history_origin_bindings(
        origin,
        str(intent["event_history_origin_record_file_sha256"]),
        str(intent["canonical_pre_sha256"]),
    )
    allowed_current_identities = {
        intent["canonical_pre_sha256"],
        intent["canonical_post_sha256"],
    }
    for source_name, declared in declarations:
        immutable_mismatches = [
            key
            for key, value in declared.items()
            if key != "event_canonical_current_identity_sha256" and expected.get(key) != value
        ]
        if immutable_mismatches:
            raise LegacyEventMigrationError(
                f"Authoritative {source_name} event-history bindings changed during append recovery: "
                + ", ".join(immutable_mismatches)
            )
        if declared["event_canonical_current_identity_sha256"] not in allowed_current_identities:
            raise LegacyEventMigrationError(
                f"Authoritative {source_name} current canonical identity is neither the append preimage nor postimage."
            )


def _validate_append_separator(preimage: bytes, intent: Mapping[str, Any]) -> None:
    expected = 1 if preimage and not preimage.endswith(b"\n") else 0
    if intent["append_separator_size_bytes"] != expected:
        raise LegacyEventMigrationError("The append transaction separator disagrees with its canonical preimage.")


def _build_append_transaction(
    run_id: str,
    *,
    directory: Path,
    canonical_bytes: bytes,
    line: bytes,
    request_sha256: str,
    expected_origin: str,
    append_surface: str,
) -> dict[str, Any]:
    if not line or not line.endswith(b"\n") or len(line) > MAX_EVENT_ROW_BYTES:
        raise LegacyEventMigrationError("The requested canonical event row is invalid or oversized.")
    origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
    if origin_path.is_symlink() or not origin_path.is_file():
        raise LegacyEventMigrationError("A new append transaction requires its preexisting event-history origin.")
    origin = _load_event_history_origin(origin_path, run_id)
    if origin["event_history_origin"] != expected_origin:
        raise LegacyEventMigrationError("A new append transaction received the wrong event-history origin.")
    separator_size = 1 if canonical_bytes and not canonical_bytes.endswith(b"\n") else 0
    payload = (b"\n" if separator_size else b"") + line
    post_bytes = canonical_bytes + payload
    intent: dict[str, Any] = {
        "schema_version": EVENT_HISTORY_TRANSACTION_SCHEMA,
        "transaction_id": uuid.uuid4().hex,
        "run_id": run_id,
        "operation": EVENT_HISTORY_TRANSACTION_APPEND,
        "canonical_event_path": EVENT_FILENAME,
        "append_surface": append_surface,
        "expected_event_history_origin": expected_origin,
        "event_history_origin_record_file_sha256": _bytes_sha256(origin_path.read_bytes()),
        "event_history_origin_record_self_sha256": origin["record_sha256"],
        "canonical_origin_identity_sha256": origin["canonical_event_identity_sha256"],
        "canonical_prefix_size_bytes": origin["canonical_prefix_size_bytes"],
        "canonical_prefix_sha256": origin["canonical_prefix_sha256"],
        "migration_required": origin["migration_required"],
        "migration_record_sha256": origin["migration_record_sha256"],
        "request_sha256": request_sha256,
        "canonical_pre_size_bytes": len(canonical_bytes),
        "canonical_pre_sha256": _bytes_sha256(canonical_bytes),
        "append_payload_base64": base64.b64encode(payload).decode("ascii"),
        "append_separator_size_bytes": separator_size,
        "append_payload_size_bytes": len(payload),
        "append_payload_sha256": _bytes_sha256(payload),
        "canonical_post_size_bytes": len(post_bytes),
        "canonical_post_sha256": _bytes_sha256(post_bytes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    intent["record_sha256"] = _migration_record_sha256(intent)
    _validate_event_history_transaction(intent, run_id)
    bound_origin = _validate_append_origin_binding(directory, intent)
    _validate_append_authoritative_bindings(directory, intent, bound_origin)
    return intent


def _finalize_event_repository_append(
    directory: Path,
    intent: Mapping[str, Any],
    event_id: int,
    origin: Mapping[str, Any],
) -> None:
    if intent["append_surface"] != EVENT_APPEND_SURFACE_REPOSITORY:
        return
    state_path = directory / "state.json"
    if not state_path.exists() and not state_path.is_symlink():
        return
    if state_path.is_symlink() or not state_path.is_file() or state_path.stat().st_size > MAX_EVENT_ROW_BYTES:
        raise LegacyEventMigrationError("Product run state is irregular during append recovery.")
    try:
        state = strict_json_loads(state_path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise LegacyEventMigrationError("Product run state is malformed during append recovery.") from exc
    if not isinstance(state, dict) or state.get("run_id") != intent["run_id"]:
        raise LegacyEventMigrationError("Product run state has the wrong identity during append recovery.")
    event = _product_event_from_append_transaction(intent)
    state.update(
        {
            "status": event.status.value,
            "stage": event.stage,
            "message": event.message,
            "last_durable_event": {
                "event_id": event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
            },
        }
    )
    if origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
        state["event_stream_migration"] = _load_migration_record(
            directory / LEGACY_MIGRATION_FILENAME,
            str(intent["run_id"]),
        )
    state.update(
        _event_history_origin_bindings(
            origin,
            _bytes_sha256((directory / EVENT_HISTORY_ORIGIN_FILENAME).read_bytes()),
            _bytes_sha256((directory / EVENT_FILENAME).read_bytes()),
        )
    )
    if event.status.value in TERMINAL_STATUSES:
        state["ended_at"] = event.timestamp
    _atomic_json(state_path, state)


def append_event_transactionally(
    run_id: str,
    directory: str | Path,
    *,
    line: bytes | None,
    request_sha256: str | None,
    expected_origin: str | None,
    append_surface: str | None,
) -> int:
    """Append exactly once and commit its current-history binding as one retryable transaction."""

    if request_sha256 is not None:
        _validate_sha256_field(request_sha256, "request_sha256")
    directory = Path(directory)
    _require_mutable_event_history(directory)
    canonical = directory / EVENT_FILENAME
    intent = pending_event_history_transaction(run_id, directory)
    if intent is None:
        if line is None or request_sha256 is None or append_surface not in EVENT_APPEND_SURFACES:
            raise LegacyEventMigrationError("A new append transaction requires an exact ProductEvent request.")
        if expected_origin not in EVENT_HISTORY_ORIGIN_STATES:
            raise LegacyEventMigrationError("A new append transaction requires a controlled event-history origin.")
        if canonical.is_symlink() or not canonical.is_file():
            raise LegacyEventMigrationError("Canonical event stream is missing or irregular before append.")
        intent = _build_append_transaction(
            run_id,
            directory=directory,
            canonical_bytes=_read_event_bytes(canonical, "Canonical event stream"),
            line=line,
            request_sha256=request_sha256,
            expected_origin=expected_origin,
            append_surface=append_surface,
        )
        _require_mutable_event_history(directory)
        _write_event_history_transaction(directory, intent)
        _event_transaction_checkpoint("append_intent_published", directory)
    elif intent["operation"] != EVENT_HISTORY_TRANSACTION_APPEND:
        raise LegacyEventMigrationError("The live event-history transaction is not an append transaction.")
    if request_sha256 is not None and intent["request_sha256"] != request_sha256:
        raise LegacyEventMigrationError("The live append transaction belongs to a different event request.")
    if expected_origin is not None and intent["expected_event_history_origin"] != expected_origin:
        raise LegacyEventMigrationError("The live append transaction binds a different event-history origin.")
    if append_surface is not None and intent["append_surface"] != append_surface:
        raise LegacyEventMigrationError("The live append transaction belongs to a different append surface.")

    bound_origin = _validate_append_origin_binding(directory, intent)
    _validate_append_authoritative_bindings(directory, intent, bound_origin)
    if canonical.is_symlink() or not canonical.is_file():
        raise LegacyEventMigrationError("Canonical event stream is missing or irregular during append recovery.")
    current = _read_event_bytes(canonical, "Canonical event stream")
    current_identity = (len(current), _bytes_sha256(current))
    pre_identity = (intent["canonical_pre_size_bytes"], intent["canonical_pre_sha256"])
    post_identity = (intent["canonical_post_size_bytes"], intent["canonical_post_sha256"])
    if current_identity == pre_identity:
        _validate_append_separator(current, intent)
        payload = base64.b64decode(intent["append_payload_base64"].encode("ascii"), validate=True)
        _require_mutable_event_history(directory)
        with canonical.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        current = _read_event_bytes(canonical, "Canonical event stream")
        if (len(current), _bytes_sha256(current)) != post_identity:
            raise LegacyEventMigrationError("Canonical event append did not reach its committed identity.")
    elif current_identity != post_identity:
        pre_size = int(intent["canonical_pre_size_bytes"])
        payload = base64.b64decode(intent["append_payload_base64"].encode("ascii"), validate=True)
        retained_prefix = current[:pre_size]
        partial_payload = current[pre_size:]
        exact_interrupted_write = (
            len(current) > pre_size
            and len(current) < int(intent["canonical_post_size_bytes"])
            and _bytes_sha256(retained_prefix) == intent["canonical_pre_sha256"]
            and payload.startswith(partial_payload)
            and len(partial_payload) < len(payload)
        )
        if not exact_interrupted_write:
            raise LegacyEventMigrationError("Canonical event stream changed outside the live append transaction.")
        _validate_append_separator(retained_prefix, intent)
        with canonical.open("r+b") as handle:
            handle.truncate(pre_size)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        current = _read_event_bytes(canonical, "Canonical event stream")
        if (len(current), _bytes_sha256(current)) != post_identity:
            raise LegacyEventMigrationError(
                "Interrupted canonical append recovery did not reach its committed identity."
            )
    else:
        _validate_append_separator(current[: int(intent["canonical_pre_size_bytes"])], intent)
    with canonical.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    current = _read_event_bytes(canonical, "Canonical event stream")
    if (len(current), _bytes_sha256(current)) != post_identity:
        raise LegacyEventMigrationError("Canonical event postimage changed before binding publication.")
    bound_origin = _validate_append_origin_binding(directory, intent)
    _validate_append_authoritative_bindings(directory, intent, bound_origin)
    _event_transaction_checkpoint("append_postimage_fsynced", directory)
    _event_transaction_checkpoint("append_event_fsynced", directory)
    origin = record_event_history_origin(
        run_id,
        directory,
        expected_origin=str(intent["expected_event_history_origin"]),
        update_current_binding=True,
        require_existing=True,
    )
    event_id = _line_count(canonical)
    _finalize_event_repository_append(directory, intent, event_id, origin)
    _event_transaction_checkpoint("append_bindings_persisted", directory)
    _clear_event_history_transaction(directory, intent)
    return event_id


def verify_event_migration(
    run_id: str,
    directory: Path,
    *,
    expected_evidence_sha256: str | None = None,
    migration_required: bool = False,
    origin_required: bool = False,
) -> EventMigrationVerification:
    """Verify retained migration evidence without inferring trust from source absence.

    ``NO_MIGRATION`` describes a native stream and is not itself migration
    evidence. Once a migration record exists (or a caller binds one), only the
    two ``VERIFIED_*`` states are resume-compatible.

    The authoritative event-history origin record is the versioned run-state
    fact that a canonical stream originated from a legacy stream.  A recorded
    ``migrated_legacy`` origin makes the migration record mandatory forever, so
    deleting the record can never downgrade the run to ``NO_MIGRATION``.
    """

    if not RUN_ID_PATTERN.fullmatch(run_id):
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "Product run ID is invalid for event migration verification.",
        )
    directory = Path(directory)
    try:
        pending = pending_event_history_transaction(run_id, directory)
    except LegacyEventMigrationError as exc:
        return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc))
    if pending is not None:
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "An incomplete event-history transaction requires explicit controlled recovery before verification.",
            details={
                "transaction_operation": pending["operation"],
                "transaction_record_sha256": pending["record_sha256"],
            },
        )
    legacy = directory / LEGACY_EVENT_FILENAME
    canonical = directory / EVENT_FILENAME
    record_path = directory / LEGACY_MIGRATION_FILENAME
    origin_path = directory / EVENT_HISTORY_ORIGIN_FILENAME
    record_exists = record_path.exists() or record_path.is_symlink()
    legacy_exists = legacy.exists() or legacy.is_symlink()
    canonical_exists = canonical.exists() or canonical.is_symlink()

    origin: dict[str, Any] | None = None
    origin_record_sha256: str | None = None
    if origin_path.exists() or origin_path.is_symlink():
        try:
            origin = _load_event_history_origin(origin_path, run_id)
            origin_record_sha256 = _bytes_sha256(origin_path.read_bytes())
        except (LegacyEventMigrationError, OSError) as exc:
            return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc))
    try:
        declared_bindings = _declared_event_history_origin_bindings(directory, run_id)
    except LegacyEventMigrationError as exc:
        return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc))
    if origin is None and declared_bindings:
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "Authoritative run state requires an event-history origin record, but that record is missing.",
        )
    if origin is not None and origin_record_sha256 is not None:
        expected_bindings = _event_history_origin_bindings(origin, origin_record_sha256, _bytes_sha256(b""))
        for source_name, declared in declared_bindings:
            mismatched = [
                key
                for key, value in declared.items()
                if key != "event_canonical_current_identity_sha256" and expected_bindings.get(key) != value
            ]
            if mismatched:
                return _migration_verification(
                    EventMigrationState.INVALID_RECORD,
                    run_id,
                    f"Authoritative {source_name} event-history bindings disagree: " + ", ".join(mismatched),
                )
    if origin is not None:
        if origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_NATIVE and (record_exists or legacy_exists):
            return _migration_verification(
                EventMigrationState.CONFLICTING_STREAMS,
                run_id,
                "A recorded native event-history origin conflicts with retained migration evidence.",
                details=_origin_evidence_details(
                    origin, origin_record_sha256, derived_origin=EVENT_HISTORY_ORIGIN_NATIVE, migration_required=False
                ),
            )
        if origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
            migration_required = True

    if origin is None and (origin_required or (canonical_exists and not legacy_exists and not record_exists)):
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "A materialized or resumable canonical event history requires its authoritative origin record.",
        )

    if not record_exists:
        if legacy_exists:
            if legacy.is_symlink() or not legacy.is_file():
                return _migration_verification(
                    EventMigrationState.NOT_COMPARABLE,
                    run_id,
                    "Legacy event stream is not a regular retained source file.",
                )
            try:
                legacy_bytes = _read_event_bytes(legacy, "Legacy event stream")
                _validate_event_stream_bytes(legacy_bytes, run_id, "Legacy event stream")
            except LegacyEventMigrationError as exc:
                return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc))
            if canonical_exists:
                if canonical.is_symlink() or not canonical.is_file():
                    return _migration_verification(
                        EventMigrationState.CONFLICTING_STREAMS,
                        run_id,
                        "Canonical and legacy event streams do not have a provable regular-file relationship.",
                    )
                try:
                    canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
                except LegacyEventMigrationError as exc:
                    return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc))
                separator_valid = not (
                    len(canonical_bytes) > len(legacy_bytes)
                    and legacy_bytes
                    and not legacy_bytes.endswith(b"\n")
                    and canonical_bytes[len(legacy_bytes) : len(legacy_bytes) + 1] != b"\n"
                )
                if not canonical_bytes.startswith(legacy_bytes) or not separator_valid:
                    return _migration_verification(
                        EventMigrationState.CONFLICTING_STREAMS,
                        run_id,
                        "Canonical and legacy event streams conflict and cannot be proven as one migration.",
                    )
            return _migration_verification(
                EventMigrationState.INVALID_RECORD,
                run_id,
                "Legacy migration evidence requires a valid migration record before resume.",
            )

        canonical_bytes = b""
        if canonical_exists:
            if canonical.is_symlink() or not canonical.is_file():
                return _migration_verification(
                    EventMigrationState.NOT_COMPARABLE,
                    run_id,
                    "Canonical event stream is not a regular file.",
                )
            try:
                canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
            except LegacyEventMigrationError as exc:
                return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc))
            native_details = {
                "canonical_present": True,
                "canonical_size_bytes": len(canonical_bytes),
                "canonical_sha256": _bytes_sha256(canonical_bytes),
            }
        else:
            native_details = {
                "canonical_present": False,
                "canonical_size_bytes": 0,
                "canonical_sha256": _bytes_sha256(b""),
            }
        try:
            _validate_current_canonical_bindings(declared_bindings, native_details["canonical_sha256"])
        except LegacyEventMigrationError as exc:
            return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc))
        native_details.update(
            _origin_evidence_details(
                origin,
                origin_record_sha256,
                derived_origin=EVENT_HISTORY_ORIGIN_NATIVE,
                migration_required=migration_required,
            )
        )
        if origin is not None and origin["event_history_origin"] == EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY:
            return _migration_verification(
                EventMigrationState.INVALID_RECORD,
                run_id,
                "The recorded migrated_legacy event-history origin requires migration evidence, but its "
                "mandatory migration record is missing; refusing native classification.",
                details=native_details,
            )
        if migration_required or _migration_marker_requires_record(directory, run_id):
            return _migration_verification(
                EventMigrationState.NOT_COMPARABLE,
                run_id,
                "Migration evidence is required but its migration record is missing.",
                details=native_details,
            )
        if origin is not None:
            prefix_size = int(origin["canonical_prefix_size_bytes"])
            if len(canonical_bytes) < prefix_size:
                return _migration_verification(
                    EventMigrationState.NOT_COMPARABLE,
                    run_id,
                    "The native canonical event stream is shorter than its recorded origin prefix.",
                    details=native_details,
                )
            if _bytes_sha256(canonical_bytes[:prefix_size]) != origin["canonical_prefix_sha256"]:
                return _migration_verification(
                    EventMigrationState.NOT_COMPARABLE,
                    run_id,
                    "The native canonical origin prefix hash changed after origin recording.",
                    details=native_details,
                )
            native_details["canonical_prefix_size_bytes"] = prefix_size
            native_details["canonical_prefix_sha256"] = origin["canonical_prefix_sha256"]
        else:
            native_details["canonical_prefix_size_bytes"] = 0
            native_details["canonical_prefix_sha256"] = _bytes_sha256(b"")
        verification = _migration_verification(
            EventMigrationState.NO_MIGRATION,
            run_id,
            "No legacy migration is recorded or required.",
            details=native_details,
        )
        if expected_evidence_sha256 is not None and verification.evidence_sha256 != expected_evidence_sha256:
            return _migration_verification(
                EventMigrationState.NOT_COMPARABLE,
                run_id,
                "Event migration evidence changed after continuation validation.",
                details=native_details,
            )
        return verification

    try:
        record = _load_migration_record(record_path, run_id)
    except (LegacyEventMigrationError, OSError) as exc:
        return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc))

    if origin is None:
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "Migration record exists without its authoritative event-history origin record.",
            record=record,
        )
    try:
        record_bytes_sha256 = _bytes_sha256(record_path.read_bytes())
    except OSError:
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "Legacy event migration record could not be reread for origin binding.",
            record=record,
        )
    if record_bytes_sha256 != origin["migration_record_sha256"]:
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "The authoritative event-history origin binds a different migration record.",
            record=record,
        )
    if (
        record["legacy_size_bytes"] != origin["legacy_source_size_bytes"]
        or record["legacy_sha256"] != origin["legacy_source_sha256"]
        or record["canonical_prefix_size_bytes"] != origin["canonical_prefix_size_bytes"]
        or record["canonical_prefix_sha256"] != origin["canonical_prefix_sha256"]
    ):
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "The authoritative event-history origin and migration record bindings disagree.",
            record=record,
        )

    if canonical.is_symlink() or not canonical.is_file():
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "The canonical event stream bound by the migration record is missing or not a regular file.",
            record=record,
        )
    try:
        canonical_bytes = _read_event_bytes(canonical, "Canonical event stream")
    except LegacyEventMigrationError as exc:
        return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc), record=record)
    try:
        _validate_current_canonical_bindings(declared_bindings, _bytes_sha256(canonical_bytes))
    except LegacyEventMigrationError as exc:
        return _migration_verification(EventMigrationState.INVALID_RECORD, run_id, str(exc), record=record)

    legacy_bytes: bytes | None = None
    if legacy_exists:
        if legacy.is_symlink() or not legacy.is_file():
            return _migration_verification(
                EventMigrationState.NOT_COMPARABLE,
                run_id,
                "Legacy event stream is not a regular retained source file.",
                record=record,
            )
        try:
            legacy_bytes = _read_event_bytes(legacy, "Legacy event stream")
        except LegacyEventMigrationError as exc:
            return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc), record=record)
        if len(legacy_bytes) != record["legacy_size_bytes"] or _bytes_sha256(legacy_bytes) != record["legacy_sha256"]:
            return _migration_verification(
                EventMigrationState.STALE_SOURCE_CHANGED,
                run_id,
                "Legacy event stream bytes changed after migration.",
                record=record,
            )

    prefix_size = int(record["canonical_prefix_size_bytes"])
    if len(canonical_bytes) < prefix_size:
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "The canonical event stream is shorter than its recorded legacy prefix.",
            record=record,
        )
    canonical_prefix = canonical_bytes[:prefix_size]
    if _bytes_sha256(canonical_prefix) != record["canonical_prefix_sha256"]:
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "The canonical legacy prefix hash changed after migration.",
            record=record,
        )
    if legacy_bytes is not None and canonical_prefix != legacy_bytes:
        return _migration_verification(
            EventMigrationState.CONFLICTING_STREAMS,
            run_id,
            "The canonical and retained legacy event streams no longer have an exact prefix relationship.",
            record=record,
        )

    try:
        prefix_metadata = _validate_event_stream_bytes(canonical_prefix, run_id, "Canonical legacy prefix")
    except LegacyEventMigrationError as exc:
        return _migration_verification(
            EventMigrationState.INVALID_CANONICAL_PREFIX,
            run_id,
            str(exc),
            record=record,
        )
    recorded_metadata = {
        "validated_event_count": record["validated_event_count"],
        "had_terminal_newline": record["had_terminal_newline"],
        "line_ending_summary": record["line_ending_summary"],
    }
    if prefix_metadata != recorded_metadata:
        return _migration_verification(
            EventMigrationState.INVALID_RECORD,
            run_id,
            "Legacy migration record validation metadata does not match the exact canonical prefix.",
            record=record,
        )
    try:
        _validate_event_stream_bytes(canonical_bytes, run_id, "Canonical event stream")
    except LegacyEventMigrationError as exc:
        return _migration_verification(EventMigrationState.NOT_COMPARABLE, run_id, str(exc), record=record)

    state = (
        EventMigrationState.VERIFIED_SOURCE_PRESENT
        if legacy_bytes is not None
        else EventMigrationState.VERIFIED_SOURCE_REMOVED
    )
    details = {
        "schema_version": record["schema_version"],
        "migration_record_sha256": record_bytes_sha256,
        "record_self_sha256": record.get("record_sha256"),
        "legacy_source_present": legacy_bytes is not None,
        "legacy_size_bytes": record["legacy_size_bytes"],
        "legacy_sha256": record["legacy_sha256"],
        "canonical_size_bytes": len(canonical_bytes),
        "canonical_sha256": _bytes_sha256(canonical_bytes),
        "canonical_prefix_size_bytes": prefix_size,
        "canonical_prefix_sha256": _bytes_sha256(canonical_prefix),
        "validated_event_count": prefix_metadata["validated_event_count"],
    }
    details.update(
        _origin_evidence_details(
            origin,
            origin_record_sha256,
            derived_origin=EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY,
            migration_required=True,
        )
    )
    verification = _migration_verification(
        state,
        run_id,
        (
            "Migration record, retained source, and canonical prefix verify exactly."
            if legacy_bytes is not None
            else "Migration record and canonical prefix verify under the explicit source-removal policy."
        ),
        record=record,
        details=details,
    )
    if expected_evidence_sha256 is not None and verification.evidence_sha256 != expected_evidence_sha256:
        return _migration_verification(
            EventMigrationState.NOT_COMPARABLE,
            run_id,
            "Event migration evidence changed after continuation validation.",
            record=record,
            details=details,
        )
    return verification


def _fsync_parent_directory(directory: Path) -> None:
    """Persist directory metadata where supported; Windows has no directory-fsync API."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _durability_barrier(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
    _fsync_parent_directory(path.parent)


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _durability_barrier(path)
    finally:
        temporary.unlink(missing_ok=True)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = strict_json_dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _durability_barrier(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_optional_text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


__all__ = [
    "EVENT_APPEND_SURFACE_REPOSITORY",
    "EVENT_APPEND_SURFACE_RUN_STATE",
    "EVENT_FILENAME",
    "EVENT_HISTORY_ORIGIN_FILENAME",
    "EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY",
    "EVENT_HISTORY_ORIGIN_NATIVE",
    "EVENT_HISTORY_ORIGIN_SCHEMA",
    "EVENT_HISTORY_ORIGIN_STATES",
    "EVENT_HISTORY_TRANSACTION_APPEND",
    "EVENT_HISTORY_TRANSACTION_FILENAME",
    "EVENT_HISTORY_TRANSACTION_MIGRATION",
    "EVENT_HISTORY_TRANSACTION_SCHEMA",
    "LEGACY_EVENT_FILENAME",
    "LEGACY_MIGRATION_FILENAME",
    "LEGACY_MIGRATION_SCHEMA",
    "LEGACY_SOURCE_REMOVAL_POLICY",
    "RUN_ID_PATTERN",
    "RUN_STATE_SCHEMA",
    "TERMINAL_STATUSES",
    "EventMigrationState",
    "EventMigrationVerification",
    "EventReplay",
    "EventRepository",
    "IndexedEvent",
    "LegacyEventMigrationError",
    "RunSnapshot",
    "append_event_transactionally",
    "event_append_request_sha256",
    "event_history_transaction_lock",
    "pending_event_history_transaction",
    "record_event_history_origin",
    "verify_event_migration",
]

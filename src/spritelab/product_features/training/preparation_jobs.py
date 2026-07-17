"""Durable, confined state for image-only baseline preparation jobs."""

from __future__ import annotations

import errno
import json
import os
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from spritelab.product_core import ProjectContext
from spritelab.training.campaign import stable_hash
from spritelab.utils.safe_fs import atomic_write_text, require_confined_path
from spritelab.v3.run_state import lock_file

PREPARATION_JOB_SCHEMA = "spritelab.training-preparation.job.v1"
PREPARATION_JOB_EVENT_SCHEMA = "spritelab.training-preparation.job-event.v1"
MAX_PREPARATION_EVENTS = 200
MAX_PREPARATION_EVENT_BYTES = 1_000_000


class PreparationJobStateError(ValueError):
    """Durable preparation state is malformed or conflicts with the active job."""


class PreparationJobRepository:
    """Persist one active/recent job plus append-only privacy-safe log events."""

    def __init__(self, context: ProjectContext) -> None:
        self.project_root = context.project_root.resolve()
        preparation_root = require_confined_path(
            self.project_root / ".spritelab" / "training-preparation",
            self.project_root,
        )
        self.root = require_confined_path(preparation_root / "jobs", preparation_root)
        self.state_path = require_confined_path(self.root / "state.json", self.root)
        self.events_path = require_confined_path(self.root / "events.jsonl", self.root)
        self.lock_path = require_confined_path(self.root / ".state.lock", self.root)
        self.recovery_lock_path = require_confined_path(self.root / ".lock-recovery.lock", self.root)
        self.stale_lock_path = require_confined_path(self.root / ".state.lock.stale", self.root)
        self.owner_token = uuid.uuid4().hex
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def reconstruct(self) -> dict[str, Any]:
        """Mark a worker lost across process restart as interrupted and retryable."""

        with self._lock:
            state = self._load_unlocked()
            if state["status"] != "running":
                return state
        with self._lock, self._state_guard():
            state = self._load_unlocked()
            if state["status"] != "running" or _pid_alive(state.get("worker_pid")):
                return state
            return self._transition_unlocked(
                state,
                status="interrupted",
                error={
                    "code": "training_preparation_interrupted",
                    "message": "The previous preparation worker stopped during application restart; retry is safe.",
                },
                message="Preparation was reconstructed as interrupted after application restart.",
            )

    def begin(self, identities: Mapping[str, str]) -> dict[str, Any]:
        with self._lock, self._state_guard():
            current = self._load_unlocked()
            if current["status"] == "running":
                raise PreparationJobStateError("A preparation job is already running.")
            required = {"input_identity", "source_identity", "config_identity", "code_identity"}
            if set(identities) != required or any(not str(value).strip() for value in identities.values()):
                raise PreparationJobStateError("Preparation identities are incomplete.")
            now = _now()
            state = {
                "schema_version": PREPARATION_JOB_SCHEMA,
                "job_id": uuid.uuid4().hex,
                "worker_pid": os.getpid(),
                "worker_owner": self.owner_token,
                "status": "running",
                "current": 0,
                "total": 0,
                "input_identity": str(identities["input_identity"]),
                "source_identity": str(identities["source_identity"]),
                "config_identity": str(identities["config_identity"]),
                "code_identity": str(identities["code_identity"]),
                "started_at": now,
                "updated_at": now,
                "error": None,
                "result": None,
                "result_identity": None,
                "logs": ["Preparation queued on a durable background worker."],
            }
            self._write_unlocked(state)
            self._record_event_unlocked(state, "Preparation queued on a durable background worker.")
            return dict(state)

    def progress(
        self,
        job_id: str,
        owner_token: str,
        current: int,
        total: int,
        message: str,
    ) -> dict[str, Any]:
        return self.transition(
            job_id,
            owner_token,
            status="running",
            current=current,
            total=total,
            message=message,
        )

    def transition(
        self,
        job_id: str,
        owner_token: str,
        *,
        status: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
        error: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._state_guard():
            state = self._load_unlocked()
            if state.get("job_id") != job_id:
                raise PreparationJobStateError("Preparation job identity changed.")
            if state.get("worker_owner") != owner_token or state.get("worker_pid") != os.getpid():
                raise PreparationJobStateError("Preparation job worker ownership changed.")
            if status not in {"running", "complete", "failed", "interrupted"}:
                raise PreparationJobStateError("Preparation job status is invalid.")
            return self._transition_unlocked(
                state,
                status=status,
                message=message,
                current=current,
                total=total,
                error=error,
                result=result,
            )

    def _transition_unlocked(
        self,
        state: dict[str, Any],
        *,
        status: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
        error: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if current is not None:
            state["current"] = max(0, int(current))
        if total is not None:
            state["total"] = max(0, int(total))
        state["status"] = status
        state["updated_at"] = _now()
        state["error"] = dict(error) if error is not None else None
        state["result"] = dict(result) if result is not None else None
        state["result_identity"] = stable_hash(dict(result)) if result is not None else None
        state["logs"] = [*state.get("logs", []), _safe_message(message)][-MAX_PREPARATION_EVENTS:]
        self._write_unlocked(state)
        self._record_event_unlocked(state, message)
        return dict(state)

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty()
        self._require_safe_regular(self.state_path, "Preparation job state")
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparationJobStateError("Preparation job state is unreadable.") from exc
        return self._validate(value)

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        require_confined_path(self.root, self.project_root)
        if self.state_path.exists():
            self._require_safe_regular(self.state_path, "Preparation job state")
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(state), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        atomic_write_text(self.state_path, payload)

    def _record_event_unlocked(self, state: Mapping[str, Any], message: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        events: list[dict[str, Any]] = []
        if self.events_path.exists():
            self._require_safe_regular(self.events_path, "Preparation event history")
            if self.events_path.stat().st_size > MAX_PREPARATION_EVENT_BYTES:
                raise PreparationJobStateError("Preparation event history exceeds its bounded size.")
            try:
                for line in self.events_path.read_text(encoding="utf-8").splitlines():
                    value = json.loads(line)
                    if not isinstance(value, dict) or value.get("schema_version") != PREPARATION_JOB_EVENT_SCHEMA:
                        raise ValueError("invalid event")
                    events.append(value)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise PreparationJobStateError("Preparation event history is invalid.") from exc
        event = {
            "schema_version": PREPARATION_JOB_EVENT_SCHEMA,
            "job_id": state["job_id"],
            "timestamp": state["updated_at"],
            "status": state["status"],
            "current": state["current"],
            "total": state["total"],
            "message": _safe_message(message),
        }
        events = [*events, event][-MAX_PREPARATION_EVENTS:]
        payload = "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in events)
        atomic_write_text(self.events_path, payload)

    def _require_safe_regular(self, path: Any, label: str) -> None:
        try:
            confined = require_confined_path(path, self.root)
        except ValueError as exc:
            raise PreparationJobStateError(f"{label} is unsafe.") from exc
        if not confined.is_file() or confined.stat().st_nlink != 1:
            raise PreparationJobStateError(f"{label} is unsafe.")

    @contextmanager
    def _state_guard(self) -> Any:
        acquired = False
        try:
            with lock_file(self.lock_path, timeout=0.5):
                acquired = True
                yield
                return
        except TimeoutError:
            if acquired:
                raise
            self._recover_stale_lock()
        acquired = False
        try:
            with lock_file(self.lock_path, timeout=5.0):
                acquired = True
                yield
        except TimeoutError as exc:
            if acquired:
                raise
            raise PreparationJobStateError("Preparation state is locked by another live process.") from exc

    def _recover_stale_lock(self) -> None:
        try:
            with lock_file(self.recovery_lock_path, timeout=1.0):
                if not self.lock_path.exists():
                    return
                self._require_safe_regular(self.lock_path, "Preparation state lock")
                try:
                    text = self.lock_path.read_text(encoding="utf-8")
                    prefix, separator, raw_pid = text.strip().partition("=")
                    if prefix != "pid" or not separator or not raw_pid.isdigit():
                        raise ValueError("invalid lock owner")
                    pid = int(raw_pid)
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    raise PreparationJobStateError("Preparation state lock ownership is invalid.") from exc
                if _pid_alive(pid):
                    raise PreparationJobStateError("Preparation state is locked by another live process.")
                if self.stale_lock_path.exists():
                    self._require_safe_regular(self.stale_lock_path, "Preparation stale-lock evidence")
                os.replace(self.lock_path, self.stale_lock_path)
        except TimeoutError as exc:
            raise PreparationJobStateError("Preparation lock recovery is already in progress.") from exc

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": PREPARATION_JOB_SCHEMA,
            "job_id": None,
            "worker_pid": None,
            "worker_owner": None,
            "status": "not_started",
            "current": 0,
            "total": 0,
            "input_identity": None,
            "source_identity": None,
            "config_identity": None,
            "code_identity": None,
            "started_at": None,
            "updated_at": None,
            "error": None,
            "result": None,
            "result_identity": None,
            "logs": [],
        }

    @staticmethod
    def _validate(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("schema_version") != PREPARATION_JOB_SCHEMA:
            raise PreparationJobStateError("Preparation job state schema is invalid.")
        if value.get("status") not in {"not_started", "running", "complete", "failed", "interrupted"}:
            raise PreparationJobStateError("Preparation job status is invalid.")
        if value.get("status") != "not_started" and not isinstance(value.get("job_id"), str):
            raise PreparationJobStateError("Preparation job identity is missing.")
        if value.get("status") != "not_started" and (
            isinstance(value.get("worker_pid"), bool)
            or not isinstance(value.get("worker_pid"), int)
            or not isinstance(value.get("worker_owner"), str)
        ):
            raise PreparationJobStateError("Preparation worker ownership is missing.")
        logs = value.get("logs")
        if not isinstance(logs, list) or not all(isinstance(item, str) for item in logs):
            raise PreparationJobStateError("Preparation job logs are invalid.")
        return dict(value)


def _safe_message(message: str) -> str:
    text = str(message).replace("\r", " ").replace("\n", " ").strip()
    return text[:1_000] or "Preparation state updated."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return False
    if value == os.getpid():
        return True
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH or (os.name == "nt" and exc.errno == errno.EINVAL):
            return False
        return True
    return True


__all__ = [
    "MAX_PREPARATION_EVENTS",
    "PREPARATION_JOB_EVENT_SCHEMA",
    "PREPARATION_JOB_SCHEMA",
    "PreparationJobRepository",
    "PreparationJobStateError",
]

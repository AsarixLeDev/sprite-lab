"""Durable, atomic orchestration state for Sprite Lab v3 runs."""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core import PRODUCT_EVENT_SCHEMA, ProductEvent, ProductStatus, strict_json_dumps
from spritelab.product_web.events import (
    EVENT_APPEND_SURFACE_RUN_STATE,
    EVENT_HISTORY_ORIGIN_NATIVE,
    EVENT_HISTORY_TRANSACTION_APPEND,
    LegacyEventMigrationError,
    append_event_transactionally,
    event_append_request_sha256,
    event_history_transaction_lock,
    pending_event_history_transaction,
    record_event_history_origin,
    verify_event_migration,
)
from spritelab.v3.config import ProjectConfig

RUN_SCHEMA = "spritelab.v3.run-state.v1"
EVENT_SCHEMA = PRODUCT_EVENT_SCHEMA


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _product_event_status(value: str) -> ProductStatus:
    try:
        return ProductStatus(value)
    except ValueError:
        return {
            "INCONCLUSIVE": ProductStatus.NEEDS_REVIEW,
            "STALE": ProductStatus.BLOCKED,
            "INVALID": ProductStatus.FAILED,
        }.get(value, ProductStatus.FAILED)


def new_run_id(command: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = command.replace(" ", "-").replace("/", "-")
    return f"{stamp}-{slug}-{secrets.token_hex(3)}"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
        if os.name != "nt":
            descriptor: int | None = None
            try:
                descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                os.fsync(descriptor)
            except OSError:
                pass
            finally:
                if descriptor is not None:
                    os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def lock_file(path: Path, *, timeout: float = 5.0) -> Iterator[None]:
    """Cross-platform lock based on atomic creation, suitable for small state files."""
    deadline = time.monotonic() + timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for state lock: {path}") from None
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class RunState:
    def __init__(self, directory: Path):
        self.directory = directory
        self.state_path = directory / "state.json"
        self.events_path = directory / "events.jsonl"
        self.command_path = directory / "command.json"
        self.log_path = directory / "logs" / "run.log"

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        *,
        command: str,
        argv: list[str],
        source_commit: str | None,
        dry_run: bool,
    ) -> RunState:
        run_id = new_run_id(command)
        instance = cls(config.runs_dir / run_id)
        for child in ("logs", "artifacts", "report", "checkpoints"):
            (instance.directory / child).mkdir(parents=True, exist_ok=False if child == "logs" else True)
        started = utc_now()
        command_value = {
            "schema_version": "spritelab.v3.command.v1",
            "run_id": run_id,
            "command": command,
            "argv": argv,
            "config_path": str(config.path) if config.path else None,
            "project_root": str(config.root),
            "source_commit": source_commit,
            "dry_run": dry_run,
            "started_at": started,
        }
        atomic_write_json(instance.command_path, command_value)
        instance.write_state(
            {
                "schema_version": RUN_SCHEMA,
                "run_id": run_id,
                "command": command,
                "status": "RUNNING",
                "stage": "project-validation",
                "started_at": started,
                "updated_at": started,
                "ended_at": None,
                "resumable": False,
                "dry_run": dry_run,
                "source_commit": source_commit,
                "config_path": str(config.path) if config.path else None,
                "backend_identity": {},
                "exit_code": None,
                "message": "Run created.",
            }
        )
        with instance.events_path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        record_event_history_origin(
            run_id,
            instance.directory,
            expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
        )
        instance.append_event(
            command=command,
            stage="project-validation",
            event_type="run_started",
            status="RUNNING",
            message="Run state created.",
        )
        instance.log(f"start command={command!r} argv={argv!r} project_root={config.root}")
        return instance

    @property
    def run_id(self) -> str:
        return self.directory.name

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state(self, value: dict[str, Any]) -> None:
        value = dict(value)
        value["updated_at"] = utc_now()
        with lock_file(self.directory / ".state.lock"):
            atomic_write_json(self.state_path, value)

    def update(self, **changes: Any) -> dict[str, Any]:
        value = self.read_state()
        value.update(changes)
        self.write_state(value)
        return value

    def append_event(
        self,
        *,
        command: str,
        stage: str,
        event_type: str,
        status: str,
        message: str,
        current_count: int = 0,
        total_count: int | None = None,
        artifact_identity: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_references: tuple[str, ...] = (),
    ) -> None:
        references = list(artifact_references)
        if artifact_identity:
            evidence = artifact_identity.get("evidence", ())
            if isinstance(evidence, list):
                references.extend(str(item) for item in evidence if item)
        event = ProductEvent(
            run_id=self.run_id,
            timestamp=utc_now(),
            feature=command,
            stage=stage,
            event_type=event_type,
            status=_product_event_status(status),
            current=current_count,
            total=total_count,
            message=message,
            metrics=metrics or {},
            artifact_references=tuple(references),
        )
        line = strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        request_event = event.to_dict()
        request_event.pop("timestamp", None)
        request_sha256 = event_append_request_sha256({"surface": "RunState", "event": request_event})
        try:
            with event_history_transaction_lock(self.directory):
                pending = pending_event_history_transaction(self.run_id, self.directory)
                if pending is not None:
                    if pending["operation"] != EVENT_HISTORY_TRANSACTION_APPEND:
                        raise LegacyEventMigrationError(
                            "The live event-history transaction cannot be recovered by a v3 event append."
                        )
                    append_event_transactionally(
                        self.run_id,
                        self.directory,
                        line=None,
                        request_sha256=None,
                        expected_origin=None,
                        append_surface=None,
                    )
                    if pending["request_sha256"] == request_sha256:
                        return
                state = self.read_state()
                verification = verify_event_migration(
                    self.run_id,
                    self.directory,
                    migration_required=bool(state.get("event_migration_required")),
                    origin_required=True,
                )
                if not verification.resume_compatible or (
                    verification.migration_required and not verification.migration_verified
                ):
                    raise LegacyEventMigrationError(f"{verification.state.value}: {verification.message}")
                append_event_transactionally(
                    self.run_id,
                    self.directory,
                    line=line,
                    request_sha256=request_sha256,
                    expected_origin=verification.event_history_origin,
                    append_surface=EVENT_APPEND_SURFACE_RUN_STATE,
                )
        except TimeoutError as exc:
            raise LegacyEventMigrationError(
                "Event history is busy; retry after the active durable write completes."
            ) from exc

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{utc_now()} {message}\n")

    def finish(
        self,
        *,
        command: str,
        status: str,
        exit_code: int,
        message: str,
        stage: str,
        resumable: bool = False,
        backend_identity: dict[str, Any] | None = None,
    ) -> None:
        self.update(
            status=status,
            exit_code=exit_code,
            message=message,
            stage=stage,
            ended_at=utc_now(),
            resumable=resumable,
            backend_identity=backend_identity or {},
        )
        self.append_event(
            command=command,
            stage=stage,
            event_type="run_finished",
            status=status,
            message=message,
        )
        self.log(f"finish status={status} exit_code={exit_code} stage={stage} message={message!r}")


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    runs = []
    for path in runs_dir.iterdir():
        state_path = path / "state.json"
        if not path.is_dir() or not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        state["directory"] = str(path)
        state["report_available"] = (path / "report" / "index.html").is_file()
        runs.append(state)
    return sorted(runs, key=lambda item: str(item.get("started_at", "")), reverse=True)


def resumable_runs(runs_dir: Path) -> list[dict[str, Any]]:
    terminal = {"COMPLETE", "FAILED", "BLOCKED", "CANCELLED"}
    return [run for run in list_runs(runs_dir) if run.get("resumable") and run.get("status") not in terminal]

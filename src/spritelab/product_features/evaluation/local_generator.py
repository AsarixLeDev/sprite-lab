"""Lazy, local-only Playground adapter for Sprite Lab challenger checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from spritelab.product_core import strict_json_dumps
from spritelab.product_features.evaluation.playground import (
    PLAYGROUND_RUNTIME_IDENTITY_SCHEMA,
    GeneratedAsset,
    GenerationCancelledError,
    GenerationTimedOutError,
    validate_runtime_identity,
)
from spritelab.utils.pinned_executable import (
    PinnedExecutableError,
    activate_windows_suspended_process,
    close_windows_handle,
    linux_parent_death_signal,
    pin_executable,
    pinned_git_ls_files,
    read_executable_identity,
    verify_process_image,
)
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    require_confined_path,
)
from spritelab.utils.write_confinement import (
    WriteConfinementError,
    create_windows_bootstrap_untrusted_process,
    prepare_windows_untrusted_integrity_workspace,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PROMPT_CHARACTERS = 2_000
_MAX_GENERATED_PNG_BYTES = 4 * 1024 * 1024
_MAX_BOUND_PLAYGROUND_WORKER_BYTES = 2 * 1024 * 1024
PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS = 5 * 60
_INTERRUPTED_CLEANUP_WAIT_SECONDS = 1.0
_PLAYGROUND_LEASE_SCHEMA = "spritelab.playground-sampler-lease.v2"
_PLAYGROUND_CONTROL_SCHEMA = "spritelab.playground-sampler-control.v2"
_PLAYGROUND_RESULT_SCHEMA = "spritelab.playground-sampler-result.v2"
_BOUND_PLAYGROUND_WORKER_BOOTSTRAP = """\
import hashlib,os,stat,sys
def _a(name):
    try:return sys.argv[sys.argv.index(name)+1]
    except (ValueError,IndexError):raise SystemExit(70)
try:
    size=int(_a('--worker-size'))
except ValueError:raise SystemExit(70)
if size<1 or size>2097152:raise SystemExit(70)
if _a('--bootstrap-sha256')!=os.environ.get('SPRITELAB_BOUND_BOOTSTRAP_SHA256'):raise SystemExit(70)
if '--worker-handle' in sys.argv:
    if os.name!='nt':raise SystemExit(70)
    import msvcrt
    try:fd=msvcrt.open_osfhandle(int(_a('--worker-handle')),os.O_RDONLY|getattr(os,'O_BINARY',0))
    except (OSError,ValueError):raise SystemExit(70)
    index=sys.argv.index('--worker-handle');sys.argv[index]='--worker-fd';sys.argv[index+1]=str(fd)
else:
    try:fd=int(_a('--worker-fd'))
    except ValueError:raise SystemExit(70)
try:
    metadata=os.fstat(fd)
    if fd<3 or not stat.S_ISREG(metadata.st_mode) or getattr(metadata,'st_nlink',1)!=1 or metadata.st_size!=size:raise SystemExit(70)
    os.lseek(fd,0,0);payload=bytearray()
    while len(payload)<size:
        chunk=os.read(fd,min(65536,size-len(payload)))
        if not chunk:raise SystemExit(70)
        payload.extend(chunk)
    if os.read(fd,1) or len(payload)!=size:raise SystemExit(70)
    os.lseek(fd,0,0)
except OSError:raise SystemExit(70)
source=bytes(payload)
if hashlib.sha256(source).hexdigest()!=_a('--worker-sha256'):raise SystemExit(70)
sys.path[:0]=[value for name in ('SPRITELAB_ISOLATED_PATHS','SPRITELAB_RUNTIME_ROOTS') for value in os.environ.get(name,'').split(os.pathsep) if value]
globals_={'__name__':'__main__','__package__':'spritelab.product_features.evaluation'}
exec(compile(source,'<bound-playground-worker>','exec',dont_inherit=True),globals_,globals_)
"""
_BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256 = hashlib.sha256(
    _BOUND_PLAYGROUND_WORKER_BOOTSTRAP.encode("utf-8")
).hexdigest()
_MISSING = object()
_PROCESS_INSTANCE_FALLBACK = (
    "portable-instance:"
    + hashlib.sha256(f"{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}".encode("ascii")).hexdigest()
)
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class LocalPlaygroundGenerationError(RuntimeError):
    """The local challenger sampler did not produce a safe bounded result."""


Sampler = Callable[[Any], Mapping[str, Any]]


class LocalCheckpointPlaygroundGenerator:
    """Run the existing challenger sampler without importing Torch on page load.

    Each explicit generation receives a new repository-local work directory.
    Those diagnostic files are retained; the durable Playground service copies
    only validated PNG bytes into its authoritative run artifacts.
    """

    remote = False
    billable = False
    requires_fresh_catalog = True

    def __init__(
        self,
        *,
        project_root: Path,
        work_root: Path,
        sampler: Sampler | None = None,
        windows_process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.work_root = require_confined_path(work_root, self.project_root)
        self._sampler = sampler
        self._windows_process_factory = windows_process_factory or create_windows_bootstrap_untrusted_process
        self.last_runtime_identity: dict[str, Any] | None = None
        self._control_local = threading.local()
        self._active_lock = threading.RLock()
        self._active: dict[str, dict[str, Any]] = {}
        self._lease_cursor_lock = threading.RLock()
        self._lease_cursors: dict[str, tuple[int, str, dict[str, Any]]] = {}

    def prepare_control(self, run_id: str, deadline_at: str) -> None:
        deadline = _parse_deadline(deadline_at)
        now = datetime.now(timezone.utc)
        if not run_id or deadline <= now:
            raise LocalPlaygroundGenerationError("The durable Playground deadline is invalid.")
        if getattr(self._control_local, "value", None) is not None:
            raise LocalPlaygroundGenerationError("A Playground operation control is already prepared on this thread.")
        cancel_event = threading.Event()
        monotonic_deadline = time.monotonic() + max(0.0, (deadline - now).total_seconds())
        with self._active_lock:
            if run_id in self._active:
                raise LocalPlaygroundGenerationError("That Playground run already has active sampling work.")
            self._active[run_id] = {"cancel": cancel_event, "process": None}
        self._control_local.value = (run_id, deadline, monotonic_deadline, cancel_event)

    def finish_control(self, run_id: str) -> None:
        value = getattr(self._control_local, "value", None)
        if isinstance(value, tuple) and len(value) == 4 and value[0] == run_id:
            del self._control_local.value
            cancel_event = value[3]
            with self._active_lock:
                active = self._active.get(run_id)
                if active is not None and active.get("cancel") is cancel_event and active.get("process") is None:
                    self._active.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        with self._active_lock:
            active = self._active.get(run_id)
            if active is None:
                return False
            active["cancel"].set()
            process = active.get("process")
        if process is not None:
            _terminate_contained_process(process)
        return True

    @property
    def code_identity_sha256(self) -> str:
        # Reuse the campaign's complete training execution inventory, then add
        # this adapter. This is evaluated only for an explicit generation plan.
        operation_check = self._prepared_operation_check()
        if operation_check is not None:
            operation_check()
        records = self._code_inventory(operation_check)
        payload = strict_json_dumps(
            {
                "schema_version": "spritelab.playground-local-code-identity.v1",
                "files": sorted(records, key=lambda row: row["path"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result = hashlib.sha256(payload).hexdigest()
        if operation_check is not None:
            operation_check()
        return result

    def _prepared_operation_check(self) -> Callable[[], None] | None:
        value = getattr(self._control_local, "value", None)
        if value is None:
            return None
        if not isinstance(value, tuple) or len(value) != 4:
            raise LocalPlaygroundGenerationError("The prepared Playground operation control is malformed.")
        run_id, deadline, monotonic_deadline, cancel_event = value
        if (
            not isinstance(run_id, str)
            or not isinstance(deadline, datetime)
            or deadline.tzinfo is None
            or not isinstance(monotonic_deadline, float)
            or not isinstance(cancel_event, threading.Event)
        ):
            raise LocalPlaygroundGenerationError("The prepared Playground operation control is malformed.")

        def operation_check() -> None:
            if cancel_event.is_set():
                raise GenerationCancelledError("Playground generation was cancelled before the next operation.")
            if datetime.now(timezone.utc) >= deadline or time.monotonic() >= monotonic_deadline:
                raise GenerationTimedOutError("Playground generation reached its durable wall-clock deadline.")

        return operation_check

    def _code_inventory(self, operation_check: Callable[[], None] | None = None) -> list[dict[str, str]]:
        source_tree_present = (self.project_root / "src/spritelab").is_dir()
        baseline = (
            _anchored_production_python_metadata(
                self.project_root,
                operation_check=operation_check,
            )
            if source_tree_present
            else None
        )
        source_paths = set(
            _operation_checked_training_code_identity_source_paths(
                self.project_root,
                operation_check=operation_check,
            )
        )
        source_paths.add(Path(__file__).resolve())
        source_paths.add(Path(__file__).with_name("playground_worker.py").resolve())
        source_paths.add(self.project_root / "src/spritelab/utils/runtime_closure.py")
        source_paths.add(self.project_root / "src/spritelab/utils/write_confinement.py")
        if baseline is None:
            baseline = _anchored_production_python_metadata(
                self.project_root,
                operation_check=operation_check,
            )
        if operation_check is not None:
            operation_check()
        ordered_source_paths = sorted(source_paths, key=lambda value: value.as_posix())
        if operation_check is not None:
            operation_check()
        records: list[dict[str, str]] = []
        for path in ordered_source_paths:
            if operation_check is not None:
                operation_check()
            expected_metadata = baseline.get(path)
            if expected_metadata is None:
                raise LocalPlaygroundGenerationError(
                    "A Playground execution source escaped the anchored production inventory."
                )
            records.append(
                {
                    "path": path.relative_to(self.project_root).as_posix(),
                    "sha256": _file_sha256(
                        path,
                        project_root=self.project_root,
                        expected_metadata=expected_metadata,
                        operation_check=operation_check,
                    ),
                }
            )
        if operation_check is not None:
            operation_check()
        if baseline != _anchored_production_python_metadata(
            self.project_root,
            operation_check=operation_check,
        ):
            raise LocalPlaygroundGenerationError("Production Python inventory changed while it was hashed.")
        return records

    def generate(
        self,
        *,
        checkpoint: Path,
        prompt: str,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
        weights: str,
        expected_sha256: str,
        expected_step: int,
        expected_variant: str,
    ) -> Sequence[GeneratedAsset]:
        if weights != expected_variant:
            raise LocalPlaygroundGenerationError("Checkpoint variant selection changed before sampling.")
        durable_control = getattr(self._control_local, "value", None)
        if isinstance(durable_control, tuple) and len(durable_control) == 4:
            run_id, deadline, monotonic_deadline, cancel_event = durable_control
            if not isinstance(run_id, str) or not run_id:
                raise LocalPlaygroundGenerationError("The durable Playground operation control is invalid.")
        elif durable_control is not None:
            raise LocalPlaygroundGenerationError("The durable Playground operation control is invalid.")
        else:
            run_id = f"direct-{uuid.uuid4().hex}"
            now = datetime.now(timezone.utc)
            deadline = now + timedelta(seconds=PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS)
            monotonic_deadline = time.monotonic() + PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS
            cancel_event = threading.Event()
        if not isinstance(deadline, datetime) or deadline.tzinfo is None:
            raise LocalPlaygroundGenerationError("The durable Playground deadline is invalid.")
        if not isinstance(monotonic_deadline, float) or not isinstance(cancel_event, threading.Event):
            raise LocalPlaygroundGenerationError("The durable Playground operation control is invalid.")

        def operation_check() -> None:
            if cancel_event.is_set():
                raise GenerationCancelledError("Playground generation was cancelled before the next operation.")
            if datetime.now(timezone.utc) >= deadline or time.monotonic() >= monotonic_deadline:
                raise GenerationTimedOutError("Playground generation reached its durable wall-clock deadline.")

        with self._active_lock:
            active = self._active.get(run_id)
            if active is None:
                self._active[run_id] = {"cancel": cancel_event, "process": None}
            elif active.get("cancel") is not cancel_event or active.get("process") is not None:
                raise LocalPlaygroundGenerationError("That Playground run already has active sampling work.")
        lease_id: str | None = None
        heartbeat_stop: threading.Event | None = None
        heartbeat: threading.Thread | None = None
        invocation_anchors: ExitStack | None = None
        lease_terminal = False
        active_terminal = False
        try:
            operation_check()
            self.last_runtime_identity = None
            normalized_prompt = self._validate_request(
                prompt=prompt,
                seed=seed,
                sampling_steps=sampling_steps,
                guidance=guidance,
                image_count=image_count,
                operation_check=operation_check,
            )
            operation_check()
            lease_id = self._acquire_lease(operation_check=operation_check)
            operation_check()
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                args=(lease_id, heartbeat_stop),
                name="spritelab-playground-lease-heartbeat",
                daemon=True,
            )
            operation_check()
            heartbeat.start()
            invocation_anchors = ExitStack()
            invocation, invocation_anchor = invocation_anchors.enter_context(
                self._new_anchored_invocation_directory(operation_check=operation_check)
            )
            self._update_lease(lease_id, operation_check=operation_check, invocation_id=invocation.name)
            snapshot = invocation / "checkpoint.snapshot.pt"
            snapshot_identity: OwnedFileIdentity | None = None
            try:
                with self._pinned_checkpoint(
                    checkpoint,
                    expected_sha256=expected_sha256,
                    operation_check=operation_check,
                ) as (checkpoint, checkpoint_descriptor):
                    snapshot_identity = self._snapshot_checkpoint(
                        checkpoint,
                        snapshot,
                        expected_sha256=expected_sha256,
                        operation_check=operation_check,
                        source_descriptor=checkpoint_descriptor,
                        destination_anchor=invocation_anchor,
                    )
            except BaseException:
                if snapshot_identity is not None:
                    self._retire_checkpoint_snapshot(
                        invocation_anchor,
                        snapshot.name,
                        snapshot_identity,
                    )
                raise
            operation_check()
            if self._sampler is not None:
                self._validate_snapshot_checkpoint(
                    snapshot,
                    expected_step=expected_step,
                    expected_variant=expected_variant,
                    operation_check=operation_check,
                )
            prompts_path = invocation / "prompts.jsonl"
            rows = [
                {
                    "prompt_id": f"playground_{index:04d}",
                    "prompt": normalized_prompt,
                    "scope": "EXPLORATORY",
                }
                for index in range(image_count)
            ]
            payload = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
            prompts_sha256 = hashlib.sha256(payload).hexdigest()
            operation_check()
            prompts_descriptor = invocation_anchor.open_file_immovable(
                prompts_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
            )
            prompts_identity = OwnedFileIdentity.from_stat(os.fstat(prompts_descriptor))
            with os.fdopen(prompts_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                if OwnedFileIdentity.from_stat(os.fstat(handle.fileno())) != prompts_identity:
                    raise LocalPlaygroundGenerationError("Local sampler prompt input changed while it was written.")
                if not prompts_identity.matches(invocation_anchor.lstat(prompts_path.name)):
                    raise LocalPlaygroundGenerationError("Local sampler prompt input changed before launch.")
            operation_check()

            output = invocation / "generated"
            operation_check()
            output_identity = invocation_anchor.mkdir(output.name)
            if not output_identity.matches(invocation_anchor.lstat(output.name)):
                raise LocalPlaygroundGenerationError("Local sampler output changed during creation.")
            output_anchor = invocation_anchors.enter_context(invocation_anchor.open_directory_immovable(output.name))
            if self._sampler is not None:
                config = self._sample_config(
                    checkpoint=snapshot,
                    prompts=prompts_path,
                    output=output,
                    seed=seed,
                    sampling_steps=sampling_steps,
                    guidance=guidance,
                    image_count=image_count,
                    expected_sha256=expected_sha256,
                    expected_step=expected_step,
                    expected_variant=expected_variant,
                    operation_check=operation_check,
                )
                report = dict(self._sampler(config))
                operation_check()
                runtime_identity = self._runtime_identity(report, operation_check=operation_check)
            else:
                report, runtime_identity = self._run_contained_sampler(
                    invocation=invocation,
                    checkpoint=snapshot,
                    prompts=prompts_path,
                    prompts_sha256=prompts_sha256,
                    output=output,
                    seed=seed,
                    sampling_steps=sampling_steps,
                    guidance=guidance,
                    image_count=image_count,
                    expected_sha256=expected_sha256,
                    expected_step=expected_step,
                    expected_variant=expected_variant,
                    run_id=run_id,
                    deadline=deadline.astimezone(timezone.utc),
                    cancel_event=cancel_event,
                    operation_check=operation_check,
                    invocation_anchor=invocation_anchor,
                )
            operation_check()
            if report.get("sample_count") != image_count:
                raise LocalPlaygroundGenerationError("Local sampler returned an unexpected sample count.")
            assets = self._load_assets(
                output,
                expected_prompt=normalized_prompt,
                expected_seed=seed,
                expected_steps=sampling_steps,
                expected_guidance=float(guidance),
                expected_count=image_count,
                operation_check=operation_check,
                output_anchor=output_anchor,
            )
            operation_check()
            invocation_anchors.close()
            invocation_anchors = None
            operation_check()
            heartbeat_stop.set()
            _join_thread_with_operation_check(heartbeat, timeout=6.0, operation_check=operation_check)
            if heartbeat.is_alive():
                raise LocalPlaygroundGenerationError("Local sampler lease heartbeat did not stop safely.")
            heartbeat_stop = None
            heartbeat = None
            operation_check()
            if not self._release_lease(
                lease_id,
                status="COMPLETE",
                retryable=False,
                operation_check=operation_check,
                defer_complete=True,
            ):
                raise LocalPlaygroundGenerationError("Local sampler lease could not be finalized safely.")
            with self._active_lock:
                active = self._active.get(run_id)
                if active is None or active.get("cancel") is not cancel_event:
                    raise LocalPlaygroundGenerationError("Local sampler operation control was lost before completion.")
                operation_check()
                if not self._release_lease(
                    lease_id,
                    status="COMPLETE",
                    retryable=False,
                    operation_check=operation_check,
                    check_after_transition=False,
                ):
                    raise LocalPlaygroundGenerationError("Local sampler lease could not be finalized safely.")
                self._active.pop(run_id, None)
                active_terminal = True
            lease_terminal = True
            self.last_runtime_identity = runtime_identity
            return assets
        except BaseException:
            if lease_id is not None and not lease_terminal:
                lease_terminal = self._release_lease(
                    lease_id,
                    status="FAILED",
                    retryable=True,
                )
            raise
        finally:
            if invocation_anchors is not None:
                invocation_anchors.close()
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat is not None:
                heartbeat.join(timeout=6.0)
            if not active_terminal:
                with self._active_lock:
                    active = self._active.get(run_id)
                    if active is not None and active.get("cancel") is cancel_event:
                        self._active.pop(run_id, None)

    @property
    def _lease_path(self) -> Path:
        return self.work_root / "sampler-lease.json"

    @property
    def _lease_lock_path(self) -> Path:
        return self.work_root / ".sampler-lease.lock"

    def _acquire_lease(self, *, operation_check: Callable[[], None]) -> str:
        operation_check()
        lease_id: str | None = None
        try:
            owner = _current_process_owner()
            operation_check()
            with self._lease_cursor_lock:
                with (
                    self._anchored_work_root(operation_check=operation_check) as lease_anchor,
                    _interprocess_lock(
                        lease_anchor,
                        self._lease_lock_path.name,
                        operation_check=operation_check,
                    ),
                ):
                    previous = _read_lease(
                        lease_anchor,
                        self._lease_path.name,
                        operation_check=operation_check,
                    )
                    recovered: dict[str, Any] | None = None
                    if previous and previous.get("status") == "ACTIVE":
                        if previous.get("schema_version") == _PLAYGROUND_LEASE_SCHEMA:
                            previous_owner = previous["owner"]
                            owner_live = _lease_owner_still_live(previous_owner)
                        else:
                            owner_pid = previous.get("owner_pid")
                            if type(owner_pid) is not int:
                                raise LocalPlaygroundGenerationError("Active local sampler lease owner is malformed.")
                            owner_live = _process_is_alive(owner_pid)
                        if owner_live:
                            raise LocalPlaygroundGenerationError("Another local Playground sampler is already active.")
                        recovered = {
                            "lease_id": str(previous.get("lease_id") or "unknown"),
                            "status": "ORPHANED",
                            "retryable": True,
                        }
                    lease_id = uuid.uuid4().hex
                    now = _utc_now()
                    previous_identity = (
                        previous.get("lease_identity")
                        if previous.get("schema_version") == _PLAYGROUND_LEASE_SCHEMA
                        else None
                    )
                    sequence = (
                        int(previous["transition_sequence"]) + 1
                        if previous.get("schema_version") == _PLAYGROUND_LEASE_SCHEMA
                        else 0
                    )
                    value: dict[str, Any] = {
                        "schema_version": _PLAYGROUND_LEASE_SCHEMA,
                        "lease_id": lease_id,
                        "lease_identity": "",
                        "transition_sequence": sequence,
                        "prior_lease_identity": previous_identity,
                        "status": "ACTIVE",
                        "owner": owner,
                        "acquired_at": now,
                        "heartbeat_at": now,
                        "ended_at": None,
                        "retryable": False,
                        "invocation_id": None,
                        "recovered_orphan": recovered,
                    }
                    value["lease_identity"] = _record_identity(value, "lease_identity")
                    _write_lease(
                        lease_anchor,
                        self._lease_path.name,
                        value,
                        operation_check=operation_check,
                    )
                    self._lease_cursors[lease_id] = (sequence, value["lease_identity"], dict(owner))
            operation_check()
            return lease_id
        except (GenerationCancelledError, GenerationTimedOutError):
            if lease_id is not None:
                self._release_lease(lease_id, status="FAILED", retryable=True)
            raise

    def _update_lease(
        self,
        lease_id: str,
        *,
        operation_check: Callable[[], None] | None = None,
        **updates: Any,
    ) -> None:
        if set(updates) - {"invocation_id"}:
            raise LocalPlaygroundGenerationError("Protected local sampler lease fields cannot be changed.")
        invocation_id = updates.get("invocation_id", _MISSING)
        self._transition_lease(
            lease_id,
            status="ACTIVE",
            retryable=False,
            invocation_id=invocation_id,
            operation_check=operation_check,
        )

    def _transition_lease(
        self,
        lease_id: str,
        *,
        status: str,
        retryable: bool,
        invocation_id: Any = None,
        operation_check: Callable[[], None] | None = None,
        check_after_transition: bool = True,
    ) -> None:
        if status not in {"ACTIVE", "COMPLETE", "FAILED"}:
            raise LocalPlaygroundGenerationError("Local sampler lease transition is invalid.")
        if retryable is not (status == "FAILED"):
            raise LocalPlaygroundGenerationError("Local sampler lease retry state is invalid.")
        with self._lease_cursor_lock:
            cursor = self._lease_cursors.get(lease_id)
            if cursor is None:
                raise LocalPlaygroundGenerationError("Local sampler lease ownership was lost.")
            expected_sequence, expected_identity, expected_owner = cursor
            with (
                self._anchored_work_root(operation_check=operation_check) as lease_anchor,
                _interprocess_lock(
                    lease_anchor,
                    self._lease_lock_path.name,
                    operation_check=operation_check,
                ),
            ):
                state = _read_lease(
                    lease_anchor,
                    self._lease_path.name,
                    operation_check=operation_check,
                )
                current_status = state.get("status")
                if (
                    state.get("schema_version") != _PLAYGROUND_LEASE_SCHEMA
                    or state.get("lease_id") != lease_id
                    or state.get("transition_sequence") != expected_sequence
                    or state.get("lease_identity") != expected_identity
                    or state.get("owner") != expected_owner
                    or not _lease_owner_is_current(expected_owner)
                    or current_status != "ACTIVE"
                ):
                    raise LocalPlaygroundGenerationError("Local sampler lease ownership was lost or stale.")
                next_state = dict(state)
                if status == "ACTIVE":
                    if invocation_id is not _MISSING:
                        if not _valid_invocation_id(invocation_id):
                            raise LocalPlaygroundGenerationError(
                                "Local sampler lease invocation identity is malformed."
                            )
                        current_invocation = state.get("invocation_id")
                        if current_invocation is not None and current_invocation != invocation_id:
                            raise LocalPlaygroundGenerationError(
                                "Local sampler lease invocation identity is immutable."
                            )
                        next_state["invocation_id"] = invocation_id
                elif invocation_id is not None:
                    raise LocalPlaygroundGenerationError("Terminal local sampler lease updates are malformed.")
                now = _utc_now()
                next_state.update(
                    {
                        "status": status,
                        "heartbeat_at": now,
                        "ended_at": None if status == "ACTIVE" else now,
                        "retryable": retryable,
                        "transition_sequence": expected_sequence + 1,
                        "prior_lease_identity": expected_identity,
                        "lease_identity": "",
                    }
                )
                next_state["lease_identity"] = _record_identity(next_state, "lease_identity")
                _write_lease(
                    lease_anchor,
                    self._lease_path.name,
                    next_state,
                    operation_check=operation_check,
                )
                self._lease_cursors[lease_id] = (
                    int(next_state["transition_sequence"]),
                    str(next_state["lease_identity"]),
                    dict(expected_owner),
                )
            if operation_check is not None and check_after_transition:
                operation_check()

    def _release_lease(
        self,
        lease_id: str,
        *,
        status: str,
        retryable: bool,
        operation_check: Callable[[], None] | None = None,
        defer_complete: bool = False,
        check_after_transition: bool = True,
    ) -> bool:
        try:
            if defer_complete:
                if status != "COMPLETE" or retryable is not False:
                    return False
                with self._lease_cursor_lock:
                    cursor = self._lease_cursors.get(lease_id)
                    if cursor is None:
                        return False
                    expected_sequence, expected_identity, expected_owner = cursor
                    with (
                        self._anchored_work_root(operation_check=operation_check) as lease_anchor,
                        _interprocess_lock(
                            lease_anchor,
                            self._lease_lock_path.name,
                            operation_check=operation_check,
                        ),
                    ):
                        state = _read_lease(
                            lease_anchor,
                            self._lease_path.name,
                            operation_check=operation_check,
                        )
                        if (
                            state.get("schema_version") != _PLAYGROUND_LEASE_SCHEMA
                            or state.get("lease_id") != lease_id
                            or state.get("transition_sequence") != expected_sequence
                            or state.get("lease_identity") != expected_identity
                            or state.get("owner") != expected_owner
                            or state.get("status") != "ACTIVE"
                            or not _lease_owner_is_current(expected_owner)
                        ):
                            return False
                if operation_check is not None:
                    operation_check()
                return True
            self._transition_lease(
                lease_id,
                status=status,
                retryable=retryable,
                operation_check=operation_check,
                check_after_transition=check_after_transition,
            )
        except (OSError, TimeoutError, LocalPlaygroundGenerationError):
            return False
        if operation_check is not None:
            operation_check()
        return True

    def _heartbeat_loop(self, lease_id: str, stop: threading.Event) -> None:
        while not stop.wait(5.0):
            try:
                self._update_lease(lease_id)
            except (OSError, TimeoutError, LocalPlaygroundGenerationError):
                return

    def _validate_checkpoint(
        self,
        checkpoint: Path,
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> Path:
        if operation_check is not None:
            operation_check()
        candidate = require_confined_path(checkpoint, self.project_root)
        current = self.project_root
        for part in candidate.relative_to(self.project_root).parts:
            if operation_check is not None:
                operation_check()
            current = current / part
            try:
                seam = current.lstat()
            except OSError as exc:
                raise LocalPlaygroundGenerationError("The selected local checkpoint is unavailable.") from exc
            if _is_link_or_reparse(seam) or os.path.ismount(current):
                raise LocalPlaygroundGenerationError("The selected local checkpoint crosses an unsafe seam.")
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise LocalPlaygroundGenerationError("The selected local checkpoint is unavailable.") from exc
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise LocalPlaygroundGenerationError("The selected local checkpoint is not a regular file.")
        if int(getattr(metadata, "st_nlink", 1)) != 1:
            raise LocalPlaygroundGenerationError("Hard-linked checkpoints are not eligible for Playground use.")
        if operation_check is not None:
            operation_check()
        return candidate

    @contextmanager
    def _pinned_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_sha256: str,
        operation_check: Callable[[], None] | None = None,
    ):
        """Pin and hash one project-root-relative checkpoint before any copy."""

        if operation_check is not None:
            operation_check()
        if not _is_sha256(expected_sha256):
            raise LocalPlaygroundGenerationError("Checkpoint SHA-256 expectation is malformed.")
        candidate = require_confined_path(checkpoint, self.project_root)
        relative = candidate.relative_to(self.project_root)
        if not relative.parts:
            raise LocalPlaygroundGenerationError("The selected local checkpoint is unavailable.")
        descriptor = -1
        with ExitStack() as anchors:
            try:
                parent = anchors.enter_context(AnchoredDirectory(self.project_root, self.project_root))
                for part in relative.parts[:-1]:
                    if operation_check is not None:
                        operation_check()
                    parent = anchors.enter_context(parent.open_directory_immovable(part))
                descriptor = parent.open_file_immovable(
                    relative.parts[-1],
                    os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
                )
                before = os.fstat(descriptor)
                identity = OwnedFileIdentity.from_stat(before)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or int(getattr(before, "st_nlink", 1)) != 1
                    or before.st_size < 0
                    or before.st_size > 8 * 1024**3
                    or not identity.matches(parent.lstat(relative.parts[-1]))
                ):
                    raise LocalPlaygroundGenerationError("The selected checkpoint is not one regular single-link file.")
                digest = hashlib.sha256()
                while True:
                    if operation_check is not None:
                        operation_check()
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after_hash = os.fstat(descriptor)
                if (
                    OwnedFileIdentity.from_stat(after_hash) != identity
                    or after_hash.st_size != before.st_size
                    or getattr(after_hash, "st_mtime_ns", None) != getattr(before, "st_mtime_ns", None)
                    or not identity.matches(parent.lstat(relative.parts[-1]))
                ):
                    raise LocalPlaygroundGenerationError("Checkpoint changed while its identity was verified.")
                if digest.hexdigest() != expected_sha256:
                    raise LocalPlaygroundGenerationError("Checkpoint does not match the durable catalog hash.")
                os.lseek(descriptor, 0, os.SEEK_SET)
                if operation_check is not None:
                    operation_check()
                yield candidate, descriptor
                final = os.fstat(descriptor)
                if (
                    OwnedFileIdentity.from_stat(final) != identity
                    or final.st_size != before.st_size
                    or getattr(final, "st_mtime_ns", None) != getattr(before, "st_mtime_ns", None)
                    or not identity.matches(parent.lstat(relative.parts[-1]))
                ):
                    raise LocalPlaygroundGenerationError("Checkpoint changed while its snapshot was consumed.")
                parent.verify()
            except LocalPlaygroundGenerationError:
                raise
            except (OSError, ValueError) as exc:
                raise LocalPlaygroundGenerationError("The selected local checkpoint is unavailable or unsafe.") from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    @staticmethod
    def _snapshot_checkpoint(
        source: Path,
        destination: Path,
        *,
        expected_sha256: str,
        operation_check: Callable[[], None] | None = None,
        source_descriptor: int | None = None,
        destination_anchor: AnchoredDirectory | None = None,
    ) -> OwnedFileIdentity:
        if destination_anchor is None:
            with AnchoredDirectory(destination.parent, destination.parent) as local_anchor:
                return LocalCheckpointPlaygroundGenerator._snapshot_checkpoint(
                    source,
                    destination,
                    expected_sha256=expected_sha256,
                    operation_check=operation_check,
                    source_descriptor=source_descriptor,
                    destination_anchor=local_anchor,
                )
        if os.path.normcase(os.path.abspath(destination.parent)) != os.path.normcase(
            os.path.abspath(destination_anchor.directory)
        ):
            raise LocalPlaygroundGenerationError("Checkpoint snapshot destination does not match its held parent.")
        if operation_check is not None:
            operation_check()
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise LocalPlaygroundGenerationError("Checkpoint SHA-256 expectation is malformed.")
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        source_fd = -1
        owns_source = source_descriptor is None
        transient_name: str | None = None
        transient_fd = -1
        transient_identity: OwnedFileIdentity | None = None
        current_name: str | None = None
        direct_final = False
        try:
            source_fd = os.open(source, flags) if source_descriptor is None else source_descriptor
            before = os.fstat(source_fd)
            source_identity = OwnedFileIdentity.from_stat(before)
            if (
                not stat.S_ISREG(before.st_mode)
                or int(getattr(before, "st_nlink", 1)) != 1
                or before.st_size < 0
                or before.st_size > 8 * 1024**3
            ):
                raise LocalPlaygroundGenerationError("The selected checkpoint is not one regular single-link file.")

            def source_unchanged(metadata: os.stat_result) -> bool:
                return (
                    OwnedFileIdentity.from_stat(metadata) == source_identity
                    and metadata.st_size == before.st_size
                    and getattr(metadata, "st_mtime_ns", None) == getattr(before, "st_mtime_ns", None)
                )

            preflight_digest = hashlib.sha256()
            os.lseek(source_fd, 0, os.SEEK_SET)
            while True:
                if operation_check is not None:
                    operation_check()
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                preflight_digest.update(chunk)
            after_preflight = os.fstat(source_fd)
            if not source_unchanged(after_preflight):
                raise LocalPlaygroundGenerationError("Checkpoint changed while its snapshot was verified.")
            if preflight_digest.hexdigest() != expected_sha256:
                raise LocalPlaygroundGenerationError("Checkpoint snapshot does not match the durable catalog hash.")
            if destination_anchor.lexists(destination.name):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot destination is already claimed.")
            os.lseek(source_fd, 0, os.SEEK_SET)
            if operation_check is not None:
                operation_check()
            if os.name == "nt":
                transient_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
                transient_fd = destination_anchor.open_file(
                    transient_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                    0o600,
                )
                current_name = transient_name
            else:
                try:
                    transient_fd = destination_anchor.open_anonymous_file(0o600)
                except (OSError, UnsafeFilesystemOperation):
                    # A fresh invocation directory is private until this
                    # method returns.  On POSIX filesystems without O_TMPFILE,
                    # create the canonical snapshot O_EXCL from birth.  It is
                    # never treated as usable until the held bytes and source
                    # binding have both been revalidated below.
                    transient_fd = destination_anchor.open_file(
                        destination.name,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                        0o600,
                    )
                    current_name = destination.name
                    direct_final = True
            transient_metadata = os.fstat(transient_fd)
            transient_identity = OwnedFileIdentity.from_stat(transient_metadata)
            expected_initial_links = 1 if transient_name is not None or direct_final else 0
            if (
                not stat.S_ISREG(transient_metadata.st_mode)
                or _is_link_or_reparse(transient_metadata)
                or int(getattr(transient_metadata, "st_nlink", 1)) != expected_initial_links
                or transient_metadata.st_size != 0
            ):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot transient object is unsafe.")
            if current_name is not None and not transient_identity.matches(destination_anchor.lstat(current_name)):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot transient name changed.")
            streamed_digest = hashlib.sha256()
            while True:
                if operation_check is not None:
                    operation_check()
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                streamed_digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(transient_fd, remaining)
                    if written <= 0:
                        raise OSError("checkpoint snapshot write made no progress")
                    remaining = remaining[written:]
                if operation_check is not None:
                    operation_check()
            os.fsync(transient_fd)
            after_stream = os.fstat(source_fd)
            if not source_unchanged(after_stream):
                raise LocalPlaygroundGenerationError("Checkpoint changed while its sampling snapshot was created.")
            if streamed_digest.hexdigest() != expected_sha256:
                raise LocalPlaygroundGenerationError("Checkpoint snapshot does not match the durable catalog hash.")
            written_metadata = os.fstat(transient_fd)
            if (
                OwnedFileIdentity.from_stat(written_metadata) != transient_identity
                or written_metadata.st_size != before.st_size
                or int(getattr(written_metadata, "st_nlink", 1)) != expected_initial_links
            ):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot changed while it was written.")
            if current_name is not None and not transient_identity.matches(destination_anchor.lstat(current_name)):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot name changed while it was written.")

            os.lseek(transient_fd, 0, os.SEEK_SET)
            retained_digest = hashlib.sha256()
            retained_size = 0
            while True:
                if operation_check is not None:
                    operation_check()
                chunk = os.read(transient_fd, 1024 * 1024)
                if not chunk:
                    break
                retained_digest.update(chunk)
                retained_size += len(chunk)
            retained_metadata = os.fstat(transient_fd)
            if (
                retained_digest.hexdigest() != expected_sha256
                or retained_size != before.st_size
                or OwnedFileIdentity.from_stat(retained_metadata) != transient_identity
                or retained_metadata.st_size != before.st_size
            ):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot bytes changed before publication.")
            if current_name is not None and not transient_identity.matches(destination_anchor.lstat(current_name)):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot name changed before publication.")
            if not source_unchanged(os.fstat(source_fd)):
                raise LocalPlaygroundGenerationError("Checkpoint changed before its snapshot was published.")
            if owns_source:
                current = os.stat(source, follow_symlinks=False)
                if not source_unchanged(current) or not source_identity.matches(current):
                    raise LocalPlaygroundGenerationError("Checkpoint changed before its snapshot was published.")
            if operation_check is not None:
                operation_check()
            if not direct_final:
                destination_anchor.publish_held_file_no_replace(
                    transient_fd,
                    transient_name,
                    destination.name,
                    identity=transient_identity,
                )
            current_name = destination.name
            held_published = os.fstat(transient_fd)
            published_metadata = destination_anchor.lstat(destination.name)
            if (
                not transient_identity.matches(published_metadata)
                or OwnedFileIdentity.from_stat(held_published) != transient_identity
                or int(getattr(held_published, "st_nlink", 1)) != 1
                or held_published.st_size != before.st_size
                or not stat.S_ISREG(held_published.st_mode)
                or _is_link_or_reparse(held_published)
                or not source_unchanged(os.fstat(source_fd))
            ):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot changed during exclusive publication.")
            destination_anchor.verify()
            return transient_identity
        except BaseException as exc:
            cleanup_error: BaseException | None = None
            cleanup_guard = -1
            can_scrub = transient_fd >= 0 and transient_identity is not None and current_name is None
            if transient_fd >= 0 and transient_identity is not None and current_name is not None:
                try:
                    if os.name == "nt":
                        cleanup_guard = destination_anchor.open_file_immovable(
                            current_name,
                            os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
                        )
                        can_scrub = OwnedFileIdentity.from_stat(
                            os.fstat(cleanup_guard)
                        ) == transient_identity and transient_identity.matches(destination_anchor.lstat(current_name))
                    else:
                        can_scrub = OwnedFileIdentity.from_stat(
                            os.fstat(transient_fd)
                        ) == transient_identity and transient_identity.matches(destination_anchor.lstat(current_name))
                except (FileNotFoundError, OSError, UnsafeFilesystemOperation):
                    can_scrub = False
            if can_scrub and transient_fd >= 0 and transient_identity is not None:
                try:
                    if OwnedFileIdentity.from_stat(os.fstat(transient_fd)) != transient_identity:
                        raise LocalPlaygroundGenerationError("Failed checkpoint snapshot changed before retirement.")
                    os.ftruncate(transient_fd, 0)
                    os.fsync(transient_fd)
                    if os.fstat(transient_fd).st_size != 0:
                        raise LocalPlaygroundGenerationError("Failed checkpoint snapshot could not be scrubbed.")
                except BaseException as error:
                    cleanup_error = error
            if cleanup_guard >= 0:
                os.close(cleanup_guard)
            if transient_fd >= 0:
                os.close(transient_fd)
                transient_fd = -1
            if transient_identity is not None and current_name is not None:
                try:
                    destination_anchor.unlink_if_owned(current_name, transient_identity)
                    if destination_anchor.lexists(current_name) and transient_identity.matches(
                        destination_anchor.lstat(current_name)
                    ):
                        raise LocalPlaygroundGenerationError("Failed checkpoint snapshot remained publicly named.")
                except BaseException as error:
                    cleanup_error = cleanup_error or error
            if cleanup_error is not None:
                raise LocalPlaygroundGenerationError("Failed checkpoint snapshot could not be retired safely.") from exc
            if isinstance(exc, LocalPlaygroundGenerationError):
                raise
            if isinstance(exc, (OSError, ValueError, UnsafeFilesystemOperation)):
                raise LocalPlaygroundGenerationError("Checkpoint snapshot could not be created safely.") from exc
            raise
        finally:
            if transient_fd >= 0:
                os.close(transient_fd)
            if source_fd >= 0 and owns_source:
                os.close(source_fd)

    @staticmethod
    def _retire_checkpoint_snapshot(
        anchor: AnchoredDirectory,
        name: str,
        identity: OwnedFileIdentity,
    ) -> None:
        descriptor = -1
        try:
            descriptor = anchor.open_file_immovable(
                name,
                os.O_RDWR | int(getattr(os, "O_BINARY", 0)),
            )
            metadata = os.fstat(descriptor)
            if (
                OwnedFileIdentity.from_stat(metadata) != identity
                or not identity.matches(anchor.lstat(name))
                or not stat.S_ISREG(metadata.st_mode)
                or int(getattr(metadata, "st_nlink", 1)) != 1
            ):
                raise LocalPlaygroundGenerationError("Failed checkpoint snapshot changed before retirement.")
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            if os.fstat(descriptor).st_size != 0:
                raise LocalPlaygroundGenerationError("Failed checkpoint snapshot could not be scrubbed.")
        except (OSError, ValueError) as exc:
            raise LocalPlaygroundGenerationError("Failed checkpoint snapshot could not be retired safely.") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            anchor.unlink_if_owned(name, identity)
            if anchor.lexists(name) and identity.matches(anchor.lstat(name)):
                raise LocalPlaygroundGenerationError("Failed checkpoint snapshot remained publicly named.")
        except (OSError, ValueError) as exc:
            raise LocalPlaygroundGenerationError("Failed checkpoint snapshot could not be retired safely.") from exc

    @staticmethod
    def _validate_snapshot_checkpoint(
        snapshot: Path,
        *,
        expected_step: int,
        expected_variant: str,
        operation_check: Callable[[], None] | None = None,
    ) -> None:
        if operation_check is not None:
            operation_check()
        from spritelab.training.checkpoint_io import load_checkpoint

        if type(expected_step) is not int or expected_step < 0:
            raise LocalPlaygroundGenerationError("Checkpoint step expectation is unavailable.")
        if expected_variant not in {"live", "ema"}:
            raise LocalPlaygroundGenerationError("Checkpoint variant expectation is malformed.")
        try:
            checkpoint = load_checkpoint(snapshot)
        except Exception as exc:
            raise LocalPlaygroundGenerationError("Checkpoint could not be loaded in safe weights-only mode.") from exc
        if operation_check is not None:
            operation_check()
        if checkpoint.get("model_type") != "generator_challenger":
            raise LocalPlaygroundGenerationError("Checkpoint model type is not supported by the local Playground.")
        if checkpoint.get("ema_weights") is not (expected_variant == "ema"):
            raise LocalPlaygroundGenerationError("Checkpoint EMA/live metadata does not match the selected variant.")
        step = checkpoint.get("step")
        global_step = checkpoint.get("global_step")
        if type(step) is not int or type(global_step) is not int or step != global_step or step != expected_step:
            raise LocalPlaygroundGenerationError("Checkpoint step metadata does not match durable catalog evidence.")
        if operation_check is not None:
            operation_check()

    @staticmethod
    def _runtime_identity(
        report: Mapping[str, Any],
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if operation_check is not None:
            operation_check()
        import torch

        report_config = report.get("config")
        selected_device = str(
            report.get("device")
            or (report_config.get("device_resolved") if isinstance(report_config, Mapping) else None)
            or "auto"
        )
        if len(selected_device) > 80:
            selected_device = "unknown"
        result = validate_runtime_identity(
            {
                "schema_version": PLAYGROUND_RUNTIME_IDENTITY_SCHEMA,
                "runtime_reported": True,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "selected_device": selected_device,
                "platform": sys.platform,
                "runtime_closure_identity": None,
                "execution_byte_policy": "in-process-generator-runtime-uncontained-v1",
                "bounded_residuals": ["generator-executed-in-server-process-without-write-confinement"],
                "paths_exposed": False,
            }
        )
        if operation_check is not None:
            operation_check()
        return result

    @staticmethod
    def _validate_request(
        *,
        prompt: str,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
        operation_check: Callable[[], None] | None = None,
    ) -> str:
        if operation_check is not None:
            operation_check()
        if not isinstance(prompt, str) or prompt != prompt.strip():
            raise LocalPlaygroundGenerationError("Prompt must already be canonicalized by the Playground request.")
        if not prompt or len(prompt) > _MAX_PROMPT_CHARACTERS:
            raise LocalPlaygroundGenerationError("Prompt length is outside the local Playground limit.")
        if any(ord(character) < 32 and character not in "\n\t" for character in prompt):
            raise LocalPlaygroundGenerationError("Prompt contains unsupported control characters.")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise LocalPlaygroundGenerationError("Seed must be an integer between 0 and 2**63 - 1.")
        if type(sampling_steps) is not int or not 1 <= sampling_steps <= 500:
            raise LocalPlaygroundGenerationError("Sampling steps are outside the supported range.")
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 0 < float(guidance) <= 50:
            raise LocalPlaygroundGenerationError("Guidance is outside the supported range.")
        if not math.isfinite(float(guidance)):
            raise LocalPlaygroundGenerationError("Guidance is outside the supported range.")
        if type(image_count) is not int or not 1 <= image_count <= 16:
            raise LocalPlaygroundGenerationError("Image count is outside the supported range.")
        if operation_check is not None:
            operation_check()
        return prompt

    def _new_invocation_directory(
        self,
        *,
        operation_check: Callable[[], None] | None = None,
    ) -> Path:
        with self._new_anchored_invocation_directory(operation_check=operation_check) as (invocation, _anchor):
            return invocation

    @contextmanager
    def _new_anchored_invocation_directory(
        self,
        *,
        operation_check: Callable[[], None] | None = None,
    ):
        if operation_check is not None:
            operation_check()
        try:
            with self._anchored_work_root(operation_check=operation_check) as work_anchor:
                name, identity = work_anchor.mkdir_unique("playground-sampler-")
                metadata = work_anchor.lstat(name)
                if not identity.matches(metadata):
                    raise LocalPlaygroundGenerationError("Could not bind the local sampling directory identity.")
                invocation = require_confined_path(self.work_root / name, self.work_root)
                with work_anchor.open_directory_immovable(name) as invocation_anchor:
                    opened = invocation_anchor.directory_metadata()
                    if not identity.matches(opened):
                        raise LocalPlaygroundGenerationError("Could not bind the local sampling directory identity.")
                    if os.name == "nt" and sys.platform == "win32":
                        try:
                            prepared = prepare_windows_untrusted_integrity_workspace(invocation)
                        except (OSError, ValueError, WriteConfinementError) as exc:
                            raise LocalPlaygroundGenerationError(
                                "Could not prepare the Windows Playground sampling boundary."
                            ) from exc
                        labeled = invocation_anchor.directory_metadata()
                        if (
                            not identity.matches(labeled)
                            or prepared.entry_count != 1
                            or prepared.identity.device != int(labeled.st_dev)
                            or prepared.identity.inode != int(labeled.st_ino)
                            or invocation_anchor.names()
                        ):
                            raise LocalPlaygroundGenerationError("The Windows Playground sampling boundary changed.")
                    invocation_anchor.verify()
                    if operation_check is not None:
                        operation_check()
                    yield invocation, invocation_anchor
                    invocation_anchor.verify()
                if operation_check is not None:
                    operation_check()
        except LocalPlaygroundGenerationError:
            raise
        except (OSError, ValueError) as exc:
            raise LocalPlaygroundGenerationError("Could not create a safe local sampling directory.") from exc

    def _ensure_work_root(self, *, operation_check: Callable[[], None] | None = None) -> None:
        with self._anchored_work_root(operation_check=operation_check):
            pass
        if operation_check is not None:
            operation_check()

    @contextmanager
    def _anchored_work_root(
        self,
        *,
        operation_check: Callable[[], None] | None = None,
    ):
        relative = self.work_root.relative_to(self.project_root)
        if not relative.parts:
            raise LocalPlaygroundGenerationError("Local Playground work root must be below the project root.")
        try:
            with ExitStack() as anchors:
                current = anchors.enter_context(AnchoredDirectory(self.project_root, self.project_root))
                for part in relative.parts:
                    if operation_check is not None:
                        operation_check()
                    if not current.lexists(part):
                        created = current.mkdir(part)
                    else:
                        created = None
                    child = anchors.enter_context(current.open_directory_immovable(part))
                    if created is not None and not created.matches(child.directory_metadata()):
                        raise LocalPlaygroundGenerationError("Local Playground work root changed during creation.")
                    current = child
                current.verify()
                if operation_check is not None:
                    operation_check()
                yield current
                current.verify()
        except LocalPlaygroundGenerationError:
            raise
        except (OSError, ValueError) as exc:
            raise LocalPlaygroundGenerationError("Local Playground work root crosses an unsafe seam.") from exc

    @staticmethod
    def _sample_config(
        *,
        checkpoint: Path,
        prompts: Path,
        output: Path,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
        expected_sha256: str,
        expected_step: int,
        expected_variant: str,
        operation_check: Callable[[], None] | None = None,
    ) -> Any:
        if operation_check is not None:
            operation_check()
        # Importing the challenger module imports Torch, so keep this behind the
        # explicit Generate action rather than application/router construction.
        from spritelab.training.generator_challenger import ChallengerSampleConfig

        result = ChallengerSampleConfig(
            checkpoint=checkpoint,
            prompts=prompts,
            out_dir=output,
            expected_checkpoint_sha256=expected_sha256,
            expected_checkpoint_step=expected_step,
            expected_checkpoint_variant=expected_variant,
            max_samples=image_count,
            steps=sampling_steps,
            cfg_scale=float(guidance),
            device="auto",
            seed=seed,
            noise_seed=seed,
            batch_size=min(image_count, 16),
            write_raw_rgba=False,
            write_hard_rgba=True,
            contact_sheet_labels="prompt",
        )
        if operation_check is not None:
            operation_check()
        return result

    def _run_contained_sampler(
        self,
        *,
        invocation: Path,
        checkpoint: Path,
        prompts: Path,
        prompts_sha256: str,
        output: Path,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
        expected_sha256: str,
        expected_step: int,
        expected_variant: str,
        run_id: str,
        deadline: datetime,
        cancel_event: threading.Event,
        operation_check: Callable[[], None],
        invocation_anchor: AnchoredDirectory,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (sys.platform.startswith("linux") or (sys.platform == "win32" and os.name == "nt")):
            raise LocalPlaygroundGenerationError(
                "Contained local Playground sampling is unavailable until this platform has an exact writable-root launcher."
            )
        from spritelab.utils.runtime_closure import (
            exact_python_runtime_environment_paths,
            prepare_exact_python_runtime_closure,
        )

        report_path = invocation / "sampler-result.json"
        control_path = invocation / "sampler-control.json"
        stdio_root = invocation / "tmp"

        def relative(path: Path) -> str:
            return path.relative_to(invocation).as_posix()

        invocation_metadata = invocation_anchor.directory_metadata()
        operation_check()
        runtime_closure = prepare_exact_python_runtime_closure(
            self.project_root,
            operation_check=operation_check,
        )
        operation_check()
        import_paths, runtime_paths = _wait_for_operation_call(
            lambda: exact_python_runtime_environment_paths(
                self.project_root,
                operation_check=operation_check,
            ),
            operation_check=operation_check,
        )
        operation_check()
        code_inventory = self._code_inventory(operation_check)
        operation_check()
        worker_relative = "src/spritelab/product_features/evaluation/playground_worker.py"
        worker_sha256 = next(
            (row["sha256"] for row in code_inventory if row["path"] == worker_relative),
            None,
        )
        operation_check()
        if worker_sha256 is None:
            raise LocalPlaygroundGenerationError("The contained Playground worker identity is unavailable.")
        worker_path = self.project_root / worker_relative
        worker_payload = _read_safe_regular_bytes(
            worker_path,
            maximum_bytes=_MAX_BOUND_PLAYGROUND_WORKER_BYTES,
            label="contained Playground worker",
            operation_check=operation_check,
        )
        operation_check()
        if not worker_payload or hashlib.sha256(worker_payload).hexdigest() != worker_sha256:
            raise LocalPlaygroundGenerationError("The contained Playground worker identity changed.")
        runtime_closure_identity = runtime_closure.get("runtime_closure_identity")
        if not _is_sha256(runtime_closure_identity):
            raise LocalPlaygroundGenerationError("The contained Playground runtime closure identity is malformed.")
        control: dict[str, Any] = {
            "schema_version": _PLAYGROUND_CONTROL_SCHEMA,
            "control_identity": "",
            "bootstrap_identity": _BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256,
            "worker_sha256": worker_sha256,
            "worker_size": len(worker_payload),
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": expected_sha256,
            "prompts": relative(prompts),
            "prompts_sha256": prompts_sha256,
            "output": relative(output),
            "report": relative(report_path),
            "seed": seed,
            "sampling_steps": sampling_steps,
            "guidance": float(guidance),
            "image_count": image_count,
            "expected_step": expected_step,
            "expected_variant": expected_variant,
            "deadline_at": deadline.isoformat(),
            "code_inventory": code_inventory,
            "code_inventory_identity": _canonical_sha256(code_inventory),
            "runtime_closure": runtime_closure,
            "runtime_closure_identity": runtime_closure_identity,
            "workspace_identity": {
                "device": int(invocation_metadata.st_dev),
                "inode": int(invocation_metadata.st_ino),
            },
        }
        control["control_identity"] = _record_identity(control, "control_identity")
        control_bytes = (strict_json_dumps(control, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        operation_check()
        control_descriptor = invocation_anchor.open_file_immovable(
            control_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        )
        control_identity = OwnedFileIdentity.from_stat(os.fstat(control_descriptor))
        with os.fdopen(control_descriptor, "wb") as handle:
            handle.write(control_bytes)
            handle.flush()
            os.fsync(handle.fileno())
            if OwnedFileIdentity.from_stat(os.fstat(handle.fileno())) != control_identity:
                raise LocalPlaygroundGenerationError("Contained Playground control changed while it was written.")
            if not control_identity.matches(invocation_anchor.lstat(control_path.name)):
                raise LocalPlaygroundGenerationError("Contained Playground control changed before launch.")
        operation_check()
        stdio_identity = invocation_anchor.mkdir(stdio_root.name)
        if not stdio_identity.matches(invocation_anchor.lstat(stdio_root.name)):
            raise LocalPlaygroundGenerationError("The contained Playground stdio directory changed.")
        operation_check()
        control_sha256 = hashlib.sha256(control_bytes).hexdigest()
        operation_check()
        executable_identity = _wait_for_operation_call(
            lambda: read_executable_identity(Path(sys.executable).resolve()),
            operation_check=operation_check,
        )
        operation_check()
        worker_job_handle: int | None = None
        process: subprocess.Popen[bytes] | None = None
        process_cleanup_delegated = threading.Event()
        monotonic_deadline = time.monotonic() + max(
            0.0,
            (deadline - datetime.now(timezone.utc)).total_seconds(),
        )
        try:
            workspace_descriptor = (
                _pinned_directory_descriptor(invocation, operation_check=operation_check)
                if sys.platform.startswith("linux")
                else nullcontext(None)
            )
            operation_check()
            with (
                workspace_descriptor as workspace_fd,
                _pinned_regular_file(
                    checkpoint,
                    expected_sha256=expected_sha256,
                    operation_check=operation_check,
                ) as checkpoint_fd,
                _pinned_regular_file(
                    prompts,
                    expected_sha256=prompts_sha256,
                    operation_check=operation_check,
                ) as prompts_fd,
                _pinned_regular_file(
                    worker_path,
                    expected_sha256=worker_sha256,
                    operation_check=operation_check,
                ) as worker_fd,
                _operation_checked_pin_executable(
                    executable_identity.resolved_path,
                    expected_sha256=executable_identity.executable_sha256,
                    expected_size=executable_identity.byte_count,
                    expected_metadata_sha256=executable_identity.metadata_sha256,
                    operation_check=operation_check,
                ) as interpreter,
            ):
                operation_check()
                argv_prefix = [
                    interpreter.launch_path,
                    "-I",
                    "-B",
                    "-S",
                    "-c",
                    _BOUND_PLAYGROUND_WORKER_BOOTSTRAP,
                    "--control",
                    relative(control_path),
                    "--control-sha256",
                    control_sha256,
                    "--bootstrap-sha256",
                    _BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256,
                    "--worker-sha256",
                    worker_sha256,
                    "--worker-size",
                    str(len(worker_payload)),
                ]
                child_environment = _minimal_sampler_environment(
                    invocation,
                    project_root=self.project_root,
                    import_paths=import_paths,
                    runtime_paths=runtime_paths,
                )
                options: dict[str, Any] = {
                    "cwd": invocation,
                    "env": child_environment,
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                    "shell": False,
                }
                if sys.platform.startswith("linux"):
                    if not isinstance(workspace_fd, int):
                        raise LocalPlaygroundGenerationError("Held Playground workspace identity is unavailable.")
                    argv_prefix.extend(
                        (
                            "--checkpoint-fd",
                            str(checkpoint_fd),
                            "--workspace-fd",
                            str(workspace_fd),
                            "--prompts-fd",
                            str(prompts_fd),
                        )
                    )
                    options["start_new_session"] = True
                    options["preexec_fn"] = linux_parent_death_signal(os.getpid())
                elif not (sys.platform == "win32" and os.name == "nt"):
                    raise LocalPlaygroundGenerationError(
                        "Contained local Playground sampling is unavailable on this platform."
                    )
                operation_check()
                with _inherited_worker_transport(worker_fd) as (worker_arguments, transport_options):
                    argv = [*argv_prefix, *worker_arguments]
                    if sys.platform.startswith("linux"):
                        transport_pass_fds = tuple(transport_options.get("pass_fds", ()))
                        options["pass_fds"] = tuple(
                            sorted(
                                {
                                    *interpreter.pass_fds,
                                    checkpoint_fd,
                                    workspace_fd,
                                    prompts_fd,
                                    *transport_pass_fds,
                                }
                            )
                        )
                        process = subprocess.Popen(argv, **options)
                    else:
                        inherited_handles = tuple(transport_options.get("inherited_handles", ()))
                        process = self._windows_process_factory(
                            argv,
                            cwd=invocation,
                            env=child_environment,
                            stdin_payload=b"",
                            writable_roots=(invocation,),
                            stdio_root=stdio_root,
                            inherited_handles=inherited_handles,
                        )
                with self._active_lock:
                    active = self._active.get(run_id)
                    if active is None or active["cancel"] is not cancel_event:
                        _terminate_contained_process(process)
                        raise GenerationCancelledError("Playground generation was cancelled before process activation.")
                    operation_check()
                    active["process"] = process

                def activation() -> int | None:
                    if os.name == "nt":
                        return activate_windows_suspended_process(
                            process,
                            verifier=lambda launched: verify_process_image(launched, interpreter),
                        )
                    verify_process_image(process, interpreter)
                    return None

                worker_job_handle = _wait_for_process_activation(
                    process,
                    activate=activation,
                    operation_check=operation_check,
                    cleanup_delegated=process_cleanup_delegated,
                )
                while process.poll() is None:
                    if cancel_event.wait(0.05):
                        cleanup_done = _start_contained_process_cleanup_reaper(
                            process,
                            job_handle=worker_job_handle,
                        )
                        process_cleanup_delegated.set()
                        worker_job_handle = None
                        cleanup_done.wait(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)
                        raise GenerationCancelledError("Playground generation was cancelled and terminated.")
                    if datetime.now(timezone.utc) >= deadline or time.monotonic() >= monotonic_deadline:
                        cleanup_done = _start_contained_process_cleanup_reaper(
                            process,
                            job_handle=worker_job_handle,
                        )
                        process_cleanup_delegated.set()
                        worker_job_handle = None
                        cleanup_done.wait(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)
                        raise GenerationTimedOutError(
                            "Contained Playground generation reached its fixed wall-clock deadline."
                        )
                if int(process.wait()) != 0:
                    raise LocalPlaygroundGenerationError("Contained Playground sampling failed safely.")
                operation_check()
        finally:
            with self._active_lock:
                active = self._active.get(run_id)
                if active is not None:
                    active["process"] = None
            if process is not None and process.poll() is None and not process_cleanup_delegated.is_set():
                _terminate_contained_process(process)
            if worker_job_handle and not process_cleanup_delegated.is_set():
                close_windows_handle(worker_job_handle)
        operation_check()
        result = self._read_contained_sampler_result(
            invocation_anchor,
            report_path.name,
            expected_control=control,
            operation_check=operation_check,
        )
        operation_check()
        return result

    @staticmethod
    def _read_contained_sampler_result(
        invocation_anchor: AnchoredDirectory,
        report_name: str,
        *,
        operation_check: Callable[[], None],
        expected_control: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        operation_check()
        payload = _read_anchored_regular_bytes(
            invocation_anchor,
            report_name,
            maximum_bytes=16 * 1024 * 1024,
            label="contained sampler result",
            operation_check=operation_check,
        )
        operation_check()
        try:
            result = _strict_json_loads_no_duplicates(payload)
        except (UnicodeError, ValueError) as exc:
            raise LocalPlaygroundGenerationError("Contained sampler result is malformed.") from exc
        if not isinstance(result, dict) or payload != (
            strict_json_dumps(result, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"):
            raise LocalPlaygroundGenerationError("Contained sampler result is malformed.")
        if expected_control is None or not _valid_sampler_result(result, expected_control):
            raise LocalPlaygroundGenerationError("Contained sampler result is malformed.")
        validated_runtime = validate_runtime_identity(result["runtime_identity"])
        operation_check()
        value = dict(result["report"]), validated_runtime
        operation_check()
        return value

    def _load_assets(
        self,
        output: Path,
        *,
        expected_prompt: str,
        expected_seed: int,
        expected_steps: int,
        expected_guidance: float,
        expected_count: int,
        operation_check: Callable[[], None] | None = None,
        output_anchor: AnchoredDirectory | None = None,
    ) -> tuple[GeneratedAsset, ...]:
        if operation_check is not None:
            operation_check()
        try:
            output_metadata = output_anchor.directory_metadata() if output_anchor is not None else output.lstat()
        except OSError as exc:
            raise LocalPlaygroundGenerationError("Local sampler output directory is unavailable.") from exc
        if (
            not stat.S_ISDIR(output_metadata.st_mode)
            or _is_link_or_reparse(output_metadata)
            or (output_anchor is None and os.path.ismount(output))
        ):
            raise LocalPlaygroundGenerationError("Local sampler output directory crosses an unsafe seam.")
        manifest_path = output / "generated_manifest.jsonl"
        manifest_bytes = (
            _read_anchored_regular_bytes(
                output_anchor,
                manifest_path.name,
                maximum_bytes=2 * 1024 * 1024,
                label="generation manifest",
                operation_check=operation_check,
            )
            if output_anchor is not None
            else _read_safe_regular_bytes(
                manifest_path,
                maximum_bytes=2 * 1024 * 1024,
                label="generation manifest",
                operation_check=operation_check,
            )
        )
        if operation_check is not None:
            operation_check()
        records: list[dict[str, Any]] = []
        try:
            manifest_text = manifest_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise LocalPlaygroundGenerationError("Local generation manifest is not valid UTF-8.") from exc
        for line in manifest_text.splitlines():
            if operation_check is not None:
                operation_check()
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocalPlaygroundGenerationError("Local generation manifest is malformed.") from exc
            if not isinstance(value, dict):
                raise LocalPlaygroundGenerationError("Local generation manifest contains a non-object row.")
            if set(value) != {
                "cfg_scale",
                "model_type",
                "noise_seed",
                "paths",
                "prompt",
                "prompt_id",
                "sample_id",
                "scope",
                "seed",
                "steps",
            }:
                raise LocalPlaygroundGenerationError("Local generation manifest contains retained private metadata.")
            if not isinstance(value.get("paths"), dict) or set(value["paths"]) != {"indexed_png"}:
                raise LocalPlaygroundGenerationError("Local generation manifest path metadata is malformed.")
            records.append(value)
        if operation_check is not None:
            operation_check()
        if len(records) != expected_count:
            raise LocalPlaygroundGenerationError("Local generation manifest has an unexpected row count.")

        assets: list[GeneratedAsset] = []
        collision_keys: set[str] = set()
        prompt_id_keys: set[str] = set()
        for index, record in enumerate(records):
            if operation_check is not None:
                operation_check()
            expected_prompt_id = f"playground_{index:04d}"
            expected_sample_id = f"sample_{index:06d}"
            prompt_id = record.get("prompt_id")
            if not isinstance(prompt_id, str):
                raise LocalPlaygroundGenerationError("Local generation manifest prompt identity is missing.")
            prompt_id_key = unicodedata.normalize("NFKC", prompt_id).casefold()
            if prompt_id_key in prompt_id_keys or prompt_id != expected_prompt_id:
                raise LocalPlaygroundGenerationError("Local generation manifest prompt identities are inconsistent.")
            prompt_id_keys.add(prompt_id_key)
            if (
                record.get("sample_id") != expected_sample_id
                or record.get("prompt") != expected_prompt
                or record.get("scope") != "EXPLORATORY"
                or type(record.get("seed")) is not int
                or record.get("seed") != expected_seed
                or type(record.get("noise_seed")) is not int
                or record.get("noise_seed") != expected_seed + index
                or record.get("model_type") != "generator_challenger"
                or type(record.get("steps")) is not int
                or record.get("steps") != expected_steps
                or isinstance(record.get("cfg_scale"), bool)
                or not isinstance(record.get("cfg_scale"), (int, float))
                or not math.isfinite(float(record["cfg_scale"]))
                or float(record["cfg_scale"]) != expected_guidance
            ):
                raise LocalPlaygroundGenerationError("Local generation manifest semantics do not match the request.")
            paths = record.get("paths")
            raw_relative = paths.get("indexed_png") if isinstance(paths, Mapping) else None
            relative = _safe_relative_png(raw_relative)
            collision_key = unicodedata.normalize("NFKC", relative.as_posix()).casefold()
            if collision_key in collision_keys:
                raise LocalPlaygroundGenerationError("Local generation manifest contains colliding output paths.")
            collision_keys.add(collision_key)
            path = require_confined_path(output / Path(*relative.parts), output)
            if output_anchor is None:
                current = output
                for part in relative.parts[:-1]:
                    current = current / part
                    try:
                        metadata = current.lstat()
                    except OSError as exc:
                        raise LocalPlaygroundGenerationError("Local sampler output is missing or unsafe.") from exc
                    if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse(metadata) or os.path.ismount(current):
                        raise LocalPlaygroundGenerationError("Local sampler output crosses an unsafe directory seam.")
            if output_anchor is None:
                content = _read_safe_regular_bytes(
                    path,
                    maximum_bytes=_MAX_GENERATED_PNG_BYTES,
                    label="sampler PNG",
                    operation_check=operation_check,
                )
            else:
                with ExitStack() as stack:
                    asset_anchor = output_anchor
                    for part in relative.parts[:-1]:
                        asset_anchor = stack.enter_context(asset_anchor.open_directory_immovable(part))
                        if operation_check is not None:
                            operation_check()
                    content = _read_anchored_regular_bytes(
                        asset_anchor,
                        relative.parts[-1],
                        maximum_bytes=_MAX_GENERATED_PNG_BYTES,
                        label="sampler PNG",
                        operation_check=operation_check,
                    )
            if not content:
                raise LocalPlaygroundGenerationError("Local sampler PNG is empty.")
            if not content.startswith(_PNG_SIGNATURE):
                raise LocalPlaygroundGenerationError("Local sampler output is not a PNG image.")
            try:
                with Image.open(io.BytesIO(content)) as image:
                    if image.format != "PNG" or image.size != (32, 32) or getattr(image, "n_frames", 1) != 1:
                        raise LocalPlaygroundGenerationError("Local sampler output is not one 32x32 PNG frame.")
                    image.verify()
            except LocalPlaygroundGenerationError:
                raise
            except Exception as exc:
                raise LocalPlaygroundGenerationError("Local sampler PNG could not be decoded safely.") from exc
            if operation_check is not None:
                operation_check()
            assets.append(GeneratedAsset(content=content, media_type="image/png"))
        if operation_check is not None:
            operation_check()
        return tuple(assets)


def _run_challenger_sampler(config: Any) -> Mapping[str, Any]:
    from spritelab.training.generator_challenger import run_sample_generator_challenger

    return run_sample_generator_challenger(config)


@contextmanager
def _pinned_directory_descriptor(
    path: Path,
    *,
    operation_check: Callable[[], None] | None = None,
):
    if operation_check is not None:
        operation_check()
    flags = (
        int(getattr(os, "O_PATH", os.O_RDONLY)) | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        lexical = path.stat(follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[int, int, int]:
            return (
                int(value.st_dev),
                int(value.st_ino),
                int(stat.S_IFMT(value.st_mode)),
            )

        if not stat.S_ISDIR(before.st_mode) or identity(before) != identity(lexical):
            raise LocalPlaygroundGenerationError("Playground workspace descriptor identity changed.")
        if operation_check is not None:
            operation_check()
        yield descriptor
        if operation_check is not None:
            operation_check()
        if identity(before) != identity(os.fstat(descriptor)) or identity(before) != identity(
            path.stat(follow_symlinks=False)
        ):
            raise LocalPlaygroundGenerationError("Playground workspace changed during contained generation.")
        if operation_check is not None:
            operation_check()
    finally:
        os.close(descriptor)


@contextmanager
def _pinned_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    operation_check: Callable[[], None] | None = None,
):
    if operation_check is not None:
        operation_check()
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        handle = kernel32.CreateFileW(
            str(path),
            0x80000000,
            0x00000001,
            None,
            3,
            0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if not handle or int(handle) == invalid:
            raise LocalPlaygroundGenerationError("Checkpoint sampling snapshot could not be held safely.")
        try:
            descriptor = int(msvcrt.open_osfhandle(int(handle), os.O_RDONLY | int(getattr(os, "O_BINARY", 0))))
        except BaseException:
            kernel32.CloseHandle(handle)
            raise
    elif sys.platform.startswith("linux"):
        descriptor = os.open(
            path,
            os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0)),
        )
    else:
        raise LocalPlaygroundGenerationError("Held checkpoint sampling is unavailable on this platform.")
    try:
        before = os.fstat(descriptor)
        lexical = path.stat(follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[Any, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                int(getattr(value, "st_nlink", 1)),
                getattr(value, "st_mtime_ns", None),
            )

        if (
            not stat.S_ISREG(before.st_mode)
            or _is_link_or_reparse(lexical)
            or int(getattr(before, "st_nlink", 1)) != 1
            or identity(before) != identity(lexical)
        ):
            raise LocalPlaygroundGenerationError("Checkpoint sampling snapshot is unsafe.")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            if operation_check is not None:
                operation_check()
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if operation_check is not None:
                operation_check()
        os.lseek(descriptor, 0, os.SEEK_SET)
        if digest.hexdigest() != expected_sha256 or identity(before) != identity(path.stat(follow_symlinks=False)):
            raise LocalPlaygroundGenerationError("Checkpoint sampling snapshot identity changed.")
        if operation_check is not None:
            operation_check()
        yield descriptor
        if operation_check is not None:
            operation_check()
        if identity(before) != identity(os.fstat(descriptor)) or identity(before) != identity(
            path.stat(follow_symlinks=False)
        ):
            raise LocalPlaygroundGenerationError("Checkpoint sampling snapshot changed during generation.")
        if operation_check is not None:
            operation_check()
    finally:
        os.close(descriptor)


@contextmanager
def _inherited_worker_transport(descriptor: int):
    """Expose exactly one held worker source handle to the isolated bootstrap."""

    if os.name != "nt":
        yield ("--worker-fd", str(descriptor)), {"pass_fds": (descriptor,)}
        return
    import msvcrt

    handle = int(msvcrt.get_osfhandle(descriptor))
    if handle == -1:
        raise LocalPlaygroundGenerationError("Held Playground worker transport is unavailable.")
    yield ("--worker-handle", str(handle)), {"inherited_handles": (handle,)}


@contextmanager
def _operation_checked_pin_executable(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_metadata_sha256: str,
    operation_check: Callable[[], None],
):
    manager = pin_executable(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        expected_metadata_sha256=expected_metadata_sha256,
    )
    interpreter = _wait_for_operation_call(
        manager.__enter__,
        operation_check=operation_check,
        interrupted_cleanup=lambda _value, exc: manager.__exit__(type(exc), exc, exc.__traceback__),
    )
    try:
        operation_check()
        yield interpreter
        operation_check()
    except BaseException as exc:
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException:
            if isinstance(exc, (GenerationCancelledError, GenerationTimedOutError)):
                raise exc from None
            raise
        raise
    else:
        suppressed = _wait_for_operation_call(
            lambda: manager.__exit__(None, None, None),
            operation_check=operation_check,
        )
        if suppressed:
            raise LocalPlaygroundGenerationError("Pinned executable cleanup suppressed an unexpected failure.")


def _minimal_sampler_environment(
    invocation: Path,
    *,
    project_root: Path,
    import_paths: Sequence[str] = (),
    runtime_paths: Sequence[str] = (),
) -> dict[str, str]:
    value = {
        "HOME": str(invocation),
        "USERPROFILE": str(invocation),
        "APPDATA": str(invocation),
        "LOCALAPPDATA": str(invocation),
        "TEMP": str(invocation),
        "TMP": str(invocation),
        "TMPDIR": str(invocation),
        "TORCH_HOME": str(invocation),
        "XDG_CACHE_HOME": str(invocation),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(invocation / "pycache"),
        "PYTHONUTF8": "1",
        "SPRITELAB_PROGRESS": "0",
        "SPRITELAB_BOUND_BOOTSTRAP_SHA256": _BOUND_PLAYGROUND_WORKER_BOOTSTRAP_SHA256,
        "SPRITELAB_PROJECT_ROOT": str(project_root),
        "SPRITELAB_ISOLATED_PATHS": os.pathsep.join(import_paths),
        "SPRITELAB_RUNTIME_ROOTS": os.pathsep.join(runtime_paths),
    }
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            if os.environ.get(name):
                value[name] = os.environ[name]
    return dict(sorted(value.items()))


def _wait_for_operation_call(
    action: Callable[[], Any],
    *,
    operation_check: Callable[[], None],
    interrupted_cleanup: Callable[[Any, BaseException], Any] | None = None,
) -> Any:
    completed = threading.Event()
    result: dict[str, Any] = {}

    def run_action() -> None:
        try:
            result["value"] = action()
        except BaseException as exc:
            result["error"] = exc
        finally:
            completed.set()

    worker = threading.Thread(
        target=run_action,
        name="spritelab-playground-operation-scan",
        daemon=True,
    )

    def await_interrupted_cleanup(interruption: BaseException) -> None:
        reaped = threading.Event()

        def reap_late_result() -> None:
            try:
                completed.wait()
                if interrupted_cleanup is not None and "value" in result:
                    interrupted_cleanup(result["value"], interruption)
            except BaseException:
                pass
            finally:
                reaped.set()

        threading.Thread(
            target=reap_late_result,
            name="spritelab-playground-operation-reaper",
            daemon=True,
        ).start()
        reaped.wait(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)

    operation_check()
    worker.start()
    try:
        while not completed.wait(0.05):
            operation_check()
        operation_check()
    except BaseException as exc:
        await_interrupted_cleanup(exc)
        raise
    worker.join(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)
    if worker.is_alive():
        error = LocalPlaygroundGenerationError("A completed Playground operation did not stop safely.")
        await_interrupted_cleanup(error)
        raise error
    error = result.get("error")
    if isinstance(error, BaseException):
        raise error
    try:
        operation_check()
    except BaseException as exc:
        await_interrupted_cleanup(exc)
        raise
    return result.get("value")


def _join_thread_with_operation_check(
    thread: threading.Thread,
    *,
    timeout: float,
    operation_check: Callable[[], None],
) -> None:
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        operation_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        thread.join(timeout=min(0.05, remaining))
    operation_check()


def _safe_close_windows_handle(handle: Any) -> None:
    if type(handle) is int and handle:
        try:
            close_windows_handle(handle)
        except BaseException:
            pass


def _start_contained_process_cleanup_reaper(
    process: subprocess.Popen[Any],
    *,
    job_handle: Any = None,
) -> threading.Event:
    completed = threading.Event()

    def reap() -> None:
        try:
            _safe_close_windows_handle(job_handle)
            _terminate_contained_process(process)
        finally:
            completed.set()

    threading.Thread(
        target=reap,
        name="spritelab-playground-process-reaper",
        daemon=True,
    ).start()
    return completed


def _wait_for_process_activation(
    process: subprocess.Popen[Any],
    *,
    activate: Callable[[], int | None],
    operation_check: Callable[[], None],
    cleanup_delegated: threading.Event | None = None,
) -> int | None:
    completed = threading.Event()
    interrupted = threading.Event()
    result_lock = threading.Lock()
    result: dict[str, Any] = {}

    def run_activation() -> None:
        try:
            handle = activate()
        except BaseException as exc:
            with result_lock:
                result["error"] = exc
        else:
            late_handle: Any = None
            with result_lock:
                if interrupted.is_set():
                    late_handle = handle
                else:
                    result["handle"] = handle
            _safe_close_windows_handle(late_handle)
        finally:
            completed.set()

    def await_interrupted_cleanup() -> None:
        with result_lock:
            interrupted.set()
            handle = result.pop("handle", None)
        reaped = _start_contained_process_cleanup_reaper(process, job_handle=handle)
        if cleanup_delegated is not None:
            cleanup_delegated.set()
        reaped.wait(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)

    activation = threading.Thread(
        target=run_activation,
        name="spritelab-playground-process-activation",
        daemon=True,
    )
    operation_check()
    activation.start()
    try:
        while not completed.wait(0.05):
            operation_check()
        operation_check()
    except BaseException:
        await_interrupted_cleanup()
        raise
    activation.join(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)
    if activation.is_alive():
        error = LocalPlaygroundGenerationError("Playground process activation did not stop safely.")
        await_interrupted_cleanup()
        raise error
    with result_lock:
        error = result.get("error")
        handle = result.pop("handle", None)
    if isinstance(error, BaseException):
        reaped = _start_contained_process_cleanup_reaper(process)
        if cleanup_delegated is not None:
            cleanup_delegated.set()
        reaped.wait(timeout=_INTERRUPTED_CLEANUP_WAIT_SECONDS)
        raise error
    try:
        operation_check()
    except BaseException:
        with result_lock:
            result["handle"] = handle
        await_interrupted_cleanup()
        raise
    return handle if type(handle) is int and handle else None


def _terminate_contained_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform.startswith("linux"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except BaseException:
        try:
            if sys.platform.startswith("linux"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        except BaseException:
            pass


def _parse_deadline(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LocalPlaygroundGenerationError("The durable Playground deadline is malformed.") from exc
    if parsed.tzinfo is None:
        raise LocalPlaygroundGenerationError("The durable Playground deadline is malformed.")
    return parsed.astimezone(timezone.utc)


def _operation_checked_training_code_identity_source_paths(
    project_root: Path,
    *,
    operation_check: Callable[[], None] | None,
) -> tuple[Path, ...]:
    from spritelab.training.campaign import (
        TRAINING_CODE_IDENTITY_MANDATORY_FILES,
        TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS,
        CampaignValidationError,
        training_code_identity_source_paths,
    )

    if operation_check is None:
        return training_code_identity_source_paths(project_root)
    operation_check()
    root = Path(project_root).resolve()
    recursive_roots = tuple(root / relative for relative in TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS)
    missing: list[Path] = []
    for path in recursive_roots:
        operation_check()
        if not path.is_dir():
            missing.append(path)
    if missing:
        raise CampaignValidationError(
            "bound code-identity source root is missing: " + ", ".join(str(path) for path in missing)
        )
    relative_roots = [path.relative_to(root).as_posix() for path in recursive_roots]
    mandatory = tuple(root / relative for relative in TRAINING_CODE_IDENTITY_MANDATORY_FILES)
    missing_files: list[Path] = []
    for path in mandatory:
        operation_check()
        if not path.is_file():
            missing_files.append(path)
    if missing_files:
        raise CampaignValidationError(
            "bound code-identity source is missing: "
            + ", ".join(path.relative_to(root).as_posix() for path in missing_files)
        )

    try:
        stdout = pinned_git_ls_files(
            root,
            (*relative_roots, *TRAINING_CODE_IDENTITY_MANDATORY_FILES),
            timeout_seconds=10.0,
            operation_check=operation_check,
        )
    except (GenerationCancelledError, GenerationTimedOutError):
        raise
    except subprocess.TimeoutExpired as exc:
        raise CampaignValidationError("tracked code-identity source inventory timed out") from exc
    except (OSError, PinnedExecutableError, subprocess.SubprocessError) as exc:
        raise CampaignValidationError(f"tracked code-identity source inventory failed: {exc}") from exc
    operation_check()
    tracked: set[Path] = set()
    for raw in stdout.split(b"\0"):
        operation_check()
        if raw:
            relative = raw.decode("utf-8")
            if relative.endswith(".py"):
                tracked.add(root / relative)
    missing_tracked: list[Path] = []
    for path in tracked:
        operation_check()
        if not path.is_file():
            missing_tracked.append(path)
    if missing_tracked:
        raise CampaignValidationError(
            "tracked production Python source is missing: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(missing_tracked))
        )

    source_root = root / "src" / "spritelab"
    discovered: set[Path] = set()
    pending = [source_root]
    while pending:
        operation_check()
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise CampaignValidationError("production Python inventory could not be scanned safely") from exc
        try:
            with entries:
                for entry in entries:
                    operation_check()
                    path = Path(entry.path)
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise CampaignValidationError("production Python inventory changed while scanning") from exc
                    attributes = int(getattr(metadata, "st_file_attributes", 0))
                    reparse = bool(attributes & 0x400)
                    if entry.is_symlink() or reparse or os.path.ismount(path):
                        raise CampaignValidationError(
                            f"production source inventory crosses an unsafe seam: {path.relative_to(root).as_posix()}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False) and path.suffix == ".py":
                        discovered.add(path)
        except OSError as exc:
            raise CampaignValidationError("production Python inventory changed while scanning") from exc
    operation_check()
    untracked = discovered - tracked
    if untracked:
        raise CampaignValidationError(
            "untracked production Python source would escape code identity: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(untracked))
        )
    if not set(mandatory).issubset(tracked):
        absent = set(mandatory) - tracked
        raise CampaignValidationError(
            "mandatory production Python source is not tracked: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(absent))
        )
    operation_check()
    result = tuple(sorted(tracked, key=lambda path: path.relative_to(root).as_posix()))
    operation_check()
    return result


def _source_metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
        int(getattr(metadata, "st_mtime_ns", 0)),
        int(getattr(metadata, "st_ctime_ns", 0)),
        int(getattr(metadata, "st_nlink", 1)),
    )


def _anchored_production_python_metadata(
    project_root: Path,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[Path, tuple[int, int, int, int, int, int, int]]:
    """Snapshot every production-Python pathname through one anchored tree."""

    root = project_root.resolve()
    source_relative = PurePosixPath("src/spritelab")
    records: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in source_relative.parts:
            if operation_check is not None:
                operation_check()
            anchor = stack.enter_context(anchor.open_directory_immovable(part))

        def walk(current: AnchoredDirectory, relative: PurePosixPath) -> None:
            if operation_check is not None:
                operation_check()
            boundary_device = int(current.directory_metadata().st_dev)
            for name in current.names():
                if operation_check is not None:
                    operation_check()
                metadata = current.lstat(name)
                if _is_link_or_reparse(metadata) or int(metadata.st_dev) != boundary_device:
                    raise LocalPlaygroundGenerationError(
                        "Production Python inventory crosses a linked or mounted filesystem seam."
                    )
                item_relative = relative / name
                if stat.S_ISDIR(metadata.st_mode):
                    with current.open_directory_immovable(name) as child:
                        walk(child, item_relative)
                elif stat.S_ISREG(metadata.st_mode):
                    if name.endswith(".py"):
                        if int(getattr(metadata, "st_nlink", 1)) != 1 or metadata.st_size < 0:
                            raise LocalPlaygroundGenerationError(
                                "Production Python inventory contains an unsafe source file."
                            )
                        records[root / Path(*item_relative.parts)] = _source_metadata_signature(metadata)
                else:
                    raise LocalPlaygroundGenerationError(
                        "Production Python inventory contains a special filesystem entry."
                    )
                if operation_check is not None:
                    operation_check()

        walk(anchor, source_relative)
    return dict(sorted(records.items(), key=lambda item: item[0].as_posix()))


def _file_sha256(
    path: Path,
    *,
    project_root: Path,
    expected_metadata: tuple[int, int, int, int, int, int, int],
    operation_check: Callable[[], None] | None = None,
) -> str:
    """Hash one exact anchored source inode from the inventory snapshot."""

    root = project_root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LocalPlaygroundGenerationError("A Playground execution source escapes the project root.") from exc
    if not relative.parts:
        raise LocalPlaygroundGenerationError("A Playground execution source path is invalid.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts[:-1]:
            if operation_check is not None:
                operation_check()
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        name = relative.parts[-1]
        before = anchor.lstat(name)
        if _source_metadata_signature(before) != expected_metadata:
            raise LocalPlaygroundGenerationError("Production Python source changed after inventory discovery.")
        payload = _read_anchored_regular_bytes(
            anchor,
            name,
            maximum_bytes=128 * 1024 * 1024,
            label="production Python source",
            operation_check=operation_check,
        )
        if _source_metadata_signature(anchor.lstat(name)) != expected_metadata:
            raise LocalPlaygroundGenerationError("Production Python source changed while it was hashed.")
        anchor.verify()
    if operation_check is not None:
        operation_check()
    return hashlib.sha256(payload).hexdigest()


def _read_safe_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    if operation_check is not None:
        operation_check()
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(getattr(before, "st_nlink", 1)) != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise LocalPlaygroundGenerationError(f"Local {label} is unsafe or exceeds its safety limit.")
        chunks: list[bytes] = []
        total = 0
        while True:
            if operation_check is not None:
                operation_check()
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise LocalPlaygroundGenerationError(f"Local {label} exceeds its safety limit.")
            if operation_check is not None:
                operation_check()
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_nlink,
            getattr(before, "st_mtime_ns", None),
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_nlink,
            getattr(after, "st_mtime_ns", None),
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_nlink,
            getattr(current, "st_mtime_ns", None),
        )
        if before_identity != after_identity or after_identity != current_identity:
            raise LocalPlaygroundGenerationError(f"Local {label} changed while it was read.")
        result = b"".join(chunks)
        if operation_check is not None:
            operation_check()
        return result
    except LocalPlaygroundGenerationError:
        raise
    except OSError as exc:
        raise LocalPlaygroundGenerationError(f"Local {label} is missing or unsafe.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_anchored_regular_bytes(
    anchor: AnchoredDirectory,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    if operation_check is not None:
        operation_check()
    descriptor = -1
    try:
        descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        before = os.fstat(descriptor)
        current = anchor.lstat(name)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_nlink,
            getattr(before, "st_mtime_ns", None),
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_nlink,
            getattr(current, "st_mtime_ns", None),
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or int(getattr(before, "st_nlink", 1)) != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
            or identity_before != identity_current
        ):
            raise LocalPlaygroundGenerationError(f"Local {label} is unsafe or exceeds its safety limit.")
        chunks: list[bytes] = []
        total = 0
        while True:
            if operation_check is not None:
                operation_check()
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise LocalPlaygroundGenerationError(f"Local {label} exceeds its safety limit.")
            if operation_check is not None:
                operation_check()
        after = os.fstat(descriptor)
        final = anchor.lstat(name)
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_nlink,
            getattr(after, "st_mtime_ns", None),
        )
        identity_final = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_nlink,
            getattr(final, "st_mtime_ns", None),
        )
        if identity_before != identity_after or identity_after != identity_final:
            raise LocalPlaygroundGenerationError(f"Local {label} changed while it was read.")
        result = b"".join(chunks)
        if operation_check is not None:
            operation_check()
        return result
    except LocalPlaygroundGenerationError:
        raise
    except OSError as exc:
        raise LocalPlaygroundGenerationError(f"Local {label} is missing or unsafe.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_identity(value: Mapping[str, Any], identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    payload = strict_json_dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = strict_json_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads_no_duplicates(payload: bytes) -> Any:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON value")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8", errors="strict"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.isoformat() != value:
        return None
    return parsed


def _valid_invocation_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value == value.strip()
        and "/" not in value
        and "\\" not in value
        and all(32 < ord(character) < 127 for character in value)
    )


def _valid_lease_owner(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"pid", "process_birth_identity"}:
        return False
    pid = value.get("pid")
    birth = value.get("process_birth_identity")
    return (
        type(pid) is int
        and pid > 0
        and isinstance(birth, str)
        and 1 <= len(birth) <= 128
        and birth == birth.strip()
        and "/" not in birth
        and "\\" not in birth
        and all(character.isascii() and (character.isalnum() or character in ":-_") for character in birth)
    )


def _validate_lease_v2(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "lease_id",
        "lease_identity",
        "transition_sequence",
        "prior_lease_identity",
        "status",
        "owner",
        "acquired_at",
        "heartbeat_at",
        "ended_at",
        "retryable",
        "invocation_id",
        "recovered_orphan",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise LocalPlaygroundGenerationError("Local sampler lease schema is malformed.")
    lease_id = value.get("lease_id")
    sequence = value.get("transition_sequence")
    prior_identity = value.get("prior_lease_identity")
    status = value.get("status")
    invocation_id = value.get("invocation_id")
    recovered = value.get("recovered_orphan")
    if (
        value.get("schema_version") != _PLAYGROUND_LEASE_SCHEMA
        or not isinstance(lease_id, str)
        or len(lease_id) != 32
        or any(character not in "0123456789abcdef" for character in lease_id)
        or not _is_sha256(value.get("lease_identity"))
        or type(sequence) is not int
        or sequence < 0
        or (sequence == 0 and prior_identity is not None)
        or (sequence > 0 and not _is_sha256(prior_identity))
        or status not in {"ACTIVE", "COMPLETE", "FAILED"}
        or not _valid_lease_owner(value.get("owner"))
        or type(value.get("retryable")) is not bool
        or value.get("retryable") is not (status == "FAILED")
        or (invocation_id is not None and not _valid_invocation_id(invocation_id))
    ):
        raise LocalPlaygroundGenerationError("Local sampler lease fields are malformed.")
    acquired_at = _parse_utc_timestamp(value.get("acquired_at"))
    heartbeat_at = _parse_utc_timestamp(value.get("heartbeat_at"))
    ended_at = _parse_utc_timestamp(value.get("ended_at")) if value.get("ended_at") is not None else None
    if (
        acquired_at is None
        or heartbeat_at is None
        or acquired_at > heartbeat_at
        or (status == "ACTIVE" and ended_at is not None)
        or (status != "ACTIVE" and (ended_at is None or heartbeat_at > ended_at))
    ):
        raise LocalPlaygroundGenerationError("Local sampler lease timestamps are malformed.")
    if recovered is not None:
        if (
            not isinstance(recovered, dict)
            or set(recovered) != {"lease_id", "status", "retryable"}
            or not isinstance(recovered.get("lease_id"), str)
            or not recovered["lease_id"]
            or len(recovered["lease_id"]) > 128
            or recovered.get("status") != "ORPHANED"
            or recovered.get("retryable") is not True
        ):
            raise LocalPlaygroundGenerationError("Local sampler recovered lease evidence is malformed.")
    if value["lease_identity"] != _record_identity(value, "lease_identity"):
        raise LocalPlaygroundGenerationError("Local sampler lease identity is malformed.")
    return dict(value)


def _read_lease(
    anchor: AnchoredDirectory,
    name: str,
    *,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if operation_check is not None:
        operation_check()
    if not anchor.lexists(name):
        if operation_check is not None:
            operation_check()
        return {}
    content = _read_anchored_regular_bytes(
        anchor,
        name,
        maximum_bytes=64 * 1024,
        label="sampler lease",
        operation_check=operation_check,
    )
    try:
        value = _strict_json_loads_no_duplicates(content)
    except (UnicodeError, ValueError) as exc:
        raise LocalPlaygroundGenerationError("Local sampler lease is malformed.") from exc
    if not isinstance(value, dict):
        raise LocalPlaygroundGenerationError("Local sampler lease schema is malformed.")
    if value.get("schema_version") == _PLAYGROUND_LEASE_SCHEMA:
        if content != (strict_json_dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"):
            raise LocalPlaygroundGenerationError("Local sampler lease is not canonical.")
        result = _validate_lease_v2(value)
        if operation_check is not None:
            operation_check()
        return result
    if value.get("schema_version") != "spritelab.playground-sampler-lease.v1":
        raise LocalPlaygroundGenerationError("Local sampler lease schema is malformed.")
    if value.get("status") not in {"ACTIVE", "COMPLETE", "FAILED"}:
        raise LocalPlaygroundGenerationError("Local sampler lease status is malformed.")
    if not isinstance(value.get("lease_id"), str) or not value["lease_id"]:
        raise LocalPlaygroundGenerationError("Local sampler lease identity is malformed.")
    if value.get("status") == "ACTIVE" and type(value.get("owner_pid")) is not int:
        raise LocalPlaygroundGenerationError("Active local sampler lease owner is malformed.")
    if operation_check is not None:
        operation_check()
    return value


def _write_lease(
    anchor: AnchoredDirectory,
    name: str,
    value: Mapping[str, Any],
    *,
    operation_check: Callable[[], None] | None = None,
) -> None:
    if operation_check is not None:
        operation_check()
    validated = _validate_lease_v2(dict(value))
    if anchor.lexists(name):
        metadata = anchor.lstat(name)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_link_or_reparse(metadata)
            or int(getattr(metadata, "st_nlink", 1)) != 1
        ):
            raise LocalPlaygroundGenerationError("Local sampler lease crosses an unsafe filesystem seam.")
    content = (strict_json_dumps(validated, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if operation_check is not None:
        operation_check()
    anchor.atomic_write_bytes(name, content)
    if operation_check is not None:
        operation_check()


@contextmanager
def _interprocess_lock(
    anchor: AnchoredDirectory,
    name: str,
    *,
    timeout: float = 5.0,
    operation_check: Callable[[], None] | None = None,
):
    if operation_check is not None:
        operation_check()
    flags = os.O_RDWR | os.O_CREAT | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = anchor.open_file_immovable(name, flags, 0o600)
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        identity = OwnedFileIdentity.from_stat(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or int(getattr(metadata, "st_nlink", 1)) != 1
            or not identity.matches(anchor.lstat(name))
        ):
            raise LocalPlaygroundGenerationError("Local sampler lock is not a regular single-link file.")
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            import msvcrt

            while not acquired:
                if operation_check is not None:
                    operation_check()
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the local sampler lease lock.") from None
                    time.sleep(0.05)
        else:  # pragma: no cover - exercised in non-Windows CI.
            import fcntl

            while not acquired:
                if operation_check is not None:
                    operation_check()
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the local sampler lease lock.") from None
                    time.sleep(0.05)
        if operation_check is not None:
            operation_check()
        if not identity.matches(os.fstat(descriptor)) or not identity.matches(anchor.lstat(name)):
            raise LocalPlaygroundGenerationError("Local sampler lock identity changed while it was acquired.")
        yield
        if not identity.matches(os.fstat(descriptor)) or not identity.matches(anchor.lstat(name)):
            raise LocalPlaygroundGenerationError("Local sampler lock identity changed while it was held.")
        if operation_check is not None:
            operation_check()
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised in non-Windows CI.
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_birth_identity(pid: int) -> str | None:
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            created = _FileTime()
            exited = _FileTime()
            kernel = _FileTime()
            user = _FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            ticks = (int(created.high) << 32) | int(created.low)
            return f"windows-filetime:{ticks:016x}"
        finally:
            kernel32.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            content = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="strict")
            close = content.rfind(")")
            fields = content[close + 2 :].split() if close >= 0 else []
            start_ticks = fields[19]
            if not start_ticks.isdecimal():
                return None
            return f"linux-proc-start:{start_ticks}"
        except (OSError, UnicodeError, IndexError):
            return None
    if pid == os.getpid():
        return _PROCESS_INSTANCE_FALLBACK
    return None


def _current_process_owner() -> dict[str, Any]:
    birth_identity = _process_birth_identity(os.getpid())
    if birth_identity is None:
        raise LocalPlaygroundGenerationError("The exact local sampler process identity is unavailable.")
    return {"pid": os.getpid(), "process_birth_identity": birth_identity}


def _lease_owner_is_current(owner: Any) -> bool:
    if not _valid_lease_owner(owner) or owner.get("pid") != os.getpid():
        return False
    return owner.get("process_birth_identity") == _process_birth_identity(os.getpid())


def _lease_owner_still_live(owner: Any) -> bool:
    if not _valid_lease_owner(owner):
        raise LocalPlaygroundGenerationError("Active local sampler lease owner is malformed.")
    pid = int(owner["pid"])
    current_birth = _process_birth_identity(pid)
    if current_birth is not None:
        return current_birth == owner["process_birth_identity"]
    return _process_is_alive(pid)


def _safe_relative_png(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise LocalPlaygroundGenerationError("Local generation manifest contains an invalid output path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LocalPlaygroundGenerationError("Local generation manifest output escapes its run.")
    for part in relative.parts:
        normalized = unicodedata.normalize("NFKC", part)
        stem = normalized.rstrip(" .").split(".", 1)[0].casefold()
        if (
            normalized in {"", ".", ".."}
            or "/" in normalized
            or "\\" in normalized
            or any(character in '<>:"|?*' or ord(character) < 32 for character in normalized)
            or stem in _WINDOWS_RESERVED_NAMES
            or normalized != normalized.rstrip(" .")
        ):
            raise LocalPlaygroundGenerationError("Local generation manifest contains an unsafe output name.")
    if relative.suffix.casefold() != ".png":
        raise LocalPlaygroundGenerationError("Local generation output must be a PNG file.")
    return relative


def _unsafe_existing_path(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    return (
        _is_link_or_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or int(getattr(metadata, "st_nlink", 1)) != 1
    )


def _valid_sampler_result(value: Any, control: Mapping[str, Any]) -> bool:
    expected_keys = {
        "schema_version",
        "result_identity",
        "control_identity",
        "bootstrap_identity",
        "worker_sha256",
        "checkpoint_sha256",
        "prompts_sha256",
        "code_inventory_identity",
        "runtime_closure_identity",
        "workspace_identity",
        "deadline_at",
        "report",
        "runtime_identity",
        "write_confinement",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version") != _PLAYGROUND_RESULT_SCHEMA
        or not _is_sha256(value.get("result_identity"))
        or value.get("result_identity") != _record_identity(value, "result_identity")
        or not isinstance(value.get("report"), dict)
        or set(value["report"]) != {"sample_count"}
        or type(value["report"].get("sample_count")) is not int
        or value["report"].get("sample_count") != control.get("image_count")
        or not isinstance(value.get("runtime_identity"), dict)
        or not isinstance(value.get("workspace_identity"), dict)
        or set(value["workspace_identity"]) != {"device", "inode"}
        or any(type(value["workspace_identity"].get(key)) is not int for key in ("device", "inode"))
        or not _valid_pathless_confinement_evidence(
            value.get("write_confinement"),
            workspace_identity=value.get("workspace_identity"),
        )
    ):
        return False
    for key in (
        "control_identity",
        "bootstrap_identity",
        "worker_sha256",
        "checkpoint_sha256",
        "prompts_sha256",
        "code_inventory_identity",
        "runtime_closure_identity",
        "workspace_identity",
        "deadline_at",
    ):
        if value.get(key) != control.get(key):
            return False
    runtime = value["runtime_identity"]
    return runtime.get("runtime_closure_identity") == control.get("runtime_closure_identity")


def _valid_pathless_confinement_evidence(
    value: Any,
    *,
    workspace_identity: Any = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "strategy",
        "platform",
        "kernel_abi",
        "root_identity_sha256",
        "handled_access_fs",
        "allowed_access_fs",
        "no_new_privileges",
        "restricted_token",
        "integrity_level_rid",
        "mandatory_no_write_up",
        "workspace_integrity_level_rid",
        "startup_integrity_level_rid",
        "bootstrap_lowered_before_worker_import",
        "new_thread_integrity_level_rid",
        "raise_to_low_denied",
        "medium_probe_write_denied",
        "low_world_probe_write_denied",
        "untrusted_world_outside_guaranteed",
        "job_kill_on_close",
        "job_active_process_limit",
        "paths_exposed",
    }
    if set(value) != expected_keys or value.get("schema_version") != "spritelab.write-confinement-evidence.v3":
        return False
    if value.get("paths_exposed") is not False:
        return False
    if (
        not isinstance(workspace_identity, Mapping)
        or set(workspace_identity) != {"device", "inode"}
        or any(type(workspace_identity.get(key)) is not int for key in ("device", "inode"))
        or any(int(workspace_identity[key]) < 0 for key in ("device", "inode"))
    ):
        return False
    expected_root_digest = hashlib.sha256(
        strict_json_dumps(
            {"device": workspace_identity["device"], "inode": workspace_identity["inode"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    digest = value.get("root_identity_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != expected_root_digest
    ):
        return False
    for key in (
        "no_new_privileges",
        "restricted_token",
        "mandatory_no_write_up",
        "bootstrap_lowered_before_worker_import",
        "raise_to_low_denied",
        "medium_probe_write_denied",
        "low_world_probe_write_denied",
        "untrusted_world_outside_guaranteed",
        "job_kill_on_close",
    ):
        if type(value.get(key)) is not bool:
            return False
    for key in (
        "kernel_abi",
        "handled_access_fs",
        "allowed_access_fs",
        "integrity_level_rid",
        "workspace_integrity_level_rid",
        "startup_integrity_level_rid",
        "new_thread_integrity_level_rid",
        "job_active_process_limit",
    ):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            return False
    if value.get("strategy") == "linux-landlock-v1":
        abi = value.get("kernel_abi")
        expected_handled = 32_754
        if abi >= 5:
            expected_handled |= 1 << 15
        if abi >= 9:
            expected_handled |= 1 << 16
        exact_linux_evidence = {
            "handled_access_fs": expected_handled,
            "allowed_access_fs": 25_010,
            "no_new_privileges": True,
            "restricted_token": False,
            "integrity_level_rid": 0,
            "mandatory_no_write_up": False,
            "workspace_integrity_level_rid": 0,
            "startup_integrity_level_rid": 0,
            "bootstrap_lowered_before_worker_import": False,
            "new_thread_integrity_level_rid": 0,
            "raise_to_low_denied": False,
            "medium_probe_write_denied": False,
            "low_world_probe_write_denied": False,
            "untrusted_world_outside_guaranteed": False,
            "job_kill_on_close": False,
            "job_active_process_limit": 0,
            "paths_exposed": False,
        }
        return (
            value.get("platform") == "linux"
            and 3 <= abi <= 10
            and all(value.get(key) == expected for key, expected in exact_linux_evidence.items())
        )
    if value.get("strategy") != "windows-bootstrap-to-untrusted-integrity-v1":
        return False
    if value.get("platform") != "windows":
        return False
    exact_windows_evidence = {
        "kernel_abi": 0,
        "handled_access_fs": 0,
        "allowed_access_fs": 0,
        "no_new_privileges": False,
        "integrity_level_rid": 0,
        "mandatory_no_write_up": True,
        "workspace_integrity_level_rid": 0,
        "startup_integrity_level_rid": 4096,
        "bootstrap_lowered_before_worker_import": True,
        "new_thread_integrity_level_rid": 0,
        "raise_to_low_denied": True,
        "medium_probe_write_denied": True,
        "low_world_probe_write_denied": True,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": True,
        "job_active_process_limit": 1,
        "paths_exposed": False,
    }
    return all(value.get(key) == expected for key, expected in exact_windows_evidence.items())


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


__all__ = ["LocalCheckpointPlaygroundGenerator", "LocalPlaygroundGenerationError"]

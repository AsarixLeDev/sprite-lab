"""Lazy, local-only Playground adapter for Sprite Lab challenger checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import stat
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from spritelab.product_core import strict_json_dumps
from spritelab.product_features.evaluation.playground import GeneratedAsset
from spritelab.utils.safe_fs import atomic_write_bytes, require_confined_path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PROMPT_CHARACTERS = 2_000
_MAX_GENERATED_PNG_BYTES = 4 * 1024 * 1024
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
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.work_root = require_confined_path(work_root, self.project_root)
        self._sampler = sampler
        self.last_runtime_identity: dict[str, Any] | None = None

    @property
    def code_identity_sha256(self) -> str:
        # Reuse the campaign's complete training execution inventory, then add
        # this adapter. This is evaluated only for an explicit generation plan.
        from spritelab.training.campaign import training_code_identity_source_paths

        source_paths = set(training_code_identity_source_paths(self.project_root))
        source_paths.add(Path(__file__).resolve())
        records = [
            {
                "path": path.relative_to(self.project_root).as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in source_paths
        ]
        payload = strict_json_dumps(
            {
                "schema_version": "spritelab.playground-local-code-identity.v1",
                "files": sorted(records, key=lambda row: row["path"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

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
        self.last_runtime_identity = None
        if weights != expected_variant:
            raise LocalPlaygroundGenerationError("Checkpoint variant selection changed before sampling.")
        checkpoint = self._validate_checkpoint(checkpoint)
        normalized_prompt = self._validate_request(
            prompt=prompt,
            seed=seed,
            sampling_steps=sampling_steps,
            guidance=guidance,
            image_count=image_count,
        )
        lease_id = self._acquire_lease()
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(lease_id, heartbeat_stop),
            name="spritelab-playground-lease-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            invocation = self._new_invocation_directory()
            self._update_lease(lease_id, invocation_id=invocation.name)
            snapshot = invocation / "checkpoint.snapshot.pt"
            self._snapshot_checkpoint(checkpoint, snapshot, expected_sha256=expected_sha256)
            self._validate_snapshot_checkpoint(
                snapshot,
                expected_step=expected_step,
                expected_variant=expected_variant,
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
            with prompts_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            output = invocation / "generated"
            output.mkdir()
            config = self._sample_config(
                checkpoint=snapshot,
                prompts=prompts_path,
                output=output,
                seed=seed,
                sampling_steps=sampling_steps,
                guidance=guidance,
                image_count=image_count,
            )
            report = dict((self._sampler or _run_challenger_sampler)(config))
            if report.get("sample_count") != image_count:
                raise LocalPlaygroundGenerationError("Local sampler returned an unexpected sample count.")
            self.last_runtime_identity = self._runtime_identity(report)
            assets = self._load_assets(
                output,
                expected_prompt=normalized_prompt,
                expected_seed=seed,
                expected_steps=sampling_steps,
                expected_guidance=float(guidance),
                expected_count=image_count,
            )
        except BaseException:
            self._release_lease(lease_id, status="FAILED", retryable=True)
            raise
        else:
            if not self._release_lease(lease_id, status="COMPLETE", retryable=False):
                raise LocalPlaygroundGenerationError("Local sampler lease could not be finalized safely.")
            return assets
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=6.0)

    @property
    def _lease_path(self) -> Path:
        return self.work_root / "sampler-lease.json"

    @property
    def _lease_lock_path(self) -> Path:
        return self.work_root / ".sampler-lease.lock"

    def _acquire_lease(self) -> str:
        self._ensure_work_root()
        with _interprocess_lock(self._lease_lock_path):
            previous = _read_lease(self._lease_path)
            recovered: dict[str, Any] | None = None
            if previous and previous.get("status") == "ACTIVE":
                owner_pid = previous.get("owner_pid")
                if type(owner_pid) is not int:
                    raise LocalPlaygroundGenerationError("Active local sampler lease owner is malformed.")
                if _process_is_alive(owner_pid):
                    raise LocalPlaygroundGenerationError("Another local Playground sampler is already active.")
                recovered = {
                    "lease_id": str(previous.get("lease_id") or "unknown"),
                    "status": "ORPHANED",
                    "retryable": True,
                }
            lease_id = uuid.uuid4().hex
            now = _utc_now()
            value = {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": lease_id,
                "status": "ACTIVE",
                "owner_pid": os.getpid(),
                "acquired_at": now,
                "heartbeat_at": now,
                "retryable": False,
                "invocation_id": None,
                "recovered_orphan": recovered,
            }
            _write_lease(self._lease_path, value)
            return lease_id

    def _update_lease(self, lease_id: str, **updates: Any) -> None:
        with _interprocess_lock(self._lease_lock_path):
            state = _read_lease(self._lease_path)
            if state.get("lease_id") != lease_id or state.get("status") != "ACTIVE":
                raise LocalPlaygroundGenerationError("Local sampler lease ownership was lost.")
            state.update(updates)
            state["heartbeat_at"] = _utc_now()
            _write_lease(self._lease_path, state)

    def _release_lease(self, lease_id: str, *, status: str, retryable: bool) -> bool:
        try:
            with _interprocess_lock(self._lease_lock_path):
                state = _read_lease(self._lease_path)
                if state.get("lease_id") != lease_id or state.get("status") != "ACTIVE":
                    return False
                state.update(
                    {
                        "status": status,
                        "heartbeat_at": _utc_now(),
                        "ended_at": _utc_now(),
                        "retryable": retryable,
                    }
                )
                _write_lease(self._lease_path, state)
        except (OSError, TimeoutError, LocalPlaygroundGenerationError):
            return False
        return True

    def _heartbeat_loop(self, lease_id: str, stop: threading.Event) -> None:
        while not stop.wait(5.0):
            try:
                self._update_lease(lease_id)
            except (OSError, TimeoutError, LocalPlaygroundGenerationError):
                return

    def _validate_checkpoint(self, checkpoint: Path) -> Path:
        candidate = require_confined_path(checkpoint, self.project_root)
        current = self.project_root
        for part in candidate.relative_to(self.project_root).parts:
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
        return candidate

    @staticmethod
    def _snapshot_checkpoint(source: Path, destination: Path, *, expected_sha256: str) -> None:
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise LocalPlaygroundGenerationError("Checkpoint SHA-256 expectation is malformed.")
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        source_fd = -1
        digest = hashlib.sha256()
        try:
            source_fd = os.open(source, flags)
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode) or int(getattr(before, "st_nlink", 1)) != 1:
                raise LocalPlaygroundGenerationError("The selected checkpoint is not one regular single-link file.")
            with destination.open("xb") as target:
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            after = os.fstat(source_fd)
            current = os.stat(source, follow_symlinks=False)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                getattr(before, "st_mtime_ns", None),
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                getattr(after, "st_mtime_ns", None),
            )
            identity_current = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                getattr(current, "st_mtime_ns", None),
            )
            if identity_before != identity_after or identity_after != identity_current:
                raise LocalPlaygroundGenerationError("Checkpoint changed while its sampling snapshot was created.")
            if digest.hexdigest() != expected_sha256:
                raise LocalPlaygroundGenerationError("Checkpoint snapshot does not match the durable catalog hash.")
        except LocalPlaygroundGenerationError:
            raise
        except OSError as exc:
            raise LocalPlaygroundGenerationError("Checkpoint snapshot could not be created safely.") from exc
        finally:
            if source_fd >= 0:
                os.close(source_fd)

    @staticmethod
    def _validate_snapshot_checkpoint(snapshot: Path, *, expected_step: int, expected_variant: str) -> None:
        from spritelab.training.checkpoint_io import load_checkpoint

        if type(expected_step) is not int or expected_step < 0:
            raise LocalPlaygroundGenerationError("Checkpoint step expectation is unavailable.")
        if expected_variant not in {"live", "ema"}:
            raise LocalPlaygroundGenerationError("Checkpoint variant expectation is malformed.")
        try:
            checkpoint = load_checkpoint(snapshot)
        except Exception as exc:
            raise LocalPlaygroundGenerationError("Checkpoint could not be loaded in safe weights-only mode.") from exc
        if checkpoint.get("model_type") != "generator_challenger":
            raise LocalPlaygroundGenerationError("Checkpoint model type is not supported by the local Playground.")
        if checkpoint.get("ema_weights") is not (expected_variant == "ema"):
            raise LocalPlaygroundGenerationError("Checkpoint EMA/live metadata does not match the selected variant.")
        step = checkpoint.get("step")
        global_step = checkpoint.get("global_step")
        if type(step) is not int or type(global_step) is not int or step != global_step or step != expected_step:
            raise LocalPlaygroundGenerationError("Checkpoint step metadata does not match durable catalog evidence.")

    @staticmethod
    def _runtime_identity(report: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        report_config = report.get("config")
        selected_device = str(
            report.get("device")
            or (report_config.get("device_resolved") if isinstance(report_config, Mapping) else None)
            or "auto"
        )
        if len(selected_device) > 80:
            selected_device = "unknown"
        return {
            "schema_version": "spritelab.playground-runtime-identity.v1",
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "selected_device": selected_device,
            "platform": sys.platform,
        }

    @staticmethod
    def _validate_request(
        *,
        prompt: str,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
    ) -> str:
        normalized = prompt.strip()
        if not normalized or len(normalized) > _MAX_PROMPT_CHARACTERS:
            raise LocalPlaygroundGenerationError("Prompt length is outside the local Playground limit.")
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise LocalPlaygroundGenerationError("Prompt contains unsupported control characters.")
        if type(seed) is not int or seed < 0:
            raise LocalPlaygroundGenerationError("Seed must be a non-negative integer.")
        if type(sampling_steps) is not int or not 1 <= sampling_steps <= 500:
            raise LocalPlaygroundGenerationError("Sampling steps are outside the supported range.")
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 0 <= float(guidance) <= 50:
            raise LocalPlaygroundGenerationError("Guidance is outside the supported range.")
        if not math.isfinite(float(guidance)):
            raise LocalPlaygroundGenerationError("Guidance is outside the supported range.")
        if type(image_count) is not int or not 1 <= image_count <= 16:
            raise LocalPlaygroundGenerationError("Image count is outside the supported range.")
        return normalized

    def _new_invocation_directory(self) -> Path:
        self._ensure_work_root()
        invocation = require_confined_path(
            self.work_root / f"playground-sampler-{uuid.uuid4().hex}",
            self.work_root,
        )
        invocation.mkdir()
        metadata = invocation.lstat()
        if _is_link_or_reparse(metadata) or invocation.is_mount() or not stat.S_ISDIR(metadata.st_mode):
            raise LocalPlaygroundGenerationError("Could not create a safe local sampling directory.")
        return invocation

    def _ensure_work_root(self) -> None:
        current = self.project_root
        relative = self.work_root.relative_to(self.project_root)
        for part in relative.parts:
            current = current / part
            try:
                current.mkdir()
            except FileExistsError:
                pass
            metadata = current.lstat()
            if _is_link_or_reparse(metadata) or current.is_mount() or not stat.S_ISDIR(metadata.st_mode):
                raise LocalPlaygroundGenerationError("Local Playground work root crosses an unsafe seam.")
            require_confined_path(current, self.project_root)

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
    ) -> Any:
        # Importing the challenger module imports Torch, so keep this behind the
        # explicit Generate action rather than application/router construction.
        from spritelab.training.generator_challenger import ChallengerSampleConfig

        return ChallengerSampleConfig(
            checkpoint=checkpoint,
            prompts=prompts,
            out_dir=output,
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

    def _load_assets(
        self,
        output: Path,
        *,
        expected_prompt: str,
        expected_seed: int,
        expected_steps: int,
        expected_guidance: float,
        expected_count: int,
    ) -> tuple[GeneratedAsset, ...]:
        try:
            output_metadata = output.lstat()
        except OSError as exc:
            raise LocalPlaygroundGenerationError("Local sampler output directory is unavailable.") from exc
        if not stat.S_ISDIR(output_metadata.st_mode) or _is_link_or_reparse(output_metadata) or os.path.ismount(output):
            raise LocalPlaygroundGenerationError("Local sampler output directory crosses an unsafe seam.")
        manifest_path = output / "generated_manifest.jsonl"
        manifest_bytes = _read_safe_regular_bytes(
            manifest_path,
            maximum_bytes=2 * 1024 * 1024,
            label="generation manifest",
        )
        records: list[dict[str, Any]] = []
        try:
            manifest_text = manifest_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise LocalPlaygroundGenerationError("Local generation manifest is not valid UTF-8.") from exc
        for line in manifest_text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocalPlaygroundGenerationError("Local generation manifest is malformed.") from exc
            if not isinstance(value, dict):
                raise LocalPlaygroundGenerationError("Local generation manifest contains a non-object row.")
            records.append(value)
        if len(records) != expected_count:
            raise LocalPlaygroundGenerationError("Local generation manifest has an unexpected row count.")

        assets: list[GeneratedAsset] = []
        collision_keys: set[str] = set()
        prompt_id_keys: set[str] = set()
        for index, record in enumerate(records):
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
            current = output
            for part in relative.parts[:-1]:
                current = current / part
                try:
                    metadata = current.lstat()
                except OSError as exc:
                    raise LocalPlaygroundGenerationError("Local sampler output is missing or unsafe.") from exc
                if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse(metadata) or os.path.ismount(current):
                    raise LocalPlaygroundGenerationError("Local sampler output crosses an unsafe directory seam.")
            content = _read_safe_regular_bytes(
                path,
                maximum_bytes=_MAX_GENERATED_PNG_BYTES,
                label="sampler PNG",
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
            assets.append(GeneratedAsset(content=content, media_type="image/png"))
        return tuple(assets)


def _run_challenger_sampler(config: Any) -> Mapping[str, Any]:
    from spritelab.training.generator_challenger import run_sample_generator_challenger

    return run_sample_generator_challenger(config)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safe_regular_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
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
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise LocalPlaygroundGenerationError(f"Local {label} exceeds its safety limit.")
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
        return b"".join(chunks)
    except LocalPlaygroundGenerationError:
        raise
    except OSError as exc:
        raise LocalPlaygroundGenerationError(f"Local {label} is missing or unsafe.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lease(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {}
    content = _read_safe_regular_bytes(path, maximum_bytes=64 * 1024, label="sampler lease")
    try:
        value = json.loads(content.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LocalPlaygroundGenerationError("Local sampler lease is malformed.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "spritelab.playground-sampler-lease.v1":
        raise LocalPlaygroundGenerationError("Local sampler lease schema is malformed.")
    if value.get("status") not in {"ACTIVE", "COMPLETE", "FAILED"}:
        raise LocalPlaygroundGenerationError("Local sampler lease status is malformed.")
    if not isinstance(value.get("lease_id"), str) or not value["lease_id"]:
        raise LocalPlaygroundGenerationError("Local sampler lease identity is malformed.")
    return value


def _write_lease(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path) and _unsafe_existing_path(path):
        raise LocalPlaygroundGenerationError("Local sampler lease crosses an unsafe filesystem seam.")
    content = (strict_json_dumps(dict(value), sort_keys=True, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, content)


@contextmanager
def _interprocess_lock(path: Path, *, timeout: float = 5.0):
    flags = os.O_RDWR | os.O_CREAT | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o600)
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, "st_nlink", 1)) != 1:
            raise LocalPlaygroundGenerationError("Local sampler lock is not a regular single-link file.")
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            import msvcrt

            while not acquired:
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
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the local sampler lease lock.") from None
                    time.sleep(0.05)
        yield
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


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


__all__ = ["LocalCheckpointPlaygroundGenerator", "LocalPlaygroundGenerationError"]

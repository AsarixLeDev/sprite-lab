"""Fail-closed checkpoint discovery from durable v3 training-run state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.product_features.evaluation.models import (
    CheckpointAvailability,
    CheckpointCandidate,
    CheckpointCatalog,
)

RUN_STATE_SCHEMA = "spritelab.v3.run-state.v1"
_STEP_PATTERN = re.compile(r"(?:step[_-]?)(\d+)", re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_PATTERNS = ("checkpoint*.pt", "checkpoint*.pth", "checkpoint*.ckpt", "checkpoint_step_*.json")
_CHECKPOINT_PATH_FIELDS = ("path", "checkpoint", "checkpoint_path", "file")
_EVIDENCE_ROWS_KEY = "_checkpoint_evidence_rows"
_EVIDENCE_SOURCE_KEY = "_checkpoint_evidence_source"


def file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(getattr(before, "st_nlink", 1)) != 1:
            raise OSError("checkpoint is not one regular single-link file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            getattr(before, "st_mtime_ns", None),
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", None),
        ) or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", None),
        ) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            getattr(current, "st_mtime_ns", None),
        ):
            raise OSError("checkpoint changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> dict[str, Any]:
    descriptor = -1
    try:
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(getattr(before, "st_nlink", 1)) != 1
            or before.st_size > 4 * 1024 * 1024
        ):
            return {}
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 4 * 1024 * 1024:
                return {}
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        before_identity = (before.st_dev, before.st_ino, before.st_size, getattr(before, "st_mtime_ns", None))
        after_identity = (after.st_dev, after.st_ino, after.st_size, getattr(after, "st_mtime_ns", None))
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            getattr(current, "st_mtime_ns", None),
        )
        if before_identity != after_identity or after_identity != current_identity:
            return {}
        value = json.loads(b"".join(chunks).decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return value if isinstance(value, dict) else {}


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & marker)


def _safe_direct_run_directory(path: Path, runs_directory: Path) -> bool:
    try:
        if path.parent != runs_directory or os.path.ismount(path):
            return False
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not _is_reparse(info)


def _safe_regular_descendant(path: Path, root: Path) -> bool:
    """Validate lexical and resolved containment without crossing link/mount seams."""

    try:
        lexical_root = Path(os.path.abspath(root))
        lexical_path = Path(os.path.abspath(path))
        relative = lexical_path.relative_to(lexical_root)
        if not relative.parts:
            return False
        current = lexical_root
        for part in relative.parts:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or os.path.ismount(current):
                return False
        final = lexical_path.lstat()
        if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
            return False
        resolved_root = root.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
        return resolved_path == lexical_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False


def _safe_directory_descendant(path: Path, root: Path) -> bool:
    try:
        lexical_root = Path(os.path.abspath(root))
        lexical_path = Path(os.path.abspath(path))
        relative = lexical_path.relative_to(lexical_root)
        if not relative.parts:
            return False
        current = lexical_root
        for part in relative.parts:
            current = current / part
            info = current.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or _is_reparse(info)
                or os.path.ismount(current)
            ):
                return False
        lexical_path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _safe_read_object(path: Path, root: Path) -> dict[str, Any]:
    return _read_object(path) if _safe_regular_descendant(path, root) else {}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _first(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value not in (None, "", [], {})), default)


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _configured_identity(*values: Any) -> str | None:
    value = _first(*values)
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def expected_dataset_identity(config: Mapping[str, Any]) -> str | None:
    """Resolve an active dataset identity without depending on one config implementation."""

    return _configured_identity(
        _nested(config, "evaluation", "dataset_identity"),
        _nested(config, "training", "dataset_identity"),
        _nested(config, "dataset", "identity"),
        _nested(config, "dataset", "freeze_identity"),
        config.get("dataset_identity"),
    )


def expected_training_view_identity(config: Mapping[str, Any]) -> str | None:
    """Resolve the independently configured training-view identity, when present."""

    return _configured_identity(
        _nested(config, "evaluation", "training_view_identity"),
        _nested(config, "training", "view_identity"),
        _nested(config, "dataset", "view_identity"),
        config.get("training_view_identity"),
    )


def _checkpoint_rows(run_directory: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completion = _safe_read_object(run_directory / "run_completion_marker.json", run_directory)
    manifest = _safe_read_object(run_directory / "checkpoint_manifest.json", run_directory)
    verification = _safe_read_object(run_directory / "checkpoint_verification.json", run_directory)
    sources = (
        ("state.checkpoints", state.get("checkpoints")),
        ("state.checkpoint", state.get("checkpoint")),
        ("state.backend_identity.checkpoints", _nested(state, "backend_identity", "checkpoints")),
        ("state.backend_identity.checkpoint", _nested(state, "backend_identity", "checkpoint")),
        ("completion.checkpoints", completion.get("checkpoints")),
        ("completion.checkpoint_series", completion.get("checkpoint_series")),
        ("completion.checkpoint", completion.get("checkpoint")),
        ("manifest.checkpoints", manifest.get("checkpoints")),
        ("manifest.checkpoint", manifest.get("checkpoint")),
        ("verification.checkpoints", verification.get("checkpoints")),
        ("verification.checkpoint", verification.get("checkpoint")),
    )
    for source_name, source in sources:
        if isinstance(source, Mapping):
            if any(field in source for field in _CHECKPOINT_PATH_FIELDS):
                source = [dict(source)]
            else:
                source = [
                    {"path": key, **(dict(value) if isinstance(value, Mapping) else {})}
                    for key, value in source.items()
                ]
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            continue
        for value in source:
            if isinstance(value, str):
                rows.append({"path": value, _EVIDENCE_SOURCE_KEY: source_name})
            elif isinstance(value, Mapping):
                rows.append({**dict(value), _EVIDENCE_SOURCE_KEY: source_name})
    if not rows:
        seen: set[Path] = set()
        for pattern in _CHECKPOINT_PATTERNS:
            for path in sorted((run_directory / "checkpoints").glob(pattern)) + sorted(run_directory.glob(pattern)):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    rows.append(
                        {
                            "path": str(resolved),
                            _EVIDENCE_SOURCE_KEY: "filesystem_discovery",
                        }
                    )

    grouped: dict[str, tuple[Path | None, list[dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        try:
            resolved = _resolve_checkpoint_path(run_directory, row)
        except (OSError, RuntimeError, ValueError):
            resolved = None
        if resolved is None:
            key = f"missing:{index}"
        else:
            raw_key = str(resolved)
            key = raw_key.casefold() if os.name == "nt" else raw_key
        if key not in grouped:
            grouped[key] = (resolved, [])
        grouped[key][1].append(row)

    aggregated: list[dict[str, Any]] = []
    for resolved, evidence_rows in grouped.values():
        aggregate: dict[str, Any] = {}
        for evidence in evidence_rows:
            for key, value in evidence.items():
                if key not in aggregate or aggregate[key] in (None, "", [], {}):
                    aggregate[key] = value
        if resolved is not None:
            aggregate["path"] = str(resolved)
        aggregate[_EVIDENCE_ROWS_KEY] = tuple(evidence_rows)
        aggregated.append(aggregate)
    return aggregated


def _evidence_rows(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = row.get(_EVIDENCE_ROWS_KEY)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        evidence = tuple(item for item in raw if isinstance(item, Mapping))
        if evidence:
            return evidence
    return (row,)


def _recursive_truthy(value: Any, keys: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in keys and child not in (None, False, "", [], {}):
                return True
            if _recursive_truthy(child, keys):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_recursive_truthy(child, keys) for child in value)
    return False


def _resolve_checkpoint_path(run_directory: Path, row: Mapping[str, Any]) -> Path | None:
    raw = _first(row.get("path"), row.get("checkpoint"), row.get("checkpoint_path"), row.get("file"))
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if ".." in path.parts:
        raise ValueError("checkpoint path cannot contain parent traversal")
    lexical = path if path.is_absolute() else run_directory / path
    return lexical.resolve()


def _checkpoint_lexical_path(run_directory: Path, row: Mapping[str, Any]) -> Path | None:
    raw = _first(row.get("path"), row.get("checkpoint"), row.get("checkpoint_path"), row.get("file"))
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if ".." in path.parts:
        return None
    return path if path.is_absolute() else run_directory / path


def _checkpoint_hash_evidence(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    hashes: set[str] = set()
    malformed = False
    for evidence in _evidence_rows(row):
        for field in ("sha256", "checkpoint_sha256", "file_sha256"):
            claim = evidence.get(field)
            if claim in (None, ""):
                continue
            if not isinstance(claim, str) or claim != claim.strip() or not _SHA256_PATTERN.fullmatch(claim.lower()):
                malformed = True
            else:
                hashes.add(claim.lower())
    if malformed:
        return None, "malformed"
    if len(hashes) > 1:
        return None, "conflict"
    return next(iter(hashes), None), None


def _checkpoint_path_safety_reason(
    row: Mapping[str, Any],
    path: Path,
    run_directory: Path,
) -> str | None:
    for evidence in _evidence_rows(row):
        lexical = _checkpoint_lexical_path(run_directory, evidence)
        if lexical is None:
            return "Checkpoint path crosses an unsafe link, mount, traversal, or hard-link seam."
        if not os.path.lexists(lexical):
            return "Checkpoint artifact is missing."
        if not _safe_regular_descendant(lexical, run_directory):
            return "Checkpoint path crosses an unsafe link, mount, traversal, or hard-link seam."
        try:
            if lexical.resolve(strict=True) != path.resolve(strict=True):
                return "Checkpoint path aliases do not identify one exact artifact."
        except OSError:
            return "Checkpoint artifact is missing."
    return None


def _step(row: Mapping[str, Any], path: Path | None, state: Mapping[str, Any]) -> int | None:
    value = _first(
        row.get("step"),
        row.get("checkpoint_step"),
        row.get("optimizer_step"),
        row.get("global_step"),
        _nested(state, "backend_identity", "checkpoint_step"),
        _nested(state, "backend_identity", "global_step"),
        state.get("checkpoint_step"),
        state.get("global_step"),
    )
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    match = _STEP_PATTERN.search(path.name if path else "")
    return int(match.group(1)) if match else None


def _weights(row: Mapping[str, Any], path: Path | None) -> str:
    raw = str(_first(row.get("weights"), row.get("variant"), row.get("checkpoint_variant"), default="")).lower()
    if row.get("ema_weights") is True or "ema" in raw or (path and "ema" in path.stem.lower()):
        return "ema"
    return "live"


def _identity_values(*values: Any) -> tuple[set[str], bool]:
    identities: set[str] = set()
    malformed = False
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str) or not value or value != value.strip():
            malformed = True
            continue
        identities.add(value)
    return identities, malformed


def _dataset_identity_values(state: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[set[str], bool]:
    row_values = tuple(
        evidence.get(field)
        for evidence in _evidence_rows(row)
        for field in ("dataset_identity", "training_dataset_identity")
    )
    return _identity_values(
        *row_values,
        _nested(state, "backend_identity", "dataset_identity"),
        _nested(state, "backend_identity", "dataset_hash"),
        _nested(state, "backend_identity", "dataset_identity_hash"),
        state.get("dataset_identity"),
        state.get("training_dataset_identity"),
        _nested(state, "dataset", "identity"),
    )


def _dataset_identity(state: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    values, malformed = _dataset_identity_values(state, row)
    return next(iter(values)) if not malformed and len(values) == 1 else None


def _dataset_summary(identity: str | None, state: Mapping[str, Any]) -> str:
    supplied = _first(
        _nested(state, "backend_identity", "dataset_identity_summary"),
        state.get("dataset_identity_summary"),
    )
    if supplied:
        return str(supplied)
    if not identity:
        return "Dataset identity unavailable"
    return f"Dataset {identity[:12]}{'…' if len(identity) > 12 else ''}"


def _view_identity_values(state: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[set[str], bool]:
    row_values = tuple(
        evidence.get(field)
        for evidence in _evidence_rows(row)
        for field in ("view_identity", "training_view_identity", "dataset_view_manifest_hash")
    )
    return _identity_values(
        *row_values,
        _nested(state, "backend_identity", "view_identity"),
        _nested(state, "backend_identity", "training_view_identity"),
        _nested(state, "backend_identity", "dataset_view_manifest_hash"),
        state.get("view_identity"),
        state.get("training_view_identity"),
        _nested(state, "dataset", "view_identity"),
    )


def _view_identity(state: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    values, malformed = _view_identity_values(state, row)
    return next(iter(values)) if not malformed and len(values) == 1 else None


def _view_summary(identity: str | None, state: Mapping[str, Any]) -> str:
    supplied = _first(
        _nested(state, "backend_identity", "view_identity_summary"),
        state.get("view_identity_summary"),
    )
    if supplied:
        return str(supplied)
    if not identity:
        return "Training view identity unavailable"
    return f"Training view {identity[:12]}{'…' if len(identity) > 12 else ''}"


def _candidate_id(run_id: str, step: int | None, weights: str, path: Path | None) -> str:
    material = f"{run_id}|{step}|{weights}|{path.name if path else 'missing'}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _verification(
    state: Mapping[str, Any],
    row: Mapping[str, Any],
    path: Path | None,
    *,
    run_directory: Path,
) -> tuple[str, list[str]]:
    if path is None:
        return "MISSING", ["Checkpoint artifact is missing."]
    path_reason = _checkpoint_path_safety_reason(row, path, run_directory)
    if path_reason:
        return ("MISSING" if "missing" in path_reason.lower() else "FAILED"), [path_reason]

    pass_states = {"PASS", "PASSED", "VERIFIED", "VALID"}
    fail_states = {
        "FAIL",
        "FAILED",
        "INVALID",
        "UNVERIFIED",
        "REVOKED",
        "NOT YET SAFE FOR RESUME",
    }
    explicit_verified = False
    explicit_failed = False
    malformed = False
    evidence_rows = _evidence_rows(row)

    state_verification = _nested(state, "backend_identity", "verification_state")
    verification_claims = [state_verification] if state_verification not in (None, "") else []
    for evidence in evidence_rows:
        verification_claims.extend(
            value
            for value in (evidence.get("verification_state"), evidence.get("verification"))
            if value not in (None, "")
        )
        if "verified" in evidence and evidence.get("verified") is not None:
            claim = evidence.get("verified")
            if not isinstance(claim, bool):
                malformed = True
            elif claim:
                explicit_verified = True
            else:
                explicit_failed = True

        for field in ("hash_verified", "safe_resume", "identity_verified"):
            if field not in evidence or evidence.get(field) is None:
                continue
            claim = evidence.get(field)
            if not isinstance(claim, bool):
                malformed = True
            elif not claim:
                explicit_failed = True

        remote = evidence.get("remote")
        if remote is not None and not isinstance(remote, bool):
            malformed = True
        if remote is True:
            for field in ("downloaded", "remote_identity_verified"):
                if field not in evidence or evidence.get(field) is None:
                    continue
                claim = evidence.get(field)
                if not isinstance(claim, bool):
                    malformed = True
                elif not claim:
                    explicit_failed = True

    for claim in verification_claims:
        if not isinstance(claim, str) or claim != claim.strip():
            malformed = True
            continue
        normalized = claim.upper()
        if normalized in pass_states:
            explicit_verified = True
        elif normalized in fail_states:
            explicit_failed = True
        else:
            malformed = True

    expected_hash, hash_error = _checkpoint_hash_evidence(row)
    if hash_error == "conflict":
        return "FAILED", ["Checkpoint SHA-256 evidence disagrees across durable sources."]
    if malformed or hash_error == "malformed":
        return "FAILED", ["Checkpoint verification or hash evidence is malformed or unsupported."]
    if expected_hash is None:
        return "UNVERIFIED", ["Checkpoint lacks a durable per-file SHA-256 binding."]
    try:
        if file_sha256(path) != expected_hash:
            return "FAILED", ["Checkpoint SHA-256 does not match verified run state."]
        explicit_verified = True
    except OSError:
        return "FAILED", ["Checkpoint artifact could not be read for verification."]
    if explicit_failed:
        return "FAILED", ["Checkpoint verification state is not passing across all durable sources."]
    if explicit_verified:
        return "VERIFIED", []
    return "UNVERIFIED", ["Checkpoint is not bound to a durable per-file SHA-256."]


def _availability(
    *,
    state: Mapping[str, Any],
    command: Mapping[str, Any],
    row: Mapping[str, Any],
    path: Path | None,
    run_directory: Path,
    project_root: Path,
    expected_dataset: str | None,
    expected_view: str | None,
) -> tuple[CheckpointAvailability, str, tuple[str, ...]]:
    reasons: list[str] = []
    schema_valid = state.get("schema_version") == RUN_STATE_SCHEMA
    status = str(state.get("status") or "UNKNOWN").upper()
    run_command = str(state.get("command") or command.get("command") or "").lower()
    if state.get("run_id") and str(state["run_id"]) != run_directory.name:
        return CheckpointAvailability.INVALID, "FAILED", ("Run identity does not match its durable directory.",)
    if not schema_valid:
        return CheckpointAvailability.INVALID, "FAILED", ("Run state schema is missing or unsupported.",)
    if command.get("_unsafe_artifact") is True:
        return CheckpointAvailability.INVALID, "FAILED", ("Run command evidence crosses an unsafe filesystem seam.",)
    if run_command not in {"train", "training", "training.start"}:
        return CheckpointAvailability.FOREIGN, "FOREIGN", ("Run is not a Sprite Lab training run.",)
    declared_root = command.get("project_root")
    if declared_root:
        try:
            same_root = Path(str(declared_root)).resolve() == project_root.resolve()
        except OSError:
            same_root = False
        if not same_root:
            return CheckpointAvailability.FOREIGN, "FOREIGN", ("Run belongs to a different project root.",)
    if path is not None and not _inside(path, run_directory):
        return CheckpointAvailability.FOREIGN, "FOREIGN", ("Checkpoint is outside its verified run directory.",)
    if path is not None and (path_reason := _checkpoint_path_safety_reason(row, path, run_directory)):
        if "missing" in path_reason.lower():
            return CheckpointAvailability.MISSING, "MISSING", (path_reason,)
        return CheckpointAvailability.INVALID, "FAILED", (path_reason,)
    if _recursive_truthy((state, row), frozenset({"unsafe_resume", "unsafe_resume_record", "unsafe_resume_requested"})):
        return (
            CheckpointAvailability.UNSAFE_RESUME,
            "REVOKED",
            ("Run contains an unsafe-resume revocation.",),
        )
    if status != "COMPLETE":
        return CheckpointAvailability.INCOMPLETE, "INCOMPLETE", (f"Training run state is {status}.",)
    verification = "UNVERIFIED"
    dataset_values, malformed_dataset = _dataset_identity_values(state, row)
    if malformed_dataset:
        return (
            CheckpointAvailability.INVALID,
            verification,
            ("Checkpoint dataset identity aliases are malformed.",),
        )
    if len(dataset_values) > 1:
        return (
            CheckpointAvailability.INVALID,
            verification,
            ("Checkpoint dataset identity aliases disagree.",),
        )
    identity = next(iter(dataset_values), None)
    if expected_dataset and identity != expected_dataset:
        return (
            CheckpointAvailability.STALE_DATASET,
            verification,
            ("Checkpoint dataset identity does not match the active dataset identity.",),
        )
    view_values, malformed_view = _view_identity_values(state, row)
    if malformed_view:
        return (
            CheckpointAvailability.INVALID,
            verification,
            ("Checkpoint training-view identity aliases are malformed.",),
        )
    if len(view_values) > 1:
        return (
            CheckpointAvailability.INVALID,
            verification,
            ("Checkpoint training-view identity aliases disagree.",),
        )
    view_identity = next(iter(view_values), None)
    if expected_view and view_identity != expected_view:
        return (
            CheckpointAvailability.STALE_VIEW,
            verification,
            ("Checkpoint training-view identity does not match the active training-view identity.",),
        )
    verification, verification_reasons = _verification(
        state,
        row,
        path,
        run_directory=run_directory,
    )
    reasons.extend(verification_reasons)
    if verification == "MISSING":
        return CheckpointAvailability.MISSING, verification, tuple(reasons)
    if verification == "FAILED":
        return CheckpointAvailability.INVALID, verification, tuple(reasons)
    if verification != "VERIFIED":
        return CheckpointAvailability.UNVERIFIED, verification, tuple(reasons)
    return CheckpointAvailability.ELIGIBLE, verification, ()


def discover_checkpoint_candidates(
    runs_directory: Path,
    *,
    project_root: Path,
    active_dataset_identity: str | None = None,
    active_view_identity: str | None = None,
) -> CheckpointCatalog:
    """Discover eligible checkpoints and preserve fail-closed reasons for advanced inspection."""

    eligible: list[CheckpointCandidate] = []
    unavailable: list[CheckpointCandidate] = []
    if not _safe_directory_descendant(runs_directory, project_root):
        return CheckpointCatalog((), (), None)
    for run_directory in sorted(
        (path for path in runs_directory.iterdir() if _safe_direct_run_directory(path, runs_directory)),
        key=lambda p: p.name,
    ):
        state_path = run_directory / "state.json"
        if not _safe_regular_descendant(state_path, run_directory):
            continue
        state = _safe_read_object(state_path, run_directory)
        command_path = run_directory / "command.json"
        command = _safe_read_object(command_path, run_directory)
        if os.path.lexists(command_path) and not command:
            command = {"_unsafe_artifact": True}
        run_id = str(state.get("run_id") or run_directory.name)
        rows = _checkpoint_rows(run_directory, state) or [{}]
        for row in rows:
            try:
                path = _resolve_checkpoint_path(run_directory, row)
            except (OSError, RuntimeError, ValueError):
                path = None
            step = _step(row, path, state)
            weights = _weights(row, path)
            identity = _dataset_identity(state, row)
            view_identity = _view_identity(state, row)
            availability, verification, reasons = _availability(
                state=state,
                command=command,
                row=row,
                path=path,
                run_directory=run_directory,
                project_root=project_root,
                expected_dataset=active_dataset_identity,
                expected_view=active_view_identity,
            )
            candidate = CheckpointCandidate(
                checkpoint_id=_candidate_id(run_id, step, weights, path),
                run_id=run_id,
                friendly_run_name=str(
                    _first(
                        _nested(state, "backend_identity", "friendly_run_name"),
                        state.get("friendly_run_name"),
                        state.get("title"),
                        run_id.replace("-", " "),
                    )
                ),
                date=str(_first(state.get("ended_at"), state.get("started_at"), default="")) or None,
                training_profile=str(
                    _first(
                        row.get("training_profile"),
                        _nested(state, "backend_identity", "training_profile"),
                        state.get("training_profile"),
                        default="Standard",
                    )
                ),
                completion_state=str(state.get("status") or "UNKNOWN").upper(),
                dataset_identity=identity,
                dataset_identity_summary=_dataset_summary(identity, state),
                view_identity=view_identity,
                view_identity_summary=_view_summary(view_identity, state),
                checkpoint_step=step,
                weights=weights,
                checkpoint_sha256=_checkpoint_hash_evidence(row)[0],
                verification_state=verification,
                availability=availability,
                unavailable_reasons=reasons,
                path=path,
                run_directory=run_directory.resolve(),
            )
            (eligible if candidate.eligible else unavailable).append(candidate)

    def sort_key(item: CheckpointCandidate) -> tuple[str, int, bool]:
        return item.date or "", item.checkpoint_step or -1, item.weights == "ema"

    eligible.sort(key=sort_key, reverse=True)
    unavailable.sort(key=sort_key, reverse=True)
    return CheckpointCatalog(tuple(eligible), tuple(unavailable), eligible[0].checkpoint_id if eligible else None)

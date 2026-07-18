"""Strict, discoverable Sprite Lab v3 project configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation

CONFIG_NAME = "spritelab.yaml"
SECTIONS: dict[str, set[str]] = {
    "project": {"name", "schema_version"},
    "paths": {"runs", "artifacts"},
    "dataset": {
        "raw_provenance_report",
        "raw_inventory",
        "extraction_report",
        "suitability_report",
        "view_manifest",
        "freeze_manifest",
    },
    "labeling": {
        "campaign_report",
        "audit_report",
        "audit_hashes",
        "audit_stage_source_commit",
        "review_queues",
        "hierarchical_enabled",
        "hierarchical_profile",
        "reference_cohort_size",
        "hierarchical_run_root",
        "hierarchical_report",
    },
    "training": {"audit_report", "audit_hashes", "campaign_config", "dataset_freeze"},
    "evaluation": {
        "checkpoint",
        "benchmark",
        "memorization_audit",
        "review_log",
        "promotion_decision",
        "candidate_evidence",
        "training_manifests",
        "dataset_identity",
        "training_view_identity",
    },
    "execution": {
        "allow_dataset_production_freeze",
        "allow_training",
        "allow_generation",
        "allow_promotion",
        "dataset_command",
        "training_command",
        "evaluation_command",
        "review_command",
    },
    "reporting": {"auto_open"},
    "ui": {"open_browser", "host", "port"},
    "providers": {"vision"},
    "compute": {"training"},
}


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "sprite-lab", "schema_version": 3},
    "paths": {"runs": "runs/v3", "artifacts": "artifacts"},
    "dataset": {
        "raw_provenance_report": "artifacts/dataset/raw_provenance_report.json",
        "raw_inventory": "artifacts/dataset/raw_inventory.jsonl",
        "extraction_report": "artifacts/dataset/extraction_report.json",
        "suitability_report": "artifacts/dataset/suitability_report.json",
        "view_manifest": "artifacts/dataset/view_manifest.json",
        "freeze_manifest": "",
    },
    "labeling": {
        "campaign_report": "artifacts/labeling/campaign_report.json",
        "audit_report": "artifacts/labeling/audit_report.json",
        "audit_hashes": "",
        "audit_stage_source_commit": "",
        "review_queues": [],
        "hierarchical_enabled": False,
        "hierarchical_profile": "fast_local",
        "reference_cohort_size": 400,
        "hierarchical_run_root": "runs/v3/hierarchical-labeling",
        "hierarchical_report": "",
    },
    "training": {
        "audit_report": "artifacts/training/audit_report.json",
        "audit_hashes": "artifacts/training/audit_hashes.json",
        "campaign_config": "",
        "dataset_freeze": "",
    },
    "evaluation": {
        "checkpoint": "",
        "benchmark": "",
        "memorization_audit": "artifacts/evaluation/memorization_audit.json",
        "review_log": "",
        "promotion_decision": "",
        "candidate_evidence": "",
        "training_manifests": [],
        "dataset_identity": "",
        "training_view_identity": "",
    },
    "execution": {
        "allow_dataset_production_freeze": False,
        "allow_training": False,
        "allow_generation": False,
        "allow_promotion": False,
        "dataset_command": [],
        "training_command": [],
        "evaluation_command": [],
        "review_command": [],
    },
    "reporting": {"auto_open": False},
    "ui": {"open_browser": True, "host": "127.0.0.1", "port": "auto"},
    "providers": {"vision": {"type": "auto"}},
    "compute": {"training": {"type": "local"}},
}

_SECRET_KEY_MARKERS = frozenset(
    {
        "api_key",
        "apikey",
        "auth_token",
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "password",
        "passwd",
        "secret",
        "token",
    }
)


class ConfigError(ValueError):
    """Invalid or missing v3 configuration."""


def configured_training_identities(values: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return the explicit v3 evaluation dataset/view binding, failing closed on malformed values."""

    evaluation = values.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None, None

    def identity(key: str) -> str | None:
        value = evaluation.get(key)
        return value if isinstance(value, str) and value and value == value.strip() else None

    return identity("dataset_identity"), identity("training_view_identity")


def _find_secret_key(value: Any, path: tuple[str, ...] = ()) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            current = (*path, str(key))
            if normalized in _SECRET_KEY_MARKERS or normalized.endswith(("_token", "_password", "_secret")):
                return ".".join(current)
            found = _find_secret_key(child, current)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_secret_key(child, (*path, str(index)))
            if found:
                return found
    return None


def _validate_project_relative_artifact_path(value: Any, *, key: str, allow_empty: bool) -> str:
    if not isinstance(value, str) or value != value.strip() or "\\" in value or "\x00" in value:
        raise ConfigError(f"{key} must be a canonical project-relative path.")
    if not value:
        if allow_empty:
            return value
        raise ConfigError(f"{key} must not be empty.")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ConfigError(f"{key} must not be absolute.")
    if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigError(f"{key} must be a canonical project-relative path without traversal.")
    return value


def _validate(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("Configuration must be a YAML mapping.")
    secret_path = _find_secret_key(data)
    if secret_path:
        raise ConfigError(
            f"Secrets are not allowed in persisted project configuration ({secret_path}). "
            "Supply credentials through a runtime-only provider mechanism."
        )
    unknown_sections = sorted(set(data) - set(SECTIONS))
    if unknown_sections:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(unknown_sections)}")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"Section '{section}' must be a mapping.")
        unknown_keys = sorted(set(values) - SECTIONS[section])
        if unknown_keys:
            raise ConfigError(f"Unknown key(s) in '{section}': {', '.join(unknown_keys)}")
        merged[section].update(values)
    if merged["project"]["schema_version"] != 3:
        raise ConfigError("project.schema_version must be 3.")
    for section, key, allow_empty in (
        ("dataset", "freeze_manifest", True),
        ("training", "dataset_freeze", True),
        ("training", "campaign_config", True),
        ("training", "audit_report", False),
        ("training", "audit_hashes", False),
    ):
        _validate_project_relative_artifact_path(
            merged[section][key],
            key=f"{section}.{key}",
            allow_empty=allow_empty,
        )
    dataset_freeze = merged["dataset"]["freeze_manifest"]
    training_freeze = merged["training"]["dataset_freeze"]
    if (dataset_freeze or training_freeze) and dataset_freeze != training_freeze:
        raise ConfigError(
            "dataset.freeze_manifest and training.dataset_freeze must name the same project-relative file."
        )
    for key in ("dataset_command", "training_command", "evaluation_command", "review_command"):
        value = merged["execution"][key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigError(f"execution.{key} must be a list of argument strings.")
    if not isinstance(merged["labeling"]["review_queues"], list):
        raise ConfigError("labeling.review_queues must be a list.")
    labeling = merged["labeling"]
    if not isinstance(labeling["hierarchical_enabled"], bool):
        raise ConfigError("labeling.hierarchical_enabled must be true or false.")
    if labeling["hierarchical_profile"] not in {"fast_local", "balanced", "high_quality"}:
        raise ConfigError("labeling.hierarchical_profile must be fast_local, balanced, or high_quality.")
    if (
        isinstance(labeling["reference_cohort_size"], bool)
        or not isinstance(labeling["reference_cohort_size"], int)
        or not 300 <= labeling["reference_cohort_size"] <= 500
    ):
        raise ConfigError("labeling.reference_cohort_size must be from 300 through 500.")
    for key in ("hierarchical_run_root", "hierarchical_report"):
        if not isinstance(labeling[key], str) or labeling[key] != labeling[key].strip():
            raise ConfigError(f"labeling.{key} must be a string without surrounding whitespace.")
    if not labeling["hierarchical_run_root"]:
        raise ConfigError("labeling.hierarchical_run_root must not be empty.")
    for key in ("dataset_identity", "training_view_identity"):
        identity = merged["evaluation"][key]
        if not isinstance(identity, str):
            raise ConfigError(f"evaluation.{key} must be a string.")
        if identity and identity != identity.strip():
            raise ConfigError(f"evaluation.{key} must not contain leading or trailing whitespace.")
    ui = merged["ui"]
    if not isinstance(ui["open_browser"], bool):
        raise ConfigError("ui.open_browser must be true or false.")
    if not isinstance(ui["host"], str) or not ui["host"].strip():
        raise ConfigError("ui.host must be a non-empty host string.")
    port = ui["port"]
    if port != "auto" and (not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535):
        raise ConfigError("ui.port must be 'auto' or an integer from 1 through 65535.")
    for section, key in (("providers", "vision"), ("compute", "training")):
        settings = merged[section][key]
        if not isinstance(settings, dict):
            raise ConfigError(f"{section}.{key} must be a mapping.")
        if not isinstance(settings.get("type"), str) or not settings["type"].strip():
            raise ConfigError(f"{section}.{key}.type must be a non-empty string.")
    return merged


def discover_config(start: Path | None = None) -> Path | None:
    override = os.environ.get("SPRITELAB_CONFIG")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"SPRITELAB_CONFIG does not name a file: {candidate}")
        return candidate.resolve()
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
        # A repository is a project boundary. Continuing above it can make an
        # unrelated user-level configuration redirect this project's writes.
        if (directory / ".git").exists():
            break
    return None


def _safe_child_bytes(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
    allow_retained_stage: bool,
) -> bytes:
    """Read one exact anchored child, optionally binding its retained POSIX stage."""

    before = anchor.lstat(name)
    if not stat.S_ISREG(before.st_mode) or _config_metadata_is_link_or_reparse(before) or before.st_size > max_bytes:
        raise ConfigError("A project configuration commit artifact is unsafe.")
    if allow_retained_stage:
        alias = _config_retained_stage_alias(anchor, name, before)
    elif before.st_nlink == 1:
        alias = None
    else:
        raise ConfigError("A project configuration commit artifact is unsafe.")
    descriptor = anchor.open_file_immovable(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != before.st_nlink
        ):
            raise ConfigError("A project configuration commit artifact changed while opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = anchor.lstat(name)
    if (
        len(content) != before.st_size
        or len(content) > max_bytes
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_nlink != before.st_nlink
        or opened_after.st_dev != before.st_dev
        or opened_after.st_ino != before.st_ino
        or opened_after.st_size != before.st_size
        or opened_after.st_mtime_ns != before.st_mtime_ns
        or opened_after.st_nlink != before.st_nlink
    ):
        raise ConfigError("A project configuration commit artifact changed while read.")
    if allow_retained_stage and _config_retained_stage_alias(anchor, name, after) != alias:
        raise ConfigError("A retained project configuration commit stage changed.")
    return content


def _config_retained_stage_alias(
    anchor: AnchoredDirectory,
    name: str,
    metadata: os.stat_result,
) -> str | None:
    prefix = f".{name}.staging-"
    candidates = [candidate for candidate in anchor.names() if candidate.startswith(prefix)]
    if metadata.st_nlink == 1:
        if candidates:
            raise ConfigError("A single-link project commit artifact has retained-stage residue.")
        return None
    if (
        metadata.st_nlink != 2
        or len(candidates) != 1
        or re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", candidates[0]) is None
    ):
        raise ConfigError("A project configuration commit artifact lost its exact retained publication stage.")
    candidate = candidates[0]
    alias = anchor.lstat(candidate)
    if (
        not stat.S_ISREG(alias.st_mode)
        or _config_metadata_is_link_or_reparse(alias)
        or alias.st_dev != metadata.st_dev
        or alias.st_ino != metadata.st_ino
        or alias.st_nlink != 2
        or alias.st_size != metadata.st_size
        or alias.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise ConfigError("A retained project configuration commit stage does not bind the exact target inode.")
    return candidate


def _config_metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _strict_json_mapping(content: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ConfigError("A project activation commit document has duplicate keys.")
            value[key] = child
        return value

    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError("A project activation commit document is invalid.") from exc
    if not isinstance(value, dict):
        raise ConfigError("A project activation commit document must be an object.")
    return value


def _activation_commit_effective_config(root: Path, canonical: bytes) -> bytes:
    """Resolve an immutable conditioned activation marker without mutating config."""

    from spritelab.product_features.training.activation_commit import (
        ACTIVATION_PROJECT_COMMIT_NAME,
        ActivationCommitError,
        canonical_activation_commit_bytes,
        validate_activation_project_commit,
    )

    try:
        try:
            root.lstat()
        except FileNotFoundError:
            # An explicitly selected project root may be prospective. With no
            # directory there cannot yet be an activation marker to apply, and
            # passive configuration loading must not create the root.
            return canonical
        with AnchoredDirectory(root, root) as root_anchor:
            if not root_anchor.lexists(ACTIVATION_PROJECT_COMMIT_NAME):
                return canonical
            marker_bytes = _safe_child_bytes(
                root_anchor,
                ACTIVATION_PROJECT_COMMIT_NAME,
                max_bytes=32 * 1024 * 1024,
                allow_retained_stage=True,
            )
            marker = _strict_json_mapping(marker_bytes)
            if canonical_activation_commit_bytes(marker) != marker_bytes:
                raise ConfigError("The project activation commit marker is not canonical.")
            job_id = marker.get("job_id")
            if not isinstance(job_id, str) or re.fullmatch(r"conditioned-[0-9a-f]{20}", job_id) is None:
                raise ConfigError("The project activation commit marker has an invalid job binding.")
            with ExitStack() as stack:
                current = root_anchor
                for part in ("runs", "v3", "conditioned-dataset-v5", job_id, "activation_receipt"):
                    current = stack.enter_context(current.open_directory_immovable(part))
                documents: dict[str, dict[str, Any]] = {}
                for name in ("receipt.json", "journal.json", "record.json"):
                    content = _safe_child_bytes(
                        current,
                        name,
                        max_bytes=1024 * 1024,
                        allow_retained_stage=True,
                    )
                    document = _strict_json_mapping(content)
                    if canonical_activation_commit_bytes(document) != content:
                        raise ConfigError("A project activation commit document is not canonical.")
                    documents[name] = document
            _summary, effective = validate_activation_project_commit(
                marker,
                receipt=documents["receipt.json"],
                journal=documents["journal.json"],
                record=documents["record.json"],
                current_config_sha256=hashlib.sha256(canonical).hexdigest(),
                expected_job_id=job_id,
            )
            return effective
    except (OSError, UnsafeFilesystemOperation, ActivationCommitError) as exc:
        raise ConfigError("The immutable project activation commit is unavailable or invalid.") from exc


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    path: Path | None
    values: dict[str, Any]

    @classmethod
    def load(cls, start: Path | None = None, *, required: bool = True) -> ProjectConfig:
        path = discover_config(start)
        if path is None:
            if required:
                raise ConfigError(f"No {CONFIG_NAME} found in this directory or its parents. Run v3 init.")
            root_override = os.environ.get("SPRITELAB_PROJECT_ROOT")
            root = Path(root_override).expanduser().resolve() if root_override else (start or Path.cwd()).resolve()
            return cls(root=root, path=None, values=copy.deepcopy(DEFAULT_CONFIG))
        root_override = os.environ.get("SPRITELAB_PROJECT_ROOT")
        root = Path(root_override).expanduser().resolve() if root_override else path.parent.resolve()
        try:
            with AnchoredDirectory(path.parent, path.parent) as config_anchor:
                canonical = _safe_child_bytes(
                    config_anchor,
                    path.name,
                    max_bytes=16 * 1024 * 1024,
                    allow_retained_stage=False,
                )
            effective = _activation_commit_effective_config(root, canonical)
            raw = yaml.safe_load(effective.decode("utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ConfigError(f"Could not read {path}: {exc}") from exc
        values = _validate(raw)
        runs_override = os.environ.get("SPRITELAB_RUNS_DIR")
        if runs_override:
            values["paths"]["runs"] = runs_override
        return cls(root=root, path=path, values=values)

    @property
    def name(self) -> str:
        return str(self.values["project"]["name"])

    def path_for(self, section: str, key: str) -> Path | None:
        raw = self.values[section].get(key)
        return self.path_for_value(raw)

    def path_for_value(self, raw: Any) -> Path | None:
        if not raw:
            return None
        candidate = Path(str(raw)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    @property
    def runs_dir(self) -> Path:
        raw = Path(str(self.values["paths"]["runs"])).expanduser()
        return raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()


def template_text() -> str:
    header = "# Sprite Lab v3 project configuration. Production actions are disabled by default.\n"
    return header + yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=True)

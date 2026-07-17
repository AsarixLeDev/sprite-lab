"""Strict, discoverable Sprite Lab v3 project configuration."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

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
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Could not read {path}: {exc}") from exc
        values = _validate(raw)
        root_override = os.environ.get("SPRITELAB_PROJECT_ROOT")
        root = Path(root_override).expanduser().resolve() if root_override else path.parent.resolve()
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

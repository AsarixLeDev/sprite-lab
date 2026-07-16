"""Atomic, project-scoped persistence for non-secret product settings."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core.contracts import ProjectContext

PRODUCT_SETTINGS_SCHEMA = "spritelab.product.settings.v1"
PRODUCT_SETTINGS_DIRECTORY = ".spritelab"
PRODUCT_SETTINGS_FILENAME = "product-settings.json"
SETTING_PATHS = {
    "provider": ("providers", "vision"),
    "compute": ("compute", "training"),
}
SECRET_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:api_?key|access_?token|auth_?token|bearer|password|passwd|private_?key|secret|token)(?:$|_)",
    re.IGNORECASE,
)
BEARER_VALUE_PATTERN = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/-]+", re.IGNORECASE)

_LOCK_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class ProductSettingsError(ValueError):
    """A product setting was unsafe, malformed, or could not be persisted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_id(root: Path) -> str:
    canonical = os.path.normcase(str(root.resolve())).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _path_lock(path: Path) -> threading.RLock:
    # ``Path.resolve`` may begin returning the Windows extended ``\\?\``
    # spelling after a parent directory is created. Use one stable absolute
    # spelling so repositories constructed on either side of that transition
    # still share the same process lock.
    absolute = os.path.abspath(os.fspath(path))
    if absolute.startswith("\\\\?\\UNC\\"):
        absolute = "\\\\" + absolute[8:]
    elif absolute.startswith("\\\\?\\"):
        absolute = absolute[4:]
    key = os.path.normcase(absolute)
    with _LOCK_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def reject_secret_settings(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject secret-bearing keys and obvious bearer material recursively."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            current = (*path, str(key))
            if SECRET_KEY_PATTERN.search(normalized) or normalized == "authorization":
                raise ProductSettingsError(
                    f"Secrets are not allowed in saved product settings ({'.'.join(current)}). "
                    "Use a credential environment-variable or credential-store reference."
                )
            reject_secret_settings(child, current)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_secret_settings(child, (*path, str(index)))
    elif isinstance(value, str) and BEARER_VALUE_PATTERN.search(value):
        raise ProductSettingsError(
            f"Bearer credentials are not allowed in saved product settings ({'.'.join(path) or 'value'})."
        )


class ProductSettingsRepository:
    """Persist one safe settings document beneath the selected project root."""

    def __init__(self, context: ProjectContext, *, path: Path | None = None) -> None:
        self.context = context
        self.project_root = context.project_root.resolve()
        self.project_id = _project_id(self.project_root)
        self.path = path or self.project_root / PRODUCT_SETTINGS_DIRECTORY / PRODUCT_SETTINGS_FILENAME
        self._lock = _path_lock(self.path)

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCT_SETTINGS_SCHEMA,
            "project_id": self.project_id,
            "updated_at": None,
            "sections": {},
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            if self.path.stat().st_size > 1_000_000:
                raise ProductSettingsError("Product settings file exceeds the safe size limit.")
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductSettingsError("Saved product settings are malformed and were not loaded.") from exc
        if not isinstance(value, dict) or value.get("schema_version") != PRODUCT_SETTINGS_SCHEMA:
            raise ProductSettingsError("Saved product settings use an unsupported schema.")
        if value.get("project_id") != self.project_id:
            raise ProductSettingsError("Saved product settings belong to another project.")
        if not isinstance(value.get("sections"), dict):
            raise ProductSettingsError("Saved product settings sections are malformed.")
        reject_secret_settings(value)
        return value

    def _write(self, document: Mapping[str, Any]) -> None:
        reject_secret_settings(document)
        payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    # Antivirus/indexer sharing locks can briefly block an
                    # otherwise atomic replacement on Windows.
                    time.sleep(0.01 * (2**attempt))
        finally:
            temporary.unlink(missing_ok=True)

    def section(self, name: str) -> dict[str, Any] | None:
        if name not in SETTING_PATHS:
            raise ProductSettingsError(f"Unknown product settings section: {name}")
        with self._lock:
            document = self._read()
            raw = document["sections"].get(name)
            return copy.deepcopy(raw) if isinstance(raw, dict) else None

    def effective_settings(self, name: str) -> tuple[dict[str, Any], int, bool]:
        saved = self.section(name)
        if saved is not None and isinstance(saved.get("settings"), Mapping):
            return dict(saved["settings"]), int(saved.get("configuration_version", 1)), True
        first, second = SETTING_PATHS[name]
        parent = self.context.config.get(first, {}) if isinstance(self.context.config, Mapping) else {}
        fallback = parent.get(second, {}) if isinstance(parent, Mapping) else {}
        return dict(fallback) if isinstance(fallback, Mapping) else {}, 0, False

    def save(self, name: str, settings: Mapping[str, Any]) -> dict[str, Any]:
        if name not in SETTING_PATHS:
            raise ProductSettingsError(f"Unknown product settings section: {name}")
        clean = copy.deepcopy(dict(settings))
        reject_secret_settings(clean)
        try:
            json.dumps(clean, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ProductSettingsError("Product settings must contain JSON-safe finite values.") from exc
        with self._lock:
            document = self._read()
            previous = document["sections"].get(name, {})
            version = int(previous.get("configuration_version", 0)) + 1 if isinstance(previous, Mapping) else 1
            section = {
                "configuration_version": version,
                "saved_at": _now(),
                "settings": clean,
            }
            document["sections"][name] = section
            document["updated_at"] = _now()
            self._write(document)
            return copy.deepcopy(section)

    def clear(self, name: str) -> bool:
        if name not in SETTING_PATHS:
            raise ProductSettingsError(f"Unknown product settings section: {name}")
        with self._lock:
            document = self._read()
            existed = name in document["sections"]
            document["sections"].pop(name, None)
            document["updated_at"] = _now()
            self._write(document)
            return existed

    def record_observation(self, name: str, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Cache an explicit probe result without treating it as timeless state."""

        clean = copy.deepcopy(dict(observation))
        reject_secret_settings(clean)
        with self._lock:
            document = self._read()
            section = document["sections"].get(name)
            if not isinstance(section, dict):
                settings, version, _saved = self.effective_settings(name)
                section = {
                    "configuration_version": version,
                    "saved_at": None,
                    "settings": settings,
                }
            stamped = {**clean, "observed_at": _now()}
            section["observation"] = stamped
            document["sections"][name] = section
            document["updated_at"] = _now()
            self._write(document)
            return copy.deepcopy(stamped)

    def effective_context(self) -> ProjectContext:
        values = copy.deepcopy(dict(self.context.config))
        for name, (first, second) in SETTING_PATHS.items():
            settings, _version, saved = self.effective_settings(name)
            if saved:
                parent = values.setdefault(first, {})
                if not isinstance(parent, dict):
                    parent = {}
                    values[first] = parent
                parent[second] = settings
        return ProjectContext(
            project_root=self.context.project_root,
            config=values,
            config_path=self.context.config_path,
            runs_directory=self.context.runs_directory,
        )


__all__ = [
    "PRODUCT_SETTINGS_FILENAME",
    "PRODUCT_SETTINGS_SCHEMA",
    "ProductSettingsError",
    "ProductSettingsRepository",
    "reject_secret_settings",
]

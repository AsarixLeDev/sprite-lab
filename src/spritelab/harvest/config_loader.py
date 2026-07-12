"""Safe, cached single-source loading for Labeling v2 configuration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

LABELING_CONFIG_SCHEMA_VERSION = "labeling_v2.1"
_CONFIG_NAMES = frozenset({"source_profiles", "hallucination_denylist", "taxonomy", "sheet_mappings"})
_REPOSITORY_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class LabelingConfigError(ValueError):
    """Raised for any invalid active Labeling v2 configuration."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise LabelingConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def override_config_dir() -> Path | None:
    """Return the explicit config override dir, without consulting cwd."""

    raw = os.environ.get("SPRITELAB_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _REPOSITORY_CONFIG_DIR if _REPOSITORY_CONFIG_DIR.is_dir() else None


@cache
def _load_config(name: str) -> tuple[dict[str, Any], str, str]:
    if name not in _CONFIG_NAMES:
        raise LabelingConfigError(f"unknown Labeling v2 config: {name}")
    override_dir = override_config_dir()
    override_path = override_dir / f"{name}.yaml" if override_dir is not None else None
    if override_path is not None and override_path.is_file():
        return (
            _parse_yaml(override_path.read_text(encoding="utf-8"), source=str(override_path)),
            str(override_path),
            "override",
        )
    if override_path is not None and override_path.exists():
        raise LabelingConfigError(f"invalid Labeling v2 override config path: {override_path}")
    package_path = resources.files("spritelab.config").joinpath(f"{name}.yaml")
    try:
        return (
            _parse_yaml(package_path.read_text(encoding="utf-8"), source=f"package:{name}.yaml"),
            f"package:{name}.yaml",
            "packaged",
        )
    except FileNotFoundError as exc:  # pragma: no cover - package integrity guard
        raise LabelingConfigError(f"missing packaged Labeling v2 config: {name}.yaml") from exc


def _parse_yaml(text: str, *, source: str) -> dict[str, Any]:
    try:
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, LabelingConfigError) as exc:
        raise LabelingConfigError(f"invalid Labeling v2 config '{source}': {exc}") from exc
    if not isinstance(data, Mapping):
        raise LabelingConfigError(f"invalid Labeling v2 config '{source}': expected an object")
    result = dict(data)
    if result.get("schema_version") != LABELING_CONFIG_SCHEMA_VERSION:
        raise LabelingConfigError(
            f"invalid Labeling v2 config '{source}': schema_version must be {LABELING_CONFIG_SCHEMA_VERSION!r}"
        )
    return result


def _load(name: str, fallback: Mapping[str, Any]) -> dict[str, Any]:
    # ``fallback`` is retained for source compatibility and explicit fallback
    # semantics if packaged defaults are intentionally unavailable in a source
    # checkout. A present malformed config always raises above.
    try:
        config, _, _ = _load_config(name)
    except LabelingConfigError:
        raise
    return _copy(config) if config else _copy(fallback)


def load_source_profiles_config(fallback: Mapping[str, Any]) -> dict[str, Any]:
    return _load("source_profiles", fallback)


def load_hallucination_denylist_config(fallback: Mapping[str, Any]) -> dict[str, Any]:
    return _load("hallucination_denylist", fallback)


def load_taxonomy_config(fallback: Mapping[str, Any]) -> dict[str, Any]:
    return _load("taxonomy", fallback)


def load_sheet_mappings_config(fallback: Mapping[str, Any]) -> dict[str, Any]:
    return _load("sheet_mappings", fallback)


def labeling_config_metadata() -> dict[str, Any]:
    """Stable schema/version/hash metadata for exports and audit reports."""

    configs: dict[str, dict[str, str]] = {}
    for name in sorted(_CONFIG_NAMES):
        config, source, source_kind = _load_config(name)
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        configs[name] = {
            "schema_version": str(config["schema_version"]),
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "source": source,
            "source_kind": source_kind,
        }
    return {"labeling_config_schema_version": LABELING_CONFIG_SCHEMA_VERSION, "configs": configs}


def clear_config_cache() -> None:
    """Test-only helper for override-location scenarios."""

    _load_config.cache_clear()


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value)))

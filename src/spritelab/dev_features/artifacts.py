"""Read-only artifact identity and reference inspection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import ProjectState

_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_KEYS = ("sha256", "sha256_before", "content_sha256", "artifact_sha256", "file_sha256", "hash")
_PATH_KEYS = ("path", "file", "artifact_path", "report_path", "manifest_path")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kind(path: str) -> str:
    lowered = Path(path).name.lower()
    if "hash" in lowered or "verification" in lowered:
        return "hashes"
    if "manifest" in lowered or lowered.endswith(".jsonl"):
        return "manifest"
    if "report" in lowered or "summary" in lowered:
        return "report"
    return "artifact"


def _configured_references(config: ProjectConfig) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for section in ("dataset", "labeling", "training", "evaluation"):
        for key, raw in config.values.get(section, {}).items():
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                references.append(
                    {
                        "reference": value,
                        "expected_sha256": None,
                        "source": f"configuration:{section}.{key}",
                        "kind": _kind(value),
                    }
                )
    return references


def _evidence_references(state: ProjectState) -> list[dict[str, Any]]:
    return [
        {
            "reference": evidence.path,
            "expected_sha256": evidence.sha256,
            "source": f"project-state:{stage.key}",
            "kind": _kind(evidence.path),
        }
        for stage in state.stages
        for evidence in stage.evidence
    ]


def _json_hash_references(value: Any, *, source: Path) -> Iterable[dict[str, Any]]:
    if isinstance(value, Mapping):
        candidate = next((value.get(key) for key in _PATH_KEYS if isinstance(value.get(key), str)), None)
        expected = next(
            (
                str(value[key]).lower()
                for key in _HASH_KEYS
                if isinstance(value.get(key), str) and _HEX_SHA256.fullmatch(str(value[key]))
            ),
            None,
        )
        if candidate and expected:
            yield {
                "reference": candidate,
                "expected_sha256": expected,
                "source": str(source),
                "kind": _kind(str(candidate)),
            }
        hashes = value.get("artifact_hashes")
        if isinstance(hashes, Mapping):
            for path, digest in hashes.items():
                if isinstance(path, str) and isinstance(digest, str) and _HEX_SHA256.fullmatch(digest):
                    yield {
                        "reference": path,
                        "expected_sha256": digest.lower(),
                        "source": str(source),
                        "kind": _kind(path),
                    }
        for child in value.values():
            yield from _json_hash_references(child, source=source)
    elif isinstance(value, list):
        for child in value:
            yield from _json_hash_references(child, source=source)


def _resolve_reference(root: Path, reference: str) -> tuple[Path | None, str | None]:
    if "\x00" in reference:
        return None, "Reference contains a NUL character."
    try:
        raw = Path(reference).expanduser()
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not raw.is_absolute() and path != root and root not in path.parents:
            return None, "Relative reference escapes the project root."
        return path, None
    except (OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)


def inspect_artifacts(config: ProjectConfig, state: ProjectState) -> list[dict[str, Any]]:
    """Inspect configured artifacts and all recognizable bound hash references."""

    references = [*_configured_references(config), *_evidence_references(state)]
    inspected_sources: set[Path] = set()
    for item in list(references):
        path, error = _resolve_reference(config.root, str(item["reference"]))
        if error or path is None or path in inspected_sources or path.suffix.lower() != ".json" or not path.is_file():
            continue
        inspected_sources.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        references.extend(_json_hash_references(payload, source=path))

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str]] = set()
    for item in references:
        reference = str(item["reference"])
        expected = item.get("expected_sha256")
        source = str(item["source"])
        marker = (reference, str(expected) if expected else None, source)
        if marker in seen:
            continue
        seen.add(marker)
        path, error = _resolve_reference(config.root, reference)
        exists = bool(path and path.exists())
        is_file = bool(path and path.is_file())
        actual = sha256_file(path) if is_file and path else None
        if error:
            status = "INVALID_REFERENCE"
        elif not exists:
            status = "MISSING"
        elif expected and not is_file:
            status = "INVALID_REFERENCE"
            error = "Hash-bound reference is not a file."
        elif expected and actual != expected:
            status = "HASH_MISMATCH"
        elif expected:
            status = "CURRENT"
        else:
            status = "PRESENT"
        results.append(
            {
                "reference": reference,
                "path": str(path) if path else None,
                "kind": item["kind"],
                "source": source,
                "exists": exists,
                "is_file": is_file,
                "sha256": actual,
                "expected_sha256": expected,
                "identity_status": status,
                "stale": status == "HASH_MISMATCH",
                "valid_reference": error is None,
                "error": error,
            }
        )
    return sorted(results, key=lambda item: (item["identity_status"], item["reference"], item["source"]))

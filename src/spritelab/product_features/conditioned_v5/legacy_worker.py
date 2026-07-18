"""Controlled child entry point for pathname-oriented Dataset intake code.

This file is executed directly, not with ``python -m``. Its top-level imports
are deliberately limited to the standard library and a direct load of the
audited write-confinement helper; importing the ``spritelab`` package and every
pathname-oriented legacy module happens only after the platform boundary is
established.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_write_confinement() -> Any:
    module = sys.modules.get("_spritelab_conditioned_write_confinement")
    if module is None:
        raise RuntimeError("audited write-confinement helper was not preloaded")
    return module


_WRITE_CONFINEMENT = _load_write_confinement()
LINUX_LANDLOCK_STRATEGY = _WRITE_CONFINEMENT.LINUX_LANDLOCK_STRATEGY
WINDOWS_PARENT_ANCHORS_STRATEGY = _WRITE_CONFINEMENT.WINDOWS_PARENT_ANCHORS_STRATEGY
WriteConfinementError = _WRITE_CONFINEMENT.WriteConfinementError
WriteConfinementUnavailable = _WRITE_CONFINEMENT.WriteConfinementUnavailable
enforce_linux_landlock_write_confinement = _WRITE_CONFINEMENT.enforce_linux_landlock_write_confinement
windows_current_process_confinement_evidence = _WRITE_CONFINEMENT.windows_current_process_confinement_evidence

REQUEST_SCHEMA = "spritelab.dataset.conditioned-legacy-intake-request.v1"
RESPONSE_SCHEMA = "spritelab.dataset.conditioned-legacy-intake-response.v1"
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "source",
        "license",
        "artifact_sha256",
    }
)


def _response(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _failure(error_code: str) -> int:
    _response(
        {
            "schema_version": RESPONSE_SCHEMA,
            "ok": False,
            "error_code": error_code,
            "paths_exposed": False,
        }
    )
    return 1


def _request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(payload) > _MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, Mapping) or set(value) != _REQUEST_KEYS or value.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("request schema mismatch")
    if not isinstance(value.get("source"), Mapping) or not isinstance(value.get("license"), Mapping):
        raise ValueError("request evidence mismatch")
    artifact_sha256 = value.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or any(
        not isinstance(key, str) or not isinstance(digest, str) for key, digest in artifact_sha256.items()
    ):
        raise ValueError("request artifact mismatch")
    return dict(value)


def _windows_evidence(workspace: Path, expected_device: int, expected_inode: int) -> dict[str, Any]:
    return windows_current_process_confinement_evidence(
        workspace,
        expected_device=expected_device,
        expected_inode=expected_inode,
    ).to_dict()


def _enable_runtime_import_roots() -> None:
    """Append parent-bound dependency roots without executing ``.pth`` files."""

    roots = getattr(sys, "_spritelab_conditioned_runtime_roots", None)
    if not isinstance(roots, tuple) or not roots:
        raise RuntimeError("runtime dependency roots unavailable")
    for raw in roots:
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RuntimeError("runtime dependency root binding invalid")
        candidate = Path(str(raw[0]))
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise RuntimeError("runtime dependency root unavailable") from exc
        reparse = getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            not candidate.is_absolute()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or reparse
            or metadata.st_dev != int(raw[1])
            or metadata.st_ino != int(raw[2])
        ):
            raise RuntimeError("runtime dependency root is unsafe")
        value = os.fspath(candidate)
        if value not in sys.path:
            sys.path.append(value)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        return _failure("legacy_worker_arguments")
    strategy = argv[1]
    workspace = Path(os.path.abspath(argv[2]))
    try:
        expected_device = int(argv[3])
        expected_inode = int(argv[4])
        if strategy == LINUX_LANDLOCK_STRATEGY:
            evidence = enforce_linux_landlock_write_confinement(
                workspace,
                expected_device=expected_device,
                expected_inode=expected_inode,
            ).to_dict()
        elif strategy == WINDOWS_PARENT_ANCHORS_STRATEGY and sys.platform == "win32":
            evidence = _windows_evidence(workspace, expected_device, expected_inode)
        else:
            raise WriteConfinementUnavailable("unsupported child strategy")

        request = _request()
        _enable_runtime_import_roots()
        # Importing this module imports every pathname-oriented legacy writer,
        # so it must remain below the irreversible Linux restriction above.
        from spritelab.product_features.conditioned_v5.intake import _run_legacy_intake_in_process

        result = _run_legacy_intake_in_process(
            work=workspace,
            source_root=workspace / "source",
            output_root=workspace / "datasets" / "managed",
            source=dict(request["source"]),
            license_record=dict(request["license"]),
            artifact_sha256={str(key): str(value) for key, value in dict(request["artifact_sha256"]).items()},
            run_id=str(request["run_id"]),
        )
        _response(
            {
                "schema_version": RESPONSE_SCHEMA,
                "ok": True,
                "result": result,
                "write_confinement": evidence,
                "paths_exposed": False,
            }
        )
        return 0
    except WriteConfinementUnavailable:
        return _failure("write_confinement_unavailable")
    except WriteConfinementError:
        return _failure("write_confinement_failed")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return _failure("legacy_intake_failed")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

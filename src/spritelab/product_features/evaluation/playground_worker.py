"""Contained subprocess entry point for one local Playground sampling request."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery as importlib_machinery
import importlib.util as importlib_util
import json
import math
import operator
import os
import stat
import sys
import time
import types
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, SupportsIndex

_MAX_CONTROL_BYTES = 128 * 1024 * 1024
_MAX_PROJECT_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_BOUND_WORKER_BYTES = 2 * 1024 * 1024
_CONTROL_SCHEMA = "spritelab.playground-sampler-control.v2"
_RESULT_SCHEMA = "spritelab.playground-sampler-result.v2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--checkpoint-fd", type=int)
    parser.add_argument("--workspace-fd", type=int)
    parser.add_argument("--prompts-fd", type=int)
    parser.add_argument("--worker-fd", type=int, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--worker-size", type=int, required=True)
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _record_identity(value: Mapping[str, Any], identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    return _canonical_sha256(body)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _strict_json_loads(payload: bytes) -> Any:
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


def _parse_control_deadline(value: Any) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise RuntimeError("contained Playground deadline is malformed")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError("contained Playground deadline is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.isoformat() != value:
        raise RuntimeError("contained Playground deadline is malformed")
    return parsed


def _deadline_operation(deadline: datetime) -> Callable[[], None]:
    remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
    monotonic_deadline = time.monotonic() + remaining

    def operation_check() -> None:
        if datetime.now(timezone.utc) >= deadline or time.monotonic() >= monotonic_deadline:
            raise RuntimeError("contained Playground deadline expired")

    operation_check()
    return operation_check


def _validate_control(
    value: Any,
    *,
    workspace: Path,
    parsed: argparse.Namespace,
) -> tuple[dict[str, Any], datetime]:
    expected_keys = {
        "schema_version",
        "control_identity",
        "bootstrap_identity",
        "worker_sha256",
        "worker_size",
        "checkpoint",
        "checkpoint_sha256",
        "prompts",
        "prompts_sha256",
        "output",
        "report",
        "seed",
        "sampling_steps",
        "guidance",
        "image_count",
        "expected_step",
        "expected_variant",
        "deadline_at",
        "code_inventory",
        "code_inventory_identity",
        "runtime_closure",
        "runtime_closure_identity",
        "workspace_identity",
    }
    if not isinstance(value, dict) or set(value) != expected_keys or value.get("schema_version") != _CONTROL_SCHEMA:
        raise RuntimeError("contained Playground control schema is malformed")
    if (
        not _is_sha256(value.get("control_identity"))
        or value["control_identity"] != _record_identity(value, "control_identity")
        or not _is_sha256(value.get("bootstrap_identity"))
        or value["bootstrap_identity"] != parsed.bootstrap_sha256
        or not _is_sha256(value.get("worker_sha256"))
        or value["worker_sha256"] != parsed.worker_sha256
        or type(value.get("worker_size")) is not int
        or value["worker_size"] != parsed.worker_size
        or not 1 <= value["worker_size"] <= _MAX_BOUND_WORKER_BYTES
        or not _is_sha256(value.get("checkpoint_sha256"))
        or not _is_sha256(value.get("prompts_sha256"))
        or not _is_sha256(value.get("code_inventory_identity"))
        or value["code_inventory_identity"] != _canonical_sha256(value.get("code_inventory"))
        or not isinstance(value.get("runtime_closure"), dict)
        or not _is_sha256(value.get("runtime_closure_identity"))
        or value["runtime_closure"].get("runtime_closure_identity") != value["runtime_closure_identity"]
    ):
        raise RuntimeError("contained Playground control identity is malformed")
    workspace_identity = value.get("workspace_identity")
    if (
        not isinstance(workspace_identity, dict)
        or set(workspace_identity) != {"device", "inode"}
        or any(
            type(workspace_identity.get(key)) is not int or workspace_identity[key] < 0 for key in ("device", "inode")
        )
    ):
        raise RuntimeError("contained Playground workspace identity is malformed")
    if (
        type(value.get("seed")) is not int
        or not 0 <= value["seed"] <= 2**63 - 1
        or type(value.get("sampling_steps")) is not int
        or not 1 <= value["sampling_steps"] <= 500
        or type(value.get("guidance")) is not float
        or not math.isfinite(value["guidance"])
        or not 0.0 < value["guidance"] <= 50.0
        or type(value.get("image_count")) is not int
        or not 1 <= value["image_count"] <= 16
        or type(value.get("expected_step")) is not int
        or value["expected_step"] < 0
        or not isinstance(value.get("expected_variant"), str)
        or not value["expected_variant"]
        or len(value["expected_variant"]) > 128
        or value["expected_variant"] != value["expected_variant"].strip()
        or "/" in value["expected_variant"]
        or "\\" in value["expected_variant"]
    ):
        raise RuntimeError("contained Playground control parameters are malformed")
    paths = [_relative(workspace, value[key]) for key in ("checkpoint", "prompts", "output", "report")]
    if len(set(paths)) != len(paths) or paths[3].parent != workspace or paths[2].parent != workspace:
        raise RuntimeError("contained Playground control paths are malformed")
    inventory = value.get("code_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("contained Playground code inventory is malformed")
    worker_rows = [
        row
        for row in inventory
        if isinstance(row, dict) and row.get("path") == "src/spritelab/product_features/evaluation/playground_worker.py"
    ]
    if len(worker_rows) != 1 or worker_rows[0].get("sha256") != value["worker_sha256"]:
        raise RuntimeError("contained Playground worker is not bound to its code inventory")
    deadline = _parse_control_deadline(value.get("deadline_at"))
    return dict(value), deadline


def _relative(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise RuntimeError("contained Playground path is malformed")
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    reserved = {"aux", "con", "nul", "prn"}
    reserved.update(f"com{number}" for number in range(1, 10))
    reserved.update(f"lpt{number}" for number in range(1, 10))
    if (
        pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or pure.as_posix() != value
        or not pure.parts
        or unicodedata.normalize("NFC", value) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(
            part[-1:] in {".", " "}
            or any(character in '<>:"|?*' for character in part)
            or part.casefold().split(".", 1)[0] in reserved
            or any(ord(character) < 32 for character in part)
            for part in pure.parts
        )
    ):
        raise RuntimeError("contained Playground path is malformed")
    candidate = root.joinpath(*pure.parts)
    candidate.resolve(strict=False).relative_to(root)
    return candidate


def _stable_bytes(
    path: Path,
    maximum: int,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    if operation_check is not None:
        operation_check()
    descriptor = os.open(
        path,
        os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0)),
    )
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
            or stat.S_ISLNK(lexical.st_mode)
            or int(getattr(lexical, "st_file_attributes", 0))
            & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            or identity(before) != identity(lexical)
            or int(getattr(before, "st_nlink", 1)) != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise RuntimeError("contained Playground file is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            if operation_check is not None:
                operation_check()
            chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("contained Playground file is too large")
            if operation_check is not None:
                operation_check()
        after = os.fstat(descriptor)
        if identity(before) != identity(after) or identity(after) != identity(path.stat(follow_symlinks=False)):
            raise RuntimeError("contained Playground file changed")
        if operation_check is not None:
            operation_check()
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_descriptor_bytes(
    descriptor: int,
    maximum: int,
    operation_check: Callable[[], None] | None = None,
) -> bytes:
    if operation_check is not None:
        operation_check()
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or int(getattr(before, "st_nlink", 1)) != 1
        or before.st_size < 0
        or before.st_size > maximum
    ):
        raise RuntimeError("contained Playground descriptor is unsafe")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        if operation_check is not None:
            operation_check()
        chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise RuntimeError("contained Playground descriptor is too large")
        if operation_check is not None:
            operation_check()
    os.lseek(descriptor, 0, os.SEEK_SET)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, getattr(before, "st_mtime_ns", None)) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", None),
    ):
        raise RuntimeError("contained Playground descriptor changed")
    if operation_check is not None:
        operation_check()
    return b"".join(chunks)


def _verify_code(
    root: Path,
    rows: Any,
    operation_check: Callable[[], None] | None = None,
) -> dict[str, str]:
    if operation_check is not None:
        operation_check()
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("contained Playground code inventory is missing")
    expected: dict[str, str] = {}
    collision_paths: dict[str, str] = {}
    paths: list[str] = []
    for row in rows:
        if operation_check is not None:
            operation_check()
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError("contained Playground code inventory is malformed")
        relative = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative.startswith("src/spritelab/")
            or not relative.endswith(".py")
            or relative in expected
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("contained Playground code inventory is malformed")
        path = _relative(root, relative)
        if path.resolve(strict=False) != path:
            raise RuntimeError("contained Playground code inventory crosses an unsafe filesystem seam")
        collision_key = "/".join(unicodedata.normalize("NFC", part).casefold() for part in relative.split("/"))
        previous = collision_paths.setdefault(collision_key, relative)
        if previous != relative:
            raise RuntimeError("contained Playground code inventory has a portable path collision")
        expected[relative] = digest
        paths.append(relative)
    if paths != sorted(paths):
        raise RuntimeError("contained Playground code inventory is not canonical")
    _verify_expected_code(root, expected, operation_check=operation_check)
    if operation_check is not None:
        operation_check()
    return expected


def _verify_expected_code(
    root: Path,
    expected: dict[str, str],
    operation_check: Callable[[], None] | None = None,
) -> None:
    for relative, digest in expected.items():
        if operation_check is not None:
            operation_check()
        path = _relative(root, relative)
        if path.resolve(strict=False) != path:
            raise RuntimeError("contained Playground code inventory crosses an unsafe filesystem seam")
        payload = _stable_bytes(path, _MAX_PROJECT_SOURCE_BYTES, operation_check=operation_check)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError("contained Playground code changed before import")
    if operation_check is not None:
        operation_check()


class _ExactProjectSourceLoader:
    """Compile only the exact verified project-source bytes selected by the finder."""

    def __init__(
        self,
        fullname: str,
        path: Path,
        expected_sha256: str,
        package: bool,
        policy_identity: object,
        operation_check: Callable[[], None] | None,
    ) -> None:
        self.fullname = fullname
        self.path = path
        self.expected_sha256 = expected_sha256
        self.package = package
        self.policy_identity = policy_identity
        self.operation_check = operation_check

    def create_module(self, spec: Any) -> None:
        return None

    def get_filename(self, fullname: str) -> str:
        if fullname != self.fullname:
            raise RuntimeError("contained Playground project import changed")
        return str(self.path)

    def get_code(self, fullname: str) -> Any:
        if self.operation_check is not None:
            self.operation_check()
        payload = _stable_bytes(
            Path(self.get_filename(fullname)),
            _MAX_PROJECT_SOURCE_BYTES,
            operation_check=self.operation_check,
        )
        if hashlib.sha256(payload).hexdigest() != self.expected_sha256:
            raise RuntimeError("contained Playground project source changed before execution")
        return compile(payload, self.get_filename(fullname), "exec", dont_inherit=True)

    def is_package(self, fullname: str) -> bool:
        if fullname != self.fullname:
            raise RuntimeError("contained Playground project import changed")
        return self.package

    def _verify_module_identity(self, module: types.ModuleType) -> None:
        spec = getattr(module, "__spec__", None)
        module_file = getattr(module, "__file__", None)
        expected_package = self.fullname if self.package else self.fullname.rpartition(".")[0]
        if (
            module.__name__ != self.fullname
            or module.__loader__ is not self
            or module.__package__ != expected_package
            or spec is None
            or spec.name != self.fullname
            or spec.loader is not self
            or not isinstance(spec.origin, str)
            or os.path.normcase(os.path.abspath(spec.origin)) != os.path.normcase(os.path.abspath(self.path))
            or not isinstance(module_file, str)
            or os.path.normcase(os.path.abspath(module_file)) != os.path.normcase(os.path.abspath(self.path))
        ):
            raise RuntimeError("contained Playground project module has an unexpected loader")
        if self.package:
            expected_location = os.path.normcase(os.path.abspath(self.path.parent))
            module_locations = getattr(module, "__path__", None)
            spec_locations = spec.submodule_search_locations
            if (
                not isinstance(module_locations, list)
                or len(module_locations) != 1
                or os.path.normcase(os.path.abspath(module_locations[0])) != expected_location
                or spec_locations is None
                or len(spec_locations) != 1
                or os.path.normcase(os.path.abspath(spec_locations[0])) != expected_location
            ):
                raise RuntimeError("contained Playground project package escaped its inventory")

    def exec_module(self, module: Any) -> None:
        if not isinstance(module, types.ModuleType):
            raise RuntimeError("contained Playground project module has an unexpected loader")
        self._verify_module_identity(module)
        exec(self.get_code(self.fullname), module.__dict__)
        self._verify_module_identity(module)


class _ExactProjectSourceFinder:
    """Permit `spritelab` imports only through content-bound source loaders."""

    def __init__(
        self,
        project_root: Path,
        expected: dict[str, str],
        operation_check: Callable[[], None] | None = None,
    ) -> None:
        self.project_path = Path(project_root)
        self.source_root = self.project_path / "src"
        self.package_root = self.source_root / "spritelab"
        self.expected = types.MappingProxyType(dict(expected))
        self.policy_identity = object()
        self.runtime_finder: Any = None
        self.enabled = True
        self.operation_check = operation_check

    def bind_runtime_finder(self, candidate: Any) -> None:
        if self.runtime_finder is not None or not _is_runtime_finder_instance(candidate, self):
            raise RuntimeError("contained Playground runtime source policy is unexpected")
        self.runtime_finder = candidate

    def require_precedence(self) -> None:
        if not self.enabled or self not in sys.meta_path:
            raise RuntimeError("contained Playground project source policy was displaced")
        for candidate in sys.meta_path[: sys.meta_path.index(self)]:
            if candidate is not self.runtime_finder or getattr(candidate, "enabled", False) is not True:
                raise RuntimeError("contained Playground project source policy lost precedence")

    def loader_is_bound(self, fullname: str, loader: Any) -> bool:
        if not isinstance(loader, _ExactProjectSourceLoader):
            return False
        if loader.fullname != fullname or loader.policy_identity is not self.policy_identity:
            return False
        try:
            relative = loader.path.relative_to(self.project_path).as_posix()
        except ValueError:
            return False
        return self.expected.get(relative) == loader.expected_sha256

    def _canonical_origin(self, fullname: str, *, package: bool) -> Path:
        parts = fullname.split(".")
        if parts[0] != "spritelab" or any(not part or not part.isidentifier() for part in parts):
            raise RuntimeError("contained Playground project import escaped its inventory")
        relative_parts = parts[1:]
        if package:
            return self.package_root.joinpath(*relative_parts, "__init__.py")
        return self.package_root.joinpath(*relative_parts).with_suffix(".py")

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if self.operation_check is not None:
            self.operation_check()
        if not self.enabled or (fullname != "spritelab" and not fullname.startswith("spritelab.")):
            return None
        self.require_precedence()
        parts = fullname.split(".")
        search_root = self.source_root if fullname == "spritelab" else self.package_root.joinpath(*parts[1:-1])
        spec = importlib_machinery.PathFinder.find_spec(fullname, [str(search_root)], target)
        if spec is None or not isinstance(spec.origin, str) or not spec.origin.endswith(".py"):
            raise RuntimeError("contained Playground project import is unavailable")
        if type(spec.loader) is not importlib_machinery.SourceFileLoader:
            raise RuntimeError("contained Playground project module has an unexpected loader")
        origin = os.path.abspath(spec.origin)
        if os.path.normcase(os.path.realpath(origin)) != os.path.normcase(origin):
            raise RuntimeError("contained Playground project import crosses an unsafe filesystem seam")
        package = spec.submodule_search_locations is not None
        canonical_origin = self._canonical_origin(fullname, package=package)
        if os.path.normcase(origin) != os.path.normcase(os.path.abspath(canonical_origin)):
            raise RuntimeError("contained Playground project import escaped its canonical package path")
        relative = canonical_origin.relative_to(self.project_path).as_posix()
        expected_sha256 = self.expected.get(relative)
        if expected_sha256 is None:
            raise RuntimeError("contained Playground project import escaped its inventory")
        source_loader = spec.loader
        if source_loader.name != fullname or os.path.normcase(os.path.abspath(source_loader.path)) != os.path.normcase(
            origin
        ):
            raise RuntimeError("contained Playground project module has an unexpected loader")
        if package:
            locations = list(spec.submodule_search_locations or ())
            if len(locations) != 1 or os.path.normcase(os.path.realpath(locations[0])) != os.path.normcase(
                os.path.abspath(canonical_origin.parent)
            ):
                raise RuntimeError("contained Playground project package escaped its inventory")
        loader = _ExactProjectSourceLoader(
            fullname,
            canonical_origin,
            expected_sha256,
            package,
            self.policy_identity,
            self.operation_check,
        )
        return importlib_util.spec_from_file_location(
            fullname,
            canonical_origin,
            loader=loader,
            submodule_search_locations=[str(canonical_origin.parent)] if package else None,
        )


def _is_runtime_finder_instance(candidate: Any, finder: _ExactProjectSourceFinder) -> bool:
    module = sys.modules.get("spritelab.training.smoke_bundle")
    if not isinstance(module, types.ModuleType) or not finder.loader_is_bound(
        "spritelab.training.smoke_bundle", getattr(module, "__loader__", None)
    ):
        return False
    runtime_type = getattr(module, "_ExactRuntimeFinder", None)
    return isinstance(runtime_type, type) and type(candidate) is runtime_type


class _ProtectedMetaPath(list[Any]):
    """Prevent a later finder from taking precedence over bound project code."""

    def __init__(self, finder: _ExactProjectSourceFinder, previous: list[Any]) -> None:
        super().__init__([finder, *previous])
        self.finder = finder
        self.runtime_finder: Any = None
        self.accepting_runtime_finder = False
        self.locked = True

    def insert(self, index: SupportsIndex, value: Any) -> None:
        normalized_index = operator.index(index)
        if self.locked and normalized_index <= self.index(self.finder):
            if (
                not self.accepting_runtime_finder
                or self.runtime_finder is not None
                or not _is_runtime_finder_instance(value, self.finder)
            ):
                raise RuntimeError("contained Playground project source policy lost precedence")
            self.runtime_finder = value
            self.finder.bind_runtime_finder(value)
            self.accepting_runtime_finder = False
        super().insert(normalized_index, value)

    def remove(self, value: Any) -> None:
        if self.locked and value is not self.runtime_finder:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().remove(value)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().__delitem__(key)

    def clear(self) -> None:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().clear()

    def pop(self, index: SupportsIndex = -1) -> Any:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        return super().pop(index)

    def reverse(self) -> None:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().reverse()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        super().sort(*args, **kwargs)

    def __iadd__(self, values: Any) -> _ProtectedMetaPath:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        return super().__iadd__(values)

    def __imul__(self, count: SupportsIndex) -> _ProtectedMetaPath:
        if self.locked:
            raise RuntimeError("contained Playground project source policy was displaced")
        return super().__imul__(count)


class _ProjectImportPolicyGuard:
    def __init__(self, finder: _ExactProjectSourceFinder) -> None:
        self.finder = finder
        self.enabled = True

    def audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if (
            self.enabled
            and event == "import"
            and arguments
            and isinstance(arguments[0], str)
            and (arguments[0] == "spritelab" or arguments[0].startswith("spritelab."))
        ):
            self.finder.require_precedence()


def _verify_loaded_project_modules(
    finder: _ExactProjectSourceFinder,
    bootstrap_packages: dict[str, types.ModuleType],
    operation_check: Callable[[], None] | None = None,
) -> None:
    for name, module in tuple(sys.modules.items()):
        if operation_check is not None:
            operation_check()
        if name != "spritelab" and not name.startswith("spritelab."):
            continue
        if not isinstance(module, types.ModuleType):
            raise RuntimeError("contained Playground project module identity changed")
        if getattr(module, "__spritelab_exact_bootstrap__", None) is True and bootstrap_packages.get(name) is module:
            continue
        if not finder.loader_is_bound(name, getattr(module, "__loader__", None)):
            raise RuntimeError("contained Playground project module escaped its exact loader")


@contextmanager
def _bound_project_source_imports(
    root: Path,
    rows: Any,
    operation_check: Callable[[], None] | None = None,
) -> Iterator[_ExactProjectSourceFinder]:
    """Install the exact source finder before the first audited project import."""

    expected = _verify_code(root, rows, operation_check=operation_check)
    if any(name == "spritelab" or name.startswith("spritelab.") for name in sys.modules):
        raise RuntimeError("contained Playground project code loaded before its exact source policy")
    finder = _ExactProjectSourceFinder(root, expected, operation_check=operation_check)
    sys.dont_write_bytecode = True
    previous_meta_path = sys.meta_path
    protected_meta_path = _ProtectedMetaPath(finder, list(previous_meta_path))
    sys.meta_path = protected_meta_path
    guard = _ProjectImportPolicyGuard(finder)
    sys.addaudithook(guard.audit)
    bootstrap_packages: dict[str, types.ModuleType] = {}
    for name in ("spritelab", "spritelab.utils", "spritelab.training"):
        if operation_check is not None:
            operation_check()
        parts = name.split(".")
        directory = root / "src" / Path(*parts)
        init_relative = (directory / "__init__.py").relative_to(root).as_posix()
        if init_relative not in expected or directory.resolve(strict=False) != directory:
            raise RuntimeError("contained Playground bootstrap package is unavailable")
        package = types.ModuleType(name)
        package_spec = importlib_machinery.ModuleSpec(name, loader=None, is_package=True)
        package_spec.submodule_search_locations = [str(directory)]
        package.__package__ = name
        package.__path__ = list(package_spec.submodule_search_locations)
        package.__spec__ = package_spec
        package.__spritelab_exact_bootstrap__ = True
        sys.modules[name] = package
        bootstrap_packages[name] = package
        if len(parts) > 1:
            setattr(sys.modules[".".join(parts[:-1])], parts[-1], package)
    try:
        yield finder
    finally:
        try:
            finder.require_precedence()
            _verify_loaded_project_modules(finder, bootstrap_packages, operation_check=operation_check)
            _verify_expected_code(root, expected, operation_check=operation_check)
        finally:
            guard.enabled = False
            finder.enabled = False
            protected_meta_path.locked = False
            sys.meta_path = previous_meta_path
            for name in tuple(sys.modules):
                if name == "spritelab" or name.startswith("spritelab."):
                    sys.modules.pop(name, None)


def _activate_exact_project_packages(
    finder: _ExactProjectSourceFinder,
    operation_check: Callable[[], None] | None = None,
) -> None:
    """Execute bootstrap package initializers after the exact runtime finder exists."""

    runtime_policy_active = (
        finder.runtime_finder is not None
        and finder.runtime_finder in sys.meta_path
        and getattr(finder.runtime_finder, "enabled", False) is True
    )
    if operation_check is not None:
        operation_check()
    finder.require_precedence()
    if not runtime_policy_active or not finder.enabled or finder not in sys.meta_path:
        raise RuntimeError("contained Playground project bootstrap identity changed")
    for name in ("spritelab", "spritelab.utils", "spritelab.training"):
        if operation_check is not None:
            operation_check()
        package = sys.modules.get(name)
        if (
            not isinstance(package, types.ModuleType)
            or getattr(package, "__spritelab_exact_bootstrap__", None) is not True
        ):
            raise RuntimeError("contained Playground project bootstrap identity changed")
        parent_path = None
        if "." in name:
            parent = sys.modules.get(name.rsplit(".", 1)[0])
            parent_path = getattr(parent, "__path__", None)
            if not isinstance(parent_path, list):
                raise RuntimeError("contained Playground project bootstrap identity changed")
        spec = finder.find_spec(name, parent_path)
        if spec is None or not isinstance(spec.loader, _ExactProjectSourceLoader):
            raise RuntimeError("contained Playground package initializer is unavailable")
        package.__spec__ = spec
        package.__loader__ = spec.loader
        package.__file__ = spec.origin
        package.__package__ = name
        package.__path__ = list(spec.submodule_search_locations or ())
        spec.loader.exec_module(package)
        if sys.modules.get(name) is not package or not finder.loader_is_bound(name, package.__loader__):
            raise RuntimeError("contained Playground project bootstrap identity changed")
        if "." in name:
            parent = sys.modules.get(name.rsplit(".", 1)[0])
            if getattr(parent, name.rsplit(".", 1)[1], None) is not package:
                raise RuntimeError("contained Playground project bootstrap identity changed")
        del package.__spritelab_exact_bootstrap__


@contextmanager
def _bound_exact_runtime_imports(
    finder: _ExactProjectSourceFinder,
    policy_factory: Any,
    project_root: Path,
    closure: Any,
    operation_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Bind the one exact-loaded runtime finder installed by the runtime policy."""

    protected_meta_path = sys.meta_path
    if (
        not isinstance(protected_meta_path, _ProtectedMetaPath)
        or protected_meta_path.finder is not finder
        or protected_meta_path.runtime_finder is not None
        or finder.runtime_finder is not None
    ):
        raise RuntimeError("contained Playground project source policy was displaced")
    with ExitStack() as stack:
        if operation_check is not None:
            operation_check()
        protected_meta_path.accepting_runtime_finder = True
        try:
            policy = (
                policy_factory(project_root, closure)
                if operation_check is None
                else policy_factory(project_root, closure, operation_check=operation_check)
            )
            verified = stack.enter_context(policy)
        finally:
            protected_meta_path.accepting_runtime_finder = False
        if (
            protected_meta_path.runtime_finder is None
            or finder.runtime_finder is not protected_meta_path.runtime_finder
            or finder.runtime_finder not in sys.meta_path
            or getattr(finder.runtime_finder, "enabled", False) is not True
        ):
            raise RuntimeError("contained Playground runtime source policy is unavailable")
        finder.require_precedence()
        if operation_check is not None:
            operation_check()
        yield verified


def _write_report(
    path: Path,
    value: dict[str, Any],
    *,
    operation_check: Callable[[], None],
) -> None:
    operation_check()
    payload = (json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
    if len(payload) > 16 * 1024 * 1024:
        raise RuntimeError("contained Playground result is too large")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("xb") as handle:
        for offset in range(0, len(payload), 1024 * 1024):
            operation_check()
            handle.write(payload[offset : offset + 1024 * 1024])
        handle.flush()
        os.fsync(handle.fileno())
    operation_check()
    os.replace(temporary, path)
    operation_check()


def _replace_diagnostic_bytes(
    path: Path,
    payload: bytes,
    *,
    operation_check: Callable[[], None],
) -> None:
    operation_check()
    temporary = path.with_name(f".{path.name}.projected-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    operation_check()
    os.replace(temporary, path)
    operation_check()


def _project_playground_diagnostics(
    workspace: Path,
    output_relative: Path,
    *,
    image_count: int,
    operation_check: Callable[[], None],
) -> dict[str, int]:
    """Retain only the path-free manifest/report fields consumed by the parent."""

    output = workspace / output_relative
    manifest = output / "generated_manifest.jsonl"
    payload = _stable_bytes(manifest, 2 * 1024 * 1024, operation_check=operation_check)
    rows: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        operation_check()
        if not raw_line.strip():
            continue
        value = _strict_json_loads(raw_line)
        if not isinstance(value, dict):
            raise RuntimeError("contained Playground manifest row is malformed")
        paths = value.get("paths")
        indexed_png = paths.get("indexed_png") if isinstance(paths, dict) else None
        if not isinstance(indexed_png, str):
            raise RuntimeError("contained Playground manifest path is malformed")
        projected_path = _relative(output, indexed_png).relative_to(output).as_posix()
        if not projected_path.casefold().endswith(".png"):
            raise RuntimeError("contained Playground manifest path is malformed")
        projected = {
            "cfg_scale": value.get("cfg_scale"),
            "model_type": value.get("model_type"),
            "noise_seed": value.get("noise_seed"),
            "paths": {"indexed_png": projected_path},
            "prompt": value.get("prompt"),
            "prompt_id": value.get("prompt_id"),
            "sample_id": value.get("sample_id"),
            "scope": value.get("scope"),
            "seed": value.get("seed"),
            "steps": value.get("steps"),
        }
        if (
            set(projected)
            != {
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
            }
            or not isinstance(projected["prompt"], str)
            or not isinstance(projected["prompt_id"], str)
            or not isinstance(projected["sample_id"], str)
            or type(projected["seed"]) is not int
            or type(projected["noise_seed"]) is not int
            or type(projected["steps"]) is not int
            or isinstance(projected["cfg_scale"], bool)
            or not isinstance(projected["cfg_scale"], (int, float))
            or not math.isfinite(float(projected["cfg_scale"]))
        ):
            raise RuntimeError("contained Playground manifest semantics are malformed")
        rows.append(projected)
    if len(rows) != image_count:
        raise RuntimeError("contained Playground manifest count is malformed")
    manifest_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _replace_diagnostic_bytes(manifest, manifest_payload, operation_check=operation_check)
    retained_report = {"manifest": "generated_manifest.jsonl", "sample_count": image_count}
    _replace_diagnostic_bytes(
        output / "generation_report.json",
        json.dumps(retained_report, allow_nan=False, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8")
        + b"\n",
        operation_check=operation_check,
    )
    _replace_diagnostic_bytes(
        output / "generation_report.md",
        f"Generated samples: {image_count}\nManifest: generated_manifest.jsonl\n".encode(),
        operation_check=operation_check,
    )
    return {"sample_count": image_count}


def _run_with_bound_project_sources(
    parsed: argparse.Namespace,
    workspace: Path,
    project_root: Path,
    value: dict[str, Any],
    project_finder: _ExactProjectSourceFinder,
    operation_check: Callable[[], None],
) -> int:
    operation_check()
    confinement_evidence: dict[str, Any]
    workspace_identity = value.get("workspace_identity")
    if not isinstance(workspace_identity, dict):
        return 70
    windows_worker = sys.platform == "win32"
    if sys.platform.startswith("linux"):
        from spritelab.utils.write_confinement import enforce_linux_landlock_write_confinement_fd

        if parsed.workspace_fd is None:
            return 70
        operation_check()
        confinement_evidence = enforce_linux_landlock_write_confinement_fd(
            parsed.workspace_fd,
            expected_device=int(workspace_identity.get("device", -1)),
            expected_inode=int(workspace_identity.get("inode", -1)),
        ).to_dict()
        operation_check()
    elif windows_worker:
        # This is intentionally the first project source executed by the
        # Windows worker.  The outer, standard-library-only bootstrap has
        # already lowered the token to Untrusted; prove that exact boundary
        # before runtime-closure policy, generator, or Torch imports.
        from spritelab.utils.write_confinement import windows_current_process_confinement_evidence

        if any(descriptor is not None for descriptor in (parsed.workspace_fd, parsed.checkpoint_fd, parsed.prompts_fd)):
            return 70
        operation_check()
        confinement_evidence = windows_current_process_confinement_evidence(
            workspace,
            expected_device=int(workspace_identity.get("device", -1)),
            expected_inode=int(workspace_identity.get("inode", -1)),
        ).to_dict()
        operation_check()
    else:
        return 70
    from spritelab.utils.runtime_closure import exact_python_runtime_import_policy

    checkpoint = _relative(workspace, value["checkpoint"])
    if parsed.checkpoint_fd is not None:
        checkpoint = Path(f"/proc/self/fd/{parsed.checkpoint_fd}")
        checkpoint_payload = _stable_descriptor_bytes(
            parsed.checkpoint_fd,
            8 * 1024**3,
            operation_check=operation_check,
        )
    else:
        checkpoint_payload = _stable_bytes(checkpoint, 8 * 1024**3, operation_check=operation_check)
    if hashlib.sha256(checkpoint_payload).hexdigest() != value.get("checkpoint_sha256"):
        return 70
    prompts = _relative(workspace, value["prompts"])
    if parsed.prompts_fd is None:
        if not windows_worker:
            return 70
        prompts_payload = _stable_bytes(prompts, 4 * 1024 * 1024, operation_check=operation_check)
    else:
        prompts_payload = _stable_descriptor_bytes(
            parsed.prompts_fd,
            4 * 1024 * 1024,
            operation_check=operation_check,
        )
        prompts = Path(f"/proc/self/fd/{parsed.prompts_fd}")
    if hashlib.sha256(prompts_payload).hexdigest() != value.get("prompts_sha256"):
        return 70
    output_relative = _relative(workspace, value["output"]).relative_to(workspace)

    # Installed runtime files are a trusted baseline. Source imports are bound
    # byte-for-byte; native/resource bytes receive pre/post drift detection with
    # the bounded residuals persisted in the runtime-closure contract.
    with _bound_exact_runtime_imports(
        project_finder,
        exact_python_runtime_import_policy,
        project_root,
        value.get("runtime_closure"),
        operation_check=operation_check,
    ):
        operation_check()
        _activate_exact_project_packages(project_finder, operation_check=operation_check)
        from spritelab.product_features.evaluation.local_generator import LocalCheckpointPlaygroundGenerator
        from spritelab.training.generator_challenger import ChallengerSampleConfig, run_sample_generator_challenger

        LocalCheckpointPlaygroundGenerator._validate_snapshot_checkpoint(
            checkpoint,
            expected_step=int(value["expected_step"]),
            expected_variant=str(value["expected_variant"]),
        )
        operation_check()
        config = ChallengerSampleConfig(
            checkpoint=checkpoint,
            prompts=prompts,
            out_dir=output_relative,
            expected_checkpoint_sha256=str(value["checkpoint_sha256"]),
            expected_checkpoint_step=int(value["expected_step"]),
            expected_checkpoint_variant=str(value["expected_variant"]),
            max_samples=int(value["image_count"]),
            steps=int(value["sampling_steps"]),
            cfg_scale=float(value["guidance"]),
            device="auto",
            seed=int(value["seed"]),
            noise_seed=int(value["seed"]),
            batch_size=min(int(value["image_count"]), 16),
            write_raw_rgba=False,
            write_hard_rgba=True,
            contact_sheet_labels="prompt",
        )
        raw_report = dict(run_sample_generator_challenger(config))
        operation_check()
        if raw_report.get("sample_count") != int(value["image_count"]):
            raise RuntimeError("contained Playground sampler count is malformed")
        import torch

        selected_device = str(raw_report.get("device") or "auto")[:80]
        report = _project_playground_diagnostics(
            workspace,
            output_relative,
            image_count=int(value["image_count"]),
            operation_check=operation_check,
        )
        operation_check()
        runtime_identity = {
            "schema_version": "spritelab.playground-runtime-identity.v2",
            "runtime_reported": True,
            "python_version": sys.version.split()[0],
            "python_implementation": sys.implementation.name,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "selected_device": selected_device,
            "platform": sys.platform,
            "runtime_closure_identity": str(value["runtime_closure"]["runtime_closure_identity"]),
            "execution_byte_policy": str(value["runtime_closure"]["execution_byte_policy"]),
            "bounded_residuals": list(value["runtime_closure"]["bounded_residuals"]),
            "paths_exposed": False,
        }
        result: dict[str, Any] = {
            "schema_version": _RESULT_SCHEMA,
            "result_identity": "",
            "control_identity": value["control_identity"],
            "bootstrap_identity": value["bootstrap_identity"],
            "worker_sha256": value["worker_sha256"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "prompts_sha256": value["prompts_sha256"],
            "code_inventory_identity": value["code_inventory_identity"],
            "runtime_closure_identity": value["runtime_closure_identity"],
            "workspace_identity": dict(value["workspace_identity"]),
            "deadline_at": value["deadline_at"],
            "report": report,
            "runtime_identity": runtime_identity,
            "write_confinement": confinement_evidence,
        }
        result["result_identity"] = _record_identity(result, "result_identity")
        operation_check()
        _write_report(
            _relative(workspace, value["report"]),
            result,
            operation_check=operation_check,
        )
        operation_check()
    return 0


def _main() -> int:
    parsed = _arguments()
    if (
        not _is_sha256(parsed.control_sha256)
        or not _is_sha256(parsed.bootstrap_sha256)
        or not _is_sha256(parsed.worker_sha256)
        or type(parsed.worker_size) is not int
        or not 1 <= parsed.worker_size <= _MAX_BOUND_WORKER_BYTES
    ):
        return 70
    worker_payload = _stable_descriptor_bytes(parsed.worker_fd, _MAX_BOUND_WORKER_BYTES)
    if len(worker_payload) != parsed.worker_size or hashlib.sha256(worker_payload).hexdigest() != parsed.worker_sha256:
        return 70
    workspace = Path.cwd().resolve()
    project_value = os.environ.get("SPRITELAB_PROJECT_ROOT")
    if not project_value:
        return 70
    project_root = Path(project_value).resolve(strict=True)
    control_path = _relative(workspace, parsed.control)
    control_payload = _stable_bytes(control_path, _MAX_CONTROL_BYTES)
    if hashlib.sha256(control_payload).hexdigest() != parsed.control_sha256:
        return 70
    value = _strict_json_loads(control_payload)
    if not isinstance(value, dict) or control_payload != _canonical_bytes(value) + b"\n":
        return 70
    value, deadline = _validate_control(value, workspace=workspace, parsed=parsed)
    operation_check = _deadline_operation(deadline)

    # The launch bootstrap executes this worker from its already hash-bound
    # descriptor.  Up to this point only the interpreter's trusted standard
    # runtime has executed.  Install the content-bound project finder before
    # importing write-confinement, runtime-policy, or any other project module.
    with _bound_project_source_imports(
        project_root,
        value.get("code_inventory"),
        operation_check=operation_check,
    ) as project_finder:
        result = _run_with_bound_project_sources(
            parsed,
            workspace,
            project_root,
            value,
            project_finder,
            operation_check,
        )
    operation_check()
    final_worker_payload = _stable_descriptor_bytes(
        parsed.worker_fd,
        _MAX_BOUND_WORKER_BYTES,
        operation_check=operation_check,
    )
    if final_worker_payload != worker_payload:
        return 70
    operation_check()
    return result


def main() -> int:
    try:
        return _main()
    except Exception:
        return 70


if __name__ == "__main__":  # pragma: no cover - exercised through the parent adapter
    raise SystemExit(main())

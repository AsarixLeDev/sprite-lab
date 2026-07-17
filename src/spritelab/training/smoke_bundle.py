"""Fail-closed infrastructure-smoke bundles for exploratory Playground use.

This module never discovers arbitrary paths.  Every artifact is derived from an
opaque smoke ID under one of the fixed project roots.  Ordinary ``--smoke``
runs are intentionally outside this contract and cannot be registered.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import secrets
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from spritelab.utils.pinned_executable import (
    PinnedExecutable,
    PinnedExecutableError,
    pin_executable,
    read_executable_identity,
    verify_process_image,
)
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity

SMOKE_PLAN_SCHEMA = "spritelab.training.smoke-plan.v1"
SMOKE_RUN_STATE_SCHEMA = "spritelab.training.smoke-run-state.v1"
SMOKE_DEVICE_RECEIPT_SCHEMA = "spritelab.training.smoke-device-receipt.v1"
SMOKE_EVIDENCE_SCHEMA = "spritelab.training.smoke-evidence.v1"
SMOKE_CHILD_ENVIRONMENT_SCHEMA = "spritelab.training.smoke-child-environment.v1"
SMOKE_INTERPRETER_SCHEMA = "spritelab.training.smoke-interpreter.v1"
SMOKE_ORCHESTRATION_CODE_SCHEMA = "spritelab.training.smoke-orchestration-code.v1"
SMOKE_SCOPE = "EXPLORATORY_INFRASTRUCTURE_SMOKE"
SMOKE_STATUS = "PROVISIONALLY_VERIFIED"
EXPLORATORY_REGISTRATION_SCHEMA = "spritelab.playground.exploratory-checkpoint-registration.v1"
SMOKE_ID_PATTERN = re.compile(r"^smoke-[0-9a-f]{20}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SMOKE_DEVICES = ("cpu", "cuda")
FALSE_ELIGIBILITY = {
    "production_eligible": False,
    "campaign_execution_evidence": False,
    "training_resume_eligible": False,
    "evaluation_eligible": False,
    "promotion_eligible": False,
}
_SANDBOXED_ENVIRONMENT_PATHS = (
    "APPDATA",
    "CUDA_CACHE_PATH",
    "HOME",
    "LOCALAPPDATA",
    "MPLCONFIGDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_HOME",
    "USERPROFILE",
    "XDG_CACHE_HOME",
)
_CHILD_PREFLIGHT_SOURCE = r"""
import hashlib as _h
import importlib.abc as _ia
import importlib.machinery as _im
import importlib.util as _iu
import json as _j
import os as _o
import re as _r
import stat as _s
import sys as _y

_RP = int(getattr(_s, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _fail():
    raise SystemExit(70)


def _canonical(value):
    return _j.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_hash(value):
    return _h.sha256(_canonical(value)).hexdigest()


def _metadata(value):
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(getattr(value, "st_nlink", 1)),
        int(getattr(value, "st_mtime_ns", 0)),
    )


def _safe_directory(path):
    value = _o.lstat(path)
    if (
        not _s.S_ISDIR(value.st_mode)
        or _s.S_ISLNK(value.st_mode)
        or int(getattr(value, "st_file_attributes", 0)) & _RP
    ):
        _fail()
    return value


def _stable_file(path, maximum=8 * 1024 * 1024):
    lexical = _o.lstat(path)
    if (
        not _s.S_ISREG(lexical.st_mode)
        or _s.S_ISLNK(lexical.st_mode)
        or int(getattr(lexical, "st_file_attributes", 0)) & _RP
        or int(getattr(lexical, "st_nlink", 1)) != 1
        or lexical.st_size < 0
        or lexical.st_size > maximum
    ):
        _fail()
    flags = _o.O_RDONLY | int(getattr(_o, "O_BINARY", 0)) | int(getattr(_o, "O_NOFOLLOW", 0))
    descriptor = _o.open(path, flags)
    try:
        before = _o.fstat(descriptor)
        if _metadata(before) != _metadata(lexical):
            _fail()
        chunks = []
        total = 0
        while True:
            chunk = _o.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail()
        after = _o.fstat(descriptor)
        if _metadata(before) != _metadata(after) or _metadata(after) != _metadata(_o.lstat(path)):
            _fail()
        return b"".join(chunks)
    finally:
        _o.close(descriptor)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail()
        value[key] = item
    return value


def _argument(name):
    indexes = [index for index, value in enumerate(_y.argv[:-1]) if value == name]
    if len(indexes) != 1:
        _fail()
    return _y.argv[indexes[0] + 1]


def _scan_sources(root):
    source = _o.path.join(root, "src", "spritelab")
    _safe_directory(_o.path.join(root, "src"))
    inventory = {}

    def walk(directory, prefix):
        before = _safe_directory(directory)
        with _o.scandir(directory) as stream:
            entries = sorted(stream, key=lambda item: item.name)
        for entry in entries:
            value = entry.stat(follow_symlinks=False)
            if _s.S_ISLNK(value.st_mode) or int(getattr(value, "st_file_attributes", 0)) & _RP:
                _fail()
            relative = f"{prefix}/{entry.name}"
            if _s.S_ISDIR(value.st_mode):
                walk(entry.path, relative)
            elif entry.name.endswith(".py"):
                payload = _stable_file(entry.path)
                inventory[relative] = (_h.sha256(payload).hexdigest(), len(payload))
        if _metadata(before) != _metadata(_safe_directory(directory)):
            _fail()

    walk(source, "src/spritelab")
    return inventory


class _BoundSourceLoader(_ia.Loader):
    def __init__(self, fullname, path, expected, package):
        self.fullname = fullname
        self.path = path
        self.expected = expected
        self.package = package

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        if fullname != self.fullname:
            _fail()
        return self.path

    def get_code(self, fullname):
        if fullname != self.fullname:
            _fail()
        payload = _stable_file(self.path)
        if _h.sha256(payload).hexdigest() != self.expected:
            _fail()
        return compile(payload, self.path, "exec", dont_inherit=True)

    def is_package(self, fullname):
        if fullname != self.fullname:
            _fail()
        return self.package

    def exec_module(self, module):
        exec(self.get_code(self.fullname), module.__dict__)


class _BoundSourceFinder(_ia.MetaPathFinder):
    def __init__(self, root, expected):
        self.root = _o.path.normcase(_o.path.abspath(root))
        self.expected = expected

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "spritelab" and not fullname.startswith("spritelab."):
            return None
        spec = _im.PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.origin, str) or not spec.origin.endswith(".py"):
            _fail()
        origin = _o.path.abspath(spec.origin)
        if _o.path.normcase(_o.path.realpath(origin)) != _o.path.normcase(origin):
            _fail()
        try:
            relative = _o.path.relpath(origin, self.root).replace(_o.sep, "/")
        except ValueError:
            _fail()
        if relative.startswith("../") or relative not in self.expected:
            _fail()
        package = spec.submodule_search_locations is not None
        loader = _BoundSourceLoader(fullname, origin, self.expected[relative], package)
        locations = [str(_o.path.dirname(origin))] if package else None
        return _iu.spec_from_file_location(
            fullname,
            origin,
            loader=loader,
            submodule_search_locations=locations,
        )


def _preflight(mode, source_sha256):
    if mode not in {"main", "worker"} or not _r.fullmatch(r"[0-9a-f]{64}", source_sha256):
        _fail()
    smoke_id = _argument("--smoke-bundle-id" if mode == "main" else "--smoke-id")
    device = _argument("--smoke-device" if mode == "main" else "--device")
    plan_identity = _argument("--smoke-plan-identity" if mode == "main" else "--plan-identity")
    if not _r.fullmatch(r"smoke-[0-9a-f]{20}", smoke_id) or device not in {"cpu", "cuda"}:
        _fail()
    root = _o.path.realpath(_o.getcwd())
    cursor = root
    for part in ("artifacts", "training", "smokes", smoke_id):
        cursor = _o.path.join(cursor, part)
        _safe_directory(cursor)
    plan_payload = _stable_file(_o.path.join(cursor, "plan.json"), 16 * 1024 * 1024)
    plan = _j.loads(plan_payload, object_pairs_hook=_unique_object)
    if not isinstance(plan, dict) or plan.get("smoke_id") != smoke_id:
        _fail()
    identity_payload = dict(plan)
    stored_plan_identity = identity_payload.pop("plan_identity", None)
    if stored_plan_identity != plan_identity or _stable_hash(identity_payload) != plan_identity:
        _fail()
    configurations = plan.get("configurations")
    if not isinstance(configurations, dict) or not isinstance(configurations.get(device), dict):
        _fail()
    child = configurations[device].get("child_environment")
    environment = dict(_o.environ)
    if not isinstance(child, dict) or _stable_hash(environment) != child.get("environment_sha256"):
        _fail()
    paths_value = environment.get("SPRITELAB_ISOLATED_PATHS")
    if not isinstance(paths_value, str):
        _fail()
    paths = paths_value.split(_o.pathsep)
    path_hashes = [_h.sha256(value.encode("utf-8", "surrogatepass")).hexdigest() for value in paths]
    if (
        not paths
        or any(not value for value in paths)
        or len(paths) != child.get("isolated_import_path_count")
        or _stable_hash(path_hashes) != child.get("isolated_import_paths_sha256")
        or _o.path.normcase(_o.path.realpath(paths[0]))
        != _o.path.normcase(_o.path.join(root, "src"))
    ):
        _fail()
    interpreter = plan.get("interpreter")
    if not isinstance(interpreter, dict):
        _fail()
    image_path = _o.path.realpath(f"/proc/{_o.getpid()}/exe") if _y.platform.startswith("linux") else _y.executable
    image = _stable_file(image_path, 2 * 1024**3)
    if (
        len(image) != interpreter.get("byte_count")
        or _h.sha256(image).hexdigest() != interpreter.get("executable_sha256")
    ):
        _fail()
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict):
        _fail()
    code = bindings.get("training_code_identity")
    if not isinstance(code, dict) or code.get("sha256") != bindings.get("training_code_identity_sha256"):
        _fail()
    code_payload = dict(code)
    code_identity = code_payload.pop("sha256", None)
    if _stable_hash(code_payload) != code_identity or not isinstance(code.get("files"), list):
        _fail()
    expected = {}
    for record in code["files"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"].startswith("src/spritelab/")
            or not record["path"].endswith(".py")
            or not _r.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""))
            or record["path"] in expected
        ):
            _fail()
        expected[record["path"]] = record["sha256"]
    actual = _scan_sources(root)
    if {path: value[0] for path, value in actual.items()} != expected:
        _fail()
    orchestration = plan.get("orchestration_code")
    if (
        not isinstance(orchestration, dict)
        or orchestration.get("preflight_sha256") != source_sha256
        or not isinstance(orchestration.get("inventory"), dict)
    ):
        _fail()
    for path, record in orchestration["inventory"].items():
        if (
            path not in actual
            or not isinstance(record, dict)
            or actual[path] != (record.get("sha256"), record.get("byte_count"))
        ):
            _fail()
    _y.path[:0] = paths
    _y.meta_path.insert(0, _BoundSourceFinder(root, expected))


def _spritelab_smoke_preflight(mode, source_sha256):
    try:
        _preflight(mode, source_sha256)
    except SystemExit:
        raise
    except BaseException:
        _fail()
"""
_ISOLATED_MAIN_BOOTSTRAP = (
    "import hashlib as _h;__spritelab_preflight_source="
    + repr(_CHILD_PREFLIGHT_SOURCE)
    + ";exec(__spritelab_preflight_source);_spritelab_smoke_preflight("
    "'main',_h.sha256(__spritelab_preflight_source.encode()).hexdigest());"
    "import runpy as _runpy;_runpy.run_module('spritelab',run_name='__main__')"
)
_ISOLATED_WORKER_BOOTSTRAP = (
    "import hashlib as _h;__spritelab_preflight_source="
    + repr(_CHILD_PREFLIGHT_SOURCE)
    + ";exec(__spritelab_preflight_source);_spritelab_smoke_preflight("
    "'worker',_h.sha256(__spritelab_preflight_source.encode()).hexdigest());"
    "import runpy as _runpy;_runpy.run_module('spritelab.training.smoke_worker',run_name='__main__')"
)
_ORCHESTRATION_CODE_PATHS = (
    "src/spritelab/__main__.py",
    "src/spritelab/training/cli/experiment_cmds.py",
    "src/spritelab/training/smoke_bundle.py",
    "src/spritelab/training/smoke_runner.py",
    "src/spritelab/training/smoke_worker.py",
    "src/spritelab/utils/pinned_executable.py",
)


class SmokeBundleError(ValueError):
    """A public, path-free smoke-bundle contract failure."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass(frozen=True)
class VerifiedSmokeCheckpoint:
    weights: str
    path: Path
    sha256: str
    byte_count: int
    step: int
    variant: str


@dataclass(frozen=True)
class VerifiedSmokeBundle:
    evidence: Mapping[str, Any]
    checkpoints: tuple[VerifiedSmokeCheckpoint, ...]


PinnedSmokeInterpreter = PinnedExecutable


def smoke_id_for_campaign(campaign_identity: str, preparation_nonce: str) -> str:
    if not SHA256_PATTERN.fullmatch(str(campaign_identity)):
        raise SmokeBundleError("campaign_identity", "The campaign identity is malformed.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,79}", str(preparation_nonce)):
        raise SmokeBundleError("preparation_nonce", "A fresh smoke preparation nonce is required.")
    identity = hashlib.sha256(f"{campaign_identity}:{preparation_nonce}".encode()).hexdigest()
    return f"smoke-{identity[:20]}"


def artifact_bundle_directory(project_root: str | Path, smoke_id: str) -> Path:
    _require_smoke_id(smoke_id)
    return Path(project_root).resolve() / "artifacts" / "training" / "smokes" / smoke_id


def run_bundle_directory(project_root: str | Path, smoke_id: str) -> Path:
    _require_smoke_id(smoke_id)
    return Path(project_root).resolve() / "runs" / "v3" / "training-smokes" / smoke_id


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + ("\n" if pretty else "")).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def prepare_smoke_environment_binding(
    project_root: str | Path,
    smoke_id: str,
    device: str,
    *,
    inherited_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind a minimal child environment without persisting private path values."""

    root = Path(project_root).resolve()
    normalized = _require_device(device)
    _require_smoke_id(smoke_id)
    public = _public_smoke_environment(normalized)
    inherited = _allowed_inherited_environment(inherited_environment or os.environ)
    temporary_relative = f"artifacts/training/smokes/{smoke_id}/execution/temp/{normalized}"
    temporary = _fixed_relative(root, temporary_relative)
    import_paths = _isolated_import_paths(root)
    child = _compose_child_environment(
        public,
        inherited,
        temporary,
        import_paths,
        _lexical_interpreter_path(),
    )
    binding = {
        "schema_version": SMOKE_CHILD_ENVIRONMENT_SCHEMA,
        "inherited_names": sorted(inherited),
        "temporary_root": temporary_relative,
        "sandboxed_path_variables": list(_SANDBOXED_ENVIRONMENT_PATHS),
        "isolated_import_path_count": len(import_paths),
        "isolated_import_paths_sha256": stable_hash(
            [hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in import_paths]
        ),
        "environment_sha256": stable_hash(child),
    }
    return public, binding


def build_smoke_child_environment(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    inherited_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Reconstruct and verify the exact full environment bound by a plan."""

    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    record = dict(validated["configurations"])[normalized]
    public = _validate_environment_record(record.get("environment"), normalized)
    binding = _validate_child_environment_binding(record.get("child_environment"))
    inherited_source = inherited_environment or os.environ
    allowed = _allowed_inherited_environment(inherited_source)
    bound_interpreter = Path(str(inherited_source.get("SPRITELAB_BOUND_INTERPRETER") or _lexical_interpreter_path()))
    if sorted(allowed) != binding["inherited_names"]:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The allowlisted host environment changed after smoke preparation; prepare a fresh bundle.",
        )
    temporary = _fixed_relative(root, str(binding["temporary_root"]))
    expected_relative = f"artifacts/training/smokes/{validated['smoke_id']}/execution/temp/{normalized}"
    if str(binding["temporary_root"]) != expected_relative:
        raise SmokeBundleError("smoke_environment", "The smoke temporary environment root is invalid.")
    import_paths = _isolated_import_paths(root)
    path_hash = stable_hash(
        [hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest() for path in import_paths]
    )
    if (
        binding["isolated_import_path_count"] != len(import_paths)
        or binding["isolated_import_paths_sha256"] != path_hash
    ):
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The isolated dependency import path changed after smoke preparation; prepare a fresh bundle.",
        )
    child = _compose_child_environment(
        public,
        allowed,
        temporary,
        import_paths,
        bound_interpreter,
    )
    if stable_hash(child) != binding["environment_sha256"]:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The exact host environment bound by the smoke plan changed; prepare a fresh bundle.",
        )
    return child


def prepare_smoke_interpreter_binding(*, lexical_path: str | Path | None = None) -> dict[str, Any]:
    """Hash the exact Python binary while exposing no executable path."""

    lexical = _lexical_interpreter_path() if lexical_path is None else Path(lexical_path).absolute()
    lexical_metadata = _lexical_interpreter_metadata(lexical)
    executable = lexical.resolve(strict=True)
    try:
        executable_identity = read_executable_identity(executable)
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter",
            "The isolated Python interpreter is unsafe.",
        ) from exc
    digest = executable_identity.executable_sha256
    byte_count = executable_identity.byte_count
    path_identity = hashlib.sha256(
        os.path.normcase(str(executable_identity.resolved_path)).encode("utf-8", "surrogatepass")
    ).hexdigest()
    return finalize_identity(
        {
            "schema_version": SMOKE_INTERPRETER_SCHEMA,
            "implementation": sys.implementation.name,
            "implementation_version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "cache_tag": sys.implementation.cache_tag,
            "python_implementation": platform.python_implementation(),
            "executable_sha256": digest,
            "byte_count": byte_count,
            "lexical_path_sha256": hashlib.sha256(
                os.path.normcase(str(lexical)).encode("utf-8", "surrogatepass")
            ).hexdigest(),
            "lexical_metadata_sha256": stable_hash(lexical_metadata),
            "lexical_kind": lexical_metadata["kind"],
            "resolved_path_sha256": path_identity,
            "resolved_metadata_sha256": executable_identity.metadata_sha256,
            "isolated_startup": True,
            "isolated_flags": ["-I", "-B"],
        },
        "interpreter_identity",
    )


def validate_smoke_interpreter(
    plan: Mapping[str, Any],
    *,
    lexical_path: str | Path | None = None,
) -> dict[str, Any]:
    validated = validate_plan(plan)
    expected = _validate_interpreter_record(validated.get("interpreter"))
    actual = prepare_smoke_interpreter_binding(lexical_path=lexical_path)
    if actual != expected:
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    return actual


@contextmanager
def pinned_smoke_interpreter(
    plan: Mapping[str, Any],
    *,
    lexical_path: str | Path | None = None,
) -> Iterator[PinnedSmokeInterpreter]:
    """Hold the plan-bound executable target across process creation.

    Windows opens the target without write/delete sharing.  Linux executes the
    already-open descriptor through ``/proc/self/fd``.  Both paths are checked
    again while the descriptor is still held after ``Popen`` returns.
    """

    validated = validate_plan(plan)
    expected = _validate_interpreter_record(validated.get("interpreter"))
    lexical = _lexical_interpreter_path() if lexical_path is None else Path(lexical_path).absolute()
    lexical_metadata = _lexical_interpreter_metadata(lexical)
    if (
        hashlib.sha256(os.path.normcase(str(lexical)).encode("utf-8", "surrogatepass")).hexdigest()
        != expected["lexical_path_sha256"]
        or stable_hash(lexical_metadata) != expected["lexical_metadata_sha256"]
        or lexical_metadata["kind"] != expected["lexical_kind"]
    ):
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    resolved = lexical.resolve(strict=True)
    if (
        hashlib.sha256(os.path.normcase(str(resolved)).encode("utf-8", "surrogatepass")).hexdigest()
        != expected["resolved_path_sha256"]
    ):
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed after smoke preparation; prepare a fresh bundle.",
        )
    try:
        with pin_executable(
            resolved,
            expected_sha256=str(expected["executable_sha256"]),
            expected_size=int(expected["byte_count"]),
            expected_metadata_sha256=str(expected["resolved_metadata_sha256"]),
        ) as pin:
            yield pin
            if stable_hash(_lexical_interpreter_metadata(lexical)) != expected["lexical_metadata_sha256"]:
                raise SmokeBundleError(
                    "smoke_interpreter_changed",
                    "The exact isolated Python interpreter changed during process launch.",
                )
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter_changed",
            "The exact isolated Python interpreter changed during process launch.",
        ) from exc


def verify_pinned_process_image(process: Any, pin: PinnedSmokeInterpreter) -> None:
    """Prove that a newly created process is executing the held target."""

    try:
        verify_process_image(process, pin)
    except PinnedExecutableError as exc:
        raise SmokeBundleError(
            "smoke_interpreter_launch",
            "The isolated Python process image differs from the pinned interpreter.",
        ) from exc


def prepare_smoke_orchestration_code_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    inventory: dict[str, dict[str, Any]] = {}
    for relative in _ORCHESTRATION_CODE_PATHS:
        payload = read_stable_single_link_bytes(
            _fixed_relative(root, relative),
            boundary=root,
            max_bytes=8 * 1024 * 1024,
        )
        inventory[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
    return finalize_identity(
        {
            "schema_version": SMOKE_ORCHESTRATION_CODE_SCHEMA,
            "paths": list(_ORCHESTRATION_CODE_PATHS),
            "inventory": inventory,
            "preflight_sha256": hashlib.sha256(_CHILD_PREFLIGHT_SOURCE.encode("utf-8")).hexdigest(),
            "bootstrap_sha256": stable_hash(
                {
                    "main": _ISOLATED_MAIN_BOOTSTRAP,
                    "worker": _ISOLATED_WORKER_BOOTSTRAP,
                }
            ),
        },
        "orchestration_code_identity",
    )


def validate_smoke_orchestration_code(
    project_root: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_plan(plan)
    expected = _validate_orchestration_code_record(validated.get("orchestration_code"))
    try:
        actual = prepare_smoke_orchestration_code_identity(project_root)
    except (OSError, SmokeBundleError) as exc:
        raise SmokeBundleError(
            "smoke_orchestration_code_changed",
            "The smoke orchestration code is unavailable; prepare a fresh bundle.",
        ) from exc
    if actual != expected:
        raise SmokeBundleError(
            "smoke_orchestration_code_changed",
            "The smoke orchestration code changed after preparation; prepare a fresh bundle.",
        )
    return actual


def validate_bound_training_code_identity(
    project_root: str | Path,
    value: Any,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Rehash every production Python file and reject inventory drift."""

    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    identity = dict(value)
    files = identity.get("files")
    if (
        identity.get("schema_version") != "spritelab_training_code_identity_v4"
        or identity.get("contract") != "all_tracked_production_python_v5_with_untracked_rejection"
        or not isinstance(files, list)
        or not files
        or not SHA256_PATTERN.fullmatch(str(identity.get("sha256") or ""))
        or identity.get("sha256") != expected_sha256
        or stable_hash({key: item for key, item in identity.items() if key != "sha256"}) != identity.get("sha256")
    ):
        raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
    expected: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
        relative = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in expected
            or not relative.startswith("src/spritelab/")
            or not relative.endswith(".py")
            or not SHA256_PATTERN.fullmatch(str(digest or ""))
        ):
            raise SmokeBundleError("smoke_training_code_changed", "The bound training code identity is invalid.")
        _fixed_relative(Path(project_root).resolve(), relative)
        expected[relative] = str(digest)
    actual = _production_python_inventory(Path(project_root).resolve())
    if actual != expected:
        raise SmokeBundleError(
            "smoke_training_code_changed",
            "Tracked or untracked production Python changed after campaign preparation.",
        )
    return identity


def smoke_launch_identity(plan: Mapping[str, Any], device: str) -> str:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    record = dict(validated["configurations"])[normalized]
    binding = _validate_child_environment_binding(record.get("child_environment"))
    return stable_hash(
        {
            "schema_version": "spritelab.training.smoke-launch.v1",
            "smoke_id": validated["smoke_id"],
            "plan_identity": validated["plan_identity"],
            "device": normalized,
            "training_argv": _base_smoke_training_argv(validated, normalized),
            "environment_sha256": binding["environment_sha256"],
            "interpreter_identity": dict(validated["interpreter"])["interpreter_identity"],
            "orchestration_code_identity": dict(validated["orchestration_code"])["orchestration_code_identity"],
            "training_code_identity_sha256": dict(validated["bindings"])["training_code_identity_sha256"],
            "retry_policy": "NEW_BUNDLE_REQUIRED",
        }
    )


def smoke_training_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    return [
        *_base_smoke_training_argv(validated, normalized),
        "--smoke-launch-identity",
        smoke_launch_identity(validated, normalized),
    ]


def smoke_worker_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    validated = validate_plan(plan)
    normalized = _require_device(device)
    return [
        "python",
        "-I",
        "-B",
        "-c",
        _ISOLATED_WORKER_BOOTSTRAP,
        "--smoke-id",
        str(validated["smoke_id"]),
        "--device",
        normalized,
        "--plan-identity",
        str(validated["plan_identity"]),
        "--launch-identity",
        smoke_launch_identity(validated, normalized),
    ]


def file_sha256(path: Path, *, boundary: Path, max_bytes: int = 2 * 1024**3) -> str:
    payload = read_stable_single_link_bytes(path, boundary=boundary, max_bytes=max_bytes)
    return hashlib.sha256(payload).hexdigest()


def finalize_identity(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise SmokeBundleError("identity_field", "The smoke identity payload is malformed.")
    result[field] = stable_hash(result)
    return result


def validate_identity(payload: Mapping[str, Any], field: str) -> None:
    expected = payload.get(field)
    if not SHA256_PATTERN.fullmatch(str(expected or "")):
        raise SmokeBundleError("identity_missing", "The smoke identity is missing or malformed.")
    body = {key: value for key, value in payload.items() if key != field}
    if stable_hash(body) != expected:
        raise SmokeBundleError("identity_changed", "The smoke evidence identity no longer matches its content.")


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    if value.get("schema_version") != SMOKE_PLAN_SCHEMA or value.get("status") != "PREPARED":
        raise SmokeBundleError("smoke_plan_schema", "The server-prepared smoke plan is unavailable.")
    _require_smoke_id(str(value.get("smoke_id") or ""))
    if value.get("scope") != SMOKE_SCOPE or any(value.get(key) is not False for key in FALSE_ELIGIBILITY):
        raise SmokeBundleError("smoke_plan_scope", "The smoke plan is not exclusively exploratory.")
    bindings = value.get("bindings")
    configurations = value.get("configurations")
    if not isinstance(bindings, Mapping) or not isinstance(configurations, Mapping):
        raise SmokeBundleError("smoke_plan_content", "The server-prepared smoke plan is incomplete.")
    _validate_interpreter_record(value.get("interpreter"))
    _validate_orchestration_code_record(value.get("orchestration_code"))
    for key in (
        "candidate_identity_sha256",
        "publication_identity_sha256",
        "activation_manifest_sha256",
        "campaign_config_sha256",
        "campaign_identity_sha256",
        "training_code_identity_sha256",
        "dataset_view_manifest_sha256",
        "split_manifest_sha256",
        "conditioning_vocabulary_sha256",
        "benchmark_manifest_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(bindings.get(key) or "")):
            raise SmokeBundleError("smoke_plan_binding", "A required smoke-plan binding is malformed.")
    training_code = bindings.get("training_code_identity")
    if not isinstance(training_code, Mapping) or training_code.get("sha256") != bindings.get(
        "training_code_identity_sha256"
    ):
        raise SmokeBundleError("smoke_plan_binding", "The full training-code binding is malformed.")
    for device in SMOKE_DEVICES:
        record = configurations.get(device)
        if not isinstance(record, Mapping):
            raise SmokeBundleError("smoke_plan_config", "The smoke plan is missing a device configuration.")
        for key in ("config_sha256", "manifest_sha256"):
            if not SHA256_PATTERN.fullmatch(str(record.get(key) or "")):
                raise SmokeBundleError("smoke_plan_config", "A smoke configuration identity is malformed.")
        _validate_environment_record(record.get("environment"), device)
        _validate_child_environment_binding(record.get("child_environment"))
    validate_identity(value, "plan_identity")
    return value


def load_plan(project_root: str | Path, smoke_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = artifact_bundle_directory(root, smoke_id) / "plan.json"
    value = _read_json(path, boundary=root, max_bytes=16 * 1024 * 1024)
    plan = validate_plan(value)
    if plan["smoke_id"] != smoke_id:
        raise SmokeBundleError("smoke_plan_id", "The smoke plan belongs to a different bundle.")
    return plan


def publish_plan(
    project_root: str | Path,
    plan: Mapping[str, Any],
    *,
    configurations: Mapping[str, bytes],
    manifests: Mapping[str, bytes],
) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    smoke_id = str(validated["smoke_id"])
    parent = ensure_managed_directory(root, ("artifacts", "training", "smokes"))
    files: dict[str, bytes] = {"plan.json": canonical_json_bytes(validated, pretty=True)}
    for device in SMOKE_DEVICES:
        files[f"configs/{device}.json"] = configurations[device]
        files[f"configs/{device}.manifest.json"] = manifests[device]
    return publish_immutable_tree(parent, root=root, final_name=smoke_id, files=files)


def publish_run_container(project_root: str | Path, plan: Mapping[str, Any]) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    parent = ensure_managed_directory(root, ("runs", "v3", "training-smokes"))
    state = {
        "schema_version": SMOKE_RUN_STATE_SCHEMA,
        "smoke_id": validated["smoke_id"],
        "status": "PREPARED",
        "scope": SMOKE_SCOPE,
        "plan_identity": validated["plan_identity"],
        "resumable": False,
        "retry_policy": "NEW_BUNDLE_REQUIRED",
        **FALSE_ELIGIBILITY,
    }
    return publish_immutable_tree(
        parent,
        root=root,
        final_name=str(validated["smoke_id"]),
        files={"state.json": canonical_json_bytes(state, pretty=True)},
    )


def begin_device_run(project_root: str | Path, plan: Mapping[str, Any], device: str) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    parent = run_bundle_directory(root, str(validated["smoke_id"]))
    state = {
        "schema_version": SMOKE_RUN_STATE_SCHEMA,
        "smoke_id": validated["smoke_id"],
        "device": normalized,
        "status": "RUNNING",
        "scope": SMOKE_SCOPE,
        "plan_identity": validated["plan_identity"],
        "resumable": False,
        "retry_policy": "NEW_BUNDLE_REQUIRED",
        **FALSE_ELIGIBILITY,
    }
    return publish_immutable_tree(
        parent,
        root=root,
        final_name=normalized,
        files={"smoke_run_state.json": canonical_json_bytes(state, pretty=True)},
    )


def write_device_receipt(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    *,
    config_sha256_before: str,
    config_sha256_after: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    validate_smoke_interpreter(
        validated,
        lexical_path=environment.get("SPRITELAB_BOUND_INTERPRETER"),
    )
    validate_smoke_environment(root, validated, normalized, environment)
    record = dict(validated["configurations"])[normalized]
    public_environment = _validate_environment_record(record.get("environment"), normalized)
    environment_binding = _validate_child_environment_binding(record.get("child_environment"))
    verify_execution_guards(root, validated)
    verification, _checkpoints = _verify_device_output(root, validated, normalized)
    verify_execution_guards(root, validated)
    receipt = finalize_identity(
        {
            "schema_version": SMOKE_DEVICE_RECEIPT_SCHEMA,
            "smoke_id": validated["smoke_id"],
            "device": normalized,
            "status": "COMPLETE",
            "scope": SMOKE_SCOPE,
            "plan_identity": validated["plan_identity"],
            "launch_identity": smoke_launch_identity(validated, normalized),
            "interpreter": dict(validated["interpreter"]),
            "interpreter_identity": dict(validated["interpreter"])["interpreter_identity"],
            "orchestration_code": dict(validated["orchestration_code"]),
            "orchestration_code_identity": dict(validated["orchestration_code"])["orchestration_code_identity"],
            "config_sha256_before": config_sha256_before,
            "config_sha256_after": config_sha256_after,
            "configuration_unchanged": config_sha256_before == config_sha256_after,
            "environment": public_environment,
            "environment_sha256": environment_binding["environment_sha256"],
            "environment_policy_sha256": stable_hash(environment_binding),
            "verification": verification,
            **FALSE_ELIGIBILITY,
        },
        "receipt_identity",
    )
    if not receipt["configuration_unchanged"]:
        raise SmokeBundleError("smoke_config_changed", "The server-generated smoke configuration changed.")
    output = run_bundle_directory(root, str(validated["smoke_id"])) / normalized
    _write_exclusive(output / "smoke_run_receipt.json", canonical_json_bytes(receipt, pretty=True), boundary=root)
    return receipt


def load_device_receipt(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Load a final receipt without importing Torch or revalidating checkpoints."""

    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    output = run_bundle_directory(root, str(validated["smoke_id"])) / normalized
    receipt = _read_json(output / "smoke_run_receipt.json", boundary=root, max_bytes=16 * 1024 * 1024)
    record = dict(validated["configurations"])[normalized]
    environment = _validate_environment_record(record.get("environment"), normalized)
    environment_binding = _validate_child_environment_binding(record.get("child_environment"))
    if (
        receipt.get("schema_version") != SMOKE_DEVICE_RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE"
        or receipt.get("smoke_id") != validated["smoke_id"]
        or receipt.get("device") != normalized
        or receipt.get("plan_identity") != validated["plan_identity"]
        or receipt.get("launch_identity") != smoke_launch_identity(validated, normalized)
        or receipt.get("interpreter") != validated["interpreter"]
        or receipt.get("interpreter_identity") != dict(validated["interpreter"])["interpreter_identity"]
        or receipt.get("orchestration_code") != validated["orchestration_code"]
        or receipt.get("orchestration_code_identity")
        != dict(validated["orchestration_code"])["orchestration_code_identity"]
        or receipt.get("config_sha256_before") != record.get("config_sha256")
        or receipt.get("config_sha256_after") != record.get("config_sha256")
        or receipt.get("configuration_unchanged") is not True
        or receipt.get("environment") != environment
        or receipt.get("environment_sha256") != environment_binding["environment_sha256"]
        or receipt.get("environment_policy_sha256") != stable_hash(environment_binding)
        or any(receipt.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("smoke_receipt_invalid", "A smoke-device completion receipt is invalid.")
    validate_identity(receipt, "receipt_identity")
    return receipt


def verify_complete_bundle(project_root: str | Path, plan: Mapping[str, Any]) -> VerifiedSmokeBundle:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    verify_execution_guards(root, validated)
    runs: dict[str, Any] = {}
    selected: list[VerifiedSmokeCheckpoint] = []
    for device in SMOKE_DEVICES:
        output = run_bundle_directory(root, str(validated["smoke_id"])) / device
        try:
            receipt = load_device_receipt(root, validated, device)
        except OSError as exc:
            raise SmokeBundleError("smoke_output_incomplete", "The two-step smoke output is incomplete.") from exc
        verification, checkpoints = _verify_device_output(root, validated, device)
        if receipt.get("verification") != verification:
            raise SmokeBundleError("smoke_receipt_stale", "Smoke output changed after its completion receipt.")
        receipt_path = output / "smoke_run_receipt.json"
        runs[device] = {
            **verification,
            "environment": dict(receipt["environment"]),
            "environment_sha256": receipt["environment_sha256"],
            "receipt_identity": receipt["receipt_identity"],
            "receipt_sha256": file_sha256(receipt_path, boundary=root, max_bytes=16 * 1024 * 1024),
        }
        if device == "cuda":
            selected.extend(checkpoints)
    verify_execution_guards(root, validated)
    evidence = finalize_identity(
        {
            "schema_version": SMOKE_EVIDENCE_SCHEMA,
            "smoke_id": validated["smoke_id"],
            "scope": SMOKE_SCOPE,
            "status": SMOKE_STATUS,
            "plan_identity": validated["plan_identity"],
            "bindings": dict(validated["bindings"]),
            "source": dict(validated["source"]),
            "derivation": dict(validated["derivation"]),
            "orchestration_code": dict(validated["orchestration_code"]),
            "config_sha256_before": validated["config_sha256_before"],
            "full_campaign_output_roots": list(validated["full_campaign_output_roots"]),
            "runs": runs,
            **FALSE_ELIGIBILITY,
        },
        "evidence_identity",
    )
    return VerifiedSmokeBundle(evidence=evidence, checkpoints=tuple(selected))


def publish_evidence(project_root: str | Path, evidence: Mapping[str, Any]) -> Path:
    root = Path(project_root).resolve()
    value = dict(evidence)
    validate_identity(value, "evidence_identity")
    smoke_id = str(value.get("smoke_id") or "")
    _require_smoke_id(smoke_id)
    target = artifact_bundle_directory(root, smoke_id) / "smoke_evidence.json"
    _write_exclusive(target, canonical_json_bytes(value, pretty=True), boundary=root)
    return target


def publish_playground_snapshot(
    project_root: str | Path,
    bundle: VerifiedSmokeBundle,
) -> tuple[Path, dict[str, Any]]:
    """Snapshot the verified CUDA live+EMA pair into a Playground-only root."""

    root = Path(project_root).resolve()
    evidence = dict(bundle.evidence)
    validate_identity(evidence, "evidence_identity")
    content_id = f"exploratory-{evidence['evidence_identity'][:24]}"
    parent = ensure_managed_directory(root, ("runs", "v3", "playground", "exploratory-checkpoints"))
    checkpoint_rows = [
        {
            "weights": checkpoint.weights,
            "path": f"checkpoint_step_000002{'_ema' if checkpoint.weights == 'ema' else ''}.pt",
            "sha256": checkpoint.sha256,
            "byte_count": checkpoint.byte_count,
            "step": checkpoint.step,
            "variant": checkpoint.variant,
        }
        for checkpoint in sorted(bundle.checkpoints, key=lambda item: item.weights)
    ]
    evidence_path = artifact_bundle_directory(root, str(evidence["smoke_id"])) / "smoke_evidence.json"
    evidence_sha256 = file_sha256(evidence_path, boundary=root, max_bytes=64 * 1024 * 1024)
    registration = finalize_identity(
        {
            "schema_version": EXPLORATORY_REGISTRATION_SCHEMA,
            "content_id": content_id,
            "smoke_id": evidence["smoke_id"],
            "status": SMOKE_STATUS,
            "purpose": "exploratory",
            "scope": SMOKE_SCOPE,
            "evidence_identity": evidence["evidence_identity"],
            "smoke_evidence_sha256": evidence_sha256,
            "plan_identity": evidence["plan_identity"],
            "bindings": dict(evidence["bindings"]),
            "checkpoints": checkpoint_rows,
            **FALSE_ELIGIBILITY,
        },
        "registration_identity",
    )
    final = parent / content_id
    if os.path.lexists(final):
        existing = load_playground_registration(root, content_id)
        if existing != registration:
            raise SmokeBundleError(
                "registration_conflict", "An exploratory checkpoint registration already has different content."
            )
        return final, existing
    staging_name: str | None = None
    staging_identity: OwnedFileIdentity | None = None
    published = False
    with anchored_directory(parent, root) as anchor:
        staging_name, staging_identity = anchor.mkdir_unique(f".{content_id}-staging-")
    staging = parent / staging_name
    try:
        by_weight = {checkpoint.weights: checkpoint for checkpoint in bundle.checkpoints}
        for row in checkpoint_rows:
            checkpoint = by_weight[str(row["weights"])]
            copy_stable_single_link_file(
                checkpoint.path,
                staging / str(row["path"]),
                source_boundary=root,
                destination_boundary=root,
                expected_sha256=checkpoint.sha256,
                expected_bytes=checkpoint.byte_count,
            )
        _write_exclusive(
            staging / "registration.json",
            canonical_json_bytes(registration, pretty=True),
            boundary=root,
        )
        with anchored_directory(parent, root) as anchor:
            if staging_name is None or staging_identity is None:
                raise SmokeBundleError("registration_state", "Checkpoint snapshot staging identity is missing.")
            if not staging_identity.matches(anchor.lstat(staging_name)):
                raise SmokeBundleError("registration_changed", "Checkpoint snapshot staging changed.")
            try:
                anchor.rename(staging_name, content_id, replace=False)
            except FileExistsError:
                anchor.quarantine_if_owned(staging_name, staging_identity, prefix=f".{content_id}-residue-")
                existing = load_playground_registration(root, content_id)
                if existing != registration:
                    raise SmokeBundleError(
                        "registration_conflict", "Concurrent checkpoint registration conflicted."
                    ) from None
                return final, existing
            published = True
            if not staging_identity.matches(anchor.lstat(content_id)):
                raise SmokeBundleError("registration_changed", "Checkpoint snapshot publication changed.")
    except BaseException:
        if not published and staging_name is not None and staging_identity is not None:
            with anchored_directory(parent, root) as anchor:
                anchor.quarantine_if_owned(staging_name, staging_identity, prefix=f".{content_id}-residue-")
        raise
    return final, registration


def load_playground_registration(project_root: str | Path, content_id: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not re.fullmatch(r"exploratory-[0-9a-f]{24}", str(content_id)):
        raise SmokeBundleError("registration_id", "The exploratory checkpoint registration ID is invalid.")
    path = root / "runs" / "v3" / "playground" / "exploratory-checkpoints" / content_id / "registration.json"
    value = _read_json(path, boundary=root, max_bytes=16 * 1024 * 1024)
    if (
        value.get("schema_version") != EXPLORATORY_REGISTRATION_SCHEMA
        or value.get("content_id") != content_id
        or value.get("status") != SMOKE_STATUS
        or value.get("purpose") != "exploratory"
        or value.get("scope") != SMOKE_SCOPE
        or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
    ):
        raise SmokeBundleError("registration_invalid", "The exploratory checkpoint registration is invalid.")
    validate_identity(value, "registration_identity")
    return value


def write_exclusive_bytes(path: Path, content: bytes, *, boundary: Path) -> None:
    """Publish one new direct-child file without replacement."""

    _write_exclusive(path, content, boundary=boundary)


def expected_config_path(project_root: str | Path, plan: Mapping[str, Any], device: str) -> Path:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    relative = str(dict(validated["configurations"])[normalized].get("config_path") or "")
    return _fixed_relative(root, relative)


def expected_manifest(project_root: str | Path, plan: Mapping[str, Any], device: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    relative = str(dict(validated["configurations"])[normalized].get("manifest_path") or "")
    path = _fixed_relative(root, relative)
    value = _read_json(path, boundary=root, max_bytes=32 * 1024 * 1024)
    expected_hash = str(dict(validated["configurations"])[normalized]["manifest_sha256"])
    if hashlib.sha256(canonical_json_bytes(value, pretty=True)).hexdigest() != expected_hash:
        raise SmokeBundleError("smoke_manifest_changed", "The server-generated smoke manifest changed.")
    return value


def validate_cli_configuration(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    config_path: Path,
) -> tuple[str, dict[str, Any]]:
    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    normalized = _require_device(device)
    expected = expected_config_path(root, validated, normalized)
    try:
        actual = config_path.resolve(strict=True)
    except OSError as exc:
        raise SmokeBundleError("smoke_config_missing", "The server-generated smoke configuration is missing.") from exc
    if actual != expected:
        raise SmokeBundleError("smoke_config_path", "Only the server-generated smoke configuration is accepted.")
    payload = read_stable_single_link_bytes(actual, boundary=root, max_bytes=16 * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    record = dict(validated["configurations"])[normalized]
    if digest != record["config_sha256"]:
        raise SmokeBundleError("smoke_config_changed", "The server-generated smoke configuration changed.")
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeBundleError("smoke_config_invalid", "The server-generated smoke configuration is invalid.") from exc
    if not isinstance(config, dict):
        raise SmokeBundleError("smoke_config_invalid", "The server-generated smoke configuration is invalid.")
    runtime = config.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or str(runtime.get("device")) != normalized
        or str(runtime.get("determinism")) != "strict"
        or str(runtime.get("out_dir")) != record.get("output_path")
    ):
        raise SmokeBundleError("smoke_config_semantics", "The smoke configuration has unsafe runtime semantics.")
    return digest, config


def validate_smoke_environment(
    project_root: str | Path,
    plan: Mapping[str, Any],
    device: str,
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Verify the exact security-relevant environment bound by a smoke plan."""

    expected = build_smoke_child_environment(
        project_root,
        plan,
        device,
        inherited_environment=environment,
    )
    if dict(environment) != expected:
        raise SmokeBundleError(
            "smoke_environment_changed",
            "The smoke runtime environment differs from the server-prepared plan.",
        )
    return expected


def verify_execution_guards(project_root: str | Path, plan: Mapping[str, Any]) -> None:
    """Recheck the real config and absent production roots without mutation."""

    root = Path(project_root).resolve()
    validated = validate_plan(plan)
    bindings = dict(validated["bindings"])
    validate_smoke_orchestration_code(root, validated)
    validate_bound_training_code_identity(
        root,
        bindings.get("training_code_identity"),
        expected_sha256=str(bindings["training_code_identity_sha256"]),
    )
    config_path = root / "spritelab.yaml"
    if file_sha256(config_path, boundary=root, max_bytes=16 * 1024 * 1024) != validated.get("config_sha256_before"):
        raise SmokeBundleError("project_config_changed", "The real project configuration changed during the smoke.")
    for sentinel in validated.get("full_campaign_output_roots") or ():
        if not isinstance(sentinel, Mapping) or sentinel.get("state") != "ABSENT":
            raise SmokeBundleError("campaign_sentinel", "A campaign output-root sentinel is invalid.")
        relative = str(sentinel.get("relative_path") or "")
        path = _fixed_relative(root, relative)
        if not anchored_path_is_absent(path, root):
            raise SmokeBundleError("campaign_output_changed", "A full-campaign output root changed during the smoke.")


def _verify_device_output(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, Any], tuple[VerifiedSmokeCheckpoint, ...]]:
    output = run_bundle_directory(root, str(plan["smoke_id"])) / device
    inventory = flat_file_inventory(output, boundary=root, exclude=("smoke_run_receipt.json",))
    required = {
        "smoke_run_state.json",
        "config.json",
        "train_report.json",
        "train_metrics.jsonl",
        "checkpoint_step_000002.pt",
        "checkpoint_step_000002_ema.pt",
    }
    if device == "cuda":
        required.add("cuda_determinism_qualification.json")
    if not required <= set(inventory):
        raise SmokeBundleError("smoke_output_incomplete", "The two-step smoke output is incomplete.")
    report = _read_json(output / "train_report.json", boundary=root, max_bytes=64 * 1024 * 1024)
    determinism = report.get("determinism")
    if (
        report.get("model_type") != "generator_challenger"
        or report.get("steps_completed") != 2
        or report.get("max_steps") != 2
        or str(report.get("device")) != device
        or not isinstance(determinism, Mapping)
        or determinism.get("mode") != "strict"
        or determinism.get("qualified") is not True
        or determinism.get("issues") not in ([], ())
    ):
        raise SmokeBundleError("smoke_report_invalid", "The smoke report is not a qualified strict two-step run.")
    _require_finite_tree(report)
    metrics = _read_metrics(output / "train_metrics.jsonl", boundary=root)
    if not metrics or metrics[-1].get("step") != 2:
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metrics do not contain a finite final step 2.")
    manifest = expected_manifest(root, plan, device)
    checkpoint_values: dict[str, Mapping[str, Any]] = {}
    for name, record in inventory.items():
        if not name.startswith("checkpoint") or not name.endswith(".pt"):
            continue
        from spritelab.training.checkpoint_io import load_checkpoint

        checkpoint_values[name] = load_checkpoint(output / name, expected_sha256=str(record["sha256"]))
        value = checkpoint_values[name]
        if (
            value.get("model_type") != "generator_challenger"
            or value.get("experiment_manifest") != manifest
            or value.get("step") not in (1, 2)
            or value.get("global_step") != value.get("step")
        ):
            raise SmokeBundleError("smoke_checkpoint_invalid", "A smoke checkpoint failed safe validation.")
    selected: list[VerifiedSmokeCheckpoint] = []
    for weights, name, variant, ema in (
        ("live", "checkpoint_step_000002.pt", "step", False),
        ("ema", "checkpoint_step_000002_ema.pt", "step_ema", True),
    ):
        value = checkpoint_values.get(name)
        record = inventory[name]
        if (
            not isinstance(value, Mapping)
            or value.get("step") != 2
            or value.get("global_step") != 2
            or value.get("checkpoint_variant") != variant
            or value.get("ema_weights") is not ema
        ):
            raise SmokeBundleError("smoke_checkpoint_variant", "The step-2 live/EMA checkpoint pair is invalid.")
        selected.append(
            VerifiedSmokeCheckpoint(
                weights=weights,
                path=output / name,
                sha256=str(record["sha256"]),
                byte_count=int(record["byte_count"]),
                step=2,
                variant=variant,
            )
        )
    qualification = None
    if device == "cuda":
        qualification = _read_json(
            output / "cuda_determinism_qualification.json",
            boundary=root,
            max_bytes=16 * 1024 * 1024,
        )
        if (
            qualification.get("qualified") is not True
            or qualification.get("mode") != "strict"
            or qualification.get("device") != "cuda"
            or qualification.get("repeated_forward_backward_bit_exact") is not True
            or qualification.get("resume_bit_exact") is not True
        ):
            raise SmokeBundleError("cuda_qualification_invalid", "Strict CUDA determinism qualification is invalid.")
    return (
        {
            "status": "COMPLETE",
            "steps_completed": 2,
            "device": device,
            "determinism": "strict",
            "determinism_qualified": True,
            "report_sha256": inventory["train_report.json"]["sha256"],
            "metrics_sha256": inventory["train_metrics.jsonl"]["sha256"],
            "output_inventory": inventory,
            "output_inventory_sha256": stable_hash(inventory),
            "checkpoints": [
                {
                    "weights": item.weights,
                    "sha256": item.sha256,
                    "byte_count": item.byte_count,
                    "step": item.step,
                    "variant": item.variant,
                }
                for item in selected
            ],
            "qualification": qualification,
            **FALSE_ELIGIBILITY,
        },
        tuple(selected),
    )


def ensure_managed_directory(
    start: Path,
    parts: Sequence[str],
    *,
    boundary: Path | None = None,
) -> Path:
    approved_root = (boundary or start).resolve()
    current = start
    with ExitStack() as stack:
        anchor = stack.enter_context(anchored_directory(current, approved_root))
        for part in parts:
            if not part or Path(part).name != part or part in {".", ".."}:
                raise SmokeBundleError("managed_path", "The fixed smoke storage path is invalid.")
            if anchor.lexists(part):
                _require_directory_metadata(anchor.lstat(part))
            else:
                try:
                    identity = anchor.mkdir(part)
                except FileExistsError:
                    _require_directory_metadata(anchor.lstat(part))
                else:
                    if not identity.matches(anchor.lstat(part)):
                        raise SmokeBundleError("managed_path_changed", "Fixed smoke storage changed during creation.")
            current = current / part
            anchor = stack.enter_context(anchor.open_directory(part))
    return current


def publish_immutable_tree(
    parent: Path,
    *,
    root: Path,
    final_name: str,
    files: Mapping[str, bytes],
) -> Path:
    boundary = root.resolve()
    if not final_name or Path(final_name).name != final_name:
        raise SmokeBundleError("publication_name", "The fixed smoke publication name is invalid.")
    staging_name: str | None = None
    staging_identity: OwnedFileIdentity | None = None
    published = False
    with anchored_directory(parent, boundary) as anchor:
        if anchor.lexists(final_name):
            raise SmokeBundleError("publication_exists", "That immutable smoke publication already exists.")
        staging_name, staging_identity = anchor.mkdir_unique(f".{final_name}-staging-")
    staging = parent / staging_name
    try:
        for relative, content in sorted(files.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or pure.as_posix() != relative or any(part in {"", ".", ".."} for part in pure.parts):
                raise SmokeBundleError("publication_inventory", "The smoke publication inventory is invalid.")
            destination_parent = (
                ensure_managed_directory(staging, pure.parts[:-1], boundary=boundary)
                if len(pure.parts) > 1
                else staging
            )
            _write_exclusive(destination_parent / pure.name, content, boundary=boundary)
        with anchored_directory(parent, boundary) as anchor:
            if staging_name is None or staging_identity is None:
                raise SmokeBundleError("publication_state", "Smoke staging identity is missing.")
            if not staging_identity.matches(anchor.lstat(staging_name)):
                raise SmokeBundleError("publication_changed", "Smoke staging changed before publication.")
            try:
                anchor.rename(staging_name, final_name, replace=False)
            except FileExistsError as exc:
                raise SmokeBundleError(
                    "publication_exists",
                    "That immutable smoke publication already exists.",
                ) from exc
            published = True
            if not staging_identity.matches(anchor.lstat(final_name)):
                raise SmokeBundleError("publication_changed", "Smoke publication identity changed.")
    except BaseException:
        if not published and staging_name is not None and staging_identity is not None:
            with anchored_directory(parent, boundary) as anchor:
                anchor.quarantine_if_owned(staging_name, staging_identity, prefix=f".{final_name}-residue-")
        raise
    return parent / final_name


def read_stable_single_link_bytes(path: Path, *, boundary: Path, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with anchored_directory(path.parent, boundary) as anchor:
        return _read_from_anchor(anchor, path.name, max_bytes=max_bytes)


def flat_file_inventory(
    directory: Path,
    *,
    boundary: Path,
    exclude: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    excluded = set(exclude)
    inventory: dict[str, dict[str, Any]] = {}
    with anchored_directory(directory, boundary) as anchor:
        names = anchor.names()
        for name in names:
            if name in excluded:
                continue
            payload = _read_from_anchor(anchor, name, max_bytes=2 * 1024**3)
            inventory[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
    return inventory


def copy_stable_single_link_file(
    source: Path,
    destination: Path,
    *,
    source_boundary: Path,
    destination_boundary: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha256) or expected_bytes < 0:
        raise SmokeBundleError("snapshot_identity", "The checkpoint snapshot identity is invalid.")
    with (
        anchored_directory(source.parent, source_boundary) as source_anchor,
        anchored_directory(destination.parent, destination_boundary) as destination_anchor,
    ):
        source_fd = source_anchor.open_file(source.name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        destination_fd = -1
        try:
            source_before = os.fstat(source_fd)
            _require_file_metadata(source_before, max_bytes=2 * 1024**3)
            if _metadata_identity(source_before) != _metadata_identity(source_anchor.lstat(source.name)):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed before snapshotting.")
            destination_fd = destination_anchor.open_file(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
            )
            destination_identity = OwnedFileIdentity.from_stat(os.fstat(destination_fd))
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
            source_after = os.fstat(source_fd)
            if (
                _metadata_identity(source_before) != _metadata_identity(source_after)
                or _metadata_identity(source_after) != _metadata_identity(source_anchor.lstat(source.name))
                or digest.hexdigest() != expected_sha256
                or byte_count != expected_bytes
            ):
                raise SmokeBundleError("snapshot_source_changed", "The checkpoint changed during snapshotting.")
            destination_after = os.fstat(destination_fd)
            if (
                not destination_identity.matches(destination_after)
                or int(getattr(destination_after, "st_nlink", 1)) != 1
                or destination_after.st_size != expected_bytes
                or not destination_identity.matches(destination_anchor.lstat(destination.name))
            ):
                raise SmokeBundleError("snapshot_destination_changed", "The checkpoint snapshot is unsafe.")
        finally:
            os.close(source_fd)
            if destination_fd >= 0:
                os.close(destination_fd)


def _write_exclusive(path: Path, content: bytes, *, boundary: Path) -> None:
    with anchored_directory(path.parent, boundary) as anchor:
        if anchor.lexists(path.name):
            raise FileExistsError(f"refusing to overwrite immutable smoke artifact: {path.name}")
        temporary = f".{path.name}.partial-{secrets.token_hex(12)}"
        descriptor = anchor.open_file(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        published = False
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or int(getattr(metadata, "st_nlink", 1)) != 1
                or not identity.matches(anchor.lstat(temporary))
            ):
                raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke write changed unexpectedly.")
            anchor.rename(temporary, path.name, replace=False)
            published = True
            if not identity.matches(anchor.lstat(path.name)):
                raise SmokeBundleError("exclusive_write_changed", "An exclusive smoke publication changed.")
        except BaseException:
            if published:
                anchor.quarantine_if_owned(path.name, identity, prefix=f".{path.name}.residue-")
            else:
                anchor.quarantine_if_owned(temporary, identity, prefix=f".{path.name}.residue-")
            raise
        finally:
            os.close(descriptor)


def _read_json(path: Path, *, boundary: Path, max_bytes: int) -> dict[str, Any]:
    payload = read_stable_single_link_bytes(path, boundary=boundary, max_bytes=max_bytes)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.") from exc
    if not isinstance(value, dict):
        raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.")
    return value


def _read_metrics(path: Path, *, boundary: Path) -> list[dict[str, Any]]:
    payload = read_stable_single_link_bytes(path, boundary=boundary, max_bytes=64 * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
        for line in text.splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            _require_finite_tree(value)
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metrics are malformed or non-finite.") from exc
    steps = [row.get("step") for row in rows]
    if any(type(step) is not int for step in steps) or steps != sorted(set(steps)):
        raise SmokeBundleError("smoke_metrics_invalid", "Smoke metric steps are not strictly ordered.")
    return rows


def _require_finite_tree(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SmokeBundleError("smoke_nonfinite", "Smoke evidence contains a non-finite metric.")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_tree(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_tree(child)


def _fixed_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.")
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SmokeBundleError("smoke_relative_path", "A server-owned smoke reference is invalid.") from exc
    return candidate


@contextmanager
def anchored_directory(path: Path, boundary: Path) -> Any:
    """Walk to ``path`` from one held project-root anchor without pathname reopening."""

    root = boundary.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SmokeBundleError("anchored_path", "A fixed smoke path is outside this project.") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise SmokeBundleError("anchored_path", "A fixed smoke path is invalid.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts:
            anchor = stack.enter_context(anchor.open_directory(part))
        yield anchor


def anchored_path_is_absent(path: Path, boundary: Path) -> bool:
    """Prove absence by walking from a held root handle; reject linked seams."""

    root = boundary.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SmokeBundleError("anchored_path", "A fixed smoke path is outside this project.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SmokeBundleError("anchored_path", "A fixed smoke path is invalid.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts[:-1]:
            if not anchor.lexists(part):
                return True
            _require_directory_metadata(anchor.lstat(part))
            anchor = stack.enter_context(anchor.open_directory(part))
        return not anchor.lexists(relative.parts[-1])


def _production_python_inventory(root: Path) -> dict[str, str]:
    source = root / "src" / "spritelab"
    inventory: dict[str, str] = {}

    def walk(anchor: AnchoredDirectory, prefix: PurePosixPath) -> None:
        for name in anchor.names():
            metadata = anchor.lstat(name)
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
                raise SmokeBundleError(
                    "smoke_training_code_changed",
                    "Production Python inventory crosses an unsafe filesystem seam.",
                )
            relative = prefix / name
            if stat.S_ISDIR(metadata.st_mode):
                with anchor.open_directory(name) as child:
                    walk(child, relative)
            elif name.endswith(".py"):
                payload = _read_from_anchor(anchor, name, max_bytes=8 * 1024 * 1024)
                inventory[relative.as_posix()] = hashlib.sha256(payload).hexdigest()

    with anchored_directory(source, root) as anchor:
        walk(anchor, PurePosixPath("src/spritelab"))
    return dict(sorted(inventory.items()))


def _read_from_anchor(anchor: AnchoredDirectory, name: str, *, max_bytes: int) -> bytes:
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        before = os.fstat(descriptor)
        _require_file_metadata(before, max_bytes=max_bytes)
        path_before = anchor.lstat(name)
        if _metadata_identity(before) != _metadata_identity(path_before):
            raise SmokeBundleError("file_changed", "A smoke artifact changed while it was opened.")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise SmokeBundleError("file_too_large", "A smoke artifact exceeds its size limit.")
        after = os.fstat(descriptor)
        path_after = anchor.lstat(name)
        if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(after) != _metadata_identity(
            path_after
        ):
            raise SmokeBundleError("file_changed", "A smoke artifact changed while it was read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_smoke_id(smoke_id: str) -> str:
    if not SMOKE_ID_PATTERN.fullmatch(str(smoke_id)):
        raise SmokeBundleError("smoke_id", "The exploratory smoke bundle ID is invalid.")
    return smoke_id


def _require_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SMOKE_DEVICES:
        raise SmokeBundleError("smoke_device", "The smoke device must be cpu or cuda.")
    return normalized


def _validate_environment_record(value: Any, device: str) -> dict[str, str]:
    expected = _public_smoke_environment(device)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise SmokeBundleError("smoke_environment", "The smoke environment binding is invalid.")
    return expected


def _public_smoke_environment(device: str) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "SPRITELAB_PROGRESS": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if device == "cuda":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return dict(sorted(environment.items()))


def _allowed_inherited_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if os.name != "nt":
        return {}
    normalized = {str(key).upper(): str(value) for key, value in environment.items()}
    return {key: normalized[key] for key in ("SYSTEMROOT", "WINDIR") if normalized.get(key)}


def _compose_child_environment(
    public: Mapping[str, str],
    inherited: Mapping[str, str],
    temporary: Path,
    import_paths: Sequence[str],
    interpreter: Path,
) -> dict[str, str]:
    result = {str(key): str(value) for key, value in inherited.items()}
    result.update({str(key): str(value) for key, value in public.items()})
    temporary_value = str(temporary)
    result.update(dict.fromkeys(_SANDBOXED_ENVIRONMENT_PATHS, temporary_value))
    result["SPRITELAB_ISOLATED_PATHS"] = os.pathsep.join(import_paths)
    result["SPRITELAB_BOUND_INTERPRETER"] = str(interpreter)
    return dict(sorted(result.items()))


def _isolated_import_paths(root: Path) -> list[str]:
    candidates = [root / "src"]
    for value in sys.path:
        if not value:
            continue
        try:
            candidate = Path(value).resolve()
        except OSError:
            continue
        if candidate.name.casefold() in {"site-packages", "dist-packages"}:
            candidates.append(candidate)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate)
        key = os.path.normcase(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _validate_child_environment_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_environment", "The smoke child-environment binding is invalid.")
    result = dict(value)
    inherited_names = result.get("inherited_names")
    sandboxed = result.get("sandboxed_path_variables")
    if (
        result.get("schema_version") != SMOKE_CHILD_ENVIRONMENT_SCHEMA
        or not isinstance(inherited_names, list)
        or any(not isinstance(name, str) for name in inherited_names)
        or inherited_names != sorted(set(inherited_names))
        or any(name not in {"SYSTEMROOT", "WINDIR"} for name in inherited_names)
        or not isinstance(result.get("temporary_root"), str)
        or sandboxed != list(_SANDBOXED_ENVIRONMENT_PATHS)
        or type(result.get("isolated_import_path_count")) is not int
        or int(result["isolated_import_path_count"]) < 1
        or not SHA256_PATTERN.fullmatch(str(result.get("isolated_import_paths_sha256") or ""))
        or not SHA256_PATTERN.fullmatch(str(result.get("environment_sha256") or ""))
    ):
        raise SmokeBundleError("smoke_environment", "The smoke child-environment binding is invalid.")
    return result


def _base_smoke_training_argv(plan: Mapping[str, Any], device: str) -> list[str]:
    record = dict(plan["configurations"])[device]
    return [
        "python",
        "-I",
        "-B",
        "-c",
        _ISOLATED_MAIN_BOOTSTRAP,
        "train",
        "experiment",
        "run",
        "--config",
        str(record["config_path"]),
        "--smoke",
        "--smoke-bundle-id",
        str(plan["smoke_id"]),
        "--smoke-device",
        device,
        "--smoke-plan-identity",
        str(plan["plan_identity"]),
    ]


def _validate_interpreter_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_interpreter", "The smoke interpreter binding is invalid.")
    result = dict(value)
    version = result.get("implementation_version")
    if (
        result.get("schema_version") != SMOKE_INTERPRETER_SCHEMA
        or not isinstance(result.get("implementation"), str)
        or not isinstance(version, list)
        or len(version) != 3
        or any(type(part) is not int or part < 0 for part in version)
        or not isinstance(result.get("cache_tag"), (str, type(None)))
        or not isinstance(result.get("python_implementation"), str)
        or not SHA256_PATTERN.fullmatch(str(result.get("executable_sha256") or ""))
        or type(result.get("byte_count")) is not int
        or int(result["byte_count"]) <= 0
        or not SHA256_PATTERN.fullmatch(str(result.get("lexical_path_sha256") or ""))
        or not SHA256_PATTERN.fullmatch(str(result.get("lexical_metadata_sha256") or ""))
        or result.get("lexical_kind") not in {"regular", "symlink", "reparse"}
        or not SHA256_PATTERN.fullmatch(str(result.get("resolved_path_sha256") or ""))
        or not SHA256_PATTERN.fullmatch(str(result.get("resolved_metadata_sha256") or ""))
        or result.get("isolated_startup") is not True
        or result.get("isolated_flags") != ["-I", "-B"]
    ):
        raise SmokeBundleError("smoke_interpreter", "The smoke interpreter binding is invalid.")
    validate_identity(result, "interpreter_identity")
    return result


def _validate_orchestration_code_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    result = dict(value)
    inventory = result.get("inventory")
    if (
        result.get("schema_version") != SMOKE_ORCHESTRATION_CODE_SCHEMA
        or result.get("paths") != list(_ORCHESTRATION_CODE_PATHS)
        or not isinstance(inventory, Mapping)
        or set(inventory) != set(_ORCHESTRATION_CODE_PATHS)
        or not SHA256_PATTERN.fullmatch(str(result.get("preflight_sha256") or ""))
        or not SHA256_PATTERN.fullmatch(str(result.get("bootstrap_sha256") or ""))
    ):
        raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    for relative in _ORCHESTRATION_CODE_PATHS:
        record = inventory.get(relative)
        if (
            not isinstance(record, Mapping)
            or not SHA256_PATTERN.fullmatch(str(record.get("sha256") or ""))
            or type(record.get("byte_count")) is not int
            or int(record["byte_count"]) <= 0
        ):
            raise SmokeBundleError("smoke_orchestration_code", "The smoke orchestration code identity is invalid.")
    validate_identity(result, "orchestration_code_identity")
    return result


def _lexical_interpreter_path() -> Path:
    return Path(os.path.abspath(sys.executable))


def _lexical_interpreter_metadata(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        target = os.readlink(path)
        link_target_sha256: str | None = hashlib.sha256(target.encode("utf-8", "surrogatepass")).hexdigest()
    elif attributes & reparse:
        kind = "reparse"
        link_target_sha256 = hashlib.sha256(str(path.resolve(strict=True)).encode("utf-8")).hexdigest()
    elif stat.S_ISREG(metadata.st_mode):
        kind = "regular"
        link_target_sha256 = None
    else:
        raise SmokeBundleError("smoke_interpreter", "The isolated Python launcher is unsafe.")
    return {
        "kind": kind,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(stat.S_IFMT(metadata.st_mode)),
        "size": int(metadata.st_size),
        "mtime_ns": int(getattr(metadata, "st_mtime_ns", 0)),
        "reparse_attributes": attributes & reparse,
        "link_target_sha256": link_target_sha256,
    }


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        int(getattr(metadata, "st_nlink", 1)),
        getattr(metadata, "st_mtime_ns", None),
    )


def _require_file_metadata(metadata: os.stat_result, *, max_bytes: int) -> None:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or attributes & reparse
        or int(getattr(metadata, "st_nlink", 1)) != 1
        or metadata.st_size < 0
        or metadata.st_size > max_bytes
    ):
        raise SmokeBundleError("smoke_file_unsafe", "A smoke artifact is not a bounded single-link file.")


def _require_directory_metadata(metadata: os.stat_result) -> None:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
        raise SmokeBundleError("smoke_directory_unsafe", "A fixed smoke directory is unsafe.")


__all__ = [
    "EXPLORATORY_REGISTRATION_SCHEMA",
    "FALSE_ELIGIBILITY",
    "SMOKE_CHILD_ENVIRONMENT_SCHEMA",
    "SMOKE_DEVICE_RECEIPT_SCHEMA",
    "SMOKE_EVIDENCE_SCHEMA",
    "SMOKE_INTERPRETER_SCHEMA",
    "SMOKE_ORCHESTRATION_CODE_SCHEMA",
    "SMOKE_PLAN_SCHEMA",
    "SMOKE_SCOPE",
    "SMOKE_STATUS",
    "PinnedSmokeInterpreter",
    "SmokeBundleError",
    "VerifiedSmokeBundle",
    "VerifiedSmokeCheckpoint",
    "anchored_directory",
    "anchored_path_is_absent",
    "artifact_bundle_directory",
    "begin_device_run",
    "build_smoke_child_environment",
    "canonical_json_bytes",
    "copy_stable_single_link_file",
    "ensure_managed_directory",
    "expected_config_path",
    "expected_manifest",
    "file_sha256",
    "finalize_identity",
    "flat_file_inventory",
    "load_device_receipt",
    "load_plan",
    "load_playground_registration",
    "pinned_smoke_interpreter",
    "prepare_smoke_environment_binding",
    "prepare_smoke_interpreter_binding",
    "prepare_smoke_orchestration_code_identity",
    "publish_evidence",
    "publish_immutable_tree",
    "publish_plan",
    "publish_playground_snapshot",
    "publish_run_container",
    "read_stable_single_link_bytes",
    "run_bundle_directory",
    "smoke_id_for_campaign",
    "smoke_launch_identity",
    "smoke_training_argv",
    "smoke_worker_argv",
    "stable_hash",
    "validate_bound_training_code_identity",
    "validate_cli_configuration",
    "validate_identity",
    "validate_plan",
    "validate_smoke_environment",
    "validate_smoke_interpreter",
    "validate_smoke_orchestration_code",
    "verify_complete_bundle",
    "verify_execution_guards",
    "verify_pinned_process_image",
    "write_device_receipt",
    "write_exclusive_bytes",
]

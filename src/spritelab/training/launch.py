"""Authoritative, CPU-only preparation and verification for training launches.

Every training process or remote transport boundary consumes a receipt produced
here.  Product status projections and adapter-supplied identity strings are not
validation evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import secrets
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    EVENT_HISTORY_TRANSACTION_FILENAME,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    MIGRATION_EVIDENCE_SCHEMA,
    EventMigrationState,
    EventMigrationVerification,
)
from spritelab.training.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    CampaignResumeError,
    CampaignValidationError,
    _CampaignFilesystemSnapshot,
    _capture_anchored_campaign_filesystem_snapshot,
    _capture_fresh_campaign_filesystem_snapshot,
    audit_resume,
    canonical_json,
    is_concrete_hash,
    plan_campaign,
    stable_hash,
    validate_campaign,
)
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)

TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION = "spritelab_training_launch_receipt_v4"
EVENT_HISTORY_ORIGIN_RECEIPT_STATES = frozenset({"native", "migrated_legacy"})
TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION = "spritelab_training_launch_context_v2"
TRAINING_LAUNCH_RECEIPT_TTL_SECONDS = 300
_MAX_CAMPAIGN_MAPPING_BYTES = 4 * 1024 * 1024
_MAX_BOUND_INPUT_BYTES = 64 * 1024 * 1024
_MAX_BOUND_OUTPUT_CONTROL_FILE_BYTES = 64 * 1024 * 1024
_MAX_BOUND_OUTPUT_CONTROL_TOTAL_BYTES = 128 * 1024 * 1024
_BOUND_OUTPUT_CONTROL_FILENAMES = frozenset(
    {
        EVENT_FILENAME,
        EVENT_HISTORY_ORIGIN_FILENAME,
        EVENT_HISTORY_TRANSACTION_FILENAME,
        LEGACY_EVENT_FILENAME,
        LEGACY_MIGRATION_FILENAME,
        "run_completion_marker.json",
        "run_identity.json",
        "state.json",
    }
)
_VALIDATOR_ISSUER_KEY = secrets.token_bytes(32)
_TRAINING_PROCESS_BOUNDARY_ENV = "SPRITELAB_VALIDATED_TRAINING_BOUNDARY"
_TRAINING_PROCESS_BOUNDARY_SCHEMA = "spritelab_validated_training_process_boundary_v1"
_TRAINING_CODE_BUNDLE_ENV = "SPRITELAB_VALIDATED_TRAINING_CODE_BUNDLE"
_TRAINING_CODE_BUNDLE_SCHEMA = "spritelab_retained_training_code_bundle_v1"
_TRAINING_CODE_MANIFEST_NAME = "spritelab-code-manifest.json"
_MAX_TRAINING_CODE_FILE_BYTES = 16 * 1024 * 1024
_MAX_TRAINING_CODE_BUNDLE_BYTES = 128 * 1024 * 1024
_DANGEROUS_TRAINING_ENV_NAMES = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GCONV_PATH",
        "PATH",
        "PATHEXT",
        "SHELLOPTS",
        "VIRTUAL_ENV",
    }
)
_DANGEROUS_TRAINING_ENV_PREFIXES = ("DYLD_", "LD_", "PYTHON")
_CAMPAIGN_RUN_CONTRACT_SCHEMA = "spritelab_generator_campaign_run_contract_v1"
_CAMPAIGN_RUN_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_identity",
        "run_id",
        "run_identity",
        "seed",
        "output_root",
        "resolved_config",
        "resolved_config_sha256",
        "execution_contract_sha256",
        "expected_checkpoint_steps",
        "expected_evaluation_steps",
        "max_optimizer_steps",
        "schedule_name",
        "evaluation_ema_policy",
        "training_code_identity_sha256",
    }
)
_PROCESS_BOUNDARY_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_identity",
        "run_identity",
        "run_id",
        "project_root",
        "logical_output_root",
        "root_token",
        "root_identity",
        "root_entries_sha256",
        "config_path",
        "config_token",
        "config_identity",
        "config_content_sha256",
        "resolved_config_sha256",
        "checkpoint_path",
        "checkpoint_token",
        "checkpoint_identity",
        "checkpoint_content_sha256",
        "campaign_run_contract",
        "campaign_run_contract_sha256",
        "input_files",
    }
)
_PROCESS_INPUT_FILE_FIELDS = frozenset(
    {"role", "logical_path", "dataset_relative_path", "token", "identity", "content_sha256"}
)
_RETAINED_CHILD_RESOURCES: list[int] = []

_ISOLATED_TRAINING_BOOTSTRAP = r"""
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import runpy
import sys
import zipfile

ENV = "SPRITELAB_VALIDATED_TRAINING_CODE_BUNDLE"
SCHEMA = "spritelab_retained_training_code_bundle_v1"
MANIFEST = "spritelab-code-manifest.json"
MAX_BYTES = 134217728

def strict_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate retained-code mapping key")
        value[key] = item
    return value

raw = os.environ.pop(ENV, None)
if raw is None:
    raise RuntimeError("retained training code capability is missing")
payload = json.loads(raw, object_pairs_hook=strict_pairs)
required = {"schema_version", "token", "size_bytes", "bundle_sha256", "code_identity_sha256", "dependency_paths"}
if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != SCHEMA:
    raise RuntimeError("retained training code capability is malformed")
kind = payload.get("token", {}).get("kind") if isinstance(payload.get("token"), dict) else None
token = payload.get("token", {}).get("value") if isinstance(payload.get("token"), dict) else None
expected_kind = "windows_handle" if os.name == "nt" else "posix_fd"
if kind != expected_kind or type(token) is not int or token < 0:
    raise RuntimeError("retained training code token is malformed")
size = payload.get("size_bytes")
if type(size) is not int or size < 1 or size > MAX_BYTES:
    raise RuntimeError("retained training code size is invalid")
if os.name == "nt":
    import msvcrt
    os.set_handle_inheritable(token, False)
    descriptor = msvcrt.open_osfhandle(token, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
else:
    descriptor = token
    os.set_inheritable(descriptor, False)
chunks = []
remaining = size
os.lseek(descriptor, 0, os.SEEK_SET)
while remaining:
    chunk = os.read(descriptor, min(1048576, remaining))
    if not chunk:
        raise RuntimeError("retained training code bundle ended early")
    chunks.append(chunk)
    remaining -= len(chunk)
if os.read(descriptor, 1):
    raise RuntimeError("retained training code bundle exceeds its bound size")
os.close(descriptor)
bundle = b"".join(chunks)
if hashlib.sha256(bundle).hexdigest() != payload.get("bundle_sha256"):
    raise RuntimeError("retained training code bundle hash changed")
with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
    names = archive.namelist()
    if len(names) != len(set(names)) or MANIFEST not in names:
        raise RuntimeError("retained training code archive inventory is ambiguous")
    manifest = json.loads(archive.read(MANIFEST), object_pairs_hook=strict_pairs)
    manifest_fields = {"schema_version", "code_identity_sha256", "files"}
    if not isinstance(manifest, dict) or set(manifest) != manifest_fields:
        raise RuntimeError("retained training code manifest is malformed")
    if manifest.get("schema_version") != SCHEMA:
        raise RuntimeError("retained training code manifest schema is unsupported")
    if manifest.get("code_identity_sha256") != payload.get("code_identity_sha256"):
        raise RuntimeError("retained training code identity changed")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("retained training code manifest has no sources")
    expected_names = {MANIFEST}
    sources = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError("retained training source record is malformed")
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not path.startswith("src/spritelab/") or not path.endswith(".py"):
            raise RuntimeError("retained training source path is unsafe")
        if path in expected_names or "\\" in path or "/../" in ("/" + path + "/"):
            raise RuntimeError("retained training source path is ambiguous")
        expected_names.add(path)
        source = archive.read(path)
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError("retained training source hash changed")
        relative = path[len("src/"):-3]
        is_package = relative.endswith("/__init__")
        module = relative[:-len("/__init__")].replace("/", ".") if is_package else relative.replace("/", ".")
        if module in sources:
            raise RuntimeError("retained training module inventory is ambiguous")
        sources[module] = (source, path, is_package)
    if set(names) != expected_names or "spritelab" not in sources or "spritelab.__main__" not in sources:
        raise RuntimeError("retained training code archive inventory changed")

class MemorySourceLoader(importlib.abc.Loader):
    def __init__(self, fullname):
        self.fullname = fullname
    def create_module(self, spec):
        return None
    def get_filename(self, fullname):
        return sources[fullname][1]
    def get_source(self, fullname):
        return sources[fullname][0].decode("utf-8")
    def get_code(self, fullname):
        source, origin, _is_package = sources[fullname]
        return compile(source, origin, "exec", dont_inherit=True)
    def is_package(self, fullname):
        return bool(sources[fullname][2])
    def exec_module(self, module):
        exec(self.get_code(self.fullname), module.__dict__)

class MemorySourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in sources:
            return None
        loader = MemorySourceLoader(fullname)
        return importlib.util.spec_from_loader(fullname, loader, is_package=loader.is_package(fullname))

for key in tuple(os.environ):
    upper = key.upper()
    if upper in {"BASH_ENV", "CDPATH", "ENV", "GCONV_PATH", "PATH", "PATHEXT", "SHELLOPTS", "VIRTUAL_ENV"} or upper.startswith(("DYLD_", "LD_", "PYTHON")):
        os.environ.pop(key, None)
if sys.argv[1:3] != ["-m", "spritelab"]:
    raise RuntimeError("retained training bootstrap received an unsupported command")
sys.meta_path.insert(0, MemorySourceFinder())
dependency_paths = payload.get("dependency_paths")
if not isinstance(dependency_paths, list) or any(not isinstance(path, str) or not os.path.isabs(path) for path in dependency_paths):
    raise RuntimeError("retained training dependency path inventory is malformed")
sys.path.extend(path for path in dependency_paths if path not in sys.path)
sys.argv = ["spritelab", *sys.argv[3:]]
runpy.run_module("spritelab", run_name="__main__", alter_sys=True)
""".strip()


class _DuplicateMappingKeyError(ValueError):
    """An untrusted campaign mapping contains an ambiguous key."""


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate keys at every nesting depth."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            keys: set[Any] = set()
            for key_node, _value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    if key in keys:
                        raise _DuplicateMappingKeyError
                    keys.add(key)
                except TypeError:
                    # The safe base constructor reports unsupported mapping keys.
                    pass
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class TrainingLaunchReceipt:
    schema_version: str
    receipt_id: str
    campaign_identity_sha256: str
    campaign_manifest_sha256: str
    campaign_validation_report_sha256: str
    training_code_identity_sha256: str
    resolved_configuration_sha256: str
    dataset_identity: str
    view_identity: str
    split_identity: str
    architecture_identity: str
    optimizer_identity: str
    schedule_identity: str
    loss_identity: str
    maximum_optimizer_steps: int
    run_identity: str
    cell_identity: str
    seed: int
    output_root_identity: str
    compute_backend_id: str
    launch_authorization_evidence_sha256: str
    execution_spec_sha256: str
    argv_sha256: str
    resume_validation_sha256: str
    event_migration_state: str
    event_migration_identity_sha256: str
    event_history_origin: str
    event_migration_required: bool
    event_migration_record_sha256: str | None
    event_canonical_prefix_sha256: str | None
    event_canonical_identity_sha256: str | None
    source_checkpoint_identity: str | None
    unsafe_resume: bool
    launch_authorized: bool
    execute_confirmed: bool
    created_at_utc: str
    expires_at_utc: str
    validator_proof_sha256: str
    receipt_sha256: str

    def body(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("receipt_sha256", None)
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingLaunchContext:
    """Authoritative inputs retained so an adapter can revalidate a receipt."""

    schema_version: str
    campaign_config_path: Path
    campaign_profile: str
    project_root: Path
    run_id: str
    resume: bool
    launch_authorization_evidence_sha256: str
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ValidatedTrainingLaunch:
    receipt: TrainingLaunchReceipt
    validator_context: TrainingLaunchContext
    campaign: Mapping[str, Any]
    run: Mapping[str, Any]
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    output_root: Path


@dataclass(frozen=True)
class _RetainedRegularFile:
    logical_path: Path
    parent: AnchoredDirectory
    name: str
    descriptor: int
    identity: OwnedFileIdentity
    content_sha256: str


@dataclass(frozen=True)
class _RetainedRunInputs:
    training_manifest: _RetainedRegularFile
    vocabulary: _RetainedRegularFile | None
    dataset_files: Mapping[str, _RetainedRegularFile]


@dataclass(frozen=True)
class TrainingProcessBoundary:
    """Exact inherited process inputs consumed before training initialization."""

    config: Mapping[str, Any]
    project_root: Path
    logical_output_root: Path
    output_root: Path
    resume_path: Path | None
    resume_descriptor: int | None
    resume_sha256: str | None
    campaign_run_contract: Mapping[str, Any]
    training_manifest_bytes: bytes
    vocabulary_bytes: bytes | None
    dataset_descriptors: Mapping[str, int]
    dataset_content_sha256: Mapping[str, str]


def _identity_payload(identity: OwnedFileIdentity) -> dict[str, int]:
    return {
        "device": int(identity.device),
        "inode": int(identity.inode),
        "file_type": int(identity.file_type),
    }


def _identity_from_payload(value: Any, *, label: str) -> OwnedFileIdentity:
    if not isinstance(value, Mapping) or set(value) != {"device", "inode", "file_type"}:
        raise CampaignValidationError(f"training process boundary {label} identity is malformed")
    fields = [value.get(field) for field in ("device", "inode", "file_type")]
    if any(type(field) is not int or field < 0 for field in fields):
        raise CampaignValidationError(f"training process boundary {label} identity is malformed")
    return OwnedFileIdentity(int(fields[0]), int(fields[1]), int(fields[2]))


def _descriptor_sha256(descriptor: int) -> str:
    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.lseek(descriptor, offset, os.SEEK_SET)
    return digest.hexdigest()


def _descriptor_bytes(descriptor: int, *, max_bytes: int, label: str) -> bytes:
    metadata = os.fstat(descriptor)
    if metadata.st_size > max_bytes:
        raise CampaignValidationError(f"training process boundary {label} exceeds the byte limit")
    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    chunks: list[bytes] = []
    remaining = int(metadata.st_size) + 1
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.lseek(descriptor, offset, os.SEEK_SET)
    payload = b"".join(chunks)
    if len(payload) != metadata.st_size:
        raise CampaignValidationError(f"training process boundary {label} changed while being read")
    return payload


def _metadata_is_reparse(metadata: os.stat_result) -> bool:
    return bool(int(getattr(metadata, "st_file_attributes", 0) or 0) & 0x400)


def _require_retained_regular_file(
    metadata: os.stat_result,
    *,
    identity: OwnedFileIdentity,
    boundary_device: int,
    label: str,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _metadata_is_reparse(metadata)
        or metadata.st_nlink != 1
        or metadata.st_dev != boundary_device
        or not identity.matches(metadata)
    ):
        raise CampaignValidationError(f"training process boundary {label} is not one exact single-link file")


def _secure_directory_chain(
    stack: ExitStack,
    target: Path,
    project_root: Path,
    *,
    create: bool,
) -> AnchoredDirectory:
    confined = require_confined_path(target, project_root, allow_root=True)
    anchor = stack.enter_context(AnchoredDirectory(project_root, project_root))
    for part in confined.relative_to(project_root).parts:
        if create:
            anchor.mkdir(part, exist_ok=True)
        anchor = stack.enter_context(anchor.open_directory_immovable(part))
    return anchor


def _retain_regular_file(
    stack: ExitStack,
    path: Path,
    project_root: Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> _RetainedRegularFile:
    confined = require_confined_path(path, project_root)
    parent = _secure_directory_chain(stack, confined.parent, project_root, create=False)
    before = parent.lstat(confined.name)
    identity = OwnedFileIdentity.from_stat(before)
    _require_retained_regular_file(
        before,
        identity=identity,
        boundary_device=int(parent.directory_metadata().st_dev),
        label=label,
    )
    descriptor = parent.open_file_immovable(
        confined.name,
        os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
    )
    stack.callback(os.close, descriptor)
    held = os.fstat(descriptor)
    _require_retained_regular_file(
        held,
        identity=identity,
        boundary_device=int(parent.directory_metadata().st_dev),
        label=label,
    )
    if max_bytes is not None and held.st_size > max_bytes:
        raise CampaignValidationError(f"training process boundary {label} exceeds the byte limit")
    content_sha256 = _descriptor_sha256(descriptor)
    after = parent.lstat(confined.name)
    _require_retained_regular_file(
        after,
        identity=identity,
        boundary_device=int(parent.directory_metadata().st_dev),
        label=label,
    )
    return _RetainedRegularFile(confined, parent, confined.name, descriptor, identity, content_sha256)


def _zip_entry(path: str, content: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info, content


def _retained_training_code_bundle(
    stack: ExitStack,
    campaign: Mapping[str, Any],
    project_root: Path,
) -> _RetainedRegularFile:
    """Build and retain one deterministic archive from exact held production sources."""

    code_identity = campaign.get("code_identity")
    if not isinstance(code_identity, Mapping) or not is_concrete_hash(code_identity.get("sha256")):
        raise CampaignValidationError("training code identity is malformed at the process boundary")
    records = code_identity.get("files")
    if not isinstance(records, list) or not records:
        raise CampaignValidationError("training code identity has no source inventory")
    code_root = Path(__file__).resolve().parents[3]
    sources: list[tuple[str, str, bytes]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise CampaignValidationError("training code identity source record is malformed")
        relative_text = record.get("path")
        declared_sha256 = record.get("sha256")
        if not isinstance(relative_text, str) or "\\" in relative_text:
            raise CampaignValidationError("training code identity source path is unsafe")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("src", "spritelab")
            or relative.suffix != ".py"
        ):
            continue
        normalized = relative.as_posix()
        if normalized in seen or not is_concrete_hash(declared_sha256):
            raise CampaignValidationError("training production source inventory is ambiguous")
        seen.add(normalized)
        retained = _retain_regular_file(
            stack,
            code_root.joinpath(*relative.parts),
            code_root,
            label="training production source",
            max_bytes=_MAX_TRAINING_CODE_FILE_BYTES,
        )
        content = _descriptor_bytes(
            retained.descriptor,
            max_bytes=_MAX_TRAINING_CODE_FILE_BYTES,
            label="training production source",
        )
        if not hmac.compare_digest(retained.content_sha256, str(declared_sha256)):
            raise CampaignValidationError("training production source differs from campaign code identity")
        sources.append((normalized, str(declared_sha256), content))
    if not sources or "src/spritelab/__init__.py" not in seen or "src/spritelab/__main__.py" not in seen:
        raise CampaignValidationError("training production source bundle is incomplete")
    source_rows = [{"path": path, "sha256": digest} for path, digest, _content in sorted(sources)]
    manifest = {
        "schema_version": _TRAINING_CODE_BUNDLE_SCHEMA,
        "code_identity_sha256": code_identity["sha256"],
        "files": source_rows,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        info, content = _zip_entry(
            _TRAINING_CODE_MANIFEST_NAME,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        archive.writestr(info, content)
        for path, _digest, source in sorted(sources):
            info, content = _zip_entry(path, source)
            archive.writestr(info, content)
    bundle_bytes = buffer.getvalue()
    if not bundle_bytes or len(bundle_bytes) > _MAX_TRAINING_CODE_BUNDLE_BYTES:
        raise CampaignValidationError("retained training code bundle exceeds the byte limit")
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    bundle_directory = _secure_directory_chain(
        stack,
        project_root / ".spritelab" / "training_code_bundles",
        project_root,
        create=True,
    )
    bundle_name = f"{code_identity['sha256']}.zip"
    if not bundle_directory.lexists(bundle_name):
        try:
            descriptor = bundle_directory.open_file(
                bundle_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                0o600,
            )
        except FileExistsError:
            pass
        else:
            try:
                view = memoryview(bundle_bytes)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("retained training code bundle write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    retained_bundle = _retain_regular_file(
        stack,
        bundle_directory.directory / bundle_name,
        project_root,
        label="retained training code bundle",
        max_bytes=_MAX_TRAINING_CODE_BUNDLE_BYTES,
    )
    if not hmac.compare_digest(retained_bundle.content_sha256, bundle_sha256):
        raise CampaignValidationError("cached retained training code bundle conflicts with exact source bytes")
    return retained_bundle


def _resolved_input_path(value: Any, project_root: Path, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        raise CampaignValidationError(f"training process boundary {label} path is malformed")
    candidate = Path(os.fspath(value))
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return require_confined_path(_lexical_absolute(candidate, label=label), project_root)


def _strict_manifest_records(content: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CampaignValidationError(f"training process boundary {label} is not UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_strict_json_mapping)
        except (_DuplicateMappingKeyError, json.JSONDecodeError) as exc:
            raise CampaignValidationError(f"training process boundary {label} line {line_number} is malformed") from exc
        if not isinstance(value, dict):
            raise CampaignValidationError(f"training process boundary {label} contains a non-object row")
        records.append(value)
    if not records:
        raise CampaignValidationError(f"training process boundary {label} has no records")
    return records


def _retain_run_inputs(
    stack: ExitStack,
    run: Mapping[str, Any],
    project_root: Path,
) -> _RetainedRunInputs:
    resolved = run.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise CampaignValidationError("training process boundary resolved input config is malformed")
    dataset = resolved.get("dataset")
    conditioning = resolved.get("conditioning")
    bindings = resolved.get("campaign_bindings")
    if not isinstance(dataset, Mapping) or not isinstance(conditioning, Mapping):
        raise CampaignValidationError("training process boundary input mappings are malformed")
    dataset_root = _resolved_input_path(dataset.get("directory"), project_root, label="dataset root")
    _secure_directory_chain(stack, dataset_root, project_root, create=False)
    manifest_path = _resolved_input_path(
        dataset.get("training_manifest"),
        project_root,
        label="training manifest",
    )
    training_manifest = _retain_regular_file(
        stack,
        manifest_path,
        project_root,
        label="training manifest",
        max_bytes=_MAX_BOUND_INPUT_BYTES,
    )
    manifest_content = _descriptor_bytes(
        training_manifest.descriptor,
        max_bytes=_MAX_BOUND_INPUT_BYTES,
        label="training manifest",
    )
    records = _strict_manifest_records(manifest_content, label="training manifest")
    if isinstance(bindings, Mapping):
        expected_manifest = bindings.get("split_manifest_hash")
        if is_concrete_hash(expected_manifest) and not hmac.compare_digest(
            training_manifest.content_sha256,
            str(expected_manifest),
        ):
            raise CampaignValidationError("retained training manifest differs from campaign split identity")
    vocabulary: _RetainedRegularFile | None = None
    vocabulary_value = conditioning.get("vocabulary_path")
    if vocabulary_value:
        vocabulary = _retain_regular_file(
            stack,
            _resolved_input_path(vocabulary_value, project_root, label="conditioning vocabulary"),
            project_root,
            label="conditioning vocabulary",
            max_bytes=_MAX_BOUND_INPUT_BYTES,
        )
        if isinstance(bindings, Mapping):
            expected_vocabulary = bindings.get("conditioning_vocabulary_hash")
            if is_concrete_hash(expected_vocabulary) and not hmac.compare_digest(
                vocabulary.content_sha256,
                str(expected_vocabulary),
            ):
                raise CampaignValidationError("retained conditioning vocabulary differs from campaign identity")
    dataset_files: dict[str, _RetainedRegularFile] = {}
    for record in records:
        relative_text = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
        if "\\" in relative_text:
            raise CampaignValidationError("training manifest dataset path is unsafe")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or any(part in {"", "."} for part in relative.parts):
            raise CampaignValidationError("training manifest dataset path is unsafe")
        normalized = relative.as_posix()
        if normalized in dataset_files:
            continue
        dataset_path = require_confined_path(
            _lexical_absolute(dataset_root.joinpath(*relative.parts), label="dataset artifact"),
            dataset_root,
        )
        dataset_files[normalized] = _retain_regular_file(
            stack,
            dataset_path,
            project_root,
            label="dataset artifact",
        )
    if not dataset_files:
        raise CampaignValidationError("training manifest retained no dataset artifacts")
    return _RetainedRunInputs(training_manifest, vocabulary, dataset_files)


def _is_bound_output_control_file(name: str) -> bool:
    if name in _BOUND_OUTPUT_CONTROL_FILENAMES:
        return True
    prefix = "checkpoint_step_"
    suffix = ".json"
    step = name[len(prefix) : -len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else ""
    return bool(step) and step.isascii() and step.isdigit()


def _snapshot_regular_content(
    descriptor: int,
    *,
    initial: os.stat_result,
    identity: OwnedFileIdentity,
    boundary_device: int,
    label: str,
) -> tuple[str, int]:
    """Hash one stable control file without materializing it in memory."""

    size = int(initial.st_size)
    if size < 0 or size > _MAX_BOUND_OUTPUT_CONTROL_FILE_BYTES:
        raise CampaignValidationError(f"{label} exceeds the retained output-control byte limit")
    opened = os.fstat(descriptor)
    if not _stable_regular_file(
        opened,
        identity=identity,
        initial=initial,
        boundary_device=boundary_device,
    ):
        raise CampaignValidationError(f"{label} changed while its content identity was opened")
    content_sha256 = _descriptor_sha256(descriptor)
    after = os.fstat(descriptor)
    if not _stable_regular_file(
        after,
        identity=identity,
        initial=initial,
        boundary_device=boundary_device,
    ):
        raise CampaignValidationError(f"{label} changed while its content identity was captured")
    return content_sha256, size


def _anchored_entry_snapshot(anchor: AnchoredDirectory) -> tuple[tuple[Any, ...], ...]:
    """Bind direct children and exact audit-control bytes without following links."""

    boundary_device = int(anchor.directory_metadata().st_dev)
    rows: list[tuple[Any, ...]] = []
    bound_control_bytes = 0
    for name in anchor.names():
        metadata = anchor.lstat(name)
        if stat.S_ISLNK(metadata.st_mode) or _metadata_is_reparse(metadata):
            raise CampaignValidationError("training output contains a linked or reparse child")
        file_type = stat.S_IFMT(metadata.st_mode)
        content_sha256: str | None = None
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1 or metadata.st_dev != boundary_device:
                raise CampaignValidationError("training output contains an aliased or cross-device file")
            if _is_bound_output_control_file(name):
                projected_total = bound_control_bytes + int(metadata.st_size)
                if projected_total > _MAX_BOUND_OUTPUT_CONTROL_TOTAL_BYTES:
                    raise CampaignValidationError("training output control files exceed the aggregate byte limit")
                identity = OwnedFileIdentity.from_stat(metadata)
                descriptor = anchor.open_file_immovable(
                    name,
                    os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
                )
                try:
                    content_sha256, captured_size = _snapshot_regular_content(
                        descriptor,
                        initial=metadata,
                        identity=identity,
                        boundary_device=boundary_device,
                        label=f"training output control file {name}",
                    )
                finally:
                    os.close(descriptor)
                visible = anchor.lstat(name)
                if not _stable_regular_file(
                    visible,
                    identity=identity,
                    initial=metadata,
                    boundary_device=boundary_device,
                ):
                    raise CampaignValidationError(
                        f"training output control file {name} changed while its content identity was captured"
                    )
                bound_control_bytes += captured_size
        elif stat.S_ISDIR(metadata.st_mode):
            if metadata.st_dev != boundary_device:
                raise CampaignValidationError("training output contains a cross-device directory")
        else:
            raise CampaignValidationError("training output contains an unsupported filesystem object")
        rows.append(
            (
                name,
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(file_type),
                int(metadata.st_nlink),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                content_sha256,
            )
        )
    return tuple(rows)


def _validate_campaign_directory_shape(
    stack: ExitStack,
    campaign: Mapping[str, Any],
    project_root: Path,
    logical_roots: Mapping[str, Path],
) -> None:
    campaign_roots = {path.parents[1] for path in logical_roots.values() if len(path.parents) >= 2}
    if len(campaign_roots) != 1:
        raise CampaignValidationError("training output roots do not share one exact campaign directory")
    campaign_root = next(iter(campaign_roots))
    expected_by_cell: dict[str, set[str]] = {}
    for run in campaign.get("expected_runs") or ():
        run_id = str(run.get("run_id"))
        output_root = logical_roots[run_id]
        relative = output_root.relative_to(campaign_root)
        if len(relative.parts) != 2:
            raise CampaignValidationError("training output root is not one exact cell/seed directory")
        expected_by_cell.setdefault(relative.parts[0], set()).add(relative.parts[1])
    campaign_anchor = _secure_directory_chain(stack, campaign_root, project_root, create=True)
    for name in campaign_anchor.names():
        metadata = campaign_anchor.lstat(name)
        if stat.S_ISLNK(metadata.st_mode) or _metadata_is_reparse(metadata):
            raise CampaignValidationError("training campaign directory contains a linked child")
        if stat.S_ISDIR(metadata.st_mode):
            if name not in expected_by_cell and name != "resolved_configs":
                raise CampaignValidationError("training campaign directory contains a foreign run directory")
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CampaignValidationError("training campaign directory contains an unsupported child")
    for cell_name, seed_names in expected_by_cell.items():
        cell_anchor = _secure_directory_chain(stack, campaign_root / cell_name, project_root, create=True)
        for name in cell_anchor.names():
            metadata = cell_anchor.lstat(name)
            if stat.S_ISLNK(metadata.st_mode) or _metadata_is_reparse(metadata):
                raise CampaignValidationError("training cell directory contains a linked child")
            if stat.S_ISDIR(metadata.st_mode):
                if name not in seed_names:
                    raise CampaignValidationError("training cell directory contains a foreign seed directory")
            elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise CampaignValidationError("training cell directory contains an unsupported child")


def _campaign_run_contract(campaign: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    training = campaign.get("training")
    schedule = campaign.get("schedule")
    evaluation = campaign.get("evaluation")
    code_identity = campaign.get("code_identity")
    if not all(isinstance(value, Mapping) for value in (training, schedule, evaluation, code_identity)):
        raise CampaignValidationError("campaign run contract source mappings are malformed")
    contract = {
        "schema_version": _CAMPAIGN_RUN_CONTRACT_SCHEMA,
        "campaign_id": campaign.get("campaign_id"),
        "campaign_identity": campaign.get("campaign_identity"),
        "run_id": run.get("run_id"),
        "run_identity": run.get("run_identity"),
        "seed": run.get("seed"),
        "output_root": run.get("output_root"),
        "resolved_config": deepcopy(run.get("resolved_config")),
        "resolved_config_sha256": run.get("resolved_config_sha256"),
        "execution_contract_sha256": run.get("execution_contract_sha256"),
        "expected_checkpoint_steps": list(run.get("expected_checkpoint_steps") or ()),
        "expected_evaluation_steps": list(run.get("expected_evaluation_steps") or ()),
        "max_optimizer_steps": training.get("max_optimizer_steps"),
        "schedule_name": schedule.get("name"),
        "evaluation_ema_policy": evaluation.get("ema_policy"),
        "training_code_identity_sha256": code_identity.get("sha256"),
    }
    if set(contract) != _CAMPAIGN_RUN_CONTRACT_FIELDS:
        raise CampaignValidationError("campaign run contract has an unsupported shape")
    protected = (
        "campaign_identity",
        "run_identity",
        "resolved_config_sha256",
        "execution_contract_sha256",
        "training_code_identity_sha256",
    )
    if any(not is_concrete_hash(contract[field]) for field in protected):
        raise CampaignValidationError("campaign run contract contains a non-concrete identity")
    if stable_hash(contract["resolved_config"]) != contract["resolved_config_sha256"]:
        raise CampaignValidationError("campaign run contract resolved config identity changed")
    if type(contract["seed"]) is not int or type(contract["max_optimizer_steps"]) is not int:
        raise CampaignValidationError("campaign run contract numeric fields are malformed")
    for field in ("expected_checkpoint_steps", "expected_evaluation_steps"):
        if any(type(step) is not int or step < 0 for step in contract[field]):
            raise CampaignValidationError(f"campaign run contract {field} is malformed")
    return contract


def _retained_dependency_paths(stack: ExitStack, code_root: Path) -> tuple[str, ...]:
    """Retain only interpreter installation package roots, never ambient PYTHONPATH entries."""

    selected: dict[str, Path] = {}
    for raw in sys.path:
        if not isinstance(raw, str) or not raw.strip():
            continue
        candidate = Path(raw)
        if not candidate.is_absolute() or not any(
            part.casefold() in {"site-packages", "dist-packages"} for part in candidate.parts
        ):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            metadata = candidate.lstat()
        except OSError as exc:
            raise CampaignValidationError("training dependency root cannot be retained") from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or _metadata_is_reparse(metadata):
            raise CampaignValidationError("training dependency root crosses a linked filesystem seam")
        try:
            resolved.relative_to(code_root)
        except ValueError:
            pass
        else:
            continue
        key = os.path.normcase(str(resolved))
        selected[key] = resolved
    if not selected:
        raise CampaignValidationError("training interpreter has no retained dependency roots")
    paths: list[str] = []
    for key in sorted(selected):
        path = selected[key]
        stack.enter_context(AnchoredDirectory(path, path))
        paths.append(str(path))
    return tuple(paths)


@contextmanager
def _inheritable_file_token(descriptor: int) -> Iterator[tuple[str, int]]:
    if os.name == "nt":
        import msvcrt

        token = int(msvcrt.get_osfhandle(descriptor))
        before = os.get_handle_inheritable(token)
        os.set_handle_inheritable(token, True)
        try:
            yield "windows_handle", token
        finally:
            os.set_handle_inheritable(token, before)
        return
    before = os.get_inheritable(descriptor)
    os.set_inheritable(descriptor, True)
    try:
        yield "posix_fd", descriptor
    finally:
        os.set_inheritable(descriptor, before)


class TrainingFilesystemCapability(AbstractContextManager["TrainingFilesystemCapability"]):
    """Retain exact campaign roots and launch inputs through one process spawn."""

    def __init__(self, campaign: Mapping[str, Any], project_root: str | Path) -> None:
        self.campaign = deepcopy(dict(campaign))
        self.project_root = _lexical_absolute(project_root, label="training project root")
        self._stack: ExitStack | None = None
        self._roots: dict[str, AnchoredDirectory] = {}
        self._logical_roots: dict[str, Path] = {}
        self._configs: dict[str, _RetainedRegularFile] = {}
        self._inputs: dict[str, _RetainedRunInputs] = {}
        self._checkpoints: dict[str, _RetainedRegularFile] = {}
        self._code_bundle: _RetainedRegularFile | None = None
        self._dependency_paths: tuple[str, ...] = ()
        self._entry_snapshots: dict[str, tuple[tuple[Any, ...], ...]] = {}
        self.filesystem_snapshot: _CampaignFilesystemSnapshot | None = None
        self.resume_report: Mapping[str, Any] | None = None

    def __enter__(self) -> TrainingFilesystemCapability:
        if self._stack is not None:
            raise CampaignValidationError("training filesystem capability cannot be entered twice")
        stack = ExitStack()
        stack.__enter__()
        self._stack = stack
        try:
            runs = list(self.campaign.get("expected_runs") or ())
            if not runs:
                raise CampaignValidationError("training filesystem capability requires campaign runs")
            for run in runs:
                run_id = str(run.get("run_id"))
                if not run_id or run_id in self._roots:
                    raise CampaignValidationError("training filesystem capability run set is ambiguous")
                logical_root = require_confined_path(Path(str(run.get("output_root"))), self.project_root)
                self._logical_roots[run_id] = logical_root
            _validate_campaign_directory_shape(stack, self.campaign, self.project_root, self._logical_roots)
            for run in runs:
                run_id = str(run["run_id"])
                root = _secure_directory_chain(stack, self._logical_roots[run_id], self.project_root, create=True)
                self._roots[run_id] = root
                self._entry_snapshots[run_id] = _anchored_entry_snapshot(root)
                config = _retain_regular_file(
                    stack,
                    Path(str(run.get("resolved_config_path"))),
                    self.project_root,
                    label="resolved run config",
                    max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
                )
                try:
                    parsed = json.loads(
                        _descriptor_bytes(
                            config.descriptor,
                            max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
                            label="resolved run config",
                        ).decode("utf-8"),
                        object_pairs_hook=_strict_json_mapping,
                    )
                except (_DuplicateMappingKeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CampaignValidationError("resolved run config cannot be retained exactly") from exc
                if parsed != run.get("resolved_config") or stable_hash(parsed) != run.get("resolved_config_sha256"):
                    raise CampaignValidationError("resolved run config identity changed at the process boundary")
                self._configs[run_id] = config
                self._inputs[run_id] = _retain_run_inputs(stack, run, self.project_root)
            self._code_bundle = _retained_training_code_bundle(stack, self.campaign, self.project_root)
            self._dependency_paths = _retained_dependency_paths(stack, Path(__file__).resolve().parents[3])
            physical_roots = {run_id: root.fixed_directory_path() for run_id, root in self._roots.items()}
            snapshot = _capture_anchored_campaign_filesystem_snapshot(self.campaign, physical_roots)
            for run_id, root in self._roots.items():
                if _anchored_entry_snapshot(root) != self._entry_snapshots[run_id]:
                    raise CampaignValidationError("training output changed during the retained resume audit")
            report = audit_resume(self.campaign, unsafe_resume=False, filesystem_snapshot=snapshot)
            if not report["safe"]:
                raise CampaignResumeError("unsafe campaign state: " + "; ".join(report["errors"]))
            states = {str(state["run_id"]): state for state in report["runs"]}
            for run in runs:
                run_id = str(run["run_id"])
                state = states[run_id]
                if state["status"] != "valid_resumable":
                    continue
                checkpoint_path = Path(str(state["checkpoint"]))
                if checkpoint_path.parent != self._logical_roots[run_id]:
                    raise CampaignResumeError("resume checkpoint is not a direct child of its retained output root")
                checkpoint = _retain_regular_file(
                    stack,
                    checkpoint_path,
                    self.project_root,
                    label="resume checkpoint",
                )
                if not hmac.compare_digest(
                    checkpoint.content_sha256,
                    str(state.get("checkpoint_content_sha256") or ""),
                ):
                    raise CampaignResumeError("resume checkpoint changed after the retained resume audit")
                self._checkpoints[run_id] = checkpoint
            self.filesystem_snapshot = snapshot
            self.resume_report = report
            return self
        except BaseException:
            stack.__exit__(*sys.exc_info())
            self._stack = None
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.__exit__(exc_type, exc_value, traceback)

    def read_run_regular_bytes(self, run_id: str, name: str, max_bytes: int) -> bytes:
        """Read one exact live run artifact relative to its retained output root."""

        if self._stack is None:
            raise CampaignValidationError("training filesystem capability is not active")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise CampaignValidationError("run artifact byte limit must be a positive integer")
        root = self._roots.get(str(run_id))
        if root is None:
            raise CampaignValidationError("run artifact is outside the retained campaign capability")
        root.verify()
        before = root.lstat(name)
        identity = OwnedFileIdentity.from_stat(before)
        boundary_device = int(root.directory_metadata().st_dev)
        _require_retained_regular_file(
            before,
            identity=identity,
            boundary_device=boundary_device,
            label="run event stream",
        )
        descriptor = root.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        try:
            opened = os.fstat(descriptor)
            _require_retained_regular_file(
                opened,
                identity=identity,
                boundary_device=boundary_device,
                label="run event stream",
            )
            payload = _descriptor_bytes(descriptor, max_bytes=max_bytes, label="run event stream")
            after = os.fstat(descriptor)
            visible = root.lstat(name)
            for metadata in (after, visible):
                _require_retained_regular_file(
                    metadata,
                    identity=identity,
                    boundary_device=boundary_device,
                    label="run event stream",
                )
            if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns) or (
                visible.st_size,
                visible.st_mtime_ns,
            ) != (after.st_size, after.st_mtime_ns):
                raise CampaignValidationError("run event stream changed while being read")
            root.verify()
            return payload
        finally:
            os.close(descriptor)

    def _revalidate_file(self, retained: _RetainedRegularFile, *, label: str) -> None:
        metadata = os.fstat(retained.descriptor)
        _require_retained_regular_file(
            metadata,
            identity=retained.identity,
            boundary_device=int(retained.parent.directory_metadata().st_dev),
            label=label,
        )
        visible = retained.parent.lstat(retained.name)
        _require_retained_regular_file(
            visible,
            identity=retained.identity,
            boundary_device=int(retained.parent.directory_metadata().st_dev),
            label=label,
        )
        if not hmac.compare_digest(_descriptor_sha256(retained.descriptor), retained.content_sha256):
            raise CampaignValidationError(f"training process boundary {label} content changed")

    def bootstrap_command(self, validated: ValidatedTrainingLaunch) -> tuple[str, ...]:
        """Wrap the exact campaign argv in an isolated stdlib-only pre-import bootstrap."""

        if self._stack is None or self._code_bundle is None:
            raise CampaignValidationError("training filesystem capability is not active")
        command = tuple(str(item) for item in validated.argv)
        if len(command) < 4 or command[1:3] != ("-m", "spritelab"):
            raise CampaignValidationError("training command cannot be isolated before project import")
        if _lexical_absolute(command[0], label="training interpreter") != _lexical_absolute(
            sys.executable,
            label="active training interpreter",
        ):
            raise CampaignValidationError("training command interpreter differs from the active validator")
        self._revalidate_file(self._code_bundle, label="retained training code bundle")
        return (command[0], "-I", "-c", _ISOLATED_TRAINING_BOOTSTRAP, *command[1:])

    @contextmanager
    def launch_inheritance(
        self,
        validated: ValidatedTrainingLaunch,
    ) -> Iterator[tuple[dict[str, str], dict[str, Any]]]:
        """Yield child-only environment and exact subprocess inheritance options."""

        if self._stack is None or self.filesystem_snapshot is None:
            raise CampaignValidationError("training filesystem capability is not active")
        run_id = str(validated.run.get("run_id"))
        root = self._roots.get(run_id)
        config = self._configs.get(run_id)
        run_inputs = self._inputs.get(run_id)
        if root is None or config is None or run_inputs is None:
            raise CampaignValidationError("validated launch is outside the retained campaign capability")
        retained_run = next(
            (run for run in self.campaign.get("expected_runs") or () if str(run.get("run_id")) == run_id),
            None,
        )
        if (
            not isinstance(retained_run, Mapping)
            or dict(validated.run) != dict(retained_run)
            or validated.campaign.get("campaign_identity") != self.campaign.get("campaign_identity")
            or validated.receipt.campaign_identity_sha256 != self.campaign.get("campaign_identity")
            or validated.validator_context.project_root != self.project_root
        ):
            raise CampaignValidationError("validated launch identity differs from its retained campaign capability")
        if validated.output_root != self._logical_roots[run_id]:
            raise CampaignValidationError("validated launch output root changed before process inheritance")
        root.verify()
        if _anchored_entry_snapshot(root) != self._entry_snapshots[run_id]:
            raise CampaignValidationError("training output changed after the retained resume audit")
        self._revalidate_file(config, label="resolved run config")
        checkpoint = self._checkpoints.get(run_id)
        code_bundle = self._code_bundle
        if code_bundle is None:
            raise CampaignValidationError("retained training code capability is unavailable")
        self._revalidate_file(code_bundle, label="retained training code bundle")
        if validated.validator_context.resume is not (checkpoint is not None):
            raise CampaignResumeError("validated launch resume mode changed before process inheritance")
        if checkpoint is not None:
            self._revalidate_file(checkpoint, label="resume checkpoint")
        self._revalidate_file(run_inputs.training_manifest, label="training manifest")
        if run_inputs.vocabulary is not None:
            self._revalidate_file(run_inputs.vocabulary, label="conditioning vocabulary")
        for retained_input in run_inputs.dataset_files.values():
            self._revalidate_file(retained_input, label="dataset artifact")
        with ExitStack() as inheritance:
            root_kind, root_token = inheritance.enter_context(root.inheritable_token())
            config_kind, config_token = inheritance.enter_context(_inheritable_file_token(config.descriptor))
            code_kind, code_token = inheritance.enter_context(_inheritable_file_token(code_bundle.descriptor))
            checkpoint_kind: str | None = None
            checkpoint_token: int | None = None
            if checkpoint is not None:
                checkpoint_kind, checkpoint_token = inheritance.enter_context(
                    _inheritable_file_token(checkpoint.descriptor)
                )
            input_payload: list[dict[str, Any]] = []
            input_tokens: list[int] = []

            def inherit_input(
                retained: _RetainedRegularFile,
                *,
                role: str,
                dataset_relative_path: str | None,
            ) -> None:
                kind, token = inheritance.enter_context(_inheritable_file_token(retained.descriptor))
                input_tokens.append(token)
                input_payload.append(
                    {
                        "role": role,
                        "logical_path": str(retained.logical_path),
                        "dataset_relative_path": dataset_relative_path,
                        "token": {"kind": kind, "value": token},
                        "identity": _identity_payload(retained.identity),
                        "content_sha256": retained.content_sha256,
                    }
                )

            inherit_input(
                run_inputs.training_manifest,
                role="training_manifest",
                dataset_relative_path=None,
            )
            if run_inputs.vocabulary is not None:
                inherit_input(
                    run_inputs.vocabulary,
                    role="conditioning_vocabulary",
                    dataset_relative_path=None,
                )
            for relative, retained_input in sorted(run_inputs.dataset_files.items()):
                inherit_input(
                    retained_input,
                    role="dataset_artifact",
                    dataset_relative_path=relative,
                )
            root_metadata = root.directory_metadata()
            campaign_run_contract = _campaign_run_contract(self.campaign, retained_run)
            payload = {
                "schema_version": _TRAINING_PROCESS_BOUNDARY_SCHEMA,
                "campaign_identity": validated.receipt.campaign_identity_sha256,
                "run_identity": validated.receipt.run_identity,
                "run_id": run_id,
                "project_root": str(self.project_root),
                "logical_output_root": str(self._logical_roots[run_id]),
                "root_token": {"kind": root_kind, "value": root_token},
                "root_identity": _identity_payload(OwnedFileIdentity.from_stat(root_metadata)),
                "root_entries_sha256": stable_hash([list(row) for row in self._entry_snapshots[run_id]]),
                "config_path": str(config.logical_path),
                "config_token": {"kind": config_kind, "value": config_token},
                "config_identity": _identity_payload(config.identity),
                "config_content_sha256": config.content_sha256,
                "resolved_config_sha256": validated.receipt.resolved_configuration_sha256,
                "checkpoint_path": None if checkpoint is None else str(checkpoint.logical_path),
                "checkpoint_token": (
                    None if checkpoint is None else {"kind": checkpoint_kind, "value": checkpoint_token}
                ),
                "checkpoint_identity": None if checkpoint is None else _identity_payload(checkpoint.identity),
                "checkpoint_content_sha256": (None if checkpoint is None else checkpoint.content_sha256),
                "campaign_run_contract": campaign_run_contract,
                "campaign_run_contract_sha256": stable_hash(campaign_run_contract),
                "input_files": input_payload,
            }
            boundary_environment = {
                _TRAINING_PROCESS_BOUNDARY_ENV: json.dumps(payload, sort_keys=True, separators=(",", ":")),
                _TRAINING_CODE_BUNDLE_ENV: json.dumps(
                    {
                        "schema_version": _TRAINING_CODE_BUNDLE_SCHEMA,
                        "token": {"kind": code_kind, "value": code_token},
                        "size_bytes": int(os.fstat(code_bundle.descriptor).st_size),
                        "bundle_sha256": code_bundle.content_sha256,
                        "code_identity_sha256": self.campaign["code_identity"]["sha256"],
                        "dependency_paths": list(self._dependency_paths),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                handles = [root_token, config_token, code_token, *input_tokens]
                if checkpoint_token is not None:
                    handles.append(checkpoint_token)
                startupinfo.lpAttributeList = {"handle_list": handles}
                spawn_options: dict[str, Any] = {"close_fds": True, "startupinfo": startupinfo}
            else:
                descriptors = [root_token, config_token, code_token, *input_tokens]
                if checkpoint_token is not None:
                    descriptors.append(checkpoint_token)
                spawn_options = {"close_fds": True, "pass_fds": tuple(descriptors)}
            yield boundary_environment, spawn_options


def _strict_process_token(value: Any, *, label: str) -> tuple[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"}:
        raise CampaignValidationError(f"training process boundary {label} token is malformed")
    kind, token = value.get("kind"), value.get("value")
    expected_kind = "windows_handle" if os.name == "nt" else "posix_fd"
    if kind != expected_kind or type(token) is not int or token < 0:
        raise CampaignValidationError(f"training process boundary {label} token is unsupported")
    return str(kind), int(token)


def _validate_visible_boundary_entry(path: Path, identity: OwnedFileIdentity, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CampaignValidationError(f"training process boundary {label} is no longer visible") from exc
    if not identity.matches(metadata):
        raise CampaignValidationError(f"training process boundary {label} path was substituted")
    return metadata


def _child_file_descriptor(
    token_value: Any,
    identity_value: Any,
    logical_path: Path,
    *,
    label: str,
) -> tuple[int, OwnedFileIdentity]:
    kind, token = _strict_process_token(token_value, label=label)
    try:
        if kind == "windows_handle":
            os.set_handle_inheritable(token, False)
        else:
            os.set_inheritable(token, False)
    except OSError as exc:
        raise CampaignValidationError(
            f"training process boundary {label} token inheritability cannot be cleared"
        ) from exc
    identity = _identity_from_payload(identity_value, label=label)
    if os.name == "nt":
        import msvcrt

        try:
            descriptor = msvcrt.open_osfhandle(
                token,
                os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
            )
        except OSError as exc:
            raise CampaignValidationError(f"training process boundary {label} handle is invalid") from exc
    else:
        descriptor = token
    try:
        metadata = os.fstat(descriptor)
        _require_retained_regular_file(
            metadata,
            identity=identity,
            boundary_device=identity.device,
            label=label,
        )
        _validate_visible_boundary_entry(logical_path, identity, label=label)
    except BaseException:
        if os.name == "nt":
            os.close(descriptor)
        raise
    _RETAINED_CHILD_RESOURCES.append(descriptor)
    return descriptor, identity


def _child_output_root(
    token_value: Any,
    identity_value: Any,
    logical_path: Path,
) -> Path:
    kind, token = _strict_process_token(token_value, label="output root")
    try:
        if kind == "windows_handle":
            os.set_handle_inheritable(token, False)
        else:
            os.set_inheritable(token, False)
    except OSError as exc:
        raise CampaignValidationError(
            "training process boundary output-root token inheritability cannot be cleared"
        ) from exc
    identity = _identity_from_payload(identity_value, label="output root")
    visible = _validate_visible_boundary_entry(logical_path, identity, label="output root")
    if not stat.S_ISDIR(visible.st_mode):
        raise CampaignValidationError("training process boundary output root is not a directory")
    if kind == "windows_handle":
        from spritelab.utils.safe_fs import _windows_handle_information

        attributes, file_index = _windows_handle_information(token)
        if not attributes & 0x10 or attributes & 0x400 or int(file_index) != identity.inode:
            raise CampaignValidationError("training process boundary output-root handle identity changed")
        _RETAINED_CHILD_RESOURCES.append(token)
        return logical_path
    try:
        metadata = os.fstat(token)
    except OSError as exc:
        raise CampaignValidationError("training process boundary output-root descriptor is invalid") from exc
    if not stat.S_ISDIR(metadata.st_mode) or not identity.matches(metadata):
        raise CampaignValidationError("training process boundary output-root descriptor identity changed")
    for namespace in (Path("/proc/self/fd"), Path("/dev/fd")):
        fixed = namespace / str(token)
        try:
            fixed_metadata = fixed.stat()
        except OSError:
            continue
        if identity.matches(fixed_metadata):
            _RETAINED_CHILD_RESOURCES.append(token)
            return fixed
    raise CampaignValidationError("training process boundary has no fixed inherited output-root namespace")


def _child_entry_snapshot(token_value: Any) -> tuple[tuple[Any, ...], ...]:
    kind, token = _strict_process_token(token_value, label="output root")
    if kind == "windows_handle":
        from spritelab.utils.safe_fs import _open_windows_child, _windows_list_directory, _windows_relative_stat

        names = tuple(sorted(_windows_list_directory(token), key=lambda item: (item.casefold(), item)))

        def stat_child(name: str) -> os.stat_result:
            return _windows_relative_stat(token, name)

        def open_child(name: str) -> int:
            return _open_windows_child(
                token,
                name,
                os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
                0o600,
                share_delete=False,
            )

    else:
        names = tuple(sorted(os.listdir(token), key=lambda item: (item.casefold(), item)))

        def stat_child(name: str) -> os.stat_result:
            return os.stat(name, dir_fd=token, follow_symlinks=False)

        def open_child(name: str) -> int:
            return os.open(
                name,
                os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0)),
                dir_fd=token,
            )

    rows: list[tuple[Any, ...]] = []
    bound_control_bytes = 0
    for name in names:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise CampaignValidationError("training process boundary output enumeration is unsafe")
        metadata = stat_child(name)
        if stat.S_ISLNK(metadata.st_mode) or _metadata_is_reparse(metadata):
            raise CampaignValidationError("training process boundary output contains a linked child")
        file_type = stat.S_IFMT(metadata.st_mode)
        content_sha256: str | None = None
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise CampaignValidationError("training process boundary output contains an aliased file")
            if _is_bound_output_control_file(name):
                projected_total = bound_control_bytes + int(metadata.st_size)
                if projected_total > _MAX_BOUND_OUTPUT_CONTROL_TOTAL_BYTES:
                    raise CampaignValidationError(
                        "training process boundary output control files exceed the aggregate byte limit"
                    )
                identity = OwnedFileIdentity.from_stat(metadata)
                try:
                    descriptor = open_child(name)
                except (OSError, UnsafeFilesystemOperation) as exc:
                    raise CampaignValidationError(
                        f"training process boundary output control file {name} cannot be retained"
                    ) from exc
                try:
                    content_sha256, captured_size = _snapshot_regular_content(
                        descriptor,
                        initial=metadata,
                        identity=identity,
                        boundary_device=int(metadata.st_dev),
                        label=f"training process boundary output control file {name}",
                    )
                finally:
                    os.close(descriptor)
                visible = stat_child(name)
                if not _stable_regular_file(
                    visible,
                    identity=identity,
                    initial=metadata,
                    boundary_device=int(metadata.st_dev),
                ):
                    raise CampaignValidationError(
                        f"training process boundary output control file {name} changed while being captured"
                    )
                bound_control_bytes += captured_size
        elif not stat.S_ISDIR(metadata.st_mode):
            raise CampaignValidationError("training process boundary output contains an unsupported object")
        rows.append(
            (
                name,
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(file_type),
                int(metadata.st_nlink),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                content_sha256,
            )
        )
    return tuple(rows)


def bootstrap_validated_training_process(
    config_path: str | Path,
    resume_path: str | Path | None,
) -> TrainingProcessBoundary | None:
    """Consume and verify the inherited parent boundary before training opens paths."""

    encoded = os.environ.pop(_TRAINING_PROCESS_BOUNDARY_ENV, None)
    if encoded is None:
        return None
    try:
        value = json.loads(encoded, object_pairs_hook=_strict_json_mapping)
    except (_DuplicateMappingKeyError, json.JSONDecodeError) as exc:
        raise CampaignValidationError("training process boundary payload is malformed") from exc
    if not isinstance(value, Mapping) or set(value) != _PROCESS_BOUNDARY_FIELDS:
        raise CampaignValidationError("training process boundary payload has an unsupported shape")
    if value.get("schema_version") != _TRAINING_PROCESS_BOUNDARY_SCHEMA:
        raise CampaignValidationError("training process boundary schema is unsupported")
    for field in (
        "campaign_identity",
        "run_identity",
        "root_entries_sha256",
        "config_content_sha256",
        "resolved_config_sha256",
    ):
        if not is_concrete_hash(value.get(field)):
            raise CampaignValidationError(f"training process boundary {field} is malformed")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CampaignValidationError("training process boundary run identity is malformed")
    project_root = _lexical_absolute(str(value.get("project_root") or ""), label="training project root")
    logical_output_root = require_confined_path(
        _lexical_absolute(str(value.get("logical_output_root") or ""), label="training output root"),
        project_root,
    )
    expected_config = require_confined_path(
        _lexical_absolute(str(value.get("config_path") or ""), label="training config path"),
        project_root,
    )
    requested_config = _lexical_absolute(config_path, label="training config path")
    if requested_config != expected_config:
        raise CampaignValidationError("training process command config differs from its inherited boundary")
    output_root = _child_output_root(value.get("root_token"), value.get("root_identity"), logical_output_root)
    child_entries_sha256 = stable_hash([list(row) for row in _child_entry_snapshot(value.get("root_token"))])
    if not hmac.compare_digest(child_entries_sha256, str(value["root_entries_sha256"])):
        raise CampaignValidationError("training process boundary output entries changed before child startup")
    config_descriptor, _config_identity = _child_file_descriptor(
        value.get("config_token"),
        value.get("config_identity"),
        expected_config,
        label="resolved run config",
    )
    config_bytes = _descriptor_bytes(
        config_descriptor,
        max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
        label="resolved run config",
    )
    if not hmac.compare_digest(hashlib.sha256(config_bytes).hexdigest(), str(value["config_content_sha256"])):
        raise CampaignValidationError("training process boundary config content identity changed")
    try:
        parsed_config = yaml.load(config_bytes.decode("utf-8"), Loader=_StrictSafeLoader)
    except (_DuplicateMappingKeyError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CampaignValidationError("training process boundary config cannot be parsed exactly") from exc
    if not isinstance(parsed_config, Mapping):
        raise CampaignValidationError("training process boundary config is not a mapping")
    config = deepcopy(dict(parsed_config))
    if stable_hash(config) != value["resolved_config_sha256"]:
        raise CampaignValidationError("training process boundary resolved config identity changed")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CampaignValidationError("training process boundary config runtime is malformed")
    configured_output = _lexical_absolute(str(runtime.get("out_dir") or ""), label="configured output root")
    if configured_output != logical_output_root:
        raise CampaignValidationError("training process boundary config output root changed")
    campaign_run_contract = value.get("campaign_run_contract")
    if not isinstance(campaign_run_contract, Mapping) or set(campaign_run_contract) != _CAMPAIGN_RUN_CONTRACT_FIELDS:
        raise CampaignValidationError("training process boundary campaign run contract is malformed")
    campaign_run_contract = deepcopy(dict(campaign_run_contract))
    if campaign_run_contract.get("schema_version") != _CAMPAIGN_RUN_CONTRACT_SCHEMA:
        raise CampaignValidationError("training process boundary campaign run contract schema is unsupported")
    if not hmac.compare_digest(
        stable_hash(campaign_run_contract),
        str(value["campaign_run_contract_sha256"]),
    ):
        raise CampaignValidationError("training process boundary campaign run contract identity changed")
    for field, expected in (
        ("campaign_identity", value["campaign_identity"]),
        ("run_identity", value["run_identity"]),
        ("run_id", run_id),
        ("output_root", str(logical_output_root)),
        ("resolved_config_sha256", value["resolved_config_sha256"]),
    ):
        if campaign_run_contract.get(field) != expected:
            raise CampaignValidationError(f"training process boundary campaign run contract {field} changed")
    if campaign_run_contract.get("resolved_config") != config or stable_hash(config) != campaign_run_contract.get(
        "resolved_config_sha256"
    ):
        raise CampaignValidationError("training process boundary campaign run contract config changed")
    for field in (
        "campaign_identity",
        "run_identity",
        "resolved_config_sha256",
        "execution_contract_sha256",
        "training_code_identity_sha256",
    ):
        if not is_concrete_hash(campaign_run_contract.get(field)):
            raise CampaignValidationError(f"training process boundary campaign run contract {field} is malformed")
    if (
        type(campaign_run_contract.get("seed")) is not int
        or type(campaign_run_contract.get("max_optimizer_steps")) is not int
    ):
        raise CampaignValidationError("training process boundary campaign run contract numeric fields are malformed")
    for field in ("expected_checkpoint_steps", "expected_evaluation_steps"):
        steps = campaign_run_contract.get(field)
        if not isinstance(steps, list) or any(type(step) is not int or step < 0 for step in steps):
            raise CampaignValidationError(f"training process boundary campaign run contract {field} is malformed")

    dataset_config = config.get("dataset")
    conditioning_config = config.get("conditioning")
    if not isinstance(dataset_config, Mapping) or not isinstance(conditioning_config, Mapping):
        raise CampaignValidationError("training process boundary retained input config is malformed")
    configured_dataset_root = _resolved_input_path(
        dataset_config.get("directory"),
        project_root,
        label="dataset root",
    )
    configured_manifest = _resolved_input_path(
        dataset_config.get("training_manifest"),
        project_root,
        label="training manifest",
    )
    configured_vocabulary = (
        None
        if not conditioning_config.get("vocabulary_path")
        else _resolved_input_path(
            conditioning_config.get("vocabulary_path"),
            project_root,
            label="conditioning vocabulary",
        )
    )
    input_files = value.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        raise CampaignValidationError("training process boundary retained input inventory is missing")
    manifest_bytes: bytes | None = None
    vocabulary_bytes: bytes | None = None
    dataset_descriptors: dict[str, int] = {}
    dataset_content_sha256: dict[str, str] = {}
    seen_input_paths: set[str] = set()
    for raw_input in input_files:
        if not isinstance(raw_input, Mapping) or set(raw_input) != _PROCESS_INPUT_FILE_FIELDS:
            raise CampaignValidationError("training process boundary retained input record is malformed")
        role = raw_input.get("role")
        logical_value = raw_input.get("logical_path")
        relative_value = raw_input.get("dataset_relative_path")
        content_sha256 = raw_input.get("content_sha256")
        if role not in {"training_manifest", "conditioning_vocabulary", "dataset_artifact"}:
            raise CampaignValidationError("training process boundary retained input role is unsupported")
        if not isinstance(logical_value, str) or not is_concrete_hash(content_sha256):
            raise CampaignValidationError("training process boundary retained input binding is malformed")
        logical_path = require_confined_path(
            _lexical_absolute(logical_value, label="retained input"),
            project_root,
        )
        path_key = os.path.normcase(str(logical_path))
        if path_key in seen_input_paths:
            raise CampaignValidationError("training process boundary retained input inventory is ambiguous")
        seen_input_paths.add(path_key)
        descriptor, _input_identity = _child_file_descriptor(
            raw_input.get("token"),
            raw_input.get("identity"),
            logical_path,
            label=str(role).replace("_", " "),
        )
        if not hmac.compare_digest(_descriptor_sha256(descriptor), str(content_sha256)):
            raise CampaignValidationError("training process boundary retained input content changed")
        if role == "training_manifest":
            if relative_value is not None or logical_path != configured_manifest or manifest_bytes is not None:
                raise CampaignValidationError("training process boundary training manifest binding is inconsistent")
            manifest_bytes = _descriptor_bytes(
                descriptor,
                max_bytes=_MAX_BOUND_INPUT_BYTES,
                label="training manifest",
            )
            _strict_manifest_records(manifest_bytes, label="training manifest")
        elif role == "conditioning_vocabulary":
            if relative_value is not None or logical_path != configured_vocabulary or vocabulary_bytes is not None:
                raise CampaignValidationError(
                    "training process boundary conditioning vocabulary binding is inconsistent"
                )
            vocabulary_bytes = _descriptor_bytes(
                descriptor,
                max_bytes=_MAX_BOUND_INPUT_BYTES,
                label="conditioning vocabulary",
            )
            try:
                vocabulary_payload = json.loads(
                    vocabulary_bytes.decode("utf-8"),
                    object_pairs_hook=_strict_json_mapping,
                )
            except (_DuplicateMappingKeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignValidationError("training process boundary conditioning vocabulary is malformed") from exc
            if not isinstance(vocabulary_payload, dict):
                raise CampaignValidationError("training process boundary conditioning vocabulary is not a mapping")
        else:
            if not isinstance(relative_value, str) or "\\" in relative_value:
                raise CampaignValidationError("training process boundary dataset artifact path is unsafe")
            relative = PurePosixPath(relative_value)
            if relative.is_absolute() or ".." in relative.parts or any(part in {"", "."} for part in relative.parts):
                raise CampaignValidationError("training process boundary dataset artifact path is unsafe")
            normalized = relative.as_posix()
            expected_dataset_path = require_confined_path(
                _lexical_absolute(
                    configured_dataset_root.joinpath(*relative.parts),
                    label="dataset artifact",
                ),
                configured_dataset_root,
            )
            if logical_path != expected_dataset_path or normalized in dataset_descriptors:
                raise CampaignValidationError("training process boundary dataset artifact binding is inconsistent")
            dataset_descriptors[normalized] = descriptor
            dataset_content_sha256[normalized] = str(content_sha256)
    if manifest_bytes is None or not dataset_descriptors:
        raise CampaignValidationError("training process boundary retained input inventory is incomplete")
    if (configured_vocabulary is None) is not (vocabulary_bytes is None):
        raise CampaignValidationError("training process boundary conditioning vocabulary inventory is incomplete")

    checkpoint_path_value = value.get("checkpoint_path")
    checkpoint_token_value = value.get("checkpoint_token")
    checkpoint_identity_value = value.get("checkpoint_identity")
    checkpoint_sha256 = value.get("checkpoint_content_sha256")
    resume_descriptor: int | None = None
    logical_resume: Path | None = None
    if checkpoint_path_value is None:
        if any(item is not None for item in (checkpoint_token_value, checkpoint_identity_value, checkpoint_sha256)):
            raise CampaignValidationError("training process boundary checkpoint fields are inconsistent")
        if resume_path is not None:
            raise CampaignResumeError("training process command unexpectedly requests resume")
    else:
        if not isinstance(checkpoint_path_value, str) or not is_concrete_hash(checkpoint_sha256):
            raise CampaignValidationError("training process boundary checkpoint binding is malformed")
        logical_resume = require_confined_path(
            _lexical_absolute(checkpoint_path_value, label="resume checkpoint"),
            project_root,
        )
        requested_resume = None if resume_path is None else _lexical_absolute(resume_path, label="resume checkpoint")
        if requested_resume != logical_resume or logical_resume.parent != logical_output_root:
            raise CampaignResumeError("training process command checkpoint differs from its inherited boundary")
        resume_descriptor, _checkpoint_identity = _child_file_descriptor(
            checkpoint_token_value,
            checkpoint_identity_value,
            logical_resume,
            label="resume checkpoint",
        )
        if not hmac.compare_digest(_descriptor_sha256(resume_descriptor), str(checkpoint_sha256)):
            raise CampaignResumeError("training process boundary checkpoint content identity changed")
    return TrainingProcessBoundary(
        config=config,
        project_root=project_root,
        logical_output_root=logical_output_root,
        output_root=output_root,
        resume_path=logical_resume,
        resume_descriptor=resume_descriptor,
        resume_sha256=None if checkpoint_sha256 is None else str(checkpoint_sha256),
        campaign_run_contract=campaign_run_contract,
        training_manifest_bytes=manifest_bytes,
        vocabulary_bytes=vocabulary_bytes,
        dataset_descriptors=dataset_descriptors,
        dataset_content_sha256=dataset_content_sha256,
    )


def _lexical_absolute(path: str | Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not raw or not raw.strip() or raw != raw.strip() or "\x00" in raw:
        raise CampaignValidationError(f"{label} is invalid")
    return Path(os.path.abspath(raw))


def _stable_regular_file(
    metadata: os.stat_result,
    *,
    identity: OwnedFileIdentity,
    initial: os.stat_result,
    boundary_device: int,
) -> bool:
    reparse = bool(int(getattr(metadata, "st_file_attributes", 0) or 0) & 0x400)
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not reparse
        and metadata.st_nlink == 1
        and metadata.st_dev == boundary_device
        and identity.matches(metadata)
        and metadata.st_size == initial.st_size
        and metadata.st_mtime_ns == initial.st_mtime_ns
    )


def _read_confined_regular_bytes(
    path: Path,
    project_root: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one exact single-link inode through an anchored project path."""

    try:
        confined = require_confined_path(path, project_root)
        with open_anchored_directory(confined.parent, project_root) as parent:
            before = parent.lstat(confined.name)
            boundary_device = parent.directory_metadata().st_dev
            if before.st_size < 0 or before.st_size > max_bytes:
                raise UnsafeFilesystemOperation("bounded file size is invalid")
            identity = OwnedFileIdentity.from_stat(before)
            if not _stable_regular_file(
                before,
                identity=identity,
                initial=before,
                boundary_device=boundary_device,
            ):
                raise UnsafeFilesystemOperation("bounded file is not a stable single-link regular file")
            descriptor = parent.open_file_immovable(
                confined.name,
                os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
            )
            try:
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    opened = os.fstat(handle.fileno())
                    if not _stable_regular_file(
                        opened,
                        identity=identity,
                        initial=before,
                        boundary_device=boundary_device,
                    ):
                        raise UnsafeFilesystemOperation("bounded file changed while being opened")
                    payload = handle.read(max_bytes + 1)
                    after = os.fstat(handle.fileno())
                    if (
                        len(payload) != before.st_size
                        or len(payload) > max_bytes
                        or not _stable_regular_file(
                            after,
                            identity=identity,
                            initial=before,
                            boundary_device=boundary_device,
                        )
                    ):
                        raise UnsafeFilesystemOperation("bounded file changed while being read")
                    visible = parent.lstat(confined.name)
                    if not _stable_regular_file(
                        visible,
                        identity=identity,
                        initial=before,
                        boundary_device=boundary_device,
                    ):
                        raise UnsafeFilesystemOperation("bounded file path changed while being read")
                    parent.verify()
                    return payload
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise CampaignValidationError(f"{label} is missing, unsafe, too large, or changed while being read") from exc


def _read_mapping_with_identity(
    path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    payload = _read_confined_regular_bytes(
        path,
        project_root,
        label="campaign configuration",
        max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
    )
    try:
        text = payload.decode("utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(text, Loader=_StrictSafeLoader)
        else:
            value = json.loads(text, object_pairs_hook=_strict_json_mapping)
    except _DuplicateMappingKeyError as exc:
        raise CampaignValidationError("campaign configuration contains an ambiguous mapping") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignValidationError("campaign configuration cannot be loaded") from exc
    if not isinstance(value, dict):
        raise CampaignValidationError("campaign configuration must contain a mapping")
    return value, hashlib.sha256(payload).hexdigest()


def _read_mapping(path: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    boundary = project_root or path.parent
    value, _identity = _read_mapping_with_identity(path, project_root=boundary)
    return value


def _strict_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise _DuplicateMappingKeyError
        mapping[key] = value
    return mapping


def _canonical_portable_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise CampaignValidationError(f"{label} must be a canonical relative path")
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if pure.is_absolute() or windows.is_absolute() or windows.drive:
        raise CampaignValidationError(f"{label} must be relative to its campaign document")
    if not pure.parts or pure.as_posix() != value:
        raise CampaignValidationError(f"{label} must be a canonical relative path")
    entered_descendant = False
    for part in pure.parts:
        if part == "..":
            if entered_descendant:
                raise CampaignValidationError(f"{label} must be a canonical relative path")
        else:
            entered_descendant = True
    return pure


def load_exact_campaign_configuration(
    path: str | Path,
    *,
    profile: str = "recommended",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and fully resolve the exact configured product or low-level campaign."""

    campaign, _manifest_identity, _source_graph_identity, _verified_file_hashes = _load_exact_campaign_configuration(
        path,
        profile=profile,
        project_root=project_root,
    )
    return campaign


def _load_exact_campaign_configuration(
    path: str | Path,
    *,
    profile: str,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], str, str, dict[Path, str]]:
    execution_root = (
        _lexical_absolute(project_root, label="training project root") if project_root is not None else None
    )
    source = _lexical_absolute(path, label="campaign configuration path")
    if execution_root is not None:
        try:
            source = require_confined_path(source, execution_root)
        except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
            raise CampaignValidationError("campaign configuration path escapes the approved project") from exc
    document, manifest_identity = _read_mapping_with_identity(
        source,
        project_root=execution_root or source.parent,
    )
    selected: Mapping[str, Any] = document
    source_graph_identity = manifest_identity
    verified_file_hashes: dict[Path, str] = {}
    selected_directory = source.parent
    profiles = document.get("product_profiles")
    selected_from_profile = isinstance(profiles, Mapping)
    if isinstance(profiles, Mapping):
        profile_entry = profiles.get(profile)
        if not isinstance(profile_entry, Mapping):
            raise CampaignValidationError(f"campaign profile {profile!r} is not configured")
        if isinstance(profile_entry.get("campaign"), Mapping):
            selected = profile_entry["campaign"]
        elif "campaign_path" in profile_entry:
            relative = _canonical_portable_relative_path(
                profile_entry["campaign_path"],
                label="campaign_path",
            )
            try:
                nested = require_confined_path(
                    source.parent.joinpath(*relative.parts),
                    source.parent,
                )
            except UnsafeFilesystemOperation as exc:
                raise CampaignValidationError(
                    "campaign_path escapes its configuration directory or crosses a link/reparse seam"
                ) from exc
            try:
                metadata = nested.lstat()
            except OSError as exc:
                raise CampaignValidationError("campaign_path cannot be verified") from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise CampaignValidationError("campaign_path must name a regular single-link file")
            selected, selected_identity = _read_mapping_with_identity(
                nested,
                project_root=execution_root or source.parent,
            )
            source_graph_identity = stable_hash(
                {
                    "campaign_configuration_sha256": manifest_identity,
                    "selected_campaign_sha256": selected_identity,
                }
            )
            selected_directory = nested.parent
        else:
            raise CampaignValidationError(f"campaign profile {profile!r} has no campaign configuration")
    campaign = dict(selected)
    if campaign.get("schema_version") == CAMPAIGN_SCHEMA_VERSION:
        if selected_from_profile:
            raise CampaignValidationError(
                "product profile campaigns must be unplanned specifications; "
                "materialized campaign manifests are forbidden"
            )
        if execution_root is not None:
            verified_file_hashes.update(_preflight_materialized_campaign_paths(campaign, execution_root))
        return campaign, manifest_identity, source_graph_identity, verified_file_hashes
    if selected_from_profile:
        if "output_root" not in campaign:
            raise CampaignValidationError("product profile campaign output_root is required")
        _canonical_portable_relative_path(campaign["output_root"], label="campaign output_root")
    if isinstance(profiles, Mapping) and execution_root is not None:
        campaign = _resolve_portable_campaign_paths(
            campaign,
            selected_directory,
            execution_root,
            verified_file_hashes=verified_file_hashes,
        )
    planned = plan_campaign(
        campaign,
        execution_root=execution_root,
        file_sha256_resolver=verified_file_hashes.__getitem__ if execution_root is not None else None,
    )
    if execution_root is not None:
        verified_file_hashes.update(_preflight_materialized_campaign_paths(planned, execution_root))
    return planned, manifest_identity, source_graph_identity, verified_file_hashes


def _resolve_portable_campaign_paths(
    spec: Mapping[str, Any],
    source_directory: Path,
    project_root: Path,
    *,
    verified_file_hashes: dict[Path, str],
) -> dict[str, Any]:
    """Resolve a product profile's portable bindings inside its approved project."""

    result = deepcopy(dict(spec))

    def resolve(value: Any, *, label: str, require_file: bool) -> str:
        pure = _canonical_portable_relative_path(value, label=label)
        try:
            candidate = require_confined_path(source_directory.joinpath(*pure.parts), project_root)
        except UnsafeFilesystemOperation as exc:
            raise CampaignValidationError(f"{label} escapes the approved project") from exc
        if require_file:
            payload = _read_confined_regular_bytes(
                candidate,
                project_root,
                label=label,
                max_bytes=_MAX_BOUND_INPUT_BYTES,
            )
            verified_file_hashes[candidate] = hashlib.sha256(payload).hexdigest()
        return str(candidate)

    identities = result.get("identities")
    if isinstance(identities, dict):
        for field in (
            "dataset_freeze_path",
            "dataset_view_manifest_path",
            "split_manifest_path",
            "conditioning_vocabulary_path",
        ):
            if identities.get(field) is not None:
                identities[field] = resolve(
                    identities[field],
                    label=f"campaign identity {field}",
                    require_file=True,
                )
    evaluation = result.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("benchmark_manifest_path") is not None:
        evaluation["benchmark_manifest_path"] = resolve(
            evaluation["benchmark_manifest_path"],
            label="campaign benchmark_manifest_path",
            require_file=True,
        )
    if result.get("output_root") is not None:
        result["output_root"] = resolve(
            result["output_root"],
            label="campaign output_root",
            require_file=False,
        )
    if result.get("campaign_artifact_root") is not None:
        result["campaign_artifact_root"] = resolve(
            result["campaign_artifact_root"],
            label="campaign campaign_artifact_root",
            require_file=False,
        )
    return result


def _confined_materialized_path(value: Any, project_root: Path, *, label: str) -> Path:
    if not isinstance(value, str):
        raise CampaignValidationError(f"{label} is invalid")
    candidate = _lexical_absolute(value, label=label)
    try:
        return require_confined_path(candidate, project_root)
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise CampaignValidationError(f"{label} escapes the approved project or crosses a link/reparse seam") from exc


def _confined_resolved_config_path(
    value: Any,
    project_root: Path,
    *,
    label: str,
    allow_project_root: bool = False,
    expected: Path | None = None,
) -> Path:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise CampaignValidationError(f"{label} is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if windows.is_absolute() or windows.drive:
        if os.name != "nt":
            raise CampaignValidationError(f"{label} is invalid")
        candidate = _lexical_absolute(value, label=label)
    elif posix.is_absolute():
        candidate = _lexical_absolute(value, label=label)
    elif "\\" in value:
        raise CampaignValidationError(f"{label} is invalid")
    elif value == "." and allow_project_root:
        candidate = project_root
    else:
        pure = _canonical_portable_relative_path(value, label=label)
        candidates = (project_root.joinpath(*pure.parts), _lexical_absolute(value, label=label))
        for relative_candidate in candidates:
            try:
                confined = require_confined_path(
                    relative_candidate,
                    project_root,
                    allow_root=allow_project_root,
                )
            except (OSError, UnsafeFilesystemOperation, ValueError):
                continue
            if expected is None or confined == expected:
                return confined
        raise CampaignValidationError(f"{label} does not bind the exact campaign path")
    try:
        return require_confined_path(candidate, project_root, allow_root=allow_project_root)
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise CampaignValidationError(f"{label} escapes the approved project or crosses a link/reparse seam") from exc


def _verify_materialized_output_path(value: Any, project_root: Path, *, label: str) -> Path:
    candidate = _confined_materialized_path(value, project_root, label=label)
    current = candidate
    while not os.path.lexists(current) and current.parent != current:
        current = current.parent
    try:
        with open_anchored_directory(current, project_root):
            pass
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise CampaignValidationError(f"{label} is not a stable project-contained output path") from exc
    return candidate


def _verify_materialized_input_path(
    value: Any,
    project_root: Path,
    *,
    label: str,
    expected_sha256: Any = None,
    verified_file_hashes: dict[Path, str] | None = None,
) -> Path:
    candidate = _confined_materialized_path(value, project_root, label=label)
    payload = _read_confined_regular_bytes(
        candidate,
        project_root,
        label=label,
        max_bytes=_MAX_BOUND_INPUT_BYTES,
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise CampaignValidationError(f"{label} identity changed")
    if verified_file_hashes is not None:
        verified_file_hashes[candidate] = actual_sha256
    return candidate


def _preflight_materialized_campaign_paths(
    campaign: Mapping[str, Any],
    project_root: Path,
) -> dict[Path, str]:
    """Reject every external or unstable path before campaign validation/audit."""

    verified_file_hashes: dict[Path, str] = {}
    identities = campaign.get("identities")
    if isinstance(identities, Mapping):
        for path_field, hash_field in (
            ("dataset_freeze_path", None),
            ("dataset_view_manifest_path", "dataset_view_manifest_hash"),
            ("split_manifest_path", "split_manifest_hash"),
            ("conditioning_vocabulary_path", "conditioning_vocabulary_hash"),
        ):
            value = identities.get(path_field)
            if value is not None:
                _verify_materialized_input_path(
                    value,
                    project_root,
                    label=f"campaign identity {path_field}",
                    expected_sha256=identities.get(hash_field) if hash_field is not None else None,
                    verified_file_hashes=verified_file_hashes,
                )

    evaluation = campaign.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("benchmark_manifest_path") is not None:
        _verify_materialized_input_path(
            evaluation["benchmark_manifest_path"],
            project_root,
            label="campaign benchmark_manifest_path",
            expected_sha256=evaluation.get("benchmark_manifest_hash"),
            verified_file_hashes=verified_file_hashes,
        )

    artifact_root = campaign.get("campaign_artifact_root")
    if artifact_root is not None:
        _verify_materialized_output_path(
            artifact_root,
            project_root,
            label="campaign campaign_artifact_root",
        )

    expected_roots = campaign.get("expected_output_roots")
    if not isinstance(expected_roots, list):
        raise CampaignValidationError("campaign expected_output_roots is invalid")
    for value in expected_roots:
        _verify_materialized_output_path(value, project_root, label="campaign expected output_root")

    runs = campaign.get("expected_runs")
    if not isinstance(runs, list):
        raise CampaignValidationError("campaign expected_runs is invalid")
    for run in runs:
        if not isinstance(run, Mapping):
            raise CampaignValidationError("campaign expected run is invalid")
        _verify_materialized_output_path(
            run.get("output_root"),
            project_root,
            label="campaign run output_root",
        )
        config_path = _confined_materialized_path(
            run.get("resolved_config_path"),
            project_root,
            label="campaign run resolved_config_path",
        )
        resolved = run.get("resolved_config")
        if not isinstance(resolved, Mapping):
            raise CampaignValidationError("campaign run resolved_config is invalid")
        dataset = resolved.get("dataset")
        conditioning = resolved.get("conditioning")
        runtime = resolved.get("runtime")
        if (
            not isinstance(dataset, Mapping)
            or not isinstance(conditioning, Mapping)
            or not isinstance(runtime, Mapping)
        ):
            raise CampaignValidationError("campaign run resolved_config path bindings are invalid")
        manifest_path = _confined_materialized_path(
            identities.get("split_manifest_path") if isinstance(identities, Mapping) else None,
            project_root,
            label="campaign identity split_manifest_path",
        )
        training_manifest = _confined_resolved_config_path(
            dataset.get("training_manifest"),
            project_root,
            label="resolved config dataset training_manifest",
            expected=manifest_path,
        )
        split_manifest = _confined_resolved_config_path(
            dataset.get("split_manifest"),
            project_root,
            label="resolved config dataset split_manifest",
            expected=manifest_path,
        )
        dataset_directory = _confined_resolved_config_path(
            dataset.get("directory"),
            project_root,
            label="resolved config dataset directory",
            allow_project_root=True,
            expected=manifest_path.parent,
        )
        expected_vocabulary = _confined_materialized_path(
            identities.get("conditioning_vocabulary_path") if isinstance(identities, Mapping) else None,
            project_root,
            label="campaign identity conditioning_vocabulary_path",
        )
        vocabulary_path = _confined_resolved_config_path(
            conditioning.get("vocabulary_path"),
            project_root,
            label="resolved config conditioning vocabulary_path",
            expected=expected_vocabulary,
        )
        run_output = _confined_materialized_path(
            run.get("output_root"),
            project_root,
            label="campaign run output_root",
        )
        runtime_output = _confined_resolved_config_path(
            runtime.get("out_dir"),
            project_root,
            label="resolved config runtime out_dir",
            expected=run_output,
        )
        if (
            training_manifest != manifest_path
            or split_manifest != manifest_path
            or dataset_directory != manifest_path.parent
            or vocabulary_path != expected_vocabulary
            or runtime_output != run_output
        ):
            raise CampaignValidationError("campaign run resolved_config path bindings do not match the exact campaign")
        command = run.get("experiment_command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise CampaignValidationError("campaign run experiment command is invalid")
        expected_command = (
            sys.executable,
            "-m",
            "spritelab",
            "train",
            "experiment",
            "run",
            "--config",
            str(run.get("resolved_config_path")),
        )
        if tuple(command) != expected_command:
            raise CampaignValidationError("campaign run experiment command does not match the exact launch command")
        command_config = _confined_materialized_path(
            command[-1],
            project_root,
            label="campaign run command config path",
        )
        if command_config != config_path:
            raise CampaignValidationError("campaign run command config path does not match resolved_config_path")
        if os.path.lexists(config_path):
            _read_confined_regular_bytes(
                config_path,
                project_root,
                label="resolved run config",
                max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
            )
    return verified_file_hashes


def _audit_resume_from_filesystem_snapshot(
    campaign: Mapping[str, Any],
    project_root: Path,
    filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
) -> dict[str, Any]:
    snapshot = filesystem_snapshot
    if snapshot is None:
        snapshot = _capture_fresh_campaign_filesystem_snapshot(campaign, project_root)
    return audit_resume(campaign, unsafe_resume=False, filesystem_snapshot=snapshot)


def _event_migration_from_resume_state(state: Mapping[str, Any]) -> EventMigrationVerification:
    value = state.get("event_migration_verification")
    if not isinstance(value, Mapping) and state.get("status") == "fresh":
        run_id = str(state.get("run_id") or "")
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        details = {
            "canonical_present": False,
            "canonical_size_bytes": 0,
            "canonical_sha256": empty_sha256,
            "event_history_origin": "native",
            "origin_record_present": False,
            "migration_required": False,
            "canonical_prefix_size_bytes": 0,
            "canonical_prefix_sha256": empty_sha256,
        }
        value = {
            "state": EventMigrationState.NO_MIGRATION.value,
            "run_id": run_id,
            "evidence_sha256": stable_hash(
                {
                    "schema_version": MIGRATION_EVIDENCE_SCHEMA,
                    "state": EventMigrationState.NO_MIGRATION.value,
                    "run_id": run_id,
                    "details": details,
                }
            ),
            "message": "No legacy migration is recorded or required.",
            "record": None,
            "details": details,
        }
    if not isinstance(value, Mapping):
        raise CampaignResumeError("resume audit did not retain exact event-migration evidence")
    try:
        migration_state = EventMigrationState(str(value.get("state")))
    except ValueError as exc:
        raise CampaignResumeError("resume audit retained an invalid event-migration state") from exc
    details = value.get("details")
    record = value.get("record")
    if not isinstance(details, Mapping) or (record is not None and not isinstance(record, Mapping)):
        raise CampaignResumeError("resume audit retained malformed event-migration evidence")
    verification = EventMigrationVerification(
        migration_state,
        str(value.get("run_id") or ""),
        str(value.get("evidence_sha256") or ""),
        str(value.get("message") or ""),
        deepcopy(dict(record)) if isinstance(record, Mapping) else None,
        deepcopy(dict(details)),
    )
    expected_evidence = stable_hash(
        {
            "schema_version": MIGRATION_EVIDENCE_SCHEMA,
            "state": verification.state.value,
            "run_id": verification.run_id,
            "details": dict(verification.details or {}),
        }
    )
    if not hmac.compare_digest(expected_evidence, verification.evidence_sha256):
        raise CampaignResumeError("resume audit event-migration evidence identity changed")
    return verification


def _normalise_environment(environment: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    supplied = dict(environment or {})
    if os.name == "nt":
        for required in ("SYSTEMROOT", "WINDIR"):
            ambient = os.environ.get(required)
            if ambient is not None and required not in supplied:
                supplied[required] = ambient
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_key, raw_value in supplied.items():
        key, value = str(raw_key), str(raw_value)
        if not key or "\x00" in key or "=" in key or "\x00" in value:
            raise CampaignValidationError("execution environment contains an unsafe key or NUL byte")
        normalized_key = key.upper()
        if normalized_key in seen:
            raise CampaignValidationError("execution environment contains an ambiguous duplicate key")
        seen.add(normalized_key)
        if normalized_key in {_TRAINING_PROCESS_BOUNDARY_ENV, _TRAINING_CODE_BUNDLE_ENV}:
            raise CampaignValidationError("execution environment contains a reserved process-boundary key")
        if normalized_key in _DANGEROUS_TRAINING_ENV_NAMES or normalized_key.startswith(
            _DANGEROUS_TRAINING_ENV_PREFIXES
        ):
            raise CampaignValidationError(f"execution environment contains forbidden loader key: {key}")
        values.append((key, value))
    return tuple(sorted(values))


def _output_root_identity(campaign: Mapping[str, Any], run: Mapping[str, Any], project_root: Path) -> str:
    output_root = _confined_materialized_path(
        run.get("output_root"),
        project_root,
        label="campaign run output_root",
    )
    return stable_hash(
        {
            "campaign_identity": campaign.get("campaign_identity"),
            "run_identity": run.get("run_identity"),
            "output_root": str(output_root),
        }
    )


def _campaign_validator_authorization_evidence(
    *,
    campaign_config_path: Path,
    campaign_profile: str,
    project_root: Path,
    run_id: str,
) -> str:
    """Bind direct campaign launches to the exact validator authorization inputs."""

    return stable_hash(
        {
            "schema_version": "spritelab_campaign_validator_authorization_v1",
            "campaign_config_path": str(campaign_config_path),
            "campaign_profile": campaign_profile,
            "project_root": str(project_root),
            "run_id": run_id,
        }
    )


def _execution_spec(
    *,
    campaign: Mapping[str, Any],
    run: Mapping[str, Any],
    backend_id: str,
    project_root: Path,
    argv: Sequence[str],
    environment: tuple[tuple[str, str], ...],
    resume: bool,
    source_checkpoint_identity: str | None,
    campaign_source_graph_sha256: str,
    launch_authorization_evidence_sha256: str,
) -> dict[str, Any]:
    output_root = _confined_materialized_path(
        run.get("output_root"),
        project_root,
        label="campaign run output_root",
    )
    return {
        "schema_version": "spritelab_training_execution_spec_v2",
        "campaign_identity_sha256": campaign.get("campaign_identity"),
        "campaign_source_graph_sha256": campaign_source_graph_sha256,
        "run_identity": run.get("run_identity"),
        "cell_id": run.get("cell_id"),
        "seed": run.get("seed"),
        "backend_id": backend_id,
        "launch_authorization_evidence_sha256": launch_authorization_evidence_sha256,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "argv": list(argv),
        "environment": dict(environment),
        "resume": resume,
        "source_checkpoint_identity": source_checkpoint_identity,
    }


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignValidationError(f"receipt {label} is malformed") from exc
    if parsed.tzinfo is None:
        raise CampaignValidationError(f"receipt {label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validator_proof(body: Mapping[str, Any]) -> str:
    protected = {key: value for key, value in body.items() if key not in {"validator_proof_sha256", "receipt_sha256"}}
    return hmac.new(_VALIDATOR_ISSUER_KEY, canonical_json(protected).encode("utf-8"), hashlib.sha256).hexdigest()


def _authoritative_snapshot(
    context: TrainingLaunchContext,
    *,
    backend_id: str,
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    output_root: str | Path | None = None,
    require_materialized_configs: bool = True,
    filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[tuple[str, str], ...], dict[str, Any]]:
    if context.schema_version != TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION:
        raise CampaignValidationError("training launch validator context schema is unsupported")
    if type(context.resume) is not bool:
        raise CampaignValidationError("training launch resume flag must be a strict boolean")
    if not is_concrete_hash(context.launch_authorization_evidence_sha256):
        raise CampaignValidationError("training launch authorization evidence identity must be concrete")
    if not backend_id.strip():
        raise CampaignValidationError("compute backend identity is required")
    (
        campaign,
        campaign_manifest_sha256,
        campaign_source_graph_sha256,
        verified_file_hashes,
    ) = _load_exact_campaign_configuration(
        context.campaign_config_path,
        profile=context.campaign_profile,
        project_root=context.project_root,
    )
    validation = validate_campaign(campaign, file_sha256_resolver=verified_file_hashes.__getitem__)
    problems = [*validation["errors"], *validation["blockers"]]
    if problems or not validation["launch_ready"]:
        raise CampaignValidationError("campaign validation failed: " + "; ".join(problems or ["not launch-ready"]))
    if not campaign.get("executable") or campaign.get("plan_status") != "ready":
        raise CampaignValidationError("campaign is blocked or executable=false")
    if campaign.get("launch_authorized") is not True:
        raise CampaignValidationError("campaign launch_authorized must be true")
    run = next((item for item in campaign.get("expected_runs", ()) if item.get("run_id") == context.run_id), None)
    if not isinstance(run, Mapping):
        raise CampaignValidationError(f"run {context.run_id!r} is not an exact campaign cell")
    expected_root = _verify_materialized_output_path(
        run["output_root"],
        context.project_root,
        label="campaign run output_root",
    )
    resume_report = _audit_resume_from_filesystem_snapshot(
        campaign,
        context.project_root,
        filesystem_snapshot,
    )
    if not resume_report["safe"]:
        raise CampaignResumeError("unsafe campaign state: " + "; ".join(resume_report["errors"]))
    state = next(item for item in resume_report["runs"] if item["run_id"] == context.run_id)
    must_resume = state["status"] == "valid_resumable"
    if context.resume != must_resume:
        raise CampaignResumeError("launch resume mode does not match the current authoritative output-root state")
    expected_command = tuple(str(item) for item in run.get("experiment_command") or ())
    source_checkpoint_identity: str | None = None
    if must_resume:
        checkpoint = Path(str(state["checkpoint"]))
        source_checkpoint_identity = state.get("checkpoint_content_sha256")
        if not is_concrete_hash(source_checkpoint_identity):
            raise CampaignResumeError("resume audit did not retain an exact checkpoint content identity")
        expected_command = (*expected_command, "--resume", str(checkpoint))
    command = expected_command if argv is None else tuple(str(item) for item in argv)
    if command != expected_command:
        raise CampaignValidationError("requested argv does not match the exact campaign execution contract")
    normalised_environment = _normalise_environment(environment)
    if normalised_environment != context.environment:
        raise CampaignValidationError("requested environment changed after launch validation")
    if output_root is not None:
        requested_root = _confined_materialized_path(
            os.fspath(output_root),
            context.project_root,
            label="requested output root",
        )
        if requested_root != expected_root:
            raise CampaignValidationError("requested output root does not match the exact campaign run")
    migration_verification = _event_migration_from_resume_state(state)
    if migration_verification.run_id != str(run["run_id"]):
        raise CampaignResumeError("resume audit event-migration evidence belongs to a different run")
    if not migration_verification.resume_compatible:
        raise CampaignResumeError(
            "event migration evidence is not safe for continuation: "
            f"{migration_verification.state.value}: {migration_verification.message}"
        )
    if must_resume and migration_verification.migration_required and not migration_verification.migration_verified:
        raise CampaignResumeError(
            "a recorded migrated event history must fully verify before continuation: "
            f"{migration_verification.state.value}: {migration_verification.message}"
        )
    for candidate in campaign.get("expected_runs", ()):
        path = _confined_materialized_path(
            candidate["resolved_config_path"],
            context.project_root,
            label="campaign run resolved_config_path",
        )
        if not require_materialized_configs:
            if stable_hash(candidate.get("resolved_config")) != candidate.get("resolved_config_sha256"):
                raise CampaignValidationError("embedded resolved run config identity changed")
            continue
        try:
            payload = _read_confined_regular_bytes(
                path,
                context.project_root,
                label="resolved run config",
                max_bytes=_MAX_CAMPAIGN_MAPPING_BYTES,
            )
            actual = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_json_mapping)
        except _DuplicateMappingKeyError as exc:
            raise CampaignValidationError("resolved run config contains an ambiguous mapping") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignValidationError("resolved run config cannot be read") from exc
        if not isinstance(actual, dict):
            raise CampaignValidationError("resolved run config must contain a mapping")
        if actual != candidate.get("resolved_config") or stable_hash(actual) != candidate.get("resolved_config_sha256"):
            raise CampaignValidationError("resolved run config identity changed")
    execution_spec = _execution_spec(
        campaign=campaign,
        run=run,
        backend_id=backend_id,
        project_root=context.project_root,
        argv=command,
        environment=normalised_environment,
        resume=must_resume,
        source_checkpoint_identity=source_checkpoint_identity,
        campaign_source_graph_sha256=campaign_source_graph_sha256,
        launch_authorization_evidence_sha256=context.launch_authorization_evidence_sha256,
    )
    snapshot = {
        "campaign": campaign,
        "run": dict(run),
        "validation": validation,
        "resume_report": resume_report,
        "event_migration_verification": migration_verification,
        "source_checkpoint_identity": source_checkpoint_identity,
        "execution_spec": execution_spec,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "campaign_source_graph_sha256": campaign_source_graph_sha256,
        "output_root": expected_root,
    }
    return campaign, dict(run), command, normalised_environment, snapshot


def validate_training_launch_plan(
    campaign_config_path: str | Path,
    *,
    compute_backend_id: str,
    project_root: str | Path,
    campaign_profile: str = "recommended",
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Perform the full CPU-only dry-run validation without issuing a receipt."""

    config_path = _lexical_absolute(campaign_config_path, label="campaign configuration path")
    execution_root = _lexical_absolute(project_root, label="training project root")
    (
        campaign,
        campaign_manifest_sha256,
        campaign_source_graph_sha256,
        verified_file_hashes,
    ) = _load_exact_campaign_configuration(
        config_path,
        profile=campaign_profile,
        project_root=execution_root,
    )
    resume_report = _audit_resume_from_filesystem_snapshot(campaign, execution_root)
    launches: list[dict[str, Any]] = []
    states = {item["run_id"]: item for item in resume_report.get("runs", ())}
    for run in campaign.get("expected_runs", ()):
        run_id = str(run["run_id"])
        context = TrainingLaunchContext(
            schema_version=TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION,
            campaign_config_path=config_path,
            campaign_profile=campaign_profile,
            project_root=execution_root,
            run_id=run_id,
            resume=states.get(run["run_id"], {}).get("status") == "valid_resumable",
            launch_authorization_evidence_sha256=_campaign_validator_authorization_evidence(
                campaign_config_path=config_path,
                campaign_profile=campaign_profile,
                project_root=execution_root,
                run_id=run_id,
            ),
            environment=_normalise_environment(environment),
        )
        _, current_run, argv, _, snapshot = _authoritative_snapshot(
            context,
            backend_id=compute_backend_id,
            environment=dict(context.environment),
            require_materialized_configs=False,
        )
        if (
            snapshot["campaign_manifest_sha256"] != campaign_manifest_sha256
            or snapshot["campaign_source_graph_sha256"] != campaign_source_graph_sha256
        ):
            raise CampaignValidationError("campaign configuration changed during launch validation")
        launches.append(
            {
                "run_id": current_run["run_id"],
                "run_identity": current_run["run_identity"],
                "seed": current_run["seed"],
                "argv_sha256": stable_hash(list(argv)),
                "execution_spec_sha256": stable_hash(snapshot["execution_spec"]),
                "resume": context.resume,
            }
        )
    return {
        "schema_version": "spritelab_training_launch_dry_run_v1",
        "campaign_id": campaign.get("campaign_id"),
        "campaign_identity_sha256": campaign.get("campaign_identity"),
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "campaign_source_graph_sha256": campaign_source_graph_sha256,
        "compute_backend_id": compute_backend_id,
        "validation": validate_campaign(campaign, file_sha256_resolver=verified_file_hashes.__getitem__),
        "resume_validation": resume_report,
        "launches": launches,
        "valid": bool(launches) and resume_report.get("safe") is True,
        "receipts_issued": 0,
        "processes_started": 0,
    }


def prepare_validated_training_launch(
    campaign_config_path: str | Path,
    *,
    run_id: str,
    compute_backend_id: str,
    project_root: str | Path,
    execute_confirmed: bool,
    campaign_profile: str = "recommended",
    environment: Mapping[str, str] | None = None,
    resume: bool = False,
    now: datetime | None = None,
    filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
    launch_authorization_evidence_sha256: str | None = None,
) -> ValidatedTrainingLaunch:
    """Run every launch gate in order and issue one short-lived receipt."""

    if execute_confirmed is not True:
        raise CampaignValidationError("training launch requires explicit execution confirmation")
    if type(resume) is not bool:
        raise CampaignValidationError("training launch resume flag must be a strict boolean")
    config_path = _lexical_absolute(campaign_config_path, label="campaign configuration path")
    execution_root = _lexical_absolute(project_root, label="training project root")
    profile = str(campaign_profile)
    selected_run_id = str(run_id)
    authorization_evidence = launch_authorization_evidence_sha256
    if authorization_evidence is None:
        authorization_evidence = _campaign_validator_authorization_evidence(
            campaign_config_path=config_path,
            campaign_profile=profile,
            project_root=execution_root,
            run_id=selected_run_id,
        )
    if not is_concrete_hash(authorization_evidence):
        raise CampaignValidationError("training launch authorization evidence identity must be concrete")
    context = TrainingLaunchContext(
        schema_version=TRAINING_LAUNCH_CONTEXT_SCHEMA_VERSION,
        campaign_config_path=config_path,
        campaign_profile=profile,
        project_root=execution_root,
        run_id=selected_run_id,
        resume=resume,
        launch_authorization_evidence_sha256=authorization_evidence,
        environment=_normalise_environment(environment),
    )
    campaign, run, argv, normalised_environment, snapshot = _authoritative_snapshot(
        context,
        backend_id=compute_backend_id,
        environment=dict(context.environment),
        filesystem_snapshot=filesystem_snapshot,
    )
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = created + timedelta(seconds=TRAINING_LAUNCH_RECEIPT_TTL_SECONDS)
    identities = dict(campaign.get("identities") or {})
    code_identity = dict(campaign.get("code_identity") or {})
    cell = next(item for item in campaign.get("architecture_cells", ()) if item.get("cell_id") == run.get("cell_id"))
    body: dict[str, Any] = {
        "schema_version": TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION,
        "receipt_id": secrets.token_hex(16),
        "campaign_identity_sha256": campaign["campaign_identity"],
        "campaign_manifest_sha256": snapshot["campaign_manifest_sha256"],
        "campaign_validation_report_sha256": stable_hash(snapshot["validation"]),
        "training_code_identity_sha256": code_identity.get("sha256"),
        "resolved_configuration_sha256": run.get("resolved_config_sha256"),
        "dataset_identity": identities.get("dataset_identity_hash", identities.get("dataset_view_manifest_hash")),
        "view_identity": identities.get("dataset_view_manifest_hash"),
        "split_identity": identities.get("split_manifest_hash"),
        "architecture_identity": stable_hash(cell),
        "optimizer_identity": identities.get("optimizer_config_hash"),
        "schedule_identity": identities.get("schedule_config_hash"),
        "loss_identity": identities.get("loss_config_hash"),
        "maximum_optimizer_steps": dict(campaign.get("training") or {}).get("max_optimizer_steps"),
        "run_identity": run.get("run_identity"),
        "cell_identity": stable_hash({"cell_id": run.get("cell_id"), "run_identity": run.get("run_identity")}),
        "seed": run.get("seed"),
        "output_root_identity": _output_root_identity(campaign, run, context.project_root),
        "compute_backend_id": compute_backend_id,
        "launch_authorization_evidence_sha256": context.launch_authorization_evidence_sha256,
        "execution_spec_sha256": stable_hash(snapshot["execution_spec"]),
        "argv_sha256": stable_hash(list(argv)),
        "resume_validation_sha256": stable_hash(snapshot["resume_report"]),
        "event_migration_state": snapshot["event_migration_verification"].state.value,
        "event_migration_identity_sha256": snapshot["event_migration_verification"].evidence_sha256,
        "event_history_origin": snapshot["event_migration_verification"].event_history_origin,
        "event_migration_required": snapshot["event_migration_verification"].migration_required,
        "event_migration_record_sha256": snapshot["event_migration_verification"].migration_record_sha256,
        "event_canonical_prefix_sha256": snapshot["event_migration_verification"].canonical_prefix_sha256,
        "event_canonical_identity_sha256": snapshot["event_migration_verification"].canonical_event_identity_sha256,
        "source_checkpoint_identity": snapshot["source_checkpoint_identity"],
        "unsafe_resume": False,
        "launch_authorized": True,
        "execute_confirmed": True,
        "created_at_utc": created.isoformat(),
        "expires_at_utc": expires.isoformat(),
    }
    hash_fields = (
        "campaign_identity_sha256",
        "campaign_manifest_sha256",
        "campaign_validation_report_sha256",
        "training_code_identity_sha256",
        "resolved_configuration_sha256",
        "dataset_identity",
        "view_identity",
        "split_identity",
        "architecture_identity",
        "optimizer_identity",
        "schedule_identity",
        "loss_identity",
        "run_identity",
        "cell_identity",
        "output_root_identity",
        "launch_authorization_evidence_sha256",
        "execution_spec_sha256",
        "argv_sha256",
        "resume_validation_sha256",
        "event_migration_identity_sha256",
        "event_canonical_prefix_sha256",
        "event_canonical_identity_sha256",
    )
    invalid = [field for field in hash_fields if not is_concrete_hash(body.get(field))]
    if invalid:
        raise CampaignValidationError("launch receipt has non-concrete protected identities: " + ", ".join(invalid))
    if body["event_history_origin"] not in EVENT_HISTORY_ORIGIN_RECEIPT_STATES:
        raise CampaignValidationError("launch receipt event-history origin is not a controlled origin state")
    if type(body["event_migration_required"]) is not bool:
        raise CampaignValidationError("launch receipt event-migration-required flag must be a strict boolean")
    optional_event_hash_fields = ("event_migration_record_sha256",)
    malformed_event_hashes = [
        field
        for field in optional_event_hash_fields
        if body.get(field) is not None and not is_concrete_hash(body.get(field))
    ]
    if malformed_event_hashes:
        raise CampaignValidationError(
            "launch receipt has malformed event evidence identities: " + ", ".join(malformed_event_hashes)
        )
    if body["event_migration_required"] and (
        body["event_migration_record_sha256"] is None or body["event_canonical_prefix_sha256"] is None
    ):
        raise CampaignValidationError(
            "launch receipt requires concrete migration-record and canonical-prefix identities for migrated runs"
        )
    if (body["event_history_origin"] == "migrated_legacy") is not body["event_migration_required"]:
        raise CampaignValidationError("launch receipt origin and migration-required classification disagree")
    if not body["event_migration_required"] and body["event_migration_record_sha256"] is not None:
        raise CampaignValidationError("native launch receipt cannot carry a migration-record identity")
    body["validator_proof_sha256"] = _validator_proof(body)
    body["receipt_sha256"] = stable_hash(body)
    receipt = TrainingLaunchReceipt(**body)
    return ValidatedTrainingLaunch(
        receipt,
        context,
        campaign,
        run,
        argv,
        dict(normalised_environment),
        snapshot["output_root"],
    )


def verify_validated_training_launch(
    receipt: TrainingLaunchReceipt,
    context: TrainingLaunchContext,
    *,
    compute_backend_id: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    output_root: str | Path,
    campaign_identity: str,
    run_identity: str,
    now: datetime | None = None,
    filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
) -> ValidatedTrainingLaunch:
    """Recompute authoritative state immediately before the process/transport seam."""

    if not isinstance(receipt, TrainingLaunchReceipt):
        raise CampaignValidationError("a typed validator-issued training launch receipt is required")
    if receipt.schema_version != TRAINING_LAUNCH_RECEIPT_SCHEMA_VERSION:
        raise CampaignValidationError("training launch receipt schema is unsupported")
    if stable_hash(receipt.body()) != receipt.receipt_sha256:
        raise CampaignValidationError("training launch receipt self-hash is invalid")
    if not hmac.compare_digest(_validator_proof(receipt.body()), receipt.validator_proof_sha256):
        raise CampaignValidationError("training launch receipt was not issued by the active validator")
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created = _parse_utc(receipt.created_at_utc, label="created_at_utc")
    expires = _parse_utc(receipt.expires_at_utc, label="expires_at_utc")
    if expires <= created or expires - created > timedelta(seconds=TRAINING_LAUNCH_RECEIPT_TTL_SECONDS):
        raise CampaignValidationError("training launch receipt lifetime is invalid")
    if checked_at < created - timedelta(seconds=5) or checked_at > expires:
        raise CampaignValidationError("training launch receipt is expired or not yet valid")
    if (
        receipt.unsafe_resume is not False
        or receipt.launch_authorized is not True
        or receipt.execute_confirmed is not True
    ):
        raise CampaignValidationError("training launch receipt does not prove safe authorization and confirmation")
    if not is_concrete_hash(campaign_identity) or not is_concrete_hash(run_identity):
        raise CampaignValidationError("compute request campaign and run identities must be concrete SHA-256 values")
    if campaign_identity != receipt.campaign_identity_sha256 or run_identity != receipt.run_identity:
        raise CampaignValidationError("compute request identity does not match its launch receipt")
    if compute_backend_id != receipt.compute_backend_id:
        raise CampaignValidationError("compute backend does not match the launch receipt")
    normalised_environment = _normalise_environment(environment)
    if normalised_environment != context.environment:
        raise CampaignValidationError("requested environment changed after launch validation")
    expected_launch = prepare_validated_training_launch(
        context.campaign_config_path,
        run_id=context.run_id,
        compute_backend_id=compute_backend_id,
        project_root=context.project_root,
        execute_confirmed=True,
        campaign_profile=context.campaign_profile,
        environment=dict(context.environment),
        resume=context.resume,
        now=created,
        filesystem_snapshot=filesystem_snapshot,
        launch_authorization_evidence_sha256=context.launch_authorization_evidence_sha256,
    )
    expected = expected_launch.receipt
    ignored = {"receipt_id", "validator_proof_sha256", "receipt_sha256"}
    stale = [key for key, value in expected.body().items() if key not in ignored and receipt.body().get(key) != value]
    if stale:
        raise CampaignValidationError("training launch receipt is stale or forged: " + ", ".join(sorted(stale)))
    if tuple(str(item) for item in argv) != expected_launch.argv:
        raise CampaignValidationError("requested argv does not match the exact campaign execution contract")
    requested_root = _confined_materialized_path(
        os.fspath(output_root),
        context.project_root,
        label="requested output root",
    )
    if requested_root != expected_launch.output_root:
        raise CampaignValidationError("requested output root does not match the exact campaign run")
    if (
        expected_launch.campaign.get("campaign_identity") != campaign_identity
        or expected_launch.run.get("run_identity") != run_identity
    ):
        raise CampaignValidationError("compute request identity changed during authoritative verification")
    return replace(expected_launch, receipt=receipt)


def receipt_with_recomputed_hash(receipt: TrainingLaunchReceipt, **changes: Any) -> TrainingLaunchReceipt:
    """Test/evidence helper: a self-consistent body still needs authoritative verification."""

    changed = replace(receipt, **changes, receipt_sha256="")
    return replace(changed, receipt_sha256=stable_hash(changed.body()))

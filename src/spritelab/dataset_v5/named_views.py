"""Contract-strict Dataset-v5 named views and immutable synthetic freezes.

The legacy Dataset-v5 preview builders remain intentionally separate.  This
module implements the seven-view contract under ``experiments/v5_view_contract_v1``
without treating existing preview rows as production truth.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from spritelab.dataset_v5.builder import _normalized_alpha_key, alpha_mask_sha256, canonical_rgba_sha256

CONTRACT_VERSION = "dataset_v5_view_contract_v1.0.0"
VIEW_MANIFEST_SCHEMA = "dataset_v5_view_manifest_v1.0.0"
RECORD_SCHEMA = "dataset_v5_record_v1.0.0"
SOURCE_RECORD_SCHEMA = "dataset_v5_source_record_v1.0.0"
POLICY_SCHEMA = "dataset_v5_named_view_policy_v1.0.0"
APPROVED_DECISIONS_SCHEMA = "dataset_v5_approved_decisions_v1.0.0"
FREEZE_SCHEMA = "dataset_v5_view_freeze_v1.0.0"

VIEW_NAMES = (
    "v5_debug",
    "v5_architecture",
    "v5_scale_check",
    "v5_eval_balanced",
    "v5_source_ood",
    "v5_open_set",
    "v5_unlabeled",
)
SUPERVISION_CLASSES = frozenset({"supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled"})
TARGET_STATES = frozenset({"known", "unknown", "missing", "abstained", "oov", "not_applicable"})
NONCONTRIBUTING_STATES = TARGET_STATES - {"known"}
UNCERTAINTY_STATES = frozenset({"not_scorable", "provisional_uncalibrated", "calibrated"})
SPLITS = frozenset({"train", "validation", "test", "source_ood_test", "open_set_test", "unsplit", "not_applicable"})
OOD_SCOPES = frozenset(
    {
        "held_out_pack",
        "held_out_artist",
        "held_out_source_family",
        "held_out_license_source",
        "combined_source_ood",
    }
)
SEMANTIC_FIELDS = (
    "canonical_object",
    "category",
    "domain",
    "role",
    "explicit_material",
    "shape",
    "palette_colors",
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "shadow_colors",
    "highlight_colors",
)
QUALITY_MULTIPLIERS = {"strict": 1.0, "standard": 0.8, "unreviewed": 0.0}
CONTRACT_FILES = (
    "dataset_v5_view_manifest.schema.json",
    "dataset_v5_record.schema.json",
    "dataset_v5_supervision.schema.json",
    "view_contracts.json",
    "freeze_contract.json",
    "leakage_contract.json",
)
APPROVED_CONTRACT_SHA256 = {
    "dataset_v5_record.schema.json": "750ee76d04aa509b96bfbb7bae92d37d33f27fc91461e23762cb1d316422dab4",
    "dataset_v5_supervision.schema.json": "1767fcb48d7cf0d16bcf0ec1782edb7916afcfb8bf3b823ece93caa5cda23d97",
    "dataset_v5_view_manifest.schema.json": "40a10cc50ceab753c67906b3732c20a886f168351665899546e50fa17ab5ffc0",
    "freeze_contract.json": "7eca2c7b10a416960134f1ca7f39cce871f32f0f69d28109b32c1e0cbca693a9",
    "leakage_contract.json": "7c32e173798c3150796e6c91309727477eab0b8e792b0c1065fdf7a1fba64595",
    "view_contracts.json": "72b31d0b51ea0e77af1a3163109186403723f7029320733d45b877621d15a597",
}
APPROVED_R2_CANDIDATE_PATH = "datasets/sprite_lab_unlabeled_pool_v1_r2/candidate_manifest.jsonl"
APPROVED_R2_CANDIDATE_SHA256 = "2f1316dbfb5ace0df1569894a83f15b9cd6b77383e5aaba98eda0b408a1f41cc"
APPROVED_R2_FREEZE_PATH = "datasets/sprite_lab_unlabeled_pool_v1_r2/freeze_manifest.json"
APPROVED_R2_FREEZE_SHA256 = "7209fd4b1fff96f7b5cf4cefd2dc09dbf945301973454a3db97b52de0bccb5ca"
VIEW_ARTIFACTS = (
    "record_manifest.jsonl",
    "excluded_record_manifest.jsonl",
    "split_manifest.json",
    "weight_manifest.jsonl",
    "evaluation_manifest.jsonl",
    "license_provenance.jsonl",
    "relation_manifest.json",
    "resolved_policy.json",
    "source_binding.json",
    "validation_report.json",
    "view_manifest.json",
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
REPO_ROOT = Path(__file__).resolve().parents[3]


class DatasetV5ViewError(ValueError):
    """Fail-closed error carrying the command-contract exit code."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int = 2,
        reason_code: str = "validation_failed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": str(self),
            "exit_code": self.exit_code,
            "reason_code": self.reason_code,
            "details": self.details,
        }


def _error(
    message: str,
    exit_code: int,
    reason_code: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> DatasetV5ViewError:
    return DatasetV5ViewError(
        message,
        exit_code=exit_code,
        reason_code=reason_code,
        details=details,
    )


def _strict_json(value: Any, *, pretty: bool) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetV5ViewError(f"value is not canonical JSON: {exc}") from None


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_strict_json(value, pretty=False).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetV5ViewError(f"cannot read JSON artifact {path}: {exc}") from None
    if not isinstance(value, dict):
        raise DatasetV5ViewError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetV5ViewError(f"cannot read JSONL artifact {path}: {exc}") from None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetV5ViewError(f"malformed JSONL at {path}:{line_number}: {exc}") from None
        if not isinstance(value, dict):
            raise DatasetV5ViewError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_strict_json(value, pretty=True) + "\n", encoding="utf-8", newline="\n")


def _record_sort_key(row: Mapping[str, Any]) -> bytes:
    return str(row.get("record_id") or row.get("source_record_id") or "").encode("utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((dict(row) for row in rows), key=_record_sort_key)
    text = "".join(_strict_json(row, pretty=False) + "\n" for row in ordered)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_report(path: str | Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise _error(f"report output already exists: {target}", 20, "existing_output_root")
    _write_json(target, dict(result))


def _resolve_file(path: str | Path, *, reason: str = "missing_source_artifact") -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = (Path.cwd() / value).resolve()
    else:
        value = value.resolve()
    if not value.is_file():
        raise _error(f"missing source artifact: {value}", 21, reason)
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raise _error(
            f"freeze-bound paths must be repository-relative: {path}",
            23,
            "unsupported_external_path",
        ) from None


def _resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / Path(value)).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        raise DatasetV5ViewError(f"path escapes repository root: {value}") from None
    return path


def _git_identity(policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    configured = (policy or {}).get("code_identity")
    if configured is not None and not isinstance(configured, Mapping):
        raise _error("invalid configured code_identity", 23, "unsupported_schema")
    if isinstance(configured, Mapping):
        if (policy or {}).get("synthetic_fixture") is not True:
            raise _error(
                "non-synthetic builds cannot configure or spoof code identity",
                23,
                "unsupported_schema",
            )
        commit = str(configured.get("git_commit") or "")
        dirty = configured.get("dirty")
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or not isinstance(dirty, bool):
            raise _error("invalid configured code_identity", 23, "unsupported_schema")
        return {"git_commit": commit, "dirty": dirty}
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    dirty_result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return {"git_commit": commit_result.stdout.strip(), "dirty": bool(dirty_result.stdout.strip())}


def _contract_paths(contract_root: str | Path) -> dict[str, Path]:
    root = Path(contract_root).resolve()
    if not root.is_dir():
        raise _error(f"contract root is missing: {root}", 21, "missing_source_artifact")
    result = {name: root / name for name in CONTRACT_FILES}
    missing = [name for name, path in result.items() if not path.is_file()]
    if missing:
        raise _error(
            f"contract root is incomplete: {', '.join(missing)}",
            21,
            "missing_source_artifact",
            details={"missing": missing},
        )
    return result


def validate_contract(
    contract_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = _contract_paths(contract_root)
    actual_contract_hashes = {name: _file_hash(path) for name, path in sorted(paths.items())}
    if actual_contract_hashes != APPROVED_CONTRACT_SHA256:
        raise _error(
            "Dataset-v5 contract artifacts differ from the approved six-file contract",
            23,
            "unsupported_schema",
            details={
                "expected": APPROVED_CONTRACT_SHA256,
                "actual": actual_contract_hashes,
            },
        )
    loaded = {name: _read_json(path) for name, path in paths.items()}
    errors: list[str] = []
    manifest_schema = loaded["dataset_v5_view_manifest.schema.json"]
    record_schema = loaded["dataset_v5_record.schema.json"]
    supervision_schema = loaded["dataset_v5_supervision.schema.json"]
    view_contracts = loaded["view_contracts.json"]
    freeze_contract = loaded["freeze_contract.json"]
    leakage_contract = loaded["leakage_contract.json"]
    expected_ids = {
        "dataset_v5_view_manifest.schema.json": "dataset_v5_view_manifest_v1.0.0",
        "dataset_v5_record.schema.json": "dataset_v5_record_v1.0.0",
        "dataset_v5_supervision.schema.json": "dataset_v5_supervision_v1.0.0",
    }
    for name, suffix in expected_ids.items():
        if suffix not in str(loaded[name].get("$id") or ""):
            errors.append(f"{name}: unsupported $id")
        if loaded[name].get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"{name}: unsupported JSON Schema dialect")
    if manifest_schema.get("properties", {}).get("contract_version", {}).get("const") != CONTRACT_VERSION:
        errors.append("view manifest contract version mismatch")
    if set(manifest_schema.get("properties", {}).get("view_name", {}).get("enum", [])) != set(VIEW_NAMES):
        errors.append("view manifest does not define exactly seven supported views")
    if set(record_schema.get("properties", {}).get("supervision_class", {}).get("enum", [])) != set(
        SUPERVISION_CLASSES
    ):
        errors.append("record schema supervision classes mismatch")
    if set(supervision_schema.get("properties", {}).get("supervision_class", {}).get("enum", [])) != set(
        SUPERVISION_CLASSES
    ):
        errors.append("supervision schema class vocabulary mismatch")
    for value, name in (
        (view_contracts.get("contract_version"), "view_contracts"),
        (freeze_contract.get("contract_version"), "freeze_contract"),
        (leakage_contract.get("contract_version"), "leakage_contract"),
    ):
        if value != CONTRACT_VERSION:
            errors.append(f"{name}: contract version mismatch")
    if set((view_contracts.get("views") or {}).keys()) != set(VIEW_NAMES):
        errors.append("view_contracts does not contain exactly seven views")
    expected_commands = {"validate-contract", "build-view", "verify-view", "freeze-view", "verify-freeze"}
    actual_commands = {
        str(name).removeprefix("dataset-v5 ")
        for name in (view_contracts.get("command_interface", {}).get("commands") or {})
    }
    if actual_commands != expected_commands:
        errors.append("command interface does not contain exactly the five required commands")
    if errors:
        raise _error(
            "Dataset-v5 contract validation failed",
            23,
            "unsupported_schema",
            details={"errors": errors},
        )
    result = {
        "ok": True,
        "schema_version": "dataset_v5_contract_validation_report_v1.0.0",
        "contract_version": CONTRACT_VERSION,
        "contract_root": _repo_relative(Path(contract_root).resolve()),
        "artifact_sha256": actual_contract_hashes,
        "views": list(VIEW_NAMES),
        "commands": sorted(expected_commands),
        "runtime_semantic_validation_required": True,
        "production_freeze_authorized": False,
    }
    if output_path is not None:
        write_report(output_path, result)
    return result


def _load_policy(path: Path, view_name: str) -> dict[str, Any]:
    policy = _read_json(path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise _error("unsupported named-view policy schema", 23, "unsupported_schema")
    if policy.get("view_name") != view_name or view_name not in VIEW_NAMES:
        raise _error("policy/view identity mismatch", 23, "unsupported_schema")
    if policy.get("synthetic_fixture") is not True and "code_identity" in policy:
        raise _error(
            "non-synthetic policies cannot configure or spoof code identity",
            23,
            "unsupported_schema",
        )
    approvals = policy.get("approvals")
    if not isinstance(approvals, Mapping) or not all(
        isinstance(key, str) and isinstance(value, bool) for key, value in approvals.items()
    ):
        raise _error("named-view policy requires a boolean approvals map", 23, "unsupported_schema")
    multipliers = policy.get("quality_multipliers")
    if multipliers != QUALITY_MULTIPLIERS:
        raise _error(
            "quality_multipliers must explicitly be strict=1.0, standard=0.8, unreviewed=0.0",
            23,
            "unsupported_schema",
        )
    status = str(policy.get("view_status") or "diagnostic")
    if status not in {"preview", "diagnostic", "candidate"}:
        raise _error("build-view cannot request a frozen or unsupported status", 23, "unsupported_schema")
    target_size = policy.get("target_size")
    if target_size is not None and (not isinstance(target_size, int) or target_size < 1):
        raise _error("target_size must be a positive integer or null", 23, "unsupported_schema")
    raking = policy.get("candidate_raking_factor", 1.0)
    if not isinstance(raking, (int, float)) or not math.isfinite(float(raking)) or float(raking) <= 0:
        raise _error("candidate_raking_factor must be finite and positive", 23, "unsupported_schema")
    if float(raking) != 1.0 and policy.get("candidate_raking_only") is not True:
        raise _error("non-unit raking is allowed only as an explicit candidate diagnostic", 23, "unsupported_schema")
    if float(raking) != 1.0 and (status != "candidate" or approvals.get("raking_policy") is not True):
        raise _error(
            "non-unit raking requires candidate status and an explicit diagnostic approval",
            23,
            "unsupported_schema",
        )
    weak_cap = policy.get("weak_weight_cap", 0.8)
    if not isinstance(weak_cap, (int, float)) or not 0 < float(weak_cap) < 1:
        raise _error("weak_weight_cap must be in (0,1)", 23, "unsupported_schema")
    return policy


def _logical_source_path(path: Path, digest: str) -> str:
    """Return a stable, POSIX source identity without leaking fixture roots."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external-fixture/{digest[:16]}/{path.name}"


def _stored_locator(path: Path, *, synthetic_fixture: bool) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        if synthetic_fixture:
            return path.resolve().as_posix()
        raise _error(
            f"non-synthetic freeze locator is outside the repository: {path}",
            23,
            "unsupported_external_path",
        ) from None


def _resolve_locator(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        return Path("<missing>")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _error(f"record requires nonempty {key}", 23, "unsupported_schema")
    return value


def _require_hash(row: Mapping[str, Any], key: str) -> str:
    value = _require_string(row, key)
    if not HASH_RE.fullmatch(value):
        raise _error(f"record has invalid {key}", 23, "unsupported_schema")
    return value


def _semantic_maps(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    names = (
        "targets",
        "field_masks",
        "field_weights",
        "field_uncertainty",
        "field_calibration_identity",
    )
    maps: list[dict[str, Any]] = []
    for name in names:
        value = row.get(name)
        if not isinstance(value, Mapping):
            raise _error(f"record requires supervision object {name}", 23, "unsupported_schema")
        if set(value) != set(SEMANTIC_FIELDS):
            raise _error(
                f"{name} must contain exactly the canonical semantic fields",
                23,
                "unsupported_schema",
                details={
                    "missing": sorted(set(SEMANTIC_FIELDS) - set(value)),
                    "extra": sorted(set(value) - set(SEMANTIC_FIELDS)),
                },
            )
        maps.append(dict(value))
    return tuple(maps)


def _validate_supervision(row: Mapping[str, Any], *, weak_cap: float) -> None:
    supervision_class = row.get("supervision_class")
    if supervision_class not in SUPERVISION_CLASSES:
        raise _error("invalid supervision class", 23, "unsupported_schema")
    targets, masks, weights, uncertainties, calibrations = _semantic_maps(row)
    if supervision_class in {"auxiliary_only", "unlabeled"} and (
        any(value != 0 for value in masks.values()) or any(value != 0 for value in weights.values())
    ):
        raise _error(
            f"{supervision_class} records require zero semantic masks and weights",
            23,
            "unsupported_schema",
        )
    for field in SEMANTIC_FIELDS:
        target = targets[field]
        if not isinstance(target, Mapping) or set(target) != {"state", "value"}:
            raise _error(f"invalid target object for {field}", 23, "unsupported_schema")
        state = target.get("state")
        value = target.get("value")
        if state not in TARGET_STATES:
            raise _error(f"invalid target state for {field}", 23, "unsupported_schema")
        if state == "known" and value is None:
            raise _error(f"known target {field} requires a value", 23, "unsupported_schema")
        if state in NONCONTRIBUTING_STATES and value is not None:
            raise _error(
                f"non-contributing target {field} must not encode an implied value",
                23,
                "unsupported_schema",
            )
        mask = masks[field]
        weight = weights[field]
        uncertainty = uncertainties[field]
        calibration = calibrations[field]
        if mask not in (0, 1):
            raise _error(f"invalid mask for {field}", 23, "unsupported_schema")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise _error(f"invalid weight for {field}", 23, "unsupported_schema")
        if not math.isfinite(float(weight)) or not 0 <= float(weight) <= 1:
            raise _error(f"invalid weight for {field}", 23, "unsupported_schema")
        if not isinstance(uncertainty, Mapping) or set(uncertainty) != {"state", "score_1_20"}:
            raise _error(f"invalid uncertainty for {field}", 23, "unsupported_schema")
        uncertainty_state = uncertainty.get("state")
        uncertainty_score = uncertainty.get("score_1_20")
        if uncertainty_state not in UNCERTAINTY_STATES:
            raise _error(f"invalid uncertainty state for {field}", 23, "unsupported_schema")
        if uncertainty_score is not None and (
            isinstance(uncertainty_score, bool)
            or not isinstance(uncertainty_score, int)
            or not 1 <= uncertainty_score <= 20
        ):
            raise _error(f"invalid uncertainty score for {field}", 23, "unsupported_schema")
        if calibration is not None and (not isinstance(calibration, str) or not calibration):
            raise _error(f"invalid calibration identity for {field}", 23, "unsupported_schema")
        if mask == 1:
            if state != "known" or float(weight) <= 0:
                raise _error(
                    f"active field {field} must have a known value and positive weight",
                    23,
                    "unsupported_schema",
                )
        elif float(weight) != 0:
            raise _error(f"masked field {field} must have zero weight", 23, "unsupported_schema")
        if state in NONCONTRIBUTING_STATES and (mask != 0 or float(weight) != 0):
            raise _error(
                f"{state} target {field} cannot contribute as a negative",
                23,
                "unsupported_schema",
            )
        if supervision_class == "supervised_strong" and mask == 1:
            if uncertainty_state != "calibrated" or not calibration:
                raise _error(
                    f"strong supervision field {field} requires calibrated evidence and identity",
                    23,
                    "unsupported_schema",
                )
        if supervision_class == "supervised_weak" and mask == 1 and float(weight) > weak_cap:
            raise _error(
                f"weak supervision field {field} exceeds policy cap {weak_cap}",
                23,
                "unsupported_schema",
            )


def _validate_license_provenance(row: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    if row.get("provenance_status") != "verified":
        raise _error("record provenance is not verified", 26, "provenance_failure")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _error("record provenance details are missing", 26, "provenance_failure")
    if provenance.get("license_confirmed") is not True:
        raise _error("record license is not confirmed", 27, "license_failure")
    for key in ("semantic_origin", "source_url", "attribution"):
        if not isinstance(provenance.get(key), str) or not provenance.get(key):
            raise _error(f"record provenance requires {key}", 26, "provenance_failure")
    if row.get("supervision_class") == "supervised_strong" and provenance.get("semantic_origin") not in {
        "human_truth",
        "deterministic_ground_truth",
    }:
        raise _error(
            "supervised_strong requires human or approved deterministic calibrated truth",
            23,
            "unsupported_schema",
        )
    if row.get("supervision_class") == "supervised_weak" and provenance.get("semantic_origin") in {
        "model_proposal",
        "provider_inference",
        "model_derived",
    }:
        masks = row.get("field_masks")
        uncertainties = row.get("field_uncertainty")
        calibrations = row.get("field_calibration_identity")
        if not all(isinstance(value, Mapping) for value in (masks, uncertainties, calibrations)):
            raise _error("model-derived weak supervision is malformed", 23, "unsupported_schema")
        for field in SEMANTIC_FIELDS:
            if masks.get(field) == 1 and (
                not isinstance(uncertainties.get(field), Mapping)
                or uncertainties[field].get("state") != "calibrated"
                or not calibrations.get(field)
            ):
                raise _error(
                    "uncalibrated model-derived evidence cannot enter supervised_weak",
                    23,
                    "unsupported_schema",
                )
    license_name = row.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        raise _error("record license is missing", 27, "license_failure")
    if license_name.strip().lower() in {"unknown", "unlicensed", "pending", "none"}:
        raise _error("record license is not approved", 27, "license_failure")
    allowed = policy.get("allowed_licenses")
    if allowed is not None and (
        not isinstance(allowed, list)
        or not all(isinstance(value, str) and value for value in allowed)
        or license_name not in allowed
    ):
        raise _error("record license is outside the policy allowlist", 27, "license_failure")
    for key in ("creator_lineage", "distribution_platform"):
        if not isinstance(row.get(key), str) or not row.get(key):
            raise _error(f"record provenance requires {key}", 26, "provenance_failure")


def _adapt_source_row(
    source: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_identity: str,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    if source.get("schema_version") != SOURCE_RECORD_SCHEMA:
        raise _error("unsupported source record schema", 23, "unsupported_schema")
    _validate_license_provenance(source, policy)
    _validate_supervision(source, weak_cap=float(policy.get("weak_weight_cap", 0.8)))
    record_id = _require_string(source, "record_id")
    blob_value = _require_string(source, "blob_path")
    blob_source = Path(blob_value)
    if policy.get("synthetic_fixture") is not True and (
        blob_source.is_absolute() or "\\" in blob_value or ".." in PurePosixPath(blob_value).parts
    ):
        raise _error(
            "non-synthetic source blob paths must be confined repository-relative POSIX paths",
            23,
            "unsupported_external_path",
        )
    if not blob_source.is_absolute():
        blob_source = (manifest_path.parent / blob_source).resolve()
    else:
        blob_source = blob_source.resolve()
    if policy.get("synthetic_fixture") is not True:
        try:
            blob_source.relative_to(REPO_ROOT)
        except ValueError:
            raise _error(
                "non-synthetic source blob escapes the repository",
                23,
                "unsupported_external_path",
            ) from None
    if not blob_source.is_file():
        raise _error(f"missing source blob for {record_id}: {blob_source}", 21, "missing_source_artifact")
    actual_blob_hash = _file_hash(blob_source)
    declared_blob_hash = _require_hash(source, "blob_sha256")
    declared_rgba_hash = _require_hash(source, "exported_rgba_hash")
    if actual_blob_hash != declared_blob_hash:
        raise _error(
            f"source blob hash changed for {record_id}",
            22,
            "changed_source_hash",
            details={
                "actual": actual_blob_hash,
                "blob_sha256": declared_blob_hash,
            },
        )
    width = source.get("exported_width")
    height = source.get("exported_height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height < 1
    ):
        raise _error("exported dimensions must be positive integers", 23, "unsupported_schema")
    if blob_source.stat().st_size != width * height * 4:
        raise _error("RGBA blob byte length does not match exported dimensions", 22, "changed_source_hash")
    rgba = np.frombuffer(blob_source.read_bytes(), dtype=np.uint8).reshape(height, width, 4)
    actual_rgba_hash = canonical_rgba_sha256(rgba)
    if actual_rgba_hash != declared_rgba_hash:
        raise _error(
            f"canonical RGBA hash changed for {record_id}",
            22,
            "changed_source_hash",
            details={"actual": actual_rgba_hash, "exported_rgba_hash": declared_rgba_hash},
        )
    actual_alpha_hash = alpha_mask_sha256(rgba[..., 3])
    declared_alpha_hash = source.get("alpha_mask_hash")
    if declared_alpha_hash is not None and declared_alpha_hash != actual_alpha_hash:
        raise _error(
            f"alpha-mask hash changed for {record_id}",
            22,
            "changed_source_hash",
            details={"actual": actual_alpha_hash, "declared": declared_alpha_hash},
        )
    actual_normalized_alpha_hash = _normalized_alpha_key(rgba[..., 3])
    source_hard_ids = source.get("hard_relation_group_ids")
    if (
        not isinstance(source_hard_ids, list)
        or not all(isinstance(value, str) and value for value in source_hard_ids)
        or source_hard_ids != sorted(set(source_hard_ids))
    ):
        raise _error(
            "hard_relation_group_ids must be a sorted-unique list of nonempty strings",
            23,
            "unsupported_schema",
        )
    hard_ids = list(source_hard_ids)
    reason_codes = source.get("suitability_reason_codes")
    if (
        not isinstance(reason_codes, list)
        or not all(isinstance(value, str) and value for value in reason_codes)
        or reason_codes != sorted(set(reason_codes))
    ):
        raise _error(
            "suitability_reason_codes must be a sorted-unique list of nonempty strings",
            23,
            "unsupported_schema",
        )
    relation_coverage = {"translation_known": False, "flip_known": False}
    for source_key, prefix, coverage_key in (
        ("known_translation_group_id", "translation", "translation_known"),
        ("known_flip_group_id", "flip", "flip_known"),
    ):
        relation = source.get(source_key)
        if relation is not None:
            if not isinstance(relation, str) or not relation:
                raise _error(f"{source_key} must be nonempty or null", 23, "unsupported_schema")
            hard_ids.append(f"{prefix}:{relation}")
            relation_coverage[coverage_key] = True
    targets = copy.deepcopy(dict(source["targets"]))
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "record_id": record_id,
        "sprite_id": _require_string(source, "sprite_id"),
        "blob_path": f"blobs/{actual_rgba_hash}.rgba",
        "blob_sha256": actual_blob_hash,
        "exported_rgba_hash": actual_rgba_hash,
        "exported_width": width,
        "exported_height": height,
        "source_artifact": manifest_identity,
        "source_record_id": _require_string(source, "source_record_id"),
        "source_pack": _require_string(source, "source_pack"),
        "sub_artist": _require_string(source, "sub_artist"),
        "source_family": _require_string(source, "source_family"),
        "license": _require_string(source, "license"),
        "provenance_status": "verified",
        "geometry_family_id": _require_string(source, "geometry_family_id"),
        "recolor_family_id": source.get("recolor_family_id"),
        "declared_variant_group_id": source.get("declared_variant_group_id"),
        "sheet_group_id": source.get("sheet_group_id"),
        "hard_relation_group_ids": sorted(set(hard_ids)),
        "suitability_status": source.get("suitability_status"),
        "suitability_reason_codes": list(reason_codes),
        "quality_eligibility": source.get("quality_eligibility"),
        "membership": "included",
        "split": "unsplit",
        "sampling_weight": 0.0,
        "evaluation_weight": 0.0,
        "view_inclusion_reason": "eligible source record",
        "view_exclusion_reason": None,
        "supervision_class": source.get("supervision_class"),
        "targets": targets,
        "field_masks": copy.deepcopy(dict(source["field_masks"])),
        "field_weights": copy.deepcopy(dict(source["field_weights"])),
        "field_uncertainty": copy.deepcopy(dict(source["field_uncertainty"])),
        "field_calibration_identity": copy.deepcopy(dict(source["field_calibration_identity"])),
        "source_ood": source.get("source_ood"),
        "source_ood_scope": source.get("source_ood_scope"),
        "source_ood_rationale": source.get("source_ood_rationale"),
        "open_set": source.get("open_set"),
        "open_set_rationale": source.get("open_set_rationale"),
        "evaluation_stratum": source.get("evaluation_stratum"),
    }
    for field in SEMANTIC_FIELDS:
        target = targets[field]
        record[field] = copy.deepcopy(target["value"] if target["state"] == "known" else None)
    record["alpha_mask_hash"] = actual_alpha_hash
    record["normalized_alpha_hash"] = actual_normalized_alpha_hash
    for nullable in ("recolor_family_id", "declared_variant_group_id", "sheet_group_id"):
        if record[nullable] is not None and (not isinstance(record[nullable], str) or not record[nullable]):
            raise _error(f"{nullable} must be nonempty or null", 23, "unsupported_schema")
    if record["suitability_status"] not in {"accept", "quarantine", "reject", "approved_existing_dataset"}:
        raise _error("invalid suitability status", 23, "unsupported_schema")
    if record["quality_eligibility"] not in {"eligible", "diagnostic_only", "ineligible"}:
        raise _error("invalid quality eligibility", 23, "unsupported_schema")
    if not isinstance(record["source_ood"], bool) or not isinstance(record["open_set"], bool):
        raise _error("OOD and open-set flags must be booleans", 23, "unsupported_schema")
    if record["source_ood"]:
        if record["source_ood_scope"] not in OOD_SCOPES or not record["source_ood_rationale"]:
            raise _error("source-OOD rows require scope and rationale", 23, "unsupported_schema")
    elif record["source_ood_scope"] is not None or record["source_ood_rationale"] is not None:
        raise _error("regular rows cannot carry source-OOD scope or rationale", 23, "unsupported_schema")
    if record["open_set"] and not record["open_set_rationale"]:
        raise _error("open-set rows require rationale", 23, "unsupported_schema")
    if not record["open_set"] and record["open_set_rationale"] is not None:
        raise _error("regular rows cannot carry open-set rationale", 23, "unsupported_schema")
    requested_split = source.get("requested_split", source.get("split", "train"))
    if requested_split not in SPLITS:
        raise _error("invalid requested split", 23, "unsupported_schema")
    review_quality = source.get("review_quality", "unreviewed")
    if review_quality not in QUALITY_MULTIPLIERS:
        raise _error("invalid review_quality", 23, "unsupported_schema")
    base_sampling = source.get("base_sampling_weight", 1.0)
    base_evaluation = source.get("base_evaluation_weight", 1.0)
    for value, name in ((base_sampling, "base_sampling_weight"), (base_evaluation, "base_evaluation_weight")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise _error(f"invalid {name}", 23, "unsupported_schema")
    metadata = {
        "requested_split": requested_split,
        "review_quality": review_quality,
        "base_sampling_weight": float(base_sampling),
        "base_evaluation_weight": float(base_evaluation),
        "creator_lineage": source["creator_lineage"],
        "distribution_platform": source["distribution_platform"],
        "provenance": copy.deepcopy(dict(source["provenance"])),
        "variant_inclusion_reason": source.get("variant_inclusion_reason"),
        "evaluation_candidate": source.get("evaluation_candidate", False),
        **relation_coverage,
    }
    if not isinstance(metadata["evaluation_candidate"], bool):
        raise _error("evaluation_candidate must be boolean", 23, "unsupported_schema")
    license_row = {
        "record_id": record_id,
        "source_artifact": manifest_identity,
        "source_record_id": record["source_record_id"],
        "source_pack": record["source_pack"],
        "sub_artist": record["sub_artist"],
        "creator_lineage": metadata["creator_lineage"],
        "distribution_platform": metadata["distribution_platform"],
        "license": record["license"],
        "provenance_status": "verified",
        "provenance": metadata["provenance"],
    }
    return record, metadata, blob_source, license_row


def _expected_manifest_hash(policy: Mapping[str, Any], *, logical_path: str, path: Path) -> str | None:
    configured = policy.get("source_manifest_sha256")
    if configured is None:
        return None
    if isinstance(configured, str):
        return configured
    if not isinstance(configured, Mapping):
        raise _error("source_manifest_sha256 must be a hash or object", 23, "unsupported_schema")
    for key in (logical_path, path.name, path.as_posix(), str(path)):
        value = configured.get(key)
        if value is not None:
            return str(value)
    raise _error(
        f"policy has no source hash binding for {logical_path}",
        21,
        "missing_source_artifact",
    )


def _load_sources(
    source_manifests: Sequence[str | Path],
    policy: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Path],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not source_manifests:
        raise _error("at least one source manifest is required", 21, "missing_source_artifact")
    records: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    blob_sources: dict[str, Path] = {}
    license_rows: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    seen_manifests: set[Path] = set()
    for source_manifest in source_manifests:
        path = _resolve_file(source_manifest)
        if path in seen_manifests:
            raise _error(f"duplicate source manifest: {path}", 23, "unsupported_schema")
        seen_manifests.add(path)
        manifest_hash = _file_hash(path)
        logical_path = _logical_source_path(path, manifest_hash)
        expected_hash = _expected_manifest_hash(policy, logical_path=logical_path, path=path)
        if expected_hash is not None:
            if not HASH_RE.fullmatch(expected_hash):
                raise _error("policy source hash is invalid", 23, "unsupported_schema")
            if manifest_hash != expected_hash:
                raise _error(
                    f"source manifest hash changed: {logical_path}",
                    22,
                    "changed_source_hash",
                    details={"expected": expected_hash, "actual": manifest_hash},
                )
        if not policy.get("synthetic_fixture"):
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                raise _error(
                    "non-synthetic source manifests must be repository-relative",
                    23,
                    "unsupported_external_path",
                ) from None
        rows = _read_jsonl(path)
        manifest_blob_bindings: list[dict[str, str]] = []
        for source in rows:
            record, record_metadata, blob_source, license_row = _adapt_source_row(
                source,
                manifest_path=path,
                manifest_identity=logical_path,
                policy=policy,
            )
            record_id = record["record_id"]
            if record_id in metadata:
                raise _error(f"duplicate record identity: {record_id}", 23, "unsupported_schema")
            records.append(record)
            metadata[record_id] = record_metadata
            blob_sources[record_id] = blob_source
            license_rows.append(license_row)
            manifest_blob_bindings.append(
                {
                    "record_id": record_id,
                    "logical_path": str(source["blob_path"]).replace("\\", "/"),
                    "locator": _stored_locator(blob_source, synthetic_fixture=policy.get("synthetic_fixture") is True),
                    "sha256": record["blob_sha256"],
                    "exported_rgba_hash": record["exported_rgba_hash"],
                }
            )
        source_bindings.append(
            {
                "logical_path": logical_path,
                "locator": _stored_locator(path, synthetic_fixture=policy.get("synthetic_fixture") is True),
                "sha256": manifest_hash,
                "record_count": len(rows),
                "blobs": sorted(manifest_blob_bindings, key=lambda item: item["record_id"]),
            }
        )
    if not records:
        raise _error("source manifests contain no records", 24, "partial_view")
    source_binding = {
        "schema_version": "dataset_v5_source_binding_v1.0.0",
        "synthetic_fixture": policy.get("synthetic_fixture") is True,
        "sources": sorted(source_bindings, key=lambda item: item["logical_path"]),
    }
    return records, metadata, blob_sources, license_rows, source_binding


def _resolve_binding_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(f"frozen-r2 binding requires {name}", 21, "missing_source_artifact")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise _error(f"frozen-r2 source is missing: {name}", 21, "missing_source_artifact")
    return path


def _validate_frozen_r2_binding(view_name: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    binding = policy.get("frozen_r2_binding")
    if view_name != "v5_unlabeled":
        if binding is not None:
            raise _error("frozen_r2_binding is valid only for v5_unlabeled", 23, "unsupported_schema")
        return {"required": False, "verified": False}
    if binding is None:
        if policy.get("synthetic_fixture") is True:
            return {
                "required": True,
                "verified": False,
                "diagnostic_warning": "synthetic fixture has no frozen-r2 binding",
            }
        raise _error("v5_unlabeled requires an exact frozen-r2 binding", 21, "missing_source_artifact")
    if not isinstance(binding, Mapping):
        raise _error("frozen_r2_binding must be an object", 23, "unsupported_schema")
    candidate = _resolve_binding_path(binding.get("candidate_manifest"), name="candidate_manifest")
    freeze_path = _resolve_binding_path(binding.get("freeze_manifest"), name="freeze_manifest")
    candidate_expected = binding.get("candidate_manifest_sha256")
    freeze_expected = binding.get("freeze_manifest_sha256")
    if not isinstance(candidate_expected, str) or not HASH_RE.fullmatch(candidate_expected):
        raise _error("invalid frozen-r2 candidate manifest hash", 23, "unsupported_schema")
    if not isinstance(freeze_expected, str) or not HASH_RE.fullmatch(freeze_expected):
        raise _error("invalid frozen-r2 freeze manifest hash", 23, "unsupported_schema")
    candidate_actual = _file_hash(candidate)
    freeze_actual = _file_hash(freeze_path)
    if candidate_actual != candidate_expected or freeze_actual != freeze_expected:
        raise _error(
            "frozen-r2 binding hash mismatch",
            22,
            "changed_source_hash",
            details={
                "candidate_expected": candidate_expected,
                "candidate_actual": candidate_actual,
                "freeze_expected": freeze_expected,
                "freeze_actual": freeze_actual,
            },
        )
    if policy.get("synthetic_fixture") is not True:
        if (
            _repo_relative(candidate) != APPROVED_R2_CANDIDATE_PATH
            or candidate_actual != APPROVED_R2_CANDIDATE_SHA256
            or _repo_relative(freeze_path) != APPROVED_R2_FREEZE_PATH
            or freeze_actual != APPROVED_R2_FREEZE_SHA256
        ):
            raise _error(
                "non-synthetic v5_unlabeled must bind the approved immutable r2 artifacts",
                22,
                "changed_source_hash",
            )
    freeze = _read_json(freeze_path)
    artifact_hashes = freeze.get("artifact_hashes")
    blob_hashes = freeze.get("blob_hashes")
    if not isinstance(artifact_hashes, Mapping) or not isinstance(blob_hashes, Mapping):
        raise _error("frozen-r2 freeze manifest is partial", 24, "partial_view")
    frozen_candidate_hash = artifact_hashes.get(candidate.name)
    if frozen_candidate_hash != candidate_actual:
        raise _error("frozen-r2 candidate is not bound by its freeze", 22, "changed_source_hash")
    candidate_rows = _read_jsonl(candidate)
    expected_count = binding.get("expected_record_count")
    expected_geometry_count = binding.get("expected_geometry_family_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise _error("invalid frozen-r2 expected record count", 23, "unsupported_schema")
    if (
        isinstance(expected_geometry_count, bool)
        or not isinstance(expected_geometry_count, int)
        or expected_geometry_count < 1
    ):
        raise _error("invalid frozen-r2 expected geometry count", 23, "unsupported_schema")
    geometry_ids = {
        str(row.get("geometry_family_id") or row.get("variant_geometry_group") or "") for row in candidate_rows
    }
    geometry_ids.discard("")
    if len(candidate_rows) != expected_count or len(geometry_ids) != expected_geometry_count:
        raise _error(
            "frozen-r2 declared counts do not match its candidate manifest",
            22,
            "changed_source_hash",
            details={
                "expected_record_count": expected_count,
                "actual_record_count": len(candidate_rows),
                "expected_geometry_family_count": expected_geometry_count,
                "actual_geometry_family_count": len(geometry_ids),
            },
        )
    for row in candidate_rows:
        blob_value = row.get("blob_path")
        if not isinstance(blob_value, str) or not blob_value:
            raise _error("frozen-r2 candidate row has no blob path", 24, "partial_view")
        blob_path = (candidate.parent / blob_value).resolve()
        if not blob_path.is_file():
            raise _error("frozen-r2 blob is missing", 21, "missing_source_artifact")
        expected_blob_hash = blob_hashes.get(blob_value)
        if expected_blob_hash != _file_hash(blob_path):
            raise _error("frozen-r2 blob hash mismatch", 22, "changed_source_hash")
    if policy.get("synthetic_fixture") is not True and (expected_count != 3233 or expected_geometry_count != 1288):
        raise _error("production r2 binding must declare 3,233 records and 1,288 geometries", 22, "changed_source_hash")
    return {
        "required": True,
        "verified": True,
        "candidate_manifest": _logical_source_path(candidate, candidate_actual),
        "candidate_manifest_sha256": candidate_actual,
        "freeze_manifest": _logical_source_path(freeze_path, freeze_actual),
        "freeze_manifest_sha256": freeze_actual,
        "record_count": len(candidate_rows),
        "geometry_family_count": len(geometry_ids),
    }


def _validate_r2_adapter(
    view_name: str,
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    binding_result: dict[str, Any],
) -> None:
    if view_name != "v5_unlabeled" or not binding_result.get("verified"):
        return
    binding = policy.get("frozen_r2_binding")
    if not isinstance(binding, Mapping):
        raise _error("verified frozen-r2 binding is unavailable", 24, "partial_view")
    candidate_path = _resolve_binding_path(binding.get("candidate_manifest"), name="candidate_manifest")
    candidates = _read_jsonl(candidate_path)
    if len(candidates) != len(records):
        raise _error(
            "named r2 adapter is not one-to-one with the frozen candidate manifest",
            22,
            "changed_source_hash",
            details={"candidate_count": len(candidates), "adapter_count": len(records)},
        )
    candidate_by_identity: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        identities = {
            str(candidate.get(key)) for key in ("record_id", "source_record_id", "sprite_id") if candidate.get(key)
        }
        for identity in identities:
            previous = candidate_by_identity.setdefault(identity, candidate)
            if previous is not candidate:
                raise _error("frozen-r2 candidate identities are ambiguous", 22, "changed_source_hash")
    matched_candidates: set[int] = set()
    for record in records:
        candidate: dict[str, Any] | None = None
        for key in ("record_id", "source_record_id", "sprite_id"):
            value = record.get(key)
            if value is not None and str(value) in candidate_by_identity:
                candidate = candidate_by_identity[str(value)]
                break
        if candidate is None:
            raise _error(
                f"named r2 adapter record has no frozen candidate identity: {record.get('record_id')}",
                22,
                "changed_source_hash",
            )
        candidate_marker = id(candidate)
        if candidate_marker in matched_candidates:
            raise _error("multiple adapter records bind one frozen-r2 candidate", 22, "changed_source_hash")
        matched_candidates.add(candidate_marker)
        expected_geometry = candidate.get("geometry_family_id") or candidate.get("variant_geometry_group")
        comparisons = {
            "sprite_id": candidate.get("sprite_id"),
            "exported_rgba_hash": candidate.get("exported_rgba_hash"),
            "exported_width": candidate.get("exported_width"),
            "exported_height": candidate.get("exported_height"),
            "geometry_family_id": expected_geometry,
        }
        for key, expected in comparisons.items():
            if expected is not None and record.get(key) != expected:
                raise _error(
                    f"named r2 adapter disagrees with frozen candidate field {key}",
                    22,
                    "changed_source_hash",
                    details={"record_id": record.get("record_id")},
                )
    binding_result["adapter_verified"] = True
    binding_result["adapter_record_count"] = len(records)


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _relation_values(record: Mapping[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {
        "exact_decoded_rgba": [str(record["exported_rgba_hash"])],
        "geometry_families": [str(record["geometry_family_id"])],
    }
    optional = {
        "alpha_masks": ("alpha_mask_hash", "normalized_alpha_hash"),
        "recolors": ("recolor_family_id",),
        "declared_variants": ("declared_variant_group_id",),
        "sheet_groups": ("sheet_group_id",),
    }
    for kind, fields in optional.items():
        values[kind] = [str(record[field]) for field in fields if record.get(field)]
    hard_ids = [str(value) for value in record.get("hard_relation_group_ids", [])]
    values["hard_relations"] = hard_ids
    values["translations"] = [value for value in hard_ids if value.startswith("translation:")]
    values["flips_where_known"] = [value for value in hard_ids if value.startswith("flip:")]
    return values


def _hard_components(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    ids = [str(record["record_id"]) for record in records]
    disjoint = _DisjointSet(ids)
    first_by_relation: dict[tuple[str, str], str] = {}
    for record in sorted(records, key=lambda item: str(item["record_id"])):
        record_id = str(record["record_id"])
        for kind, relation_values in _relation_values(record).items():
            for value in relation_values:
                key = (kind, value)
                previous = first_by_relation.setdefault(key, record_id)
                disjoint.union(previous, record_id)
    components: dict[str, list[str]] = defaultdict(list)
    for record_id in ids:
        components[disjoint.find(record_id)].append(record_id)
    return {min(members): sorted(members) for members in components.values()}


def _crossing_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    included = [record for record in records if record.get("membership") == "included"]
    crossings: list[dict[str, Any]] = []
    coverage = Counter({"translation_known": 0, "flip_known": 0})
    for record in included:
        hard = record.get("hard_relation_group_ids") or []
        coverage["translation_known"] += int(any(str(value).startswith("translation:") for value in hard))
        coverage["flip_known"] += int(any(str(value).startswith("flip:") for value in hard))
    relation_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in included:
        for kind, values in _relation_values(record).items():
            for value in values:
                relation_groups[(kind, value)].append(record)
    for (kind, value), members in sorted(relation_groups.items()):
        splits = sorted({str(member.get("split")) for member in members})
        if len(splits) > 1:
            crossings.append(
                {
                    "kind": kind,
                    "relation": value,
                    "splits": splits,
                    "record_ids": sorted(str(member["record_id"]) for member in members),
                }
            )
    regular = [record for record in included if record.get("split") in {"train", "validation", "test"}]
    source_ood = [record for record in included if record.get("split") == "source_ood_test"]
    for record in source_ood:
        scope = record.get("source_ood_scope")
        axes: tuple[tuple[str, Any], ...]
        if scope == "held_out_pack":
            axes = (("held_out_packs", record.get("source_pack")),)
        elif scope == "held_out_artist":
            axes = (("held_out_artists", record.get("sub_artist")),)
        elif scope == "held_out_source_family":
            axes = (("held_out_source_families", record.get("source_family")),)
        elif scope == "held_out_license_source":
            axes = (("held_out_license_sources", (record.get("license"), record.get("source_family"))),)
        elif scope == "combined_source_ood":
            axes = (
                ("held_out_packs", record.get("source_pack")),
                ("held_out_artists", record.get("sub_artist")),
                ("held_out_source_families", record.get("source_family")),
                ("held_out_license_sources", (record.get("license"), record.get("source_family"))),
            )
        else:
            axes = ()
        for kind, identity in axes:
            matches = [
                candidate
                for candidate in regular
                if (
                    candidate.get("source_pack")
                    if kind == "held_out_packs"
                    else candidate.get("sub_artist")
                    if kind == "held_out_artists"
                    else candidate.get("source_family")
                    if kind == "held_out_source_families"
                    else (candidate.get("license"), candidate.get("source_family"))
                )
                == identity
            ]
            if matches:
                crossings.append(
                    {
                        "kind": kind,
                        "relation": identity,
                        "splits": ["regular", "source_ood_test"],
                        "record_ids": sorted(
                            [str(record["record_id"])] + [str(candidate["record_id"]) for candidate in matches]
                        ),
                    }
                )
    open_rows = [record for record in included if record.get("split") == "open_set_test"]
    regular_concepts = defaultdict(list)
    for record in regular:
        regular_concepts[_strict_json(record.get("canonical_object"), pretty=False)].append(record)
    for record in open_rows:
        key = _strict_json(record.get("canonical_object"), pretty=False)
        if key in regular_concepts:
            crossings.append(
                {
                    "kind": "open_set_concept",
                    "relation": record.get("canonical_object"),
                    "splits": ["regular", "open_set_test"],
                    "record_ids": sorted(
                        [str(record["record_id"])]
                        + [str(candidate["record_id"]) for candidate in regular_concepts[key]]
                    ),
                }
            )
    return {
        "passed": not crossings,
        "crossing_count": len(crossings),
        "crossings": crossings,
        "known_relation_coverage": dict(sorted(coverage.items())),
        "unknown_translation_count": len(included) - coverage["translation_known"],
        "unknown_flip_count": len(included) - coverage["flip_known"],
    }


def validate_leakage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return all known hard-relation and configured OOD crossings."""

    return _crossing_report(records)


def _preselection_independence_check(
    view_name: str,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    if view_name not in {"v5_eval_balanced", "v5_source_ood", "v5_open_set"}:
        return
    components = _hard_components(records)
    by_id = {str(record["record_id"]): record for record in records}

    def is_evaluation_member(record_id: str) -> bool:
        if view_name == "v5_eval_balanced":
            return bool(metadata[record_id].get("evaluation_candidate"))
        flag = "source_ood" if view_name == "v5_source_ood" else "open_set"
        return bool(by_id[record_id].get(flag))

    crossings = [
        members
        for members in components.values()
        if {is_evaluation_member(record_id) for record_id in members} == {False, True}
    ]
    if crossings:
        raise _error(
            f"{view_name} candidate pool has hard-relation leakage",
            25,
            "leakage_failure",
            details={"crossing_components": crossings},
        )
    if view_name == "v5_eval_balanced":
        return
    if view_name == "v5_source_ood":
        regular = [record for record in records if not record.get("source_ood")]
        for record in (item for item in records if item.get("source_ood")):
            scope = record.get("source_ood_scope")

            def matches(
                other: Mapping[str, Any],
                held: Mapping[str, Any] = record,
                held_scope: Any = scope,
            ) -> bool:
                pack = other.get("source_pack") == held.get("source_pack")
                artist = other.get("sub_artist") == held.get("sub_artist")
                family = other.get("source_family") == held.get("source_family")
                license_source = (other.get("license"), other.get("source_family")) == (
                    held.get("license"),
                    held.get("source_family"),
                )
                return {
                    "held_out_pack": pack,
                    "held_out_artist": artist,
                    "held_out_source_family": family,
                    "held_out_license_source": license_source,
                    "combined_source_ood": any((pack, artist, family, license_source)),
                }.get(str(held_scope), False)

            if any(matches(other) for other in regular):
                raise _error(
                    "source-OOD identity leakage detected before view selection",
                    25,
                    "leakage_failure",
                )
    else:

        def concept_token(value: Any) -> str:
            return "_".join(str(value).casefold().replace("-", " ").split())

        regular_concepts = {
            concept_token(record.get("canonical_object")) for record in records if not record.get("open_set")
        }
        aliases = policy.get("open_set_aliases")
        if aliases is not None and (
            not isinstance(aliases, Mapping)
            or not all(
                isinstance(key, str)
                and key
                and isinstance(values, list)
                and all(isinstance(value, str) and value for value in values)
                for key, values in aliases.items()
            )
        ):
            raise _error("open_set_aliases must map concepts to alias lists", 23, "unsupported_schema")
        open_tokens: set[str] = set()
        for record in records:
            if not record.get("open_set"):
                continue
            concept = concept_token(record.get("canonical_object"))
            open_tokens.add(concept)
            if isinstance(aliases, Mapping):
                covered = False
                for key, values in aliases.items():
                    normalized = {concept_token(key), *(concept_token(value) for value in values)}
                    if concept in normalized:
                        covered = True
                        open_tokens.update(normalized)
                if not covered:
                    raise _error(
                        f"open-set alias/taxonomy policy does not cover concept {concept}",
                        24,
                        "partial_view",
                    )
        if regular_concepts & open_tokens:
            raise _error("open-set concept leakage detected", 25, "leakage_failure")


def _exclude(record: dict[str, Any], reason: str) -> None:
    record["membership"] = "excluded"
    record["split"] = "not_applicable"
    record["sampling_weight"] = 0.0
    record["evaluation_weight"] = 0.0
    record["view_inclusion_reason"] = None
    record["view_exclusion_reason"] = reason


def _selection_rank(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    geometry_position: int,
    view_name: str,
) -> tuple[int, bytes]:
    if geometry_position == 0:
        tier = 0
    elif metadata.get("variant_inclusion_reason") and record.get("explicit_material") is not None:
        tier = 1
    elif record.get("recolor_family_id"):
        tier = 2
    else:
        tier = 3
    if view_name == "v5_debug":
        tier = 0
    digest = hashlib.sha256(str(record["record_id"]).encode("utf-8")).digest()
    return tier, digest


def _stratified_order(
    records: Sequence[dict[str, Any]],
    *,
    stratum: Any,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[_strict_json(stratum(record), pretty=False)].append(record)
    for rows in buckets.values():
        rows.sort(key=lambda record: hashlib.sha256(str(record["record_id"]).encode()).digest())
    ordered: list[dict[str, Any]] = []
    index = 0
    while True:
        added = False
        for key in sorted(buckets):
            if index < len(buckets[key]):
                ordered.append(buckets[key][index])
                added = True
        if not added:
            return ordered
        index += 1


def _select_records(
    view_name: str,
    records: list[dict[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    approvals = policy.get("approvals")
    if not isinstance(approvals, Mapping):
        approvals = {}
    evaluation_views = {"v5_eval_balanced", "v5_source_ood", "v5_open_set"}
    eligible_classes = (
        {"supervised_strong"}
        if view_name in evaluation_views
        else {"unlabeled", "auxiliary_only"}
        if view_name == "v5_unlabeled"
        else set(SUPERVISION_CLASSES)
    )
    for record in records:
        reason: str | None = None
        if record["supervision_class"] not in eligible_classes:
            reason = "supervision class is ineligible for view"
        elif record["suitability_status"] in {"reject"} or record["quality_eligibility"] == "ineligible":
            reason = "record is quality-ineligible"
        elif record["suitability_status"] == "quarantine":
            diagnostic_allowed = (
                approvals.get("quarantine_policy") is True
                and record["quality_eligibility"] == "diagnostic_only"
                and view_name in {"v5_debug", "v5_architecture", "v5_scale_check", "v5_unlabeled"}
            )
            if not diagnostic_allowed:
                reason = "quarantine is not approved for this view"
        elif view_name in evaluation_views and record["suitability_status"] != "accept":
            reason = "evaluation views require accept-only records"
        elif view_name == "v5_eval_balanced" and not metadata[record["record_id"]]["evaluation_candidate"]:
            reason = "record is not an evaluation candidate"
        elif view_name == "v5_source_ood" and (
            not record["source_ood"] or not metadata[record["record_id"]]["evaluation_candidate"]
        ):
            reason = "record is not in the source-OOD evaluation set"
        elif view_name == "v5_open_set" and (
            not record["open_set"] or not metadata[record["record_id"]]["evaluation_candidate"]
        ):
            reason = "record is not in the open-set evaluation set"
        if reason:
            _exclude(record, reason)
    representative_quality = {"strict": 0, "standard": 1, "unreviewed": 2}

    def quality_key(item: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
        return (
            representative_quality[str(metadata[str(item["record_id"])]["review_quality"])],
            0 if item["quality_eligibility"] == "eligible" else 1,
            0 if item["suitability_status"] in {"accept", "approved_existing_dataset"} else 1,
            0 if item["supervision_class"] == "supervised_strong" else 1,
            str(item["record_id"]),
        )

    representatives: dict[str, str] = {}
    for record in sorted(records, key=quality_key):
        if record["membership"] != "included":
            continue
        rgba_hash = str(record["exported_rgba_hash"])
        representative = representatives.setdefault(rgba_hash, str(record["record_id"]))
        if representative != record["record_id"]:
            _exclude(record, f"exact RGBA duplicate of {representative}")
    candidates = [record for record in records if record["membership"] == "included"]
    geometry_members: dict[str, list[str]] = defaultdict(list)
    for record in sorted(candidates, key=quality_key):
        geometry_members[str(record["geometry_family_id"])].append(str(record["record_id"]))
    per_record_index = {
        record_id: index for members in geometry_members.values() for index, record_id in enumerate(members)
    }
    if view_name == "v5_debug":
        ranked = _stratified_order(
            candidates,
            stratum=lambda record: (
                record["supervision_class"],
                record["suitability_status"],
                record["source_pack"],
            ),
        )
    elif view_name == "v5_eval_balanced":
        ranked = _stratified_order(
            candidates,
            stratum=lambda record: record["evaluation_stratum"],
        )
    elif view_name == "v5_source_ood":
        ranked = _stratified_order(
            candidates,
            stratum=lambda record: (record["source_ood_scope"], record["evaluation_stratum"]),
        )
    elif view_name == "v5_open_set":
        ranked = _stratified_order(
            candidates,
            stratum=lambda record: (record["canonical_object"], record["evaluation_stratum"]),
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda record: _selection_rank(
                record,
                metadata[str(record["record_id"])],
                per_record_index[str(record["record_id"])],
                view_name,
            ),
        )
    target_size = policy.get("target_size")
    if target_size is not None:
        selected_ids = {str(record["record_id"]) for record in ranked[: int(target_size)]}
        for record in candidates:
            if str(record["record_id"]) not in selected_ids:
                _exclude(record, "outside deterministic target-size selection")
    if view_name == "v5_scale_check":
        retained_geometry: set[str] = set()
        for record in ranked:
            if record["membership"] != "included":
                continue
            geometry = str(record["geometry_family_id"])
            if geometry in retained_geometry and not metadata[str(record["record_id"])].get("variant_inclusion_reason"):
                _exclude(record, "scale-check variant lacks explicit inclusion rationale")
            retained_geometry.add(geometry)


def _assign_splits_and_weights(
    view_name: str,
    records: list[dict[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    included = [record for record in records if record["membership"] == "included"]
    components = _hard_components(included)
    by_id = {str(record["record_id"]): record for record in included}
    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
        "source_ood_test": 3,
        "open_set_test": 4,
        "unsplit": 5,
        "not_applicable": 6,
    }
    for members in components.values():
        if view_name == "v5_eval_balanced":
            split = "test"
        elif view_name == "v5_source_ood":
            split = "source_ood_test"
        elif view_name == "v5_open_set":
            split = "open_set_test"
        elif view_name == "v5_unlabeled":
            split = "unsplit"
        else:
            requested = {str(metadata[record_id]["requested_split"]) for record_id in members}
            split = min(requested, key=lambda value: (split_order[value], value))
        for record_id in members:
            by_id[record_id]["split"] = split
    evaluation_views = {"v5_eval_balanced", "v5_source_ood", "v5_open_set"}
    raking = float(policy.get("candidate_raking_factor", 1.0))
    recolor_counts = Counter(
        str(record["recolor_family_id"]) for record in included if record.get("recolor_family_id") is not None
    )
    weight_details: dict[str, dict[str, float]] = {}
    for record in included:
        record_id = str(record["record_id"])
        record_metadata = metadata[record_id]
        recolor_family = record.get("recolor_family_id")
        recolor_multiplier = 1.0 / recolor_counts[str(recolor_family)] if recolor_family is not None else 1.0
        quality_multiplier = QUALITY_MULTIPLIERS[str(record_metadata["review_quality"])]
        if view_name in evaluation_views:
            record["sampling_weight"] = 0.0
            record["evaluation_weight"] = record_metadata["base_evaluation_weight"]
        else:
            record["sampling_weight"] = (
                record_metadata["base_sampling_weight"] * quality_multiplier * recolor_multiplier * raking
            )
            record["evaluation_weight"] = 0.0
        weight_details[record_id] = {
            "quality_multiplier": quality_multiplier,
            "recolor_multiplier": recolor_multiplier,
            "candidate_raking_factor": raking,
        }
        record["view_inclusion_reason"] = (
            "synthetic diagnostic selection"
            if policy.get("synthetic_fixture") is True
            else "policy-eligible deterministic selection"
        )
        record["view_exclusion_reason"] = None
    return components, weight_details


def _validate_record(
    record: Mapping[str, Any],
    *,
    record_schema: Mapping[str, Any],
    view_name: str,
    weak_cap: float,
) -> None:
    properties = record_schema.get("properties")
    required = record_schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise _error("record contract schema is malformed", 23, "unsupported_schema")
    missing = sorted(set(required) - set(record))
    extra = sorted(set(record) - set(properties))
    if missing or extra:
        raise _error(
            "record does not match the common record contract",
            23,
            "unsupported_schema",
            details={"record_id": record.get("record_id"), "missing": missing, "extra": extra},
        )
    if record.get("schema_version") != RECORD_SCHEMA:
        raise _error("unsupported record schema", 23, "unsupported_schema")
    for key in (
        "record_id",
        "sprite_id",
        "blob_path",
        "source_artifact",
        "source_record_id",
        "source_pack",
        "sub_artist",
        "source_family",
        "license",
        "geometry_family_id",
    ):
        _require_string(record, key)
    _require_hash(record, "blob_sha256")
    _require_hash(record, "exported_rgba_hash")
    blob_path = record.get("blob_path")
    expected_blob_path = f"blobs/{record['exported_rgba_hash']}.rgba"
    if (
        not isinstance(blob_path, str)
        or "\\" in blob_path
        or PurePosixPath(blob_path).is_absolute()
        or ".." in PurePosixPath(blob_path).parts
        or blob_path != expected_blob_path
    ):
        raise _error(
            "record blob_path must be the confined canonical content-addressed blob path",
            23,
            "unsupported_schema",
        )
    for key in ("exported_width", "exported_height"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _error(f"invalid {key}", 23, "unsupported_schema")
    for key in ("alpha_mask_hash", "normalized_alpha_hash"):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not HASH_RE.fullmatch(value)):
            raise _error(f"invalid {key}", 23, "unsupported_schema")
    if record.get("provenance_status") != "verified":
        raise _error("record provenance is not verified", 26, "provenance_failure")
    if record.get("suitability_status") not in {
        "accept",
        "quarantine",
        "reject",
        "approved_existing_dataset",
    }:
        raise _error("invalid suitability status", 23, "unsupported_schema")
    if record.get("quality_eligibility") not in {"eligible", "diagnostic_only", "ineligible"}:
        raise _error("invalid quality eligibility", 23, "unsupported_schema")
    for key in ("suitability_reason_codes", "hard_relation_group_ids"):
        values = record.get(key)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) and value for value in values)
            or values != sorted(set(values))
        ):
            raise _error(f"{key} must be sorted unique nonempty strings", 23, "unsupported_schema")
    for key in ("recolor_family_id", "declared_variant_group_id", "sheet_group_id"):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise _error(f"invalid {key}", 23, "unsupported_schema")
    if record.get("split") not in SPLITS:
        raise _error("invalid record split", 23, "unsupported_schema")
    if record.get("membership") not in {"included", "excluded"}:
        raise _error("invalid record membership", 23, "unsupported_schema")
    for key in ("sampling_weight", "evaluation_weight"):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise _error(f"invalid {key}", 23, "unsupported_schema")
    if record["membership"] == "included":
        if not record.get("view_inclusion_reason") or record.get("view_exclusion_reason") is not None:
            raise _error("included record has invalid inclusion/exclusion reasons", 23, "unsupported_schema")
    else:
        if (
            record.get("sampling_weight") != 0
            or record.get("evaluation_weight") != 0
            or not record.get("view_exclusion_reason")
        ):
            raise _error("excluded record must have zero weights and a reason", 23, "unsupported_schema")
    _validate_supervision(record, weak_cap=weak_cap)
    targets = record["targets"]
    for field in SEMANTIC_FIELDS:
        expected = targets[field]["value"] if targets[field]["state"] == "known" else None
        if record[field] != expected:
            raise _error(
                f"top-level semantic field {field} disagrees with targets",
                23,
                "unsupported_schema",
            )
    if record.get("source_ood"):
        if record.get("source_ood_scope") not in OOD_SCOPES or not record.get("source_ood_rationale"):
            raise _error("source-OOD row requires declared scope and rationale", 23, "unsupported_schema")
    if record.get("open_set") and not record.get("open_set_rationale"):
        raise _error("open-set row requires rationale", 23, "unsupported_schema")
    if not isinstance(record.get("source_ood"), bool) or not isinstance(record.get("open_set"), bool):
        raise _error("OOD and open-set flags must be booleans", 23, "unsupported_schema")
    if view_name in {"v5_eval_balanced", "v5_source_ood", "v5_open_set"} and record["membership"] == "included":
        if record["supervision_class"] != "supervised_strong":
            raise _error("evaluation views require strong supervision", 23, "unsupported_schema")
        if record["suitability_status"] != "accept":
            raise _error("evaluation views require accept-only records", 23, "unsupported_schema")
        if record["sampling_weight"] != 0 or record["evaluation_weight"] <= 0:
            raise _error(
                "evaluation views require zero sampling and positive evaluation weight", 23, "unsupported_schema"
            )
        if not record.get("evaluation_stratum"):
            raise _error("evaluation views require a calibrated stratum", 23, "unsupported_schema")
    if view_name == "v5_unlabeled" and record["membership"] == "included":
        if record["supervision_class"] not in {"unlabeled", "auxiliary_only"}:
            raise _error("v5_unlabeled contains a supervised record", 23, "unsupported_schema")
        if record["evaluation_weight"] != 0:
            raise _error("v5_unlabeled records cannot carry evaluation weight", 23, "unsupported_schema")
        if record["split"] != "unsplit":
            raise _error("v5_unlabeled records must remain unsplit", 23, "unsupported_schema")
    if view_name == "v5_eval_balanced" and record["membership"] == "included":
        if record["split"] != "test":
            raise _error("balanced evaluation records require the test split", 23, "unsupported_schema")
    if view_name == "v5_source_ood" and record["membership"] == "included":
        if record["split"] != "source_ood_test" or record["source_ood"] is not True:
            raise _error("source-OOD view requires flagged source_ood_test records", 23, "unsupported_schema")
    if view_name == "v5_open_set" and record["membership"] == "included":
        if record["split"] != "open_set_test" or record["open_set"] is not True:
            raise _error("open-set view requires flagged open_set_test records", 23, "unsupported_schema")


def _count_scalar(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = Counter(_strict_json(record.get(field), pretty=False) for record in records)
    return dict(sorted(values.items()))


def _blob_map(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        path = str(record["blob_path"])
        blob_hash = str(record["blob_sha256"])
        previous = result.setdefault(path, blob_hash)
        if previous != blob_hash:
            raise _error("blob identity maps to inconsistent hashes", 22, "changed_source_hash")
    return dict(sorted(result.items()))


def _contract_identity(contract_root: Path) -> dict[str, Any]:
    paths = _contract_paths(contract_root)
    return {
        "version": CONTRACT_VERSION,
        "artifact_sha256": {name: _file_hash(path) for name, path in sorted(paths.items())},
    }


def _copy_view_blobs(
    staging: Path,
    records: Sequence[Mapping[str, Any]],
    blob_sources: Mapping[str, Path],
) -> None:
    for record in records:
        target = staging / str(record["blob_path"])
        source = blob_sources[str(record["record_id"])]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _file_hash(target) != record["blob_sha256"]:
                raise _error("inconsistent duplicate blob content", 22, "changed_source_hash")
            continue
        shutil.copyfile(source, target)
        if _file_hash(target) != record["blob_sha256"]:
            raise _error("copied blob failed hash verification", 22, "changed_source_hash")


def _purpose(contract_root: Path, view_name: str) -> str:
    contracts = _read_json(contract_root / "view_contracts.json")
    value = (contracts.get("views") or {}).get(view_name, {}).get("purpose")
    if not isinstance(value, str) or not value:
        raise _error("view contract has no purpose", 23, "unsupported_schema")
    return value


def _production_blockers(
    view_name: str,
    policy: Mapping[str, Any],
    frozen_r2: Mapping[str, Any],
    leakage: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not policy.get("independent_builder_audit"):
        blockers.append("fresh independent Dataset-v5 builder audit is required")
    if policy.get("synthetic_fixture") is True:
        blockers.append("synthetic fixture cannot be production-frozen")
    if view_name == "v5_debug":
        blockers.append("v5_debug is permanently diagnostic and promotion-forbidden")
    if view_name in {"v5_architecture", "v5_scale_check"}:
        blockers.append(f"{view_name} cannot authorize production training under the approved contract")
    if view_name == "v5_scale_check" and not policy.get("architecture_core_binding"):
        blockers.append("scale-check policy has no exact architecture-core binding")
    if view_name in {"v5_eval_balanced", "v5_source_ood", "v5_open_set"}:
        minimum = policy.get("approved_minimum_size")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or len(records) < minimum:
            blockers.append("evaluation minimum size is unapproved or unmet")
    if view_name == "v5_open_set" and not policy.get("open_set_aliases"):
        blockers.append("open-set alias/taxonomy closure is not freeze-bound")
    if view_name == "v5_open_set" and not policy.get("open_set_taxonomy"):
        blockers.append("open-set taxonomy identity is not freeze-bound")
    if view_name == "v5_source_ood":
        scopes = {record.get("source_ood_scope") for record in records if record.get("source_ood")}
        if policy.get("source_ood_scope") not in OOD_SCOPES or scopes != {policy.get("source_ood_scope")}:
            blockers.append("source-OOD scope is not singly policy-bound across included records")
    if view_name == "v5_unlabeled" and not frozen_r2.get("adapter_verified"):
        blockers.append("frozen-r2 one-to-one named adapter is not verified")
    if not policy.get("source_manifest_sha256"):
        blockers.append("source manifest hashes were not predeclared by policy")
    if float(policy.get("candidate_raking_factor", 1.0)) != 1.0:
        blockers.append("candidate diagnostic raking is non-unit")
    if leakage.get("unknown_translation_count") or leakage.get("unknown_flip_count"):
        blockers.append("translation/flip relation coverage is incomplete")
    if not policy.get("near_duplicate_detector"):
        blockers.append("near-duplicate detector is not freeze-bound")
    return sorted(set(blockers))


def build_view(
    contract_root: str | Path,
    view_name: str,
    policy_path: str | Path,
    source_manifests: Sequence[str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise _error(f"output root already exists: {output}", 20, "existing_output_root")
    output.parent.mkdir(parents=True, exist_ok=True)
    contract = Path(contract_root).resolve()
    validate_contract(contract)
    if view_name not in VIEW_NAMES:
        raise _error("unsupported Dataset-v5 view", 23, "unsupported_schema")
    policy_file = _resolve_file(policy_path)
    policy = _load_policy(policy_file, view_name)
    frozen_r2 = _validate_frozen_r2_binding(view_name, policy)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        records, metadata, blob_sources, license_rows, source_binding = _load_sources(source_manifests, policy)
        _validate_r2_adapter(view_name, policy, records, frozen_r2)
        _preselection_independence_check(view_name, records, metadata, policy)
        _select_records(view_name, records, metadata, policy)
        components, weight_details = _assign_splits_and_weights(view_name, records, metadata, policy)
        included = sorted(
            (record for record in records if record["membership"] == "included"),
            key=lambda record: str(record["record_id"]),
        )
        excluded = sorted(
            (record for record in records if record["membership"] == "excluded"),
            key=lambda record: str(record["record_id"]),
        )
        if not included:
            raise _error("view selection produced no included records", 24, "partial_view")
        leakage = validate_leakage(records)
        if not leakage["passed"]:
            raise _error(
                "view has hard-relation leakage",
                25,
                "leakage_failure",
                details={"crossings": leakage["crossings"]},
            )
        record_schema = _read_json(contract / "dataset_v5_record.schema.json")
        for record in records:
            _validate_record(
                record,
                record_schema=record_schema,
                view_name=view_name,
                weak_cap=float(policy.get("weak_weight_cap", 0.8)),
            )
        _copy_view_blobs(staging, included, blob_sources)
        _write_jsonl(staging / "record_manifest.jsonl", included)
        _write_jsonl(staging / "excluded_record_manifest.jsonl", excluded)
        _write_json(
            staging / "split_manifest.json",
            {
                "schema_version": "dataset_v5_split_manifest_v1.0.0",
                "view_name": view_name,
                "assignments": [
                    {"record_id": record["record_id"], "membership": record["membership"], "split": record["split"]}
                    for record in sorted(records, key=lambda item: str(item["record_id"]))
                ],
                "hard_relation_components": [
                    {"component_id": component_id, "record_ids": members}
                    for component_id, members in sorted(components.items())
                ],
            },
        )
        _write_jsonl(
            staging / "weight_manifest.jsonl",
            [
                {
                    "record_id": record["record_id"],
                    "membership": record["membership"],
                    "sampling_weight": record["sampling_weight"],
                    "evaluation_weight": record["evaluation_weight"],
                    "review_quality": metadata[str(record["record_id"])]["review_quality"],
                    "quality_multiplier": weight_details.get(
                        str(record["record_id"]),
                        {
                            "quality_multiplier": QUALITY_MULTIPLIERS[
                                str(metadata[str(record["record_id"])]["review_quality"])
                            ]
                        },
                    )["quality_multiplier"],
                    "recolor_multiplier": weight_details.get(str(record["record_id"]), {}).get(
                        "recolor_multiplier", 0.0
                    ),
                    "candidate_raking_factor": weight_details.get(str(record["record_id"]), {}).get(
                        "candidate_raking_factor", 0.0
                    ),
                }
                for record in records
            ],
        )
        _write_jsonl(
            staging / "evaluation_manifest.jsonl",
            [
                {
                    "record_id": record["record_id"],
                    "split": record["split"],
                    "evaluation_weight": record["evaluation_weight"],
                    "evaluation_stratum": record["evaluation_stratum"],
                    "source_ood": record["source_ood"],
                    "source_ood_scope": record["source_ood_scope"],
                    "open_set": record["open_set"],
                }
                for record in included
                if record["evaluation_weight"] > 0
            ],
        )
        _write_jsonl(staging / "license_provenance.jsonl", license_rows)
        _write_json(
            staging / "relation_manifest.json",
            {
                "schema_version": "dataset_v5_relation_manifest_v1.0.0",
                "view_name": view_name,
                "components": [
                    {"component_id": component_id, "record_ids": members}
                    for component_id, members in sorted(components.items())
                ],
                "leakage": leakage,
            },
        )
        resolved_policy = copy.deepcopy(policy)
        resolved_policy["policy_sha256"] = _file_hash(policy_file)
        resolved_policy["frozen_r2_validation"] = frozen_r2
        _write_json(staging / "resolved_policy.json", resolved_policy)
        source_binding["policy"] = {
            "logical_path": _logical_source_path(policy_file, _file_hash(policy_file)),
            "locator": _stored_locator(policy_file, synthetic_fixture=policy.get("synthetic_fixture") is True),
            "sha256": _file_hash(policy_file),
        }
        _write_json(staging / "source_binding.json", source_binding)
        preliminary_artifacts = {
            path.relative_to(staging).as_posix(): _file_hash(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        validation_report = {
            "schema_version": "dataset_v5_view_validation_report_v1.0.0",
            "contract_version": CONTRACT_VERSION,
            "view_name": view_name,
            "ok": True,
            "record_count": len(included),
            "excluded_record_count": len(excluded),
            "supervision_runtime_validation": True,
            "license_provenance_validation": True,
            "frozen_r2_validation": frozen_r2,
            "leakage": leakage,
            "artifact_sha256": preliminary_artifacts,
            "production_blockers": _production_blockers(view_name, policy, frozen_r2, leakage, included),
            "production_freeze_authorized": False,
        }
        _write_json(staging / "validation_report.json", validation_report)
        blob_map = _blob_map(included)
        source_artifacts = [source["logical_path"] for source in source_binding["sources"]]
        source_hashes = {source["logical_path"]: source["sha256"] for source in source_binding["sources"]}
        freeze_artifacts = sorted([*preliminary_artifacts, "validation_report.json", "view_manifest.json"])
        manifest = {
            "schema_version": VIEW_MANIFEST_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "example_only": policy.get("synthetic_fixture") is True,
            "view_name": view_name,
            "view_purpose": _purpose(contract, view_name),
            "view_status": policy["view_status"],
            "production_frozen": False,
            "promotion_forbidden": True,
            "created_by_command": f"dataset-v5 build-view --view {view_name} --output <fresh-output>",
            "code_identity": _git_identity(policy),
            "contract_identity": _contract_identity(contract),
            "source_artifacts": sorted(source_artifacts),
            "source_artifact_sha256": dict(sorted(source_hashes.items())),
            "record_manifest_sha256": _file_hash(staging / "record_manifest.jsonl"),
            "blob_store_identity": "content-addressed-canonical-rgba-v1",
            "blob_store_sha256": _stable_hash(blob_map),
            "blob_store_frozen_hash_map": blob_map,
            "record_count": len(included),
            "unique_geometry_count": len({str(record["geometry_family_id"]) for record in included}),
            "unique_rgba_count": len({str(record["exported_rgba_hash"]) for record in included}),
            "split_counts": dict(sorted(Counter(str(record["split"]) for record in included).items())),
            "supervision_class_counts": dict(
                sorted(Counter(str(record["supervision_class"]) for record in included).items())
            ),
            "suitability_counts": dict(
                sorted(Counter(str(record["suitability_status"]) for record in included).items())
            ),
            "license_counts": dict(sorted(Counter(str(record["license"]) for record in included).items())),
            "category_counts": _count_scalar(included, "category"),
            "pack_counts": dict(sorted(Counter(str(record["source_pack"]) for record in included).items())),
            "artist_counts": dict(sorted(Counter(str(record["sub_artist"]) for record in included).items())),
            "hard_relation_validation": {"passed": True, "crossing_count": 0},
            "freeze_boundary": {
                "complete": False,
                "artifacts": freeze_artifacts,
                "artifact_sha256": {
                    **preliminary_artifacts,
                    "validation_report.json": _file_hash(staging / "validation_report.json"),
                },
                "production_freeze_authorized": False,
            },
        }
        _write_json(staging / "view_manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "ok": True,
        "schema_version": "dataset_v5_view_build_result_v1.0.0",
        "view_name": view_name,
        "view_status": policy["view_status"],
        "record_count": len(included),
        "excluded_record_count": len(excluded),
        "production_frozen": False,
        "promotion_forbidden": True,
    }


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted((dict(row) for row in rows), key=_record_sort_key)
    return "".join(_strict_json(row, pretty=False) + "\n" for row in ordered).encode("utf-8")


def _verify_source_bindings(binding: Mapping[str, Any], errors: list[str]) -> None:
    sources = binding.get("sources")
    if not isinstance(sources, list):
        errors.append("source binding has no source list")
        return
    for source in sources:
        if not isinstance(source, Mapping):
            errors.append("source binding row is malformed")
            continue
        locator = source.get("locator")
        expected = source.get("sha256")
        path = _resolve_locator(locator)
        if not path.is_file():
            errors.append(f"source manifest is missing: {source.get('logical_path')}")
        elif _file_hash(path) != expected:
            errors.append(f"source manifest hash mismatch: {source.get('logical_path')}")
        blobs = source.get("blobs")
        if not isinstance(blobs, list):
            errors.append(f"source blob binding is missing: {source.get('logical_path')}")
            continue
        for blob in blobs:
            if not isinstance(blob, Mapping):
                errors.append("source blob binding row is malformed")
                continue
            blob_locator = blob.get("locator")
            blob_path = _resolve_locator(blob_locator)
            if not blob_path.is_file():
                errors.append(f"source blob is missing: {blob.get('record_id')}")
            elif _file_hash(blob_path) != blob.get("sha256"):
                errors.append(f"source blob hash mismatch: {blob.get('record_id')}")
    policy = binding.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy source binding is missing")
    else:
        locator = policy.get("locator")
        path = _resolve_locator(locator)
        if not path.is_file():
            errors.append("policy source is missing")
        elif _file_hash(path) != policy.get("sha256"):
            errors.append("policy source hash mismatch")


def _replay_view_from_bound_sources(
    view_name: str,
    policy: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = binding.get("sources")
    if not isinstance(sources, list) or not sources:
        raise _error("source replay requires bound source manifests", 24, "partial_view")
    locators: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("locator"), str):
            raise _error("source replay binding is malformed", 24, "partial_view")
        locators.append(str(_resolve_locator(source["locator"])))
    policy_binding = binding.get("policy")
    if not isinstance(policy_binding, Mapping):
        raise _error("source replay requires a bound policy", 24, "partial_view")
    policy_path = _resolve_locator(policy_binding.get("locator"))
    replay_policy = _load_policy(policy_path, view_name)
    policy_hash = _file_hash(policy_path)
    if policy_hash != policy_binding.get("sha256"):
        raise _error("bound policy source hash changed", 22, "changed_source_hash")
    frozen_r2 = _validate_frozen_r2_binding(view_name, replay_policy)
    replay, metadata, _, license_rows, replay_binding = _load_sources(locators, replay_policy)
    expected_binding_sources = sorted(
        (dict(source) for source in sources), key=lambda source: str(source.get("logical_path"))
    )
    if replay_binding.get("sources") != expected_binding_sources:
        raise _error("source replay binding differs from the view binding", 22, "changed_source_hash")
    _validate_r2_adapter(view_name, replay_policy, replay, frozen_r2)
    expected_resolved_policy = copy.deepcopy(replay_policy)
    expected_resolved_policy["policy_sha256"] = policy_hash
    expected_resolved_policy["frozen_r2_validation"] = frozen_r2
    if dict(policy) != expected_resolved_policy:
        raise _error(
            "resolved policy differs from deterministic resolution of its bound source",
            22,
            "changed_source_hash",
        )
    _preselection_independence_check(view_name, replay, metadata, replay_policy)
    _select_records(view_name, replay, metadata, replay_policy)
    _assign_splits_and_weights(view_name, replay, metadata, replay_policy)
    included = sorted(
        (record for record in replay if record["membership"] == "included"),
        key=lambda record: str(record["record_id"]),
    )
    excluded = sorted(
        (record for record in replay if record["membership"] == "excluded"),
        key=lambda record: str(record["record_id"]),
    )
    return included, excluded, license_rows


def _verify_view_impl(contract_root: Path, view_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = set(VIEW_ARTIFACTS)
    missing = sorted(name for name in required if not (view_root / name).is_file())
    if missing:
        return {
            "ok": False,
            "schema_version": "dataset_v5_view_verification_report_v1.0.0",
            "errors": [f"partial view; missing artifacts: {', '.join(missing)}"],
            "record_count": 0,
            "leakage": {"passed": False, "crossing_count": 0, "crossings": []},
        }
    try:
        manifest = _read_json(view_root / "view_manifest.json")
        policy = _read_json(view_root / "resolved_policy.json")
        binding = _read_json(view_root / "source_binding.json")
        records = _read_jsonl(view_root / "record_manifest.jsonl")
        excluded = _read_jsonl(view_root / "excluded_record_manifest.jsonl")
        split_manifest = _read_json(view_root / "split_manifest.json")
        weight_rows = _read_jsonl(view_root / "weight_manifest.jsonl")
        evaluation_rows = _read_jsonl(view_root / "evaluation_manifest.jsonl")
        license_rows = _read_jsonl(view_root / "license_provenance.jsonl")
        relation_manifest = _read_json(view_root / "relation_manifest.json")
        validation_report = _read_json(view_root / "validation_report.json")
    except DatasetV5ViewError as exc:
        return {
            "ok": False,
            "schema_version": "dataset_v5_view_verification_report_v1.0.0",
            "errors": [str(exc)],
            "record_count": 0,
            "leakage": {"passed": False, "crossing_count": 0, "crossings": []},
        }
    view_name = manifest.get("view_name")
    if view_name not in VIEW_NAMES:
        errors.append("view manifest has unsupported view name")
        view_name = "v5_debug"
    if policy.get("schema_version") != POLICY_SCHEMA or policy.get("view_name") != view_name:
        errors.append("resolved policy schema/view identity mismatch")
    if (policy.get("synthetic_fixture") is True) is not (manifest.get("example_only") is True):
        errors.append("resolved policy synthetic status disagrees with view manifest")
    if not manifest.get("production_frozen") and policy.get("view_status") != manifest.get("view_status"):
        errors.append("resolved policy status disagrees with non-production view manifest")
    manifest_schema = _read_json(contract_root / "dataset_v5_view_manifest.schema.json")
    manifest_properties = manifest_schema.get("properties") or {}
    manifest_required = manifest_schema.get("required") or []
    missing_manifest = sorted(set(manifest_required) - set(manifest))
    extra_manifest = sorted(set(manifest) - set(manifest_properties))
    if missing_manifest or extra_manifest:
        errors.append(f"view manifest schema mismatch: missing={missing_manifest}, extra={extra_manifest}")
    if manifest.get("schema_version") != VIEW_MANIFEST_SCHEMA or manifest.get("contract_version") != CONTRACT_VERSION:
        errors.append("view manifest has unsupported schema or contract version")
    status = manifest.get("view_status")
    production_frozen = manifest.get("production_frozen")
    example_only = manifest.get("example_only")
    promotion_forbidden = manifest.get("promotion_forbidden")
    if status not in {"preview", "diagnostic", "candidate", "frozen_production", "deprecated"}:
        errors.append("view manifest has invalid status")
    if not all(isinstance(value, bool) for value in (production_frozen, example_only, promotion_forbidden)):
        errors.append("view manifest production/example/promotion flags must be booleans")
    if production_frozen is True and (
        status != "frozen_production" or example_only is not False or promotion_forbidden is not False
    ):
        errors.append("production-frozen manifest has incoherent status or flags")
    if status == "frozen_production" and production_frozen is not True:
        errors.append("frozen-production status lacks a production freeze")
    if example_only is True and (production_frozen is not False or promotion_forbidden is not True):
        errors.append("example-only manifest has a production or promotion claim")
    if manifest.get("view_name") == "v5_debug" and production_frozen is not False:
        errors.append("v5_debug cannot be production-frozen")
    code_identity = manifest.get("code_identity")
    if (
        not isinstance(code_identity, Mapping)
        or set(code_identity) != {"git_commit", "dirty"}
        or not isinstance(code_identity.get("git_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", str(code_identity.get("git_commit")))
        or not isinstance(code_identity.get("dirty"), bool)
    ):
        errors.append("view manifest code identity is malformed")
    if manifest.get("production_frozen") is not False and not (view_root / "FREEZE.json").is_file():
        errors.append("production-frozen claim has no FREEZE.json")
    if manifest.get("promotion_forbidden") is not True and not manifest.get("production_frozen"):
        errors.append("non-production view must remain promotion-forbidden")
    expected_contract = _contract_identity(contract_root)
    if manifest.get("contract_identity") != expected_contract:
        errors.append("contract artifact hash binding mismatch")
    if _canonical_jsonl_bytes(records) != (view_root / "record_manifest.jsonl").read_bytes():
        errors.append("record manifest is not canonical JSONL")
    if _canonical_jsonl_bytes(excluded) != (view_root / "excluded_record_manifest.jsonl").read_bytes():
        errors.append("excluded-record manifest is not canonical JSONL")
    for name, value in (
        ("split_manifest.json", split_manifest),
        ("resolved_policy.json", policy),
        ("source_binding.json", binding),
        ("relation_manifest.json", relation_manifest),
        ("validation_report.json", validation_report),
        ("view_manifest.json", manifest),
    ):
        canonical = (_strict_json(value, pretty=True) + "\n").encode("utf-8")
        if canonical != (view_root / name).read_bytes():
            errors.append(f"artifact is not canonical JSON: {name}")
    for name, rows in (
        ("weight_manifest.jsonl", weight_rows),
        ("evaluation_manifest.jsonl", evaluation_rows),
        ("license_provenance.jsonl", license_rows),
    ):
        if _canonical_jsonl_bytes(rows) != (view_root / name).read_bytes():
            errors.append(f"artifact is not canonical JSONL: {name}")
    if manifest.get("record_manifest_sha256") != _file_hash(view_root / "record_manifest.jsonl"):
        errors.append("record manifest hash mismatch")
    record_schema = _read_json(contract_root / "dataset_v5_record.schema.json")
    seen_ids: set[str] = set()
    for record in [*records, *excluded]:
        record_id = str(record.get("record_id") or "")
        if not record_id or record_id in seen_ids:
            errors.append(f"duplicate or empty record identity: {record_id}")
        seen_ids.add(record_id)
        try:
            _validate_record(
                record,
                record_schema=record_schema,
                view_name=str(view_name),
                weak_cap=float(policy.get("weak_weight_cap", 0.8)),
            )
        except DatasetV5ViewError as exc:
            errors.append(f"record {record_id}: {exc}")
    if any(record.get("membership") != "included" for record in records):
        errors.append("record_manifest contains excluded membership")
    if any(record.get("membership") != "excluded" for record in excluded):
        errors.append("excluded_record_manifest contains included membership")
    all_records = sorted([*records, *excluded], key=lambda item: str(item.get("record_id")))
    expected_components = _hard_components(records)
    expected_split_manifest = {
        "schema_version": "dataset_v5_split_manifest_v1.0.0",
        "view_name": view_name,
        "assignments": [
            {
                "record_id": record.get("record_id"),
                "membership": record.get("membership"),
                "split": record.get("split"),
            }
            for record in all_records
        ],
        "hard_relation_components": [
            {"component_id": component_id, "record_ids": members}
            for component_id, members in sorted(expected_components.items())
        ],
    }
    if split_manifest != expected_split_manifest:
        errors.append("split manifest disagrees with record membership/splits or hard closure")
    record_by_id = {str(record.get("record_id")): record for record in all_records}
    weights_by_id = {str(row.get("record_id")): row for row in weight_rows if isinstance(row.get("record_id"), str)}
    if len(weights_by_id) != len(weight_rows) or set(weights_by_id) != set(record_by_id):
        errors.append("weight manifest record inventory mismatch")
    recolor_counts = Counter(
        str(record.get("recolor_family_id")) for record in records if record.get("recolor_family_id") is not None
    )
    for record_id, record in record_by_id.items():
        row = weights_by_id.get(record_id)
        if row is None:
            continue
        if row.get("membership") != record.get("membership"):
            errors.append(f"weight manifest membership mismatch: {record_id}")
        if row.get("sampling_weight") != record.get("sampling_weight") or row.get("evaluation_weight") != record.get(
            "evaluation_weight"
        ):
            errors.append(f"weight manifest weight mismatch: {record_id}")
        review_quality = row.get("review_quality")
        if review_quality not in QUALITY_MULTIPLIERS or row.get("quality_multiplier") != QUALITY_MULTIPLIERS.get(
            str(review_quality)
        ):
            errors.append(f"weight manifest quality multiplier mismatch: {record_id}")
        if record.get("membership") == "included":
            recolor_family = record.get("recolor_family_id")
            expected_recolor = 1.0 / recolor_counts[str(recolor_family)] if recolor_family is not None else 1.0
            if row.get("recolor_multiplier") != expected_recolor:
                errors.append(f"weight manifest recolor multiplier mismatch: {record_id}")
            if row.get("candidate_raking_factor") != float(policy.get("candidate_raking_factor", 1.0)):
                errors.append(f"weight manifest raking factor mismatch: {record_id}")
        elif row.get("recolor_multiplier") != 0.0 or row.get("candidate_raking_factor") != 0.0:
            errors.append(f"excluded record has active weight factors: {record_id}")
    expected_evaluation = [
        {
            "record_id": record.get("record_id"),
            "split": record.get("split"),
            "evaluation_weight": record.get("evaluation_weight"),
            "evaluation_stratum": record.get("evaluation_stratum"),
            "source_ood": record.get("source_ood"),
            "source_ood_scope": record.get("source_ood_scope"),
            "open_set": record.get("open_set"),
        }
        for record in records
        if float(record.get("evaluation_weight", 0)) > 0
    ]
    if evaluation_rows != expected_evaluation:
        errors.append("evaluation manifest disagrees with canonical view records")
    license_by_id = {str(row.get("record_id")): row for row in license_rows if isinstance(row.get("record_id"), str)}
    if len(license_by_id) != len(license_rows) or set(license_by_id) != set(record_by_id):
        errors.append("license/provenance manifest record inventory mismatch")
    for record_id, record in record_by_id.items():
        row = license_by_id.get(record_id)
        if row is None:
            continue
        for key in (
            "source_artifact",
            "source_record_id",
            "source_pack",
            "sub_artist",
            "license",
            "provenance_status",
        ):
            if row.get(key) != record.get(key):
                errors.append(f"license/provenance field mismatch for {record_id}: {key}")
    try:
        replay_records, replay_excluded, replay_license_rows = _replay_view_from_bound_sources(
            str(view_name), policy, binding
        )
        if replay_records != records:
            errors.append("record manifest differs from deterministic replay of exact bound sources")
        if replay_excluded != excluded:
            errors.append("excluded-record manifest differs from deterministic source replay")
        if _canonical_jsonl_bytes(replay_license_rows) != _canonical_jsonl_bytes(license_rows):
            errors.append("license/provenance manifest differs from deterministic source replay")
    except DatasetV5ViewError as exc:
        errors.append(f"bound source replay failed: {exc}")
    output_blob_map = _blob_map(records) if records else {}
    if manifest.get("blob_store_frozen_hash_map") != output_blob_map:
        errors.append("blob-store frozen hash map mismatch")
    if manifest.get("blob_store_sha256") != _stable_hash(output_blob_map):
        errors.append("blob-store aggregate hash mismatch")
    for record in records:
        blob_path = (view_root / str(record.get("blob_path"))).resolve()
        try:
            blob_path.relative_to(view_root.resolve())
        except ValueError:
            errors.append(f"view blob path escapes the view root: {record.get('record_id')}")
            continue
        if not blob_path.is_file():
            errors.append(f"view blob is missing: {record.get('record_id')}")
            continue
        if _file_hash(blob_path) != record.get("blob_sha256"):
            errors.append(f"view blob hash mismatch: {record.get('record_id')}")
            continue
        width = record.get("exported_width")
        height = record.get("exported_height")
        if isinstance(width, int) and isinstance(height, int) and blob_path.stat().st_size == width * height * 4:
            rgba = np.frombuffer(blob_path.read_bytes(), dtype=np.uint8).reshape(height, width, 4)
            if canonical_rgba_sha256(rgba) != record.get("exported_rgba_hash"):
                errors.append(f"view blob canonical RGBA identity mismatch: {record.get('record_id')}")
            if alpha_mask_sha256(rgba[..., 3]) != record.get("alpha_mask_hash"):
                errors.append(f"view blob alpha-mask identity mismatch: {record.get('record_id')}")
            if _normalized_alpha_key(rgba[..., 3]) != record.get("normalized_alpha_hash"):
                errors.append(f"view blob normalized-alpha identity mismatch: {record.get('record_id')}")
        else:
            errors.append(f"view blob dimensions mismatch: {record.get('record_id')}")
    expected_counts = {
        "record_count": len(records),
        "unique_geometry_count": len({str(record.get("geometry_family_id")) for record in records}),
        "unique_rgba_count": len({str(record.get("exported_rgba_hash")) for record in records}),
        "split_counts": dict(sorted(Counter(str(record.get("split")) for record in records).items())),
        "supervision_class_counts": dict(
            sorted(Counter(str(record.get("supervision_class")) for record in records).items())
        ),
        "suitability_counts": dict(
            sorted(Counter(str(record.get("suitability_status")) for record in records).items())
        ),
        "license_counts": dict(sorted(Counter(str(record.get("license")) for record in records).items())),
        "category_counts": _count_scalar(records, "category"),
        "pack_counts": dict(sorted(Counter(str(record.get("source_pack")) for record in records).items())),
        "artist_counts": dict(sorted(Counter(str(record.get("sub_artist")) for record in records).items())),
    }
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            errors.append(f"view manifest count mismatch: {key}")
    source_hashes = {
        str(source.get("logical_path")): source.get("sha256")
        for source in binding.get("sources", [])
        if isinstance(source, Mapping)
    }
    if manifest.get("source_artifacts") != sorted(source_hashes):
        errors.append("source artifact identity list mismatch")
    if manifest.get("source_artifact_sha256") != dict(sorted(source_hashes.items())):
        errors.append("source artifact hash map mismatch")
    _verify_source_bindings(binding, errors)
    freeze_boundary = manifest.get("freeze_boundary")
    if not isinstance(freeze_boundary, Mapping):
        errors.append("view manifest freeze boundary is missing")
    else:
        expected_inventory = {
            path.relative_to(view_root).as_posix()
            for path in view_root.rglob("*")
            if path.is_file() and path.name != "FREEZE.json"
        }
        if freeze_boundary.get("artifacts") != sorted(expected_inventory):
            errors.append("view manifest freeze-boundary artifact inventory is incomplete")
        if freeze_boundary.get("complete") is not (production_frozen is True):
            errors.append("freeze-boundary completeness disagrees with production status")
        artifact_hashes = freeze_boundary.get("artifact_sha256")
        if not isinstance(artifact_hashes, Mapping):
            errors.append("view manifest artifact hash map is missing")
        else:
            if set(artifact_hashes) != expected_inventory - {"view_manifest.json"}:
                errors.append("view manifest freeze-boundary hash inventory is incomplete")
            for relative, expected in artifact_hashes.items():
                path = view_root / str(relative)
                if not path.is_file():
                    errors.append(f"freeze-bound view artifact is missing: {relative}")
                elif _file_hash(path) != expected:
                    errors.append(f"freeze-bound view artifact hash mismatch: {relative}")
    leakage = validate_leakage([*records, *excluded])
    if not leakage["passed"]:
        errors.append(f"leakage validation failed with {leakage['crossing_count']} crossing(s)")
    if manifest.get("hard_relation_validation") != {
        "passed": leakage["passed"],
        "crossing_count": leakage["crossing_count"],
    }:
        errors.append("hard-relation validation summary mismatch")
    expected_relation = {
        "schema_version": "dataset_v5_relation_manifest_v1.0.0",
        "view_name": view_name,
        "components": [
            {"component_id": component_id, "record_ids": members}
            for component_id, members in sorted(expected_components.items())
        ],
        "leakage": leakage,
    }
    if relation_manifest != expected_relation:
        errors.append("relation manifest disagrees with recomputed hard closure/leakage")
    if validation_report.get("ok") is not True or validation_report.get("view_name") != view_name:
        errors.append("validation report identity or verdict mismatch")
    if validation_report.get("record_count") != len(records) or validation_report.get("excluded_record_count") != len(
        excluded
    ):
        errors.append("validation report record counts mismatch")
    if validation_report.get("leakage") != leakage:
        errors.append("validation report leakage result mismatch")
    expected_validation_hashes = {
        path.relative_to(view_root).as_posix(): _file_hash(path)
        for path in sorted(view_root.rglob("*"))
        if path.is_file() and path.name not in {"validation_report.json", "view_manifest.json", "FREEZE.json"}
    }
    if validation_report.get("artifact_sha256") != expected_validation_hashes:
        errors.append("validation report artifact hash map is incomplete or stale")
    if validation_report.get("supervision_runtime_validation") is not True:
        errors.append("validation report omits runtime supervision validation")
    if validation_report.get("license_provenance_validation") is not True:
        errors.append("validation report omits license/provenance validation")
    if validation_report.get("frozen_r2_validation") != policy.get("frozen_r2_validation"):
        errors.append("validation report frozen-r2 result disagrees with resolved policy")
    if validation_report.get("production_blockers") != _production_blockers(
        str(view_name),
        policy,
        policy.get("frozen_r2_validation") or {},
        leakage,
        records,
    ):
        errors.append("validation report production blockers are stale or incomplete")
    if validation_report.get("production_freeze_authorized") is not False:
        errors.append("candidate validation report must not authorize production freezing")
    policy_binding = binding.get("policy")
    if not isinstance(policy_binding, Mapping) or policy.get("policy_sha256") != policy_binding.get("sha256"):
        errors.append("resolved policy hash disagrees with exact policy source binding")
    return {
        "ok": not errors,
        "schema_version": "dataset_v5_view_verification_report_v1.0.0",
        "contract_version": CONTRACT_VERSION,
        "view_name": manifest.get("view_name"),
        "record_count": len(records),
        "excluded_record_count": len(excluded),
        "errors": errors,
        "leakage": leakage,
        "source_ood_scopes": sorted(
            {str(record["source_ood_scope"]) for record in records if record.get("source_ood")}
        ),
        "open_set_concepts": sorted(
            {_strict_json(record.get("canonical_object"), pretty=False) for record in records if record.get("open_set")}
        ),
        "production_frozen": manifest.get("production_frozen"),
        "promotion_forbidden": manifest.get("promotion_forbidden"),
    }


def verify_view(contract_root: str | Path, view_root: str | Path) -> dict[str, Any]:
    contract = Path(contract_root).resolve()
    validate_contract(contract)
    root = Path(view_root).resolve()
    if not root.is_dir():
        return {
            "ok": False,
            "schema_version": "dataset_v5_view_verification_report_v1.0.0",
            "errors": [f"view root is missing: {root}"],
            "record_count": 0,
            "leakage": {"passed": False, "crossing_count": 0, "crossings": []},
        }
    try:
        return _verify_view_impl(contract, root)
    except (DatasetV5ViewError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {
            "ok": False,
            "schema_version": "dataset_v5_view_verification_report_v1.0.0",
            "errors": [str(exc)],
            "record_count": 0,
            "leakage": {"passed": False, "crossing_count": 0, "crossings": []},
        }


def _approved_decisions(path: Path) -> dict[str, Any]:
    decisions = _read_json(path)
    if decisions.get("schema_version") != APPROVED_DECISIONS_SCHEMA:
        raise _error("unsupported approved-decisions schema", 23, "unsupported_schema")
    approvals = decisions.get("approvals")
    if not isinstance(approvals, Mapping) or not all(
        isinstance(key, str) and isinstance(value, bool) for key, value in approvals.items()
    ):
        raise _error("approved decisions require a boolean approvals map", 23, "unsupported_schema")
    return decisions


def _all_view_file_hashes(view_root: Path) -> dict[str, str]:
    return {
        path.relative_to(view_root).as_posix(): _file_hash(path)
        for path in sorted(view_root.rglob("*"))
        if path.is_file() and path.name != "FREEZE.json"
    }


def _verification_failure_code(errors: Sequence[str]) -> tuple[int, str]:
    text = "\n".join(errors).casefold()
    if any(
        token in text
        for token in (
            "partial view",
            "missing artifacts",
            "inventory is incomplete",
            "view artifact is missing",
            "view blob is missing",
            "source replay requires a bound",
        )
    ):
        return 24, "partial_view"
    if "leakage" in text or "crossing" in text:
        return 25, "leakage_failure"
    if "license" in text and ("failed" in text or "mismatch" in text or "missing" in text):
        return 27, "license_failure"
    if "provenance" in text and ("failed" in text or "mismatch" in text or "missing" in text):
        return 26, "provenance_failure"
    if "source" in text and "missing" in text:
        return 21, "missing_source_artifact"
    if "source" in text and ("hash" in text or "changed" in text or "replay" in text):
        return 22, "changed_source_hash"
    if "schema" in text or "unsupported" in text or "malformed" in text:
        return 23, "unsupported_schema"
    return 2, "validation_failed"


def _mandatory_production_approvals(view_name: str) -> set[str]:
    common = {"independent_builder_audit", "production_freeze", "weighting_policy"}
    by_view = {
        "v5_eval_balanced": {"evaluation_minimum", "human_truth", "calibrated_strata"},
        "v5_source_ood": {
            "evaluation_minimum",
            "human_truth",
            "calibrated_strata",
            "source_ood_scope",
        },
        "v5_open_set": {
            "evaluation_minimum",
            "human_truth",
            "calibrated_strata",
            "open_set_concept",
        },
        "v5_unlabeled": {"frozen_r2_binding", "quarantine_policy"},
    }
    return common | by_view.get(view_name, set())


def _validate_independent_builder_audit(policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    binding = policy.get("independent_builder_audit")
    if not isinstance(binding, Mapping):
        raise _error(
            "production freeze requires a fresh independent builder audit binding",
            24,
            "partial_view",
        )
    audit_path = _resolve_locator(binding.get("path"))
    try:
        audit_path.relative_to(REPO_ROOT)
    except ValueError:
        raise _error(
            "independent builder audit must be repository-bound",
            23,
            "unsupported_external_path",
        ) from None
    expected_hash = binding.get("sha256")
    if (
        not audit_path.is_file()
        or not isinstance(expected_hash, str)
        or not HASH_RE.fullmatch(expected_hash)
        or _file_hash(audit_path) != expected_hash
    ):
        raise _error("independent builder audit is missing or changed", 22, "changed_source_hash")
    audit = _read_json(audit_path)
    if (
        audit.get("schema_version") != "dataset_v5_independent_builder_audit_v1.0.0"
        or audit.get("verdict") != "pass"
        or audit.get("independent") is not True
        or not isinstance(audit.get("auditor_session_id"), str)
        or not audit.get("auditor_session_id")
        or audit.get("contract_identity") != manifest.get("contract_identity")
        or audit.get("builder_code_identity") != manifest.get("code_identity")
    ):
        raise _error(
            "independent builder audit does not certify this contract/code identity",
            24,
            "partial_view",
        )


def _validate_production_authorization(
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    decisions: Mapping[str, Any],
    verification: Mapping[str, Any],
    require_current_code: bool,
) -> None:
    approvals = decisions.get("approvals")
    if not isinstance(approvals, Mapping):
        raise _error("production freeze approvals are missing", 24, "partial_view")
    view_name = str(manifest.get("view_name"))
    required = _mandatory_production_approvals(view_name) | {
        key for key, value in (policy.get("approvals") or {}).items() if value is True
    }
    missing = sorted(key for key in required if approvals.get(key) is not True)
    if missing:
        raise _error(
            "production freeze is missing mandatory contract approval decisions",
            24,
            "partial_view",
            details={"missing_approvals": missing},
        )
    if policy.get("synthetic_fixture") is True:
        raise _error("synthetic fixtures cannot receive a production freeze", 24, "partial_view")
    if policy.get("production_freeze_authorized") is not True:
        raise _error("policy does not authorize a production freeze", 24, "partial_view")
    if view_name in {"v5_debug", "v5_architecture", "v5_scale_check"}:
        raise _error("this view contract cannot authorize production freezing", 24, "partial_view")
    code_identity = manifest.get("code_identity")
    if not isinstance(code_identity, Mapping) or code_identity.get("dirty") is not False:
        raise _error("production freeze requires clean code identity", 24, "partial_view")
    if require_current_code and _git_identity() != code_identity:
        raise _error("production freeze code identity no longer matches HEAD", 24, "partial_view")
    if not (manifest.get("hard_relation_validation") or {}).get("passed"):
        raise _error("production freeze requires zero leakage", 25, "leakage_failure")
    source_hash_policy = policy.get("source_manifest_sha256")
    if not isinstance(source_hash_policy, (str, Mapping)) or not source_hash_policy:
        raise _error(
            "production freeze requires predeclared source hash bindings",
            21,
            "missing_source_artifact",
        )
    if float(policy.get("candidate_raking_factor", 1.0)) != 1.0:
        raise _error("candidate raking cannot enter a production freeze", 24, "partial_view")
    approved_minimum = policy.get("approved_minimum_size")
    if (
        isinstance(approved_minimum, bool)
        or not isinstance(approved_minimum, int)
        or approved_minimum < 1
        or int(manifest.get("record_count", 0)) < approved_minimum
    ):
        raise _error("production freeze requires a met approved minimum size", 24, "partial_view")
    leakage = verification.get("leakage") or {}
    if leakage.get("unknown_translation_count") or leakage.get("unknown_flip_count"):
        raise _error(
            "production freeze requires complete known translation and flip coverage",
            24,
            "partial_view",
        )
    detector = policy.get("near_duplicate_detector")
    if not isinstance(detector, Mapping):
        raise _error("production freeze requires a bound near-duplicate detector", 24, "partial_view")
    detector_path = _resolve_locator(detector.get("path"))
    try:
        detector_path.relative_to(REPO_ROOT)
    except ValueError:
        raise _error(
            "production near-duplicate detector must be repository-bound",
            23,
            "unsupported_external_path",
        ) from None
    detector_hash = detector.get("sha256")
    if (
        not detector_path.is_file()
        or not isinstance(detector_hash, str)
        or not HASH_RE.fullmatch(detector_hash)
        or _file_hash(detector_path) != detector_hash
    ):
        raise _error(
            "production near-duplicate detector binding is missing or changed",
            22,
            "changed_source_hash",
        )
    if view_name == "v5_open_set" and not (
        isinstance(policy.get("open_set_aliases"), Mapping) and policy["open_set_aliases"]
    ):
        raise _error("production open-set freeze requires alias/taxonomy closure", 24, "partial_view")
    if view_name == "v5_open_set":
        taxonomy = policy.get("open_set_taxonomy")
        if not isinstance(taxonomy, Mapping):
            raise _error("production open-set freeze requires a bound taxonomy", 24, "partial_view")
        taxonomy_path = _resolve_locator(taxonomy.get("path"))
        taxonomy_hash = taxonomy.get("sha256")
        try:
            taxonomy_path.relative_to(REPO_ROOT)
        except ValueError:
            raise _error(
                "production open-set taxonomy must be repository-bound",
                23,
                "unsupported_external_path",
            ) from None
        if (
            not taxonomy_path.is_file()
            or not isinstance(taxonomy_hash, str)
            or not HASH_RE.fullmatch(taxonomy_hash)
            or _file_hash(taxonomy_path) != taxonomy_hash
        ):
            raise _error("production open-set taxonomy is missing or changed", 22, "changed_source_hash")
        taxonomy_document = _read_json(taxonomy_path)
        if taxonomy_document.get("schema_version") != "dataset_v5_open_set_taxonomy_v1.0.0" or taxonomy_document.get(
            "aliases"
        ) != policy.get("open_set_aliases"):
            raise _error("open-set taxonomy identity/aliases mismatch", 24, "partial_view")
    if view_name == "v5_source_ood":
        configured_scope = policy.get("source_ood_scope")
        if configured_scope not in OOD_SCOPES or verification.get("source_ood_scopes") != [configured_scope]:
            raise _error(
                "production source-OOD view requires one policy-bound consistent scope",
                24,
                "partial_view",
            )
    if view_name == "v5_unlabeled" and not (policy.get("frozen_r2_validation") or {}).get("adapter_verified"):
        raise _error("production unlabeled freeze requires verified r2 adapter", 22, "changed_source_hash")
    _validate_independent_builder_audit(policy, manifest)


def freeze_view(
    contract_root: str | Path,
    view_root: str | Path,
    approved_decisions: str | Path,
    command_line: str,
) -> dict[str, Any]:
    if not isinstance(command_line, str) or not command_line.strip():
        raise _error("freeze command line must be nonempty", 23, "unsupported_schema")
    contract = Path(contract_root).resolve()
    validate_contract(contract)
    root = Path(view_root).resolve()
    if not root.is_dir():
        raise _error("view root is missing", 21, "missing_source_artifact")
    freeze_path = root / "FREEZE.json"
    if freeze_path.exists():
        raise _error("FREEZE.json already exists", 20, "existing_output_root")
    verification = verify_view(contract, root)
    if not verification["ok"]:
        exit_code, reason_code = _verification_failure_code(verification["errors"])
        raise _error(
            "view verification failed before freeze",
            exit_code,
            reason_code,
            details={"errors": verification["errors"]},
        )
    decisions_path = _resolve_file(approved_decisions)
    decisions = _approved_decisions(decisions_path)
    approvals = decisions["approvals"]
    policy = _read_json(root / "resolved_policy.json")
    manifest_path = root / "view_manifest.json"
    manifest = _read_json(manifest_path)
    policy_approvals = policy.get("approvals") or {}
    mandatory_by_view = {
        "v5_eval_balanced": {"evaluation_minimum", "human_truth", "calibrated_strata"},
        "v5_source_ood": {
            "evaluation_minimum",
            "human_truth",
            "calibrated_strata",
            "source_ood_scope",
        },
        "v5_open_set": {
            "evaluation_minimum",
            "human_truth",
            "calibrated_strata",
            "open_set_concept",
        },
        "v5_unlabeled": {"frozen_r2_binding", "quarantine_policy", "weighting_policy"},
    }
    required_approvals = sorted(
        {key for key, value in policy_approvals.items() if value is True}
        | mandatory_by_view.get(str(manifest.get("view_name")), set())
    )
    missing_approvals = [key for key in required_approvals if approvals.get(key) is not True]
    if missing_approvals:
        raise _error(
            "required approval decisions do not satisfy the resolved policy and contract",
            24,
            "partial_view",
            details={"missing_approvals": missing_approvals},
        )
    production_requested = approvals.get("production_freeze") is True
    final_manifest = copy.deepcopy(manifest)
    if production_requested:
        _validate_production_authorization(
            manifest=manifest,
            policy=policy,
            decisions=decisions,
            verification=verification,
            require_current_code=True,
        )
        final_manifest["view_status"] = "frozen_production"
        final_manifest["production_frozen"] = True
        final_manifest["promotion_forbidden"] = False
        final_manifest["example_only"] = False
        final_manifest["freeze_boundary"]["complete"] = True
        final_manifest["freeze_boundary"]["production_freeze_authorized"] = True
        freeze_kind = "production"
    else:
        freeze_kind = "synthetic_diagnostic"
    artifact_hashes = _all_view_file_hashes(root)
    if production_requested:
        artifact_hashes["view_manifest.json"] = hashlib.sha256(
            (_strict_json(final_manifest, pretty=True) + "\n").encode("utf-8")
        ).hexdigest()
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "freeze_kind": freeze_kind,
        "view_name": final_manifest["view_name"],
        "view_status": final_manifest["view_status"],
        "production_frozen": final_manifest["production_frozen"],
        "promotion_forbidden": final_manifest["promotion_forbidden"],
        "exact_command_line": command_line,
        "code_identity": final_manifest["code_identity"],
        "contract_identity": _contract_identity(contract),
        "contract_root_locator": _repo_relative(contract),
        "approved_decisions": {
            "logical_path": _logical_source_path(decisions_path, _file_hash(decisions_path)),
            "locator": _stored_locator(decisions_path, synthetic_fixture=final_manifest["example_only"] is True),
            "sha256": _file_hash(decisions_path),
        },
        "resolved_policy_sha256": _file_hash(root / "resolved_policy.json"),
        "source_binding_sha256": _file_hash(root / "source_binding.json"),
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }
    try:
        if production_requested:
            _write_json(manifest_path, final_manifest)
        _write_json(freeze_path, freeze)
        result = verify_freeze(root)
        if not result["ok"]:
            raise _error(
                "new freeze failed immediate verification",
                2,
                "validation_failed",
                details={"errors": result["errors"]},
            )
    except BaseException:
        freeze_path.unlink(missing_ok=True)
        if production_requested:
            _write_json(manifest_path, manifest)
        raise
    return {
        "ok": True,
        "schema_version": "dataset_v5_freeze_result_v1.0.0",
        "view_name": final_manifest["view_name"],
        "freeze_kind": freeze_kind,
        "production_frozen": final_manifest["production_frozen"],
        "promotion_forbidden": final_manifest["promotion_forbidden"],
        "freeze_sha256": _file_hash(freeze_path),
    }


def _verify_freeze_impl(view_root: str | Path) -> dict[str, Any]:
    root = Path(view_root).resolve()
    freeze_path = root / "FREEZE.json"
    errors: list[str] = []
    if not freeze_path.is_file():
        return {
            "ok": False,
            "schema_version": "dataset_v5_freeze_verification_report_v1.0.0",
            "errors": ["FREEZE.json is missing"],
        }
    try:
        freeze = _read_json(freeze_path)
    except DatasetV5ViewError as exc:
        return {
            "ok": False,
            "schema_version": "dataset_v5_freeze_verification_report_v1.0.0",
            "errors": [str(exc)],
        }
    if freeze.get("schema_version") != FREEZE_SCHEMA or freeze.get("contract_version") != CONTRACT_VERSION:
        errors.append("unsupported freeze schema or contract")
    required_fields = {
        "schema_version",
        "contract_version",
        "freeze_kind",
        "view_name",
        "view_status",
        "production_frozen",
        "promotion_forbidden",
        "exact_command_line",
        "code_identity",
        "contract_identity",
        "contract_root_locator",
        "approved_decisions",
        "resolved_policy_sha256",
        "source_binding_sha256",
        "artifact_sha256",
    }
    if set(freeze) != required_fields:
        errors.append(
            f"freeze metadata fields mismatch: missing={sorted(required_fields - set(freeze))}, "
            f"extra={sorted(set(freeze) - required_fields)}"
        )
    if (_strict_json(freeze, pretty=True) + "\n").encode("utf-8") != freeze_path.read_bytes():
        errors.append("FREEZE.json is not canonical JSON")
    freeze_kind = freeze.get("freeze_kind")
    if freeze_kind not in {"synthetic_diagnostic", "production"}:
        errors.append("unsupported freeze kind")
    elif freeze_kind == "synthetic_diagnostic" and (
        freeze.get("production_frozen") is not False
        or freeze.get("promotion_forbidden") is not True
        or freeze.get("view_status") == "frozen_production"
    ):
        errors.append("synthetic diagnostic freeze has a production claim")
    elif freeze_kind == "production" and (
        freeze.get("production_frozen") is not True
        or freeze.get("promotion_forbidden") is not False
        or freeze.get("view_status") != "frozen_production"
    ):
        errors.append("production freeze identity is incomplete")
    if not isinstance(freeze.get("exact_command_line"), str) or not str(freeze.get("exact_command_line")).strip():
        errors.append("freeze exact command line is empty")
    for field in ("resolved_policy_sha256", "source_binding_sha256"):
        value = freeze.get(field)
        if not isinstance(value, str) or not HASH_RE.fullmatch(value):
            errors.append(f"freeze has invalid {field}")
    policy_path = root / "resolved_policy.json"
    if policy_path.is_file() and freeze.get("resolved_policy_sha256") != _file_hash(policy_path):
        errors.append("freeze resolved-policy hash mismatch")
    binding_path = root / "source_binding.json"
    if binding_path.is_file() and freeze.get("source_binding_sha256") != _file_hash(binding_path):
        errors.append("freeze source-binding hash mismatch")
    artifact_hashes = freeze.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        errors.append("freeze artifact hash map is missing")
        artifact_hashes = {}
    current_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "FREEZE.json"
    }
    if set(artifact_hashes) != current_files:
        errors.append("freeze artifact inventory is incomplete or has unexpected files")
    for relative, expected in artifact_hashes.items():
        path = root / str(relative)
        if not path.is_file():
            errors.append(f"freeze-bound artifact is missing: {relative}")
        elif _file_hash(path) != expected:
            errors.append(f"freeze-bound artifact hash mismatch: {relative}")
    decisions = freeze.get("approved_decisions")
    decision_document: dict[str, Any] | None = None
    if not isinstance(decisions, Mapping):
        errors.append("approved-decisions freeze binding is missing")
    else:
        locator = decisions.get("locator")
        path = _resolve_locator(locator)
        if not path.is_file():
            errors.append("approved-decisions source is missing")
        elif _file_hash(path) != decisions.get("sha256"):
            errors.append("approved-decisions hash mismatch")
        else:
            try:
                decision_document = _approved_decisions(path)
            except DatasetV5ViewError as exc:
                errors.append(f"approved-decisions validation failed: {exc}")
    contract_locator = freeze.get("contract_root_locator")
    contract = _resolve_locator(contract_locator)
    view_verification: dict[str, Any] | None = None
    if not contract.is_dir():
        errors.append("contract root is missing")
    else:
        try:
            if _contract_identity(contract) != freeze.get("contract_identity"):
                errors.append("freeze contract artifact hash mismatch")
            view_verification = verify_view(contract, root)
            if not view_verification["ok"]:
                errors.extend(f"view verification: {error}" for error in view_verification["errors"])
        except DatasetV5ViewError as exc:
            errors.append(str(exc))
    manifest_path = root / "view_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        for key in ("view_name", "view_status", "production_frozen", "promotion_forbidden", "code_identity"):
            if freeze.get(key) != manifest.get(key):
                errors.append(f"freeze/view manifest identity mismatch: {key}")
        if freeze_kind == "production" and decision_document is not None and view_verification is not None:
            try:
                policy = _read_json(root / "resolved_policy.json")
                _validate_production_authorization(
                    manifest=manifest,
                    policy=policy,
                    decisions=decision_document,
                    verification=view_verification,
                    require_current_code=False,
                )
            except DatasetV5ViewError as exc:
                errors.append(f"production authorization replay failed: {exc}")
    return {
        "ok": not errors,
        "schema_version": "dataset_v5_freeze_verification_report_v1.0.0",
        "contract_version": CONTRACT_VERSION,
        "view_name": freeze.get("view_name"),
        "freeze_kind": freeze.get("freeze_kind"),
        "freeze_sha256": _file_hash(freeze_path),
        "artifact_count": len(artifact_hashes),
        "errors": errors,
    }


def verify_freeze(view_root: str | Path) -> dict[str, Any]:
    try:
        return _verify_freeze_impl(view_root)
    except (DatasetV5ViewError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {
            "ok": False,
            "schema_version": "dataset_v5_freeze_verification_report_v1.0.0",
            "errors": [str(exc)],
        }

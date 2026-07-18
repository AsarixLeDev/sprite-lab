"""Transactional Harvest-to-Dataset intake for conditioned Dataset-v5.

The callback is intentionally narrower than the conditioned builder.  It copies
an already verified Harvest artifact set into unique repository-managed work,
runs the ordinary Dataset intake contract there, and publishes only an atomic
opaque receipt after every byte and managed document has been revalidated.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import unicodedata
import uuid
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_v5.identity import decoded_rgba_sha256 as dataset_decoded_rgba_sha256
from spritelab.product_core import ProductStatus, ProjectContext, strict_json_dumps, strict_json_loads
from spritelab.product_features.conditioned_v5.identity import (
    ConditionedCodeIdentityError,
    conditioned_callback_runtime_inventory,
    conditioned_code_inventory,
    controlled_worker_dependency_roots,
    controlled_worker_environment,
    controlled_worker_executable,
    controlled_worker_launch_arguments,
    controlled_worker_runtime,
)
from spritelab.product_features.dataset.intake import DatasetIntakeService, discover_source_packs
from spritelab.product_features.dataset.managed import validate_managed_dataset_output
from spritelab.product_features.dataset.sheets import EXTRACTION_POLICY_VERSION
from spritelab.product_features.dataset.sidecar import save_grouping, save_pack_metadata, sidecar_path
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_RELATIVE_PATH,
    HarvestSource,
    load_trusted_catalog,
    trusted_catalog_identity,
)
from spritelab.product_features.harvest.certification import (
    BACKEND_AUDIT_REPORT_RELATIVE_PATH,
    BACKEND_CAPABILITIES_RELATIVE_PATH,
    BackendCapabilityEvidence,
    load_backend_capability_evidence,
)
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import (
    AcquiredFile,
    CertifiedBackendCapabilities,
    DatasetImportCancelled,
    DatasetImportDeadlineExceeded,
    DatasetImportRequest,
    DatasetImportResult,
    HarvestLimits,
)
from spritelab.training.campaign import stable_hash
from spritelab.utils.pinned_executable import (
    PinnedExecutableError,
    activate_windows_suspended_process,
    close_windows_handle,
    linux_parent_death_signal,
    pin_executable,
    verify_process_image,
)
from spritelab.utils.portable_paths import (
    canonical_portable_relative_path,
    portable_path_collision_key,
)
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    require_confined_path,
)
from spritelab.utils.write_confinement import (
    LINUX_LANDLOCK_STRATEGY,
    WINDOWS_PARENT_ANCHORS_STRATEGY,
    DirectoryIdentity,
    WriteConfinementError,
    WriteConfinementUnavailable,
    create_windows_bootstrap_untrusted_process,
    prepare_windows_untrusted_integrity_workspace,
    write_confinement_strategy,
)

HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
ARTIFACT_MANIFEST_SCHEMA = "spritelab.harvest.artifact-manifest.v1"
HARVEST_IMPORT_RECEIPT_SCHEMA = "spritelab.harvest.dataset-import-receipt.v1"
HARVEST_REQUEST_SCHEMA = "spritelab.harvest.request.v2"
HARVEST_AUTHORIZATION_SCHEMA = "spritelab.harvest.authorization-receipt.v2"
HARVEST_ACQUISITION_RECEIPT_SCHEMA = "spritelab.harvest.acquisition-receipt.v2"
MANAGED_IMPORT_RECEIPT_SCHEMA = "spritelab.dataset.conditioned-import-receipt.v2"
MANAGED_IMPORT_INVENTORY_SCHEMA = "spritelab.dataset.conditioned-import-inventory.v1"
DERIVED_SHEET_MANIFEST_SCHEMA = "spritelab.dataset.conditioned-derived-sheet-manifest.v1"
DERIVED_SHEET_FRAME_SCHEMA = "spritelab.dataset.conditioned-derived-sheet-frame.v1"
DERIVED_SHEET_RECIPE = {
    "schema_version": "spritelab.dataset.conditioned-derived-sheet-recipe.v1",
    "input_encoding": "decoded-rgba8",
    "crop_semantics": "left-top-inclusive-right-bottom-exclusive",
    "output_encoding": "png-rgba8-filter-none-zlib-level-9",
    "resize_or_resample": False,
    "augmentation": False,
}
DERIVED_SHEET_RECIPE_IDENTITY = stable_hash(DERIVED_SHEET_RECIPE)
_WORKER_MODULE_MANIFEST_SCHEMA = "spritelab.dataset.conditioned-worker-module-manifest.v2"
_OPERATION_CONTROL_SCHEMA = "spritelab.dataset.conditioned-operation-control.v1"
_MAX_OPERATION_SECONDS = 86_400.0
_OPERATION_POLL_SECONDS = 0.05

ALLOWED_LICENSES = frozenset({"cc0-1.0", "public-domain"})
REFERENCE_PATTERN = re.compile(r"^dataset\.[0-9a-f]{24}$")
_WORK_NAME_PATTERN = re.compile(r"^intake-[0-9a-f]{32}$")
RUN_ID_PATTERN = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,80}$")
KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_MAX_FILES = 5_000
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_DEPTH = 8
_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_LEGACY_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_DERIVED_PARENT_PIXELS = 16_777_216
_MAX_DERIVED_FRAMES = 5_000
_MAX_DERIVED_TOTAL_BYTES = 512 * 1024 * 1024
_LEGACY_REQUEST_SCHEMA = "spritelab.dataset.conditioned-legacy-intake-request.v1"
_LEGACY_RESPONSE_SCHEMA = "spritelab.dataset.conditioned-legacy-intake-response.v1"
_DERIVED_SHEET_FRAME_KEYS = frozenset(
    {
        "schema_version",
        "dataset_item_id",
        "parent_source_relative_path",
        "parent_source_raw_sha256",
        "parent_source_decoded_rgba_sha256",
        "crop_rectangle",
        "frame_index",
        "recipe_version",
        "recipe_identity",
        "decoded_rgba_sha256",
        "width",
        "height",
        "source_provenance_identity",
        "source_group_identity",
        "semantic_relative_path",
        "output_relative_path",
        "encoded_output_sha256",
        "encoded_output_byte_count",
        "derivation_identity",
        "record_identity",
        "source_derived_not_augmentation",
    }
)
_DERIVED_SHEET_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "recipe",
        "recipe_identity",
        "records",
        "record_count",
        "total_bytes",
        "portable_relative_paths",
        "raw_source_mutated",
        "source_derived_not_augmentation",
        "paths_exposed",
        "manifest_identity",
    }
)
_SHEET_EXTRACTION_KEYS = frozenset(
    {
        "source_item_id",
        "source_relative_path",
        "source_byte_sha256",
        "source_decoded_rgba_sha256",
        "crop_rectangle",
        "frame_index",
        "output_decoded_rgba_sha256",
        "extraction_policy_version",
        "source_sheet_modified",
    }
)
_MANAGED_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "dataset_reference",
        "request_identity",
        "callback_id",
        "callback_code_identity_sha256",
        "callback_code_inventory",
        "operation_control",
        "harvest",
        "handoff_document",
        "artifact_manifest",
        "source",
        "license",
        "managed",
        "accepted_count",
        "quarantined_count",
        "raw_harvest_mutated",
        "atomic_publication",
        "portable_relative_paths",
        "paths_exposed",
        "created_at",
        "receipt_identity",
    }
)
_MANAGED_RECEIPT_MANAGED_KEYS = frozenset(
    {
        "work_relative_path",
        "source_relative_path",
        "output_relative_path",
        "derived_root_relative_path",
        "source_inventory",
        "source_inventory_sha256",
        "output_inventory",
        "output_inventory_sha256",
        "derived_inventory",
        "derived_inventory_sha256",
        "derived_sheet_manifest",
        "derived_sheet_manifest_identity",
        "intake_result_identity",
        "accepted_relative_paths",
        "covered_source_relative_paths",
        "write_confinement",
        "worker_runtime",
        "sidecar_relative_path",
        "sidecar_identity",
        "sidecar_record_identity",
        "grouping_relative_path",
        "grouping_identity",
    }
)
_MANAGED_RECEIPT_HARVEST_KEYS = frozenset(
    {
        "run_id",
        "handoff_identity",
        "request_handoff_identity",
        "artifact_manifest_identity",
        "artifact_manifest_file_sha256",
        "artifact_set_identity",
        "provenance_identity",
        "source_evidence_binding_identity",
        "trusted_catalog_identity",
        "source_catalog_identity",
        "backend_capability_identity",
        "backend_capability_evidence_identity",
        "backend_certificate_identity",
        "backend_audit_report_sha256",
        "backend_audit_report_identity",
        "backend_capability_issued_at",
        "backend_capability_expires_at",
        "authorization_receipt_identity",
        "acquisition_receipt_identity",
        "request_document_identity",
    }
)
_WRITE_CONFINEMENT_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "strategy",
        "platform",
        "kernel_abi",
        "root_identity_sha256",
        "handled_access_fs",
        "allowed_access_fs",
        "no_new_privileges",
        "restricted_token",
        "integrity_level_rid",
        "mandatory_no_write_up",
        "workspace_integrity_level_rid",
        "startup_integrity_level_rid",
        "bootstrap_lowered_before_worker_import",
        "new_thread_integrity_level_rid",
        "raise_to_low_denied",
        "medium_probe_write_denied",
        "low_world_probe_write_denied",
        "untrusted_world_outside_guaranteed",
        "job_kill_on_close",
        "job_active_process_limit",
        "paths_exposed",
    }
)


class ConditionedIntakeError(ValueError):
    """A managed import cannot establish the required immutable bindings."""


class ConditionedWorkerTimeout(ConditionedIntakeError):
    """The isolated legacy worker exceeded its separately bounded runtime."""


CatalogLoader = Callable[[str | Path], Sequence[HarvestSource]]
CapabilityEvidenceLoader = Callable[[str | Path], BackendCapabilityEvidence | None]


def _never_cancel() -> bool:
    return False


def _normalize_operation_control(
    deadline_monotonic: float | None,
    cancel_requested: Callable[[], bool] | None,
    *,
    started_monotonic: float,
) -> tuple[float, Callable[[], bool]]:
    if cancel_requested is None:
        cancel_requested = _never_cancel
    if not callable(cancel_requested):
        raise ConditionedIntakeError("Dataset import cancellation control is invalid.")
    deadline = started_monotonic + 600.0 if deadline_monotonic is None else deadline_monotonic
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise ConditionedIntakeError("Dataset import deadline control is invalid.")
    normalized = float(deadline)
    if not math.isfinite(normalized) or normalized > started_monotonic + _MAX_OPERATION_SECONDS:
        raise ConditionedIntakeError("Dataset import deadline control is invalid.")
    return normalized, cancel_requested


def _check_operation_control(deadline_monotonic: float, cancel_requested: Callable[[], bool]) -> None:
    try:
        cancelled = cancel_requested()
    except Exception as exc:
        raise ConditionedIntakeError("Dataset import cancellation control failed closed.") from exc
    if type(cancelled) is not bool:
        raise ConditionedIntakeError("Dataset import cancellation control returned an invalid value.")
    if cancelled:
        raise DatasetImportCancelled("Conditioned Dataset import was cancelled.")
    if time.monotonic() >= deadline_monotonic:
        raise DatasetImportDeadlineExceeded("Conditioned Dataset import exceeded its whole-operation deadline.")


def _check_optional_operation_control(
    deadline_monotonic: float | None,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    if deadline_monotonic is None and cancel_requested is None:
        return
    if deadline_monotonic is None or cancel_requested is None:
        raise ConditionedIntakeError("Dataset import operation control is incomplete.")
    _check_operation_control(deadline_monotonic, cancel_requested)


def _remaining_operation_seconds(
    deadline_monotonic: float,
    cancel_requested: Callable[[], bool],
) -> float:
    _check_operation_control(deadline_monotonic, cancel_requested)
    return max(0.0, deadline_monotonic - time.monotonic())


@dataclass(frozen=True)
class _IntakeStorage:
    datasets_root: Path
    receipts_root: Path
    work_root: Path
    project_anchor: AnchoredDirectory
    datasets_anchor: AnchoredDirectory
    receipts_anchor: AnchoredDirectory
    work_root_anchor: AnchoredDirectory


@dataclass(frozen=True)
class _ManagedTransactionAnchors:
    """Exact already-held transaction roots used across Windows publication."""

    work: AnchoredDirectory
    source: AnchoredDirectory
    datasets: AnchoredDirectory
    output: AnchoredDirectory
    metadata: AnchoredDirectory
    derived: AnchoredDirectory


class ConditionedDatasetImportAdapter:
    """Harvest callback that publishes an opaque, DatasetIntake-backed receipt."""

    callback_id = "dataset.conditioned-intake"
    supports_operation_control = True
    code_identity_sha256 = stable_hash(
        {
            "adapter": callback_id,
            "contract": MANAGED_IMPORT_RECEIPT_SCHEMA,
            "revision": 2,
            "publication": "atomic-receipt-after-managed-validation",
        }
    )

    def __init__(
        self,
        project_root: str | Path,
        *,
        catalog_loader: CatalogLoader = load_trusted_catalog,
        capability_evidence_loader: CapabilityEvidenceLoader = load_backend_capability_evidence,
    ) -> None:
        try:
            root = Path(project_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ConditionedIntakeError("The trusted project root is unavailable.") from exc
        if not _safe_directory(root):
            raise ConditionedIntakeError("The trusted project root is linked, mounted, or unsafe.")
        self.project_root = root
        self.code_inventory = conditioned_code_inventory()
        self.code_identity_sha256 = str(self.code_inventory["inventory_sha256"])
        self.runtime_inventory = conditioned_callback_runtime_inventory(self.code_inventory)
        self.runtime_identity_sha256 = str(self.runtime_inventory["runtime_identity_sha256"])
        self._catalog_loader = catalog_loader
        self._capability_evidence_loader = capability_evidence_loader

    def import_harvest(
        self,
        request: DatasetImportRequest,
        *,
        idempotency_key: str,
        deadline_monotonic: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> DatasetImportResult:
        operation_started = time.monotonic()
        deadline, cancel_probe = _normalize_operation_control(
            deadline_monotonic,
            cancel_requested,
            started_monotonic=operation_started,
        )
        operation_control = {
            "schema_version": _OPERATION_CONTROL_SCHEMA,
            "deadline_monotonic": deadline,
            "started_monotonic": operation_started,
            "initial_budget_seconds": max(0.0, deadline - operation_started),
            "cancellation_probe_bound": True,
            "paths_exposed": False,
        }
        if not isinstance(request, DatasetImportRequest):
            raise ConditionedIntakeError("Harvest supplied an invalid Dataset import request.")
        if KEY_PATTERN.fullmatch(str(idempotency_key)) is None:
            raise ConditionedIntakeError("Dataset import requires a valid idempotency key.")

        verified = _verify_harvest_request(
            request,
            project_root=self.project_root,
            catalog_loader=self._catalog_loader,
            capability_evidence_loader=self._capability_evidence_loader,
        )
        project_root = self.project_root
        request_identity = stable_hash(
            {
                "schema_version": "spritelab.dataset.conditioned-import-request.v1",
                "callback_id": self.callback_id,
                "callback_code_identity_sha256": self.code_identity_sha256,
                "run_id": request.run_id,
                "idempotency_key": idempotency_key,
                "handoff_identity": verified["handoff_identity"],
                "artifact_manifest_identity": verified["artifact_manifest_identity"],
            }
        )
        reference = f"dataset.{request_identity[:24]}"
        receipt_name = f"{reference}.json"
        lock_name = f".{reference}.lock"

        with (
            _intake_storage(project_root) as storage,
            _receipt_lock(
                storage.receipts_anchor,
                lock_name,
                timeout=600.0,
                deadline_monotonic=deadline,
                cancel_requested=cancel_probe,
            ),
        ):
            work_root = storage.work_root
            if storage.receipts_anchor.lexists(receipt_name):
                receipt = _validate_managed_receipt(
                    _read_anchored_json(storage.receipts_anchor, receipt_name),
                    project_root,
                    reference,
                )
                if receipt["request_identity"] != request_identity:
                    raise ConditionedIntakeError("The Dataset import reference is bound to another request.")
                _require_receipt_verified_projection(receipt, verified)
                _check_operation_control(deadline, cancel_probe)
                _revalidate_managed_transaction(
                    project_root,
                    receipt,
                    work_root_anchor=storage.work_root_anchor,
                    verification=verified,
                    deadline_monotonic=deadline,
                    cancel_requested=cancel_probe,
                )
                _check_operation_control(deadline, cancel_probe)
                return DatasetImportResult(
                    reference,
                    int(receipt["accepted_count"]),
                    int(receipt["quarantined_count"]),
                )

            _check_operation_control(deadline, cancel_probe)
            try:
                write_strategy = write_confinement_strategy()
            except (WriteConfinementError, WriteConfinementUnavailable) as exc:
                raise ConditionedIntakeError(
                    "Conditioned Dataset intake is unavailable without an approved OS write boundary."
                ) from exc
            work_name, work_identity = storage.work_root_anchor.mkdir_unique("intake-")
            work = work_root / work_name
            workspace_stack = ExitStack()
            try:
                work_anchor = workspace_stack.enter_context(
                    storage.work_root_anchor.open_directory_immovable(work_name)
                )
                _require_created_anchor_identity(
                    work_identity,
                    work_anchor,
                    "The private intake workspace changed between creation and retained anchoring.",
                )
                if write_strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
                    try:
                        prepare_windows_untrusted_integrity_workspace(work)
                    except (WriteConfinementError, WriteConfinementUnavailable) as exc:
                        raise ConditionedIntakeError(
                            "Conditioned Dataset intake could not label its exact empty private workspace."
                        ) from exc
                    _require_created_anchor_identity(
                        work_identity,
                        work_anchor,
                        "The private intake workspace changed while its Windows integrity label was applied.",
                    )
                tmp_identity = work_anchor.mkdir("tmp", exist_ok=False)
                source_identity = work_anchor.mkdir("source", exist_ok=False)
                work_datasets_identity = work_anchor.mkdir("datasets", exist_ok=False)
                derived_identity = work_anchor.mkdir("derived_sprites", exist_ok=False)
                tmp_anchor = workspace_stack.enter_context(work_anchor.open_directory_immovable("tmp"))
                source_anchor = workspace_stack.enter_context(work_anchor.open_directory_immovable("source"))
                work_datasets_anchor = workspace_stack.enter_context(work_anchor.open_directory_immovable("datasets"))
                derived_anchor = workspace_stack.enter_context(work_anchor.open_directory_immovable("derived_sprites"))
                _require_created_anchor_identity(
                    tmp_identity,
                    tmp_anchor,
                    "The private intake temporary directory changed while it was anchored.",
                )
                _require_created_anchor_identity(
                    source_identity,
                    source_anchor,
                    "The private intake source directory changed while it was anchored.",
                )
                _require_created_anchor_identity(
                    work_datasets_identity,
                    work_datasets_anchor,
                    "The private intake Dataset directory changed while it was anchored.",
                )
                _require_created_anchor_identity(
                    derived_identity,
                    derived_anchor,
                    "The private direct-final derived-frame tree changed while it was anchored.",
                )
                output_identity = work_datasets_anchor.mkdir("managed", exist_ok=False)
                output_anchor = workspace_stack.enter_context(work_datasets_anchor.open_directory_immovable("managed"))
                _require_created_anchor_identity(
                    output_identity,
                    output_anchor,
                    "The private managed Dataset directory changed while it was anchored.",
                )
                held_legacy_anchors: list[AnchoredDirectory] = [
                    storage.project_anchor,
                    work_anchor,
                    tmp_anchor,
                    source_anchor,
                    work_datasets_anchor,
                    output_anchor,
                    derived_anchor,
                ]
                source_metadata_anchor: AnchoredDirectory | None = None
                if write_strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
                    source_metadata_identity = work_datasets_anchor.mkdir("source_metadata", exist_ok=False)
                    source_metadata_anchor = workspace_stack.enter_context(
                        work_datasets_anchor.open_directory_immovable("source_metadata")
                    )
                    _require_created_anchor_identity(
                        source_metadata_identity,
                        source_metadata_anchor,
                        "The private source-metadata directory changed while it was anchored.",
                    )
                    transactions_identity = source_metadata_anchor.mkdir(".transactions", exist_ok=False)
                    transactions_anchor = workspace_stack.enter_context(
                        source_metadata_anchor.open_directory_immovable(".transactions")
                    )
                    _require_created_anchor_identity(
                        transactions_identity,
                        transactions_anchor,
                        "The private metadata transaction directory changed while it was anchored.",
                    )
                    runs_identity = work_anchor.mkdir("runs", exist_ok=False)
                    runs_anchor = workspace_stack.enter_context(work_anchor.open_directory_immovable("runs"))
                    _require_created_anchor_identity(
                        runs_identity,
                        runs_anchor,
                        "The private runs directory changed while it was anchored.",
                    )
                    runs_v3_identity = runs_anchor.mkdir("v3", exist_ok=False)
                    runs_v3_anchor = workspace_stack.enter_context(runs_anchor.open_directory_immovable("v3"))
                    _require_created_anchor_identity(
                        runs_v3_identity,
                        runs_v3_anchor,
                        "The private v3 runs directory changed while it was anchored.",
                    )
                    held_legacy_anchors.extend(
                        (source_metadata_anchor, transactions_anchor, runs_anchor, runs_v3_anchor)
                    )
                harvest_anchor = workspace_stack.enter_context(
                    _open_descendant_directory(
                        storage.project_anchor,
                        Path(verified["artifacts_root"]),
                        project_root,
                    )
                )
            except BaseException:
                workspace_stack.close()
                raise
            source_root = work / "source"
            output_root = work / "datasets" / "managed"
            derived_root = work / "derived_sprites"
            stage = "copy"
            receipt_committed = False
            try:
                _check_operation_control(deadline, cancel_probe)
                _copy_verified_artifacts(
                    harvest_anchor,
                    source_anchor,
                    verified["artifact_manifest"],
                )
                _check_operation_control(deadline, cancel_probe)
                # A second source scan proves that the callback did not change or
                # replace raw Harvest bytes while publishing its independent copy.
                _verify_artifact_tree(
                    verified["artifacts_root"],
                    verified["artifact_manifest"],
                    anchor=harvest_anchor,
                )
                copied_manifest = _verify_artifact_tree(
                    source_root,
                    verified["artifact_manifest"],
                    anchor=source_anchor,
                )
                if copied_manifest != verified["artifact_manifest"]:
                    raise ConditionedIntakeError("The managed source copy does not reproduce the Harvest manifest.")
                _check_operation_control(deadline, cancel_probe)

                stage = "dataset_intake"
                legacy = _run_legacy_intake_boundary(
                    work=work,
                    source_root=source_root,
                    output_root=output_root,
                    derived_root=derived_root,
                    project_anchor=storage.project_anchor,
                    work_anchor=work_anchor,
                    source_anchor=source_anchor,
                    output_anchor=output_anchor,
                    derived_anchor=derived_anchor,
                    verified=verified,
                    run_id=request.run_id,
                    write_strategy=write_strategy,
                    held_anchors=tuple(held_legacy_anchors),
                    code_inventory=self.code_inventory,
                    deadline_monotonic=deadline,
                    cancel_requested=cancel_probe,
                )
                _check_operation_control(deadline, cancel_probe)
                accepted_paths = list(legacy["accepted_relative_paths"])
                derived_sheet_manifest = dict(legacy["derived_sheet_manifest"])
                covered_source_paths = list(legacy["covered_source_relative_paths"])

                stage = "derived_tree_validation"
                _require_created_anchor_identity(
                    derived_identity,
                    derived_anchor,
                    "The direct-final derived-frame tree changed before receipt publication.",
                )
                if derived_anchor.directory != derived_root:
                    raise ConditionedIntakeError("The derived-frame tree occupies an unexpected private root.")

                stage = "publication_validation"
                source_inventory = _inventory_from_anchor(source_anchor)
                output_inventory = _inventory_from_anchor(output_anchor)
                derived_inventory = _inventory_from_anchor(derived_anchor)
                sidecar_file = sidecar_path(work, str(legacy["pack_id"]))
                if write_strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
                    if source_metadata_anchor is None or sidecar_file.parent != source_metadata_anchor.directory:
                        raise ConditionedIntakeError(
                            "The controlled legacy sidecar escaped its held Windows metadata root."
                        )
                    sidecar_parent_anchor = source_metadata_anchor
                else:
                    sidecar_parent_anchor = workspace_stack.enter_context(
                        _open_descendant_directory(work_anchor, sidecar_file.parent, work)
                    )
                    source_metadata_anchor = sidecar_parent_anchor
                sidecar_identity = _stable_anchored_file_identity(sidecar_parent_anchor, sidecar_file.name)
                sidecar = _read_anchored_json(sidecar_parent_anchor, sidecar_file.name)
                if stable_hash(sidecar) != legacy["sidecar_record_identity"]:
                    raise ConditionedIntakeError("The controlled legacy sidecar identity differs from its exact file.")
                grouping_file = _grouping_file_for(work, source_root)
                if source_metadata_anchor is None or grouping_file.parent != source_metadata_anchor.directory:
                    raise ConditionedIntakeError("The controlled legacy grouping escaped its held metadata root.")
                grouping_parent_anchor = source_metadata_anchor
                grouping_identity = _stable_anchored_file_identity(grouping_parent_anchor, grouping_file.name)
                _check_operation_control(deadline, cancel_probe)
                fresh = _verify_harvest_request(
                    request,
                    project_root=self.project_root,
                    catalog_loader=self._catalog_loader,
                    capability_evidence_loader=self._capability_evidence_loader,
                )
                _check_operation_control(deadline, cancel_probe)
                _require_same_harvest_verification(verified, fresh)
                verified = fresh
                if conditioned_code_inventory() != self.code_inventory:
                    raise ConditionedIntakeError("Conditioned intake code changed during the controlled import.")
                source = verified["source"]
                license_record = verified["license"]
                artifact_count = int(verified["artifact_manifest"]["artifact_count"])
                receipt_without_identity = {
                    "schema_version": MANAGED_IMPORT_RECEIPT_SCHEMA,
                    "dataset_reference": reference,
                    "request_identity": request_identity,
                    "callback_id": self.callback_id,
                    "callback_code_identity_sha256": self.code_identity_sha256,
                    "callback_code_inventory": self.code_inventory,
                    "operation_control": operation_control,
                    "harvest": {
                        "run_id": request.run_id,
                        "handoff_identity": verified["handoff_identity"],
                        "request_handoff_identity": verified["request_handoff_identity"],
                        "artifact_manifest_identity": verified["artifact_manifest_identity"],
                        "artifact_manifest_file_sha256": verified["artifact_manifest_file_sha256"],
                        "artifact_set_identity": verified["artifact_manifest"]["artifact_set_identity"],
                        "provenance_identity": verified["handoff"]["provenance_identity"],
                        "source_evidence_binding_identity": verified["handoff"]["source_evidence_binding_identity"],
                        "trusted_catalog_identity": verified["trusted_catalog_identity"],
                        "source_catalog_identity": verified["source_catalog_identity"],
                        "backend_capability_identity": verified["backend_capability_identity"],
                        "backend_capability_evidence_identity": verified["backend_capability_evidence_identity"],
                        "backend_certificate_identity": verified["backend_certificate_identity"],
                        "backend_audit_report_sha256": verified["backend_audit_report_sha256"],
                        "backend_audit_report_identity": verified["backend_audit_report_identity"],
                        "backend_capability_issued_at": verified["backend_capability_issued_at"],
                        "backend_capability_expires_at": verified["backend_capability_expires_at"],
                        "authorization_receipt_identity": verified["authorization_receipt_identity"],
                        "acquisition_receipt_identity": verified["acquisition_receipt_identity"],
                        "request_document_identity": verified["request_document_identity"],
                    },
                    "handoff_document": verified["handoff"],
                    "artifact_manifest": verified["artifact_manifest"],
                    "source": source,
                    "license": license_record,
                    "managed": {
                        "work_relative_path": work.relative_to(project_root).as_posix(),
                        "source_relative_path": source_root.relative_to(project_root).as_posix(),
                        "output_relative_path": output_root.relative_to(project_root).as_posix(),
                        "derived_root_relative_path": derived_root.relative_to(project_root).as_posix(),
                        "source_inventory": source_inventory,
                        "source_inventory_sha256": stable_hash(source_inventory),
                        "output_inventory": output_inventory,
                        "output_inventory_sha256": stable_hash(output_inventory),
                        "derived_inventory": derived_inventory,
                        "derived_inventory_sha256": stable_hash(derived_inventory),
                        "derived_sheet_manifest": derived_sheet_manifest,
                        "derived_sheet_manifest_identity": derived_sheet_manifest["manifest_identity"],
                        "intake_result_identity": legacy["intake_result_identity"],
                        "accepted_relative_paths": accepted_paths,
                        "covered_source_relative_paths": covered_source_paths,
                        "write_confinement": legacy["write_confinement"],
                        "worker_runtime": legacy["worker_runtime"],
                        "sidecar_relative_path": sidecar_file.relative_to(project_root).as_posix(),
                        "sidecar_identity": sidecar_identity,
                        "sidecar_record_identity": stable_hash(sidecar),
                        "grouping_relative_path": grouping_file.relative_to(project_root).as_posix(),
                        "grouping_identity": grouping_identity,
                    },
                    "accepted_count": len(accepted_paths) + int(derived_sheet_manifest["record_count"]),
                    "quarantined_count": artifact_count - len(covered_source_paths),
                    "raw_harvest_mutated": False,
                    "atomic_publication": "receipt_pointer_after_validation",
                    "portable_relative_paths": True,
                    "paths_exposed": False,
                    "created_at": _now(),
                }
                receipt = {
                    **receipt_without_identity,
                    "receipt_identity": stable_hash(receipt_without_identity),
                }
                _validate_managed_receipt(receipt, project_root, reference)
                if source_metadata_anchor is None:
                    raise ConditionedIntakeError("The managed metadata root is not retained for publication.")
                transaction_anchors = _ManagedTransactionAnchors(
                    work=work_anchor,
                    source=source_anchor,
                    datasets=work_datasets_anchor,
                    output=output_anchor,
                    metadata=source_metadata_anchor,
                    derived=derived_anchor,
                )
                _check_operation_control(deadline, cancel_probe)
                _revalidate_managed_transaction(
                    project_root,
                    receipt,
                    work_root_anchor=storage.work_root_anchor,
                    verification=verified,
                    held_anchors=transaction_anchors,
                    deadline_monotonic=deadline,
                    cancel_requested=cancel_probe,
                )
                receipt_bytes = (strict_json_dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
                _check_operation_control(deadline, cancel_probe)
                try:
                    _publish_anchored_file_noreplace(
                        storage.receipts_anchor,
                        receipt_name,
                        receipt_bytes,
                        residue_prefix=f".rollback-{reference}-",
                    )
                except BaseException:
                    # ``AnchoredDirectory.rename`` syncs its directory after the
                    # no-replace rename.  If that sync (or an injected fault)
                    # raises after the namespace commit, the exact immutable
                    # receipt is still authoritative and must never be described
                    # by transaction-local evidence as unpublished.
                    receipt_committed = _anchored_regular_file_equals(
                        storage.receipts_anchor,
                        receipt_name,
                        receipt_bytes,
                    )
                    raise
                receipt_committed = True
                _check_operation_control(deadline, cancel_probe)
                # Load through the same boundary used by the conditioned builder.
                published = _load_managed_receipt_from_anchor(storage.receipts_anchor, project_root, reference)
                _require_receipt_verified_projection(published, verified)
                _revalidate_managed_transaction(
                    project_root,
                    published,
                    work_root_anchor=storage.work_root_anchor,
                    verification=verified,
                    held_anchors=transaction_anchors,
                    deadline_monotonic=deadline,
                    cancel_requested=cancel_probe,
                )
                _check_operation_control(deadline, cancel_probe)
                return DatasetImportResult(
                    reference,
                    int(published["accepted_count"]),
                    int(published["quarantined_count"]),
                )
            except BaseException as exc:
                if not receipt_committed:
                    failure = {
                        "schema_version": "spritelab.dataset.conditioned-import-failure.v1",
                        "request_identity": request_identity,
                        "operation_control": operation_control,
                        "stage": stage,
                        "error_code": _conditioned_failure_code(exc),
                        "published": False,
                        "retained_for_safe_inspection": True,
                        "paths_exposed": False,
                        "created_at": _now(),
                    }
                    try:
                        work_anchor.atomic_write_bytes(
                            "failure.json",
                            (strict_json_dumps(failure, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                        )
                    except (OSError, UnsafeFilesystemOperation):
                        pass
                if receipt_committed and not isinstance(exc, ConditionedIntakeError):
                    raise ConditionedIntakeError(
                        "The managed Dataset receipt committed, but callback confirmation failed."
                    ) from exc
                raise
            finally:
                workspace_stack.close()

    def load_managed_intake(self, dataset_reference: str) -> dict[str, Any]:
        """Revalidate one receipt under this adapter's exact trust context."""

        return load_managed_intake(
            self.project_root,
            dataset_reference,
            catalog_loader=self._catalog_loader,
            capability_evidence_loader=self._capability_evidence_loader,
        )


def _conditioned_failure_code(exc: BaseException) -> str:
    if isinstance(exc, DatasetImportCancelled):
        return "dataset_import_cancelled"
    if isinstance(exc, DatasetImportDeadlineExceeded):
        return "dataset_import_duration_exceeded"
    if isinstance(exc, ConditionedIntakeError):
        return "conditioned_intake_rejected"
    return "conditioned_intake_internal_failure"


def managed_intake_inventory(project_root: str | Path) -> list[dict[str, Any]]:
    """Passively inventory only atomically published managed intake receipts."""

    project = Path(project_root).resolve()
    root = project / "datasets" / "conditioned_intake_receipts"
    if not _safe_directory(root):
        return []
    results: list[dict[str, Any]] = []
    with AnchoredDirectory(root, project) as anchor:
        for name in anchor.names():
            if not name.startswith("dataset.") or not name.endswith(".json"):
                continue
            reference = name.removesuffix(".json")
            if REFERENCE_PATTERN.fullmatch(reference) is None:
                continue
            try:
                receipt = _validate_managed_receipt(
                    _read_anchored_json(anchor, name),
                    project,
                    reference,
                    require_current_code=False,
                )
            except ConditionedIntakeError:
                continue
            harvest = receipt.get("harvest") if isinstance(receipt.get("harvest"), Mapping) else {}
            source = receipt.get("source") if isinstance(receipt.get("source"), Mapping) else {}
            results.append(
                {
                    "dataset_reference": reference,
                    "harvest_run_id": str(harvest.get("run_id") or ""),
                    "source_id": str(source.get("source_id") or ""),
                    "source_title": str(source.get("title") or ""),
                    "accepted_count": _integer(receipt.get("accepted_count"), 0),
                    "quarantined_count": _integer(receipt.get("quarantined_count"), 0),
                    "status": "COMPLETE",
                    "paths_exposed": False,
                }
            )
    return results


def _require_receipt_verified_projection(
    receipt: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> None:
    harvest = receipt["harvest"]
    _require_receipt_harvest_bindings(harvest, verification)
    if (
        receipt["handoff_document"] != verification["handoff"]
        or receipt["artifact_manifest"] != verification["artifact_manifest"]
        or receipt["source"] != verification["source"]
        or receipt["license"] != verification["license"]
    ):
        raise ConditionedIntakeError("Managed Dataset receipt differs from its verified Harvest projection.")


@contextmanager
def _managed_transaction_anchor_scope(
    work_root_anchor: AnchoredDirectory,
    work_name: str,
    *,
    held_anchors: _ManagedTransactionAnchors | None,
) -> Iterator[_ManagedTransactionAnchors]:
    """Reuse retained publication handles or open an exact read-only set."""

    if held_anchors is not None:
        _verify_anchors(
            work_root_anchor,
            held_anchors.work,
            held_anchors.source,
            held_anchors.datasets,
            held_anchors.output,
            held_anchors.metadata,
            held_anchors.derived,
        )
        yield held_anchors
        _verify_anchors(
            work_root_anchor,
            held_anchors.work,
            held_anchors.source,
            held_anchors.datasets,
            held_anchors.output,
            held_anchors.metadata,
            held_anchors.derived,
        )
        return

    with ExitStack() as stack:
        work_anchor = stack.enter_context(work_root_anchor.open_directory_immovable(work_name))
        source_anchor = stack.enter_context(work_anchor.open_directory_immovable("source"))
        datasets_anchor = stack.enter_context(work_anchor.open_directory_immovable("datasets"))
        output_anchor = stack.enter_context(datasets_anchor.open_directory_immovable("managed"))
        metadata_anchor = stack.enter_context(datasets_anchor.open_directory_immovable("source_metadata"))
        derived_anchor = stack.enter_context(work_anchor.open_directory_immovable("derived_sprites"))
        opened = _ManagedTransactionAnchors(
            work=work_anchor,
            source=source_anchor,
            datasets=datasets_anchor,
            output=output_anchor,
            metadata=metadata_anchor,
            derived=derived_anchor,
        )
        _verify_anchors(
            work_root_anchor,
            opened.work,
            opened.source,
            opened.datasets,
            opened.output,
            opened.metadata,
            opened.derived,
        )
        yield opened
        _verify_anchors(
            work_root_anchor,
            opened.work,
            opened.source,
            opened.datasets,
            opened.output,
            opened.metadata,
            opened.derived,
        )


def _revalidate_managed_transaction(
    project: Path,
    receipt: Mapping[str, Any],
    *,
    work_root_anchor: AnchoredDirectory,
    verification: Mapping[str, Any],
    held_anchors: _ManagedTransactionAnchors | None = None,
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Revalidate every managed byte without requiring the downstream Harvest receipt."""

    _check_optional_operation_control(deadline_monotonic, cancel_requested)
    managed = receipt["managed"]
    work_relative = _canonical_relative(str(managed["work_relative_path"]))
    work_parts = PurePosixPath(work_relative).parts
    if (
        len(work_parts) != 3
        or work_parts[:2] != ("datasets", "conditioned_intake_work")
        or _WORK_NAME_PATTERN.fullmatch(work_parts[2]) is None
        or work_root_anchor.directory != project / "datasets" / "conditioned_intake_work"
    ):
        raise ConditionedIntakeError("Managed Dataset transaction root is outside its held intake namespace.")
    work_name = work_parts[2]
    work = project.joinpath(*work_parts)
    source_root = work / "source"
    output_root = work / "datasets" / "managed"
    derived_root = work / "derived_sprites"
    if (
        _project_relative_path(project, managed["source_relative_path"], expected="directory") != source_root
        or _project_relative_path(project, managed["output_relative_path"], expected="directory") != output_root
        or _project_relative_path(project, managed["derived_root_relative_path"], expected="directory") != derived_root
    ):
        raise ConditionedIntakeError("Managed Dataset roots differ from their receipt-bound transaction root.")
    context = ProjectContext(project_root=project, config={}, runs_directory=project / "runs" / "v3")
    try:
        with _managed_transaction_anchor_scope(
            work_root_anchor,
            work_name,
            held_anchors=held_anchors,
        ) as transaction_anchors:
            work_anchor = transaction_anchors.work
            source_anchor = transaction_anchors.source
            datasets_anchor = transaction_anchors.datasets
            output_anchor = transaction_anchors.output
            metadata_anchor = transaction_anchors.metadata
            derived_anchor = transaction_anchors.derived
            expected_directories = (
                (work_anchor, work),
                (source_anchor, source_root),
                (datasets_anchor, work / "datasets"),
                (output_anchor, output_root),
                (metadata_anchor, work / "datasets" / "source_metadata"),
                (derived_anchor, derived_root),
            )
            if any(anchor.directory != expected for anchor, expected in expected_directories):
                raise ConditionedIntakeError("Managed transaction anchors differ from their receipt-bound roots.")
            _verify_anchors(
                work_root_anchor,
                work_anchor,
                source_anchor,
                datasets_anchor,
                output_anchor,
                metadata_anchor,
                derived_anchor,
            )
            validate_managed_dataset_output(output_root, context=context, require_datasets_root=True)
            _check_optional_operation_control(deadline_monotonic, cancel_requested)
            _verify_anchors(source_anchor, output_anchor, derived_anchor)
            if _inventory_from_anchor(source_anchor) != managed["source_inventory"]:
                raise ConditionedIntakeError("Managed intake source bytes changed after publication.")
            if _inventory_from_anchor(output_anchor) != managed["output_inventory"]:
                raise ConditionedIntakeError("Managed Dataset intake output changed after publication.")
            if _inventory_from_anchor(derived_anchor) != managed["derived_inventory"]:
                raise ConditionedIntakeError("Managed derived-frame bytes changed after publication.")
            _validate_persisted_intake_result(
                output_anchor,
                str(managed["intake_result_identity"]),
            )
            _validate_persisted_confinement_binding(
                managed["write_confinement"],
                work_anchor,
            )
            _check_optional_operation_control(deadline_monotonic, cancel_requested)

            metadata_relative = PurePosixPath(work_relative) / "datasets" / "source_metadata"
            sidecar_relative = PurePosixPath(_canonical_relative(str(managed["sidecar_relative_path"])))
            grouping_relative = PurePosixPath(_canonical_relative(str(managed["grouping_relative_path"])))
            if sidecar_relative.parent != metadata_relative or grouping_relative.parent != metadata_relative:
                raise ConditionedIntakeError("Managed Dataset metadata escaped its held transaction root.")
            sidecar_name = sidecar_relative.name
            grouping_name = grouping_relative.name
            if _stable_anchored_file_identity(metadata_anchor, sidecar_name) != managed["sidecar_identity"]:
                raise ConditionedIntakeError("Managed intake provenance sidecar changed after publication.")
            if _stable_anchored_file_identity(metadata_anchor, grouping_name) != managed["grouping_identity"]:
                raise ConditionedIntakeError("Managed intake pack grouping changed after publication.")
            sidecar = _read_anchored_json(metadata_anchor, sidecar_name)
            if stable_hash(sidecar) != managed["sidecar_record_identity"]:
                raise ConditionedIntakeError("Managed intake provenance sidecar identity is inconsistent.")

            manifest = verification["artifact_manifest"]
            if _verify_artifact_tree(source_root, manifest, anchor=source_anchor) != manifest:
                raise ConditionedIntakeError("Managed source files no longer reproduce the Harvest manifest.")
            accepted = _accepted_source_paths(
                output_root,
                source_root,
                manifest,
                output_anchor=output_anchor,
                source_anchor=source_anchor,
            )
            if accepted != managed["accepted_relative_paths"]:
                raise ConditionedIntakeError("Managed intake dispositions changed after publication.")
            derived_sheet_records = _validate_derived_sheet_tree(
                managed["derived_sheet_manifest"],
                output_anchor=output_anchor,
                source_anchor=source_anchor,
                derived_anchor=derived_anchor,
                artifact_manifest=manifest,
                source=verification["source"],
                license_record=verification["license"],
                run_id=str(verification["handoff"]["run_id"]),
                deadline_monotonic=deadline_monotonic,
                cancel_requested=cancel_requested,
            )
            covered_source_paths = sorted(
                {
                    *accepted,
                    *(str(record["parent_source_relative_path"]) for record in derived_sheet_records),
                }
            )
            if covered_source_paths != managed["covered_source_relative_paths"]:
                raise ConditionedIntakeError("Managed intake source coverage changed after publication.")
            _verify_anchors(work_anchor, source_anchor, output_anchor, metadata_anchor, derived_anchor)
    except ConditionedIntakeError:
        raise
    except (OSError, UnsafeFilesystemOperation, ValueError, TypeError) as exc:
        raise ConditionedIntakeError("Managed Dataset transaction bytes are unavailable or unsafe.") from exc
    return {
        "source_root": source_root,
        "output_root": output_root,
        "derived_root": derived_root,
        "accepted_relative_paths": accepted,
        "derived_sheet_records": derived_sheet_records,
        "covered_source_relative_paths": covered_source_paths,
    }


def load_managed_intake(
    project_root: str | Path,
    dataset_reference: str,
    *,
    catalog_loader: CatalogLoader = load_trusted_catalog,
    capability_evidence_loader: CapabilityEvidenceLoader = load_backend_capability_evidence,
) -> dict[str, Any]:
    """Revalidate a published intake, its managed bytes, and original handoff."""

    try:
        return _load_managed_intake(
            project_root,
            dataset_reference,
            catalog_loader=catalog_loader,
            capability_evidence_loader=capability_evidence_loader,
        )
    except ConditionedIntakeError:
        raise
    except (OSError, UnsafeFilesystemOperation, ValueError, TypeError) as exc:
        raise ConditionedIntakeError("Managed Dataset intake is unavailable, changed, or unsafe.") from exc


def _load_managed_intake(
    project_root: str | Path,
    dataset_reference: str,
    *,
    catalog_loader: CatalogLoader,
    capability_evidence_loader: CapabilityEvidenceLoader,
) -> dict[str, Any]:
    """Implementation boundary kept separate so filesystem failures normalize."""

    project = Path(project_root).resolve(strict=True)
    with AnchoredDirectory(project, project) as project_anchor:
        receipts_root = project / "datasets" / "conditioned_intake_receipts"
        with _open_descendant_directory(
            project_anchor,
            receipts_root,
            project,
            immovable=True,
        ) as receipt_anchor:
            receipt = _load_managed_receipt_from_anchor(receipt_anchor, project, dataset_reference)

    harvest = receipt["harvest"]
    run_id = str(harvest["run_id"])
    manifest = receipt["artifact_manifest"]
    run_root = _project_relative_path(project, f"harvest_runs/{run_id}", expected="directory")
    verification = _verify_harvest_request(
        DatasetImportRequest(run_id, run_root / "artifacts", receipt["handoff_document"], manifest),
        project_root=project,
        catalog_loader=catalog_loader,
        capability_evidence_loader=capability_evidence_loader,
    )
    _require_receipt_verified_projection(receipt, verification)
    handoff = verification["handoff"]
    work_root = project / "datasets" / "conditioned_intake_work"
    with AnchoredDirectory(project, project) as project_anchor:
        with _open_descendant_directory(
            project_anchor,
            work_root,
            project,
            immovable=True,
        ) as work_root_anchor:
            transaction = _revalidate_managed_transaction(
                project,
                receipt,
                work_root_anchor=work_root_anchor,
                verification=verification,
            )
    source_root = transaction["source_root"]
    output_root = transaction["output_root"]
    derived_root = transaction["derived_root"]
    accepted = transaction["accepted_relative_paths"]
    derived_sheet_records = transaction["derived_sheet_records"]
    covered_source_paths = transaction["covered_source_relative_paths"]
    managed = receipt["managed"]

    with AnchoredDirectory(project, project) as project_anchor:
        harvest_receipt = _read_project_anchored_json(
            project_anchor,
            run_root / "dataset_import_receipt.json",
            project,
        )
    if (
        harvest_receipt.get("schema_version") != HARVEST_IMPORT_RECEIPT_SCHEMA
        or harvest_receipt.get("run_id") != run_id
        or harvest_receipt.get("dataset_reference") != dataset_reference
        or harvest_receipt.get("callback_id") != receipt["callback_id"]
        or harvest_receipt.get("callback_code_identity_sha256") != receipt["callback_code_identity_sha256"]
        or harvest_receipt.get("artifact_manifest_identity") != harvest["artifact_manifest_identity"]
        or harvest_receipt.get("accepted_count") != receipt["accepted_count"]
        or harvest_receipt.get("quarantined_count") != receipt["quarantined_count"]
        or harvest_receipt.get("paths_exposed") is not False
    ):
        raise ConditionedIntakeError("The Harvest Dataset import receipt is missing or inconsistent.")
    harvest_receipt_identity = stable_hash(harvest_receipt)

    source = verification["source"]
    license_record = verification["license"]
    return {
        "dataset_reference": dataset_reference,
        "run_id": run_id,
        "run_root": run_root,
        "artifacts_root": source_root,
        "managed_output_root": output_root,
        "derived_root": derived_root,
        "handoff": handoff,
        "handoff_identity": harvest["handoff_identity"],
        "trusted_catalog_identity": harvest["trusted_catalog_identity"],
        "source_catalog_identity": harvest["source_catalog_identity"],
        "backend_capability_identity": harvest["backend_capability_identity"],
        "backend_capability_evidence_identity": harvest["backend_capability_evidence_identity"],
        "backend_certificate_identity": harvest["backend_certificate_identity"],
        "backend_audit_report_sha256": harvest["backend_audit_report_sha256"],
        "backend_audit_report_identity": harvest["backend_audit_report_identity"],
        "backend_capability_issued_at": harvest["backend_capability_issued_at"],
        "backend_capability_expires_at": harvest["backend_capability_expires_at"],
        "authorization_receipt_identity": harvest["authorization_receipt_identity"],
        "acquisition_receipt_identity": harvest["acquisition_receipt_identity"],
        "harvest_import_receipt_identity": harvest_receipt_identity,
        "managed_intake_receipt_identity": receipt["receipt_identity"],
        "managed_output_inventory_sha256": managed["output_inventory_sha256"],
        "managed_source_inventory_sha256": managed["source_inventory_sha256"],
        "managed_derived_inventory_sha256": managed["derived_inventory_sha256"],
        "derived_sheet_manifest_identity": managed["derived_sheet_manifest_identity"],
        "derived_sheet_manifest": managed["derived_sheet_manifest"],
        "derived_sheet_records": derived_sheet_records,
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": harvest["artifact_manifest_file_sha256"],
        "source_id": source["source_id"],
        "source_title": source["title"],
        "creator": source["creator"],
        "license_id": str(license_record["identifier"]).casefold(),
        "license_evidence": license_record,
        "artifact_count": int(manifest["artifact_count"]),
        "accepted_relative_paths": accepted,
        "covered_source_relative_paths": covered_source_paths,
    }


def _verify_harvest_request(
    request: DatasetImportRequest,
    *,
    project_root: Path,
    catalog_loader: CatalogLoader,
    capability_evidence_loader: CapabilityEvidenceLoader,
) -> dict[str, Any]:
    try:
        with AnchoredDirectory(project_root, project_root) as project_anchor:
            return _verify_harvest_request_anchored(
                request,
                project_root=project_root,
                project_anchor=project_anchor,
                catalog_loader=catalog_loader,
                capability_evidence_loader=capability_evidence_loader,
            )
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Harvest trust storage changed while it was being verified.") from exc


def _verify_harvest_request_anchored(
    request: DatasetImportRequest,
    *,
    project_root: Path,
    project_anchor: AnchoredDirectory,
    catalog_loader: CatalogLoader,
    capability_evidence_loader: CapabilityEvidenceLoader,
) -> dict[str, Any]:
    if RUN_ID_PATTERN.fullmatch(str(request.run_id)) is None:
        raise ConditionedIntakeError("Harvest Dataset import run identity is invalid.")
    expected_harvest_root = project_root / "harvest_runs"
    expected_run_root = expected_harvest_root / request.run_id
    expected_artifacts = expected_run_root / "artifacts"
    artifacts = Path(os.path.abspath(os.path.expanduser(os.fspath(request.artifacts_directory))))
    if artifacts != expected_artifacts:
        raise ConditionedIntakeError("Harvest artifacts are outside the expected managed run boundary.")
    try:
        require_confined_path(artifacts, project_root)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Harvest artifacts are outside the project boundary.") from exc
    for directory in (project_root, expected_harvest_root, expected_run_root, artifacts):
        if not _safe_directory(directory):
            raise ConditionedIntakeError("Harvest Dataset import crosses an unsafe directory boundary.")
    run_root = expected_run_root

    projected = dict(request.handoff)
    request_handoff_identity = stable_hash(projected)
    projected.pop("dataset_import_available", None)
    disk_handoff = _read_project_anchored_json(project_anchor, run_root / "handoff.json", project_root)
    if projected != disk_handoff:
        raise ConditionedIntakeError("Harvest handoff projection disagrees with its durable document.")
    manifest = dict(request.artifact_manifest)
    disk_manifest_path = run_root / "artifact_manifest.json"
    disk_manifest_bytes = _read_project_anchored_file(project_anchor, disk_manifest_path, project_root, _MAX_JSON_BYTES)
    try:
        disk_manifest_value = strict_json_loads(disk_manifest_bytes)
    except ValueError as exc:
        raise ConditionedIntakeError("Harvest artifact manifest is invalid.") from exc
    if not isinstance(disk_manifest_value, Mapping) or manifest != dict(disk_manifest_value):
        raise ConditionedIntakeError("Harvest artifact manifest projection disagrees with its durable document.")
    _validate_handoff_document(request.run_id, disk_handoff, manifest)

    catalog_path = project_root / TRUSTED_CATALOG_RELATIVE_PATH
    catalog_bytes = _read_project_anchored_file(project_anchor, catalog_path, project_root, _MAX_JSON_BYTES)
    try:
        catalog_document = strict_json_loads(catalog_bytes)
    except ValueError as exc:
        raise ConditionedIntakeError("The current trusted Harvest catalog is invalid.") from exc
    if not isinstance(catalog_document, Mapping):
        raise ConditionedIntakeError("The current trusted Harvest catalog is invalid.")
    try:
        catalog = tuple(catalog_loader(project_root))
    except (OSError, TypeError, ValueError) as exc:
        raise ConditionedIntakeError("The current trusted Harvest catalog is unavailable or invalid.") from exc
    if not catalog or any(not isinstance(item, HarvestSource) for item in catalog):
        raise ConditionedIntakeError("The current trusted Harvest catalog is unavailable or invalid.")
    source_matches = [item for item in catalog if item.source_id == disk_handoff.get("source_id")]
    if len(source_matches) != 1:
        raise ConditionedIntakeError("The Harvest source is absent or duplicated in the current trusted catalog.")
    trusted_source = source_matches[0]
    try:
        trusted_source.evidence_binding.validate(trusted_source.source_page, trusted_source.license_evidence_url)
        source_snapshot = trusted_source.to_public_dict()
        catalog_identity = trusted_catalog_identity(catalog)
    except ValueError as exc:
        raise ConditionedIntakeError("The current Harvest source evidence attestation is invalid or expired.") from exc
    if source_snapshot != disk_handoff.get("source"):
        raise ConditionedIntakeError("The Harvest handoff is stale against the current trusted source catalog.")
    if (
        _read_project_anchored_file(project_anchor, catalog_path, project_root, _MAX_JSON_BYTES) != catalog_bytes
        or catalog_document.get("catalog_identity") != catalog_identity
    ):
        raise ConditionedIntakeError("The trusted Harvest catalog changed while it was loaded.")

    certificate_path = project_root / BACKEND_CAPABILITIES_RELATIVE_PATH
    audit_report_path = project_root / BACKEND_AUDIT_REPORT_RELATIVE_PATH
    certificate_bytes = _read_project_anchored_file(project_anchor, certificate_path, project_root, _MAX_JSON_BYTES)
    audit_report_bytes = _read_project_anchored_file(project_anchor, audit_report_path, project_root, _MAX_JSON_BYTES)
    try:
        certificate_document = strict_json_loads(certificate_bytes)
    except ValueError as exc:
        raise ConditionedIntakeError("The current Harvest backend certificate is invalid.") from exc
    if not isinstance(certificate_document, Mapping):
        raise ConditionedIntakeError("The current Harvest backend certificate is invalid.")
    try:
        capability_evidence = capability_evidence_loader(project_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ConditionedIntakeError("The current Harvest backend certificate is unavailable or invalid.") from exc
    if not isinstance(capability_evidence, BackendCapabilityEvidence):
        raise ConditionedIntakeError("A current independently certified Harvest backend is required.")
    capabilities = capability_evidence.capabilities
    backend_evidence = {**capability_evidence.to_dict(), "evidence_identity": capability_evidence.identity}
    if (
        _read_project_anchored_file(project_anchor, certificate_path, project_root, _MAX_JSON_BYTES)
        != certificate_bytes
        or _read_project_anchored_file(project_anchor, audit_report_path, project_root, _MAX_JSON_BYTES)
        != audit_report_bytes
        or certificate_document.get("certificate_identity") != capability_evidence.certificate_identity
        or hashlib.sha256(audit_report_bytes).hexdigest() != capability_evidence.audit_report_sha256
    ):
        raise ConditionedIntakeError("The Harvest backend certificate changed while it was loaded.")

    request_document = _read_project_anchored_json(project_anchor, run_root / "request.json", project_root)
    authorization = _read_project_anchored_json(project_anchor, run_root / "authorization_receipt.json", project_root)
    acquisition = _read_project_anchored_json(project_anchor, run_root / "acquisition_receipt.json", project_root)
    _validate_harvest_receipts(
        request.run_id,
        request_document=request_document,
        authorization=authorization,
        acquisition=acquisition,
        source=trusted_source,
        capabilities=capabilities,
        backend_evidence=backend_evidence,
        handoff=disk_handoff,
        manifest=manifest,
    )
    with _open_descendant_directory(project_anchor, artifacts, project_root) as artifacts_anchor:
        if _verify_artifact_tree(artifacts, manifest, anchor=artifacts_anchor) != manifest:
            raise ConditionedIntakeError("Harvest artifact bytes do not reproduce their exact manifest.")
    return {
        "project_root": project_root,
        "run_root": run_root,
        "artifacts_root": artifacts,
        "handoff": disk_handoff,
        "handoff_identity": stable_hash(disk_handoff),
        "request_handoff_identity": request_handoff_identity,
        "artifact_manifest": manifest,
        "artifact_manifest_identity": stable_hash(manifest),
        "artifact_manifest_file_sha256": hashlib.sha256(disk_manifest_bytes).hexdigest(),
        "trusted_catalog_identity": catalog_identity,
        "source_catalog_identity": trusted_source.catalog_identity,
        "backend_capability_identity": capabilities.identity,
        "backend_capability_evidence_identity": capability_evidence.identity,
        "backend_certificate_identity": capability_evidence.certificate_identity,
        "backend_audit_report_sha256": capability_evidence.audit_report_sha256,
        "backend_audit_report_identity": capability_evidence.audit_report_identity,
        "backend_capability_issued_at": capability_evidence.issued_at,
        "backend_capability_expires_at": capability_evidence.expires_at,
        "request_document_identity": stable_hash(request_document),
        "authorization_receipt_identity": stable_hash(authorization),
        "acquisition_receipt_identity": str(acquisition["acquisition_receipt_identity"]),
        "source": dict(disk_handoff["source"]),
        "license": dict(disk_handoff["license"]),
    }


def _validate_harvest_receipts(
    run_id: str,
    *,
    request_document: Mapping[str, Any],
    authorization: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    source: HarvestSource,
    capabilities: CertifiedBackendCapabilities,
    backend_evidence: Mapping[str, Any],
    handoff: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected_capabilities = {**capabilities.to_dict(), "capability_identity": capabilities.identity}
    authorization_limits = authorization.get("limits")
    if not isinstance(authorization_limits, Mapping):
        raise ConditionedIntakeError("Harvest authorization limits evidence is missing.")
    limits_identity = authorization_limits.get("limits_identity")
    if (
        request_document.get("schema_version") != HARVEST_REQUEST_SCHEMA
        or request_document.get("run_id") != run_id
        or request_document.get("source_id") != source.source_id
        or request_document.get("source_catalog_identity") != source.catalog_identity
        or request_document.get("backend_capability_identity") != capabilities.identity
        or request_document.get("backend_capability_evidence_identity") != backend_evidence["evidence_identity"]
        or request_document.get("backend_capability_certificate_identity") != backend_evidence["certificate_identity"]
        or request_document.get("backend_capability_audit_report_sha256") != backend_evidence["audit_report_sha256"]
        or request_document.get("backend_capability_audit_report_identity") != backend_evidence["audit_report_identity"]
        or request_document.get("backend_capability_issued_at") != backend_evidence["issued_at"]
        or request_document.get("backend_capability_expires_at") != backend_evidence["expires_at"]
        or request_document.get("limits_identity") != limits_identity
        or request_document.get("browser_paths_accepted") is not False
    ):
        raise ConditionedIntakeError("Harvest request evidence is stale or inconsistent.")
    authorizations = authorization.get("authorizations")
    if (
        authorization.get("schema_version") != HARVEST_AUTHORIZATION_SCHEMA
        or authorization.get("run_id") != run_id
        or authorization.get("source") != source.to_public_dict()
        or authorization.get("backend_capabilities") != expected_capabilities
        or authorization.get("backend_capability_evidence") != backend_evidence
        or not isinstance(authorizations, Mapping)
        or set(authorizations) != {"explicit_action", "zero_cost", "permissive_license", "existing_inventory_reviewed"}
        or any(value is not True for value in authorizations.values())
        or authorization.get("network_actions_before_receipt") != 0
        or authorization.get("paths_exposed") is not False
    ):
        raise ConditionedIntakeError("Harvest authorization evidence is stale or inconsistent.")
    acquisition_identity = acquisition.get("acquisition_receipt_identity")
    acquisition_payload = dict(acquisition)
    acquisition_payload.pop("acquisition_receipt_identity", None)
    _validate_acquisition_kind_binding(acquisition, handoff, manifest)
    if (
        acquisition.get("schema_version") != HARVEST_ACQUISITION_RECEIPT_SCHEMA
        or acquisition.get("source_id") != source.source_id
        or acquisition.get("source_catalog_identity") != source.catalog_identity
        or acquisition.get("source_evidence_binding_identity") != source.evidence_binding.identity
        or acquisition.get("backend_capabilities") != expected_capabilities
        or acquisition.get("backend_capability_evidence") != backend_evidence
        or acquisition.get("backend_capability_evidence_identity") != backend_evidence["evidence_identity"]
        or acquisition.get("limits") != authorization_limits
        or acquisition.get("artifact_manifest_identity") != stable_hash(dict(manifest))
        or SHA256_PATTERN.fullmatch(str(acquisition_identity or "")) is None
        or stable_hash(acquisition_payload) != acquisition_identity
        or handoff.get("acquisition_receipt_identity") != acquisition_identity
        or handoff.get("backend_capability_identity") != capabilities.identity
        or handoff.get("backend_capability_evidence") != backend_evidence
        or handoff.get("backend_capability_evidence_identity") != backend_evidence["evidence_identity"]
        or handoff.get("limits_identity") != limits_identity
    ):
        raise ConditionedIntakeError("Harvest acquisition or backend-certificate evidence is inconsistent.")


def _validate_acquisition_kind_binding(
    acquisition: Mapping[str, Any],
    handoff: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    response_kind = acquisition.get("response_kind")
    direct = acquisition.get("direct_image_derivation")
    if (
        response_kind not in {"archive", "direct_static_image"}
        or SHA256_PATTERN.fullmatch(str(acquisition.get("actual_response_sha256") or "")) is None
        or handoff.get("acquisition_kind") != response_kind
        or handoff.get("direct_image_derivation") != direct
    ):
        raise ConditionedIntakeError("Harvest acquisition kind or raw-response binding is invalid.")
    if response_kind == "archive":
        if direct is not None:
            raise ConditionedIntakeError("Harvest archive acquisition has unexpected direct-image derivation evidence.")
        return
    expected_keys = {
        "schema_version",
        "kind",
        "source_format",
        "source_mime_type",
        "raw_byte_count",
        "raw_sha256",
        "frame_count",
        "width",
        "height",
        "decoded_rgba_sha256",
        "output_relative_path",
        "output_mime_type",
        "output_byte_count",
        "output_sha256",
        "recipe_identity",
        "derived",
        "source_bytes_modified",
    }
    files = manifest.get("files")
    if not isinstance(direct, Mapping) or set(direct) != expected_keys or not isinstance(files, list):
        raise ConditionedIntakeError("Harvest direct-image derivation evidence is invalid.")
    output_relative = str(direct.get("output_relative_path") or "")
    matching = [row for row in files if isinstance(row, Mapping) and row.get("relative_path") == output_relative]
    if (
        direct.get("schema_version") != "spritelab.harvest.direct-image-derivation.v1"
        or direct.get("kind") != "direct_static_image"
        or direct.get("raw_sha256") != acquisition.get("actual_response_sha256")
        or direct.get("raw_byte_count") != acquisition.get("response_bytes")
        or direct.get("frame_count") != 1
        or type(direct.get("width")) is not int
        or type(direct.get("height")) is not int
        or not 0 < int(direct["width"]) * int(direct["height"]) <= 16_777_216
        or SHA256_PATTERN.fullmatch(str(direct.get("decoded_rgba_sha256") or "")) is None
        or direct.get("output_mime_type") != "image/png"
        or type(direct.get("output_byte_count")) is not int
        or int(direct["output_byte_count"]) <= 0
        or SHA256_PATTERN.fullmatch(str(direct.get("output_sha256") or "")) is None
        or direct.get("recipe_identity") != "spritelab.harvest.direct-static-image-to-png.v1"
        or direct.get("source_bytes_modified") is not False
        or len(matching) != 1
        or matching[0].get("actual_sha256") != direct.get("output_sha256")
        or matching[0].get("byte_count") != direct.get("output_byte_count")
    ):
        raise ConditionedIntakeError("Harvest direct-image derivation evidence is invalid.")


def _require_same_harvest_verification(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    keys = (
        "handoff_identity",
        "request_handoff_identity",
        "artifact_manifest_identity",
        "artifact_manifest_file_sha256",
        "trusted_catalog_identity",
        "source_catalog_identity",
        "backend_capability_identity",
        "backend_capability_evidence_identity",
        "backend_certificate_identity",
        "backend_audit_report_sha256",
        "backend_audit_report_identity",
        "backend_capability_issued_at",
        "backend_capability_expires_at",
        "request_document_identity",
        "authorization_receipt_identity",
        "acquisition_receipt_identity",
    )
    if any(previous.get(key) != current.get(key) for key in keys):
        raise ConditionedIntakeError("Harvest trust or artifact evidence changed during Dataset import.")


def _require_receipt_harvest_bindings(receipt: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    bindings = {
        "handoff_identity": "handoff_identity",
        "request_handoff_identity": "request_handoff_identity",
        "artifact_manifest_identity": "artifact_manifest_identity",
        "artifact_manifest_file_sha256": "artifact_manifest_file_sha256",
        "trusted_catalog_identity": "trusted_catalog_identity",
        "source_catalog_identity": "source_catalog_identity",
        "backend_capability_identity": "backend_capability_identity",
        "backend_capability_evidence_identity": "backend_capability_evidence_identity",
        "backend_certificate_identity": "backend_certificate_identity",
        "backend_audit_report_sha256": "backend_audit_report_sha256",
        "backend_audit_report_identity": "backend_audit_report_identity",
        "backend_capability_issued_at": "backend_capability_issued_at",
        "backend_capability_expires_at": "backend_capability_expires_at",
        "request_document_identity": "request_document_identity",
        "authorization_receipt_identity": "authorization_receipt_identity",
        "acquisition_receipt_identity": "acquisition_receipt_identity",
    }
    if any(receipt.get(receipt_key) != current.get(current_key) for receipt_key, current_key in bindings.items()):
        raise ConditionedIntakeError("The managed import is stale against current Harvest trust evidence.")


def _validate_handoff_document(run_id: str, handoff: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("run_id") != run_id
        or handoff.get("handoff_ready") is not True
        or handoff.get("portable_relative_paths") is not True
        or handoff.get("paths_exposed") is not False
        or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA
    ):
        raise ConditionedIntakeError("Harvest handoff is incomplete or uses an unsupported contract.")
    managed = handoff.get("managed_reference")
    if not isinstance(managed, Mapping) or managed != {"kind": "harvest_run", "run_id": run_id}:
        raise ConditionedIntakeError("Harvest handoff is not bound to its managed run.")
    source = handoff.get("source")
    license_record = handoff.get("license")
    if not isinstance(source, Mapping) or not isinstance(license_record, Mapping):
        raise ConditionedIntakeError("Harvest source or license evidence is missing.")
    source_license = source.get("license")
    if (
        source.get("source_id") != handoff.get("source_id")
        or source_license != license_record
        or str(license_record.get("identifier") or "").casefold() not in ALLOWED_LICENSES
        or license_record.get("permissive_policy") is not True
        or not str(source.get("title") or "").strip()
        or not str(source.get("creator") or "").strip()
        or not str(source.get("source_page") or "").strip()
        or not str(license_record.get("evidence_url") or "").strip()
    ):
        raise ConditionedIntakeError("Harvest provenance or permissive-license evidence is incomplete.")
    identity_fields = (
        "provenance_identity",
        "source_evidence_binding_identity",
        "backend_capability_identity",
        "backend_capability_evidence_identity",
        "limits_identity",
        "acquisition_receipt_identity",
        "artifact_manifest_identity",
        "artifact_set_identity",
    )
    if any(SHA256_PATTERN.fullmatch(str(handoff.get(key) or "")) is None for key in identity_fields):
        raise ConditionedIntakeError("Harvest handoff identity bindings are invalid.")
    expected_provenance = stable_hash(
        {"source": dict(source), "acquisition_receipt_identity": handoff["acquisition_receipt_identity"]}
    )
    evidence = source.get("evidence_binding")
    backend_evidence = handoff.get("backend_capability_evidence")
    if (
        handoff["provenance_identity"] != expected_provenance
        or not isinstance(evidence, Mapping)
        or evidence.get("binding_identity") != handoff["source_evidence_binding_identity"]
        or not isinstance(backend_evidence, Mapping)
        or backend_evidence.get("evidence_identity") != handoff["backend_capability_evidence_identity"]
        or backend_evidence.get("backend_capability_identity") != handoff["backend_capability_identity"]
        or stable_hash(dict(manifest)) != handoff["artifact_manifest_identity"]
        or manifest.get("artifact_set_identity") != handoff["artifact_set_identity"]
        or manifest.get("files") != handoff.get("files")
        or manifest.get("artifact_count") != handoff.get("artifact_count")
        or manifest.get("total_bytes") != handoff.get("total_bytes")
    ):
        raise ConditionedIntakeError("Harvest handoff provenance or artifact bindings are inconsistent.")


def _verify_artifact_tree(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    anchor: AnchoredDirectory | None = None,
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > _MAX_FILES:
        raise ConditionedIntakeError("Harvest artifact manifest is missing or oversized.")
    expected: list[AcquiredFile] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ConditionedIntakeError("Harvest artifact manifest contains an invalid row.")
        expected_sha = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        if raw.get("expected_sha256") not in {None, "", expected_sha}:
            raise ConditionedIntakeError("Harvest expected and actual artifact identities disagree.")
        try:
            expected.append(
                AcquiredFile(
                    relative_path=_canonical_relative(str(raw.get("relative_path") or "")),
                    byte_count=int(raw.get("byte_count")),
                    sha256=expected_sha,
                    mime_type=str(raw.get("mime_type") or ""),
                    usable=raw.get("usable") is True,
                    quarantine_reason=(
                        str(raw.get("quarantine_reason")) if raw.get("quarantine_reason") is not None else None
                    ),
                    taxonomy=tuple(str(value) for value in raw.get("taxonomy") or ()),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConditionedIntakeError("Harvest artifact manifest contains an invalid file identity.") from exc
    limits = HarvestLimits(
        max_files=_MAX_FILES,
        max_file_bytes=_MAX_FILE_BYTES,
        max_total_bytes=_MAX_TOTAL_BYTES,
        max_response_bytes=_MAX_TOTAL_BYTES,
        max_depth=_MAX_DEPTH,
        max_archive_uncompressed_bytes=1024 * 1024 * 1024,
    )
    expected_inventory = {
        item.relative_path: {"sha256": item.sha256, "byte_count": item.byte_count} for item in expected
    }
    if anchor is None:
        with AnchoredDirectory(root, root) as local_anchor:
            return _verify_artifact_tree(root, manifest, anchor=local_anchor)
    anchor.verify()
    before_inventory = _inventory_from_anchor(anchor)["files"]
    if before_inventory != expected_inventory:
        raise ConditionedIntakeError("Managed artifact bytes do not match the exact declared inventory.")
    try:
        scanned = scan_artifacts(
            root,
            limits,
            expected_files=tuple(expected),
            artifacts_anchor=anchor,
        )
    except (OSError, ValueError) as exc:
        raise ConditionedIntakeError("Managed artifact bytes failed complete per-file re-verification.") from exc
    anchor.verify()
    if _inventory_from_anchor(anchor)["files"] != before_inventory:
        raise ConditionedIntakeError("Managed artifact bytes changed during complete re-verification.")
    return scanned


def _copy_verified_artifacts(
    source_anchor: AnchoredDirectory,
    destination_anchor: AnchoredDirectory,
    manifest: Mapping[str, Any],
) -> None:
    for raw in manifest["files"]:
        relative = _canonical_relative(str(raw["relative_path"]))
        content = _read_anchored_relative(source_anchor, relative, _MAX_FILE_BYTES)
        expected = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        if len(content) != int(raw["byte_count"]) or hashlib.sha256(content).hexdigest() != expected:
            raise ConditionedIntakeError("A Harvest artifact changed while it was copied.")
        _write_anchored_relative(destination_anchor, relative, content)
        copied = _read_anchored_relative(destination_anchor, relative, _MAX_FILE_BYTES)
        if copied != content:
            raise ConditionedIntakeError("A managed source copy changed during atomic publication.")


def _read_anchored_relative(anchor: AnchoredDirectory, relative: str, max_bytes: int) -> bytes:
    parts = PurePosixPath(relative).parts
    with ExitStack() as stack:
        current = anchor
        for name in parts[:-1]:
            current = stack.enter_context(current.open_directory(name))
        return _read_anchored_regular_bytes(current, parts[-1], max_bytes)


def _write_anchored_relative(anchor: AnchoredDirectory, relative: str, content: bytes) -> None:
    parts = PurePosixPath(relative).parts
    with ExitStack() as stack:
        current = anchor
        for name in parts[:-1]:
            current.mkdir(name, exist_ok=True)
            current = stack.enter_context(current.open_directory(name))
        current.atomic_write_bytes(parts[-1], content)


def _run_legacy_intake_boundary(
    *,
    work: Path,
    source_root: Path,
    output_root: Path,
    derived_root: Path,
    project_anchor: AnchoredDirectory,
    work_anchor: AnchoredDirectory,
    source_anchor: AnchoredDirectory,
    output_anchor: AnchoredDirectory,
    derived_anchor: AnchoredDirectory,
    verified: Mapping[str, Any],
    run_id: str,
    write_strategy: str,
    held_anchors: Sequence[AnchoredDirectory],
    code_inventory: Mapping[str, Any],
    deadline_monotonic: float,
    cancel_requested: Callable[[], bool],
) -> dict[str, Any]:
    """Run pathname-oriented legacy intake only in the controlled child."""

    if derived_anchor.directory != derived_root:
        raise ConditionedIntakeError("The derived-frame anchor differs from its exact private root.")
    required_anchors = (
        project_anchor,
        work_anchor,
        source_anchor,
        output_anchor,
        derived_anchor,
        *held_anchors,
    )
    _check_operation_control(deadline_monotonic, cancel_requested)
    _verify_anchors(*required_anchors)
    workspace_identity = DirectoryIdentity.from_stat(work_anchor.directory_metadata())
    if write_strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
        _require_windows_legacy_anchors(work, derived_root, required_anchors)
    elif write_strategy != LINUX_LANDLOCK_STRATEGY:
        raise ConditionedIntakeError("Conditioned legacy intake has no approved write-confinement strategy.")

    artifact_sha256: dict[str, str] = {}
    for raw in verified["artifact_manifest"]["files"]:
        if not isinstance(raw, Mapping):
            raise ConditionedIntakeError("Harvest artifact identities are unavailable for controlled intake.")
        relative = _canonical_relative(str(raw.get("relative_path") or ""))
        digest = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        if relative in artifact_sha256 or SHA256_PATTERN.fullmatch(digest) is None:
            raise ConditionedIntakeError("Harvest artifact identities are unavailable for controlled intake.")
        artifact_sha256[relative] = digest

    request_payload = {
        "schema_version": _LEGACY_REQUEST_SCHEMA,
        "run_id": run_id,
        "source": dict(verified["source"]),
        "license": dict(verified["license"]),
        "artifact_sha256": artifact_sha256,
    }
    _audit_writable_workspace(work_anchor, workspace_identity, held_anchors=required_anchors)
    _check_operation_control(deadline_monotonic, cancel_requested)
    if conditioned_code_inventory() != code_inventory:
        raise ConditionedIntakeError("Conditioned intake code changed before controlled worker launch.")
    worker_runtime = controlled_worker_runtime()
    with ExitStack() as fixed_anchor_stack:
        if write_strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
            opened: set[Path] = set()
            for held in required_anchors:
                if held.directory in opened:
                    continue
                expected = DirectoryIdentity.from_stat(held.directory_metadata())
                fixed = fixed_anchor_stack.enter_context(AnchoredDirectory(held.directory, held.directory))
                if DirectoryIdentity.from_stat(fixed.directory_metadata()) != expected:
                    raise ConditionedIntakeError("A Windows writable root changed before child launch.")
                opened.add(held.directory)
        response = _run_legacy_intake_child(
            work,
            strategy=write_strategy,
            workspace_identity=workspace_identity,
            request_payload=request_payload,
            code_inventory=code_inventory,
            deadline_monotonic=deadline_monotonic,
            cancel_requested=cancel_requested,
        )
    _check_operation_control(deadline_monotonic, cancel_requested)
    _assert_pathless_worker_response(response)
    _verify_anchors(*required_anchors)
    if controlled_worker_runtime() != worker_runtime:
        raise ConditionedIntakeError("The controlled worker runtime changed during legacy intake.")
    _verify_artifact_tree(source_root, verified["artifact_manifest"], anchor=source_anchor)

    if (
        set(response)
        != {
            "schema_version",
            "ok",
            "result",
            "write_confinement",
            "paths_exposed",
        }
        or response.get("schema_version") != _LEGACY_RESPONSE_SCHEMA
    ):
        raise ConditionedIntakeError("The controlled legacy intake response is invalid.")
    if response.get("ok") is not True or response.get("paths_exposed") is not False:
        raise ConditionedIntakeError("The controlled legacy intake failed closed.")
    _validate_write_confinement_evidence(
        response.get("write_confinement"),
        strategy=write_strategy,
        workspace_identity=workspace_identity,
    )
    raw_result = response.get("result")
    if not isinstance(raw_result, Mapping) or set(raw_result) != {
        "schema_version",
        "pack_id",
        "sidecar_record_identity",
        "intake_result_identity",
        "result_identity",
    }:
        raise ConditionedIntakeError("The controlled legacy intake result is invalid.")
    result = dict(raw_result)
    result_payload = dict(result)
    result_identity = result_payload.pop("result_identity", None)
    if (
        result.get("schema_version") != "spritelab.dataset.conditioned-legacy-intake-result.v1"
        or not isinstance(result.get("pack_id"), str)
        or SHA256_PATTERN.fullmatch(str(result.get("sidecar_record_identity") or "")) is None
        or SHA256_PATTERN.fullmatch(str(result.get("intake_result_identity") or "")) is None
        or SHA256_PATTERN.fullmatch(str(result_identity or "")) is None
        or stable_hash(result_payload) != result_identity
    ):
        raise ConditionedIntakeError("The controlled legacy intake result is invalid.")
    _validate_persisted_intake_result(
        output_anchor,
        str(result["intake_result_identity"]),
    )
    accepted_paths = _accepted_source_paths(
        output_root,
        source_root,
        verified["artifact_manifest"],
        output_anchor=output_anchor,
        source_anchor=source_anchor,
    )
    derived_sheet_manifest = _publish_derived_sheet_tree(
        output_anchor=output_anchor,
        source_anchor=source_anchor,
        derived_anchor=derived_anchor,
        artifact_manifest=verified["artifact_manifest"],
        source=verified["source"],
        license_record=verified["license"],
        run_id=run_id,
        deadline_monotonic=deadline_monotonic,
        cancel_requested=cancel_requested,
    )
    covered_source_paths = sorted(
        {
            *accepted_paths,
            *(str(record["parent_source_relative_path"]) for record in derived_sheet_manifest["records"]),
        }
    )
    if not accepted_paths and not derived_sheet_manifest["records"]:
        raise ConditionedIntakeError("Dataset intake accepted no source images.")
    _verify_anchors(*required_anchors)
    return {
        **result,
        "accepted_relative_paths": accepted_paths,
        "covered_source_relative_paths": covered_source_paths,
        "derived_sheet_manifest": derived_sheet_manifest,
        "write_confinement": dict(response["write_confinement"]),
        "worker_runtime": worker_runtime,
    }


def _run_legacy_intake_child(
    work: Path,
    *,
    strategy: str,
    workspace_identity: DirectoryIdentity,
    request_payload: Mapping[str, Any],
    code_inventory: Mapping[str, Any],
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    _worker_path: Path | None = None,
    _helper_path: Path | None = None,
    _source_root: Path | None = None,
    _before_worker_launch: Callable[[], None] | None = None,
    _worker_timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Execute one bounded worker without exposing its stderr or local paths."""

    if isinstance(_worker_timeout_seconds, bool) or not isinstance(_worker_timeout_seconds, (int, float)):
        raise ConditionedIntakeError("The controlled worker timeout is invalid.")
    if not 0 < float(_worker_timeout_seconds) <= 600:
        raise ConditionedIntakeError("The controlled worker timeout is invalid.")
    operation_started = time.monotonic()
    operation_deadline, cancel_probe = _normalize_operation_control(
        deadline_monotonic,
        cancel_requested,
        started_monotonic=operation_started,
    )
    _check_operation_control(operation_deadline, cancel_probe)
    source_root = (
        Path(__file__).resolve(strict=True).parents[3]
        if _source_root is None
        else Path(_source_root).resolve(strict=True)
    )
    expected_worker = source_root / "spritelab" / "product_features" / "conditioned_v5" / "legacy_worker.py"
    expected_helper = source_root / "spritelab" / "utils" / "write_confinement.py"
    worker = (expected_worker if _worker_path is None else Path(_worker_path)).resolve(strict=True)
    helper = expected_helper if _helper_path is None else Path(_helper_path).resolve(strict=True)
    if worker != expected_worker.resolve(strict=True) or helper != expected_helper.resolve(strict=True):
        raise ConditionedIntakeError("The controlled worker entry paths differ from their audited module names.")
    runtime_roots = controlled_worker_dependency_roots()
    manifest_bytes = _worker_module_manifest(code_inventory, runtime_roots=runtime_roots)
    manifest_name = f".conditioned-worker-modules-{uuid.uuid4().hex}.json"
    with AnchoredDirectory(work, work) as work_anchor:
        _publish_anchored_file_noreplace(
            work_anchor,
            manifest_name,
            manifest_bytes,
            residue_prefix=".conditioned-worker-manifest-residue-",
        )
    manifest_path = work / manifest_name
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    environment = controlled_worker_environment(work / "tmp")
    runtime_root_arguments = [
        value
        for root, metadata, _distributions in runtime_roots
        for value in (str(root), str(metadata.st_dev), str(metadata.st_ino))
    ]
    executable = controlled_worker_executable()
    executable_binding = _worker_executable_binding(code_inventory)
    payload = (strict_json_dumps(dict(request_payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if _before_worker_launch is not None:
        _before_worker_launch()
    _check_operation_control(operation_deadline, cancel_probe)
    try:
        worker_launch_arguments = controlled_worker_launch_arguments()
    except ConditionedCodeIdentityError as exc:
        raise ConditionedIntakeError("The controlled worker launch source changed before execution.") from exc
    process: Any | None = None
    job_handle = 0
    stdout = b""
    returncode: int | None = None
    try:
        with pin_executable(
            executable,
            expected_sha256=executable_binding[0],
            expected_size=executable_binding[1],
            expected_metadata_sha256=executable_binding[2],
        ) as pinned:
            command = [
                pinned.launch_path,
                *worker_launch_arguments,
                str(source_root),
                str(manifest_path),
                manifest_sha256,
                str(len(manifest_bytes)),
                str(len(runtime_roots)),
                *runtime_root_arguments,
                strategy,
                str(work),
                str(workspace_identity.device),
                str(workspace_identity.inode),
            ]
            if os.name == "nt":
                process = create_windows_bootstrap_untrusted_process(
                    command,
                    cwd=work,
                    env=environment,
                    stdin_payload=payload,
                )
            else:
                options: dict[str, Any] = {
                    "stdin": subprocess.PIPE,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.DEVNULL,
                    "cwd": work,
                    "env": environment,
                    "close_fds": True,
                    "shell": False,
                    "start_new_session": True,
                    "pass_fds": pinned.pass_fds,
                }
                if sys.platform.startswith("linux"):
                    options["preexec_fn"] = linux_parent_death_signal(os.getpid())
                process = subprocess.Popen(command, **options)
            if os.name == "nt":
                job_handle = activate_windows_suspended_process(
                    process,
                    verifier=lambda child: verify_process_image(child, pinned),
                )
            else:
                verify_process_image(process, pinned)
            worker_deadline = min(
                operation_deadline,
                time.monotonic() + float(_worker_timeout_seconds),
            )
            first_communication = True
            while True:
                remaining = _remaining_operation_seconds(operation_deadline, cancel_probe)
                worker_remaining = worker_deadline - time.monotonic()
                if worker_remaining <= 0:
                    _terminate_worker_group(process, job_handle=job_handle)
                    job_handle = 0
                    if time.monotonic() >= operation_deadline:
                        raise DatasetImportDeadlineExceeded(
                            "Conditioned Dataset import exceeded its whole-operation deadline."
                        )
                    raise ConditionedIntakeError("The controlled legacy intake worker exceeded its time limit.")
                try:
                    stdout, _stderr = process.communicate(
                        input=payload if first_communication else None,
                        timeout=min(_OPERATION_POLL_SECONDS, remaining, worker_remaining),
                    )
                    break
                except subprocess.TimeoutExpired:
                    first_communication = False
                    continue
            _check_operation_control(operation_deadline, cancel_probe)
            returncode = process.returncode
    except (DatasetImportCancelled, DatasetImportDeadlineExceeded):
        if process is not None:
            _terminate_worker_group(process, job_handle=job_handle)
            job_handle = 0
        raise
    except ConditionedIntakeError:
        raise
    except (OSError, subprocess.SubprocessError, PinnedExecutableError, WriteConfinementError) as exc:
        if process is not None:
            _terminate_worker_group(process, job_handle=job_handle)
            job_handle = 0
        raise ConditionedIntakeError("The controlled legacy intake worker was unavailable.") from exc
    finally:
        if job_handle:
            close_windows_handle(job_handle)
    if len(stdout) > _MAX_LEGACY_RESPONSE_BYTES:
        raise ConditionedIntakeError("The controlled legacy intake response exceeded its byte limit.")
    try:
        response = strict_json_loads(stdout)
    except ValueError as exc:
        raise ConditionedIntakeError("The controlled legacy intake response is invalid.") from exc
    if not isinstance(response, Mapping):
        raise ConditionedIntakeError("The controlled legacy intake response is invalid.")
    value = dict(response)
    if returncode != 0:
        if set(value) != {"schema_version", "ok", "error_code", "paths_exposed"}:
            raise ConditionedIntakeError("The controlled legacy intake failed without valid evidence.")
        if value.get("error_code") == "write_confinement_unavailable":
            raise ConditionedIntakeError(
                "Conditioned Dataset intake is unavailable because OS write confinement could not be established."
            )
        raise ConditionedIntakeError("The controlled legacy intake failed closed.")
    _check_operation_control(operation_deadline, cancel_probe)
    return value


def _worker_executable_binding(code_inventory: Mapping[str, Any]) -> tuple[str, int, str]:
    runtime = code_inventory.get("worker_runtime")
    if not isinstance(runtime, Mapping):
        raise ConditionedIntakeError("The controlled worker lacks an exact interpreter binding.")
    digest = str(runtime.get("executable_sha256") or "")
    size = runtime.get("executable_byte_count")
    metadata = str(runtime.get("executable_metadata_sha256") or "")
    if (
        SHA256_PATTERN.fullmatch(digest) is None
        or type(size) is not int
        or int(size) <= 0
        or SHA256_PATTERN.fullmatch(metadata) is None
    ):
        raise ConditionedIntakeError("The controlled worker lacks an exact interpreter binding.")
    return digest, int(size), metadata


def _terminate_worker_group(process: Any, *, job_handle: int) -> None:
    if os.name == "nt" and job_handle:
        close_windows_handle(job_handle)
    elif sys.platform.startswith("linux"):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def _worker_module_manifest(
    code_inventory: Mapping[str, Any],
    *,
    runtime_roots: Sequence[tuple[Path, os.stat_result, tuple[str, ...]]] | None = None,
) -> bytes:
    files = code_inventory.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ConditionedIntakeError("The controlled worker module inventory is unavailable.")
    modules: dict[str, dict[str, Any]] = {}
    resource_packages: dict[str, dict[str, dict[str, Any]]] = {}
    collision_keys: set[str] = set()
    for raw_relative in sorted(files):
        relative = str(raw_relative)
        if not relative.startswith("spritelab/") or "\\" in relative:
            raise ConditionedIntakeError("The controlled worker module inventory contains an invalid path.")
        parts = PurePosixPath(relative).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ConditionedIntakeError("The controlled worker module inventory contains an invalid path.")
        binding = _code_file_binding(code_inventory, relative)
        if not relative.endswith(".py"):
            if len(parts) != 3 or parts[:2] != ("spritelab", "config") or not parts[-1].endswith(".yaml"):
                raise ConditionedIntakeError("The controlled worker inventory contains an unsupported resource.")
            package_resources = resource_packages.setdefault("spritelab.config", {})
            package_resources[parts[-1]] = {
                "relative_path": relative,
                "sha256": binding[0],
                "byte_count": binding[1],
            }
            continue
        if parts[-1] == "__init__.py":
            module_parts = parts[:-1]
            is_package = True
        else:
            module_parts = (*parts[:-1], parts[-1].removesuffix(".py"))
            is_package = False
        module_name = ".".join(module_parts)
        collision = unicodedata.normalize("NFC", module_name).casefold()
        if not module_name or module_name in modules or collision in collision_keys:
            raise ConditionedIntakeError("The controlled worker module inventory contains a module collision.")
        collision_keys.add(collision)
        modules[module_name] = {
            "relative_path": relative,
            "sha256": binding[0],
            "byte_count": binding[1],
            "is_package": is_package,
        }
    worker_module = "spritelab.product_features.conditioned_v5.legacy_worker"
    helper_module = "spritelab.utils.write_confinement"
    if worker_module not in modules or helper_module not in modules:
        raise ConditionedIntakeError("The controlled worker inventory lacks its exact entry modules.")
    roots = tuple(controlled_worker_dependency_roots() if runtime_roots is None else runtime_roots)
    root_index_by_distribution: dict[str, int] = {}
    for index, (_root, _metadata, distributions) in enumerate(roots):
        for distribution in distributions:
            if distribution in root_index_by_distribution:
                raise ConditionedIntakeError("A controlled dependency is bound to more than one runtime root.")
            root_index_by_distribution[distribution] = index
    runtime_dependencies = code_inventory.get("runtime_dependencies")
    if not isinstance(runtime_dependencies, Mapping):
        runtime_dependencies = conditioned_code_inventory().get("runtime_dependencies")
    if not isinstance(runtime_dependencies, Mapping) or set(runtime_dependencies) != set(root_index_by_distribution):
        raise ConditionedIntakeError("The controlled worker dependency inventories are unavailable.")
    dependency_bindings: dict[str, dict[str, Any]] = {}
    for distribution, raw_inventory in sorted(runtime_dependencies.items()):
        if not isinstance(raw_inventory, Mapping):
            raise ConditionedIntakeError("A controlled worker dependency inventory is invalid.")
        inventory = dict(raw_inventory)
        identity = str(inventory.get("inventory_sha256") or "")
        base = dict(inventory)
        base.pop("inventory_sha256", None)
        if SHA256_PATTERN.fullmatch(identity) is None or stable_hash(base) != identity:
            raise ConditionedIntakeError("A controlled worker dependency inventory identity is invalid.")
        dependency_bindings[str(distribution)] = {
            "runtime_root_index": root_index_by_distribution[str(distribution)],
            "inventory": inventory,
        }
    value = {
        "schema_version": _WORKER_MODULE_MANIFEST_SCHEMA,
        "worker_module": worker_module,
        "helper_module": helper_module,
        "modules": dict(sorted(modules.items())),
        "resource_packages": {
            package: dict(sorted(resources.items())) for package, resources in sorted(resource_packages.items())
        },
        "runtime_dependencies": dependency_bindings,
    }
    return (strict_json_dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _code_file_binding(code_inventory: Mapping[str, Any], relative_path: str) -> tuple[str, int]:
    files = code_inventory.get("files")
    entry = files.get(relative_path) if isinstance(files, Mapping) else None
    if (
        not isinstance(entry, Mapping)
        or set(entry) != {"sha256", "byte_count"}
        or SHA256_PATTERN.fullmatch(str(entry.get("sha256") or "")) is None
        or type(entry.get("byte_count")) is not int
        or int(entry["byte_count"]) <= 0
    ):
        raise ConditionedIntakeError("The controlled worker lacks an exact audited code binding.")
    return str(entry["sha256"]), int(entry["byte_count"])


def _audit_writable_workspace(
    anchor: AnchoredDirectory,
    identity: DirectoryIdentity,
    *,
    held_anchors: Sequence[AnchoredDirectory] = (),
) -> None:
    """Reject mount/reparse/hard-link seams before granting child writes."""

    anchor.verify()
    if DirectoryIdentity.from_stat(anchor.directory_metadata()) != identity:
        raise ConditionedIntakeError("The controlled workspace identity changed before launch.")
    if os.name == "posix" and _linux_mount_descendants(anchor.directory, include_root=True):
        raise ConditionedIntakeError("The controlled workspace contains a nested mount boundary.")
    held_by_path = {held.directory: held for held in held_anchors}
    held_by_path[anchor.directory] = anchor
    _audit_writable_anchor(anchor, expected_device=identity.device, held_by_path=held_by_path)


def _audit_writable_anchor(
    anchor: AnchoredDirectory,
    *,
    expected_device: int,
    held_by_path: Mapping[Path, AnchoredDirectory],
) -> None:
    anchor.verify()
    if anchor.directory_metadata().st_dev != expected_device:
        raise ConditionedIntakeError("The controlled workspace crosses a filesystem device boundary.")
    names = anchor.names()
    retained_aliases = {name for name in names if _intake_retained_stage_target(anchor, name, names) is not None}
    for name in names:
        metadata = anchor.lstat(name)
        if name in retained_aliases:
            continue
        if _metadata_is_link_or_reparse(metadata) or metadata.st_dev != expected_device:
            raise ConditionedIntakeError("The controlled workspace contains a linked or mounted seam.")
        if stat.S_ISDIR(metadata.st_mode):
            child_path = anchor.directory / name
            held = held_by_path.get(child_path)
            if held is not None:
                if not OwnedFileIdentity.from_stat(metadata).matches(held.directory_metadata()):
                    raise ConditionedIntakeError("A held workspace directory changed during its recursive audit.")
                _audit_writable_anchor(
                    held,
                    expected_device=expected_device,
                    held_by_path=held_by_path,
                )
            else:
                with anchor.open_directory_immovable(name) as child:
                    _audit_writable_anchor(
                        child,
                        expected_device=expected_device,
                        held_by_path=held_by_path,
                    )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink not in {1, 2}:
            raise ConditionedIntakeError("The controlled workspace contains a non-owned filesystem entry.")
        if metadata.st_nlink == 2:
            _intake_retained_stage_alias(anchor, name, metadata)
    anchor.verify()


def _linux_mount_descendants(root: Path, *, include_root: bool = False) -> tuple[Path, ...]:
    if not sys.platform.startswith("linux"):
        return ()
    try:
        with Path("/proc/self/mountinfo").open("rb") as handle:
            payload = handle.read(8 * 1024 * 1024 + 1)
    except OSError as exc:
        raise ConditionedIntakeError("Linux mount boundaries could not be audited before child launch.") from exc
    if len(payload) > 8 * 1024 * 1024:
        raise ConditionedIntakeError("Linux mount-boundary evidence exceeded its byte limit.")
    absolute_root = Path(os.path.abspath(root))
    descendants: list[Path] = []
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ConditionedIntakeError("Linux mount-boundary evidence is malformed.") from exc
    for line in lines:
        fields = line.split(" ")
        if len(fields) < 6:
            raise ConditionedIntakeError("Linux mount-boundary evidence is malformed.")
        mount_point = Path(os.path.abspath(_decode_mountinfo_field(fields[4])))
        try:
            relative = mount_point.relative_to(absolute_root)
        except ValueError:
            continue
        if relative.parts or include_root:
            descendants.append(mount_point)
    return tuple(sorted(set(descendants), key=lambda path: (len(path.parts), os.fspath(path))))


def _decode_mountinfo_field(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    return re.sub(r"\\([0-7]{3})", replace, value)


def _assert_pathless_worker_response(value: Any) -> None:
    """Reject any response string that could carry a private pathname."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_pathless_worker_response(key)
            _assert_pathless_worker_response(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_pathless_worker_response(item)
        return
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if (
            "\x00" in value
            or "/" in value
            or "\\" in value
            or windows.drive
            or value.startswith("~")
            or value.casefold().startswith("file:")
        ):
            raise ConditionedIntakeError("The controlled legacy intake response exposed a private path.")


def _validate_write_confinement_evidence(
    value: Any,
    *,
    strategy: str,
    workspace_identity: DirectoryIdentity,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _WRITE_CONFINEMENT_EVIDENCE_KEYS:
        raise ConditionedIntakeError("The controlled legacy intake lacks exact write-confinement evidence.")
    if (
        value.get("schema_version") != "spritelab.write-confinement-evidence.v3"
        or value.get("strategy") != strategy
        or value.get("root_identity_sha256") != workspace_identity.identity_sha256
        or value.get("paths_exposed") is not False
        or type(value.get("kernel_abi")) is not int
        or type(value.get("handled_access_fs")) is not int
        or type(value.get("allowed_access_fs")) is not int
    ):
        raise ConditionedIntakeError("The controlled legacy intake write-confinement evidence is invalid.")
    if strategy == LINUX_LANDLOCK_STRATEGY:
        if (
            value.get("platform") != "linux"
            or int(value["kernel_abi"]) < 3
            or value.get("no_new_privileges") is not True
            or int(value["handled_access_fs"]) <= 0
            or int(value["allowed_access_fs"]) <= 0
            or value.get("restricted_token") is not False
            or value.get("integrity_level_rid") != 0
            or value.get("mandatory_no_write_up") is not False
            or value.get("workspace_integrity_level_rid") != 0
            or value.get("startup_integrity_level_rid") != 0
            or value.get("bootstrap_lowered_before_worker_import") is not False
            or value.get("new_thread_integrity_level_rid") != 0
            or value.get("raise_to_low_denied") is not False
            or value.get("medium_probe_write_denied") is not False
            or value.get("low_world_probe_write_denied") is not False
            or value.get("untrusted_world_outside_guaranteed") is not False
            or value.get("job_kill_on_close") is not False
            or value.get("job_active_process_limit") != 0
        ):
            raise ConditionedIntakeError("Linux legacy intake did not establish the required Landlock boundary.")
    elif strategy == WINDOWS_PARENT_ANCHORS_STRATEGY:
        if (
            value.get("platform") != "windows"
            or value.get("kernel_abi") != 0
            or value.get("no_new_privileges") is not False
            or value.get("handled_access_fs") != 0
            or value.get("allowed_access_fs") != 0
            or type(value.get("restricted_token")) is not bool
            or value.get("integrity_level_rid") != 0
            or value.get("mandatory_no_write_up") is not True
            or value.get("workspace_integrity_level_rid") != 0
            or value.get("startup_integrity_level_rid") != 4096
            or value.get("bootstrap_lowered_before_worker_import") is not True
            or value.get("new_thread_integrity_level_rid") != 0
            or value.get("raise_to_low_denied") is not True
            or value.get("medium_probe_write_denied") is not True
            or value.get("low_world_probe_write_denied") is not True
            or value.get("untrusted_world_outside_guaranteed") is not False
            or value.get("job_kill_on_close") is not True
            or value.get("job_active_process_limit") != 1
        ):
            raise ConditionedIntakeError("Windows legacy intake did not establish the bootstrap-to-Untrusted boundary.")
    else:
        raise ConditionedIntakeError("The controlled legacy intake write-confinement strategy is invalid.")


def _require_windows_legacy_anchors(
    work: Path,
    derived_root: Path,
    anchors: Sequence[AnchoredDirectory],
) -> None:
    required = {
        work,
        work / "tmp",
        work / "source",
        work / "datasets",
        work / "datasets" / "managed",
        derived_root,
        work / "datasets" / "source_metadata",
        work / "datasets" / "source_metadata" / ".transactions",
        work / "runs",
        work / "runs" / "v3",
    }
    held = {anchor.directory for anchor in anchors}
    if not required <= held:
        raise ConditionedIntakeError("Windows legacy intake is unavailable without every audited writable root held.")
    _verify_anchors(*anchors)


def _run_legacy_intake_in_process(
    *,
    work: Path,
    source_root: Path,
    output_root: Path,
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    artifact_sha256: Mapping[str, str],
    run_id: str,
) -> dict[str, Any]:
    """Legacy implementation imported and called only by the confined child."""

    context = ProjectContext(
        project_root=work,
        config={},
        runs_directory=work / "runs" / "v3",
    )
    save_grouping(work, source_root, ["."])
    _root, png_paths, packs = discover_source_packs(source_root, context=context)
    if len(packs) != 1 or packs[0].relative_root != ".":
        raise ConditionedIntakeError("The copied Harvest source did not resolve to one bound source pack.")
    covered_hashes: list[str] = []
    for path in png_paths:
        relative = _canonical_relative(path.relative_to(source_root).as_posix())
        digest = str(artifact_sha256.get(relative) or "")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ConditionedIntakeError("Dataset intake discovered a PNG outside the bound artifact manifest.")
        covered_hashes.append(digest)
    sidecar = save_pack_metadata(
        work,
        source_root,
        packs[0],
        {
            "creator_or_rights_holder": source["creator"],
            "pack_title": source["title"],
            "source_type": "other_downloaded",
            "source_page_url": source["source_page"],
            "original_work_declaration": False,
            "license_identifier": _sidecar_license(str(license_record["identifier"])),
            "license_url": license_record["evidence_url"],
            "license_evidence_file": None,
            "attribution_text": str(license_record.get("attribution_text") or "") or None,
            "permission_confirmed": False,
            "notes": f"Verified Harvest handoff {run_id}.",
        },
        covered_byte_hashes=covered_hashes,
    )
    result = DatasetIntakeService().build(
        source_root,
        output_root=output_root,
        context=context,
        private_fresh_output=True,
    )
    if result.status in {ProductStatus.BLOCKED, ProductStatus.FAILED, ProductStatus.UNAVAILABLE}:
        raise ConditionedIntakeError("Dataset intake could not publish a training-eligible managed output.")
    validate_managed_dataset_output(output_root, context=context, require_datasets_root=True)
    payload = {
        "schema_version": "spritelab.dataset.conditioned-legacy-intake-result.v1",
        "pack_id": packs[0].pack_id,
        "sidecar_record_identity": stable_hash(sidecar),
        "intake_result_identity": stable_hash(result.to_dict()),
    }
    return {**payload, "result_identity": stable_hash(payload)}


def _publish_derived_sheet_tree(
    *,
    output_anchor: AnchoredDirectory,
    source_anchor: AnchoredDirectory,
    derived_anchor: AnchoredDirectory,
    artifact_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    run_id: str,
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish canonical accepted sheet frames below one new immutable root."""

    _check_optional_operation_control(deadline_monotonic, cancel_requested)
    _verify_anchors(output_anchor, source_anchor, derived_anchor)
    if derived_anchor.names():
        raise ConditionedIntakeError("The private derived-frame root was not empty before publication.")
    expected = _expected_derived_sheet_records(
        output_anchor=output_anchor,
        source_anchor=source_anchor,
        artifact_manifest=artifact_manifest,
        source=source,
        license_record=license_record,
        run_id=run_id,
        deadline_monotonic=deadline_monotonic,
        cancel_requested=cancel_requested,
    )
    records = [record for record, _content in expected]
    manifest_payload = {
        "schema_version": DERIVED_SHEET_MANIFEST_SCHEMA,
        "recipe": DERIVED_SHEET_RECIPE,
        "recipe_identity": DERIVED_SHEET_RECIPE_IDENTITY,
        "records": records,
        "record_count": len(records),
        "total_bytes": sum(int(record["encoded_output_byte_count"]) for record in records),
        "portable_relative_paths": True,
        "raw_source_mutated": False,
        "source_derived_not_augmentation": True,
        "paths_exposed": False,
    }
    manifest = {**manifest_payload, "manifest_identity": stable_hash(manifest_payload)}
    _validate_derived_sheet_manifest_document(manifest)

    frames_identity = derived_anchor.mkdir("frames", exist_ok=False)
    with derived_anchor.open_directory_immovable("frames") as frames_anchor:
        _require_created_anchor_identity(
            frames_identity,
            frames_anchor,
            "The private derived-frame directory changed while it was anchored.",
        )
        for record, content in expected:
            _check_optional_operation_control(deadline_monotonic, cancel_requested)
            name = PurePosixPath(str(record["output_relative_path"])).name
            _publish_anchored_file_noreplace(
                frames_anchor,
                name,
                content,
                residue_prefix=".derived-frame-residue-",
            )
            if _read_anchored_regular_bytes(frames_anchor, name, _MAX_FILE_BYTES) != content:
                raise ConditionedIntakeError("A derived frame changed immediately after publication.")
    manifest_bytes = (strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_anchored_file_noreplace(
        derived_anchor,
        "manifest.json",
        manifest_bytes,
        residue_prefix=".derived-manifest-residue-",
    )
    _validate_derived_sheet_tree(
        manifest,
        output_anchor=output_anchor,
        source_anchor=source_anchor,
        derived_anchor=derived_anchor,
        artifact_manifest=artifact_manifest,
        source=source,
        license_record=license_record,
        run_id=run_id,
        deadline_monotonic=deadline_monotonic,
        cancel_requested=cancel_requested,
    )
    _verify_anchors(output_anchor, source_anchor, derived_anchor)
    return manifest


def _validate_derived_sheet_tree(
    manifest: Mapping[str, Any],
    *,
    output_anchor: AnchoredDirectory,
    source_anchor: AnchoredDirectory,
    derived_anchor: AnchoredDirectory,
    artifact_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    run_id: str,
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Recompute every derived byte and reject extras or pathname substitution."""

    _check_optional_operation_control(deadline_monotonic, cancel_requested)
    normalized = _validate_derived_sheet_manifest_document(manifest)
    expected = _expected_derived_sheet_records(
        output_anchor=output_anchor,
        source_anchor=source_anchor,
        artifact_manifest=artifact_manifest,
        source=source,
        license_record=license_record,
        run_id=run_id,
        deadline_monotonic=deadline_monotonic,
        cancel_requested=cancel_requested,
    )
    expected_records = [record for record, _content in expected]
    if normalized != expected_records:
        raise ConditionedIntakeError("The derived-frame manifest differs from the exact intake/crop reconstruction.")
    derived_names = derived_anchor.names()
    derived_aliases = {
        name for name in derived_names if _intake_retained_stage_target(derived_anchor, name, derived_names) is not None
    }
    if set(derived_names) - derived_aliases != {"frames", "manifest.json"}:
        raise ConditionedIntakeError("The derived-frame tree contains an unknown or missing root entry.")
    expected_manifest_bytes = (strict_json_dumps(dict(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _read_anchored_regular_bytes(derived_anchor, "manifest.json", _MAX_JSON_BYTES) != expected_manifest_bytes:
        raise ConditionedIntakeError("The derived-frame manifest file differs from its receipt-bound document.")
    with derived_anchor.open_directory_immovable("frames") as frames_anchor:
        expected_names = {PurePosixPath(str(record["output_relative_path"])).name for record in expected_records}
        frame_names = frames_anchor.names()
        frame_aliases = {
            name for name in frame_names if _intake_retained_stage_target(frames_anchor, name, frame_names) is not None
        }
        if set(frame_names) - frame_aliases != expected_names:
            raise ConditionedIntakeError("The derived-frame directory contains an unknown or missing entry.")
        for record, expected_content in expected:
            _check_optional_operation_control(deadline_monotonic, cancel_requested)
            name = PurePosixPath(str(record["output_relative_path"])).name
            actual = _read_anchored_regular_bytes(frames_anchor, name, _MAX_FILE_BYTES)
            if actual != expected_content:
                raise ConditionedIntakeError("A derived frame differs from its exact parent/crop recipe.")
    _verify_anchors(output_anchor, source_anchor, derived_anchor)
    return normalized


def read_receipt_bound_derived_frame(
    *,
    source_anchor: AnchoredDirectory,
    derived_anchor: AnchoredDirectory,
    record: Mapping[str, Any],
    max_bytes: int,
) -> bytes:
    """Reconstruct one receipt-bound frame while its parent and output roots are held."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= _MAX_FILE_BYTES:
        raise ConditionedIntakeError("The derived-frame read bound is invalid.")
    record_value = dict(record)
    manifest_payload = {
        "schema_version": DERIVED_SHEET_MANIFEST_SCHEMA,
        "recipe": DERIVED_SHEET_RECIPE,
        "recipe_identity": DERIVED_SHEET_RECIPE_IDENTITY,
        "records": [record_value],
        "record_count": 1,
        "total_bytes": record_value.get("encoded_output_byte_count"),
        "portable_relative_paths": True,
        "raw_source_mutated": False,
        "source_derived_not_augmentation": True,
        "paths_exposed": False,
    }
    manifest = {**manifest_payload, "manifest_identity": stable_hash(manifest_payload)}
    normalized = _validate_derived_sheet_manifest_document(manifest)
    bound = normalized[0]
    if int(bound["encoded_output_byte_count"]) > max_bytes:
        raise ConditionedIntakeError("The receipt-bound derived frame exceeds its consumer byte limit.")

    _verify_anchors(source_anchor, derived_anchor)
    parent_content = _read_anchored_relative(
        source_anchor,
        str(bound["parent_source_relative_path"]),
        _MAX_FILE_BYTES,
    )
    if hashlib.sha256(parent_content).hexdigest() != bound["parent_source_raw_sha256"]:
        raise ConditionedIntakeError("A derived-frame parent changed before consumer inspection.")
    parent_width, parent_height, parent_rgba = _decode_single_png_rgba(parent_content)
    if _decoded_rgba_identity(parent_width, parent_height, parent_rgba) != bound["parent_source_decoded_rgba_sha256"]:
        raise ConditionedIntakeError("A derived-frame parent changed its decoded pixel identity.")
    crop = _strict_crop_rectangle(bound["crop_rectangle"], parent_width, parent_height)
    cell_rgba = _crop_rgba(parent_rgba, parent_width, crop)
    if _decoded_rgba_identity(int(bound["width"]), int(bound["height"]), cell_rgba) != bound["decoded_rgba_sha256"]:
        raise ConditionedIntakeError("A derived frame differs from its receipt-bound parent crop.")
    encoded = _encode_canonical_rgba_png(int(bound["width"]), int(bound["height"]), cell_rgba)
    if (
        len(encoded) != bound["encoded_output_byte_count"]
        or hashlib.sha256(encoded).hexdigest() != bound["encoded_output_sha256"]
    ):
        raise ConditionedIntakeError("A derived frame differs from the canonical receipt-bound recipe.")
    actual = _read_anchored_relative(
        derived_anchor,
        str(bound["output_relative_path"]),
        max_bytes,
    )
    if actual != encoded:
        raise ConditionedIntakeError("A receipt-bound derived frame changed before consumer inspection.")
    _verify_anchors(source_anchor, derived_anchor)
    return actual


def _expected_derived_sheet_records(
    *,
    output_anchor: AnchoredDirectory,
    source_anchor: AnchoredDirectory,
    artifact_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    run_id: str,
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[tuple[dict[str, Any], bytes]]:
    """Reconstruct canonical records from accepted legacy sheet-child rows."""

    _check_optional_operation_control(deadline_monotonic, cancel_requested)
    rows = _read_anchored_jsonl(output_anchor, "items.jsonl")
    rows_by_relative: dict[str, dict[str, Any]] = {}
    rows_by_item_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        relative = _canonical_relative(str(raw.get("relative_path") or ""))
        item_id = str(raw.get("item_id") or "")
        if not item_id or relative in rows_by_relative or item_id in rows_by_item_id:
            raise ConditionedIntakeError("Dataset intake emitted duplicate or invalid item identities.")
        rows_by_relative[relative] = dict(raw)
        rows_by_item_id[item_id] = dict(raw)

    artifacts: dict[str, dict[str, Any]] = {}
    for raw in artifact_manifest.get("files", ()):
        if not isinstance(raw, Mapping):
            raise ConditionedIntakeError("The Harvest artifact manifest contains an invalid file record.")
        relative = _canonical_relative(str(raw.get("relative_path") or ""))
        digest = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        byte_count = raw.get("byte_count")
        if (
            relative in artifacts
            or SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise ConditionedIntakeError("The Harvest artifact manifest cannot bind derived parents exactly.")
        artifacts[relative] = {
            "sha256": digest,
            "byte_count": byte_count,
            "eligible": _harvest_artifact_is_eligible_png(raw),
        }

    accepted_children = [
        row
        for row in rows_by_relative.values()
        if row.get("current_disposition") == "accepted" and row.get("sheet_extraction") is not None
    ]
    if len(accepted_children) > _MAX_DERIVED_FRAMES:
        raise ConditionedIntakeError("Dataset intake produced too many derived sheet frames.")
    parent_cache: dict[str, tuple[int, int, bytes, str]] = {}
    results: list[tuple[dict[str, Any], bytes]] = []
    collision_keys: set[str] = set()
    derivation_ids: set[str] = set()
    total_encoded_bytes = 0
    for item in sorted(accepted_children, key=lambda value: str(value["relative_path"])):
        _check_optional_operation_control(deadline_monotonic, cancel_requested)
        extraction = item.get("sheet_extraction")
        if not isinstance(extraction, Mapping) or set(extraction) != _SHEET_EXTRACTION_KEYS:
            raise ConditionedIntakeError("An accepted derived frame has an invalid extraction contract.")
        parent_relative = _canonical_relative(str(extraction.get("source_relative_path") or ""))
        parent_artifact = artifacts.get(parent_relative)
        parent_item = rows_by_relative.get(parent_relative)
        if parent_artifact is None or parent_item is None:
            raise ConditionedIntakeError("A derived frame is not bound to one original Harvest parent.")
        if parent_artifact["eligible"] is not True:
            continue
        parent_raw_sha256 = str(parent_artifact["sha256"])
        if (
            extraction.get("source_item_id") != parent_item.get("item_id")
            or extraction.get("source_byte_sha256") != parent_raw_sha256
            or parent_item.get("byte_sha256") != parent_raw_sha256
            or extraction.get("extraction_policy_version") != EXTRACTION_POLICY_VERSION
            or extraction.get("source_sheet_modified") is not False
        ):
            raise ConditionedIntakeError("A derived frame changed its parent or extraction-policy binding.")
        if parent_relative not in parent_cache:
            parent_content = _read_anchored_relative(source_anchor, parent_relative, _MAX_FILE_BYTES)
            if (
                len(parent_content) != int(parent_artifact["byte_count"])
                or hashlib.sha256(parent_content).hexdigest() != parent_raw_sha256
            ):
                raise ConditionedIntakeError("A derived-frame parent changed after Harvest verification.")
            width, height, rgba = _decode_single_png_rgba(parent_content)
            parent_cache[parent_relative] = (
                width,
                height,
                rgba,
                _decoded_rgba_identity(width, height, rgba),
            )
        parent_width, parent_height, parent_rgba, parent_decoded_sha256 = parent_cache[parent_relative]
        if (
            extraction.get("source_decoded_rgba_sha256") != parent_decoded_sha256
            or parent_item.get("decoded_rgba_sha256") != parent_decoded_sha256
        ):
            raise ConditionedIntakeError("A derived frame changed its decoded parent identity.")
        crop = _strict_crop_rectangle(extraction.get("crop_rectangle"), parent_width, parent_height)
        frame_index = extraction.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ConditionedIntakeError("A derived frame has an invalid frame index.")
        semantic_relative = _canonical_relative(str(item.get("relative_path") or ""))
        if semantic_relative != f"{parent_relative}#frame{frame_index:04d}":
            raise ConditionedIntakeError("A derived frame changed its deterministic semantic path.")
        cell_rgba = _crop_rgba(parent_rgba, parent_width, crop)
        width = crop[2] - crop[0]
        height = crop[3] - crop[1]
        decoded_sha256 = _decoded_rgba_identity(width, height, cell_rgba)
        if (
            extraction.get("output_decoded_rgba_sha256") != decoded_sha256
            or item.get("decoded_rgba_sha256") != decoded_sha256
            or item.get("width") != width
            or item.get("height") != height
        ):
            raise ConditionedIntakeError("A derived frame differs from the legacy intake crop result.")
        dataset_item_id = str(item.get("item_id") or "")
        if not dataset_item_id:
            raise ConditionedIntakeError("A derived frame lacks a Dataset intake item identity.")
        source_provenance_identity = _derived_source_provenance_identity(
            source=source,
            license_record=license_record,
            run_id=run_id,
            parent_relative=parent_relative,
            parent_raw_sha256=parent_raw_sha256,
        )
        source_group_identity = _derived_source_group_identity(
            source=source,
            run_id=run_id,
            parent_relative=parent_relative,
            parent_raw_sha256=parent_raw_sha256,
        )
        derivation_payload = {
            "schema_version": "spritelab.dataset.conditioned-derived-sheet-derivation.v1",
            "dataset_item_id": dataset_item_id,
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_raw_sha256,
            "parent_source_decoded_rgba_sha256": parent_decoded_sha256,
            "crop_rectangle": list(crop),
            "frame_index": frame_index,
            "recipe_identity": DERIVED_SHEET_RECIPE_IDENTITY,
            "decoded_rgba_sha256": decoded_sha256,
            "source_provenance_identity": source_provenance_identity,
            "source_group_identity": source_group_identity,
        }
        derivation_identity = stable_hash(derivation_payload)
        if derivation_identity in derivation_ids:
            raise ConditionedIntakeError("Dataset intake emitted a duplicate derived-frame identity.")
        derivation_ids.add(derivation_identity)
        output_relative = f"frames/{derivation_identity}.png"
        for portable in (semantic_relative, output_relative):
            collision = portable_path_collision_key(portable)
            if collision in collision_keys:
                raise ConditionedIntakeError("Derived frames contain a case or Unicode path collision.")
            collision_keys.add(collision)
        encoded = _encode_canonical_rgba_png(width, height, cell_rgba)
        total_encoded_bytes += len(encoded)
        if total_encoded_bytes > _MAX_DERIVED_TOTAL_BYTES:
            raise ConditionedIntakeError("Dataset intake produced too many derived sheet bytes.")
        record_payload = {
            "schema_version": DERIVED_SHEET_FRAME_SCHEMA,
            "dataset_item_id": dataset_item_id,
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_raw_sha256,
            "parent_source_decoded_rgba_sha256": parent_decoded_sha256,
            "crop_rectangle": list(crop),
            "frame_index": frame_index,
            "recipe_version": DERIVED_SHEET_RECIPE["schema_version"],
            "recipe_identity": DERIVED_SHEET_RECIPE_IDENTITY,
            "decoded_rgba_sha256": decoded_sha256,
            "width": width,
            "height": height,
            "source_provenance_identity": source_provenance_identity,
            "source_group_identity": source_group_identity,
            "semantic_relative_path": semantic_relative,
            "output_relative_path": output_relative,
            "encoded_output_sha256": hashlib.sha256(encoded).hexdigest(),
            "encoded_output_byte_count": len(encoded),
            "derivation_identity": derivation_identity,
            "source_derived_not_augmentation": True,
        }
        results.append(({**record_payload, "record_identity": stable_hash(record_payload)}, encoded))
    return results


def _validate_derived_sheet_manifest_document(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise ConditionedIntakeError("The derived-frame manifest is malformed or inconsistent.")
    value = dict(manifest)
    identity = value.get("manifest_identity")
    payload = dict(value)
    payload.pop("manifest_identity", None)
    records = value.get("records")
    if (
        set(value) != _DERIVED_SHEET_MANIFEST_KEYS
        or value.get("schema_version") != DERIVED_SHEET_MANIFEST_SCHEMA
        or value.get("recipe") != DERIVED_SHEET_RECIPE
        or value.get("recipe_identity") != DERIVED_SHEET_RECIPE_IDENTITY
        or not isinstance(records, list)
        or isinstance(value.get("record_count"), bool)
        or not isinstance(value.get("record_count"), int)
        or not 0 <= int(value["record_count"]) <= _MAX_DERIVED_FRAMES
        or isinstance(value.get("total_bytes"), bool)
        or not isinstance(value.get("total_bytes"), int)
        or not 0 <= int(value["total_bytes"]) <= _MAX_DERIVED_TOTAL_BYTES
        or value.get("portable_relative_paths") is not True
        or value.get("raw_source_mutated") is not False
        or value.get("source_derived_not_augmentation") is not True
        or value.get("paths_exposed") is not False
        or SHA256_PATTERN.fullmatch(str(identity or "")) is None
        or stable_hash(payload) != identity
    ):
        raise ConditionedIntakeError("The derived-frame manifest is malformed or inconsistent.")
    normalized: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    seen_semantic_paths: set[str] = set()
    seen_output_paths: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _DERIVED_SHEET_FRAME_KEYS:
            raise ConditionedIntakeError("A derived-frame record is malformed.")
        record = dict(raw)
        record_identity = record.get("record_identity")
        record_payload = dict(record)
        record_payload.pop("record_identity", None)
        crop = record.get("crop_rectangle")
        width = record.get("width")
        height = record.get("height")
        frame_index = record.get("frame_index")
        encoded_bytes = record.get("encoded_output_byte_count")
        item_id = record.get("dataset_item_id")
        semantic = record.get("semantic_relative_path")
        output = record.get("output_relative_path")
        if (
            record.get("schema_version") != DERIVED_SHEET_FRAME_SCHEMA
            or not isinstance(item_id, str)
            or not item_id
            or not isinstance(crop, list)
            or len(crop) != 4
            or any(isinstance(part, bool) or not isinstance(part, int) for part in crop)
            or not crop[0] < crop[2]
            or not crop[1] < crop[3]
            or isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width != crop[2] - crop[0]
            or height != crop[3] - crop[1]
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or record.get("recipe_version") != DERIVED_SHEET_RECIPE["schema_version"]
            or record.get("recipe_identity") != DERIVED_SHEET_RECIPE_IDENTITY
            or isinstance(encoded_bytes, bool)
            or not isinstance(encoded_bytes, int)
            or not 0 < encoded_bytes <= _MAX_FILE_BYTES
            or record.get("source_derived_not_augmentation") is not True
            or any(
                SHA256_PATTERN.fullmatch(str(record.get(name) or "")) is None
                for name in (
                    "parent_source_raw_sha256",
                    "parent_source_decoded_rgba_sha256",
                    "decoded_rgba_sha256",
                    "source_provenance_identity",
                    "source_group_identity",
                    "encoded_output_sha256",
                    "derivation_identity",
                    "record_identity",
                )
            )
            or stable_hash(record_payload) != record_identity
        ):
            raise ConditionedIntakeError("A derived-frame record is malformed or inconsistent.")
        parent = _canonical_relative(str(record.get("parent_source_relative_path") or ""))
        semantic = _canonical_relative(str(semantic or ""))
        output = _canonical_relative(str(output or ""))
        derivation_payload = {
            "schema_version": "spritelab.dataset.conditioned-derived-sheet-derivation.v1",
            "dataset_item_id": item_id,
            "parent_source_relative_path": parent,
            "parent_source_raw_sha256": record["parent_source_raw_sha256"],
            "parent_source_decoded_rgba_sha256": record["parent_source_decoded_rgba_sha256"],
            "crop_rectangle": crop,
            "frame_index": frame_index,
            "recipe_identity": DERIVED_SHEET_RECIPE_IDENTITY,
            "decoded_rgba_sha256": record["decoded_rgba_sha256"],
            "source_provenance_identity": record["source_provenance_identity"],
            "source_group_identity": record["source_group_identity"],
        }
        if (
            stable_hash(derivation_payload) != record["derivation_identity"]
            or semantic != f"{parent}#frame{frame_index:04d}"
            or output != f"frames/{record['derivation_identity']}.png"
            or item_id in seen_item_ids
            or portable_path_collision_key(semantic) in seen_semantic_paths
            or portable_path_collision_key(output) in seen_output_paths
        ):
            raise ConditionedIntakeError("A derived-frame identity or portable path is inconsistent.")
        seen_item_ids.add(item_id)
        seen_semantic_paths.add(portable_path_collision_key(semantic))
        seen_output_paths.add(portable_path_collision_key(output))
        normalized.append(record)
    if (
        value["record_count"] != len(normalized)
        or value["total_bytes"] != sum(int(record["encoded_output_byte_count"]) for record in normalized)
        or normalized != sorted(normalized, key=lambda record: str(record["semantic_relative_path"]))
    ):
        raise ConditionedIntakeError("The derived-frame manifest count, bytes, or ordering is inconsistent.")
    return normalized


def _decode_single_png_rgba(content: bytes) -> tuple[int, int, bytes]:
    try:
        with Image.open(io.BytesIO(content)) as opened:
            width, height = opened.size
            if (
                opened.format != "PNG"
                or getattr(opened, "n_frames", 1) != 1
                or width < 1
                or height < 1
                or width * height > _MAX_DERIVED_PARENT_PIXELS
            ):
                raise ConditionedIntakeError("A derived-frame parent is not one bounded static PNG.")
            opened.load()
            rgba = opened.convert("RGBA").tobytes()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ConditionedIntakeError("A derived-frame parent could not be decoded safely.") from exc
    if len(rgba) != width * height * 4:
        raise ConditionedIntakeError("A derived-frame parent decoded to an invalid RGBA buffer.")
    return width, height, rgba


def _decoded_rgba_identity(width: int, height: int, rgba: bytes) -> str:
    """Return the canonical Dataset-v5 decoded-pixel identity."""

    if width < 1 or height < 1 or len(rgba) != width * height * 4:
        raise ConditionedIntakeError("Decoded RGBA pixels have an invalid shape.")
    pixels = np.frombuffer(rgba, dtype=np.uint8).reshape((height, width, 4))
    return dataset_decoded_rgba_sha256(pixels)


def _strict_crop_rectangle(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
    ):
        raise ConditionedIntakeError("A derived frame has an invalid crop rectangle.")
    left, top, right, bottom = (int(part) for part in value)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ConditionedIntakeError("A derived frame crop escapes its exact parent image.")
    return left, top, right, bottom


def _crop_rgba(parent: bytes, parent_width: int, crop: tuple[int, int, int, int]) -> bytes:
    left, top, right, bottom = crop
    row_bytes = parent_width * 4
    return b"".join(
        parent[(row * row_bytes) + (left * 4) : (row * row_bytes) + (right * 4)] for row in range(top, bottom)
    )


def _encode_canonical_rgba_png(width: int, height: int, rgba: bytes) -> bytes:
    if width < 1 or height < 1 or len(rgba) != width * height * 4:
        raise ConditionedIntakeError("A derived frame cannot be encoded from an invalid RGBA buffer.")
    stride = width * 4
    scanlines = b"".join(b"\x00" + rgba[offset : offset + stride] for offset in range(0, len(rgba), stride))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


def _derived_source_provenance_identity(
    *,
    source: Mapping[str, Any],
    license_record: Mapping[str, Any],
    run_id: str,
    parent_relative: str,
    parent_raw_sha256: str,
) -> str:
    return stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-provenance.v1",
            "run_id": run_id,
            "source": dict(source),
            "license": dict(license_record),
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_raw_sha256,
        }
    )


def _derived_source_group_identity(
    *,
    source: Mapping[str, Any],
    run_id: str,
    parent_relative: str,
    parent_raw_sha256: str,
) -> str:
    return stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-group.v1",
            "run_id": run_id,
            "source_id": str(source.get("source_id") or ""),
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_raw_sha256,
        }
    )


def _accepted_source_paths(
    output: Path,
    source: Path,
    manifest: Mapping[str, Any],
    *,
    output_anchor: AnchoredDirectory,
    source_anchor: AnchoredDirectory,
) -> list[str]:
    del output
    rows = _read_anchored_jsonl(output_anchor, "items.jsonl")
    known = {
        str(row["relative_path"]): {
            "sha256": str(row.get("actual_sha256") or row.get("sha256") or ""),
            "eligible": _harvest_artifact_is_eligible_png(row),
        }
        for row in manifest["files"]
        if isinstance(row, Mapping)
    }
    accepted: list[str] = []
    for item in rows:
        if item.get("current_disposition") != "accepted":
            continue
        extraction = item.get("sheet_extraction")
        if extraction is not None:
            if not isinstance(extraction, Mapping):
                raise ConditionedIntakeError("Dataset intake emitted an invalid derived-frame marker.")
            continue
        relative = _canonical_relative(str(item.get("relative_path") or ""))
        binding = known.get(relative)
        if binding is None or item.get("byte_sha256") != binding["sha256"]:
            raise ConditionedIntakeError("Dataset intake accepted an image outside the original artifact manifest.")
        if binding["eligible"] is not True:
            continue
        content = _read_anchored_relative(source_anchor, relative, _MAX_FILE_BYTES)
        if hashlib.sha256(content).hexdigest() != binding["sha256"]:
            raise ConditionedIntakeError("Dataset intake source bytes changed after preprocessing.")
        expected_path = os.path.normcase(os.path.abspath(source.joinpath(*PurePosixPath(relative).parts)))
        raw_item_path = str(item.get("source_path") or "")
        item_path = Path(raw_item_path)
        if not item_path.is_absolute() or os.path.normcase(os.path.abspath(item_path)) != expected_path:
            raise ConditionedIntakeError("Dataset intake source binding changed after preprocessing.")
        accepted.append(relative)
    if len(accepted) != len(set(accepted)):
        raise ConditionedIntakeError("Dataset intake accepted duplicate source identities.")
    return sorted(accepted)


def _harvest_artifact_is_eligible_png(value: Mapping[str, Any]) -> bool:
    return (
        value.get("usable") is True and value.get("quarantine_reason") is None and value.get("mime_type") == "image/png"
    )


def _load_managed_receipt_from_anchor(
    anchor: AnchoredDirectory,
    project_root: Path,
    dataset_reference: str,
) -> dict[str, Any]:
    if REFERENCE_PATTERN.fullmatch(str(dataset_reference)) is None:
        raise ConditionedIntakeError("Managed Dataset reference is invalid.")
    return _validate_managed_receipt(
        _read_anchored_json(anchor, f"{dataset_reference}.json"),
        project_root,
        dataset_reference,
    )


def _validate_managed_receipt(
    receipt: Mapping[str, Any],
    project_root: Path,
    dataset_reference: str,
    *,
    require_current_code: bool = True,
) -> dict[str, Any]:
    receipt = dict(receipt)
    stored_code_inventory = _validate_stored_code_inventory(receipt.get("callback_code_inventory"))
    current_code_inventory = conditioned_code_inventory() if require_current_code else stored_code_inventory
    identity = receipt.get("receipt_identity")
    payload = dict(receipt)
    payload.pop("receipt_identity", None)
    accepted_count = receipt.get("accepted_count")
    quarantined_count = receipt.get("quarantined_count")
    if (
        set(receipt) != _MANAGED_RECEIPT_KEYS
        or receipt.get("schema_version") != MANAGED_IMPORT_RECEIPT_SCHEMA
        or receipt.get("dataset_reference") != dataset_reference
        or receipt.get("callback_id") != ConditionedDatasetImportAdapter.callback_id
        or SHA256_PATTERN.fullmatch(str(receipt.get("request_identity") or "")) is None
        or receipt.get("paths_exposed") is not False
        or receipt.get("portable_relative_paths") is not True
        or receipt.get("raw_harvest_mutated") is not False
        or receipt.get("atomic_publication") != "receipt_pointer_after_validation"
        or isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or accepted_count < 1
        or isinstance(quarantined_count, bool)
        or not isinstance(quarantined_count, int)
        or quarantined_count < 0
        or not isinstance(receipt.get("created_at"), str)
        or not str(receipt["created_at"])
        or not SHA256_PATTERN.fullmatch(str(identity or ""))
        or stable_hash(payload) != identity
        or receipt.get("callback_code_inventory") != stored_code_inventory
        or receipt.get("callback_code_identity_sha256") != stored_code_inventory["inventory_sha256"]
        or (require_current_code and stored_code_inventory != current_code_inventory)
        or not isinstance(receipt.get("managed"), Mapping)
        or not isinstance(receipt.get("operation_control"), Mapping)
        or not isinstance(receipt.get("harvest"), Mapping)
        or not isinstance(receipt.get("handoff_document"), Mapping)
        or not isinstance(receipt.get("artifact_manifest"), Mapping)
        or not isinstance(receipt.get("source"), Mapping)
        or not isinstance(receipt.get("license"), Mapping)
    ):
        raise ConditionedIntakeError("Managed Dataset import receipt is malformed or inconsistent.")
    if dataset_reference != f"dataset.{str(receipt['request_identity'])[:24]}":
        raise ConditionedIntakeError("Managed Dataset reference differs from its request identity.")
    handoff_document = receipt["handoff_document"]
    if handoff_document.get("source") != receipt["source"] or handoff_document.get("license") != receipt["license"]:
        raise ConditionedIntakeError("Managed Dataset source or license differs from its Harvest handoff.")
    _validate_managed_harvest_document(
        receipt["harvest"],
        handoff_document=handoff_document,
        artifact_manifest=receipt["artifact_manifest"],
    )
    _validate_stored_operation_control(receipt["operation_control"])
    managed = dict(receipt["managed"])
    if set(managed) != _MANAGED_RECEIPT_MANAGED_KEYS:
        raise ConditionedIntakeError("Managed Dataset receipt has unknown or missing managed fields.")
    _validate_stored_write_confinement(managed.get("write_confinement"))
    if managed.get("worker_runtime") != stored_code_inventory.get("worker_runtime"):
        raise ConditionedIntakeError("Managed Dataset worker-runtime evidence is invalid.")
    inventories: dict[str, dict[str, Any]] = {}
    for name in ("source_inventory", "output_inventory", "derived_inventory"):
        inventory = _validate_managed_inventory_document(managed.get(name))
        if stable_hash(inventory) != managed.get(f"{name}_sha256"):
            raise ConditionedIntakeError("Managed Dataset inventory identity is invalid.")
        inventories[name] = inventory
    derived_manifest = _validate_derived_sheet_manifest_document(managed.get("derived_sheet_manifest", {}))
    if managed.get("derived_sheet_manifest_identity") != managed["derived_sheet_manifest"].get("manifest_identity"):
        raise ConditionedIntakeError("Managed Dataset derived-frame manifest identity is invalid.")
    _validate_derived_inventory_binding(
        inventories["derived_inventory"],
        managed["derived_sheet_manifest"],
        derived_manifest,
    )
    accepted = managed.get("accepted_relative_paths")
    covered = managed.get("covered_source_relative_paths")
    if not isinstance(accepted, list) or not isinstance(covered, list):
        raise ConditionedIntakeError("Managed Dataset source coverage is invalid.")
    accepted_paths = [_canonical_relative(str(value or "")) for value in accepted]
    covered_paths = [_canonical_relative(str(value or "")) for value in covered]
    artifact_files = receipt["artifact_manifest"].get("files")
    if not isinstance(artifact_files, list):
        raise ConditionedIntakeError("Managed Dataset artifact manifest is invalid.")
    artifact_paths = {
        _canonical_relative(str(value.get("relative_path") or ""))
        for value in artifact_files
        if isinstance(value, Mapping)
    }
    derived_parents = {str(record["parent_source_relative_path"]) for record in derived_manifest}
    expected_covered = sorted({*accepted_paths, *derived_parents})
    artifact_count = receipt["artifact_manifest"].get("artifact_count")
    if (
        accepted_paths != sorted(set(accepted_paths))
        or covered_paths != sorted(set(covered_paths))
        or not set(accepted_paths) <= artifact_paths
        or not derived_parents <= artifact_paths
        or covered_paths != expected_covered
        or accepted_count != len(accepted_paths) + len(derived_manifest)
        or isinstance(artifact_count, bool)
        or not isinstance(artifact_count, int)
        or quarantined_count != artifact_count - len(covered_paths)
    ):
        raise ConditionedIntakeError("Managed Dataset counts or source coverage are inconsistent.")
    work_relative = _canonical_relative(str(managed.get("work_relative_path") or ""))
    work_parts = PurePosixPath(work_relative).parts
    if (
        len(work_parts) != 3
        or work_parts[:2] != ("datasets", "conditioned_intake_work")
        or _WORK_NAME_PATTERN.fullmatch(work_parts[2]) is None
    ):
        raise ConditionedIntakeError("Managed Dataset transaction root is outside its exact intake namespace.")
    expected_paths = {
        "source_relative_path": f"{work_relative}/source",
        "output_relative_path": f"{work_relative}/datasets/managed",
        "derived_root_relative_path": f"{work_relative}/derived_sprites",
    }
    if any(_canonical_relative(str(managed.get(name) or "")) != expected for name, expected in expected_paths.items()):
        raise ConditionedIntakeError("Managed Dataset roots differ from their unique transaction root.")
    expected_metadata_parent = f"{work_relative}/datasets/source_metadata"
    for name in ("sidecar_relative_path", "grouping_relative_path"):
        relative = _canonical_relative(str(managed.get(name) or ""))
        if PurePosixPath(relative).parent.as_posix() != expected_metadata_parent:
            raise ConditionedIntakeError("Managed Dataset metadata escaped its unique transaction root.")
    for name in ("sidecar_identity", "grouping_identity"):
        file_identity = managed.get(name)
        if (
            not isinstance(file_identity, Mapping)
            or set(file_identity) != {"sha256", "byte_count"}
            or SHA256_PATTERN.fullmatch(str(file_identity.get("sha256") or "")) is None
            or isinstance(file_identity.get("byte_count"), bool)
            or not isinstance(file_identity.get("byte_count"), int)
            or int(file_identity["byte_count"]) < 1
        ):
            raise ConditionedIntakeError("Managed Dataset metadata identity is invalid.")
    for name in ("intake_result_identity", "sidecar_record_identity"):
        if SHA256_PATTERN.fullmatch(str(managed.get(name) or "")) is None:
            raise ConditionedIntakeError("Managed Dataset result identity is invalid.")
    return receipt


def _validate_managed_harvest_document(
    value: Mapping[str, Any],
    *,
    handoff_document: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
) -> None:
    harvest = dict(value)
    digest_fields = _MANAGED_RECEIPT_HARVEST_KEYS - {
        "run_id",
        "backend_capability_issued_at",
        "backend_capability_expires_at",
    }
    if (
        set(harvest) != _MANAGED_RECEIPT_HARVEST_KEYS
        or RUN_ID_PATTERN.fullmatch(str(harvest.get("run_id") or "")) is None
        or any(SHA256_PATTERN.fullmatch(str(harvest.get(name) or "")) is None for name in digest_fields)
        or not isinstance(harvest.get("backend_capability_issued_at"), str)
        or not str(harvest["backend_capability_issued_at"])
        or not isinstance(harvest.get("backend_capability_expires_at"), str)
        or not str(harvest["backend_capability_expires_at"])
        or harvest.get("run_id") != handoff_document.get("run_id")
        or harvest.get("handoff_identity") != stable_hash(dict(handoff_document))
        or harvest.get("request_handoff_identity") != stable_hash(dict(handoff_document))
        or harvest.get("artifact_manifest_identity") != stable_hash(dict(artifact_manifest))
        or harvest.get("artifact_set_identity") != artifact_manifest.get("artifact_set_identity")
        or harvest.get("provenance_identity") != handoff_document.get("provenance_identity")
        or harvest.get("source_evidence_binding_identity") != handoff_document.get("source_evidence_binding_identity")
    ):
        raise ConditionedIntakeError("Managed Dataset Harvest bindings are malformed or inconsistent.")


def _validate_derived_inventory_binding(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    manifest_bytes = (strict_json_dumps(dict(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected: dict[str, dict[str, Any]] = {
        "manifest.json": {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "byte_count": len(manifest_bytes),
        }
    }
    for record in records:
        relative = _canonical_relative(str(record["output_relative_path"]))
        if relative in expected:
            raise ConditionedIntakeError("Managed derived-frame inventory contains a duplicate output path.")
        expected[relative] = {
            "sha256": str(record["encoded_output_sha256"]),
            "byte_count": int(record["encoded_output_byte_count"]),
        }
    expected = dict(sorted(expected.items()))
    if (
        inventory.get("files") != expected
        or inventory.get("file_count") != len(expected)
        or inventory.get("total_bytes") != sum(int(item["byte_count"]) for item in expected.values())
    ):
        raise ConditionedIntakeError("Managed derived-frame inventory differs from its exact manifest and frames.")


def _validate_stored_code_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConditionedIntakeError("Managed Dataset code inventory is malformed.")
    inventory = dict(value)
    identity = inventory.get("inventory_sha256")
    payload = dict(inventory)
    payload.pop("inventory_sha256", None)
    files = inventory.get("files")
    if (
        set(inventory)
        != {
            "schema_version",
            "files",
            "file_count",
            "total_bytes",
            "runtime_dependencies",
            "worker_runtime",
            "inventory_sha256",
        }
        or inventory.get("schema_version") != "spritelab.dataset.conditioned-code-inventory.v3"
        or not isinstance(files, Mapping)
        or isinstance(inventory.get("file_count"), bool)
        or not isinstance(inventory.get("file_count"), int)
        or isinstance(inventory.get("total_bytes"), bool)
        or not isinstance(inventory.get("total_bytes"), int)
        or not isinstance(inventory.get("runtime_dependencies"), Mapping)
        or not isinstance(inventory.get("worker_runtime"), Mapping)
        or SHA256_PATTERN.fullmatch(str(identity or "")) is None
        or stable_hash(payload) != identity
    ):
        raise ConditionedIntakeError("Managed Dataset code inventory is malformed or inconsistent.")
    total_bytes = 0
    normalized_files: dict[str, dict[str, Any]] = {}
    for raw_relative, raw_binding in files.items():
        relative = _canonical_relative(str(raw_relative or ""))
        if (
            not relative.startswith("spritelab/")
            or not isinstance(raw_binding, Mapping)
            or set(raw_binding) != {"sha256", "byte_count"}
            or SHA256_PATTERN.fullmatch(str(raw_binding.get("sha256") or "")) is None
            or isinstance(raw_binding.get("byte_count"), bool)
            or not isinstance(raw_binding.get("byte_count"), int)
            or int(raw_binding["byte_count"]) < 1
            or relative in normalized_files
        ):
            raise ConditionedIntakeError("Managed Dataset code inventory contains an invalid file binding.")
        normalized_files[relative] = dict(raw_binding)
        total_bytes += int(raw_binding["byte_count"])
    if (
        dict(files) != dict(sorted(normalized_files.items()))
        or inventory["file_count"] != len(normalized_files)
        or inventory["total_bytes"] != total_bytes
    ):
        raise ConditionedIntakeError("Managed Dataset code inventory counts or ordering are inconsistent.")
    return inventory


def _validate_managed_inventory_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "files", "file_count", "total_bytes"}:
        raise ConditionedIntakeError("Managed Dataset inventory is malformed.")
    inventory = dict(value)
    files = inventory.get("files")
    file_count = inventory.get("file_count")
    total_bytes = inventory.get("total_bytes")
    if (
        inventory.get("schema_version") != MANAGED_IMPORT_INVENTORY_SCHEMA
        or not isinstance(files, Mapping)
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
    ):
        raise ConditionedIntakeError("Managed Dataset inventory is malformed.")
    normalized: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    for raw_relative, raw_identity in files.items():
        relative = _canonical_relative(str(raw_relative or ""))
        collision = portable_path_collision_key(relative)
        if (
            collision in collision_keys
            or not isinstance(raw_identity, Mapping)
            or set(raw_identity) != {"sha256", "byte_count"}
            or SHA256_PATTERN.fullmatch(str(raw_identity.get("sha256") or "")) is None
            or isinstance(raw_identity.get("byte_count"), bool)
            or not isinstance(raw_identity.get("byte_count"), int)
            or int(raw_identity["byte_count"]) < 0
        ):
            raise ConditionedIntakeError("Managed Dataset inventory contains an invalid file binding.")
        collision_keys.add(collision)
        normalized[relative] = dict(raw_identity)
    normalized = dict(sorted(normalized.items()))
    if (
        dict(files) != normalized
        or file_count != len(normalized)
        or total_bytes != sum(int(item["byte_count"]) for item in normalized.values())
    ):
        raise ConditionedIntakeError("Managed Dataset inventory counts or ordering are inconsistent.")
    return inventory


def _validate_stored_operation_control(value: Any) -> None:
    expected_keys = {
        "schema_version",
        "deadline_monotonic",
        "started_monotonic",
        "initial_budget_seconds",
        "cancellation_probe_bound",
        "paths_exposed",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ConditionedIntakeError("Managed Dataset operation-control evidence is invalid.")
    deadline = value.get("deadline_monotonic")
    started = value.get("started_monotonic")
    budget = value.get("initial_budget_seconds")
    if (
        value.get("schema_version") != _OPERATION_CONTROL_SCHEMA
        or isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or isinstance(started, bool)
        or not isinstance(started, (int, float))
        or isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(deadline))
        or not math.isfinite(float(started))
        or not math.isfinite(float(budget))
        or not 0 < float(budget) <= _MAX_OPERATION_SECONDS
        or float(deadline) - float(started) != float(budget)
        or value.get("cancellation_probe_bound") is not True
        or value.get("paths_exposed") is not False
    ):
        raise ConditionedIntakeError("Managed Dataset operation-control evidence is invalid.")


def _validate_stored_write_confinement(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _WRITE_CONFINEMENT_EVIDENCE_KEYS
        or value.get("schema_version") != "spritelab.write-confinement-evidence.v3"
        or SHA256_PATTERN.fullmatch(str(value.get("root_identity_sha256") or "")) is None
        or value.get("paths_exposed") is not False
    ):
        raise ConditionedIntakeError("Managed Dataset write-confinement evidence is invalid.")
    strategy = value.get("strategy")
    if strategy == LINUX_LANDLOCK_STRATEGY:
        valid = (
            value.get("platform") == "linux"
            and type(value.get("kernel_abi")) is int
            and int(value["kernel_abi"]) >= 3
            and value.get("no_new_privileges") is True
            and type(value.get("handled_access_fs")) is int
            and int(value["handled_access_fs"]) > 0
            and type(value.get("allowed_access_fs")) is int
            and int(value["allowed_access_fs"]) > 0
            and value.get("restricted_token") is False
            and value.get("integrity_level_rid") == 0
            and value.get("mandatory_no_write_up") is False
            and value.get("workspace_integrity_level_rid") == 0
            and value.get("startup_integrity_level_rid") == 0
            and value.get("bootstrap_lowered_before_worker_import") is False
            and value.get("new_thread_integrity_level_rid") == 0
            and value.get("raise_to_low_denied") is False
            and value.get("medium_probe_write_denied") is False
            and value.get("low_world_probe_write_denied") is False
            and value.get("untrusted_world_outside_guaranteed") is False
            and value.get("job_kill_on_close") is False
            and value.get("job_active_process_limit") == 0
        )
    else:
        valid = (
            strategy == WINDOWS_PARENT_ANCHORS_STRATEGY
            and value.get("platform") == "windows"
            and value.get("kernel_abi") == 0
            and value.get("handled_access_fs") == 0
            and value.get("allowed_access_fs") == 0
            and value.get("no_new_privileges") is False
            and type(value.get("restricted_token")) is bool
            and value.get("integrity_level_rid") == 0
            and value.get("mandatory_no_write_up") is True
            and value.get("workspace_integrity_level_rid") == 0
            and value.get("startup_integrity_level_rid") == 4096
            and value.get("bootstrap_lowered_before_worker_import") is True
            and value.get("new_thread_integrity_level_rid") == 0
            and value.get("raise_to_low_denied") is True
            and value.get("medium_probe_write_denied") is True
            and value.get("low_world_probe_write_denied") is True
            and value.get("untrusted_world_outside_guaranteed") is False
            and value.get("job_kill_on_close") is True
            and value.get("job_active_process_limit") == 1
        )
    if not valid:
        raise ConditionedIntakeError("Managed Dataset write-confinement evidence is invalid.")


def _project_relative_path(project: Path, value: Any, *, expected: str) -> Path:
    relative = _canonical_relative(str(value or ""))
    try:
        path = require_confined_path(project.joinpath(*PurePosixPath(relative).parts), project)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Managed Dataset receipt contains an unsafe relative path.") from exc
    if expected == "directory" and not _safe_directory(path):
        raise ConditionedIntakeError("Managed Dataset directory is unavailable or unsafe.")
    if expected not in {"directory", "file"}:
        raise ConditionedIntakeError("Managed Dataset receipt requested an unsupported path kind.")
    return path


def _grouping_file_for(project: Path, source: Path) -> Path:
    from spritelab.product_features.dataset.sidecar import grouping_path

    return grouping_path(project, source)


def _sidecar_license(identifier: str) -> str:
    normalized = identifier.casefold()
    if normalized == "cc0-1.0":
        return "cc0"
    if normalized == "public-domain":
        return "public_domain"
    raise ConditionedIntakeError("The Harvest license is outside the conditioned intake policy.")


def _inventory(root: Path) -> dict[str, Any]:
    if not _safe_directory(root):
        raise ConditionedIntakeError("Managed inventory root is unavailable or unsafe.")
    with AnchoredDirectory(root, root) as anchor:
        return _inventory_from_anchor(anchor)


def _inventory_from_anchor(anchor: AnchoredDirectory) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    anchor.verify()
    _inventory_anchored_tree(anchor, (), files, collision_keys)
    anchor.verify()
    normalized = dict(sorted(files.items()))
    return {
        "schema_version": MANAGED_IMPORT_INVENTORY_SCHEMA,
        "files": normalized,
        "file_count": len(normalized),
        "total_bytes": sum(int(value["byte_count"]) for value in normalized.values()),
    }


def _inventory_anchored_tree(
    anchor: AnchoredDirectory,
    parents: tuple[str, ...],
    files: dict[str, dict[str, Any]],
    collision_keys: set[str],
) -> None:
    names = anchor.names()
    retained_aliases = {name for name in names if _intake_retained_stage_target(anchor, name, names) is not None}
    for name in names:
        metadata = anchor.lstat(name)
        if name in retained_aliases:
            continue
        if stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata):
            with anchor.open_directory(name) as child:
                _inventory_anchored_tree(child, (*parents, name), files, collision_keys)
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink not in {1, 2}
        ):
            raise ConditionedIntakeError("Managed inventory crosses a link or unsupported filesystem entry.")
        if metadata.st_nlink == 2:
            _intake_retained_stage_alias(anchor, name, metadata)
        relative = _canonical_relative(PurePosixPath(*parents, name).as_posix())
        collision = unicodedata.normalize("NFC", relative).casefold()
        if collision in collision_keys:
            raise ConditionedIntakeError("Managed inventory contains a case or Unicode path collision.")
        collision_keys.add(collision)
        content = _read_anchored_regular_bytes(anchor, name, max(_MAX_JSON_BYTES, metadata.st_size))
        files[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "byte_count": metadata.st_size}


def _stable_anchored_file_identity(anchor: AnchoredDirectory, name: str) -> dict[str, Any]:
    before = anchor.lstat(name)
    if not stat.S_ISREG(before.st_mode) or _metadata_is_link_or_reparse(before) or before.st_nlink not in {1, 2}:
        raise ConditionedIntakeError("Managed inventory entries must be owned single-link regular files.")
    if before.st_nlink == 2:
        _intake_retained_stage_alias(anchor, name, before)
    content = _read_anchored_regular_bytes(anchor, name, max(_MAX_JSON_BYTES, before.st_size))
    after = anchor.lstat(name)
    if (
        not stat.S_ISREG(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or after.st_nlink != before.st_nlink
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ConditionedIntakeError("A managed file changed while its identity was computed.")
    return {"sha256": hashlib.sha256(content).hexdigest(), "byte_count": before.st_size}


def _validate_persisted_intake_result(
    output_anchor: AnchoredDirectory,
    expected_identity: str,
) -> None:
    """Bind the child result claim to the immutable managed ``result.json``."""

    if SHA256_PATTERN.fullmatch(expected_identity) is None:
        raise ConditionedIntakeError("Managed Dataset result identity is invalid.")
    result = _read_anchored_json(output_anchor, "result.json")
    if stable_hash(result) != expected_identity:
        raise ConditionedIntakeError("Managed Dataset result identity differs from its persisted result.")


def _validate_persisted_confinement_binding(
    value: Mapping[str, Any],
    work_anchor: AnchoredDirectory,
) -> None:
    """Bind stored confinement evidence to the exact reopened transaction root."""

    work_anchor.verify()
    workspace_identity = DirectoryIdentity.from_stat(work_anchor.directory_metadata())
    try:
        current_strategy = write_confinement_strategy()
    except (WriteConfinementError, WriteConfinementUnavailable) as exc:
        raise ConditionedIntakeError("Managed Dataset write confinement is unavailable.") from exc
    if (
        value.get("root_identity_sha256") != workspace_identity.identity_sha256
        or value.get("strategy") != current_strategy
    ):
        raise ConditionedIntakeError("Managed Dataset write-confinement evidence differs from its transaction root.")
    work_anchor.verify()


def _read_project_anchored_file(
    project_anchor: AnchoredDirectory,
    path: Path,
    project_root: Path,
    max_bytes: int,
) -> bytes:
    try:
        with _open_descendant_directory(project_anchor, path.parent, project_root) as parent_anchor:
            return _read_anchored_regular_bytes(parent_anchor, path.name, max_bytes)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("A trusted project file crossed an unsafe directory boundary.") from exc


def _read_project_anchored_json(
    project_anchor: AnchoredDirectory,
    path: Path,
    project_root: Path,
) -> dict[str, Any]:
    try:
        value = strict_json_loads(_read_project_anchored_file(project_anchor, path, project_root, _MAX_JSON_BYTES))
    except ValueError as exc:
        raise ConditionedIntakeError("A required managed JSON document is invalid.") from exc
    if not isinstance(value, Mapping):
        raise ConditionedIntakeError("A required managed JSON document must be an object.")
    return dict(value)


def _intake_retained_stage_alias(
    anchor: AnchoredDirectory,
    target_name: str,
    metadata: os.stat_result,
) -> str:
    prefix = f".{target_name}.staging-"
    candidates = [
        candidate
        for candidate in anchor.names()
        if re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", candidate) is not None
    ]
    if len(candidates) != 1:
        raise ConditionedIntakeError("A two-link managed file lost its sole retained publication stage.")
    candidate = candidates[0]
    current = anchor.lstat(candidate)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or metadata.st_nlink != 2
        or not stat.S_ISREG(current.st_mode)
        or _metadata_is_link_or_reparse(current)
        or current.st_dev != metadata.st_dev
        or current.st_ino != metadata.st_ino
        or current.st_nlink != 2
        or current.st_size != metadata.st_size
    ):
        raise ConditionedIntakeError("A retained managed publication stage differs from its exact target inode.")
    return candidate


def _intake_retained_stage_target(
    anchor: AnchoredDirectory,
    alias_name: str,
    names: Sequence[str],
) -> str | None:
    marker = ".staging-"
    if not alias_name.startswith(".") or marker not in alias_name:
        return None
    target_name, separator, suffix = alias_name[1:].rpartition(marker)
    if separator != marker or re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
        return None
    if target_name not in names:
        raise ConditionedIntakeError("A retained managed publication stage has no exact target.")
    target = anchor.lstat(target_name)
    if _intake_retained_stage_alias(anchor, target_name, target) != alias_name:
        raise ConditionedIntakeError("A retained managed publication stage is ambiguous.")
    return target_name


def _read_anchored_regular_bytes(anchor: AnchoredDirectory, name: str, max_bytes: int) -> bytes:
    """Read one bounded file, binding any exact retained POSIX stage alias."""

    try:
        before = anchor.lstat(name)
    except OSError as exc:
        raise ConditionedIntakeError("A required managed file is unavailable or unsafe.") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink not in {1, 2}
        or before.st_size > max_bytes
    ):
        raise ConditionedIntakeError("A required managed file is not a bounded single-link regular file.")
    retained_alias = _intake_retained_stage_alias(anchor, name, before) if before.st_nlink == 2 else None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    descriptor = anchor.open_file(name, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != before.st_nlink
        ):
            raise ConditionedIntakeError("A managed file changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(content) > max_bytes or len(content) != before.st_size:
        raise ConditionedIntakeError("A required managed file exceeds its byte limit.")
    after = anchor.lstat(name)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != before.st_nlink
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or opened_after.st_dev != before.st_dev
        or opened_after.st_ino != before.st_ino
        or opened_after.st_size != before.st_size
        or opened_after.st_mtime_ns != before.st_mtime_ns
        or opened_after.st_nlink != before.st_nlink
        or _metadata_is_link_or_reparse(after)
    ):
        raise ConditionedIntakeError("A managed file changed while it was read.")
    if retained_alias is not None:
        alias_after = anchor.lstat(retained_alias)
        if alias_after.st_dev != before.st_dev or alias_after.st_ino != before.st_ino or alias_after.st_nlink != 2:
            raise ConditionedIntakeError("A retained managed publication stage changed while read.")
    return content


def _anchored_regular_file_equals(anchor: AnchoredDirectory, name: str, expected: bytes) -> bool:
    """Return true only for one exact visible immutable receipt candidate."""

    try:
        metadata = anchor.lstat(name)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink not in {1, 2}
            or metadata.st_size != len(expected)
        ):
            return False
        if metadata.st_nlink == 2:
            _intake_retained_stage_alias(anchor, name, metadata)
        return _read_anchored_regular_bytes(anchor, name, len(expected)) == expected
    except (ConditionedIntakeError, OSError, UnsafeFilesystemOperation):
        return False


def _read_anchored_json(anchor: AnchoredDirectory, name: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(_read_anchored_regular_bytes(anchor, name, _MAX_JSON_BYTES))
    except ValueError as exc:
        raise ConditionedIntakeError("A required managed JSON document is invalid.") from exc
    if not isinstance(value, Mapping):
        raise ConditionedIntakeError("A required managed JSON document must be an object.")
    return dict(value)


def _read_anchored_jsonl(anchor: AnchoredDirectory, name: str) -> list[dict[str, Any]]:
    return _parse_jsonl(_read_anchored_regular_bytes(anchor, name, _MAX_JSON_BYTES))


def _parse_jsonl(content: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("JSONL row is not an object")
            rows.append(dict(value))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ConditionedIntakeError("A required managed JSONL document is invalid.") from exc
    return rows


@contextmanager
def _intake_storage(project_root: Path) -> Any:
    """Hold the full project-to-work-root chain for one callback transaction."""

    stack = ExitStack()
    try:
        project_anchor = stack.enter_context(AnchoredDirectory(project_root, project_root))
        datasets_root = project_root / "datasets"
        receipts_root = datasets_root / "conditioned_intake_receipts"
        work_root = datasets_root / "conditioned_intake_work"
        datasets_identity = project_anchor.mkdir("datasets", exist_ok=True)
        if datasets_root in _linux_mount_descendants(project_root):
            raise ConditionedIntakeError("Managed Dataset storage crosses a Linux mount boundary.")
        datasets_anchor = stack.enter_context(project_anchor.open_directory_immovable("datasets"))
        _require_created_anchor_identity(
            datasets_identity,
            datasets_anchor,
            "Managed Dataset storage changed while its Dataset root was anchored.",
        )
        receipts_identity = datasets_anchor.mkdir("conditioned_intake_receipts", exist_ok=True)
        work_root_identity = datasets_anchor.mkdir("conditioned_intake_work", exist_ok=True)
        storage_mounts = set(_linux_mount_descendants(project_root))
        if storage_mounts.intersection({receipts_root, work_root}):
            raise ConditionedIntakeError("Managed Dataset storage crosses a Linux mount boundary.")
        receipts_anchor = stack.enter_context(datasets_anchor.open_directory_immovable("conditioned_intake_receipts"))
        work_root_anchor = stack.enter_context(datasets_anchor.open_directory_immovable("conditioned_intake_work"))
        _require_created_anchor_identity(
            receipts_identity,
            receipts_anchor,
            "Managed Dataset receipt storage changed while it was anchored.",
        )
        _require_created_anchor_identity(
            work_root_identity,
            work_root_anchor,
            "Managed Dataset work storage changed while it was anchored.",
        )
        storage_mounts = set(_linux_mount_descendants(project_root))
        if storage_mounts.intersection({datasets_root, receipts_root, work_root}):
            raise ConditionedIntakeError("Managed Dataset storage changed to a Linux mount boundary.")
    except ConditionedIntakeError:
        stack.close()
        raise
    except (OSError, UnsafeFilesystemOperation) as exc:
        stack.close()
        raise ConditionedIntakeError("Managed Dataset storage is linked, mounted, or unsafe.") from exc
    try:
        yield _IntakeStorage(
            datasets_root=datasets_root,
            receipts_root=receipts_root,
            work_root=work_root,
            project_anchor=project_anchor,
            datasets_anchor=datasets_anchor,
            receipts_anchor=receipts_anchor,
            work_root_anchor=work_root_anchor,
        )
    finally:
        stack.close()


@contextmanager
def _open_descendant_directory(
    root_anchor: AnchoredDirectory,
    target: Path,
    boundary: Path,
    *,
    immovable: bool = False,
) -> Any:
    """Open a descendant exclusively through the already-held root handle."""

    root = Path(os.path.abspath(boundary))
    candidate = Path(os.path.abspath(target))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ConditionedIntakeError("A managed directory escaped the trusted project root.") from exc
    if not relative.parts:
        yield root_anchor
        return
    with ExitStack() as stack:
        current = root_anchor
        for name in relative.parts:
            child = current.open_directory_immovable(name) if immovable else current.open_directory(name)
            current = stack.enter_context(child)
        yield current


def _verify_anchors(*anchors: AnchoredDirectory) -> None:
    for anchor in anchors:
        anchor.verify()


def _require_created_anchor_identity(
    identity: OwnedFileIdentity,
    anchor: AnchoredDirectory,
    message: str,
) -> None:
    """Bind one exclusively observed directory entry to its retained anchor."""

    if not identity.matches(anchor.directory_metadata()):
        raise ConditionedIntakeError(message)
    anchor.verify()


def _publish_anchored_file_noreplace(
    anchor: AnchoredDirectory,
    target_name: str,
    content: bytes,
    *,
    residue_prefix: str,
) -> OwnedFileIdentity:
    """Publish one immutable direct-child file through its exact held inode."""

    temporary: str | None = None
    try:
        descriptor = anchor.open_anonymous_file()
    except (UnsafeFilesystemOperation, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise
        temporary = f".{target_name}.staging-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
        descriptor = anchor.open_file(temporary, flags, 0o600)
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if OwnedFileIdentity.from_stat(os.fstat(handle.fileno())) != identity:
                raise ConditionedIntakeError("A managed staging file changed while it was written.")
        if temporary is not None and not identity.matches(anchor.lstat(temporary)):
            raise ConditionedIntakeError("A managed staging path changed before publication.")
        anchor.publish_held_file_no_replace(
            descriptor,
            temporary,
            target_name,
            identity=identity,
        )
        if not identity.matches(anchor.lstat(target_name)):
            raise ConditionedIntakeError("A managed file identity changed during publication.")
        if _read_anchored_regular_bytes(anchor, target_name, len(content)) != content:
            raise ConditionedIntakeError("Managed file bytes changed during publication.")
    except BaseException:
        anchor.quarantine_if_owned(target_name, identity, prefix=residue_prefix)
        if temporary is not None:
            anchor.quarantine_if_owned(temporary, identity, prefix=residue_prefix)
        raise
    finally:
        os.close(descriptor)
    return identity


@contextmanager
def _receipt_lock(
    anchor: AnchoredDirectory,
    name: str,
    *,
    timeout: float,
    deadline_monotonic: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Any:
    """Serialize one receipt through a persistent anchored lock file."""

    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    for _attempt in range(8):
        try:
            if anchor.lexists(name):
                before = anchor.lstat(name)
                _require_safe_lock_metadata(before)
                descriptor = anchor.open_file(name, flags)
            else:
                descriptor = anchor.open_file(name, flags | os.O_CREAT | os.O_EXCL, 0o600)
                before = os.fstat(descriptor)
            break
        except (FileExistsError, FileNotFoundError):
            continue
    else:
        raise ConditionedIntakeError("The Dataset import receipt lock changed repeatedly.")
    handle: Any = None
    try:
        opened = os.fstat(descriptor)
        _require_same_lock_metadata(before, opened)
        _require_same_lock_metadata(opened, anchor.lstat(name))
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = -1
        if opened.st_size == 0:
            if handle.write(b"0") != 1:
                raise ConditionedIntakeError("The Dataset import receipt lock could not be initialized.")
            handle.flush()
            os.fsync(handle.fileno())
        operation_started = time.monotonic()
        operation_deadline, cancel_probe = _normalize_operation_control(
            deadline_monotonic,
            cancel_requested,
            started_monotonic=operation_started,
        )
        lock_deadline = min(operation_deadline, operation_started + timeout)
        while True:
            try:
                _lock_receipt_handle(handle)
                break
            except (BlockingIOError, OSError):
                _check_operation_control(operation_deadline, cancel_probe)
                if time.monotonic() >= lock_deadline:
                    raise ConditionedIntakeError("Another process is publishing this Dataset import.") from None
                time.sleep(0.01)
        _require_same_lock_metadata(os.fstat(handle.fileno()), anchor.lstat(name))
        try:
            yield
        finally:
            _require_same_lock_metadata(os.fstat(handle.fileno()), anchor.lstat(name))
            _unlock_receipt_handle(handle)
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _require_safe_lock_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or int(getattr(metadata, "st_nlink", 1)) != 1
    ):
        raise ConditionedIntakeError("The Dataset import receipt lock is unsafe.")


def _require_same_lock_metadata(before: os.stat_result, after: os.stat_result) -> None:
    _require_safe_lock_metadata(after)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ConditionedIntakeError("The Dataset import receipt lock identity changed.")


def _lock_receipt_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_receipt_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata) and not path.is_mount()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except OSError:
        return False


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _canonical_relative(value: str) -> str:
    try:
        return canonical_portable_relative_path(value)
    except ValueError as exc:
        raise ConditionedIntakeError("A managed relative path is not canonical or portable.") from exc


def _integer(value: Any, default: int) -> int:
    return value if type(value) is int else default


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "MANAGED_IMPORT_RECEIPT_SCHEMA",
    "ConditionedDatasetImportAdapter",
    "ConditionedIntakeError",
    "load_managed_intake",
    "managed_intake_inventory",
    "read_receipt_bound_derived_frame",
]

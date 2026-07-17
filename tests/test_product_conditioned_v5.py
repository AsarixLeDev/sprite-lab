from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import marshal
import os
import stat
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

import spritelab.product_features.conditioned_v5.audit_runner as conditioned_audit_module
import spritelab.product_features.conditioned_v5.identity as conditioned_identity_module
import spritelab.product_features.conditioned_v5.intake as conditioned_intake_module
import spritelab.product_features.conditioned_v5.service as conditioned_service_module
from spritelab.dataset_v5.identity import decoded_rgba_sha256 as dataset_decoded_rgba_sha256
from spritelab.product_core import ProjectContext
from spritelab.product_features.conditioned_v5 import (
    CandidatePolicy,
    ConditionedDatasetImportAdapter,
    ConditionedDatasetService,
)
from spritelab.product_features.conditioned_v5.identity import (
    TRUSTED_AUDITOR_IDS,
    conditioned_code_inventory,
    conditioned_code_module_paths,
    trusted_auditor_inventory,
)
from spritelab.product_features.conditioned_v5.plugin import create_plugin
from spritelab.product_features.conditioned_v5.service import (
    DATASET_VALIDATION_GATES,
    DATASET_VALIDATION_SCHEMA,
    HANDOFF_SCHEMA,
    LABEL_AUDIT_GATES,
    LABEL_AUDIT_SCHEMA,
)
from spritelab.product_features.conditioned_v5.web import create_router
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_RELATIVE_PATH,
    TRUSTED_CATALOG_SCHEMA,
    CatalogAutomationTermsBinding,
    CatalogEvidenceBinding,
    HarvestSource,
    automation_terms_decision_identity,
    trusted_catalog_identity,
    url_identity,
)
from spritelab.product_features.harvest.catalog_verifier import (
    CATALOG_EVIDENCE_VERIFIER_ID,
    catalog_evidence_verifier_code_identity,
)
from spritelab.product_features.harvest.certification import (
    BACKEND_AUDIT_REPORT_RELATIVE_PATH,
    BACKEND_AUDIT_REPORT_SCHEMA,
    BACKEND_CAPABILITIES_RELATIVE_PATH,
    BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
    REQUIRED_BACKEND_AUDIT_GATES,
    BackendCapabilityEvidence,
    load_backend_capability_evidence,
)
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    DatasetImportRequest,
    HarvestLimits,
    conditioned_dataset_import_callback_binding,
    hardened_backend_code_identity,
    hardened_backend_module_hashes,
    hardened_backend_runtime_dependencies,
)
from spritelab.product_web.app import create_app
from spritelab.training.campaign import stable_hash
from spritelab.utils.pinned_executable import read_executable_identity
from spritelab.utils.safe_fs import atomic_write_bytes
from spritelab.v3.config import DEFAULT_CONFIG


def _png(path: Path, color: tuple[int, int, int, int], marker: int) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(32):
        for x in range(32):
            if abs(x - 16) + abs(y - 16) <= 11:
                shade = ((x // 3 + y // 3 + marker) % 3) * 36
                image.putpixel(
                    (x, y),
                    (
                        (color[0] + shade) % 255,
                        (color[1] + shade // 2) % 255,
                        (color[2] + shade // 3) % 255,
                        255,
                    ),
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _identity(value: Any) -> str:
    return stable_hash(value)


def _controlled_worker_work(tmp_path: Path) -> Path:
    work = tmp_path / "private-work"
    work.mkdir()
    if os.name == "nt":
        conditioned_intake_module.prepare_windows_untrusted_integrity_workspace(work)
    (work / "tmp").mkdir()
    return work


def _source(source_id: str) -> HarvestSource:
    source_url = f"https://catalog.example.test/{source_id}/source"
    license_url = f"https://catalog.example.test/{source_id}/license"
    now = datetime.now(timezone.utc)
    verified_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    expires_at = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    source_content_sha256 = "b" * 64
    automation_terms = CatalogAutomationTermsBinding(
        mode="source_page_no_governing_terms_link",
        decision="NO_PROHIBITION_OBSERVED",
        evidence_url=source_url,
        evidence_request_url_sha256=url_identity(source_url),
        evidence_final_url=source_url,
        evidence_http_status=200,
        evidence_content_sha256=source_content_sha256,
        matched_declaration=None,
        limited_evidence=True,
        decision_identity_sha256=automation_terms_decision_identity(
            mode="source_page_no_governing_terms_link",
            evidence_url=source_url,
            content_sha256=source_content_sha256,
            matched_declaration=None,
            decision="NO_PROHIBITION_OBSERVED",
        ),
        verified_at=verified_at,
        expires_at=expires_at,
    )
    provisional = CatalogEvidenceBinding(
        verifier_id=CATALOG_EVIDENCE_VERIFIER_ID,
        verifier_code_identity_sha256=catalog_evidence_verifier_code_identity(),
        verified_at=verified_at,
        expires_at=expires_at,
        source_request_url_sha256=url_identity(source_url),
        source_final_url=source_url,
        source_http_status=200,
        source_content_sha256=source_content_sha256,
        license_request_url_sha256=url_identity(license_url),
        license_final_url=license_url,
        license_http_status=200,
        license_content_sha256="c" * 64,
        automation_terms=automation_terms,
        attestation_identity_sha256="0" * 64,
    )
    binding = replace(provisional, attestation_identity_sha256=provisional.expected_attestation_identity)
    return HarvestSource(
        source_id=source_id,
        title=f"Source {source_id}",
        creator=f"Creator {source_id}",
        source_page=source_url,
        license_id="cc0-1.0",
        license_evidence_url=license_url,
        license_evidence_text="CC0 public-domain dedication.",
        attribution_text=f"Creator {source_id}",
        acquisition_reference=f"https://downloads.example.test/{source_id}.zip",
        allowed_download_hosts=("downloads.example.test",),
        expected_response_sha256="d" * 64,
        evidence_binding=binding,
        taxonomy_hints=("item",),
    )


def _source_record(source: HarvestSource) -> dict[str, Any]:
    binding = source.evidence_binding
    return {
        "source_id": source.source_id,
        "title": source.title,
        "creator": source.creator,
        "source_page": source.source_page,
        "license_id": source.license_id,
        "license_evidence_url": source.license_evidence_url,
        "license_evidence_text": source.license_evidence_text,
        "attribution_text": source.attribution_text,
        "acquisition_reference": source.acquisition_reference,
        "allowed_download_hosts": list(source.allowed_download_hosts),
        "expected_response_sha256": source.expected_response_sha256,
        "evidence_binding": asdict(binding),
        "zero_cost": True,
        "permissive": True,
        "taxonomy_hints": list(source.taxonomy_hints),
    }


def _capabilities() -> CertifiedBackendCapabilities:
    return CertifiedBackendCapabilities(
        backend_id="audit.backend",
        backend_version="1.0",
        downloader_id="audit.downloader",
        downloader_version="1.0",
        code_identity_sha256=hardened_backend_code_identity(),
        **conditioned_dataset_import_callback_binding(),
        enforces_http_success=True,
        enforces_https_direct_url=True,
        resolves_and_blocks_private_networks=True,
        validates_every_redirect=True,
        enforces_response_mime_allowlist=True,
        enforces_expected_response_hash=True,
        enforces_per_file_hashes=True,
        enforces_file_count_and_byte_limits=True,
        enforces_depth_and_name_policy=True,
        enforces_archive_limits=True,
        enforces_duration_and_cancellation=True,
        enforces_bounded_evidence_fetch=True,
        enforces_quarantine_hash_probe=True,
        enforces_probe_no_decode_extract_import=True,
        enforces_deterministic_evidence_verification=True,
        enforces_transactional_catalog_promotion=True,
        enforces_direct_static_image_derivation=True,
        enforces_retained_anchored_state=True,
        enforces_whole_operation_deadline=True,
        enforces_durable_import_control=True,
        enforces_same_pack_license_and_zero_cost=True,
        enforces_technical_usability_and_pixel_uniqueness=True,
        enforces_non_self_attested_production_bindings=True,
    )


def _write_trust(project: Path, sources: tuple[HarvestSource, ...]) -> None:
    catalog_path = project / TRUSTED_CATALOG_RELATIVE_PATH
    catalog_path.parent.mkdir(parents=True)
    catalog = {
        "schema_version": TRUSTED_CATALOG_SCHEMA,
        "sources": [_source_record(source) for source in sources],
        "catalog_identity": trusted_catalog_identity(sources),
    }
    catalog_path.write_text(json.dumps(catalog, sort_keys=True), encoding="utf-8")
    capabilities = _capabilities()
    modules = hardened_backend_module_hashes()
    runtime_dependencies = hardened_backend_runtime_dependencies()
    issued = datetime.now(timezone.utc) - timedelta(minutes=5)
    report = {
        "schema_version": BACKEND_AUDIT_REPORT_SCHEMA,
        "outcome": "PASS",
        "auditor_id": "independent.audit",
        "audited_at": (issued - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "implementation_identity_sha256": hardened_backend_code_identity(),
        "module_sha256": modules,
        "runtime_dependencies": runtime_dependencies,
        "gate_results": dict.fromkeys(sorted(REQUIRED_BACKEND_AUDIT_GATES), "PASS"),
    }
    report["report_identity"] = _identity(report)
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report_path = project / BACKEND_AUDIT_REPORT_RELATIVE_PATH
    report_path.write_bytes(report_bytes)
    certificate = {
        "schema_version": BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
        "auditor_id": "independent.audit",
        "issued_at": issued.isoformat().replace("+00:00", "Z"),
        "expires_at": (issued + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        "audit_report_relative_path": BACKEND_AUDIT_REPORT_RELATIVE_PATH.as_posix(),
        "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "module_sha256": modules,
        "runtime_dependencies": runtime_dependencies,
        "capabilities": dict(capabilities.__dict__),
    }
    certificate["certificate_identity"] = _identity(certificate)
    (project / BACKEND_CAPABILITIES_RELATIVE_PATH).write_text(
        json.dumps(certificate, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _handoff(
    project: Path,
    run_id: str,
    source: HarvestSource,
    offset: int,
    *,
    sheet: bool = False,
    capability_evidence: BackendCapabilityEvidence | None = None,
) -> dict[str, Any]:
    source_id = source.source_id
    run = project / "harvest_runs" / run_id
    artifacts = run / "artifacts"
    if sheet:
        path = artifacts / "weapons" / "iron_swords.png"
        path.parent.mkdir(parents=True)
        image = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
        for frame_index, left in enumerate((0, 32), start=offset):
            color = ((frame_index * 29) % 255, (frame_index * 53) % 255, (frame_index * 71) % 255, 255)
            for y in range(2, 30):
                for x in range(left + 2, left + 30):
                    image.putpixel((x, y), color)
        image.save(path, format="PNG")
    else:
        names = (
            "weapons/iron_sword.png",
            "tools/copper_pickaxe.png",
            "armor/steel_helmet.png",
            "potions/blue_elixir.png",
        )
        for index, name in enumerate(names, start=offset):
            _png(artifacts / name, ((index * 29) % 255, (index * 53) % 255, (index * 71) % 255, 255), index)
    manifest = scan_artifacts(artifacts, HarvestLimits(max_files=32, max_total_bytes=8 * 1024 * 1024))
    manifest_path = run / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_snapshot = source.to_public_dict()
    license_value = source_snapshot["license"]
    capability_evidence = capability_evidence or load_backend_capability_evidence(project)
    assert capability_evidence is not None
    capabilities = capability_evidence.capabilities
    backend_evidence = {**capability_evidence.to_dict(), "evidence_identity": capability_evidence.identity}
    limits = HarvestLimits(max_files=32, max_total_bytes=8 * 1024 * 1024)
    limits_record = {**limits.to_dict(), "limits_identity": limits.identity}
    request_document = {
        "schema_version": "spritelab.harvest.request.v2",
        "run_id": run_id,
        "source_id": source_id,
        "source_catalog_identity": source.catalog_identity,
        "backend_capability_identity": capabilities.identity,
        "backend_capability_evidence_identity": capability_evidence.identity,
        "backend_capability_certificate_identity": capability_evidence.certificate_identity,
        "backend_capability_audit_report_sha256": capability_evidence.audit_report_sha256,
        "backend_capability_audit_report_identity": capability_evidence.audit_report_identity,
        "backend_capability_issued_at": capability_evidence.issued_at,
        "backend_capability_expires_at": capability_evidence.expires_at,
        "limits_identity": limits.identity,
        "browser_paths_accepted": False,
    }
    authorization = {
        "schema_version": "spritelab.harvest.authorization-receipt.v2",
        "run_id": run_id,
        "source": source_snapshot,
        "backend_capabilities": {**capabilities.to_dict(), "capability_identity": capabilities.identity},
        "backend_capability_evidence": backend_evidence,
        "limits": limits_record,
        "authorizations": {
            "explicit_action": True,
            "zero_cost": True,
            "permissive_license": True,
            "existing_inventory_reviewed": True,
        },
        "network_actions_before_receipt": 0,
        "paths_exposed": False,
    }
    acquisition = {
        "schema_version": "spritelab.harvest.acquisition-receipt.v2",
        "source_id": source_id,
        "source_catalog_identity": source.catalog_identity,
        "source_evidence_binding_identity": source.evidence_binding.identity,
        "backend_capabilities": {**capabilities.to_dict(), "capability_identity": capabilities.identity},
        "backend_capability_evidence": backend_evidence,
        "backend_capability_evidence_identity": capability_evidence.identity,
        "limits": limits_record,
        "actual_response_sha256": hashlib.sha256(b"synthetic-bound-archive").hexdigest(),
        "response_bytes": len(b"synthetic-bound-archive"),
        "response_kind": "archive",
        "direct_image_derivation": None,
        "artifact_manifest_identity": stable_hash(manifest),
    }
    acquisition_identity = stable_hash(acquisition)
    acquisition["acquisition_receipt_identity"] = acquisition_identity
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "run_id": run_id,
        "source_id": source_id,
        "managed_reference": {"kind": "harvest_run", "run_id": run_id},
        "source": source_snapshot,
        "provenance_identity": stable_hash(
            {"source": source_snapshot, "acquisition_receipt_identity": acquisition_identity}
        ),
        "source_evidence_binding_identity": source.evidence_binding.identity,
        "backend_capability_identity": capabilities.identity,
        "backend_capability_evidence": backend_evidence,
        "backend_capability_evidence_identity": capability_evidence.identity,
        "limits_identity": limits.identity,
        "acquisition_receipt_identity": acquisition_identity,
        "acquisition_kind": "archive",
        "direct_image_derivation": None,
        "artifact_manifest_identity": stable_hash(manifest),
        "artifact_set_identity": manifest["artifact_set_identity"],
        "artifact_count": manifest["artifact_count"],
        "usable_count": manifest["usable_count"],
        "quarantined_count": manifest["quarantined_count"],
        "total_bytes": manifest["total_bytes"],
        "taxonomy_counts": manifest["taxonomy_counts"],
        "files": manifest["files"],
        "license": license_value,
        "handoff_ready": True,
        "portable_relative_paths": True,
        "paths_exposed": False,
    }
    (run / "request.json").write_text(json.dumps(request_document, sort_keys=True), encoding="utf-8")
    (run / "authorization_receipt.json").write_text(json.dumps(authorization, sort_keys=True), encoding="utf-8")
    (run / "acquisition_receipt.json").write_text(json.dumps(acquisition, sort_keys=True), encoding="utf-8")
    (run / "handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff


def _service(project: Path, *, activation_loader: Any = None) -> ConditionedDatasetService:
    def campaign_builder(*_args: Any, **_kwargs: Any) -> Any:
        portable = {
            "campaign_id": "conditioned_test",
            "seeds": [731001, 731002, 731003],
            "training": {"max_optimizer_steps": 5000},
            "identities": {"dataset_freeze_hash": _kwargs["activation_manifest_sha256"]},
            "executable": True,
            "launch_authorized": True,
        }
        return SimpleNamespace(
            portable_campaign=portable,
            campaign={"campaign_identity": "e" * 64},
            validation={"launch_ready": True},
        )

    return ConditionedDatasetService(
        project,
        campaign_builder=campaign_builder,
        activation_loader=activation_loader,
        policy=CandidatePolicy(min_images=4, target_images=8, max_images=10, max_source_files=32),
    )


def _import_handoff(
    project: Path,
    run_id: str,
    handoff: dict[str, Any],
    *,
    adapter: ConditionedDatasetImportAdapter | None = None,
) -> str:
    run = project / "harvest_runs" / run_id
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    work_root = project / "datasets" / "conditioned_intake_work"
    prior_work_count = len(list(work_root.iterdir())) if work_root.is_dir() else 0
    before = {
        path.relative_to(run / "artifacts").as_posix(): path.read_bytes()
        for path in sorted((run / "artifacts").rglob("*"))
        if path.is_file()
    }
    adapter = adapter or ConditionedDatasetImportAdapter(project)
    result = adapter.import_harvest(
        DatasetImportRequest(run_id, run / "artifacts", handoff, manifest),
        idempotency_key=f"dataset-import-{run_id}",
    )
    repeated = adapter.import_harvest(
        DatasetImportRequest(run_id, run / "artifacts", handoff, manifest),
        idempotency_key=f"dataset-import-{run_id}",
    )
    assert repeated == result
    assert len(list(work_root.iterdir())) == prior_work_count + 1
    receipt = {
        "schema_version": "spritelab.harvest.dataset-import-receipt.v1",
        "run_id": run_id,
        "idempotency_key": f"dataset-import-{run_id}",
        "callback_id": adapter.callback_id,
        "callback_code_identity_sha256": adapter.code_identity_sha256,
        "dataset_reference": result.dataset_reference,
        "accepted_count": result.accepted_count,
        "quarantined_count": result.quarantined_count,
        "artifact_manifest_identity": stable_hash(manifest),
        "paths_exposed": False,
        "created_at": "2026-07-17T00:00:00Z",
    }
    (run / "dataset_import_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    after = {
        path.relative_to(run / "artifacts").as_posix(): path.read_bytes()
        for path in sorted((run / "artifacts").rglob("*"))
        if path.is_file()
    }
    assert after == before
    loaded = adapter.load_managed_intake(result.dataset_reference)
    assert loaded["dataset_reference"] == result.dataset_reference
    assert loaded["accepted_relative_paths"] or loaded["derived_sheet_records"]
    return result.dataset_reference


def _derived_sheet_fixture(root: Path, *, derived_name: str = "derived") -> dict[str, Any]:
    source_root = root / "source"
    output_root = root / "output"
    derived_root = root / derived_name
    source_root.mkdir(parents=True)
    output_root.mkdir()
    derived_root.mkdir()
    sheet_path = source_root / "weapons" / "iron_swords.png"
    sheet_path.parent.mkdir()
    sheet = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    sheet.paste(Image.new("RGBA", (32, 32), (220, 48, 48, 255)), (0, 0))
    sheet.paste(Image.new("RGBA", (32, 32), (48, 96, 220, 255)), (32, 0))
    sheet.save(sheet_path, format="PNG")
    parent_content = sheet_path.read_bytes()
    parent_sha256 = hashlib.sha256(parent_content).hexdigest()
    parent_decoded_sha256 = dataset_decoded_rgba_sha256(np.asarray(sheet, dtype=np.uint8))
    rows: list[dict[str, Any]] = [
        {
            "item_id": "item_parent_sheet",
            "relative_path": "weapons/iron_swords.png",
            "current_disposition": "sheet_split",
            "byte_sha256": parent_sha256,
            "decoded_rgba_sha256": parent_decoded_sha256,
        }
    ]
    for index, (left, color) in enumerate(((0, (220, 48, 48, 255)), (32, (48, 96, 220, 255)))):
        cell = Image.new("RGBA", (32, 32), color).tobytes()
        decoded_sha256 = dataset_decoded_rgba_sha256(np.frombuffer(cell, dtype=np.uint8).reshape((32, 32, 4)))
        rows.append(
            {
                "item_id": f"item_sheet_frame_{index}",
                "relative_path": f"weapons/iron_swords.png#frame{index:04d}",
                "current_disposition": "accepted",
                "decoded_rgba_sha256": decoded_sha256,
                "width": 32,
                "height": 32,
                "sheet_extraction": {
                    "source_item_id": "item_parent_sheet",
                    "source_relative_path": "weapons/iron_swords.png",
                    "source_byte_sha256": parent_sha256,
                    "source_decoded_rgba_sha256": parent_decoded_sha256,
                    "crop_rectangle": [left, 0, left + 32, 32],
                    "frame_index": index,
                    "output_decoded_rgba_sha256": decoded_sha256,
                    "extraction_policy_version": "spritelab.dataset.sheet_extraction_policy.v1",
                    "source_sheet_modified": False,
                },
            }
        )
    (output_root / "items.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "source_root": source_root,
        "output_root": output_root,
        "derived_root": derived_root,
        "parent_content": parent_content,
        "artifact_manifest": {
            "files": [
                {
                    "relative_path": "weapons/iron_swords.png",
                    "actual_sha256": parent_sha256,
                    "byte_count": len(parent_content),
                    "mime_type": "image/png",
                    "usable": True,
                    "quarantine_reason": None,
                }
            ]
        },
        "source": {"source_id": "source.sheet", "title": "Sheet source", "creator": "Creator"},
        "license": {"identifier": "cc0-1.0", "evidence_url": "https://example.test/license"},
        "run_id": "harvest-sheet-fixture",
    }


def _wait(service: ConditionedDatasetService, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = service.job(job_id)
        if job["status"] != "RUNNING":
            return job
        time.sleep(0.02)
    raise AssertionError("conditioned build did not finish")


def _evidence(kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
    label = kind == "label_audit"
    inventory = trusted_auditor_inventory(kind)
    audit_subjects = candidate["label_audit_subjects"]
    metrics = (
        {
            "audited_record_ids": audit_subjects["required_label_audit_ids"],
            "stratified_sample_ids": audit_subjects["stratified_sample_ids"],
            "low_confidence_ids": audit_subjects["low_confidence_ids"],
            "disagreement_ids": audit_subjects["disagreement_ids"],
            "high_impact_ids": audit_subjects["high_impact_ids"],
            "generic_label_ids": audit_subjects["generic_label_ids"],
            "distributions": audit_subjects["distributions"],
            "quality_rates_basis_points": audit_subjects["quality_rates_basis_points"],
            "recomputed_visual_descriptor_bindings": audit_subjects["visual_descriptor_bindings"],
            "local_pixel_vision_config_identity": audit_subjects["local_pixel_vision_config_identity"],
        }
        if label
        else {
            "split_counts": candidate["split_counts"],
            "category_counts": candidate["category_counts"],
            "source_counts": candidate["source_counts"],
            "benchmark_category_counts": candidate["benchmark_category_counts"],
            "payload_inventory_sha256": candidate["payload_inventory_sha256"],
            "verified_file_count": len(candidate["payload_inventory"]),
            "near_duplicate_recomputation": {
                "algorithm_id": conditioned_service_module.NEAR_DUPLICATE_ALGORITHM,
                "config_identity": conditioned_service_module.NEAR_DUPLICATE_CONFIG_IDENTITY,
                "retained_count": candidate["image_count"],
                "checked_same_category_pairs": sum(
                    int(count) * (int(count) - 1) // 2 for count in candidate["category_counts"].values()
                ),
                "violation_count": 0,
                "gate_identity": candidate["near_duplicate_retained_gate"]["gate_identity"],
            },
        }
    )
    report = {
        "schema_version": LABEL_AUDIT_SCHEMA if label else DATASET_VALIDATION_SCHEMA,
        "verdict": "PASS",
        "independent": True,
        "generated_by_conditioned_workflow": False,
        "auditor": {
            "auditor_id": TRUSTED_AUDITOR_IDS[kind],
            "code_identity_sha256": inventory["inventory_sha256"],
            "implementation_inventory": inventory,
        },
        "bindings": {
            "candidate_identity": candidate["candidate_identity"],
            "payload_inventory_sha256": candidate["payload_inventory_sha256"],
            "image_count": candidate["image_count"],
            "production_code_identity": candidate["production_code_identity"],
            "label_audit_subjects_identity": candidate["label_audit_subjects_identity"],
        },
        "subject_files": candidate["payload_inventory"],
        "checks": dict.fromkeys(LABEL_AUDIT_GATES if label else DATASET_VALIDATION_GATES, "PASS"),
        "audit_subjects": audit_subjects,
        "metrics": metrics,
    }
    return {**report, "audit_run_identity": stable_hash(report)}


def _built_candidate(
    project: Path,
    *,
    production_builder: bool = False,
    write_config: bool = False,
    activation_loader: Any = None,
) -> tuple[ConditionedDatasetService, str, dict[str, Any]]:
    project.mkdir()
    if write_config:
        _write_config(project)
    sources = (_source("source.one"), _source("source.two"))
    _write_trust(project, sources)
    references = [
        _import_handoff(project, run_id, _handoff(project, run_id, source, offset))
        for run_id, source, offset in (
            ("harvest-source-one", sources[0], 1),
            ("harvest-source-two", sources[1], 20),
        )
    ]
    conditioned = (
        ConditionedDatasetService(
            project,
            policy=CandidatePolicy(min_images=4, target_images=8, max_images=10, max_source_files=32),
        )
        if production_builder
        else _service(project, activation_loader=activation_loader)
    )
    started, created = conditioned.start_build(
        references,
        idempotency_key="conditioned-build-fixture-0001",
        explicit_action=True,
    )
    assert created is True
    job = _wait(conditioned, started["job_id"])
    assert job["status"] == "NEEDS_REVIEW", job["message"]
    root = project / "runs" / "v3" / "conditioned-dataset-v5" / started["job_id"]
    candidate = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    return conditioned, started["job_id"], candidate


def _write_config(project: Path) -> bytes:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    payload = yaml.safe_dump(values, sort_keys=False, allow_unicode=True).encode("utf-8")
    atomic_write_bytes(project / "spritelab.yaml", payload)
    return payload


def _attach_evidence_pair(
    service: ConditionedDatasetService,
    job_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    service.attach_evidence(job_id, kind="label_audit", document=_evidence("label_audit", candidate))
    return service.attach_evidence(
        job_id,
        kind="dataset_validation",
        document=_evidence("dataset_validation", candidate),
    )


def _publish_kwargs(candidate: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_identity": candidate["candidate_identity"],
        "label_audit_sha256": job["evidence"]["label_audit"]["sha256"],
        "dataset_validation_sha256": job["evidence"]["dataset_validation"]["sha256"],
        "authorization_id": "freeze-authorization-fixture-0001",
        "explicit_action": True,
        "authorize_one_time_freeze": True,
    }


def _activation_loader(config: Any, **_kwargs: Any) -> Any:
    freeze = config.root / config.values["dataset"]["freeze_manifest"]
    campaign = config.root / config.values["training"]["campaign_config"]
    return SimpleNamespace(
        ready=True,
        freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
        campaign_config_sha256=hashlib.sha256(campaign.read_bytes()).hexdigest(),
        campaign={
            "campaign_identity": "e" * 64,
            "seeds": [731001, 731002, 731003],
            "training": {"max_optimizer_steps": 5_000},
        },
    )


def _activation_kwargs(
    candidate: dict[str, Any],
    publication: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "candidate_identity": candidate["candidate_identity"],
        "publication_identity_sha256": publication["publication_identity_sha256"],
        "activation_manifest_sha256": publication["activation_manifest_sha256"],
        "campaign_config_sha256": publication["campaign_config_sha256"],
        "campaign_identity_sha256": publication["campaign_identity_sha256"],
        "expected_config_sha256": config_sha256,
        "activation_authorization_id": "activation-authorization-fixture-0001",
        "explicit_action": True,
        "authorize_dataset_freeze": True,
        "authorize_training": True,
    }


def _published_configured_candidate(
    project: Path,
) -> tuple[ConditionedDatasetService, str, dict[str, Any], dict[str, Any], bytes]:
    service, job_id, candidate = _built_candidate(
        project,
        write_config=True,
        activation_loader=_activation_loader,
    )
    before_config = (project / "spritelab.yaml").read_bytes()
    evidence_job = _attach_evidence_pair(service, job_id, candidate)
    published = service.publish(job_id, **_publish_kwargs(candidate, evidence_job))
    return service, job_id, candidate, published["publication"], before_config


def _module_relative_path(module_name: str) -> str | None:
    package_root = Path(conditioned_identity_module.__file__).resolve(strict=True).parents[2]
    if module_name == "spritelab":
        return "__init__.py"
    if not module_name.startswith("spritelab."):
        return None
    relative = module_name.removeprefix("spritelab.").replace(".", "/")
    if (package_root / f"{relative}.py").is_file():
        return f"{relative}.py"
    if (package_root / relative / "__init__.py").is_file():
        return f"{relative}/__init__.py"
    return None


def test_conditioned_code_inventory_closes_direct_imports_parents_integrations_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = conditioned_code_inventory()
    relative_files = {name.removeprefix("spritelab/") for name in inventory["files"]}
    package_root = Path(conditioned_identity_module.__file__).resolve(strict=True).parents[2]
    for relative in (
        "product_features/conditioned_v5/service.py",
        "product_features/conditioned_v5/intake.py",
    ):
        tree = ast.parse((package_root / relative).read_text(encoding="utf-8"))
        direct_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("spritelab")
        }
        direct_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("spritelab")
        )
        for module_name in direct_modules:
            imported = _module_relative_path(module_name)
            assert imported is not None
            assert imported in relative_files

    resource_files = {relative for relative in relative_files if not relative.endswith(".py")}
    assert resource_files == {
        "config/hallucination_denylist.yaml",
        "config/sheet_mappings.yaml",
        "config/source_profiles.yaml",
        "config/taxonomy.yaml",
    }
    for relative in relative_files - resource_files:
        tree = ast.parse((package_root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
                or not node.args[0].value.startswith("spritelab.")
            ):
                continue
            function = node.func
            is_dynamic_import = (isinstance(function, ast.Name) and function.id in {"__import__", "import_module"}) or (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
                and isinstance(function.value, ast.Name)
                and function.value.id == "importlib"
            )
            if is_dynamic_import:
                imported = _module_relative_path(node.args[0].value)
                assert imported is not None
                assert imported in relative_files
        directory = Path(relative).parent
        while directory.parts:
            initializer = (directory / "__init__.py").as_posix()
            if (package_root / initializer).is_file():
                assert initializer in relative_files
            directory = directory.parent
    assert "__init__.py" in relative_files
    dependencies = inventory["runtime_dependencies"]
    assert set(dependencies) == {
        "anyio",
        "idna",
        "numpy",
        "Pillow",
        "PyYAML",
        "setuptools",
        "starlette",
        "typing_extensions",
    }
    assert dependencies["numpy"]["version"] == pytest.importorskip("numpy").__version__
    assert dependencies["Pillow"]["version"] == pytest.importorskip("PIL").__version__
    assert dependencies["PyYAML"]["version"] == yaml.__version__
    for dependency in dependencies.values():
        assert dependency["schema_version"] == "spritelab.runtime.installed-distribution-inventory.v2"
        assert dependency["file_count"] == len(dependency["files"]) > 0
        assert dependency["record_file_count"] == len(dependency["record_declared_paths"]) > 0
        assert set(dependency["record_declared_paths"]) <= set(dependency["files"])
        assert dependency["unrecorded_file_count"] == dependency["file_count"] - dependency["record_file_count"]
        assert dependency["owned_roots"]
        assert dependency["total_bytes"] == sum(item["byte_count"] for item in dependency["files"].values())
        assert len(dependency["record_sha256"]) == 64
        assert len(dependency["inventory_sha256"]) == 64
        assert dependency["paths_exposed"] is False
    assert {
        "product_features/conditioned_v5/legacy_worker.py",
        "utils/write_confinement.py",
    } <= relative_files
    worker_runtime = inventory["worker_runtime"]
    assert worker_runtime["schema_version"] == "spritelab.dataset.conditioned-worker-runtime.v1"
    assert worker_runtime["paths_exposed"] is False
    assert len(worker_runtime["executable_sha256"]) == 64
    assert len(worker_runtime["environment_policy_identity"]) == 64
    assert worker_runtime["environment_policy"]["interpreter_flags"] == ["-I", "-S", "-B", "-c"]
    assert worker_runtime["runtime_dependency_inventory_identities"] == {
        name: value["inventory_sha256"] for name, value in dependencies.items()
    }


def test_runtime_dependency_inventory_detects_same_size_installed_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = tmp_path / "installation"
    package = installation / "fake_dependency"
    metadata = installation / "fake_dependency-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    runtime_file = package / "__init__.py"
    runtime_file.write_bytes(b"trusted-runtime-A\n")
    record = metadata / "RECORD"
    record.write_text(
        "fake_dependency/__init__.py,,\nfake_dependency-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
        newline="",
    )

    class FakeDistribution:
        def __init__(self) -> None:
            self._path = metadata
            self.version = "1.0"
            self.metadata = {"Name": "fake-dependency"}

        @staticmethod
        def locate_file(value: str) -> Path:
            return installation / value

    monkeypatch.setattr(
        conditioned_identity_module.importlib.metadata, "distribution", lambda _name: FakeDistribution()
    )
    before = conditioned_identity_module._installed_distribution_inventory("fake-dependency")
    runtime_file.write_bytes(b"tamperd-runtime-B\n")
    assert runtime_file.stat().st_size == before["files"]["fake_dependency/__init__.py"]["byte_count"]
    after = conditioned_identity_module._installed_distribution_inventory("fake-dependency")
    assert after["inventory_sha256"] != before["inventory_sha256"]
    assert (
        after["files"]["fake_dependency/__init__.py"]["sha256"]
        != before["files"]["fake_dependency/__init__.py"]["sha256"]
    )


def test_runtime_dependency_inventory_binds_unrecorded_valid_pyc_and_supplemental_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = tmp_path / "installation"
    package = installation / "fake_dependency"
    metadata = installation / "fake_dependency-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    record = metadata / "RECORD"
    record.write_text(
        "fake_dependency/__init__.py,,\nfake_dependency-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
        newline="",
    )

    class FakeDistribution:
        def __init__(self) -> None:
            self._path = metadata
            self.version = "1.0"
            self.metadata = {"Name": "fake-dependency"}

        @staticmethod
        def locate_file(value: str) -> Path:
            return installation / value

    monkeypatch.setattr(
        conditioned_identity_module.importlib.metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    before = conditioned_identity_module.installed_distribution_inventory("fake-dependency")
    supplemental = package / "native-data.bin"
    supplemental.write_bytes(b"supplemental-runtime")
    after_supplemental = conditioned_identity_module.installed_distribution_inventory("fake-dependency")
    assert "fake_dependency/native-data.bin" in after_supplemental["files"]
    assert after_supplemental["unrecorded_file_count"] == 1
    assert after_supplemental["inventory_sha256"] != before["inventory_sha256"]

    pycache = package / "__pycache__"
    pycache.mkdir()
    code = compile("VALUE = 2\n", "fake_dependency/__init__.py", "exec")
    valid_pyc = importlib.util.MAGIC_NUMBER + b"\x00" * 12 + marshal.dumps(code)
    (pycache / "__init__.cpython-test.pyc").write_bytes(valid_pyc)
    after_pyc = conditioned_identity_module.installed_distribution_inventory("fake-dependency")
    assert "fake_dependency/__pycache__/__init__.cpython-test.pyc" in after_pyc["files"]
    assert after_pyc["unrecorded_file_count"] == 2
    assert after_pyc["inventory_sha256"] != after_supplemental["inventory_sha256"]


@pytest.mark.skipif(os.name != "nt", reason="Windows held directories deny rename at the kernel handle")
def test_runtime_dependency_inventory_blocks_owned_root_rename_swap_while_anchored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = tmp_path / "installation"
    package = installation / "fake_dependency"
    metadata = installation / "fake_dependency-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    (package / "__init__.py").write_bytes(b"VALUE = 'trusted'\n")
    (metadata / "RECORD").write_text(
        "fake_dependency/__init__.py,,\nfake_dependency-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
        newline="",
    )
    replacement = installation / "replacement"
    replacement.mkdir()
    (replacement / "__init__.py").write_bytes(b"VALUE = 'malicious'\n")

    class FakeDistribution:
        def __init__(self) -> None:
            self._path = metadata
            self.version = "1.0"
            self.metadata = {"Name": "fake-dependency"}

        @staticmethod
        def locate_file(value: str) -> Path:
            return installation / value

    monkeypatch.setattr(
        conditioned_identity_module.importlib.metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    original_scan = conditioned_identity_module._scan_distribution_owned_anchor
    attempted = False

    def scan_with_swap_attempt(*args: Any, **kwargs: Any) -> None:
        nonlocal attempted
        if not attempted and kwargs.get("relative_directory") == "fake_dependency":
            attempted = True
            with pytest.raises(OSError):
                package.rename(installation / "held-original")
        original_scan(*args, **kwargs)

    monkeypatch.setattr(conditioned_identity_module, "_scan_distribution_owned_anchor", scan_with_swap_attempt)
    inventory = conditioned_identity_module.installed_distribution_inventory("fake-dependency")
    assert attempted is True
    assert package.is_dir()
    assert (
        inventory["files"]["fake_dependency/__init__.py"]["sha256"]
        == hashlib.sha256(b"VALUE = 'trusted'\n").hexdigest()
    )

    monkeypatch.setattr(
        conditioned_identity_module,
        "_PRODUCTION_INTEGRATION_MODULES",
        ("evaluation/cli.py",),
    )
    expanded = set(conditioned_code_module_paths())
    assert "evaluation/cli.py" in expanded
    assert "evaluation/promotion_decision.py" in expanded
    assert "evaluation/__init__.py" in expanded


def test_controlled_worker_ignores_hostile_python_startup_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _controlled_worker_work(tmp_path)
    attacker = tmp_path / "attacker-pythonpath"
    attacker.mkdir()
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    startup = f"from pathlib import Path\nPath({str(sentinel)!r}).write_bytes(b'compromised')\n"
    (attacker / "sitecustomize.py").write_text(startup, encoding="utf-8")
    (attacker / "usercustomize.py").write_text(startup, encoding="utf-8")
    (attacker / "startup.py").write_text(startup, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(attacker))
    monkeypatch.setenv("PYTHONHOME", str(attacker))
    monkeypatch.setenv("PYTHONSTARTUP", str(attacker / "startup.py"))
    monkeypatch.setenv("SPRITELAB_PROJECT_ROOT", str(attacker))
    monkeypatch.setenv("HF_HOME", str(attacker))

    environment = conditioned_identity_module.controlled_worker_environment(work / "tmp")
    assert set(environment) <= {"TEMP", "TMP", "TMPDIR", "SystemRoot", "WINDIR"}
    assert not any(name.startswith("PYTHON") or name in {"HF_HOME", "SPRITELAB_PROJECT_ROOT"} for name in environment)
    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    strategy = conditioned_intake_module.write_confinement_strategy()
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=strategy,
            workspace_identity=identity,
            request_payload={},
            code_inventory=conditioned_code_inventory(),
        )

    assert sentinel.read_bytes() == b"unchanged"


def test_controlled_worker_bootstrap_transport_is_exact_and_fits_windows_command_line() -> None:
    import spritelab.utils.write_confinement as write_confinement_module

    source = conditioned_identity_module.WORKER_BOOTSTRAP_SOURCE
    command_source = conditioned_identity_module.WORKER_BOOTSTRAP_COMMAND_SOURCE
    compressed_hex = zlib.compress(source.encode("utf-8"), level=9).hex()

    assert repr(compressed_hex) in command_source
    assert zlib.decompress(bytes.fromhex(compressed_hex)).decode("utf-8") == source

    representative_path = "C:\\" + "p" * 256
    command = [
        sys.executable,
        *conditioned_identity_module.WORKER_INTERPRETER_FLAGS,
        command_source,
        *([representative_path, "123", "456"] * 12),
    ]
    wrapped = write_confinement_module._windows_untrusted_bootstrap_arguments(command)
    assert len(subprocess.list2cmdline(list(wrapped))) < 32_767


def test_controlled_worker_launch_policy_rejects_runtime_source_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert conditioned_identity_module.controlled_worker_launch_arguments()[-1]
    monkeypatch.setattr(conditioned_identity_module, "WORKER_BOOTSTRAP_COMMAND_SOURCE", "raise SystemExit(0)")

    with pytest.raises(conditioned_identity_module.ConditionedCodeIdentityError, match="audited policy"):
        conditioned_identity_module.controlled_worker_launch_arguments()


def test_windows_outer_bootstrap_rejects_runtime_source_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.utils.write_confinement as write_confinement_module

    command = [sys.executable, "-I", "-S", "-B", "-c", "pass", "argument"]
    monkeypatch.setattr(write_confinement_module, "WINDOWS_UNTRUSTED_BOOTSTRAP_SOURCE", "pass")

    with pytest.raises(write_confinement_module.WriteConfinementError, match="audited source"):
        write_confinement_module._windows_untrusted_bootstrap_arguments(command)


@pytest.mark.parametrize("swapped_name", ["worker", "helper"])
def test_controlled_worker_rejects_swapped_preconfinement_code_bytes(
    tmp_path: Path,
    swapped_name: str,
) -> None:
    work = _controlled_worker_work(tmp_path)
    worker_source = Path(conditioned_intake_module.__file__).with_name("legacy_worker.py")
    helper_source = (
        Path(conditioned_intake_module.__file__).resolve(strict=True).parents[2] / "utils" / "write_confinement.py"
    )
    worker = tmp_path / "audited-worker.py"
    helper = tmp_path / "audited-helper.py"
    worker.write_bytes(worker_source.read_bytes())
    helper.write_bytes(helper_source.read_bytes())

    def binding(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}

    inventory = {
        "worker_runtime": conditioned_identity_module.controlled_worker_runtime(),
        "files": {
            "spritelab/product_features/conditioned_v5/legacy_worker.py": binding(worker),
            "spritelab/utils/write_confinement.py": binding(helper),
        },
    }
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    malicious = f"from pathlib import Path\nPath({str(sentinel)!r}).write_bytes(b'compromised')\n".encode()
    (worker if swapped_name == "worker" else helper).write_bytes(malicious)
    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    strategy = conditioned_intake_module.write_confinement_strategy()

    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=strategy,
            workspace_identity=identity,
            request_payload={},
            code_inventory=inventory,
            _worker_path=worker,
            _helper_path=helper,
        )

    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("swapped_module", ["intake", "transitive"])
def test_controlled_worker_inventory_finder_rejects_same_size_module_swap_before_import(
    tmp_path: Path,
    swapped_module: str,
) -> None:
    source_root = tmp_path / "audited-src"
    work = _controlled_worker_work(tmp_path)
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    module_bytes: dict[str, bytes] = {
        "spritelab/__init__.py": b"# package\n",
        "spritelab/product_features/__init__.py": b"# package\n",
        "spritelab/product_features/conditioned_v5/__init__.py": b"# package\n",
        "spritelab/utils/__init__.py": b"# package\n",
        "spritelab/utils/write_confinement.py": b"BOUND_HELPER = True\n",
        "spritelab/product_features/conditioned_v5/legacy_worker.py": (
            b"from spritelab.product_features.conditioned_v5 import intake\n"
        ),
    }
    if swapped_module == "transitive":
        module_bytes["spritelab/product_features/conditioned_v5/intake.py"] = (
            b"from spritelab.conditioned_transitive_guard import VALUE\n"
        )
        target_relative = "spritelab/conditioned_transitive_guard.py"
    else:
        target_relative = "spritelab/product_features/conditioned_v5/intake.py"
    malicious = f"from pathlib import Path\nPath({str(sentinel)!r}).write_bytes(b'compromised')\n".encode()
    module_bytes[target_relative] = b"VALUE = 1\n" + b"#" * (len(malicious) - len(b"VALUE = 1\n"))
    for relative, payload in module_bytes.items():
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def binding(payload: bytes) -> dict[str, Any]:
        return {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}

    inventory = {
        "worker_runtime": conditioned_identity_module.controlled_worker_runtime(),
        "files": {relative: binding(payload) for relative, payload in module_bytes.items()},
    }

    def swap() -> None:
        assert len(malicious) == len(module_bytes[target_relative])
        source_root.joinpath(*PurePosixPath(target_relative).parts).write_bytes(malicious)

    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=conditioned_intake_module.write_confinement_strategy(),
            workspace_identity=identity,
            request_payload={},
            code_inventory=inventory,
            _source_root=source_root,
            _before_worker_launch=swap,
        )
    assert sentinel.read_bytes() == b"unchanged"


def test_controlled_worker_inventory_finder_rejects_unbound_dynamic_spritelab_import(tmp_path: Path) -> None:
    source_root = tmp_path / "audited-src"
    work = _controlled_worker_work(tmp_path)
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    module_bytes = {
        "spritelab/__init__.py": b"# package\n",
        "spritelab/product_features/__init__.py": b"# package\n",
        "spritelab/product_features/conditioned_v5/__init__.py": b"# package\n",
        "spritelab/utils/__init__.py": b"# package\n",
        "spritelab/utils/write_confinement.py": b"BOUND_HELPER = True\n",
        "spritelab/product_features/conditioned_v5/legacy_worker.py": (
            b"import importlib\nimportlib.import_module('spritelab.unbound_payload')\n"
        ),
    }
    for relative, payload in module_bytes.items():
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    unbound = source_root / "spritelab" / "unbound_payload.py"
    unbound.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_bytes(b'compromised')\n",
        encoding="utf-8",
    )
    inventory = {
        "worker_runtime": conditioned_identity_module.controlled_worker_runtime(),
        "files": {
            relative: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
            for relative, payload in module_bytes.items()
        },
    }
    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=conditioned_intake_module.write_confinement_strategy(),
            workspace_identity=identity,
            request_payload={},
            code_inventory=inventory,
            _source_root=source_root,
        )
    assert sentinel.read_bytes() == b"unchanged"


def test_controlled_worker_rejects_interpreter_substitution_between_inventory_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _controlled_worker_work(tmp_path)
    interpreter = tmp_path / ("python-copy.exe" if os.name == "nt" else "python-copy")
    original = Path(sys.executable).resolve(strict=True)
    interpreter.write_bytes(original.read_bytes())
    interpreter.chmod(stat.S_IMODE(original.stat().st_mode))
    executable_identity = read_executable_identity(interpreter)
    inventory = conditioned_code_inventory()
    runtime = {
        **dict(inventory["worker_runtime"]),
        "executable_sha256": executable_identity.executable_sha256,
        "executable_byte_count": executable_identity.byte_count,
        "executable_metadata_sha256": executable_identity.metadata_sha256,
    }
    inventory = {**inventory, "worker_runtime": runtime}
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")

    def substitute() -> None:
        payload = bytearray(interpreter.read_bytes())
        payload[len(payload) // 2] ^= 1
        interpreter.write_bytes(payload)

    monkeypatch.setattr(conditioned_intake_module, "controlled_worker_executable", lambda: interpreter)
    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=conditioned_intake_module.write_confinement_strategy(),
            workspace_identity=identity,
            request_payload={},
            code_inventory=inventory,
            _before_worker_launch=substitute,
        )
    assert sentinel.read_bytes() == b"unchanged"


def test_controlled_worker_timeout_terminates_descendant_group_before_outside_write(tmp_path: Path) -> None:
    source_root = tmp_path / "audited-src"
    work = _controlled_worker_work(tmp_path)
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    descendant = (
        f"import time\nfrom pathlib import Path\ntime.sleep(0.5)\nPath({str(sentinel)!r}).write_bytes(b'compromised')\n"
    )
    worker = (
        "import subprocess,sys,time\n"
        "try:\n"
        f" subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        "except OSError:\n"
        " pass\n"
        "time.sleep(30)\n"
    ).encode()
    module_bytes = {
        "spritelab/__init__.py": b"# package\n",
        "spritelab/product_features/__init__.py": b"# package\n",
        "spritelab/product_features/conditioned_v5/__init__.py": b"# package\n",
        "spritelab/utils/__init__.py": b"# package\n",
        "spritelab/utils/write_confinement.py": b"BOUND_HELPER = True\n",
        "spritelab/product_features/conditioned_v5/legacy_worker.py": worker,
    }
    for relative, payload in module_bytes.items():
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    inventory = {
        "worker_runtime": conditioned_identity_module.controlled_worker_runtime(),
        "files": {
            relative: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
            for relative, payload in module_bytes.items()
        },
    }
    identity = conditioned_intake_module.DirectoryIdentity.from_stat(work.stat())
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="time limit"):
        conditioned_intake_module._run_legacy_intake_child(
            work,
            strategy=conditioned_intake_module.write_confinement_strategy(),
            workspace_identity=identity,
            request_payload={},
            code_inventory=inventory,
            _source_root=source_root,
            _worker_timeout_seconds=0.2,
        )
    time.sleep(0.7)
    assert sentinel.read_bytes() == b"unchanged"


def test_controlled_worker_response_rejects_private_path_strings(tmp_path: Path) -> None:
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="private path"):
        conditioned_intake_module._assert_pathless_worker_response(
            {"schema_version": "response.v1", "result": {"leak": str(tmp_path)}}
        )


def test_controlled_workspace_audit_rejects_an_outside_hard_link(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    try:
        os.link(sentinel, work / "injected.bin")
    except OSError:
        pytest.skip("hard links are unavailable in this test session")
    with conditioned_intake_module.AnchoredDirectory(work, work) as anchor:
        identity = conditioned_intake_module.DirectoryIdentity.from_stat(anchor.directory_metadata())
        with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="non-owned"):
            conditioned_intake_module._audit_writable_workspace(anchor, identity)
    assert sentinel.read_bytes() == b"unchanged"


def test_trusted_auditor_identity_binds_transitive_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    label_before = trusted_auditor_inventory("label_audit")
    validation_before = trusted_auditor_inventory("dataset_validation")
    original_read = conditioned_identity_module._read_single_link

    label_helper = (
        Path(conditioned_identity_module.__file__).resolve(strict=True).parents[2] / "dataset_v5" / "identity.py"
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            conditioned_identity_module,
            "_read_single_link",
            lambda path: (
                original_read(path) + b"\n# injected helper drift\n" if path == label_helper else original_read(path)
            ),
        )
        assert trusted_auditor_inventory("label_audit")["inventory_sha256"] != label_before["inventory_sha256"]
        assert (
            trusted_auditor_inventory("dataset_validation")["inventory_sha256"] != validation_before["inventory_sha256"]
        )

    validation_helper = (
        Path(conditioned_identity_module.__file__).resolve(strict=True).parents[2]
        / "harvest"
        / "semantic_extractors.py"
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            conditioned_identity_module,
            "_read_single_link",
            lambda path: (
                original_read(path) + b"\n# injected helper drift\n"
                if path == validation_helper
                else original_read(path)
            ),
        )
        assert (
            trusted_auditor_inventory("dataset_validation")["inventory_sha256"] != validation_before["inventory_sha256"]
        )

    original_dependency = conditioned_identity_module._installed_distribution_inventory
    with monkeypatch.context() as patch:
        patch.setattr(
            conditioned_identity_module,
            "_installed_distribution_inventory",
            lambda distribution: (
                {
                    **original_dependency(distribution),
                    "inventory_sha256": "f" * 64,
                }
                if distribution == "numpy"
                else original_dependency(distribution)
            ),
        )
        assert trusted_auditor_inventory("label_audit")["inventory_sha256"] != label_before["inventory_sha256"]
        assert (
            trusted_auditor_inventory("dataset_validation")["inventory_sha256"] != validation_before["inventory_sha256"]
        )


@pytest.mark.parametrize(
    "value",
    [
        "C:drive-relative.png",
        "C:/absolute.png",
        "\\\\?\\C:\\device.png",
        "\\\\.\\PhysicalDrive0",
        "safe/path\\mixed.png",
        "safe/../escape.png",
        "safe/CON.png",
        "safe/con.txt",
        "safe/CLOCK$.json",
        "safe/name:stream.png",
        "safe/less<than.png",
        "safe/greater>than.png",
        'safe/quote"name.png',
        "safe/pipe|name.png",
        "safe/question?.png",
        "safe/star*.png",
        "safe/control\x1f.png",
        "safe/delete\x7f.png",
        "safe/trailing-dot.png.",
        "safe/trailing-space.png ",
        "/rooted.png",
        "//server/share.png",
        "safe/e\u0301.png",
    ],
)
def test_independent_auditor_rejects_noncanonical_portable_paths(value: str) -> None:
    assert conditioned_audit_module._portable_relative_path(value) is False
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError):
        conditioned_intake_module._canonical_relative(value)
    with pytest.raises(conditioned_service_module.ConditionedDatasetError):
        conditioned_service_module._canonical_relative(value)


def test_independent_auditor_accepts_only_canonical_posix_relative_paths() -> None:
    assert conditioned_audit_module._portable_relative_path("harvest/source.one/artifacts/weapon/red_sword.png") is True
    assert conditioned_audit_module._is_private_path("C:drive-relative.png") is True
    assert conditioned_audit_module._is_private_path("safe/path\\mixed.png") is True


def test_local_pixel_vision_is_deterministic_factual_and_nonsemantic() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 10:22] = (220, 48, 48, 255)

    first = conditioned_service_module._local_pixel_vision(rgba)
    second = conditioned_service_module._local_pixel_vision(rgba.copy())

    assert first == second
    assert first["algorithm_id"] == "local_pixel_vision_v1"
    assert first["config_identity"] == conditioned_service_module.LOCAL_PIXEL_VISION_CONFIG_IDENTITY
    assert first["semantic_category_inferred"] is False
    assert first["provider_contacted"] is False
    assert first["model_weights_loaded"] is False
    assert first["metrics"]["alpha_bbox"] == [10, 8, 22, 24]
    assert first["metrics"]["dominant_coarse_color"] == "red"
    assert "dominant_red" in first["visual_tags"]
    changed_config = {**conditioned_service_module.LOCAL_PIXEL_VISION_CONFIG, "alpha_threshold": 254}
    assert stable_hash(changed_config) != conditioned_service_module.LOCAL_PIXEL_VISION_CONFIG_IDENTITY


def test_near_duplicate_collapse_uses_category_alpha_bbox_and_perceptual_evidence(tmp_path: Path) -> None:
    def record(name: str, color: tuple[int, int, int], *, category: str, object_name: str) -> Any:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[8:24, 10:22] = (*color, 255)
        alpha = rgba[:, :, 3]
        visual = conditioned_service_module._local_pixel_vision(rgba)
        return conditioned_service_module._SourceRecord(
            relative_path=f"{name}.png",
            path=tmp_path / f"{name}.png",
            byte_count=100,
            byte_sha256=hashlib.sha256(f"bytes:{name}".encode()).hexdigest(),
            pixel_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
            alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
            alpha_bitmap=np.packbits(alpha == 255).tobytes(),
            alpha_bbox=(10, 8, 22, 24),
            perceptual_hash=conditioned_service_module._perceptual_hash(rgba),
            category=category,
            object_name=object_name,
            tokens=(object_name,),
            source_id="source.one",
            source_title="Source",
            creator="Creator",
            license_id="cc0-1.0",
            license_evidence={},
            visual_descriptor=visual,
            visual_tags=tuple(visual["visual_tags"]),
        )

    sword = record("sword", (210, 40, 40), category="weapon", object_name="sword")
    differently_named_recolor = record("blade", (190, 30, 30), category="weapon", object_name="blade")
    icon = record("icon", (180, 45, 30), category="icon", object_name="blade")
    kept, exclusions, evidence = conditioned_service_module._deduplicate_records(
        (sword, differently_named_recolor, icon)
    )

    assert len(kept) == 2
    assert exclusions == ["near_duplicate"]
    assert evidence[0]["disposition"] == "near_duplicate"
    assert evidence[0]["metric_evidence"]["same_taxonomy_category"] is True
    assert evidence[0]["metric_evidence"]["is_near_duplicate"] is True
    assert (
        evidence[0]["metric_evidence"]["config_identity"] == conditioned_service_module.NEAR_DUPLICATE_CONFIG_IDENTITY
    )
    gate = conditioned_service_module._retained_near_duplicate_gate(kept)
    assert gate["ok"] is True


def test_conditioned_intake_publishes_receipt_bound_canonical_sheet_frames(tmp_path: Path) -> None:
    fixture = _derived_sheet_fixture(tmp_path / "first")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
        validated = conditioned_intake_module._validate_derived_sheet_tree(
            manifest,
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
    assert fixture["parent_content"] == (fixture["source_root"] / "weapons" / "iron_swords.png").read_bytes()
    assert manifest["record_count"] == 2
    assert validated == manifest["records"]
    assert len({record["source_group_identity"] for record in validated}) == 1
    assert len({record["source_provenance_identity"] for record in validated}) == 1
    assert [record["crop_rectangle"] for record in validated] == [[0, 0, 32, 32], [32, 0, 64, 32]]
    assert all(
        record["recipe_identity"] == conditioned_intake_module.DERIVED_SHEET_RECIPE_IDENTITY for record in validated
    )
    assert all(record["source_derived_not_augmentation"] is True for record in validated)
    assert sorted(path.name for path in (fixture["derived_root"] / "frames").iterdir()) == sorted(
        f"{record['derivation_identity']}.png" for record in validated
    )

    second = _derived_sheet_fixture(tmp_path / "second")
    with (
        conditioned_intake_module.AnchoredDirectory(second["source_root"], second["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(second["output_root"], second["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(second["derived_root"], second["derived_root"]) as derived_anchor,
    ):
        repeated = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=second["artifact_manifest"],
            source=second["source"],
            license_record=second["license"],
            run_id=second["run_id"],
        )
    assert repeated == manifest
    for record in validated:
        relative = PurePosixPath(record["output_relative_path"])
        assert (
            fixture["derived_root"].joinpath(*relative.parts).read_bytes()
            == second["derived_root"].joinpath(*relative.parts).read_bytes()
        )


def test_conditioned_intake_derived_tree_rejects_republication_and_byte_drift(tmp_path: Path) -> None:
    fixture = _derived_sheet_fixture(tmp_path / "fixture")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
        with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="not empty"):
            conditioned_intake_module._publish_derived_sheet_tree(
                output_anchor=output_anchor,
                source_anchor=source_anchor,
                derived_anchor=derived_anchor,
                artifact_manifest=fixture["artifact_manifest"],
                source=fixture["source"],
                license_record=fixture["license"],
                run_id=fixture["run_id"],
            )
    record = manifest["records"][0]
    frame = fixture["derived_root"].joinpath(*PurePosixPath(record["output_relative_path"]).parts)
    frame.write_bytes(frame.read_bytes() + b"drift")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
        pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="exact parent/crop recipe"),
    ):
        conditioned_intake_module._validate_derived_sheet_tree(
            manifest,
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )


def test_conditioned_receipt_rejects_derived_inventory_manifest_frame_mismatch(tmp_path: Path) -> None:
    fixture = _derived_sheet_fixture(tmp_path / "fixture")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
        inventory = conditioned_intake_module._inventory_from_anchor(derived_anchor)
    mismatched = json.loads(json.dumps(inventory))
    first_output = manifest["records"][0]["output_relative_path"]
    mismatched["files"][first_output]["sha256"] = "0" * 64
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="exact manifest and frames"):
        conditioned_intake_module._validate_derived_inventory_binding(
            mismatched,
            manifest,
            manifest["records"],
        )


def test_conditioned_builder_consumes_held_derived_bytes_with_parent_grouping(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    fixture = _derived_sheet_fixture(project / "managed")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
    source = {
        "artifact_manifest": fixture["artifact_manifest"],
        "accepted_relative_paths": [],
        "covered_source_relative_paths": ["weapons/iron_swords.png"],
        "artifacts_root": fixture["source_root"],
        "derived_root": fixture["derived_root"],
        "derived_sheet_records": manifest["records"],
        "source_id": "source.sheet",
        "source_title": "Sheet source",
        "creator": "Creator",
        "license_id": "cc0-1.0",
        "license_evidence": fixture["license"],
    }
    records, exclusions = _service(project)._inspect_records(source)
    assert exclusions == []
    assert len(records) == 2
    assert len({conditioned_service_module._source_group(record) for record in records}) == 1
    assert all(record.derivation is not None for record in records)
    assert all(hashlib.sha256(record.content).hexdigest() == record.byte_sha256 for record in records)

    first = records[0]
    first.path.write_bytes(first.path.read_bytes() + b"later-path-drift")
    imported = conditioned_service_module.import_png_bytes_as_dataset_item(
        first.content,
        source_name=first.path.name,
        options=conditioned_service_module.ImportOptions(
            max_palette_slots=32,
            allow_quantize_overcolor=False,
            quantize_overcolor=False,
            allow_nearest_resize=False,
            infer_role_map=True,
            canonicalize_palette=True,
        ),
        default_category=first.category,
        default_tags=first.tokens,
    )
    assert imported.errors == ()
    assert imported.bundle is not None
    assert hashlib.sha256(first.content).hexdigest() == first.byte_sha256


def test_conditioned_builder_rejects_parent_drift_after_managed_load(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    fixture = _derived_sheet_fixture(project / "managed")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
    source = {
        "artifact_manifest": fixture["artifact_manifest"],
        "accepted_relative_paths": [],
        "covered_source_relative_paths": ["weapons/iron_swords.png"],
        "artifacts_root": fixture["source_root"],
        "derived_root": fixture["derived_root"],
        "derived_sheet_records": manifest["records"],
        "source_id": "source.sheet",
        "source_title": "Sheet source",
        "creator": "Creator",
        "license_id": "cc0-1.0",
        "license_evidence": fixture["license"],
    }
    parent = fixture["source_root"] / "weapons" / "iron_swords.png"
    parent.write_bytes(parent.read_bytes() + b"parent-drift")
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as captured:
        _service(project)._inspect_records(source)
    assert captured.value.code == "derived_frame_changed"
    assert os.fspath(project) not in captured.value.public_message


def test_conditioned_auditor_reconstructs_receipt_bound_parent_and_rejects_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    work_name = f"intake-{'1' * 32}"
    work = project / "datasets" / "conditioned_intake_work" / work_name
    fixture = _derived_sheet_fixture(work, derived_name="derived_sprites")
    with (
        conditioned_intake_module.AnchoredDirectory(fixture["source_root"], fixture["source_root"]) as source_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["output_root"], fixture["output_root"]) as output_anchor,
        conditioned_intake_module.AnchoredDirectory(fixture["derived_root"], fixture["derived_root"]) as derived_anchor,
    ):
        manifest = conditioned_intake_module._publish_derived_sheet_tree(
            output_anchor=output_anchor,
            source_anchor=source_anchor,
            derived_anchor=derived_anchor,
            artifact_manifest=fixture["artifact_manifest"],
            source=fixture["source"],
            license_record=fixture["license"],
            run_id=fixture["run_id"],
        )
        source_inventory = conditioned_intake_module._inventory_from_anchor(source_anchor)
        derived_inventory = conditioned_intake_module._inventory_from_anchor(derived_anchor)
    reference = f"dataset.{'2' * 24}"
    source_inventory_identity = stable_hash(source_inventory)
    derived_inventory_identity = stable_hash(derived_inventory)
    receipt_payload = {
        "schema_version": "spritelab.dataset.conditioned-import-receipt.v2",
        "dataset_reference": reference,
        "harvest": {"run_id": fixture["run_id"]},
        "source": fixture["source"],
        "license": fixture["license"],
        "artifact_manifest": fixture["artifact_manifest"],
        "managed": {
            "work_relative_path": f"datasets/conditioned_intake_work/{work_name}",
            "source_relative_path": f"datasets/conditioned_intake_work/{work_name}/source",
            "derived_root_relative_path": f"datasets/conditioned_intake_work/{work_name}/derived_sprites",
            "source_inventory": source_inventory,
            "source_inventory_sha256": source_inventory_identity,
            "derived_inventory": derived_inventory,
            "derived_inventory_sha256": derived_inventory_identity,
            "derived_sheet_manifest": manifest,
        },
    }
    receipt = {**receipt_payload, "receipt_identity": stable_hash(receipt_payload)}
    receipts_root = project / "datasets" / "conditioned_intake_receipts"
    receipts_root.mkdir(parents=True)
    (receipts_root / f"{reference}.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    binding = {
        "dataset_reference": reference,
        "harvest_run_id": fixture["run_id"],
        "managed_intake_receipt_identity": receipt["receipt_identity"],
        "managed_source_inventory_sha256": source_inventory_identity,
        "managed_derived_inventory_sha256": derived_inventory_identity,
        "derived_sheet_manifest_identity": manifest["manifest_identity"],
        "source_id": fixture["source"]["source_id"],
        "title": fixture["source"]["title"],
        "creator": fixture["source"]["creator"],
        "license_id": fixture["license"]["identifier"],
        "license_evidence": fixture["license"],
        "source_document": fixture["source"],
        "license_document": fixture["license"],
    }
    derivation = manifest["records"][0]
    frame = fixture["derived_root"].joinpath(*PurePosixPath(derivation["output_relative_path"]).parts)
    with Image.open(frame) as opened:
        rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    record = {
        "source_id": fixture["source"]["source_id"],
        "source_pack": fixture["source"]["title"],
        "creator": fixture["source"]["creator"],
        "license_id": fixture["license"]["identifier"],
        "source_relative_path": derivation["semantic_relative_path"],
        "source_sha256": derivation["encoded_output_sha256"],
        "source_byte_count": derivation["encoded_output_byte_count"],
        "source_group": derivation["source_group_identity"],
        "source_derivation": derivation,
    }
    candidate = {"input_bindings": [binding]}
    dataset = {"sprites": {"sprite-1": {"record": record, "rgba": rgba}}}
    job_root = project / "runs" / "v3" / "conditioned-dataset-v5" / "conditioned-test"
    job_root.mkdir(parents=True)

    conditioned_audit_module._verify_source_derivation(
        record,
        {"source_derivation": derivation},
        rgba=rgba,
        source_binding=binding,
    )
    conditioned_audit_module._verify_parent_bound_derivations(
        project,
        job_root,
        candidate,
        dataset,
        progress=lambda *_args: None,
        cancelled=lambda: False,
    )

    parent = fixture["source_root"] / "weapons" / "iron_swords.png"
    parent.write_bytes(parent.read_bytes() + b"raw-parent-drift")
    with pytest.raises(conditioned_audit_module.IndependentAuditError) as captured:
        conditioned_audit_module._verify_parent_bound_derivations(
            project,
            job_root,
            candidate,
            dataset,
            progress=lambda *_args: None,
            cancelled=lambda: False,
        )
    assert os.fspath(project) not in captured.value.public_message


def test_conditioned_sheet_intake_publishes_one_staged_tree_and_builder_reloads_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    worker_runtime = {"schema_version": "test.conditioned-worker-runtime.v1", "paths_exposed": False}
    inventory_file = Path(conditioned_intake_module.__file__).read_bytes()
    inventory_payload = {
        "schema_version": "spritelab.dataset.conditioned-code-inventory.v3",
        "files": {
            "spritelab/product_features/conditioned_v5/intake.py": {
                "sha256": hashlib.sha256(inventory_file).hexdigest(),
                "byte_count": len(inventory_file),
            }
        },
        "file_count": 1,
        "total_bytes": len(inventory_file),
        "runtime_dependencies": {},
        "worker_runtime": worker_runtime,
    }
    code_inventory = {**inventory_payload, "inventory_sha256": stable_hash(inventory_payload)}
    monkeypatch.setattr(
        conditioned_identity_module,
        "conditioned_code_inventory",
        lambda: code_inventory,
    )
    monkeypatch.setattr(
        conditioned_identity_module,
        "conditioned_callback_runtime_inventory",
        lambda _inventory: {"runtime_identity_sha256": "0" * 64},
    )
    backend_identity = "f" * 64
    callback_binding = {
        "dataset_import_callback_id": ConditionedDatasetImportAdapter.callback_id,
        "dataset_import_callback_code_identity_sha256": code_inventory["inventory_sha256"],
        "dataset_import_callback_runtime_identity_sha256": "0" * 64,
    }
    test_module = sys.modules[__name__]
    monkeypatch.setattr(test_module, "hardened_backend_code_identity", lambda: backend_identity)
    monkeypatch.setattr(test_module, "hardened_backend_module_hashes", lambda: {})
    monkeypatch.setattr(test_module, "hardened_backend_runtime_dependencies", lambda: {})
    monkeypatch.setattr(test_module, "conditioned_dataset_import_callback_binding", lambda: callback_binding)
    source = _source("sheet-source")
    _write_trust(project, (source,))
    report_path = project / BACKEND_AUDIT_REPORT_RELATIVE_PATH
    certificate_path = project / BACKEND_CAPABILITIES_RELATIVE_PATH
    report_bytes = report_path.read_bytes()
    report_document = json.loads(report_bytes)
    certificate_document = json.loads(certificate_path.read_bytes())
    capability_evidence = BackendCapabilityEvidence(
        capabilities=_capabilities(),
        auditor_id=str(report_document["auditor_id"]),
        audited_at=str(report_document["audited_at"]),
        issued_at=str(certificate_document["issued_at"]),
        expires_at=str(certificate_document["expires_at"]),
        audit_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        audit_report_identity=str(report_document["report_identity"]),
        certificate_identity=str(certificate_document["certificate_identity"]),
        implementation_identity_sha256=backend_identity,
    )
    run_id = "harvest-sheet-worker"
    handoff = _handoff(
        project,
        run_id,
        source,
        1,
        sheet=True,
        capability_evidence=capability_evidence,
    )
    adapter = object.__new__(ConditionedDatasetImportAdapter)
    adapter.project_root = project
    adapter.code_inventory = code_inventory
    adapter.code_identity_sha256 = code_inventory["inventory_sha256"]
    adapter.runtime_inventory = {}
    adapter.runtime_identity_sha256 = "0" * 64
    adapter._catalog_loader = conditioned_intake_module.load_trusted_catalog
    adapter._capability_evidence_loader = lambda _root: capability_evidence
    monkeypatch.setattr(
        conditioned_intake_module,
        "conditioned_code_inventory",
        lambda: code_inventory,
    )
    monkeypatch.setattr(conditioned_intake_module, "controlled_worker_runtime", lambda: worker_runtime)

    def run_bound_intake_in_process(
        work: Path,
        *,
        strategy: str,
        workspace_identity: Any,
        request_payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        result = conditioned_intake_module._run_legacy_intake_in_process(
            work=work,
            source_root=work / "source",
            output_root=work / "datasets" / "managed",
            source=request_payload["source"],
            license_record=request_payload["license"],
            artifact_sha256=request_payload["artifact_sha256"],
            run_id=request_payload["run_id"],
        )
        windows = strategy == conditioned_intake_module.WINDOWS_PARENT_ANCHORS_STRATEGY
        evidence = {
            "schema_version": "spritelab.write-confinement-evidence.v3",
            "strategy": strategy,
            "platform": "windows" if windows else "linux",
            "kernel_abi": 0 if windows else 3,
            "root_identity_sha256": workspace_identity.identity_sha256,
            "handled_access_fs": 0 if windows else 1,
            "allowed_access_fs": 0 if windows else 1,
            "no_new_privileges": not windows,
            "restricted_token": False,
            "integrity_level_rid": 0,
            "mandatory_no_write_up": windows,
            "workspace_integrity_level_rid": 0,
            "startup_integrity_level_rid": 4096 if windows else 0,
            "bootstrap_lowered_before_worker_import": windows,
            "new_thread_integrity_level_rid": 0,
            "raise_to_low_denied": windows,
            "medium_probe_write_denied": windows,
            "low_world_probe_write_denied": windows,
            "untrusted_world_outside_guaranteed": False,
            "job_kill_on_close": windows,
            "job_active_process_limit": 1 if windows else 0,
            "paths_exposed": False,
        }
        return {
            "schema_version": conditioned_intake_module._LEGACY_RESPONSE_SCHEMA,
            "ok": True,
            "result": result,
            "write_confinement": evidence,
            "paths_exposed": False,
        }

    monkeypatch.setattr(conditioned_intake_module, "_run_legacy_intake_child", run_bound_intake_in_process)
    observed: list[tuple[str, str]] = []
    original_rename = conditioned_intake_module.AnchoredDirectory.rename_held_directory_noreplace

    def observe_tree_rename(
        parent: conditioned_intake_module.AnchoredDirectory,
        child: conditioned_intake_module.AnchoredDirectory,
        destination_name: str,
    ) -> None:
        if destination_name == "derived_sprites":
            assert child.directory.name.startswith(".derived-sprites-")
            assert not (parent.directory / destination_name).exists()
            assert set(child.names()) == {"frames", "manifest.json"}
            before = child.directory.name
            original_rename(parent, child, destination_name)
            assert child.directory == parent.directory / destination_name
            observed.append((before, child.directory.name))
            return
        original_rename(parent, child, destination_name)

    monkeypatch.setattr(
        conditioned_intake_module.AnchoredDirectory,
        "rename_held_directory_noreplace",
        observe_tree_rename,
    )
    reference = _import_handoff(project, run_id, handoff, adapter=adapter)
    loaded = adapter.load_managed_intake(reference)
    assert len(observed) == 1
    assert observed[0][0].startswith(".derived-sprites-")
    assert observed[0][1] == "derived_sprites"
    assert len(loaded["derived_sheet_records"]) == 2
    assert loaded["accepted_relative_paths"] == []
    work = loaded["derived_root"].parent
    assert not any(path.name.startswith(".derived-sprites-") for path in work.iterdir())
    records, exclusions = _service(project)._inspect_records(loaded)
    assert exclusions == []
    assert len(records) == 2
    assert len({conditioned_service_module._source_group(record) for record in records}) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows parent anchors use non-share-delete handles")
def test_conditioned_intake_parent_swap_fails_before_outside_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = _source("swap-source")
    _write_trust(project, (source,))
    run_id = "harvest-swap-boundary"
    handoff = _handoff(project, run_id, source, 1)
    run = project / "harvest_runs" / run_id
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"preserve")
    blocked: list[str] = []

    def inject_swaps(work: Path, **_kwargs: Any) -> dict[str, Any]:
        derived_staging = [path for path in work.iterdir() if path.name.startswith(".derived-sprites-")]
        assert len(derived_staging) == 1
        writable_roots = (
            work,
            work / "tmp",
            work / "source",
            work / "datasets",
            work / "datasets" / "managed",
            derived_staging[0],
            work / "datasets" / "source_metadata",
            work / "datasets" / "source_metadata" / ".transactions",
            work / "runs",
            work / "runs" / "v3",
        )
        for index, target in enumerate(writable_roots):
            moved = target.parent / f".injected-move-{index}"
            try:
                os.replace(target, moved)
            except OSError:
                blocked.append(target.relative_to(work).as_posix() if target != work else ".")
                continue
            os.replace(moved, target)
            raise AssertionError(f"writable root rename was not blocked: {target.name}")
        raise conditioned_intake_module.ConditionedIntakeError("injected fixed-root swaps were blocked")

    monkeypatch.setattr(conditioned_intake_module, "_run_legacy_intake_child", inject_swaps)
    adapter = ConditionedDatasetImportAdapter(project)
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError, match="fixed-root"):
        adapter.import_harvest(
            DatasetImportRequest(run_id, run / "artifacts", handoff, manifest),
            idempotency_key="dataset-import-swap-boundary",
        )
    assert blocked[:5] == [
        ".",
        "tmp",
        "source",
        "datasets",
        "datasets/managed",
    ]
    assert len(blocked) == 10
    assert blocked[5].startswith(".derived-sprites-")
    assert blocked[6:] == [
        "datasets/source_metadata",
        "datasets/source_metadata/.transactions",
        "runs",
        "runs/v3",
    ]
    assert sentinel.read_bytes() == b"preserve"
    assert sorted(path.relative_to(outside).as_posix() for path in outside.rglob("*")) == ["sentinel.bin"]


def test_conditioned_intake_receipt_rename_is_commit_point_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = _source("commit-source")
    _write_trust(project, (source,))
    run_id = "harvest-commit-retry"
    handoff = _handoff(project, run_id, source, 1)
    run = project / "harvest_runs" / run_id
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    request = DatasetImportRequest(run_id, run / "artifacts", handoff, manifest)
    adapter = ConditionedDatasetImportAdapter(project)
    monkeypatch.setattr(conditioned_intake_module, "conditioned_code_inventory", lambda: adapter.code_inventory)
    original_rename = conditioned_intake_module.AnchoredDirectory.rename
    injected = False

    def fail_after_receipt_rename(
        anchor: Any,
        source_name: str,
        destination_name: str,
        *,
        replace: bool,
    ) -> None:
        nonlocal injected
        original_rename(anchor, source_name, destination_name, replace=replace)
        if (
            not injected
            and anchor.directory == project / "datasets" / "conditioned_intake_receipts"
            and destination_name.startswith("dataset.")
        ):
            injected = True
            raise OSError("injected fault after the receipt namespace commit")

    monkeypatch.setattr(conditioned_intake_module.AnchoredDirectory, "rename", fail_after_receipt_rename)
    with pytest.raises(conditioned_intake_module.ConditionedIntakeError) as caught:
        adapter.import_harvest(request, idempotency_key="dataset-import-commit-retry")
    receipts = list((project / "datasets" / "conditioned_intake_receipts").glob("dataset.*.json"))
    assert len(receipts) == 1, f"unexpected pre-commit failure: {caught.value!r}; cause={caught.value.__cause__!r}"
    work_directories = [
        path for path in (project / "datasets" / "conditioned_intake_work").iterdir() if path.name.startswith("intake-")
    ]
    assert len(work_directories) == 1
    assert not (work_directories[0] / "failure.json").exists()

    repeated = adapter.import_harvest(request, idempotency_key="dataset-import-commit-retry")
    assert repeated.dataset_reference == receipts[0].stem
    assert (
        len(
            [
                path
                for path in (project / "datasets" / "conditioned_intake_work").iterdir()
                if path.name.startswith("intake-")
            ]
        )
        == 1
    )


def test_preview_build_evidence_and_publication_are_bound_and_portable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    sources = (_source("source.one"), _source("source.two"))
    _write_trust(project, sources)
    handoffs = {
        "harvest-source-one": _handoff(project, "harvest-source-one", sources[0], 1),
        "harvest-source-two": _handoff(project, "harvest-source-two", sources[1], 20),
    }
    references = [_import_handoff(project, run_id, handoff) for run_id, handoff in handoffs.items()]
    service = _service(project)

    preview = service.preview(references)
    assert preview["ready_to_build"] is True
    assert preview["selected_images"] == 8
    assert preview["source_counts"] == {"source.one": 4, "source.two": 4}
    assert preview["labels_are_human_truth"] is False

    started, created = service.start_build(
        references, idempotency_key="conditioned-build-test-0001", explicit_action=True
    )
    assert created is True
    job = _wait(service, started["job_id"])
    assert job["status"] == "NEEDS_REVIEW", job["message"]
    assert job["candidate"]["image_count"] == 8
    root = project / "runs" / "v3" / "conditioned-dataset-v5" / started["job_id"]
    candidate = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert candidate["dataset_references"] == sorted(references)
    assert len(candidate["managed_intake_receipt_identities"]) == 2
    phase7 = root / "candidate" / "phase7"
    assert all((phase7 / f"{split}.npz").is_file() for split in ("train", "val", "test"))
    assert str(project) not in (phase7 / "training_manifest.jsonl").read_text(encoding="utf-8")
    assert json.loads((phase7 / "split_integrity_report.json").read_text(encoding="utf-8"))["ok"] is True

    job = service.attach_evidence(started["job_id"], kind="label_audit", document=_evidence("label_audit", candidate))
    job = service.attach_evidence(
        started["job_id"], kind="dataset_validation", document=_evidence("dataset_validation", candidate)
    )
    published = service.publish(
        started["job_id"],
        candidate_identity=candidate["candidate_identity"],
        label_audit_sha256=job["evidence"]["label_audit"]["sha256"],
        dataset_validation_sha256=job["evidence"]["dataset_validation"]["sha256"],
        authorization_id="freeze-authorization-test-0001",
        explicit_action=True,
        authorize_one_time_freeze=True,
    )
    assert published["status"] == "COMPLETE"
    publication = published["publication"]
    activation = project / publication["activation_manifest"]
    activation_value = json.loads(activation.read_text(encoding="utf-8"))
    assert activation_value["schema_version"] == "spritelab.dataset.freeze.conditioned.v5"
    assert activation_value["publication_inventory"]["file_count"] > 10
    assert set(next(iter(activation_value["publication_inventory"]["files"].values()))) == {
        "sha256",
        "byte_count",
    }
    assert publication["campaign_seeds"] == [731001, 731002, 731003]
    assert publication["campaign_steps"] == 5000
    assert publication["configuration_activated"] is False
    assert not (project / "spritelab.yaml").exists()


def test_plugin_and_api_reject_browser_paths_and_expose_navigation(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    project = tmp_path / "project"
    project.mkdir()
    config_bytes = _write_config(project)
    source = _source("source.one")
    _write_trust(project, (source,))
    handoff = _handoff(project, "harvest-source-one", source, 1)
    reference = _import_handoff(project, "harvest-source-one", handoff)
    service = _service(project)
    plugin = create_plugin(service_factory=lambda _context: service)
    assert plugin.navigation[0].path == "/dataset-v5"
    assert plugin.api_prefixes == ("/dataset-v5/api",)

    runs = project / "runs-shell"
    runs.mkdir()
    shell = TestClient(create_app(ProjectContext(project, runs_directory=runs), plugins=(plugin,)))
    page = shell.get("/dataset-v5")
    assert page.status_code == 200
    assert 'meta name="spritelab-csrf"' in page.text
    assert 'id="cv5-config-sha"' in page.text
    assert 'id="cv5-activation-auth"' in page.text
    assert 'id="cv5-authorize-dataset"' in page.text
    assert 'id="cv5-authorize-training"' in page.text
    assert 'id="cv5-activate"' in page.text
    static = shell.get("/dataset-v5/static/conditioned-v5.js")
    assert static.status_code == 200
    assert "expected_config_sha256" in static.text
    assert "Training was not started" in static.text
    assert static.text.count('request("/dataset-v5/api/inventory")') == 1
    assert static.text.count('get("cv5-refresh")?.addEventListener') == 1
    assert "if (busy.size > 0) return" in static.text
    no_csrf = shell.post("/dataset-v5/api/preview", json={"dataset_references": [reference]})
    assert no_csrf.status_code == 403
    assert no_csrf.json()["error_code"] == "csrf_validation_failed"

    app = FastAPI()
    app.include_router(create_router(ProjectContext(project), service=service))
    client = TestClient(app)
    assert client.get("/dataset-v5").status_code == 200
    inventory = client.get("/dataset-v5/api/inventory")
    assert inventory.status_code == 200
    assert inventory.json()["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    invalid_activation = client.post(
        "/dataset-v5/api/jobs/conditioned-00000000000000000000/activate",
        json={"config_path": "C:/private"},
    )
    assert invalid_activation.status_code == 422
    assert invalid_activation.json()["error_code"] == "invalid_conditioned_v5_payload"
    rejected = client.post(
        "/dataset-v5/api/jobs",
        json={
            "dataset_references": [reference],
            "output_path": "C:/private",
            "idempotency_key": "conditioned-build-test-0002",
            "explicit_action": True,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == "invalid_conditioned_v5_payload"


def test_independent_evidence_requires_trusted_auditor_exact_schema_and_coverage(tmp_path: Path) -> None:
    service, job_id, candidate = _built_candidate(tmp_path / "project")

    arbitrary = _evidence("label_audit", candidate)
    arbitrary["auditor"]["auditor_id"] = "independent.arbitrary"
    arbitrary["audit_run_identity"] = stable_hash(
        {key: value for key, value in arbitrary.items() if key != "audit_run_identity"}
    )
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as arbitrary_error:
        service.attach_evidence(job_id, kind="label_audit", document=arbitrary)
    assert arbitrary_error.value.code == "evidence_auditor"

    extra = _evidence("label_audit", candidate)
    extra["unexpected"] = True
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as extra_error:
        service.attach_evidence(job_id, kind="label_audit", document=extra)
    assert extra_error.value.code == "evidence_schema"

    private = _evidence("label_audit", candidate)
    private["auditor"]["auditor_id"] = "C:/Users/Private/auditor"
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as private_error:
        service.attach_evidence(job_id, kind="label_audit", document=private)
    assert private_error.value.code == "evidence_schema"

    incomplete = _evidence("label_audit", candidate)
    incomplete["metrics"]["audited_record_ids"] = incomplete["metrics"]["audited_record_ids"][:-1]
    incomplete["audit_run_identity"] = stable_hash(
        {key: value for key, value in incomplete.items() if key != "audit_run_identity"}
    )
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as coverage_error:
        service.attach_evidence(job_id, kind="label_audit", document=incomplete)
    assert coverage_error.value.code == "evidence_metrics"


def test_publication_rolls_back_exact_outputs_after_campaign_rename_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    service, job_id, candidate = _built_candidate(project)
    job = _attach_evidence_pair(service, job_id, candidate)
    original_publish = conditioned_service_module._publish_directory_noreplace

    def fault_after_campaign_rename(anchor: Any, source_name: str, target_name: str, identity: Any) -> None:
        original_publish(anchor, source_name, target_name, identity)
        if anchor.directory == service.campaigns_root:
            raise OSError("injected fault after campaign rename")

    monkeypatch.setattr(conditioned_service_module, "_publish_directory_noreplace", fault_after_campaign_rename)
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as error:
        service.publish(job_id, **_publish_kwargs(candidate, job))
    assert error.value.code == "publication_failed"
    assert not list(service.datasets_root.glob("conditioned-v5-*"))
    assert not list(service.campaigns_root.glob("conditioned-v5-*"))
    assert service.job(job_id)["status"] == "FAILED"


def test_publication_rollback_quarantines_owned_content_drift_with_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    service, job_id, candidate = _built_candidate(project)
    job = _attach_evidence_pair(service, job_id, candidate)
    original_publish = conditioned_service_module._publish_directory_noreplace

    def drift_after_campaign_rename(anchor: Any, source_name: str, target_name: str, identity: Any) -> None:
        original_publish(anchor, source_name, target_name, identity)
        if anchor.directory == service.campaigns_root:
            target = anchor.directory / target_name
            campaign = target / "campaign.json"
            atomic_write_bytes(campaign, campaign.read_bytes() + b"\n")
            raise OSError("injected foreign-byte drift after campaign rename")

    monkeypatch.setattr(conditioned_service_module, "_publish_directory_noreplace", drift_after_campaign_rename)
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as error:
        service.publish(job_id, **_publish_kwargs(candidate, job))
    assert error.value.code == "publication_failed"
    assert not list(service.datasets_root.glob("conditioned-v5-*"))
    assert not list(service.campaigns_root.glob("conditioned-v5-*"))
    residues = list(service.campaigns_root.glob(".rollback-drift-conditioned-v5-*"))
    assert len(residues) == 1
    assert (residues[0] / "campaign.json").read_bytes().endswith(b"\n\n")
    assert service.job(job_id)["status"] == "FAILED"


def test_publication_rollback_refuses_foreign_directory_inode_but_continues_owned_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    service, job_id, candidate = _built_candidate(project)
    job = _attach_evidence_pair(service, job_id, candidate)
    original_publish = conditioned_service_module._publish_directory_noreplace

    def replace_after_campaign_rename(anchor: Any, source_name: str, target_name: str, identity: Any) -> None:
        original_publish(anchor, source_name, target_name, identity)
        if anchor.directory == service.campaigns_root:
            target = anchor.directory / target_name
            target.rename(anchor.directory / f".detached-{target_name}")
            target.mkdir()
            (target / "foreign.txt").write_bytes(b"foreign-directory-bytes")
            raise OSError("injected foreign inode after campaign rename")

    monkeypatch.setattr(conditioned_service_module, "_publish_directory_noreplace", replace_after_campaign_rename)
    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as error:
        service.publish(job_id, **_publish_kwargs(candidate, job))
    assert error.value.code == "publication_rollback_identity_changed"
    assert not list(service.datasets_root.glob("conditioned-v5-*"))
    campaigns = list(service.campaigns_root.glob("conditioned-v5-*"))
    assert len(campaigns) == 1
    assert (campaigns[0] / "foreign.txt").read_bytes() == b"foreign-directory-bytes"


def test_publication_final_state_failure_rolls_back_exact_dataset_and_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    service, job_id, candidate = _built_candidate(project)
    job = _attach_evidence_pair(service, job_id, candidate)
    original_write = service._write_state_unlocked

    def fail_complete_state(root: Path, state: dict[str, Any]) -> None:
        if state.get("status") == "COMPLETE" and state.get("publication") is not None:
            raise OSError("injected final state failure")
        original_write(root, state)

    monkeypatch.setattr(service, "_write_state_unlocked", fail_complete_state)
    with pytest.raises(OSError, match="injected final state failure"):
        service.publish(job_id, **_publish_kwargs(candidate, job))
    assert not list(service.datasets_root.glob("conditioned-v5-*"))
    assert not list(service.campaigns_root.glob("conditioned-v5-*"))
    assert service.job(job_id)["status"] == "FAILED"


def test_direct_production_campaign_builder_publishes_bound_launch_ready_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.product_features.training.activation as training_activation

    monkeypatch.setattr(training_activation, "MIN_CONDITIONED_IMAGES", 4)
    project = tmp_path / "project"
    service, job_id, candidate = _built_candidate(project, production_builder=True)
    job = _attach_evidence_pair(service, job_id, candidate)
    published = service.publish(job_id, **_publish_kwargs(candidate, job))

    publication = published["publication"]
    campaign_path = project / publication["campaign_config"]
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    recommended = campaign["product_profiles"]["recommended"]["campaign"]
    assert publication["campaign_launch_ready"] is True
    assert publication["campaign_seeds"] == [731001, 731002, 731003]
    assert recommended["training"]["max_optimizer_steps"] == 5_000
    assert recommended["identities"]["dataset_freeze_hash"] == publication["activation_manifest_sha256"]


def test_activation_is_explicit_cas_and_does_not_start_training(tmp_path: Path) -> None:
    project = tmp_path / "project"
    service, job_id, candidate, publication, before_config = _published_configured_candidate(project)
    before_sha256 = hashlib.sha256(before_config).hexdigest()

    activated = service.activate(job_id, **_activation_kwargs(candidate, publication, before_sha256))
    assert activated["stage"] == "activated"
    assert activated["publication"]["configuration_activated"] is True
    assert activated["publication"]["training_started"] is False
    receipt = project / "runs" / "v3" / "conditioned-dataset-v5" / job_id / "activation_receipt" / "receipt.json"
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_value["training_started"] is False
    assert receipt_value["config_before_sha256"] == before_sha256
    assert receipt_value["config_after_sha256"] == hashlib.sha256((project / "spritelab.yaml").read_bytes()).hexdigest()

    reloaded = yaml.safe_load((project / "spritelab.yaml").read_text(encoding="utf-8"))
    original = yaml.safe_load(before_config.decode("utf-8"))
    assert reloaded["dataset"]["view_manifest"] != original["dataset"]["view_manifest"]
    assert reloaded["dataset"]["freeze_manifest"] == publication["activation_manifest"]
    assert reloaded["training"]["dataset_freeze"] == publication["activation_manifest"]
    assert reloaded["training"]["campaign_config"] == publication["campaign_config"]
    assert reloaded["execution"]["allow_dataset_production_freeze"] is True
    assert reloaded["execution"]["allow_training"] is True
    assert reloaded["project"] == original["project"]
    assert reloaded["paths"] == original["paths"]

    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as consumed:
        service.activate(job_id, **_activation_kwargs(candidate, activated["publication"], before_sha256))
    assert consumed.value.code == "activation_authorization_consumed"


def test_activation_refuses_stale_config_without_mutation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    service, job_id, candidate, publication, before_config = _published_configured_candidate(project)
    kwargs = _activation_kwargs(candidate, publication, "0" * 64)

    with pytest.raises(conditioned_service_module.ConditionedDatasetError) as stale:
        service.activate(job_id, **kwargs)
    assert stale.value.code == "activation_config_changed"
    assert (project / "spritelab.yaml").read_bytes() == before_config
    assert not (project / "runs" / "v3" / "conditioned-dataset-v5" / job_id / "activation_receipt").exists()
    assert service.job(job_id)["publication"]["configuration_activated"] is False


def test_activation_final_state_failure_restores_config_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    service, job_id, candidate, publication, before_config = _published_configured_candidate(project)
    original_write = service._write_state_unlocked

    def fail_activated_state(root: Path, state: dict[str, Any]) -> None:
        if state.get("stage") == "activated":
            raise OSError("injected activation state failure")
        original_write(root, state)

    monkeypatch.setattr(service, "_write_state_unlocked", fail_activated_state)
    with pytest.raises(OSError, match="injected activation state failure"):
        service.activate(
            job_id,
            **_activation_kwargs(candidate, publication, hashlib.sha256(before_config).hexdigest()),
        )
    assert (project / "spritelab.yaml").read_bytes() == before_config
    assert not (project / "runs" / "v3" / "conditioned-dataset-v5" / job_id / "activation_receipt").exists()
    restored = service.job(job_id)
    assert restored["stage"] == "published"
    assert restored["activation_authorization"] is None
    assert restored["publication"]["configuration_activated"] is False

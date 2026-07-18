from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from spritelab.product_core import ProjectContext
from spritelab.product_features.training import audit as audit_module
from spritelab.product_features.training.activation import (
    MANDATORY_TRAINING_AUDIT_GATES,
    TRAINING_AUDIT_HASHES_SCHEMA,
    TRAINING_AUDIT_REPORT_SCHEMA,
    ConditionedTrainingActivation,
)
from spritelab.product_features.training.audit import (
    TRAINING_AUDIT_ACTION_RECORD_SCHEMA,
    TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA,
    TRAINING_AUDIT_TEST_HARNESS_SCHEMA,
    TRAINING_AUDITOR_ID,
    TrainingAuditExecution,
    TrainingAuditExecutionError,
    run_training_infrastructure_audit,
    training_audit_action_record_path,
    training_audit_receipt_path,
    verify_training_audit_execution,
)
from spritelab.product_features.training.models import TrainingProfile
from spritelab.training.campaign import file_sha256, stable_hash
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _activation(root: Path) -> ConditionedTrainingActivation:
    freeze = root / "dataset" / "activation.json"
    campaign = root / "campaign" / "campaign.json"
    validation = root / "dataset" / "evidence" / "dataset_validation.json"
    _write_json(freeze, {"fixture": "freeze"})
    _write_json(campaign, {"fixture": "campaign"})
    _write_json(validation, {"checks": {"training_loader_all_splits": "PASS"}})
    config = ProjectConfig(root, None, deepcopy(DEFAULT_CONFIG))
    return ConditionedTrainingActivation(
        config=config,
        profile=TrainingProfile.RECOMMENDED,
        freeze_path=freeze,
        freeze_sha256=file_sha256(freeze),
        campaign_config_path=campaign,
        campaign_config_sha256=file_sha256(campaign),
        manifest={},
        artifacts={"validation_report": validation},
        selected_spec={},
        campaign={
            "campaign_identity": "c" * 64,
            "code_identity": {"sha256": "d" * 64, "files": []},
        },
        audit_status=AuditStatus.NOT_AUDITED,
    )


def _runner_inventory() -> dict[str, Any]:
    payload = {
        "schema_version": TRAINING_AUDIT_RUNNER_INVENTORY_SCHEMA,
        "auditor_id": TRAINING_AUDITOR_ID,
        "files": [],
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}


def _test_harness_inventory() -> dict[str, Any]:
    payload = {
        "schema_version": TRAINING_AUDIT_TEST_HARNESS_SCHEMA,
        "files": [],
        "interpreter": {
            "sha256": "7" * 64,
            "byte_count": 1,
            "python_version": "fixture",
        },
        "pytest": {"version": "fixture", "entrypoint_sha256": "8" * 64},
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}


def _patch_successful_execution(
    monkeypatch: pytest.MonkeyPatch,
    activation: ConditionedTrainingActivation,
) -> None:
    runner = _runner_inventory()
    test_harness = _test_harness_inventory()
    monkeypatch.setattr(audit_module, "load_conditioned_training_activation", lambda *_args, **_kwargs: activation)
    monkeypatch.setattr(audit_module, "training_audit_runner_inventory", lambda _root: runner)
    monkeypatch.setattr(audit_module, "training_audit_test_harness_inventory", lambda _root: test_harness)
    monkeypatch.setattr(audit_module, "_training_code_inventory", lambda _root, _activation: [])
    monkeypatch.setattr(
        audit_module,
        "_verify_smoke_evidence",
        lambda *_args: (
            AuditStatus.PASS,
            [
                {
                    "kind": "cpu_cuda_smoke_bundle",
                    "smoke_id": "smoke-fixture",
                    "evidence_identity": "e" * 64,
                    "plan_identity": "f" * 64,
                    "cpu_receipt_identity": "1" * 64,
                    "cuda_receipt_identity": "2" * 64,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(audit_module, "_smoke_bundle_inventory", lambda *_args: [])
    monkeypatch.setattr(
        audit_module,
        "_execute_fixed_test_plan",
        lambda *_args: [
            audit_module._test_result("curated", "PASS", 0, b"curated passed", b""),
            audit_module._test_result("full", "PASS", 0, b"full passed", b""),
        ],
    )
    monkeypatch.setattr(audit_module, "_revalidate_before_publication", lambda *_args: None)


def _publish_action_record(
    root: Path,
    execution: TrainingAuditExecution,
    config: ProjectConfig,
    source_job_id: str,
) -> None:
    receipt = json.loads(execution.receipt_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": TRAINING_AUDIT_ACTION_RECORD_SCHEMA,
        "source_job_id": source_job_id,
        "operation_identity": execution.operation_identity,
        "prospective_configuration_identity_sha256": stable_hash(config.values),
        "base_config_sha256": "b" * 64,
        "verdict": execution.verdict.value,
        "report_sha256": file_sha256(execution.report_path),
        "hash_inventory_sha256": file_sha256(execution.hashes_path),
        "receipt_sha256": file_sha256(execution.receipt_path),
        "receipt_identity": receipt["receipt_identity"],
        "config_unchanged": True,
        "configuration_activated": False,
        "training_started": False,
        "paths_exposed": False,
    }
    _write_json(
        training_audit_action_record_path(root, source_job_id, execution.operation_identity),
        {**payload, "record_identity": stable_hash(payload)},
    )


def test_server_action_writes_receipt_last_and_refuses_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)

    source_job_id = "conditioned-" + "d" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-001",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )

    assert execution.verdict is AuditStatus.PASS
    assert execution.report_path.is_file()
    assert execution.hashes_path.is_file()
    assert execution.receipt_path == training_audit_receipt_path(activation.config)
    assert execution.receipt_path.is_file()
    assert all(
        path.stat(follow_symlinks=False).st_nlink == 1
        for path in (
            execution.report_path,
            execution.hashes_path,
            execution.receipt_path,
        )
    )
    report = json.loads(execution.report_path.read_text(encoding="utf-8"))
    assert report["gates"] == dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.PASS

    with pytest.raises(TrainingAuditExecutionError) as captured:
        run_training_infrastructure_audit(
            activation.config,
            operation_nonce="audit-operation-002",
            smoke_id="smoke-fixture",
        )
    assert captured.value.code == "audit_output_exists"


def test_coherent_audit_snapshot_holds_all_authorization_bytes_and_exposes_one_launch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    source_job_id = "conditioned-" + "6" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-snapshot-001",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)

    with audit_module.open_training_audit_execution_snapshot(
        activation.config,
        None,
        activation,
    ) as snapshot:
        assert snapshot.status is AuditStatus.PASS
        assert snapshot.operation_identity == execution.operation_identity
        assert snapshot.report == json.loads(execution.report_path.read_text(encoding="utf-8"))
        expected_evidence = {
            "schema_version": audit_module.TRAINING_AUDIT_LAUNCH_AUTHORIZATION_SCHEMA,
            "status": "PASS",
            "operation_identity": execution.operation_identity,
            "report_sha256": snapshot.report_sha256,
            "hash_inventory_sha256": snapshot.hash_inventory_sha256,
            "receipt_sha256": snapshot.receipt_sha256,
            "receipt_identity": snapshot.receipt_identity,
            "action_record_sha256": snapshot.action_record_sha256,
        }
        assert snapshot.launch_authorization_evidence_sha256 == stable_hash(expected_evidence)
        snapshot.verify_unchanged()

    with pytest.raises(audit_module.UnsafeFilesystemOperation, match="closed"):
        snapshot.verify_unchanged()


def test_coherent_audit_snapshot_rejects_one_file_opened_from_a_transient_foreign_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    source_job_id = "conditioned-" + "7" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-snapshot-002",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)
    parked = execution.hashes_path.with_name("parked-audit-hashes.json")
    foreign = execution.hashes_path.with_name("foreign-audit-hashes.json")
    foreign_bytes = b'{"foreign":true}\n'
    foreign.write_bytes(foreign_bytes)
    outside = tmp_path / "outside-sentinel.bin"
    outside.write_bytes(b"outside-byte-identical")
    original_open = audit_module._open_held_audit_file

    def open_transient_foreign(stack, path: Path, root: Path):
        if path == execution.hashes_path:
            os.replace(execution.hashes_path, parked)
            os.replace(foreign, execution.hashes_path)
            held = original_open(stack, path, root)
            os.replace(execution.hashes_path, foreign)
            os.replace(parked, execution.hashes_path)
            return held
        return original_open(stack, path, root)

    monkeypatch.setattr(audit_module, "_open_held_audit_file", open_transient_foreign)

    with pytest.raises((audit_module.UnsafeFilesystemOperation, OSError)):
        with audit_module.open_training_audit_execution_snapshot(activation.config, None, activation):
            raise AssertionError("an incoherent audit snapshot was yielded")

    if os.name == "nt":
        assert execution.hashes_path.read_bytes() == foreign_bytes
        assert parked.is_file()
    else:
        assert execution.hashes_path.read_bytes() != foreign_bytes
        assert foreign.read_bytes() == foreign_bytes
    assert outside.read_bytes() == b"outside-byte-identical"


def test_exclusive_audit_publication_rejects_same_length_foreign_pass_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "artifacts" / "training"
    output_root.mkdir(parents=True)
    target = output_root / "audit_report.json"
    displaced_owned = output_root / "attacker-displaced-owned.json"
    expected = audit_module._canonical_bytes({"verdict": "FAIL"})
    forged_pass = audit_module._canonical_bytes({"verdict": "PASS"})
    assert len(expected) == len(forged_pass)

    outside_sentinel = tmp_path.parent / f"{tmp_path.name}-outside-pass-sentinel.json"
    outside_sentinel.write_bytes(forged_pass)
    sentinel_before = outside_sentinel.read_bytes()
    sentinel_sha256 = hashlib.sha256(sentinel_before).hexdigest()
    original_publish = audit_module.AnchoredDirectory.publish_held_file_no_replace
    foreign_staging: Path | None = None

    def substitute_before_publication(
        anchor: audit_module.AnchoredDirectory,
        source_descriptor: int,
        source_name: str | None,
        destination_name: str,
        *,
        identity,
    ) -> None:
        nonlocal foreign_staging
        if destination_name == target.name:
            if source_name is None:
                target.write_bytes(forged_pass)
            else:
                anchor.rename(source_name, displaced_owned.name, replace=False)
                foreign_staging = output_root / source_name
                foreign_staging.write_bytes(forged_pass)
        return original_publish(
            anchor,
            source_descriptor,
            source_name,
            destination_name,
            identity=identity,
        )

    monkeypatch.setattr(
        audit_module.AnchoredDirectory,
        "publish_held_file_no_replace",
        substitute_before_publication,
    )

    with pytest.raises((audit_module.UnsafeFilesystemOperation, TrainingAuditExecutionError)):
        audit_module._write_exclusive(tmp_path, target, expected)

    if foreign_staging is None:
        assert target.read_bytes() == forged_pass
    else:
        assert not target.exists()
        assert foreign_staging.read_bytes() == forged_pass
        assert displaced_owned.read_bytes() == expected
    assert outside_sentinel.read_bytes() == sentinel_before
    assert hashlib.sha256(outside_sentinel.read_bytes()).hexdigest() == sentinel_sha256
    assert not tuple(output_root.glob(".training-audit-residue-*"))


def test_audit_inventory_rejects_lstat_open_restore_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    target = evidence_root / "artifact.json"
    foreign = evidence_root / "foreign.json"
    parked = evidence_root / "parked-original.json"
    original_bytes = audit_module._canonical_bytes({"verdict": "PASS"})
    forged_bytes = audit_module._canonical_bytes({"verdict": "FAIL"})
    assert len(original_bytes) == len(forged_bytes)
    target.write_bytes(original_bytes)
    foreign.write_bytes(forged_bytes)
    expected = audit_module._file_record(tmp_path, target)
    original_open = audit_module.AnchoredDirectory.open_file
    raced = {"value": False}

    def open_foreign_then_restore(
        anchor: audit_module.AnchoredDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        if not raced["value"] and anchor.directory == evidence_root and name == target.name:
            raced["value"] = True
            os.replace(target, parked)
            os.replace(foreign, target)
            descriptor = original_open(anchor, name, flags, mode)
            os.replace(target, foreign)
            os.replace(parked, target)
            return descriptor
        return original_open(anchor, name, flags, mode)

    monkeypatch.setattr(audit_module.AnchoredDirectory, "open_file", open_foreign_then_restore)

    with pytest.raises(TrainingAuditExecutionError) as captured:
        audit_module._file_record(tmp_path, target)
    assert captured.value.code == "audit_artifact_unsafe"
    assert target.read_bytes() == original_bytes
    assert foreign.read_bytes() == forged_bytes

    raced["value"] = False
    assert audit_module._verify_artifact_inventory(tmp_path, [expected]) is False
    assert target.read_bytes() == original_bytes
    assert foreign.read_bytes() == forged_bytes


def test_fully_rehashed_caller_authored_v2_report_without_execution_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    activation = _activation(tmp_path)
    report_path = tmp_path / DEFAULT_CONFIG["training"]["audit_report"]
    hashes_path = tmp_path / DEFAULT_CONFIG["training"]["audit_hashes"]
    bindings = {
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign["campaign_identity"],
        "training_code_identity_sha256": activation.campaign["code_identity"]["sha256"],
    }
    report = {
        "schema_version": TRAINING_AUDIT_REPORT_SCHEMA,
        "bindings": bindings,
        "gates": dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS"),
    }
    _write_json(report_path, report)
    _write_json(
        hashes_path,
        {
            "schema_version": TRAINING_AUDIT_HASHES_SCHEMA,
            "audit_report_sha256": file_sha256(report_path),
            "bindings": bindings,
            "files": [],
        },
    )

    assert not training_audit_receipt_path(activation.config).exists()
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.STALE


def test_rehashing_report_and_hash_inventory_cannot_reuse_server_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    source_job_id = "conditioned-" + "f" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-003",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )
    receipt_before = execution.receipt_path.read_bytes()
    report = json.loads(execution.report_path.read_text(encoding="utf-8"))
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.PASS
    report["gates"]["api_ui_privacy"] = "FAIL"
    evidence = report["gate_evidence"]["api_ui_privacy"]
    evidence["verdict"] = "FAIL"
    evidence["evidence_identity"] = stable_hash(
        {key: value for key, value in evidence.items() if key != "evidence_identity"}
    )
    report["verdict"] = "FAIL"
    report["report_identity"] = stable_hash({key: value for key, value in report.items() if key != "report_identity"})
    _write_json(execution.report_path, report)

    hashes = json.loads(execution.hashes_path.read_text(encoding="utf-8"))
    report_bytes = execution.report_path.read_bytes()
    hashes["audit_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    hashes["audit_report_byte_count"] = len(report_bytes)
    hashes["inventory_identity"] = stable_hash(
        {key: value for key, value in hashes.items() if key != "inventory_identity"}
    )
    _write_json(execution.hashes_path, hashes)

    assert execution.receipt_path.read_bytes() == receipt_before
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.STALE


def test_receipt_is_stale_for_a_different_prospective_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    source_job_id = "conditioned-" + "e" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-004",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )
    report = json.loads(execution.report_path.read_text(encoding="utf-8"))
    receipt = json.loads(execution.receipt_path.read_text(encoding="utf-8"))
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.PASS

    changed_values = deepcopy(activation.config.values)
    changed_values["training"]["campaign_config"] = "campaign/changed.json"
    changed_config = ProjectConfig(tmp_path, None, changed_values)
    changed_activation = replace(activation, config=changed_config)

    assert receipt["bindings"]["prospective_configuration_identity_sha256"] == stable_hash(activation.config.values)
    assert receipt["bindings"]["prospective_configuration_identity_sha256"] != stable_hash(changed_values)
    assert verify_training_audit_execution(changed_config, report, changed_activation) is AuditStatus.STALE


def test_fully_recreated_report_hashes_and_receipt_cannot_replace_server_action_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    source_job_id = "conditioned-" + "9" * 20
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-008",
        smoke_id="smoke-fixture",
        source_job_id=source_job_id,
    )
    report = json.loads(execution.report_path.read_text(encoding="utf-8"))
    _publish_action_record(tmp_path, execution, activation.config, source_job_id)
    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.PASS

    evidence = report["gate_evidence"]["api_ui_privacy"]
    evidence["sources"].append({"kind": "caller_fabricated"})
    evidence["evidence_identity"] = stable_hash(
        {key: value for key, value in evidence.items() if key != "evidence_identity"}
    )
    report["report_identity"] = stable_hash({key: value for key, value in report.items() if key != "report_identity"})
    _write_json(execution.report_path, report)

    hashes = json.loads(execution.hashes_path.read_text(encoding="utf-8"))
    report_bytes = execution.report_path.read_bytes()
    hashes["audit_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    hashes["audit_report_byte_count"] = len(report_bytes)
    hashes["inventory_identity"] = stable_hash(
        {key: value for key, value in hashes.items() if key != "inventory_identity"}
    )
    _write_json(execution.hashes_path, hashes)

    receipt = json.loads(execution.receipt_path.read_text(encoding="utf-8"))
    hashes_bytes = execution.hashes_path.read_bytes()
    receipt["report"] = {
        "path": execution.report_path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
        "byte_count": len(report_bytes),
        "report_identity": report["report_identity"],
    }
    receipt["hash_inventory"] = {
        "path": execution.hashes_path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(hashes_bytes).hexdigest(),
        "byte_count": len(hashes_bytes),
        "inventory_identity": hashes["inventory_identity"],
    }
    receipt["receipt_identity"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_identity"}
    )
    _write_json(execution.receipt_path, receipt)

    assert verify_training_audit_execution(activation.config, report, activation) is AuditStatus.STALE


def test_fixed_pytest_execution_strips_hostile_environment_and_uses_isolated_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "caller_plugin")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "caller"))
    monkeypatch.setattr(audit_module, "_pytest_site_roots", lambda: ["C:/trusted/site-packages"])

    def execute(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((list(argv), dict(kwargs["env"])))
        return SimpleNamespace(returncode=0, stdout=b"passed", stderr=b"")

    monkeypatch.setattr(audit_module.subprocess, "run", execute)
    results = audit_module._execute_fixed_test_plan(tmp_path, "a" * 64)

    assert [result["verdict"] for result in results] == ["PASS", "PASS"]
    assert len(calls) == 2
    for argv, environment in calls:
        assert argv[1:4] == ["-I", "-B", "-S"]
        assert audit_module._PYTEST_BOOTSTRAP in argv
        assert "--import-mode=importlib" in argv
        assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "PYTEST_ADDOPTS" not in environment
        assert "PYTEST_PLUGINS" not in environment
        assert "PYTHONPATH" not in environment


def test_audit_outputs_are_confined_to_managed_training_namespace(tmp_path: Path) -> None:
    values = deepcopy(DEFAULT_CONFIG)
    values["training"]["audit_report"] = "datasets/freeze/audit_report.json"
    values["training"]["audit_hashes"] = "datasets/freeze/audit_hashes.json"
    config = ProjectConfig(tmp_path, None, values)

    with pytest.raises(TrainingAuditExecutionError) as captured:
        run_training_infrastructure_audit(
            config,
            operation_nonce="audit-operation-009",
            smoke_id="smoke-fixture",
        )
    assert captured.value.code == "audit_output_path"
    assert not (tmp_path / "datasets").exists()


def test_complete_smoke_inventory_detects_added_output_file(tmp_path: Path) -> None:
    smoke_id = "smoke-" + "a" * 20
    artifact = tmp_path / "artifacts/training/smokes" / smoke_id
    run = tmp_path / "runs/v3/training-smokes" / smoke_id
    _write_json(artifact / "plan.json", {"fixture": "plan"})
    _write_json(artifact / "configs/cpu.json", {"fixture": "cpu"})
    _write_json(artifact / "configs/cpu.manifest.json", {"fixture": "cpu-manifest"})
    _write_json(artifact / "configs/cuda.json", {"fixture": "cuda"})
    _write_json(artifact / "configs/cuda.manifest.json", {"fixture": "cuda-manifest"})
    (artifact / "bootstrap").mkdir(parents=True)
    (artifact / "bootstrap/preflight.py").write_text("# bound\n", encoding="utf-8")
    _write_json(run / "state.json", {"status": "PREPARED"})
    _write_json(run / "cpu/smoke_run_receipt.json", {"device": "cpu"})
    _write_json(run / "cuda/smoke_run_receipt.json", {"device": "cuda"})

    before = audit_module._smoke_bundle_inventory(tmp_path, smoke_id)
    _write_json(run / "cuda/unexpected.json", {"caller": "addition"})
    _write_json(run / "unexpected-root.json", {"caller": "root-addition"})
    after = audit_module._smoke_bundle_inventory(tmp_path, smoke_id)

    assert before != after
    assert any(item["path"].endswith("/cuda/unexpected.json") for item in after)
    assert any(item["path"].endswith("/unexpected-root.json") for item in after)


def test_registered_smoke_evidence_must_match_recomputed_bundle(tmp_path: Path) -> None:
    smoke_id = "smoke-" + "b" * 20
    recomputed_body = {
        "schema_version": "spritelab.training.smoke-evidence.v1",
        "smoke_id": smoke_id,
        "runs": {"cpu": {"receipt_identity": "1" * 64}, "cuda": {"receipt_identity": "2" * 64}},
    }
    recomputed = {**recomputed_body, "evidence_identity": stable_hash(recomputed_body)}
    published_body = {
        **deepcopy(recomputed_body),
        "server_execution_identities": {"cpu": "3" * 64, "cuda": "4" * 64},
    }
    published = {**published_body, "evidence_identity": stable_hash(published_body)}
    evidence_path = tmp_path / "artifacts/training/smokes" / smoke_id / "smoke_evidence.json"
    _write_json(evidence_path, published)

    assert audit_module._registered_smoke_evidence(tmp_path, smoke_id, recomputed) == published

    published["runs"]["cuda"]["receipt_identity"] = "5" * 64
    hostile_body = {key: value for key, value in published.items() if key != "evidence_identity"}
    published["evidence_identity"] = stable_hash(hostile_body)
    _write_json(evidence_path, published)
    with pytest.raises(audit_module.SmokeBundleError):
        audit_module._registered_smoke_evidence(tmp_path, smoke_id, recomputed)


def test_receipt_strictness_rejects_bad_timestamp_result_semantics_and_recursive_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = audit_module._test_result("curated", "PASS", 0, b"ok", b"")
    result["return_code"] = 1
    result["result_identity"] = stable_hash({key: value for key, value in result.items() if key != "result_identity"})
    assert audit_module._valid_test_result(result) is False

    activation = _activation(tmp_path)
    _patch_successful_execution(monkeypatch, activation)
    execution = run_training_infrastructure_audit(
        activation.config,
        operation_nonce="audit-operation-011",
        smoke_id="smoke-fixture",
        source_job_id="conditioned-" + "6" * 20,
    )
    receipt = json.loads(execution.receipt_path.read_text(encoding="utf-8"))
    receipt["completed_at"] = None
    receipt["receipt_identity"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_identity"}
    )
    assert audit_module._validate_receipt(receipt) is False

    nonfinite = tmp_path / "artifacts/training/nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        audit_module._read_mapping(nonfinite, tmp_path)

    monkeypatch.setattr(audit_module, "_read_mapping", lambda *_args: (_ for _ in ()).throw(RecursionError()))
    assert verify_training_audit_execution(activation.config, {}, activation) is AuditStatus.STALE


def test_conditioned_service_audits_prospective_overlay_without_mutating_config(
    tmp_path: Path,
) -> None:
    from spritelab.product_features.conditioned_v5.service import ConditionedDatasetService

    config_path = tmp_path / "spritelab.yaml"
    config_before = yaml.safe_dump(deepcopy(DEFAULT_CONFIG), sort_keys=False).encode("utf-8")
    config_path.write_bytes(config_before)
    activation_path = tmp_path / "datasets/conditioned/freeze/activation.json"
    campaign_path = tmp_path / "campaigns/conditioned/campaign.json"
    _write_json(activation_path, {"fixture": "activation"})
    _write_json(campaign_path, {"fixture": "campaign"})
    activation_sha256 = file_sha256(activation_path)
    campaign_sha256 = file_sha256(campaign_path)
    candidate_identity = "1" * 64
    publication_identity = "2" * 64
    campaign_identity = "3" * 64
    report = tmp_path / "artifacts/training/audit_report.json"
    hashes = tmp_path / "artifacts/training/audit_hashes.json"
    receipt = tmp_path / "artifacts/training/audit_receipt.json"
    calls: list[dict[str, Any]] = []

    def execute(prospective: ProjectConfig, **kwargs: Any) -> TrainingAuditExecution:
        assert config_path.read_bytes() == config_before
        assert prospective.values["dataset"]["view_manifest"] == "datasets/conditioned/freeze/view_manifest.json"
        assert prospective.values["dataset"]["freeze_manifest"] == "datasets/conditioned/freeze/activation.json"
        assert prospective.values["training"]["dataset_freeze"] == "datasets/conditioned/freeze/activation.json"
        assert prospective.values["training"]["campaign_config"] == "campaigns/conditioned/campaign.json"
        assert prospective.values["execution"]["allow_dataset_production_freeze"] is True
        assert prospective.values["execution"]["allow_training"] is True
        calls.append(kwargs)
        _write_json(report, {"fixture": "report"})
        _write_json(hashes, {"fixture": "hashes"})
        _write_json(receipt, {"receipt_identity": "9" * 64})
        return TrainingAuditExecution("a" * 64, AuditStatus.PASS, report, hashes, receipt)

    service = ConditionedDatasetService(tmp_path, training_infrastructure_audit_runner=execute)
    job_id = "conditioned-" + "a" * 20
    job_root = service.jobs_root / job_id
    job_root.mkdir(parents=True)
    service._write_state(
        job_root,
        {
            "schema_version": "spritelab.dataset.conditioned-job.v1",
            "job_id": job_id,
            "status": "COMPLETE",
            "candidate": {"candidate_identity": candidate_identity},
            "publication": {
                "publication_identity_sha256": publication_identity,
                "activation_manifest": "datasets/conditioned/freeze/activation.json",
                "activation_manifest_sha256": activation_sha256,
                "campaign_config": "campaigns/conditioned/campaign.json",
                "campaign_config_sha256": campaign_sha256,
                "campaign_identity_sha256": campaign_identity,
                "campaign_launch_ready": True,
                "campaign_seeds": [731001, 731002, 731003],
                "campaign_steps": 5_000,
                "configuration_activated": False,
                "training_started": False,
                "paths_exposed": False,
            },
            "paths_exposed": False,
        },
    )

    result = service.run_training_infrastructure_audit(
        job_id,
        candidate_identity=candidate_identity,
        publication_identity_sha256=publication_identity,
        activation_manifest_sha256=activation_sha256,
        campaign_config_sha256=campaign_sha256,
        campaign_identity_sha256=campaign_identity,
        expected_config_sha256=hashlib.sha256(config_before).hexdigest(),
        smoke_id="smoke-fixture",
        operation_nonce="audit-operation-005",
        explicit_action=True,
    )

    assert result["verdict"] == "PASS"
    assert result["config_unchanged"] is True
    assert result["action_record_identity"]
    assert result["configuration_activated"] is False
    assert result["training_started"] is False
    assert config_path.read_bytes() == config_before
    reloaded = ProjectConfig.load(tmp_path)
    assert reloaded.values["execution"]["allow_dataset_production_freeze"] is False
    assert reloaded.values["execution"]["allow_training"] is False
    assert calls == [
        {
            "operation_nonce": "audit-operation-005",
            "smoke_id": "smoke-fixture",
            "source_job_id": job_id,
        }
    ]


def test_conditioned_service_discards_apparent_pass_when_runner_mutates_config(
    tmp_path: Path,
) -> None:
    from spritelab.product_features.conditioned_v5.service import (
        ConditionedDatasetError,
        ConditionedDatasetService,
    )

    config_path = tmp_path / "spritelab.yaml"
    config_before = yaml.safe_dump(deepcopy(DEFAULT_CONFIG), sort_keys=False).encode("utf-8")
    config_path.write_bytes(config_before)
    activation_path = tmp_path / "datasets/conditioned/freeze/activation.json"
    campaign_path = tmp_path / "campaigns/conditioned/campaign.json"
    _write_json(activation_path, {"fixture": "activation"})
    _write_json(campaign_path, {"fixture": "campaign"})
    activation_sha256 = file_sha256(activation_path)
    campaign_sha256 = file_sha256(campaign_path)
    candidate_identity = "1" * 64
    publication_identity = "2" * 64
    campaign_identity = "3" * 64
    report = tmp_path / "artifacts/training/audit_report.json"
    hashes = tmp_path / "artifacts/training/audit_hashes.json"
    receipt = tmp_path / "artifacts/training/audit_receipt.json"

    def hostile_execute(_prospective: ProjectConfig, **_kwargs: Any) -> TrainingAuditExecution:
        mutated = deepcopy(DEFAULT_CONFIG)
        mutated["execution"]["allow_training"] = True
        config_path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
        return TrainingAuditExecution("a" * 64, AuditStatus.PASS, report, hashes, receipt)

    service = ConditionedDatasetService(tmp_path, training_infrastructure_audit_runner=hostile_execute)
    job_id = "conditioned-" + "c" * 20
    job_root = service.jobs_root / job_id
    job_root.mkdir(parents=True)
    service._write_state(
        job_root,
        {
            "schema_version": "spritelab.dataset.conditioned-job.v1",
            "job_id": job_id,
            "status": "COMPLETE",
            "candidate": {"candidate_identity": candidate_identity},
            "publication": {
                "publication_identity_sha256": publication_identity,
                "activation_manifest": "datasets/conditioned/freeze/activation.json",
                "activation_manifest_sha256": activation_sha256,
                "campaign_config": "campaigns/conditioned/campaign.json",
                "campaign_config_sha256": campaign_sha256,
                "campaign_identity_sha256": campaign_identity,
                "campaign_launch_ready": True,
                "campaign_seeds": [731001, 731002, 731003],
                "campaign_steps": 5_000,
                "configuration_activated": False,
                "training_started": False,
                "paths_exposed": False,
            },
            "paths_exposed": False,
        },
    )

    with pytest.raises(ConditionedDatasetError) as captured:
        service.run_training_infrastructure_audit(
            job_id,
            candidate_identity=candidate_identity,
            publication_identity_sha256=publication_identity,
            activation_manifest_sha256=activation_sha256,
            campaign_config_sha256=campaign_sha256,
            campaign_identity_sha256=campaign_identity,
            expected_config_sha256=hashlib.sha256(config_before).hexdigest(),
            smoke_id="smoke-fixture",
            operation_nonce="audit-operation-007",
            explicit_action=True,
        )

    assert captured.value.code == "training_audit_config_mutated"
    assert "not applicable" in captured.value.public_message
    config_after = config_path.read_bytes()
    assert config_after != config_before
    assert hashlib.sha256(config_after).hexdigest() != hashlib.sha256(config_before).hexdigest()
    assert ProjectConfig.load(tmp_path).values["execution"]["allow_training"] is True
    state = service.job(job_id)
    assert state["publication"]["configuration_activated"] is False
    assert state["publication"]["training_started"] is False


def test_conditioned_service_refuses_config_change_before_action_record_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.product_features.conditioned_v5.service import (
        ConditionedDatasetError,
        ConditionedDatasetService,
    )

    config_path = tmp_path / "spritelab.yaml"
    config_before = yaml.safe_dump(deepcopy(DEFAULT_CONFIG), sort_keys=False).encode("utf-8")
    config_path.write_bytes(config_before)
    activation_path = tmp_path / "datasets/conditioned/freeze/activation.json"
    campaign_path = tmp_path / "campaigns/conditioned/campaign.json"
    _write_json(activation_path, {"fixture": "activation"})
    _write_json(campaign_path, {"fixture": "campaign"})
    report = tmp_path / "artifacts/training/audit_report.json"
    hashes = tmp_path / "artifacts/training/audit_hashes.json"
    receipt = tmp_path / "artifacts/training/audit_receipt.json"

    def execute(_prospective: ProjectConfig, **_kwargs: Any) -> TrainingAuditExecution:
        _write_json(report, {"fixture": "report"})
        _write_json(hashes, {"fixture": "hashes"})
        _write_json(receipt, {"receipt_identity": "9" * 64})
        return TrainingAuditExecution("a" * 64, AuditStatus.PASS, report, hashes, receipt)

    service = ConditionedDatasetService(tmp_path, training_infrastructure_audit_runner=execute)
    job_id = "conditioned-" + "7" * 20
    candidate_identity = "1" * 64
    publication_identity = "2" * 64
    campaign_identity = "3" * 64
    activation_sha256 = file_sha256(activation_path)
    campaign_sha256 = file_sha256(campaign_path)
    job_root = service.jobs_root / job_id
    job_root.mkdir(parents=True)
    service._write_state(
        job_root,
        {
            "schema_version": "spritelab.dataset.conditioned-job.v1",
            "job_id": job_id,
            "status": "COMPLETE",
            "candidate": {"candidate_identity": candidate_identity},
            "publication": {
                "publication_identity_sha256": publication_identity,
                "activation_manifest": "datasets/conditioned/freeze/activation.json",
                "activation_manifest_sha256": activation_sha256,
                "campaign_config": "campaigns/conditioned/campaign.json",
                "campaign_config_sha256": campaign_sha256,
                "campaign_identity_sha256": campaign_identity,
                "campaign_launch_ready": True,
                "campaign_seeds": [731001, 731002, 731003],
                "campaign_steps": 5_000,
                "configuration_activated": False,
                "training_started": False,
                "paths_exposed": False,
            },
            "paths_exposed": False,
        },
    )

    original_receipt_path = audit_module.training_audit_receipt_path

    def mutate_in_commit_gap(config: ProjectConfig) -> Path:
        mutated = deepcopy(DEFAULT_CONFIG)
        mutated["execution"]["allow_training"] = True
        config_path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
        return original_receipt_path(config)

    monkeypatch.setattr(audit_module, "training_audit_receipt_path", mutate_in_commit_gap)
    with pytest.raises(ConditionedDatasetError) as captured:
        service.run_training_infrastructure_audit(
            job_id,
            candidate_identity=candidate_identity,
            publication_identity_sha256=publication_identity,
            activation_manifest_sha256=activation_sha256,
            campaign_config_sha256=campaign_sha256,
            campaign_identity_sha256=campaign_identity,
            expected_config_sha256=hashlib.sha256(config_before).hexdigest(),
            smoke_id="smoke-fixture",
            operation_nonce="audit-operation-010",
            explicit_action=True,
        )

    assert captured.value.code == "training_audit_config_mutated"
    record = training_audit_action_record_path(tmp_path, job_id, "a" * 64)
    assert not record.exists()
    state = service.job(job_id)
    assert state["publication"]["configuration_activated"] is False
    assert state["publication"]["training_started"] is False


def test_conditioned_training_audit_api_rejects_caller_verdicts_and_forwards_exact_selection(
    tmp_path: Path,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from spritelab.product_features.conditioned_v5.web import create_router

    context = ProjectContext(tmp_path, deepcopy(DEFAULT_CONFIG), tmp_path / "spritelab.yaml", tmp_path / "runs/v3")
    calls: list[tuple[str, dict[str, Any]]] = []

    class Service:
        def inventory(self) -> dict[str, Any]:
            return {}

        def run_training_infrastructure_audit(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((job_id, kwargs))
            return {
                "schema_version": "spritelab.training.infrastructure-audit-action.v1",
                "job_id": job_id,
                "smoke_id": kwargs["smoke_id"],
                "operation_identity": "a" * 64,
                "prospective_configuration_identity_sha256": "b" * 64,
                "base_config_sha256": kwargs["expected_config_sha256"],
                "action_record_identity": "c" * 64,
                "verdict": "INCONCLUSIVE",
                "config_unchanged": True,
                "report_path": "file:///C:/private/audit_report.json",
                "hashes_path": "C:/private/audit_hashes.json",
                "receipt_path": "C:/private/audit_receipt.json",
                "action_record_path": "C:/private/action.json",
                "configuration_activated": False,
                "training_started": False,
                "paths_exposed": False,
                "secret_token": "PRIVATE_AUDIT_TOKEN",
            }

    app = FastAPI()
    app.include_router(create_router(context, service=Service()))  # type: ignore[arg-type]
    client = TestClient(app)
    job_id = "conditioned-" + "b" * 20
    payload = {
        "candidate_identity": "1" * 64,
        "publication_identity_sha256": "2" * 64,
        "activation_manifest_sha256": "3" * 64,
        "campaign_config_sha256": "4" * 64,
        "campaign_identity_sha256": "5" * 64,
        "expected_config_sha256": "6" * 64,
        "smoke_id": "smoke-" + "d" * 20,
        "operation_nonce": "audit-operation-006",
        "explicit_action": True,
    }
    hostile = client.post(
        f"/dataset-v5/api/jobs/{job_id}/training-audit",
        json={**payload, "gates": dict.fromkeys(MANDATORY_TRAINING_AUDIT_GATES, "PASS")},
    )
    assert hostile.status_code == 422
    assert calls == []

    accepted = client.post(f"/dataset-v5/api/jobs/{job_id}/training-audit", json=payload)
    assert accepted.status_code == 200
    assert accepted.json() == {
        "schema_version": "spritelab.training.infrastructure-audit-action-public.v1",
        "job_id": job_id,
        "smoke_id": "smoke-" + "d" * 20,
        "operation_identity": "a" * 64,
        "prospective_configuration_identity_sha256": "b" * 64,
        "base_config_sha256": "6" * 64,
        "action_record_identity": "c" * 64,
        "verdict": "INCONCLUSIVE",
        "config_unchanged": True,
        "configuration_activated": False,
        "training_started": False,
        "paths_exposed": False,
    }
    assert "PRIVATE_AUDIT_TOKEN" not in accepted.text
    assert "C:/private" not in accepted.text
    assert calls == [
        (
            job_id,
            {
                "candidate_identity": "1" * 64,
                "publication_identity_sha256": "2" * 64,
                "activation_manifest_sha256": "3" * 64,
                "campaign_config_sha256": "4" * 64,
                "campaign_identity_sha256": "5" * 64,
                "expected_config_sha256": "6" * 64,
                "smoke_id": "smoke-" + "d" * 20,
                "operation_nonce": "audit-operation-006",
                "explicit_action": True,
            },
        )
    ]


def test_conditioned_training_audit_options_expose_only_bound_registered_smokes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from spritelab.product_features.conditioned_v5.web import create_router
    from spritelab.product_features.evaluation import exploratory_smoke as exploratory_smoke_module

    job_id = "conditioned-" + "c" * 20
    activation_sha256 = "3" * 64
    campaign_identity = "5" * 64
    job_state: dict[str, Any] = {
        "job_id": job_id,
        "status": "COMPLETE",
        "candidate": {"candidate_identity": "1" * 64},
        "publication": {
            "activation_manifest_sha256": activation_sha256,
            "campaign_identity_sha256": campaign_identity,
            "configuration_activated": False,
        },
    }

    class Service:
        def inventory(self) -> dict[str, Any]:
            return {}

        def job(self, selected_job_id: str) -> dict[str, Any]:
            assert selected_job_id == job_id
            return job_state

    eligible = (
        SimpleNamespace(
            weights="live",
            smoke_id="smoke-" + "a" * 20,
            registration_id="exploratory-" + "a" * 24,
            freeze_identity=activation_sha256,
            campaign_identity=campaign_identity,
            path=tmp_path / "private-live.pt",
        ),
        SimpleNamespace(
            weights="ema",
            smoke_id="smoke-" + "a" * 20,
            registration_id="exploratory-" + "a" * 24,
            freeze_identity=activation_sha256,
            campaign_identity=campaign_identity,
            path=tmp_path / "private-ema.pt",
        ),
        SimpleNamespace(
            weights="ema",
            smoke_id="smoke-" + "b" * 20,
            registration_id="exploratory-" + "b" * 24,
            freeze_identity=activation_sha256,
            campaign_identity="9" * 64,
            path=tmp_path / "wrong-campaign.pt",
        ),
        SimpleNamespace(
            weights="ema",
            smoke_id="smoke-" + "c" * 20,
            registration_id="exploratory-" + "c" * 24,
            freeze_identity=activation_sha256,
            campaign_identity=campaign_identity,
            path=tmp_path / "wrong-job.pt",
        ),
    )

    class Workflow:
        def __init__(self, project_root: Path, **kwargs: Any) -> None:
            assert project_root == tmp_path
            assert callable(kwargs["job_loader"])

        def prepared_plans(self) -> dict[str, Any]:
            return {
                "eligible": [
                    {"conditioned_job_id": job_id, "smoke_id": "smoke-" + "a" * 20},
                    {"conditioned_job_id": job_id, "smoke_id": "smoke-" + "b" * 20},
                    {"conditioned_job_id": "conditioned-" + "d" * 20, "smoke_id": "smoke-" + "c" * 20},
                ]
            }

        def catalog(self) -> SimpleNamespace:
            return SimpleNamespace(eligible=eligible)

    monkeypatch.setattr(exploratory_smoke_module, "ExploratorySmokeWorkflow", Workflow)
    context = ProjectContext(tmp_path, deepcopy(DEFAULT_CONFIG), tmp_path / "spritelab.yaml", tmp_path / "runs/v3")
    app = FastAPI()
    app.include_router(create_router(context, service=Service()))  # type: ignore[arg-type]
    client = TestClient(app)

    response = client.get(f"/dataset-v5/api/jobs/{job_id}/training-audit-options")
    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "spritelab.training.conditioned-audit-options.v1",
        "job_id": job_id,
        "eligible": [
            {
                "smoke_id": "smoke-" + "a" * 20,
                "registration_id": "exploratory-" + "a" * 24,
                "status": "PROVISIONALLY_VERIFIED",
                "purpose": "exploratory",
            }
        ],
        "count": 1,
        "ready": True,
        "paths_exposed": False,
    }
    assert "private" not in response.text

    job_state["publication"]["configuration_activated"] = True
    blocked = client.get(f"/dataset-v5/api/jobs/{job_id}/training-audit-options")
    assert blocked.status_code == 200
    assert blocked.json()["eligible"] == []
    assert blocked.json()["ready"] is False


def test_conditioned_page_exposes_registered_smoke_training_audit_control(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from spritelab.product_features.conditioned_v5.plugin import create_plugin
    from spritelab.product_web.app import create_app

    class Service:
        def inventory(self) -> dict[str, Any]:
            return {"managed_intakes": [], "jobs": [], "config_sha256": "6" * 64}

    runs = tmp_path / "runs"
    runs.mkdir()
    context = ProjectContext(tmp_path, deepcopy(DEFAULT_CONFIG), tmp_path / "spritelab.yaml", runs)
    plugin = create_plugin(service_factory=lambda _context: Service())  # type: ignore[arg-type]
    client = TestClient(create_app(context, plugins=(plugin,)))

    page = client.get("/dataset-v5")
    assert page.status_code == 200
    assert 'id="cv5-training-audit-controls" hidden' in page.text
    assert 'id="cv5-training-smoke"' in page.text
    assert 'id="cv5-training-audit"' in page.text
    assert 'id="cv5-training-audit-result"' in page.text
    assert 'type="file"' not in page.text

    javascript = client.get("/dataset-v5/static/conditioned-v5.js")
    assert javascript.status_code == 200
    assert "/training-audit-options" in javascript.text
    assert "/training-audit`" in javascript.text
    assert 'operation_nonce: authorizationId("training-audit")' in javascript.text
    assert "gates:" not in javascript.text
    assert "/training/api/start" not in javascript.text
    assert "item.source_title" not in javascript.text
    assert "item.source_id" not in javascript.text
    assert "trainingAuditResult.report_path" not in javascript.text


def test_conditioned_browser_state_uses_closed_privacy_projection(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from spritelab.product_features.conditioned_v5.web import create_router

    job_id = "conditioned-" + "e" * 20
    dataset_reference = "dataset." + "f" * 24
    hostile_job = {
        "schema_version": "spritelab.dataset.conditioned-job.v1",
        "job_id": job_id,
        "status": "COMPLETE",
        "stage": "secret_stage",
        "current": 8,
        "total": 8,
        "message": "file:///C:/private/PRIVATE_JOB_MESSAGE",
        "idempotency_key": "PRIVATE_IDEMPOTENCY_SECRET",
        "lease": {"owner_pid": 1234, "secret_token": "PRIVATE_LEASE_TOKEN"},
        "events": [{"message": "PRIVATE_EVENT", "absolute_path": "C:/private/event.json"}],
        "candidate": {
            "candidate_identity": "1" * 64,
            "payload_inventory_sha256": "2" * 64,
            "image_count": 2500,
            "production_authorized": "true",
            "input_bindings": [{"source_title": "PRIVATE_SOURCE_TITLE", "api_key": "PRIVATE_API_KEY"}],
        },
        "evidence": {
            "label_audit": {
                "sha256": "3" * 64,
                "byte_count": 123,
                "audit_run_identity": "4" * 64,
                "relative_path": "file:///C:/private/label.json",
                "receipt": {"receipt_identity": "5" * 64, "relative_path": "C:/private/receipt.json"},
                "action": {"record_identity": "6" * 64, "relative_path": "C:/private/action.json"},
                "password": "PRIVATE_EVIDENCE_PASSWORD",
            },
            "dataset_validation": {
                "sha256": "7" * 64,
                "byte_count": 456,
                "audit_run_identity": "8" * 64,
                "relative_path": "C:/private/validation.json",
                "receipt": {"receipt_identity": "9" * 64},
                "action": {"record_identity": "a" * 64},
            },
            "unknown_report": {"file_uri": "file:///C:/private/unknown.json"},
        },
        "publication": {
            "publication_identity_sha256": "b" * 64,
            "activation_manifest": "file:///C:/private/activation.json",
            "activation_manifest_sha256": "c" * 64,
            "campaign_config": "C:/private/campaign.json",
            "campaign_config_sha256": "d" * 64,
            "campaign_identity_sha256": "e" * 64,
            "campaign_launch_ready": "true",
            "campaign_seeds": [731001, 731002, 731003, "PRIVATE_SEED"],
            "campaign_steps": 5000,
            "configuration_activated": "false",
            "training_started": "false",
            "secret": "PRIVATE_PUBLICATION_SECRET",
        },
        "activation_authorization": {
            "status": "COMMITTED",
            "config_after_sha256": "f" * 64,
            "one_time": True,
            "receipt_relative_path": "C:/private/activation-receipt.json",
            "authorization_token": "PRIVATE_ACTIVATION_TOKEN",
        },
        "unknown_nested_state": {"source_title": "PRIVATE_UNKNOWN_TITLE"},
        "paths_exposed": False,
    }
    hostile_inventory = {
        "managed_intakes": [
            {
                "dataset_reference": dataset_reference,
                "harvest_run_id": "PRIVATE_HARVEST_RUN",
                "source_id": "PRIVATE_SOURCE_ID",
                "source_title": "PRIVATE_SOURCE_TITLE",
                "accepted_count": 2500,
                "quarantined_count": 2,
                "status": "COMPLETE",
                "file_uri": "file:///C:/private/source.zip",
                "secret_key": "PRIVATE_INTAKE_SECRET",
            }
        ],
        "jobs": [hostile_job],
        "count_policy": {"minimum": 2000, "target": 2500, "maximum": 3000, "secret": "PRIVATE_POLICY"},
        "taxonomy": ["weapon", "PRIVATE_TAXONOMY"],
        "config_sha256": "0" * 64,
        "api_token": "PRIVATE_INVENTORY_TOKEN",
    }

    class Service:
        def inventory(self) -> dict[str, Any]:
            return hostile_inventory

        def job(self, selected_job_id: str) -> dict[str, Any]:
            assert selected_job_id == job_id
            return hostile_job

        def preview(self, references: list[str]) -> dict[str, Any]:
            assert references == [dataset_reference]
            return {
                "dataset_references": references,
                "source_ids": ["PRIVATE_SOURCE_ID"],
                "eligible_unique_images": 2500,
                "selected_images": 2500,
                "ready_to_build": True,
                "near_duplicate_exclusions": [{"source_path": "C:/private/source.png"}],
                "blockers": ["PRIVATE_BLOCKER"],
                "labels_are_human_truth": False,
                "secret": "PRIVATE_PREVIEW_SECRET",
            }

    def assert_closed(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = key.casefold()
                assert "secret" not in lowered
                assert "token" not in lowered
                assert "password" not in lowered
                assert "authorization" not in lowered
                assert "title" not in lowered
                assert "uri" not in lowered
                assert "url" not in lowered
                assert "path" not in lowered or key == "paths_exposed"
                assert_closed(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_closed(nested)
        elif isinstance(value, str):
            assert not value.casefold().startswith("file:")
            assert not value.startswith("/")
            assert not value.startswith("\\\\")
            assert "C:/private" not in value
            assert "C:\\private" not in value
            assert "PRIVATE_" not in value
            assert "secret_" not in value.casefold()

    context = ProjectContext(tmp_path, deepcopy(DEFAULT_CONFIG), tmp_path / "spritelab.yaml", tmp_path / "runs/v3")
    app = FastAPI()
    app.include_router(create_router(context, service=Service()))  # type: ignore[arg-type]
    client = TestClient(app)

    page = client.get("/dataset-v5")
    assert page.status_code == 200
    assert "PRIVATE_" not in page.text
    assert "file:///C:/private" not in page.text

    inventory = client.get("/dataset-v5/api/inventory")
    assert inventory.status_code == 200
    inventory_value = inventory.json()
    assert set(inventory_value) == {
        "schema_version",
        "managed_intakes",
        "jobs",
        "count_policy",
        "config_sha256",
        "network_actions",
        "paths_exposed",
    }
    assert set(inventory_value["managed_intakes"][0]) == {
        "dataset_reference",
        "accepted_count",
        "quarantined_count",
        "status",
        "paths_exposed",
    }
    assert set(inventory_value["jobs"][0]) == {
        "job_id",
        "status",
        "stage",
        "current",
        "total",
        "message",
        "paths_exposed",
    }
    assert_closed(inventory_value)

    job = client.get(f"/dataset-v5/api/jobs/{job_id}")
    assert job.status_code == 200
    job_value = job.json()
    assert set(job_value) == {
        "schema_version",
        "job_id",
        "status",
        "stage",
        "current",
        "total",
        "message",
        "candidate",
        "evidence",
        "publication",
        "activated_config_sha256",
        "events",
        "paths_exposed",
    }
    assert set(job_value["candidate"]) == {
        "candidate_identity",
        "payload_inventory_sha256",
        "image_count",
        "paths_exposed",
    }
    assert job_value["stage"] == "unavailable"
    assert job_value["activated_config_sha256"] == "f" * 64
    assert set(job_value["evidence"]) == {"label_audit", "dataset_validation"}
    assert "activation_manifest" not in job_value["publication"]
    assert "campaign_config" not in job_value["publication"]
    assert job_value["publication"]["campaign_launch_ready"] is False
    assert job_value["publication"]["configuration_activated"] is True
    assert job_value["publication"]["training_started"] is True
    assert job_value["publication"]["campaign_seeds"] == [731001, 731002, 731003]
    assert job_value["events"] == []
    assert_closed(job_value)

    preview = client.post("/dataset-v5/api/preview", json={"dataset_references": [dataset_reference]})
    assert preview.status_code == 200
    assert preview.json() == {
        "schema_version": "spritelab.dataset.conditioned-preview-public.v1",
        "dataset_references": [dataset_reference],
        "eligible_unique_images": 2500,
        "selected_images": 2500,
        "ready_to_build": True,
        "blockers": [],
        "labels_are_human_truth": False,
        "paths_exposed": False,
    }
    assert_closed(preview.json())

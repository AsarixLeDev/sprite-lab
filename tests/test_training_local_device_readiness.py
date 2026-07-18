from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.training.config import ComputeSettings
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.plans import (
    TrainingPlanResolver,
    synthetic_training_path_contract_for_tests,
)
from spritelab.remote_compute import LocalComputeBackend
from spritelab.remote_compute import local as local_module
from spritelab.training import campaign as campaign_module
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, plan_campaign, stable_hash
from spritelab.training.launch import prepare_validated_training_launch
from spritelab.v3.config import DEFAULT_CONFIG
from spritelab.v3.model import AuditStatus


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _context(root: Path, policy: str) -> ProjectContext:
    return ProjectContext(
        root,
        {"compute": {"training": {"type": "local", "device_policy": policy}}},
        root / "spritelab.yaml",
        root / "runs/v3",
    )


def _campaign_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    inputs = tmp_path / "inputs"
    dataset, split, vocabulary, benchmark = [
        inputs / name for name in ("dataset.json", "split.json", "vocabulary.json", "benchmark.json")
    ]
    for path, payload in (
        (dataset, {"records": ["sprite"]}),
        (split, {"train": ["sprite"]}),
        (vocabulary, {"tokens": ["sprite"]}),
        (benchmark, {"prompts": ["sprite"]}),
    ):
        _write_json(path, payload)
    optimizer = {"name": "adamw"}
    schedule = {"name": "none", "warmup_steps": 0}
    loss = {"name": "uniform_velocity"}
    determinism = {"mode": "strict"}
    evaluation = {
        "cadence": 5,
        "include_step_zero": False,
        "benchmark_manifest_hash": file_sha256(benchmark),
        "benchmark_manifest_path": str(benchmark),
        "ema_policy": "both",
        "live_weight_evaluation_policy": "required",
    }
    evaluation["evaluation_config_hash"] = stable_hash(
        {key: value for key, value in evaluation.items() if not key.startswith("benchmark_manifest_")}
    )
    spec = {
        "campaign_id": "local-device-readiness",
        "purpose": "Validate the local device gate without starting training.",
        "architecture_cells": [{"cell_id": "base", "comparison_values": {}}],
        "identities": {
            "dataset_view_manifest_hash": file_sha256(dataset),
            "dataset_view_manifest_path": str(dataset),
            "split_manifest_hash": file_sha256(split),
            "split_manifest_path": str(split),
            "conditioning_vocabulary_hash": file_sha256(vocabulary),
            "conditioning_vocabulary_path": str(vocabulary),
            "model_config_hash": stable_hash({}),
            "optimizer_config_hash": stable_hash(optimizer),
            "schedule_config_hash": stable_hash(schedule),
            "loss_config_hash": stable_hash(loss),
            "determinism_config_hash": stable_hash(determinism),
        },
        "seeds": list(DEFAULT_SEEDS),
        "training": {
            "max_optimizer_steps": 10,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "effective_batch_size": 1,
            "precision": "fp32",
            "sampler_policy": "weighted_replacement_v1",
            "positive_sampling_mass_records": 1.0,
        },
        "optimizer": optimizer,
        "schedule": schedule,
        "loss": loss,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": {"cadence": 5},
        "output_root": str(tmp_path / "training-output"),
        "executable": True,
        "launch_authorized": True,
    }
    synthetic_code_identity = {
        "schema_version": "synthetic_training_code_identity_v1",
        "sha256": "a" * 64,
        "files": [],
    }
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: synthetic_code_identity)
    campaign = plan_campaign(spec, execution_root=tmp_path)
    for run in campaign["expected_runs"]:
        _write_json(Path(str(run["resolved_config_path"])), run["resolved_config"])
    config_path = tmp_path / "campaign.json"
    _write_json(config_path, campaign)
    return config_path, campaign


def test_explicit_cuda_probe_uses_fixed_bounded_command_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "nvidia-smi"
    calls: list[list[str]] = []
    monkeypatch.setattr(local_module, "_nvidia_smi_executable", lambda: executable.resolve())

    def run(command: list[str]) -> dict[str, Any]:
        calls.append(command)
        return {"status": "ok", "returncode": 0, "stdout": b"0, GPU-abc123, Default\n"}

    monkeypatch.setattr(local_module, "_run_bounded_probe_command", run)
    before = set(sys.modules)

    capability = LocalComputeBackend().probe(_context(tmp_path, "cuda"))[0]

    assert capability.status is ProductStatus.READY
    assert capability.details == {
        "disk_free_bytes": capability.details["disk_free_bytes"],
        "cuda_initialized": False,
        "device_policy": "cuda",
        "pytorch_compatibility_verified": False,
        "cuda_host_probe": "PASS",
        "reason": "nvidia_smi_pass",
        "usable_nvidia_device_count": 1,
    }
    assert calls == [
        [
            str(executable.resolve()),
            "--id=0",
            "--query-gpu=index,uuid,compute_mode",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert not any(name.startswith("torch") for name in set(sys.modules) - before)


def test_probe_command_discards_stderr_and_caps_stdout_before_killing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_sizes: list[int] = []

    class EndlessStdout:
        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b"x" * size

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = EndlessStdout()
            self.returncode: int | None = None
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, *, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

    process = FakeProcess()
    observed: list[tuple[list[str], dict[str, Any]]] = []

    def popen(command: list[str], **options: Any) -> FakeProcess:
        observed.append((command, options))
        return process

    monkeypatch.setattr(local_module.subprocess, "Popen", popen)
    result = local_module._run_bounded_probe_command(["trusted-nvidia-smi", "--fixed"])

    assert result == {"status": "output_limit"}
    assert process.killed is True
    assert sum(read_sizes) == 16 * 1024 + 1
    assert max(read_sizes) <= 4_096
    command, options = observed[0]
    assert command == ["trusted-nvidia-smi", "--fixed"]
    assert options["shell"] is False
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.PIPE
    assert options["stderr"] is subprocess.DEVNULL


def test_probe_command_captures_small_stdout_with_fixed_process_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SmallStdout:
        def __init__(self) -> None:
            self._chunks = [b"0, GPU-abc, Default\n", b""]

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            return None

    process = SimpleNamespace(
        stdout=SmallStdout(),
        returncode=0,
        poll=lambda: 0,
        wait=lambda timeout: 0,
        kill=lambda: pytest.fail("completed probe must not be killed"),
    )
    observed: list[tuple[list[str], dict[str, Any]]] = []

    def popen(command: list[str], **options: Any) -> SimpleNamespace:
        observed.append((command, options))
        return process

    monkeypatch.setattr(local_module.subprocess, "Popen", popen)
    result = local_module._run_bounded_probe_command(["trusted-nvidia-smi", "--fixed"])

    assert result == {"status": "ok", "returncode": 0, "stdout": b"0, GPU-abc, Default\n"}
    assert observed[0][0] == ["trusted-nvidia-smi", "--fixed"]
    assert observed[0][1]["shell"] is False
    assert observed[0][1]["stderr"] is subprocess.DEVNULL


def test_probe_command_timeout_kills_the_exact_process_and_unblocks_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = local_module.threading.Event()

    class BlockingStdout:
        def read(self, _size: int) -> bytes:
            closed.wait(timeout=1.0)
            return b""

        def close(self) -> None:
            closed.set()

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = BlockingStdout()
            self.returncode: int | None = None
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, *, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

    process = FakeProcess()
    monkeypatch.setattr(local_module, "_NVIDIA_SMI_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(local_module.subprocess, "Popen", lambda _command, **_options: process)

    result = local_module._run_bounded_probe_command(["trusted-nvidia-smi", "--fixed"])

    assert result == {"status": "timeout"}
    assert process.killed is True
    assert closed.is_set()


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("missing", "nvidia_smi_missing"),
        ("timeout", "nvidia_smi_timeout"),
        ("os_error", "nvidia_smi_unavailable"),
        ("failure", "nvidia_smi_failed"),
        ("empty", "nvidia_smi_malformed"),
        ("oversized", "nvidia_smi_output_limit"),
        ("bad_columns", "nvidia_smi_malformed"),
        ("bad_index", "nvidia_smi_malformed"),
        ("bad_uuid", "nvidia_smi_malformed"),
        ("bad_mode", "nvidia_smi_malformed"),
        ("prohibited", "nvidia_smi_compute_prohibited"),
    ],
)
def test_explicit_cuda_probe_fails_closed_for_unproved_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_reason: str,
) -> None:
    executable = tmp_path / "nvidia-smi"
    monkeypatch.setattr(
        local_module,
        "_nvidia_smi_executable",
        lambda: None if case == "missing" else executable,
    )
    outputs = {
        "timeout": {"status": "timeout"},
        "os_error": {"status": "unavailable"},
        "failure": {"status": "ok", "returncode": 1, "stdout": b""},
        "empty": {"status": "ok", "returncode": 0, "stdout": b""},
        "oversized": {"status": "output_limit"},
        "bad_columns": {"status": "ok", "returncode": 0, "stdout": b"0, GPU-abc\n"},
        "bad_index": {"status": "ok", "returncode": 0, "stdout": b"one, GPU-abc, Default\n"},
        "bad_uuid": {"status": "ok", "returncode": 0, "stdout": b"0, not-a-gpu, Default\n"},
        "bad_mode": {"status": "ok", "returncode": 0, "stdout": b"0, GPU-abc, Mystery\n"},
        "prohibited": {"status": "ok", "returncode": 0, "stdout": b"0, GPU-abc, Prohibited\n"},
    }
    monkeypatch.setattr(
        local_module,
        "_run_bounded_probe_command",
        lambda _command: pytest.fail("missing executable must not spawn") if case == "missing" else outputs[case],
    )

    capability = LocalComputeBackend().probe(_context(tmp_path, "cuda"))[0]

    assert capability.status is ProductStatus.UNAVAILABLE
    assert capability.details["reason"] == expected_reason
    assert capability.details["cuda_host_probe"] == "FAIL"
    assert capability.details["cuda_initialized"] is False
    assert capability.details["pytorch_compatibility_verified"] is False
    assert "private" not in capability.message
    assert "private" not in json.dumps(dict(capability.details))


@pytest.mark.parametrize("policy", ["auto", "cpu"])
def test_auto_and_cpu_probe_never_run_nvidia_smi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    monkeypatch.setattr(
        local_module,
        "_nvidia_smi_executable",
        lambda: pytest.fail("passive CPU/auto capability must not resolve nvidia-smi"),
    )
    capability = LocalComputeBackend().probe(_context(tmp_path, policy))[0]
    assert capability.status is ProductStatus.READY
    assert capability.details["cuda_host_probe"] == "NOT_REQUIRED"
    assert capability.details["reason"] == ("cpu_forced" if policy == "cpu" else "auto_allows_cpu_fallback")


def test_local_device_policies_have_distinct_receipt_bound_environments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_config, campaign = _campaign_config(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    auto_environment = ComputeSettings(device_policy="auto").execution_environment()
    cpu_environment = ComputeSettings(device_policy="cpu").execution_environment()
    cuda_environment = ComputeSettings(device_policy="cuda").execution_environment()

    assert auto_environment == {
        "SPRITELAB_PREVIEW_INTERVAL": "500",
        "SPRITELAB_DEVICE_POLICY": "auto",
    }
    assert cpu_environment == {
        "SPRITELAB_PREVIEW_INTERVAL": "500",
        "SPRITELAB_DEVICE_POLICY": "cpu",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    assert cuda_environment == {
        "SPRITELAB_PREVIEW_INTERVAL": "500",
        "SPRITELAB_DEVICE_POLICY": "cuda",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    auto = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=tmp_path,
        execute_confirmed=True,
        environment=auto_environment,
    )
    cuda = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=tmp_path,
        execute_confirmed=True,
        environment=cuda_environment,
    )

    assert auto.argv == cuda.argv
    assert auto.output_root == cuda.output_root
    assert auto.validator_context.environment != cuda.validator_context.environment
    assert auto.receipt.execution_spec_sha256 != cuda.receipt.execution_spec_sha256


def test_start_time_resolver_blocks_explicit_cuda_when_host_probe_is_unproved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_config, campaign = _campaign_config(tmp_path, monkeypatch)
    freeze = tmp_path / "freeze.json"
    _write_json(freeze, {"image_count": 2_417})
    values = deepcopy(DEFAULT_CONFIG)
    values["dataset"]["freeze_manifest"] = freeze.relative_to(tmp_path).as_posix()
    values["training"]["dataset_freeze"] = freeze.relative_to(tmp_path).as_posix()
    values["training"]["campaign_config"] = campaign_config.relative_to(tmp_path).as_posix()
    values["compute"]["training"] = {"type": "local", "device_policy": "cuda"}
    values["execution"]["allow_training"] = True
    values.update(synthetic_training_path_contract_for_tests(tmp_path))
    context = ProjectContext(tmp_path, values, tmp_path / "spritelab.yaml", tmp_path / "runs/v3")
    monkeypatch.setattr(local_module, "_nvidia_smi_executable", lambda: None)

    resolver = TrainingPlanResolver(
        activation_loader=lambda *_args, **_kwargs: SimpleNamespace(
            audit_status=AuditStatus.PASS,
            campaign=campaign,
            manifest={"image_count": 2_417},
        )
    )
    plan = resolver.resolve(
        context,
        TrainingProfile.RECOMMENDED,
        LocalComputeBackend(),
        probe_backend=True,
    )

    assert plan.ready is False
    device_gate = next(gate for gate in plan.gates if gate.gate_id == "device")
    assert device_gate.passed is False
    assert "NVIDIA host probe" in device_gate.message
    assert all(not Path(str(run["output_root"])).exists() for run in campaign["expected_runs"])

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import marshal
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import spritelab.training.smoke_bundle as smoke_bundle_module
import spritelab.training.smoke_runner as smoke_runner_module
from spritelab.product_features.evaluation.exploratory_smoke import (
    ExploratorySmokeWorkflow,
    SmokePreparationRequest,
    SmokeRegistrationRequest,
)
from spritelab.product_features.evaluation.models import CheckpointCatalog
from spritelab.product_features.evaluation.playground import PlaygroundService
from spritelab.training.smoke_bundle import (
    SmokeBundleError,
    artifact_bundle_directory,
    begin_device_run,
    build_smoke_child_environment,
    expected_manifest,
    load_plan,
    load_playground_registration,
    pinned_smoke_interpreter,
    portable_relative_parts,
    read_stable_single_link_bytes,
    run_bundle_directory,
    smoke_launch_identity,
    smoke_training_argv,
    smoke_worker_argv,
    stable_hash,
    verify_pinned_process_image,
    verify_prepared_runtime_closure,
    write_device_receipt,
)
from spritelab.training.smoke_runner import ExploratorySmokeRunner, SmokeExecutionError
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation
from spritelab.v3.config import DEFAULT_CONFIG


def _sha(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _contained_child_environment(
    fixture: SmokeFixture,
    plan: dict[str, Any],
    device: str,
):
    environment = build_smoke_child_environment(fixture.root, plan, device)
    descriptors: list[int] = []
    try:
        if sys.platform.startswith("linux"):
            flags = (
                int(getattr(os, "O_PATH", os.O_RDONLY))
                | int(getattr(os, "O_DIRECTORY", 0))
                | int(getattr(os, "O_NOFOLLOW", 0))
            )
            for relative in plan["configurations"][device]["writable_roots"]:
                descriptors.append(os.open(fixture.root / relative, flags))
            environment["SPRITELAB_WRITABLE_ROOT_FDS"] = ",".join(str(value) for value in descriptors)
        yield environment, tuple(descriptors)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _direct_bound_child_command(
    fixture: SmokeFixture,
    plan: dict[str, Any],
    device: str,
    argv: list[str],
) -> tuple[list[str], Path]:
    """Exercise the inner loader after the Windows outer bootstrap contract."""

    if os.name != "nt":
        return [sys.executable, *argv[1:]], fixture.root
    assert argv[1:5] == ["-I", "-B", "-S", "-c"]
    source = (
        "import sys as _spritelab_test_sys\n"
        f"_spritelab_test_sys._spritelab_windows_project_root={os.fspath(fixture.root)!r}\n" + argv[5]
    )
    execution = artifact_bundle_directory(fixture.root, str(plan["smoke_id"])) / "execution" / device
    return [sys.executable, *argv[1:5], source, *argv[6:]], execution


@dataclass
class SmokeFixture:
    root: Path
    workflow: ExploratorySmokeWorkflow
    request: SmokePreparationRequest
    job: dict[str, Any]
    activation: Any
    activation_calls: list[bool]
    real_config_before: bytes

    def prepare(self, nonce: str = "nonce-00000001") -> dict[str, Any]:
        return self.workflow.prepare(
            SmokePreparationRequest(
                **self.publication_fields(),
                preparation_nonce=nonce,
                explicit_action=True,
            )
        )

    def publication_fields(self) -> dict[str, str]:
        return {
            "conditioned_job_id": self.request.conditioned_job_id,
            "candidate_identity_sha256": self.request.candidate_identity_sha256,
            "publication_identity_sha256": self.request.publication_identity_sha256,
            "activation_manifest_sha256": self.request.activation_manifest_sha256,
            "campaign_config_sha256": self.request.campaign_config_sha256,
            "campaign_identity_sha256": self.request.campaign_identity_sha256,
        }

    def registration_request(
        self,
        prepared: dict[str, Any],
        receipts: dict[str, dict[str, Any]],
    ) -> SmokeRegistrationRequest:
        return SmokeRegistrationRequest(
            **self.publication_fields(),
            smoke_id=prepared["smoke_id"],
            plan_identity=prepared["plan_identity"],
            cpu_receipt_identity=receipts["cpu"]["receipt_identity"],
            cuda_receipt_identity=receipts["cuda"]["receipt_identity"],
            explicit_action=True,
        )


@pytest.fixture
def smoke_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SmokeFixture:
    root = tmp_path / "project"
    root.mkdir()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = root / "spritelab.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    real_config_before = config_path.read_bytes()
    for relative in (
        "src/spritelab/__init__.py",
        "src/spritelab/__main__.py",
        "src/spritelab/training/__init__.py",
        "src/spritelab/training/cli/experiment_cmds.py",
        "src/spritelab/training/generator_challenger.py",
        "src/spritelab/training/smoke_bundle.py",
        "src/spritelab/training/smoke_runner.py",
        "src/spritelab/training/smoke_worker.py",
        "src/spritelab/utils/pinned_executable.py",
    ):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# isolated smoke fixture: {relative}\n", encoding="utf-8")
    (root / "src/spritelab/__main__.py").write_text(
        "print('ISOLATED_BOOTSTRAP_OK')\n",
        encoding="utf-8",
    )
    runtime_root = root / "runtime" / "site-packages"
    runtime_root.mkdir(parents=True)
    stdlib_root = root / "runtime" / "stdlib"
    stdlib_root.mkdir()
    (stdlib_root / "runpy.py").write_bytes(Path(runpy.__file__).read_bytes())
    import io

    (stdlib_root / "io.py").write_bytes(Path(io.__file__).read_bytes())
    (stdlib_root / "bound_stdlib.py").write_text("VALUE = 'bound-stdlib'\n", encoding="utf-8")
    stdlib_cache = stdlib_root / "__pycache__"
    stdlib_cache.mkdir()
    (stdlib_cache / "bound_stdlib.test.pyc").write_bytes(b"bound-pyc-inventory")
    sourceless_payload = (
        importlib.util.MAGIC_NUMBER
        + b"\0" * 12
        + marshal.dumps(compile("print('SOURCELESS_PYC_EXECUTED')\n", "sourceless_only.pyc", "exec"))
    )
    (stdlib_root / "sourceless_only.pyc").write_bytes(sourceless_payload)
    for distribution_name, import_name in (
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("PyYAML", "yaml"),
        ("torch", "torch"),
    ):
        package = runtime_root / import_name
        package.mkdir()
        (package / "__init__.py").write_text(
            f"__version__ = '1.0'  # {distribution_name}\n",
            encoding="utf-8",
        )
        info = runtime_root / f"{distribution_name.replace('-', '_')}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 1.0\n",
            encoding="utf-8",
        )
        records = [
            f"{import_name}/__init__.py,,",
            f"{info.name}/METADATA,,",
            f"{info.name}/RECORD,,",
        ]
        (info / "RECORD").write_text("\n".join(records) + "\n", encoding="utf-8")
    import spritelab.training.smoke_bundle as smoke_bundle_module

    monkeypatch.setattr(
        smoke_bundle_module,
        "_isolated_import_paths",
        lambda project: [str(Path(project).resolve() / "src"), str(runtime_root.resolve())],
    )
    monkeypatch.setattr(
        smoke_bundle_module,
        "_standard_runtime_root_specs",
        lambda: [
            (
                stdlib_root.resolve(),
                ("destshared", "platstdlib", "runtime-libraries", "stdlib"),
                ("__pycache__", "bound_stdlib.py", "io.py", "runpy.py", "sourceless_only.pyc"),
            )
        ],
    )

    artifacts: dict[str, Path] = {}
    for key in ("view_manifest", "split_manifest", "conditioning_vocabulary", "benchmark_manifest"):
        path = root / "artifacts" / "conditioned" / f"{key}.json"
        _write_json(path, {"artifact": key})
        artifacts[key] = path
    activation_path = root / "artifacts" / "conditioned" / "activation.json"
    campaign_path = root / "artifacts" / "conditioned" / "campaign.json"
    _write_json(activation_path, {"schema_version": "test.activation.v1"})
    _write_json(campaign_path, {"schema_version": "test.campaign.v1"})

    identities = {
        "dataset_view_manifest_hash": _sha(artifacts["view_manifest"].read_bytes()),
        "split_manifest_hash": _sha(artifacts["split_manifest"].read_bytes()),
        "conditioning_vocabulary_hash": _sha(artifacts["conditioning_vocabulary"].read_bytes()),
    }
    campaign_identity = _sha("campaign")
    code_identity = {
        "schema_version": "spritelab_training_code_identity_v4",
        "contract": "all_tracked_production_python_v5_with_untracked_rejection",
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "binding": "whole_file",
                "semantic_role": "isolated smoke test production source",
                "sha256": _sha(path.read_bytes()),
            }
            for path in sorted((root / "src" / "spritelab").rglob("*.py"))
        ],
    }
    code_identity["sha256"] = stable_hash(code_identity)
    base_config = {
        "name": "conditioned-campaign-run",
        "dataset": {"training_manifest": "artifacts/conditioned/split_manifest.json"},
        "conditioning": {"mode": "caption"},
        "runtime": {
            "max_steps": 5000,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "determinism": "strict",
            "device": "cuda",
            "out_dir": "runs/v3/training/full-run-001",
        },
    }
    campaign = {
        "campaign_identity": campaign_identity,
        "code_identity": code_identity,
        "identities": identities,
        "evaluation": {"benchmark_manifest_hash": _sha(artifacts["benchmark_manifest"].read_bytes())},
        "expected_runs": [
            {
                "run_id": "full-run-001",
                "run_identity": _sha("full-run-001"),
                "resolved_config": base_config,
                "resolved_config_sha256": stable_hash(base_config),
                "seed": 7,
                "output_root": "runs/v3/training/full-run-001",
            }
        ],
    }
    activation = SimpleNamespace(
        freeze_sha256=_sha(activation_path.read_bytes()),
        campaign_config_sha256=_sha(campaign_path.read_bytes()),
        campaign=campaign,
        artifacts=artifacts,
    )
    candidate_identity = _sha("conditioned-candidate")
    publication_identity = _sha("conditioned-publication")
    job = {
        "job_id": "conditioned-job-0001",
        "status": "COMPLETE",
        "candidate": {"candidate_identity": candidate_identity},
        "publication": {
            "publication_identity_sha256": publication_identity,
            "activation_manifest": "artifacts/conditioned/activation.json",
            "activation_manifest_sha256": activation.freeze_sha256,
            "campaign_config": "artifacts/conditioned/campaign.json",
            "campaign_config_sha256": activation.campaign_config_sha256,
            "campaign_identity_sha256": campaign_identity,
            "configuration_activated": False,
        },
    }
    activation_calls: list[bool] = []

    def activation_loader(_config: Any, *, require_audit: bool) -> Any:
        activation_calls.append(require_audit)
        return activation

    def manifest_builder(
        _path: Path,
        *,
        write: bool,
        config: dict[str, Any],
        runtime_overrides: dict[str, Any],
        resolution_root: Path,
    ) -> dict[str, Any]:
        assert write is False
        assert resolution_root == root
        effective = copy.deepcopy(config)
        effective["runtime"].update(runtime_overrides)
        return {
            "schema_version": "test.smoke-manifest.v1",
            "name": effective["name"],
            "runtime": effective["runtime"],
            "experiment_hash": stable_hash(effective),
        }

    request = SmokePreparationRequest(
        conditioned_job_id="conditioned-job-0001",
        candidate_identity_sha256=candidate_identity,
        publication_identity_sha256=publication_identity,
        activation_manifest_sha256=activation.freeze_sha256,
        campaign_config_sha256=activation.campaign_config_sha256,
        campaign_identity_sha256=campaign_identity,
        preparation_nonce="nonce-00000001",
        explicit_action=True,
    )
    workflow = ExploratorySmokeWorkflow(
        root,
        job_loader=lambda _job_id: job,
        job_inventory_loader=lambda: [job],
        activation_loader=activation_loader,
        manifest_builder=manifest_builder,
    )
    return SmokeFixture(root, workflow, request, job, activation, activation_calls, real_config_before)


def _complete_outputs(fixture: SmokeFixture, prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plan = load_plan(fixture.root, prepared["smoke_id"])
    return {device: _complete_device(fixture, plan, device) for device in ("cpu", "cuda")}


def _complete_device(fixture: SmokeFixture, plan: dict[str, Any], device: str) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    output = begin_device_run(fixture.root, plan, device)
    manifest = expected_manifest(fixture.root, plan, device)
    _write_json(output / "config.json", {"device": device, "smoke": True})
    _write_json(
        output / "train_report.json",
        {
            "model_type": "generator_challenger",
            "steps_completed": 2,
            "max_steps": 2,
            "device": device,
            "loss": 0.25,
            "determinism": {"mode": "strict", "qualified": True, "issues": []},
        },
    )
    (output / "train_metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 0.5}) + "\n" + json.dumps({"step": 2, "loss": 0.25}) + "\n",
        encoding="utf-8",
    )
    for name, variant, ema in (
        ("checkpoint_step_000002.pt", "step", False),
        ("checkpoint_step_000002_ema.pt", "step_ema", True),
    ):
        torch.save(
            {
                "model_type": "generator_challenger",
                "experiment_manifest": manifest,
                "step": 2,
                "global_step": 2,
                "checkpoint_variant": variant,
                "ema_weights": ema,
                "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            },
            output / name,
        )
    if device == "cuda":
        (output / "cuda_determinism_qualification.json").write_bytes(
            smoke_bundle_module.canonical_json_bytes(
                {
                    "qualified": True,
                    "mode": "strict",
                    "device": "cuda",
                    "steps": 2,
                    "interrupted_after": 1,
                    "repeated_forward_backward_bit_exact": True,
                    "resume_bit_exact": True,
                    "environment": {
                        "platform": "test-platform",
                        "torch_version": str(torch.__version__),
                        "cuda_runtime_version": "12.0",
                        "cuda_driver_version": 12000,
                        "cudnn_version": 9000,
                        "gpus": [
                            {
                                "index": 0,
                                "name": "test-gpu",
                                "compute_capability": "8.0",
                                "total_memory_bytes": 8 * 1024**3,
                            }
                        ],
                    },
                    "guarantee_scope": "same GPU model, driver, CUDA, cuDNN, Torch, code, and inputs only",
                    "cross_gpu_or_version_identity_claimed": False,
                },
                pretty=True,
            )
        )
    record = plan["configurations"][device]
    return write_device_receipt(
        fixture.root,
        plan,
        device,
        config_sha256_before=record["config_sha256"],
        config_sha256_after=record["config_sha256"],
        environment=build_smoke_child_environment(fixture.root, plan, device),
    )


def _activate_fixture(fixture: SmokeFixture) -> None:
    fixture.job["publication"]["configuration_activated"] = True
    values = copy.deepcopy(DEFAULT_CONFIG)
    values["dataset"]["freeze_manifest"] = "artifacts/conditioned/activation.json"
    values["training"]["dataset_freeze"] = "artifacts/conditioned/activation.json"
    values["training"]["campaign_config"] = "artifacts/conditioned/campaign.json"
    values["execution"]["allow_training"] = True
    (fixture.root / "spritelab.yaml").write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _execution_identities(smoke_id: str) -> dict[str, str]:
    return {device: _sha(f"server-execution:{smoke_id}:{device}") for device in ("cpu", "cuda")}


def _register(fixture: SmokeFixture, request: SmokeRegistrationRequest) -> dict[str, Any]:
    return fixture.workflow.register(
        request,
        server_execution_identities=_execution_identities(request.smoke_id),
    )


def test_prepare_is_fixed_idempotent_and_never_touches_production_roots(smoke_fixture: SmokeFixture) -> None:
    first = smoke_fixture.prepare()
    second = smoke_fixture.prepare()

    assert first == second
    assert first["status"] == "PREPARED"
    assert first["production_eligible"] is False
    assert first["evaluation_eligible"] is False
    assert first["training_resume_eligible"] is False
    assert first["promotion_eligible"] is False
    assert smoke_fixture.root.as_posix() not in json.dumps(first)
    assert (smoke_fixture.root / "spritelab.yaml").read_bytes() == smoke_fixture.real_config_before
    assert not (smoke_fixture.root / "runs/v3/training/full-run-001").exists()
    plan = load_plan(smoke_fixture.root, first["smoke_id"])
    assert plan["configurations"]["cpu"]["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert plan["configurations"]["cpu"]["environment"]["SPRITELAB_PROGRESS"] == "0"
    assert plan["configurations"]["cpu"]["environment"]["PYTHONNOUSERSITE"] == "1"
    assert plan["configurations"]["cpu"]["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert plan["configurations"]["cuda"]["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert plan["configurations"]["cpu"]["child_environment"]["temporary_root"].startswith("artifacts/")
    assert smoke_fixture.root.as_posix() not in json.dumps(plan["configurations"]["cpu"]["child_environment"])
    assert plan["interpreter"]["isolated_flags"] == ["-I", "-B", "-S"]
    assert plan["interpreter"]["byte_count"] > 0
    assert str(Path(sys.executable).parent) not in json.dumps(plan["interpreter"])
    assert plan["runtime_closure"]["paths_exposed"] is False
    assert smoke_fixture.root.as_posix() not in json.dumps(plan["runtime_closure"])
    fresh = smoke_fixture.prepare("nonce-00000002")
    assert fresh["smoke_id"] != first["smoke_id"]
    recovered = smoke_fixture.workflow.prepared_plans()
    assert {item["smoke_id"] for item in recovered["eligible"]} == {first["smoke_id"], fresh["smoke_id"]}


@pytest.mark.parametrize(
    "relative",
    (
        "",
        ".",
        "../escape",
        "/absolute",
        "C:drive-relative",
        "C:/absolute",
        "//server/share",
        "\\\\server\\share",
        "\\\\?\\C:\\device",
        "\\\\.\\pipe\\name",
        "mixed\\separator/file",
        "double//separator",
        "CON.txt",
        "nested/aux.json",
        "trailing-dot.",
        "trailing-space ",
        "invalid<name",
        "invalid>name",
        'invalid"name',
        "invalid|name",
        "invalid?name",
        "invalid*name",
        "decomposed-e\u0301.txt",
    ),
)
def test_portable_relative_path_rejects_hostile_windows_and_unicode_grammar(relative: str) -> None:
    with pytest.raises(SmokeBundleError, match="server-owned smoke reference"):
        portable_relative_parts(relative)


def test_portable_relative_path_accepts_canonical_paths_and_rejects_alias_collisions() -> None:
    assert portable_relative_parts("artifacts/training/smokes/configs/cpu.json") == (
        "artifacts",
        "training",
        "smokes",
        "configs",
        "cpu.json",
    )
    with pytest.raises(SmokeBundleError, match="collision"):
        smoke_bundle_module._reject_portable_collisions(("Package/File.py", "package/file.PY"))
    with pytest.raises(SmokeBundleError, match="collision"):
        smoke_bundle_module._reject_portable_collisions(("Straße.py", "STRASSE.py"))


def test_runtime_closure_rejects_same_size_file_drift(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-runtime-same-size-drift")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    target = smoke_fixture.root / "runtime/site-packages/torch/__init__.py"
    original = target.read_bytes()
    replacement = bytes((byte ^ 1) if index == 0 else byte for index, byte in enumerate(original))
    assert len(replacement) == len(original)
    target.write_bytes(replacement)

    with pytest.raises(SmokeBundleError, match="exact bound runtime files changed"):
        verify_prepared_runtime_closure(smoke_fixture.root, plan["runtime_closure"])


def test_runtime_closure_rejects_unowned_file_added_after_preparation(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-runtime-unowned")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    (smoke_fixture.root / "runtime/site-packages/torch/hostile.py").write_text(
        "raise RuntimeError('must never import')\n",
        encoding="utf-8",
    )

    with pytest.raises(SmokeBundleError, match="exact bound runtime files changed"):
        verify_prepared_runtime_closure(smoke_fixture.root, plan["runtime_closure"])


@pytest.mark.parametrize(
    "relative",
    ("bound_stdlib.py", "__pycache__/bound_stdlib.test.pyc"),
)
def test_runtime_closure_rejects_same_size_stdlib_and_pyc_drift(
    smoke_fixture: SmokeFixture,
    relative: str,
) -> None:
    prepared = smoke_fixture.prepare(f"nonce-stdlib-drift-{hashlib.sha256(relative.encode()).hexdigest()[:8]}")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    target = smoke_fixture.root / "runtime/stdlib" / relative
    original = target.read_bytes()
    replacement = bytes((byte ^ 1) if index == len(original) - 1 else byte for index, byte in enumerate(original))
    assert replacement != original
    assert len(replacement) == len(original)
    target.write_bytes(replacement)

    with pytest.raises(SmokeBundleError, match="exact bound runtime files changed"):
        verify_prepared_runtime_closure(smoke_fixture.root, plan["runtime_closure"])


def test_runtime_scan_holds_the_original_tree_and_rejects_root_rename_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    package = runtime / "package"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 'original'\n", encoding="utf-8")
    original_module_bytes = (package / "module.py").read_bytes()
    outside_sentinel = tmp_path / "outside-sentinel.bin"
    outside_before = b"outside-must-remain-identical"
    outside_sentinel.write_bytes(outside_before)
    entered = threading.Event()
    release = threading.Event()
    original_hash = smoke_bundle_module._hash_runtime_anchored_file

    def paused_hash(anchor: AnchoredDirectory, name: str) -> tuple[str, int]:
        entered.set()
        assert release.wait(timeout=5)
        return original_hash(anchor, name)

    monkeypatch.setattr(smoke_bundle_module, "_hash_runtime_anchored_file", paused_hash)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(smoke_bundle_module._scan_runtime_files_with_identity, runtime, ("package",))
        assert entered.wait(timeout=5)
        moved = tmp_path / "runtime-original-held"
        try:
            runtime.rename(moved)
        except OSError:
            rename_succeeded = False
        else:
            rename_succeeded = True
            replacement = runtime / "package"
            replacement.mkdir(parents=True)
            (replacement / "module.py").write_text("VALUE = 'replacement'\n", encoding="utf-8")
        finally:
            release.set()
        if rename_succeeded:
            with pytest.raises(SmokeBundleError, match="root path changed"):
                future.result(timeout=5)
        else:
            inventory, _metadata = future.result(timeout=5)
            assert inventory["package/module.py"]["sha256"] == _sha(original_module_bytes)
    assert outside_sentinel.read_bytes() == outside_before


def test_register_requires_exact_receipts_and_is_idempotent(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    request = smoke_fixture.registration_request(prepared, receipts)

    first = _register(smoke_fixture, request)
    second = _register(smoke_fixture, request)

    assert first == second
    assert first["playground_eligible"] is True
    assert first["evaluation_eligible"] is False
    assert first["training_resume_eligible"] is False
    assert first["promotion_eligible"] is False
    catalog = smoke_fixture.workflow.catalog()
    assert {item.weights for item in catalog.eligible} == {"live", "ema"}
    assert catalog.to_dict()["production_catalog_merged"] is False
    assert CheckpointCatalog((), (), None).find(first["registration_id"]) is None
    playground = PlaygroundService(
        CheckpointCatalog((), (), None),
        output_root=smoke_fixture.root / "runs/v3/playground/generations",
        exploratory_catalog=catalog,
    )
    assert playground.defaults()["checkpoint_id"] == catalog.default_checkpoint_id
    assert (smoke_fixture.root / "spritelab.yaml").read_bytes() == smoke_fixture.real_config_before
    assert not (smoke_fixture.root / "runs/v3/training/full-run-001").exists()


def test_receipt_replay_and_cross_bundle_swap_are_rejected(smoke_fixture: SmokeFixture) -> None:
    first = smoke_fixture.prepare("nonce-00000011")
    first_receipts = _complete_outputs(smoke_fixture, first)
    second = smoke_fixture.prepare("nonce-00000012")
    second_receipts = _complete_outputs(smoke_fixture, second)
    request = smoke_fixture.registration_request(first, first_receipts)
    swapped = SmokeRegistrationRequest(
        **smoke_fixture.publication_fields(),
        smoke_id=request.smoke_id,
        plan_identity=request.plan_identity,
        cpu_receipt_identity=second_receipts["cpu"]["receipt_identity"],
        cuda_receipt_identity=request.cuda_receipt_identity,
        explicit_action=True,
    )

    with pytest.raises(SmokeBundleError, match="receipt"):
        _register(smoke_fixture, swapped)


def test_snapshot_fault_after_evidence_converges_on_retry(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    request = smoke_fixture.registration_request(prepared, receipts)
    import spritelab.product_features.evaluation.exploratory_smoke as module

    publish = module.publish_playground_snapshot
    monkeypatch.setattr(
        module, "publish_playground_snapshot", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fault"))
    )
    with pytest.raises(OSError, match="fault"):
        _register(smoke_fixture, request)
    evidence = artifact_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "smoke_evidence.json"
    assert evidence.is_file()
    monkeypatch.setattr(module, "publish_playground_snapshot", publish)

    result = _register(smoke_fixture, request)

    assert result["playground_eligible"] is True
    assert smoke_fixture.workflow.catalog().default_checkpoint_id is not None


def test_postactivation_catalog_requires_current_pass_audit(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    _activate_fixture(smoke_fixture)

    eligible = smoke_fixture.workflow.catalog()

    assert len(eligible.eligible) == 2
    assert smoke_fixture.activation_calls[-1] is True

    def stale_audit(_config: Any, *, require_audit: bool) -> Any:
        if require_audit:
            raise ValueError("stale or missing audit")
        return smoke_fixture.activation

    stale = ExploratorySmokeWorkflow(
        smoke_fixture.root,
        job_loader=lambda _job_id: smoke_fixture.job,
        activation_loader=stale_audit,
    )
    unavailable = stale.catalog()
    assert unavailable.eligible == ()
    assert unavailable.unavailable_count == 1


def test_catalog_rejects_tamper_and_hardlink(smoke_fixture: SmokeFixture, tmp_path: Path) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    result = _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    snapshot = (
        smoke_fixture.root
        / "runs/v3/playground/exploratory-checkpoints"
        / result["registration_id"]
        / "checkpoint_step_000002.pt"
    )
    hardlink = tmp_path / "checkpoint-hardlink.pt"
    try:
        os.link(snapshot, hardlink)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    catalog = smoke_fixture.workflow.catalog()

    assert catalog.eligible == ()
    assert catalog.unavailable_count == 1


def test_catalog_rejects_checkpoint_byte_tamper(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    result = _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    snapshot = (
        smoke_fixture.root
        / "runs/v3/playground/exploratory-checkpoints"
        / result["registration_id"]
        / "checkpoint_step_000002_ema.pt"
    )
    with snapshot.open("ab") as handle:
        handle.write(b"tamper")

    catalog = smoke_fixture.workflow.catalog()

    assert catalog.eligible == ()
    assert catalog.unavailable_count == 1


@pytest.mark.parametrize("binding", ["code", "freeze", "campaign"])
def test_catalog_rejects_stale_code_freeze_or_campaign_binding(
    smoke_fixture: SmokeFixture,
    binding: str,
) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    if binding == "code":
        smoke_fixture.activation.campaign["code_identity"]["sha256"] = _sha("changed-code")
    elif binding == "freeze":
        smoke_fixture.activation.freeze_sha256 = _sha("changed-freeze")
    else:
        smoke_fixture.activation.campaign["campaign_identity"] = _sha("changed-campaign")

    catalog = smoke_fixture.workflow.catalog()

    assert catalog.eligible == ()
    assert catalog.unavailable_count == 1


def test_prepare_rejects_campaign_output_path_escape(smoke_fixture: SmokeFixture) -> None:
    smoke_fixture.activation.campaign["expected_runs"][0]["output_root"] = "../outside/full-run"

    with pytest.raises(SmokeBundleError, match="invalid"):
        smoke_fixture.prepare()


def test_concurrent_prepare_and_register_converge(smoke_fixture: SmokeFixture) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared_values = list(pool.map(lambda _index: smoke_fixture.prepare(), range(4)))
    assert all(value == prepared_values[0] for value in prepared_values)
    receipts = _complete_outputs(smoke_fixture, prepared_values[0])
    request = smoke_fixture.registration_request(prepared_values[0], receipts)

    with ThreadPoolExecutor(max_workers=4) as pool:
        registered_values = list(pool.map(lambda _index: _register(smoke_fixture, request), range(4)))

    assert all(value == registered_values[0] for value in registered_values)
    assert len(smoke_fixture.workflow.catalog().eligible) == 2


@pytest.mark.parametrize("target", ["plan", "device", "catalog"])
def test_ancestor_rename_to_outside_symlink_fails_closed(
    smoke_fixture: SmokeFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    prepared = smoke_fixture.prepare()
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    if target == "device":
        begin_device_run(smoke_fixture.root, plan, "cpu")
    elif target == "catalog":
        receipts = _complete_outputs(smoke_fixture, prepared)
        _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    seam = "artifacts" if target == "plan" else "runs"
    original_directory = smoke_fixture.root / seam
    renamed_directory = smoke_fixture.root / f"{seam}-held-by-test"
    outside = tmp_path / f"outside-{target}"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside-byte-identical", encoding="utf-8")
    original_open = AnchoredDirectory.open_directory_immovable
    attacked = False

    def swap_then_open(anchor: AnchoredDirectory, child: str) -> AnchoredDirectory:
        nonlocal attacked
        if not attacked and child == seam:
            attacked = True
            os.rename(original_directory, renamed_directory)
            try:
                os.symlink(outside, original_directory, target_is_directory=True)
            except OSError as exc:
                pytest.skip(f"directory symlinks unavailable: {exc}")
        return original_open(anchor, child)

    monkeypatch.setattr(AnchoredDirectory, "open_directory_immovable", swap_then_open)
    if target == "plan":
        with pytest.raises((OSError, SmokeBundleError, UnsafeFilesystemOperation)):
            load_plan(smoke_fixture.root, prepared["smoke_id"])
    elif target == "device":
        with pytest.raises((OSError, SmokeBundleError, UnsafeFilesystemOperation)):
            read_stable_single_link_bytes(
                run_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "cpu/smoke_run_state.json",
                boundary=smoke_fixture.root,
                max_bytes=1024 * 1024,
            )
    else:
        assert smoke_fixture.workflow.catalog().eligible == ()
    assert attacked is True
    assert sentinel.read_text(encoding="utf-8") == "outside-byte-identical"


def test_catalog_rejects_linked_ancestor_without_touching_outside_sentinel(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside-byte-identical", encoding="utf-8")
    try:
        os.symlink(outside, project / "runs", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    workflow = ExploratorySmokeWorkflow(project, job_loader=lambda _job_id: {})

    catalog = workflow.catalog()

    assert catalog.eligible == ()
    assert catalog.unavailable_count == 1
    assert sentinel.read_text(encoding="utf-8") == "outside-byte-identical"


def test_passive_catalog_imports_no_torch_and_creates_no_directories(tmp_path: Path) -> None:
    project = tmp_path / "passive-project"
    project.mkdir()
    script = """
import json
import sys
from pathlib import Path
from spritelab.product_features.evaluation.exploratory_smoke import ExploratorySmokeWorkflow
root = Path(sys.argv[1])
before = sorted(path.name for path in root.iterdir())
catalog = ExploratorySmokeWorkflow(root, job_loader=lambda _job_id: {}).catalog()
after = sorted(path.name for path in root.iterdir())
print(json.dumps({"before": before, "after": after, "torch": "torch" in sys.modules, "catalog": catalog.to_dict()}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(project)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["torch"] is False
    assert payload["before"] == payload["after"] == []
    assert payload["catalog"]["eligible"] == []


def test_incomplete_device_output_is_not_resumable(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    begin_device_run(smoke_fixture.root, plan, "cpu")
    state = json.loads(
        (run_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "cpu/smoke_run_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "RUNNING"
    assert state["execution_mode"] == ("windows-direct-trainer-v1" if os.name == "nt" else "linux-worker-trainer-v1")
    assert state["resumable"] is False
    assert state["retry_policy"] == "NEW_BUNDLE_REQUIRED"
    with pytest.raises(SmokeBundleError, match="incomplete"):
        smoke_fixture.workflow.register(
            SmokeRegistrationRequest(
                **smoke_fixture.publication_fields(),
                smoke_id=prepared["smoke_id"],
                plan_identity=prepared["plan_identity"],
                cpu_receipt_identity="0" * 64,
                cuda_receipt_identity="1" * 64,
                explicit_action=True,
            ),
            server_execution_identities=_execution_identities(prepared["smoke_id"]),
        )


@pytest.mark.parametrize("target", ["run-container", "device"])
def test_run_loader_does_not_repair_missing_completion_marker(
    smoke_fixture: SmokeFixture,
    target: str,
) -> None:
    prepared = smoke_fixture.prepare(f"nonce-run-completion-{target}")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    run_root = run_bundle_directory(smoke_fixture.root, prepared["smoke_id"])
    if target == "device":
        begin_device_run(smoke_fixture.root, plan, "cpu")
        publication = run_root / "cpu"
    else:
        publication = run_root
    marker = publication / smoke_bundle_module._PUBLICATION_COMPLETION_FILENAME
    marker.rename(publication / ".held-completion-marker.json")

    with pytest.raises(SmokeBundleError, match="completion marker"):
        begin_device_run(smoke_fixture.root, plan, "cpu")

    assert not marker.exists()


def test_environment_mismatch_is_rejected_before_receipt(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    receipts = _complete_outputs(smoke_fixture, prepared)
    assert receipts["cuda"]["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    with pytest.raises(SmokeBundleError, match="environment"):
        write_device_receipt(
            smoke_fixture.root,
            plan,
            "cuda",
            config_sha256_before=plan["configurations"]["cuda"]["config_sha256"],
            config_sha256_after=plan["configurations"]["cuda"]["config_sha256"],
            environment={"CUDA_VISIBLE_DEVICES": "1", "SPRITELAB_PROGRESS": "0"},
        )


def test_plan_and_registration_artifacts_are_immutable_single_publications(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    plan_path = artifact_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "plan.json"
    plan_before = plan_path.read_bytes()
    smoke_fixture.prepare()
    assert plan_path.read_bytes() == plan_before
    assert not list(plan_path.parent.glob("*.partial-*"))


@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_plan_loader_requires_untampered_completion_marker(
    smoke_fixture: SmokeFixture,
    change: str,
) -> None:
    prepared = smoke_fixture.prepare(f"nonce-plan-completion-{change}")
    publication = artifact_bundle_directory(smoke_fixture.root, prepared["smoke_id"])
    marker = publication / smoke_bundle_module._PUBLICATION_COMPLETION_FILENAME
    outside_sentinel = smoke_fixture.root.parent / f"outside-plan-{change}.bin"
    outside_sentinel.write_bytes(b"outside-byte-identical")
    if change == "missing":
        marker.rename(publication / ".held-completion-marker.json")
    else:
        marker.write_bytes(marker.read_bytes() + b" ")

    with pytest.raises(SmokeBundleError, match=r"completion marker|publication"):
        load_plan(smoke_fixture.root, prepared["smoke_id"])

    assert outside_sentinel.read_bytes() == b"outside-byte-identical"


def test_registration_loader_rejects_incomplete_final_without_marker(
    smoke_fixture: SmokeFixture,
) -> None:
    prepared = smoke_fixture.prepare("nonce-registration-completion")
    receipts = _complete_outputs(smoke_fixture, prepared)
    registered = _register(smoke_fixture, smoke_fixture.registration_request(prepared, receipts))
    content_id = str(registered["registration_id"])
    publication = smoke_fixture.root / "runs/v3/playground/exploratory-checkpoints" / content_id
    marker = publication / smoke_bundle_module._PUBLICATION_COMPLETION_FILENAME
    marker.rename(publication / ".held-completion-marker.json")

    with pytest.raises(SmokeBundleError, match="completion marker"):
        load_playground_registration(smoke_fixture.root, content_id)

    catalog = smoke_fixture.workflow.catalog()
    assert catalog.eligible == ()
    assert catalog.unavailable_count == 1


class _ControlledProcess:
    _next_pid = 43000

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = os.getpid()
        self._done = threading.Event()
        self._code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self._code if self._done.is_set() else None

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout=10 if timeout is None else timeout)
        return int(self._code if self._code is not None else 99)

    def terminate(self) -> None:
        self.terminated = True
        self.finish(143)

    def kill(self) -> None:
        self.terminated = True
        self.finish(137)

    def finish(self, code: int) -> None:
        self._code = code
        self._done.set()


def _controlled_runner(
    root: Path,
) -> tuple[ExploratorySmokeRunner, list[tuple[list[str], dict[str, Any], _ControlledProcess]]]:
    calls: list[tuple[list[str], dict[str, Any], _ControlledProcess]] = []

    def factory(argv: list[str], **options: Any) -> _ControlledProcess:
        process = _ControlledProcess()
        process.bootstrap_identity_sha256 = smoke_runner_module.WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256
        process.private_desktop_identity_sha256 = "a" * 64
        process.restricted_token = True
        process.restricted_sid_hashes_identity_sha256 = "b" * 64
        calls.append((argv, options, process))
        return process

    def activate(
        process: _ControlledProcess,
        *,
        verifier: Any,
    ) -> int:
        verifier(process)
        return 0

    return (
        ExploratorySmokeRunner(
            root,
            process_factory=factory,
            windows_process_factory=factory,
            windows_suspended_activator=activate,
            containment_supported=lambda: True,
        ),
        calls,
    )


def _wait_for_status(runner: ExploratorySmokeRunner, smoke_id: str, device: str, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = runner.status(smoke_id, device)
        if value["status"] == expected:
            return value
        time.sleep(0.02)
    raise AssertionError(f"{device} did not reach {expected}")


def test_server_runner_uses_fixed_argv_shell_false_and_exact_environment(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare()
    runner, calls = _controlled_runner(smoke_fixture.root)

    with pytest.raises(SmokeExecutionError, match="CPU"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cuda", explicit_action=True)
    assert calls == []

    state = runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)

    assert state["status"] == "RUNNING"
    assert len(calls) == 1
    argv, options, process = calls[0]
    assert argv[:5] == [sys.executable, "-I", "-B", "-S", "-c"]
    if os.name == "nt":
        assert "spritelab.training.smoke_worker" not in argv[5]
    else:
        assert "spritelab.training.smoke_worker" in argv[5]
    assert argv[-2:] == [
        "--smoke-launch-identity" if os.name == "nt" else "--launch-identity",
        smoke_launch_identity(load_plan(smoke_fixture.root, prepared["smoke_id"]), "cpu"),
    ]
    assert isinstance(argv, list)
    if os.name == "nt":
        import spritelab.utils.write_confinement as confinement_module

        execution = artifact_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "execution" / "cpu"
        output = run_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "cpu"
        assert options["cwd"] == execution
        assert options["stdin_payload"] == b""
        assert options["stdio_root"] == execution / "temp"
        assert next(iter(options["writable_roots"])) == execution
        assert state["confinement"]["bootstrap_identity_sha256"] == (
            confinement_module.WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256
        )
        for root in (execution, output):
            for candidate in (root, *root.rglob("*")):
                assert confinement_module._windows_path_integrity_label(candidate) == (0, True)
    else:
        assert options["shell"] is False
        assert options["cwd"] == smoke_fixture.root
        assert options["stdin"] is subprocess.DEVNULL
        assert options["stdout"] is subprocess.DEVNULL
        assert options["stderr"] is subprocess.DEVNULL
    if sys.platform.startswith("linux"):
        assert callable(options["preexec_fn"])
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert options["env"]["SPRITELAB_PROGRESS"] == "0"
    assert options["env"]["PYTHONNOUSERSITE"] == "1"
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)["status"]
        == "RUNNING"
    )
    assert len(calls) == 1
    process.finish(7)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")
    with pytest.raises(SmokeExecutionError, match="fresh"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Linux has inherited-FD Landlock confinement.")
def test_server_runner_fails_closed_without_an_exact_platform_write_boundary(
    smoke_fixture: SmokeFixture,
) -> None:
    prepared = smoke_fixture.prepare("nonce-platform-fail-closed")
    runner = ExploratorySmokeRunner(smoke_fixture.root, containment_supported=lambda: False)

    with pytest.raises(SmokeExecutionError, match="containment is unavailable"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)


def test_server_runner_cancellation_is_durable_and_terminates_the_live_process(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = smoke_fixture.prepare("nonce-runner-cancel")
    runner, calls = _controlled_runner(smoke_fixture.root)
    terminated: list[_ControlledProcess] = []

    def terminate(process: _ControlledProcess) -> None:
        terminated.append(process)
        process.terminate()

    monkeypatch.setattr(smoke_runner_module, "_terminate_worker_process", terminate)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    process = calls[0][2]

    with pytest.raises(SmokeExecutionError, match="explicit action"):
        runner.cancel(
            prepared["smoke_id"],
            prepared["plan_identity"],
            "cpu",
            explicit_action=False,
        )
    cancelled = runner.cancel(
        prepared["smoke_id"],
        prepared["plan_identity"],
        "cpu",
        explicit_action=True,
    )

    assert cancelled["status"] == "CANCELLED"
    assert cancelled["exit_code"] == 130
    assert process.terminated is True
    assert process in terminated
    assert ExploratorySmokeRunner(smoke_fixture.root).status(prepared["smoke_id"], "cpu")["status"] == "CANCELLED"


def test_server_runner_enforces_the_durable_wall_clock_deadline(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = smoke_fixture.prepare("nonce-runner-deadline")
    runner, calls = _controlled_runner(smoke_fixture.root)
    terminated: list[_ControlledProcess] = []

    def terminate(process: _ControlledProcess) -> None:
        terminated.append(process)
        process.terminate()

    monkeypatch.setattr(smoke_runner_module, "_terminate_worker_process", terminate)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    state_path = artifact_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "execution/cpu/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    limit = int(state["wall_clock_limit_seconds"])
    started = datetime.now(timezone.utc) - timedelta(seconds=limit + 5)
    deadline = started + timedelta(seconds=limit)
    state["started_at"] = started.isoformat()
    state["deadline_at"] = deadline.isoformat()
    state["updated_at"] = (deadline - timedelta(seconds=1)).isoformat()
    state.pop("state_identity")
    state = smoke_runner_module._finalize_state_identity(state)
    runner._validate_state(state)
    payload = smoke_runner_module.canonical_json_bytes(state, pretty=True)
    with state_path.open("r+b") as handle:
        handle.write(payload)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())

    timed_out = runner.status(prepared["smoke_id"], "cpu")

    assert timed_out["status"] == "TIMED_OUT"
    assert timed_out["exit_code"] == 124
    assert calls[0][2].terminated is True
    assert calls[0][2] in terminated
    assert ExploratorySmokeRunner(smoke_fixture.root).status(prepared["smoke_id"], "cpu")["status"] == "TIMED_OUT"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux inherited-FD execution contract")
def test_smoke_worker_repasses_retained_writable_root_fds_to_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.training.smoke_worker as worker_module

    roots = [tmp_path / "execution", tmp_path / "output"]
    for root in roots:
        root.mkdir()
    flags = (
        int(getattr(os, "O_PATH", os.O_RDONLY)) | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptors = tuple(os.open(root, flags) for root in roots)
    monkeypatch.setenv("SPRITELAB_WRITABLE_ROOT_FDS", ",".join(str(value) for value in descriptors))
    parsed = SimpleNamespace(
        smoke_id="smoke-" + "1" * 20,
        device="cpu",
        plan_identity="plan-identity",
        launch_identity="launch-identity",
    )
    plan = {"smoke_id": parsed.smoke_id, "plan_identity": parsed.plan_identity}
    process_options: dict[str, Any] = {}
    validation_environments: list[dict[str, str]] = []

    class Process:
        pid = os.getpid()

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait() -> int:
            return 0

    class Containment:
        name = "LINUX_PDEATHSIG"

        def __init__(self, process: Process) -> None:
            self.process = process

        def activate(self, *, verifier: Any) -> None:
            verifier(self.process)

        def terminate(self) -> None:
            raise AssertionError("completed trainer was unexpectedly terminated")

        def close(self) -> None:
            return None

    @contextmanager
    def pinned(_plan: dict[str, Any], **_kwargs: Any):
        yield SimpleNamespace(launch_path=sys.executable, pass_fds=())

    def process_factory(_argv: list[str], **options: Any) -> Process:
        process_options.update(options)
        return Process()

    def validate_environment(
        _root: Path,
        _plan: dict[str, Any],
        _device: str,
        environment: dict[str, str],
    ) -> None:
        validation_environments.append(dict(environment))

    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    monkeypatch.setattr(worker_module, "_parse_args", lambda: parsed)
    monkeypatch.setattr(worker_module, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(worker_module, "smoke_launch_identity", lambda *_args, **_kwargs: parsed.launch_identity)
    monkeypatch.setattr(worker_module, "validate_smoke_environment", validate_environment)
    monkeypatch.setattr(worker_module, "validate_smoke_interpreter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "smoke_containment_supported", lambda: True)
    monkeypatch.setattr(worker_module, "verify_execution_guards", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "_process_identity", lambda pid: {"pid": pid, "birth_token": "1"})
    monkeypatch.setattr(worker_module, "_publish_heartbeat", lambda *_args, **_kwargs: {"heartbeat_identity": "a" * 64})
    monkeypatch.setattr(
        worker_module,
        "_read_execution_state",
        lambda *_args, **_kwargs: {"status": "RUNNING", "deadline_at": future},
    )
    monkeypatch.setattr(worker_module, "smoke_training_argv", lambda *_args, **_kwargs: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(worker_module, "pinned_smoke_interpreter", pinned)
    monkeypatch.setattr(worker_module.subprocess, "Popen", process_factory)
    monkeypatch.setattr(worker_module, "_Containment", Containment)
    monkeypatch.setattr(worker_module, "verify_pinned_process_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "load_device_receipt", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(worker_module, "_finish", lambda *_args, **_kwargs: 0)

    assert worker_module.main() == 0
    assert len(validation_environments) == 1
    assert "SPRITELAB_WRITABLE_ROOT_FDS" not in validation_environments[0]
    assert set(descriptors) <= set(process_options["pass_fds"])
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    ("terminal_status", "exit_code"),
    (("CANCELLED", 130), ("TIMED_OUT", 124)),
)
def test_smoke_worker_guard_scan_publishes_the_exact_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    exit_code: int,
) -> None:
    import spritelab.training.smoke_worker as worker_module

    parsed = SimpleNamespace(
        smoke_id="smoke-" + "2" * 20,
        device="cpu",
        plan_identity="plan-identity",
        launch_identity="launch-identity",
    )
    plan = {"smoke_id": parsed.smoke_id, "plan_identity": parsed.plan_identity}
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    execution_state = {"status": "RUNNING", "deadline_at": future}
    finish_calls: list[dict[str, Any]] = []

    def verify_guards(
        _root: Path,
        _plan: dict[str, Any],
        *,
        operation_check: Any,
    ) -> None:
        execution_state["status"] = terminal_status
        operation_check()

    def finish(
        _root: Path,
        _plan: dict[str, Any],
        _device: str,
        _launch_identity: str,
        _heartbeat: dict[str, Any],
        actual_exit_code: int,
        *,
        status_override: str | None = None,
        **_kwargs: Any,
    ) -> int:
        finish_calls.append({"status": status_override, "exit_code": actual_exit_code})
        return actual_exit_code

    monkeypatch.delenv("SPRITELAB_WRITABLE_ROOT_FDS", raising=False)
    monkeypatch.setattr(worker_module, "_parse_args", lambda: parsed)
    monkeypatch.setattr(worker_module, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(worker_module, "smoke_launch_identity", lambda *_args, **_kwargs: parsed.launch_identity)
    monkeypatch.setattr(worker_module, "_parse_writable_root_fds", lambda _value: ())
    monkeypatch.setattr(worker_module, "validate_smoke_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "validate_smoke_interpreter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "smoke_containment_supported", lambda: True)
    monkeypatch.setattr(worker_module, "_process_identity", lambda pid: {"pid": pid, "birth_token": "1"})
    monkeypatch.setattr(
        worker_module,
        "_publish_heartbeat",
        lambda *_args, **_kwargs: {"heartbeat_identity": "a" * 64},
    )
    monkeypatch.setattr(worker_module, "_read_execution_state", lambda *_args, **_kwargs: dict(execution_state))
    monkeypatch.setattr(worker_module, "verify_execution_guards", verify_guards)
    monkeypatch.setattr(worker_module, "_finish", finish)
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("the trainer started after a terminal worker state"),
    )

    assert worker_module.main() == exit_code
    assert finish_calls == [{"status": terminal_status, "exit_code": exit_code}]


def test_smoke_worker_finish_rejects_a_terminal_exit_code_mismatch(tmp_path: Path) -> None:
    import spritelab.training.smoke_worker as worker_module

    with pytest.raises(SmokeBundleError, match="exit code"):
        worker_module._finish(
            tmp_path,
            {"smoke_id": "smoke-" + "3" * 20},
            "cpu",
            "launch-identity",
            {"heartbeat_identity": "a" * 64},
            124,
            status_override="CANCELLED",
        )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (("COMPLETE", 70), ("FAILED", 0), ("CANCELLED", 124), ("TIMED_OUT", 130)),
)
def test_smoke_worker_outcome_loader_rejects_status_exit_code_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
) -> None:
    import spritelab.training.smoke_worker as worker_module

    plan = {"smoke_id": "smoke-" + "4" * 20, "plan_identity": "plan-identity"}
    outcome = worker_module.finalize_identity(
        {
            "schema_version": worker_module.SMOKE_WORKER_OUTCOME_SCHEMA,
            "smoke_id": plan["smoke_id"],
            "device": "cpu",
            "plan_identity": plan["plan_identity"],
            "launch_identity": "launch-identity",
            "status": status,
            "exit_code": exit_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat_identity": "a" * 64,
            "message": "Terminal outcome fixture.",
        },
        "outcome_identity",
    )
    monkeypatch.setattr(worker_module, "_read_json", lambda *_args, **_kwargs: outcome)

    with pytest.raises(SmokeBundleError, match="outcome is invalid"):
        worker_module.load_worker_outcome(tmp_path, plan, "cpu", "launch-identity")


def test_smoke_orchestration_source_change_stales_plan_before_launch(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-orchestration-stale")
    source = smoke_fixture.root / "src/spritelab/training/smoke_worker.py"
    source.write_text("# changed after preparation\n", encoding="utf-8")
    runner, calls = _controlled_runner(smoke_fixture.root)
    with pytest.raises(SmokeBundleError, match="orchestration code changed"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert calls == []


def test_non_orchestration_training_source_change_stales_full_code_identity(
    smoke_fixture: SmokeFixture,
) -> None:
    prepared = smoke_fixture.prepare("nonce-full-code-stale")
    source = smoke_fixture.root / "src/spritelab/training/generator_challenger.py"
    source.write_text("# non-orchestration training source changed\n", encoding="utf-8")
    runner, calls = _controlled_runner(smoke_fixture.root)
    with pytest.raises(SmokeBundleError, match="production Python changed"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert calls == []


def test_untracked_production_python_stales_full_code_inventory(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-untracked-code-stale")
    source = smoke_fixture.root / "src/spritelab/training/untracked_injected.py"
    source.write_text("# must never escape the campaign identity\n", encoding="utf-8")
    runner, calls = _controlled_runner(smoke_fixture.root)
    with pytest.raises(SmokeBundleError, match="production Python changed"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert calls == []


def test_server_runner_drops_hostile_inherited_environment(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "CUDA_HOME",
    ):
        monkeypatch.setenv(name, f"hostile-{name}")
    prepared = smoke_fixture.prepare("nonce-hostile-environment")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    environment = calls[0][1]["env"]
    expected_environment = build_smoke_child_environment(smoke_fixture.root, plan, "cpu")
    if os.name == "nt":
        expected_environment["SPRITELAB_CONFINEMENT_PROJECT_ROOT"] = os.fspath(smoke_fixture.root)
    assert environment == expected_environment
    assert not {
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "CUDA_HOME",
    } & set(environment)
    for name in plan["configurations"]["cpu"]["child_environment"]["sandboxed_path_variables"]:
        assert Path(environment[name]).is_relative_to(smoke_fixture.root)
    assert smoke_fixture.root.as_posix() not in json.dumps(plan)
    calls[0][2].finish(8)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")


def test_cuda_waits_for_cpu_wrapper_exit_after_receipt(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-delayed-wrapper")
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    device_root = run_bundle_directory(smoke_fixture.root, prepared["smoke_id"]) / "cpu"
    assert (device_root / smoke_bundle_module._PUBLICATION_COMPLETION_FILENAME).is_file()
    _complete_device(smoke_fixture, plan, "cpu")
    assert runner.status(prepared["smoke_id"], "cpu")["status"] == "RUNNING"
    with pytest.raises(SmokeExecutionError, match="CPU"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cuda", explicit_action=True)
    assert len(calls) == 1
    calls[0][2].finish(0)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "COMPLETE")


def test_project_wide_scan_blocks_a_fresh_bundle_while_worker_is_active(smoke_fixture: SmokeFixture) -> None:
    first = smoke_fixture.prepare("nonce-project-worker-one")
    second = smoke_fixture.prepare("nonce-project-worker-two")
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(first["smoke_id"], first["plan_identity"], "cpu", explicit_action=True)
    with pytest.raises(SmokeExecutionError, match="Another contained"):
        runner.launch(second["smoke_id"], second["plan_identity"], "cpu", explicit_action=True)
    calls[0][2].finish(7)
    _wait_for_status(runner, first["smoke_id"], "cpu", "FAILED")
    assert (
        runner.launch(second["smoke_id"], second["plan_identity"], "cpu", explicit_action=True)["status"] == "RUNNING"
    )
    calls[1][2].finish(7)
    _wait_for_status(runner, second["smoke_id"], "cpu", "FAILED")


def test_server_runner_reconstructs_receipt_completion_and_registers_without_pasted_hashes(
    smoke_fixture: SmokeFixture,
) -> None:
    prepared = smoke_fixture.prepare()
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    _complete_device(smoke_fixture, plan, "cpu")
    calls[0][2].finish(0)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "COMPLETE")
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cuda", explicit_action=True)
    if os.name != "nt":
        assert calls[1][1]["shell"] is False
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[1][1]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls[1][1]["env"]["SPRITELAB_PROGRESS"] == "0"
    _complete_device(smoke_fixture, plan, "cuda")
    calls[1][2].finish(0)
    _wait_for_status(runner, prepared["smoke_id"], "cuda", "COMPLETE")

    restarted = ExploratorySmokeRunner(smoke_fixture.root)
    assert restarted.bundle_status(prepared["smoke_id"])["registration_ready"] is True
    result = smoke_fixture.workflow.register_job(
        smoke_fixture.request.conditioned_job_id,
        prepared["smoke_id"],
        prepared["plan_identity"],
        explicit_action=True,
        server_execution_identities={
            device: value["execution_identity"]
            for device, value in restarted.require_complete(prepared["smoke_id"]).items()
        },
    )
    assert result["playground_eligible"] is True
    assert "receipt" not in json.dumps(result).casefold()


def test_runner_refuses_manual_output_adoption_and_reconstructs_interruption(smoke_fixture: SmokeFixture) -> None:
    manual = smoke_fixture.prepare("nonce-manual-output")
    _complete_outputs(smoke_fixture, manual)
    runner, _calls = _controlled_runner(smoke_fixture.root)
    assert runner.status(manual["smoke_id"], "cpu")["status"] == "NOT_STARTED"
    with pytest.raises(SmokeExecutionError, match="server-owned"):
        runner.launch(manual["smoke_id"], manual["plan_identity"], "cpu", explicit_action=True)

    interrupted = smoke_fixture.prepare("nonce-interrupted-output")
    active, _active_calls = _controlled_runner(smoke_fixture.root)
    active.launch(interrupted["smoke_id"], interrupted["plan_identity"], "cpu", explicit_action=True)
    restarted = ExploratorySmokeRunner(smoke_fixture.root)
    assert restarted.status(interrupted["smoke_id"], "cpu")["status"] == "RUNNING"
    _active_calls[0][2].finish(9)
    _wait_for_status(restarted, interrupted["smoke_id"], "cpu", "FAILED")
    with pytest.raises(SmokeExecutionError, match="fresh"):
        restarted.launch(interrupted["smoke_id"], interrupted["plan_identity"], "cpu", explicit_action=True)


@pytest.mark.skipif(os.name == "nt", reason="Windows launches the trainer directly without a nested worker heartbeat.")
def test_restart_recognizes_exact_durable_worker_heartbeat(smoke_fixture: SmokeFixture) -> None:
    import spritelab.training.smoke_runner as runner_module
    import spritelab.training.smoke_worker as worker_module

    prepared = smoke_fixture.prepare("nonce-durable-heartbeat")
    active, calls = _controlled_runner(smoke_fixture.root)
    active.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    launch_identity = smoke_launch_identity(plan, "cpu")
    process_identity = worker_module._process_identity(os.getpid())
    assert process_identity is not None
    worker_module._publish_heartbeat(
        smoke_fixture.root,
        plan,
        "cpu",
        launch_identity,
        worker_module._now(),
        1,
        process_identity,
        status="RUNNING",
        containment=worker_module._containment_name(),
    )
    runner_module._forget_process(smoke_fixture.root, prepared["smoke_id"], "cpu", launch_identity)
    restarted = ExploratorySmokeRunner(smoke_fixture.root)
    assert restarted.status(prepared["smoke_id"], "cpu")["status"] == "RUNNING"
    calls[0][2].finish(7)
    _wait_for_status(restarted, prepared["smoke_id"], "cpu", "FAILED")


def test_web_actions_derive_identities_and_require_csrf_without_manual_receipts(
    smoke_fixture: SmokeFixture,
) -> None:
    from fastapi.testclient import TestClient

    from spritelab.product_core import ProjectContext
    from spritelab.product_features.evaluation.web import create_evaluation_router
    from spritelab.product_web.app import create_app as create_product_app

    runner, calls = _controlled_runner(smoke_fixture.root)
    context = ProjectContext(
        smoke_fixture.root,
        config=copy.deepcopy(DEFAULT_CONFIG),
        runs_directory=smoke_fixture.root / "runs",
    )
    app = create_product_app(context)
    app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/evaluation"]
    app.state.spritelab_render_plugin_template = None
    app.include_router(
        create_evaluation_router(
            context,
            smoke_workflow=smoke_fixture.workflow,
            smoke_runner=runner,
        )
    )
    client = TestClient(app)
    page = client.get("/evaluation")
    assert page.status_code == 200
    assert "Eligible conditioned publication" in page.text
    assert "Run CPU smoke" in page.text
    assert "Run CUDA smoke" in page.text
    assert "receipt identity" not in page.text.casefold()
    assert "Candidate identity" not in page.text
    headers = {"X-CSRF-Token": app.state.spritelab_csrf_token}
    payload = {
        "conditioned_job_id": smoke_fixture.request.conditioned_job_id,
        "preparation_nonce": "nonce-web-operated",
        "explicit_action": True,
    }
    assert client.post("/evaluation/api/playground/smokes/prepare", json=payload).status_code == 403
    prepared_response = client.post(
        "/evaluation/api/playground/smokes/prepare",
        json=payload,
        headers=headers,
    )
    assert prepared_response.status_code == 200
    prepared = prepared_response.json()
    run_payload = {
        "conditioned_job_id": smoke_fixture.request.conditioned_job_id,
        "smoke_id": prepared["smoke_id"],
        "plan_identity": prepared["plan_identity"],
        "explicit_action": True,
    }
    run_response = client.post(
        "/evaluation/api/playground/smokes/run-cpu",
        json=run_payload,
        headers=headers,
    )
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "RUNNING"
    assert len(calls) == 1
    with pytest.raises(SmokeExecutionError, match="Both server-run CPU and CUDA"):
        runner.require_complete(prepared["smoke_id"])
    register_response = client.post(
        "/evaluation/api/playground/smokes/register",
        json=run_payload,
        headers=headers,
    )
    assert register_response.status_code == 409
    assert register_response.json()["error_code"] == "smoke_devices_incomplete"
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    _complete_device(smoke_fixture, plan, "cpu")
    calls[0][2].finish(0)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "COMPLETE")
    cuda_response = client.post(
        "/evaluation/api/playground/smokes/run-cuda",
        json=run_payload,
        headers=headers,
    )
    assert cuda_response.status_code == 200
    assert cuda_response.json()["status"] == "RUNNING"
    _complete_device(smoke_fixture, plan, "cuda")
    calls[1][2].finish(0)
    _wait_for_status(runner, prepared["smoke_id"], "cuda", "COMPLETE")
    completed_registration = client.post(
        "/evaluation/api/playground/smokes/register",
        json=run_payload,
        headers=headers,
    )
    assert completed_registration.status_code == 200
    assert completed_registration.json()["playground_eligible"] is True
    rejected = client.post(
        "/evaluation/api/playground/smokes/prepare",
        json={**payload, "candidate_identity_sha256": "0" * 64},
        headers=headers,
    )
    assert rejected.status_code == 422
    assert smoke_fixture.root.as_posix() not in rejected.text


def test_passive_evaluation_page_uses_no_torch_subprocess_or_directory_creation(tmp_path: Path) -> None:
    project = tmp_path / "passive-page-project"
    project.mkdir()
    script = """
import copy
import json
import subprocess
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from spritelab.product_core import ProjectContext
from spritelab.product_features.evaluation.web import create_evaluation_router
from spritelab.product_web.app import create_app
from spritelab.v3.config import DEFAULT_CONFIG
root = Path(sys.argv[1])
before = sorted(path.name for path in root.iterdir())
process_calls = []
def forbidden_process(*args, **kwargs):
    process_calls.append([args, kwargs])
    raise AssertionError("passive page launched a subprocess")
subprocess.Popen = forbidden_process
context = ProjectContext(root, config=copy.deepcopy(DEFAULT_CONFIG), runs_directory=root / "runs")
app = create_app(context)
app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/evaluation"]
app.state.spritelab_render_plugin_template = None
app.include_router(create_evaluation_router(context))
response = TestClient(app).get("/evaluation")
after = sorted(path.name for path in root.iterdir())
print(json.dumps({"status": response.status_code, "before": before, "after": after, "torch": "torch" in sys.modules, "subprocesses": len(process_calls)}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(project)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {"status": 200, "before": [], "after": [], "torch": False, "subprocesses": 0}


def test_cli_rejects_invalid_server_environment_before_torch_import(smoke_fixture: SmokeFixture) -> None:
    prepared = smoke_fixture.prepare("nonce-cli-preflight")
    script = """
import argparse
import json
import os
import sys
from pathlib import Path
from spritelab.training.cli.experiment_cmds import _run
from spritelab.training.smoke_bundle import load_plan, smoke_launch_identity
os.chdir(sys.argv[1])
plan = load_plan(Path.cwd(), sys.argv[2])
parsed = argparse.Namespace(
    smoke_bundle_id=sys.argv[2], smoke_device="cpu", smoke=True,
    smoke_plan_identity=plan["plan_identity"],
    smoke_launch_identity=smoke_launch_identity(plan, "cpu"),
    resume=None, unsafe_resume=False, unsafe_resume_reason=None,
    config=Path(sys.argv[3]),
)
try:
    _run(parsed)
except Exception as exc:
    print(json.dumps({"error": str(exc), "torch": "torch" in sys.modules}))
else:
    raise AssertionError("invalid environment was accepted")
"""
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("SPRITELAB_PROGRESS", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    config_path = f"artifacts/training/smokes/{prepared['smoke_id']}/configs/cpu.json"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(smoke_fixture.root), prepared["smoke_id"], config_path],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["torch"] is False
    assert "environment" in payload["error"].casefold()


def test_isolated_bootstrap_imports_real_package_and_ignores_hostile_sitecustomize(
    smoke_fixture: SmokeFixture,
) -> None:
    prepared = smoke_fixture.prepare("nonce-isolated-bootstrap")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    argv = smoke_training_argv(plan, "cpu")
    assert argv[1:5] == ["-I", "-B", "-S", "-c"]
    marker = smoke_fixture.root / "sitecustomize-imported.txt"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe', encoding='utf-8')\n"
    (smoke_fixture.root / "sitecustomize.py").write_text(payload, encoding="utf-8")
    (smoke_fixture.root / "src" / "sitecustomize.py").write_text(payload, encoding="utf-8")
    with _contained_child_environment(smoke_fixture, plan, "cpu") as (environment, writable_fds):
        assert "PYTHONPATH" not in environment
        isolated_command, isolated_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", argv)
        isolated = subprocess.run(
            isolated_command,
            cwd=isolated_cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            pass_fds=writable_fds,
        )
        assert isolated.returncode == 0, isolated.stderr
        assert isolated.stdout.strip() == "ISOLATED_BOOTSTRAP_OK"
        assert not marker.exists()
        worker_argv = smoke_worker_argv(plan, "cpu")
        worker_command, worker_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", worker_argv)
        worker = subprocess.run(
            worker_command,
            cwd=worker_cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            pass_fds=writable_fds,
        )
        assert worker.returncode == 0, worker.stderr

    (smoke_fixture.root / "src/spritelab/training/generator_challenger.py").write_text(
        "print('MUTATED_BEFORE_IMPORT')\n",
        encoding="utf-8",
    )
    rejected_command, rejected_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", argv)
    rejected = subprocess.run(
        rejected_command,
        cwd=rejected_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 70
    assert rejected.stdout == ""
    calls[0][2].finish(1)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")


def test_bound_runtime_refuses_sourceless_pyc_execution(smoke_fixture: SmokeFixture) -> None:
    (smoke_fixture.root / "src/spritelab/__main__.py").write_text(
        "import sourceless_only\nprint('UNSAFE_SOURCELESS_IMPORT_COMPLETED')\n",
        encoding="utf-8",
    )
    code_identity = smoke_fixture.activation.campaign["code_identity"]
    code_identity.pop("sha256", None)
    for record in code_identity["files"]:
        record["sha256"] = _sha((smoke_fixture.root / record["path"]).read_bytes())
    code_identity["sha256"] = stable_hash(code_identity)
    prepared = smoke_fixture.prepare("nonce-sourceless-pyc")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    argv = smoke_training_argv(plan, "cpu")
    rejected_command, rejected_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", argv)

    rejected = subprocess.run(
        rejected_command,
        cwd=rejected_cwd,
        env=build_smoke_child_environment(smoke_fixture.root, plan, "cpu"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert rejected.returncode == 70
    assert "SOURCELESS_PYC_EXECUTED" not in rejected.stdout
    assert "UNSAFE_SOURCELESS_IMPORT_COMPLETED" not in rejected.stdout
    calls[0][2].finish(1)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")


def test_bound_runtime_denies_preexisting_unowned_package_code(smoke_fixture: SmokeFixture) -> None:
    (smoke_fixture.root / "runtime/site-packages/torch/unowned.py").write_text(
        "print('UNOWNED_PACKAGE_CODE_EXECUTED')\n",
        encoding="utf-8",
    )
    (smoke_fixture.root / "src/spritelab/__main__.py").write_text(
        "import torch.unowned\nprint('UNSAFE_UNOWNED_IMPORT_COMPLETED')\n",
        encoding="utf-8",
    )
    code_identity = smoke_fixture.activation.campaign["code_identity"]
    code_identity.pop("sha256", None)
    for record in code_identity["files"]:
        record["sha256"] = _sha((smoke_fixture.root / record["path"]).read_bytes())
    code_identity["sha256"] = stable_hash(code_identity)
    prepared = smoke_fixture.prepare("nonce-unowned-runtime-code")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    argv = smoke_training_argv(plan, "cpu")
    rejected_command, rejected_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", argv)

    rejected = subprocess.run(
        rejected_command,
        cwd=rejected_cwd,
        env=build_smoke_child_environment(smoke_fixture.root, plan, "cpu"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert rejected.returncode == 70
    assert "UNOWNED_PACKAGE_CODE_EXECUTED" not in rejected.stdout
    assert "UNSAFE_UNOWNED_IMPORT_COMPLETED" not in rejected.stdout
    calls[0][2].finish(1)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")


def test_child_loader_rejects_source_changed_between_preflight_and_import(
    smoke_fixture: SmokeFixture,
) -> None:
    target = smoke_fixture.root / "src/spritelab/training/generator_challenger.py"
    (smoke_fixture.root / "src/spritelab/__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(target)!r}).write_text(\"print('MUTATED_DURING_IMPORT')\\n\", encoding='utf-8')\n",
        encoding="utf-8",
    )
    (smoke_fixture.root / "src/spritelab/__main__.py").write_text(
        "import spritelab.training.generator_challenger\nprint('UNSAFE_IMPORT_COMPLETED')\n",
        encoding="utf-8",
    )
    code_identity = smoke_fixture.activation.campaign["code_identity"]
    code_identity.pop("sha256", None)
    for record in code_identity["files"]:
        record["sha256"] = _sha((smoke_fixture.root / record["path"]).read_bytes())
    code_identity["sha256"] = stable_hash(code_identity)
    prepared = smoke_fixture.prepare("nonce-loader-race")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    runner, calls = _controlled_runner(smoke_fixture.root)
    runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    argv = smoke_training_argv(plan, "cpu")
    rejected_command, rejected_cwd = _direct_bound_child_command(smoke_fixture, plan, "cpu", argv)
    rejected = subprocess.run(
        rejected_command,
        cwd=rejected_cwd,
        env=build_smoke_child_environment(smoke_fixture.root, plan, "cpu"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 70
    assert "UNSAFE_IMPORT_COMPLETED" not in rejected.stdout
    calls[0][2].finish(1)
    _wait_for_status(runner, prepared["smoke_id"], "cpu", "FAILED")


def test_interpreter_swap_is_rejected_before_worker_launch(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import spritelab.training.smoke_bundle as smoke_bundle_module

    prepared = smoke_fixture.prepare("nonce-interpreter-swap")
    replacement = tmp_path / "not-the-bound-python.exe"
    replacement.write_bytes(b"not the bound interpreter")
    monkeypatch.setattr(smoke_bundle_module.sys, "executable", str(replacement))
    runner, calls = _controlled_runner(smoke_fixture.root)
    with pytest.raises(SmokeBundleError, match="interpreter changed"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert calls == []


def test_interpreter_lexical_symlink_swap_is_rejected_when_supported(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import spritelab.training.smoke_bundle as smoke_bundle_module

    original = Path(sys.executable)
    first = tmp_path / "python-bound-link.exe"
    second = tmp_path / "python-swapped-link.exe"
    try:
        os.symlink(original, first)
        os.symlink(original, second)
    except OSError as exc:
        pytest.skip(f"interpreter symlinks are unavailable: {type(exc).__name__}")
    monkeypatch.setattr(smoke_bundle_module.sys, "executable", str(first))
    prepared = smoke_fixture.prepare("nonce-interpreter-link-swap")
    monkeypatch.setattr(smoke_bundle_module.sys, "executable", str(second))
    runner, calls = _controlled_runner(smoke_fixture.root)
    with pytest.raises(SmokeBundleError, match="interpreter changed"):
        runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert calls == []


def test_pinned_interpreter_denies_target_mutation_across_launch_window(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import spritelab.training.smoke_bundle as smoke_bundle_module

    original = Path(sys.executable)
    copied = tmp_path / original.name
    copied.write_bytes(original.read_bytes())
    copied.chmod(original.stat().st_mode)
    monkeypatch.setattr(smoke_bundle_module.sys, "executable", str(copied))
    prepared = smoke_fixture.prepare("nonce-pinned-interpreter")
    plan = load_plan(smoke_fixture.root, prepared["smoke_id"])
    before = copied.read_bytes()
    with pinned_smoke_interpreter(plan, lexical_path=copied) as pin:
        if os.name == "nt":
            with pytest.raises(OSError):
                descriptor = os.open(copied, os.O_WRONLY)
                os.close(descriptor)
        else:
            process = subprocess.Popen(
                [pin.launch_path, "-I", "-B", "-c", "import time;time.sleep(2)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=pin.pass_fds,
            )
            try:
                verify_pinned_process_image(process, pin)
                with pytest.raises(OSError):
                    descriptor = os.open(copied, os.O_WRONLY)
                    os.close(descriptor)
            finally:
                process.terminate()
                process.wait(timeout=5)
    assert copied.read_bytes() == before


def test_worker_image_verification_failure_terminates_launched_process(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.training.smoke_runner as runner_module

    prepared = smoke_fixture.prepare("nonce-worker-image-mismatch")
    runner, calls = _controlled_runner(smoke_fixture.root)
    monkeypatch.setattr(
        runner_module,
        "verify_pinned_process_image",
        lambda _process, _pin: (_ for _ in ()).throw(
            SmokeBundleError("smoke_interpreter_launch", "Injected process image mismatch.")
        ),
    )
    state = runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert state["status"] == "FAILED"
    assert len(calls) == 1
    assert calls[0][2].terminated is True


def test_monitor_start_failure_terminates_launched_worker(
    smoke_fixture: SmokeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spritelab.training.smoke_runner as runner_module

    prepared = smoke_fixture.prepare("nonce-monitor-start-failure")
    runner, calls = _controlled_runner(smoke_fixture.root)

    class BrokenThread:
        def __init__(self, **_options: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("injected monitor start failure")

    monkeypatch.setattr(runner_module.threading, "Thread", BrokenThread)
    state = runner.launch(prepared["smoke_id"], prepared["plan_identity"], "cpu", explicit_action=True)
    assert state["status"] == "FAILED"
    assert len(calls) == 1
    assert calls[0][2].terminated is True


def test_suspended_windows_containment_failure_terminates_before_resume() -> None:
    from spritelab.utils.pinned_executable import activate_windows_suspended_process

    process = _ControlledProcess()
    resumed: list[bool] = []
    closed: list[int] = []

    def assignment_failure(_process: Any) -> int:
        raise OSError("injected assignment failure")

    with pytest.raises(OSError, match="assignment"):
        activate_windows_suspended_process(
            process,
            assigner=assignment_failure,
            verifier=lambda _process: None,
            resumer=lambda _process: resumed.append(True),
            closer=closed.append,
        )
    assert process.terminated is True
    assert resumed == []
    assert closed == []

    process = _ControlledProcess()
    with pytest.raises(OSError, match="injected resume"):
        activate_windows_suspended_process(
            process,
            assigner=lambda _process: 12345,
            verifier=lambda _process: None,
            resumer=lambda _process: (_ for _ in ()).throw(OSError("injected resume failure")),
            closer=closed.append,
        )
    assert process.terminated is True
    assert closed == [12345]


def test_suspended_windows_image_verification_precedes_resume_and_fails_closed() -> None:
    from spritelab.utils.pinned_executable import activate_windows_suspended_process

    events: list[str] = []
    process = _ControlledProcess()
    handle = activate_windows_suspended_process(
        process,
        assigner=lambda _process: events.append("assigned") or 77,
        verifier=lambda _process: events.append("verified"),
        resumer=lambda _process: events.append("resumed"),
        closer=lambda _handle: events.append("closed"),
    )
    assert handle == 77
    assert events == ["assigned", "verified", "resumed"]
    assert process.terminated is False

    events.clear()
    process = _ControlledProcess()

    def mismatch(_process: Any) -> None:
        events.append("verification_failed")
        raise SmokeBundleError("smoke_interpreter_launch", "injected image mismatch")

    with pytest.raises(SmokeBundleError, match="mismatch"):
        activate_windows_suspended_process(
            process,
            assigner=lambda _process: events.append("assigned") or 88,
            verifier=mismatch,
            resumer=lambda _process: events.append("resumed"),
            closer=lambda _handle: events.append("closed"),
        )
    assert events == ["assigned", "verification_failed", "closed"]
    assert process.terminated is True


def test_linux_parent_death_guard_rejects_parent_race_before_exec() -> None:
    from spritelab.utils.pinned_executable import linux_parent_death_signal

    calls: list[tuple[int, int, int, int, int]] = []
    exits: list[int] = []

    class Libc:
        def prctl(self, *arguments: int) -> int:
            calls.append(arguments)
            return 0

    guarded = linux_parent_death_signal(
        101,
        libc_factory=Libc,
        parent_pid=lambda: 202,
        exit_process=exits.append,
    )
    guarded()
    assert calls == [(1, int(getattr(signal, "SIGKILL", 9)), 0, 0, 0)]
    assert exits == [127]

    exits.clear()
    stable = linux_parent_death_signal(
        101,
        libc_factory=Libc,
        parent_pid=lambda: 101,
        exit_process=exits.append,
    )
    stable()
    assert exits == []

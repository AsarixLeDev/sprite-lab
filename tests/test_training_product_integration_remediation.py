from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from spritelab.product_core import ProductResult, ProductRun, ProductStatus, ProjectContext
from spritelab.product_features.training import activation as activation_module
from spritelab.remote_compute import (
    ArtifactReference,
    ComputeJob,
    ComputeStatus,
    FakeComputeBackend,
    HostedBackendRegistry,
    LocalComputeBackend,
    PreparedCompute,
    ResumeRequest,
    SSHComputeBackend,
    SSHSettings,
    TrainingLaunchRejected,
)
from spritelab.remote_compute.ssh import RemoteResult
from spritelab.training import campaign as campaign_module
from spritelab.training.campaign import CampaignValidationError, stable_hash
from spritelab.training.launch import receipt_with_recomputed_hash
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState, StageStatus
from spritelab.v3.orchestration import ExecutionOptions, _validated_product_training_identity, train
from training_launch_test_utils import compute_request, validated_launch


class RecordingSSHTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, script: str, payload: dict) -> RemoteResult:
        self.calls.append((script, dict(payload)))
        return RemoteResult(0, json.dumps({"status": "RUNNING", "changed": True, **payload}))

    def upload(self, local_path: Path, remote_path: str) -> RemoteResult:
        self.calls.append(("upload", {"local": str(local_path), "remote": remote_path}))
        return RemoteResult(0)

    def download(self, remote_path: str, local_path: Path) -> RemoteResult:
        self.calls.append(("download", {"remote": remote_path, "local": str(local_path)}))
        return RemoteResult(0)


def _ssh_backend(transport: RecordingSSHTransport) -> SSHComputeBackend:
    return SSHComputeBackend(
        SSHSettings("example.test", "trainer", "/workspace/sprite-lab", cloud=False),
        transport=transport,
    )


def _ready_state(root: Path) -> ProjectState:
    return ProjectState(
        "synthetic",
        root,
        root / "spritelab.yaml",
        "synthetic-source",
        [
            StageState(
                "dataset-freeze",
                "Dataset freeze",
                StageStatus.COMPLETE,
                "Frozen.",
                production_authorized=True,
            ),
            StageState(
                "training-infrastructure-audit",
                "Training audit",
                StageStatus.COMPLETE,
                "Applicable PASS.",
                audit=AuditStatus.PASS,
            ),
            StageState(
                "training-campaign",
                "Training campaign",
                StageStatus.READY,
                "Authorized.",
                production_authorized=True,
            ),
        ],
    )


def _config(root: Path) -> ProjectConfig:
    values = copy.deepcopy(DEFAULT_CONFIG)
    values["paths"]["runs"] = "runs/v3"
    values["execution"]["allow_training"] = True
    return ProjectConfig(root, root / "spritelab.yaml", values)


@pytest.mark.parametrize("inert_command", [[], ["never", "run"]])
def test_v3_train_missing_campaign_is_blocked_and_legacy_command_is_never_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inert_command: list[str]
) -> None:
    config = _config(tmp_path)
    config.values["execution"]["training_command"] = inert_command
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _ready_state(tmp_path))
    monkeypatch.setattr(
        "spritelab.v3.orchestration.backend_from_context",
        lambda *_: pytest.fail("backend selection must not occur without a campaign"),
    )
    result = train(config, [], ExecutionOptions(dry_run=True))
    assert result.status == "BLOCKED"
    assert result.data["backend_launches"] == 0
    assert any("campaign_config" in blocker for blocker in result.blockers)


def test_v3_invalid_campaign_and_changed_code_identity_block_without_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    campaign_path = tmp_path / "invalid-campaign.json"
    campaign_path.write_text("{}", encoding="utf-8")
    config.values["training"]["campaign_config"] = str(campaign_path)
    backend = FakeComputeBackend()
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _ready_state(tmp_path))
    monkeypatch.setattr(
        "spritelab.product_features.training.plans.build_project_state", lambda *_: _ready_state(tmp_path)
    )
    monkeypatch.setattr("spritelab.v3.orchestration.backend_from_context", lambda *_: backend)
    result = train(config, [], ExecutionOptions(dry_run=True))
    assert result.status == "BLOCKED"
    assert "prepare" not in backend.calls and "launch" not in backend.calls

    launch = validated_launch(tmp_path / "stale", "fake")
    config.values["training"]["campaign_config"] = str(launch.validator_context.campaign_config_path)
    changed = copy.deepcopy(launch.campaign["code_identity"])
    changed["files"][0]["sha256"] = "f" * 64
    changed["sha256"] = stable_hash({key: value for key, value in changed.items() if key != "sha256"})
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: changed)
    result = train(config, [], ExecutionOptions(dry_run=True))
    assert result.status == "BLOCKED"
    assert "prepare" not in backend.calls and "launch" not in backend.calls


def test_v3_valid_dry_run_issues_no_receipt_and_valid_fake_launches_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = validated_launch(tmp_path, "fake")
    config = _config(tmp_path)
    config.values["training"]["campaign_config"] = str(prepared.validator_context.campaign_config_path)
    backend = FakeComputeBackend()
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _ready_state(tmp_path))
    monkeypatch.setattr(
        "spritelab.product_features.training.plans.build_project_state", lambda *_: _ready_state(tmp_path)
    )
    monkeypatch.setattr("spritelab.v3.orchestration.backend_from_context", lambda *_: backend)

    class AuthorizedActivation:
        def __init__(self, campaign: dict) -> None:
            self.campaign = campaign
            self.selected_spec = campaign

        def to_contract_dict(self) -> dict:
            return {
                "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
                "ready": True,
                "campaign_identity_sha256": self.campaign["campaign_identity"],
                "paths_exposed": False,
            }

    monkeypatch.setattr(
        "spritelab.product_features.training.service.load_conditioned_training_activation",
        lambda *_args, **kwargs: AuthorizedActivation(kwargs["expected_campaign"]),
    )
    dry_run = train(config, [], ExecutionOptions(dry_run=True))
    assert dry_run.status == "COMPLETE"
    assert dry_run.data["validation"]["receipts_issued"] == 0
    assert "prepare" not in backend.calls and "launch" not in backend.calls

    monkeypatch.setattr("spritelab.v3.orchestration.sys.stdin.isatty", lambda: False)
    launched = train(config, [], ExecutionOptions(yes=True, non_interactive_confirm=True))
    assert launched.status == ProductStatus.RUNNING.value
    assert backend.calls.count("launch") == 3
    projected = launched.data["product_result"]["data"]["training_identity"]
    assert projected["dataset_identity"] == prepared.receipt.dataset_identity
    assert projected["view_identity"] == prepared.receipt.view_identity
    product_state = json.loads(
        (config.runs_dir / prepared.campaign["campaign_id"] / "state.json").read_text(encoding="utf-8")
    )
    assert product_state["backend_identity"]["view_identity"] == prepared.receipt.view_identity
    v3_state = json.loads((config.runs_dir / launched.run_id / "state.json").read_text(encoding="utf-8"))
    assert v3_state["backend_identity"]["dataset_identity"] == prepared.receipt.dataset_identity
    assert v3_state["backend_identity"]["training_view_identity"] == prepared.receipt.view_identity
    assert v3_state["resumable"] is False


def test_v3_declined_confirmation_is_not_misreported_as_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = validated_launch(tmp_path, "fake")
    config = _config(tmp_path)
    config.values["training"]["campaign_config"] = str(prepared.validator_context.campaign_config_path)
    backend = FakeComputeBackend()
    monkeypatch.setattr("spritelab.v3.orchestration.build_project_state", lambda *_: _ready_state(tmp_path))
    monkeypatch.setattr(
        "spritelab.product_features.training.plans.build_project_state", lambda *_: _ready_state(tmp_path)
    )
    monkeypatch.setattr("spritelab.v3.orchestration.backend_from_context", lambda *_: backend)
    monkeypatch.setattr("spritelab.v3.orchestration.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    result = train(config, [], ExecutionOptions())

    state = json.loads((config.runs_dir / result.run_id / "state.json").read_text(encoding="utf-8"))
    assert result.status == ProductStatus.PAUSED.value
    assert result.next_command == "python -m spritelab v3 train"
    assert state["resumable"] is False
    assert "prepare" not in backend.calls and "launch" not in backend.calls


@pytest.mark.parametrize(
    "training_identity",
    [
        None,
        {},
        {"dataset_identity": 7, "view_identity": "b" * 64, "training_view_identity": "b" * 64},
        {"dataset_identity": "a" * 64, "view_identity": " padded ", "training_view_identity": " padded "},
        {"dataset_identity": "a" * 64, "view_identity": "b" * 64, "training_view_identity": "c" * 64},
    ],
)
def test_v3_outer_run_rejects_malformed_product_training_identity_projection(
    training_identity: object,
) -> None:
    product_result = ProductResult(
        ProductStatus.RUNNING,
        "synthetic",
        feature="training",
        run=ProductRun("product-run", "training", "start", ProductStatus.RUNNING),
        data={"training_identity": training_identity},
    )

    with pytest.raises(CampaignValidationError, match=r"training|dataset|view"):
        _validated_product_training_identity(product_result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "wrong_schema",
        "malformed_campaign",
        "placeholder_run",
        "forged",
        "other_run",
        "other_seed",
        "other_backend",
        "argv",
        "environment",
        "output_root",
        "expired",
    ],
)
def test_ssh_rejects_invalid_receipts_before_launch_transport(tmp_path: Path, mutation: str) -> None:
    request = compute_request(tmp_path / mutation, "ssh")
    transport = RecordingSSHTransport()
    backend = _ssh_backend(transport)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    transport.calls.clear()
    changed = request
    if mutation == "missing_receipt":
        changed = replace(request, launch_receipt=None)
    elif mutation == "wrong_schema":
        receipt = receipt_with_recomputed_hash(request.launch_receipt, schema_version="unsupported")
        changed = replace(request, launch_receipt=receipt)
    elif mutation == "malformed_campaign":
        changed = replace(request, campaign_identity="not-a-hash")
    elif mutation == "placeholder_run":
        changed = replace(request, run_identity="0" * 64)
    elif mutation == "forged":
        forged = receipt_with_recomputed_hash(request.launch_receipt, campaign_identity_sha256="f" * 64)
        changed = replace(request, campaign_identity="f" * 64, launch_receipt=forged)
    elif mutation == "other_run":
        changed = replace(request, run_identity="f" * 64)
    elif mutation == "other_seed":
        forged = receipt_with_recomputed_hash(request.launch_receipt, seed=request.launch_receipt.seed + 1)
        changed = replace(request, launch_receipt=forged)
    elif mutation == "other_backend":
        forged = receipt_with_recomputed_hash(request.launch_receipt, compute_backend_id="local")
        changed = replace(request, launch_receipt=forged)
    elif mutation == "argv":
        changed = replace(request, command=(*request.command, "--forged"))
    elif mutation == "environment":
        changed = replace(request, environment={"FORGED": "1"})
    elif mutation == "output_root":
        changed = replace(request, output_root=tmp_path / "other-output")
    elif mutation == "expired":
        created = request.launch_receipt.created_at_utc
        forged = receipt_with_recomputed_hash(request.launch_receipt, expires_at_utc=created)
        changed = replace(request, launch_receipt=forged)
    with pytest.raises(TrainingLaunchRejected):
        backend.launch(prepared, changed)
    assert transport.calls == []


def test_valid_receipt_for_another_campaign_is_rejected_before_transport(tmp_path: Path) -> None:
    request = compute_request(tmp_path / "campaign-a", "ssh")
    other = compute_request(tmp_path / "campaign-b", "ssh")
    transport = RecordingSSHTransport()
    backend = _ssh_backend(transport)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    transport.calls.clear()
    with pytest.raises(TrainingLaunchRejected):
        backend.launch(prepared, replace(request, launch_receipt=other.launch_receipt))
    assert transport.calls == []


def test_valid_ssh_receipt_reaches_one_launch_transport_call(tmp_path: Path) -> None:
    request = compute_request(tmp_path, "ssh")
    transport = RecordingSSHTransport()
    backend = _ssh_backend(transport)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    transport.calls.clear()
    job = backend.launch(prepared, request)
    assert job.status == ComputeStatus.RUNNING
    assert len(transport.calls) == 1


def test_stale_receipt_after_code_identity_change_is_rejected_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = compute_request(tmp_path, "ssh")
    transport = RecordingSSHTransport()
    backend = _ssh_backend(transport)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    changed = copy.deepcopy(request.launch_receipt)
    current = copy.deepcopy(campaign_module._code_identity())
    current["files"][0]["sha256"] = "f" * 64
    current["sha256"] = stable_hash({key: value for key, value in current.items() if key != "sha256"})
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: current)
    transport.calls.clear()
    with pytest.raises(TrainingLaunchRejected):
        backend.launch(prepared, replace(request, launch_receipt=changed))
    assert transport.calls == []


def test_direct_resume_without_a_current_receipt_launches_nothing(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    request = compute_request(tmp_path, "fake")
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    missing = replace(request, idempotency_key="resume-without-receipt", launch_receipt=None)
    checkpoint = ArtifactReference(
        "checkpoint.pt",
        "a" * 64,
        prepared.remote_identity,
        tmp_path / "checkpoint.pt",
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )
    with pytest.raises(TrainingLaunchRejected):
        backend.resume(prepared, ResumeRequest(missing, checkpoint, safe_resume=True))
    assert "launch" not in backend.calls


def test_local_and_plugin_boundaries_cannot_opt_out_of_receipt_verification(tmp_path: Path) -> None:
    processes: list[list[str]] = []
    local = LocalComputeBackend(process_factory=lambda argv, **_: processes.append(argv))
    missing = replace(compute_request(tmp_path / "local", "local"), launch_receipt=None)
    with pytest.raises(TrainingLaunchRejected):
        local.launch(PreparedCompute("local", missing.idempotency_key, str(missing.output_root), "x"), missing)
    assert processes == []

    class UnsafePlugin:
        backend_id = "unsafe-plugin"
        title = "Unsafe test plugin"
        is_cloud = True

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.delegate = FakeComputeBackend(is_cloud=True)

        def __getattr__(self, name: str):
            return getattr(self.delegate, name)

        def prepare(self, context, request):
            self.calls.append("prepare")
            return PreparedCompute(self.backend_id, request.idempotency_key, "/unsafe", "x")

        def launch(self, prepared, request, *, cloud_confirmation=False):
            self.calls.append("launch")
            return ComputeJob(self.backend_id, request.idempotency_key, request.run_id, ComputeStatus.RUNNING, "x")

        def resume(self, prepared, resume, *, cloud_confirmation=False):
            self.calls.append("resume")
            return self.launch(prepared, resume.request, cloud_confirmation=cloud_confirmation)

    plugin = UnsafePlugin()
    wrapped = HostedBackendRegistry([plugin]).get(plugin.backend_id)
    invalid = replace(compute_request(tmp_path / "plugin", "unsafe-plugin"), launch_receipt=None)
    with pytest.raises(TrainingLaunchRejected):
        wrapped.prepare(ProjectContext(tmp_path, {}), invalid)
    assert plugin.calls == []


def test_idempotent_adapter_calls_still_reverify_before_returning_existing_job(tmp_path: Path) -> None:
    processes: list[list[str]] = []
    local = LocalComputeBackend(process_factory=lambda argv, **_: processes.append(argv))
    request = compute_request(tmp_path / "local", "local")
    prepared = local.prepare(ProjectContext(tmp_path, {}), request)
    first = local.launch(prepared, request)
    assert len(processes) == 1
    forged = replace(request, launch_receipt=None)
    with pytest.raises(TrainingLaunchRejected):
        local.launch(prepared, forged)
    assert len(processes) == 1
    assert local.launch(prepared, request) is first
    assert len(processes) == 1


def test_training_code_identity_covers_every_product_launch_surface() -> None:
    identity = campaign_module._code_identity()
    paths = {item["path"] for item in identity["files"]}
    assert identity["schema_version"] == "spritelab_training_code_identity_v4"
    assert {
        "src/spritelab/product_features/training/plans.py",
        "src/spritelab/product_features/training/service.py",
        "src/spritelab/product_features/training/web.py",
        "src/spritelab/product_features/dataset/certification.py",
        "src/spritelab/product_core/contracts.py",
        "src/spritelab/product_core/events.py",
        "src/spritelab/product_runtime.py",
        "src/spritelab/product_web/app.py",
        "src/spritelab/product_web/cli.py",
        "src/spritelab/product_web/events.py",
        "src/spritelab/remote_compute/contracts.py",
        "src/spritelab/remote_compute/local.py",
        "src/spritelab/remote_compute/ssh.py",
        "src/spritelab/training/launch.py",
        "src/spritelab/v3/orchestration.py",
        "src/spritelab/v3/cli.py",
        "src/spritelab/__main__.py",
    }.issubset(paths)
    assert not any(path.startswith("docs/") or path.endswith(".css") for path in paths)


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/spritelab/product_features/training/plans.py",
        "src/spritelab/product_features/training/service.py",
        "src/spritelab/product_features/dataset/certification.py",
        "src/spritelab/product_runtime.py",
        "src/spritelab/product_web/app.py",
        "src/spritelab/product_web/cli.py",
        "src/spritelab/remote_compute/local.py",
        "src/spritelab/remote_compute/ssh.py",
        "src/spritelab/v3/orchestration.py",
        "src/spritelab/training/launch.py",
        "src/spritelab/training/campaign.py",
        "src/spritelab/product_web/events.py",
    ],
)
def test_each_bound_product_launch_surface_mutation_changes_code_identity(
    monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    baseline = campaign_module._code_identity()
    original = campaign_module.file_sha256

    def changed_hash(path: str | Path) -> str:
        candidate = Path(path).resolve().relative_to(Path(campaign_module.__file__).resolve().parents[3]).as_posix()
        if candidate == relative_path:
            return "f" * 64 if original(path) != "f" * 64 else "e" * 64
        return original(path)

    monkeypatch.setattr(campaign_module, "file_sha256", changed_hash)
    assert campaign_module._code_identity()["sha256"] != baseline["sha256"]


@pytest.mark.parametrize(
    "semantic_anchor",
    [
        'EVENT_FILENAME = "events.jsonl"',
        'LEGACY_EVENT_FILENAME = "product_events.jsonl"',
        'LEGACY_SOURCE_REMOVAL_POLICY = "may_be_removed_after_verified_migration"',
        'EVENT_HISTORY_ORIGIN_FILENAME = "event_history_origin.json"',
        'EVENT_HISTORY_ORIGIN_MIGRATED_LEGACY = "migrated_legacy"',
        'EVENT_HISTORY_TRANSACTION_FILENAME = "event_history_transaction.json"',
        "def record_event_history_origin(",
        "def append_event_transactionally(",
        "def _validate_event_history_origin_record(",
        "canonical identity does not match its immutable canonical prefix",
        "cannot be reconstructed from current file presence",
        "mandatory migration record is missing; refusing native classification",
        "_atomic_bytes(canonical, legacy_bytes)",
        "Legacy event stream bytes changed after migration.",
        "canonical legacy prefix hash changed after migration.",
        "def verify_event_migration(",
        "incomplete event-history transaction requires explicit controlled recovery",
        "strict_json_loads",
        "def replay(",
    ],
)
def test_each_shared_event_semantic_mutation_changes_training_identity(
    monkeypatch: pytest.MonkeyPatch, semantic_anchor: str
) -> None:
    relative_path = "src/spritelab/product_web/events.py"
    source = Path(campaign_module.__file__).resolve().parents[3] / relative_path
    assert semantic_anchor in source.read_text(encoding="utf-8")
    baseline = campaign_module._code_identity()
    record = next(item for item in baseline["files"] if item["path"] == relative_path)
    assert record["binding"] == "whole_file"
    assert record["semantic_role"] == (
        "canonical filename, transactional append/migration, event-history origin, revalidation, and durable replay semantics"
    )
    original = campaign_module.file_sha256

    def changed_hash(path: str | Path) -> str:
        return "f" * 64 if Path(path).resolve() == source else original(path)

    monkeypatch.setattr(campaign_module, "file_sha256", changed_hash)
    assert campaign_module._code_identity()["sha256"] != baseline["sha256"]


def test_missing_shared_event_source_fails_training_identity_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.is_file

    def missing_event_source(path: Path) -> bool:
        if path.as_posix().endswith("src/spritelab/product_web/events.py"):
            return False
        return original(path)

    monkeypatch.setattr(Path, "is_file", missing_event_source)
    with pytest.raises(CampaignValidationError, match=r"product_web/events\.py"):
        campaign_module.training_code_identity_source_paths()


def test_missing_dataset_certification_source_fails_training_identity_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.is_file

    def missing_certification_source(path: Path) -> bool:
        if path.as_posix().endswith("src/spritelab/product_features/dataset/certification.py"):
            return False
        return original(path)

    monkeypatch.setattr(Path, "is_file", missing_certification_source)
    with pytest.raises(CampaignValidationError, match=r"dataset/certification\.py"):
        campaign_module.training_code_identity_source_paths()


def test_training_audit_surface_stales_on_adapter_change_but_not_documentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "src/spritelab/remote_compute/local.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("validated = True\n", encoding="utf-8")
    docs = tmp_path / "docs/readme.md"
    docs.parent.mkdir()
    docs.write_text("one\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(adapter.read_bytes()).hexdigest()
    files = [{"path": adapter.relative_to(tmp_path).as_posix(), "sha256_before": digest}]
    monkeypatch.setattr(
        activation_module,
        "TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS",
        ("src/spritelab",),
    )
    monkeypatch.setattr(activation_module, "training_code_identity_source_paths", lambda _root: (adapter,))

    assert activation_module._verify_audited_code_inventory(tmp_path, files) is True
    docs.write_text("two\n", encoding="utf-8")
    assert activation_module._verify_audited_code_inventory(tmp_path, files) is True
    adapter.write_text("validated = False\n", encoding="utf-8")
    assert activation_module._verify_audited_code_inventory(tmp_path, files) is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/spritelab/training/campaign.py",
        "src/spritelab/training/launch.py",
        "src/spritelab/product_features/training/plans.py",
        "src/spritelab/product_features/training/service.py",
        "src/spritelab/product_runtime.py",
        "src/spritelab/product_web/app.py",
        "src/spritelab/product_web/cli.py",
        "src/spritelab/v3/orchestration.py",
        "src/spritelab/remote_compute/local.py",
        "src/spritelab/remote_compute/ssh.py",
        "src/spritelab/product_web/events.py",
    ],
)
def test_training_audit_is_stale_after_each_bound_launch_surface_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    files = [{"path": relative_path, "sha256_before": digest}]
    monkeypatch.setattr(
        activation_module,
        "TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS",
        ("src/spritelab",),
    )
    monkeypatch.setattr(activation_module, "training_code_identity_source_paths", lambda _root: (target,))

    assert activation_module._verify_audited_code_inventory(tmp_path, files) is True
    target.write_text("after\n", encoding="utf-8")
    assert activation_module._verify_audited_code_inventory(tmp_path, files) is False

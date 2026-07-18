from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from spritelab.product_features.training.activation import (
    CONDITIONED_DATASET_FREEZE_SCHEMA,
    build_conditioned_three_seed_campaign,
)
from spritelab.training import campaign as campaign_module
from spritelab.training import launch as launch_module
from spritelab.training.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    DEFAULT_SEEDS,
    CampaignValidationError,
    file_sha256,
)
from spritelab.training.launch import (
    TrainingFilesystemCapability,
    load_exact_campaign_configuration,
    prepare_validated_training_launch,
    validate_training_launch_plan,
    verify_validated_training_launch,
)
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation
from training_launch_test_utils import validated_launch


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_isolated_bootstrap_imports_project_code_only_from_retained_bundle(tmp_path: Path) -> None:
    launch = validated_launch(tmp_path, "local")
    with TrainingFilesystemCapability(launch.campaign, tmp_path) as capability:
        verified = verify_validated_training_launch(
            launch.receipt,
            launch.validator_context,
            compute_backend_id="local",
            argv=launch.argv,
            environment=launch.environment,
            output_root=launch.output_root,
            campaign_identity=launch.receipt.campaign_identity_sha256,
            run_identity=launch.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        child_command = (*capability.bootstrap_command(verified), "--help")
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            environment = dict(verified.environment)
            environment.update(boundary_environment)
            result = subprocess.run(
                child_command,
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                **spawn_options,
            )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("loader_key", ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"])
def test_launch_rejects_environment_loader_injection_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader_key: str,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)

    with pytest.raises(CampaignValidationError, match="forbidden loader key"):
        prepare_validated_training_launch(
            campaign_config,
            run_id=str(campaign["expected_runs"][0]["run_id"]),
            compute_backend_id="local",
            project_root=root,
            execute_confirmed=True,
            environment={loader_key: str(root)},
        )


def _portable_campaign_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "project"
    publication = root / "artifacts" / "dataset" / "conditioned-v5"
    campaign_directory = root / "artifacts" / "training"
    publication.mkdir(parents=True)
    campaign_directory.mkdir(parents=True)

    repository_root = Path(campaign_module.__file__).resolve().parents[3]
    synthetic_sources = ("src/spritelab/__init__.py", "src/spritelab/__main__.py")
    synthetic_code_identity = {
        "schema_version": "synthetic_training_code_identity_v1",
        "sha256": "a" * 64,
        "files": [
            {"path": relative, "sha256": file_sha256(repository_root / relative)} for relative in synthetic_sources
        ],
    }
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: synthetic_code_identity)

    artifacts: dict[str, Path] = {}
    for name in ("view", "split", "vocabulary", "benchmark"):
        path = publication / f"{name}.json"
        value = (
            {"split": "train", "sprite_id": "synthetic", "npz_file": "train.npz", "npz_row": 0}
            if name == "split"
            else {"artifact": name}
        )
        if name == "split":
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        else:
            _write_json(path, value)
        artifacts[name] = path
    (publication / "train.npz").write_bytes(b"synthetic retained dataset artifact")
    activation = publication / "activation.json"
    _write_json(
        activation,
        {
            "schema_version": CONDITIONED_DATASET_FREEZE_SCHEMA,
            "image_count": 2_417,
        },
    )

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    built = build_conditioned_three_seed_campaign(
        root,
        campaign_directory=relative(campaign_directory),
        activation_manifest=relative(activation),
        activation_manifest_sha256=file_sha256(activation),
        view_manifest=relative(artifacts["view"]),
        split_manifest=relative(artifacts["split"]),
        conditioning_vocabulary=relative(artifacts["vocabulary"]),
        benchmark_manifest=relative(artifacts["benchmark"]),
        output_root="runs/training",
        campaign_id="conditioned-v5-production",
    )
    campaign_config = campaign_directory / "campaigns.json"
    _write_json(
        campaign_config,
        {"product_profiles": {"recommended": {"campaign": dict(built.portable_campaign)}}},
    )
    for run in built.campaign["expected_runs"]:
        _write_json(Path(str(run["resolved_config_path"])), run["resolved_config"])
    return root, campaign_config, dict(built.campaign)


def _spawn_boundary_probe(
    *,
    project_root: Path,
    config_path: Path,
    resume_path: Path | None,
    boundary_environment: dict[str, str],
    spawn_options: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(boundary_environment)
    script = (
        "import json,os,sys;"
        "encoded=os.environ['SPRITELAB_VALIDATED_TRAINING_BOUNDARY'];"
        "payload=json.loads(encoded);"
        "tokens=[payload['root_token'],payload['config_token']]+[item['token'] for item in payload['input_files']];"
        "tokens += ([] if payload['checkpoint_token'] is None else [payload['checkpoint_token']]);"
        "from spritelab.training.launch import bootstrap_validated_training_process;"
        "resume=None if sys.argv[2]=='-' else sys.argv[2];"
        "boundary=bootstrap_validated_training_process(sys.argv[1],resume);"
        "inheritable=[(os.get_handle_inheritable(token['value']) if token['kind']=='windows_handle' "
        "else os.get_inheritable(token['value'])) for token in tokens];"
        "print(json.dumps({'logical':str(boundary.logical_output_root),"
        "'physical':str(boundary.output_root),'resume':boundary.resume_descriptor,"
        "'boundary_env_present':'SPRITELAB_VALIDATED_TRAINING_BOUNDARY' in os.environ,"
        "'inheritable':inheritable}))"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(config_path), "-" if resume_path is None else str(resume_path)],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        **spawn_options,
    )


def _native_resumable_launch(
    root: Path,
    campaign_config: Path,
    campaign: dict[str, Any],
) -> tuple[Any, Path]:
    from spritelab.product_web.events import EVENT_FILENAME, EVENT_HISTORY_ORIGIN_NATIVE, record_event_history_origin

    run = campaign["expected_runs"][0]
    output_root = Path(str(run["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "run_identity.json",
        {
            "campaign_id": campaign["campaign_id"],
            "campaign_identity": campaign["campaign_identity"],
            "run_id": run["run_id"],
            "run_identity": run["run_identity"],
            "output_root": run["output_root"],
            "resolved_config_sha256": run["resolved_config_sha256"],
            "execution_contract_sha256": run["execution_contract_sha256"],
        },
    )
    (output_root / EVENT_FILENAME).write_bytes(b"")
    record_event_history_origin(
        str(run["run_id"]),
        output_root,
        expected_origin=EVENT_HISTORY_ORIGIN_NATIVE,
        allow_binding_population=True,
    )
    step = int(run["expected_checkpoint_steps"][0])
    checkpoint = output_root / f"checkpoint_step_{step:06d}.pt"
    checkpoint.write_bytes(b"exact retained checkpoint")
    checkpoint_sha256 = file_sha256(checkpoint)
    _write_json(
        output_root / f"checkpoint_step_{step:06d}.json",
        {
            "optimizer_step": step,
            "campaign_identity": campaign["campaign_identity"],
            "run_identity": run["run_identity"],
            "resumability_metadata": {
                "schema_version": campaign_module.RESUME_CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_relative_path": checkpoint.name,
                "checkpoint_content_sha256": checkpoint_sha256,
                "source_checkpoint_identity": checkpoint_sha256,
                "target_runtime_identity": run["run_identity"],
                "experiment_manifest_identity": campaign_module.stable_hash(run["resolved_config"]),
                "exact_replay_eligible": True,
                "unsafe_resume": False,
                "max_optimizer_steps": campaign["training"]["max_optimizer_steps"],
                "gradient_accumulation_position": 0,
                "state_presence": {
                    "model_state_dict": True,
                    "optimizer_state_dict": True,
                    "scheduler_state_dict": True,
                    "ema_state_dict": True,
                    "rng_states": True,
                    "sampler_state": True,
                    "dataloader_generator_state": True,
                },
            },
        },
    )
    return (
        prepare_validated_training_launch(
            campaign_config,
            run_id=str(run["run_id"]),
            compute_backend_id="local",
            project_root=root,
            execute_confirmed=True,
            resume=True,
        ),
        checkpoint,
    )


def test_builder_portable_profile_reaches_an_exact_verified_launch_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    environment = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "SPRITELAB_PROGRESS": "0"}
    portable = json.loads(campaign_config.read_text(encoding="utf-8"))["product_profiles"]["recommended"]["campaign"]

    assert portable["identities"]["dataset_freeze_path"].startswith("../dataset/")

    dry_run = validate_training_launch_plan(
        campaign_config,
        compute_backend_id="local",
        project_root=root,
        campaign_profile="recommended",
        environment=environment,
    )

    assert dry_run["valid"] is True
    assert dry_run["campaign_identity_sha256"] == built_campaign["campaign_identity"]
    assert [item["seed"] for item in dry_run["launches"]] == list(DEFAULT_SEEDS)
    assert dry_run["receipts_issued"] == 0
    assert dry_run["processes_started"] == 0

    first_run = built_campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(first_run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
        campaign_profile="recommended",
        environment=environment,
    )
    verified = verify_validated_training_launch(
        prepared.receipt,
        prepared.validator_context,
        compute_backend_id="local",
        argv=prepared.argv,
        environment=prepared.environment,
        output_root=prepared.output_root,
        campaign_identity=str(built_campaign["campaign_identity"]),
        run_identity=str(first_run["run_identity"]),
    )

    assert verified.receipt == prepared.receipt
    assert prepared.argv == tuple(first_run["experiment_command"])
    assert prepared.receipt.argv_sha256 == dry_run["launches"][0]["argv_sha256"]
    assert prepared.output_root == Path(str(first_run["output_root"])).resolve()
    assert prepared.output_root.is_relative_to(root)
    assert not prepared.output_root.exists()


def test_retained_training_boundary_is_revalidated_inside_the_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
    )
    with TrainingFilesystemCapability(prepared.campaign, root) as capability:
        verified = verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.receipt.campaign_identity_sha256,
            run_identity=prepared.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            result = _spawn_boundary_probe(
                project_root=root,
                config_path=Path(str(run["resolved_config_path"])),
                resume_path=None,
                boundary_environment=boundary_environment,
                spawn_options=spawn_options,
            )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["logical"] == str(prepared.output_root)
    assert payload["resume"] is None
    assert payload["boundary_env_present"] is False
    assert payload["inheritable"] and not any(payload["inheritable"])
    assert Path(payload["physical"]).samefile(prepared.output_root)


@pytest.mark.parametrize("target", ["manifest", "dataset"])
def test_child_boundary_rejects_equal_length_retained_input_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
    )
    dataset = run["resolved_config"]["dataset"]
    manifest = root / str(dataset["training_manifest"])
    retained_input = manifest if target == "manifest" else root / str(dataset["directory"]) / "train.npz"
    original = retained_input.read_bytes()
    mutated = bytearray(original)
    mutated[0] ^= 1

    with TrainingFilesystemCapability(prepared.campaign, root) as capability:
        verified = verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.receipt.campaign_identity_sha256,
            run_identity=prepared.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            before = retained_input.stat()
            with retained_input.open("r+b", buffering=0) as handle:
                handle.write(mutated)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(retained_input, ns=(before.st_atime_ns, before.st_mtime_ns))
            restored = retained_input.stat()
            assert restored.st_ino == before.st_ino
            assert restored.st_size == before.st_size
            assert restored.st_mtime_ns == before.st_mtime_ns

            result = _spawn_boundary_probe(
                project_root=root,
                config_path=Path(str(run["resolved_config_path"])),
                resume_path=None,
                boundary_environment=boundary_environment,
                spawn_options=spawn_options,
            )

    assert result.returncode != 0
    assert "retained input content changed" in result.stderr


def test_child_boundary_blocks_output_root_replacement_and_preserves_outside_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
    )
    outside = tmp_path / "outside-output-root-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-output-root-unchanged")
    before = sentinel.read_bytes()
    blocked_by_held_handle = False
    exit_error: UnsafeFilesystemOperation | None = None
    result: subprocess.CompletedProcess[str] | None = None
    try:
        with TrainingFilesystemCapability(prepared.campaign, root) as capability:
            verified = verify_validated_training_launch(
                prepared.receipt,
                prepared.validator_context,
                compute_backend_id="local",
                argv=prepared.argv,
                environment=prepared.environment,
                output_root=prepared.output_root,
                campaign_identity=prepared.receipt.campaign_identity_sha256,
                run_identity=prepared.receipt.run_identity,
                filesystem_snapshot=capability.filesystem_snapshot,
            )
            with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
                replacement = prepared.output_root.with_name(f"{prepared.output_root.name}-replacement")
                replacement.mkdir()
                try:
                    os.replace(replacement, prepared.output_root)
                except OSError:
                    blocked_by_held_handle = True
                result = _spawn_boundary_probe(
                    project_root=root,
                    config_path=Path(str(run["resolved_config_path"])),
                    resume_path=None,
                    boundary_environment=boundary_environment,
                    spawn_options=spawn_options,
                )
    except UnsafeFilesystemOperation as exc:
        exit_error = exc

    assert result is not None
    if blocked_by_held_handle:
        assert result.returncode == 0, result.stderr
        assert exit_error is None
    else:
        assert result.returncode != 0
        assert "substituted" in result.stderr
        assert exit_error is not None
    assert sentinel.read_bytes() == before


def test_child_boundary_uses_exact_retained_checkpoint_across_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    prepared, checkpoint = _native_resumable_launch(root, campaign_config, campaign)
    outside = tmp_path / "outside-checkpoint-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-checkpoint-unchanged")
    before = sentinel.read_bytes()
    replacement = checkpoint.with_name("replacement-checkpoint.pt")
    replacement.write_bytes(b"hostile replacement checkpoint")

    with TrainingFilesystemCapability(prepared.campaign, root) as capability:
        verified = verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.receipt.campaign_identity_sha256,
            run_identity=prepared.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            try:
                os.replace(replacement, checkpoint)
            except OSError:
                blocked_by_held_handle = True
            else:
                blocked_by_held_handle = False
            result = _spawn_boundary_probe(
                project_root=root,
                config_path=Path(str(prepared.run["resolved_config_path"])),
                resume_path=checkpoint,
                boundary_environment=boundary_environment,
                spawn_options=spawn_options,
            )

    if blocked_by_held_handle:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["resume"] is not None
    else:
        assert result.returncode != 0
        assert "substituted" in result.stderr
    assert sentinel.read_bytes() == before


def test_retained_audit_rejects_event_hardlink_before_any_event_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
    )
    outside = tmp_path / "outside-event-audit"
    outside.mkdir()
    sentinel = outside / "sentinel.jsonl"
    sentinel.write_bytes(b'{"outside":"unchanged"}\n')
    before = sentinel.read_bytes()
    prepared.output_root.mkdir(parents=True)
    event_path = prepared.output_root / "events.jsonl"
    try:
        os.link(sentinel, event_path)
    except (NotImplementedError, OSError):
        pytest.skip("hard links are unavailable in this test session")

    def unexpected_event_traversal(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("retained audit must reject an aliased event file before path traversal")

    monkeypatch.setattr(campaign_module, "_classify_run_root", unexpected_event_traversal)

    with pytest.raises(CampaignValidationError, match="aliased"):
        with TrainingFilesystemCapability(prepared.campaign, root):
            pass

    assert sentinel.read_bytes() == before


def test_child_boundary_rejects_event_alias_inserted_after_parent_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    run = campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
    )
    outside = tmp_path / "outside-child-event-race"
    outside.mkdir()
    sentinel = outside / "sentinel.jsonl"
    sentinel.write_bytes(b'{"outside":"unchanged"}\n')
    before = sentinel.read_bytes()

    with TrainingFilesystemCapability(prepared.campaign, root) as capability:
        verified = verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.receipt.campaign_identity_sha256,
            run_identity=prepared.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            try:
                os.link(sentinel, prepared.output_root / "events.jsonl")
            except (NotImplementedError, OSError):
                pytest.skip("hard links are unavailable in this test session")
            result = _spawn_boundary_probe(
                project_root=root,
                config_path=Path(str(run["resolved_config_path"])),
                resume_path=None,
                boundary_environment=boundary_environment,
                spawn_options=spawn_options,
            )

    assert result.returncode != 0
    assert "aliased" in result.stderr or "entries changed" in result.stderr
    assert sentinel.read_bytes() == before


def test_child_boundary_rejects_equal_length_in_place_sidecar_mutation_with_restored_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    prepared, checkpoint = _native_resumable_launch(root, campaign_config, campaign)
    sidecar = checkpoint.with_suffix(".json")
    original = sidecar.read_bytes()
    mutated = bytearray(original)
    mutated[0] = ord("[") if mutated[0] != ord("[") else ord("{")
    assert len(mutated) == len(original)
    assert bytes(mutated) != original

    outside = tmp_path / "outside-sidecar-content-race"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside-sidecar-content-unchanged")
    sentinel_before = sentinel.read_bytes()

    with TrainingFilesystemCapability(prepared.campaign, root) as capability:
        verified = verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.receipt.campaign_identity_sha256,
            run_identity=prepared.receipt.run_identity,
            filesystem_snapshot=capability.filesystem_snapshot,
        )
        with capability.launch_inheritance(verified) as (boundary_environment, spawn_options):
            before = sidecar.stat()
            with sidecar.open("r+b", buffering=0) as handle:
                handle.write(mutated)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(sidecar, ns=(before.st_atime_ns, before.st_mtime_ns))
            restored = sidecar.stat()
            assert restored.st_ino == before.st_ino
            assert restored.st_size == before.st_size
            assert restored.st_mtime_ns == before.st_mtime_ns
            assert sidecar.read_bytes() == bytes(mutated)

            result = _spawn_boundary_probe(
                project_root=root,
                config_path=Path(str(prepared.run["resolved_config_path"])),
                resume_path=checkpoint,
                boundary_environment=boundary_environment,
                spawn_options=spawn_options,
            )

    assert result.returncode != 0
    assert "output entries changed" in result.stderr
    assert sentinel.read_bytes() == sentinel_before


def test_retained_output_control_snapshot_fails_closed_above_its_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bounded-output"
    root.mkdir()
    (root / "run_identity.json").write_bytes(b"{}")
    monkeypatch.setattr(launch_module, "_MAX_BOUND_OUTPUT_CONTROL_FILE_BYTES", 1)

    with AnchoredDirectory(root, root) as anchor:
        with pytest.raises(CampaignValidationError, match="output-control byte limit"):
            launch_module._anchored_entry_snapshot(anchor)


def test_portable_profile_refuses_an_output_root_outside_the_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, _campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    escaped = deepcopy(document)
    escaped["product_profiles"]["recommended"]["campaign"]["output_root"] = "../../../../outside"
    escaped_config = campaign_config.with_name("escaped-campaigns.json")
    _write_json(escaped_config, escaped)

    with pytest.raises(CampaignValidationError, match="output_root escapes the approved project"):
        validate_training_launch_plan(
            escaped_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )


@pytest.mark.parametrize("selection", ["inline", "campaign_path"])
def test_product_profile_refuses_a_materialized_manifest_before_launch_work_or_external_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    external = tmp_path / "external-dataset.json"
    external.write_text("external sentinel\n", encoding="utf-8")
    materialized = deepcopy(built_campaign)
    materialized["identities"]["dataset_view_manifest_path"] = str(external)
    nested = campaign_config.with_name("materialized.json")
    if selection == "inline":
        profile_entry: dict[str, Any] = {"campaign": materialized}
    else:
        _write_json(nested, materialized)
        profile_entry = {"campaign_path": nested.name}
    hostile_config = campaign_config.with_name(f"materialized-{selection}-profile.json")
    _write_json(hostile_config, {"product_profiles": {"recommended": profile_entry}})

    external_reads: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.resolve() == external.resolve():
            external_reads.append(path)
        return original_open(path, *args, **kwargs)

    planning_calls: list[str] = []

    def unexpected_planning(*_args: Any, **_kwargs: Any) -> Any:
        planning_calls.append("called")
        raise AssertionError("launch planning must not run for a profile-selected materialized manifest")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(launch_module, "plan_campaign", unexpected_planning)
    monkeypatch.setattr(launch_module, "audit_resume", unexpected_planning)
    monkeypatch.setattr(launch_module, "_authoritative_snapshot", unexpected_planning)

    with pytest.raises(CampaignValidationError, match="must be unplanned specifications"):
        validate_training_launch_plan(
            hostile_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert planning_calls == []
    assert external_reads == []
    assert external.read_text(encoding="utf-8") == "external sentinel\n"
    assert external_reads == [external]


def test_direct_top_level_materialized_manifest_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    direct = campaign_config.with_name("direct-materialized.json")
    _write_json(direct, built_campaign)
    protected = {
        Path(os.path.abspath(path))
        for path in (
            direct,
            *(Path(str(run["resolved_config_path"])) for run in built_campaign["expected_runs"]),
        )
    }
    original_read_text = Path.read_text

    def refuse_plain_launch_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if Path(os.path.abspath(path)) in protected:
            raise AssertionError("launch configuration must be read through an anchored descriptor")
        return original_read_text(path, *args, **kwargs)

    bound_inputs = {
        Path(os.path.abspath(path))
        for path in (
            built_campaign["identities"]["dataset_view_manifest_path"],
            built_campaign["identities"]["split_manifest_path"],
            built_campaign["identities"]["conditioning_vocabulary_path"],
            built_campaign["evaluation"]["benchmark_manifest_path"],
        )
    }
    original_campaign_file_sha256 = campaign_module.file_sha256

    def refuse_reopened_bound_input(path: Path, *args: Any, **kwargs: Any) -> str:
        if Path(os.path.abspath(path)) in bound_inputs:
            raise AssertionError("campaign validation must consume retained anchored input hashes")
        return original_campaign_file_sha256(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_plain_launch_read)
    monkeypatch.setattr(campaign_module, "file_sha256", refuse_reopened_bound_input)

    loaded = load_exact_campaign_configuration(direct, profile="recommended")
    dry_run = validate_training_launch_plan(
        direct,
        compute_backend_id="local",
        project_root=root,
        campaign_profile="recommended",
    )
    prepared = prepare_validated_training_launch(
        direct,
        run_id=str(built_campaign["expected_runs"][0]["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
        campaign_profile="recommended",
    )

    assert loaded == built_campaign
    assert dry_run["valid"] is True
    assert dry_run["processes_started"] == 0
    assert prepared.receipt.campaign_manifest_sha256 == file_sha256(direct)


def test_campaign_resume_audit_accepts_a_retained_snapshot_without_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, _campaign_config, campaign = _portable_campaign_project(tmp_path, monkeypatch)
    campaign = deepcopy(campaign)
    for index, run in enumerate(campaign["expected_runs"]):
        run["output_root"] = str(_root / "fresh-audit" / f"cell-{index}" / "seed")
    campaign["expected_output_roots"] = [run["output_root"] for run in campaign["expected_runs"]]
    campaign["campaign_identity"] = campaign_module.stable_hash(campaign_module._campaign_identity_payload(campaign))
    snapshot = campaign_module._capture_fresh_campaign_filesystem_snapshot(campaign, _root)
    assert snapshot is not None

    def unexpected_traversal(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a retained campaign filesystem snapshot must not reopen output paths")

    monkeypatch.setattr(campaign_module, "_classify_run_root", unexpected_traversal)
    monkeypatch.setattr(campaign_module, "_foreign_run_roots", unexpected_traversal)

    report = campaign_module.audit_resume(campaign, filesystem_snapshot=snapshot)

    assert report["safe"] is True
    assert report["root_state"] == "fresh"

    with pytest.raises(AttributeError, match="immutable"):
        snapshot.campaign_identity = "b" * 64
    with pytest.raises(TypeError):
        snapshot.runs[0]["status"] = "foreign"

    with pytest.raises(CampaignValidationError, match="trusted capture seam"):
        campaign_module.audit_resume(
            campaign,
            filesystem_snapshot={
                "schema_version": "spritelab_campaign_filesystem_snapshot_v1",
                "campaign_identity": campaign["campaign_identity"],
                "foreign_run_roots": [],
                "runs": snapshot.runs,
            },
        )


def test_nested_campaign_source_bytes_are_bound_into_receipt_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    profile = document["product_profiles"]["recommended"]
    nested_campaign = profile.pop("campaign")
    nested_path = campaign_config.with_name("nested-campaign.json")
    _write_json(nested_path, nested_campaign)
    profile["campaign_path"] = nested_path.name
    wrapper = campaign_config.with_name("nested-wrapper.json")
    _write_json(wrapper, document)

    prepared = prepare_validated_training_launch(
        wrapper,
        run_id=str(built_campaign["expected_runs"][0]["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
        campaign_profile="recommended",
    )
    original_manifest_sha256 = prepared.receipt.campaign_manifest_sha256
    nested_path.write_text(json.dumps(nested_campaign, sort_keys=True), encoding="utf-8")

    with pytest.raises(CampaignValidationError, match="execution_spec_sha256"):
        verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.campaign["campaign_identity"],
            run_identity=prepared.run["run_identity"],
        )

    assert file_sha256(wrapper) == original_manifest_sha256


def test_verification_rejects_first_hostile_snapshot_without_a_second_a_b_a_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(built_campaign["expected_runs"][0]["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
        campaign_profile="recommended",
    )
    original_snapshot = launch_module._authoritative_snapshot
    calls = 0

    def alternating_snapshot(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        campaign, run, command, environment, snapshot = original_snapshot(*args, **kwargs)
        if calls % 2 == 1:
            run = deepcopy(run)
            run["seed"] = int(run["seed"]) + 1
        return campaign, run, command, environment, snapshot

    monkeypatch.setattr(launch_module, "_authoritative_snapshot", alternating_snapshot)

    with pytest.raises(CampaignValidationError, match="stale or forged"):
        verify_validated_training_launch(
            prepared.receipt,
            prepared.validator_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=prepared.campaign["campaign_identity"],
            run_identity=prepared.run["run_identity"],
        )

    assert calls == 1


@pytest.mark.parametrize(
    "hostile_field",
    [
        "expected_output_root",
        "run_output_root",
        "resolved_config_path",
        "command_config_path",
        "campaign_artifact_root",
        "dataset_freeze_path",
        "dataset_view_manifest_path",
        "split_manifest_path",
        "conditioning_vocabulary_path",
        "benchmark_manifest_path",
        "resolved_dataset_directory",
        "resolved_training_manifest",
        "resolved_split_manifest",
        "resolved_vocabulary_path",
        "resolved_runtime_out_dir",
    ],
)
def test_direct_materialized_manifest_refuses_every_external_path_before_validation_or_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_field: str,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    outside = tmp_path / "private-outside"
    outside.mkdir()
    sentinel = outside / "sentinel.json"
    sentinel.write_bytes(b'{"private":"unchanged"}\n')
    before = sentinel.read_bytes()
    hostile = deepcopy(built_campaign)
    outside_output = str(outside / "run")
    outside_input = str(sentinel)
    first_run = hostile["expected_runs"][0]
    if hostile_field == "expected_output_root":
        hostile["expected_output_roots"][0] = outside_output
    elif hostile_field == "run_output_root":
        first_run["output_root"] = outside_output
    elif hostile_field == "resolved_config_path":
        first_run["resolved_config_path"] = outside_input
    elif hostile_field == "command_config_path":
        first_run["experiment_command"][-1] = outside_input
    elif hostile_field == "campaign_artifact_root":
        hostile["campaign_artifact_root"] = outside_output
    elif hostile_field == "benchmark_manifest_path":
        hostile["evaluation"]["benchmark_manifest_path"] = outside_input
    elif hostile_field == "resolved_dataset_directory":
        first_run["resolved_config"]["dataset"]["directory"] = outside_output
    elif hostile_field == "resolved_training_manifest":
        first_run["resolved_config"]["dataset"]["training_manifest"] = outside_input
    elif hostile_field == "resolved_split_manifest":
        first_run["resolved_config"]["dataset"]["split_manifest"] = outside_input
    elif hostile_field == "resolved_vocabulary_path":
        first_run["resolved_config"]["conditioning"]["vocabulary_path"] = outside_input
    elif hostile_field == "resolved_runtime_out_dir":
        first_run["resolved_config"]["runtime"]["out_dir"] = outside_output
    else:
        hostile["identities"][hostile_field] = outside_input
    direct = campaign_config.with_name(f"direct-external-{hostile_field}.json")
    _write_json(direct, hostile)

    consumer_calls: list[str] = []

    def unexpected_consumer(*_args: Any, **_kwargs: Any) -> Any:
        consumer_calls.append("called")
        raise AssertionError("external materialized paths must fail before campaign consumers")

    original_confined_read = launch_module._read_confined_regular_bytes

    def guarded_confined_read(path: Path, project_root: Path, **kwargs: Any) -> bytes:
        candidate = Path(os.path.abspath(path))
        if candidate.is_relative_to(outside):
            raise AssertionError("external materialized input was opened")
        return original_confined_read(path, project_root, **kwargs)

    monkeypatch.setattr(launch_module, "validate_campaign", unexpected_consumer)
    monkeypatch.setattr(launch_module, "audit_resume", unexpected_consumer)
    monkeypatch.setattr(launch_module, "plan_campaign", unexpected_consumer)
    monkeypatch.setattr(launch_module, "_read_confined_regular_bytes", guarded_confined_read)

    with pytest.raises(CampaignValidationError) as captured:
        validate_training_launch_plan(
            direct,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    message = str(captured.value)
    if hostile_field == "command_config_path":
        assert "exact launch command" in message
    else:
        assert "approved project" in message
    assert str(outside) not in str(captured.value)
    assert consumer_calls == []
    assert sentinel.read_bytes() == before


@pytest.mark.parametrize("command_attack", ["executable", "module", "extra_argument"])
def test_direct_materialized_manifest_refuses_noncanonical_command_before_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_attack: str,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    hostile = deepcopy(built_campaign)
    command = hostile["expected_runs"][0]["experiment_command"]
    if command_attack == "executable":
        command[0] = "powershell"
    elif command_attack == "module":
        command[2] = "hostile.module"
    else:
        command.insert(-2, "--hostile-extra")
    direct = campaign_config.with_name(f"direct-command-{command_attack}.json")
    _write_json(direct, hostile)

    def unexpected_consumer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a noncanonical command must fail before validation or audit")

    monkeypatch.setattr(launch_module, "validate_campaign", unexpected_consumer)
    monkeypatch.setattr(launch_module, "audit_resume", unexpected_consumer)

    with pytest.raises(CampaignValidationError, match="exact launch command"):
        validate_training_launch_plan(
            direct,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )


def test_direct_materialized_manifest_outside_source_is_rejected_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    outside = tmp_path / "private-outside-campaign.json"
    _write_json(outside, built_campaign)
    before = outside.read_bytes()

    def unexpected_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an external campaign source must not be read")

    monkeypatch.setattr(launch_module, "_read_confined_regular_bytes", unexpected_read)

    with pytest.raises(CampaignValidationError) as captured:
        validate_training_launch_plan(
            outside,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert str(captured.value) == "campaign configuration path escapes the approved project"
    assert str(outside) not in str(captured.value)
    assert outside.read_bytes() == before


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_direct_materialized_manifest_linked_source_is_rejected_before_content_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    root, _campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    source_directory = root if link_kind == "symlink" else tmp_path
    source = source_directory / f"private-{link_kind}-campaign.json"
    _write_json(source, built_campaign)
    linked = root / f"linked-{link_kind}-campaign.json"
    try:
        if link_kind == "symlink":
            os.symlink(source, linked)
        else:
            os.link(source, linked)
    except (NotImplementedError, OSError):
        pytest.skip(f"{link_kind} is unavailable in this test session")
    before = source.read_bytes()
    opened_source: list[str] = []
    original_open = AnchoredDirectory.open_file_immovable

    def guarded_open(
        anchor: AnchoredDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        if anchor.directory == linked.parent and name == linked.name:
            opened_source.append(name)
        return original_open(anchor, name, flags, mode)

    monkeypatch.setattr(AnchoredDirectory, "open_file_immovable", guarded_open)

    with pytest.raises(CampaignValidationError) as captured:
        validate_training_launch_plan(
            linked,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert "unsafe" in str(captured.value) or "approved project" in str(captured.value)
    assert str(source) not in str(captured.value)
    assert opened_source == []
    assert source.read_bytes() == before


def test_direct_materialized_manifest_source_swap_is_rejected_before_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    direct = campaign_config.with_name("direct-source-swap.json")
    _write_json(direct, built_campaign)
    original_lstat = AnchoredDirectory.lstat
    source_lstats = 0

    def raced_lstat(anchor: AnchoredDirectory, name: str) -> os.stat_result:
        nonlocal source_lstats
        metadata = original_lstat(anchor, name)
        if anchor.directory == direct.parent and name == direct.name:
            source_lstats += 1
            if source_lstats >= 2:
                values = list(metadata)
                values[1] = int(values[1]) + 1
                return os.stat_result(values)
        return metadata

    def unexpected_consumer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a raced campaign source must fail before validation or audit")

    monkeypatch.setattr(AnchoredDirectory, "lstat", raced_lstat)
    monkeypatch.setattr(launch_module, "validate_campaign", unexpected_consumer)
    monkeypatch.setattr(launch_module, "audit_resume", unexpected_consumer)

    with pytest.raises(CampaignValidationError, match="changed while being read"):
        validate_training_launch_plan(
            direct,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert source_lstats == 2


@pytest.mark.parametrize(
    "reducible",
    [
        "runs/./training",
        "runs//training",
        "runs/training/",
        "runs/../training",
    ],
)
def test_product_profile_refuses_noncanonical_portable_path_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reducible: str,
) -> None:
    root, campaign_config, _campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    document["product_profiles"]["recommended"]["campaign"]["output_root"] = reducible
    hostile_config = campaign_config.with_name("noncanonical-portable-path.json")
    _write_json(hostile_config, document)

    planning_calls: list[str] = []

    def unexpected_planning(*_args: Any, **_kwargs: Any) -> Any:
        planning_calls.append("called")
        raise AssertionError("planning must not run for a noncanonical portable path")

    monkeypatch.setattr(launch_module, "plan_campaign", unexpected_planning)

    with pytest.raises(CampaignValidationError, match="output_root must be a canonical relative path"):
        validate_training_launch_plan(
            hostile_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert planning_calls == []


def test_product_profile_requires_an_explicit_output_root_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, _campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    document["product_profiles"]["recommended"]["campaign"].pop("output_root")
    hostile_config = campaign_config.with_name("missing-output-root.json")
    _write_json(hostile_config, document)
    planning_calls: list[str] = []

    def unexpected_planning(*_args: Any, **_kwargs: Any) -> Any:
        planning_calls.append("called")
        raise AssertionError("planning must not run without an explicit product-profile output root")

    monkeypatch.setattr(launch_module, "plan_campaign", unexpected_planning)

    with pytest.raises(CampaignValidationError, match="output_root is required"):
        validate_training_launch_plan(
            hostile_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert planning_calls == []


def test_product_profile_resolves_and_confines_optional_campaign_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, _campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    portable = document["product_profiles"]["recommended"]["campaign"]
    portable["campaign_artifact_root"] = "../campaign-evidence"
    configured = campaign_config.with_name("campaign-artifact-root.json")
    _write_json(configured, document)

    loaded = load_exact_campaign_configuration(configured, profile="recommended", project_root=root)

    expected = root / "artifacts" / "campaign-evidence"
    assert Path(str(loaded["campaign_artifact_root"])) == expected.resolve()
    assert expected.resolve().is_relative_to(root)


@pytest.mark.parametrize(
    "artifact_root", ["../campaign-evidence/", "../training/../campaign-evidence", "../../../outside"]
)
def test_product_profile_refuses_unsafe_campaign_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_root: str,
) -> None:
    root, campaign_config, _campaign = _portable_campaign_project(tmp_path, monkeypatch)
    document = json.loads(campaign_config.read_text(encoding="utf-8"))
    document["product_profiles"]["recommended"]["campaign"]["campaign_artifact_root"] = artifact_root
    hostile_config = campaign_config.with_name("unsafe-campaign-artifact-root.json")
    _write_json(hostile_config, document)

    message = "canonical relative path" if artifact_root != "../../../outside" else "escapes the approved project"
    with pytest.raises(CampaignValidationError, match=message):
        validate_training_launch_plan(
            hostile_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )


@pytest.mark.parametrize(
    "field,value", [("execute_confirmed", 1), ("execute_confirmed", "true"), ("resume", 0), ("resume", "false")]
)
def test_launch_issue_gate_refuses_coercible_boolean_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    first_run = built_campaign["expected_runs"][0]
    arguments: dict[str, Any] = {
        "run_id": str(first_run["run_id"]),
        "compute_backend_id": "local",
        "project_root": root,
        "execute_confirmed": True,
        "campaign_profile": "recommended",
        "resume": False,
    }
    arguments[field] = value

    with pytest.raises(CampaignValidationError, match=r"explicit execution confirmation|resume flag"):
        prepare_validated_training_launch(campaign_config, **arguments)


def test_launch_verification_refuses_a_coercible_context_resume_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, campaign_config, built_campaign = _portable_campaign_project(tmp_path, monkeypatch)
    first_run = built_campaign["expected_runs"][0]
    prepared = prepare_validated_training_launch(
        campaign_config,
        run_id=str(first_run["run_id"]),
        compute_backend_id="local",
        project_root=root,
        execute_confirmed=True,
        campaign_profile="recommended",
    )
    forged_context = replace(prepared.validator_context, resume=0)

    with pytest.raises(CampaignValidationError, match="resume flag must be a strict boolean"):
        verify_validated_training_launch(
            prepared.receipt,
            forged_context,
            compute_backend_id="local",
            argv=prepared.argv,
            environment=prepared.environment,
            output_root=prepared.output_root,
            campaign_identity=str(built_campaign["campaign_identity"]),
            run_identity=str(first_run["run_identity"]),
        )


@pytest.mark.parametrize("reducible", ["nested/./campaign.json", "nested//campaign.json", "nested/campaign.json/"])
def test_product_profile_refuses_noncanonical_campaign_path_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reducible: str,
) -> None:
    root = tmp_path / "project"
    config_directory = root / "artifacts" / "training"
    nested = config_directory / "nested" / "campaign.json"
    _write_json(nested, {"schema_version": CAMPAIGN_SCHEMA_VERSION})
    hostile_config = config_directory / "campaigns.json"
    _write_json(
        hostile_config,
        {"product_profiles": {"recommended": {"campaign_path": reducible}}},
    )
    nested_reads: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.resolve() == nested.resolve():
            nested_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    with pytest.raises(CampaignValidationError, match="campaign_path must be a canonical relative path"):
        validate_training_launch_plan(
            hostile_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert nested_reads == []


def test_product_profile_refuses_an_in_tree_symlink_seam_before_nested_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    config_directory = root / "artifacts" / "training"
    real_directory = config_directory / "real"
    real_campaign = real_directory / "campaign.json"
    _write_json(real_campaign, {"campaign_id": "must-not-be-read"})
    linked_directory = config_directory / "linked"
    try:
        os.symlink(real_directory, linked_directory, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable in this test session")
    campaign_config = config_directory / "campaigns.json"
    _write_json(
        campaign_config,
        {"product_profiles": {"recommended": {"campaign_path": "linked/campaign.json"}}},
    )
    nested_reads: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.resolve() == real_campaign.resolve():
            nested_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    with pytest.raises(CampaignValidationError, match=r"link/reparse seam"):
        validate_training_launch_plan(
            campaign_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert nested_reads == []
    assert real_campaign.read_text(encoding="utf-8")
    assert nested_reads == [real_campaign]


def test_product_profile_refuses_a_hardlinked_campaign_before_nested_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    config_directory = root / "artifacts" / "training"
    owned_campaign = config_directory / "owned-campaign.json"
    _write_json(owned_campaign, {"campaign_id": "must-not-be-read"})
    hardlinked_campaign = config_directory / "hardlinked-campaign.json"
    try:
        os.link(owned_campaign, hardlinked_campaign)
    except (NotImplementedError, OSError):
        pytest.skip("hard links are unavailable in this test session")
    campaign_config = config_directory / "campaigns.json"
    _write_json(
        campaign_config,
        {"product_profiles": {"recommended": {"campaign_path": hardlinked_campaign.name}}},
    )
    nested_reads: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.resolve() == hardlinked_campaign.resolve():
            nested_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    with pytest.raises(CampaignValidationError, match="regular single-link file"):
        validate_training_launch_plan(
            campaign_config,
            compute_backend_id="local",
            project_root=root,
            campaign_profile="recommended",
        )

    assert nested_reads == []
    assert hardlinked_campaign.read_text(encoding="utf-8")
    assert nested_reads == [hardlinked_campaign]


@pytest.mark.parametrize(
    "suffix,payload",
    [
        (
            ".json",
            '{"product_profiles":{"recommended":{"campaign":'
            '{"output_root":"runs/first","output_root":"runs/second"}}}}',
        ),
        (
            ".yaml",
            "product_profiles:\n"
            "  recommended:\n"
            "    campaign:\n"
            "      output_root: runs/first\n"
            "      output_root: runs/second\n",
        ),
    ],
)
def test_launch_loader_refuses_nested_duplicate_keys_without_disclosing_the_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    payload: str,
) -> None:
    campaign_config = tmp_path / f"private-campaign{suffix}"
    campaign_config.write_text(payload, encoding="utf-8")
    planning_calls: list[str] = []

    def unexpected_planning(*_args: Any, **_kwargs: Any) -> Any:
        planning_calls.append("called")
        raise AssertionError("planning must not interpret a campaign with duplicate keys")

    monkeypatch.setattr(launch_module, "plan_campaign", unexpected_planning)

    with pytest.raises(CampaignValidationError) as captured:
        validate_training_launch_plan(
            campaign_config,
            compute_backend_id="local",
            project_root=tmp_path,
            campaign_profile="recommended",
        )

    assert str(captured.value) == "campaign configuration contains an ambiguous mapping"
    assert str(campaign_config) not in str(captured.value)
    assert planning_calls == []

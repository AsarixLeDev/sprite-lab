from __future__ import annotations

import json
from pathlib import Path

from spritelab.remote_compute import ComputeJobRequest
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, plan_campaign, stable_hash
from spritelab.training.launch import ValidatedTrainingLaunch, prepare_validated_training_launch


class _StaticLaunchAuthorizationVerifier:
    def __init__(self, evidence_sha256: str) -> None:
        self.launch_authorization_evidence_sha256 = evidence_sha256

    def verify_unchanged(self) -> None:
        return None


def launch_authorization_verifier(launch: ValidatedTrainingLaunch) -> _StaticLaunchAuthorizationVerifier:
    return _StaticLaunchAuthorizationVerifier(launch.receipt.launch_authorization_evidence_sha256)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def validated_launch(tmp_path: Path, backend_id: str = "fake") -> ValidatedTrainingLaunch:
    inputs = tmp_path / "receipt-inputs"
    dataset, split, vocabulary, benchmark = [
        inputs / name for name in ("dataset.json", "split.json", "vocab.json", "benchmark.json")
    ]
    for path, value in (
        (dataset, {"records": ["sprite"]}),
        (split, {"split": "train", "sprite_id": "sprite", "npz_file": "train.npz", "npz_row": 0}),
        (vocabulary, {"tokens": ["sprite"]}),
        (benchmark, {"prompts": ["sprite"]}),
    ):
        if path == split:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        else:
            _write_json(path, value)
    (inputs / "train.npz").write_bytes(b"synthetic retained dataset artifact")
    optimizer = {"name": "adamw"}
    schedule = {"name": "cosine"}
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
        "campaign_id": "validated_adapter_test",
        "purpose": "Synthetic receipt validation without real execution.",
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
        "output_root": str(tmp_path / "validated-runs"),
        "executable": True,
        "launch_authorized": True,
    }
    campaign = plan_campaign(spec, execution_root=tmp_path)
    for run in campaign["expected_runs"]:
        _write_json(Path(run["resolved_config_path"]), run["resolved_config"])
    config_path = tmp_path / "validated-campaign.json"
    _write_json(config_path, campaign)
    return prepare_validated_training_launch(
        config_path,
        run_id=campaign["expected_runs"][0]["run_id"],
        compute_backend_id=backend_id,
        project_root=tmp_path,
        execute_confirmed=True,
    )


def compute_request(tmp_path: Path, backend_id: str = "fake") -> ComputeJobRequest:
    launch = validated_launch(tmp_path, backend_id)
    return ComputeJobRequest(
        run_id=str(launch.run["run_id"]),
        command=launch.argv,
        idempotency_key=str(launch.run["run_id"]),
        campaign_identity=launch.receipt.campaign_identity_sha256,
        run_identity=launch.receipt.run_identity,
        local_project_root=tmp_path,
        output_root=launch.output_root,
        environment=launch.environment,
        execution_spec_identity=launch.receipt.execution_spec_sha256,
        output_root_identity=launch.receipt.output_root_identity,
        launch_authorization_evidence_sha256=launch.receipt.launch_authorization_evidence_sha256,
        compute_backend_id=backend_id,
        launch_receipt=launch.receipt,
        validator_context=launch.validator_context,
        launch_authorization_verifier=launch_authorization_verifier(launch),
    )

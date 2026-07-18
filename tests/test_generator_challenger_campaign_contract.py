from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.product_core import ProductEvent, ProductStatus
from spritelab.product_web.events import EVENT_FILENAME, EventRepository, LegacyEventMigrationError
from spritelab.training.campaign import PER_RUN_ARTIFACTS, audit_resume
from spritelab.training.experiment_system import stable_hash
from spritelab.training.generator_challenger import (
    CAMPAIGN_RUN_CONTRACT_SCHEMA_VERSION,
    ChallengerTrainConfig,
    _CampaignRunWriter,
    run_challenger_training,
)
from spritelab.utils.safe_fs import UnsafeFilesystemOperation


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "dataset", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _campaign_contract(tmp_path: Path, *, seed: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_id = "generator_writer_integration"
    run_id = f"base-seed-{seed}"
    output_root = tmp_path / "runs" / "base" / run_id
    campaign_identity = stable_hash({"campaign_id": campaign_id, "purpose": "writer integration"})
    resolved_config = {
        "training": {"max_optimizer_steps": 2},
        "schedule": {"name": "none"},
        "evaluation": {"ema_policy": "both"},
    }
    resolved_config_sha256 = stable_hash(resolved_config)
    execution_contract_sha256 = stable_hash(
        {"resolved_config_sha256": resolved_config_sha256, "output_root": str(output_root)}
    )
    run_identity = stable_hash(
        {
            "campaign_identity": campaign_identity,
            "run_id": run_id,
            "seed": seed,
            "resolved_config_sha256": resolved_config_sha256,
            "execution_contract_sha256": execution_contract_sha256,
        }
    )
    run = {
        "run_id": run_id,
        "run_identity": run_identity,
        "seed": seed,
        "output_root": str(output_root),
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "execution_contract_sha256": execution_contract_sha256,
        "expected_checkpoint_steps": [1, 2],
        "expected_evaluation_steps": [1, 2],
    }
    campaign = {
        "campaign_id": campaign_id,
        "campaign_identity": campaign_identity,
        "training": {"max_optimizer_steps": 2},
        "schedule": {"name": "none"},
        "evaluation": {"ema_policy": "both"},
        "expected_runs": [run],
    }
    contract = {
        "schema_version": CAMPAIGN_RUN_CONTRACT_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_identity": campaign_identity,
        "run_id": run_id,
        "run_identity": run_identity,
        "seed": seed,
        "output_root": str(output_root),
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "execution_contract_sha256": execution_contract_sha256,
        "expected_checkpoint_steps": [1, 2],
        "expected_evaluation_steps": [1, 2],
        "max_optimizer_steps": 2,
        "schedule_name": "none",
        "evaluation_ema_policy": "both",
        "training_code_identity_sha256": "a" * 64,
    }
    return campaign, contract


def _move_aside(path: Path, retained: Path) -> None:
    assert not retained.exists()
    path.rename(retained)


def _restore_from(path: Path, retained: Path, adversarial: Path) -> None:
    assert not adversarial.exists()
    path.rename(adversarial)
    retained.rename(path)


def test_campaign_training_is_resumable_then_commits_complete_artifacts_last(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    campaign, contract = _campaign_contract(tmp_path)
    run = campaign["expected_runs"][0]
    root = Path(run["output_root"])
    sentinel = tmp_path / "outside-sentinel.txt"
    sentinel.write_bytes(b"outside remains unchanged")
    initial = ChallengerTrainConfig(
        dataset_dir=dataset,
        training_manifest=manifest,
        out_dir=root,
        batch_size=2,
        max_steps=2,
        device="cpu",
        seed=run["seed"],
        base_channels=8,
        channel_mults="1,2",
        res_blocks_per_level=1,
        embed_dim=8,
        sample_every=0,
        save_every=0,
        validation_mode="auto",
        eval_max_batches=1,
        ema_decay=0.9,
        auxiliary_heads_mode="absent",
        experiment_manifest={},
        campaign_run_contract=contract,
        stop_after_step=1,
    )

    partial_report = run_challenger_training(initial)
    assert partial_report["steps_completed"] == 1
    resume_audit = audit_resume(campaign)
    assert resume_audit["root_state"] == "valid_resumable"
    assert resume_audit["safe"] is True
    assert not (root / "run_completion_marker.json").exists()

    sidecar = root / "checkpoint_step_000001.json"
    retained_sidecar = tmp_path / "retained-sidecar.json"
    _move_aside(sidecar, retained_sidecar)
    assert audit_resume(campaign)["root_state"] == "partial_invalid"
    retained_sidecar.rename(sidecar)

    _move_aside(sidecar, retained_sidecar)
    tampered_sidecar = json.loads(retained_sidecar.read_text(encoding="utf-8"))
    tampered_sidecar["run_identity"] = "f" * 64
    sidecar.write_text(json.dumps(tampered_sidecar), encoding="utf-8")
    assert audit_resume(campaign)["root_state"] == "partial_invalid"
    _restore_from(sidecar, retained_sidecar, tmp_path / "tampered-sidecar.json")

    resume_entry = resume_audit["runs"][0]
    resume_config = replace(
        initial,
        stop_after_step=None,
        resume_from=Path(resume_entry["checkpoint"]),
        expected_resume_sha256=resume_entry["checkpoint_content_sha256"],
    )
    hostile_link = root / "foreign-hardlink.bin"
    os.link(sentinel, hostile_link)
    with pytest.raises(UnsafeFilesystemOperation, match="unsafe entry"):
        run_challenger_training(resume_config)
    assert sentinel.read_bytes() == b"outside remains unchanged"
    hostile_link.rename(tmp_path / "retained-foreign-hardlink.bin")

    completed_report = run_challenger_training(resume_config)
    assert completed_report["steps_completed"] == 2
    assert audit_resume(campaign)["root_state"] == "complete"
    assert sorted(path.name for path in root.glob("checkpoint*.pt")) == [
        "checkpoint_step_000001.pt",
        "checkpoint_step_000002.pt",
    ]
    assert (root / "samples_final.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all(
        (root / name).stat().st_nlink == 1
        for name in ("config.json", "samples_final.png", "train_metrics.jsonl", "train_report.json", "vocab.json")
    )
    assert all((root / f"{name}.json").stat().st_nlink == 1 for name in PER_RUN_ARTIFACTS)

    terminal_before = {
        name: (root / name).read_bytes() for name in (EVENT_FILENAME, "run_identity.json", "run_completion_marker.json")
    }
    with pytest.raises(LegacyEventMigrationError, match="complet"):
        EventRepository(root.parent).append(
            ProductEvent(
                run_id=run["run_id"],
                timestamp="2026-07-18T12:00:00+00:00",
                feature="training",
                stage="complete",
                event_type="late_progress",
                status=ProductStatus.RUNNING,
                current=2,
                total=2,
                message="This late append must not reopen a completed campaign run.",
            )
        )
    assert {
        name: (root / name).read_bytes() for name in (EVENT_FILENAME, "run_identity.json", "run_completion_marker.json")
    } == terminal_before

    marker = root / "run_completion_marker.json"
    retained_marker = tmp_path / "retained-marker.json"
    _move_aside(marker, retained_marker)
    assert audit_resume(campaign)["root_state"] == "valid_resumable"
    retained_marker.rename(marker)

    _move_aside(marker, retained_marker)
    tampered_marker = json.loads(retained_marker.read_text(encoding="utf-8"))
    tampered_marker["run_identity"] = "f" * 64
    marker.write_text(json.dumps(tampered_marker), encoding="utf-8")
    assert audit_resume(campaign)["root_state"] == "contradictory"
    _restore_from(marker, retained_marker, tmp_path / "tampered-marker.json")

    _move_aside(marker, retained_marker)
    marker.write_bytes(b"{")
    assert audit_resume(campaign)["root_state"] == "corrupt"
    _restore_from(marker, retained_marker, tmp_path / "partial-marker.json")
    assert audit_resume(campaign)["root_state"] == "complete"
    assert sentinel.read_bytes() == b"outside remains unchanged"


def test_campaign_resume_requires_exact_hash_before_output_mutation(tmp_path: Path) -> None:
    campaign, contract = _campaign_contract(tmp_path)
    root = Path(campaign["expected_runs"][0]["output_root"])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not opened")
    config = ChallengerTrainConfig(
        dataset_dir=tmp_path / "dataset",
        training_manifest=tmp_path / "manifest.jsonl",
        out_dir=root,
        max_steps=2,
        seed=campaign["expected_runs"][0]["seed"],
        resume_from=checkpoint,
        campaign_run_contract=contract,
    )

    with pytest.raises(ValueError, match="exact retained checkpoint SHA-256"):
        run_challenger_training(config)
    assert not root.exists()


def test_retained_manifest_and_dataset_descriptors_are_the_only_training_reads(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    manifest_bytes = manifest.read_bytes()
    retained_records = tuple(json.loads(line) for line in manifest_bytes.splitlines() if line.strip())
    npz_names = sorted({str(row.get("npz_file") or f"{row['split']}.npz") for row in retained_records})
    descriptors = {name: os.open(dataset / name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0))) for name in npz_names}
    content_sha256 = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in npz_names}
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    output_root = tmp_path / "retained-input-run"
    try:
        report = run_challenger_training(
            ChallengerTrainConfig(
                dataset_dir=tmp_path / "lexical-dataset-path-is-unavailable",
                training_manifest=tmp_path / "lexical-manifest-is-unavailable.jsonl",
                out_dir=output_root,
                batch_size=2,
                max_steps=1,
                device="cpu",
                seed=13,
                base_channels=8,
                channel_mults="1,2",
                res_blocks_per_level=1,
                embed_dim=8,
                sample_every=0,
                save_every=0,
                validation_mode="none",
                auxiliary_heads_mode="absent",
                experiment_manifest={
                    "dataset_manifest_hash": manifest_sha256,
                    "split_manifest_hash": manifest_sha256,
                },
                retained_training_manifest_records=retained_records,
                retained_dataset_descriptors=descriptors,
                retained_dataset_content_sha256=content_sha256,
            )
        )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    assert report["steps_completed"] == 1
    persisted = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    assert not {
        "retained_training_manifest_records",
        "retained_dataset_descriptors",
        "retained_dataset_content_sha256",
    }.intersection(persisted)


def test_partial_direct_final_completion_marker_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spritelab.training import generator_challenger as generator_module

    root = tmp_path / "run"
    root.mkdir()
    sentinel = tmp_path / "outside-sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    marker = {
        "campaign_identity": "a" * 64,
        "run_identity": "b" * 64,
        "seed": 7,
        "complete": True,
    }

    def interrupted_write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:7])
        raise OSError("simulated interruption")

    with _CampaignRunWriter(root, root) as writer:
        monkeypatch.setattr(generator_module, "_write_descriptor_all", interrupted_write)
        with pytest.raises(OSError, match="simulated interruption"):
            writer.write_json_idempotent("run_completion_marker.json", marker)
    assert (root / "run_completion_marker.json").read_bytes() != (
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    monkeypatch.undo()
    with _CampaignRunWriter(root, root) as writer:
        with pytest.raises(UnsafeFilesystemOperation, match="conflicts with retained bytes"):
            writer.write_json_idempotent("run_completion_marker.json", marker)
    assert sentinel.read_bytes() == b"unchanged"


def test_campaign_atomic_publication_replaces_post_scan_hardlink_without_touching_outside(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside-config.json"
    outside.write_bytes(b"outside-preserved")

    with _CampaignRunWriter(root, root) as writer:
        try:
            os.link(outside, root / "config.json")
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"hard links are unavailable for this filesystem: {exc}")
        writer.write_bytes_atomic("config.json", b'{"safe":true}\n')

    assert outside.read_bytes() == b"outside-preserved"
    assert (root / "config.json").read_bytes() == b'{"safe":true}\n'
    assert (root / "config.json").stat().st_nlink == 1


def test_campaign_metrics_append_keeps_one_descriptor_across_rename_recreate_attack(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    metrics_path = root / "train_metrics.jsonl"
    displaced = root / "displaced-metrics.jsonl"
    outside = tmp_path / "outside-metrics.jsonl"
    outside.write_bytes(b"outside-preserved\n")

    with _CampaignRunWriter(root, root) as writer:
        metrics = writer.open_jsonl_append("train_metrics.jsonl", initialize=True)
        try:
            os.replace(metrics_path, displaced)
        except OSError:
            metrics.write_record({"step": 1})
            metrics.close()
            assert metrics_path.read_text(encoding="utf-8") == '{"step": 1}\n'
        else:
            try:
                os.link(outside, metrics_path)
            except (NotImplementedError, OSError) as exc:
                with pytest.raises(UnsafeFilesystemOperation):
                    metrics.close()
                pytest.skip(f"hard links are unavailable for this filesystem: {exc}")
            with pytest.raises(UnsafeFilesystemOperation, match="identity changed"):
                metrics.write_record({"step": 1})
            with pytest.raises(UnsafeFilesystemOperation, match="identity changed"):
                metrics.close()

    assert outside.read_bytes() == b"outside-preserved\n"

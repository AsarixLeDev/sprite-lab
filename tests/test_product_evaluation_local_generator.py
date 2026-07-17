from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from spritelab.product_features.evaluation.local_generator import (
    LocalCheckpointPlaygroundGenerator,
    LocalPlaygroundGenerationError,
)


def _checkpoint(root: Path) -> Path:
    import torch

    path = root / "runs" / "train-smoke" / "checkpoint_last.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_type": "generator_challenger",
            "ema_weights": False,
            "step": 7,
            "global_step": 7,
        },
        path,
    )
    return path


def _sampler(config):
    indexed = config.out_dir / "indexed_png"
    indexed.mkdir()
    rows = []
    prompt_rows = [json.loads(line) for line in config.prompts.read_text(encoding="utf-8").splitlines()]
    for index in range(config.max_samples):
        relative = f"indexed_png/sample_{index:04d}.png"
        Image.new("RGBA", (32, 32), (index, 20, 30, 255)).save(config.out_dir / relative)
        rows.append(
            {
                **prompt_rows[index],
                "sample_id": f"sample_{index:06d}",
                "seed": config.seed,
                "noise_seed": config.noise_seed + index,
                "model_type": "generator_challenger",
                "steps": config.steps,
                "cfg_scale": config.cfg_scale,
                "paths": {"indexed_png": relative},
            }
        )
    (config.out_dir / "generated_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"sample_count": len(rows)}


def _manifest_row(config, *, index: int, relative: str) -> dict[str, object]:
    prompt = json.loads(config.prompts.read_text(encoding="utf-8").splitlines()[index])
    return {
        **prompt,
        "sample_id": f"sample_{index:06d}",
        "seed": config.seed,
        "noise_seed": config.noise_seed + index,
        "model_type": "generator_challenger",
        "steps": config.steps,
        "cfg_scale": config.cfg_scale,
        "paths": {"indexed_png": relative},
    }


def _invoke(generator: LocalCheckpointPlaygroundGenerator, checkpoint: Path, **overrides):
    values = {
        "checkpoint": checkpoint,
        "prompt": "small shield",
        "seed": 1,
        "sampling_steps": 2,
        "guidance": 2.0,
        "image_count": 1,
        "weights": "live",
        "expected_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "expected_step": 7,
        "expected_variant": "live",
    }
    values.update(overrides)
    return generator.generate(**values)


def test_local_generator_is_passive_until_generate_and_returns_exact_pngs(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    assert generator.remote is False
    assert generator.billable is False
    assert not work_root.exists()

    checkpoint = _checkpoint(tmp_path)
    assets = generator.generate(
        checkpoint=checkpoint,
        prompt="small blue potion",
        seed=71,
        sampling_steps=2,
        guidance=2.5,
        image_count=2,
        weights="live",
        expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        expected_step=7,
        expected_variant="live",
    )

    assert len(assets) == 2
    assert all(asset.media_type == "image/png" for asset in assets)
    assert all(asset.content.startswith(b"\x89PNG\r\n\x1a\n") for asset in assets)
    invocations = [path for path in work_root.iterdir() if path.is_dir()]
    assert len(invocations) == 1
    prompt_rows = [
        json.loads(line) for line in (invocations[0] / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["prompt"] for row in prompt_rows] == ["small blue potion", "small blue potion"]
    assert {row["scope"] for row in prompt_rows} == {"EXPLORATORY"}


def test_local_generator_rejects_checkpoint_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-checkpoint.pt"
    outside.write_bytes(b"outside")
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=_sampler,
    )
    with pytest.raises(ValueError, match="escapes"):
        generator.generate(
            checkpoint=outside,
            prompt="small shield",
            seed=1,
            sampling_steps=2,
            guidance=2.0,
            image_count=1,
            weights="ema",
            expected_sha256=sha256(outside.read_bytes()).hexdigest(),
            expected_step=7,
            expected_variant="ema",
        )
    assert outside.read_bytes() == b"outside"


def test_local_generator_rejects_manifest_traversal_without_reading_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.png"
    sentinel.write_bytes(b"outside-sentinel")

    def malicious_sampler(config):
        (config.out_dir / "generated_manifest.jsonl").write_text(
            json.dumps(_manifest_row(config, index=0, relative="../../sentinel.png")) + "\n",
            encoding="utf-8",
        )
        return {"sample_count": 1}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=malicious_sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="escapes"):
        generator.generate(
            checkpoint=checkpoint,
            prompt="small shield",
            seed=1,
            sampling_steps=2,
            guidance=2.0,
            image_count=1,
            weights="live",
            expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
            expected_step=7,
            expected_variant="live",
        )
    assert sentinel.read_bytes() == b"outside-sentinel"


def test_local_generator_rejects_count_and_image_contract_mismatches(tmp_path: Path) -> None:
    def wrong_image_sampler(config):
        indexed = config.out_dir / "indexed_png"
        indexed.mkdir()
        Image.new("RGBA", (16, 16), (0, 0, 0, 255)).save(indexed / "wrong.png")
        (config.out_dir / "generated_manifest.jsonl").write_text(
            json.dumps(_manifest_row(config, index=0, relative="indexed_png/wrong.png")) + "\n",
            encoding="utf-8",
        )
        return {"sample_count": 1}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=wrong_image_sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="32x32"):
        generator.generate(
            checkpoint=checkpoint,
            prompt="small shield",
            seed=1,
            sampling_steps=2,
            guidance=2.0,
            image_count=1,
            weights="live",
            expected_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
            expected_step=7,
            expected_variant="live",
        )


def test_local_generator_rejects_catalog_hash_mismatch_before_sampler(tmp_path: Path) -> None:
    called = False

    def sampler(config):
        nonlocal called
        called = True
        return _sampler(config)

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(LocalPlaygroundGenerationError, match="durable catalog hash"):
        _invoke(generator, checkpoint, expected_sha256="0" * 64)
    assert called is False


def test_local_generator_rejects_manifest_prompt_semantics(tmp_path: Path) -> None:
    def sampler(config):
        report = _sampler(config)
        manifest = config.out_dir / "generated_manifest.jsonl"
        row = json.loads(manifest.read_text(encoding="utf-8"))
        row["prompt"] = "different prompt"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return report

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="semantics"):
        _invoke(generator, _checkpoint(tmp_path))


def test_local_generator_serializes_active_process_lease(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    work_root.mkdir(parents=True)
    (work_root / "sampler-lease.json").write_text(
        json.dumps(
            {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": "active-owner",
                "status": "ACTIVE",
                "owner_pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="already active"):
        _invoke(generator, _checkpoint(tmp_path))


def test_local_generator_recovers_dead_owner_as_retryable_orphan(tmp_path: Path) -> None:
    work_root = tmp_path / "runs" / "playground-sampler-work"
    work_root.mkdir(parents=True)
    (work_root / "sampler-lease.json").write_text(
        json.dumps(
            {
                "schema_version": "spritelab.playground-sampler-lease.v1",
                "lease_id": "dead-owner",
                "status": "ACTIVE",
                "owner_pid": 2_147_483_647,
            }
        ),
        encoding="utf-8",
    )
    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=work_root,
        sampler=_sampler,
    )
    assert len(_invoke(generator, _checkpoint(tmp_path))) == 1
    lease = json.loads((work_root / "sampler-lease.json").read_text(encoding="utf-8"))
    assert lease["status"] == "COMPLETE"
    assert lease["recovered_orphan"] == {
        "lease_id": "dead-owner",
        "retryable": True,
        "status": "ORPHANED",
    }


def test_local_generator_rejects_unicode_normalized_output_collision(tmp_path: Path) -> None:
    def sampler(config):
        indexed = config.out_dir / "indexed_png"
        indexed.mkdir()
        relatives = ("indexed_png/A.png", "indexed_png/\uff21.png")
        rows = []
        for index, relative in enumerate(relatives):
            Image.new("RGBA", (32, 32), (index, 2, 3, 255)).save(config.out_dir / relative)
            rows.append(_manifest_row(config, index=index, relative=relative))
        (config.out_dir / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {"sample_count": 2}

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=sampler,
    )
    with pytest.raises(LocalPlaygroundGenerationError, match="colliding"):
        _invoke(generator, _checkpoint(tmp_path), image_count=2)


def test_local_generator_binds_actual_sampler_load_to_original_snapshot_hash(tmp_path: Path) -> None:
    def swapping_sampler(config):
        import torch

        from spritelab.training.generator_challenger import run_sample_generator_challenger

        replacement = config.checkpoint.with_name("replacement-step-999.pt")
        torch.save(
            {
                "model_type": "generator_challenger",
                "ema_weights": False,
                "step": 999,
                "global_step": 999,
            },
            replacement,
        )
        os.replace(replacement, config.checkpoint)
        return run_sample_generator_challenger(config)

    generator = LocalCheckpointPlaygroundGenerator(
        project_root=tmp_path,
        work_root=tmp_path / "runs" / "playground-sampler-work",
        sampler=swapping_sampler,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        _invoke(generator, _checkpoint(tmp_path))

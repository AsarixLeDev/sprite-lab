from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset

from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training import generator_challenger, v1_gallery
from spritelab.training.cli import main as train_cli
from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training
from spritelab.training.sample_generator import read_prompt_records
from spritelab.training.v1_gallery import (
    BuildV1GalleryConfig,
    build_default_v1_gallery_prompts,
    build_v1_gallery_demo,
)


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _tiny_challenger_checkpoint(tmp_path: Path) -> Path:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "tiny_challenger_run"
    run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=run_dir,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=7,
            base_channels=8,
            channel_mults="1,2",
            res_blocks_per_level=1,
            embed_dim=8,
            sample_every=0,
            save_every=0,
            validation_mode="none",
        )
    )
    return run_dir / "checkpoint_last_ema.pt"


def _fake_run_sample_generator_challenger(config: Any) -> dict[str, Any]:
    """CPU/torch-free stand-in for run_sample_generator_challenger used by report/metadata tests."""

    from PIL import Image

    prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, prompt in enumerate(prompts):
        sample_id = f"sample_{index:06d}"
        image = Image.new("RGBA", (32, 32), (200, 60, 60, 255))
        for key in ("raw_rgba", "hard_rgba", "indexed_png"):
            sub = out_dir / key
            sub.mkdir(parents=True, exist_ok=True)
            image.save(sub / f"{sample_id}.png")
        records.append(
            {
                **prompt,
                "sample_id": sample_id,
                "checkpoint": str(config.checkpoint),
                "max_colors": int(config.max_colors),
                "alpha_threshold": float(config.alpha_threshold),
                "visible_color_count": 1,
                "alpha_opaque_count": 1024,
                "paths": {
                    "raw_rgba": f"raw_rgba/{sample_id}.png",
                    "hard_rgba": f"hard_rgba/{sample_id}.png",
                    "indexed_png": f"indexed_png/{sample_id}.png",
                },
                "warnings": [],
            }
        )
    (out_dir / "generated_manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (out_dir / "generation_report.json").write_text(
        json.dumps(
            {
                "sample_count": len(records),
                "contact_sheet": None,
                "config": {
                    "checkpoint": str(config.checkpoint),
                    "checkpoint_resolved": str(config.checkpoint),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if config.project_palette:
        (out_dir / "palette_projection_report.json").write_text(
            json.dumps(
                {
                    "sample_count": len(records),
                    "median_visible_color_count_before": 32,
                    "median_visible_color_count_after": 12,
                    "mean_visible_color_count_before": 30.0,
                    "mean_visible_color_count_after": 11.0,
                    "mean_rgb_mae_visible": 0.0206,
                    "destructive_rate": 0.0,
                    "safe_count": len(records),
                    "moderate_count": 0,
                    "destructive_count": 0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "sample_count": len(records),
        "config": {"checkpoint_resolved": str(config.checkpoint)},
    }


def test_default_prompt_set_is_deterministic_and_in_range() -> None:
    first = build_default_v1_gallery_prompts()
    second = build_default_v1_gallery_prompts()
    assert first == second
    assert 48 <= len(first) <= 96

    categories = {row["category"] for row in first}
    assert {"weapon", "armor", "item_icon", "tool", "material", "effect_icon", "plant"} <= categories


def test_default_prompts_are_compatible_with_sampler(tmp_path: Path) -> None:
    rows = build_default_v1_gallery_prompts()
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    records = read_prompt_records(prompts_path)
    assert len(records) == len(rows)
    prompt_ids = set()
    for record in records:
        assert str(record["prompt"]).strip()
        assert str(record["prompt_id"]).strip()
        prompt_ids.add(record["prompt_id"])
    assert len(prompt_ids) == len(records)


def test_build_v1_gallery_creates_expected_output_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    def _fail_if_trained(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("build_v1_gallery_demo must never invoke training")

    monkeypatch.setattr(generator_challenger, "run_challenger_training", _fail_if_trained)

    out_dir = tmp_path / "gallery"
    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=out_dir,
            checkpoint=tmp_path / "checkpoint_last_ema.pt",
            device="cpu",
            seed=123,
            batch_size=8,
            num_samples=6,
        )
    )

    assert (out_dir / "v1_gallery_prompts.jsonl").is_file()
    assert (out_dir / "v1_gallery_report.json").is_file()
    assert (out_dir / "v1_gallery_report.md").is_file()
    assert (out_dir / "samples" / "generated_manifest.jsonl").is_file()
    assert (out_dir / "contact_sheets").is_dir()
    assert any((out_dir / "contact_sheets").glob("*.png"))
    assert report["sample_count"] == 6


def test_report_json_contains_v1_preset_and_projection_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    out_dir = tmp_path / "gallery"
    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=out_dir,
            checkpoint=tmp_path / "checkpoint_last_ema.pt",
            device="cpu",
            num_samples=4,
        )
    )

    assert report["preset"]["name"] == "v1"
    assert report["preset"]["cfg_scale"] == pytest.approx(3.0)
    assert report["preset"]["steps"] == 30
    assert report["preset"]["projection_method"] == "deterministic_kmeans"
    assert report["preset"]["projection_target_colors"] == 16
    assert report["preset"]["projection_min_pixel_share"] == pytest.approx(0.01)

    projection = report["projection_summary"]
    assert projection["median_visible_colors_before"] == 32
    assert projection["median_visible_colors_after"] == 12
    assert projection["mean_rgb_mae_visible"] == pytest.approx(0.0206)
    assert projection["destructive_rate"] == pytest.approx(0.0)

    assert "generated_qa" in report
    assert "generated_review" in report
    assert "Official v1 default" in report["official_statement"]

    on_disk = json.loads((out_dir / "v1_gallery_report.json").read_text(encoding="utf-8"))
    assert on_disk["preset"]["name"] == "v1"
    assert on_disk["preset"]["factored_cfg"] is False
    assert on_disk["validated_v1_1_factored_cfg_reference"] is None


@pytest.mark.parametrize("preset_flag", ["v1.1", "v1_1", "phase1_v1_1"])
def test_report_json_contains_v1_1_preset_and_factored_cfg_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset_flag: str
) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    out_dir = tmp_path / f"gallery_{preset_flag.replace('.', '_')}"
    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=out_dir,
            checkpoint=tmp_path / "checkpoint_last_ema.pt",
            device="cpu",
            num_samples=4,
            export_preset=preset_flag,
        )
    )

    assert report["preset"]["name"] == "v1.1"
    assert report["preset"]["cfg_scale"] == pytest.approx(3.0)
    assert report["preset"]["steps"] == 30
    assert report["preset"]["factored_cfg"] is True
    assert report["preset"]["cfg_base_scale"] == pytest.approx(2.5)
    assert report["preset"]["cfg_color_scale"] == pytest.approx(3.0)

    reference = report["validated_v1_1_factored_cfg_reference"]
    assert reference is not None
    assert reference["deltas_v1_1_minus_v1"]["color_consistency"] == pytest.approx(0.0312)

    markdown = (out_dir / "v1_gallery_report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# v1.1 Gallery Report")
    assert "Factored CFG: base=2.5, color=3.0" in markdown

    on_disk = json.loads((out_dir / "v1_gallery_report.json").read_text(encoding="utf-8"))
    assert on_disk["preset"]["name"] == "v1.1"
    assert on_disk["preset"]["factored_cfg"] is True


def test_v1_gallery_rejects_unknown_export_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    with pytest.raises(ValueError):
        build_v1_gallery_demo(
            BuildV1GalleryConfig(
                out_dir=tmp_path / "gallery_bad_preset",
                checkpoint=tmp_path / "checkpoint_last_ema.pt",
                device="cpu",
                num_samples=2,
                export_preset="not_a_real_preset",
            )
        )


def test_custom_prompt_file_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    custom_prompts = tmp_path / "custom_prompts.jsonl"
    custom_rows = [
        {"prompt_id": "custom_0", "prompt": "blue lantern 32x32 pixel art icon", "category": "item_icon"},
        {"prompt_id": "custom_1", "prompt": "green fern 32x32 pixel art icon", "category": "plant"},
    ]
    custom_prompts.write_text("".join(json.dumps(row) + "\n" for row in custom_rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "gallery"
    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=out_dir,
            checkpoint=tmp_path / "checkpoint_last_ema.pt",
            prompts=custom_prompts,
            device="cpu",
        )
    )

    assert report["prompt_set"]["prompt_count"] == 2
    written = [
        json.loads(line)
        for line in (out_dir / "v1_gallery_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["prompt_id"] for row in written] == ["custom_0", "custom_1"]


def test_no_training_invoked_end_to_end_with_real_checkpoint(tmp_path: Path) -> None:
    checkpoint = _tiny_challenger_checkpoint(tmp_path)
    sibling_checkpoint = checkpoint.parent / "checkpoint_last.pt"
    mtime_before = sibling_checkpoint.stat().st_mtime

    out_dir = tmp_path / "gallery_real"
    report = build_v1_gallery_demo(
        BuildV1GalleryConfig(
            out_dir=out_dir,
            checkpoint=checkpoint,
            device="cpu",
            seed=9,
            batch_size=3,
            num_samples=3,
        )
    )

    assert report["sample_count"] == 3
    assert (out_dir / "samples" / "generated_manifest.jsonl").is_file()
    assert (out_dir / "v1_gallery_report.json").is_file()
    # Training artifacts must be untouched: the gallery build never re-invokes training.
    assert sibling_checkpoint.stat().st_mtime == mtime_before


def test_build_v1_gallery_cli_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    out_dir = tmp_path / "cli_gallery"
    train_cli(
        [
            "build-v1-gallery",
            "--out",
            str(out_dir),
            "--checkpoint",
            str(tmp_path / "checkpoint_last_ema.pt"),
            "--device",
            "cpu",
            "--num-samples",
            "5",
        ]
    )
    assert (out_dir / "v1_gallery_report.json").is_file()
    report = json.loads((out_dir / "v1_gallery_report.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 5
    assert report["preset"]["name"] == "v1"


def test_build_v1_gallery_cli_accepts_export_preset_v1_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_gallery, "run_sample_generator_challenger", _fake_run_sample_generator_challenger)

    out_dir = tmp_path / "cli_gallery_v1_1"
    train_cli(
        [
            "build-v1-gallery",
            "--out",
            str(out_dir),
            "--checkpoint",
            str(tmp_path / "checkpoint_last_ema.pt"),
            "--export-preset",
            "v1.1",
            "--device",
            "cpu",
            "--num-samples",
            "5",
        ]
    )
    assert (out_dir / "v1_gallery_report.json").is_file()
    report = json.loads((out_dir / "v1_gallery_report.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 5
    assert report["preset"]["name"] == "v1.1"
    assert report["preset"]["factored_cfg"] is True
    assert report["preset"]["cfg_base_scale"] == pytest.approx(2.5)
    assert report["preset"]["cfg_color_scale"] == pytest.approx(3.0)


def test_docs_mention_official_v1_default() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "v1_default.md"
    text = docs_path.read_text(encoding="utf-8")
    assert "Official v1 default" in text
    assert "checkpoint_last_ema.pt" in text
    assert "CFG 3.0" in text
    assert "build-v1-gallery" in text

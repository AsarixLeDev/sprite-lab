from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.cli import main as train_cli
from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.sample_generator import SampleGeneratorConfig, run_sample_generator
from spritelab.training.tokenization import SpriteTextTokenizer


def _fake_checkpoint(path: Path) -> Path:
    tokenizer = SpriteTextTokenizer.build(["red potion", "gold sword"], max_length=8)
    model = TinyCaptionSpriteGenerator(
        vocab_size=len(tokenizer),
        embed_dim=8,
        latent_dim=4,
        hidden_channels=8,
        pad_token_id=tokenizer.pad_id,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "vocab": tokenizer.to_json_dict(),
            "checkpoint_type": "caption_rgba_generator_v0",
            "step": 0,
        },
        path,
    )
    return path


def _prompts(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "p0",
                        "prompt": "red potion",
                        "category": "seen_object",
                        "target_semantics": {"base_object": "potion"},
                    }
                ),
                json.dumps({"prompt_id": "p1", "prompt": "gold sword", "category": "seen_object"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_sample_generator_writes_manifest_reports_contact_sheet_and_metadata(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "generated"
    report = run_sample_generator(
        SampleGeneratorConfig(
            checkpoint=ckpt,
            prompts=prompts,
            out_dir=out,
            max_samples=2,
            max_colors=8,
            device="cpu",
            seed=77,
            noise_seed=1000,
            batch_size=2,
        )
    )
    assert report["sample_count"] == 2
    assert (out / "generated_manifest.jsonl").is_file()
    assert (out / "generation_report.json").is_file()
    assert (out / "generation_report.md").is_file()
    assert (out / "generation_contact_sheet.png").is_file()

    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prompt_id"] == "p0"
    assert rows[0]["prompt"] == "red potion"
    assert rows[0]["target_semantics"] == {"base_object": "potion"}
    assert rows[0]["noise_seed"] == 1000
    assert (out / rows[0]["paths"]["raw_rgba"]).is_file()
    assert (out / rows[0]["paths"]["hard_rgba"]).is_file()
    assert (out / rows[0]["paths"]["indexed_png"]).is_file()


def test_sample_generator_cli_runs(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "cli_generated"
    train_cli(
        [
            "sample-generator",
            "--checkpoint",
            str(ckpt),
            "--prompts",
            str(prompts),
            "--out",
            str(out),
            "--max-samples",
            "1",
            "--max-colors",
            "8",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--noise-seed",
            "2000",
        ]
    )
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["noise_seed"] == 2000


def test_sample_generator_projection_writes_projected_outputs_and_metadata(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "generated_projected"

    report = run_sample_generator(
        SampleGeneratorConfig(
            checkpoint=ckpt,
            prompts=prompts,
            out_dir=out,
            max_samples=1,
            max_colors=8,
            project_palette=True,
            project_palette_target_colors=4,
            project_palette_min_pixel_share=0.01,
            device="cpu",
            seed=77,
            noise_seed=1000,
            batch_size=1,
        )
    )

    row = json.loads((out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["palette_projection_applied"] is True
    assert row["palette_projection_method"] == "deterministic_kmeans"
    assert row["palette_projection_target_colors"] == 4
    assert row["palette_projection_min_pixel_share"] == pytest.approx(0.01)
    assert row["visible_color_count_after_projection"] <= 4
    assert row["visible_color_count"] == row["visible_color_count_after_projection"]
    assert row["max_colors"] == 4
    assert row["canonical_max_colors_before_projection"] == 8
    assert row["projection_destructiveness"] in {"safe", "moderate", "destructive"}

    paths = row["paths"]
    assert (out / paths["raw_rgba"]).is_file()
    assert (out / paths["hard_rgba"]).is_file()
    assert (out / paths["pre_projection_indexed_png"]).is_file()
    assert paths["indexed_png"] == paths["projected_png"]
    assert paths["indexed_png"].startswith("projected/")
    assert (out / paths["indexed_png"]).is_file()
    assert (out / "palette_projection_report.json").is_file()
    assert (out / "palette_projection_samples.jsonl").is_file()
    assert (out / "contact_sheet_projected.png").is_file()
    assert report["palette_projection"]["applied"] is True


def test_sample_generator_challenger_v1_preset_expands_expected_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--export-preset",
            "v1",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / "challenger_out"),
            "--device",
            "cpu",
        ]
    )

    assert len(captured) == 1
    config = captured[0]
    assert config.export_preset == "v1"
    assert config.steps == 30
    assert config.cfg_scale == pytest.approx(3.0)
    assert config.max_colors == 32
    assert config.alpha_threshold == pytest.approx(0.5)
    assert config.dither is False
    assert config.write_raw_rgba is True
    assert config.write_hard_rgba is True
    assert config.project_palette is True
    assert config.project_palette_target_colors == 16
    assert config.project_palette_min_pixel_share == pytest.approx(0.01)
    assert config.project_palette_method == "deterministic_kmeans"


def test_sample_generator_challenger_v2_phase0_flags_default_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / "challenger_out"),
            "--device",
            "cpu",
        ]
    )

    config = captured[0]
    assert config.factored_cfg is False
    assert config.cfg_base_scale is None
    assert config.cfg_color_scale is None
    assert config.null_fields == ""


def test_sample_generator_challenger_v2_phase0_flags_parse_from_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / "challenger_out2"),
            "--device",
            "cpu",
            "--factored-cfg",
            "--cfg-base-scale",
            "2.0",
            "--cfg-color-scale",
            "4.5",
            "--null-fields",
            "colors,object_id",
        ]
    )

    config = captured[0]
    assert config.factored_cfg is True
    assert config.cfg_base_scale == pytest.approx(2.0)
    assert config.cfg_color_scale == pytest.approx(4.5)
    assert config.null_fields == "colors,object_id"


@pytest.mark.parametrize("preset_flag", ["v1.1", "v1_1", "phase1_v1_1"])
def test_sample_generator_challenger_v1_1_preset_aliases_enable_factored_cfg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset_flag: str,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / f"challenger_out_{preset_flag.replace('.', '_')}"),
            "--device",
            "cpu",
            "--export-preset",
            preset_flag,
        ]
    )

    config = captured[0]
    assert config.export_preset == preset_flag
    assert config.factored_cfg is True
    assert config.cfg_base_scale == pytest.approx(2.5)
    assert config.cfg_color_scale == pytest.approx(3.0)
    # v1.1 still carries the v1 base defaults.
    assert config.steps == 30
    assert config.cfg_scale == pytest.approx(3.0)
    assert config.project_palette is True
    assert config.project_palette_target_colors == 16


def test_sample_generator_challenger_v1_1_preset_explicit_scales_take_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / "challenger_out_explicit"),
            "--device",
            "cpu",
            "--export-preset",
            "v1.1",
            "--cfg-base-scale",
            "1.0",
        ]
    )

    config = captured[0]
    assert config.factored_cfg is True
    assert config.cfg_base_scale == pytest.approx(1.0)  # explicit flag beats preset default
    assert config.cfg_color_scale == pytest.approx(3.0)  # preset default still applies


def test_sample_generator_challenger_v1_preset_does_not_enable_factored_cfg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger

    prompts = _prompts(tmp_path / "prompts.jsonl")
    captured = []

    def fake_sample(config: object) -> dict[str, object]:
        captured.append(config)
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return {"sample_count": 0, "max_visible_color_count": 0}

    monkeypatch.setattr(generator_challenger, "run_sample_generator_challenger", fake_sample)

    train_cli(
        [
            "sample-generator-challenger",
            "--checkpoint",
            str(tmp_path / "checkpoint_last.pt"),
            "--prompts",
            str(prompts),
            "--out",
            str(tmp_path / "challenger_out_v1_unchanged"),
            "--device",
            "cpu",
            "--export-preset",
            "v1",
        ]
    )

    config = captured[0]
    assert config.factored_cfg is False
    assert config.cfg_base_scale is None
    assert config.cfg_color_scale is None
    assert config.cfg_scale == pytest.approx(3.0)
    assert config.steps == 30

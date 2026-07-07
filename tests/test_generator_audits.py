from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spritelab.training import generator_audits
from spritelab.training.generator_audits import (
    ChallengerGeneratorAuditConfig,
    FullV4ChallengerAuditConfig,
    RegressionGeneratorAuditConfig,
    build_balanced_eval_prompts,
    compare_challenger_conditioning_audits,
    decide_full_v4_challenger_audit,
    default_micro_overfit_steps,
    run_challenger_generator_audit,
    run_full_v4_challenger_audit,
    run_regression_generator_audit,
)
from spritelab.training.overfit_subset import read_sprite_id_list
from spritelab.training.sample_generator import read_prompt_records


def _write_manifest(path: Path, *, count: int = 6) -> list[dict[str, Any]]:
    rows = [
        {
            "sprite_id": f"sprite_{index}",
            "split": "train",
            "category": "weapon" if index % 2 else "potion",
            "caption": f"sprite {index}",
            "object_name": f"Object {index}",
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def _patch_common_audit_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sample(config: Any) -> dict[str, Any]:
        prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, prompt in enumerate(prompts):
            records.append(
                {
                    **prompt,
                    "sample_id": f"sample_{index:06d}",
                    "paths": {"indexed_png": f"sample_{index:06d}.png"},
                    "warnings": [],
                }
            )
        (config.out_dir / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return {"sample_count": len(records)}

    monkeypatch.setattr(generator_audits, "run_sample_generator", fake_sample)
    monkeypatch.setattr(generator_audits, "run_sample_generator_challenger", fake_sample)
    monkeypatch.setattr(
        generator_audits,
        "qa_generated_sprites",
        lambda generated_dir: SimpleNamespace(to_json_dict=lambda: {"ok": True, "errors": []}),
    )
    monkeypatch.setattr(
        generator_audits,
        "review_generated_sprites",
        lambda config: SimpleNamespace(report={"overall": {"total_warnings": 0, "mean_alpha_coverage": 1.0}}),
    )
    def fake_source_match(config: Any) -> dict[str, Any]:
        manifest = config.generated / "generated_manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        return {
            "mean_visible_rgb_mae": 0.0,
            "mean_alpha_iou": 1.0,
            "near_match_rate": 1.0,
            "samples": [{"target_sprite_id": row["target_sprite_id"]} for row in rows],
        }

    monkeypatch.setattr(generator_audits, "run_source_match_review", fake_source_match)
    monkeypatch.setattr(generator_audits, "run_prompt_sensitivity", lambda config: {"warnings": []})
    monkeypatch.setattr(
        generator_audits,
        "run_prompt_faithfulness",
        lambda config: {"repeated_silhouette_rate": 0.0, "color_consistency_rate": 1.0},
    )


def _train_report_from_sprite_id_list(config: Any) -> dict[str, Any]:
    assert config.sprite_id_list is not None
    assert config.max_train_sprites is None
    sprite_ids = read_sprite_id_list(config.sprite_id_list)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "final_train_loss": 0.1,
        "loss_decrease": 0.9,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "steps_completed": config.max_steps,
        "train_records": len(sprite_ids),
        "overfit_subset": {"sprite_ids": sprite_ids},
    }


def test_default_micro_overfit_steps_scale_by_sprite_count() -> None:
    assert default_micro_overfit_steps(16) == 3000
    assert default_micro_overfit_steps(32) == 6000
    assert default_micro_overfit_steps(64) == 12000


def test_challenger_overfit_specs_default_to_normalized_steps() -> None:
    budgets = {
        str(spec["name"]): generator_audits._resolve_micro_overfit_budget(spec)
        for spec in generator_audits.CHALLENGER_SPECS
    }
    assert budgets["overfit_16_sprites"]["steps"] == 3000
    assert budgets["overfit_16_sprites"]["step_count_defaulted_from_helper"] is True
    assert budgets["overfit_32_sprites"]["steps"] == 6000
    assert budgets["overfit_32_sprites"]["step_count_defaulted_from_helper"] is True
    assert budgets["overfit_64_sprites"]["steps"] == 12000
    assert budgets["overfit_64_sprites"]["step_count_defaulted_from_helper"] is True
    assert budgets["full_v4_smoke_2k"]["steps"] == 2000
    assert budgets["full_v4_smoke_2k"]["step_count_defaulted_from_helper"] is False


def test_challenger_has_64_sprite_run_with_normalized_budget() -> None:
    spec = next(
        spec for spec in generator_audits.CHALLENGER_SPECS if spec["name"] == "overfit_64_sprites"
    )
    assert spec["count"] == 64
    assert "steps" not in spec  # relies on the normalized budget helper
    budget = generator_audits._resolve_micro_overfit_budget(spec)
    assert budget["steps"] == 12000
    assert budget["step_count_source"] == "default_micro_overfit_budget"
    assert budget["step_count_defaulted_from_helper"] is True

    train_report = {
        "batch_size": spec["batch_size"],
        "max_steps": budget["steps"],
        "steps_completed": budget["steps"],
        "train_records": 64,
        "effective_train_records": 64,
    }
    metadata = generator_audits._audit_budget_metadata(spec, budget, train_report)
    assert metadata["sprite_count"] == 64
    assert metadata["train_row_count"] == 64
    assert metadata["steps"] == 12000
    assert metadata["steps_per_sprite"] == 187.5
    assert metadata["batch_size"] == spec["batch_size"]
    assert metadata["step_count_source"] == "default_micro_overfit_budget"
    assert metadata["step_count_defaulted_from_helper"] is True
    assert metadata["micro_overfit_budget_policy"]["default_steps_per_sprite"] == 188
    assert metadata["approx_update_exposure"]["optimizer_steps_per_sprite"] == 187.5


def test_64_sprite_status_thresholds_do_not_reclassify_smaller_overfits() -> None:
    qa = {"errors": []}
    borderline_metrics = {
        "mean_alpha_iou": 0.8743,
        "mean_visible_rgb_mae": 0.1179,
        "near_match_rate": 0.75,
    }
    assert generator_audits._run_status(qa, borderline_metrics, sprite_count=32) == "pass"
    assert generator_audits._run_status(qa, borderline_metrics, sprite_count=64) == "warn"
    assert (
        generator_audits._run_status(
            qa,
            {"mean_alpha_iou": 0.91, "mean_visible_rgb_mae": 0.09, "near_match_rate": 0.70},
            sprite_count=64,
        )
        == "pass"
    )


def test_balanced_eval_prompt_builder_dedupes_sprite_ids_and_round_robins_categories(tmp_path: Path) -> None:
    rows = [
        {"sprite_id": "p0", "split": "train", "category": "potion", "caption": "red potion", "object_name": "potion"},
        {"sprite_id": "p0", "split": "train", "category": "potion", "caption": "duplicate potion"},
        {"sprite_id": "w0", "split": "train", "category": "weapon", "caption": "iron sword", "object_name": "sword"},
        {"sprite_id": "a0", "split": "train", "category": "armor", "caption": "steel helm", "object_name": "helm"},
        {"sprite_id": "p1", "split": "train", "category": "potion", "caption": "blue potion", "object_name": "potion"},
        {"sprite_id": "w1", "split": "train", "category": "weapon", "caption": "gold axe", "object_name": "axe"},
    ]
    manifest = tmp_path / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = build_balanced_eval_prompts(
        training_manifest=manifest,
        out=tmp_path / "prompts.jsonl",
        max_prompts=5,
    )
    prompts = read_prompt_records(tmp_path / "prompts.jsonl")

    assert [prompt["target_sprite_id"] for prompt in prompts] == ["a0", "p0", "w0", "p1", "w1"]
    assert len({prompt["target_sprite_id"] for prompt in prompts}) == len(prompts)
    assert summary["category_counts"] == {"armor": 1, "potion": 2, "weapon": 2}
    assert summary["target_ids_unique"] is True
    assert prompts[0]["prompt"] == "steel helm"
    assert prompts[0]["object_name"] == "helm"


def test_full_v4_decision_emits_control_warnings_for_known_bad_metrics() -> None:
    decision = decide_full_v4_challenger_audit(
        {
            "training": {"val_train_loss_ratio": 8.0},
            "generated_qa": {"errors": 0},
            "generated_review": {"touches_border_rate": 0.2, "too_many_rare_colors_rate": 0.1},
            "prompt_faithfulness": {
                "category_consistency_rate": 0.50,
                "color_consistency_rate": 0.60,
                "repeated_silhouette_rate": 0.30,
                "generic_blob_collapse_rate": 0.25,
            },
            "prompt_sensitivity": {"prompt_pair_near_duplicate_rate": 0.60},
        }
    )

    assert decision["code"] == "D"
    warning_names = {warning["name"] for warning in decision["warnings"]}
    assert "category_consistency_rate" in warning_names
    assert "prompt_pair_near_duplicate_rate" in warning_names


def test_ood_control_score_orders_better_control_above_degraded_control() -> None:
    good = {
        "qa_errors": 0,
        "qa_error_rate": 0.0,
        "category": 0.85,
        "color": 0.90,
        "rare_color_rate": 0.20,
        "blob_collapse_rate": 0.10,
        "repeated_silhouette_rate": 0.05,
        "potion_collapse_rate": 0.05,
    }
    degraded = {
        **good,
        "color": 0.70,
        "rare_color_rate": 0.45,
        "blob_collapse_rate": 0.35,
        "repeated_silhouette_rate": 0.30,
    }

    assert generator_audits._ood_control_score_v1(good) > generator_audits._ood_control_score_v1(degraded)


def test_ood_control_guardrails_pass_and_fail() -> None:
    passing = {
        "qa_errors": 0,
        "category": 0.75,
        "color": 0.80,
        "rare_color_rate": 0.35,
        "blob_collapse_rate": 0.35,
    }
    failing = {
        **passing,
        "qa_errors": 1,
        "color": 0.79,
        "rare_color_rate": 0.36,
    }

    assert generator_audits._ood_guardrail_failures(passing) == []
    assert set(generator_audits._ood_guardrail_failures(failing)) == {
        "qa_errors",
        "ood_color",
        "ood_rare_color_rate",
    }


def test_full_v4_audit_summary_schema_with_patched_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest, count=6)

    def fake_train(config: Any) -> dict[str, Any]:
        assert config.checkpoint_steps == ()
        config.out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "train_records": 6,
            "val_records": 2,
            "initial_train_loss": 2.0,
            "final_train_loss": 0.1,
            "last_step_loss": 0.08,
            "val_loss": 0.7,
            "steps_completed": config.max_steps,
            "batch_size": config.batch_size,
            "elapsed_seconds": 1.5,
            "model_config": {"base_channels": 64},
        }

    def fake_sample(config: Any) -> dict[str, Any]:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "sample_count": config.max_samples,
            "warnings": 0,
            "fully_transparent_count": 0,
            "max_visible_color_count": 32,
            "contact_sheet": "generation_contact_sheet.png",
        }

    monkeypatch.setattr(generator_audits, "run_challenger_training", fake_train)
    monkeypatch.setattr(generator_audits, "run_sample_generator_challenger", fake_sample)
    monkeypatch.setattr(
        generator_audits,
        "qa_generated_sprites",
        lambda generated_dir: SimpleNamespace(
            to_json_dict=lambda: {"sample_count": 4, "errors": [], "warnings": [], "ok": True, "checks": {}}
        ),
    )
    monkeypatch.setattr(
        generator_audits,
        "review_generated_sprites",
        lambda config: SimpleNamespace(
            report={
                "sample_count": 4,
                "overall": {
                    "mean_alpha_coverage": 0.5,
                    "median_visible_color_count": 32,
                    "warning_counts": {"touches_border": 1},
                    "mean_raw_indexed_rgb_mae_visible": 0.01,
                },
                "groups": {"potion": {"count": 2}},
                "contact_sheets": {"overall": "review_contact_sheet.png"},
            }
        ),
    )
    def fake_faithfulness(config: Any) -> dict[str, Any]:
        report = {
            "sample_count": 4,
            "source_selection": {
                "mode": "all",
                "max_sources": int(config.max_sources) if config.max_sources is not None else None,
                "source_count_total": 8,
                "source_count_used": 8,
                "source_candidate_hash": "deadbeef",
                "source_category_counts": {"weapon": 4, "item_icon": 4},
            },
            "nearest_source_summary": {"mean_distance": 0.2},
            "category_consistency_rate": 0.5,
            "nearest_source_category_consistency_rate": 0.5,
            "color_consistency_rate": 0.7,
            "shape_bbox_consistency_rate": 1.0,
            "repeated_silhouette_rate": 0.25,
            "nearest_neighbor_duplicate_rate": 0.0,
            "generic_potion_collapse_rate": 0.0,
            "generic_flame_collapse_rate": 0.0,
            "generic_blob_collapse_rate": 0.3,
            "object_families_worst_faithfulness": [],
            "color_prompts_failed": [],
        }
        out_json = Path(config.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    monkeypatch.setattr(generator_audits, "run_prompt_faithfulness", fake_faithfulness)
    monkeypatch.setattr(
        generator_audits,
        "run_prompt_sensitivity",
        lambda config: {
            "sets": {
                "same_noise_different_prompts": {"metrics": {"mean_pairwise_difference": 0.3, "near_duplicate_rate": 0.0}},
                "same_prompt_different_noise": {"metrics": {"diversity_score": 0.1}},
                "prompt_pairs": {"metrics": {"near_duplicate_rate": 0.6, "pairs": []}},
            },
            "warnings": ["near_duplicate_pair"],
        },
    )

    report = run_full_v4_challenger_audit(
        FullV4ChallengerAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            max_steps=2,
            max_eval_prompts=4,
            max_sensitivity_prompts=2,
            noise_samples=1,
            device="cpu",
            amp=False,
            lr_schedule="none",
            lr_warmup_steps=0,
        )
    )

    assert report["training"]["steps"] == 2
    assert "checkpoint_evaluation" not in report
    assert report["artifacts"]["sample_checkpoint"].endswith("checkpoint_last.pt")
    assert report["prompt_set"]["prompt_count"] == 4
    assert report["generated_qa"]["errors"] == 0
    assert report["generated_review"]["warning_counts"]["touches_border"] == 1
    assert report["prompt_faithfulness"]["generic_blob_collapse_rate"] == 0.3
    assert report["palette_swap"]["palette_swap_augmentation"] is False
    assert report["palette_swap"]["palette_swap_prob"] == 0.0
    assert report["palette_swap"]["palette_swap_preserve_outline"] is True
    assert report["prompt_faithfulness"]["source_selection"]["source_candidate_hash"] == "deadbeef"
    assert report["prompt_sensitivity"]["prompt_pair_near_duplicate_rate"] == 0.6
    assert report["decision"]["code"] == "D"
    assert (tmp_path / "audit" / "full_v4_challenger_audit.json").is_file()
    assert (tmp_path / "audit" / "full_v4_challenger_audit.md").is_file()

    # Full-v4 audit must embed exactly the prompt-faithfulness metrics written to disk.
    report_path = tmp_path / "audit" / "generated_eval" / "prompt_faithfulness_report.json"
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    embedded = report["prompt_faithfulness"]
    assert embedded["generic_blob_collapse_rate"] == on_disk["generic_blob_collapse_rate"]
    assert embedded["category_consistency_rate"] == on_disk["category_consistency_rate"]

    # A stale on-disk report must trigger the consistency guard.
    report_path.write_text(json.dumps({"generic_blob_collapse_rate": 0.99}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt faithfulness mismatch"):
        generator_audits._verify_faithfulness_matches_disk(on_disk, report_path)


def test_full_v4_checkpoint_evaluation_leaderboard_and_selected_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest, count=4)
    ood_prompts = tmp_path / "ood_prompts.jsonl"
    ood_rows = [
        {"prompt_id": "ood0", "prompt": "blue sword", "category": "weapon"},
        {"prompt_id": "ood1", "prompt": "red shield", "category": "armor"},
    ]
    ood_prompts.write_text("".join(json.dumps(row) + "\n" for row in ood_rows), encoding="utf-8")
    sample_calls: list[Any] = []

    def fake_train(config: Any) -> dict[str, Any]:
        assert config.checkpoint_steps == (1, 2)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        for step in config.checkpoint_steps:
            (config.out_dir / f"checkpoint_step_{step:06d}.pt").write_text("checkpoint\n", encoding="utf-8")
            (config.out_dir / f"checkpoint_step_{step:06d}_ema.pt").write_text("ema\n", encoding="utf-8")
        (config.out_dir / "checkpoint_last.pt").write_text("last\n", encoding="utf-8")
        (config.out_dir / "checkpoint_last_ema.pt").write_text("last ema\n", encoding="utf-8")
        return {
            "train_records": 4,
            "val_records": 0,
            "initial_train_loss": 2.0,
            "final_train_loss": 0.2,
            "last_step_loss": 0.2,
            "val_loss": None,
            "steps_completed": config.max_steps,
            "batch_size": config.batch_size,
            "model_config": {"base_channels": 8},
            "ema_enabled": True,
            "ema_decay": config.ema_decay,
        }

    def fake_sample(config: Any) -> dict[str, Any]:
        sample_calls.append(config)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "sample_id": f"sample_{index:06d}",
                "prompt_id": f"p{index}",
                "prompt": "prompt",
                "category": "weapon",
                "checkpoint": str(config.checkpoint),
                "paths": {"indexed_png": f"sample_{index:06d}.png"},
                "warnings": [],
            }
            for index in range(int(config.max_samples))
        ]
        (config.out_dir / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return {"sample_count": len(records), "warnings": 0, "fully_transparent_count": 0, "contact_sheet": "sheet.png"}

    def checkpoint_text(generated_dir: Path) -> str:
        manifest_path = generated_dir / "generated_manifest.jsonl"
        if not manifest_path.is_file():
            return ""
        first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        return str(first.get("checkpoint") or "")

    def fake_review(config: Any) -> SimpleNamespace:
        checkpoint = checkpoint_text(config.generated_dir)
        rare = 0.20 if "000001_ema" in checkpoint else 0.45
        return SimpleNamespace(
            report={
                "sample_count": 2,
                "overall": {
                    "mean_alpha_coverage": 0.5,
                    "median_visible_color_count": 8,
                    "warning_counts": {},
                    "touches_border_rate": 0.10,
                    "too_many_rare_colors_rate": rare,
                },
                "contact_sheets": {},
            }
        )

    def fake_faithfulness(config: Any) -> dict[str, Any]:
        checkpoint = checkpoint_text(config.generated)
        good = "000001_ema" in checkpoint
        report = {
            "sample_count": 2,
            "source_selection": {"mode": "all", "source_candidate_hash": "same"},
            "nearest_source_summary": {"mean_distance": 0.2},
            "category_consistency_rate": 0.85 if good else 0.90,
            "nearest_source_category_consistency_rate": 0.85 if good else 0.90,
            "color_consistency_rate": 0.90 if good else 0.70,
            "shape_bbox_consistency_rate": 1.0,
            "repeated_silhouette_rate": 0.05 if good else 0.30,
            "nearest_neighbor_duplicate_rate": 0.0,
            "generic_potion_collapse_rate": 0.05,
            "generic_flame_collapse_rate": 0.0,
            "generic_blob_collapse_rate": 0.10 if good else 0.35,
            "object_families_worst_faithfulness": [],
            "color_prompts_failed": [],
        }
        Path(config.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    monkeypatch.setattr(generator_audits, "run_challenger_training", fake_train)
    monkeypatch.setattr(generator_audits, "run_sample_generator_challenger", fake_sample)
    monkeypatch.setattr(
        generator_audits,
        "qa_generated_sprites",
        lambda generated_dir: SimpleNamespace(
            to_json_dict=lambda: {"sample_count": 2, "errors": [], "warnings": [], "ok": True, "checks": {}}
        ),
    )
    monkeypatch.setattr(generator_audits, "review_generated_sprites", fake_review)
    monkeypatch.setattr(generator_audits, "run_prompt_faithfulness", fake_faithfulness)
    monkeypatch.setattr(
        generator_audits,
        "run_prompt_sensitivity",
        lambda config: {
            "sets": {
                "same_noise_different_prompts": {"metrics": {"mean_pairwise_difference": 0.3, "near_duplicate_rate": 0.0}},
                "same_prompt_different_noise": {"metrics": {"diversity_score": 0.1}},
                "prompt_pairs": {"metrics": {"near_duplicate_rate": 0.0, "pairs": []}},
            },
            "warnings": [],
        },
    )

    report = run_full_v4_challenger_audit(
        FullV4ChallengerAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            max_steps=2,
            max_eval_prompts=2,
            max_sensitivity_prompts=1,
            noise_samples=1,
            sample_ema=True,
            eval_checkpoints=True,
            eval_checkpoint_steps="1,2",
            ood_prompts=ood_prompts,
            device="cpu",
            amp=False,
            lr_schedule="none",
            lr_warmup_steps=0,
        )
    )

    checkpoint_eval = report["checkpoint_evaluation"]
    assert checkpoint_eval["enabled"] is True
    assert checkpoint_eval["selection_metric"] == "ood_control_score_v1"
    assert checkpoint_eval["selected_step"] == 1
    assert checkpoint_eval["selected_checkpoint"].endswith("checkpoint_step_000001_ema.pt")
    assert checkpoint_eval["selected_metrics"]["guardrails_passed"] is True
    assert checkpoint_eval["final_step_metrics"]["step"] == 2
    assert all(entry["ema"] is True for entry in checkpoint_eval["leaderboard"])
    assert {"step", "checkpoint_path", "ood_score", "qa_errors", "category", "color", "rare_color_rate", "selected"} <= set(
        checkpoint_eval["leaderboard"][0]
    )
    final_generation_call = next(call for call in sample_calls if Path(call.out_dir).name == "generated_eval")
    assert str(final_generation_call.checkpoint).endswith("checkpoint_step_000001_ema.pt")
    assert report["artifacts"]["sample_checkpoint"].endswith("checkpoint_step_000001_ema.pt")
    assert report["artifacts"]["final_step_sample_checkpoint"].endswith("checkpoint_last_ema.pt")
    markdown = (tmp_path / "audit" / "full_v4_challenger_audit.md").read_text(encoding="utf-8")
    assert "## Checkpoint OOD Leaderboard" in markdown


def test_full_v4_audit_cli_accepts_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spritelab.training.cli import main as train_cli

    captured: list[FullV4ChallengerAuditConfig] = []

    def fake_audit(config: FullV4ChallengerAuditConfig) -> dict[str, Any]:
        captured.append(config)
        return {"decision": {"code": "F", "label": "Unknown."}}

    monkeypatch.setattr(generator_audits, "run_full_v4_challenger_audit", fake_audit)
    train_cli(
        [
            "audit-challenger-full-v4",
            "--dataset",
            str(tmp_path / "dataset"),
            "--training-manifest",
            str(tmp_path / "training_manifest.jsonl"),
            "--out",
            str(tmp_path / "audit"),
            "--architecture",
            "rectified_flow",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--max-steps",
            "2",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
            "--lr",
            "0.0002",
            "--conditioning-mode",
            "caption_semantic_structured",
            "--cfg-dropout",
            "0.1",
            "--structured-field-dropout",
            "0.1",
            "--ema-decay",
            "0.99",
            "--sample-ema",
            "--foreground-rgb-loss-weight",
            "2.0",
            "--background-rgb-loss-weight",
            "0.25",
            "--palette-loss-weight",
            "0.1",
            "--palette-loss-temperature",
            "0.07",
            "--sample-steps",
            "2",
            "--cfg-scale",
            "2.0",
            "--max-colors",
            "8",
            "--alpha-threshold",
            "0.5",
            "--max-eval-prompts",
            "4",
            "--max-sensitivity-prompts",
            "2",
            "--noise-samples",
            "1",
            "--sample-batch-size",
            "8",
            "--run-ood-compositional",
            "--ood-prompts",
            str(tmp_path / "ood.jsonl"),
            "--eval-checkpoints",
            "--eval-checkpoint-every",
            "5",
            "--eval-checkpoint-steps",
            "1,2",
            "--palette-swap-augmentation",
            "--palette-swap-prob",
            "0.5",
            "--palette-swap-families",
            "red,blue,green",
            "--palette-swap-require-explicit-caption-color",
            "--palette-swap-require-explicit-semantic-color",
            "--palette-swap-no-caption-prepend",
            "--no-palette-swap-update-prompts",
            "--no-amp",
            "--lr-schedule",
            "none",
            "--lr-warmup-steps",
            "0",
        ]
    )

    assert captured
    assert captured[0].max_steps == 2
    assert captured[0].max_eval_prompts == 4
    assert captured[0].conditioning_mode == "caption_semantic_structured"
    assert captured[0].palette_swap_augmentation is True
    assert captured[0].palette_swap_prob == 0.5
    assert captured[0].palette_swap_families == "red,blue,green"
    assert captured[0].palette_swap_require_explicit_caption_color is True
    assert captured[0].palette_swap_require_explicit_semantic_color is True
    assert captured[0].palette_swap_no_caption_prepend is True
    assert captured[0].palette_swap_preserve_outline is True
    assert captured[0].palette_swap_update_prompts is False
    assert captured[0].structured_field_dropout == 0.1
    assert captured[0].ema_decay == 0.99
    assert captured[0].sample_ema is True
    assert captured[0].foreground_rgb_loss_weight == 2.0
    assert captured[0].background_rgb_loss_weight == 0.25
    assert captured[0].palette_loss_weight == 0.1
    assert captured[0].palette_loss_temperature == 0.07
    assert captured[0].num_workers == 2
    assert captured[0].sample_batch_size == 8
    assert captured[0].run_ood_compositional is True
    assert captured[0].ood_prompts == tmp_path / "ood.jsonl"
    assert captured[0].eval_checkpoints is True
    assert captured[0].eval_checkpoint_every == 5
    assert captured[0].eval_checkpoint_steps == "1,2"
    assert captured[0].amp is False


def test_challenger_conditioning_comparison_reports_improvements(tmp_path: Path) -> None:
    def report(mode: str, *, category: float, color: float, blob: float) -> dict[str, Any]:
        return {
            "conditioning_mode": mode,
            "training": {"final_train_loss": 0.1, "val_loss": 0.5, "val_train_loss_ratio": 5.0},
            "ood_compositional": {
                "generated_qa": {"errors": 0},
                "generated_review": {"touches_border_rate": 0.4, "too_many_rare_colors_rate": 0.2},
                "prompt_faithfulness": {
                    "category_consistency_rate": category,
                    "color_consistency_rate": color,
                    "repeated_silhouette_rate": 0.2,
                    "generic_potion_collapse_rate": 0.3,
                    "generic_blob_collapse_rate": blob,
                },
                "prompt_sensitivity": {
                    "prompt_pair_near_duplicate_rate": 0.2,
                    "same_noise_mean_difference": 0.4,
                    "same_prompt_diversity": 0.1,
                },
            },
        }

    comparison = compare_challenger_conditioning_audits(
        baseline=report("caption_semantic", category=0.35, color=0.61, blob=0.64),
        structured=report("caption_semantic_structured", category=0.50, color=0.70, blob=0.30),
        out_dir=tmp_path / "comparison",
    )

    assert comparison["answers"]["ood_category_consistency_improved"] is True
    assert comparison["answers"]["ood_color_consistency_improved"] is True
    assert comparison["answers"]["generic_blob_collapse_reduced"] is True
    assert comparison["answers"]["ood_category"] == "improved"
    assert "dataset_grounded" in comparison
    assert "ood_compositional" in comparison
    assert "source_distribution_deltas" in comparison
    markdown = (tmp_path / "comparison" / "challenger_conditioning_comparison.md").read_text(encoding="utf-8")
    assert "Dataset-grounded eval comparison" in markdown
    assert "OOD compositional eval comparison" in markdown
    assert "Source-distribution deltas" in markdown
    assert "Category improved:" not in markdown
    assert (tmp_path / "comparison" / "challenger_conditioning_comparison.json").is_file()
    assert (tmp_path / "comparison" / "challenger_conditioning_comparison.md").is_file()


def test_challenger_conditioning_comparison_warns_on_source_hash_mismatch(tmp_path: Path) -> None:
    def report(mode: str, *, source_hash: str) -> dict[str, Any]:
        return {
            "conditioning_mode": mode,
            "training": {"final_train_loss": 0.1, "val_loss": 0.5, "val_train_loss_ratio": 5.0},
            "ood_compositional": {
                "generated_qa": {"errors": 0},
                "generated_review": {"touches_border_rate": 0.4, "too_many_rare_colors_rate": 0.2},
                "prompt_faithfulness": {
                    "category_consistency_rate": 0.5,
                    "color_consistency_rate": 0.7,
                    "repeated_silhouette_rate": 0.2,
                    "generic_potion_collapse_rate": 0.3,
                    "generic_blob_collapse_rate": 0.3,
                    "source_selection": {
                        "mode": "deterministic_first_n",
                        "source_count_total": 928,
                        "source_count_used": 128,
                        "source_candidate_hash": source_hash,
                    },
                },
                "prompt_sensitivity": {
                    "prompt_pair_near_duplicate_rate": 0.2,
                    "same_noise_mean_difference": 0.4,
                    "same_prompt_diversity": 0.1,
                },
            },
        }

    comparison = compare_challenger_conditioning_audits(
        baseline=report("caption_semantic", source_hash="aaaa"),
        structured=report("caption_semantic_structured", source_hash="bbbb"),
        out_dir=tmp_path / "comparison_mismatch",
    )

    assert any("not directly comparable" in warning for warning in comparison["warnings"])
    markdown = (tmp_path / "comparison_mismatch" / "challenger_conditioning_comparison.md").read_text(encoding="utf-8")
    assert "Warnings" in markdown
    assert "not directly comparable" in markdown


def test_challenger_conditioning_comparison_reports_checkpoint_selection_metadata(tmp_path: Path) -> None:
    def report(mode: str, *, selection_enabled: bool, cfg_scale: float, selected_step: int | None) -> dict[str, Any]:
        checkpoint_eval = (
            {
                "enabled": True,
                "selection_metric": "ood_control_score_v1",
                "selected_checkpoint": f"run/checkpoint_step_{selected_step:06d}_ema.pt",
                "selected_step": selected_step,
                "selected_score": 2.5,
                "selected_deployable": True,
                "final_step_checkpoint": "run/checkpoint_step_000002_ema.pt",
                "leaderboard": [],
            }
            if selection_enabled and selected_step is not None
            else {}
        )
        return {
            "conditioning_mode": mode,
            "config": {"cfg_scale": cfg_scale},
            "artifacts": {
                "sample_checkpoint": checkpoint_eval.get("selected_checkpoint", "run/checkpoint_last.pt"),
                "final_step_sample_checkpoint": "run/checkpoint_last.pt",
            },
            "checkpoint_evaluation": checkpoint_eval,
            "training": {"final_train_loss": 0.1, "val_loss": 0.5, "val_train_loss_ratio": 5.0},
            "ood_compositional": {
                "generated_qa": {"errors": 0},
                "generated_review": {"touches_border_rate": 0.4, "too_many_rare_colors_rate": 0.2},
                "prompt_faithfulness": {
                    "category_consistency_rate": 0.5,
                    "color_consistency_rate": 0.7,
                    "repeated_silhouette_rate": 0.2,
                    "generic_potion_collapse_rate": 0.3,
                    "generic_blob_collapse_rate": 0.3,
                    "source_selection": {"source_candidate_hash": "same"},
                },
                "prompt_sensitivity": {
                    "prompt_pair_near_duplicate_rate": 0.2,
                    "same_noise_mean_difference": 0.4,
                    "same_prompt_diversity": 0.1,
                },
            },
        }

    comparison = compare_challenger_conditioning_audits(
        baseline=report("caption_semantic", selection_enabled=False, cfg_scale=2.0, selected_step=None),
        structured=report("caption_semantic_structured", selection_enabled=True, cfg_scale=3.0, selected_step=1),
        out_dir=tmp_path / "comparison_selection",
    )

    assert comparison["structured"]["selected_step"] == 1
    assert comparison["structured"]["selection_score"] == 2.5
    assert comparison["checkpoint_selection_comparison"]["both_used_checkpoint_selection"] is False
    assert any("final-step run vs selected-checkpoint run" in warning for warning in comparison["warnings"])
    assert any("CFG scales differ" in warning for warning in comparison["warnings"])
    markdown = (tmp_path / "comparison_selection" / "challenger_conditioning_comparison.md").read_text(encoding="utf-8")
    assert "Checkpoint selection" in markdown
    assert "checkpoint_step_000001_ema.pt" in markdown


def test_category_weighted_source_baseline_computation() -> None:
    by_category = {
        "weapon": {"metrics": {"border_touch_rate": 0.75, "mean_alpha_coverage": 0.5, "mean_bbox_width": 20.0, "mean_bbox_height": 24.0, "mean_center_offset": 1.0, "mean_visible_color_count": 10.0}},
        "item_icon": {"metrics": {"border_touch_rate": 0.25, "mean_alpha_coverage": 0.3, "mean_bbox_width": 12.0, "mean_bbox_height": 14.0, "mean_center_offset": 3.0, "mean_visible_color_count": 6.0}},
    }
    weighted = generator_audits._category_weighted_source_baseline(
        by_category,
        {"weapon": 3, "item_icon": 1},
    )

    assert weighted["metrics"]["border_touch_rate"] == pytest.approx(0.625)
    assert weighted["metrics"]["mean_alpha_coverage"] == pytest.approx(0.45)
    assert weighted["metrics"]["mean_visible_color_count"] == pytest.approx(9.0)
    assert weighted["missing_categories"] == []


def test_full_v4_decision_border_warning_is_source_relative() -> None:
    base_sections = {
        "generated_qa": {"errors": 0},
        "generated_review": {"touches_border_rate": 0.95, "median_visible_colors_pinned": False},
        "prompt_faithfulness": {"category_consistency_rate": 0.9, "color_consistency_rate": 0.9, "repeated_silhouette_rate": 0.0, "generic_blob_collapse_rate": 0.0},
        "prompt_sensitivity": {"prompt_pair_near_duplicate_rate": 0.0},
        "generated_vs_source": {"grounded_eval": {"deltas": {"border_touch_rate": 0.05, "mean_visible_color_count": 0.0}}},
    }
    no_warning = decide_full_v4_challenger_audit(base_sections)
    assert "border_touch_source_delta" not in {warning["name"] for warning in no_warning["warnings"]}

    high_delta = dict(base_sections)
    high_delta["generated_vs_source"] = {"grounded_eval": {"deltas": {"border_touch_rate": 0.11, "mean_visible_color_count": 0.0}}}
    warning = decide_full_v4_challenger_audit(high_delta)
    assert "border_touch_source_delta" in {item["name"] for item in warning["warnings"]}


def test_full_v4_decision_rare_color_warning_is_source_relative() -> None:
    sections = {
        "generated_qa": {"errors": 0},
        "generated_review": {"median_visible_color_count": 32, "max_colors": 32, "median_visible_colors_pinned": False},
        "prompt_faithfulness": {"category_consistency_rate": 0.9, "color_consistency_rate": 0.9, "repeated_silhouette_rate": 0.0, "generic_blob_collapse_rate": 0.0},
        "prompt_sensitivity": {"prompt_pair_near_duplicate_rate": 0.0},
        "generated_vs_source": {"grounded_eval": {"deltas": {"border_touch_rate": 0.0, "mean_visible_color_count": 8.5}}},
    }
    decision = decide_full_v4_challenger_audit(sections)
    assert "visible_color_count_source_delta" in {warning["name"] for warning in decision["warnings"]}

    pinned = dict(sections)
    pinned["generated_review"] = {"median_visible_color_count": 32, "max_colors": 32, "median_visible_colors_pinned": True}
    decision = decide_full_v4_challenger_audit(pinned)
    assert "median_visible_colors_pinned" in {warning["name"] for warning in decision["warnings"]}


def test_full_v4_summary_extracts_current_review_schema_variants() -> None:
    review_summary = generator_audits._full_v4_generated_review_summary(
        {
            "sample_count": 2,
            "samples": [
                {
                    "metrics": {
                        "touches_border": True,
                        "alpha_coverage": 0.5,
                        "bbox_width": 20,
                        "bbox_height": 22,
                        "center_offset_from_image_center": 1.5,
                        "visible_color_count": 32,
                    },
                    "warnings": ["touches_border", "too_many_rare_colors"],
                },
                {
                    "metrics": {
                        "touches_border": False,
                        "alpha_coverage": 0.25,
                        "bbox_width": 12,
                        "bbox_height": 14,
                        "center_offset_from_image_center": 2.5,
                        "visible_color_count": 16,
                    },
                    "warnings": [],
                },
            ],
        },
        max_colors=32,
    )
    faithfulness_summary = generator_audits._full_v4_faithfulness_summary(
        {"nearest_source_summary": {"mean_dist": 0.42}}
    )

    assert review_summary["touches_border_rate"] == 0.5
    assert review_summary["too_many_rare_colors_rate"] == 0.5
    assert review_summary["mean_bbox_width"] == 16.0
    assert review_summary["mean_visible_color_count"] == 24.0
    assert faithfulness_summary["mean_nearest_source_distance"] == 0.42


def test_regression_audit_trains_samples_and_source_matches_same_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest)
    _patch_common_audit_steps(monkeypatch)
    monkeypatch.setattr(
        generator_audits,
        "REGRESSION_SPECS",
        ({"name": "overfit_2", "count": 2, "steps": 1, "margin": False, "batch_size": 2},),
    )
    monkeypatch.setattr(generator_audits, "run_generator_training", _train_report_from_sprite_id_list)

    report = run_regression_generator_audit(
        RegressionGeneratorAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            device="cpu",
            seed=123,
        )
    )

    run = report["runs"][0]
    persisted = json.loads((tmp_path / "audit" / "runs" / "overfit_2" / "overfit_sprite_ids.json").read_text())
    assert run["subset_equality"] is True
    assert run["overfit_subset"]["sets_equal"] is True
    assert run["generated_target_subset"]["sets_equal"] is True
    assert run["source_match_target_subset"]["sets_equal"] is True
    assert run["steps"] == 1
    assert run["steps_per_sprite"] == 0.5
    assert run["train_row_count"] == 2
    assert run["step_count_source"] == "explicit"
    assert run["step_count_defaulted_from_helper"] is False
    assert set(run["overfit_subset"]["train_sprite_ids"]) == set(persisted["sprite_ids"])
    assert set(run["overfit_subset"]["prompt_sprite_ids"]) == set(persisted["sprite_ids"])


def test_challenger_audit_trains_samples_and_source_matches_same_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest)
    _patch_common_audit_steps(monkeypatch)
    monkeypatch.setattr(
        generator_audits,
        "CHALLENGER_SPECS",
        ({"name": "overfit_2", "count": 2, "steps": 1, "batch_size": 2},),
    )
    monkeypatch.setattr(generator_audits, "run_challenger_training", _train_report_from_sprite_id_list)

    report = run_challenger_generator_audit(
        ChallengerGeneratorAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            architecture="rectified_flow",
            device="cpu",
            seed=123,
        )
    )

    run = report["runs"][0]
    persisted = json.loads((tmp_path / "audit" / "runs" / "overfit_2" / "overfit_sprite_ids.json").read_text())
    assert run["subset_equality"] is True
    assert run["overfit_subset"]["sets_equal"] is True
    assert run["generated_target_subset"]["sets_equal"] is True
    assert run["source_match_target_subset"]["sets_equal"] is True
    assert run["steps"] == 1
    assert run["steps_per_sprite"] == 0.5
    assert run["train_row_count"] == 2
    assert run["batch_size"] == 2
    assert run["approx_update_exposure"]["sample_slots_per_sprite"] == 1.0
    assert run["step_count_source"] == "explicit"
    assert run["step_count_defaulted_from_helper"] is False
    assert set(run["overfit_subset"]["train_sprite_ids"]) == set(persisted["sprite_ids"])
    assert set(run["overfit_subset"]["prompt_sprite_ids"]) == set(persisted["sprite_ids"])


def test_challenger_64_audit_uses_persisted_subset_and_normalized_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest, count=72)
    captured_steps: list[int] = []
    _patch_common_audit_steps(monkeypatch)
    monkeypatch.setattr(
        generator_audits,
        "CHALLENGER_SPECS",
        ({"name": "overfit_64_sprites", "count": 64, "batch_size": 8},),
    )

    def fake_train(config: Any) -> dict[str, Any]:
        captured_steps.append(config.max_steps)
        return _train_report_from_sprite_id_list(config)

    monkeypatch.setattr(generator_audits, "run_challenger_training", fake_train)

    report = run_challenger_generator_audit(
        ChallengerGeneratorAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            architecture="rectified_flow",
            device="cpu",
            seed=123,
        )
    )

    run = report["runs"][0]
    persisted = json.loads(
        (tmp_path / "audit" / "runs" / "overfit_64_sprites" / "overfit_sprite_ids.json").read_text()
    )
    assert captured_steps == [12000]
    assert run["name"] == "overfit_64_sprites"
    assert run["sprite_count"] == 64
    assert run["train_row_count"] == 64
    assert run["steps"] == 12000
    assert run["steps_per_sprite"] == 187.5
    assert run["step_count_source"] == "default_micro_overfit_budget"
    assert run["step_count_defaulted_from_helper"] is True
    assert run["subset_equality"] is True
    assert run["overfit_subset"]["sets_equal"] is True
    assert run["generated_target_subset"]["sets_equal"] is True
    assert run["source_match_target_subset"]["sets_equal"] is True
    assert len(persisted["sprite_ids"]) == 64
    assert set(run["overfit_subset"]["train_sprite_ids"]) == set(persisted["sprite_ids"])
    assert set(run["source_match_target_subset"]["source_match_sprite_ids"]) == set(persisted["sprite_ids"])


def test_challenger_audit_explicit_step_override_still_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest)
    captured_steps: list[int] = []
    _patch_common_audit_steps(monkeypatch)
    monkeypatch.setattr(
        generator_audits,
        "CHALLENGER_SPECS",
        ({"name": "overfit_2", "count": 2, "steps": 123, "batch_size": 2},),
    )

    def fake_train(config: Any) -> dict[str, Any]:
        captured_steps.append(config.max_steps)
        return _train_report_from_sprite_id_list(config)

    monkeypatch.setattr(generator_audits, "run_challenger_training", fake_train)

    report = run_challenger_generator_audit(
        ChallengerGeneratorAuditConfig(
            dataset=tmp_path / "dataset",
            training_manifest=manifest,
            out_dir=tmp_path / "audit",
            architecture="rectified_flow",
            device="cpu",
            seed=123,
        )
    )

    run = report["runs"][0]
    assert captured_steps == [123]
    assert run["steps"] == 123
    assert run["steps_per_sprite"] == 61.5
    assert run["step_count_source"] == "explicit"
    assert run["step_count_defaulted_from_helper"] is False


def test_audit_fails_before_sampling_when_train_and_prompt_subsets_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "training_manifest.jsonl"
    _write_manifest(manifest)
    sample_called = False
    _patch_common_audit_steps(monkeypatch)
    monkeypatch.setattr(
        generator_audits,
        "REGRESSION_SPECS",
        ({"name": "overfit_2", "count": 2, "steps": 1, "margin": False, "batch_size": 2},),
    )

    def fake_train(config: Any) -> dict[str, Any]:
        sprite_ids = read_sprite_id_list(config.sprite_id_list)
        return {
            "final_train_loss": 0.1,
            "loss_decrease": 0.9,
            "overfit_subset": {"sprite_ids": sprite_ids[1:]},
        }

    def fake_sample(config: Any) -> dict[str, Any]:
        nonlocal sample_called
        sample_called = True
        return {"sample_count": 0}

    monkeypatch.setattr(generator_audits, "run_generator_training", fake_train)
    monkeypatch.setattr(generator_audits, "run_sample_generator", fake_sample)

    with pytest.raises(ValueError, match="overfit subset mismatch"):
        run_regression_generator_audit(
            RegressionGeneratorAuditConfig(
                dataset=tmp_path / "dataset",
                training_manifest=manifest,
                out_dir=tmp_path / "audit",
                device="cpu",
                seed=123,
            )
        )
    assert sample_called is False

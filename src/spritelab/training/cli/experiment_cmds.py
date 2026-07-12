"""Config-driven experiment creation, validation, execution, and comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("experiment", help="Manage reproducible generator experiments.")
    commands = parser.add_subparsers(dest="experiment_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--ablation", default="baseline")
    create.add_argument("--out", required=True, type=Path)
    create.set_defaults(func=_create)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(func=_validate)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--resume", type=Path)
    run.add_argument("--unsafe-resume", action="store_true")
    run.set_defaults(func=_run)
    sample = commands.add_parser("sample")
    sample.add_argument("--config", required=True, type=Path)
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--prompts", required=True, type=Path)
    sample.add_argument("--out", required=True, type=Path)
    sample.add_argument("--paired-seeds", action="store_true")
    sample.set_defaults(func=_sample)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--samples", required=True, type=Path)
    evaluate.set_defaults(func=_evaluate)
    compare = commands.add_parser("compare")
    compare.add_argument("--runs", required=True, nargs="+", type=Path)
    compare.add_argument("--out", required=True, type=Path)
    compare.set_defaults(func=_compare)
    qualify = commands.add_parser("qualify-determinism")
    qualify.add_argument("--mode", choices=["off", "warn", "strict"], default="strict")
    qualify.add_argument("--device", default="cuda")
    qualify.add_argument("--steps", type=int, default=4)
    qualify.add_argument("--out", type=Path)
    qualify.set_defaults(func=_qualify_determinism)
    _register_campaign_commands(subparsers)


def _register_campaign_commands(subparsers: argparse._SubParsersAction) -> None:
    plan = subparsers.add_parser("campaign-plan", help="Plan a versioned fixed-step three-seed campaign.")
    plan.add_argument("--config", required=True, type=Path, help="Campaign specification JSON.")
    plan.add_argument("--out", required=True, type=Path, help="New campaign manifest path.")
    plan.add_argument("--plan-only", action="store_true", default=True, help="Plan only (the safe default).")
    plan.set_defaults(func=_campaign_plan)

    validate = subparsers.add_parser("campaign-validate", help="Validate identities and fixed-step fairness.")
    validate.add_argument("--campaign", required=True, type=Path)
    validate.add_argument("--out", type=Path, help="Optional new validation report path.")
    validate.add_argument("--validate-only", action="store_true", default=True, help="Validate only; launch nothing.")
    validate.set_defaults(func=_campaign_validate)

    run = subparsers.add_parser("campaign-run", help="Execute an already-ready campaign after explicit gates.")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--execute", action="store_true", help="Required explicit execution mode.")
    run.add_argument("--confirm-execute", action="store_true", help="Required non-interactive launch confirmation.")
    run.add_argument("--resume", action="store_true", help="Resume identity-verified partial runs.")
    run.add_argument("--unsafe-resume", action="store_true", help="Rejected for fair-comparison campaigns.")
    run.set_defaults(func=_campaign_run)

    status = subparsers.add_parser("campaign-status", help="Audit run roots and artifact completeness without launch.")
    status.add_argument("--campaign", required=True, type=Path)
    status.add_argument("--campaign-artifact-root", type=Path)
    status.add_argument("--out", type=Path, help="Optional new status report path.")
    status.set_defaults(func=_campaign_status)


def _create(parsed: argparse.Namespace) -> None:
    from spritelab.training.experiment_system import ablation_registry

    registry = ablation_registry()
    if parsed.ablation not in registry:
        raise ValueError(f"unknown ablation {parsed.ablation!r}; choose from {sorted(registry)}")
    template = Path(__file__).resolve().parents[1] / "experiment_configs" / "baseline_v1.yaml"
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    data["name"] = parsed.name
    data["ablation"] = parsed.ablation
    for dotted, value in registry[parsed.ablation]["overrides"].items():
        _set_dotted(data, dotted, value)
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    if parsed.out.exists():
        raise FileExistsError(f"refusing to overwrite experiment config: {parsed.out}")
    parsed.out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Created {parsed.out}")


def _validate(parsed: argparse.Namespace) -> None:
    manifest = _prepare_manifest(parsed.config, write=True)
    print(f"Valid experiment: {manifest['name']} ({manifest['experiment_hash']})")


def _prepare_manifest(config_path: Path, *, write: bool) -> dict:
    from spritelab.training.data import read_jsonl
    from spritelab.training.experiment_system import build_experiment_manifest, load_config, validate_ablation_config
    from spritelab.training.structured_conditioning import build_structured_conditioning_vocab
    from spritelab.training.tokenization import SpriteTextTokenizer

    config = load_config(config_path)
    validate_ablation_config(config)
    dataset = config["dataset"]
    manifest_path = _resolve(config_path, dataset["training_manifest"])
    rows = read_jsonl(manifest_path)
    train_rows = [row for row in rows if row.get("split") == dataset.get("split", "train")]
    if not train_rows:
        raise ValueError("training manifest has no rows for configured split")
    tokenizer = SpriteTextTokenizer.build_from_records(
        train_rows, max_length=int(config["conditioning"].get("caption_max_length", 32))
    )
    structured = None
    if "structured" in str(config["conditioning"]["mode"]):
        structured = build_structured_conditioning_vocab(train_rows).to_json_dict()
    result = build_experiment_manifest(
        config,
        dataset_manifest=manifest_path,
        split_manifest=_resolve(config_path, dataset.get("split_manifest", dataset["training_manifest"])),
        tokenizer=tokenizer.to_json_dict(),
        structured_vocab=structured,
        repo=Path.cwd(),
    )
    if write:
        output = config_path.with_suffix(".manifest.json")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run(parsed: argparse.Namespace) -> None:
    from spritelab.training.experiment_system import load_config
    from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training

    manifest = _prepare_manifest(parsed.config, write=True)
    config = load_config(parsed.config)
    dataset, model = config["dataset"], config["model"]
    conditioning, loss = config["conditioning"], config["loss"]
    optimizer, augmentation = config["optimizer"], config["augmentation"]
    runtime, ema = config["runtime"], config["ema"]
    out_dir = _resolve(parsed.config, runtime["out_dir"])
    if not parsed.resume and out_dir.exists() and any(out_dir.glob("checkpoint*.pt")):
        raise FileExistsError(f"refusing to overwrite checkpoints in {out_dir}; choose a new run directory or resume")
    max_steps = min(2, int(runtime["max_steps"])) if parsed.smoke else int(runtime["max_steps"])
    kwargs = {
        "dataset_dir": _resolve(parsed.config, dataset["directory"]),
        "training_manifest": _resolve(parsed.config, dataset["training_manifest"]),
        "out_dir": out_dir,
        "split": str(dataset.get("split", "train")),
        "batch_size": min(2, int(runtime["batch_size"])) if parsed.smoke else int(runtime["batch_size"]),
        "max_steps": max_steps,
        "learning_rate": float(optimizer["learning_rate"]),
        "device": str(runtime.get("device", "cpu")),
        "seed": int(config["seeds"]["training"]),
        "num_workers": int(runtime.get("num_workers", 0)),
        "conditioning_mode": str(conditioning["mode"]),
        "cfg_dropout": float(conditioning["cfg_dropout"]),
        "structured_field_dropout_rates": conditioning.get("field_dropout") or None,
        "ema_decay": float(ema["decay"]),
        "foreground_rgb_loss_weight": float(loss.get("foreground_rgb_weight", 1.0)),
        "background_rgb_loss_weight": float(loss.get("background_rgb_weight", 1.0)),
        "palette_loss_weight": float(loss.get("palette_aux_weight", 0.0)),
        "index_head_loss_weight": float(loss.get("index_head_weight", 0.0)) if loss.get("auxiliary_heads") else 0.0,
        "palette_head_loss_weight": float(loss.get("palette_head_weight", 0.0)) if loss.get("auxiliary_heads") else 0.0,
        "palette_presence_loss_weight": float(loss.get("palette_presence_weight", 0.0))
        if loss.get("auxiliary_heads")
        else 0.0,
        "palette_swap_augmentation": float(augmentation.get("palette_swap_probability", 0.0)) > 0,
        "palette_swap_prob": float(augmentation.get("palette_swap_probability", 0.0)),
        "base_channels": int(model["base_channels"]),
        "channel_mults": ",".join(str(value) for value in model["channel_mults"]),
        "res_blocks_per_level": int(model["res_blocks_per_level"]),
        "embed_dim": int(model["embed_dim"]),
        "sample_every": 0 if parsed.smoke else int(runtime.get("sample_every", 250)),
        "save_every": 1 if parsed.smoke else int(runtime.get("save_every", 1000)),
        "validation_mode": str(runtime.get("validation_mode", "auto")),
        "amp": str(runtime.get("precision", "fp32")) != "fp32",
        "grad_clip": float(optimizer.get("gradient_clip", 0.0)),
        "lr_schedule": str(optimizer.get("schedule", "none")),
        "lr_warmup_steps": int(optimizer.get("warmup_steps", 0)),
        "film_conditioning": bool(model.get("film_conditioning", False)),
        "bottleneck_attention": bool(model.get("bottleneck_attention", False)),
        "palette_conditioning": bool(conditioning.get("palette", {}).get("enabled", False)),
        "palette_conditioning_dropout": float(conditioning.get("palette", {}).get("dropout", 0.0)),
        "experiment_manifest": manifest,
        "resume_from": parsed.resume,
        "unsafe_resume": bool(parsed.unsafe_resume),
        "determinism": str(runtime.get("determinism", "off")),
        "timestep_validation_boundaries": tuple(
            float(value) for value in runtime.get("timestep_validation_boundaries", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ),
    }
    report = run_challenger_training(ChallengerTrainConfig(**kwargs))
    print(json.dumps({"steps_completed": report["steps_completed"], "out_dir": str(out_dir)}, sort_keys=True))


def _sample(parsed: argparse.Namespace) -> None:
    from spritelab.training.checkpoint_io import load_checkpoint
    from spritelab.training.experiment_system import load_config, validate_inference_parity
    from spritelab.training.generator_challenger import ChallengerSampleConfig, run_sample_generator_challenger

    config = load_config(parsed.config)
    validate_inference_parity(_prepare_manifest(parsed.config, write=False), load_checkpoint(parsed.checkpoint))
    sampling = config["sampling"]
    seeds = [int(config["seeds"]["sampling"])]
    if parsed.paired_seeds:
        seeds.append(seeds[0] + 1)
    for seed in seeds:
        out = parsed.out / f"seed_{seed}" if len(seeds) > 1 else parsed.out
        run_sample_generator_challenger(
            ChallengerSampleConfig(
                checkpoint=parsed.checkpoint,
                prompts=parsed.prompts,
                out_dir=out,
                max_samples=int(sampling.get("max_samples", 8)),
                steps=int(sampling["steps"]),
                cfg_scale=float(sampling["cfg_scale"]),
                seed=seed,
                noise_seed=seed,
                device=str(config["runtime"].get("device", "cpu")),
            )
        )
    print(f"Sampled {len(seeds)} paired run(s) under {parsed.out}")


def _evaluate(parsed: argparse.Namespace) -> None:
    report_path = parsed.samples / "generation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"generation report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {"evaluation_suite_version": "spritelab_eval_v1", "sample_count": report.get("sample_count", 0)}
    (parsed.samples / "experiment_evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def _compare(parsed: argparse.Namespace) -> None:
    rows = []
    for run in parsed.runs:
        report_path = run / "train_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {"run": str(run), "final_train_loss": report.get("final_train_loss"), "val_loss": report.get("val_loss")}
        )
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compared {len(rows)} runs in {parsed.out}")


def _qualify_determinism(parsed: argparse.Namespace) -> None:
    from spritelab.training.determinism import qualify_determinism

    result = qualify_determinism(mode=parsed.mode, device=parsed.device, steps=parsed.steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if parsed.out is not None:
        parsed.out.parent.mkdir(parents=True, exist_ok=True)
        if parsed.out.exists():
            raise FileExistsError(f"refusing to overwrite qualification report: {parsed.out}")
        parsed.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


def _campaign_plan(parsed: argparse.Namespace) -> None:
    from spritelab.training.campaign import plan_campaign, write_json_exclusive

    spec = json.loads(parsed.config.read_text(encoding="utf-8"))
    manifest = plan_campaign(spec)
    write_json_exclusive(parsed.out, manifest)
    print(
        json.dumps(
            {
                "campaign_id": manifest["campaign_id"],
                "plan_status": manifest["plan_status"],
                "executable": manifest["executable"],
                "manifest": str(parsed.out),
            },
            sort_keys=True,
        )
    )


def _campaign_validate(parsed: argparse.Namespace) -> None:
    from spritelab.training.campaign import load_campaign, validate_campaign, write_json_exclusive

    report = validate_campaign(load_campaign(parsed.campaign))
    if parsed.out is not None:
        write_json_exclusive(parsed.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _campaign_run(parsed: argparse.Namespace) -> None:
    from spritelab.training.campaign import execute_campaign, load_campaign

    report = execute_campaign(
        load_campaign(parsed.campaign),
        execute=bool(parsed.execute),
        confirm_execute=bool(parsed.confirm_execute),
        resume=bool(parsed.resume),
        unsafe_resume=bool(parsed.unsafe_resume),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _campaign_status(parsed: argparse.Namespace) -> None:
    from spritelab.training.campaign import (
        audit_artifact_completeness,
        audit_resume,
        load_campaign,
        write_json_exclusive,
    )

    campaign = load_campaign(parsed.campaign)
    report = {
        "resume": audit_resume(campaign),
        "artifacts": audit_artifact_completeness(campaign, campaign_artifact_root=parsed.campaign_artifact_root),
    }
    if parsed.out is not None:
        write_json_exclusive(parsed.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _resolve(config_path: Path, value: str | Path) -> Path:
    del config_path
    path = Path(value)
    return path if path.is_absolute() else path.resolve()


def _set_dotted(data: dict, dotted: str, value: object) -> None:
    target = data
    keys = dotted.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value

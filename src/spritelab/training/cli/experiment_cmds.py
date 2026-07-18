"""Config-driven experiment creation, validation, execution, and comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    run.add_argument(
        "--smoke-bundle-id",
        help="Opaque ID from the web-prepared exploratory infrastructure-smoke plan.",
    )
    run.add_argument("--smoke-device", choices=("cpu", "cuda"))
    run.add_argument("--smoke-plan-identity", help=argparse.SUPPRESS)
    run.add_argument("--smoke-launch-identity", help="Exact server-owned contained-worker launch identity.")
    run.add_argument("--resume", type=Path)
    run.add_argument("--unsafe-resume", action="store_true")
    run.add_argument(
        "--unsafe-resume-reason",
        help="Required nonempty human-supplied reason when --unsafe-resume is requested.",
    )
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


def prepare_experiment_manifest(
    config_path: Path,
    *,
    write: bool,
    config: Mapping[str, object] | None = None,
    runtime_overrides: Mapping[str, object] | None = None,
    resolution_root: Path | None = None,
    training_manifest_bytes: bytes | None = None,
    vocabulary_bytes: bytes | None = None,
    software_version_override: Mapping[str, Any] | None = None,
    hardware_summary_override: Mapping[str, Any] | None = None,
) -> dict:
    """Build the exact effective manifest, optionally from server-owned config bytes.

    ``runtime_overrides`` is used by the two-step smoke contract so the adjacent
    derived manifest describes the checkpoint that is actually written rather
    than the untouched 5,000-step source campaign.
    """
    from spritelab.training.data import read_jsonl
    from spritelab.training.experiment_system import build_experiment_manifest, load_config, validate_ablation_config
    from spritelab.training.structured_conditioning import build_structured_conditioning_vocab
    from spritelab.training.tokenization import SpriteTextTokenizer

    effective = load_config(config_path) if config is None else deepcopy(dict(config))
    if runtime_overrides:
        runtime = effective.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("experiment runtime configuration must be an object")
        runtime.update(deepcopy(dict(runtime_overrides)))
    validate_ablation_config(effective)
    dataset = effective["dataset"]
    resolver = (
        (lambda value: _resolve(config_path, value))
        if resolution_root is None
        else (lambda value: _resolve_from_root(resolution_root, value))
    )
    manifest_path = resolver(dataset["training_manifest"])
    if training_manifest_bytes is None:
        rows = read_jsonl(manifest_path)
        manifest_sha256 = None
    else:
        rows = []
        try:
            manifest_text = training_manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("retained training manifest is not UTF-8") from exc
        for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"retained training manifest line {line_number} is invalid") from exc
            if not isinstance(row, dict):
                raise ValueError(f"retained training manifest line {line_number} is not an object")
            rows.append(row)
        manifest_sha256 = hashlib.sha256(training_manifest_bytes).hexdigest()
    train_rows = [row for row in rows if row.get("split") == dataset.get("split", "train")]
    if not train_rows:
        raise ValueError("training manifest has no rows for configured split")
    generated_tokenizer = SpriteTextTokenizer.build_from_records(
        train_rows, max_length=int(effective["conditioning"].get("caption_max_length", 32))
    )
    vocabulary_path = effective["conditioning"].get("vocabulary_path")
    if vocabulary_path:
        vocabulary_file = resolver(vocabulary_path)
        try:
            tokenizer_payload = json.loads(
                vocabulary_file.read_text(encoding="utf-8")
                if vocabulary_bytes is None
                else vocabulary_bytes.decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("configured conditioning vocabulary is unreadable") from exc
        if not isinstance(tokenizer_payload, dict) or tokenizer_payload != generated_tokenizer.to_json_dict():
            raise ValueError("configured conditioning vocabulary does not match the training split")
        tokenizer = tokenizer_payload
    else:
        tokenizer = generated_tokenizer.to_json_dict()
    bindings = effective.get("campaign_bindings")
    if isinstance(bindings, dict):
        from spritelab.training.experiment_system import file_sha256

        expected_manifest_hash = bindings.get("split_manifest_hash")
        actual_manifest_hash = file_sha256(manifest_path) if manifest_sha256 is None else manifest_sha256
        if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
            raise ValueError("campaign split-manifest identity does not match the experiment input")
        expected_vocabulary_hash = bindings.get("conditioning_vocabulary_hash")
        if expected_vocabulary_hash and (
            not vocabulary_path
            or expected_vocabulary_hash
            != (
                file_sha256(resolver(vocabulary_path))
                if vocabulary_bytes is None
                else hashlib.sha256(vocabulary_bytes).hexdigest()
            )
        ):
            raise ValueError("campaign conditioning-vocabulary identity does not match the experiment input")
    structured = None
    if "structured" in str(effective["conditioning"]["mode"]):
        structured = build_structured_conditioning_vocab(train_rows).to_json_dict()
    split_manifest = resolver(dataset.get("split_manifest", dataset["training_manifest"]))
    if training_manifest_bytes is not None and split_manifest != manifest_path:
        raise ValueError("retained training input requires one exact dataset/split manifest path")
    result = build_experiment_manifest(
        effective,
        dataset_manifest=manifest_path,
        split_manifest=split_manifest,
        tokenizer=tokenizer,
        structured_vocab=structured,
        repo=Path.cwd() if resolution_root is None else resolution_root.resolve(),
        dataset_manifest_sha256=manifest_sha256,
        split_manifest_sha256=manifest_sha256,
        software_version_override=software_version_override,
        hardware_summary_override=hardware_summary_override,
    )
    if write:
        output = config_path.with_suffix(".manifest.json")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _prepare_manifest(
    config_path: Path,
    *,
    write: bool,
    config: Mapping[str, object] | None = None,
    runtime_overrides: Mapping[str, object] | None = None,
    resolution_root: Path | None = None,
    training_manifest_bytes: bytes | None = None,
    vocabulary_bytes: bytes | None = None,
    software_version_override: Mapping[str, Any] | None = None,
    hardware_summary_override: Mapping[str, Any] | None = None,
) -> dict:
    return prepare_experiment_manifest(
        config_path,
        write=write,
        config=config,
        runtime_overrides=runtime_overrides,
        resolution_root=resolution_root,
        training_manifest_bytes=training_manifest_bytes,
        vocabulary_bytes=vocabulary_bytes,
        software_version_override=software_version_override,
        hardware_summary_override=hardware_summary_override,
    )


def _authoritative_lr_schedule(config: Mapping[str, Any]) -> tuple[str, int]:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("optimizer configuration must be a mapping")
    schedule = config.get("schedule")
    if schedule is None:
        name = optimizer.get("schedule", "none")
        warmup_steps = optimizer.get("warmup_steps", 0)
    else:
        if not isinstance(schedule, Mapping):
            raise ValueError("schedule configuration must be a mapping")
        name = schedule.get("name")
        warmup_steps = schedule.get("warmup_steps", 0)
        if optimizer.get("schedule", name) != name or optimizer.get("warmup_steps", warmup_steps) != warmup_steps:
            raise ValueError("optimizer schedule projection differs from the authoritative campaign schedule")
    if not isinstance(name, str) or name not in {"none", "cosine"}:
        raise ValueError("learning-rate schedule must be 'none' or 'cosine'")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("learning-rate warmup_steps must be a nonnegative integer")
    return name, warmup_steps


def _require_exact_smoke_manifest(
    rebuilt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if rebuilt != expected:
        raise ValueError("the effective smoke manifest differs from the server-prepared manifest")


def _run(parsed: argparse.Namespace) -> None:
    bundle_id = parsed.smoke_bundle_id
    bundle_device = parsed.smoke_device
    bundle_plan_identity = getattr(parsed, "smoke_plan_identity", None)
    bundle_launch_identity = getattr(parsed, "smoke_launch_identity", None)
    if (bundle_id is None) != (bundle_device is None):
        raise ValueError("--smoke-bundle-id and --smoke-device must be supplied together")
    if bundle_id is not None and not parsed.smoke:
        raise ValueError("registrable smoke bundles require --smoke")
    if bundle_id is None and bundle_launch_identity is not None:
        raise ValueError("--smoke-launch-identity requires a registrable smoke bundle")
    if bundle_id is None and bundle_plan_identity is not None:
        raise ValueError("--smoke-plan-identity requires a registrable smoke bundle")
    if bundle_id is not None and bundle_launch_identity is None:
        raise ValueError("registrable smoke bundles require --smoke-launch-identity")
    if bundle_id is not None and bundle_plan_identity is None:
        raise ValueError("registrable smoke bundles require --smoke-plan-identity")
    if bundle_id is not None and (parsed.resume is not None or parsed.unsafe_resume):
        raise ValueError("registrable smoke bundles cannot resume or use unsafe resume")
    if parsed.unsafe_resume:
        if parsed.resume is None:
            raise ValueError("--unsafe-resume requires --resume")
        if not isinstance(parsed.unsafe_resume_reason, str) or not parsed.unsafe_resume_reason.strip():
            raise ValueError("--unsafe-resume requires a nonempty explicit --unsafe-resume-reason")
    elif parsed.unsafe_resume_reason is not None:
        raise ValueError("--unsafe-resume-reason may only be used with --unsafe-resume")
    boundary = None
    if bundle_id is None:
        from spritelab.training.launch import bootstrap_validated_training_process

        boundary = bootstrap_validated_training_process(parsed.config, parsed.resume)
        if boundary is not None and (parsed.smoke or parsed.unsafe_resume or parsed.unsafe_resume_reason is not None):
            raise ValueError("validated campaign process boundaries forbid smoke and unsafe-resume overrides")
    bundle_plan = None
    bundle_expected_manifest = None
    bundle_config_sha256 = None
    qualification = None
    if bundle_id is not None:
        import os

        from spritelab.training.smoke_bundle import (
            expected_manifest,
            load_plan,
            smoke_launch_identity,
            validate_cli_configuration,
            validate_smoke_environment,
            validate_smoke_interpreter,
            verify_execution_guards,
        )

        project_root = Path.cwd().resolve()
        bundle_plan = load_plan(project_root, bundle_id)
        if bundle_plan_identity != bundle_plan["plan_identity"]:
            raise ValueError("the contained smoke plan identity changed")
        if bundle_launch_identity != smoke_launch_identity(bundle_plan, str(bundle_device)):
            raise ValueError("the contained smoke launch identity changed")
        validate_smoke_interpreter(
            bundle_plan,
            lexical_path=os.environ.get("SPRITELAB_BOUND_INTERPRETER"),
        )
        validate_smoke_environment(project_root, bundle_plan, str(bundle_device), os.environ)
        bundle_config_sha256, config = validate_cli_configuration(
            project_root,
            bundle_plan,
            str(bundle_device),
            parsed.config,
        )
        verify_execution_guards(project_root, bundle_plan)
        bundle_expected_manifest = expected_manifest(project_root, bundle_plan, str(bundle_device))
        if not isinstance(bundle_expected_manifest.get("software_version"), Mapping):
            raise ValueError("the server-prepared smoke software version is missing or malformed")
        if not isinstance(bundle_expected_manifest.get("hardware_summary"), Mapping):
            raise ValueError("the server-prepared smoke hardware summary is missing or malformed")
    else:
        if boundary is None:
            from spritelab.training.experiment_system import load_config

            config = load_config(parsed.config)
        else:
            config = dict(boundary.config)
    runtime = config["runtime"]
    smoke_runtime = (
        {
            "max_steps": min(2, int(runtime["max_steps"])),
            "batch_size": min(2, int(runtime["batch_size"])),
            "micro_batch_size": min(2, int(runtime.get("micro_batch_size", runtime["batch_size"]))),
            "global_batch_size": min(2, int(runtime["batch_size"])),
            "effective_batch_size": min(2, int(runtime["batch_size"]))
            * int(runtime.get("gradient_accumulation_steps", 1)),
            "sample_every": 0,
            "save_every": 1,
        }
        if parsed.smoke
        else None
    )
    manifest = _prepare_manifest(
        parsed.config,
        write=bundle_id is None and boundary is None,
        config=config,
        runtime_overrides=smoke_runtime,
        resolution_root=(
            Path.cwd().resolve() if bundle_id is not None else (boundary.project_root if boundary is not None else None)
        ),
        training_manifest_bytes=None if boundary is None else boundary.training_manifest_bytes,
        vocabulary_bytes=None if boundary is None else boundary.vocabulary_bytes,
        software_version_override=(
            None if bundle_expected_manifest is None else bundle_expected_manifest["software_version"]
        ),
        hardware_summary_override=(
            None if bundle_expected_manifest is None else bundle_expected_manifest["hardware_summary"]
        ),
    )
    if bundle_id is not None:
        from spritelab.training.smoke_bundle import (
            begin_device_run,
            run_bundle_directory,
        )

        project_root = Path.cwd().resolve()
        if bundle_plan is None or bundle_config_sha256 is None:
            raise ValueError("the server-prepared smoke plan was not retained safely")
        if bundle_expected_manifest is None:
            raise ValueError("the server-prepared smoke manifest is missing")
        _require_exact_smoke_manifest(manifest, bundle_expected_manifest)
        out_dir = begin_device_run(project_root, bundle_plan, str(bundle_device))
        expected_output = run_bundle_directory(project_root, bundle_id) / str(bundle_device)
        if out_dir != expected_output:
            raise ValueError("the fixed smoke output identity changed")
        if bundle_device == "cuda":
            from spritelab.training.determinism import qualify_determinism

            qualification = qualify_determinism(mode="strict", device="cuda", steps=2)
    elif boundary is not None:
        out_dir = boundary.output_root
    else:
        out_dir = _resolve(parsed.config, runtime["out_dir"])
    logical_out_dir = boundary.logical_output_root if boundary is not None else out_dir
    dataset, model = config["dataset"], config["model"]
    conditioning, loss = config["conditioning"], config["loss"]
    optimizer, augmentation = config["optimizer"], config["augmentation"]
    lr_schedule, lr_warmup_steps = _authoritative_lr_schedule(config)
    ema = config["ema"]
    if not parsed.resume and out_dir.exists() and any(out_dir.glob("checkpoint*.pt")):
        raise FileExistsError(f"refusing to overwrite checkpoints in {out_dir}; choose a new run directory or resume")
    max_steps = min(2, int(runtime["max_steps"])) if parsed.smoke else int(runtime["max_steps"])
    resolver = (
        (lambda value: _resolve(parsed.config, value))
        if boundary is None
        else (lambda value: _resolve_from_root(boundary.project_root, value))
    )
    kwargs = {
        "dataset_dir": resolver(dataset["directory"]),
        "training_manifest": resolver(dataset["training_manifest"]),
        "out_dir": logical_out_dir,
        "retained_output_root": None if boundary is None else out_dir,
        "campaign_run_contract": None if boundary is None else boundary.campaign_run_contract,
        "retained_training_manifest_records": (
            None
            if boundary is None
            else tuple(
                json.loads(line)
                for line in boundary.training_manifest_bytes.decode("utf-8").splitlines()
                if line.strip()
            )
        ),
        "retained_dataset_descriptors": None if boundary is None else boundary.dataset_descriptors,
        "retained_dataset_content_sha256": None if boundary is None else boundary.dataset_content_sha256,
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
        "lr_schedule": lr_schedule,
        "lr_warmup_steps": lr_warmup_steps,
        "film_conditioning": bool(model.get("film_conditioning", False)),
        "bottleneck_attention": bool(model.get("bottleneck_attention", False)),
        "palette_conditioning": bool(conditioning.get("palette", {}).get("enabled", False)),
        "palette_conditioning_dropout": float(conditioning.get("palette", {}).get("dropout", 0.0)),
        "palette_conditioning_dim": int(model.get("palette_conditioning_dim", 64)),
        "palette_conditioning_inject": str(model.get("palette_conditioning_inject", "decoder")),
        "auxiliary_heads_mode": model.get("auxiliary_heads_mode"),
        "gradient_accumulation_steps": int(runtime.get("gradient_accumulation_steps", 1)),
        "caption_max_length": int(conditioning.get("caption_max_length", 32)),
        "semantic_max_length": int(conditioning.get("semantic_max_length", 48)),
        "experiment_manifest": manifest,
        "resume_from": parsed.resume if boundary is None else boundary.resume_path,
        "resume_descriptor": None if boundary is None else boundary.resume_descriptor,
        "expected_resume_sha256": None if boundary is None else boundary.resume_sha256,
        "unsafe_resume": bool(parsed.unsafe_resume),
        "unsafe_resume_reason": parsed.unsafe_resume_reason,
        "determinism": str(runtime.get("determinism", "off")),
        "timestep_validation_boundaries": tuple(
            float(value) for value in runtime.get("timestep_validation_boundaries", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ),
    }
    from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training

    report = run_challenger_training(ChallengerTrainConfig(**kwargs))
    if bundle_id is not None:
        from spritelab.training.smoke_bundle import (
            canonical_json_bytes,
            read_stable_single_link_bytes,
            verify_execution_guards,
            write_device_receipt,
            write_exclusive_bytes,
        )

        if bundle_plan is None or bundle_config_sha256 is None:
            raise ValueError("the server-prepared smoke identity was not retained safely")
        project_root = Path.cwd().resolve()
        verify_execution_guards(project_root, bundle_plan)
        if qualification is not None:
            write_exclusive_bytes(
                out_dir / "cuda_determinism_qualification.json",
                canonical_json_bytes(qualification, pretty=True),
                boundary=project_root,
            )
        config_after = read_stable_single_link_bytes(
            parsed.config.resolve(strict=True),
            boundary=project_root,
            max_bytes=16 * 1024 * 1024,
        )
        import hashlib

        receipt = write_device_receipt(
            project_root,
            bundle_plan,
            str(bundle_device),
            config_sha256_before=bundle_config_sha256,
            config_sha256_after=hashlib.sha256(config_after).hexdigest(),
            environment=os.environ,
        )
        verify_execution_guards(project_root, bundle_plan)
        print(
            json.dumps(
                {
                    "device": bundle_device,
                    "receipt_identity": receipt["receipt_identity"],
                    "smoke_id": bundle_id,
                    "status": "COMPLETE",
                    "steps_completed": report["steps_completed"],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {"steps_completed": report["steps_completed"], "out_dir": str(logical_out_dir)},
                sort_keys=True,
            )
        )


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
        campaign_config_path=parsed.campaign,
        project_root=Path.cwd(),
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


def _resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _set_dotted(data: dict, dotted: str, value: object) -> None:
    target = data
    keys = dotted.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value

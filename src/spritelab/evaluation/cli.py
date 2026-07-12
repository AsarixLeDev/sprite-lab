"""CLI for generation benchmark v1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from spritelab.evaluation.suite import compare_reports, human_package, score_suite, write_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spritelab eval", description="Reproducible sprite generation benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-suite", help="Generate frozen prompts with paired noise seeds.")
    generate.add_argument("--suite", required=True, type=Path)
    generate.add_argument(
        "--checkpoint", required=True, action="append", help="LABEL=PATH; repeat for paired checkpoints."
    )
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--seeds", help="Comma-separated override; defaults to frozen suite seeds.")
    generate.add_argument("--cfg-scale", type=float, action="append", default=[])
    generate.add_argument("--steps", type=int, action="append", default=[])
    generate.add_argument("--export-preset", action="append", default=[])
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--batch-size", type=int, default=16)
    generate.add_argument("--max-samples", type=int, default=0)
    generate.add_argument("--palette-conditioning-source", choices=("none", "source", "retrieved"), default="none")
    generate.add_argument("--palette-conditioning-dataset", type=Path)
    generate.add_argument("--palette-conditioning-training-manifest", type=Path)
    generate.add_argument("--palette-conditioning-exclude-exact-target", action="store_true")
    generate.add_argument(
        "--allow-legacy-conditioning-v1",
        action="store_true",
        help="Explicitly use the recorded schema-v1 structured vocabulary without remapping IDs.",
    )
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.set_defaults(func=_generate)

    score = sub.add_parser(
        "score-suite", help="Score generated artifacts on CPU and optionally retrieve train neighbors."
    )
    score.add_argument("--generated", required=True, type=Path)
    score.add_argument("--out", required=True, type=Path)
    score.add_argument("--training-manifest", action="append", type=Path, default=[])
    score.add_argument("--limit", type=int, default=0)
    score.add_argument("--gates", type=Path)
    score.set_defaults(func=_score)

    compare = sub.add_parser("compare", help="Compare two scored suites using paired prompt/noise seeds.")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--out", required=True, type=Path)
    compare.add_argument("--architecture-change", action="store_true")
    compare.set_defaults(func=_compare)

    human = sub.add_parser("human-package", help="Create a static blind A/B package and import schema.")
    human.add_argument("--a", required=True, type=Path)
    human.add_argument("--b", required=True, type=Path)
    human.add_argument("--out", required=True, type=Path)
    human.add_argument("--seed", type=int, default=731001)
    human.add_argument("--mode", choices=("side-by-side", "shuffled"), default="side-by-side")
    human.set_defaults(func=_human)

    review = sub.add_parser("review-memorization", help="Review benchmark v1 exact-alpha training matches in a GUI.")
    review.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/generation_benchmark_v1/baseline_smoke"),
        help="Scored generation benchmark v1 report directory.",
    )
    review.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/generation_benchmark_v1/exact_alpha_review"),
        help="Append-only review output directory.",
    )
    review.set_defaults(func=_review_memorization)
    return parser


def _checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return _slug(path.stem), path
    label, raw = value.split("=", 1)
    return _slug(label), Path(raw)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "run"


def _generate(parsed: argparse.Namespace) -> None:
    from spritelab.training.generator_challenger import ChallengerSampleConfig, run_sample_generator_challenger

    manifest = json.loads(parsed.suite.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if parsed.max_samples:
        cases = cases[: parsed.max_samples]
    prompts = parsed.out / "frozen_prompts.jsonl"
    prompt_rows: list[dict[str, Any]] = []
    for case in cases:
        conditions = dict(case.get("conditions") or {})
        prompt_rows.append(
            {
                **conditions,
                "prompt_id": case["id"],
                "prompt": case["prompt"],
                "conditions": conditions,
                "target_palette": case.get("target_palette"),
                "benchmark_noise_offset": case.get("noise_offset"),
                "trusted_source": case.get("trusted_source"),
            }
        )
    parsed.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompts, prompt_rows)
    seeds = (
        [int(value) for value in parsed.seeds.split(",")]
        if parsed.seeds
        else [int(value) for value in manifest["seeds"]]
    )
    cfg_scales = parsed.cfg_scale or [3.0]
    steps = parsed.steps or [30]
    presets = parsed.export_preset or [""]
    plan: list[dict[str, Any]] = []
    for label, checkpoint in map(_checkpoint, parsed.checkpoint):
        for cfg_scale in cfg_scales:
            for step_count in steps:
                for preset in presets:
                    factored = preset.lower() in {"v1.1", "v1_1", "phase1_v1_1"}
                    for seed in seeds:
                        run = (
                            parsed.out
                            / label
                            / f"cfg_{cfg_scale:g}_steps_{step_count}_preset_{_slug(preset or 'none')}"
                            / f"seed_{seed}"
                        )
                        cell = {
                            "checkpoint": str(checkpoint),
                            "out": str(run),
                            "seed": seed,
                            "noise_seed": seed * 100000,
                            "cfg_scale": cfg_scale,
                            "steps": step_count,
                            "export_preset": preset,
                        }
                        plan.append(cell)
                        if parsed.dry_run or (parsed.resume and (run / "generated_manifest.jsonl").is_file()):
                            continue
                        run_sample_generator_challenger(
                            ChallengerSampleConfig(
                                checkpoint=checkpoint,
                                prompts=prompts,
                                out_dir=run,
                                device=parsed.device,
                                batch_size=parsed.batch_size,
                                max_samples=len(prompt_rows),
                                seed=seed,
                                noise_seed=seed * 100000,
                                cfg_scale=cfg_scale,
                                steps=step_count,
                                export_preset=preset or None,
                                project_palette=True,
                                project_palette_target_colors=16,
                                factored_cfg=factored,
                                cfg_base_scale=2.5 if factored else None,
                                cfg_color_scale=3.0 if factored else None,
                                palette_conditioning_source=parsed.palette_conditioning_source,
                                palette_conditioning_dataset=parsed.palette_conditioning_dataset,
                                palette_conditioning_training_manifest=parsed.palette_conditioning_training_manifest,
                                palette_conditioning_exclude_exact_prompt_target=(
                                    parsed.palette_conditioning_exclude_exact_target
                                ),
                                allow_legacy_conditioning_v1=parsed.allow_legacy_conditioning_v1,
                            )
                        )
    (parsed.out / "generation_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generation cells: {len(plan)}; output: {parsed.out}")


def _score(parsed: argparse.Namespace) -> None:
    report = score_suite(
        parsed.generated,
        parsed.out,
        training_manifests=parsed.training_manifest,
        limit=parsed.limit,
        gates_path=parsed.gates,
    )
    print(f"Scored {report['summary']['sample_count']} samples; promotion pass={report['promotion']['pass']}")


def _compare(parsed: argparse.Namespace) -> None:
    report = compare_reports(
        parsed.baseline, parsed.candidate, parsed.out, architecture_change=parsed.architecture_change
    )
    print(f"Paired outputs: {report['paired_count']}; promotion pass={report['promotion']['pass']}")


def _human(parsed: argparse.Namespace) -> None:
    report = human_package(parsed.a, parsed.b, parsed.out, seed=parsed.seed, mode=parsed.mode)
    print(f"Blind pairs: {report['pair_count']}; output: {parsed.out}")


def _review_memorization(parsed: argparse.Namespace) -> None:
    from spritelab.evaluation.memorization_review import launch_gui, load_review_pairs

    pairs = load_review_pairs(parsed.report)
    print(f"Loaded {len(pairs)} exact-alpha benchmark pairs; output: {parsed.out}")
    launch_gui(pairs, parsed.out)


def main(argv: list[str] | None = None) -> None:
    parsed = _parser().parse_args(argv)
    parsed.func(parsed)


if __name__ == "__main__":
    main()

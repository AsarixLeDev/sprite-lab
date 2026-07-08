"""Generate, canonicalize, and export prompt-conditioned sprite samples."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.eval_baseline import resolve_device
from spritelab.training.eval_generator import _load_checkpoint, _tokenizer_from_checkpoint
from spritelab.training.conditioning import (
    apply_conditioning_mode,
    checkpoint_conditioning_mode,
    checkpoint_semantic_max_length,
)
from spritelab.training.generated_canonicalizer import (
    build_generation_contact_sheet,
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.generator_models import TinyCaptionSpriteGenerator


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab generator sampling.")
    return torch


@dataclass(frozen=True)
class SampleGeneratorConfig:
    checkpoint: Path
    prompts: Path
    out_dir: Path
    max_samples: int = 64
    max_colors: int = 32
    alpha_threshold: float = 0.5
    device: str = "cpu"
    seed: int = 123
    noise_seed: int | None = None
    dither: bool = False
    write_raw_rgba: bool = True
    write_hard_rgba: bool = True
    batch_size: int = 16
    contact_sheet_labels: str = "prompt"
    project_palette: bool = False
    project_palette_target_colors: int = 16
    project_palette_min_pixel_share: float = 0.01
    project_palette_method: str = "deterministic_kmeans"


def run_sample_generator(config: SampleGeneratorConfig) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    _set_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    ckpt = _load_checkpoint(config.checkpoint)
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    conditioning_mode = checkpoint_conditioning_mode(ckpt)
    semantic_max_length = checkpoint_semantic_max_length(ckpt)
    model = TinyCaptionSpriteGenerator(**dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
    manifest_records: list[dict[str, Any]] = []
    base_noise_seed = int(config.noise_seed) if config.noise_seed is not None else int(config.seed) * 100000

    for batch_start in range(0, len(prompts), max(1, int(config.batch_size))):
        batch_records = prompts[batch_start : batch_start + max(1, int(config.batch_size))]
        if not batch_records:
            continue
        noise_seeds = [base_noise_seed + batch_start + index for index in range(len(batch_records))]
        caption_tokens = th.as_tensor(
            [tokenizer.encode(str(record["prompt"]), max_length=tokenizer.max_length) for record in batch_records],
            dtype=th.long,
            device=device,
        )
        semantic_tokens = th.as_tensor(
            [
                tokenizer.encode_record_semantics(record, max_length=semantic_max_length)
                for record in batch_records
            ],
            dtype=th.long,
            device=device,
        )
        noise = th.cat(
            [model.sample_noise(1, device=device, seed=noise_seed) for noise_seed in noise_seeds],
            dim=0,
        )
        model_inputs = apply_conditioning_mode(
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
        )
        with th.no_grad():
            outputs = model(**model_inputs, noise=noise)
        rgba_batch = _outputs_to_rgba(outputs)

        for item_index, prompt_record in enumerate(batch_records):
            sample_index = batch_start + item_index
            sample_id = f"sample_{sample_index:06d}"
            noise_seed = noise_seeds[item_index]
            sprite = canonicalize_generated_rgba(
                rgba_batch[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
                dither=config.dither,
            )
            metadata = {
                **prompt_record,
                "checkpoint": str(config.checkpoint),
                "seed": int(config.seed),
                "noise_seed": int(noise_seed),
                "conditioning_mode": conditioning_mode,
                "alpha_threshold": float(config.alpha_threshold),
                "max_colors": int(config.max_colors),
                "dither": bool(config.dither),
            }
            record = write_generated_sprite_artifacts(
                sprite,
                out_dir,
                sample_id,
                metadata,
                write_raw_rgba=config.write_raw_rgba,
                write_hard_rgba=config.write_hard_rgba,
            )
            if config.project_palette:
                from spritelab.training.palette_projection import project_generated_sprite_record

                record = project_generated_sprite_record(
                    sprite,
                    out_dir,
                    record,
                    target_colors=config.project_palette_target_colors,
                    min_pixel_share=config.project_palette_min_pixel_share,
                    alpha_threshold=config.alpha_threshold,
                    method=config.project_palette_method,
                )
            manifest_records.append(record)

    contact_sheet_path = build_generation_contact_sheet(
        out_dir,
        manifest_records,
        out_dir / "generation_contact_sheet.png",
        include_raw=config.write_raw_rgba,
    )
    _write_contact_sheet_label_mapping(out_dir, manifest_records, label_mode=config.contact_sheet_labels)
    projection_report = None
    if config.project_palette:
        from spritelab.training.palette_projection import write_runtime_projection_report

        projection_report = write_runtime_projection_report(
            out_dir,
            manifest_records,
            target_colors=config.project_palette_target_colors,
            min_pixel_share=config.project_palette_min_pixel_share,
            alpha_threshold=config.alpha_threshold,
            method=config.project_palette_method,
        )
    config_json = {key: _jsonable(value) for key, value in asdict(config).items()}
    if not config.project_palette:
        for key in (
            "project_palette",
            "project_palette_target_colors",
            "project_palette_min_pixel_share",
            "project_palette_method",
        ):
            config_json.pop(key, None)
    report = write_generation_reports(
        out_dir=out_dir,
        records=manifest_records,
        config={
            **config_json,
            "device_resolved": str(device),
            "conditioning_mode": conditioning_mode,
            "semantic_max_length": semantic_max_length,
            "elapsed_seconds": time.perf_counter() - started,
        },
        contact_sheet=None if contact_sheet_path is None else contact_sheet_path.name,
    )
    if projection_report is not None:
        report["palette_projection"] = {
            "applied": True,
            "report": "palette_projection_report.json",
            "contact_sheet": "contact_sheet_projected.png"
            if (out_dir / "contact_sheet_projected.png").is_file()
            else None,
            "method": str(config.project_palette_method),
            "target_colors": int(config.project_palette_target_colors),
            "min_pixel_share": float(config.project_palette_min_pixel_share),
            "mean_rgb_mae_visible": projection_report.get("mean_rgb_mae_visible"),
            "destructive_rate": projection_report.get("destructive_rate"),
        }
        (out_dir / "generation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def read_prompt_records(path: str | Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    """Read eval prompt JSONL while preserving metadata fields."""

    path = Path(path)
    if max_records is not None and int(max_records) <= 0:
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if isinstance(value, str):
            record = {"prompt": value, "prompt_id": f"prompt_{line_no:04d}"}
        elif isinstance(value, dict):
            record = dict(value)
            record["prompt"] = str(record.get("prompt") or record.get("caption") or "")
            record.setdefault("prompt_id", f"prompt_{line_no:04d}")
        else:
            raise ValueError(f"{path}:{line_no}: expected JSON object or string")
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"{path}:{line_no}: prompt is empty")
        records.append(record)
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


def _outputs_to_rgba(outputs: dict[str, Any]) -> np.ndarray:
    rgb_logits = outputs["rgb_logits"].detach().cpu().numpy().astype(np.float32)
    alpha_logits = outputs["alpha_logits"].detach().cpu().numpy().astype(np.float32)
    rgb = _sigmoid(rgb_logits)
    alpha = _sigmoid(alpha_logits)
    rgba_chw = np.concatenate([rgb, alpha], axis=1)
    return np.moveaxis(rgba_chw, 1, -1).astype(np.float32, copy=False)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def _set_seed(seed: int) -> None:
    th = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_contact_sheet_label_mapping(out_dir: Path, records: list[dict[str, Any]], *, label_mode: str) -> None:
    rows = []
    for record in records:
        paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "sample_filename": paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba"),
                "prompt": record.get("prompt"),
                "prompt_id": record.get("prompt_id"),
                "seed": record.get("seed"),
                "noise_seed": record.get("noise_seed"),
                "conditioning": record.get("conditioning_mode"),
                "label_mode": label_mode,
                "nearest_source_object": record.get("nearest_source_object"),
                "nearest_source_category": record.get("nearest_source_category"),
            }
        )
    (out_dir / "contact_sheet_labels.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Contact Sheet Labels", ""]
    for row in rows:
        lines.append(
            f"- `{row.get('sample_id')}` `{row.get('prompt_id')}` seed={row.get('noise_seed')}: {row.get('prompt')}"
        )
    lines.append("")
    (out_dir / "contact_sheet_labels.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sample and canonicalize a caption-conditioned RGBA generator.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-colors", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--dither", action="store_true", default=False)
    parser.add_argument("--no-dither", action="store_false", dest="dither")
    parser.add_argument("--write-raw-rgba", action="store_true", dest="write_raw_rgba", default=True)
    parser.add_argument("--no-write-raw-rgba", action="store_false", dest="write_raw_rgba")
    parser.add_argument("--write-hard-rgba", action="store_true", dest="write_hard_rgba", default=True)
    parser.add_argument("--no-write-hard-rgba", action="store_false", dest="write_hard_rgba")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--contact-sheet-labels",
        choices=["prompt", "prompt_and_seed", "prompt_and_nearest_source"],
        default="prompt",
    )
    parser.add_argument("--project-palette", action="store_true", default=False)
    parser.add_argument("--project-palette-target-colors", type=int, default=16)
    parser.add_argument("--project-palette-min-pixel-share", type=float, default=0.01)
    parser.add_argument("--project-palette-method", choices=["deterministic_kmeans"], default="deterministic_kmeans")
    parsed = parser.parse_args(argv)
    report = run_sample_generator(SampleGeneratorConfig(**vars(parsed)))
    print(f"Generated samples: {report['sample_count']}")
    print(f"Max visible colors: {report['max_visible_color_count']}")
    print(f"Outputs written to {parsed.out_dir}")

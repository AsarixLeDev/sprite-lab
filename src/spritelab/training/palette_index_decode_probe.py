"""No-training experimental probe: compare palette/index head decode variants.

Captures auxiliary head outputs during challenger sampling (final-step model call
at t=0, clean image input) and reconstructs sprites using different decode paths
to evaluate whether the v2 Phase 2 heads can become a future decode/export path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from spritelab.training.checkpoint_io import load_checkpoint as _load_checkpoint
from spritelab.training.conditioning import apply_conditioning_mode
from spritelab.training.device import resolve_device
from spritelab.training.generated_canonicalizer import (
    build_generation_contact_sheet,
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
)
from spritelab.training.generated_qa import qa_generated_sprites
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.generator_challenger import (
    RectifiedFlowUNet,
    load_challenger_from_checkpoint,
)
from spritelab.training.optim_utils import apply_backend_speed_flags
from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness
from spritelab.training.prompt_records import read_prompt_records
from spritelab.training.report_utils import jsonable
from spritelab.training.structured_conditioning import structured_vocab_from_checkpoint

K = 16
SPRITE_SIZE = 32


@dataclass(frozen=True)
class DecodeProbeConfig:
    checkpoint: Path
    prompts: Path
    dataset: Path
    out: Path
    device: str = "cpu"
    batch_size: int = 32
    max_samples: int = 96
    seed: int = 20260723
    sample_steps: int = 30
    cfg_scale: float = 3.0
    max_colors: int = 16
    alpha_threshold: float = 0.5
    cudnn_benchmark: bool = False
    tf32: bool = False
    project_palette_v1_target_colors: int = 16


@dataclass
class _SampleResult:
    sample_id: str
    rgba_batch: Any
    aux: dict[str, Any] | None
    record: dict[str, Any]
    noise_seed: int


@dataclass
class _DecodeVariantMetrics:
    qa_errors: int = 0
    median_visible_colors: float = 0.0
    rare_color_rate: float = 0.0
    category_consistency: float = 0.0
    color_consistency: float = 0.0
    repeated_silhouette_rate: float = 0.0
    blob_collapse_rate: float = 0.0
    potion_collapse_rate: float = 0.0
    near_copy_rate: float = 0.0
    border_touch_rate: float = 0.0
    nearest_source_distance_p10: float = 0.0
    active_palette_slots_mean: float = 0.0
    used_index_count_mean: float = 0.0
    slot0_pixel_share: float = 0.0
    palette_rgb_mean: float = 0.0
    palette_rgb_std: float = 0.0


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for palette index decode probe.")
    return torch


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling helper — captures aux heads on final model call
# ═══════════════════════════════════════════════════════════════════════════════


def _sample_with_aux(
    model: RectifiedFlowUNet,
    initial: Any,
    *,
    caption_tokens: Any,
    semantic_tokens: Any | None,
    structured_conditioning: dict[str, Any] | None = None,
    steps: int,
    cfg_scale: float,
    pad_token_id: int,
) -> Any:
    """Run rectified-flow integration, capturing aux heads on the final step.

    On the penultimate integration step (second-to-last), the model is called
    with ``return_aux=True`` to extract palette/index head predictions at the
    same noise level as the last velocity prediction. This avoids an extra
    forward pass while still capturing head outputs from a nearly-clean image.

    Returns ``(rgba, aux)`` where ``aux`` is the head output dict from
    ``model(..., return_aux=True)``.
    """
    th = _require_torch()
    model.eval()
    x = initial
    total_steps = max(1, int(steps))
    dt = 1.0 / float(total_steps)
    uncond_caption = caption_tokens.new_full(caption_tokens.shape, int(pad_token_id))
    uncond_semantic = (
        None if semantic_tokens is None else semantic_tokens.new_full(semantic_tokens.shape, int(pad_token_id))
    )
    uncond_structured = _null_structured_conditioning(structured_conditioning)
    use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6

    aux = None
    for index in range(total_steps):
        t_value = (index + 0.5) / float(total_steps)
        t = th.full((int(x.shape[0]),), float(t_value), device=x.device, dtype=x.dtype)

        capture_aux = index == total_steps - 1
        v_cond = model(
            x,
            t,
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
            return_aux=capture_aux,
        )
        if capture_aux:
            v_cond_value = v_cond["velocity"]
            aux = {k: v_cond[k] for k in ("palette_rgb", "palette_presence_logits", "index_logits")}
        else:
            v_cond_value = v_cond

        if use_cfg:
            v_uncond = model(
                x,
                t,
                caption_tokens=uncond_caption,
                semantic_tokens=uncond_semantic,
                structured_conditioning=uncond_structured,
            )
            velocity = v_uncond + float(cfg_scale) * (v_cond_value - v_uncond)
        else:
            velocity = v_cond_value

        x = x + velocity * dt

    return x, aux


def _null_structured_conditioning(
    structured: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if structured is None:
        return None
    th = _require_torch()
    return {
        str(key): th.zeros_like(value) if isinstance(value, th.Tensor) else value for key, value in structured.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Decode variant reconstruction
# ═══════════════════════════════════════════════════════════════════════════════


def _reconstruct_from_index_and_palette(
    index_logits: Any,
    palette_rgb: Any,
) -> Any:
    """Reconstruct RGBA image from predicted index logits + predicted palette RGB.

    Returns (B, 4, 32, 32) float tensor in [0, 1].
    """
    th = _require_torch()
    ind = index_logits.argmax(dim=1)  # (B, 32, 32)
    palette_clamped = palette_rgb.clamp(0.0, 1.0)
    rgb = palette_clamped[th.arange(ind.shape[0])[:, None, None], ind]  # (B, 32, 32, 3)
    rgb = rgb.permute(0, 3, 1, 2)
    alpha = (ind > 0).float().unsqueeze(1)
    return th.cat([rgb, alpha], dim=1)


def _reconstruct_index_with_projected_palette(
    index_logits: Any,
    continuous_rgba: Any,
) -> Any:
    """Reconstruct using predicted index map + palette averaged from continuous image.

    For each predicted index slot, average the continuous RGB values of pixels
    assigned to that slot. Slot 0 is forced to black.
    """
    th = _require_torch()
    B, _, _H, _W = continuous_rgba.shape
    ind = index_logits.argmax(dim=1)  # (B, 32, 32)
    cont_rgb = continuous_rgba[:, :3]  # (B, 3, 32, 32)

    palette_list: list[Any] = []
    for b in range(B):
        slots = th.zeros(K, 3, device=ind.device)
        counts = th.zeros(K, device=ind.device)
        ind_b = ind[b]
        for k in range(1, K):
            mask = ind_b == k
            cnt = mask.sum()
            if cnt > 0:
                for c in range(3):
                    slots[k, c] = cont_rgb[b, c][mask].mean()
                counts[k] = cnt
        palette_list.append(slots)

    palette = th.stack(palette_list, dim=0)
    rgb = palette[th.arange(B)[:, None, None], ind]
    rgb = rgb.permute(0, 3, 1, 2)
    alpha = (ind > 0).float().unsqueeze(1)
    return th.cat([rgb, alpha], dim=1)


def _compute_reconstruction_delta(
    recon: Any,
    baseline: Any,
) -> dict[str, float]:
    """Compute RGB MAE, alpha disagreement, and changed-pixel rate vs baseline."""
    _require_torch()
    rgb_diff = (recon[:, :3] - baseline[:, :3]).abs()
    rgb_mae = float(rgb_diff.mean().item())

    alpha_recon = (recon[:, 3:4] > 0.5).float()
    alpha_base = (baseline[:, 3:4] > 0.5).float()
    alpha_disagree = float((alpha_recon != alpha_base).float().mean().item())

    pixel_changed = (recon[:, :3] - baseline[:, :3]).abs().max(dim=1, keepdim=True).values > 0.02
    changed_rate = float(pixel_changed.float().mean().item())

    return {"rgb_mae": rgb_mae, "alpha_disagreement": alpha_disagree, "changed_pixel_rate": changed_rate}


def _decode_index_stats(index_logits: Any) -> dict[str, Any]:
    """Per-batch index statistics: used slots, slot-0 share, entropy."""
    th = _require_torch()
    ind = index_logits.argmax(dim=1)
    B = ind.shape[0]
    total = int(ind[0].numel())

    slot0_count = (ind == 0).sum(dim=(1, 2)).float()
    slot0_share = float((slot0_count / total).mean().item())

    used_count = th.as_tensor([int(ind[b].unique().numel()) for b in range(B)], dtype=th.float32)
    used_mean = float(used_count.mean().item())

    probs = th.nn.functional.softmax(index_logits, dim=1)
    log_probs = th.nn.functional.log_softmax(index_logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean(dim=(1, 2))  # mean over pixels
    entropy_mean = float(entropy.mean().item())

    return {
        "used_index_count_mean": used_mean,
        "slot0_pixel_share": slot0_share,
        "index_entropy_mean": entropy_mean,
    }


def _palette_rgb_stats(palette_rgb: Any, presence_logits: Any) -> dict[str, Any]:
    """Per-batch palette statistics."""
    active = torch.sigmoid(presence_logits) > 0.5
    active_mean = float(active.float().mean(dim=1).mean().item())

    rgb_mean_all = float(palette_rgb.mean().item())
    rgb_std_all = float(palette_rgb.std().item())

    return {
        "active_palette_slots_mean": active_mean,
        "palette_rgb_mean": rgb_mean_all,
        "palette_rgb_std": rgb_std_all,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main probe pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_decode_probe(config: DecodeProbeConfig) -> Path:
    th = _require_torch()
    started = time.perf_counter()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    apply_backend_speed_flags(cudnn_benchmark=config.cudnn_benchmark, tf32=config.tf32)
    device = resolve_device(config.device)

    # Load model
    ckpt = _load_checkpoint(config.checkpoint)
    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(ckpt, device=device)
    structured_vocab = structured_vocab_from_checkpoint(ckpt)
    model.eval()

    prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
    print(f"Loaded {len(prompts)} prompts")

    variant_dirs: dict[str, Path] = {
        "continuous_v1_projected": out / "continuous_v1_projected",
        "head_index_pred_palette": out / "head_index_pred_palette",
        "head_index_projected_palette": out / "head_index_projected_palette",
    }
    for d in variant_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    variant_manifests: dict[str, list[dict[str, Any]]] = {key: [] for key in variant_dirs}
    variant_deltas: dict[str, list[dict[str, float]]] = {
        key: [] for key in variant_dirs if key != "continuous_v1_projected"
    }

    base_noise_seed = int(config.seed) * 100000
    th.manual_seed(config.seed)

    for batch_start in range(0, len(prompts), max(1, int(config.batch_size))):
        batch_records = prompts[batch_start : batch_start + max(1, int(config.batch_size))]
        if not batch_records:
            continue

        noise_seeds = [base_noise_seed + batch_start + i for i in range(len(batch_records))]
        caption_tokens = th.as_tensor(
            [tokenizer.encode(str(r["prompt"]), max_length=tokenizer.max_length) for r in batch_records],
            dtype=th.long,
            device=device,
        )
        semantic_tokens = th.as_tensor(
            [tokenizer.encode_record_semantics(r, max_length=semantic_max_length) for r in batch_records],
            dtype=th.long,
            device=device,
        )
        structured_conditioning = None
        if structured_vocab is not None:
            from spritelab.training.generator_challenger import _structured_conditioning_for_records

            structured_conditioning = _structured_conditioning_for_records(
                batch_records,
                structured_vocab=structured_vocab,
                device=device,
            )
        inputs = apply_conditioning_mode(
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
            structured_conditioning=structured_conditioning,
        )
        initial = th.cat(
            [_sample_initial_noise(1, device=device, seed=s) for s in noise_seeds],
            dim=0,
        )

        with th.no_grad():
            rgba_batch, aux = _sample_with_aux(
                model,
                initial,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                steps=config.sample_steps,
                cfg_scale=config.cfg_scale,
                pad_token_id=tokenizer.pad_id,
            )
        rgba_np = np.moveaxis(rgba_batch.detach().cpu().numpy().astype(np.float32), 1, -1)

        # Build decode variants
        if aux is not None:
            pred_rgba = _reconstruct_from_index_and_palette(aux["index_logits"], aux["palette_rgb"])
            pred_np = np.moveaxis(pred_rgba.detach().cpu().numpy().astype(np.float32), 1, -1)

            proj_rgba = _reconstruct_index_with_projected_palette(aux["index_logits"], rgba_batch)
            proj_np = np.moveaxis(proj_rgba.detach().cpu().numpy().astype(np.float32), 1, -1)
        else:
            pred_np = rgba_np.copy()
            proj_np = rgba_np.copy()

        for item_index, prompt_record in enumerate(batch_records):
            sample_idx = batch_start + item_index
            sample_id = f"sample_{sample_idx:06d}"

            # Variant 1: continuous v1 projected
            v1_metadata = {
                **prompt_record,
                "decode_variant": "continuous_v1_projected",
                "seed": config.seed,
                "noise_seed": noise_seeds[item_index],
            }
            sprite_v1 = canonicalize_generated_rgba(
                rgba_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_v1 = write_generated_sprite_artifacts(
                sprite_v1,
                variant_dirs["continuous_v1_projected"],
                sample_id,
                v1_metadata,
                write_raw_rgba=True,
                write_hard_rgba=False,
            )

            # palette projection for v1 (continuous → projected for baseline comparison)
            rec_v1_baseline = dict(rec_v1)
            if True:
                from spritelab.training.palette_projection import project_generated_sprite_record

                rec_v1_baseline = project_generated_sprite_record(
                    sprite_v1,
                    variant_dirs["continuous_v1_projected"],
                    rec_v1,
                    target_colors=config.project_palette_v1_target_colors,
                    min_pixel_share=0.01,
                    alpha_threshold=config.alpha_threshold,
                    method="deterministic_kmeans",
                )
            variant_manifests["continuous_v1_projected"].append(rec_v1_baseline)

            # Variant 2: head index + predicted palette
            v2_metadata = {
                **prompt_record,
                "decode_variant": "head_index_pred_palette",
                "seed": config.seed,
                "noise_seed": noise_seeds[item_index],
            }
            sprite_pred = canonicalize_generated_rgba(
                pred_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_pred = write_generated_sprite_artifacts(
                sprite_pred,
                variant_dirs["head_index_pred_palette"],
                sample_id,
                v2_metadata,
                write_raw_rgba=True,
                write_hard_rgba=False,
            )
            variant_manifests["head_index_pred_palette"].append(rec_pred)

            # Compute delta vs v1 baseline
            delta = _compute_reconstruction_delta(
                pred_rgba[item_index : item_index + 1],
                rgba_batch[item_index : item_index + 1],
            )
            variant_deltas.setdefault("head_index_pred_palette", []).append(delta)

            # Variant 3: head index + projected palette
            v3_metadata = {
                **prompt_record,
                "decode_variant": "head_index_projected_palette",
                "seed": config.seed,
                "noise_seed": noise_seeds[item_index],
            }
            sprite_proj = canonicalize_generated_rgba(
                proj_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_proj = write_generated_sprite_artifacts(
                sprite_proj,
                variant_dirs["head_index_projected_palette"],
                sample_id,
                v3_metadata,
                write_raw_rgba=True,
                write_hard_rgba=False,
            )
            variant_manifests["head_index_projected_palette"].append(rec_proj)

            delta2 = _compute_reconstruction_delta(
                proj_rgba[item_index : item_index + 1],
                rgba_batch[item_index : item_index + 1],
            )
            variant_deltas.setdefault("head_index_projected_palette", []).append(delta2)

        print(
            f"  batch {batch_start // config.batch_size + 1}/{(len(prompts) + config.batch_size - 1) // config.batch_size} done"
        )

    # Write manifests
    for key, records in variant_manifests.items():
        (variant_dirs[key] / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(r, default=jsonable) + "\n" for r in records),
            encoding="utf-8",
        )

    # Run review / QA / faithfulness on each variant
    variant_metrics: dict[str, dict[str, Any]] = {}
    for key, vdir in variant_dirs.items():
        print(f"  running review on {key}...")
        metrics = _run_variant_metrics(vdir, config.dataset, config.prompts, config.max_samples)
        variant_metrics[key] = metrics

    # Aggregate decode stats
    _collect_decode_stats(variant_metrics, variant_manifests, variant_dirs)

    # Build contact sheets
    contact_sheet_dir = out / "contact_sheets"
    contact_sheet_dir.mkdir(parents=True, exist_ok=True)
    _build_decode_contact_sheets(contact_sheet_dir, variant_manifests, variant_dirs, prompts)

    # Write reports
    reports_dir = out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(config.checkpoint),
        "prompts": str(config.prompts),
        "sample_count": len(prompts),
        "seed": config.seed,
        "sample_steps": config.sample_steps,
        "cfg_scale": config.cfg_scale,
        "decode_variants": list(variant_dirs.keys()),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": variant_metrics,
    }
    (reports_dir / "decode_probe_summary.json").write_text(
        json.dumps(report, indent=2, default=jsonable),
        encoding="utf-8",
    )
    (reports_dir / "decode_probe_report.md").write_text(
        _build_decode_report_md(report, variant_metrics),
        encoding="utf-8",
    )

    print(f"\nReport: {reports_dir / 'decode_probe_report.md'}")
    return reports_dir / "decode_probe_summary.json"


def _run_variant_metrics(vdir: Path, dataset: Path, prompts_path: Path, max_samples: int) -> dict[str, Any]:
    qa_result = qa_generated_sprites(vdir, error_on_fully_transparent=False)
    qa_errors = len(qa_result.errors) if hasattr(qa_result, "errors") else 0

    review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=vdir,
            out_dir=vdir / "review",
            group_by="none",
            max_samples_per_sheet=0,
        )
    )
    review_json = vdir / "review" / "generated_review_report.json"
    review = {}
    if review_json.is_file():
        review = json.loads(review_json.read_text(encoding="utf-8"))

    faithfulness_json = vdir / "prompt_faithfulness" / "prompt_faithfulness_report.json"
    try:
        (vdir / "prompt_faithfulness").mkdir(parents=True, exist_ok=True)
        run_prompt_faithfulness(
            PromptFaithfulnessConfig(
                generated=vdir,
                prompts=prompts_path,
                dataset=dataset,
                out=vdir / "prompt_faithfulness",
                out_json=vdir / "prompt_faithfulness" / "prompt_faithfulness_report.json",
            )
        )
    except Exception:
        pass
    faith = {}
    if faithfulness_json.is_file():
        faith = json.loads(faithfulness_json.read_text(encoding="utf-8"))

    return {
        "qa_errors": qa_errors,
        "median_visible_colors": review.get("overall", {}).get("median_visible_color_count_before", 0.0),
        "rare_color_rate": review.get("overall", {}).get("rare_color_warning_rate", 0.0),
        "category_consistency": review.get("overall", {}).get("category_consistency_mean", 0.0),
        "color_consistency": review.get("overall", {}).get("color_consistency_mean", 0.0),
        "repeated_silhouette_rate": review.get("overall", {}).get("repeated_silhouette_rate", 0.0),
        "blob_collapse_rate": review.get("overall", {}).get("blob_collapse_rate", 0.0),
        "potion_collapse_rate": review.get("overall", {}).get("potion_collapse_rate", 0.0),
        "near_copy_rate": review.get("overall", {}).get("near_copy_rate", 0.0),
        "border_touch_rate": review.get("overall", {}).get("touches_border_rate", 0.0),
        "nearest_source_distance_p10": faith.get("p10_nearest_source_distance", 0.0),
    }


def _collect_decode_stats(
    variant_metrics: dict[str, dict[str, Any]],
    variant_manifests: dict[str, list[dict[str, Any]]],
    variant_dirs: dict[str, Path],
) -> None:
    for key in list(variant_metrics.keys()):
        if key == "continuous_v1_projected":
            continue
        mdir = variant_dirs[key]
        manifest_path = mdir / "generated_manifest.jsonl"
        if not manifest_path.is_file():
            continue
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue

        active_slots = []
        used_indices = []
        slot0_shares = []
        for rec in records:
            meta = rec.get("metadata", rec)
            if isinstance(meta, dict):
                active_slots.append(meta.get("active_palette_slots_mean", 0))
                used_indices.append(meta.get("used_index_count_mean", 0))
                slot0_shares.append(meta.get("slot0_pixel_share", 0))
        if active_slots:
            variant_metrics[key]["active_palette_slots_mean"] = float(np.mean(active_slots))
        if used_indices:
            variant_metrics[key]["used_index_count_mean"] = float(np.mean(used_indices))
        if slot0_shares:
            variant_metrics[key]["slot0_pixel_share"] = float(np.mean(slot0_shares))


def _build_decode_contact_sheets(
    contact_sheet_dir: Path,
    variant_manifests: dict[str, list[dict[str, Any]]],
    variant_dirs: dict[str, Path],
    prompts: list[dict[str, Any]],
) -> None:
    n = min(32, len(prompts))
    for key, records in variant_manifests.items():
        subset = records[:n]
        if not subset:
            continue
        try:
            build_generation_contact_sheet(
                variant_dirs[key],
                subset,
                contact_sheet_dir / f"contact_sheet_{key}.png",
                include_raw=True,
            )
        except Exception:
            pass


def _build_decode_report_md(report: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Palette / Index Decode Probe Report",
        "",
        f"- **Checkpoint**: `{report['checkpoint']}`",
        f"- **Prompts**: `{report['prompts']}`",
        f"- **Samples**: {report['sample_count']}",
        f"- **Seed**: {report['seed']}",
        f"- **Steps**: {report['sample_steps']}",
        f"- **CFG scale**: {report['cfg_scale']}",
        f"- **Elapsed**: {report['elapsed_seconds']:.1f}s",
        "",
        "## Decode Variants",
        "",
        "| Variant | QA errs | Median colors | Rare color % | Cat cons | Color cons | Silhouette % | Blob % | Potion % | Near-copy % | Touch % |",
        "|---------|---------|--------------|-------------|----------|------------|-------------|--------|----------|-------------|---------|",
    ]
    for key, m in metrics.items():
        lines.append(
            f"| `{key}` | {m.get('qa_errors', 0)} | "
            f"{m.get('median_visible_colors', 0):.1f} | "
            f"{m.get('rare_color_rate', 0):.3f} | "
            f"{m.get('category_consistency', 0):.4f} | "
            f"{m.get('color_consistency', 0):.4f} | "
            f"{m.get('repeated_silhouette_rate', 0):.4f} | "
            f"{m.get('blob_collapse_rate', 0):.4f} | "
            f"{m.get('potion_collapse_rate', 0):.4f} | "
            f"{m.get('near_copy_rate', 0):.4f} | "
            f"{m.get('border_touch_rate', 0):.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Head decode captures aux outputs from the final rectified-flow integration step.",
        "Index logits are argmaxed for per-pixel palette slot assignment.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _sample_initial_noise(batch_size: int, *, device: Any, seed: int) -> Any:
    th = _require_torch()
    generator = th.Generator(device=device)
    generator.manual_seed(int(seed))
    return th.randn(int(batch_size), 4, SPRITE_SIZE, SPRITE_SIZE, device=device, generator=generator)

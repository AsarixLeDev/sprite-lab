"""No-training experimental probe: compare palette/index head decode variants.

Captures auxiliary head outputs on a separate post-sampling model call at t=0
(clean generated image) to avoid any risk of destabilising the normal
``integrate_rectified_flow`` path.

Three decode variants are compared:
  1. ``continuous_v1_projected`` — normal v1 path (baseline)
  2. ``head_index_pred_palette`` — predicted index + predicted palette, alpha from continuous image
  3. ``head_index_projected_palette`` — predicted index + averaged palette, alpha from continuous image
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
    write_generation_reports,
)
from spritelab.training.generated_qa import qa_generated_sprites
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.generator_challenger import (
    integrate_rectified_flow,
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


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for palette index decode probe.")
    return torch


# ═══════════════════════════════════════════════════════════════════════════════
# Decode variant reconstruction
# ═══════════════════════════════════════════════════════════════════════════════


def _reconstruct_from_index_and_palette(
    index_logits: Any,
    palette_rgb: Any,
    *,
    alpha_mask: Any,
) -> Any:
    """Reconstruct RGBA from predicted index + palette, using *alpha_mask* for alpha.

    ``alpha_mask`` is a (B, 32, 32) float tensor where > 0.5 means visible.
    Slot-0 (predicted) pixels get black RGB regardless of mask.
    Returns (B, 4, 32, 32) float tensor in [0, 1].
    """
    th = _require_torch()
    ind = index_logits.argmax(dim=1)
    palette_clamped = palette_rgb.clamp(0.0, 1.0)
    rgb = palette_clamped[th.arange(ind.shape[0])[:, None, None], ind]
    rgb = rgb.permute(0, 3, 1, 2)
    # Slot 0 → transparent/black
    rgb = th.where((ind == 0).unsqueeze(1), th.zeros_like(rgb), rgb)
    vis = (alpha_mask > 0.5).float().unsqueeze(1)
    rgb = rgb * vis
    alpha = vis
    return th.cat([rgb, alpha], dim=1)


def _reconstruct_index_with_projected_palette(
    index_logits: Any,
    continuous_rgba: Any,
    *,
    alpha_mask: Any,
) -> Any:
    """Reconstruct using predicted index + palette averaged from continuous image."""
    th = _require_torch()
    B = continuous_rgba.shape[0]
    ind = index_logits.argmax(dim=1)
    cont_rgb = continuous_rgba[:, :3]

    palette_list: list[Any] = []
    for b in range(B):
        slots = th.zeros(K, 3, device=ind.device)
        ind_b = ind[b]
        for k in range(1, K):
            mask = ind_b == k
            cnt = mask.sum()
            if cnt > 0:
                for c in range(3):
                    slots[k, c] = cont_rgb[b, c][mask].mean()
        palette_list.append(slots)

    palette = th.stack(palette_list, dim=0)
    rgb = palette[th.arange(B)[:, None, None], ind]
    rgb = rgb.permute(0, 3, 1, 2)
    rgb = th.where((ind == 0).unsqueeze(1), th.zeros_like(rgb), rgb)
    vis = (alpha_mask > 0.5).float().unsqueeze(1)
    rgb = rgb * vis
    alpha = vis
    return th.cat([rgb, alpha], dim=1)


def _compute_reconstruction_delta(
    recon: Any,
    baseline: Any,
    *,
    visible_only: bool = True,
) -> dict[str, float]:
    """Compute RGB MAE, alpha disagreement, and changed-pixel rate vs baseline."""
    _require_torch()
    if visible_only:
        mask = (baseline[:, 3:4] > 0.5).float()
    else:
        mask = None

    rgb_diff = (recon[:, :3] - baseline[:, :3]).abs()
    if mask is not None:
        rgb_mae = float((rgb_diff * mask).sum().item() / max(mask.sum().item(), 1))
    else:
        rgb_mae = float(rgb_diff.mean().item())

    alpha_recon = (recon[:, 3:4] > 0.5).float()
    alpha_base = (baseline[:, 3:4] > 0.5).float()
    alpha_disagree = float((alpha_recon != alpha_base).float().mean().item())

    pixel_changed = (recon[:, :3] - baseline[:, :3]).abs().max(dim=1, keepdim=True).values > 0.02
    if mask is not None:
        changed_rate = float((pixel_changed.float() * mask).sum().item() / max(mask.sum().item(), 1))
    else:
        changed_rate = float(pixel_changed.float().mean().item())

    return {"rgb_mae": rgb_mae, "alpha_disagreement": alpha_disagree, "changed_pixel_rate": changed_rate}


def _decode_index_stats(index_logits: Any, alpha_mask: Any) -> dict[str, Any]:
    """Per-batch index statistics."""
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
    entropy = -(probs * log_probs).sum(dim=1).mean(dim=(1, 2))
    entropy_mean = float(entropy.mean().item())

    cont_vis_pct = float(alpha_mask.float().mean().item())

    return {
        "used_index_count_mean": used_mean,
        "slot0_pixel_share": slot0_share,
        "index_entropy_mean": entropy_mean,
        "continuous_alpha_visible_pct": cont_vis_pct,
    }


def _validate_variant_images(vdir: Path, sample_count: int, *, min_visible_pct: float = 0.95) -> list[str]:
    """Validate generated PNGs before review. Returns list of error messages."""
    errors: list[str] = []
    manifest_path = vdir / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        return [f"Missing manifest: {manifest_path}"]
    try:
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return [f"Failed to read manifest: {exc}"]

    transparent_count = 0
    full_count = 0
    for record in records:
        opaque = int(record.get("alpha_opaque_count", 0))
        if opaque == 0:
            transparent_count += 1
        elif opaque >= SPRITE_SIZE * SPRITE_SIZE:
            full_count += 1

    if len(records) != sample_count:
        errors.append(f"Manifest has {len(records)} records, expected {sample_count}")

    if transparent_count / max(len(records), 1) > (1.0 - min_visible_pct):
        errors.append(
            f"{transparent_count}/{len(records)} samples are fully transparent — baseline sampling likely broken"
        )
    if transparent_count > 0:
        print(f"    {transparent_count}/{len(records)} fully transparent")

    return errors


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

    variant_records: dict[str, list[dict[str, Any]]] = {key: [] for key in variant_dirs}
    variant_deltas: dict[str, list[dict[str, float]]] = {
        key: [] for key in variant_dirs if key != "continuous_v1_projected"
    }
    variant_aux_stats: dict[str, list[dict[str, Any]]] = {
        key: [] for key in variant_dirs if key != "continuous_v1_projected"
    }

    base_noise_seed = int(config.seed) * 100000
    th.manual_seed(config.seed)

    shared_meta = {
        "checkpoint": str(config.checkpoint),
        "conditioning_mode": conditioning_mode,
        "model_type": "generator_challenger",
        "architecture": "rectified_flow",
        "cfg_scale": float(config.cfg_scale),
        "steps": int(config.sample_steps),
        "alpha_threshold": float(config.alpha_threshold),
        "max_colors": int(config.max_colors),
        "seed": int(config.seed),
        "export_preset": "",
    }

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
            # Use the NORMAL integrate_rectified_flow — guaranteed identical to the v1 sampler.
            rgba_batch = integrate_rectified_flow(
                model,
                initial,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                steps=config.sample_steps,
                cfg_scale=config.cfg_scale,
                pad_token_id=tokenizer.pad_id,
            )

            # Capture auxiliary head outputs on a SEPARATE model call at t=0
            # (clean generated image). This avoids any modification to the
            # normal sampling path.
            t_zero = th.zeros((rgba_batch.shape[0],), device=device, dtype=rgba_batch.dtype)
            aux_outputs = model(
                rgba_batch,
                t_zero,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                return_aux=True,
            )
            aux = {
                "palette_rgb": aux_outputs["palette_rgb"],
                "palette_presence_logits": aux_outputs["palette_presence_logits"],
                "index_logits": aux_outputs["index_logits"],
            }

        rgba_np = np.moveaxis(rgba_batch.detach().cpu().numpy().astype(np.float32), 1, -1)
        alpha_mask = rgba_batch[:, 3]  # (B, 32, 32) — continuous alpha, used for head decode variants

        # Build head decode variants using continuous alpha mask
        pred_rgba = _reconstruct_from_index_and_palette(
            aux["index_logits"],
            aux["palette_rgb"],
            alpha_mask=alpha_mask,
        )
        pred_np = np.moveaxis(pred_rgba.detach().cpu().numpy().astype(np.float32), 1, -1)

        proj_rgba = _reconstruct_index_with_projected_palette(
            aux["index_logits"],
            rgba_batch,
            alpha_mask=alpha_mask,
        )
        proj_np = np.moveaxis(proj_rgba.detach().cpu().numpy().astype(np.float32), 1, -1)

        for item_index, prompt_record in enumerate(batch_records):
            sample_idx = batch_start + item_index
            sample_id = f"sample_{sample_idx:06d}"
            per_sample_meta = {
                **shared_meta,
                **prompt_record,
                "sample_id": sample_id,
                "noise_seed": int(noise_seeds[item_index]),
            }

            # ── Variant 1: continuous v1 projected (baseline) ──
            sprite_v1 = canonicalize_generated_rgba(
                rgba_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_v1 = write_generated_sprite_artifacts(
                sprite_v1,
                variant_dirs["continuous_v1_projected"],
                sample_id,
                {**per_sample_meta, "decode_variant": "continuous_v1_projected"},
                write_raw_rgba=True,
                write_hard_rgba=True,
            )
            from spritelab.training.palette_projection import project_generated_sprite_record

            rec_v1_projected = project_generated_sprite_record(
                sprite_v1,
                variant_dirs["continuous_v1_projected"],
                rec_v1,
                target_colors=config.project_palette_v1_target_colors,
                min_pixel_share=0.01,
                alpha_threshold=config.alpha_threshold,
                method="deterministic_kmeans",
            )
            variant_records["continuous_v1_projected"].append(rec_v1_projected)

            # ── Variant 2: head index + predicted palette (alpha from continuous) ──
            sprite_pred = canonicalize_generated_rgba(
                pred_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_pred = write_generated_sprite_artifacts(
                sprite_pred,
                variant_dirs["head_index_pred_palette"],
                sample_id,
                {**per_sample_meta, "decode_variant": "head_index_pred_palette"},
                write_raw_rgba=True,
                write_hard_rgba=True,
            )
            variant_records["head_index_pred_palette"].append(rec_pred)

            delta = _compute_reconstruction_delta(
                pred_rgba[item_index : item_index + 1],
                rgba_batch[item_index : item_index + 1],
            )
            variant_deltas.setdefault("head_index_pred_palette", []).append(delta)

            # ── Variant 3: head index + projected palette (alpha from continuous) ──
            sprite_proj = canonicalize_generated_rgba(
                proj_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
            )
            rec_proj = write_generated_sprite_artifacts(
                sprite_proj,
                variant_dirs["head_index_projected_palette"],
                sample_id,
                {**per_sample_meta, "decode_variant": "head_index_projected_palette"},
                write_raw_rgba=True,
                write_hard_rgba=True,
            )
            variant_records["head_index_projected_palette"].append(rec_proj)

            delta2 = _compute_reconstruction_delta(
                proj_rgba[item_index : item_index + 1],
                rgba_batch[item_index : item_index + 1],
            )
            variant_deltas.setdefault("head_index_projected_palette", []).append(delta2)

            # Per-sample aux stats
            aux_stats = _decode_index_stats(
                aux["index_logits"][item_index : item_index + 1],
                alpha_mask[item_index : item_index + 1],
            )
            variant_aux_stats.setdefault("head_index_pred_palette", []).append(aux_stats)
            variant_aux_stats.setdefault("head_index_projected_palette", []).append(aux_stats)

        print(
            f"  batch {batch_start // config.batch_size + 1}/"
            f"{(len(prompts) + config.batch_size - 1) // config.batch_size} done"
        )

    # Write manifests and generation reports
    for key, records in variant_records.items():
        write_generation_reports(
            out_dir=variant_dirs[key],
            records=records,
            config={"archive": "challenger", "architecture": "rectified_flow", "conditioning_mode": conditioning_mode},
            contact_sheet=None,
        )

    # Validate images before running reviews
    for key in variant_dirs:
        errors = _validate_variant_images(variant_dirs[key], len(prompts))
        if errors:
            print(f"  FAIL: {key}:")
            for e in errors:
                print(f"    {e}")
    # Abort if baseline is broken
    baseline_errors = _validate_variant_images(variant_dirs["continuous_v1_projected"], len(prompts))
    if any("fully transparent" in e for e in baseline_errors):
        print("\nERROR: continuous_v1_projected baseline is fully transparent. Aborting to avoid misleading reports.")
        raise SystemExit("Baseline sampling failed — check model/checkpoint/config.")

    # Run review / QA / faithfulness on each variant
    variant_metrics: dict[str, dict[str, Any]] = {}
    for key, vdir in variant_dirs.items():
        print(f"  running review on {key}...")
        metrics = _run_variant_metrics(vdir, config.dataset, config.prompts)
        variant_metrics[key] = metrics

    # Aggregate decode-specific stats from aux
    _collect_decode_stats(variant_metrics, variant_aux_stats)

    # Build contact sheets
    contact_sheet_dir = out / "contact_sheets"
    contact_sheet_dir.mkdir(parents=True, exist_ok=True)
    _build_decode_contact_sheets(contact_sheet_dir, variant_records, variant_dirs, prompts)

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
        "max_colors": config.max_colors,
        "alpha_threshold": config.alpha_threshold,
        "decode_variants": list(variant_dirs.keys()),
        "head_decode_policy": {
            "alpha_source": "continuous_image",
            "transparent_slot": 0,
            "description": (
                "Alpha mask from continuous generated image (alpha > 0.5). "
                "Predicted index argmax used for RGB lookup. "
                "Slot 0 (predicted) mapped to transparent; RGB zeroed for invisible pixels."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": variant_metrics,
    }
    (reports_dir / "decode_probe_summary.json").write_text(
        json.dumps(report, indent=2, default=jsonable),
        encoding="utf-8",
    )
    (reports_dir / "decode_probe_report.md").write_text(
        _build_decode_report_md(report, variant_metrics, variant_deltas),
        encoding="utf-8",
    )

    print(f"\nReport: {reports_dir / 'decode_probe_report.md'}")
    return reports_dir / "decode_probe_summary.json"


def _run_variant_metrics(vdir: Path, dataset: Path, prompts_path: Path) -> dict[str, Any]:
    qa_result = qa_generated_sprites(vdir, error_on_fully_transparent=False)
    qa_errors = len(qa_result.errors) if hasattr(qa_result, "errors") else 0

    review_result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=vdir,
            out_dir=vdir / "review",
            group_by="none",
            max_samples_per_sheet=64,
        )
    )
    review = review_result.report if hasattr(review_result, "report") else {}

    faith = {}
    try:
        faith_dir = vdir / "prompt_faithfulness"
        faith_dir.mkdir(parents=True, exist_ok=True)
        faith_result = run_prompt_faithfulness(
            PromptFaithfulnessConfig(
                generated=vdir,
                prompts=prompts_path,
                dataset=dataset,
                out=faith_dir,
                out_json=faith_dir / "prompt_faithfulness_report.json",
            )
        )
        if isinstance(faith_result, dict):
            faith = faith_result
    except Exception:
        pass

    overall = review.get("overall", {}) if isinstance(review, dict) else {}
    warning_counts = overall.get("warning_counts", {}) if isinstance(overall, dict) else {}
    sample_count = int(overall.get("sample_count", review.get("sample_count", faith.get("sample_count", 1))))

    def _rate(count_key: str) -> float | None:
        count = warning_counts.get(count_key)
        if count is None:
            return None
        return float(count) / max(sample_count, 1)

    def _faith_rate(key: str) -> float | None:
        val = faith.get(key)
        return float(val) if val is not None else None

    def _faith_or_overall(faith_key: str, overall_key: str) -> float | None:
        val = faith.get(faith_key)
        if val is not None:
            return float(val)
        val = overall.get(overall_key)
        return float(val) if val is not None else None

    return {
        "qa_errors": qa_errors,
        "sample_count": sample_count,
        "median_visible_colors": float(overall.get("median_visible_color_count"))
        if overall.get("median_visible_color_count") is not None
        else None,
        "rare_color_rate": _rate("too_many_rare_colors"),
        "category_consistency": _faith_or_overall("category_consistency_rate", "category_consistency_mean"),
        "color_consistency": _faith_or_overall("color_consistency_rate", "color_consistency_mean"),
        "repeated_silhouette_rate": _faith_rate("repeated_silhouette_rate"),
        "blob_collapse_rate": _faith_rate("generic_blob_collapse_rate"),
        "potion_collapse_rate": _faith_rate("generic_potion_collapse_rate"),
        "near_copy_rate": _faith_rate("near_copy_rate"),
        "border_touch_rate": _rate("touches_border"),
        "nearest_source_distance_p10": float(faith.get("p10_nearest_source_distance"))
        if faith.get("p10_nearest_source_distance") is not None
        else None,
    }


def _collect_decode_stats(
    variant_metrics: dict[str, dict[str, Any]],
    variant_aux_stats: dict[str, list[dict[str, Any]]],
) -> None:
    for key, stats_list in variant_aux_stats.items():
        if key not in variant_metrics:
            continue
        if not stats_list:
            continue
        keys = [k for k in stats_list[0] if k not in ("continuous_alpha_visible_pct",)]
        for k in keys:
            variant_metrics[key][k] = float(np.mean([s[k] for s in stats_list]))
        # continuous alpha visible pct is the same for both head variants (derived from same rgba_batch)
        variant_metrics[key]["continuous_alpha_visible_pct"] = float(
            np.mean([s["continuous_alpha_visible_pct"] for s in stats_list])
        )


def _build_decode_contact_sheets(
    contact_sheet_dir: Path,
    variant_records: dict[str, list[dict[str, Any]]],
    variant_dirs: dict[str, Path],
    prompts: list[dict[str, Any]],
) -> None:
    n = min(32, len(prompts))
    for key, records in variant_records.items():
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


def _build_decode_report_md(
    report: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    deltas: dict[str, list[dict[str, float]]] | None = None,
) -> str:
    def _fmt_md(val: Any, fmt: str) -> str:
        if val is None:
            return "NA"
        return f"{val:{fmt}}"

    policy = report.get("head_decode_policy", {})
    lines = [
        "# Palette / Index Decode Probe Report",
        "",
        f"- **Checkpoint**: `{report['checkpoint']}`",
        f"- **Prompts**: `{report['prompts']}`",
        f"- **Samples**: {report['sample_count']}",
        f"- **Seed**: {report['seed']}",
        f"- **Steps**: {report['sample_steps']}",
        f"- **CFG scale**: {report['cfg_scale']}",
        f"- **Max colors**: {report.get('max_colors', 16)}",
        f"- **Alpha threshold**: {report.get('alpha_threshold', 0.5)}",
        f"- **Elapsed**: {report['elapsed_seconds']:.1f}s",
        "",
        "## Head Decode Policy",
        "",
        f"- Alpha source: **{policy.get('alpha_source', 'unknown')}**",
        f"- Transparent slot: **{policy.get('transparent_slot', '?')}**",
        f"- {policy.get('description', '')}",
        "",
        "## Decode Variant Metrics",
        "",
        "| Variant | QA errs | Med colors | Rare color % | Cat cons | Color cons | Silhouette % | Blob % | Potion % | Near-copy % | Touch % |",
        "|---------|---------|------------|-------------|----------|------------|-------------|--------|----------|-------------|---------|",
    ]
    for key, m in metrics.items():
        lines.append(
            f"| `{key}` | {m.get('qa_errors', 0)} | "
            f"{_fmt_md(m.get('median_visible_colors'), '.1f')} | "
            f"{_fmt_md(m.get('rare_color_rate'), '.3f')} | "
            f"{_fmt_md(m.get('category_consistency'), '.4f')} | "
            f"{_fmt_md(m.get('color_consistency'), '.4f')} | "
            f"{_fmt_md(m.get('repeated_silhouette_rate'), '.4f')} | "
            f"{_fmt_md(m.get('blob_collapse_rate'), '.4f')} | "
            f"{_fmt_md(m.get('potion_collapse_rate'), '.4f')} | "
            f"{_fmt_md(m.get('near_copy_rate'), '.4f')} | "
            f"{_fmt_md(m.get('border_touch_rate'), '.4f')} |"
        )

    # Decode-specific stats
    hd_keys = [
        ("used_index_count_mean", ".1f"),
        ("slot0_pixel_share", ".4f"),
        ("index_entropy_mean", ".4f"),
        ("continuous_alpha_visible_pct", ".4f"),
    ]
    for hk, hfmt in hd_keys:
        vals = {k: m.get(hk) for k, m in metrics.items() if k != "continuous_v1_projected"}
        vals = {k: v for k, v in vals.items() if v is not None}
        if vals:
            lines += ["", f"## {hk.replace('_', ' ').title()}", ""]
            for key, val in vals.items():
                lines.append(f"- `{key}`: {_fmt_md(val, hfmt)}")

    if deltas:
        lines += [
            "",
            "## Reconstruction Deltas vs Continuous v1 Projected (visible pixels only)",
            "",
            "| Variant | RGB MAE | Alpha disagreement | Changed pixel rate |",
            "|---------|---------|-------------------|-------------------|",
        ]
        for key, dlist in deltas.items():
            if dlist:
                avg = {k: float(np.mean([d[k] for d in dlist])) for k in dlist[0]}
                lines.append(
                    f"| `{key}` | {avg['rgb_mae']:.6f} | {avg['alpha_disagreement']:.6f} | {avg['changed_pixel_rate']:.4f} |"
                )

    lines += [
        "",
        "## Interpretation",
        "",
        "Head decode captures aux outputs on a separate post-sampling model call",
        "at t=0 (clean generated image). The continuous image's alpha mask (threshold 0.5)",
        "is used for head decode variants. Slot 0 is reserved for transparent pixels.",
        "Metrics use the same keys as v2 Phase 0 eval (review.warning_counts + faithfulness rates).",
        "",
    ]
    return "\n".join(lines) + "\n"


def _sample_initial_noise(batch_size: int, *, device: Any, seed: int) -> Any:
    th = _require_torch()
    generator = th.Generator(device=device)
    generator.manual_seed(int(seed))
    return th.randn(int(batch_size), 4, SPRITE_SIZE, SPRITE_SIZE, device=device, generator=generator)

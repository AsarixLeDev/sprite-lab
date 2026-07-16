"""No-training diagnostic command for v2 Phase 2 palette/index auxiliary heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

from spritelab.training.checkpoint_io import load_checkpoint as _load_checkpoint
from spritelab.training.conditioning import apply_conditioning_mode
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch
from spritelab.training.device import move_batch_to_device, resolve_device
from spritelab.training.generator_challenger import (
    RectifiedFlowUNet,
    _structured_conditioning_from_batch,
    load_challenger_from_checkpoint,
)
from spritelab.training.optim_utils import (
    apply_backend_speed_flags,
    dataloader_perf_kwargs,
)
from spritelab.training.structured_conditioning import structured_vocab_from_checkpoint

ROLE_LABELS: dict[int, str] = {
    0: "background",
    1: "outline",
    2: "shadow",
    3: "fill",
    4: "highlight",
}

VISIBLE_ALPHA_THRESHOLD = 0.05
PALETTE_PRESENCE_THRESHOLD = 0.5
K = 16


@dataclass(frozen=True)
class PaletteIndexHeadInspectConfig:
    checkpoint: Path
    dataset: Path
    training_manifest: Path
    out: Path
    device: str = "cpu"
    batch_size: int = 32
    max_batches: int = 32
    split: str = "train"
    cudnn_benchmark: bool = False
    tf32: bool = False


@dataclass
class _IndexMetrics:
    visible_pixel_accuracy: float = 0.0
    top2_accuracy: float = 0.0
    cross_entropy: float = 0.0
    per_role_accuracy: dict[str, float] = field(default_factory=dict)
    invalid_target_count: int = 0
    ignored_pixel_count: int = 0
    visible_pixel_count: int = 0
    index_entropy_mean: float = 0.0


@dataclass
class _PaletteRGBMetrics:
    mse: float = 0.0
    mae: float = 0.0
    per_slot_mae: list[float] = field(default_factory=lambda: [0.0] * K)
    active_slot_count_mean: float = 0.0
    slot0_is_transparent: bool = False


@dataclass
class _PalettePresenceMetrics:
    bce: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    predicted_active_mean: float = 0.0
    target_active_mean: float = 0.0
    false_positives_rate: float = 0.0
    false_negatives_rate: float = 0.0


@dataclass
class _BatchMetrics:
    index: _IndexMetrics = field(default_factory=_IndexMetrics)
    palette_rgb: _PaletteRGBMetrics = field(default_factory=_PaletteRGBMetrics)
    palette_presence: _PalettePresenceMetrics = field(default_factory=_PalettePresenceMetrics)


def _has_palette_index_heads(model: RectifiedFlowUNet) -> bool:
    return (
        hasattr(model, "palette_head_rgb") and hasattr(model, "palette_head_presence") and hasattr(model, "index_head")
    )


def compute_index_head_metrics(
    predicted_logits: Any,
    target_index_map: Any,
    alpha: Any,
    role_map: Any | None = None,
) -> _IndexMetrics:
    th = _require_torch()
    _B, C, _H, _W = predicted_logits.shape

    visible_mask = alpha.squeeze(1) > VISIBLE_ALPHA_THRESHOLD
    target = target_index_map.long()
    pred = predicted_logits.argmax(dim=1)

    invalid_mask = target >= C
    invalid_count = int(invalid_mask.sum().item())
    target = target.clamp(0, C - 1)

    ignored_mask = ~visible_mask
    ignored_count = int(ignored_mask.sum().item())

    eval_mask = visible_mask & ~invalid_mask
    visible_count = int(eval_mask.sum().item())

    if visible_count == 0:
        return _IndexMetrics(
            visible_pixel_count=0,
            invalid_target_count=invalid_count,
            ignored_pixel_count=ignored_count,
        )

    correct = (pred == target) & eval_mask
    accuracy = float(correct.sum().item()) / visible_count

    top2_correct = _compute_topk_correct(predicted_logits, target, eval_mask, k=2)
    top2_accuracy = float(top2_correct.sum().item()) / visible_count

    ce = th.nn.functional.cross_entropy(
        predicted_logits.permute(0, 2, 3, 1).reshape(-1, C)[eval_mask.reshape(-1)],
        target[eval_mask],
        reduction="mean",
    )
    cross_entropy = float(ce.item())

    entropy = _compute_entropy(predicted_logits, eval_mask)
    entropy_mean = float(entropy.item())

    per_role: dict[str, float] = {}
    if role_map is not None:
        for role_id, role_name in sorted(ROLE_LABELS.items()):
            role_mask = (role_map == role_id) & eval_mask
            role_count = int(role_mask.sum().item())
            if role_count > 0:
                role_acc = float((correct & role_mask).sum().item()) / role_count
                per_role[role_name] = role_acc
        other_mask = eval_mask.clone()
        for role_id in ROLE_LABELS:
            other_mask &= role_map != role_id
        other_count = int(other_mask.sum().item())
        if other_count > 0:
            per_role["other"] = float((correct & other_mask).sum().item()) / other_count

    return _IndexMetrics(
        visible_pixel_accuracy=accuracy,
        top2_accuracy=top2_accuracy,
        cross_entropy=cross_entropy,
        per_role_accuracy=per_role,
        invalid_target_count=invalid_count,
        ignored_pixel_count=ignored_count,
        visible_pixel_count=visible_count,
        index_entropy_mean=entropy_mean,
    )


def _compute_topk_correct(logits: Any, target: Any, mask: Any, k: int) -> Any:
    _require_torch()
    topk = logits.topk(k, dim=1).indices
    correct = (topk == target.unsqueeze(1)).any(dim=1) & mask
    return correct


def _compute_entropy(logits: Any, mask: Any) -> Any:
    th = _require_torch()
    probs = th.nn.functional.softmax(logits, dim=1)
    log_probs = th.nn.functional.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy[mask].mean()


def compute_palette_rgb_metrics(
    predicted_palette_rgb: Any,
    target_palette: Any,
    palette_mask: Any,
) -> _PaletteRGBMetrics:
    th = _require_torch()
    B = predicted_palette_rgb.shape[0]

    K_gt = target_palette.shape[1]
    K_pred = predicted_palette_rgb.shape[1]
    if K_gt > K_pred:
        target_palette = target_palette[:, :K_pred].contiguous()
        palette_mask = palette_mask[:, :K_pred].contiguous()
    elif K_gt < K_pred:
        pad_k = K_pred - K_gt
        target_palette = th.cat(
            [
                target_palette,
                th.zeros(B, pad_k, 3, device=target_palette.device, dtype=target_palette.dtype),
            ],
            dim=1,
        )
        palette_mask = th.cat(
            [
                palette_mask,
                th.zeros(B, pad_k, device=palette_mask.device, dtype=palette_mask.dtype),
            ],
            dim=1,
        )

    mask = palette_mask.bool()
    active_count = mask.sum(dim=1).float().mean()
    active_count_mean = float(active_count.item())

    mask_3d = mask.unsqueeze(-1).float()

    diff = predicted_palette_rgb - target_palette.to(dtype=predicted_palette_rgb.dtype)
    squared = (diff * diff) * mask_3d
    absolute = diff.abs() * mask_3d

    total_active = mask.sum()
    if total_active == 0:
        return _PaletteRGBMetrics(active_slot_count_mean=active_count_mean)

    mse = float(squared.sum().item() / (total_active.item() * 3))
    mae = float(absolute.sum().item() / (total_active.item() * 3))

    per_slot = []
    for slot in range(K):
        slot_mask = mask[:, slot]
        slot_count = slot_mask.sum()
        if slot_count > 0:
            slot_abs = (
                predicted_palette_rgb[:, slot] - target_palette[:, slot].to(dtype=predicted_palette_rgb.dtype)
            ).abs()
            slot_mae = float((slot_abs * slot_mask.unsqueeze(-1).float()).sum().item() / (slot_count.item() * 3))
            per_slot.append(slot_mae)
        else:
            per_slot.append(0.0)

    slot0_transparent_likely = _check_slot0_transparent(target_palette, palette_mask)

    return _PaletteRGBMetrics(
        mse=mse,
        mae=mae,
        per_slot_mae=per_slot,
        active_slot_count_mean=active_count_mean,
        slot0_is_transparent=slot0_transparent_likely,
    )


def _check_slot0_transparent(target_palette: Any, palette_mask: Any) -> bool:
    slot0_visible = palette_mask[:, 0].any().item()
    if not slot0_visible:
        return False
    slot0_avg_rgb = target_palette[:, 0].mean(dim=0)
    (slot0_avg_rgb > 0.95).all().item()
    slot0_dark = (slot0_avg_rgb < 0.05).all().item()
    return slot0_dark


def compute_palette_presence_metrics(
    predicted_logits: Any,
    target_mask: Any,
) -> _PalettePresenceMetrics:
    th = _require_torch()

    K_gt = target_mask.shape[1]
    K_pred = predicted_logits.shape[1]
    if K_gt > K_pred:
        predicted_logits = th.cat(
            [
                predicted_logits,
                th.full(
                    (predicted_logits.shape[0], K_gt - K_pred),
                    -10.0,
                    device=predicted_logits.device,
                    dtype=predicted_logits.dtype,
                ),
            ],
            dim=1,
        )
    elif K_gt < K_pred:
        target_mask = th.cat(
            [
                target_mask,
                th.zeros(target_mask.shape[0], K_pred - K_gt, device=target_mask.device, dtype=target_mask.dtype),
            ],
            dim=1,
        )

    target = target_mask.float()

    bce = float(
        th.nn.functional.binary_cross_entropy_with_logits(
            predicted_logits,
            target,
            reduction="mean",
        ).item()
    )

    pred_prob = th.sigmoid(predicted_logits)
    pred_binary = (pred_prob > PALETTE_PRESENCE_THRESHOLD).float()

    accuracy = float((pred_binary == target).float().mean().item())

    true_pos = (pred_binary * target).sum().float()
    pred_pos = pred_binary.sum().float()
    target_pos = target.sum().float()

    precision = float((true_pos / pred_pos.clamp(min=1)).item())
    recall = float((true_pos / target_pos.clamp(min=1)).item())
    f1 = float((2 * true_pos / (pred_pos + target_pos).clamp(min=1)).item())

    pred_active_mean = float(pred_pos.item() / target.shape[0])
    target_active_mean = float(target_pos.item() / target.shape[0])

    false_pos = (pred_binary * (1 - target)).sum().float()
    ((1 - pred_binary) * (1 - target)).sum().float()
    total_neg = (1 - target).sum().float()
    false_neg = ((1 - pred_binary) * target).sum().float()
    total_pos = target.sum().float()

    fp_rate = float((false_pos / total_neg.clamp(min=1)).item())
    fn_rate = float((false_neg / total_pos.clamp(min=1)).item())

    return _PalettePresenceMetrics(
        bce=bce,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_active_mean=pred_active_mean,
        target_active_mean=target_active_mean,
        false_positives_rate=fp_rate,
        false_negatives_rate=fn_rate,
    )


def write_inspect_report(
    batches: list[_BatchMetrics],
    config: PaletteIndexHeadInspectConfig,
    out: Path,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)

    aggregate = _aggregate_metrics(batches)

    report = {
        "checkpoint": str(config.checkpoint),
        "dataset": str(config.dataset),
        "manifest": str(config.training_manifest),
        "max_batches": config.max_batches,
        "batches_evaluated": len(batches),
        "aggregate": _metrics_to_dict(aggregate),
    }

    json_path = out / "palette_index_head_inspect.json"
    json_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    md_lines = _build_markdown(report)
    md_path = out / "palette_index_head_inspect.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return json_path


def _aggregate_metrics(batches: list[_BatchMetrics]) -> _BatchMetrics:
    if not batches:
        return _BatchMetrics()

    idx_batches = [b for b in batches if b.index.visible_pixel_count > 0]
    if not idx_batches:
        return _BatchMetrics()

    # Index metrics
    total_visible = sum(b.index.visible_pixel_count for b in idx_batches)
    idx_acc = sum(b.index.visible_pixel_accuracy * b.index.visible_pixel_count for b in idx_batches) / total_visible
    top2_acc = sum(b.index.top2_accuracy * b.index.visible_pixel_count for b in idx_batches) / total_visible
    idx_ce = sum(b.index.cross_entropy * b.index.visible_pixel_count for b in idx_batches) / total_visible
    total_invalid = sum(b.index.invalid_target_count for b in batches)
    total_ignored = sum(b.index.ignored_pixel_count for b in batches)
    idx_entropy = sum(b.index.index_entropy_mean * b.index.visible_pixel_count for b in idx_batches) / total_visible

    per_role: dict[str, float] = {}
    for role_name in sorted(ROLE_LABELS.values()):
        role_values = [
            (b.index.per_role_accuracy.get(role_name, 0.0), b.index.visible_pixel_count)
            for b in idx_batches
            if role_name in b.index.per_role_accuracy
        ]
        if role_values:
            weighted = sum(acc * cnt for acc, cnt in role_values) / sum(cnt for _, cnt in role_values)
            per_role[role_name] = weighted
    other_values = [
        (b.index.per_role_accuracy.get("other", 0.0), b.index.visible_pixel_count)
        for b in idx_batches
        if "other" in b.index.per_role_accuracy
    ]
    if other_values:
        weighted = sum(acc * cnt for acc, cnt in other_values) / sum(cnt for _, cnt in other_values)
        per_role["other"] = weighted

    index = _IndexMetrics(
        visible_pixel_accuracy=idx_acc,
        top2_accuracy=top2_acc,
        cross_entropy=idx_ce,
        per_role_accuracy=per_role,
        invalid_target_count=total_invalid,
        ignored_pixel_count=total_ignored,
        visible_pixel_count=total_visible,
        index_entropy_mean=idx_entropy,
    )

    # Palette RGB
    rgb_batches = list(batches)
    active_total = sum(b.palette_rgb.active_slot_count_mean for b in rgb_batches)
    rgb_mae = sum(b.palette_rgb.mae * b.palette_rgb.active_slot_count_mean for b in rgb_batches)
    rgb_mse = sum(b.palette_rgb.mse * b.palette_rgb.active_slot_count_mean for b in rgb_batches)
    if active_total > 0:
        rgb_mae /= active_total
        rgb_mse /= active_total
    else:
        rgb_mae = 0.0
        rgb_mse = 0.0
    per_slot_mae = [0.0] * K
    if active_total > 0:
        for slot in range(K):
            slot_sum = sum(b.palette_rgb.per_slot_mae[slot] * b.palette_rgb.active_slot_count_mean for b in rgb_batches)
            per_slot_mae[slot] = slot_sum / active_total
    active_count_mean = active_total / max(len(rgb_batches), 1)
    slot0_transparent = any(b.palette_rgb.slot0_is_transparent for b in rgb_batches)

    palette_rgb = _PaletteRGBMetrics(
        mse=rgb_mse,
        mae=rgb_mae,
        per_slot_mae=per_slot_mae,
        active_slot_count_mean=active_count_mean,
        slot0_is_transparent=slot0_transparent,
    )

    # Palette presence
    n = len(list(batches))
    pres_bce = sum(b.palette_presence.bce for b in batches) / max(n, 1)
    pres_acc = sum(b.palette_presence.accuracy for b in batches) / max(n, 1)
    pres_prec = sum(b.palette_presence.precision for b in batches) / max(n, 1)
    pres_rec = sum(b.palette_presence.recall for b in batches) / max(n, 1)
    pres_f1 = sum(b.palette_presence.f1 for b in batches) / max(n, 1)
    pred_act = sum(b.palette_presence.predicted_active_mean for b in batches) / max(n, 1)
    targ_act = sum(b.palette_presence.target_active_mean for b in batches) / max(n, 1)
    fp_rate = sum(b.palette_presence.false_positives_rate for b in batches) / max(n, 1)
    fn_rate = sum(b.palette_presence.false_negatives_rate for b in batches) / max(n, 1)

    palette_presence = _PalettePresenceMetrics(
        bce=pres_bce,
        accuracy=pres_acc,
        precision=pres_prec,
        recall=pres_rec,
        f1=pres_f1,
        predicted_active_mean=pred_act,
        target_active_mean=targ_act,
        false_positives_rate=fp_rate,
        false_negatives_rate=fn_rate,
    )

    return _BatchMetrics(index=index, palette_rgb=palette_rgb, palette_presence=palette_presence)


def _metrics_to_dict(metrics: _BatchMetrics) -> dict[str, Any]:
    return {
        "index_head": asdict(metrics.index),
        "palette_rgb_head": asdict(metrics.palette_rgb),
        "palette_presence_head": asdict(metrics.palette_presence),
    }


def _build_markdown(report: dict[str, Any]) -> list[str]:
    agg = report["aggregate"]
    idx = agg["index_head"]
    prgb = agg["palette_rgb_head"]
    ppr = agg["palette_presence_head"]

    lines = [
        "# Palette / Index Head Inspection Report",
        "",
        "## Configuration",
        "",
        f"- **Checkpoint**: `{report['checkpoint']}`",
        f"- **Dataset**: `{report['dataset']}`",
        f"- **Manifest**: `{report['manifest']}`",
        f"- **Max batches**: {report['max_batches']}",
        f"- **Batches evaluated**: {report['batches_evaluated']}",
        "",
        "## Aggregate Metrics",
        "",
        "### Index Head",
        "",
        f"- **Visible pixel accuracy**: {idx['visible_pixel_accuracy']:.4f}",
        f"- **Top-2 accuracy**: {idx['top2_accuracy']:.4f}",
        f"- **Cross-entropy**: {idx['cross_entropy']:.6f}",
        f"- **Index entropy mean**: {idx['index_entropy_mean']:.4f}",
        f"- **Visible pixels**: {idx['visible_pixel_count']}",
        f"- **Invalid targets**: {idx['invalid_target_count']}",
        f"- **Ignored pixels**: {idx['ignored_pixel_count']}",
        "",
    ]

    per_role = idx.get("per_role_accuracy", {})
    if per_role:
        lines.append("#### Per-Role Accuracy")
        lines.append("")
        for role, acc in per_role.items():
            lines.append(f"- **{role}**: {acc:.4f}")
        lines.append("")

    lines += [
        "### Palette RGB Head",
        "",
        f"- **MSE (active slots)**: {prgb['mse']:.6f}",
        f"- **MAE (active slots)**: {prgb['mae']:.6f}",
        f"- **Active slot count mean**: {prgb['active_slot_count_mean']:.2f}",
        f"- **Slot 0 is transparent/reserved**: {prgb['slot0_is_transparent']}",
        "",
    ]

    per_slot = prgb.get("per_slot_mae", [])
    if per_slot:
        lines.append("#### Per-Slot MAE")
        lines.append("")
        for slot, mae_s in enumerate(per_slot[:8]):
            lines.append(f"- Slot {slot}: {mae_s:.6f}")
        if len(per_slot) > 8:
            lines.append(f"- ... ({len(per_slot) - 8} more)")
        lines.append("")

    lines += [
        "### Palette Presence Head",
        "",
        f"- **BCE**: {ppr['bce']:.6f}",
        f"- **Accuracy (thr=0.5)**: {ppr['accuracy']:.4f}",
        f"- **Precision**: {ppr['precision']:.4f}",
        f"- **Recall**: {ppr['recall']:.4f}",
        f"- **F1**: {ppr['f1']:.4f}",
        f"- **Predicted active mean**: {ppr['predicted_active_mean']:.2f}",
        f"- **Target active mean**: {ppr['target_active_mean']:.2f}",
        f"- **False positive rate**: {ppr['false_positives_rate']:.4f}",
        f"- **False negative rate**: {ppr['false_negatives_rate']:.4f}",
        "",
    ]

    # Interpretation
    lines += [
        "## Interpretation",
        "",
    ]

    if idx["visible_pixel_accuracy"] > 0.90:
        lines.append("- Index head accuracy is high - the model reliably assigns pixels to the correct palette slots.")
    elif idx["visible_pixel_accuracy"] > 0.70:
        lines.append("- Index head accuracy is moderate - there is room for improvement in palette slot assignment.")
    else:
        lines.append("- Index head accuracy is low - the model struggles to assign pixels to correct palette slots.")

    if ppr["f1"] > 0.80:
        lines.append("- Palette presence F1 is high - the model reliably predicts which slots are active.")
    elif ppr["f1"] > 0.50:
        lines.append(
            "- Palette presence F1 is moderate - the model partially distinguishes active from inactive slots."
        )
    else:
        lines.append("- Palette presence F1 is low - the model struggles with slot presence prediction.")

    if ppr["predicted_active_mean"] > ppr["target_active_mean"] * 1.2:
        lines.append("- Model tends to over-predict active slots (predicted > target).")
    elif ppr["predicted_active_mean"] < ppr["target_active_mean"] * 0.8:
        lines.append("- Model tends to under-predict active slots (predicted < target).")

    if prgb["mae"] < 0.05:
        lines.append("- Palette RGB MAE is very low - colors are predicted accurately.")
    elif prgb["mae"] < 0.15:
        lines.append("- Palette RGB MAE is moderate - colors are approximated but could be sharper.")
    else:
        lines.append("- Palette RGB MAE is high - color prediction needs improvement.")

    return lines


def run_inspect_palette_index_heads(config: PaletteIndexHeadInspectConfig) -> Path:
    th = _require_torch()

    ckpt = _load_checkpoint(config.checkpoint)
    if ckpt.get("model_type") != "generator_challenger":
        raise SystemExit(
            "Checkpoint is not a generator_challenger checkpoint. "
            f"Found model_type={ckpt.get('model_type')!r}. "
            "This command requires a v2 Phase 2-capable checkpoint."
        )

    device = resolve_device(config.device)
    apply_backend_speed_flags(cudnn_benchmark=config.cudnn_benchmark, tf32=config.tf32)

    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(ckpt, device=device)

    structured_vocab = structured_vocab_from_checkpoint(ckpt)

    if not _has_palette_index_heads(model):
        raise SystemExit(
            "Model does not include palette/index heads. "
            "The checkpoint may be from an older version without v2 Phase 2 auxiliary heads."
        )

    dataset = SpriteTrainingDataset(
        dataset_dir=config.dataset,
        training_manifest=config.training_manifest,
        split=config.split,
        tokenizer=tokenizer,
        caption_max_length=ckpt.get("train_config", {}).get("caption_max_length") or 32,
        semantic_max_length=semantic_max_length,
        structured_vocab=structured_vocab,
    )
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sprite_batch,
        **dataloader_perf_kwargs(device, num_workers=0),
    )

    model.eval()
    all_batches: list[_BatchMetrics] = []

    with th.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            inputs = apply_conditioning_mode(
                caption_tokens=batch.get("caption_tokens"),
                semantic_tokens=batch.get("semantic_tokens"),
                mode=conditioning_mode,
                pad_token_id=tokenizer.pad_id,
                structured_conditioning=_structured_conditioning_from_batch(batch),
            )

            t = th.zeros(batch["rgba"].shape[0], device=device)

            aux = model(
                batch["rgba"],
                t,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                return_aux=True,
            )

            index_metrics = compute_index_head_metrics(
                aux["index_logits"],
                batch["index_map"],
                batch["rgba"][:, 3:4],
                role_map=batch.get("role_map"),
            )
            palette_rgb_metrics = compute_palette_rgb_metrics(
                aux["palette_rgb"],
                batch["palette"],
                batch["palette_mask"],
            )
            presence_metrics = compute_palette_presence_metrics(
                aux["palette_presence_logits"],
                batch["palette_mask"],
            )

            batch_metrics = _BatchMetrics(
                index=index_metrics,
                palette_rgb=palette_rgb_metrics,
                palette_presence=presence_metrics,
            )
            all_batches.append(batch_metrics)

            if config.max_batches > 0 and batch_idx + 1 >= config.max_batches:
                break

    json_path = write_inspect_report(all_batches, config, config.out)

    agg = _aggregate_metrics(all_batches)
    print(f"Index visible-pixel accuracy:  {agg.index.visible_pixel_accuracy:.4f}")
    print(f"Index top-2 accuracy:          {agg.index.top2_accuracy:.4f}")
    print(f"Index cross-entropy:           {agg.index.cross_entropy:.6f}")
    print(f"Palette RGB MAE:               {agg.palette_rgb.mae:.6f}")
    print(f"Palette RGB MSE:               {agg.palette_rgb.mse:.6f}")
    print(
        f"Presence precision/recall/F1:  {agg.palette_presence.precision:.4f} / {agg.palette_presence.recall:.4f} / {agg.palette_presence.f1:.4f}"
    )
    print(f"Predicted active slots mean:   {agg.palette_presence.predicted_active_mean:.2f}")
    print(f"Target active slots mean:      {agg.palette_presence.target_active_mean:.2f}")
    print(f"Batches evaluated:             {len(all_batches)}")
    print(f"Report written to {json_path}")
    return json_path


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for palette/index head inspection.")
    return torch

"""Tiny overfit smoke-test trainer. Not the production model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spritelab.codec.bundle import INDEX_MASK
from spritelab.ml.dataset import SpriteBundleDataset
from spritelab.ml.masking import FixedOpaqueMask
from spritelab.ml.metrics import (
    average_reconstruction_metrics,
    compute_reconstruction_metrics,
    metrics_to_dict,
)
from spritelab.ml.previews import save_prediction_grid


class TinyIndexMapModel(nn.Module):
    """Small convolutional sanity-check model over masked index maps.

    Inputs may contain ``INDEX_MASK``; output logits cover only the real
    index tokens ``0..max_palette_slots`` (``num_tokens`` classes).
    """

    def __init__(
        self,
        num_tokens: int,
        *,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_categories: int = 64,
        category_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        # +1 so the INDEX_MASK token has an embedding row of its own.
        self.token_embedding = nn.Embedding(max(num_tokens, INDEX_MASK + 1), embed_dim)
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        in_channels = embed_dim + 1 + category_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_tokens, kernel_size=1),
        )

    def forward(
        self,
        input_index_map: torch.Tensor,
        alpha: torch.Tensor,
        category_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape [B, num_tokens, 32, 32]."""

        embedded = self.token_embedding(input_index_map.long())
        embedded = embedded.permute(0, 3, 1, 2)
        alpha_channel = alpha.float().unsqueeze(1)
        batch, _, height, width = embedded.shape
        if category_id is None:
            category_id = torch.zeros(batch, dtype=torch.long, device=embedded.device)
        category = self.category_embedding(category_id.long().clamp(min=0))
        category = category[:, :, None, None].expand(-1, -1, height, width)
        features = torch.cat([embedded, alpha_channel, category], dim=1)
        return self.net(features)


def compute_masked_index_loss(
    logits: torch.Tensor,
    target_index_map: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy over masked pixels only; zero (graph-connected) if none."""

    mask = loss_mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    num_tokens = logits.shape[1]
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, num_tokens)
    flat_targets = target_index_map.long().reshape(-1)
    flat_mask = mask.reshape(-1)
    return nn.functional.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


@dataclass(frozen=True)
class OverfitConfig:
    dataset_root: Path
    split: str = "train"
    output_dir: Path = Path("outputs/overfit_smoke")
    max_samples: int = 16
    steps: int = 300
    batch_size: int = 8
    learning_rate: float = 1e-3
    mask_fraction: float = 0.5
    seed: int = 1337
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_overfit_smoke_test(config: OverfitConfig) -> dict[str, Any]:
    """Train the tiny model on a small subset and report before/after metrics."""

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)

    dataset = SpriteBundleDataset(
        config.dataset_root,
        config.split,
        transform=FixedOpaqueMask(mask_fraction=config.mask_fraction, seed=config.seed),
    )
    count = min(len(dataset), config.max_samples)
    if count == 0:
        raise ValueError("dataset split has no samples")
    samples = [dataset[index] for index in range(count)]

    num_tokens = int(samples[0]["palette_mask"].shape[0])
    input_maps = torch.stack([sample["input_index_map"] for sample in samples]).to(device)
    targets = torch.stack([sample["target_index_map"] for sample in samples]).to(device)
    alphas = torch.stack([sample["alpha"] for sample in samples]).to(device)
    loss_masks = torch.stack([sample["loss_mask"] for sample in samples]).to(device)
    categories = torch.stack([sample["category_id"] for sample in samples]).to(device)

    model = TinyIndexMapModel(
        num_tokens,
        num_categories=int(categories.max()) + 2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)

    def evaluate() -> tuple[float, dict[str, Any], list[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            logits = model(input_maps, alphas, categories)
            loss = compute_masked_index_loss(logits, targets, loss_masks)
            predictions = logits.argmax(dim=1)
        per_sample = [
            compute_reconstruction_metrics(
                predictions[index].cpu(),
                samples[index]["target_index_map"],
                samples[index]["alpha"],
                samples[index]["palette_mask"],
                samples[index]["loss_mask"],
            )
            for index in range(count)
        ]
        model.train()
        return (
            float(loss),
            metrics_to_dict(average_reconstruction_metrics(per_sample)),
            [predictions[index].cpu() for index in range(count)],
        )

    initial_loss, initial_metrics, _ = evaluate()

    batch_size = min(config.batch_size, count)
    for _ in range(config.steps):
        indices = torch.randperm(count, generator=generator)[:batch_size]
        logits = model(input_maps[indices], alphas[indices], categories[indices])
        loss = compute_masked_index_loss(logits, targets[indices], loss_masks[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss, final_metrics, predictions = evaluate()

    passed = final_loss < initial_loss and (
        final_metrics["masked_accuracy"] >= initial_metrics["masked_accuracy"]
    )
    result: dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "split": config.split,
        "sample_count": count,
        "steps": config.steps,
        "mask_fraction": config.mask_fraction,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_masked_accuracy": initial_metrics["masked_accuracy"],
        "final_masked_accuracy": final_metrics["masked_accuracy"],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "passed": bool(passed),
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_prediction_grid(samples, predictions, output_dir / "predictions.png")
    return result

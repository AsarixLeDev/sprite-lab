"""Best-effort exploratory previews from scheduled intermediate checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from spritelab.product_core import ProductEvent, ProductStatus

PreviewGenerator = Callable[..., Path]


@dataclass(frozen=True)
class PreviewConfiguration:
    enabled: bool = True
    interval_steps: int = 5_000
    prompts: tuple[str, ...] = ()
    generation_seeds: tuple[int, ...] = (91001,)
    parameters: tuple[tuple[str, object], ...] = ()

    def validate(self, checkpoint_schedule: Sequence[int]) -> None:
        if self.interval_steps <= 0:
            raise ValueError("Preview interval must be positive.")
        if self.enabled and not self.prompts:
            raise ValueError("Enabled previews require a fixed configured prompt set.")
        if self.enabled and not any(step % self.interval_steps == 0 for step in checkpoint_schedule):
            raise ValueError("Preview interval must coincide with at least one configured checkpoint.")


class PreviewScheduler:
    def __init__(self, configuration: PreviewConfiguration, generator: PreviewGenerator) -> None:
        self.configuration = configuration
        self.generator = generator

    def generate(
        self,
        *,
        run_id: str,
        run_root: Path,
        checkpoint: Path,
        checkpoint_step: int,
        training_seed: int,
        checkpoint_schedule: Sequence[int],
    ) -> tuple[ProductEvent, ...]:
        config = self.configuration
        if not config.enabled:
            return ()
        config.validate(checkpoint_schedule)
        if checkpoint_step not in checkpoint_schedule or checkpoint_step % config.interval_steps:
            return ()
        events = []
        parameters = dict(config.parameters)
        for prompt_index, prompt in enumerate(config.prompts):
            for generation_seed in config.generation_seeds:
                output = (
                    run_root
                    / "previews"
                    / f"checkpoint_{checkpoint_step}"
                    / f"seed_{training_seed}"
                    / f"prompt_{prompt_index}_generation_{generation_seed}.png"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    generated = self.generator(
                        checkpoint=checkpoint,
                        prompt=prompt,
                        generation_seed=generation_seed,
                        parameters=parameters,
                        output_path=output,
                    )
                    events.append(
                        ProductEvent(
                            run_id=run_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            feature="training",
                            stage="preview",
                            event_type="exploratory_preview",
                            status=ProductStatus.RUNNING,
                            current=checkpoint_step,
                            message="Exploratory intermediate preview generated.",
                            metrics={
                                "checkpoint": str(checkpoint),
                                "checkpoint_step": checkpoint_step,
                                "training_seed": training_seed,
                                "prompt": prompt,
                                "generation_seed": generation_seed,
                                "parameters": parameters,
                                "output": str(generated),
                                "exploratory": True,
                                "benchmark_evidence": False,
                                "promotion_evidence": False,
                            },
                            artifact_references=(str(generated),),
                        )
                    )
                except Exception as exc:  # best-effort by contract; training must continue
                    events.append(
                        ProductEvent(
                            run_id=run_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            feature="training",
                            stage="preview",
                            event_type="preview_failed",
                            status=ProductStatus.RUNNING,
                            current=checkpoint_step,
                            message=f"Exploratory preview failed and training continued: {exc}",
                            metrics={"checkpoint_step": checkpoint_step, "training_seed": training_seed},
                        )
                    )
        return tuple(events)

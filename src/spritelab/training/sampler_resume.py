"""Versioned exact-resume sampler state for single-process training."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sized
from typing import Any

SAMPLER_STATE_VERSION = "spritelab_sampler_state_v1"


class UnsupportedExactResumeError(RuntimeError):
    """Raised when a loader topology cannot provide exact resume guarantees."""


class StatefulPermutationSampler:
    """Random permutation sampler whose in-epoch cursor is serializable."""

    def __init__(self, data_source: Sized, *, generator: Any, rank: int = 0, world_size: int = 1) -> None:
        if int(world_size) != 1 or int(rank) != 0:
            raise UnsupportedExactResumeError("exact sampler resume currently supports only rank=0, world_size=1")
        self.data_source = data_source
        self.generator = generator
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.permutation: list[int] = []
        self.sample_cursor = 0

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.sample_cursor)

    def __iter__(self) -> Iterator[int]:
        import torch

        size = len(self.data_source)
        if len(self.permutation) != size or self.sample_cursor >= size:
            if self.permutation:
                self.epoch += 1
            self.permutation = torch.randperm(size, generator=self.generator).tolist()
            self.sample_cursor = 0
        while self.sample_cursor < size:
            index = int(self.permutation[self.sample_cursor])
            self.sample_cursor += 1
            yield index

    def state_dict(
        self,
        *,
        batch_size: int,
        loader_generator: Any,
        accumulation_position: int = 0,
        invalid_batch_count: int = 0,
        skipped_batch_reasons: Mapping[str, int] | None = None,
        num_workers: int = 0,
        worker_seed_base: int = 0,
    ) -> dict[str, Any]:
        return {
            "schema_version": SAMPLER_STATE_VERSION,
            "epoch": int(self.epoch),
            "batch_index": int(self.sample_cursor // max(1, int(batch_size))),
            "sample_cursor": int(self.sample_cursor),
            "permutation": list(self.permutation),
            "sampler_generator_state": self.generator.get_state(),
            "dataloader_generator_state": loader_generator.get_state(),
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_size": len(self.data_source),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "worker_seed_derivation": {
                "algorithm": "torch_dataloader_base_seed_plus_worker_id",
                "base_seed": int(worker_seed_base),
            },
            "gradient_accumulation_position": int(accumulation_position),
            "invalid_batch_count": int(invalid_batch_count),
            "skipped_batch_reasons": dict(skipped_batch_reasons or {}),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        batch_size: int,
        loader_generator: Any,
        num_workers: int = 0,
        unsafe: bool = False,
    ) -> None:
        version = state.get("schema_version")
        if version != SAMPLER_STATE_VERSION:
            raise UnsupportedExactResumeError(f"unsupported sampler state schema: {version!r}")
        problems: list[str] = []
        expected = {
            "dataset_size": len(self.data_source),
            "batch_size": int(batch_size),
            "rank": self.rank,
            "world_size": self.world_size,
        }
        for field, value in expected.items():
            if int(state.get(field, -1)) != value:
                problems.append(f"{field} checkpoint={state.get(field)!r} current={value!r}")
        saved_workers = int(state.get("num_workers", 0))
        if saved_workers != int(num_workers):
            problems.append(f"num_workers checkpoint={saved_workers} current={num_workers}")
        if int(num_workers) != 0:
            message = "exact mid-epoch resume is unsupported with num_workers > 0 because prefetch advances the sampler"
            if not unsafe:
                raise UnsupportedExactResumeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        accumulation = int(state.get("gradient_accumulation_position", 0))
        if accumulation != 0:
            problems.append("gradient_accumulation_position is non-zero but gradient tensors are not checkpointed")
        if problems and not unsafe:
            raise UnsupportedExactResumeError("incompatible sampler resume: " + "; ".join(problems))
        if problems:
            warnings.warn("unsafe sampler resume: " + "; ".join(problems), RuntimeWarning, stacklevel=2)
        permutation = [int(value) for value in state.get("permutation", ())]
        cursor = int(state.get("sample_cursor", 0))
        if len(permutation) != len(self.data_source) or not 0 <= cursor <= len(permutation):
            raise UnsupportedExactResumeError("sampler permutation/cursor does not match the current dataset")
        self.epoch = int(state.get("epoch", 0))
        self.permutation = permutation
        self.sample_cursor = cursor
        self.generator.set_state(state["sampler_generator_state"])
        loader_generator.set_state(state["dataloader_generator_state"])


def validate_worker_mode(*, num_workers: int, exact_resume: bool, unsafe: bool = False) -> list[str]:
    if not exact_resume or int(num_workers) == 0:
        return []
    message = "exact mid-epoch resume is unsupported with num_workers > 0"
    if not unsafe:
        raise UnsupportedExactResumeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return [message]

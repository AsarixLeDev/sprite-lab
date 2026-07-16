"""Immutable multisource dataset-v5 assembly."""

from importlib import import_module
from typing import Any

from spritelab.dataset_v5.builder import BuilderConfig, build_dataset, canonical_rgba_sha256, verify_dataset
from spritelab.dataset_v5.policy_v2 import PolicyV2Config, build_policy_preview, verify_policy_preview

__all__ = [
    "BuilderConfig",
    "DatasetV5ViewError",
    "PolicyV2Config",
    "build_dataset",
    "build_policy_preview",
    "build_view",
    "canonical_rgba_sha256",
    "freeze_view",
    "validate_contract",
    "verify_dataset",
    "verify_freeze",
    "verify_policy_preview",
    "verify_view",
    "write_report",
]

_NAMED_VIEW_EXPORTS = {
    "DatasetV5ViewError",
    "build_view",
    "freeze_view",
    "validate_contract",
    "verify_freeze",
    "verify_view",
    "write_report",
}


def __getattr__(name: str) -> Any:
    if name not in _NAMED_VIEW_EXPORTS:
        raise AttributeError(name)
    return getattr(import_module("spritelab.dataset_v5.named_views"), name)

"""Immutable multisource dataset-v5 assembly."""

from spritelab.dataset_v5.builder import BuilderConfig, build_dataset, canonical_rgba_sha256, verify_dataset
from spritelab.dataset_v5.policy_v2 import PolicyV2Config, build_policy_preview, verify_policy_preview

__all__ = [
    "BuilderConfig",
    "PolicyV2Config",
    "build_dataset",
    "build_policy_preview",
    "canonical_rgba_sha256",
    "verify_dataset",
    "verify_policy_preview",
]

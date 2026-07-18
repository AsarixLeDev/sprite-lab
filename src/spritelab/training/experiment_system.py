"""Versioned, deterministic experiment specifications for generator ablations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

EXPERIMENT_MANIFEST_VERSION = "spritelab_experiment_v3"
UNSAFE_RESUME_REVOCATION_VERSION = "spritelab_unsafe_resume_revocation_v1"
ARCHITECTURE_SCHEMA_VERSION = "spritelab_generator_architecture_v2"
FORWARD_OUTPUT_SCHEMA_VERSION = "spritelab_generator_forward_v2"
CONDITIONING_SCHEMA_VERSION_V1 = "spritelab_conditioning_v1"
CONDITIONING_SCHEMA_VERSION = "spritelab_conditioning_v2"
EVALUATION_SUITE_VERSION = "spritelab_eval_v1"
TOKENIZER_VERSION = "sprite_text_tokenizer_v1"
RESUME_HARD_FIELDS = (
    "dataset_manifest_hash",
    "split_manifest_hash",
    "model_architecture_hash",
    "conditioning_schema_hash",
    "optimizer_identity_hash",
    "schedule_identity_hash",
    "micro_batch_size",
    "global_batch_size",
    "effective_batch_size",
    "gradient_accumulation_steps",
    "precision_policy",
    "autocast_policy",
    "ema_enabled",
    "ema_decay",
    "ema_identity_hash",
    "loss_configuration_hash",
    "auxiliary_heads_mode",
    "seed_identity_hash",
    "sampler_policy_hash",
    "determinism_policy_hash",
    "evaluation_cadence",
    "checkpoint_cadence",
    "max_optimizer_steps",
    "experiment_configuration_hash",
    "lineage_parent_identity",
)

LEGACY_EXPERIMENT_MANIFEST_VERSIONS = ("spritelab_experiment_v1", "spritelab_experiment_v2")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PRECISION_POLICIES = frozenset({"fp32", "fp16", "bf16", "amp"})
_AUXILIARY_HEAD_MODES = frozenset({"absent", "palette_index"})

# Resume classifications are deliberately explicit.  Hard fields affect exact
# replay or fair architectural comparison; warnings affect interpretation but
# not optimizer continuation; runtime information is descriptive and may vary
# safely between machines or output locations.
RESUME_FIELD_CLASSIFICATION = {
    "resume_hard": RESUME_HARD_FIELDS,
    "resume_warning": (
        "software_version",
        "augmentation_configuration",
        "timestep_validation_boundaries",
    ),
    "runtime_informational": (
        "name",
        "ablation",
        "dataset_manifest",
        "split_manifest",
        "hardware_summary",
    ),
}


class IncompatibleResumeError(RuntimeError):
    """Raised when a checkpoint belongs to a different experiment identity."""

    def __init__(self, message: str, *, mismatches: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.mismatches = tuple(mismatches)

    def to_report(self) -> dict[str, Any]:
        """Return the complete machine-readable failure, without parsing prose."""

        return {
            "schema_version": "spritelab_resume_compatibility_report_v1",
            "compatible": False,
            "mismatches": list(self.mismatches),
            "message": str(self),
        }


class UnsafeResumeMismatches(list[str]):
    """Backward-compatible mismatch list carrying an immutable revocation record."""

    def __init__(self, values: Sequence[str], record: Mapping[str, Any]) -> None:
        super().__init__(values)
        self.revocation_record = _deep_freeze(deepcopy(dict(record)))


def canonical_json(value: Any) -> str:
    """Serialize configuration deterministically for hashes and reviews."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conditioning_schema(*, mode: str, tokenizer: Mapping[str, Any], structured_vocab: Any = None) -> dict[str, Any]:
    fields = ["caption", "semantic"]
    if "structured" in str(mode):
        fields.extend(
            [
                "category_id",
                "object_id",
                "base_object_id",
                "primary_color_id",
                "color_multi_hot",
                "material_multi_hot",
                "shape_multi_hot",
                "function_multi_hot",
                "style_multi_hot",
            ]
        )
    schema = {
        "version": CONDITIONING_SCHEMA_VERSION,
        "mode": str(mode),
        "field_order": fields,
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_hash": stable_hash(tokenizer),
        "null_semantics": {
            "text": "all pad tokens",
            "structured_missing_id": 0,
            "structured_unknown_id": 1,
            "multi_hot_missing": "index 0 set",
            "multi_hot_unknown": "index 1 set",
        },
        "structured_vocab": structured_vocab,
    }
    schema["hash"] = stable_hash(schema)
    return schema


def architecture_identity(
    model_config: Mapping[str, Any],
    *,
    auxiliary_heads_instantiated: bool | None = None,
    parameter_names: Sequence[str] | None = None,
    state_dict_keys: Sequence[str] | None = None,
    parameter_count: int | None = None,
) -> dict[str, Any]:
    """Build the immutable physical model identity.

    A missing ``auxiliary_heads_mode`` is the documented legacy adapter: it
    preserves the historical, physically enabled palette/index heads and marks
    the identity as legacy.  Loss weights never select physical construction.
    """
    raw_config = dict(model_config)
    explicit_mode = raw_config.get("auxiliary_heads_mode")
    legacy = explicit_mode is None
    mode = "palette_index" if legacy else str(explicit_mode)
    if mode not in {"absent", "palette_index"}:
        raise ValueError("auxiliary_heads_mode must be 'absent' or 'palette_index'")
    instantiated = mode == "palette_index"
    if auxiliary_heads_instantiated is not None and bool(auxiliary_heads_instantiated) != instantiated:
        raise ValueError("auxiliary head instantiation disagrees with auxiliary_heads_mode")
    normalized_config = dict(raw_config)
    normalized_config["auxiliary_heads_mode"] = mode
    names = None if parameter_names is None else list(parameter_names)
    keys = None if state_dict_keys is None else list(state_dict_keys)
    identity = {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "model_type": "generator_challenger",
        "base_model_family": str(raw_config.get("architecture", "rectified_flow")),
        "sprite_size": int(raw_config.get("sprite_size", 32)),
        "model_config": normalized_config,
        "channel_widths_depth": {
            "base_channels": raw_config.get("base_channels"),
            "channel_mults": raw_config.get("channel_mults"),
            "res_blocks_per_level": raw_config.get("res_blocks_per_level"),
        },
        "conditioning_dimensions": {
            "vocab_size": raw_config.get("vocab_size"),
            "embed_dim": raw_config.get("embed_dim"),
            "palette_conditioning_dim": raw_config.get("palette_conditioning_dim"),
        },
        "text_pooling_configuration": "masked_mean_v1",
        "structured_conditioning_schema": raw_config.get("structured_vocab_sizes"),
        "auxiliary_heads_mode": mode,
        "auxiliary_heads_instantiated": instantiated,
        "auxiliary_head_configuration": {"palette_slots": 16, "index_classes": 16} if instantiated else None,
        "identity_kind": "legacy_adapter" if legacy else "explicit",
        "promotion_eligible": not legacy,
        "parameter_count": None if parameter_count is None else int(parameter_count),
        "ordered_parameter_name_hash": None if names is None else stable_hash(names),
        "state_dict_key_hash": None if keys is None else stable_hash(keys),
        "forward_output_schema_version": FORWARD_OUTPUT_SCHEMA_VERSION,
    }
    identity["hash"] = stable_hash(identity)
    return identity


def measured_architecture_identity(model: Any) -> dict[str, Any]:
    """Measure names, keys and parameter ownership from a constructed model."""
    names = [name for name, _parameter in model.named_parameters()]
    keys = list(model.state_dict())
    config = model.config()
    if bool(getattr(model, "legacy_auxiliary_heads_adapter", False)):
        config.pop("auxiliary_heads_mode", None)
    return architecture_identity(
        config,
        parameter_names=names,
        state_dict_keys=keys,
        parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
    )


def _measure_configured_architecture(
    model_config: Mapping[str, Any], *, tokenizer: Mapping[str, Any], structured_vocab: Any
) -> dict[str, Any]:
    """Construct the declared architecture on CPU without perturbing RNG state."""
    if torch is None:
        return architecture_identity(model_config)
    from spritelab.training.generator_challenger import RectifiedFlowUNet
    from spritelab.training.structured_conditioning import StructuredConditioningVocab

    token_to_id = tokenizer.get("token_to_id", {})
    if not isinstance(token_to_id, Mapping) or not token_to_id:
        raise ValueError("tokenizer must contain a nonempty token_to_id mapping")
    structured_sizes = None
    if isinstance(structured_vocab, Mapping):
        structured_sizes = StructuredConditioningVocab.from_json_dict(structured_vocab).sizes()
    palette = model_config.get("palette_conditioning", False)
    if isinstance(palette, Mapping):
        palette_enabled = bool(palette.get("enabled", False))
        palette_dropout = float(palette.get("dropout", 0.0))
    else:
        palette_enabled = bool(palette)
        palette_dropout = float(model_config.get("palette_conditioning_dropout", 0.0))
    kwargs = {
        "vocab_size": len(token_to_id),
        "embed_dim": int(model_config.get("embed_dim", 64)),
        "base_channels": int(model_config.get("base_channels", 64)),
        "channel_mults": tuple(int(value) for value in model_config.get("channel_mults", (1, 2, 4))),
        "res_blocks_per_level": int(model_config.get("res_blocks_per_level", 2)),
        "pad_token_id": int(token_to_id.get("<pad>", 0)),
        "structured_vocab_sizes": structured_sizes,
        "film_conditioning": bool(model_config.get("film_conditioning", False)),
        "bottleneck_attention": bool(model_config.get("bottleneck_attention", False)),
        "palette_conditioning": palette_enabled,
        "palette_conditioning_dropout": palette_dropout,
        "palette_conditioning_dim": int(model_config.get("palette_conditioning_dim", 64)),
        "palette_conditioning_inject": str(model_config.get("palette_conditioning_inject", "decoder")),
        "auxiliary_heads_mode": model_config.get("auxiliary_heads_mode"),
    }
    with torch.random.fork_rng(devices=[]):
        model = RectifiedFlowUNet(**kwargs)
    return measured_architecture_identity(model)


def software_version(repo: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo or Path.cwd())
    commit = "unknown"
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "package_version": "0.1.0",
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": None if torch is None else str(torch.__version__),
    }


def hardware_summary(*, target_device: str | None = None) -> dict[str, Any]:
    """Describe hardware visible to this process or to one smoke-device projection.

    Smoke preparation runs before the contained child receives its minimal
    environment.  Projecting ``cpu`` hides CUDA exactly as the CPU child does;
    projecting ``cuda`` records only device zero because the child is bound to
    ``CUDA_VISIBLE_DEVICES=0``.  Ordinary experiment manifests continue to
    describe every device visible to their current process.
    """

    if target_device not in {None, "cpu", "cuda"}:
        raise ValueError("hardware-summary target_device must be 'cpu', 'cuda', or None")
    cuda_requested = bool(target_device != "cpu" and torch is not None and torch.cuda.is_available())
    visible_cuda_count = torch.cuda.device_count() if cuda_requested else 0
    cuda = bool(cuda_requested and visible_cuda_count > 0)
    if not cuda:
        cuda_device_count = 0
        cuda_devices: list[str] = []
    elif target_device == "cuda":
        cuda_device_count = 1
        cuda_devices = [torch.cuda.get_device_name(0)]
    else:
        cuda_device_count = visible_cuda_count
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(cuda_device_count)]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": cuda_devices,
    }


def _validated_software_version(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("precomputed software version must be an object")
    result = deepcopy(dict(value))
    if set(result) != {"package_version", "git_commit", "git_dirty", "python", "torch"}:
        raise ValueError("precomputed software version has an inexact schema")
    for key in ("package_version", "python"):
        field = result[key]
        if not isinstance(field, str) or not field or len(field) > 256:
            raise ValueError(f"precomputed software version {key} is malformed")
    commit = result["git_commit"]
    if not isinstance(commit, str) or (commit != "unknown" and not _GIT_COMMIT_RE.fullmatch(commit)):
        raise ValueError("precomputed software version git_commit is malformed")
    if result["git_dirty"] is not None and type(result["git_dirty"]) is not bool:
        raise ValueError("precomputed software version git_dirty is malformed")
    torch_version = result["torch"]
    if torch_version is not None and (
        not isinstance(torch_version, str) or not torch_version or len(torch_version) > 256
    ):
        raise ValueError("precomputed software version torch is malformed")
    return result


def _validated_hardware_summary(
    value: Mapping[str, Any],
    *,
    target_device: Any = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("precomputed hardware summary must be an object")
    result = deepcopy(dict(value))
    if set(result) != {
        "platform",
        "machine",
        "cpu_count",
        "cuda_available",
        "cuda_device_count",
        "cuda_devices",
    }:
        raise ValueError("precomputed hardware summary has an inexact schema")
    for key in ("platform", "machine"):
        field = result[key]
        if not isinstance(field, str) or len(field) > 4096:
            raise ValueError(f"precomputed hardware summary {key} is malformed")
    cpu_count = result["cpu_count"]
    if cpu_count is not None and (type(cpu_count) is not int or cpu_count < 1):
        raise ValueError("precomputed hardware summary cpu_count is malformed")
    available = result["cuda_available"]
    count = result["cuda_device_count"]
    devices = result["cuda_devices"]
    if (
        type(available) is not bool
        or type(count) is not int
        or not 0 <= count <= 1024
        or not isinstance(devices, list)
        or len(devices) != count
        or any(not isinstance(name, str) or not name or len(name) > 4096 for name in devices)
        or available != (count > 0)
    ):
        raise ValueError("precomputed hardware summary CUDA projection is malformed")
    if target_device == "cpu" and (available or count != 0 or devices):
        raise ValueError("precomputed CPU hardware summary must hide CUDA")
    if target_device == "cuda" and available and count != 1:
        raise ValueError("precomputed CUDA hardware summary must bind exactly one visible device")
    return result


def _strict_manifest_integer(value: Any, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer; bool and coercible strings are forbidden")
    return value


def build_experiment_manifest(
    config: Mapping[str, Any],
    *,
    dataset_manifest: str | Path,
    split_manifest: str | Path | None = None,
    tokenizer: Mapping[str, Any],
    structured_vocab: Any = None,
    repo: str | Path | None = None,
    checkpoint_lineage: Sequence[str] = (),
    dataset_manifest_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    software_version_override: Mapping[str, Any] | None = None,
    hardware_summary_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    # Config-only manifests retain null measured fields.  Production training
    # replaces this identity with measured_architecture_identity(model) before
    # checkpointing, so checkpoint identities always bind physical tensors.
    architecture = _measure_configured_architecture(
        model_config, tokenizer=tokenizer, structured_vocab=structured_vocab
    )
    conditioning = conditioning_schema(
        mode=str(config.get("conditioning", {}).get("mode", "caption_semantic")),
        tokenizer=tokenizer,
        structured_vocab=structured_vocab,
    )
    dataset_path = Path(dataset_manifest)
    split_path = Path(split_manifest or dataset_manifest)
    dataset_hash = file_sha256(dataset_path) if dataset_manifest_sha256 is None else dataset_manifest_sha256
    split_hash = file_sha256(split_path) if split_manifest_sha256 is None else split_manifest_sha256
    if not isinstance(dataset_hash, str) or not _SHA256_RE.fullmatch(dataset_hash):
        raise ValueError("dataset manifest SHA-256 is malformed")
    if not isinstance(split_hash, str) or not _SHA256_RE.fullmatch(split_hash):
        raise ValueError("split manifest SHA-256 is malformed")
    optimizer = deepcopy(config.get("optimizer", {}))
    loss = deepcopy(config.get("loss", {}))
    runtime = deepcopy(config.get("runtime", {}))
    ema = deepcopy(config.get("ema", {}))
    if not isinstance(optimizer, Mapping) or not optimizer:
        raise ValueError("optimizer configuration must be a nonempty object")
    optimizer_name = optimizer.get("name")
    if not isinstance(optimizer_name, str) or not optimizer_name.strip():
        raise ValueError("optimizer name must be a nonempty string")
    schedule_name = optimizer.get("schedule", "none")
    if schedule_name not in {"none", "cosine"}:
        raise ValueError("optimizer schedule must be 'none' or 'cosine'")
    warmup_steps = _strict_manifest_integer(optimizer.get("warmup_steps", 0), "optimizer warmup_steps", minimum=0)
    optimizer["warmup_steps"] = warmup_steps
    if not isinstance(loss, Mapping) or not loss:
        raise ValueError("loss configuration must be a nonempty object")
    ema_decay_value = ema.get("decay", 0.0)
    if isinstance(ema_decay_value, bool) or not isinstance(ema_decay_value, (int, float)):
        raise ValueError("EMA decay must be numeric and may not be a coercible string")
    ema_decay = float(ema_decay_value)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    ema_enabled_value = ema.get("enabled", ema_decay > 0.0)
    if type(ema_enabled_value) is not bool:
        raise ValueError("EMA enabled must be a boolean")
    ema_enabled = ema_enabled_value
    ema["enabled"] = ema_enabled
    ema["decay"] = ema_decay
    seeds = deepcopy(config.get("seeds", {}))
    if not isinstance(seeds, Mapping) or not seeds or any(type(value) is not int for value in seeds.values()):
        raise ValueError("seed identity must be a nonempty object of integers; bool and strings are forbidden")
    sampling_policy = deepcopy(
        config.get("sampler", config.get("sampling_policy", {"name": "stateful_permutation_v1", "shuffle": True}))
    )
    if not isinstance(sampling_policy, Mapping) or not sampling_policy:
        raise ValueError("sampler policy must be a nonempty object")
    determinism_policy = runtime.get("determinism", "off")
    if determinism_policy not in {"off", "warn", "strict"}:
        raise ValueError("determinism policy must be one of: off, warn, strict")
    precision_policy = runtime.get("precision", "fp32")
    if precision_policy not in _PRECISION_POLICIES:
        raise ValueError(f"precision policy must be one of: {sorted(_PRECISION_POLICIES)}")
    autocast_policy = deepcopy(runtime.get("autocast_policy", {"enabled": precision_policy != "fp32"}))
    if not isinstance(autocast_policy, Mapping) or type(autocast_policy.get("enabled")) is not bool:
        raise ValueError("autocast policy must be an object with an explicit boolean enabled value")
    if "dtype" in autocast_policy and autocast_policy["dtype"] not in {"fp16", "bf16"}:
        raise ValueError("autocast dtype must be fp16 or bf16 when present")
    gradient_accumulation = _strict_manifest_integer(
        runtime.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps", minimum=1
    )
    micro_batch = _strict_manifest_integer(
        runtime.get("micro_batch_size", runtime.get("batch_size")), "micro_batch_size", minimum=1
    )
    global_batch = _strict_manifest_integer(runtime.get("batch_size", micro_batch), "global_batch_size", minimum=1)
    effective_batch = _strict_manifest_integer(
        runtime.get("effective_batch_size", global_batch * gradient_accumulation),
        "effective_batch_size",
        minimum=1,
    )
    evaluation_cadence = _strict_manifest_integer(runtime.get("sample_every", 0), "evaluation_cadence", minimum=0)
    checkpoint_cadence = _strict_manifest_integer(runtime.get("save_every", 0), "checkpoint_cadence", minimum=0)
    max_optimizer_steps = _strict_manifest_integer(runtime.get("max_steps"), "max_optimizer_steps", minimum=1)
    lineage = list(checkpoint_lineage)
    lineage_parent_identity = lineage[-1] if lineage else "root"
    hard_config = {
        "model": architecture,
        "conditioning_hash": conditioning["hash"],
        "loss": loss,
        "optimizer": optimizer,
        "micro_batch_size": micro_batch,
        "batch_size": global_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "precision": precision_policy,
        "autocast_policy": autocast_policy,
        "ema": ema,
        "seeds": seeds,
        "sampler": sampling_policy,
        "determinism": determinism_policy,
        "evaluation_cadence": evaluation_cadence,
        "checkpoint_cadence": checkpoint_cadence,
        "max_optimizer_steps": max_optimizer_steps,
        "lineage_parent_identity": lineage_parent_identity,
    }
    manifest = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "name": str(config.get("name", "unnamed")),
        "ablation": str(config.get("ablation", "baseline")),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_hash": dataset_hash,
        "split_manifest": str(split_path),
        "split_manifest_hash": split_hash,
        "model_architecture": architecture,
        "model_architecture_hash": architecture["hash"],
        "conditioning_schema": conditioning,
        "conditioning_schema_hash": conditioning["hash"],
        "loss_configuration": loss,
        "loss_configuration_hash": stable_hash(loss),
        "optimizer_configuration": optimizer,
        "optimizer_identity_hash": stable_hash(optimizer),
        "schedule_identity_hash": stable_hash({"schedule": schedule_name, "warmup_steps": warmup_steps}),
        "augmentation_configuration": deepcopy(config.get("augmentation", {})),
        "random_seeds": seeds,
        "seed_identity_hash": stable_hash(seeds),
        "software_version": (
            software_version(repo)
            if software_version_override is None
            else _validated_software_version(software_version_override)
        ),
        "precision_mode": precision_policy,
        "precision_policy": precision_policy,
        "autocast_policy": autocast_policy,
        "hardware_summary": (
            hardware_summary()
            if hardware_summary_override is None
            else _validated_hardware_summary(
                hardware_summary_override,
                target_device=runtime.get("device"),
            )
        ),
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "checkpoint_lineage": lineage,
        "lineage_parent_identity": lineage_parent_identity,
        "sampling": deepcopy(config.get("sampling", {})),
        "ema": ema,
        "ema_identity_hash": stable_hash(ema),
        "ema_enabled": ema_enabled,
        "ema_decay": float(ema.get("decay", 0.0)),
        "timestep_sampling": deepcopy(config.get("timestep_sampling", {"strategy": "uniform"})),
        "determinism_mode": determinism_policy,
        "determinism_policy": determinism_policy,
        "determinism_policy_hash": stable_hash(determinism_policy),
        "sampler_policy": sampling_policy,
        "sampler_policy_hash": stable_hash(sampling_policy),
        "micro_batch_size": micro_batch,
        "global_batch_size": global_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "evaluation_cadence": evaluation_cadence,
        "checkpoint_cadence": checkpoint_cadence,
        "max_optimizer_steps": max_optimizer_steps,
        "auxiliary_heads_mode": architecture["auxiliary_heads_mode"],
        "legacy_architecture": architecture["identity_kind"] == "legacy_adapter",
        "fair_architecture_comparison_eligible": bool(architecture["promotion_eligible"]),
        "checkpoint_promotion_eligible": bool(architecture["promotion_eligible"]),
        "resume_field_classification": RESUME_FIELD_CLASSIFICATION,
        "timestep_validation_boundaries": deepcopy(
            config.get("runtime", {}).get("timestep_validation_boundaries", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ),
    }
    manifest["experiment_configuration_hash"] = stable_hash(hard_config)
    manifest["experiment_hash"] = stable_hash(manifest)
    return manifest


def validate_resume_compatibility(
    current: Mapping[str, Any],
    saved: Mapping[str, Any],
    *,
    unsafe: bool = False,
    unsafe_reason: str | None = None,
    unsafe_record: dict[str, Any] | None = None,
    additional_mismatches: Sequence[str] = (),
) -> list[str]:
    """Validate exact current-schema resume or create an explicit revocation.

    A current manifest is current only when it declares the one supported
    schema version.  Legacy and versionless manifests are never inferred from
    their shape.  All schema and content mismatches are collected before the
    operation is rejected so callers receive complete structured evidence.
    """

    mismatches: list[str] = []
    for label, manifest in (("current", current), ("checkpoint", saved)):
        version = manifest.get("manifest_version")
        if version != EXPERIMENT_MANIFEST_VERSION:
            suffix = "missing" if version is None else "unsupported"
            mismatches.append(f"{label}.manifest_version.{suffix}")
        mismatches.extend(_protected_field_mismatches(label, manifest))
        mismatches.extend(_manifest_content_mismatches(label, manifest))
    direct_mismatches = [field for field in RESUME_HARD_FIELDS if current.get(field) != saved.get(field)]
    mismatches.extend(direct_mismatches)
    mismatches.extend(additional_mismatches)
    mismatches = list(dict.fromkeys(mismatches))
    if mismatches and not unsafe:
        detail = ", ".join(mismatches)
        raise IncompatibleResumeError(
            f"incompatible resume ({detail}); pass --unsafe-resume with an explicit reason to override",
            mismatches=mismatches,
        )
    if unsafe:
        record = create_unsafe_resume_revocation(
            reason=unsafe_reason,
            mismatches=mismatches,
            source_checkpoint_identity=saved.get("experiment_hash"),
            target_runtime_identity=current.get("experiment_hash"),
        )
        if unsafe_record is not None:
            unsafe_record.update(record)
        return UnsafeResumeMismatches(direct_mismatches or mismatches, record)
    return mismatches


def create_unsafe_resume_revocation(
    *,
    reason: str | None,
    mismatches: Sequence[str],
    source_checkpoint_identity: Any,
    target_runtime_identity: Any,
    event_identity: str | None = None,
) -> Mapping[str, Any]:
    """Create the one mandatory, append-only unsafe-resume revocation record."""

    if not isinstance(reason, str) or not reason.strip():
        raise IncompatibleResumeError("unsafe resume requires an explicit recorded reason")
    mismatch_list = list(mismatches)
    if not mismatch_list:
        raise IncompatibleResumeError("unsafe resume is unnecessary without a detected mismatch")
    if any(not isinstance(item, str) or not item.strip() for item in mismatch_list):
        raise IncompatibleResumeError("unsafe resume mismatches must be nonempty strings")
    for label, identity in (
        ("source checkpoint", source_checkpoint_identity),
        ("target runtime", target_runtime_identity),
    ):
        if not isinstance(identity, str) or not _SHA256_RE.fullmatch(identity):
            raise IncompatibleResumeError(f"unsafe resume requires a concrete SHA-256 {label} identity")
    timestamp = event_identity or datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": UNSAFE_RESUME_REVOCATION_VERSION,
        "unsafe_resume": True,
        "reason": reason.strip(),
        "all_mismatches": mismatch_list,
        "source_checkpoint_identity": source_checkpoint_identity,
        "target_runtime_identity": target_runtime_identity,
        "timestamp": timestamp,
        "exact_replay_eligible": False,
        "fair_comparison_eligible": False,
        "promotion_eligible": False,
        "event_identity": timestamp,
        # Retained aliases keep older report readers honest while the v3 names
        # above are the authoritative eligibility contract.
        "mismatches": mismatch_list,
        "exact_replay_claimed": False,
        "fair_architecture_comparison_eligible": False,
        "checkpoint_promotion_eligible": False,
    }
    payload["revocation_identity"] = stable_hash(payload)
    return _deep_freeze(payload)


def write_unsafe_resume_revocation(directory: str | Path, record: Mapping[str, Any]) -> Path:
    """Persist one immutable revocation as a new append-only record file."""

    identity = record.get("revocation_identity")
    if not isinstance(identity, str) or not _SHA256_RE.fullmatch(identity):
        raise IncompatibleResumeError("unsafe resume revocation has no concrete record identity")
    target = Path(directory) / "unsafe_resume_revocations" / f"{identity}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise IncompatibleResumeError(f"unsafe resume revocation already exists: {target}") from exc
    return target


def validate_resume_against_runtime(
    saved: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    dataset_manifest: str | Path,
    split_manifest: str | Path | None,
    tokenizer: Mapping[str, Any],
    structured_vocab: Any = None,
    unsafe: bool = False,
    unsafe_reason: str | None = None,
    unsafe_record: dict[str, Any] | None = None,
) -> list[str]:
    """Rebuild identity from runtime configuration and actual input bytes."""

    runtime = build_experiment_manifest(
        config,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        tokenizer=tokenizer,
        structured_vocab=structured_vocab,
        checkpoint_lineage=list(saved.get("checkpoint_lineage") or []),
    )
    return validate_resume_compatibility(
        runtime,
        saved,
        unsafe=unsafe,
        unsafe_reason=unsafe_reason,
        unsafe_record=unsafe_record,
    )


def _protected_field_mismatches(label: str, manifest: Mapping[str, Any]) -> list[str]:
    mismatches: list[str] = []
    hash_fields = {
        "dataset_manifest_hash",
        "split_manifest_hash",
        "model_architecture_hash",
        "conditioning_schema_hash",
        "optimizer_identity_hash",
        "schedule_identity_hash",
        "ema_identity_hash",
        "loss_configuration_hash",
        "seed_identity_hash",
        "sampler_policy_hash",
        "determinism_policy_hash",
        "experiment_configuration_hash",
    }
    positive_integer_fields = {
        "micro_batch_size",
        "global_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "max_optimizer_steps",
    }
    nonnegative_integer_fields = {"evaluation_cadence", "checkpoint_cadence"}
    for field in RESUME_HARD_FIELDS:
        if field not in manifest:
            mismatches.append(f"{label}.{field}.missing")
            continue
        value = manifest[field]
        if value is None:
            mismatches.append(f"{label}.{field}.null")
            continue
        if field in hash_fields and (not isinstance(value, str) or not _SHA256_RE.fullmatch(value)):
            mismatches.append(f"{label}.{field}.malformed_sha256")
        elif field in positive_integer_fields and (type(value) is not int or value <= 0):
            mismatches.append(f"{label}.{field}.invalid_positive_integer")
        elif field in nonnegative_integer_fields and (type(value) is not int or value < 0):
            mismatches.append(f"{label}.{field}.invalid_cadence")
        elif field == "precision_policy" and (not isinstance(value, str) or value not in _PRECISION_POLICIES):
            mismatches.append(f"{label}.{field}.unsupported")
        elif field == "autocast_policy":
            if not isinstance(value, Mapping) or not value or type(value.get("enabled")) is not bool:
                mismatches.append(f"{label}.{field}.malformed")
            elif "dtype" in value and value["dtype"] not in {"fp16", "bf16"}:
                mismatches.append(f"{label}.{field}.unsupported_dtype")
        elif field == "ema_enabled" and type(value) is not bool:
            mismatches.append(f"{label}.{field}.invalid_boolean")
        elif field == "ema_decay" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) < 1.0
        ):
            mismatches.append(f"{label}.{field}.invalid_decay")
        elif field == "auxiliary_heads_mode" and value not in _AUXILIARY_HEAD_MODES:
            mismatches.append(f"{label}.{field}.unsupported")
        elif field == "lineage_parent_identity" and (
            not isinstance(value, str) or (value != "root" and not _SHA256_RE.fullmatch(value))
        ):
            mismatches.append(f"{label}.{field}.malformed")
    return mismatches


def _manifest_content_mismatches(label: str, manifest: Mapping[str, Any]) -> list[str]:
    mismatches: list[str] = []
    experiment_identity = manifest.get("experiment_hash")
    if not isinstance(experiment_identity, str) or not _SHA256_RE.fullmatch(experiment_identity):
        mismatches.append(f"{label}.experiment_hash.malformed_sha256")
    elif experiment_identity != stable_hash(
        {key: value for key, value in manifest.items() if key != "experiment_hash"}
    ):
        mismatches.append(f"{label}.experiment_hash.content_hash")
    architecture = manifest.get("model_architecture")
    if isinstance(architecture, Mapping):
        architecture_content = {key: value for key, value in architecture.items() if key != "hash"}
        if architecture.get("hash") != stable_hash(architecture_content):
            mismatches.append(f"{label}.tampered_model_architecture_hash")
    checks = {
        "model_architecture_hash": (
            manifest.get("model_architecture"),
            lambda value: value.get("hash") if isinstance(value, Mapping) else None,
        ),
        "conditioning_schema_hash": (manifest.get("conditioning_schema"), stable_hash),
        "optimizer_identity_hash": (manifest.get("optimizer_configuration"), stable_hash),
        "loss_configuration_hash": (manifest.get("loss_configuration"), stable_hash),
        "seed_identity_hash": (manifest.get("random_seeds"), stable_hash),
        "sampler_policy_hash": (manifest.get("sampler_policy"), stable_hash),
    }
    for field, (content, hasher) in checks.items():
        if not isinstance(content, Mapping) or not content:
            mismatches.append(f"{label}.{field}.content_missing")
            continue
        if field == "conditioning_schema_hash" and isinstance(content, Mapping):
            content = {key: value for key, value in content.items() if key != "hash"}
            expected = manifest.get("conditioning_schema", {}).get("hash")
            if stable_hash(content) != expected:
                mismatches.append(f"{label}.conditioning_schema.content_hash")
        calculated = hasher(content)
        if manifest.get(field) != calculated:
            mismatches.append(f"{label}.{field}.content_hash")
    for hash_field, path_field in (
        ("dataset_manifest_hash", "dataset_manifest"),
        ("split_manifest_hash", "split_manifest"),
    ):
        path = manifest.get(path_field)
        if not path or not Path(str(path)).is_file():
            mismatches.append(f"{label}.{path_field}.missing_file")
        elif manifest.get(hash_field) != file_sha256(Path(str(path))):
            mismatches.append(f"{label}.{hash_field}.file_hash")
    optimizer = manifest.get("optimizer_configuration")
    if isinstance(optimizer, Mapping) and optimizer:
        schedule = {"schedule": optimizer.get("schedule", "none"), "warmup_steps": optimizer.get("warmup_steps", 0)}
        if manifest.get("schedule_identity_hash") != stable_hash(schedule):
            mismatches.append(f"{label}.schedule_identity_hash.content_hash")
        if not isinstance(optimizer.get("name"), str) or not optimizer["name"].strip():
            mismatches.append(f"{label}.optimizer_configuration.name")
        if optimizer.get("schedule", "none") not in {"none", "cosine"}:
            mismatches.append(f"{label}.optimizer_configuration.schedule")
        if type(optimizer.get("warmup_steps", 0)) is not int or optimizer.get("warmup_steps", 0) < 0:
            mismatches.append(f"{label}.optimizer_configuration.warmup_steps")
    else:
        mismatches.append(f"{label}.optimizer_configuration.missing")
    seeds = manifest.get("random_seeds")
    if isinstance(seeds, Mapping) and seeds:
        if any(not isinstance(key, str) or not key.strip() or type(value) is not int for key, value in seeds.items()):
            mismatches.append(f"{label}.random_seeds.malformed")
    else:
        mismatches.append(f"{label}.random_seeds.missing")
    sampler_policy = manifest.get("sampler_policy")
    if not isinstance(sampler_policy, Mapping) or not sampler_policy:
        mismatches.append(f"{label}.sampler_policy.missing")
    determinism_policy = manifest.get("determinism_policy")
    if determinism_policy not in {"off", "warn", "strict"}:
        mismatches.append(f"{label}.determinism_policy.unsupported")
    if manifest.get("determinism_policy_hash") != stable_hash(determinism_policy):
        mismatches.append(f"{label}.determinism_policy_hash.content_hash")
    ema = manifest.get("ema")
    if isinstance(ema, Mapping) and ema:
        enabled = bool(ema.get("enabled", float(ema.get("decay", 0.0)) > 0.0))
        decay = ema.get("decay")
        if (
            type(ema.get("enabled")) is not bool
            or isinstance(decay, bool)
            or not isinstance(decay, (int, float))
            or manifest.get("ema_enabled") != enabled
            or manifest.get("ema_decay") != float(decay)
        ):
            mismatches.append(f"{label}.ema.policy_content")
    else:
        mismatches.append(f"{label}.ema.missing")
    try:
        global_batch = manifest.get("global_batch_size")
        accumulation = manifest.get("gradient_accumulation_steps")
        effective = manifest.get("effective_batch_size")
        if any(type(value) is not int for value in (global_batch, accumulation, effective)):
            raise TypeError
        expected_effective = global_batch * accumulation
        if effective != expected_effective:
            mismatches.append(f"{label}.batch_configuration")
    except (TypeError, ValueError):
        mismatches.append(f"{label}.batch_configuration")
    return mismatches


def adapt_conditioning_schema_v1_manifest(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Return a marked, non-mutating compatibility view of an old manifest.

    The adapter does not pretend v1 is v2 and therefore retains the original
    conditioning hash.  Callers must still use ``unsafe=True`` to cross schemas.
    """
    schema = saved.get("conditioning_schema")
    if not isinstance(schema, Mapping) or schema.get("version") != CONDITIONING_SCHEMA_VERSION_V1:
        raise ValueError("conditioning schema-v1 adapter requires a schema-v1 experiment manifest")
    adapted = deepcopy(dict(saved))
    adapted["compatibility_adapter"] = {
        "from": CONDITIONING_SCHEMA_VERSION_V1,
        "semantics": "ID 0 remains legacy missing/OOV; no IDs were remapped",
    }
    return adapted


def validate_inference_parity(current: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    """Require inference to use the dataset/model/conditioning identity it declares."""
    saved = checkpoint.get("experiment_manifest")
    if not isinstance(saved, Mapping):
        raise IncompatibleResumeError("checkpoint has no experiment manifest; use the legacy sampler explicitly")
    current_version = (current.get("conditioning_schema") or {}).get("version")
    saved_version = (saved.get("conditioning_schema") or {}).get("version")
    if current_version != saved_version:
        raise IncompatibleResumeError(
            f"conditioning schema mismatch: inference={current_version!r}, checkpoint={saved_version!r}"
        )
    validate_resume_compatibility(current, saved)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "schema_version": "spritelab.numpy-rng-state.v1",
            "bit_generator": str(numpy_state[0]),
            "keys": [int(value) for value in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, Mapping):
        if numpy_state.get("schema_version") != "spritelab.numpy-rng-state.v1":
            raise ValueError("unsupported NumPy RNG-state schema")
        keys = numpy_state.get("keys")
        if not isinstance(keys, (list, tuple)) or not all(type(value) is int for value in keys):
            raise ValueError("NumPy RNG-state keys must be primitive integers")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(keys, dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    elif isinstance(numpy_state, (tuple, list)):
        # Compatibility is limited to a legacy state already present in trusted
        # memory. Checkpoint loading never falls back to unsafe pickle mode.
        np.random.set_state(tuple(numpy_state))
    else:
        raise ValueError("NumPy RNG state is malformed")
    if torch is not None and "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("experiment config must be a mapping")
    return dict(data)


def validate_ablation_config(config: Mapping[str, Any]) -> None:
    required = {
        "name",
        "ablation",
        "dataset",
        "model",
        "conditioning",
        "loss",
        "optimizer",
        "augmentation",
        "seeds",
        "runtime",
        "sampling",
        "ema",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"experiment config missing fields: {', '.join(missing)}")
    if config["ablation"] not in ablation_registry():
        raise ValueError(f"unknown ablation {config['ablation']!r}")
    if config.get("self_conditioning"):
        raise ValueError("self-conditioning is not implemented safely by RectifiedFlowUNet")
    schedule = str(config.get("noise_schedule", "rectified_flow_linear_path"))
    if schedule != "rectified_flow_linear_path":
        raise ValueError("alternative noise schedules are not supported cleanly by the current sampler")
    if str(config.get("timestep_sampling", {}).get("strategy", "uniform")) != "uniform":
        raise ValueError("only uniform timestep sampling is currently implemented")
    model = config.get("model", {})
    mode = model.get("auxiliary_heads_mode") if isinstance(model, Mapping) else None
    if mode is not None and str(mode) not in {"absent", "palette_index"}:
        raise ValueError("model.auxiliary_heads_mode must be 'absent' or 'palette_index'")
    loss = config.get("loss", {})
    auxiliary_weights = (
        "index_head_weight",
        "palette_head_weight",
        "palette_presence_weight",
    )
    if mode == "absent" and isinstance(loss, Mapping):
        nonzero = [field for field in auxiliary_weights if float(loss.get(field, 0.0)) != 0.0]
        if nonzero:
            raise ValueError(
                "model.auxiliary_heads_mode='absent' is incompatible with nonzero auxiliary loss fields: "
                + ", ".join(nonzero)
            )


def ablation_registry() -> dict[str, dict[str, Any]]:
    return {
        "baseline": {"factor": "none", "overrides": {}},
        "aux_heads_off": {"factor": "auxiliary_loss", "overrides": {"loss.auxiliary_heads": False}},
        "aux_heads_on": {"factor": "auxiliary_loss", "overrides": {"loss.auxiliary_heads": True}},
        "cfg_dropout_0": {"factor": "cfg_dropout", "overrides": {"conditioning.cfg_dropout": 0.0}},
        "cfg_dropout_0_2": {"factor": "cfg_dropout", "overrides": {"conditioning.cfg_dropout": 0.2}},
        "cfg_scale_1": {"factor": "inference_cfg", "overrides": {"sampling.cfg_scale": 1.0}},
        "cfg_scale_3": {"factor": "inference_cfg", "overrides": {"sampling.cfg_scale": 3.0}},
        "sampling_steps_15": {"factor": "sampling_steps", "overrides": {"sampling.steps": 15}},
        "sampling_steps_30": {"factor": "sampling_steps", "overrides": {"sampling.steps": 30}},
        "foreground_weighted_loss": {"factor": "loss_weighting", "overrides": {"loss.strategy": "foreground_weighted"}},
        "uniform_timestep": {"factor": "timestep_sampling", "overrides": {"timestep_sampling.strategy": "uniform"}},
        "ema_off": {"factor": "ema", "overrides": {"ema.decay": 0.0}},
        "ema_0_999": {"factor": "ema", "overrides": {"ema.decay": 0.999}},
        "field_dropout": {"factor": "field_dropout", "overrides": {"conditioning.field_dropout": {"object_id": 0.2}}},
        "palette_conditioning": {"factor": "palette_conditioning", "overrides": {"conditioning.palette.enabled": True}},
        "object_hierarchy": {
            "factor": "object_hierarchy",
            "overrides": {"conditioning.mode": "caption_semantic_structured"},
        },
        "palette_swap_augmentation": {
            "factor": "augmentation",
            "overrides": {"augmentation.palette_swap_probability": 0.2},
        },
    }


def windows_command(arguments: Sequence[str | Path]) -> str:
    """Return a PowerShell-safe, copy/paste command without POSIX quoting."""

    def quote(value: str | Path) -> str:
        text = str(value)
        if not text or any(char.isspace() or char in "'`$&|;()[]{}" for char in text):
            return "'" + text.replace("'", "''") + "'"
        return text

    return " ".join(quote(item) for item in arguments)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value

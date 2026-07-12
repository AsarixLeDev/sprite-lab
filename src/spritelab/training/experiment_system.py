"""Versioned, deterministic experiment specifications for generator ablations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

EXPERIMENT_MANIFEST_VERSION = "spritelab_experiment_v2"
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
    "global_batch_size",
    "effective_batch_size",
    "gradient_accumulation_steps",
    "precision_policy",
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


def hardware_summary() -> dict[str, Any]:
    cuda = bool(torch is not None and torch.cuda.is_available())
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda
        else [],
    }


def build_experiment_manifest(
    config: Mapping[str, Any],
    *,
    dataset_manifest: str | Path,
    split_manifest: str | Path | None = None,
    tokenizer: Mapping[str, Any],
    structured_vocab: Any = None,
    repo: str | Path | None = None,
    checkpoint_lineage: Sequence[str] = (),
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
    optimizer = deepcopy(config.get("optimizer", {}))
    loss = deepcopy(config.get("loss", {}))
    runtime = deepcopy(config.get("runtime", {}))
    ema = deepcopy(config.get("ema", {}))
    seeds = deepcopy(config.get("seeds", {}))
    sampling_policy = deepcopy(config.get("sampler", config.get("sampling_policy", {})))
    gradient_accumulation = int(runtime.get("gradient_accumulation_steps", 1))
    global_batch = int(runtime.get("batch_size", 0))
    effective_batch = int(runtime.get("effective_batch_size", global_batch * gradient_accumulation))
    lineage = list(checkpoint_lineage)
    hard_config = {
        "model": architecture,
        "conditioning_hash": conditioning["hash"],
        "loss": loss,
        "optimizer": optimizer,
        "batch_size": global_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "precision": runtime.get("precision", "fp32"),
        "ema": ema,
        "seeds": seeds,
        "sampler": sampling_policy,
        "determinism": runtime.get("determinism", "off"),
        "evaluation_cadence": runtime.get("sample_every", 0),
        "checkpoint_cadence": runtime.get("save_every", 0),
        "max_optimizer_steps": runtime.get("max_steps"),
        "lineage_parent_identity": lineage[-1] if lineage else None,
    }
    manifest = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "name": str(config.get("name", "unnamed")),
        "ablation": str(config.get("ablation", "baseline")),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_hash": file_sha256(dataset_path),
        "split_manifest": str(split_path),
        "split_manifest_hash": file_sha256(split_path),
        "model_architecture": architecture,
        "model_architecture_hash": architecture["hash"],
        "conditioning_schema": conditioning,
        "conditioning_schema_hash": conditioning["hash"],
        "loss_configuration": loss,
        "loss_configuration_hash": stable_hash(loss),
        "optimizer_configuration": optimizer,
        "optimizer_identity_hash": stable_hash(optimizer),
        "schedule_identity_hash": stable_hash(
            {"schedule": optimizer.get("schedule", "none"), "warmup_steps": optimizer.get("warmup_steps", 0)}
        ),
        "augmentation_configuration": deepcopy(config.get("augmentation", {})),
        "random_seeds": seeds,
        "seed_identity_hash": stable_hash(seeds),
        "software_version": software_version(repo),
        "precision_mode": str(runtime.get("precision", "fp32")),
        "precision_policy": str(runtime.get("precision", "fp32")),
        "hardware_summary": hardware_summary(),
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "checkpoint_lineage": lineage,
        "lineage_parent_identity": lineage[-1] if lineage else None,
        "sampling": deepcopy(config.get("sampling", {})),
        "ema": ema,
        "ema_identity_hash": stable_hash({"enabled": float(ema.get("decay", 0.0)) > 0.0, **ema}),
        "timestep_sampling": deepcopy(config.get("timestep_sampling", {"strategy": "uniform"})),
        "determinism_mode": str(runtime.get("determinism", "off")),
        "determinism_policy_hash": stable_hash(runtime.get("determinism", "off")),
        "sampler_policy_hash": stable_hash(sampling_policy),
        "global_batch_size": global_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "evaluation_cadence": runtime.get("sample_every", 0),
        "checkpoint_cadence": runtime.get("save_every", 0),
        "max_optimizer_steps": runtime.get("max_steps"),
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
) -> list[str]:
    for label, manifest in (("current", current), ("checkpoint", saved)):
        architecture = manifest.get("model_architecture")
        if isinstance(architecture, Mapping):
            claimed = architecture.get("hash")
            content = dict(architecture)
            content.pop("hash", None)
            if claimed != stable_hash(content) or manifest.get("model_architecture_hash") != claimed:
                raise IncompatibleResumeError(
                    f"{label} has tampered model_architecture_hash or architecture identity content"
                )
    mismatches = [field for field in RESUME_HARD_FIELDS if current.get(field) != saved.get(field)]
    if mismatches and not unsafe:
        detail = ", ".join(
            f"{field}: current={current.get(field)!r}, checkpoint={saved.get(field)!r}" for field in mismatches
        )
        raise IncompatibleResumeError(f"incompatible resume ({detail}); pass --unsafe-resume to override explicitly")
    if mismatches and unsafe_record is not None:
        if not str(unsafe_reason or "").strip():
            raise IncompatibleResumeError("unsafe resume requires an explicit recorded unsafe reason")
        unsafe_record.update(
            {
                "unsafe_resume": True,
                "unsafe_reason": str(unsafe_reason),
                "mismatches": list(mismatches),
                "exact_replay_claimed": False,
                "fair_architecture_comparison_eligible": False,
                "checkpoint_promotion_eligible": False,
            }
        )
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
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
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

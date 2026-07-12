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

EXPERIMENT_MANIFEST_VERSION = "spritelab_experiment_v1"
CONDITIONING_SCHEMA_VERSION_V1 = "spritelab_conditioning_v1"
CONDITIONING_SCHEMA_VERSION = "spritelab_conditioning_v2"
EVALUATION_SUITE_VERSION = "spritelab_eval_v1"
TOKENIZER_VERSION = "sprite_text_tokenizer_v1"
RESUME_HARD_FIELDS = (
    "dataset_manifest_hash",
    "split_manifest_hash",
    "model_architecture_hash",
    "conditioning_schema_hash",
)


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
    model_config: Mapping[str, Any], *, auxiliary_heads_instantiated: bool = True
) -> dict[str, Any]:
    identity = {
        "model_type": "generator_challenger",
        "architecture": "rectified_flow",
        "sprite_size": 32,
        "model_config": dict(model_config),
        "auxiliary_heads_instantiated": bool(auxiliary_heads_instantiated),
    }
    identity["hash"] = stable_hash(identity)
    return identity


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
    architecture = architecture_identity(model_config)
    conditioning = conditioning_schema(
        mode=str(config.get("conditioning", {}).get("mode", "caption_semantic")),
        tokenizer=tokenizer,
        structured_vocab=structured_vocab,
    )
    dataset_path = Path(dataset_manifest)
    split_path = Path(split_manifest or dataset_manifest)
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
        "loss_configuration": deepcopy(config.get("loss", {})),
        "optimizer_configuration": deepcopy(config.get("optimizer", {})),
        "augmentation_configuration": deepcopy(config.get("augmentation", {})),
        "random_seeds": deepcopy(config.get("seeds", {})),
        "software_version": software_version(repo),
        "precision_mode": str(config.get("runtime", {}).get("precision", "fp32")),
        "hardware_summary": hardware_summary(),
        "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        "checkpoint_lineage": list(checkpoint_lineage),
        "sampling": deepcopy(config.get("sampling", {})),
        "ema": deepcopy(config.get("ema", {})),
        "timestep_sampling": deepcopy(config.get("timestep_sampling", {"strategy": "uniform"})),
        "determinism_mode": str(config.get("runtime", {}).get("determinism", "off")),
        "timestep_validation_boundaries": deepcopy(
            config.get("runtime", {}).get("timestep_validation_boundaries", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ),
    }
    manifest["experiment_hash"] = stable_hash(manifest)
    return manifest


def validate_resume_compatibility(
    current: Mapping[str, Any], saved: Mapping[str, Any], *, unsafe: bool = False
) -> list[str]:
    mismatches = [field for field in RESUME_HARD_FIELDS if current.get(field) != saved.get(field)]
    if mismatches and not unsafe:
        detail = ", ".join(
            f"{field}: current={current.get(field)!r}, checkpoint={saved.get(field)!r}" for field in mismatches
        )
        raise IncompatibleResumeError(f"incompatible resume ({detail}); pass --unsafe-resume to override explicitly")
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

"""CUDA determinism policy and target-device qualification helpers."""

from __future__ import annotations

import os
import platform
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

DETERMINISM_MODES = ("off", "warn", "strict")
_CUBLAS_CONFIGS = {":4096:8", ":16:8"}


class DeterminismQualificationError(RuntimeError):
    """Raised when strict determinism cannot be guaranteed on this runtime."""


def configure_determinism(
    mode: str,
    *,
    device: Any,
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    import torch as real_torch

    th = real_torch if torch_module is None else torch_module
    normalized = str(mode).strip().lower()
    if normalized not in DETERMINISM_MODES:
        raise ValueError(f"determinism mode must be one of: {', '.join(DETERMINISM_MODES)}")
    is_cuda = str(getattr(device, "type", device)).startswith("cuda")
    issues: list[str] = []
    env = os.environ if environ is None else environ
    if is_cuda and env.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_CONFIGS:
        issues.append("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before Python starts")
    if normalized == "off":
        th.use_deterministic_algorithms(False)
        th.backends.cudnn.deterministic = False
    else:
        th.use_deterministic_algorithms(True, warn_only=normalized == "warn")
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
        if hasattr(th.backends, "cuda") and hasattr(th.backends.cuda, "matmul"):
            th.backends.cuda.matmul.allow_tf32 = False
        if hasattr(th.backends.cudnn, "allow_tf32"):
            th.backends.cudnn.allow_tf32 = False
    if issues and normalized == "strict":
        raise DeterminismQualificationError("strict CUDA determinism unavailable: " + "; ".join(issues))
    if issues and normalized == "warn":
        warnings.warn("CUDA determinism not guaranteed: " + "; ".join(issues), RuntimeWarning, stacklevel=2)
    return {
        "mode": normalized,
        "cuda_target": bool(is_cuda),
        "qualified": normalized != "off" and not issues,
        "issues": issues,
        "environment": runtime_environment(th, device=device),
        "guarantee_scope": "same GPU model, driver, CUDA, cuDNN, Torch, code, and inputs only",
        "cross_gpu_or_version_identity_claimed": False,
    }


def runtime_environment(torch_module: Any, *, device: Any) -> dict[str, Any]:
    th = torch_module
    is_cuda = str(getattr(device, "type", device)).startswith("cuda")
    cuda_version = getattr(getattr(th, "version", None), "cuda", None)
    cudnn_version = th.backends.cudnn.version() if hasattr(th.backends, "cudnn") else None
    devices: list[dict[str, Any]] = []
    if is_cuda and th.cuda.is_available():
        for index in range(th.cuda.device_count()):
            props = th.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": th.cuda.get_device_name(index),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    driver = None
    get_driver = getattr(getattr(th, "_C", None), "_cuda_getDriverVersion", None)
    if callable(get_driver):
        try:
            driver = int(get_driver())
        except RuntimeError:
            driver = None
    return {
        "platform": platform.platform(),
        "torch_version": str(th.__version__),
        "cuda_runtime_version": cuda_version,
        "cuda_driver_version": driver,
        "cudnn_version": cudnn_version,
        "gpus": devices,
    }


def assert_repeated_state_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    """Fail with a useful key when qualification artifacts differ."""
    import torch

    if set(first) != set(second):
        raise DeterminismQualificationError("qualification state keys differ")
    for key in first:
        left, right = first[key], second[key]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not equal:
            raise DeterminismQualificationError(f"bit-exact qualification mismatch at {key}")


def qualify_determinism(*, mode: str = "strict", device: str = "cuda", steps: int = 4) -> dict[str, Any]:
    """Run repeated forward/backward and a two-step interruption comparison."""
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise DeterminismQualificationError("CUDA qualification requested but torch.cuda.is_available() is false")
    policy = configure_determinism(mode, device=resolved, torch_module=torch)
    total_steps = max(2, int(steps))
    interrupt_at = total_steps // 2

    def new_run() -> tuple[Any, Any]:
        torch.manual_seed(20260711)
        if resolved.type == "cuda":
            torch.cuda.manual_seed_all(20260711)
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 8)).to(resolved)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train(model: Any, optimizer: Any, start: int, stop: int) -> None:
        for _step in range(start, stop):
            inputs = torch.randn(4, 16, device=resolved)
            target = torch.randn(4, 8, device=resolved)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs) - target).square().mean()
            loss.backward()
            optimizer.step()

    full_model, full_optimizer = new_run()
    train(full_model, full_optimizer, 0, total_steps)
    full = {"model": deepcopy(full_model.state_dict()), "optimizer": deepcopy(full_optimizer.state_dict())}

    repeat_model, repeat_optimizer = new_run()
    train(repeat_model, repeat_optimizer, 0, total_steps)
    repeated = {"model": repeat_model.state_dict(), "optimizer": repeat_optimizer.state_dict()}
    _assert_nested_equal(full, repeated, path="repeated")

    interrupted_model, interrupted_optimizer = new_run()
    train(interrupted_model, interrupted_optimizer, 0, interrupt_at)
    saved = {
        "model": deepcopy(interrupted_model.state_dict()),
        "optimizer": deepcopy(interrupted_optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if resolved.type == "cuda" else [],
    }
    resumed_model, resumed_optimizer = new_run()
    resumed_model.load_state_dict(saved["model"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    if resolved.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    train(resumed_model, resumed_optimizer, interrupt_at, total_steps)
    resumed = {"model": resumed_model.state_dict(), "optimizer": resumed_optimizer.state_dict()}
    _assert_nested_equal(full, resumed, path="resume")
    return {
        "qualified": True,
        "mode": policy["mode"],
        "device": str(resolved),
        "steps": total_steps,
        "interrupted_after": interrupt_at,
        "repeated_forward_backward_bit_exact": True,
        "resume_bit_exact": True,
        "environment": policy["environment"],
        "guarantee_scope": policy["guarantee_scope"],
        "cross_gpu_or_version_identity_claimed": False,
    }


def _assert_nested_equal(first: Any, second: Any, *, path: str) -> None:
    import torch

    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or not torch.equal(first, second):
            raise DeterminismQualificationError(f"bit-exact qualification mismatch at {path}")
        return
    if isinstance(first, Mapping):
        if not isinstance(second, Mapping) or set(first) != set(second):
            raise DeterminismQualificationError(f"qualification mapping mismatch at {path}")
        for key in first:
            _assert_nested_equal(first[key], second[key], path=f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)):
        if not isinstance(second, (list, tuple)) or len(first) != len(second):
            raise DeterminismQualificationError(f"qualification sequence mismatch at {path}")
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_nested_equal(left, right, path=f"{path}[{index}]")
        return
    if first != second:
        raise DeterminismQualificationError(f"qualification value mismatch at {path}")

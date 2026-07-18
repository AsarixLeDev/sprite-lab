from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from spritelab.training import smoke_bundle
from spritelab.training.smoke_bundle import SmokeBundleError


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _production_cuda_qualification() -> dict[str, Any]:
    return {
        "qualified": True,
        "mode": "strict",
        "device": "cuda",
        "steps": 2,
        "interrupted_after": 1,
        "repeated_forward_backward_bit_exact": True,
        "resume_bit_exact": True,
        "environment": {
            "platform": "Windows-11-test",
            "torch_version": "2.9.0+cu128",
            "cuda_runtime_version": "12.8",
            "cuda_driver_version": 12080,
            "cudnn_version": 91002,
            "gpus": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "compute_capability": "8.9",
                    "total_memory_bytes": 24 * 1024**3,
                }
            ],
        },
        "guarantee_scope": "same GPU model, driver, CUDA, cuDNN, Torch, code, and inputs only",
        "cross_gpu_or_version_identity_claimed": False,
    }


def _qualification_summary() -> dict[str, Any]:
    return {
        "qualified": True,
        "mode": "strict",
        "device": "cuda",
        "steps": 2,
        "repeated_forward_backward_bit_exact": True,
        "resume_bit_exact": True,
    }


def _verification_record() -> dict[str, Any]:
    inventory = {
        "checkpoint_step_000002.pt": {"sha256": _sha("live"), "byte_count": 101},
        "checkpoint_step_000002_ema.pt": {"sha256": _sha("ema"), "byte_count": 102},
        "cuda_determinism_qualification.json": {"sha256": _sha("qualification"), "byte_count": 103},
        "train_metrics.jsonl": {"sha256": _sha("metrics"), "byte_count": 104},
        "train_report.json": {"sha256": _sha("report"), "byte_count": 105},
    }
    return {
        "status": "COMPLETE",
        "steps_completed": 2,
        "device": "cuda",
        "determinism": "strict",
        "determinism_qualified": True,
        "report_sha256": inventory["train_report.json"]["sha256"],
        "metrics_sha256": inventory["train_metrics.jsonl"]["sha256"],
        "output_inventory": inventory,
        "output_inventory_sha256": smoke_bundle.stable_hash(inventory),
        "checkpoints": [
            {
                "weights": "live",
                "sha256": inventory["checkpoint_step_000002.pt"]["sha256"],
                "byte_count": inventory["checkpoint_step_000002.pt"]["byte_count"],
                "step": 2,
                "variant": "step",
            },
            {
                "weights": "ema",
                "sha256": inventory["checkpoint_step_000002_ema.pt"]["sha256"],
                "byte_count": inventory["checkpoint_step_000002_ema.pt"]["byte_count"],
                "step": 2,
                "variant": "step_ema",
            },
        ],
        "qualification": _qualification_summary(),
        **smoke_bundle.FALSE_ELIGIBILITY,
    }


def test_real_producer_qualification_is_strictly_validated_and_normalized() -> None:
    raw = _production_cuda_qualification()

    assert smoke_bundle._validate_cuda_qualification(raw, code="test") == raw
    assert smoke_bundle._cuda_qualification_summary(raw, code="test") == _qualification_summary()
    assert smoke_bundle._validate_verification_record(_verification_record(), "cuda")["qualification"] == (
        _qualification_summary()
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("interrupted_after",), 0),
        (("steps",), True),
        (("cross_gpu_or_version_identity_claimed",), True),
        (("guarantee_scope",), "all hardware and versions"),
        (("environment", "platform"), ""),
        (("environment", "cuda_driver_version"), True),
        (("environment", "gpus"), []),
        (("environment", "gpus", 0, "index"), 1),
        (("environment", "gpus", 0, "compute_capability"), "unknown"),
        (("environment", "gpus", 0, "total_memory_bytes"), 0),
    ],
)
def test_real_producer_qualification_rejects_mismatched_evidence(
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    raw: Any = _production_cuda_qualification()
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(SmokeBundleError):
        smoke_bundle._validate_cuda_qualification(raw, code="test")


@pytest.mark.parametrize("change", ["missing", "extra", "nested_missing", "nested_extra", "gpu_extra"])
def test_real_producer_qualification_rejects_inexact_schema(change: str) -> None:
    raw = _production_cuda_qualification()
    if change == "missing":
        raw.pop("environment")
    elif change == "extra":
        raw["unexpected"] = False
    elif change == "nested_missing":
        raw["environment"].pop("torch_version")
    elif change == "nested_extra":
        raw["environment"]["hostname"] = "private-host"
    else:
        raw["environment"]["gpus"][0]["uuid"] = "unexpected"

    with pytest.raises(SmokeBundleError):
        smoke_bundle._validate_cuda_qualification(raw, code="test")


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("missing", None),
        ("extra", None),
        ("steps", True),
        ("device", "cuda:0"),
        ("resume_bit_exact", False),
    ],
)
def test_receipt_rejects_inexact_or_mismatched_qualification_summary(change: str, value: Any) -> None:
    verification = _verification_record()
    summary = verification["qualification"]
    if change == "missing":
        summary.pop("steps")
    elif change == "extra":
        summary["environment"] = {}
    else:
        summary[change] = value

    with pytest.raises(SmokeBundleError, match=r"qualification summary|inexact schema"):
        smoke_bundle._validate_verification_record(verification, "cuda")


def test_receipt_rejects_missing_qualification_inventory_binding() -> None:
    verification = _verification_record()
    verification["output_inventory"].pop("cuda_determinism_qualification.json")
    verification["output_inventory_sha256"] = smoke_bundle.stable_hash(verification["output_inventory"])

    with pytest.raises(SmokeBundleError, match="evidence is missing"):
        smoke_bundle._validate_verification_record(verification, "cuda")


def test_qualification_reader_binds_exact_bytes_to_inventory(tmp_path: Path) -> None:
    path = tmp_path / "cuda_determinism_qualification.json"
    payload = smoke_bundle.canonical_json_bytes(_production_cuda_qualification(), pretty=True)
    path.write_bytes(payload)
    inventory_record = {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}

    assert (
        smoke_bundle._read_cuda_qualification_artifact(
            path,
            boundary=tmp_path,
            inventory_record=inventory_record,
        )
        == _production_cuda_qualification()
    )

    for mismatch in (
        {**inventory_record, "sha256": _sha("wrong")},
        {**inventory_record, "byte_count": len(payload) + 1},
    ):
        with pytest.raises(SmokeBundleError, match="do not match"):
            smoke_bundle._read_cuda_qualification_artifact(
                path,
                boundary=tmp_path,
                inventory_record=mismatch,
            )


def test_qualification_reader_rejects_duplicate_and_extra_evidence_with_matching_hash(tmp_path: Path) -> None:
    for name, payload in (
        (
            "duplicate.json",
            b'{"qualified":true,"qualified":true}',
        ),
        (
            "extra.json",
            smoke_bundle.canonical_json_bytes({**_production_cuda_qualification(), "unexpected": False}),
        ),
        (
            "noncanonical.json",
            smoke_bundle.canonical_json_bytes(_production_cuda_qualification()),
        ),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        inventory_record = {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
        with pytest.raises(SmokeBundleError):
            smoke_bundle._read_cuda_qualification_artifact(
                path,
                boundary=tmp_path,
                inventory_record=inventory_record,
            )

"""Explicit, non-benchmark exploratory generation playground."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from spritelab.product_core import (
    ProductEvent,
    ProductRun,
    ProductStatus,
    strict_json_bytes,
    strict_json_dumps,
    strict_json_loads,
)
from spritelab.product_features.evaluation.exploratory_smoke import ExploratoryCheckpointCatalog
from spritelab.product_features.evaluation.models import CheckpointCatalog
from spritelab.product_web.events import EventRepository, event_history_transaction_lock
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    ExactPublicationUnsupported,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
)

PLAYGROUND_SCHEMA = "spritelab.product.playground-generation.v1"
PLAYGROUND_RUN_SCHEMA = "spritelab.product.playground-run.v1"
PLAYGROUND_COMMAND_SCHEMA = "spritelab.product.playground-command.v1"
EXPLORATORY_SCOPE = "EXPLORATORY"
DEFAULT_SEED = 42
DEFAULT_SAMPLING_STEPS = 30
DEFAULT_GUIDANCE = 3.0
DEFAULT_IMAGE_COUNT = 4
PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS = 5 * 60
PLAYGROUND_RUNTIME_IDENTITY_SCHEMA = "spritelab.playground-runtime-identity.v2"
PLAYGROUND_TERMINAL_SCHEMA = "spritelab.product.playground-terminal.v1"
PLAYGROUND_LIFECYCLE_COMMIT_SCHEMA = "spritelab.product.playground-lifecycle-commit.v1"
PLAYGROUND_TRANSITION_SCHEMA = "spritelab.product.playground-transition.v1"
CONTAINED_RUNTIME_BYTE_POLICY = "trusted-installed-runtime-source-exact-native-resource-drift-detected-v1"
CONTAINED_RUNTIME_RESIDUALS = [
    "installed-runtime-bootstrap-is-a-trusted-baseline-before-loader-policy",
    "dependent-native-libraries-are-pre-post-hashed-but-not-fd-pinned",
    "runtime-resource-opens-are-prechecked-and-posthashed-but-not-fd-pinned",
]
_RUNTIME_IDENTITY_KEYS = {
    "schema_version",
    "runtime_reported",
    "python_version",
    "python_implementation",
    "torch_version",
    "torch_cuda_version",
    "cuda_available",
    "selected_device",
    "platform",
    "runtime_closure_identity",
    "execution_byte_policy",
    "bounded_residuals",
    "paths_exposed",
}
_COMMAND_KEYS = {
    "schema_version",
    "action",
    "run_id",
    "request",
    "checkpoint_identity",
    "generation_adapter_identity",
    "scope",
    "benchmark_eligible",
    "promotion_evidence_eligible",
}
_REQUEST_KEYS = {
    "prompt",
    "checkpoint_id",
    "weights",
    "seed",
    "sampling_steps",
    "guidance",
    "image_count",
}
_CHECKPOINT_IDENTITY_KEYS = {
    "checkpoint_id",
    "checkpoint_run_id",
    "checkpoint_step",
    "weights",
    "sha256",
    "classification",
    "purpose",
    "registration_identity",
    "evidence_identity",
    "dataset_freeze_identity",
    "campaign_identity",
    "training_code_identity",
    "production_eligible",
    "evaluation_eligible",
    "training_resume_eligible",
    "promotion_eligible",
}
_ADAPTER_IDENTITY_REQUIRED_KEYS = {"adapter", "remote", "billable", "sha256"}
_REPORT_KEYS = {
    "schema_version",
    "run_id",
    "generation_id",
    "product_run",
    "scope",
    "request",
    "checkpoint_identity",
    "generation_adapter_identity",
    "runtime_identity",
    "started_at",
    "ended_at",
    "results",
    "artifact_identities",
    "excluded_from_frozen_benchmark",
    "excluded_from_promotion_evidence",
}
_PRODUCT_RUN_KEYS = {
    "run_id",
    "feature",
    "action_id",
    "status",
    "backend_id",
    "started_at",
    "ended_at",
    "artifact_references",
}
_RESULT_KEYS = {
    "result_id",
    "checkpoint_identity",
    "checkpoint_run_id",
    "checkpoint_step",
    "weights",
    "prompt",
    "seed",
    "generation_parameters",
    "timestamp",
    "output_hash",
    "application_version",
    "media_type",
    "output_reference",
    "scope",
    "frozen_benchmark_eligible",
    "promotion_evidence_eligible",
}
_ARTIFACT_IDENTITY_KEYS = {"artifact_id", "reference", "sha256", "size_bytes", "media_type"}
_CACHE_KEYS = {
    "request",
    "checkpoint_identity",
    "generation_adapter_identity",
    "command_sha256",
    "prompt",
    "weights",
    "seed",
    "generation_parameters",
    "results",
    "artifact_identities",
    "output_hashes",
    "progress",
    "runtime_identity",
    "report_identity",
    "failure",
    "cancellation",
    "timeout",
    "terminal_status",
    "started_at",
    "ended_at",
    "deadline_at",
}
_LIFECYCLE_COMMIT_KEYS = {
    "schema_version",
    "sequence",
    "prior_event_stream_identity",
    "event_identity",
    "event",
    "cache_identity",
}
_TRANSITION_KEYS = {
    "schema_version",
    "sequence",
    "prior_event_stream_identity",
    "prior_state_identity",
    "prior_cache_identity",
    "prior_commit_identity",
}
_TERMINAL_RECORD_KEYS = {
    "schema_version",
    "terminal_status",
    "command_sha256",
    "request",
    "checkpoint_identity",
    "generation_adapter_identity",
    "cache",
    "cache_identity",
}
_TERMINAL_EVENT_TYPES = {"generation_completed", "failed", "cancelled", "timed_out"}
_TERMINAL_STATUS_BY_EVENT = {
    "generation_completed": "COMPLETE",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
    "timed_out": "TIMED_OUT",
}
_PRODUCT_STATUS_BY_TERMINAL = {
    "COMPLETE": ProductStatus.COMPLETE,
    "FAILED": ProductStatus.FAILED,
    "CANCELLED": ProductStatus.FAILED,
    "TIMED_OUT": ProductStatus.FAILED,
}


class GenerationSafetyError(ValueError):
    """An explicit generation or billing safety precondition was not met."""


class GeneratorUnavailableError(RuntimeError):
    """No typed generator was supplied to the playground."""


class GenerationCancelledError(RuntimeError):
    """A typed generator reported explicit cancellation."""


class GenerationTimedOutError(RuntimeError):
    """A contained generator reached its durable wall-clock deadline."""


def _pathless_runtime_text(value: Any, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        raise GenerationSafetyError("The generation runtime identity is malformed.")
    if "\\" in value or value.startswith("/") or (len(value) >= 3 and value[1] == ":" and value[2] in "/\\"):
        raise GenerationSafetyError("The generation runtime identity exposed a private path.")
    if any(ord(character) < 32 for character in value):
        raise GenerationSafetyError("The generation runtime identity is malformed.")
    return value


def validate_runtime_identity(value: Any) -> dict[str, Any]:
    """Return one exact, pathless runtime report or fail closed."""

    if not isinstance(value, Mapping) or set(value) != _RUNTIME_IDENTITY_KEYS:
        raise GenerationSafetyError("The generation runtime identity is missing or malformed.")
    result = dict(value)
    if result.get("schema_version") != PLAYGROUND_RUNTIME_IDENTITY_SCHEMA or result.get("paths_exposed") is not False:
        raise GenerationSafetyError("The generation runtime identity schema is unsupported.")
    reported = result.get("runtime_reported")
    if type(reported) is not bool:
        raise GenerationSafetyError("The generation runtime identity boolean fields are malformed.")
    cuda_available = result.get("cuda_available")
    if cuda_available is not None and type(cuda_available) is not bool:
        raise GenerationSafetyError("The generation runtime identity boolean fields are malformed.")
    for key in (
        "python_version",
        "python_implementation",
        "torch_version",
        "torch_cuda_version",
        "selected_device",
        "platform",
    ):
        result[key] = _pathless_runtime_text(result.get(key), nullable=True)
    closure_identity = result.get("runtime_closure_identity")
    if closure_identity is not None and (
        not isinstance(closure_identity, str)
        or len(closure_identity) != 64
        or any(character not in "0123456789abcdef" for character in closure_identity)
    ):
        raise GenerationSafetyError("The generation runtime closure identity is malformed.")
    policy = _pathless_runtime_text(result.get("execution_byte_policy"))
    residuals = result.get("bounded_residuals")
    if not isinstance(residuals, list) or any(not isinstance(item, str) for item in residuals):
        raise GenerationSafetyError("The generation runtime residuals are malformed.")
    for item in residuals:
        _pathless_runtime_text(item)
    required = ("python_version", "python_implementation", "torch_version", "selected_device", "platform")
    if reported:
        if any(result.get(key) is None for key in required):
            raise GenerationSafetyError("The reported generation runtime identity is incomplete.")
        if policy == CONTAINED_RUNTIME_BYTE_POLICY:
            if closure_identity is None or residuals != CONTAINED_RUNTIME_RESIDUALS:
                raise GenerationSafetyError("The contained runtime residual contract is malformed.")
        elif policy == "in-process-generator-runtime-uncontained-v1":
            if closure_identity is not None or residuals != [
                "generator-executed-in-server-process-without-write-confinement"
            ]:
                raise GenerationSafetyError("The in-process runtime residual contract is malformed.")
        else:
            raise GenerationSafetyError("The generation runtime byte policy is unsupported.")
    elif (
        policy != "runtime-identity-not-reported-v1"
        or closure_identity is not None
        or cuda_available is not None
        or any(result.get(key) is not None for key in (*required, "torch_cuda_version"))
        or residuals != ["generator-did-not-report-runtime-identity"]
    ):
        raise GenerationSafetyError("The unreported runtime identity contract is malformed.")
    return result


def _unreported_runtime_identity() -> dict[str, Any]:
    return {
        "schema_version": PLAYGROUND_RUNTIME_IDENTITY_SCHEMA,
        "runtime_reported": False,
        "python_version": None,
        "python_implementation": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "selected_device": None,
        "platform": None,
        "runtime_closure_identity": None,
        "execution_byte_policy": "runtime-identity-not-reported-v1",
        "bounded_residuals": ["generator-did-not-report-runtime-identity"],
        "paths_exposed": False,
    }


@dataclass(frozen=True)
class GeneratedAsset:
    content: bytes
    media_type: str = "image/png"


class PlaygroundGenerator(Protocol):
    remote: bool
    billable: bool

    def generate(
        self,
        *,
        checkpoint: Path,
        prompt: str,
        seed: int,
        sampling_steps: int,
        guidance: float,
        image_count: int,
        weights: str,
        expected_sha256: str,
        expected_step: int,
        expected_variant: str,
    ) -> Sequence[GeneratedAsset | bytes | Mapping[str, Any] | Path]: ...


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    checkpoint_id: str
    weights: str = "ema"
    seed: int = DEFAULT_SEED
    sampling_steps: int = DEFAULT_SAMPLING_STEPS
    guidance: float = DEFAULT_GUIDANCE
    image_count: int = DEFAULT_IMAGE_COUNT

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise ValueError("Prompt must be text.")
        normalized_prompt = self.prompt.strip()
        if not normalized_prompt or len(normalized_prompt) > 2_000:
            raise ValueError("Prompt must contain between 1 and 2,000 characters.")
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized_prompt):
            raise ValueError("Prompt contains unsupported control characters.")
        object.__setattr__(self, "prompt", normalized_prompt)
        if self.weights not in {"live", "ema"}:
            raise ValueError("weights must be 'live' or 'ema'.")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1.")
        if type(self.sampling_steps) is not int or not 1 <= self.sampling_steps <= 500:
            raise ValueError("sampling_steps must be between 1 and 500.")
        if isinstance(self.guidance, bool) or not isinstance(self.guidance, (int, float)):
            raise ValueError("guidance must be a finite number greater than 0 and at most 50.")
        if not math.isfinite(float(self.guidance)) or not 0.0 < self.guidance <= 50.0:
            raise ValueError("guidance must be greater than 0 and at most 50.")
        if type(self.image_count) is not int or not 1 <= self.image_count <= 16:
            raise ValueError("image_count must be between 1 and 16.")

    @classmethod
    def defaults(cls, checkpoint_id: str = "") -> dict[str, Any]:
        return asdict(cls(prompt="Describe a 32x32 sprite", checkpoint_id=checkpoint_id))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_copy(value: Any) -> Any:
    return strict_json_loads(strict_json_bytes(value, sort_keys=True, separators=(",", ":")))


def _identity(value: Any) -> str:
    return hashlib.sha256(strict_json_bytes(value, sort_keys=True, separators=(",", ":"))).hexdigest()


def _event_identity(event: ProductEvent | Mapping[str, Any]) -> str:
    value = event.to_dict() if isinstance(event, ProductEvent) else dict(event)
    return _identity(value)


def _event_stream_identity(events: Sequence[ProductEvent]) -> str:
    return _identity([event.to_dict() for event in events])


def _canonical_event_bytes(events: Sequence[ProductEvent]) -> bytes:
    return b"".join(
        strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for event in events
    )


def _aware_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise GenerationSafetyError("Playground timestamp metadata is malformed.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GenerationSafetyError("Playground timestamp metadata is malformed.") from exc
    if parsed.tzinfo is None:
        raise GenerationSafetyError("Playground timestamp metadata is malformed.")
    return value


def _validate_request_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_KEYS:
        raise GenerationSafetyError("The Playground request schema is malformed.")
    request = dict(value)
    checkpoint_id = request.get("checkpoint_id")
    if (
        not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or len(checkpoint_id) > 240
        or any(ord(character) < 32 for character in checkpoint_id)
    ):
        raise GenerationSafetyError("The Playground checkpoint request identity is malformed.")
    try:
        validated = GenerationRequest(**request)
    except (TypeError, ValueError) as exc:
        raise GenerationSafetyError("The Playground request schema is malformed.") from exc
    if asdict(validated) != request:
        raise GenerationSafetyError("The Playground request is not canonical.")
    return request


def _validate_checkpoint_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_IDENTITY_KEYS:
        raise GenerationSafetyError("The Playground checkpoint identity schema is malformed.")
    identity = dict(value)
    for key in ("checkpoint_id", "checkpoint_run_id", "weights", "classification", "purpose"):
        if not isinstance(identity.get(key), str) or not identity[key] or len(identity[key]) > 240:
            raise GenerationSafetyError("The Playground checkpoint identity schema is malformed.")
    if identity["weights"] not in {"live", "ema"} or type(identity.get("checkpoint_step")) is not int:
        raise GenerationSafetyError("The Playground checkpoint identity schema is malformed.")
    if identity["checkpoint_step"] < 0 or not _is_sha256(identity.get("sha256")):
        raise GenerationSafetyError("The Playground checkpoint identity schema is malformed.")
    for key in (
        "registration_identity",
        "evidence_identity",
        "dataset_freeze_identity",
        "campaign_identity",
        "training_code_identity",
    ):
        if identity.get(key) is not None and not isinstance(identity[key], str):
            raise GenerationSafetyError("The Playground checkpoint identity schema is malformed.")
    for key in (
        "production_eligible",
        "evaluation_eligible",
        "training_resume_eligible",
        "promotion_eligible",
    ):
        if type(identity.get(key)) is not bool:
            raise GenerationSafetyError("The Playground checkpoint identity booleans are malformed.")
    return identity


def _validate_adapter_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationSafetyError("The Playground adapter identity schema is malformed.")
    identity = dict(value)
    keys = set(identity)
    if keys != _ADAPTER_IDENTITY_REQUIRED_KEYS and keys != _ADAPTER_IDENTITY_REQUIRED_KEYS | {"code_identity_sha256"}:
        raise GenerationSafetyError("The Playground adapter identity schema is malformed.")
    if not isinstance(identity.get("adapter"), str) or not identity["adapter"] or len(identity["adapter"]) > 300:
        raise GenerationSafetyError("The Playground adapter identity schema is malformed.")
    if type(identity.get("remote")) is not bool or type(identity.get("billable")) is not bool:
        raise GenerationSafetyError("The Playground adapter identity booleans are malformed.")
    if "code_identity_sha256" in identity and not _is_sha256(identity["code_identity_sha256"]):
        raise GenerationSafetyError("The Playground adapter code identity is malformed.")
    expected_sha256 = _identity({key: item for key, item in identity.items() if key != "sha256"})
    if identity.get("sha256") != expected_sha256:
        raise GenerationSafetyError("The Playground adapter identity does not match its fields.")
    return identity


def _validate_command(value: Any, *, run_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMMAND_KEYS:
        raise GenerationSafetyError("The Playground command schema is malformed.")
    command = dict(value)
    if (
        command.get("schema_version") != PLAYGROUND_COMMAND_SCHEMA
        or command.get("action") != "playground.generate"
        or command.get("scope") != EXPLORATORY_SCOPE
        or type(command.get("benchmark_eligible")) is not bool
        or command["benchmark_eligible"] is not False
        or type(command.get("promotion_evidence_eligible")) is not bool
        or command["promotion_evidence_eligible"] is not False
        or not isinstance(command.get("run_id"), str)
        or (run_id is not None and command["run_id"] != run_id)
    ):
        raise GenerationSafetyError("The Playground command schema is malformed.")
    command["request"] = _validate_request_payload(command.get("request"))
    command["checkpoint_identity"] = _validate_checkpoint_identity(command.get("checkpoint_identity"))
    command["generation_adapter_identity"] = _validate_adapter_identity(command.get("generation_adapter_identity"))
    return command


def _validate_report_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"reference", "sha256", "size_bytes"}:
        raise GenerationSafetyError("The Playground report identity schema is malformed.")
    identity = dict(value)
    if (
        identity.get("reference") != "report/report.json"
        or not _is_sha256(identity.get("sha256"))
        or type(identity.get("size_bytes")) is not int
        or identity["size_bytes"] <= 0
    ):
        raise GenerationSafetyError("The Playground report identity schema is malformed.")
    return identity


def _cache_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not _CACHE_KEYS.issubset(state):
        raise GenerationSafetyError("The Playground state cache schema is incomplete.")
    cache = {key: _json_copy(state.get(key)) for key in sorted(_CACHE_KEYS)}
    cache["request"] = _validate_request_payload(cache["request"])
    cache["checkpoint_identity"] = _validate_checkpoint_identity(cache["checkpoint_identity"])
    cache["generation_adapter_identity"] = _validate_adapter_identity(cache["generation_adapter_identity"])
    if not _is_sha256(cache.get("command_sha256")):
        raise GenerationSafetyError("The Playground state command identity is malformed.")
    if cache.get("prompt") != cache["request"]["prompt"] or cache.get("weights") != cache["request"]["weights"]:
        raise GenerationSafetyError("The Playground state request bindings are inconsistent.")
    if cache.get("seed") != cache["request"]["seed"]:
        raise GenerationSafetyError("The Playground state request bindings are inconsistent.")
    expected_parameters = {
        "sampling_steps": cache["request"]["sampling_steps"],
        "guidance": cache["request"]["guidance"],
        "image_count": cache["request"]["image_count"],
    }
    if cache.get("generation_parameters") != expected_parameters:
        raise GenerationSafetyError("The Playground state generation parameters are inconsistent.")
    if not isinstance(cache.get("results"), list) or not isinstance(cache.get("artifact_identities"), list):
        raise GenerationSafetyError("The Playground state result cache is malformed.")
    if not isinstance(cache.get("output_hashes"), list) or not isinstance(cache.get("progress"), Mapping):
        raise GenerationSafetyError("The Playground state result cache is malformed.")
    results = [
        _validate_result(
            item,
            index=index,
            request=cache["request"],
            checkpoint_identity=cache["checkpoint_identity"],
            generation_parameters=expected_parameters,
        )
        for index, item in enumerate(cache["results"])
    ]
    if len(cache["artifact_identities"]) != len(results):
        raise GenerationSafetyError("The Playground state result and artifact caches disagree.")
    artifacts = [
        _validate_artifact_identity(item, result=results[index])
        for index, item in enumerate(cache["artifact_identities"])
    ]
    if len(artifacts) != len(results) or cache["output_hashes"] != [item["sha256"] for item in artifacts]:
        raise GenerationSafetyError("The Playground state result and artifact caches disagree.")
    progress = dict(cache["progress"])
    if (
        set(progress) != {"current", "total"}
        or type(progress.get("current")) is not int
        or type(progress.get("total")) is not int
        or progress != {"current": len(results), "total": cache["request"]["image_count"]}
    ):
        raise GenerationSafetyError("The Playground state progress cache is malformed.")
    cache["results"] = results
    cache["artifact_identities"] = artifacts
    cache["progress"] = progress
    if cache.get("runtime_identity") is not None:
        cache["runtime_identity"] = validate_runtime_identity(cache["runtime_identity"])
    if cache.get("report_identity") is not None:
        cache["report_identity"] = _validate_report_identity(cache["report_identity"])
    if cache.get("terminal_status") not in {None, "COMPLETE", "FAILED", "CANCELLED", "TIMED_OUT"}:
        raise GenerationSafetyError("The Playground state terminal status is malformed.")
    _aware_timestamp(cache.get("started_at"))
    _aware_timestamp(cache.get("deadline_at"))
    if cache.get("ended_at") is not None:
        _aware_timestamp(cache["ended_at"])
    return cache


def _lifecycle_commit_for(events: Sequence[ProductEvent], cache: Mapping[str, Any]) -> dict[str, Any]:
    if not events:
        raise GenerationSafetyError("A Playground lifecycle commit requires an event.")
    event = events[-1]
    return {
        "schema_version": PLAYGROUND_LIFECYCLE_COMMIT_SCHEMA,
        "sequence": len(events),
        "prior_event_stream_identity": _event_stream_identity(events[:-1]),
        "event_identity": _event_identity(event),
        "event": event.to_dict(),
        "cache_identity": _identity(cache),
    }


def _validate_lifecycle_commit(
    value: Any,
    *,
    events: Sequence[ProductEvent],
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LIFECYCLE_COMMIT_KEYS:
        raise GenerationSafetyError("The Playground lifecycle commit schema is malformed.")
    commit = dict(value)
    expected = _lifecycle_commit_for(events, cache)
    if commit != expected:
        raise GenerationSafetyError("The Playground lifecycle commit does not match authoritative evidence.")
    return commit


def _transition_for(
    *,
    sequence: int,
    prior_events: Sequence[ProductEvent],
    prior_state: Mapping[str, Any],
    prior_cache: Mapping[str, Any],
) -> dict[str, Any]:
    prior_commit = _validate_lifecycle_commit(
        prior_state.get("playground_lifecycle_commit"),
        events=prior_events,
        cache=prior_cache,
    )
    return {
        "schema_version": PLAYGROUND_TRANSITION_SCHEMA,
        "sequence": sequence,
        "prior_event_stream_identity": _event_stream_identity(prior_events),
        "prior_state_identity": _identity(prior_state),
        "prior_cache_identity": _identity(prior_cache),
        "prior_commit_identity": _identity(prior_commit),
    }


def _validate_transition(
    value: Any,
    *,
    sequence: int,
    prior_events: Sequence[ProductEvent],
    prior_cache: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRANSITION_KEYS:
        raise GenerationSafetyError("A Playground lifecycle transition schema is malformed.")
    transition = dict(value)
    prior_commit = _lifecycle_commit_for(prior_events, prior_cache)
    if (
        transition.get("schema_version") != PLAYGROUND_TRANSITION_SCHEMA
        or transition.get("sequence") != sequence
        or transition.get("prior_event_stream_identity") != _event_stream_identity(prior_events)
        or transition.get("prior_cache_identity") != _identity(prior_cache)
        or transition.get("prior_commit_identity") != _identity(prior_commit)
        or not _is_sha256(transition.get("prior_state_identity"))
    ):
        raise GenerationSafetyError("A Playground lifecycle transition binding is malformed.")
    return transition


def _playground_lifecycle_checkpoint(stage: str, directory: Path) -> None:
    """Test hook for exact cross-file lifecycle publication boundaries."""


def _application_version() -> str:
    try:
        return version("sprite-lab")
    except PackageNotFoundError:
        return "0.1.0"


def _asset_bytes(value: GeneratedAsset | bytes | Mapping[str, Any] | Path) -> tuple[bytes, str]:
    if isinstance(value, GeneratedAsset):
        return value.content, value.media_type
    if isinstance(value, bytes):
        return value, "image/png"
    if isinstance(value, Path):
        return value.read_bytes(), "image/png"
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, bytes):
            return content, str(value.get("media_type") or "image/png")
    raise TypeError("Generator outputs must be bytes, paths, mappings with bytes, or GeneratedAsset values.")


def _validate_result(
    value: Any,
    *,
    index: int,
    request: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    generation_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESULT_KEYS:
        raise GenerationSafetyError("A Playground result schema is malformed.")
    result = dict(value)
    expected_id_suffix = f"-{index:03d}"
    if not isinstance(result.get("result_id"), str) or not result["result_id"].endswith(expected_id_suffix):
        raise GenerationSafetyError("A Playground result identity is malformed.")
    if (
        result.get("checkpoint_identity") != checkpoint_identity.get("checkpoint_id")
        or result.get("checkpoint_run_id") != checkpoint_identity.get("checkpoint_run_id")
        or result.get("checkpoint_step") != checkpoint_identity.get("checkpoint_step")
        or result.get("weights") != request.get("weights")
        or result.get("prompt") != request.get("prompt")
        or result.get("seed") != request.get("seed", 0) + index
        or result.get("generation_parameters") != dict(generation_parameters)
    ):
        raise GenerationSafetyError("A Playground result does not match its generation request.")
    _aware_timestamp(result.get("timestamp"))
    if not _is_sha256(result.get("output_hash")):
        raise GenerationSafetyError("A Playground result output identity is malformed.")
    if not isinstance(result.get("application_version"), str) or not result["application_version"]:
        raise GenerationSafetyError("A Playground result application identity is malformed.")
    media_type = result.get("media_type")
    if not isinstance(media_type, str) or not media_type or len(media_type) > 100:
        raise GenerationSafetyError("A Playground result media type is malformed.")
    expected_reference = f"artifacts/image_{index:03d}{'.png' if media_type == 'image/png' else '.bin'}"
    if result.get("output_reference") != expected_reference:
        raise GenerationSafetyError("A Playground result reference is malformed.")
    if (
        result.get("scope") != EXPLORATORY_SCOPE
        or result.get("frozen_benchmark_eligible") is not False
        or result.get("promotion_evidence_eligible") is not False
    ):
        raise GenerationSafetyError("A Playground result eligibility contract is malformed.")
    return result


def _validate_artifact_identity(
    value: Any, *, result: Mapping[str, Any], content_size: int | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_IDENTITY_KEYS:
        raise GenerationSafetyError("A Playground artifact identity schema is malformed.")
    identity = dict(value)
    if (
        identity.get("artifact_id") != result.get("result_id")
        or identity.get("reference") != result.get("output_reference")
        or identity.get("sha256") != result.get("output_hash")
        or identity.get("media_type") != result.get("media_type")
        or type(identity.get("size_bytes")) is not int
        or identity["size_bytes"] < 0
        or (content_size is not None and identity["size_bytes"] != content_size)
    ):
        raise GenerationSafetyError("A Playground artifact identity does not match its result.")
    return identity


def _validate_terminal_record(value: Any, *, base_cache: Mapping[str, Any], event: ProductEvent) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TERMINAL_RECORD_KEYS:
        raise GenerationSafetyError("The Playground terminal record schema is malformed.")
    terminal = dict(value)
    expected_status = _TERMINAL_STATUS_BY_EVENT.get(event.event_type)
    if (
        terminal.get("schema_version") != PLAYGROUND_TERMINAL_SCHEMA
        or terminal.get("terminal_status") != expected_status
    ):
        raise GenerationSafetyError("The Playground terminal record status is malformed.")
    for key in ("command_sha256", "request", "checkpoint_identity", "generation_adapter_identity"):
        if terminal.get(key) != base_cache.get(key):
            raise GenerationSafetyError("The Playground terminal record changed an immutable run binding.")
    cache_value = terminal.get("cache")
    if not isinstance(cache_value, Mapping) or set(cache_value) != _CACHE_KEYS:
        raise GenerationSafetyError("The Playground terminal cache is malformed.")
    cache = _cache_from_state(cache_value)
    if terminal.get("cache_identity") != _identity(cache):
        raise GenerationSafetyError("The Playground terminal cache identity is malformed.")
    for key in (
        "request",
        "checkpoint_identity",
        "generation_adapter_identity",
        "command_sha256",
        "prompt",
        "weights",
        "seed",
        "generation_parameters",
        "results",
        "artifact_identities",
        "output_hashes",
        "progress",
        "started_at",
        "deadline_at",
    ):
        if cache.get(key) != base_cache.get(key):
            raise GenerationSafetyError("The Playground terminal cache changed pre-terminal evidence.")
    if cache.get("terminal_status") != expected_status or cache.get("ended_at") != event.timestamp:
        raise GenerationSafetyError("The Playground terminal cache status is inconsistent.")
    if expected_status == "COMPLETE":
        if (
            cache.get("failure") is not None
            or cache.get("cancellation") is not None
            or cache.get("timeout") is not None
            or cache.get("report_identity") is None
        ):
            raise GenerationSafetyError("The completed Playground terminal cache is malformed.")
        _validate_report_identity(cache["report_identity"])
        validate_runtime_identity(cache.get("runtime_identity"))
    else:
        if cache.get("report_identity") is not None or cache.get("runtime_identity") is not None:
            raise GenerationSafetyError("A non-complete Playground terminal carries completed evidence.")
        if expected_status == "FAILED":
            failure = cache.get("failure")
            if not isinstance(failure, Mapping) or set(failure) != {"type", "message", "timestamp"}:
                raise GenerationSafetyError("The Playground failure terminal is malformed.")
            if not isinstance(failure.get("type"), str) or failure.get("message") != "Generation adapter failed.":
                raise GenerationSafetyError("The Playground failure terminal is malformed.")
            if _aware_timestamp(failure.get("timestamp")) != event.timestamp:
                raise GenerationSafetyError("The Playground failure terminal timestamp is malformed.")
            if cache.get("cancellation") is not None or cache.get("timeout") is not None:
                raise GenerationSafetyError("The Playground failure terminal is malformed.")
        elif expected_status == "CANCELLED":
            cancellation = cache.get("cancellation")
            if not isinstance(cancellation, Mapping) or set(cancellation) != {"reason", "timestamp"}:
                raise GenerationSafetyError("The Playground cancellation terminal is malformed.")
            if (
                not isinstance(cancellation.get("reason"), str)
                or not cancellation["reason"]
                or len(cancellation["reason"]) > 2_000
            ):
                raise GenerationSafetyError("The Playground cancellation terminal is malformed.")
            if _aware_timestamp(cancellation.get("timestamp")) != event.timestamp:
                raise GenerationSafetyError("The Playground cancellation terminal timestamp is malformed.")
            if cache.get("failure") is not None or cache.get("timeout") is not None:
                raise GenerationSafetyError("The Playground cancellation terminal is malformed.")
        elif expected_status == "TIMED_OUT":
            timeout = cache.get("timeout")
            if not isinstance(timeout, Mapping) or set(timeout) != {"reason", "timestamp", "deadline_at"}:
                raise GenerationSafetyError("The Playground timeout terminal is malformed.")
            if not isinstance(timeout.get("reason"), str) or not timeout["reason"] or len(timeout["reason"]) > 2_000:
                raise GenerationSafetyError("The Playground timeout terminal is malformed.")
            if _aware_timestamp(timeout.get("timestamp")) != event.timestamp:
                raise GenerationSafetyError("The Playground timeout terminal timestamp is malformed.")
            if timeout.get("deadline_at") != cache.get("deadline_at"):
                raise GenerationSafetyError("The Playground timeout deadline binding is malformed.")
            if cache.get("failure") is not None or cache.get("cancellation") is not None:
                raise GenerationSafetyError("The Playground timeout terminal is malformed.")
    return terminal


@dataclass(frozen=True)
class _LifecycleReduction:
    cache: dict[str, Any]
    status: str
    stage: str
    event_count: int
    terminal_event_type: str | None
    stages: tuple[str, ...]


def _reduce_lifecycle_events(events: Sequence[ProductEvent]) -> _LifecycleReduction:
    if not events or events[0].event_type != "planned":
        raise GenerationSafetyError("The authoritative planned Playground event is missing.")
    fingerprints: set[str] = set()
    for event in events:
        fingerprint = _event_identity(event)
        if fingerprint in fingerprints:
            raise GenerationSafetyError("The Playground event stream contains a duplicate event.")
        fingerprints.add(fingerprint)
        if event.feature != "playground":
            raise GenerationSafetyError("The Playground event stream changed feature identity.")
        _aware_timestamp(event.timestamp)

    planned = events[0]
    planned_metric_keys = {
        "command_sha256",
        "prompt",
        "checkpoint_identity",
        "request",
        "generation_adapter_identity",
        "deadline_at",
        "exploratory",
        "benchmark_eligible",
        "promotion_evidence_eligible",
    }
    if (
        planned.stage != "planned"
        or planned.status is not ProductStatus.RUNNING
        or planned.current != 0
        or set(planned.metrics) != planned_metric_keys
        or planned.metrics.get("exploratory") is not True
        or planned.metrics.get("benchmark_eligible") is not False
        or planned.metrics.get("promotion_evidence_eligible") is not False
        or planned.message != "Exploratory Playground generation planned."
        or planned.artifact_references
    ):
        raise GenerationSafetyError("The authoritative planned Playground event is malformed.")
    request = _validate_request_payload(planned.metrics.get("request"))
    checkpoint_identity = _validate_checkpoint_identity(planned.metrics.get("checkpoint_identity"))
    adapter_identity = _validate_adapter_identity(planned.metrics.get("generation_adapter_identity"))
    if planned.metrics.get("prompt") != request["prompt"] or not _is_sha256(planned.metrics.get("command_sha256")):
        raise GenerationSafetyError("The authoritative planned Playground event bindings are malformed.")
    deadline_at = _aware_timestamp(planned.metrics.get("deadline_at"))
    if planned.total != request["image_count"]:
        raise GenerationSafetyError("The authoritative planned Playground event count is malformed.")
    generation_parameters = {
        "sampling_steps": request["sampling_steps"],
        "guidance": request["guidance"],
        "image_count": request["image_count"],
    }
    cache: dict[str, Any] = {
        "request": request,
        "checkpoint_identity": checkpoint_identity,
        "generation_adapter_identity": adapter_identity,
        "command_sha256": planned.metrics["command_sha256"],
        "prompt": request["prompt"],
        "weights": request["weights"],
        "seed": request["seed"],
        "generation_parameters": generation_parameters,
        "results": [],
        "artifact_identities": [],
        "output_hashes": [],
        "progress": {"current": 0, "total": request["image_count"]},
        "runtime_identity": None,
        "report_identity": None,
        "failure": None,
        "cancellation": None,
        "timeout": None,
        "terminal_status": None,
        "started_at": planned.timestamp,
        "ended_at": None,
        "deadline_at": deadline_at,
    }
    seen_validated = False
    seen_started = False
    terminal_event_type: str | None = None
    stage = "planned"
    status = ProductStatus.RUNNING.value
    for event_index, event in enumerate(events[1:], start=1):
        if terminal_event_type is not None:
            raise GenerationSafetyError("The Playground event stream contains a post-terminal event.")
        if event.total != request["image_count"]:
            raise GenerationSafetyError("A Playground event changed the requested image count.")
        _validate_transition(
            event.metrics.get("transition"),
            sequence=event_index + 1,
            prior_events=events[:event_index],
            prior_cache=cache,
        )
        if event.event_type == "validated":
            if seen_validated or seen_started or cache["results"]:
                raise GenerationSafetyError("The Playground validated event is duplicated or out of order.")
            if (
                event.stage != "validated"
                or event.status is not ProductStatus.RUNNING
                or event.current != 0
                or set(event.metrics) != {"transition", "generation_adapter_identity", "exploratory"}
                or event.metrics.get("generation_adapter_identity") != adapter_identity
                or event.metrics.get("exploratory") is not True
                or event.message != "Checkpoint, parameters, and adapter identity validated."
                or event.artifact_references
            ):
                raise GenerationSafetyError("The Playground validated event is malformed.")
            seen_validated = True
        elif event.event_type == "generation_started":
            if not seen_validated or seen_started or cache["results"]:
                raise GenerationSafetyError("The Playground generation-start event is duplicated or out of order.")
            if (
                event.stage != "generation_started"
                or event.status is not ProductStatus.RUNNING
                or event.current != 0
                or set(event.metrics) != {"transition", "exploratory"}
                or event.metrics.get("exploratory") is not True
                or event.message != "Exploratory generation started after explicit user action."
                or event.artifact_references
            ):
                raise GenerationSafetyError("The Playground generation-start event is malformed.")
            seen_started = True
        elif event.event_type == "image_completed":
            if not seen_started:
                raise GenerationSafetyError("A Playground image event precedes generation start.")
            index = len(cache["results"])
            if (
                event.stage != "image_completed"
                or event.status is not ProductStatus.RUNNING
                or event.current != index + 1
                or set(event.metrics) != {"transition", "result", "artifact_identity", "exploratory"}
                or event.metrics.get("exploratory") is not True
            ):
                raise GenerationSafetyError("A Playground image event is malformed.")
            result = _validate_result(
                event.metrics.get("result"),
                index=index,
                request=request,
                checkpoint_identity=checkpoint_identity,
                generation_parameters=generation_parameters,
            )
            if result["result_id"] != f"{planned.run_id}-{index:03d}":
                raise GenerationSafetyError("A Playground image event has the wrong run identity.")
            artifact = _validate_artifact_identity(event.metrics.get("artifact_identity"), result=result)
            if tuple(event.artifact_references) != (artifact["reference"],):
                raise GenerationSafetyError("A Playground image event artifact binding is malformed.")
            if event.message != f"Exploratory image {index + 1} of {request['image_count']} completed.":
                raise GenerationSafetyError("A Playground image event message is malformed.")
            cache["results"].append(result)
            cache["artifact_identities"].append(artifact)
            cache["output_hashes"].append(artifact["sha256"])
            cache["progress"] = {"current": len(cache["results"]), "total": request["image_count"]}
        elif event.event_type in _TERMINAL_EVENT_TYPES:
            if (
                event.stage
                != {
                    "generation_completed": "generation_completed",
                    "failed": "failed",
                    "cancelled": "cancelled",
                    "timed_out": "timed_out",
                }[event.event_type]
                or event.status is not _PRODUCT_STATUS_BY_TERMINAL[_TERMINAL_STATUS_BY_EVENT[event.event_type]]
                or event.current != len(cache["results"])
                or set(event.metrics) != {"transition", "terminal", "exploratory"}
                or event.metrics.get("exploratory") is not True
            ):
                raise GenerationSafetyError("The Playground terminal event is malformed.")
            if event.event_type == "generation_completed" and (
                not seen_started or len(cache["results"]) != request["image_count"]
            ):
                raise GenerationSafetyError("The Playground completion event precedes complete image evidence.")
            terminal = _validate_terminal_record(event.metrics.get("terminal"), base_cache=cache, event=event)
            expected_message = {
                "generation_completed": "Exploratory Playground generation completed durably.",
                "failed": "Exploratory Playground generation failed.",
                "cancelled": "Exploratory Playground generation was cancelled.",
                "timed_out": "Exploratory Playground generation reached its fixed deadline.",
            }[event.event_type]
            if event.message != expected_message or list(event.artifact_references) != [
                item["reference"] for item in cache["artifact_identities"]
            ]:
                raise GenerationSafetyError("The Playground terminal event bindings are malformed.")
            cache = dict(terminal["cache"])
            terminal_event_type = event.event_type
            status = _TERMINAL_STATUS_BY_EVENT[event.event_type]
        else:
            raise GenerationSafetyError("The Playground event stream contains an unsupported event type.")
        stage = event.stage
    return _LifecycleReduction(
        cache=_json_copy(cache),
        status=status,
        stage=stage,
        event_count=len(fingerprints),
        terminal_event_type=terminal_event_type,
        stages=tuple(event.stage for event in events),
    )


def _validate_report(value: Any, *, cache: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REPORT_KEYS:
        raise GenerationSafetyError("The Playground report schema is malformed.")
    report = dict(value)
    if (
        report.get("schema_version") != PLAYGROUND_SCHEMA
        or report.get("run_id") != run_id
        or report.get("generation_id") != run_id
        or report.get("scope") != EXPLORATORY_SCOPE
        or report.get("request") != cache.get("request")
        or report.get("checkpoint_identity") != cache.get("checkpoint_identity")
        or report.get("generation_adapter_identity") != cache.get("generation_adapter_identity")
        or report.get("runtime_identity") != cache.get("runtime_identity")
        or report.get("started_at") != cache.get("started_at")
        or report.get("ended_at") != cache.get("ended_at")
        or report.get("results") != cache.get("results")
        or report.get("artifact_identities") != cache.get("artifact_identities")
        or report.get("excluded_from_frozen_benchmark") is not True
        or report.get("excluded_from_promotion_evidence") is not True
    ):
        raise GenerationSafetyError("The Playground report bindings are inconsistent.")
    product_run = report.get("product_run")
    if not isinstance(product_run, Mapping) or set(product_run) != _PRODUCT_RUN_KEYS:
        raise GenerationSafetyError("The Playground report ProductRun schema is malformed.")
    expected_artifacts = [item["reference"] for item in cache["artifact_identities"]]
    if (
        product_run.get("run_id") != run_id
        or product_run.get("feature") != "playground"
        or product_run.get("action_id") != "generate"
        or product_run.get("status") != "COMPLETE"
        or product_run.get("backend_id") != cache["generation_adapter_identity"]["adapter"]
        or product_run.get("started_at") != cache.get("started_at")
        or product_run.get("ended_at") != cache.get("ended_at")
        or product_run.get("artifact_references") != expected_artifacts
    ):
        raise GenerationSafetyError("The Playground report ProductRun bindings are inconsistent.")
    validate_runtime_identity(report.get("runtime_identity"))
    return report


def _terminal_record(status: str, cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAYGROUND_TERMINAL_SCHEMA,
        "terminal_status": status,
        "command_sha256": cache["command_sha256"],
        "request": _json_copy(cache["request"]),
        "checkpoint_identity": _json_copy(cache["checkpoint_identity"]),
        "generation_adapter_identity": _json_copy(cache["generation_adapter_identity"]),
        "cache": _json_copy(cache),
        "cache_identity": _identity(cache),
    }


class PromptPresetStore:
    """Small JSON preset store; values are generation parameters, never executable code."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            value = strict_json_loads(self.path.read_bytes())
        except (OSError, UnicodeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def list(self) -> list[dict[str, Any]]:
        return [{"name": name, **value} for name, value in sorted(self._read().items())]

    def save(self, name: str, request: GenerationRequest) -> dict[str, Any]:
        normalized = name.strip()
        if not normalized or len(normalized) > 80:
            raise ValueError("Preset name must contain 1 through 80 characters.")
        values = self._read()
        values[normalized] = asdict(request)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(strict_json_dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"name": normalized, **values[normalized]}

    def load(self, name: str) -> GenerationRequest:
        value = self._read().get(name)
        if value is None:
            raise KeyError(f"Unknown prompt preset: {name}")
        return GenerationRequest(**value)


class PlaygroundService:
    def __init__(
        self,
        catalog: CheckpointCatalog,
        *,
        output_root: Path,
        generator: PlaygroundGenerator | None = None,
        runs_directory: Path | None = None,
        run_id_factory: Callable[[], str] | None = None,
        initialization_hook: Callable[[str, Path], None] | None = None,
        catalog_provider: Callable[[], CheckpointCatalog] | None = None,
        exploratory_catalog: ExploratoryCheckpointCatalog | None = None,
        exploratory_catalog_provider: Callable[[], ExploratoryCheckpointCatalog] | None = None,
    ) -> None:
        self.catalog = catalog
        self.catalog_provider = catalog_provider
        self.exploratory_catalog = exploratory_catalog or ExploratoryCheckpointCatalog()
        self.exploratory_catalog_provider = exploratory_catalog_provider
        self.output_root = output_root.resolve()
        self.generator = generator
        self.run_id_factory = run_id_factory or _new_run_id
        self.initialization_hook = initialization_hook
        self.presets = PromptPresetStore(self.output_root / "presets.json")
        self.repository = EventRepository(
            (runs_directory or self.output_root).resolve(),
            private_roots=(self.output_root.parent,),
        )
        self._recover_startup_projections()
        self._reconcile_expired_generations()
        self.startup_incomplete_run_ids = tuple(item["run_id"] for item in self.incomplete_initializations())

    def defaults(self) -> dict[str, Any]:
        checkpoint_id = self.catalog.default_checkpoint_id or self.exploratory_catalog.default_checkpoint_id or ""
        return GenerationRequest.defaults(checkpoint_id)

    @property
    def confirmation_required(self) -> bool:
        if self.generator is None:
            return False
        remote = getattr(self.generator, "remote", False)
        billable = getattr(self.generator, "billable", False)
        if type(remote) is not bool or type(billable) is not bool:
            raise GenerationSafetyError("The generation adapter boolean capabilities are malformed.")
        return remote or billable

    @contextmanager
    def _run_transaction(self, run_id: str) -> Iterator[Path]:
        directory = self.repository.run_directory(run_id)
        if directory is None or not directory.is_dir():
            raise KeyError(f"Unknown Playground run: {run_id}")
        with self.repository._lock, event_history_transaction_lock(directory):
            yield directory

    def _strict_snapshot_locked(self, run_id: str) -> tuple[dict[str, Any], list[ProductEvent]]:
        state = self.repository.state(run_id)
        replay = self.repository.replay(run_id)
        indexed = list(replay.events)
        if (
            not state
            or replay.integrity_status != "VALID"
            or replay.invalid_event_count
            or [item.event_id for item in indexed] != list(range(1, len(indexed) + 1))
        ):
            raise GenerationSafetyError("The Playground lifecycle snapshot is not authoritative.")
        return state, [item.event for item in indexed]

    @staticmethod
    def _snapshot_token(state: Mapping[str, Any], events: Sequence[ProductEvent]) -> tuple[Any, ...]:
        return (
            _identity(state),
            len(events),
            _event_stream_identity(events),
            _event_identity(events[-1]) if events else None,
        )

    @staticmethod
    def _validate_state_projection(
        state: Mapping[str, Any],
        events: Sequence[ProductEvent],
    ) -> _LifecycleReduction:
        reduction = _reduce_lifecycle_events(events)
        cache = _cache_from_state(state)
        if cache != reduction.cache:
            raise GenerationSafetyError("The Playground state cache diverges from authoritative events.")
        _validate_lifecycle_commit(
            state.get("playground_lifecycle_commit"),
            events=events,
            cache=reduction.cache,
        )
        last = events[-1]
        expected_last = {
            "event_id": len(events),
            "event_type": last.event_type,
            "timestamp": last.timestamp,
        }
        if (
            state.get("status") != last.status.value
            or state.get("stage") != last.stage
            or state.get("message") != last.message
            or state.get("last_durable_event") != expected_last
        ):
            raise GenerationSafetyError("The Playground state lifecycle projection is inconsistent.")
        return reduction

    @staticmethod
    def _prior_state_before_last_append(
        state: Mapping[str, Any],
        prior_events: Sequence[ProductEvent],
        prior_cache: Mapping[str, Any],
    ) -> dict[str, Any]:
        prior = _json_copy(state)
        previous = prior_events[-1]
        prior.update(prior_cache)
        prior.update(
            {
                "status": previous.status.value,
                "stage": previous.stage,
                "message": previous.message,
                "last_durable_event": {
                    "event_id": len(prior_events),
                    "event_type": previous.event_type,
                    "timestamp": previous.timestamp,
                },
                "event_canonical_current_identity_sha256": hashlib.sha256(
                    _canonical_event_bytes(prior_events)
                ).hexdigest(),
            }
        )
        return prior

    def _recover_projection_locked(self, run_id: str) -> bool:
        state, events = self._strict_snapshot_locked(run_id)
        reduction = _reduce_lifecycle_events(events)
        try:
            self._validate_state_projection(state, events)
        except GenerationSafetyError:
            pass
        else:
            return False
        if len(events) < 2:
            raise GenerationSafetyError("The Playground state is ahead of or divergent from its event stream.")
        prior_events = events[:-1]
        prior_reduction = _reduce_lifecycle_events(prior_events)
        prior_state = self._prior_state_before_last_append(state, prior_events, prior_reduction.cache)
        if _cache_from_state(prior_state) != prior_reduction.cache:
            raise GenerationSafetyError("The Playground state is not an exact one-event projection lag.")
        prior_commit = _validate_lifecycle_commit(
            prior_state.get("playground_lifecycle_commit"),
            events=prior_events,
            cache=prior_reduction.cache,
        )
        last = events[-1]
        transition = _validate_transition(
            last.metrics.get("transition"),
            sequence=len(events),
            prior_events=prior_events,
            prior_cache=prior_reduction.cache,
        )
        if (
            transition["prior_state_identity"] != _identity(prior_state)
            or transition["prior_commit_identity"] != _identity(prior_commit)
            or state.get("status") != last.status.value
            or state.get("stage") != last.stage
            or state.get("message") != last.message
            or state.get("last_durable_event")
            != {"event_id": len(events), "event_type": last.event_type, "timestamp": last.timestamp}
            or state.get("event_canonical_current_identity_sha256")
            != hashlib.sha256(_canonical_event_bytes(events)).hexdigest()
        ):
            raise GenerationSafetyError("The lagging Playground projection does not authenticate its prior state.")
        self.repository.update_state(
            run_id,
            **reduction.cache,
            playground_lifecycle_commit=_lifecycle_commit_for(events, reduction.cache),
        )
        _playground_lifecycle_checkpoint("state_projection_recovered", self.repository.run_directory(run_id))  # type: ignore[arg-type]
        repaired, repaired_events = self._strict_snapshot_locked(run_id)
        self._validate_state_projection(repaired, repaired_events)
        return True

    def _recover_projection(self, run_id: str) -> bool:
        with self._run_transaction(run_id):
            return self._recover_projection_locked(run_id)

    def _authoritative_cache(self, run_id: str) -> dict[str, Any]:
        self._recover_projection(run_id)
        with self._run_transaction(run_id):
            state, events = self._strict_snapshot_locked(run_id)
            return _json_copy(self._validate_state_projection(state, events).cache)

    def _commit_lifecycle_event(
        self,
        run_id: str,
        *,
        timestamp: str,
        stage: str,
        event_type: str,
        status: ProductStatus,
        current: int,
        total: int,
        message: str,
        cache_transform: Callable[[dict[str, Any]], dict[str, Any]],
        metrics_factory: Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]],
        artifacts: tuple[str, ...] = (),
        publish: Callable[[Path], None] | None = None,
        publish_checkpoint: str | None = None,
        operation_check: Callable[[], None] | None = None,
    ) -> bool:
        self._recover_projection(run_id)
        with self._run_transaction(run_id) as directory:
            if operation_check is not None:
                operation_check()
            expected_state, expected_events = self._strict_snapshot_locked(run_id)
            expected_reduction = self._validate_state_projection(expected_state, expected_events)
            if expected_reduction.terminal_event_type is not None:
                return False
            expected_token = self._snapshot_token(expected_state, expected_events)
        _playground_lifecycle_checkpoint("expected_snapshot_captured", directory)

        with self._run_transaction(run_id) as directory:
            if operation_check is not None:
                operation_check()
            state, events = self._strict_snapshot_locked(run_id)
            reduction = self._validate_state_projection(state, events)
            if self._snapshot_token(state, events) != expected_token:
                if operation_check is not None:
                    operation_check()
                if reduction.terminal_event_type is not None:
                    return False
                raise GenerationSafetyError("The Playground lifecycle changed before its CAS commit.")
            if reduction.terminal_event_type is not None:
                return False
            prior_cache = reduction.cache
            next_cache = _json_copy(cache_transform(_json_copy(prior_cache)))
            transition = _transition_for(
                sequence=len(events) + 1,
                prior_events=events,
                prior_state=state,
                prior_cache=prior_cache,
            )
            metrics = {"transition": transition, **dict(metrics_factory(prior_cache, next_cache))}
            event = ProductEvent(
                run_id=run_id,
                timestamp=timestamp,
                feature="playground",
                stage=stage,
                event_type=event_type,
                status=status,
                current=current,
                total=total,
                message=message,
                metrics=metrics,
                artifact_references=artifacts,
            )
            prospective = _reduce_lifecycle_events([*events, event])
            if next_cache != prospective.cache:
                raise GenerationSafetyError("The Playground lifecycle cache does not match its proposed event.")
            if operation_check is not None:
                operation_check()
            if publish is not None:
                publish(directory)
                if publish_checkpoint is not None:
                    _playground_lifecycle_checkpoint(publish_checkpoint, directory)
            if operation_check is not None:
                operation_check()
            self.repository.append(event)
            _playground_lifecycle_checkpoint("event_committed", directory)
            committed_events = [*events, event]
            self.repository.update_state(
                run_id,
                **prospective.cache,
                playground_lifecycle_commit=_lifecycle_commit_for(committed_events, prospective.cache),
            )
            _playground_lifecycle_checkpoint("state_projected", directory)
            projected_state, projected_events = self._strict_snapshot_locked(run_id)
            self._validate_state_projection(projected_state, projected_events)
            return True

    def generate(
        self,
        request: GenerationRequest,
        *,
        explicit_action: bool,
        confirm_billable: bool = False,
        _run_id: str | None = None,
        _initialize: bool = True,
    ) -> dict[str, Any]:
        if type(explicit_action) is not bool or explicit_action is not True:
            raise GenerationSafetyError("Generation requires an explicit Generate action.")
        if type(confirm_billable) is not bool:
            raise GenerationSafetyError("The billable confirmation must be an exact boolean.")
        if not isinstance(request, GenerationRequest):
            raise GenerationSafetyError("The Playground request must be a validated GenerationRequest.")
        if self.generator is None:
            raise GeneratorUnavailableError("No typed generation adapter is configured.")

        run_id = _run_id or self.run_id_factory()
        if not isinstance(run_id, str) or not run_id:
            raise GenerationSafetyError("The Playground run identity is malformed.")
        if _initialize:
            started_at = _utc_now()
            deadline_at = (
                datetime.fromisoformat(started_at) + timedelta(seconds=PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS)
            ).isoformat()
        else:
            continuation_state = self.repository.state(run_id)
            if continuation_state.get("feature") != "playground":
                raise GenerationSafetyError("The planned Playground run is unavailable for continuation.")
            started_at = _aware_timestamp(continuation_state.get("started_at"))
            deadline_at = _aware_timestamp(continuation_state.get("deadline_at"))

        prepare_control = getattr(self.generator, "prepare_control", None)
        finish_control = getattr(self.generator, "finish_control", None)
        if prepare_control is not None and not callable(prepare_control):
            raise GenerationSafetyError("The generation adapter control hook is malformed.")
        if finish_control is not None and not callable(finish_control):
            raise GenerationSafetyError("The generation adapter control hook is malformed.")

        control_started = False
        try:
            if callable(prepare_control):
                prepare_control(run_id, deadline_at)
            control_started = True
            prepared_check_factory = getattr(self.generator, "prepared_operation_check", None)
            if prepared_check_factory is None:
                prepared_check_factory = getattr(self.generator, "_prepared_operation_check", None)
            prepared_check = prepared_check_factory() if callable(prepared_check_factory) else None
            if prepared_check is not None and not callable(prepared_check):
                raise GenerationSafetyError("The generation adapter operation check is malformed.")

            def preinitialization_check() -> None:
                deadline = _parse_deadline(deadline_at)
                if deadline is None or datetime.now(timezone.utc) >= deadline:
                    raise GenerationTimedOutError("Generation reached its durable wall-clock deadline.")
                if prepared_check is not None:
                    prepared_check()

            preinitialization_check()
            requires_fresh_catalog = getattr(self.generator, "requires_fresh_catalog", False)
            if type(requires_fresh_catalog) is not bool:
                raise GenerationSafetyError("The generation adapter catalog capability is malformed.")
            if requires_fresh_catalog and self.catalog_provider is None and self.exploratory_catalog_provider is None:
                raise GenerationSafetyError(
                    "The local generation adapter requires a fresh checkpoint catalog provider."
                )
            if self.confirmation_required and confirm_billable is not True:
                raise GenerationSafetyError("Remote or billable generation requires explicit cost confirmation.")

            preinitialization_check()
            current_catalog = self.catalog_provider() if self.catalog_provider is not None else self.catalog
            preinitialization_check()
            current_exploratory = (
                self.exploratory_catalog_provider()
                if self.exploratory_catalog_provider is not None
                else self.exploratory_catalog
            )
            preinitialization_check()
            self.catalog = current_catalog
            self.exploratory_catalog = current_exploratory
            checkpoint = current_catalog.find(request.checkpoint_id, weights=request.weights)
            checkpoint_classification = "production_complete"
            if checkpoint is None:
                checkpoint = current_exploratory.find(request.checkpoint_id, weights=request.weights)
                checkpoint_classification = "exploratory_smoke"
            if (
                checkpoint is None
                or checkpoint.path is None
                or checkpoint.checkpoint_sha256 is None
                or checkpoint.checkpoint_step is None
            ):
                raise GenerationSafetyError("The selected checkpoint and live/EMA variant is not eligible.")

            request_payload = _validate_request_payload(asdict(request))
            checkpoint_run_id = str(
                getattr(checkpoint, "run_id", None) or getattr(checkpoint, "registration_id", "exploratory-smoke")
            )
            checkpoint_identity = _validate_checkpoint_identity(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_run_id": checkpoint_run_id,
                    "checkpoint_step": checkpoint.checkpoint_step,
                    "weights": request.weights,
                    "sha256": checkpoint.checkpoint_sha256,
                    "classification": checkpoint_classification,
                    "purpose": str(getattr(checkpoint, "purpose", "production")),
                    "registration_identity": getattr(checkpoint, "registration_identity", None),
                    "evidence_identity": getattr(checkpoint, "evidence_identity", None),
                    "dataset_freeze_identity": getattr(checkpoint, "freeze_identity", None),
                    "campaign_identity": getattr(checkpoint, "campaign_identity", None),
                    "training_code_identity": getattr(checkpoint, "code_identity", None),
                    "production_eligible": checkpoint_classification != "exploratory_smoke",
                    "evaluation_eligible": checkpoint_classification != "exploratory_smoke",
                    "training_resume_eligible": checkpoint_classification != "exploratory_smoke",
                    "promotion_eligible": checkpoint_classification != "exploratory_smoke",
                }
            )
            preinitialization_check()
            adapter_identity = self._adapter_identity()
            preinitialization_check()
            command = _validate_command(
                {
                    "schema_version": PLAYGROUND_COMMAND_SCHEMA,
                    "action": "playground.generate",
                    "run_id": run_id,
                    "request": request_payload,
                    "checkpoint_identity": checkpoint_identity,
                    "generation_adapter_identity": adapter_identity,
                    "scope": EXPLORATORY_SCOPE,
                    "benchmark_eligible": False,
                    "promotion_evidence_eligible": False,
                },
                run_id=run_id,
            )
            command_bytes = _json_bytes(command)
            command_sha256 = hashlib.sha256(command_bytes).hexdigest()
            generation_parameters = {
                "sampling_steps": request.sampling_steps,
                "guidance": request.guidance,
                "image_count": request.image_count,
            }
            planned_event = ProductEvent(
                run_id=run_id,
                timestamp=started_at,
                feature="playground",
                stage="planned",
                event_type="planned",
                status=ProductStatus.RUNNING,
                current=0,
                total=request.image_count,
                message="Exploratory Playground generation planned.",
                metrics={
                    "command_sha256": command_sha256,
                    "prompt": request.prompt,
                    "checkpoint_identity": checkpoint_identity,
                    "request": request_payload,
                    "generation_adapter_identity": adapter_identity,
                    "deadline_at": deadline_at,
                    "exploratory": True,
                    "benchmark_eligible": False,
                    "promotion_evidence_eligible": False,
                },
            )
            planned_reduction = _reduce_lifecycle_events([planned_event])
            state_extra = {
                "playground_schema_version": PLAYGROUND_RUN_SCHEMA,
                "playground_terminal_protocol": PLAYGROUND_TERMINAL_SCHEMA,
                "command_reference": "command.json",
                "wall_clock_limit_seconds": PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS,
                "exploratory": True,
                "benchmark_eligible": False,
                "promotion_evidence_eligible": False,
                "exploratory_classification": EXPLORATORY_SCOPE,
                "benchmark_eligibility": False,
                "promotion_evidence_eligibility": False,
                **planned_reduction.cache,
                "playground_lifecycle_commit": _lifecycle_commit_for([planned_event], planned_reduction.cache),
            }

            preinitialization_check()
            if _initialize:
                self.repository.initialize_run(
                    run_id,
                    feature="playground",
                    command="playground.generate",
                    command_payload=command,
                    planned_event=planned_event,
                    status=ProductStatus.RUNNING.value,
                    stage="planned",
                    started_at=started_at,
                    resumable=False,
                    backend_id=str(adapter_identity["adapter"]),
                    backend_identity=adapter_identity,
                    report_reference="report/report.json",
                    extra=state_extra,
                    on_step=self.initialization_hook,
                )
            else:
                self._recover_projection(run_id)
                with self._run_transaction(run_id) as continuation_directory:
                    state, events = self._strict_snapshot_locked(run_id)
                    reduction = self._validate_state_projection(state, events)
                    stored_command_path = continuation_directory / "command.json"
                    try:
                        stored_command, stored_command_sha256 = _read_authorized_command_snapshot(
                            stored_command_path,
                            root=continuation_directory,
                        )
                    except GenerationSafetyError:
                        stored_command, stored_command_sha256 = {}, None
                    if (
                        reduction.terminal_event_type is not None
                        or [event.event_type for event in events] != ["planned"]
                        or stored_command_sha256 != command_sha256
                        or stored_command != command
                        or _validate_command(stored_command, run_id=run_id) != command
                        or any(not (continuation_directory / name).is_dir() for name in ("logs", "artifacts", "report"))
                    ):
                        raise GenerationSafetyError("The planned Playground run is not safe for explicit continuation.")

            directory = self.repository.run_directory(run_id)
            if directory is None:
                raise OSError("Playground runs directory is unavailable.")

            def operation_check() -> None:
                preinitialization_check()
                self._assert_active(run_id)

            try:
                operation_check()
                if not self._commit_lifecycle_event(
                    run_id,
                    timestamp=_utc_now(),
                    stage="validated",
                    event_type="validated",
                    status=ProductStatus.RUNNING,
                    current=0,
                    total=request.image_count,
                    message="Checkpoint, parameters, and adapter identity validated.",
                    cache_transform=lambda cache: cache,
                    metrics_factory=lambda _prior, _next: {
                        "generation_adapter_identity": adapter_identity,
                        "exploratory": True,
                    },
                    operation_check=operation_check,
                ):
                    operation_check()
                    raise GenerationSafetyError("A terminal Playground transition already won.")

                if not self._commit_lifecycle_event(
                    run_id,
                    timestamp=_utc_now(),
                    stage="generation_started",
                    event_type="generation_started",
                    status=ProductStatus.RUNNING,
                    current=0,
                    total=request.image_count,
                    message="Exploratory generation started after explicit user action.",
                    cache_transform=lambda cache: cache,
                    metrics_factory=lambda _prior, _next: {"exploratory": True},
                    operation_check=operation_check,
                ):
                    operation_check()
                    raise GenerationSafetyError("A terminal Playground transition already won.")

                operation_check()
                raw_assets = list(
                    self.generator.generate(
                        checkpoint=checkpoint.path,
                        prompt=request.prompt,
                        seed=request.seed,
                        sampling_steps=request.sampling_steps,
                        guidance=request.guidance,
                        image_count=request.image_count,
                        weights=request.weights,
                        expected_sha256=checkpoint.checkpoint_sha256,
                        expected_step=checkpoint.checkpoint_step,
                        expected_variant=checkpoint.weights,
                    )
                )
                operation_check()
                if len(raw_assets) != request.image_count:
                    raise RuntimeError("Generator returned a different number of images than requested.")

                for index, raw in enumerate(raw_assets):
                    operation_check()
                    content, media_type = _asset_bytes(raw)
                    operation_check()
                    digest = hashlib.sha256(content).hexdigest()
                    extension = ".png" if media_type == "image/png" else ".bin"
                    reference = f"artifacts/image_{index:03d}{extension}"
                    completed_at = _utc_now()
                    result = _validate_result(
                        {
                            "result_id": f"{run_id}-{index:03d}",
                            "checkpoint_identity": checkpoint.checkpoint_id,
                            "checkpoint_run_id": checkpoint_run_id,
                            "checkpoint_step": checkpoint.checkpoint_step,
                            "weights": request.weights,
                            "prompt": request.prompt,
                            "seed": request.seed + index,
                            "generation_parameters": generation_parameters,
                            "timestamp": completed_at,
                            "output_hash": digest,
                            "application_version": _application_version(),
                            "media_type": media_type,
                            "output_reference": reference,
                            "scope": EXPLORATORY_SCOPE,
                            "frozen_benchmark_eligible": False,
                            "promotion_evidence_eligible": False,
                        },
                        index=index,
                        request=request_payload,
                        checkpoint_identity=checkpoint_identity,
                        generation_parameters=generation_parameters,
                    )
                    artifact_identity = _validate_artifact_identity(
                        {
                            "artifact_id": result["result_id"],
                            "reference": reference,
                            "sha256": digest,
                            "size_bytes": len(content),
                            "media_type": media_type,
                        },
                        result=result,
                        content_size=len(content),
                    )

                    def append_result(
                        cache: dict[str, Any],
                        *,
                        expected_index: int = index,
                        result_value: dict[str, Any] = result,
                        artifact_value: dict[str, Any] = artifact_identity,
                    ) -> dict[str, Any]:
                        if len(cache["results"]) != expected_index:
                            raise GenerationSafetyError("The Playground result sequence changed before publication.")
                        cache["results"] = [*cache["results"], result_value]
                        cache["artifact_identities"] = [
                            *cache["artifact_identities"],
                            artifact_value,
                        ]
                        cache["output_hashes"] = [item["sha256"] for item in cache["artifact_identities"]]
                        cache["progress"] = {
                            "current": len(cache["results"]),
                            "total": request.image_count,
                        }
                        return cache

                    def publish_artifact(
                        run_directory: Path,
                        *,
                        reference_value: str = reference,
                        content_value: bytes = content,
                    ) -> None:
                        _publish_exact_bytes(run_directory, reference_value, content_value)

                    if not self._commit_lifecycle_event(
                        run_id,
                        timestamp=completed_at,
                        stage="image_completed",
                        event_type="image_completed",
                        status=ProductStatus.RUNNING,
                        current=index + 1,
                        total=request.image_count,
                        message=f"Exploratory image {index + 1} of {request.image_count} completed.",
                        cache_transform=append_result,
                        metrics_factory=lambda _prior, _next, result_value=result, artifact_value=artifact_identity: {
                            "result": result_value,
                            "artifact_identity": artifact_value,
                            "exploratory": True,
                        },
                        artifacts=(reference,),
                        publish=publish_artifact,
                        publish_checkpoint="artifact_published",
                        operation_check=operation_check,
                    ):
                        operation_check()
                        raise GenerationSafetyError("A terminal Playground transition already won.")

                operation_check()
                refreshed_catalog = self.catalog_provider() if self.catalog_provider is not None else self.catalog
                operation_check()
                refreshed_exploratory = (
                    self.exploratory_catalog_provider()
                    if self.exploratory_catalog_provider is not None
                    else self.exploratory_catalog
                )
                operation_check()
                self.catalog = refreshed_catalog
                self.exploratory_catalog = refreshed_exploratory
                refreshed = refreshed_catalog.find(request.checkpoint_id, weights=request.weights)
                if refreshed is None:
                    refreshed = refreshed_exploratory.find(request.checkpoint_id, weights=request.weights)
                if (
                    refreshed is None
                    or refreshed.path != checkpoint.path
                    or refreshed.checkpoint_sha256 != checkpoint.checkpoint_sha256
                    or refreshed.checkpoint_step != checkpoint.checkpoint_step
                    or refreshed.weights != checkpoint.weights
                ):
                    raise GenerationSafetyError("Checkpoint eligibility changed before generation completion.")
                operation_check()
                final_adapter_identity = self._adapter_identity()
                operation_check()
                if final_adapter_identity != adapter_identity:
                    raise GenerationSafetyError("Generation adapter code identity changed while sampling.")
                runtime_identity = validate_runtime_identity(
                    getattr(self.generator, "last_runtime_identity", None) or _unreported_runtime_identity()
                )
                operation_check()

                base_cache = self._authoritative_cache(run_id)
                ended_at = _utc_now()
                complete_cache = {
                    **base_cache,
                    "runtime_identity": runtime_identity,
                    "failure": None,
                    "cancellation": None,
                    "timeout": None,
                    "terminal_status": "COMPLETE",
                    "ended_at": ended_at,
                }
                product_run = ProductRun(
                    run_id=run_id,
                    feature="playground",
                    action_id="generate",
                    status=ProductStatus.COMPLETE,
                    backend_id=str(adapter_identity["adapter"]),
                    started_at=started_at,
                    ended_at=ended_at,
                    artifact_references=tuple(item["reference"] for item in complete_cache["artifact_identities"]),
                )
                report = {
                    "schema_version": PLAYGROUND_SCHEMA,
                    "run_id": run_id,
                    "generation_id": run_id,
                    "product_run": _product_run_dict(product_run),
                    "scope": EXPLORATORY_SCOPE,
                    "request": request_payload,
                    "checkpoint_identity": checkpoint_identity,
                    "generation_adapter_identity": adapter_identity,
                    "runtime_identity": runtime_identity,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "results": complete_cache["results"],
                    "artifact_identities": complete_cache["artifact_identities"],
                    "excluded_from_frozen_benchmark": True,
                    "excluded_from_promotion_evidence": True,
                }
                report_bytes = _json_bytes(report)
                report_identity = _validate_report_identity(
                    {
                        "reference": "report/report.json",
                        "sha256": hashlib.sha256(report_bytes).hexdigest(),
                        "size_bytes": len(report_bytes),
                    }
                )
                complete_cache["report_identity"] = report_identity
                complete_cache = _cache_from_state(complete_cache)
                _validate_report(report, cache=complete_cache, run_id=run_id)
                terminal = _terminal_record("COMPLETE", complete_cache)

                def complete_transform(cache: dict[str, Any]) -> dict[str, Any]:
                    if cache != base_cache:
                        raise GenerationSafetyError("The Playground evidence changed before terminal publication.")
                    return _json_copy(complete_cache)

                def publish_report(run_directory: Path) -> None:
                    _publish_exact_bytes(run_directory, "report/report.json", report_bytes)

                won = self._commit_lifecycle_event(
                    run_id,
                    timestamp=ended_at,
                    stage="generation_completed",
                    event_type="generation_completed",
                    status=ProductStatus.COMPLETE,
                    current=request.image_count,
                    total=request.image_count,
                    message="Exploratory Playground generation completed durably.",
                    cache_transform=complete_transform,
                    metrics_factory=lambda _prior, _next: {
                        "terminal": terminal,
                        "exploratory": True,
                    },
                    artifacts=tuple(item["reference"] for item in complete_cache["artifact_identities"]),
                    publish=publish_report,
                    publish_checkpoint="report_published",
                    operation_check=operation_check,
                )
                if not won:
                    operation_check()
                    raise GenerationSafetyError("A terminal Playground transition already won.")
                result = self.reconstruct(run_id)
                if result.get("status") != "COMPLETE" or result.get("comparability") != "CURRENT":
                    raise GenerationSafetyError("The completed Playground terminal evidence is not reconstructable.")
                return result
            except GenerationCancelledError:
                self._record_cancellation(
                    run_id,
                    "Generation adapter cancelled the run.",
                )
                raise
            except GenerationTimedOutError:
                self._record_timeout(
                    run_id,
                    "Generation reached its wall-clock deadline.",
                )
                raise
            except Exception as exc:
                self._record_failure(run_id, exc)
                raise
        finally:
            if control_started and callable(finish_control):
                finish_control(run_id)

    def continue_run(
        self,
        run_id: str,
        *,
        explicit_action: bool,
        confirm_billable: bool = False,
    ) -> dict[str, Any]:
        """Continue one fully initialized planned run only after an explicit action."""

        state = self.repository.state(run_id)
        request = state.get("request") if isinstance(state.get("request"), Mapping) else None
        if request is None:
            raise GenerationSafetyError("The planned Playground request is unavailable for continuation.")
        try:
            generation_request = GenerationRequest(**dict(request))
        except (TypeError, ValueError) as exc:
            raise GenerationSafetyError("The planned Playground request is invalid.") from exc
        return self.generate(
            generation_request,
            explicit_action=explicit_action,
            confirm_billable=confirm_billable,
            _run_id=run_id,
            _initialize=False,
        )

    def reconstruct(self, run_id: str) -> dict[str, Any]:
        """Rebuild one run from strict events, treating state as a verified cache."""

        directory = self.repository.run_directory(run_id)
        if directory is None or not directory.is_dir():
            raise KeyError(f"Unknown Playground run: {run_id}")
        state = self.repository.state(run_id)
        if not state:
            if (directory / "state.json").exists():
                return _not_comparable(run_id, "state.json is malformed or identity-inconsistent.")
            return _initialization_incomplete(run_id, directory)

        reasons: list[str] = []
        try:
            self._recover_projection(run_id)
        except (GenerationSafetyError, OSError, ValueError):
            reasons.append("Playground recovery could not verify an internal artifact safely.")

        with self._run_transaction(run_id):
            state = self.repository.state(run_id)
            replay = self.repository.replay(run_id)
        indexed = list(replay.events)
        events = [item.event for item in indexed]
        unique_event_count = len({_event_identity(event) for event in events})
        if replay.integrity_status != "VALID" or replay.invalid_event_count:
            reasons.extend(replay.warnings or ("Playground event history is not comparable.",))
        if [item.event_id for item in indexed] != list(range(1, len(indexed) + 1)):
            reasons.append("The Playground event stream is not contiguous.")

        if state.get("playground_schema_version") != PLAYGROUND_RUN_SCHEMA:
            reasons.append("Playground run schema is missing or unsupported.")
        if state.get("playground_terminal_protocol") != PLAYGROUND_TERMINAL_SCHEMA:
            reasons.append("Playground terminal protocol is missing or unsupported.")
        if state.get("feature") != "playground" or state.get("command") != "playground.generate":
            reasons.append("ProductRun feature or action identity is inconsistent.")
        exact_flags = {
            "exploratory": True,
            "benchmark_eligible": False,
            "promotion_evidence_eligible": False,
            "exploratory_classification": EXPLORATORY_SCOPE,
            "benchmark_eligibility": False,
            "promotion_evidence_eligibility": False,
        }
        for key, expected in exact_flags.items():
            if (type(expected) is bool and type(state.get(key)) is not bool) or state.get(key) != expected:
                reasons.append(f"Playground {key} identity changed.")
        if type(state.get("wall_clock_limit_seconds")) is not int or (
            state.get("wall_clock_limit_seconds") != PLAYGROUND_WALL_CLOCK_LIMIT_SECONDS
        ):
            reasons.append("Playground wall-clock policy changed.")
        for name in ("logs", "artifacts", "report"):
            path = directory / name
            if path.is_symlink() or not path.is_dir():
                reasons.append(f"Required {name}/ directory is missing or irregular.")
        event_path = directory / "events.jsonl"
        if event_path.is_symlink() or not event_path.is_file():
            reasons.append("events.jsonl is missing from authoritative Playground state.")

        command_path = directory / "command.json"
        command: dict[str, Any] | None = None
        try:
            raw_command, command_sha256 = _read_authorized_command_snapshot(command_path, root=directory)
        except GenerationSafetyError:
            reasons.append("command.json is missing or irregular.")
        else:
            try:
                command = _validate_command(raw_command, run_id=run_id)
            except GenerationSafetyError as exc:
                reasons.append(str(exc))
            if command_sha256 != state.get("command_sha256"):
                reasons.append("command.json identity changed.")

        reduction: _LifecycleReduction | None = None
        if events:
            try:
                reduction = _reduce_lifecycle_events(events)
            except GenerationSafetyError as exc:
                reasons.append(str(exc))
        else:
            reasons.append("The authoritative planned ProductEvent is missing.")

        cache: dict[str, Any] | None = None
        try:
            cache = _cache_from_state(state)
        except GenerationSafetyError as exc:
            reasons.append(str(exc))
        if reduction is not None and cache is not None:
            if cache != reduction.cache:
                reasons.append("The Playground state cache does not exactly match authoritative events.")
            try:
                _validate_lifecycle_commit(
                    state.get("playground_lifecycle_commit"),
                    events=events,
                    cache=reduction.cache,
                )
            except GenerationSafetyError as exc:
                reasons.append(str(exc))
            last = events[-1]
            if (
                state.get("status") != last.status.value
                or state.get("stage") != last.stage
                or state.get("message") != last.message
                or state.get("last_durable_event")
                != {
                    "event_id": len(events),
                    "event_type": last.event_type,
                    "timestamp": last.timestamp,
                }
            ):
                reasons.append("The generic run state does not match the last authoritative event.")
            if (
                state.get("event_canonical_current_identity_sha256")
                != hashlib.sha256(_canonical_event_bytes(events)).hexdigest()
            ):
                reasons.append("The generic run state event-stream identity changed.")
            if command is not None:
                planned = events[0]
                if (
                    command.get("request") != reduction.cache["request"]
                    or command.get("checkpoint_identity") != reduction.cache["checkpoint_identity"]
                    or command.get("generation_adapter_identity") != reduction.cache["generation_adapter_identity"]
                    or planned.metrics.get("command_sha256") != state.get("command_sha256")
                    or state.get("backend_identity") != reduction.cache["generation_adapter_identity"]
                    or state.get("backend_id") != reduction.cache["generation_adapter_identity"]["adapter"]
                    or state.get("report_reference") != "report/report.json"
                ):
                    reasons.append("Command, event, and run identity bindings disagree.")

        authoritative_cache = reduction.cache if reduction is not None else cache
        if authoritative_cache is None:
            authoritative_cache = {
                "request": {},
                "checkpoint_identity": {},
                "generation_adapter_identity": {},
                "command_sha256": None,
                "prompt": None,
                "weights": None,
                "seed": None,
                "generation_parameters": {},
                "results": [],
                "artifact_identities": [],
                "output_hashes": [],
                "progress": {"current": 0, "total": None},
                "runtime_identity": None,
                "report_identity": None,
                "failure": None,
                "cancellation": None,
                "timeout": None,
                "terminal_status": None,
                "started_at": state.get("started_at"),
                "ended_at": state.get("ended_at"),
                "deadline_at": state.get("deadline_at"),
            }

        stale_reasons: list[str] = []
        results = [dict(item) for item in authoritative_cache.get("results", ())]
        identities = [dict(item) for item in authoritative_cache.get("artifact_identities", ())]
        for index, identity in enumerate(identities):
            reference = _safe_reference(identity.get("reference"))
            if reference is None:
                reasons.append("Artifact reference is unsafe or malformed.")
                continue
            artifact = directory / reference
            if artifact.is_symlink() or not artifact.is_file():
                stale_reasons.append(f"Missing artifact: {reference.as_posix()}.")
                continue
            if artifact.stat().st_size != identity.get("size_bytes") or _file_sha256(artifact) != identity.get(
                "sha256"
            ):
                stale_reasons.append(f"Artifact bytes changed: {reference.as_posix()}.")
            if index >= len(results) or results[index].get("output_reference") != reference.as_posix():
                reasons.append("Ordered result and artifact references disagree.")

        report_available = False
        report_identity = authoritative_cache.get("report_identity")
        terminal_status = (
            reduction.status
            if reduction is not None
            else authoritative_cache.get("terminal_status") or state.get("status") or ProductStatus.NOT_STARTED.value
        )
        if terminal_status == "COMPLETE":
            try:
                validated_report_identity = _validate_report_identity(report_identity)
            except GenerationSafetyError as exc:
                stale_reasons.append(str(exc))
            else:
                report_path = directory / validated_report_identity["reference"]
                if report_path.is_symlink() or not report_path.is_file():
                    stale_reasons.append("Durable Playground report is missing.")
                elif (
                    report_path.stat().st_size != validated_report_identity["size_bytes"]
                    or _file_sha256(report_path) != validated_report_identity["sha256"]
                ):
                    stale_reasons.append("Durable Playground report bytes changed.")
                else:
                    report_value = _read_json(report_path)
                    try:
                        _validate_report(
                            report_value,
                            cache=authoritative_cache,
                            run_id=run_id,
                        )
                    except GenerationSafetyError as exc:
                        reasons.append(str(exc))
                    else:
                        report_available = True
        elif report_identity is not None:
            reasons.append("A non-complete Playground run carries a report identity.")

        status = str(terminal_status)
        stage = reduction.stage if reduction is not None else str(state.get("stage") or "planned")
        comparability = "CURRENT"
        if reasons:
            status, comparability = "NOT_COMPARABLE", "NOT_COMPARABLE"
        elif stale_reasons:
            status, comparability = "STALE", "STALE"
        artifact_references = tuple(item["reference"] for item in identities if isinstance(item.get("reference"), str))
        product_status = ProductStatus(status) if status in ProductStatus._value2member_map_ else ProductStatus.FAILED
        product_run = ProductRun(
            run_id=run_id,
            feature="playground",
            action_id="generate",
            status=product_status,
            backend_id=str(
                dict(authoritative_cache.get("generation_adapter_identity") or {}).get("adapter") or "playground"
            ),
            started_at=(str(authoritative_cache.get("started_at")) if authoritative_cache.get("started_at") else None),
            ended_at=(str(authoritative_cache.get("ended_at")) if authoritative_cache.get("ended_at") else None),
            artifact_references=artifact_references,
        )
        return {
            "schema_version": PLAYGROUND_SCHEMA,
            "durable": True,
            "authoritative": True,
            "run_id": run_id,
            "generation_id": run_id,
            "product_run": _product_run_dict(product_run, status=status),
            "status": status,
            "stage": stage,
            "scope": EXPLORATORY_SCOPE,
            "exploratory": True,
            "benchmark_eligible": False,
            "promotion_evidence_eligible": False,
            "prompt": authoritative_cache.get("prompt"),
            "request": dict(authoritative_cache.get("request") or {}),
            "checkpoint_identity": dict(authoritative_cache.get("checkpoint_identity") or {}),
            "weights": authoritative_cache.get("weights"),
            "seed": authoritative_cache.get("seed"),
            "generation_parameters": dict(authoritative_cache.get("generation_parameters") or {}),
            "generation_adapter_identity": dict(authoritative_cache.get("generation_adapter_identity") or {}),
            "runtime_identity": dict(authoritative_cache.get("runtime_identity") or {}),
            "started_at": authoritative_cache.get("started_at"),
            "ended_at": authoritative_cache.get("ended_at"),
            "results": results,
            "artifact_identities": identities,
            "output_hashes": list(authoritative_cache.get("output_hashes") or ()),
            "progress": dict(authoritative_cache.get("progress") or {"current": len(results), "total": None}),
            "failure": authoritative_cache.get("failure"),
            "cancellation": authoritative_cache.get("cancellation"),
            "timeout": authoritative_cache.get("timeout"),
            "report_available": report_available,
            "event_count": unique_event_count,
            "stages": [event.stage for event in events],
            "comparability": comparability,
            "integrity_reasons": [*reasons, *stale_reasons],
            "excluded_from_frozen_benchmark": True,
            "excluded_from_promotion_evidence": True,
            "frozen_benchmark_eligible": False,
        }

    def latest_run(self) -> dict[str, Any] | None:
        run_ids = self.repository.recent_run_ids(feature="playground", limit=1)
        return self.reconstruct(run_ids[0]) if run_ids else None

    def recent_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return [
            self.reconstruct(run_id) for run_id in self.repository.recent_run_ids(feature="playground", limit=limit)
        ]

    def incomplete_initializations(self) -> list[dict[str, Any]]:
        root = self.repository.runs_directory
        if root is None or not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or self.repository.run_directory(child.name) != child.resolve():
                continue
            if self.repository.state(child.name):
                continue
            rows.append(_initialization_incomplete(child.name, child))
        return rows

    def _recover_startup_projections(self) -> None:
        """Repair only an authenticated one-event cache lag after a crash."""

        for run_id in self.repository.recent_run_ids(feature="playground", limit=1_000):
            try:
                self._recover_projection(run_id)
            except (GenerationSafetyError, OSError, ValueError, KeyError):
                continue

    def _reconcile_expired_generations(self) -> None:
        """Durably time out abandoned generation work after a server restart."""

        for run_id in self.repository.recent_run_ids(feature="playground", limit=1_000):
            state = self.repository.state(run_id)
            if (
                state.get("status") != ProductStatus.RUNNING.value
                or state.get("terminal_status") is not None
                or state.get("stage") not in {"validated", "generation_started", "image_completed"}
            ):
                continue
            deadline = _parse_deadline(state.get("deadline_at"))
            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                self._record_timeout(
                    run_id,
                    "Generation was abandoned past its durable wall-clock deadline.",
                )

    def cancel(self, run_id: str, *, reason: str = "Cancelled by explicit user action.") -> dict[str, Any]:
        if not isinstance(reason, str) or not reason or len(reason) > 2_000:
            raise GenerationSafetyError("The Playground cancellation reason is malformed.")
        state = self.repository.state(run_id)
        if state.get("feature") != "playground":
            raise KeyError(f"Unknown Playground run: {run_id}")
        won = self._record_cancellation(run_id, reason)
        cancel_adapter = getattr(self.generator, "cancel", None)
        if won and callable(cancel_adapter):
            cancel_adapter(run_id)
        return self.reconstruct(run_id)

    def _assert_active(self, run_id: str) -> None:
        state = self.repository.state(run_id)
        terminal_status = state.get("terminal_status")
        if terminal_status == "CANCELLED":
            cancellation = state.get("cancellation")
            reason = cancellation.get("reason") if isinstance(cancellation, Mapping) else None
            raise GenerationCancelledError(str(reason or "Generation was cancelled."))
        if terminal_status == "TIMED_OUT":
            timeout = state.get("timeout")
            reason = timeout.get("reason") if isinstance(timeout, Mapping) else None
            raise GenerationTimedOutError(str(reason or "Generation reached its wall-clock deadline."))
        if terminal_status is not None or state.get("status") != ProductStatus.RUNNING.value:
            raise GenerationSafetyError("Playground run is no longer active.")
        deadline = _parse_deadline(state.get("deadline_at"))
        if deadline is None or datetime.now(timezone.utc) >= deadline:
            raise GenerationTimedOutError("Generation reached its durable wall-clock deadline.")

    def legacy_generations(self) -> list[dict[str, Any]]:
        """Expose metadata-only historical outputs without upgrading their authority."""

        root = self.output_root / "generations"
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(root.glob("*/metadata.json")):
            metadata = _read_json(path)
            if metadata:
                raw_results = metadata.get("results")
                result_count = len(raw_results) if isinstance(raw_results, list) else 0
                rows.append(
                    {
                        "generation_id": f"legacy-{hashlib.sha256(path.parent.name.encode('utf-8')).hexdigest()[:16]}",
                        "status": "NOT_COMPARABLE",
                        "comparability": "NOT_COMPARABLE",
                        "durable": False,
                        "scope": EXPLORATORY_SCOPE,
                        "reason": "Legacy metadata has no atomic ProductRun state or canonical ProductEvents.",
                        "result_count": result_count,
                        "results": [],
                        "frozen_benchmark_eligible": False,
                        "promotion_evidence_eligible": False,
                    }
                )
        return rows

    def _adapter_identity(self) -> dict[str, Any]:
        assert self.generator is not None
        adapter_type = type(self.generator)
        remote = getattr(self.generator, "remote", False)
        billable = getattr(self.generator, "billable", False)
        if type(remote) is not bool or type(billable) is not bool:
            raise GenerationSafetyError("The generation adapter boolean capabilities are malformed.")
        identity = {
            "adapter": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
            "remote": remote,
            "billable": billable,
        }
        code_identity = getattr(self.generator, "code_identity_sha256", None)
        if code_identity is not None and not _is_sha256(code_identity):
            raise GenerationSafetyError("The generation adapter code identity is malformed.")
        if code_identity is not None:
            identity["code_identity_sha256"] = code_identity
        identity["sha256"] = hashlib.sha256(
            strict_json_bytes(identity, sort_keys=True, separators=(",", ":"))
        ).hexdigest()
        return _validate_adapter_identity(identity)

    def _record_terminal(
        self,
        run_id: str,
        *,
        terminal_status: str,
        reason: str,
    ) -> bool:
        definitions = {
            "FAILED": (
                "failed",
                "failed",
                "Exploratory Playground generation failed.",
            ),
            "CANCELLED": (
                "cancelled",
                "cancelled",
                "Exploratory Playground generation was cancelled.",
            ),
            "TIMED_OUT": (
                "timed_out",
                "timed_out",
                "Exploratory Playground generation reached its fixed deadline.",
            ),
        }
        if terminal_status not in definitions:
            raise GenerationSafetyError("The Playground terminal transition is unsupported.")
        event_type, stage, message = definitions[terminal_status]
        for _attempt in range(4):
            base_cache = self._authoritative_cache(run_id)
            if base_cache.get("terminal_status") is not None:
                return False
            ended_at = _utc_now()
            terminal_cache = {
                **base_cache,
                "runtime_identity": None,
                "report_identity": None,
                "failure": None,
                "cancellation": None,
                "timeout": None,
                "terminal_status": terminal_status,
                "ended_at": ended_at,
            }
            if terminal_status == "FAILED":
                terminal_cache["failure"] = {
                    "type": "GenerationAdapterError",
                    "message": "Generation adapter failed.",
                    "timestamp": ended_at,
                }
            elif terminal_status == "CANCELLED":
                terminal_cache["cancellation"] = {
                    "reason": reason,
                    "timestamp": ended_at,
                }
            else:
                terminal_cache["timeout"] = {
                    "reason": reason,
                    "timestamp": ended_at,
                    "deadline_at": base_cache["deadline_at"],
                }
            terminal_cache = _cache_from_state(terminal_cache)
            terminal = _terminal_record(terminal_status, terminal_cache)

            def terminal_transform(
                cache: dict[str, Any],
                *,
                expected_cache: dict[str, Any] = base_cache,
                next_cache: dict[str, Any] = terminal_cache,
            ) -> dict[str, Any]:
                if cache != expected_cache:
                    raise GenerationSafetyError("The Playground evidence changed before terminal publication.")
                return _json_copy(next_cache)

            try:
                return self._commit_lifecycle_event(
                    run_id,
                    timestamp=ended_at,
                    stage=stage,
                    event_type=event_type,
                    status=ProductStatus.FAILED,
                    current=int(base_cache["progress"]["current"]),
                    total=int(base_cache["progress"]["total"]),
                    message=message,
                    cache_transform=terminal_transform,
                    metrics_factory=lambda _prior, _next, terminal_value=terminal: {
                        "terminal": terminal_value,
                        "exploratory": True,
                    },
                    artifacts=tuple(item["reference"] for item in base_cache["artifact_identities"]),
                )
            except GenerationSafetyError:
                state = self.repository.state(run_id)
                if state.get("terminal_status") is not None:
                    return False
        raise GenerationSafetyError("The Playground terminal CAS could not obtain a stable preimage.")

    def _record_failure(self, run_id: str, _error: Exception) -> bool:
        try:
            return self._record_terminal(
                run_id,
                terminal_status="FAILED",
                reason="Generation adapter failed.",
            )
        except (GenerationSafetyError, OSError, ValueError, KeyError):
            return False

    def _record_cancellation(self, run_id: str, reason: str) -> bool:
        return self._record_terminal(
            run_id,
            terminal_status="CANCELLED",
            reason=reason,
        )

    def _record_timeout(self, run_id: str, reason: str) -> bool:
        return self._record_terminal(
            run_id,
            terminal_status="TIMED_OUT",
            reason=reason,
        )

    def rerun(
        self,
        preset_name: str,
        *,
        explicit_action: bool,
        confirm_billable: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        request = self.presets.load(preset_name)
        if seed is not None:
            request = replace(request, seed=seed)
        return self.generate(request, explicit_action=explicit_action, confirm_billable=confirm_billable)


def _new_run_id() -> str:
    return f"playground-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_deadline(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return strict_json_bytes(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + b"\n"


def _verify_anchored_publication(
    parent: AnchoredDirectory,
    name: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Verify immutable publication bytes through one held parent handle."""

    content = _read_held_regular_file(parent, name, maximum_size=expected_size)
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise GenerationSafetyError("A Playground publication target already contains different bytes.")


def _read_held_regular_file(
    parent: AnchoredDirectory,
    name: str,
    *,
    maximum_size: int,
) -> bytes:
    """Consume one single-link regular file from a stable held descriptor."""

    descriptor = parent.open_file_immovable(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        identity = OwnedFileIdentity.from_stat(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_size
        ):
            raise GenerationSafetyError("A Playground held file is irregular or exceeds its byte limit.")
        content = handle.read(maximum_size + 1)
        after = os.fstat(handle.fileno())
        if (
            len(content) > maximum_size
            or len(content) != before.st_size
            or OwnedFileIdentity.from_stat(after) != identity
            or after.st_size != before.st_size
        ):
            raise GenerationSafetyError("A Playground held file changed during consumption.")
        if not identity.matches(parent.lstat(name)):
            raise GenerationSafetyError("A Playground held file name changed during consumption.")
    return content


def _write_held_file(descriptor: int, content: bytes) -> OwnedFileIdentity:
    """Write and verify bytes while retaining the original publication descriptor."""

    before = os.fstat(descriptor)
    identity = OwnedFileIdentity.from_stat(before)
    if not stat.S_ISREG(before.st_mode):
        raise GenerationSafetyError("A Playground publication staging object is irregular.")
    writer_descriptor = os.dup(descriptor)
    try:
        with os.fdopen(writer_descriptor, "wb") as handle:
            writer_descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
            if OwnedFileIdentity.from_stat(after) != identity or after.st_size != len(content):
                raise GenerationSafetyError("A Playground publication changed while writing.")
    finally:
        if writer_descriptor >= 0:
            os.close(writer_descriptor)
    return identity


def _publish_exact_bytes(run_directory: Path, reference: str, content: bytes) -> None:
    """Publish immutable bytes exclusively below one anchored Playground run."""

    relative = _safe_reference(reference)
    if relative is None or len(relative.parts) < 2 or any(part in {"", "."} for part in relative.parts):
        raise GenerationSafetyError("A Playground publication reference is unsafe.")
    parent_path = run_directory.joinpath(*relative.parts[:-1])
    name = relative.parts[-1]
    expected_sha256 = hashlib.sha256(content).hexdigest()
    try:
        with open_anchored_directory(parent_path, run_directory) as parent:
            if parent.lexists(name):
                _verify_anchored_publication(
                    parent,
                    name,
                    expected_size=len(content),
                    expected_sha256=expected_sha256,
                )
                return

            temporary: str | None = None
            direct_final = False
            descriptor = -1
            identity: OwnedFileIdentity | None = None
            try:
                if os.name == "nt":
                    temporary = f".spritelab-playground-{uuid.uuid4().hex}.tmp"
                    descriptor = parent.open_file(
                        temporary,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                    )
                else:
                    try:
                        descriptor = parent.open_anonymous_file()
                    except (OSError, UnsafeFilesystemOperation):
                        descriptor = parent.open_file(
                            name,
                            os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                        )
                        direct_final = True
                identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
                if _write_held_file(descriptor, content) != identity:
                    raise GenerationSafetyError("A Playground publication staging identity changed.")

                if direct_final:
                    if not identity.matches(parent.lstat(name)):
                        raise GenerationSafetyError("A Playground direct publication changed while writing.")
                elif temporary is not None and not identity.matches(parent.lstat(temporary)):
                    raise GenerationSafetyError("A Playground publication temporary changed before publication.")

                if not direct_final:
                    try:
                        parent.publish_held_file_no_replace(
                            descriptor,
                            temporary,
                            name,
                            identity=identity,
                        )
                    except ExactPublicationUnsupported:
                        if os.name == "nt":
                            raise
                        os.close(descriptor)
                        descriptor = -1
                        try:
                            descriptor = parent.open_file(
                                name,
                                os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
                            )
                        except FileExistsError:
                            _verify_anchored_publication(
                                parent,
                                name,
                                expected_size=len(content),
                                expected_sha256=expected_sha256,
                            )
                        else:
                            direct_final = True
                            identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
                            if _write_held_file(descriptor, content) != identity:
                                raise GenerationSafetyError("A Playground publication staging identity changed.")
                            if not identity.matches(parent.lstat(name)):
                                raise GenerationSafetyError(
                                    "A Playground direct publication changed while writing."
                                ) from None
                    except FileExistsError:
                        _verify_anchored_publication(
                            parent,
                            name,
                            expected_size=len(content),
                            expected_sha256=expected_sha256,
                        )

                _verify_anchored_publication(
                    parent,
                    name,
                    expected_size=len(content),
                    expected_sha256=expected_sha256,
                )
            finally:
                if temporary is not None and identity is not None:
                    parent.unlink_if_owned(temporary, identity)
                if descriptor >= 0:
                    os.close(descriptor)
    except GenerationSafetyError:
        raise
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise GenerationSafetyError("A Playground publication path could not be anchored safely.") from exc


def _read_authorized_command_snapshot(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    """Read command authorization bytes and their digest from one held inode."""

    try:
        with open_anchored_directory(path.parent, root) as parent:
            content = _read_held_regular_file(parent, path.name, maximum_size=1_000_000)
        value = strict_json_loads(content)
    except GenerationSafetyError:
        raise
    except (OSError, UnicodeError, ValueError, UnsafeFilesystemOperation) as exc:
        raise GenerationSafetyError("The Playground command snapshot could not be held safely.") from exc
    if not isinstance(value, dict):
        raise GenerationSafetyError("The Playground command snapshot is malformed.")
    return value, hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 1_000_000:
        return {}
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_reference(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    reference = Path(value)
    return None if reference.is_absolute() or ".." in reference.parts else reference


def _product_run_dict(run: ProductRun, *, status: str | None = None) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "feature": run.feature,
        "action_id": run.action_id,
        "status": status or run.status.value,
        "backend_id": run.backend_id,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "artifact_references": list(run.artifact_references),
    }


def _initialization_incomplete(run_id: str, directory: Path) -> dict[str, Any]:
    present = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    return {
        "schema_version": PLAYGROUND_SCHEMA,
        "durable": False,
        "authoritative": False,
        "resumable": False,
        "run_id": run_id,
        "generation_id": run_id,
        "status": "initialization_incomplete",
        "stage": "initialization_incomplete",
        "comparability": "NOT_COMPARABLE",
        "scope": EXPLORATORY_SCOPE,
        "exploratory": True,
        "benchmark_eligible": False,
        "promotion_evidence_eligible": False,
        "results": [],
        "progress": {"current": 0, "total": None},
        "report_available": False,
        "present_skeleton_entries": present,
        "integrity_reasons": ["Playground initialization stopped before authoritative state publication."],
        "excluded_from_frozen_benchmark": True,
        "excluded_from_promotion_evidence": True,
        "frozen_benchmark_eligible": False,
    }


def _not_comparable(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": PLAYGROUND_SCHEMA,
        "durable": False,
        "run_id": run_id,
        "generation_id": run_id,
        "status": "NOT_COMPARABLE",
        "comparability": "NOT_COMPARABLE",
        "scope": EXPLORATORY_SCOPE,
        "results": [],
        "progress": {"current": 0, "total": None},
        "report_available": False,
        "integrity_reasons": [reason],
        "excluded_from_frozen_benchmark": True,
        "excluded_from_promotion_evidence": True,
        "frozen_benchmark_eligible": False,
        "promotion_evidence_eligible": False,
    }

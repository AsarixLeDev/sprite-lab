"""Explicit, non-benchmark exploratory generation playground."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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
from spritelab.product_features.evaluation.models import CheckpointCatalog
from spritelab.product_web.events import EventRepository
from spritelab.v3.run_state import lock_file

PLAYGROUND_SCHEMA = "spritelab.product.playground-generation.v1"
PLAYGROUND_RUN_SCHEMA = "spritelab.product.playground-run.v1"
PLAYGROUND_COMMAND_SCHEMA = "spritelab.product.playground-command.v1"
EXPLORATORY_SCOPE = "EXPLORATORY"
DEFAULT_SEED = 42
DEFAULT_SAMPLING_STEPS = 30
DEFAULT_GUIDANCE = 3.0
DEFAULT_IMAGE_COUNT = 4


class GenerationSafetyError(ValueError):
    """An explicit generation or billing safety precondition was not met."""


class GeneratorUnavailableError(RuntimeError):
    """No typed generator was supplied to the playground."""


class GenerationCancelledError(RuntimeError):
    """A typed generator reported explicit cancellation."""


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
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if type(self.sampling_steps) is not int or not 1 <= self.sampling_steps <= 500:
            raise ValueError("sampling_steps must be between 1 and 500.")
        if isinstance(self.guidance, bool) or not isinstance(self.guidance, (int, float)):
            raise ValueError("guidance must be a finite number between 0 and 50.")
        if not math.isfinite(float(self.guidance)) or not 0.0 <= self.guidance <= 50.0:
            raise ValueError("guidance must be between 0 and 50.")
        if type(self.image_count) is not int or not 1 <= self.image_count <= 16:
            raise ValueError("image_count must be between 1 and 16.")

    @classmethod
    def defaults(cls, checkpoint_id: str = "") -> dict[str, Any]:
        return asdict(cls(prompt="Describe a 32x32 sprite", checkpoint_id=checkpoint_id))


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
    ) -> None:
        self.catalog = catalog
        self.catalog_provider = catalog_provider
        self.output_root = output_root.resolve()
        self.generator = generator
        self.run_id_factory = run_id_factory or _new_run_id
        self.initialization_hook = initialization_hook
        self.presets = PromptPresetStore(self.output_root / "presets.json")
        self.repository = EventRepository(
            (runs_directory or self.output_root).resolve(),
            private_roots=(self.output_root.parent,),
        )
        self.startup_incomplete_run_ids = tuple(item["run_id"] for item in self.incomplete_initializations())

    def defaults(self) -> dict[str, Any]:
        return GenerationRequest.defaults(self.catalog.default_checkpoint_id or "")

    @property
    def confirmation_required(self) -> bool:
        return bool(
            self.generator and (getattr(self.generator, "remote", False) or getattr(self.generator, "billable", False))
        )

    def generate(
        self,
        request: GenerationRequest,
        *,
        explicit_action: bool,
        confirm_billable: bool = False,
        _run_id: str | None = None,
        _initialize: bool = True,
    ) -> dict[str, Any]:
        if not explicit_action:
            raise GenerationSafetyError("Generation requires an explicit Generate action.")
        if self.generator is None:
            raise GeneratorUnavailableError("No typed generation adapter is configured.")
        if getattr(self.generator, "requires_fresh_catalog", False) and self.catalog_provider is None:
            raise GenerationSafetyError("The local generation adapter requires a fresh checkpoint catalog provider.")
        if self.confirmation_required and not confirm_billable:
            raise GenerationSafetyError("Remote or billable generation requires explicit cost confirmation.")
        current_catalog = self.catalog_provider() if self.catalog_provider is not None else self.catalog
        self.catalog = current_catalog
        checkpoint = current_catalog.find(request.checkpoint_id, weights=request.weights)
        if (
            checkpoint is None
            or checkpoint.path is None
            or checkpoint.checkpoint_sha256 is None
            or checkpoint.checkpoint_step is None
        ):
            raise GenerationSafetyError("The selected checkpoint and live/EMA variant is not eligible.")
        started_at = _utc_now()
        run_id = _run_id or self.run_id_factory()
        request_payload = asdict(request)
        checkpoint_identity = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_run_id": checkpoint.run_id,
            "checkpoint_step": checkpoint.checkpoint_step,
            "weights": request.weights,
            "sha256": checkpoint.checkpoint_sha256,
        }
        adapter_identity = self._adapter_identity()
        command = {
            "schema_version": PLAYGROUND_COMMAND_SCHEMA,
            "action": "playground.generate",
            "run_id": run_id,
            "request": request_payload,
            "checkpoint_identity": checkpoint_identity,
            "generation_adapter_identity": adapter_identity,
            "scope": EXPLORATORY_SCOPE,
            "benchmark_eligible": False,
            "promotion_evidence_eligible": False,
        }
        command_bytes = _json_bytes(command)
        command_sha256 = hashlib.sha256(command_bytes).hexdigest()
        generation_parameters = {
            "sampling_steps": request.sampling_steps,
            "guidance": request.guidance,
            "image_count": request.image_count,
        }
        state_extra = {
            "playground_schema_version": PLAYGROUND_RUN_SCHEMA,
            "prompt": request.prompt,
            "request": request_payload,
            "checkpoint_identity": checkpoint_identity,
            "weights": request.weights,
            "seed": request.seed,
            "generation_parameters": generation_parameters,
            "generation_adapter_identity": adapter_identity,
            "command_reference": "command.json",
            "command_sha256": command_sha256,
            "results": [],
            "artifact_identities": [],
            "output_hashes": [],
            "progress": {"current": 0, "total": request.image_count},
            "exploratory": True,
            "benchmark_eligible": False,
            "promotion_evidence_eligible": False,
            "exploratory_classification": EXPLORATORY_SCOPE,
            "benchmark_eligibility": False,
            "promotion_evidence_eligibility": False,
            "failure": None,
            "cancellation": None,
            "report_identity": None,
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
                "exploratory": True,
                "benchmark_eligible": False,
                "promotion_evidence_eligible": False,
            },
        )
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
            state = self.repository.state(run_id)
            replay = self.repository.replay(run_id)
            event_types = [item.event.event_type for item in replay.events]
            continuation_directory = self.repository.run_directory(run_id)
            if (
                not state
                or continuation_directory is None
                or not (continuation_directory / "command.json").is_file()
                or _file_sha256(continuation_directory / "command.json") != command_sha256
                or any(not (continuation_directory / name).is_dir() for name in ("logs", "artifacts", "report"))
                or state.get("stage") != "planned"
                or state.get("status") != ProductStatus.RUNNING.value
                or state.get("command_sha256") != command_sha256
                or state.get("request") != request_payload
                or event_types != ["planned"]
                or not replay.safe_for_resume
            ):
                raise GenerationSafetyError("The planned Playground run is not safe for explicit continuation.")
            started_at = str(state.get("started_at") or started_at)
        directory = self.repository.run_directory(run_id)
        if directory is None:
            raise OSError("Playground runs directory is unavailable.")
        results: list[dict[str, Any]] = []
        artifact_identities: list[dict[str, Any]] = []
        try:
            self._append(
                run_id,
                stage="validated",
                event_type="validated",
                status=ProductStatus.RUNNING,
                current=0,
                total=request.image_count,
                message="Checkpoint, parameters, and adapter identity validated.",
                metrics={"generation_adapter_identity": adapter_identity},
            )
            self.repository.update_state(run_id, stage="generation_started")
            self._append(
                run_id,
                stage="generation_started",
                event_type="generation_started",
                status=ProductStatus.RUNNING,
                current=0,
                total=request.image_count,
                message="Exploratory generation started after explicit user action.",
            )
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
            self._assert_active(run_id)
            for index, raw in enumerate(raw_assets):
                with lock_file(directory / ".playground-lifecycle.lock"):
                    self._assert_active(run_id)
                    if index >= request.image_count:
                        raise RuntimeError("Generator returned more images than requested.")
                    content, media_type = _asset_bytes(raw)
                    digest = hashlib.sha256(content).hexdigest()
                    extension = ".png" if media_type == "image/png" else ".bin"
                    reference = f"artifacts/image_{index:03d}{extension}"
                    _atomic_bytes(directory / reference, content)
                    completed_at = _utc_now()
                    result = {
                        "result_id": f"{run_id}-{index:03d}",
                        "checkpoint_identity": checkpoint.checkpoint_id,
                        "checkpoint_run_id": checkpoint.run_id,
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
                    }
                    identity = {
                        "artifact_id": result["result_id"],
                        "reference": reference,
                        "sha256": digest,
                        "size_bytes": len(content),
                        "media_type": media_type,
                    }
                    results.append(result)
                    artifact_identities.append(identity)
                    self.repository.update_state(
                        run_id,
                        stage="image_completed",
                        results=results,
                        artifact_identities=artifact_identities,
                        output_hashes=[item["sha256"] for item in artifact_identities],
                        progress={"current": len(results), "total": request.image_count},
                    )
                    self._append(
                        run_id,
                        stage="image_completed",
                        event_type="image_completed",
                        status=ProductStatus.RUNNING,
                        current=len(results),
                        total=request.image_count,
                        message=f"Exploratory image {index + 1} of {request.image_count} completed.",
                        metrics={"result": result, "artifact_identity": identity, "exploratory": True},
                        artifacts=(reference,),
                    )
            if len(results) != request.image_count:
                raise RuntimeError("Generator returned a different number of images than requested.")
            refreshed_catalog = self.catalog_provider() if self.catalog_provider is not None else self.catalog
            self.catalog = refreshed_catalog
            refreshed = refreshed_catalog.find(request.checkpoint_id, weights=request.weights)
            if (
                refreshed is None
                or refreshed.path != checkpoint.path
                or refreshed.checkpoint_sha256 != checkpoint.checkpoint_sha256
                or refreshed.checkpoint_step != checkpoint.checkpoint_step
                or refreshed.weights != checkpoint.weights
            ):
                raise GenerationSafetyError("Checkpoint eligibility changed before generation completion.")
            final_adapter_identity = self._adapter_identity()
            if final_adapter_identity != adapter_identity:
                raise GenerationSafetyError("Generation adapter code identity changed while sampling.")
            runtime_identity = getattr(self.generator, "last_runtime_identity", None)
            if not isinstance(runtime_identity, Mapping):
                runtime_identity = {
                    "schema_version": "spritelab.playground-runtime-identity.v1",
                    "runtime_reported": False,
                }
            else:
                runtime_identity = dict(runtime_identity)
            ended_at = _utc_now()
            product_run = ProductRun(
                run_id=run_id,
                feature="playground",
                action_id="generate",
                status=ProductStatus.COMPLETE,
                backend_id=str(adapter_identity["adapter"]),
                started_at=started_at,
                ended_at=ended_at,
                artifact_references=tuple(item["reference"] for item in artifact_identities),
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
                "results": results,
                "artifact_identities": artifact_identities,
                "excluded_from_frozen_benchmark": True,
                "excluded_from_promotion_evidence": True,
            }
            with lock_file(directory / ".playground-lifecycle.lock"):
                self._assert_active(run_id)
                report_path = directory / "report" / "report.json"
                report_bytes = _json_bytes(report)
                _atomic_bytes(report_path, report_bytes)
                report_identity = {
                    "reference": "report/report.json",
                    "sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "size_bytes": len(report_bytes),
                }
                self.repository.update_state(
                    run_id,
                    status=ProductStatus.COMPLETE.value,
                    stage="generation_completed",
                    ended_at=ended_at,
                    results=results,
                    artifact_identities=artifact_identities,
                    output_hashes=[item["sha256"] for item in artifact_identities],
                    progress={"current": len(results), "total": request.image_count},
                    report_identity=report_identity,
                    runtime_identity=runtime_identity,
                )
                self._append(
                    run_id,
                    stage="generation_completed",
                    event_type="generation_completed",
                    status=ProductStatus.COMPLETE,
                    current=len(results),
                    total=request.image_count,
                    message="Exploratory Playground generation completed durably.",
                    metrics={
                        "output_hashes": [item["sha256"] for item in artifact_identities],
                        "runtime_identity": runtime_identity,
                        "exploratory": True,
                        "benchmark_eligible": False,
                        "promotion_evidence_eligible": False,
                    },
                    artifacts=tuple(item["reference"] for item in artifact_identities),
                )
        except GenerationCancelledError as exc:
            with lock_file(directory / ".playground-lifecycle.lock"):
                if self.repository.state(run_id).get("status") != "CANCELLED":
                    self._record_cancellation(run_id, str(exc) or "Generation adapter cancelled the run.")
            raise
        except Exception as exc:
            with lock_file(directory / ".playground-lifecycle.lock"):
                self._record_failure(run_id, exc)
            raise
        return self.reconstruct(run_id)

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
        """Rebuild one run from durable state and events without invoking a generator."""

        directory = self.repository.run_directory(run_id)
        if directory is None or not directory.is_dir():
            raise KeyError(f"Unknown Playground run: {run_id}")
        state = self.repository.state(run_id)
        if not state:
            if (directory / "state.json").exists():
                return _not_comparable(run_id, "state.json is malformed or identity-inconsistent.")
            return _initialization_incomplete(run_id, directory)
        reasons: list[str] = []
        if state.get("playground_schema_version") != PLAYGROUND_RUN_SCHEMA:
            reasons.append("Playground run schema is missing or unsupported.")
        if state.get("feature") != "playground" or state.get("command") != "playground.generate":
            reasons.append("ProductRun feature or action identity is inconsistent.")
        command_path = directory / "command.json"
        command = _read_json(command_path)
        if not command:
            reasons.append("command.json is missing or malformed.")
        else:
            if _file_sha256(command_path) != state.get("command_sha256"):
                reasons.append("command.json identity changed.")
            if command.get("schema_version") != PLAYGROUND_COMMAND_SCHEMA:
                reasons.append("command.json schema is unsupported.")
            for key in ("request", "checkpoint_identity", "generation_adapter_identity"):
                if command.get(key) != state.get(key):
                    reasons.append(f"{key} metadata changed after planning.")
            if command.get("run_id") != run_id:
                reasons.append("command.json run identity changed.")
        if state.get("exploratory_classification") != EXPLORATORY_SCOPE:
            reasons.append("Exploratory classification changed.")
        if state.get("benchmark_eligibility") is not False:
            reasons.append("Playground benchmark eligibility must remain false.")
        if state.get("promotion_evidence_eligibility") is not False:
            reasons.append("Playground promotion-evidence eligibility must remain false.")
        if state.get("exploratory") is not True:
            reasons.append("Playground exploratory flag must remain true.")
        if state.get("benchmark_eligible") is not False:
            reasons.append("Playground benchmark_eligible flag must remain false.")
        if state.get("promotion_evidence_eligible") is not False:
            reasons.append("Playground promotion_evidence_eligible flag must remain false.")
        request_state = state.get("request") if isinstance(state.get("request"), Mapping) else {}
        if state.get("prompt") != request_state.get("prompt"):
            reasons.append("Prompt metadata changed after planning.")
        if state.get("weights") != request_state.get("weights"):
            reasons.append("Live/EMA metadata changed after planning.")
        if state.get("seed") != request_state.get("seed"):
            reasons.append("Seed metadata changed after planning.")
        expected_parameters = {
            "sampling_steps": request_state.get("sampling_steps"),
            "guidance": request_state.get("guidance"),
            "image_count": request_state.get("image_count"),
        }
        if state.get("generation_parameters") != expected_parameters:
            reasons.append("Generation parameter metadata changed after planning.")

        for name in ("logs", "artifacts", "report"):
            if not (directory / name).is_dir():
                reasons.append(f"Required {name}/ directory is missing.")
        event_path = directory / "events.jsonl"
        if not event_path.is_file():
            reasons.append("events.jsonl is missing from authoritative Playground state.")
        replay = self.repository.replay(run_id)
        if replay.integrity_status != "VALID" or replay.invalid_event_count:
            reasons.extend(replay.warnings or ("Playground event history is not comparable.",))
        indexed = list(replay.events)
        if not indexed or indexed[0].event_id != 1 or indexed[0].event.event_type != "planned":
            reasons.append("The authoritative planned ProductEvent is missing or out of order.")
        else:
            planned = indexed[0].event
            if (
                planned.metrics.get("command_sha256") != state.get("command_sha256")
                or planned.metrics.get("exploratory") is not True
                or planned.metrics.get("benchmark_eligible") is not False
                or planned.metrics.get("promotion_evidence_eligible") is not False
            ):
                reasons.append("The planned ProductEvent identity or exploratory flags changed.")
        unique_events: list[ProductEvent] = []
        fingerprints: set[str] = set()
        event_results: dict[str, dict[str, Any]] = {}
        for item in indexed:
            fingerprint = hashlib.sha256(
                strict_json_bytes(item.event.to_dict(), sort_keys=True, separators=(",", ":"))
            ).hexdigest()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            unique_events.append(item.event)
            result = item.event.metrics.get("result") if item.event.event_type == "image_completed" else None
            if isinstance(result, Mapping) and isinstance(result.get("result_id"), str):
                event_results.setdefault(str(result["result_id"]), dict(result))
        state_results = [dict(item) for item in state.get("results", ()) if isinstance(item, Mapping)]
        state_by_id = {str(item["result_id"]): item for item in state_results if isinstance(item.get("result_id"), str)}
        if event_results and event_results != state_by_id:
            reasons.append("Durable image event metadata does not match atomic state.")

        stale_reasons: list[str] = []
        identities = [dict(item) for item in state.get("artifact_identities", ()) if isinstance(item, Mapping)]
        identities_by_id = {
            str(item["artifact_id"]): item for item in identities if isinstance(item.get("artifact_id"), str)
        }
        if set(identities_by_id) != set(state_by_id):
            reasons.append("Atomic state result and artifact identities disagree.")
        for result_id, result in state_by_id.items():
            identity = identities_by_id.get(result_id)
            if identity and (
                result.get("output_reference") != identity.get("reference")
                or result.get("output_hash") != identity.get("sha256")
            ):
                reasons.append(f"Output metadata changed for {result_id}.")
        for identity in identities:
            reference = _safe_reference(identity.get("reference"))
            if reference is None:
                reasons.append("Artifact reference is unsafe or malformed.")
                continue
            artifact = directory / reference
            if not artifact.is_file():
                stale_reasons.append(f"Missing artifact: {reference.as_posix()}.")
                continue
            if _file_sha256(artifact) != identity.get("sha256"):
                stale_reasons.append(f"Artifact bytes changed: {reference.as_posix()}.")
        report_available = False
        report_identity = state.get("report_identity")
        if isinstance(report_identity, Mapping):
            report_reference = _safe_reference(report_identity.get("reference"))
            report_path = directory / report_reference if report_reference is not None else None
            if report_path is None or not report_path.is_file():
                stale_reasons.append("Durable Playground report is missing.")
            elif _file_sha256(report_path) != report_identity.get("sha256"):
                stale_reasons.append("Durable Playground report bytes changed.")
            else:
                report_available = True
        elif state.get("status") == ProductStatus.COMPLETE.value:
            stale_reasons.append("Completed Playground run has no report identity.")

        status = str(state.get("status") or ProductStatus.NOT_STARTED.value)
        comparability = "CURRENT"
        if reasons:
            status, comparability = "NOT_COMPARABLE", "NOT_COMPARABLE"
        elif stale_reasons:
            status, comparability = "STALE", "STALE"
        artifact_references = tuple(
            str(item.get("reference")) for item in identities if isinstance(item.get("reference"), str)
        )
        product_status = ProductStatus(status) if status in ProductStatus._value2member_map_ else ProductStatus.FAILED
        product_run = ProductRun(
            run_id=run_id,
            feature="playground",
            action_id="generate",
            status=product_status,
            backend_id=str(state.get("backend_id") or "playground"),
            started_at=str(state.get("started_at")) if state.get("started_at") else None,
            ended_at=str(state.get("ended_at")) if state.get("ended_at") else None,
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
            "stage": str(state.get("stage") or "planned"),
            "scope": EXPLORATORY_SCOPE,
            "exploratory": True,
            "benchmark_eligible": False,
            "promotion_evidence_eligible": False,
            "prompt": state.get("prompt"),
            "request": dict(state.get("request") or {}),
            "checkpoint_identity": dict(state.get("checkpoint_identity") or {}),
            "weights": state.get("weights"),
            "seed": state.get("seed"),
            "generation_parameters": dict(state.get("generation_parameters") or {}),
            "generation_adapter_identity": dict(state.get("generation_adapter_identity") or {}),
            "runtime_identity": dict(state.get("runtime_identity") or {}),
            "started_at": state.get("started_at"),
            "ended_at": state.get("ended_at"),
            "results": state_results,
            "artifact_identities": identities,
            "output_hashes": list(state.get("output_hashes") or ()),
            "progress": dict(state.get("progress") or {"current": len(state_results), "total": None}),
            "failure": state.get("failure"),
            "cancellation": state.get("cancellation"),
            "report_available": report_available,
            "event_count": len(unique_events),
            "stages": [event.stage for event in unique_events],
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

    def cancel(self, run_id: str, *, reason: str = "Cancelled by explicit user action.") -> dict[str, Any]:
        state = self.repository.state(run_id)
        if state.get("feature") != "playground":
            raise KeyError(f"Unknown Playground run: {run_id}")
        if str(state.get("status")) in {ProductStatus.COMPLETE.value, ProductStatus.FAILED.value, "CANCELLED"}:
            return self.reconstruct(run_id)
        directory = self.repository.run_directory(run_id)
        if directory is None:
            raise KeyError(f"Unknown Playground run: {run_id}")
        with lock_file(directory / ".playground-lifecycle.lock"):
            state = self.repository.state(run_id)
            if str(state.get("status")) not in {ProductStatus.COMPLETE.value, ProductStatus.FAILED.value, "CANCELLED"}:
                self._record_cancellation(run_id, reason)
        return self.reconstruct(run_id)

    def _assert_active(self, run_id: str) -> None:
        state = self.repository.state(run_id)
        if state.get("status") == "CANCELLED":
            cancellation = state.get("cancellation")
            reason = cancellation.get("reason") if isinstance(cancellation, Mapping) else None
            raise GenerationCancelledError(str(reason or "Generation was cancelled."))
        if state.get("status") != ProductStatus.RUNNING.value:
            raise GenerationSafetyError("Playground run is no longer active.")

    def legacy_generations(self) -> list[dict[str, Any]]:
        """Expose metadata-only historical outputs without upgrading their authority."""

        root = self.output_root / "generations"
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(root.glob("*/metadata.json")):
            metadata = _read_json(path)
            if metadata:
                rows.append(
                    {
                        "generation_id": metadata.get("generation_id") or path.parent.name,
                        "status": "NOT_COMPARABLE",
                        "comparability": "NOT_COMPARABLE",
                        "durable": False,
                        "scope": EXPLORATORY_SCOPE,
                        "reason": "Legacy metadata has no atomic ProductRun state or canonical ProductEvents.",
                        "results": list(metadata.get("results") or ()),
                        "frozen_benchmark_eligible": False,
                        "promotion_evidence_eligible": False,
                    }
                )
        return rows

    def _adapter_identity(self) -> dict[str, Any]:
        assert self.generator is not None
        adapter_type = type(self.generator)
        identity = {
            "adapter": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
            "remote": bool(getattr(self.generator, "remote", False)),
            "billable": bool(getattr(self.generator, "billable", False)),
        }
        code_identity = getattr(self.generator, "code_identity_sha256", None)
        if (
            isinstance(code_identity, str)
            and len(code_identity) == 64
            and code_identity == code_identity.lower()
            and all(character in "0123456789abcdef" for character in code_identity)
        ):
            identity["code_identity_sha256"] = code_identity
        identity["sha256"] = hashlib.sha256(
            strict_json_bytes(identity, sort_keys=True, separators=(",", ":"))
        ).hexdigest()
        return identity

    def _append(
        self,
        run_id: str,
        *,
        stage: str,
        event_type: str,
        status: ProductStatus,
        current: int,
        total: int | None,
        message: str,
        metrics: Mapping[str, Any] | None = None,
        artifacts: tuple[str, ...] = (),
    ) -> None:
        self.repository.append(
            ProductEvent(
                run_id=run_id,
                timestamp=_utc_now(),
                feature="playground",
                stage=stage,
                event_type=event_type,
                status=status,
                current=current,
                total=total,
                message=message,
                metrics=metrics or {},
                artifact_references=artifacts,
            )
        )

    def _record_failure(self, run_id: str, error: Exception) -> None:
        state = self.repository.state(run_id)
        if state.get("status") in {"CANCELLED", ProductStatus.COMPLETE.value}:
            return
        ended_at = _utc_now()
        failure = {
            "type": type(error).__name__,
            "message": "Generation adapter failed.",
            "timestamp": ended_at,
        }
        try:
            self._append(
                run_id,
                stage="failed",
                event_type="failed",
                status=ProductStatus.FAILED,
                current=int(dict(state.get("progress") or {}).get("current", 0)),
                total=int(dict(state.get("progress") or {}).get("total", 0)) or None,
                message="Exploratory Playground generation failed.",
                metrics={"failure": failure, "exploratory": True},
            )
            self.repository.update_state(
                run_id,
                status=ProductStatus.FAILED.value,
                stage="failed",
                ended_at=ended_at,
                failure=failure,
            )
        except (OSError, ValueError, FileNotFoundError):
            pass

    def _record_cancellation(self, run_id: str, reason: str) -> None:
        state = self.repository.state(run_id)
        if state.get("status") in {"CANCELLED", ProductStatus.COMPLETE.value, ProductStatus.FAILED.value}:
            return
        ended_at = _utc_now()
        cancellation = {"reason": reason, "timestamp": ended_at}
        self._append(
            run_id,
            stage="cancelled",
            event_type="cancelled",
            status=ProductStatus.FAILED,
            current=int(dict(state.get("progress") or {}).get("current", 0)),
            total=int(dict(state.get("progress") or {}).get("total", 0)) or None,
            message="Exploratory Playground generation was cancelled.",
            metrics={"cancellation": cancellation, "exploratory": True},
        )
        self.repository.update_state(
            run_id,
            status="CANCELLED",
            stage="cancelled",
            ended_at=ended_at,
            cancellation=cancellation,
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


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return strict_json_bytes(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + b"\n"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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

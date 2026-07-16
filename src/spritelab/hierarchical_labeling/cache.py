"""Complete artifact identities and durable item-level labeling resume state."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.cache import CacheIdentity, LabelV4Cache, sha256_hex
from spritelab.harvest.label_v4.pixel_evidence import PIXEL_EVIDENCE_VERSION
from spritelab.hierarchical_labeling.contracts import ContextEvidence, FeatureValue, TechnicalVisualEvidence
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    canonical_json,
    content_identity,
    require_text,
    strict_json_value,
)
from spritelab.hierarchical_labeling.technical import TECHNICAL_EVIDENCE_VERSION
from spritelab.v3.run_state import atomic_write_json, lock_file, utc_now

LABELING_CACHE_SCHEMA = "spritelab.labeling.complete-cache-identity.v2"
LABELING_RUN_SCHEMA = "spritelab.labeling.durable-run.v1"
NO_CONTEXT_IDENTITY = content_identity(
    "spritelab-labeling-no-context-v1",
    {"context": None},
)
TECHNICAL_SCHEMA_IDENTITY = content_identity(
    "spritelab-labeling-technical-schema-binding-v1",
    {
        "technical_evidence_schema": TechnicalVisualEvidence.SCHEMA_VERSION,
        "feature_schema": FeatureValue.SCHEMA_VERSION,
        "technical_extractor_version": TECHNICAL_EVIDENCE_VERSION,
        "pixel_evidence_version": PIXEL_EVIDENCE_VERSION,
    },
)


@dataclass(frozen=True)
class LabelingCacheIdentity:
    artifact_stage: str
    source_image_identity: str
    decoded_rgba_identity: str
    render_bundle_identity: str
    provider_identity: str
    model_identity: str
    prompt_identity: str
    taxonomy_identity: str
    description_schema: str
    hypothesis_schema: str
    embedding_identity: str
    retrieval_index_identity: str
    reference_set_identity: str
    decision_policy_identity: str
    calibration_identity: str
    metadata_identity: str
    provider_configuration_identity: str
    reviewed_truth_identity: str
    context_identity: str
    technical_evidence_identity: str
    technical_extraction_identity: str
    technical_schema_identity: str

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            require_text(value, name.replace("_", " "))

    def to_dict(self) -> dict[str, str]:
        return {"schema_version": LABELING_CACHE_SCHEMA, **self.__dict__}

    @classmethod
    def from_evidence(
        cls,
        *,
        technical: TechnicalVisualEvidence,
        context: ContextEvidence | None,
        **dimensions: str,
    ) -> LabelingCacheIdentity:
        bound = {
            "context_identity": context.identity if context is not None else NO_CONTEXT_IDENTITY,
            "technical_evidence_identity": technical.identity,
            "technical_extraction_identity": technical.extraction_identity,
            "technical_schema_identity": TECHNICAL_SCHEMA_IDENTITY,
        }
        repeated = set(bound) & set(dimensions)
        if repeated:
            raise ContractValidationError(
                f"evidence-bound cache dimensions cannot be supplied manually: {', '.join(sorted(repeated))}"
            )
        return cls(**dimensions, **bound)

    @property
    def identity(self) -> str:
        return content_identity(LABELING_CACHE_SCHEMA, self.to_dict())

    def legacy_cache_identity(self) -> CacheIdentity:
        """Adapt to the existing atomic cache while retaining every dimension."""

        complete = canonical_json(self.to_dict())
        return CacheIdentity(
            namespace="hierarchical_labeling_v2",
            stage=self.artifact_stage,
            image_hash=_as_sha256(self.decoded_rgba_identity),
            model_identity=f"{self.provider_identity}:{self.model_identity}",
            provider=self.provider_identity,
            prompt_version=self.prompt_identity,
            prompt_hash=_as_sha256(self.prompt_identity),
            schema_version=f"{self.description_schema}|{self.hypothesis_schema}",
            request_hash=sha256_hex(complete),
        )


def _as_sha256(value: str) -> str:
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return sha256_hex(value)


class HierarchicalLabelingCache:
    def __init__(self, root: str | Path):
        self._cache = LabelV4Cache(root)

    def get(self, identity: LabelingCacheIdentity) -> Any | None:
        envelope = self._cache.get(identity.legacy_cache_identity())
        if envelope is None:
            return None
        if not isinstance(envelope, Mapping) or envelope.get("complete_identity") != identity.to_dict():
            raise ContractValidationError("hierarchical cache entry does not match its complete identity")
        return envelope.get("artifact")

    def put(self, identity: LabelingCacheIdentity, artifact: Any) -> Any:
        if artifact is None:
            raise ContractValidationError("a failed or absent artifact cannot be cached as a success")
        strict_json_value(artifact)
        envelope = {"complete_identity": identity.to_dict(), "artifact": artifact}
        stored = self._cache.put(identity.legacy_cache_identity(), envelope)
        return stored["artifact"]


class LabelingRunStore:
    """Durable state that checkpoints each completed item independently."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.state_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"
        self.command_path = self.directory / "command.json"
        self.items_path = self.directory / "artifacts" / "item_state.json"
        self.log_path = self.directory / "logs" / "run.log"
        self.cancel_path = self.directory / "cancel.requested"

    @classmethod
    def open_or_create(
        cls,
        directory: str | Path,
        *,
        run_identity: str,
        command: Sequence[str],
        architecture_identity: str,
    ) -> LabelingRunStore:
        for name, value in (("run identity", run_identity), ("architecture identity", architecture_identity)):
            require_text(value, name)
        if not command or any(not isinstance(item, str) or not item.strip() for item in command):
            raise ContractValidationError("durable labeling command must be non-empty text tokens")
        instance = cls(directory)
        if instance.state_path.is_file():
            state = instance.read_state()
            if state.get("run_identity") != run_identity or state.get("architecture_identity") != architecture_identity:
                raise ContractValidationError("durable labeling run identity changed; stale state cannot be resumed")
            persisted = json.loads(instance.command_path.read_text(encoding="utf-8"))
            if persisted.get("argv") != list(command):
                raise ContractValidationError("durable labeling command changed; stale state cannot be resumed")
            return instance
        for child in ("logs", "artifacts", "report"):
            (instance.directory / child).mkdir(parents=True, exist_ok=True)
        now = utc_now()
        atomic_write_json(
            instance.command_path,
            {
                "schema_version": "spritelab.labeling.command.v1",
                "run_identity": run_identity,
                "architecture_identity": architecture_identity,
                "argv": list(command),
                "created_at": now,
            },
        )
        atomic_write_json(
            instance.state_path,
            {
                "schema_version": LABELING_RUN_SCHEMA,
                "run_identity": run_identity,
                "architecture_identity": architecture_identity,
                "status": "running",
                "stage": "created",
                "resumable": True,
                "created_at": now,
                "updated_at": now,
            },
        )
        atomic_write_json(instance.items_path, {"schema_version": "spritelab.labeling.item-state.v1", "items": {}})
        instance.events_path.touch(exist_ok=False)
        instance.log_path.touch(exist_ok=False)
        instance.append_event("run_created", stage="created", details={})
        return instance

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_state(self, *, status: str, stage: str, message: str | None = None) -> None:
        if status not in {"running", "partial", "completed", "cancelled", "failed"}:
            raise ContractValidationError("durable labeling state status is not controlled")
        with lock_file(self.directory / ".state.lock"):
            state = self.read_state()
            state.update({"status": status, "stage": stage, "message": message, "updated_at": utc_now()})
            atomic_write_json(self.state_path, state)

    def mark_item(
        self,
        *,
        stage: str,
        record_identity: str,
        status: str,
        artifact_identity: str | None = None,
        error_code: str | None = None,
    ) -> None:
        require_text(stage, "item stage")
        require_text(record_identity, "item record identity")
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ContractValidationError("durable labeling item status is not controlled")
        if status == "succeeded" and not artifact_identity:
            raise ContractValidationError("successful durable item requires an artifact identity")
        if status == "failed" and not error_code:
            raise ContractValidationError("failed durable item requires an error code")
        key = content_identity("spritelab-labeling-item-state-key-v1", {"stage": stage, "record": record_identity})
        with lock_file(self.directory / ".items.lock"):
            payload = json.loads(self.items_path.read_text(encoding="utf-8"))
            prior = payload["items"].get(key)
            if prior and prior.get("status") == "succeeded" and status != "succeeded":
                raise ContractValidationError("a successful durable item cannot be downgraded during resume")
            payload["items"][key] = {
                "stage": stage,
                "record_identity": record_identity,
                "status": status,
                "artifact_identity": artifact_identity,
                "error_code": error_code,
                "updated_at": utc_now(),
            }
            atomic_write_json(self.items_path, payload)
        self.append_event("item_checkpointed", stage=stage, details=payload["items"][key])

    def successful_items(self, stage: str) -> dict[str, str]:
        payload = json.loads(self.items_path.read_text(encoding="utf-8"))
        return {
            item["record_identity"]: item["artifact_identity"]
            for item in payload["items"].values()
            if item["stage"] == stage and item["status"] == "succeeded"
        }

    def pending_items(self, stage: str, record_identities: Sequence[str]) -> tuple[str, ...]:
        completed = self.successful_items(stage)
        return tuple(record for record in record_identities if record not in completed)

    def request_cancel(self) -> None:
        if self.cancel_path.exists():
            return
        with self.cancel_path.open("x", encoding="utf-8") as handle:
            handle.write(utc_now() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.append_event("cancellation_requested", stage=self.read_state()["stage"], details={})

    @property
    def cancellation_requested(self) -> bool:
        return self.cancel_path.is_file()

    def append_event(self, event_type: str, *, stage: str, details: Mapping[str, Any]) -> None:
        require_text(event_type, "event type")
        require_text(stage, "event stage")
        strict_json_value(dict(details))
        with lock_file(self.directory / ".events.lock"):
            event = {
                "schema_version": "spritelab.labeling.run-event.v1",
                "event_type": event_type,
                "stage": stage,
                "timestamp": utc_now(),
                "details": dict(details),
            }
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

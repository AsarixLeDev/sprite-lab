"""Stable runtime contracts for vision-provider adapters and orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ProviderKind(str, Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    MOCK = "mock"
    PLUGIN = "plugin"
    UNAVAILABLE = "unavailable"


class PrivacyClass(str, Enum):
    LOCAL = "local"
    HOSTED = "hosted"


class PrivacyPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    ALLOW_HOSTED = "allow_hosted"
    HOSTED_ONLY = "hosted_only"
    ASK_BEFORE_HOSTED = "ask_before_hosted"


class ProviderMode(str, Enum):
    AUTOMATIC = "automatic"
    LOCAL = "local"
    HOSTED = "hosted"
    SPECIFIC = "specific"
    CUSTOM_PLUGIN = "custom_plugin"


class DiscoveryState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MISCONFIGURED = "misconfigured"
    AUTHENTICATION_REQUIRED = "authentication_required"
    UNSUPPORTED_MODEL = "unsupported_model"


class LabelState(str, Enum):
    LABELED = "labeled"
    ABSTAINED = "abstained"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class VisionCapabilities:
    vision: bool
    structured_output: bool
    multiple_images: bool
    batching: bool
    maximum_image_count: int
    maximum_payload_size: int
    timeout_support: bool
    cancellation_support: bool
    local_or_hosted: str

    def __post_init__(self) -> None:
        if self.maximum_image_count < 1:
            raise ValueError("maximum_image_count must be positive")
        if self.maximum_payload_size < 1:
            raise ValueError("maximum_payload_size must be positive")
        if self.local_or_hosted not in {"local", "hosted"}:
            raise ValueError("local_or_hosted must be local or hosted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision": self.vision,
            "structured_output": self.structured_output,
            "multiple_images": self.multiple_images,
            "batching": self.batching,
            "maximum_image_count": self.maximum_image_count,
            "maximum_payload_size": self.maximum_payload_size,
            "timeout_support": self.timeout_support,
            "cancellation_support": self.cancellation_support,
            "local_or_hosted": self.local_or_hosted,
        }


@dataclass(frozen=True)
class ProviderModel:
    model_id: str
    display_name: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def vision(self) -> bool | None:
        if not self.capabilities:
            return None
        return "vision" in self.capabilities


@dataclass(frozen=True)
class ProbeResult:
    state: DiscoveryState
    message: str
    provider_id: str
    latency_ms: float | None = None
    models: tuple[ProviderModel, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.state == DiscoveryState.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "state": self.state.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "models": [
                {
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "capabilities": list(model.capabilities),
                    "metadata": dict(model.metadata),
                }
                for model in self.models
            ],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ModelValidation:
    state: DiscoveryState
    model_id: str | None
    message: str
    capabilities: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.state == DiscoveryState.AVAILABLE


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    data: bytes
    media_type: str = "image/png"

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id cannot be empty")
        if not self.data:
            raise ValueError("image data cannot be empty")
        if not self.media_type.startswith("image/"):
            raise ValueError("media_type must be an image media type")


@dataclass(frozen=True)
class LabelRequest:
    request_id: str
    model_id: str
    prompt: str
    images: tuple[ImageInput, ...]
    timeout_seconds: float
    schema_version: str = "spritelab.vision.label-request.v1"


@dataclass(frozen=True)
class NormalizedLabel:
    state: LabelState
    domain: str | None
    category: str | None
    canonical_object: str | None
    role: str | None
    description: str | None
    confidence: float
    abstention_reasons: tuple[str, ...]
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "domain": self.domain,
            "category": self.category,
            "canonical_object": self.canonical_object,
            "role": self.role,
            "description": self.description,
            "confidence": self.confidence,
            "abstention_reasons": list(self.abstention_reasons),
            "provider_metadata": dict(self.provider_metadata),
        }


@dataclass(frozen=True)
class ImageLabelResult:
    image_id: str
    request_id: str
    response_id: str
    label: NormalizedLabel | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.label is not None and self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "label": self.label.to_dict() if self.label else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class BatchResponse:
    request_id: str
    response_id: str
    results: tuple[ImageLabelResult, ...]
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestEstimate:
    image_count: int
    payload_bytes: int
    estimated_cost: float | None
    currency: str | None = None
    basis: str = "unknown"

    @property
    def cost_display(self) -> str:
        if self.estimated_cost is None:
            return "unknown"
        suffix = f" {self.currency}" if self.currency else ""
        return f"{self.estimated_cost:.6g}{suffix}"


@dataclass(frozen=True)
class ProviderHealth:
    state: DiscoveryState
    message: str
    checked_at: str
    latency_ms: float | None = None


ConfirmationCallback = Callable[[str], bool]


@runtime_checkable
class VisionProvider(Protocol):
    provider_id: str
    display_name: str
    provider_kind: ProviderKind
    privacy_class: PrivacyClass

    def probe(self) -> ProbeResult: ...

    def list_models(self) -> tuple[ProviderModel, ...]: ...

    def validate_model(self, model_id: str | None) -> ModelValidation: ...

    def capabilities(self, model_id: str | None = None) -> VisionCapabilities: ...

    def estimate_request(self, request: LabelRequest) -> RequestEstimate: ...

    def label_batch(self, request: LabelRequest) -> BatchResponse: ...

    def cancel(self, request_id: str) -> bool: ...

    def health(self) -> ProviderHealth: ...


def validate_provider_contract(provider: object) -> None:
    required_attributes = ("provider_id", "display_name", "provider_kind", "privacy_class")
    required_methods: Sequence[str] = (
        "probe",
        "list_models",
        "validate_model",
        "capabilities",
        "estimate_request",
        "label_batch",
        "cancel",
        "health",
    )
    missing = [name for name in required_attributes if not getattr(provider, name, None)]
    missing.extend(name for name in required_methods if not callable(getattr(provider, name, None)))
    if missing:
        raise TypeError(f"Vision provider is missing contract members: {', '.join(missing)}")

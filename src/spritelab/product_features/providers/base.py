"""Common bounded behavior shared by built-in provider adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Lock

from spritelab.product_features.providers.contracts import (
    DiscoveryState,
    LabelRequest,
    PrivacyClass,
    ProbeResult,
    ProviderHealth,
    RequestEstimate,
    VisionCapabilities,
)
from spritelab.product_features.providers.errors import ProviderCancelledError, ProviderError


class ProviderBase:
    pricing_per_request: float | None
    pricing_per_image: float | None
    pricing_currency: str | None

    def __init__(
        self,
        *,
        pricing_per_request: float | None = None,
        pricing_per_image: float | None = None,
        pricing_currency: str | None = None,
    ) -> None:
        self.pricing_per_request = pricing_per_request
        self.pricing_per_image = pricing_per_image
        self.pricing_currency = pricing_currency
        self._cancellations: dict[str, Event] = {}
        self._cancellation_lock = Lock()

    def estimate_request(self, request: LabelRequest) -> RequestEstimate:
        amount = None
        if self.pricing_per_request is not None or self.pricing_per_image is not None:
            amount = float(self.pricing_per_request or 0) + len(request.images) * float(self.pricing_per_image or 0)
        return RequestEstimate(
            image_count=len(request.images),
            payload_bytes=sum(len(image.data) for image in request.images),
            estimated_cost=amount,
            currency=self.pricing_currency if amount is not None else None,
            basis="explicit_configuration" if amount is not None else "unknown",
        )

    def cancel(self, request_id: str) -> bool:
        with self._cancellation_lock:
            event = self._cancellations.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def _begin_request(self, request: LabelRequest) -> Event:
        capabilities = self.capabilities(request.model_id)
        if not capabilities.vision:
            raise ProviderError(
                "provider_no_vision_capability", "The selected provider does not declare vision support."
            )
        if not capabilities.structured_output:
            raise ProviderError(
                "provider_structured_output_unsupported",
                "The selected provider does not declare structured-output support.",
            )
        if len(request.images) > capabilities.maximum_image_count:
            raise ProviderError("provider_batch_too_large", "The provider batch exceeds its image-count limit.")
        payload_size = sum(len(image.data) for image in request.images)
        if payload_size > capabilities.maximum_payload_size:
            raise ProviderError("provider_payload_too_large", "The provider batch exceeds its payload-size limit.")
        event = Event()
        with self._cancellation_lock:
            self._cancellations[request.request_id] = event
        return event

    def _end_request(self, request_id: str) -> None:
        with self._cancellation_lock:
            self._cancellations.pop(request_id, None)

    @staticmethod
    def _check_cancelled(event: Event) -> None:
        if event.is_set():
            raise ProviderCancelledError()

    def health(self) -> ProviderHealth:
        result = self.probe()
        return ProviderHealth(
            state=result.state,
            message=result.message,
            checked_at=datetime.now(UTC).isoformat(),
            latency_ms=result.latency_ms,
        )


def probe_from_error(provider_id: str, error: ProviderError, latency_ms: float | None = None) -> ProbeResult:
    states = {
        "provider_authentication_required": DiscoveryState.AUTHENTICATION_REQUIRED,
        "provider_credential_missing": DiscoveryState.MISCONFIGURED,
        "provider_endpoint_not_found": DiscoveryState.MISCONFIGURED,
        "provider_invalid_configuration": DiscoveryState.MISCONFIGURED,
        "provider_invalid_response": DiscoveryState.MISCONFIGURED,
        "provider_unsupported_model": DiscoveryState.UNSUPPORTED_MODEL,
    }
    return ProbeResult(states.get(error.code, DiscoveryState.UNAVAILABLE), str(error), provider_id, latency_ms)


def capability_set(
    privacy_class: PrivacyClass,
    *,
    vision: bool = True,
    structured_output: bool = True,
    multiple_images: bool = False,
    batching: bool = False,
    maximum_image_count: int = 1,
    maximum_payload_size: int = 20 * 1024 * 1024,
    timeout_support: bool = True,
    cancellation_support: bool = True,
) -> VisionCapabilities:
    return VisionCapabilities(
        vision=vision,
        structured_output=structured_output,
        multiple_images=multiple_images,
        batching=batching,
        maximum_image_count=maximum_image_count,
        maximum_payload_size=maximum_payload_size,
        timeout_support=timeout_support,
        cancellation_support=cancellation_support,
        local_or_hosted=privacy_class.value,
    )

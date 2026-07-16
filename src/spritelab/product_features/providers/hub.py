"""Privacy-gated, bounded, retrying, resumable vision-label orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.contracts import (
    ConfirmationCallback,
    ImageInput,
    ImageLabelResult,
    LabelRequest,
    PrivacyClass,
    PrivacyPolicy,
    RequestEstimate,
    VisionProvider,
)
from spritelab.product_features.providers.discovery import DiscoveredProvider, VisionProviderRegistry
from spritelab.product_features.providers.errors import (
    ProviderCancelledError,
    ProviderDeadlineExceededError,
    ProviderError,
    ProviderPolicyError,
)
from spritelab.product_features.providers.identity import request_identity, response_identity


@dataclass(frozen=True)
class RetryPolicy:
    maximum_retries: int = 2
    initial_backoff_seconds: float = 0.25
    maximum_backoff_seconds: float = 2.0

    @property
    def maximum_attempts(self) -> int:
        return self.maximum_retries + 1

    def delay(self, retry_number: int, retry_after: float | None = None) -> float:
        delay = (
            retry_after if retry_after is not None else self.initial_backoff_seconds * (2 ** max(0, retry_number - 1))
        )
        return max(0.0, min(float(delay), self.maximum_backoff_seconds))


@dataclass(frozen=True)
class ProviderRequestAttempt:
    request_id: str
    provider_id: str
    model_id: str
    image_ids: tuple[str, ...]
    attempt_number: int
    estimate: RequestEstimate
    request_context: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class HubRunResult:
    provider_id: str
    model_id: str
    results: tuple[ImageLabelResult, ...]
    estimates: tuple[RequestEstimate, ...]
    attempts: int
    resumed_count: int
    request_attempts: tuple[ProviderRequestAttempt, ...] = ()

    @property
    def successful_count(self) -> int:
        return sum(result.ok for result in self.results)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.successful_count

    @property
    def estimated_cost(self) -> float | None:
        if any(estimate.estimated_cost is None for estimate in self.estimates):
            return None
        return sum(float(estimate.estimated_cost or 0) for estimate in self.estimates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "results": [result.to_dict() for result in self.results],
            "attempts": self.attempts,
            "resumed_count": self.resumed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "estimated_cost": self.estimated_cost if self.estimated_cost is not None else "unknown",
            "request_attempts": [
                {
                    "request_id": attempt.request_id,
                    "provider_id": attempt.provider_id,
                    "model_id": attempt.model_id,
                    "image_ids": list(attempt.image_ids),
                    "attempt_number": attempt.attempt_number,
                    "estimated_cost": (
                        attempt.estimate.estimated_cost if attempt.estimate.estimated_cost is not None else "unknown"
                    ),
                    "request_context": dict(attempt.request_context),
                }
                for attempt in self.request_attempts
            ],
        }


class VisionProviderHub:
    def __init__(
        self,
        registry: VisionProviderRegistry,
        *,
        settings: ProviderSettings | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.settings = settings or registry.settings
        self.sleep = sleep
        self.clock = clock
        self._hosted_confirmed = False
        self._active: tuple[VisionProvider, str] | None = None

    def discover(self) -> tuple[DiscoveredProvider, ...]:
        return self.registry.discover()

    def selected_provider(self, *, provider_id: str | None = None) -> DiscoveredProvider | None:
        discovered = self.discover()
        return self.registry.select(discovered, provider_id=provider_id)

    def label_images(
        self,
        provider: VisionProvider,
        images: Sequence[ImageInput],
        *,
        prompt: str,
        model_id: str | None = None,
        batch_size: int | None = None,
        completed: Mapping[str, ImageLabelResult] | None = None,
        confirm_hosted: ConfirmationCallback | None = None,
        retry_policy: RetryPolicy | None = None,
        maximum_elapsed_seconds: float | None = None,
        request_context: Mapping[str, str] | None = None,
        on_request_started: Callable[[ProviderRequestAttempt], None] | None = None,
    ) -> HubRunResult:
        if maximum_elapsed_seconds is not None and (
            isinstance(maximum_elapsed_seconds, bool)
            or not isinstance(maximum_elapsed_seconds, (int, float))
            or not math.isfinite(maximum_elapsed_seconds)
            or maximum_elapsed_seconds <= 0
        ):
            raise ProviderPolicyError(
                "provider_deadline_exceeded", "The provider run has no positive elapsed-time budget remaining."
            )
        context = tuple(sorted(dict(request_context or {}).items()))
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not key.strip() or not value.strip()
            for key, value in context
        ):
            raise ProviderPolicyError(
                "provider_invalid_request_context", "Provider request context keys and values must be non-empty."
            )
        deadline = self.clock() + maximum_elapsed_seconds if maximum_elapsed_seconds is not None else None
        self._check_deadline(deadline)
        self._authorize_images(provider, confirm_hosted)
        validation = provider.validate_model(model_id)
        if not validation.valid or not validation.model_id:
            raise ProviderError("provider_unsupported_model", validation.message)
        selected_model = validation.model_id
        capabilities = provider.capabilities(selected_model)
        if not capabilities.vision:
            raise ProviderError("provider_no_vision_capability", "The selected provider does not support vision.")
        if not capabilities.structured_output:
            raise ProviderError(
                "provider_structured_output_unsupported", "The selected provider does not support structured output."
            )
        if deadline is not None and not capabilities.timeout_support:
            raise ProviderPolicyError(
                "provider_deadline_unsupported",
                "The selected provider cannot enforce the required request deadline.",
            )
        prior = {key: result for key, result in dict(completed or {}).items() if result.ok}
        pending = [image for image in images if image.image_id not in prior]
        maximum_count = capabilities.maximum_image_count if capabilities.batching else 1
        effective_batch_size = min(batch_size or self.settings.batch_size, maximum_count)
        chunks = _bounded_chunks(
            pending,
            maximum_count=effective_batch_size,
            maximum_payload_size=capabilities.maximum_payload_size,
        )
        policy = retry_policy or RetryPolicy(
            maximum_retries=self.settings.maximum_retries,
            initial_backoff_seconds=self.settings.backoff_seconds,
        )
        all_results: dict[str, ImageLabelResult] = dict(prior)
        estimates: list[RequestEstimate] = []
        request_attempts: list[ProviderRequestAttempt] = []
        attempts = 0
        for chunk in chunks:
            chunk_results, chunk_estimates, chunk_attempts, chunk_request_attempts = self._run_chunk(
                provider,
                tuple(chunk),
                prompt=prompt,
                model_id=selected_model,
                retry_policy=policy,
                deadline=deadline,
                request_context=context,
                on_request_started=on_request_started,
            )
            all_results.update((result.image_id, result) for result in chunk_results)
            estimates.extend(chunk_estimates)
            request_attempts.extend(chunk_request_attempts)
            attempts += chunk_attempts
        ordered = tuple(all_results[image.image_id] for image in images if image.image_id in all_results)
        return HubRunResult(
            provider.provider_id,
            selected_model,
            ordered,
            tuple(estimates),
            attempts,
            len(prior),
            tuple(request_attempts),
        )

    def cancel_active(self) -> bool:
        if self._active is None:
            return False
        provider, request_id = self._active
        return provider.cancel(request_id)

    def _run_chunk(
        self,
        provider: VisionProvider,
        images: tuple[ImageInput, ...],
        *,
        prompt: str,
        model_id: str,
        retry_policy: RetryPolicy,
        deadline: float | None,
        request_context: tuple[tuple[str, str], ...],
        on_request_started: Callable[[ProviderRequestAttempt], None] | None,
    ) -> tuple[
        tuple[ImageLabelResult, ...],
        tuple[RequestEstimate, ...],
        int,
        tuple[ProviderRequestAttempt, ...],
    ]:
        pending = images
        results: dict[str, ImageLabelResult] = {}
        estimates: list[RequestEstimate] = []
        request_attempts: list[ProviderRequestAttempt] = []
        attempts = 0
        last_error: ProviderError | None = None
        for attempt_number in range(1, retry_policy.maximum_attempts + 1):
            if not pending:
                break
            timeout_seconds = self._request_timeout(deadline)
            attempts += 1
            request_id = request_identity(
                provider.provider_id,
                model_id,
                prompt,
                pending,
                options={
                    "timeout_seconds": round(timeout_seconds, 6),
                    "attempt": attempt_number,
                    "request_context": dict(request_context),
                },
            )
            request = LabelRequest(
                request_id=request_id,
                model_id=model_id,
                prompt=prompt,
                images=pending,
                timeout_seconds=timeout_seconds,
            )
            estimate = provider.estimate_request(request)
            estimates.append(estimate)
            request_attempt = ProviderRequestAttempt(
                request_id,
                provider.provider_id,
                model_id,
                tuple(image.image_id for image in pending),
                attempt_number,
                estimate,
                request_context,
            )
            request_attempts.append(request_attempt)
            if on_request_started is not None:
                on_request_started(request_attempt)
            self._active = (provider, request_id)
            try:
                response = provider.label_batch(request)
                self._check_deadline(deadline)
                if response.request_id != request_id:
                    raise ProviderError(
                        "provider_invalid_output", "The provider response does not match the request identity."
                    )
                last_error = None
            except ProviderCancelledError:
                raise
            except ProviderDeadlineExceededError:
                raise
            except ProviderError as exc:
                self._check_deadline(deadline)
                last_error = exc
                if not exc.retryable or attempt_number >= retry_policy.maximum_attempts:
                    break
                self._sleep_before_deadline(retry_policy.delay(attempt_number, exc.retry_after), deadline)
                continue
            except Exception as exc:
                last_error = ProviderError(
                    "provider_internal_error",
                    f"The provider adapter failed safely ({type(exc).__name__}).",
                )
                break
            finally:
                self._active = None
            by_id = {result.image_id: result for result in response.results}
            retry_images: list[ImageInput] = []
            for image in pending:
                result = by_id.get(image.image_id)
                if result is None:
                    result = ImageLabelResult(
                        image.image_id,
                        request_id,
                        response.response_id,
                        error_code="provider_invalid_output",
                        error_message="The provider omitted this image result.",
                    )
                elif result.request_id != request_id or result.response_id != response.response_id:
                    result = ImageLabelResult(
                        image.image_id,
                        request_id,
                        response.response_id,
                        error_code="provider_invalid_output",
                        error_message="The provider returned inconsistent request or response identity.",
                    )
                if result.ok or not result.retryable:
                    results[image.image_id] = result
                elif attempt_number < retry_policy.maximum_attempts:
                    retry_images.append(image)
                else:
                    results[image.image_id] = _retry_exhausted(image.image_id, request_id, result.error_code)
            pending = tuple(retry_images)
            if pending:
                self._sleep_before_deadline(retry_policy.delay(attempt_number), deadline)
        if pending:
            error = last_error or ProviderError("provider_retry_exhausted", "Provider retries were exhausted.")
            terminal_code = (
                error.code if not error.retryable or retry_policy.maximum_retries == 0 else "provider_retry_exhausted"
            )
            for image in pending:
                request_id = request_identity(provider.provider_id, model_id, prompt, (image,))
                response_id = response_identity(request_id, {"error": terminal_code})
                results[image.image_id] = ImageLabelResult(
                    image.image_id,
                    request_id,
                    response_id,
                    error_code=terminal_code,
                    error_message=f"Provider request failed ({error.code}).",
                    retryable=False,
                )
        return (
            tuple(results[image.image_id] for image in images),
            tuple(estimates),
            attempts,
            tuple(request_attempts),
        )

    def _request_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.settings.timeout_seconds
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ProviderDeadlineExceededError()
        return min(self.settings.timeout_seconds, remaining)

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and self.clock() >= deadline:
            raise ProviderDeadlineExceededError()

    def _sleep_before_deadline(self, delay: float, deadline: float | None) -> None:
        if deadline is not None and delay >= deadline - self.clock():
            raise ProviderDeadlineExceededError()
        self.sleep(delay)
        self._check_deadline(deadline)

    def _authorize_images(self, provider: VisionProvider, confirm_hosted: ConfirmationCallback | None) -> None:
        policy = self.settings.privacy_policy
        if provider.privacy_class == PrivacyClass.LOCAL:
            if policy == PrivacyPolicy.HOSTED_ONLY:
                raise ProviderPolicyError(
                    "provider_policy_blocked", "The project policy permits hosted providers only."
                )
            return
        if policy == PrivacyPolicy.LOCAL_ONLY:
            raise ProviderPolicyError(
                "provider_policy_blocked",
                "The local_only policy blocks every image-bearing request to hosted providers.",
            )
        if policy == PrivacyPolicy.ASK_BEFORE_HOSTED and not self._hosted_confirmed:
            if confirm_hosted is None or not confirm_hosted(
                f"Send this and future image batches in this run to hosted provider {provider.display_name}?"
            ):
                raise ProviderPolicyError(
                    "provider_hosted_confirmation_required",
                    "Hosted image transfer was not confirmed.",
                )
            self._hosted_confirmed = True


def _bounded_chunks(
    images: Sequence[ImageInput], *, maximum_count: int, maximum_payload_size: int
) -> tuple[tuple[ImageInput, ...], ...]:
    chunks: list[tuple[ImageInput, ...]] = []
    current: list[ImageInput] = []
    current_size = 0
    for image in images:
        if len(image.data) > maximum_payload_size:
            raise ProviderError(
                "provider_payload_too_large", f"Image {image.image_id} exceeds the provider payload limit."
            )
        if current and (len(current) >= maximum_count or current_size + len(image.data) > maximum_payload_size):
            chunks.append(tuple(current))
            current = []
            current_size = 0
        current.append(image)
        current_size += len(image.data)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _retry_exhausted(image_id: str, request_id: str, last_code: str | None) -> ImageLabelResult:
    return ImageLabelResult(
        image_id,
        request_id,
        response_identity(request_id, {"error": "provider_retry_exhausted", "last_code": last_code}),
        error_code="provider_retry_exhausted",
        error_message=f"Provider retries were exhausted ({last_code or 'unknown error'}).",
    )

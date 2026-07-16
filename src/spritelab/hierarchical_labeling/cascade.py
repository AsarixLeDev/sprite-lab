"""Bounded labeling-cascade profiles built on the existing provider hub."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from spritelab.hierarchical_labeling.json_utils import ContractValidationError, content_identity, require_text
from spritelab.product_features.providers.contracts import (
    ImageInput,
    ImageLabelResult,
    PrivacyClass,
    VisionProvider,
)
from spritelab.product_features.providers.errors import (
    ProviderCancelledError,
    ProviderDeadlineExceededError,
    ProviderPolicyError,
)
from spritelab.product_features.providers.hub import (
    ProviderRequestAttempt,
    RetryPolicy,
    VisionProviderHub,
)

CASCADE_POLICY_SCHEMA = "spritelab.labeling.cascade-policy.v2"


class CascadeProfile(str, Enum):
    FAST_LOCAL = "fast_local"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass(frozen=True)
class HostedCostRate:
    provider_id: str
    model_id: str
    stage: str
    cost_per_request: float = 0.0
    cost_per_image: float = 0.0

    def __post_init__(self) -> None:
        for name in ("provider_id", "model_id", "stage"):
            require_text(getattr(self, name), name.replace("_", " "))
        for name in ("cost_per_request", "cost_per_image"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ContractValidationError(f"{name.replace('_', ' ')} must be finite and non-negative")

    def charge(self, image_count: int) -> float:
        return float(self.cost_per_request) + image_count * float(self.cost_per_image)


@dataclass(frozen=True)
class CascadeBudget:
    maximum_hosted_records: int = 0
    maximum_requests: int = 1000
    maximum_estimated_cost: float | None = None
    maximum_elapsed_seconds: float = 3600.0
    maximum_retries: int = 2
    trusted_hosted_cost_per_record: float | None = None
    trusted_hosted_cost_rates: tuple[HostedCostRate, ...] = ()

    def __post_init__(self) -> None:
        for name in ("maximum_hosted_records", "maximum_requests", "maximum_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractValidationError(f"{name.replace('_', ' ')} must be a non-negative integer")
        if (
            isinstance(self.maximum_elapsed_seconds, bool)
            or not isinstance(self.maximum_elapsed_seconds, (int, float))
            or not math.isfinite(self.maximum_elapsed_seconds)
            or self.maximum_elapsed_seconds <= 0
        ):
            raise ContractValidationError("maximum elapsed seconds must be finite and positive")
        for name in ("maximum_estimated_cost", "trusted_hosted_cost_per_record"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ContractValidationError(f"{name.replace('_', ' ')} must be finite and non-negative")
        if not all(isinstance(rate, HostedCostRate) for rate in self.trusted_hosted_cost_rates):
            raise ContractValidationError("trusted hosted cost rates must use HostedCostRate records")
        routes = [(rate.provider_id, rate.model_id, rate.stage) for rate in self.trusted_hosted_cost_rates]
        if len(routes) != len(set(routes)):
            raise ContractValidationError("trusted hosted cost routes cannot repeat")

    def hosted_rate(self, provider_id: str, model_id: str, stage: str) -> HostedCostRate | None:
        exact = next(
            (
                rate
                for rate in self.trusted_hosted_cost_rates
                if (rate.provider_id, rate.model_id, rate.stage) == (provider_id, model_id, stage)
            ),
            None,
        )
        if exact is not None:
            return exact
        if self.trusted_hosted_cost_per_record is None:
            return None
        return HostedCostRate(
            provider_id,
            model_id,
            stage,
            cost_per_image=self.trusted_hosted_cost_per_record,
        )


@dataclass(frozen=True)
class CascadePolicy:
    profile: CascadeProfile
    budget: CascadeBudget = field(default_factory=CascadeBudget)
    local_provider_role: str = "local"
    primary_provider_role: str = "primary"
    verifier_provider_role: str = "verifier"

    def __post_init__(self) -> None:
        for name in ("local_provider_role", "primary_provider_role", "verifier_provider_role"):
            require_text(getattr(self, name), name.replace("_", " "))

    @property
    def identity(self) -> str:
        return content_identity(
            CASCADE_POLICY_SCHEMA,
            {
                "profile": self.profile.value,
                "budget": self.budget.__dict__,
                "local_provider_role": self.local_provider_role,
                "primary_provider_role": self.primary_provider_role,
                "verifier_provider_role": self.verifier_provider_role,
            },
        )


@dataclass(frozen=True)
class CascadeStageResult:
    stage: str
    provider_role: str
    provider_id: str
    model_id: str
    requested_record_ids: tuple[str, ...]
    results: tuple[ImageLabelResult, ...]
    attempts: int
    resumed_count: int
    estimated_cost: float | None

    @property
    def successful_count(self) -> int:
        return sum(result.ok for result in self.results)


@dataclass(frozen=True)
class HostedRequestCharge:
    stage: str
    provider_id: str
    model_id: str
    request_id: str
    image_ids: tuple[str, ...]
    attempt_number: int
    estimated_cost: float | None


@dataclass(frozen=True)
class CascadeRunResult:
    profile: CascadeProfile
    policy_identity: str
    stages: tuple[CascadeStageResult, ...]
    status: str
    stop_reason: str | None
    hosted_record_count: int
    request_count: int
    estimated_cost: float | None
    elapsed_seconds: float
    exception_queue: tuple[str, ...]
    hosted_request_count: int
    hosted_charges: tuple[HostedRequestCharge, ...]


@dataclass
class _StageReservation:
    stage: str
    provider_id: str
    model_id: str
    hosted: bool
    allowed_image_ids: frozenset[str]
    maximum_requests: int
    maximum_image_attempts: int
    rate: HostedCostRate | None
    observed_requests: int = 0
    observed_image_attempts: int = 0


@dataclass
class _BudgetLedger:
    budget: CascadeBudget
    started: float
    hosted_records: set[str] = field(default_factory=set)
    requests: int = 0
    estimated_cost: float | None = 0.0
    hosted_charges: list[HostedRequestCharge] = field(default_factory=list)

    def elapsed(self, clock: Callable[[], float]) -> float:
        return max(0.0, clock() - self.started)

    def reserve(
        self,
        provider: VisionProvider,
        stage: str,
        model_id: str,
        images: Sequence[ImageInput],
        *,
        minimum_requests: int,
        maximum_attempts: int,
        maximum_image_attempts: int,
        clock: Callable[[], float],
    ) -> _StageReservation:
        if self.elapsed(clock) >= self.budget.maximum_elapsed_seconds:
            raise ProviderPolicyError(
                "labeling_elapsed_budget_exhausted", "The labeling elapsed-time budget is exhausted."
            )
        if self.requests + maximum_attempts > self.budget.maximum_requests:
            raise ProviderPolicyError("labeling_request_budget_exhausted", "The labeling request budget is exhausted.")
        if minimum_requests < 1 or maximum_attempts < minimum_requests:
            raise ContractValidationError("cascade request reservation is invalid")
        if maximum_image_attempts < len(images):
            raise ContractValidationError("cascade image-attempt reservation is invalid")
        hosted = provider.privacy_class == PrivacyClass.HOSTED
        image_ids = frozenset(image.image_id for image in images)
        rate = self.budget.hosted_rate(provider.provider_id, model_id, stage) if hosted else None
        if hosted and len(self.hosted_records | set(image_ids)) > self.budget.maximum_hosted_records:
            raise ProviderPolicyError(
                "labeling_hosted_record_budget_exhausted", "The maximum hosted-record budget would be exceeded."
            )
        if hosted and self.budget.maximum_estimated_cost is not None:
            if rate is None or self.estimated_cost is None:
                raise ProviderPolicyError(
                    "labeling_cost_unknown_budget_unenforceable",
                    "Hosted cost is unknown, so the configured cost ceiling cannot be enforced safely.",
                )
            projected = (
                float(self.estimated_cost)
                + maximum_attempts * rate.cost_per_request
                + maximum_image_attempts * rate.cost_per_image
            )
            if projected > self.budget.maximum_estimated_cost:
                raise ProviderPolicyError(
                    "labeling_cost_budget_exhausted", "The trusted estimated-cost budget is exhausted."
                )
        return _StageReservation(
            stage,
            provider.provider_id,
            model_id,
            hosted,
            image_ids,
            maximum_attempts,
            maximum_image_attempts,
            rate,
        )

    def record_request(self, reservation: _StageReservation, attempt: ProviderRequestAttempt) -> None:
        route = (attempt.provider_id, attempt.model_id, dict(attempt.request_context).get("cascade_stage"))
        expected = (reservation.provider_id, reservation.model_id, reservation.stage)
        if route != expected:
            raise ProviderPolicyError(
                "labeling_request_route_mismatch", "The provider request does not match its budget reservation."
            )
        image_ids = frozenset(attempt.image_ids)
        if not image_ids or not image_ids.issubset(reservation.allowed_image_ids):
            raise ProviderPolicyError(
                "labeling_request_scope_mismatch", "The provider request exceeds its reserved image scope."
            )
        if attempt.estimate.image_count != len(attempt.image_ids):
            raise ProviderPolicyError(
                "labeling_request_estimate_mismatch", "The provider estimate does not match the request image count."
            )
        if (
            reservation.observed_requests + 1 > reservation.maximum_requests
            or reservation.observed_image_attempts + len(attempt.image_ids) > reservation.maximum_image_attempts
            or self.requests + 1 > self.budget.maximum_requests
        ):
            raise ProviderPolicyError(
                "labeling_request_budget_exhausted", "The provider request exceeds its worst-case reservation."
            )
        charge = None
        if reservation.hosted:
            charge = (
                reservation.rate.charge(len(attempt.image_ids))
                if reservation.rate is not None
                else attempt.estimate.estimated_cost
            )
            if charge is not None and (
                isinstance(charge, bool)
                or not isinstance(charge, (int, float))
                or not math.isfinite(charge)
                or charge < 0
            ):
                raise ProviderPolicyError(
                    "labeling_request_cost_invalid", "The provider request cost is not finite and non-negative."
                )
        reservation.observed_requests += 1
        reservation.observed_image_attempts += len(attempt.image_ids)
        self.requests += 1
        if not reservation.hosted:
            return
        self.hosted_records.update(image_ids)
        if charge is None:
            self.estimated_cost = None
        elif self.estimated_cost is not None:
            self.estimated_cost += float(charge)
        self.hosted_charges.append(
            HostedRequestCharge(
                reservation.stage,
                reservation.provider_id,
                reservation.model_id,
                attempt.request_id,
                attempt.image_ids,
                attempt.attempt_number,
                float(charge) if charge is not None else None,
            )
        )


class LabelingCascade:
    """Execute configured providers without embedding provider/model defaults.

    The provider hub remains authoritative for privacy consent, credentials,
    structured-output capability, retries, partial results, and cancellation.
    This layer adds profile routing and whole-run budgets.
    """

    def __init__(
        self,
        hub: VisionProviderHub,
        providers: Mapping[str, VisionProvider],
        *,
        model_ids: Mapping[str, str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.hub = hub
        self.providers = dict(providers)
        self.model_ids = dict(model_ids or {})
        self.clock = clock
        self._cancelled = False

    def cancel(self) -> bool:
        already = self._cancelled
        self._cancelled = True
        return self.hub.cancel_active() or not already

    def run(
        self,
        images: Sequence[ImageInput],
        *,
        policy: CascadePolicy,
        prompts: Mapping[str, str],
        uncertain_record_ids: Sequence[str] = (),
        high_impact_record_ids: Sequence[str] = (),
        confirm_hosted: Callable[[str], bool] | None = None,
        completed_by_stage: Mapping[str, Mapping[str, ImageLabelResult]] | None = None,
    ) -> CascadeRunResult:
        if not images:
            raise ContractValidationError("labeling cascade requires at least one image")
        image_ids = [image.image_id for image in images]
        if len(image_ids) != len(set(image_ids)):
            raise ContractValidationError("labeling cascade image identities cannot repeat")
        required_prompts = {"description", "hypotheses"}
        if policy.profile != CascadeProfile.FAST_LOCAL:
            required_prompts.add("verification")
        if not required_prompts.issubset(prompts) or any(not str(prompts[key]).strip() for key in required_prompts):
            raise ContractValidationError("labeling cascade is missing a configured stage prompt")
        self._cancelled = False
        started = self.clock()
        ledger = _BudgetLedger(policy.budget, started)
        completed = dict(completed_by_stage or {})
        planned = self._plan(policy, images, uncertain_record_ids, high_impact_record_ids)
        stages: list[CascadeStageResult] = []
        stop_reason: str | None = None
        for stage, role, stage_images in planned:
            if self._cancelled:
                stop_reason = "cancelled"
                break
            try:
                stages.append(
                    self._run_stage(
                        stage,
                        role,
                        stage_images,
                        prompt=prompts[stage],
                        policy=policy,
                        ledger=ledger,
                        confirm_hosted=confirm_hosted,
                        completed=completed.get(stage, {}),
                    )
                )
                if ledger.elapsed(self.clock) >= policy.budget.maximum_elapsed_seconds:
                    stop_reason = "labeling_elapsed_budget_exhausted"
                    break
            except ProviderCancelledError:
                stop_reason = "cancelled"
                break
            except ProviderDeadlineExceededError:
                stop_reason = "labeling_elapsed_budget_exhausted"
                break
            except ProviderPolicyError as exc:
                stop_reason = (
                    "labeling_elapsed_budget_exhausted" if exc.code == "provider_deadline_exceeded" else exc.code
                )
                break
        failed = {result.image_id for stage in stages for result in stage.results if not result.ok}
        missing = set(image_ids) if stop_reason else set()
        exception_queue = tuple(sorted(failed | missing))
        status = (
            "cancelled" if stop_reason == "cancelled" else "partial" if stop_reason or exception_queue else "completed"
        )
        elapsed = ledger.elapsed(self.clock)
        return CascadeRunResult(
            policy.profile,
            policy.identity,
            tuple(stages),
            status,
            stop_reason,
            len(ledger.hosted_records),
            ledger.requests,
            ledger.estimated_cost,
            elapsed,
            exception_queue,
            len(ledger.hosted_charges),
            tuple(ledger.hosted_charges),
        )

    def _plan(
        self,
        policy: CascadePolicy,
        images: Sequence[ImageInput],
        uncertain_record_ids: Sequence[str],
        high_impact_record_ids: Sequence[str],
    ) -> tuple[tuple[str, str, tuple[ImageInput, ...]], ...]:
        by_id = {image.image_id: image for image in images}
        escalation = tuple(
            by_id[record_id]
            for record_id in sorted(set(uncertain_record_ids) | set(high_impact_record_ids))
            if record_id in by_id
        )
        local = policy.local_provider_role
        primary = policy.primary_provider_role
        verifier = policy.verifier_provider_role
        if policy.profile == CascadeProfile.FAST_LOCAL:
            self._require_local(local)
            return (("description", local, tuple(images)), ("hypotheses", local, tuple(images)))
        if policy.profile == CascadeProfile.BALANCED:
            self._require_local(local)
            planned: list[tuple[str, str, tuple[ImageInput, ...]]] = [
                ("description", local, tuple(images)),
                ("hypotheses", local, tuple(images)),
            ]
            if escalation:
                planned.append(("verification", verifier, escalation))
            return tuple(planned)
        return (
            ("description", primary, tuple(images)),
            ("hypotheses", primary, tuple(images)),
            ("verification", verifier, tuple(images)),
        )

    def _require_local(self, role: str) -> None:
        provider = self._provider(role)
        if provider.privacy_class != PrivacyClass.LOCAL:
            raise ProviderPolicyError(
                "labeling_profile_requires_local", "This labeling profile requires a local provider."
            )

    def _provider(self, role: str) -> VisionProvider:
        try:
            return self.providers[role]
        except KeyError as exc:
            raise ProviderPolicyError(
                "labeling_provider_role_unconfigured", f"No provider is configured for cascade role {role}."
            ) from exc

    def _run_stage(
        self,
        stage: str,
        role: str,
        images: Sequence[ImageInput],
        *,
        prompt: str,
        policy: CascadePolicy,
        ledger: _BudgetLedger,
        confirm_hosted: Callable[[str], bool] | None,
        completed: Mapping[str, ImageLabelResult],
    ) -> CascadeStageResult:
        provider = self._provider(role)
        model_id = self.model_ids.get(role)
        validation = provider.validate_model(model_id)
        if not validation.valid or not validation.model_id:
            raise ProviderPolicyError("labeling_model_unavailable", "The configured cascade model is unavailable.")
        pending = tuple(
            image for image in images if image.image_id not in completed or not completed[image.image_id].ok
        )
        if not pending:
            return CascadeStageResult(
                stage,
                role,
                provider.provider_id,
                validation.model_id,
                tuple(image.image_id for image in images),
                tuple(completed[image.image_id] for image in images),
                0,
                len(images),
                0.0,
            )
        capabilities = provider.capabilities(validation.model_id)
        maximum_count = capabilities.maximum_image_count if capabilities.batching else 1
        effective_batch_size = min(self.hub.settings.batch_size, maximum_count)
        batches = _minimum_batch_count(pending, effective_batch_size, capabilities.maximum_payload_size)
        available_attempts = max(0, policy.budget.maximum_requests - ledger.requests)
        retries = min(policy.budget.maximum_retries, max(0, available_attempts // batches - 1))
        maximum_attempts = batches * (retries + 1)
        reservation = ledger.reserve(
            provider,
            stage,
            validation.model_id,
            pending,
            minimum_requests=batches,
            maximum_attempts=maximum_attempts,
            maximum_image_attempts=len(pending) * (retries + 1),
            clock=self.clock,
        )
        remaining = policy.budget.maximum_elapsed_seconds - ledger.elapsed(self.clock)
        result = self.hub.label_images(
            provider,
            images,
            prompt=prompt,
            model_id=validation.model_id,
            completed=completed,
            confirm_hosted=confirm_hosted,
            retry_policy=RetryPolicy(maximum_retries=retries),
            maximum_elapsed_seconds=remaining,
            request_context={"cascade_stage": stage, "cascade_policy": policy.identity},
            on_request_started=lambda attempt: ledger.record_request(reservation, attempt),
        )
        return CascadeStageResult(
            stage,
            role,
            result.provider_id,
            result.model_id,
            tuple(image.image_id for image in images),
            result.results,
            result.attempts,
            result.resumed_count,
            result.estimated_cost,
        )


def _minimum_batch_count(images: Sequence[ImageInput], maximum_count: int, maximum_payload_size: int) -> int:
    count = 0
    current_count = 0
    current_size = 0
    for image in images:
        if len(image.data) > maximum_payload_size:
            raise ProviderPolicyError("labeling_payload_too_large", "An image exceeds the provider payload limit.")
        if current_count and (current_count >= maximum_count or current_size + len(image.data) > maximum_payload_size):
            count += 1
            current_count = 0
            current_size = 0
        current_count += 1
        current_size += len(image.data)
    return count + int(bool(current_count))

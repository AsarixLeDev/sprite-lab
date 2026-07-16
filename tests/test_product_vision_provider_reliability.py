from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from spritelab.product_features.providers import (
    DeterministicMockVisionProvider,
    ImageInput,
    PrivacyClass,
    PrivacyPolicy,
    ProviderCancelledError,
    ProviderDeadlineExceededError,
    ProviderError,
    ProviderPolicyError,
    ProviderSettings,
    ProviderTimeoutError,
    RetryPolicy,
    VisionProviderHub,
    VisionProviderRegistry,
)


def label(object_name: str = "sword", *, abstain: bool = False) -> dict[str, object]:
    return {
        "state": "abstained" if abstain else "labeled",
        "domain": None if abstain else "fantasy",
        "category": None if abstain else "weapon",
        "canonical_object": None if abstain else object_name,
        "role": None if abstain else "equipment",
        "description": "Unclear silhouette." if abstain else f"A {object_name}.",
        "confidence": 0.1 if abstain else 0.9,
        "abstention_reasons": ["ambiguous_silhouette"] if abstain else [],
        "provider_metadata": {"fixture": True},
    }


def hub_for(provider, **settings_overrides) -> VisionProviderHub:
    defaults = {
        "privacy_policy": PrivacyPolicy.ALLOW_HOSTED,
        "maximum_retries": 0,
        "batch_size": 4,
        "timeout_seconds": 1,
    }
    defaults.update(settings_overrides)
    settings = ProviderSettings(**defaults)
    registry = VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=())
    return VisionProviderHub(registry, settings=settings, sleep=lambda _seconds: None)


def images(*names: str) -> tuple[ImageInput, ...]:
    return tuple(ImageInput(name, f"bytes-{name}".encode()) for name in names)


def test_partial_batch_failure_preserves_successful_item() -> None:
    provider = DeterministicMockVisionProvider(
        {"good": label(), "bad": ProviderError("provider_item_failed", "fixture item failure")}
    )
    result = hub_for(provider).label_images(provider, images("good", "bad"), prompt="Label.")
    assert result.successful_count == 1
    assert result.results[0].label.canonical_object == "sword"
    assert result.results[1].error_code == "provider_item_failed"


def test_malformed_response_becomes_provider_invalid_output() -> None:
    provider = DeterministicMockVisionProvider({"bad": {**label(), "confidence": "high"}})
    result = hub_for(provider).label_images(provider, images("bad"), prompt="Label.")
    assert result.results[0].error_code == "provider_invalid_output"


def test_timeout_is_normalized_when_retries_are_disabled() -> None:
    provider = DeterministicMockVisionProvider(failures=(ProviderTimeoutError(),))
    result = hub_for(provider).label_images(
        provider, images("sprite"), prompt="Label.", retry_policy=RetryPolicy(maximum_retries=0)
    )
    assert result.results[0].error_code == "provider_timeout"


def test_retry_exhaustion_is_bounded() -> None:
    provider = DeterministicMockVisionProvider(
        failures=(
            ProviderError("provider_rate_limited", "limited", retryable=True, retry_after=100),
            ProviderError("provider_rate_limited", "limited", retryable=True),
            ProviderError("provider_rate_limited", "limited", retryable=True),
        )
    )
    delays: list[float] = []
    settings = ProviderSettings(privacy_policy=PrivacyPolicy.ALLOW_HOSTED, maximum_retries=2)
    hub = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=()),
        settings=settings,
        sleep=delays.append,
    )
    result = hub.label_images(provider, images("sprite"), prompt="Label.")
    assert result.attempts == 3
    assert result.results[0].error_code == "provider_retry_exhausted"
    assert max(delays) <= 2


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class LateMock(DeterministicMockVisionProvider):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__({"sprite": label()})
        self.clock = clock
        self.seen_timeout: float | None = None

    def label_batch(self, request):
        self.seen_timeout = request.timeout_seconds
        self.clock.advance(2.0)
        return super().label_batch(request)


def test_deadline_is_propagated_observed_and_late_response_is_rejected() -> None:
    clock = FakeClock()
    provider = LateMock(clock)
    settings = ProviderSettings(
        privacy_policy=PrivacyPolicy.ALLOW_HOSTED,
        maximum_retries=0,
        timeout_seconds=30.0,
    )
    hub = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=()),
        settings=settings,
        sleep=lambda _seconds: None,
        clock=clock,
    )
    attempts = []
    with pytest.raises(ProviderDeadlineExceededError):
        hub.label_images(
            provider,
            images("sprite"),
            prompt="Label.",
            maximum_elapsed_seconds=1.0,
            request_context={"cascade_stage": "description"},
            on_request_started=attempts.append,
        )
    assert provider.seen_timeout == 1.0
    assert provider.call_count == len(attempts) == 1
    assert attempts[0].image_ids == ("sprite",)
    assert dict(attempts[0].request_context) == {"cascade_stage": "description"}


def test_deadline_unsupported_provider_fails_before_call() -> None:
    class NoDeadlineMock(DeterministicMockVisionProvider):
        def capabilities(self, model_id=None):
            return replace(super().capabilities(model_id), timeout_support=False)

    provider = NoDeadlineMock({"sprite": label()})
    with pytest.raises(ProviderPolicyError, match="cannot enforce"):
        hub_for(provider).label_images(
            provider,
            images("sprite"),
            prompt="Label.",
            maximum_elapsed_seconds=1.0,
        )
    assert provider.call_count == 0


class CancellingMock(DeterministicMockVisionProvider):
    def label_batch(self, request):
        event = self._begin_request(request)
        try:
            assert self.cancel(request.request_id)
            self._check_cancelled(event)
            raise AssertionError("unreachable")
        finally:
            self._end_request(request.request_id)


def test_cancellation_stops_batch_without_retry() -> None:
    provider = CancellingMock()
    with pytest.raises(ProviderCancelledError):
        hub_for(provider).label_images(provider, images("sprite"), prompt="Label.")


class HostedMock(DeterministicMockVisionProvider):
    privacy_class = PrivacyClass.HOSTED


def test_local_only_policy_blocks_before_any_image_call() -> None:
    provider = HostedMock({"sprite": label()})
    hub = hub_for(provider, privacy_policy=PrivacyPolicy.LOCAL_ONLY)
    with pytest.raises(ProviderPolicyError, match="local_only"):
        hub.label_images(provider, images("sprite"), prompt="Label.")
    assert provider.call_count == 0


def test_ask_before_hosted_confirms_once_before_first_batch() -> None:
    provider = HostedMock({"one": label(), "two": label("shield")})
    hub = hub_for(provider, privacy_policy=PrivacyPolicy.ASK_BEFORE_HOSTED, batch_size=1)
    prompts: list[str] = []
    result = hub.label_images(
        provider,
        images("one", "two"),
        prompt="Label.",
        confirm_hosted=lambda message: prompts.append(message) or True,
    )
    assert result.successful_count == 2
    assert len(prompts) == 1
    assert provider.call_count == 2


def test_ask_before_hosted_decline_sends_no_image() -> None:
    provider = HostedMock({"sprite": label()})
    hub = hub_for(provider, privacy_policy=PrivacyPolicy.ASK_BEFORE_HOSTED)
    with pytest.raises(ProviderPolicyError, match="not confirmed"):
        hub.label_images(provider, images("sprite"), prompt="Label.", confirm_hosted=lambda _message: False)
    assert provider.call_count == 0


def test_bounded_batches_and_per_image_order() -> None:
    provider = DeterministicMockVisionProvider({name: label(name) for name in ("one", "two", "three")})
    result = hub_for(provider, batch_size=2).label_images(provider, images("one", "two", "three"), prompt="Label.")
    assert provider.call_count == 2
    assert [item.image_id for item in result.results] == ["one", "two", "three"]


def test_resumability_skips_completed_images() -> None:
    first_provider = DeterministicMockVisionProvider({"one": label("sword")})
    first = hub_for(first_provider).label_images(first_provider, images("one"), prompt="Label.")
    provider = DeterministicMockVisionProvider({"two": label("shield")})
    resumed = hub_for(provider).label_images(
        provider,
        images("one", "two"),
        prompt="Label.",
        completed={"one": first.results[0]},
    )
    assert resumed.resumed_count == 1
    assert provider.call_count == 1
    assert [item.label.canonical_object for item in resumed.results] == ["sword", "shield"]


def test_request_and_response_identity_are_stable() -> None:
    provider = DeterministicMockVisionProvider({"sprite": label()})
    first = hub_for(provider).label_images(provider, images("sprite"), prompt="Label.")
    second = hub_for(provider).label_images(provider, images("sprite"), prompt="Label.")
    assert first.results[0].request_id == second.results[0].request_id
    assert first.results[0].response_id == second.results[0].response_id


def test_cost_is_unknown_without_explicit_pricing() -> None:
    provider = DeterministicMockVisionProvider({"sprite": label()})
    result = hub_for(provider).label_images(provider, images("sprite"), prompt="Label.")
    assert result.estimated_cost is None
    assert result.estimates[0].cost_display == "unknown"


def test_abstention_survives_hub_unchanged() -> None:
    provider = DeterministicMockVisionProvider({"sprite": label(abstain=True)})
    result = hub_for(provider).label_images(provider, images("sprite"), prompt="Label.")
    assert result.results[0].label.abstention_reasons == ("ambiguous_silhouette",)


def test_no_vision_and_no_structured_output_fail_before_label_call() -> None:
    class CapabilityMock(DeterministicMockVisionProvider):
        def __init__(self, capabilities: Mapping[str, bool]) -> None:
            super().__init__()
            self.flags = capabilities

        def capabilities(self, model_id=None):
            original = super().capabilities(model_id)
            return type(original)(
                **{
                    **original.to_dict(),
                    "vision": self.flags.get("vision", original.vision),
                    "structured_output": self.flags.get("structured_output", original.structured_output),
                }
            )

    no_vision = CapabilityMock({"vision": False})
    with pytest.raises(ProviderError, match="vision"):
        hub_for(no_vision).label_images(no_vision, images("sprite"), prompt="Label.")
    no_structured = CapabilityMock({"structured_output": False})
    with pytest.raises(ProviderError, match="structured"):
        hub_for(no_structured).label_images(no_structured, images("sprite"), prompt="Label.")
    assert no_vision.call_count == no_structured.call_count == 0

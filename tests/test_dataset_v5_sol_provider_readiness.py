"""Synthetic, no-network fail-closed checks for the Sol provider-readiness audit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from spritelab.dataset_v5 import sol as sol_module
from spritelab.dataset_v5.sol import (
    SolConfig,
    SolEndpointIdentity,
    SolModelUnavailable,
    SolProvider,
)

EXACT_MODEL = "gpt-5.6-sol"
EXACT_ENDPOINT = "https://api.openai.com/v1"


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "SPRITELAB_SOL_API_KEY": "x",
        "SPRITELAB_SOL_BACKEND": "openai_responses_v1",
        "SPRITELAB_SOL_BASE_URL": EXACT_ENDPOINT,
        "SPRITELAB_SOL_MODEL": EXACT_MODEL,
    }
    values.update(overrides)
    return values


def _config(*, backend: str = "openai_responses_v1", base_url: str = EXACT_ENDPOINT) -> SolConfig:
    return SolConfig(backend=backend, model=EXACT_MODEL, base_url=base_url, api_key="x")


def _metadata(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "family": "GPT-5.6 Sol",
        "id": EXACT_MODEL,
        "provider": "openai",
        "request_schema_version": "responses-v1",
        "version": "fixture-version",
    }
    value.update(overrides)
    return value


def _identity() -> SolEndpointIdentity:
    return SolEndpointIdentity(
        backend="openai_responses_v1",
        endpoint_identity=EXACT_ENDPOINT,
        model_identifier=EXACT_MODEL,
        model_family="GPT-5.6 Sol",
        model_version="fixture-version",
        provider="openai",
        request_schema_version="responses-v1",
    )


def test_requested_sol_response_claiming_another_model_fails_closed() -> None:
    body = {
        "model": "gpt-5.6-terra",
        "model_version": "fixture-version",
        "output_text": "{}",
        "provider": "openai",
    }
    with pytest.raises(SolModelUnavailable):
        sol_module._parse_provider_body(body, _identity())


def test_model_absent_fails_closed() -> None:
    values = _environment()
    values.pop("SPRITELAB_SOL_MODEL")
    with pytest.raises(SolModelUnavailable):
        SolConfig.from_env(values)


def test_endpoint_change_reported_by_transport_fails_closed() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {
            "final_url": "https://changed.example/v1/models/gpt-5.6-sol",
            "json": _metadata(),
            "status": 200,
        }

    with pytest.raises(SolModelUnavailable):
        SolProvider(_config(), transport=transport).preflight()


def test_model_alias_change_fails_closed() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"json": _metadata(id="gpt-5.6"), "status": 200}

    with pytest.raises(SolModelUnavailable):
        SolProvider(_config(), transport=transport).preflight()


def test_version_missing_fails_closed() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"json": _metadata(version=""), "status": 200}

    with pytest.raises(SolModelUnavailable):
        SolProvider(_config(), transport=transport).preflight()


def test_api_key_absent_fails_closed() -> None:
    values = _environment()
    values.pop("SPRITELAB_SOL_API_KEY")
    with pytest.raises(SolModelUnavailable):
        SolConfig.from_env(values)


def test_non_openai_provider_is_rejected_even_when_it_self_claims_sol() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"json": _metadata(provider="not-openai"), "status": 200}

    provider = SolProvider(
        _config(backend="openai_compatible_responses_v1", base_url="https://other.example/v1"),
        transport=transport,
    )
    with pytest.raises(SolModelUnavailable):
        provider.preflight()


def test_official_model_object_without_extra_private_fields_fails_closed() -> None:
    def transport(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {
            "json": {"created": 1783872000, "id": EXACT_MODEL, "object": "model", "owned_by": "openai"},
            "status": 200,
        }

    with pytest.raises(SolModelUnavailable):
        SolProvider(_config(), transport=transport).preflight()

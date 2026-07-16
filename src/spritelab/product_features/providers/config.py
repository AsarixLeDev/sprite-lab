"""Validated provider settings derived from project and environment configuration."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from spritelab.product_core.settings import reject_secret_settings
from spritelab.product_features.providers.contracts import PrivacyClass, PrivacyPolicy, ProviderMode


@dataclass(frozen=True)
class ProviderSettings:
    mode: ProviderMode = ProviderMode.AUTOMATIC
    adapter: str | None = None
    provider_id: str | None = None
    endpoint: str | None = None
    model: str | None = None
    credential_env: str | None = None
    privacy_policy: PrivacyPolicy = PrivacyPolicy.ASK_BEFORE_HOSTED
    location: PrivacyClass | None = None
    timeout_seconds: float = 30.0
    batch_size: int = 4
    maximum_retries: int = 2
    backoff_seconds: float = 0.25
    vision: bool = True
    structured_output: bool = True
    multiple_images: bool = True
    batching: bool = True
    maximum_image_count: int = 8
    maximum_payload_size: int = 20 * 1024 * 1024
    pricing_per_request: float | None = None
    pricing_per_image: float | None = None
    pricing_currency: str | None = None
    source: str = "project"
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, environ: Mapping[str, str] | None = None
    ) -> ProviderSettings:
        raw = dict(value or {})
        reject_secret_settings(raw)
        provider_type = str(raw.get("type", raw.get("mode", "auto"))).strip().lower()
        aliases = {
            "auto": (ProviderMode.AUTOMATIC, None, None),
            "automatic": (ProviderMode.AUTOMATIC, None, None),
            "local": (ProviderMode.LOCAL, None, PrivacyClass.LOCAL),
            "ollama": (ProviderMode.SPECIFIC, "ollama", PrivacyClass.LOCAL),
            "vllm": (ProviderMode.SPECIFIC, "openai_compatible", PrivacyClass.LOCAL),
            "openai_compatible": (ProviderMode.SPECIFIC, "openai_compatible", None),
            "openai-compatible": (ProviderMode.SPECIFIC, "openai_compatible", None),
            "hosted": (ProviderMode.HOSTED, "openai_compatible", PrivacyClass.HOSTED),
            "plugin": (ProviderMode.CUSTOM_PLUGIN, "plugin", None),
            "custom_plugin": (ProviderMode.CUSTOM_PLUGIN, "plugin", None),
            "runpod": (ProviderMode.SPECIFIC, "runpod_native", PrivacyClass.HOSTED),
        }
        if provider_type not in aliases:
            mode, adapter, location = ProviderMode.SPECIFIC, provider_type, None
        else:
            mode, adapter, location = aliases[provider_type]
        adapter = str(raw.get("adapter", adapter or "")).strip().lower() or None
        location_raw = raw.get("location")
        if location_raw:
            location = PrivacyClass(str(location_raw).strip().lower())
        endpoint = _optional_text(raw.get("endpoint"))
        environment = environ or os.environ
        source = "project"
        if mode == ProviderMode.AUTOMATIC and not endpoint and environment.get("SPRITELAB_VISION_ENDPOINT"):
            endpoint = environment["SPRITELAB_VISION_ENDPOINT"].strip()
            environment_adapter = environment.get("SPRITELAB_VISION_ADAPTER")
            adapter = environment_adapter.strip().lower() if environment_adapter else None
            location = PrivacyClass(environment.get("SPRITELAB_VISION_LOCATION", "local").strip().lower())
            raw.setdefault("model", environment.get("SPRITELAB_VISION_MODEL"))
            raw.setdefault("credential_env", environment.get("SPRITELAB_VISION_CREDENTIAL_ENV"))
            source = "environment"
        if endpoint:
            validate_endpoint(endpoint)
        if adapter == "openai_compatible" and mode != ProviderMode.AUTOMATIC and not endpoint:
            raise ValueError("An OpenAI-compatible provider requires an explicit endpoint base URL.")
        if endpoint and adapter in {None, "openai_compatible"} and location is None:
            if endpoint and is_loopback_endpoint(endpoint):
                location = PrivacyClass.LOCAL
            else:
                raise ValueError("A non-loopback OpenAI-compatible endpoint must declare location: hosted.")
        capabilities = raw.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            raise ValueError("provider capabilities must be a mapping")
        pricing = raw.get("pricing", {})
        if not isinstance(pricing, Mapping):
            raise ValueError("provider pricing must be a mapping")
        credential_env = _optional_text(raw.get("credential_env"))
        if credential_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", credential_env):
            raise ValueError("Credential environment-variable name is invalid.")
        return cls(
            mode=mode,
            adapter=adapter,
            provider_id=_optional_text(raw.get("provider_id")),
            endpoint=endpoint,
            model=_optional_text(raw.get("model")),
            credential_env=credential_env,
            privacy_policy=PrivacyPolicy(str(raw.get("privacy_policy", "ask_before_hosted")).strip().lower()),
            location=location,
            timeout_seconds=_bounded_float(raw.get("timeout", raw.get("timeout_seconds", 30)), 0.1, 3600),
            batch_size=_bounded_int(raw.get("batch_size", 4), 1, 1024),
            maximum_retries=_bounded_int(raw.get("maximum_retries", 2), 0, 10),
            backoff_seconds=_bounded_float(raw.get("backoff_seconds", 0.25), 0, 30),
            vision=bool(capabilities.get("vision", True)),
            structured_output=bool(capabilities.get("structured_output", True)),
            multiple_images=bool(capabilities.get("multiple_images", True)),
            batching=bool(capabilities.get("batching", True)),
            maximum_image_count=_bounded_int(capabilities.get("maximum_image_count", 8), 1, 1024),
            maximum_payload_size=_bounded_int(
                capabilities.get("maximum_payload_size", 20 * 1024 * 1024), 1, 1024 * 1024 * 1024
            ),
            pricing_per_request=_optional_nonnegative_float(pricing.get("per_request")),
            pricing_per_image=_optional_nonnegative_float(pricing.get("per_image")),
            pricing_currency=_optional_text(pricing.get("currency")),
            source=source,
            extra={key: item for key, item in raw.items() if key not in {"api_key", "token", "password", "secret"}},
        )

    def to_persisted_dict(self) -> dict[str, Any]:
        """Return the single non-secret representation used by UI and execution."""

        value: dict[str, Any] = {
            "mode": self.mode.value,
            "adapter": self.adapter,
            "provider_id": self.provider_id,
            "endpoint": self.endpoint,
            "model": self.model,
            "credential_env": self.credential_env,
            "privacy_policy": self.privacy_policy.value,
            "location": self.location.value if self.location else None,
            "timeout": self.timeout_seconds,
            "batch_size": self.batch_size,
            "batch_policy": {"maximum_images_per_request": self.batch_size},
        }
        reject_secret_settings(value)
        return value

    @classmethod
    def from_context_config(
        cls, config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
    ) -> ProviderSettings:
        providers = config.get("providers", {})
        if not isinstance(providers, Mapping):
            raise ValueError("providers configuration must be a mapping")
        vision = providers.get("vision", {})
        if not isinstance(vision, Mapping):
            raise ValueError("providers.vision configuration must be a mapping")
        return cls.from_mapping(vision, environ=environ)


def validate_endpoint(endpoint: str) -> None:
    parts = urlsplit(str(endpoint))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Provider endpoint must be an explicit http:// or https:// base URL.")
    if parts.username or parts.password:
        raise ValueError("Provider credentials must not be embedded in endpoint URLs.")
    secret_query_names = {
        key.casefold().replace("-", "_") for key, _value in parse_qsl(parts.query, keep_blank_values=True)
    }
    if secret_query_names.intersection({"api_key", "apikey", "token", "access_token", "auth_token", "password"}):
        raise ValueError("Provider credentials must not be embedded in endpoint query parameters.")


def is_loopback_endpoint(endpoint: str) -> bool:
    host = (urlsplit(endpoint).hostname or "").lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid integer provider settings")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"provider setting must be between {minimum} and {maximum}")
    return result


def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid numeric provider settings")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"provider setting must be between {minimum} and {maximum}")
    return result


def _optional_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result < 0:
        raise ValueError("configured provider pricing cannot be negative")
    return result

"""Passive configured/cached provider state projections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from spritelab.product_core import ProductSettingsRepository, ProjectContext
from spritelab.product_features.providers.config import ProviderSettings


def passive_provider_projection(context: ProjectContext, *, maximum_age_seconds: int = 900) -> dict[str, Any]:
    """Project configured/cached state with no provider construction or request."""

    repository = ProductSettingsRepository(context)
    raw, version, saved = repository.effective_settings("provider")
    settings = ProviderSettings.from_mapping(raw)
    section = repository.section("provider") if saved else None
    observation = section.get("observation") if isinstance(section, dict) else None
    configured = saved or bool(
        settings.endpoint
        or settings.model
        or settings.provider_id
        or settings.adapter
        or settings.mode.value != "automatic"
    )
    state = "configured_unverified" if configured else "not_configured"
    observed_at: str | None = None
    if isinstance(observation, dict):
        observed_at = str(observation.get("observed_at") or "") or None
        observation_version = int(observation.get("configuration_version", version))
        stale = observation_version != version
        if observed_at:
            try:
                observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
                stale = stale or age > maximum_age_seconds
            except ValueError:
                stale = True
        else:
            stale = True
        if stale:
            state = "stale_cached"
        else:
            cached = str(observation.get("state") or "unavailable")
            state = {
                "available": "previously_verified",
                "authentication_required": "authentication_required_cached",
            }.get(cached, "unavailable_cached")
    return {
        "state": state,
        "observation_timestamp": observed_at,
        "configuration_version": version,
        "configured": configured,
        "privacy_policy": settings.privacy_policy.value,
        "provider_requests": 0,
    }


__all__ = ["passive_provider_projection"]

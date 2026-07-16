"""Adapter from the provider hub to the foundation VisionProvider contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core import (
    ProductAction,
    ProductCapability,
    ProductEvent,
    ProductResult,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
)
from spritelab.product_features.providers.config import ProviderSettings
from spritelab.product_features.providers.contracts import ImageInput, LabelState
from spritelab.product_features.providers.discovery import DiscoveredProvider, VisionProviderRegistry
from spritelab.product_features.providers.hub import VisionProviderHub

ConfirmationCallback = Callable[[str], bool]
RegistryFactory = Callable[[ProviderSettings], VisionProviderRegistry]


class HubProductVisionProvider:
    """Expose one selected hub provider through the product-core protocol."""

    provider_id = "vision.provider-hub"
    title = "Vision provider hub"

    def __init__(
        self,
        context: ProjectContext,
        *,
        confirm_hosted: ConfirmationCallback | None = None,
        registry_factory: RegistryFactory = VisionProviderRegistry,
    ) -> None:
        self.context = ProductSettingsRepository(context).effective_context()
        self.settings = ProviderSettings.from_context_config(self.context.config)
        self.registry = registry_factory(self.settings)
        self.hub = VisionProviderHub(self.registry, settings=self.settings)
        self.confirm_hosted = confirm_hosted
        self._selected: DiscoveredProvider | None = None

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        del context
        try:
            self._selected = self.hub.selected_provider()
        except Exception as exc:
            return (
                ProductCapability(
                    "vision.labeling",
                    "Automatic descriptions",
                    ProductStatus.UNAVAILABLE,
                    f"Provider discovery failed safely ({type(exc).__name__}).",
                ),
            )
        if self._selected is None:
            return (
                ProductCapability(
                    "vision.labeling",
                    "Automatic descriptions",
                    ProductStatus.UNAVAILABLE,
                    "No compatible vision provider is available; image-only preparation remains available.",
                ),
            )
        return (
            ProductCapability(
                "vision.labeling",
                "Automatic descriptions",
                ProductStatus.READY,
                "A compatible vision provider is available.",
                details={
                    "provider_id": self._selected.provider.provider_id,
                    "privacy_class": self._selected.provider.privacy_class.value,
                },
            ),
        )

    def execute(
        self,
        action: ProductAction,
        context: ProjectContext,
        emit: Callable[[ProductEvent], None],
    ) -> ProductResult:
        del context
        if action.action_id != "dataset.semantic.propose":
            return ProductResult(
                ProductStatus.BLOCKED,
                "The provider hub refused an unsupported product action.",
                feature="providers",
                data={"proposals": (), "provider_calls": 0},
            )
        selected = self._selected or self.hub.selected_provider()
        if selected is None:
            return ProductResult(
                ProductStatus.UNAVAILABLE,
                "No compatible vision provider is available.",
                feature="providers",
                data={"proposals": (), "provider_calls": 0},
            )
        values = action.parameters.get("items", ())
        rows = [value for value in values if isinstance(value, Mapping)] if isinstance(values, Sequence) else []
        images = tuple(_image_input(value) for value in rows)
        emit(_event("provider_started", ProductStatus.RUNNING, 0, len(images)))
        run = self.hub.label_images(
            selected.provider,
            images,
            prompt=_conservative_prompt(),
            model_id=self.settings.model,
            confirm_hosted=self.confirm_hosted,
        )
        proposals = tuple(_proposal(row, result) for row, result in zip(rows, run.results, strict=True))
        emit(_event("provider_completed", ProductStatus.COMPLETE, len(images), len(images)))
        return ProductResult(
            ProductStatus.COMPLETE,
            "Conservative description proposals are ready for review.",
            feature="providers",
            data={
                "proposals": proposals,
                "provider_id": run.provider_id,
                "provider_calls": run.attempts,
                "estimated_cost": run.estimated_cost if run.estimated_cost is not None else "unknown",
            },
        )


def _image_input(row: Mapping[str, Any]) -> ImageInput:
    path = Path(str(row.get("image_path", ""))).expanduser().resolve()
    if not path.is_file():
        raise ValueError("A selected dataset image is no longer available.")
    return ImageInput(str(row.get("item_id", "")), path.read_bytes())


def _proposal(row: Mapping[str, Any], result: Any) -> dict[str, Any]:
    item_id = str(row.get("item_id", ""))
    label = result.label
    if not result.ok or label is None or label.state == LabelState.ABSTAINED:
        reasons = tuple(label.abstention_reasons) if label else ()
        return {
            "item_id": item_id,
            "abstained": True,
            "reason": reasons[0] if reasons else str(result.error_code or "provider_abstained"),
        }
    labels = {
        key: value
        for key, value in {
            "domain": label.domain,
            "category": label.category,
            "canonical_object": label.canonical_object,
            "role": label.role,
            "description": label.description,
        }.items()
        if value not in (None, "")
    }
    return {
        "item_id": item_id,
        "labels": labels,
        "confidence": label.confidence,
        "conflicts": ["provider_requires_review"] if label.state == LabelState.NEEDS_REVIEW else [],
        "health_ok": True,
    }


def _event(event_type: str, status: ProductStatus, current: int, total: int) -> ProductEvent:
    return ProductEvent(
        run_id="dataset-semantic-provider",
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature="dataset",
        stage="automatic-descriptions",
        event_type=event_type,
        status=status,
        current=current,
        total=total,
        message="Checking optional automatic descriptions.",
    )


def _conservative_prompt() -> str:
    return (
        "Describe each sprite conservatively. Abstain when identity, category, role, or description is uncertain. "
        "Return structured fields and never treat a proposal as human-verified truth."
    )


__all__ = ["HubProductVisionProvider"]

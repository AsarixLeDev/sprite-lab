"""Typed contracts shared by Sprite Lab product extensions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from spritelab.product_core.events import ProductEventValidationError, StrictJSONError, validate_finite_json

if TYPE_CHECKING:
    from spritelab.product_core.cli import ProductCliRegistry

PRODUCT_RESULT_SCHEMA = "spritelab.product.result.v1"
PRODUCT_EVENT_SCHEMA = "spritelab.product.event.v1"
PRODUCT_STATUS_SCHEMA = "spritelab.product.status.v1"
PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class ProductStatus(str, Enum):
    """Statuses suitable for the final user-facing product."""

    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


class DeveloperEvidenceStatus(str, Enum):
    """Additional evidence states reserved for developer surfaces."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    STALE = "STALE"
    NOT_AUDITED = "NOT_AUDITED"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True)
class ProjectContext:
    """Read-only project information passed into extensions."""

    project_root: Path
    config: Mapping[str, Any] = field(default_factory=dict)
    config_path: Path | None = None
    runs_directory: Path | None = None


@dataclass(frozen=True)
class ProductBlocker:
    code: str
    message: str
    resolution: str | None = None


@dataclass(frozen=True)
class ProductWarning:
    code: str
    message: str
    resolution: str | None = None


@dataclass(frozen=True)
class ProductCapability:
    capability_id: str
    title: str
    status: ProductStatus
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status in {ProductStatus.READY, ProductStatus.RUNNING, ProductStatus.COMPLETE}


@dataclass(frozen=True)
class ProductAction:
    action_id: str
    feature: str
    title: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ProductRun:
    run_id: str
    feature: str
    action_id: str
    status: ProductStatus
    backend_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    artifact_references: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductEvent:
    """One progress/event structure for CLI, web, local, and hosted execution."""

    run_id: str
    timestamp: str
    feature: str
    stage: str
    event_type: str
    status: ProductStatus
    current: int = 0
    total: int | None = None
    message: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact_references: tuple[str, ...] = ()
    schema_version: str = field(default=PRODUCT_EVENT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if type(self.current) is not int:
            raise ProductEventValidationError(
                "invalid_counter", "$.current", "ProductEvent.current must be an integer."
            )
        if self.total is not None and type(self.total) is not int:
            raise ProductEventValidationError("invalid_counter", "$.total", "ProductEvent.total must be an integer.")
        if self.current < 0:
            raise ValueError("ProductEvent.current cannot be negative.")
        if self.total is not None and self.total < 0:
            raise ValueError("ProductEvent.total cannot be negative.")
        if not isinstance(self.metrics, Mapping):
            raise ProductEventValidationError(
                "invalid_metrics", "$.metrics", "ProductEvent.metrics must be a JSON object."
            )
        try:
            validate_finite_json(dict(self.metrics), path="$.metrics")
        except StrictJSONError as exc:
            raise ProductEventValidationError(exc.code, exc.path, str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "feature": self.feature,
            "stage": self.stage,
            "event_type": self.event_type,
            "status": self.status.value,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "metrics": dict(self.metrics),
            "artifact_references": list(self.artifact_references),
        }
        try:
            validate_finite_json(payload)
        except StrictJSONError as exc:
            raise ProductEventValidationError(exc.code, exc.path, str(exc)) from exc
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProductEvent:
        try:
            validate_finite_json(value)
        except StrictJSONError as exc:
            raise ProductEventValidationError(exc.code, exc.path, str(exc)) from exc
        schema = value.get("schema_version")
        if schema != PRODUCT_EVENT_SCHEMA:
            raise ValueError(f"Unsupported product event schema: {schema!r}")
        metrics = value.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ProductEventValidationError(
                "invalid_metrics", "$.metrics", "ProductEvent.metrics must be a JSON object."
            )
        return cls(
            run_id=str(value["run_id"]),
            timestamp=str(value["timestamp"]),
            feature=str(value["feature"]),
            stage=str(value["stage"]),
            event_type=str(value["event_type"]),
            status=ProductStatus(str(value["status"])),
            current=int(value.get("current", 0)),
            total=int(value["total"]) if value.get("total") is not None else None,
            message=str(value.get("message", "")),
            metrics=dict(metrics),
            artifact_references=tuple(str(item) for item in value.get("artifact_references", ())),
        )


@dataclass(frozen=True)
class ProductResult:
    status: ProductStatus
    message: str
    feature: str | None = None
    action: ProductAction | None = None
    run: ProductRun | None = None
    capabilities: tuple[ProductCapability, ...] = ()
    blockers: tuple[ProductBlocker, ...] = ()
    warnings: tuple[ProductWarning, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=PRODUCT_RESULT_SCHEMA, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "message": self.message,
            "feature": self.feature,
            "action": asdict(self.action) if self.action else None,
            "run": _run_to_dict(self.run) if self.run else None,
            "capabilities": [_capability_to_dict(item) for item in self.capabilities],
            "blockers": [asdict(item) for item in self.blockers],
            "warnings": [asdict(item) for item in self.warnings],
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class WebNavigationItem:
    navigation_id: str
    title: str
    path: str
    order: int = 100
    capability_id: str | None = None

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError("Web navigation paths must start with '/'.")


@dataclass(frozen=True)
class WebAssetBundle:
    """Package-relative templates and static assets owned by one feature."""

    package: str
    templates: str = "templates"
    static: str = "static"


WebRouterFactory = Callable[[ProjectContext], object]
ProductStatusProvider = Callable[[ProjectContext], ProductResult]
ProductCapabilityProbe = Callable[[ProjectContext], Sequence[ProductCapability]]
ProductCliRegistration = Callable[["ProductCliRegistry"], None]
EventSink = Callable[[ProductEvent], None]


@dataclass(frozen=True)
class WebPlugin:
    plugin_id: str
    router_factory: WebRouterFactory
    navigation: tuple[WebNavigationItem, ...] = ()
    assets: tuple[WebAssetBundle, ...] = ()
    route_prefix: str = ""

    def __post_init__(self) -> None:
        _validate_plugin_id(self.plugin_id)
        if self.route_prefix and not self.route_prefix.startswith("/"):
            raise ValueError("WebPlugin.route_prefix must be empty or start with '/'.")


@dataclass(frozen=True)
class ProductPlugin:
    """Feature-owned product extension returned by ``build_plugin()``."""

    plugin_id: str
    title: str
    cli_registration: ProductCliRegistration
    status_provider: ProductStatusProvider
    capability_probe: ProductCapabilityProbe
    web_router_factory: WebRouterFactory | None = None
    navigation: tuple[WebNavigationItem, ...] = ()
    required_backend_capabilities: tuple[str, ...] = ()
    settings_schema: Mapping[str, Any] | type[Any] | None = None
    web_assets: tuple[WebAssetBundle, ...] = ()
    api_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_plugin_id(self.plugin_id)
        if not self.title.strip():
            raise ValueError("ProductPlugin.title cannot be empty.")
        if len(set(self.required_backend_capabilities)) != len(self.required_backend_capabilities):
            raise ValueError("ProductPlugin.required_backend_capabilities cannot contain duplicates.")
        if any(not value.strip() for value in self.required_backend_capabilities):
            raise ValueError("ProductPlugin.required_backend_capabilities cannot contain empty values.")
        if any(not value.startswith("/") for value in self.api_prefixes):
            raise ValueError("ProductPlugin.api_prefixes entries must start with '/'.")

    def web_plugin(self) -> WebPlugin | None:
        if self.web_router_factory is None:
            return None
        return WebPlugin(
            plugin_id=self.plugin_id,
            router_factory=self.web_router_factory,
            navigation=self.navigation,
            assets=self.web_assets,
        )


@runtime_checkable
class VisionProvider(Protocol):
    provider_id: str
    title: str

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]: ...

    def execute(self, action: ProductAction, context: ProjectContext, emit: EventSink) -> ProductResult: ...


@runtime_checkable
class ComputeBackend(Protocol):
    backend_id: str
    title: str

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]: ...

    def execute(self, action: ProductAction, context: ProjectContext, emit: EventSink) -> ProductResult: ...


@runtime_checkable
class ReviewQueue(Protocol):
    queue_id: str
    title: str

    def status(self, context: ProjectContext) -> ProductResult: ...

    def items(self, context: ProjectContext, *, limit: int | None = None) -> Sequence[Mapping[str, Any]]: ...

    def apply(self, action: ProductAction, context: ProjectContext) -> ProductResult: ...


@dataclass(frozen=True)
class DatasetImportResult:
    dataset_id: str | None
    status: ProductStatus
    imported: int = 0
    skipped: int = 0
    needs_review: int = 0
    blockers: tuple[ProductBlocker, ...] = ()
    warnings: tuple[ProductWarning, ...] = ()
    artifact_references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["blockers"] = [asdict(item) for item in self.blockers]
        value["warnings"] = [asdict(item) for item in self.warnings]
        value["artifact_references"] = list(self.artifact_references)
        return value


def _validate_plugin_id(plugin_id: str) -> None:
    if not PLUGIN_ID_PATTERN.fullmatch(plugin_id):
        raise ValueError(
            "Product plugin IDs must be lowercase dotted, dashed, or underscored identifiers beginning with a letter."
        )


def _capability_to_dict(value: ProductCapability) -> dict[str, Any]:
    result = asdict(value)
    result["status"] = value.status.value
    result["details"] = dict(value.details)
    return result


def _run_to_dict(value: ProductRun) -> dict[str, Any]:
    result = asdict(value)
    result["status"] = value.status.value
    result["artifact_references"] = list(value.artifact_references)
    return result

"""No-build local web host and feature router composition contracts."""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spritelab.product_core.contracts import ProductPlugin, ProjectContext
from spritelab.product_core.plugins import ProductPluginRegistry

if TYPE_CHECKING:
    from fastapi import FastAPI

DEFAULT_WEB_HOST = "127.0.0.1"


class WebSecurityError(ValueError):
    """Unsafe web bind configuration was rejected before server startup."""


@dataclass(frozen=True)
class WebServerSettings:
    host: str = DEFAULT_WEB_HOST
    port: int | str = "auto"
    open_browser: bool = True
    allow_non_loopback: bool = False
    authentication_token: str | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        allow_non_loopback: bool = False,
        authentication_token: str | None = None,
    ) -> WebServerSettings:
        ui = config.get("ui", {})
        if not isinstance(ui, Mapping):
            raise WebSecurityError("ui configuration must be a mapping.")
        settings = cls(
            host=str(ui.get("host", DEFAULT_WEB_HOST)),
            port=ui.get("port", "auto"),
            open_browser=bool(ui.get("open_browser", True)),
            allow_non_loopback=allow_non_loopback,
            authentication_token=authentication_token,
        )
        settings.validate()
        return settings

    @property
    def is_loopback(self) -> bool:
        host = self.host.strip().lower().rstrip(".")
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def resolved_port(self) -> int:
        return 0 if self.port == "auto" else int(self.port)

    def validate(self) -> None:
        if self.port != "auto" and (not isinstance(self.port, int) or isinstance(self.port, bool)):
            raise WebSecurityError("Web port must be 'auto' or an integer from 1 through 65535.")
        if isinstance(self.port, int) and not 1 <= self.port <= 65535:
            raise WebSecurityError("Web port must be 'auto' or an integer from 1 through 65535.")
        if not self.is_loopback:
            if not self.allow_non_loopback:
                raise WebSecurityError("Non-loopback binding requires the explicit allow_non_loopback option.")
            if not self.authentication_token:
                raise WebSecurityError("Non-loopback binding requires a runtime authentication token.")


def create_product_app(
    context: ProjectContext,
    *,
    plugins: Iterable[ProductPlugin] = (),
    settings: WebServerSettings | None = None,
) -> FastAPI:
    """Create a FastAPI app without launching a server or importing feature pages eagerly."""

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    effective = settings or WebServerSettings()
    effective.validate()
    registry = ProductPluginRegistry(plugins)
    app = FastAPI(title="Sprite Lab", docs_url=None, redoc_url=None)

    if not effective.is_loopback:
        token = effective.authentication_token or ""

        @app.middleware("http")
        async def require_runtime_token(request: Request, call_next: Any) -> Any:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {token}"
            if not hmac.compare_digest(supplied, expected):
                return JSONResponse(status_code=401, content={"detail": "Authentication required."})
            return await call_next(request)

    for plugin in registry:
        web_plugin = plugin.web_plugin()
        if web_plugin is None:
            continue
        router = web_plugin.router_factory(context)
        app.include_router(router, prefix=web_plugin.route_prefix)

    app.state.spritelab_context = context
    app.state.spritelab_plugins = tuple(registry)
    return app


def serve_product_app(
    context: ProjectContext,
    *,
    plugins: Iterable[ProductPlugin] = (),
    settings: WebServerSettings | None = None,
) -> None:
    """Launch the local server; callers own browser opening and runtime tokens."""

    import uvicorn

    effective = settings or WebServerSettings()
    effective.validate()
    app = create_product_app(context, plugins=plugins, settings=effective)
    uvicorn.run(app, host=effective.host, port=effective.resolved_port)

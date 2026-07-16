"""Launch contract for the local Sprite Lab v3 product."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import webbrowser
from collections.abc import Iterable, Sequence
from typing import Any

from spritelab.product_core import ProductPlugin, ProjectContext, WebSecurityError, WebServerSettings
from spritelab.product_web.app import create_app
from spritelab.v3.config import ConfigError, ProjectConfig


def _port(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be 'auto' or an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be from 1 through 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spritelab v3 app", description="Open the local Sprite Lab product."
    )
    parser.add_argument("--host", help="Explicit bind host (default: project setting or 127.0.0.1).")
    parser.add_argument("--port", type=_port, help="Port number or 'auto'.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the default browser.")
    parser.add_argument(
        "--auth-token",
        help="Runtime-only token required for non-loopback binding; SPRITELAB_WEB_TOKEN is preferred.",
    )
    return parser


def _interactive_desktop() -> bool:
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _browser_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def run_server(
    app: Any,
    settings: WebServerSettings,
    *,
    open_browser: bool,
    open_path: str = "/",
) -> None:
    """Bind first so an automatic port can be printed and opened deterministically."""

    import uvicorn

    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
    server_socket = socket.socket(family, socket.SOCK_STREAM)
    state = getattr(app, "state", None)
    shutdown_callback: Any = None
    server: Any = None
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((settings.host, settings.resolved_port))
        server_socket.listen(128)
        port = int(server_socket.getsockname()[1])
        browser_host = _browser_host(settings.host)
        bracketed = f"[{browser_host}]" if ":" in browser_host else browser_host
        if not open_path.startswith("/") or open_path.startswith("//"):
            raise ValueError("Product start paths must be local absolute URL paths.")
        url = f"http://{bracketed}:{port}{open_path}"
        print(f"Sprite Lab is available at {url}")
        if not settings.is_loopback:
            print("WARNING: Sprite Lab is bound beyond this computer. Authentication and CSRF protection are active.")
        if open_browser:
            webbrowser.open(url)
        config = uvicorn.Config(app, log_level="info", access_log=False)
        server = uvicorn.Server(config)

        def request_shutdown() -> None:
            server.should_exit = True

        shutdown_callback = request_shutdown
        if state is not None:
            state.spritelab_request_shutdown = shutdown_callback
        server.run(sockets=[server_socket])
    finally:
        if server is not None:
            server.should_exit = True
        if (
            state is not None
            and shutdown_callback is not None
            and getattr(state, "spritelab_request_shutdown", None) is shutdown_callback
        ):
            del state.spritelab_request_shutdown
        close_socket = getattr(server_socket, "close", None)
        if callable(close_socket):
            close_socket()


def _settings(config: ProjectConfig, args: argparse.Namespace) -> WebServerSettings:
    ui = config.values.get("ui", {})
    configured_host = str(ui.get("host", "127.0.0.1"))
    host = str(args.host or configured_host)
    port = args.port if args.port is not None else ui.get("port", "auto")
    token = args.auth_token or os.environ.get("SPRITELAB_WEB_TOKEN")
    candidate = WebServerSettings(
        host=host,
        port=port,
        open_browser=bool(ui.get("open_browser", True)) and not args.no_open,
        allow_non_loopback=args.host is not None,
        authentication_token=token,
    )
    candidate.validate()
    return candidate


def main(argv: Sequence[str] | None = None, *, plugins: Iterable[ProductPlugin] | None = None) -> None:
    args = build_parser().parse_args(list(argv or ()))
    try:
        config = ProjectConfig.load(required=False)
        settings = _settings(config, args)
    except (ConfigError, WebSecurityError) as exc:
        print(f"Could not start Sprite Lab: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    context = ProjectContext(
        project_root=config.root,
        config=config.values,
        config_path=config.path,
        runs_directory=config.runs_dir,
    )
    if plugins is None:
        from spritelab.product_runtime import build_product_runtime

        plugins = build_product_runtime().plugins
    app = create_app(context, plugins=plugins, settings=settings)
    should_open = settings.open_browser and _interactive_desktop()
    run_server(app, settings, open_browser=should_open)


__all__ = ["build_parser", "main", "run_server"]

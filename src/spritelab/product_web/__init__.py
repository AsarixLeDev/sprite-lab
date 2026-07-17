"""Unified, offline Sprite Lab v3 web product shell with lazy imports."""

__all__ = ["EventRepository", "RunSnapshot", "create_app", "main", "run_server"]


def __getattr__(name: str) -> object:
    if name in {"EventRepository", "RunSnapshot"}:
        from spritelab.product_web.events import EventRepository, RunSnapshot

        return {"EventRepository": EventRepository, "RunSnapshot": RunSnapshot}[name]
    if name == "create_app":
        from spritelab.product_web.app import create_app

        return create_app
    if name in {"main", "run_server"}:
        from spritelab.product_web.cli import main, run_server

        return {"main": main, "run_server": run_server}[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)

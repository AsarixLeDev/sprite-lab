"""Compatibility facade for the separately owned developer command suite."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from spritelab.dev_features.cli import (
    DeveloperCommandEnvironment,
    register_developer_commands,
)
from spritelab.dev_features.cli import (
    build_parser as _build_parser,
)
from spritelab.dev_features.cli import (
    main as _main,
)
from spritelab.v3.config import ProjectConfig
from spritelab.v3.status import build_project_state


def _load() -> ProjectConfig:
    """Compatibility seam retained for foundation callers and tests."""

    return ProjectConfig.load(Path.cwd(), required=False)


def _environment() -> DeveloperCommandEnvironment:
    return DeveloperCommandEnvironment(load_config=_load, build_project_state=build_project_state)


def build_parser():
    return _build_parser(environment=_environment())


def main(argv: Sequence[str] | None = None) -> None:
    _main(argv, environment=_environment())


__all__ = ["build_parser", "main", "register_developer_commands"]


if __name__ == "__main__":
    main()

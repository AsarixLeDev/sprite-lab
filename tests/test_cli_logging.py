from __future__ import annotations

import argparse
import logging
from types import ModuleType, SimpleNamespace

import pytest

from spritelab.harvest import cli as harvest_cli
from spritelab.training import cli as training_cli


@pytest.mark.parametrize("cli_module", [training_cli, harvest_cli])
def test_verbose_reconfigures_top_level_logger(cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    configured: dict[str, object] = {}
    parsed = SimpleNamespace(verbose=True, func=lambda _: None)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *_: parsed)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: configured.update(kwargs))

    cli_module.main([])

    assert configured["level"] == logging.DEBUG
    assert configured["force"] is True

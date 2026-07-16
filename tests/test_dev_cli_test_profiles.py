from __future__ import annotations

import copy
import io
import subprocess
from pathlib import Path

import pytest

from spritelab.dev_features import test_profiles
from spritelab.dev_features._process import run_process
from spritelab.dev_features.cli import DeveloperCommandEnvironment, build_parser, main
from spritelab.dev_features.test_profiles import build_test_plan, execute_test_plan
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import ProjectState


def _environment(root: Path) -> DeveloperCommandEnvironment:
    values = copy.deepcopy(DEFAULT_CONFIG)
    for section in ("dataset", "labeling", "training", "evaluation"):
        for key in values[section]:
            values[section][key] = [] if key == "review_queues" else ""
    config = ProjectConfig(root, None, values)
    state = ProjectState("synthetic", root, None, "a" * 40, [])
    return DeveloperCommandEnvironment(lambda: config, lambda _config: state)


def test_test_profile_dry_run_prints_exact_command_without_execution(monkeypatch, tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_dev_cli_synthetic.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    plan = build_test_plan(tmp_path, "quick", ("-x",))
    monkeypatch.setattr(test_profiles, "run_process", lambda *_args, **_kwargs: pytest.fail("executed dry-run"))
    stream = io.StringIO()
    result = execute_test_plan(plan, root=tmp_path, dry_run=True, announcement=stream)
    assert result is None
    assert stream.getvalue().strip() == f"Planned command: {plan.display_command}"
    assert plan.arguments[-1] == "-x"


def test_safe_argument_arrays_are_passed_without_shell(monkeypatch, tmp_path: Path) -> None:
    plan = build_test_plan(tmp_path, "full", ("-k", "path with spaces"))
    observed = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed.update(kwargs)
        return subprocess.CompletedProcess(list(arguments), 0, "", "")

    monkeypatch.setattr(test_profiles, "run_process", fake_run)
    execute_test_plan(plan, root=tmp_path, dry_run=False, capture_output=True)
    assert isinstance(observed["arguments"], tuple)
    assert observed["arguments"][-1] == "path with spaces"
    assert "shell" not in observed
    with pytest.raises(TypeError, match="shell command string"):
        run_process("pytest -q", cwd=tmp_path)  # type: ignore[arg-type]


def test_registration_exports_all_developer_commands(tmp_path: Path) -> None:
    parser = build_parser(environment=_environment(tmp_path))
    action = next(item for item in parser._actions if hasattr(item, "choices") and item.choices)
    assert {"status", "audits", "branches", "artifacts", "doctor", "test", "explain", "report"} <= set(action.choices)


def test_cli_test_defaults_to_quick_and_supports_output_flags(tmp_path: Path, capsys) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_dev_cli_synthetic.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["--no-color", "test", "--dry-run", "--json"], environment=_environment(tmp_path))
    output = capsys.readouterr()
    assert caught.value.code == 0
    assert '"profile": "quick"' in output.out
    assert '"shell": false' in output.out
    assert "Planned command:" in output.err


def test_profile_options_after_profile_are_not_forwarded_to_pytest(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["test", "training", "--dry-run", "--json"], environment=_environment(tmp_path))
    output = capsys.readouterr()
    assert caught.value.code == 0
    assert '"profile": "training"' in output.out
    assert '"dry_run": true' in output.out
    assert output.err.rstrip().endswith("-m pytest")
    assert "--dry-run" not in output.err
    assert "--json" not in output.err

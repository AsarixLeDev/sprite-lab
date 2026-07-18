from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

from spritelab.training import smoke_bundle


def _project_fixture(root: Path) -> list[dict[str, str]]:
    package = root / "src" / "spritelab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
import sys

if not any(
    type(candidate).__name__ == "_ExactProjectSourceFinder"
    and getattr(candidate, "enabled", False) is True
    for candidate in sys.meta_path
):
    raise RuntimeError("spritelab initializer executed before its exact finder")
INITIALIZED_AFTER_EXACT_FINDER = True
""".lstrip(),
        encoding="utf-8",
    )
    (package / "target.py").write_text("VALUE = 'trusted'\n", encoding="utf-8")
    for name in ("training", "utils"):
        child = package / name
        child.mkdir()
        (child / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "training" / "leaf.py").write_text("VALUE = 'bound leaf'\n", encoding="utf-8")
    (package / "training" / "smoke_bundle.py").write_text(
        """
class _ExactRuntimeFinder:
    def __init__(self):
        self.enabled = True
""".lstrip(),
        encoding="utf-8",
    )
    escape = package / "escape"
    escape.mkdir()
    (escape / "__init__.py").write_text("\n", encoding="utf-8")
    (escape / "target.py").write_text("VALUE = 'escaped alias'\n", encoding="utf-8")
    rows: list[dict[str, str]] = []
    for path in sorted(package.rglob("*.py")):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def _run_isolated(worker_path: Path, root: Path, body: str) -> subprocess.CompletedProcess[str]:
    source = textwrap.dedent(
        f"""
        import hashlib
        import importlib
        import importlib.util
        import pathlib
        import sys

        class EarlyProjectImportRejector:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "spritelab" or fullname.startswith("spritelab."):
                    raise AssertionError("project import ran while loading the policy bootstrap")
                return None

        worker_path = pathlib.Path(sys.argv[1])
        root = pathlib.Path(sys.argv[2])
        early_guard = EarlyProjectImportRejector()
        sys.meta_path.insert(0, early_guard)
        try:
            spec = importlib.util.spec_from_file_location("bound_playground_worker_test", worker_path)
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
        finally:
            sys.meta_path.remove(early_guard)
        assert not any(name == "spritelab" or name.startswith("spritelab.") for name in sys.modules)
        rows = []
        for path in sorted((root / "src" / "spritelab").rglob("*.py")):
            payload = path.read_bytes()
            rows.append({{
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }})
        sys.path.insert(0, str(root / "src"))

        {textwrap.indent(textwrap.dedent(body).strip(), "        ").lstrip()}
        """
    )
    return subprocess.run(
        [sys.executable, "-I", "-B", "-S", "-c", source, str(worker_path), str(root)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _assert_isolated_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_project_source_policy_refuses_late_installation(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        import types
        sys.modules["spritelab"] = types.ModuleType("spritelab")
        try:
            with worker._bound_project_source_imports(root, rows):
                raise AssertionError("late source policy unexpectedly started")
        except RuntimeError as exc:
            assert "loaded before its exact source policy" in str(exc)
        else:
            raise AssertionError("late source policy was accepted")
        """,
    )
    _assert_isolated_success(result)


def test_project_source_loader_rehashes_after_finder_selection(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        expected = worker._verify_code(root, rows)
        finder = worker._ExactProjectSourceFinder(root, expected)
        try:
            sys.meta_path.insert(0, finder)
            module_spec = finder.find_spec("spritelab.target", [str(root / "src" / "spritelab")])
            (root / "src" / "spritelab" / "target.py").write_text("VALUE = 'substituted'\\n", encoding="utf-8")
            module = importlib.util.module_from_spec(module_spec)
            try:
                module_spec.loader.exec_module(module)
            except RuntimeError as exc:
                assert "changed before execution" in str(exc)
            else:
                raise AssertionError("substituted project source executed")
        finally:
            sys.meta_path.remove(finder)
        """,
    )
    _assert_isolated_success(result)


def test_spritelab_initializer_executes_only_behind_exact_finder(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        expected = worker._verify_code(root, rows)
        finder = worker._ExactProjectSourceFinder(root, expected)
        try:
            sys.meta_path.insert(0, finder)
            package = importlib.import_module("spritelab")
            assert package.INITIALIZED_AFTER_EXACT_FINDER is True
            assert finder.loader_is_bound("spritelab", package.__loader__)
        finally:
            sys.meta_path.remove(finder)
            sys.modules.pop("spritelab", None)
        """,
    )
    _assert_isolated_success(result)


def test_project_source_finder_ignores_mutated_package_search_path(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        expected = worker._verify_code(root, rows)
        finder = worker._ExactProjectSourceFinder(root, expected)
        try:
            sys.meta_path.insert(0, finder)
            escaped_path = root / "src" / "spritelab" / "escape"
            module_spec = finder.find_spec("spritelab.target", [str(escaped_path)])
            assert pathlib.Path(module_spec.origin) == root / "src" / "spritelab" / "target.py"
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            assert module.VALUE == "trusted"
        finally:
            sys.meta_path.remove(finder)
        """,
    )
    _assert_isolated_success(result)


def test_project_source_finder_rejects_unexpected_selected_loader(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        class UnexpectedLoader:
            pass

        expected = worker._verify_code(root, rows)
        finder = worker._ExactProjectSourceFinder(root, expected)
        path_finder = worker.importlib_machinery.PathFinder
        original = path_finder.__dict__["find_spec"]

        def forged_spec(cls, fullname, path=None, target=None):
            origin = root / "src" / "spritelab" / "target.py"
            return worker.importlib_machinery.ModuleSpec(fullname, UnexpectedLoader(), origin=str(origin))

        try:
            sys.meta_path.insert(0, finder)
            path_finder.find_spec = classmethod(forged_spec)
            try:
                finder.find_spec("spritelab.target", [str(root / "src" / "spritelab")])
            except RuntimeError as exc:
                assert "unexpected loader" in str(exc)
            else:
                raise AssertionError("unexpected project loader was accepted")
        finally:
            path_finder.find_spec = original
            sys.meta_path.remove(finder)
        """,
    )
    _assert_isolated_success(result)


def test_project_source_inventory_rejects_portable_collisions(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        target = next(row for row in rows if row["path"].endswith("/target.py"))
        rows.append({**target, "path": "src/spritelab/TARGET.py"})
        try:
            worker._verify_code(root, rows)
        except RuntimeError as exc:
            assert "portable path collision" in str(exc)
        else:
            raise AssertionError("portable project-source collision was accepted")
        """,
    )
    _assert_isolated_success(result)


def test_project_source_policy_rejects_a_preceding_rogue_finder(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        class RogueFinder:
            def find_spec(self, fullname, path=None, target=None):
                raise AssertionError("rogue finder ran before the exact policy guard")

        rogue = RogueFinder()
        with worker._bound_project_source_imports(root, rows):
            try:
                sys.meta_path.insert(0, rogue)
            except RuntimeError as exc:
                assert "lost precedence" in str(exc)
            else:
                raise AssertionError("preceding rogue finder was accepted")
        """,
    )
    _assert_isolated_success(result)


def test_project_source_policy_rejects_spoofed_runtime_finder_ahead(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        runtime_type = type("_ExactRuntimeFinder", (), {})
        runtime_type.__module__ = "spritelab.training.smoke_bundle"
        runtime_finder = runtime_type()
        runtime_finder.enabled = True
        with worker._bound_project_source_imports(root, rows):
            sys.meta_path.accepting_runtime_finder = True
            try:
                sys.meta_path.insert(0, runtime_finder)
            except RuntimeError as exc:
                assert "lost precedence" in str(exc)
            else:
                raise AssertionError("spoofed runtime finder was accepted")
            finally:
                sys.meta_path.accepting_runtime_finder = False
        """,
    )
    _assert_isolated_success(result)


def test_project_source_policy_binds_exact_loaded_runtime_finder(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        import contextlib

        with worker._bound_project_source_imports(root, rows) as project_finder:
            runtime_module = importlib.import_module("spritelab.training.smoke_bundle")

            @contextlib.contextmanager
            def exact_runtime_policy(project_root, closure):
                assert project_root == root
                assert closure == {"identity": "bound"}
                runtime_finder = runtime_module._ExactRuntimeFinder()
                sys.meta_path.insert(0, runtime_finder)
                try:
                    yield {"verified": True}
                finally:
                    runtime_finder.enabled = False
                    sys.meta_path.remove(runtime_finder)

            with worker._bound_exact_runtime_imports(
                project_finder,
                exact_runtime_policy,
                root,
                {"identity": "bound"},
            ) as verified:
                project_finder.require_precedence()
                assert verified == {"verified": True}
                assert project_finder.runtime_finder is sys.meta_path[0]
                worker._activate_exact_project_packages(project_finder)
                assert sys.modules["spritelab"].INITIALIZED_AFTER_EXACT_FINDER is True
                assert project_finder.loader_is_bound("spritelab", sys.modules["spritelab"].__loader__)
        """,
    )
    _assert_isolated_success(result)


def test_project_source_policy_revalidates_after_exact_import(tmp_path: Path) -> None:
    _project_fixture(tmp_path)
    worker = (
        Path(__file__).parents[1] / "src" / "spritelab" / "product_features" / "evaluation" / "playground_worker.py"
    )
    result = _run_isolated(
        worker,
        tmp_path,
        """
        try:
            with worker._bound_project_source_imports(root, rows) as finder:
                assert getattr(sys.modules["spritelab"], "__spritelab_exact_bootstrap__", None) is True
                try:
                    worker._activate_exact_project_packages(finder)
                except RuntimeError as exc:
                    assert "bootstrap identity changed" in str(exc)
                else:
                    raise AssertionError("project initializer ran before the runtime policy")
                leaf = importlib.import_module("spritelab.training.leaf")
                assert leaf.VALUE == "bound leaf"
                assert type(leaf.__loader__).__name__ == "_ExactProjectSourceLoader"
                module = importlib.import_module("spritelab.target")
                assert module.VALUE == "trusted"
                assert type(module.__loader__).__name__ == "_ExactProjectSourceLoader"
                drift = b"VALUE = 'late drift'\\n"
                (root / "src" / "spritelab" / "target.py").write_bytes(drift)
                next(row for row in rows if row["path"].endswith("/target.py"))["sha256"] = hashlib.sha256(drift).hexdigest()
        except RuntimeError as exc:
            assert "changed before import" in str(exc)
        else:
            raise AssertionError("post-import project-source drift was accepted")
        """,
    )
    _assert_isolated_success(result)


def test_smoke_bootstrap_installs_source_finder_before_bound_project_run() -> None:
    source = smoke_bundle._CHILD_PREFLIGHT_SOURCE
    source_finder = source.index("_y.meta_path.insert(0, _BoundSourceFinder(root, expected))")
    bound_run = source.index('"spritelab" if mode == "main"')
    assert source_finder < bound_run

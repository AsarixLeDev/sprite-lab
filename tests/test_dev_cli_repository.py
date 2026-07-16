from __future__ import annotations

import subprocess
from pathlib import Path

from spritelab.dev_features.repository import branch_heads, list_local_branches


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )


def _repository(root: Path) -> None:
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "-c", "user.name=Sprite Lab", "-c", "user.email=sprite@example.invalid", "commit", "-m", "initial")
    _git(root, "branch", "topic/contained")


def test_branch_listing_is_read_only_and_reports_containment(tmp_path: Path) -> None:
    root = tmp_path / "repository with spaces"
    _repository(root)
    before = branch_heads(root)
    branches = list_local_branches(root)
    after = branch_heads(root)
    assert before == after
    assert {item["branch"] for item in branches} == {"main", "topic/contained"}
    assert all(item["merged"] is True for item in branches)
    assert next(item for item in branches if item["branch"] == "main")["current"] is True


def test_dirty_worktree_is_visible_without_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "dirty repository"
    _repository(root)
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    branch = next(item for item in list_local_branches(root) if item["branch"] == "main")
    assert branch["clean"] is False
    assert any("tracked.txt" in line for line in branch["changes"])
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "changed\n"

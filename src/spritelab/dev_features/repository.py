"""Read-only Git repository and worktree inspection."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from spritelab.dev_features._process import run_process

_READ_ONLY_GIT_COMMANDS = frozenset(
    {
        "branch",
        "config",
        "diff",
        "for-each-ref",
        "merge-base",
        "rev-list",
        "rev-parse",
        "status",
        "worktree",
    }
)


def git_result(root: Path, *arguments: str, timeout: float = 15) -> subprocess.CompletedProcess[str] | None:
    """Run an explicitly read-only Git operation."""

    if not arguments or arguments[0] not in _READ_ONLY_GIT_COMMANDS:
        command = arguments[0] if arguments else ""
        raise ValueError(f"Developer repository inspection does not allow git {command!r}.")
    try:
        return run_process(["git", *arguments], cwd=root, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def git_text(root: Path, *arguments: str, timeout: float = 15) -> str | None:
    result = git_result(root, *arguments, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip()


def _status_lines(root: Path) -> list[str] | None:
    result = git_result(root, "status", "--short", "--untracked-files=all")
    if result is None or result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def list_worktrees(root: Path) -> list[dict[str, Any]]:
    output = git_text(root, "worktree", "list", "--porcelain")
    if output is None:
        return []
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in [*output.splitlines(), ""]:
        if not line:
            if current:
                path = Path(str(current.get("path", root)))
                changes = _status_lines(path) if path.exists() else None
                current["clean"] = changes == [] if changes is not None else None
                current["changes"] = changes or []
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["commit"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached", "locked", "prunable"}:
            current[key] = value or True
    return records


def repository_state(root: Path) -> dict[str, Any]:
    branch = git_text(root, "branch", "--show-current")
    commit = git_text(root, "rev-parse", "HEAD")
    changes = _status_lines(root)
    worktrees = list_worktrees(root)
    resolved = str(root.resolve())
    worktree = next(
        (item for item in worktrees if _same_path(str(item.get("path", "")), resolved)),
        None,
    )
    return {
        "branch": branch or None,
        "commit": commit,
        "worktree": worktree,
        "clean": changes == [] if changes is not None else None,
        "changes": changes or [],
        "repository_available": commit is not None,
    }


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return left.casefold().replace("\\", "/") == right.casefold().replace("\\", "/")


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool | None:
    result = git_result(root, "merge-base", "--is-ancestor", ancestor, descendant)
    if result is None:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _branches_matching(root: Path, option: str) -> set[str] | None:
    output = git_text(root, "for-each-ref", "--format=%(refname:short)", option, "refs/heads")
    return set(output.splitlines()) if output is not None else None


def branch_heads(root: Path) -> dict[str, str]:
    output = git_text(root, "for-each-ref", "--format=%(refname:short)%00%(objectname)", "refs/heads")
    if output is None:
        return {}
    return {values[0]: values[1] for line in output.splitlines() if len(values := line.split("\x00")) == 2}


def _ahead_behind(root: Path, branch: str, upstream: str | None) -> tuple[int | None, int | None]:
    if not upstream:
        return None, None
    output = git_text(root, "rev-list", "--left-right", "--count", f"{branch}...{upstream}")
    if output is None:
        return None, None
    values = output.replace("\t", " ").split()
    if len(values) != 2:
        return None, None
    try:
        return int(values[0]), int(values[1])
    except ValueError:
        return None, None


def _supersession_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    patterns = ("experiments/**/supersession_matrix.json", "experiments/**/merge_plan.json")
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            entries = payload.get("entries", []) if isinstance(payload, dict) else []
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict) or not isinstance(entry.get("branch"), str):
                    continue
                disposition = str(entry.get("disposition", "")).upper()
                if disposition not in {"SUPERSEDED", "CONTENT_EQUIVALENT"}:
                    continue
                records[entry["branch"]] = {
                    "disposition": disposition,
                    "replacement": entry.get("replacement_or_equivalent") or entry.get("related_identity"),
                    "evidence": entry.get("evidence") or entry.get("rationale"),
                    "record": str(path),
                }
    return records


def list_local_branches(root: Path) -> list[dict[str, Any]]:
    """Return local branch facts without changing repository state."""

    current = git_text(root, "branch", "--show-current")
    head = git_text(root, "rev-parse", "HEAD")
    output = git_text(
        root,
        "for-each-ref",
        "--format=%(refname:short)%00%(objectname)%00%(upstream:short)%00%(worktreepath)",
        "refs/heads",
    )
    if output is None:
        return []
    supersession = _supersession_records(root)
    worktree_states = {str(item.get("path")): item for item in list_worktrees(root)}
    merged_branches = _branches_matching(root, "--merged=HEAD")
    containing_branches = _branches_matching(root, "--contains=HEAD")
    branches: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = line.split("\x00")
        if len(values) != 4:
            continue
        name, commit, upstream, worktree_path = values
        ahead, behind = _ahead_behind(root, name, upstream or None)
        worktree = worktree_states.get(worktree_path)
        branches.append(
            {
                "branch": name,
                "commit": commit,
                "current": name == current,
                "worktree": worktree_path or None,
                "clean": worktree.get("clean") if worktree else None,
                "changes": worktree.get("changes", []) if worktree else [],
                "upstream": upstream or None,
                "ahead": ahead,
                "behind": behind,
                "merged": name in merged_branches if merged_branches is not None and head else None,
                "contained": name in merged_branches if merged_branches is not None and head else None,
                "contains_current": name in containing_branches if containing_branches is not None and head else None,
                "likely_superseded": name in supersession,
                "supersession": supersession.get(name),
            }
        )
    return branches

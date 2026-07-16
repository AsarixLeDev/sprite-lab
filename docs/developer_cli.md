# Sprite Lab developer CLI

The developer command suite keeps repository evidence and engineering diagnostics separate from the final product experience. It is available under:

```text
python -m spritelab dev <command>
```

The suite is read-only except for `dev test`, which runs tests, and `dev report --output`, which writes the explicitly requested report. It never merges, deletes, resets, checks out, or rewrites branches or artifacts. Existing low-level Sprite Lab commands remain available.

## Shared output options

Every developer command accepts the options before or after its command name:

- `--json` emits a stable machine-readable result.
- `--no-color` disables terminal color. Current output is deliberately plain even without this option.
- `--quiet` suppresses successful human output. Test failures remain visible.
- `--debug` includes a traceback for unexpected internal failures.

## Commands

### `dev status`

Shows the current branch and commit, worktree state, subsystem implementation and execution status, independent audit verdicts, audit applicability and freshness, failed gates, artifact identities, dataset freeze identities, training and promotion authorization, active developer runs, and the recommended engineering action.

This is the intended home for source commits, SHA-256 values, branch names, audit matrices, and gate evidence.

### `dev audits`

Lists, for each known independent audit surface:

- subsystem and verdict;
- bound and current commits;
- applicability, freshness, and whether it is a current certification;
- failed gates and report path;
- consequence for downstream authorization.

A stale report has `applicable=false`, `current_certification=false`, and the consequence `NO_CURRENT_CERTIFICATION`. A fresh pass makes the dependent action eligible for its other checks; it does not authorize training or promotion by itself.

### `dev branches`

Lists local branch heads, registered worktrees, clean or dirty state, upstream ahead/behind counts, whether the branch is merged or contained in the current branch, whether it contains the current commit, and likely supersession recorded in repository evidence.

The command snapshots branch heads before and after inspection and reports whether they remained unchanged. It invokes only read-only Git operations.

### `dev artifacts`

Inspects configured reports, manifests, hash files, project-state evidence, and recognizable path/hash bindings inside JSON artifacts. Each reference is classified as `PRESENT`, `CURRENT`, `MISSING`, `HASH_MISMATCH`, or `INVALID_REFERENCE`. Files are never repaired or rewritten.

### `dev doctor`

Runs the developer-specific, read-only diagnostics for repository state, worktrees, optional executables, test dependencies, fixture availability, audit artifact integrity, Windows path-length risk, generated caches, and configured external fixtures. It does not initialize CUDA or call providers.

### `dev test [profile] [--dry-run] [-- pytest arguments]`

The default profile is `quick`. The exact planned command is printed before execution. Commands are executed as argument arrays with no shell. Pytest's exit code is returned unchanged, and failures are not hidden.

Profiles:

- `quick`: developer CLI tests and the product-foundation compatibility test;
- `dataset`: dataset tests and v3 configuration/state tests;
- `labeling`: label and semantic tests;
- `training`: training tests;
- `evaluation`: memorization tests and the v3 run-report tests;
- `full`: the complete configured pytest suite.

Examples:

```text
python -m spritelab dev test
python -m spritelab dev test training --dry-run
python -m spritelab dev test evaluation -- -x
```

### `dev explain [subsystem]`

Explains the requested subsystem with its blockers, evidence, audit state, and next engineering action. With no subsystem, it explains the currently recommended action. Common aliases such as `training-audit`, `memorization`, `promotion`, and `freeze` are accepted.

### `dev report [--output PATH]`

Combines the full developer state with the safe product projection. An output ending in `.json` receives JSON; other output paths receive Markdown. Without `--output`, the report is returned only through the selected output renderer.

## User/developer separation

`spritelab.dev_features.project_user_status()` is an allowlisted reusable projection. It constructs four simple areas—Dataset, Training, Evaluation, and Project result—from detailed state. It never copies branch names, commits, hashes, evidence paths, audit verdicts, gate matrices, or technical blockers.

For example, detailed developer evidence can say:

```text
Training implementation: IMPLEMENTED
Training audit: FAIL
Failed gates: 1, 5, 6, 8, 9
Audit commit: ...
Current commit: ...
```

The product projection says only:

```text
Training
  Not available yet
```

The developer package does not modify the final product UI.

## Extension registration

The foundation-facing callback is:

```python
from spritelab.dev_features import register_developer_commands

register_developer_commands(subparsers, parents=[common_output_parser])
```

Its complete signature is `register_developer_commands(subparsers, *, parents=(), environment=None) -> None`. The optional `DeveloperCommandEnvironment` supplies the configuration loader and v3 project-state builder for integration or tests. The package-level command registry in `spritelab.__main__` is not modified by feature registration.

# Sprite Lab agent guide

This file applies to the whole repository and is the first file every agent
must read. Keep it concise: feature detail belongs in the nearest source, test,
or document.

## Mandatory filesystem safety

Filesystem safety outranks task completion. User data, generated datasets,
credentials, ignored files, and files outside this repository are never
disposable scratch space.

- Treat the resolved repository root as the only default mutation boundary.
  Anything outside it is read-only unless the user explicitly names the exact
  target and authorizes that outside-workspace change.
- Never delete or recursively move a drive/filesystem root, home/profile
  directory, repository root, or a path computed from an empty value, wildcard,
  `.`, or `..`.
- Do not run `rm -rf`, recursive `Remove-Item`, `rmdir /s`, `git clean`,
  `git reset --hard`, `git checkout --`, or ad-hoc `shutil.rmtree`. Do not route
  around this rule through Python, another shell, or a generated script.
- Prefer additive edits, `apply_patch`, exclusive creation, and unique
  same-directory temporary files followed by atomic replacement. Never
  overwrite source datasets or user artifacts when a fresh output works.
- Before any necessary delete, replacement, or recursive move: inspect Git
  status; resolve the exact target; prove it is an owned generated path strictly
  below its approved root; reject symlink, junction, reparse-point, mount, and
  hard-link hazards; enumerate the impact; and obtain explicit user approval.
- Use one exact literal target in one shell. Afterward, verify the target, root,
  worktree status, and any outside sentinel used by the security test.
- Never discard unrelated worktree changes. Existing changes belong to the
  user. Preserve them or stop when a requested edit overlaps them.
- Keep tests/scratch output in a unique repository-local `.pytest_tmp_<task>`
  or another explicitly owned directory. Never clean another task's temp root.
- New path-management code must fail closed: validate lexical and resolved
  containment, reject link/reparse seams, separate input from managed output,
  publish atomically, and test that outside files remain byte-identical.
- Use `spritelab.utils.safe_fs` for recursive cleanup and atomic file replacement
  instead of adding new raw deletion helpers.

Inspect these high-risk areas before changing or invoking them:

- `src/spritelab/dataset_maker/exporter.py`: transactional export replacement.
- `src/spritelab/product_features/dataset/intake.py` and `sidecar.py`: directory
  publication, rollback, metadata transactions, and hardened cleanup.
- `src/spritelab/dataset_v5/`: staging trees and immutable/refuse-overwrite
  freezes.
- `src/spritelab/remote_compute/ssh.py`: local/remote staging and SSH cleanup.
- `src/spritelab/harvest/download.py`: exclusive partial downloads and atomic
  destination replacement.
- `scripts/reset_autoonly_imported.py`: confirmed in-place JSONL rewrite after a
  backup; never run noninteractively without explicit authorization.

For path safety, begin with `tests/test_safe_filesystem.py`,
`tests/test_product_universal_dataset_intake.py`,
`tests/test_product_complete_remediation.py`,
`tests/test_event_history_transaction_atomicity.py`, and the relevant
Dataset-v5/remote-compute tests.

## Fast start and search discipline

1. Run `git status --short` and `git branch --show-current` before edits.
2. Read this file, the task-specific source, its closest tests, and the relevant
   feature document. Do not preload every design document.
3. Use `rg --files <area>` and `rg -n "symbol|schema|command" src tests docs`.
   Ripgrep respects `.gitignore`; do not add `-uuu` or crawl ignored data unless
   the task explicitly concerns it.
4. Find tests first with `rg -n "def test_.*keyword|SymbolName" tests`. Tests
   usually state safety, determinism, and compatibility contracts precisely.
5. Open narrow ranges around matches. Several modules/tests have thousands of
   lines; reading them whole wastes context.
6. Trace CLI work from `src/spritelab/__main__.py`, then search `add_parser(`,
   `set_defaults(`, or `ProductCliRegistry` in the owning area.
7. For artifacts, find the schema/version string plus its writer, loader,
   validator, identity/hash code, and tests before changing fields.
8. Search `rmtree`, `unlink`, `replace`, `move`, `overwrite`, `resolve`, and
   `relative_to` whenever a change can affect files.

Normally skip ignored machine-local areas: `.git/`, tool caches,
`.pytest_tmp_*/`, virtual environments, build metadata, `artifacts/`, `data/`,
`data_sources/`, `datasets/`, `evals/`, `experiments/`, `generated/`,
`harvest_runs/`, `out/`, `outputs/`, `runs/`, and `.prefill_cache*/`.

## Source-of-truth order

When sources disagree, use this order and report material drift:

1. The current user request and this `AGENTS.md`.
2. Current code contracts and focused tests.
3. `pyproject.toml`, `.github/workflows/ci.yml`, and
   `spritelab.example.yaml` for tooling, CI, and configuration.
4. Feature-specific documents under `docs/`.
5. `CLAUDE.md` as a compact developer reference and `README.md` as an overview.
6. `PROJECT_BRAIN.md` only as historical milestone context; its early
   codec-only non-goals are no longer current.

Do not assume that v1, v2, product v3, Label v2/v3/v4, or Dataset-v5 documents
replace one another. These versions describe coexisting subsystem scopes.

## Repository map

| Work area | Start in source | Tests and docs |
|---|---|---|
| Root dispatch/composition | `src/spritelab/__main__.py`, `product_runtime.py` | CLI logging and product foundation tests |
| Guarded v3 workflow/developer evidence | `v3/`, `dev/`, `dev_features/`, `product_core/` | `test_v3_*`, `test_dev_cli_*`; `docs/v3_*.md`, `docs/developer_cli.md` |
| Local web product/plugins | `product_web/`, `product_features/`, `product_ux/` | `test_product_*`; `docs/product/`, `docs/product_*.md` |
| Intake/provenance/views/freezes | `product_features/dataset/`, `dataset_v5/`, `provenance/`, `unlabeled_pool/`, `suitability/` | Dataset-v5, provenance, pool, universal-intake tests; intake docs |
| Harvesting/semantic labeling | `harvest/`, `hierarchical_labeling/`, `annotation_scheduler/` | harvest, label, semantic, hierarchical tests; harvester/label/Qwen docs |
| Sprite representation/classic tools | `codec/`, `data/`, `curation/`, `dataset_maker/` | matching tests; `README.md`, curation/maker docs |
| Training/evaluation/compute | `training/`, `evaluation/`, `remote_compute/`, `ml/` | training, generator, memorization, remote, ML tests; training/v1/v2/evaluation/hosted docs |
| Config/web assets | `src/spritelab/config/`, `spritelab.example.yaml`, feature `templates/` and `static/` | config/state and product web tests |
| One-off utilities/examples | `scripts/`, `examples/` | Read an entire script before invoking; outputs are usually ignored |

The installed entry point is `spritelab` / `python -m spritelab`. Root command
families are `v3`, `dev`, `curation`, `train` (`training` alias), `harvest`,
`ml`, `eval`, and `dataset-maker`, plus legacy direct commands. Prefer `v3` for
the guarded product workflow and low-level commands for targeted backend work.

## Engineering conventions

- Python is `>=3.10` with a `src/` layout. Preserve Windows, macOS, and Linux.
- Keep CLI imports lazy so help/status does not initialize optional Torch, UI,
  provider, CUDA, or GPU dependencies.
- Argparse CLIs register handlers with `set_defaults`; keep registry dispatch.
- Reuse canonical JSONL, report, checkpoint, device, path, and identity helpers.
- Preserve deterministic ordering, stable schemas, SHA-256 bindings,
  immutable/fail-closed freezes, append-only evidence, and explicit review or
  authorization gates.
- Passive status/discovery/rendering must not contact providers, initialize
  CUDA, spend money, reveal secrets, or mutate inputs.
- Never persist credentials or expose absolute private paths in web/API errors
  and artifacts.
- SpriteBundle invariants: exact 32x32 sprites; palette/index slot 0 is
  transparent; opaque pixels use visible slots; metadata is JSON-serializable;
  canonicalization preserves decoded RGBA.
- Keep edits focused. Do not rewrite unrelated files or alter local/generated
  datasets to make tests pass.

## Tests and handoff

Start narrow and scale with risk:

```powershell
python -m pytest tests\test_relevant_file.py -q --basetemp=.pytest_tmp_<task> -p no:cacheprovider
python -m ruff check <changed paths>
python -m ruff format --check <changed paths>
```

- Before running any broad, release, or historically slow test suite, optimize
  the test path first. Inspect prior `--durations` output or profile one
  representative slow test; identify repeated hashing, process startup,
  fixture construction, polling, and full-tree scans; remove or safely reuse
  avoidable work; then benchmark the representative test again before
  launching the broad suite. Do not spend a full-suite run merely to discover
  a known performance problem.
- Test optimization must preserve the contract. Keep dedicated drift, tamper,
  confinement, cancellation, and live-reload tests on fresh/live state; cache
  or snapshot only inputs that the scenario declares immutable. A faster test
  is not acceptable if it can conceal a production change or weaken a gate.
- Never launch a monolithic broad suite when its measured or reasonably
  expected runtime can exceed 10 minutes. Inventory the test modules first,
  partition them deterministically into disjoint exhaustive shards, and fail
  closed on any overlap or gap. Balance historically slow modules separately;
  do not assume equal file counts mean equal workloads.
- Keep each shard below a 10-minute target and below the command/tool timeout.
  Cap native math-library threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS`) at `1` during parallel pytest work, and disable
  tokenizer parallelism unless the test specifically requires it. Give every
  worker unique process `TEMP`/`TMP` and `--basetemp` roots.
- Record each shard's module inventory, elapsed time, pass/fail/skip counts,
  failed node IDs, and reviewed skip reasons as soon as it completes. A slow
  worker must not hide already completed results. After a timeout, terminate
  only the verified process owned by that shard, preserve completed evidence,
  split the unfinished shard further, and resume only the uncovered portion;
  never repeat the same long monolith with a larger timeout.
- Use a unique basetemp for concurrent agents.
- `python -m spritelab dev test <quick|dataset|labeling|training|evaluation|full> --dry-run`
  shows curated profiles.
- Run `python -m mypy src` for broad/type-sensitive API changes.
- Run full pytest and repository-wide Ruff for cross-cutting/release changes;
  focused docs-only edits do not need ML tests.
- Tests are CPU-first. Do not weaken tests because optional dependencies or a
  GPU are unavailable.
- Training hot-loop changes retain sync-free behavior and A/B bit-identical
  validation.

Before handoff: inspect `git diff --check`, `git diff --stat`, the actual diff,
and `git status --short`; run relevant checks; confirm only intended files are
changed. Update the nearest docs when commands, schemas, configuration, safety,
or user-visible behavior changes.

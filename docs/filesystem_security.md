# Filesystem security

Sprite Lab handles local datasets, generated artifacts, resumable runs, and
remote training staging. Those paths may contain valuable untracked data. A
cleanup convenience must never widen into deletion of a project, profile,
drive, or remote workspace.

## Threat model

Filesystem mutations must account for:

- empty, relative, traversal, drive-relative, root, and unexpectedly normalized
  paths;
- a symbolic link, Windows junction/reparse point, or mount inserted at a
  managed path;
- a hard link planted at a predictable temporary filename so a normal write
  truncates an outside file;
- an incomplete export replacing the only valid previous export;
- a caller-constructed remote-compute record containing a different workspace
  or a traversal operation ID;
- failure cleanup that follows a path no longer owned by the operation.

The application does not assume ignored paths are disposable. Git status cannot
inventory ignored datasets or runs, so containment and ownership must be proved
independently.

Configuration discovery stops at the current Git repository boundary. A
`spritelab.yaml` in a user profile or unrelated parent project cannot redirect
writes from a repository that does not contain its own configuration. Explicit
environment overrides remain available when cross-boundary configuration is
intentional.

## Local guarantees

`spritelab.utils.safe_fs` owns the shared primitives:

- `require_confined_path(path, root)` checks lexical and resolved containment,
  refuses the root itself, and rejects existing descendant links and Windows
  reparse points.
- `remove_confined_tree(path, root)` is a compatibility retirement primitive,
  not recursive deletion: it rejects mounts/links, moves the exact owned child
  through a held parent to an unpredictable `.spritelab-retired-tree-*`
  residue, and retains every byte for explicit later cleanup review.
- `atomic_write_bytes` and `atomic_write_text` create unpredictable exclusive
  sibling files, flush them, and replace the destination entry atomically. They
  do not write through a predictable `.tmp` hard link.

Dataset Maker exports are built completely in a unique staging directory. An
overwrite moves the previous export to a unique confined backup, publishes the
staging tree, restores the previous export if publication fails, and retires
only the verified backup to recovery residue after success.

Dataset-v5 builders use fresh staging directories and refuse existing output
roots. Failure retirement is confined to the staging directory's parent and
retains a hidden recovery residue. Product
dataset intake confines its fixed `raw_extraction` candidate, previous, and
published directories to the managed output root. The product sidecar has a
stricter transaction-specific implementation that also guards lexical targets,
reparse seams, hard links, durability barriers, and recovery journals.

Downloads and local SSH artifact transfers use unpredictable exclusive partial
files. Final publication uses `os.replace`, which replaces a destination link
entry instead of truncating the file it references.

## Remote cleanup guarantees

`SSHComputeBackend` binds every `PreparedCompute` used for upload, launch, and
cleanup to:

- the `ssh` backend;
- the configured canonical remote workspace;
- a controlled operation identifier; and
- a concrete SHA-256 remote identity.

The remote cleanup script independently validates the operation identifier,
proves the resolved staging base remains directly below the configured
workspace metadata directory, requires the target's parent to equal that base,
rejects symlinks and non-directories, and removes only that one operation
directory. Run artifacts are outside this cleanup scope.

Remote marker creation and uploads use exclusive or unpredictable partial
names. Preparation rejects linked metadata, staging, and operation directories;
upload finalization rechecks staging and destination containment and rejects
linked or missing partial files.

## Required review pattern

For any new delete, overwrite, rollback, extraction, or remote-cleanup code:

1. Identify the approved root and exact owned child name.
2. Validate before reading or mutating the target.
3. Keep input roots disjoint from every run, staging, backup, and output root.
4. Use exclusive unpredictable staging; do not use a fixed `.tmp` or `.part`.
5. Validate the completed artifact before replacing the previous version.
6. Make rollback preserve the last valid version.
7. Add tests for root/traversal rejection, link/reparse escape, a hard-linked
   predictable-temp sentinel, injected failure, and outside-file preservation.
8. Search the production tree for new raw deletion primitives:

   ```powershell
   rg -n --glob '*.py' 'shutil\.rmtree|\.unlink\(|\.rmdir\(' src
   ```

Specialized code may retire a unique temporary file or remove an already
validated empty directory through an exact handle. New certified flows do not
perform recursive path deletion; unresolved cleanup remains explicit recovery
evidence for a separately authorized cleanup task.

## Verification

Run the focused security set with a unique basetemp:

```powershell
python -m pytest `
  tests/test_safe_filesystem.py `
  tests/test_dataset_maker_exporter.py `
  tests/test_remote_compute_ssh.py `
  tests/test_harvest_archive.py `
  tests/test_product_universal_dataset_intake.py `
  -q --basetemp=.pytest_tmp_security -p no:cacheprovider
```

Then run the relevant Dataset-v5 and product transaction tests, Ruff, mypy, and
the full test suite for cross-cutting changes.

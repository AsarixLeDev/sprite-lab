# Sprite Lab v3 project configuration

Sprite Lab uses one discoverable YAML file: `spritelab.yaml`. Commands search the current directory and then its parents. `spritelab.example.yaml` is the documented template.

## Sections

- `project`: human name and schema version (`3`).
- `paths`: run and artifact roots. Relative paths resolve from the configuration directory.
- `dataset`: authoritative provenance, raw inventory, extraction, suitability, candidate-view, and production-freeze artifacts.
- `labeling`: blind campaign, independent disagreement audit, and review queues.
- `training`: independent audit report, frozen audited-file hashes, campaign configuration, and dataset freeze identity.
- `evaluation`: checkpoint, benchmark, memorization audit, review log, and promotion decision.
- `execution`: explicit production permissions and existing backend adapter argument arrays.
- `reporting`: report preferences.

Unknown sections and keys are errors. Backend commands must be YAML lists so arguments remain distinct:

```yaml
execution:
  allow_training: false
  training_command:
  - python
  - -m
  - spritelab.training.cli
  - run-campaign
```

The adapter uses an argument array with `shell=False`. Configuration does not bypass backend manifests, resume identities, campaign gates, review integrity, or promotion policy.

## Defaults and overrides

All production permissions default to `false`:

```yaml
execution:
  allow_dataset_production_freeze: false
  allow_training: false
  allow_generation: false
  allow_promotion: false
```

Environment overrides are intentionally small:

- `SPRITELAB_CONFIG`: explicit configuration file.
- `SPRITELAB_PROJECT_ROOT`: explicit project root.
- `SPRITELAB_RUNS_DIR`: explicit run-state root.

The resolved paths, artifact hashes, source commit, run ID, and backend arguments remain visible in machine-readable state. Tokens and credentials do not belong in this file and are never printed.

`v3 init --dry-run` previews the target. `v3 init` uses exclusive creation and refuses to overwrite a file, including one that appears concurrently.

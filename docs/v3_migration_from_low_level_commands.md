# Migrating from low-level commands

Use the v3 surface for normal operations:

| Normal intent | v3 command |
|---|---|
| Understand the project | `python -m spritelab v3 status` |
| Build Dataset-v5 safely | `python -m spritelab v3 dataset build` |
| Plan or run training | `python -m spritelab v3 train` |
| Plan or run evaluation | `python -m spritelab v3 eval` |
| Find a review queue | `python -m spritelab v3 review` |

The v3 layer discovers paths from `spritelab.yaml`, derives identities from authoritative manifests, writes durable run state, and presents backend blockers in plain language. It does not reimplement Dataset-v5, training, evaluation, resume, or promotion algorithms.

Existing commands such as `training`, `train`, `eval`, `curation`, `dataset-maker`, `harvest`, and `ml` remain registered. They are appropriate for backend development, controlled remediation, and independent audit reproduction where an expert intentionally supplies low-level manifests and arguments.

Do not migrate by embedding a low-level shell command in a string. Configure an argument list only after the low-level workflow's own safety contract has been reviewed. The wrapper preserves `shell=False`, stdout/stderr logs, exact arguments, protected identities, and backend exit status.

Moving a command to v3 does not convert candidate data to a production freeze, proposals to calibrated truth, synthetic checks to audit certification, or reviewable evidence to promotion permission.

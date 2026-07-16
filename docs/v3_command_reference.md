# Sprite Lab v3 command reference

The mandatory module form is `python -m spritelab v3 <command>`. An installed package also exposes `spritelab v3 <command>`.

## Main workflow

| Command | Purpose | Normal safe stop |
|---|---|---|
| `dataset build` | Validate and orchestrate raw provenance, extraction, suitability, labeling, calibration, view construction, and freeze gates | Health failure, review, or missing freeze authorization |
| `train` | Validate dataset identity, audit, environment, campaign plan, confirmation, and safe backend execution | Missing freeze, failed/stale audit, policy, or confirmation |
| `eval` | Validate checkpoint/benchmark, generation, metrics, memorization review, and promotion gates | Missing identities, failed/stale review audit, or policy |

State-changing commands accept `--dry-run`. Training and evaluation never interpret a missing TTY as consent. Authorized noninteractive execution requires both `--yes` and `--non-interactive-confirm`.

## Project utilities

| Command | Purpose |
|---|---|
| `init` | Create `spritelab.yaml` safely; never overwrite it |
| `status` | Derive unified state from authoritative artifacts and hashes |
| `doctor` | Check environment, paths, artifacts, disk, Git, GPU visibility, and audit freshness without initializing CUDA |
| `resume [--run-id ID]` | Revalidate protected identities before delegating to a safe backend resume mechanism |
| `review` | Discover actionable review queues and the configured review interface |
| `report [--open]` | Show or open the latest static offline report |
| `runs [--limit N]` | List recent runs, status, elapsed time, resume state, and report availability |
| `logs [--run-id ID] [--follow]` | Read the selected or latest log; follow only active TTY sessions |
| `explain STAGE` | Explain status, dependencies, evidence, blockers, and exact next command |

Every command supports `--help`, `--json`, `--no-color`, `--quiet`, and `--debug`.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Completed successfully |
| 1 | Unexpected internal failure |
| 2 | Invalid configuration or usage |
| 3 | Mandatory project gate blocked execution |
| 4 | Human review required |
| 5 | Resumable pause or interruption |
| 6 | Stale or non-comparable evidence |
| 7 | Mandatory doctor/environment failure |

The JSON `exit_code` is identical to the process exit code. Human rendering and JSON consume the same typed result and project-state objects.

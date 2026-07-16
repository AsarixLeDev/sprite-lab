# Sprite Lab v3 run state and resume

Each state-changing command creates:

```text
runs/v3/<run-id>/
  state.json
  events.jsonl
  command.json
  logs/run.log
  artifacts/
  report/
  checkpoints/
```

Source datasets are referenced, not duplicated. `command.json` records the argument array, resolved configuration, project root, source commit, start time, and dry-run status. Secrets are not recorded.

`state.json` is written to a same-directory temporary file, flushed, and atomically replaced under a small cross-platform lock. `events.jsonl` is append-only and flushed after each event. Events include schema version, run ID, timestamp, command, stage, type, status, counts, message, and relevant artifact identity.

This state describes orchestration only. It does not replace campaign, checkpoint, dataset, evaluation, or review identities owned by backend code.

## Resume rules

`v3 resume` selects only incomplete runs explicitly marked resumable. Completed, failed, or blocked runs cannot be manufactured into success. Before continuation, it compares the protected source and backend identities with the current project.

- One resumable run: it is selected automatically.
- Several runs in a TTY: a compact numbered choice is shown.
- Several runs without a TTY: `--run-id` is mandatory.
- Changed identity: resume fails with exit code 6.
- No owning safe backend adapter: state remains paused; no generic subprocess is invented.

Use `v3 resume --dry-run` to validate identities without continuation, `v3 runs` to locate IDs, and `v3 logs --run-id ID` to inspect the preserved log.

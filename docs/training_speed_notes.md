# Training Speed Notes — `generator-challenger` / `audit-challenger-full-v4`

Findings from profiling the full-v4 challenger audit's 25,000-step training
loop and adding opt-in speed flags. Every flag below defaults to today's exact
numeric behaviour — see [[docs/v2_phase1_conditioning.md]] for the same
default-off convention applied to architecture flags.

## 1. TL;DR

Measured with `tools_benchmark_train_step.py` against the archived
`experiments/challenger_full_v4_v2_phase1_conditioning/train_25k/config.json`
run (RTX 5060 Ti, torch 2.11.0+cu128, batch 32, `--amp` on, 4 dataloader
workers), 400 measured steps + 30 warmup steps per variant:

| Variant | ms/step | it/s | vs. baseline |
|---|---:|---:|---:|
| baseline (today's defaults) | 61.56 | 16.24 | — |
| `--metrics-every 25` alone | 46.09 | 21.70 | -25.1% |
| fast EMA path (unconditional, no flag) | 41.35 | 24.18 | -32.8% |
| `--fused-adamw` alone | 43.57 | 22.95 | -29.2% |
| `--cudnn-benchmark` alone | 44.87 | 22.29 | -27.1% |
| `--tf32` alone | 44.90 | 22.27 | -27.1% |
| all combined (`all_on`) | 38.13 | 26.23 | **-38.1%** |

Recommended command for a full 25k run:

```powershell
python -m spritelab train audit-challenger-full-v4 `
  ... `
  --metrics-every 25 `
  --fused-adamw `
  --cudnn-benchmark `
  --tf32 `
  --num-workers 4
```

(`--num-workers 4` is already the archived run's value; see [§6](#6-measured-results) —
`--num-workers 0` measured *slower* on this dataset, so there is nothing to
change there.)

At the measured -38.1% per-step reduction, a 25,000-step training loop drops
from roughly 25.7 min to roughly 15.9 min on this machine today; see
[§2](#2-a-caveat-on-absolute-numbers-environmental-gpu-contention) for why
the archived run's own historical figure (40.7 ms/step, ~17 min total) sits
*between* the "all flags off" and "all flags on" numbers above rather than
matching either exactly, and why that doesn't change the recommendation.

## 2. Measure before optimizing

The archived 25k run's `train_metrics.jsonl` already had per-step
`elapsed_seconds`, so attribution was nearly free:

| Phase | Time | Share |
|---|---:|---:|
| Training loop (25,000 steps) | ~1018 s | ~93% |
| Setup + initial full-train-set loss eval | ~17 s | ~1.6% |
| Final train + val loss evals | ~10-20 s | ~1.5% |
| Sampling/audits (eval + OOD prompts, sensitivity, faithfulness, QA, export) | ~20 s | ~2% |

Per-step dt: median 40.7 ms (24.6 it/s), p10 39.2 ms, p90 43.3 ms, uniform from
step 0 through step 20,000+ — no epoch-boundary stalls, no warmup ramp. That
ruled out the audits and the dataloader as suspects before writing a single
line of optimization code: **the step loop was the only thing worth
optimizing**, and its tight, un-ramping distribution said the cost was
per-step overhead, not intermittent stalls.

### A caveat on absolute numbers: environmental GPU contention

`tools_benchmark_train_step.py`'s `baseline` variant, which reproduces
today's exact code path, measured 61.56 ms/step here — 1.5x the archived
40.7 ms/step reference, even after fixing two harness bugs (orphaned
persistent-worker processes contending for the GPU across variants, and an
accidental explicit `fused=False` that disabled torch's own default
optimizer auto-selection; both described in [§4](#4-anti-patterns-found)).
`nvidia-smi` showed 8-10 other processes holding GPU contexts throughout
(browser tabs, Discord, Steam, Wallpaper Engine, NVIDIA overlay) and
utilization sitting at 40% even during the benchmark's own steady state —
consistent with Windows' shared WDDM scheduler adding real per-kernel-launch
latency when interleaving with other GPU clients, which hits a
launch-overhead-bound workload (this one) hardest. The relative deltas
between variants, measured back-to-back under the same contention, are the
trustworthy numbers here; the absolute floor an uncontended machine would hit
is closer to (or better than) the archived 40.7 ms/step, since every
single-flag variant above already lands at or below it.

## 3. The overhead-bound regime

A ~5-10M-parameter U-Net at batch 32 on 32x32 images should not need 40+ ms
per step on this GPU if it were purely compute-bound. Signs this is instead
launch/sync-bound:

* GPU utilization plateaus around 40% during steady-state training, not
  90-100% — the GPU is idle waiting on the CPU between kernel launches more
  than it is saturated.
* Two *independent* changes (dropping metrics sync, or replacing the loop-
  based EMA update) each cut ~25-33% off the step time on their own, and
  combining more of them keeps helping — a compute-bound step would not
  respond this way to changes that touch zero FLOPs.
* The two costs share a mechanism: `float(loss.detach().cpu())` forces a
  device sync every step, and the old EMA loop issued ~2 tiny kernel launches
  per model tensor *per step* from Python — both serialize the CPU behind
  the GPU queue instead of letting the CPU race ahead and keep the queue full.

## 4. Anti-patterns found

* **Per-step forced device sync for logging**
  ([generator_challenger.py:803](../src/spritelab/training/generator_challenger.py:803),
  historically unconditional every step): `float(loss.detach().cpu())` plus
  building a full loss-components dict blocks the CPU on the GPU finishing
  the current step before it can enqueue the next one's kernels. Measured
  cost: ~15.5 ms/step (`baseline` 61.56 -> `no_metrics_sync` 46.09).
* **Rebuilding the EMA update from a fresh `state_dict()` every step, one
  tensor at a time** ([generator_challenger.py:1760](../src/spritelab/training/generator_challenger.py:1760)):
  the old `_update_ema_state` looped over every named tensor in the model and
  issued `mul_`/`add_`/`copy_` as separate kernel launches per tensor, all
  from Python. Measured cost: ~15.8-20.2 ms/step depending on what else was
  already removed (`baseline` -> `no_ema` or -> `ema_foreach`).
* **Plain `AdamW` without a fused kernel path**
  ([generator_challenger.py:704](../src/spritelab/training/generator_challenger.py:704)
  now takes `--fused-adamw`): ~18 ms/step (`baseline` -> `fused_adamw`,
  43.57 ms/step) — bigger than the single-digit estimate that motivated
  looking at it in the first place.
* **No `cudnn.benchmark` for fixed conv shapes**
  ([generator_challenger.py:708](../src/spritelab/training/generator_challenger.py:708)
  now takes `--cudnn-benchmark`): with a constant 32x32/batch-32 shape
  through the entire run, letting cuDNN search once for the best algorithm
  measured ~14-17 ms/step, larger than expected for a model this size —
  likely because FiLM conditioning and bottleneck attention add conv/matmul
  shapes beyond the plain U-Net path this project started with.
* **Benchmark-harness-specific bug (not in production code): explicit
  `fused=False`**. The first version of `tools_benchmark_train_step.py`
  called `torch.optim.AdamW(params, lr=lr, fused=fused_adamw)` for every
  variant, including "baseline". Passing `fused=False` *explicitly* is not
  the same as omitting the kwarg (`fused=None`, torch's real default):
  passing `False` explicitly measured 62 ms/step across the board for every
  non-`fused_adamw` variant, while the *actual* production call (no `fused`
  kwarg at all, exercised by `optim_utils.build_adamw(..., fused=False)`,
  which omits the kwarg rather than passing `False`) does not have this
  penalty. Fixed by routing the benchmark's optimizer construction through
  the same `build_adamw` helper production uses. Lesson: an explicit
  `False` and an implicit default are not always interchangeable in
  optimizer constructors that auto-select internals from `None`.
* **Benchmark-harness-specific bug: orphaned persistent DataLoader
  workers on Windows**. `persistent_workers=True` (production's own
  `dataloader_perf_kwargs`) keeps worker processes alive until every
  reference to the DataLoader/iterator is dropped and garbage-collected;
  an unclosed infinite generator wrapping `for batch in loader` left worker
  processes (and sometimes the main process) alive *after* the benchmark
  script had already printed its results and returned control, contending
  for the GPU with the next invocation and inflating its numbers. Fixed with
  an explicit `close()`/`del`/`gc.collect()` per variant plus a defensive
  `multiprocessing.active_children()` sweep at the end of `main()`. Not a
  production code path (production builds its loaders once and keeps
  running until the process really does exit), but worth knowing if this
  benchmark script is reused for shorter probes later.

## 5. Transferable checklist

For similarly small/fast models on a fast GPU:

* Sync and log on a cadence, not every step — but **always sync the final
  step** so `last_step_loss`/final metrics stay exact.
* Cache tensor references once (e.g. from a single `state_dict()` call) in
  hot loops instead of re-deriving them every iteration — safe as long as
  the underlying params are mutated in place and never reassigned.
* Prefer `torch._foreach_*` (or a fused kernel) over a Python loop issuing
  one kernel launch per tensor.
* Turn on `cudnn.benchmark` when input shapes and batch size are fixed for
  the whole run.
* bf16 `autocast` needs no `GradScaler` (unlike fp16) — one fewer moving
  part when adding mixed precision.
* `pin_memory` + `non_blocking=True` transfers, and `persistent_workers` on
  Windows (process spawn is expensive there) — already in place via
  `optim_utils.dataloader_perf_kwargs`.
* Benchmark `--num-workers 0` vs. workers explicitly when samples are
  small enough to be fully RAM-cached — worker IPC can be a net loss. (Not
  the case here: measured `--num-workers 0` at 64.06 ms/step, *slower* than
  4 workers' 61.56 ms/step.)
* Check batch-size headroom separately from per-step latency — it changes
  training dynamics, so it's a probe, not a numerics-identical flag (`batch64`
  measured 59.98 ms/step for **double** the batch, i.e. ~2.05x samples/s).
* Remember CFG dropout means every "conditional" step already does one
  forward pass, not two — don't double-count when estimating sampling-time
  CFG cost against training-time cost.

## 6. Measured results

Full sweep, `tools_benchmark_train_step.py --steps 400 --warmup 30` against
the archived config (see [§1](#1-tldr) for the summary table):

| Variant | ms/step | it/s | samples/s | batch | workers |
|---|---:|---:|---:|---:|---:|
| baseline | 61.56 | 16.24 | 519.8 | 32 | 4 |
| no_metrics_sync | 46.09 | 21.70 | 694.3 | 32 | 4 |
| no_ema | 45.76 | 21.85 | 699.3 | 32 | 4 |
| ema_foreach | 41.35 | 24.18 | 773.9 | 32 | 4 |
| fused_adamw | 43.57 | 22.95 | 734.4 | 32 | 4 |
| cudnn_benchmark | 44.87 | 22.29 | 713.1 | 32 | 4 |
| tf32 | 44.90 | 22.27 | 712.7 | 32 | 4 |
| workers0 | 64.06 | 15.61 | 499.5 | 32 | 0 |
| batch64 | 59.98 | 16.67 | 1067.0 | 64 | 4 |
| all_on | 38.13 | 26.23 | 839.2 | 32 | 4 |

`all_on` = `--metrics-every` (simulated via skipped sync in the benchmark) +
fast EMA + `--fused-adamw` + `--cudnn-benchmark` + `--tf32`, at `--num-workers 4`
/ `--batch-size 32` (unchanged) since those two are throughput probes, not
part of the numerics-identical flag set.

Correctness verification (all passing):

* `tests/test_training_speed_options.py` — exact `torch.equal` EMA-path
  equivalence (including an injected non-floating-point tensor), `--metrics-every`
  logs only synced steps plus always the final step, `--eval-max-batches`
  produces a well-formed report, `build_adamw`/`apply_backend_speed_flags`
  are no-ops at their defaults, and a CPU run comparing two
  `ChallengerTrainConfig`s (bare dataclass defaults vs. every new field set
  explicitly to its default) produces **bit-identical** `train_metrics.jsonl`
  content field-for-field (excluding wall-clock `elapsed_seconds`).
* Full project test suite: 1153 passed, 0 failed, 0 regressions.

## 7. Deliberately not done

* **`torch.compile` / CUDA graphs.** Windows + Triton + sm_120 (Blackwell) is
  the least-tested combination for torch 2.11; `reduce-overhead` mode (CUDA
  graphs) is the theoretically correct fix for a launch-overhead-bound model
  this small, but it captures a fixed sequence of kernel launches against
  fixed tensor storage, which fights the in-place EMA update on live param
  storage and the CFG-dropout branching (a different set of kernels gets
  launched depending on the dropout draw) in this training loop. If this
  becomes worth revisiting: try `mode="reduce-overhead"` on just the U-Net
  forward (not the EMA update or the loss), pin the CFG-dropout draw to a
  per-graph-capture-compatible form (e.g. always compute both branches and
  select, rather than branching), and validate on Linux/CUDA first to
  separate "doesn't work on this stack" from "doesn't work at all."
* **Eval-pass micro-optimizations.** `_evaluate_challenger_losses` is ~2-3%
  of total wall time (see [§2](#2-measure-before-optimizing)); `--eval-max-batches`
  exists as a knob for when a *smaller* estimate is acceptable, but the full
  pass was never the bottleneck and wasn't touched otherwise.
* **Changing any default** for the two training subcommands above. Every
  flag on `generator-challenger`/`audit-challenger-full-v4` ships off; this
  project's determinism/reproducibility philosophy (see `optim_utils.py`'s
  own docstring convention) treats "the training run you get without asking
  for anything new" as a contract, and archived runs/checkpoints should stay
  reproducible from their `config.json` without needing to know which speed
  flags happened to be the default at the time.

## 8. Exception: `run-v2-phase0-eval` defaults its backend flags ON

`run-v2-phase0-eval` never trains (no optimizer, no EMA, no `train_metrics.jsonl`
contract to keep reproducible) — it only repeatedly calls
`run_sample_generator_challenger` against the *same* checkpoint at the *same*
fixed shape across many preset/ablation cells and seeds. That's exactly the
profile `cudnn.benchmark`'s one-time algorithm search amortizes well over, so
its `--speed-optimizations` flag (`cudnn_benchmark` + `tf32` via
`optim_utils.apply_backend_speed_flags`) defaults to **on**, unlike every
training-loop flag above. `--no-speed-optimizations` restores the plain path
if a bitwise-exact-with-older-runs comparison is ever needed. `metrics_every`,
the fast EMA path, `--fused-adamw`, and `--eval-max-batches` don't apply here
at all — there's no training step, optimizer, or EMA state to speed up.

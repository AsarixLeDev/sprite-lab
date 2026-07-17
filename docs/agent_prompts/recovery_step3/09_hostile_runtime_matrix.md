# Independent hostile runtime matrix

Run only after every step-3 production writer is quiescent. Follow the common
safety preamble in `README.md`.

This is an adversarial test/audit lane. Do not edit production code. Add only
new uniquely named hostile test modules. If a test exposes a defect, report it
to the owning implementation lane and stop.

Cover project-source and descriptor ABA; runtime-root/native/resource drift;
same-prior-state concurrent writers; COMPLETE/CANCELLED/TIMED_OUT races; every
report/event/state interruption; stale same-sequence CAS; execution of all real
bootstraps; wrong-hash/preloaded-module no-execution markers; Windows one-
process Job no-spawn with outside sentinels; Linux's actual process-group/
parent-death guarantees without overstating Landlock; dependent DSO/resource
final rehash; exact documented residuals; cancellation during scans, lock wait,
checkpoint copy, report read, inventory, and final publication; and passive
status/reconstruction with no Torch import, subprocess, provider call, or
mutation.

Use platform skips only for genuinely unavailable mechanisms. Test a normal
Medium Windows token outside the inherited restricted Codex shell and
separately prove the restricted caller refuses before native token derivation.
Return exact hashes, commands, results, and counterexamples. Do not certify.


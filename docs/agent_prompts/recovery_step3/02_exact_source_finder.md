# Verify exact early project-source loading

Follow the common safety preamble in `README.md`.

Exclusive production file: `product_features/evaluation/playground_worker.py`.
Exclusive new tests: `tests/test_smoke_source_finder_step3.py` or a new
non-colliding successor module. Do not edit `smoke_bundle.py`,
`local_generator.py`, smoke runner/worker, or existing large test modules.

Prove that no project `spritelab` module, including `spritelab.__init__`, can
execute before a finder bound to the already verified project inventory is
installed. Reject preloaded project modules, late installation, source
substitution after spec selection, unexpected loaders, namespace/package
escape, and post-import drift. The bootstrap must not import unverified project
helpers to install its own policy.

Preserve exact native/runtime closure checks and pathless diagnostics. Run
focused hostile tests, Ruff, format-check, and `git diff --check`.


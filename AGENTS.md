# Working on Faster-Motus

This repository is upstream Motus plus host-side training-speed changes behind `MOTUS_OPT_*` switches, the selected
preset (`.harness/config.json`, `scripts/faster-motus.env.sh`: DS_HOOK + LANG_MMAP), validation tools and documentation.
The research harness, its judge, datasets and checkpoints are separate artifacts. Do not copy a machine's workspace,
weights, data, run records or caches into the repository.

Read [docs/agent-environment.md](docs/agent-environment.md) before GPU work and
[skills/validate-faster-motus/SKILL.md](skills/validate-faster-motus/SKILL.md) before claiming a speedup or numerical
equivalence on new hardware.

Keep upstream's recipe: 8 GPUs per node, DeepSpeed ZeRO-1, bf16, batch 8 per GPU, 16 workers, upstream learning rate
and schedule. A change belongs behind a new `MOTUS_OPT_*` switch, default off, and must not change numerics: with
`DET=1`, a run with the switch must be bitwise identical to a run without it (`tools/compare_runs.py`). Keep
`MOTUS_OPT_LOADER_SPEC` off: it prefetches two epochs into pinned host memory without bound and can leave GPU-holding
workers behind on a hard exit (docs/full-data-results.md).

Use an isolated environment and fresh output directories. Check that GPUs are idle, honor scheduler and locking rules,
and never kill another user's job; after an abnormal exit, kill only your own leftover processes. Do not start full
training unless requested. Validate code changes with `python -m pytest tests/` and a bounded run. Publish only code,
instructions and selected aggregate results; never commit credentials, weights, datasets or raw run records.

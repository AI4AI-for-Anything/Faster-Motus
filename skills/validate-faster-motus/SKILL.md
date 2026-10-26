---
name: validate-faster-motus
description: Check Faster-Motus against the baseline (upstream Motus, no switches) on a new machine - bitwise numerics of whole training runs and time per iteration - with the repository's tools. Use before claiming equivalence or a speedup on new hardware or library versions; not for convergence or quality evaluation.
---

# Validate Faster-Motus on a machine

Read [the environment and assets guide](../../docs/agent-environment.md) first. You need the converted RoboTwin 2.0
data, the three model directories, a baseline checkout (upstream Motus at the matching commit: `git clone https://github.com/thu-ml/Motus
../motus-upstream && git -C ../motus-upstream checkout 88f6fe3`) and 8 idle GPUs on one node.

1. CPU checks: `python -m pytest tests/`.
2. Bitwise runs, deterministic mode, the same steps for both (cross at least one epoch boundary: 532 iterations on the
   full data; validation runs every 500):

   ```bash
   export DATA=... WAN=... QWEN=... MOTUS_CKPT=...
   OUT=runs/det_baseline TREE=../motus-upstream SWITCHES="" STEPS=1100 DET=1 bash tools/run_validation.sh
   OUT=runs/det_faster STEPS=1100 DET=1 bash tools/run_validation.sh          # the preset
   python tools/compare_runs.py runs/det_baseline runs/det_faster --json runs/compare.json
   ```

   Require `bitwise over 8 ranks: True`: every step, the validation metrics and the final parameters, master weights
   and Adam moments equal on every rank. A difference at step 1 with equal batch digests means the two runs did not
   use the same mode or environment; a first difference later localizes the problem.
3. Timing, default mode (leave `DET` unset), on the same machine: the baseline for 150 iterations, the preset for 650
   (one validation and one epoch boundary). Run `bash tools/mem_sample.sh <run dir> > <run dir>/mem.log` alongside to
   record host memory. Compare the mean host time per iteration over iterations 21-150 of each run's
   `rank0.jsonl` (`iter_s`); `tools/compare_runs.py` prints each run's overall mean and steady median.

Deterministic mode is slower and only for step 2. Report hardware, driver, library versions, both commits, the
switches, the comparison result and the timing table, as in [docs/full-data-results.md](../../docs/full-data-results.md).
Bitwise training over a bounded run does not replace a quality evaluation of a full training run.

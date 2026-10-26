# Release boundary

`Faster-Motus` contains upstream Motus with the agent's training-speed changes, the selected preset, tools and
instructions to validate them on new hardware, and selected aggregate results. Every change is behind a
`MOTUS_OPT_*` environment switch; with no switch set, the training code runs upstream's code paths.

The research harness that produced the changes is not part of this repository: the task formulation, the frozen judge
(benchmark fixture, timing protocol, bitwise and trajectory evaluators), the optimizer's work log, and the experiment
records. Faster-Motus does not depend on it to train or to validate.

The interface between them is a source snapshot (per-file hashes of the judged snapshot in
[`source-snapshot.json`](source-snapshot.json)), an environment mapping ([`.harness/config.json`](../.harness/config.json)),
the training config and machine-local asset paths.

The tools here (`tools/run_validation.sh`, `tools/fullrun.py`, `tools/compare_runs.py`) compare whole training runs
bitwise on the hardware they run on. A PASS on one machine does not establish equivalence on other GPUs, other
library versions, other world sizes or multi-node runs; repeat the comparison there.

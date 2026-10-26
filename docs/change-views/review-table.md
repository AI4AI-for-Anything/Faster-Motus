# What changed in Faster-Motus

File-by-file review against upstream Motus `88f6fe3`; this table and changes.json are not listed. The [module map](module-map.svg) summarizes the code changes. Regenerate this table with `python3 docs/change-views/generate.py` and the map with `python3 docs/performance-views/render.py`.

**Attribution:** line counts describe the diff, not runtime benefit. The measured gains of the four switches are in [the optimization history](../performance-views/history.json) and [full-data results](../full-data-results.md).

## Agent optimization (judged snapshot)

**3 changed paths · +449 / −1 lines** (binary files not counted).

| Module | File | Change | Purpose |
|---|---|---:|---|
| Training loop | [train/train.py](../../train/train.py) | M +4 / −0 | Imports utils/fastpath.py; calls apply_all() at import (DS_HOOK) and before_train_loop() right before the first data iterator (loader switches). |
| Fast paths | [utils/fastpath.py](../../utils/fastpath.py) | A +423 / −0 | DS_HOOK: per-graph-task cache of DeepSpeed's used-parameter count. LOADER_REUSE: persistent data-loader workers re-seeded like fresh ones. LOADER_SPEC: speculative next-epoch worker pool. MOTUS_DIAG_LOADER: a diagnostic loader watchdog. |
| Data | [data/robotwin2/robotwin_agilex_dataset.py](../../data/robotwin2/robotwin_agilex_dataset.py) | M +22 / −1 | LANG_MMAP: memory-mapped umt5 embedding lists, cached per worker; marks the dataset index-free for LOADER_SPEC. |

## Release: preset, validation tools, tests and documentation

**29 changed paths · +3223 / −3 lines** (binary files not counted).

| Module | File | Change | Purpose |
|---|---|---:|---|
| Preset | [.harness/config.json](../../.harness/config.json) | A +6 / −0 | The selected switches as an environment mapping: DS_HOOK + LANG_MMAP. |
| Preset | [scripts/faster-motus.env.sh](../../scripts/faster-motus.env.sh) | A +8 / −0 | Shell version of the preset; lists the two switches it leaves out. |
| Agent workflow | [AGENTS.md](../../AGENTS.md) | A +21 / −0 | Rules for agents: upstream recipe, switches default off, bitwise checks, LOADER_SPEC off, shared-GPU etiquette, what never to commit. |
| Agent workflow | [skills/validate-faster-motus/SKILL.md](../../skills/validate-faster-motus/SKILL.md) | A +33 / −0 | Checklist to compare Faster-Motus with upstream bitwise and in time per iteration on a new machine. |
| Validation tools | [tools/compare_runs.py](../../tools/compare_runs.py) | A +91 / −0 | Bitwise comparison of two recorded runs, rank by rank; timing summary. |
| Validation tools | [tools/fullrun.py](../../tools/fullrun.py) | A +187 / −0 | Runs train/train.py unchanged under torchrun with per-rank seeds and optional determinism; records every step and the final state digests. |
| Validation tools | [tools/mem_sample.sh](../../tools/mem_sample.sh) | A +21 / −0 | Samples host memory while a run trains. |
| Validation tools | [tools/run_validation.sh](../../tools/run_validation.sh) | A +46 / −0 | Launches one recorded run with the config's paths, steps and switches filled in. |
| Data preparation | [tools/convert_robotwin_full.py](../../tools/convert_robotwin_full.py) | A +107 / −0 | Runs upstream's converter on all 100 RoboTwin 2.0 aloha-agilex archives in parallel. |
| Data preparation | [tools/encode_t5_sharded.py](../../tools/encode_t5_sharded.py) | A +117 / −0 | Upstream's umt5 encoder over the converted data, shared by any number of GPU workers. |
| Tests | [tests/test_ds_hook.py](../../tests/test_ds_hook.py) | A +67 / −0 | CPU: cached count equals DeepSpeed's per-backward maximum, per graph task and parameter list. |
| Tests | [tests/test_lang_mmap.py](../../tests/test_lang_mmap.py) | A +52 / −0 | CPU: mapped umt5 list equals torch.load's list bitwise; cached per worker. |
| Results | [docs/performance-views/01-benchmark.png](../../docs/performance-views/01-benchmark.png) | A binary | PNG copy of the SVG figure of the same name. |
| Results | [docs/performance-views/01-benchmark.svg](../../docs/performance-views/01-benchmark.svg) | A +163 / −0 | README headline figure: baseline vs Faster-Motus time per iteration (PNG copy alongside). |
| Results | [docs/performance-views/02-history.png](../../docs/performance-views/02-history.png) | A binary | PNG copy of the SVG figure of the same name. |
| Results | [docs/performance-views/02-history.svg](../../docs/performance-views/02-history.svg) | A +323 / −0 | Cumulative speedup over the agent's rounds R01-R04 on the benchmark fixture (PNG copy alongside). |
| Results | [docs/performance-views/full-data-benchmark.json](../../docs/performance-views/full-data-benchmark.json) | A +80 / −0 | Full-data timing and memory measurements behind the README table and figures. |
| Results | [docs/performance-views/history.json](../../docs/performance-views/history.json) | A +42 / −0 | The agent's rounds R01-R04 on the benchmark fixture, with their timing screens and the final full check. |
| Results | [docs/performance-views/render.py](../../docs/performance-views/render.py) | A +121 / −0 | Draws the README figures and the module map from the two records (matplotlib, no GPU). |
| Documentation | [README.md](../../README.md) | M +67 / −2 | Faster-Motus overview, results and switches; upstream README kept below. |
| Documentation | [README.zh-CN.md](../../README.zh-CN.md) | A +266 / −0 | Chinese version of the README. |
| Documentation | [docs/agent-environment.md](../../docs/agent-environment.md) | A +103 / −0 | Validated stack, assets, full-data conversion, NCCL fix, upstream converter workarounds and troubleshooting. |
| Documentation | [docs/change-views/generate.py](../../docs/change-views/generate.py) | A +156 / −0 | These change views and their generator. |
| Documentation | [docs/change-views/module-map.png](../../docs/change-views/module-map.png) | A binary | PNG copy of the module map. |
| Documentation | [docs/change-views/module-map.svg](../../docs/change-views/module-map.svg) | A +181 / −0 | Cards summarizing the four code changes (drawn by render.py; PNG copy alongside). |
| Documentation | [docs/full-data-results.md](../../docs/full-data-results.md) | A +96 / −0 | Full-data setup, bitwise results, timing of every switch set, why LOADER_SPEC is off by default, and limits. |
| Documentation | [docs/release-boundary.md](../../docs/release-boundary.md) | A +17 / −0 | What this repository contains and what stays in the research harness. |
| Documentation | [docs/source-snapshot.json](../../docs/source-snapshot.json) | A +846 / −0 | Per-file SHA-256 of the judged source snapshot. |
| Repository | [.gitignore](../../.gitignore) | M +6 / −1 | Keeps the rendered charts despite upstream's *.png rule; ignores local runs. |

A = added; M = modified. Upstream code is otherwise unchanged; every new code path is behind a `MOTUS_OPT_*` switch that defaults to off.

# Full-data results

Faster-Motus was validated on the **full RoboTwin 2.0 data** with upstream's training recipe on **8 GPUs**:

- **Numerics:** 1,100 deterministic training iterations are **bitwise identical** to the baseline (upstream Motus, no switches) on all 8 ranks.
  This covers every iteration's losses, learning rates and batches, two epoch boundaries, two validations, and the
  final parameters, fp32 master weights and Adam moments.
- **Speed:** the preset is **9.59× faster per iteration** (7.468 → 0.778 s).

## Setup

| | |
|---|---|
| Hardware | one node, 8 GPUs (NVLink), 2.9 TB host RAM, driver 580.126.09 |
| Software | Python 3.10, PyTorch 2.7.1+cu128 with nvidia-nccl-cu12 2.27.3, DeepSpeed 0.18.5, accelerate 1.12.0, transformers 5.0.0rc0, flash-attn 2.8.3 |
| Recipe | upstream `scripts/train.sh`: torchrun with 8 ranks, `train/train.py --deepspeed configs/zero1.json --config configs/robotwin.yaml` (DeepSpeed ZeRO-1, bf16, batch 8 per GPU, 16 data-loader workers per GPU, validation every 500 iterations) |
| Model | Motus: Wan2.2-TI2V-5B video expert, frozen Qwen3-VL-2B, action and understanding experts; 8.0B parameters, 5.9B trainable; fine-tuned from the released Stage-2 checkpoint |
| Data | RoboTwin 2.0, all 50 tasks: every aloha-agilex clean (50) and randomized (500) demonstration (27,500), converted with upstream's converter (27,244 kept after its qpos check) and umt5-embedded with upstream's T5 code. Dataset length 272,440, **532 iterations per epoch** |
| Config | `configs/robotwin.yaml` with only the machine paths, `training.max_steps` and the checkpoint directory changed |

Each run used `tools/run_validation.sh`, which runs the tree's own `train/train.py` under `tools/fullrun.py`. That
wrapper seeds rank r's Python, numpy and torch streams with a fixed seed + r; upstream never seeds, so two unseeded runs
draw different data and noise. It records every iteration on every rank and changes nothing in the training.
`tools/compare_runs.py` compares two runs.

## Numerical equivalence

The bitwise runs use deterministic mode: torch's deterministic algorithms, and flash-attn's deterministic backward
(upstream's flash-attn backward accumulates with atomics, so even two upstream runs differ without it).

| Comparison | Iterations | Result |
|---|---:|---|
| Baseline vs Faster-Motus preset (DS_HOOK + LANG_MMAP) | 1,100 | **bitwise identical** on 8 ranks |
| Baseline vs all four switches | 1,100 | **bitwise identical** on 8 ranks |
| Baseline vs all four switches, benchmark fixture (9-iteration epochs) | 12 | bitwise identical on 8 ranks |
| Negative control: deterministic vs default-mode baseline | 150 | losses differ from iteration 1 (same batches) |

"Bitwise identical" covers, on every rank:
- every iteration's loss dict, both learning rates and a digest of the batch;
- the validation metrics at iterations 500 and 1,000;
- at the end, the SHA-256 of all 1,988 model parameters, of the ZeRO fp32 master partitions, and of the Adam moments.

The 1,100 iterations include the end of the 200-iteration warmup and two epoch boundaries (iterations 533 and 1,065).
The baseline and the all-switch run ran on one machine; the preset run ran on a second machine of the same type.

During the agent's optimization, the task's judge also verified bitwise equality on its benchmark fixture: 87,328
recorded tensors over 8 ranks × 2 iterations, in deterministic mode. It also compared 100-iteration loss trajectories.

## Speed

These runs use upstream's default (non-deterministic) mode. The time is rank 0's host time between the starts of
consecutive iterations, so it includes any wait for data. All runs were on one machine.

| Configuration | Mean, iterations 21–150 (s) | Mean, iterations 21–650 (s) | Validation iteration (s) | Epoch-boundary iteration (s) | Speedup (21–150) |
|---|---:|---:|---:|---:|---:|
| Baseline | 7.468 | — | — | — | 1.00× |
| **Faster-Motus preset** (DS_HOOK + LANG_MMAP) | **0.778** | **0.813** | 17.3 | 7.1 | **9.59×** |
| Preset + LOADER_REUSE | 0.783 | 0.808 | 17.1 | 5.4 | 9.54× |
| All four switches (+ LOADER_REUSE, LOADER_SPEC) | 1.799 | 1.097 | 16.3 | 5.9 | 4.15× |

The baseline run was 150 iterations, so its window has no validation and no epoch boundary. Steady iterations of the
preset after iteration 150 average 0.776 s; including one validation and one epoch boundary, iterations 21–650
average 0.813 s, 9.19× faster than the baseline's 7.468 s.

In deterministic mode, which is slower and used only for the comparisons above, iterations excluding validation and
epoch boundaries took: baseline 7.733 s; all four switches 1.250 s; preset 0.972 s (that run was on the other
machine).

## Why the preset leaves out LOADER_SPEC

The agent optimized on a benchmark fixture of 496 episodes. There, accelerate's re-sharding of upstream's
DistributedSampler loader makes an epoch only 9 iterations long, and the loader's restart at every epoch (~4.4 s)
was a large share of the time. LOADER_REUSE and LOADER_SPEC remove that restart. On the full data an epoch has 532
iterations, so the restart costs about 1%, and LOADER_SPEC has two problems at this scale:

- **No backpressure.** At each epoch start, LOADER_SPEC's pool submits every batch of the epoch plus a speculative copy
  of the next epoch. The workers produce all of them ahead of the training, and the pin-memory thread keeps them in
  pinned host memory, so the memory it needs grows with the epoch length. While the workers produce this burst, the training processes are starved of CPU, so the first ~100 iterations of an
  epoch run 2–3× slower.
- **Workers left behind on a hard exit.** The pool's worker loop never calls `result_q.cancel_join_thread()`, which
  torch's own worker loop does. If the training processes exit abruptly (SIGKILL, an NCCL abort, `os._exit`), a
  worker that still has results to send blocks forever at exit. Forked after CUDA initialization, it keeps that GPU's
  memory allocated until it is killed.

LOADER_REUSE is safe (it uses torch's worker loop) but saves only ~1.7 s per epoch boundary (~0.3%). Both switches
stay in the source, off by default.

## Not covered

- **Quality:** no full training run and no RoboTwin evaluation. Bitwise-identical training over 1,100 iterations in
  deterministic mode means the same computations. In default mode, the runs differ only the way two baseline runs
  differ from each other, through non-deterministic kernels.
- **Other setups:** multi-node, other GPUs and other DeepSpeed versions were not tested. DS_HOOK patches
  `count_used_parameters_in_backward` in DeepSpeed's ZeRO stage 1/2 module (DeepSpeed 0.18; validated with 0.18.5).
  With a DeepSpeed that does not have it, enabling the switch fails at start-up.
- **Training time:** these timings exclude start-up (about 3 minutes) and checkpoint saving.

# Environment and assets

Faster-Motus needs the same environment, weights and data as upstream Motus. This page records the stack it was
validated on and the steps that differed from upstream's instructions.

## Install the runtime

Follow upstream's installation (Python 3.10, PyTorch 2.7.1 + torchvision 0.22.1 for CUDA 12.8, flash-attn,
`requirements.txt`) in an isolated environment:

```bash
conda create -n faster-motus python=3.10 -y
conda activate faster-motus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation      # or the matching 2.8.3 release wheel
pip install -r requirements.txt
```

The validated stack is PyTorch 2.7.1+cu128, DeepSpeed 0.18.5, accelerate 1.12.0, transformers 5.0.0rc0, flash-attn 2.8.3,
decord 0.6.0 and diffusers 0.41.0.

**NCCL:** PyTorch 2.7.1+cu128 pins `nvidia-nccl-cu12==2.26.2`. With it, every NCCL communicator
initialization hung on the machines used here (driver 580.126.09). NCCL 2.27.3 (PyTorch 2.8's pin) works.
After `pip install -r requirements.txt`, which reinstalls the pin, run:

```bash
pip install --no-deps nvidia-nccl-cu12==2.27.3
```

`torch.cuda.nccl.version()` still reports the version PyTorch was compiled with; the library loaded at run time is
2.27.3.

**DS_HOOK and DeepSpeed:** the switch patches `count_used_parameters_in_backward` in
`deepspeed.runtime.zero.stage_1_and_2` (DeepSpeed 0.18). If your DeepSpeed does not have that function, leave
`MOTUS_OPT_DS_HOOK` unset.

## Prepare assets once

| Input | Source | Config key in `configs/robotwin.yaml` |
|---|---|---|
| Motus Stage-2 checkpoint (fine-tuning start) | `motus-robotics/Motus` | `finetune.checkpoint_path` |
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | `model.vlm.checkpoint_path` |
| Wan2.2-TI2V-5B (config, VAE `Wan2.2_VAE.pth`, umt5 encoder for data preparation) | `Wan-AI/Wan2.2-TI2V-5B` | `model.wan.config_path`, `model.wan.checkpoint_path`, `model.wan.vae_path` |
| RoboTwin 2.0 data, converted | `TianxingChen/RoboTwin2.0` archives, converted as below | `dataset.dataset_dir` |

Keep weights and data out of the repository and read-only during training. Set `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` once the weights are local.

### Convert the RoboTwin 2.0 data

Upstream's converter (`data/robotwin2/robotwin_data_convert/`, see its README) turns each episode's HDF5 file into the
layout the training dataset reads: `<subset>/<task>/{videos,qpos,metas,umt5_wan}/<episode>.{mp4,pt,txt,pt}`. The full
data is 27,500 episodes. The converter processes episodes one at a time, so two tools here run it in parallel:

```bash
# CPU: every aloha-agilex episode of the 50 tasks, upstream's converter, one (subset, task) per process
python tools/convert_robotwin_full.py --archives /data/robotwin2.0 --target /data/robotwin2 \
    --scratch /dev/shm/robotwin_raw --logs logs/convert --jobs 24

# GPU: umt5 embeddings with upstream's T5 code; start any number of workers (two per GPU worked well), on any
# hosts that see the same file system
for g in 0 1 2 3 4 5 6 7; do for k in 1 2; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. nohup python tools/encode_t5_sharded.py --target /data/robotwin2 \
      --wan /models/Wan2.2-TI2V-5B --claims /data/robotwin2_t5_claims --logs logs/convert \
      --worker $(hostname -s)_g${g}_$k > logs/t5_g${g}_$k.log 2>&1 &
done; done
```

On one 8-GPU machine plus 14 GPUs elsewhere, the conversion took about 20 minutes and the embeddings about 30 minutes.
The result: 27,244 episodes; the converter's own qpos check drops 256. The data takes 1.4 TB, almost all of it umt5
files: each episode stores the embeddings of its ~100 prompt lines, about 52 MB.

Two upstream converter problems are handled by these tools:

- Upstream's T5 step imports `utils.device` from the repository root, so it fails with `No module named 'utils'`
  unless the repository root is on `PYTHONPATH`.
- Its multi-GPU wrapper sets `CUDA_VISIBLE_DEVICES=<n>` and then selects `cuda:<n>`, which only works for GPU 0.
  `tools/encode_t5_sharded.py` calls upstream's per-file encoder on the one GPU it is given.

Embeddings cached for other projects cannot be reused. Motus encodes each full meta line, a fixed scene description
followed by the instruction, with umt5's bidirectional encoder, so the instruction's embeddings depend on that prefix.

## Check before training

```bash
python -m pytest tests/                 # CPU only: DS_HOOK and LANG_MMAP equivalence checks
nvidia-smi                              # all 8 GPUs idle
```

Data loading uses 16 workers per GPU (128 processes). With LOADER_SPEC off, the host needs about 250 GB of RAM above
its idle use.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Training hangs at the first collective | NCCL 2.26.2 on the machines used here | Install nvidia-nccl-cu12 2.27.3 |
| T5 conversion: `No module named 'utils'` | upstream T5 code imports from the repository root | Run with `PYTHONPATH=<repo root>` |
| T5 conversion runs on the wrong GPU or fails for GPU n > 0 | upstream wrapper selects `cuda:<n>` after masking | Use `tools/encode_t5_sharded.py` |
| Host memory grows sharply at each epoch start | `MOTUS_OPT_LOADER_SPEC=1` | Unset it (not in the preset) |
| GPU memory stays allocated after a killed run, no training process left | Orphaned LOADER_SPEC workers | Kill your leftover `pt_data_worker` processes; leave LOADER_SPEC off |
| `AttributeError: ... count_used_parameters_in_backward` at start-up | DeepSpeed without that function | Unset `MOTUS_OPT_DS_HOOK` or use DeepSpeed 0.18 |
| Loss is NaN from iteration 2 on a single GPU | Upstream behaviour: upstream Motus does the same on 1 GPU (seen in deterministic mode, bitwise identical with and without the switches) | Use the 8-GPU recipe; it trains with finite losses |

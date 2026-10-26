#!/bin/bash
# One observed training run for a bitwise or timing comparison: tools/fullrun.py under torchrun with scripts/train.sh's
# arguments. The config is the tree's configs/robotwin.yaml with only the machine paths, training.max_steps and the
# checkpoint directory changed. Runs in the foreground; per-rank records go to $OUT.
#
#   OUT=runs/val_faster STEPS=1100 DET=1 \
#   DATA=/data/robotwin2 WAN=/models/Wan2.2-TI2V-5B QWEN=/models/Qwen3-VL-2B-Instruct MOTUS_CKPT=/models/Motus \
#   bash tools/run_validation.sh
#
# TREE      tree to run (default: this repository); use an upstream checkout for the reference run
# SWITCHES  MOTUS_OPT_ switches to set, without the prefix (default: the preset, "DS_HOOK LANG_MMAP"; "" sets none)
# DET=1     deterministic mode, for bitwise comparisons; leave unset for timing
# NPROC     GPUs (default 8, as scripts/train.sh); PYTHON interpreter (default python); PORT rendezvous port
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "${TREE:-$HERE/..}" && pwd)
OUT=$(mkdir -p "${OUT:?set OUT}" && cd "$OUT" && pwd)
PYTHON=${PYTHON:-python}
: "${STEPS:?set STEPS}" "${DATA:?set DATA}" "${WAN:?set WAN}" "${QWEN:?set QWEN}" "${MOTUS_CKPT:?set MOTUS_CKPT}"

"$PYTHON" - "$TREE/configs/robotwin.yaml" "$OUT/robotwin.yaml" "$DATA" "$WAN" "$QWEN" "$MOTUS_CKPT" "$STEPS" \
  "$OUT/checkpoints" <<'EOF'
import sys
from omegaconf import OmegaConf
src, dest, data, wan, qwen, motus, steps, ckpt = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.dataset.dataset_dir = data
cfg.model.wan.config_path = cfg.model.wan.checkpoint_path = wan
cfg.model.wan.vae_path = f'{wan}/Wan2.2_VAE.pth'
cfg.model.vlm.checkpoint_path = qwen
cfg.finetune.checkpoint_path = motus
cfg.training.max_steps = int(steps)
cfg.system.checkpoint_dir = ckpt
open(dest, 'w').write(OmegaConf.to_yaml(cfg))
EOF

SW=()
for s in ${SWITCHES-DS_HOOK LANG_MMAP}; do SW+=("MOTUS_OPT_$s=1"); done
DFLAG=()
[ "${DET:-0}" = 1 ] && DFLAG=(--deterministic)
echo "tree=$TREE out=$OUT steps=$STEPS det=${DET:-0} switches=${SW[*]:-none}"
cd "$TREE"
exec env "${SW[@]}" "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="${NPROC:-8}" --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${PORT:-$((29600 + RANDOM % 300))}" "$HERE/fullrun.py" --tree "$TREE" \
  --out "$OUT" "${DFLAG[@]}" -- --deepspeed configs/zero1.json --config "$OUT/robotwin.yaml" --run_name robotwin \
  --report_to tensorboard

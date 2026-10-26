# Source this file before launching training (e.g. `source scripts/faster-motus.env.sh && bash scripts/train.sh`).
# The Faster-Motus preset, the same as .harness/config.json: both switches validated bitwise against upstream on the
# full RoboTwin 2.0 data (docs/full-data-results.md).
export MOTUS_OPT_DS_HOOK=1      # cache DeepSpeed ZeRO-1/2's per-hook parameter count (the main speedup)
export MOTUS_OPT_LANG_MMAP=1    # map the umt5 embedding files instead of reading ~50 MB per sample
# Available but not in the preset (see docs/full-data-results.md):
#   MOTUS_OPT_LOADER_REUSE=1    keep data-loader workers across epochs (about 0.3% on full data)
#   MOTUS_OPT_LOADER_SPEC=1     prepare the next epoch ahead; NOT recommended: unbounded pinned host memory

#!/bin/bash
# Samples host memory while a tools/run_validation.sh run trains on this host, until its rank0.jsonl has the exit
# record: the host's used and available memory (free -g), the summed RSS of the training processes and of their
# data-loader workers (RSS double-counts pages shared after fork; compare the host's used memory instead), and the
# step reached.
#   bash tools/mem_sample.sh <run dir> [interval seconds] > <run dir>/mem.log
OUT=${1:?run directory}
INTERVAL=${2:-5}
echo "time used_GB avail_GB ranks_rss_GB workers_rss_GB n_workers step"
while ! grep -q '"exit"' "$OUT/rank0.jsonl" 2>/dev/null; do
  read -r used avail < <(free -g | awk '/^Mem:/ {print $3, $7}')
  procs=$(ps -u "$(id -u)" -o rss=,comm=,args= | awk '
    index($0, "fullrun.py") && !index($0, "awk") {
      if ($2 ~ /^pt_data/) {w += $1; nw++} else {r += $1}
    }
    END {printf "%.1f %.1f %d", r/1048576, w/1048576, nw}')
  step=$(tail -c 2000 "$OUT/rank0.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | tail -1 | awk '{print $2}')
  echo "$(date +%T) $used $avail $procs ${step:-0}"
  sleep "$INTERVAL"
done
echo "exit seen"

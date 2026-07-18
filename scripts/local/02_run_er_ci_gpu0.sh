#!/bin/bash
# Run ER-Balanced CI (b=1,5,10) sequentially on GPU0.
# After all three finish, emits a sentinel that watch_and_launch_m10.sh waits for.
# Takes ~12-15 min total on RTX 3090.
#
# Usage: bash 02_run_er_ci_gpu0.sh   (run in background with nohup)

set -euo pipefail

REPODIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
QLOG="$LOGDIR/_run_gpu0.out"

cd "$REPODIR"
mkdir -p "$LOGDIR"

for mem in 1 5 10; do
    tag="er_balanced_ci_m${mem}"
    echo "[$(date '+%H:%M:%S')] Starting $tag on GPU0" | tee -a "$QLOG"

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 RUN_TAG="${tag}_" \
        uv run python main.py --er --balanced True --mem "$mem" --scenario 1 --seed 42 \
        2>&1 | tr '\r' '\n' | grep -v '^$' > "$LOGDIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] === DONE ${tag} ===" | tee -a "$QLOG"
done

echo "[$(date '+%H:%M:%S')] ALL GPU0 ER-BALANCED CI DONE" | tee -a "$QLOG"

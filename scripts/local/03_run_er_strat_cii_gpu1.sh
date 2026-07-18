#!/bin/bash
# Run ER-Strat CI (m1,m5) then ER-Strat CII (m1,m5) on GPU1.
# m10 is handled separately by 05_watch_and_launch_m10.sh after GPU0 CI finishes.
# Takes ~20-25 min total on RTX 3090.
#
# Usage: bash 03_run_er_strat_cii_gpu1.sh   (run in background with nohup)

set -euo pipefail

REPODIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
QLOG="$LOGDIR/_run_gpu1.out"

cd "$REPODIR"
mkdir -p "$LOGDIR"

for scenario in 1 2; do
    scenario_label=$([ "$scenario" -eq 1 ] && echo "ci" || echo "cii")
    for mem in 1 5; do
        tag="er_strat_${scenario_label}_m${mem}"
        echo "[$(date '+%H:%M:%S')] Starting $tag on GPU1" | tee -a "$QLOG"

        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 RUN_TAG="${tag}_" \
            uv run python main.py --er --mem "$mem" --scenario "$scenario" --seed 42 \
            2>&1 | tr '\r' '\n' | grep -v '^$' > "$LOGDIR/${tag}.log"

        echo "[$(date '+%H:%M:%S')] === DONE ${tag} ===" | tee -a "$QLOG"
    done
done

echo "[$(date '+%H:%M:%S')] ALL GPU1 ER-STRAT DONE" | tee -a "$QLOG"

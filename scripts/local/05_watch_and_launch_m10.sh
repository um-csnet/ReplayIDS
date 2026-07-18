#!/bin/bash
# Wait for GPU0 ER-Balanced CI to finish, then run ER-Strat m10 (CI then CII) on GPU0.
# Start this BEFORE 02_run_er_ci_gpu0.sh so it is already watching when CI finishes.
#
# Usage: nohup bash 05_watch_and_launch_m10.sh &

set -euo pipefail

REPODIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
QLOG="$LOGDIR/_watch_m10.out"
M10LOG="$LOGDIR/_run_gpu0_strat_m10.out"

mkdir -p "$LOGDIR"
echo "[$(date '+%H:%M:%S')] Watcher started, waiting for GPU0 ER-Balanced CI to finish..." | tee -a "$QLOG"

until grep -q "ALL GPU0 ER-BALANCED CI DONE" "$LOGDIR/_run_gpu0.out" 2>/dev/null; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] GPU0 CI done. Launching ER-Strat m10..." | tee -a "$QLOG"

cd "$REPODIR"
for scenario in 1 2; do
    scenario_label=$([ "$scenario" -eq 1 ] && echo "ci" || echo "cii")
    tag="er_strat_${scenario_label}_m10"
    echo "[$(date '+%H:%M:%S')] Starting $tag on GPU0" | tee -a "$M10LOG"

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 RUN_TAG="${tag}_" \
        uv run python main.py --er --mem 10 --scenario "$scenario" --seed 42 \
        2>&1 | tr '\r' '\n' | grep -v '^$' > "$LOGDIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] === DONE ${tag} ===" | tee -a "$M10LOG"
done

echo "[$(date '+%H:%M:%S')] ER-STRAT m10 ALL DONE" | tee -a "$M10LOG"

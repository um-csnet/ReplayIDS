#!/bin/bash
# Run all non-replay baselines: Naive, EWC, LwF, iCaRL — both CI and CII scenarios.
# Assign to whichever GPU is free; defaults to GPU0.
# Logs go to $LOGDIR. Each run takes ~10-15 min on RTX 3090.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash 01_run_baselines.sh
#   LOGDIR=/custom/path bash 01_run_baselines.sh

set -euo pipefail

REPODIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPODIR"
mkdir -p "$LOGDIR"

run_baseline() {
    local flag="$1"    # --ewc | --lwf | --icarl | (empty for naive)
    local scenario="$2"
    local tag="$3"

    echo "[$(date '+%H:%M:%S')] Starting $tag (scenario $scenario, GPU $GPU)"

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$GPU" RUN_TAG="${tag}_" \
        uv run python main.py $flag --scenario "$scenario" --seed 42 \
        2>&1 | tr '\r' '\n' | grep -v '^$' > "$LOGDIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] === DONE $tag ===" | tee -a "$LOGDIR/${tag}.log"
}

# Naive (no extra flag)
run_baseline ""        1 "tabT_naive_ci"
run_baseline ""        2 "tabT_naive_cii"

# EWC (lambda=1000, default)
run_baseline "--ewc"   1 "tabT_ewc_ci"
run_baseline "--ewc"   2 "tabT_ewc_cii"

# LwF (alpha=0.5, T=2.0, defaults)
run_baseline "--lwf"   1 "tabT_lwf_ci"
run_baseline "--lwf"   2 "tabT_lwf_cii"

# iCaRL (exemplar budget K=2000, default)
run_baseline "--icarl" 1 "tabT_icarl_ci"
run_baseline "--icarl" 2 "tabT_icarl_cii"

echo "[$(date '+%H:%M:%S')] ALL BASELINES DONE"

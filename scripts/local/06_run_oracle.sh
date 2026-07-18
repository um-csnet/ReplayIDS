#!/bin/bash
# Run oracle (joint training) for both CI and CII scenarios.
# Oracle trains once on ALL data pooled — no continual learning.
# Used to compute intransigence: oracle_acc - method_acc
#
# NOTE: the oracle model in main.py uses dim=64, heads=8 (more capacity than
# the CL model which uses dim=32, heads=10). This is intentional — the oracle
# has more headroom since it sees all data at once. See main.py ~line 350.
# Hardcoded oracle values from these runs (seed=42, RTX 3090):
#   CI:  Acc=0.9997, F1=0.9979
#   CII: Acc=0.9955, F1=0.9795
#
# Usage: bash v2/scripts/06_run_oracle.sh

set -euo pipefail

REPODIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$LOGDIR"
cd "$REPODIR"

for scenario in 1 2; do
    scenario_label=$([ "$scenario" -eq 1 ] && echo "ci" || echo "cii")
    tag="oracle_${scenario_label}"
    echo "[$(date '+%H:%M:%S')] Starting $tag on GPU$GPU"

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$GPU" RUN_TAG="${tag}_" \
        uv run python main.py --oracle --scenario "$scenario" --seed 42 \
        2>&1 | tr '\r' '\n' | grep -v '^$' > "$LOGDIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] === DONE ${tag} ==="
    grep -E "Overall accuracy|Macro F1|BWT" "$LOGDIR/${tag}.log" | tail -5
done

echo "Oracle runs complete. Logs: $LOGDIR/oracle_{ci,cii}.log"

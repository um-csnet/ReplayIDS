#!/bin/bash
# Run ER-Balanced CII (b=1,5,10) on remote RTX 5090 via SSH.
# Requires: SSH key at ~/.ssh/csnet_proxmox, 5090 accessible at 10.84.0.14,
#           wandb==0.21.3 installed in /root/ReplayIDS/.venv on the 5090
#           (NOT 0.28.0 — its Go daemon crashes asynchronously).
#
# The 5090 logs are written locally to LOGDIR via SSH stdout redirect.
# Takes ~15-20 min total.
#
# Usage: bash 04_run_er_bal_cii_5090.sh

set -euo pipefail

LOGDIR="${LOGDIR:-/home/minda/synology/2026/0705_AziziAfif/experiments/lwf/results/logs}"
SSH="ssh -i ~/.ssh/csnet_proxmox -o StrictHostKeyChecking=no root@10.84.0.14"
QLOG="$LOGDIR/_run_5090.out"

mkdir -p "$LOGDIR"

# Verify 5090 is reachable
if ! $SSH "echo OK" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach 5090 at 10.84.0.14 — check VPN or SSH key" >&2
    exit 1
fi

for mem in 1 5 10; do
    tag="er_balanced_cii_m${mem}"
    echo "[$(date '+%H:%M:%S')] Starting $tag on 5090" | tee -a "$QLOG"

    $SSH "cd /root/ReplayIDS && WANDB_MODE=disabled RUN_TAG=${tag}_ \
        .venv/bin/python main.py --er --balanced True --mem $mem --scenario 2 --seed 42 \
        2>&1 | tr '\r' '\n' | grep -v '^\$'" \
        > "$LOGDIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] === DONE ${tag} ===" | tee -a "$QLOG"
done

echo "[$(date '+%H:%M:%S')] ALL 5090 ER-BALANCED CII DONE" | tee -a "$QLOG"

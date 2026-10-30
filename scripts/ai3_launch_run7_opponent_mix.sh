#!/usr/bin/env bash
# Launch run7: opponent-pool self-play (fictitious-play) experiment.
#
# Starts from the run6 best checkpoint (best_calling_station_512.pth,
# +116.55 bb/100 vs calling_station) with the opponent_mix feature enabled
# (probability 0.3, strategy calling_station). This gives the advantage
# network direct exploitation signal so the learned equilibrium robustly
# beats calling_station rather than only transiently peaking.
#
# Usage: bash scripts/ai3_launch_run7_opponent_mix.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="ai3-rd-${TS}-run7-opponent-mix"
LOG_PATH="/root/work/software/cfr/logs/ai3-rd-${TS}.log"

# Ensure the best checkpoint is the "latest" .pth by mtime so run_training.py
# auto-loads it. We touch the best checkpoint symlink target to make it newest.
PREP_CMD="cd /root/work/software/cfr && \
  cp -f models/best/best_calling_station_latest.pth models/run7_start.pth && \
  touch models/run7_start.pth && \
  ls -la models/run7_start.pth models/best/best_calling_station_latest.pth"

echo "[run7] Preparing best checkpoint as latest on ai3..."
scripts/ai3_shell.sh "$PREP_CMD" >/dev/null

echo "[run7] Launching training job: $JOB_NAME"
echo "[run7] Log: $LOG_PATH"
# NPU budget policy: cap at 8 NPUs per environment (default 1).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/npu_limit.sh"
NPU_DEVICES="$(npu_visible_devices "${ASI3_NUM_NPUS:-1}" 8)" || exit 1
scripts/ai3_job.sh start "$JOB_NAME" "$LOG_PATH" \
  "cd /root/work/software/cfr && export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES && python3 run_training.py --config poc_ai3_npu.yaml --algorithm deep_cfr --npus"

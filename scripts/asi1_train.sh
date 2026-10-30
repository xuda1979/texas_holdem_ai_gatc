#!/usr/bin/env bash
set -euo pipefail

# Launch a Deep CFR training run on ASI1 (Ascend NPU, 4x NPUs available).
#
# Usage:
#   scripts/asi1_train.sh start <job-name> [extra train.py args...]
#   scripts/asi1_train.sh status <job-id>
#   scripts/asi1_train.sh logs <job-id> [tail-lines]
#   scripts/asi1_train.sh list
#
# Environment overrides:
#   HUANXIN_ASI1_REMOTE_ROOT  - remote working directory (default: ~/software/cfr)
#   ASI1_CONFIG               - config file to use (default: poc_asi1_npu.yaml)
#   ASI1_ALGORITHM            - training algorithm (default: deep_cfr)
#   ASI1_NUM_NPUS             - number of NPUs to use (default: 1; hard cap 8)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Shared NPU budget policy: never more than 8 NPUs per environment.
source scripts/npu_limit.sh
NPU_CAP=8
ASI1_NUM_NPUS="${ASI1_NUM_NPUS:-1}"
NPU_DEVICES="$(npu_visible_devices "$ASI1_NUM_NPUS" "$NPU_CAP")" || exit 1

CONFIG_FILE="${ASI1_CONFIG:-poc_asi1_npu.yaml}"
ALGORITHM="${ASI1_ALGORITHM:-deep_cfr}"
REMOTE_ROOT="${HUANXIN_ASI1_REMOTE_ROOT:-~/software/cfr}"

usage() {
  cat >&2 <<EOF
Usage:
  scripts/asi1_train.sh start <job-name> [extra train.py args...]
  scripts/asi1_train.sh status <job-id>
  scripts/asi1_train.sh logs <job-id> [tail-lines]
  scripts/asi1_train.sh list

Config: $CONFIG_FILE
Algorithm: $ALGORITHM
Remote root: $REMOTE_ROOT
EOF
  exit 1
}

[[ $# -ge 1 ]] || usage
ACTION="$1"; shift || true

case "$ACTION" in
  start)
    [[ $# -ge 1 ]] || usage
    job_name="$1"; shift
    log_path="logs/${job_name}-$(date -u +%Y%m%dT%H%M%SZ).log"

    # Compose the remote training command.  We export POKER_AI_HUANXIN_REMOTE_ROOT
    # so the trainer writes checkpoints/models to /workspace/cfr/models.
    # PYTHONPATH includes both repo root (for gatc_holdem) and src/ (for poker_ai).
    extra_args="$*"
    remote_cmd="cd $REMOTE_ROOT && export POKER_AI_HUANXIN_REMOTE_ROOT=$REMOTE_ROOT && export POKER_AI_ALLOW_LOCAL_MODEL_WRITES=1 && export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES && export PYTHONPATH=\$PWD:src:\$PYTHONPATH && python3 run_training.py --config $CONFIG_FILE --algorithm $ALGORITHM --npus $extra_args"

    echo "[asi1_train] Launching training job '$job_name' on ASI1..." >&2
    echo "[asi1_train] Remote log: $log_path" >&2
    exec scripts/asi1_job.sh start "$job_name" "$log_path" "$remote_cmd"
    ;;
  status|logs|list)
    exec scripts/asi1_job.sh "$ACTION" "$@"
    ;;
  *)
    usage
    ;;
esac

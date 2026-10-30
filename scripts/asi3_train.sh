#!/usr/bin/env bash
set -euo pipefail

# Launch a Deep CFR training run on ASI3 (Ascend NPU).
#
# Usage:
#   scripts/asi3_train.sh start <job-name> [extra train.py args...]
#   scripts/asi3_train.sh status <job-id>
#   scripts/asi3_train.sh logs <job-id> [tail-lines]
#   scripts/asi3_train.sh list
#
# The script:
#   1. Pushes the local repo to ASI3 via scripts/ai3_push_to_s3.sh's sync.
#   2. Launches `python run_training.py --config poc_ai3_npu.yaml --npus`
#      as a backgrounded nohup job on ASI3.
#   3. Records the job in .huanxin_jobs/ so scripts/asi3_job.sh status works.
#
# Environment overrides:
#   HUANXIN_ASI3_REMOTE_ROOT  - remote working directory (default: ~/software/cfr)
#   ASI3_CONFIG               - config file to use (default: poc_ai3_npu.yaml)
#   ASI3_ALGORITHM            - training algorithm (default: deep_cfr)
#   ASI3_SYNC_FIRST           - if "1", rsync the repo before launching (default: 1)
#   ASI3_NUM_NPUS             - number of NPUs to use (default: 1; hard cap 8)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Shared NPU budget policy: never more than 8 NPUs per environment.
source scripts/npu_limit.sh
NPU_CAP=8
ASI3_NUM_NPUS="${ASI3_NUM_NPUS:-1}"
NPU_DEVICES="$(npu_visible_devices "$ASI3_NUM_NPUS" "$NPU_CAP")" || exit 1

CONFIG_FILE="${ASI3_CONFIG:-poc_ai3_npu.yaml}"
ALGORITHM="${ASI3_ALGORITHM:-deep_cfr}"
SYNC_FIRST="${ASI3_SYNC_FIRST:-1}"
REMOTE_ROOT="${HUANXIN_ASI3_REMOTE_ROOT:-~/software/cfr}"

usage() {
  cat >&2 <<EOF
Usage:
  scripts/asi3_train.sh start <job-name> [extra train.py args...]
  scripts/asi3_train.sh status <job-id>
  scripts/asi3_train.sh logs <job-id> [tail-lines]
  scripts/asi3_train.sh list

Config: $CONFIG_FILE
Algorithm: $ALGORITHM
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

    if [[ "$SYNC_FIRST" == "1" ]]; then
      echo "[asi3_train] Syncing repo to ASI3..." >&2
      scripts/ai3_push_to_s3.sh --exclude '/.git/**' --exclude '/.venv/**' \
        --exclude '/__pycache__/**' --exclude '/.mypy_cache/**' \
        --exclude '/runs/**' --exclude '/outputs/**' \
        --exclude '/models/**' --exclude '/trained_models/**' \
        --exclude '/checkpoints/**' --exclude '/artifacts/**' \
        --exclude '/memory/**' --exclude '/logs/**' \
        >/dev/null 2>&1 || {
          echo "[asi3_train] warning: sync failed, continuing with existing remote state" >&2
        }
    fi

    # Compose the remote training command.
    extra_args="$*"
    remote_cmd="cd $REMOTE_ROOT && export POKER_AI_HUANXIN_REMOTE_ROOT=$REMOTE_ROOT && export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES && export PYTHONPATH=\$PWD/src:\$PYTHONPATH && python3 run_training.py --config $CONFIG_FILE --algorithm $ALGORITHM --npus $extra_args"

    echo "[asi3_train] Launching training job '$job_name' on ASI3..." >&2
    echo "[asi3_train] Remote log: $log_path" >&2
    exec scripts/asi3_job.sh start "$job_name" "$log_path" "$remote_cmd"
    ;;
  status|logs|list)
    exec scripts/asi3_job.sh "$ACTION" "$@"
    ;;
  *)
    usage
    ;;
esac

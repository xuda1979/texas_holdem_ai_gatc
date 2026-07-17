#!/usr/bin/env bash
set -euo pipefail

# Run comprehensive evaluation against checkpoints on ASI3.
#
# Usage:
#   scripts/asi3_evaluate.sh checkpoint <ckpt-path> [--h2h-hands N] [--output report.json]
#   scripts/asi3_evaluate.sh sweep <models-dir>    [--prefix P] [--latest-only] [--output sweep.json]
#   scripts/asi3_evaluate.sh health   <ckpt-path>
#   scripts/asi3_evaluate.sh h2h      <model-a> <model-b> [--hands N]
#   scripts/asi3_evaluate.sh pull     <job-id>   # pull eval history + reports to ./runs/asi3/
#
# All commands execute *on ASI3* via scripts/ai3_shell.sh; the only local
# operation is the optional ``pull`` subcommand which copies artifacts back
# to the local filesystem via scripts/ai3_sync_from_s3.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_ROOT="${HUANXIN_ASI3_REMOTE_ROOT:-~/software/cfr}"
USAGE=$(cat <<'EOF'
Usage:
  scripts/asi3_evaluate.sh checkpoint <ckpt-path> [extra evaluate.py args...]
  scripts/asi3_evaluate.sh sweep      <models-dir> [extra evaluate.py args...]
  scripts/asi3_evaluate.sh health     <ckpt-path>  [extra evaluate.py args...]
  scripts/asi3_evaluate.sh h2h        <model-a> <model-b> [extra evaluate.py args...]
  scripts/asi3_evaluate.sh pull       <local-dir>
EOF
)

[[ $# -ge 1 ]] || { echo "$USAGE" >&2; exit 1; }
ACTION="$1"; shift || true

case "$ACTION" in
  checkpoint|sweep|health|h2h|baselines)
    # Forward to evaluate.py on the remote host.
    remote_cmd="cd $REMOTE_ROOT && export PYTHONPATH=\$PWD/src:\$PYTHONPATH && python3 evaluate.py $ACTION $*"
    exec scripts/ai3_shell.sh "$remote_cmd"
    ;;
  pull)
    [[ $# -ge 1 ]] || { echo "pull requires a local destination directory" >&2; exit 1; }
    local_dir="$1"
    mkdir -p "$local_dir"
    # Pull the remote models directory and the eval history file.
    REMOTE_PATH="${REMOTE_ROOT/#\~/}/models"
    echo "[asi3_evaluate] Pulling $REMOTE_PATH -> $local_dir" >&2
    scripts/ai3_sync_from_s3.sh "$REMOTE_PATH" "$local_dir" 2>/dev/null || true
    # Also pull logs (training event stream).
    REMOTE_LOGS="${REMOTE_ROOT/#\~/}/logs"
    if [[ -d "$REMOTE_LOGS" ]]; then
      scripts/ai3_sync_from_s3.sh "$REMOTE_LOGS" "$local_dir/logs" 2>/dev/null || true
    fi
    echo "[asi3_evaluate] Done. Artifacts in $local_dir" >&2
    ;;
  *)
    echo "$USAGE" >&2; exit 1
    ;;
esac

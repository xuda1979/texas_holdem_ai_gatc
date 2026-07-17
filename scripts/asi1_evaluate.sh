#!/usr/bin/env bash
set -euo pipefail

# Run comprehensive evaluation against checkpoints on ASI1.
#
# Usage:
#   scripts/asi1_evaluate.sh checkpoint <ckpt-path> [--h2h-hands N] [--output report.json]
#   scripts/asi1_evaluate.sh sweep      <models-dir> [--latest-only] [--output sweep.json]
#   scripts/asi1_evaluate.sh health     <ckpt-path>
#   scripts/asi1_evaluate.sh h2h        <model-a> <model-b> [--hands N]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_ROOT="${HUANXIN_ASI1_REMOTE_ROOT:-/workspace/cfr}"
USAGE=$(cat <<'EOF'
Usage:
  scripts/asi1_evaluate.sh checkpoint <ckpt-path> [extra evaluate.py args...]
  scripts/asi1_evaluate.sh sweep      <models-dir> [extra evaluate.py args...]
  scripts/asi1_evaluate.sh health     <ckpt-path>  [extra evaluate.py args...]
  scripts/asi1_evaluate.sh h2h        <model-a> <model-b> [extra evaluate.py args...]
  scripts/asi1_evaluate.sh baselines  <ckpt-path>  [extra evaluate.py args...]
EOF
)

[[ $# -ge 1 ]] || { echo "$USAGE" >&2; exit 1; }
ACTION="$1"; shift || true

case "$ACTION" in
  checkpoint|sweep|health|h2h|baselines)
    remote_cmd="cd $REMOTE_ROOT && export POKER_AI_ALLOW_LOCAL_MODEL_WRITES=1 && export PYTHONPATH=\$PWD:src:\$PYTHONPATH && python3 evaluate.py $ACTION $*"
    exec scripts/huanxin_shell.sh ASI1 "$remote_cmd"
    ;;
  *)
    echo "$USAGE" >&2; exit 1
    ;;
esac

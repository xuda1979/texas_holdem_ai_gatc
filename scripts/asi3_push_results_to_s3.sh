#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="${HUANXIN_ASI3_S3_ROOT:-${HUANXIN_AI_S3_ROOT:-nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/cfr}}"
REMOTE_ROOT="${HUANXIN_ASI3_REMOTE_ROOT:-~/software/cfr}"

cd "$ROOT_DIR"

remote_quoted="$(python3 -c "import shlex,sys; print(shlex.quote(sys.argv[1]))" "$REMOTE_ROOT")"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1; shift
fi

remote_paths=("$@")
if [[ ${#remote_paths[@]} -eq 0 ]]; then
  remote_paths=("models" "logs" "outputs" "artifacts")
fi

RCLONE_ENV="${RCLONE_ENV:-}"

for remote_path in "${remote_paths[@]}"; do
  REMOTE_CMD="cd $remote_quoted && if [[ -e '$remote_path' ]]; then $RCLONE_ENV rclone copy '$remote_path' '$S3_ROOT/$remote_path' --s3-no-check-bucket --progress"
  if [[ $DRY_RUN -eq 1 ]]; then
    REMOTE_CMD+=" --dry-run"
  fi
  REMOTE_CMD+="; else echo 'skip missing: $remote_path'; fi"
  JSON_OUT="$(scripts/ai3_shell.sh "$REMOTE_CMD")"
  python3 - <<'PY' "$JSON_OUT"
import json
import sys

payload = json.loads(sys.argv[1])
text = payload.get("after", "") or ""
bad_markers = ("ERROR :", "NOTICE: Failed", "AccessDenied", "Failed to")
if any(marker in text for marker in bad_markers):
    raise SystemExit(text)
PY
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="${HUANXIN_ASI3_S3_ROOT:-${HUANXIN_AI_S3_ROOT:-nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/cfr}}"
REMOTE_ROOT="${HUANXIN_ASI3_REMOTE_ROOT:-~/software/cfr}"

cd "$ROOT_DIR"

RCLONE_ENV="${RCLONE_ENV:-}"

# Sync the local repo to the S3-backed remote root on ASI3.
RCLONE_CMD="$RCLONE_ENV rclone copy . '$S3_ROOT/' --s3-no-check-bucket --exclude '/.git/**' --exclude '/.venv/**' --exclude '/__pycache__/**' --exclude '/.mypy_cache/**' --exclude '/outputs/**' --exclude '/models/**' --exclude '/trained_models/**' --exclude '/checkpoints/**' --exclude '/artifacts/**' --exclude '/memory/**' --exclude '/logs/**' --progress"
if [[ "${1:-}" == "--dry-run" ]]; then
  RCLONE_CMD+=" --dry-run"
fi

JSON_OUT="$(scripts/ai3_shell.sh "$RCLONE_CMD")"

python3 - <<'PY' "$JSON_OUT"
import json
import sys

payload = json.loads(sys.argv[1])
text = payload.get("after", "") or ""
bad_markers = ("ERROR :", "NOTICE: Failed", "AccessDenied", "Failed to")
if any(marker in text for marker in bad_markers):
    raise SystemExit(text)
PY

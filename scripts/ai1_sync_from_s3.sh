#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/cfr"
REMOTE_ROOT="/root/root/work/cfr"

cd "$ROOT_DIR"

RCLONE_CMD="mkdir -p '$REMOTE_ROOT' && rclone sync $S3_ROOT '$REMOTE_ROOT' --exclude '/outputs/**' --exclude '/models/**' --exclude '/trained_models/**' --exclude '/checkpoints/**' --exclude '/artifacts/**' --exclude '/memory/**' --exclude '/logs/**' --progress"
if [[ "${1:-}" == "--dry-run" ]]; then
  RCLONE_CMD+=" --dry-run"
fi

JSON_OUT="$(scripts/ai1_shell.sh "$RCLONE_CMD")"

python3 - <<'PY' "$JSON_OUT"
import json
import sys

payload = json.loads(sys.argv[1])
text = payload.get("after", "") or ""
bad_markers = ("ERROR :", "NOTICE: Failed", "AccessDenied", "Failed to")
if any(marker in text for marker in bad_markers):
    raise SystemExit(text)
PY

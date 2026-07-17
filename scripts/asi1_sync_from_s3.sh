#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="${HUANXIN_ASI1_S3_ROOT:-${HUANXIN_AI_S3_ROOT:-nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/cfr}}"
REMOTE_ROOT="${HUANXIN_ASI1_REMOTE_ROOT:-~/software/cfr}"

cd "$ROOT_DIR"

remote_quote_path() {
  python3 - <<'PY' "$1"
import shlex
import sys

path = sys.argv[1]
if path == "~":
    print('"$HOME"')
elif path.startswith("~/"):
    print('"$HOME"/' + shlex.quote(path[2:]))
else:
    print(shlex.quote(path))
PY
}

REMOTE_ROOT_Q="$(remote_quote_path "$REMOTE_ROOT")"

RCLONE_ENV=""
if [[ -n "${HUANXIN_AI_S3_ACCESS_KEY_ID:-}" && -n "${HUANXIN_AI_S3_SECRET_ACCESS_KEY:-}" ]]; then
  RCLONE_ENV="RCLONE_CONFIG_HXAI_TYPE=s3 RCLONE_CONFIG_HXAI_PROVIDER=Other RCLONE_CONFIG_HXAI_ACCESS_KEY_ID='${HUANXIN_AI_S3_ACCESS_KEY_ID}' RCLONE_CONFIG_HXAI_SECRET_ACCESS_KEY='${HUANXIN_AI_S3_SECRET_ACCESS_KEY}' RCLONE_CONFIG_HXAI_ENDPOINT='${HUANXIN_AI_S3_ENDPOINT:-https://iner.aihuanxin.cn}' RCLONE_CONFIG_HXAI_FORCE_PATH_STYLE=true"
fi

RCLONE_CMD="mkdir -p $REMOTE_ROOT_Q && $RCLONE_ENV rclone sync $S3_ROOT $REMOTE_ROOT_Q --exclude '/outputs/**' --exclude '/models/**' --exclude '/trained_models/**' --exclude '/checkpoints/**' --exclude '/artifacts/**' --exclude '/memory/**' --exclude '/logs/**' --progress"
if [[ "${1:-}" == "--dry-run" ]]; then
  RCLONE_CMD+=" --dry-run"
fi

JSON_OUT="$(scripts/asi1_shell.sh "$RCLONE_CMD")"

python3 - <<'PY' "$JSON_OUT"
import json
import sys

payload = json.loads(sys.argv[1])
text = payload.get("after", "") or ""
bad_markers = ("ERROR :", "NOTICE: Failed", "AccessDenied", "Failed to")
if any(marker in text for marker in bad_markers):
    raise SystemExit(text)
PY

#!/usr/bin/env bash
set -euo pipefail

# Pull a path from the ASI3 S3 bucket to the local filesystem.
# Usage: scripts/asi3_sync_from_s3.sh <remote-path> <local-dest> [--dry-run]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="${HUANXIN_ASI3_S3_ROOT:-${HUANXIN_AI_S3_ROOT:-nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3/cfr}}"

cd "$ROOT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1; shift
fi

[[ $# -ge 1 ]] || { echo "Usage: $0 <remote-path> [local-dest] [--dry-run]" >&2; exit 1; }
remote_path="$1"
local_dest="${2:-.}"

RCLONE_ENV="${RCLONE_ENV:-}"
RCLONE_CMD="$RCLONE_ENV rclone copy '$S3_ROOT/$remote_path' '$local_dest' --s3-no-check-bucket --progress"
if [[ $DRY_RUN -eq 1 ]]; then
  RCLONE_CMD+=" --dry-run"
fi

eval "$RCLONE_CMD"

#!/usr/bin/env bash
# Backup important CFR project files to NAS.
#
# Default target is the Huanxin AI environment's backup directory
# (/root/work/backups/cfr/), reached via the ai3 shell.  If Huanxin
# auth is expired (the current state as of 2026-07-19), the script
# falls back to a local tarball under $HOME/backups/cfr/ which can be
# pushed to NAS once auth is restored.
#
# Usage:
#   scripts/backup_to_nas.sh                # auto: remote if auth ok, else local
#   scripts/backup_to_nas.sh --local        # force local tarball
#   scripts/backup_to_nas.sh --remote       # force remote (fails if auth expired)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_BACKUP_DIR="${HOME}/backups/cfr"
REMOTE_BACKUP_DIR="/root/work/backups/cfr"
MODE="${1:-auto}"

mkdir -p "$LOCAL_BACKUP_DIR"

backup_local() {
    local out="${LOCAL_BACKUP_DIR}/cfr-${TIMESTAMP}.tar.gz"
    echo "[backup] writing local tarball -> $out"
    tar -czf "$out" \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.mypy_cache' \
        --exclude='.ruff_cache' \
        --exclude='*.pyc' \
        --exclude='models' \
        --exclude='logs' \
        --exclude='.git' \
        --exclude='train.ipynb' \
        -C "$PROJECT_ROOT" \
        src/ scripts/ docs/ memory/ tests/ \
        MEMORY.md SOUL.md AGENTS.md USER.md TOOLS.md \
        pyproject.toml uv.lock readme.md 2>/dev/null || true
    local size
    size=$(du -h "$out" | awk '{print $1}')
    local files
    files=$(tar -tzf "$out" | wc -l | tr -d ' ')
    echo "[backup] done: $files files, $size"
}

backup_remote() {
    echo "[backup] attempting remote sync to ai3:$REMOTE_BACKUP_DIR"
    if ! scripts/ai3_shell.sh "mkdir -p $REMOTE_BACKUP_DIR" 2>/dev/null; then
        echo "[backup] ERROR: cannot reach ai3 (Huanxin auth likely expired)."
        echo "[backup] Falling back to local tarball."
        backup_local
        return
    fi
    # Sync the important directories via ai3_sync_to_s3-style rsync over SSH.
    # NOTE: this requires `scripts/ai3_shell.sh` to expose an rsync-compatible
    # transport.  For now, we tar locally then scp the tarball.
    local out="${LOCAL_BACKUP_DIR}/cfr-${TIMESTAMP}.tar.gz"
    if [ ! -f "$out" ]; then
        backup_local
    fi
    scripts/ai3_shell.sh "cat > $REMOTE_BACKUP_DIR/$(basename "$out")" < "$out"
    echo "[backup] remote copy complete: $REMOTE_BACKUP_DIR/$(basename "$out")"
}

case "$MODE" in
    --local)  backup_local ;;
    --remote) backup_remote ;;
    auto)
        if scripts/ai3_shell.sh "true" 2>/dev/null; then
            backup_remote
        else
            echo "[backup] ai3 unreachable; using local fallback."
            backup_local
        fi
        ;;
    *)
        echo "Usage: $0 [--local|--remote|auto]" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# cfr_rd_loop.sh — recurring R&D continuation for the CFR poker AI project.
#
# Invoked every 20 minutes by the com.quantumgpt.cfr-rd-loop LaunchAgent.
# Runs a non-interactive Claude Code session that continues training and R&D
# on the Huanxin ASI1/ASI2/ASI3 NPU environments.
set -euo pipefail

REPO_DIR="/Users/daxu/software/cfr"
LOG_DIR="$REPO_DIR/tools/logs"
mkdir -p "$LOG_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_LOG="$LOG_DIR/cfr-rd-loop.$TS.out.log"
ERR_LOG="$LOG_DIR/cfr-rd-loop.$TS.err.log"

# Keep only the last 20 run logs to avoid unbounded growth.
find "$LOG_DIR" -name "cfr-rd-loop.*.out.log" -type f -mtime +7 -delete 2>/dev/null || true
find "$LOG_DIR" -name "cfr-rd-loop.*.err.log" -type f -mtime +7 -delete 2>/dev/null || true

cd "$REPO_DIR"

# Path for claude + node (fnm multishell).
export PATH="/Users/daxu/.local/bin:/Users/daxu/.local/state/fnm_multishells/1002_1777257073261/bin:/Users/daxu/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Non-interactive R&D continuation prompt.  This is the verbatim prompt the
# /loop command parsed: continue research and development until full success,
# using available Huanxin ASI NPUs (max 4).
PROMPT='Continue the CFR poker AI research and development. Check the ASI1 training status (PID in .huanxin_jobs/asi1-rd-4layer-*.json, log at /workspace/texas-holdem/logs/asi1-rd-4layer.log on the remote). Monitor the 4-layer Deep CFR training: if the process died, restart it; if loss has stalled, investigate and fix; if a comprehensive evaluation has fired, analyze the h2h bb/100 results and address any weaknesses. Use scripts/asi1_shell.sh for remote commands. Do not use more than 2 NPUs. Push the work toward full success — a trained policy that beats all baselines (always_fold, calling_station, random_uniform) with positive bb/100.'

# Run Claude Code in print mode (non-interactive).  --dangerously-skip-permissions
# is safe here because this runs in the user's own sandbox with no untrusted input.
exec /Users/daxu/.local/bin/claude \
  --print \
  --dangerously-skip-permissions \
  --output-format text \
  "$PROMPT" \
  > "$OUT_LOG" 2> "$ERR_LOG"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${HUANXIN_ASI3_REMOTE_ROOT:-~/software/cfr}"
SHELL_WRAPPER="$ROOT_DIR/scripts/ai3_shell.sh"
META_DIR="$ROOT_DIR/.huanxin_jobs"
TAIL_LINES_DEFAULT="${HUANXIN_JOB_TAIL_LINES:-40}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/asi3_job.sh start <job-name> <log-path> "<remote command>"
  scripts/asi3_job.sh status <job-id> [tail-lines]
  scripts/asi3_job.sh logs <job-id> [tail-lines]
  scripts/asi3_job.sh list
EOF
  exit 1
}

combine_transcript() {
  python3 - "$@" <<'PY'
import json, sys
payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
before = (payload.get("before") or "").strip()
after = (payload.get("after") or "").strip()
output = (payload.get("output") or "").strip()
parts = [p for p in (before, output, after) if p]
print("\n".join(parts))
PY
}

[[ $# -ge 1 ]] || usage
ACTION="$1"; shift || true

mkdir -p "$META_DIR"

case "$ACTION" in
  start)
    [[ $# -eq 3 ]] || usage
    job_name="$1"
    log_path="$2"
    remote_command="$3"
    job_id="${job_name}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

    # Compose the nohup wrapper.  The PID is captured after backgrounding and
    # echoed in a known marker format so we can parse it from the transcript.
    wrapped_command="cd $REMOTE_ROOT && mkdir -p $(dirname "$log_path") && nohup bash -c '$(echo "$remote_command" | sed "s/'/'\\\\''/g")' > '$log_path' 2>&1 & echo \"__OPENCLAW_JOB_PID__ \$! __OPENCLAW_JOB_LOG__ $log_path __OPENCLAW_JOB_ROOT__ $REMOTE_ROOT __OPENCLAW_JOB_END__\""

    JSON_OUT="$("$SHELL_WRAPPER" "$wrapped_command")"

    python3 - <<'PY' "$JSON_OUT" "$job_id" "$job_name" "$log_path" "$remote_command" "$META_DIR"
import json, os, re, sys
from datetime import datetime, timezone

payload = json.loads(sys.argv[1])
job_id = sys.argv[2]
job_name = sys.argv[3]
log_path = sys.argv[4]
remote_command = sys.argv[5]
meta_dir = sys.argv[6]

transcript = "\n".join(str(payload.get(k, "")) for k in ("before", "output", "after"))
match = re.search(r"__OPENCLAW_JOB_PID__\s+(\d+)\s+__OPENCLAW_JOB_LOG__\s+(\S+)\s+__OPENCLAW_JOB_ROOT__\s+(\S+)\s+__OPENCLAW_JOB_END__", transcript)
if not match:
    raise SystemExit(f"Could not parse job PID from transcript:\n{transcript}")

meta_file = os.path.join(meta_dir, f"{job_id}.json")
job_payload = {
    'job_id': match.group(1) if False else job_id,
    'pid': int(match.group(1)),
    'name': job_name,
    'log_path': match.group(2),
    'remote_root': match.group(3),
    'command': remote_command,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'transport': payload.get('transport'),
    'duration_ms': payload.get('durationMs'),
    'meta_file': meta_file,
}
with open(meta_file, 'w', encoding='utf-8') as handle:
    json.dump(job_payload, handle, indent=2)
print(json.dumps(job_payload, indent=2))
PY
    ;;
  status|logs)
    [[ $# -ge 1 && $# -le 2 ]] || usage
    job_id="$1"
    tail_lines="${2:-$TAIL_LINES_DEFAULT}"
    meta_file="$META_DIR/${job_id}.json"
    [[ -f "$meta_file" ]] || { echo "No such job: $job_id" >&2; exit 2; }

    python3 - "$meta_file" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(f.read())
PY

    # Check whether the PID is still running on the remote host.
    meta_json="$(cat "$meta_file")"
    pid="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['pid'])" "$meta_json")"
    log_path_remote="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['log_path'])" "$meta_json")"
    remote_root="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['remote_root'])" "$meta_json")"

    status_cmd="ps -p $pid >/dev/null 2>&1 && echo RUNNING || echo DONE; tail -n $tail_lines $log_path_remote 2>/dev/null || true"
    JSON_OUT="$("$SHELL_WRAPPER" "cd $remote_root && $status_cmd")"
    combine_transcript "$JSON_OUT"
    ;;
  list)
    python3 - <<'PY' "$META_DIR"
import glob, json, os, sys
items = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    with open(path, encoding='utf-8') as handle:
        items.append(json.load(handle))
print(json.dumps(items, indent=2))
PY
    ;;
  *)
    usage
    ;;
esac

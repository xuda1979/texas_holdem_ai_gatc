#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/huanxin_shell.sh <AI|ai1|ai2|ai3> "<remote command>"

Examples:
  scripts/huanxin_shell.sh AI "cd ~/software/cfr && pwd"
  scripts/huanxin_shell.sh ai1 "cd /root/root/work/cfr && pwd"
  scripts/huanxin_shell.sh ai2 "echo ok"
  scripts/huanxin_shell.sh ai3 "cd ~/software/cfr && pwd"
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

ENV_NAME="$1"
shift

case "$ENV_NAME" in
  AI) DEFAULT_PORT="19006" ;;
  ai1) DEFAULT_PORT="19001" ;;
  ai2) DEFAULT_PORT="19002" ;;
  ai3) DEFAULT_PORT="19003" ;;
  ASI1) DEFAULT_PORT="20646" ;;
  ASI2) DEFAULT_PORT="19004" ;;
  ASI3) DEFAULT_PORT="20653" ;;
  *) echo "Unsupported env: $ENV_NAME" >&2; exit 1 ;;
esac

TRAIN_DEV_BASE_URL="https://aihuanxin.cn/kunlun/kl-web?poolId=6&projectId=21b4208dde424e96b159362ef49c9c96#/train-dev"
case "$ENV_NAME" in
  AI)
    DEFAULT_TRAIN_DEV_URL="${TRAIN_DEV_BASE_URL}/environment/dl-9a5a098accce31c28cf4c6ca23391341?name=AI"
    ;;
  ai3)
    DEFAULT_TRAIN_DEV_URL="${TRAIN_DEV_BASE_URL}/environment/dl-c72bd81a96e33134bbe0ae4a478fbab0?name=ai3"
    ;;
  *)
    DEFAULT_TRAIN_DEV_URL="$TRAIN_DEV_BASE_URL"
    ;;
esac
export HUANXIN_TRAIN_DEV_URL="${HUANXIN_TRAIN_DEV_URL:-$DEFAULT_TRAIN_DEV_URL}"

PROFILE_COPY_NAME="${HUANXIN_PROFILE_COPY_NAME:-cfr}"
export HUANXIN_PROFILE_COPY_NAME="$PROFILE_COPY_NAME"
export HUANXIN_HEADLESS="${HUANXIN_HEADLESS:-1}"
WAIT_MS="${HUANXIN_WAIT_MS:-180000}"
USE_DAEMON="${HUANXIN_USE_DAEMON:-1}"
ALLOW_STANDALONE_FALLBACK="${HUANXIN_ALLOW_STANDALONE_FALLBACK:-1}"
AUTO_REFRESH_SAFARI_ON_CALLBACK_FAILURE="${HUANXIN_AUTO_REFRESH_SAFARI_ON_CALLBACK_FAILURE:-1}"
export HUANXIN_ALLOW_STANDALONE_FALLBACK="$ALLOW_STANDALONE_FALLBACK"

if [[ -n "${HUANXIN_BROWSER_ROOT:-}" ]]; then
  BROWSER_ROOT="${HUANXIN_BROWSER_ROOT}"
elif [[ -d "${HOME}/software/quantum-gpt/browser-automation" ]]; then
  BROWSER_ROOT="${HOME}/software/quantum-gpt"
else
  BROWSER_ROOT="${HOME}/.openclaw/workspace-quantum-rnd"
fi
BROWSER_DIR="${BROWSER_ROOT}/browser-automation"
KEEPALIVE_SCRIPT="${HUANXIN_KEEPALIVE_SCRIPT:-${HOME}/software/quantum-gpt/scripts/huanxin_safari_keepalive.sh}"

if [[ ! -d "$BROWSER_DIR" ]]; then
  echo "[huanxin_shell:$ENV_NAME] browser automation directory not found at $BROWSER_DIR" >&2
  exit 1
fi

cd "$BROWSER_ROOT"

RUN_PREFIX="$(printf '%s' "$ENV_NAME" | tr '[:lower:]' '[:upper:]')"
RUN_MARKER="__${RUN_PREFIX}_RUN_$(date +%s%N)__"
REMOTE_CMD="$*; echo ${RUN_MARKER}"

DAEMON_PORT_FILE="/tmp/huanxin-daemon-${ENV_NAME}.port"
DAEMON_PID_FILE="/tmp/huanxin-daemon-${ENV_NAME}.pid"
DAEMON_LOG_FILE="/tmp/huanxin-daemon-${ENV_NAME}.log"

daemon_health_ok() {
  local port=""
  if [[ -f "$DAEMON_PORT_FILE" ]]; then
    port="$(tr -d '[:space:]' < "$DAEMON_PORT_FILE" 2>/dev/null)"
  fi
  if [[ -z "$port" ]]; then
    port="$DEFAULT_PORT"
  fi
  curl --max-time 2 -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1
}

daemon_health_json() {
  local port=""
  if [[ -f "$DAEMON_PORT_FILE" ]]; then
    port="$(tr -d '[:space:]' < "$DAEMON_PORT_FILE" 2>/dev/null)"
  fi
  if [[ -z "$port" ]]; then
    port="$DEFAULT_PORT"
  fi
  curl --max-time 2 -sf "http://127.0.0.1:${port}/health" 2>/dev/null || true
}

daemon_health_ready() {
  python3 - <<'PY' "$1"
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("0")
    raise SystemExit(0)

print("1" if payload.get("ready") else "0")
PY
}

daemon_terminal_prompt_failure() {
  python3 - <<'PY' "$1"
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("0")
    raise SystemExit(0)

startup_error = str(payload.get("startupError") or "")
needs = (
    payload.get("startupState") == "error"
    and "did not reach a prompt" in startup_error
)
print("1" if needs else "0")
PY
}

daemon_port_file_ready() {
  [[ -f "$DAEMON_PORT_FILE" ]] && [[ -n "$(tr -d '[:space:]' < "$DAEMON_PORT_FILE" 2>/dev/null)" ]]
}

shell_exec_capture() {
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  SHELL_EXEC_OUTPUT="$output"
  SHELL_EXEC_STATUS="$status"
}

callback_capture_failed() {
  [[ "$1" == *"Safari callback capture failed after retries"* ]]
}

refresh_existing_safari_tab() {
  if [[ "$AUTO_REFRESH_SAFARI_ON_CALLBACK_FAILURE" != "1" ]]; then
    return 1
  fi
  if [[ ! -x "$KEEPALIVE_SCRIPT" ]]; then
    echo "[huanxin_shell:$ENV_NAME] Safari keepalive script not found at $KEEPALIVE_SCRIPT" >&2
    return 1
  fi
  echo "[huanxin_shell:$ENV_NAME] Refreshing existing Safari Huanxin tab before retry..." >&2
  set +e
  local refresh_output refresh_status
  refresh_output="$("$KEEPALIVE_SCRIPT" --refresh --activate 2>&1)"
  refresh_status=$?
  set -e
  if [[ $refresh_status -ne 0 ]]; then
    echo "$refresh_output" >&2
    return 1
  fi
  echo "$refresh_output" >&2
  return 0
}

run_shell_exec_with_retry() {
  shell_exec_capture "$@"
  if [[ "$SHELL_EXEC_STATUS" -eq 0 ]]; then
    JSON_OUT="$SHELL_EXEC_OUTPUT"
    return 0
  fi

  if callback_capture_failed "$SHELL_EXEC_OUTPUT" && refresh_existing_safari_tab; then
    shell_exec_capture "$@"
  fi

  if [[ "$SHELL_EXEC_STATUS" -ne 0 ]]; then
    echo "$SHELL_EXEC_OUTPUT" >&2
    return "$SHELL_EXEC_STATUS"
  fi

  JSON_OUT="$SHELL_EXEC_OUTPUT"
  return 0
}

start_daemon_if_needed() {
  local health_json=""
  if daemon_health_ok; then
    if ! daemon_port_file_ready; then
      printf '%s' "$DEFAULT_PORT" > "$DAEMON_PORT_FILE"
    fi
    health_json="$(daemon_health_json)"
    if [[ -n "$health_json" && "$(daemon_health_ready "$health_json")" == "1" ]]; then
      return 0
    fi
  fi

  echo "[huanxin_shell:$ENV_NAME] Starting browser daemon..." >&2
  rm -f "$DAEMON_PORT_FILE" "$DAEMON_PID_FILE"
  HUANXIN_PROFILE_COPY_NAME="$PROFILE_COPY_NAME" HUANXIN_HEADLESS="${HUANXIN_HEADLESS}" \
    nohup node "$BROWSER_DIR/huanxin_browser_daemon.js" "$ENV_NAME" > "$DAEMON_LOG_FILE" 2>&1 &

  for _poll in $(seq 1 30); do
    if daemon_health_ok; then
      if ! daemon_port_file_ready; then
        printf '%s' "$DEFAULT_PORT" > "$DAEMON_PORT_FILE"
      fi
      health_json="$(daemon_health_json)"
      if [[ -n "$health_json" && "$(daemon_health_ready "$health_json")" == "1" ]]; then
        if [[ -f "$DAEMON_PID_FILE" ]]; then
          echo "[huanxin_shell:$ENV_NAME] Daemon ready (PID $(cat "$DAEMON_PID_FILE" 2>/dev/null))" >&2
        else
          echo "[huanxin_shell:$ENV_NAME] Daemon ready" >&2
        fi
        return 0
      fi
    fi
    sleep 2
  done

  echo "[huanxin_shell:$ENV_NAME] Daemon unavailable after startup attempt." >&2
  if [[ -f "$DAEMON_LOG_FILE" ]]; then
    tail -n 20 "$DAEMON_LOG_FILE" >&2 || true
  fi
  return 1
}

DAEMON_READY=0
if [[ "$USE_DAEMON" == "1" ]] && start_daemon_if_needed && daemon_port_file_ready; then
  DAEMON_READY=1
fi

if [[ "$USE_DAEMON" == "1" && "$DAEMON_READY" == "1" ]]; then
  echo "[huanxin_shell:$ENV_NAME] Using daemon transport." >&2
  run_shell_exec_with_retry \
    env HUANXIN_REQUIRE_DAEMON=1 \
    node "$BROWSER_DIR/huanxin_shell_exec.js" "$ENV_NAME" --require-daemon --wait-ms "$WAIT_MS" --command "$REMOTE_CMD"
else
  if [[ "$USE_DAEMON" == "1" ]]; then
    health_json="$(daemon_health_json)"
    if [[ -n "$health_json" && "$(daemon_terminal_prompt_failure "$health_json")" == "1" ]]; then
      echo "[huanxin_shell:$ENV_NAME] Shell terminal failed to reach a prompt; refusing slow standalone retry." >&2
      echo "$health_json" >&2
      exit 1
    fi
  fi
  if [[ "$USE_DAEMON" == "1" && "$ALLOW_STANDALONE_FALLBACK" != "1" ]]; then
    echo "[huanxin_shell:$ENV_NAME] Refusing standalone fallback because daemon startup failed and fallback is disabled." >&2
    exit 1
  fi
  if [[ "$USE_DAEMON" == "1" ]]; then
    echo "[huanxin_shell:$ENV_NAME] Falling back to standalone transport because daemon startup did not hold." >&2
  else
    echo "[huanxin_shell:$ENV_NAME] Using standalone transport because HUANXIN_USE_DAEMON=0 was set." >&2
  fi
  run_shell_exec_with_retry \
    node "$BROWSER_DIR/huanxin_shell_exec.js" "$ENV_NAME" --skip-daemon --wait-ms "$WAIT_MS" --command "$REMOTE_CMD"
fi

python3 - <<'PY' "$JSON_OUT" "$RUN_MARKER" "$ENV_NAME"
import json
import sys

payload = json.loads(sys.argv[1])
run_marker = sys.argv[2]
env_name = sys.argv[3]
combined = "\n".join(str(payload.get(key, "")) for key in ("output", "after", "before"))

if not payload.get("ok", False):
    raise SystemExit(f"huanxin_shell({env_name}) command failed: {payload}")

if run_marker not in combined:
    raise SystemExit(
        f"Detected stale Huanxin shell output for {env_name}: expected run marker {run_marker} was not observed."
    )

print(json.dumps(payload, indent=2))
PY

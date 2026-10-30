#!/usr/bin/env bash
# Shared NPU budget enforcement for ASI1/ASI2/ASI3 training launches.
#
# Policy: a single training environment must never use more than 8 NPUs
# on the shared ASI machines.  Source this file and call:
#
#   npu_visible_devices <requested> <max>
#
# It prints the ASCEND_RT_VISIBLE_DEVICES value (comma-separated device
# indices, 0-based) honouring both the request and the hard cap, and
# prints an error to stderr if the request exceeds the cap.
#
# Examples:
#   npu_visible_devices 1 8   -> 0
#   npu_visible_devices 4 8   -> 0,1,2,3
#   npu_visible_devices 16 8  -> error, prints nothing (caller aborts)

npu_visible_devices() {
  local requested="${1:-1}"
  local max="${2:-8}"

  if ! [[ "$requested" =~ ^[0-9]+$ ]] || [ "$requested" -lt 1 ]; then
    echo "[npu_limit] Invalid NPU request '$requested' (must be >= 1)" >&2
    return 1
  fi
  if [ "$requested" -gt "$max" ]; then
    echo "[npu_limit] Requested $requested NPUs exceeds the per-environment cap of $max (ASI1/2/3 shared machines)." >&2
    return 1
  fi

  local devices=()
  for ((i = 0; i < requested; i++)); do
    devices+=("$i")
  done
  local IFS=','
  echo "${devices[*]}"
}

# Derive the effective NPU count from an env var, clamped to the cap.
# Usage: NUM=$(npu_clamped_count "$ASI3_NUM_NPUS" 8); [ $? -eq 0 ] || exit 1
npu_clamped_count() {
  local value="${1:-}"
  local max="${2:-8}"
  local requested="${value:-1}"

  if ! [[ "$requested" =~ ^[0-9]+$ ]] || [ "$requested" -lt 1 ]; then
    echo "[npu_limit] Invalid NPU count '$requested'" >&2
    return 1
  fi
  if [ "$requested" -gt "$max" ]; then
    echo "[npu_limit] NPU count $requested exceeds the per-environment cap of $max; clamping to $max." >&2
    requested="$max"
  fi
  echo "$requested"
}

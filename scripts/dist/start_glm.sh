#!/usr/bin/env bash
# Start GLM server (default: NVFP4 config). Modes: pure_cpu | hybrid_gpu | pure_gpu
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server_glm"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_glm.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_glm not found — build the project first" >&2
  exit 1
fi
CFG="${1:-configs/engine_glm_nvfp4.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi
echo "Starting GLM server with $CFG"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
exec "$BIN" --config "$CFG"

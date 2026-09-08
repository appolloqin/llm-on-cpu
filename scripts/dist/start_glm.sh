#!/usr/bin/env bash
# Start GLM MoE NVFP4 (siblings: llmoc_server_glm_bf16 / _int4)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server_glm_nvfp4"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_glm_nvfp4.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_glm_nvfp4 not found — build the project first" >&2
  exit 1
fi
CFG="${1:-configs/engine_glm_nvfp4.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi
if [[ ! -f models/GLM-5.3-Flash.nvfp4.glmq ]]; then
  echo "ERROR: models/GLM-5.3-Flash.nvfp4.glmq missing — run ./download_glm.sh first" >&2
  exit 1
fi
echo "Starting GLM NVFP4 server with $CFG"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
exec "$BIN" --config "$CFG"

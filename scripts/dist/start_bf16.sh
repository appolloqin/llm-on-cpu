#!/usr/bin/env bash
# Start BF16 server. Modes: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server not found" >&2
  exit 1
fi
CFG="${1:-configs/engine.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.lwc ]]; then
  echo "WARN: models/Qwen3.5-4B.lwc missing — run ./download_bf16.sh first" >&2
fi
echo "Starting BF16 server with $CFG"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
exec "$BIN" --config "$CFG"

#!/usr/bin/env bash
# Start INT4 server. Modes: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream
# Optional: ./start_int4.sh configs/engine_int4_mtp.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server_int4"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_int4.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_int4 not found" >&2
  exit 1
fi
CFG="${1:-configs/engine_int4.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.int4.qlwc ]]; then
  echo "WARN: models/Qwen3.5-4B.int4.qlwc missing — run ./download_int4.sh first" >&2
fi
echo "Starting INT4 server with $CFG"
echo "  hybrid/pure_gpu: set tiers.gpu_vram_gb; layer_stream: see yaml layer_stream block"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
exec "$BIN" --config "$CFG"

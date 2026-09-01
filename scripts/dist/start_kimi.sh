#!/usr/bin/env bash
# Start Kimi stub (Kimi-STUB-v0). Single-card pure_gpu → auto layer_stream
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="./bin/llmoc_server_kimi"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_kimi.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_kimi not found" >&2
  exit 1
fi
CFG="${1:-configs/engine_kimi_hybrid.yaml}"
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
[[ -f models/fake_kimi.kimiq ]] || echo "WARN: models/fake_kimi.kimiq missing — run ./download_kimi.sh" >&2
echo "Starting Kimi stub with $CFG (port ~15087)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$BIN" --config "$CFG"

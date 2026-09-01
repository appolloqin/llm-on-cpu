#!/usr/bin/env bash
# Start DeepSeek stub (DS-STUB-v0)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="./bin/llmoc_server_ds"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_ds.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_ds not found" >&2
  exit 1
fi
CFG="${1:-configs/engine_ds_nvfp4.yaml}"
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
[[ -f models/fake_ds.dskq ]] || echo "WARN: models/fake_ds.dskq missing — run ./download_ds.sh" >&2
echo "Starting DS stub with $CFG (port ~15086)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$BIN" --config "$CFG"

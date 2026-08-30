#!/usr/bin/env bash
# Start INT4 server on port 15085
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server_int4"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_int4.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_int4 not found" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.int4.qlwc ]]; then
  echo "WARN: models/Qwen3.5-4B.int4.qlwc missing — run ./download_int4.sh first" >&2
fi
echo "Starting INT4 server on http://127.0.0.1:15085/"
exec "$BIN" --config configs/engine_int4.yaml

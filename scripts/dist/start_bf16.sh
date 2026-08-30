#!/usr/bin/env bash
# Start BF16 (unquantized) server on port 15085
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server not found" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.lwc ]]; then
  echo "WARN: models/Qwen3.5-4B.lwc missing — run ./download_bf16.sh first" >&2
fi
echo "Starting BF16 server on http://127.0.0.1:15085/"
exec "$BIN" --config configs/engine.yaml

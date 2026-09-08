#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="./bin/llmoc_server_kimi_nvfp4"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_kimi_nvfp4.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/llmoc_server_kimi_nvfp4 not found" >&2
  exit 1
fi
exec "$BIN" --config "${1:-configs/engine_kimi_nvfp4.yaml}"

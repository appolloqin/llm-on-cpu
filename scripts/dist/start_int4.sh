#!/usr/bin/env bash
# 启动 INT4 服务，端口 15085
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server_int4"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server_int4.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: 找不到 bin/llmoc_server_int4" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.int4.qlwc ]]; then
  echo "WARN: 未找到 models/Qwen3.5-4B.int4.qlwc — 请先运行 ./download_int4.sh" >&2
fi
echo "Starting INT4 server on http://127.0.0.1:15085/"
exec "$BIN" --config configs/engine_int4.yaml

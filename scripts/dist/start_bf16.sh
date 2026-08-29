#!/usr/bin/env bash
# 启动 BF16（无量化）服务，端口 15085
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BIN="./bin/llmoc_server"
[[ -x "$BIN" ]] || BIN="./bin/llmoc_server.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: 找不到 bin/llmoc_server" >&2
  exit 1
fi
if [[ ! -f models/Qwen3.5-4B.lwc ]]; then
  echo "WARN: 未找到 models/Qwen3.5-4B.lwc — 请先运行 ./download_bf16.sh" >&2
fi
echo "Starting BF16 server on http://127.0.0.1:15085/"
exec "$BIN" --config configs/engine.yaml

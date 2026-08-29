#!/usr/bin/env bash
# 下载 → 转 LWC → 删除原 HF 大权重（保留 config/tokenizer）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: 需要 Node.js >= 18，并加入 PATH。" >&2
  exit 1
fi

MODEL="Qwen/Qwen3.5-4B"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

SHORT="${MODEL##*/}"
echo "== [1/2] 下载 + 转 LWC + 删除原 safetensors  ${MODEL}"
node tools/prepare_model.mjs --model "${MODEL}" --prune-hf "${EXTRA[@]}"
echo
echo "OK. 引擎权重: models/${SHORT}.lwc"
echo "    旁路保留: models/${SHORT}-hf/ （仅 config/tokenizer，大权重已删）"
echo "若模型不是 Qwen3.5-4B，请改 configs/engine.yaml。"
echo "然后运行: ./start_bf16.sh"

#!/usr/bin/env bash
# 下载 → 转 LWC → INT4 量化 → 删除原 HF 大权重 + 中间 BF16 LWC
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
echo "== [1/3] 下载 + 转 LWC + 删除原 safetensors  ${MODEL}"
node tools/prepare_model.mjs --model "${MODEL}" --prune-hf "${EXTRA[@]}"

echo "== [2/3] INT4 量化"
node tools/quantize_int4.mjs \
  --src "models/${SHORT}.lwc" \
  --out "models/${SHORT}.int4.qlwc" \
  --method gptq

echo "== [3/3] 删除中间 BF16 LWC"
if [[ -f "models/${SHORT}.lwc" ]]; then
  rm -f "models/${SHORT}.lwc"
  echo "  removed models/${SHORT}.lwc"
fi

echo
echo "OK. 引擎权重: models/${SHORT}.int4.qlwc"
echo "    旁路保留: models/${SHORT}-hf/ （仅 config/tokenizer，大权重已删）"
echo "若模型不是 Qwen3.5-4B，请改 configs/engine_int4.yaml。"
echo "然后运行: ./start_int4.sh"

#!/usr/bin/env bash
# Auto: download / convert / INT4 / prune as needed
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js >= 18 required in PATH." >&2
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
echo "== [INT4] prepare ${MODEL} (AWQ/compressed-tensors→QLWC import; or BF16→LWC→QLWC)"
echo "    Tip: pre-quant (cyankiwi/Qwen3.8-27B-AWQ-INT4) or BF16 base (Qwen/Qwen3.8-27B)"
node tools/prepare_model.mjs --model "${MODEL}" --prune-hf --int4 "${EXTRA[@]}"
echo
echo "OK. Engine weights: models/${SHORT}.int4.qlwc"
echo "    Tokenizer keep: models/${SHORT}-hf/"
echo "Next: ./start_int4.sh"
echo "    Modes: edit configs/engine_int4.yaml (pure_cpu|hybrid_gpu|pure_gpu|auto|layer_stream)"
echo "    GPU: tiers.gpu_vram_gb; MTP: ./start_int4.sh configs/engine_int4_mtp.yaml"

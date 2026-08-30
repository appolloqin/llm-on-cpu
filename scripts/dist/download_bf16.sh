#!/usr/bin/env bash
# Auto: download / convert / prune as needed (BF16)
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
echo "== [BF16] prepare ${MODEL} (auto-skip done steps)"
node tools/prepare_model.mjs --model "${MODEL}" --prune-hf "${EXTRA[@]}"
echo
echo "OK. Engine weights: models/${SHORT}.lwc"
echo "    Tokenizer keep: models/${SHORT}-hf/"
echo "Next: ./start_bf16.sh"

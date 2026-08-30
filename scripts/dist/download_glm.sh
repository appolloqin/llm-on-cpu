#!/usr/bin/env bash
# Auto: download / import NVFP4 (default) or AWQ for GLM-5.3-Flash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js >= 18 required in PATH." >&2
  exit 1
fi

QUANT="nvfp4"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quant) QUANT="${2:?}"; shift 2 ;;
    --nvfp4) QUANT="nvfp4"; shift ;;
    --awq) QUANT="awq"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

echo "== [GLM] prepare quant=${QUANT} (default: LibertAIDAI/GLM-5.3-Flash-NVFP4)"
node tools/glm/prepare_glm.mjs --quant "${QUANT}" --prune-hf "${EXTRA[@]}"
echo
if [[ "${QUANT}" == "awq" ]]; then
  echo "Next: ./start_glm.sh configs/engine_glm_int4.yaml"
else
  echo "Next: ./start_glm.sh"
fi

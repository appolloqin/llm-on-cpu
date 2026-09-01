#!/usr/bin/env bash
# Write tiny DeepSeek stub weights (DSS1) — NOT a real model download
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="./bin/make_fake_ds"
[[ -x "$BIN" ]] || BIN="./bin/make_fake_ds.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/make_fake_ds not found — build first" >&2
  exit 1
fi
mkdir -p models
OUT="${1:-models/fake_ds.dskq}"
"$BIN" "$OUT"
echo "OK. Stub weights: $OUT"
echo "Next: ./start_ds.sh"
echo "    (DS-STUB-v0 — smoke only; see configs/engine_ds_nvfp4.yaml)"

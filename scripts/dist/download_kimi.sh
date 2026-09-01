#!/usr/bin/env bash
# Write tiny Kimi stub weights (KIM1) — NOT a real model download
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="./bin/make_fake_kimi"
[[ -x "$BIN" ]] || BIN="./bin/make_fake_kimi.exe"
if [[ ! -e "$BIN" ]]; then
  echo "ERROR: bin/make_fake_kimi not found — build first" >&2
  exit 1
fi
mkdir -p models
OUT="${1:-models/fake_kimi.kimiq}"
"$BIN" "$OUT"
echo "OK. Stub weights: $OUT"
echo "Next: ./start_kimi.sh"
echo "    (Kimi-STUB-v0; single-card pure_gpu → auto layer_stream)"

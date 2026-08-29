#!/usr/bin/env bash
# llm-on-cpu :: 将 build/bin 产物打成可分发应用包
# 用法:
#   BUILD_DIR=build/release PLATFORM_ID=linux-x64 ./scripts/package_app.sh
#   BUILD_DIR=build/msvc-x64 PLATFORM_ID=windows-x64 ./scripts/package_app.sh
set -euo pipefail
# Git Bash/MSYS 的 pwd 是 /d/...，原生 Windows Python 会当成 \\d\\... 失败；优先用 Windows 路径。
if ROOT_WIN="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null)"; then
  ROOT="${ROOT_WIN}"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
# 统一为正斜杠，便于传给 Python pathlib
ROOT="${ROOT//\\//}"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
PLATFORM_ID="${PLATFORM_ID:-unknown}"
VER="${PKG_VER:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
BIN_DIR="${BUILD_DIR}/bin"
DIST="${ROOT}/dist"
STAGE="${DIST}/llm-on-cpu-${VER}-${PLATFORM_ID}"

mkdir -p "${STAGE}/bin" "${STAGE}/configs" "${STAGE}/docs"

BINS=(
  llmoc_server
  llmoc_server_int4
  lwc_verify
  m0_bandwidth
  m0_isa
  int4_gemm_bench
)

copied=0
for b in "${BINS[@]}"; do
  if [[ -f "${BIN_DIR}/${b}.exe" ]]; then
    cp "${BIN_DIR}/${b}.exe" "${STAGE}/bin/"
    copied=$((copied + 1))
  elif [[ -f "${BIN_DIR}/${b}" ]]; then
    cp "${BIN_DIR}/${b}" "${STAGE}/bin/"
    copied=$((copied + 1))
  else
    echo "WARN: missing ${BIN_DIR}/${b}[.exe]" >&2
  fi
done
if [[ "$copied" -lt 2 ]]; then
  echo "ERROR: expected at least llmoc_server + llmoc_server_int4 under ${BIN_DIR}" >&2
  exit 1
fi

cp configs/engine.yaml configs/engine_int4.yaml "${STAGE}/configs/"
cp README.md "${STAGE}/"
[[ -f README.en.md ]] && cp README.en.md "${STAGE}/"
[[ -f docs/USAGE.md ]] && cp docs/USAGE.md "${STAGE}/docs/"
[[ -f docs/PLATFORM.md ]] && cp docs/PLATFORM.md "${STAGE}/docs/"

cat > "${STAGE}/RUN.txt" <<EOF
llm-on-cpu application package (${PLATFORM_ID})
version: ${VER}

1) Place model weights (see docs/USAGE.md), e.g. models/Qwen3.5-4B.lwc
2) Edit configs/engine.yaml or configs/engine_int4.yaml if needed
3) Start server:

   Windows:  .\\bin\\llmoc_server.exe --config .\\configs\\engine.yaml
             .\\bin\\llmoc_server_int4.exe --config .\\configs\\engine_int4.yaml

   Linux/macOS:
             ./bin/llmoc_server --config ./configs/engine.yaml
             ./bin/llmoc_server_int4 --config ./configs/engine_int4.yaml

4) Open http://127.0.0.1:15085/

Notes:
- Windows may need Visual C++ Redistributable (x64) for OpenMP runtime.
- macOS package is for logic/API validation; production target is Linux (SPR).
- Models are NOT included (download/convert separately).
EOF

ZIP_PATH="${DIST}/llm-on-cpu-${VER}-${PLATFORM_ID}.zip"
rm -f "${ZIP_PATH}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON=python3
  elif command -v python >/dev/null 2>&1; then PYTHON=python
  else echo "ERROR: python3/python required to zip" >&2; exit 1; fi
fi

"${PYTHON}" - <<PY
import zipfile
from pathlib import Path
stage = Path(r"${STAGE}")
out = Path(r"${ZIP_PATH}")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in stage.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(stage.parent).as_posix())
print("wrote", out, "bytes", out.stat().st_size)
PY

echo "OK ${ZIP_PATH}"

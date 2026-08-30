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

mkdir -p "${STAGE}/bin" "${STAGE}/configs" "${STAGE}/docs" "${STAGE}/tools" "${STAGE}/models"

BINS=(
  llmoc_server
  llmoc_server_int4
  llmoc_server_glm
  make_fake_glmq
  lwc_verify
  m0_bandwidth
  m0_isa
  int4_gemm_bench
)

copied=0
have_glm=0
for b in "${BINS[@]}"; do
  if [[ -f "${BIN_DIR}/${b}.exe" ]]; then
    cp "${BIN_DIR}/${b}.exe" "${STAGE}/bin/"
    copied=$((copied + 1))
    [[ "$b" == "llmoc_server_glm" ]] && have_glm=1
  elif [[ -f "${BIN_DIR}/${b}" ]]; then
    cp "${BIN_DIR}/${b}" "${STAGE}/bin/"
    copied=$((copied + 1))
    [[ "$b" == "llmoc_server_glm" ]] && have_glm=1
  else
    echo "WARN: missing ${BIN_DIR}/${b}[.exe]" >&2
  fi
done
if [[ "$copied" -lt 2 ]]; then
  echo "ERROR: expected at least llmoc_server + llmoc_server_int4 under ${BIN_DIR}" >&2
  exit 1
fi
if [[ "$have_glm" -eq 0 ]]; then
  echo "ERROR: llmoc_server_glm missing under ${BIN_DIR} (GLM must be in the app package)" >&2
  exit 1
fi

cp configs/engine.yaml configs/engine_int4.yaml \
   configs/engine_glm_int4.yaml configs/engine_glm_nvfp4.yaml "${STAGE}/configs/"
cp README.md "${STAGE}/"
[[ -f README.en.md ]] && cp README.en.md "${STAGE}/"
[[ -f docs/USAGE.md ]] && cp docs/USAGE.md "${STAGE}/docs/"
[[ -f docs/PLATFORM.md ]] && cp docs/PLATFORM.md "${STAGE}/docs/"
[[ -f docs/MODEL_GLM53_FLASH.md ]] && cp docs/MODEL_GLM53_FLASH.md "${STAGE}/docs/"

# 发布包一键脚本 + Node 工具链（下载/转换/量化；需本机 Node≥18）
for t in download_model.mjs convert_lwc.mjs prepare_model.mjs quantize_int4.mjs; do
  cp "tools/${t}" "${STAGE}/tools/"
done
mkdir -p "${STAGE}/tools/glm"
for t in convert_glm_lwc.mjs quantize_glm_awq.mjs import_glm_nvfp4.mjs prepare_glm.mjs; do
  cp "tools/glm/${t}" "${STAGE}/tools/glm/"
done
mkdir -p "${STAGE}/models/recipes"
[[ -f models/recipes/glm53_flash.json ]] && cp models/recipes/glm53_flash.json "${STAGE}/models/recipes/"
[[ -f models/recipes/qwen3_5.json ]] && cp models/recipes/qwen3_5.json "${STAGE}/models/recipes/"

cp scripts/dist/download_bf16.cmd scripts/dist/download_int4.cmd \
   scripts/dist/download_glm.cmd \
   scripts/dist/start_bf16.cmd scripts/dist/start_int4.cmd \
   scripts/dist/start_glm.cmd "${STAGE}/"
cp scripts/dist/download_bf16.sh scripts/dist/download_int4.sh \
   scripts/dist/download_glm.sh \
   scripts/dist/start_bf16.sh scripts/dist/start_int4.sh \
   scripts/dist/start_glm.sh "${STAGE}/"
chmod +x "${STAGE}/download_bf16.sh" "${STAGE}/download_int4.sh" \
         "${STAGE}/download_glm.sh" \
         "${STAGE}/start_bf16.sh" "${STAGE}/start_int4.sh" \
         "${STAGE}/start_glm.sh" || true
# 空 models 占位，避免用户找不到目录
: > "${STAGE}/models/.gitkeep"

cat > "${STAGE}/RUN.txt" <<EOF
llm-on-cpu application package (${PLATFORM_ID})
version: ${VER}

Need: Node.js >= 18 on PATH.

download_* = auto pipeline (skip steps already done):
  detect HF shards / good LWC / good QLWC / GLMQ -> download -> convert -> verify -> prune
  INT4 also quantizes and removes mid .lwc
  GLM default: LibertAIDAI/GLM-5.3-Flash-NVFP4 → nvfp4.glmq; optional --awq for AWQ path
  Force redo: --force / --force-download / --force-convert / --force-int4 / --force-quant

-- INT4 (recommended) --
  Windows:     download_int4.cmd
               start_int4.cmd
  Linux/macOS: ./download_int4.sh && ./start_int4.sh

-- BF16 (unquantized) --
  Windows:     download_bf16.cmd
               start_bf16.cmd
  Linux/macOS: ./download_bf16.sh && ./start_bf16.sh

-- GLM-5.3-Flash (default: LibertAIDAI/GLM-5.3-Flash-NVFP4) --
  Windows:     download_glm.cmd
               download_glm.cmd --awq
               start_glm.cmd
               start_glm.cmd configs\\engine_glm_int4.yaml
  Linux/macOS: ./download_glm.sh
               ./download_glm.sh --awq
               ./start_glm.sh
               ./start_glm.sh configs/engine_glm_int4.yaml
  Docs:        docs/MODEL_GLM53_FLASH.md

Optional: download_*.cmd --model org/name  (then edit configs)

Open http://127.0.0.1:15085/

Notes:
- Weights are NOT in the zip; run download_* once (large / slow).
- Logs: logs/llmoc-YYYY-MM-DD.log (set LLMOC_LOG_DIR to change; LLMOC_PROFILE=1 for layer timing).
- Windows may need VC++ Redistributable x64 (OpenMP).
- Server uses plain HTTP only (no OpenSSL DLL required).
- See docs/USAGE.md for details.
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

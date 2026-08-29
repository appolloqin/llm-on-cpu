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

# 发布包一键脚本 + Node 工具链（下载/转换/量化；需本机 Node≥18）
for t in download_model.mjs convert_lwc.mjs prepare_model.mjs quantize_int4.mjs; do
  cp "tools/${t}" "${STAGE}/tools/"
done
cp scripts/dist/download_bf16.cmd scripts/dist/download_int4.cmd \
   scripts/dist/start_bf16.cmd scripts/dist/start_int4.cmd "${STAGE}/"
cp scripts/dist/download_bf16.sh scripts/dist/download_int4.sh \
   scripts/dist/start_bf16.sh scripts/dist/start_int4.sh "${STAGE}/"
chmod +x "${STAGE}/download_bf16.sh" "${STAGE}/download_int4.sh" \
         "${STAGE}/start_bf16.sh" "${STAGE}/start_int4.sh" || true
# 空 models 占位，避免用户找不到目录
: > "${STAGE}/models/.gitkeep"

cat > "${STAGE}/RUN.txt" <<EOF
llm-on-cpu application package (${PLATFORM_ID})
version: ${VER}

前置: 安装 Node.js >= 18，并保证 node 在 PATH 中。

下载脚本均包含：下载 HF → 转引擎格式 → 删除原模型大权重
（保留 *-hf 里的 config/tokenizer 供服务加载；INT4 还会删掉中间 .lwc）

—— INT4（推荐，省内存）——
  Windows:     download_int4.cmd
               start_int4.cmd
  Linux/macOS: chmod +x download_int4.sh start_int4.sh
               ./download_int4.sh
               ./start_int4.sh

—— BF16（无量化）——
  Windows:     download_bf16.cmd
               start_bf16.cmd
  Linux/macOS: chmod +x download_bf16.sh start_bf16.sh
               ./download_bf16.sh
               ./start_bf16.sh

可选换模型: download_*.cmd --model org/name  （并改 configs 里 path）

浏览器打开 http://127.0.0.1:15085/

Notes:
- 权重不打进压缩包；首次需跑对应 download_*（体积大、耗时长）。
- Windows 可能需要 Visual C++ Redistributable (x64)（OpenMP）。
- macOS 包偏逻辑/API 验证；生产目标为 Linux。
- 详细说明见 docs/USAGE.md。
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

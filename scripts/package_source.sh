#!/usr/bin/env bash
# llm-on-cpu :: 打源码包（不含 build/models/本地忽略文件）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VER="${PKG_VER:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
DIST="${ROOT}/dist"
STAGE="${DIST}/llm-on-cpu-${VER}-src"
ZIP_PATH="${DIST}/llm-on-cpu-${VER}-src.zip"

rm -rf "${STAGE}"
mkdir -p "${STAGE}"
git archive --format=tar --prefix="llm-on-cpu/" HEAD | tar -x -C "${STAGE}"

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

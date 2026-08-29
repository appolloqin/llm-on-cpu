#!/usr/bin/env bash
# llm-on-cpu :: Linux/macOS 构建(生产主目标为 Linux; macOS 仅逻辑层验证)
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build/release}
GENERATOR="Ninja"
if ! command -v ninja >/dev/null 2>&1; then GENERATOR="Unix Makefiles"; fi

cmake -S . -B "$BUILD_DIR" -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR"

echo "--- run unit tests ---"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "--- done ---"

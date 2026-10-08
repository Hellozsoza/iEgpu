#!/usr/bin/env bash
# Build the pinned Linux engine and a RAM-only development RPC worker.
# BACKEND=vulkan adds local GPU support (Mesa/Vulkan headers required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ $(uname -s) != Linux ]]; then
  echo "build-linux: run this on Linux" >&2
  exit 1
fi
if [[ ! -f llama.cpp/CMakeLists.txt ]]; then
  git submodule update --init --recursive
fi
# Apply Linux portability fixes without moving the engine pin or changing its math.
for PATCH in "$ROOT/patches/llama-linux.patch" "$ROOT/patches/llama-remote-stream.patch"; do
  if git -C llama.cpp apply --check "$PATCH" 2>/dev/null; then
    git -C llama.cpp apply "$PATCH"
  elif ! git -C llama.cpp apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "build-linux: $PATCH does not match this engine; inspect local llama.cpp changes" >&2
    exit 1
  fi
done
BACKEND=${BACKEND:-cpu}
case "$BACKEND" in
  cpu) VULKAN=OFF ;;
  vulkan) VULKAN=ON ;;
  *) echo "BACKEND must be cpu or vulkan" >&2; exit 1 ;;
esac
cmake -S llama.cpp -B "build/linux-$BACKEND" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF \
  -DGGML_VULKAN="$VULKAN" \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF
# Bound memory use on laptops; JOBS may override the default.
if [[ -z "${JOBS:-}" ]]; then
  JOBS=$(nproc)
  if (( JOBS > 4 )); then JOBS=4; fi
fi
cmake --build "build/linux-$BACKEND" --target llama-server ggml-rpc-server -j "$JOBS"
echo "Built engine and development worker. Run ./iegpu --help"

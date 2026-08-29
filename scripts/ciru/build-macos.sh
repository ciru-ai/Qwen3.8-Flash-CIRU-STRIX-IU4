#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-metal}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_HIP=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF

cmake --build "${build_dir}" \
    --target llama-server llama-cli llama-bench \
    -j "${BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}"

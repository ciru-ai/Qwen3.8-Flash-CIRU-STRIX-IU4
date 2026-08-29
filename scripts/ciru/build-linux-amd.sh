#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rocm_root="${ROCM_ROOT:-/opt/rocm}"
build_dir="${BUILD_DIR:-${repo_root}/build-gfx1151}"
gpu_target="${GPU_TARGET:-gfx1151}"

if [[ -x "${rocm_root}/bin/amdclang++" ]]; then
    hip_compiler="${rocm_root}/bin/amdclang++"
elif [[ -x "${rocm_root}/bin/hipcc" ]]; then
    hip_compiler="${rocm_root}/bin/hipcc"
else
    echo "No AMD HIP compiler found under ROCM_ROOT=${rocm_root}" >&2
    exit 2
fi

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="${hip_compiler}" \
    -DCMAKE_PREFIX_PATH="${rocm_root}" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_MTMD=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_LLGUIDANCE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_HIP=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_SYCL=OFF \
    -DGPU_TARGETS="${gpu_target}" \
    -DAMDGPU_TARGETS="${gpu_target}"

cmake --build "${build_dir}" \
    --target llama-server llama-cli llama-bench \
    -j "${BUILD_JOBS:-$(nproc)}"

echo "Built release binaries under ${build_dir}/bin"

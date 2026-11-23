#!/usr/bin/env bash
# Configure the same kernel target and compiler modes used for the Sozo candidate.
set -euo pipefail
src=$(realpath "${1:?usage: build-gfx1151.sh SOURCE_DIRECTORY BUILD_DIRECTORY BORINGSSL_SOURCE_DIRECTORY}")
build=$(realpath -m "${2:?pass a build directory}")
boringssl=$(realpath "${3:?pass the pinned BoringSSL source directory}")
sdk=$(realpath "${ROCM_PATH:-/opt/rocm}")
clang=$sdk/llvm/bin/clang
clangxx=$sdk/llvm/bin/clang++
test -x "$clang"
test -x "$clangxx"
test -f "$boringssl/CMakeLists.txt"
test -f "$src/scripts/ciru/flash-v6/gdn-registered.hip"

export HIP_PLATFORM=amd ROCM_PATH="$sdk" HIP_PATH="$sdk"
export HIP_DEVICE_LIB_PATH="$sdk/amdgcn/bitcode"
export HIPFLAGS="--rocm-path=$sdk --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH"
cmake -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_C_COMPILER="$clang" -DCMAKE_CXX_COMPILER="$clangxx" \
    -DCMAKE_CXX_FLAGS="-I$sdk/include" -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_CROSSCOMPILING=ON -DCMAKE_HIP_COMPILER="$clangxx" -DCMAKE_PREFIX_PATH="$sdk" \
    -DAMDGPU_TARGETS=gfx1151 -DGPU_TARGETS=gfx1151 -DGGML_HIP=ON \
    -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DFETCHCONTENT_SOURCE_DIR_BORINGSSL="$boringssl" -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_BORINGSSL=ON -DGGML_STATIC=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DLLAMA_BUILD_UI=OFF \
    -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_EXPORT_METRICS=OFF \
    -DGGML_CCACHE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$build" --target llama-bench llama-server --parallel "${BUILD_JOBS:-4}"
for kernel in registered norm; do
    "$clangxx" -x hip --rocm-path="$sdk" --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" \
        --offload-arch=gfx1151 -O3 -std=c++17 -fPIC -shared \
        "$src/scripts/ciru/flash-v6/gdn-$kernel.hip" -o "$build/bin/libciru-gdn-$kernel-gfx1151.so" \
        -L "$sdk/lib" -lamdhip64
done
(
    cd "$build/bin"
    sha256sum llama-server llama-bench libggml-hip.so.1.0.0 \
        libllama-common.so.1.0.0 libllama.so.1.0.0 libllama-server-impl.so libciru-gdn-*.so
) > "$build/FLASH-V6-BUILD-SHA256SUMS"

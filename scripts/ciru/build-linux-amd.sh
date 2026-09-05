#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
check_only=0
case "${1:-}" in
    --check) check_only=1; shift ;;
    --help|-h)
        echo "Usage: $0 [--check]"
        echo "ROCM_ROOT selects a complete SDK. --check configures in a temporary directory without building or using a GPU."
        exit 0 ;;
esac
[[ $# == 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

fail() {
    echo "AMD build preflight: $*" >&2
    echo "See docs/BUILD_LINUX.md, or use ./scripts/ciru/setup-linux-amd.sh for an isolated SDK." >&2
    exit 2
}

for tool in cmake ninja cc c++ readlink; do
    command -v "${tool}" >/dev/null || fail "Missing host tool: ${tool}"
done

rocm_root="${ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}"
[[ -d "${rocm_root}" ]] || fail "ROCM_ROOT does not exist: ${rocm_root}"
rocm_root="$(cd "${rocm_root}" && pwd -P)"
build_dir="${BUILD_DIR:-${repo_root}/build-gfx1151}"
gpu_target="${GPU_TARGET:-gfx1151}"

hip_compiler=""
for candidate in bin/amdclang++ llvm/bin/clang++ bin/clang++; do
    if [[ -x "${rocm_root}/${candidate}" ]]; then
        hip_compiler="${rocm_root}/${candidate}"
        break
    fi
done
[[ -n "${hip_compiler}" ]] || fail "No AMD clang compiler under ${rocm_root}. CMake's HIP language cannot use hipcc as CMAKE_HIP_COMPILER."

# Pin each package to this SDK instead of falling back to another installation.
package_args=()
for package in hip hipblas rocblas; do
    package_dir=""
    for lib_dir in "${rocm_root}/lib" "${rocm_root}/lib64" "${rocm_root}"/lib/*-linux-gnu; do
        for config in "${package}-config.cmake" "${package}Config.cmake"; do
            if [[ -f "${lib_dir}/cmake/${package}/${config}" ]]; then
                package_dir="${lib_dir}/cmake/${package}"
                break 2
            fi
        done
    done
    [[ -n "${package_dir}" ]] || fail "Missing ${package} development CMake package under ${rocm_root}; install the complete matching SDK, including hipBLAS and rocBLAS."
    package_args+=("-D${package}_DIR=${package_dir}")
    echo "${package}: ${package_dir}"
done

if (( check_only )); then
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ciru-amd-check.XXXXXX")"
    trap 'rm -rf -- "${build_dir}"' EXIT
elif [[ -f "${build_dir}/CMakeCache.txt" ]]; then
    cached_compiler="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "${build_dir}/CMakeCache.txt")"
    if [[ -n "${cached_compiler}" && "$(readlink -f "${cached_compiler}")" != "$(readlink -f "${hip_compiler}")" ]]; then
        fail "${build_dir} was configured with ${cached_compiler}. Select a new BUILD_DIR when changing SDKs; existing builds are not removed."
    fi
fi

export ROCM_PATH="${rocm_root}"
export HIP_PATH="${rocm_root}"
export HIP_PLATFORM=amd
export PATH="${rocm_root}/bin:${PATH}"

if ! cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="${hip_compiler}" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="${rocm_root}" \
    -DCMAKE_PREFIX_PATH="${rocm_root}" \
    "-DCMAKE_BUILD_RPATH=${rocm_root}/lib;${rocm_root}/lib64" \
    "${package_args[@]}" \
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
    -DAMDGPU_TARGETS="${gpu_target}"; then
    fail "CMake configuration failed. An AMDDeviceLibs include error indicates an incomplete SDK or mixed package paths. Do not repair it with cross-version symlinks."
fi

if (( check_only )); then
    echo "AMD SDK configuration passed for ${gpu_target}. No GPU workload or model was started."
    exit 0
fi

cmake --build "${build_dir}" \
    --target llama-server llama-cli llama-bench \
    -j "${BUILD_JOBS:-$(nproc)}"

echo "Built release binaries under ${build_dir}/bin"

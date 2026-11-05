#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${ROCM_ROOT:?Set ROCM_ROOT to the complete ROCm 10.0.0 SDK; see INSTALL.md.}"
sdk="$(cd "$sdk" && pwd -P)"
prefix="${CIRU_RUNTIME_ROOT:-$repo_root/runtime}"
mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd -P)"
sources="$repo_root/dependencies/rocm-systems"
revision=7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1
jobs="${BUILD_JOBS:-2}"
for tool in git cmake ninja cc c++ pkg-config python3; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; see INSTALL.md" >&2; exit 2; }
done
python3 -c 'import CppHeaderParser, ply; assert ply.__version__ == "3.11"'
for module in libdrm libdrm_amdgpu libelf; do pkg-config --exists "$module"; done
if [[ ! -d "$sources/.git" ]]; then
  mkdir -p "$(dirname "$sources")"
  git clone --filter=blob:none https://github.com/pwilkin/rocm-systems.git "$sources"
fi
if [[ -n "$(git -C "$sources" status --porcelain)" ]]; then
  echo "The runtime dependency checkout has edits; use a fresh source package." >&2; exit 2
fi
git -C "$sources" fetch origin "$revision"
git -C "$sources" checkout --detach "$revision"
[[ "$(git -C "$sources" rev-parse HEAD)" == "$revision" ]]
llvm="$sdk/lib/llvm"
[[ -d "$llvm" ]] || llvm="$sdk/llvm"
[[ -x "$llvm/bin/clang++" ]] || { echo "SDK LLVM compiler missing: $llvm" >&2; exit 2; }
export PATH="$sdk/bin:$PATH"
export HIP_PLATFORM=amd
common=(
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  "-DLLVM_DIR=$llvm/lib/cmake/llvm" "-DClang_DIR=$llvm/lib/cmake/clang"
  "-DCMAKE_INCLUDE_PATH=$sdk/lib/rocm_sysdeps/include"
  "-DCMAKE_LIBRARY_PATH=$sdk/lib/rocm_sysdeps/lib;$sdk/lib"
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
)
cmake -S "$sources/projects/rocr-runtime" -B "$repo_root/build-rocr-v5" \
  "${common[@]}" "-DCMAKE_INSTALL_PREFIX=$prefix/rocr" \
  "-DCMAKE_PREFIX_PATH=$sdk;$sdk/lib/rocm_sysdeps;$llvm"
cmake --build "$repo_root/build-rocr-v5" --parallel "$jobs"
cmake --install "$repo_root/build-rocr-v5"
cmake -S "$sources/projects/clr" -B "$repo_root/build-hip-v5" \
  "${common[@]}" "-DCMAKE_INSTALL_PREFIX=$prefix/hip" \
  "-DCMAKE_PREFIX_PATH=$prefix/rocr;$sdk;$sdk/lib/rocm_sysdeps;$llvm" \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  "-DHIP_COMMON_DIR=$sources/projects/hip" "-DHIPCC_BIN_DIR=$sdk/bin" \
  "-DLLVM_ROOT=$llvm" "-DHIP_LLVM_ROOT=$llvm" "-DClang_ROOT=$llvm" \
  "-DROCM_PATH=$prefix/rocr" "-Dhsa-runtime64_DIR=$prefix/rocr/lib/cmake/hsa-runtime64" \
  "-DPython3_EXECUTABLE=$(command -v python3)" \
  -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF \
  -DHIP_ENABLE_ROCPROFILER_REGISTER=ON -DUSE_PROF_API=ON -D__HIP_ENABLE_PCH=ON
cmake --build "$repo_root/build-hip-v5" --parallel "$jobs"
cmake --install "$repo_root/build-hip-v5"
test -f "$prefix/hip/lib/libamdhip64.so"
test -f "$prefix/rocr/lib/libhsa-runtime64.so"
printf 'Custom HIP and ROCr installed: %s\n' "$prefix"

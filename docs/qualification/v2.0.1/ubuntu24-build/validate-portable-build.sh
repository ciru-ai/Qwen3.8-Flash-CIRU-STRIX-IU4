#!/usr/bin/env bash
set -euo pipefail
cd /work/source
export DEBIAN_FRONTEND=noninteractive
export BUILD_JOBS=4
export BUILD_DIR=/work/build-gfx1151-sdk
export ROCM_VENV=/work/rocm-sdk-venv
export PIP_NO_CACHE_DIR=1
export CCACHE_DISABLE=1
printf 'Build started UTC: '; date -u --iso-8601=seconds
# The release target is AMD; do not inherit Intel build-host native features.
cmake -S /work/source -B "$BUILD_DIR" \
  -DGGML_NATIVE=OFF \
  -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_BMI2=ON \
  -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
./scripts/ciru/setup-linux-amd.sh
./scripts/ciru/setup-linux-amd.sh --check
mkdir -p /work/checks
{
  cat /etc/os-release
  cmake --version
  ninja --version
  cc --version
  c++ --version
  python3 --version
  "$ROCM_VENV/bin/python" -m pip freeze
  "$ROCM_VENV/bin/rocm-sdk" path --root
  "$("$ROCM_VENV/bin/rocm-sdk" path --root)/bin/amdclang++" --version
} > /work/checks/toolchain.txt
for bin in llama-server llama-cli llama-bench; do
  ldd "$BUILD_DIR/bin/$bin" > "/work/checks/$bin.ldd.txt"
  if grep -q 'not found' "/work/checks/$bin.ldd.txt"; then exit 1; fi
  readelf -d "$BUILD_DIR/bin/$bin" > "/work/checks/$bin.dynamic.txt"
  "$BUILD_DIR/bin/$bin" --version > "/work/checks/$bin.version.txt" 2>&1
  "$BUILD_DIR/bin/$bin" --help > "/work/checks/$bin.help.txt" 2>&1
  sha256sum "$BUILD_DIR/bin/$bin"
done
readelf -d "$BUILD_DIR/bin/libggml-hip.so" > /work/checks/libggml-hip.dynamic.txt
ldd "$BUILD_DIR/bin/libggml-hip.so" > /work/checks/libggml-hip.ldd.txt
if grep -q 'not found' /work/checks/libggml-hip.ldd.txt; then exit 1; fi
grep -E 'ple-sidecar|spec-draft-model' /work/checks/llama-server.help.txt
printf 'Build verified UTC: '; date -u --iso-8601=seconds
touch /work/BUILD-PASSED

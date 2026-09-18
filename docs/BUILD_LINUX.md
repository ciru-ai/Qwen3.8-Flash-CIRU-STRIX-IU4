# Build v4.4.1

The v4.4.1 NixOS package rebuilds `libllama-common` for the vision/MTP fix and retains the v4.4.0 inference kernels and pinned HIP/ROCr libraries. The compiler is stock TheRock ROCm10 with the recorded Nix host dependencies. See `binary-identity.json`; this is not a portable Ubuntu binary.

The existing Ubuntu/Debian engine setup below remains the source-build entry point. To reproduce the new PM4 runtime path, build pwilkin's HIP/ROCr at commit `7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1` separately. A mainstream Linux build outline is below; only the recorded NixOS build was executed and qualified here. It needs CMake 3.27+, Ninja, a complete TheRock10 SDK, libdrm, libelf, NUMA, OpenGL development files, CppHeaderParser 2.7.4 and ply 3.11. On Ubuntu/Debian install `build-essential ninja-build pkg-config libdrm-dev libelf-dev libnuma-dev libgl-dev python3-venv`; use a venv for the Python packages.

```bash
export ROCM_ROOT=/path/to/complete/therock-10-sdk
export CIRU_RUNTIME_ROOT="$PWD/runtime"
git clone --filter=blob:none https://github.com/pwilkin/rocm-systems.git rocm-systems
git -C rocm-systems checkout 7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1
cmake -S rocm-systems/projects/rocr-runtime -B build-rocr -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$CIRU_RUNTIME_ROOT/rocr" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT;$ROCM_ROOT/lib/rocm_sysdeps;$ROCM_ROOT/lib/llvm"
cmake --build build-rocr --parallel 4
cmake --install build-rocr
cmake -S rocm-systems/projects/clr -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CIRU_RUNTIME_ROOT/hip" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH="$CIRU_RUNTIME_ROOT/rocr;$ROCM_ROOT;$ROCM_ROOT/lib/rocm_sysdeps;$ROCM_ROOT/lib/llvm" \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  -DHIP_COMMON_DIR="$PWD/rocm-systems/projects/hip" -DHIPCC_BIN_DIR="$ROCM_ROOT/bin" \
  -DLLVM_ROOT="$ROCM_ROOT/lib/llvm" -DHIP_LLVM_ROOT="$ROCM_ROOT/lib/llvm" \
  -DClang_ROOT="$ROCM_ROOT/lib/llvm" -DROCM_PATH="$CIRU_RUNTIME_ROOT/rocr" \
  -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF \
  -DHIP_ENABLE_ROCPROFILER_REGISTER=ON -DUSE_PROF_API=ON -D__HIP_ENABLE_PCH=ON
cmake --build build-hip --parallel 4
cmake --install build-hip
```

CIRU's launcher selects these two directories before the engine/SDK libraries. Check its printed loader identity. The source-build outline is not evidence of speed or numerical equivalence on another distribution. See the published custom-runtime build receipt for the exact qualified configuration and local Nix adaptations. The historical build notes below describe previous releases, not v4.4 binary identity.

## Retained engine build instructions

# Build the v4.3.0 source package

The tested binary is specific to its recorded NixOS and stock TheRock ROCm10 dependencies. It is not a portable Ubuntu binary. Other systems build the matching release source.

The retained setup helper installs a private ROCm10 SDK and, when requested, Ubuntu/Debian host tools including `libssl-dev`. It does not change GPU drivers or start a model:

```bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
```

For an existing complete SDK, set `ROCM_ROOT` and a fresh `BUILD_DIR`, then run `scripts/ciru/build-linux-amd.sh`. The helper requests OpenSSL; inspect CMake's effective result before claiming HTTPS support. `LLAMA_OPENSSL=ON` alone is insufficient: the retained Nix v3 build printed `OpenSSL not found, HTTPS support disabled`, with all OpenSSL include/library cache entries set to `NOTFOUND`.

The profile uses external UI assets. Inference builds explicitly set `LLAMA_BUILD_UI=OFF` and `LLAMA_USE_PREBUILT_UI=OFF`, avoiding a moving prebuilt-UI download. The release package includes its hashed `ui/` directory. To reproduce those assets separately from the pinned source:

```bash
cd tools/ui
npm ci --ignore-scripts --no-audit --no-fund
LLAMA_BUILD_NUMBER=ciru-v4.3.0-ui-f5daaa3 npm run build
```

Copy the resulting `dist/` contents to the release's `ui/` directory and create a new asset manifest. The UI build used Node v26.7.0 and the exact package-lock.json; see UI-PROVENANCE.json for the npm version, hashes and complete file inventory. This local UI build does not alter inference kernels or the server binary.

Keep the complete matched `bin/` executable/shared-library set. Never combine a new server with old `libllama`, `libggml`, `libmtmd` or server implementation libraries. Keep the SDK directory used at build time in place. For a relocated tested Nix package, use its exact recorded dependency paths and put that package's `bin/` ahead of older project libraries in `LD_LIBRARY_PATH`, followed by the SDK library directories as documented in its binary receipt.

The v4.3 UI preserves the qualified application assets and updates only its release version descriptor. The NixOS binary archive reuses the complete corrected v4.2 executable and library set. Only launch defaults, documentation and release metadata change for v4.3. Its embedded upstream build text remains R2; CIRU_RELEASE.json and binary-identity.json identify this complete v4.3 package. Source builds use the v4.3 Git revision.

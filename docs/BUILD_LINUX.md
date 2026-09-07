# Build on Linux and WSL2

## Support status

| Environment | Status | Notes |
|---|---|---|
| NixOS x86-64 + TheRock ROCm + gfx1151 | **v2.0.1 runtime validated** | Matched incremental GPU build; correctness and performance regression checks |
| Ubuntu 24.04 container + isolated ROCm 10.0.0 | **v2.0.1 clean build and GPU smoke validated** | ELF checks; 66 QSA cases and 30 batch tests; 520-token completion and MTP counts matched; NixOS host GPU driver |
| Ubuntu 26.04 + isolated TheRock ROCm 10.0.0 SDK | **Historical v1.1.1 build-validated** | Clean container, all three binaries; GPU inference remains unvalidated on this distro |
| Other Ubuntu/Debian + ROCm | Unvalidated | Requires a complete, matching SDK with gfx1151 support |
| Fedora/RHEL + ROCm | Source-compatible, unvalidated | Use the distro's supported ROCm packages |
| Arch + ROCm | Source-compatible, unvalidated | Rolling packages can change behavior |
| Ubuntu under WSL2 + ROCDXG | Experimental, unvalidated | Verify `/dev/dxg`; store PLE on WSL ext4 |
| Linux CPU-only | **v2.0.1 clean build validated** | 66 QSA mapping/state cases and batch-allocation test |

The performance comparisons use NixOS; Ubuntu qualification is a separate compatibility check. v2.0.1 passed a clean Ubuntu 24.04 / ROCm 10.0.0 build on Dunamis, plus ELF/help checks and the existing 66 QSA mapping/state cases and 30 batch-allocation tests (198 assertions). No prior build cache was imported. The build used the documented helper, with `GGML_NATIVE=OFF`, AVX2/F16C/FMA/BMI2 on and AVX512/VNNI off because Dunamis has an Intel CPU; the final compile commands verify these portable settings.

The resulting binaries ran in an Ubuntu 24.04 container on Ciru with its NixOS host GPU driver and matching ROCm 10 SDK. At 262144 configured context, the 57-token coding prompt produced all 520 reference tokens exactly, with 674 drafted / 400 accepted tokens. GPU device identity and all 15 binary hashes were verified; the original overly strict device-label verifier failure is preserved in the qualification report. This does not qualify a native Ubuntu host or a filled 262144-token conversation. See [qualification and separate build identities](QSA_BACKPORT_STATUS.md). The earlier v2.0 Ubuntu smoke remains historical; the build scripts and HIP sources are unchanged in v2.0.1.

Historical v1.1.1 Ubuntu build validation on 2026-09-05 used CMake 4.2.3, GNU 15.2.0, Python 3.14, and AMD's stable ROCm 10.0.0 SDK in a clean Ubuntu 26.04 container with no GPU devices. All three binaries compiled, the server's shared libraries resolved, and `--version`/`--help` returned successfully with the PLE and MTP options present. The expected no-GPU diagnostic in that container is not an inference test.

## Prerequisites

Install CMake 3.21 or newer (required by the HIP backend), Ninja, a C/C++ toolchain, Git, Python, and OpenSSL development headers. The HIP compiler, AMD device libraries, HIP development files, hipBLAS, and rocBLAS must come from one complete SDK. Setting `ROCM_ROOT` does not install these components.

Ubuntu/Debian host dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build libssl-dev python3-venv ca-certificates
```

Fedora/RHEL host dependencies:

```bash
sudo dnf group install -y "Development Tools"
sudo dnf install -y cmake git ninja-build openssl-devel python3
```

Arch host dependencies:

```bash
sudo pacman -S --needed base-devel cmake git ninja openssl python
```

## Ubuntu setup with an isolated SDK

From a checkout containing the setup script:

```bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
```

This installs the host build packages above with `apt`, then installs AMD's stable `rocm[libraries,devel,device-gfx1151]==10.0.0` into `.venv-rocm/`, expands the development SDK with `rocm-sdk init`, and builds the three release binaries into `build-gfx1151-sdk/`. The compiler and BLAS libraries come from the same pinned SDK, including the gfx1151 device package. Downloads and SDK/build files need additional disk space beyond the model package. The default is four build jobs; lower `BUILD_JOBS` on a busy machine.

The script does not install a GPU driver, edit `/opt/rocm`, change permissions or boot settings, manage services, download model weights, or launch a model. Run it without `sudo`; only the explicit host-package step uses sudo. If host tools are already installed, omit `--install-host-deps`. The SDK and build directories are reusable. Keep `.venv-rocm/` in place because the binaries link to its libraries. Use a new `ROCM_VENV` and `BUILD_DIR` when testing a different `ROCM_VERSION`.

### Reproduce the tested portable CPU configuration

The helper's fresh-build default is native CPU compilation. When building on another CPU for transfer to Strix Halo, use the explicit AVX2 configuration tested for v2.0.1. After setup has initialized `build-gfx1151-sdk/`, reconfigure that same directory and rebuild:

```bash
cmake -S . -B build-gfx1151-sdk \
  -DGGML_NATIVE=OFF \
  -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_BMI2=ON \
  -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
./scripts/ciru/setup-linux-amd.sh
```

If you changed the defaults, replace CMake's `-B build-gfx1151-sdk` with your chosen build directory, and pass the matching `BUILD_DIR` and `ROCM_VENV` to the setup helper. This rebuilds affected CPU objects while retaining unchanged HIP objects from that source build. The clean qualification used these settings on an Intel build host and ran the resulting binaries on Strix Halo; it does not require changing the runtime profile.

Check an existing private SDK without installing or building:

```bash
./scripts/ciru/setup-linux-amd.sh --check
```

This performs CMake configuration in a temporary directory and removes that directory afterward. It does not execute a GPU workload. Successful configuration alone does not verify inference.

Before launching on the intended Strix Halo host, verify the kernel driver and current user's GPU access:

```bash
test -r /dev/kfd && test -w /dev/kfd
ls -l /dev/dri/renderD*
./.venv-rocm/bin/rocminfo > /tmp/ciru-rocminfo.txt
grep gfx1151 /tmp/ciru-rocminfo.txt
```

The user needs access to `/dev/kfd` and the relevant render node. Follow [AMD's Strix Halo system guidance](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) for kernel and memory requirements and [TheRock's release instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for SDK details. A container or private SDK supplies user-space libraries; it still relies on a compatible host GPU driver.

Launch separately when the host has enough free memory:

```bash
BUILD_DIR="$PWD/build-gfx1151-sdk" \
  MODEL_DIR=/absolute/path/to/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  ./scripts/ciru/run-server.sh
```

## Build the gfx1151 release runtime

If you already have a complete ROCm/TheRock SDK, use the lower-level build script from this extracted v2.0.1 source directory. The v1.1.1 SDK setup fixes are retained.

~~~bash
ROCM_ROOT=/opt/rocm ./scripts/ciru/build-linux-amd.sh
~~~

The build script sets `ROCM_PATH` and `HIP_PATH` to the selected root, pins the HIP/hipBLAS/rocBLAS CMake packages there, uses AMD clang directly, and records SDK library directories in the build's runtime search paths. `hipcc` is not a valid `CMAKE_HIP_COMPILER` for CMake's native HIP language.

Override `ROCM_ROOT` when using a TheRock bundle or another self-contained ROCm distribution:

```bash
ROCM_ROOT=/path/to/therock-gfx1151 \
  BUILD_DIR=build-gfx1151 \
  ./scripts/ciru/build-linux-amd.sh
```

The script configures:

```text
Release; shared libraries; HIP on; HIP VMM off; HIP graphs on;
HIP MMQ MFMA on; CUDA/Vulkan/SYCL off; GPU_TARGETS=gfx1151;
AMDGPU_TARGETS=gfx1151
```

It builds `llama-server`, `llama-cli`, and `llama-bench`.

### Repairing the Ubuntu 26 build report

An error involving `/opt/rocm/lib/cmake/AMDDeviceLibs/../../llvm/lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake`, followed by a missing `hipblasConfig.cmake`, occurs during SDK discovery before the model is loaded. The report also resolves HIP from `/usr/lib/x86_64-linux-gnu/cmake/hip`, so it crosses the Ubuntu and `/opt/rocm` package layouts. The log cannot establish which packages or symlinks are missing; a fresh Ubuntu installation by itself does not supply a complete matching SDK at `/opt/rocm`.

Use the isolated setup above to avoid that mixed installation. Existing model, MTP, and PLE files can be reused. Select a new build directory when changing SDKs; do not reuse the failed CMake cache or add cross-version symlinks to make an include error disappear.

For a system SDK, first check its package origin and development packages. Ubuntu's archive ROCm packages normally use `/usr`; AMD's and TheRock's layouts differ. Ubuntu 26.04 supplies a `rocm` metapackage, but its toolchain has not been validated for this custom runtime. Do not combine archive `libhipblas-dev` with a different AMD SDK. A compiler or `rocminfo` alone is insufficient. Diagnose a selected SDK with:

```bash
ROCM_ROOT=/path/to/complete/sdk ./scripts/ciru/build-linux-amd.sh --check
```

The ccache warning is optional and does not cause either reported error.

### NixOS

The measured build used a pinned TheRock toolchain in a Nix environment with explicit AMD-clang GCC/glibc include, link, and runtime paths. The generic script above works when your dev shell has already supplied a coherent C/C++ sysroot. Do not mix an AMD clang from one environment with headers or libraries from another.

## WSL2 / ROCDXG

The WSL2 path is experimental for this release.

1. Install the AMD Windows driver and the matching ROCDXG user-space stack inside Ubuntu WSL.
2. Confirm the device and architecture:

   ```bash
   test -e /dev/dxg
   rocminfo | grep -m1 gfx1151
   ```

3. Clone and build with the same `build-linux-amd.sh` script.
4. Download the model under the WSL Linux filesystem, for example `~/models/Qwen3.8-Flash-CIRU-STRIX-IU4`.

Do **not** place `ple/ple.payload.bin` under `/mnt/c`, `/mnt/d`, or another DrvFS mount. The optimized Linux pager opens the payload using `O_DIRECT`; the filesystem must support it.

## CPU-only compatibility build

This path verifies that the clean source export builds without ROCm. It is not practical for the advertised model performance.

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_HIP=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cpu --target llama-server llama-cli llama-bench -j "$(nproc)"
```

## Verify the build

```bash
BUILD_DIR="${BUILD_DIR:-$PWD/build-gfx1151-sdk}" # use build-gfx1151 for a manual SDK build
"$BUILD_DIR/bin/llama-server" --version
"$BUILD_DIR/bin/llama-server" --help | grep -E 'ple-sidecar|spec-draft-model'
```

Then follow [RUNNING.md](RUNNING.md). A successful compile on an unvalidated environment is not proof of the full PLE/MTP runtime path; begin with a short local smoke request and inspect the server log.

# Build on Linux and WSL2

## Support status

| Environment | Status | Notes |
|---|---|---|
| NixOS x86-64 + TheRock ROCm + gfx1151 | **Validated** | Release performance path |
| Ubuntu/Debian + ROCm | Source-compatible, unvalidated | Use the distro's supported ROCm packages |
| Fedora/RHEL + ROCm | Source-compatible, unvalidated | Use the distro's supported ROCm packages |
| Arch + ROCm | Source-compatible, unvalidated | Rolling packages can change behavior |
| Ubuntu under WSL2 + ROCDXG | Experimental, unvalidated | Verify `/dev/dxg`; store PLE on WSL ext4 |
| Linux CPU-only | Build-validated | Compatibility and diagnostics only |

Only the first row is represented by the published speed numbers.

## Prerequisites

Install CMake 3.14 or newer, Ninja, a C/C++ toolchain, Git, Python, and OpenSSL development headers. Install an AMD ROCm distribution that recognizes the APU as `gfx1151`.

Ubuntu/Debian host dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build libssl-dev python3
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

ROCm packaging and kernel requirements vary by distribution. Finish the vendor installation first, then require both commands below to succeed:

```bash
rocminfo | grep -m1 gfx1151
hipconfig --full
```

## Build the gfx1151 release runtime

```bash
git clone --branch v1.0.0-h121 \
  https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git
cd Qwen3.8-Flash-CIRU-STRIX-IU4
ROCM_ROOT=/opt/rocm ./scripts/ciru/build-linux-amd.sh
```

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
./build-gfx1151/bin/llama-server --version
./build-gfx1151/bin/llama-server --help | grep -E 'ple-sidecar|spec-draft-model'
```

Then follow [RUNNING.md](RUNNING.md). A successful compile on an unvalidated environment is not proof of the full PLE/MTP runtime path; begin with a short local smoke request and inspect the server log.

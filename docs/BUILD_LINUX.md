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
git clone --branch v3.0.0 \
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

The WSL2 path is experimental for this release but is the only Windows path
that reaches the Strix Halo GPU. It was verified end to end on Windows 11
25H2 with WSL 2.7.12, Ubuntu 26.04.1, ROCm 10.0 (`amdgpu-install` 31.50,
`--no-dkms`), and `rocdxg-roct` 1.2.2; measured results and a full recipe
are in [BUILD_WINDOWS.md](BUILD_WINDOWS.md).

1. Install WSL2 and Ubuntu (26+ recommended, 26.04.1 verified; 24.04
   probably works; other distros are not supported). A reboot is
   required after `wsl --install --no-distribution`. A large Strix Halo GPU
   carve-out leaves only ~31.6 GiB visible to Windows; cap the VM with a
   `.wslconfig` (`memory=28GB`).
2. Inside the distro, install ROCm 10 with the Windows-driver-matching
   installer and **no kernel driver** (WSL has no amdgpu module):

   ```bash
   wget -q https://repo.radeon.com/amdgpu-install/31.50/ubuntu/resolute/amdgpu-install_31.50.315000-1_all.deb
   apt-get install -y ./amdgpu-install_31.50.315000-1_all.deb
   amdgpu-install --usecase=rocm --no-dkms -y
   ```

   The cmake build additionally needs `amdrocm-runtime-dev10.0` (provides
   `hip-lang-config.cmake`) and `amdrocm-blas10.0-gfx1151` +
   `amdrocm-blas-dev10.0` (provides `hipblasConfig.cmake`). Do not install
   the distro's `libhipblas-dev` (7.1.x); it lacks the ROCm 10 cmake configs.
3. Install the ROCDXG runtime bridge, then confirm the device:

   ```bash
   wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
   apt-get install -y ./rocdxg-roct_1.2.2_amd64.deb
   test -e /dev/dxg
   rocminfo | grep -m1 gfx1151
   ```

4. Clone and build with the same `build-linux-amd.sh` script.
5. Download the model under the WSL Linux filesystem, for example
   `~/models/Qwen3.8-Flash-CIRU-STRIX-IU4`.

Do **not** place `ple/ple.payload.bin` under `/mnt/c`, `/mnt/d`, or another
DrvFS mount. The optimized Linux pager opens the payload using `O_DIRECT`;
the filesystem must support it.

Two WSL 2.6+ (2.7.12) lifecycle notes, both covered in
[BUILD_WINDOWS.md](BUILD_WINDOWS.md): when the last `wsl.exe` client
detaches, WSL tears the session down - without `vmIdleTimeout=-1` the VM
powers off, and even with it set, systemd units are stopped roughly 15 s
after detach (confirmed regression,
[microsoft/WSL#13416](https://github.com/microsoft/WSL/issues/13416)).
Fix with `vmIdleTimeout=-1` in `.wslconfig` plus a systemd "self-keeper"
unit whose `ExecStart` runs `wsl.exe` against the distro itself, keeping a
client session permanently attached. A systemd unit is also more reliable
than a `[boot] command`, which can re-fire on session starts.

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

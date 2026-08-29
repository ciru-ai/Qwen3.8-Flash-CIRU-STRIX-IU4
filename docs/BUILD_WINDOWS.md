# Build on Windows

## Recommended AMD route: WSL2

For Strix Halo GPU acceleration, use Ubuntu under WSL2 with AMD's ROCDXG stack and follow [BUILD_LINUX.md](BUILD_LINUX.md#WSL2--ROCDXG). That path is experimental and is not yet represented by the published performance figures.

Keep the 52.4 GB PLE payload on WSL's native Linux filesystem, not an `/mnt/c` path.

## Native Windows CPU compatibility build

Native Windows is currently a CPU compatibility and source-build path. It does not provide the release's validated gfx1151 HIP fast path, Linux `O_DIRECT` PLE pager, or advertised performance.

Prerequisites:

- Visual Studio 2022 with **Desktop development with C++**
- CMake available from a Developer PowerShell
- Git

From **Developer PowerShell for VS 2022**:

```powershell
git clone --branch v1.0.0-h121 https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git
Set-Location Qwen3.8-Flash-CIRU-STRIX-IU4

cmake -S . -B build-win-cpu -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HIP=OFF `
  -DGGML_CUDA=OFF `
  -DGGML_VULKAN=OFF `
  -DGGML_METAL=OFF `
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build-win-cpu --config Release --target llama-server llama-cli llama-bench -j
```

Expected binaries are under `build-win-cpu\bin\Release\` or `build-win-cpu\bin\`, depending on generator and CMake version.

This compile recipe is provided for portability work and diagnostics; full model execution on native Windows has not been release-validated. Use WSL2 or the validated Linux stack for the actual Strix profile.

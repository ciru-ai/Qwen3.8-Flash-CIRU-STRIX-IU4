[CmdletBinding()]
param(
    [string]$BuildDir = "build-win-cpu"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

cmake -S $repoRoot -B $BuildDir -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_HIP=OFF `
    -DGGML_CUDA=OFF `
    -DGGML_VULKAN=OFF `
    -DGGML_METAL=OFF `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_TESTS=OFF

cmake --build $BuildDir --config Release --target llama-server llama-cli llama-bench -j

# Build on macOS

macOS is an unvalidated compatibility path. The source has portable CPU support for the stored GGUF types, and `llama.cpp` can build with Metal, but CIRU's gfx1151 IU4 kernels and Linux P16/`O_DIRECT` PLE prefill path are not available. No published performance or full-runtime claim applies to macOS.

Install tools:

```bash
xcode-select --install
brew install cmake git ninja
```

Build:

```bash
git clone --branch v1.0.0-h121 \
  https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git
cd Qwen3.8-Flash-CIRU-STRIX-IU4

cmake -S . -B build-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_HIP=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build-metal --target llama-server llama-cli llama-bench \
  -j "$(sysctl -n hw.logicalcpu)"
```

Expected binaries are under `build-metal/bin/`.

This recipe documents source portability; the complete 126.6 GiB target/MTP/PLE package has not been validated on Apple hardware. The optimized release remains Linux + AMD ROCm + gfx1151.

# Rebuild the Flash v6.1 gfx1151 runtime

This recipe reconstructs the qualified HaloBox b1064-derived runtime from the pinned upstream Git tree and a patch. The patch carries the C1 port, the qualified prefill and serving changes, and the sources of both required Gated DeltaNet sidecars. HaloBox and ggml’s source license and attribution are in the resulting tree. See the [b1064 release](https://github.com/halo-box/strix-llama.cpp/releases/tag/b1064).

The tested build environment uses Linux x86-64, the ROCm 10.1.0a20260910 gfx1151 compiler, CMake 3.31.10, Ninja, and a Linux C and C++ build environment with Git, Python, and standard build tools. Run the following commands from this directory on a compatible Linux builder. The pinned SDK archive is about 2 GB. Set `BUILD_JOBS` to match available memory.

```bash
work=$(mktemp -d)
curl -fL --retry 2 \
  https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.1.0a20260910.tar.gz \
  -o "$work/rocm.tar.gz"
curl -fL --retry 2 \
  https://codeload.github.com/google/boringssl/tar.gz/refs/tags/0.20261105.0 \
  -o "$work/boringssl.tar.gz"
(
  cd "$work"
  sha256sum -c <<'SUMS'
cd63c4be1a7fef912340d29b745e593f357542aaedc99a53e90473df56c15b9f  rocm.tar.gz
2df2b287f3406a3d2f7f51dccb82cffbddf6cf981811e8edcfffe8fbf02c3ac4  boringssl.tar.gz
SUMS
)
mkdir "$work/rocm" "$work/boringssl"
tar -xzf "$work/rocm.tar.gz" -C "$work/rocm" --strip-components=1
tar -xzf "$work/boringssl.tar.gz" -C "$work/boringssl" --strip-components=1

bash prepare.sh "$work/source"
ROCM_PATH="$work/rocm" bash build-gfx1151.sh "$work/source" "$work/build" "$work/boringssl"
```

`prepare.sh` checks the pinned upstream commit, applies the patch and verifies the patched tree against `SOURCE-SHA256SUMS`. See `$work/build/bin` for `llama-server`, `llama-bench`, and their libraries. The build writes `FLASH-V6-BUILD-SHA256SUMS` in the build directory. Keep both `libciru-gdn-*` sidecars beside `libggml-hip.so.1.0.0` so the HIP backend can load the qualified recurrent kernels.

The research release pairs the runtime with a hybrid C1 target, an original donor PLE shard and a matching MTP draft. Use the release’s model and launcher instructions for those files and the measured serving configuration.

The source rebuilt successfully with the pinned Linux toolchain. The server executable, HIP backend and context library matched the measured binaries byte for byte. The common and server implementation libraries matched their executable code, read-only data and dynamic symbol entries, with different ELF file hashes. Both recurrent sidecars reproduced byte for byte when built at the original input paths. The downloadable binary package uses the measured library files.

# Independent Ubuntu smoke review

PASS for the stated build/inference smoke scope. All 24 independent checks passed using saved local evidence; no new inference, remote actions, or documentation edits were performed.

The raw SSE contains exactly 520 target token IDs, identical to retained baseline A1 and the expected fixture. Their SHA-256 is `5115118aef25826d9d904cd30f47a28fc9aeac842457b17ff906deba653fb731`. The original 57-token prompt and 520-token request budget match, `/slots` reports one speculative slot at 262,144 allocation, and native MTP counters match at 674 drafted / 400 accepted. The database has exactly one matching request row and its hash chain verifies. All 15 binary hashes match the clean build. The server container exited normally without OOM or runtime abort diagnostics.

The original `result.json` failure is preserved correctly. It came from requiring the literal `gfx1151` in a verbosity-3 server log, after token and MTP checks had passed. Saved evidence supplies the missing identity: the binary enumerates ROCm0 as AMD Radeon 8060S, the sole GPU KFD node reports `gfx_target_version=110501`, and the actual inference log records an RDNA3.5 kernel admission. This supports the reviewed pass without repeating inference. The separate device-enumeration Docker bridge failure was an infrastructure failure before enumeration; its preserved host-network retry did not load the model.

There is a harmless image-ID representation difference worth preserving in provenance. Dunamis reports `b350ab...` with an OCI manifest descriptor; Ciru reports `3fb497...` through its classic image store. Their complete `Config` objects and `RootFS` layer lists are identical. The loaded container references the reviewed Ciru image. Requiring equal engine-reported IDs would be another false identity failure.

The reviewed scope remains one short coding completion using Ubuntu 24.04 userspace and ROCm 10 on Ciru's NixOS host GPU driver. It does not establish native Ubuntu driver support, filled 262,144-token context behavior, multi-slot behavior, or a matched Ubuntu speed result. Those claims require their separate evidence. Cleanup receipts show the service, governors, and absence of owned servers/containers consistent with intake.

## Reproducing the tested CPU flags

The public helper alone defaults to `GGML_NATIVE=ON`, so it does not promise the same CPU backend on AMD and Intel hosts. The tested final tree explicitly set native mode off, enabled SSE4.2/AVX/AVX2/F16C/FMA/BMI2, and disabled AVX-VNNI and all AVX512 variants. HIP source, gfx1151 target and ROCm flags were unchanged. The actual build initially started in native mode, was interrupted, then rebuilt every affected CPU object with those explicit flags. No previous-build object cache was imported.

The final `docs/BUILD_LINUX.md` portable subsection now gives the same reconfiguration used by the tested build: run setup to initialize the SDK/build directory, set the explicit CPU cache options, then rerun setup so Ninja rebuilds affected CPU objects and retains unchanged HIP objects. All 12 listed CPU options match the saved final CMake cache. No build-script change is required.

The following uses explicit variables so custom build/SDK directories cannot diverge between CMake and the helper. It works on AMD or Intel hosts; use Ubuntu 24.04, the pinned ROCm 10.0.0 SDK and the recorded compiler versions when claiming the same toolchain qualification.

```bash
build_dir="$PWD/build-gfx1151-sdk"
sdk_venv="$PWD/.venv-rocm"
BUILD_DIR="$build_dir" ROCM_VENV="$sdk_venv" BUILD_JOBS=4 \
  ./scripts/ciru/setup-linux-amd.sh --install-host-deps

cmake -S "$PWD" -B "$build_dir" \
  -DGGML_NATIVE=OFF \
  -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_BMI2=ON \
  -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF

BUILD_DIR="$build_dir" ROCM_VENV="$sdk_venv" BUILD_JOBS=4 \
  ./scripts/ciru/setup-linux-amd.sh
```

The final cache must report `GGML_HIP=ON`, `GGML_NATIVE=OFF`, these CPU options, and `GPU_TARGETS=gfx1151` / `AMDGPU_TARGETS=gfx1151`. Keep the SDK venv for runtime libraries. The saved final compile commands confirm 15 CPU translation units have the portable flags and no `-march=native`, AVX512, or AVX-VNNI flags.

For custom paths, CMake takes the build location through `-B`; exporting `BUILD_DIR` alone does not override a literal `-B build-gfx1151-sdk`. `ROCM_VENV` is consumed by the setup helper, not plain CMake. The explicit-variable example above removes that ambiguity. This is a recipe inspection, not another source build or GPU test.

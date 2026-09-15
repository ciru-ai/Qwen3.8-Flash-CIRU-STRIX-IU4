# Build the v4.2.0 source package

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
LLAMA_BUILD_NUMBER=ciru-v4.2.0-ui-f5daaa3 npm run build
```

Copy the resulting `dist/` contents to the release's `ui/` directory and create a new asset manifest. The UI build used Node v26.7.0 and the exact package-lock.json; see UI-PROVENANCE.json for the npm version, hashes and complete file inventory. This local UI build does not alter inference kernels or the server binary.

Keep the complete matched `bin/` executable/shared-library set. Never combine a new server with old `libllama`, `libggml`, `libmtmd` or server implementation libraries. Keep the SDK directory used at build time in place. For a relocated tested Nix package, use its exact recorded dependency paths and put that package's `bin/` ahead of older project libraries in `LD_LIBRARY_PATH`, followed by the SDK library directories as documented in its binary receipt.

The v4.2 UI preserves the qualified application assets and updates only its release version descriptor. The NixOS binary archive keeps the v4.1 IO32 executable and matching libraries, replacing only libggml-hip with the independently tested selected-key fallback correction. Its embedded upstream build text remains R2; CIRU_RELEASE.json and binary-identity.json identify this complete v4.2 package. Source builds use the v4.2 Git revision.

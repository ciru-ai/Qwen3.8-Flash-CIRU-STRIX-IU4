# Build the v5.5.1 source candidate on Linux

See [INSTALL.md](../INSTALL.md) for the complete Ubuntu/Debian setup.
The v5.5 baseline was built on Arch Linux with GNU 16.2.1 and qualified on
NixOS with TheRock 10.0.0 and the pinned custom HIP/ROCr runtime. This v5.5.1
source candidate has passed configuration preflight; a full candidate build
and serving qualification are still pending. A new distribution build is not
a reproduction of the measured executable or a guarantee of identical performance.

## Manual build with an existing complete ROCm 10.0.0 SDK

Install the host prerequisites listed in INSTALL.md. From the source root:

```bash
python3 -m venv .venv-build
.venv-build/bin/python -m pip install 'cmake==4.1.2' 'CppHeaderParser==2.7.4' 'ply==3.11'
export PATH="$PWD/.venv-build/bin:$PATH"
export ROCM_ROOT=/absolute/path/to/complete/rocm-10.0.0-sdk
export CIRU_RUNTIME_ROOT="$PWD/runtime"
export BUILD_DIR="$PWD/build-gfx1151-sdk"
export BUILD_JOBS=2
bash scripts/ciru/build-custom-runtime.sh
bash scripts/ciru/build-linux-amd.sh --check
bash scripts/ciru/build-linux-amd.sh
```

Use the SDK root returned by `rocm-sdk path --root` for a pip SDK. Keep the
matching SDK, HIP/ROCr and complete executable/shared-library set together.
The build disables automatic UI downloading; API serving needs no UI build.

# Build and run on Windows

Windows has two paths:

- **WSL2 + ROCDXG (recommended):** the only path that reaches the Strix Halo
  GPU (`gfx1151`) with the release HIP runtime. This is the path verified end
  to end on a production Windows 11 host in August 2026; measured numbers are
  in [BENCHMARKS.md](BENCHMARKS.md).
- **Native Windows CPU build:** a source/diagnostic path only. No gfx1151 HIP
  fast path, no `O_DIRECT` PLE pager, and the model cannot hold in the
  ~32 GiB of system-visible RAM a Strix Halo exposes once the GPU carve-out is
  taken. Not useful for running the release profile.

Verified environment for the WSL2 path:

| Component | Version used |
|---|---|
| Windows | 11 Pro 25H2 (build 26200) |
| WSL | 2.7.12 (kernel 6.18.33.2) |
| Distro | Ubuntu 26.04.1 LTS ("resolute"); 24.04 also supported by the same steps |
| ROCm | 10.0.0 (amdgpu-install 31.50, `--no-dkms`) |
| ROCDXG | rocdxg-roct 1.2.2 (librocdxg) |
| GPU pool seen by ROCm | 117,076,066 KB (~111.7 GiB) |

A scripted version of every section below ships with this repository as
[`ci/setup-strix-halo-windows.ps1`](../ci/setup-strix-halo-windows.ps1).
It runs the phases in order, gates on preflight (installed RAM, WSL
version, distro, systemd, `/dev/dxg`, disk space, ROCm pool size), sizes
`CONTEXT_SIZE` from the measured pool, and prints each manual step (BIOS
carve-out, reboot, driver, HF login) before the phase that depends on it:

```powershell
# read-only check of the host first:
.\ci\setup-strix-halo-windows.ps1 -Phase preflight
# then, elevated, from a clean Windows install through running server:
.\ci\setup-strix-halo-windows.ps1 -Phase all -Distro Ubuntu-24.04
```

Individual phases (`wsl`, `rocm`, `build`, `model`, `service`,
`portproxy`) can be re-run independently; long downloads resume.

## 1. Install WSL2 and a distro

From an elevated PowerShell:

```powershell
wsl --install --no-distribution
```

A reboot is required before the first distro can start. Then install Ubuntu:

```powershell
wsl --install -d Ubuntu-24.04
```

If the Store download stalls (observed), install the msstore package instead
and register it with `wsl --import`, or use winget:

```powershell
winget install --id 9PDXGNCFSCZV --source msstore
```

Regardless of the route, keep the model and build **inside the WSL Linux
filesystem** (ext4). Do not place `ple/ple.payload.bin` under `/mnt/c`: the
pager opens it with `O_DIRECT`, which DrvFS does not support. The PLE payload
alone is 52.4 GB, plus ~74 GB GGUF and ~4 GB MTP draft, so plan for at least
160 GiB free.

### .wslconfig

A Strix Halo with a large GPU carve-out shows only ~31.6 GiB of system RAM to
Windows. Size the WSL VM below that ceiling:

```ini
[wsl2]
memory=28GB
processors=32
swap=0
vmIdleTimeout=-1
```

`vmIdleTimeout=-1` disables the VM idle shutoff. A large positive value
does NOT work around the WSL 2.6+ idle regression; it must be `-1`.

Note: even with the VM kept alive, WSL 2.6.1+ (confirmed regression
[microsoft/WSL#13416](https://github.com/microsoft/WSL/issues/13416), still
open) tears down systemd units when the last `wsl.exe` client detaches,
roughly 15 seconds after you close your last shell. Use the self-keeper unit
in [Keepalive](#5-keepalive) so a client session always exists.

## 2. Install ROCm 10 and ROCDXG inside WSL

From inside the distro as root:

```bash
cd /tmp
wget -q https://repo.radeon.com/amdgpu-install/31.50/ubuntu/resolute/amdgpu-install_31.50.315000-1_all.deb
apt-get install -y ./amdgpu-install_31.50.315000-1_all.deb
amdgpu-install --usecase=rocm --no-dkms -y
```

`--no-dkms` is required: there is no amdgpu kernel module under WSL; the GPU
arrives through `/dev/dxg`.

The `--usecase=rocm` selection does **not** install everything CMake needs.
The CIRU build fails with two specific missing-package errors; install the
dev packages up front:

```bash
apt-get install -y amdrocm-runtime-dev10.0 \
                   amdrocm-blas10.0 \
                   amdrocm-blas10.0-gfx1151 \
                   amdrocm-blas-dev10.0 \
                   libssl-dev cmake ninja-build build-essential git python3 python3-pip
```

- Missing `hip-lang-config.cmake` (CMake fails in
  `CMakeDetermineHIPCompiler.cmake`) is fixed by `amdrocm-runtime-dev10.0`.
- Missing `hipblasConfig.cmake` (CMake fails in `ggml-hip/CMakeLists.txt`)
  is fixed by `amdrocm-blas10.0-gfx1151` + `amdrocm-blas-dev10.0`.

Do not install the Ubuntu distro's `libhipblas-dev` (7.1.x); it does not
provide the ROCm 10 cmake configs and conflicts.

Then install the ROCDXG user-space bridge (the runtime half, not the
amd-smi-lib half):

```bash
wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
apt-get install -y ./rocdxg-roct_1.2.2_amd64.deb
```

Verify the GPU is visible:

```bash
test -e /dev/dxg && echo dxg-ok
LD_LIBRARY_PATH=/opt/rocm/lib /opt/rocm/bin/rocminfo | grep -m1 gfx1151
```

Expected: `/dev/dxg` exists and rocminfo lists `Agent 2: gfx1151` with a
~111.7 GiB memory pool.

## 3. Build the runtime

```bash
git clone --branch v1.1 https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git /opt/runtime
cd /opt/runtime
ROCM_ROOT=/opt/rocm ./scripts/ciru/build-linux-amd.sh
```

The script configures HIP on, VMM off, graphs on, MFMA on, and
`GPU_TARGETS=gfx1151` (see the script and [BUILD_LINUX.md](BUILD_LINUX.md)).

Requires ROCm's `amdclang++` or `hipcc` on `ROCM_ROOT/bin`; the ROCm 10
install provides `amdclang++`.

Verify the build:

```bash
LD_LIBRARY_PATH=build-gfx1151/bin build-gfx1151/bin/llama-server --version
LD_LIBRARY_PATH=build-gfx1151/bin build-gfx1151/bin/llama-server --help | grep -E 'ple-sidecar|spec-draft-model'
```

## 4. Download and verify the model

```bash
python3 -m venv /opt/hf-venv
/opt/hf-venv/bin/pip install -U "huggingface_hub[cli]" hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 \
  /opt/hf-venv/bin/hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  --local-dir /models/Qwen3.8-Flash-CIRU-STRIX-IU4
cd /models/Qwen3.8-Flash-CIRU-STRIX-IU4
sha256sum -c checksums.sha256
```

All five files (79.4 GB GGUF, 52.4 GB PLE payload, 4.1 GB MTP draft, manifest,
scale) must report OK before first launch. At ~62 MB/s the download takes
roughly 35 minutes.

## 5. Run as a supervised service

A systemd unit is more reliable than a `[boot] command` under WSL 2.7
(observed: the boot command can re-fire on session starts and race the
server). Create `/etc/systemd/system/qwen-ciru-server.service`:

```ini
[Unit]
Description=Qwen3.8-Flash-CIRU-STRIX-IU4 llama-server (gfx1151)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
Environment=LD_LIBRARY_PATH=/opt/runtime/build-gfx1151/bin
Environment=MODEL_DIR=/models/Qwen3.8-Flash-CIRU-STRIX-IU4
Environment=HOST=127.0.0.1
Environment=CONTEXT_SIZE=131072
WorkingDirectory=/opt/runtime
ExecStart=/opt/runtime/scripts/ciru/run-server.sh

[Install]
WantedBy=multi-user.target
```

Then:

```bash
systemctl daemon-reload
systemctl enable --now qwen-ciru-server.service
journalctl -u qwen-ciru-server.service -f
```

### Context size on a 96 GiB carve-out

The release profile defaults to `CONTEXT_SIZE=262144`. On a Strix Halo with
the default large GPU carve-out (98,128 MB dedicated reported by dxdiag), the
draft model load OOMs during startup:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating ... on device 0:
cudaMalloc failed: out of memory
```

F16 target KV at 262,144 tokens alone is ~37.5 GiB plus 18.8 GiB for the
Q8_0 draft KV. Use `CONTEXT_SIZE=131072` (KV drops to ~28.1 GiB total) on
such machines; the full profile needs the complete 128 GiB pool reported by
rocminfo.

### Keepalive

WSL 2.6.1+ has a confirmed regression ([microsoft/WSL#13416](https://github.com/microsoft/WSL/issues/13416), still open): when the last `wsl.exe` client disconnects, the VM is shut down and systemd units are torn down roughly 15 seconds later, even with `vmIdleTimeout` set to a large positive value. Without a workaround the server dies whenever you close your shell, and is not running when a client reconnects.

Two complementary fixes, both verified on WSL 2.7.12:

1. `vmIdleTimeout=-1` in `.wslconfig` (see above) keeps the VM itself alive. A positive value is not enough: only `-1` disables the idle shutoff.
2. A **self-keeper unit** keeps a systemd unit alive across client disconnects. It works by running `wsl.exe` from inside the distro, so there is always an attached client.

Create `/etc/systemd/system/wsl-session-keeper.service`. Replace `-d Ubuntu-24.04` with your distro name from `wsl -l -v`:

```ini
[Unit]
Description=Keep WSL session alive (workaround for microsoft/WSL#13416)
After=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/mnt/c/Windows/System32/wsl.exe -d Ubuntu-24.04 -u root -- sh -c "sleep infinity"

[Install]
WantedBy=multi-user.target
```

Then:

```bash
systemctl daemon-reload
systemctl enable --now wsl-session-keeper.service
systemctl is-active wsl-session-keeper.service   # expect: active
```

It is the same mechanism as a permanent `wsl.exe` foreground process, but self-contained: no Startup-folder scripts, no scheduled tasks, no admin rights, and it survives the WSL VM restarting (the unit starts at boot and immediately re-attaches a client).

With both fixes in place, the VM stays up and the server unit keeps running after you close every shell.

## 6. Optional: expose the server on the LAN

The server binds loopback by default. WSL2 is NAT'd, so binding `0.0.0.0`
inside the guest is not reachable from the LAN by itself; add a portproxy on
Windows (elevated) pointing at the current WSL IP (`wsl hostname -I`):

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8080 connectaddress=<WSL_IP> connectport=8080
netsh advfirewall firewall add rule name="Qwen CIRU 8080" dir=in action=allow protocol=TCP localport=8080
```

The VM's NAT IP can change across reboots; re-sync the proxy after each
boot. Do not rely on `networkingMode=mirrored` for this: on the verified
host, mirrored mode broke Windows-to-WSL loopback access to the server.

## 7. Smoke test

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-CIRU-STRIX-IU4",
       "messages":[{"role":"user","content":"Say hi in two words."}],
       "max_tokens":200}' | head -c 400
```

The model is a thinking model by default; give it a generous `max_tokens`
budget or the reasoning pass consumes the whole budget and `content` comes
back empty.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CMake Error ... CMakeDetermineHIPCompiler.cmake` | missing `amdrocm-runtime-dev10.0` |
| `Could not find hipblasConfig.cmake` | missing `amdrocm-blas10.0-gfx1151` / `amdrocm-blas-dev10.0` |
| `cudaMalloc failed: out of memory` at startup on a large carve-out | `CONTEXT_SIZE=262144` does not fit ~95.8 GiB carve-out; use 131072 |
| VM `poweroff` / unit death ~15 s after closing the shell | WSL 2.6+ regression (microsoft/WSL#13416); set `vmIdleTimeout=-1` and install the self-keeper unit (Keepalive above) |
| Server exits when a boot command re-fires | use the systemd unit, not `[boot] command` |
| `wsl --install -d Ubuntu-24.04` stalls for many minutes | Store backend hang; use `winget install --id 9PDXGNCFSCZV --source msstore` or any msstore Ubuntu package, then `wsl --import` |
| Build succeeds but server OOMs at load on a large carve-out | see `CONTEXT_SIZE` guidance; 131072 is verified on ~96 GiB carve-outs |

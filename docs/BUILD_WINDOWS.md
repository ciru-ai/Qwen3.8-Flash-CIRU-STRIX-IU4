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
| Distro | Ubuntu 26.04.1 LTS ("resolute") - verified; 26+ recommended, 24.04 probably works |
| ROCm | 10.0.0 (amdgpu-install 31.50, `--no-dkms`) |
| ROCDXG | rocdxg-roct 1.2.2 (librocdxg) |
| Runtime tag | v3.0.0 (cut over from v1.1 via the section 8 upgrade flow) |
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
.\ci\setup-strix-halo-windows.ps1 -Phase all -Distro Ubuntu-26.04
```

Individual phases (`wsl`, `rocm`, `build`, `model`, `service`,
`portproxy`, `upgrade`) can be re-run independently; long downloads resume.

## 1. Install WSL2 and a distro

From an elevated PowerShell:

```powershell
wsl --install --no-distribution
```

A reboot is required before the first distro can start. Then install Ubuntu
26.04. Stay on Ubuntu: 26+ is recommended, 26.04.1 is the verified release,
and 24.04 probably works; other distros are not supported:

```powershell
wsl --install -d Ubuntu-26.04
```

If the Store download stalls (observed), install the msstore package with
winget and register its rootfs directly (the Store package is not registered
by winget alone; `wsl --import` accepts the tarball as-is):

```powershell
winget install --id 9PDXGNCFSCZV --source msstore
$rootfs = (Get-ChildItem 'C:\Program Files\WindowsApps\*Ubuntu*\install.tar.gz' | Select-Object -First 1).FullName
wsl --import Ubuntu-26.04 "$env:LOCALAPPDATA\wsl\Ubuntu-26.04" $rootfs --version 2
```

Regardless of the route, keep the model and build **inside the WSL Linux
filesystem** (ext4). Do not place `ple/ple.payload.bin` under `/mnt/c`: the
pager opens it with `O_DIRECT`, which DrvFS does not support. The PLE payload
alone is 52.4 GB, plus a 79.4 GB GGUF and a 4.1 GB MTP draft (~127 GiB
total), so keep at least 170 GiB free - the installer's preflight gate.

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

Keep the cap a few GiB below the Windows-visible total: while the model
loads (6-8 minutes) the VM climbs toward its cap, and Windows will kill
foreground apps if there is no room left.

Note: even with the VM kept alive, WSL 2.6.1+ (confirmed regression
[microsoft/WSL#13416](https://github.com/microsoft/WSL/issues/13416), still
open) tears down systemd units when the last `wsl.exe` client detaches,
roughly 15 seconds after you close your last shell. Use the self-keeper unit
in [Keepalive](#keepalive) so a client session always exists.

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
git clone --branch v3.0.0 https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git /opt/runtime
cd /opt/runtime
ROCM_ROOT=/opt/rocm ./scripts/ciru/build-linux-amd.sh
```

The script configures HIP on, VMM off, graphs on, MFMA on, and
`GPU_TARGETS=gfx1151` (see the script and [BUILD_LINUX.md](BUILD_LINUX.md)).
The compile pool defaults to `nproc`; cap it with `BUILD_JOBS=n` only when
a loaded server is already competing for the same WSL VM RAM (section 8).

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

The units need systemd in the distro: `/etc/wsl.conf` must contain the
two-line stanza

```ini
[boot]
systemd=true
```

(Ubuntu WSL images ship this; the installer's `wsl`
phase verifies and writes it). Reload WSL (`wsl --shutdown`, then reopen a
session) if you had to add it.

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
Environment=HOST=0.0.0.0
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
such machines; the 262,144 default does not fit a carve-out pool.

### Keepalive

WSL 2.6.1+ has a confirmed regression ([microsoft/WSL#13416](https://github.com/microsoft/WSL/issues/13416), still open): when the last `wsl.exe` client disconnects, WSL tears the session down - without `vmIdleTimeout=-1` the VM powers off, and even with it set, systemd units are stopped roughly 15 seconds after the last client detaches. Without a workaround the server dies whenever you close your shell, and is not running when a client reconnects.

Two complementary fixes:

1. `vmIdleTimeout=-1` in `.wslconfig` (see above) keeps the VM itself alive. A positive value is not enough: only `-1` disables the idle shutoff.
2. A **self-keeper unit** keeps a systemd unit alive across client disconnects. It works by running `wsl.exe` from inside the distro, so there is always an attached client.

Create `/etc/systemd/system/wsl-session-keeper.service`. If your Ubuntu
distro is registered under a different name, replace `-d Ubuntu-26.04` with
the name from `wsl -l -v`:

```ini
[Unit]
Description=Keep WSL session alive (workaround for microsoft/WSL#13416)
After=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/mnt/c/Windows/System32/wsl.exe -d Ubuntu-26.04 -u root -- sh -c "sleep infinity"

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

The server binds loopback by default. Use `Environment=HOST=0.0.0.0` in the
unit (as in the example above) when LAN access is wanted; keep `127.0.0.1`
for loopback-only use. WSL2 is NAT'd, so binding `0.0.0.0` inside the guest
is not reachable from the LAN by itself; add a portproxy on Windows
(elevated) pointing at the current WSL IP (`wsl hostname -I`):

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8080 connectaddress=<WSL_IP> connectport=8080
netsh advfirewall firewall add rule name="Qwen CIRU 8080" dir=in action=allow protocol=TCP localport=8080
```

Do not rely on `networkingMode=mirrored` for this: on this host,
mirrored mode broke Windows-to-WSL loopback access to the server.

### Reboot persistence

After a Windows reboot nothing starts the WSL VM until the first interactive
shell opens, and the VM's NAT IP can change. One scheduled task (elevated,
once) closes both gaps. It must run as the distro owner (distros are
registered per-user), with S4U so no password is stored:

```powershell
$keeper = Join-Path $env:USERPROFILE 'qwen-boot-keeper.ps1'
$a = New-ScheduledTaskAction -Execute powershell.exe `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$keeper`""
$t = New-ScheduledTaskTrigger -AtStartup
$p = New-ScheduledTaskPrincipal -UserId <user> -LogonType S4U -RunLevel Highest
Register-ScheduledTask -TaskName 'Qwen CIRU boot' -Action $a -Trigger $t -Principal $p
```

`%USERPROFILE%` is not expanded inside PowerShell - build the path as above.

`qwen-boot-keeper.ps1` boots the distro (systemd then starts the units) and
re-points the portproxy at the fresh NAT IP. `-Phase portproxy` of the
installer script writes this keeper and registers/starts the task for you.
After each reboot: the task boots the VM, systemd starts both enabled units,
the keeper holds the session, and the server answers again once the model has
loaded (6-8 minutes measured). Interactive auto-login is not required by this
chain; use it only if other Startup-folder items must run unattended.

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

## 8. Upgrading to a new runtime tag

A new release tag builds while the current server keeps serving, so the
upgrade is one short restart instead of a re-setup. This is the flow run on
this host for `v1.1` -> `v3.0.0`; the scripted version is `-Phase upgrade`
(add `-WhatIfUpgrade` to stop after staging the new unit, before cutover).

Clone the new tag into its own tree and build with the compile pool sized to
the RAM the running server leaves free - HIP compile jobs peak around
2.2 GiB each, and this host completed 492 objects in about 5 minutes with
`BUILD_JOBS=8` while the old server served requests the whole time:

```bash
git clone --branch v3.0.0 \
  https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git /opt/runtime-v3
cd /opt/runtime-v3
ROCM_ROOT=/opt/rocm BUILD_JOBS=8 ./scripts/ciru/build-linux-amd.sh
```

Before paying for another ~136 GiB download, check whether the tag actually
changed the weights. v3.0.0's `checksums.sha256` is byte-identical to
v1.1's (the release notes say the same), so the installed model files were
reused as-is:

```bash
/opt/hf-venv/bin/hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  checksums.sha256 --revision v3.0.0 --local-dir /tmp/v3-sha
cmp -s /tmp/v3-sha/checksums.sha256 \
  /models/Qwen3.8-Flash-CIRU-STRIX-IU4/checksums.sha256 && echo weights unchanged
```

Install a second unit for the new tree: the section 5 unit with every
`/opt/runtime` path replaced by `/opt/runtime-v3` (name it
`qwen-ciru-server-v3.service`, keep the same `CONTEXT_SIZE`), then cut over.
Only one server can hold port 8080 and the GPU pool at a time:

```bash
systemctl daemon-reload
systemctl stop qwen-ciru-server.service
systemctl disable qwen-ciru-server.service
systemctl enable --now qwen-ciru-server-v3.service
```

`/health` reaches green in about 6 minutes after the cutover (357 s
measured). Leave the old unit stopped but its tree on disk: rollback is the
same command in reverse. The boot keeper and the keeper unit are
name-agnostic - they start the distro, and systemd starts whichever server
unit is enabled - so reboot persistence needs no changes after an upgrade.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CMake Error ... CMakeDetermineHIPCompiler.cmake` | missing `amdrocm-runtime-dev10.0` |
| `Could not find hipblasConfig.cmake` | missing `amdrocm-blas10.0-gfx1151` / `amdrocm-blas-dev10.0` |
| `cudaMalloc failed: out of memory` at startup on a large carve-out | `CONTEXT_SIZE=262144` does not fit ~95.8 GiB carve-out; use 131072 |
| VM `poweroff` / unit death ~15 s after closing the shell | WSL 2.6+ regression (microsoft/WSL#13416); set `vmIdleTimeout=-1` and install the self-keeper unit (Keepalive above) |
| Server exits when a boot command re-fires | use the systemd unit, not `[boot] command` |
| `wsl --install -d Ubuntu-26.04` stalls for many minutes | Store backend hang; use `winget install --id 9PDXGNCFSCZV --source msstore` or any msstore Ubuntu package, then `wsl --import` |
| Build succeeds but server OOMs at load on a large carve-out | see `CONTEXT_SIZE` guidance; use 131072 on ~96 GiB carve-outs |

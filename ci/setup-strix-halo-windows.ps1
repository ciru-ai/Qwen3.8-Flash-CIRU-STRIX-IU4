#Requires -Version 5.1
<#
setup-strix-halo-windows.ps1 - reproducible install of the CIRU
Qwen3.8-Flash Strix Halo runtime on Windows through WSL2 + ROCDXG.

Phases mirror docs/BUILD_WINDOWS.md 1:1. Preflight is read-only and
gates every phase: it verifies the hardware, the BIOS GPU carve-out,
disk space, WSL version, and the keepalive config, and prints each
manual step that must happen before the next phase is allowed.

  -Phase preflight   host checks only (default)
  -Phase wsl         section 1: WSL engine, distro, .wslconfig   [admin]
  -Phase rocm        section 2: ROCm 10, dev packages, ROCDXG
  -Phase build       section 3: build for gfx1151
  -Phase model       section 4: model download + sha256 verify
  -Phase service     section 5: server unit + self-keeper unit
  -Phase portproxy   section 6: LAN exposure                     [admin]
  -Phase all         preflight through service, stopping at gates

Verified on: Windows 11 25H2, WSL 2.7.12, Ubuntu, ROCm 10.0,
rocdxg-roct 1.2.2, Ryzen AI Max+ 395 (gfx1151), 128 GB unified.
#>
[CmdletBinding()]
param(
    [ValidateSet('preflight', 'wsl', 'rocm', 'build', 'model', 'service', 'portproxy', 'all')]
    [string]$Phase = 'preflight',
    [string]$Distro = 'Ubuntu-26.04',
    [string]$RepoTag = 'v1.1',
    [string]$RepoRoot = '/opt/runtime',
    [string]$ModelDir = '/models/Qwen3.8-Flash-CIRU-STRIX-IU4',
    # 0 = derive from the ROCm pool (see Get-AutoContext)
    [int]$ContextSize = 0,
    # 0 = derive as (total RAM GB - 100), floored at 16
    [int]$WslMemoryGB = 0,
    # 0 = every logical processor
    [int]$WslProcessors = 0
)

$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git'
$ModelRepo = 'jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4'
$MinDiskGiB = 170

function Write-Step([string]$m)   { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok([string]$m)     { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Write-Bad([string]$m)    { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Write-Manual([string]$m) { Write-Host "  [MANUAL] $m" -ForegroundColor Yellow }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Require-Admin([string]$what) {
    if (-not (Test-Admin)) {
        Write-Bad "$what needs an elevated PowerShell (Run as Administrator)."
        exit 1
    }
}

function Invoke-WslText([string]$cmdline) {
    $raw = & wsl.exe -d $Distro -u root -- bash -lc $cmdline 2>$null
    return (($raw | Out-String) -replace "`0", '').Trim()
}

function Invoke-Wsl([string]$cmdline) {
    Write-Host "  \$ $cmdline" -ForegroundColor DarkGray
    $raw = & wsl.exe -d $Distro -u root -- bash -lc $cmdline 2>&1
    foreach ($line in ($raw -split "`n")) {
        $t = ($line -replace "`0", '').TrimEnd()
        if ($t) { Write-Host "    $t" }
    }
    return ($LASTEXITCODE -eq 0)
}

function Get-DistroNames {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { return @() }
    $raw = (& wsl.exe -l -q 2>$null | Out-String) -replace "`0", ''
    return @($raw -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Get-WslEngineVersion {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { return $null }
    $raw = (& wsl.exe --version 2>$null | Out-String) -replace "`0", ''
    if ($raw -match 'WSL version:\s*([\d.]+)') { return [version]$Matches[1] }
    return $null
}

# ROCm pool size in GiB measured inside the distro; 0 when ROCm is absent.
function Get-GpuPoolGiB {
    # No double quotes in bash command strings: PowerShell 5.1 corrupts
    # them when passing arguments to native exes. Use single quotes.
    # The largest pool Size not exceeding installed physical RAM is the
    # GPU carve-out pool (e.g. 117076066 KB); larger values are virtual
    # GTT aperture windows (4 TiB), not real memory.
    $ramKb = [double](Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum / 1KB
    $t = Invoke-WslText 'LD_LIBRARY_PATH=/opt/rocm/lib /opt/rocm/bin/rocminfo 2>/dev/null | grep -oE ''Size:[[:space:]]*[0-9]+'' | grep -oE ''[0-9]+'''
    $best = 0.0
    foreach ($line in ($t -split "`n")) {
        $v = $line.Trim()
        if ($v -match '^\d+$') {
            $kb = [double]$v
            if ($kb -le $ramKb -and $kb -gt $best) { $best = $kb }
        }
    }
    return $best / 1MB   # KB to GiB
}

# CONTEXT_SIZE from the measured pool. 131072 is the value verified on a
# ~111.7 GiB pool; the 262144 release profile needs the full 128 GiB pool.
function Get-AutoContext([double]$poolGiB) {
    if ($poolGiB -ge 110) { return 131072 }
    if ($poolGiB -ge 80)  { return 65536 }
    if ($poolGiB -ge 56)  { return 32768 }
    return 16384
}

function Get-ContextForUnits {
    if ($ContextSize -gt 0) { return $ContextSize }
    $pool = Get-GpuPoolGiB
    if ($pool -le 0) {
        Write-Bad 'Cannot measure the ROCm pool. Run -Phase rocm first, or pass -ContextSize explicitly.'
        exit 1
    }
    $auto = Get-AutoContext $pool
    Write-Ok ("ROCm pool {0:N1} GiB -> CONTEXT_SIZE={1}" -f $pool, $auto)
    return $auto
}

function Write-ManualChecklist {
    Write-Host ''
    Write-Host 'Manual prerequisites (verify before or between phases):' -ForegroundColor Yellow
    Write-Manual 'BIOS: raise the GPU memory carve-out (UMA framebuffer) toward 96 GB on 128 GB machines. The verified profile is 96 GiB GPU / 32 GiB OS. A smaller carve-out works but shrinks the ROCm pool, and CONTEXT_SIZE is sized from it.'
    Write-Manual 'Reboot after the WSL engine install, before starting any distro.'
    Write-Manual ("Keep at least {0} GiB free on the drive holding the WSL vhdx (model ~136 GB plus build)." -f $MinDiskGiB)
    Write-Manual 'Install the current AMD Radeon Software driver on Windows; the GPU reaches WSL through /dev/dxg.'
    Write-Manual 'If the model repository requires a Hugging Face login, run: wsl -d <distro> -- bash -lc "/opt/hf-venv/bin/hf auth login"'
    Write-Host ''
}

function Get-DimmRamGiB {
    # Total installed memory including the GPU carve-out (ComputerSystem
    # reports only what Windows keeps after the carve-out).
    $sum = (Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum
    return [double]$sum / 1GB
}

function Get-ModelFilesPresent {
    $t = Invoke-WslText "test -f '$ModelDir/Qwen3.8-Flash-CIRU-STRIX-IU4.gguf' && test -f '$ModelDir/mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf' && test -f '$ModelDir/ple/ple.payload.bin' && echo yes || echo no"
    return ($t -eq 'yes')
}

function Invoke-Preflight {
    $fail = $false
    Write-Step 'Preflight (read-only)'

    $ram = Get-DimmRamGiB
    if ($ram -ge 120) { Write-Ok ("installed RAM {0:N1} GiB" -f $ram) }
    else { Write-Bad ("installed RAM {0:N1} GiB: this release profile is built for 128 GB unified memory machines." -f $ram); $fail = $true }

    $ver = Get-WslEngineVersion
    if ($null -eq $ver) {
        Write-Manual "WSL engine not installed: run -Phase wsl elevated to install it."
    } elseif ($ver -lt [version]'2.6.1') {
        Write-Ok "WSL $ver"
        Write-Manual 'WSL < 2.6.1 predates the microsoft/WSL#13416 regression; vmIdleTimeout=-1 and the keeper unit are harmless there, keep them for future updates.'
    } else {
        Write-Ok "WSL $ver"
        Write-Manual "WSL $ver has microsoft/WSL#13416: the -Phase wsl .wslconfig (vmIdleTimeout=-1) and the -Phase service keeper unit are mandatory for a resident server."
    }

    $names = Get-DistroNames
    if ($names -notcontains $Distro) {
        Write-Manual "Distro '$Distro' not registered: -Phase wsl installs it."
    } else {
        Write-Ok "Distro '$Distro' registered"
        $systemd = Invoke-WslText 'systemctl is-system-running 2>/dev/null || echo unknown'
        if ($systemd -match 'running|degraded|starting') { Write-Ok "systemd active ($systemd)" }
        else { Write-Bad "systemd not enabled in '$Distro': add [boot] systemd=true to /etc/wsl.conf, wsl --shutdown, retry"; $fail = $true }

        $dxg = Invoke-WslText 'test -e /dev/dxg && echo yes || echo no'
        if ($dxg -eq 'yes') { Write-Ok '/dev/dxg present (GPU passthrough)' }
        else { Write-Bad "/dev/dxg missing: update the AMD Windows driver, then wsl --shutdown and retry"; $fail = $true }
    }

    $cfg = Join-Path $env:USERPROFILE '.wslconfig'
    if (Test-Path $cfg) {
        $text = Get-Content $cfg -Raw
        if ($text -match '(?im)^\s*vmIdleTimeout\s*=\s*-1\s*$') { Write-Ok '.wslconfig has vmIdleTimeout=-1' }
        else { Write-Manual ".wslconfig lacks vmIdleTimeout=-1 (only -1 disables the idle VM shutoff; large positive values do not work): -Phase wsl writes it." }
    } else {
        Write-Manual '.wslconfig missing: -Phase wsl writes it (memory, processors, swap, vmIdleTimeout=-1).'
    }

    $pool = Get-GpuPoolGiB
    if ($pool -gt 0) {
        Write-Ok ("ROCm pool {0:N1} GiB -> suggested CONTEXT_SIZE={1}" -f $pool, (Get-AutoContext $pool))
        if ($pool -lt 56) {
            Write-Bad ("ROCm pool {0:N1} GiB is too small for any useful context: raise the BIOS GPU carve-out first." -f $pool); $fail = $true
        }
    } elseif ($names -contains $Distro) {
        Write-Manual 'rocminfo not available yet: run -Phase rocm, then rerun preflight to gate on the real pool size.'
    }

    $drive = (Get-Item $env:LOCALAPPDATA).PSDrive.Name
    $free = (Get-PSDrive $drive).Free / 1GB
    if ($free -ge $MinDiskGiB) { Write-Ok ("{0}: free {1:N0} GiB" -f $drive, $free) }
    else { Write-Bad ("{0}: free {1:N0} GiB, need {2}" -f $drive, $free, $MinDiskGiB); $fail = $true }

    Write-ManualChecklist
    if ($fail) { Write-Bad 'Preflight found blocking issues. Resolve them, then rerun.'; exit 1 }
    Write-Ok 'Preflight passed.'
}

function Invoke-WslPhase {
    Require-Admin 'The wsl phase'
    Write-Step 'Section 1: WSL engine, distro, .wslconfig'

    if ($null -eq (Get-WslEngineVersion)) {
        Write-Host 'Installing WSL engine (no distro)...'
        & wsl.exe --install --no-distribution
        if ($LASTEXITCODE -ne 0) { Write-Bad 'wsl --install failed'; exit 1 }
        Write-Manual 'Reboot now, then rerun: setup-strix-halo-windows.ps1 -Phase all'
        exit 0
    }
    Write-Ok "WSL engine $((Get-WslEngineVersion))"

    if ((Get-DistroNames) -notcontains $Distro) {
        Write-Host "Installing distro $Distro..."
        & wsl.exe --install -d $Distro --no-launch
        if ($LASTEXITCODE -ne 0 -or (Get-DistroNames) -notcontains $Distro) {
            Write-Host 'Store route failed or stalled: falling back to the msstore package + wsl --import.'
            & winget.exe install --id 9PDXGNCFSCZV --source msstore --accept-package-agreements --accept-source-agreements
            $tar = Get-ChildItem 'C:\Program Files\WindowsApps\*Ubuntu*\install.tar.gz' -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $tar) { Write-Bad 'install.tar.gz not found under WindowsApps'; exit 1 }
            & wsl.exe --import $Distro "$env:LOCALAPPDATA\wsl\$Distro" $tar.FullName --version 2
            if ($LASTEXITCODE -ne 0) { Write-Bad 'wsl --import failed'; exit 1 }
        }
        Write-Ok "Distro $Distro registered"
    }

    # systemd is required for the service and keeper units (default on Ubuntu 24.04+)
    Invoke-Wsl 'grep -q ^systemd=true /etc/wsl.conf 2>/dev/null || { printf ''[boot]\nsystemd=true\n'' >> /etc/wsl.conf; echo added; }' | Out-Null
    $t = Invoke-WslText 'grep -q ^systemd=true /etc/wsl.conf && echo yes || echo no'
    if ($t -eq 'yes') { Write-Ok 'systemd enabled in wsl.conf' }
    else { Write-Bad 'could not enable systemd in wsl.conf'; exit 1 }

    $ram = Get-DimmRamGiB
    $mem = if ($WslMemoryGB -gt 0) { $WslMemoryGB } else { [math]::Max(16, [math]::Floor($ram - 100)) }
    $cpu = if ($WslProcessors -gt 0) { $WslProcessors } else { [Environment]::ProcessorCount }
    $cfg = Join-Path $env:USERPROFILE '.wslconfig'
    $ini = @"
[wsl2]
memory=${mem}GB
processors=${cpu}
swap=0
vmIdleTimeout=-1
"@
    Set-Content -Path $cfg -Value $ini -Encoding ASCII
    Write-Ok ".wslconfig: memory=${mem}GB processors=${cpu} swap=0 vmIdleTimeout=-1"

    & wsl.exe --shutdown | Out-Null
    Start-Sleep -Seconds 3
    $dxg = Invoke-WslText 'test -e /dev/dxg && echo yes || echo no'
    if ($dxg -ne 'yes') { Write-Bad '/dev/dxg missing after boot: install the current AMD Radeon Software driver on Windows, wsl --shutdown, rerun'; exit 1 }
    Write-Ok '/dev/dxg present'
}

function Invoke-RocmPhase {
    Write-Step 'Section 2: ROCm 10 + dev packages + ROCDXG'
    $dxg = Invoke-WslText 'test -e /dev/dxg && echo yes || echo no'
    if ($dxg -ne 'yes') { Write-Bad '/dev/dxg missing: complete -Phase wsl and the AMD Windows driver first'; exit 1 }

    if (-not (Invoke-Wsl 'apt-get update -y')) { Write-Bad 'apt update failed'; exit 1 }
    if (-not (Invoke-Wsl 'test -f /tmp/amdgpu-install_31.50.315000-1_all.deb || wget -q -P /tmp https://repo.radeon.com/amdgpu-install/31.50/ubuntu/resolute/amdgpu-install_31.50.315000-1_all.deb')) {
        Write-Bad 'amdgpu-install download failed'; exit 1
    }
    if (-not (Invoke-Wsl 'DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/amdgpu-install_31.50.315000-1_all.deb')) { Write-Bad 'amdgpu-install pkg install failed'; exit 1 }
    if (-not (Invoke-Wsl 'DEBIAN_FRONTEND=noninteractive amdgpu-install --usecase=rocm --no-dkms -y')) { Write-Bad 'ROCm install failed'; exit 1 }
    # CMake needs both dev packages; do not use the distro libhipblas-dev (7.x)
    if (-not (Invoke-Wsl 'DEBIAN_FRONTEND=noninteractive apt-get install -y amdrocm-runtime-dev10.0 amdrocm-blas10.0 amdrocm-blas10.0-gfx1151 amdrocm-blas-dev10.0')) { Write-Bad 'ROCm dev packages failed'; exit 1 }
    if (-not (Invoke-Wsl 'test -f /tmp/rocdxg-roct_1.2.2_amd64.deb || wget -q -P /tmp https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb')) {
        Write-Bad 'rocdxg download failed'; exit 1
    }
    if (-not (Invoke-Wsl 'DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/rocdxg-roct_1.2.2_amd64.deb')) { Write-Bad 'rocdxg install failed'; exit 1 }

    $agent = Invoke-WslText 'LD_LIBRARY_PATH=/opt/rocm/lib /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 gfx1151'
    if ($agent -notmatch 'gfx1151') { Write-Bad 'gfx1151 not visible to rocminfo'; exit 1 }
    $pool = Get-GpuPoolGiB
    Write-Ok ("gfx1151 visible; ROCm pool {0:N1} GiB (CONTEXT_SIZE will be sized from this)" -f $pool)
}

function Invoke-BuildPhase {
    Write-Step 'Section 3: build the runtime'
    if ((Invoke-WslText 'test -x /opt/rocm/bin/rocminfo && echo yes || echo no') -ne 'yes') {
        Write-Bad 'ROCm missing: run -Phase rocm first'; exit 1
    }
    if (-not (Invoke-Wsl 'DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake ninja-build git python3 python3-venv libssl-dev')) { Write-Bad 'base packages failed'; exit 1 }

    if ((Invoke-WslText "test -d $RepoRoot/.git && echo yes || echo no") -eq 'yes') {
        Write-Ok "$RepoRoot already cloned"
    } elseif (-not (Invoke-Wsl "git clone --branch $RepoTag $RepoUrl $RepoRoot")) {
        Write-Bad 'clone failed'; exit 1
    }

    Write-Host 'Building (this takes a while)...'
    if (-not (Invoke-Wsl "cd $RepoRoot && ROCM_ROOT=/opt/rocm ./scripts/ciru/build-linux-amd.sh")) { Write-Bad 'build failed; see the output above'; exit 1 }

    $v = Invoke-WslText "LD_LIBRARY_PATH=$RepoRoot/build-gfx1151/bin $RepoRoot/build-gfx1151/bin/llama-server --version 2>/dev/null | head -1"
    if (-not $v) { Write-Bad 'llama-server --version failed'; exit 1 }
    Write-Ok "llama-server built: $v"
}

function Invoke-ModelPhase {
    Write-Step 'Section 4: model download + verify'
    $free = (Get-PSDrive (Get-Item $env:LOCALAPPDATA).PSDrive.Name).Free / 1GB
    if ($free -lt $MinDiskGiB) { Write-Bad ("free {0:N0} GiB, need {1}: free disk before downloading ~136 GiB" -f $free, $MinDiskGiB); exit 1 }

    if (-not (Invoke-Wsl 'test -d /opt/hf-venv || python3 -m venv /opt/hf-venv')) { Write-Bad 'venv failed'; exit 1 }
    if (-not (Invoke-Wsl '/opt/hf-venv/bin/pip install -q -U ''huggingface_hub[cli]'' hf_transfer')) { Write-Bad 'pip install failed'; exit 1 }
    Write-Host 'Downloading the model (~136 GiB, resumable; rerun this phase if interrupted)...'
    if (-not (Invoke-Wsl "HF_HUB_ENABLE_HF_TRANSFER=1 /opt/hf-venv/bin/hf download $ModelRepo --local-dir $ModelDir")) { Write-Bad 'model download failed; rerun -Phase model to resume'; exit 1 }
    if (-not (Invoke-Wsl "cd $ModelDir && sha256sum -c checksums.sha256")) { Write-Bad 'checksum verification failed; rerun the download'; exit 1 }
    Write-Ok 'model files verified'
}

function Write-Unit([string]$name, [string]$content) {
    $tmp = Join-Path $env:TEMP $name
    Set-Content -Path $tmp -Value $content -Encoding ASCII
    # Convert C:\...\Temp\name to /mnt/c/.../Temp/name without wslpath
    $drv = $tmp.Substring(0, 1).ToLower()
    $u = '/mnt/' + $drv + ($tmp.Substring(2) -replace '\\', '/')
    if (-not (Invoke-Wsl "cp $u /etc/systemd/system/$name")) { Write-Bad "installing $name failed"; exit 1 }
}

function Invoke-ServicePhase {
    Write-Step 'Section 5: server unit + self-keeper unit'
    if ((Invoke-WslText "test -x $RepoRoot/build-gfx1151/bin/llama-server && echo yes || echo no") -ne 'yes') {
        Write-Bad 'llama-server not built: run -Phase build first'; exit 1
    }
    if (-not (Get-ModelFilesPresent)) { Write-Bad 'model files missing: run -Phase model first'; exit 1 }
    $ctx = Get-ContextForUnits

    Write-Unit 'qwen-ciru-server.service' @"
[Unit]
Description=Qwen3.8-Flash-CIRU-STRIX-IU4 llama-server (gfx1151)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
Environment=LD_LIBRARY_PATH=$RepoRoot/build-gfx1151/bin
Environment=MODEL_DIR=$ModelDir
Environment=HOST=0.0.0.0
Environment=CONTEXT_SIZE=$ctx
WorkingDirectory=$RepoRoot
ExecStart=$RepoRoot/scripts/ciru/run-server.sh

[Install]
WantedBy=multi-user.target
"@

    Write-Unit 'wsl-session-keeper.service' @"
[Unit]
Description=Keep WSL session alive (workaround for microsoft/WSL#13416)
After=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/mnt/c/Windows/System32/wsl.exe -d $Distro -u root -- sh -c "sleep infinity"

[Install]
WantedBy=multi-user.target
"@

    if (-not (Invoke-Wsl 'systemctl daemon-reload && systemctl enable --now qwen-ciru-server.service wsl-session-keeper.service')) { Write-Bad 'unit start failed'; exit 1 }
    $a1 = Invoke-WslText 'systemctl is-active qwen-ciru-server.service'
    $a2 = Invoke-WslText 'systemctl is-active wsl-session-keeper.service'
    if ($a1 -eq 'active' -and $a2 -eq 'active') {
        Write-Ok "both units active (server CONTEXT_SIZE=$ctx)"
        Write-Host "First load takes ~10-12 minutes. Follow it with: wsl -d $Distro -u root -- journalctl -u qwen-ciru-server.service -f"
        Write-Host 'Then smoke test: curl http://127.0.0.1:8080/health'
    } else { Write-Bad "unit not active: server=$a1 keeper=$a2 (journalctl -u <unit> to inspect)"; exit 1 }
}

function Invoke-PortproxyPhase {
    Require-Admin 'The portproxy phase'
    Write-Step 'Section 6: expose the server on the LAN (optional)'
    $ip = ((& wsl.exe -d $Distro -- hostname -I 2>$null | Out-String) -replace "`0", '').Trim()
    if (-not $ip) { Write-Bad 'cannot resolve the WSL IP; is the distro running?'; exit 1 }
    $ip = ($ip -split ' ')[0]
    & netsh.exe interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080 2>$null | Out-Null
    & netsh.exe interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8080 connectaddress=$ip connectport=8080
    if ($LASTEXITCODE -ne 0) { Write-Bad 'netsh portproxy failed'; exit 1 }
    $rule = & netsh.exe advfirewall firewall show rule name='Qwen CIRU 8080' 2>$null | Out-String
    if ($rule -notmatch 'Qwen CIRU 8080') {
        & netsh.exe advfirewall firewall add rule name='Qwen CIRU 8080' dir=in action=allow protocol=TCP localport=8080 | Out-Null
    }

    # Boot task: after a Windows reboot nothing starts the WSL VM until a
    # shell opens, and the NAT IP changes. This task boots the distro at
    # startup and re-points the portproxy at the fresh IP. It must run as
    # the current user (distros are registered per-user), not SYSTEM.
    $keeper = Join-Path $env:USERPROFILE 'qwen-boot-keeper.ps1'
    $q = [char]39
    $keeperBody = @"
param([string]`${Distro} = ${q}$Distro${q}, [int]`${Port} = 8080)
`${ErrorActionPreference} = ${q}SilentlyContinue${q}
& wsl.exe -d `${Distro} -u root -- true | Out-Null
`${ip} = ${q}${q}
for (`$i = 0; `$i -lt 40; `$i++) {
    `${ip} = ((& wsl.exe -d `${Distro} -- hostname -I 2>`$null | Out-String) -replace "`0", '').Trim()
    if (`$ip) { break }
    Start-Sleep -Seconds 3
}
if (-not `$ip) { exit 1 }
`$ip = (`$ip -split ' ')[0]
& netsh.exe interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=`$Port | Out-Null
& netsh.exe interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=`$Port connectaddress=`$ip connectport=`$Port
if (`$LASTEXITCODE -ne 0) { exit 1 }
"@
    Set-Content -Path $keeper -Value $keeperBody -Encoding ASCII
    $a = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$keeper`""
    $t = New-ScheduledTaskTrigger -AtStartup
    $t.Delay = 'PT10S'
    $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
    $s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName 'Qwen CIRU boot' -Action $a -Trigger $t -Principal $p -Settings $s -Force -Description 'Boot the WSL VM and re-sync the 8080 portproxy at startup' | Out-Null
    Start-ScheduledTask -TaskName 'Qwen CIRU boot'
    Write-Ok "portproxy 0.0.0.0:8080 -> ${ip}:8080; boot task 'Qwen CIRU boot' registered (re-syncs after every reboot)"
}

switch ($Phase) {
    'preflight' { Invoke-Preflight }
    'wsl'       { Invoke-WslPhase }
    'rocm'      { Invoke-RocmPhase }
    'build'     { Invoke-BuildPhase }
    'model'     { Invoke-ModelPhase }
    'service'   { Invoke-ServicePhase }
    'portproxy' { Invoke-PortproxyPhase }
    'all' {
        Invoke-Preflight
        Invoke-WslPhase
        Invoke-RocmPhase
        Invoke-BuildPhase
        Invoke-ModelPhase
        Invoke-ServicePhase
    }
}

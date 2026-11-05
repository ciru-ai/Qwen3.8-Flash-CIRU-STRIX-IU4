#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
install_host=0
case "${1:-}" in
  --install-host-deps) install_host=1; shift ;;
  --help|-h) echo "Usage: $0 [--install-host-deps]"; echo "Install the pinned private SDK, build HIP/ROCr and the engine. Does not start a model."; exit 0 ;;
esac
[[ $# == 0 ]] || { echo "Unexpected argument: $1" >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Linux x86-64 required" >&2; exit 2; }
if (( install_host )); then
  command -v apt-get >/dev/null || { echo "Install distro prerequisites from INSTALL.md" >&2; exit 2; }
  sudo_cmd=(); if (( EUID != 0 )); then sudo_cmd=(sudo); fi
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y build-essential git ninja-build pkg-config \
    libssl-dev libdrm-dev libelf-dev libnuma-dev libgl-dev zlib1g-dev \
    python3-venv ca-certificates curl
fi
for tool in python3 git ninja cc c++ pkg-config; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; see INSTALL.md" >&2; exit 2; }
done
venv="${ROCM_VENV:-$repo_root/.venv-rocm}"
if [[ -d "$venv" && ! -f "$venv/pyvenv.cfg" ]]; then
  echo "$venv exists and is not a Python virtual environment" >&2; exit 2
fi
[[ -f "$venv/pyvenv.cfg" ]] || python3 -m venv "$venv"
version="$("$venv/bin/python" -m pip show rocm 2>/dev/null | sed -n 's/^Version: //p' || true)"
[[ -z "$version" || "$version" == 10.0.0 ]] || { echo "Use a fresh ROCM_VENV for ROCm 10.0.0" >&2; exit 2; }
"$venv/bin/python" -m pip install 'cmake==4.1.2' 'CppHeaderParser==2.7.4' 'ply==3.11'
"$venv/bin/python" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'rocm[libraries,devel,device-gfx1151]==10.0.0'
"$venv/bin/rocm-sdk" init
export PATH="$venv/bin:$PATH"
export ROCM_ROOT
ROCM_ROOT="$("$venv/bin/rocm-sdk" path --root)"
export CIRU_RUNTIME_ROOT="${CIRU_RUNTIME_ROOT:-$repo_root/runtime}"
export BUILD_DIR="${BUILD_DIR:-$repo_root/build-gfx1151-sdk}"
export BUILD_JOBS="${BUILD_JOBS:-2}"
bash "$repo_root/scripts/ciru/build-custom-runtime.sh"
bash "$repo_root/scripts/ciru/build-linux-amd.sh" --check
bash "$repo_root/scripts/ciru/build-linux-amd.sh"
printf '\nBuild complete. Keep the source, .venv-rocm, runtime and build directories together.\nSee INSTALL.md for model download and launch instructions.\n'

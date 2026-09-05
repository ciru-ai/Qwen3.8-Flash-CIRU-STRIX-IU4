#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
install_host=0
check_only=0
for arg in "$@"; do
    case "${arg}" in
        --install-host-deps) install_host=1 ;;
        --check) check_only=1 ;;
        --help|-h)
            echo "Usage: $0 [--install-host-deps | --check]"
            echo "Install a private ROCm 10.0.0 SDK and build. Does not start a model, change GPU drivers, or manage services."
            echo "--install-host-deps: install Ubuntu/Debian build tools with apt first."
            echo "--check: check an existing private SDK without installing or building."
            echo "Overrides: ROCM_VENV, ROCM_VERSION, BUILD_DIR, BUILD_JOBS."
            exit 0 ;;
        *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
if (( install_host && check_only )); then
    echo "--check cannot be combined with --install-host-deps" >&2
    exit 2
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo "This setup script requires Linux x86-64." >&2; exit 2;
}
if (( install_host )); then
    command -v apt-get >/dev/null || { echo "Install host tools using your distro package manager; see docs/BUILD_LINUX.md." >&2; exit 2; }
    sudo_cmd=()
    if (( EUID != 0 )); then sudo_cmd=(sudo); fi
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y build-essential cmake git ninja-build libssl-dev python3-venv ca-certificates
fi
for tool in python3 cmake ninja cc c++; do
    command -v "${tool}" >/dev/null || {
        echo "Missing ${tool}. On Ubuntu/Debian rerun with --install-host-deps; otherwise see docs/BUILD_LINUX.md." >&2
        exit 2
    }
done

sdk_venv="${ROCM_VENV:-${repo_root}/.venv-rocm}"
sdk_version="${ROCM_VERSION:-10.0.0}"
if (( ! check_only )); then
    if [[ -d "${sdk_venv}" && ! -f "${sdk_venv}/pyvenv.cfg" ]]; then
        echo "${sdk_venv} exists but is not a virtual environment. Choose a new ROCM_VENV." >&2
        exit 2
    fi
    if [[ ! -f "${sdk_venv}/pyvenv.cfg" ]]; then
        python3 -m venv "${sdk_venv}"
    fi
    installed_version="$("${sdk_venv}/bin/python" -m pip show rocm 2>/dev/null | sed -n 's/^Version: //p' || true)"
    if [[ -n "${installed_version}" && "${installed_version}" != "${sdk_version}" ]]; then
        echo "${sdk_venv} contains ROCm ${installed_version}; choose a new ROCM_VENV and BUILD_DIR for ${sdk_version}." >&2
        exit 2
    fi
    "${sdk_venv}/bin/python" -m pip install \
        --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "rocm[libraries,devel,device-gfx1151]==${sdk_version}"
    "${sdk_venv}/bin/rocm-sdk" init
fi
[[ -x "${sdk_venv}/bin/rocm-sdk" ]] || { echo "No private SDK at ${sdk_venv}; run setup without --check first." >&2; exit 2; }
export ROCM_ROOT
ROCM_ROOT="$("${sdk_venv}/bin/rocm-sdk" path --root)"
export BUILD_DIR="${BUILD_DIR:-${repo_root}/build-gfx1151-sdk}"
export BUILD_JOBS="${BUILD_JOBS:-4}"
args=()
if (( check_only )); then args=(--check); fi
"${repo_root}/scripts/ciru/build-linux-amd.sh" "${args[@]}"
if (( ! check_only )); then
    printf '\nBuild complete. Keep the SDK directory in place for runtime libraries.\n'
    printf 'To start the server separately, set MODEL_DIR and run:\nBUILD_DIR=%q ./scripts/ciru/run-server.sh\n' "${BUILD_DIR}"
fi

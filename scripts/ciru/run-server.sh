#!/usr/bin/env bash
set -euo pipefail

repo_root="${RUNTIME_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
repo_root="$(cd "$repo_root" && pwd)"
package_root="$repo_root"
build_dir="${BUILD_DIR:-${repo_root}/build-gfx1151-sdk}"
if [[ -z "${BUILD_DIR:-}" && -x "${repo_root}/bin/llama-server" ]]; then
    build_dir="${repo_root}"
fi
server_bin="${SERVER_BIN:-${build_dir}/bin/llama-server}"
model_dir="${MODEL_DIR:-${repo_root}/model}"
case "${MODEL_VARIANT:-IU4}" in
    IU4)
        model_name=Qwen3.8-Flash-CIRU-STRIX-IU4
        projector_name=mmproj-Qwen3.8-Flash-F16.mmproj
        slot_name=v4.1.0
        ;;
    Orca)
        model_name=Qwen3.8-Flash-CIRU-STRIX-Orca
        projector_name=mmproj-Qwen3.8-Flash-Orca-F16.mmproj
        slot_name=orca-v4.1.0
        ;;
    *) echo "MODEL_VARIANT must be IU4 or Orca." >&2; exit 2 ;;
esac
model="${model_dir}/${model_name}.gguf"
draft="${model_dir}/mtp/${model_name}-MTP-Q8_0.gguf"
ple_dir="${model_dir}/ple"
slot_dir="${SLOT_DIR:-${package_root}/slot-state/${slot_name}}"

# Launcher-only opt-in; remaining arguments are passed to llama-server.
enable_vision="${ENABLE_VISION:-0}"
server_args=()
for arg in "$@"; do
    case "$arg" in
        --vision) enable_vision=1 ;;
        *) server_args+=("$arg") ;;
    esac
done
set -- "${server_args[@]}"
if [[ "$enable_vision" != 0 && "$enable_vision" != 1 ]]; then
    echo "ENABLE_VISION must be 0 or 1." >&2
    exit 2
fi
mmproj="${MMPROJ:-${model_dir}/vision/${projector_name}}"
if [[ "$enable_vision" == 1 && ! -f "$mmproj" ]]; then
    echo "Vision is enabled but its projector is missing: $mmproj" >&2
    echo "Download vision/${projector_name} from the model repository, or set MMPROJ." >&2
    exit 2
fi

# Image batches are qualified with target-only generation.
enable_mtp="${ENABLE_MTP:-1}"
if [[ "$enable_vision" == 1 ]]; then
    enable_mtp=0
    echo "Vision mode uses target-only generation (MTP disabled)." >&2
fi

# The release qualifies MTP with exactly one slot; preserve the public guard.
# Include trailing CLI overrides so --parallel/-np cannot bypass this check.
parallel_slots="${PARALLEL_SLOTS:-1}"
extra_args=("$@")
for ((i = 0; i < ${#extra_args[@]}; i++)); do
    case "${extra_args[i]}" in
        --parallel|-np)
            parallel_slots="${extra_args[i+1]:-}"
            ((i += 1))
            ;;
        --parallel=*) parallel_slots="${extra_args[i]#*=}" ;;
        -np=*) parallel_slots="${extra_args[i]#*=}" ;;
    esac
done
if [[ "${enable_mtp}" != "0" && "${parallel_slots}" != "1" ]]; then
    echo "The CIRU v4.1.0 MTP profile requires exactly one slot (--parallel 1)." >&2
    echo "For parallel target-only serving, set ENABLE_MTP=0 and PARALLEL_SLOTS=2." >&2
    echo "See docs/RUNNING.md: Parallel requests and unified KV cache." >&2
    exit 2
fi

for required in "${server_bin}" "${model}" "${ple_dir}/ple.payload.bin" "${ple_dir}/ple.manifest.json" "${ple_dir}/ple.scale.bf16"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required release file is missing: ${required}" >&2
        exit 2
    fi
done

server_bin="$(realpath "${server_bin}")"
export LD_LIBRARY_PATH="$(dirname "${server_bin}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${slot_dir}"

set -a
# shellcheck source=/dev/null
source "${repo_root}/profiles/strix-halo-production.env"
set +a

args=(
    --model "${model}"
    --alias "${model_name}"
    --host "${HOST:-127.0.0.1}"
    --port "${PORT:-8080}"
    --jinja
    --ple-sidecar "${ple_dir}"
    --ple-cache-mib "${PLE_CACHE_MIB:-4096}"
    --slot-save-path "${slot_dir}"
    -ngl all
    -sm none
    --fit off
    -lm none
    -lzm on-direct
    --no-kv-unified
    --no-context-shift
    -c "${CONTEXT_SIZE:-262144}"
    -b "${BATCH_SIZE:-8192}"
    -ub "${UBATCH_SIZE:-8192}"
    --parallel "${PARALLEL_SLOTS:-1}"
    -t "${THREADS:-8}"
    -tb "${BATCH_THREADS:-8}"
    -ctk f16
    -ctv f16
    -fa on
    --cont-batching
    --cache-prompt
    --cache-ram "${PROMPT_CACHE_MIB:-1024}"
    --cache-idle-slots
    --ctx-checkpoints "${CTX_CHECKPOINTS:-32}"
    --checkpoint-min-step "${CHECKPOINT_MIN_STEP:-8192}"
    --temp "${TEMPERATURE:-1.0}"
    --top-p "${TOP_P:-0.95}"
    --top-k "${TOP_K:-20}"
    --min-p "${MIN_P:-0.0}"
    --metrics
    --slots
)

# External assets preserve the Web UI without rebuilding the inference binary.
enable_ui="${ENABLE_UI:-1}"
ui_dir="${UI_DIR:-${repo_root}/ui}"
for ((i = 0; i < ${#extra_args[@]}; i++)); do
    case "${extra_args[i]}" in
        --no-ui|--no-webui) enable_ui=0 ;;
        --ui|--webui) enable_ui=1 ;;
        --path)
            ui_dir="${extra_args[i+1]:-}"
            ((i += 1))
            ;;
        --path=*) ui_dir="${extra_args[i]#*=}" ;;
    esac
done
if [[ "$enable_ui" == 1 ]]; then
    if [[ ! -f "${ui_dir}/index.html" ]]; then
        echo "Web UI assets are missing: ${ui_dir}/index.html" >&2
        echo "Install this runtime's ui/ directory, set UI_DIR, or use ENABLE_UI=0 for API-only serving." >&2
        exit 2
    fi
    args+=(--ui --path "$ui_dir")
elif [[ "$enable_ui" == 0 ]]; then
    args+=(--no-ui)
else
    echo "ENABLE_UI must be 0 or 1." >&2
    exit 2
fi

if [[ "$enable_vision" == 1 ]]; then
    args+=(--mmproj "$mmproj")
fi

if [[ "${enable_mtp}" != "0" ]]; then
    if [[ ! -f "${draft}" ]]; then
        echo "MTP is enabled but the draft model is missing: ${draft}" >&2
        exit 2
    fi
    args+=(
        --spec-type draft-mtp
        --spec-draft-model "${draft}"
        --spec-draft-ngl all
        --spec-draft-device "${DRAFT_DEVICE:-ROCm0}"
        --spec-draft-type-k f16
        --spec-draft-type-v f16
        --spec-draft-threads "${DRAFT_THREADS:-8}"
        --spec-draft-threads-batch "${DRAFT_BATCH_THREADS:-8}"
        --spec-draft-n-max "${MTP_DEPTH:-6}"
        --spec-draft-n-min 0
        --spec-draft-p-min 0.0
        --spec-draft-p-split 0.10
    )
else
    args+=(--spec-type none)
fi

unset GGML_HIP_GRAPH_EXEC_UPDATE CIRU_MTP_GPU_CONFIDENCE CIRU_MTP_GPU_ADAPTIVE CIRU_MTP_GPU_CONF_MIN CIRU_MOE_EXPERT_REUSE CIRU_MTP_TRACE CIRU_MTP_CONF_TRACE LD_PRELOAD
# Backend discovery must not scan an unrelated working directory.
cd -- "${repo_root}"
exec "${server_bin}" "${args[@]}" "$@"

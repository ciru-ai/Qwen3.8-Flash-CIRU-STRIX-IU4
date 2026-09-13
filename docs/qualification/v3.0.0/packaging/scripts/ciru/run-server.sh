#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-gfx1151}"
server_bin="${SERVER_BIN:-${build_dir}/bin/llama-server}"
model_dir="${MODEL_DIR:-${repo_root}/model}"
model="${model_dir}/Qwen3.8-Flash-CIRU-STRIX-IU4.gguf"
draft="${model_dir}/mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf"
ple_dir="${model_dir}/ple"
slot_dir="${SLOT_DIR:-${repo_root}/slot-state}"

# The released shortlist has one shared row map and requires exactly one slot.
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
if [[ "${ENABLE_MTP:-1}" != "0" && "${parallel_slots}" != "1" ]]; then
    echo "The CIRU MTP shortlist requires exactly one slot (--parallel 1)." >&2
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

mkdir -p "${slot_dir}"

set -a
# shellcheck source=/dev/null
source "${repo_root}/profiles/strix-halo-production.env"
set +a

args=(
    --model "${model}"
    --alias Qwen3.8-Flash-CIRU-STRIX-IU4
    --host "${HOST:-127.0.0.1}"
    --port "${PORT:-8080}"
    --jinja
    --ple-sidecar "${ple_dir}"
    --ple-cache-mib "${PLE_CACHE_MIB:-4096}"
    --slot-save-path "${slot_dir}"
    -ngl all
    -sm none
    --fit off
    -c "${CONTEXT_SIZE:-262144}"
    -b "${BATCH_SIZE:-1024}"
    -ub "${UBATCH_SIZE:-1024}"
    --parallel "${PARALLEL_SLOTS:-1}"
    -t "${THREADS:-8}"
    -tb "${BATCH_THREADS:-8}"
    -ctk f16
    -ctv f16
    -fa on
    --cont-batching
    --cache-prompt
    --cache-ram "${PROMPT_CACHE_MIB:-8192}"
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

if [[ "${ENABLE_MTP:-1}" != "0" ]]; then
    if [[ ! -f "${draft}" ]]; then
        echo "MTP is enabled but the draft model is missing: ${draft}" >&2
        exit 2
    fi
    args+=(
        --spec-type draft-mtp
        --spec-draft-model "${draft}"
        --spec-draft-ngl all
        --spec-draft-device "${DRAFT_DEVICE:-ROCm0}"
        --spec-draft-type-k q8_0
        --spec-draft-type-v q8_0
        --spec-draft-threads "${DRAFT_THREADS:-8}"
        --spec-draft-threads-batch "${DRAFT_BATCH_THREADS:-8}"
        --spec-draft-n-max "${MTP_DEPTH:-6}"
        --spec-draft-n-min 0
        --spec-draft-p-min 0.0
        --spec-draft-p-split 0.10
    )
fi

unset GGML_HIP_GRAPH_EXEC_UPDATE CIRU_MTP_GPU_CONFIDENCE CIRU_MTP_GPU_ADAPTIVE CIRU_MTP_GPU_CONF_MIN CIRU_MOE_EXPERT_REUSE CIRU_MTP_TRACE CIRU_MTP_CONF_TRACE LD_PRELOAD
# Backend discovery must not scan an unrelated working directory.
cd -- "${repo_root}"
exec "${server_bin}" "${args[@]}" "$@"

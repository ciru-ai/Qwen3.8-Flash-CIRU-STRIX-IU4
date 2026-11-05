# CIRU shared-context launcher policy. Sourced after effective CLI slot/context parsing.
# The runtime must include the qualified unified-QSA and sequence-dense changes.
ciru_memory_policy() {
    local slots="$1" context="$2" policy="${CONTEXT_POOL:-auto}"
    local total_default=32 checkpoints_total
    [[ "$slots" =~ ^[1-9][0-9]*$ ]] || {
        echo 'PARALLEL_SLOTS/--parallel must be a positive integer.' >&2; return 2;
    }
    if ((slots > 1)); then total_default=8; fi
    checkpoints_total="${CTX_CHECKPOINTS_TOTAL:-$total_default}"
    [[ "$checkpoints_total" =~ ^[0-9]+$ ]] || {
        echo 'CTX_CHECKPOINTS_TOTAL must be a nonnegative integer.' >&2; return 2;
    }
    ciru_checkpoints="${CTX_CHECKPOINTS:-$((10#$checkpoints_total / slots))}"
    [[ "$ciru_checkpoints" =~ ^[0-9]+$ ]] || {
        echo 'CTX_CHECKPOINTS must be a nonnegative integer.' >&2; return 2;
    }
    if [[ "$policy" == auto ]]; then
        if ((slots > 1)); then policy=1; else policy=0; fi
    fi
    ciru_kv_args=(--no-kv-unified)
    ciru_idle_args=(--cache-idle-slots)
    case "$policy" in
        0)
            unset CIRU_QSA_UNIFIED_SERIAL CIRU_QSA_SEQUENCE_DENSE
            ;;
        1)
            ciru_kv_args=(--kv-unified --kv-unified-per-slot "$context")
            ciru_idle_args=(--no-cache-idle-slots)
            export CIRU_QSA_UNIFIED_SERIAL=1 CIRU_QSA_SEQUENCE_DENSE=1
            ;;
        *) echo 'CONTEXT_POOL must be auto, 0, or 1.' >&2; return 2 ;;
    esac
    # Cache/PLE budgets stay global. Keep the established single-slot microbatch.
    ciru_ubatch_default="$ubatch_default"
    if ((slots > 1)); then ciru_ubatch_default=2048; fi
}

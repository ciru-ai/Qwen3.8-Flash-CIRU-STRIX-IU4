#pragma once
#include "ggml.h"
#include <cstdint>
struct ciru_hc_record {
    int32_t type, op, flags, external;
    int64_t ne[4];
    uint64_t nb[4];
    int32_t params[GGML_MAX_OP_PARAMS / 4];
    int32_t src[GGML_MAX_SRC];
    int32_t view, bf16, cache, output;
    uint64_t view_offs, bytes, data_address, cache_address;
    char name[GGML_MAX_NAME];
};

#pragma once
#include "common.cuh"
// Ciru replays up to four cached tokens before a logical 512-token append.
// The donor's per-query scheduling is also valid for that measured 516-row
// shape; keep a small margin while larger ubatches remain on QSA3.
#define QSA_DECODE_MAX_QUERIES 520
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

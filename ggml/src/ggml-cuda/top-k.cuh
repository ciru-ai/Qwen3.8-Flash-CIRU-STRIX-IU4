#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_supports_mtp_top_k(const ggml_tensor * dst);

bool ggml_cuda_qsa_prefill_rows_supported(int64_t nrows);

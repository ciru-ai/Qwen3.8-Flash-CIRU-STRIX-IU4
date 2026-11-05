#pragma once

#include "common.cuh"

// Experimental C1 dispatch: native/same-value vector paths for short batches,
// grouped WMMA for contiguous larger batches, and F32 reference fallback.
// Source compilation alone is not a full-model performance result.
void ggml_cuda_mul_mat_id_c1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

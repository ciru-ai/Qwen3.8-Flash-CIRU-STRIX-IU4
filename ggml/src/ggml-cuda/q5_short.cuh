#pragma once
#include "common.cuh"

bool ggml_cuda_q5_short_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
                           const ggml_tensor * x, ggml_tensor * y);

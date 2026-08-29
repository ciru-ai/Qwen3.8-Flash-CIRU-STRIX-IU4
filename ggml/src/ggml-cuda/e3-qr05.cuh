#pragma once

#include <cstdint>

// Launch the complete resident E3 node on the caller's graph stream.
// M=1 keeps the 19,680-byte workspace. M>1 reserves 19,800 bytes per token
// plus 16 KiB; the leading [2,560, M] F32 region is the public output.
void ggml_cuda_e3_qr05_launch(
        const float * input_f32,
        const std::uint8_t * resident_layer_bank,
        unsigned bank_bytes,
        const std::int32_t * logical_expert_ids,
        const float * route_weights,
        float * workspace_f32,
        unsigned n_tokens,
        void * stream);

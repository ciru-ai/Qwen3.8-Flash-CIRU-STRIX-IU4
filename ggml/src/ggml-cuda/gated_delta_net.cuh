#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Chunked prefill (RDNA3.5): the q/k L2 norms that precede GATED_DELTA_NET (RMS_NORM(x, eps/n) -> SCALE(1/sqrt(n)),
// i.e. x * mul * rsqrt(sum(x^2) + eps) per 128-row) are folded into the chunk kernels. ggml_cuda_gdn_chunk_eligible
// is the exact dispatch condition of the chunked path for this GDN node with these q/k tensors; set_qk_norm registers
// the raw q/k views for the next evaluation of that node (consumed by it).
bool ggml_cuda_gdn_chunk_eligible(const ggml_tensor * gdn, const ggml_tensor * q, const ggml_tensor * k);
void ggml_cuda_gdn_set_qk_norm(const ggml_tensor * gdn, const ggml_tensor * q_raw, const ggml_tensor * k_raw, float eps, float mul);
// chunked prefill: the gated RMS norm of the output (RMS_NORM -> MUL(w) -> MUL(sigmoid(z))) written by the scan kernel
// straight into out (+ optional BF16 copy out16); z / out laid out like the GDN output rows [S_v, H, T]
void ggml_cuda_gdn_set_out_norm(const ggml_tensor * gdn, const float * w, const float * z, float * out, uint16_t * out16, float eps);
// chunked prefill: the causal conv + SiLU producing q/k/v is computed by the chunk kernels from the raw conv input
// x [T, C] (row stride C floats), history st [C, 3] and taps cw [C, 4]; qc/kc/vc = channel of element 0 of each view
void ggml_cuda_gdn_set_conv(const ggml_tensor * gdn, const float * x, const float * st, const float * cw,
        int64_t C, int64_t qc, int64_t kc, int64_t vc);
// ggml_cuda_gdn_chunk_eligible with the q/k views registered by set_qk_norm (if any)
bool ggml_cuda_gdn_chunk_eligible_node(const ggml_tensor * gdn);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

#include "c1.cuh"

#include <cstdint>
#include <cstdlib>

#if defined(__HIP_PLATFORM_AMD__)
struct C1PackedQ8K32Split {
    __fp16 d, s0, s1, pad;
    uint32_t low[4], high[4];
};
static_assert(sizeof(C1PackedQ8K32Split) == 40);

// Same activation producer and integer arithmetic as the isolated frozen
// C1 graph. Both IU4 and IU8 arms pay this producer and use its exact bytes.
static __global__ void c1_pack_q8_runtime(const float * source,
        C1PackedQ8K32Split * output, int groups32, int rows) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unsigned(groups32 * rows)) return;
    const float * x = source + size_t(i) * 32;
    float vmax = 0, sum0 = 0, sum1 = 0;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const float value = x[k];
        vmax = __builtin_fmaxf(vmax, __builtin_fabsf(value));
        if (k < 16) sum0 += value;
        else sum1 += value;
    }
    const float d = vmax / 127.0f;
    C1PackedQ8K32Split result = {};
    result.d = __fp16(d);
    result.s0 = __fp16(sum0);
    result.s1 = __fp16(sum1);
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int q = vmax == 0 ? 0 : int(__builtin_roundf(x[k] / d));
        const uint32_t shift = 4 * (k & 7);
        result.low[k >> 3] |= (uint32_t(q) & 15u) << shift;
        result.high[k >> 3] |= ((uint32_t(q) >> 4) & 15u) << shift;
    }
    output[i] = result;
}

static __device__ __forceinline__ uint32_t c1_load32(const void * source) {
    uint32_t value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}
static __device__ __forceinline__ uint32_t c1_unpack4(uint32_t value) {
    return (value & 0x000fu) | ((value & 0x00f0u) << 4) |
           ((value & 0x0f00u) << 8) | ((value & 0xf000u) << 12);
}
static __device__ __forceinline__ uint32_t c1_unpack_centered4(uint32_t value) {
    // Flip the nibble sign bit, then sign-extend each nibble independently.
    // The returned four bytes are exactly int8_t(code - 8), without a carry
    // between bytes as a packed subtraction would introduce.
    uint32_t bytes = c1_unpack4(value ^ 0x8888u);
    const uint32_t sign = bytes & 0x08080808u;
    bytes |= (sign << 1) | (sign << 2) | (sign << 3) | (sign << 4);
    return bytes;
}
static __device__ __forceinline__ uint32_t c1_restore_q8(uint32_t low, uint32_t high) {
    return c1_unpack4(low) | (c1_unpack4(high) << 4);
}

template<int K, int N, bool NativeIU4, bool Centered>
static __global__ __launch_bounds__(32) void c1_vector_runtime(
        const uint8_t * __restrict__ main, const half * __restrict__ rows,
        const half * __restrict__ tables, const C1PackedQ8K32Split * __restrict__ activation,
        const int32_t * __restrict__ ids, float * __restrict__ output,
        int experts, int routes, int input_channels, int ids_token_stride) {
    constexpr int Groups16 = K / 16;
    constexpr int Groups32 = K / 32;
    constexpr int CodeRowBytes = K / 2;
    constexpr int MainRowBytes = K / 2 + K / 16;
    constexpr int MainExpertBytes = N * MainRowBytes;
    const unsigned lane = threadIdx.x;
    const unsigned n = blockIdx.x;
    const unsigned route = blockIdx.y;
    const unsigned token = blockIdx.z;
    const int expert = ids[size_t(token) * ids_token_stride + route];
    if (expert < 0 || expert >= experts) return;
    const uint8_t * row_main = main + size_t(expert) * MainExpertBytes + n * MainRowBytes;
    const C1PackedQ8K32Split * x = activation +
        (size_t(token) * input_channels + route % input_channels) * Groups32;
    const float r = float(rows[size_t(expert) * N + n]);
    const half * table = tables + size_t(expert) * 512;
    float total = 0.0f;
#pragma clang loop unroll(disable)
    for (unsigned g = lane; g < Groups16; g += 32) {
        const unsigned half_group = g & 1, full_group = g >> 1;
        const uint8_t * code_ptr = row_main + g * 8;
        const C1PackedQ8K32Split & a = x[full_group];
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t code = c1_load32(code_ptr + j * 4);
            const uint32_t low = a.low[half_group * 2 + j];
            const uint32_t high = a.high[half_group * 2 + j];
            if constexpr (NativeIU4) {
                const uint32_t w = Centered ? code ^ 0x88888888u : code;
                dot = __builtin_amdgcn_sudot8(Centered, int(w), false, int(low), dot, false);
                const int hi = __builtin_amdgcn_sudot8(Centered, int(w), true, int(high), 0, false);
                dot += 16 * hi;
            } else {
                const uint32_t w0 = Centered ? c1_unpack_centered4(code & 0xffffu) : c1_unpack4(code & 0xffffu);
                const uint32_t w1 = Centered ? c1_unpack_centered4(code >> 16) : c1_unpack4(code >> 16);
                const uint32_t x0 = c1_restore_q8(low & 0xffffu, high & 0xffffu);
                const uint32_t x1 = c1_restore_q8(low >> 16, high >> 16);
                dot = __builtin_amdgcn_sudot4(Centered, int(w0), true, int(x0), dot, false);
                dot = __builtin_amdgcn_sudot4(Centered, int(w1), true, int(x1), dot, false);
            }
        }
        const uint8_t index = row_main[CodeRowBytes + g];
        const float table_d = float(table[2 * unsigned(index)]);
        const float table_m = float(table[2 * unsigned(index) + 1]);
        const float d = r * table_d;
        const float m = r * (table_m + (Centered ? 8.0f * table_d : 0.0f));
        const float sx = float(half_group ? a.s1 : a.s0);
        total += d * float(a.d) * float(dot) + m * sx;
    }
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
        const uint32_t bits = __builtin_bit_cast(uint32_t, total);
        const uint32_t peer = __builtin_amdgcn_ds_bpermute((lane ^ offset) * 4, bits);
        total += __builtin_bit_cast(float, peer);
    }
    if (lane == 0) output[(size_t(token) * routes + route) * N + n] = total;
}

#include "c1-vector-r4.cuh"

static bool c1_use_same_value_iu8() {
    static const bool use_iu8 = [] {
        const char * value = std::getenv("LLAMA_C1_IU8");
        return value && value[0] == '1';
    }();
    return use_iu8;
}

static bool c1_use_centered_activation() {
    static const bool centered = [] {
        const char * value = std::getenv("LLAMA_C1_CENTERED");
        const bool enabled = value && value[0] == '1';
        if (enabled) std::fprintf(stderr, "C1 centered activation enabled (LLAMA_C1_CENTERED=1)\n");
        return enabled;
    }();
    return centered;
}
#endif

#include "c1-grouped.cuh"

// A deliberately simple F32 reference backend for integration/quality checks.
// The GGUF main tensor contains codes and indices adjacent per output row;
// row multipliers and affine tables are two independent FP16 source tensors.
// It is not the native IU4 execution path and cannot establish speed.
static __global__ void c1_mat_id_reference(
        const uint8_t * __restrict__ main,
        const half * __restrict__ rows,
        const half * __restrict__ tables,
        const float * __restrict__ input,
        const int32_t * __restrict__ ids,
        float * __restrict__ output,
        int k, int n, int experts, int routes, int tokens, int input_channels,
        size_t main_expert_stride, size_t rows_expert_stride, size_t table_expert_stride,
        size_t input_route_stride, size_t input_token_stride,
        size_t ids_token_stride, size_t output_route_stride, size_t output_token_stride) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int route = blockIdx.y;
    const int token = blockIdx.z;
    if (row >= n || route >= routes || token >= tokens) return;

    const int expert = *reinterpret_cast<const int32_t *>(
        reinterpret_cast<const uint8_t *>(ids) + token * ids_token_stride + route * sizeof(int32_t));
    if (expert < 0 || expert >= experts) return;
    const int code_row_bytes = k / 2;
    const int index_row_bytes = k / 16;
    const int main_row_bytes = code_row_bytes + index_row_bytes;
    const uint8_t * row_main = main + expert * main_expert_stride + row * main_row_bytes;
    const uint8_t * code = row_main;
    const uint8_t * index = row_main + code_row_bytes;
    const half * table = reinterpret_cast<const half *>(
        reinterpret_cast<const uint8_t *>(tables) + expert * table_expert_stride);
    const float * x = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(input) + token * input_token_stride
            + (route % input_channels) * input_route_stride);
    const float r = __half2float(*reinterpret_cast<const half *>(
        reinterpret_cast<const uint8_t *>(rows) + expert * rows_expert_stride + row * sizeof(half)));
    float sum = 0.0f;
    for (int group = 0; group < k / 16; ++group) {
        const int idx = index[group];
        const float d = __half2float(table[2 * idx]);
        const float m = __half2float(table[2 * idx + 1]);
        const int base = group * 16;
        for (int j = 0; j < 8; ++j) {
            const uint8_t both = code[group * 8 + j];
            sum += x[base + 2 * j] * (r * (d * float(both & 15) + m));
            sum += x[base + 2 * j + 1] * (r * (d * float(both >> 4) + m));
        }
    }
    *reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(output)
            + token * output_token_stride + route * output_route_stride + row * sizeof(float)) = sum;
}

void ggml_cuda_mul_mat_id_c1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * rows = dst->src[3];
    const ggml_tensor * tables = dst->src[4];
    GGML_ASSERT(w && x && ids && rows && tables);
    GGML_ASSERT(w->type == GGML_TYPE_C1_G16_K2560 || w->type == GGML_TYPE_C1_G16_K640);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(rows->type == GGML_TYPE_F16 && tables->type == GGML_TYPE_F16);
    GGML_ASSERT(w->ne[0] == (w->type == GGML_TYPE_C1_G16_K2560 ? 2560 : 640));
    GGML_ASSERT(rows->ne[0] == w->ne[1] && rows->ne[1] == w->ne[2]);
    GGML_ASSERT(tables->ne[0] == 2 && tables->ne[1] == 256 && tables->ne[2] == w->ne[2]);
    GGML_ASSERT(x->ne[0] == w->ne[0] && ids->ne[0] == dst->ne[1] && ids->ne[1] == dst->ne[2]);
    GGML_ASSERT(dst->ne[0] == w->ne[1]);
    GGML_ASSERT(w->nb[1] == size_t(w->ne[0] / 2 + w->ne[0] / 16));
    GGML_ASSERT(x->nb[0] == sizeof(float) && ids->nb[0] == sizeof(int32_t) && dst->nb[0] == sizeof(float));

    static int dispatch_hits = 0;
    if (std::getenv("LLAMA_C1_TRACE") && dispatch_hits++ < 12) {
        std::fprintf(stderr,
            "C1 dispatch %s centered=%d K=%lld N=%lld experts=%lld routes=%lld tokens=%lld x_channels=%lld "
            "contig(w,x,ids,dst,rows,tables)=%d,%d,%d,%d,%d,%d x_strides=%zu,%zu,%zu ids_strides=%zu,%zu\n",
            c1_use_same_value_iu8() ? "IU8" : "IU4", int(c1_use_centered_activation()),
            (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
            (long long) dst->ne[1], (long long) dst->ne[2], (long long) x->ne[1],
            int(ggml_is_contiguous(w)), int(ggml_is_contiguous(x)), int(ggml_is_contiguous(ids)),
            int(ggml_is_contiguous(dst)), int(ggml_is_contiguous(rows)), int(ggml_is_contiguous(tables)),
            x->nb[0], x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);
    }

#if defined(__HIP_PLATFORM_AMD__)
    if (c1_grouped::run(ctx, dst, c1_use_same_value_iu8(), c1_use_centered_activation())) {
        static int grouped_hits = 0;
        if (std::getenv("LLAMA_C1_TRACE") && grouped_hits++ < 12) {
            std::fprintf(stderr, "C1 grouped %s centered=%d K=%lld N=%lld experts=%lld routes=%lld tokens=%lld x_channels=%lld\n",
                c1_use_same_value_iu8() ? "IU8" : "IU4", int(c1_use_centered_activation()),
                (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                (long long) dst->ne[1], (long long) dst->ne[2], (long long) x->ne[1]);
        }
        return;
    }
#endif

#if defined(__HIP_PLATFORM_AMD__)
    if (dst->ne[2] <= 8 && ggml_is_contiguous(x) && ids->nb[0] == sizeof(int32_t) &&
            ids->nb[1] % sizeof(int32_t) == 0 && ggml_is_contiguous(dst) &&
            ggml_is_contiguous(rows) && ggml_is_contiguous(tables)) {
        static int vector_hits = 0;
        if (std::getenv("LLAMA_C1_TRACE") && vector_hits++ < 12) {
            std::fprintf(stderr, "C1 vector %s centered=%d K=%lld N=%lld experts=%lld routes=%lld tokens=%lld x_channels=%lld\n",
                c1_use_same_value_iu8() ? "IU8" : "IU4", int(c1_use_centered_activation()),
                (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
                (long long) dst->ne[1], (long long) dst->ne[2], (long long) x->ne[1]);
        }
        const int k = int(w->ne[0]);
        const int n = int(w->ne[1]);
        const int groups32 = k / 32;
        const int activation_rows = int(x->ne[1] * x->ne[2]);
        ggml_cuda_pool_alloc<C1PackedQ8K32Split> packed(ctx.pool(), size_t(groups32) * activation_rows);
        const int blocks = (groups32 * activation_rows + 255) / 256;
        c1_pack_q8_runtime<<<blocks, 256, 0, ctx.stream()>>>(
            static_cast<const float *>(x->data), packed.ptr, groups32, activation_rows);
        const dim3 vector_grid(unsigned(n), unsigned(dst->ne[1]), unsigned(dst->ne[2]));
        const auto * main = static_cast<const uint8_t *>(w->data);
        const auto * row_scales = static_cast<const half *>(rows->data);
        const auto * table = static_cast<const half *>(tables->data);
        const auto * selected = static_cast<const int32_t *>(ids->data);
        auto * output = static_cast<float *>(dst->data);
        const int ids_token_stride = int(ids->nb[1] / sizeof(int32_t));
        if (k == 2560 && n == 640) {
            if (c1_use_same_value_iu8()) {
                if (c1_use_centered_activation()) {
                    c1_launch_vector<2560, 640, false, true>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                } else {
                    c1_launch_vector<2560, 640, false, false>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                }
            } else {
                if (c1_use_centered_activation()) {
                    c1_launch_vector<2560, 640, true, true>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                } else {
                    c1_launch_vector<2560, 640, true, false>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                }
            }
        } else if (k == 640 && n == 2560) {
            if (c1_use_same_value_iu8()) {
                if (c1_use_centered_activation()) {
                    c1_launch_vector<640, 2560, false, true>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                } else {
                    c1_launch_vector<640, 2560, false, false>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                }
            } else {
                if (c1_use_centered_activation()) {
                    c1_launch_vector<640, 2560, true, true>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                } else {
                    c1_launch_vector<640, 2560, true, false>(ctx, vector_grid,
                        main, row_scales, table, packed.ptr, selected, output, int(w->ne[2]), int(dst->ne[1]), int(x->ne[1]), ids_token_stride);
                }
            }
        } else {
            GGML_ABORT("unsupported C1 projection geometry");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif

    static int reference_hits = 0;
    if (std::getenv("LLAMA_C1_TRACE") && reference_hits++ < 12) {
        std::fprintf(stderr, "C1 F32 reference K=%lld N=%lld experts=%lld routes=%lld tokens=%lld x_channels=%lld\n",
            (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
            (long long) dst->ne[1], (long long) dst->ne[2], (long long) x->ne[1]);
    }
    const dim3 grid((unsigned(w->ne[1]) + 127) / 128, unsigned(dst->ne[1]), unsigned(dst->ne[2]));
    c1_mat_id_reference<<<grid, 128, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(w->data), static_cast<const half *>(rows->data),
        static_cast<const half *>(tables->data), static_cast<const float *>(x->data),
        static_cast<const int32_t *>(ids->data), static_cast<float *>(dst->data),
        int(w->ne[0]), int(w->ne[1]), int(w->ne[2]), int(dst->ne[1]), int(dst->ne[2]), int(x->ne[1]),
        w->nb[2], rows->nb[1], tables->nb[2], x->ne[1] > 1 ? x->nb[1] : 0,
        x->nb[2], ids->nb[1], dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

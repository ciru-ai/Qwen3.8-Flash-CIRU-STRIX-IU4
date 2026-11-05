#pragma once

// Experimental Strix Halo C1 grouped projection for MUL_MAT_ID with more
// than eight tokens. Four N16 waves share each per-expert table per CTA.
// The frozen 4.5175-bpw GGUF values are consumed directly:
// main tensor = per-output-row [K/2 packed codes][K/16 table indices], plus
// separate FP16 row multipliers and FP16 affine tables. The same packed
// activation is used by the IU4 and same-value IU8 arms.
//
// Include this after C1PackedQ8K32Split and c1_pack_q8_runtime in c1.cu.

#if defined(__HIP_PLATFORM_AMD__)
namespace c1_grouped {

using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));

struct AffinePair { half d, m; };
static_assert(sizeof(AffinePair) == 4);

static __device__ __forceinline__ uint32_t unpack4(uint32_t v) {
    return (v & 0x000fu) | ((v & 0x00f0u) << 4) |
           ((v & 0x0f00u) << 8) | ((v & 0xf000u) << 12);
}
static __device__ __forceinline__ uint32_t unpack_centered4(uint32_t v) {
    uint32_t bytes = unpack4(v ^ 0x8888u);
    const uint32_t sign = bytes & 0x08080808u;
    bytes |= (sign << 1) | (sign << 2) | (sign << 3) | (sign << 4);
    return bytes;
}
static __device__ __forceinline__ uint32_t restore_q8(uint32_t l, uint32_t h) {
    return unpack4(l) | (unpack4(h) << 4);
}

// A K32 group has two adjacent K16 table indices per output row. The run()
// alignment guard and even C1 row strides make this a legal halfword load.
static __device__ __forceinline__ uint16_t load_index_pair(const uint8_t * source) {
    uint16_t pair;
    __builtin_memcpy(&pair, __builtin_assume_aligned(source, alignof(uint16_t)), sizeof(pair));
    return pair;
}

template<bool NativeIU4, bool Centered>
static __device__ __forceinline__ i32x8 contract(uint32_t w0, uint32_t w1,
        uint32_t l0, uint32_t l1, uint32_t h0, uint32_t h1) {
    i32x8 dot = {};
    if constexpr (NativeIU4) {
        const i32x2 w = {int(Centered ? w0 ^ 0x88888888u : w0),
                         int(Centered ? w1 ^ 0x88888888u : w1)};
        const i32x2 hi = {int(h0), int(h1)};
        const i32x2 lo = {int(l0), int(l1)};
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(Centered, w, true, hi, dot, false);
#pragma unroll
        for (int i = 0; i < 8; ++i) dot[i] *= 16;
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(Centered, w, false, lo, dot, false);
    } else {
        const i32x4 w = {int(Centered ? unpack_centered4(w0) : unpack4(w0)),
                         int(Centered ? unpack_centered4(w0 >> 16) : unpack4(w0 >> 16)),
                         int(Centered ? unpack_centered4(w1) : unpack4(w1)),
                         int(Centered ? unpack_centered4(w1 >> 16) : unpack4(w1 >> 16))};
        const i32x4 x = {int(restore_q8(l0, h0)), int(restore_q8(l0 >> 16, h0 >> 16)),
                         int(restore_q8(l1, h1)), int(restore_q8(l1 >> 16, h1 >> 16))};
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(Centered, w, true, x, dot, false);
    }
    return dot;
}

static __global__ void route_count(const int32_t * ids, uint32_t * count,
        int all_routes, int routes_per_token, int ids_token_stride, int experts) {
    const unsigned route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= unsigned(all_routes)) return;
    const int expert = ids[(route / routes_per_token) * ids_token_stride + route % routes_per_token];
    if (expert >= 0 && expert < experts) atomicAdd(count + expert, 1u);
}

static __global__ __launch_bounds__(32) void route_describe(
        const uint32_t * count, uint32_t * cursor, uint32_t * prefix,
        int32_t * tile_expert, int32_t * tile_active, uint32_t * total,
        int experts, int capacity) {
    if (threadIdx.x) return;
    unsigned next = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const unsigned n = count[expert];
        const unsigned tiles = (n + 15) / 16;
        prefix[expert] = next;
        cursor[expert] = 0;
        for (unsigned t = 0; t < tiles && next + t < unsigned(capacity); ++t) {
            tile_expert[next + t] = expert;
            tile_active[next + t] = int(n - t * 16 < 16 ? n - t * 16 : 16);
        }
        next += tiles;
    }
    for (unsigned t = next; t < unsigned(capacity); ++t) {
        tile_expert[t] = -1;
        tile_active[t] = 0;
    }
    *total = next;
}

static __global__ void route_scatter(const int32_t * ids,
        const uint32_t * prefix, uint32_t * cursor,
        int32_t * slot_for_row, int all_routes, int routes_per_token,
        int ids_token_stride, int experts) {
    const unsigned route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= unsigned(all_routes)) return;
    const int expert = ids[(route / routes_per_token) * ids_token_stride + route % routes_per_token];
    if (expert < 0 || expert >= experts) return;
    const unsigned ordinal = atomicAdd(cursor + expert, 1u);
    const unsigned row = (prefix[expert] + ordinal / 16) * 16 + ordinal % 16;
    slot_for_row[row] = int(route);
}

template<int K, int N, bool NativeIU4, bool Centered>
static __global__ __launch_bounds__(128) void project(
        const uint8_t * __restrict__ main,
        const half * __restrict__ row_scales,
        const half * __restrict__ tables,
        const C1PackedQ8K32Split * __restrict__ activation,
        const int32_t * __restrict__ tile_expert,
        const int32_t * __restrict__ tile_active,
        const int32_t * __restrict__ slot_for_row,
        float * __restrict__ output,
        int capacity, int experts, int routes, int input_channels,
        size_t main_expert_stride, size_t row_expert_stride,
        size_t table_expert_stride) {
    constexpr int G16 = K / 16, G32 = K / 32;
    constexpr int CodeRowBytes = K / 2;
    constexpr int MainRowBytes = K / 2 + K / 16;
    static_assert(CodeRowBytes % 2 == 0 && MainRowBytes % 2 == 0);
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31;
    const unsigned wave = thread >> 5;
    const unsigned lane16 = lane & 15, half_lane = lane >> 4;
    const unsigned tile_n = blockIdx.x * 4 + wave, tile_m = blockIdx.y;
    if (tile_m >= unsigned(capacity)) return;
    const int expert = tile_expert[tile_m], active = tile_active[tile_m];
    if (expert < 0 || expert >= experts || active <= 0) return;
    const uint8_t * expert_main = main + size_t(expert) * main_expert_stride;
    const half * expert_rows = reinterpret_cast<const half *>(
        reinterpret_cast<const uint8_t *>(row_scales) + size_t(expert) * row_expert_stride);
    const uint32_t * expert_table = reinterpret_cast<const uint32_t *>(
        reinterpret_cast<const uint8_t *>(tables) + size_t(expert) * table_expert_stride);
    __shared__ uint32_t table_shared[256];
#pragma unroll
    for (int i = 0; i < 2; ++i) table_shared[thread + 128 * i] = expert_table[thread + 128 * i];
    __syncthreads();

    const unsigned feature_lane = tile_n * 16 + lane16;
    const unsigned grouped_row = tile_m * 16 + lane16;
    const int slot = lane16 < unsigned(active) ? slot_for_row[grouped_row] : 0;
    const int token = slot / routes, route = slot % routes;
    const unsigned input_row = unsigned(token * input_channels + route % input_channels);
    float totals[8] = {}, scales[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned feature = tile_n * 16 + 2 * i + half_lane;
        scales[i] = float(expert_rows[feature]);
    }
#pragma clang loop unroll(disable)
    for (unsigned g = 0; g < G32; ++g) {
        // Submit eight halfword index loads before the WMMA and table lookup.
        // Keeping the active-lane predicate avoids fetching unused rows in a
        // partially occupied routed tile. The loaded values are used by both
        // h iterations, so this does not alter the F32 accumulation order.
        uint16_t index_pairs[8];
        if (lane16 < unsigned(active)) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned feature = tile_n * 16 + 2 * i + half_lane;
                index_pairs[i] = load_index_pair(
                    expert_main + feature * MainRowBytes + CodeRowBytes + g * 2);
            }
        }
        const uint32_t * weight = reinterpret_cast<const uint32_t *>(
            expert_main + feature_lane * MainRowBytes + g * 16);
        uint32_t w[4], lo[4] = {}, hi[4] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = weight[j];
        float dx = 0, sum0 = 0, sum1 = 0;
        if (lane16 < unsigned(active)) {
            const C1PackedQ8K32Split & a = activation[size_t(input_row) * G32 + g];
            dx = float(a.d); sum0 = float(a.s0); sum1 = float(a.s1);
#pragma unroll
            for (int j = 0; j < 4; ++j) { lo[j] = a.low[j]; hi[j] = a.high[j]; }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const i32x8 dot = contract<NativeIU4, Centered>(
                w[2 * h], w[2 * h + 1], lo[2 * h], lo[2 * h + 1],
                hi[2 * h], hi[2 * h + 1]);
#pragma unroll
            for (int i = 0; i < 8; ++i) if (lane16 < unsigned(active)) {
                const uint8_t index = uint8_t(index_pairs[i] >> (8 * h));
                AffinePair pair;
                const uint32_t bits = table_shared[index];
                __builtin_memcpy(&pair, &bits, sizeof(bits));
                const float s = h ? sum1 : sum0;
                const float table_d = float(pair.d);
                const float table_m = float(pair.m);
                totals[i] += scales[i] * table_d * dx * float(dot[i])
                           + scales[i] * (table_m + (Centered ? 8.0f * table_d : 0.0f)) * s;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) if (lane16 < unsigned(active)) {
        const unsigned feature = tile_n * 16 + 2 * i + half_lane;
        output[size_t(slot) * N + feature] = totals[i];
    }
}

static bool run(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        bool same_value_iu8, bool centered) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    const ggml_tensor * rows = dst->src[3], * tables = dst->src[4];
    const int tokens = int(dst->ne[2]), routes = int(dst->ne[1]);
    const int experts = int(w->ne[2]), k = int(w->ne[0]), n = int(w->ne[1]);
    const int input_channels = int(x->ne[1]);
    if (tokens <= 8 || experts != 512 || routes <= 0 ||
            (input_channels != 1 && input_channels != routes) ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
            ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
            !ggml_is_contiguous(dst) ||
            !ggml_is_contiguous(rows) || !ggml_is_contiguous(tables)) return false;
    if (!((k == 2560 && n == 640) || (k == 640 && n == 2560))) return false;
    if ((reinterpret_cast<uintptr_t>(w->data) & 1u) ||
            (w->nb[1] & 1u) || (w->nb[2] & 1u)) return false;
    const int all_routes = tokens * routes;
    const int occupied_max = experts < all_routes ? experts : all_routes;
    const int capacity = occupied_max + (all_routes - occupied_max) / 16;
    const int groups32 = k / 32;
    ggml_cuda_pool_alloc<uint32_t> count(ctx.pool(), experts);
    ggml_cuda_pool_alloc<uint32_t> cursor(ctx.pool(), experts);
    ggml_cuda_pool_alloc<uint32_t> prefix(ctx.pool(), experts);
    ggml_cuda_pool_alloc<uint32_t> total(ctx.pool(), 1);
    ggml_cuda_pool_alloc<int32_t> tile_expert(ctx.pool(), capacity);
    ggml_cuda_pool_alloc<int32_t> tile_active(ctx.pool(), capacity);
    ggml_cuda_pool_alloc<int32_t> slot_for_row(ctx.pool(), size_t(capacity) * 16);
    ggml_cuda_pool_alloc<C1PackedQ8K32Split> activation(
        ctx.pool(), size_t(groups32) * input_channels * tokens);
    CUDA_CHECK(cudaMemsetAsync(count.ptr, 0, experts * sizeof(uint32_t), ctx.stream()));
    const auto * selected = static_cast<const int32_t *>(ids->data);
    const int ids_token_stride = int(ids->nb[1] / sizeof(int32_t));
    route_count<<<(all_routes + 255) / 256, 256, 0, ctx.stream()>>>(
        selected, count.ptr, all_routes, routes, ids_token_stride, experts);
    route_describe<<<1, 32, 0, ctx.stream()>>>(
        count.ptr, cursor.ptr, prefix.ptr, tile_expert.ptr, tile_active.ptr,
        total.ptr, experts, capacity);
    route_scatter<<<(all_routes + 255) / 256, 256, 0, ctx.stream()>>>(
        selected, prefix.ptr, cursor.ptr, slot_for_row.ptr,
        all_routes, routes, ids_token_stride, experts);
    c1_pack_q8_runtime<<<(groups32 * input_channels * tokens + 255) / 256,
        256, 0, ctx.stream()>>>(static_cast<const float *>(x->data),
        activation.ptr, groups32, input_channels * tokens);
    const dim3 grid(unsigned(n / 64), unsigned(capacity));
#define C1_GROUPED_LAUNCH(K, N, IU4, CENTERED) \
        project<K, N, IU4, CENTERED><<<grid, 128, 0, ctx.stream()>>>( \
            static_cast<const uint8_t *>(w->data), static_cast<const half *>(rows->data), \
            static_cast<const half *>(tables->data), activation.ptr, \
            tile_expert.ptr, tile_active.ptr, slot_for_row.ptr, \
            static_cast<float *>(dst->data), capacity, experts, routes, input_channels, \
            w->nb[2], rows->nb[1], tables->nb[2])
    if (k == 2560 && n == 640) {
        if (same_value_iu8) {
            if (centered) { C1_GROUPED_LAUNCH(2560, 640, false, true); }
            else          { C1_GROUPED_LAUNCH(2560, 640, false, false); }
        } else {
            if (centered) { C1_GROUPED_LAUNCH(2560, 640, true, true); }
            else          { C1_GROUPED_LAUNCH(2560, 640, true, false); }
        }
    } else {
        if (same_value_iu8) {
            if (centered) { C1_GROUPED_LAUNCH(640, 2560, false, true); }
            else          { C1_GROUPED_LAUNCH(640, 2560, false, false); }
        } else {
            if (centered) { C1_GROUPED_LAUNCH(640, 2560, true, true); }
            else          { C1_GROUPED_LAUNCH(640, 2560, true, false); }
        }
    }
#undef C1_GROUPED_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace c1_grouped
#endif

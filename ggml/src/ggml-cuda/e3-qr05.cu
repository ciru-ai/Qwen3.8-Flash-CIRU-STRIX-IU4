#include "e3-qr05.cuh"

#if defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>

// Reuse the evidence-locked H30/H24 source bodies. The nested source tree
// preserves H30's original relative include and vendors only the two Opus
// headers those kernels consume.
#define H30_NO_HOST_MAIN
#include "e3-qr05-locked/h30-e3-fused-middle-q8/code/h30_e3_fused_middle_q8.cu"
#undef H30_NO_HOST_MAIN

namespace {

#if defined(__HIP_DEVICE_COMPILE__)
using e3_i8_t = h24d::i8_t;
using e3_fp16_t = h24d::fp16_t;
#else
using e3_i8_t = signed char;
using e3_fp16_t = _Float16;
#endif

constexpr unsigned kModelWidth = 2560;
constexpr unsigned kExpertWidth = 640;
constexpr unsigned kTopK = 10;
constexpr unsigned kGateGroup = 128;
constexpr unsigned kDownGroup = 64;
constexpr unsigned kGateGroups = kModelWidth / kGateGroup;
constexpr unsigned kDownGroups = kExpertWidth / kDownGroup;
constexpr unsigned kLayerBytes = 1363148800u;
constexpr unsigned kOutputBytes = kModelWidth * sizeof(float);
constexpr unsigned kInputQ8Bytes = kModelWidth;
constexpr unsigned kInputMetaBytes = kGateGroups * 2 * sizeof(e3_fp16_t);
constexpr unsigned kMiddleQ8Bytes = kTopK * kExpertWidth;
constexpr unsigned kMiddleMetaBytes =
    kTopK * kDownGroups * 2 * sizeof(e3_fp16_t);
constexpr unsigned kProductionWorkspaceBytes = kOutputBytes + kInputQ8Bytes +
    kInputMetaBytes + kMiddleQ8Bytes + kMiddleMetaBytes;
constexpr unsigned kExperts = 512;
constexpr unsigned kRouteTile = 4;
constexpr unsigned kWmmaMinRoutes = 4;
constexpr unsigned kWmmaRouteTile = 16;
constexpr unsigned kOutputTile = 32;
constexpr unsigned kWmmaOutputTile = 16;
constexpr unsigned kInputDigitBytes = kModelWidth / 2;
constexpr unsigned kMiddleDigitBytes = kTopK * kExpertWidth / 2;

static_assert(kProductionWorkspaceBytes == 19680);
static_assert(kLayerBytes == 512u * 2662400u);
static_assert(kModelWidth == H24_MODEL_WIDTH);
static_assert(kExpertWidth == H24_EXPERT_WIDTH);
static_assert(kTopK == H24_TOP_K);

#if defined(__HIP_DEVICE_COMPILE__)
using e3_i32x2 = std::int32_t __attribute__((ext_vector_type(2)));
using e3_i32x8 = std::int32_t __attribute__((ext_vector_type(8)));
#endif

extern "C" __global__ __launch_bounds__(32)
void h36_quantize_input_q8_g128(
        const float * input_f32,
        e3_i8_t * input_q8,
        e3_fp16_t * input_meta);

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" __global__ __launch_bounds__(32)
void h36_quantize_input_q8_g128(
        const float * __restrict__ input_f32,
        e3_i8_t * __restrict__ input_q8,
        e3_fp16_t * __restrict__ input_meta) {
    constexpr int kPerLane = kGateGroup / 32;
    const int group = opus::block_id_x();
    const int lane = opus::thread_id_x();
    const int group_base = group * kGateGroup;
    auto input = opus::make_gmem(
        input_f32, kModelWidth * sizeof(float), h24d::kRdnaBufferConfig);
    auto output = opus::make_gmem(
        input_q8, kModelWidth, h24d::kRdnaBufferConfig);
    auto meta = opus::make_gmem(
        input_meta, kGateGroups * 2 * sizeof(h24d::fp16_t),
        h24d::kRdnaBufferConfig);

    float values[kPerLane];
    float max_abs = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        values[pass] = input.template load<1>(
            group_base + lane + pass * 32)[0];
        max_abs = opus::max(max_abs, __builtin_fabsf(values[pass]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const float peer = opus::shfl(max_abs, lane + delta, 32);
        if (lane < delta) {
            max_abs = opus::max(max_abs, peer);
        }
    }

    float d = max_abs / 127.0f;
    if (lane == 0) {
        d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
    }
    d = opus::shfl(d, 0, 32);

    int local_sum = 0;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        int q = d == 0.0f
            ? 0
            : static_cast<int>(__builtin_rintf(values[pass] / d));
        q = opus::max(-127, opus::min(127, q));
        output.template store<1>(
            static_cast<e3_i8_t>(q), group_base + lane + pass * 32);
        local_sum += q;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int peer = opus::shfl(local_sum, lane + delta, 32);
        if (lane < delta) {
            local_sum += peer;
        }
    }
    if (lane == 0) {
        opus::vector_t<e3_fp16_t, 2> pair;
        pair[0] = opus::fp32_to_fp16(d);
        pair[1] = opus::fp32_to_fp16(d * static_cast<float>(local_sum));
        meta.template store<2>(pair, group * 2);
    }
}

__device__ __forceinline__ h24d::u32_t h47_pack_even(
        const e3_i8_t * q) {
    return static_cast<h24d::u32_t>(static_cast<unsigned char>(q[0])) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[2])) << 8) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[4])) << 16) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[6])) << 24);
}

__device__ __forceinline__ h24d::u32_t h47_pack_odd(
        const e3_i8_t * q) {
    return static_cast<h24d::u32_t>(static_cast<unsigned char>(q[1])) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[3])) << 8) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[5])) << 16) |
        (static_cast<h24d::u32_t>(static_cast<unsigned char>(q[7])) << 24);
}

__device__ __forceinline__ void h47_dot8_b4(
        const std::uint8_t * code,
        const e3_i8_t * const q[kRouteTile],
        int count,
        int dot[kRouteTile]) {
    const h24d::u32_t packed =
        *reinterpret_cast<const h24d::u32_t *>(code);
    const h24d::u32_t even = packed & 0x0f0f0f0fu;
    const h24d::u32_t odd = (packed >> 4) & 0x0f0f0f0fu;
#pragma unroll
    for (int b = 0; b < kRouteTile; ++b) {
        if (b < count) {
            dot[b] = __builtin_amdgcn_sudot4(
                false, even, true, h47_pack_even(q[b]), dot[b], false);
            dot[b] = __builtin_amdgcn_sudot4(
                false, odd, true, h47_pack_odd(q[b]), dot[b], false);
        }
    }
}

extern "C" __global__ __launch_bounds__(32)
void h47_zero_output(float * output, unsigned n_tokens) {
    const unsigned index = opus::block_id_x() * 32 + opus::thread_id_x();
    const unsigned elements = n_tokens * kModelWidth;
    if (index < elements) output[index] = 0.0f;
}

extern "C" __global__ __launch_bounds__(32)
void h47_quantize_input_q8_g128_batched(
        const float * input_f32,
        e3_i8_t * input_q8,
        e3_fp16_t * input_meta,
        std::uint8_t * input_low,
        std::uint8_t * input_high,
        unsigned n_tokens) {
    constexpr int kPerLane = kGateGroup / 32;
    const unsigned linear_group = opus::block_id_x();
    const unsigned token = linear_group / kGateGroups;
    const unsigned group = linear_group - token * kGateGroups;
    if (token >= n_tokens) return;
    const int lane = opus::thread_id_x();
    const unsigned group_base = token * kModelWidth + group * kGateGroup;
    float values[kPerLane];
    float max_abs = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        values[pass] = input_f32[group_base + lane + pass * 32];
        max_abs = opus::max(max_abs, __builtin_fabsf(values[pass]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const float peer = opus::shfl(max_abs, lane + delta, 32);
        if (lane < delta) max_abs = opus::max(max_abs, peer);
    }
    float d = max_abs / 127.0f;
    if (lane == 0) d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
    d = opus::shfl(d, 0, 32);
    int local_sum = 0;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        int q = d == 0.0f ? 0 :
            static_cast<int>(__builtin_rintf(values[pass] / d));
        q = opus::max(-127, opus::min(127, q));
        input_q8[group_base + lane + pass * 32] = static_cast<e3_i8_t>(q);
        local_sum += q;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int peer = opus::shfl(local_sum, lane + delta, 32);
        if (lane < delta) local_sum += peer;
    }
    if (lane == 0) {
        const unsigned meta_base = token * kGateGroups * 2 + group * 2;
        input_meta[meta_base] = opus::fp32_to_fp16(d);
        input_meta[meta_base + 1] = opus::fp32_to_fp16(
            d * static_cast<float>(local_sum));
    }
    opus::sync_threads();

    // Preserve the signed-Q8 codes for the vector path and additionally emit
    // exact radix-16 low-U4/high-S4 fragments for the B16 WMMA path.  Each
    // byte packs two consecutive K values in the same order as the resident
    // E3 packet; no persistent bank transform is involved.
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const unsigned byte = static_cast<unsigned>(lane + pass * 32);
        const unsigned k = byte * 2;
        const int q0 = static_cast<int>(input_q8[group_base + k]);
        const int q1 = static_cast<int>(input_q8[group_base + k + 1]);
        const int lo0 = q0 & 0x0f;
        const int lo1 = q1 & 0x0f;
        const int hi0 = (q0 - lo0) / 16;
        const int hi1 = (q1 - lo1) / 16;
        const size_t digit_index = static_cast<size_t>(token) * kInputDigitBytes +
            group * (kGateGroup / 2) + byte;
        input_low[digit_index] = static_cast<std::uint8_t>(lo0 | (lo1 << 4));
        input_high[digit_index] = static_cast<std::uint8_t>(
            (hi0 & 0x0f) | ((hi1 & 0x0f) << 4));
    }
}

// One device thread creates the expert-major route plan on the caller stream.
extern "C" __global__ __launch_bounds__(1)
void h47_plan_routes(
        const int * expert_ids,
        unsigned n_tokens,
        int * header,
        int * counts,
        int * starts,
        int * cursors,
        int * route_indices,
        int * descriptors) {
    if (opus::block_id_x() != 0 || opus::thread_id_x() != 0) return;
    const int routes = static_cast<int>(n_tokens * kTopK);
    header[0] = 0;
    header[1] = 0;
    header[2] = 0;
    header[3] = routes;
    for (unsigned expert = 0; expert < kExperts; ++expert) counts[expert] = 0;
    for (unsigned token = 0; token < n_tokens; ++token) {
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            const int route = static_cast<int>(token * kTopK + slot);
            const int expert = expert_ids[route];
            if (expert < 0 || expert >= static_cast<int>(kExperts)) {
                __builtin_trap();
            }
            for (unsigned prior = 0; prior < slot; ++prior) {
                if (expert_ids[token * kTopK + prior] == expert) {
                    __builtin_trap();
                }
            }
            ++counts[expert];
        }
    }
    starts[0] = 0;
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        starts[expert + 1] = starts[expert] + counts[expert];
        cursors[expert] = starts[expert];
    }
    for (int route = 0; route < routes; ++route) {
        const int expert = expert_ids[route];
        route_indices[cursors[expert]++] = route;
    }
    int descriptor_count = 0;
    // Put every expert with at least four routes on the masked B4..B16 WMMA
    // path.  Avoid stranding a B1..B3 tail after a full B16 tile: 17/18/19
    // routes become 13+4 / 14+4 / 15+4.  Only experts whose complete bucket
    // has one to three routes are left for the unchanged vector fallback.
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        const int count = counts[expert];
        if (count < static_cast<int>(kWmmaMinRoutes)) continue;
        int offset = 0;
        int remaining = count;
        while (remaining >= static_cast<int>(kWmmaMinRoutes)) {
            int take = remaining < static_cast<int>(kWmmaRouteTile) ?
                remaining : static_cast<int>(kWmmaRouteTile);
            if (remaining > static_cast<int>(kWmmaRouteTile) &&
                remaining - take < static_cast<int>(kWmmaMinRoutes)) {
                take = remaining - static_cast<int>(kWmmaMinRoutes);
            }
            descriptors[descriptor_count * 2] = starts[expert] + offset;
            descriptors[descriptor_count * 2 + 1] =
                static_cast<int>(expert) | (take << 16);
            ++descriptor_count;
            offset += take;
            remaining -= take;
        }
    }
    header[4] = descriptor_count;
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        const int count = counts[expert];
        if (count > 0 && count < static_cast<int>(kWmmaMinRoutes)) {
            descriptors[descriptor_count * 2] = starts[expert];
            descriptors[descriptor_count * 2 + 1] =
                static_cast<int>(expert) | (count << 16);
            ++descriptor_count;
            header[2] += count;
        }
    }
    header[1] = descriptor_count;
    header[5] = descriptor_count - header[4];
}

#if 0
// Retained only as the rejected H57/H57A evidence body.  H59 does not compile
// or launch either cross-wave implementation below.
extern "C" __global__ __launch_bounds__(128)
void h57_grouped_gate_up_wmma_b16_d64(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const e3_fp16_t * input_meta,
        const std::uint8_t * input_low,
        const std::uint8_t * input_high,
        e3_i8_t * middle_q8,
        e3_fp16_t * middle_meta,
        std::uint8_t * middle_low,
        std::uint8_t * middle_high) {
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor = linear / kDownGroups;
    const unsigned down_group = linear - descriptor * kDownGroups;
    if (descriptor >= max_descriptors ||
        descriptor >= static_cast<unsigned>(header[4])) return;

    const int thread = opus::thread_id_x();
    const int wave = thread >> 5;
    const int lane = thread & 31;
    const int lane16 = lane & 15;
    const int lane_half = lane >> 4;
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count != static_cast<int>(kWmmaRouteTile)) return;

    __shared__ int routes[kWmmaRouteTile];
    __shared__ std::uint32_t activation_low[kGateGroup / 8][kWmmaRouteTile];
    __shared__ std::uint32_t activation_high[kGateGroup / 8][kWmmaRouteTile];
    __shared__ e3_fp16_t activation_meta[kWmmaRouteTile][2];
    __shared__ float middle[kWmmaOutputTile][kWmmaRouteTile];

    if (thread < static_cast<int>(kWmmaRouteTile)) {
        routes[thread] = route_indices[start + thread];
    }
    opus::sync_threads();

    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;
    float gate_even[8] = {};
    float gate_odd[8] = {};
    float up_even[8] = {};
    float up_odd[8] = {};

#pragma clang loop unroll(disable)
    for (unsigned group = 0; group < kGateGroups; ++group) {
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int flat = thread + pass * 128;
            const int route_local = flat / static_cast<int>(kGateGroup / 8);
            const int word = flat - route_local * static_cast<int>(kGateGroup / 8);
            const int token = routes[route_local] / kTopK;
            const size_t source_word = static_cast<size_t>(token) * (kModelWidth / 8) +
                group * (kGateGroup / 8) + word;
            activation_low[word][route_local] =
                reinterpret_cast<const std::uint32_t *>(input_low)[source_word];
            activation_high[word][route_local] =
                reinterpret_cast<const std::uint32_t *>(input_high)[source_word];
        }
        if (thread < static_cast<int>(kWmmaRouteTile * 2)) {
            const int route_local = thread >> 1;
            const int component = thread & 1;
            const int token = routes[route_local] / kTopK;
            activation_meta[route_local][component] =
                input_meta[static_cast<size_t>(token) * kGateGroups * 2 +
                    group * 2 + component];
        }
        opus::sync_threads();

        const int input_row = static_cast<int>(down_group) * kWmmaOutputTile +
            wave * 16 + lane16;
        const std::uint8_t * gate_code = expert_bank + expert_base +
            h24d::kGateCodeOffset + static_cast<size_t>(input_row) * (kModelWidth / 2) +
            group * (kGateGroup / 2);
        const std::uint8_t * up_code = expert_bank + expert_base +
            h24d::kGateCodeOffset +
            static_cast<size_t>(kExpertWidth + input_row) * (kModelWidth / 2) +
            group * (kGateGroup / 2);

        const float activation_d = opus::fp16_to_fp32(activation_meta[lane16][0]);
        const float activation_sum = opus::fp16_to_fp32(activation_meta[lane16][1]);
        {
            e3_i32x8 low_acc = {};
            e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
            for (int tile = 0; tile < static_cast<int>(kGateGroup / 16); ++tile) {
                e3_i32x2 weight_fragment;
                e3_i32x2 low_fragment;
                e3_i32x2 high_fragment;
                const std::uint32_t * weight_words =
                    reinterpret_cast<const std::uint32_t *>(gate_code + tile * 8);
                weight_fragment[0] = static_cast<std::int32_t>(weight_words[0]);
                weight_fragment[1] = static_cast<std::int32_t>(weight_words[1]);
                low_fragment[0] = static_cast<std::int32_t>(activation_low[tile * 2][lane16]);
                low_fragment[1] = static_cast<std::int32_t>(activation_low[tile * 2 + 1][lane16]);
                high_fragment[0] = static_cast<std::int32_t>(activation_high[tile * 2][lane16]);
                high_fragment[1] = static_cast<std::int32_t>(activation_high[tile * 2 + 1][lane16]);
                low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, false, low_fragment, low_acc, false);
                high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, true, high_fragment, high_acc, false);
            }
#pragma unroll
            for (int value = 0; value < 8; ++value) {
                const int local_row = wave * 16 + 2 * value + lane_half;
                const int output_row = static_cast<int>(down_group) * kWmmaOutputTile + local_row;
                const e3_fp16_t * weight_meta = reinterpret_cast<const e3_fp16_t *>(
                    expert_bank + expert_base + h24d::kGateMetaOffset) +
                    static_cast<size_t>(output_row) * kGateGroups * 2 + group * 2;
                const int dot = low_acc[value] + 16 * high_acc[value];
                if (group & 1) gate_odd[value] +=
                    opus::fp16_to_fp32(weight_meta[0]) *
                        (activation_d * static_cast<float>(dot)) +
                    opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
                else gate_even[value] +=
                    opus::fp16_to_fp32(weight_meta[0]) *
                        (activation_d * static_cast<float>(dot)) +
                    opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
            }
        }
        {
            e3_i32x8 low_acc = {};
            e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
            for (int tile = 0; tile < static_cast<int>(kGateGroup / 16); ++tile) {
                e3_i32x2 weight_fragment;
                e3_i32x2 low_fragment;
                e3_i32x2 high_fragment;
                const std::uint32_t * weight_words =
                    reinterpret_cast<const std::uint32_t *>(up_code + tile * 8);
                weight_fragment[0] = static_cast<std::int32_t>(weight_words[0]);
                weight_fragment[1] = static_cast<std::int32_t>(weight_words[1]);
                low_fragment[0] = static_cast<std::int32_t>(activation_low[tile * 2][lane16]);
                low_fragment[1] = static_cast<std::int32_t>(activation_low[tile * 2 + 1][lane16]);
                high_fragment[0] = static_cast<std::int32_t>(activation_high[tile * 2][lane16]);
                high_fragment[1] = static_cast<std::int32_t>(activation_high[tile * 2 + 1][lane16]);
                low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, false, low_fragment, low_acc, false);
                high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, true, high_fragment, high_acc, false);
            }
#pragma unroll
            for (int value = 0; value < 8; ++value) {
                const int local_row = wave * 16 + 2 * value + lane_half;
                const int output_row = static_cast<int>(down_group) * kWmmaOutputTile + local_row;
                const e3_fp16_t * weight_meta = reinterpret_cast<const e3_fp16_t *>(
                    expert_bank + expert_base + h24d::kGateMetaOffset) +
                    static_cast<size_t>(kExpertWidth + output_row) * kGateGroups * 2 + group * 2;
                const int dot = low_acc[value] + 16 * high_acc[value];
                if (group & 1) up_odd[value] +=
                    opus::fp16_to_fp32(weight_meta[0]) *
                        (activation_d * static_cast<float>(dot)) +
                    opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
                else up_even[value] +=
                    opus::fp16_to_fp32(weight_meta[0]) *
                        (activation_d * static_cast<float>(dot)) +
                    opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
            }
        }
        opus::sync_threads();
    }

#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int local_row = wave * 16 + 2 * value + lane_half;
        const float gate = gate_even[value] + gate_odd[value];
        const float up = up_even[value] + up_odd[value];
        middle[local_row][lane16] = h24d::silu(gate) * up;
    }
    opus::sync_threads();

    // Four waves quantize four routes each.  The Q8 codes remain the source of
    // truth for vector tails; the packed digit planes are an exact view used
    // only by B16 down descriptors.
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int route_local = wave * 4 + pass;
        const int channel0 = lane * 2;
        const int channel1 = channel0 + 1;
        float v0 = middle[channel0][route_local];
        float v1 = middle[channel1][route_local];
        float max_abs = opus::max(__builtin_fabsf(v0), __builtin_fabsf(v1));
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            const float peer = opus::shfl(max_abs, lane + delta, 32);
            if (lane < delta) max_abs = opus::max(max_abs, peer);
        }
        float d = max_abs / 127.0f;
        if (lane == 0) d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
        d = opus::shfl(d, 0, 32);
        int q0 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(v0 / d));
        int q1 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(v1 / d));
        q0 = opus::max(-127, opus::min(127, q0));
        q1 = opus::max(-127, opus::min(127, q1));
        const int route = routes[route_local];
        const size_t code_base = static_cast<size_t>(route) * kExpertWidth +
            down_group * kWmmaOutputTile;
        middle_q8[code_base + channel0] = static_cast<e3_i8_t>(q0);
        middle_q8[code_base + channel1] = static_cast<e3_i8_t>(q1);
        const int lo0 = q0 & 0x0f;
        const int lo1 = q1 & 0x0f;
        const int hi0 = (q0 - lo0) / 16;
        const int hi1 = (q1 - lo1) / 16;
        const size_t digit_base = static_cast<size_t>(route) * (kExpertWidth / 2) +
            down_group * (kWmmaOutputTile / 2) + lane;
        middle_low[digit_base] = static_cast<std::uint8_t>(lo0 | (lo1 << 4));
        middle_high[digit_base] = static_cast<std::uint8_t>(
            (hi0 & 0x0f) | ((hi1 & 0x0f) << 4));
        int sum = q0 + q1;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            const int peer = opus::shfl(sum, lane + delta, 32);
            if (lane < delta) sum += peer;
        }
        if (lane == 0) {
            const size_t meta_base = static_cast<size_t>(route) * kDownGroups * 2 +
                down_group * 2;
            middle_meta[meta_base] = opus::fp32_to_fp16(d);
            middle_meta[meta_base + 1] = opus::fp32_to_fp16(
                d * static_cast<float>(sum));
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void h57_grouped_down_wmma_b16_m64(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const float * route_weights,
        const e3_fp16_t * middle_meta,
        const std::uint8_t * middle_low,
        const std::uint8_t * middle_high,
        float * output) {
    constexpr unsigned kRowTiles = kModelWidth / kWmmaOutputTile;
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor = linear / kRowTiles;
    const unsigned row_tile = linear - descriptor * kRowTiles;
    if (descriptor >= max_descriptors ||
        descriptor >= static_cast<unsigned>(header[4])) return;

    const int thread = opus::thread_id_x();
    const int wave = thread >> 5;
    const int lane = thread & 31;
    const int lane16 = lane & 15;
    const int lane_half = lane >> 4;
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count != static_cast<int>(kWmmaRouteTile)) return;

    __shared__ int routes[kWmmaRouteTile];
    __shared__ std::uint32_t activation_low[kExpertWidth / 8][kWmmaRouteTile];
    __shared__ std::uint32_t activation_high[kExpertWidth / 8][kWmmaRouteTile];
    __shared__ e3_fp16_t activation_meta[kDownGroups * 2][kWmmaRouteTile];
    if (thread < static_cast<int>(kWmmaRouteTile)) {
        routes[thread] = route_indices[start + thread];
    }
    opus::sync_threads();

    // A B16 down descriptor reuses the complete 16x640 activation tile across
    // all forty M64 row blocks.  Stage it once, transposed by route, instead
    // of paying a barrier and global reload for every G64 group.
#pragma unroll
    for (int pass = 0; pass < 10; ++pass) {
        const int flat = thread + pass * 128;
        const int word = flat / static_cast<int>(kWmmaRouteTile);
        const int route_local = flat - word * static_cast<int>(kWmmaRouteTile);
        const int route = routes[route_local];
        const size_t source_word = static_cast<size_t>(route) * (kExpertWidth / 8) + word;
        activation_low[word][route_local] =
            reinterpret_cast<const std::uint32_t *>(middle_low)[source_word];
        activation_high[word][route_local] =
            reinterpret_cast<const std::uint32_t *>(middle_high)[source_word];
    }
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const int flat = thread + pass * 128;
        if (flat < static_cast<int>(kDownGroups * 2 * kWmmaRouteTile)) {
            const int meta_index = flat / static_cast<int>(kWmmaRouteTile);
            const int route_local = flat - meta_index * static_cast<int>(kWmmaRouteTile);
            const int route = routes[route_local];
            activation_meta[meta_index][route_local] =
                middle_meta[static_cast<size_t>(route) * kDownGroups * 2 + meta_index];
        }
    }
    opus::sync_threads();

    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;
    float values[8] = {};
#pragma clang loop unroll(disable)
    for (unsigned group = 0; group < kDownGroups; ++group) {
        const int input_row = static_cast<int>(row_tile) * kWmmaOutputTile +
            wave * 16 + lane16;
        const std::uint8_t * code = expert_bank + expert_base +
            h24d::kDownCodeOffset + static_cast<size_t>(input_row) * (kExpertWidth / 2) +
            group * (kDownGroup / 2);
        e3_i32x8 low_acc = {};
        e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
        for (int tile = 0; tile < static_cast<int>(kDownGroup / 16); ++tile) {
            e3_i32x2 weight_fragment;
            e3_i32x2 low_fragment;
            e3_i32x2 high_fragment;
            const std::uint32_t * weight_words =
                reinterpret_cast<const std::uint32_t *>(code + tile * 8);
            weight_fragment[0] = static_cast<std::int32_t>(weight_words[0]);
            weight_fragment[1] = static_cast<std::int32_t>(weight_words[1]);
            const int word = static_cast<int>(group) * (kDownGroup / 8) + tile * 2;
            low_fragment[0] = static_cast<std::int32_t>(activation_low[word][lane16]);
            low_fragment[1] = static_cast<std::int32_t>(activation_low[word + 1][lane16]);
            high_fragment[0] = static_cast<std::int32_t>(activation_high[word][lane16]);
            high_fragment[1] = static_cast<std::int32_t>(activation_high[word + 1][lane16]);
            low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, weight_fragment, false, low_fragment, low_acc, false);
            high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, weight_fragment, true, high_fragment, high_acc, false);
        }

        const float activation_d = opus::fp16_to_fp32(activation_meta[group * 2][lane16]);
        const float activation_sum = opus::fp16_to_fp32(activation_meta[group * 2 + 1][lane16]);
#pragma unroll
        for (int value = 0; value < 8; ++value) {
            const int local_row = wave * 16 + 2 * value + lane_half;
            const int output_row = static_cast<int>(row_tile) * kWmmaOutputTile + local_row;
            const e3_fp16_t * weight_meta = reinterpret_cast<const e3_fp16_t *>(
                expert_bank + expert_base + h24d::kDownMetaOffset) +
                static_cast<size_t>(output_row) * kDownGroups * 2 + group * 2;
            const int dot = low_acc[value] + 16 * high_acc[value];
            values[value] += opus::fp16_to_fp32(weight_meta[0]) *
                    (activation_d * static_cast<float>(dot)) +
                opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
        }
    }

    const int route = routes[lane16];
    const int token = route / kTopK;
    const float weighted_route = route_weights[route];
#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int local_row = wave * 16 + 2 * value + lane_half;
        const int output_row = static_cast<int>(row_tile) * kWmmaOutputTile + local_row;
        atomicAdd(output + static_cast<size_t>(token) * kModelWidth + output_row,
            weighted_route * values[value]);
    }
}
#endif

// One wave owns one expert descriptor and one N16 output tile.  Counts from
// four through sixteen are represented by zero-padding inactive WMMA columns;
// every global route read/write remains predicated by lane16 < count.
extern "C" __global__ __launch_bounds__(32)
void h59_grouped_gate_up_wmma_b4_16_n16(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const e3_fp16_t * input_meta,
        const std::uint8_t * input_low,
        const std::uint8_t * input_high,
        float * middle_f32) {
    constexpr unsigned kOutputTiles = kExpertWidth / kWmmaOutputTile;
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor = linear / kOutputTiles;
    const unsigned output_tile = linear - descriptor * kOutputTiles;
    if (descriptor >= max_descriptors ||
        descriptor >= static_cast<unsigned>(header[4])) return;

    const int lane = opus::thread_id_x();
    const int lane16 = lane & 15;
    const int lane_half = lane >> 4;
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count < static_cast<int>(kWmmaMinRoutes) ||
        count > static_cast<int>(kWmmaRouteTile)) return;

    const bool active = lane16 < count;
    int route = 0;
    if (lane_half == 0 && active) route = route_indices[start + lane16];
    route = opus::shfl(route, lane16, 32);
    const int token = route / kTopK;
    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;
    const int fragment_row = static_cast<int>(output_tile) * kWmmaOutputTile + lane16;

    // Padding the second dimension removes the power-of-two bank stride while
    // retaining the exact H47 even/odd F32 association.
    __shared__ float partial[kWmmaOutputTile][kWmmaRouteTile + 1];
    __shared__ float gate_tile[kWmmaOutputTile][kWmmaRouteTile + 1];

    // Gate projection: even G128 groups, then odd G128 groups.
    for (int parity = 0; parity < 2; ++parity) {
        float values[8] = {};
#pragma clang loop unroll(disable)
        for (unsigned group = static_cast<unsigned>(parity);
             group < kGateGroups; group += 2) {
            float activation_d = 0.0f;
            float activation_sum = 0.0f;
            if (lane_half == 0 && active) {
                const e3_fp16_t * xm = input_meta +
                    static_cast<size_t>(token) * kGateGroups * 2 + group * 2;
                activation_d = opus::fp16_to_fp32(xm[0]);
                activation_sum = opus::fp16_to_fp32(xm[1]);
            }
            activation_d = opus::shfl(activation_d, lane16, 32);
            activation_sum = opus::shfl(activation_sum, lane16, 32);

            e3_i32x8 low_acc = {};
            e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
            for (int tile = 0; tile < static_cast<int>(kGateGroup / 16); ++tile) {
                int weight0 = 0;
                int weight1 = 0;
                if (lane_half == 0) {
                    const std::uint8_t * code = expert_bank + expert_base +
                        h24d::kGateCodeOffset +
                        static_cast<size_t>(fragment_row) * (kModelWidth / 2) +
                        group * (kGateGroup / 2) + tile * 8;
                    const std::uint32_t * words =
                        reinterpret_cast<const std::uint32_t *>(code);
                    weight0 = static_cast<int>(words[0]);
                    weight1 = static_cast<int>(words[1]);
                }
                weight0 = opus::shfl(weight0, lane16, 32);
                weight1 = opus::shfl(weight1, lane16, 32);

                int low0 = 0;
                int low1 = 0;
                int high0 = 0;
                int high1 = 0;
                if (lane_half == 0 && active) {
                    const size_t source_word = static_cast<size_t>(token) *
                            (kModelWidth / 8) +
                        group * (kGateGroup / 8) + tile * 2;
                    low0 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_low)[source_word]);
                    low1 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_low)[source_word + 1]);
                    high0 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_high)[source_word]);
                    high1 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_high)[source_word + 1]);
                }
                low0 = opus::shfl(low0, lane16, 32);
                low1 = opus::shfl(low1, lane16, 32);
                high0 = opus::shfl(high0, lane16, 32);
                high1 = opus::shfl(high1, lane16, 32);

                e3_i32x2 weight_fragment;
                e3_i32x2 low_fragment;
                e3_i32x2 high_fragment;
                weight_fragment[0] = weight0;
                weight_fragment[1] = weight1;
                low_fragment[0] = low0;
                low_fragment[1] = low1;
                high_fragment[0] = high0;
                high_fragment[1] = high1;
                low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, false, low_fragment, low_acc, false);
                high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, true, high_fragment, high_acc, false);
            }

#pragma unroll
            for (int value = 0; value < 8; ++value) {
                if (active) {
                    const int local_row = 2 * value + lane_half;
                    const int output_row = static_cast<int>(output_tile) *
                        kWmmaOutputTile + local_row;
                    const e3_fp16_t * weight_meta =
                        reinterpret_cast<const e3_fp16_t *>(expert_bank +
                            expert_base + h24d::kGateMetaOffset) +
                        static_cast<size_t>(output_row) * kGateGroups * 2 + group * 2;
                    const int dot = low_acc[value] + 16 * high_acc[value];
                    values[value] += opus::fp16_to_fp32(weight_meta[0]) *
                            (activation_d * static_cast<float>(dot)) +
                        opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
                }
            }
        }

#pragma unroll
        for (int value = 0; value < 8; ++value) {
            const int local_row = 2 * value + lane_half;
            if (parity == 0) {
                partial[local_row][lane16] = values[value];
            } else {
                gate_tile[local_row][lane16] =
                    partial[local_row][lane16] + values[value];
            }
        }
        opus::sync_threads();
    }

    // Up projection uses the same association and reuses the partial tile.
    for (int parity = 0; parity < 2; ++parity) {
        float values[8] = {};
#pragma clang loop unroll(disable)
        for (unsigned group = static_cast<unsigned>(parity);
             group < kGateGroups; group += 2) {
            float activation_d = 0.0f;
            float activation_sum = 0.0f;
            if (lane_half == 0 && active) {
                const e3_fp16_t * xm = input_meta +
                    static_cast<size_t>(token) * kGateGroups * 2 + group * 2;
                activation_d = opus::fp16_to_fp32(xm[0]);
                activation_sum = opus::fp16_to_fp32(xm[1]);
            }
            activation_d = opus::shfl(activation_d, lane16, 32);
            activation_sum = opus::shfl(activation_sum, lane16, 32);

            e3_i32x8 low_acc = {};
            e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
            for (int tile = 0; tile < static_cast<int>(kGateGroup / 16); ++tile) {
                int weight0 = 0;
                int weight1 = 0;
                if (lane_half == 0) {
                    const std::uint8_t * code = expert_bank + expert_base +
                        h24d::kGateCodeOffset +
                        static_cast<size_t>(kExpertWidth + fragment_row) *
                            (kModelWidth / 2) +
                        group * (kGateGroup / 2) + tile * 8;
                    const std::uint32_t * words =
                        reinterpret_cast<const std::uint32_t *>(code);
                    weight0 = static_cast<int>(words[0]);
                    weight1 = static_cast<int>(words[1]);
                }
                weight0 = opus::shfl(weight0, lane16, 32);
                weight1 = opus::shfl(weight1, lane16, 32);

                int low0 = 0;
                int low1 = 0;
                int high0 = 0;
                int high1 = 0;
                if (lane_half == 0 && active) {
                    const size_t source_word = static_cast<size_t>(token) *
                            (kModelWidth / 8) +
                        group * (kGateGroup / 8) + tile * 2;
                    low0 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_low)[source_word]);
                    low1 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_low)[source_word + 1]);
                    high0 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_high)[source_word]);
                    high1 = static_cast<int>(
                        reinterpret_cast<const std::uint32_t *>(input_high)[source_word + 1]);
                }
                low0 = opus::shfl(low0, lane16, 32);
                low1 = opus::shfl(low1, lane16, 32);
                high0 = opus::shfl(high0, lane16, 32);
                high1 = opus::shfl(high1, lane16, 32);

                e3_i32x2 weight_fragment;
                e3_i32x2 low_fragment;
                e3_i32x2 high_fragment;
                weight_fragment[0] = weight0;
                weight_fragment[1] = weight1;
                low_fragment[0] = low0;
                low_fragment[1] = low1;
                high_fragment[0] = high0;
                high_fragment[1] = high1;
                low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, false, low_fragment, low_acc, false);
                high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    false, weight_fragment, true, high_fragment, high_acc, false);
            }

#pragma unroll
            for (int value = 0; value < 8; ++value) {
                if (active) {
                    const int local_row = 2 * value + lane_half;
                    const int output_row = static_cast<int>(output_tile) *
                        kWmmaOutputTile + local_row;
                    const e3_fp16_t * weight_meta =
                        reinterpret_cast<const e3_fp16_t *>(expert_bank +
                            expert_base + h24d::kGateMetaOffset) +
                        static_cast<size_t>(kExpertWidth + output_row) *
                            kGateGroups * 2 + group * 2;
                    const int dot = low_acc[value] + 16 * high_acc[value];
                    values[value] += opus::fp16_to_fp32(weight_meta[0]) *
                            (activation_d * static_cast<float>(dot)) +
                        opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
                }
            }
        }

#pragma unroll
        for (int value = 0; value < 8; ++value) {
            const int local_row = 2 * value + lane_half;
            if (parity == 0) {
                partial[local_row][lane16] = values[value];
            } else if (active) {
                const int output_row = static_cast<int>(output_tile) *
                    kWmmaOutputTile + local_row;
                const float gate = gate_tile[local_row][lane16];
                const float up = partial[local_row][lane16] + values[value];
                middle_f32[static_cast<size_t>(route) * kExpertWidth + output_row] =
                    h24d::silu(gate) * up;
            }
        }
        opus::sync_threads();
    }
}

extern "C" __global__ __launch_bounds__(32)
void h59_quantize_middle_q8_d64(
        const int * logical_expert_ids,
        const int * counts,
        const float * middle_f32,
        e3_i8_t * middle_q8,
        e3_fp16_t * middle_meta,
        std::uint8_t * middle_low,
        std::uint8_t * middle_high,
        unsigned routes) {
    const unsigned linear = opus::block_id_x();
    const unsigned route = linear / kDownGroups;
    const unsigned down_group = linear - route * kDownGroups;
    if (route >= routes) return;
    const int expert = logical_expert_ids[route];
    if (expert < 0 || expert >= static_cast<int>(kExperts) ||
        counts[expert] < static_cast<int>(kWmmaMinRoutes)) return;

    const int lane = opus::thread_id_x();
    const int channel0 = lane * 2;
    const int channel1 = channel0 + 1;
    const size_t source_base = static_cast<size_t>(route) * kExpertWidth +
        down_group * kDownGroup;
    const float v0 = middle_f32[source_base + channel0];
    const float v1 = middle_f32[source_base + channel1];
    float max_abs = opus::max(__builtin_fabsf(v0), __builtin_fabsf(v1));
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const float peer = opus::shfl(max_abs, lane + delta, 32);
        if (lane < delta) max_abs = opus::max(max_abs, peer);
    }
    float d = max_abs / 127.0f;
    if (lane == 0) d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
    d = opus::shfl(d, 0, 32);
    int q0 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(v0 / d));
    int q1 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(v1 / d));
    q0 = opus::max(-127, opus::min(127, q0));
    q1 = opus::max(-127, opus::min(127, q1));
    middle_q8[source_base + channel0] = static_cast<e3_i8_t>(q0);
    middle_q8[source_base + channel1] = static_cast<e3_i8_t>(q1);

    const int lo0 = q0 & 0x0f;
    const int lo1 = q1 & 0x0f;
    const int hi0 = (q0 - lo0) / 16;
    const int hi1 = (q1 - lo1) / 16;
    const size_t digit_base = static_cast<size_t>(route) * (kExpertWidth / 2) +
        down_group * (kDownGroup / 2) + lane;
    middle_low[digit_base] = static_cast<std::uint8_t>(lo0 | (lo1 << 4));
    middle_high[digit_base] = static_cast<std::uint8_t>(
        (hi0 & 0x0f) | ((hi1 & 0x0f) << 4));

    int sum = q0 + q1;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int peer = opus::shfl(sum, lane + delta, 32);
        if (lane < delta) sum += peer;
    }
    if (lane == 0) {
        const size_t meta_base = static_cast<size_t>(route) * kDownGroups * 2 +
            down_group * 2;
        middle_meta[meta_base] = opus::fp32_to_fp16(d);
        middle_meta[meta_base + 1] = opus::fp32_to_fp16(
            d * static_cast<float>(sum));
    }
}

extern "C" __global__ __launch_bounds__(32)
void h59_grouped_down_wmma_b4_16_n16(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const float * route_weights,
        const e3_fp16_t * middle_meta,
        const std::uint8_t * middle_low,
        const std::uint8_t * middle_high,
        float * output) {
    constexpr unsigned kRowTiles = kModelWidth / kWmmaOutputTile;
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor = linear / kRowTiles;
    const unsigned row_tile = linear - descriptor * kRowTiles;
    if (descriptor >= max_descriptors ||
        descriptor >= static_cast<unsigned>(header[4])) return;

    const int lane = opus::thread_id_x();
    const int lane16 = lane & 15;
    const int lane_half = lane >> 4;
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count < static_cast<int>(kWmmaMinRoutes) ||
        count > static_cast<int>(kWmmaRouteTile)) return;

    const bool active = lane16 < count;
    int route = 0;
    if (lane_half == 0 && active) route = route_indices[start + lane16];
    route = opus::shfl(route, lane16, 32);
    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;
    const int fragment_row = static_cast<int>(row_tile) * kWmmaOutputTile + lane16;
    float values[8] = {};

#pragma clang loop unroll(disable)
    for (unsigned group = 0; group < kDownGroups; ++group) {
        float activation_d = 0.0f;
        float activation_sum = 0.0f;
        if (lane_half == 0 && active) {
            const e3_fp16_t * xm = middle_meta +
                static_cast<size_t>(route) * kDownGroups * 2 + group * 2;
            activation_d = opus::fp16_to_fp32(xm[0]);
            activation_sum = opus::fp16_to_fp32(xm[1]);
        }
        activation_d = opus::shfl(activation_d, lane16, 32);
        activation_sum = opus::shfl(activation_sum, lane16, 32);

        e3_i32x8 low_acc = {};
        e3_i32x8 high_acc = {};
#pragma clang loop unroll(disable)
        for (int tile = 0; tile < static_cast<int>(kDownGroup / 16); ++tile) {
            int weight0 = 0;
            int weight1 = 0;
            if (lane_half == 0) {
                const std::uint8_t * code = expert_bank + expert_base +
                    h24d::kDownCodeOffset +
                    static_cast<size_t>(fragment_row) * (kExpertWidth / 2) +
                    group * (kDownGroup / 2) + tile * 8;
                const std::uint32_t * words =
                    reinterpret_cast<const std::uint32_t *>(code);
                weight0 = static_cast<int>(words[0]);
                weight1 = static_cast<int>(words[1]);
            }
            weight0 = opus::shfl(weight0, lane16, 32);
            weight1 = opus::shfl(weight1, lane16, 32);

            int low0 = 0;
            int low1 = 0;
            int high0 = 0;
            int high1 = 0;
            if (lane_half == 0 && active) {
                const size_t source_word = static_cast<size_t>(route) *
                        (kExpertWidth / 8) +
                    group * (kDownGroup / 8) + tile * 2;
                low0 = static_cast<int>(
                    reinterpret_cast<const std::uint32_t *>(middle_low)[source_word]);
                low1 = static_cast<int>(
                    reinterpret_cast<const std::uint32_t *>(middle_low)[source_word + 1]);
                high0 = static_cast<int>(
                    reinterpret_cast<const std::uint32_t *>(middle_high)[source_word]);
                high1 = static_cast<int>(
                    reinterpret_cast<const std::uint32_t *>(middle_high)[source_word + 1]);
            }
            low0 = opus::shfl(low0, lane16, 32);
            low1 = opus::shfl(low1, lane16, 32);
            high0 = opus::shfl(high0, lane16, 32);
            high1 = opus::shfl(high1, lane16, 32);

            e3_i32x2 weight_fragment;
            e3_i32x2 low_fragment;
            e3_i32x2 high_fragment;
            weight_fragment[0] = weight0;
            weight_fragment[1] = weight1;
            low_fragment[0] = low0;
            low_fragment[1] = low1;
            high_fragment[0] = high0;
            high_fragment[1] = high1;
            low_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, weight_fragment, false, low_fragment, low_acc, false);
            high_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, weight_fragment, true, high_fragment, high_acc, false);
        }

#pragma unroll
        for (int value = 0; value < 8; ++value) {
            if (active) {
                const int local_row = 2 * value + lane_half;
                const int output_row = static_cast<int>(row_tile) *
                    kWmmaOutputTile + local_row;
                const e3_fp16_t * weight_meta =
                    reinterpret_cast<const e3_fp16_t *>(expert_bank + expert_base +
                        h24d::kDownMetaOffset) +
                    static_cast<size_t>(output_row) * kDownGroups * 2 + group * 2;
                const int dot = low_acc[value] + 16 * high_acc[value];
                values[value] += opus::fp16_to_fp32(weight_meta[0]) *
                        (activation_d * static_cast<float>(dot)) +
                    opus::fp16_to_fp32(weight_meta[1]) * activation_sum;
            }
        }
    }

    if (active) {
        const int token = route / kTopK;
        const float weighted_route = route_weights[route];
#pragma unroll
        for (int value = 0; value < 8; ++value) {
            const int local_row = 2 * value + lane_half;
            const int output_row = static_cast<int>(row_tile) *
                kWmmaOutputTile + local_row;
            atomicAdd(output + static_cast<size_t>(token) * kModelWidth + output_row,
                weighted_route * values[value]);
        }
    }
}

extern "C" __global__ __launch_bounds__(32)
void h47_grouped_gate_up_q8_d64(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const e3_i8_t * input_q8,
        const e3_fp16_t * input_meta,
        e3_i8_t * middle_q8,
        e3_fp16_t * middle_meta) {
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor_local = linear / kDownGroups;
    const unsigned down_group = linear - descriptor_local * kDownGroups;
    if (descriptor_local >= max_descriptors ||
        descriptor_local >= static_cast<unsigned>(header[5])) return;
    const unsigned descriptor = static_cast<unsigned>(header[4]) + descriptor_local;
    const int lane = opus::thread_id_x();
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count > static_cast<int>(kRouteTile)) return;
    __shared__ int routes[kRouteTile];
    __shared__ float middle[kRouteTile][64];
    __shared__ float scales[kRouteTile];
    if (lane < count) routes[lane] = route_indices[start + lane];
    opus::sync_threads();
    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int channel = static_cast<int>(down_group) * 64 + half * kOutputTile + lane;
        const int gate_row = channel;
        const int up_row = kExpertWidth + channel;
        const std::uint8_t * gate_code = expert_bank + expert_base +
            h24d::kGateCodeOffset + static_cast<size_t>(gate_row) * (kModelWidth / 2);
        const std::uint8_t * up_code = expert_bank + expert_base +
            h24d::kGateCodeOffset + static_cast<size_t>(up_row) * (kModelWidth / 2);
        const e3_fp16_t * gate_meta = reinterpret_cast<const e3_fp16_t *>(
            expert_bank + expert_base + h24d::kGateMetaOffset) + gate_row * kGateGroups * 2;
        const e3_fp16_t * up_meta = reinterpret_cast<const e3_fp16_t *>(
            expert_bank + expert_base + h24d::kGateMetaOffset) + up_row * kGateGroups * 2;
        float gate_even[kRouteTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        float gate_odd[kRouteTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_even[kRouteTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_odd[kRouteTile] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
        for (unsigned group = 0; group < kGateGroups; ++group) {
            int gate_dot[kRouteTile] = {0, 0, 0, 0};
            int up_dot[kRouteTile] = {0, 0, 0, 0};
#pragma unroll
            for (unsigned k = 0; k < kGateGroup; k += 8) {
                const e3_i8_t * q[kRouteTile];
#pragma unroll
                for (int b = 0; b < kRouteTile; ++b) {
                    const int token = b < count ? routes[b] / kTopK : 0;
                    q[b] = input_q8 + static_cast<size_t>(token) * kModelWidth +
                        group * kGateGroup + k;
                }
                h47_dot8_b4(gate_code + (group * kGateGroup + k) / 2,
                    q, count, gate_dot);
                h47_dot8_b4(up_code + (group * kGateGroup + k) / 2,
                    q, count, up_dot);
            }
            const float gate_scale = opus::fp16_to_fp32(gate_meta[group * 2]);
            const float gate_offset = opus::fp16_to_fp32(gate_meta[group * 2 + 1]);
            const float up_scale = opus::fp16_to_fp32(up_meta[group * 2]);
            const float up_offset = opus::fp16_to_fp32(up_meta[group * 2 + 1]);
#pragma unroll
            for (int b = 0; b < kRouteTile; ++b) {
                if (b < count) {
                    const int token = routes[b] / kTopK;
                    const e3_fp16_t * xm = input_meta +
                        static_cast<size_t>(token) * kGateGroups * 2 + group * 2;
                    const float d = opus::fp16_to_fp32(xm[0]);
                    const float sum = opus::fp16_to_fp32(xm[1]);
                    float * gate_partial = (group & 1) ? gate_odd : gate_even;
                    float * up_partial = (group & 1) ? up_odd : up_even;
                    gate_partial[b] += gate_scale * (d * static_cast<float>(gate_dot[b])) + gate_offset * sum;
                    up_partial[b] += up_scale * (d * static_cast<float>(up_dot[b])) + up_offset * sum;
                }
            }
        }
#pragma unroll
        for (int b = 0; b < kRouteTile; ++b) {
            if (b < count) {
                const float gate = gate_even[b] + gate_odd[b];
                const float up = up_even[b] + up_odd[b];
                middle[b][half * kOutputTile + lane] = h24d::silu(gate) * up;
            }
        }
    }
    opus::sync_threads();

#pragma unroll
    for (int b = 0; b < kRouteTile; ++b) {
        if (b < count) {
            float max_abs = opus::max(
                __builtin_fabsf(middle[b][lane]),
                __builtin_fabsf(middle[b][kOutputTile + lane]));
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                const float peer = opus::shfl(max_abs, lane + delta, 32);
                if (lane < delta) max_abs = opus::max(max_abs, peer);
            }
            float d = max_abs / 127.0f;
            if (lane == 0) {
                d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
                scales[b] = d;
            }
            d = opus::shfl(d, 0, 32);
            int q0 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(middle[b][lane] / d));
            int q1 = d == 0.0f ? 0 : static_cast<int>(__builtin_rintf(middle[b][kOutputTile + lane] / d));
            q0 = opus::max(-127, opus::min(127, q0));
            q1 = opus::max(-127, opus::min(127, q1));
            const int route = routes[b];
            const size_t output_base = static_cast<size_t>(route) * kExpertWidth + down_group * 64;
            middle_q8[output_base + lane] = static_cast<e3_i8_t>(q0);
            middle_q8[output_base + kOutputTile + lane] = static_cast<e3_i8_t>(q1);
            int sum = q0 + q1;
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                const int peer = opus::shfl(sum, lane + delta, 32);
                if (lane < delta) sum += peer;
            }
            if (lane == 0) {
                const size_t meta_base = static_cast<size_t>(route) * kDownGroups * 2 + down_group * 2;
                middle_meta[meta_base] = opus::fp32_to_fp16(scales[b]);
                middle_meta[meta_base + 1] = opus::fp32_to_fp16(scales[b] * static_cast<float>(sum));
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(32)
void h47_grouped_down_b4_n32(
        const std::uint8_t * expert_bank,
        const int * header,
        const int * route_indices,
        const int * descriptors,
        unsigned max_descriptors,
        const float * route_weights,
        const e3_i8_t * middle_q8,
        const e3_fp16_t * middle_meta,
        float * output) {
    constexpr unsigned kRowTiles = kModelWidth / kOutputTile;
    const unsigned linear = opus::block_id_x();
    const unsigned descriptor_local = linear / kRowTiles;
    const unsigned row_tile = linear - descriptor_local * kRowTiles;
    if (descriptor_local >= max_descriptors ||
        descriptor_local >= static_cast<unsigned>(header[5])) return;
    const unsigned descriptor = static_cast<unsigned>(header[4]) + descriptor_local;
    const int lane = opus::thread_id_x();
    const int row = row_tile * kOutputTile + lane;
    const int start = descriptors[descriptor * 2];
    const int packed_descriptor = descriptors[descriptor * 2 + 1];
    const int expert = packed_descriptor & 0xffff;
    const int count = packed_descriptor >> 16;
    if (count > static_cast<int>(kRouteTile)) return;
    int routes[kRouteTile] = {0, 0, 0, 0};
#pragma unroll
    for (int b = 0; b < kRouteTile; ++b) {
        if (b < count) routes[b] = route_indices[start + b];
    }
    const size_t expert_base = static_cast<size_t>(expert) * h24d::kExpertStride;
    const std::uint8_t * code = expert_bank + expert_base + h24d::kDownCodeOffset +
        static_cast<size_t>(row) * (kExpertWidth / 2);
    const e3_fp16_t * weight_meta = reinterpret_cast<const e3_fp16_t *>(
        expert_bank + expert_base + h24d::kDownMetaOffset) + row * kDownGroups * 2;
    float values[kRouteTile] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
    for (unsigned group = 0; group < kDownGroups; ++group) {
        int dot[kRouteTile] = {0, 0, 0, 0};
#pragma unroll
        for (unsigned k = 0; k < kDownGroup; k += 8) {
            const e3_i8_t * q[kRouteTile];
#pragma unroll
            for (int b = 0; b < kRouteTile; ++b) {
                q[b] = middle_q8 + static_cast<size_t>(routes[b]) * kExpertWidth +
                    group * kDownGroup + k;
            }
            h47_dot8_b4(code + (group * kDownGroup + k) / 2, q, count, dot);
        }
        const float scale = opus::fp16_to_fp32(weight_meta[group * 2]);
        const float offset = opus::fp16_to_fp32(weight_meta[group * 2 + 1]);
#pragma unroll
        for (int b = 0; b < kRouteTile; ++b) {
            if (b < count) {
                const e3_fp16_t * xm = middle_meta +
                    static_cast<size_t>(routes[b]) * kDownGroups * 2 + group * 2;
                const float d = opus::fp16_to_fp32(xm[0]);
                const float sum = opus::fp16_to_fp32(xm[1]);
                values[b] += scale * (d * static_cast<float>(dot[b])) + offset * sum;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < kRouteTile; ++b) {
        if (b < count) {
            const int route = routes[b];
            const int token = route / kTopK;
            atomicAdd(output + static_cast<size_t>(token) * kModelWidth + row,
                route_weights[route] * values[b]);
        }
    }
}
#else
extern "C" __global__ __launch_bounds__(32)
void h36_quantize_input_q8_g128(
        const float *,
        e3_i8_t *,
        e3_fp16_t *) {}
extern "C" __global__ void h47_zero_output(float *, unsigned) {}
extern "C" __global__ void h47_quantize_input_q8_g128_batched(
    const float *, e3_i8_t *, e3_fp16_t *, std::uint8_t *, std::uint8_t *, unsigned) {}
extern "C" __global__ void h47_plan_routes(
    const int *, unsigned, int *, int *, int *, int *, int *, int *) {}
extern "C" __global__ void h59_grouped_gate_up_wmma_b4_16_n16(
    const std::uint8_t *, const int *, const int *, const int *, unsigned,
    const e3_fp16_t *, const std::uint8_t *, const std::uint8_t *, float *) {}
extern "C" __global__ void h59_quantize_middle_q8_d64(
    const int *, const int *, const float *, e3_i8_t *, e3_fp16_t *,
    std::uint8_t *, std::uint8_t *, unsigned) {}
extern "C" __global__ void h59_grouped_down_wmma_b4_16_n16(
    const std::uint8_t *, const int *, const int *, const int *, unsigned,
    const float *, const e3_fp16_t *, const std::uint8_t *, const std::uint8_t *,
    float *) {}
extern "C" __global__ void h47_grouped_gate_up_q8_d64(
    const std::uint8_t *, const int *, const int *, const int *, unsigned,
    const e3_i8_t *, const e3_fp16_t *, e3_i8_t *, e3_fp16_t *) {}
extern "C" __global__ void h47_grouped_down_b4_n32(
    const std::uint8_t *, const int *, const int *, const int *, unsigned,
    const float *, const e3_i8_t *, const e3_fp16_t *, float *) {}
#endif

} // namespace

void ggml_cuda_e3_qr05_launch(
        const float * input_f32,
        const std::uint8_t * resident_layer_bank,
        unsigned bank_bytes,
        const std::int32_t * logical_expert_ids,
        const float * route_weights,
        float * workspace_f32,
        unsigned n_tokens,
        void * stream_ptr) {
    auto * workspace = reinterpret_cast<unsigned char *>(workspace_f32);
    const hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

    // Preserve H36/H30/H24 M=1 byte offsets, launch geometry and arithmetic.
    if (n_tokens == 1) {
        auto * output_f32 = reinterpret_cast<float *>(workspace);
        auto * input_q8 = reinterpret_cast<e3_i8_t *>(workspace + kOutputBytes);
        auto * input_meta = reinterpret_cast<e3_fp16_t *>(
            workspace + kOutputBytes + kInputQ8Bytes);
        const unsigned middle_q8_offset = kOutputBytes + kInputQ8Bytes + kInputMetaBytes;
        auto * middle_q8 = reinterpret_cast<e3_i8_t *>(workspace + middle_q8_offset);
        auto * middle_meta = reinterpret_cast<e3_fp16_t *>(
            workspace + middle_q8_offset + kMiddleQ8Bytes);

        hipLaunchKernelGGL(h36_quantize_input_q8_g128,
            dim3(kGateGroups), dim3(32), 0, stream,
            input_f32, input_q8, input_meta);
        hipLaunchKernelGGL(h30_e3_gate_up_swiglu_q8_d64,
            dim3(kTopK * kDownGroups), dim3(256), 0, stream,
            resident_layer_bank, bank_bytes, logical_expert_ids,
            input_q8, input_meta, middle_q8, middle_meta);
        hipLaunchKernelGGL(h24_down_top10,
            dim3((kModelWidth + 255) / 256), dim3(256), 0, stream,
            resident_layer_bank, bank_bytes, logical_expert_ids, route_weights,
            middle_q8, middle_meta, output_f32);
        return;
    }

    auto * output_f32 = reinterpret_cast<float *>(workspace);
    const size_t output_bytes = static_cast<size_t>(n_tokens) * kOutputBytes;

    // Very-large shapes use the unchanged packet-native token loop on the
    // existing stream and reuse one scratch slice.
    if (n_tokens > 8192) {
        auto * input_q8 = reinterpret_cast<e3_i8_t *>(workspace + output_bytes);
        auto * input_meta = reinterpret_cast<e3_fp16_t *>(workspace + output_bytes + kInputQ8Bytes);
        const size_t middle_q8_offset = output_bytes + kInputQ8Bytes + kInputMetaBytes;
        auto * middle_q8 = reinterpret_cast<e3_i8_t *>(workspace + middle_q8_offset);
        auto * middle_meta = reinterpret_cast<e3_fp16_t *>(workspace + middle_q8_offset + kMiddleQ8Bytes);
        for (unsigned token = 0; token < n_tokens; ++token) {
            hipLaunchKernelGGL(h36_quantize_input_q8_g128,
                dim3(kGateGroups), dim3(32), 0, stream,
                input_f32 + static_cast<size_t>(token) * kModelWidth,
                input_q8, input_meta);
            hipLaunchKernelGGL(h30_e3_gate_up_swiglu_q8_d64,
                dim3(kTopK * kDownGroups), dim3(256), 0, stream,
                resident_layer_bank, bank_bytes, logical_expert_ids + token * kTopK,
                input_q8, input_meta, middle_q8, middle_meta);
            hipLaunchKernelGGL(h24_down_top10,
                dim3((kModelWidth + 255) / 256), dim3(256), 0, stream,
                resident_layer_bank, bank_bytes, logical_expert_ids + token * kTopK,
                route_weights + token * kTopK, middle_q8, middle_meta,
                output_f32 + static_cast<size_t>(token) * kModelWidth);
        }
        return;
    }

    const size_t routes = static_cast<size_t>(n_tokens) * kTopK;
    const size_t max_descriptors = (routes + kRouteTile - 1) / kRouteTile + kExperts;
    const size_t max_wmma_by_routes = (routes + kWmmaMinRoutes - 1) / kWmmaMinRoutes;
    const size_t max_wmma_by_experts =
        (routes + kWmmaRouteTile - 1) / kWmmaRouteTile + kExperts;
    const size_t max_wmma_descriptors = max_wmma_by_routes < max_wmma_by_experts ?
        max_wmma_by_routes : max_wmma_by_experts;
    const size_t max_vector_descriptors = routes < kExperts ? routes : kExperts;
    auto * input_q8 = reinterpret_cast<e3_i8_t *>(workspace + output_bytes);
    auto * input_meta = reinterpret_cast<e3_fp16_t *>(workspace + output_bytes +
        static_cast<size_t>(n_tokens) * kInputQ8Bytes);
    // Keep every H47 Q8/meta offset stable.  The four transient digit planes
    // and H59's route-major F32 middle live after the established middle
    // representation.
    auto * middle_q8 = reinterpret_cast<e3_i8_t *>(
        reinterpret_cast<unsigned char *>(input_meta) +
        static_cast<size_t>(n_tokens) * kInputMetaBytes);
    auto * middle_meta = reinterpret_cast<e3_fp16_t *>(
        reinterpret_cast<unsigned char *>(middle_q8) + routes * kExpertWidth);
    auto * input_low = reinterpret_cast<std::uint8_t *>(
        reinterpret_cast<unsigned char *>(middle_meta) +
        routes * kDownGroups * 2 * sizeof(e3_fp16_t));
    auto * input_high = input_low + static_cast<size_t>(n_tokens) * kInputDigitBytes;
    auto * middle_low = input_high + static_cast<size_t>(n_tokens) * kInputDigitBytes;
    auto * middle_high = middle_low + routes * (kExpertWidth / 2);
    auto * middle_f32 = reinterpret_cast<float *>(
        middle_high + routes * (kExpertWidth / 2));
    auto * plan = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(middle_f32) +
            routes * kExpertWidth * sizeof(float));
    int * header = plan;
    int * counts = header + 6;
    int * starts = counts + kExperts;
    int * cursors = starts + kExperts + 1;
    int * route_indices = cursors + kExperts;
    int * descriptors = route_indices + routes;

    hipLaunchKernelGGL(h47_zero_output,
        dim3((static_cast<size_t>(n_tokens) * kModelWidth + 31) / 32), dim3(32), 0, stream,
        output_f32, n_tokens);
    hipLaunchKernelGGL(h47_quantize_input_q8_g128_batched,
        dim3(static_cast<size_t>(n_tokens) * kGateGroups), dim3(32), 0, stream,
        input_f32, input_q8, input_meta, input_low, input_high, n_tokens);
    hipLaunchKernelGGL(h47_plan_routes,
        dim3(1), dim3(1), 0, stream,
        logical_expert_ids, n_tokens, header, counts, starts, cursors,
        route_indices, descriptors);
    hipLaunchKernelGGL(h59_grouped_gate_up_wmma_b4_16_n16,
        dim3(max_wmma_descriptors * (kExpertWidth / kWmmaOutputTile)),
        dim3(32), 0, stream,
        resident_layer_bank, header, route_indices, descriptors,
        static_cast<unsigned>(max_wmma_descriptors), input_meta, input_low,
        input_high, middle_f32);
    hipLaunchKernelGGL(h47_grouped_gate_up_q8_d64,
        dim3(max_vector_descriptors * kDownGroups), dim3(32), 0, stream,
        resident_layer_bank, header, route_indices, descriptors,
        static_cast<unsigned>(max_vector_descriptors), input_q8, input_meta,
        middle_q8, middle_meta);
    hipLaunchKernelGGL(h59_quantize_middle_q8_d64,
        dim3(routes * kDownGroups), dim3(32), 0, stream,
        logical_expert_ids, counts, middle_f32, middle_q8, middle_meta,
        middle_low, middle_high, static_cast<unsigned>(routes));
    hipLaunchKernelGGL(h59_grouped_down_wmma_b4_16_n16,
        dim3(max_wmma_descriptors * (kModelWidth / kWmmaOutputTile)),
        dim3(32), 0, stream,
        resident_layer_bank, header, route_indices, descriptors,
        static_cast<unsigned>(max_wmma_descriptors), route_weights, middle_meta,
        middle_low, middle_high, output_f32);
    hipLaunchKernelGGL(h47_grouped_down_b4_n32,
        dim3(max_vector_descriptors * (kModelWidth / kOutputTile)), dim3(32), 0, stream,
        resident_layer_bank, header, route_indices, descriptors,
        static_cast<unsigned>(max_vector_descriptors), route_weights, middle_q8,
        middle_meta, output_f32);
}

#else

void ggml_cuda_e3_qr05_launch(
    const float *, const std::uint8_t *, unsigned, const std::int32_t *,
        const float *, float *, unsigned, void *) {
    // The scheduler never admits GGML_OP_E3_QR05 on non-HIP backends.
}

#endif

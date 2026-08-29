// H24: one-wave-per-output gate/up mappings for the H22 finalists.
// Compile as E2 (G128/D128), E3 (G128/D64), or T500 (G32/D32).

#ifndef H24_GATE_GROUP
#define H24_GATE_GROUP 128
#endif
#ifndef H24_DOWN_GROUP
#define H24_DOWN_GROUP 64
#endif
#ifndef H24_SHAPE_KIND
#define H24_SHAPE_KIND 3
#endif

#define H24_MODEL_WIDTH 2560
#define H24_EXPERT_WIDTH 640
#define H24_TOP_K 10
#define H24_BLOCK_SIZE 256

#ifdef __HIP_DEVICE_COMPILE__

#include "opus/hip_minimal.hpp"
#include "opus/opus.hpp"

namespace h24d {

using opus::fp16_t;
using opus::i8_t;
using opus::u8_t;
using opus::u32_t;

constexpr int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr int kGateGroups = H24_MODEL_WIDTH / H24_GATE_GROUP;
constexpr int kDownGroups = H24_EXPERT_WIDTH / H24_DOWN_GROUP;
constexpr int kGateRows = 2 * H24_EXPERT_WIDTH;
constexpr int kGateCodeBytes = kGateRows * H24_MODEL_WIDTH / 2;
constexpr int kGateMetaBytes = kGateRows * kGateGroups * 4;
constexpr int kDownCodeBytes = H24_MODEL_WIDTH * H24_EXPERT_WIDTH / 2;
constexpr int kDownMetaBytes = H24_MODEL_WIDTH * kDownGroups * 4;
constexpr int kGateCodeOffset = 0;
constexpr int kGateMetaOffset = align_up(kGateCodeBytes, 256);
constexpr int kDownCodeOffset = align_up(kGateMetaOffset + kGateMetaBytes, 256);
constexpr int kDownMetaOffset = align_up(kDownCodeOffset + kDownCodeBytes, 256);
constexpr int kExpertStride = align_up(kDownMetaOffset + kDownMetaBytes, 4096);
constexpr unsigned kRdnaBufferConfig = 0x31004000u;

static_assert(H24_MODEL_WIDTH % H24_GATE_GROUP == 0);
static_assert(H24_EXPERT_WIDTH % H24_DOWN_GROUP == 0);
static_assert(H24_GATE_GROUP % 8 == 0 && H24_DOWN_GROUP % 32 == 0);
static_assert((H24_GATE_GROUP == 128 && kGateGroups == 20)
              || (H24_GATE_GROUP == 32 && kGateGroups == 80));

__device__ __forceinline__ u32_t pack_even(const opus::vector_t<i8_t, 8>& q) {
    return static_cast<u32_t>(static_cast<unsigned char>(q[0]))
        | (static_cast<u32_t>(static_cast<unsigned char>(q[2])) << 8)
        | (static_cast<u32_t>(static_cast<unsigned char>(q[4])) << 16)
        | (static_cast<u32_t>(static_cast<unsigned char>(q[6])) << 24);
}

__device__ __forceinline__ u32_t pack_odd(const opus::vector_t<i8_t, 8>& q) {
    return static_cast<u32_t>(static_cast<unsigned char>(q[1]))
        | (static_cast<u32_t>(static_cast<unsigned char>(q[3])) << 8)
        | (static_cast<u32_t>(static_cast<unsigned char>(q[5])) << 16)
        | (static_cast<u32_t>(static_cast<unsigned char>(q[7])) << 24);
}

template<int K, int Group>
__device__ __forceinline__ float affine_row(
    opus::gmem<u8_t>& bank_u8,
    opus::gmem<fp16_t>& bank_f16,
    int code_base,
    int meta_base,
    int row,
    opus::gmem<i8_t>& activation_q8,
    opus::gmem<fp16_t>& activation_meta,
    int activation_base,
    int activation_meta_base) {
    constexpr int groups = K / Group;
    constexpr int packed_row_bytes = K / 2;
    int row_code = code_base + row * packed_row_bytes;
    int row_meta = meta_base + row * groups * 2;
    float total = 0.0f;
#pragma clang loop unroll(disable)
    for (int group = 0; group < groups; ++group) {
        auto pair = bank_f16.template load<2>(row_meta + group * 2);
        auto activation_pair = activation_meta.template load<2>(
            activation_meta_base + group * 2);
        int dot = 0;
#pragma unroll
        for (int k = 0; k < Group; k += 8) {
            auto bytes = bank_u8.template load<4>(
                row_code + (group * Group + k) / 2);
            u32_t packed = __builtin_bit_cast(u32_t, bytes);
            u32_t even = packed & 0x0f0f0f0fu;
            u32_t odd = (packed >> 4) & 0x0f0f0f0fu;
            auto q8 = activation_q8.template load<8>(
                activation_base + group * Group + k);
            dot = __builtin_amdgcn_sudot4(
                false, even, true, pack_even(q8), dot, false);
            dot = __builtin_amdgcn_sudot4(
                false, odd, true, pack_odd(q8), dot, false);
        }
        float d = opus::fp16_to_fp32(activation_pair[0]);
        float sum = opus::fp16_to_fp32(activation_pair[1]);
        total += opus::fp16_to_fp32(pair[0])
                * (d * static_cast<float>(dot))
            + opus::fp16_to_fp32(pair[1]) * sum;
    }
    return total;
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + __builtin_expf(-value));
}

}  // namespace h24d

extern "C" __global__ __launch_bounds__(H24_BLOCK_SIZE)
void h24_scalar_gate_up_swiglu(
    const h24d::u8_t* __restrict__ expert_bank,
    unsigned bank_bytes,
    const int* __restrict__ expert_ids,
    const h24d::i8_t* __restrict__ input_q8,
    const h24d::fp16_t* __restrict__ input_meta,
    float* __restrict__ middle_f32) {
    int output = opus::block_id_x() * H24_BLOCK_SIZE + opus::thread_id_x();
    if (output >= H24_TOP_K * H24_EXPERT_WIDTH) return;
    int slot = output / H24_EXPERT_WIDTH;
    int channel = output - slot * H24_EXPERT_WIDTH;
    auto bank_u8 = opus::make_gmem(
        expert_bank, bank_bytes, h24d::kRdnaBufferConfig);
    auto bank_f16 = opus::make_gmem(
        expert_bank == nullptr
            ? static_cast<const h24d::fp16_t*>(nullptr)
            : reinterpret_cast<const h24d::fp16_t*>(expert_bank),
        bank_bytes, h24d::kRdnaBufferConfig);
    auto ids = opus::make_gmem(
        expert_ids, H24_TOP_K * sizeof(int), h24d::kRdnaBufferConfig);
    auto xq = opus::make_gmem(
        input_q8, H24_MODEL_WIDTH, h24d::kRdnaBufferConfig);
    auto xm = opus::make_gmem(
        input_meta, 2 * h24d::kGateGroups * sizeof(h24d::fp16_t),
        h24d::kRdnaBufferConfig);
    auto out = opus::make_gmem(
        middle_f32, H24_TOP_K * H24_EXPERT_WIDTH * sizeof(float),
        h24d::kRdnaBufferConfig);
    int expert_base = ids.template load<1>(slot)[0] * h24d::kExpertStride;
    int gate_row = channel;
    int up_row = H24_EXPERT_WIDTH + channel;
    float gate = h24d::affine_row<H24_MODEL_WIDTH, H24_GATE_GROUP>(
        bank_u8, bank_f16, expert_base + h24d::kGateCodeOffset,
        (expert_base + h24d::kGateMetaOffset) / 2, gate_row,
        xq, xm, 0, 0);
    float up = h24d::affine_row<H24_MODEL_WIDTH, H24_GATE_GROUP>(
        bank_u8, bank_f16, expert_base + h24d::kGateCodeOffset,
        (expert_base + h24d::kGateMetaOffset) / 2, up_row,
        xq, xm, 0, 0);
    out.template store<1>(h24d::silu(gate) * up, output);
}

extern "C" __global__ __launch_bounds__(H24_BLOCK_SIZE)
void h24_wave_gate_up_swiglu(
    const h24d::u8_t* __restrict__ expert_bank,
    unsigned bank_bytes,
    const int* __restrict__ expert_ids,
    const h24d::i8_t* __restrict__ input_q8,
    const h24d::fp16_t* __restrict__ input_meta,
    float* __restrict__ middle_f32) {
    constexpr int kWave = 32;
    constexpr int kWavesPerBlock = H24_BLOCK_SIZE / kWave;
    constexpr int kPasses = (h24d::kGateGroups + kWave - 1) / kWave;
    constexpr int kPackedRowBytes = H24_MODEL_WIDTH / 2;
    int thread = opus::thread_id_x();
    int lane = thread & 31;
    int wave = thread >> 5;
    int output = opus::block_id_x() * kWavesPerBlock + wave;

    auto xq_global = opus::make_gmem(
        input_q8, H24_MODEL_WIDTH, h24d::kRdnaBufferConfig);
    auto xm_global = opus::make_gmem(
        input_meta, 2 * h24d::kGateGroups * sizeof(h24d::fp16_t),
        h24d::kRdnaBufferConfig);
    __shared__ h24d::i8_t shared_xq[H24_MODEL_WIDTH];
    __shared__ h24d::fp16_t shared_xm[2 * h24d::kGateGroups];
    int first = thread * 8;
    if (first < 2048) {
        auto values = xq_global.template load<8>(first);
#pragma unroll
        for (int i = 0; i < 8; ++i) shared_xq[first + i] = values[i];
    }
    if (thread < 64) {
        int second = 2048 + thread * 8;
        auto values = xq_global.template load<8>(second);
#pragma unroll
        for (int i = 0; i < 8; ++i) shared_xq[second + i] = values[i];
    }
    if (thread < h24d::kGateGroups) {
        auto pair = xm_global.template load<2>(thread * 2);
        shared_xm[thread * 2] = pair[0];
        shared_xm[thread * 2 + 1] = pair[1];
    }
    opus::sync_threads();

    int slot = output / H24_EXPERT_WIDTH;
    int channel = output - slot * H24_EXPERT_WIDTH;
    auto bank_u8 = opus::make_gmem(
        expert_bank, bank_bytes, h24d::kRdnaBufferConfig);
    auto bank_f16 = opus::make_gmem(
        expert_bank == nullptr
            ? static_cast<const h24d::fp16_t*>(nullptr)
            : reinterpret_cast<const h24d::fp16_t*>(expert_bank),
        bank_bytes, h24d::kRdnaBufferConfig);
    auto ids = opus::make_gmem(
        expert_ids, H24_TOP_K * sizeof(int), h24d::kRdnaBufferConfig);
    auto out = opus::make_gmem(
        middle_f32, H24_TOP_K * H24_EXPERT_WIDTH * sizeof(float),
        h24d::kRdnaBufferConfig);
    int expert = lane == 0 ? ids.template load<1>(slot)[0] : 0;
    expert = opus::shfl(expert, 0, kWave);
    int expert_base = expert * h24d::kExpertStride;
    int gate_row = channel;
    int up_row = H24_EXPERT_WIDTH + channel;
    int gate_code = expert_base + h24d::kGateCodeOffset
        + gate_row * kPackedRowBytes;
    int up_code = expert_base + h24d::kGateCodeOffset
        + up_row * kPackedRowBytes;
    int gate_meta = (expert_base + h24d::kGateMetaOffset) / 2
        + gate_row * h24d::kGateGroups * 2;
    int up_meta = (expert_base + h24d::kGateMetaOffset) / 2
        + up_row * h24d::kGateGroups * 2;

    float gate_total = 0.0f;
    float up_total = 0.0f;
    if constexpr (H24_GATE_GROUP == 128) {
        constexpr int kChunksPerGroup = H24_GATE_GROUP / 8;
        constexpr int kGroupsPerIteration = kWave / kChunksPerGroup;
        constexpr int kIterations = h24d::kGateGroups / kGroupsPerIteration;
        static_assert(kChunksPerGroup == 16 && kGroupsPerIteration == 2
                      && kIterations == 10);
        int subgroup_lane = lane & (kChunksPerGroup - 1);
#pragma clang loop unroll(disable)
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            int flat_chunk = iteration * kWave + lane;
            int group = flat_chunk / kChunksPerGroup;
            int k = (flat_chunk - group * kChunksPerGroup) * 8;
            auto gate_bytes = bank_u8.template load<4>(
                gate_code + flat_chunk * 4);
            auto up_bytes = bank_u8.template load<4>(
                up_code + flat_chunk * 4);
            h24d::u32_t gate_packed =
                __builtin_bit_cast(h24d::u32_t, gate_bytes);
            h24d::u32_t up_packed =
                __builtin_bit_cast(h24d::u32_t, up_bytes);
            h24d::u32_t gate_even = gate_packed & 0x0f0f0f0fu;
            h24d::u32_t gate_odd = (gate_packed >> 4) & 0x0f0f0f0fu;
            h24d::u32_t up_even = up_packed & 0x0f0f0f0fu;
            h24d::u32_t up_odd = (up_packed >> 4) & 0x0f0f0f0fu;
            opus::vector_t<h24d::i8_t, 8> q8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q8[i] = shared_xq[group * H24_GATE_GROUP + k + i];
            }
            h24d::u32_t even_q8 = h24d::pack_even(q8);
            h24d::u32_t odd_q8 = h24d::pack_odd(q8);
            int gate_dot = __builtin_amdgcn_sudot4(
                false, gate_even, true, even_q8, 0, false);
            gate_dot = __builtin_amdgcn_sudot4(
                false, gate_odd, true, odd_q8, gate_dot, false);
            int up_dot = __builtin_amdgcn_sudot4(
                false, up_even, true, even_q8, 0, false);
            up_dot = __builtin_amdgcn_sudot4(
                false, up_odd, true, odd_q8, up_dot, false);
#pragma unroll
            for (int delta = 8; delta > 0; delta >>= 1) {
                int gate_peer = opus::shfl(
                    gate_dot, subgroup_lane + delta, kChunksPerGroup);
                int up_peer = opus::shfl(
                    up_dot, subgroup_lane + delta, kChunksPerGroup);
                if (subgroup_lane < delta) {
                    gate_dot += gate_peer;
                    up_dot += up_peer;
                }
            }
            if (subgroup_lane == 0) {
                auto gate_pair = bank_f16.template load<2>(
                    gate_meta + group * 2);
                auto up_pair = bank_f16.template load<2>(
                    up_meta + group * 2);
                float d = opus::fp16_to_fp32(shared_xm[group * 2]);
                float sum = opus::fp16_to_fp32(shared_xm[group * 2 + 1]);
                gate_total += opus::fp16_to_fp32(gate_pair[0])
                        * (d * static_cast<float>(gate_dot))
                    + opus::fp16_to_fp32(gate_pair[1]) * sum;
                up_total += opus::fp16_to_fp32(up_pair[0])
                        * (d * static_cast<float>(up_dot))
                    + opus::fp16_to_fp32(up_pair[1]) * sum;
            }
        }
    } else {
#pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            int group = lane + pass * kWave;
            if (group < h24d::kGateGroups) {
                auto gate_pair = bank_f16.template load<2>(
                    gate_meta + group * 2);
                auto up_pair = bank_f16.template load<2>(
                    up_meta + group * 2);
                int gate_dot = 0;
                int up_dot = 0;
#pragma unroll
                for (int k = 0; k < H24_GATE_GROUP; k += 8) {
                    int code_offset = (group * H24_GATE_GROUP + k) / 2;
                    auto gate_bytes = bank_u8.template load<4>(
                        gate_code + code_offset);
                    auto up_bytes = bank_u8.template load<4>(
                        up_code + code_offset);
                    h24d::u32_t gate_packed =
                        __builtin_bit_cast(h24d::u32_t, gate_bytes);
                    h24d::u32_t up_packed =
                        __builtin_bit_cast(h24d::u32_t, up_bytes);
                    h24d::u32_t gate_even = gate_packed & 0x0f0f0f0fu;
                    h24d::u32_t gate_odd = (gate_packed >> 4) & 0x0f0f0f0fu;
                    h24d::u32_t up_even = up_packed & 0x0f0f0f0fu;
                    h24d::u32_t up_odd = (up_packed >> 4) & 0x0f0f0f0fu;
                    opus::vector_t<h24d::i8_t, 8> q8;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        q8[i] = shared_xq[group * H24_GATE_GROUP + k + i];
                    }
                    h24d::u32_t even_q8 = h24d::pack_even(q8);
                    h24d::u32_t odd_q8 = h24d::pack_odd(q8);
                    gate_dot = __builtin_amdgcn_sudot4(
                        false, gate_even, true, even_q8, gate_dot, false);
                    gate_dot = __builtin_amdgcn_sudot4(
                        false, gate_odd, true, odd_q8, gate_dot, false);
                    up_dot = __builtin_amdgcn_sudot4(
                        false, up_even, true, even_q8, up_dot, false);
                    up_dot = __builtin_amdgcn_sudot4(
                        false, up_odd, true, odd_q8, up_dot, false);
                }
                float d = opus::fp16_to_fp32(shared_xm[group * 2]);
                float sum = opus::fp16_to_fp32(shared_xm[group * 2 + 1]);
                gate_total += opus::fp16_to_fp32(gate_pair[0])
                        * (d * static_cast<float>(gate_dot))
                    + opus::fp16_to_fp32(gate_pair[1]) * sum;
                up_total += opus::fp16_to_fp32(up_pair[0])
                        * (d * static_cast<float>(up_dot))
                    + opus::fp16_to_fp32(up_pair[1]) * sum;
            }
        }
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        float gate_peer = opus::shfl(gate_total, lane + delta, kWave);
        float up_peer = opus::shfl(up_total, lane + delta, kWave);
        if (lane < delta) {
            gate_total += gate_peer;
            up_total += up_peer;
        }
    }
    if (lane == 0) {
        out.template store<1>(h24d::silu(gate_total) * up_total, output);
    }
}

extern "C" __global__ __launch_bounds__(32)
void h24_quantize_middle_q8(
    const float* __restrict__ middle_f32,
    h24d::i8_t* __restrict__ middle_q8,
    h24d::fp16_t* __restrict__ middle_meta) {
    constexpr int kPerLane = H24_DOWN_GROUP / 32;
    int group_linear = opus::block_id_x();
    int lane = opus::thread_id_x();
    int group_base = group_linear * H24_DOWN_GROUP;
    auto input = opus::make_gmem(
        middle_f32, H24_TOP_K * H24_EXPERT_WIDTH * sizeof(float),
        h24d::kRdnaBufferConfig);
    auto output = opus::make_gmem(
        middle_q8, H24_TOP_K * H24_EXPERT_WIDTH, h24d::kRdnaBufferConfig);
    auto meta = opus::make_gmem(
        middle_meta, H24_TOP_K * h24d::kDownGroups * 2
            * sizeof(h24d::fp16_t), h24d::kRdnaBufferConfig);
    float values[kPerLane];
    float max_abs = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        values[pass] = input.template load<1>(group_base + lane + pass * 32)[0];
        max_abs = opus::max(max_abs, __builtin_fabsf(values[pass]));
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        float peer = opus::shfl(max_abs, lane + delta, 32);
        if (lane < delta) max_abs = opus::max(max_abs, peer);
    }
    float d = max_abs / 127.0f;
    if (lane == 0) d = opus::fp16_to_fp32(opus::fp32_to_fp16(d));
    d = opus::shfl(d, 0, 32);
    int local_sum = 0;
#pragma unroll
    for (int pass = 0; pass < kPerLane; ++pass) {
        int q = d == 0.0f ? 0
            : static_cast<int>(__builtin_rintf(values[pass] / d));
        q = opus::max(-127, opus::min(127, q));
        output.template store<1>(
            static_cast<h24d::i8_t>(q), group_base + lane + pass * 32);
        local_sum += q;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        int peer = opus::shfl(local_sum, lane + delta, 32);
        if (lane < delta) local_sum += peer;
    }
    if (lane == 0) {
        opus::vector_t<h24d::fp16_t, 2> pair;
        pair[0] = opus::fp32_to_fp16(d);
        pair[1] = opus::fp32_to_fp16(d * static_cast<float>(local_sum));
        meta.template store<2>(pair, group_linear * 2);
    }
}

extern "C" __global__ __launch_bounds__(H24_BLOCK_SIZE)
void h24_down_top10(
    const h24d::u8_t* __restrict__ expert_bank,
    unsigned bank_bytes,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    const h24d::i8_t* __restrict__ middle_q8,
    const h24d::fp16_t* __restrict__ middle_meta,
    float* __restrict__ output_f32) {
    int row = opus::block_id_x() * H24_BLOCK_SIZE + opus::thread_id_x();
    if (row >= H24_MODEL_WIDTH) return;
    auto bank_u8 = opus::make_gmem(
        expert_bank, bank_bytes, h24d::kRdnaBufferConfig);
    auto bank_f16 = opus::make_gmem(
        expert_bank == nullptr
            ? static_cast<const h24d::fp16_t*>(nullptr)
            : reinterpret_cast<const h24d::fp16_t*>(expert_bank),
        bank_bytes, h24d::kRdnaBufferConfig);
    auto ids = opus::make_gmem(
        expert_ids, H24_TOP_K * sizeof(int), h24d::kRdnaBufferConfig);
    auto routes = opus::make_gmem(
        route_weights, H24_TOP_K * sizeof(float), h24d::kRdnaBufferConfig);
    auto mq = opus::make_gmem(
        middle_q8, H24_TOP_K * H24_EXPERT_WIDTH, h24d::kRdnaBufferConfig);
    auto mm = opus::make_gmem(
        middle_meta, H24_TOP_K * h24d::kDownGroups * 2
            * sizeof(h24d::fp16_t), h24d::kRdnaBufferConfig);
    auto out = opus::make_gmem(
        output_f32, H24_MODEL_WIDTH * sizeof(float), h24d::kRdnaBufferConfig);
    float total = 0.0f;
#pragma clang loop unroll(disable)
    for (int slot = 0; slot < H24_TOP_K; ++slot) {
        int expert_base = ids.template load<1>(slot)[0] * h24d::kExpertStride;
        float value = h24d::affine_row<H24_EXPERT_WIDTH, H24_DOWN_GROUP>(
            bank_u8, bank_f16, expert_base + h24d::kDownCodeOffset,
            (expert_base + h24d::kDownMetaOffset) / 2, row, mq, mm,
            slot * H24_EXPERT_WIDTH, slot * h24d::kDownGroups * 2);
        total += routes.template load<1>(slot)[0] * value;
    }
    out.template store<1>(total, row);
}

#else

#include "opus/hip_minimal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" __global__ void h24_scalar_gate_up_swiglu(
    const unsigned char*, unsigned, const int*, const signed char*,
    const _Float16*, float*) {}
extern "C" __global__ void h24_wave_gate_up_swiglu(
    const unsigned char*, unsigned, const int*, const signed char*,
    const _Float16*, float*) {}
extern "C" __global__ void h24_quantize_middle_q8(
    const float*, signed char*, _Float16*) {}
extern "C" __global__ void h24_down_top10(
    const unsigned char*, unsigned, const int*, const float*,
    const signed char*, const _Float16*, float*) {}

namespace h24 {

constexpr int kModelWidth = H24_MODEL_WIDTH;
constexpr int kExpertWidth = H24_EXPERT_WIDTH;
constexpr int kTopK = H24_TOP_K;
constexpr int kBlock = H24_BLOCK_SIZE;
constexpr int kExperts = 10;
constexpr int kGateGroup = H24_GATE_GROUP;
constexpr int kDownGroup = H24_DOWN_GROUP;
constexpr int kGateGroups = kModelWidth / kGateGroup;
constexpr int kDownGroups = kExpertWidth / kDownGroup;
constexpr int kGateRows = 2 * kExpertWidth;
constexpr int kGateCodeBytes = kGateRows * kModelWidth / 2;
constexpr int kGateMetaBytes = kGateRows * kGateGroups * 4;
constexpr int kDownCodeBytes = kModelWidth * kExpertWidth / 2;
constexpr int kDownMetaBytes = kModelWidth * kDownGroups * 4;
constexpr int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
constexpr int kGateMetaOffset = align_up(kGateCodeBytes, 256);
constexpr int kDownCodeOffset = align_up(kGateMetaOffset + kGateMetaBytes, 256);
constexpr int kDownMetaOffset = align_up(kDownCodeOffset + kDownCodeBytes, 256);
constexpr int kExpertStride = align_up(kDownMetaOffset + kDownMetaBytes, 4096);
static_assert(
    (H24_SHAPE_KIND == 2 && kGateGroup == 128 && kDownGroup == 128
        && kExpertStride == 2613248)
    || (H24_SHAPE_KIND == 3 && kGateGroup == 128 && kDownGroup == 64
        && kExpertStride == 2662400)
    || (H24_SHAPE_KIND == 500 && kGateGroup == 32 && kDownGroup == 32
        && kExpertStride == 3072000));

constexpr const char* shape_name() {
    if constexpr (H24_SHAPE_KIND == 2) return "E2";
    if constexpr (H24_SHAPE_KIND == 3) return "E3";
    return "T500";
}

int code_value(int plane, int expert, int row, int k) {
    return (1 + plane * 3 + expert * 5 + row * 7 + (k / 8) * 11 + k) & 15;
}

void write_half_pair(std::vector<unsigned char>& bank, size_t offset,
                     float first, float second) {
    _Float16 pair[2] = {
        static_cast<_Float16>(first), static_cast<_Float16>(second)};
    std::memcpy(bank.data() + offset, pair, sizeof(pair));
}

void serialize_packets(std::vector<unsigned char>& bank) {
    for (int expert = 0; expert < kExperts; ++expert) {
        size_t base = static_cast<size_t>(expert) * kExpertStride;
        for (int row = 0; row < kGateRows; ++row) {
            size_t row_base = base + static_cast<size_t>(row) * kModelWidth / 2;
            for (int byte = 0; byte < kModelWidth / 2; ++byte) {
                int k = 2 * byte;
                int lo = code_value(0, expert, row, k);
                int hi = code_value(0, expert, row, k + 1);
                bank[row_base + byte] = static_cast<unsigned char>(lo | (hi << 4));
            }
            for (int group = 0; group < kGateGroups; ++group) {
                float scale = static_cast<float>(
                    1 + ((expert + row + group) % 3)) / 2048.0f;
                float offset = static_cast<float>(
                    ((expert * 3 + row + group) % 5) - 2) / 2048.0f;
                write_half_pair(bank, base + kGateMetaOffset
                    + static_cast<size_t>(row * kGateGroups + group) * 4,
                    scale, offset);
            }
        }
        for (int row = 0; row < kModelWidth; ++row) {
            size_t row_base = base + kDownCodeOffset
                + static_cast<size_t>(row) * kExpertWidth / 2;
            for (int byte = 0; byte < kExpertWidth / 2; ++byte) {
                int k = 2 * byte;
                int lo = code_value(1, expert, row, k);
                int hi = code_value(1, expert, row, k + 1);
                bank[row_base + byte] = static_cast<unsigned char>(lo | (hi << 4));
            }
            for (int group = 0; group < kDownGroups; ++group) {
                float scale = static_cast<float>(
                    1 + ((expert + row + group) % 2)) / 1024.0f;
                float offset = static_cast<float>(
                    ((expert + row * 2 + group) % 5) - 2) / 2048.0f;
                write_half_pair(bank, base + kDownMetaOffset
                    + static_cast<size_t>(row * kDownGroups + group) * 4,
                    scale, offset);
            }
        }
    }
}

int read_code(const std::vector<unsigned char>& bank, size_t row_base, int k) {
    unsigned byte = bank[row_base + static_cast<size_t>(k / 2)];
    return (k & 1) ? static_cast<int>(byte >> 4)
                   : static_cast<int>(byte & 15);
}

void read_half_pair(const std::vector<unsigned char>& bank, size_t offset,
                    float& first, float& second) {
    _Float16 pair[2];
    std::memcpy(pair, bank.data() + offset, sizeof(pair));
    first = static_cast<float>(pair[0]);
    second = static_cast<float>(pair[1]);
}

float affine_host(const std::vector<unsigned char>& bank, int expert,
                  bool down, int row, const signed char* q8,
                  const _Float16* activation_meta) {
    int width = down ? kExpertWidth : kModelWidth;
    int group_width = down ? kDownGroup : kGateGroup;
    int groups = width / group_width;
    size_t base = static_cast<size_t>(expert) * kExpertStride;
    size_t code_plane = base + (down ? kDownCodeOffset : 0);
    size_t meta_plane = base + (down ? kDownMetaOffset : kGateMetaOffset);
    size_t row_code = code_plane + static_cast<size_t>(row) * width / 2;
    float total = 0.0f;
    for (int group = 0; group < groups; ++group) {
        float scale = 0.0f;
        float offset = 0.0f;
        read_half_pair(bank, meta_plane
            + static_cast<size_t>(row * groups + group) * 4, scale, offset);
        int dot = 0;
        for (int k = 0; k < group_width; ++k) {
            int index = group * group_width + k;
            dot += read_code(bank, row_code, index) * static_cast<int>(q8[index]);
        }
        float d = static_cast<float>(activation_meta[group * 2]);
        float sum = static_cast<float>(activation_meta[group * 2 + 1]);
        total += scale * (d * static_cast<float>(dot)) + offset * sum;
    }
    return total;
}

void quantize_host(const std::vector<float>& middle,
                   std::vector<signed char>& q8,
                   std::vector<_Float16>& meta) {
    for (int slot = 0; slot < kTopK; ++slot) {
        for (int group = 0; group < kDownGroups; ++group) {
            int base = slot * kExpertWidth + group * kDownGroup;
            float max_abs = 0.0f;
            for (int k = 0; k < kDownGroup; ++k) {
                max_abs = std::max(max_abs, std::fabs(middle[base + k]));
            }
            _Float16 d_half = static_cast<_Float16>(max_abs / 127.0f);
            float d = static_cast<float>(d_half);
            int sum_q = 0;
            for (int k = 0; k < kDownGroup; ++k) {
                int value = d == 0.0f ? 0
                    : static_cast<int>(std::nearbyint(middle[base + k] / d));
                value = std::max(-127, std::min(127, value));
                q8[base + k] = static_cast<signed char>(value);
                sum_q += value;
            }
            int meta_base = (slot * kDownGroups + group) * 2;
            meta[meta_base] = d_half;
            meta[meta_base + 1] = static_cast<_Float16>(d * sum_q);
        }
    }
}

struct ErrorStats { double relative_l2; float max_abs; };

ErrorStats error_stats(const std::vector<float>& got,
                       const std::vector<float>& expected) {
    double error2 = 0.0;
    double reference2 = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float delta = got[i] - expected[i];
        error2 += static_cast<double>(delta) * delta;
        reference2 += static_cast<double>(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::fabs(delta));
    }
    return {std::sqrt(error2 / reference2), max_abs};
}

}  // namespace h24

#define H24_HIP(call) do { \
    hipError_t h24_error = (call); \
    if (h24_error != hipSuccess) { \
        std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            static_cast<int>(h24_error), hipGetErrorString(h24_error), \
            __FILE__, __LINE__); \
        return 2; \
    } \
} while (0)

int main(int argc, char** argv) {
    using namespace h24;
    std::string candidate = argc > 1 ? argv[1] : "wave";
    if (candidate != "scalar" && candidate != "wave") {
        std::fprintf(stderr, "usage: %s [scalar|wave]\n", argv[0]);
        return 2;
    }
    const int ids_host[kTopK] = {9, 0, 8, 1, 7, 2, 6, 3, 5, 4};
    const float routes_host[kTopK] = {
        0.125f, 0.0625f, 0.1875f, 0.09375f, 0.15625f,
        0.046875f, 0.109375f, 0.078125f, 0.0859375f, 0.0546875f};

    std::vector<unsigned char> bank(
        static_cast<size_t>(kExperts) * kExpertStride, 0);
    serialize_packets(bank);
    std::vector<signed char> input_q8(kModelWidth);
    std::vector<_Float16> input_meta(2 * kGateGroups);
    for (int k = 0; k < kModelWidth; ++k) {
        int value = 1 + (k % 7);
        input_q8[k] = static_cast<signed char>((k % 8) == 0 ? -7 : value);
    }
    for (int group = 0; group < kGateGroups; ++group) {
        _Float16 d_half = static_cast<_Float16>(1.0f / 32.0f);
        float d = static_cast<float>(d_half);
        int sum_q = 0;
        for (int k = 0; k < kGateGroup; ++k) {
            sum_q += input_q8[group * kGateGroup + k];
        }
        input_meta[group * 2] = d_half;
        input_meta[group * 2 + 1] = static_cast<_Float16>(d * sum_q);
    }

    std::vector<float> middle_oracle(kTopK * kExpertWidth);
    for (int slot = 0; slot < kTopK; ++slot) {
        for (int channel = 0; channel < kExpertWidth; ++channel) {
            float gate = affine_host(bank, ids_host[slot], false, 2 * channel,
                                     input_q8.data(), input_meta.data());
            float up = affine_host(bank, ids_host[slot], false, 2 * channel + 1,
                                   input_q8.data(), input_meta.data());
            middle_oracle[slot * kExpertWidth + channel] =
                (gate / (1.0f + std::exp(-gate))) * up;
        }
    }
    std::vector<signed char> q8_oracle(kTopK * kExpertWidth);
    std::vector<_Float16> meta_oracle(kTopK * kDownGroups * 2);
    quantize_host(middle_oracle, q8_oracle, meta_oracle);
    std::vector<float> output_oracle(kModelWidth, 0.0f);
    for (int row = 0; row < kModelWidth; ++row) {
        float total = 0.0f;
        for (int slot = 0; slot < kTopK; ++slot) {
            total += routes_host[slot] * affine_host(
                bank, ids_host[slot], true, row,
                q8_oracle.data() + slot * kExpertWidth,
                meta_oracle.data() + slot * kDownGroups * 2);
        }
        output_oracle[row] = total;
    }

    unsigned char* bank_device = nullptr;
    int* ids_device = nullptr;
    float* routes_device = nullptr;
    signed char* input_q8_device = nullptr;
    _Float16* input_meta_device = nullptr;
    float* middle_device = nullptr;
    signed char* middle_q8_device = nullptr;
    _Float16* middle_meta_device = nullptr;
    float* output_device = nullptr;
    size_t bank_bytes = bank.size();
    H24_HIP(hipMalloc(&bank_device, bank_bytes));
    H24_HIP(hipMalloc(&ids_device, sizeof(ids_host)));
    H24_HIP(hipMalloc(&routes_device, sizeof(routes_host)));
    H24_HIP(hipMalloc(&input_q8_device, input_q8.size()));
    H24_HIP(hipMalloc(
        &input_meta_device, input_meta.size() * sizeof(_Float16)));
    H24_HIP(hipMalloc(&middle_device, middle_oracle.size() * sizeof(float)));
    H24_HIP(hipMalloc(&middle_q8_device, q8_oracle.size()));
    H24_HIP(hipMalloc(
        &middle_meta_device, meta_oracle.size() * sizeof(_Float16)));
    H24_HIP(hipMalloc(&output_device, output_oracle.size() * sizeof(float)));
    H24_HIP(hipMemcpy(
        bank_device, bank.data(), bank_bytes, hipMemcpyHostToDevice));
    H24_HIP(hipMemcpy(
        ids_device, ids_host, sizeof(ids_host), hipMemcpyHostToDevice));
    H24_HIP(hipMemcpy(
        routes_device, routes_host, sizeof(routes_host), hipMemcpyHostToDevice));
    H24_HIP(hipMemcpy(input_q8_device, input_q8.data(), input_q8.size(),
                      hipMemcpyHostToDevice));
    H24_HIP(hipMemcpy(input_meta_device, input_meta.data(),
        input_meta.size() * sizeof(_Float16), hipMemcpyHostToDevice));

    if (candidate == "scalar") {
        hipLaunchKernelGGL(h24_scalar_gate_up_swiglu,
            dim3((kTopK * kExpertWidth + kBlock - 1) / kBlock), dim3(kBlock),
            0, nullptr, bank_device, static_cast<unsigned>(bank_bytes),
            ids_device, input_q8_device, input_meta_device, middle_device);
    } else {
        constexpr int waves_per_block = kBlock / 32;
        hipLaunchKernelGGL(h24_wave_gate_up_swiglu,
            dim3((kTopK * kExpertWidth + waves_per_block - 1)
                / waves_per_block),
            dim3(kBlock), 0, nullptr, bank_device,
            static_cast<unsigned>(bank_bytes), ids_device, input_q8_device,
            input_meta_device, middle_device);
    }
    H24_HIP(hipGetLastError());
    hipLaunchKernelGGL(h24_quantize_middle_q8,
        dim3(kTopK * kDownGroups), dim3(32), 0, nullptr,
        middle_device, middle_q8_device, middle_meta_device);
    H24_HIP(hipGetLastError());
    hipLaunchKernelGGL(h24_down_top10,
        dim3((kModelWidth + kBlock - 1) / kBlock), dim3(kBlock), 0, nullptr,
        bank_device, static_cast<unsigned>(bank_bytes), ids_device,
        routes_device, middle_q8_device, middle_meta_device, output_device);
    H24_HIP(hipGetLastError());
    H24_HIP(hipDeviceSynchronize());

    std::vector<float> middle_gpu(middle_oracle.size());
    std::vector<signed char> q8_gpu(q8_oracle.size());
    std::vector<_Float16> meta_gpu(meta_oracle.size());
    std::vector<float> output_gpu(output_oracle.size());
    H24_HIP(hipMemcpy(middle_gpu.data(), middle_device,
        middle_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
    H24_HIP(hipMemcpy(q8_gpu.data(), middle_q8_device,
        q8_gpu.size(), hipMemcpyDeviceToHost));
    H24_HIP(hipMemcpy(meta_gpu.data(), middle_meta_device,
        meta_gpu.size() * sizeof(_Float16), hipMemcpyDeviceToHost));
    H24_HIP(hipMemcpy(output_gpu.data(), output_device,
        output_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));

    auto middle_error = error_stats(middle_gpu, middle_oracle);
    int q8_mismatches = 0;
    for (size_t i = 0; i < q8_gpu.size(); ++i) {
        q8_mismatches += q8_gpu[i] != q8_oracle[i];
    }
    int meta_mismatches = 0;
    for (size_t i = 0; i < meta_gpu.size(); ++i) {
        uint16_t got = 0;
        uint16_t expected = 0;
        std::memcpy(&got, &meta_gpu[i], sizeof(got));
        std::memcpy(&expected, &meta_oracle[i], sizeof(expected));
        meta_mismatches += got != expected;
    }
    auto output_error = error_stats(output_gpu, output_oracle);
    bool nonzero = false;
    for (float value : output_gpu) nonzero = nonzero || value != 0.0f;
    bool pass = middle_error.relative_l2 < 2.0e-5
        && q8_mismatches == 0 && meta_mismatches == 0
        && output_error.relative_l2 < 2.0e-5 && nonzero;
    std::printf(
        "{\"status\":\"%s\",\"candidate\":\"%s\","
        "\"fixture\":\"distinct_top10\",\"shape\":\"%s\","
        "\"gate_group\":%d,\"down_group\":%d,"
        "\"gate_groups\":%d,\"down_groups\":%d,"
        "\"experts\":10,\"expert_ids\":[9,0,8,1,7,2,6,3,5,4],"
        "\"route_sum\":1,\"expert_stride\":%d,\"packet_bytes\":%zu,"
        "\"gate_grid_blocks\":%d,\"gate_block_threads\":256,"
        "\"middle_relative_l2\":%.9g,\"middle_max_abs_error\":%.9g,"
        "\"q8_mismatches\":%d,\"meta_bit_mismatches\":%d,"
        "\"output_relative_l2\":%.9g,\"output_max_abs_error\":%.9g,"
        "\"output_probe\":[%.9g,%.9g,%.9g]}\n",
        pass ? "pass" : "fail", candidate.c_str(), shape_name(),
        kGateGroup, kDownGroup, kGateGroups, kDownGroups, kExpertStride,
        bank_bytes, candidate == "scalar" ? 25 : 800,
        middle_error.relative_l2, middle_error.max_abs,
        q8_mismatches, meta_mismatches,
        output_error.relative_l2, output_error.max_abs,
        output_gpu[0], output_gpu[1279], output_gpu[2559]);

    hipFree(output_device);
    hipFree(middle_meta_device);
    hipFree(middle_q8_device);
    hipFree(middle_device);
    hipFree(input_meta_device);
    hipFree(input_q8_device);
    hipFree(routes_device);
    hipFree(ids_device);
    hipFree(bank_device);
    return pass ? 0 : 1;
}

#endif

// H30: fuse H24's accepted E3 G128 gate/up/SwiGLU with natural D64 Q8 output.

#define H24_GATE_GROUP 128
#define H24_DOWN_GROUP 64
#define H24_SHAPE_KIND 3
#define main h24_reference_main
#include "../../h24-finalist-wave32-gate-up/code/h24_finalist_gate_up.cu"
#undef main

#ifdef __HIP_DEVICE_COMPILE__

namespace h30d {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = H24_BLOCK_SIZE / kWave;
constexpr int kChannelsPerGroup = H24_DOWN_GROUP;
constexpr int kBatches = kChannelsPerGroup / kWavesPerBlock;
constexpr int kChunksPerGateGroup = H24_GATE_GROUP / 8;
constexpr int kGateGroupsPerIteration = kWave / kChunksPerGateGroup;
constexpr int kGateIterations = h24d::kGateGroups / kGateGroupsPerIteration;
constexpr int kPackedGateRowBytes = H24_MODEL_WIDTH / 2;

static_assert(kWavesPerBlock == 8);
static_assert(kChannelsPerGroup == 64 && kBatches == 8);
static_assert(kChunksPerGateGroup == 16);
static_assert(kGateGroupsPerIteration == 2 && kGateIterations == 10);
static_assert(h24d::kExpertStride == 2662400);

__device__ __forceinline__ float middle_channel(
    opus::gmem<h24d::u8_t>& bank_u8,
    opus::gmem<h24d::fp16_t>& bank_f16,
    int expert_base,
    int channel,
    int lane,
    const h24d::i8_t* shared_xq,
    const h24d::fp16_t* shared_xm) {
    int gate_row = channel;
    int up_row = H24_EXPERT_WIDTH + channel;
    int gate_code = expert_base + h24d::kGateCodeOffset
        + gate_row * kPackedGateRowBytes;
    int up_code = expert_base + h24d::kGateCodeOffset
        + up_row * kPackedGateRowBytes;
    int gate_meta = (expert_base + h24d::kGateMetaOffset) / 2
        + gate_row * h24d::kGateGroups * 2;
    int up_meta = (expert_base + h24d::kGateMetaOffset) / 2
        + up_row * h24d::kGateGroups * 2;
    int subgroup_lane = lane & (kChunksPerGateGroup - 1);
    float gate_total = 0.0f;
    float up_total = 0.0f;
#pragma clang loop unroll(disable)
    for (int iteration = 0; iteration < kGateIterations; ++iteration) {
        int flat_chunk = iteration * kWave + lane;
        int group = flat_chunk / kChunksPerGateGroup;
        int k = (flat_chunk - group * kChunksPerGateGroup) * 8;
        auto gate_bytes = bank_u8.template load<4>(
            gate_code + flat_chunk * 4);
        auto up_bytes = bank_u8.template load<4>(up_code + flat_chunk * 4);
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
                gate_dot, subgroup_lane + delta, kChunksPerGateGroup);
            int up_peer = opus::shfl(
                up_dot, subgroup_lane + delta, kChunksPerGateGroup);
            if (subgroup_lane < delta) {
                gate_dot += gate_peer;
                up_dot += up_peer;
            }
        }
        if (subgroup_lane == 0) {
            auto gate_pair = bank_f16.template load<2>(
                gate_meta + group * 2);
            auto up_pair = bank_f16.template load<2>(up_meta + group * 2);
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
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        float gate_peer = opus::shfl(gate_total, lane + delta, kWave);
        float up_peer = opus::shfl(up_total, lane + delta, kWave);
        if (lane < delta) {
            gate_total += gate_peer;
            up_total += up_peer;
        }
    }
    return h24d::silu(gate_total) * up_total;
}

}  // namespace h30d

extern "C" __global__ __launch_bounds__(H24_BLOCK_SIZE)
void h30_e3_gate_up_swiglu_q8_d64(
    const h24d::u8_t* __restrict__ expert_bank,
    unsigned bank_bytes,
    const int* __restrict__ expert_ids,
    const h24d::i8_t* __restrict__ input_q8,
    const h24d::fp16_t* __restrict__ input_meta,
    h24d::i8_t* __restrict__ middle_q8,
    h24d::fp16_t* __restrict__ middle_meta) {
    int thread = opus::thread_id_x();
    int lane = thread & 31;
    int wave = thread >> 5;
    int group_linear = opus::block_id_x();
    int slot = group_linear / h24d::kDownGroups;
    int down_group = group_linear - slot * h24d::kDownGroups;

    auto xq_global = opus::make_gmem(
        input_q8, H24_MODEL_WIDTH, h24d::kRdnaBufferConfig);
    auto xm_global = opus::make_gmem(
        input_meta, 2 * h24d::kGateGroups * sizeof(h24d::fp16_t),
        h24d::kRdnaBufferConfig);
    auto ids = opus::make_gmem(
        expert_ids, H24_TOP_K * sizeof(int), h24d::kRdnaBufferConfig);
    auto bank_u8 = opus::make_gmem(
        expert_bank, bank_bytes, h24d::kRdnaBufferConfig);
    auto bank_f16 = opus::make_gmem(
        expert_bank == nullptr
            ? static_cast<const h24d::fp16_t*>(nullptr)
            : reinterpret_cast<const h24d::fp16_t*>(expert_bank),
        bank_bytes, h24d::kRdnaBufferConfig);
    auto q8_out = opus::make_gmem(
        middle_q8, H24_TOP_K * H24_EXPERT_WIDTH,
        h24d::kRdnaBufferConfig);
    auto meta_out = opus::make_gmem(
        middle_meta,
        H24_TOP_K * h24d::kDownGroups * 2 * sizeof(h24d::fp16_t),
        h24d::kRdnaBufferConfig);

    __shared__ h24d::i8_t shared_xq[H24_MODEL_WIDTH];
    __shared__ h24d::fp16_t shared_xm[2 * h24d::kGateGroups];
    __shared__ float shared_middle[h30d::kChannelsPerGroup];
    __shared__ float shared_wave_max[2];
    __shared__ int shared_wave_sum[2];
    __shared__ float shared_d;
    __shared__ int shared_expert;

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
    if (thread == 0) shared_expert = ids.template load<1>(slot)[0];
    opus::sync_threads();

    int expert_base = shared_expert * h24d::kExpertStride;
#pragma unroll
    for (int batch = 0; batch < h30d::kBatches; ++batch) {
        int local_channel = batch * h30d::kWavesPerBlock + wave;
        int channel = down_group * h30d::kChannelsPerGroup + local_channel;
        float middle = h30d::middle_channel(
            bank_u8, bank_f16, expert_base, channel, lane,
            shared_xq, shared_xm);
        if (lane == 0) shared_middle[local_channel] = middle;
    }
    opus::sync_threads();

    float max_abs = 0.0f;
    if (thread < h30d::kChannelsPerGroup) {
        max_abs = __builtin_fabsf(shared_middle[thread]);
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            float peer = opus::shfl(max_abs, lane + delta, h30d::kWave);
            if (lane < delta) max_abs = opus::max(max_abs, peer);
        }
        if (lane == 0) shared_wave_max[wave] = max_abs;
    }
    opus::sync_threads();
    if (thread == 0) {
        float group_max = opus::max(shared_wave_max[0], shared_wave_max[1]);
        shared_d = opus::fp16_to_fp32(
            opus::fp32_to_fp16(group_max / 127.0f));
    }
    opus::sync_threads();

    int local_sum = 0;
    if (thread < h30d::kChannelsPerGroup) {
        float d = shared_d;
        int q = d == 0.0f ? 0
            : static_cast<int>(__builtin_rintf(shared_middle[thread] / d));
        q = opus::max(-127, opus::min(127, q));
        int output_index = slot * H24_EXPERT_WIDTH
            + down_group * h30d::kChannelsPerGroup + thread;
        q8_out.template store<1>(static_cast<h24d::i8_t>(q), output_index);
        local_sum = q;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            int peer = opus::shfl(local_sum, lane + delta, h30d::kWave);
            if (lane < delta) local_sum += peer;
        }
        if (lane == 0) shared_wave_sum[wave] = local_sum;
    }
    opus::sync_threads();
    if (thread == 0) {
        int sum = shared_wave_sum[0] + shared_wave_sum[1];
        opus::vector_t<h24d::fp16_t, 2> pair;
        pair[0] = opus::fp32_to_fp16(shared_d);
        pair[1] = opus::fp32_to_fp16(
            shared_d * static_cast<float>(sum));
        meta_out.template store<2>(pair, group_linear * 2);
    }
}

#else

extern "C" __global__ void h30_e3_gate_up_swiglu_q8_d64(
    const unsigned char*, unsigned, const int*, const signed char*,
    const _Float16*, signed char*, _Float16*) {}

#define H30_HIP(call) do { \
    hipError_t h30_error = (call); \
    if (h30_error != hipSuccess) { \
        std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            static_cast<int>(h30_error), hipGetErrorString(h30_error), \
            __FILE__, __LINE__); \
        return 2; \
    } \
} while (0)

#ifndef H30_NO_HOST_MAIN
int main() {
    using namespace h24;
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
            float up = affine_host(bank, ids_host[slot], false,
                                   2 * channel + 1,
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
    signed char* middle_q8_device = nullptr;
    _Float16* middle_meta_device = nullptr;
    float* output_device = nullptr;
    size_t bank_bytes = bank.size();
    H30_HIP(hipMalloc(&bank_device, bank_bytes));
    H30_HIP(hipMalloc(&ids_device, sizeof(ids_host)));
    H30_HIP(hipMalloc(&routes_device, sizeof(routes_host)));
    H30_HIP(hipMalloc(&input_q8_device, input_q8.size()));
    H30_HIP(hipMalloc(
        &input_meta_device, input_meta.size() * sizeof(_Float16)));
    H30_HIP(hipMalloc(&middle_q8_device, q8_oracle.size()));
    H30_HIP(hipMalloc(
        &middle_meta_device, meta_oracle.size() * sizeof(_Float16)));
    H30_HIP(hipMalloc(&output_device, output_oracle.size() * sizeof(float)));
    H30_HIP(hipMemcpy(
        bank_device, bank.data(), bank_bytes, hipMemcpyHostToDevice));
    H30_HIP(hipMemcpy(
        ids_device, ids_host, sizeof(ids_host), hipMemcpyHostToDevice));
    H30_HIP(hipMemcpy(
        routes_device, routes_host, sizeof(routes_host), hipMemcpyHostToDevice));
    H30_HIP(hipMemcpy(
        input_q8_device, input_q8.data(), input_q8.size(), hipMemcpyHostToDevice));
    H30_HIP(hipMemcpy(input_meta_device, input_meta.data(),
        input_meta.size() * sizeof(_Float16), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(h30_e3_gate_up_swiglu_q8_d64,
        dim3(kTopK * kDownGroups), dim3(kBlock), 0, nullptr,
        bank_device, static_cast<unsigned>(bank_bytes), ids_device,
        input_q8_device, input_meta_device,
        middle_q8_device, middle_meta_device);
    H30_HIP(hipGetLastError());
    hipLaunchKernelGGL(h24_down_top10,
        dim3((kModelWidth + kBlock - 1) / kBlock), dim3(kBlock), 0, nullptr,
        bank_device, static_cast<unsigned>(bank_bytes), ids_device,
        routes_device, middle_q8_device, middle_meta_device, output_device);
    H30_HIP(hipGetLastError());
    H30_HIP(hipDeviceSynchronize());

    std::vector<signed char> q8_gpu(q8_oracle.size());
    std::vector<_Float16> meta_gpu(meta_oracle.size());
    std::vector<float> output_gpu(output_oracle.size());
    H30_HIP(hipMemcpy(
        q8_gpu.data(), middle_q8_device, q8_gpu.size(), hipMemcpyDeviceToHost));
    H30_HIP(hipMemcpy(meta_gpu.data(), middle_meta_device,
        meta_gpu.size() * sizeof(_Float16), hipMemcpyDeviceToHost));
    H30_HIP(hipMemcpy(output_gpu.data(), output_device,
        output_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));

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
    bool pass = q8_mismatches == 0 && meta_mismatches == 0
        && output_error.relative_l2 < 2.0e-5 && nonzero;
    std::printf(
        "{\"schema\":\"ciru.h30.e3-fused-middle-q8.v1\","
        "\"status\":\"%s\",\"candidate\":\"E3_G128_D64_FUSED_Q8_GROUP\","
        "\"fixture\":\"distinct_top10\",\"experts\":10,"
        "\"expert_stride\":%d,\"packet_bytes\":%zu,"
        "\"fused_grid_blocks\":%d,\"fused_block_threads\":256,"
        "\"global_middle_f32_allocated\":false,"
        "\"global_middle_f32_bytes_eliminated\":25600,"
        "\"global_middle_f32_write_read_bytes_eliminated\":51200,"
        "\"standalone_quantize_dispatch_eliminated\":true,"
        "\"q8_mismatches\":%d,\"meta_bit_mismatches\":%d,"
        "\"output_relative_l2\":%.9g,\"output_max_abs_error\":%.9g,"
        "\"output_nonzero\":%s,\"output_probe\":[%.9g,%.9g,%.9g]}\n",
        pass ? "pass" : "fail", kExpertStride, bank_bytes,
        kTopK * kDownGroups, q8_mismatches, meta_mismatches,
        output_error.relative_l2, output_error.max_abs,
        nonzero ? "true" : "false",
        output_gpu[0], output_gpu[1279], output_gpu[2559]);

    hipFree(output_device);
    hipFree(middle_meta_device);
    hipFree(middle_q8_device);
    hipFree(input_meta_device);
    hipFree(input_q8_device);
    hipFree(routes_device);
    hipFree(ids_device);
    hipFree(bank_device);
    return pass ? 0 : 1;
}
#endif

#endif

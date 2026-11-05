// Four adjacent output rows share one activation packet and one wave.
// Packet, correction, K16 assignment and each row's reduction match R1.
template<int K, int N, bool NativeIU4, bool Centered>
static __global__ __launch_bounds__(32) void c1_vector_r4_runtime(
        const uint8_t * __restrict__ main, const half * __restrict__ rows,
        const half * __restrict__ tables, const C1PackedQ8K32Split * __restrict__ activation,
        const int32_t * __restrict__ ids, float * __restrict__ output,
        int experts, int routes, int input_channels, int ids_token_stride) {
    static_assert(N % 4 == 0);
    constexpr int Groups16 = K / 16, Groups32 = K / 32;
    constexpr int CodeRowBytes = K / 2, MainRowBytes = K / 2 + K / 16;
    constexpr int MainExpertBytes = N * MainRowBytes;
    const unsigned lane = threadIdx.x, group_n = blockIdx.x;
    const unsigned route = blockIdx.y, token = blockIdx.z;
    const int expert = ids[size_t(token) * ids_token_stride + route];
    if (expert < 0 || expert >= experts) return;
    const uint8_t * expert_main = main + size_t(expert) * MainExpertBytes;
    const C1PackedQ8K32Split * x = activation +
        (size_t(token) * input_channels + route % input_channels) * Groups32;
    const half * table = tables + size_t(expert) * 512;
    float total[4] = {}, scale[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        scale[row] = float(rows[size_t(expert) * N + group_n * 4 + row]);
    }
#pragma clang loop unroll(disable)
    for (unsigned g = lane; g < Groups16; g += 32) {
        const unsigned half_group = g & 1, full_group = g >> 1;
        const C1PackedQ8K32Split & a = x[full_group];
        const uint32_t low0 = a.low[half_group * 2], low1 = a.low[half_group * 2 + 1];
        const uint32_t high0 = a.high[half_group * 2], high1 = a.high[half_group * 2 + 1];
        const float dx = float(a.d), sx = float(half_group ? a.s1 : a.s0);
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint8_t * row_main = expert_main + size_t(group_n * 4 + row) * MainRowBytes;
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t code = c1_load32(row_main + g * 8 + j * 4);
                const uint32_t low = j ? low1 : low0, high = j ? high1 : high0;
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
            const float d = scale[row] * table_d;
            const float m = scale[row] * (table_m + (Centered ? 8.0f * table_d : 0.0f));
            // Match the parent ISA: round the dot product, then FMA the sum correction.
            total[row] += __builtin_fmaf(m, sx, (d * dx) * float(dot));
        }
    }
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint32_t bits = __builtin_bit_cast(uint32_t, total[row]);
            const uint32_t peer = __builtin_amdgcn_ds_bpermute((lane ^ offset) * 4, bits);
            total[row] += __builtin_bit_cast(float, peer);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            output[(size_t(token) * routes + route) * N + group_n * 4 + row] = total[row];
        }
    }
}

// Short/decode discriminator: one CTA stages the expert LUT for four
// otherwise independent R4 waves. The packet, four per-thread totals, K16
// assignment, reduction, and output values have no format change.
template<int K, int N, bool NativeIU4, bool Centered>
static __global__ __launch_bounds__(128) void c1_vector_r4_lut_runtime(
        const uint8_t * __restrict__ main, const half * __restrict__ rows,
        const half * __restrict__ tables, const C1PackedQ8K32Split * __restrict__ activation,
        const int32_t * __restrict__ ids, float * __restrict__ output,
        int experts, int routes, int input_channels, int ids_token_stride) {
    static_assert(N % 16 == 0);
    constexpr int Groups16 = K / 16, Groups32 = K / 32;
    constexpr int CodeRowBytes = K / 2, MainRowBytes = K / 2 + K / 16;
    constexpr int MainExpertBytes = N * MainRowBytes;
    const unsigned thread = threadIdx.x, lane = thread & 31, wave = thread >> 5;
    const unsigned group_n = blockIdx.x * 4 + wave;
    const unsigned route = blockIdx.y, token = blockIdx.z;
    const int expert = ids[size_t(token) * ids_token_stride + route];
    // Expert is CTA-uniform: every wave either stages its table or returns.
    if (expert < 0 || expert >= experts) return;
    const uint8_t * expert_main = main + size_t(expert) * MainExpertBytes;
    const C1PackedQ8K32Split * x = activation +
        (size_t(token) * input_channels + route % input_channels) * Groups32;
    const half * table = tables + size_t(expert) * 512;
    // Four independent R4 waves cooperate only on the expert LUT. Each
    // thread stages two entries, once per 16-row CTA. This is 2 KiB of LDS.
    struct Coeff { float d, m; };
    __shared__ Coeff lut[256];
#pragma unroll
    for (unsigned part = 0; part < 2; ++part) {
        const unsigned index = thread + part * 128;
        const float d = float(table[2 * index]);
        const float m = float(table[2 * index + 1]);
        lut[index].d = d;
        lut[index].m = m + (Centered ? 8.0f * d : 0.0f);
    }
    __syncthreads();
    float total[4] = {}, scale[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        scale[row] = float(rows[size_t(expert) * N + group_n * 4 + row]);
    }
#pragma clang loop unroll(disable)
    for (unsigned g = lane; g < Groups16; g += 32) {
        const unsigned half_group = g & 1, full_group = g >> 1;
        const C1PackedQ8K32Split & a = x[full_group];
        const uint32_t low0 = a.low[half_group * 2], low1 = a.low[half_group * 2 + 1];
        const uint32_t high0 = a.high[half_group * 2], high1 = a.high[half_group * 2 + 1];
        const float dx = float(a.d), sx = float(half_group ? a.s1 : a.s0);
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint8_t * row_main = expert_main + size_t(group_n * 4 + row) * MainRowBytes;
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t code = c1_load32(row_main + g * 8 + j * 4);
                const uint32_t low = j ? low1 : low0, high = j ? high1 : high0;
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
            const Coeff coeff = lut[unsigned(index)];
            const float d = scale[row] * coeff.d;
            const float m = scale[row] * coeff.m;
            // Match the parent ISA: round the dot product, then FMA the sum correction.
            total[row] += __builtin_fmaf(m, sx, (d * dx) * float(dot));
        }
    }
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint32_t bits = __builtin_bit_cast(uint32_t, total[row]);
            const uint32_t peer = __builtin_amdgcn_ds_bpermute((lane ^ offset) * 4, bits);
            total[row] += __builtin_bit_cast(float, peer);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            output[(size_t(token) * routes + route) * N + group_n * 4 + row] = total[row];
        }
    }
}


static bool c1_use_vector_r4_lut() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_C1_VECTOR_R4_LUT");
        const bool on = value && value[0] == '1';
        if (on) std::fprintf(stderr, "C1 vector R4 LUT enabled (LLAMA_C1_VECTOR_R4_LUT=1)\n");
        return on;
    }();
    return enabled;
}

static bool c1_use_vector_r4() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_C1_VECTOR_R4");
        const bool on = value && value[0] == '1';
        if (on) std::fprintf(stderr, "C1 vector R4 enabled (LLAMA_C1_VECTOR_R4=1)\n");
        return on;
    }();
    return enabled;
}

template<int K, int N, bool NativeIU4, bool Centered>
static void c1_launch_vector(ggml_backend_cuda_context & ctx, dim3 grid,
        const uint8_t * main, const half * rows, const half * tables,
        const C1PackedQ8K32Split * activation, const int32_t * ids, float * output,
        int experts, int routes, int input_channels, int ids_token_stride) {
    if (c1_use_vector_r4_lut()) {
        grid.x /= 16;
        c1_vector_r4_lut_runtime<K, N, NativeIU4, Centered><<<grid, 128, 0, ctx.stream()>>>(
            main, rows, tables, activation, ids, output, experts, routes, input_channels, ids_token_stride);
    } else if (c1_use_vector_r4()) {
        grid.x /= 4;
        c1_vector_r4_runtime<K, N, NativeIU4, Centered><<<grid, 32, 0, ctx.stream()>>>(
            main, rows, tables, activation, ids, output, experts, routes, input_channels, ids_token_stride);
    } else {
        c1_vector_runtime<K, N, NativeIU4, Centered><<<grid, 32, 0, ctx.stream()>>>(
            main, rows, tables, activation, ids, output, experts, routes, input_channels, ids_token_stride);
    }
}

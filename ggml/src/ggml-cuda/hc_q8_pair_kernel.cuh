#include "hc_q8_pair.cuh"

// Standalone device prototype extracted from the current C1 BF16 MMB Q8_0
// arithmetic. Fixed model geometry: Q8_0 K10240, HC down M320, inject M4.
// It adds only virtual row selection and separate destination stores. No
// permanent BF16 weights, packed weight concat, activation conversion, or
// additional downstream HC fusion lives in this translation unit.
namespace {

using v16s = short __attribute__((ext_vector_type(16)));
using v8f  = float __attribute__((ext_vector_type(8)));
constexpr int BK = 64, NT = 256, LDS_STRIDE = BK + 8;
constexpr int K = 10240, MD = 320, MI = 4, M = MD + MI;
constexpr int BM = 128, BN = 128, WTM = 32, WTN = 64;
constexpr int ROW_BYTES = (K / 32) * 34;

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    return (uint32_t)f2bf(a) | ((uint32_t)f2bf(b) << 16);
}
__device__ __forceinline__ float h2f(uint16_t h) {
    return (float)__builtin_bit_cast(_Float16, h);
}

// Exact current mmb_dq_row68 arithmetic: Q8_0 FP16 scale, signed bytes,
// F32 multiply, then BF16 RNE for the WMMA LDS tile.
__device__ __forceinline__ void dq_row68(
        const uint4 w0, const uint4 w1, const uint4 w2,
        const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w,
                             w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = h2f((uint16_t)(ws[0] & 0xffff));
    const float d1 = h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0;
        uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      );
            const float e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16);
            const float e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = pack2(e0, e1);
            out[2*w + 1] = pack2(e2, e3);
        }
    }
}

// Same LDS transpose and F32 WMMA store order as mmb_store_tile. The only
// changed address calculation chooses the original down or injection buffer.
__device__ __forceinline__ void store_tile_pair(
        const v8f & acc, float * __restrict__ stg,
        float * __restrict__ Ydown, float * __restrict__ Yinject,
        int T, int n_base, int m_base, int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) stg[(2 * e + cn) * 16 + cm] = acc[e];
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = n_base + n;
    if (dr < T) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8);
        const float4 v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        if (m_base + 16 <= MD) {
            const size_t base = (size_t)dr * MD + m_base + half * 8;
            *(float4 *)(Ydown + base) = v0;
            *(float4 *)(Ydown + base + 4) = v1;
        } else {
            const float vv[8] = {v0.x, v0.y, v0.z, v0.w,
                                 v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int mr = m_base + half * 8 + j;
                if (mr >= MD && mr < M) Yinject[(size_t)dr * MI + mr - MD] = vv[j];
            }
        }
    }
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(NT, 2) hc_q8_pair_kernel(
        const uint8_t * __restrict__ Wdown,
        const uint8_t * __restrict__ Winject,
        const uint16_t * __restrict__ Xh,
        float * __restrict__ Ydown,
        float * __restrict__ Yinject,
        int T) {
    __shared__ __align__(16) uint16_t As[BM * LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const int a_rows = M - m0;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int B_ITEMS = (BN * 8) / NT;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0, a1, a3, a4;
    uint32_t a2;
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
        const int c = tid + i * NT;
        const int t = t0 + (c >> 3);
        brow[i] = t < T ? t : -1;
    }
    auto load_regs = [&](int ks) {
        const int mr = m0 + tid;
        if (tid < BM && mr < M) {
            const uint8_t * base = mr < MD ? Wdown + (size_t)mr * ROW_BYTES
                                            : Winject + (size_t)(mr - MD) * ROW_BYTES;
            const uint8_t * p = base + (size_t)ks * 68;
            // Q8_0 K64 packets are 68 bytes; every other packet is only
            // four-byte aligned. memcpy avoids a false uint4 alignment claim.
            __builtin_memcpy(&a0, p, 16);
            __builtin_memcpy(&a1, p + 16, 16);
            __builtin_memcpy(&a3, p + 32, 16);
            __builtin_memcpy(&a4, p + 48, 16);
            __builtin_memcpy(&a2, p + 64, 4);
        } else {
            a0 = a1 = a3 = a4 = make_uint4(0, 0, 0, 0);
            a2 = 0;
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * NT;
            const int off = (c & 7) * 8;
            bst[i] = brow[i] >= 0 ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * BK + off)
                                  : make_uint4(0, 0, 0, 0);
        }
    };
    auto store_lds = [&]() {
        if (tid < BM) dq_row68(a0, a1, a3, a4, a2,
                               (uint32_t *)(As + tid * LDS_STRIDE));
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * NT;
            *(uint4 *)(Bs + (c >> 3) * LDS_STRIDE + (c & 7) * 8) = bst[i];
        }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = (wn * WTN + j * 16) < (T - t0);

    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < K / BK; ++ks) {
        if (ks + 1 < K / BK) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            v16s a[TM], b[TN];
            const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const uint16_t * p = As + (wm * WTM + i * 16 + r) * LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8);
                a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1});
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                if (!jact[j]) continue;
                const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8);
                b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1});
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if (!jact[j]) continue;
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]);
                }
        }
        __syncthreads();
        if (ks + 1 < K / BK) store_lds();
        __syncthreads();
    }

    float * stg = (float *)Bs + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16;
        const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) store_tile_pair(acc[i][j], stg, Ydown, Yinject,
                                                T, t0 + wn * WTN + j * 16,
                                                m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

} // namespace

extern "C" hipError_t hc_q8_pair_launch(
        const uint8_t * Wdown, const uint8_t * Winject,
        const uint16_t * Xbf16, float * Ydown, float * Yinject,
        int T, hipStream_t stream) {
    if (!Wdown || !Winject || !Xbf16 || !Ydown || !Yinject || T < 1) return hipErrorInvalidValue;
    const dim3 grid((M + BM - 1) / BM, (T + BN - 1) / BN);
    hipLaunchKernelGGL(hc_q8_pair_kernel, grid, dim3(NT), 0, stream,
                       Wdown, Winject, Xbf16, Ydown, Yinject, T);
    return hipGetLastError();
}

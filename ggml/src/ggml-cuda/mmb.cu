#include "mmb.cuh"
#include "unary.cuh"
#include "dequantize.cuh"
#include <unordered_map>
#include <map>
#include <utility>
#include "mmid.cuh"
#include "hc_q8_pair_kernel.cuh"
#include <cstdlib>
#include <vector>
#include <unordered_set>

namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;

__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) { return (uint32_t)mmb_f2bf(a) | ((uint32_t)mmb_f2bf(b) << 16); }
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
struct mmb_quant_slice {
    uint16_t * dst;
    int begin;
    int offset = 0;
    struct element {
        uint16_t * dst;
        int index;
        __device__ void operator=(float value) const {
            if (index >= 0 && index < 64) dst[index] = mmb_f2bf(value);
        }
    };
    __device__ mmb_quant_slice operator+(int64_t n) const { return {dst, begin, offset + (int) n}; }
    __device__ element operator[](int64_t n) const { return {dst, offset + (int) n - begin}; }
};

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void mmb_halo_dequantize_q5_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    auto y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

// HaloBox 25fda3478 Q5_K tile loader, retaining its eight-lane row decode.
template <int BM>
__device__ __forceinline__ void mmb_halo_q5_tile(const uint8_t * weights, size_t row_bytes, int rows, int ks, uint16_t * tile) {
    for (int row = threadIdx.x / 8; row < BM; row += blockDim.x / 8) {
        uint16_t * dst = tile + row * MMB_LDS_STRIDE;
        const int lane = threadIdx.x % 8;
        if (row < rows) {
            const mmb_quant_slice out{dst, (ks * 64) % QK_K};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int tid = (ks & 3) * 16 + lane + j * 8;
                mmb_halo_dequantize_q5_K<float>(weights + row * row_bytes, ks / 4, out, tid);
            }
        } else {
#pragma unroll
            for (int k = lane; k < 64; k += 8) dst[k] = 0;
        }
    }
}

// Q5_K metadata and codes are prefetched before the current WMMA tile.
template <int MODE = 0>
__device__ __forceinline__ void mmb_halo_q5_store(uint4 h, uint4 q, int il, int lane, uint16_t * dst) {
    const int is = 2 * il;
    const float d = mmb_h2f(uint16_t(h.x)), dm = mmb_h2f(uint16_t(h.x >> 16));
    int sc[2], mn[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int k = is + j;
        if (k < 4) {
            sc[j] = (h.y >> (8*k)) & 63;
            mn[j] = (h.z >> (8*k)) & 63;
        } else {
            const int shift = 8 * (k - 4);
            sc[j] = ((h.w >> shift) & 15) | (((h.y >> shift) >> 6 & 3) << 4);
            mn[j] = ((h.w >> shift) >> 4 & 15) | (((h.z >> shift) >> 6 & 3) << 4);
        }
    }
    const float d0 = d*sc[0], d1 = d*sc[1], m0 = dm*mn[0], m1 = dm*mn[1];
    const uint32_t mask = 1u << (2*il);
#pragma unroll
    for (int seg = 0; seg < 2; ++seg) {
        const uint32_t ql = seg ? q.y : q.x, qh = seg ? q.w : q.z;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int sh = 16 * pair;
            const uint32_t a = ql >> sh, b = qh >> sh;
            const uint32_t lq0 = (a & 15) + ((b & mask) ? 16 : 0);
            const uint32_t lq1 = ((a >> 8) & 15) + (((b >> 8) & mask) ? 16 : 0);
            const uint32_t hq0 = ((a >> 4) & 15) + ((b & (mask << 1)) ? 16 : 0);
            const uint32_t hq1 = ((a >> 12) & 15) + (((b >> 8) & (mask << 1)) ? 16 : 0);
            uint32_t lo, hi;
            if constexpr (MODE == 2) {
                // Exact q in [0,31]: construct q+32, then packed subtract 32.
                const half2 thirtytwo = __builtin_bit_cast(half2, 0x50005000u);
                const half2 lq = __hsub2(__builtin_bit_cast(half2, 0x50005000u | (lq0 << 5) | (lq1 << 21)), thirtytwo);
                const half2 hq = __hsub2(__builtin_bit_cast(half2, 0x50005000u | (hq0 << 5) | (hq1 << 21)), thirtytwo);
                lo = __builtin_bit_cast(uint32_t, __hfma2(__floats2half2_rn(d0,d0), lq, __floats2half2_rn(-m0,-m0)));
                hi = __builtin_bit_cast(uint32_t, __hfma2(__floats2half2_rn(d1,d1), hq, __floats2half2_rn(-m1,-m1)));
            } else {
                const float l0 = d0 * float(lq0) - m0, l1 = d0 * float(lq1) - m0;
                const float h0 = d1 * float(hq0) - m1, h1 = d1 * float(hq1) - m1;
                if constexpr (MODE == 1) {
                    lo = __builtin_bit_cast(uint32_t, __floats2half2_rn(l0,l1));
                    hi = __builtin_bit_cast(uint32_t, __floats2half2_rn(h0,h1));
                } else {
                    lo = mmb_pack2(l0,l1); hi = mmb_pack2(h0,h1);
                }
            }
            const int col = seg*16 + lane*4 + pair*2;
            *(uint32_t *)(dst + col) = lo;
            *(uint32_t *)(dst + col + 32) = hi;
        }
    }
}

// K640 C1 rows are 360 bytes, so alternating rows are only 8-byte aligned.
// Let the compiler lower these 16-byte packet copies without a false uint4
// alignment promise; the 32-bit table-index load remains naturally aligned.
__device__ __forceinline__ uint4 mmb_c1_load16(const uint8_t * p) {
    uint4 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

__device__ __forceinline__ uint2 mmb_c1_load8(const uint8_t * p) {
    uint2 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

__global__ void mmb_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2bf(x[i]);
    }
}

// dequantize one weight row's two consecutive IQ4_NL blocks (36 bytes) into 64 bf16 in LDS.
// LUT held in registers as (kv + 128) bytes and applied with v_perm_b32 (4 nibbles per op pair) instead of a per-lane
// indexed constant array (which lowers to one scalar-byte memory load per element). The value kv*d is produced as
// fma(kv+128, d, -128*d): -128*d is exact, so the single rounding equals RN(kv*d) -> bitwise the same BF16 as before.
__device__ __forceinline__ void mmb_dq_row36(const uint4 w0, const uint4 w1, const uint32_t w2, uint32_t * arow) {
    const uint32_t ws[9] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[4] >> 16));
    const uint32_t q0[4] = { (ws[0] >> 16) | (ws[1] << 16), (ws[1] >> 16) | (ws[2] << 16), (ws[2] >> 16) | (ws[3] << 16), (ws[3] >> 16) | (ws[4] << 16) };
    const uint32_t q1[4] = { ws[5], ws[6], ws[7], ws[8] };
    // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; const float md = -128.0f * d; const uint32_t * q = blk ? q1 : q0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);   // entries 0..7
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);   // entries 8..15
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;      // 0xFF where the nibble >= 8
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w] = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1] = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w] = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// Q4_1: two exact 20-byte GGUF blocks -> 64 BF16 LDS values.
__device__ __forceinline__ void mmb_dq_row40(uint4 w0, uint4 w1, uint2 w2, uint32_t * out) {
    const uint32_t words[10] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y};
#pragma unroll
    for (int block = 0; block < 2; ++block) {
        const int base = block * 5;
        const float d = mmb_h2f(uint16_t(words[base]));
        const float m = mmb_h2f(uint16_t(words[base] >> 16));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = words[base + 1 + j];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint32_t v = (q >> (half * 4)) & 0x0f0f0f0fu;
                const float a = d * float(v & 15) + m;
                const float b = d * float((v >> 8) & 15) + m;
                const float c = d * float((v >> 16) & 15) + m;
                const float e = d * float(v >> 24) + m;
                out[block * 16 + half * 8 + j * 2] = mmb_pack2(a, b);
                out[block * 16 + half * 8 + j * 2 + 1] = mmb_pack2(c, e);
            }
        }
    }
}

// C1: one K64 slice of a frozen row is 32 interleaved 4-bit code bytes and four
// byte indices into the expert's 256-entry FP16 affine table. Row multipliers
// and table pairs are independent tensors. Only this on-chip BF16 tile expands;
// the resident weight packet stays at 4.5175 bpw.
__device__ __forceinline__ void mmb_dq_row_c1(uint4 w0, uint4 w1, uint32_t indices,
        uint16_t row_scale, const uint32_t * __restrict__ table, uint32_t * out) {
    const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    const float r = mmb_h2f(row_scale);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t pair = table[(indices >> (g * 8)) & 255u];
        const float d = mmb_h2f(uint16_t(pair));
        const float m = mmb_h2f(uint16_t(pair >> 16));
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t q = words[g * 2 + w];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t packed = uint8_t(q >> (b * 8));
                const float a = r * fmaf(d, float(packed & 15), m);
                const float z = r * fmaf(d, float(packed >> 4), m);
                out[g * 8 + w * 4 + b] = mmb_pack2(a, z);
            }
        }
    }
}

__device__ __forceinline__ void mmb_dq_group_c1(uint2 packed, uint8_t index,
        uint16_t row_scale, const uint32_t * __restrict__ table, uint32_t * out) {
    const float r = mmb_h2f(row_scale);
    const uint32_t pair = table[index];
    const float d = mmb_h2f(uint16_t(pair));
    const float m = mmb_h2f(uint16_t(pair >> 16));
    const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t q = words[w];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint8_t code = uint8_t(q >> (b * 8));
            const float a = r * fmaf(d, float(code & 15), m);
            const float z = r * fmaf(d, float(code >> 4), m);
            out[w * 4 + b] = mmb_pack2(a, z);
        }
    }
}

// The coefficients are already FP16. Decode two integer levels with one
// packed FP16 FMA and retain the row multiplier for the F32 epilogue.
__device__ __forceinline__ void mmb_dq_group_c1_half(uint2 packed, uint8_t index,
        const uint32_t * table, uint32_t * out) {
    const uint32_t pair=table[index];
    const uint32_t d=pair & 65535u, m=pair >> 16;
    const half2 dd=__builtin_bit_cast(half2,d|(d<<16));
    const half2 mm=__builtin_bit_cast(half2,m|(m<<16));
    const half2 sixteen=__builtin_bit_cast(half2,0x4c004c00u);
#pragma unroll
    for (int b=0;b<8;++b) {
        const uint32_t word=b<4?packed.x:packed.y;
        const uint8_t code=uint8_t(word >> ((b&3)*8));
        const uint32_t bits=0x4c004c00u | (uint32_t(code&15)<<6) | (uint32_t(code>>4)<<22);
        const half2 q=__hsub2(__builtin_bit_cast(half2,bits),sixteen);
        out[b]=__builtin_bit_cast(uint32_t,__hfma2(dd,q,mm));
    }
}
__device__ __forceinline__ uint32_t mmb_bf16_pair_to_half(uint32_t v) {
    return __builtin_bit_cast(uint32_t,__floats2half2_rn(__uint_as_float(v<<16),__uint_as_float(v&0xffff0000u)));
}

__global__ void mmb_cvt_bf16_f16(const uint16_t * __restrict__ x, uint16_t * __restrict__ y, size_t n) {
    size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const uint4 v = *(const uint4 *)(x+i);
        const uint4 o = make_uint4(mmb_bf16_pair_to_half(v.x), mmb_bf16_pair_to_half(v.y),
                                  mmb_bf16_pair_to_half(v.z), mmb_bf16_pair_to_half(v.w));
        *(uint4 *)(y+i) = o;
    } else {
        for (; i<n; ++i) y[i] = __builtin_bit_cast(uint16_t, __float2half_rn(__uint_as_float(uint32_t(x[i]) << 16)));
    }
}

// dequantize one weight row's two consecutive Q8_0 blocks (68 bytes: d0 qs0[32] d1 qs1[32]) into 64 bf16 in LDS
__device__ __forceinline__ void mmb_dq_row68(const uint4 w0, const uint4 w1, const uint4 w2, const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w, w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      ), e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16), e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = mmb_pack2(e0, e1); out[2*w + 1] = mmb_pack2(e2, e3);
        }
    }
}

template <typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
        }
    }
    __syncthreads();
}
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs,
        const uint16_t * c1_rows = nullptr, const uint32_t * c1_table = nullptr) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS], a5[A_ITEMS], a6[A_ITEMS], a7[A_ITEMS], a8[A_ITEMS]; uint32_t a2[A_ITEMS];
    constexpr int Q5_ITEMS = (BM + MMB_NT/4 - 1) / (MMB_NT/4);
    uint4 q5_head[Q5_ITEMS], q5_codes[Q5_ITEMS];
    uint16_t c1_scale[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
    if constexpr (WTYPE == 4) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            c1_scale[i] = row < BM && row < a_rows ? c1_rows[row] : 0;
        }
    }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }

    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (WTYPE >= 5 && WTYPE <= 8) {
#pragma unroll
            for (int i = 0; i < Q5_ITEMS; ++i) {
                const int row = tid / 4 + i * (MMB_NT / 4), lane4 = tid & 3;
                if (row < BM && row < a_rows) {
                    const uint8_t * p = Wbase + size_t(row)*wrow_bytes + size_t(ks/4)*sizeof(block_q5_K);
                    q5_head[i] = *(const uint4 *)p;
                    const uint8_t * ql = p + 48 + 32*(ks & 3) + lane4*4;
                    const uint8_t * qh = p + 16 + lane4*4;
                    q5_codes[i] = make_uint4(*(const uint32_t *)ql, *(const uint32_t *)(ql+16),
                                            *(const uint32_t *)qh, *(const uint32_t *)(qh+16));
                } else {
                    q5_head[i] = make_uint4(0,0,0,0);
                    q5_codes[i] = make_uint4(0,0,0,0);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a2[i] = *(const uint32_t *)(p + 32); }
                else if constexpr (WTYPE == 3) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 40;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16);
                    const uint2 tail = *(const uint2 *)(p + 32); a3[i] = make_uint4(tail.x, tail.y, 0, 0); }
                else if constexpr (WTYPE == 4) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes;
                    a0[i] = mmb_c1_load16(p + (size_t)ks * 32); a1[i] = mmb_c1_load16(p + (size_t)ks * 32 + 16);
                    a2[i] = *(const uint32_t *)(p + K / 2 + (size_t)ks * 4); }
                else if constexpr (WTYPE == 1) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a3[i] = *(const uint4 *)(p + 32); a4[i] = *(const uint4 *)(p + 48); a2[i] = *(const uint32_t *)(p + 64); }
                else if constexpr (WTYPE == 2) { const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2]; a4[i] = p[3]; a5[i] = p[4]; a6[i] = p[5]; a7[i] = p[6]; a8[i] = p[7]; }
            } else { a0[i] = make_uint4(0,0,0,0); a1[i] = make_uint4(0,0,0,0); a3[i] = make_uint4(0,0,0,0); a4[i] = make_uint4(0,0,0,0); a5[i] = a6[i] = a7[i] = a8[i] = make_uint4(0,0,0,0); a2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE >= 5 && WTYPE <= 8) {
#pragma unroll
            for (int i = 0; i < Q5_ITEMS; ++i) {
                const int row = tid / 4 + i * (MMB_NT / 4);
                if (row < BM) mmb_halo_q5_store<WTYPE == 8 ? 2 : WTYPE - 5>(q5_head[i], q5_codes[i], weight_ks & 3, tid & 3,
                                               As + row*MMB_LDS_STRIDE);
            }
        }
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0[i], a1[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 3) mmb_dq_row40(a0[i], a1[i], make_uint2(a3[i].x, a3[i].y), (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 4) mmb_dq_row_c1(a0[i], a1[i], a2[i], c1_scale[i], c1_table, (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 1) mmb_dq_row68(a0[i], a1[i], a3[i], a4[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 2) { uint4 * d = (uint4 *)(As + row * MMB_LDS_STRIDE); d[0] = a0[i]; d[1] = a1[i]; d[2] = a3[i]; d[3] = a4[i]; d[4] = a5[i]; d[5] = a6[i]; d[6] = a7[i]; d[7] = a8[i]; } } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT;
            uint4 v = bst[i];
            if constexpr (WTYPE == 8) {
                v = make_uint4(mmb_bf16_pair_to_half(v.x), mmb_bf16_pair_to_half(v.y),
                               mmb_bf16_pair_to_half(v.z), mmb_bf16_pair_to_half(v.w));
            }
            *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = v;
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
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;   // whole fragments past the valid columns are never stored
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } if constexpr (WTYPE >= 6 && WTYPE <= 8)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a[i], acc[i][j]);
                else acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]); }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue through LDS (per-wave 1 KB stage in the now-free A/B tile area); tiles are 16 rows, a_rows is a
    // multiple of 32 in this model, so whole tiles are either valid or beyond a_rows
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : As) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmb_store_tile(acc[i][j], stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = (WTYPE >= 5 && WTYPE <= 8) ? (size_t)(K / QK_K) * sizeof(block_q5_K) : WTYPE == 2 ? (size_t) K * 2 : (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, false>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, Dh, store_f32, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs);
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif
__device__ __forceinline__ float gm_sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float gm_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }

template <int HC, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
hc_gate_mix_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Lo, const uint16_t * __restrict__ Xn, float * __restrict__ Out,
        uint16_t * __restrict__ OutH, const bool store_f32,
        const int E, const int K, const int T, const float scale, const float bias) {
    constexpr int CH = 32, BN = 128, BM = HC * CH;
    static_assert(BM <= MMB_NT, "one A row per thread");
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 1, wn = wave >> 1;                 // wave: 16 channels (all HC streams) x 32 tokens
    const int e0 = blockIdx.x * CH, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    uint4 a0 = make_uint4(0,0,0,0), a1 = make_uint4(0,0,0,0); uint32_t a2 = 0; uint4 a3 = make_uint4(0,0,0,0), a4 = make_uint4(0,0,0,0); uint4 bst[B_ITEMS]; int brow[B_ITEMS];
    const uint8_t * arow = W;
    if (tid < BM) { const int c = tid / CH, i = tid - c * CH; arow = W + (size_t)(c * E + e0 + i) * wrow_bytes; }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = t0 + (c >> 3); brow[i] = t < T ? t : -1; }
    auto load_regs = [&](const int ks) {
        if (tid < BM) {
            const uint8_t * p = arow + (size_t)ks * (WTYPE == 0 ? 36 : 68);
            a0 = *(const uint4 *)(p); a1 = *(const uint4 *)(p + 16);
            if constexpr (WTYPE == 0) { a2 = *(const uint32_t *)(p + 32); }
            else { a3 = *(const uint4 *)(p + 32); a4 = *(const uint4 *)(p + 48); a2 = *(const uint32_t *)(p + 64); }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Lo + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0); }
    };
    auto store_lds = [&]() {
        if (tid < BM) {
            if constexpr (WTYPE == 0) { mmb_dq_row36(a0, a1, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE)); }
            else { mmb_dq_row68(a0, a1, a3, a4, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE)); }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f acc[HC][2];
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[c][j][e] = 0.f;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[HC], b[2]; const int r = lane & 15;
#pragma unroll
            for (int c = 0; c < HC; ++c) { const uint16_t * p = As + (c * CH + wm * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[c] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < 2; ++j) { const uint16_t * p = Bs + (wn * 32 + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int c = 0; c < HC; ++c)
#pragma unroll
                for (int j = 0; j < 2; ++j) acc[c][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[c], acc[c][j]);
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue: lane holds channel (lane & 15) of the wave's 16 and tokens 2e + (lane >> 4) of each 16-token fragment
    const int cm = lane & 15, cn = lane >> 4; const int ch = e0 + wm * 16 + cm;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = t0 + wn * 32 + j * 16 + 2 * e + cn;
            if (t >= T) continue;
            const uint16_t * xr = Xn + (size_t)t * ((size_t)HC * E) + ch;
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float g = __uint_as_float(((uint32_t) mmb_f2bf(acc[c][j][e])) << 16);   // the gate GEMM's BF16 epilogue rounding
                const float term = gm_mul_rn(gm_bf2f(xr[(size_t)c * E]), gm_sigmoid(g));
                s = (c == 0) ? term : gm_add_rn(s, term);
            }
            const float o = scale * s + bias;
            if (store_f32) Out[(size_t)t * E + ch] = o;
            if (OutH) OutH[(size_t)t * E + ch] = mmb_f2bf(o);
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;   // uniform across the block, before any barrier
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 3 ? 20 : 18);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, true>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
}

template <int BM, int BN, int WTM, int WTN>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_c1_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes,
        const uint16_t * __restrict__ rows, const uint32_t * __restrict__ tables,
        const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint32_t table[256];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    table[threadIdx.x] = tables[(size_t)e * 256 + threadIdx.x];
    __syncthreads();
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (size_t)K / 2 + (size_t)K / 16;
    mmb_tile_gemm<BM, BN, WTM, WTN, 4, true>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes,
        wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M,
        [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs,
        rows + (size_t)e * M + m0, table);
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, bool C1_GROUP4, bool C1_HALF = false, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs,
        const uint16_t * c1_grows = nullptr, const uint16_t * c1_urows = nullptr,
        const uint32_t * c1_gtable = nullptr, const uint32_t * c1_utable = nullptr) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = ((C1_GROUP4 ? BM * 4 : BM) + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int part = C1_GROUP4 ? tid & 3 : 0;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS], g3[A_ITEMS], u3[A_ITEMS];
    uint16_t c1_gscale[A_ITEMS], c1_uscale[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
    if constexpr (WTYPE == 4 && !C1_HALF) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = C1_GROUP4 ? (tid + i * MMB_NT) / 4 : tid + i * MMB_NT;
            c1_gscale[i] = row < BM && row < a_rows ? c1_grows[row] : 0;
            c1_uscale[i] = row < BM && row < a_rows ? c1_urows[row] : 0;
        }
    }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    auto load_regs = [&](const int ks) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = C1_GROUP4 ? (tid + i * MMB_NT) / 4 : tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 4) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes;
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes;
                    if constexpr (C1_GROUP4) {
                        const uint2 gp = mmb_c1_load8(pg + (size_t)ks * 32 + part * 8);
                        const uint2 up = mmb_c1_load8(pu + (size_t)ks * 32 + part * 8);
                        g0[i] = make_uint4(gp.x, gp.y, 0, 0);
                        u0[i] = make_uint4(up.x, up.y, 0, 0);
                        g2[i] = *(pg + K / 2 + (size_t)ks * 4 + part);
                        u2[i] = *(pu + K / 2 + (size_t)ks * 4 + part);
                    } else {
                        g0[i] = mmb_c1_load16(pg + (size_t)ks * 32); g1[i] = mmb_c1_load16(pg + (size_t)ks * 32 + 16);
                        u0[i] = mmb_c1_load16(pu + (size_t)ks * 32); u1[i] = mmb_c1_load16(pu + (size_t)ks * 32 + 16);
                        g2[i] = *(const uint32_t *)(pg + K / 2 + (size_t)ks * 4);
                        u2[i] = *(const uint32_t *)(pu + K / 2 + (size_t)ks * 4);
                    }
                } else {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * (WTYPE == 3 ? 40 : 36);
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * (WTYPE == 3 ? 40 : 36);
                    g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                    u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
                    if constexpr (WTYPE == 3) { g3[i] = *(const uint32_t *)(pg + 36); u3[i] = *(const uint32_t *)(pu + 36); }
                }
            } else { g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = g3[i] = u3[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = C1_GROUP4 ? (tid + i * MMB_NT) / 4 : tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 3) {
                mmb_dq_row40(g0[i], g1[i], make_uint2(g2[i], g3[i]), (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                mmb_dq_row40(u0[i], u1[i], make_uint2(u2[i], u3[i]), (uint32_t *)(Au + row * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 4) {
                if constexpr (C1_HALF) {
                    static_assert(C1_GROUP4 && WTYPE == 4);
                    mmb_dq_group_c1_half(make_uint2(g0[i].x,g0[i].y),uint8_t(g2[i]),c1_gtable,(uint32_t *)(Ag+row*MMB_LDS_STRIDE)+part*8);
                    mmb_dq_group_c1_half(make_uint2(u0[i].x,u0[i].y),uint8_t(u2[i]),c1_utable,(uint32_t *)(Au+row*MMB_LDS_STRIDE)+part*8);
                } else if constexpr (C1_GROUP4) {
                    mmb_dq_group_c1(make_uint2(g0[i].x, g0[i].y), (uint8_t)g2[i], c1_gscale[i], c1_gtable, (uint32_t *)(Ag + row * MMB_LDS_STRIDE) + part * 8);
                    mmb_dq_group_c1(make_uint2(u0[i].x, u0[i].y), (uint8_t)u2[i], c1_uscale[i], c1_utable, (uint32_t *)(Au + row * MMB_LDS_STRIDE) + part * 8);
                } else {
                    mmb_dq_row_c1(g0[i], g1[i], g2[i], c1_gscale[i], c1_gtable, (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_row_c1(u0[i], u1[i], u2[i], c1_uscale[i], c1_utable, (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                }
            } else {
                mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + row * MMB_LDS_STRIDE));
            } } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT;
            uint4 v=bst[i];
            if constexpr (C1_HALF) v=make_uint4(mmb_bf16_pair_to_half(v.x),mmb_bf16_pair_to_half(v.y),mmb_bf16_pair_to_half(v.z),mmb_bf16_pair_to_half(v.w));
            *(uint4 *)(Bs + (c >> 3)*MMB_LDS_STRIDE + (c & 7)*8)=v;
        }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (TAIL) { if (!jact[j]) continue; }
                    if constexpr (C1_HALF) {
                        accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], ag[i], accg[i][j]);
                        accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], au[i], accu[i][j]);
                    } else {
                        accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                        accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                    }
                }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
                const float gr=C1_HALF?mmb_h2f(c1_grows[ml+(lane&15)]):1.0f;
                const float ur=C1_HALF?mmb_h2f(c1_urows[ml+(lane&15)]):1.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(gr*accg[i][j][e]) * (ur*accu[i][j][e]); }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 3 ? 20 : 18);
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, WTYPE, true, false>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
}

template <int BM, int BN, int WTM, int WTN, bool C1_GROUP4, bool C1_HALF = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_c1_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu,
        const size_t expert_bytes, const uint16_t * __restrict__ grows, const uint16_t * __restrict__ urows,
        const uint32_t * __restrict__ gtables, const uint32_t * __restrict__ utables,
        const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint32_t gtable[256], utable[256];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    gtable[threadIdx.x] = gtables[(size_t)e * 256 + threadIdx.x];
    utable[threadIdx.x] = utables[(size_t)e * 256 + threadIdx.x];
    __syncthreads();
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (size_t)K / 2 + (size_t)K / 16;
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, 4, true, C1_GROUP4, C1_HALF>(
        Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes,
        Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes,
        wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M,
        [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs,
        grows + (size_t)e * M + m0, urows + (size_t)e * M + m0, gtable, utable);
}

// F32 x F32 -> F32 GEMM with F32-equivalent precision on WMMA: each operand is split into F16 hi + F16 lo at tile load and
// the product is accumulated as hi*hi + hi*lo + lo*hi in F32 (lo*lo ~2^-22 relative, dropped). Used for the F32 MoE router.
__device__ __forceinline__ void mmb_split2(float x, uint16_t & hi, uint16_t & lo) {
    hi = __builtin_bit_cast(uint16_t, (_Float16) x); lo = __builtin_bit_cast(uint16_t, (_Float16) (x - (float) __builtin_bit_cast(_Float16, hi)));
}
template <int BM, int BN, int WTM, int WTN, bool TWO>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32split_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T) {
    constexpr int BKs = 32, LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    __shared__ __align__(16) uint16_t Ah[BM * LS], Al[BM * LS], Bh[BN * LS], Bl[BN * LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    constexpr int A_CH = BM * BKs / 4, B_CH = BN * BKs / 4;
    for (int k0 = 0; k0 < K; k0 += BKs) {
        for (int idx = tid; idx < A_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int m = m0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f); if (m < M) v = *(const float4 *)(W + (size_t) m * K + k0 + c4);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Al + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        for (int idx = tid; idx < B_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int t = t0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f); if (t < T) v = *(const float4 *)(X + (size_t) t * K + k0 + c4);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        __syncthreads();
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], al[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)});
                al[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Al + off), *(const uint4 *)(Al + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                    if constexpr (!TWO) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], al[i], acc[i][j]);
                }
        }
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) D[(size_t) t * M + m] = acc[i][j][e]; } }
}

// HaloBox prefill kernel (c750386bb); use its published narrow default tiles.
template <int BM, int BN, int WTM, int WTN, bool TWO, bool XSPLIT>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32_halo_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T,
        const float * __restrict__ W2 = nullptr, float * __restrict__ D2 = nullptr, const int M1 = 0) {
    // W2 != nullptr: two GEMMs on the same X, rows [0, M1) from W -> D (ld M1) and rows [M1, M) from W2 -> D2 (ld M - M1)
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    // WMMA (wave32, 16x16x16 bf16/f16 into f32) exists only on RDNA3; the host gate keeps other devices off this path
    NO_DEVICE_CODE;
#else
    constexpr int BKs = 32, LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    static_assert(WAVES_M * (BN / WTN) * 32 == MMB_NT, "tile must use every wave");
    __shared__ __align__(16) uint16_t Ah[BM * LS], Al[TWO ? 8 : BM * LS], Bh[BN * LS], Bl[XSPLIT ? BN * LS : 8];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    constexpr int A_CH = BM * BKs / 4, B_CH = BN * BKs / 4;
    // one register stage: the next K slice is fetched while the current one is multiplied (unguarded loads with
    // clamped rows, zeroed at the LDS store: a divergent guard would make the compiler drain them at the join)
    constexpr int A_IT = (A_CH + MMB_NT - 1) / MMB_NT, B_IT = (B_CH + MMB_NT - 1) / MMB_NT;
    static_assert(A_CH % MMB_NT == 0 && B_CH % MMB_NT == 0, "f32split staging assumes whole passes");
    float4 ra[A_IT], rb[B_IT];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < A_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const int gr = min(m0 + row, M - 1);
            const float * wr = (W2 && gr >= M1) ? W2 + (size_t) (gr - M1) * K : W + (size_t) gr * K;
            ra[i] = *(const float4 *)(wr + k0 + c4); }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            rb[i] = *(const float4 *)(X + (size_t) min(t0 + row, T - 1) * K + k0 + c4); }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < A_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const float4 v = m0 + row < M ? ra[i] : make_float4(0.f,0.f,0.f,0.f);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            if constexpr (!TWO) *(uint2 *)(Al + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const float4 v = t0 + row < T ? rb[i] : make_float4(0.f,0.f,0.f,0.f);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            if constexpr (XSPLIT) *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += BKs) {
        lstore();
        __syncthreads();
        gload(min(k0 + BKs, K - BKs));
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], al[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)});
                if constexpr (!TWO) al[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Al + off), *(const uint4 *)(Al + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                if constexpr (XSPLIT) bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    if constexpr (XSPLIT) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                    if constexpr (!TWO) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], al[i], acc[i][j]);
                }
        }
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) {
                if (W2) { if (m < M1) D[(size_t) t * M1 + m] = acc[i][j][e]; else D2[(size_t) t * (M - M1) + (m - M1)] = acc[i][j][e]; }
                else D[(size_t) t * M + m] = acc[i][j][e]; } } }
#endif
}

// two tile classes: experts with >= thresh rows get BN_BIG-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows on tiny experts)
__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

struct mmb_cache_entry { const ggml_tensor * root; const void * data; size_t n; ggml_cuda_pool_alloc<uint16_t> * buf; };
static std::vector<mmb_cache_entry> g_mmb_cache;
static mmb_cache_entry g_mmb_slots[4] = {{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr}};
static std::unordered_set<const ggml_tensor *> g_mmb_bf16_only;

static size_t mmb_cache_max() { static const int v = getenv("LLAMA_MMB_CACHE") ? atoi(getenv("LLAMA_MMB_CACHE")) : 4; return (size_t) v; }
static const ggml_tensor * mmb_root(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }
static uint16_t * mmb_cache_insert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const size_t n) {
    if (g_mmb_cache.size() >= mmb_cache_max()) { delete g_mmb_cache.front().buf; g_mmb_cache.erase(g_mmb_cache.begin()); }
    auto * buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n);
    g_mmb_cache.push_back({mmb_root(t), t->data, n, buf});
    return buf->get();
}
static const uint16_t * mmb_bf16_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream) {
    const ggml_tensor * root = mmb_root(src1);
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == src1->data && e.n == n) return e.buf->get();
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == src1->data && e.n == n) return e.buf->get();
    uint16_t * buf = mmb_cache_insert(ctx, src1, n);
    { static const int lg = getenv("LLAMA_MMB_CVT_LOG") ? atoi(getenv("LLAMA_MMB_CVT_LOG")) : 0; static unsigned cnt = 0;
      if (lg && cnt++ < 200) fprintf(stderr, "MMB_CVT %s op=%s ne=[%lld,%lld,%lld,%lld] view_src=%s n=%zu\n", src1->name, ggml_op_name(src1->op), (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3], src1->view_src ? src1->view_src->name : "-", n); }
    mmb_cvt_f32_bf16<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, n);
    return buf;
}

// Shadow BF16 copies of IQ4_NL dense weights: dequantised once (same LUT*scale -> BF16 RNE as mmb_dq_row36, so the
// WMMA inputs are bitwise identical) so the dense GEMM runs the dequant-free WTYPE=2 path.
__global__ void mmb_dq_q6k_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p  = W + b * 210;
    const uint8_t * ql = p, * qh = p + 128;
    const int8_t  * sc = (const int8_t *) (p + 192);
    const float d = mmb_h2f(*(const uint16_t *) (p + 208));
    uint16_t * o = out + b * 256;
    for (int n = 0; n < 2; ++n) {
        const uint8_t * QL = ql + 64 * n; const uint8_t * QH = qh + 32 * n; const int8_t * S = sc + 8 * n; uint16_t * Y = o + 128 * n;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((QL[l +  0] & 0xF) | (((QH[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((QL[l +  0] >>  4) | (((QH[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4)) - 32;
            Y[l +  0] = mmb_f2bf(d * S[is + 0] * q1);
            Y[l + 32] = mmb_f2bf(d * S[is + 2] * q2);
            Y[l + 64] = mmb_f2bf(d * S[is + 4] * q3);
            Y[l + 96] = mmb_f2bf(d * S[is + 6] * q4);
        }
    }
}

__global__ void mmb_dq_iq4nl_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p = W + b * 18;
    const float d = mmb_h2f(*(const uint16_t *) p);
    const uint32_t * q = (const uint32_t *) (p + 2);
    uint32_t * o = (uint32_t *) (out + b * 32);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = q[w];
        const float l0 = d * mmb_kv_iq4nl[(v      ) & 0xF], h0 = d * mmb_kv_iq4nl[(v >>  4) & 0xF];
        const float l1 = d * mmb_kv_iq4nl[(v >>  8) & 0xF], h1 = d * mmb_kv_iq4nl[(v >> 12) & 0xF];
        const float l2 = d * mmb_kv_iq4nl[(v >> 16) & 0xF], h2 = d * mmb_kv_iq4nl[(v >> 20) & 0xF];
        const float l3 = d * mmb_kv_iq4nl[(v >> 24) & 0xF], h3 = d * mmb_kv_iq4nl[(v >> 28) & 0xF];
        o[2*w] = mmb_pack2(l0, l1); o[2*w + 1] = mmb_pack2(l2, l3); o[8 + 2*w] = mmb_pack2(h0, h1); o[8 + 2*w + 1] = mmb_pack2(h2, h3);
    }
}
static std::unordered_map<const void *, uint16_t *> g_mmb_shadow;
static std::map<std::pair<const void *, const void *>, uint16_t *> g_mmb_shadow_pair;   // concat(w0, w1) along rows -> BF16 copy
static size_t g_mmb_shadow_bytes = 0;
int    mmb_shadow_mode(){ static const int v = getenv("LLAMA_MMB_SHADOW") ? atoi(getenv("LLAMA_MMB_SHADOW")) : 0; return v; }
bool   mmb_shadow()    { return mmb_shadow_mode() != 0; }
bool   mmb_shadow_q6k(){ return mmb_shadow_mode() >= 1; }
size_t mmb_shadow_cap(){ static const long v = getenv("LLAMA_MMB_SHADOW_MB") ? atol(getenv("LLAMA_MMB_SHADOW_MB")) : 6144; return (size_t) v << 20; }
static bool mmb_is_resident_q6k(const ggml_tensor * w) { return w && w->type == GGML_TYPE_Q6_K && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->ne[0] % 256 == 0 && w->ne[1] <= 32768; }
static bool mmb_is_resident_iq4(const ggml_tensor * w) { return w && w->type == GGML_TYPE_IQ4_NL && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w); }
static bool mmb_is_row_concat(const ggml_tensor * w) {
    return w && w->op == GGML_OP_CONCAT && w->type == GGML_TYPE_IQ4_NL && ggml_get_op_params_i32(w, 0) == 1 && mmb_is_resident_iq4(w->src[0]) && mmb_is_resident_iq4(w->src[1]) &&
           w->src[0]->ne[0] == w->src[1]->ne[0] && w->ne[0] == w->src[0]->ne[0] && w->ne[1] == w->src[0]->ne[1] + w->src[1]->ne[1];
}
static const uint16_t * mmb_shadow_lookup(const ggml_tensor * w) {
    if (w->op == GGML_OP_CONCAT) { auto it = g_mmb_shadow_pair.find({w->src[0]->data, w->src[1]->data}); return it == g_mmb_shadow_pair.end() ? nullptr : it->second; }
    auto it = g_mmb_shadow.find(w->data); return it == g_mmb_shadow.end() ? nullptr : it->second;
}

bool mmb_enabled() { static const int v = getenv("LLAMA_MMB") ? atoi(getenv("LLAMA_MMB")) : 0; return v != 0; }
int  mmb_min_t()   { static const int v = getenv("LLAMA_MMB_MIN_T") ? atoi(getenv("LLAMA_MMB_MIN_T")) : 512; return v; }
int  mmb_f32split_mode(){ static const int v = getenv("LLAMA_MMB_F32SPLIT") ? atoi(getenv("LLAMA_MMB_F32SPLIT")) : 0; return v; }
bool mmb_f32split() { static const int v = getenv("LLAMA_MMB_F32SPLIT") ? atoi(getenv("LLAMA_MMB_F32SPLIT")) : 0; return v != 0; }
bool mmb_bf16w()    { static const int v = getenv("LLAMA_MMB_BF16W") ? atoi(getenv("LLAMA_MMB_BF16W")) : 0; return v != 0; }
bool mmb_hc16()    { static const int v = getenv("LLAMA_MMB_HC16") ? atoi(getenv("LLAMA_MMB_HC16")) : 0; return v != 0; }
int  mmb_tall_mode(){ static const int v = getenv("LLAMA_MMB_TALL") ? atoi(getenv("LLAMA_MMB_TALL")) : 0; return v; }
bool mmb_tall()    { return mmb_tall_mode() != 0; }
bool mmb_gatemix_flag() { static const int v = getenv("LLAMA_HC_GATEMIX") ? atoi(getenv("LLAMA_HC_GATEMIX")) : 0; return v != 0; }
bool mmb_down16_flag() { static const int v = getenv("LLAMA_MMB_DOWN16") ? atoi(getenv("LLAMA_MMB_DOWN16")) : 0; return v != 0; }
bool mmb_glu()     { static const int v = getenv("LLAMA_MMB_GLU") ? atoi(getenv("LLAMA_MMB_GLU")) : 0; return v != 0; }

} // namespace

const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t) {
    const ggml_tensor * root = mmb_root(t);
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == t->data) return e.buf->get();
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == t->data) return e.buf->get();
    return nullptr;
}
static size_t g_mmb_slot_cap[4] = {0, 0, 0};
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n) {
    mmb_cache_entry & e = g_mmb_slots[slot];
    if (e.buf && g_mmb_slot_cap[slot] < n) { delete e.buf; e.buf = nullptr; }
    if (!e.buf) { e.buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n); g_mmb_slot_cap[slot] = n; }
    e.root = mmb_root(t); e.data = t->data; e.n = n;
    return e.buf->get();
}
void ggml_cuda_mmb_marks_clear() { g_mmb_bf16_only.clear(); }
size_t ggml_cuda_mmb_marks_count() { return g_mmb_bf16_only.size(); }
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t) { g_mmb_bf16_only.insert(t); }
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t) { return g_mmb_bf16_only.count(t) > 0; }
void ggml_cuda_mmb_begin_graph() { for (auto & e : g_mmb_cache) delete e.buf; g_mmb_cache.clear(); for (auto & e : g_mmb_slots) { e.root = nullptr; e.data = nullptr; e.n = 0; } }
void ggml_cuda_mmb_release_all() {
    ggml_cuda_mmb_begin_graph();
    for (int i = 0; i < 4; ++i) { if (g_mmb_slots[i].buf) delete g_mmb_slots[i].buf; g_mmb_slots[i].buf = nullptr; g_mmb_slot_cap[i] = 0; }
    // the shadow weights are raw cudaMalloc, keyed by data pointer and held for the life of the
    // process. A model has finitely many weights so this never mattered, but a long-lived process
    // that sees many distinct tensors (test-backend-ops) keeps every one of them.
    for (auto & e : g_mmb_shadow)      { if (e.second) cudaFree(e.second); }
    for (auto & e : g_mmb_shadow_pair) { if (e.second) cudaFree(e.second); }
    g_mmb_shadow.clear();
    g_mmb_shadow_pair.clear();
    g_mmb_shadow_bytes = 0;
}
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n) {
    if (!mmb_enabled() || ggml_nrows(t) < mmb_min_t()) return nullptr;
    return ggml_cuda_mmb_slot_reserve(ctx, 0, t, n);
}

static bool mmb_halo_q5_enabled() {
    static const bool enabled = getenv("CIRU_HALO_Q5_MMB") && atoi(getenv("CIRU_HALO_Q5_MMB")) != 0;
    return enabled;
}

bool ggml_cuda_mmb_supported_mm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmb_enabled()) return false;
    const bool quant = src0->type == GGML_TYPE_IQ4_NL || src0->type == GGML_TYPE_Q8_0 ||
                       (mmb_halo_q5_enabled() && src0->type == GGML_TYPE_Q5_K && src0->ne[0] % QK_K == 0) ||
                       (src0->type == GGML_TYPE_Q6_K && mmb_shadow_q6k() && mmb_is_resident_q6k(src0) &&
                        mmb_shadow_lookup(src0) != nullptr);
    const bool bf16w = src0->type == GGML_TYPE_BF16 && mmb_bf16w();
    const bool f32w  = src0->type == GGML_TYPE_F32 && mmb_f32split();
    if ((!quant && !bf16w && !f32w) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if ((f32w ? K % 32 : K % 64) != 0 || src1->ne[0] != K || dst->ne[0] != M) return false;
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (T < mmb_min_t() || T > INT32_MAX / 4) return false;
    return ggml_nrows(dst) == T;
}

bool ggml_cuda_mmb_hc_q8_pair(ggml_backend_cuda_context & ctx,
        const ggml_tensor * down, const ggml_tensor * inject) {
    if (!down || !inject || down->op != GGML_OP_MUL_MAT || inject->op != GGML_OP_MUL_MAT ||
            down->src[1] != inject->src[1] || !down->src[1]) return false;
    const ggml_tensor * xn = down->src[1];
    const ggml_tensor * wd = down->src[0];
    const ggml_tensor * wi = inject->src[0];
    if (!wd || !wi || wd->type != GGML_TYPE_Q8_0 || wi->type != GGML_TYPE_Q8_0 ||
            wd->ne[0] != 10240 || wd->ne[1] != 320 || wi->ne[0] != 10240 || wi->ne[1] != 4 ||
            !ggml_cuda_mmb_supported_mm(wd, xn, down) ||
            !ggml_cuda_mmb_supported_mm(wi, xn, inject) ||
            ggml_cuda_mmb_is_bf16_only(down) || ggml_cuda_mmb_is_bf16_only(inject)) return false;
    const int T = (int)(xn->ne[1] * xn->ne[2] * xn->ne[3]);
    if (T < 4096 || T > INT32_MAX / 4 || down->ne[0] != 320 || inject->ne[0] != 4 ||
            ggml_nrows(down) != T || ggml_nrows(inject) != T) return false;
    cudaStream_t stream = ctx.stream();
    const uint16_t * xhp = mmb_bf16_activation(ctx, xn, (size_t)T * 10240, stream);
    CUDA_CHECK(hc_q8_pair_launch((const uint8_t *)wd->data, (const uint8_t *)wi->data,
            xhp, (float *)down->data, (float *)inject->data, T, stream));
    static unsigned logged = 0;
    if (logged++ < 12) fprintf(stderr, "HC_Q8_PAIR K=10240 M=320+4 T=%d\n", T);
    return true;
}

static bool mmb_c1_enabled() {
    static const bool enabled = getenv("LLAMA_C1_MMB") && atoi(getenv("LLAMA_C1_MMB")) != 0;
    return enabled;
}
static int mmb_c1_min_t() {
    static const int requested = getenv("LLAMA_C1_MMB_MIN_T") ? atoi(getenv("LLAMA_C1_MMB_MIN_T")) : 4096;
    const int floor = requested < 512 ? 512 : requested;
    return floor > mmb_min_t() ? floor : mmb_min_t();
}
static bool mmb_c1_aux_valid(const ggml_tensor * w, const ggml_tensor * dst) {
    if (!w || !dst || dst->src[0] != w) return false;
    const ggml_tensor * rows = dst->src[3], * tables = dst->src[4];
    if (!rows || !tables || rows->type != GGML_TYPE_F16 || tables->type != GGML_TYPE_F16) return false;
    if (!ggml_is_contiguous(rows) || !ggml_is_contiguous(tables)) return false;
    const int64_t K = w->ne[0], M = w->ne[1], E = w->ne[2];
    if (w->nb[1] != (size_t)(K / 2 + K / 16)) return false;
    if (rows->ne[0] != M || rows->ne[1] != E || rows->ne[2] != 1 || rows->ne[3] != 1) return false;
    if (tables->ne[0] != 2 || tables->ne[1] != 256 || tables->ne[2] != E || tables->ne[3] != 1) return false;
    return true;
}

bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!mmb_enabled()) return false;
    const bool c1 = src0->type == GGML_TYPE_C1_G16_K2560 || src0->type == GGML_TYPE_C1_G16_K640;
    if (c1 && (!mmb_c1_enabled() || !mmb_c1_aux_valid(src0, dst))) return false;
    if ((!c1 && src0->type != GGML_TYPE_IQ4_NL && src0->type != GGML_TYPE_Q4_1) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    if (src0->ne[3] != 1 || K % 64 != 0 || E < 1 || E > 1024) return false;
    const int64_t n_used = ids->ne[0], T = ids->ne[1];
    // Admit only the unchanged CIRU target's routed gate/up and down geometries.
    // IQ4_NL admission and its existing kernels remain unchanged.
    if ((src0->type == GGML_TYPE_Q4_1 || c1) &&
        (E != 512 || n_used != 10 || T > 32768 ||
         !((K == 2560 && M == 640) || (K == 640 && M == 2560)))) return false;
    if (src1->ne[0] != K || src1->ne[3] != 1 || src1->ne[2] != T) return false;
    if (src1->ne[1] != 1 && src1->ne[1] != n_used) return false;
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1) return false;
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
    if (T < (c1 ? mmb_c1_min_t() : mmb_min_t()) || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;   // tile index must fit in 16 bits per expert
    return true;
}

void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (src0->type == GGML_TYPE_F32) {
        static const bool halo_prefill = getenv("CIRU_HALO_PREFILL") && atoi(getenv("CIRU_HALO_PREFILL")) != 0;
        if (halo_prefill) {
            const float * W = (const float *) src0->data, * X = (const float *) src1->data;
            float * D = (float *) dst->data;
            if (M <= 64) {
                dim3 grid((M + 63) / 64, (T + 31) / 32);
                mmb_f32_halo_kernel<64, 32, 16, 16, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
            } else if (M <= 1024) {
                dim3 grid((M + 31) / 32, (T + 63) / 64);
                mmb_f32_halo_kernel<32, 64, 16, 16, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
            } else {
                dim3 grid((M + 127) / 128, (T + 127) / 128);
                mmb_f32_halo_kernel<128, 128, 32, 64, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
            }
            CUDA_CHECK(cudaGetLastError());
            static unsigned hits = 0;
            if (hits++ < 4) fprintf(stderr, "CIRU_HALO_F32 M=%d K=%d T=%d\n", M, K, T);
            return;
        }
        dim3 grid((M + 127) / 128, (T + 127) / 128);
        static const bool two = getenv("LLAMA_MMB_F32SPLIT") && atoi(getenv("LLAMA_MMB_F32SPLIT")) >= 2;
        if (two) mmb_f32split_kernel<128, 128, 32, 64, true ><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T);
        else     mmb_f32split_kernel<128, 128, 32, 64, false><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T);
        CUDA_CHECK(cudaGetLastError()); return;
    }
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) T * K, stream);
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data;
    if (mmb_tall() && src0->type == GGML_TYPE_IQ4_NL && M <= 384 && K >= 4096 && T >= 2048) {   // tall-M tile: HC down|inject [10240 -> 324], activations read once
        static const int wide = mmb_tall_mode() >= 2;
        if (wide) { dim3 grid(1, (T + 63) / 64); mmb_dense_kernel<384, 64, 96, 32, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        else      { dim3 grid(1, (T + 31) / 32); mmb_dense_kernel<384, 32, 96, 16, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        CUDA_CHECK(cudaGetLastError());
        static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_TALL%s dense M=%d K=%d T=%d\n", wide ? "(wide 384x64)" : "(384x32)", M, K, T);
        return;
    }
    const uint16_t * shadow_pre = ((src0->type == GGML_TYPE_IQ4_NL && mmb_shadow()) || src0->type == GGML_TYPE_Q6_K) ? mmb_shadow_lookup(src0) : nullptr;
    const bool big = (M >= 6144 && K >= 2560) || (shadow_pre && K >= 2560 && T >= 4096);
    uint16_t * Dh = (mmb_hc16() && K == 320 && M == 10240) ? ggml_cuda_mmb_slot_reserve(ctx, 1, dst, (size_t) T * M) : nullptr;
    bool store_f32 = !(Dh && ggml_cuda_mmb_is_bf16_only(dst));
    if (ggml_cuda_mmb_blk16() && !Dh && ggml_cuda_mmb_is_bf16_only(dst) && (M & 7) == 0) {
        Dh = (uint16_t *) dst->data; store_f32 = false;
        static unsigned h = 0; if (h++ < 2) fprintf(stderr, "MMB_BLK16 dense BF16 in place: M=%d K=%d T=%d\n", M, K, T);
    }
    dim3 grid((M + 127) / 128, big ? (T + 255) / 256 : (T + 127) / 128);
    const uint16_t * shadow = shadow_pre;
    if (src0->type == GGML_TYPE_Q6_K && !shadow) { GGML_ABORT("MMB: Q6_K weight %s has no BF16 shadow", src0->name); }
    if (src0->type == GGML_TYPE_Q5_K) {
        static const int requested_half = getenv("CIRU_Q5_PACKED_HALF") ? atoi(getenv("CIRU_Q5_PACKED_HALF")) : 0;
        const int half_mode = M >= 2560 && (requested_half >= 1 && requested_half <= 3) ? requested_half : 0;
        ggml_cuda_pool_alloc<uint16_t> temp(ctx.pool());
        if (half_mode == 1 || half_mode == 2) {
            const size_t n = size_t(T) * K;
            uint16_t * fp16 = temp.alloc(n);
            mmb_cvt_bf16_f16<<<(n+MMB_NT*8-1)/(MMB_NT*8), MMB_NT, 0, stream>>>(xhp, fp16, n);
            xhp = fp16;
        }
        if (half_mode == 3) {
            if (big) mmb_dense_kernel<128, 256, 64, 64, 8><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
            else     mmb_dense_kernel<128, 128, 32, 64, 8><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        } else if (half_mode == 2) {
            if (big) mmb_dense_kernel<128, 256, 64, 64, 7><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
            else     mmb_dense_kernel<128, 128, 32, 64, 7><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        } else if (half_mode == 1) {
            if (big) mmb_dense_kernel<128, 256, 64, 64, 6><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
            else     mmb_dense_kernel<128, 128, 32, 64, 6><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        } else {
            if (big) mmb_dense_kernel<128, 256, 64, 64, 5><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
            else     mmb_dense_kernel<128, 128, 32, 64, 5><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        }
        static unsigned hits = 0;
        if (hits++ < 12) fprintf(stderr, "CIRU_HALO_Q5_MMB tensor=%s K=%d M=%d T=%d cached_input=%d bf16_output=%d half_mode=%d\n", src0->name, K, M, T, ggml_cuda_mmb_cache_lookup(src1) != nullptr, !store_f32, half_mode);
    } else if (shadow) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T);
    } else if (src0->type == GGML_TYPE_IQ4_NL) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    } else if (src0->type == GGML_TYPE_Q8_0) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    } else {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_gatemix() { return mmb_gatemix_flag(); }
bool ggml_cuda_mmb_down16() { return mmb_down16_flag(); }
bool ggml_cuda_mmb_blk16() { static const int v = getenv("LLAMA_HC_BLK16") ? atoi(getenv("LLAMA_HC_BLK16")) : 0; return v != 0; }
bool ggml_cuda_mmb_res16()  { static const int v = getenv("LLAMA_HC_RES16") ? atoi(getenv("LLAMA_HC_RES16")) : 0; return v != 0; }
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst,
        const int hc, const float scale, const float bias) {
    static const bool q8_enabled = getenv("CIRU_HC_Q8_FUSE") && atoi(getenv("CIRU_HC_Q8_FUSE")) != 0;
    if (!mmb_gatemix_flag() || hc != 4 || (w->type != GGML_TYPE_IQ4_NL && !(q8_enabled && w->type == GGML_TYPE_Q8_0)) || lo->type != GGML_TYPE_F32 || !ggml_is_contiguous(lo) || !ggml_is_contiguous(dst)) return false;
    const int K = (int) w->ne[0], M = (int) w->ne[1], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    if (K % MMB_BK != 0 || M != hc * E || E % 32 != 0 || lo->ne[0] != K || ggml_nrows(lo) != T || xn->ne[0] != M || ggml_nrows(xn) != T || T < mmb_min_t()) return false;
    const uint16_t * xn16 = ggml_cuda_mmb_cache_lookup(xn);
    if (!xn16) return false;
    cudaStream_t stream = ctx.stream();
    const uint16_t * lo16 = mmb_bf16_activation(ctx, lo, (size_t) T * K, stream);
    uint16_t * outh = ggml_cuda_mmb_slot_reserve(ctx, 3, dst, (size_t) T * E);
    const bool store_f32 = !(outh && ggml_cuda_mmb_is_bf16_only(dst));
    dim3 grid(E / 32, (T + 127) / 128);
    if (w->type == GGML_TYPE_Q8_0) {
    hc_gate_mix_kernel<4, 1><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, (float *) dst->data, outh, store_f32, E, K, T, scale, bias);
    } else {
    hc_gate_mix_kernel<4><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, (float *) dst->data, outh, store_f32, E, K, T, scale, bias);
    }
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "HC_GATEMIX fused gate GEMM + sigmoid + mix: type=%d E=%d K=%d T=%d\n", (int) w->type, E, K, T);
    return true;
}

void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;

    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    constexpr int BN_SMALL = 32;
    // HaloBox 1298b8680: admit medium expert buckets to the wide tile.
    static const int THRESH = getenv("CIRU_HALO_PREFILL") && atoi(getenv("CIRU_HALO_PREFILL")) != 0 ? 32 : 128;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);

    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data; const size_t eb = (size_t) src0->nb[2];
    uint16_t * Dh = (mmb_down16_flag() && ggml_cuda_mmb_is_bf16_only(dst)) ? (uint16_t *) dst->data : nullptr;
    const bool store_f32 = Dh == nullptr;
    if (Dh) { static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_DOWN16 routed down BF16 in place: M=%d K=%d rows=%d\n", M, K, n_rows); }
    dim3 gbig((M + 127) / 128, nbig_max), gsmall((M + 127) / 128, nsmall_max);
    if (src0->type == GGML_TYPE_Q4_1) {
        mmb_routed_kernel<128, BN, 32, 64, 3><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_kernel<128, BN_SMALL, 32, 16, 3><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    } else if (src0->type == GGML_TYPE_C1_G16_K640 || src0->type == GGML_TYPE_C1_G16_K2560) {
        const auto * rows = (const uint16_t *) dst->src[3]->data;
        const auto * tables = (const uint32_t *) dst->src[4]->data;
        mmb_routed_c1_kernel<128, BN, 32, 64><<<gbig, MMB_NT, 0, stream>>>(W, eb, rows, tables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_c1_kernel<128, BN_SMALL, 32, 16><<<gsmall, MMB_NT, 0, stream>>>(W, eb, rows, tables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
        static unsigned hits_c1 = 0;
        if (hits_c1++ < 12) fprintf(stderr, "C1_MMB_PROJECT K=%d N=%d T=%d rows=%d store_f32=%d\n", K, M, T, n_rows, (int)store_f32);
    } else {
        mmb_routed_kernel<128, BN, 32, 64><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_kernel<128, BN_SMALL, 32, 16><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu) {
    if (!mmb_enabled() || !mmb_glu() || !gw || !uw || !src1 || !ids || !glu) return false;
    const bool c1 = gw->type == GGML_TYPE_C1_G16_K2560;
    if (gw->type != uw->type || (gw->type != GGML_TYPE_IQ4_NL && gw->type != GGML_TYPE_Q4_1 && !c1)) return false;
    if ((gw->type == GGML_TYPE_Q4_1 || c1) && (gw->ne[0] != 2560 || gw->ne[1] != 640)) return false;
    if (!ggml_are_same_shape(gw, uw) || gw->nb[1] != uw->nb[1] || gw->nb[2] != uw->nb[2]) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) return false;
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !glu->src[0] || !glu->src[1]) return false;
    if (glu->src[0]->op != GGML_OP_MUL_MAT_ID || glu->src[1]->op != GGML_OP_MUL_MAT_ID) return false;
    if (glu->src[0]->src[0] != gw || glu->src[1]->src[0] != uw || glu->src[0]->src[1] != src1 || glu->src[1]->src[1] != src1 || glu->src[0]->src[2] != ids || glu->src[1]->src[2] != ids) return false;
    if (ggml_nelements(glu) != ggml_nelements(glu->src[0]) || glu->ne[0] != gw->ne[1]) return false;
    return ggml_cuda_mmb_supported_mmid(gw, src1, ids, glu->src[0]) && ggml_cuda_mmb_supported_mmid(uw, src1, ids, glu->src[1]);
}

void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) gw->ne[0], M = (int) gw->ne[1], E = (int) gw->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    constexpr int BN_SMALL = 32;
    // HaloBox 1298b8680: admit medium expert buckets to the wide tile.
    static const int THRESH = getenv("CIRU_HALO_PREFILL") && atoi(getenv("CIRU_HALO_PREFILL")) != 0 ? 32 : 128;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);
    uint16_t * Dh = ggml_cuda_mmb_slot_reserve(ctx, 2, glu, (size_t) n_rows * M);
    const bool store_f32 = !ggml_cuda_mmb_is_bf16_only(glu);
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_GLU fused gate/up+swiglu: M=%d K=%d rows=%d store_f32=%d\n", M, K, n_rows, (int) store_f32);
    const uint8_t * Wg = (const uint8_t *) gw->data, * Wu = (const uint8_t *) uw->data; float * D = (float *) glu->data; const size_t eb = (size_t) gw->nb[2];
    dim3 gbig((M + 63) / 64, nbig_max), gsmall((M + 63) / 64, nsmall_max);
    if (gw->type == GGML_TYPE_Q4_1) {
        mmb_routed_glu_kernel<64, BN, 32, 32, 3><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, 3><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    } else if (gw->type == GGML_TYPE_C1_G16_K2560) {
        const auto * grows = (const uint16_t *) glu->src[0]->src[3]->data;
        const auto * urows = (const uint16_t *) glu->src[1]->src[3]->data;
        const auto * gtables = (const uint32_t *) glu->src[0]->src[4]->data;
        const auto * utables = (const uint32_t *) glu->src[1]->src[4]->data;
        static const bool group4 = getenv("LLAMA_C1_GROUP4") && atoi(getenv("LLAMA_C1_GROUP4")) != 0;
        static const bool packed_half = getenv("LLAMA_C1_PACKED_HALF") && atoi(getenv("LLAMA_C1_PACKED_HALF")) != 0;
        if (packed_half) {
            mmb_routed_c1_glu_kernel<64, BN, 32, 32, true, true><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, grows, urows, gtables, utables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        } else if (group4) {
            mmb_routed_c1_glu_kernel<64, BN, 32, 32, true><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, grows, urows, gtables, utables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        } else {
            mmb_routed_c1_glu_kernel<64, BN, 32, 32, false><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, grows, urows, gtables, utables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        }
        mmb_routed_c1_glu_kernel<64, BN_SMALL, 16, 16, false><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, grows, urows, gtables, utables, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
        static unsigned hits_c1 = 0;
        if (hits_c1++ < 12) fprintf(stderr, "C1_MMB_GLU K=%d N=%d T=%d rows=%d store_f32=%d group4=%d packed_half=%d\n", K, M, T, n_rows, (int)store_f32, (int)group4, (int)packed_half);
    } else {
        mmb_routed_glu_kernel<64, BN, 32, 32><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Called from graph_optimize (outside stream capture): create the shadow for an eligible IQ4_NL dense weight.
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!w) return;
    if (mmb_is_resident_q6k(w)) {
        if (!mmb_shadow_q6k() || g_mmb_shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { fprintf(stderr, "MMB_SHADOW cap reached; %s stays Q6_K\n", w->name); return; }
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { fprintf(stderr, "MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        mmb_dq_q6k_bf16_kernel<<<(unsigned) ((n / 256 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 256);
        CUDA_CHECK(cudaGetLastError());
        g_mmb_shadow[w->data] = buf; g_mmb_shadow_bytes += bytes;
        static unsigned q6 = 0; if (q6++ < 3) fprintf(stderr, "MMB_SHADOW Q6_K %s [%lld x %lld] -> BF16 (%.1f MB total)\n", w->name, (long long) w->ne[0], (long long) w->ne[1], g_mmb_shadow_bytes / 1048576.0);
        return;
    }
    if (mmb_shadow_mode() != 1) return;               // mode 2: Q6_K only
    const bool concat = mmb_is_row_concat(w);
    if (!concat && !mmb_is_resident_iq4(w)) return;
    if (concat ? g_mmb_shadow_pair.count({w->src[0]->data, w->src[1]->data}) > 0 : g_mmb_shadow.count(w->data) > 0) return;
    const size_t n = (size_t) w->ne[0] * w->ne[1];
    const size_t bytes = n * 2;
    if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { static bool warned = false; if (!warned) { fprintf(stderr, "MMB_SHADOW cap reached at %.1f MB; further weights stay IQ4_NL\n", g_mmb_shadow_bytes / 1048576.0); warned = true; } return; }
    uint16_t * buf = nullptr;
    if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { fprintf(stderr, "MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
    if (concat) {
        const size_t n0 = (size_t) w->src[0]->ne[0] * w->src[0]->ne[1], n1 = (size_t) w->src[1]->ne[0] * w->src[1]->ne[1];
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n0 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[0]->data, buf, n0 / 32);
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n1 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[1]->data, buf + n0, n1 / 32);
        g_mmb_shadow_pair[{w->src[0]->data, w->src[1]->data}] = buf;
    } else {
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 32);
        g_mmb_shadow[w->data] = buf;
    }
    CUDA_CHECK(cudaGetLastError());
    g_mmb_shadow_bytes += bytes;
    static unsigned hits = 0; if (hits++ < 3 || (hits % 50) == 0) fprintf(stderr, "MMB_SHADOW %s [%lld x %lld] -> BF16 (%.1f MB total)\n", w->name, (long long) w->ne[0], (long long) w->ne[1], g_mmb_shadow_bytes / 1048576.0);
}

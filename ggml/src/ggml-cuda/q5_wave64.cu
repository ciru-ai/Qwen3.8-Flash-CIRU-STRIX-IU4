// Isolated Q5_K wave64 component, default absent from every serving runtime.
// The GEMM body is the exact Gufo 990fdce WKQuantA8BlockedWmmaGEMMKernel
// excerpt in donor_wkquant_kernel.cuh. This TU supplies only Q5_K decode,
// Gufo-format Q8 activation packing/sums, and launch glue.
#include "q5_wave64.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace core {
enum class GgmlType { kQ3_K, kQ4_K, kQ5_K, kQ6_K };
}
namespace gufo::hip {

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
constexpr std::size_t kQ8ActSumTileBytes = 16 * sizeof(float);

struct Q5KBlock {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176);

struct QuantSub16 {
    union { std::int8_t q[16]; std::uint32_t w[4]; };
    float scale;
    float offset;
};

__host__ __device__ inline std::size_t Q8ActTiledBytes(std::size_t batch, std::size_t k) {
    return ((batch + 15) / 16) * (k / 32) * kQ8ActTileBytes;
}
__host__ __device__ inline std::size_t Q8ActSumBytes(std::size_t batch, std::size_t k) {
    return ((batch + 15) / 16) * (k / 32) * kQ8ActSumTileBytes;
}
__device__ __forceinline__ const float * Q8ActSumSidecar(const void * base, std::size_t batch, std::size_t k) {
    return reinterpret_cast<const float *>(static_cast<const std::uint8_t *>(base) + Q8ActTiledBytes(batch, k));
}
__device__ __forceinline__ std::int8_t * Q8ActTile(void * base, std::size_t num_blocks,
                                                   std::size_t tt, std::size_t kb) {
    return static_cast<std::int8_t *>(base) + ((tt * num_blocks + kb) * kQ8ActTileBytes);
}
__device__ __forceinline__ std::size_t QuantRowBytes(core::GgmlType, std::size_t k) {
    return (k / 256) * sizeof(Q5KBlock);
}

// Exact Q5_K branch and shared-scale spelling from pinned Gufo quant_ops.hpp.
// Each call decodes one 16-value sub-block of the resident 176-byte GGUF block.
template<bool SharedScales>
__device__ __forceinline__ void DecodeQuantSub16(core::GgmlType,
                                                  const void * __restrict__ row,
                                                  std::size_t sub16,
                                                  QuantSub16 & out) {
    const auto & blk = static_cast<const Q5KBlock *>(row)[sub16 / 16];
    const std::size_t sb32 = (sub16 / 2) % 8;
    const std::size_t lane0 = (sub16 % 2) * 16;
    const std::size_t base = ((sb32 / 2) * 32) + lane0;
    const unsigned shift = ((sb32 % 2) == 0) ? 0U : 4U;
    std::uint32_t packed[4], high[4];
    __builtin_memcpy(packed, blk.qs + base, 16);
    __builtin_memcpy(high, blk.qh + lane0, 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (packed[i] >> shift) & 0x0F0F0F0FU;
        const std::uint32_t bit = (high[i] >> static_cast<unsigned>(sb32)) & 0x01010101U;
        out.w[i] = low | (bit << 4U);
    }
    const std::size_t ix = sb32 & 3U;
    const unsigned lo = blk.scales[ix], mid = blk.scales[ix + 4], hi = blk.scales[ix + 8];
    const unsigned mask = 0U - static_cast<unsigned>(sb32 >> 2U);
    const unsigned lower_scale = lo & 63U, upper_scale = (hi & 15U) | ((lo >> 6U) << 4U);
    const unsigned lower_min = mid & 63U, upper_min = (hi >> 4U) | ((mid >> 6U) << 4U);
    const unsigned sc = lower_scale ^ ((lower_scale ^ upper_scale) & mask);
    const unsigned m = lower_min ^ ((lower_min ^ upper_min) & mask);
    out.scale = __half2float(blk.d) * static_cast<float>(sc);
    out.offset = __half2float(blk.dmin) * static_cast<float>(m);
    (void) SharedScales;
}

template<int WaveSize>
using WmmaQuantAccumulator = std::conditional_t<WaveSize == 64, int32x4_t, int32x8_t>;
template<int WaveSize>
__device__ __forceinline__ WmmaQuantAccumulator<WaveSize> WmmaQuant(
    int32x4_t a, int32x4_t b, WmmaQuantAccumulator<WaveSize> c) {
    if constexpr (WaveSize == 64)
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c, true);
    else
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

#include "donor_wkquant_kernel.cuh"

// Gufo's source quantizer uses one 32-element F32 activation group per Q8
// block, roundf, F32 scale, and a dequantized sum sidecar for Q5's offset.
__global__ void pack_q8_and_sums(const float * __restrict__ x, void * __restrict__ q8,
                                  std::size_t tokens, std::size_t k) {
    const std::size_t nblocks = k / 32;
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (linear >= tokens * nblocks) return;
    const int lane = threadIdx.x & 31;
    const std::size_t tok = linear / nblocks, kb = linear % nblocks;
    const float value = x[tok * k + kb * 32 + lane];
    float max_abs = fabsf(value);
    for (int off = 16; off > 0; off >>= 1) max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    const float d = max_abs / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::int8_t code = static_cast<std::int8_t>(roundf(value * id));
    const std::size_t tt = tok / 16, tl = tok % 16;
    std::int8_t * tile = Q8ActTile(q8, nblocks, tt, kb);
    tile[(lane / 16) * 256 + tl * 16 + (lane & 15)] = code;
    int sum = static_cast<int>(code);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
    if (lane == 0) {
        *reinterpret_cast<float *>(tile + 512 + tl * sizeof(float)) = d;
        float * sums = reinterpret_cast<float *>(static_cast<std::uint8_t *>(q8) + Q8ActTiledBytes(tokens,k));
        sums[(tt * nblocks + kb) * 16 + tl] = d * static_cast<float>(sum);
    }
}

__global__ void zero_q8_tail(void * q8, std::size_t tokens, std::size_t k) {
    const std::size_t tail = tokens % 16, nblocks = k / 32;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tail == 0 || idx >= nblocks * (16 - tail)) return;
    const std::size_t kb = idx / (16 - tail), tl = tail + idx % (16 - tail);
    std::int8_t * tile = Q8ActTile(q8,nblocks,tokens / 16,kb);
    *reinterpret_cast<int32x4_t *>(tile + tl * 16) = {};
    *reinterpret_cast<int32x4_t *>(tile + 256 + tl * 16) = {};
    *reinterpret_cast<float *>(tile + 512 + tl * sizeof(float)) = 0.0f;
    float * sums = reinterpret_cast<float *>(static_cast<std::uint8_t *>(q8) + Q8ActTiledBytes(tokens,k));
    sums[((tokens / 16 * nblocks) + kb) * 16 + tl] = 0.0f;
}

} // namespace gufo::hip

extern "C" std::size_t c1_q5_wave64_q8_bytes(std::size_t tokens, std::size_t k) {
    return gufo::hip::Q8ActTiledBytes(tokens,k) + gufo::hip::Q8ActSumBytes(tokens,k);
}

extern "C" hipError_t c1_q5_wave64_launch(const void * w, const float * x, void * q8,
                                            float * y, std::size_t tokens,
                                            std::size_t m, std::size_t k,
                                            hipStream_t stream) {
    if (!w || !x || !q8 || !y || tokens < 4096 || tokens > 32768 ||
        (k != 2560 && k != 6144) ||
        k % 256 != 0 ||
        !((k == 2560 && (m == 6144 || m == 10240 || m == 12288)) ||
          (k == 6144 && m == 2560)))
        return hipErrorInvalidValue;
    const std::size_t blocks = tokens * (k / 32);
    gufo::hip::pack_q8_and_sums<<<(blocks + 7) / 8, 256, 0, stream>>>(x,q8,tokens,k);
    hipError_t rc = hipGetLastError(); if (rc != hipSuccess) return rc;
    if (tokens % 16 != 0) {
        const std::size_t tail_blocks = (k / 32) * (16 - (tokens % 16));
        gufo::hip::zero_q8_tail<<<(tail_blocks + 255) / 256, 256, 0, stream>>>(q8,tokens,k);
        rc = hipGetLastError(); if (rc != hipSuccess) return rc;
    }
    const dim3 grid((tokens + 127) / 128, (m + 127) / 128);
    gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128,128,2,4,1,core::GgmlType::kQ5_K,64>
        <<<grid,256,0,stream>>>(w,q8,y,tokens,m,k);
    return hipGetLastError();
}

#include <climits>
#include <cstdlib>
#include <cstring>

#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t qsa_ordered_f32(const float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// One block owns one QSA query row.  Four radix passes find the exact kth
// score without sorting the entire history.  The selected block IDs are then
// sorted by position, which is both deterministic and the useful order for
// the indexed K/V tile loader.  Ties at the threshold prefer the lower block
// position.
static __global__ void top_k_qsa_blocks_f32_i32(
        const float * src, const int32_t * positions, int * dst,
        const int ncols, const int k, const int ratio,
        const int cell_base, const int end_pos) {
    const int tid = threadIdx.x;
    const float * row_src = src + (int64_t) blockIdx.x*ncols;
    const int dst_width = ratio*k + ratio - 1;
    int * row_dst = dst + (int64_t) blockIdx.x*dst_width;
    const int neligible = min(ncols, (positions[blockIdx.x] + 1)/ratio);

    __shared__ int histogram[256];
    __shared__ int scan[256];
    __shared__ int selected[512];
    __shared__ uint32_t prefix;
    __shared__ int rank;
    __shared__ int emitted;

    if (tid == 0) {
        prefix = 0;
        rank = k - 1;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        histogram[tid] = 0;
        __syncthreads();

        const int shift = 24 - 8*pass;
        const uint32_t prefix_mask = pass == 0 ? 0u : (0xffffffffu << (32 - 8*pass));
        for (int col = tid; col < neligible; col += blockDim.x) {
            const uint32_t key = qsa_ordered_f32(row_src[col]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int skipped = 0;
            for (int bucket = 255; bucket >= 0; --bucket) {
                const int count = histogram[bucket];
                if (rank < skipped + count) {
                    prefix |= (uint32_t) bucket << shift;
                    rank -= skipped;
                    break;
                }
                skipped += count;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        emitted = 0;
    }
    __syncthreads();

    // Deterministic block-wide compaction in source-position order.
    for (int mode = 0; mode < 2; ++mode) {
        for (int base = 0; base < neligible && emitted < k; base += blockDim.x) {
            const int col = base + tid;
            const uint32_t key = col < neligible ? qsa_ordered_f32(row_src[col]) : 0u;
            const bool take = col < neligible && (mode == 0 ? key > prefix : key == prefix);
            scan[tid] = take ? 1 : 0;
            __syncthreads();

#pragma unroll
            for (int offset = 1; offset < 256; offset <<= 1) {
                const int add = tid >= offset ? scan[tid - offset] : 0;
                __syncthreads();
                scan[tid] += add;
                __syncthreads();
            }

            const int out = emitted + scan[tid] - 1;
            if (take && out < k) {
                selected[out] = col;
            }
            const int count = scan[255];
            __syncthreads();
            if (tid == 0) {
                emitted = min(k, emitted + count);
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < 512; i += blockDim.x) {
        if (i >= k) {
            selected[i] = INT_MAX;
        }
    }
    __syncthreads();

    // Position-sort a fixed 512-wide shared tile.  Padding sorts to the end.
#pragma unroll
    for (int size = 2; size <= 512; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < 512; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = (i & size) == 0;
                    const int lhs = selected[i];
                    const int rhs = selected[j];
                    if ((ascending && lhs > rhs) || (!ascending && lhs < rhs)) {
                        selected[i] = rhs;
                        selected[j] = lhs;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += blockDim.x) {
        const int block = selected[i];
        for (int t = 0; t < ratio; ++t) {
            row_dst[i*ratio + t] = cell_base + block*ratio + t;
        }
    }
    if (tid < ratio - 1) {
        const int q = positions[blockIdx.x];
        const int tail_start = ((q + 1)/ratio)*ratio;
        row_dst[ratio*k + tid] = cell_base + min(tail_start + tid, end_pos);
    }
}

static __global__ void ciru_top_k_radix_cells(
        const float * src, int * dst, const int ncols, const int k) {
    const int tid = threadIdx.x;
    const float * row_src = src + (int64_t) blockIdx.x*ncols;
    const int dst_width = k;
    int * row_dst = dst + (int64_t) blockIdx.x*dst_width;
    const int neligible = ncols;

    __shared__ int histogram[256];
    __shared__ int scan[256];
    __shared__ int selected[4096];
    __shared__ uint32_t prefix;
    __shared__ int rank;
    __shared__ int emitted;

    if (tid == 0) {
        prefix = 0;
        rank = k - 1;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        histogram[tid] = 0;
        __syncthreads();

        const int shift = 24 - 8*pass;
        const uint32_t prefix_mask = pass == 0 ? 0u : (0xffffffffu << (32 - 8*pass));
        for (int col = tid; col < neligible; col += blockDim.x) {
            const uint32_t key = qsa_ordered_f32(row_src[col]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int skipped = 0;
            for (int bucket = 255; bucket >= 0; --bucket) {
                const int count = histogram[bucket];
                if (rank < skipped + count) {
                    prefix |= (uint32_t) bucket << shift;
                    rank -= skipped;
                    break;
                }
                skipped += count;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        emitted = 0;
    }
    __syncthreads();

    // Deterministic block-wide compaction in source-position order.
    for (int mode = 0; mode < 2; ++mode) {
        for (int base = 0; base < neligible && emitted < k; base += blockDim.x) {
            const int col = base + tid;
            const uint32_t key = col < neligible ? qsa_ordered_f32(row_src[col]) : 0u;
            const bool take = col < neligible && (mode == 0 ? key > prefix : key == prefix);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            const unsigned long long votes = __ballot(take);
            if (lane == 0) scan[warp] = __popcll(votes);
            __syncthreads();
            int rank_in_tile = __popcll(votes & ((1ull << lane) - 1));
            for (int w = 0; w < warp; ++w) rank_in_tile += scan[w];
            const int out = emitted + rank_in_tile;
            if (take && out < k) selected[out] = col;
            __syncthreads();
            if (tid == 0) {
                int count = 0;
                for (int w = 0; w < 8; ++w) count += scan[w];
                emitted = min(k, emitted + count);
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        if (i >= k) {
            selected[i] = INT_MAX;
        }
    }
    __syncthreads();

    // Position-sort a fixed 4096-wide shared tile.  Padding sorts to the end.
#pragma unroll
    for (int size = 2; size <= 4096; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < 4096; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = (i & size) == 0;
                    const int lhs = selected[i];
                    const int rhs = selected[j];
                    if ((ascending && lhs > rhs) || (!ascending && lhs < rhs)) {
                        selected[i] = rhs;
                        selected[j] = lhs;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += blockDim.x) {
        row_dst[i] = selected[i];
    }
}

static void top_k_qsa_blocks_f32_i32_launch(
        const float * src, const int32_t * positions, int * dst,
        const int ncols, const int nrows, const int k, const int ratio,
        const int cell_base, const int end_pos, cudaStream_t stream) {
    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(256, 1, 1);
    top_k_qsa_blocks_f32_i32<<<blocks, threads, 0, stream>>>(
        src, positions, dst, ncols, k, ratio, cell_base, end_pos);
}

static __device__ __forceinline__ bool h109c_heap_comp_f32(
        const float * values, const int lhs, const int rhs) {
    return values[lhs] > values[rhs];
}

static __device__ __forceinline__ void h109c_heap_push_f32(
        const float * values, int * heap, int hole, const int top, const int value) {
    int parent = (hole - 1) / 2;
    while (hole > top && h109c_heap_comp_f32(values, heap[parent], value)) {
        heap[hole] = heap[parent];
        hole = parent;
        parent = (hole - 1) / 2;
    }
    heap[hole] = value;
}

static __device__ __forceinline__ void h109c_heap_adjust_f32(
        const float * values, int * heap, int hole, const int len, const int value) {
    const int top = hole;
    int second_child = hole;
    while (second_child < (len - 1) / 2) {
        second_child = 2 * (second_child + 1);
        if (h109c_heap_comp_f32(values, heap[second_child], heap[second_child - 1])) {
            --second_child;
        }
        heap[hole] = heap[second_child];
        hole = second_child;
    }
    if ((len & 1) == 0 && second_child == (len - 2) / 2) {
        second_child = 2 * (second_child + 1);
        heap[hole] = heap[second_child - 1];
        hole = second_child - 1;
    }
    h109c_heap_push_f32(values, heap, hole, top, value);
}

// H109C reuses H49's exact GCC-15.2 heap semantics for the legacy expanded
// cell scores.  One wave owns a row; lane zero intentionally owns all heap
// comparisons so four-way score ties retain the exact H101 cell membership.
template<bool warp_scan>
static __global__ void h109c_top_k_cells_heap_exact(
        const float * src, int * dst, const int ncols, const int k) {
    const int row = blockIdx.x;
    const float * row_src = src + (int64_t) row*ncols;
    int * row_dst = dst + (int64_t) row*k;
    extern __shared__ int heap[];

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        heap[i] = i;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (k >= 2) {
            int parent = (k - 2) / 2;
            while (true) {
                const int value = heap[parent];
                h109c_heap_adjust_f32(row_src, heap, parent, k, value);
                if (parent == 0) {
                    break;
                }
                --parent;
            }
        }
        if constexpr (!warp_scan) {
            for (int i = k; i < ncols; ++i) {
                if (h109c_heap_comp_f32(row_src, i, heap[0])) {
                    h109c_heap_adjust_f32(row_src, heap, 0, k, i);
                }
            }
        }
        // Membership is the only consumer contract. Sorting the selected heap
        // and GGML's final slot swap change order only, not the selected set.
    }
    __syncthreads();

    if constexpr (warp_scan) {
        for (int base = k; base < ncols; base += 32) {
            const int i = base + threadIdx.x;
            // The minimum retained score only increases. A rejected candidate
            // cannot become eligible after an earlier lane updates the heap.
            unsigned int candidates = __ballot_sync(0xffffffffffffffffULL,
                    i < ncols && h109c_heap_comp_f32(row_src, i, heap[0]));
            if (threadIdx.x == 0) {
                while (candidates) {
                    const int candidate = base + __ffs(candidates) - 1;
                    candidates &= candidates - 1;
                    if (h109c_heap_comp_f32(row_src, candidate, heap[0])) {
                        h109c_heap_adjust_f32(row_src, heap, 0, k, candidate);
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        row_dst[i] = heap[i];
    }
}

static void h109c_top_k_cells_heap_exact_launch(
        const float * src, int * dst, const int ncols, const int nrows,
        const int k, cudaStream_t stream) {
    const char * scan = std::getenv("CIRU_QSA_WARP_SCAN");
    if (scan && std::strcmp(scan, "1") == 0) {
        h109c_top_k_cells_heap_exact<true><<<dim3(nrows), dim3(32), k*sizeof(int), stream>>>(
            src, dst, ncols, k);
    } else {
        h109c_top_k_cells_heap_exact<false><<<dim3(nrows), dim3(WARP_SIZE), k*sizeof(int), stream>>>(
            src, dst, ncols, k);
    }
}

static __global__ void h111_top_k_identity(int * dst, const int ncols, const int total) {
    const int i = int(blockIdx.x)*int(blockDim.x) + int(threadIdx.x);
    if (i < total) {
        dst[i] = i % ncols;
    }
}

static void h111_top_k_identity_launch(
        int * dst, const int ncols, const int nrows, cudaStream_t stream) {
    constexpr int threads = 256;
    const int total = ncols*nrows;
    h111_top_k_identity<<<dim3((total + threads - 1)/threads), dim3(threads), 0, stream>>>(
        dst, ncols, total);
}

#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// The vocabulary is split into 1024-value tiles.  Keeping ten winners per
// tile is sufficient to recover the global top ten through smaller reduction passes.
// Equal scores use token order; the TOP_K operation does not prescribe tie order.
static __device__ __forceinline__ bool mtp_top10_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

template<int items, bool merge, bool mapped = false>
static __global__ void mtp_top10_f32_i32(
        const float * src, const int * src_ids, float * values, int * indices,
        const int ncols, const int ntiles) {
    constexpr int threads = 256;
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = threads / warp;
    const int row = blockIdx.y;
    const int start = merge ? 0 : int(blockIdx.x) * threads * items;
    const float * row_src = src + int64_t(row) * ncols;
    const int * row_ids = (merge || mapped) ? src_ids + int64_t(row) * ncols : nullptr;

    float local_values[items];
    int local_ids[items];
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int col = start + threadIdx.x + i * threads;
        local_values[i] = col < ncols ? row_src[col] : -INFINITY;
        local_ids[i] = col < ncols ? ((merge || mapped) ? row_ids[col] : col) : INT_MAX;
    }

    __shared__ float warp_values[nwarps];
    __shared__ int warp_ids[nwarps];
    __shared__ int selected;
    const int out = merge ? row * 10 : (row * ntiles + int(blockIdx.x)) * 10;

#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        float best = -INFINITY;
        int best_id = INT_MAX;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            if (mtp_top10_better(local_values[i], local_ids[i], best, best_id)) {
                best = local_values[i];
                best_id = local_ids[i];
            }
        }
#pragma unroll
        for (int offset = warp / 2; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, offset, warp);
            const int other_id = __shfl_xor_sync(0xffffffffu, best_id, offset, warp);
            if (mtp_top10_better(other, other_id, best, best_id)) {
                best = other;
                best_id = other_id;
            }
        }
        if (threadIdx.x % warp == 0) {
            warp_values[threadIdx.x / warp] = best;
            warp_ids[threadIdx.x / warp] = best_id;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            best = -INFINITY;
            best_id = INT_MAX;
#pragma unroll
            for (int i = 0; i < nwarps; ++i) {
                if (mtp_top10_better(warp_values[i], warp_ids[i], best, best_id)) {
                    best = warp_values[i];
                    best_id = warp_ids[i];
                }
            }
            indices[out + rank] = best_id;
            if constexpr (!merge) {
                values[out + rank] = best;
            }
            selected = best_id;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < items; ++i) {
            if (local_ids[i] == selected) {
                local_values[i] = -INFINITY;
                local_ids[i] = INT_MAX;
            }
        }
    }
}

static void mtp_top10_launch(ggml_cuda_pool & pool, const float * src, int * dst,
        int ncols, int nrows, cudaStream_t stream) {
    const int ntiles = (ncols + 1023) / 1024;
    GGML_ASSERT(ncols >= 10 && ncols <= 1048576);
    ggml_cuda_pool_alloc<float> values(pool, int64_t(nrows) * ntiles * 10);
    ggml_cuda_pool_alloc<int> indices(pool, int64_t(nrows) * ntiles * 10);
    mtp_top10_f32_i32<4, false><<<dim3(ntiles, nrows), dim3(256), 0, stream>>>(
        src, nullptr, values.get(), indices.get(), ncols, ntiles);
    CUDA_CHECK(cudaGetLastError());
    if (ntiles * 10 <= 4096) {
        mtp_top10_f32_i32<16, true><<<dim3(1, nrows), dim3(256), 0, stream>>>(
            values.get(), indices.get(), nullptr, dst, ntiles * 10, 1);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const int reduced_tiles = (ntiles * 10 + 1023) / 1024;
        ggml_cuda_pool_alloc<float> reduced_values(pool, int64_t(nrows) * reduced_tiles * 10);
        ggml_cuda_pool_alloc<int> reduced_indices(pool, int64_t(nrows) * reduced_tiles * 10);
        mtp_top10_f32_i32<4, false, true><<<dim3(reduced_tiles, nrows), dim3(256), 0, stream>>>(
            values.get(), indices.get(), reduced_values.get(), reduced_indices.get(), ntiles * 10, reduced_tiles);
        CUDA_CHECK(cudaGetLastError());
        mtp_top10_f32_i32<16, true><<<dim3(1, nrows), dim3(256), 0, stream>>>(
            reduced_values.get(), reduced_indices.get(), nullptr, dst, reduced_tiles * 10, 1);
        CUDA_CHECK(cudaGetLastError());
    }
}

#endif

bool ggml_cuda_supports_mtp_top_k(const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] {
        const char * value = std::getenv("CIRU_MTP_TOPK10");
        return value && value[0] == '1';
    }();
    const ggml_tensor * src = dst->src[0];
    return enabled && !ggml_top_k_is_qsa_blocks(dst) && src->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_I32 && (src->ne[0] >= 10 && src->ne[0] <= 1048576) && dst->ne[0] == 10 &&
        ggml_nrows(src) >= 1 && ggml_nrows(src) <= 16 && ggml_nrows(dst) == ggml_nrows(src) &&
        ggml_is_contiguous(src) && ggml_is_contiguous(dst);
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

bool ggml_cuda_qsa_prefill_rows_supported(int64_t nrows) {
    static const bool all_rows = [] {
        const char * value = std::getenv("GGML_QSA_ALL_ROWS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (all_rows && nrows >= 1 && nrows <= 2048) {
        return true;
    }
    static const bool wide_prefill = [] {
        const char * value = std::getenv("GGML_QSA_PREFILL_WIDE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return nrows == 512 || (wide_prefill && nrows > 8 && nrows <= 2048 && nrows % 4 == 0);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#if defined(GGML_USE_HIP)
    if (ggml_cuda_supports_mtp_top_k(dst)) {
        if (std::getenv("CIRU_MTP_TOPK_TRACE")) {
            GGML_LOG_INFO("CIRU_MTP_TOPK10 GPU cols=%lld rows=%lld\n", (long long) ncols, (long long) nrows);
        }
        mtp_top10_launch(pool, src0_d, dst_d, ncols, nrows, stream);
        return;
    }
#endif

#if defined(GGML_USE_HIP)
    if (ggml_top_k_is_qsa_blocks(dst)) {
        const ggml_tensor * positions = dst->src[1];
        const int ratio = ggml_get_op_params_i32(dst, 1);
        const int k_blocks = ggml_get_op_params_i32(dst, 2);
        const int cell_base = ggml_get_op_params_i32(dst, 3);
        const int end_pos = ggml_get_op_params_i32(dst, 4);
        GGML_ASSERT(ncols <= 65536);
        GGML_ASSERT(nrows <= 512);
        GGML_ASSERT(k_blocks <= 512);
        GGML_ASSERT(positions != nullptr);
        GGML_ASSERT(positions->type == GGML_TYPE_I32);
        GGML_ASSERT(positions->ne[0] >= nrows);
        GGML_ASSERT(ratio > 0);
        GGML_ASSERT(ggml_is_contiguous(dst));
        top_k_qsa_blocks_f32_i32_launch(
            src0_d, (const int32_t *) positions->data, dst_d,
            ncols, nrows, k_blocks, ratio, cell_base, end_pos, stream);
        return;
    }
    if (ggml_cuda_info().devices[ctx.device].warp_size == 32 &&
            ncols > k && k == 2051 && nrows <= 2048 &&
            (std::getenv("CIRU_QSA_RADIX_SELECT") && std::strcmp(std::getenv("CIRU_QSA_RADIX_SELECT"), "1") == 0)) {
        // Same top-k score budget; ties prefer earlier positions and output
        // is position-sorted. This differs from legacy heap tie/list order.
        ciru_top_k_radix_cells<<<dim3(nrows), dim3(256), 0, stream>>>(
            src0_d, dst_d, ncols, k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ncols > 1024) {
        static bool h109c_reported = false;
        if (!h109c_reported) {
            fprintf(stderr, "H111 exact H49 M512 selector cleanup active: ncols=%lld k=%lld rows=%lld\n",
                    (long long) ncols, (long long) k, (long long) nrows);
            h109c_reported = true;
        }
        // Heap storage is per query; increasing rows only increases the block count.
        GGML_ASSERT(src0->ne[1] == nrows && ggml_cuda_qsa_prefill_rows_supported(nrows));
        GGML_ASSERT(ncols <= 262144 && k == std::min<int64_t>(ncols, 2051));
        GGML_ASSERT(ggml_is_contiguous(dst));
        if (ncols == k) {
            h111_top_k_identity_launch(dst_d, ncols, nrows, stream);
            return;
        }
        GGML_ASSERT(k*sizeof(int) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
        h109c_top_k_cells_heap_exact_launch(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#endif
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}

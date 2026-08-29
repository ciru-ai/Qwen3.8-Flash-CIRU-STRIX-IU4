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
        for (int i = k; i < ncols; ++i) {
            if (h109c_heap_comp_f32(row_src, i, heap[0])) {
                h109c_heap_adjust_f32(row_src, heap, 0, k, i);
            }
        }
        // Membership is the only consumer contract. Sorting the selected heap
        // and GGML's final slot swap change order only, not the selected set.
    }
    __syncthreads();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        row_dst[i] = heap[i];
    }
}

static void h109c_top_k_cells_heap_exact_launch(
        const float * src, int * dst, const int ncols, const int nrows,
        const int k, cudaStream_t stream) {
    h109c_top_k_cells_heap_exact<<<dim3(nrows), dim3(WARP_SIZE), k*sizeof(int), stream>>>(
        src, dst, ncols, k);
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
    if (ncols > 1024) {
        static bool h109c_reported = false;
        if (!h109c_reported) {
            fprintf(stderr, "H111 exact H49 M512 selector cleanup active: ncols=%lld k=%lld rows=%lld\n",
                    (long long) ncols, (long long) k, (long long) nrows);
            h109c_reported = true;
        }
        GGML_ASSERT(src0->ne[1] == 512 && nrows == 512);
        GGML_ASSERT(ncols <= 8192 && k == std::min<int64_t>(ncols, 2051));
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

#include "fattn-common.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(2); }
}

// Call the production kernel, with an independent scalar specification. Poison
// storage beyond the logical mask detects scans of padding without provoking a
// GPU fault. Tight allocations separately exercise the real boundary shape.
template<int TILE>
static size_t run(int queries, int sequences, int mask_sequences, int layout, bool masked) {
    constexpr int keys = 768;
    const int tiles = (queries + TILE - 1) / TILE;
    const int rows = layout == 0 ? queries : tiles*TILE + 3;
    const int row_stride = keys + (layout == 2 ? 32 : 0);
    const int allocated_sequences = layout == 0 ? mask_sequences : sequences + 1;
    const int64_t sequence_stride = int64_t(rows)*row_stride;
    std::vector<half> mask(sequence_stride*allocated_sequences, __float2half(0.f));
    for (int s=0;s<mask_sequences;++s) {
        for (int q=0;q<queries;++q) {
            const int visible = masked ? 0 : (((q/7+s)%3)*256);
            for (int k=0;k<keys;++k) {
                mask[s*sequence_stride+q*row_stride+k] = __float2half(k<visible ? -.25f : -INFINITY);
            }
        }
    }
    std::vector<int> expected(tiles*sequences, 0), got(expected.size(), -1);
    for (int s=0;s<sequences;++s) for (int t=0;t<tiles;++t) {
        for (int q=t*TILE;q<std::min((t+1)*TILE,queries);++q) {
            for (int k=0;k<keys;++k) {
                if (std::isfinite(__half2float(mask[(s%mask_sequences)*sequence_stride+q*row_stride+k]))) {
                    expected[s*tiles+t] = std::max(expected[s*tiles+t], (k/256+1)*256);
                }
            }
        }
    }
    half * dmask = nullptr; int * dout = nullptr;
    check(hipMalloc(&dmask, mask.size()*sizeof(half)));
    check(hipMalloc(&dout, got.size()*sizeof(int)));
    check(hipMemcpy(dmask,mask.data(),mask.size()*sizeof(half),hipMemcpyHostToDevice));
    flash_attn_mask_to_KV_max<TILE><<<dim3(tiles,sequences),128>>>(
        reinterpret_cast<half2 *>(dmask),dout,keys/256,queries,mask_sequences,row_stride/2,sequence_stride/2);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(got.data(),dout,got.size()*sizeof(int),hipMemcpyDeviceToHost));
    for (size_t i=0;i<got.size();++i) if (got[i]!=expected[i]) {
        std::fprintf(stderr,"FAIL tile=%d queries=%d sequences=%d mask_sequences=%d layout=%d masked=%d index=%zu got=%d expected=%d\n",
                     TILE,queries,sequences,mask_sequences,layout,masked,i,got[i],expected[i]);
        std::exit(1);
    }
    check(hipFree(dout)); check(hipFree(dmask)); return got.size();
}

template<int TILE> static size_t suite() {
    size_t n=0;
    for (int q : {1,17,1025,2044,2048}) for (int layout : {0,1,2}) for (bool masked : {false,true}) {
        for (auto shape : {std::pair<int,int>{1,1},{4,1},{4,2},{4,4}}) {
            n += run<TILE>(q,shape.first,shape.second,layout,masked);
        }
    }
    return n;
}

int main() {
    check(hipSetDevice(0));
    size_t n=suite<1>()+suite<2>()+suite<4>()+suite<8>()+suite<16>()+suite<32>()+suite<64>();
    std::printf("PASS: 840 cases; %zu exact KV tile limits, tight/poisoned/strided masks, tails and sequence broadcasting\n",n);
}

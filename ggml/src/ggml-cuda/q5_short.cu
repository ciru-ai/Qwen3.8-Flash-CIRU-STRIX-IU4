/*
 * Adapted from Gufo SmallBatchKQuantExactFp32GEMMKernel and Q5_K decoding.
 * MIT License
 * Copyright (c) 2026 gufo contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#if defined(GGML_USE_HIP)
#include "q5_short.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(CIRU_Q5_SHORT_GFX1151)
namespace {
struct q5_block {
    __half d, dmin;
    uint8_t scales[12], qh[32], qs[128];
};
static_assert(sizeof(q5_block)==176);
struct sub16 {
    union { int8_t q[16]; uint32_t words[4]; };
    float scale, offset;
};
__device__ inline void decode16(const void * __restrict__ row, size_t sub, sub16 & out) {
    const auto & block=static_cast<const q5_block *>(row)[sub/16];
    const size_t s=(sub/2)%8, lane0=(sub%2)*16, base=(s/2)*32+lane0;
    const unsigned shift=(s%2)==0?0u:4u;
    uint32_t packed[4], high[4];
    __builtin_memcpy(packed,block.qs+base,16); __builtin_memcpy(high,block.qh+lane0,16);
#pragma unroll
    for(int i=0;i<4;++i) {
        const uint32_t low=(packed[i]>>shift)&0x0f0f0f0fu;
        const uint32_t bit=(high[i]>>unsigned(s))&0x01010101u;
        out.words[i]=low|(bit<<4u);
    }
    const size_t b=s&3u;
    const unsigned low=block.scales[b], middle=block.scales[b+4], high_scale=block.scales[b+8];
    const unsigned mask=0u-unsigned(s>>2u);
    const unsigned sl=low&63u, su=(high_scale&15u)|((low>>6u)<<4u);
    const unsigned ml=middle&63u, mu=(high_scale>>4u)|((middle>>6u)<<4u);
    const uint8_t sc=uint8_t(sl^((sl^su)&mask)), mn=uint8_t(ml^((ml^mu)&mask));
    out.scale=__half2float(block.d)*float(sc); out.offset=__half2float(block.dmin)*float(mn);
}
template<int Offset=16>
__device__ __forceinline__ float reduce32(float value) {
    int other=__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,value),(Offset<<10)|31);
    float sum=value+__builtin_bit_cast(float,other);
    if constexpr(Offset>1) return reduce32<Offset/2>(sum);
    return sum;
}
__device__ __forceinline__ __attribute__((convergent)) void sync_tile() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}
__launch_bounds__(512,2) __global__
void q5_short_f32_t4_kernel(const void * __restrict__ w,const float * __restrict__ x,
                            float * __restrict__ y,size_t wide_m,size_t wide_k) {
    const uint32_t m=uint32_t(wide_m), k=uint32_t(wide_k);
    constexpr uint32_t stride=20, tile_stride=32*stride;
    __shared__ float staged_x[4*tile_stride];
    __shared__ float staged_sums[4*32];
    const uint32_t lane=threadIdx.x%32, row_base=(blockIdx.x*16+threadIdx.x/32)*4;
    const uint32_t num_sub=k/16, row_bytes=(k/256)*176;
    float sums[4][4]={};
    for(uint32_t tile_base=0;tile_base<num_sub;tile_base+=32) {
        for(uint32_t flat=threadIdx.x;flat<4*32;flat+=blockDim.x) {
            const uint32_t token=flat/32, sub=flat%32, source_sub=tile_base+sub;
            float * dst=staged_x+token*tile_stride+sub*stride;
            float total=0.0f;
#pragma unroll
            for(uint32_t v=0;v<4;++v) {
                float4 value={0,0,0,0};
                if(source_sub<num_sub) value=*reinterpret_cast<const float4 *>(x+token*k+source_sub*16+v*4);
                *reinterpret_cast<float4 *>(dst+v*4)=value;
                total+=value.x; total+=value.y; total+=value.z; total+=value.w;
            }
            staged_sums[token*32+sub]=total;
        }
        sync_tile();
        const uint32_t sub=tile_base+lane;
        if(sub<num_sub) {
            sub16 decoded[4];
#pragma unroll
            for(uint32_t r=0;r<4;++r) {
                const uint32_t row=row_base+r, safe_row=row<m?row:m-1;
                decode16(static_cast<const uint8_t *>(w)+safe_row*row_bytes,sub,decoded[r]);
            }
#pragma unroll
            for(uint32_t token_base=0;token_base<4;token_base+=2) {
                float dots[4][2]={};
#pragma unroll
                for(uint32_t group=0;group<4;++group) {
#pragma unroll
                    for(uint32_t local_token=0;local_token<2;++local_token) {
                        const uint32_t token=token_base+local_token;
                        const float4 xv=*reinterpret_cast<const float4 *>(staged_x+token*tile_stride+lane*stride+group*4);
#pragma unroll
                        for(uint32_t r=0;r<4;++r) {
                            const int8_t * q=decoded[r].q+group*4;
                            dots[r][local_token]+=float(q[0])*xv.x;
                            dots[r][local_token]+=float(q[1])*xv.y;
                            dots[r][local_token]+=float(q[2])*xv.z;
                            dots[r][local_token]+=float(q[3])*xv.w;
                        }
                    }
                }
#pragma unroll
                for(uint32_t r=0;r<4;++r) {
#pragma unroll
                    for(uint32_t local_token=0;local_token<2;++local_token) {
                        const uint32_t token=token_base+local_token;
                        sums[r][token]+=(decoded[r].scale*dots[r][local_token])-
                            (decoded[r].offset*staged_sums[token*32+lane]);
                    }
                }
            }
        }
        sync_tile();
    }
#pragma unroll
    for(uint32_t r=0;r<4;++r) {
#pragma unroll
        for(uint32_t token=0;token<4;++token) sums[r][token]=reduce32(sums[r][token]);
    }
    if(lane==0) {
#pragma unroll
        for(uint32_t r=0;r<4;++r) {
            const uint32_t row=row_base+r;
            if(row>=m) continue;
#pragma unroll
            for(uint32_t token=0;token<4;++token) y[token*m+row]=sums[r][token];
        }
    }
}

// T5 keeps the existing T4 kernel unchanged. The odd token tail is explicitly guarded.
__launch_bounds__(512,2) __global__
void q5_short_f32_t5_kernel(const void * __restrict__ w,const float * __restrict__ x,
                            float * __restrict__ y,size_t wide_m,size_t wide_k) {
    const uint32_t m=uint32_t(wide_m), k=uint32_t(wide_k);
    constexpr uint32_t stride=20, tile_stride=32*stride;
    __shared__ float staged_x[5*tile_stride];
    __shared__ float staged_sums[5*32];
    const uint32_t lane=threadIdx.x%32, row_base=(blockIdx.x*16+threadIdx.x/32)*4;
    const uint32_t num_sub=k/16, row_bytes=(k/256)*176;
    float sums[4][5]={};
    for(uint32_t tile_base=0;tile_base<num_sub;tile_base+=32) {
        for(uint32_t flat=threadIdx.x;flat<5*32;flat+=blockDim.x) {
            const uint32_t token=flat/32, sub=flat%32, source_sub=tile_base+sub;
            float * dst=staged_x+token*tile_stride+sub*stride;
            float total=0.0f;
#pragma unroll
            for(uint32_t v=0;v<4;++v) {
                float4 value={0,0,0,0};
                if(source_sub<num_sub) value=*reinterpret_cast<const float4 *>(x+token*k+source_sub*16+v*4);
                *reinterpret_cast<float4 *>(dst+v*4)=value;
                total+=value.x; total+=value.y; total+=value.z; total+=value.w;
            }
            staged_sums[token*32+sub]=total;
        }
        sync_tile();
        const uint32_t sub=tile_base+lane;
        if(sub<num_sub) {
            sub16 decoded[4];
#pragma unroll
            for(uint32_t r=0;r<4;++r) {
                const uint32_t row=row_base+r, safe_row=row<m?row:m-1;
                decode16(static_cast<const uint8_t *>(w)+safe_row*row_bytes,sub,decoded[r]);
            }
#pragma unroll
            for(uint32_t token_base=0;token_base<5;token_base+=2) {
                float dots[4][2]={};
#pragma unroll
                for(uint32_t group=0;group<4;++group) {
#pragma unroll
                    for(uint32_t local_token=0;local_token<2;++local_token) {
                        const uint32_t token=token_base+local_token;
                        if (token >= 5) continue;
                        const float4 xv=*reinterpret_cast<const float4 *>(staged_x+token*tile_stride+lane*stride+group*4);
#pragma unroll
                        for(uint32_t r=0;r<4;++r) {
                            const int8_t * q=decoded[r].q+group*4;
                            dots[r][local_token]+=float(q[0])*xv.x;
                            dots[r][local_token]+=float(q[1])*xv.y;
                            dots[r][local_token]+=float(q[2])*xv.z;
                            dots[r][local_token]+=float(q[3])*xv.w;
                        }
                    }
                }
#pragma unroll
                for(uint32_t r=0;r<4;++r) {
#pragma unroll
                    for(uint32_t local_token=0;local_token<2;++local_token) {
                        const uint32_t token=token_base+local_token;
                        if (token >= 5) continue;
                        sums[r][token]+=(decoded[r].scale*dots[r][local_token])-
                            (decoded[r].offset*staged_sums[token*32+lane]);
                    }
                }
            }
        }
        sync_tile();
    }
#pragma unroll
    for(uint32_t r=0;r<4;++r) {
#pragma unroll
        for(uint32_t token=0;token<5;++token) sums[r][token]=reduce32(sums[r][token]);
    }
    if(lane==0) {
#pragma unroll
        for(uint32_t r=0;r<4;++r) {
            const uint32_t row=row_base+r;
            if(row>=m) continue;
#pragma unroll
            for(uint32_t token=0;token<5;++token) y[token*m+row]=sums[r][token];
        }
    }
}
} // namespace
#endif

bool ggml_cuda_q5_short_f32(ggml_backend_cuda_context & ctx,const ggml_tensor * w,
                           const ggml_tensor * x,ggml_tensor * y) {
#if defined(CIRU_Q5_SHORT_GFX1151)
    static const bool enabled=[]() {const char * p=std::getenv("CIRU_Q5_SHORT_F32");return p && std::strcmp(p,"1")==0;}();
    static const bool enabled_t5=[]() {const char * p=std::getenv("CIRU_Q5_SHORT_F32_T5");return p && std::strcmp(p,"1")==0;}();
    const bool is_t5 = x->ne[1] == 5 && enabled_t5;
    if(!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_OFFSET_AMD+0x1151 ||
       w->type!=GGML_TYPE_Q5_K || w->ne[0]!=2560 || (w->ne[1]!=10240 && w->ne[1]!=248320) ||
       w->ne[2]!=1 || w->ne[3]!=1 || x->type!=GGML_TYPE_F32 || x->ne[0]!=2560 || (x->ne[1]!=4 && !is_t5) ||
       x->ne[2]!=1 || x->ne[3]!=1 || y->type!=GGML_TYPE_F32 || y->ne[0]!=w->ne[1] || y->ne[1]!=x->ne[1] ||
       y->ne[2]!=1 || y->ne[3]!=1 || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(y)) return false;
    if (is_t5) {
        q5_short_f32_t5_kernel<<<(w->ne[1]+63)/64,512,0,ctx.stream()>>>(w->data,static_cast<const float *>(x->data),
            static_cast<float *>(y->data),w->ne[1],w->ne[0]);
    } else {
    q5_short_f32_t4_kernel<<<(w->ne[1]+63)/64,512,0,ctx.stream()>>>(w->data,static_cast<const float *>(x->data),
        static_cast<float *>(y->data),w->ne[1],w->ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
    static bool logged[2]={false,false};
    if(!logged[is_t5]) { GGML_LOG_INFO("CIRU_Q5_SHORT_F32 active T=%lld K=2560 M=%lld\n",(long long)x->ne[1],(long long)w->ne[1]);logged[is_t5]=true; }
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(w); GGML_UNUSED(x); GGML_UNUSED(y); return false;
#endif
}
#endif

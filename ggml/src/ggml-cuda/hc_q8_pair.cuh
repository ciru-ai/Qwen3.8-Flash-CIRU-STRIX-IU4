#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

// Prototype interface. All pointers are device pointers. Wdown and Winject are
// the original, separate GGUF Q8_0 matrices [10240,320] and [10240,4]. Xbf16
// is the existing MMB BF16 activation [T,10240]. Outputs retain the two GGML
// F32 row-major layouts [T,320] and [T,4]. T may have a partial 128-token tail.
extern "C" hipError_t hc_q8_pair_launch(
        const uint8_t * Wdown, const uint8_t * Winject,
        const uint16_t * Xbf16, float * Ydown, float * Yinject,
        int T, hipStream_t stream);

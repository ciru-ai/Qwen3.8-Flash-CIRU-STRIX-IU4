#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

// Isolated Gufo-derived Q5_K wave64 operator. The caller allocates transient
// scratch from the existing ggml HIP pool and supplies the current stream.
extern "C" std::size_t c1_q5_wave64_q8_bytes(std::size_t tokens, std::size_t k);
extern "C" hipError_t c1_q5_wave64_launch(const void * w, const float * x,
                                           void * q8, float * y,
                                           std::size_t tokens, std::size_t m,
                                           std::size_t k, hipStream_t stream);

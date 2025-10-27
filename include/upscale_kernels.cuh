#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace upscaler {

void bilinear2x_rgb(
    const uint8_t* src, int srcW, int srcH, int srcPitchBytes,
    uint8_t*       dst, int dstW, int dstH, int dstPitchBytes,
    cudaStream_t stream = nullptr);

} 

#pragma once 
#include "common.hpp"

namespace upscaler {

void upscaleCPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t*       dst, int dstWidth, int dstHeight,
    const Options& options);

void upscaleCUDA(
    const uint8_t* dSrc, int srcWidth, int srcHeight, int srcPitch,
    uint8_t*       dDst, int dstWidth, int dstHeight, int dstPitch,
    const Options& options,
    cudaStream_t stream = nullptr);
} // namespace upscaler
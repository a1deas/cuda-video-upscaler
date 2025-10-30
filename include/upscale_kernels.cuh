#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "common.hpp"

namespace upscaler {

// 2× upscale, RGB (3 bytes/px), pitch-aware
void bilinear2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* dst, int dstWidth, int dstHeight, int dstPitchBytes,
    const Options& options,
    cudaStream_t stream = nullptr);

void bicubic2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* dst, int dstWidth, int dstHeight, int dstPitchBytes,
    const Options& options,
    cudaStream_t stream = nullptr);

void upscaleCUDA(
    const uint8_t* dSrc, int srcWidth, int srcHeight, int srcPitch,
    uint8_t* dDst, int dstWidth, int dstHeight, int dstPitch,
    const Options& options, 
    cudaStream_t stream = nullptr);
} // namespace upscaler

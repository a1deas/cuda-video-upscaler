#pragma once
#include <cstdint>
#include "common.hpp"

// CPU upscale realization
namespace upscaler {

void bilinear2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    Border border = Border::Clamp, bool gamma = false);

void bicubic2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight, 
    uint8_t* dst, int dstWidth, int dstHeight,
    Border border = Border::Clamp, bool gamma = false);

void upscaleCPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    const Options& options);
} // namespace upscaler
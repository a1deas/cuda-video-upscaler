#pragma once
#include <cstdint>
#include "common.hpp"

// CPU upscale realization
namespace upscaler {

void upscaleCPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    const Options& options);
} // namespace upscaler
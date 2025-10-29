#pragma once
#include <cstdint>

namespace upscaler {

void bilinear2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight, 
    uint8_t* dst, int dstWidth, int dstHeight);
    
} // namespace upscaler
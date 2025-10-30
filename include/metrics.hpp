#pragma once 
#include <cstdint>
#include <tuple>

namespace upscaler {
    // PSNR: the more, the better
    double psnrRGB(const uint8_t* a, const uint8_t* b, int width, int height);

    // Simple SSIM(one window = full picture), 0..1
    double ssimRGB(const uint8_t* a, const uint8_t* b, int width, int height);
} // namespace upscaler
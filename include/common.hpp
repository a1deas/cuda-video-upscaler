#pragma once 
#include <cstdint>

namespace upscaler {

enum class Mode { Bilinear, Bicubic };
enum class Device { CUDA, CPU };
enum class Border { Clamp, Reflect };

struct Options { 
    int scale = 2;                  // 2 | 4
    Mode mode = Mode::Bilinear;     // Bilinear | Bicubic
    Border border = Border::Clamp;  // Clamp    | Reflect
    bool gammaCorrect = false;      // sRGB->Linear interpolation
};

} // namespace upscaler
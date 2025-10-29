#include <upscale_cpu.hpp>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace upscaler {

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void bilinear2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t*       dst, int dstWidth, int dstHeight)
{
     const float scaleX = (float)srcWidth / (float)dstWidth;
     const float scaleY = (float)srcHeight / (float)dstHeight;

    #pragma omp parallel for if(dstHeight > 64)
    for (int y = 0; y < dstHeight; ++y) {
        const float srcYf = (y + 0.5f) * scaleY - 0.5f;
        const int iy = (int)std::floor(srcYf);
        const float fy = srcYf - iy;

        const int iy0 = clampi(iy, 0, srcHeight - 1);
        const int iy1 = clampi(iy + 1, 0, srcHeight - 1);

        const uint8_t* row0 = src + iy0 * srcWidth * 3;
        const uint8_t* row1 = src + iy1 * srcWidth * 3;

        uint8_t* drow = dst + y * dstWidth * 3;

        for (int x = 0; x < dstWidth; ++x) {
            const float srcXf = (x + 0.5f) * scaleX - 0.5f;
            const int ix = (int)std::floor(srcXf);
            const float fx = srcXf - ix;

            const int ix0 = clampi(ix, 0, srcWidth - 1);
            const int ix1 = clampi(ix + 1, 0, srcWidth - 1);
            
            const uint8_t* p00 = row0 + ix0 * 3;
            const uint8_t* p10 = row0 + ix1 * 3;
            const uint8_t* p01 = row1 + ix0 * 3;
            const uint8_t* p11 = row1 + ix1 * 3;

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w10 = (      fx) * (1.0f - fy);
            const float w01 = (1.0f - fx) * (      fy);
            const float w11 = (      fx) * (      fy);

            uint8_t* dp = drow + x * 3;

            for (int c = 0; c < 3; ++c) {
                float v = w00 * p00[c]
                        + w10 * p10[c]
                        + w01 * p01[c]
                        + w11 * p11[c];
                int out = (int)std::lround(v);
                dp[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
            }
        }
    }
}

} // namespace upscaler
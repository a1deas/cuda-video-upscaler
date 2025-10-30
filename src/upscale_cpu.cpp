#include "upscale_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace upscaler {

// Border type: clamp
static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
// Border type: reflect
static inline int reflectIndex(int i, int n) {
    if (n <= 1) return 0;
    int p = 2*(n-1);
    int m = i % p; if (m < 0) m += p;
    return (m < n) ? m : p - m;
}

// Helper to choose border type
static inline int borderIdx(int i, int n, Border b) {
    return (b == Border::Clamp) ? clampi(i, 0, n-1) : reflectIndex(i, n);
}

// sRGB gamma helpers
static inline float srgb2lin(float v){
    return (v <= 0.04045f) ? (v/12.92f) : std::pow((v+0.055f)/1.055f, 2.4f);
}
static inline float lin2srgb(float x){
    return (x <= 0.0031308f) ? (12.92f*x) : (1.055f*std::pow(x, 1.0f/2.4f) - 0.055f);
}

static inline float wCubic(float x, float a = -0.5f){
    x = std::fabs(x);
    if (x < 1.0f)      return (a+2)*x*x*x - (a+3)*x*x + 1;
    else if (x < 2.0f) return a*x*x*x - 5*a*x*x + 8*a*x - 4*a;
    else               return 0.0f;
}

static void bilinear2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    Border border, bool gamma)
{
    const float scaleX = float(srcWidth) / float(dstWidth);
    const float scaleY = float(srcHeight) / float(dstHeight);

    #pragma omp parallel for if (dstHeight > 64)
    for (int y = 0; y < dstHeight; ++y) {
        const float srcYf = (y + 0.5f) * scaleY - 0.5f;
        const int iy = int(std::floor(srcYf));
        const float fy = srcYf - iy;

        const int iy0 = borderIdx(iy,   srcHeight, border);
        const int iy1 = borderIdx(iy+1, srcHeight, border);

        const uint8_t* row0 = src + iy0 * srcWidth * 3;
        const uint8_t* row1 = src + iy1 * srcWidth * 3;

        uint8_t* drow = dst + y * dstWidth * 3;

        for (int x = 0; x < dstWidth; ++x) {
            const float srcXf = (x + 0.5f) * scaleX - 0.5f;
            const int   ix    = int(std::floor(srcXf));
            const float fx    = srcXf - ix;

            const int ix0 = borderIdx(ix,     srcWidth, border);
            const int ix1 = borderIdx(ix + 1, srcWidth, border);

            const uint8_t* p00 = row0 + ix0 * 3;
            const uint8_t* p10 = row0 + ix1 * 3;
            const uint8_t* p01 = row1 + ix0 * 3;
            const uint8_t* p11 = row1 + ix1 * 3;

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w10 = (fx) * (1.0f - fy);
            const float w01 = (1.0f - fx) * (fy);
            const float w11 = (fx) * (fy);

            uint8_t* dp = drow + x * 3;

            for (int c = 0; c < 3; ++c) {
                float v00 = p00[c], v10 = p10[c], v01 = p01[c], v11 = p11[c];
                if (gamma) {
                    v00 = srgb2lin(v00/255.f); v10 = srgb2lin(v10/255.f);
                    v01 = srgb2lin(v01/255.f); v11 = srgb2lin(v11/255.f);
                    const float lin = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                    const float s   = lin2srgb(std::clamp(lin, 0.f, 1.f));
                    const int out   = (int)std::lround(s * 255.f);
                    dp[c] = (uint8_t)std::clamp(out, 0, 255);
                } else {
                    const float s = w00*v00 + w10*v10 + w01*v01 + w11*v11;
                    const int out = (int)std::lround(s);
                    dp[c] = (uint8_t)std::clamp(out, 0, 255);
                }
            }
        }
    }
}

static void bicubic2xRGB_CPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    Border border, bool gamma)
{
    const float scaleX = float(srcWidth) / float(dstWidth);
    const float scaleY = float(srcHeight) / float(dstHeight);

    #pragma omp parallel for if (dstHeight > 32)
    for (int y = 0; y < dstHeight; ++y) {
        const float srcYf = (y + 0.5f) * scaleY - 0.5f;
        const int iy = int(std::floor(srcYf));
        const float fy = srcYf - iy;

        const int ys[4] = {
            borderIdx(iy-1, srcHeight, border),
            borderIdx(iy+0, srcHeight, border),
            borderIdx(iy+1, srcHeight, border),
            borderIdx(iy+2, srcHeight, border)
        };
        const float wy[4] = {
            wCubic(1.f + fy), wCubic(0.f + fy),
            wCubic(1.f - fy), wCubic(2.f - fy)
        };

        uint8_t* drow = dst + y*dstWidth*3;

        for (int x = 0; x < dstWidth; ++x) {
            const float srcXf = (x + 0.5f) * scaleX - 0.5f;
            const int   ix    = int(std::floor(srcXf));
            const float fx    = srcXf - ix;

            const int xs[4] = {
                borderIdx(ix-1, srcWidth, border),
                borderIdx(ix+0, srcWidth, border),
                borderIdx(ix+1, srcWidth, border),
                borderIdx(ix+2, srcWidth, border)
            };
            const float wx[4] = {
                wCubic(1.f + fx), wCubic(0.f + fx),
                wCubic(1.f - fx), wCubic(2.f - fx)
            };

            float sum[3] = {0,0,0};
            for (int j = 0; j < 4; ++j){
                const uint8_t* r = src + ys[j]*srcWidth*3;
                const float wyj = wy[j];
                for (int i = 0; i < 4; ++i){
                    const uint8_t* p = r + xs[i]*3;
                    const float w = wyj * wx[i];
                    if (gamma){
                        const float r0 = srgb2lin(p[0]/255.f);
                        const float g0 = srgb2lin(p[1]/255.f);
                        const float b0 = srgb2lin(p[2]/255.f);
                        sum[0] += w*r0; sum[1] += w*g0; sum[2] += w*b0;
                    } else {
                        sum[0] += w*p[0]; sum[1] += w*p[1]; sum[2] += w*p[2];
                    }
                }
            }
            uint8_t* dp = drow + x*3;
            if (gamma){
                for (int c=0;c<3;++c){
                    const float s = lin2srgb(std::clamp(sum[c], 0.f, 1.f));
                    const int out = (int)std::lround(s * 255.f);
                    dp[c] = (uint8_t)std::clamp(out, 0, 255);
                }
            } else {
                for (int c=0;c<3;++c){
                    const int out = (int)std::lround(sum[c]);
                    dp[c] = (uint8_t)std::clamp(out, 0, 255);
                }
            }
        }
    }
}

void upscaleCPU(
    const uint8_t* src, int srcWidth, int srcHeight,
    uint8_t* dst, int dstWidth, int dstHeight,
    const Options& options)
{
    if (options.scale == 2) {
        if (options.mode == Mode::Bilinear)
            bilinear2xRGB_CPU(src, srcWidth, srcHeight, dst, dstWidth, dstHeight, options.border, options.gammaCorrect);
        else
            bicubic2xRGB_CPU (src, srcWidth, srcHeight, dst, dstWidth, dstHeight, options.border, options.gammaCorrect);
    } else if (options.scale == 4) {
        std::vector<uint8_t> mid((size_t)(srcWidth*2) * (srcHeight*2) * 3);
        Options opt2 = options; opt2.scale = 2;
        upscaleCPU(src, srcWidth, srcHeight, mid.data(), srcWidth*2, srcHeight*2, opt2);
        upscaleCPU(mid.data(), srcWidth*2, srcHeight*2, dst, dstWidth, dstHeight, opt2);
    } else {
        throw std::runtime_error("CPU: only scale 2 or 4 supported");
    }
}

} // namespace upscaler

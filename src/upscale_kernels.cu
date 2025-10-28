#include "upscale_kernels.cuh"
#include <cmath>

namespace {

__device__ __forceinline__ int clampInt(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float wCubic(float x, float a = -0.5f) {
    x = fabsf(x);
    if (x < 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    } else {
        return 0.0f;
    }
}

__global__ void kBilinear2xRGB(
    const uint8_t* __restrict__ src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t*       __restrict__ dst, int dstWidth, int dstHeight, int dstPitchBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    // reverse mapping for scale=2 (center-aligned)
    const float srcX = (x + 0.5f) * 0.5f - 0.5f;
    const float srcY = (y + 0.5f) * 0.5f - 0.5f;

    const int   ix = (int)floorf(srcX);
    const int   iy = (int)floorf(srcY);
    const float fx = srcX - ix;
    const float fy = srcY - iy;

    const int ix0 = clampInt(ix,     0, srcWidth  - 1);
    const int iy0 = clampInt(iy,     0, srcHeight - 1);
    const int ix1 = clampInt(ix + 1, 0, srcWidth  - 1);
    const int iy1 = clampInt(iy + 1, 0, srcHeight - 1);

    const uint8_t* row0 = src + iy0 * srcPitchBytes;
    const uint8_t* row1 = src + iy1 * srcPitchBytes;

    const uint8_t* p00 = row0 + ix0 * 3;
    const uint8_t* p10 = row0 + ix1 * 3;
    const uint8_t* p01 = row1 + ix0 * 3;
    const uint8_t* p11 = row1 + ix1 * 3;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = (      fx) * (1.0f - fy);
    const float w01 = (1.0f - fx) * (      fy);
    const float w11 = (      fx) * (      fy);

    uint8_t* drow = dst + y * dstPitchBytes;
    uint8_t* dpix = drow + x * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = w00 * p00[c]
                + w10 * p10[c]
                + w01 * p01[c]
                + w11 * p11[c];
        int out = __float2int_rn(v);
        dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
    }
}

__global__ void kBicubic2xRGB(
    const uint8_t* __restrict__ src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t*       __restrict__ dst, int dstWidth, int dstHeight, int dstPitchBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    const float srcX = (x + 0.5f) * 0.5f - 0.5f;
    const float srcY = (y + 0.5f) * 0.5f - 0.5f;

    const int   ix = (int)floorf(srcX);
    const int   iy = (int)floorf(srcY);
    const float fx = srcX - ix;
    const float fy = srcY - iy;

    // 4×4 neighborhood around (ix,iy)
    const int x0 = clampInt(ix - 1, 0, srcWidth  - 1);
    const int x1 = clampInt(ix + 0, 0, srcWidth  - 1);
    const int x2 = clampInt(ix + 1, 0, srcWidth  - 1);
    const int x3 = clampInt(ix + 2, 0, srcWidth  - 1);

    const int y0 = clampInt(iy - 1, 0, srcHeight - 1);
    const int y1 = clampInt(iy + 0, 0, srcHeight - 1);
    const int y2 = clampInt(iy + 1, 0, srcHeight - 1);
    const int y3 = clampInt(iy + 2, 0, srcHeight - 1);

    const float wx[4] = {
        wCubic(1.0f + fx), // ix-1
        wCubic(0.0f + fx), // ix
        wCubic(1.0f - fx), // ix+1
        wCubic(2.0f - fx)  // ix+2
    };
    const float wy[4] = {
        wCubic(1.0f + fy),
        wCubic(0.0f + fy),
        wCubic(1.0f - fy),
        wCubic(2.0f - fy)
    };

    float sum[3] = {0,0,0};
    const int xs[4] = { x0, x1, x2, x3 };
    const int ys[4] = { y0, y1, y2, y3 };

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t* row = src + ys[j] * srcPitchBytes;
        float wyj = wy[j];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* p = row + xs[i] * 3;
            float wxy = wyj * wx[i];
            sum[0] += wxy * p[0];
            sum[1] += wxy * p[1];
            sum[2] += wxy * p[2];
        }
    }

    uint8_t* drow = dst + y * dstPitchBytes;
    uint8_t* dpix = drow + x * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int out = __float2int_rn(sum[c]);
        dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
    }
}

} // anonymous namespace

namespace upscaler {

void bilinear2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t*       dst, int dstWidth, int dstHeight, int dstPitchBytes,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);
    kBilinear2xRGB<<<grid, block, 0, stream>>>(
        src, srcWidth, srcHeight, srcPitchBytes,
        dst, dstWidth, dstHeight, dstPitchBytes);
}

void bicubic2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t*       dst, int dstWidth, int dstHeight, int dstPitchBytes,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);
    kBicubic2xRGB<<<grid, block, 0, stream>>>(
        src, srcWidth, srcHeight, srcPitchBytes,
        dst, dstWidth, dstHeight, dstPitchBytes);
}

} // namespace upscaler

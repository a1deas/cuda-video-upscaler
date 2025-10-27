#include "upscale_kernels.cuh"

namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void k_bilinear2x_rgb(
    const uint8_t* __restrict__ src, int srcW, int srcH, int srcPitch,
    uint8_t*       __restrict__ dst, int dstW, int dstH, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // dst x
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // dst y
    if (x >= dstW || y >= dstH) return;

    // reverse (scale=2)
    const float srcX = (x + 0.5f) * 0.5f - 0.5f;
    const float srcY = (y + 0.5f) * 0.5f - 0.5f;

    const int ix = (int)floorf(srcX);
    const int iy = (int)floorf(srcY);
    const float fx = srcX - ix;
    const float fy = srcY - iy;

    const int ix0 = clampi(ix,     0, srcW - 1);
    const int iy0 = clampi(iy,     0, srcH - 1);
    const int ix1 = clampi(ix + 1, 0, srcW - 1);
    const int iy1 = clampi(iy + 1, 0, srcH - 1);

    const uint8_t* row0 = src + iy0 * srcPitch;
    const uint8_t* row1 = src + iy1 * srcPitch;

    const uint8_t* p00 = row0 + ix0 * 3;
    const uint8_t* p10 = row0 + ix1 * 3;
    const uint8_t* p01 = row1 + ix0 * 3;
    const uint8_t* p11 = row1 + ix1 * 3;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = (      fx) * (1.0f - fy);
    const float w01 = (1.0f - fx) * (      fy);
    const float w11 = (      fx) * (      fy);

    uint8_t* drow = dst + y * dstPitch;
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

} 

namespace upscaler {

void bilinear2x_rgb(
    const uint8_t* src, int srcW, int srcH, int srcPitchBytes,
    uint8_t*       dst, int dstW, int dstH, int dstPitchBytes,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    k_bilinear2x_rgb<<<grid, block, 0, stream>>>(
        src, srcW, srcH, srcPitchBytes,
        dst, dstW, dstH, dstPitchBytes);
}

}

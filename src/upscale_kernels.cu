#include "upscale_kernels.cuh"
#include <cmath>
#include <cstdio>

namespace {

// Border Type: Clamp
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Border Type: Reflect
__device__ __forceinline__ int reflectIndex(int i, int n) {
    if (n <= 1) return 0;
    int p = 2 * (n - 1);
    int m = i % p; if (m < 0) m += p;
    return (m < n) ? m : p - m;
}

// Helper to choose border type
__device__ __forceinline__ int borderIdx(int i, int n, upscaler::Border b){
    return (b == upscaler::Border::Clamp) ? clampi(i,0,n-1) : reflectIndex(i,n);
}

// sRGB gamma helpers
__device__ __forceinline__ float srgb2lin(float v){
    return (v <= 0.04045f) ? (v/12.92f) : powf((v+0.055f)/1.055f, 2.4f);
}
__device__ __forceinline__ float lin2srgb(float x){
    return (x <= 0.0031308f) ? (12.92f*x) : (1.055f*powf(x, 1.f/2.4f) - 0.055f);
}

__device__ __forceinline__ float wCubic(float x, float a=-0.5f){
    x = fabsf(x);
    if (x < 1.f)        return (a+2)*x*x*x - (a+3)*x*x + 1;
    else if (x < 2.f)   return a*x*x*x - 5*a*x*x + 8*a*x - 4*a;
    else                return 0.f;
}


__global__ void kBilinear2xRGB(
    const uint8_t* __restrict__ src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* __restrict__ dst, int dstWidth, int dstHeight, int dstPitchBytes,
    upscaler::Border border, bool gammaCorrect)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    // reverse mapping (scale=2)
    const float srcX = (x + 0.5f) * 0.5f - 0.5f;
    const float srcY = (y + 0.5f) * 0.5f - 0.5f;

    const int   ix = (int)floorf(srcX);
    const int   iy = (int)floorf(srcY);
    const float fx = srcX - ix;
    const float fy = srcY - iy;

    const int ix0 = borderIdx(ix,   srcWidth,  border);
    const int ix1 = borderIdx(ix+1, srcWidth,  border);
    const int iy0 = borderIdx(iy,   srcHeight, border);
    const int iy1 = borderIdx(iy+1, srcHeight, border);

    const uint8_t* row0 = src + (size_t)iy0 * srcPitchBytes;
    const uint8_t* row1 = src + (size_t)iy1 * srcPitchBytes;

    const uint8_t* p00 = row0 + ix0 * 3;
    const uint8_t* p10 = row0 + ix1 * 3;
    const uint8_t* p01 = row1 + ix0 * 3;
    const uint8_t* p11 = row1 + ix1 * 3;

    const float w00 = (1.f - fx) * (1.f - fy);
    const float w10 = (fx) * (1.f - fy);
    const float w01 = (1.f - fx) * (fy);
    const float w11 = (fx) * (fy);

    uint8_t* drow = dst + (size_t)y * dstPitchBytes;
    uint8_t* dpix = drow + x * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        if (!gammaCorrect) {
            float s = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
            int out = __float2int_rn(s);
            dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
        } else {
            float v00 = srgb2lin(p00[c] / 255.f);
            float v10 = srgb2lin(p10[c] / 255.f);
            float v01 = srgb2lin(p01[c] / 255.f);
            float v11 = srgb2lin(p11[c] / 255.f);
            float lin = w00*v00 + w10*v10 + w01*v01 + w11*v11;
            float s = lin2srgb(fminf(fmaxf(lin, 0.f), 1.f));
            int out = __float2int_rn(s * 255.f);
            dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
        }
    }
}


__global__ void kBicubic2xRGB(
    const uint8_t* __restrict__ src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* __restrict__ dst, int dstWidth, int dstHeight, int dstPitchBytes,
    upscaler::Border border, bool gammaCorrect)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    const float srcX = (x + 0.5f) * 0.5f - 0.5f;
    const float srcY = (y + 0.5f) * 0.5f - 0.5f;

    const int ix = (int)floorf(srcX);
    const int iy = (int)floorf(srcY);
    const float fx = srcX - ix;
    const float fy = srcY - iy;

    const int xs[4] = {
        borderIdx(ix-1, srcWidth,  border),
        borderIdx(ix+0, srcWidth,  border),
        borderIdx(ix+1, srcWidth,  border),
        borderIdx(ix+2, srcWidth,  border)
    };
    const int ys[4] = {
        borderIdx(iy-1, srcHeight, border),
        borderIdx(iy+0, srcHeight, border),
        borderIdx(iy+1, srcHeight, border),
        borderIdx(iy+2, srcHeight, border)
    };

    const float wx[4] = {
        wCubic(1.f + fx), wCubic(0.f + fx),
        wCubic(1.f - fx), wCubic(2.f - fx)
    };
    const float wy[4] = {
        wCubic(1.f + fy), wCubic(0.f + fy),
        wCubic(1.f - fy), wCubic(2.f - fy)
    };

    float sum[3] = {0,0,0};

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t* row = src + (size_t)ys[j] * srcPitchBytes;
        const float wyj = wy[j];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* p = row + xs[i] * 3;
            const float wxy = wyj * wx[i];

            if (!gammaCorrect) {
                sum[0] += wxy * p[0];
                sum[1] += wxy * p[1];
                sum[2] += wxy * p[2];
            } else {
                float r = srgb2lin(p[0] / 255.f);
                float g = srgb2lin(p[1] / 255.f);
                float b = srgb2lin(p[2] / 255.f);
                sum[0] += wxy * r;
                sum[1] += wxy * g;
                sum[2] += wxy * b;
            }
        }
    }

    uint8_t* drow = dst + (size_t)y * dstPitchBytes;
    uint8_t* dpix = drow + x * 3;

    if (!gammaCorrect) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            int out = __float2int_rn(sum[c]);
            dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float s = lin2srgb(fminf(fmaxf(sum[c], 0.f), 1.f));
            int out = __float2int_rn(s * 255.f);
            dpix[c] = (uint8_t)(out < 0 ? 0 : (out > 255 ? 255 : out));
        }
    }
}

} // anonymous namespace

namespace upscaler {

static inline void checkCuda(const char* where){
#if !defined(NDEBUG)
    auto err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("[CUDA] %s: %s\n", where, cudaGetErrorString(err));
    }
#endif
}

void bilinear2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* dst, int dstWidth, int dstHeight, int dstPitchBytes,
    const Options& options,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((dstWidth  + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);

    kBilinear2xRGB<<<grid, block, 0, stream>>>(
        src, srcWidth, srcHeight, srcPitchBytes,
        dst, dstWidth, dstHeight, dstPitchBytes,
        options.border, options.gammaCorrect);
    checkCuda("kBilinear2xRGB");
}

void bicubic2xRGB(
    const uint8_t* src, int srcWidth, int srcHeight, int srcPitchBytes,
    uint8_t* dst, int dstWidth, int dstHeight, int dstPitchBytes,
    const Options& options,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((dstWidth  + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);

    kBicubic2xRGB<<<grid, block, 0, stream>>>(
        src, srcWidth, srcHeight, srcPitchBytes,
        dst, dstWidth, dstHeight, dstPitchBytes,
        options.border, options.gammaCorrect);
    checkCuda("kBicubic2xRGB");
}

void upscaleCUDA(
    const uint8_t* dSrc, int srcWidth, int srcHeight, int srcPitch,
    uint8_t* dDst, int dstWidth, int dstHeight, int dstPitch,
    const Options& options,
    cudaStream_t stream)
{
    if (options.scale == 2) {
        (options.mode == Mode::Bilinear)
            ? bilinear2xRGB(dSrc, srcWidth, srcHeight, srcPitch, dDst, dstWidth, dstHeight, dstPitch, options, stream)
            : bicubic2xRGB (dSrc, srcWidth, srcHeight, srcPitch, dDst, dstWidth, dstHeight, dstPitch, options, stream);
    } else if (options.scale == 4) {
        uint8_t* dMid = nullptr; size_t midPitch = 0;
        cudaError_t e1 = cudaMallocPitch(&dMid, &midPitch, (size_t)(srcWidth*2) * 3, (size_t)(srcHeight*2));
        if (e1 != cudaSuccess) return;

        Options o2 = options; o2.scale = 2;
        upscaleCUDA(dSrc, srcWidth, srcHeight, srcPitch,
                    dMid, srcWidth*2, srcHeight*2, (int)midPitch, o2, stream);
        upscaleCUDA(dMid, srcWidth*2, srcHeight*2, (int)midPitch,
                    dDst, dstWidth, dstHeight, dstPitch, o2, stream);

        cudaFree(dMid);
        checkCuda("upscaleCUDA 4x chain");
    } else {
        // later
    }
}

} // namespace upscaler

#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include <vector> 
#include <cuda_runtime.h>
#include "upscale_kernels.cuh"

static void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: video_upscaler <input.png> <output.png>\n";
        return 1;
    }
    int w,h,nc;
    unsigned char* img = stbi_load(argv[1], &w, &h, &nc, 3);
    if (!img) { std::cerr << "Failed to load image\n"; return 1; }

    int dstW = w * 2, dstH = h * 2;

    // pitch-alloc
    unsigned char *d_src=nullptr, *d_dst=nullptr;
    size_t srcPitch=0, dstPitch=0;
    check(cudaMallocPitch(&d_src, &srcPitch, w*3, h), "cudaMallocPitch d_src");
    check(cudaMallocPitch(&d_dst, &dstPitch, dstW*3, dstH), "cudaMallocPitch d_dst");

    // H2D
    for (int y=0; y<h; ++y) {
        check(cudaMemcpy(d_src + y*srcPitch, img + y*w*3, w*3, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D");
    }

    upscaler::bilinear2x_rgb(d_src, w, h, (int)srcPitch,
                             d_dst, dstW, dstH, (int)dstPitch);

    // D2H
    std::vector<unsigned char> out(dstW*dstH*3);
    for (int y=0; y<dstH; ++y) {
        check(cudaMemcpy(out.data() + y*dstW*3, d_dst + y*dstPitch, dstW*3, cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H");
    }

    stbi_write_png(argv[2], dstW, dstH, 3, out.data(), dstW*3);

    cudaFree(d_src); cudaFree(d_dst);
    stbi_image_free(img);
    std::cout << "Saved: " << argv[2] << "\n";
    return 0;
}

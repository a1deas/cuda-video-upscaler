#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "upscale_kernels.cuh"
#include "upscale_cpu.hpp"

enum class Device { CUDA, CPU };
enum class Mode { Bilinear, Bicubic };

struct Args {
    std::string in, out;
    int  scale = 2;          // 2 | 4
    Mode mode  = Mode::Bilinear;
    Device device = Device::CUDA; // CPU | GPU(CUDA)
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 3) {
        throw std::runtime_error("Usage: video_upscaler <in.png> <out.png> [--mode bilinear|bicubic] [--scale 2|4] [--device cuda|cpu]");
    }
    a.in  = argv[1];
    a.out = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--mode" && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "bilinear") a.mode = Mode::Bilinear;
            else if (v == "bicubic")  a.mode = Mode::Bicubic;
            else throw std::runtime_error("Unknown --mode");
        } else if (s == "--scale" && i + 1 < argc) {
            a.scale = std::stoi(argv[++i]);
            if (a.scale != 2 && a.scale != 4)
                throw std::runtime_error("Only scale=2 or 4 supported");
        } else if (s == "--device" && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "cuda") a.device = Device::CUDA;
            else if (v == "cpu")  a.device = Device::CPU;
            else throw std::runtime_error("Unknown --device (use cuda|cpu)");
        } else {
            throw std::runtime_error("Unknown argument: " + s);
        }
    }
    return a;
}

static inline void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    Args args;
    try { args = parseArgs(argc, argv); }
    catch (const std::exception& ex) { std::cerr << ex.what() << "\n"; return 1; }

    int width, height, nc;
    unsigned char* img = stbi_load(args.in.c_str(), &width, &height, &nc, 3);
    if (!img) { std::cerr << "Failed to load image: " << args.in << "\n"; return 1; }

    if (args.device == Device::CPU) {
        // CPU 
        const int dstW = width * args.scale;
        const int dstH = height * args.scale;

        std::vector<unsigned char> out(dstW * dstH * 3);

        if (args.scale == 2) {
            upscaler::bilinear2xRGB_CPU(
                img, width, height,
                out.data(), dstW, dstH);
        } else if (args.scale == 4) {
            // 2x + 2x (CPU)
            std::vector<unsigned char> mid((size_t)(width*2) * (height*2) * 3);
            upscaler::bilinear2xRGB_CPU(img, width, height, mid.data(), width*2, height*2);
            upscaler::bilinear2xRGB_CPU(mid.data(), width*2, height*2, out.data(), dstW, dstH);
        } else {
            std::cerr << "CPU backend: scale must be 2 or 4.\n";
            stbi_image_free(img);
            return 1;
        }

        stbi_write_png(args.out.c_str(), dstW, dstH, 3, out.data(), dstW*3);
        stbi_image_free(img);
        std::cout << "Saved: " << args.out << " (" << dstW << "x" << dstH << ") [CPU]\n";
        return 0;
    }
    const int dstWidth  = width  * args.scale;
    const int dstHeight = height * args.scale;

    // GPU alloc (pitch-aware)
    unsigned char *dSrc = nullptr, *dDst = nullptr;
    size_t srcPitchBytes = 0, dstPitchBytes = 0;
    check(cudaMallocPitch(&dSrc, &srcPitchBytes, width*3,  height),  "cudaMallocPitch dSrc");
    check(cudaMallocPitch(&dDst, &dstPitchBytes, dstWidth*3, dstHeight), "cudaMallocPitch dDst");

    // H2D (row by row with pitch)
    for (int y = 0; y < height; ++y) {
        check(cudaMemcpy(dSrc + y*srcPitchBytes, img + y*width*3, width*3, cudaMemcpyHostToDevice), "H2D");
    }

    if (args.scale == 2) {
        if (args.mode == Mode::Bilinear)
            upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes, dDst, dstWidth, dstHeight, (int)dstPitchBytes);
        else
            upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes, dDst, dstWidth, dstHeight, (int)dstPitchBytes);
    } else { // scale == 4 (2x + 2x) (MVP)
        unsigned char *dMid = nullptr; size_t midPitch = 0;
        check(cudaMallocPitch(&dMid, &midPitch, (width*2)*3, (height*2)), "cudaMallocPitch dMid");

        if (args.mode == Mode::Bilinear) {
            upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes, dMid, width*2, height*2, (int)midPitch);
            upscaler::bilinear2xRGB(dMid, width*2, height*2, (int)midPitch, dDst, dstWidth, dstHeight, (int)dstPitchBytes);
        } else {
            upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes, dMid, width*2, height*2, (int)midPitch);
            upscaler::bicubic2xRGB (dMid, width*2, height*2, (int)midPitch, dDst, dstWidth, dstHeight, (int)dstPitchBytes);
        }
        cudaFree(dMid);
    }

    // D2H
    std::vector<unsigned char> out(dstWidth * dstHeight * 3);
    for (int y = 0; y < dstHeight; ++y) {
        check(cudaMemcpy(out.data() + y*dstWidth*3, dDst + y*dstPitchBytes, dstWidth*3, cudaMemcpyDeviceToHost), "D2H");
    }

    stbi_write_png(args.out.c_str(), dstWidth, dstHeight, 3, out.data(), dstWidth*3);

    cudaFree(dSrc); cudaFree(dDst);
    stbi_image_free(img);

    std::cout << "Saved: " << args.out << " (" << dstWidth << "x" << dstHeight << ")\n";
    return 0;
}

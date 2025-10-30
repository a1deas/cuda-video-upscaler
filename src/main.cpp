#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#include "common.hpp"            
#include "upscale_cpu.hpp"      
#include "upscale_kernels.cuh"   

using upscaler::Mode;
using upscaler::Border;
using upscaler::Options;

enum class Device { CUDA, CPU };

struct Args {
    std::string in, out;            // input    | output
    int   scale = 2;                // 2        | 4
    Mode  mode  = Mode::Bilinear;   // bilinear | bicubic
    Device device = Device::CUDA;   // cpu      | cuda
    Border border = Border::Clamp;  // clamp    | reflect
    bool  gammaCorrect = false;     // sRGB->Linear interpolation
    bool  selftest = false;
    int   benchIters = 0;           // 0=off    | >0 iterations
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: video_upscaler <in.png> <out.png> "
            "[--mode bilinear|bicubic] [--scale 2|4] "
            "[--device cuda|cpu] [--border clamp|reflect] "
            "[--gamma-correct] [--bench N] [--selftest]");
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
        } else if (s == "--border" && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "clamp")   a.border = Border::Clamp;
            else if (v == "reflect") a.border = Border::Reflect;
            else throw std::runtime_error("Unknown --border");
        } else if (s == "--gamma-correct") {
            a.gammaCorrect = true;
        } else if (s == "--bench" && i + 1 < argc) {
            a.benchIters = std::max(0, std::stoi(argv[++i]));
        } else if (s == "--selftest") {
            a.selftest = true;
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

    Options opt;
    opt.scale         = args.scale;
    opt.mode          = args.mode;
    opt.border        = args.border;
    opt.gammaCorrect  = args.gammaCorrect;

    // Selftest
    if (args.selftest) {
        const int W = 64, H = 48;
        std::vector<uint8_t> src(W*H*3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                src[(y*W+x)*3+0] = (uint8_t)((x*3  + y*5 ) & 255);
                src[(y*W+x)*3+1] = (uint8_t)((x*7  + y*11) & 255);
                src[(y*W+x)*3+2] = (uint8_t)((x*13 + y*17) & 255);
            }
        const int dW = W * opt.scale, dH = H * opt.scale;
        std::vector<uint8_t> cpuOut(dW*dH*3), cudaOut(dW*dH*3);

        // CPU
        upscaler::upscaleCPU(src.data(), W, H, cpuOut.data(), dW, dH, opt);

        // CUDA
        uint8_t *dSrc=nullptr, *dDst=nullptr; size_t pS=0, pD=0;
        check(cudaMallocPitch(&dSrc, &pS, (size_t)W*3,  H),  "malloc dSrc");
        check(cudaMallocPitch(&dDst, &pD, (size_t)dW*3, dH), "malloc dDst");
        for (int y=0; y<H; ++y)
            check(cudaMemcpy(dSrc + y*pS, src.data() + (size_t)y*W*3, (size_t)W*3, cudaMemcpyHostToDevice), "H2D");
        upscaler::upscaleCUDA(dSrc, W, H, (int)pS, dDst, dW, dH, (int)pD, opt);
        for (int y=0; y<dH; ++y)
            check(cudaMemcpy(cudaOut.data() + (size_t)y*dW*3, dDst + y*pD, (size_t)dW*3, cudaMemcpyDeviceToHost), "D2H");
        cudaFree(dSrc); cudaFree(dDst);

        float maxAbs = 0.f;
        for (size_t i=0;i<cpuOut.size();++i) {
            float a = cpuOut[i] / 255.f, b = cudaOut[i] / 255.f;
            maxAbs = std::max(maxAbs, std::abs(a-b));
        }
        std::cout << "[SELFTEST] max |CPU-CUDA| = " << maxAbs
                  << " (target ≤ ~5e-4)\n";
        return (maxAbs <= 5e-4f) ? 0 : 2;
    }

    int width=0, height=0, nc=0;
    unsigned char* img = stbi_load(args.in.c_str(), &width, &height, &nc, 3);
    if (!img) { std::cerr << "Failed to load image: " << args.in << "\n"; return 1; }

    const int dstW = width  * opt.scale;
    const int dstH = height * opt.scale;
    std::vector<uint8_t> out((size_t)dstW * dstH * 3);

    // --- CPU backend ---
    if (args.device == Device::CPU) {
        if (args.benchIters > 0) {
            using clk = std::chrono::high_resolution_clock;
            using msf = std::chrono::duration<double, std::milli>;
            double accMs = 0.0;
            for (int it = 0; it < args.benchIters; ++it) {
                auto t0 = clk::now();
                upscaler::upscaleCPU(img, width, height, out.data(), dstW, dstH, opt);
                auto t1 = clk::now();
                accMs += std::chrono::duration_cast<msf>(t1 - t0).count();
            }
            std::cout << "[CPU] avg: " << (accMs / args.benchIters)
                      << " ms over " << args.benchIters << " iters\n";
        }

        // final launch
        upscaler::upscaleCPU(img, width, height, out.data(), dstW, dstH, opt);
        stbi_write_png(args.out.c_str(), dstW, dstH, 3, out.data(), dstW*3);
        stbi_image_free(img);
        std::cout << "Saved: " << args.out << " (" << dstW << "x" << dstH << ") [CPU]\n";
        return 0;
    }

    // --- CUDA backend ---
    unsigned char *dSrc=nullptr, *dDst=nullptr;
    size_t srcPitchBytes=0, dstPitchBytes=0;
    check(cudaMallocPitch(&dSrc, &srcPitchBytes, (size_t)width*3,  height), "cudaMallocPitch dSrc");
    check(cudaMallocPitch(&dDst, &dstPitchBytes, (size_t)dstW*3,   dstH),   "cudaMallocPitch dDst");

    for (int y = 0; y < height; ++y) {
        check(cudaMemcpy(dSrc + (size_t)y*srcPitchBytes,
                         img  + (size_t)y*width*3,
                         (size_t)width*3, cudaMemcpyHostToDevice), "H2D");
    }

    if (args.benchIters > 0) {
        cudaDeviceSynchronize();
        float accMs = 0.0f;
        for (int it = 0; it < args.benchIters; ++it) {
            cudaEvent_t a{}, b{}; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            upscaler::upscaleCUDA(dSrc, width, height, (int)srcPitchBytes,
                                  dDst, dstW,   dstH,   (int)dstPitchBytes,
                                  opt);
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (bench)");
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
            accMs += ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        std::cout << "[CUDA] kernel avg: " << (accMs / args.benchIters)
                  << " ms over " << args.benchIters << " iters\n";
    }

    upscaler::upscaleCUDA(dSrc, width, height, (int)srcPitchBytes,
                          dDst, dstW,   dstH,   (int)dstPitchBytes,
                          opt);

    for (int y = 0; y < dstH; ++y) {
        check(cudaMemcpy(out.data() + (size_t)y*dstW*3,
                         dDst + (size_t)y*dstPitchBytes,
                         (size_t)dstW*3, cudaMemcpyDeviceToHost), "D2H");
    }

    stbi_write_png(args.out.c_str(), dstW, dstH, 3, out.data(), dstW*3);

    cudaFree(dSrc); cudaFree(dDst);
    stbi_image_free(img);

    std::cout << "Saved: " << args.out << " (" << dstW << "x" << dstH << ") [CUDA]\n";
    return 0;
}

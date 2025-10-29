#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#include "upscale_kernels.cuh"
#include "upscale_cpu.hpp"

enum class Device { CUDA, CPU };
enum class Mode   { Bilinear, Bicubic };

struct Args {
    std::string in, out;                // input    | output
    int scale = 2;                      // 2        | 4
    Mode mode = Mode::Bilinear;         // bilinear | bicubic
    Device device = Device::CUDA;       // cpu      | cuda
    int benchIters = 0;                 // 0 = off  |  >0 = iterations
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: video_upscaler <in.png> <out.png> "
            "[--mode bilinear|bicubic] [--scale 2|4] "
            "[--device cuda|cpu] [--bench N]");
    }
    a.in  = argv[1];
    a.out = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--mode" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "bilinear") a.mode = Mode::Bilinear;
            else if (v == "bicubic")  a.mode = Mode::Bicubic;
            else throw std::runtime_error("Unknown --mode");
        } else if (s == "--scale" && i + 1 < argc) {
            a.scale = std::stoi(argv[++i]);
            if (a.scale != 2 && a.scale != 4)
                throw std::runtime_error("Only scale=2 or 4 supported");
        } else if (s == "--device" && i + 1 < argc) {
            std::string v = argv[++i];
            if(v == "cuda") a.device = Device::CUDA;
            else if (v == "cpu")  a.device = Device::CPU;
            else throw std::runtime_error("Unknown --device (use cuda|cpu)");
        } else if (s == "--bench" && i + 1 < argc) {
            a.benchIters = std::max(0, std::stoi(argv[++i]));
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

    int width = 0, height = 0, nc = 0;
    unsigned char* img = stbi_load(args.in.c_str(), &width, &height, &nc, 3);
    if (!img) { std::cerr << "Failed to load image: " << args.in << "\n"; return 1; }

    // -------- CPU BACKEND --------
    if (args.device == Device::CPU) {
        const int dstW = width * args.scale;
        const int dstH = height * args.scale;

        std::vector<unsigned char> out((size_t)dstW * dstH * 3);

        // Bench (optional)
        if (args.benchIters > 0) {
            using clk = std::chrono::high_resolution_clock;
            using msf = std::chrono::duration<double, std::milli>;
            double accMs = 0.0;
            for (int it = 0; it < args.benchIters; ++it) {
                auto t0 = clk::now();
                if (args.scale == 2) {
                    if (args.mode == Mode::Bilinear)
                        upscaler::bilinear2xRGB_cpu(img, width, height, out.data(), dstW, dstH);
                    else
                        upscaler::bicubic2xRGB_cpu (img, width, height, out.data(), dstW, dstH);
                } else { // scale == 4 (2x + 2x)
                    std::vector<unsigned char> mid((size_t)(width*2) * (height*2) * 3);
                    if (args.mode == Mode::Bilinear) {
                        upscaler::bilinear2xRGB_cpu(img, width, height, mid.data(), width*2, height*2);
                        upscaler::bilinear2xRGB_cpu(mid.data(), width*2, height*2, out.data(), dstW, dstH);
                    } else {
                        upscaler::bicubic2xRGB_cpu (img, width, height, mid.data(), width*2, height*2);
                        upscaler::bicubic2xRGB_cpu (mid.data(), width*2, height*2, out.data(), dstW, dstH);
                    }
                }
                auto t1 = clk::now();
                accMs += std::chrono::duration_cast<msf>(t1 - t0).count();
            }
            std::cout << "[CPU] avg: " << (accMs / args.benchIters) << " ms over " << args.benchIters << " iters\n";
        }

        // One final run for saving
        if (args.scale == 2) {
            if (args.mode == Mode::Bilinear)
                upscaler::bilinear2xRGB_cpu(img, width, height, out.data(), dstW, dstH);
            else
                upscaler::bicubic2xRGB_cpu (img, width, height, out.data(), dstW, dstH);
        } else { // 4x via cascade
            std::vector<unsigned char> mid((size_t)(width*2) * (height*2) * 3);
            if (args.mode == Mode::Bilinear) {
                upscaler::bilinear2xRGB_cpu(img, width, height, mid.data(), width*2, height*2);
                upscaler::bilinear2xRGB_cpu(mid.data(), width*2, height*2, out.data(), dstW, dstH);
            } else {
                upscaler::bicubic2xRGB_cpu (img, width, height, mid.data(), width*2, height*2);
                upscaler::bicubic2xRGB_cpu (mid.data(), width*2, height*2, out.data(), dstW, dstH);
            }
        }

        stbi_write_png(args.out.c_str(), dstW, dstH, 3, out.data(), dstW*3);
        stbi_image_free(img);
        std::cout << "Saved: " << args.out << " (" << dstW << "x" << dstH << ") [CPU]\n";
        return 0;
    }

    // -------- CUDA BACKEND --------
    const int dstW = width  * args.scale;
    const int dstH = height * args.scale;

    unsigned char *dSrc = nullptr, *dDst = nullptr, *dMid = nullptr;
    size_t srcPitchBytes = 0, dstPitchBytes = 0, midPitchBytes = 0;

    check(cudaMallocPitch(&dSrc, &srcPitchBytes, (size_t)width*3,  height),   "cudaMallocPitch dSrc");
    check(cudaMallocPitch(&dDst, &dstPitchBytes, (size_t)dstW*3,   dstH),     "cudaMallocPitch dDst");
    if (args.scale == 4) {
        check(cudaMallocPitch(&dMid, &midPitchBytes, (size_t)(width*2)*3, (height*2)), "cudaMallocPitch dMid");
    }

    // H2D
    for (int y = 0; y < height; ++y) {
        check(cudaMemcpy(dSrc + y*srcPitchBytes, img + (size_t)y*width*3, (size_t)width*3, cudaMemcpyHostToDevice), "H2D");
    }

    // Bench (optional)
    if (args.benchIters > 0) {
        // warm-up 
        cudaDeviceSynchronize();

        float accMs = 0.0f;
        for (int it = 0; it < args.benchIters; ++it) {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);

            if (args.scale == 2) {
                if (args.mode == Mode::Bilinear)
                    upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes,
                                            dDst,  dstW,  dstH,  (int)dstPitchBytes);
                else
                    upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes,
                                            dDst,  dstW,  dstH,  (int)dstPitchBytes);
            } else { // 4x = 2 times 2x
                if (args.mode == Mode::Bilinear) {
                    upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes,
                                            dMid, width*2, height*2, (int)midPitchBytes);
                    upscaler::bilinear2xRGB(dMid, width*2, height*2, (int)midPitchBytes,
                                            dDst, dstW, dstH, (int)dstPitchBytes);
                } else {
                    upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes,
                                            dMid, width*2, height*2, (int)midPitchBytes);
                    upscaler::bicubic2xRGB (dMid, width*2, height*2, (int)midPitchBytes,
                                            dDst, dstW, dstH, (int)dstPitchBytes);
                }
            }

            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (bench)");
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            accMs += ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        std::cout << "[CUDA] kernel avg: " << (accMs / args.benchIters) << " ms over " << args.benchIters << " iters\n";
    }

    // real launch
    if (args.scale == 2) {
        if (args.mode == Mode::Bilinear)
            upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes,
                                    dDst,  dstW,  dstH,  (int)dstPitchBytes);
        else
            upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes,
                                    dDst,  dstW,  dstH,  (int)dstPitchBytes);
    } else {
        if (args.mode == Mode::Bilinear) {
            upscaler::bilinear2xRGB(dSrc, width, height, (int)srcPitchBytes,
                                    dMid, width*2, height*2, (int)midPitchBytes);
            upscaler::bilinear2xRGB(dMid, width*2, height*2, (int)midPitchBytes,
                                    dDst, dstW, dstH, (int)dstPitchBytes);
        } else {
            upscaler::bicubic2xRGB (dSrc, width, height, (int)srcPitchBytes,
                                    dMid, width*2, height*2, (int)midPitchBytes);
            upscaler::bicubic2xRGB (dMid, width*2, height*2, (int)midPitchBytes,
                                    dDst, dstW, dstH, (int)dstPitchBytes);
        }
    }

    // D2H
    std::vector<unsigned char> out((size_t)dstW * dstH * 3);
    for (int y = 0; y < dstH; ++y) {
        check(cudaMemcpy(out.data() + (size_t)y*dstW*3, dDst + (size_t)y*dstPitchBytes,
                         (size_t)dstW*3, cudaMemcpyDeviceToHost), "D2H");
    }

    stbi_write_png(args.out.c_str(), dstW, dstH, 3, out.data(), dstW*3);

    // cleanup
    if (dMid) cudaFree(dMid);
    cudaFree(dSrc);
    cudaFree(dDst);
    stbi_image_free(img);

    std::cout << "Saved: " << args.out << " (" << dstW << "x" << dstH << ") [CUDA]\n";
    return 0;
}

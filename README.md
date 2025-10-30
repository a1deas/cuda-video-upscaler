# CUDA Video Upscaler
High-performance **image and video upscaler** built with **CUDA**.

This project demonstrates image and video upscaling with bilinear and bicubic interpolation.
Demonstrates CUDA memory handling (pitched allocations, kernel launches) and CPU-GPU comparison.

# Devices 
- CPU 
- GPU(CUDA of course)

## Features
- 2x/4x scaling with high-quality interpolation
- Bilinear or Bicubic filters
- Gamma-correct interpolation(sRGB <-> Linear)
- Benchmarks: `--bench N` (ms/iter)
- Optional OpenMP
- Clamp and Reflect borders
- `ffmpeg` pite via PPM `--stdin/--stdout`

## Building:
Requirements:
- **CUDA 13.0** (or newer)
- **CMake 3.22+**
- **Visual Studio 2022**
- [**vcpkg**](https://github.com/microsoft/vcpkg) *(for stb_image dependency)*

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Commands Examples:
```bash
# CPU 
.\build\bin\Release\video_upscaler.exe imageIn.png imageOut.png --device cpu --mode bilinear --scale 2 --border clamp --gamma-correct --bench 5

# CUDA
.\build\bin\Release\video_upscaler.exe imageIn.png imageOut.png --device cuda --mode bicubic --scale 4 --border reflect --gamma-correct --bench 20
```

## CLI Parametres 
```text
--mode              bilinear/bicubic
--scale             2/4
--device            cpu/cuda
--border            clamp/reflect
--gamma-correct     true/false
--bench <N>         benchmark for N iterations
--selftest          Run automatic CPU/GPU parity test
--stdin             Read via PPM stdin(P6)
--stdout            Write via PPM stdout(P6)
```

## Chess Pattern Original Image - 600x600

![screenshot](demo/chess.png)

## Chess Pattern Bilinear 2x Upscale - 1200x1200

![screenshot](demo/bilinear_2x_chess.png)

## Chess Pattern Bicubic 4x Upscale - 2400x2400

![screenshot](demo/bicubic_4x_chess.png)

## Next Steps
- FFmpeg stdin/stdout
- Video upscaling
- NVDEC/NVENC
- Texture Objects / CUDA surface reads
- Bench Tables(720p/1080p/1440p, 2x/4x, ms/frame and MPix/s)
- Autotest Package, PSNR/SSIM report

## Tech: 
C++17, CUDA 13.0, CMake, stb_image, OpenMP


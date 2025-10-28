cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(cuda_video_upscaler LANGUAGES CXX CUDA)

# --- Standards ---
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Hybrid) 

# --- Arch ---
set(CMAKE_CUDA_ARCHITECTURES native)

# IntelliSense
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Output dirs ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Sources ---
file(GLOB SRC
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
    ${CMAKE_SOURCE_DIR}/src/*.h
)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    "C:/tools/vcpkg/installed/x64-windows/include"
)

# --- Target ---
add_executable(video_upscaler ${SRC})

# --- CUDA properties ---
set_target_properties(video_upscaler PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Link CUDA runtime ---
find_package(CUDAToolkit REQUIRED)
target_link_libraries(video_upscaler PRIVATE CUDA::cudart)

if (TARGET CUDA::cudadevrt)
  target_link_libraries(video_upscaler PRIVATE CUDA::cudadevrt)
endif()

# --- Flags ---
if (MSVC)
  target_compile_options(video_upscaler PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
else()
  target_compile_options(video_upscaler PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
  )
endif()

# --- Windows specifics ---
if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

# --- Print summary ---
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

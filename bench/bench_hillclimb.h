// Hill-Climb Benchmark & Parity Harness for brokernel:
// - Tier 1: FastNoise2 Node Suite across 2D, 3D, 4D + f64 planetary origin (43 configs)
//   * Parity oracle: FastNoise2 C++ SIMD
//   * CPU speed baseline: FastNoise2 C++ SIMD (best of kCpuReps runs, single thread)
//   * GPU speed baseline: the same node as a GLSL 450 compute shader compiled by glslc (-O),
//     every f32 node (kGlslNoiseHeader); its output is checked against FastNoise2 before
//     it counts. The f64 rows are CPU only (@cpu): no GPU column. The cheap nodes are memory-bound at
//     ~130,000 Mpt/s on a 4090 either way; the fractal, cellular and warp rows are the
//     ones whose ratio measures the code.
// - Tier 2: Stencil & Grid Kernels (Separable Blur, Gradient/Normals, Thermal Erosion,
//           Hydraulic Erosion, Flow Accumulation, and the planet's rivers: PlanetFlow,
//           its erosion pass, five fields read at eight neighbours). On the GPU the
//           planet's flow is memory-bound per iteration (~36 bytes a cell).
//   * Parity oracle & CPU speed baseline: Plain C++ reference
//   * GPU speed baseline: Hand-written GLSL 450 compute shader compiled by glslc (-O)
// - Capstone: the planet's noise fields (broworkshop 1fbd228^:demos/planet/terrain.js),
//   eight FractalFBm-of-Simplex fields summed, in f64 about a double origin on the sphere
//   (on the CPU: f64 has no GPU build).
//   * Parity in metres against FastNoise2's tree of the same nodes; CPU speed against it.
// The GPU is timed with timestamp queries on 4M points (a 2048 x 2048 grid).
// --write-golden writes bench/goldens.windows.json; --check-golden fails on any
// parity failure or a geomean more than 10% under the golden's.

#pragma once

#include "brokernel/brokernel.h"
#include <FastNoise/FastNoise.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace brokernel;

// Parity runs on kParityPoints (and stencils on a 256 x 256 grid); the GPU is
// timed on kGpuTimePoints (a 2048 x 2048 grid), large enough to fill it: at
// 65536 points every kernel took the same ~2 us of dispatch and nothing else.
inline constexpr uint32_t kParityPoints = 65536;
inline constexpr uint32_t kGpuTimeSide = 2048;
inline constexpr uint32_t kGpuTimePoints = kGpuTimeSide * kGpuTimeSide;
inline constexpr uint32_t kGpuRepeat = 20;
inline constexpr int kCpuReps = 3;

// Parity against FastNoise2, absolute. Both builds are fast math (fused
// multiply-adds, at different sites; FastLengthSqrt is an approximate
// inverse square root): the lattice noises differ by up to ~9e-5 where a
// corner's falloff is steep, the cellular distances and the warps, which
// move positions by them, by up to ~1e-4.
inline constexpr float kTol = 1.5e-4f;
inline constexpr float kTolRelaxed = 1e-4f;

uint32_t ulp_distance(float a, float b);

struct ErrorStats {
    float max_abs = 0.0f;
    uint32_t max_ulp = 0;
    bool pass = false;
};

ErrorStats compare_buffers(const std::vector<float>& got, const std::vector<float>& ref, float tol);

// The fastest of kCpuReps runs of f, in ms (the first warms caches and code).
template <class F>
double best_cpu_ms(F&& f) {
    double best = 1e30;
    for (int r = 0; r < kCpuReps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return std::max(best, 1e-4);
}

inline double mpts(size_t points, double ms) { return (points / 1e6) / (ms / 1000.0); }

// Compile GLSL 450 compute shader (using buffer_reference physical storage buffers) to SPIR-V via glslc
std::vector<uint32_t> compile_glsl_compute(const std::string& glsl_src, const std::string& tag);

// Wrap a glslc-compiled SPIR-V binary into a brass::target::SpirvKernel with the given push-constant layout
brass::gpu::VulkanModule load_glsl_vulkan_module(
    const std::vector<uint32_t>& words,
    const std::vector<brass::Type>& param_types,
    uint32_t lx = 64, uint32_t ly = 1, uint32_t lz = 1);

// Time a compiled brokernel on GPU using Vulkan timestamp queries (with pre-uploaded device buffers)
// A device dispatch argument: a buffer, or a scalar. A brokernel kernel takes
// a store as its address and element count; a GLSL baseline takes the
// address alone (glsl_args).
struct GpuArg {
    GpuArg(const brass::gpu::VulkanBuffer& b) : buf(&b), arg(b) {}  // NOLINT
    GpuArg(int32_t v) : arg(v) {}                                    // NOLINT
    GpuArg(float v) : arg(v) {}                                      // NOLINT
    GpuArg(double v) : arg(v) {}                                     // NOLINT
    const brass::gpu::VulkanBuffer* buf = nullptr;
    brass::gpu::VulkanArg arg;
};

std::vector<brass::gpu::VulkanArg> glsl_args(const std::vector<GpuArg>& args);

double time_bk_gpu_ms(
    const CompiledKernel* k,
    GridShape shape,
    const std::vector<GpuArg>& args_without_grid_and_out,
    uint32_t repeat = kGpuRepeat);

double time_glsl_gpu_ms(
    const brass::gpu::VulkanModule& vk_mod,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t lx, uint32_t ly, uint32_t lz,
    const std::vector<brass::gpu::VulkanArg>& vk_args,
    uint32_t repeat = kGpuRepeat);

struct BenchRow {
    std::string tier;
    std::string name;
    int dim = 2;
    size_t points = 0;
    bool cpu_pass = false;
    bool gpu_pass = false;
    float cpu_max_abs = 0.0f;
    uint32_t cpu_max_ulp = 0;
    float gpu_max_abs = 0.0f;
    uint32_t gpu_max_ulp = 0;
    double cpu_bk_mpts = 0.0;
    double cpu_ref_mpts = 0.0;
    double cpu_ratio = 0.0;
    double gpu_bk_mpts = 0.0;
    double gpu_glsl_mpts = 0.0;
    double gpu_ratio = 0.0;
    bool has_gpu_baseline = false;   // a GLSL kernel of the same node, checked against the oracle
};

extern const char* kGlslNoiseHeader;
extern const char* kBkNoiseSuiteSource;

std::string make_glsl_noise_shader(int dim, const std::string& expr);
std::vector<BenchRow> run_stencil_tier();

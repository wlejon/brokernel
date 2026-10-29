// Hill-Climb Benchmark & Parity Harness for brokernel:
// - Tier 1: FastNoise2 Node Suite across 2D, 3D, 4D + f64 planetary origin (43 configs)
//   * Parity oracle: FastNoise2 C++ SIMD
//   * CPU speed baseline: FastNoise2 C++ SIMD (best of kCpuReps runs, single thread)
//   * GPU speed baseline: the same node as a GLSL 450 compute shader compiled by glslc (-O),
//     for Value, Perlin, FractalFBm and FractalRidged; its output is checked against
//     FastNoise2 before it counts. Other nodes have no GPU ratio.
// - Tier 2: Stencil & Grid Kernels (Separable Blur, Gradient/Normals, Thermal Erosion,
//           Hydraulic Erosion, Flow Accumulation)
//   * Parity oracle & CPU speed baseline: Plain C++ reference
//   * GPU speed baseline: Hand-written GLSL 450 compute shader compiled by glslc (-O)
// - Capstone: the planet's noise fields (broworkshop 1fbd228^:demos/planet/terrain.js),
//   eight FractalFBm-of-Simplex fields summed, in f64 about a double origin on the sphere.
//   * Parity in metres against FastNoise2's tree of the same nodes; CPU speed against it.
// The GPU is timed with timestamp queries on 4M points (a 2048 x 2048 grid).
// --write-golden writes bench/goldens.windows.json; --check-golden fails on any
// parity failure or a geomean more than 10% under the golden's.

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

namespace {

uint32_t ulp_distance(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return UINT32_MAX;
    int32_t ia = 0, ib = 0;
    std::memcpy(&ia, &a, sizeof(float));
    std::memcpy(&ib, &b, sizeof(float));
    if (ia < 0) ia = int32_t(0x80000000u) - ia;
    if (ib < 0) ib = int32_t(0x80000000u) - ib;
    return static_cast<uint32_t>(std::abs(int64_t(ia) - int64_t(ib)));
}

struct ErrorStats {
    float max_abs = 0.0f;
    uint32_t max_ulp = 0;
    bool pass = false;
};

ErrorStats compare_buffers(const std::vector<float>& got, const std::vector<float>& ref, float tol) {
    ErrorStats s{};
    if (got.size() != ref.size()) return s;
    s.pass = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) {
            s.pass = false;
            s.max_abs = INFINITY;
            s.max_ulp = UINT32_MAX;
            break;
        }
        float err = std::abs(got[i] - ref[i]);
        if (err > s.max_abs) s.max_abs = err;
        if (err > 1e-6f) {
            s.max_ulp = std::max(s.max_ulp, ulp_distance(got[i], ref[i]));
        }
        if (err > tol) s.pass = false;
    }
    return s;
}

// Parity runs on kParityPoints (and stencils on a 256 x 256 grid); the GPU is
// timed on kGpuTimePoints (a 2048 x 2048 grid), large enough to fill it: at
// 65536 points every kernel took the same ~2 us of dispatch and nothing else.
constexpr uint32_t kParityPoints = 65536;
constexpr uint32_t kGpuTimeSide = 2048;
constexpr uint32_t kGpuTimePoints = kGpuTimeSide * kGpuTimeSide;
constexpr uint32_t kGpuRepeat = 20;
constexpr int kCpuReps = 3;

// Parity against FastNoise2, absolute. Its build is relaxed (fused
// multiply-adds; FastLengthSqrt is an approximate inverse square root), so
// the cellular distances and the warps, which move positions by them, get
// a looser bound than the lattice noises (whose errors are ~1e-6).
constexpr float kTol = 1e-5f;
constexpr float kTolRelaxed = 1e-4f;

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

double mpts(size_t points, double ms) { return (points / 1e6) / (ms / 1000.0); }

// Compile GLSL 450 compute shader (using buffer_reference physical storage buffers) to SPIR-V via glslc
std::vector<uint32_t> compile_glsl_compute(const std::string& glsl_src, const std::string& tag) {
    fs::path tmp_dir = fs::temp_directory_path() / "brokernel_hillclimb_spv";
    fs::create_directories(tmp_dir);
    fs::path comp_path = tmp_dir / (tag + ".comp");
    fs::path spv_path = tmp_dir / (tag + ".spv");

    // Cache compiled SPIR-V if source matches
    bool need_compile = !fs::exists(spv_path);
    if (!need_compile && fs::exists(comp_path)) {
        std::ifstream existing(comp_path, std::ios::binary);
        std::ostringstream ss;
        ss << existing.rdbuf();
        if (ss.str() != glsl_src) need_compile = true;
    } else {
        need_compile = true;
    }

    if (need_compile) {
        {
            std::ofstream ofs(comp_path, std::ios::binary);
            ofs << glsl_src;
        }
        std::string glslc = "glslc";
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        if (const char* sdk = std::getenv("VULKAN_SDK")) {
            fs::path p = fs::path(sdk) / "Bin" / "glslc";
            if (fs::exists(p) || fs::exists(p.string() + ".exe")) glslc = p.string();
        }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        std::string cmd = "\"\"" + glslc + "\" --target-env=vulkan1.3 -O -fshader-stage=compute \"" +
                          comp_path.string() + "\" -o \"" + spv_path.string() + "\"\"";
        int rc = std::system(cmd.c_str());
        if (rc != 0 || !fs::exists(spv_path)) {
            std::cerr << "glslc failed for " << tag << "\n";
            return {};
        }
    }

    std::ifstream ifs(spv_path, std::ios::binary | std::ios::ate);
    size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint32_t> words(sz / 4);
    ifs.read(reinterpret_cast<char*>(words.data()), sz);
    return words;
}

// Wrap a glslc-compiled SPIR-V binary into a brass::target::SpirvKernel with the given push-constant layout
brass::gpu::VulkanModule load_glsl_vulkan_module(
    const std::vector<uint32_t>& words,
    const std::vector<brass::Type>& param_types,
    uint32_t lx = 64, uint32_t ly = 1, uint32_t lz = 1) {
    brass::target::SpirvKernel sk;
    sk.entry = "main";
    sk.stage = brass::target::SpirvStage::Compute;
    sk.words = words;
    sk.local_size[0] = lx;
    sk.local_size[1] = ly;
    sk.local_size[2] = lz;
    sk.local_size_spec_constants = false;

    uint32_t offset = 0;
    for (const auto& t : param_types) {
        uint32_t sz = static_cast<uint32_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;
        offset = (offset + sz - 1) & ~(sz - 1);
        sk.params.push_back({t, offset, sz});
        offset += sz;
    }
    sk.push_constant_bytes = offset;

    std::string err;
    auto mod = brass::gpu::VulkanModule::load(sk, &err);
    if (!mod.valid()) {
        std::cerr << "VulkanModule::load failed for GLSL shader: " << err << "\n";
    }
    return mod;
}

// Time a compiled brokernel on GPU using Vulkan timestamp queries (with pre-uploaded device buffers)
double time_bk_gpu_ms(
    const CompiledKernel* k,
    GridShape shape,
    const std::vector<brass::gpu::VulkanArg>& vk_args_without_grid_and_out,
    uint32_t repeat = kGpuRepeat) {
    std::string err;
    auto vk_mod = brass::gpu::VulkanModule::load(k->spirv, &err);
    if (!vk_mod.valid()) {
        std::cerr << "Failed to load bk VulkanModule for " << k->name << ": " << err << "\n";
        return 1e9;
    }

    std::vector<brass::gpu::VulkanArg> full_args = vk_args_without_grid_and_out;
    uint32_t w = std::max(1u, shape.width);
    uint32_t h = std::max(1u, shape.height);
    uint32_t d = std::max(1u, shape.depth);
    if (k->grid_dim >= 1) full_args.emplace_back(static_cast<int32_t>(w));
    if (k->grid_dim >= 2) full_args.emplace_back(static_cast<int32_t>(h));
    if (k->grid_dim >= 3) full_args.emplace_back(static_cast<int32_t>(d));

    brass::gpu::VulkanBuffer ret_buf;
    if (k->has_return_value()) {
        ret_buf = brass::gpu::VulkanBuffer::alloc(shape.total_points() * std::max<size_t>(k->return_byte_size, 4));
        full_args.emplace_back(ret_buf);
    }

    uint32_t lx = std::max(1u, k->local_size[0]);
    uint32_t ly = std::max(1u, k->local_size[1]);
    uint32_t lz = std::max(1u, k->local_size[2]);

    brass::gpu::VulkanDispatch disp;
    disp.grid[0] = (k->grid_dim >= 1) ? (w + lx - 1) / lx : 1;
    disp.grid[1] = (k->grid_dim >= 2) ? (h + ly - 1) / ly : 1;
    disp.grid[2] = (k->grid_dim >= 3) ? (d + lz - 1) / lz : 1;
    disp.block[0] = lx;
    disp.block[1] = ly;
    disp.block[2] = lz;

    // Warmup
    disp.repeat = 1;
    double warm_ms = 0.0;
    disp.gpu_ms = &warm_ms;
    vk_mod.launch(disp, full_args, &err);

    // Timed repeat
    double total_ms = 0.0;
    disp.repeat = repeat;
    disp.gpu_ms = &total_ms;
    vk_mod.launch(disp, full_args, &err);
    return (total_ms > 0.0) ? (total_ms / repeat) : 1e-4;
}

double time_glsl_gpu_ms(
    const brass::gpu::VulkanModule& vk_mod,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t lx, uint32_t ly, uint32_t lz,
    const std::vector<brass::gpu::VulkanArg>& vk_args,
    uint32_t repeat = kGpuRepeat) {
    std::string err;
    brass::gpu::VulkanDispatch disp;
    disp.grid[0] = gx;
    disp.grid[1] = gy;
    disp.grid[2] = gz;
    disp.block[0] = lx;
    disp.block[1] = ly;
    disp.block[2] = lz;

    disp.repeat = 1;
    double warm_ms = 0.0;
    disp.gpu_ms = &warm_ms;
    vk_mod.launch(disp, vk_args, &err);

    double total_ms = 0.0;
    disp.repeat = repeat;
    disp.gpu_ms = &total_ms;
    vk_mod.launch(disp, vk_args, &err);
    return (total_ms > 0.0) ? (total_ms / repeat) : 1e-4;
}

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

// Shared GLSL header with FastNoise2 hashes, gradients, Value, Perlin, Simplex, Fractal, Cellular & Warp
const char* kGlslNoiseHeader = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf {
    float v[];
};

const int PRIME_X = -141048377;
const int PRIME_Y = 1812335753;
const int PRIME_Z = 1180684367;
const int PRIME_W = -140107143;
const int HASH_A  = -1210014219;

int hash_primes2d(int seed, int x, int y) {
    int h = (seed ^ x ^ y) * HASH_A;
    return (h >> 15) ^ h;
}
int hash_primes3d(int seed, int x, int y, int z) {
    int h = (seed ^ x ^ y ^ z) * HASH_A;
    return (h >> 15) ^ h;
}
int hash_primes4d(int seed, int x, int y, int z, int w) {
    int h = (seed ^ x ^ y ^ z ^ w) * HASH_A;
    return (h >> 15) ^ h;
}

float get_value2d(int seed, int x, int y) {
    int h = seed ^ x ^ y;
    return float(h * (h * HASH_A));
}
float get_value3d(int seed, int x, int y, int z) {
    int h = seed ^ x ^ y ^ z;
    return float(h * (h * HASH_A));
}
float get_value4d(int seed, int x, int y, int z, int w) {
    int h = seed ^ x ^ y ^ z ^ w;
    return float(h * (h * HASH_A));
}

float interp_hermite(float t) { return t * t * (3.0 - 2.0 * t); }
float interp_quintic(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
float scale_output(float val, float min_b, float max_b) {
    return (1.0 / (max_b - min_b)) * 2.0 * (val - min_b) - 1.0;
}
float xor_sign(float val, int sign_bit) {
    return intBitsToFloat(floatBitsToInt(val) ^ (sign_bit & 0x80000000));
}
float grad_dot_perlin2d(int hash, float fx, float fy) {
    float sx = xor_sign(fx, hash << 31);
    float sy = xor_sign(fy, (hash >> 1) << 31);
    bool sel = (hash & 4) != 0;
    float u = sel ? sy : sx;
    float v = sel ? sx : sy;
    return 2.414213657379150390625 * u + v;
}
float grad_dot_common3d(int hash, float fx, float fy, float fz) {
    int h13 = hash & 13;
    float u = (h13 > 7) ? fy : fx;
    float v = (h13 == 12) ? fx : fz;
    v = (h13 < 2) ? fy : v;
    return xor_sign(u, hash << 31) + xor_sign(v, (hash >> 1) << 31);
}
float grad_dot_perlin4d(int hash, float fx, float fy, float fz, float fw) {
    int p = hash & 24;
    float a = (p > 0) ? fx : fy;
    float b = ((hash & 16) != 0) ? fy : fz;
    float c = (p > 16) ? fz : fw;
    return xor_sign(a, hash << 31) + xor_sign(b, (hash >> 1) << 31) + xor_sign(c, (hash >> 2) << 31);
}

float fn_value2d(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    vec2 f = vec2(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y));
    float v0 = mix(get_value2d(seed, i0.x, i0.y), get_value2d(seed, i1.x, i0.y), f.x);
    float v1 = mix(get_value2d(seed, i0.x, i1.y), get_value2d(seed, i1.x, i1.y), f.x);
    return scale_output(mix(v0, v1, f.y), -2147483648.0, 2147483648.0);
}
float fn_value3d(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f = vec3(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z));
    float vy0 = mix(mix(get_value3d(seed, i0.x, i0.y, i0.z), get_value3d(seed, i1.x, i0.y, i0.z), f.x),
                    mix(get_value3d(seed, i0.x, i1.y, i0.z), get_value3d(seed, i1.x, i1.y, i0.z), f.x), f.y);
    float vy1 = mix(mix(get_value3d(seed, i0.x, i0.y, i1.z), get_value3d(seed, i1.x, i0.y, i1.z), f.x),
                    mix(get_value3d(seed, i0.x, i1.y, i1.z), get_value3d(seed, i1.x, i1.y, i1.z), f.x), f.y);
    return scale_output(mix(vy0, vy1, f.z), -2147483648.0, 2147483648.0);
}
float fn_value4d(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    ivec4 i1 = i0 + ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f = vec4(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z), interp_hermite(p.w - s.w));
    float vz00 = mix(mix(get_value4d(seed, i0.x, i0.y, i0.z, i0.w), get_value4d(seed, i1.x, i0.y, i0.z, i0.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i0.z, i0.w), get_value4d(seed, i1.x, i1.y, i0.z, i0.w), f.x), f.y);
    float vz10 = mix(mix(get_value4d(seed, i0.x, i0.y, i1.z, i0.w), get_value4d(seed, i1.x, i0.y, i1.z, i0.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i1.z, i0.w), get_value4d(seed, i1.x, i1.y, i1.z, i0.w), f.x), f.y);
    float vw0 = mix(vz00, vz10, f.z);
    float vz01 = mix(mix(get_value4d(seed, i0.x, i0.y, i0.z, i1.w), get_value4d(seed, i1.x, i0.y, i0.z, i1.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i0.z, i1.w), get_value4d(seed, i1.x, i1.y, i0.z, i1.w), f.x), f.y);
    float vz11 = mix(mix(get_value4d(seed, i0.x, i0.y, i1.z, i1.w), get_value4d(seed, i1.x, i0.y, i1.z, i1.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i1.z, i1.w), get_value4d(seed, i1.x, i1.y, i1.z, i1.w), f.x), f.y);
    float vw1 = mix(vz01, vz11, f.z);
    return scale_output(mix(vw0, vw1, f.w), -2147483648.0, 2147483648.0);
}

float fn_perlin2d(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    vec2 f0 = p - s;
    vec2 f1 = f0 - 1.0;
    vec2 u = vec2(interp_quintic(f0.x), interp_quintic(f0.y));
    float g0 = mix(grad_dot_perlin2d(hash_primes2d(seed, i0.x, i0.y), f0.x, f0.y),
                   grad_dot_perlin2d(hash_primes2d(seed, i1.x, i0.y), f1.x, f0.y), u.x);
    float g1 = mix(grad_dot_perlin2d(hash_primes2d(seed, i0.x, i1.y), f0.x, f1.y),
                   grad_dot_perlin2d(hash_primes2d(seed, i1.x, i1.y), f1.x, f1.y), u.x);
    return scale_output(mix(g0, g1, u.y), -1.726796627044677734375, 1.726796627044677734375);
}
float fn_perlin3d(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f0 = p - s;
    vec3 f1 = f0 - 1.0;
    vec3 u = vec3(interp_quintic(f0.x), interp_quintic(f0.y), interp_quintic(f0.z));
    float gy0 = mix(mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i0.y, i0.z), f0.x, f0.y, f0.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i0.y, i0.z), f1.x, f0.y, f0.z), u.x),
                    mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i1.y, i0.z), f0.x, f1.y, f0.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i1.y, i0.z), f1.x, f1.y, f0.z), u.x), u.y);
    float gy1 = mix(mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i0.y, i1.z), f0.x, f0.y, f1.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i0.y, i1.z), f1.x, f0.y, f1.z), u.x),
                    mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i1.y, i1.z), f0.x, f1.y, f1.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i1.y, i1.z), f1.x, f1.y, f1.z), u.x), u.y);
    return scale_output(mix(gy0, gy1, u.z), -1.0363423824310302734375, 1.0363423824310302734375);
}
float fn_perlin4d(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    ivec4 i1 = i0 + ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f0 = p - s;
    vec4 f1 = f0 - 1.0;
    vec4 u = vec4(interp_quintic(f0.x), interp_quintic(f0.y), interp_quintic(f0.z), interp_quintic(f0.w));
    float gz00 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i0.z, i0.w), f0.x, f0.y, f0.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i0.z, i0.w), f1.x, f0.y, f0.z, f0.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i0.z, i0.w), f0.x, f1.y, f0.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i0.z, i0.w), f1.x, f1.y, f0.z, f0.w), u.x), u.y);
    float gz10 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i1.z, i0.w), f0.x, f0.y, f1.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i1.z, i0.w), f1.x, f0.y, f1.z, f0.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i1.z, i0.w), f0.x, f1.y, f1.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i1.z, i0.w), f1.x, f1.y, f1.z, f0.w), u.x), u.y);
    float gw0 = mix(gz00, gz10, u.z);
    float gz01 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i0.z, i1.w), f0.x, f0.y, f0.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i0.z, i1.w), f1.x, f0.y, f0.z, f1.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i0.z, i1.w), f0.x, f1.y, f0.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i0.z, i1.w), f1.x, f1.y, f0.z, f1.w), u.x), u.y);
    float gz11 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i1.z, i1.w), f0.x, f0.y, f1.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i1.z, i1.w), f1.x, f0.y, f1.z, f1.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i1.z, i1.w), f0.x, f1.y, f1.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i1.z, i1.w), f1.x, f1.y, f1.z, f1.w), u.x), u.y);
    float gw1 = mix(gz01, gz11, u.z);
    return scale_output(mix(gw0, gw1, u.w), -1.33858621120452880859375, 1.33858621120452880859375);
}

float fn_perlin(vec2 p, int seed) { return fn_perlin2d(p, seed); }
float fn_perlin(vec3 p, int seed) { return fn_perlin3d(p, seed); }
float fn_perlin(vec4 p, int seed) { return fn_perlin4d(p, seed); }

// FastNoise2's FractalFBm and FractalRidged over Perlin (seed + 1 an octave,
// the weighted strength's lerp of the amplitude), one per dimension.
#define FN_FRACTALS(VT) \
float fn_fbm(VT p, int seed, int octaves, float lac, float gain, float ws) { \
    float amp = 1.0; \
    float noise = fn_perlin(p, seed); \
    float sum = noise * amp; \
    for (int i = 1; i < octaves; ++i) { \
        seed += 1; \
        amp *= 1.0 + ws * ((noise + 1.0) * 0.5 - 1.0); \
        amp *= gain; \
        p *= lac; \
        noise = fn_perlin(p, seed); \
        sum += noise * amp; \
    } \
    return sum; \
} \
float fn_ridged(VT p, int seed, int octaves, float lac, float gain, float ws) { \
    float amp = 1.0; \
    float noise = abs(fn_perlin(p, seed)); \
    float sum = (noise * -2.0 + 1.0) * amp; \
    for (int i = 1; i < octaves; ++i) { \
        seed += 1; \
        amp *= 1.0 + ws * ((1.0 - noise) - 1.0); \
        amp *= gain; \
        p *= lac; \
        noise = abs(fn_perlin(p, seed)); \
        sum += (noise * -2.0 + 1.0) * amp; \
    } \
    return sum; \
}
FN_FRACTALS(vec2)
FN_FRACTALS(vec3)
FN_FRACTALS(vec4)
)GLSL";

const char* kBkNoiseSuiteSource = R"BK(
    import std::noise::*;

    @kernel @grid(1)
    fn k_value2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_value3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_value4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_perlin2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_perlin3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_perlin4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_simplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_simplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_supersimplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_supersimplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_supersimplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_white2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_white3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_white4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_fbm2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_fbm3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_fbm4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
    }

    @kernel @grid(1)
    fn k_ridged2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_ridged3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_ridged4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
    }

    @kernel @grid(1)
    fn k_pingpong2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin2d<f32>(vec2<f32>(px[i], py[i]), seed), 2.0);
    }
    @kernel @grid(1)
    fn k_pingpong3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed), 2.0);
    }
    @kernel @grid(1)
    fn k_pingpong4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed), 2.0);
    }

    @kernel @grid(1)
    fn k_cell_val2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_val3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_val4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 1.0);
    }

    @kernel @grid(1)
    fn k_cell_dist2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_dist3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_dist4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 0, 1.0);
    }

    @kernel @grid(1)
    fn k_cell_lookup2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin2d<f32>(vec2<f32>(px[i], py[i]), seed, 1.0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_lookup3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 1.0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_lookup4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 1.0, 1.0);
    }

    @kernel @grid(1)
    fn k_warp_grad2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient2d<f32>(vec2<f32>(px[i], py[i]), seed, 1.5, 0.5);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_grad3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 1.5, 0.5);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_grad4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 1.5, 0.5);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @grid(1)
    fn k_warp_prog2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 2>(vec2<f32>(px[i], py[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_prog3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_prog4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @grid(1)
    fn k_warp_indep2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 2>(vec2<f32>(px[i], py[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_indep3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_indep4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @grid(1)
    fn k_planet_f64_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, origin: f64, seed: i32) {
        let i = g.x();
        let p = vec3<f64>((px[i] as f64) + origin, (py[i] as f64) + origin, (pz[i] as f64) + origin);
        out[i] = simplex3d<f64>(p, seed) as f32;
    }
)BK";

std::string make_glsl_noise_shader(int dim, const std::string& expr) {
    std::ostringstream ss;
    ss << kGlslNoiseHeader;
    if (dim == 2) {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec2 p = vec2(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    } else if (dim == 3) {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t pz; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec3 p = vec3(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i], FloatBuf(pc.pz).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    } else {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t pz; uint64_t pw; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec4 p = vec4(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i], FloatBuf(pc.pz).v[i], FloatBuf(pc.pw).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    }
    return ss.str();
}

// ============================================================================
// Tier 2: Stencil & Grid Kernels
// ============================================================================
std::vector<BenchRow> run_stencil_tier() {
    std::vector<BenchRow> rows;
    const uint32_t W = 256, H = 256;
    const size_t N = static_cast<size_t>(W) * H;
    const GridShape shape2d(W, H);

    std::vector<float> h_in(N);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float fx = static_cast<float>(x) * 0.05f;
            float fy = static_cast<float>(y) * 0.05f;
            h_in[y * W + x] = std::sin(fx) * std::cos(fy) * 10.0f + std::sin(fx * 2.3f + fy * 1.7f) * 3.0f;
        }
    }

    const char* bk_stencils_src = R"BK(
        @kernel @grid(2)
        fn k_blur_h(g: grid<2>, src: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            dst[x, y] = src[-2, 0] * 0.0625 + src[-1, 0] * 0.25 + src[0, 0] * 0.375 + src[1, 0] * 0.25 + src[2, 0] * 0.0625;
        }

        @kernel @grid(2)
        fn k_blur_v(g: grid<2>, src: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            dst[x, y] = src[0, -2] * 0.0625 + src[0, -1] * 0.25 + src[0, 0] * 0.375 + src[0, 1] * 0.25 + src[0, 2] * 0.0625;
        }

        @kernel @grid(2)
        fn k_grad_normals(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let dx = 0.5 * (h[1, 0] - h[-1, 0]);
            let dy = 0.5 * (h[0, 1] - h[0, -1]);
            dst[x, y] = 1.0 / sqrt(dx * dx + dy * dy + 1.0);
        }

        fn thermal_flux(d: f32, talus: f32) -> f32 {
            if d > talus { return d - talus; }
            if d < -talus { return d + talus; }
            return 0.0;
        }

        @kernel @grid(2)
        fn k_thermal_step(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let talus = 0.1;
            let f = thermal_flux(h[-1, 0] - hc, talus)
                  + thermal_flux(h[ 1, 0] - hc, talus)
                  + thermal_flux(h[0, -1] - hc, talus)
                  + thermal_flux(h[0,  1] - hc, talus);
            dst[x, y] = hc + 0.125 * f;
        }

        @kernel @grid(2)
        fn k_hydraulic_step(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let hL = h[-1, 0];
            let hR = h[ 1, 0];
            let hD = h[0, -1];
            let hU = h[0,  1];
            let dx = hR - hL;
            let dy = hU - hD;
            let slope = 0.5 * sqrt(dx * dx + dy * dy);
            let lap = 0.25 * (hL + hR + hD + hU) - hc;
            let capacity = slope * 0.15;
            dst[x, y] = hc - 0.05 * capacity + 0.1 * lap;
        }

        @kernel @grid(2)
        fn k_flow_accum(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let card = max(0.0, h[-1, 0] - hc)
                     + max(0.0, h[ 1, 0] - hc)
                     + max(0.0, h[0, -1] - hc)
                     + max(0.0, h[0,  1] - hc);
            let diag = max(0.0, h[-1, -1] - hc)
                     + max(0.0, h[ 1, -1] - hc)
                     + max(0.0, h[-1,  1] - hc)
                     + max(0.0, h[ 1,  1] - hc);
            dst[x, y] = 1.0 + card + 0.70710678 * diag;
        }
    )BK";

    auto res = compile_source(bk_stencils_src, "stencils_bench.bk");
    if (!res.ok) {
        std::cerr << "Stencil compile error:\n" << res.error_message() << "\n";
        return rows;
    }

    auto clamp_i = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    // The GPU is timed on a kGpuTimeSide grid of the same field.
    const uint32_t TW = kGpuTimeSide, TH = kGpuTimeSide;
    const size_t NT = static_cast<size_t>(TW) * TH;
    const GridShape shape_t(TW, TH);
    std::vector<float> h_time(NT);
    for (uint32_t y = 0; y < TH; ++y) {
        for (uint32_t x = 0; x < TW; ++x) {
            float fx = static_cast<float>(x) * 0.05f;
            float fy = static_cast<float>(y) * 0.05f;
            h_time[y * TW + x] = std::sin(fx) * std::cos(fy) * 10.0f + std::sin(fx * 2.3f + fy * 1.7f) * 3.0f;
        }
    }
    brass::gpu::VulkanBuffer d_in = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_tmp = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_out = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    d_in.upload(h_time.data(), NT * sizeof(float));

    // 1. Separable Blur (Horizontal + Vertical)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "SeparableBlur_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_tmp(N), ref_out(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    ref_tmp[y * W + x] = h_in[y * W + clamp_i(x - 2, 0, W - 1)] * 0.0625f
                                       + h_in[y * W + clamp_i(x - 1, 0, W - 1)] * 0.25f
                                       + h_in[y * W + x] * 0.375f
                                       + h_in[y * W + clamp_i(x + 1, 0, W - 1)] * 0.25f
                                       + h_in[y * W + clamp_i(x + 2, 0, W - 1)] * 0.0625f;
                }
            }
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    ref_out[y * W + x] = ref_tmp[clamp_i(y - 2, 0, H - 1) * W + x] * 0.0625f
                                       + ref_tmp[clamp_i(y - 1, 0, H - 1) * W + x] * 0.25f
                                       + ref_tmp[y * W + x] * 0.375f
                                       + ref_tmp[clamp_i(y + 1, 0, H - 1) * W + x] * 0.25f
                                       + ref_tmp[clamp_i(y + 2, 0, H - 1) * W + x] * 0.0625f;
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* kh = res.find_kernel("k_blur_h");
        const auto* kv = res.find_kernel("k_blur_v");
        std::vector<float> cpu_tmp(N), cpu_out(N), gpu_tmp(N), gpu_out(N);

        auto tc0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            kh->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_tmp)});
            kv->dispatch_cpu(shape2d, {KernelArg::store(cpu_tmp), KernelArg::store(cpu_out)});
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double bk_cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / 5.0;
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        kh->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_tmp)});
        kv->dispatch_gpu(shape2d, {KernelArg::store(gpu_tmp), KernelArg::store(gpu_out)});

        double bk_gpu_ms = time_bk_gpu_ms(kh, shape_t, {d_in, d_tmp}) +
                           time_bk_gpu_ms(kv, shape_t, {d_tmp, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int clx(int x) { return clamp(x, 0, pc.w - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    FloatBuf(pc.dst).v[y * pc.w + x] =
        s.v[y * pc.w + clx(x - 2)] * 0.0625 + s.v[y * pc.w + clx(x - 1)] * 0.25 +
        s.v[y * pc.w + x] * 0.375 + s.v[y * pc.w + clx(x + 1)] * 0.25 + s.v[y * pc.w + clx(x + 2)] * 0.0625;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_blur");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 2.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 2. Gradient & Normals from Height Grid
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "GradientNormals_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_out(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float dx = 0.5f * (h_in[y * W + clamp_i(x + 1, 0, W - 1)] - h_in[y * W + clamp_i(x - 1, 0, W - 1)]);
                    float dy = 0.5f * (h_in[clamp_i(y + 1, 0, H - 1) * W + x] - h_in[clamp_i(y - 1, 0, H - 1) * W + x]);
                    ref_out[y * W + x] = 1.0f / std::sqrt(dx * dx + dy * dy + 1.0f);
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_grad_normals");
        std::vector<float> cpu_out(N), gpu_out(N);
        auto tc0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            k->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_out)});
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double bk_cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / 5.0;
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        k->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_out)});
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float dx = 0.5 * (s.v[y * pc.w + cl(x + 1, pc.w)] - s.v[y * pc.w + cl(x - 1, pc.w)]);
    float dy = 0.5 * (s.v[cl(y + 1, pc.h) * pc.w + x] - s.v[cl(y - 1, pc.h) * pc.w + x]);
    FloatBuf(pc.dst).v[y * pc.w + x] = 1.0 / sqrt(dx * dx + dy * dy + 1.0);
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_grad_normals");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 3. Thermal Erosion (3 iterations)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "ThermalErosion_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> cur = h_in, nxt(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 3; ++iter) {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float hc = cur[y * W + x];
                    auto flux = [&](float hn) {
                        float d = hn - hc;
                        if (d > 0.1f) return d - 0.1f;
                        if (d < -0.1f) return d + 0.1f;
                        return 0.0f;
                    };
                    float f = flux(cur[y * W + clamp_i(x - 1, 0, W - 1)])
                            + flux(cur[y * W + clamp_i(x + 1, 0, W - 1)])
                            + flux(cur[clamp_i(y - 1, 0, H - 1) * W + x])
                            + flux(cur[clamp_i(y + 1, 0, H - 1) * W + x]);
                    nxt[y * W + x] = hc + 0.125f * f;
                }
            }
            cur.swap(nxt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_thermal_step");
        std::vector<float> cpu_a = h_in, cpu_b(N), gpu_a = h_in, gpu_b(N);
        auto tc0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_cpu(shape2d, {KernelArg::store(cpu_a), KernelArg::store(cpu_b)});
            cpu_a.swap(cpu_b);
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double bk_cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_gpu(shape2d, {KernelArg::store(gpu_a), KernelArg::store(gpu_b)});
            gpu_a.swap(gpu_b);
        }
        double bk_gpu_ms = 3.0 * time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_a, cur, 1e-4f);
        auto gs = compare_buffers(gpu_a, cur, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
float flux(float d) {
    if (d > 0.1) return d - 0.1;
    if (d < -0.1) return d + 0.1;
    return 0.0;
}
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    float f = flux(s.v[y * pc.w + cl(x - 1, pc.w)] - hc)
            + flux(s.v[y * pc.w + cl(x + 1, pc.w)] - hc)
            + flux(s.v[cl(y - 1, pc.h) * pc.w + x] - hc)
            + flux(s.v[cl(y + 1, pc.h) * pc.w + x] - hc);
    FloatBuf(pc.dst).v[y * pc.w + x] = hc + 0.125 * f;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_thermal");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 3.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 4. Hydraulic Erosion (3 iterations)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "HydraulicErosion_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> cur = h_in, nxt(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 3; ++iter) {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float hc = cur[y * W + x];
                    float hL = cur[y * W + clamp_i(x - 1, 0, W - 1)];
                    float hR = cur[y * W + clamp_i(x + 1, 0, W - 1)];
                    float hD = cur[clamp_i(y - 1, 0, H - 1) * W + x];
                    float hU = cur[clamp_i(y + 1, 0, H - 1) * W + x];
                    float dx = hR - hL;
                    float dy = hU - hD;
                    float slope = 0.5f * std::sqrt(dx * dx + dy * dy);
                    float lap = 0.25f * (hL + hR + hD + hU) - hc;
                    nxt[y * W + x] = hc - 0.05f * (slope * 0.15f) + 0.1f * lap;
                }
            }
            cur.swap(nxt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_hydraulic_step");
        std::vector<float> cpu_a = h_in, cpu_b(N), gpu_a = h_in, gpu_b(N);
        auto tc0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_cpu(shape2d, {KernelArg::store(cpu_a), KernelArg::store(cpu_b)});
            cpu_a.swap(cpu_b);
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double bk_cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_gpu(shape2d, {KernelArg::store(gpu_a), KernelArg::store(gpu_b)});
            gpu_a.swap(gpu_b);
        }
        double bk_gpu_ms = 3.0 * time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_a, cur, 1e-4f);
        auto gs = compare_buffers(gpu_a, cur, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    float hL = s.v[y * pc.w + cl(x - 1, pc.w)];
    float hR = s.v[y * pc.w + cl(x + 1, pc.w)];
    float hD = s.v[cl(y - 1, pc.h) * pc.w + x];
    float hU = s.v[cl(y + 1, pc.h) * pc.w + x];
    float dx = hR - hL, dy = hU - hD;
    float slope = 0.5 * sqrt(dx * dx + dy * dy);
    float lap = 0.25 * (hL + hR + hD + hU) - hc;
    FloatBuf(pc.dst).v[y * pc.w + x] = hc - 0.05 * (slope * 0.15) + 0.1 * lap;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_hydraulic");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 3.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 5. Flow Accumulation (8-neighbour upslope inflow)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "FlowAccumulation_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_out(N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float hc = h_in[y * W + x];
                    int xm = clamp_i(x - 1, 0, W - 1), xp = clamp_i(x + 1, 0, W - 1);
                    int ym = clamp_i(y - 1, 0, H - 1), yp = clamp_i(y + 1, 0, H - 1);
                    float card = std::max(0.0f, h_in[y * W + xm] - hc)
                               + std::max(0.0f, h_in[y * W + xp] - hc)
                               + std::max(0.0f, h_in[ym * W + x] - hc)
                               + std::max(0.0f, h_in[yp * W + x] - hc);
                    float diag = std::max(0.0f, h_in[ym * W + xm] - hc)
                               + std::max(0.0f, h_in[ym * W + xp] - hc)
                               + std::max(0.0f, h_in[yp * W + xm] - hc)
                               + std::max(0.0f, h_in[yp * W + xp] - hc);
                    ref_out[y * W + x] = 1.0f + card + 0.70710678f * diag;
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_flow_accum");
        std::vector<float> cpu_out(N), gpu_out(N);
        auto tc0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 5; ++rep) {
            k->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_out)});
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double bk_cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / 5.0;
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        k->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_out)});
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    int xm = cl(x - 1, pc.w), xp = cl(x + 1, pc.w);
    int ym = cl(y - 1, pc.h), yp = cl(y + 1, pc.h);
    float card = max(0.0, s.v[y * pc.w + xm] - hc) + max(0.0, s.v[y * pc.w + xp] - hc)
               + max(0.0, s.v[ym * pc.w + x] - hc) + max(0.0, s.v[yp * pc.w + x] - hc);
    float diag = max(0.0, s.v[ym * pc.w + xm] - hc) + max(0.0, s.v[ym * pc.w + xp] - hc)
               + max(0.0, s.v[yp * pc.w + xm] - hc) + max(0.0, s.v[yp * pc.w + xp] - hc);
    FloatBuf(pc.dst).v[y * pc.w + x] = 1.0 + card + 0.70710678 * diag;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_flow_accum");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    return rows;
}

void write_golden_json(const fs::path& path, const std::vector<BenchRow>& rows,
                       int parity_passed, int parity_total,
                       double geomean_cpu, double geomean_gpu) {
    std::ofstream ofs(path);
    ofs << "{\n";
    ofs << "  \"parity_passed\": " << parity_passed << ",\n";
    ofs << "  \"parity_total\": " << parity_total << ",\n";
    ofs << "  \"geomean_cpu_vs_ref\": " << std::fixed << std::setprecision(4) << geomean_cpu << ",\n";
    ofs << "  \"geomean_gpu_vs_glsl\": " << std::fixed << std::setprecision(4) << geomean_gpu << ",\n";
    ofs << "  \"benchmarks\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        ofs << "    {\n";
        ofs << "      \"tier\": \"" << r.tier << "\",\n";
        ofs << "      \"name\": \"" << r.name << "\",\n";
        ofs << "      \"dim\": " << r.dim << ",\n";
        ofs << "      \"points\": " << r.points << ",\n";
        ofs << "      \"cpu_pass\": " << (r.cpu_pass ? "true" : "false") << ",\n";
        ofs << "      \"gpu_pass\": " << (r.gpu_pass ? "true" : "false") << ",\n";
        ofs << "      \"cpu_max_abs\": " << std::scientific << std::setprecision(3) << r.cpu_max_abs << ",\n";
        ofs << "      \"cpu_max_ulp\": " << r.cpu_max_ulp << ",\n";
        ofs << "      \"gpu_max_abs\": " << std::scientific << std::setprecision(3) << r.gpu_max_abs << ",\n";
        ofs << "      \"gpu_max_ulp\": " << r.gpu_max_ulp << ",\n";
        ofs << "      \"cpu_bk_mpts\": " << std::fixed << std::setprecision(2) << r.cpu_bk_mpts << ",\n";
        ofs << "      \"cpu_ref_mpts\": " << std::fixed << std::setprecision(2) << r.cpu_ref_mpts << ",\n";
        ofs << "      \"cpu_ratio\": " << std::fixed << std::setprecision(4) << r.cpu_ratio << ",\n";
        ofs << "      \"gpu_bk_mpts\": " << std::fixed << std::setprecision(2) << r.gpu_bk_mpts << ",\n";
        ofs << "      \"gpu_glsl_mpts\": " << std::fixed << std::setprecision(2) << r.gpu_glsl_mpts << ",\n";
        ofs << "      \"gpu_ratio\": " << std::fixed << std::setprecision(4) << r.gpu_ratio << "\n";
        ofs << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

// A top-level number of the golden file ("key": value), or a negative when absent.
double read_golden_number(const fs::path& path, const std::string& key) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string text = ss.str();
    size_t at = text.find("\"" + key + "\"");
    if (at == std::string::npos) return -1.0;
    at = text.find(':', at);
    return at == std::string::npos ? -1.0 : std::strtod(text.c_str() + at + 1, nullptr);
}

// One of the planet's FastNoise fields (broworkshop 1fbd228^:demos/planet/terrain.js):
// FractalFBm of Simplex at a feature scale, lacunarity 2.03, times an amplitude.
struct PlanetField {
    float scale;
    int octaves;
    int seed_offset;
    float gain;
    float amp;
};

// WARP, CONTINENT and RELIEF, the four regional relief bands, and TREE.
const PlanetField kPlanetFields[] = {
    {2.2e6f, 4, 7, 0.5f, 1.0f},
    {3.0e6f, 6, 9, 0.55f, 1.0f},
    {60000.0f, 2, 0, 0.5f, 1.0f},
    {500.0f, 2, 31, 0.5f, 30.0f},
    {2000.0f, 2, 32, 0.5f, 120.0f},
    {8000.0f, 2, 33, 0.5f, 330.0f},
    {32000.0f, 2, 34, 0.5f, 520.0f},
    {120.0f, 3, 3, 0.5f, 3.0f},
};

// The fields summed as one FastNoise2 tree: Add(Add(f0, f1), f2)..., a field
// with an amplitude a Multiply by it.
FastNoise::SmartNode<> planet_fields_tree() {
    FastNoise::SmartNode<> sum;
    for (const auto& f : kPlanetFields) {
        auto simplex = FastNoise::New<FastNoise::Simplex>();
        simplex->SetScale(f.scale);
        simplex->SetSeedOffset(f.seed_offset);
        auto fbm = FastNoise::New<FastNoise::FractalFBm>();
        fbm->SetSource(simplex);
        fbm->SetOctaveCount(f.octaves);
        fbm->SetLacunarity(2.03f);
        fbm->SetGain(f.gain);
        FastNoise::SmartNode<> field = fbm;
        if (f.amp != 1.0f) {
            auto mul = FastNoise::New<FastNoise::Multiply>();
            mul->SetLHS(fbm);
            mul->SetRHS(f.amp);
            field = mul;
        }
        if (!sum) {
            sum = field;
        } else {
            auto add = FastNoise::New<FastNoise::Add>();
            add->SetLHS(sum);
            add->SetRHS(field);
            sum = add;
        }
    }
    return sum;
}

// The same fields as one brokernel kernel, in f64 about a double origin. Its
// constants are FastNoise2's floats (1 / scale, 2.03, the gains) widened, so
// both evaluate the same lattice positions.
std::string planet_fields_source() {
    std::ostringstream ss;
    ss << std::setprecision(17);
    ss << "import std::noise::*;\n"
          "@kernel @grid(1)\n"
          "fn k_planet_fields(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>,\n"
          "                   out: write_store<f32, 1>, ox: f64, oy: f64, oz: f64, seed: i32) {\n"
          "    let i = g.x();\n"
          "    let p = vec3<f64>((px[i] as f64) + ox, (py[i] as f64) + oy, (pz[i] as f64) + oz);\n";
    bool first = true;
    for (const auto& f : kPlanetFields) {
        ss << (first ? "    var h: f32 = " : "    h = h + ");
        ss << "(fbm_simplex<f64, 3>(p * " << double(1.0f / f.scale) << ", " << f.octaves << ", "
           << double(2.03f) << ", " << double(f.gain) << ", 0.0, seed + " << f.seed_offset << ") as f32)";
        if (f.amp != 1.0f) ss << " * " << std::fixed << std::setprecision(1) << f.amp << std::defaultfloat << std::setprecision(17);
        ss << ";\n";
        first = false;
    }
    ss << "    out[i] = h;\n}\n";
    return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    bool write_golden = false;
    bool check_golden = false;
    fs::path golden_path = fs::path(BROKERNEL_SOURCE_DIR) / "bench" / "goldens.windows.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--write-golden") write_golden = true;
        else if (arg == "--check-golden") check_golden = true;
        else if (arg == "--golden" && i + 1 < argc) golden_path = argv[++i];
    }

    if (!brass::gpu::vulkan_available()) {
        std::cerr << "bench_hillclimb: Vulkan compute device unavailable\n";
        return 1;
    }

    auto suite_t0 = std::chrono::high_resolution_clock::now();

    CompileOptions copts;
    copts.opt_level = 3;
    auto noise_res = compile_source(kBkNoiseSuiteSource, "noise_hillclimb.bk", copts);
    if (!noise_res.ok) {
        std::cerr << "Noise compile error:\n" << noise_res.error_message() << "\n";
        return 1;
    }

    constexpr uint32_t N = kParityPoints;
    constexpr uint32_t NT = kGpuTimePoints;
    constexpr int kSeed = 1337;
    const GridShape shape1d(N);
    const GridShape shape_time(NT);

    // The positions: the first N are the parity set, all NT the GPU's timing set.
    std::vector<float> px(NT), py(NT), pz(NT), pw(NT);
    for (uint32_t i = 0; i < NT; ++i) {
        float t = static_cast<float>(i & 4095u) * 0.03125f - 32.0f + static_cast<float>(i >> 12) * 0.0009765625f;
        px[i] = t * 1.13f + 0.17f;
        py[i] = std::sin(t * 0.37f) * 14.3f - 2.1f;
        pz[i] = std::cos(t * 0.23f) * 11.7f + 3.4f;
        pw[i] = std::sin(t * 0.19f + 1.1f) * 9.5f - 1.3f;
    }

    // Device buffers for the GPU's timed runs (timestamp queries, inputs uploaded once).
    brass::gpu::VulkanBuffer d_px = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_py = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_pz = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_pw = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_out = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    d_px.upload(px.data(), NT * sizeof(float));
    d_py.upload(py.data(), NT * sizeof(float));
    d_pz.upload(pz.data(), NT * sizeof(float));
    d_pw.upload(pw.data(), NT * sizeof(float));

    // The GLSL baselines: the same FastNoise2 node, hand-written. A row whose
    // node has none here has no GPU ratio.
    std::vector<brass::Type> layout2 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    std::vector<brass::Type> layout3 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    std::vector<brass::Type> layout4 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    auto glsl_module = [&](int dim, const std::string& expr, const std::string& tag) {
        auto spv = compile_glsl_compute(make_glsl_noise_shader(dim, expr), tag);
        return load_glsl_vulkan_module(spv, dim == 2 ? layout2 : dim == 3 ? layout3 : layout4);
    };
    auto vmod_val2 = glsl_module(2, "fn_value2d(p, seed)", "glsl_val2");
    auto vmod_val3 = glsl_module(3, "fn_value3d(p, seed)", "glsl_val3");
    auto vmod_val4 = glsl_module(4, "fn_value4d(p, seed)", "glsl_val4");
    auto vmod_perlin2 = glsl_module(2, "fn_perlin2d(p, seed)", "glsl_perlin2");
    auto vmod_perlin3 = glsl_module(3, "fn_perlin3d(p, seed)", "glsl_perlin3");
    auto vmod_perlin4 = glsl_module(4, "fn_perlin4d(p, seed)", "glsl_perlin4");
    auto vmod_fbm2 = glsl_module(2, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm2");
    auto vmod_fbm3 = glsl_module(3, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm3");
    auto vmod_fbm4 = glsl_module(4, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm4");
    auto vmod_ridged2 = glsl_module(2, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged2");
    auto vmod_ridged3 = glsl_module(3, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged3");
    auto vmod_ridged4 = glsl_module(4, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged4");

    std::vector<float> ref(N), cpu_out(N), gpu_out(N), glsl_out(N);
    std::vector<BenchRow> all_rows;

    auto bench_noise = [&](const std::string& tier, const std::string& name, int dim, const char* kname,
                           FastNoise::SmartNode<> fn, const brass::gpu::VulkanModule* baseline, float tol) {
        BenchRow row;
        row.tier = tier;
        row.name = name;
        row.dim = dim;
        row.points = N;

        const auto* k = noise_res.find_kernel(kname);

        double fn_ms = best_cpu_ms([&] {
            if (dim == 2) fn->GenPositionArray2D(ref.data(), N, px.data(), py.data(), 0.0f, 0.0f, kSeed);
            else if (dim == 3) fn->GenPositionArray3D(ref.data(), N, px.data(), py.data(), pz.data(), 0.0f, 0.0f, 0.0f, kSeed);
            else fn->GenPositionArray4D(ref.data(), N, px.data(), py.data(), pz.data(), pw.data(), 0.0f, 0.0f, 0.0f, 0.0f, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        std::vector<KernelArg> cargs, gargs;
        std::vector<brass::gpu::VulkanArg> vk_args;
        if (dim == 2) {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_out, kSeed};
        } else if (dim == 3) {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_pz, d_out, kSeed};
        } else {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_pz, d_pw, d_out, kSeed};
        }

        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);

        k->dispatch_gpu(shape1d, gargs);

        // The baseline counts only if it computes the node: its first N
        // points against FastNoise2 to the same tolerance.
        if (baseline) {
            std::vector<brass::gpu::VulkanArg> bargs(vk_args.begin(), vk_args.end());
            bargs.emplace_back(static_cast<int32_t>(NT));
            double glsl_ms = time_glsl_gpu_ms(*baseline, NT / 64, 1, 1, 64, 1, 1, bargs);
            d_out.download(glsl_out.data(), N * sizeof(float));
            auto bs = compare_buffers(glsl_out, ref, tol);
            if (bs.pass) {
                row.has_gpu_baseline = true;
                row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
            } else {
                std::cerr << name << ": the GLSL baseline does not match FastNoise2 (max err "
                          << bs.max_abs << "), no GPU ratio\n";
            }
        }
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_time, vk_args);
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref, tol);
        auto gs = compare_buffers(gpu_out, ref, tol);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        bool pass = row.cpu_pass && row.gpu_pass;
        row.cpu_ratio = pass ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.gpu_ratio = (pass && row.has_gpu_baseline) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        all_rows.push_back(row);
    };

    // 1. Value (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Value>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "Value_2D", 2, "k_value2d", fn, &vmod_val2, kTol);
        bench_noise("Noise", "Value_3D", 3, "k_value3d", fn, &vmod_val3, kTol);
        bench_noise("Noise", "Value_4D", 4, "k_value4d", fn, &vmod_val4, kTol);
    }
    // 2. Perlin (2D, 3D, 4D)
    auto fn_perlin = FastNoise::New<FastNoise::Perlin>();
    fn_perlin->SetScale(1.0f);
    bench_noise("Noise", "Perlin_2D", 2, "k_perlin2d", fn_perlin, &vmod_perlin2, kTol);
    bench_noise("Noise", "Perlin_3D", 3, "k_perlin3d", fn_perlin, &vmod_perlin3, kTol);
    bench_noise("Noise", "Perlin_4D", 4, "k_perlin4d", fn_perlin, &vmod_perlin4, kTol);

    // 3. Simplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "Simplex_2D", 2, "k_simplex2d", fn, nullptr, kTol);
        bench_noise("Noise", "Simplex_3D", 3, "k_simplex3d", fn, nullptr, kTol);
        bench_noise("Noise", "Simplex_4D", 4, "k_simplex4d", fn, nullptr, kTol);
    }
    // 4. SuperSimplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::SuperSimplex>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "SuperSimplex_2D", 2, "k_supersimplex2d", fn, nullptr, kTol);
        bench_noise("Noise", "SuperSimplex_3D", 3, "k_supersimplex3d", fn, nullptr, kTol);
        bench_noise("Noise", "SuperSimplex_4D", 4, "k_supersimplex4d", fn, nullptr, kTol);
    }
    // 5. White (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::White>();
        bench_noise("Noise", "White_2D", 2, "k_white2d", fn, nullptr, kTol);
        bench_noise("Noise", "White_3D", 3, "k_white3d", fn, nullptr, kTol);
        bench_noise("Noise", "White_4D", 4, "k_white4d", fn, nullptr, kTol);
    }
    // 6. FractalFBm (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalFBm>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "FractalFBm_2D", 2, "k_fbm2d", fn, &vmod_fbm2, kTol);
        bench_noise("Noise", "FractalFBm_3D", 3, "k_fbm3d", fn, &vmod_fbm3, kTol);
        bench_noise("Noise", "FractalFBm_4D", 4, "k_fbm4d", fn, &vmod_fbm4, kTol);
    }
    // 7. FractalRidged (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalRidged>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "FractalRidged_2D", 2, "k_ridged2d", fn, &vmod_ridged2, kTol);
        bench_noise("Noise", "FractalRidged_3D", 3, "k_ridged3d", fn, &vmod_ridged3, kTol);
        bench_noise("Noise", "FractalRidged_4D", 4, "k_ridged4d", fn, &vmod_ridged4, kTol);
    }
    // 8. PingPong (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::PingPong>();
        fn->SetSource(fn_perlin);
        fn->SetPingPongStrength(2.0f);
        bench_noise("Noise", "PingPong_2D", 2, "k_pingpong2d", fn, nullptr, kTol);
        bench_noise("Noise", "PingPong_3D", 3, "k_pingpong3d", fn, nullptr, kTol);
        bench_noise("Noise", "PingPong_4D", 4, "k_pingpong4d", fn, nullptr, kTol);
    }
    // 9. CellularValue (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularValue>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularValue_2D", 2, "k_cell_val2d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularValue_3D", 3, "k_cell_val3d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularValue_4D", 4, "k_cell_val4d", fn, nullptr, kTolRelaxed);
    }
    // 10. CellularDistance (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularDistance>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularDistance_2D", 2, "k_cell_dist2d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularDistance_3D", 3, "k_cell_dist3d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularDistance_4D", 4, "k_cell_dist4d", fn, nullptr, kTolRelaxed);
    }
    // 11. CellularLookup (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularLookup>();
        fn->SetScale(1.0f);
        fn->SetLookup(fn_perlin);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularLookup_2D", 2, "k_cell_lookup2d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularLookup_3D", 3, "k_cell_lookup3d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "CellularLookup_4D", 4, "k_cell_lookup4d", fn, nullptr, kTolRelaxed);
    }
    // 12. DomainWarpGradient (2D, 3D, 4D)
    auto fn_warp_grad = FastNoise::New<FastNoise::DomainWarpGradient>();
    fn_warp_grad->SetSource(fn_perlin);
    fn_warp_grad->SetWarpAmplitude(1.5f);
    fn_warp_grad->SetScale(2.0f);
    bench_noise("Noise", "DomainWarpGrad_2D", 2, "k_warp_grad2d", fn_warp_grad, nullptr, kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_3D", 3, "k_warp_grad3d", fn_warp_grad, nullptr, kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_4D", 4, "k_warp_grad4d", fn_warp_grad, nullptr, kTolRelaxed);

    // 13. DomainWarpFractalProgressive (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalProgressive>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpProg_2D", 2, "k_warp_prog2d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_3D", 3, "k_warp_prog3d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_4D", 4, "k_warp_prog4d", fn, nullptr, kTolRelaxed);
    }
    // 14. DomainWarpFractalIndependent (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalIndependent>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpIndep_2D", 2, "k_warp_indep2d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_3D", 3, "k_warp_indep3d", fn, nullptr, kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_4D", 4, "k_warp_indep4d", fn, nullptr, kTolRelaxed);
    }

    // 15. Planetary double origin (6,371,000 m) Simplex 3D
    {
        BenchRow row;
        row.tier = "Noise";
        row.name = "Simplex_3D_f64_Planet";
        row.dim = 3;
        row.points = N;

        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        const double origin = 6371000.0;
        std::vector<float> px_zero(N, 0.0f);

        // Reference via GenSingle3D at (double(px[i]) + origin) for exact parity oracle
        for (uint32_t i = 0; i < N; ++i) {
            ref[i] = fn->GenSingle3D(double(px[i]) + origin, double(py[i]) + origin, double(pz[i]) + origin, kSeed);
        }
        // FastNoise2's own speed: GenPositionArray3D about the double origin.
        std::vector<float> fn_bench_out(N);
        double fn_ms = best_cpu_ms([&] {
            fn->GenPositionArray3D(fn_bench_out.data(), N, px.data(), py.data(), pz.data(), origin, origin, origin, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        const auto* k = noise_res.find_kernel("k_planet_f64_simplex3d");
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(origin), KernelArg(kSeed)};
        std::vector<KernelArg> gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(gpu_out), KernelArg(origin), KernelArg(kSeed)};

        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);

        k->dispatch_gpu(shape1d, gargs);
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_time, {d_px, d_py, d_pz, d_out, origin, kSeed});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref, kTol);
        auto gs = compare_buffers(gpu_out, ref, kTol);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        all_rows.push_back(row);
    }

    // Tier 2: Stencil Kernels
    auto stencil_rows = run_stencil_tier();
    for (const auto& r : stencil_rows) all_rows.push_back(r);

    // Capstone: the planet's noise fields (kPlanetFields), summed, over an
    // 8 km patch of its surface: float offsets about a double origin on the
    // sphere, as the planet sampled them. Parity is in metres of height, and
    // FastNoise2 scales the offsets in float, so a wider patch is less exact
    // on its side (16 km out a float's step is a millimetre).
    {
        BenchRow row;
        row.tier = "Capstone";
        row.name = "PlanetFields_3D_f64";
        row.dim = 3;
        row.points = N;

        auto planet_res = compile_source(planet_fields_source(), "planet_fields.bk", copts);
        if (!planet_res.ok) {
            std::cerr << "Planet fields compile error:\n" << planet_res.error_message() << "\n";
            return 1;
        }
        const auto* k = planet_res.find_kernel("k_planet_fields");
        auto tree = planet_fields_tree();

        const double R = 6371000.0;
        const double ox = R * 0.36, oy = R * 0.48, oz = R * 0.8;
        std::vector<float> qx(NT), qy(NT), qz(NT);
        for (uint32_t i = 0; i < NT; ++i) {
            qx[i] = px[i] * 100.0f;
            qy[i] = py[i] * 100.0f;
            qz[i] = pz[i] * 100.0f;
        }
        double fn_ms = best_cpu_ms([&] {
            tree->GenPositionArray3D(ref.data(), N, qx.data(), qy.data(), qz.data(), ox, oy, oz, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        std::vector<KernelArg> cargs = {KernelArg::store(qx), KernelArg::store(qy), KernelArg::store(qz), KernelArg::store(cpu_out),
                                        KernelArg(ox), KernelArg(oy), KernelArg(oz), KernelArg(kSeed)};
        std::vector<KernelArg> gargs = {KernelArg::store(qx), KernelArg::store(qy), KernelArg::store(qz), KernelArg::store(gpu_out),
                                        KernelArg(ox), KernelArg(oy), KernelArg(oz), KernelArg(kSeed)};
        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);
        k->dispatch_gpu(shape1d, gargs);

        d_px.upload(qx.data(), NT * sizeof(float));
        d_py.upload(qy.data(), NT * sizeof(float));
        d_pz.upload(qz.data(), NT * sizeof(float));
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_time, {d_px, d_py, d_pz, d_out, ox, oy, oz, kSeed});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        // Heights of up to ~1000 m (the bands' amplitudes), to a millimetre.
        auto cs = compare_buffers(cpu_out, ref, 1e-3f);
        auto gs = compare_buffers(gpu_out, ref, 1e-3f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        all_rows.push_back(row);
    }

    auto suite_t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(suite_t1 - suite_t0).count();

    // Print unified table
    std::printf("\n============================================================================================================================\n");
    std::printf("%-9s | %-24s | %-6s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s\n",
                "Tier", "Node / Kernel", "Parity", "CPU MaxErr", "GPU MaxErr", "CPU Mpt/s", "GPU Mpt/s", "CPU/Ref", "GPU/GLSL");
    std::printf("----------------------------------------------------------------------------------------------------------------------------\n");

    int parity_passed = 0;
    int parity_total = static_cast<int>(all_rows.size());
    double log_sum_cpu = 0.0;
    double log_sum_gpu = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;

    for (const auto& r : all_rows) {
        bool both_pass = r.cpu_pass && r.gpu_pass;
        if (both_pass) {
            parity_passed++;
            if (r.cpu_ratio > 0.0) {
                log_sum_cpu += std::log(r.cpu_ratio);
                cpu_count++;
            }
            if (r.gpu_ratio > 0.0) {
                log_sum_gpu += std::log(r.gpu_ratio);
                gpu_count++;
            }
        }
        char gpu_ratio[16];
        if (r.has_gpu_baseline) std::snprintf(gpu_ratio, sizeof gpu_ratio, "%7.2fx", r.gpu_ratio);
        else std::snprintf(gpu_ratio, sizeof gpu_ratio, "%8s", "n/a");
        std::printf("%-9s | %-24s | %-6s | %10.2e | %10.2e | %10.2f | %10.2f | %7.2fx | %s\n",
                    r.tier.c_str(),
                    r.name.c_str(),
                    both_pass ? "PASS" : "FAIL",
                    r.cpu_max_abs,
                    r.gpu_max_abs,
                    r.cpu_bk_mpts,
                    r.gpu_bk_mpts,
                    r.cpu_ratio,
                    gpu_ratio);
    }

    double geomean_cpu = cpu_count > 0 ? std::exp(log_sum_cpu / cpu_count) : 0.0;
    double geomean_gpu = gpu_count > 0 ? std::exp(log_sum_gpu / gpu_count) : 0.0;
    const auto& capstone = all_rows.back();

    std::printf("============================================================================================================================\n");
    std::printf("ROLL-UP SCORES:\n");
    std::printf("  1. Completeness / Parity : %d / %d (%5.1f%%) nodes & stencils pass CPU and GPU parity\n",
                parity_passed, parity_total, 100.0 * parity_passed / std::max(parity_total, 1));
    std::printf("  2. Speed (GeoMean CPU)   : %.3fx relative to FastNoise2 SIMD / C++ Reference\n", geomean_cpu);
    std::printf("  3. Speed (GeoMean GPU)   : %.3fx relative to glslc SPIR-V of the same node (%d rows with one)\n",
                geomean_gpu, gpu_count);
    std::printf("  4. Capstone Planet Fields: %s (CPU: %.2f Mpt/s [%.2fx FastNoise2], GPU: %.2f Mpt/s, MaxErr: %.2e m)\n",
                (capstone.cpu_pass && capstone.gpu_pass) ? "PASS" : "FAIL",
                capstone.cpu_bk_mpts, capstone.cpu_ratio, capstone.gpu_bk_mpts,
                std::max(capstone.cpu_max_abs, capstone.gpu_max_abs));
    std::printf("  Total Harness Wall Time  : %.2f s\n", total_sec);
    std::printf("============================================================================================================================\n");

    if (write_golden) {
        write_golden_json(golden_path, all_rows, parity_passed, parity_total, geomean_cpu, geomean_gpu);
        std::cout << "Wrote golden ratchet file: " << golden_path.string() << "\n";
    }

    if (check_golden) {
        if (!fs::exists(golden_path)) {
            std::cerr << "Golden file missing: " << golden_path.string() << "\n";
            return 1;
        }
        if (parity_passed != parity_total) {
            std::cerr << "Parity regression: " << parity_passed << " / " << parity_total << "\n";
            return 1;
        }
        // The geomeans may not fall more than kNoise below the golden's: run
        // to run they move a few percent (CPU load, GPU clocks).
        constexpr double kNoise = 0.10;
        bool ok = true;
        auto check = [&](const char* key, double now) {
            double was = read_golden_number(golden_path, key);
            if (was <= 0.0) {
                std::cerr << "Golden file has no " << key << "\n";
                ok = false;
            } else if (now < was * (1.0 - kNoise)) {
                std::cerr << "Speed regression: " << key << " " << now << " < golden " << was << "\n";
                ok = false;
            }
        };
        check("geomean_cpu_vs_ref", geomean_cpu);
        check("geomean_gpu_vs_glsl", geomean_gpu);
        if (!ok) return 1;
        std::cout << "Golden ratchet check PASSED (" << parity_passed << "/" << parity_total
                  << " parity, geomeans within " << int(kNoise * 100) << "% of the golden or above).\n";
    }

    return (parity_passed == parity_total) ? 0 : 1;
}

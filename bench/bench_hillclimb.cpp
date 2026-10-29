// Hill-Climb Benchmark & Parity Harness for brokernel:
// - Tier 1: FastNoise2 Node Suite across 2D, 3D, 4D + f64 planetary origin (43 configs)
//   * Parity oracle: FastNoise2 C++ SIMD
//   * CPU speed baseline: FastNoise2 C++ SIMD (best of kCpuReps runs, single thread)
//   * GPU speed baseline: the same node as a GLSL 450 compute shader compiled by glslc (-O),
//     every f32 node (kGlslNoiseHeader); its output is checked against FastNoise2 before
//     it counts. The f64 rows have no GPU ratio. The cheap nodes are memory-bound at
//     ~130,000 Mpt/s on a 4090 either way; the fractal, cellular and warp rows are the
//     ones whose ratio measures the code.
// - Tier 2: Stencil & Grid Kernels (Separable Blur, Gradient/Normals, Thermal Erosion,
//           Hydraulic Erosion, Flow Accumulation, and the planet's rivers: PlanetFlow,
//           its erosion pass, five fields read at eight neighbours). On the GPU the
//           planet's flow is memory-bound per iteration (~36 bytes a cell).
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

// The GLSL baselines: FastNoise2's nodes in f32, in std::noise's order of operations
// (Value, Perlin, the fractals, Simplex, SuperSimplex, White, PingPong, Cellular, the
// gradient warps). Split into literals under MSVC's 16 KB limit.
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
)GLSL" R"GLSL(
float fn_simplex2d(vec2 p, int seed) {
    const float k_unskew = -0.211324870586395263671875;
    float skew_delta = 0.3660254180431365966796875 * (p.x + p.y);
    float x_skewed = p.x + skew_delta;
    float y_skewed = p.y + skew_delta;
    float x_base = floor(x_skewed);
    float y_base = floor(y_skewed);
    float dx_skewed = x_skewed - x_base;
    float dy_skewed = y_skewed - y_base;
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    bool x_ge_y = dx_skewed >= dy_skewed;
    float unskew_delta = k_unskew * (dx_skewed + dy_skewed);
    float dx0 = dx_skewed + unskew_delta;
    float dy0 = dy_skewed + unskew_delta;
    float dx1 = (x_ge_y ? dx0 : (dx0 + 1.0)) - (k_unskew + 1.0);
    float dy1 = (x_ge_y ? (dy0 + 1.0) : dy0) - (k_unskew + 1.0);
    float dx2 = dx0 - (k_unskew * 2.0 + 1.0);
    float dy2 = dy0 - (k_unskew * 2.0 + 1.0);
    float f0 = 0.5 - dy0 * dy0 - dx0 * dx0;
    float f1 = 0.5 - dy1 * dy1 - dx1 * dx1;
    float f2 = f0 + (unskew_delta * -3.1547005176544189453125 - 0.666666686534881591796875);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1); f2 = (f2 * f2) * (f2 * f2);
    float g0 = grad_dot_perlin2d(hash_primes2d(seed, x_primed, y_primed), dx0, dy0);
    float g1 = grad_dot_perlin2d(hash_primes2d(seed, x_ge_y ? (x_primed + PRIME_X) : x_primed,
                                               x_ge_y ? y_primed : (y_primed + PRIME_Y)), dx1, dy1);
    float g2 = grad_dot_perlin2d(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y), dx2, dy2);
    float val = g0 * f0 + (g1 * f1 + g2 * f2);
    return scale_output(val, -0.0261208079755306243896484375, 0.0261208079755306243896484375);
}
float fn_simplex3d(vec3 p, int seed) {
    float skew_delta = 0.3333333432674407958984375 * (p.x + p.y + p.z);
    vec3 skewed = p + skew_delta;
    vec3 base = floor(skewed);
    vec3 d = skewed - base;
    int x_primed = int(base.x) * PRIME_X;
    int y_primed = int(base.y) * PRIME_Y;
    int z_primed = int(base.z) * PRIME_Z;
    bool x_ge_y = d.x >= d.y;
    bool y_ge_z = d.y >= d.z;
    bool x_ge_z = d.x >= d.z;
    float unskew_delta = -0.5 * (d.x + d.y + d.z);
    float dx0 = d.x + unskew_delta;
    float dy0 = d.y + unskew_delta;
    float dz0 = d.z + unskew_delta;
    bool mask_x1 = x_ge_y && x_ge_z;
    bool mask_y1 = (!x_ge_y) && y_ge_z;
    bool mask_z1 = x_ge_z || y_ge_z;
    bool nmask_x2 = x_ge_y || x_ge_z;
    bool nmask_y2 = (!y_ge_z) && x_ge_y;
    bool nmask_z2 = x_ge_z && y_ge_z;
    float dx3 = dx0 + 0.5;
    float dy3 = dy0 + 0.5;
    float dz3 = dz0 + 0.5;
    float dx1 = mask_x1 ? (dx3 - 1.0) : dx3;
    float dy1 = mask_y1 ? (dy3 - 1.0) : dy3;
    float dz1 = (!mask_z1) ? (dz3 - 1.0) : dz3;
    float dx2 = (!nmask_x2) ? (dx0 + 1.0) : dx0;
    float dy2 = nmask_y2 ? (dy0 + 1.0) : dy0;
    float dz2 = nmask_z2 ? (dz0 + 1.0) : dz0;
    float f0 = (0.6 - dx0 * dx0 - dy0 * dy0) - dz0 * dz0;
    float f1 = (0.6 - dx1 * dx1 - dy1 * dy1) - dz1 * dz1;
    float f2 = (0.6 - dx2 * dx2 - dy2 * dy2) - dz2 * dz2;
    float f3 = f0 - (unskew_delta + 0.75);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0); f3 = max(f3, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1);
    f2 = (f2 * f2) * (f2 * f2); f3 = (f3 * f3) * (f3 * f3);
    float g0 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed), dx0, dy0, dz0);
    float g1 = grad_dot_common3d(hash_primes3d(seed,
        mask_x1 ? (x_primed + PRIME_X) : x_primed,
        mask_y1 ? (y_primed + PRIME_Y) : y_primed,
        (!mask_z1) ? (z_primed + PRIME_Z) : z_primed), dx1, dy1, dz1);
    float g2 = grad_dot_common3d(hash_primes3d(seed,
        nmask_x2 ? (x_primed + PRIME_X) : x_primed,
        (!nmask_y2) ? (y_primed + PRIME_Y) : y_primed,
        (!nmask_z2) ? (z_primed + PRIME_Z) : z_primed), dx2, dy2, dz2);
    float g3 = grad_dot_common3d(hash_primes3d(seed, x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z), dx3, dy3, dz3);
    float val = g3 * f3 + (g2 * f2 + (g1 * f1 + g0 * f0));
    return scale_output(val, -0.030586399137973785400390625, 0.030586399137973785400390625);
}
float fn_simplex4d(vec4 p, int seed) {
    const float k_unskew = -0.13819660112501051517954131656344;
    float skew_delta = 0.30901699437494742410229341718282 * (p.x + p.y + p.z + p.w);
    vec4 skewed = p + skew_delta;
    vec4 base = floor(skewed);
    vec4 d = skewed - base;
    int x_primed = int(base.x) * PRIME_X;
    int y_primed = int(base.y) * PRIME_Y;
    int z_primed = int(base.z) * PRIME_Z;
    int w_primed = int(base.w) * PRIME_W;
    float unskew_delta = k_unskew * (d.x + d.y + d.z + d.w);
    float dx0 = d.x + unskew_delta;
    float dy0 = d.y + unskew_delta;
    float dz0 = d.z + unskew_delta;
    float dw0 = d.w + unskew_delta;
    int rx = 0, ry = 0, rz = 0, rw = 0;
    if (dx0 >= dy0) rx += 1; else ry += 1;
    if (dx0 >= dz0) rx += 1; else rz += 1;
    if (dx0 >= dw0) rx += 1; else rw += 1;
    if (dy0 >= dz0) ry += 1; else rz += 1;
    if (dy0 >= dw0) ry += 1; else rw += 1;
    if (dz0 >= dw0) rz += 1; else rw += 1;
    float dx1 = (rx > 2 ? (dx0 - 1.0) : dx0) - k_unskew;
    float dy1 = (ry > 2 ? (dy0 - 1.0) : dy0) - k_unskew;
    float dz1 = (rz > 2 ? (dz0 - 1.0) : dz0) - k_unskew;
    float dw1 = (rw > 2 ? (dw0 - 1.0) : dw0) - k_unskew;
    float dx2 = (rx > 1 ? (dx0 - 1.0) : dx0) - (k_unskew * 2.0);
    float dy2 = (ry > 1 ? (dy0 - 1.0) : dy0) - (k_unskew * 2.0);
    float dz2 = (rz > 1 ? (dz0 - 1.0) : dz0) - (k_unskew * 2.0);
    float dw2 = (rw > 1 ? (dw0 - 1.0) : dw0) - (k_unskew * 2.0);
    float dx3 = (rx > 0 ? (dx0 - 1.0) : dx0) - (k_unskew * 3.0);
    float dy3 = (ry > 0 ? (dy0 - 1.0) : dy0) - (k_unskew * 3.0);
    float dz3 = (rz > 0 ? (dz0 - 1.0) : dz0) - (k_unskew * 3.0);
    float dw3 = (rw > 0 ? (dw0 - 1.0) : dw0) - (k_unskew * 3.0);
    float dx4 = dx0 - (k_unskew * 4.0 + 1.0);
    float dy4 = dy0 - (k_unskew * 4.0 + 1.0);
    float dz4 = dz0 - (k_unskew * 4.0 + 1.0);
    float dw4 = dw0 - (k_unskew * 4.0 + 1.0);
    float f0 = (((0.6 - dx0 * dx0) - dy0 * dy0) - dz0 * dz0) - dw0 * dw0;
    float f1 = (((0.6 - dx1 * dx1) - dy1 * dy1) - dz1 * dz1) - dw1 * dw1;
    float f2 = (((0.6 - dx2 * dx2) - dy2 * dy2) - dz2 * dz2) - dw2 * dw2;
    float f3 = (((0.6 - dx3 * dx3) - dy3 * dy3) - dz3 * dz3) - dw3 * dw3;
    float f4 = f0 + (unskew_delta * -2.89442729949951171875 - 0.800000011920928955078125);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0); f3 = max(f3, 0.0); f4 = max(f4, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1); f2 = (f2 * f2) * (f2 * f2);
    f3 = (f3 * f3) * (f3 * f3); f4 = (f4 * f4) * (f4 * f4);
    float g0 = grad_dot_perlin4d(hash_primes4d(seed, x_primed, y_primed, z_primed, w_primed), dx0, dy0, dz0, dw0);
    float g1 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 2 ? (x_primed + PRIME_X) : x_primed, ry > 2 ? (y_primed + PRIME_Y) : y_primed,
        rz > 2 ? (z_primed + PRIME_Z) : z_primed, rw > 2 ? (w_primed + PRIME_W) : w_primed), dx1, dy1, dz1, dw1);
    float g2 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 1 ? (x_primed + PRIME_X) : x_primed, ry > 1 ? (y_primed + PRIME_Y) : y_primed,
        rz > 1 ? (z_primed + PRIME_Z) : z_primed, rw > 1 ? (w_primed + PRIME_W) : w_primed), dx2, dy2, dz2, dw2);
    float g3 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 0 ? (x_primed + PRIME_X) : x_primed, ry > 0 ? (y_primed + PRIME_Y) : y_primed,
        rz > 0 ? (z_primed + PRIME_Z) : z_primed, rw > 0 ? (w_primed + PRIME_W) : w_primed), dx3, dy3, dz3, dw3);
    float g4 = grad_dot_perlin4d(hash_primes4d(seed,
        x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z, w_primed + PRIME_W), dx4, dy4, dz4, dw4);
    float val = g0 * f0 + (g1 * f1 + (g2 * f2 + (g3 * f3 + g4 * f4)));
    return scale_output(val, -0.036730043590068817138671875, 0.036730043590068817138671875);
}
)GLSL" R"GLSL(
float fn_supersimplex2d(vec2 p, int seed) {
    const float k_unskew = -0.211324870586395263671875;
    const float k_rad2 = 0.666666686534881591796875;
    float s = 0.3660254180431365966796875 * (p.x + p.y);
    float x_skewed = p.x + s;
    float y_skewed = p.y + s;
    float x_base = floor(x_skewed);
    float y_base = floor(y_skewed);
    float dx_skewed = x_skewed - x_base;
    float dy_skewed = y_skewed - y_base;
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    bool fwd_xy = (dx_skewed + dy_skewed) > 1.0;
    float bound_xy = fwd_xy ? -1.0 : 0.0;
    bool fwd_x = (dx_skewed * -2.0 + dy_skewed) < bound_xy;
    bool fwd_y = (dy_skewed * -2.0 + dx_skewed) < bound_xy;
    float unskew_d = k_unskew * (dx_skewed + dy_skewed);
    float dx_base = dx_skewed + unskew_d;
    float dy_base = dy_skewed + unskew_d;
    float g0 = grad_dot_perlin2d(hash_primes2d(seed, x_primed, y_primed), dx_base, dy_base);
    float fb0 = (k_rad2 - dy_base * dy_base) - dx_base * dx_base;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    float g1 = grad_dot_perlin2d(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y),
                                 dx_base - (2.0 * k_unskew + 1.0), dy_base - (2.0 * k_unskew + 1.0));
    float fb1 = unskew_d * -3.1547005176544189453125 + (fb0 - k_rad2);
    value += ((fb1 * fb1) * (fb1 * fb1)) * g1;
    float xy_delta = fwd_xy ? (k_unskew + 1.0) : (-k_unskew);
    dx_base -= xy_delta;
    dy_base -= xy_delta;
    int xp2_tmp = fwd_x ? (x_primed + (PRIME_X * 2)) : x_primed;
    int xp2 = (!fwd_xy) ? (xp2_tmp - PRIME_X) : xp2_tmp;
    int yp2 = fwd_xy ? (y_primed + PRIME_Y) : y_primed;
    float dx2 = dx_base - (fwd_x ? (1.0 + 2.0 * k_unskew) : -1.0);
    float dy2 = fwd_x ? (dy_base - 2.0 * k_unskew) : dy_base;
    float g2 = grad_dot_perlin2d(hash_primes2d(seed, xp2, yp2), dx2, dy2);
    float fb2 = max((k_rad2 - dy2 * dy2) - dx2 * dx2, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;
    int xp3 = fwd_xy ? (x_primed + PRIME_X) : x_primed;
    int yp3_tmp = fwd_y ? (y_primed + (PRIME_Y * 2)) : y_primed;
    int yp3 = (!fwd_xy) ? (yp3_tmp - PRIME_Y) : yp3_tmp;
    float dx3 = fwd_y ? (dx_base - 2.0 * k_unskew) : dx_base;
    float dy3 = dy_base - (fwd_y ? (1.0 + 2.0 * k_unskew) : -1.0);
    float g3 = grad_dot_perlin2d(hash_primes2d(seed, xp3, yp3), dx3, dy3);
    float fb3 = max((k_rad2 - dy3 * dy3) - dx3 * dx3, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;
    return scale_output(value, -0.14084912836551666259765625, 0.14084912836551666259765625);
}
float fn_supersimplex3d(vec3 p, int seed) {
    const float k_twice_unskew = -0.25;
    float s = 0.3333333432674407958984375 * (p.x + p.y + p.z);
    float x_base = floor(p.x + s);
    float y_base = floor(p.y + s);
    float z_base = floor(p.z + s);
    float dx_skewed = (p.x + s) - x_base;
    float dy_skewed = (p.y + s) - y_base;
    float dz_skewed = (p.z + s) - z_base;
    float tu0 = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed);
    float xn0 = dx_skewed + tu0;
    float yn0 = dy_skewed + tu0;
    float zn0 = dz_skewed + tu0;
    float xyzn0 = -tu0;
    float max_score = 0.375;
    int move_bits = (xyzn0 > max_score) ? -1 : 0;
    max_score = max(max_score, xyzn0);
    if (xn0 > max_score) { move_bits = 1; max_score = xn0; }
    if (yn0 > max_score) { move_bits = 2; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 4; max_score = zn0; }
    max_score += 0.125 - xyzn0;
    if (-zn0 > max_score) { move_bits = 3; max_score = -zn0; }
    if (-yn0 > max_score) { move_bits = 5; max_score = -yn0; }
    if (-xn0 > max_score) { move_bits = 6; max_score = -xn0; }
    if ((move_bits & 1) != 0) { x_base += 1.0; dx_skewed -= 1.0; }
    if ((move_bits & 2) != 0) { y_base += 1.0; dy_skewed -= 1.0; }
    if ((move_bits & 4) != 0) { z_base += 1.0; dz_skewed -= 1.0; }
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    int z_primed = int(z_base) * PRIME_Z;
    float sk_sum = dx_skewed + dy_skewed + dz_skewed;
    float tu = k_twice_unskew * sk_sum;
    float xn = dx_skewed + tu;
    float yn = dy_skewed + tu;
    float zn = dz_skewed + tu;
    float xyzn = -tu;
    float unskew_d = -0.5 * sk_sum;
    float dx_b = dx_skewed + unskew_d;
    float dy_b = dy_skewed + unskew_d;
    float dz_b = dz_skewed + unskew_d;
    float coord_sum = -0.5 * sk_sum;
    float g0 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed), dx_b, dy_b, dz_b);
    float fb0 = (((0.75 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) * 0.5;
    float stem_a = fb0 - 0.375;
    float stem_b = fb0 - 0.5;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    bool sm1 = xyzn < 0.0;
    float off1 = sm1 ? 0.5 : -0.5;
    float g1 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm1 ? -PRIME_X : PRIME_X),
        y_primed + (sm1 ? -PRIME_Y : PRIME_Y), z_primed + (sm1 ? -PRIME_Z : PRIME_Z)),
        dx_b - off1, dy_b - off1, dz_b - off1);
    float fb1 = max(off1 * coord_sum + stem_a, 0.0);
    value += ((fb1 * fb1) * (fb1 * fb1)) * g1;
    bool sm2 = xyzn < zn;
    float off2 = sm2 ? 1.0 : -1.0;
    float g2 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm2 ? -PRIME_X : PRIME_X), y_primed + (sm2 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b, dy_b, dz_b - off2);
    float fb2 = min((sm2 ? -dz_b : dz_b) - stem_b, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;
    bool sm3 = xyzn < yn;
    float off3 = sm3 ? 1.0 : -1.0;
    float g3 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm3 ? -PRIME_X : PRIME_X), y_primed, z_primed + (sm3 ? -PRIME_Z : PRIME_Z)),
        dx_b, dy_b - off3, dz_b);
    float fb3 = min((sm3 ? -dy_b : dy_b) - stem_b, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;
    bool sm4 = xyzn < xn;
    float off4 = sm4 ? 1.0 : -1.0;
    float g4 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed + (sm4 ? -PRIME_Y : PRIME_Y), z_primed + (sm4 ? -PRIME_Z : PRIME_Z)),
        dx_b - off4, dy_b, dz_b);
    float fb4 = min((sm4 ? -dx_b : dx_b) - stem_b, 0.0);
    value += ((fb4 * fb4) * (fb4 * fb4)) * g4;
    bool sm5 = xn < 0.0;
    float off5 = sm5 ? 0.5 : -0.5;
    float g5 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm5 ? -PRIME_X : PRIME_X), y_primed, z_primed),
        dx_b + off5, dy_b - off5, dz_b - off5);
    float fb5 = max((off5 * coord_sum + stem_a) + (sm5 ? -dx_b : dx_b), 0.0);
    value += ((fb5 * fb5) * (fb5 * fb5)) * g5;
    bool sm6 = yn < 0.0;
    float off6 = sm6 ? 0.5 : -0.5;
    float g6 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed + (sm6 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b - off6, dy_b + off6, dz_b - off6);
    float fb6 = max((off6 * coord_sum + stem_a) + (sm6 ? -dy_b : dy_b), 0.0);
    value += ((fb6 * fb6) * (fb6 * fb6)) * g6;
    bool sm7 = zn < 0.0;
    float off7 = sm7 ? 0.5 : -0.5;
    float g7 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed + (sm7 ? -PRIME_Z : PRIME_Z)),
        dx_b - off7, dy_b - off7, dz_b + off7);
    float fb7 = max((off7 * coord_sum + stem_a) + (sm7 ? -dz_b : dz_b), 0.0);
    value += ((fb7 * fb7) * (fb7 * fb7)) * g7;
    return scale_output(value, -0.0069091119803488254547119140625, 0.0069091119803488254547119140625);
}
)GLSL" R"GLSL(
float supersimplex4d_v3(int seed, bool sm, int xp, int yp, int zp, int wp,
                        float dx_b, float dy_b, float dz_b, float dw_b,
                        float off_x, float off_y, float off_z, float off_w,
                        float off_mul, float coord_sum, float stem, float sub_axis) {
    float sgn = sm ? -1.0 : 1.0;
    float g = grad_dot_perlin4d(hash_primes4d(seed, xp, yp, zp, wp),
        dx_b - off_x * sgn, dy_b - off_y * sgn, dz_b - off_z * sgn, dw_b - off_w * sgn);
    float fb = max(((off_mul * sgn) * coord_sum + stem) + sub_axis * sgn, 0.0);
    return ((fb * fb) * (fb * fb)) * g;
}
float fn_supersimplex4d(vec4 p, int seed) {
    const float k_unskew = -0.13819660112501051517954131656344;
    const float k_twice_unskew = -0.2;
    float s = 0.30901699437494742410229341718282 * (p.x + p.y + p.z + p.w);
    vec4 base = floor(p + s);
    vec4 dsk = (p + s) - base;
    float x_base = base.x, y_base = base.y, z_base = base.z, w_base = base.w;
    float dx_skewed = dsk.x, dy_skewed = dsk.y, dz_skewed = dsk.z, dw_skewed = dsk.w;
    float tu0 = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed + dw_skewed);
    float xn0 = dx_skewed + tu0;
    float yn0 = dy_skewed + tu0;
    float zn0 = dz_skewed + tu0;
    float wn0 = dw_skewed + tu0;
    float xyzwn0 = -tu0;
    float max_score = 0.6 - xyzwn0;
    int move_bits = (0.2 > max_score) ? -1 : 0;
    max_score = max(max_score, 0.2);
    if (-wn0 > max_score) { move_bits = 7; max_score = -wn0; }
    if (-zn0 > max_score) { move_bits = 11; max_score = -zn0; }
    if (-yn0 > max_score) { move_bits = 13; max_score = -yn0; }
    if (-xn0 > max_score) { move_bits = 14; max_score = -xn0; }
    max_score += xyzwn0 - 0.2;
    if (xn0 > max_score) { move_bits = 1; max_score = xn0; }
    if (yn0 > max_score) { move_bits = 2; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 4; max_score = zn0; }
    if (wn0 > max_score) { move_bits = 8; max_score = wn0; }
    max_score += 0.2 - xn0;
    if (yn0 > max_score) { move_bits = 3; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 5; max_score = zn0; }
    if (wn0 > max_score) { move_bits = 9; max_score = wn0; }
    max_score += xn0;
    if (yn0 + zn0 > max_score) { move_bits = 6; max_score = yn0 + zn0; }
    max_score -= wn0;
    if (yn0 > max_score) { move_bits = 10; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 12; max_score = zn0; }
    if ((move_bits & 1) != 0) { x_base += 1.0; dx_skewed -= 1.0; }
    if ((move_bits & 2) != 0) { y_base += 1.0; dy_skewed -= 1.0; }
    if ((move_bits & 4) != 0) { z_base += 1.0; dz_skewed -= 1.0; }
    if ((move_bits & 8) != 0) { w_base += 1.0; dw_skewed -= 1.0; }
    int x_p = int(x_base) * PRIME_X;
    int y_p = int(y_base) * PRIME_Y;
    int z_p = int(z_base) * PRIME_Z;
    int w_p = int(w_base) * PRIME_W;
    float sk_sum = dx_skewed + dy_skewed + dz_skewed + dw_skewed;
    float tu = k_twice_unskew * sk_sum;
    float xn = dx_skewed + tu;
    float yn = dy_skewed + tu;
    float zn = dz_skewed + tu;
    float wn = dw_skewed + tu;
    float xyzwn = -tu;
    float unskew_d = k_unskew * sk_sum;
    float dx_b = dx_skewed + unskew_d;
    float dy_b = dy_skewed + unskew_d;
    float dz_b = dz_skewed + unskew_d;
    float dw_b = dw_skewed + unskew_d;
    float coord_sum = (1.0 + 4.0 * k_unskew) * sk_sum;
    float g0 = grad_dot_perlin4d(hash_primes4d(seed, x_p, y_p, z_p, w_p), dx_b, dy_b, dz_b, dw_b);
    float fb0 = ((((0.8 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) - dw_b * dw_b) * 0.5;
    float stem_a = fb0 - 0.4;
    float stem_b = fb0 - 0.6;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    float u1_0 = k_unskew, u1_1 = k_unskew + 1.0;
    float u2_0 = 2.0 * k_unskew, u2_1 = 2.0 * k_unskew + 1.0;
    float u3_0 = 3.0 * k_unskew, u3_1 = 3.0 * k_unskew + 1.0;
    float u4_1 = 4.0 * k_unskew + 1.0;
    bool sm1 = xyzwn < 0.0;
    value += supersimplex4d_v3(seed, sm1, x_p + (sm1 ? -PRIME_X : PRIME_X), y_p + (sm1 ? -PRIME_Y : PRIME_Y), z_p + (sm1 ? -PRIME_Z : PRIME_Z), w_p + (sm1 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u4_1, u4_1, u4_1, u4_1, u4_1, coord_sum, stem_a, 0.0);
    bool sm2 = xyzwn < wn;
    value += supersimplex4d_v3(seed, sm2, x_p + (sm2 ? -PRIME_X : PRIME_X), y_p + (sm2 ? -PRIME_Y : PRIME_Y), z_p + (sm2 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_1, u3_0, u3_1, coord_sum, stem_b, -dw_b);
    bool sm3 = xyzwn < zn;
    value += supersimplex4d_v3(seed, sm3, x_p + (sm3 ? -PRIME_X : PRIME_X), y_p + (sm3 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm3 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_0, u3_1, u3_1, coord_sum, stem_b, -dz_b);
    bool sm4 = xyzwn < yn;
    value += supersimplex4d_v3(seed, sm4, x_p + (sm4 ? -PRIME_X : PRIME_X), y_p, z_p + (sm4 ? -PRIME_Z : PRIME_Z), w_p + (sm4 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_0, u3_1, u3_1, u3_1, coord_sum, stem_b, -dy_b);
    bool sm5 = xyzwn < xn;
    value += supersimplex4d_v3(seed, sm5, x_p, y_p + (sm5 ? -PRIME_Y : PRIME_Y), z_p + (sm5 ? -PRIME_Z : PRIME_Z), w_p + (sm5 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_0, u3_1, u3_1, u3_1, u3_1, coord_sum, stem_b, -dx_b);
    bool sm6 = xn < 0.0;
    value += supersimplex4d_v3(seed, sm6, x_p + (sm6 ? -PRIME_X : PRIME_X), y_p, z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_1, u1_0, u1_0, u1_0, u1_0, coord_sum, stem_a, dx_b);
    bool sm7 = xn < -yn;
    value += supersimplex4d_v3(seed, sm7, x_p + (sm7 ? -PRIME_X : PRIME_X), y_p + (sm7 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_1, u2_0, u2_0, u2_0, coord_sum, stem_b, dx_b + dy_b);
    bool sm8 = xn < -zn;
    value += supersimplex4d_v3(seed, sm8, x_p + (sm8 ? -PRIME_X : PRIME_X), y_p, z_p + (sm8 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_1, u2_0, u2_0, coord_sum, stem_b, dx_b + dz_b);
    bool sm9 = xn < -wn;
    value += supersimplex4d_v3(seed, sm9, x_p + (sm9 ? -PRIME_X : PRIME_X), y_p, z_p, w_p + (sm9 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_0, u2_1, u2_0, coord_sum, stem_b, dx_b + dw_b);
    bool sm10 = yn < 0.0;
    value += supersimplex4d_v3(seed, sm10, x_p, y_p + (sm10 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_1, u1_0, u1_0, u1_0, coord_sum, stem_a, dy_b);
    bool sm11 = yn < -zn;
    value += supersimplex4d_v3(seed, sm11, x_p, y_p + (sm11 ? -PRIME_Y : PRIME_Y), z_p + (sm11 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_1, u2_0, u2_0, coord_sum, stem_b, dy_b + dz_b);
    bool sm12 = yn < -wn;
    value += supersimplex4d_v3(seed, sm12, x_p, y_p + (sm12 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm12 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_0, u2_1, u2_0, coord_sum, stem_b, dy_b + dw_b);
    bool sm13 = zn < 0.0;
    value += supersimplex4d_v3(seed, sm13, x_p, y_p, z_p + (sm13 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_1, u1_0, u1_0, coord_sum, stem_a, dz_b);
    bool sm14 = zn < -wn;
    value += supersimplex4d_v3(seed, sm14, x_p, y_p, z_p + (sm14 ? -PRIME_Z : PRIME_Z), w_p + (sm14 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_0, u2_1, u2_1, u2_0, coord_sum, stem_b, dz_b + dw_b);
    bool sm15 = wn < 0.0;
    value += supersimplex4d_v3(seed, sm15, x_p, y_p, z_p, w_p + (sm15 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_0, u1_1, u1_0, coord_sum, stem_a, dw_b);
    return scale_output(value, -0.011310013942420482635498046875, 0.011310013942420482635498046875);
}
)GLSL" R"GLSL(
int white_coord(float x, int prime) {
    int bits = floatBitsToInt(x);
    return (bits ^ (bits >> 16)) * prime;
}
float fn_white2d(vec2 p, int seed) {
    return scale_output(get_value2d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y)), -2147483648.0, 2147483648.0);
}
float fn_white3d(vec3 p, int seed) {
    return scale_output(get_value3d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y), white_coord(p.z, PRIME_Z)),
                        -2147483648.0, 2147483648.0);
}
float fn_white4d(vec4 p, int seed) {
    return scale_output(get_value4d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y), white_coord(p.z, PRIME_Z),
                                    white_coord(p.w, PRIME_W)), -2147483648.0, 2147483648.0);
}

float fn_ping_pong(float val, float strength) {
    float t0 = val * strength;
    float t = t0 - floor(t0 * 0.5) * 2.0;
    return (t < 1.0) ? t : (2.0 - t);
}

// FastNoise2's cellular nodes with the Euclidean-squared distance (the
// bench's), a 3^D neighbourhood of jittered points: the nearest point's hash
// (value), the nearest distance (distance, rooted), or Perlin at the nearest
// point (lookup, seed + 1).
const float kJitter2 = 0.437015950679779052734375;
const float kJitter3 = 0.3961435258388519287109375;
const float kJitter4 = 0.3660254180431365966796875;
void cell_search2d(vec2 p, int seed, out float d0, out int best_hash, out vec2 best_p) {
    vec2 r = roundEven(p);
    int xc = (int(r.x) - 1) * PRIME_X;
    float xcf = (r.x - 1.0) - p.x;
    d0 = 1.0e10; best_hash = 0; best_p = vec2(0.0);
    for (int ix = 0; ix < 3; ++ix) {
        int yc = (int(r.y) - 1) * PRIME_Y;
        float ycf = (r.y - 1.0) - p.y;
        for (int iy = 0; iy < 3; ++iy) {
            int h = (seed ^ xc ^ yc) * HASH_A;
            float xd = float(h & 0x7ff) - 1023.5;
            float yd = float((h >> 21) & 0x7ff) - 1023.5;
            float inv_mag = kJitter2 / sqrt(xd * xd + yd * yd);
            float px_d = xd * inv_mag + xcf;
            float py_d = yd * inv_mag + ycf;
            float d = px_d * px_d + py_d * py_d;
            if (d < d0) { d0 = d; best_hash = h; best_p = vec2(px_d, py_d) + p; }
            yc += PRIME_Y; ycf += 1.0;
        }
        xc += PRIME_X; xcf += 1.0;
    }
}
void cell_search3d(vec3 p, int seed, out float d0, out int best_hash, out vec3 best_p) {
    vec3 r = roundEven(p);
    int xc = (int(r.x) - 1) * PRIME_X;
    float xcf = (r.x - 1.0) - p.x;
    d0 = 1.0e10; best_hash = 0; best_p = vec3(0.0);
    for (int ix = 0; ix < 3; ++ix) {
        int yc = (int(r.y) - 1) * PRIME_Y;
        float ycf = (r.y - 1.0) - p.y;
        for (int iy = 0; iy < 3; ++iy) {
            int zc = (int(r.z) - 1) * PRIME_Z;
            float zcf = (r.z - 1.0) - p.z;
            for (int iz = 0; iz < 3; ++iz) {
                int h = (seed ^ xc ^ yc ^ zc) * HASH_A;
                float xd = float(h & 0x3ff) - 511.5;
                float yd = float((h >> 11) & 0x3ff) - 511.5;
                float zd = float((h >> 22) & 0x3ff) - 511.5;
                float inv_mag = kJitter3 / sqrt(xd * xd + yd * yd + zd * zd);
                vec3 pd = vec3(xd, yd, zd) * inv_mag + vec3(xcf, ycf, zcf);
                float d = pd.x * pd.x + pd.y * pd.y + pd.z * pd.z;
                if (d < d0) { d0 = d; best_hash = h; best_p = pd + p; }
                zc += PRIME_Z; zcf += 1.0;
            }
            yc += PRIME_Y; ycf += 1.0;
        }
        xc += PRIME_X; xcf += 1.0;
    }
}
void cell_search4d(vec4 p, int seed, out float d0, out int best_hash, out vec4 best_p) {
    vec4 r = roundEven(p);
    ivec4 cb = (ivec4(r) - 1) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 cf = (r - 1.0) - p;
    d0 = 1.0e10; best_hash = 0; best_p = vec4(0.0);
    for (int ix = 0; ix < 3; ++ix)
    for (int iy = 0; iy < 3; ++iy)
    for (int iz = 0; iz < 3; ++iz)
    for (int iw = 0; iw < 3; ++iw) {
        int h = (seed ^ (cb.x + ix * PRIME_X) ^ (cb.y + iy * PRIME_Y) ^ (cb.z + iz * PRIME_Z) ^ (cb.w + iw * PRIME_W)) * HASH_A;
        vec4 dv = vec4(float(h & 0xff), float((h >> 8) & 0xff), float((h >> 16) & 0xff), float((h >> 24) & 0xff)) - 127.5;
        float inv_mag = kJitter4 / sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z + dv.w * dv.w);
        vec4 pd = dv * inv_mag + (cf + vec4(float(ix), float(iy), float(iz), float(iw)));
        float d = pd.x * pd.x + pd.y * pd.y + pd.z * pd.z + pd.w * pd.w;
        if (d < d0) { d0 = d; best_hash = h; best_p = pd + p; }
    }
}
// The second-nearest distance is not tracked: the bench's distance rows
// return d0 (FastNoise2's Index0).
#define FN_CELLULAR(VT, D, JIT) \
float fn_cell_value##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return scale_output(float(h), -2147483648.0, 2147483648.0); \
} \
float fn_cell_distance##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return scale_output(sqrt(d0), 0.0, (1.0 + JIT) * (1.0 + JIT)); \
} \
float fn_cell_lookup##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return fn_perlin(bp, seed + 1); \
}
FN_CELLULAR(vec2, 2, kJitter2)
FN_CELLULAR(vec3, 3, kJitter3)
FN_CELLULAR(vec4, 4, kJitter4)
)GLSL" R"GLSL(
// FastNoise2's gradient domain warp (hermite-weighted lattice hashes, its
// bits as the offset), alone and as the progressive / independent fractals.
vec2 warp_delta(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    float xs1 = interp_hermite(p.x - s.x), ys1 = interp_hermite(p.y - s.y);
    float xs0 = 1.0 - xs1, ys0 = 1.0 - ys1;
    int h00 = (seed ^ i0.x ^ i0.y) * HASH_A;
    int h10 = (seed ^ i1.x ^ i0.y) * HASH_A;
    int h01 = (seed ^ i0.x ^ i1.y) * HASH_A;
    int h11 = (seed ^ i1.x ^ i1.y) * HASH_A;
    float c00 = xs0 * ys0, c10 = xs1 * ys0, c01 = xs0 * ys1, c11 = xs1 * ys1;
    float xw = c00 * float(h00 & 0xffff) + c10 * float(h10 & 0xffff) + c01 * float(h01 & 0xffff) + c11 * float(h11 & 0xffff);
    float yw = c00 * float((h00 >> 16) & 0xffff) + c10 * float((h10 >> 16) & 0xffff) + c01 * float((h01 >> 16) & 0xffff) + c11 * float((h11 >> 16) & 0xffff);
    float norm = 1.0 / (65535.0 / 2.0);
    return vec2(xw * norm - 1.0, yw * norm - 1.0);
}
vec3 warp_delta(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f1 = vec3(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z));
    vec3 f0 = 1.0 - f1;
    vec3 acc = vec3(0.0);
    for (int iz = 0; iz < 2; ++iz)
    for (int iy = 0; iy < 2; ++iy)
    for (int ix = 0; ix < 2; ++ix) {
        float c = (ix == 0 ? f0.x : f1.x) * (iy == 0 ? f0.y : f1.y) * (iz == 0 ? f0.z : f1.z);
        int h = (seed ^ (ix == 0 ? i0.x : i1.x) ^ (iy == 0 ? i0.y : i1.y) ^ (iz == 0 ? i0.z : i1.z)) * HASH_A;
        acc += c * vec3(float(h & 0x3ff), float((h >> 11) & 0x3ff), float((h >> 22) & 0x3ff));
    }
    return acc * (1.0 / (1023.0 / 2.0)) - 1.0;
}
vec4 warp_delta(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f = vec4(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z), interp_hermite(p.w - s.w));
    vec4 acc = vec4(0.0);
    for (int iw = 0; iw < 2; ++iw) {
        float wt = (iw == 0) ? (1.0 - f.w) : f.w;
        for (int iz = 0; iz < 2; ++iz) {
            float zt = wt * ((iz == 0) ? (1.0 - f.z) : f.z);
            for (int iy = 0; iy < 2; ++iy) {
                float yt = zt * ((iy == 0) ? (1.0 - f.y) : f.y);
                for (int ix = 0; ix < 2; ++ix) {
                    float xt = yt * ((ix == 0) ? (1.0 - f.x) : f.x);
                    int h = (seed ^ (i0.x + ix * PRIME_X) ^ (i0.y + iy * PRIME_Y) ^ (i0.z + iz * PRIME_Z) ^ (i0.w + iw * PRIME_W)) * HASH_A;
                    acc += xt * vec4(float(h & 0xff), float((h >> 8) & 0xff), float((h >> 16) & 0xff), float((h >> 24) & 0xff));
                }
            }
        }
    }
    return acc * (1.0 / (255.0 / 2.0)) - 1.0;
}
#define FN_WARPS(VT) \
VT fn_warp_grad(VT p, int seed, float amp, float freq) { return p + warp_delta(p * freq, seed) * amp; } \
VT fn_warp_fractal(VT p, int seed, int octaves, float lac, float gain, float ws, float amp, float freq, bool progressive) { \
    VT wp = p; \
    VT w = warp_delta(p * freq, seed); \
    wp = wp + w * amp; \
    float strength = dot(w, w); \
    for (int i = 1; i < octaves; ++i) { \
        strength = sqrt(strength); \
        seed += 1; \
        freq *= lac; \
        amp *= 1.0 + ws * ((1.0 - strength) - 1.0); \
        amp *= gain; \
        w = warp_delta((progressive ? wp : p) * freq, seed); \
        wp = wp + w * amp; \
        strength = dot(w, w); \
    } \
    return wp; \
}
FN_WARPS(vec2)
FN_WARPS(vec3)
FN_WARPS(vec4)
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
// The planet's flow state: per cell the ground h, drainage a, the drainage
// handed on a unit of weight z, the steepest slope down w, the water level wl.
struct FlowState {
    std::vector<float> h, a, z, w, wl;
    explicit FlowState(size_t n) : h(n), a(n), z(n), w(n), wl(n) {}
};

constexpr int kPlanetFlowIters = 8;
constexpr float kPlanetFlowCell = 19500.0f;   // the global grid's cells, metres

// The start procInit leaves: the ground in metres, the water high over the
// land (it drains from the coasts), nothing routed yet.
FlowState planet_flow_start(const std::vector<float>& field) {
    FlowState s(field.size());
    for (size_t i = 0; i < field.size(); ++i) {
        s.h[i] = field[i] * 100.0f;
        s.a[i] = 1.0f;
        s.wl[i] = s.h[i] < 0.0f ? s.h[i] : 1e5f;
    }
    return s;
}

// k_planet_flow in C++, the same operations in the same order.
void planet_flow_ref(const FlowState& in, FlowState& out, int W, int H, float cell) {
    const int nbx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int nby[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float nd[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f, 1.0f, 1.41421356f, 1.0f, 1.41421356f};
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t c = static_cast<size_t>(y) * W + x;
            float h = in.h[c];
            const float w = in.wl[c];
            const bool sea = h < 0.0f;
            float wn[8], a = 1.0f, rest = 1e30f, mean4 = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const size_t n = static_cast<size_t>(std::clamp(y + nby[k], 0, H - 1)) * W + std::clamp(x + nbx[k], 0, W - 1);
                const float dist = nd[k] * cell;
                wn[k] = in.wl[n];
                if (wn[k] > w && in.w[n] > 0.0f) {
                    float t = std::min((wn[k] - w) / dist / in.w[n], 1.0f);
                    a += in.z[n] * ((t * t) * (t * t));
                }
                rest = std::min(rest, wn[k] + 2e-4f * dist);
                if (nd[k] == 1.0f) mean4 += 0.25f * in.h[n];
            }
            if (!sea) {
                if (w - h < 1.0f) {
                    float drop = 0.0f;
                    for (int k = 0; k < 8; ++k) drop = std::max(drop, (h - wn[k]) / nd[k]);
                    float e = 2.5e-4f * std::sqrt(a);
                    float hc = (h - drop * e / (1.0f + e)) + 0.0015f * (mean4 - h);
                    h = std::max(hc, std::min(h, 1.0f));
                } else {
                    h = h + 0.01f * (w - h);
                }
            }
            const float wl = sea ? h : std::max(h, rest);
            float smax = 0.0f, s = 0.0f;
            for (int k = 0; k < 8; ++k) smax = std::max(smax, (wl - wn[k]) / (nd[k] * cell));
            if (!sea && smax > 0.0f) {
                for (int k = 0; k < 8; ++k) {
                    float down = (wl - wn[k]) / (nd[k] * cell);
                    if (down > 0.0f) { float t = down / smax; s += (t * t) * (t * t); }
                }
            }
            out.h[c] = h;
            out.a[c] = a;
            out.z[c] = s > 0.0f ? a / s : 0.0f;
            out.w[c] = sea ? 0.0f : smax;
            out.wl[c] = wl;
        }
    }
}

const char* kGlslPlanetFlow = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC {
    uint64_t h; uint64_t z; uint64_t w; uint64_t wl;
    uint64_t h_out; uint64_t a_out; uint64_t z_out; uint64_t w_out; uint64_t wl_out;
    float cell; int width; int height;
} pc;
const ivec2 NB[8] = ivec2[8](ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0),
                             ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
const float ND[8] = float[8](1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356);
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= pc.width || p.y >= pc.height) return;
    int c = p.y * pc.width + p.x;
    float L = pc.cell;
    float h = FloatBuf(pc.h).v[c];
    float W = FloatBuf(pc.wl).v[c];
    bool sea = h < 0.0;
    float Wn[8];
    float A = 1.0, rest = 1e30, mean4 = 0.0;
    for (int k = 0; k < 8; ++k) {
        ivec2 q = clamp(p + NB[k], ivec2(0), ivec2(pc.width - 1, pc.height - 1));
        int n = q.y * pc.width + q.x;
        float dist = ND[k] * L;
        Wn[k] = FloatBuf(pc.wl).v[n];
        float nw = FloatBuf(pc.w).v[n];
        if (Wn[k] > W && nw > 0.0) {
            float t = min((Wn[k] - W) / dist / nw, 1.0);
            A += FloatBuf(pc.z).v[n] * ((t * t) * (t * t));
        }
        rest = min(rest, Wn[k] + 2e-4 * dist);
        if (ND[k] == 1.0) mean4 += 0.25 * FloatBuf(pc.h).v[n];
    }
    if (!sea) {
        if (W - h < 1.0) {
            float drop = 0.0;
            for (int k = 0; k < 8; ++k) drop = max(drop, (h - Wn[k]) / ND[k]);
            float e = 2.5e-4 * sqrt(A);
            float hc = (h - drop * e / (1.0 + e)) + 0.0015 * (mean4 - h);
            h = max(hc, min(h, 1.0));
        } else {
            h += 0.01 * (W - h);
        }
    }
    W = sea ? h : max(h, rest);
    float smax = 0.0, S = 0.0;
    for (int k = 0; k < 8; ++k) smax = max(smax, (W - Wn[k]) / (ND[k] * L));
    if (!sea && smax > 0.0)
        for (int k = 0; k < 8; ++k) {
            float down = (W - Wn[k]) / (ND[k] * L);
            if (down > 0.0) { float t = down / smax; S += (t * t) * (t * t); }
        }
    FloatBuf(pc.h_out).v[c] = h;
    FloatBuf(pc.a_out).v[c] = A;
    FloatBuf(pc.z_out).v[c] = S > 0.0 ? A / S : 0.0;
    FloatBuf(pc.w_out).v[c] = sea ? 0.0 : smax;
    FloatBuf(pc.wl_out).v[c] = W;
}
)GLSL";

BenchRow run_planet_flow_row(const CompileResult& res, uint32_t W, uint32_t H, const std::vector<float>& field,
                             uint32_t TW, uint32_t TH, const std::vector<float>& field_time) {
    BenchRow row;
    row.tier = "Stencil";
    row.name = "PlanetFlow_2D";
    row.dim = 2;
    const size_t N = static_cast<size_t>(W) * H;
    row.points = N;
    const GridShape shape(W, H);

    FlowState ref = planet_flow_start(field), ref_next(N);
    double ref_ms = best_cpu_ms([&] {
        FlowState a = planet_flow_start(field), b(N);
        for (int it = 0; it < kPlanetFlowIters; ++it) {
            planet_flow_ref(a, b, W, H, kPlanetFlowCell);
            std::swap(a, b);
        }
        ref = std::move(a);
    });
    row.cpu_ref_mpts = mpts(N * kPlanetFlowIters, ref_ms);

    const auto* k = res.find_kernel("k_planet_flow");
    auto args = [&](FlowState& in, FlowState& out) {
        return std::vector<KernelArg>{KernelArg::store(in.h), KernelArg::store(in.z), KernelArg::store(in.w),
                                      KernelArg::store(in.wl), KernelArg::store(out.h), KernelArg::store(out.a),
                                      KernelArg::store(out.z), KernelArg::store(out.w), KernelArg::store(out.wl),
                                      KernelArg(kPlanetFlowCell)};
    };
    FlowState cpu = planet_flow_start(field);
    double bk_cpu_ms = best_cpu_ms([&] {
        FlowState a = planet_flow_start(field), b(N);
        for (int it = 0; it < kPlanetFlowIters; ++it) {
            k->dispatch_cpu(shape, args(a, b));
            std::swap(a, b);
        }
        cpu = std::move(a);
    });
    row.cpu_bk_mpts = mpts(N * kPlanetFlowIters, bk_cpu_ms);

    FlowState gpu = planet_flow_start(field), gpu_b(N);
    for (int it = 0; it < kPlanetFlowIters; ++it) {
        k->dispatch_gpu(shape, args(gpu, gpu_b));
        std::swap(gpu, gpu_b);
    }

    // Heights to a millimetre (they are hundreds of metres), the drainage
    // to 1e-4 of a cell, the water's level as the heights.
    auto check = [&](const FlowState& got, bool& pass, float& max_abs, uint32_t& max_ulp) {
        pass = true;
        max_abs = 0.0f;
        max_ulp = 0;
        for (const auto& [g, r, tol] : {std::tuple{&got.h, &ref.h, 1e-3f}, std::tuple{&got.a, &ref.a, 1e-4f},
                                        std::tuple{&got.wl, &ref.wl, 1e-3f}}) {
            auto s = compare_buffers(*g, *r, tol);
            pass = pass && s.pass;
            max_abs = std::max(max_abs, s.max_abs);
            max_ulp = std::max(max_ulp, s.max_ulp);
        }
    };
    check(cpu, row.cpu_pass, row.cpu_max_abs, row.cpu_max_ulp);
    check(gpu, row.gpu_pass, row.gpu_max_abs, row.gpu_max_ulp);

    // The GPU timed an iteration on the TW x TH field, from its start.
    const size_t NT = static_cast<size_t>(TW) * TH;
    FlowState t0 = planet_flow_start(field_time);
    std::vector<brass::gpu::VulkanBuffer> bufs;
    for (int i = 0; i < 9; ++i) bufs.push_back(brass::gpu::VulkanBuffer::alloc(NT * sizeof(float)));
    bufs[0].upload(t0.h.data(), NT * sizeof(float));
    bufs[1].upload(t0.z.data(), NT * sizeof(float));
    bufs[2].upload(t0.w.data(), NT * sizeof(float));
    bufs[3].upload(t0.wl.data(), NT * sizeof(float));
    std::vector<brass::gpu::VulkanArg> vk_args;
    for (auto& b : bufs) vk_args.emplace_back(b);
    vk_args.emplace_back(kPlanetFlowCell);
    double bk_gpu_ms = time_bk_gpu_ms(k, GridShape(TW, TH), vk_args);
    row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

    auto spv = compile_glsl_compute(kGlslPlanetFlow, "stencil_planet_flow");
    std::vector<brass::Type> layout(9, brass::Type::ptr());
    layout.push_back(brass::Type::f32());
    layout.push_back(brass::Type::i32());
    layout.push_back(brass::Type::i32());
    auto vmod = load_glsl_vulkan_module(spv, layout, 8, 8, 1);
    vk_args.emplace_back(static_cast<int32_t>(TW));
    vk_args.emplace_back(static_cast<int32_t>(TH));
    double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, vk_args);
    row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));

    const bool pass = row.cpu_pass && row.gpu_pass;
    row.cpu_ratio = pass ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
    row.has_gpu_baseline = true;
    row.gpu_ratio = pass ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
    return row;
}

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

        // The planet's rivers (broworkshop 1fbd228^:demos/planet/terrain.js,
        // flow() as erode ran it on the global grid: no region edge, seedA
        // and ks 0, so its bed and slide clamps do nothing; FLOW_P 4 and M
        // 0.5 as squares and a root). Water level W drains from the sea
        // inward (Planchon-Darboux), drainage A is the cell and what its
        // higher neighbours hand it by slope^4, and land out of the water is
        // cut by implicit stream power and crept by diffusion.
        fn flow_in(wn: f32, nz: f32, nw: f32, w: f32, dist: f32) -> f32 {
            if wn > w && nw > 0.0 {
                let t = min((wn - w) / dist / nw, 1.0);
                return nz * ((t * t) * (t * t));
            }
            return 0.0;
        }
        fn flow_share(w: f32, wn: f32, dist: f32, smax: f32) -> f32 {
            let down = (w - wn) / dist;
            if down > 0.0 {
                let t = down / smax;
                return (t * t) * (t * t);
            }
            return 0.0;
        }

        @kernel @grid(2)
        fn k_planet_flow(g: grid<2>, hs: stencil<f32, 2>, zs: stencil<f32, 2>, ws: stencil<f32, 2>, wl: stencil<f32, 2>,
                         h_out: write_store<f32, 2>, a_out: write_store<f32, 2>, z_out: write_store<f32, 2>,
                         w_out: write_store<f32, 2>, wl_out: write_store<f32, 2>, cell: f32) {
            let x = g.x();
            let y = g.y();
            let fill = 2e-4;
            let k_rate = 2.5e-4;
            let creep = 0.0015;
            let silt = 0.01;
            var h = hs[0, 0];
            let w = wl[0, 0];
            let sea = h < 0.0;
            let d1 = cell;
            let d2 = 1.41421356 * cell;
            let w0 = wl[-1, -1]; let w1 = wl[0, -1]; let w2 = wl[1, -1]; let w3 = wl[-1, 0];
            let w4 = wl[1, 0]; let w5 = wl[-1, 1]; let w6 = wl[0, 1]; let w7 = wl[1, 1];
            var a = 1.0;
            a += flow_in(w0, zs[-1, -1], ws[-1, -1], w, d2);
            a += flow_in(w1, zs[0, -1], ws[0, -1], w, d1);
            a += flow_in(w2, zs[1, -1], ws[1, -1], w, d2);
            a += flow_in(w3, zs[-1, 0], ws[-1, 0], w, d1);
            a += flow_in(w4, zs[1, 0], ws[1, 0], w, d1);
            a += flow_in(w5, zs[-1, 1], ws[-1, 1], w, d2);
            a += flow_in(w6, zs[0, 1], ws[0, 1], w, d1);
            a += flow_in(w7, zs[1, 1], ws[1, 1], w, d2);
            let rest = min(min(min(w0 + fill * d2, w1 + fill * d1), min(w2 + fill * d2, w3 + fill * d1)),
                           min(min(w4 + fill * d1, w5 + fill * d2), min(w6 + fill * d1, w7 + fill * d2)));
            let mean4 = ((0.25 * hs[0, -1] + 0.25 * hs[-1, 0]) + 0.25 * hs[1, 0]) + 0.25 * hs[0, 1];
            if !sea {
                if w - h < 1.0 {
                    let drop = max(max(max(max(0.0, (h - w0) / 1.41421356), h - w1), max((h - w2) / 1.41421356, h - w3)),
                                   max(max(h - w4, (h - w5) / 1.41421356), max(h - w6, (h - w7) / 1.41421356)));
                    let e = k_rate * sqrt(a);
                    let hc = (h - drop * e / (1.0 + e)) + creep * (mean4 - h);
                    h = max(hc, min(h, 1.0));
                } else {
                    h = h + silt * (w - h);
                }
            }
            let wn = sea ? h : max(h, rest);
            let smax = max(max(max(max(0.0, (wn - w0) / d2), (wn - w1) / d1), max((wn - w2) / d2, (wn - w3) / d1)),
                           max(max((wn - w4) / d1, (wn - w5) / d2), max((wn - w6) / d1, (wn - w7) / d2)));
            var s = 0.0;
            if !sea && smax > 0.0 {
                s = flow_share(wn, w0, d2, smax) + flow_share(wn, w1, d1, smax) + flow_share(wn, w2, d2, smax)
                  + flow_share(wn, w3, d1, smax) + flow_share(wn, w4, d1, smax) + flow_share(wn, w5, d2, smax)
                  + flow_share(wn, w6, d1, smax) + flow_share(wn, w7, d2, smax);
            }
            h_out[x, y] = h;
            a_out[x, y] = a;
            z_out[x, y] = s > 0.0 ? a / s : 0.0;
            w_out[x, y] = sea ? 0.0 : smax;
            wl_out[x, y] = wn;
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

    // 6. The planet's rivers (k_planet_flow), kPlanetFlowIters iterations
    // from the sea-drained start, over the field in metres (half of it sea).
    rows.push_back(run_planet_flow_row(res, W, H, h_in, TW, TH, h_time));

    return rows;
}

void write_golden_json(const fs::path& path, const std::vector<BenchRow>& rows,
                       int parity_passed, int parity_total,
                       double geomean_cpu, double geomean_gpu, double geomean_gpu_compute) {
    std::ofstream ofs(path);
    ofs << "{\n";
    ofs << "  \"parity_passed\": " << parity_passed << ",\n";
    ofs << "  \"parity_total\": " << parity_total << ",\n";
    ofs << "  \"geomean_cpu_vs_ref\": " << std::fixed << std::setprecision(4) << geomean_cpu << ",\n";
    ofs << "  \"geomean_gpu_vs_glsl\": " << std::fixed << std::setprecision(4) << geomean_gpu << ",\n";
    ofs << "  \"geomean_gpu_compute_vs_glsl\": " << std::fixed << std::setprecision(4) << geomean_gpu_compute << ",\n";
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
    // One module per (node, dimension): expr's %D is the dimension.
    auto glsl_set = [&](const std::string& expr, const std::string& tag) {
        std::vector<brass::gpu::VulkanModule> mods;
        for (int dim = 2; dim <= 4; ++dim) {
            std::string e = expr;
            for (size_t at; (at = e.find("%D")) != std::string::npos;) e.replace(at, 2, std::to_string(dim));
            mods.push_back(glsl_module(dim, e, tag + std::to_string(dim)));
        }
        return mods;
    };
    auto vmod_simplex = glsl_set("fn_simplex%Dd(p, seed)", "glsl_simplex");
    auto vmod_supersimplex = glsl_set("fn_supersimplex%Dd(p, seed)", "glsl_supersimplex");
    auto vmod_white = glsl_set("fn_white%Dd(p, seed)", "glsl_white");
    auto vmod_pingpong = glsl_set("fn_ping_pong(fn_perlin(p, seed), 2.0)", "glsl_pingpong");
    auto vmod_cell_val = glsl_set("fn_cell_value%Dd(p, seed)", "glsl_cell_val");
    auto vmod_cell_dist = glsl_set("fn_cell_distance%Dd(p, seed)", "glsl_cell_dist");
    auto vmod_cell_lookup = glsl_set("fn_cell_lookup%Dd(p, seed)", "glsl_cell_lookup");
    auto vmod_warp_grad = glsl_set("fn_perlin(fn_warp_grad(p, seed, 1.5, 0.5), seed)", "glsl_warp_grad");
    auto vmod_warp_prog = glsl_set("fn_perlin(fn_warp_fractal(p, seed, 3, 2.0, 0.5, 0.25, 1.5, 0.5, true), seed)", "glsl_warp_prog");
    auto vmod_warp_indep = glsl_set("fn_perlin(fn_warp_fractal(p, seed, 3, 2.0, 0.5, 0.25, 1.5, 0.5, false), seed)", "glsl_warp_indep");

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
        // points against FastNoise2. Not to the kernel's tolerance: the GLSL
        // is not precise, so the driver fuses multiply-adds, and Simplex's
        // small output bound turns that into ~1e-4. A wrong node is off by O(1).
        if (baseline) {
            std::vector<brass::gpu::VulkanArg> bargs(vk_args.begin(), vk_args.end());
            bargs.emplace_back(static_cast<int32_t>(NT));
            double glsl_ms = time_glsl_gpu_ms(*baseline, NT / 64, 1, 1, 64, 1, 1, bargs);
            d_out.download(glsl_out.data(), N * sizeof(float));
            auto bs = compare_buffers(glsl_out, ref, std::max(tol, 1e-3f));
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
        bench_noise("Noise", "Simplex_2D", 2, "k_simplex2d", fn, &vmod_simplex[0], kTol);
        bench_noise("Noise", "Simplex_3D", 3, "k_simplex3d", fn, &vmod_simplex[1], kTol);
        bench_noise("Noise", "Simplex_4D", 4, "k_simplex4d", fn, &vmod_simplex[2], kTol);
    }
    // 4. SuperSimplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::SuperSimplex>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "SuperSimplex_2D", 2, "k_supersimplex2d", fn, &vmod_supersimplex[0], kTol);
        bench_noise("Noise", "SuperSimplex_3D", 3, "k_supersimplex3d", fn, &vmod_supersimplex[1], kTol);
        bench_noise("Noise", "SuperSimplex_4D", 4, "k_supersimplex4d", fn, &vmod_supersimplex[2], kTol);
    }
    // 5. White (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::White>();
        bench_noise("Noise", "White_2D", 2, "k_white2d", fn, &vmod_white[0], kTol);
        bench_noise("Noise", "White_3D", 3, "k_white3d", fn, &vmod_white[1], kTol);
        bench_noise("Noise", "White_4D", 4, "k_white4d", fn, &vmod_white[2], kTol);
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
        bench_noise("Noise", "PingPong_2D", 2, "k_pingpong2d", fn, &vmod_pingpong[0], kTol);
        bench_noise("Noise", "PingPong_3D", 3, "k_pingpong3d", fn, &vmod_pingpong[1], kTol);
        bench_noise("Noise", "PingPong_4D", 4, "k_pingpong4d", fn, &vmod_pingpong[2], kTol);
    }
    // 9. CellularValue (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularValue>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularValue_2D", 2, "k_cell_val2d", fn, &vmod_cell_val[0], kTolRelaxed);
        bench_noise("Noise", "CellularValue_3D", 3, "k_cell_val3d", fn, &vmod_cell_val[1], kTolRelaxed);
        bench_noise("Noise", "CellularValue_4D", 4, "k_cell_val4d", fn, &vmod_cell_val[2], kTolRelaxed);
    }
    // 10. CellularDistance (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularDistance>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularDistance_2D", 2, "k_cell_dist2d", fn, &vmod_cell_dist[0], kTolRelaxed);
        bench_noise("Noise", "CellularDistance_3D", 3, "k_cell_dist3d", fn, &vmod_cell_dist[1], kTolRelaxed);
        bench_noise("Noise", "CellularDistance_4D", 4, "k_cell_dist4d", fn, &vmod_cell_dist[2], kTolRelaxed);
    }
    // 11. CellularLookup (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularLookup>();
        fn->SetScale(1.0f);
        fn->SetLookup(fn_perlin);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularLookup_2D", 2, "k_cell_lookup2d", fn, &vmod_cell_lookup[0], kTolRelaxed);
        bench_noise("Noise", "CellularLookup_3D", 3, "k_cell_lookup3d", fn, &vmod_cell_lookup[1], kTolRelaxed);
        bench_noise("Noise", "CellularLookup_4D", 4, "k_cell_lookup4d", fn, &vmod_cell_lookup[2], kTolRelaxed);
    }
    // 12. DomainWarpGradient (2D, 3D, 4D)
    auto fn_warp_grad = FastNoise::New<FastNoise::DomainWarpGradient>();
    fn_warp_grad->SetSource(fn_perlin);
    fn_warp_grad->SetWarpAmplitude(1.5f);
    fn_warp_grad->SetScale(2.0f);
    bench_noise("Noise", "DomainWarpGrad_2D", 2, "k_warp_grad2d", fn_warp_grad, &vmod_warp_grad[0], kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_3D", 3, "k_warp_grad3d", fn_warp_grad, &vmod_warp_grad[1], kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_4D", 4, "k_warp_grad4d", fn_warp_grad, &vmod_warp_grad[2], kTolRelaxed);

    // 13. DomainWarpFractalProgressive (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalProgressive>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpProg_2D", 2, "k_warp_prog2d", fn, &vmod_warp_prog[0], kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_3D", 3, "k_warp_prog3d", fn, &vmod_warp_prog[1], kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_4D", 4, "k_warp_prog4d", fn, &vmod_warp_prog[2], kTolRelaxed);
    }
    // 14. DomainWarpFractalIndependent (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalIndependent>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpIndep_2D", 2, "k_warp_indep2d", fn, &vmod_warp_indep[0], kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_3D", 3, "k_warp_indep3d", fn, &vmod_warp_indep[1], kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_4D", 4, "k_warp_indep4d", fn, &vmod_warp_indep[2], kTolRelaxed);
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

    // The GPU rows the code decides: a noise row whose GLSL baseline runs
    // under kComputeBound of White's rate in its dimension (White hashes its
    // inputs and nothing else, so it runs at the memory's rate).
    constexpr double kComputeBound = 0.6;
    double white_mpts[5] = {};
    for (const auto& r : all_rows) {
        if (r.name.rfind("White_", 0) == 0 && r.has_gpu_baseline) white_mpts[r.dim] = r.gpu_glsl_mpts;
    }
    double log_sum_compute = 0.0;
    int compute_count = 0;
    for (const auto& r : all_rows) {
        if (r.tier != "Noise" || r.gpu_ratio <= 0.0 || white_mpts[r.dim] <= 0.0) continue;
        if (r.gpu_glsl_mpts >= kComputeBound * white_mpts[r.dim]) continue;
        log_sum_compute += std::log(r.gpu_ratio);
        compute_count++;
    }
    double geomean_gpu_compute = compute_count > 0 ? std::exp(log_sum_compute / compute_count) : 0.0;
    const auto& capstone = all_rows.back();

    std::printf("============================================================================================================================\n");
    std::printf("ROLL-UP SCORES:\n");
    std::printf("  1. Completeness / Parity : %d / %d (%5.1f%%) nodes & stencils pass CPU and GPU parity\n",
                parity_passed, parity_total, 100.0 * parity_passed / std::max(parity_total, 1));
    std::printf("  2. Speed (GeoMean CPU)   : %.3fx relative to FastNoise2 SIMD / C++ Reference\n", geomean_cpu);
    std::printf("  3. Speed (GeoMean GPU)   : %.3fx relative to glslc SPIR-V of the same node (%d rows with one)\n",
                geomean_gpu, gpu_count);
    std::printf("     of which compute-bound: %.3fx (%d noise rows under %.0f%% of White's memory-bound rate)\n",
                geomean_gpu_compute, compute_count, kComputeBound * 100);
    std::printf("  4. Capstone Planet Fields: %s (CPU: %.2f Mpt/s [%.2fx FastNoise2], GPU: %.2f Mpt/s, MaxErr: %.2e m)\n",
                (capstone.cpu_pass && capstone.gpu_pass) ? "PASS" : "FAIL",
                capstone.cpu_bk_mpts, capstone.cpu_ratio, capstone.gpu_bk_mpts,
                std::max(capstone.cpu_max_abs, capstone.gpu_max_abs));
    std::printf("  Total Harness Wall Time  : %.2f s\n", total_sec);
    std::printf("============================================================================================================================\n");

    if (write_golden) {
        write_golden_json(golden_path, all_rows, parity_passed, parity_total, geomean_cpu, geomean_gpu,
                          geomean_gpu_compute);
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
        check("geomean_gpu_compute_vs_glsl", geomean_gpu_compute);
        if (!ok) return 1;
        std::cout << "Golden ratchet check PASSED (" << parity_passed << "/" << parity_total
                  << " parity, geomeans within " << int(kNoise * 100) << "% of the golden or above).\n";
    }

    return (parity_passed == parity_total) ? 0 : 1;
}

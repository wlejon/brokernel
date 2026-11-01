// FastNoise2 CPU and GPU parity test suite for brokernel.
// Compares std::noise (in 2D, 3D, 4D, and f64 planetary origin) on CPU and GPU
// directly against FastNoise2's C++ SIMD reference implementation.

#include "brokernel/brokernel.h"
#include <FastNoise/FastNoise.h>

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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

struct ParityStats {
    float max_abs_err = 0.0f;
    uint32_t max_ulp = 0;
};

ParityStats compare_buffers(const std::vector<float>& got, const std::vector<float>& ref) {
    assert(got.size() == ref.size());
    ParityStats s{};
    for (size_t i = 0; i < got.size(); ++i) {
        float err = std::abs(got[i] - ref[i]);
        s.max_abs_err = std::max(s.max_abs_err, err);
        if (err > 1e-6f) {
            s.max_ulp = std::max(s.max_ulp, ulp_distance(got[i], ref[i]));
        }
    }
    return s;
}

static int g_failures = 0;

void check_node_parity(
    const char* name,
    const CompiledKernel* k,
    const std::vector<float>& ref,
    std::vector<float>& cpu_buf,
    std::vector<float>& gpu_buf,
    std::vector<KernelArg>& cpu_args,
    std::vector<KernelArg>& gpu_args,
    uint32_t N,
    float tol = 6e-5f) {
    assert(k != nullptr);
    std::fill(cpu_buf.begin(), cpu_buf.end(), -999.0f);
    std::fill(gpu_buf.begin(), gpu_buf.end(), -999.0f);

    bool ok_cpu = k->dispatch_cpu(GridShape(N), cpu_args);
    assert(ok_cpu);
    bool ok_gpu = k->dispatch_gpu(GridShape(N), gpu_args);
    assert(ok_gpu);

    ParityStats cs = compare_buffers(cpu_buf, ref);
    ParityStats gs = compare_buffers(gpu_buf, ref);
    if (cs.max_abs_err > tol || gs.max_abs_err > tol) {
        ++g_failures;
        std::cerr << "FAIL " << name << ": CPU max_err=" << cs.max_abs_err
                  << " (ulp=" << cs.max_ulp << "), GPU max_err=" << gs.max_abs_err
                  << " (ulp=" << gs.max_ulp << ") tol=" << tol << "\n";
        for (size_t i = 0; i < std::min<size_t>(N, 6); ++i) {
            std::cerr << "  [" << i << "] ref=" << ref[i]
                      << " cpu=" << cpu_buf[i] << " gpu=" << gpu_buf[i] << "\n";
        }
    } else {
        std::cout << "PASS " << name << " (cpu_max=" << cs.max_abs_err << ", gpu_max=" << gs.max_abs_err << ")\n";
    }
}

} // namespace

int main() {
    if (!brass::gpu::vulkan_available()) {
        std::cout << "test_fastnoise2_parity: no Vulkan device, skipped\n";
        return 77;
    }

    const char* bk_src = R"(
        import std::noise::*;

        @kernel @grid(1)
        fn k_value2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = value2d<f32>(vec2<f32>(px[i], py[i]), seed);
        }
        @kernel @grid(1)
        fn k_value3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
        }
        @kernel @grid(1)
        fn k_value4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
        }

        @kernel @grid(1)
        fn k_perlin2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = perlin2d<f32>(vec2<f32>(px[i], py[i]), seed);
        }
        @kernel @grid(1)
        fn k_perlin3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
        }
        @kernel @grid(1)
        fn k_perlin4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
        }

        @kernel @grid(1)
        fn k_simplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = simplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
        }
        @kernel @grid(1)
        fn k_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = simplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
        }
        @kernel @grid(1)
        fn k_simplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = simplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
        }

        @kernel @grid(1)
        fn k_supersimplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = supersimplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
        }
        @kernel @grid(1)
        fn k_supersimplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = supersimplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
        }
        @kernel @grid(1)
        fn k_supersimplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = supersimplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
        }

        @kernel @grid(1)
        fn k_white2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = white2d<f32>(vec2<f32>(px[i], py[i]), seed);
        }
        @kernel @grid(1)
        fn k_white3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = white3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
        }
        @kernel @grid(1)
        fn k_white4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = white4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
        }

        @kernel @grid(1)
        fn k_fbm2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = fbm_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
        }
        @kernel @grid(1)
        fn k_fbm3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = fbm_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
        }
        @kernel @grid(1)
        fn k_fbm4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = fbm_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
        }

        @kernel @grid(1)
        fn k_ridged2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ridged_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
        }
        @kernel @grid(1)
        fn k_ridged3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ridged_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
        }
        @kernel @grid(1)
        fn k_ridged4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ridged_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
        }

        @kernel @grid(1)
        fn k_pingpong2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ping_pong_fn<f32>(perlin2d<f32>(vec2<f32>(px[i], py[i]), seed), 2.0);
        }
        @kernel @grid(1)
        fn k_pingpong3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ping_pong_fn<f32>(perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed), 2.0);
        }
        @kernel @grid(1)
        fn k_pingpong4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = ping_pong_fn<f32>(perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed), 2.0);
        }

        @kernel @grid(1)
        fn k_cell_val2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_value2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_val3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_val4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 1.0);
        }

        @kernel @grid(1)
        fn k_cell_dist2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_distance2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_dist3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_distance3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_dist4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_distance4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 0, 1.0);
        }

        @kernel @grid(1)
        fn k_cell_lookup2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_lookup_perlin2d<f32>(vec2<f32>(px[i], py[i]), seed, 1.0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_lookup3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_lookup_perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 1.0, 1.0);
        }
        @kernel @grid(1)
        fn k_cell_lookup4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let i = g.x();
            out[i] = cellular_lookup_perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 1.0, 1.0);
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

        @kernel @cpu @grid(1)
        fn k_planet_f64_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, origin: f64, seed: i32) {
            let i = g.x();
            let p = vec3<f64>((px[i] as f64) + origin, (py[i] as f64) + origin, (pz[i] as f64) + origin);
            out[i] = simplex3d<f64>(p, seed) as f32;
        }
    )";

    auto res = compile_source(bk_src, "fastnoise2_parity.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);

    constexpr uint32_t N = 4096;
    constexpr int kSeed = 1337;
    std::vector<float> px(N), py(N), pz(N), pw(N);
    for (uint32_t i = 0; i < N; ++i) {
        float t = static_cast<float>(i) * 0.03125f - 32.0f;
        px[i] = t * 1.13f + 0.17f;
        py[i] = std::sin(t * 0.37f) * 14.3f - 2.1f;
        pz[i] = std::cos(t * 0.23f) * 11.7f + 3.4f;
        pw[i] = std::sin(t * 0.19f + 1.1f) * 9.5f - 1.3f;
    }

    std::vector<float> ref(N), cpu_out(N), gpu_out(N);

    // Both builds are fast math (fused multiply-adds, fused at different
    // sites; FastNoise2's FastLengthSqrt is an approximate inverse square
    // root), so a lattice noise differs by up to ~6e-5 where a corner's
    // falloff is steep, and the cellular distances and the warps, which move
    // positions by them, by up to ~1e-4.
    auto tol_for = [](const char*) { return 1.5e-4f; };
    auto run_2d = [&](const char* name, const char* kname, FastNoise::SmartNode<> fn) {
        const float tol = tol_for(name);
        fn->GenPositionArray2D(ref.data(), N, px.data(), py.data(), 0.0f, 0.0f, kSeed);
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(cpu_out), KernelArg(kSeed)};
        std::vector<KernelArg> gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(gpu_out), KernelArg(kSeed)};
        check_node_parity(name, res.find_kernel(kname), ref, cpu_out, gpu_out, cargs, gargs, N, tol);
    };
    auto run_3d = [&](const char* name, const char* kname, FastNoise::SmartNode<> fn) {
        const float tol = tol_for(name);
        fn->GenPositionArray3D(ref.data(), N, px.data(), py.data(), pz.data(), 0.0f, 0.0f, 0.0f, kSeed);
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(kSeed)};
        std::vector<KernelArg> gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(gpu_out), KernelArg(kSeed)};
        check_node_parity(name, res.find_kernel(kname), ref, cpu_out, gpu_out, cargs, gargs, N, tol);
    };
    auto run_4d = [&](const char* name, const char* kname, FastNoise::SmartNode<> fn) {
        const float tol = tol_for(name);
        fn->GenPositionArray4D(ref.data(), N, px.data(), py.data(), pz.data(), pw.data(), 0.0f, 0.0f, 0.0f, 0.0f, kSeed);
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(cpu_out), KernelArg(kSeed)};
        std::vector<KernelArg> gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(gpu_out), KernelArg(kSeed)};
        check_node_parity(name, res.find_kernel(kname), ref, cpu_out, gpu_out, cargs, gargs, N, tol);
    };

    // 1. Value (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Value>();
        fn->SetScale(1.0f);
        run_2d("Value 2D", "k_value2d", fn);
        run_3d("Value 3D", "k_value3d", fn);
        run_4d("Value 4D", "k_value4d", fn);
    }

    // 2. Perlin (2D, 3D, 4D)
    auto fn_perlin = FastNoise::New<FastNoise::Perlin>();
    fn_perlin->SetScale(1.0f);
    run_2d("Perlin 2D", "k_perlin2d", fn_perlin);
    run_3d("Perlin 3D", "k_perlin3d", fn_perlin);
    run_4d("Perlin 4D", "k_perlin4d", fn_perlin);

    // 3. Simplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        run_2d("Simplex 2D", "k_simplex2d", fn);
        run_3d("Simplex 3D", "k_simplex3d", fn);
        run_4d("Simplex 4D", "k_simplex4d", fn);
    }

    // 4. SuperSimplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::SuperSimplex>();
        fn->SetScale(1.0f);
        run_2d("SuperSimplex 2D", "k_supersimplex2d", fn);
        run_3d("SuperSimplex 3D", "k_supersimplex3d", fn);
        run_4d("SuperSimplex 4D", "k_supersimplex4d", fn);
    }

    // 5. White (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::White>();
        run_2d("White 2D", "k_white2d", fn);
        run_3d("White 3D", "k_white3d", fn);
        run_4d("White 4D", "k_white4d", fn);
    }

    // 6. FractalFBm (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalFBm>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        run_2d("FractalFBm 2D", "k_fbm2d", fn);
        run_3d("FractalFBm 3D", "k_fbm3d", fn);
        run_4d("FractalFBm 4D", "k_fbm4d", fn);
    }

    // 7. FractalRidged (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalRidged>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        run_2d("FractalRidged 2D", "k_ridged2d", fn);
        run_3d("FractalRidged 3D", "k_ridged3d", fn);
        run_4d("FractalRidged 4D", "k_ridged4d", fn);
    }

    // 8. PingPong (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::PingPong>();
        fn->SetSource(fn_perlin);
        fn->SetPingPongStrength(2.0f);
        run_2d("PingPong 2D", "k_pingpong2d", fn);
        run_3d("PingPong 3D", "k_pingpong3d", fn);
        run_4d("PingPong 4D", "k_pingpong4d", fn);
    }

    // 9. CellularValue (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularValue>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        run_2d("CellularValue 2D", "k_cell_val2d", fn);
        run_3d("CellularValue 3D", "k_cell_val3d", fn);
        run_4d("CellularValue 4D", "k_cell_val4d", fn);
    }

    // 10. CellularDistance (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularDistance>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        fn->SetGridJitter(1.0f);
        run_2d("CellularDistance 2D", "k_cell_dist2d", fn);
        run_3d("CellularDistance 3D", "k_cell_dist3d", fn);
        run_4d("CellularDistance 4D", "k_cell_dist4d", fn);
    }

    // 11. CellularLookup (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularLookup>();
        fn->SetScale(1.0f);
        fn->SetLookup(fn_perlin);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        run_2d("CellularLookup 2D", "k_cell_lookup2d", fn);
        run_3d("CellularLookup 3D", "k_cell_lookup3d", fn);
        run_4d("CellularLookup 4D", "k_cell_lookup4d", fn);
    }

    // 12. DomainWarpGradient (2D, 3D, 4D)
    auto fn_warp_grad = FastNoise::New<FastNoise::DomainWarpGradient>();
    fn_warp_grad->SetSource(fn_perlin);
    fn_warp_grad->SetWarpAmplitude(1.5f);
    fn_warp_grad->SetScale(2.0f); // frequency = 0.5
    run_2d("DomainWarpGradient 2D", "k_warp_grad2d", fn_warp_grad);
    run_3d("DomainWarpGradient 3D", "k_warp_grad3d", fn_warp_grad);
    run_4d("DomainWarpGradient 4D", "k_warp_grad4d", fn_warp_grad);

    // 13. DomainWarpFractalProgressive (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalProgressive>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        run_2d("DomainWarpFractalProgressive 2D", "k_warp_prog2d", fn);
        run_3d("DomainWarpFractalProgressive 3D", "k_warp_prog3d", fn);
        run_4d("DomainWarpFractalProgressive 4D", "k_warp_prog4d", fn);
    }

    // 14. DomainWarpFractalIndependent (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalIndependent>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        run_2d("DomainWarpFractalIndependent 2D", "k_warp_indep2d", fn);
        run_3d("DomainWarpFractalIndependent 3D", "k_warp_indep3d", fn);
        run_4d("DomainWarpFractalIndependent 4D", "k_warp_indep4d", fn);
    }

    // 15. Planetary double origin (6,371,000 m) Simplex 3D, on the CPU (f64
    // has no GPU build).
    {
        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        const double origin = 6371000.0;
        for (uint32_t i = 0; i < N; ++i) {
            ref[i] = fn->GenSingle3D(double(px[i]) + origin, double(py[i]) + origin, double(pz[i]) + origin, kSeed);
        }
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(origin), KernelArg(kSeed)};
        const CompiledKernel* k = res.find_kernel("k_planet_f64_simplex3d");
        assert(k != nullptr && k->cpu_only && k->spirv.words.empty());
        std::fill(cpu_out.begin(), cpu_out.end(), -999.0f);
        assert(k->dispatch_cpu(GridShape(N), cargs));
        ParityStats cs = compare_buffers(cpu_out, ref);
        if (cs.max_abs_err > 1e-5f) {
            ++g_failures;
            std::cerr << "FAIL Simplex 3D (f64 planet origin): CPU max_err=" << cs.max_abs_err << "\n";
        }
    }

    assert(g_failures == 0);
    std::cout << "test_fastnoise2_parity: ALL 43 NODE/DIMENSION CONFIGURATIONS PASSED\n";
    return 0;
}

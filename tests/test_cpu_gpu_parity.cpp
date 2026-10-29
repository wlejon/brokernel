// CPU vs GPU target agreement and point-query parity tests for brokernel.

#include "brokernel/brokernel.h"

#include <brass/target/host_cpu.hpp>

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace brokernel;

static void test_planetary_elevation_f64_parity() {
    const char* src = R"(
        import std::noise::{fbm, ridged};
        import std::sphere;

        @kernel @grid(2)
        fn bake_planet_tile(g: grid<2>, out_elev: write_store<f64, 2>, freq: f64) {
            let uv: vec2<f64> = g.uv_f64();
            let lat: f64 = (uv.y - 0.5) * 3.141592653589793;
            let lon: f64 = (uv.x - 0.5) * 6.283185307179586;
            let dir: vec3<f64> = sphere::latlon_to_unit<f64>(lat, lon);
            let base: f64 = fbm<f64, 3>(dir * freq, 4, 2.0, 0.5);
            let ridge: f64 = ridged<f64, 3>(dir * (freq * 1.5), 3, 2.0, 0.5);
            out_elev.write(base * 3000.0 + ridge * 1500.0);
        }
    )";

    auto res = compile_source(src, "planet_tile.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    assert(k->uses_f64);
    assert(k->grid_dim == 2);

    const uint32_t W = 32, H = 16;
    std::vector<double> elev(W * H, 0.0);
    std::vector<KernelArg> args = {KernelArg::store(elev), KernelArg(2.5)};

    auto rep = k->compare_cpu_gpu(GridShape(W, H), args, 1e-9, 1e-9);
    if (!rep.ok) {
        std::cerr << "planet_tile parity failed: " << rep.message << "\n";
    }
    assert(rep.ok && rep.ran_gpu);
    assert(rep.elements_compared == W * H);
    // Agreement on a constant (or unwritten) tile proves nothing: the GPU's
    // heights, written back into elev, must vary and be finite.
    double lo = elev[0], hi = elev[0];
    for (double e : elev) {
        assert(std::isfinite(e));
        lo = std::min(lo, e);
        hi = std::max(hi, e);
    }
    assert(hi - lo > 100.0);
}

static void test_simplex3d_range_and_parity() {
    const char* src = R"(
        import std::noise::{simplex};

        @kernel @grid(2)
        fn slice(g: grid<2>, out: write_store<f64, 2>, z: f64) {
            let uv: vec2<f64> = g.uv_f64();
            out.write(simplex<f64, 3>(vec3<f64>(uv.x * 23.0, uv.y * 23.0, z), 7));
        }
    )";
    auto res = compile_source(src, "simplex3d.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.default_kernel();
    const uint32_t W = 128, H = 128;
    std::vector<double> v(W * H, -99.0);
    std::vector<KernelArg> args = {KernelArg::store(v), KernelArg(0.37)};
    auto rep = k->compare_cpu_gpu(GridShape(W, H), args, 1e-12, 1e-12);
    if (!rep.ok) std::cerr << "simplex3d parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);
    double lo = 1e9, hi = -1e9, sum = 0.0;
    for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        sum += x;
    }
    std::cout << "simplex3d range [" << lo << ", " << hi << "] mean " << sum / v.size() << "\n";
    assert(lo >= -1.0 && hi <= 1.0);   // within the documented range
    assert(lo < -0.6 && hi > 0.6);     // and using most of it
    assert(std::abs(sum / v.size()) < 0.1);
}

static void test_stencil_diffusion_f64_and_point_query() {
    const char* src = R"(
        from std::filter import laplacian2d;

        @kernel @grid(2)
        fn diffuse_height(h: stencil<f64, 2, 1>, out: write_store<f64, 2>, dt: f64) -> f64 {
            let next_h: f64 = h.at(0, 0) + dt * laplacian2d(h);
            out.write(next_h);
            return next_h;
        }
    )";

    auto res = compile_source(src, "diffuse.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);

    const uint32_t W = 24, H = 24;
    std::vector<double> h_in(W * H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            h_in[y * W + x] = std::sin(0.2 * x) * std::cos(0.15 * y) * 100.0;
        }
    }
    std::vector<double> h_out(W * H, 0.0);
    std::vector<KernelArg> args = {
        KernelArg::store(h_in),
        KernelArg::store(h_out),
        KernelArg(0.125),
    };

    // 1. Compare full-grid CPU vs GPU dispatch
    auto rep = k->compare_cpu_gpu(GridShape(W, H), args, 1e-11, 1e-11);
    if (!rep.ok) {
        std::cerr << "diffuse parity failed: " << rep.message << "\n";
    }
    assert(rep.ok && rep.ran_gpu);

    // The GPU's result (written back into h_out) against the host: one
    // explicit diffusion step, neighbours clamped at the edges.
    auto at = [&](int x, int y) {
        x = std::clamp(x, 0, int(W) - 1);
        y = std::clamp(y, 0, int(H) - 1);
        return h_in[y * W + x];
    };
    for (int y = 0; y < int(H); ++y) {
        for (int x = 0; x < int(W); ++x) {
            double lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            double want = at(x, y) + 0.125 * lap;
            assert(std::abs(h_out[y * W + x] - want) < 1e-11);
        }
    }

    // 2. Verify single-point CPU query (`eval_point_2d_f64`) matches grid output at every cell
    std::vector<double> scratch_out(W * H, 0.0);
    std::vector<KernelArg> pt_args = {
        KernelArg::store(h_in),
        KernelArg::store(scratch_out),
        KernelArg(0.125),
    };
    for (uint32_t y = 0; y < H; y += 5) {
        for (uint32_t x = 0; x < W; x += 5) {
            double pt_val = k->eval_point_2d_f64(static_cast<int32_t>(x), static_cast<int32_t>(y), W, H, pt_args);
            assert(std::abs(pt_val - h_out[y * W + x]) < 1e-12);
        }
    }
}

static void test_3d_voxel_and_control_flow_parity() {
    const char* src = R"(
        fn march_steps(base: f32, max_iters: i32) -> f32 {
            if base < 0.0 {
                return 0.0;
            }
            var acc: f32 = base;
            for i in 0..max_iters {
                if i == 2 {
                    continue;
                }
                if acc > 50.0 {
                    break;
                }
                acc += (i as f32) * 1.5 + 0.25;
            }
            return acc;
        }

        @kernel @grid(3)
        fn fill_voxels(out: write_store<f32, 3>, scale: f32, iters: i32) {
            let x = grid_x() as f32;
            let y = grid_y() as f32;
            let z = grid_z() as f32;
            let seed = (x * 0.5 + y * 0.25 + z * 0.125) * scale;
            out.write(march_steps(seed, iters));
        }
    )";

    auto res = compile_source(src, "voxels.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    assert(k->grid_dim == 3);

    const uint32_t W = 8, H = 8, D = 6;
    std::vector<float> vox(W * H * D, 0.0f);
    std::vector<KernelArg> args = {KernelArg::store(vox), KernelArg(1.2f), KernelArg(7)};

    auto rep = k->compare_cpu_gpu(GridShape(W, H, D), args, 1e-5, 1e-5);
    if (!rep.ok) {
        std::cerr << "voxel 3D parity failed: " << rep.message << "\n";
    }
    assert(rep.ok && rep.ran_gpu);

    // march_steps on the host, at every voxel (exact: the same f32 operations).
    auto march = [](float base, int iters) {
        if (base < 0.0f) return 0.0f;
        float acc = base;
        for (int i = 0; i < iters; ++i) {
            if (i == 2) continue;
            if (acc > 50.0f) break;
            acc += float(i) * 1.5f + 0.25f;
        }
        return acc;
    };
    for (uint32_t z = 0; z < D; ++z) {
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float seed = (float(x) * 0.5f + float(y) * 0.25f + float(z) * 0.125f) * 1.2f;
                assert(vox[(z * H + y) * W + x] == march(seed, 7));
            }
        }
    }
}

// The CPU grid loop's vector build: 8 lanes for f32 work, 4 for f64, on a
// ragged width (the scalar remainder), with varying branches and a loop
// whose trip count differs per lane; and a return kernel dispatched with no
// return store (the lanes write scratch).
static void test_vector_grid_lanes() {
    const char* src = R"(
        @kernel @grid(2)
        fn k_f32(g: grid<2>, src: read_store<f32, 2>, out: write_store<f32, 2>, k: f32) -> f32 {
            let x = g.x();
            let y = g.y();
            var v: f32 = src[x, y] * k;
            if (v > 0.5) { v = v - 1.0; } else { v = v * 2.0; }
            var n: i32 = 0;
            for (var i: i32 = 0; i < (x & 7); i += 1) { n += i * 3 + y; }
            out[x, y] = v + f32(n);
            return v * 0.5;
        }
        @kernel @grid(2)
        fn k_f64(g: grid<2>, out: write_store<f64, 2>) {
            let x = g.x();
            let p = f64(x) * 0.25 + f64(g.y());
            out[x, g.y()] = select((x % 3) == 1, -p, p * p);
        }
    )";
    auto res = compile_source(src, "vector_grid.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k32 = res.find_kernel("k_f32");
    const auto* k64 = res.find_kernel("k_f64");
    assert(k32 && k64);
    if (brass::host_has_avx2()) {
        assert(k32->cpu_grid_lanes == 8);
        assert(k64->cpu_grid_lanes == 4);
    }
    const uint32_t W = 37, H = 5;
    std::vector<float> src_v(W * H);
    for (uint32_t i = 0; i < W * H; ++i) src_v[i] = static_cast<float>((i * 37) % 101) / 101.0f;
    std::vector<float> out_cpu(W * H, -9.0f), out_gpu(W * H, -9.0f);
    std::vector<float> ret_cpu(W * H, -9.0f), ret_gpu(W * H, -9.0f);
    std::string err;
    {
        std::vector<KernelArg> args = {KernelArg::store(src_v), KernelArg::store(out_cpu), KernelArg(0.8f)};
        assert(k32->dispatch_cpu(GridShape(W, H), args, ret_cpu.data(), &err));
        std::vector<float> dummy(W * H, -9.0f);
        std::vector<KernelArg> args2 = {KernelArg::store(src_v), KernelArg::store(dummy), KernelArg(0.8f)};
        assert(k32->dispatch_cpu(GridShape(W, H), args2, nullptr, &err));
        assert(dummy == out_cpu);
    }
    {
        std::vector<KernelArg> args = {KernelArg::store(src_v), KernelArg::store(out_gpu), KernelArg(0.8f)};
        assert(k32->dispatch_gpu(GridShape(W, H), args, ret_gpu.data(), &err));
    }
    for (uint32_t i = 0; i < W * H; ++i) {
        assert(std::abs(out_cpu[i] - out_gpu[i]) <= 1e-5f);
        assert(std::abs(ret_cpu[i] - ret_gpu[i]) <= 1e-5f);
    }
    std::vector<double> d(W * H, -9.0);
    std::vector<KernelArg> dargs = {KernelArg::store(d)};
    auto rep = k64->compare_cpu_gpu(GridShape(W, H), dargs, 1e-12, 1e-12);
    if (!rep.ok) std::cerr << "f64 vector grid parity failed: " << rep.message << "\n";
    assert(rep.ok);
}

// A vector build runs a group whose stencil reads all land inside its row
// without their x clamps, and a group within the halo of a row's end with
// them: every width here, from none of a row's groups inside (5, 8, 13) to
// most (64), against the clamped reads on the host and on the GPU.
static void test_vector_stencil_edges() {
    const char* src = R"(
        @kernel @grid(2)
        fn k_reach(g: grid<2>, s: stencil<f32, 2>, out: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            out[x, y] = s[-3, 0] * 1.0 + s[-1, 1] * 10.0 + s[0, 0] * 100.0 + s[2, -1] * 1000.0 + s[3, 0] * 10000.0;
        }
    )";
    auto res = compile_source(src, "vector_stencil.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_reach");
    assert(k && k->stencil_halo[0] == 3);
    if (brass::host_has_avx2() && k->cpu_grid_lanes != 8) {
        std::cerr << "k_reach runs " << k->cpu_grid_lanes << " lanes\n";
        assert(false);
    }
    const uint32_t H = 3;
    for (uint32_t W : {5u, 8u, 13u, 16u, 37u, 64u}) {
        std::vector<float> s(W * H);
        for (uint32_t i = 0; i < W * H; ++i) s[i] = static_cast<float>(i % 7) + 0.25f * static_cast<float>(i / 7);
        auto at = [&](int x, int y) {
            x = std::clamp(x, 0, static_cast<int>(W) - 1);
            y = std::clamp(y, 0, static_cast<int>(H) - 1);
            return s[y * W + x];
        };
        std::vector<float> out(W * H, -1.0f);
        std::string err;
        assert(k->dispatch_cpu(GridShape(W, H), {KernelArg::store(s), KernelArg::store(out)}, nullptr, &err));
        for (int y = 0; y < static_cast<int>(H); ++y) {
            for (int x = 0; x < static_cast<int>(W); ++x) {
                float want = at(x - 3, y) * 1.0f + at(x - 1, y + 1) * 10.0f + at(x, y) * 100.0f +
                             at(x + 2, y - 1) * 1000.0f + at(x + 3, y) * 10000.0f;
                if (out[y * W + x] != want) {
                    std::cerr << "stencil edge: width " << W << " cell (" << x << ", " << y << ") " << out[y * W + x]
                              << ", want " << want << "\n";
                    assert(false);
                }
            }
        }
        std::vector<float> gout(W * H, -1.0f);
        std::vector<KernelArg> gargs = {KernelArg::store(s), KernelArg::store(gout)};
        auto rep = k->compare_cpu_gpu(GridShape(W, H), gargs, 0.0, 0.0);
        if (!rep.ok) std::cerr << "stencil edge parity, width " << W << ": " << rep.message << "\n";
        assert(rep.ok);
    }
}

int main() {
    if (!brass::gpu::vulkan_available()) {
        std::cout << "test_cpu_gpu_parity: no Vulkan device, skipped\n";
        return 77;
    }
    test_planetary_elevation_f64_parity();
    test_simplex3d_range_and_parity();
    test_stencil_diffusion_f64_and_point_query();
    test_3d_voxel_and_control_flow_parity();
    test_vector_grid_lanes();
    test_vector_stencil_edges();
    std::cout << "test_cpu_gpu_parity: ALL PASSED\n";
    return 0;
}

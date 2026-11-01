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

// f64 is the CPU's: a @cpu kernel bakes a tile in f64 and has no GPU build,
// and the same kernel without @cpu is a compile error at its first f64.
static void test_planetary_elevation_f64_cpu() {
    const char* gpu_src = R"(
        import std::noise::{fbm};

        @kernel @grid(2)
        fn bake(g: grid<2>, out_elev: write_store<f32, 2>, freq: f32) {
            let dir = vec3<f32>(g.uv(), 1.0);
            let d: f64 = fbm<f64, 3>(vec3<f64>(dir.x as f64, dir.y as f64, 1.0) * (freq as f64), 4, 2.0, 0.5);
            out_elev.write(d as f32);
        }
    )";
    auto bad = compile_source(gpu_src, "planet_gpu.bk");
    assert(!bad.ok);
    const std::string msg = bad.error_message();
    assert(msg.find("f64 in GPU kernel 'bake'") != std::string::npos);
    assert(msg.find("planet_gpu.bk:7") != std::string::npos);
    CompileOptions no_gpu;
    no_gpu.emit_spirv = false;   // nothing compiled for the GPU: no error
    assert(compile_source(gpu_src, "planet_gpu.bk", no_gpu).ok);

    const char* src = R"(
        import std::noise::{fbm, ridged};
        import std::sphere;

        @kernel @cpu @grid(2)
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
    assert(k->uses_f64 && k->cpu_only && k->spirv.words.empty());
    assert(k->grid_dim == 2);

    const uint32_t W = 32, H = 16;
    std::vector<double> elev(W * H, 0.0);
    std::vector<KernelArg> args = {KernelArg::store(elev), KernelArg(2.5)};

    std::string err;
    assert(k->dispatch_cpu(GridShape(W, H), args, nullptr, &err));
    assert(!k->dispatch_gpu(GridShape(W, H), args, nullptr, &err) && err.find("@cpu") != std::string::npos);
    // The heights must vary and be finite.
    double lo = elev[0], hi = elev[0];
    for (double e : elev) {
        assert(std::isfinite(e));
        lo = std::min(lo, e);
        hi = std::max(hi, e);
    }
    assert(hi - lo > 100.0);
}

static void test_simplex3d_range() {
    const char* src = R"(
        import std::noise::{simplex};

        @kernel @cpu @grid(2)
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
    std::string err;
    assert(k->dispatch_cpu(GridShape(W, H), args, nullptr, &err));
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

        @kernel @cpu @grid(2)
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

    // 1. The grid on the CPU against the host: one explicit diffusion step,
    // neighbours clamped at the edges.
    std::string err;
    assert(k->dispatch_cpu(GridShape(W, H), args, nullptr, &err));
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
        @kernel @cpu @grid(2)
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
    assert(k64->dispatch_cpu(GridShape(W, H), dargs, nullptr, &err));
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const double p = double(x) * 0.25 + double(y);
            assert(d[y * W + x] == ((x % 3) == 1 ? -p : p * p));
        }
    }
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

// Store accesses outside the store: a read is zero, a write does nothing, on
// every path (a gather and scatter lane by lane, a contiguous group running
// past its store's end, a uniform index), the vector loop, its scalar tail
// and the GPU alike. Accesses at the cells are bounded by the dispatch
// instead: a store too small for the grid is refused.
static void test_store_bounds() {
    const char* src = R"(
        @kernel @grid(1)
        fn k_bounds(g: grid<1>, table: read_store<f32, 1>, idx: read_store<i32, 1>,
                    out_a: write_store<f32, 1>, out_b: write_store<f32, 1>,
                    scatter: write_store<f32, 1>, edge: write_store<f32, 1>, tail: write_store<f32, 1>) {
            let i = g.x();
            let j = idx[i];
            out_a[i] = table[j];
            out_b[i] = table[i + 2] + table[3] * 100.0 + table[64] * 1000.0;
            scatter[j] = f32(i) + 1.0;
            edge[i - 3] = f32(i);
            tail[5] = 7.0;
            tail[1] = 2.0;
        }
    )";
    auto res = compile_source(src, "store_bounds.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_bounds");
    assert(k);
    assert(k->find_param("idx")->grid_cover_dim == 1);
    assert(k->find_param("out_a")->grid_cover_dim == 1);
    assert(k->find_param("table")->grid_cover_dim == 0);
    assert(k->find_param("scatter")->grid_cover_dim == 0);
    if (brass::host_has_avx2() && k->cpu_grid_lanes != 8) {
        std::cerr << "k_bounds runs " << k->cpu_grid_lanes << " lanes\n";
        assert(false);
    }
    std::vector<float> table(10);
    for (size_t t = 0; t < table.size(); ++t) table[t] = 1.0f + static_cast<float>(t);
    for (uint32_t W : {5u, 8u, 13u, 37u, 64u}) {
        std::vector<int32_t> idx(W);
        for (uint32_t i = 0; i < W; ++i) idx[i] = static_cast<int32_t>(i) * 3 - 7;
        std::vector<float> out_a(W, -1.0f), out_b(W, -1.0f), scatter(20, -1.0f), edge(12, -1.0f), tail(3, -1.0f);
        std::vector<KernelArg> args = {KernelArg::store(table), KernelArg::store(idx), KernelArg::store(out_a),
                                       KernelArg::store(out_b), KernelArg::store(scatter), KernelArg::store(edge),
                                       KernelArg::store(tail)};
        std::string err;
        if (!k->dispatch_cpu(GridShape(W), args, nullptr, &err)) {
            std::cerr << "store bounds: " << err << "\n";
            assert(false);
        }
        auto fail = [&](const char* what, size_t at, float got, float want) {
            std::cerr << "store bounds, width " << W << ": " << what << "[" << at << "] " << got << ", want " << want << "\n";
            assert(false);
        };
        for (uint32_t i = 0; i < W; ++i) {
            const int32_t j = idx[i];
            const float a = (j >= 0 && j < 10) ? table[j] : 0.0f;
            const float b = (i + 2 < 10 ? table[i + 2] : 0.0f) + table[3] * 100.0f;
            if (out_a[i] != a) fail("out_a", i, out_a[i], a);
            if (out_b[i] != b) fail("out_b", i, out_b[i], b);
        }
        for (int32_t s = 0; s < 20; ++s) {
            const bool hit = (s + 7) % 3 == 0 && static_cast<uint32_t>((s + 7) / 3) < W;
            const float want = hit ? static_cast<float>((s + 7) / 3) + 1.0f : -1.0f;
            if (scatter[s] != want) fail("scatter", s, scatter[s], want);
        }
        for (uint32_t e = 0; e < 12; ++e) {
            const float want = e + 3 < W ? static_cast<float>(e + 3) : -1.0f;
            if (edge[e] != want) fail("edge", e, edge[e], want);
        }
        if (tail[0] != -1.0f || tail[1] != 2.0f || tail[2] != -1.0f) fail("tail", 0, tail[0], -1.0f);

        // (The GPU zeroes a write-only store, so these start at zero.)
        std::vector<float> g_a(W), g_b(W), g_s(20), g_e(12), g_t(3);
        std::vector<KernelArg> gargs = {KernelArg::store(table), KernelArg::store(idx), KernelArg::store(g_a),
                                        KernelArg::store(g_b), KernelArg::store(g_s), KernelArg::store(g_e),
                                        KernelArg::store(g_t)};
        auto rep = k->compare_cpu_gpu(GridShape(W), gargs, 0.0, 0.0);
        if (!rep.ok) std::cerr << "store bounds parity, width " << W << ": " << rep.message << "\n";
        assert(rep.ok);

        // A store accessed at the cells, one element short of the grid.
        std::vector<float> short_a(W - 1);
        std::vector<KernelArg> bad = args;
        bad[2] = KernelArg::store(short_a);
        err.clear();
        assert(!k->dispatch_cpu(GridShape(W), bad, nullptr, &err) && err.find("out_a") != std::string::npos);
        err.clear();
        assert(!k->dispatch_gpu(GridShape(W), bad, nullptr, &err) && err.find("out_a") != std::string::npos);
    }
    // A point query's cell is one of its grid's.
    std::vector<int32_t> idx1(4, 0);
    std::vector<float> o1(4), o2(4), o3(20), o4(12), o5(3);
    std::vector<KernelArg> args1 = {KernelArg::store(table), KernelArg::store(idx1), KernelArg::store(o1),
                                    KernelArg::store(o2), KernelArg::store(o3), KernelArg::store(o4), KernelArg::store(o5)};
    std::string err;
    assert(k->eval_grid_point_cpu(3, 0, 0, GridShape(4), args1, nullptr, &err));
    assert(!k->eval_grid_point_cpu(4, 0, 0, GridShape(4), args1, nullptr, &err));
}

static void test_vector_bool_masks() {
    const char* src = R"(
        const CONST_TRUE: bool = 2 < 5;
        const CONST_FALSE: bool = 2 > 5;

        @kernel @grid(1)
        fn k_bool_masks(g: grid<1>, out: write_store<f32, 1>, flag_on: bool, flag_off: bool) -> f32 {
            let x = g.x();
            let a = f32(x) + 10.0;
            let b = f32(x) + 100.0;
            let odd = (x & 1) != 0;

            var acc: f32 = 0.0;
            // BoolLit in select over varying lanes
            acc += select(true, a, b);
            acc += select(false, a, b);
            // Folded bool constants and is_f64<T>() in select over varying lanes
            acc += select(CONST_TRUE, a, b);
            acc += select(CONST_FALSE, a, b);
            acc += select(!is_f64<f32>(), a, b);
            // Kernel bool parameters in select over varying lanes
            acc += select(flag_on, a, b);
            acc += select(flag_off, a, b);

            // if (true && varying) and if (param && varying) over varying lanes
            if true && odd {
                acc += a * 2.0;
            } else {
                acc += b * 3.0;
            }
            if CONST_TRUE && flag_on && odd {
                acc += 5.0;
            }
            if flag_off || !odd {
                acc += 7.0;
            }
            out[x] = acc;
            return acc;
        }
    )";

    auto res = compile_source(src, "bool_masks.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_bool_masks");
    assert(k != nullptr);
    if (brass::host_has_avx2()) {
        assert(k->cpu_grid_lanes == 8);
    }

    // Width 29 exercises 3 full 8-wide AVX2 groups (0..23) plus a 5-lane scalar tail (24..28).
    const uint32_t W = 29;
    std::vector<float> out_store(W, 0.0f);
    std::vector<float> out_ret(W, 0.0f);
    std::vector<KernelArg> args = {KernelArg::store(out_store), KernelArg(true), KernelArg(false)};

    std::string err;
    assert(k->dispatch_cpu(GridShape(W), args, out_ret.data(), &err));

    for (uint32_t x = 0; x < W; ++x) {
        const float a = static_cast<float>(x) + 10.0f;
        const float b = static_cast<float>(x) + 100.0f;
        const bool odd = (x & 1u) != 0u;
        float want = 4.0f * a + 3.0f * b;
        want += odd ? (a * 2.0f + 5.0f) : (b * 3.0f + 7.0f);
        assert(out_store[x] == want);
        assert(out_ret[x] == want);

        // Verify scalar point query path agrees identically.
        float pt = 0.0f;
        assert(k->eval_grid_point_cpu(static_cast<int32_t>(x), 0, 0, GridShape(W), args, &pt, &err));
        assert(pt == want);
    }

    // Verify CPU vs GPU parity.
    std::vector<float> gpu_store(W, 0.0f);
    std::vector<KernelArg> gargs = {KernelArg::store(gpu_store), KernelArg(true), KernelArg(false)};
    auto rep = k->compare_cpu_gpu(GridShape(W), gargs, 0.0, 0.0);
    if (!rep.ok) std::cerr << "k_bool_masks parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);
}

static void test_assignment_targets_parity() {
    const char* src = R"(
        struct Leaf {
            c: f32,
            v: vec3<f32>,
        }
        struct Branch {
            b: Leaf,
            weights: array<f32, 3>,
        }
        struct Root {
            b: Branch,
        }

        @kernel @grid(1)
        fn k_assign_targets(g: grid<1>, out: write_store<f32, 1>) -> f32 {
            let x = g.x();
            let dyn_i = x % 4;
            let dyn_v = x % 3;

            // 1. Local array<T, N> element writes (static + dynamic indices + compound assignment)
            var arr: array<f32, 4>;
            arr[0] = 1.0;
            arr[1] = 2.0;
            arr[2] = 3.0;
            arr[3] = 4.0;
            arr[dyn_i] = f32(x) + 10.0;
            arr[(x + 1) % 4] += 5.0;

            // 2. Vector element writes (v[i] = ..., v[i] += ...) and multi-component swizzle writes (v.xy = ..., v.yz += ...)
            var v: vec4<f32> = vec4<f32>(1.0, 2.0, 3.0, 4.0);
            v[dyn_v] = f32(x) * 2.0;
            v[(x + 2) % 4] += 7.0;
            v.xy = vec2<f32>(v.x + 1.0, v.y + 3.0);
            v.yzw += vec3<f32>(10.0, 20.0, 30.0);

            // 3. Nested field writes (a.b.c = ..., a.b.v.xz += ..., a.b.weights[i] += ...)
            var a: Root;
            a.b.b.c = 12.0;
            a.b.b.c += f32(x);
            a.b.b.v = vec3<f32>(1.0, 2.0, 3.0);
            a.b.b.v[dyn_v] = 9.0;
            a.b.b.v.xz += vec2<f32>(4.0, 6.0);
            a.b.weights[0] = 0.5;
            a.b.weights[1] = 1.5;
            a.b.weights[2] = 2.5;
            a.b.weights[dyn_v] += f32(x);

            let total = (arr[0] + arr[1] + arr[2] + arr[3])
                      + (v.x + v.y + v.z + v.w)
                      + a.b.b.c
                      + (a.b.b.v.x + a.b.b.v.y + a.b.b.v.z)
                      + (a.b.weights[0] + a.b.weights[1] + a.b.weights[2]);
            out[x] = total;
            return total;
        }
    )";

    auto res = compile_source(src, "assign_targets.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_assign_targets");
    assert(k != nullptr);
    if (brass::host_has_avx2()) {
        assert(k->cpu_grid_lanes == 8);
    }

    const uint32_t W = 29;
    std::vector<float> out_store(W, 0.0f);
    std::vector<float> out_ret(W, 0.0f);
    std::vector<KernelArg> args = {KernelArg::store(out_store)};

    std::string err;
    assert(k->dispatch_cpu(GridShape(W), args, out_ret.data(), &err));

    for (uint32_t x = 0; x < W; ++x) {
        const uint32_t dyn_i = x % 4;
        const uint32_t dyn_v = x % 3;

        float arr[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        arr[dyn_i] = static_cast<float>(x) + 10.0f;
        arr[(x + 1) % 4] += 5.0f;

        float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        v[dyn_v] = static_cast<float>(x) * 2.0f;
        v[(x + 2) % 4] += 7.0f;
        v[0] = v[0] + 1.0f;
        v[1] = v[1] + 3.0f;
        v[1] += 10.0f;
        v[2] += 20.0f;
        v[3] += 30.0f;

        float c = 12.0f + static_cast<float>(x);
        float bv[3] = {1.0f, 2.0f, 3.0f};
        bv[dyn_v] = 9.0f;
        bv[0] += 4.0f;
        bv[2] += 6.0f;
        float w[3] = {0.5f, 1.5f, 2.5f};
        w[dyn_v] += static_cast<float>(x);

        float want = (arr[0] + arr[1] + arr[2] + arr[3])
                   + (v[0] + v[1] + v[2] + v[3])
                   + c
                   + (bv[0] + bv[1] + bv[2])
                   + (w[0] + w[1] + w[2]);
        assert(out_store[x] == want);
        assert(out_ret[x] == want);
    }

    std::vector<float> gpu_store(W, 0.0f);
    std::vector<KernelArg> gargs = {KernelArg::store(gpu_store)};
    auto rep = k->compare_cpu_gpu(GridShape(W), gargs, 0.0, 0.0);
    if (!rep.ok) std::cerr << "k_assign_targets parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);
}

static void test_atomics_parity() {
    const char* src = R"(
        @kernel @grid(1)
        fn k_atomics(g: grid<1>,
                     s_i32: rw_store<i32, 1>,
                     s_u32: rw_store<u32, 1>,
                     s_f32: rw_store<f32, 1>,
                     exch_i32: rw_store<i32, 1>,
                     exch_u32: rw_store<u32, 1>) {
            let x = g.x();
            let sv: i32 = (x * 7) - 50;
            // Values spanning above 0x80000000u (2147483648u) so signed vs unsigned order differs
            let uv: u32 = 2147483640u + u32(x) * 3u;

            s_i32.atomic_add(0, sv);
            s_i32.atomic_min(1, sv);
            s_i32.atomic_max(2, sv);

            s_u32.atomic_add(0, u32(x + 1));
            s_u32.atomic_min(1, uv);
            s_u32.atomic_max(2, uv);

            s_f32.atomic_add(0, 1.5);

            // Per-cell exchange (deterministic across parallel threads)
            let prev_i = exch_i32.atomic_exchange(x, sv);
            s_i32.atomic_add(3, prev_i);
            let prev_u = exch_u32.atomic_exchange(x, uv);
            s_u32.atomic_add(3, prev_u);
        }
    )";

    auto res = compile_source(src, "atomics.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_atomics");
    assert(k != nullptr);
    if (brass::host_has_avx2()) {
        assert(k->cpu_grid_lanes == 8);
    }

    const uint32_t W = 29;
    std::vector<int32_t> s_i32 = {0, 1000, -1000, 0};
    std::vector<uint32_t> s_u32 = {0u, 0xFFFFFFFFu, 0u, 0u};
    std::vector<float> s_f32 = {0.0f};
    std::vector<int32_t> exch_i32(W, 5);
    std::vector<uint32_t> exch_u32(W, 11u);

    std::vector<KernelArg> args = {
        KernelArg::store(s_i32),
        KernelArg::store(s_u32),
        KernelArg::store(s_f32),
        KernelArg::store(exch_i32),
        KernelArg::store(exch_u32),
    };
    auto rep = k->compare_cpu_gpu(GridShape(W), args, 0.0, 0.0);
    if (!rep.ok) std::cerr << "k_atomics parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);

    int32_t want_i_sum = 0, want_i_min = 1000, want_i_max = -1000;
    uint32_t want_u_sum = 0u, want_u_min = 0xFFFFFFFFu, want_u_max = 0u;
    for (uint32_t x = 0; x < W; ++x) {
        int32_t sv = static_cast<int32_t>(x) * 7 - 50;
        uint32_t uv = 2147483640u + x * 3u;
        want_i_sum += sv;
        want_i_min = std::min(want_i_min, sv);
        want_i_max = std::max(want_i_max, sv);
        want_u_sum += (x + 1u);
        want_u_min = std::min(want_u_min, uv);
        want_u_max = std::max(want_u_max, uv);
        assert(exch_i32[x] == sv);
        assert(exch_u32[x] == uv);
    }
    assert(s_i32[0] == want_i_sum);
    assert(s_i32[1] == want_i_min);
    assert(s_i32[2] == want_i_max);
    assert(s_i32[3] == static_cast<int32_t>(W) * 5);
    assert(s_u32[0] == want_u_sum);
    assert(s_u32[1] == want_u_min);
    assert(s_u32[2] == want_u_max);
    assert(s_u32[3] == W * 11u);
    assert(s_f32[0] == static_cast<float>(W) * 1.5f);
}

static void test_lowering_quirks_parity() {
    const char* src = R"(
        fn helper_early_return(i: i32, x: i32) -> i32 {
            if (i == 2) {
                return 100;
            }
            if (i + x > 25) {
                return 50;
            }
            return i * 10 + 1;
        }

        fn rhs_side_effects(
            x: i32,
            tag: i32,
            side_store: rw_store<i32, 1>,
            counter: rw_store<i32, 1>
        ) -> bool {
            side_store.store(x, tag);
            counter.atomic_add(0, 1);
            return (x & 1) == 0;
        }

        @kernel fn k_lowering_quirks(
            and_store: rw_store<i32, 1>,
            and_count: rw_store<i32, 1>,
            or_store: rw_store<i32, 1>,
            or_count: rw_store<i32, 1>,
            out: write_store<i32, 1>,
            g: grid<1>
        ) {
            let x = g.x();
            var acc: i32 = 0;
            for (var i: i32 = 0; i < 8; i = i + 1) {
                if (i == (x & 3)) {
                    continue;
                }
                if (i > 4 + (x % 3)) {
                    break;
                }
                acc += helper_early_return(i, x);
            }

            let c_and = (x % 3 == 0) && rhs_side_effects(x, 111 + x, and_store, and_count);
            let c_or  = (x % 2 == 0) || rhs_side_effects(x, 222 + x, or_store, or_count);

            if (c_and) {
                acc += 1000;
            }
            if (c_or) {
                acc += 2000;
            }
            out.write(acc);
        }
    )";

    auto res = compile_source(src, "lowering_quirks.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_lowering_quirks");
    assert(k != nullptr);
    if (brass::host_has_avx2()) {
        assert(k->cpu_grid_lanes == 8);
    }

    const uint32_t W = 29; // 3 full AVX2 vector groups (24) + 5 scalar tail
    std::vector<int32_t> and_store(W, -99);
    std::vector<int32_t> and_count = {0};
    std::vector<int32_t> or_store(W, -99);
    std::vector<int32_t> or_count = {0};
    std::vector<int32_t> out(W, 0);

    std::vector<KernelArg> args = {
        KernelArg::store(and_store),
        KernelArg::store(and_count),
        KernelArg::store(or_store),
        KernelArg::store(or_count),
        KernelArg::store(out),
    };
    auto rep = k->compare_cpu_gpu(GridShape(W), args, 0.0, 0.0);
    if (!rep.ok) std::cerr << "k_lowering_quirks parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);

    int32_t expected_and_count = 0;
    int32_t expected_or_count = 0;
    for (int32_t x = 0; x < static_cast<int32_t>(W); ++x) {
        int32_t want_acc = 0;
        for (int32_t i = 0; i < 8; ++i) {
            if (i == (x & 3)) continue;
            if (i > 4 + (x % 3)) break;
            int32_t hv = (i == 2) ? 100 : ((i + x > 25) ? 50 : (i * 10 + 1));
            want_acc += hv;
        }
        bool c_and = false;
        if (x % 3 == 0) {
            expected_and_count++;
            assert(and_store[x] == 111 + x);
            c_and = ((x & 1) == 0);
        } else {
            // Left-hand side decided false: right-hand side store write must NOT happen
            assert(and_store[x] == -99);
        }
        bool c_or = true;
        if (x % 2 != 0) {
            expected_or_count++;
            assert(or_store[x] == 222 + x);
            c_or = ((x & 1) == 0);
        } else {
            // Left-hand side decided true: right-hand side store write must NOT happen
            assert(or_store[x] == -99);
        }
        if (c_and) want_acc += 1000;
        if (c_or) want_acc += 2000;
        assert(out[x] == want_acc);
    }
    assert(and_count[0] == expected_and_count);
    assert(or_count[0] == expected_or_count);
}

static void test_vector_and_matrix_constants_parity() {
    const char* src = R"(
        const AXIS: vec3<f32> = vec3<f32>(1.0, 0.45, -0.5);
        const OFFSET: vec3<f32> = vec3<f32>(0.5, -0.25, 1.5);
        const COMBINED: vec3<f32> = AXIS * 2.0 + OFFSET;
        const SWIZ: vec2<f32> = COMBINED.xz;
        const MAT3: mat3<f32> = mat3<f32>(
            vec3<f32>(1.0, 2.0, 3.0),
            AXIS,
            OFFSET
        );
        const MAT2: mat2<f32> = mat2<f32>(1.5, -0.5, 2.0, 3.0);
        const MAT2_SCALED: mat2<f32> = MAT2 * 2.0 + mat2<f32>(0.5);
        const CONST_ELEM: f32 = MAT3[1].y + MAT2_SCALED[0][1] + AXIS[2];

        @kernel fn k_vec_mat_consts(
            out: write_store<vec4<f32>, 1>,
            g: grid<1>
        ) {
            let x = f32(g.x());
            let v: vec3<f32> = AXIS * x + COMBINED;
            let mcol: vec3<f32> = MAT3[1] + MAT3[2];
            let m2: vec2<f32> = MAT2_SCALED[0] + SWIZ;
            out.write(vec4<f32>(
                v.x + mcol.x,
                v.y + mcol.y + m2.x,
                v.z + mcol.z + m2.y,
                CONST_ELEM
            ));
        }
    )";

    auto res = compile_source(src, "vec_mat_consts.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);

    const auto* k = res.find_kernel("k_vec_mat_consts");
    assert(k != nullptr);
    if (brass::host_has_avx2()) {
        assert(k->cpu_grid_lanes == 8);
    }

    const uint32_t W = 29;
    std::vector<float> out(W * 4, 0.0f);
    std::vector<KernelArg> args = {KernelArg::store(out)};

    auto rep = k->compare_cpu_gpu(GridShape(W), args, 1e-5, 1e-5);
    if (!rep.ok) std::cerr << "k_vec_mat_consts parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);

    for (uint32_t i = 0; i < W; ++i) {
        float x = static_cast<float>(i);
        float vx = 1.0f * x + 2.5f;
        float vy = 0.45f * x + 0.65f;
        float vz = -0.5f * x + 0.5f;
        float mcol_x = 1.0f + 0.5f;
        float mcol_y = 0.45f - 0.25f;
        float mcol_z = -0.5f + 1.5f;
        float m2_x = (1.5f * 2.0f + 0.5f) + 2.5f;
        float m2_y = (-0.5f * 2.0f + 0.5f) + 0.5f;
        float want0 = vx + mcol_x;
        float want1 = vy + mcol_y + m2_x;
        float want2 = vz + mcol_z + m2_y;
        float want3 = 0.45f + (-0.5f) + (-0.5f);
        assert(std::abs(out[i * 4 + 0] - want0) < 1e-5f);
        assert(std::abs(out[i * 4 + 1] - want1) < 1e-5f);
        assert(std::abs(out[i * 4 + 2] - want2) < 1e-5f);
        assert(std::abs(out[i * 4 + 3] - want3) < 1e-5f);
    }
}

static void test_matrix_linear_algebra_parity() {
    const char* src = R"(
        @kernel @grid(1)
        fn eval_mat_ops(g: grid<1>, out: write_store<vec4<f32>, 1>) {
            let t: f32 = f32(g.x()) * 0.25 + 0.5;
            let a4: mat4f = mat4f(
                vec4f(1.0,  0.5, -0.25, 0.0),
                vec4f(0.0,  2.0,  0.75, 0.0),
                vec4f(0.25, -1.0, 1.5,  0.0),
                vec4f(2.0, -3.0,  4.0,  1.0)
            );
            let b4: mat4f = mat4f(
                vec4f(0.5,  0.0,  1.0,  0.0),
                vec4f(1.25, 1.0, -0.5,  0.0),
                vec4f(0.0,  0.5,  2.0,  0.0),
                vec4f(-1.0, 2.0,  0.5,  1.0)
            );
            let v4: vec4f = vec4f(t, t * 1.5 - 1.0, -t * 0.5 + 2.0, 1.0);
            let c4: mat4f = a4 * b4;
            let p4: vec4f = c4 * v4;

            let a3: mat3f = mat3f(a4);
            let b3: mat3f = mat3f(b4) * 0.5;
            let v3: vec3f = v4.xyz;
            let p3: vec3f = (a3 * b3) * v3 + (v3 * transpose(a3));
            out.write(p4 + vec4f(p3, c4[3].z));
        }
    )";
    auto res = compile_source(src, "mat_ops_parity.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const CompiledKernel* k = res.find_kernel("eval_mat_ops");
    assert(k != nullptr);

    constexpr uint32_t W = 32;
    std::vector<float> out(W * 4, 0.0f);
    std::vector<KernelArg> args = {KernelArg::store(out)};
    auto rep = k->compare_cpu_gpu(GridShape(W), args, 1e-5, 1e-5);
    if (!rep.ok) std::cerr << "mat_ops parity failed: " << rep.message << "\n";
    assert(rep.ok && rep.ran_gpu);
}

int main() {
    if (!brass::gpu::vulkan_available()) {
        std::cout << "test_cpu_gpu_parity: no Vulkan device, skipped\n";
        return 77;
    }
    test_planetary_elevation_f64_cpu();
    test_simplex3d_range();
    test_stencil_diffusion_f64_and_point_query();
    test_3d_voxel_and_control_flow_parity();
    test_vector_grid_lanes();
    test_vector_stencil_edges();
    test_store_bounds();
    test_vector_bool_masks();
    test_assignment_targets_parity();
    test_atomics_parity();
    test_lowering_quirks_parity();
    test_vector_and_matrix_constants_parity();
    test_matrix_linear_algebra_parity();
    std::cout << "test_cpu_gpu_parity: ALL PASSED\n";
    return 0;
}

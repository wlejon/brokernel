// Tests for Phase 2 runtime diagnostics: `check("label", cond)`, `trap_non_finite`,
// `probe_if("label", cond, expr)`, and compile-cache isolation/serialization.
#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace brokernel;

static void test_check_cpu_and_gpu() {
    const char* src = R"(
        fn guard_positive(v: f32) -> f32 {
            if (!check("positive", v > 0.0)) {
                return 0.0;
            }
            return v;
        }

        @kernel @grid(2)
        fn k_checked(g: grid<2>, inp: read_store<f32, 2>, out: write_store<f32, 2>) {
            let x = inp[g.x(), g.y()];
            check("in_range", x < 100.0);
            out[g.x(), g.y()] = guard_positive(x);
        }
    )";

    // 1. Default compile: checks_compiled == false, check_labels recorded, normal dispatch works.
    auto res_off = compile_source(src, "checks_off.bk");
    if (!res_off.ok) std::cerr << res_off.error_message() << "\n";
    assert(res_off.ok);
    const auto* k_off = res_off.find_kernel("k_checked");
    assert(k_off != nullptr);
    assert(!k_off->checks_compiled);
    assert(!k_off->trap_non_finite_compiled);
    assert(k_off->check_labels.size() == 2);

    std::vector<float> inp(6 * 4, 2.0f);
    std::vector<float> out(6 * 4, -99.0f);
    std::string err;
    assert(k_off->dispatch_cpu(GridShape(6, 4), {KernelArg::store(inp), KernelArg::store(out)}, nullptr, &err));
    assert(std::abs(out[0] - 2.0f) < 1e-6f);

    // Passing case via lazy checked variant on CPU (both trap_nf=false and trap_nf=true).
    CheckReport rep = k_off->check_cpu(GridShape(6, 4), {KernelArg::store(inp), KernelArg::store(out)}, false, &err);
    if (!rep.ok) std::cerr << "unexpected check_cpu failure: " << rep.message << "\n";
    assert(rep.ok);
    assert(rep.kind == 0);

    // Failing case at exact cell (4, 2) via lazy checked variant; out buffer must remain intact.
    std::vector<float> inp_bad = inp;
    inp_bad[2 * 6 + 4] = -3.5f;
    std::vector<float> out_sentinel(6 * 4, 77.0f);
    rep = k_off->check_cpu(GridShape(6, 4), {KernelArg::store(inp_bad), KernelArg::store(out_sentinel)}, false, &err);
    assert(!rep.ok);
    assert(rep.kind == 1);
    assert(rep.coord[0] == 4 && rep.coord[1] == 2 && rep.coord[2] == 0);
    assert(rep.label == "positive");
    assert(rep.message.find("positive") != std::string::npos);
    assert(out_sentinel[2 * 6 + 4] == 77.0f); // diagnostic run did not mutate caller buffer

    // Second check label ("in_range") failing at (1, 3).
    std::vector<float> inp_big = inp;
    inp_big[3 * 6 + 1] = 150.0f;
    rep = k_off->check_cpu(GridShape(6, 4), {KernelArg::store(inp_big), KernelArg::store(out_sentinel)}, false, &err);
    assert(!rep.ok);
    assert(rep.kind == 1);
    assert(rep.coord[0] == 1 && rep.coord[1] == 3 && rep.coord[2] == 0);
    assert(rep.label == "in_range");

    // 2. Explicit enable_checks = true compile.
    CompileOptions copts;
    copts.enable_checks = true;
    auto res_on = compile_source(src, "checks_on.bk", copts);
    assert(res_on.ok);
    const auto* k_on = res_on.find_kernel("k_checked");
    assert(k_on != nullptr);
    assert(k_on->checks_compiled);
    assert(!k_on->trap_non_finite_compiled);
    // Normal dispatch_cpu still works when compiled with enable_checks = true.
    assert(k_on->dispatch_cpu(GridShape(6, 4), {KernelArg::store(inp_bad), KernelArg::store(out)}, nullptr, &err));
    assert(out[2 * 6 + 4] == 0.0f);

    if (brass::gpu::vulkan_available()) {
        std::cout << "[test_diagnostics] Vulkan GPU device available: running GPU checks\n";
        // Lazy GPU check on k_off and direct GPU check on k_on.
        CheckReport gok = k_off->check_gpu(GridShape(6, 4), {KernelArg::store(inp), KernelArg::store(out_sentinel)}, false, &err);
        if (!gok.ok) std::cerr << "unexpected check_gpu failure: " << gok.message << "\n";
        assert(gok.ok);

        CheckReport gfail = k_off->check_gpu(GridShape(6, 4), {KernelArg::store(inp_bad), KernelArg::store(out_sentinel)}, false, &err);
        assert(!gfail.ok);
        assert(gfail.kind == 1);
        assert(gfail.coord[0] == 4 && gfail.coord[1] == 2 && gfail.coord[2] == 0);
        assert(gfail.label == "positive");
        assert(out_sentinel[2 * 6 + 4] == 77.0f);

        CheckReport gfail2 = k_on->check_gpu(GridShape(6, 4), {KernelArg::store(inp_big), KernelArg::store(out_sentinel)}, false, &err);
        assert(!gfail2.ok);
        assert(gfail2.kind == 1);
        assert(gfail2.coord[0] == 1 && gfail2.coord[1] == 3 && gfail2.coord[2] == 0);
        assert(gfail2.label == "in_range");

        // Normal dispatch_gpu with enable_checks = true.
        assert(k_on->dispatch_gpu(GridShape(6, 4), {KernelArg::store(inp_bad), KernelArg::store(out)}, nullptr, &err));
    }
}

static void test_trap_non_finite() {
    const char* src = R"(
        @kernel @grid(2)
        fn k_div_store(g: grid<2>, num: read_store<f32, 2>, den: read_store<f32, 2>, out: write_store<f32, 2>) {
            out[g.x(), g.y()] = num[g.x(), g.y()] / den[g.x(), g.y()];
        }

        @kernel @grid(2)
        fn k_div_ret(g: grid<2>, num: read_store<f32, 2>, den: read_store<f32, 2>) -> vec2<f32> {
            let v = num[g.x(), g.y()] / den[g.x(), g.y()];
            return vec2<f32>(1.0, v);
        }

        @kernel @cpu @grid(1)
        fn k_div_f64(g: grid<1>, num: read_store<f64, 1>, den: read_store<f64, 1>, out: write_store<f64, 1>) -> f64 {
            let v = num[g.x()] / den[g.x()];
            out[g.x()] = v;
            return v;
        }

        @binding(0, 0) var out_buf: buffer<f32>;
        @kernel @grid(1)
        fn k_div_buf(g: grid<1>, num: read_store<f32, 1>, den: read_store<f32, 1>) {
            out_buf[g.x()] = num[g.x()] / den[g.x()];
        }
    )";

    auto res = compile_source(src, "trap_nf.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);

    const auto* k_store = res.find_kernel("k_div_store");
    const auto* k_ret = res.find_kernel("k_div_ret");
    const auto* k_f64 = res.find_kernel("k_div_f64");
    assert(k_store && k_ret && k_f64);

    constexpr uint32_t W = 5, H = 4;
    std::vector<float> num(W * H, 2.0f);
    std::vector<float> den(W * H, 2.0f);
    std::vector<float> out(W * H, 0.0f);
    std::string err;

    // Finite inputs pass cleanly.
    CheckReport ok_rep = k_store->check_cpu(GridShape(W, H), {KernelArg::store(num), KernelArg::store(den), KernelArg::store(out)}, true, &err);
    assert(ok_rep.ok);
    CheckReport ok_ret = k_ret->check_cpu(GridShape(W, H), {KernelArg::store(num), KernelArg::store(den)}, true, &err);
    assert(ok_ret.ok);

    // 0.0 / 0.0 -> NaN at (3, 2) on store write.
    std::vector<float> num_nan = num, den_nan = den;
    num_nan[2 * W + 3] = 0.0f;
    den_nan[2 * W + 3] = 0.0f;
    CheckReport nan_rep = k_store->check_cpu(GridShape(W, H), {KernelArg::store(num_nan), KernelArg::store(den_nan), KernelArg::store(out)}, true, &err);
    assert(!nan_rep.ok);
    assert(nan_rep.kind == 2);
    assert(nan_rep.coord[0] == 3 && nan_rep.coord[1] == 2 && nan_rep.coord[2] == 0);
    assert(nan_rep.label == "out");
    assert(std::isnan(nan_rep.value));

    // 1.0 / 0.0 -> +Inf at (4, 1) on return value (vec2.y).
    std::vector<float> num_inf = num, den_inf = den;
    num_inf[1 * W + 4] = 1.0f;
    den_inf[1 * W + 4] = 0.0f;
    CheckReport inf_rep = k_ret->check_cpu(GridShape(W, H), {KernelArg::store(num_inf), KernelArg::store(den_inf)}, true, &err);
    assert(!inf_rep.ok);
    assert(inf_rep.kind == 2);
    assert(inf_rep.coord[0] == 4 && inf_rep.coord[1] == 1 && inf_rep.coord[2] == 0);
    assert(std::isinf(inf_rep.value) && inf_rep.value > 0.0);

    // f64 Inf on CPU @cpu kernel at x = 3.
    std::vector<double> dnum(6, 4.0), dden(6, 2.0), dout(6, 0.0);
    assert(k_f64->check_cpu(GridShape(6), {KernelArg::store(dnum), KernelArg::store(dden), KernelArg::store(dout)}, true, &err).ok);
    dnum[3] = -1.0;
    dden[3] = 0.0;
    CheckReport d_rep = k_f64->check_cpu(GridShape(6), {KernelArg::store(dnum), KernelArg::store(dden), KernelArg::store(dout)}, true, &err);
    assert(!d_rep.ok);
    assert(d_rep.kind == 2);
    assert(d_rep.coord[0] == 3 && d_rep.coord[1] == 0 && d_rep.coord[2] == 0);
    assert(std::isinf(d_rep.value) && d_rep.value < 0.0);

    // Direct compile with trap_non_finite = true also works and detects violations.
    CompileOptions nf_opts;
    nf_opts.trap_non_finite = true;
    auto res_nf = compile_source(src, "trap_nf_direct.bk", nf_opts);
    assert(res_nf.ok);
    const auto* k_nf = res_nf.find_kernel("k_div_store");
    assert(k_nf && k_nf->trap_non_finite_compiled);
    CheckReport direct_rep = k_nf->check_cpu(GridShape(W, H), {KernelArg::store(num_nan), KernelArg::store(den_nan), KernelArg::store(out)}, true, &err);
    assert(!direct_rep.ok && direct_rep.kind == 2 && direct_rep.coord[0] == 3 && direct_rep.coord[1] == 2);

    // buffer<f32> write trap_non_finite on CPU & GPU.
    const auto* k_buf = res_nf.find_kernel("k_div_buf");
    assert(k_buf != nullptr);
    std::vector<float> bnum = {1.0f, 0.0f, 2.0f, 3.0f};
    std::vector<float> bden = {1.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> bout(4, 0.0f);
    CheckReport buf_rep = k_buf->check_cpu(GridShape(4), {KernelArg::store(bnum), KernelArg::store(bden), KernelArg::store(bout)}, true, &err);
    assert(!buf_rep.ok && buf_rep.kind == 2 && buf_rep.coord[0] == 1 && buf_rep.label == "out_buf" && std::isnan(buf_rep.value));
    if (brass::gpu::vulkan_available()) {
        CheckReport g_buf_rep = k_buf->check_gpu(GridShape(4), {KernelArg::store(bnum), KernelArg::store(bden), KernelArg::store(bout)}, true, &err);
        assert(!g_buf_rep.ok && g_buf_rep.kind == 2 && g_buf_rep.coord[0] == 1 && g_buf_rep.label == "out_buf" && std::isnan(g_buf_rep.value));
    }

    if (brass::gpu::vulkan_available()) {
        assert(k_store->check_gpu(GridShape(W, H), {KernelArg::store(num), KernelArg::store(den), KernelArg::store(out)}, true, &err).ok);
        CheckReport g_nan = k_store->check_gpu(GridShape(W, H), {KernelArg::store(num_nan), KernelArg::store(den_nan), KernelArg::store(out)}, true, &err);
        assert(!g_nan.ok);
        assert(g_nan.kind == 2);
        assert(g_nan.coord[0] == 3 && g_nan.coord[1] == 2 && g_nan.coord[2] == 0);
        assert(g_nan.label == "out");
        assert(std::isnan(g_nan.value));

        CheckReport g_inf = k_ret->check_gpu(GridShape(W, H), {KernelArg::store(num_inf), KernelArg::store(den_inf)}, true, &err);
        assert(!g_inf.ok);
        assert(g_inf.kind == 2);
        assert(g_inf.coord[0] == 4 && g_inf.coord[1] == 1 && g_inf.coord[2] == 0);
        assert(std::isinf(g_inf.value) && g_inf.value > 0.0);

        // Normal dispatch_gpu with trap_non_finite = true must remain safe.
        assert(k_nf->dispatch_gpu(GridShape(W, H), {KernelArg::store(num), KernelArg::store(den), KernelArg::store(out)}, nullptr, &err));
    }
}

static void test_probe_if_cpu_and_gpu() {
    const char* src = R"(
        @kernel @grid(2)
        fn k_probe_if(g: grid<2>, inp: read_store<f32, 2>, out: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let v = inp[x, y];
            let kept = probe_if("outlier", v > 10.0, vec2<f32>(v, f32(x + y * 10)));
            let idx_val = probe_if("corner", x == 0 && y == 0, x + 42);
            out[x, y] = kept.x + f32(idx_val) * 0.0;
        }
    )";

    auto res = compile_source(src, "probe_if.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    const auto* k = res.find_kernel("k_probe_if");
    assert(k != nullptr);
    assert(!k->probes_compiled);
    assert(k->probes.size() == 2);
    assert(k->probes[0].is_conditional && k->probes[1].is_conditional);

    constexpr uint32_t W = 8, H = 4;
    std::vector<float> inp(W * H, 1.0f);
    inp[1 * W + 2] = 25.5f;
    inp[3 * W + 6] = 42.25f;
    std::vector<float> out(W * H, 0.0f);
    std::string err;

    // Normal dispatch preserves exact values.
    assert(k->dispatch_cpu(GridShape(W, H), {KernelArg::store(inp), KernelArg::store(out)}, nullptr, &err));
    assert(std::abs(out[1 * W + 2] - 25.5f) < 1e-5f);
    assert(std::abs(out[3 * W + 6] - 42.25f) < 1e-5f);

    std::vector<ConditionalProbeSample> samples;
    assert(k->probe_if_cpu(GridShape(W, H), {KernelArg::store(inp), KernelArg::store(out)}, samples, 16, &err));
    assert(samples.size() == 3);
    bool saw_corner = false, saw_outlier_1 = false, saw_outlier_2 = false;
    for (const auto& s : samples) {
        if (s.label == "corner") {
            saw_corner = true;
            assert(s.coord[0] == 0 && s.coord[1] == 0 && s.coord[2] == 0);
            assert(s.values.size() == 1 && std::abs(s.values[0] - 42.0) < 1e-6);
        } else if (s.label == "outlier") {
            assert(s.values.size() == 2);
            if (s.coord[0] == 2 && s.coord[1] == 1) {
                saw_outlier_1 = true;
                assert(std::abs(s.values[0] - 25.5) < 1e-5);
                assert(std::abs(s.values[1] - 12.0) < 1e-5);
            } else if (s.coord[0] == 6 && s.coord[1] == 3) {
                saw_outlier_2 = true;
                assert(std::abs(s.values[0] - 42.25) < 1e-5);
                assert(std::abs(s.values[1] - 36.0) < 1e-5);
            }
        }
    }
    assert(saw_corner && saw_outlier_1 && saw_outlier_2);

    // max_samples cap is respected cleanly without overflow.
    std::vector<ConditionalProbeSample> capped;
    assert(k->probe_if_cpu(GridShape(W, H), {KernelArg::store(inp), KernelArg::store(out)}, capped, 2, &err));
    assert(capped.size() == 2);

    if (brass::gpu::vulkan_available()) {
        std::vector<ConditionalProbeSample> gsamples;
        assert(k->probe_if_gpu(GridShape(W, H), {KernelArg::store(inp), KernelArg::store(out)}, gsamples, 16, &err));
        assert(gsamples.size() == 3);
        bool gsaw_c = false, gsaw_1 = false, gsaw_2 = false;
        for (const auto& s : gsamples) {
            if (s.label == "corner" && s.coord[0] == 0 && s.coord[1] == 0 && s.values.size() == 1 && std::abs(s.values[0] - 42.0) < 1e-4) {
                gsaw_c = true;
            }
            if (s.label == "outlier" && s.coord[0] == 2 && s.coord[1] == 1 && s.values.size() == 2 && std::abs(s.values[0] - 25.5) < 1e-4) {
                gsaw_1 = true;
            }
            if (s.label == "outlier" && s.coord[0] == 6 && s.coord[1] == 3 && s.values.size() == 2 && std::abs(s.values[0] - 42.25) < 1e-4) {
                gsaw_2 = true;
            }
        }
        assert(gsaw_c && gsaw_1 && gsaw_2);

        std::vector<ConditionalProbeSample> gcapped;
        assert(k->probe_if_gpu(GridShape(W, H), {KernelArg::store(inp), KernelArg::store(out)}, gcapped, 1, &err));
        assert(gcapped.size() == 1);
    }
}

static void test_compile_cache_separation() {
    const std::filesystem::path cache_dir = std::filesystem::temp_directory_path() / "brokernel_diag_cache_test";
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
    set_compile_cache(CompileCacheConfig{cache_dir, 32 * 1024 * 1024, "diag-test-v1"});

    const char* src = R"(
        @kernel @grid(1)
        fn k_cache(g: grid<1>, inp: read_store<f32, 1>, out: write_store<f32, 1>) {
            let x = inp[g.x()];
            check("nonneg", x >= 0.0);
            out[g.x()] = probe_if("big", x > 5.0, x * 2.0);
        }
    )";

    CompileOptions normal_opts;
    CompileOptions probed_opts;
    probed_opts.enable_probes = true;
    CompileOptions checked_opts;
    checked_opts.enable_checks = true;
    checked_opts.trap_non_finite = true;

    Compiler c;
    auto r0 = c.compile_source(src, "diag_cache.bk", normal_opts);
    assert(r0.ok && r0.cache_status == CacheStatus::Miss);
    auto r1 = c.compile_source(src, "diag_cache.bk", probed_opts);
    assert(r1.ok && r1.cache_status == CacheStatus::Miss);
    auto r2 = c.compile_source(src, "diag_cache.bk", checked_opts);
    assert(r2.ok && r2.cache_status == CacheStatus::Miss);

    auto r0_hit = c.compile_source(src, "diag_cache.bk", normal_opts);
    assert(r0_hit.ok && r0_hit.cache_status == CacheStatus::Hit);
    auto r1_hit = c.compile_source(src, "diag_cache.bk", probed_opts);
    assert(r1_hit.ok && r1_hit.cache_status == CacheStatus::Hit);
    auto r2_hit = c.compile_source(src, "diag_cache.bk", checked_opts);
    assert(r2_hit.ok && r2_hit.cache_status == CacheStatus::Hit);

    const auto* k_norm = r0_hit.find_kernel("k_cache");
    const auto* k_prob = r1_hit.find_kernel("k_cache");
    const auto* k_chk = r2_hit.find_kernel("k_cache");
    assert(k_norm && !k_norm->probes_compiled && !k_norm->checks_compiled && !k_norm->trap_non_finite_compiled);
    assert(k_prob && k_prob->probes_compiled && k_prob->probes.size() == 1 && k_prob->probes[0].is_conditional);
    assert(k_chk && k_chk->checks_compiled && k_chk->trap_non_finite_compiled && k_chk->check_labels.size() >= 1);

    // Lazy check_cpu & probe_if_cpu work on cache-hit normal kernel!
    std::vector<float> inp = {1.0f, -2.0f, 8.0f, 3.0f};
    std::vector<float> out(4, 0.0f);
    std::string err;
    CheckReport rep = k_norm->check_cpu(GridShape(4), {KernelArg::store(inp), KernelArg::store(out)}, false, &err);
    assert(!rep.ok && rep.kind == 1 && rep.coord[0] == 1 && rep.label == "nonneg");

    std::vector<ConditionalProbeSample> pif;
    assert(k_norm->probe_if_cpu(GridShape(4), {KernelArg::store(inp), KernelArg::store(out)}, pif, 8, &err));
    assert(pif.size() == 1 && pif[0].coord[0] == 2 && std::abs(pif[0].values[0] - 16.0) < 1e-5);

    set_compile_cache(CompileCacheConfig{});
    std::filesystem::remove_all(cache_dir, ec);
}

static void test_parity_report_worst_coord() {
    const char* src = R"(
        @kernel @grid(2)
        fn k_store_2d(g: grid<2>, out: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            var v: f32 = f32(x + y * 10);
            if (x == 4 && y == 2) {
                v = 0.75;
                for (var i: i32 = 0; i < 8; i = i + 1) {
                    v = sin(v * 1.7 + 0.3) * 0.5 + cos(v * 2.1 - 0.4) * 0.5 + exp(-abs(v) * 0.25) * 0.1;
                }
            }
            out[x, y] = v;
        }

        @kernel @grid(3)
        fn k_ret_3d(g: grid<3>) -> vec2<f32> {
            let x = g.x();
            let y = g.y();
            let z = g.z();
            var v: f32 = f32(x + y * 4 + z * 12);
            if (x == 2 && y == 1 && z == 1) {
                v = 0.75;
                for (var i: i32 = 0; i < 8; i = i + 1) {
                    v = sin(v * 1.7 + 0.3) * 0.5 + cos(v * 2.1 - 0.4) * 0.5 + exp(-abs(v) * 0.25) * 0.1;
                }
            }
            return vec2<f32>(1.0, v);
        }
    )";

    auto res = compile_source(src, "parity_coord.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);

    const auto* k2 = res.find_kernel("k_store_2d");
    const auto* k3 = res.find_kernel("k_ret_3d");
    assert(k2 && k3);

    if (!brass::gpu::vulkan_available()) {
        std::vector<float> out(6 * 4, 0.0f);
        ParityReport rep = k2->compare_cpu_gpu(GridShape(6, 4), {KernelArg::store(out)}, 0.0, 0.0);
        assert(!rep.ran_gpu && !rep.ok);
        return;
    }

    // 1. Store mismatch at (4, 2, 0) under zero tolerance.
    std::vector<float> out(6 * 4, 0.0f);
    ParityReport rep2 = k2->compare_cpu_gpu(GridShape(6, 4), {KernelArg::store(out)}, 0.0, 0.0);
    assert(rep2.ran_gpu);
    assert(rep2.elements_compared == 6 * 4);
    assert(rep2.worst_coord[0] == 4 && rep2.worst_coord[1] == 2 && rep2.worst_coord[2] == 0);
    assert(rep2.cpu_val != 0.0 && rep2.gpu_val != 0.0);
    if (!rep2.ok) {
        assert(rep2.cpu_val != rep2.gpu_val);
        assert(rep2.message.find("(4, 2, 0)") != std::string::npos);
        assert(rep2.message.find("cpu=") != std::string::npos);
        assert(rep2.message.find("gpu=") != std::string::npos);
    }

    // With normal tolerance, both pass and worst_coord tracks max_abs_diff cell (4, 2, 0).
    ParityReport rep2_ok = k2->compare_cpu_gpu(GridShape(6, 4), {KernelArg::store(out)}, 1e-4, 1e-5);
    assert(rep2_ok.ran_gpu && rep2_ok.ok);
    assert(rep2_ok.worst_coord[0] == 4 && rep2_ok.worst_coord[1] == 2 && rep2_ok.worst_coord[2] == 0);

    // 2. Return-store vec2<f32> 3D mismatch at (2, 1, 1).
    ParityReport rep3 = k3->compare_cpu_gpu(GridShape(4, 3, 2), {}, 0.0, 0.0);
    assert(rep3.ran_gpu);
    assert(rep3.elements_compared == 4 * 3 * 2 * 2);
    assert(rep3.worst_coord[0] == 2 && rep3.worst_coord[1] == 1 && rep3.worst_coord[2] == 1);
    if (!rep3.ok) {
        assert(rep3.cpu_val != rep3.gpu_val);
        assert(rep3.message.find("(2, 1, 1)") != std::string::npos);
    }
}

int main() {
    test_check_cpu_and_gpu();
    test_trap_non_finite();
    test_probe_if_cpu_and_gpu();
    test_compile_cache_separation();
    test_parity_report_worst_coord();
    std::cout << "All diagnostics tests passed!\n";
    return 0;
}

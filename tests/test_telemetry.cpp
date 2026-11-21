// Tests for runtime multi-entity telemetry (`telemetry(slot, val)`) and AstCache.
#include "brokernel/brokernel.h"
#include "../src/ast_cache.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr const char* kTelemetrySrc = R"bk(
@knob
const SCALE: f32 = 2.0;

fn helper_record(idx: i32, v: f32) {
    telemetry(idx, v * SCALE);
    telemetry(-1, 999.0);
    telemetry(99, 888.0);
}

@kernel @grid(1)
fn k_telemetry(g: grid<1>, out: write_store<f32, 1>) {
    let x = g.x();
    let val = f32(x + 1);
    helper_record(x, val);
    let pv = probe("val", val);
    out[x] = pv * SCALE;
}
)bk";

constexpr const char* kAstBenchSrc = R"bk(
import std::noise;
import std::math;
import std::filter;
import std::color;

@knob
const GAIN: f32 = 1.5;

@kernel @grid(2)
fn k_bench(g: grid<2>, out: write_store<f32, 2>) {
    let p = vec2<f32>(f32(g.x()) * 0.05, f32(g.y()) * 0.05);
    out[g.x(), g.y()] = (noise::perlin2d(p, 42) + 1.0) * 0.5 * GAIN;
}
)bk";

bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void test_telemetry_disabled_and_null_safe() {
    brokernel::Compiler c;
    brokernel::CompileOptions opts;
    opts.telemetry_slots = 0;
    auto res = c.compile_source(kTelemetrySrc, "telem_off.bk", opts);
    CHECK(res.ok);
    const auto* k = res.find_kernel("k_telemetry");
    CHECK(k != nullptr);
    CHECK(k->telemetry_slots == 0);
    CHECK(k->uses_telemetry);
    std::vector<float> out(8, 0.0f);
    CHECK(k->dispatch_cpu(brokernel::GridShape(8), {brokernel::KernelArg::store(out)}, nullptr, nullptr));
    for (int i = 0; i < 8; ++i) {
        CHECK(approx(out[i], float(i + 1) * 2.0f));
    }

    // Enabled telemetry with null buffer must remain safe on CPU & GPU.
    opts.telemetry_slots = 16;
    auto res_on = c.compile_source(kTelemetrySrc, "telem_null.bk", opts);
    CHECK(res_on.ok);
    const auto* kon = res_on.find_kernel("k_telemetry");
    CHECK(kon != nullptr);
    CHECK(kon->telemetry_slots == 16);
    CHECK(kon->uses_telemetry);
    std::fill(out.begin(), out.end(), 0.0f);
    CHECK(kon->dispatch_cpu(brokernel::GridShape(8), {brokernel::KernelArg::store(out)}, nullptr, nullptr));
    for (int i = 0; i < 8; ++i) {
        CHECK(approx(out[i], float(i + 1) * 2.0f));
    }
}

void test_telemetry_cpu_and_probe_knobs_coexistence() {
    brokernel::Compiler c;
    brokernel::CompileOptions opts;
    opts.telemetry_slots = 12;
    opts.runtime_knobs = true;
    opts.enable_probes = true;
    auto res = c.compile_source(kTelemetrySrc, "telem_coexist.bk", opts);
    CHECK(res.ok);
    const auto* k = res.find_kernel("k_telemetry");
    CHECK(k != nullptr);
    CHECK(k->probes_compiled);
    CHECK(!k->runtime_knobs.empty());

    std::vector<float> telem(12, -1.0f);
    std::vector<float> out(16, 0.0f);
    k->set_telemetry_buffer(telem.data(), static_cast<uint32_t>(telem.size()), 0);
    CHECK(k->dispatch_cpu(brokernel::GridShape(16), {brokernel::KernelArg::store(out)}, nullptr, nullptr));
    // Slots 0..11 written; out-of-range slots (-1, 12..15, 99) safely skipped.
    for (int i = 0; i < 12; ++i) {
        CHECK(approx(telem[i], float(i + 1) * 2.0f));
    }

    // Mutate runtime knob and verify telemetry updates without recompile.
    CHECK(k->set_knobs({{"SCALE", brokernel::ConstValue::from_f32(3.0f)}}));
    std::fill(telem.begin(), telem.end(), 0.0f);
    CHECK(k->dispatch_cpu(brokernel::GridShape(16), {brokernel::KernelArg::store(out)}, nullptr, nullptr));
    for (int i = 0; i < 12; ++i) {
        CHECK(approx(telem[i], float(i + 1) * 3.0f));
    }

    // Probe must still work accurately alongside runtime knobs and telemetry.
    std::vector<brokernel::ProbeSample> samples;
    CHECK(k->probe_cpu(5, 0, 0, brokernel::GridShape(16), {brokernel::KernelArg::store(out)}, samples, nullptr));
    CHECK(samples.size() == 1);
    CHECK(samples[0].hit);
    CHECK(approx(samples[0].values[0], 6.0f));

    if (brass::gpu::vulkan_available()) {
        // Test GPU host-bounce telemetry AND probe + runtime knobs on GPU.
        std::fill(telem.begin(), telem.end(), -5.0f);
        std::fill(out.begin(), out.end(), 0.0f);
        std::string err;
        CHECK(k->dispatch_gpu(brokernel::GridShape(16), {brokernel::KernelArg::store(out)}, nullptr, &err));
        for (int i = 0; i < 12; ++i) {
            CHECK(approx(telem[i], float(i + 1) * 3.0f));
        }
        for (int i = 0; i < 16; ++i) {
            CHECK(approx(out[i], float(i + 1) * 3.0f));
        }
        std::vector<brokernel::ProbeSample> gpu_samples;
        CHECK(k->probe_gpu(4, 0, 0, brokernel::GridShape(16), {brokernel::KernelArg::store(out)}, gpu_samples, &err));
        CHECK(gpu_samples.size() == 1);
        CHECK(gpu_samples[0].hit);
        CHECK(approx(gpu_samples[0].values[0], 5.0f));
    }
}

void test_ast_cache_and_cold_compile_benchmark() {
    using clock = std::chrono::steady_clock;
    constexpr int kIters = 20;

    auto bench = [&](bool cache_on, brokernel::CompileResult::CompileTimings& sum_t, double& ctor_ms) -> double {
        brokernel::detail::clear_ast_cache();
        brokernel::detail::set_ast_cache_enabled(cache_on);
        sum_t = {};
        ctor_ms = 0.0;
        const auto t0 = clock::now();
        for (int i = 0; i < kIters; ++i) {
            const auto tc0 = clock::now();
            brokernel::Compiler c;
            const auto tc1 = clock::now();
            ctor_ms += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
            brokernel::CompileOptions opts;
            opts.compile_cpu = false;
            opts.knob_overrides["GAIN"] = brokernel::ConstValue::from_f32(1.0f + 0.01f * float(i));
            auto r = c.compile_source(kAstBenchSrc, "bench_ast.bk", opts);
            CHECK(r.ok);
            sum_t.parse_ms += r.timings.parse_ms;
            sum_t.sema_ms += r.timings.sema_ms;
            sum_t.lower_ms += r.timings.lower_ms;
            sum_t.gpu_opt_ms += r.timings.gpu_opt_ms;
            sum_t.spirv_ms += r.timings.spirv_ms;
            sum_t.total_ms += r.timings.total_ms;
        }
        const auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    brokernel::CompileResult::CompileTimings t_off{}, t_on{};
    double ctor_off = 0.0, ctor_on = 0.0;
    const double off_ms = bench(false, t_off, ctor_off);
    const double on_ms = bench(true, t_on, ctor_on);
    const auto stats = brokernel::detail::ast_cache_stats();
    CHECK(stats.hits > 0);
    CHECK(stats.entries >= 1);

    // Changing source must miss and invalidate cleanly.
    std::string modified = std::string(kAstBenchSrc) + "\n// changed\n";
    brokernel::Compiler c2;
    brokernel::CompileOptions opts2;
    opts2.compile_cpu = false;
    auto r2 = c2.compile_source(modified, "bench_ast.bk", opts2);
    CHECK(r2.ok);
    const auto stats2 = brokernel::detail::ast_cache_stats();
    CHECK(stats2.misses > stats.misses);

    brokernel::detail::set_ast_cache_enabled(true);
    std::printf(
        "cold_compile_%d_iters: ast_cache_off=%.2f ms, ast_cache_on=%.2f ms (hits=%llu, misses=%llu)\n"
        "  stages_on_ms : ctor=%.2f parse=%.2f sema=%.2f lower=%.2f gpu_opt=%.2f spirv=%.2f total_uncached=%.2f\n",
        kIters, off_ms, on_ms,
        static_cast<unsigned long long>(stats.hits),
        static_cast<unsigned long long>(stats.misses),
        ctor_on, t_on.parse_ms, t_on.sema_ms, t_on.lower_ms, t_on.gpu_opt_ms, t_on.spirv_ms, t_on.total_ms);
}

} // namespace

int main() {
    test_telemetry_disabled_and_null_safe();
    test_telemetry_cpu_and_probe_knobs_coexistence();
    test_ast_cache_and_cold_compile_benchmark();
    return 0;
}

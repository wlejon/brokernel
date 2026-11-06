// Tests for agent-focused wishlist features:
// 1. Re-exports (`pub import`, `pub use`, `use`, `pub from ... import`)
// 2. Dynamic module resolver (`Compiler::set_module_resolver`)
// 3. Shared `pub const` reflection & struct layout reflection
// 4. Knobs without source edits (`@knob const`, `CompileResult::with_knobs`)
// 5. Named probes (`probe("label", expr)` on CPU, GPU compute, and fragment stages)
// 6. Forward-mode automatic differentiation (`grad(f, x, ...)`)
// 7. Quadrature & moments (`expect`, `moments`, `cell_expect`)

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace brokernel;

static void test_reexports_and_use_synonym() {
    Compiler compiler;
    compiler.register_module("world::leaf", R"(
        module world::leaf;
        pub const LEAF_SCALE: f32 = 2.5;
        pub struct LeafData {
            base: f32,
            gain: f32,
        }
        pub fn eval_leaf(d: LeafData, x: f32) -> f32 {
            return d.base + x * d.gain * LEAF_SCALE;
        }
    )");

    compiler.register_module("world::mid", R"(
        module world::mid;
        pub use world::leaf::*;
        pub const MID_BIAS: f32 = 10.0;
    )");

    compiler.register_module("world::facade", R"(
        module world::facade;
        pub from world::mid import eval_leaf, LeafData, LEAF_SCALE;
        pub use world::mid::{MID_BIAS};
    )");

    const char* src = R"(
        use world::facade::*;
        pub use world::facade::{LEAF_SCALE};

        @kernel fn run_facade(x: f32) -> f32 {
            let d = LeafData { base: MID_BIAS, gain: 4.0 };
            return eval_leaf(d, x);
        }
    )";

    auto res = compiler.compile_source(src, "reexport.bk");
    if (!res.ok) {
        std::cerr << "test_reexports_and_use_synonym failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.find_kernel("run_facade");
    assert(k != nullptr);
    // 10.0 + 3.0 * 4.0 * 2.5 = 40.0
    float v = k->eval_f32({KernelArg(3.0f)});
    assert(std::abs(v - 40.0f) < 1e-5f);

    // Check that re-exported `pub use world::facade::{LEAF_SCALE}` is in `res.constants`
    bool found_leaf_scale = false;
    for (const auto& c : res.constants) {
        if (c.name == "LEAF_SCALE") {
            found_leaf_scale = true;
            assert(std::abs(c.value.float_val - 2.5) < 1e-6);
        }
    }
    assert(found_leaf_scale);
}

static void test_module_resolver() {
    Compiler compiler;
    int resolve_calls = 0;
    compiler.set_module_resolver([&](std::string_view name) -> std::optional<std::string> {
        ++resolve_calls;
        if (name == "dyn::math_ext" || name == "dyn/math_ext") {
            return R"(
                module dyn::math_ext;
                pub const GOLDEN: f32 = 1.61803398875;
                pub fn triple(x: f32) -> f32 {
                    return x * 3.0;
                }
            )";
        }
        return std::nullopt;
    });

    const char* src = R"(
        use dyn::math_ext::{triple, GOLDEN};

        @kernel fn k_dyn(x: f32) -> f32 {
            return triple(x) + GOLDEN;
        }
    )";

    auto res = compiler.compile_source(src, "dyn_test.bk");
    if (!res.ok) {
        std::cerr << "test_module_resolver failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);
    assert(resolve_calls >= 1);
    const auto* k = res.find_kernel("k_dyn");
    assert(k != nullptr);
    float got = k->eval_f32({KernelArg(2.0f)});
    assert(std::abs(got - (6.0f + 1.61803398875f)) < 1e-5f);
}

static void test_constants_and_struct_layout() {
    const char* src = R"(
        pub const MAX_STEPS: i32 = 16;
        pub const SUN_DIR: vec3<f32> = vec3<f32>(0.0, 0.8, 0.6);
        pub const WEIGHTS: array<f32, 3> = array<f32, 3>(0.25, 0.5, 0.25);

        pub struct AtmosphereParams {
            planet_radius: f32,
            sun_dir: vec3<f32>,
            rayleigh: vec3<f32>,
            mie_g: f32,
        }

        @kernel fn dummy(p: AtmosphereParams) -> f32 {
            return p.planet_radius + p.mie_g + SUN_DIR.y + WEIGHTS[1] + (MAX_STEPS as f32);
        }
    )";

    auto res = compile_source(src, "consts_layout.bk");
    if (!res.ok) {
        std::cerr << "test_constants_and_struct_layout failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);
    assert(res.constants.size() == 3);

    const StructDef* sdef = res.types.find_struct("AtmosphereParams");
    assert(sdef != nullptr);
    Type st = Type::make_struct(sdef->id, sdef->name);
    assert(res.types.byte_size(st) == 32u);
    auto l140 = res.types.block_layout(st, true);
    auto l430 = res.types.block_layout(st, false);
    assert(l140.align == 16u);
    assert(l430.align == 16u);
    assert(res.types.block_field_offset(*sdef, "planet_radius", false) == 0u);
    assert(res.types.block_field_offset(*sdef, "sun_dir", false) == 16u);
    assert(res.types.block_field_offset(*sdef, "rayleigh", false) == 32u);
    assert(res.types.block_field_offset(*sdef, "mie_g", false) == 44u);
}

static void test_knobs() {
    const char* src = R"(
        @knob pub const CU_CAP: f32 = 0.06;
        @knob const GAIN: f32 = 2.0;

        @kernel fn tune_me(x: f32) -> f32 {
            return min(x * GAIN, CU_CAP);
        }
    )";

    auto res0 = compile_source(src, "knobs.bk");
    if (!res0.ok) {
        std::cerr << "test_knobs failed:\n" << res0.error_message() << "\n";
    }
    assert(res0.ok);
    const auto* k0 = res0.find_kernel("tune_me");
    assert(k0 != nullptr);
    assert(std::abs(k0->eval_f32({KernelArg(0.05f)}) - 0.06f) < 1e-6f);
    assert(std::abs(k0->eval_f32({KernelArg(0.02f)}) - 0.04f) < 1e-6f);

    // Override via with_knobs
    std::unordered_map<std::string, ConstValue> overrides;
    overrides["CU_CAP"] = ConstValue::from_f32(0.25f);
    overrides["GAIN"] = ConstValue::from_f32(3.0f);
    auto res1 = res0.with_knobs(overrides);
    assert(res1.ok);
    const auto* k1 = res1.find_kernel("tune_me");
    assert(k1 != nullptr);
    // min(0.05 * 3.0, 0.25) = 0.15
    assert(std::abs(k1->eval_f32({KernelArg(0.05f)}) - 0.15f) < 1e-6f);
}

static void test_probes() {
    const char* src = R"(
        fn helper(v: f32) -> f32 {
            let scaled = probe("scaled", v * 2.5);
            return scaled + 1.0;
        }

        @kernel @grid(2)
        fn k_probe_grid(g: grid<2>, out: write_store<f32, 2>, factor: f32) {
            let uv = probe("uv", vec2<f32>(g.x() as f32, g.y() as f32));
            let h = helper((uv.x + uv.y * 10.0) * factor);
            out[g.x(), g.y()] = probe("final_h", h);
        }
    )";

    // Default compile (enable_probes == false): zero-cost in normal runs,
    // and on-demand probe variant when probe_cpu / probe_gpu is called.
    auto res = compile_source(src, "probes.bk");
    if (!res.ok) {
        std::cerr << "test_probes failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);
    assert(res.probes.size() == 3);
    const auto* k = res.find_kernel("k_probe_grid");
    assert(k != nullptr);
    assert(k->probes.size() == 3);

    std::vector<float> out_buf(16, 0.0f);
    std::vector<ProbeSample> samples;
    std::string err;
    // Probe cell (2, 3) on a 4x4 grid with factor = 0.5:
    // uv = (2, 3), v = (2 + 30) * 0.5 = 16, scaled = 40, final_h = 41
    bool ok = k->probe_cpu(2, 3, 0, GridShape(4, 4, 1),
                           {KernelArg::store(out_buf), KernelArg(0.5f)},
                           samples, &err);
    if (!ok) std::cerr << "probe_cpu error: " << err << "\n";
    assert(ok);
    assert(samples.size() == 3);

    bool saw_uv = false, saw_scaled = false, saw_final = false;
    for (const auto& s : samples) {
        assert(s.hit);
        if (s.name == "uv") {
            saw_uv = true;
            assert(s.lanes == 2);
            assert(std::abs(s.values[0] - 2.0f) < 1e-5f);
            assert(std::abs(s.values[1] - 3.0f) < 1e-5f);
        } else if (s.name == "scaled") {
            saw_scaled = true;
            assert(s.lanes == 1);
            assert(std::abs(s.values[0] - 40.0f) < 1e-5f);
        } else if (s.name == "final_h") {
            saw_final = true;
            assert(s.lanes == 1);
            assert(std::abs(s.values[0] - 41.0f) < 1e-5f);
        }
    }
    assert(saw_uv && saw_scaled && saw_final);

    if (brass::gpu::vulkan_available()) {
        std::vector<ProbeSample> gpu_samples;
        bool gok = k->probe_gpu(2, 3, 0, GridShape(4, 4, 1),
                                {KernelArg::store(out_buf), KernelArg(0.5f)},
                                gpu_samples, &err);
        if (!gok) std::cerr << "probe_gpu error: " << err << "\n";
        assert(gok);
        for (const auto& s : gpu_samples) {
            assert(s.hit);
            if (s.name == "scaled") assert(std::abs(s.values[0] - 40.0f) < 1e-4f);
            if (s.name == "final_h") assert(std::abs(s.values[0] - 41.0f) < 1e-4f);
        }
    }

    // Fragment stage probe compilation check (`@binding(4, 23)`)
    const char* frag_src = R"(
        @fragment fn fs_main(@location(0) uv: vec2<f32>) {
            let c = probe("albedo", vec3<f32>(uv.x, uv.y, 0.5));
            set_frag_color(vec4<f32>(c.x, c.y, c.z, 1.0));
        }
    )";
    CompileOptions fopts;
    fopts.enable_fragment_probes = true;
    auto fres = compile_source(frag_src, "frag_probe.bk", fopts);
    if (!fres.ok) {
        std::cerr << "fragment probe compile failed:\n" << fres.error_message() << "\n";
    }
    assert(fres.ok);
    const auto* fk = fres.find_kernel("fs_main");
    assert(fk != nullptr);
    assert(!fk->spirv.words.empty());

    // A user function named probe (tiles::probe is one) stays a different
    // function from the builtin: the probe beside a call of it is the
    // builtin's, and the call reads its buffer argument.
    const char* clash_src = R"(
        fn probe(table: read_buffer<f32>, i: i32) -> f32 {
            return table[i] * 2.0;
        }
        @kernel @grid(1)
        fn k_clash(g: grid<1>, table: read_buffer<f32>, out: write_store<f32>) {
            let v = probe(table, g.x() as i32);
            out[g.x()] = probe("v", v);
        }
    )";
    auto cres = compile_source(clash_src, "probe_clash.bk");
    if (!cres.ok) std::cerr << "probe beside a user fn named probe failed:\n" << cres.error_message() << "\n";
    assert(cres.ok);
    const auto* ck = cres.find_kernel("k_clash");
    assert(ck != nullptr && ck->probes.size() == 1);
    std::vector<float> table{1.0f, 2.0f, 3.0f};
    std::vector<float> clash_out(3, 0.0f);
    std::vector<ProbeSample> clash_samples;
    bool cok = ck->probe_cpu(1, 0, 0, GridShape(3, 1, 1),
                             {KernelArg::store(table), KernelArg::store(clash_out)},
                             clash_samples, &err);
    if (!cok) std::cerr << "probe_cpu error: " << err << "\n";
    assert(cok && clash_samples.size() == 1);
    assert(clash_samples[0].hit && std::abs(clash_samples[0].values[0] - 4.0f) < 1e-5f);
}

static void test_grad_forward_ad() {
    const char* src = R"(
        struct WaveCfg {
            freq: f32,
            amp: f32,
        }

        fn poly_wave(x: f32, cfg: WaveCfg) -> f32 {
            var acc: f32 = 0.0;
            for (var i: i32 = 1; i <= 3; i += 1) {
                let fi = i as f32;
                acc += sin(x * cfg.freq * fi) / fi;
            }
            return cfg.amp * acc + x * x * x;
        }

        fn field3d(p: vec3<f32>, scale: f32) -> f32 {
            let q = p * scale;
            return dot(q, q) + sin(q.x) * cos(q.y) + exp(q.z * 0.5);
        }

        fn field3d_f64(p: vec3<f64>, scale: f64) -> f64 {
            let q = p * scale;
            return dot(q, q) + sin(q.x) * cos(q.y);
        }

        @kernel fn d_poly_wave(x: f32, freq: f32, amp: f32) -> f32 {
            let cfg = WaveCfg { freq: freq, amp: amp };
            return grad(poly_wave, x, cfg);
        }

        @kernel @grid(1)
        fn k_grad_field3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>,
                          gx_out: write_store<f32, 1>, gy_out: write_store<f32, 1>, gz_out: write_store<f32, 1>,
                          scale: f32) {
            let i = g.x();
            let p = vec3<f32>(px[i], py[i], pz[i]);
            let gr = grad(field3d, p, scale);
            gx_out[i] = gr.x;
            gy_out[i] = gr.y;
            gz_out[i] = gr.z;
        }

        @kernel @cpu fn d_field3d_f64(x: f64, y: f64, z: f64, scale: f64) -> f64 {
            let g = grad(field3d_f64, vec3<f64>(x, y, z), scale);
            return g.x + g.y + g.z;
        }
    )";

    auto res = compile_source(src, "grad_test.bk");
    if (!res.ok) {
        std::cerr << "test_grad_forward_ad failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);

    // 1. Scalar derivative check
    const auto* kw = res.find_kernel("d_poly_wave");
    assert(kw != nullptr);
    const float x = 0.7f, freq = 1.5f, amp = 2.0f;
    float got_dw = kw->eval_f32({KernelArg(x), KernelArg(freq), KernelArg(amp)});
    float expected_dw = amp * freq * (std::cos(x * freq * 1.0f) + std::cos(x * freq * 2.0f) + std::cos(x * freq * 3.0f))
                      + 3.0f * x * x;
    assert(std::abs(got_dw - expected_dw) < 1e-4f);

    // 2. Vector gradient check (CPU & GPU parity)
    const auto* kg = res.find_kernel("k_grad_field3d");
    assert(kg != nullptr);
    std::vector<float> px = {0.4f, -0.7f, 1.2f, 0.1f};
    std::vector<float> py = {0.9f, 0.3f, -0.5f, 1.1f};
    std::vector<float> pz = {-0.2f, 0.8f, 0.6f, -0.4f};
    std::vector<float> gx(4), gy(4), gz(4);
    const float sc = 1.25f;
    bool ok = kg->dispatch_cpu(GridShape(4, 1, 1),
                               {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz),
                                KernelArg::store(gx), KernelArg::store(gy), KernelArg::store(gz),
                                KernelArg(sc)});
    assert(ok);
    for (int i = 0; i < 4; ++i) {
        float qx = px[i] * sc, qy = py[i] * sc, qz = pz[i] * sc;
        float ex = sc * (2.0f * qx + std::cos(qx) * std::cos(qy));
        float ey = sc * (2.0f * qy - std::sin(qx) * std::sin(qy));
        float ez = sc * (2.0f * qz + 0.5f * std::exp(qz * 0.5f));
        assert(std::abs(gx[i] - ex) < 1e-4f);
        assert(std::abs(gy[i] - ey) < 1e-4f);
        assert(std::abs(gz[i] - ez) < 1e-4f);
    }

    if (brass::gpu::vulkan_available()) {
        auto cmp = kg->compare_cpu_gpu(
            GridShape(4, 1, 1),
            {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz),
             KernelArg::store(gx), KernelArg::store(gy), KernelArg::store(gz),
             KernelArg(sc)},
            1e-4, 1e-4);
        assert(cmp.ok);
    }

    // 3. f64 gradient check
    const auto* kd64 = res.find_kernel("d_field3d_f64");
    assert(kd64 != nullptr);
    double g64 = kd64->eval_f64({KernelArg(0.5), KernelArg(0.8), KernelArg(-0.3), KernelArg(2.0)});
    double qx = 1.0, qy = 1.6, qz = -0.6;
    double ex64 = 2.0 * (2.0 * qx + std::cos(qx) * std::cos(qy))
                + 2.0 * (2.0 * qy - std::sin(qx) * std::sin(qy))
                + 2.0 * (2.0 * qz);
    assert(std::abs(g64 - ex64) < 1e-11);
}

static void test_quadrature_expect_moments_cell() {
    const char* src = R"(
        fn cubic(x: f32, a: f32, b: f32) -> f32 {
            return a * x * x * x + b * x * x + 2.0 * x + 5.0;
        }

        fn linear_fn(x: f32, slope: f32, bias: f32) -> f32 {
            return slope * x + bias;
        }

        fn patch_fn(p: vec2<f32>, k: f32) -> f32 {
            return k * (p.x * p.x + p.y * p.y);
        }

        @kernel fn k_expect(mu: f32, sigma: f32, a: f32, b: f32) -> f32 {
            return expect(cubic, mu, sigma, a, b);
        }

        @kernel fn k_moments(mu: f32, sigma: f32, slope: f32, bias: f32) -> f32 {
            let m = moments(linear_fn, mu, sigma, slope, bias);
            return m.x * 100.0 + m.y;
        }

        @kernel fn k_cell_expect(cx: f32, cy: f32, L: f32, k: f32) -> f32 {
            return cell_expect(patch_fn, vec2<f32>(cx, cy), vec2<f32>(L, 0.0), vec2<f32>(0.0, L), k);
        }
    )";

    auto res = compile_source(src, "quadrature_test.bk");
    if (!res.ok) {
        std::cerr << "test_quadrature_expect_moments_cell failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);

    // 1. `expect` on cubic polynomial: E[a X^3 + b X^2 + 2 X + 5] = a(mu^3 + 3*mu*sigma^2) + b(mu^2 + sigma^2) + 2*mu + 5
    const auto* ke = res.find_kernel("k_expect");
    assert(ke != nullptr);
    float mu = 1.5f, sigma = 0.8f, a = 0.5f, b = -1.2f;
    float got_e = ke->eval_f32({KernelArg(mu), KernelArg(sigma), KernelArg(a), KernelArg(b)});
    float exact_e = a * (mu * mu * mu + 3.0f * mu * sigma * sigma)
                  + b * (mu * mu + sigma * sigma)
                  + 2.0f * mu + 5.0f;
    assert(std::abs(got_e - exact_e) < 1e-4f);

    // 2. `moments` on linear function: E = slope * mu + bias, Var = slope^2 * sigma^2
    const auto* km = res.find_kernel("k_moments");
    assert(km != nullptr);
    float slope = 3.0f, bias = -2.0f;
    float got_m = km->eval_f32({KernelArg(mu), KernelArg(sigma), KernelArg(slope), KernelArg(bias)});
    float exact_mean = slope * mu + bias;
    float exact_var = slope * slope * sigma * sigma;
    assert(std::abs(got_m - (exact_mean * 100.0f + exact_var)) < 1e-3f);

    // 3. `cell_expect` on k * (x^2 + y^2) over square of side L centered at (cx, cy):
    // E = k * (cx^2 + cy^2 + L^2 / 6)
    const auto* kc = res.find_kernel("k_cell_expect");
    assert(kc != nullptr);
    float cx = 2.0f, cy = -1.5f, L = 3.0f, k = 2.0f;
    float got_c = kc->eval_f32({KernelArg(cx), KernelArg(cy), KernelArg(L), KernelArg(k)});
    float exact_c = k * (cx * cx + cy * cy + (L * L) / 6.0f);
    assert(std::abs(got_c - exact_c) < 1e-4f);
}

int main() {
    test_reexports_and_use_synonym();
    test_module_resolver();
    test_constants_and_struct_layout();
    test_knobs();
    test_probes();
    test_grad_forward_ad();
    test_quadrature_expect_moments_cell();
    std::cout << "All wishlist tests passed!\n";
    return 0;
}

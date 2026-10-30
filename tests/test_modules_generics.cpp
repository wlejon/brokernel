// Module system and generic monomorphization tests for brokernel.

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace brokernel;

static void test_builtin_stdlib_modules() {
    const char* src = R"(
        import std::math;
        import std::noise::{fbm, simplex3d};
        import std::sphere;
        import std::color;

        @kernel fn planet_height(lat: f64, lon: f64) -> f64 {
            let p: vec3<f64> = sphere::latlon_to_unit<f64>(lat, lon);
            let h: f64 = fbm<f64, 3>(p * 2.0, 4, 2.0, 0.5);
            return math::remap<f64>(h, -1.0, 1.0, -4000.0, 8000.0);
        }

        @kernel fn tonemap_pixel(r: f32, g: f32, b: f32) -> f32 {
            let hdr = vec3<f32>(r, g, b);
            let ldr = color::aces_tonemap<f32>(hdr);
            return color::luminance<f32>(ldr);
        }
    )";

    auto res = compile_source(src, "planet.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    assert(res.kernels.size() == 2);

    const auto* kh = res.find_kernel("planet_height");
    assert(kh != nullptr);
    assert(kh->uses_f64);
    double h0 = kh->eval_f64({KernelArg(0.25), KernelArg(1.1)});
    double h1 = kh->eval_f64({KernelArg(-0.4), KernelArg(2.3)});
    assert(h0 >= -5000.0 && h0 <= 9000.0);
    assert(std::abs(h0 - h1) > 1.0);

    const auto* kt = res.find_kernel("tonemap_pixel");
    assert(kt != nullptr);
    float lum = kt->eval_f32({KernelArg(1.2f), KernelArg(0.8f), KernelArg(0.4f)});
    assert(lum > 0.0f && lum < 1.0f);
}

static void test_custom_and_filesystem_modules_and_cycles() {
    Compiler compiler;
    compiler.register_module("biome::moisture", R"(
        module biome::moisture;
        import std::math;

        pub struct ClimateSample<T> {
            temp: T,
            rain: T,
        }

        pub fn aridity_index<T>(s: ClimateSample<T>) -> T {
            return math::saturate<T>(s.rain / (s.temp + 30.0));
        }
    )");

    const char* main_src = R"(
        from biome::moisture import ClimateSample, aridity_index;

        @kernel fn compute_aridity(temp: f64, rain: f64) -> f64 {
            let sample = ClimateSample<f64> { temp: temp, rain: rain };
            return aridity_index(sample);
        }
    )";

    auto res = compiler.compile_source(main_src, "main.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    double arid = k->eval_f64({KernelArg(20.0), KernelArg(25.0)});
    assert(std::abs(arid - 0.5) < 1e-12);

    // Test import cycle detection
    compiler.register_module("cyc_a", "module cyc_a; import cyc_b; pub fn fa(x: f32) -> f32 { return x; }");
    compiler.register_module("cyc_b", "module cyc_b; import cyc_a; pub fn fb(x: f32) -> f32 { return x; }");
    auto cyc_res = compiler.compile_source("import cyc_a; @kernel fn k(x: f32) -> f32 { return x; }", "cyc.bk");
    assert(!cyc_res.ok);
    assert(cyc_res.error_message().find("circular import") != std::string::npos);
}

static void test_generic_stencil_filter_module() {
    const char* src = R"(
        from std::filter import laplacian2d, box_blur2d;

        @kernel fn diffuse_step(h: stencil<f64, 2, 1>, out: write_store<f64, 2>, alpha: f64) {
            let lap: f64 = laplacian2d(h);
            out.write(h.at(0, 0) + alpha * lap);
        }

        @kernel fn blur_r2(img: stencil<f32, 2, 2>, out: write_store<f32, 2>) {
            out.write(box_blur2d<f32, 2>(img));
        }
    )";

    auto res = compile_source(src, "filters.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);

    const auto* kd = res.find_kernel("diffuse_step");
    assert(kd != nullptr);
    assert(kd->max_stencil_radius == 1);
    assert(kd->find_param("h")->stencil_offsets.size() == 5);

    const auto* kb = res.find_kernel("blur_r2");
    assert(kb != nullptr);
    assert(kb->max_stencil_radius == 2);
    assert(kb->stencil_halo[0] == 2 && kb->stencil_halo[1] == 2);
    // A 5x5 box blur accesses all 25 offsets in [-2..2] x [-2..2]
    assert(kb->find_param("img")->stencil_offsets.size() == 25);
}

static void test_generic_type_pattern_inference() {
    const char* src = R"(
        fn f<Scalar>(v: vec3<Scalar>) -> Scalar {
            return v.x + v.y + v.z;
        }

        fn g<Data>(s: store<Data, 2>, x: i32, y: i32) -> Data {
            return s.load(x, y);
        }

        @kernel fn run_inferred(inp: store<f64, 2>, out: write_store<f64, 2>, dom: grid<2>) {
            let v = vec3<f64>(1.5, 2.5, 3.0);
            let sum_v: f64 = f(v);
            let cell: f64 = g(inp, dom.x(), dom.y());
            out.write(sum_v + cell);
        }

        @kernel fn run_inferred_i32(inp: store<i32, 2>, out: write_store<i32, 2>, dom: grid<2>) {
            let v = vec3<i32>(10, 20, 30);
            out.write(f(v) + g(inp, dom.x(), dom.y()));
        }
    )";

    auto res = compile_source(src, "generic_infer.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);

    const auto* k64 = res.find_kernel("run_inferred");
    assert(k64 != nullptr);
    std::vector<double> in64 = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> out64(4, 0.0);
    bool disp64 = k64->dispatch_cpu(
        GridShape(2, 2),
        {KernelArg::store(in64), KernelArg::store(out64)});
    assert(disp64);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(out64[i] - (7.0 + in64[i])) < 1e-12);
    }

    const auto* k32 = res.find_kernel("run_inferred_i32");
    assert(k32 != nullptr);
    std::vector<int32_t> in32 = {1, 2, 3, 4};
    std::vector<int32_t> out32(4, 0);
    bool disp32 = k32->dispatch_cpu(
        GridShape(2, 2),
        {KernelArg::store(in32), KernelArg::store(out32)});
    assert(disp32);
    for (int i = 0; i < 4; ++i) {
        assert(out32[i] == 60 + in32[i]);
    }
}

int main() {
    test_builtin_stdlib_modules();
    test_custom_and_filesystem_modules_and_cycles();
    test_generic_stencil_filter_module();
    test_generic_type_pattern_inference();
    std::cout << "test_modules_generics: ALL PASSED\n";
    return 0;
}

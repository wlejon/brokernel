// SPIR-V 1.5/1.6 emission and validation tests across Compute, Vertex, and Fragment stages.

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

using namespace brokernel;

static void test_compute_spirv_and_ptx() {
    const char* src = R"(
        @kernel @grid(2) @workgroup_size(16, 16, 1)
        fn height_erode(h: stencil<f64, 2, 1>, out: write_store<f64, 2>, k: f64) {
            let c: f64 = h.at(0, 0);
            let n: f64 = 0.25 * (h.at(-1, 0) + h.at(1, 0) + h.at(0, -1) + h.at(0, 1));
            out.write(c + k * (n - c));
        }
    )";

    CompileOptions opts;
    opts.emit_spirv = true;
    opts.emit_spirv_disassembly = true;
    opts.emit_ptx = true;
    opts.spirv_version = 0x00010600u; // SPIR-V 1.6

    auto res = compile_source(src, "height_erode.bk", opts);
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    assert(k->stage == Stage::Compute);
    assert(k->local_size[0] == 16 && k->local_size[1] == 16 && k->local_size[2] == 1);
    assert(!k->spirv.words.empty());
    // Verify SPIR-V magic word 0x07230203
    assert(k->spirv.words[0] == 0x07230203u);
    assert(k->spirv_disassembly.find("OpCapability Float64") != std::string::npos);
    assert(k->spirv_disassembly.find("OpEntryPoint GLCompute") != std::string::npos);
    assert(!k->ptx.empty());
    assert(k->ptx.find(".visible .entry height_erode") != std::string::npos);
}

static void test_vertex_stage_spirv() {
    const char* src = R"(
        import std::sphere;

        @vertex fn planet_patch_vs(scale: f32) {
            let vid: i32 = vertex_index();
            let iid: i32 = instance_index();
            let u: f32 = (vid & 15) as f32 / 15.0;
            let v: f32 = (vid >> 4) as f32 / 15.0;
            let p = sphere::cube_to_sphere<f32>(iid % 6, u * 2.0 - 1.0, v * 2.0 - 1.0) * scale;
            set_vertex_position(vec4<f32>(p.x, p.y, p.z, 1.0));
            set_vertex_output(0, vec4<f32>(u, v, p.z, 1.0));
        }
    )";

    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "planet_vs.bk", opts);
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    assert(k->stage == Stage::Vertex);
    assert(!k->spirv.words.empty());
    assert(k->spirv_disassembly.find("OpEntryPoint Vertex") != std::string::npos);
    assert(k->spirv_disassembly.find("gl_Position") != std::string::npos);
    assert(k->spirv_disassembly.find("out_loc0") != std::string::npos);
}

static void test_fragment_stage_spirv() {
    const char* src = R"(
        import std::color;

        @fragment fn atmosphere_fs(exposure: f32) {
            let fc: vec4<f32> = frag_coord();
            let dx: f32 = dpdx(fc.x);
            let dy: f32 = dpdy(fc.y);
            if fc.w < 0.0 {
                discard_fragment();
            }
            let hdr = vec3<f32>(fc.x * 0.001 + dx, fc.y * 0.001 + dy, 0.5) * exposure;
            let ldr = color::aces_tonemap<f32>(hdr);
            set_frag_color(vec4<f32>(ldr.x, ldr.y, ldr.z, 1.0));
        }
    )";

    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "atmosphere_fs.bk", opts);
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);
    assert(k->stage == Stage::Fragment);
    assert(!k->spirv.words.empty());
    assert(k->spirv_disassembly.find("OpEntryPoint Fragment") != std::string::npos);
    assert(k->spirv_disassembly.find("out_loc0") != std::string::npos);
    // Real derivatives and a real discard, not stand-ins.
    assert(k->spirv_disassembly.find("OpDPdx") != std::string::npos);
    assert(k->spirv_disassembly.find("OpDPdy") != std::string::npos);
    assert(k->spirv_disassembly.find("OpDemoteToHelperInvocation") != std::string::npos);
    // A fragment stage has no CPU build.
    assert(!k->has_cpu_build() && !k->cpu_point_fn().is_valid() && !k->cpu_grid_fn().is_valid());
}

// Render math: a fragment stage's f32 transcendentals are the device's own
// instructions and its arithmetic may fuse; a compute kernel's are the
// CPU-exact polynomials under NoContraction unless it is marked @fast.
static void test_render_math() {
    const char* src = R"(
        @fragment fn shade(k: f32) {
            let fc: vec4<f32> = frag_coord();
            let a: f32 = sin(fc.x * k) + exp(fc.y * k) + pow(fc.x, k) + atan2(fc.y, fc.x) + tan(k);
            set_frag_color(vec4<f32>(a, a * k + 1.0, 0.0, 1.0));
        }
        @kernel @grid(1) fn exact(out: write_store<f32, 1>, k: f32) {
            out.write(sin(k) * k + 1.0);
        }
        @kernel @grid(1) @fast fn quick(out: write_store<f32, 1>, k: f32) {
            out.write(sin(k) * k + 1.0);
        }
    )";
    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "render_math.bk", opts);
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    auto find = [&](const char* name) -> const CompiledKernel* {
        for (const auto& k : res.kernels) if (k.name == name) return &k;
        return nullptr;
    };
    const auto* fs = find("shade");
    const auto* exact = find("exact");
    const auto* quick = find("quick");
    assert(fs && exact && quick);
    // Whether the disassembly calls GLSL.std.450 instruction `n`
    // ("%r = OpExtInst %type %set n ...").
    auto has_ext = [](const std::string& dis, int n) {
        std::istringstream lines(dis);
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream w(line);
            std::string r, eq, op, type, set, num;
            if (w >> r >> eq >> op >> type >> set >> num && op == "OpExtInst" && num == std::to_string(n)) return true;
        }
        return false;
    };
    const std::string& d = fs->spirv_disassembly;
    // Sin 13, Exp 27, Pow 26, Atan2 25, Tan 15.
    for (int op : {13, 27, 26, 25, 15}) {
        if (!has_ext(d, op)) {
            std::cerr << "render math: no GLSL.std.450 " << op << " in\n" << d << "\n";
            assert(false);
        }
    }
    // Whether any result is decorated NoContraction (42): "OpDecorate %r 42".
    auto exact_math = [](const std::string& dis) {
        std::istringstream lines(dis);
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream w(line);
            std::string op, id, dec, more;
            if (w >> op >> id >> dec && !(w >> more) && op == "OpDecorate" && dec == "42") return true;
        }
        return false;
    };
    assert(!exact_math(d));
    assert(exact_math(exact->spirv_disassembly));
    assert(!has_ext(exact->spirv_disassembly, 13));
    assert(!exact_math(quick->spirv_disassembly) && has_ext(quick->spirv_disassembly, 13));
    // A @fast kernel's CPU build keeps the exact polynomials.
    assert(quick->has_cpu_build());
}

static void test_builtin_type_diagnostics() {
    auto fails_with = [](const char* src, const char* needle) {
        auto res = compile_source(src, "bad_builtin.bk");
        if (res.ok || res.error_message().find(needle) == std::string::npos) {
            std::cerr << "expected an error containing '" << needle << "', got:\n" << res.error_message() << "\n";
            assert(false);
        }
    };
    fails_with("@compute fn k(x: f32) { let d: f32 = dpdx(x); }", "@fragment");
    fails_with("@fragment fn k(x: f64) { let d: f64 = dpdx(x); }", "f32");
    fails_with("@kernel @grid(1) fn k(out: write_store<f64, 1>, x: f64) { out.write(subgroup_add(x)); }",
               "subgroup_add");
    fails_with("@kernel @grid(1) fn k(out: write_store<i32, 1>, x: i32) { out.write(isnan(x) ? 1 : 0); }", "isnan");
}

static void test_stage_misuse_diagnostics() {
    const char* bad_src = R"(
        @compute fn bad_compute() {
            let fc = frag_coord();
        }
    )";
    auto res = compile_source(bad_src, "bad_stage.bk");
    assert(!res.ok);
    assert(res.error_message().find("@fragment") != std::string::npos);
}

int main() {
    test_compute_spirv_and_ptx();
    test_vertex_stage_spirv();
    test_fragment_stage_spirv();
    test_render_math();
    test_builtin_type_diagnostics();
    test_stage_misuse_diagnostics();
    std::cout << "test_spirv_stages: ALL PASSED\n";
    return 0;
}

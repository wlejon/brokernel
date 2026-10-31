// SPIR-V 1.5/1.6 emission and validation tests across Compute, Vertex, and Fragment stages.

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace brokernel;

// spirv-val --target-env vulkan1.3 on a module: true when it is valid or
// the tool was not found at configure.
static bool spirv_valid(const CompiledKernel& k) {
    const std::string tool = BROKERNEL_SPIRV_VAL;
    if (tool.empty()) return true;
    auto dir = std::filesystem::temp_directory_path();
    auto spv = dir / ("brokernel_" + k.name + ".spv");
    auto log = dir / ("brokernel_" + k.name + ".txt");
    {
        std::ofstream out(spv, std::ios::binary);
        out.write(reinterpret_cast<const char*>(k.spirv.words.data()),
                  static_cast<std::streamsize>(k.spirv.words.size() * sizeof(uint32_t)));
    }
    std::string cmd = "\"\"" + tool + "\" --target-env vulkan1.3 \"" + spv.string() + "\" > \"" + log.string() + "\" 2>&1\"";
#ifndef _WIN32
    cmd = cmd.substr(1, cmd.size() - 2);
#endif
    if (std::system(cmd.c_str()) == 0) return true;
    std::ifstream in(log);
    std::cerr << "spirv-val rejected '" << k.name << "':\n" << in.rdbuf() << "\n";
    return false;
}

// Every kernel of `res` validates.
static void require_valid(const CompileResult& res) {
    for (const auto& k : res.kernels) assert(spirv_valid(k));
}

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
    require_valid(res);
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
    require_valid(res);
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
    require_valid(res);
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

// Typed stage inputs (@location parameters, @flat, integers flat) and outputs
// of any lane count, a second fragment output and the fragment's depth.
static void test_stage_io() {
    const char* src = R"(
        @vertex fn vs(@location(0) pos: vec3<f32>, @location(1) id: i32, k: f32) {
            set_vertex_position(vec4<f32>(pos * k, 1.0));
            set_vertex_output(0, pos.xy);
            set_vertex_output(1, id);
            set_vertex_output(2, pos.z);
        }
        @fragment fn fs(@location(0) uv: vec2<f32>, @location(1) id: i32, @location(2) @flat w: f32) {
            set_frag_color(vec4<f32>(uv, w, 1.0));
            set_frag_output(1, f32(id) * w);
            set_frag_output(2, vec2<u32>(u32(id), 7u));
            set_frag_depth(uv.x * 0.5);
        }
    )";
    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "stage_io.bk", opts);
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    require_valid(res);
    const CompiledKernel* vs = nullptr;
    const CompiledKernel* fs = nullptr;
    for (const auto& k : res.kernels) {
        if (k.name == "vs") vs = &k;
        if (k.name == "fs") fs = &k;
    }
    assert(vs && fs);
    // Only k is pushed: the stage inputs are no parameter data.
    size_t pushed = 0;
    for (const auto& p : vs->params) pushed += p.mir_param_count;
    assert(pushed == 1);
    // Lines "<op> %id <value>": decorations and modes print as numbers.
    auto has_line = [](const std::string& dis, const std::string& op, const std::string& value) {
        std::istringstream lines(dis);
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream w(line);
            std::string o, id, v, more;
            if (w >> o >> id >> v && !(w >> more) && o == op && v == value) return true;
        }
        return false;
    };
    const std::string& d = fs->spirv_disassembly;
    assert(has_line(d, "OpExecutionMode", "12"));   // DepthReplacing
    size_t flats = 0;
    for (size_t at = d.find("OpDecorate"); at != std::string::npos; at = d.find("OpDecorate", at + 1)) {
        size_t end = d.find('\n', at);
        std::istringstream w(d.substr(at, end - at));
        std::string o, id, v, more;
        if (w >> o >> id >> v && !(w >> more) && v == "14") ++flats;
    }
    assert(flats == 2);   // id (an integer) and w
    for (const char* name : {"out_loc0", "out_loc1", "out_loc2", "gl_FragDepth"}) {
        if (d.find(name) == std::string::npos) {
            std::cerr << "stage io: no " << name << " in\n" << d << "\n";
            assert(false);
        }
    }

    for (const char* bad : {
             "@kernel @grid(1) fn k(@location(0) x: f32, g: grid<1>) { }",
             "@fragment fn f(@location(0) x: f64) { set_frag_color(vec4<f32>(0.0, 0.0, 0.0, 1.0)); }",
             "@fragment fn f(@flat x: f32) { set_frag_color(vec4<f32>(x, 0.0, 0.0, 1.0)); }",
             "@vertex fn v(n: i32) { set_vertex_output(n, 1.0); }",
         }) {
        auto r = compile_source(bad, "bad_stage_io.bk");
        if (r.ok) std::cerr << "stage io: accepted\n" << bad << "\n";
        assert(!r.ok);
    }
}

// Buffers at descriptors: a uniform block laid out std140 (a vec3 padded to
// 16 bytes, an array of structs at a 16-byte multiple), read through fields,
// a dynamic index and a helper function; a storage buffer written.
static void test_buffers() {
    const char* src = R"(
        struct Light {
            pos: vec3<f32>,
            radius: f32,
            color: vec3<f32>,
        }
        struct Scene {
            eye: vec3<f32>,
            exposure: f32,
            lights: array<Light, 4>,
            count: i32,
        }
        fn light_color(scene: uniform<Scene>, i: i32) -> vec3<f32> {
            return scene.lights[i].color * scene.lights[i].radius;
        }
        @fragment fn shade(@binding(0, 0) scene: uniform<Scene>,
                           @binding(0, 1) hits: buffer<u32>,
                           @binding(0, 2) tints: read_buffer<vec4<f32>>,
                           @location(0) @flat which: i32) {
            var c: vec3<f32> = scene.eye.xyz * scene.exposure;
            for (var i: i32 = 0; i < scene.count; i += 1) {
                c += light_color(scene, i);
                hits[i] += 1u;
            }
            hits[which] = u32(scene.count);
            set_frag_color(vec4<f32>(c, 1.0) * tints[which]);
        }
    )";
    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "buffers.bk", opts);
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    require_valid(res);
    const CompiledKernel* fs = res.default_kernel();
    assert(fs);
    const auto& desc = fs->spirv.descriptors;
    assert(desc.size() == 3);
    for (const auto& d : desc) {
        if (d.binding == 0) {
            assert(d.kind == brass::target::SpirvDescriptorKind::UniformBuffer);
            // eye 0, exposure 12, lights at 16 (4 x 32: pos, radius, color
            // padded to 16), count at 144: 148 bytes, a block of 160.
            assert(d.bytes == 160);
        } else {
            assert(d.kind == brass::target::SpirvDescriptorKind::StorageBuffer);
        }
    }
    for (const char* bad : {
             "@fragment fn f(@binding(0) u: uniform<f32>) { u = 1.0; set_frag_color(vec4<f32>(0.0, 0.0, 0.0, 1.0)); }",
             "@fragment fn f(@binding(0) b: read_buffer<f32>) { b[0] = 1.0; set_frag_color(vec4<f32>(0.0, 0.0, 0.0, 1.0)); }",
             "@fragment fn f(@binding(0) b: buffer<f64>) { set_frag_color(vec4<f32>(0.0, 0.0, 0.0, 1.0)); }",
         }) {
        auto r = compile_source(bad, "bad_buffer.bk");
        if (r.ok) std::cerr << "buffers: accepted\n" << bad << "\n";
        assert(!r.ok);
    }
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
    require_valid(res);
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

// Samplers of every shape and storage images, through a helper function, to
// their descriptors and image instructions.
static void test_textures() {
    const char* src = R"(
        fn lit(albedo: sampler2d<vec4<f32>>, uv: vec2<f32>) -> vec4<f32> {
            return albedo.sample(uv) + albedo.sample_offset(uv, vec2<i32>(1, -1));
        }
        @fragment fn shade(@binding(0, 0) albedo: sampler2d<vec4<f32>>,
                           @binding(0, 1) sky: sampler_cube<vec3<f32>>,
                           @binding(0, 2) shadows: sampler2d_array_shadow,
                           @binding(1, 0) mask: sampler2d<u32>,
                           k: f32) {
            let fc: vec4<f32> = frag_coord();
            let uv: vec2<f32> = fc.xy * k;
            let c: vec4<f32> = lit(albedo, uv) + albedo.sample_lod(uv, 2.0) +
                               albedo.sample_grad(uv, vec2<f32>(k, 0.0), vec2<f32>(0.0, k)) +
                               albedo.fetch(vec2<i32>(3, 4), 0) + albedo.gather(uv, 1);
            let s: vec3<f32> = sky.sample(vec3<f32>(uv, 1.0));
            let lit_by: f32 = shadows.compare(vec3<f32>(uv, 2.0), k) + shadows.compare_lod(vec3<f32>(uv, 1.0), k, 0.0);
            let size: vec2<i32> = albedo.size(0);
            let m: u32 = mask.fetch(vec2<i32>(size.x - 1, 0), albedo.levels() - 1);
            set_frag_color(c + vec4<f32>(s * lit_by, f32(m)));
        }
        @kernel @grid(2) fn blur(@binding(0, 0) @format(rgba16f) dst: image2d<vec4<f32>>,
                                 @binding(0, 1) src: sampler2d<vec4<f32>>, g: grid<2>) {
            let uv: vec2<f32> = g.uv();
            dst.store(vec2<i32>(g.x(), g.y()), src.sample_lod(uv, 0.0) * 0.5 + dst.load(vec2<i32>(g.x(), g.y())) * 0.5);
        }
    )";
    CompileOptions opts;
    opts.emit_spirv_disassembly = true;
    auto res = compile_source(src, "textures.bk", opts);
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    require_valid(res);
    const CompiledKernel* fs = nullptr;
    const CompiledKernel* cs = nullptr;
    for (const auto& k : res.kernels) {
        if (k.name == "shade") fs = &k;
        if (k.name == "blur") cs = &k;
    }
    assert(fs && cs);
    const std::string& d = fs->spirv_disassembly;
    for (const char* op : {"OpImageSampleImplicitLod", "OpImageSampleExplicitLod", "OpImageFetch", "OpImageGather",
                           "OpImageSampleDrefImplicitLod", "OpImageSampleDrefExplicitLod", "OpImageQuerySizeLod",
                           "OpImageQueryLevels"}) {
        if (d.find(op) == std::string::npos) {
            std::cerr << "textures: no " << op << " in\n" << d << "\n";
            assert(false);
        }
    }
    using brass::target::SpirvImageDim;
    const auto& desc = fs->spirv.descriptors;
    assert(desc.size() == 4);
    auto at = [&](uint32_t set, uint32_t binding) -> const brass::target::SpirvDescriptorBinding& {
        for (const auto& b : desc) if (b.set == set && b.binding == binding) return b;
        assert(false);
        return desc[0];
    };
    assert(at(0, 0).dim == SpirvImageDim::D2 && !at(0, 0).arrayed);
    assert(at(0, 1).dim == SpirvImageDim::Cube);
    assert(at(0, 2).depth && at(0, 2).arrayed);
    assert(at(1, 0).scalar == brass::target::SpirvImageScalar::U32);
    // A kernel reading textures runs on the GPU alone.
    assert(!cs->has_cpu_build());
    assert(cs->spirv.descriptors.size() == 2);
    for (const auto& b : cs->spirv.descriptors) {
        if (b.binding == 0) {
            assert(b.kind == brass::target::SpirvDescriptorKind::StorageImage);
            assert(b.format == 2);   // Rgba16f
        }
    }

    // A texel no format holds, a format of the wrong kind, and a shadow
    // sampler's filtering are refused.
    for (const char* bad : {
             "@kernel @grid(2) fn k(@binding(0) im: image2d<vec3<f32>>, g: grid<2>) { }",
             "@kernel @grid(2) fn k(@binding(0) @format(rgba8ui) im: image2d<vec4<f32>>, g: grid<2>) { }",
             "@fragment fn f(@binding(0) s: sampler2d_shadow) { set_frag_color(vec4<f32>(s.sample(vec2<f32>(0.0, 0.0)), 0.0, 0.0, 1.0)); }",
         }) {
        auto r = compile_source(bad, "bad_texture.bk");
        if (r.ok) std::cerr << "textures: accepted\n" << bad << "\n";
        assert(!r.ok);
    }
}

// Textures sampled through one separate sampler (passed to a helper beside
// its texture), a shadow texture through a comparison sampler, and a separate
// texture fetched and sized with none.
static void test_separate_samplers() {
    const char* src = R"(
        fn tap(t: texture2d<vec4<f32>>, s: sampler, uv: vec2<f32>) -> vec4<f32> {
            return t.sample(s, uv) + t.sample_lod_offset(s, uv, 1.0, vec2<i32>(-1, 0));
        }
        @fragment fn shade(@binding(1, 8) linear: sampler,
                           @binding(1, 16) cmp: sampler,
                           @binding(1, 1) albedo: texture2d<vec4<f32>>,
                           @binding(1, 2) sky: texture_cube<vec3<f32>>,
                           @binding(1, 3) shadow: texture2d_array_shadow,
                           @binding(1, 4) lut: texture3d<f32>) {
            let fc: vec4<f32> = frag_coord();
            let uv: vec2<f32> = fc.xy * 0.01;
            let s: vec3<f32> = sky.sample_lod(linear, vec3<f32>(uv, 1.0), 0.0);
            let lit: f32 = shadow.compare(cmp, vec3<f32>(uv, 0.0), 0.5);
            let l: f32 = lut.sample(linear, vec3<f32>(uv, 0.5));
            let size: vec2<i32> = albedo.size(0);
            let texel: vec4<f32> = albedo.fetch(vec2<i32>(size.x - 1, 0), albedo.levels() - 1);
            set_frag_color(tap(albedo, linear, uv) + texel + vec4<f32>(s * lit, l));
        }
    )";
    auto res = compile_source(src, "separate.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    require_valid(res);
    const auto& desc = res.kernels[0].spirv.descriptors;
    assert(desc.size() == 6);
    using brass::target::SpirvDescriptorKind;
    for (const auto& b : desc) {
        assert(b.set == 1);
        bool sampler = b.binding == 8 || b.binding == 16;
        assert(b.kind == (sampler ? SpirvDescriptorKind::Sampler : SpirvDescriptorKind::SampledImage));
    }

    // A separate texture sampled with no sampler or through a texture, and a
    // bare sampler sampled, are refused.
    for (const char* bad : {
             "@fragment fn f(@binding(0) t: texture2d<vec4<f32>>) { set_frag_color(t.sample(vec2<f32>(0.0, 0.0))); }",
             "@fragment fn f(@binding(0) t: texture2d<vec4<f32>>, @binding(1) u: texture2d<vec4<f32>>) { set_frag_color(t.sample(u, vec2<f32>(0.0, 0.0))); }",
             "@fragment fn f(@binding(0) s: sampler) { set_frag_color(s.sample(vec2<f32>(0.0, 0.0))); }",
         }) {
        auto r = compile_source(bad, "bad_separate.bk");
        if (r.ok) std::cerr << "separate samplers: accepted\n" << bad << "\n";
        assert(!r.ok);
    }
}

// Module-level resources: a module declares the descriptors it reads
// (`@binding(s, b) var name: T;`), and a function of it, or of a module
// importing it, reads them by name, unqualified, through a symbol import,
// or by path; a buffer's elements are written through one.
static void test_module_resources() {
    Compiler compiler;
    compiler.register_module("scene", R"(
        module scene;
        pub struct Camera { view_proj: vec4<f32>, eye: vec3<f32>, frame: vec4<f32> }
        pub struct Light { color: vec4<f32>, dir: vec4<f32> }
        pub struct Ray { dir: vec3<f32>, t: f32 }
        @binding(0, 0) pub var camera: uniform<Camera>;
        @binding(1, 0) pub var lights: read_buffer<Light>;
        @binding(1, 8) pub var linear: sampler;
        @binding(1, 1) pub var sky: texture_cube<vec3<f32>>;
        @binding(2, 5) pub var meter: buffer<vec4<f32>>;
        pub fn ray(d: vec3<f32>) -> Ray { return Ray { dir: d, t: 2.0 }; }
        pub fn sky_light(dir: vec3<f32>) -> vec3<f32> {
            return sky.sample_lod(linear, dir, 0.0) * lights[0].color.rgb;
        }
    )");
    const char* src = R"(
        import scene;
        from scene import camera;
        struct Ray { a: f32 }   // not scene's
        @fragment fn shade(@location(0) dir: vec3<f32>) {
            let own = Ray { a: 3.0 };
            let theirs = scene::ray(dir);
            let l: vec3<f32> = scene::sky_light(normalize(dir)) + camera.eye * scene::camera.frame.x +
                               theirs.dir * (theirs.t * own.a);
            scene::meter[scene::meter.len() - 1] = vec4<f32>(l, 1.0);
            set_frag_color(vec4<f32>(l, 1.0));
        }
        @kernel @grid(1) fn plain(out: write_store<f32, 1>, g: grid<1>) { out.write(1.0); }
    )";
    auto res = compiler.compile_source(src, "uses_scene.bk");
    if (!res.ok) std::cerr << res.error_message() << "\n";
    assert(res.ok);
    require_valid(res);
    const CompiledKernel* fs = nullptr;
    const CompiledKernel* cs = nullptr;
    for (const auto& k : res.kernels) {
        if (k.name == "shade") fs = &k;
        if (k.name == "plain") cs = &k;
    }
    assert(fs && cs);
    using brass::target::SpirvDescriptorKind;
    // Only what the stage reads is bound; the kernel reading none keeps its CPU build.
    assert(fs->spirv.descriptors.size() == 5);
    for (const auto& b : fs->spirv.descriptors) {
        if (b.set == 0) assert(b.kind == SpirvDescriptorKind::UniformBuffer);
        if (b.set == 2) assert(b.kind == SpirvDescriptorKind::StorageBuffer && b.binding == 5);
    }
    assert(cs->spirv.descriptors.empty() && cs->has_cpu_build());
    for (const auto& p : fs->params) assert(p.name == "dir");
    // Every resource reflected, with a uniform struct's std140 offsets.
    assert(res.resources.size() == 5);
    assert(res.resources[0].name == "camera" && res.resources[0].module == "scene");
    const StructDef* cam = res.types.find_struct(res.resources[0].type.struct_id);
    assert(cam);
    assert(res.types.block_field_offset(*cam, "eye", true) == 16);
    assert(res.types.block_field_offset(*cam, "frame", true) == 32);
    assert(res.types.block_layout(res.resources[0].type.element_type(), true).size == 48);

    // A module-level value that is no resource, one without a binding, a
    // read-only buffer written and an unexported module's resource refused.
    Compiler bad;
    for (const char* src_bad : {
             "@binding(0) var x: f32;",
             "var c: uniform<vec4<f32>>;",
             "@binding(0) var b: read_buffer<f32>;\n@fragment fn f() { b[0] = 1.0; }",
         }) {
        auto r = bad.compile_source(src_bad, "bad_resource.bk");
        if (r.ok) std::cerr << "module resources: accepted\n" << src_bad << "\n";
        assert(!r.ok);
    }
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
    test_stage_io();
    test_buffers();
    test_render_math();
    test_textures();
    test_separate_samplers();
    test_module_resources();
    test_builtin_type_diagnostics();
    test_stage_misuse_diagnostics();
    std::cout << "test_spirv_stages: ALL PASSED\n";
    return 0;
}

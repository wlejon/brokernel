// The on-disk compile cache (compile_cache.h): a hit is the compile, field by
// field, and runs as the compile does on the CPU; what a compile depends on
// (source, imported modules, options, knobs) misses when it changes; a
// damaged entry misses; the directory is held to its limit; a hit is fast.
// CPU only: no GPU is used.
#include "brokernel/brokernel.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace brokernel;
namespace fs = std::filesystem;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

namespace {

const char* kBaseSrc = R"(
    module shapes_base;
    pub fn bias(x: f32) -> f32 { return x + 1.0; }
)";

const char* kShapesSrc = R"(
    module shapes;
    import shapes_base;
    pub struct Light { color: vec4<f32>, dir: vec4<f32> }
    @binding(0, 0) pub var light: uniform<Light>;
    pub const HALF: f32 = 0.5;
    pub fn shade(x: f32) -> f32 { return shapes_base::bias(x * 2.0); }
)";

const char* kRootSrc = R"(
    import shapes;
    @knob pub const SCALE: f32 = 3.0;
    @knob const OFFSET: f32 = 0.0;
    struct Pair { a: f32, b: vec2<f32> }
    fn pair_sum(p: Pair) -> f32 { return p.a + p.b.x + p.b.y; }

    @kernel fn point(x: f32) -> f32 {
        let p = Pair { a: x, b: vec2<f32>(1.0, 2.0) };
        return probe("ps", pair_sum(p)) * SCALE + shapes::shade(x) + OFFSET;
    }
    @kernel @grid(2) fn field(g: grid<2>, out: write_store<f32, 2>, k: f32) {
        let v = probe("v", g.x() as f32 + g.y() as f32 * 10.0);
        out[g.x(), g.y()] = shapes::shade(v * k);
    }
    @kernel @cpu fn precise(a: f64, b: f64) -> f64 { return a * b + 0.5; }
    @vertex fn vs(@location(0) pos: vec3<f32>, k: f32) {
        set_vertex_position(vec4<f32>(pos * k, 1.0));
        set_vertex_output(0, pos.xy);
    }
    @fragment fn fs(@location(0) uv: vec2<f32>, @binding(0, 1) tints: read_buffer<vec4<f32>>) {
        set_frag_color(vec4<f32>(uv, 0.0, 1.0) * tints[0] * shapes::light.color);
    }
)";

fs::path g_root;

fs::path fresh_dir(const char* name) {
    fs::path d = g_root / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

void use_cache(const fs::path& dir, uint64_t max_bytes = 512ull << 20) {
    CompileCacheConfig c;
    c.dir = dir;
    c.max_bytes = max_bytes;
    set_compile_cache(c);
}

CompileResult compile_with(const std::string& root, const std::string& shapes, const CompileOptions& opts,
                           bool cached = true) {
    Compiler c;
    c.set_use_compile_cache(cached);
    c.register_module("shapes_base", kBaseSrc);
    c.register_module("shapes", shapes);
    return c.compile_source(root, "root.bk", opts);
}

CompileOptions base_options() {
    CompileOptions o;
    o.emit_spirv_disassembly = true;
    o.knob_overrides["OFFSET"] = ConstValue::from_f32(0.25f);
    return o;
}

// --- Field-by-field equality -------------------------------------------------

bool same(const Type& a, const Type& b) {
    if (a.kind != b.kind || a.scalar != b.scalar || a.dimension != b.dimension ||
        a.stencil_radius != b.stencil_radius || a.access != b.access || a.struct_id != b.struct_id ||
        a.struct_name != b.struct_name || a.elem_kind != b.elem_kind || a.elem_lanes != b.elem_lanes ||
        a.image_flags != b.image_flags || (a.nested_elem == nullptr) != (b.nested_elem == nullptr)) {
        return false;
    }
    return !a.nested_elem || same(*a.nested_elem, *b.nested_elem);
}

bool same(const ConstValue& a, const ConstValue& b) {
    if (!same(a.type, b.type) || a.int_val != b.int_val || a.float_val != b.float_val || a.bool_val != b.bool_val ||
        a.elements.size() != b.elements.size()) {
        return false;
    }
    for (size_t i = 0; i < a.elements.size(); ++i) {
        if (!same(a.elements[i], b.elements[i])) return false;
    }
    return true;
}

bool same(const ProbeInfo& a, const ProbeInfo& b) { return a.name == b.name && same(a.type, b.type) && a.index == b.index; }

bool same(const KernelParamInfo& a, const KernelParamInfo& b) {
    if (a.name != b.name || !same(a.type, b.type) || a.role != b.role || a.access != b.access ||
        a.dimension != b.dimension || a.declared_radius != b.declared_radius || a.inferred_radius != b.inferred_radius ||
        a.mir_param_index != b.mir_param_index || a.mir_param_count != b.mir_param_count ||
        a.element_bytes != b.element_bytes || a.grid_cover_dim != b.grid_cover_dim ||
        a.push_constant_offset != b.push_constant_offset || a.push_constant_size != b.push_constant_size ||
        a.descriptor_set != b.descriptor_set || a.descriptor_binding != b.descriptor_binding ||
        a.image_format != b.image_format || a.stencil_offsets.size() != b.stencil_offsets.size()) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (a.halo[i] != b.halo[i]) return false;
    }
    for (size_t i = 0; i < a.stencil_offsets.size(); ++i) {
        if (!(a.stencil_offsets[i] == b.stencil_offsets[i])) return false;
    }
    return true;
}

template <typename T>
bool same_list(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!same(a[i], b[i])) return false;
    }
    return true;
}

void check_same_spirv(const brass::target::SpirvKernel& a, const brass::target::SpirvKernel& b) {
    CHECK(a.entry == b.entry && a.stage == b.stage);
    CHECK(a.words.size() == b.words.size());
    CHECK(std::memcmp(a.words.data(), b.words.data(), a.words.size() * sizeof(uint32_t)) == 0);
    CHECK(a.params.size() == b.params.size());
    for (size_t i = 0; i < a.params.size(); ++i) {
        CHECK(a.params[i].type == b.params[i].type && a.params[i].offset == b.params[i].offset &&
              a.params[i].size == b.params[i].size);
    }
    CHECK(a.descriptors == b.descriptors);
    CHECK(a.push_constant_bytes == b.push_constant_bytes && a.shared_bytes == b.shared_bytes);
    for (int i = 0; i < 3; ++i) CHECK(a.local_size[i] == b.local_size[i]);
    CHECK(a.local_size_spec_constants == b.local_size_spec_constants);
    CHECK(a.capabilities == b.capabilities && a.extensions == b.extensions);
}

void check_same_kernel(const CompiledKernel& a, const CompiledKernel& b) {
    CHECK(a.name == b.name && a.is_entry_kernel == b.is_entry_kernel && a.is_pub == b.is_pub);
    CHECK(a.stage == b.stage && a.grid_dim == b.grid_dim);
    for (int i = 0; i < 3; ++i) CHECK(a.local_size[i] == b.local_size[i] && a.stencil_halo[i] == b.stencil_halo[i]);
    CHECK(same_list(a.params, b.params));
    CHECK(same_list(a.all_mir_params, b.all_mir_params));
    CHECK(same_list(a.block_bindings, b.block_bindings));
    CHECK(same_list(a.probes, b.probes));
    CHECK(a.function_costs.size() == b.function_costs.size());
    for (size_t i = 0; i < a.function_costs.size(); ++i) {
        CHECK(a.function_costs[i].name == b.function_costs[i].name &&
              a.function_costs[i].self == b.function_costs[i].self &&
              a.function_costs[i].total == b.function_costs[i].total);
    }
    CHECK(a.code_hash == b.code_hash);
    CHECK(a.probes_compiled == b.probes_compiled);
    CHECK((a.build_probed_variant == nullptr) == (b.build_probed_variant == nullptr));
    CHECK(same(a.return_type, b.return_type));
    CHECK(a.return_flat_leaves == b.return_flat_leaves && a.return_byte_size == b.return_byte_size);
    CHECK(a.return_leaf_kinds == b.return_leaf_kinds);
    CHECK(a.uses_f64 == b.uses_f64 && a.cpu_only == b.cpu_only && a.uses_grid_coords == b.uses_grid_coords);
    CHECK(a.uses_barriers_or_shared == b.uses_barriers_or_shared && a.max_stencil_radius == b.max_stencil_radius);
    check_same_spirv(a.spirv, b.spirv);
    CHECK(a.spirv_disassembly == b.spirv_disassembly && a.ptx == b.ptx && a.spirv_equivalent == b.spirv_equivalent);
    CHECK(a.has_cpu_build() == b.has_cpu_build());
}

void check_same_result(const CompileResult& a, const CompileResult& b) {
    CHECK(a.ok == b.ok && a.module_name == b.module_name);
    CHECK(a.diagnostics.size() == b.diagnostics.size());
    for (size_t i = 0; i < a.diagnostics.size(); ++i) {
        const Diagnostic& x = a.diagnostics.items()[i];
        const Diagnostic& y = b.diagnostics.items()[i];
        CHECK(x.severity == y.severity && x.message == y.message && x.line_text == y.line_text &&
              x.span.file == y.span.file && x.span.start.offset == y.span.start.offset &&
              x.span.end.offset == y.span.end.offset && x.span.start.line == y.span.start.line);
    }
    CHECK(a.types.structs().size() == b.types.structs().size());
    for (size_t i = 0; i < a.types.structs().size(); ++i) {
        const StructDef& x = a.types.structs()[i];
        const StructDef& y = b.types.structs()[i];
        CHECK(x.id == y.id && x.name == y.name && x.byte_size == y.byte_size && x.flat_leaf_count == y.flat_leaf_count);
        CHECK(x.fields.size() == y.fields.size());
        for (size_t f = 0; f < x.fields.size(); ++f) {
            CHECK(x.fields[f].name == y.fields[f].name && same(x.fields[f].type, y.fields[f].type) &&
                  x.fields[f].byte_offset == y.fields[f].byte_offset);
        }
    }
    CHECK(a.resources.size() == b.resources.size());
    for (size_t i = 0; i < a.resources.size(); ++i) {
        const ResourceInfo& x = a.resources[i];
        const ResourceInfo& y = b.resources[i];
        CHECK(x.name == y.name && x.module == y.module && x.set == y.set && x.binding == y.binding &&
              same(x.type, y.type) && x.image_format == y.image_format);
    }
    CHECK(a.constants.size() == b.constants.size());
    for (size_t i = 0; i < a.constants.size(); ++i) {
        const ConstInfo& x = a.constants[i];
        const ConstInfo& y = b.constants[i];
        CHECK(x.name == y.name && x.module == y.module && same(x.type, y.type) && same(x.value, y.value) &&
              x.is_knob == y.is_knob);
    }
    CHECK(same_list(a.probes, b.probes));
    CHECK(a.exported_functions == b.exported_functions);
    CHECK(a.kernels.size() == b.kernels.size());
    for (size_t i = 0; i < a.kernels.size(); ++i) check_same_kernel(a.kernels[i], b.kernels[i]);
    CHECK(a.source == b.source && a.file_name == b.file_name);
    CHECK(a.loaded_modules == b.loaded_modules);
    CHECK(a.imports == b.imports);
    CHECK(a.search_paths == b.search_paths);
    CHECK((a.module_resolver == nullptr) == (b.module_resolver == nullptr));
}

// The kernels run on the CPU as a compile's do.
void check_same_runs(const CompileResult& a, const CompileResult& b) {
    const CompiledKernel* pa = a.find_kernel("point");
    const CompiledKernel* pb = b.find_kernel("point");
    CHECK(pa && pb);
    for (float x : {0.0f, 0.75f, -3.5f}) CHECK(pa->eval_f32({KernelArg(x)}) == pb->eval_f32({KernelArg(x)}));

    const CompiledKernel* fa = a.find_kernel("field");
    const CompiledKernel* fb = b.find_kernel("field");
    CHECK(fa && fb);
    std::vector<float> oa(8 * 4, -1.0f), ob(8 * 4, -1.0f);
    std::string err;
    CHECK(fa->dispatch_cpu(GridShape(8, 4), {KernelArg::store(oa), KernelArg(0.5f)}, nullptr, &err));
    CHECK(fb->dispatch_cpu(GridShape(8, 4), {KernelArg::store(ob), KernelArg(0.5f)}, nullptr, &err));
    CHECK(oa == ob);
    CHECK(fa->cpu_grid_lanes() == fb->cpu_grid_lanes());
    CHECK(ob[1 + 2 * 8] == (1.0f + 20.0f) * 0.5f * 2.0f + 1.0f);

    const CompiledKernel* ca = a.find_kernel("precise");
    const CompiledKernel* cb = b.find_kernel("precise");
    CHECK(ca && cb && cb->cpu_only && cb->spirv.words.empty());
    CHECK(ca->eval_f64({KernelArg(1.25), KernelArg(3.0)}) == cb->eval_f64({KernelArg(1.25), KernelArg(3.0)}));
    CHECK(cb->eval_f64({KernelArg(1.25), KernelArg(3.0)}) == 1.25 * 3.0 + 0.5);

    std::vector<ProbeSample> sa, sb;
    CHECK(fa->probe_cpu(3, 1, 0, GridShape(8, 4), {KernelArg::store(oa), KernelArg(0.5f)}, sa, &err));
    CHECK(fb->probe_cpu(3, 1, 0, GridShape(8, 4), {KernelArg::store(ob), KernelArg(0.5f)}, sb, &err));
    CHECK(sa.size() == sb.size() && !sb.empty());
    for (size_t i = 0; i < sa.size(); ++i) {
        CHECK(sa[i].name == sb[i].name && sa[i].hit == sb[i].hit && sa[i].values[0] == sb[i].values[0]);
    }
}

// --- Tests -------------------------------------------------------------------

void test_hit_equals_compile() {
    use_cache(fresh_dir("equal"));
    const CompileOptions opts = base_options();
    const CompileResult plain = compile_with(kRootSrc, kShapesSrc, opts, false);
    if (!plain.ok) std::cerr << plain.error_message() << "\n";
    CHECK(plain.ok && plain.cache_status == CacheStatus::Off);
    CHECK(plain.kernels.size() == 5 && !plain.resources.empty() && !plain.types.structs().empty());

    const CompileResult miss = compile_with(kRootSrc, kShapesSrc, opts);
    CHECK(miss.ok && miss.cache_status == CacheStatus::Miss && miss.cache_note == "no entry");
    const CompileResult hit = compile_with(kRootSrc, kShapesSrc, opts);
    if (hit.cache_status != CacheStatus::Hit) std::cerr << "not a hit: " << hit.cache_note << "\n";
    CHECK(hit.ok && hit.cache_status == CacheStatus::Hit);
    CHECK(hit.imports == std::vector<std::string>({"shapes", "shapes_base"}));
    check_same_result(plain, hit);
    check_same_result(miss, hit);
    check_same_runs(plain, hit);

    // Knob overrides in another order are the same key.
    CompileOptions reordered = opts;
    reordered.knob_overrides.clear();
    reordered.knob_overrides["SCALE"] = ConstValue::from_f32(3.0f);
    reordered.knob_overrides["OFFSET"] = ConstValue::from_f32(0.25f);
    CompileOptions forward = opts;
    forward.knob_overrides.clear();
    forward.knob_overrides["OFFSET"] = ConstValue::from_f32(0.25f);
    forward.knob_overrides["SCALE"] = ConstValue::from_f32(3.0f);
    CHECK(compile_with(kRootSrc, kShapesSrc, forward).ok);
    CHECK(compile_with(kRootSrc, kShapesSrc, reordered).cache_status == CacheStatus::Hit);

    // with_knobs and with_sources from a hit compile again from what it holds.
    std::unordered_map<std::string, ConstValue> knobs{{"SCALE", ConstValue::from_f32(10.0f)}};
    const CompileResult tuned = hit.with_knobs(knobs);
    CHECK(tuned.ok && tuned.cache_status == CacheStatus::Miss);
    const float pt = tuned.find_kernel("point")->eval_f32({KernelArg(1.0f)});
    CHECK(pt == (1.0f + 3.0f) * 10.0f + 3.0f + 0.25f);
    CHECK(plain.with_knobs(knobs).cache_status == CacheStatus::Hit);

    const std::string shapes2 = std::string(kShapesSrc) + "\n    pub fn extra() -> f32 { return 1.0; }\n";
    const CompileResult edited = hit.with_sources({{"shapes", shapes2}});
    CHECK(edited.ok && edited.cache_status == CacheStatus::Miss && edited.cache_note == "module changed: shapes");
    CHECK(edited.loaded_modules.at("shapes") == shapes2);
    const CompileResult edited_plain = plain.with_sources({{"shapes", shapes2}});
    CHECK(edited_plain.cache_status == CacheStatus::Hit);
    check_same_runs(edited_plain, edited);
    std::cout << "  hit equals compile: ok\n";
}

void test_invalidation() {
    use_cache(fresh_dir("invalidate"));
    const CompileOptions opts = base_options();
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Miss);
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Hit);

    const std::string root2 = std::string(kRootSrc) + "\n    fn unused() -> f32 { return 2.0; }\n";
    CHECK(compile_with(root2, kShapesSrc, opts).cache_note == "no entry");

    // A module the root imports through another one.
    {
        Compiler c;
        c.register_module("shapes_base", "module shapes_base;\npub fn bias(x: f32) -> f32 { return x + 2.0; }\n");
        c.register_module("shapes", kShapesSrc);
        const CompileResult r = c.compile_source(kRootSrc, "root.bk", opts);
        CHECK(r.ok && r.cache_status == CacheStatus::Miss && r.cache_note == "module changed: shapes_base");
        CHECK(r.find_kernel("point")->eval_f32({KernelArg(0.0f)}) == 3.0f * 3.0f + 2.0f + 0.25f);
    }
    // A module no longer found.
    {
        Compiler c;
        c.register_module("shapes", kShapesSrc);
        const CompileResult r = c.compile_source(kRootSrc, "root.bk", opts);
        CHECK(!r.ok && r.cache_status == CacheStatus::Miss && r.cache_note == "module missing: shapes_base");
    }
    CompileOptions o2 = opts;
    o2.opt_level = 1;
    CHECK(compile_with(kRootSrc, kShapesSrc, o2).cache_note == "no entry");
    CompileOptions o3 = opts;
    o3.knob_overrides["OFFSET"] = ConstValue::from_f32(0.5f);
    const CompileResult k3 = compile_with(kRootSrc, kShapesSrc, o3);
    CHECK(k3.cache_note == "no entry");
    CHECK(k3.find_kernel("point")->eval_f32({KernelArg(0.0f)}) == 9.0f + 1.0f + 0.5f);
    CompileOptions o4 = opts;
    o4.local_size[0] = 32;
    CHECK(compile_with(kRootSrc, kShapesSrc, o4).cache_note == "no entry");
    // One entry per key: the compile with the edited module replaced it, so
    // the original modules miss once and are stored again.
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_note == "module changed: shapes_base");
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Hit);
    std::cout << "  invalidation: ok\n";
}

std::vector<fs::path> entries(const fs::path& dir) {
    std::vector<fs::path> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".bkc") out.push_back(e.path());
    }
    return out;
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void test_damaged_entries() {
    const fs::path dir = fresh_dir("damaged");
    use_cache(dir);
    const CompileOptions opts = base_options();
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Miss);
    const auto files = entries(dir);
    CHECK(files.size() == 1);
    const std::string good = read_file(files[0]);
    CHECK(good.size() > 100);

    auto attempt = [&](const std::string& bytes, const char* note) {
        write_file(files[0], bytes);
        const CompileResult r = compile_with(kRootSrc, kShapesSrc, opts);
        if (r.cache_note != note) std::cerr << "note: " << r.cache_note << " (wanted " << note << ")\n";
        CHECK(r.ok && r.cache_status == CacheStatus::Miss && r.cache_note == note);
        check_same_runs(r, r);
    };
    attempt(good.substr(0, good.size() / 2), "entry truncated");
    attempt(good.substr(0, 20), "entry truncated");
    std::string flipped = good;
    flipped[flipped.size() - 7] ^= 0x5a;
    attempt(flipped, "entry damaged");
    attempt(std::string(good.size(), 'x'), "entry of another format");
    std::string other_key = good;
    other_key[9] ^= 1;
    attempt(other_key, "entry of another key");
    // Each miss stored the entry again.
    CHECK(read_file(files[0]) == good);
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Hit);

    // Deleting the directory is safe: the next compile misses and makes it again.
    fs::remove_all(dir);
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_note == "no entry");
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Hit);
    std::cout << "  damaged entries: ok\n";
}

uint64_t dir_disk_bytes(const fs::path& dir) {
    uint64_t total = 0;
    for (const auto& e : fs::directory_iterator(dir)) total += (e.file_size() + 4095) / 4096 * 4096;
    return total;
}

void test_size_limit() {
    const fs::path dir = fresh_dir("limit");
    const uint64_t limit = 96 * 1024;
    use_cache(dir, limit);
    CompileOptions opts;
    std::string last;
    for (int i = 0; i < 60; ++i) {
        last = "@kernel fn k(x: f32) -> f32 { return x * " + std::to_string(i) + ".0 + 1.0; }\n";
        const CompileResult r = compile_source(last, "limit.bk", opts);
        CHECK(r.ok && r.cache_status == CacheStatus::Miss);
    }
    flush_compile_cache();
    const uint64_t used = dir_disk_bytes(dir);
    const size_t n = entries(dir).size();
    std::cout << "  size limit: " << n << " entries, " << used / 1024 << " KiB of " << limit / 1024 << " KiB\n";
    CHECK(used <= limit && n > 0 && n < 60);
    // The newest stays.
    CHECK(compile_source(last, "limit.bk", opts).cache_status == CacheStatus::Hit);
}

void test_threads() {
    use_cache(fresh_dir("threads"));
    const CompileOptions opts = base_options();
    const CompileResult want = compile_with(kRootSrc, kShapesSrc, opts, false);
    std::vector<CompileResult> got(8);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < got.size(); ++i) {
        threads.emplace_back([&, i] { got[i] = compile_with(kRootSrc, kShapesSrc, opts); });
    }
    for (auto& t : threads) t.join();
    for (const auto& r : got) {
        CHECK(r.ok);
        check_same_result(want, r);
    }
    CHECK(compile_with(kRootSrc, kShapesSrc, opts).cache_status == CacheStatus::Hit);
    std::cout << "  threads: ok\n";
}

void test_speed() {
    use_cache(fresh_dir("speed"));
    const char* src = R"(
        import std::noise::*;
        @kernel @grid(2) fn terrain(g: grid<2>, out: write_store<f32, 2>, seed: i32) {
            let p = vec2<f32>(g.x() as f32, g.y() as f32) * 0.01;
            out[g.x(), g.y()] = perlin2d<f32>(p, seed) + value2d<f32>(p * 2.0, seed) +
                                cellular_distance2d<f32>(p * 4.0, seed) + perlin3d<f32>(vec3<f32>(p, 0.5), seed);
        }
        @kernel @grid(1) fn warp(g: grid<1>, px: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
            let q = domain_warp_gradient2d<f32>(vec2<f32>(px[g.x()], 0.25), seed);
            out[g.x()] = perlin2d<f32>(q, seed) + cellular_lookup_perlin2d<f32>(q, seed);
        }
    )";
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const CompileResult miss = compile_source(src, "speed.bk");
    const auto t1 = clock::now();
    const CompileResult hit = compile_source(src, "speed.bk");
    const auto t2 = clock::now();
    if (!miss.ok) std::cerr << miss.error_message() << "\n";
    CHECK(miss.ok && miss.cache_status == CacheStatus::Miss && hit.cache_status == CacheStatus::Hit);
    check_same_result(miss, hit);
    const double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double hit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double best_ms = hit_ms;
    for (int i = 0; i < 10; ++i) {
        const auto a = clock::now();
        CHECK(compile_source(src, "speed.bk").cache_status == CacheStatus::Hit);
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(clock::now() - a).count());
    }
    std::cout << "  speed: compile " << compile_ms << " ms, first hit " << hit_ms << " ms, best of 10 hits "
              << best_ms << " ms (" << miss.kernels[0].spirv.words.size() + miss.kernels[1].spirv.words.size()
              << " SPIR-V words)\n";
    CHECK(hit_ms * 5 < compile_ms);

    // The CPU build from a hit, lowered on first use, matches.
    std::vector<float> a(16 * 16), b(16 * 16);
    std::string err;
    CHECK(miss.find_kernel("terrain")->dispatch_cpu(GridShape(16, 16), {KernelArg::store(a), KernelArg(int32_t(7))},
                                                    nullptr, &err));
    CHECK(hit.find_kernel("terrain")->dispatch_cpu(GridShape(16, 16), {KernelArg::store(b), KernelArg(int32_t(7))},
                                                   nullptr, &err));
    CHECK(a == b);
}

} // namespace

int main() {
    g_root = fs::temp_directory_path() / ("brokernel_compile_cache_test_" + std::to_string(std::rand()) + "_" +
                                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::cout << "test_compile_cache (" << g_root.string() << ")\n";
    test_hit_equals_compile();
    test_invalidation();
    test_damaged_entries();
    test_size_limit();
    test_threads();
    test_speed();
    set_compile_cache({});
    flush_compile_cache();
    std::error_code ec;
    fs::remove_all(g_root, ec);
    std::cout << "test_compile_cache: all passed\n";
    return 0;
}

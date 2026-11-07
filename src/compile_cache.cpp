// Compiler::compile_source through the compile cache (compile_cache.h): a
// lookup by key, the entry's modules resolved again, and a hit rebuilt into
// the result a compile gives; a miss compiles and stores what it read.
#include "brokernel/brokernel.h"
#include "compile_cache_internal.h"
#include "compile_internal.h"

#include <brass/target/host_cpu.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>

namespace brokernel {

namespace {

struct ActiveCache {
    CompileCacheConfig config;
    std::string key_material;   // what every key starts with: format, version, build, host
};

std::mutex g_mu;
std::shared_ptr<const ActiveCache> g_cache;
std::atomic<uint64_t> g_hits{0}, g_misses{0}, g_stores{0}, g_store_failures{0};

std::shared_ptr<const ActiveCache> active_cache() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_cache;
}

detail::Digest entry_key(const ActiveCache& c, std::string_view source, const std::string& file_name,
                         const CompileOptions& opts,
                         const std::unordered_map<std::string, std::string>& registered) {
    std::string material = c.key_material;
    detail::ByteWriter w(material);
    w.str(source);
    w.str(file_name);
    detail::encode_options(opts, w);
    // The modules given by name (register_module), so one source compiled
    // against different ones (a stage beside each of its hooks) keeps an
    // entry each; those a resolver or search path finds are the entry's
    // records, checked at the lookup.
    std::vector<std::pair<std::string_view, detail::Digest>> mods;
    mods.reserve(registered.size());
    for (const auto& [name, src] : registered) mods.emplace_back(name, detail::hash128(src));
    std::sort(mods.begin(), mods.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    w.u32(static_cast<uint32_t>(mods.size()));
    for (const auto& [name, d] : mods) {
        w.str(name);
        w.u64(d.lo);
        w.u64(d.hi);
    }
    return detail::hash128(material);
}

std::string dir_string(const std::filesystem::path& p) {
    const std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path dir_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

bool is_builtin(const std::string& path) { return path.starts_with("<builtin:"); }

// What a hit's kernels compile their CPU builds again from, one copy for all.
struct CompileInputs {
    std::string source;
    std::string file_name;
    CompileOptions options;
    std::unordered_map<std::string, std::string> modules;
    std::vector<std::filesystem::path> search_paths;
    ModuleRegistry::ModuleResolver resolver;
};

// The CPU build of a kernel lowered by compiling the source again for it
// alone, with no GPU build (which lowers the CPU functions at once): the
// cached result has no checked module to lower them from.
std::function<void(CompiledKernel::CpuBuild&)> recompiling_lower(std::shared_ptr<const CompileInputs> in,
                                                                 const CompiledKernel& k) {
    return [in = std::move(in), name = k.name, stage = k.stage, grid_dim = k.grid_dim](CompiledKernel::CpuBuild& b) {
        const CompileOptions& opts = in->options;
        CompileOptions o = opts;
        o.emit_spirv = false;
        o.emit_spirv_disassembly = false;
        o.emit_ptx = false;
        o.verify_spirv_equivalence = false;
        o.compile_cpu = true;
        std::string err;
        // For the kernel alone first; if that does not give it (an entry
        // name also matches a function of an imported module), every kernel.
        for (const bool alone : {true, false}) {
            Compiler c;
            c.set_use_compile_cache(false);
            for (const auto& sp : in->search_paths) c.add_search_path(sp);
            for (const auto& [mn, ms] : in->modules) c.register_module(mn, ms);
            if (in->resolver) c.set_module_resolver(in->resolver);
            o.entry_name = alone ? name : opts.entry_name;
            CompileResult r = c.compile_source(in->source, in->file_name, o);
            if (!r.ok) {
                err = r.diagnostics.format_all();
                continue;
            }
            for (auto& rk : r.kernels) {
                if (rk.name != name || rk.stage != stage || rk.grid_dim != grid_dim || !rk.cpu_) continue;
                CompiledKernel::CpuBuild& from = *rk.cpu_;
                if (!from.error.empty()) {
                    b.error = from.error;
                    return;
                }
                b.module = std::move(from.module);
                b.point = from.point;
                b.grid = from.grid;
                b.grid_lanes = from.grid_lanes;
                b.shared_bytes = from.shared_bytes;
                return;
            }
            err = "the recompile has no CPU build of kernel '" + name + "'";
        }
        b.error = err;
    };
}

} // namespace

void set_compile_cache(CompileCacheConfig config) {
    std::shared_ptr<const ActiveCache> next;
    if (!config.dir.empty()) {
        const std::string stamp = config.build_stamp.empty() ? detail::build_identity() : config.build_stamp;
        if (!stamp.empty()) {
            auto c = std::make_shared<ActiveCache>();
            c->config = std::move(config);
            detail::ByteWriter w(c->key_material);
            w.str("brokernel compile cache");
            w.u32(detail::kEntryFormat);
            w.str(version());
            w.str(stamp);
            w.boolean(brass::host_has_avx2());
            w.boolean(brass::host_has_fma());
            next = std::move(c);
        }
    }
    std::lock_guard<std::mutex> lock(g_mu);
    g_cache = std::move(next);
}

CompileCacheConfig compile_cache_config() {
    auto c = active_cache();
    return c ? c->config : CompileCacheConfig{{}, 0, {}};
}

void flush_compile_cache() { detail::wait_for_trims(); }

CompileCacheStats compile_cache_stats() {
    return {g_hits.load(), g_misses.load(), g_stores.load(), g_store_failures.load()};
}

CompileResult Compiler::compile_source(std::string_view source, std::string file_name, const CompileOptions& opts) {
    const std::shared_ptr<const ActiveCache> cache = use_cache_ ? active_cache() : nullptr;
    if (!cache) return compile_uncached(source, std::move(file_name), opts);

    const detail::Digest key = entry_key(*cache, source, file_name, opts, modules_.given_modules());
    const std::filesystem::path file = detail::entry_path(cache->config.dir, key);
    std::string why;
    std::string payload;
    if (detail::read_entry(file, key, payload, why)) {
        detail::ByteReader r(payload.data(), payload.size());
        std::vector<detail::ModuleRecord> mods;
        bool fresh = detail::decode_modules(r, mods);
        if (!fresh) why = "entry damaged";
        // In the order the compile asked for them, an importer before what
        // it imports: each resolution made is one a compile would make.
        for (size_t i = 0; fresh && i < mods.size(); ++i) {
            const detail::ModuleRecord& m = mods[i];
            const auto got = modules_.resolve(m.name, dir_path(m.importing_dir));
            if (!got) {
                why = "module missing: " + m.key;
                fresh = false;
            } else if (got->key != m.key || *got->path != m.path || !(detail::hash128(*got->source) == m.source)) {
                why = "module changed: " + m.key;
                fresh = false;
            }
        }
        CompileResult hit;
        std::vector<uint8_t> had_cpu;
        if (fresh && !detail::decode_result(r, hit, had_cpu)) {
            why = "entry damaged";
            fresh = false;
        }
        if (fresh) {
            hit.ok = true;
            hit.source = std::string(source);
            hit.file_name = std::move(file_name);
            hit.options = opts;
            hit.search_paths = modules_.search_paths();
            hit.module_resolver = modules_.resolver();
            hit.loaded_modules = modules_.custom_modules();
            hit.imports = modules_.imported_modules();
            for (const auto& m : mods) {
                if (!is_builtin(m.path)) hit.imports.push_back(m.key);
            }
            std::sort(hit.imports.begin(), hit.imports.end());
            hit.imports.erase(std::unique(hit.imports.begin(), hit.imports.end()), hit.imports.end());
            std::shared_ptr<const CompileInputs> inputs;
            for (size_t i = 0; i < hit.kernels.size(); ++i) {
                CompiledKernel& k = hit.kernels[i];
                if (!opts.enable_probes && !k.probes.empty()) k.build_probed_variant = detail::probed_variant_builder(hit, k.name);
                if (had_cpu[i]) {
                    if (!inputs) {
                        inputs = std::make_shared<const CompileInputs>(CompileInputs{
                            hit.source, hit.file_name, hit.options, hit.loaded_modules, hit.search_paths,
                            hit.module_resolver});
                    }
                    k.cpu_ = detail::new_cpu_build(opts);
                    k.cpu_->lower = recompiling_lower(inputs, k);
                }
            }
            hit.cache_status = CacheStatus::Hit;
            detail::touch_entry(file);
            g_hits.fetch_add(1, std::memory_order_relaxed);
            return hit;
        }
    }

    modules_.record_requests();
    CompileResult res = compile_uncached(source, file_name, opts);
    const std::vector<ModuleRegistry::ModuleRequest> requests = modules_.take_requests();
    g_misses.fetch_add(1, std::memory_order_relaxed);
    res.cache_status = CacheStatus::Miss;
    res.cache_note = why;
    if (!res.ok) return res;

    std::vector<detail::ModuleRecord> mods;
    mods.reserve(requests.size());
    for (const auto& rq : requests) {
        const auto got = modules_.resolve(rq.name, rq.importing_dir);
        if (!got) return res;   // nothing to check a lookup against
        mods.push_back({rq.name, dir_string(rq.importing_dir), got->key, *got->path, detail::hash128(*got->source)});
    }
    std::string out;
    detail::ByteWriter w(out);
    detail::encode_modules(mods, w);
    detail::encode_result(res, w);
    if (detail::write_entry(file, key, out)) {
        g_stores.fetch_add(1, std::memory_order_relaxed);
        detail::note_store(cache->config.dir, cache->config.max_bytes, key, out.size() + 48);
    } else {
        g_store_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return res;
}

} // namespace brokernel

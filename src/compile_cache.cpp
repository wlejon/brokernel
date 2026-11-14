// Compiler::compile_source through the compile cache (compile_cache.h): a
// lookup by key in brass's artifact cache, the entry's modules resolved
// again, and a hit rebuilt into the result a compile gives; a miss compiles
// and stores what it read. A kernel's CPU build, on its first run, is stored
// as native code through brass's native module cache, or loaded from it.
#include "brokernel/brokernel.h"
#include "compile_cache_internal.h"
#include "compile_internal.h"

#include <brass/cache/artifact_cache.hpp>
#include <brass/cache/native_module_cache.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace brokernel {

namespace {

using brass::cache::BlobRef;
using brass::cache::CacheKey;
using brass::cache::KeyBuilder;

// The key schemas: a change to what a key holds, or to what an entry's blobs
// hold (detail::kEntryFormat), is a new domain or format, never a misread.
constexpr std::string_view kBaseDomain = "brokernel.compile-base/1";
constexpr std::string_view kReadsDomain = "brokernel.compile-reads/1";
constexpr std::string_view kEntryDomain = "brokernel.compile/3";
// The lists of module requests kept for one source (the oldest dropped).
constexpr size_t kMaxReadLists = 8;
constexpr std::string_view kCpuDomain = "brokernel.cpu/1";
constexpr std::string_view kCpuMetaDomain = "brokernel.cpu-meta/1";
constexpr const char* kUndecodable = "the entry does not decode";

// "path|size|mtime" of the binary brokernel's code is in, or empty.
// add_brass_build names the binary brass is in; when brass is a shared
// library of its own, that is another file, and a rebuild of brokernel
// (its lowering, its built-in modules) would not change it.
std::string brokernel_binary_identity() {
    std::filesystem::path self;
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&brokernel_binary_identity), &module)) {
        std::wstring buf(1024, L'\0');
        for (;;) {
            const DWORD n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
            if (n == 0) break;
            if (n < buf.size()) {
                buf.resize(n);
                self = std::filesystem::path(buf);
                break;
            }
            buf.resize(buf.size() * 2);
        }
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&brokernel_binary_identity), &info) && info.dli_fname) self = info.dli_fname;
#endif
    if (self.empty()) return {};
    std::error_code ec;
    const auto size = std::filesystem::file_size(self, ec);
    if (ec) return {};
    const auto mtime = std::filesystem::last_write_time(self, ec);
    if (ec) return {};
    return self.generic_string() + "|" + std::to_string(size) + "|" +
           std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

// The cache brokernel kept before brass had one: "<32 hex>.bkc" entries and
// their temporaries "<...>.bkc.tmp<pid>_<n>". Nothing reads them (their names
// are not brass's); they only take space.
void remove_old_format(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        const bool old = (name.size() > 4 && name.compare(name.size() - 4, 4, ".bkc") == 0) ||
                         name.find(".bkc.tmp") != std::string::npos;
        std::error_code rm;
        if (old && it->is_regular_file(rm)) std::filesystem::remove(it->path(), rm);
    }
}

struct ActiveCache {
    ActiveCache(CompileCacheConfig c, std::string b)
        : config(std::move(c)), build(std::move(b)), store(config.dir, config.max_bytes), natives(store) {}
    CompileCacheConfig config;
    std::string build;                          // brokernel's binary, or the build stamp
    brass::cache::ArtifactCache store;
    brass::cache::NativeModuleCache natives;    // CPU builds, in `store`
};

std::mutex g_mu;
std::shared_ptr<ActiveCache> g_cache;
std::atomic<uint64_t> g_hits{0}, g_misses{0}, g_stores{0}, g_store_failures{0};
std::atomic<uint64_t> g_cpu_loads{0}, g_cpu_stores{0}, g_cpu_in_memory{0};

std::shared_ptr<ActiveCache> active_cache() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_cache;
}

// What a compile is of, apart from the modules it reads: the compiler, the
// source, its file name and the options.
CacheKey base_key(const ActiveCache& c, std::string_view source, const std::string& file_name,
                  const CompileOptions& opts) {
    std::string options;
    detail::ByteWriter w(options);
    detail::encode_options(opts, w);
    return KeyBuilder(kBaseDomain)
        .add_brass_build()
        .add_u64("brokernel.format", detail::kEntryFormat)
        .add("brokernel.version", version())
        .add("brokernel.build", c.build)
        .add("source", source)
        .add("file_name", file_name)
        .add("options", options)
        .finish();
}

// --- CPU builds as native code ------------------------------------------------

// Whether KernelJit compiles a CPU build with these options as
// optimize_for_kernel_jit does (the options new_cpu_build makes).
bool storable(const brass::codegen::KernelOptions& o) {
    return !o.enable_parallel && !(o.enable_optimizations && o.opt_level >= 2 && (o.enable_superopt || o.enable_smt_guard_elim));
}

// A kernel's CPU build: the compile's entry and module records (what the
// MIR is lowered from), the kernel, and what it is compiled with.
CacheKey cpu_key(const CacheKey& entry, std::string_view records, size_t index, const CompiledKernel& k,
                 const brass::codegen::KernelOptions& o) {
    return KeyBuilder(kCpuDomain)
        .add_key("entry", entry)
        .add("modules", records)
        .add_u64("kernel.index", index)
        .add("kernel.name", k.name)
        .add_u64("kernel.stage", static_cast<uint64_t>(k.stage))
        .add_u64("kernel.grid_dim", k.grid_dim)
        .add_u64("opt.level", static_cast<uint64_t>(o.opt_level))
        .add_bool("opt.enabled", o.enable_optimizations)
        .add_bool("opt.sroa", o.enable_sroa)
        .add_bool("opt.bce", o.enable_bce)
        .add_bool("opt.avx2", o.enable_avx2)
        .add_bool("opt.fma", o.enable_fma)
        .add_bool("opt.fp_reassociation", o.enable_fp_reassociation)
        .add_bool("opt.vectorize", o.enable_vectorize)
        .add_bool("opt.slp", o.enable_slp)
        .add_bool("opt.unroll", o.enable_unroll)
        .add_u64("opt.unroll_factor", o.unroll_factor)
        .finish();
}

CacheKey cpu_meta_key(const CacheKey& cpu) { return KeyBuilder(kCpuMetaDomain).add_key("cpu", cpu).finish(); }

// What a stored CPU build is besides its code: its functions' names, the
// grid function's lanes and the shared memory a run needs.
struct CpuMeta {
    std::string point, grid;
    uint32_t grid_lanes = 1;
    uint32_t shared_bytes = 0;
};

std::string encode_meta(const CpuMeta& m) {
    std::string out;
    detail::ByteWriter w(out);
    w.str(m.point);
    w.str(m.grid);
    w.u32(m.grid_lanes);
    w.u32(m.shared_bytes);
    return out;
}

bool decode_meta(std::span<const uint8_t> bytes, CpuMeta& m) {
    detail::ByteReader r(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    m.point = r.str();
    m.grid = r.str();
    m.grid_lanes = r.u32();
    m.shared_bytes = r.u32();
    return r.ok() && r.at_end() && (!m.point.empty() || !m.grid.empty()) && m.grid_lanes >= 1;
}

// The build's functions from a loaded image. A KernelFunction keeps its code
// alive through its engine; this one's engine pointer shares the image's
// ownership and points at no engine (the code is a loaded library, not a
// JIT's), so a copy of the function keeps the image loaded as a JIT
// function keeps its engine.
bool bind_image(const std::shared_ptr<const brass::cache::NativeModule>& m, const CpuMeta& meta,
          CompiledKernel::CpuBuild& b) {
    void* point = meta.point.empty() ? nullptr : m->symbol(meta.point);
    void* grid = meta.grid.empty() ? nullptr : m->symbol(meta.grid);
    if ((!meta.point.empty() && !point) || (!meta.grid.empty() && !grid)) return false;
    const std::shared_ptr<brass::codegen::JitExecutionEngine> owner(m, nullptr);
    b.point_fn = point ? brass::codegen::KernelFunction(meta.point, point, 0, owner) : brass::codegen::KernelFunction();
    b.grid_fn = grid ? brass::codegen::KernelFunction(meta.grid, grid, 0, owner) : brass::codegen::KernelFunction();
    b.grid_lanes = meta.grid_lanes;
    b.shared_bytes = meta.shared_bytes;
    // Nothing left for cpu_build to compile.
    b.point = nullptr;
    b.grid = nullptr;
    b.module.reset();
    return true;
}

// KernelJit::compile's MIR pipeline (brass src/codegen/kernel_jit.cpp:311,
// run_kernel_optimizations at :247, which is private) for the options a CPU
// build has (storable()): no auto-parallelization, no SMT guard elimination,
// no superoptimizer. A copy because brass offers that pipeline to no one but
// its JIT; once KernelJit has a public "prepare this module as compile()
// would" (run_kernel_optimizations and the verify, made callable), this
// becomes that one call and storable() goes.
bool build_cpu_module(const brass::Module& src, const brass::codegen::KernelOptions& o, brass::Module& mod,
                      std::string& error) {
    mod.set_allow_fp_reassociation(src.allow_fp_reassociation() || o.enable_fp_reassociation);
    mod.copy_declarations_from(src);
    for (const brass::Function* fn : src.functions()) {
        if (fn) brass::clone_function(*fn, mod);
    }
    if (o.enable_optimizations && o.opt_level >= 1) {
        brass::LoopOptOptions l;
        l.enable_sroa = o.enable_sroa && o.opt_level >= 2;
        l.enable_bce = o.enable_bce && o.opt_level >= 2;
        l.enable_avx2 = o.enable_avx2;
        l.enable_fma = o.enable_fma;
        l.enable_fp_reassociation = o.enable_fp_reassociation || mod.allow_fp_reassociation();
        l.enable_vectorize = o.enable_vectorize;
        l.enable_slp = o.enable_slp;
        l.enable_unroll = o.enable_unroll;
        l.unroll_factor = o.unroll_factor;
        l.enable_f64_demote = false;
        l.enable_parallel_loops = false;
        std::vector<brass::Function*> fns;
        for (brass::Function* fn : mod.functions()) {
            if (fn) fns.push_back(fn);
        }
        for (brass::Function* fn : fns) {
            brass::optimize_function(*fn, l);
            if (o.opt_level >= 2) brass::eliminate_dead_code(*fn);
        }
    }
    for (brass::Function* fn : mod.functions()) {
        if (fn) fn->rebuild_cfg_predecessors();
    }
    brass::DiagnosticReporter diag;
    if (!brass::verify_module(mod, &diag)) {
        error = "verification failed: " + diag.format_all();
        return false;
    }
    return true;
}

// The lowered build (b.module) compiled, stored and loaded as native code;
// false (the build untouched, for the JIT) when it cannot be.
bool store_cpu(ActiveCache& c, const CacheKey& key, CompiledKernel::CpuBuild& b) {
    if (!storable(b.options) || !b.module) return false;
    const brass::Module* src = b.module.get();
    if ((b.point && b.point->parent() != src) || (b.grid && b.grid->parent() != src)) return false;
    CpuMeta meta;
    if (b.point) meta.point = std::string(b.point->name());
    if (b.grid) meta.grid = std::string(b.grid->name());
    meta.grid_lanes = b.grid_lanes ? b.grid_lanes : 1;
    meta.shared_bytes = b.shared_bytes;
    // A module calling a function it does not define fails to link (the
    // image cannot point into this process) and is compiled in memory.
    const auto m = c.natives.get_or_build(
        key, [&](brass::Module& mod, std::string& error) { return build_cpu_module(*src, b.options, mod, error); });
    if (!m || !bind_image(m, meta, b)) return false;
    c.store.put(cpu_meta_key(key), {BlobRef("meta", encode_meta(meta))});
    return true;
}

bool load_cpu(ActiveCache& c, const CacheKey& key, CompiledKernel::CpuBuild& b) {
    if (!storable(b.options)) return false;
    const std::optional<brass::cache::Artifact> art = c.store.get(cpu_meta_key(key));
    CpuMeta meta;
    if (!art || !decode_meta(art->bytes("meta"), meta)) return false;
    const auto m = c.natives.find(key);
    return m && bind_image(m, meta, b);
}

// A CPU build's `lower` through the cache: the stored code when there is
// some; else `lower` (none: the build is lowered already), then the code
// stored. The JIT compiles what is left (b.point / b.grid) as without a cache.
std::function<void(CompiledKernel::CpuBuild&)> cached_lower(std::shared_ptr<ActiveCache> cache, CacheKey key,
                                                           std::function<void(CompiledKernel::CpuBuild&)> lower) {
    return [cache = std::move(cache), key, lower = std::move(lower)](CompiledKernel::CpuBuild& b) {
        if (load_cpu(*cache, key, b)) {
            g_cpu_loads.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (lower) lower(b);
        if (!b.error.empty() || !b.module || (!b.point && !b.grid)) return;
        if (store_cpu(*cache, key, b)) {
            g_cpu_stores.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_cpu_in_memory.fetch_add(1, std::memory_order_relaxed);
        }
    };
}

// --- Hits ----------------------------------------------------------------------

std::string dir_string(const std::filesystem::path& p) {
    const std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path dir_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

bool is_builtin(const std::string& path) { return path.starts_with("<builtin:"); }

std::string_view as_text(std::span<const uint8_t> bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

// The modules a compile asked for, in the order it asked: (name, importing
// directory) each. A source's entry is keyed by what these resolve to now,
// so a module the compile never reads (one registered for other sources)
// is no part of its key, and one source compiled against different modules
// of the same names keeps an entry each.
using ReadList = std::vector<std::pair<std::string, std::string>>;

CacheKey reads_key(const CacheKey& base) { return KeyBuilder(kReadsDomain).add_key("base", base).finish(); }

// The lists stored for `base`, the newest last.
std::vector<ReadList> load_read_lists(ActiveCache& c, const CacheKey& base, std::string& why) {
    std::vector<ReadList> lists;
    brass::cache::Miss miss = brass::cache::Miss::None;
    const std::optional<brass::cache::Artifact> art = c.store.get(reads_key(base), &miss);
    if (!art) {
        why = brass::cache::to_string(miss);
        return lists;
    }
    why = kUndecodable;
    if (!art->has("reads")) return lists;
    const std::string_view bytes = as_text(art->bytes("reads"));
    detail::ByteReader r(bytes.data(), bytes.size());
    const size_t n = r.count();
    for (size_t i = 0; r.ok() && i < n; ++i) {
        ReadList list(r.count(16));
        for (auto& [name, dir] : list) {
            name = r.str();
            dir = r.str();
        }
        lists.push_back(std::move(list));
    }
    if (!r.ok() || !r.at_end()) lists.clear();
    if (!lists.empty()) why.clear();
    return lists;
}

void store_read_lists(ActiveCache& c, const CacheKey& base, const std::vector<ReadList>& lists) {
    std::string out;
    detail::ByteWriter w(out);
    w.u64(lists.size());
    for (const ReadList& list : lists) {
        w.u64(list.size());
        for (const auto& [name, dir] : list) {
            w.str(name);
            w.str(dir);
        }
    }
    c.store.put(reads_key(base), {BlobRef("reads", out)});
}

// The entry of `base` compiled with `reads` resolving as they do now in
// `modules` (one not found is part of the key as that).
CacheKey entry_key(const CacheKey& base, const ReadList& reads, ModuleRegistry& modules) {
    KeyBuilder kb(kEntryDomain);
    kb.add_key("base", base).add_u64("reads", reads.size());
    for (const auto& [name, dir] : reads) {
        kb.add("module.name", name).add("module.dir", dir);
        if (const auto got = modules.resolve(name, dir_path(dir))) {
            kb.add("module.key", got->key).add_digest("module.source", brass::hash128(*got->source));
        } else {
            kb.add("module.key", "<missing>");
        }
    }
    return kb.finish();
}

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
// cached result has no checked module to lower them from. Only when the
// cache does not hold the build's code (cached_lower).
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
    std::shared_ptr<ActiveCache> next;
    if (!config.dir.empty()) {
        std::string build = config.build_stamp.empty() ? brokernel_binary_identity() : config.build_stamp;
        // Nothing is cached that its key cannot invalidate: brass's binary
        // and brokernel's must both be known.
        const bool brass_known = KeyBuilder(kEntryDomain).add_brass_build().finish().valid;
        if (!build.empty() && brass_known) {
            remove_old_format(config.dir);
            next = std::make_shared<ActiveCache>(std::move(config), std::move(build));
        }
    }
    std::shared_ptr<ActiveCache> prev;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        prev = std::move(g_cache);
        g_cache = std::move(next);
    }
    // `prev` goes here, outside the lock: its store waits for a trim in
    // progress (unless a compile or a CPU build still holds it).
}

CompileCacheConfig compile_cache_config() {
    auto c = active_cache();
    return c ? c->config : CompileCacheConfig{{}, 0, {}};
}

void flush_compile_cache() {
    if (auto c = active_cache()) c->store.flush();
}

CompileCacheStats compile_cache_stats() {
    CompileCacheStats s;
    s.hits = g_hits.load();
    s.misses = g_misses.load();
    s.stores = g_stores.load();
    s.store_failures = g_store_failures.load();
    s.cpu_loads = g_cpu_loads.load();
    s.cpu_stores = g_cpu_stores.load();
    s.cpu_in_memory = g_cpu_in_memory.load();
    return s;
}

CompileResult Compiler::compile_source(std::string_view source, std::string file_name, const CompileOptions& opts) {
    const std::shared_ptr<ActiveCache> cache = use_cache_ ? active_cache() : nullptr;
    if (!cache) return compile_uncached(source, std::move(file_name), opts);

    const CacheKey base = base_key(*cache, source, file_name, opts);
    std::string why;
    std::vector<ReadList> lists = load_read_lists(*cache, base, why);
    // The newest list first: the modules the last compile of this source read.
    for (size_t li = lists.size(); li-- > 0;) {
        const CacheKey key = entry_key(base, lists[li], modules_);
        brass::cache::Miss miss = brass::cache::Miss::None;
        const std::optional<brass::cache::Artifact> art = cache->store.get(key, &miss);
        if (!art) {
            if (why.empty()) why = brass::cache::to_string(miss);
            continue;
        }
        const std::string_view records = as_text(art->bytes("modules"));
        const std::string_view result = as_text(art->bytes("result"));
        detail::ByteReader mr(records.data(), records.size());
        std::vector<detail::ModuleRecord> mods;
        bool fresh = art->has("modules") && art->has("result") && detail::decode_modules(mr, mods) && mr.at_end();
        if (!fresh) why = kUndecodable;
        // In the order the compile asked for them, an importer before what
        // it imports: each resolution made is one a compile would make.
        for (size_t i = 0; fresh && i < mods.size(); ++i) {
            const detail::ModuleRecord& m = mods[i];
            const auto got = modules_.resolve(m.name, dir_path(m.importing_dir));
            if (!got) {
                why = "module missing: " + m.key;
                fresh = false;
            } else if (got->key != m.key || *got->path != m.path || brass::hash128(*got->source) != m.source) {
                why = "module changed: " + m.key;
                fresh = false;
            }
        }
        CompileResult hit;
        std::vector<uint8_t> had_cpu;
        detail::ByteReader rr(result.data(), result.size());
        if (fresh && !detail::decode_result(rr, hit, had_cpu)) {
            why = kUndecodable;
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
                    k.cpu_->lower = cached_lower(cache, cpu_key(key, records, i, k, k.cpu_->options),
                                                 recompiling_lower(inputs, k));
                }
            }
            hit.cache_status = CacheStatus::Hit;
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
    ReadList reads;
    reads.reserve(requests.size());
    for (const auto& rq : requests) {
        const auto got = modules_.resolve(rq.name, rq.importing_dir);
        if (!got) return res;   // nothing to check a lookup against
        mods.push_back({rq.name, dir_string(rq.importing_dir), got->key, *got->path, brass::hash128(*got->source)});
        reads.emplace_back(rq.name, dir_string(rq.importing_dir));
    }
    const CacheKey key = entry_key(base, reads, modules_);
    // This compile's list the newest of the source's.
    std::erase(lists, reads);
    lists.push_back(std::move(reads));
    if (lists.size() > kMaxReadLists) lists.erase(lists.begin(), lists.end() - kMaxReadLists);
    store_read_lists(*cache, base, lists);
    std::string records;
    detail::ByteWriter mw(records);
    detail::encode_modules(mods, mw);
    for (size_t i = 0; i < res.kernels.size(); ++i) {
        CompiledKernel& k = res.kernels[i];
        if (k.cpu_) k.cpu_->lower = cached_lower(cache, cpu_key(key, records, i, k, k.cpu_->options), std::move(k.cpu_->lower));
    }
    std::string out;
    detail::ByteWriter w(out);
    detail::encode_result(res, w);
    if (cache->store.put(key, {BlobRef("modules", records), BlobRef("result", out)})) {
        g_stores.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_store_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return res;
}

} // namespace brokernel

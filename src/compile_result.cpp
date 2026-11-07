// A compile made again from what it was compiled from: with other knobs, or
// with new sources for some of its modules.
#include "brokernel/brokernel.h"
#include "compile_internal.h"

#include <brass/target/host_cpu.hpp>

#include <algorithm>

namespace brokernel {

namespace detail {

std::function<std::shared_ptr<const CompiledKernel>()> probed_variant_builder(const CompileResult& res,
                                                                              const std::string& kernel) {
    CompileOptions popts = res.options;
    popts.enable_probes = true;
    popts.entry_name = kernel;
    return [src = res.source, fname = res.file_name, popts, mods = res.loaded_modules, sps = res.search_paths,
            resolver = res.module_resolver]() -> std::shared_ptr<const CompiledKernel> {
        Compiler c;
        for (const auto& sp : sps) c.add_search_path(sp);
        for (const auto& [mn, ms] : mods) c.register_module(mn, ms);
        if (resolver) c.set_module_resolver(resolver);
        auto pres = c.compile_source(src, fname, popts);
        if (!pres.ok || pres.kernels.empty()) return nullptr;
        return std::make_shared<CompiledKernel>(std::move(pres.kernels.front()));
    };
}

std::shared_ptr<CompiledKernel::CpuBuild> new_cpu_build(const CompileOptions& opts) {
    auto cpu = std::make_shared<CompiledKernel::CpuBuild>();
    cpu->options.opt_level = opts.opt_level;
    cpu->options.enable_optimizations = (opts.opt_level > 0);
    // Fast math fuses multiply-adds where the host has FMA.
    cpu->options.enable_fma = brass::host_has_fma();
    cpu->options.enable_parallel = false;
    cpu->options.enable_superopt = false;
    cpu->options.enable_smt_guard_elim = false;
    return cpu;
}

} // namespace detail

CompileResult CompileResult::with_knobs(const std::unordered_map<std::string, ConstValue>& overrides) const {
    Compiler compiler;
    for (const auto& sp : search_paths) compiler.add_search_path(sp);
    for (const auto& [mname, msrc] : loaded_modules) compiler.register_module(mname, msrc);
    if (module_resolver) compiler.set_module_resolver(module_resolver);
    CompileOptions new_opts = options;
    for (const auto& [k, v] : overrides) {
        new_opts.knob_overrides[k] = v;
    }
    return compiler.compile_source(source, file_name, new_opts);
}

bool CompileResult::depends_on(const std::string& key) const {
    return key == file_name || std::binary_search(imports.begin(), imports.end(), key);
}

CompileResult CompileResult::with_sources(const std::unordered_map<std::string, std::string>& sources,
                                          bool use_resolver) const {
    Compiler compiler;
    for (const auto& sp : search_paths) compiler.add_search_path(sp);
    for (const auto& [mname, msrc] : loaded_modules) {
        auto it = sources.find(mname);
        compiler.register_module(mname, it != sources.end() ? it->second : msrc);
    }
    // A changed module this one did not hold yet (a new import of a changed
    // module): registered too, so it wins over the resolver.
    for (const auto& [mname, msrc] : sources) {
        if (mname != file_name && !loaded_modules.count(mname)) compiler.register_module(mname, msrc);
    }
    compiler.set_module_resolver(use_resolver ? module_resolver : ModuleRegistry::ModuleResolver{});
    auto root = sources.find(file_name);
    return compiler.compile_source(root != sources.end() ? root->second : source, file_name, options);
}

} // namespace brokernel

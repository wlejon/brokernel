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

std::function<std::shared_ptr<const CompiledKernel>(bool trap_nf)> checked_variant_builder(
    const CompileResult& res, const std::string& kernel) {
    CompileOptions base_opts = res.options;
    base_opts.entry_name = kernel;
    return [src = res.source, fname = res.file_name, base_opts, mods = res.loaded_modules, sps = res.search_paths,
            resolver = res.module_resolver](bool trap_nf) -> std::shared_ptr<const CompiledKernel> {
        CompileOptions copts = base_opts;
        copts.enable_checks = true;
        copts.trap_non_finite = trap_nf;
        Compiler c;
        for (const auto& sp : sps) c.add_search_path(sp);
        for (const auto& [mn, ms] : mods) c.register_module(mn, ms);
        if (resolver) c.set_module_resolver(resolver);
        auto cres = c.compile_source(src, fname, copts);
        if (!cres.ok || cres.kernels.empty()) return nullptr;
        return std::make_shared<CompiledKernel>(std::move(cres.kernels.front()));
    };
}

std::function<std::string(std::string* err)> ptx_variant_builder(const CompileResult& res,
                                                                 const std::string& kernel) {
    CompileOptions popts = res.options;
    popts.emit_ptx = true;
    popts.emit_spirv = false;
    popts.compile_cpu = false;
    popts.entry_name = kernel;
    return [src = res.source, fname = res.file_name, popts, mods = res.loaded_modules, sps = res.search_paths,
            resolver = res.module_resolver](std::string* err) -> std::string {
        try {
            Compiler c;
            for (const auto& sp : sps) c.add_search_path(sp);
            for (const auto& [mn, ms] : mods) c.register_module(mn, ms);
            if (resolver) c.set_module_resolver(resolver);
            auto pres = c.compile_source(src, fname, popts);
            if (!pres.ok || pres.kernels.empty() || pres.kernels.front().ptx.empty()) {
                if (err) {
                    const std::string d = pres.diagnostics.format_all();
                    *err = d.empty() ? ("failed to emit PTX for kernel '" + popts.entry_name + "'") : d;
                }
                return {};
            }
            return std::move(pres.kernels.front().ptx);
        } catch (const std::exception& ex) {
            if (err) *err = ex.what();
            return {};
        }
    };
}

std::function<MslArtifact(std::string* err)> msl_variant_builder(const CompileResult& res,
                                                                 const std::string& kernel) {
    CompileOptions popts = res.options;
    popts.emit_msl = true;
    popts.emit_spirv = false;
    popts.compile_cpu = false;
    popts.entry_name = kernel;
    return [src = res.source, fname = res.file_name, popts, mods = res.loaded_modules, sps = res.search_paths,
            resolver = res.module_resolver](std::string* err) -> MslArtifact {
        try {
            Compiler c;
            for (const auto& sp : sps) c.add_search_path(sp);
            for (const auto& [mn, ms] : mods) c.register_module(mn, ms);
            if (resolver) c.set_module_resolver(resolver);
            auto pres = c.compile_source(src, fname, popts);
            if (!pres.ok || pres.kernels.empty() || pres.kernels.front().msl.empty()) {
                if (err) {
                    const std::string d = pres.diagnostics.format_all();
                    *err = d.empty() ? ("failed to emit MSL for kernel '" + popts.entry_name + "'") : d;
                }
                return {};
            }
            auto& fk = pres.kernels.front();
            MslArtifact out;
            out.source = std::move(fk.msl);
            out.entry_name = std::move(fk.msl_entry);
            out.num_buffers = fk.msl_num_buffers;
            out.scalar_params_bytes = fk.msl_scalar_params_bytes;
            out.scalar_param_offsets = std::move(fk.msl_scalar_param_offsets);
            return out;
        } catch (const std::exception& ex) {
            if (err) *err = ex.what();
            return {};
        }
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

const std::string& CompiledKernel::ensure_ptx(std::string* err) const {
    static const std::string kEmpty;
    if (!ptx.empty()) return ptx;
    if (stage != Stage::Compute || cpu_only || uses_f64) {
        if (err) *err = "kernel '" + name + "' cannot be compiled to PTX";
        return kEmpty;
    }
    if (!gpu_cache_) {
        if (err) *err = "kernel '" + name + "' has no PTX cache";
        return kEmpty;
    }
    std::lock_guard<std::mutex> lock(gpu_cache_->mu);
    if (!gpu_cache_->ptx_tried) {
        gpu_cache_->ptx_tried = true;
        if (build_ptx_variant) {
            gpu_cache_->ptx_cached = build_ptx_variant(&gpu_cache_->ptx_error);
        } else {
            gpu_cache_->ptx_error = "kernel '" + name + "' has no PTX builder";
        }
    }
    if (gpu_cache_->ptx_cached.empty() && err) {
        *err = gpu_cache_->ptx_error.empty() ? ("kernel '" + name + "' PTX compilation failed") : gpu_cache_->ptx_error;
    }
    return gpu_cache_->ptx_cached;
}

const MslArtifact& CompiledKernel::ensure_msl(std::string* err) const {
    static const MslArtifact kEmpty;
    if (stage != Stage::Compute || cpu_only || uses_f64) {
        if (err) *err = "kernel '" + name + "' cannot be compiled to MSL";
        return kEmpty;
    }
    if (!gpu_cache_) {
        if (err) *err = "kernel '" + name + "' has no MSL cache";
        return kEmpty;
    }
    std::lock_guard<std::mutex> lock(gpu_cache_->mu);
    if (!gpu_cache_->msl_tried) {
        gpu_cache_->msl_tried = true;
        if (!msl.empty()) {
            gpu_cache_->msl_cached = MslArtifact{
                msl, msl_entry, msl_num_buffers, msl_scalar_params_bytes, msl_scalar_param_offsets};
        } else if (build_msl_variant) {
            gpu_cache_->msl_cached = build_msl_variant(&gpu_cache_->msl_error);
        } else {
            gpu_cache_->msl_error = "kernel '" + name + "' has no MSL builder";
        }
    }
    if (gpu_cache_->msl_cached.source.empty() && err) {
        *err = gpu_cache_->msl_error.empty() ? ("kernel '" + name + "' MSL compilation failed") : gpu_cache_->msl_error;
    }
    return gpu_cache_->msl_cached;
}

std::shared_ptr<const CompiledKernel> CompiledKernel::ensure_probed_variant() const {
    if (probes_compiled || probes.empty() || !build_probed_variant) return nullptr;
    if (!gpu_cache_) return build_probed_variant();
    std::lock_guard<std::mutex> lock(gpu_cache_->mu);
    if (!gpu_cache_->probed_variant) {
        gpu_cache_->probed_variant = build_probed_variant();
    }
    return gpu_cache_->probed_variant;
}

std::shared_ptr<const CompiledKernel> CompiledKernel::ensure_checked_variant(bool trap_non_finite) const {
    if (checks_compiled && trap_non_finite_compiled == trap_non_finite) return nullptr;
    if (!build_checked_variant) return nullptr;
    if (!gpu_cache_) return build_checked_variant(trap_non_finite);
    std::lock_guard<std::mutex> lock(gpu_cache_->mu);
    auto& slot = trap_non_finite ? gpu_cache_->checked_variant_nf : gpu_cache_->checked_variant_no_nf;
    if (!slot) {
        slot = build_checked_variant(trap_non_finite);
    }
    return slot;
}

} // namespace brokernel

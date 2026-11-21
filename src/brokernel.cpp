#include "brokernel/brokernel.h"
#include "ast_cache.h"
#include "compile_internal.h"
#include "kernel_report.h"

#include <brass/mir/gpu_opt.hpp>
#include <brass/target/host_cpu.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/verify/gpu_equiv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace brokernel {

// ---------------------------------------------------------------------------
// Compiler Implementation
// ---------------------------------------------------------------------------

Compiler::Compiler() = default;

void Compiler::register_module(std::string name, std::string source) {
    modules_.register_module(std::move(name), std::move(source));
}

void Compiler::add_search_path(std::filesystem::path dir) {
    modules_.add_search_path(std::move(dir));
}

CompileResult Compiler::compile_uncached(
    std::string_view source,
    std::string file_name,
    const CompileOptions& opts) {
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_total0 = clock::now();

    CompileResult res;
    res.source = std::string(source);
    res.file_name = file_name;
    res.options = opts;
    res.search_paths = modules_.search_paths();
    res.module_resolver = modules_.resolver();

    const auto t_parse0 = clock::now();
    auto ast = detail::parse_source_cached(source, file_name, res.diagnostics);
    res.timings.parse_ms = ms_since(t_parse0, clock::now());
    if (!ast || res.diagnostics.has_errors()) {
        res.timings.total_ms = ms_since(t_total0, clock::now());
        res.ok = false;
        return res;
    }

    std::filesystem::path root_dir;
    if (!file_name.empty()) {
        std::filesystem::path p(file_name);
        if (p.has_parent_path()) root_dir = p.parent_path();
    }

    // Shared: a kernel's CPU functions are lowered from it when first asked for.
    const auto t_sema0 = clock::now();
    std::shared_ptr<CheckedModule> checked = check_module(*ast, modules_, res.diagnostics, root_dir, opts.entry_name,
                                                          opts.knob_overrides.empty() ? nullptr : &opts.knob_overrides,
                                                          opts.runtime_knobs, opts.telemetry_slots);
    res.timings.sema_ms = ms_since(t_sema0, clock::now());
    res.loaded_modules = modules_.custom_modules();
    res.imports = modules_.imported_modules();
    if (!checked || res.diagnostics.has_errors()) {
        res.ok = false;
        return res;
    }
    checked->enable_probes = opts.enable_probes;
    checked->enable_fragment_probes = opts.enable_fragment_probes;

    res.module_name = checked->module_name;
    res.types = checked->types;
    res.constants = checked->exported_constants;
    res.probes = checked->probes;
    res.exported_functions = checked->exported_functions;
    for (const auto& [key, index] : checked->global_index) {
        const CheckedParam& g = checked->globals[index];
        if (g.role == ParamRole::Shared) continue;
        res.resources.push_back({g.name, key.substr(0, key.size() - g.name.size() - 2), g.descriptor_set,
                                 g.descriptor_binding, g.type, g.image_format});
    }
    std::sort(res.resources.begin(), res.resources.end(), [](const ResourceInfo& a, const ResourceInfo& b) {
        return a.set != b.set ? a.set < b.set : a.binding < b.binding;
    });

    bool any_explicit_kernel = false;
    for (const auto& fn : checked->functions) {
        if (fn.module_name == checked->module_name && fn.is_entry_kernel) {
            any_explicit_kernel = true;
            break;
        }
    }

    std::vector<const CheckedFunction*> targets;
    for (const auto& fn : checked->functions) {
        if (!opts.entry_name.empty()) {
            if (fn.name == opts.entry_name || fn.source_name == opts.entry_name) {
                targets.push_back(&fn);
            }
            continue;
        }
        if (fn.module_name != checked->module_name) continue;
        if (opts.compile_all_functions || !any_explicit_kernel || fn.is_entry_kernel) {
            targets.push_back(&fn);
        }
    }

    if (!opts.entry_name.empty() && targets.empty()) {
        res.diagnostics.error(ast->span, "entry function '" + opts.entry_name + "' not found in module");
        res.ok = false;
        return res;
    }

    for (const CheckedFunction* fn : targets) {
        // GPU code is f32: a kernel built for the GPU that reaches f64 is an
        // error at the f64. A @cpu kernel has no GPU build and may use it; a
        // plain function compiled as an entry (no @kernel in the module) has
        // only its CPU build when it does.
        bool gpu_build = opts.emit_spirv && !fn->cpu_only;
        if (fn->cpu_only && fn->stage != Stage::Compute) {
            res.diagnostics.error(fn->span, "'" + fn->source_name + "' is a vertex or fragment stage: it cannot be @cpu");
            continue;
        }
        if (gpu_build && fn->uses_f64) {
            if (fn->is_entry_kernel) {
                res.diagnostics.error(fn->f64_site ? *fn->f64_site : fn->span,
                                      "f64 in GPU kernel '" + fn->source_name +
                                          "': GPU code is f32 (compute in f64 on the CPU and pass f32 offsets from an "
                                          "anchor near the work), or mark the kernel @cpu");
                continue;
            }
            gpu_build = false;
        }
        const auto t_lower0 = clock::now();
        LoweredKernelMir lowered = lower_kernel_to_mir(*checked, *fn, res.diagnostics);
        res.timings.lower_ms += ms_since(t_lower0, clock::now());
        if (res.diagnostics.has_errors()) {
            res.ok = false;
            return res;
        }

        CompiledKernel ck;
        ck.name = fn->source_name;
        ck.is_entry_kernel = fn->is_entry_kernel;
        ck.is_pub = fn->is_pub;
        ck.stage = fn->stage;
        ck.grid_dim = fn->grid_dim;
        for (int i = 0; i < 3; ++i) {
            ck.local_size[i] = fn->has_explicit_local_size ? fn->local_size[i]
                               : (fn->grid_dim >= 2 ? fn->local_size[i] : opts.local_size[i]);
        }
        ck.params = lowered.param_infos;
        ck.block_bindings = lowered.block_bindings;
        ck.probes = fn->probes;
        ck.probes_compiled = opts.enable_probes && !fn->probes.empty();
        if (!opts.enable_probes && !fn->probes.empty()) ck.build_probed_variant = detail::probed_variant_builder(res, ck.name);
        ck.uses_barriers_or_shared = lowered.uses_barriers_or_shared;
        ck.return_type = fn->return_type;
        ck.return_leaf_kinds = checked->types.flatten_scalars(fn->return_type);
        ck.return_flat_leaves = static_cast<uint32_t>(ck.return_leaf_kinds.size());
        ck.return_byte_size = checked->types.byte_size(fn->return_type);
        ck.uses_f64 = fn->uses_f64;
        ck.cpu_only = fn->cpu_only;
        ck.uses_grid_coords = fn->uses_grid_coords;
        ck.max_stencil_radius = fn->max_stencil_radius;
        ck.stencil_halo[0] = fn->stencil_halo[0];
        ck.stencil_halo[1] = fn->stencil_halo[1];
        ck.stencil_halo[2] = fn->stencil_halo[2];
        for (const auto& rk : checked->runtime_knobs) {
            ck.runtime_knobs.push_back(RuntimeKnobInfo{
                rk.name, rk.module, rk.key, rk.type, rk.byte_offset, rk.byte_size, rk.default_value});
        }
        ck.runtime_knob_bytes = checked->runtime_knob_bytes;
        ck.init_runtime_knobs();
        ck.telemetry_slots = opts.telemetry_slots;
        ck.uses_telemetry = fn->uses_telemetry;
        ck.init_telemetry();

        try {
            const bool emit_text_now = (opts.emit_ptx || opts.emit_msl) && ck.stage == Stage::Compute && !fn->cpu_only && !fn->uses_f64;
            if (opts.optimize_gpu_mir && opts.opt_level > 0 && (gpu_build || emit_text_now) &&
                lowered.gpu_function && lowered.gpu_module) {
                const auto t_opt0 = clock::now();
                brass::GpuOptOptions gpu_opts;
                gpu_opts.opt_level = opts.opt_level;
                // Superoptimization & SMT guard elimination run expensive bitvector SAT queries;
                // reserve them for explicit opt_level >= 3 (matching new_cpu_build).
                gpu_opts.enable_superopt = (opts.opt_level >= 3);
                gpu_opts.enable_gpu_superopt = (opts.opt_level >= 3);
                gpu_opts.enable_smt_guard_elim = (opts.opt_level >= 3);
                brass::optimize_gpu_function(*lowered.gpu_function, *lowered.gpu_module, gpu_opts);
                res.timings.gpu_opt_ms += ms_since(t_opt0, clock::now());
            }
            if (gpu_build) {
                const auto t_spv0 = clock::now();
                brass::target::SpirvOptions spv_opts;
                spv_opts.stage = ck.stage;
                spv_opts.local_size_x = ck.local_size[0];
                spv_opts.local_size_y = ck.local_size[1];
                spv_opts.local_size_z = ck.local_size[2];
                spv_opts.local_size_spec_constants = opts.local_size_spec_constants;
                spv_opts.spirv_version = opts.spirv_version;
                spv_opts.loads_through_variable = opts.spirv_loads_through_variable;

                ck.spirv = brass::target::SpirvTarget::compile(*lowered.gpu_function, spv_opts);
                res.timings.spirv_ms += ms_since(t_spv0, clock::now());
                ck.function_costs = detail::function_costs(*lowered.gpu_function, lowered.gpu_module->debug_context(), ck.name);
                if (opts.emit_spirv_disassembly) {
                    ck.spirv_disassembly = brass::target::SpirvTarget::dump_function(*lowered.gpu_function, spv_opts);
                }

                for (auto& p : ck.params) {
                    if (p.mir_param_count > 0 && p.mir_param_index < ck.spirv.params.size()) {
                        p.push_constant_offset = ck.spirv.params[p.mir_param_index].offset;
                        uint32_t total_sz = 0;
                        for (uint32_t k = 0; k < p.mir_param_count && (p.mir_param_index + k) < ck.spirv.params.size(); ++k) {
                            total_sz += ck.spirv.params[p.mir_param_index + k].size;
                        }
                        p.push_constant_size = total_sz;
                    }
                }

                if (opts.verify_spirv_equivalence && ck.stage == Stage::Compute) {
                    auto eq = brass::verify::verify_spirv_equivalence(*lowered.gpu_function, spv_opts);
                    ck.spirv_equivalent = eq.is_equivalent();
                }
            }

            if (lowered.gpu_function) {
                detail::emit_optional_gpu_targets(opts, *fn, *lowered.gpu_function, ck);
            }

            if (opts.compile_cpu && has_cpu_lowering(*fn)) {
                auto cpu = detail::new_cpu_build(opts);
                // The CPU functions from the checked module, into the build.
                auto lower = [checked, fn, bindings = lowered.block_bindings](CompiledKernel::CpuBuild& b,
                                                                             DiagnosticBag& diags) {
                    LoweredKernelMir cpu_mir;
                    cpu_mir.block_bindings = bindings;
                    lower_kernel_cpu_to_mir(*checked, *fn, cpu_mir, diags);
                    b.module = std::move(cpu_mir.cpu_module);
                    b.point = cpu_mir.cpu_point_function;
                    b.grid = cpu_mir.cpu_grid_function;
                    b.grid_lanes = cpu_mir.cpu_grid_lanes ? cpu_mir.cpu_grid_lanes : 1;
                    b.shared_bytes = cpu_mir.cpu_shared_bytes;
                };
                if (gpu_build) {
                    // Lowered the first time the kernel runs on the CPU.
                    cpu->lower = [lower](CompiledKernel::CpuBuild& b) {
                        DiagnosticBag diags;
                        lower(b, diags);
                        if (diags.has_errors()) b.error = diags.format_all();
                    };
                } else {
                    // The CPU is its only build: lowered now, its errors the compile's.
                    lower(*cpu, res.diagnostics);
                    if (res.diagnostics.has_errors()) {
                        res.ok = false;
                        return res;
                    }
                }
                ck.cpu_ = std::move(cpu);
            }
            ck.code_hash = detail::code_hash(ck);
        } catch (const std::exception& ex) {
            res.diagnostics.error(fn->span, std::string("backend compilation error: ") + ex.what());
            res.ok = false;
            return res;
        }

        res.kernels.push_back(std::move(ck));
    }

    res.timings.total_ms = ms_since(t_total0, clock::now());
    res.ok = !res.diagnostics.has_errors();
    return res;
}

CompileResult Compiler::compile_file(
    const std::filesystem::path& path,
    const CompileOptions& opts) {
    CompileResult res;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        res.diagnostics.error({path.string(), 1, 1, 0, 0},
                              "could not open kernel file '" + path.string() + "'");
        res.ok = false;
        return res;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return compile_source(ss.str(), path.string(), opts);
}

CompileResult compile_source(
    std::string_view source,
    std::string file_name,
    const CompileOptions& opts) {
    Compiler compiler;
    return compiler.compile_source(source, std::move(file_name), opts);
}

CompileResult compile_file(
    const std::filesystem::path& path,
    const CompileOptions& opts) {
    Compiler compiler;
    return compiler.compile_file(path, opts);
}

} // namespace brokernel

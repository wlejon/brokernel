// Formal SPIR-V verification and multi-target text emitters (PTX, MSL).
#include "compile_internal.h"

#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>

#include <algorithm>
#include <sstream>

namespace brokernel {

KernelVerifyReport CompiledKernel::verify() const {
    KernelVerifyReport rep;
    if (spirv.words.empty()) {
        rep.diagnostic = cpu_only
                             ? "kernel '" + name + "' is @cpu (no SPIR-V binary to verify)"
                             : "kernel '" + name + "' has no SPIR-V binary (emit_spirv=false)";
        return rep;
    }

    brass::spirv::Module spv_mod;
    std::string parse_err;
    if (!brass::spirv::parse_binary(spirv.words, spv_mod, &parse_err)) {
        rep.diagnostic = parse_err.empty() ? "failed to parse SPIR-V binary" : parse_err;
        return rep;
    }
    rep.spirv_parsed = true;

    constexpr size_t kMaxSmtSpirvWords = 3072;
    if (spirv.words.size() > kMaxSmtSpirvWords) {
        const auto audit = brass::verify::prove_spirv_shader_safety(spv_mod);
        const bool struct_ok = audit.parse_ok && audit.structural_ok && audit.dominance_ok;
        rep.memory_safe = struct_ok && audit.shared_bounds_ok;
        rep.barrier_uniform = struct_ok && audit.barriers_uniform;
        for (const auto& sa : audit.shared_accesses) {
            if (sa.proven_in_bounds) ++rep.shared_accesses_proven;
            else rep.violations.push_back(sa.detail.empty() ? "shared memory safety violation" : sa.detail);
        }
        for (const auto& b : audit.barriers) {
            if (b.uniform_control_flow) ++rep.barriers_proven;
            else rep.violations.push_back(b.detail.empty() ? "barrier divergence" : b.detail);
        }
        for (const auto& d : audit.verifier_diagnostics) {
            if (!d.message.empty()) rep.violations.push_back(d.message);
        }
        rep.diagnostic = rep.violations.empty()
                             ? "verified via static SPIR-V compute shader audit"
                             : rep.violations.front();
        return rep;
    }

    brass::verify::GpuProverOptions opts;
    opts.local_size_x = std::max(1u, local_size[0]);
    opts.local_size_y = std::max(1u, local_size[1]);
    opts.local_size_z = std::max(1u, local_size[2]);
    std::vector<int32_t> grid_dim_indices;
    uint32_t max_store_mir_end = 0;
    for (const auto& p : params) {
        if ((p.role == ParamRole::Store || p.role == ParamRole::Stencil) && p.mir_param_count >= 2) {
            brass::verify::GpuBufferSpec spec;
            spec.param_index = p.mir_param_index;
            spec.length_param_index = static_cast<int32_t>(p.mir_param_index + 1);
            spec.elem_bytes = p.element_bytes ? p.element_bytes : 4;
            spec.base_alignment = 4;
            opts.buffers.push_back(spec);
            max_store_mir_end = std::max(max_store_mir_end, p.mir_param_index + p.mir_param_count);
            if (grid_dim > 0) {
                brass::verify::GpuParamConstraint pc;
                pc.param_index = p.mir_param_index + 1;
                pc.min_val = 1;
                opts.param_constraints.push_back(pc);
            }
        } else if (p.mir_param_count > 0) {
            max_store_mir_end = std::max(max_store_mir_end, p.mir_param_index + p.mir_param_count);
        }
    }
    // Note any trailing implicit return-store pointer (which appears after grid dims in spirv.params).
    if (!return_type.is_void()) {
        for (size_t i = max_store_mir_end; i < spirv.params.size(); ++i) {
            if (spirv.params[i].size == 8 && spirv.params[i].type == brass::Type::ptr()) {
                brass::verify::GpuBufferSpec spec;
                spec.param_index = static_cast<uint32_t>(i);
                spec.elem_bytes = return_byte_size ? return_byte_size : 4;
                spec.base_alignment = 4;
                if (grid_dim >= 1 && max_store_mir_end + grid_dim <= i) {
                    spec.length_param_index = static_cast<int32_t>(max_store_mir_end);
                    if (grid_dim >= 2) spec.length_param_index2 = static_cast<int32_t>(max_store_mir_end + 1);
                    if (grid_dim >= 3) spec.length_param_index3 = static_cast<int32_t>(max_store_mir_end + 2);
                } else {
                    spec.static_byte_size = spec.elem_bytes;
                }
                opts.buffers.push_back(spec);
                break;
            }
        }
    }
    auto mem = brass::verify::verify_spirv_memory_safety(spv_mod, opts);
    auto bar = brass::verify::verify_spirv_barrier_uniformity(spv_mod, opts);

    rep.memory_safe = mem.is_safe();
    rep.barrier_uniform = bar.is_uniform();
    rep.shared_accesses_proven = mem.shared_accesses_proven;
    rep.global_accesses_proven = mem.global_accesses_proven;
    rep.barriers_proven = bar.barriers_proven;

    for (const auto& v : mem.violations) {
        rep.violations.push_back(v.description.empty() ? "memory safety violation" : v.description);
    }
    for (const auto& d : bar.divergences) {
        rep.violations.push_back(d.description.empty() ? "barrier divergence" : d.description);
    }

    std::ostringstream oss;
    if (!mem.diagnostic.empty()) oss << mem.diagnostic;
    if (!bar.diagnostic.empty()) {
        if (!mem.diagnostic.empty()) oss << "; ";
        oss << bar.diagnostic;
    }
    rep.diagnostic = oss.str();
    return rep;
}

namespace detail {

void emit_optional_gpu_targets(const CompileOptions& opts, const CheckedFunction& fn,
                               const brass::Function& gpu_fn, CompiledKernel& ck) {
    if (ck.stage != Stage::Compute || fn.cpu_only || fn.uses_f64) return;
    if (opts.emit_ptx) {
        ck.ptx = brass::target::PtxTarget::emit_function(gpu_fn);
    }
    if (opts.emit_msl) {
        brass::target::MslOptions mopts;
        mopts.workgroup_x = std::max(1u, ck.local_size[0]);
        mopts.workgroup_y = std::max(1u, ck.local_size[1]);
        mopts.workgroup_z = std::max(1u, ck.local_size[2]);
        brass::target::MslKernel mk = brass::target::MslTarget::compile(gpu_fn, mopts);
        ck.msl = std::move(mk.source);
        ck.msl_entry = std::move(mk.entry_name);
    }
}

} // namespace detail
} // namespace brokernel

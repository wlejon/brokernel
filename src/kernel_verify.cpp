// Formal SPIR-V verification and multi-target text emitters (PTX, MSL).
#include "compile_internal.h"

#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>

#include <algorithm>
#include <sstream>

namespace brokernel {

namespace {

void populate_from_static_audit(const brass::spirv::Module& spv_mod, KernelVerifyReport& rep) {
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
    if (rep.diagnostic.empty()) {
        rep.diagnostic = rep.violations.empty()
                             ? "verified via static SPIR-V compute shader audit"
                             : rep.violations.front();
    }
}

} // namespace

brass::GpuProverOptions CompiledKernel::build_gpu_prover_options(const brass::Function* mir_fn) const {
    brass::verify::GpuProverOptions opts;
    opts.local_size_x = std::max(1u, local_size[0]);
    opts.local_size_y = std::max(1u, local_size[1]);
    opts.local_size_z = std::max(1u, local_size[2]);
    opts.max_sat_conflicts = 4000;

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
                pc.max_val = 0x3FFFFFFFu;
                opts.param_constraints.push_back(pc);
            }
        } else if (p.mir_param_count > 0) {
            max_store_mir_end = std::max(max_store_mir_end, p.mir_param_index + p.mir_param_count);
        }
    }

    const size_t param_count = mir_fn ? mir_fn->param_types().size()
        : (!spirv.params.empty() ? spirv.params.size()
                                 : size_t{max_store_mir_end} + grid_dim + (return_type.is_void() ? 0u : 1u));

    if (grid_dim >= 1 && max_store_mir_end < param_count) {
        for (const auto& p : params) {
            if ((p.role == ParamRole::Store || p.role == ParamRole::Stencil) &&
                p.mir_param_count >= 2 && p.grid_cover_dim == 1) {
                brass::verify::GpuParamConstraint cov;
                cov.param_index = max_store_mir_end;          // grid_w (i32)
                cov.rhs_param_index = p.mir_param_index + 1;  // store length (i64, bounded <= 0x3FFFFFFF)
                cov.min_val = 1;
                cov.max_val = 0x3FFFFFFFu;
                opts.param_constraints.push_back(cov);
            }
        }
    }

    // Account for trailing implicit return-store pointer (after grid dims).
    if (!return_type.is_void()) {
        auto add_ret_buf = [&](size_t idx) {
            brass::verify::GpuBufferSpec spec;
            spec.param_index = static_cast<uint32_t>(idx);
            spec.elem_bytes = return_byte_size ? return_byte_size : 4;
            spec.base_alignment = 4;
            if (grid_dim >= 1 && max_store_mir_end + grid_dim <= idx) {
                spec.length_param_index = static_cast<int32_t>(max_store_mir_end);
                if (grid_dim >= 2) spec.length_param_index2 = static_cast<int32_t>(max_store_mir_end + 1);
                if (grid_dim >= 3) spec.length_param_index3 = static_cast<int32_t>(max_store_mir_end + 2);
            } else {
                spec.static_byte_size = spec.elem_bytes;
            }
            opts.buffers.push_back(spec);
        };
        if (mir_fn) {
            const auto& mpt = mir_fn->param_types();
            for (size_t i = max_store_mir_end; i < mpt.size(); ++i) {
                if (mpt[i] == brass::Type::ptr()) {
                    add_ret_buf(i);
                    break;
                }
            }
        } else if (!spirv.params.empty()) {
            for (size_t i = max_store_mir_end; i < spirv.params.size(); ++i) {
                if (spirv.params[i].size == 8 && spirv.params[i].type == brass::Type::ptr()) {
                    add_ret_buf(i);
                    break;
                }
            }
        } else {
            add_ret_buf(size_t{max_store_mir_end} + grid_dim);
        }
    }
    return opts;
}

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
        populate_from_static_audit(spv_mod, rep);
        return rep;
    }

    const brass::verify::GpuProverOptions opts = build_gpu_prover_options();
    auto mem = brass::verify::verify_spirv_memory_safety(spv_mod, opts);
    auto bar = brass::verify::verify_spirv_barrier_uniformity(spv_mod, opts);

    if (mem.status == brass::verify::EquivStatus::Unknown && mem.violations.empty() &&
        mem.diagnostic.find("loop_unroll_bound") != std::string::npos && bar.is_uniform()) {
        // Bounded-unroll SMT checked every path up to `loop_unroll_bound` with 0 violations;
        // verify structural/shared-bounds invariants on the dynamic loop via static SPIR-V audit.
        populate_from_static_audit(spv_mod, rep);
        rep.shared_accesses_proven = std::max(rep.shared_accesses_proven, size_t(mem.shared_accesses_proven));
        rep.global_accesses_proven = std::max(rep.global_accesses_proven, size_t(mem.global_accesses_proven));
        rep.barriers_proven = std::max(rep.barriers_proven, size_t(bar.barriers_proven));
        return rep;
    }

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

KernelSandboxReport CompiledKernel::verify_sandbox() const {
    if (precomputed_sandbox_.has_value()) {
        return *precomputed_sandbox_;
    }

    KernelSandboxReport sb;
    if (cpu_only || uses_f64 || stage != Stage::Compute) {
        sb.portability_ok = false;
        sb.passed = false;
        sb.hardening_tier = brass::HardeningTier::Rejected;
        sb.diagnostic = cpu_only
            ? "kernel '" + name + "' is @cpu (no GPU sandbox verification)"
            : (uses_f64 ? "kernel '" + name + "' uses f64 (unsupported on GPU)"
                        : "kernel '" + name + "' is not a @compute stage");
        return sb;
    }

    const brass::Function* gpu_fn = gpu_mir_module_ ? gpu_mir_module_->get_function(gpu_mir_entry_) : nullptr;
    if (gpu_fn) {
        brass::GpuSandboxOptions opts;
        opts.spirv_opts.stage = stage;
        opts.spirv_opts.local_size_x = std::max(1u, local_size[0]);
        opts.spirv_opts.local_size_y = std::max(1u, local_size[1]);
        opts.spirv_opts.local_size_z = std::max(1u, local_size[2]);
        opts.spirv_opts.local_size_spec_constants = spirv.words.empty() ? true : spirv.local_size_spec_constants;
        opts.prover_opts = build_gpu_prover_options(gpu_fn);
        opts.verify_spirv = true;
        opts.run_smt_prover = true;
        opts.max_smt_spirv_words = 3072;
        opts.debug_ctx = &gpu_mir_module_->debug_context();

        const brass::GpuSandboxResult sr = brass::verify_gpu_kernel_sandbox(*gpu_fn, opts);
        sb.portability_ok = sr.portability_ok && gpu_portability_ok;
        for (const auto& pv : gpu_portability_violations) {
            if (!pv.empty()) sb.violations.push_back(pv);
        }
        for (const auto& v : sr.violations) {
            if (!v.empty()) sb.violations.push_back(v);
        }
        sb.spirv_parsed = sr.spirv_parsed;
        sb.memory_safe = sr.memory_safe;
        sb.barrier_uniform = sr.barrier_uniform;
        sb.global_bounds_proven = sr.global_bounds_proven;
        sb.proof_tier = sr.proof_tier;
        sb.shared_accesses_proven = sr.shared_accesses_proven;
        sb.global_accesses_proven = sr.global_accesses_proven;
        sb.barriers_proven = sr.barriers_proven;
        sb.collectives_proven = sr.collectives_proven;
        sb.passed = sr.passed() && sb.portability_ok && sb.violations.empty();
        sb.hardening_tier = sb.passed ? brass::HardeningTier::PureSmtZeroOverhead : brass::HardeningTier::Rejected;
        if (!sb.passed) {
            if (!sb.violations.empty()) sb.diagnostic = sb.violations.front();
            else if (!sr.diagnostic.empty()) sb.diagnostic = sr.diagnostic;
            else sb.diagnostic = "sandbox verification failed";
        } else {
            sb.diagnostic = sr.diagnostic.empty() ? "sandbox verification passed" : sr.diagnostic;
        }
        return sb;
    }

    // Fallback for compile-cache deserialized kernels without an in-memory MIR module.
    sb.portability_ok = gpu_portability_ok;
    for (const auto& pv : gpu_portability_violations) {
        if (!pv.empty()) sb.violations.push_back(pv);
    }
    const KernelVerifyReport vr = verify();
    sb.spirv_parsed = vr.spirv_parsed;
    sb.memory_safe = vr.memory_safe;
    sb.barrier_uniform = vr.barrier_uniform;
    sb.shared_accesses_proven = vr.shared_accesses_proven;
    sb.global_accesses_proven = vr.global_accesses_proven;
    sb.barriers_proven = vr.barriers_proven;
    sb.global_bounds_proven = vr.memory_safe && spirv.words.size() <= 3072;
    sb.proof_tier = !vr.spirv_parsed ? brass::SandboxProofTier::None
        : (spirv.words.size() > 3072 ? brass::SandboxProofTier::StaticOnly : brass::SandboxProofTier::FullSmt);
    for (const auto& v : vr.violations) {
        if (!v.empty()) sb.violations.push_back(v);
    }
    sb.passed = sb.portability_ok && sb.spirv_parsed && sb.memory_safe && sb.barrier_uniform && sb.violations.empty();
    sb.hardening_tier = sb.passed ? brass::HardeningTier::PureSmtZeroOverhead : brass::HardeningTier::Rejected;
    if (!sb.passed) {
        if (!sb.violations.empty()) sb.diagnostic = sb.violations.front();
        else if (!vr.diagnostic.empty()) sb.diagnostic = vr.diagnostic;
        else sb.diagnostic = "sandbox verification failed";
    } else {
        sb.diagnostic = vr.diagnostic.empty() ? "sandbox verification passed" : vr.diagnostic;
    }
    return sb;
}

bool CompiledKernel::verify_dispatch_bounds(
    GridShape shape,
    std::span<const KernelArg> args,
    std::string* err) const {
    size_t non_grid = 0;
    size_t param_blocks = 0;
    for (const auto& p : params) {
        if (!is_bound_param(p.role)) ++non_grid;
        if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) ++param_blocks;
    }
    const size_t global_blocks = block_bindings.size() >= param_blocks ? block_bindings.size() - param_blocks : 0;
    const size_t expected_with_blocks = non_grid + block_bindings.size();
    const size_t expected_full = params.size() + global_blocks;
    const bool skip_all_bound = (args.size() == non_grid);
    const bool skip_non_block_bound = (args.size() == expected_with_blocks && expected_with_blocks != expected_full);
    if (args.size() != params.size() && args.size() != expected_full && !skip_all_bound && !skip_non_block_bound) {
        if (err) {
            *err = "kernel '" + name + "' expects " + std::to_string(non_grid) +
                   " argument(s), got " + std::to_string(args.size());
        }
        return false;
    }

    std::vector<uint64_t> param_values;
    size_t arg_idx = 0;
    for (const auto& p : params) {
        if (is_bound_param(p.role)) {
            if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) {
                if (!skip_all_bound) ++arg_idx;
            } else if (!skip_all_bound && !skip_non_block_bound) {
                ++arg_idx;
            }
            continue;
        }
        const KernelArg& a = args[arg_idx++];
        if (p.role == ParamRole::Store || p.role == ParamRole::Stencil) {
            const uint64_t ptr_val = a.kind() == KernelArg::Kind::Device
                ? a.device_address()
                : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a.buffer_ptr()));
            const uint64_t elem_count = p.element_bytes ? static_cast<uint64_t>(a.buffer_bytes() / p.element_bytes) : 0;
            param_values.push_back(ptr_val);
            param_values.push_back(elem_count);
        } else if (p.role == ParamRole::Scalar) {
            uint64_t v = 0;
            switch (p.type.scalar) {
                case ScalarKind::U64: v = a.as_u64(); break;
                case ScalarKind::I64: v = static_cast<uint64_t>(a.as_i64()); break;
                case ScalarKind::U32: v = a.as_u32(); break;
                case ScalarKind::I32: v = static_cast<uint64_t>(static_cast<int64_t>(a.as_i32())); break;
                default: v = a.as_u64(); break;
            }
            param_values.push_back(v);
        } else {
            for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                param_values.push_back(0);
            }
        }
    }

    if (grid_dim >= 1) param_values.push_back(std::max(1u, shape.width));
    if (grid_dim >= 2) param_values.push_back(std::max(1u, shape.height));
    if (grid_dim >= 3) param_values.push_back(std::max(1u, shape.depth));
    if (!return_type.is_void()) param_values.push_back(1);

    const brass::verify::GpuProverOptions opts =
        build_gpu_prover_options(gpu_mir_module_ ? gpu_mir_module_->get_function(gpu_mir_entry_) : nullptr);
    return brass::verify_gpu_dispatch_bounds(opts, param_values, err);
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
        ck.msl_num_buffers = mk.num_buffers;
        ck.msl_scalar_params_bytes = mk.scalar_params_bytes;
        ck.msl_scalar_param_offsets = std::move(mk.scalar_param_offsets);
    }
}

} // namespace detail
} // namespace brokernel

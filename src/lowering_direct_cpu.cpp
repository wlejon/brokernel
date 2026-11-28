#include "lowering_internal.h"

#include <brass/mir/cross_module.hpp>
#include <brass/target/host_cpu.hpp>

#include <cmath>

namespace brokernel {
namespace {

bool uniform_store_array_scalar(const TypeRegistry& types,
                                const Type& elem,
                                ScalarKind& out_bucket,
                                uint32_t& out_elems_per_item) {
    auto kinds = types.flatten_scalars(elem);
    if (kinds.empty()) return false;
    auto to_bucket = [](ScalarKind sk, ScalarKind& bkt) -> bool {
        switch (sk) {
            case ScalarKind::F32:
            case ScalarKind::DF32: bkt = ScalarKind::F32; return true;
            case ScalarKind::F64:  bkt = ScalarKind::F64; return true;
            case ScalarKind::I32:
            case ScalarKind::Bool: bkt = ScalarKind::I32; return true;
            case ScalarKind::U32:  bkt = ScalarKind::U32; return true;
            default: return false;
        }
    };
    ScalarKind first{};
    if (!to_bucket(kinds[0], first)) return false;
    for (size_t i = 1; i < kinds.size(); ++i) {
        ScalarKind bkt{};
        if (!to_bucket(kinds[i], bkt) || bkt != first) return false;
    }
    uint32_t scalar_bytes = 0;
    switch (first) {
        case ScalarKind::F64: scalar_bytes = 8; break;
        case ScalarKind::F32:
        case ScalarKind::I32:
        case ScalarKind::U32: scalar_bytes = 4; break;
        default: return false;
    }
    const uint32_t elem_bytes = types.byte_size(elem);
    if (elem_bytes == 0 || elem_bytes % scalar_bytes != 0) return false;
    out_bucket = first;
    out_elems_per_item = elem_bytes / scalar_bytes;
    return true;
}

bool to_bronze_mir_scalar(ScalarKind sk, brass::Type& out_type) {
    switch (sk) {
        case ScalarKind::F32:
        case ScalarKind::F64:
        case ScalarKind::DF32:
            out_type = brass::Type::f64();
            return true;
        case ScalarKind::Bool:
        case ScalarKind::I32:
        case ScalarKind::U32:
            out_type = brass::Type::i32();
            return true;
        default:
            return false;
    }
}

brass::Value* unpack_bronze_scalar_leaf(brass::Builder& b, brass::Value* raw, ScalarKind sk, std::vector<brass::Value*>& out) {
    switch (sk) {
        case ScalarKind::F32:
            out.push_back(b.build_fptrunc_f32_f64(raw));
            return out.back();
        case ScalarKind::F64:
            out.push_back(raw);
            return out.back();
        case ScalarKind::DF32: {
            brass::Value* hi = b.build_fptrunc_f32_f64(raw);
            brass::Value* lo = b.build_fptrunc_f32_f64(b.build_sub(raw, b.build_fpext_f64_f32(hi)));
            out.push_back(hi);
            out.push_back(lo);
            return hi;
        }
        case ScalarKind::Bool:
            out.push_back(b.build_ne(raw, b.build_iconst_i32(0)));
            return out.back();
        case ScalarKind::I32:
        case ScalarKind::U32:
            out.push_back(raw);
            return out.back();
        default:
            return nullptr;
    }
}

} // namespace

bool lower_kernel_direct_cpu_mir(
    const CheckedModule& mod,
    const CheckedFunction& fn,
    std::shared_ptr<const brass::Module>& out_mod,
    std::string& out_entry,
    std::string* err) {
    out_mod.reset();
    out_entry.clear();

    if (!fn.is_entry_kernel || fn.stage != Stage::Compute || fn.uses_textures ||
        fn.uses_grid_coords || fn.uses_barriers_or_shared || fn.uses_telemetry) {
        if (err) *err = "direct CPU MIR only supports stateless non-grid kernels";
        return false;
    }

    brass::Type mir_ret = brass::Type::void_type();
    if (!fn.return_type.is_void()) {
        if (!fn.return_type.is_scalar() || !to_bronze_mir_scalar(fn.return_type.scalar, mir_ret)) {
            if (err) *err = "unsupported return type for direct CPU MIR";
            return false;
        }
    }

    struct ParamPlan {
        bool is_store = false;
        ScalarKind scalar = ScalarKind::Void;
        uint32_t elems_per_item = 1;
        size_t mir_idx = 0;
    };
    std::vector<ParamPlan> plans(fn.params.size());
    std::vector<brass::Type> mir_params;
    for (size_t i = 0; i < fn.params.size(); ++i) {
        const CheckedParam& cp = fn.params[i];
        plans[i].mir_idx = mir_params.size();
        if (cp.role == ParamRole::Scalar) {
            brass::Type pt = brass::Type::void_type();
            if (!cp.type.is_scalar() || !to_bronze_mir_scalar(cp.type.scalar, pt)) {
                if (err) *err = "unsupported scalar parameter '" + cp.name + "' for direct CPU MIR";
                return false;
            }
            plans[i].scalar = cp.type.scalar;
            mir_params.push_back(pt);
        } else if (cp.role == ParamRole::Store && cp.type.dimension <= 1) {
            ScalarKind bkt = ScalarKind::Void;
            uint32_t per_item = 1;
            if (!uniform_store_array_scalar(mod.types, cp.type.element_type(), bkt, per_item)) {
                if (err) *err = "store parameter '" + cp.name + "' cannot be mapped to typed array";
                return false;
            }
            plans[i].is_store = true;
            plans[i].elems_per_item = per_item;
            mir_params.push_back(brass::Type::i64()); // TypedArray raw data pointer
            mir_params.push_back(brass::Type::i32()); // TypedArray element length
        } else {
            if (err) *err = "unsupported parameter role '" + cp.name + "' for direct CPU MIR";
            return false;
        }
    }

    auto mir_mod = std::make_shared<brass::Module>("brokernel_direct_" + fn.name);
    std::string entry_name = "__bk_direct_" + fn.name;
    brass::Function* mir_fn = mir_mod->create_function(entry_name, mir_ret, mir_params);
    if (!mir_fn) {
        if (err) *err = "failed to create direct CPU MIR function";
        return false;
    }

    brass::codegen::KernelBuilder kb(*mir_mod, mir_fn);
    brass::Builder& b = kb.builder();
    brass::BasicBlock* entry = b.append_block("entry");
    std::vector<brass::Value*> arg_vals;
    arg_vals.reserve(mir_params.size());
    for (const auto& pt : mir_params) arg_vals.push_back(b.add_block_param(entry, pt));
    b.position_at_end(entry);

    DiagnosticBag diags;
    detail::MirBodyLowerer lowerer(mod, fn, diags, kb, /*is_gpu=*/false);
    lowerer.scopes.emplace_back();
    lowerer.gx = b.build_iconst_i32(0);
    lowerer.gy = b.build_iconst_i32(0);
    lowerer.gz = b.build_iconst_i32(0);
    lowerer.gw = b.build_iconst_i32(1);
    lowerer.gh = b.build_iconst_i32(1);
    lowerer.gd = b.build_iconst_i32(1);
    lowerer.lid_x = b.build_iconst_i32(0);
    lowerer.lid_y = b.build_iconst_i32(0);
    lowerer.lid_z = b.build_iconst_i32(0);
    lowerer.wgid_x = b.build_iconst_i32(0);
    lowerer.wgid_y = b.build_iconst_i32(0);
    lowerer.wgid_z = b.build_iconst_i32(0);
    lowerer.lsz_x = b.build_iconst_i32(1);
    lowerer.lsz_y = b.build_iconst_i32(1);
    lowerer.lsz_z = b.build_iconst_i32(1);
    lowerer.nwg_x = b.build_iconst_i32(1);
    lowerer.nwg_y = b.build_iconst_i32(1);
    lowerer.nwg_z = b.build_iconst_i32(1);
    lowerer.cell_coords_in_grid = false;
    lowerer.grid_cover.assign(fn.params.size(), 0);
    lowerer.collect_dyn_indexed_vars(fn.body);
    lowerer.active_return_type = fn.return_type;
    lowerer.has_returned_var_id = lowerer.alloc_var("__has_returned", {b.build_iconst_i32(0)});
    if (!fn.return_type.is_void()) {
        lowerer.ret_val_var_id = lowerer.alloc_var("__ret_val", lowerer.make_zero_leaves(fn.return_type));
    }

    for (size_t i = 0; i < fn.params.size(); ++i) {
        const CheckedParam& cp = fn.params[i];
        const ParamPlan& pp = plans[i];
        if (pp.is_store) {
            brass::Value* ptr_v = arg_vals[pp.mir_idx];
            brass::Value* ta_len = arg_vals[pp.mir_idx + 1];
            brass::Value* count64 = b.build_zext_i64(ta_len);
            if (pp.elems_per_item > 1) {
                count64 = b.build_udiv(count64, b.build_iconst_i64(pp.elems_per_item));
            }
            detail::ValueList store_leaves = {ptr_v, count64};
            lowerer.alloc_var(cp.name, store_leaves);
            lowerer.bind_store_param(i, store_leaves);
        } else {
            detail::ValueList pvals;
            unpack_bronze_scalar_leaf(b, arg_vals[pp.mir_idx], pp.scalar, pvals);
            lowerer.alloc_var(cp.name, std::move(pvals));
        }
    }

    lowerer.lower_stmt_list(fn.body);

    if (diags.has_errors() || !lowerer.used_block_params.empty() ||
        !lowerer.texture_param.empty() || !lowerer.shared_vars.empty()) {
        if (err) {
            *err = diags.has_errors() ? diags.format_all()
                                      : "direct CPU MIR kernel accessed unsupported module resource";
        }
        return false;
    }

    if (fn.return_type.is_void()) {
        b.build_ret_void();
    } else {
        const auto& rvals = lowerer.var_values[lowerer.ret_val_var_id];
        brass::Value* out_val = nullptr;
        switch (fn.return_type.scalar) {
            case ScalarKind::F32:
                out_val = b.build_fpext_f64_f32(rvals[0]);
                break;
            case ScalarKind::F64:
                out_val = rvals[0];
                break;
            case ScalarKind::DF32:
                out_val = b.build_add(b.build_fpext_f64_f32(rvals[0]),
                                      b.build_fpext_f64_f32(rvals[1]));
                break;
            case ScalarKind::Bool:
                out_val = b.build_ne(rvals[0], b.build_iconst_i32(0));
                break;
            case ScalarKind::I32:
            case ScalarKind::U32:
                out_val = rvals[0];
                break;
            default:
                if (err) *err = "unsupported return scalar";
                return false;
        }
        b.build_ret(out_val);
    }

    mir_fn->rebuild_cfg_predecessors();
    brass::DomainAuditReport audit = brass::audit_function_domain(*mir_fn, *mir_mod);
    if (!audit.ok) {
        if (err) *err = "direct CPU MIR domain audit failed: " + audit.diagnostic;
        return false;
    }

    out_mod = std::move(mir_mod);
    out_entry = std::move(entry_name);
    return true;
}

} // namespace brokernel

// Runtime diagnostics lowering: `check("label", cond)`, `trap_non_finite`, and `probe_if("label", cond, expr)`.
#include "lowering_internal.h"

#include <algorithm>

namespace brokernel::detail {

namespace {

brass::Value* extract_scalar_bits_lo(brass::Builder& sb, brass::Value* sv, ScalarKind sk) {
    if (sk == ScalarKind::F32) return sb.build_bitcast_i32_f32(sv);
    if (sk == ScalarKind::F64) return sb.build_trunc_i32(sb.build_bitcast_i64_f64(sv));
    return sb.build_iconst_i32(0);
}

brass::Value* extract_scalar_bits_hi(brass::Builder& sb, brass::Value* sv, ScalarKind sk) {
    if (sk == ScalarKind::F64) {
        brass::Value* bits = sb.build_bitcast_i64_f64(sv);
        return sb.build_trunc_i32(sb.build_lshr(bits, sb.build_iconst_i64(32)));
    }
    return sb.build_iconst_i32(0);
}

brass::Value* encode_probe_if_scalar_u32(brass::Builder& sb, brass::Value* sv, ScalarKind sk) {
    switch (sk) {
        case ScalarKind::F32: return sb.build_bitcast_i32_f32(sv);
        case ScalarKind::F64: return sb.build_bitcast_i32_f32(sb.build_fptrunc_f32_f64(sv));
        case ScalarKind::Bool: return sb.build_ne(sv, sb.build_iconst_i32(0));
        case ScalarKind::I32:
        case ScalarKind::U32: return sv;
        default: return sb.build_iconst_i32(0);
    }
}

void emit_record_check_scalar(
    brass::codegen::KernelBuilder& kb,
    Ops& b,
    bool is_gpu,
    brass::Value* ptr,
    uint32_t kind,
    uint32_t label_id,
    brass::Value* gx_s,
    brass::Value* gy_s,
    brass::Value* gz_s,
    brass::Value* lo_u32,
    brass::Value* hi_u32) {
    brass::Builder& sb = b.builder();
    brass::Value* zero32 = sb.build_iconst_i32(0);
    brass::Value* one32 = sb.build_iconst_i32(1);
    brass::Value* old = is_gpu
        ? kb.atom_cas_u32(ptr, zero32, one32)
        : sb.build_load_indexed(brass::Type::i32(), ptr, sb.build_iconst_i64(0), 4, 0);

    brass::BasicBlock* win_bb = b.side_block("check_win");
    brass::BasicBlock* done_bb = b.side_block("check_done");
    sb.build_br_if(sb.build_eq(old, zero32), win_bb, done_bb);
    sb.position_at_end(win_bb);
    auto write_w = [&](int32_t idx, brass::Value* v) {
        sb.build_store_indexed(brass::Type::i32(), ptr, sb.build_iconst_i64(idx), 4, 0, v);
    };
    if (!is_gpu) write_w(0, one32);
    write_w(1, sb.build_iconst_i32(static_cast<int32_t>(kind)));
    write_w(2, gx_s);
    write_w(3, gy_s);
    write_w(4, gz_s);
    write_w(5, sb.build_iconst_i32(static_cast<int32_t>(label_id)));
    write_w(6, lo_u32);
    write_w(7, hi_u32);
    sb.build_br(done_bb);
    sb.position_at_end(done_bb);
}

void emit_record_probe_if_scalar(
    brass::codegen::KernelBuilder& kb,
    Ops& b,
    bool is_gpu,
    brass::Value* ptr,
    uint32_t probe_idx,
    uint32_t lanes_count,
    brass::Value* gx_s,
    brass::Value* gy_s,
    brass::Value* gz_s,
    brass::Value* w0,
    brass::Value* w1,
    brass::Value* w2,
    brass::Value* w3) {
    brass::Builder& sb = b.builder();
    brass::Value* one32 = sb.build_iconst_i32(1);
    brass::Value* slot = nullptr;
    if (is_gpu) {
        slot = kb.atom_add_u32(ptr, one32);
    } else {
        brass::Value* zero64 = sb.build_iconst_i64(0);
        slot = sb.build_load_indexed(brass::Type::i32(), ptr, zero64, 4, 0);
        sb.build_store_indexed(brass::Type::i32(), ptr, zero64, 4, 0, sb.build_add(slot, one32));
    }
    brass::Value* cap = sb.build_load_indexed(brass::Type::i32(), ptr, sb.build_iconst_i64(1), 4, 0);
    brass::BasicBlock* write_bb = b.side_block("probe_if_write");
    brass::BasicBlock* done_bb = b.side_block("probe_if_done");
    sb.build_br_if(sb.build_ult(slot, cap), write_bb, done_bb);
    sb.position_at_end(write_bb);
    brass::Value* base_idx =
        sb.build_add(sb.build_iconst_i64(2), sb.build_mul(sb.build_zext_i64(slot), sb.build_iconst_i64(9)));
    auto write_w = [&](int32_t off, brass::Value* v) {
        brass::Value* idx = off == 0 ? base_idx : sb.build_add(base_idx, sb.build_iconst_i64(off));
        sb.build_store_indexed(brass::Type::i32(), ptr, idx, 4, 0, v);
    };
    write_w(0, gx_s);
    write_w(1, gy_s);
    write_w(2, gz_s);
    write_w(3, sb.build_iconst_i32(static_cast<int32_t>(probe_idx)));
    write_w(4, sb.build_iconst_i32(static_cast<int32_t>(lanes_count)));
    write_w(5, w0);
    write_w(6, w1);
    write_w(7, w2);
    write_w(8, w3);
    sb.build_br(done_bb);
    sb.position_at_end(done_bb);
}

} // namespace

uint32_t MirBodyLowerer::register_check_label(const std::string& label) {
    for (size_t i = 0; i < check_labels.size(); ++i) {
        if (check_labels[i] == label) return static_cast<uint32_t>(i);
    }
    uint32_t id = static_cast<uint32_t>(check_labels.size());
    check_labels.push_back(label);
    return id;
}

void MirBodyLowerer::bind_diagnostic_gpu_params(LoweredKernelMir& out, uint32_t& mir_idx) {
    brass::Builder& sb = b.builder();
    if (mod.enable_probes && root_fn.stage == Stage::Compute && root_fn.has_conditional_probes()) {
        out.probe_if_out_mir_index = mir_idx;
        gpu_probe_if_ptr = sb.add_param(brass::Type::ptr());
        mir_idx += 1;
    }
    if ((mod.enable_checks || mod.trap_non_finite) && root_fn.stage == Stage::Compute) {
        out.check_out_mir_index = mir_idx;
        gpu_check_ptr = sb.add_param(brass::Type::ptr());
        mir_idx += 1;
    }
}

void MirBodyLowerer::bind_diagnostic_cpu_params(brass::Value* args_pack, int32_t& slot) {
    brass::Builder& sb = b.builder();
    brass::Value* zero64 = sb.build_iconst_i64(0);
    if (mod.enable_probes && root_fn.has_conditional_probes()) {
        cpu_probe_if_ptr = sb.build_load(brass::Type::ptr(), args_pack, slot * 8);
        cpu_probe_if_active = sb.build_ne(sb.build_load(brass::Type::i64(), args_pack, slot * 8), zero64);
        slot += 1;
    }
    if (mod.enable_checks || mod.trap_non_finite) {
        cpu_check_ptr = sb.build_load(brass::Type::ptr(), args_pack, slot * 8);
        cpu_check_active = sb.build_ne(sb.build_load(brass::Type::i64(), args_pack, slot * 8), zero64);
        slot += 1;
    }
}

void MirBodyLowerer::copy_diagnostic_cpu_ptrs(
    brass::Value* check_p, brass::Value* check_act, brass::Value* pif_p, brass::Value* pif_act) {
    cpu_check_ptr = check_p;
    cpu_check_active = check_act;
    cpu_probe_if_ptr = pif_p;
    cpu_probe_if_active = pif_act;
}

void MirBodyLowerer::emit_check(const std::string& label, brass::Value* cond) {
    if (!mod.enable_checks) return;
    brass::Value* ptr = is_gpu ? gpu_check_ptr : cpu_check_ptr;
    if (!ptr) return;

    brass::Builder& sb = b.builder();
    brass::Value* zero32 = sb.build_iconst_i32(0);
    brass::Value* bad = b.build_eq(cond, b.build_iconst_i32(0));
    if (b.mask) bad = b.build_and(b.mask, bad);
    brass::Value* any_bad = b.any(bad);
    brass::Value* trig = sb.build_ne(any_bad, zero32);
    if (!is_gpu) {
        if (!cpu_check_active) return;
        trig = sb.build_and(cpu_check_active, trig);
    }

    const uint32_t label_id = register_check_label(label);
    brass::BasicBlock* trap_bb = b.side_block("check_trap");
    brass::BasicBlock* cont_bb = b.side_block("check_cont");
    sb.build_br_if(trig, trap_bb, cont_bb);
    sb.position_at_end(trap_bb);
    if (!b.vector_mode()) {
        emit_record_check_scalar(kb, b, is_gpu, ptr, 1u, label_id, gx, gy, gz, zero32, zero32);
    } else {
        for (uint32_t i = 0; i < b.lanes; ++i) {
            brass::Value* lbad = b.lane_of(bad, i);
            brass::BasicBlock* lhit = b.side_block("check_lane_hit");
            brass::BasicBlock* lnext = b.side_block("check_lane_next");
            sb.build_br_if(sb.build_ne(lbad, zero32), lhit, lnext);
            sb.position_at_end(lhit);
            emit_record_check_scalar(
                kb, b, is_gpu, ptr, 1u, label_id,
                b.lane_of(gx, i), b.lane_of(gy, i), b.lane_of(gz, i), zero32, zero32);
            sb.build_br(lnext);
            sb.position_at_end(lnext);
        }
    }
    sb.build_br(cont_bb);
    sb.position_at_end(cont_bb);
}

ValueList MirBodyLowerer::lower_check_expr(const Expr* e) {
    ValueList cv = lower_expr(e->args[1].get());
    brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
    emit_check(e->args[0]->name, cond);
    return {cond};
}

void MirBodyLowerer::emit_probe_if(uint32_t probe_idx, brass::Value* cond, const Type& val_type, const ValueList& vals) {
    if (!mod.enable_probes) return;
    brass::Value* ptr = is_gpu ? gpu_probe_if_ptr : cpu_probe_if_ptr;
    if (!ptr) return;

    brass::Builder& sb = b.builder();
    brass::Value* zero32 = sb.build_iconst_i32(0);
    brass::Value* hit = cond;
    if (b.mask) hit = b.build_and(b.mask, hit);
    brass::Value* any_hit = b.any(hit);
    brass::Value* trig = sb.build_ne(any_hit, zero32);
    if (!is_gpu) {
        if (!cpu_probe_if_active) return;
        trig = sb.build_and(cpu_probe_if_active, trig);
    }

    const uint32_t lanes_count = val_type.is_vector() ? val_type.dimension : 1u;
    brass::BasicBlock* cap_bb = b.side_block("probe_if_cap");
    brass::BasicBlock* cont_bb = b.side_block("probe_if_cont");
    sb.build_br_if(trig, cap_bb, cont_bb);
    sb.position_at_end(cap_bb);
    if (!b.vector_mode()) {
        brass::Value* w[4] = {zero32, zero32, zero32, zero32};
        for (size_t c = 0; c < 4 && c < vals.size(); ++c) {
            w[c] = encode_probe_if_scalar_u32(sb, vals[c], val_type.scalar);
        }
        emit_record_probe_if_scalar(kb, b, is_gpu, ptr, probe_idx, lanes_count, gx, gy, gz, w[0], w[1], w[2], w[3]);
    } else {
        for (uint32_t i = 0; i < b.lanes; ++i) {
            brass::Value* lhit = b.lane_of(hit, i);
            brass::BasicBlock* lcap = b.side_block("probe_if_lane_cap");
            brass::BasicBlock* lnext = b.side_block("probe_if_lane_next");
            sb.build_br_if(sb.build_ne(lhit, zero32), lcap, lnext);
            sb.position_at_end(lcap);
            brass::Value* w[4] = {zero32, zero32, zero32, zero32};
            for (size_t c = 0; c < 4 && c < vals.size(); ++c) {
                w[c] = encode_probe_if_scalar_u32(sb, b.lane_of(vals[c], i), val_type.scalar);
            }
            emit_record_probe_if_scalar(
                kb, b, is_gpu, ptr, probe_idx, lanes_count,
                b.lane_of(gx, i), b.lane_of(gy, i), b.lane_of(gz, i), w[0], w[1], w[2], w[3]);
            sb.build_br(lnext);
            sb.position_at_end(lnext);
        }
    }
    sb.build_br(cont_bb);
    sb.position_at_end(cont_bb);
}

ValueList MirBodyLowerer::lower_probe_if_expr(const Expr* e) {
    if (!mod.enable_probes) {
        return lower_expr(e->args[2].get());
    }
    ValueList cv = lower_expr(e->args[1].get());
    ValueList vals = lower_expr(e->args[2].get());
    brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
    emit_probe_if(static_cast<uint32_t>(e->args[0]->int_val), cond, e->resolved_type, vals);
    return vals;
}

void MirBodyLowerer::emit_trap_non_finite(
    const std::string& label,
    const std::vector<ScalarKind>& kinds,
    const ValueList& vals,
    const Ops::Bound* bound) {
    if (!mod.trap_non_finite) return;
    brass::Value* ptr = is_gpu ? gpu_check_ptr : cpu_check_ptr;
    if (!ptr) return;

    brass::Builder& sb = b.builder();
    brass::Value* zero32 = sb.build_iconst_i32(0);

    for (size_t k = 0; k < kinds.size() && k < vals.size(); ++k) {
        const ScalarKind sk = kinds[k];
        if (sk != ScalarKind::F32 && sk != ScalarKind::F64) continue;
        brass::Value* v = vals[k];
        brass::Value* bad = nullptr;
        if (sk == ScalarKind::F64) {
            brass::Value* bits = b.build_bitcast_i64_f64(v);
            brass::Value* exp_mask = b.build_iconst_i64(0x7ff0000000000000LL);
            bad = b.build_eq(b.build_and(bits, exp_mask), exp_mask);
        } else {
            brass::Value* bits = b.build_bitcast_i32_f32(v);
            brass::Value* exp_mask = b.build_iconst_i32(0x7f800000);
            bad = b.build_eq(b.build_and(bits, exp_mask), exp_mask);
        }
        if (b.mask) bad = b.build_and(b.mask, bad);
        brass::Value* any_bad = b.any(bad);
        brass::Value* trig = sb.build_ne(any_bad, zero32);
        if (!is_gpu) {
            if (!cpu_check_active) return;
            trig = sb.build_and(cpu_check_active, trig);
        }
        const uint32_t label_id = register_check_label(label);
        brass::BasicBlock* trap_bb = b.side_block("nf_trap");
        brass::BasicBlock* cont_bb = b.side_block("nf_cont");
        sb.build_br_if(trig, trap_bb, cont_bb);
        sb.position_at_end(trap_bb);
        if (!b.vector_mode()) {
            brass::Value* cond_s = sb.build_iconst_i32(1);
            if (bound && bound->count) {
                cond_s = sb.build_ult(bound->index, bound->count);
            }
            brass::BasicBlock* lhit = b.side_block("nf_hit");
            brass::BasicBlock* ldone = b.side_block("nf_done");
            sb.build_br_if(cond_s, lhit, ldone);
            sb.position_at_end(lhit);
            emit_record_check_scalar(
                kb, b, is_gpu, ptr, 2u, label_id, gx, gy, gz,
                extract_scalar_bits_lo(sb, v, sk), extract_scalar_bits_hi(sb, v, sk));
            sb.build_br(ldone);
            sb.position_at_end(ldone);
        } else {
            for (uint32_t i = 0; i < b.lanes; ++i) {
                brass::Value* lbad = sb.build_ne(b.lane_of(bad, i), zero32);
                if (bound && bound->count) {
                    lbad = sb.build_and(lbad, sb.build_ult(b.lane_of(bound->index, i), bound->count));
                }
                brass::BasicBlock* lhit = b.side_block("nf_lane_hit");
                brass::BasicBlock* lnext = b.side_block("nf_lane_next");
                sb.build_br_if(lbad, lhit, lnext);
                sb.position_at_end(lhit);
                brass::Value* sv = b.lane_of(v, i);
                emit_record_check_scalar(
                    kb, b, is_gpu, ptr, 2u, label_id,
                    b.lane_of(gx, i), b.lane_of(gy, i), b.lane_of(gz, i),
                    extract_scalar_bits_lo(sb, sv, sk), extract_scalar_bits_hi(sb, sv, sk));
                sb.build_br(lnext);
                sb.position_at_end(lnext);
            }
        }
        sb.build_br(cont_bb);
        sb.position_at_end(cont_bb);
    }
}

void MirBodyLowerer::trap_non_finite_type(
    const std::string& label, const Type& t, const ValueList& vals, const Ops::Bound* bound) {
    if (!mod.trap_non_finite) return;
    emit_trap_non_finite(label, mod.types.flatten_scalars(t), vals, bound);
}

void MirBodyLowerer::trap_non_finite_store(
    const Type& elem_type, const ValueList& store, const ValueList& vals, const Ops::Bound* bound) {
    if (!mod.trap_non_finite || store.empty()) return;
    auto it = store_param.find(store[0]);
    if (it == store_param.end() || it->second >= root_fn.params.size()) return;
    trap_non_finite_type(root_fn.params[it->second].name, elem_type, vals, bound);
}

void MirBodyLowerer::trap_non_finite_return(const Type& ret_type, const ValueList& vals) {
    if (!mod.trap_non_finite) return;
    if (root_fn.uses_barriers_or_shared && root_fn.grid_dim > 0) {
        brass::Builder& sb = b.builder();
        brass::Value* inb = sb.build_ult(gx, gw);
        if (root_fn.grid_dim >= 2) inb = sb.build_and(inb, sb.build_ult(gy, gh));
        if (root_fn.grid_dim >= 3) inb = sb.build_and(inb, sb.build_ult(gz, gd));
        brass::BasicBlock* ok_bb = b.side_block("nf_ret_inb");
        brass::BasicBlock* skip_bb = b.side_block("nf_ret_skip");
        sb.build_br_if(inb, ok_bb, skip_bb);
        sb.position_at_end(ok_bb);
        trap_non_finite_type("<return>", ret_type, vals, nullptr);
        sb.build_br(skip_bb);
        sb.position_at_end(skip_bb);
        return;
    }
    trap_non_finite_type("<return>", ret_type, vals, nullptr);
}

} // namespace brokernel::detail

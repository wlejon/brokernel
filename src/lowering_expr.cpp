#include "lowering_internal.h"
#include <algorithm>
#include <cmath>
namespace brokernel::detail {
brass::Type to_mir_scalar_type(ScalarKind sk) {
    switch (sk) {
        case ScalarKind::Bool:
        case ScalarKind::I32:
        case ScalarKind::U32:
            return brass::Type::i32();
        case ScalarKind::I64:
        case ScalarKind::U64:
            return brass::Type::i64();
        case ScalarKind::F32:
            return brass::Type::f32();
        case ScalarKind::F64:
            return brass::Type::f64();
        default:
            return brass::Type::i32();
    }
}
uint32_t swizzle_index(char c) {
    if (c == 'x' || c == 'r' || c == 's') return 0;
    if (c == 'y' || c == 'g' || c == 't') return 1;
    if (c == 'z' || c == 'b' || c == 'p') return 2;
    if (c == 'w' || c == 'a' || c == 'q') return 3;
    return 0;
}
bool stmt_has_return(const Stmt* s) {
    if (!s) return false;
    switch (s->kind) {
        case Stmt::Kind::Return:
        case Stmt::Kind::Break:
        case Stmt::Kind::Continue:
            return true;
        case Stmt::Kind::Block:
        case Stmt::Kind::While:
        case Stmt::Kind::ForRange:
        case Stmt::Kind::ForC:
            for (const auto& b : s->body) {
                if (stmt_has_return(b.get())) return true;
            }
            return false;
        case Stmt::Kind::If:
            for (const auto& b : s->body) {
                if (stmt_has_return(b.get())) return true;
            }
            for (const auto& eb : s->else_body) {
                if (stmt_has_return(eb.get())) return true;
            }
            return false;
        default:
            return false;
    }
}
int MirBodyLowerer::alloc_var(const std::string& name, ValueList initial_leaves) {
        int id = next_var_id++;
        var_values[id] = std::move(initial_leaves);
        if (!name.empty() && !scopes.empty()) {
            scopes.back()[name] = id;
        }
        return id;
    }
int MirBodyLowerer::lookup_var_id(const std::string& name) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto vit = it->find(name);
            if (vit != it->end()) return vit->second;
        }
        return -1;
    }
ValueList MirBodyLowerer::make_zero_leaves(const Type& t) {
        auto kinds = mod.types.flatten_scalars(t);
        ValueList out;
        out.reserve(kinds.size());
        for (ScalarKind sk : kinds) {
            switch (sk) {
                case ScalarKind::F32: out.push_back(b.build_fconst_f32(0.0f)); break;
                case ScalarKind::F64: out.push_back(b.build_fconst_f64(0.0)); break;
                case ScalarKind::I64:
                case ScalarKind::U64: out.push_back(b.build_iconst_i64(0)); break;
                default: out.push_back(b.build_iconst_i32(0)); break;
            }
        }
        return out;
    }
std::vector<int> MirBodyLowerer::sorted_live_var_ids() const {
        std::vector<int> ids;
        ids.reserve(var_values.size());
        for (const auto& [id, _] : var_values) ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        return ids;
    }
std::vector<brass::Value*> MirBodyLowerer::pack_var_ids(const std::vector<int>& ids) const {
        std::vector<brass::Value*> flat;
        for (int id : ids) {
            auto it = var_values.find(id);
            if (it != var_values.end()) {
                for (brass::Value* v : it->second) flat.push_back(v);
            }
        }
        return flat;
    }
ValueList MirBodyLowerer::lower_inlined_call(
    const CheckedFunction& callee,
    const std::vector<ValueList>& arg_vals,
    const std::vector<std::optional<BlockPlace>>& arg_places) {
    // Save outer return tracking, loop stack, and scopes
    int saved_has_ret = has_returned_var_id;
    int saved_ret_val = ret_val_var_id;
    Type saved_ret_type = active_return_type;
    std::string saved_fn_name = active_fn_name;
    auto saved_scopes = std::move(scopes);
    auto saved_loops = std::move(loop_stack);
    scopes.clear();
    loop_stack.clear();
    push_scope();
    active_return_type = callee.return_type;
    active_fn_name = callee.name;
    has_returned_var_id = alloc_var("__callee_has_ret", {b.build_iconst_i32(0)});
    if (!callee.return_type.is_void()) {
        ret_val_var_id = alloc_var("__callee_ret_val", make_zero_leaves(callee.return_type));
    } else {
        ret_val_var_id = -1;
    }
    for (size_t i = 0; i < callee.params.size() && i < arg_vals.size(); ++i) {
        int id = alloc_var(callee.params[i].name, arg_vals[i]);
        if (i < arg_places.size() && arg_places[i].has_value()) {
            block_places[id] = *arg_places[i];
        }
    }
    lower_stmt_list(callee.body);
    ValueList result;
    if (ret_val_var_id >= 0) {
        result = var_values[ret_val_var_id];
    }
    pop_scope();
    var_values.erase(has_returned_var_id);
    if (ret_val_var_id >= 0) var_values.erase(ret_val_var_id);
    scopes = std::move(saved_scopes);
    loop_stack = std::move(saved_loops);
    has_returned_var_id = saved_has_ret;
    ret_val_var_id = saved_ret_val;
    active_return_type = saved_ret_type;
    active_fn_name = std::move(saved_fn_name);
    return result;
}
ValueList MirBodyLowerer::lower_expr(const Expr* e) {
    if (!e) return {};
    switch (e->kind) {
        case Expr::Kind::IntLit: {
            ScalarKind sk = e->resolved_type.scalar;
            if (sk == ScalarKind::I64 || sk == ScalarKind::U64) {
                return {b.build_iconst_i64(e->int_val)};
            }
            return {b.build_iconst_i32(static_cast<int32_t>(e->int_val))};
        }
        case Expr::Kind::FloatLit: {
            if (e->resolved_type.scalar == ScalarKind::F64) {
                return {b.build_fconst_f64(e->float_val)};
            }
            return {b.build_fconst_f32(static_cast<float>(e->float_val))};
        }
        case Expr::Kind::BoolLit:
            return {e->bool_val ? b.true_value() : b.build_iconst_i32(0)};
        case Expr::Kind::Ident: {
            if (auto v = try_shared_read(e)) return *v;
            if (e->global_index >= 0) return global_value(e);
            if (auto v = try_block_read(e)) return *v;
            int id = lookup_var_id(e->name);
            if (id >= 0) return var_values[id];
            return make_zero_leaves(e->resolved_type);
        }
        case Expr::Kind::Path:
            if (auto v = try_shared_read(e)) return *v;
            if (e->global_index >= 0) return global_value(e);
            return make_zero_leaves(e->resolved_type);
        case Expr::Kind::Unary: {
            ValueList sub = lower_expr(e->lhs.get());
            ValueList out;
            out.reserve(sub.size());
            for (brass::Value* v : sub) {
                if (e->unary_op == UnaryOp::Neg) {
                    out.push_back(b.build_neg(v));
                } else if (e->unary_op == UnaryOp::Not) {
                    out.push_back(b.build_eq(v, b.build_iconst_i32(0)));
                } else if (e->unary_op == UnaryOp::BitNot) {
                    out.push_back(b.build_not(v));
                } else {
                    out.push_back(v);
                }
            }
            return out;
        }
        case Expr::Kind::Binary: {
            if (e->binary_op == BinaryOp::LogAnd || e->binary_op == BinaryOp::LogOr) {
                ValueList lv = lower_expr(e->lhs.get());
                brass::Value* l = lv.empty() ? b.build_iconst_i32(0) : lv[0];
                brass::Value* cond = (e->binary_op == BinaryOp::LogAnd)
                                         ? l
                                         : b.build_eq(l, b.build_iconst_i32(0));
                if (b.vector_mode()) {
                    brass::Value* outer = b.mask;
                    b.mask = outer ? b.build_and(outer, cond) : cond;
                    ValueList rv = lower_expr(e->rhs.get());
                    b.mask = outer;
                    brass::Value* r = rv.empty() ? b.build_iconst_i32(0) : rv[0];
                    return {(e->binary_op == BinaryOp::LogAnd) ? b.build_and(l, r) : b.build_or(l, r)};
                }
                int res_id = alloc_var("", {l});
                lower_branch(
                    cond,
                    [&] { var_values[res_id] = lower_expr(e->rhs.get()); },
                    nullptr);
                brass::Value* res = var_values[res_id][0];
                var_values.erase(res_id);
                return {res};
            }
            ValueList lv = lower_expr(e->lhs.get());
            ValueList rv = lower_expr(e->rhs.get());
            size_t n = std::max(lv.size(), rv.size());
            ValueList out;
            out.reserve(n);
            ScalarKind sk = e->lhs->resolved_type.scalar;
            for (size_t i = 0; i < n; ++i) {
                brass::Value* l = lv[i < lv.size() ? i : 0];
                brass::Value* r = rv[i < rv.size() ? i : 0];
                out.push_back(lower_scalar_binary(e->binary_op, l, r, sk));
            }
            return out;
        }
        case Expr::Kind::Select: {
            ValueList cv = lower_expr(e->cond.get());
            brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
            if (const brass::Instruction* def = cond->is_instruction() ? cond->defining_instruction() : nullptr;
                def && def->opcode() == brass::Opcode::iconst_i32) {
                return lower_expr(def->imm_i32() != 0 ? e->lhs.get() : e->rhs.get());
            }
            ValueList lv = lower_expr(e->lhs.get());
            ValueList rv = lower_expr(e->rhs.get());
            ValueList out;
            out.reserve(lv.size());
            for (size_t i = 0; i < lv.size() && i < rv.size(); ++i) {
                out.push_back(b.build_select(cond, lv[i], rv[i]));
            }
            return out;
        }
        case Expr::Kind::Cast: {
            ValueList sub = lower_expr(e->lhs.get());
            ScalarKind from_sk = e->lhs->resolved_type.scalar;
            ScalarKind to_sk = e->resolved_type.scalar;
            ValueList out;
            out.reserve(sub.size());
            for (brass::Value* v : sub) {
                out.push_back(lower_scalar_cast(v, from_sk, to_sk));
            }
            return out;
        }
        case Expr::Kind::Field: {
            if (auto v = try_block_read(e)) return *v;
            if (auto v = try_shared_read(e)) return *v;
            ValueList recv = lower_expr(e->lhs.get());
            const Type& rt = e->lhs->resolved_type;
            if (rt.is_scalar()) return recv;
            if (rt.is_vector()) {
                ValueList out;
                out.reserve(e->name.size());
                for (char c : e->name) {
                    uint32_t idx = swizzle_index(c);
                    out.push_back(idx < recv.size() ? recv[idx] : recv[0]);
                }
                return out;
            }
            if (rt.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(rt.struct_id);
                if (sdef) {
                    if (const StructField* sf = sdef->find_field(e->name)) {
                        ValueList out;
                        for (uint32_t i = 0; i < sf->flat_leaf_count; ++i) {
                            uint32_t idx = sf->flat_leaf_offset + i;
                            if (idx < recv.size()) out.push_back(recv[idx]);
                        }
                        return out;
                    }
                }
            }
            return recv;
        }
        case Expr::Kind::Index: {
            if (auto v = try_block_read(e)) return *v;
            if (auto v = try_shared_read(e)) return *v;
            const Type& rt = e->lhs->resolved_type;
            if (rt.is_stencil()) return lower_stencil_read(e->lhs.get(), rt, e->args);
            if (rt.is_store()) {
                ValueList base = lower_expr(e->lhs.get());
                ValueList x = e->args.size() >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList y = e->args.size() >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList z = e->args.size() >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                brass::Value* lin = compute_linear_index(rt.dimension, x[0], y[0], z[0]);
                return load_element_from_buffer(rt.element_type(), store_access(base, rt.dimension, x[0], y[0], z[0]), lin);
            }
            // Vector or array indexing
            ValueList recv = lower_expr(e->lhs.get());
            if (rt.is_scalar() || recv.size() <= 1) return recv;
            ValueList idx_v = lower_expr(e->args[0].get());
            brass::Value* idx = idx_v.empty() ? b.build_iconst_i32(0) : idx_v[0];
            if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
            Type elem_t = rt.element_type();
            uint32_t elem_leaves = mod.types.flat_leaf_count(elem_t);
            uint32_t count = rt.dimension;
            ValueList res(elem_leaves, recv[0]);
            for (uint32_t k = 0; k < elem_leaves; ++k) {
                brass::Value* acc = recv[k];
                for (uint32_t i = 1; i < count; ++i) {
                    brass::Value* eq = b.build_eq(idx, b.build_iconst_i32(static_cast<int32_t>(i)));
                    acc = b.build_select(eq, recv[i * elem_leaves + k], acc);
                }
                res[k] = acc;
            }
            return res;
        }
        case Expr::Kind::StructLit: {
            const StructDef* sdef = mod.types.find_struct(e->resolved_type.struct_id);
            ValueList res = make_zero_leaves(e->resolved_type);
            if (!sdef) return res;
            for (const auto& fi : e->fields) {
                if (const StructField* sf = sdef->find_field(fi.name)) {
                    ValueList fv = lower_expr(fi.value.get());
                    for (uint32_t k = 0; k < sf->flat_leaf_count && k < fv.size(); ++k) {
                        res[sf->flat_leaf_offset + k] = fv[k];
                    }
                }
            }
            return res;
        }
        case Expr::Kind::MethodCall: {
            const Type& rt = e->lhs->resolved_type;
            const std::string& m = e->name;
            if (rt.is_grid()) {
                if (m == "x") return {gx};
                if (m == "y") return {gy};
                if (m == "z") return {gz};
                if (m == "width") return {gw};
                if (m == "height") return {gh};
                if (m == "depth") return {gd};
                if (m == "linear_index") {
                    brass::Value* lin = compute_linear_index(rt.dimension, gx, gy, gz);
                    linear_indices.insert(lin);
                    return {lin};
                }
                if (m == "total_cells") {
                    brass::Value* total = b.build_sext_i64(gw);
                    if (rt.dimension >= 2) total = b.build_mul(total, b.build_sext_i64(gh));
                    if (rt.dimension >= 3) total = b.build_mul(total, b.build_sext_i64(gd));
                    return {total};
                }
                if (m == "coord" || m == "pos") {
                    if (rt.dimension == 1) return {gx};
                    if (rt.dimension == 2) return {gx, gy};
                    return {gx, gy, gz};
                }
                if (m == "uv") {
                    auto uv_comp = [&](brass::Value* c, brass::Value* ext) {
                        brass::Value* cf = b.build_add(b.build_sitofp_f32_i32(c), b.build_fconst_f32(0.5f));
                        return b.build_sdiv(cf, b.build_sitofp_f32_i32(ext));
                    };
                    if (rt.dimension == 1) return {uv_comp(gx, gw)};
                    if (rt.dimension == 2) return {uv_comp(gx, gw), uv_comp(gy, gh)};
                    return {uv_comp(gx, gw), uv_comp(gy, gh), uv_comp(gz, gd)};
                }
                if (m == "uv_f64") {
                    auto uv_comp = [&](brass::Value* c, brass::Value* ext) {
                        brass::Value* cf = b.build_add(b.build_sitofp_f64_i32(c), b.build_fconst_f64(0.5));
                        return b.build_sdiv(cf, b.build_sitofp_f64_i32(ext));
                    };
                    if (rt.dimension == 1) return {uv_comp(gx, gw)};
                    if (rt.dimension == 2) return {uv_comp(gx, gw), uv_comp(gy, gh)};
                    return {uv_comp(gx, gw), uv_comp(gy, gh), uv_comp(gz, gd)};
                }
            }
            if (rt.is_stencil()) return lower_stencil_read(e->lhs.get(), rt, e->args);
            if (rt.is_image() || rt.is_sampler()) return lower_texture_call(e);
            if (rt.is_buffer() && e->name == "len") {
                // The runtime array's words over an element's.
                ValueList base = lower_expr(e->lhs.get());
                auto found = base.empty() ? block_param.end() : block_param.find(base[0]);
                if (found == block_param.end()) return {b.build_iconst_i32(0)};
                used_block_params.insert(found->second);
                if (!is_gpu) {
                    auto it = cpu_block_ptrs.find(found->second);
                    if (it == cpu_block_ptrs.end()) return {b.build_iconst_i32(0)};
                    const uint32_t stride = std::max(4u, block_stride(rt.element_type(), false));
                    return {b.build_trunc_i32(b.build_udiv(it->second.second, b.build_iconst_i64(stride)))};
                }
                const CheckedParam& p = bound_param(found->second);
                const uint32_t words = std::max(1u, block_stride(rt.element_type(), false) / 4);
                brass::Value* n = kb.buffer_length(p.descriptor_set, p.descriptor_binding);
                return {words == 1 ? n : b.build_udiv(n, b.build_iconst_i32(static_cast<int32_t>(words)))};
            }
            if (m.rfind("atomic_", 0) == 0 && is_block_path(e->lhs.get())) {
                BlockPlace bp;
                if (!block_place(e->lhs.get(), bp)) {
                    diags.error(e->span, "this atomic access to a buffer is not a path of fields, indices and swizzles");
                    return make_zero_leaves(e->resolved_type);
                }
                std::vector<ValueList> avs;
                avs.reserve(e->args.size());
                for (const auto& a : e->args) avs.push_back(lower_expr(a.get()));
                return block_atomic(e, bp, m, avs);
            }
            if (m.rfind("atomic_", 0) == 0 && is_shared_path(e->lhs.get())) {
                SharedPlace sp;
                if (!shared_place(e->lhs.get(), sp)) {
                    diags.error(e->span, "this atomic access to shared memory is not a path of fields, indices and swizzles");
                    return make_zero_leaves(e->resolved_type);
                }
                std::vector<ValueList> avs;
                avs.reserve(e->args.size());
                for (const auto& a : e->args) avs.push_back(lower_expr(a.get()));
                return shared_atomic(e, sp, m, avs);
            }
            if (rt.is_store()) {
                ValueList base = lower_expr(e->lhs.get());
                uint32_t dim = rt.dimension;
                Type elem_t = rt.element_type();
                if (m == "read") {
                    brass::Value* lin = compute_linear_index(dim, gx, gy, gz);
                    return load_element_from_buffer(elem_t, store_access(base, dim, gx, gy, gz), lin);
                }
                if (m == "write") {
                    ValueList val = lower_expr(e->args[0].get());
                    brass::Value* lin = compute_linear_index(dim, gx, gy, gz);
                    store_element_to_buffer(elem_t, store_access(base, dim, gx, gy, gz), lin, val);
                    return {};
                }
                if (m == "load") {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    return load_element_from_buffer(elem_t, store_access(base, dim, x[0], y[0], z[0]), lin);
                }
                if (m == "store") {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList val = lower_expr(e->args[dim].get());
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    store_element_to_buffer(elem_t, store_access(base, dim, x[0], y[0], z[0]), lin, val);
                    return {};
                }
                if (m.rfind("atomic_", 0) == 0) {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList val = lower_expr(e->args[dim].get());
                    ValueList val2 = (m == "atomic_cas" && e->args.size() > dim + 1)
                                         ? lower_expr(e->args[dim + 1].get())
                                         : ValueList{};
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    base = store_access(base, dim, x[0], y[0], z[0]);
                    if (is_gpu) {
                        uint32_t esz = elem_t.scalar_size_bytes();
                        brass::Value* lin64 = (lin->type() == brass::Type::i64()) ? lin : b.build_sext_i64(lin);
                        brass::Value* byte_off = b.build_mul(lin64, b.build_iconst_i64(esz));
                        brass::Value* addr = b.build_add(base[0], byte_off);
                        auto atomic = [&]() -> brass::Value* {
                            if (m == "atomic_add") {
                                if (elem_t.is_f32()) return kb.atom_add_f32(addr, val[0]);
                                return elem_t.is_u32() ? kb.atom_add_u32(addr, val[0]) : kb.atom_add_i32(addr, val[0]);
                            }
                            if (m == "atomic_min") {
                                return elem_t.is_u32() ? kb.atom_min_u32(addr, val[0]) : kb.atom_min_i32(addr, val[0]);
                            }
                            if (m == "atomic_max") {
                                return elem_t.is_u32() ? kb.atom_max_u32(addr, val[0]) : kb.atom_max_i32(addr, val[0]);
                            }
                            if (m == "atomic_or") {
                                return elem_t.is_u32() ? kb.atom_or_u32(addr, val[0]) : kb.atom_or_i32(addr, val[0]);
                            }
                            if (m == "atomic_and") {
                                return elem_t.is_u32() ? kb.atom_and_u32(addr, val[0]) : kb.atom_and_i32(addr, val[0]);
                            }
                            if (m == "atomic_xor") {
                                return elem_t.is_u32() ? kb.atom_xor_u32(addr, val[0]) : kb.atom_xor_i32(addr, val[0]);
                            }
                            if (m == "atomic_cas") {
                                return elem_t.is_u32() ? kb.atom_cas_u32(addr, val[0], val2[0])
                                                       : kb.atom_cas_i32(addr, val[0], val2[0]);
                            }
                            if (elem_t.is_f32()) return kb.atom_exch_f32(addr, val[0]);
                            return elem_t.is_u32() ? kb.atom_exch_u32(addr, val[0]) : kb.atom_exch_i32(addr, val[0]);
                        };
                        if (base.size() < 2) return {atomic()};
                        brass::BasicBlock* atom_bb = b.side_block("bounded_atomic");
                        brass::BasicBlock* done_bb = b.side_block("bounded_atomic_done");
                        brass::Value* old = b.add_block_param(done_bb, to_mir_scalar_type(elem_t.scalar));
                        brass::Value* zero = elem_t.is_f32() ? b.build_fconst_f32(0.0f) : b.build_iconst_i32(0);
                        b.builder().build_br_if(b.build_ult(lin64, base[1]), atom_bb, {}, done_bb, {zero});
                        b.position_at_end(atom_bb);
                        b.build_br(done_bb, {atomic()});
                        b.position_at_end(done_bb);
                        return {old};
                    } else {
                        brass::Type mt = to_mir_scalar_type(elem_t.scalar);
                        brass::Value* idx64 = (lin->type() == brass::Type::i64()) ? lin : b.build_sext_i64(lin);
                        const Ops::Bound bound_v{idx64, base.size() > 1 ? base[1] : nullptr};
                        const Ops::Bound* bound = bound_v.count ? &bound_v : nullptr;
                        brass::Builder& sb = b.builder();
                        bool is_u = elem_t.is_u32();
                        brass::Value* rmw_val = (m == "atomic_cas") ? val2[0] : val[0];
                        brass::Value* cmp_val = (m == "atomic_cas") ? val[0] : nullptr;
                        if (cmp_val && b.vector_mode() && Ops::is_varying(cmp_val)) {
                            throw VectorUnsupported("varying atomic_cas");
                        }
                        brass::Value* old_v = b.build_atomic_rmw(
                            mt, base[0], idx64, static_cast<uint8_t>(mt.size_in_bytes()), rmw_val, bound,
                            [&](brass::Value* cur, brass::Value* v) -> brass::Value* {
                                if (m == "atomic_add") return sb.build_add(cur, v);
                                if (m == "atomic_min") {
                                    return sb.build_select(is_u ? sb.build_ult(cur, v) : sb.build_slt(cur, v), cur, v);
                                }
                                if (m == "atomic_max") {
                                    return sb.build_select(is_u ? sb.build_ugt(cur, v) : sb.build_sgt(cur, v), cur, v);
                                }
                                if (m == "atomic_or") return sb.build_or(cur, v);
                                if (m == "atomic_and") return sb.build_and(cur, v);
                                if (m == "atomic_xor") return sb.build_xor(cur, v);
                                if (m == "atomic_cas") return sb.build_select(sb.build_eq(cur, cmp_val), v, cur);
                                return v;
                            });
                        return {old_v};
                    }
                }
            }
            return make_zero_leaves(e->resolved_type);
        }
        case Expr::Kind::Call: {
            const std::string& n = e->name;
            // 1. User / stdlib function call (inlined)
            if (const CheckedFunction* callee = mod.find_function(n)) {
                std::vector<ValueList> arg_vals;
                std::vector<std::optional<BlockPlace>> arg_places;
                arg_vals.reserve(e->args.size());
                arg_places.reserve(e->args.size());
                for (size_t i = 0; i < e->args.size(); ++i) {
                    const Expr* a = e->args[i].get();
                    BlockPlace bp;
                    if (i < callee->params.size() &&
                        (callee->params[i].type.is_struct() || callee->params[i].type.is_array()) &&
                        is_block_path(a) && block_place(a, bp)) {
                        arg_vals.emplace_back();
                        arg_places.emplace_back(std::move(bp));
                    } else {
                        arg_vals.push_back(lower_expr(a));
                        arg_places.emplace_back(std::nullopt);
                    }
                }
                return lower_inlined_call(*callee, arg_vals, arg_places);
            }
            // 2. Internal splat helper inserted by Sema
            if (n == "__splat") {
                ValueList sv = lower_expr(e->args[0].get());
                uint32_t count = mod.types.flat_leaf_count(e->resolved_type);
                return ValueList(count, sv[0]);
            }
            // 3. Scalar casts f32(x), f64(x), i32(x), u32(x), i64(x), u64(x), bool(x)
            if (n == "f32" || n == "f64" || n == "i32" || n == "u32" || n == "i64" || n == "u64" || n == "bool") {
                ValueList sv = lower_expr(e->args[0].get());
                ScalarKind from_sk = e->args[0]->resolved_type.scalar;
                ScalarKind to_sk = e->resolved_type.scalar;
                ValueList out;
                for (brass::Value* v : sv) out.push_back(lower_scalar_cast(v, from_sk, to_sk));
                return out;
            }
            if (n == "bitcast") {
                ValueList sv = lower_expr(e->args[0].get());
                ScalarKind from_sk = e->args[0]->resolved_type.scalar;
                ScalarKind to_sk = e->resolved_type.scalar;
                ValueList out;
                for (brass::Value* v : sv) {
                    if (from_sk == ScalarKind::F32 && (to_sk == ScalarKind::I32 || to_sk == ScalarKind::U32)) {
                        out.push_back(b.build_bitcast_i32_f32(v));
                    } else if ((from_sk == ScalarKind::I32 || from_sk == ScalarKind::U32) && to_sk == ScalarKind::F32) {
                        out.push_back(b.build_bitcast_f32_i32(v));
                    } else if (from_sk == ScalarKind::F64 && (to_sk == ScalarKind::I64 || to_sk == ScalarKind::U64)) {
                        out.push_back(b.build_bitcast_i64_f64(v));
                    } else if ((from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) && to_sk == ScalarKind::F64) {
                        out.push_back(b.build_bitcast_f64_i64(v));
                    } else {
                        out.push_back(v);
                    }
                }
                return out;
            }
            // 4. Vector / Matrix / Array / Struct constructors
            if (n == "vec2" || n == "vec3" || n == "vec4" ||
                n == "vec2f" || n == "vec3f" || n == "vec4f" ||
                n == "vec2d" || n == "vec3d" || n == "vec4d" ||
                n == "vec2i" || n == "vec3i" || n == "vec4i" ||
                n == "vec2u" || n == "vec3u" || n == "vec4u" ||
                n == "mat2" || n == "mat3" || n == "mat4" ||
                n == "mat2f" || n == "mat3f" || n == "mat4f" ||
                n == "mat2d" || n == "mat3d" || n == "mat4d" ||
                n == "vec" || n == "mat" || n == "array" || e->resolved_type.is_struct()) {
                uint32_t target_leaves = mod.types.flat_leaf_count(e->resolved_type);
                std::vector<ScalarKind> target_sks = mod.types.flatten_scalars(e->resolved_type);
                ValueList flat;
                for (const auto& a : e->args) {
                    ValueList av = lower_expr(a.get());
                    std::vector<ScalarKind> arg_sks = mod.types.flatten_scalars(a->resolved_type);
                    for (size_t i = 0; i < av.size(); ++i) {
                        brass::Value* v = av[i];
                        if (!e->resolved_type.is_struct() && i < arg_sks.size() && !target_sks.empty()) {
                            ScalarKind to_sk = target_sks[std::min(flat.size(), target_sks.size() - 1)];
                            if (arg_sks[i] != to_sk) {
                                v = lower_scalar_cast(v, arg_sks[i], to_sk);
                            }
                        }
                        flat.push_back(v);
                    }
                }
                if (flat.empty()) return make_zero_leaves(e->resolved_type);
                if (flat.size() == 1 && target_leaves > 1) {
                    return ValueList(target_leaves, flat[0]);
                }
                while (flat.size() < target_leaves) flat.push_back(flat.back());
                flat.resize(target_leaves);
                return flat;
            }
            // 5. Grid / Thread / Stage built-in queries
            if (n == "grid_x") return {gx};
            if (n == "grid_y") return {gy};
            if (n == "grid_z") return {gz};
            if (n == "grid_width") return {gw};
            if (n == "grid_height") return {gh};
            if (n == "grid_depth") return {gd};
            if (n == "grid_linear_index") {
                return {compute_linear_index(std::max(1u, root_fn.grid_dim), gx, gy, gz)};
            }
            if (n == "grid_total_cells") {
                uint32_t d = std::max(1u, root_fn.grid_dim);
                brass::Value* total = b.build_sext_i64(gw);
                if (d >= 2) total = b.build_mul(total, b.build_sext_i64(gh));
                if (d >= 3) total = b.build_mul(total, b.build_sext_i64(gd));
                return {total};
            }
            if (n == "grid_coord" || n == "grid_pos") {
                if (e->resolved_type.is_scalar()) return {gx};
                if (e->resolved_type.dimension == 2) return {gx, gy};
                return {gx, gy, gz};
            }
            if (n == "grid_uv") {
                auto uv_comp = [&](brass::Value* c, brass::Value* ext) {
                    brass::Value* cf = b.build_add(b.build_sitofp_f32_i32(c), b.build_fconst_f32(0.5f));
                    return b.build_sdiv(cf, b.build_sitofp_f32_i32(ext));
                };
                if (e->resolved_type.is_scalar()) return {uv_comp(gx, gw)};
                if (e->resolved_type.dimension == 2) return {uv_comp(gx, gw), uv_comp(gy, gh)};
                return {uv_comp(gx, gw), uv_comp(gy, gh), uv_comp(gz, gd)};
            }
            if (n == "grid_uv_f64") {
                auto uv_comp = [&](brass::Value* c, brass::Value* ext) {
                    brass::Value* cf = b.build_add(b.build_sitofp_f64_i32(c), b.build_fconst_f64(0.5));
                    return b.build_sdiv(cf, b.build_sitofp_f64_i32(ext));
                };
                if (e->resolved_type.is_scalar()) return {uv_comp(gx, gw)};
                if (e->resolved_type.dimension == 2) return {uv_comp(gx, gw), uv_comp(gy, gh)};
                return {uv_comp(gx, gw), uv_comp(gy, gh), uv_comp(gz, gd)};
            }
            auto get_lid_x = [&] { return is_gpu ? kb.tid_x() : (lid_x ? lid_x : b.build_iconst_i32(0)); };
            auto get_lid_y = [&] { return is_gpu ? kb.tid_y() : (lid_y ? lid_y : b.build_iconst_i32(0)); };
            auto get_lid_z = [&] { return is_gpu ? kb.tid_z() : (lid_z ? lid_z : b.build_iconst_i32(0)); };
            auto get_wgid_x = [&] { return is_gpu ? kb.ctaid_x() : (wgid_x ? wgid_x : b.build_iconst_i32(0)); };
            auto get_wgid_y = [&] { return is_gpu ? kb.ctaid_y() : (wgid_y ? wgid_y : b.build_iconst_i32(0)); };
            auto get_wgid_z = [&] { return is_gpu ? kb.ctaid_z() : (wgid_z ? wgid_z : b.build_iconst_i32(0)); };
            auto get_nwg_x = [&] { return is_gpu ? kb.nctaid_x() : (nwg_x ? nwg_x : b.build_iconst_i32(1)); };
            auto get_nwg_y = [&] { return is_gpu ? kb.nctaid_y() : (nwg_y ? nwg_y : b.build_iconst_i32(1)); };
            auto get_nwg_z = [&] { return is_gpu ? kb.nctaid_z() : (nwg_z ? nwg_z : b.build_iconst_i32(1)); };
            auto get_lsz_x = [&] { return is_gpu ? kb.ntid_x() : (lsz_x ? lsz_x : b.build_iconst_i32(1)); };
            auto get_lsz_y = [&] { return is_gpu ? kb.ntid_y() : (lsz_y ? lsz_y : b.build_iconst_i32(1)); };
            auto get_lsz_z = [&] { return is_gpu ? kb.ntid_z() : (lsz_z ? lsz_z : b.build_iconst_i32(1)); };
            auto get_gid_x = [&] { return gx ? gx : (is_gpu ? kb.global_tid_x() : b.build_iconst_i32(0)); };
            auto get_gid_y = [&] {
                if (is_gpu && root_fn.grid_dim < 2) return b.build_add(b.build_mul(kb.ctaid_y(), kb.ntid_y()), kb.tid_y());
                return gy ? gy : b.build_iconst_i32(0);
            };
            auto get_gid_z = [&] {
                if (is_gpu && root_fn.grid_dim < 3) return b.build_add(b.build_mul(kb.ctaid_z(), kb.ntid_z()), kb.tid_z());
                return gz ? gz : b.build_iconst_i32(0);
            };
            if (n == "thread_id_x" || n == "local_id_x") return {get_lid_x()};
            if (n == "thread_id_y" || n == "local_id_y") return {get_lid_y()};
            if (n == "thread_id_z" || n == "local_id_z") return {get_lid_z()};
            if (n == "local_id" || n == "thread_id") return {get_lid_x(), get_lid_y(), get_lid_z()};
            if (n == "group_id_x" || n == "workgroup_id_x") return {get_wgid_x()};
            if (n == "group_id_y" || n == "workgroup_id_y") return {get_wgid_y()};
            if (n == "group_id_z" || n == "workgroup_id_z") return {get_wgid_z()};
            if (n == "workgroup_id" || n == "group_id") return {get_wgid_x(), get_wgid_y(), get_wgid_z()};
            if (n == "num_workgroups_x") return {get_nwg_x()};
            if (n == "num_workgroups_y") return {get_nwg_y()};
            if (n == "num_workgroups_z") return {get_nwg_z()};
            if (n == "num_workgroups") return {get_nwg_x(), get_nwg_y(), get_nwg_z()};
            if (n == "global_id_x") return {get_gid_x()};
            if (n == "global_id_y") return {get_gid_y()};
            if (n == "global_id_z") return {get_gid_z()};
            if (n == "global_id") return {get_gid_x(), get_gid_y(), get_gid_z()};
            if (n == "workgroup_size_x") return {get_lsz_x()};
            if (n == "workgroup_size_y") return {get_lsz_y()};
            if (n == "workgroup_size_z") return {get_lsz_z()};
            if (n == "workgroup_size") return {get_lsz_x(), get_lsz_y(), get_lsz_z()};
            if (n == "local_invocation_index" || n == "local_index") {
                brass::Value* sx = get_lsz_x();
                brass::Value* sy = get_lsz_y();
                brass::Value* xy = b.build_add(get_lid_x(), b.build_mul(get_lid_y(), sx));
                return {b.build_add(xy, b.build_mul(get_lid_z(), b.build_mul(sx, sy)))};
            }
            if (n == "subgroup_size") return {is_gpu ? b.build_call("spirv_subgroup_size", brass::Type::i32()) : b.build_iconst_i32(1)};
            if (n == "subgroup_invocation_id") return {is_gpu ? b.build_call("spirv_subgroup_local_id", brass::Type::i32()) : b.build_iconst_i32(0)};
            if (n == "workgroup_barrier" || n == "storage_barrier" || n == "barrier") {
                if (is_gpu) {
                    kb.sync();
                } else if (cpu_barrier_fn && cpu_barrier_ctx) {
                    b.builder().build_call_indirect(cpu_barrier_fn, brass::Type::void_type(), {cpu_barrier_ctx});
                }
                return {};
            }
            // Vertex & Fragment intrinsics
            if (n == "vertex_index") return {is_gpu ? b.build_call("spirv_vertex_index", brass::Type::i32()) : gx};
            if (n == "instance_index") return {is_gpu ? b.build_call("spirv_instance_index", brass::Type::i32()) : gy};
            if (n == "set_vertex_position") {
                ValueList v = lower_expr(e->args[0].get());
                if (is_gpu && v.size() >= 4) {
                    b.build_call("spirv_set_vertex_position", brass::Type::void_type(), {v[0], v[1], v[2], v[3]});
                }
                return {};
            }
            if (n == "set_vertex_output" || n == "set_frag_output") {
                ValueList loc = lower_expr(e->args[0].get());
                ValueList v = lower_expr(e->args[1].get());
                const brass::Instruction* def = loc[0]->is_instruction() ? loc[0]->defining_instruction() : nullptr;
                if (!def || def->opcode() != brass::Opcode::iconst_i32) {
                    diags.error(e->args[0]->span, "a stage output's location must be a constant");
                    return {};
                }
                if (is_gpu) kb.stage_output(static_cast<uint32_t>(def->imm_i32()), v);
                return {};
            }
            if (n == "set_frag_color") {
                ValueList v = lower_expr(e->args[0].get());
                if (is_gpu) kb.stage_output(0, v);
                return {};
            }
            if (n == "set_frag_depth") {
                ValueList v = lower_expr(e->args[0].get());
                if (is_gpu) kb.set_frag_depth(v[0]);
                return {};
            }
            if (n == "frag_coord") {
                if (is_gpu) {
                    return {
                        b.build_call("spirv_frag_coord_x", brass::Type::f32()),
                        b.build_call("spirv_frag_coord_y", brass::Type::f32()),
                        b.build_call("spirv_frag_coord_z", brass::Type::f32()),
                        b.build_call("spirv_frag_coord_w", brass::Type::f32()),
                    };
                }
                return {b.build_sitofp_f32_i32(gx), b.build_sitofp_f32_i32(gy), b.build_fconst_f32(0.0f), b.build_fconst_f32(1.0f)};
            }
            if (n == "front_facing") {
                return {is_gpu ? b.build_call("spirv_front_facing", brass::Type::i32()) : b.build_iconst_i32(1)};
            }
            if (n == "discard_fragment") {
                // Demote drops the fragment's outputs and keeps the invocation running in uniform control flow.
                if (is_gpu) b.build_call("spirv_demote_to_helper", brass::Type::void_type());
                return {};
            }
            // 6. Vector reductions: dot, length, distance, cross, normalize, reflect
            if (n == "dot") {
                ValueList a = lower_expr(e->args[0].get());
                ValueList c = lower_expr(e->args[1].get());
                brass::Value* sum = b.build_mul(a[0], c[0]);
                for (size_t i = 1; i < a.size() && i < c.size(); ++i) {
                    sum = b.build_add(sum, b.build_mul(a[i], c[i]));
                }
                return {sum};
            }
            if (n == "length") {
                ValueList a = lower_expr(e->args[0].get());
                if (a.size() == 1) return {lower_scalar_math_1("abs", a[0], e->resolved_type.scalar)};
                brass::Value* sum = b.build_mul(a[0], a[0]);
                for (size_t i = 1; i < a.size(); ++i) {
                    sum = b.build_add(sum, b.build_mul(a[i], a[i]));
                }
                return {b.build_sqrt(sum)};
            }
            if (n == "distance") {
                ValueList a = lower_expr(e->args[0].get());
                ValueList c = lower_expr(e->args[1].get());
                brass::Value* d0 = b.build_sub(a[0], c[0]);
                if (a.size() == 1) return {lower_scalar_math_1("abs", d0, e->resolved_type.scalar)};
                brass::Value* sum = b.build_mul(d0, d0);
                for (size_t i = 1; i < a.size() && i < c.size(); ++i) {
                    brass::Value* di = b.build_sub(a[i], c[i]);
                    sum = b.build_add(sum, b.build_mul(di, di));
                }
                return {b.build_sqrt(sum)};
            }
            if (n == "normalize") {
                ValueList a = lower_expr(e->args[0].get());
                brass::Value* sum = b.build_mul(a[0], a[0]);
                for (size_t i = 1; i < a.size(); ++i) {
                    sum = b.build_add(sum, b.build_mul(a[i], a[i]));
                }
                brass::Value* len = b.build_sqrt(sum);
                ValueList out;
                out.reserve(a.size());
                for (brass::Value* v : a) out.push_back(b.build_sdiv(v, len));
                return out;
            }
            if (n == "cross") {
                ValueList a = lower_expr(e->args[0].get());
                ValueList c = lower_expr(e->args[1].get());
                return {
                    b.build_sub(b.build_mul(a[1], c[2]), b.build_mul(a[2], c[1])),
                    b.build_sub(b.build_mul(a[2], c[0]), b.build_mul(a[0], c[2])),
                    b.build_sub(b.build_mul(a[0], c[1]), b.build_mul(a[1], c[0])),
                };
            }
            if (n == "reflect") {
                ValueList iv = lower_expr(e->args[0].get());
                ValueList nv = lower_expr(e->args[1].get());
                brass::Value* dt = b.build_mul(iv[0], nv[0]);
                for (size_t i = 1; i < iv.size(); ++i) dt = b.build_add(dt, b.build_mul(iv[i], nv[i]));
                brass::Value* two_dt = b.build_add(dt, dt);
                ValueList out;
                for (size_t i = 0; i < iv.size(); ++i) {
                    out.push_back(b.build_sub(iv[i], b.build_mul(two_dt, nv[i])));
                }
                return out;
            }
            if (n == "faceforward") {
                // N if dot(Nref, I) < 0, else -N.
                ValueList nv = lower_expr(e->args[0].get());
                ValueList iv = lower_expr(e->args[1].get());
                ValueList rv = lower_expr(e->args[2].get());
                bool d = e->resolved_type.scalar == ScalarKind::F64;
                brass::Value* zero = d ? b.build_fconst_f64(0.0) : b.build_fconst_f32(0.0f);
                brass::Value* minus = d ? b.build_fconst_f64(-1.0) : b.build_fconst_f32(-1.0f);
                brass::Value* dt = b.build_mul(rv[0], iv[0]);
                for (size_t i = 1; i < rv.size(); ++i) dt = b.build_add(dt, b.build_mul(rv[i], iv[i]));
                brass::Value* front = b.build_slt(dt, zero);
                ValueList out;
                for (brass::Value* v : nv) out.push_back(b.build_select(front, v, b.build_mul(v, minus)));
                return out;
            }
            if (n == "refract") {
                // k = 1 - eta^2 (1 - dot(N, I)^2); 0 if k < 0, else eta I - (eta dot(N, I) + sqrt(k)) N.
                ValueList iv = lower_expr(e->args[0].get());
                ValueList nv = lower_expr(e->args[1].get());
                brass::Value* eta = lower_expr(e->args[2].get())[0];
                bool d = e->resolved_type.scalar == ScalarKind::F64;
                brass::Value* zero = d ? b.build_fconst_f64(0.0) : b.build_fconst_f32(0.0f);
                brass::Value* one = d ? b.build_fconst_f64(1.0) : b.build_fconst_f32(1.0f);
                brass::Value* dt = b.build_mul(nv[0], iv[0]);
                for (size_t i = 1; i < nv.size(); ++i) dt = b.build_add(dt, b.build_mul(nv[i], iv[i]));
                brass::Value* k = b.build_sub(one, b.build_mul(b.build_mul(eta, eta), b.build_sub(one, b.build_mul(dt, dt))));
                brass::Value* tir = b.build_slt(k, zero);
                brass::Value* f = b.build_add(b.build_mul(eta, dt), b.build_sqrt(b.build_fmax(k, zero)));
                ValueList out;
                for (size_t i = 0; i < iv.size(); ++i) {
                    brass::Value* r = b.build_sub(b.build_mul(eta, iv[i]), b.build_mul(f, nv[i]));
                    out.push_back(b.build_select(tir, zero, r));
                }
                return out;
            }
            if (n == "select") {
                ValueList cv = lower_expr(e->args[0].get());
                if (const brass::Instruction* def = cv[0]->is_instruction() ? cv[0]->defining_instruction() : nullptr;
                    def && def->opcode() == brass::Opcode::iconst_i32) {
                    return lower_expr(def->imm_i32() != 0 ? e->args[1].get() : e->args[2].get());
                }
                ValueList av = lower_expr(e->args[1].get());
                ValueList bv = lower_expr(e->args[2].get());
                ValueList out;
                for (size_t i = 0; i < av.size() && i < bv.size(); ++i) {
                    out.push_back(b.build_select(cv[0], av[i], bv[i]));
                }
                return out;
            }
            // 7. Component-wise 1-, 2-, 3-arg math functions
            ScalarKind sk = e->resolved_type.scalar;
            if (e->args.size() == 1) {
                ValueList a = lower_expr(e->args[0].get());
                ValueList out;
                // The argument's kind: isnan / isinf / subgroup votes return bool.
                ScalarKind ask = e->args[0]->resolved_type.scalar;
                for (brass::Value* v : a) out.push_back(lower_scalar_math_1(n, v, ask));
                return out;
            }
            if (e->args.size() == 2) {
                ValueList a = lower_expr(e->args[0].get());
                ValueList c = lower_expr(e->args[1].get());
                ValueList out;
                for (size_t i = 0; i < a.size() && i < c.size(); ++i) {
                    out.push_back(lower_scalar_math_2(n, a[i], c[i], sk));
                }
                return out;
            }
            if (e->args.size() == 3) {
                ValueList x = lower_expr(e->args[0].get());
                ValueList y = lower_expr(e->args[1].get());
                ValueList z = lower_expr(e->args[2].get());
                ValueList out;
                for (size_t i = 0; i < x.size() && i < y.size() && i < z.size(); ++i) {
                    out.push_back(lower_scalar_math_3(n, x[i], y[i], z[i], sk));
                }
                return out;
            }
            return make_zero_leaves(e->resolved_type);
        }
    }
    return {};
}
} // namespace brokernel::detail

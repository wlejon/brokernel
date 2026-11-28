// Structured store sub-element places: `st[i].field`, `st[i].arr[k] = val`,
// `st[i].pos.xy += v`. Resolves a Field/Index chain rooted at `store[coords...]`
// to a byte offset within the element so only the targeted scalar leaves are
// loaded or stored, with full store bounds checking on both CPU and GPU.

#include "lowering_internal.h"
#include <algorithm>

namespace brokernel::detail {

namespace {

brass::Value* add_i32_bytes(Ops& b, brass::Value* base, uint32_t bytes) {
    if (bytes == 0) return base;
    const brass::Instruction* def = base && base->is_instruction() ? base->defining_instruction() : nullptr;
    if (def && def->opcode() == brass::Opcode::iconst_i32) {
        return b.build_iconst_i32(def->imm_i32() + static_cast<int32_t>(bytes));
    }
    return b.build_add(base, b.build_iconst_i32(static_cast<int32_t>(bytes)));
}

brass::Value* compute_byte_base_i32(Ops& b, const MirBodyLowerer::StorePlace& p) {
    brass::Value* lin32 = (p.linear_idx->type() == brass::Type::i32()) ? p.linear_idx : b.build_trunc_i32(p.linear_idx);
    brass::Value* rel32 = (p.rel_byte_off->type() == brass::Type::i32()) ? p.rel_byte_off : b.build_trunc_i32(p.rel_byte_off);
    return b.build_add(b.build_mul(lin32, b.build_iconst_i32(static_cast<int32_t>(p.elem_bytes))), rel32);
}

brass::Value* compute_byte_base_i64(Ops& b, const MirBodyLowerer::StorePlace& p) {
    brass::Value* lin64 = (p.linear_idx->type() == brass::Type::i64()) ? p.linear_idx : b.build_sext_i64(p.linear_idx);
    brass::Value* rel64 = (p.rel_byte_off->type() == brass::Type::i64()) ? p.rel_byte_off : b.build_sext_i64(p.rel_byte_off);
    return b.build_add(b.build_mul(lin64, b.build_iconst_i64(static_cast<int64_t>(p.elem_bytes))), rel64);
}

void collect_store_leaf_offsets(
    const TypeRegistry& types,
    const Type& t,
    uint32_t cur_byte_off,
    std::vector<uint32_t>& offsets,
    std::vector<ScalarKind>& kinds) {
    if (t.is_scalar()) {
        offsets.push_back(cur_byte_off);
        kinds.push_back(t.scalar);
        if (t.scalar == ScalarKind::DF32) {
            offsets.push_back(cur_byte_off + 4);
            kinds.push_back(ScalarKind::DF32);
        }
    } else if (t.is_vector()) {
        uint32_t sb = t.scalar_size_bytes();
        for (uint32_t i = 0; i < t.dimension; ++i) {
            offsets.push_back(cur_byte_off + i * sb);
            kinds.push_back(t.scalar);
            if (t.scalar == ScalarKind::DF32) {
                offsets.push_back(cur_byte_off + i * sb + 4);
                kinds.push_back(ScalarKind::DF32);
            }
        }
    } else if (t.is_array()) {
        Type elem = t.element_type();
        uint32_t stride = types.byte_size(elem);
        for (uint32_t i = 0; i < t.dimension; ++i) {
            collect_store_leaf_offsets(types, elem, cur_byte_off + i * stride, offsets, kinds);
        }
    } else if (t.is_struct()) {
        if (const StructDef* sdef = types.find_struct(t.struct_id)) {
            for (const auto& f : sdef->fields) {
                collect_store_leaf_offsets(types, f.type, cur_byte_off + f.byte_offset, offsets, kinds);
            }
        }
    }
}

} // namespace

bool MirBodyLowerer::is_store_subpath(const Expr* e) const {
    if (!e || (e->kind != Expr::Kind::Field && e->kind != Expr::Kind::Index)) return false;
    // Direct `st[coords...]` uses the fast whole-element path (contiguous AVX2).
    if (e->kind == Expr::Kind::Index && e->lhs && e->lhs->resolved_type.is_store()) return false;
    const Expr* cur = e->lhs.get();
    while (cur && (cur->kind == Expr::Kind::Field || cur->kind == Expr::Kind::Index)) {
        if (cur->kind == Expr::Kind::Index && cur->lhs && cur->lhs->resolved_type.is_store()) {
            return true;
        }
        cur = cur->lhs.get();
    }
    return false;
}

bool MirBodyLowerer::store_place(const Expr* e, StorePlace& p) {
    if (!e) return false;
    if (e->kind == Expr::Kind::Index && e->lhs && e->lhs->resolved_type.is_store()) {
        const Type& st_t = e->lhs->resolved_type;
        Type elem_t = st_t.element_type();
        ValueList base = lower_expr(e->lhs.get());
        uint32_t dim = st_t.dimension;
        ValueList x = (dim >= 1 && e->args.size() >= 1) ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
        ValueList y = (dim >= 2 && e->args.size() >= 2) ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
        ValueList z = (dim >= 3 && e->args.size() >= 3) ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
        brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
        p.store = store_access(base, dim, x[0], y[0], z[0]);
        p.linear_idx = (lin->type() == brass::Type::i64()) ? lin : b.build_sext_i64(lin);
        p.elem_bytes = mod.types.byte_size(elem_t);
        p.rel_byte_off = b.build_iconst_i32(0);
        p.type = elem_t;
        p.lanes.clear();
        return true;
    }
    if (e->kind == Expr::Kind::Field) {
        if (!store_place(e->lhs.get(), p) || !p.lanes.empty()) return false;
        if (p.type.is_newtype()) {
            if (e->name != "0") return false;
            p.type = mod.types.unwrap_newtype(p.type);
            return true;
        }
        if (p.type.is_scalar()) return true;
        if (p.type.is_struct()) {
            const StructDef* sdef = mod.types.find_struct(p.type.struct_id);
            const StructField* sf = sdef ? sdef->find_field(e->name) : nullptr;
            if (!sf) return false;
            p.rel_byte_off = add_i32_bytes(b, p.rel_byte_off, sf->byte_offset);
            p.type = sf->type;
            return true;
        }
        if (p.type.is_vector()) {
            uint32_t sb = p.type.scalar_size_bytes();
            if (e->name.size() == 1) {
                uint32_t lane = swizzle_index(e->name[0]);
                p.rel_byte_off = add_i32_bytes(b, p.rel_byte_off, lane * sb);
                p.type = Type::scalar_type(p.type.scalar);
            } else {
                for (char c : e->name) p.lanes.push_back(swizzle_index(c));
                p.type = Type::vec(p.type.scalar, static_cast<uint32_t>(p.lanes.size()));
            }
            return true;
        }
        return false;
    }
    if (e->kind == Expr::Kind::Index) {
        if (!store_place(e->lhs.get(), p) || !p.lanes.empty() || e->args.size() != 1) return false;
        if (!p.type.is_array() && !p.type.is_vector()) return false;
        Type elem = p.type.element_type();
        uint32_t stride = mod.types.byte_size(elem);
        brass::Value* idx = lower_expr(e->args[0].get())[0];
        if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
        idx = clamp_coord(idx, b.build_iconst_i32(static_cast<int32_t>(p.type.dimension)));
        p.rel_byte_off = b.build_add(p.rel_byte_off, b.build_mul(idx, b.build_iconst_i32(static_cast<int32_t>(stride))));
        p.type = elem;
        return true;
    }
    return false;
}

ValueList MirBodyLowerer::store_place_load(const StorePlace& p) {
    brass::Value* base_ptr = p.store[0];
    const Ops::Bound bound_v{p.linear_idx, p.store.size() > 1 ? p.store[1] : nullptr};
    const Ops::Bound* bound = bound_v.count ? &bound_v : nullptr;
    const bool vec_i32 = b.vector_mode();
    brass::Value* elem_byte_base = vec_i32 ? compute_byte_base_i32(b, p) : compute_byte_base_i64(b, p);

    std::vector<uint32_t> offs;
    std::vector<ScalarKind> kinds;
    if (!p.lanes.empty()) {
        uint32_t sb = p.type.scalar_size_bytes();
        for (uint32_t lane : p.lanes) {
            offs.push_back(lane * sb);
            kinds.push_back(p.type.scalar);
            if (p.type.scalar == ScalarKind::DF32) {
                offs.push_back(lane * sb + 4);
                kinds.push_back(ScalarKind::DF32);
            }
        }
    } else {
        collect_store_leaf_offsets(mod.types, p.type, 0, offs, kinds);
    }

    ValueList out;
    out.reserve(offs.size());
    for (size_t i = 0; i < offs.size(); ++i) {
        brass::Type mt = to_mir_scalar_type(kinds[i]);
        brass::Value* byte_idx = nullptr;
        if (vec_i32) {
            byte_idx = b.build_sext_i64(add_i32_bytes(b, elem_byte_base, offs[i]));
        } else {
            byte_idx = offs[i] == 0 ? elem_byte_base
                                    : b.build_add(elem_byte_base, b.build_iconst_i64(static_cast<int64_t>(offs[i])));
        }
        brass::Value* v = b.build_load_indexed(mt, base_ptr, byte_idx, 1, 0, bound);
        if (kinds[i] == ScalarKind::Bool && b.vector_mode()) v = b.build_ne(v, b.build_iconst_i32(0));
        out.push_back(v);
    }
    return out;
}

void MirBodyLowerer::store_place_store(const StorePlace& p, const ValueList& value) {
    brass::Value* base_ptr = p.store[0];
    const Ops::Bound bound_v{p.linear_idx, p.store.size() > 1 ? p.store[1] : nullptr};
    const Ops::Bound* bound = bound_v.count ? &bound_v : nullptr;
    const bool vec_i32 = b.vector_mode();
    brass::Value* elem_byte_base = vec_i32 ? compute_byte_base_i32(b, p) : compute_byte_base_i64(b, p);

    std::vector<uint32_t> offs;
    std::vector<ScalarKind> kinds;
    if (!p.lanes.empty()) {
        uint32_t sb = p.type.scalar_size_bytes();
        for (uint32_t lane : p.lanes) {
            offs.push_back(lane * sb);
            kinds.push_back(p.type.scalar);
            if (p.type.scalar == ScalarKind::DF32) {
                offs.push_back(lane * sb + 4);
                kinds.push_back(ScalarKind::DF32);
            }
        }
    } else {
        collect_store_leaf_offsets(mod.types, p.type, 0, offs, kinds);
    }

    for (size_t i = 0; i < offs.size() && i < value.size(); ++i) {
        brass::Type mt = to_mir_scalar_type(kinds[i]);
        brass::Value* byte_idx = nullptr;
        if (vec_i32) {
            byte_idx = b.build_sext_i64(add_i32_bytes(b, elem_byte_base, offs[i]));
        } else {
            byte_idx = offs[i] == 0 ? elem_byte_base
                                    : b.build_add(elem_byte_base, b.build_iconst_i64(static_cast<int64_t>(offs[i])));
        }
        brass::Value* val = (kinds[i] == ScalarKind::Bool && b.vector_mode())
                                ? b.build_and(value[i], b.build_iconst_i32(1))
                                : value[i];
        b.build_store_indexed(mt, base_ptr, byte_idx, 1, 0, val, bound);
    }
    if (mod.trap_non_finite && !p.store.empty()) {
        if (auto it = store_param.find(p.store[0]); it != store_param.end() && it->second < root_fn.params.size()) {
            emit_trap_non_finite(root_fn.params[it->second].name, kinds, value, bound);
        }
    }
}

std::optional<ValueList> MirBodyLowerer::try_store_place_read(const Expr* e) {
    if (!is_store_subpath(e)) return std::nullopt;
    StorePlace p;
    if (!store_place(e, p)) return std::nullopt;
    return store_place_load(p);
}

bool MirBodyLowerer::try_store_place_assign(const Stmt* s, ValueList rhs) {
    const Expr* target = s->assign_target.get();
    if (!is_store_subpath(target)) return false;
    StorePlace p;
    if (!store_place(target, p)) return false;
    if (s->compound_op.has_value()) {
        ValueList old_v = store_place_load(p);
        for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
            rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], target->resolved_type.scalar);
        }
    }
    store_place_store(p, rhs);
    return true;
}

} // namespace brokernel::detail

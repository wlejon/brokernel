// Image and sampler methods, lowered to brass's texture calls on the
// parameter's descriptor (set, binding). A texture parameter's value is a
// placeholder the lowerer maps back to the parameter, so a texture passed to
// a helper function, inlined, still names its descriptor. Texels come back
// as four lanes (f32, or i32 for a u32 texture) of which the texel type takes
// its first; a size as four i32 lanes.

#include "lowering_internal.h"
#include <algorithm>
#include <cstring>

namespace brokernel::detail {

namespace {

brass::Value* add_words(Ops& b, brass::Value* base, uint32_t words) {
    if (words == 0) return base;
    const brass::Instruction* def = base && base->is_instruction() ? base->defining_instruction() : nullptr;
    if (def && def->opcode() == brass::Opcode::iconst_i32) {
        return b.build_iconst_i32(def->imm_i32() + static_cast<int32_t>(words));
    }
    return b.build_add(base, b.build_iconst_i32(static_cast<int32_t>(words)));
}

brass::codegen::KernelBuilder::ImageRef image_ref(const CheckedParam& p) {
    brass::codegen::KernelBuilder::ImageRef r;
    r.set = p.descriptor_set;
    r.binding = p.descriptor_binding;
    const Type& t = p.type;
    if (t.image_cube()) r.dim = brass::target::SpirvImageDim::Cube;
    else if (t.dimension == 1) r.dim = brass::target::SpirvImageDim::D1;
    else if (t.dimension == 3) r.dim = brass::target::SpirvImageDim::D3;
    else r.dim = brass::target::SpirvImageDim::D2;
    r.arrayed = t.image_arrayed();
    r.depth = t.image_shadow();
    r.scalar = t.scalar == ScalarKind::U32 ? brass::target::SpirvImageScalar::U32 : brass::target::SpirvImageScalar::F32;
    r.format = t.is_image() ? p.image_format : 0;
    if (t.image_separate()) r.sampler_set = r.sampler_binding = -2;   // read with no sampler, until given one
    return r;
}

} // namespace

ValueList MirBodyLowerer::lower_texture_call(const Expr* e) {
    brass::Builder& mb = kb.builder();
    const Type& rt = e->lhs->resolved_type;
    const std::string& m = e->name;
    ValueList recv = lower_expr(e->lhs.get());
    auto found = recv.empty() ? texture_param.end() : texture_param.find(recv[0]);
    if (found == texture_param.end()) {
        diags.error(e->span, "a texture must be a kernel parameter (with @binding) to be used");
        return make_zero_leaves(e->resolved_type);
    }
    const CheckedParam& param = bound_param(found->second);
    auto img = image_ref(param);
    // A separate texture's filtering methods name their sampler first.
    size_t first = 0;
    if (rt.image_separate() && m != "size" && m != "levels" && m != "fetch") {
        ValueList s = lower_expr(e->args[0].get());
        auto sp = s.empty() ? texture_param.end() : texture_param.find(s[0]);
        if (sp == texture_param.end()) {
            diags.error(e->args[0]->span, "a sampler must be a kernel parameter (with @binding) to be used");
            return make_zero_leaves(e->resolved_type);
        }
        const CheckedParam& sparam = bound_param(sp->second);
        img.sampler_set = static_cast<int32_t>(sparam.descriptor_set);
        img.sampler_binding = static_cast<int32_t>(sparam.descriptor_binding);
        first = 1;
    }

    auto arg = [&](size_t i) { return lower_expr(e->args[first + i].get()); };
    auto scalar = [&](size_t i) { return arg(i)[0]; };
    // The first `lanes` lanes of a four-lane result.
    auto lanes_of = [&](brass::Value* v, uint32_t lanes) {
        ValueList out;
        for (uint32_t i = 0; i < lanes; ++i) out.push_back(mb.build_vextract_lane(v, i));
        return out;
    };
    auto texel = [&](brass::Value* v) { return lanes_of(v, rt.element_type().is_vector() ? rt.elem_lanes : 1); };

    if (m == "size") {
        brass::Value* s = rt.is_image() ? kb.image_size(img) : kb.texture_size(img, scalar(0));
        const Type& t = e->resolved_type;
        return lanes_of(s, t.is_vector() ? t.dimension : 1);
    }
    if (m == "levels") return {kb.texture_levels(img)};
    if (m == "load") return texel(kb.image_load(img, arg(0)));
    if (m == "store") {
        ValueList v = arg(1);
        brass::Value* zero = rt.scalar == ScalarKind::U32 ? b.build_iconst_i32(0) : b.build_fconst_f32(0.0f);
        brass::Value* vec = mb.build_vbroadcast(rt.scalar == ScalarKind::U32 ? brass::Type::i32x4() : brass::Type::f32x4(), zero);
        for (uint32_t i = 0; i < v.size() && i < 4; ++i) vec = mb.build_vinsert_lane(vec, v[i], i);
        kb.image_store(img, arg(0), vec);
        return {};
    }
    if (m == "sample") return texel(kb.texture_sample(img, arg(0)));
    if (m == "sample_offset") return texel(kb.texture_sample(img, arg(0), arg(1)));
    if (m == "sample_lod") return texel(kb.texture_sample_lod(img, arg(0), scalar(1)));
    if (m == "sample_lod_offset") return texel(kb.texture_sample_lod(img, arg(0), scalar(1), arg(2)));
    if (m == "sample_grad") return texel(kb.texture_sample_grad(img, arg(0), arg(1), arg(2)));
    if (m == "sample_grad_offset") return texel(kb.texture_sample_grad(img, arg(0), arg(1), arg(2), arg(3)));
    if (m == "fetch") return texel(kb.texture_fetch(img, arg(0), scalar(1)));
    if (m == "gather") return lanes_of(kb.texture_gather(img, arg(0), scalar(1)), 4);
    if (m == "compare") return {kb.texture_compare(img, arg(0), scalar(1))};
    if (m == "compare_lod") return {kb.texture_compare(img, arg(0), scalar(1), scalar(2))};
    diags.error(e->span, "type '" + rt.to_string() + "' has no method '" + m + "'");
    return make_zero_leaves(e->resolved_type);
}

// ---------------------------------------------------------------------------
// Function-local and constant array places (`alloca_` + indexed load/store)
// ---------------------------------------------------------------------------

namespace {

void scan_expr_dyn_arrays(const Expr* e, std::set<std::string>& out) {
    if (!e) return;
    if (e->kind == Expr::Kind::Index && e->lhs && e->lhs->resolved_type.is_array() &&
        !e->args.empty() && e->args[0]->kind != Expr::Kind::IntLit) {
        const Expr* root = e->lhs.get();
        while (root && (root->kind == Expr::Kind::Field || root->kind == Expr::Kind::Index)) {
            root = root->lhs.get();
        }
        if (root && root->kind == Expr::Kind::Ident) out.insert(root->name);
    }
    scan_expr_dyn_arrays(e->lhs.get(), out);
    scan_expr_dyn_arrays(e->rhs.get(), out);
    scan_expr_dyn_arrays(e->cond.get(), out);
    for (const auto& a : e->args) scan_expr_dyn_arrays(a.get(), out);
    for (const auto& f : e->fields) scan_expr_dyn_arrays(f.value.get(), out);
}

void scan_stmt_dyn_arrays(const Stmt* s, std::set<std::string>& out) {
    if (!s) return;
    scan_expr_dyn_arrays(s->init_expr.get(), out);
    scan_expr_dyn_arrays(s->assign_target.get(), out);
    scan_expr_dyn_arrays(s->assign_value.get(), out);
    scan_expr_dyn_arrays(s->cond.get(), out);
    scan_expr_dyn_arrays(s->range_start.get(), out);
    scan_expr_dyn_arrays(s->range_end.get(), out);
    scan_expr_dyn_arrays(s->expr.get(), out);
    for (const auto& c : s->body) scan_stmt_dyn_arrays(c.get(), out);
    for (const auto& c : s->else_body) scan_stmt_dyn_arrays(c.get(), out);
    scan_stmt_dyn_arrays(s->for_init.get(), out);
    scan_stmt_dyn_arrays(s->for_step.get(), out);
}

bool is_const_leaf(const brass::Value* v, uint64_t* bits_out = nullptr) {
    if (!v || !v->is_instruction() || !v->defining_instruction()) return false;
    const brass::Instruction* def = v->defining_instruction();
    if (def->opcode() == brass::Opcode::iconst_i32) {
        if (bits_out) *bits_out = static_cast<uint32_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == brass::Opcode::iconst_i64) {
        if (bits_out) *bits_out = static_cast<uint64_t>(def->imm_i64());
        return true;
    }
    if (def->opcode() == brass::Opcode::fconst_f64) {
        if (bits_out) {
            double d = def->imm_f64();
            uint64_t u = 0;
            std::memcpy(&u, &d, 8);
            *bits_out = u;
        }
        return true;
    }
    return false;
}

} // namespace

void MirBodyLowerer::collect_dyn_indexed_vars(const std::vector<StmtPtr>& stmts) {
    if (b.vector_mode()) return;
    for (const auto& s : stmts) scan_stmt_dyn_arrays(s.get(), dyn_indexed_vars);
}

brass::Value* MirBodyLowerer::alloc_entry_slot(uint32_t bytes, uint32_t align) {
    brass::Builder& raw = b.builder();
    brass::BasicBlock* saved = raw.current_block();
    brass::BasicBlock* entry = raw.current_function()->entry_block();
    if (brass::Instruction* term = entry->terminator()) raw.position_before(term);
    else raw.position_at_end(entry);
    brass::Value* slot = raw.build_alloca(bytes, align);
    raw.position_at_end(saved);
    return slot;
}

MirBodyLowerer::LocalPlace MirBodyLowerer::get_or_create_const_slot(const Type& t, const ValueList& leaves) {
    auto kinds = mod.types.flatten_scalars(t);
    uint32_t elem_bytes = 4;
    for (ScalarKind sk : kinds) {
        if (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) elem_bytes = 8;
    }
    uint32_t total_words = std::max<uint32_t>(1u, static_cast<uint32_t>(kinds.size()));
    std::vector<uint64_t> key;
    key.reserve(leaves.size() + 2);
    key.push_back(elem_bytes);
    key.push_back(kinds.empty() ? 0 : static_cast<uint64_t>(kinds.front()));
    for (const brass::Value* v : leaves) {
        uint64_t bits = 0;
        is_const_leaf(v, &bits);
        key.push_back(bits);
    }
    auto it = const_array_slots.find(key);
    if (it != const_array_slots.end()) {
        LocalPlace lp = it->second;
        lp.type = t;
        lp.word_offset = b.build_iconst_i32(0);
        lp.lanes.clear();
        return lp;
    }

    brass::Builder& raw = b.builder();
    brass::BasicBlock* saved = raw.current_block();
    brass::BasicBlock* entry = raw.current_function()->entry_block();
    if (brass::Instruction* term = entry->terminator()) raw.position_before(term);
    else raw.position_at_end(entry);
    brass::Value* slot = raw.build_alloca(total_words * elem_bytes, elem_bytes);
    for (size_t i = 0; i < kinds.size() && i < leaves.size(); ++i) {
        const brass::Instruction* def = leaves[i]->defining_instruction();
        brass::Value* c = nullptr;
        if (def->opcode() == brass::Opcode::iconst_i32) {
            c = raw.build_iconst_i32(def->imm_i32());
        } else if (def->opcode() == brass::Opcode::iconst_i64) {
            c = raw.build_iconst_i64(def->imm_i64());
        } else {
            c = def->type() == brass::Type::f32() ? raw.build_fconst_f32(static_cast<float>(def->imm_f64()))
                                                  : raw.build_fconst_f64(def->imm_f64());
        }
        raw.build_store(to_mir_scalar_type(kinds[i]), slot, static_cast<int32_t>(i * elem_bytes), c);
    }
    raw.position_at_end(saved);

    LocalPlace lp{slot, b.build_iconst_i32(0), t, {}, elem_bytes, total_words};
    const_array_slots[key] = lp;
    return lp;
}

MirBodyLowerer::LocalPlace MirBodyLowerer::materialize_array_slot(const Type& t, const ValueList& leaves,
                                                                  bool allow_const_slot) {
    bool all_const = allow_const_slot && !leaves.empty();
    for (const brass::Value* v : leaves) {
        if (b.vector_mode() && Ops::is_varying(v)) return {};
        if (!is_const_leaf(v)) all_const = false;
    }
    if (all_const) return get_or_create_const_slot(t, leaves);
    if (b.vector_mode()) return {};

    auto kinds = mod.types.flatten_scalars(t);
    uint32_t elem_bytes = 4;
    for (ScalarKind sk : kinds) {
        if (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) elem_bytes = 8;
    }
    uint32_t total_words = std::max<uint32_t>(1u, static_cast<uint32_t>(kinds.size()));
    brass::Value* slot = alloc_entry_slot(total_words * elem_bytes, elem_bytes);
    for (size_t i = 0; i < kinds.size() && i < leaves.size(); ++i) {
        b.builder().build_store(to_mir_scalar_type(kinds[i]), slot, static_cast<int32_t>(i * elem_bytes), leaves[i]);
    }
    return LocalPlace{slot, b.build_iconst_i32(0), t, {}, elem_bytes, total_words};
}

int MirBodyLowerer::alloc_local_place_var(const std::string& name, const Type& t, const ValueList& init_leaves,
                                          bool is_const_init) {
    int id = alloc_var(name, {});
    bool all_const = is_const_init && !init_leaves.empty();
    for (const brass::Value* v : init_leaves) {
        if (b.vector_mode() && Ops::is_varying(v)) {
            throw VectorUnsupported("varying local array with dynamic indexing");
        }
        if (!is_const_leaf(v)) all_const = false;
    }
    if (all_const) {
        local_places[id] = get_or_create_const_slot(t, init_leaves);
        return id;
    }
    local_places[id] = materialize_array_slot(t, init_leaves);
    return id;
}

bool MirBodyLowerer::is_local_path(const Expr* e, bool for_assign) const {
    bool has_dyn_array_idx = false;
    const Expr* cur = e;
    while (cur && (cur->kind == Expr::Kind::Field || cur->kind == Expr::Kind::Index)) {
        if (cur->kind == Expr::Kind::Index && cur->lhs && cur->lhs->resolved_type.is_array() &&
            !cur->args.empty() && cur->args[0]->kind != Expr::Kind::IntLit) {
            has_dyn_array_idx = true;
        }
        cur = cur->lhs.get();
    }
    if (!cur) return false;
    if (cur->kind == Expr::Kind::Ident) {
        int id = lookup_var_id(cur->name);
        if (id >= 0 && local_places.count(id)) return true;
    }
    if (!for_assign && has_dyn_array_idx && cur->resolved_type.is_array() &&
        !is_block_path(cur) && !is_shared_path(cur)) {
        return true;
    }
    return false;
}

bool MirBodyLowerer::local_place(const Expr* e, LocalPlace& p) {
    if (!e) return false;
    if (e->kind == Expr::Kind::Ident) {
        int id = lookup_var_id(e->name);
        auto lp = local_places.find(id);
        if (lp != local_places.end()) {
            p = lp->second;
            return true;
        }
    } else if (e->kind == Expr::Kind::Field) {
        if (local_place(e->lhs.get(), p) && p.lanes.empty()) {
            if (p.type.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(p.type.struct_id);
                const StructField* sf = sdef ? sdef->find_field(e->name) : nullptr;
                if (!sf) return false;
                p.word_offset = add_words(b, p.word_offset, sf->flat_leaf_offset);
                p.type = sf->type;
                return true;
            }
            if (p.type.is_vector()) {
                for (char c : e->name) p.lanes.push_back(swizzle_index(c));
                if (p.lanes.size() == 1) {
                    p.word_offset = add_words(b, p.word_offset, p.lanes[0]);
                    p.lanes.clear();
                    p.type = Type::scalar_type(p.type.scalar);
                } else {
                    p.type = Type::vec(p.type.scalar, static_cast<uint32_t>(p.lanes.size()));
                }
                return true;
            }
            return false;
        }
    } else if (e->kind == Expr::Kind::Index) {
        if (local_place(e->lhs.get(), p) && p.lanes.empty() && e->args.size() == 1 &&
            (p.type.is_array() || p.type.is_vector())) {
            Type elem = p.type.element_type();
            uint32_t count = std::max(1u, p.type.dimension);
            uint32_t stride_words = std::max(1u, mod.types.flat_leaf_count(elem));
            brass::Value* idx = lower_expr(e->args[0].get())[0];
            if (b.vector_mode() && Ops::is_varying(idx)) return false;
            if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
            const brass::Instruction* def = idx && idx->is_instruction() ? idx->defining_instruction() : nullptr;
            if (def && def->opcode() == brass::Opcode::iconst_i32) {
                uint32_t c = static_cast<uint32_t>(
                    std::clamp<int32_t>(def->imm_i32(), 0, static_cast<int32_t>(count) - 1));
                p.word_offset = add_words(b, p.word_offset, c * stride_words);
            } else {
                if (!is_gpu && count > 1) {
                    brass::Value* zero = b.build_iconst_i32(0);
                    brass::Value* max_i = b.build_iconst_i32(static_cast<int32_t>(count - 1));
                    idx = b.build_select(b.build_slt(idx, zero), zero,
                                         b.build_select(b.build_sgt(idx, max_i), max_i, idx));
                }
                brass::Value* delta =
                    stride_words == 1 ? idx : b.build_mul(idx, b.build_iconst_i32(static_cast<int32_t>(stride_words)));
                const brass::Instruction* wdef =
                    p.word_offset && p.word_offset->is_instruction() ? p.word_offset->defining_instruction() : nullptr;
                if (wdef && wdef->opcode() == brass::Opcode::iconst_i32 && wdef->imm_i32() == 0) {
                    p.word_offset = delta;
                } else {
                    p.word_offset = b.build_add(p.word_offset, delta);
                }
            }
            p.type = elem;
            return true;
        }
    }
    if (e->resolved_type.is_array()) {
        p = materialize_array_slot(e->resolved_type, lower_expr(e), true);
        return p.base != nullptr;
    }
    return false;
}

ValueList MirBodyLowerer::local_load(const Expr* e, const LocalPlace& p) {
    (void)e;
    ValueList out;
    auto load_word = [&](ScalarKind sk, uint32_t k) -> brass::Value* {
        brass::Type mt = to_mir_scalar_type(sk);
        auto esz = static_cast<uint8_t>(p.elem_bytes);
        const brass::Instruction* def =
            p.word_offset && p.word_offset->is_instruction() ? p.word_offset->defining_instruction() : nullptr;
        brass::Value* v = nullptr;
        if (def && def->opcode() == brass::Opcode::iconst_i32) {
            int32_t byte_off = static_cast<int32_t>((static_cast<uint32_t>(def->imm_i32()) + k) * esz);
            v = b.builder().build_load(mt, p.base, byte_off);
        } else {
            brass::Value* idx = is_gpu ? p.word_offset : b.build_sext_i64(p.word_offset);
            v = b.builder().build_load_indexed(mt, p.base, idx, esz, static_cast<int32_t>(k * esz));
        }
        return sk == ScalarKind::Bool ? b.build_ne(v, b.build_iconst_i32(0)) : v;
    };
    if (!p.lanes.empty()) {
        for (uint32_t lane : p.lanes) out.push_back(load_word(p.type.scalar, lane));
        return out;
    }
    auto kinds = mod.types.flatten_scalars(p.type);
    out.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        out.push_back(load_word(kinds[i], static_cast<uint32_t>(i)));
    }
    return out;
}

void MirBodyLowerer::local_store(const Expr* e, const LocalPlace& p, const ValueList& value) {
    (void)e;
    auto store_word = [&](ScalarKind sk, uint32_t k, brass::Value* val) {
        if (b.vector_mode() && (Ops::is_varying(val) || b.mask)) {
            throw VectorUnsupported("varying store to local array");
        }
        brass::Type mt = to_mir_scalar_type(sk);
        auto esz = static_cast<uint8_t>(p.elem_bytes);
        const brass::Instruction* def =
            p.word_offset && p.word_offset->is_instruction() ? p.word_offset->defining_instruction() : nullptr;
        if (def && def->opcode() == brass::Opcode::iconst_i32) {
            int32_t byte_off = static_cast<int32_t>((static_cast<uint32_t>(def->imm_i32()) + k) * esz);
            b.builder().build_store(mt, p.base, byte_off, val);
        } else {
            brass::Value* idx = is_gpu ? p.word_offset : b.build_sext_i64(p.word_offset);
            b.builder().build_store_indexed(mt, p.base, idx, esz, static_cast<int32_t>(k * esz), val);
        }
    };
    if (!p.lanes.empty()) {
        for (size_t i = 0; i < p.lanes.size() && i < value.size(); ++i) {
            store_word(p.type.scalar, p.lanes[i], value[i]);
        }
        return;
    }
    auto kinds = mod.types.flatten_scalars(p.type);
    for (size_t i = 0; i < kinds.size() && i < value.size(); ++i) {
        store_word(kinds[i], static_cast<uint32_t>(i), value[i]);
    }
}

std::optional<ValueList> MirBodyLowerer::try_local_read(const Expr* e) {
    if (!is_local_path(e, false)) return std::nullopt;
    LocalPlace p;
    if (!local_place(e, p)) return std::nullopt;
    return local_load(e, p);
}

bool MirBodyLowerer::try_local_assign(const Stmt* s, ValueList rhs) {
    const Expr* target = s->assign_target.get();
    if (!is_local_path(target, true)) return false;
    LocalPlace p;
    if (!local_place(target, p)) return false;
    if (s->compound_op.has_value()) {
        ValueList old_v = local_load(target, p);
        for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
            rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], target->resolved_type.scalar);
        }
    }
    local_store(target, p, rhs);
    return true;
}

void MirBodyLowerer::assign_local_lvalue(const Stmt* s, const ValueList& rhs) {
    std::function<void(const Expr*, const std::function<ValueList(const ValueList&)>&)> update_lvalue;
    update_lvalue = [&](const Expr* target, const std::function<ValueList(const ValueList&)>& write_fn) {
        if (!target) return;
        if (target->kind == Expr::Kind::Ident) {
            int id = lookup_var_id(target->name);
            if (id >= 0) var_values[id] = write_fn(var_values[id]);
            return;
        }
        if (target->kind == Expr::Kind::Field && target->lhs) {
            const Type& base_t = target->lhs->resolved_type;
            if (base_t.is_scalar()) {
                update_lvalue(target->lhs.get(), write_fn);
                return;
            }
            if (base_t.is_vector()) {
                const std::string& swz = target->name;
                update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                    ValueList old_sub;
                    old_sub.reserve(swz.size());
                    for (char c : swz) {
                        uint32_t idx = swizzle_index(c);
                        old_sub.push_back(idx < cur.size() ? cur[idx] : cur[0]);
                    }
                    ValueList new_sub = write_fn(old_sub);
                    ValueList next = cur;
                    for (size_t k = 0; k < swz.size() && k < new_sub.size(); ++k) {
                        uint32_t idx = swizzle_index(swz[k]);
                        if (idx < next.size()) next[idx] = new_sub[k];
                    }
                    return next;
                });
                return;
            }
            if (base_t.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(base_t.struct_id);
                const StructField* sf = sdef ? sdef->find_field(target->name) : nullptr;
                if (!sf) return;
                update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                    ValueList old_sub;
                    old_sub.reserve(sf->flat_leaf_count);
                    for (uint32_t k = 0; k < sf->flat_leaf_count; ++k) {
                        uint32_t idx = sf->flat_leaf_offset + k;
                        old_sub.push_back(idx < cur.size() ? cur[idx] : cur[0]);
                    }
                    ValueList new_sub = write_fn(old_sub);
                    ValueList next = cur;
                    for (uint32_t k = 0; k < sf->flat_leaf_count && k < new_sub.size(); ++k) {
                        uint32_t idx = sf->flat_leaf_offset + k;
                        if (idx < next.size()) next[idx] = new_sub[k];
                    }
                    return next;
                });
                return;
            }
            return;
        }
        if (target->kind == Expr::Kind::Index && target->lhs && !target->args.empty()) {
            const Type& base_t = target->lhs->resolved_type;
            if (base_t.is_scalar()) {
                update_lvalue(target->lhs.get(), write_fn);
                return;
            }
            ValueList idx_v = lower_expr(target->args[0].get());
            brass::Value* idx = idx_v.empty() ? b.build_iconst_i32(0) : idx_v[0];
            if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
            Type elem_t = base_t.element_type();
            uint32_t elem_leaves = mod.types.flat_leaf_count(elem_t);
            uint32_t count = base_t.dimension;
            const brass::Instruction* def = idx && idx->is_instruction() ? idx->defining_instruction() : nullptr;
            if (def && def->opcode() == brass::Opcode::iconst_i32) {
                uint32_t c = static_cast<uint32_t>(
                    std::clamp<int32_t>(def->imm_i32(), 0, static_cast<int32_t>(count) - 1));
                update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                    if (cur.empty() || elem_leaves == 0 || count == 0) return cur;
                    ValueList old_sub(elem_leaves, cur[0]);
                    for (uint32_t k = 0; k < elem_leaves; ++k) {
                        uint32_t off = c * elem_leaves + k;
                        if (off < cur.size()) old_sub[k] = cur[off];
                    }
                    ValueList new_sub = write_fn(old_sub);
                    ValueList next = cur;
                    for (uint32_t k = 0; k < elem_leaves && k < new_sub.size(); ++k) {
                        uint32_t off = c * elem_leaves + k;
                        if (off < next.size()) next[off] = new_sub[k];
                    }
                    return next;
                });
                return;
            }
            update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                if (cur.empty() || elem_leaves == 0 || count == 0) return cur;
                ValueList old_sub(elem_leaves, cur[0]);
                for (uint32_t k = 0; k < elem_leaves; ++k) {
                    brass::Value* acc = k < cur.size() ? cur[k] : cur[0];
                    for (uint32_t i = 1; i < count; ++i) {
                        uint32_t off = i * elem_leaves + k;
                        if (off < cur.size()) {
                            brass::Value* eq = b.build_eq(idx, b.build_iconst_i32(static_cast<int32_t>(i)));
                            acc = b.build_select(eq, cur[off], acc);
                        }
                    }
                    old_sub[k] = acc;
                }
                ValueList new_sub = write_fn(old_sub);
                ValueList next = cur;
                for (uint32_t i = 0; i < count; ++i) {
                    brass::Value* eq = b.build_eq(idx, b.build_iconst_i32(static_cast<int32_t>(i)));
                    for (uint32_t k = 0; k < elem_leaves && k < new_sub.size(); ++k) {
                        uint32_t off = i * elem_leaves + k;
                        if (off < next.size()) next[off] = b.build_select(eq, new_sub[k], next[off]);
                    }
                }
                return next;
            });
        }
    };
    ScalarKind sk = s->assign_target->resolved_type.scalar;
    update_lvalue(s->assign_target.get(), [&](const ValueList& old_v) -> ValueList {
        if (!s->compound_op.has_value()) return rhs;
        ValueList res = old_v;
        for (size_t k = 0; k < res.size() && k < rhs.size(); ++k) {
            res[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], sk);
        }
        return res;
    });
}

} // namespace brokernel::detail

// Buffers at descriptors: uniform<T> (std140) and buffer<T> / read_buffer<T>
// (an array of T, std430). An access is a path of fields, indices and
// swizzles from the block parameter, lowered to a byte offset (a dynamic
// index is arithmetic on it) and to brass's word loads and stores at that
// offset, of only the scalars the path names: a light of a 32-light array
// is read as its own words, not the array selected from. A block passed to a
// helper function is the placeholder value its parameter was given, which
// names the parameter, as a texture's does.

#include "lowering_internal.h"
#include <algorithm>
#include <cstring>

namespace brokernel::detail {

namespace {

uint32_t round_up(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

brass::Value* add_bytes(Ops& b, brass::Value* base, uint32_t bytes) {
    if (bytes == 0) return base;
    const brass::Instruction* def = base && base->is_instruction() ? base->defining_instruction() : nullptr;
    if (def && def->opcode() == brass::Opcode::iconst_i32) {
        return b.build_iconst_i32(def->imm_i32() + static_cast<int32_t>(bytes));
    }
    return b.build_add(base, b.build_iconst_i32(static_cast<int32_t>(bytes)));
}

} // namespace

// `e` as a place in a block: which parameter, the byte offset, the type
// there and, after a swizzle, the lanes it picks. False when `e` is not a
// path from a block parameter.
bool MirBodyLowerer::block_place(const Expr* e, BlockPlace& p) {
    if (!e) return false;
    switch (e->kind) {
        case Expr::Kind::Ident:
        case Expr::Kind::Path: {
            if (e->kind == Expr::Kind::Ident && e->global_index < 0) {
                int id = lookup_var_id(e->name);
                auto bp = block_places.find(id);
                if (bp != block_places.end()) {
                    p = bp->second;
                    return true;
                }
            }
            if (!e->resolved_type.is_block()) return false;
            ValueList root;
            if (e->global_index >= 0) {
                root = global_value(e);
            } else {
                int id = e->kind == Expr::Kind::Ident ? lookup_var_id(e->name) : -1;
                if (id < 0 || var_values[id].empty()) return false;
                root = var_values[id];
            }
            auto found = block_param.find(root[0]);
            if (found == block_param.end()) return false;
            p.param = found->second;
            used_block_params.insert(p.param);
            p.std140 = e->resolved_type.is_uniform();
            p.aligned16 = true;
            p.vec4_base = false;
            p.offset = b.build_iconst_i32(0);
            p.type = e->resolved_type.is_uniform() ? e->resolved_type.element_type() : e->resolved_type;
            p.lanes.clear();
            return true;
        }
        case Expr::Kind::Field: {
            if (!block_place(e->lhs.get(), p) || !p.lanes.empty()) return false;
            if (p.type.is_buffer() && p.type.element_type().is_struct()) {
                p.type = p.type.element_type();
            }
            if (p.type.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(p.type.struct_id);
                const StructField* sf = sdef ? sdef->find_field(e->name) : nullptr;
                if (!sf) return false;
                uint32_t foff = block_field_offset(*sdef, e->name, p.std140);
                p.offset = add_bytes(b, p.offset, foff);
                p.aligned16 = p.aligned16 && (foff % 16 == 0);
                p.type = sf->type;
                return true;
            }
            if (p.type.is_vector()) {
                for (char c : e->name) p.lanes.push_back(swizzle_index(c));
                if (p.lanes.size() == 1) {
                    uint32_t foff = 4 * p.lanes[0];
                    p.offset = add_bytes(b, p.offset, foff);
                    p.aligned16 = p.aligned16 && (foff % 16 == 0);
                    p.lanes.clear();
                    p.type = Type::scalar_type(p.type.scalar);
                } else {
                    p.vec4_base = p.aligned16 && (p.type.dimension == 4);
                    p.type = Type::vec(p.type.scalar, static_cast<uint32_t>(p.lanes.size()));
                }
                return true;
            }
            return false;
        }
        case Expr::Kind::Index: {
            if (!block_place(e->lhs.get(), p) || !p.lanes.empty() || e->args.size() != 1) return false;
            Type elem;
            uint32_t stride = 0;
            if (p.type.is_buffer()) {
                elem = p.type.element_type();
                stride = block_stride(elem, false);
            } else if (p.type.is_array()) {
                elem = p.type.element_type();
                stride = block_stride(elem, p.std140);
            } else if (p.type.is_vector()) {
                elem = Type::scalar_type(p.type.scalar);
                stride = 4;
            } else {
                return false;
            }
            brass::Value* idx = lower_expr(e->args[0].get())[0];
            if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
            const brass::Instruction* def = idx && idx->is_instruction() ? idx->defining_instruction() : nullptr;
            if (def && def->opcode() == brass::Opcode::iconst_i32 && def->imm_i32() >= 0) {
                uint32_t ioff = static_cast<uint32_t>(def->imm_i32()) * stride;
                p.offset = add_bytes(b, p.offset, ioff);
                p.aligned16 = p.aligned16 && (ioff % 16 == 0);
            } else {
                p.offset = b.build_add(p.offset, b.build_mul(idx, b.build_iconst_i32(static_cast<int32_t>(stride))));
                p.aligned16 = p.aligned16 && (stride % 16 == 0);
            }
            p.type = elem;
            return true;
        }
        default:
            return false;
    }
}

// Whether `e` is a path rooted at a block (its leftmost operand one).
bool MirBodyLowerer::is_block_path(const Expr* e) const {
    while (e && (e->kind == Expr::Kind::Field || e->kind == Expr::Kind::Index)) {
        if (e->lhs && e->lhs->resolved_type.is_block()) return true;
        e = e->lhs.get();
    }
    if (e && e->kind == Expr::Kind::Ident && e->global_index < 0) {
        int id = lookup_var_id(e->name);
        if (id >= 0 && block_places.count(id)) return true;
    }
    return false;
}

// The word offsets of a type's scalars, in leaf order, from `base`.
void MirBodyLowerer::block_words(const Type& t, bool std140, brass::Value* base, std::vector<brass::Value*>& out,
                                 std::vector<ScalarKind>& kinds) {
    auto at = [&](uint32_t bytes) { return add_bytes(b, base, bytes); };
    if (t.is_scalar()) {
        out.push_back(base);
        kinds.push_back(t.scalar);
    } else if (t.is_vector()) {
        for (uint32_t i = 0; i < t.dimension; ++i) {
            out.push_back(at(4 * i));
            kinds.push_back(t.scalar);
        }
    } else if (t.is_array()) {
        uint32_t stride = block_stride(t.element_type(), std140);
        for (uint32_t i = 0; i < t.dimension; ++i) block_words(t.element_type(), std140, at(stride * i), out, kinds);
    } else if (t.is_struct()) {
        if (const StructDef* sdef = mod.types.find_struct(t.struct_id)) {
            for (const auto& f : sdef->fields) {
                block_words(f.type, std140, at(block_field_offset(*sdef, f.name, std140)), out, kinds);
            }
        }
    }
}

ValueList MirBodyLowerer::block_load(const Expr* e, const BlockPlace& p) {
    const CheckedParam* param = &bound_param(p.param);
    for (ScalarKind sk : mod.types.flatten_scalars(p.type)) {
        if (sk == ScalarKind::Bool || sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) {
            diags.error(e->span, "a buffer holds 32-bit scalars, not '" + Type::scalar_type(sk).to_string() + "'");
            return make_zero_leaves(e->resolved_type);
        }
    }
    ValueList out;
    if (!is_gpu) {
        auto it = cpu_block_ptrs.find(p.param);
        brass::Value* base_ptr = it != cpu_block_ptrs.end() ? it->second.first : b.build_iconst_i64(0);
        auto load_cpu_scalar = [&](ScalarKind sk, brass::Value* byte_off) {
            brass::Type mt = sk == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
            return b.build_load_indexed(mt, base_ptr, b.build_sext_i64(byte_off), 1, 0);
        };
        if (!p.lanes.empty()) {
            for (uint32_t lane : p.lanes) {
                out.push_back(load_cpu_scalar(p.type.scalar, add_bytes(b, p.offset, 4 * lane)));
            }
            return out;
        }
        std::vector<brass::Value*> words;
        std::vector<ScalarKind> kinds;
        block_words(p.type, p.std140, p.offset, words, kinds);
        for (size_t i = 0; i < words.size(); ++i) {
            out.push_back(load_cpu_scalar(kinds[i], words[i]));
        }
        return out;
    }
    const uint32_t ubytes = uniform_bytes(*param);
    auto load_vec4 = [&](ScalarKind sk, brass::Value* off, const std::vector<uint32_t>& pick) {
        brass::Type vt = sk == ScalarKind::F32 ? brass::Type::f32x4() : brass::Type::i32x4();
        brass::Value* vec = kb.buffer_load(param->descriptor_set, param->descriptor_binding, ubytes, off, vt);
        brass::Value* lanes[4] = {};
        for (uint32_t l : pick) {
            if (!lanes[l]) lanes[l] = kb.builder().build_vextract_lane(vec, l);
            out.push_back(lanes[l]);
        }
    };
    if (!p.lanes.empty()) {
        if (p.vec4_base) {
            load_vec4(p.type.scalar, p.offset, p.lanes);
            return out;
        }
        brass::Type t = p.type.scalar == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
        for (uint32_t lane : p.lanes) {
            out.push_back(kb.buffer_load(param->descriptor_set, param->descriptor_binding, ubytes,
                                         add_bytes(b, p.offset, 4 * lane), t));
        }
        return out;
    }
    std::function<void(const Type&, uint32_t)> emit_loads = [&](const Type& t, uint32_t rel) {
        if (t.is_vector() && t.dimension == 4 && p.aligned16 && (rel % 16 == 0)) {
            load_vec4(t.scalar, add_bytes(b, p.offset, rel), {0, 1, 2, 3});
        } else if (t.is_scalar()) {
            brass::Type mt = t.scalar == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
            out.push_back(kb.buffer_load(param->descriptor_set, param->descriptor_binding, ubytes,
                                         add_bytes(b, p.offset, rel), mt));
        } else if (t.is_vector()) {
            brass::Type mt = t.scalar == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
            for (uint32_t i = 0; i < t.dimension; ++i) {
                out.push_back(kb.buffer_load(param->descriptor_set, param->descriptor_binding, ubytes,
                                             add_bytes(b, p.offset, rel + 4 * i), mt));
            }
        } else if (t.is_array()) {
            uint32_t stride = block_stride(t.element_type(), p.std140);
            for (uint32_t i = 0; i < t.dimension; ++i) emit_loads(t.element_type(), rel + stride * i);
        } else if (t.is_struct()) {
            if (const StructDef* sdef = mod.types.find_struct(t.struct_id)) {
                for (const auto& f : sdef->fields) {
                    emit_loads(f.type, rel + block_field_offset(*sdef, f.name, p.std140));
                }
            }
        }
    };
    emit_loads(p.type, 0);
    return out;
}

void MirBodyLowerer::block_store(const Expr* e, const BlockPlace& p, const ValueList& value) {
    const CheckedParam* param = &bound_param(p.param);
    if (!param->type.is_buffer() || param->type.access == AccessMode::Read) {
        diags.error(e->span, "'" + param->name + "' is read-only (" + param->type.to_string() + ")");
        return;
    }
    std::vector<brass::Value*> words;
    std::vector<ScalarKind> kinds;
    if (!p.lanes.empty()) {
        for (uint32_t lane : p.lanes) {
            words.push_back(add_bytes(b, p.offset, 4 * lane));
            kinds.push_back(p.type.scalar);
        }
    } else {
        block_words(p.type, p.std140, p.offset, words, kinds);
    }
    if (!is_gpu) {
        auto it = cpu_block_ptrs.find(p.param);
        brass::Value* base_ptr = it != cpu_block_ptrs.end() ? it->second.first : b.build_iconst_i64(0);
        for (size_t i = 0; i < words.size() && i < value.size(); ++i) {
            brass::Type mt = kinds[i] == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
            b.build_store_indexed(mt, base_ptr, b.build_sext_i64(words[i]), 1, 0, value[i]);
        }
        return;
    }
    for (size_t i = 0; i < words.size() && i < value.size(); ++i) {
        kb.buffer_store(param->descriptor_set, param->descriptor_binding, words[i], value[i]);
    }
}

uint32_t MirBodyLowerer::uniform_bytes(const CheckedParam& param) const {
    if (!param.type.is_uniform()) return 0;
    return round_up(block_layout(param.type.element_type(), true).size, 16);
}

// A read through a block, or nullopt when `e` is no path from one.
std::optional<ValueList> MirBodyLowerer::try_block_read(const Expr* e) {
    if (!is_block_path(e)) return std::nullopt;
    BlockPlace p;
    if (!block_place(e, p)) {
        diags.error(e->span, "this access to a buffer is not a path of fields, indices and swizzles");
        return make_zero_leaves(e->resolved_type);
    }
    return block_load(e, p);
}

// `target = value` (`target op= value`) when the target is a place in a
// storage buffer; false when it is not one.
bool MirBodyLowerer::try_block_assign(const Stmt* s, ValueList rhs) {
    const Expr* target = s->assign_target.get();
    if (!is_block_path(target)) return false;
    BlockPlace p;
    if (!block_place(target, p)) {
        diags.error(target->span, "this access to a buffer is not a path of fields, indices and swizzles");
        return true;
    }
    if (s->compound_op.has_value()) {
        ValueList old_v = block_load(target, p);
        for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
            rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], target->resolved_type.scalar);
        }
    }
    block_store(target, p, rhs);
    return true;
}

namespace {

brass::Value* cpu_atomic_op(Ops& b, brass::Type mt, bool is_u, brass::Value* base_ptr, brass::Value* index64,
                            uint8_t scale, const std::string& op, const std::vector<ValueList>& args,
                            const Ops::Bound* bound = nullptr) {
    brass::Value* val = (op == "atomic_cas") ? args[1][0] : args[0][0];
    brass::Value* cmp = (op == "atomic_cas") ? args[0][0] : nullptr;
    if (cmp && b.vector_mode() && Ops::is_varying(cmp)) {
        throw VectorUnsupported("varying atomic_cas");
    }
    brass::Builder& sb = b.builder();
    return b.build_atomic_rmw(mt, base_ptr, index64, scale, val, bound,
                              [&](brass::Value* cur, brass::Value* v) -> brass::Value* {
                                  if (op == "atomic_add") return sb.build_add(cur, v);
                                  if (op == "atomic_min") {
                                      return sb.build_select(is_u ? sb.build_ult(cur, v) : sb.build_slt(cur, v), cur, v);
                                  }
                                  if (op == "atomic_max") {
                                      return sb.build_select(is_u ? sb.build_ugt(cur, v) : sb.build_sgt(cur, v), cur, v);
                                  }
                                  if (op == "atomic_or") return sb.build_or(cur, v);
                                  if (op == "atomic_and") return sb.build_and(cur, v);
                                  if (op == "atomic_xor") return sb.build_xor(cur, v);
                                  if (op == "atomic_cas") return sb.build_select(sb.build_eq(cur, cmp), v, cur);
                                  return v;
                              });
}

} // namespace

ValueList MirBodyLowerer::block_atomic(const Expr* e, const BlockPlace& p, const std::string& op,
                                       const std::vector<ValueList>& args) {
    const CheckedParam* param = &bound_param(p.param);
    if (!param->type.is_buffer() || param->type.access == AccessMode::Read) {
        diags.error(e->span, "'" + param->name + "' is read-only (" + param->type.to_string() + ")");
        return make_zero_leaves(e->resolved_type);
    }
    if (!is_gpu) {
        auto it = cpu_block_ptrs.find(p.param);
        brass::Value* base_ptr = it != cpu_block_ptrs.end() ? it->second.first : b.build_iconst_i64(0);
        return {cpu_atomic_op(b, to_mir_scalar_type(p.type.scalar), p.type.is_u32(), base_ptr,
                              b.build_sext_i64(p.offset), 1, op, args)};
    }
    const uint32_t ds = param->descriptor_set;
    const uint32_t db = param->descriptor_binding;
    brass::Value* byte_off = p.offset;
    if (p.type.is_f32()) {
        if (op == "atomic_exchange") {
            brass::Value* vi = b.build_bitcast_i32_f32(args[0][0]);
            brass::Value* old = kb.buffer_atomic_exch_u32(ds, db, byte_off, vi);
            return {b.build_bitcast_f32_i32(old)};
        }
        // CAS loop for f32 atomic_add on a descriptor storage buffer.
        brass::Value* init_bits = kb.buffer_load(ds, db, 0, byte_off, brass::Type::i32());
        int cur_id = alloc_var("__fadd_cur", {init_bits});
        int old_id = alloc_var("__fadd_old", {b.build_fconst_f32(0.0f)});
        int broke_id = alloc_var("__fadd_broke", {b.build_iconst_i32(0)});
        int cont_id = alloc_var("__fadd_cont", {b.build_iconst_i32(0)});
        Stmt dummy_assign;
        dummy_assign.kind = Stmt::Kind::Assign;
        dummy_assign.assign_target = std::make_unique<Expr>();
        dummy_assign.assign_target->kind = Expr::Kind::Ident;
        dummy_assign.assign_target->name = "__fadd_cur";
        Stmt dummy_assign2;
        dummy_assign2.kind = Stmt::Kind::Assign;
        dummy_assign2.assign_target = std::make_unique<Expr>();
        dummy_assign2.assign_target->kind = Expr::Kind::Ident;
        dummy_assign2.assign_target->name = "__fadd_old";
        lower_loop(
            broke_id, cont_id, -1, {&dummy_assign, &dummy_assign2},
            [&] { return b.true_value(); },
            [&] {
                brass::Value* expected = var_values[cur_id][0];
                brass::Value* cur_f = b.build_bitcast_f32_i32(expected);
                brass::Value* next_f = b.build_add(cur_f, args[0][0]);
                brass::Value* desired = b.build_bitcast_i32_f32(next_f);
                brass::Value* prev = kb.buffer_atomic_cas_u32(ds, db, byte_off, expected, desired);
                var_values[old_id] = {cur_f};
                var_values[cur_id] = {prev};
                var_values[broke_id] = {b.build_eq(prev, expected)};
            },
            [] {});
        brass::Value* res = var_values[old_id][0];
        var_values.erase(cur_id);
        var_values.erase(old_id);
        var_values.erase(broke_id);
        var_values.erase(cont_id);
        return {res};
    }
    const bool u = p.type.is_u32();
    brass::Value* v0 = args[0][0];
    if (op == "atomic_add") return {u ? kb.buffer_atomic_add_u32(ds, db, byte_off, v0) : kb.buffer_atomic_add_i32(ds, db, byte_off, v0)};
    if (op == "atomic_min") return {u ? kb.buffer_atomic_min_u32(ds, db, byte_off, v0) : kb.buffer_atomic_min_i32(ds, db, byte_off, v0)};
    if (op == "atomic_max") return {u ? kb.buffer_atomic_max_u32(ds, db, byte_off, v0) : kb.buffer_atomic_max_i32(ds, db, byte_off, v0)};
    if (op == "atomic_exchange") return {u ? kb.buffer_atomic_exch_u32(ds, db, byte_off, v0) : kb.buffer_atomic_exch_i32(ds, db, byte_off, v0)};
    if (op == "atomic_or") return {u ? kb.buffer_atomic_or_u32(ds, db, byte_off, v0) : kb.buffer_atomic_or_i32(ds, db, byte_off, v0)};
    if (op == "atomic_and") return {u ? kb.buffer_atomic_and_u32(ds, db, byte_off, v0) : kb.buffer_atomic_and_i32(ds, db, byte_off, v0)};
    if (op == "atomic_xor") return {u ? kb.buffer_atomic_xor_u32(ds, db, byte_off, v0) : kb.buffer_atomic_xor_i32(ds, db, byte_off, v0)};
    if (op == "atomic_cas") {
        brass::Value* v1 = args[1][0];
        return {u ? kb.buffer_atomic_cas_u32(ds, db, byte_off, v0, v1) : kb.buffer_atomic_cas_i32(ds, db, byte_off, v0, v1)};
    }
    return {v0};
}

// ---------------------------------------------------------------------------
// Shared workgroup memory
// ---------------------------------------------------------------------------

void MirBodyLowerer::init_shared_memory() {
    auto alloc_shared = [&](const std::string& key, const Type& t) {
        if (shared_vars.count(key)) return;
        uint32_t words = std::max(1u, mod.types.flat_leaf_count(t));
        SharedVar sv;
        sv.type = t;
        if (is_gpu) {
            sv.gpu_base = kb.shared_alloc_i32(words);
        } else {
            sv.cpu_byte_offset = total_cpu_shared_bytes;
            total_cpu_shared_bytes += words * 4;
        }
        shared_vars.emplace(key, sv);
    };

    std::unordered_set<std::string> visited_fns;
    std::function<void(const Expr*)> scan_expr;
    std::function<void(const Stmt*, const std::string&)> scan_stmt;
    std::function<void(const CheckedFunction&)> scan_fn;

    scan_expr = [&](const Expr* e) {
        if (!e) return;
        if (e->global_index >= 0) {
            const CheckedParam& g = mod.globals[static_cast<size_t>(e->global_index)];
            if (g.role == ParamRole::Shared) {
                alloc_shared("g:" + std::to_string(e->global_index), g.type);
            }
        }
        if (e->kind == Expr::Kind::Call) {
            if (const CheckedFunction* callee = mod.find_function(e->name)) {
                scan_fn(*callee);
            }
        }
        scan_expr(e->lhs.get());
        scan_expr(e->rhs.get());
        scan_expr(e->cond.get());
        for (const auto& a : e->args) scan_expr(a.get());
        for (const auto& fi : e->fields) scan_expr(fi.value.get());
    };

    scan_stmt = [&](const Stmt* s, const std::string& fn_name) {
        if (!s) return;
        if (s->kind == Stmt::Kind::Var && s->is_shared) {
            alloc_shared("l:" + fn_name + ":" + s->var_name, s->resolved_var_type);
        }
        scan_expr(s->init_expr.get());
        scan_expr(s->assign_target.get());
        scan_expr(s->assign_value.get());
        scan_expr(s->cond.get());
        scan_expr(s->range_start.get());
        scan_expr(s->range_end.get());
        scan_expr(s->expr.get());
        scan_stmt(s->for_init.get(), fn_name);
        scan_stmt(s->for_step.get(), fn_name);
        for (const auto& b_stmt : s->body) scan_stmt(b_stmt.get(), fn_name);
        for (const auto& e_stmt : s->else_body) scan_stmt(e_stmt.get(), fn_name);
    };

    scan_fn = [&](const CheckedFunction& fn) {
        if (!visited_fns.insert(fn.name).second) return;
        for (const auto& s : fn.body) scan_stmt(s.get(), fn.name);
    };

    scan_fn(root_fn);
}

bool MirBodyLowerer::is_shared_path(const Expr* e) const {
    while (e && (e->kind == Expr::Kind::Field || e->kind == Expr::Kind::Index)) {
        e = e->lhs.get();
    }
    if (!e) return false;
    if (e->global_index >= 0) {
        return mod.globals[static_cast<size_t>(e->global_index)].role == ParamRole::Shared;
    }
    if (e->kind == Expr::Kind::Ident) {
        int id = lookup_var_id(e->name);
        if (id >= 0 && shared_places.count(id)) return true;
    }
    return false;
}

bool MirBodyLowerer::shared_place(const Expr* e, SharedPlace& p) {
    if (!e) return false;
    switch (e->kind) {
        case Expr::Kind::Ident:
        case Expr::Kind::Path: {
            if (e->global_index >= 0) {
                const CheckedParam& g = mod.globals[static_cast<size_t>(e->global_index)];
                if (g.role != ParamRole::Shared) return false;
                auto it = shared_vars.find("g:" + std::to_string(e->global_index));
                if (it == shared_vars.end()) return false;
                p.var = &it->second;
                p.word_offset = b.build_iconst_i32(0);
                p.type = g.type;
                p.lanes.clear();
                return true;
            }
            if (e->kind == Expr::Kind::Ident) {
                int id = lookup_var_id(e->name);
                auto sp = shared_places.find(id);
                if (sp != shared_places.end()) {
                    p = sp->second;
                    return true;
                }
            }
            return false;
        }
        case Expr::Kind::Field: {
            if (!shared_place(e->lhs.get(), p) || !p.lanes.empty()) return false;
            if (p.type.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(p.type.struct_id);
                const StructField* sf = sdef ? sdef->find_field(e->name) : nullptr;
                if (!sf) return false;
                p.word_offset = add_bytes(b, p.word_offset, sf->flat_leaf_offset);
                p.type = sf->type;
                return true;
            }
            if (p.type.is_vector()) {
                for (char c : e->name) p.lanes.push_back(swizzle_index(c));
                if (p.lanes.size() == 1) {
                    p.word_offset = add_bytes(b, p.word_offset, p.lanes[0]);
                    p.lanes.clear();
                    p.type = Type::scalar_type(p.type.scalar);
                } else {
                    p.type = Type::vec(p.type.scalar, static_cast<uint32_t>(p.lanes.size()));
                }
                return true;
            }
            return false;
        }
        case Expr::Kind::Index: {
            if (!shared_place(e->lhs.get(), p) || !p.lanes.empty() || e->args.size() != 1) return false;
            if (!p.type.is_array() && !p.type.is_vector()) return false;
            Type elem = p.type.element_type();
            uint32_t stride_words = std::max(1u, mod.types.flat_leaf_count(elem));
            brass::Value* idx = lower_expr(e->args[0].get())[0];
            if (idx->type() == brass::Type::i64()) idx = b.build_trunc_i32(idx);
            const brass::Instruction* def = idx && idx->is_instruction() ? idx->defining_instruction() : nullptr;
            if (def && def->opcode() == brass::Opcode::iconst_i32 && def->imm_i32() >= 0) {
                p.word_offset = add_bytes(b, p.word_offset, static_cast<uint32_t>(def->imm_i32()) * stride_words);
            } else {
                brass::Value* delta = stride_words == 1
                                          ? idx
                                          : b.build_mul(idx, b.build_iconst_i32(static_cast<int32_t>(stride_words)));
                p.word_offset = b.build_add(p.word_offset, delta);
            }
            p.type = elem;
            return true;
        }
        default:
            return false;
    }
}

ValueList MirBodyLowerer::shared_load(const Expr* e, const SharedPlace& p) {
    (void)e;
    ValueList out;
    auto load_word = [&](ScalarKind sk, brass::Value* w_off) -> brass::Value* {
        brass::Value* v = nullptr;
        if (is_gpu) {
            v = sk == ScalarKind::F32 ? kb.shared_load_f32_indexed(p.var->gpu_base, w_off)
                                      : kb.shared_load_i32_indexed(p.var->gpu_base, w_off);
        } else {
            brass::Value* base_ptr = b.build_add(cpu_shared_base, b.build_iconst_i64(p.var->cpu_byte_offset));
            brass::Type mt = sk == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
            v = b.build_load_indexed(mt, base_ptr, b.build_sext_i64(w_off), 4, 0);
        }
        return sk == ScalarKind::Bool ? b.build_ne(v, b.build_iconst_i32(0)) : v;
    };
    if (!p.lanes.empty()) {
        for (uint32_t lane : p.lanes) {
            out.push_back(load_word(p.type.scalar, add_bytes(b, p.word_offset, lane)));
        }
        return out;
    }
    auto kinds = mod.types.flatten_scalars(p.type);
    out.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        out.push_back(load_word(kinds[i], add_bytes(b, p.word_offset, static_cast<uint32_t>(i))));
    }
    return out;
}

void MirBodyLowerer::shared_store(const Expr* e, const SharedPlace& p, const ValueList& value) {
    (void)e;
    auto store_word = [&](ScalarKind sk, brass::Value* w_off, brass::Value* val) {
        if (is_gpu) {
            if (sk == ScalarKind::F32) {
                kb.shared_store_f32_indexed(p.var->gpu_base, w_off, val);
            } else {
                kb.shared_store_i32_indexed(p.var->gpu_base, w_off, val);
            }
            return;
        }
        brass::Value* base_ptr = b.build_add(cpu_shared_base, b.build_iconst_i64(p.var->cpu_byte_offset));
        brass::Type mt = sk == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
        b.build_store_indexed(mt, base_ptr, b.build_sext_i64(w_off), 4, 0, val);
    };
    if (!p.lanes.empty()) {
        for (size_t i = 0; i < p.lanes.size() && i < value.size(); ++i) {
            store_word(p.type.scalar, add_bytes(b, p.word_offset, p.lanes[i]), value[i]);
        }
        return;
    }
    auto kinds = mod.types.flatten_scalars(p.type);
    for (size_t i = 0; i < kinds.size() && i < value.size(); ++i) {
        store_word(kinds[i], add_bytes(b, p.word_offset, static_cast<uint32_t>(i)), value[i]);
    }
}

std::optional<ValueList> MirBodyLowerer::try_shared_read(const Expr* e) {
    if (!is_shared_path(e)) return std::nullopt;
    SharedPlace p;
    if (!shared_place(e, p)) {
        diags.error(e->span, "this access to shared memory is not a path of fields, indices and swizzles");
        return make_zero_leaves(e->resolved_type);
    }
    return shared_load(e, p);
}

bool MirBodyLowerer::try_shared_assign(const Stmt* s, ValueList rhs) {
    const Expr* target = s->assign_target.get();
    if (!is_shared_path(target)) return false;
    SharedPlace p;
    if (!shared_place(target, p)) {
        diags.error(target->span, "this access to shared memory is not a path of fields, indices and swizzles");
        return true;
    }
    if (s->compound_op.has_value()) {
        ValueList old_v = shared_load(target, p);
        for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
            rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], target->resolved_type.scalar);
        }
    }
    shared_store(target, p, rhs);
    return true;
}

ValueList MirBodyLowerer::shared_atomic(const Expr* e, const SharedPlace& p, const std::string& op,
                                        const std::vector<ValueList>& args) {
    (void)e;
    if (!is_gpu) {
        brass::Value* base_ptr = b.build_add(cpu_shared_base, b.build_iconst_i64(p.var->cpu_byte_offset));
        return {cpu_atomic_op(b, to_mir_scalar_type(p.type.scalar), p.type.is_u32(), base_ptr,
                              b.build_sext_i64(p.word_offset), 4, op, args)};
    }
    brass::Value* addr = p.var->gpu_base;
    const brass::Instruction* def =
        p.word_offset && p.word_offset->is_instruction() ? p.word_offset->defining_instruction() : nullptr;
    if (def && def->opcode() == brass::Opcode::iconst_i32) {
        if (def->imm_i32() != 0) {
            addr = b.build_add(addr, b.build_iconst_i64(static_cast<int64_t>(def->imm_i32()) * 4));
        }
    } else {
        addr = b.build_add(addr, b.build_mul(b.build_sext_i64(p.word_offset), b.build_iconst_i64(4)));
    }
    const bool u = p.type.is_u32();
    const bool f = p.type.is_f32();
    brass::Value* v0 = args[0][0];
    if (op == "atomic_add") {
        if (f) return {kb.atom_shared_add_f32(addr, v0)};
        return {u ? kb.atom_shared_add_u32(addr, v0) : kb.atom_shared_add_i32(addr, v0)};
    }
    if (op == "atomic_min") return {u ? kb.atom_shared_min_u32(addr, v0) : kb.atom_shared_min_i32(addr, v0)};
    if (op == "atomic_max") return {u ? kb.atom_shared_max_u32(addr, v0) : kb.atom_shared_max_i32(addr, v0)};
    if (op == "atomic_exchange") {
        if (f) return {kb.atom_shared_exch_f32(addr, v0)};
        return {u ? kb.atom_shared_exch_u32(addr, v0) : kb.atom_shared_exch_i32(addr, v0)};
    }
    if (op == "atomic_or") return {u ? kb.atom_shared_or_u32(addr, v0) : kb.atom_shared_or_i32(addr, v0)};
    if (op == "atomic_and") return {u ? kb.atom_shared_and_u32(addr, v0) : kb.atom_shared_and_i32(addr, v0)};
    if (op == "atomic_xor") return {u ? kb.atom_shared_xor_u32(addr, v0) : kb.atom_shared_xor_i32(addr, v0)};
    if (op == "atomic_cas") {
        brass::Value* v1 = args[1][0];
        return {u ? kb.atom_shared_cas_u32(addr, v0, v1) : kb.atom_shared_cas_i32(addr, v0, v1)};
    }
    return {v0};
}

} // namespace brokernel::detail

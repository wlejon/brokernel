// Buffers at descriptors: uniform<T> (std140) and buffer<T> / read_buffer<T>
// (an array of T, std430). An access is a path of fields, indices and
// swizzles from the block parameter, lowered to a byte offset (a dynamic
// index is arithmetic on it) and to brass's word loads and stores at that
// offset, of only the scalars the path names: a light of a 32-light array
// is read as its own words, not the array selected from. A block passed to a
// helper function is the placeholder value its parameter was given, which
// names the parameter, as a texture's does.

#include "lowering_internal.h"

namespace brokernel::detail {

namespace {

uint32_t round_up(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

} // namespace

// The std140 or std430 size and alignment of a type in a block.
MirBodyLowerer::BlockLayout MirBodyLowerer::block_layout(const Type& t, bool std140) const {
    if (t.is_scalar()) return {4, 4};
    if (t.is_vector()) {
        if (t.dimension == 2) return {8, 8};
        return {t.dimension * 4, 16};
    }
    if (t.is_array()) {
        uint32_t stride = block_stride(t.element_type(), std140);
        BlockLayout e = block_layout(t.element_type(), std140);
        return {stride * t.dimension, std140 ? std::max(e.align, 16u) : e.align};
    }
    if (t.is_struct()) {
        const StructDef* sdef = mod.types.find_struct(t.struct_id);
        uint32_t offset = 0, align = 4;
        if (sdef) {
            for (const auto& f : sdef->fields) {
                BlockLayout l = block_layout(f.type, std140);
                offset = round_up(offset, l.align) + l.size;
                align = std::max(align, l.align);
            }
        }
        if (std140) align = std::max(align, 16u);
        return {round_up(offset, align), align};
    }
    return {4, 4};
}

// The distance between an array's elements.
uint32_t MirBodyLowerer::block_stride(const Type& elem, bool std140) const {
    BlockLayout l = block_layout(elem, std140);
    return round_up(l.size, std140 ? std::max(l.align, 16u) : l.align);
}

uint32_t MirBodyLowerer::block_field_offset(const StructDef& sdef, const std::string& field, bool std140) const {
    uint32_t offset = 0;
    for (const auto& f : sdef.fields) {
        BlockLayout l = block_layout(f.type, std140);
        offset = round_up(offset, l.align);
        if (f.name == field) return offset;
        offset += l.size;
    }
    return offset;
}

// `e` as a place in a block: which parameter, the byte offset, the type
// there and, after a swizzle, the lanes it picks. False when `e` is not a
// path from a block parameter.
bool MirBodyLowerer::block_place(const Expr* e, BlockPlace& p) {
    if (!e) return false;
    switch (e->kind) {
        case Expr::Kind::Ident: {
            if (!e->resolved_type.is_block()) return false;
            int id = lookup_var_id(e->name);
            if (id < 0 || var_values[id].empty()) return false;
            auto found = block_param.find(var_values[id][0]);
            if (found == block_param.end()) return false;
            p.param = found->second;
            p.std140 = e->resolved_type.is_uniform();
            p.offset = b.build_iconst_i32(0);
            p.type = e->resolved_type.is_uniform() ? e->resolved_type.element_type() : e->resolved_type;
            p.lanes.clear();
            return true;
        }
        case Expr::Kind::Field: {
            if (!block_place(e->lhs.get(), p) || !p.lanes.empty()) return false;
            if (p.type.is_struct()) {
                const StructDef* sdef = mod.types.find_struct(p.type.struct_id);
                const StructField* sf = sdef ? sdef->find_field(e->name) : nullptr;
                if (!sf) return false;
                p.offset = b.build_add(p.offset, b.build_iconst_i32(static_cast<int32_t>(block_field_offset(*sdef, e->name, p.std140))));
                p.type = sf->type;
                return true;
            }
            if (p.type.is_vector()) {
                for (char c : e->name) p.lanes.push_back(swizzle_index(c));
                if (p.lanes.size() == 1) {
                    p.offset = b.build_add(p.offset, b.build_iconst_i32(static_cast<int32_t>(4 * p.lanes[0])));
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
            p.offset = b.build_add(p.offset, b.build_mul(idx, b.build_iconst_i32(static_cast<int32_t>(stride))));
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
    return false;
}

// The word offsets of a type's scalars, in leaf order, from `base`.
void MirBodyLowerer::block_words(const Type& t, bool std140, brass::Value* base, std::vector<brass::Value*>& out,
                                 std::vector<ScalarKind>& kinds) {
    auto at = [&](uint32_t bytes) { return bytes ? b.build_add(base, b.build_iconst_i32(static_cast<int32_t>(bytes))) : base; };
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

namespace {

const CheckedParam* block_param_of(const CheckedFunction& fn, size_t index) {
    return index < fn.params.size() ? &fn.params[index] : nullptr;
}

} // namespace

ValueList MirBodyLowerer::block_load(const Expr* e, const BlockPlace& p) {
    const CheckedParam* param = block_param_of(root_fn, p.param);
    std::vector<brass::Value*> words;
    std::vector<ScalarKind> kinds;
    if (!p.lanes.empty()) {
        for (uint32_t lane : p.lanes) {
            words.push_back(b.build_add(p.offset, b.build_iconst_i32(static_cast<int32_t>(4 * lane))));
            kinds.push_back(p.type.scalar);
        }
    } else {
        block_words(p.type, p.std140, p.offset, words, kinds);
    }
    ValueList out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (kinds[i] == ScalarKind::Bool || kinds[i] == ScalarKind::F64 || kinds[i] == ScalarKind::I64 ||
            kinds[i] == ScalarKind::U64) {
            diags.error(e->span, "a buffer holds 32-bit scalars, not '" + Type::scalar_type(kinds[i]).to_string() + "'");
            return make_zero_leaves(e->resolved_type);
        }
        brass::Type t = kinds[i] == ScalarKind::F32 ? brass::Type::f32() : brass::Type::i32();
        out.push_back(kb.buffer_load(param->descriptor_set, param->descriptor_binding, uniform_bytes(*param), words[i], t));
    }
    return out;
}

void MirBodyLowerer::block_store(const Expr* e, const BlockPlace& p, const ValueList& value) {
    const CheckedParam* param = block_param_of(root_fn, p.param);
    if (!param->type.is_buffer() || param->type.access == AccessMode::Read) {
        diags.error(e->span, "'" + param->name + "' is read-only (" + param->type.to_string() + ")");
        return;
    }
    std::vector<brass::Value*> words;
    std::vector<ScalarKind> kinds;
    if (!p.lanes.empty()) {
        for (uint32_t lane : p.lanes) {
            words.push_back(b.build_add(p.offset, b.build_iconst_i32(static_cast<int32_t>(4 * lane))));
            kinds.push_back(p.type.scalar);
        }
    } else {
        block_words(p.type, p.std140, p.offset, words, kinds);
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

} // namespace brokernel::detail

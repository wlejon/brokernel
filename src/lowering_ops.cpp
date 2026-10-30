#include "lowering_ops.h"

namespace brokernel::detail {

using brass::BasicBlock;
using brass::Opcode;
using brass::Type;
using brass::Value;

namespace {

bool is_64(Type t) { return t.element_type().size_in_bytes() == 8; }

} // namespace

Type Ops::vec_type(Type s) const {
    if (s.is_vector()) return s;
    const bool w8 = lanes == 8;
    if (s == Type::i32()) return w8 ? Type::i32x8() : Type::i32x4();
    if (s == Type::f32()) return w8 ? Type::f32x8() : Type::f32x4();
    if (s == Type::i64() || s == Type::f64()) {
        if (w8) throw NeedNarrowLanes();
        return s == Type::i64() ? Type::i64x4() : Type::f64x4();
    }
    throw VectorUnsupported("a varying " + std::string(s.name()));
}

void Ops::check_lanes(Type t) const {
    if (lanes == 8 && t.element_type().size_in_bytes() == 8) throw NeedNarrowLanes();
}

Value* Ops::widen(Value* v) {
    if (is_varying(v)) return v;
    auto m = markers_.find(v);
    if (m == markers_.end()) return b_.build_vbroadcast(vec_type(v->type()), v);
    const Type vt = vec_type(Type::i64());
    check_lanes(vt);
    const Marker mk = m->second;
    Value* steps = b_.build_vzero(vt);
    if (mk.off) {
        steps = b_.build_vmul(convert(Type::i64(), mk.off), b_.build_vbroadcast(vt, b_.build_iconst_i64(mk.stride)));
    } else {
        for (uint32_t i = 1; i < lanes; ++i) {
            steps = b_.build_vinsert_lane(steps, b_.build_iconst_i64(static_cast<int64_t>(i) * mk.stride), i);
        }
    }
    return b_.build_vadd(b_.build_vbroadcast(vt, v), steps);
}

Value* Ops::coerce(Value* v, Type t) {
    if (t.is_vector() && !is_varying(v)) return widen(v);
    return v;
}

Value* Ops::real(Value* v) {
    return is_index_marker(v) ? widen(v) : v;
}

Value* Ops::mark(Value* base, Marker m) {
    markers_[base] = m;
    return base;
}

std::optional<Ops::Affine> Ops::affine_of(Value* v) const {
    auto it = affine_.find(v);
    if (it == affine_.end()) return std::nullopt;
    return it->second;
}

Value* Ops::index_marker(Value* v) {
    if (auto a = affine_of(v)) return mark(b_.build_sext_i64(a->base), Marker{a->stride, nullptr});
    return mark(b_.build_sext_i64(b_.build_iconst_i32(0)), Marker{1, v});
}

Value* Ops::marker_lane(Value* m, uint32_t i) {
    const Marker mk = markers_.at(m);
    if (!mk.off) return i == 0 ? m : b_.build_add(m, b_.build_iconst_i64(static_cast<int64_t>(i) * mk.stride));
    Value* o = b_.build_sext_i64(b_.build_vextract_lane(mk.off, i));
    if (mk.stride != 1) o = b_.build_mul(o, b_.build_iconst_i64(mk.stride));
    return b_.build_add(m, o);
}

namespace {

std::optional<int64_t> int_constant(Value* v) {
    const brass::Instruction* d = v->defining_instruction();
    if (!d) return std::nullopt;
    if (d->opcode() == Opcode::iconst_i32) return d->imm_i32();
    if (d->opcode() == Opcode::iconst_i64) return d->imm_i64();
    return std::nullopt;
}

} // namespace

Value* Ops::affine_bin(Opcode op, Value* a, Value* c) {
    if (!vector_mode()) return nullptr;
    // A stride per operand: a marker's or an affine vector's, 0 for a uniform.
    auto stride_of = [&](Value* v, bool& known) -> int64_t {
        if (auto m = markers_.find(v); m != markers_.end()) return m->second.stride;
        if (auto f = affine_of(v)) return f->stride;
        known = !is_varying(v);
        return 0;
    };
    auto base_of = [&](Value* v) -> Value* {
        if (auto f = affine_of(v)) return f->base;
        return v;   // a marker is its own base, a uniform its own value
    };
    bool ka = true, kc = true;
    const int64_t sa = stride_of(a, ka), sc = stride_of(c, kc);
    if (!ka || !kc || (sa == 0 && sc == 0)) return nullptr;
    const bool marker = is_index_marker(a) || is_index_marker(c);
    if (marker && (is_varying(a) || is_varying(c))) return nullptr;
    // Two markers combine only over the same lane function.
    Value* off = nullptr;
    if (marker) {
        Value* const* oa = is_index_marker(a) ? &markers_.at(a).off : nullptr;
        Value* const* oc = is_index_marker(c) ? &markers_.at(c).off : nullptr;
        if (oa && oc && *oa != *oc) return nullptr;
        off = oa ? *oa : *oc;
    }
    int64_t stride = 0;
    Value* base = nullptr;
    switch (op) {
        case Opcode::vadd:
            stride = sa + sc;
            base = b_.build_add(base_of(a), base_of(c));
            break;
        case Opcode::vsub:
            stride = sa - sc;
            base = b_.build_sub(base_of(a), base_of(c));
            break;
        case Opcode::vmul: {
            // Affine times a constant.
            auto k = sa == 0 ? int_constant(a) : sc == 0 ? int_constant(c) : std::nullopt;
            if (!k) return nullptr;
            stride = (sa == 0 ? sc : sa) * *k;
            base = b_.build_mul(base_of(a), base_of(c));
            break;
        }
        default: return nullptr;
    }
    if (marker) return stride == 0 ? base : mark(base, Marker{stride, off});
    Value* r = vec_bin(op, a, c);
    if (stride != 0) affine_[r] = Affine{base, stride};
    return r;
}

Value* Ops::iota_i32() {
    Value* v = b_.build_vzero(vec_type(Type::i32()));
    for (uint32_t i = 1; i < lanes; ++i) v = b_.build_vinsert_lane(v, b_.build_iconst_i32(static_cast<int32_t>(i)), i);
    affine_[v] = Affine{b_.build_iconst_i32(0), 1};
    return v;
}

Value* Ops::any(Value* m) {
    return is_varying(m) ? b_.build_vany(m) : m;
}

Value* Ops::narrow_mask(Value* m) {
    if (m->type() == Type::i64x4()) return b_.build_vconvert(Type::i32x4(), m);
    return m;
}

Value* Ops::mask_for(Value* m, Type t) {
    m = widen(m);
    if (is_64(t) && m->type() == Type::i32x4()) return b_.build_vconvert(Type::i64x4(), m);
    return m;
}

brass::Instruction* Ops::build_br_if(Value* c, brass::BasicBlock* t, brass::BasicBlock* f) {
    if (is_varying(c)) throw std::logic_error("brokernel: a branch on a varying condition");
    return b_.build_br_if(c, t, f);
}

Value* Ops::lane_of(Value* v, uint32_t i) {
    if (is_index_marker(v)) return marker_lane(v, i);
    return is_varying(v) ? b_.build_vextract_lane(v, i) : v;
}

template <typename F>
Value* Ops::per_lane(Type res_scalar, std::initializer_list<Value*> operands, F scalar_op) {
    Value* out = b_.build_vzero(vec_type(res_scalar));
    std::vector<Value*> s(operands.size());
    for (uint32_t i = 0; i < lanes; ++i) {
        size_t k = 0;
        for (Value* v : operands) s[k++] = lane_of(v, i);
        out = b_.build_vinsert_lane(out, scalar_op(s), i);
    }
    return out;
}

Value* Ops::vec_bin(Opcode op, Value* a, Value* c) {
    if (!is_varying(a) && !is_varying(c)) return nullptr;
    a = widen(a);
    c = widen(c);
    switch (op) {
        case Opcode::vadd: return b_.build_vadd(a, c);
        case Opcode::vsub: return b_.build_vsub(a, c);
        case Opcode::vmul: return b_.build_vmul(a, c);
        case Opcode::vdiv: return b_.build_vdiv(a, c);
        case Opcode::vmin: return b_.build_vmin(a, c);
        case Opcode::vmax: return b_.build_vmax(a, c);
        case Opcode::vand: return b_.build_vand(a, c);
        case Opcode::vor:  return b_.build_vor(a, c);
        case Opcode::vxor: return b_.build_vxor(a, c);
        case Opcode::vshl: return b_.build_vshl(a, c);
        case Opcode::vlshr: return b_.build_vlshr(a, c);
        case Opcode::vashr: return b_.build_vashr(a, c);
        default: throw std::logic_error("brokernel: vec_bin of a non-binary opcode");
    }
}

Value* Ops::convert(Type to_scalar, Value* v) {
    if (v->type() == vec_type(to_scalar)) return v;
    return b_.build_vconvert(vec_type(to_scalar), v);
}

Value* Ops::bitcast(Type to_scalar, Value* v) {
    return b_.build_vbitcast(vec_type(to_scalar), v);
}

// --- Conversions ------------------------------------------------------------

Value* Ops::build_sext_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sext_i64(v);
    if (v->type().element_type() == Type::i32()) return index_marker(v);
    return convert(Type::i64(), v);
}
Value* Ops::build_zext_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_zext_i64(v);
    return b_.build_vand(convert(Type::i64(), v), widen(b_.build_iconst_i64(0xffffffffll)));
}
Value* Ops::build_trunc_i32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_trunc_i32(v);
    return convert(Type::i32(), v);
}
Value* Ops::build_fptosi_i32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fptosi_i32(v);
    return convert(Type::i32(), v);
}
Value* Ops::build_fptosi_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fptosi_i64(v);
    return per_lane(Type::i64(), {v}, [&](std::vector<Value*>& s) { return b_.build_fptosi_i64(s[0]); });
}
Value* Ops::build_fptosi_i32_f32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fptosi_i32_f32(v);
    return convert(Type::i32(), v);
}
Value* Ops::build_fptosi_i64_f32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fptosi_i64_f32(v);
    check_lanes(Type::i64());
    return per_lane(Type::i64(), {v}, [&](std::vector<Value*>& s) { return b_.build_fptosi_i64_f32(s[0]); });
}
Value* Ops::build_sitofp_f64_i32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sitofp_f64_i32(v);
    return convert(Type::f64(), v);
}
Value* Ops::build_sitofp_f64_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sitofp_f64_i64(v);
    return per_lane(Type::f64(), {v}, [&](std::vector<Value*>& s) { return b_.build_sitofp_f64_i64(s[0]); });
}
Value* Ops::build_sitofp_f32_i32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sitofp_f32_i32(v);
    return convert(Type::f32(), v);
}
Value* Ops::build_sitofp_f32_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sitofp_f32_i64(v);
    return per_lane(Type::f32(), {v}, [&](std::vector<Value*>& s) { return b_.build_sitofp_f32_i64(s[0]); });
}
Value* Ops::build_fptrunc_f32_f64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fptrunc_f32_f64(v);
    return convert(Type::f32(), v);
}
Value* Ops::build_fpext_f64_f32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fpext_f64_f32(v);
    return convert(Type::f64(), v);
}
Value* Ops::build_bitcast_i64_f64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_bitcast_i64_f64(v);
    return bitcast(Type::i64(), v);
}
Value* Ops::build_bitcast_f64_i64(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_bitcast_f64_i64(v);
    return bitcast(Type::f64(), v);
}
Value* Ops::build_bitcast_i32_f32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_bitcast_i32_f32(v);
    return bitcast(Type::i32(), v);
}
Value* Ops::build_bitcast_f32_i32(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_bitcast_f32_i32(v);
    return bitcast(Type::f32(), v);
}

// --- Arithmetic and logic -----------------------------------------------------

Value* Ops::build_add(Value* a, Value* c) {
    if (Value* r = affine_bin(Opcode::vadd, a, c)) return r;
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vadd, a, c)) return r;
    return b_.build_add(a, c);
}
Value* Ops::build_sub(Value* a, Value* c) {
    if (Value* r = affine_bin(Opcode::vsub, a, c)) return r;
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vsub, a, c)) return r;
    return b_.build_sub(a, c);
}
Value* Ops::build_mul(Value* a, Value* c) {
    if (Value* r = affine_bin(Opcode::vmul, a, c)) return r;
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_mul(a, c);
    const Type t = merge_type(a, c);
    if (t.element_type() == Type::i64()) {   // no 64-bit lane multiply in AVX2
        return per_lane(Type::i64(), {a, c}, [&](std::vector<Value*>& s) { return b_.build_mul(s[0], s[1]); });
    }
    return vec_bin(Opcode::vmul, a, c);
}
Value* Ops::build_fma(Value* a, Value* c, Value* d) {
    a = real(a);
    c = real(c);
    d = real(d);
    if (!is_varying(a) && !is_varying(c) && !is_varying(d)) return b_.build_fma(a, c, d);
    return b_.build_vfma(widen(a), widen(c), widen(d));
}

// A divisor no lane traps on: 1 where it is 0, or where a / d is MIN / -1
// (both undefined; the GPU does not trap either).
Value* Ops::safe_divisor(Value* a, Value* d) {
    const bool wide = d->type() == Type::i64();
    Value* zero = wide ? b_.build_iconst_i64(0) : b_.build_iconst_i32(0);
    Value* one = wide ? b_.build_iconst_i64(1) : b_.build_iconst_i32(1);
    Value* minus_one = wide ? b_.build_iconst_i64(-1) : b_.build_iconst_i32(-1);
    Value* min = wide ? b_.build_iconst_i64(INT64_MIN) : b_.build_iconst_i32(INT32_MIN);
    Value* bad = b_.build_or(b_.build_eq(d, zero), b_.build_and(b_.build_eq(a, min), b_.build_eq(d, minus_one)));
    return b_.build_select(bad, one, d);
}

Value* Ops::build_sdiv(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_sdiv(a, c);
    const Type t = merge_type(a, c);
    if (t.element_type().is_float()) return vec_bin(Opcode::vdiv, a, c);
    return per_lane(t.element_type(), {a, c}, [&](std::vector<Value*>& s) {
        return b_.build_sdiv(s[0], safe_divisor(s[0], s[1]));
    });
}
Value* Ops::build_udiv(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_udiv(a, c);
    return per_lane(merge_type(a, c).element_type(), {a, c}, [&](std::vector<Value*>& s) {
        return b_.build_udiv(s[0], safe_divisor(s[0], s[1]));
    });
}
Value* Ops::build_smod(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_smod(a, c);
    return per_lane(merge_type(a, c).element_type(), {a, c}, [&](std::vector<Value*>& s) {
        return b_.build_smod(s[0], safe_divisor(s[0], s[1]));
    });
}
Value* Ops::build_umod(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_umod(a, c);
    return per_lane(merge_type(a, c).element_type(), {a, c}, [&](std::vector<Value*>& s) {
        return b_.build_umod(s[0], safe_divisor(s[0], s[1]));
    });
}
Value* Ops::build_neg(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_neg(v);
    return b_.build_vneg(v);
}
Value* Ops::build_not(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_not(v);
    return b_.build_vnot(v);
}
Value* Ops::build_and(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vand, a, c)) return r;
    return b_.build_and(a, c);
}
Value* Ops::build_or(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vor, a, c)) return r;
    return b_.build_or(a, c);
}
Value* Ops::build_xor(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vxor, a, c)) return r;
    return b_.build_xor(a, c);
}

// The scalar shifts take the amount modulo the width (x64's shl/shr), the
// vector ones saturate: a varying amount is masked to agree.
Value* Ops::build_shl(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_shl(a, c);
    const bool wide = merge_type(a, c).element_type() == Type::i64();
    return vec_bin(Opcode::vshl, a, build_and(c, wide ? b_.build_iconst_i64(63) : b_.build_iconst_i32(31)));
}
Value* Ops::build_lshr(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_lshr(a, c);
    const bool wide = merge_type(a, c).element_type() == Type::i64();
    return vec_bin(Opcode::vlshr, a, build_and(c, wide ? b_.build_iconst_i64(63) : b_.build_iconst_i32(31)));
}
Value* Ops::build_ashr(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (!is_varying(a) && !is_varying(c)) return b_.build_ashr(a, c);
    const bool wide = merge_type(a, c).element_type() == Type::i64();
    return vec_bin(Opcode::vashr, a, build_and(c, wide ? b_.build_iconst_i64(63) : b_.build_iconst_i32(31)));
}
Value* Ops::build_sqrt(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_sqrt(v);
    return b_.build_vsqrt(v);
}
Value* Ops::build_floor(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_floor(v);
    return b_.build_vfloor(v);
}
Value* Ops::build_ceil(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_ceil(v);
    return b_.build_vceil(v);
}
Value* Ops::build_round(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_round(v);
    return b_.build_vround(v);
}
Value* Ops::build_fabs(Value* v) {
    v = real(v);
    if (!is_varying(v)) return b_.build_fabs(v);
    const bool wide = is_64(v->type());
    Value* bits = wide ? widen(b_.build_iconst_i64(0x7fffffffffffffffll)) : widen(b_.build_iconst_i32(0x7fffffff));
    return b_.build_vand(v, b_.build_vbitcast(v->type(), bits));
}
Value* Ops::build_fmin(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vmin, a, c)) return r;
    return b_.build_fmin(a, c);
}
Value* Ops::build_fmax(Value* a, Value* c) {
    a = real(a);
    c = real(c);
    if (Value* r = vec_bin(Opcode::vmax, a, c)) return r;
    return b_.build_fmax(a, c);
}

// --- Compares -------------------------------------------------------------------

Value* Ops::scalar_compare(Cmp c, bool u, Value* a, Value* d) {
    switch (c) {
        case Cmp::eq: return b_.build_eq(a, d);
        case Cmp::ne: return b_.build_ne(a, d);
        case Cmp::lt: return u ? b_.build_ult(a, d) : b_.build_slt(a, d);
        case Cmp::le: return u ? b_.build_ule(a, d) : b_.build_sle(a, d);
        case Cmp::gt: return u ? b_.build_ugt(a, d) : b_.build_sgt(a, d);
        default:      return u ? b_.build_uge(a, d) : b_.build_sge(a, d);
    }
}

Value* Ops::compare(Cmp c, bool u, Value* a, Value* d) {
    a = real(a);
    d = real(d);
    if (!is_varying(a) && !is_varying(d)) {
        Value* r = scalar_compare(c, u, a, d);
        return vector_mode() ? bool_to_mask(r) : r;
    }
    a = widen(a);
    d = widen(d);
    if (u && c != Cmp::eq && c != Cmp::ne) {
        // Unsigned order is signed order with the sign bits flipped.
        Value* sign = is_64(a->type()) ? widen(b_.build_iconst_i64(INT64_MIN)) : widen(b_.build_iconst_i32(INT32_MIN));
        a = b_.build_vxor(a, sign);
        d = b_.build_vxor(d, sign);
    }
    const Opcode op = c == Cmp::eq ? Opcode::vcmp_eq : c == Cmp::ne ? Opcode::vcmp_ne
                    : c == Cmp::lt ? Opcode::vcmp_lt : c == Cmp::le ? Opcode::vcmp_le
                    : c == Cmp::gt ? Opcode::vcmp_gt : Opcode::vcmp_ge;
    return narrow_mask(b_.build_vcmp(op, a, d));
}

Value* Ops::build_eq(Value* a, Value* c) { return compare(Cmp::eq, false, a, c); }
Value* Ops::build_ne(Value* a, Value* c) { return compare(Cmp::ne, false, a, c); }
Value* Ops::build_slt(Value* a, Value* c) { return compare(Cmp::lt, false, a, c); }
Value* Ops::build_sle(Value* a, Value* c) { return compare(Cmp::le, false, a, c); }
Value* Ops::build_sgt(Value* a, Value* c) { return compare(Cmp::gt, false, a, c); }
Value* Ops::build_sge(Value* a, Value* c) { return compare(Cmp::ge, false, a, c); }
Value* Ops::build_ult(Value* a, Value* c) { return compare(Cmp::lt, true, a, c); }
Value* Ops::build_ule(Value* a, Value* c) { return compare(Cmp::le, true, a, c); }
Value* Ops::build_ugt(Value* a, Value* c) { return compare(Cmp::gt, true, a, c); }
Value* Ops::build_uge(Value* a, Value* c) { return compare(Cmp::ge, true, a, c); }

Value* Ops::build_select(Value* c, Value* t, Value* f) {
    // A uniform choice of two indices of one stride is an index of that stride.
    if (!is_varying(c) && is_index_marker(t) && is_index_marker(f) && markers_.at(t) == markers_.at(f)) {
        const Marker m = markers_.at(t);
        return mark(b_.build_select(c, t, f), m);
    }
    c = real(c);
    t = real(t);
    f = real(f);
    if (!is_varying(c) && !is_varying(t) && !is_varying(f)) return b_.build_select(c, t, f);
    const Type vt = is_varying(t) ? t->type() : is_varying(f) ? f->type() : vec_type(t->type());
    return b_.build_vselect(mask_for(c, vt), coerce(t, vt), coerce(f, vt));
}

// --- Memory ---------------------------------------------------------------------

Value* Ops::build_load(Type t, Value* base, int32_t offset) {
    return b_.build_load(t, base, offset);
}

bool Ops::is_contiguous(const Value* index, uint8_t scale, Type t) const {
    auto m = markers_.find(index);
    return m != markers_.end() && !mask && !m->second.off &&
           m->second.stride * scale == static_cast<int64_t>(t.size_in_bytes());
}

static Value* zero_index(brass::Builder& b, const Value* index) {
    return index->type().element_type() == Type::i64() ? b.build_iconst_i64(0) : b.build_iconst_i32(0);
}

static Value* zero_scalar(brass::Builder& b, Type t) {
    if (t == Type::f32()) return b.build_fconst_f32(0.0f);
    if (t == Type::f64()) return b.build_fconst_f64(0.0);
    if (t == Type::i64()) return b.build_iconst_i64(0);
    return b.build_iconst_i32(0);
}

Value* Ops::in_bounds(Value* index, const Bound& bound) {
    // Unsigned, so a negative index is out too.
    return b_.build_ult(index, bound.count);
}

Value* Ops::group_in_bounds(const Bound& bound) {
    auto m = markers_.find(bound.index);
    if (m == markers_.end() || m->second.off || m->second.stride <= 0) return nullptr;
    Value* last = b_.build_add(bound.index, b_.build_iconst_i64(m->second.stride * (lanes - 1)));
    return b_.build_and(in_bounds(bound.index, bound), in_bounds(last, bound));
}

BasicBlock* Ops::side_block(std::string_view name) {
    BasicBlock* prev = b_.current_block();
    BasicBlock* bb = b_.append_block(name);
    if (prev) b_.position_at_end(prev);
    return bb;
}

Value* Ops::load_lanes(Type t, Value* base, Value* index, uint8_t scale, int32_t offset, const Bound* bound) {
    check_lanes(t);
    Value* out = b_.build_vzero(vec_type(t));
    for (uint32_t i = 0; i < lanes; ++i) {
        Value* idx = lane_of(index, i);
        Value* on = mask ? lane_of(mask, i) : nullptr;
        if (bound) {
            Value* in = in_bounds(lane_of(bound->index, i), *bound);
            on = on ? b_.build_select(in, on, b_.build_iconst_i32(0)) : in;
        }
        if (on) idx = b_.build_select(on, idx, zero_index(b_, idx));
        Value* v = b_.build_load_indexed(t, base, idx, scale, offset);
        if (bound) v = b_.build_select(on, v, zero_scalar(b_, t));
        out = b_.build_vinsert_lane(out, v, i);
    }
    return out;
}

Value* Ops::build_load_indexed(Type t, Value* base, Value* index, uint8_t scale, int32_t offset, const Bound* bound) {
    if (is_contiguous(index, scale, t)) {
        // The lanes' elements are consecutive.
        check_lanes(t);
        Value* addr = b_.build_add(base, b_.build_mul(index, b_.build_iconst_i64(scale)));
        Value* ok = bound ? group_in_bounds(*bound) : nullptr;
        if (!bound) return b_.build_vload(vec_type(t), addr, offset);
        if (ok) {
            // One test for the group; only a group running past its store
            // goes lane by lane.
            BasicBlock* fast = side_block("bounded_load_group");
            BasicBlock* slow = side_block("bounded_load_lanes");
            BasicBlock* join = side_block("bounded_load_join");
            Value* v = b_.add_block_param(join, vec_type(t));
            b_.build_br_if(ok, fast, slow);
            b_.position_at_end(fast);
            b_.build_br(join, {b_.build_vload(vec_type(t), addr, offset)});
            b_.position_at_end(slow);
            b_.build_br(join, {load_lanes(t, base, index, scale, offset, bound)});
            b_.position_at_end(join);
            return v;
        }
    }
    if (!is_varying(index) && !is_index_marker(index) && !(bound && is_varying(bound->index))) {
        if (!bound) return b_.build_load_indexed(t, base, index, scale, offset);
        Value* in = in_bounds(bound->index, *bound);
        Value* v = b_.build_load_indexed(t, base, b_.build_select(in, index, zero_index(b_, index)), scale, offset);
        return b_.build_select(in, v, zero_scalar(b_, t));
    }
    return load_lanes(t, base, index, scale, offset, bound);
}

void Ops::store_lanes(Type t, Value* base, Value* index, uint8_t scale, int32_t offset, Value* val,
                      const Bound* bound) {
    check_lanes(t);
    for (uint32_t i = 0; i < lanes; ++i) {
        Value* idx = lane_of(index, i);
        Value* where = base;
        Value* on = mask ? lane_of(mask, i) : nullptr;
        if (bound) {
            Value* in = in_bounds(lane_of(bound->index, i), *bound);
            on = on ? b_.build_select(in, on, b_.build_iconst_i32(0)) : in;
        }
        if (on) {
            where = b_.build_select(on, base, scratch);
            idx = b_.build_select(on, idx, zero_index(b_, idx));
        }
        b_.build_store_indexed(t, where, idx, scale, offset, lane_of(val, i));
    }
}

void Ops::build_store_indexed(Type t, Value* base, Value* index, uint8_t scale, int32_t offset, Value* val,
                              const Bound* bound) {
    val = real(val);
    const bool marker = is_index_marker(index);
    if ((mask || bound) && vector_mode() && (offset < 0 || offset > 248)) {
        throw VectorUnsupported("a masked store at offset " + std::to_string(offset));
    }
    if (is_contiguous(index, scale, t)) {
        check_lanes(t);
        Value* addr = b_.build_add(base, b_.build_mul(index, b_.build_iconst_i64(scale)));
        Value* ok = bound ? group_in_bounds(*bound) : nullptr;
        if (!bound) {
            b_.build_vstore(vec_type(t), addr, offset, widen(val));
            return;
        }
        if (ok) {
            BasicBlock* fast = side_block("bounded_store_group");
            BasicBlock* slow = side_block("bounded_store_lanes");
            BasicBlock* join = side_block("bounded_store_join");
            b_.build_br_if(ok, fast, slow);
            b_.position_at_end(fast);
            b_.build_vstore(vec_type(t), addr, offset, widen(val));
            b_.build_br(join);
            b_.position_at_end(slow);
            store_lanes(t, base, index, scale, offset, val, bound);
            b_.build_br(join);
            b_.position_at_end(join);
            return;
        }
    }
    const bool uniform = !marker && !is_varying(index) && !is_varying(val) && !(bound && is_varying(bound->index));
    if (uniform && !mask) {
        if (!bound) {
            b_.build_store_indexed(t, base, index, scale, offset, val);
            return;
        }
        // Stored only in bounds: a branch (the GPU has no scratch slot to
        // send it to).
        BasicBlock* store_bb = side_block("bounded_store");
        BasicBlock* done_bb = side_block("bounded_store_done");
        b_.build_br_if(in_bounds(bound->index, *bound), store_bb, done_bb);
        b_.position_at_end(store_bb);
        b_.build_store_indexed(t, base, index, scale, offset, val);
        b_.build_br(done_bb);
        b_.position_at_end(done_bb);
        return;
    }
    if (uniform) {
        // One store, by any active lane.
        Value* on = any(mask);
        if (bound) on = b_.build_select(in_bounds(bound->index, *bound), on, b_.build_iconst_i32(0));
        b_.build_store_indexed(t, b_.build_select(on, base, scratch), b_.build_select(on, index, zero_index(b_, index)),
                               scale, offset, val);
        return;
    }
    store_lanes(t, base, index, scale, offset, val, bound);
}

Value* Ops::build_atomic_rmw(Type t, Value* base, Value* index, uint8_t scale, Value* val, const Bound* bound,
                             const std::function<Value*(Value*, Value*)>& scalar_op) {
    val = real(val);
    if (!vector_mode()) {
        if (!bound) {
            Value* old_v = b_.build_load_indexed(t, base, index, scale, 0);
            b_.build_store_indexed(t, base, index, scale, 0, scalar_op(old_v, val));
            return old_v;
        }
        BasicBlock* atom_bb = side_block("bounded_atomic");
        BasicBlock* done_bb = side_block("bounded_atomic_done");
        Value* old = b_.add_block_param(done_bb, t);
        b_.build_br_if(in_bounds(bound->index, *bound), atom_bb, {}, done_bb, {zero_scalar(b_, t)});
        b_.position_at_end(atom_bb);
        Value* old_v = b_.build_load_indexed(t, base, index, scale, 0);
        b_.build_store_indexed(t, base, index, scale, 0, scalar_op(old_v, val));
        b_.build_br(done_bb, {old_v});
        b_.position_at_end(done_bb);
        return old;
    }
    check_lanes(t);
    Value* out = b_.build_vzero(vec_type(t));
    for (uint32_t i = 0; i < lanes; ++i) {
        Value* idx = lane_of(index, i);
        Value* v = lane_of(val, i);
        Value* on = mask ? lane_of(mask, i) : nullptr;
        if (bound) {
            Value* in = in_bounds(lane_of(bound->index, i), *bound);
            on = on ? b_.build_select(in, on, b_.build_iconst_i32(0)) : in;
        }
        Value* where = on ? b_.build_select(on, base, scratch) : base;
        Value* eff_idx = on ? b_.build_select(on, idx, zero_index(b_, idx)) : idx;
        Value* old_v = b_.build_load_indexed(t, where, eff_idx, scale, 0);
        if (on) old_v = b_.build_select(on, old_v, zero_scalar(b_, t));
        b_.build_store_indexed(t, where, eff_idx, scale, 0, scalar_op(old_v, v));
        out = b_.build_vinsert_lane(out, old_v, i);
    }
    return out;
}

Value* Ops::build_call(std::string_view callee, Type ret, std::initializer_list<Value*> args) {
    return build_call(callee, ret, std::vector<Value*>(args));
}
Value* Ops::build_call(std::string_view callee, Type ret, const std::vector<Value*>& args) {
    for (Value* a : args) {
        if (is_varying(a) || is_index_marker(a)) throw VectorUnsupported("a call to " + std::string(callee) + " with varying arguments");
    }
    return b_.build_call(callee, ret, args);
}

} // namespace brokernel::detail

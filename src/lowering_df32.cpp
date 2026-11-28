// Double-float (df32) MIR lowering:
//   - Represents each df32 scalar as two f32 MIR values `(hi, lo)`.
//   - On GPU: error-free Dekker/Knuth transforms using fma_f32.
//   - On CPU: exact native f64 arithmetic with (hi, lo) conversion at boundaries.
#include "lowering_internal.h"

#include <cmath>
#include <utility>

namespace brokernel::detail {

namespace {

using DfPair = std::pair<brass::Value*, brass::Value*>;

brass::Value* exact(brass::Value* v) {
    if (v && v->is_instruction()) {
        if (brass::Instruction* inst = v->defining_instruction()) {
            inst->set_no_contraction(true);
        }
    }
    return v;
}

DfPair quick_two_sum(Ops& b, brass::Value* x, brass::Value* y) {
    brass::Value* s = exact(b.build_add(x, y));
    brass::Value* e = exact(b.build_sub(y, exact(b.build_sub(s, x))));
    return {s, e};
}

DfPair two_sum(Ops& b, brass::Value* x, brass::Value* y) {
    brass::Value* s = exact(b.build_add(x, y));
    brass::Value* v = exact(b.build_sub(s, x));
    brass::Value* e = exact(b.build_add(
        exact(b.build_sub(x, exact(b.build_sub(s, v)))),
        exact(b.build_sub(y, v))));
    return {s, e};
}

DfPair two_prod(Ops& b, brass::Value* x, brass::Value* y) {
    brass::Value* p = exact(b.build_mul(x, y));
    brass::Value* e = exact(b.build_fma(x, y, exact(b.build_neg(p))));
    return {p, e};
}

brass::Value* df_to_f64(Ops& b, DfPair a) {
    return b.build_add(b.build_fpext_f64_f32(a.first), b.build_fpext_f64_f32(a.second));
}

DfPair f64_to_df(Ops& b, brass::Value* d) {
    brass::Value* hi = b.build_fptrunc_f32_f64(d);
    brass::Value* lo = b.build_fptrunc_f32_f64(b.build_sub(d, b.build_fpext_f64_f32(hi)));
    return {hi, lo};
}

DfPair df_add_gpu(Ops& b, DfPair x, DfPair y) {
    DfPair s = two_sum(b, x.first, y.first);
    DfPair t = two_sum(b, x.second, y.second);
    brass::Value* s2 = exact(b.build_add(s.second, t.first));
    DfPair v = quick_two_sum(b, s.first, s2);
    brass::Value* w = exact(b.build_add(v.second, t.second));
    return quick_two_sum(b, v.first, w);
}

DfPair df_sub_gpu(Ops& b, DfPair x, DfPair y) {
    return df_add_gpu(b, x, {b.build_neg(y.first), b.build_neg(y.second)});
}

DfPair df_mul_gpu(Ops& b, DfPair x, DfPair y) {
    DfPair p = two_prod(b, x.first, y.first);
    brass::Value* cross = exact(b.build_fma(x.first, y.second, exact(b.build_mul(x.second, y.first))));
    return quick_two_sum(b, p.first, exact(b.build_add(p.second, cross)));
}

DfPair df_div_gpu(Ops& b, DfPair x, DfPair y) {
    brass::Value* zero = b.build_fconst_f32(0.0f);
    brass::Value* q1 = exact(b.build_sdiv(x.first, y.first));
    DfPair p1 = df_mul_gpu(b, {q1, zero}, y);
    DfPair r = df_sub_gpu(b, x, p1);
    brass::Value* q2 = exact(b.build_sdiv(r.first, y.first));
    DfPair q12 = quick_two_sum(b, q1, q2);
    DfPair p2 = df_mul_gpu(b, q12, y);
    DfPair r2 = df_sub_gpu(b, x, p2);
    brass::Value* q3 = exact(b.build_sdiv(r2.first, y.first));
    return df_add_gpu(b, q12, {q3, zero});
}

DfPair df_sqrt_gpu(Ops& b, DfPair x) {
    brass::Value* zero = b.build_fconst_f32(0.0f);
    brass::Value* half = b.build_fconst_f32(0.5f);
    brass::Value* pos = b.build_sgt(x.first, zero);
    brass::Value* safe_hi = b.build_select(pos, x.first, b.build_fconst_f32(1.0f));
    brass::Value* xi = b.build_sqrt(safe_hi);
    DfPair xi2 = df_mul_gpu(b, {xi, zero}, {xi, zero});
    DfPair diff = df_sub_gpu(b, x, xi2);
    brass::Value* corr = exact(b.build_sdiv(exact(b.build_mul(diff.first, half)), xi));
    DfPair res = quick_two_sum(b, xi, corr);
    return {b.build_select(pos, res.first, zero), b.build_select(pos, res.second, zero)};
}

DfPair df_floor_gpu(Ops& b, DfPair x) {
    brass::Value* zero = b.build_fconst_f32(0.0f);
    brass::Value* fhi = b.build_floor(x.first);
    brass::Value* exact = b.build_eq(fhi, x.first);
    brass::Value* flo = b.build_floor(x.second);
    DfPair adj = quick_two_sum(b, fhi, flo);
    return {b.build_select(exact, adj.first, fhi), b.build_select(exact, adj.second, zero)};
}

DfPair df_add_pair(Ops& b, bool is_gpu, DfPair x, DfPair y) {
    return is_gpu ? df_add_gpu(b, x, y) : f64_to_df(b, b.build_add(df_to_f64(b, x), df_to_f64(b, y)));
}

DfPair df_sub_pair(Ops& b, bool is_gpu, DfPair x, DfPair y) {
    return is_gpu ? df_sub_gpu(b, x, y) : f64_to_df(b, b.build_sub(df_to_f64(b, x), df_to_f64(b, y)));
}

DfPair df_mul_pair(Ops& b, bool is_gpu, DfPair x, DfPair y) {
    return is_gpu ? df_mul_gpu(b, x, y) : f64_to_df(b, b.build_mul(df_to_f64(b, x), df_to_f64(b, y)));
}

DfPair df_div_pair(Ops& b, bool is_gpu, DfPair x, DfPair y) {
    return is_gpu ? df_div_gpu(b, x, y) : f64_to_df(b, b.build_sdiv(df_to_f64(b, x), df_to_f64(b, y)));
}

brass::Value* df_cmp_pair(Ops& b, bool is_gpu, BinaryOp op, DfPair x, DfPair y) {
    if (!is_gpu) {
        brass::Value* dx = df_to_f64(b, x);
        brass::Value* dy = df_to_f64(b, y);
        switch (op) {
            case BinaryOp::Eq: return b.build_eq(dx, dy);
            case BinaryOp::Ne: return b.build_ne(dx, dy);
            case BinaryOp::Lt: return b.build_slt(dx, dy);
            case BinaryOp::Le: return b.build_sle(dx, dy);
            case BinaryOp::Gt: return b.build_sgt(dx, dy);
            case BinaryOp::Ge: return b.build_sge(dx, dy);
            default: return b.build_iconst_i32(0);
        }
    }
    DfPair d = df_sub_gpu(b, x, y);
    brass::Value* zero = b.build_fconst_f32(0.0f);
    brass::Value* hi_z = b.build_eq(d.first, zero);
    brass::Value* lo_z = b.build_eq(d.second, zero);
    switch (op) {
        case BinaryOp::Eq: return b.build_and(hi_z, lo_z);
        case BinaryOp::Ne: return b.build_ne(b.build_and(hi_z, lo_z), b.build_iconst_i32(1));
        case BinaryOp::Lt: return b.build_select(hi_z, b.build_slt(d.second, zero), b.build_slt(d.first, zero));
        case BinaryOp::Le: return b.build_select(hi_z, b.build_sle(d.second, zero), b.build_slt(d.first, zero));
        case BinaryOp::Gt: return b.build_select(hi_z, b.build_sgt(d.second, zero), b.build_sgt(d.first, zero));
        case BinaryOp::Ge: return b.build_select(hi_z, b.build_sge(d.second, zero), b.build_sgt(d.first, zero));
        default: return b.build_iconst_i32(0);
    }
}

DfPair df_abs_pair(Ops& b, DfPair x) {
    brass::Value* zero = b.build_fconst_f32(0.0f);
    brass::Value* neg = b.build_slt(x.first, zero);
    return {b.build_select(neg, b.build_neg(x.first), x.first),
            b.build_select(neg, b.build_neg(x.second), x.second)};
}

DfPair df_select_pair(Ops& b, brass::Value* cond, DfPair t, DfPair f) {
    return {b.build_select(cond, t.first, f.first), b.build_select(cond, t.second, f.second)};
}

} // namespace

ValueList MirBodyLowerer::lower_df32_const(double val) {
    float hi = static_cast<float>(val);
    float lo = std::isfinite(val) ? static_cast<float>(val - static_cast<double>(hi)) : 0.0f;
    return {b.build_fconst_f32(hi), b.build_fconst_f32(lo)};
}

ValueList MirBodyLowerer::lower_df32_cast_to(const ValueList& src, ScalarKind from_sk) {
    if (from_sk == ScalarKind::DF32 && src.size() >= 2) return {src[0], src[1]};
    brass::Value* zero = b.build_fconst_f32(0.0f);
    if (from_sk == ScalarKind::F64 && !src.empty()) {
        DfPair p = f64_to_df(b, src[0]);
        return {p.first, p.second};
    }
    if (!src.empty()) {
        brass::Value* f = lower_scalar_cast(src[0], from_sk, ScalarKind::F32);
        return {f, zero};
    }
    return {zero, zero};
}

brass::Value* MirBodyLowerer::lower_df32_cast_from(brass::Value* hi, brass::Value* lo, ScalarKind to_sk) {
    if (to_sk == ScalarKind::F64) {
        return df_to_f64(b, {hi, lo});
    }
    if (to_sk == ScalarKind::F32) {
        return b.build_add(hi, lo);
    }
    if (to_sk == ScalarKind::Bool) {
        return b.build_ne(b.build_add(hi, lo), b.build_fconst_f32(0.0f));
    }
    return lower_scalar_cast(df_to_f64(b, {hi, lo}), ScalarKind::F64, to_sk);
}

ValueList MirBodyLowerer::lower_df32_binary(BinaryOp op, const ValueList& lhs, const ValueList& rhs) {
    const size_t lanes_n = std::max(lhs.size(), rhs.size()) / 2;
    ValueList out;
    out.reserve(op >= BinaryOp::Eq && op <= BinaryOp::Ge ? lanes_n : lanes_n * 2);
    for (size_t i = 0; i < lanes_n; ++i) {
        const size_t li = (2 * i + 1 < lhs.size()) ? 2 * i : 0;
        const size_t ri = (2 * i + 1 < rhs.size()) ? 2 * i : 0;
        DfPair x{lhs[li], lhs[li + 1]};
        DfPair y{rhs[ri], rhs[ri + 1]};
        if (op == BinaryOp::Add) {
            DfPair r = df_add_pair(b, is_gpu, x, y);
            out.push_back(r.first);
            out.push_back(r.second);
        } else if (op == BinaryOp::Sub) {
            DfPair r = df_sub_pair(b, is_gpu, x, y);
            out.push_back(r.first);
            out.push_back(r.second);
        } else if (op == BinaryOp::Mul) {
            DfPair r = df_mul_pair(b, is_gpu, x, y);
            out.push_back(r.first);
            out.push_back(r.second);
        } else if (op == BinaryOp::Div) {
            DfPair r = df_div_pair(b, is_gpu, x, y);
            out.push_back(r.first);
            out.push_back(r.second);
        } else {
            out.push_back(df_cmp_pair(b, is_gpu, op, x, y));
        }
    }
    return out;
}

ValueList MirBodyLowerer::lower_df32_math(const std::string& name, const std::vector<ValueList>& args) {
    if (name == "dot" && args.size() == 2) {
        const ValueList& a = args[0];
        const ValueList& c = args[1];
        DfPair acc{b.build_fconst_f32(0.0f), b.build_fconst_f32(0.0f)};
        for (size_t i = 0; i + 1 < a.size() && i + 1 < c.size(); i += 2) {
            DfPair prod = df_mul_pair(b, is_gpu, {a[i], a[i + 1]}, {c[i], c[i + 1]});
            acc = (i == 0) ? prod : df_add_pair(b, is_gpu, acc, prod);
        }
        return {acc.first, acc.second};
    }
    if (name == "length" && args.size() == 1) {
        const ValueList& a = args[0];
        if (a.size() == 2) {
            DfPair r = df_abs_pair(b, {a[0], a[1]});
            return {r.first, r.second};
        }
        DfPair acc{b.build_fconst_f32(0.0f), b.build_fconst_f32(0.0f)};
        for (size_t i = 0; i + 1 < a.size(); i += 2) {
            DfPair p{a[i], a[i + 1]};
            DfPair sq = df_mul_pair(b, is_gpu, p, p);
            acc = (i == 0) ? sq : df_add_pair(b, is_gpu, acc, sq);
        }
        DfPair r = is_gpu ? df_sqrt_gpu(b, acc) : f64_to_df(b, b.build_sqrt(df_to_f64(b, acc)));
        return {r.first, r.second};
    }
    if (name == "distance" && args.size() == 2) {
        ValueList diff = lower_df32_binary(BinaryOp::Sub, args[0], args[1]);
        return lower_df32_math("length", {diff});
    }
    if (name == "select" && args.size() == 3) {
        const ValueList& cond = args[0];
        const ValueList& a = args[1];
        const ValueList& c = args[2];
        const size_t lanes_n = std::min(a.size(), c.size()) / 2;
        ValueList out;
        out.reserve(lanes_n * 2);
        for (size_t i = 0; i < lanes_n; ++i) {
            brass::Value* ci = cond[i < cond.size() ? i : 0];
            DfPair r = df_select_pair(b, ci, {a[2 * i], a[2 * i + 1]}, {c[2 * i], c[2 * i + 1]});
            out.push_back(r.first);
            out.push_back(r.second);
        }
        return out;
    }

    const size_t lanes_n = args.empty() ? 0 : (args[0].size() / 2);
    ValueList out;
    out.reserve(lanes_n * 2);
    for (size_t i = 0; i < lanes_n; ++i) {
        DfPair a0{args[0][2 * i], args[0][2 * i + 1]};
        DfPair r{b.build_fconst_f32(0.0f), b.build_fconst_f32(0.0f)};
        if (name == "abs") {
            r = df_abs_pair(b, a0);
        } else if (name == "sqrt") {
            r = is_gpu ? df_sqrt_gpu(b, a0) : f64_to_df(b, b.build_sqrt(df_to_f64(b, a0)));
        } else if (name == "floor") {
            r = is_gpu ? df_floor_gpu(b, a0) : f64_to_df(b, b.build_floor(df_to_f64(b, a0)));
        } else if (name == "fract") {
            DfPair fl = is_gpu ? df_floor_gpu(b, a0) : f64_to_df(b, b.build_floor(df_to_f64(b, a0)));
            r = df_sub_pair(b, is_gpu, a0, fl);
        } else if (name == "min" && args.size() >= 2) {
            const size_t i1 = (2 * i + 1 < args[1].size()) ? 2 * i : 0;
            DfPair a1{args[1][i1], args[1][i1 + 1]};
            r = df_select_pair(b, df_cmp_pair(b, is_gpu, BinaryOp::Lt, a0, a1), a0, a1);
        } else if (name == "max" && args.size() >= 2) {
            const size_t i1 = (2 * i + 1 < args[1].size()) ? 2 * i : 0;
            DfPair a1{args[1][i1], args[1][i1 + 1]};
            r = df_select_pair(b, df_cmp_pair(b, is_gpu, BinaryOp::Gt, a0, a1), a0, a1);
        } else if (name == "clamp" && args.size() >= 3) {
            const size_t i1 = (2 * i + 1 < args[1].size()) ? 2 * i : 0;
            const size_t i2 = (2 * i + 1 < args[2].size()) ? 2 * i : 0;
            DfPair lo{args[1][i1], args[1][i1 + 1]};
            DfPair hi{args[2][i2], args[2][i2 + 1]};
            DfPair c1 = df_select_pair(b, df_cmp_pair(b, is_gpu, BinaryOp::Gt, lo, a0), lo, a0);
            r = df_select_pair(b, df_cmp_pair(b, is_gpu, BinaryOp::Gt, c1, hi), hi, c1);
        } else if ((name == "mix" || name == "lerp") && args.size() >= 3) {
            const size_t i1 = (2 * i + 1 < args[1].size()) ? 2 * i : 0;
            const size_t i2 = (2 * i + 1 < args[2].size()) ? 2 * i : 0;
            DfPair y{args[1][i1], args[1][i1 + 1]};
            DfPair t{args[2][i2], args[2][i2 + 1]};
            r = df_add_pair(b, is_gpu, a0, df_mul_pair(b, is_gpu, df_sub_pair(b, is_gpu, y, a0), t));
        } else if (name == "fma" && args.size() >= 3) {
            const size_t i1 = (2 * i + 1 < args[1].size()) ? 2 * i : 0;
            const size_t i2 = (2 * i + 1 < args[2].size()) ? 2 * i : 0;
            DfPair y{args[1][i1], args[1][i1 + 1]};
            DfPair z{args[2][i2], args[2][i2 + 1]};
            r = df_add_pair(b, is_gpu, df_mul_pair(b, is_gpu, a0, y), z);
        } else {
            diags.error(SourceSpan{}, "internal: unsupported df32 builtin '" + name + "'");
            r = a0;
        }
        out.push_back(r.first);
        out.push_back(r.second);
    }
    return out;
}

ValueList MirBodyLowerer::lower_df32_field(const ValueList& recv, const Type& rt, std::string_view fname) {
    if (fname == "hi" || fname == "lo") {
        const size_t off = (fname == "hi") ? 0 : 1;
        ValueList out;
        for (size_t i = off; i < recv.size(); i += 2) out.push_back(recv[i]);
        return out;
    }
    if (rt.is_vector()) {
        ValueList out;
        out.reserve(fname.size() * 2);
        for (char c : fname) {
            const uint32_t idx = swizzle_index(c);
            const size_t base = static_cast<size_t>(idx) * 2;
            out.push_back(base < recv.size() ? recv[base] : recv[0]);
            out.push_back(base + 1 < recv.size() ? recv[base + 1] : recv[1]);
        }
        return out;
    }
    return recv;
}

ValueList MirBodyLowerer::lower_df32_cast_expr(const ValueList& sub, ScalarKind from_sk, ScalarKind to_sk) {
    if (to_sk == ScalarKind::DF32) {
        if (from_sk == ScalarKind::DF32) return sub;
        ValueList out;
        out.reserve(sub.size() * 2);
        for (brass::Value* v : sub) {
            ValueList p = lower_df32_cast_to({v}, from_sk);
            out.push_back(p[0]);
            out.push_back(p[1]);
        }
        return out;
    }
    ValueList out;
    out.reserve(sub.size() / 2);
    for (size_t i = 0; i + 1 < sub.size(); i += 2) {
        out.push_back(lower_df32_cast_from(sub[i], sub[i + 1], to_sk));
    }
    return out;
}

std::optional<ValueList> MirBodyLowerer::try_lower_df32_call(const Expr* e) {
    const std::string& n = e->name;
    if (n == "df32") {
        if (e->args.size() == 2) {
            ValueList h = lower_expr(e->args[0].get());
            ValueList l = lower_expr(e->args[1].get());
            return ValueList{h.empty() ? b.build_fconst_f32(0.0f) : h[0],
                             l.empty() ? b.build_fconst_f32(0.0f) : l[0]};
        }
        if (e->args.size() == 1) {
            ValueList sv = lower_expr(e->args[0].get());
            return lower_df32_cast_to(sv, e->args[0]->resolved_type.scalar);
        }
        return make_zero_leaves(e->resolved_type);
    }
    if ((n == "f32" || n == "f64" || n == "i32" || n == "u32" || n == "i64" || n == "u64" || n == "bool") &&
        !e->args.empty() && e->args[0]->resolved_type.scalar == ScalarKind::DF32) {
        return lower_df32_cast_expr(lower_expr(e->args[0].get()), ScalarKind::DF32, e->resolved_type.scalar);
    }
    if (n == "__splat" && e->resolved_type.scalar == ScalarKind::DF32) {
        ValueList elem = lower_expr(e->args[0].get());
        if (elem.size() < 2) return make_zero_leaves(e->resolved_type);
        uint32_t count = mod.types.flat_leaf_count(e->resolved_type) / 2;
        ValueList out;
        out.reserve(count * 2);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(elem[0]);
            out.push_back(elem[1]);
        }
        return out;
    }
    if ((n == "vec2" || n == "vec3" || n == "vec4" || n == "vec") && e->resolved_type.scalar == ScalarKind::DF32) {
        uint32_t target_leaves = mod.types.flat_leaf_count(e->resolved_type);
        ValueList flat;
        for (const auto& a : e->args) {
            ValueList av = lower_expr(a.get());
            if (a->resolved_type.scalar != ScalarKind::DF32) {
                av = lower_df32_cast_expr(av, a->resolved_type.scalar, ScalarKind::DF32);
            }
            flat.insert(flat.end(), av.begin(), av.end());
        }
        if (flat.empty()) return make_zero_leaves(e->resolved_type);
        if (flat.size() == 2 && target_leaves > 2) {
            ValueList out;
            out.reserve(target_leaves);
            for (uint32_t i = 0; i + 1 < target_leaves; i += 2) {
                out.push_back(flat[0]);
                out.push_back(flat[1]);
            }
            return out;
        }
        while (flat.size() + 2 <= target_leaves) {
            flat.push_back(flat[flat.size() - 2]);
            flat.push_back(flat[flat.size() - 1]);
        }
        flat.resize(target_leaves);
        return flat;
    }
    const bool arg0_df = !e->args.empty() && e->args[0]->resolved_type.scalar == ScalarKind::DF32;
    const bool arg1_df = e->args.size() >= 2 && e->args[1]->resolved_type.scalar == ScalarKind::DF32;
    if (!arg0_df && !arg1_df && e->resolved_type.scalar != ScalarKind::DF32) {
        return std::nullopt;
    }
    if ((n == "isnan" || n == "isinf") && !e->args.empty() && e->args[0]->resolved_type.scalar == ScalarKind::DF32) {
        ValueList a = lower_expr(e->args[0].get());
        return ValueList{lower_float_class(b, n, a[0], ScalarKind::F32)};
    }
    if (n == "abs" || n == "sqrt" || n == "floor" || n == "fract" || n == "min" || n == "max" ||
        n == "clamp" || n == "mix" || n == "lerp" || n == "fma" || n == "dot" || n == "length" ||
        n == "distance" || n == "select") {
        std::vector<ValueList> lowered;
        lowered.reserve(e->args.size());
        for (const auto& a : e->args) lowered.push_back(lower_expr(a.get()));
        return lower_df32_math(n, lowered);
    }
    return std::nullopt;
}

} // namespace brokernel::detail

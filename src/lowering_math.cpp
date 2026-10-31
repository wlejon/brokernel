// The transcendentals, built from MIR arithmetic (add, mul, div, floor,
// select, sqrt and integer bit operations) rather than called: the CPU's libm
// and the GPU's GLSL.std.450 are different algorithms with different
// accuracy (a driver's f32 sin may be good to 2^-11 absolute), and there is
// no f64 GLSL.std.450 sin/exp/log at all. One algorithm lowered for both
// targets gives one answer, to within the rounding of a fused multiply-add
// the GPU compiler may form.
//
// f64: Cody-Waite reduction and fdlibm's minimax polynomials, within a few
// ulp. f32: the same reductions with Cephes' single-precision polynomials.

#include "lowering_internal.h"

#include <cmath>
#include <initializer_list>
#include <limits>

namespace brokernel::detail {

namespace {

struct Math {
    Ops& b;
    bool d; // f64 (else f32)

    brass::Value* c(double v) { return d ? b.build_fconst_f64(v) : b.build_fconst_f32(static_cast<float>(v)); }
    brass::Value* add(brass::Value* x, brass::Value* y) { return b.build_add(x, y); }
    brass::Value* sub(brass::Value* x, brass::Value* y) { return b.build_sub(x, y); }
    brass::Value* mul(brass::Value* x, brass::Value* y) { return b.build_mul(x, y); }
    brass::Value* div(brass::Value* x, brass::Value* y) { return b.build_sdiv(x, y); }
    brass::Value* neg(brass::Value* x) { return mul(x, c(-1.0)); }
    brass::Value* lt(brass::Value* x, brass::Value* y) { return b.build_slt(x, y); }
    brass::Value* gt(brass::Value* x, brass::Value* y) { return b.build_sgt(x, y); }
    brass::Value* sel(brass::Value* cond, brass::Value* t, brass::Value* f) { return b.build_select(cond, t, f); }
    brass::Value* abs(brass::Value* x) { return b.build_fabs(x); }
    brass::Value* floor(brass::Value* x) { return b.build_floor(x); }
    brass::Value* clamp(brass::Value* x, double lo, double hi) { return b.build_fmin(b.build_fmax(x, c(lo)), c(hi)); }
    brass::Value* inf() { return c(std::numeric_limits<double>::infinity()); }
    brass::Value* nan() { return c(std::numeric_limits<double>::quiet_NaN()); }

    // Coefficients highest degree first.
    brass::Value* horner(brass::Value* x, std::initializer_list<double> coeffs) {
        auto it = coeffs.begin();
        brass::Value* p = c(*it++);
        for (; it != coeffs.end(); ++it) p = add(mul(p, x), c(*it));
        return p;
    }

    // The sign/exponent bits: x's magnitude bits as an integer of x's width.
    brass::Value* magnitude_bits(brass::Value* x) {
        if (d) return b.build_and(b.build_bitcast_i64_f64(x), b.build_iconst_i64(0x7fffffffffffffffLL));
        return b.build_and(b.build_bitcast_i32_f32(x), b.build_iconst_i32(0x7fffffff));
    }
    brass::Value* is_nan(brass::Value* x) {
        if (d) return b.build_sgt(magnitude_bits(x), b.build_iconst_i64(0x7ff0000000000000LL));
        return b.build_sgt(magnitude_bits(x), b.build_iconst_i32(0x7f800000));
    }
    brass::Value* is_inf(brass::Value* x) {
        if (d) return b.build_eq(magnitude_bits(x), b.build_iconst_i64(0x7ff0000000000000LL));
        return b.build_eq(magnitude_bits(x), b.build_iconst_i32(0x7f800000));
    }

    // 2^k for an integral k in the normal exponent range.
    brass::Value* pow2_normal(brass::Value* k) {
        if (d) {
            brass::Value* e = b.build_add(b.build_fptosi_i64(k), b.build_iconst_i64(1023));
            return b.build_bitcast_f64_i64(b.build_shl(e, b.build_iconst_i64(52)));
        }
        brass::Value* e = b.build_add(b.build_fptosi_i32_f32(k), b.build_iconst_i32(127));
        return b.build_bitcast_f32_i32(b.build_shl(e, b.build_iconst_i32(23)));
    }
    // p * 2^k for |k| up to twice the exponent bias, as two in-range factors
    // (the result underflows gracefully to a subnormal).
    brass::Value* scale(brass::Value* p, brass::Value* k) {
        brass::Value* k1 = floor(mul(k, c(0.5)));
        return mul(mul(p, pow2_normal(k1)), pow2_normal(sub(k, k1)));
    }

    // ---- sin / cos / tan ------------------------------------------------

    // x = k pi/2 + r, |r| <= pi/4; q = k mod 4 (an exact float in 0..3).
    void reduce_half_pi(brass::Value* x, brass::Value*& r, brass::Value*& q) {
        brass::Value* k = floor(add(mul(x, c(0.63661977236758134308)), c(0.5)));
        if (d) {
            r = sub(x, mul(k, c(1.57079632673412561417e+00)));
            r = sub(r, mul(k, c(6.07710050650619224932e-11)));
            r = sub(r, mul(k, c(2.02226624879595063154e-21)));
        } else {
            r = sub(x, mul(k, c(1.5703125)));
            r = sub(r, mul(k, c(4.837512969970703125e-4)));
            r = sub(r, mul(k, c(7.54978995489188216e-8)));
        }
        q = sub(k, mul(c(4.0), floor(mul(k, c(0.25)))));
    }
    brass::Value* sin_poly(brass::Value* r) {
        brass::Value* z = mul(r, r);
        brass::Value* p = d ? horner(z, {1.58969099521155010221e-10, -2.50507602534068634195e-08,
                                         2.75573137070700676789e-06, -1.98412698298579493134e-04,
                                         8.33333333332248946124e-03, -1.66666666666666324348e-01})
                            : horner(z, {-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1});
        return add(r, mul(mul(r, z), p));
    }
    brass::Value* cos_poly(brass::Value* r) {
        brass::Value* z = mul(r, r);
        brass::Value* p = d ? horner(z, {-1.13596475577881948265e-11, 2.08757232129817482790e-09,
                                         -2.75573143513906633035e-07, 2.48015872894767294178e-05,
                                         -1.38888888888741095749e-03, 4.16666666666666019037e-02})
                            : horner(z, {2.443315711809948e-5, -1.388731625493765e-3, 4.166664568298827e-2});
        return add(sub(c(1.0), mul(c(0.5), z)), mul(mul(z, z), p));
    }
    // Picks by quadrant: q0, q1, q2, q3.
    brass::Value* by_quadrant(brass::Value* q, brass::Value* v0, brass::Value* v1, brass::Value* v2, brass::Value* v3) {
        return sel(lt(q, c(0.5)), v0, sel(lt(q, c(1.5)), v1, sel(lt(q, c(2.5)), v2, v3)));
    }
    brass::Value* sin(brass::Value* x) {
        brass::Value *r, *q;
        reduce_half_pi(x, r, q);
        brass::Value* s = sin_poly(r);
        brass::Value* co = cos_poly(r);
        return by_quadrant(q, s, co, neg(s), neg(co));
    }
    brass::Value* cos(brass::Value* x) {
        brass::Value *r, *q;
        reduce_half_pi(x, r, q);
        brass::Value* s = sin_poly(r);
        brass::Value* co = cos_poly(r);
        return by_quadrant(q, co, neg(s), neg(co), s);
    }
    brass::Value* tan(brass::Value* x) {
        brass::Value *r, *q;
        reduce_half_pi(x, r, q);
        brass::Value* t = div(sin_poly(r), cos_poly(r));
        brass::Value* cot = div(c(-1.0), t);
        return by_quadrant(q, t, cot, t, cot);
    }

    // ---- exp / log --------------------------------------------------------

    // e^r for |r| <= ln2/2 (Taylor; the f64 tail past r^13/13! is below 1e-17).
    brass::Value* exp_poly(brass::Value* r) {
        if (d) {
            return horner(r, {1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
                              1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0,
                              1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0});
        }
        return horner(r, {1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0});
    }
    // Overflow to +inf above `hi`, to 0 below `lo`, NaN through.
    brass::Value* exp_edges(brass::Value* x, brass::Value* v, double lo, double hi) {
        brass::Value* r = sel(gt(x, c(hi)), inf(), sel(lt(x, c(lo)), c(0.0), v));
        return sel(is_nan(x), x, r);
    }
    brass::Value* exp(brass::Value* x) {
        const double lo = d ? -745.2 : -104.0;
        const double hi = d ? 709.782712893384 : 88.72283905206835;
        brass::Value* xc = clamp(x, lo, hi);
        brass::Value* k = floor(add(mul(xc, c(1.44269504088896340736)), c(0.5)));
        brass::Value* r = d ? sub(sub(xc, mul(k, c(6.93147180369123816490e-01))), mul(k, c(1.90821492927058770002e-10)))
                            : sub(sub(xc, mul(k, c(0.693359375))), mul(k, c(-2.12194440e-4)));
        return exp_edges(x, scale(exp_poly(r), k), lo, hi);
    }
    brass::Value* exp2(brass::Value* x) {
        const double lo = d ? -1075.0 : -150.0;
        const double hi = d ? 1024.0 : 128.0;
        brass::Value* xc = clamp(x, lo, hi);
        brass::Value* k = floor(add(xc, c(0.5)));
        brass::Value* r = mul(sub(xc, k), c(0.69314718055994530942)); // x - k is exact
        brass::Value* v = scale(exp_poly(r), k);
        // exp2 of exactly hi overflows; of lo underflows: the edges are exclusive.
        return exp_edges(x, v, lo, hi - (d ? 1e-9 : 1e-4));
    }

    // x = m 2^e with m in [sqrt(1/2), sqrt(2)); returns ln m, sets e (float).
    brass::Value* log_mantissa(brass::Value* x, brass::Value*& e) {
        // Subnormals: scale into the normal range first.
        const double tiny = d ? 2.2250738585072014e-308 : 1.17549435e-38;
        const double up = d ? 18014398509481984.0 : 16777216.0; // 2^54 / 2^24
        brass::Value* small = lt(x, c(tiny));
        brass::Value* xs = sel(small, mul(x, c(up)), x);
        brass::Value* m;
        if (d) {
            brass::Value* bits = b.build_bitcast_i64_f64(xs);
            brass::Value* ebits = b.build_and(b.build_lshr(bits, b.build_iconst_i64(52)), b.build_iconst_i64(0x7ff));
            e = b.build_sitofp_f64_i64(b.build_sub(ebits, b.build_iconst_i64(1023)));
            brass::Value* mbits = b.build_or(b.build_and(bits, b.build_iconst_i64(0x000fffffffffffffLL)),
                                             b.build_iconst_i64(0x3ff0000000000000LL));
            m = b.build_bitcast_f64_i64(mbits);
        } else {
            brass::Value* bits = b.build_bitcast_i32_f32(xs);
            brass::Value* ebits = b.build_and(b.build_lshr(bits, b.build_iconst_i32(23)), b.build_iconst_i32(0xff));
            e = b.build_sitofp_f32_i32(b.build_sub(ebits, b.build_iconst_i32(127)));
            brass::Value* mbits = b.build_or(b.build_and(bits, b.build_iconst_i32(0x007fffff)),
                                             b.build_iconst_i32(0x3f800000));
            m = b.build_bitcast_f32_i32(mbits);
        }
        e = sub(e, sel(small, c(d ? 54.0 : 24.0), c(0.0)));
        brass::Value* hi = gt(m, c(1.41421356237309504880));
        m = sel(hi, mul(m, c(0.5)), m);
        e = add(e, sel(hi, c(1.0), c(0.0)));
        // ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.1716.
        brass::Value* s = div(sub(m, c(1.0)), add(m, c(1.0)));
        brass::Value* z = mul(s, s);
        brass::Value* p = d ? horner(z, {2.0 / 21.0, 2.0 / 19.0, 2.0 / 17.0, 2.0 / 15.0, 2.0 / 13.0, 2.0 / 11.0,
                                         2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0, 2.0})
                            : horner(z, {2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0, 2.0});
        return mul(s, p);
    }
    // log of 0 is -inf, of +inf +inf, of a negative (-inf too) NaN; NaN through.
    brass::Value* log_edges(brass::Value* x, brass::Value* v) {
        brass::Value* r = sel(gt(x, c(0.0)), v, neg(inf()));
        r = sel(is_inf(x), inf(), r);
        r = sel(lt(x, c(0.0)), nan(), r);
        return sel(is_nan(x), x, r);
    }
    brass::Value* log(brass::Value* x) {
        brass::Value* e;
        brass::Value* lm = log_mantissa(x, e);
        brass::Value* v = d ? add(mul(e, c(6.93147180369123816490e-01)), add(lm, mul(e, c(1.90821492927058770002e-10))))
                            : add(mul(e, c(0.693359375)), add(lm, mul(e, c(-2.12194440e-4))));
        return log_edges(x, v);
    }
    brass::Value* log2(brass::Value* x) {
        brass::Value* e;
        brass::Value* lm = log_mantissa(x, e);
        return log_edges(x, add(e, mul(lm, c(1.44269504088896340736))));
    }

    // ---- atan family --------------------------------------------------------

    // atan(t) for t in [0, 1].
    brass::Value* atan01(brass::Value* t) {
        // Above tan(pi/8): atan(t) = pi/4 + atan((t - 1) / (t + 1)).
        brass::Value* big = gt(t, c(0.41421356237309504880));
        brass::Value* u = sel(big, div(sub(t, c(1.0)), add(t, c(1.0))), t);
        brass::Value* z = mul(u, u);
        brass::Value* a;
        if (d) {
            brass::Value* p = horner(z, {1.62858201153657823623e-02, -3.65315727442169155270e-02,
                                         4.97687799461593236017e-02, -5.83357013379057348645e-02,
                                         6.66107313738753120669e-02, -7.69187620504482999495e-02,
                                         9.09088713343650656196e-02, -1.11111104054623557880e-01,
                                         1.42857142725034663711e-01, -1.99999999998764832476e-01,
                                         3.33333333333329318027e-01});
            a = sub(u, mul(mul(u, z), p));
        } else {
            brass::Value* p = horner(z, {8.05374449538e-2, -1.38776856032e-1, 1.99777106478e-1, -3.33329491539e-1});
            a = add(u, mul(mul(u, z), p));
        }
        return add(a, sel(big, c(0.78539816339744830962), c(0.0)));
    }
    // The sign bit, so -0 counts as negative.
    brass::Value* sign_bit(brass::Value* x) {
        if (d) return b.build_slt(b.build_bitcast_i64_f64(x), b.build_iconst_i64(0));
        return b.build_slt(b.build_bitcast_i32_f32(x), b.build_iconst_i32(0));
    }
    brass::Value* copysign_of(brass::Value* mag, brass::Value* sign_src) {
        return sel(sign_bit(sign_src), neg(mag), mag);
    }
    brass::Value* atan(brass::Value* x) {
        brass::Value* a = abs(x);
        brass::Value* inv = gt(a, c(1.0));
        brass::Value* t = sel(inv, div(c(1.0), a), a);
        brass::Value* r = atan01(t);
        r = sel(inv, sub(c(1.57079632679489661923), r), r);
        return copysign_of(r, x);
    }
    brass::Value* atan2(brass::Value* y, brass::Value* x) {
        brass::Value* ax = abs(x);
        brass::Value* ay = abs(y);
        brass::Value* mx = b.build_fmax(ax, ay);
        brass::Value* mn = b.build_fmin(ax, ay);
        brass::Value* t = sel(gt(mx, c(0.0)), div(mn, mx), c(0.0));
        // Both infinite: the ratio is 1 (a 45-degree multiple).
        t = sel(b.build_and(is_inf(ax), is_inf(ay)), c(1.0), t);
        brass::Value* r = atan01(t);
        r = sel(gt(ay, ax), sub(c(1.57079632679489661923), r), r);
        r = sel(sign_bit(x), sub(c(3.14159265358979323846), r), r);
        r = copysign_of(r, y);
        return sel(b.build_or(is_nan(x), is_nan(y)), add(x, y), r);
    }
    brass::Value* asin(brass::Value* x) {
        return atan2(x, b.build_sqrt(mul(sub(c(1.0), x), add(c(1.0), x))));
    }
    brass::Value* acos(brass::Value* x) {
        return atan2(b.build_sqrt(mul(sub(c(1.0), x), add(c(1.0), x))), x);
    }

    // ---- hyperbolic -----------------------------------------------------------

    // sinh for |x| <= 1/2 by its series (e^x - e^-x cancels there).
    brass::Value* sinh_small(brass::Value* x) {
        brass::Value* z = mul(x, x);
        brass::Value* p = d ? horner(z, {1.0 / 355687428096000.0, 1.0 / 1307674368000.0, 1.0 / 6227020800.0,
                                         1.0 / 39916800.0, 1.0 / 362880.0, 1.0 / 5040.0, 1.0 / 120.0, 1.0 / 6.0})
                            : horner(z, {1.0 / 362880.0, 1.0 / 5040.0, 1.0 / 120.0, 1.0 / 6.0});
        return add(x, mul(mul(x, z), p));
    }
    brass::Value* sinh(brass::Value* x) {
        brass::Value* ex = exp(x);
        brass::Value* big = mul(c(0.5), sub(ex, div(c(1.0), ex)));
        return sel(gt(abs(x), c(0.5)), big, sinh_small(x));
    }
    brass::Value* cosh(brass::Value* x) {
        brass::Value* ex = exp(abs(x));
        return mul(c(0.5), add(ex, div(c(1.0), ex)));
    }
    brass::Value* tanh(brass::Value* x) {
        // Past 22 (f64) / 10 (f32) tanh rounds to +-1.
        const double lim = d ? 22.0 : 10.0;
        brass::Value* xc = clamp(x, -lim, lim);
        brass::Value* e2 = exp(mul(c(2.0), abs(xc)));
        brass::Value* big = div(sub(e2, c(1.0)), add(e2, c(1.0)));
        brass::Value* s = sinh_small(xc);
        brass::Value* small = div(s, b.build_sqrt(add(c(1.0), mul(s, s))));
        brass::Value* r = sel(gt(abs(xc), c(0.5)), copysign_of(big, xc), small);
        return sel(is_nan(x), x, r);
    }

    // exp2(y log2 x), with C's values where that is undefined: pow(x, 0) and
    // pow(1, y) are 1, pow(0, y < 0) is +inf. A negative base is NaN (no
    // odd-integer exponent case, as in GLSL).
    brass::Value* pow(brass::Value* x, brass::Value* y) {
        brass::Value* r = exp2(mul(y, log2(x)));
        return sel(b.build_or(b.build_eq(y, c(0.0)), b.build_eq(x, c(1.0))), c(1.0), r);
    }
};

} // namespace

brass::Value* lower_transcendental_1(Ops& b, std::string_view name, brass::Value* x, ScalarKind sk) {
    if (sk != ScalarKind::F32 && sk != ScalarKind::F64) return nullptr;
    Math m{b, sk == ScalarKind::F64};
    if (name == "sin") return m.sin(x);
    if (name == "cos") return m.cos(x);
    if (name == "tan") return m.tan(x);
    if (name == "asin") return m.asin(x);
    if (name == "acos") return m.acos(x);
    if (name == "atan") return m.atan(x);
    if (name == "sinh") return m.sinh(x);
    if (name == "cosh") return m.cosh(x);
    if (name == "tanh") return m.tanh(x);
    if (name == "exp") return m.exp(x);
    if (name == "exp2") return m.exp2(x);
    if (name == "log") return m.log(x);
    if (name == "log2") return m.log2(x);
    return nullptr;
}

brass::Value* lower_transcendental_2(Ops& b, std::string_view name, brass::Value* x, brass::Value* y,
                                     ScalarKind sk) {
    if (sk != ScalarKind::F32 && sk != ScalarKind::F64) return nullptr;
    Math m{b, sk == ScalarKind::F64};
    if (name == "pow") return m.pow(x, y);
    if (name == "atan2") return m.atan2(x, y);
    return nullptr;
}

brass::Value* lower_float_class(Ops& b, std::string_view name, brass::Value* x, ScalarKind sk) {
    if (sk != ScalarKind::F32 && sk != ScalarKind::F64) return b.build_iconst_i32(0);
    Math m{b, sk == ScalarKind::F64};
    return name == "isnan" ? m.is_nan(x) : m.is_inf(x);
}

brass::Value* MirBodyLowerer::clamp_coord(brass::Value* c, brass::Value* extent) {
    brass::Value* zero = b.build_iconst_i32(0);
    brass::Value* one = b.build_iconst_i32(1);
    brass::Value* hi = b.build_sub(extent, one);
    brass::Value* c_lo = b.build_select(b.build_slt(c, zero), zero, c);
    return b.build_select(b.build_sgt(c_lo, hi), hi, c_lo);
}

ValueList MirBodyLowerer::at_cells(const ValueList& store, uint32_t dim) {
    auto it = store_param.find(store[0]);
    if (it == store_param.end()) return store;
    grid_cover[it->second] = std::max(grid_cover[it->second], dim);
    return {store[0]};
}

ValueList MirBodyLowerer::store_access(const ValueList& store, uint32_t dim, brass::Value* x, brass::Value* y,
                                       brass::Value* z) {
    if (!cell_coords_in_grid) return store;
    if (dim == 1 && linear_indices.count(x)) return at_cells(store, std::max(1u, root_fn.grid_dim));
    if (x != gx || (dim >= 2 && y != gy) || (dim >= 3 && z != gz)) return store;
    return at_cells(store, dim);
}

ValueList MirBodyLowerer::lower_stencil_read(const Expr* stencil, const Type& st, const std::vector<ExprPtr>& offsets) {
    ValueList base = lower_expr(stencil);
    auto offset = [&](size_t i) -> brass::Value* {
        return i < offsets.size() ? lower_expr(offsets[i].get())[0] : b.build_iconst_i32(0);
    };
    brass::Value* x = b.build_add(gx, offset(0));
    if (!x_interior) x = clamp_coord(x, gw);
    brass::Value* y = (st.dimension >= 2) ? clamp_coord(b.build_add(gy, offset(1)), gh) : b.build_iconst_i32(0);
    brass::Value* z = (st.dimension >= 3) ? clamp_coord(b.build_add(gz, offset(2)), gd) : b.build_iconst_i32(0);
    return load_element_from_buffer(st.element_type(), at_cells(base, st.dimension),
                                    compute_linear_index(st.dimension, x, y, z));
}

brass::Value* MirBodyLowerer::compute_linear_index(uint32_t dim, brass::Value* x, brass::Value* y, brass::Value* z) {
    auto to_i64 = [&](brass::Value* v) -> brass::Value* {
        return (v->type() == brass::Type::i64()) ? v : b.build_sext_i64(v);
    };
    brass::Value* x64 = to_i64(x);
    if (dim == 1) return x64;
    brass::Value* y64 = to_i64(y);
    brass::Value* gw64 = to_i64(gw);
    if (dim == 2) return b.build_add(b.build_mul(y64, gw64), x64);
    brass::Value* z64 = to_i64(z);
    brass::Value* gh64 = to_i64(gh);
    brass::Value* zy = b.build_add(b.build_mul(z64, gh64), y64);
    return b.build_add(b.build_mul(zy, gw64), x64);
}

ValueList MirBodyLowerer::load_element_from_buffer(const Type& elem_type, const ValueList& store, brass::Value* linear_idx) {
    brass::Value* base_ptr = store[0];
    brass::Value* idx64 = (linear_idx->type() == brass::Type::i64()) ? linear_idx : b.build_sext_i64(linear_idx);
    const Ops::Bound bound_v{idx64, store.size() > 1 ? store[1] : nullptr};
    const Ops::Bound* bound = bound_v.count ? &bound_v : nullptr;
    auto kinds = mod.types.flatten_scalars(elem_type);
    ValueList res;
    res.reserve(kinds.size());
    if (kinds.size() == 1) {
        brass::Type mt = to_mir_scalar_type(kinds[0]);
        brass::Value* v = b.build_load_indexed(mt, base_ptr, idx64, static_cast<uint8_t>(mt.size_in_bytes()), 0, bound);
        if (kinds[0] == ScalarKind::Bool && b.vector_mode()) v = b.build_ne(v, b.build_iconst_i32(0));
        res.push_back(v);
        return res;
    }
    bool uniform = true;
    for (size_t i = 1; i < kinds.size(); ++i) {
        if (kinds[i] != kinds[0]) { uniform = false; break; }
    }
    if (uniform) {
        brass::Type mt = to_mir_scalar_type(kinds[0]);
        uint8_t scale = static_cast<uint8_t>(mt.size_in_bytes());
        brass::Value* base_elem = b.build_mul(idx64, b.build_iconst_i64(static_cast<int64_t>(kinds.size())));
        for (size_t i = 0; i < kinds.size(); ++i) {
            brass::Value* idx_i = (i == 0) ? base_elem : b.build_add(base_elem, b.build_iconst_i64(static_cast<int64_t>(i)));
            brass::Value* v = b.build_load_indexed(mt, base_ptr, idx_i, scale, 0, bound);
            if (kinds[i] == ScalarKind::Bool && b.vector_mode()) v = b.build_ne(v, b.build_iconst_i32(0));
            res.push_back(v);
        }
        return res;
    }
    uint32_t elem_bytes = mod.types.byte_size(elem_type);
    brass::Value* byte_base = b.build_mul(idx64, b.build_iconst_i64(static_cast<int64_t>(elem_bytes)));
    uint32_t cur_off = 0;
    for (ScalarKind sk : kinds) {
        brass::Type mt = to_mir_scalar_type(sk);
        brass::Value* v = b.build_load_indexed(mt, base_ptr, byte_base, 1, static_cast<int32_t>(cur_off), bound);
        if (sk == ScalarKind::Bool && b.vector_mode()) v = b.build_ne(v, b.build_iconst_i32(0));
        res.push_back(v);
        cur_off += static_cast<uint32_t>(mt.size_in_bytes());
    }
    return res;
}

void MirBodyLowerer::store_element_to_buffer(const Type& elem_type, const ValueList& store, brass::Value* linear_idx,
                                             const ValueList& vals) {
    brass::Value* base_ptr = store[0];
    brass::Value* idx64 = (linear_idx->type() == brass::Type::i64()) ? linear_idx : b.build_sext_i64(linear_idx);
    const Ops::Bound bound_v{idx64, store.size() > 1 ? store[1] : nullptr};
    const Ops::Bound* bound = bound_v.count ? &bound_v : nullptr;
    auto kinds = mod.types.flatten_scalars(elem_type);
    auto norm_bool = [&](ScalarKind sk, brass::Value* v) -> brass::Value* {
        return (sk == ScalarKind::Bool && b.vector_mode()) ? b.build_and(v, b.build_iconst_i32(1)) : v;
    };
    if (kinds.size() == 1 && !vals.empty()) {
        brass::Type mt = to_mir_scalar_type(kinds[0]);
        b.build_store_indexed(mt, base_ptr, idx64, static_cast<uint8_t>(mt.size_in_bytes()), 0, norm_bool(kinds[0], vals[0]), bound);
        return;
    }
    bool uniform = true;
    for (size_t i = 1; i < kinds.size(); ++i) {
        if (kinds[i] != kinds[0]) { uniform = false; break; }
    }
    if (uniform) {
        brass::Type mt = to_mir_scalar_type(kinds[0]);
        uint8_t scale = static_cast<uint8_t>(mt.size_in_bytes());
        brass::Value* base_elem = b.build_mul(idx64, b.build_iconst_i64(static_cast<int64_t>(kinds.size())));
        for (size_t i = 0; i < kinds.size() && i < vals.size(); ++i) {
            brass::Value* idx_i = (i == 0) ? base_elem : b.build_add(base_elem, b.build_iconst_i64(static_cast<int64_t>(i)));
            b.build_store_indexed(mt, base_ptr, idx_i, scale, 0, norm_bool(kinds[i], vals[i]), bound);
        }
        return;
    }
    uint32_t elem_bytes = mod.types.byte_size(elem_type);
    brass::Value* byte_base = b.build_mul(idx64, b.build_iconst_i64(static_cast<int64_t>(elem_bytes)));
    uint32_t cur_off = 0;
    for (size_t i = 0; i < kinds.size() && i < vals.size(); ++i) {
        brass::Type mt = to_mir_scalar_type(kinds[i]);
        b.build_store_indexed(mt, base_ptr, byte_base, 1, static_cast<int32_t>(cur_off), norm_bool(kinds[i], vals[i]), bound);
        cur_off += static_cast<uint32_t>(mt.size_in_bytes());
    }
}

brass::Value* MirBodyLowerer::lower_scalar_cast(brass::Value* v, ScalarKind from_sk, ScalarKind to_sk) {
    if (from_sk == to_sk) return v;
    if (to_sk == ScalarKind::Bool) {
        if (from_sk == ScalarKind::F32) return b.build_ne(v, b.build_fconst_f32(0.0f));
        if (from_sk == ScalarKind::F64) return b.build_ne(v, b.build_fconst_f64(0.0));
        if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_ne(v, b.build_iconst_i64(0));
        return b.build_ne(v, b.build_iconst_i32(0));
    }
    if (from_sk == ScalarKind::Bool && b.vector_mode()) {
        v = b.build_and(v, b.build_iconst_i32(1));
    }
    if (to_sk == ScalarKind::F32) {
        if (from_sk == ScalarKind::F64) return b.build_fptrunc_f32_f64(v);
        if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_sitofp_f32_i64(v);
        if (from_sk == ScalarKind::U32) {
            return (is_gpu && !b.vector_mode()) ? b.build_call("ptx_u32_to_f32", brass::Type::f32(), {v})
                                                : b.build_sitofp_f32_i64(b.build_zext_i64(v));
        }
        return b.build_sitofp_f32_i32(v);
    }
    if (to_sk == ScalarKind::F64) {
        if (from_sk == ScalarKind::F32) return b.build_fpext_f64_f32(v);
        if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_sitofp_f64_i64(v);
        if (from_sk == ScalarKind::U32) return b.build_sitofp_f64_i64(b.build_zext_i64(v));
        return b.build_sitofp_f64_i32(v);
    }
    if (to_sk == ScalarKind::I64 || to_sk == ScalarKind::U64) {
        if (from_sk == ScalarKind::F32) return b.build_fptosi_i64_f32(v);
        if (from_sk == ScalarKind::F64) return b.build_fptosi_i64(v);
        if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return v;
        return (from_sk == ScalarKind::U32 || from_sk == ScalarKind::Bool) ? b.build_zext_i64(v) : b.build_sext_i64(v);
    }
    if (from_sk == ScalarKind::F32) {
        if (to_sk == ScalarKind::U32) {
            return (is_gpu && !b.vector_mode()) ? b.build_call("ptx_f32_to_u32", brass::Type::i32(), {v})
                                                : b.build_trunc_i32(b.build_fptosi_i64_f32(v));
        }
        return b.build_fptosi_i32_f32(v);
    }
    if (from_sk == ScalarKind::F64) {
        return to_sk == ScalarKind::U32 ? b.build_trunc_i32(b.build_fptosi_i64(v)) : b.build_fptosi_i32(v);
    }
    if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_trunc_i32(v);
    return v;
}

brass::Value* MirBodyLowerer::lower_scalar_binary(BinaryOp op, brass::Value* l, brass::Value* r, ScalarKind sk) {
    bool is_unsigned = (sk == ScalarKind::U32 || sk == ScalarKind::U64);
    bool is_fp = (sk == ScalarKind::F32 || sk == ScalarKind::F64);
    switch (op) {
        case BinaryOp::Add: return b.build_add(l, r);
        case BinaryOp::Sub: return b.build_sub(l, r);
        case BinaryOp::Mul: return b.build_mul(l, r);
        case BinaryOp::Div:
            if (is_fp) return b.build_sdiv(l, r);
            return is_unsigned ? b.build_udiv(l, r) : b.build_sdiv(l, r);
        case BinaryOp::Mod:
            if (is_fp) {
                brass::Value* q = b.build_floor(b.build_sdiv(l, r));
                return b.build_sub(l, b.build_mul(r, q));
            }
            return is_unsigned ? b.build_umod(l, r) : b.build_smod(l, r);
        case BinaryOp::BitAnd:
        case BinaryOp::LogAnd: return b.build_and(l, r);
        case BinaryOp::BitOr:
        case BinaryOp::LogOr:  return b.build_or(l, r);
        case BinaryOp::BitXor: return b.build_xor(l, r);
        case BinaryOp::Shl:    return b.build_shl(l, r);
        case BinaryOp::Shr:    return is_unsigned ? b.build_lshr(l, r) : b.build_ashr(l, r);
        case BinaryOp::Eq:     return b.build_eq(l, r);
        case BinaryOp::Ne:     return b.build_ne(l, r);
        case BinaryOp::Lt:     return (!is_fp && is_unsigned) ? b.build_ult(l, r) : b.build_slt(l, r);
        case BinaryOp::Le:     return (!is_fp && is_unsigned) ? b.build_ule(l, r) : b.build_sle(l, r);
        case BinaryOp::Gt:     return (!is_fp && is_unsigned) ? b.build_ugt(l, r) : b.build_sgt(l, r);
        case BinaryOp::Ge:     return (!is_fp && is_unsigned) ? b.build_uge(l, r) : b.build_sge(l, r);
    }
    return l;
}

brass::Value* MirBodyLowerer::lower_native_math_1(const std::string& name, brass::Value* x) {
    static const char* const kNative[][2] = {
        {"sin", "sinf"}, {"cos", "cosf"}, {"tan", "tanf"}, {"asin", "asinf"}, {"acos", "acosf"},
        {"atan", "atanf"}, {"sinh", "sinhf"}, {"cosh", "coshf"}, {"tanh", "tanhf"},
        {"exp", "expf"}, {"exp2", "ex2f"}, {"log", "logf"}, {"log2", "lg2f"},
    };
    for (const auto& n : kNative) {
        if (name == n[0]) return b.build_call(n[1], brass::Type::f32(), {x});
    }
    return nullptr;
}

brass::Value* MirBodyLowerer::lower_scalar_math_1(const std::string& name, brass::Value* x, ScalarKind sk) {
    bool is_fp = (sk == ScalarKind::F32 || sk == ScalarKind::F64);
    brass::Value* zero = (sk == ScalarKind::F64) ? b.build_fconst_f64(0.0)
                         : (sk == ScalarKind::F32) ? b.build_fconst_f32(0.0f)
                         : (sk == ScalarKind::I64 || sk == ScalarKind::U64) ? b.build_iconst_i64(0)
                                                                            : b.build_iconst_i32(0);
    brass::Value* one = (sk == ScalarKind::F64) ? b.build_fconst_f64(1.0)
                        : (sk == ScalarKind::F32) ? b.build_fconst_f32(1.0f)
                        : (sk == ScalarKind::I64 || sk == ScalarKind::U64) ? b.build_iconst_i64(1)
                                                                           : b.build_iconst_i32(1);
    if (name == "abs") {
        if (is_fp) return b.build_fabs(x);
        return b.build_select(b.build_slt(x, zero), b.build_neg(x), x);
    }
    if (name == "sign") {
        brass::Value* neg_one = b.build_neg(one);
        return b.build_select(b.build_sgt(x, zero), one, b.build_select(b.build_slt(x, zero), neg_one, zero));
    }
    if (name == "sqrt") return b.build_sqrt(x);
    if (name == "rsqrt" || name == "inversesqrt") return b.build_sdiv(one, b.build_sqrt(x));
    if (name == "floor") return b.build_floor(x);
    if (name == "ceil") return b.build_ceil(x);
    if (name == "round") return b.build_round(x);
    if (name == "trunc") return b.build_select(b.build_sge(x, zero), b.build_floor(x), b.build_ceil(x));
    if (name == "fract") return b.build_sub(x, b.build_floor(x));
    if (name == "saturate") return b.build_fmin(b.build_fmax(x, zero), one);
    if (name == "degrees") {
        constexpr double kRadToDeg = 57.2957795130823208768;
        brass::Value* c = (sk == ScalarKind::F64) ? b.build_fconst_f64(kRadToDeg) : b.build_fconst_f32(static_cast<float>(kRadToDeg));
        return b.build_mul(x, c);
    }
    if (name == "radians") {
        constexpr double kDegToRad = 0.017453292519943295769;
        brass::Value* c = (sk == ScalarKind::F64) ? b.build_fconst_f64(kDegToRad) : b.build_fconst_f32(static_cast<float>(kDegToRad));
        return b.build_mul(x, c);
    }
    if (is_gpu && root_fn.fast_math && sk == ScalarKind::F32) {
        if (brass::Value* t = lower_native_math_1(name, x)) return t;
    }
    if (brass::Value* t = lower_transcendental_1(b, name, x, sk)) return t;
    if (name == "dpdx" || name == "dpdy" || name == "fwidth") {
        return b.build_call("spirv_" + name + "_f32", brass::Type::f32(), {x});
    }
    if (name == "subgroup_broadcast_first" || name == "subgroup_add" || name == "subgroup_min" ||
        name == "subgroup_max") {
        if (!is_gpu) return x;
        return b.build_call("spirv_" + name + (sk == ScalarKind::F32 ? "_f32" : "_i32"), to_mir_scalar_type(sk), {x});
    }
    if (name == "subgroup_all" || name == "subgroup_any") {
        if (!is_gpu) return b.build_ne(x, b.build_iconst_i32(0));
        return b.build_call("spirv_" + name, brass::Type::i32(), {x});
    }
    if (name == "isnan" || name == "isinf") return lower_float_class(b, name, x, sk);
    diags.error(SourceSpan{}, "internal: no lowering for builtin '" + name + "'");
    return x;
}

brass::Value* MirBodyLowerer::lower_scalar_math_2(const std::string& name, brass::Value* a, brass::Value* c, ScalarKind sk) {
    bool is_fp = (sk == ScalarKind::F32 || sk == ScalarKind::F64);
    bool is_unsigned = (sk == ScalarKind::U32 || sk == ScalarKind::U64);
    if (name == "min") {
        if (is_fp) return b.build_fmin(a, c);
        return b.build_select(is_unsigned ? b.build_ult(a, c) : b.build_slt(a, c), a, c);
    }
    if (name == "max") {
        if (is_fp) return b.build_fmax(a, c);
        return b.build_select(is_unsigned ? b.build_ugt(a, c) : b.build_sgt(a, c), a, c);
    }
    if (is_gpu && root_fn.fast_math && sk == ScalarKind::F32 && (name == "pow" || name == "atan2")) {
        return b.build_call(name == "pow" ? "powf" : "atan2f", brass::Type::f32(), {a, c});
    }
    if (brass::Value* t = lower_transcendental_2(b, name, a, c, sk)) return t;
    if (name == "step") {
        brass::Value* zero = (sk == ScalarKind::F64) ? b.build_fconst_f64(0.0) : b.build_fconst_f32(0.0f);
        brass::Value* one = (sk == ScalarKind::F64) ? b.build_fconst_f64(1.0) : b.build_fconst_f32(1.0f);
        return b.build_select(b.build_slt(c, a), zero, one);
    }
    if (name == "mod") {
        brass::Value* q = b.build_floor(b.build_sdiv(a, c));
        return b.build_sub(a, b.build_mul(c, q));
    }
    if (name == "fmod") {
        if (!is_fp) return is_unsigned ? b.build_umod(a, c) : b.build_smod(a, c);
        brass::Value* zero = (sk == ScalarKind::F64) ? b.build_fconst_f64(0.0) : b.build_fconst_f32(0.0f);
        brass::Value* q = b.build_sdiv(a, c);
        q = b.build_select(b.build_sge(q, zero), b.build_floor(q), b.build_ceil(q));
        return b.build_sub(a, b.build_mul(c, q));
    }
    diags.error(SourceSpan{}, "internal: no lowering for builtin '" + name + "'");
    return a;
}

brass::Value* MirBodyLowerer::lower_scalar_math_3(const std::string& name, brass::Value* x, brass::Value* y, brass::Value* z, ScalarKind sk) {
    bool is_fp = (sk == ScalarKind::F32 || sk == ScalarKind::F64);
    if (name == "clamp") {
        if (is_fp) return b.build_fmin(b.build_fmax(x, y), z);
        bool is_unsigned = (sk == ScalarKind::U32 || sk == ScalarKind::U64);
        brass::Value* lo = b.build_select(is_unsigned ? b.build_ult(x, y) : b.build_slt(x, y), y, x);
        return b.build_select(is_unsigned ? b.build_ugt(lo, z) : b.build_sgt(lo, z), z, lo);
    }
    if (name == "mix" || name == "lerp") {
        return b.build_add(x, b.build_mul(b.build_sub(y, x), z));
    }
    if (name == "fma") {
        if (is_fp) return b.build_fma(x, y, z);
        return b.build_add(b.build_mul(x, y), z);
    }
    if (name == "smoothstep") {
        brass::Value* zero = (sk == ScalarKind::F64) ? b.build_fconst_f64(0.0) : b.build_fconst_f32(0.0f);
        brass::Value* one = (sk == ScalarKind::F64) ? b.build_fconst_f64(1.0) : b.build_fconst_f32(1.0f);
        brass::Value* two = (sk == ScalarKind::F64) ? b.build_fconst_f64(2.0) : b.build_fconst_f32(2.0f);
        brass::Value* three = (sk == ScalarKind::F64) ? b.build_fconst_f64(3.0) : b.build_fconst_f32(3.0f);
        brass::Value* raw = b.build_sdiv(b.build_sub(z, x), b.build_sub(y, x));
        brass::Value* t = b.build_fmin(b.build_fmax(raw, zero), one);
        brass::Value* tt = b.build_mul(t, t);
        return b.build_mul(tt, b.build_sub(three, b.build_mul(two, t)));
    }
    diags.error(SourceSpan{}, "internal: no lowering for builtin '" + name + "'");
    return x;
}

} // namespace brokernel::detail

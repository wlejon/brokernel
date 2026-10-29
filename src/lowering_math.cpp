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

} // namespace brokernel::detail

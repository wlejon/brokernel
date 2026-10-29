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
brass::Value* MirBodyLowerer::clamp_coord(brass::Value* c, brass::Value* extent) {
        brass::Value* zero = b.build_iconst_i32(0);
        brass::Value* one = b.build_iconst_i32(1);
        brass::Value* hi = b.build_sub(extent, one);
        brass::Value* c_lo = b.build_select(b.build_slt(c, zero), zero, c);
        return b.build_select(b.build_sgt(c_lo, hi), hi, c_lo);
    }
brass::Value* MirBodyLowerer::compute_linear_index(uint32_t dim, brass::Value* x, brass::Value* y, brass::Value* z) {
        if (dim == 1) return x;
        if (dim == 2) {
            return b.build_add(b.build_mul(y, gw), x);
        }
        brass::Value* zy = b.build_add(b.build_mul(z, gh), y);
        return b.build_add(b.build_mul(zy, gw), x);
    }
ValueList MirBodyLowerer::load_element_from_buffer(const Type& elem_type, brass::Value* base_ptr, brass::Value* linear_idx) {
        auto kinds = mod.types.flatten_scalars(elem_type);
        ValueList res;
        res.reserve(kinds.size());
        if (kinds.size() == 1) {
            brass::Type mt = to_mir_scalar_type(kinds[0]);
            res.push_back(b.build_load_indexed(mt, base_ptr, linear_idx, static_cast<uint8_t>(mt.size_in_bytes())));
            return res;
        }
        // Check if all leaves have identical scalar kind
        bool uniform = true;
        for (size_t i = 1; i < kinds.size(); ++i) {
            if (kinds[i] != kinds[0]) { uniform = false; break; }
        }
        if (uniform) {
            brass::Type mt = to_mir_scalar_type(kinds[0]);
            uint8_t scale = static_cast<uint8_t>(mt.size_in_bytes());
            brass::Value* base_elem = b.build_mul(linear_idx, b.build_iconst_i32(static_cast<int32_t>(kinds.size())));
            for (size_t i = 0; i < kinds.size(); ++i) {
                brass::Value* idx_i = (i == 0) ? base_elem : b.build_add(base_elem, b.build_iconst_i32(static_cast<int32_t>(i)));
                res.push_back(b.build_load_indexed(mt, base_ptr, idx_i, scale));
            }
            return res;
        }
        // Heterogeneous struct: compute byte offset = linear_idx * elem_byte_size + leaf_off
        uint32_t elem_bytes = mod.types.byte_size(elem_type);
        brass::Value* byte_base = b.build_mul(linear_idx, b.build_iconst_i32(static_cast<int32_t>(elem_bytes)));
        uint32_t cur_off = 0;
        for (ScalarKind sk : kinds) {
            brass::Type mt = to_mir_scalar_type(sk);
            res.push_back(b.build_load_indexed(mt, base_ptr, byte_base, 1, static_cast<int32_t>(cur_off)));
            cur_off += static_cast<uint32_t>(mt.size_in_bytes());
        }
        return res;
    }
void MirBodyLowerer::store_element_to_buffer(const Type& elem_type, brass::Value* base_ptr, brass::Value* linear_idx,
                                 const ValueList& vals) {
        auto kinds = mod.types.flatten_scalars(elem_type);
        if (kinds.size() == 1 && !vals.empty()) {
            brass::Type mt = to_mir_scalar_type(kinds[0]);
            b.build_store_indexed(mt, base_ptr, linear_idx, static_cast<uint8_t>(mt.size_in_bytes()), vals[0]);
            return;
        }
        bool uniform = true;
        for (size_t i = 1; i < kinds.size(); ++i) {
            if (kinds[i] != kinds[0]) { uniform = false; break; }
        }
        if (uniform) {
            brass::Type mt = to_mir_scalar_type(kinds[0]);
            uint8_t scale = static_cast<uint8_t>(mt.size_in_bytes());
            brass::Value* base_elem = b.build_mul(linear_idx, b.build_iconst_i32(static_cast<int32_t>(kinds.size())));
            for (size_t i = 0; i < kinds.size() && i < vals.size(); ++i) {
                brass::Value* idx_i = (i == 0) ? base_elem : b.build_add(base_elem, b.build_iconst_i32(static_cast<int32_t>(i)));
                b.build_store_indexed(mt, base_ptr, idx_i, scale, vals[i]);
            }
            return;
        }
        uint32_t elem_bytes = mod.types.byte_size(elem_type);
        brass::Value* byte_base = b.build_mul(linear_idx, b.build_iconst_i32(static_cast<int32_t>(elem_bytes)));
        uint32_t cur_off = 0;
        for (size_t i = 0; i < kinds.size() && i < vals.size(); ++i) {
            brass::Type mt = to_mir_scalar_type(kinds[i]);
            b.build_store_indexed(mt, base_ptr, byte_base, 1, static_cast<int32_t>(cur_off), vals[i]);
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
        if (to_sk == ScalarKind::F32) {
            if (from_sk == ScalarKind::F64) return b.build_fptrunc_f32_f64(v);
            if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_sitofp_f32_i64(v);
            return b.build_sitofp_f32_i32(v);
        }
        if (to_sk == ScalarKind::F64) {
            if (from_sk == ScalarKind::F32) return b.build_fpext_f64_f32(v);
            if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return b.build_sitofp_f64_i64(v);
            return b.build_sitofp_f64_i32(v);
        }
        if (to_sk == ScalarKind::I64 || to_sk == ScalarKind::U64) {
            if (from_sk == ScalarKind::F32) return b.build_fptosi_i64_f32(v);
            if (from_sk == ScalarKind::F64) return b.build_fptosi_i64(v);
            if (from_sk == ScalarKind::I64 || from_sk == ScalarKind::U64) return v;
            return (from_sk == ScalarKind::U32 || from_sk == ScalarKind::Bool) ? b.build_zext_i64(v) : b.build_sext_i64(v);
        }
        // Target is I32 or U32
        if (from_sk == ScalarKind::F32) return b.build_fptosi_i32_f32(v);
        if (from_sk == ScalarKind::F64) return b.build_fptosi_i32(v);
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
                    // x - y * floor(x / y) for well-defined GLSL/SPIR-V/CPU parity
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
        if (name == "trunc") {
            return b.build_select(b.build_sge(x, zero), b.build_floor(x), b.build_ceil(x));
        }
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
        if (brass::Value* t = lower_transcendental_1(b, name, x, sk)) return t;
        if (name == "dpdx" || name == "dpdy" || name == "fwidth") {
            // Fragment-only (sema), and fragments have no CPU build.
            return b.build_call("spirv_" + name + "_f32", brass::Type::f32(), {x});
        }
        if (name == "subgroup_broadcast_first" || name == "subgroup_add" || name == "subgroup_min" ||
            name == "subgroup_max") {
            // The CPU's subgroup is one invocation: every reduction is the value itself.
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
            // C fmod: x - y trunc(x / y), the sign of x.
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
ValueList MirBodyLowerer::lower_inlined_call(
    const CheckedFunction& callee,
    const std::vector<ValueList>& arg_vals) {
    // Save outer return tracking and scopes
    int saved_has_ret = has_returned_var_id;
    int saved_ret_val = ret_val_var_id;
    Type saved_ret_type = active_return_type;
    auto saved_scopes = std::move(scopes);
    scopes.clear();
    push_scope();
    active_return_type = callee.return_type;
    has_returned_var_id = alloc_var("__callee_has_ret", {b.build_iconst_i32(0)});
    if (!callee.return_type.is_void()) {
        ret_val_var_id = alloc_var("__callee_ret_val", make_zero_leaves(callee.return_type));
    } else {
        ret_val_var_id = -1;
    }
    for (size_t i = 0; i < callee.params.size() && i < arg_vals.size(); ++i) {
        alloc_var(callee.params[i].name, arg_vals[i]);
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
    has_returned_var_id = saved_has_ret;
    ret_val_var_id = saved_ret_val;
    active_return_type = saved_ret_type;
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
            return {b.build_iconst_i32(e->bool_val ? 1 : 0)};
        case Expr::Kind::Ident: {
            int id = lookup_var_id(e->name);
            if (id >= 0) return var_values[id];
            return make_zero_leaves(e->resolved_type);
        }
        case Expr::Kind::Path:
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
            ValueList lv = lower_expr(e->lhs.get());
            ValueList rv = lower_expr(e->rhs.get());
            brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
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
            const Type& rt = e->lhs->resolved_type;
            if (rt.is_stencil()) {
                ValueList base = lower_expr(e->lhs.get());
                ValueList dx = e->args.size() >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList dy = e->args.size() >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList dz = e->args.size() >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                brass::Value* x = clamp_coord(b.build_add(gx, dx[0]), gw);
                brass::Value* y = (rt.dimension >= 2) ? clamp_coord(b.build_add(gy, dy[0]), gh) : b.build_iconst_i32(0);
                brass::Value* z = (rt.dimension >= 3) ? clamp_coord(b.build_add(gz, dz[0]), gd) : b.build_iconst_i32(0);
                brass::Value* lin = compute_linear_index(rt.dimension, x, y, z);
                return load_element_from_buffer(rt.element_type(), base[0], lin);
            }
            if (rt.is_store()) {
                ValueList base = lower_expr(e->lhs.get());
                ValueList x = e->args.size() >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList y = e->args.size() >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList z = e->args.size() >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                brass::Value* lin = compute_linear_index(rt.dimension, x[0], y[0], z[0]);
                return load_element_from_buffer(rt.element_type(), base[0], lin);
            }
            // Vector or array indexing
            ValueList recv = lower_expr(e->lhs.get());
            if (rt.is_scalar() || recv.size() <= 1) return recv;
            ValueList idx_v = lower_expr(e->args[0].get());
            Type elem_t = rt.element_type();
            uint32_t elem_leaves = mod.types.flat_leaf_count(elem_t);
            uint32_t count = rt.dimension;
            ValueList res(elem_leaves, recv[0]);
            for (uint32_t k = 0; k < elem_leaves; ++k) {
                brass::Value* acc = recv[k];
                for (uint32_t i = 1; i < count; ++i) {
                    brass::Value* eq = b.build_eq(idx_v[0], b.build_iconst_i32(static_cast<int32_t>(i)));
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
            if (rt.is_stencil()) {
                ValueList base = lower_expr(e->lhs.get());
                ValueList dx = e->args.size() >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList dy = e->args.size() >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList dz = e->args.size() >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                brass::Value* x = clamp_coord(b.build_add(gx, dx[0]), gw);
                brass::Value* y = (rt.dimension >= 2) ? clamp_coord(b.build_add(gy, dy[0]), gh) : b.build_iconst_i32(0);
                brass::Value* z = (rt.dimension >= 3) ? clamp_coord(b.build_add(gz, dz[0]), gd) : b.build_iconst_i32(0);
                brass::Value* lin = compute_linear_index(rt.dimension, x, y, z);
                return load_element_from_buffer(rt.element_type(), base[0], lin);
            }
            if (rt.is_store()) {
                ValueList base = lower_expr(e->lhs.get());
                uint32_t dim = rt.dimension;
                Type elem_t = rt.element_type();
                if (m == "read") {
                    brass::Value* lin = compute_linear_index(dim, gx, gy, gz);
                    return load_element_from_buffer(elem_t, base[0], lin);
                }
                if (m == "write") {
                    ValueList val = lower_expr(e->args[0].get());
                    brass::Value* lin = compute_linear_index(dim, gx, gy, gz);
                    store_element_to_buffer(elem_t, base[0], lin, val);
                    return {};
                }
                if (m == "load") {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    return load_element_from_buffer(elem_t, base[0], lin);
                }
                if (m == "store") {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList val = lower_expr(e->args[dim].get());
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    store_element_to_buffer(elem_t, base[0], lin, val);
                    return {};
                }
                if (m == "atomic_add" || m == "atomic_min" || m == "atomic_max" || m == "atomic_exchange") {
                    ValueList x = dim >= 1 ? lower_expr(e->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList y = dim >= 2 ? lower_expr(e->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList z = dim >= 3 ? lower_expr(e->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                    ValueList val = lower_expr(e->args[dim].get());
                    brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                    if (is_gpu) {
                        uint32_t esz = elem_t.scalar_size_bytes();
                        brass::Value* byte_off = b.build_mul(b.build_sext_i64(lin), b.build_iconst_i64(esz));
                        brass::Value* addr = b.build_add(base[0], byte_off);
                        if (m == "atomic_add") {
                            return {elem_t.is_f32() ? kb.atom_add_f32(addr, val[0]) : kb.atom_add_i32(addr, val[0])};
                        }
                        if (m == "atomic_min") return {b.build_call("ptx_atom_min_i32", brass::Type::i32(), {addr, val[0]})};
                        if (m == "atomic_max") return {b.build_call("ptx_atom_max_i32", brass::Type::i32(), {addr, val[0]})};
                        return {b.build_call("ptx_atom_exch_i32", brass::Type::i32(), {addr, val[0]})};
                    } else {
                        // Sequential CPU evaluation of atomic op
                        ValueList old_v = load_element_from_buffer(elem_t, base[0], lin);
                        brass::Value* new_v = val[0];
                        if (m == "atomic_add") new_v = b.build_add(old_v[0], val[0]);
                        else if (m == "atomic_min") new_v = lower_scalar_math_2("min", old_v[0], val[0], elem_t.scalar);
                        else if (m == "atomic_max") new_v = lower_scalar_math_2("max", old_v[0], val[0], elem_t.scalar);
                        store_element_to_buffer(elem_t, base[0], lin, {new_v});
                        return old_v;
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
                arg_vals.reserve(e->args.size());
                for (const auto& a : e->args) {
                    arg_vals.push_back(lower_expr(a.get()));
                }
                return lower_inlined_call(*callee, arg_vals);
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
            // 4. Vector / Array / Struct constructors
            if (n == "vec2" || n == "vec3" || n == "vec4" ||
                n == "vec2f" || n == "vec3f" || n == "vec4f" ||
                n == "vec2d" || n == "vec3d" || n == "vec4d" ||
                n == "vec2i" || n == "vec3i" || n == "vec4i" ||
                n == "vec2u" || n == "vec3u" || n == "vec4u" ||
                n == "vec" || n == "array" || e->resolved_type.is_struct()) {
                uint32_t target_leaves = mod.types.flat_leaf_count(e->resolved_type);
                ValueList flat;
                for (const auto& a : e->args) {
                    ValueList av = lower_expr(a.get());
                    for (brass::Value* v : av) flat.push_back(v);
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
            if (n == "thread_id_x" || n == "local_id_x") return {is_gpu ? kb.tid_x() : b.build_iconst_i32(0)};
            if (n == "thread_id_y" || n == "local_id_y") return {is_gpu ? kb.tid_y() : b.build_iconst_i32(0)};
            if (n == "thread_id_z" || n == "local_id_z") return {is_gpu ? kb.tid_z() : b.build_iconst_i32(0)};
            if (n == "group_id_x") return {is_gpu ? kb.ctaid_x() : b.build_iconst_i32(0)};
            if (n == "group_id_y") return {is_gpu ? kb.ctaid_y() : b.build_iconst_i32(0)};
            if (n == "group_id_z") return {is_gpu ? kb.ctaid_z() : b.build_iconst_i32(0)};
            if (n == "global_id_x") return {gx ? gx : (is_gpu ? kb.global_tid_x() : b.build_iconst_i32(0))};
            if (n == "global_id_y") return {gy ? gy : b.build_iconst_i32(0)};
            if (n == "global_id_z") return {gz ? gz : b.build_iconst_i32(0)};
            if (n == "workgroup_size_x") return {is_gpu ? kb.ntid_x() : b.build_iconst_i32(1)};
            if (n == "workgroup_size_y") return {is_gpu ? kb.ntid_y() : b.build_iconst_i32(1)};
            if (n == "workgroup_size_z") return {is_gpu ? kb.ntid_z() : b.build_iconst_i32(1)};
            if (n == "subgroup_size") return {is_gpu ? b.build_call("spirv_subgroup_size", brass::Type::i32()) : b.build_iconst_i32(1)};
            if (n == "subgroup_invocation_id") return {is_gpu ? b.build_call("spirv_subgroup_local_id", brass::Type::i32()) : b.build_iconst_i32(0)};
            if (n == "workgroup_barrier" || n == "storage_barrier" || n == "barrier") {
                if (is_gpu) kb.sync();
                return {};
            }
            // Vertex & Fragment intrinsics
            if (n == "vertex_index") return {is_gpu ? b.build_call("spirv_vertex_index", brass::Type::i32()) : gx};
            if (n == "instance_index") return {is_gpu ? b.build_call("spirv_instance_index", brass::Type::i32()) : gy};
            if (n == "vertex_input" || n == "frag_input") {
                ValueList loc = lower_expr(e->args[0].get());
                if (is_gpu) {
                    return {
                        b.build_call("spirv_stage_input_f32", brass::Type::f32(), {loc[0], b.build_iconst_i32(0)}),
                        b.build_call("spirv_stage_input_f32", brass::Type::f32(), {loc[0], b.build_iconst_i32(1)}),
                        b.build_call("spirv_stage_input_f32", brass::Type::f32(), {loc[0], b.build_iconst_i32(2)}),
                        b.build_call("spirv_stage_input_f32", brass::Type::f32(), {loc[0], b.build_iconst_i32(3)}),
                    };
                }
                return make_zero_leaves(e->resolved_type);
            }
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
                if (is_gpu && v.size() >= 4) {
                    b.build_call("spirv_stage_output_vec4", brass::Type::void_type(), {loc[0], v[0], v[1], v[2], v[3]});
                }
                return {};
            }
            if (n == "set_frag_color") {
                ValueList v = lower_expr(e->args[0].get());
                if (is_gpu && v.size() >= 4) {
                    b.build_call("spirv_stage_output_vec4", brass::Type::void_type(),
                                 {b.build_iconst_i32(0), v[0], v[1], v[2], v[3]});
                }
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
                // Demote drops the fragment's outputs; the rest of the body is skipped like a return.
                if (is_gpu) b.build_call("spirv_demote_to_helper", brass::Type::void_type());
                if (has_returned_var_id >= 0) {
                    var_values[has_returned_var_id] = {b.build_iconst_i32(1)};
                }
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

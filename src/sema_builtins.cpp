#include "sema_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
namespace brokernel::detail {
Type FnCheckState::check_method_call(Expr* e) {
    Type recv_t = check_expr(e->lhs.get());
    const std::string& m = e->name;
    if (recv_t.is_grid()) {
        uint32_t dim = recv_t.dimension;
        ensure_grid_dim(dim);
        if (m == "x" || m == "y" || m == "z" || m == "width" || m == "height" || m == "depth") {
            e->resolved_type = Type::i32();
            return Type::i32();
        }
        if (m == "linear_index" || m == "total_cells") {
            e->resolved_type = Type::i64();
            return Type::i64();
        }
        if (m == "coord" || m == "pos") {
            Type rt = (dim == 1) ? Type::i32() : Type::vec(ScalarKind::I32, dim);
            e->resolved_type = rt;
            return rt;
        }
        if (m == "uv") {
            Type rt = (dim == 1) ? Type::f32() : Type::vec(ScalarKind::F32, dim);
            e->resolved_type = rt;
            return rt;
        }
        if (m == "uv_f64") {
            fn.uses_f64 = true;
            Type rt = (dim == 1) ? Type::f64() : Type::vec(ScalarKind::F64, dim);
            e->resolved_type = rt;
            return rt;
        }
        ctx.diags.error(e->span, "unknown grid method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_stencil()) {
        LocalVar* lv = (e->lhs->kind == Expr::Kind::Ident) ? lookup_var(e->lhs->name) : nullptr;
        CheckedParam* cparam = (lv && lv->param_index >= 0) ? &fn.params[lv->param_index] : nullptr;
        if (m == "radius") {
            int32_t r = cparam && cparam->declared_radius >= 0 ? cparam->declared_radius : recv_t.stencil_radius;
            e->kind = Expr::Kind::IntLit;
            e->int_val = std::max(0, r);
            e->resolved_type = Type::i32();
            return Type::i32();
        }
        if (m == "write" || m == "store") {
            ctx.diags.error(e->span, "cannot write to read-only stencil");
            return Type::void_type();
        }
        if (m == "read" || m == "center") {
            if (!e->args.empty()) {
                ctx.diags.error(e->span, "stencil.read() takes 0 arguments");
            }
            e->args.clear();
            for (uint32_t d = 0; d < recv_t.dimension; ++d) {
                auto zero = std::make_unique<Expr>();
                zero->kind = Expr::Kind::IntLit;
                zero->span = e->span;
                zero->int_val = 0;
                zero->resolved_type = Type::i32();
                e->args.push_back(std::move(zero));
            }
            e->name = "at";
        }
        if (e->name == "at") {
            if (cparam) {
                validate_stencil_access(*cparam, e->args, e->span);
            }
            Type elem = recv_t.element_type();
            mark_type_usage(elem);
            e->resolved_type = elem;
            return elem;
        }
        ctx.diags.error(e->span, "unknown stencil method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_store()) {
        uint32_t dim = recv_t.dimension;
        Type elem = recv_t.element_type();
        mark_type_usage(elem);
        if (m == "read") {
            if (!can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "cannot read from write-only store");
            }
            ensure_grid_dim(dim);
            if (!e->args.empty()) {
                ctx.diags.error(e->span, "store.read() takes 0 arguments (use store.load(...) for explicit coordinates)");
            }
            e->resolved_type = elem;
            return elem;
        }
        if (m == "write") {
            if (!can_Write(recv_t.access)) {
                ctx.diags.error(e->span, "cannot write to read-only store");
            }
            ensure_grid_dim(dim);
            if (e->args.size() != 1) {
                ctx.diags.error(e->span, "store.write(value) expects 1 argument");
            } else {
                check_expr(e->args[0].get(), elem);
                check_assignable(e->args[0].get(), elem, ctx.diags, e->args[0]->span);
            }
            e->resolved_type = Type::void_type();
            return Type::void_type();
        }
        if (m == "load") {
            if (!can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "cannot read from write-only store");
            }
            if (e->args.size() != dim) {
                ctx.diags.error(e->span, "store.load expects " + std::to_string(dim) + " coordinate argument(s)");
            }
            for (auto& a : e->args) {
                Type at = check_expr(a.get(), Type::i32());
                if (!at.is_integer() || !at.is_scalar()) {
                    ctx.diags.error(a->span, "store.load coordinate must be a scalar integer, got '" + at.to_string() + "'");
                }
            }
            e->resolved_type = elem;
            return elem;
        }
        if (m == "store") {
            if (!can_Write(recv_t.access)) {
                ctx.diags.error(e->span, "cannot write to read-only store");
            }
            if (e->args.size() != dim + 1) {
                ctx.diags.error(e->span, "store.store expects " + std::to_string(dim) + " coordinate argument(s) and 1 value argument");
            } else {
                for (size_t i = 0; i < dim; ++i) {
                    Type at = check_expr(e->args[i].get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(e->args[i]->span, "store.store coordinate must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                check_expr(e->args[dim].get(), elem);
                check_assignable(e->args[dim].get(), elem, ctx.diags, e->args[dim]->span);
            }
            e->resolved_type = Type::void_type();
            return Type::void_type();
        }
        if (m == "atomic_add" || m == "atomic_min" || m == "atomic_max" || m == "atomic_exchange") {
            if (!can_Write(recv_t.access) || !can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "atomic store operation requires a read-write store ('rw_store' or 'store')");
            }
            if (e->args.size() != dim + 1) {
                ctx.diags.error(e->span, m + " expects " + std::to_string(dim) + " coordinate argument(s) and 1 value argument");
            } else {
                for (size_t i = 0; i < dim; ++i) {
                    Type at = check_expr(e->args[i].get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(e->args[i]->span, m + " coordinate must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                check_expr(e->args[dim].get(), elem);
                check_assignable(e->args[dim].get(), elem, ctx.diags, e->args[dim]->span);
            }
            e->resolved_type = elem;
            return elem;
        }
        ctx.diags.error(e->span, "unknown store method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_image()) {
        Type elem = recv_t.element_type();
        for (auto& a : e->args) check_expr(a.get());
        if (m == "load") {
            if (!can_Read(recv_t.access)) ctx.diags.error(e->span, "cannot load from write-only image");
            e->resolved_type = elem;
            return elem;
        }
        if (m == "store") {
            if (!can_Write(recv_t.access)) ctx.diags.error(e->span, "cannot store to read-only image");
            e->resolved_type = Type::void_type();
            return Type::void_type();
        }
    }
    if (recv_t.is_sampler()) {
        Type elem = recv_t.element_type();
        for (auto& a : e->args) check_expr(a.get());
        if (m == "sample" || m == "sample_lod") {
            e->resolved_type = elem;
            return elem;
        }
    }
    ctx.diags.error(e->span, "type '" + recv_t.to_string() + "' has no method '" + m + "'");
    return Type::void_type();
}
static void harmonize_args(std::vector<ExprPtr>& args, std::optional<Type> expected) {
    // Determine target type among concrete (non-unsuffixed) arguments or expected
    std::optional<Type> concrete;
    for (const auto& a : args) {
        if (!is_unsuffixed_literal(a.get()) && !a->resolved_type.is_void()) {
            if (!concrete || a->resolved_type.is_vector()) {
                concrete = a->resolved_type;
            }
        }
    }
    if (!concrete && expected && !expected->is_void()) {
        concrete = *expected;
    }
    if (concrete) {
        for (auto& a : args) {
            coerce_expr_to_type(a.get(), *concrete);
        }
    }
}
Type FnCheckState::check_builtin_call(Expr* e, std::optional<Type> expected) {
    const std::string& n = e->name;
    // 1. Scalar constructors / explicit casts: f32(x), f64(x), i32(x), u32(x), i64(x), u64(x), bool(x)
    if ((n == "f32" || n == "f64" || n == "i32" || n == "u32" || n == "i64" || n == "u64" || n == "bool") &&
        e->generic_args.empty()) {
        TypeExpr te{e->span, n, {}};
        Type target = ctx.resolve_type(&te, fn.module_name, genv);
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "type conversion '" + n + "(x)' expects 1 argument");
            return target;
        }
        check_expr(e->args[0].get(), target);
        mark_type_usage(target);
        e->resolved_type = target;
        return target;
    }
    if (n == "is_f64" || n == "sizeof") {
        if (auto cv = ctx.eval_const_expr(e, fn.module_name, genv, scopes)) {
            if (cv->type.is_bool()) {
                e->kind = Expr::Kind::BoolLit;
                e->bool_val = cv->bool_val;
            } else {
                e->kind = Expr::Kind::IntLit;
                e->int_val = cv->int_val;
            }
            e->generic_args.clear();
            e->args.clear();
            e->resolved_type = cv->type;
            return cv->type;
        }
        ctx.diags.error(e->span, "'" + n + "<T>()' expects 1 type argument and 0 value arguments");
        return Type::bool_type();
    }
    if (n == "bitcast") {
        if (e->generic_args.size() != 1 || e->args.size() != 1) {
            ctx.diags.error(e->span, "'bitcast<T>(x)' expects 1 type argument and 1 value argument");
            return Type::i32();
        }
        Type target = Type::i32();
        const auto& ga = e->generic_args[0];
        if (ga.type_arg) {
            target = ctx.resolve_type(ga.type_arg.get(), fn.module_name, genv);
        } else if (!ga.ident_or_const_name.empty()) {
            TypeExpr tmp{ga.span, ga.ident_or_const_name, {}};
            target = ctx.resolve_type(&tmp, fn.module_name, genv);
        }
        Type src_t = check_expr(e->args[0].get());
        auto bit_width = [](ScalarKind sk) -> uint32_t {
            switch (sk) {
                case ScalarKind::F32:
                case ScalarKind::I32:
                case ScalarKind::U32:
                    return 32;
                case ScalarKind::F64:
                case ScalarKind::I64:
                case ScalarKind::U64:
                    return 64;
                default:
                    return 0;
            }
        };
        if (target.kind != src_t.kind || target.dimension != src_t.dimension ||
            bit_width(target.scalar) == 0 || bit_width(target.scalar) != bit_width(src_t.scalar)) {
            ctx.diags.error(e->span, "cannot bitcast '" + src_t.to_string() + "' to '" + target.to_string() + "'");
        }
        mark_type_usage(target);
        e->resolved_type = target;
        return target;
    }
    // 2. Vector / Array constructors: vec2, vec3, vec4, vec2f, vec3d, vec<T, D>, array<T, N>
    if (n == "vec2" || n == "vec3" || n == "vec4" ||
        n == "vec2f" || n == "vec3f" || n == "vec4f" ||
        n == "vec2d" || n == "vec3d" || n == "vec4d" ||
        n == "vec2i" || n == "vec3i" || n == "vec4i" ||
        n == "vec2u" || n == "vec3u" || n == "vec4u" ||
        n == "vec" || n == "array") {
        for (auto& a : e->args) check_expr(a.get());
        Type target;
        if (n == "vec" || n == "array" || !e->generic_args.empty()) {
            TypeExpr te;
            te.span = e->span;
            te.name = n;
            for (const auto& ga : e->generic_args) {
                GenericArg copy;
                copy.span = ga.span;
                copy.type_arg = clone_type_expr(ga.type_arg.get());
                copy.int_arg = ga.int_arg;
                copy.ident_or_const_name = ga.ident_or_const_name;
                te.args.push_back(std::move(copy));
            }
            target = ctx.resolve_type(&te, fn.module_name, genv);
        } else if (n == "vec2" || n == "vec3" || n == "vec4") {
            uint32_t lanes = static_cast<uint32_t>(n.back() - '0');
            ScalarKind sk = ScalarKind::F32;
            for (const auto& a : e->args) {
                if (!is_unsuffixed_literal(a.get()) && !a->resolved_type.is_void()) {
                    sk = a->resolved_type.scalar;
                    break;
                }
            }
            if (expected && expected->is_vector() && expected->dimension == lanes) {
                sk = expected->scalar;
            }
            target = Type::vec(sk, lanes);
        } else {
            TypeExpr te{e->span, n, {}};
            target = ctx.resolve_type(&te, fn.module_name, genv);
        }
        Type elem_t = target.element_type();
        for (auto& a : e->args) {
            if (a->resolved_type.is_scalar()) {
                coerce_expr_to_type(a.get(), elem_t);
            }
        }
        mark_type_usage(target);
        e->resolved_type = target;
        return target;
    }
    // 3. Grid / Thread / Stage built-in queries
    if (n == "grid_x" || n == "grid_width") {
        ensure_grid_dim(1);
        e->resolved_type = Type::i32();
        return Type::i32();
    }
    if (n == "grid_y" || n == "grid_height") {
        ensure_grid_dim(2);
        e->resolved_type = Type::i32();
        return Type::i32();
    }
    if (n == "grid_z" || n == "grid_depth") {
        ensure_grid_dim(3);
        e->resolved_type = Type::i32();
        return Type::i32();
    }
    if (n == "grid_linear_index" || n == "grid_total_cells") {
        uint32_t d = std::max(1u, fn.grid_dim);
        ensure_grid_dim(d);
        e->resolved_type = Type::i64();
        return Type::i64();
    }
    if (n == "grid_coord" || n == "grid_pos") {
        uint32_t d = std::max(1u, fn.grid_dim);
        ensure_grid_dim(d);
        Type rt = (d == 1) ? Type::i32() : Type::vec(ScalarKind::I32, d);
        e->resolved_type = rt;
        return rt;
    }
    if (n == "grid_uv") {
        uint32_t d = std::max(1u, fn.grid_dim);
        ensure_grid_dim(d);
        Type rt = (d == 1) ? Type::f32() : Type::vec(ScalarKind::F32, d);
        e->resolved_type = rt;
        return rt;
    }
    if (n == "grid_uv_f64") {
        uint32_t d = std::max(1u, fn.grid_dim);
        ensure_grid_dim(d);
        fn.uses_f64 = true;
        Type rt = (d == 1) ? Type::f64() : Type::vec(ScalarKind::F64, d);
        e->resolved_type = rt;
        return rt;
    }
    if (n == "thread_id_x" || n == "thread_id_y" || n == "thread_id_z" ||
        n == "local_id_x" || n == "local_id_y" || n == "local_id_z" ||
        n == "group_id_x" || n == "group_id_y" || n == "group_id_z" ||
        n == "global_id_x" || n == "global_id_y" || n == "global_id_z" ||
        n == "workgroup_size_x" || n == "workgroup_size_y" || n == "workgroup_size_z" ||
        n == "subgroup_size" || n == "subgroup_invocation_id") {
        e->resolved_type = Type::i32();
        return Type::i32();
    }
    if (n == "workgroup_barrier" || n == "storage_barrier" || n == "barrier") {
        if (fn.stage != Stage::Compute) {
            ctx.diags.error(e->span, "'" + n + "()' is only valid in compute kernels");
        }
        fn.uses_barriers_or_shared = true;
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    // Vertex stage intrinsics
    if (n == "vertex_index" || n == "instance_index") {
        if (fn.stage != Stage::Vertex) {
            ctx.diags.error(e->span, "'" + n + "()' is only valid in @vertex shaders");
        }
        e->resolved_type = Type::i32();
        return Type::i32();
    }
    if (n == "set_vertex_position") {
        if (fn.stage != Stage::Vertex) {
            ctx.diags.error(e->span, "'set_vertex_position' is only valid in @vertex shaders");
        }
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'set_vertex_position' expects a vec4<f32> argument");
        } else {
            Type v4 = Type::vec(ScalarKind::F32, 4);
            check_expr(e->args[0].get(), v4);
            check_assignable(e->args[0].get(), v4, ctx.diags, e->args[0]->span);
        }
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    if (n == "set_vertex_output") {
        if (fn.stage != Stage::Vertex) {
            ctx.diags.error(e->span, "'set_vertex_output' is only valid in @vertex shaders");
        }
        if (e->args.size() == 2) {
            check_expr(e->args[0].get(), Type::i32());
            Type v4 = Type::vec(ScalarKind::F32, 4);
            check_expr(e->args[1].get(), v4);
            check_assignable(e->args[1].get(), v4, ctx.diags, e->args[1]->span);
        } else {
            ctx.diags.error(e->span, "'set_vertex_output(loc, val)' expects 2 arguments");
        }
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    // Fragment stage intrinsics
    if (n == "frag_coord") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'frag_coord()' is only valid in @fragment shaders");
        }
        Type v4 = Type::vec(ScalarKind::F32, 4);
        e->resolved_type = v4;
        return v4;
    }
    if (n == "front_facing") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'front_facing()' is only valid in @fragment shaders");
        }
        e->resolved_type = Type::bool_type();
        return Type::bool_type();
    }
    if (n == "set_frag_color") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'set_frag_color' is only valid in @fragment shaders");
        }
        if (e->args.size() == 1) {
            Type v4 = Type::vec(ScalarKind::F32, 4);
            check_expr(e->args[0].get(), v4);
            check_assignable(e->args[0].get(), v4, ctx.diags, e->args[0]->span);
        } else {
            ctx.diags.error(e->span, "'set_frag_color' expects 1 vec4<f32> argument");
        }
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    if (n == "set_frag_output") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'set_frag_output' is only valid in @fragment shaders");
        }
        if (e->args.size() == 2) {
            check_expr(e->args[0].get(), Type::i32());
            Type v4 = Type::vec(ScalarKind::F32, 4);
            check_expr(e->args[1].get(), v4);
            check_assignable(e->args[1].get(), v4, ctx.diags, e->args[1]->span);
        } else {
            ctx.diags.error(e->span, "'set_frag_output(loc, val)' expects 2 arguments");
        }
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    if (n == "discard_fragment") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'discard_fragment()' is only valid in @fragment shaders");
        }
        e->resolved_type = Type::void_type();
        return Type::void_type();
    }
    if (n == "dpdx" || n == "dpdy" || n == "fwidth") {
        if (fn.stage != Stage::Fragment) {
            ctx.diags.error(e->span, "'" + n + "()' is only valid in @fragment shaders");
        }
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'" + n + "' expects 1 argument");
            return Type::f32();
        }
        Type t = check_expr(e->args[0].get(), expected);
        if (t.scalar != ScalarKind::F32 || !(t.is_scalar() || t.is_vector())) {
            ctx.diags.error(e->span, "'" + n + "' takes f32 or an f32 vector, not '" + t.to_string() + "'");
        }
        e->resolved_type = t;
        return t;
    }
    if (n == "subgroup_broadcast_first" || n == "subgroup_add" || n == "subgroup_min" || n == "subgroup_max") {
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'" + n + "' expects 1 argument");
            return Type::f32();
        }
        Type t = check_expr(e->args[0].get(), expected);
        if (!(t.is_f32() || t.is_i32() || t.is_u32()) || (n != "subgroup_broadcast_first" && t.is_u32())) {
            ctx.diags.error(e->span, "'" + n + "' takes " +
                                     (n == "subgroup_broadcast_first" ? "f32, i32 or u32" : "f32 or i32") +
                                     ", not '" + t.to_string() + "'");
        }
        e->resolved_type = t;
        return t;
    }
    // 4. Math and vector built-in functions
    if (n == "abs" || n == "sign" || n == "sqrt" || n == "rsqrt" || n == "inversesqrt" ||
        n == "sin" || n == "cos" || n == "tan" || n == "asin" || n == "acos" || n == "atan" ||
        n == "sinh" || n == "cosh" || n == "tanh" || n == "exp" || n == "exp2" ||
        n == "log" || n == "log2" || n == "floor" || n == "ceil" || n == "round" ||
        n == "trunc" || n == "fract" || n == "saturate" || n == "degrees" || n == "radians" ||
        n == "normalize") {
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'" + n + "' expects 1 argument");
            return Type::f32();
        }
        Type t = check_expr(e->args[0].get(), expected);
        mark_type_usage(t);
        e->resolved_type = t;
        return t;
    }
    if (n == "min" || n == "max" || n == "pow" || n == "atan2" || n == "step" ||
        n == "mod" || n == "fmod" || n == "reflect") {
        if (e->args.size() != 2) {
            ctx.diags.error(e->span, "'" + n + "' expects 2 arguments");
            return Type::f32();
        }
        for (auto& a : e->args) check_expr(a.get());
        harmonize_args(e->args, expected);
        Type t0 = e->args[0]->resolved_type;
        Type t1 = e->args[1]->resolved_type;
        if (t0 != t1 && !coerce_expr_to_type(e->args[0].get(), t1) && !coerce_expr_to_type(e->args[1].get(), t0)) {
            ctx.diags.error(e->span, "mismatched argument types for '" + n + "': '" +
                                     t0.to_string() + "' and '" + t1.to_string() + "'");
        }
        Type res = e->args[0]->resolved_type;
        mark_type_usage(res);
        e->resolved_type = res;
        return res;
    }
    if (n == "refract") {
        // refract(I, N, eta): I and N vectors of one type, eta their scalar.
        if (e->args.size() != 3) {
            ctx.diags.error(e->span, "'refract(I, N, eta)' expects 3 arguments");
            return Type::vec(ScalarKind::F32, 3);
        }
        check_expr(e->args[0].get(), expected);
        check_expr(e->args[1].get(), e->args[0]->resolved_type);
        Type t = e->args[0]->resolved_type;
        if (!t.is_vector() || !t.is_float() ||
            (e->args[1]->resolved_type != t && !coerce_expr_to_type(e->args[1].get(), t))) {
            ctx.diags.error(e->span, "'refract' requires two float vectors of one type");
        }
        Type eta = Type::scalar_type(t.scalar);
        check_expr(e->args[2].get(), eta);
        if (e->args[2]->resolved_type != eta && !coerce_expr_to_type(e->args[2].get(), eta)) {
            ctx.diags.error(e->args[2]->span, "'refract' eta must be '" + eta.to_string() + "'");
        }
        mark_type_usage(t);
        e->resolved_type = t;
        return t;
    }
    if (n == "clamp" || n == "mix" || n == "lerp" || n == "smoothstep" || n == "fma" || n == "faceforward") {
        if (e->args.size() != 3) {
            ctx.diags.error(e->span, "'" + n + "' expects 3 arguments");
            return Type::f32();
        }
        for (auto& a : e->args) check_expr(a.get());
        harmonize_args(e->args, expected);
        Type res = e->args[0]->resolved_type.is_vector() ? e->args[0]->resolved_type : e->args[1]->resolved_type;
        for (auto& a : e->args) {
            if (a->resolved_type != res && !coerce_expr_to_type(a.get(), res)) {
                ctx.diags.error(a->span, "argument type '" + a->resolved_type.to_string() +
                                         "' does not match '" + res.to_string() + "' in '" + n + "'");
            }
        }
        mark_type_usage(res);
        e->resolved_type = res;
        return res;
    }
    if (n == "select") {
        if (e->args.size() != 3) {
            ctx.diags.error(e->span, "'select(cond, a, b)' expects 3 arguments");
            return Type::f32();
        }
        check_expr(e->args[0].get(), Type::bool_type());
        check_expr(e->args[1].get(), expected);
        check_expr(e->args[2].get(), expected);
        if (!coerce_expr_to_type(e->args[2].get(), e->args[1]->resolved_type)) {
            coerce_expr_to_type(e->args[1].get(), e->args[2]->resolved_type);
        }
        Type res = e->args[1]->resolved_type;
        mark_type_usage(res);
        e->resolved_type = res;
        return res;
    }
    if (n == "dot" || n == "distance") {
        if (e->args.size() != 2) {
            ctx.diags.error(e->span, "'" + n + "' expects 2 arguments");
            return Type::f32();
        }
        check_expr(e->args[0].get());
        check_expr(e->args[1].get());
        harmonize_args(e->args, std::nullopt);
        if (e->args[0]->resolved_type != e->args[1]->resolved_type) {
            ctx.diags.error(e->span, "'" + n + "' requires matching operand types");
        }
        Type res = Type::scalar_type(e->args[0]->resolved_type.scalar);
        mark_type_usage(res);
        e->resolved_type = res;
        return res;
    }
    if (n == "length") {
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'length' expects 1 argument");
            return Type::f32();
        }
        Type t = check_expr(e->args[0].get());
        Type res = Type::scalar_type(t.scalar);
        mark_type_usage(res);
        e->resolved_type = res;
        return res;
    }
    if (n == "cross") {
        if (e->args.size() != 2) {
            ctx.diags.error(e->span, "'cross' expects 2 vec3 arguments");
            return Type::vec(ScalarKind::F32, 3);
        }
        check_expr(e->args[0].get());
        check_expr(e->args[1].get());
        harmonize_args(e->args, expected);
        Type t = e->args[0]->resolved_type;
        if (!t.is_vector() || t.dimension != 3 || e->args[1]->resolved_type != t) {
            ctx.diags.error(e->span, "'cross' requires two vec3 arguments of identical element type");
        }
        mark_type_usage(t);
        e->resolved_type = t;
        return t;
    }
    if (n == "isnan" || n == "isinf" || n == "subgroup_all" || n == "subgroup_any") {
        if (e->args.size() != 1) {
            ctx.diags.error(e->span, "'" + n + "' expects 1 argument");
            return Type::bool_type();
        }
        Type t = check_expr(e->args[0].get());
        bool float_test = (n == "isnan" || n == "isinf");
        if (float_test ? !t.is_float_scalar() : !t.is_bool()) {
            ctx.diags.error(e->span, "'" + n + "' takes " + (float_test ? "an f32 or f64" : "a bool") +
                                     ", not '" + t.to_string() + "'");
        }
        e->resolved_type = Type::bool_type();
        return Type::bool_type();
    }
    // 5. Struct constructor syntax `StructName(field0, field1, ...)`
    std::string st_mod;
    if (const StructDecl* sdecl = ctx.find_struct_decl(fn.module_name, n, &st_mod)) {
        TypeExpr te;
        te.span = e->span;
        te.name = n;
        for (const auto& ga : e->generic_args) {
            GenericArg copy;
            copy.span = ga.span;
            copy.type_arg = clone_type_expr(ga.type_arg.get());
            copy.int_arg = ga.int_arg;
            copy.ident_or_const_name = ga.ident_or_const_name;
            te.args.push_back(std::move(copy));
        }
        Type st_type = ctx.resolve_type(&te, fn.module_name, genv);
        const StructDef* sdef = ctx.out->types.find_struct(st_type.struct_id);
        if (sdef) {
            if (e->args.size() != sdef->fields.size()) {
                ctx.diags.error(e->span, "struct '" + sdef->name + "' constructor expects " +
                                         std::to_string(sdef->fields.size()) + " argument(s)");
            } else {
                for (size_t i = 0; i < e->args.size(); ++i) {
                    check_expr(e->args[i].get(), sdef->fields[i].type);
                    check_assignable(e->args[i].get(), sdef->fields[i].type, ctx.diags, e->args[i]->span);
                }
            }
        }
        mark_type_usage(st_type);
        e->resolved_type = st_type;
        return st_type;
    }
    ctx.diags.error(e->span, "unknown function '" + n + "'");
    return Type::void_type();
}
bool is_unsuffixed_literal(const Expr* e) {
    if (!e) return false;
    if (e->kind == Expr::Kind::IntLit || e->kind == Expr::Kind::FloatLit) {
        return e->literal_suffix == ScalarKind::Void;
    }
    if (e->kind == Expr::Kind::Unary &&
        (e->unary_op == UnaryOp::Neg || e->unary_op == UnaryOp::Pos)) {
        return is_unsuffixed_literal(e->lhs.get());
    }
    return false;
}
static bool is_unsuffixed_float(const Expr* e) {
    if (!e) return false;
    if (e->kind == Expr::Kind::FloatLit) return e->literal_suffix == ScalarKind::Void;
    if (e->kind == Expr::Kind::Unary &&
        (e->unary_op == UnaryOp::Neg || e->unary_op == UnaryOp::Pos)) {
        return is_unsuffixed_float(e->lhs.get());
    }
    return false;
}
static bool is_unsuffixed_int(const Expr* e) {
    if (!e) return false;
    if (e->kind == Expr::Kind::IntLit) return e->literal_suffix == ScalarKind::Void;
    if (e->kind == Expr::Kind::Unary &&
        (e->unary_op == UnaryOp::Neg || e->unary_op == UnaryOp::Pos)) {
        return is_unsuffixed_int(e->lhs.get());
    }
    return false;
}
static void retype_literal(Expr* e, const Type& t) {
    if (!e) return;
    e->resolved_type = t;
    if (e->kind == Expr::Kind::Unary && e->lhs) {
        retype_literal(e->lhs.get(), t);
    }
}
bool coerce_expr_to_type(Expr* e, const Type& target_type) {
    if (!e) return false;
    if (e->resolved_type == target_type) return true;
    // Unsuffixed float literal adapts to f32 or f64
    if (is_unsuffixed_float(e) && target_type.is_float_scalar()) {
        retype_literal(e, target_type);
        return true;
    }
    // Unsuffixed int literal adapts to any integer scalar (or float scalar if constant)
    if (is_unsuffixed_int(e) && target_type.is_integer() && target_type.is_scalar()) {
        retype_literal(e, target_type);
        return true;
    }
    // Scalar broadcast to vector of matching scalar type (or unsuffixed literal to vector)
    if (target_type.is_vector()) {
        Type elem_target = Type::scalar_type(target_type.scalar);
        if (e->resolved_type == elem_target || coerce_expr_to_type(e, elem_target)) {
            auto wrapped = std::make_unique<Expr>();
            wrapped->kind = Expr::Kind::Call;
            wrapped->span = e->span;
            wrapped->name = "__splat";
            wrapped->resolved_type = target_type;
            wrapped->args.push_back(clone_expr(e));
            *e = std::move(*wrapped);
            return true;
        }
    }
    return false;
}
bool check_assignable(Expr* expr, const Type& to, DiagnosticBag& diags, const SourceSpan& span) {
    if (!expr) return false;
    if (coerce_expr_to_type(expr, to)) return true;
    const Type& from = expr->resolved_type;
    if (from.is_void() || to.is_void()) return false;
    if (from.is_f64() && to.is_f32()) {
        diags.error(span, "implicit narrowing conversion from 'f64' to 'f32' is forbidden; use an explicit cast ('as f32' or 'f32(...)')");
        return false;
    }
    if (from.is_f32() && to.is_f64()) {
        diags.error(span, "implicit precision conversion from 'f32' to 'f64' is forbidden; use an explicit cast ('as f64' or 'f64(...)')");
        return false;
    }
    diags.error(span, "type mismatch: expected '" + to.to_string() + "', got '" + from.to_string() + "'");
    return false;
}
// ---------------------------------------------------------------------------
// Function Body Checker
// ---------------------------------------------------------------------------
bool FnCheckState::validate_stencil_access(
    CheckedParam& param,
    const std::vector<ExprPtr>& offset_args,
    const SourceSpan& span) {
    uint32_t dim = param.type.dimension;
    ensure_grid_dim(dim);
    if (offset_args.size() != dim) {
        ctx.diags.error(span, "stencil of dimension " + std::to_string(dim) +
                              " expects " + std::to_string(dim) + " offset argument(s), got " +
                              std::to_string(offset_args.size()));
        return false;
    }
    int32_t offs[3] = {0, 0, 0};
    for (size_t i = 0; i < offset_args.size(); ++i) {
        check_expr(offset_args[i].get(), Type::i32());
        auto val = ctx.eval_const_int(offset_args[i].get(), fn.module_name, genv, scopes);
        if (!val.has_value()) {
            ctx.diags.error(offset_args[i]->span,
                            "stencil offset must be a compile-time integer constant");
            return false;
        }
        offs[i] = static_cast<int32_t>(*val);
        // Replace offset expression with concrete IntLit so MIR lowering sees exact constant
        offset_args[i]->kind = Expr::Kind::IntLit;
        offset_args[i]->int_val = offs[i];
        offset_args[i]->resolved_type = Type::i32();
    }
    param.record_stencil_offset(offs[0], offs[1], offs[2]);
    if (param.declared_radius >= 0) {
        int32_t max_off = std::max({std::abs(offs[0]), std::abs(offs[1]), std::abs(offs[2])});
        if (max_off > param.declared_radius) {
            ctx.diags.error(span, "stencil offset (" + std::to_string(offs[0]) +
                                  (dim >= 2 ? ", " + std::to_string(offs[1]) : "") +
                                  (dim >= 3 ? ", " + std::to_string(offs[2]) : "") +
                                  ") exceeds declared stencil radius " +
                                  std::to_string(param.declared_radius));
            return false;
        }
    }
    fn.max_stencil_radius = std::max(fn.max_stencil_radius, param.inferred_radius);
    for (int a = 0; a < 3; ++a) {
        fn.stencil_halo[a] = std::max(fn.stencil_halo[a], param.halo[a]);
    }
    return true;
}
} // namespace brokernel::detail

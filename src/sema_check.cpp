#include "sema_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
namespace brokernel::detail {
static bool match_type_pattern(
    const TypeExpr* pattern,
    const Type& actual,
    const std::string& decl_mod,
    SemaContext& ctx,
    GenericEnv& inferred) {
    if (!pattern || actual.is_void()) return false;
    // Direct generic parameter name
    if (pattern->args.empty()) {
        // Check if pattern->name is one of the type params
        if (pattern->name == "T" || pattern->name == "U" || pattern->name == "V" || pattern->name == "Elem") {
            auto it = inferred.type_params.find(pattern->name);
            if (it == inferred.type_params.end()) {
                inferred.type_params[pattern->name] = actual;
                return true;
            }
            return it->second == actual;
        }
    }
    if ((pattern->name == "vec2" || pattern->name == "vec3" || pattern->name == "vec4") &&
        actual.is_vector()) {
        uint32_t lanes = static_cast<uint32_t>(pattern->name.back() - '0');
        if (actual.dimension != lanes) return false;
        if (!pattern->args.empty() && pattern->args[0].type_arg) {
            return match_type_pattern(pattern->args[0].type_arg.get(), Type::scalar_type(actual.scalar), decl_mod, ctx, inferred);
        }
        return true;
    }
    if (pattern->name == "vec" && pattern->args.size() == 2) {
        if (!actual.is_vector() && !actual.is_scalar()) return false;
        uint32_t lanes = actual.is_vector() ? actual.dimension : 1;
        if (pattern->args[0].type_arg) {
            match_type_pattern(pattern->args[0].type_arg.get(), Type::scalar_type(actual.scalar), decl_mod, ctx, inferred);
        }
        if (!pattern->args[1].ident_or_const_name.empty()) {
            ConstValue cv;
            cv.type = Type::i32();
            cv.int_val = lanes;
            inferred.const_params[pattern->args[1].ident_or_const_name] = cv;
        }
        return true;
    }
    if ((pattern->name == "stencil" || pattern->name == "stencil1d" ||
         pattern->name == "stencil2d" || pattern->name == "stencil3d") &&
        actual.is_stencil()) {
        if (!pattern->args.empty() && pattern->args[0].type_arg) {
            match_type_pattern(pattern->args[0].type_arg.get(), actual.element_type(), decl_mod, ctx, inferred);
        }
        size_t rad_idx = 1;
        if (pattern->name == "stencil") {
            if (pattern->args.size() >= 2 && !pattern->args[1].ident_or_const_name.empty()) {
                ConstValue cv{Type::i32(), static_cast<int64_t>(actual.dimension), 0.0, false};
                inferred.const_params[pattern->args[1].ident_or_const_name] = cv;
            }
            rad_idx = 2;
        }
        if (pattern->args.size() > rad_idx && !pattern->args[rad_idx].ident_or_const_name.empty()) {
            int32_t r = actual.stencil_radius >= 0 ? actual.stencil_radius : 1;
            ConstValue cv{Type::i32(), r, 0.0, false};
            inferred.const_params[pattern->args[rad_idx].ident_or_const_name] = cv;
        }
        return true;
    }
    if ((pattern->name == "store" || pattern->name == "read_store" ||
         pattern->name == "write_store" || pattern->name == "rw_store") &&
        actual.is_store()) {
        if (!pattern->args.empty() && pattern->args[0].type_arg) {
            match_type_pattern(pattern->args[0].type_arg.get(), actual.element_type(), decl_mod, ctx, inferred);
        }
        if (pattern->args.size() >= 2 && !pattern->args[1].ident_or_const_name.empty()) {
            ConstValue cv{Type::i32(), static_cast<int64_t>(actual.dimension), 0.0, false};
            inferred.const_params[pattern->args[1].ident_or_const_name] = cv;
        }
        return true;
    }
    if (actual.is_struct() && !pattern->args.empty()) {
        std::string st_mod;
        const StructDecl* sdecl = ctx.find_struct_decl(decl_mod, pattern->name, &st_mod);
        const StructDef* sdef = ctx.out->types.find_struct(actual.struct_id);
        if (sdecl && sdef) {
            for (size_t fi = 0; fi < sdecl->fields.size() && fi < sdef->fields.size(); ++fi) {
                match_type_pattern(sdecl->fields[fi].type.get(), sdef->fields[fi].type, st_mod, ctx, inferred);
            }
        }
        return true;
    }
    return true;
}
Type FnCheckState::check_user_call(Expr* e, const FnDecl& callee_decl, const std::string& callee_mod) {
    if (e->args.size() < callee_decl.params.size()) {
        for (size_t i = e->args.size(); i < callee_decl.params.size(); ++i) {
            if (!callee_decl.params[i].default_value) break;
            e->args.push_back(clone_expr(callee_decl.params[i].default_value.get()));
        }
    }
    if (e->args.size() != callee_decl.params.size()) {
        ctx.diags.error(e->span, "function '" + callee_decl.name + "' expects " +
                                 std::to_string(callee_decl.params.size()) + " argument(s), got " +
                                 std::to_string(e->args.size()));
        return Type::void_type();
    }
    // First type-check arguments without forced coercion if generic
    for (size_t i = 0; i < e->args.size(); ++i) {
        if (callee_decl.generic_params.empty()) {
            Type ptype = ctx.resolve_type(callee_decl.params[i].type.get(), callee_mod, {});
            check_expr(e->args[i].get(), ptype);
        } else {
            check_expr(e->args[i].get());
        }
    }
    GenericEnv call_env;
    if (!callee_decl.generic_params.empty()) {
        if (e->generic_args.size() > callee_decl.generic_params.size()) {
            ctx.diags.error(e->span, "too many generic arguments for function '" + callee_decl.name + "'");
            return Type::void_type();
        }
        // 1. Bind explicit generic arguments
        for (size_t i = 0; i < e->generic_args.size(); ++i) {
            const auto& gp = callee_decl.generic_params[i];
            const auto& ga = e->generic_args[i];
            if (gp.is_const) {
                std::optional<int64_t> val = ga.int_arg;
                if (!val && !ga.ident_or_const_name.empty()) {
                    auto cv = ctx.find_const(fn.module_name, {}, ga.ident_or_const_name, genv, scopes);
                    if (cv && cv->type.is_integer()) val = cv->int_val;
                }
                if (!val) {
                    ctx.diags.error(ga.span, "expected compile-time integer for const generic parameter '" + gp.name + "'");
                    val = 1;
                }
                call_env.const_params[gp.name] = ConstValue{Type::i32(), *val, static_cast<double>(*val), false};
            } else {
                Type targ = Type::f32();
                if (ga.type_arg) {
                    targ = ctx.resolve_type(ga.type_arg.get(), fn.module_name, genv);
                } else if (!ga.ident_or_const_name.empty()) {
                    TypeExpr tmp{ga.span, ga.ident_or_const_name, {}};
                    targ = ctx.resolve_type(&tmp, fn.module_name, genv);
                }
                call_env.type_params[gp.name] = targ;
            }
        }
        // 2. Infer remaining generic parameters from argument types
        for (size_t i = 0; i < callee_decl.params.size(); ++i) {
            const TypeExpr* ptexpr = callee_decl.params[i].type.get();
            const Type& arg_t = e->args[i]->resolved_type;
            if (!ptexpr) continue;
            // Check direct generic param name
            bool is_gp_name = false;
            for (const auto& gp : callee_decl.generic_params) {
                if (!gp.is_const && gp.name == ptexpr->name && ptexpr->args.empty()) {
                    is_gp_name = true;
                    if (call_env.type_params.find(gp.name) == call_env.type_params.end() &&
                        !is_unsuffixed_literal(e->args[i].get())) {
                        call_env.type_params[gp.name] = arg_t;
                    }
                }
            }
            if (!is_gp_name) {
                match_type_pattern(ptexpr, arg_t, callee_mod, ctx, call_env);
            }
        }
        // 3. Apply defaults or report missing generic parameters
        for (const auto& gp : callee_decl.generic_params) {
            if (gp.is_const) {
                if (call_env.const_params.find(gp.name) == call_env.const_params.end()) {
                    if (gp.default_int.has_value()) {
                        call_env.const_params[gp.name] = ConstValue{Type::i32(), *gp.default_int, 0.0, false};
                    } else {
                        ctx.diags.error(e->span, "could not infer const generic parameter '" + gp.name + "' for '" + callee_decl.name + "'");
                        call_env.const_params[gp.name] = ConstValue{Type::i32(), 1, 0.0, false};
                    }
                }
            } else {
                if (call_env.type_params.find(gp.name) == call_env.type_params.end()) {
                    if (gp.default_type) {
                        call_env.type_params[gp.name] = ctx.resolve_type(gp.default_type.get(), callee_mod, call_env);
                    } else {
                        // Second pass if all args were unsuffixed literals
                        call_env.type_params[gp.name] = Type::f32();
                    }
                }
            }
        }
    }
    const CheckedFunction* inst = ctx.instantiate_function(callee_decl, callee_mod, call_env, e->span, false);
    if (!inst) return Type::void_type();
    // Update call name to the mangled instantiation name so MIR lowering can look it up directly
    std::string inst_name = inst->name;
    std::vector<CheckedParam> inst_params = inst->params;
    Type inst_ret = inst->return_type;
    bool inst_f64 = inst->uses_f64;
    bool inst_grid = inst->uses_grid_coords;
    uint32_t inst_grid_dim = inst->grid_dim;
    e->kind = Expr::Kind::Call;
    e->path.clear();
    e->name = inst_name;
    if (inst_f64) fn.uses_f64 = true;
    if (inst_grid) ensure_grid_dim(std::max(1u, inst_grid_dim));
    // Coerce and validate arguments against instantiated parameter types, and propagate stencil offsets
    for (size_t i = 0; i < e->args.size() && i < inst_params.size(); ++i) {
        const auto& cparam = inst_params[i];
        if (cparam.type.is_stencil()) {
            if (!e->args[i]->resolved_type.is_stencil() ||
                e->args[i]->resolved_type.element_type() != cparam.type.element_type() ||
                e->args[i]->resolved_type.dimension != cparam.type.dimension) {
                ctx.diags.error(e->args[i]->span, "stencil argument type mismatch: expected '" +
                                                  cparam.type.to_string() + "', got '" +
                                                  e->args[i]->resolved_type.to_string() + "'");
            }
            // Propagate stencil offsets from callee parameter back to caller's stencil parameter
            if (e->args[i]->kind == Expr::Kind::Ident) {
                if (LocalVar* lv = lookup_var(e->args[i]->name)) {
                    if (lv->param_index >= 0 && static_cast<size_t>(lv->param_index) < fn.params.size()) {
                        auto& caller_param = fn.params[lv->param_index];
                        for (const auto& off : cparam.stencil_offsets) {
                            caller_param.record_stencil_offset(off.dx, off.dy, off.dz);
                            if (caller_param.declared_radius >= 0) {
                                int32_t m = std::max({std::abs(off.dx), std::abs(off.dy), std::abs(off.dz)});
                                if (m > caller_param.declared_radius) {
                                    ctx.diags.error(e->span, "called function '" + callee_decl.name +
                                                             "' accesses stencil offset exceeding declared radius " +
                                                             std::to_string(caller_param.declared_radius));
                                }
                            }
                        }
                        fn.max_stencil_radius = std::max(fn.max_stencil_radius, caller_param.inferred_radius);
                        for (int a = 0; a < 3; ++a) {
                            fn.stencil_halo[a] = std::max(fn.stencil_halo[a], caller_param.halo[a]);
                        }
                    }
                }
            }
        } else if (cparam.type.is_store()) {
            if (!e->args[i]->resolved_type.is_store() ||
                e->args[i]->resolved_type.element_type() != cparam.type.element_type() ||
                e->args[i]->resolved_type.dimension != cparam.type.dimension) {
                ctx.diags.error(e->args[i]->span, "store argument type mismatch");
            }
            if (can_Write(cparam.type.access) && !can_Write(e->args[i]->resolved_type.access)) {
                ctx.diags.error(e->args[i]->span, "cannot pass read-only store to writable store parameter");
            }
            if (can_Read(cparam.type.access) && !can_Read(e->args[i]->resolved_type.access)) {
                ctx.diags.error(e->args[i]->span, "cannot pass write-only store to readable store parameter");
            }
        } else {
            check_assignable(e->args[i].get(), cparam.type, ctx.diags, e->args[i]->span);
        }
    }
    e->resolved_type = inst_ret;
    mark_type_usage(inst_ret);
    return inst_ret;
}
Type FnCheckState::check_expr(Expr* e, std::optional<Type> expected) {
    if (!e) return Type::void_type();
    switch (e->kind) {
        case Expr::Kind::IntLit: {
            ScalarKind sk = (e->literal_suffix != ScalarKind::Void) ? e->literal_suffix : ScalarKind::I32;
            if (e->literal_suffix == ScalarKind::Void && expected && expected->is_integer() && expected->is_scalar()) {
                sk = expected->scalar;
            }
            e->resolved_type = Type::scalar_type(sk);
            mark_type_usage(e->resolved_type);
            return e->resolved_type;
        }
        case Expr::Kind::FloatLit: {
            ScalarKind sk = (e->literal_suffix != ScalarKind::Void) ? e->literal_suffix : ScalarKind::F32;
            if (e->literal_suffix == ScalarKind::Void && expected) {
                if (expected->is_f64() || (expected->is_vector() && expected->scalar == ScalarKind::F64)) {
                    sk = ScalarKind::F64;
                }
            }
            e->resolved_type = Type::scalar_type(sk);
            mark_type_usage(e->resolved_type);
            return e->resolved_type;
        }
        case Expr::Kind::BoolLit:
            e->resolved_type = Type::bool_type();
            return e->resolved_type;
        case Expr::Kind::Ident: {
            if (LocalVar* lv = lookup_var(e->name)) {
                e->resolved_type = lv->type;
                mark_type_usage(lv->type);
                return lv->type;
            }
            auto cv = ctx.find_const(fn.module_name, {}, e->name, genv, scopes);
            if (cv) {
                if (cv->type.is_float()) {
                    e->kind = Expr::Kind::FloatLit;
                    e->float_val = cv->float_val;
                    e->literal_suffix = cv->type.scalar;
                } else if (cv->type.is_bool()) {
                    e->kind = Expr::Kind::BoolLit;
                    e->bool_val = cv->bool_val;
                } else {
                    e->kind = Expr::Kind::IntLit;
                    e->int_val = cv->int_val;
                    e->literal_suffix = cv->type.scalar;
                }
                e->resolved_type = cv->type;
                mark_type_usage(cv->type);
                return cv->type;
            }
            ctx.diags.error(e->span, "undeclared identifier '" + e->name + "'");
            return Type::void_type();
        }
        case Expr::Kind::Path: {
            auto cv = ctx.find_const(fn.module_name, e->path, e->name, genv, scopes);
            if (cv) {
                if (cv->type.is_float()) {
                    e->kind = Expr::Kind::FloatLit;
                    e->float_val = cv->float_val;
                    e->literal_suffix = cv->type.scalar;
                } else if (cv->type.is_bool()) {
                    e->kind = Expr::Kind::BoolLit;
                    e->bool_val = cv->bool_val;
                } else {
                    e->kind = Expr::Kind::IntLit;
                    e->int_val = cv->int_val;
                    e->literal_suffix = cv->type.scalar;
                }
                e->resolved_type = cv->type;
                mark_type_usage(cv->type);
                return cv->type;
            }
            ctx.diags.error(e->span, "undeclared constant or symbol '" + ModuleRegistry::format_module_path(e->path) + "'");
            return Type::void_type();
        }
        case Expr::Kind::Unary: {
            Type sub = check_expr(e->lhs.get(), expected);
            if (e->unary_op == UnaryOp::Not) {
                if (!sub.is_bool()) {
                    ctx.diags.error(e->span, "logical '!' requires a bool operand");
                }
                e->resolved_type = Type::bool_type();
                return e->resolved_type;
            }
            if (e->unary_op == UnaryOp::BitNot) {
                if (!sub.is_integer()) {
                    ctx.diags.error(e->span, "bitwise '~' requires an integer operand");
                }
            }
            e->resolved_type = sub;
            mark_type_usage(sub);
            return sub;
        }
        case Expr::Kind::Binary: {
            Type lt = check_expr(e->lhs.get(), expected);
            Type rt = check_expr(e->rhs.get(), expected);
            // Harmonize unsuffixed literals or scalar-vector broadcasts
            if (lt != rt) {
                if (!coerce_expr_to_type(e->rhs.get(), lt)) {
                    coerce_expr_to_type(e->lhs.get(), rt);
                }
                lt = e->lhs->resolved_type;
                rt = e->rhs->resolved_type;
            }
            if (lt != rt) {
                if ((lt.is_f64() && rt.is_f32()) || (lt.is_f32() && rt.is_f64())) {
                    ctx.diags.error(e->span, "cannot mix 'f64' and 'f32' without an explicit cast");
                } else {
                    ctx.diags.error(e->span, "binary operator type mismatch: '" + lt.to_string() +
                                             "' vs '" + rt.to_string() + "'");
                }
            }
            mark_type_usage(lt);
            switch (e->binary_op) {
                case BinaryOp::Eq:
                case BinaryOp::Ne:
                case BinaryOp::Lt:
                case BinaryOp::Le:
                case BinaryOp::Gt:
                case BinaryOp::Ge:
                case BinaryOp::LogAnd:
                case BinaryOp::LogOr:
                    e->resolved_type = Type::bool_type();
                    return Type::bool_type();
                default:
                    e->resolved_type = lt;
                    return lt;
            }
        }
        case Expr::Kind::Select: {
            Type ct = check_expr(e->cond.get(), Type::bool_type());
            if (!ct.is_bool()) {
                ctx.diags.error(e->cond->span, "ternary condition must be 'bool'");
            }
            Type lt = check_expr(e->lhs.get(), expected);
            Type rt = check_expr(e->rhs.get(), expected);
            if (lt != rt) {
                if (!coerce_expr_to_type(e->rhs.get(), lt)) {
                    coerce_expr_to_type(e->lhs.get(), rt);
                }
                lt = e->lhs->resolved_type;
                rt = e->rhs->resolved_type;
            }
            if (lt != rt) {
                ctx.diags.error(e->span, "ternary branches must have identical types ('" +
                                         lt.to_string() + "' vs '" + rt.to_string() + "')");
            }
            mark_type_usage(lt);
            e->resolved_type = lt;
            return lt;
        }
        case Expr::Kind::Cast: {
            check_expr(e->lhs.get());
            Type target = ctx.resolve_type(e->target_type.get(), fn.module_name, genv);
            mark_type_usage(target);
            e->resolved_type = target;
            return target;
        }
        case Expr::Kind::Call: {
            std::string callee_mod;
            const FnDecl* fdecl = ctx.find_fn_decl(fn.module_name, e->path, e->name, &callee_mod);
            if (fdecl) {
                return check_user_call(e, *fdecl, callee_mod);
            }
            if (!e->path.empty()) {
                ctx.diags.error(e->span, "unknown function '" + ModuleRegistry::format_module_path(e->path) + "'");
                return Type::void_type();
            }
            return check_builtin_call(e, expected);
        }
        case Expr::Kind::MethodCall:
            return check_method_call(e);
        case Expr::Kind::Field: {
            Type recv = check_expr(e->lhs.get());
            const std::string& f = e->name;
            // Allow .x on scalar (for D=1 generic vector code)
            if (recv.is_scalar() && (f == "x" || f == "r")) {
                e->resolved_type = recv;
                return recv;
            }
            if (recv.is_vector()) {
                if (f.empty() || f.size() > 4) {
                    ctx.diags.error(e->span, "invalid vector swizzle '." + f + "'");
                    return Type::scalar_type(recv.scalar);
                }
                for (char c : f) {
                    uint32_t idx = 4;
                    if (c == 'x' || c == 'r' || c == 's') idx = 0;
                    else if (c == 'y' || c == 'g' || c == 't') idx = 1;
                    else if (c == 'z' || c == 'b' || c == 'p') idx = 2;
                    else if (c == 'w' || c == 'a' || c == 'q') idx = 3;
                    if (idx >= recv.dimension) {
                        ctx.diags.error(e->span, "vector swizzle component '" + std::string(1, c) +
                                                 "' out of bounds for '" + recv.to_string() + "'");
                    }
                }
                Type rt = (f.size() == 1) ? Type::scalar_type(recv.scalar)
                                          : Type::vec(recv.scalar, static_cast<uint32_t>(f.size()));
                e->resolved_type = rt;
                return rt;
            }
            if (recv.is_struct()) {
                const StructDef* sdef = ctx.out->types.find_struct(recv.struct_id);
                if (sdef) {
                    if (const StructField* sf = sdef->find_field(f)) {
                        e->resolved_type = sf->type;
                        mark_type_usage(sf->type);
                        return sf->type;
                    }
                }
                ctx.diags.error(e->span, "struct '" + recv.struct_name + "' has no field '" + f + "'");
                return Type::void_type();
            }
            ctx.diags.error(e->span, "cannot access field '" + f + "' on type '" + recv.to_string() + "'");
            return Type::void_type();
        }
        case Expr::Kind::Index: {
            Type recv = check_expr(e->lhs.get());
            if (recv.is_stencil()) {
                LocalVar* lv = (e->lhs->kind == Expr::Kind::Ident) ? lookup_var(e->lhs->name) : nullptr;
                if (lv && lv->param_index >= 0) {
                    validate_stencil_access(fn.params[lv->param_index], e->args, e->span);
                }
                Type elem = recv.element_type();
                mark_type_usage(elem);
                e->resolved_type = elem;
                return elem;
            }
            if (recv.is_store()) {
                if (!can_Read(recv.access)) {
                    ctx.diags.error(e->span, "cannot read from write-only store");
                }
                if (e->args.size() != recv.dimension) {
                    ctx.diags.error(e->span, "store indexing expects " + std::to_string(recv.dimension) + " index argument(s)");
                }
                for (auto& a : e->args) {
                    Type at = check_expr(a.get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(a->span, "store index must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                Type elem = recv.element_type();
                mark_type_usage(elem);
                e->resolved_type = elem;
                return elem;
            }
            if (recv.is_vector() || recv.is_array() || recv.is_scalar()) {
                if (e->args.size() != 1) {
                    ctx.diags.error(e->span, "indexing expects 1 index expression");
                } else {
                    Type at = check_expr(e->args[0].get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(e->args[0]->span, "index must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                Type elem = recv.element_type();
                e->resolved_type = elem;
                return elem;
            }
            ctx.diags.error(e->span, "type '" + recv.to_string() + "' is not indexable");
            return Type::void_type();
        }
        case Expr::Kind::StructLit: {
            Type st_type = ctx.resolve_type(e->target_type.get(), fn.module_name, genv);
            if (!st_type.is_struct()) {
                ctx.diags.error(e->span, "'" + (e->target_type ? e->target_type->name : "") + "' is not a struct type");
                return Type::void_type();
            }
            const StructDef* sdef = ctx.out->types.find_struct(st_type.struct_id);
            if (sdef) {
                for (auto& fi : e->fields) {
                    const StructField* sf = sdef->find_field(fi.name);
                    if (!sf) {
                        ctx.diags.error(fi.span, "struct '" + sdef->name + "' has no field '" + fi.name + "'");
                        check_expr(fi.value.get());
                    } else {
                        check_expr(fi.value.get(), sf->type);
                        check_assignable(fi.value.get(), sf->type, ctx.diags, fi.span);
                    }
                }
            }
            mark_type_usage(st_type);
            e->resolved_type = st_type;
            return st_type;
        }
    }
    return Type::void_type();
}
void FnCheckState::check_block_stmts(std::vector<StmtPtr>& stmts) {
    for (auto& s : stmts) {
        check_stmt(s.get());
    }
}
void FnCheckState::check_stmt(Stmt* s) {
    if (!s) return;
    switch (s->kind) {
        case Stmt::Kind::Block:
            push_scope();
            check_block_stmts(s->body);
            pop_scope();
            break;
        case Stmt::Kind::Let:
        case Stmt::Kind::Var: {
            std::optional<Type> annotated;
            if (s->var_type) {
                annotated = ctx.resolve_type(s->var_type.get(), fn.module_name, genv);
            }
            Type init_t = Type::void_type();
            if (s->init_expr) {
                init_t = check_expr(s->init_expr.get(), annotated);
            }
            Type final_t = annotated ? *annotated : init_t;
            if (final_t.is_void()) {
                ctx.diags.error(s->span, "variable '" + s->var_name + "' must have a non-void type");
                final_t = Type::i32();
            }
            if (annotated && s->init_expr) {
                check_assignable(s->init_expr.get(), *annotated, ctx.diags, s->span);
            }
            s->resolved_var_type = final_t;
            mark_type_usage(final_t);
            LocalVar lv;
            lv.name = s->var_name;
            lv.type = final_t;
            lv.is_mutable = (s->kind == Stmt::Kind::Var);
            if (!lv.is_mutable && s->init_expr) {
                if (auto cv = ctx.eval_const_expr(s->init_expr.get(), fn.module_name, genv, scopes)) {
                    lv.is_const = true;
                    lv.const_val = *cv;
                }
            }
            scopes.back()[s->var_name] = lv;
            break;
        }
        case Stmt::Kind::Assign: {
            // Special case: assigning to store index `st[x, y] = val`
            if (s->assign_target && s->assign_target->kind == Expr::Kind::Index && s->assign_target->lhs) {
                Type recv = check_expr(s->assign_target->lhs.get());
                if (recv.is_stencil()) {
                    ctx.diags.error(s->span, "cannot write to read-only stencil");
                    return;
                }
                if (recv.is_store()) {
                    if (!can_Write(recv.access)) {
                        ctx.diags.error(s->span, "cannot write to read-only store");
                    }
                    if (s->compound_op.has_value() && !can_Read(recv.access)) {
                        ctx.diags.error(s->span, "compound assignment on store requires read-write access");
                    }
                    if (s->assign_target->args.size() != recv.dimension) {
                        ctx.diags.error(s->span, "store indexing expects " + std::to_string(recv.dimension) + " coordinate(s)");
                    }
                    for (auto& a : s->assign_target->args) {
                        Type at = check_expr(a.get(), Type::i32());
                        if (!at.is_integer() || !at.is_scalar()) {
                            ctx.diags.error(a->span, "store coordinate must be a scalar integer, got '" + at.to_string() + "'");
                        }
                    }
                    Type elem = recv.element_type();
                    if (s->compound_op.has_value() && !elem.is_numeric()) {
                        ctx.diags.error(s->span, "compound assignment on store requires numeric scalar or vector element type");
                    }
                    s->assign_target->resolved_type = elem;
                    check_expr(s->assign_value.get(), elem);
                    if (s->compound_op.has_value() && elem.is_vector()) {
                        Type elem_s = Type::scalar_type(elem.scalar);
                        if (s->assign_value->resolved_type == elem_s || coerce_expr_to_type(s->assign_value.get(), elem_s)) {
                            coerce_expr_to_type(s->assign_value.get(), elem);
                        }
                    }
                    check_assignable(s->assign_value.get(), elem, ctx.diags, s->span);
                    mark_type_usage(elem);
                    return;
                }
            }
            // Validate lvalue chain and mutability on base identifier
            const Expr* base = s->assign_target.get();
            while (base && (base->kind == Expr::Kind::Field || base->kind == Expr::Kind::Index)) {
                base = base->lhs.get();
            }
            if (!base || base->kind != Expr::Kind::Ident) {
                ctx.diags.error(s->span, "invalid assignment target");
                if (s->assign_value) check_expr(s->assign_value.get());
                return;
            }
            LocalVar* lv = lookup_var(base->name);
            if (!lv) {
                ctx.diags.error(s->span, "cannot assign to '" + base->name + "'");
                if (s->assign_value) check_expr(s->assign_value.get());
                return;
            }
            if (lv->type.is_stencil()) {
                ctx.diags.error(s->span, "cannot write to read-only stencil");
                return;
            }
            if (lv->type.is_store()) {
                ctx.diags.error(s->span, "cannot assign to sub-element of store '" + base->name + "' directly; assign a full store element");
                return;
            }
            if (!lv->is_mutable) {
                ctx.diags.error(s->span, "cannot assign to immutable variable '" + base->name + "' (declare with 'var' instead of 'let')");
            }
            Type target_t = check_expr(s->assign_target.get());
            for (const Expr* cur = s->assign_target.get();
                 cur && (cur->kind == Expr::Kind::Field || cur->kind == Expr::Kind::Index);
                 cur = cur->lhs.get()) {
                if (cur->kind == Expr::Kind::Field && cur->lhs && cur->lhs->resolved_type.is_vector() && cur->name.size() > 1) {
                    uint32_t seen_mask = 0;
                    for (char c : cur->name) {
                        uint32_t idx = (c == 'x' || c == 'r' || c == 's') ? 0 :
                                       (c == 'y' || c == 'g' || c == 't') ? 1 :
                                       (c == 'z' || c == 'b' || c == 'p') ? 2 : 3;
                        if (seen_mask & (1u << idx)) {
                            ctx.diags.error(cur->span, "cannot assign to vector swizzle with duplicate components '." + cur->name + "'");
                            break;
                        }
                        seen_mask |= (1u << idx);
                    }
                }
            }
            if (s->compound_op.has_value() && !target_t.is_numeric()) {
                ctx.diags.error(s->span, "compound assignment requires a numeric scalar or vector target, got '" + target_t.to_string() + "'");
            }
            check_expr(s->assign_value.get(), target_t);
            if (s->compound_op.has_value() && target_t.is_vector()) {
                // Allow `vec *= scalar`
                Type elem_t = Type::scalar_type(target_t.scalar);
                if (s->assign_value->resolved_type == elem_t || coerce_expr_to_type(s->assign_value.get(), elem_t)) {
                    coerce_expr_to_type(s->assign_value.get(), target_t);
                }
            }
            check_assignable(s->assign_value.get(), target_t, ctx.diags, s->span);
            break;
        }
        case Stmt::Kind::If: {
            // Check if condition is a compile-time constant (e.g. `if D == 1`)
            if (auto cv = ctx.eval_const_expr(s->cond.get(), fn.module_name, genv, scopes)) {
                if (cv->type.is_bool()) {
                    std::vector<StmtPtr> chosen = cv->bool_val ? std::move(s->body) : std::move(s->else_body);
                    s->kind = Stmt::Kind::Block;
                    s->cond.reset();
                    s->body = std::move(chosen);
                    s->else_body.clear();
                    push_scope();
                    check_block_stmts(s->body);
                    pop_scope();
                    return;
                }
            }
            Type ct = check_expr(s->cond.get(), Type::bool_type());
            if (!ct.is_bool()) {
                ctx.diags.error(s->cond->span, "if condition must be 'bool', got '" + ct.to_string() + "'");
            }
            push_scope();
            check_block_stmts(s->body);
            pop_scope();
            if (!s->else_body.empty()) {
                push_scope();
                check_block_stmts(s->else_body);
                pop_scope();
            }
            break;
        }
        case Stmt::Kind::ForRange: {
            bool has_unroll = false;
            for (const auto& attr : s->attributes) {
                if (attr.name == "unroll") has_unroll = true;
            }
            auto start_c = ctx.eval_const_int(s->range_start.get(), fn.module_name, genv, scopes);
            auto end_c = ctx.eval_const_int(s->range_end.get(), fn.module_name, genv, scopes);
            // Unroll @unroll loops when bounds are compile-time constants
            if (has_unroll && start_c.has_value() && end_c.has_value()) {
                int64_t lo = *start_c;
                int64_t hi = s->inclusive_range ? *end_c : (*end_c - 1);
                std::vector<StmtPtr> orig_body = std::move(s->body);
                std::string loop_var = s->var_name;
                s->kind = Stmt::Kind::Block;
                s->body.clear();
                for (int64_t v = lo; v <= hi; ++v) {
                    auto block = std::make_unique<Stmt>();
                    block->kind = Stmt::Kind::Block;
                    block->span = s->span;
                    auto let_v = std::make_unique<Stmt>();
                    let_v->kind = Stmt::Kind::Let;
                    let_v->span = s->span;
                    let_v->var_name = loop_var;
                    let_v->resolved_var_type = Type::i32();
                    let_v->init_expr = std::make_unique<Expr>();
                    let_v->init_expr->kind = Expr::Kind::IntLit;
                    let_v->init_expr->span = s->span;
                    let_v->init_expr->int_val = v;
                    let_v->init_expr->resolved_type = Type::i32();
                    block->body.push_back(std::move(let_v));
                    for (const auto& bs : orig_body) {
                        block->body.push_back(clone_stmt(bs.get()));
                    }
                    s->body.push_back(std::move(block));
                }
                push_scope();
                check_block_stmts(s->body);
                pop_scope();
                return;
            }
            Type st = check_expr(s->range_start.get(), Type::i32());
            check_expr(s->range_end.get(), st);
            check_assignable(s->range_end.get(), st, ctx.diags, s->range_end->span);
            s->resolved_var_type = st;
            push_scope();
            LocalVar lv;
            lv.name = s->var_name;
            lv.type = st;
            lv.is_mutable = false;
            scopes.back()[s->var_name] = lv;
            ++loop_depth;
            check_block_stmts(s->body);
            --loop_depth;
            pop_scope();
            break;
        }
        case Stmt::Kind::ForC: {
            push_scope();
            if (s->for_init) check_stmt(s->for_init.get());
            if (s->cond) {
                Type ct = check_expr(s->cond.get(), Type::bool_type());
                if (!ct.is_bool()) ctx.diags.error(s->cond->span, "for loop condition must be 'bool'");
            }
            if (s->for_step) check_stmt(s->for_step.get());
            ++loop_depth;
            check_block_stmts(s->body);
            --loop_depth;
            pop_scope();
            break;
        }
        case Stmt::Kind::While: {
            Type ct = check_expr(s->cond.get(), Type::bool_type());
            if (!ct.is_bool()) {
                ctx.diags.error(s->cond->span, "while condition must be 'bool'");
            }
            push_scope();
            ++loop_depth;
            check_block_stmts(s->body);
            --loop_depth;
            pop_scope();
            break;
        }
        case Stmt::Kind::Break:
        case Stmt::Kind::Continue:
            if (loop_depth <= 0) {
                ctx.diags.error(s->span, std::string(s->kind == Stmt::Kind::Break ? "'break'" : "'continue'") +
                                         " statement outside of a loop");
            }
            break;
        case Stmt::Kind::Return: {
            if (s->expr) {
                if (fn.return_type.is_void()) {
                    ctx.diags.error(s->span, "void function '" + fn.source_name + "' cannot return a value");
                } else {
                    check_expr(s->expr.get(), fn.return_type);
                    check_assignable(s->expr.get(), fn.return_type, ctx.diags, s->span);
                }
            } else if (!fn.return_type.is_void()) {
                ctx.diags.error(s->span, "non-void function '" + fn.source_name + "' must return a value of type '" +
                                         fn.return_type.to_string() + "'");
            }
            break;
        }
        case Stmt::Kind::Expr:
            check_expr(s->expr.get());
            break;
    }
}
const CheckedFunction* SemaContext::instantiate_function(
    const FnDecl& decl,
    const std::string& decl_mod,
    const GenericEnv& genv,
    const SourceSpan& call_site,
    bool as_entry_kernel) {
    // Build mangled name
    std::string mangled = (decl_mod == out->module_name) ? decl.name : (decl_mod + "__" + decl.name);
    if (!decl.generic_params.empty()) {
        for (const auto& gp : decl.generic_params) {
            mangled += "_";
            if (gp.is_const) {
                auto it = genv.const_params.find(gp.name);
                mangled += (it != genv.const_params.end()) ? std::to_string(it->second.int_val) : "0";
            } else {
                auto it = genv.type_params.find(gp.name);
                std::string ts = (it != genv.type_params.end()) ? it->second.to_string() : "void";
                for (char c : ts) {
                    if (std::isalnum(static_cast<unsigned char>(c))) mangled.push_back(c);
                    else mangled.push_back('_');
                }
            }
        }
    }
    // Check recursion first
    for (const auto& active : active_fn_stack) {
        if (active == mangled) {
            diags.error(call_site, "recursive function call to '" + decl.name + "' is not allowed in GPU/world kernels");
            return nullptr;
        }
    }
    auto existing_it = fn_instances.find(mangled);
    if (existing_it != fn_instances.end()) {
        return &out->functions[existing_it->second];
    }
    CheckedFunction cfn;
    cfn.span = decl.span;
    cfn.name = mangled;
    cfn.source_name = decl.name;
    cfn.module_name = decl_mod;
    cfn.is_entry_kernel = as_entry_kernel || decl.is_entry_kernel();
    // Stage & attributes
    if (decl.has_attribute("vertex")) cfn.stage = Stage::Vertex;
    else if (decl.has_attribute("fragment")) cfn.stage = Stage::Fragment;
    else cfn.stage = Stage::Compute;
    if (const Attribute* grid_attr = decl.find_attribute("grid")) {
        if (!grid_attr->int_args.empty()) {
            int64_t gd = grid_attr->int_args[0];
            if (gd < 1 || gd > 3) {
                diags.error(grid_attr->span, "@grid(D) dimension D must be 1, 2, or 3");
            } else {
                cfn.grid_dim = static_cast<uint32_t>(gd);
            }
        } else {
            cfn.grid_dim = 2;
        }
    }
    if (const Attribute* wg_attr = decl.find_attribute("workgroup_size")) {
        cfn.has_explicit_local_size = true;
        for (size_t i = 0; i < 3; ++i) {
            int64_t sz = (i < wg_attr->int_args.size()) ? wg_attr->int_args[i] : 1;
            if (sz <= 0) {
                diags.error(wg_attr->span, "@workgroup_size dimensions must be > 0");
                sz = 1;
            }
            cfn.local_size[i] = static_cast<uint32_t>(sz);
        }
    }
    // Resolve parameters
    for (const auto& pdecl : decl.params) {
        CheckedParam cp;
        cp.span = pdecl.span;
        cp.name = pdecl.name;
        cp.type = resolve_type(pdecl.type.get(), decl_mod, genv);
        cp.access = cp.type.access;
        if (cp.type.scalar == ScalarKind::F64) cfn.uses_f64 = true;
        if (cp.type.is_scalar()) cp.role = ParamRole::Scalar;
        else if (cp.type.is_vector()) cp.role = ParamRole::Vector;
        else if (cp.type.is_struct()) cp.role = ParamRole::Struct;
        else if (cp.type.is_array()) cp.role = ParamRole::Array;
        else if (cp.type.is_grid()) {
            cp.role = ParamRole::Grid;
            cfn.uses_grid_coords = true;
            cfn.grid_dim = std::max(cfn.grid_dim, cp.type.dimension);
        } else if (cp.type.is_store()) {
            cp.role = ParamRole::Store;
            cfn.grid_dim = std::max(cfn.grid_dim, cp.type.dimension);
        } else if (cp.type.is_stencil()) {
            cp.role = ParamRole::Stencil;
            cp.declared_radius = cp.type.stencil_radius;
            cfn.grid_dim = std::max(cfn.grid_dim, cp.type.dimension);
        } else if (cp.type.is_image()) {
            cp.role = ParamRole::Image;
        } else if (cp.type.is_sampler()) {
            cp.role = ParamRole::Sampler;
        }
        for (const auto& pattr : pdecl.attributes) {
            if (pattr.name == "binding") {
                if (pattr.int_args.size() == 1) {
                    cp.descriptor_binding = static_cast<uint32_t>(pattr.int_args[0]);
                } else if (pattr.int_args.size() >= 2) {
                    cp.descriptor_set = static_cast<uint32_t>(pattr.int_args[0]);
                    cp.descriptor_binding = static_cast<uint32_t>(pattr.int_args[1]);
                }
            }
        }
        cfn.params.push_back(std::move(cp));
    }
    if (!cfn.has_explicit_local_size) {
        if (cfn.grid_dim == 2) {
            cfn.local_size[0] = 8;
            cfn.local_size[1] = 8;
            cfn.local_size[2] = 1;
        } else if (cfn.grid_dim == 3) {
            cfn.local_size[0] = 4;
            cfn.local_size[1] = 4;
            cfn.local_size[2] = 4;
        }
    }
    cfn.return_type = decl.return_type ? resolve_type(decl.return_type.get(), decl_mod, genv) : Type::void_type();
    if (cfn.return_type.scalar == ScalarKind::F64) cfn.uses_f64 = true;
    // Clone body and type-check
    for (const auto& s : decl.body) {
        cfn.body.push_back(clone_stmt(s.get()));
    }
    active_fn_stack.push_back(mangled);
    {
        FnCheckState state{*this, cfn, genv, {}, 0};
        state.push_scope();
        for (size_t i = 0; i < cfn.params.size(); ++i) {
            LocalVar lv;
            lv.name = cfn.params[i].name;
            lv.type = cfn.params[i].type;
            lv.is_mutable = false;
            lv.param_index = static_cast<int>(i);
            state.scopes.back()[lv.name] = lv;
        }
        for (const auto& [cname, cval] : genv.const_params) {
            LocalVar lv;
            lv.name = cname;
            lv.type = cval.type;
            lv.is_mutable = false;
            lv.is_const = true;
            lv.const_val = cval;
            state.scopes.back()[cname] = lv;
        }
        state.check_block_stmts(cfn.body);
        state.pop_scope();
    }
    active_fn_stack.pop_back();
    size_t idx = out->functions.size();
    out->functions.push_back(std::move(cfn));
    fn_instances[mangled] = idx;
    return &out->functions[idx];
}
} // namespace brokernel::detail

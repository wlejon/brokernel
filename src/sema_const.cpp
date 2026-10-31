#include "sema_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace brokernel::detail {

static ConstValue convert_const_scalar(const ConstValue& cv, ScalarKind sk) {
    ConstValue out;
    out.type = Type::scalar_type(sk);
    if (sk == ScalarKind::F32 || sk == ScalarKind::F64) {
        double v = cv.type.is_float_scalar() ? cv.float_val : static_cast<double>(cv.int_val);
        out.float_val = v;
        out.int_val = static_cast<int64_t>(v);
        out.bool_val = (v != 0.0);
    } else if (sk == ScalarKind::Bool) {
        bool b = cv.type.is_bool() ? cv.bool_val
               : cv.type.is_float_scalar() ? (cv.float_val != 0.0)
                                           : (cv.int_val != 0);
        out.bool_val = b;
        out.int_val = b ? 1 : 0;
        out.float_val = b ? 1.0 : 0.0;
    } else {
        int64_t iv = cv.type.is_float_scalar() ? static_cast<int64_t>(cv.float_val) : cv.int_val;
        if (sk == ScalarKind::I32) iv = static_cast<int32_t>(iv);
        else if (sk == ScalarKind::U32) iv = static_cast<uint32_t>(iv);
        out.int_val = iv;
        out.float_val = static_cast<double>(iv);
        out.bool_val = (iv != 0);
    }
    return out;
}

static void flatten_const_leaves(const ConstValue& cv, std::vector<ConstValue>& out) {
    if (cv.elements.empty()) {
        out.push_back(cv);
        return;
    }
    for (const auto& el : cv.elements) {
        flatten_const_leaves(el, out);
    }
}

static ConstValue unflatten_const_value(const Type& t, const std::vector<ConstValue>& leaves, size_t& idx) {
    if (t.is_scalar()) {
        ConstValue src = (idx < leaves.size()) ? leaves[idx++] : ConstValue{};
        return convert_const_scalar(src, t.scalar);
    }
    ConstValue out;
    out.type = t;
    Type elem_t = t.element_type();
    out.elements.reserve(t.dimension);
    for (uint32_t i = 0; i < t.dimension; ++i) {
        out.elements.push_back(unflatten_const_value(elem_t, leaves, idx));
    }
    return out;
}

ConstValue convert_const_value(const ConstValue& cv, const Type& target, const TypeRegistry& types) {
    if (target.is_scalar()) {
        return convert_const_scalar(cv, target.scalar);
    }
    std::vector<ConstValue> leaves;
    flatten_const_leaves(cv, leaves);
    if (leaves.empty()) return cv;
    uint32_t needed = types.flat_leaf_count(target);
    if (leaves.size() == 1 && needed > 1) {
        leaves.assign(needed, leaves[0]);
    } else {
        while (leaves.size() < needed) leaves.push_back(leaves.back());
    }
    size_t idx = 0;
    return unflatten_const_value(target, leaves, idx);
}

static std::optional<ConstValue> apply_const_unary(UnaryOp op, const ConstValue& sub) {
    if (!sub.elements.empty()) {
        ConstValue res;
        res.type = sub.type;
        res.elements.reserve(sub.elements.size());
        for (const auto& el : sub.elements) {
            auto r = apply_const_unary(op, el);
            if (!r) return std::nullopt;
            res.elements.push_back(*r);
        }
        return res;
    }
    ConstValue res = sub;
    if (op == UnaryOp::Neg) {
        res.int_val = -sub.int_val;
        res.float_val = -sub.float_val;
    } else if (op == UnaryOp::Not) {
        res.type = Type::bool_type();
        res.bool_val = !sub.bool_val;
        res.int_val = res.bool_val ? 1 : 0;
    } else if (op == UnaryOp::BitNot) {
        res.int_val = ~sub.int_val;
    }
    return res;
}

static std::optional<ConstValue> apply_const_binary(BinaryOp op, const ConstValue& l, const ConstValue& r) {
    if (!l.elements.empty() || !r.elements.empty()) {
        if (!l.elements.empty() && !r.elements.empty()) {
            if (l.elements.size() != r.elements.size()) return std::nullopt;
            ConstValue res;
            res.type = l.type;
            res.elements.reserve(l.elements.size());
            for (size_t i = 0; i < l.elements.size(); ++i) {
                auto sub = apply_const_binary(op, l.elements[i], r.elements[i]);
                if (!sub) return std::nullopt;
                res.elements.push_back(*sub);
            }
            return res;
        }
        if (!l.elements.empty()) {
            ConstValue res;
            res.type = l.type;
            res.elements.reserve(l.elements.size());
            for (const auto& el : l.elements) {
                auto sub = apply_const_binary(op, el, r);
                if (!sub) return std::nullopt;
                res.elements.push_back(*sub);
            }
            return res;
        }
        ConstValue res;
        res.type = r.type;
        res.elements.reserve(r.elements.size());
        for (const auto& el : r.elements) {
            auto sub = apply_const_binary(op, l, el);
            if (!sub) return std::nullopt;
            res.elements.push_back(*sub);
        }
        return res;
    }
    ConstValue res;
    bool is_fp = l.type.is_float_scalar() || r.type.is_float_scalar();
    res.type = is_fp ? (l.type.is_f64() || r.type.is_f64() ? Type::f64() : Type::f32()) : Type::i32();
    switch (op) {
        case BinaryOp::Add:
            res.int_val = l.int_val + r.int_val;
            res.float_val = l.float_val + r.float_val;
            return res;
        case BinaryOp::Sub:
            res.int_val = l.int_val - r.int_val;
            res.float_val = l.float_val - r.float_val;
            return res;
        case BinaryOp::Mul:
            res.int_val = l.int_val * r.int_val;
            res.float_val = l.float_val * r.float_val;
            return res;
        case BinaryOp::Div:
            if (!is_fp && r.int_val == 0) return std::nullopt;
            if (!is_fp) res.int_val = l.int_val / r.int_val;
            res.float_val = l.float_val / r.float_val;
            return res;
        case BinaryOp::Mod:
            if (r.int_val == 0) return std::nullopt;
            res.int_val = l.int_val % r.int_val;
            return res;
        case BinaryOp::Eq:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val == r.float_val) : (l.int_val == r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::Ne:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val != r.float_val) : (l.int_val != r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::Lt:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val < r.float_val) : (l.int_val < r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::Le:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val <= r.float_val) : (l.int_val <= r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::Gt:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val > r.float_val) : (l.int_val > r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::Ge:
            res.type = Type::bool_type();
            res.bool_val = is_fp ? (l.float_val >= r.float_val) : (l.int_val >= r.int_val);
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::LogAnd:
            res.type = Type::bool_type();
            res.bool_val = l.bool_val && r.bool_val;
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::LogOr:
            res.type = Type::bool_type();
            res.bool_val = l.bool_val || r.bool_val;
            res.int_val = res.bool_val ? 1 : 0;
            return res;
        case BinaryOp::BitAnd:
            res.int_val = l.int_val & r.int_val;
            res.bool_val = (res.int_val != 0);
            return res;
        case BinaryOp::BitOr:
            res.int_val = l.int_val | r.int_val;
            res.bool_val = (res.int_val != 0);
            return res;
        case BinaryOp::BitXor:
            res.int_val = l.int_val ^ r.int_val;
            res.bool_val = (res.int_val != 0);
            return res;
        case BinaryOp::Shl:
            res.int_val = l.int_val << (r.int_val & 63);
            res.bool_val = (res.int_val != 0);
            return res;
        case BinaryOp::Shr:
            res.int_val = l.int_val >> (r.int_val & 63);
            res.bool_val = (res.int_val != 0);
            return res;
        default:
            return std::nullopt;
    }
}

std::optional<ConstValue> SemaContext::eval_const_expr(
    const Expr* e,
    const std::string& current_mod,
    const GenericEnv& genv,
    const std::vector<std::unordered_map<std::string, LocalVar>>& scopes) {
    if (!e) return std::nullopt;
    switch (e->kind) {
        case Expr::Kind::IntLit: {
            ConstValue cv;
            cv.type = (e->literal_suffix != ScalarKind::Void) ? Type::scalar_type(e->literal_suffix) : Type::i32();
            cv.int_val = e->int_val;
            cv.float_val = static_cast<double>(e->int_val);
            cv.bool_val = (e->int_val != 0);
            return cv;
        }
        case Expr::Kind::FloatLit: {
            ConstValue cv;
            cv.type = (e->literal_suffix != ScalarKind::Void) ? Type::scalar_type(e->literal_suffix) : Type::f32();
            cv.float_val = e->float_val;
            cv.int_val = static_cast<int64_t>(e->float_val);
            cv.bool_val = (e->float_val != 0.0);
            return cv;
        }
        case Expr::Kind::BoolLit: {
            ConstValue cv;
            cv.type = Type::bool_type();
            cv.bool_val = e->bool_val;
            cv.int_val = e->bool_val ? 1 : 0;
            return cv;
        }
        case Expr::Kind::Ident:
            return find_const(current_mod, {}, e->name, genv, scopes);
        case Expr::Kind::Path:
            return find_const(current_mod, e->path, e->name, genv, scopes);
        case Expr::Kind::Unary: {
            auto sub = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            if (!sub) return std::nullopt;
            return apply_const_unary(e->unary_op, *sub);
        }
        case Expr::Kind::Binary: {
            auto l = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            if (l && l->type.is_bool()) {
                if (e->binary_op == BinaryOp::LogAnd && !l->bool_val) return l;
                if (e->binary_op == BinaryOp::LogOr && l->bool_val) return l;
            }
            auto r = eval_const_expr(e->rhs.get(), current_mod, genv, scopes);
            if (!l || !r) return std::nullopt;
            return apply_const_binary(e->binary_op, *l, *r);
        }
        case Expr::Kind::Select: {
            auto c = eval_const_expr(e->cond.get(), current_mod, genv, scopes);
            if (!c || !c->type.is_bool()) return std::nullopt;
            return eval_const_expr(c->bool_val ? e->lhs.get() : e->rhs.get(), current_mod, genv, scopes);
        }
        case Expr::Kind::Field: {
            auto sub = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            if (!sub) return std::nullopt;
            if (sub->type.is_scalar() && (e->name == "x" || e->name == "r")) {
                return sub;
            }
            if (sub->type.is_vector() && sub->elements.size() == sub->type.dimension) {
                if (e->name.empty() || e->name.size() > 4) return std::nullopt;
                auto swz_idx = [](char c) -> uint32_t {
                    if (c == 'x' || c == 'r' || c == 's') return 0;
                    if (c == 'y' || c == 'g' || c == 't') return 1;
                    if (c == 'z' || c == 'b' || c == 'p') return 2;
                    if (c == 'w' || c == 'a' || c == 'q') return 3;
                    return 4;
                };
                if (e->name.size() == 1) {
                    uint32_t idx = swz_idx(e->name[0]);
                    if (idx >= sub->elements.size()) return std::nullopt;
                    return sub->elements[idx];
                }
                ConstValue out;
                out.type = Type::vec(sub->type.scalar, static_cast<uint32_t>(e->name.size()));
                out.elements.reserve(e->name.size());
                for (char c : e->name) {
                    uint32_t idx = swz_idx(c);
                    if (idx >= sub->elements.size()) return std::nullopt;
                    out.elements.push_back(sub->elements[idx]);
                }
                return out;
            }
            return std::nullopt;
        }
        case Expr::Kind::Index: {
            if (e->args.size() != 1) return std::nullopt;
            auto sub = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            auto idx = eval_const_int(e->args[0].get(), current_mod, genv, scopes);
            if (!sub || !idx) return std::nullopt;
            if (*idx < 0 || static_cast<size_t>(*idx) >= sub->elements.size()) return std::nullopt;
            return sub->elements[static_cast<size_t>(*idx)];
        }
        case Expr::Kind::Cast: {
            auto sub = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            if (!sub) return std::nullopt;
            Type target = resolve_type(e->target_type.get(), current_mod, genv);
            return convert_const_value(*sub, target, out->types);
        }
        case Expr::Kind::Call: {
            const std::string& n = e->name;
            if (n == "select" && e->args.size() == 3) {
                auto c = eval_const_expr(e->args[0].get(), current_mod, genv, scopes);
                if (!c || !c->type.is_bool()) return std::nullopt;
                return eval_const_expr(c->bool_val ? e->args[1].get() : e->args[2].get(), current_mod, genv, scopes);
            }
            if ((n == "bool" || n == "i32" || n == "u32" || n == "i64" || n == "u64" || n == "f32" || n == "f64") &&
                e->generic_args.empty() && e->args.size() == 1) {
                auto sub = eval_const_expr(e->args[0].get(), current_mod, genv, scopes);
                if (!sub) return std::nullopt;
                TypeExpr te{e->span, n, {}};
                return convert_const_value(*sub, resolve_type(&te, current_mod, genv), out->types);
            }
            if ((n == "is_f64" || n == "sizeof") && e->generic_args.size() == 1 && e->args.empty()) {
                Type t = Type::f32();
                const auto& ga = e->generic_args[0];
                if (ga.type_arg) {
                    t = resolve_type(ga.type_arg.get(), current_mod, genv);
                } else if (!ga.ident_or_const_name.empty()) {
                    TypeExpr tmp{ga.span, ga.ident_or_const_name, {}};
                    t = resolve_type(&tmp, current_mod, genv);
                }
                if (n == "is_f64") {
                    bool b = (t.scalar == ScalarKind::F64);
                    return ConstValue{Type::bool_type(), b ? 1 : 0, b ? 1.0 : 0.0, b};
                }
                int64_t sz = static_cast<int64_t>(out->types.byte_size(t));
                return ConstValue{Type::i32(), sz, static_cast<double>(sz), false};
            }
            if (n == "vec2" || n == "vec3" || n == "vec4" ||
                n == "vec2f" || n == "vec3f" || n == "vec4f" ||
                n == "vec2d" || n == "vec3d" || n == "vec4d" ||
                n == "vec2i" || n == "vec3i" || n == "vec4i" ||
                n == "vec2u" || n == "vec3u" || n == "vec4u" ||
                n == "mat2" || n == "mat3" || n == "mat4" ||
                n == "mat2f" || n == "mat3f" || n == "mat4f" ||
                n == "mat2d" || n == "mat3d" || n == "mat4d" ||
                n == "vec" || n == "mat" || n == "array") {
                std::vector<ConstValue> arg_vals;
                arg_vals.reserve(e->args.size());
                for (const auto& a : e->args) {
                    auto av = eval_const_expr(a.get(), current_mod, genv, scopes);
                    if (!av) return std::nullopt;
                    arg_vals.push_back(*av);
                }
                Type target = e->resolved_type;
                if (target.is_void()) {
                    if (!e->generic_args.empty() || n == "vec" || n == "mat") {
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
                        target = resolve_type(&te, current_mod, genv);
                    } else if (n == "array") {
                        Type elem_t = arg_vals.empty() ? Type::f32() : arg_vals[0].type;
                        target = Type::array_of(elem_t, std::max<uint32_t>(1, static_cast<uint32_t>(arg_vals.size())));
                    } else if (n == "vec2" || n == "vec3" || n == "vec4") {
                        uint32_t lanes = static_cast<uint32_t>(n.back() - '0');
                        ScalarKind sk = ScalarKind::F32;
                        for (size_t i = 0; i < arg_vals.size(); ++i) {
                            if (e->args[i]->literal_suffix != ScalarKind::Void || !arg_vals[i].type.is_scalar()) {
                                sk = arg_vals[i].type.scalar;
                                break;
                            }
                            if (arg_vals[i].type.is_integer()) {
                                sk = arg_vals[i].type.scalar;
                            }
                        }
                        target = Type::vec(sk, lanes);
                    } else if (n == "mat2" || n == "mat3" || n == "mat4") {
                        uint32_t dim = static_cast<uint32_t>(n.back() - '0');
                        ScalarKind sk = ScalarKind::F32;
                        for (size_t i = 0; i < arg_vals.size(); ++i) {
                            if (e->args[i]->literal_suffix != ScalarKind::Void || !arg_vals[i].type.is_scalar()) {
                                sk = arg_vals[i].type.scalar;
                                break;
                            }
                        }
                        target = Type::array_of(Type::vec(sk, dim), dim);
                    } else {
                        TypeExpr te{e->span, n, {}};
                        target = resolve_type(&te, current_mod, genv);
                    }
                }
                std::vector<ConstValue> leaves;
                for (const auto& av : arg_vals) {
                    flatten_const_leaves(av, leaves);
                }
                uint32_t needed = out->types.flat_leaf_count(target);
                if (leaves.empty()) {
                    leaves.assign(needed, ConstValue{});
                } else if (leaves.size() == 1 && needed > 1) {
                    leaves.assign(needed, leaves[0]);
                } else {
                    while (leaves.size() < needed) leaves.push_back(leaves.back());
                }
                size_t idx = 0;
                return unflatten_const_value(target, leaves, idx);
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
    }
}

std::optional<int64_t> SemaContext::eval_const_int(
    const Expr* e,
    const std::string& current_mod,
    const GenericEnv& genv,
    const std::vector<std::unordered_map<std::string, LocalVar>>& scopes) {
    auto cv = eval_const_expr(e, current_mod, genv, scopes);
    if (!cv || !cv->type.is_integer()) return std::nullopt;
    return cv->int_val;
}

void fold_const_into_expr(Expr* e, const ConstValue& cv) {
    if (!e) return;
    e->path.clear();
    e->generic_args.clear();
    e->args.clear();
    e->fields.clear();
    e->lhs.reset();
    e->rhs.reset();
    e->cond.reset();
    e->target_type.reset();
    e->resolved_type = cv.type;
    if (cv.type.is_scalar()) {
        if (cv.type.is_float_scalar()) {
            e->kind = Expr::Kind::FloatLit;
            e->float_val = cv.float_val;
            e->literal_suffix = cv.type.scalar;
        } else if (cv.type.is_bool()) {
            e->kind = Expr::Kind::BoolLit;
            e->bool_val = cv.bool_val;
        } else {
            e->kind = Expr::Kind::IntLit;
            e->int_val = cv.int_val;
            e->literal_suffix = cv.type.scalar;
        }
        return;
    }
    if (cv.type.is_vector() || cv.type.is_array()) {
        e->kind = Expr::Kind::Call;
        e->name = cv.type.is_vector() ? ("vec" + std::to_string(cv.type.dimension)) : "array";
        e->args.reserve(cv.elements.size());
        for (const auto& el : cv.elements) {
            auto arg = std::make_unique<Expr>();
            arg->span = e->span;
            fold_const_into_expr(arg.get(), el);
            e->args.push_back(std::move(arg));
        }
        return;
    }
}

bool validate_shared_type(
    const Type& t,
    const TypeRegistry& types,
    DiagnosticBag& diags,
    const SourceSpan& span,
    const std::string& name) {
    auto leaves = types.flatten_scalars(t);
    if (leaves.empty()) {
        diags.error(span, "shared variable '" + name + "' must have a concrete value type");
        return false;
    }
    for (ScalarKind sk : leaves) {
        if (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64 || sk == ScalarKind::Void) {
            diags.error(span, "shared variable '" + name + "' must use 32-bit scalar, vector, struct, or array types");
            return false;
        }
    }
    return true;
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
                if (s->is_shared) {
                    ctx.diags.error(s->span, "shared variable '" + s->var_name + "' cannot have a declaration initializer");
                }
                init_t = check_expr(s->init_expr.get(), annotated);
            }
            Type final_t = annotated ? *annotated : init_t;
            if (final_t.is_void()) {
                ctx.diags.error(s->span, "variable '" + s->var_name + "' must have a non-void type");
                final_t = Type::i32();
            }
            if (s->is_shared) {
                require_stage(Stage::Compute, "shared variable '" + s->var_name + "'", s->span);
                fn.uses_barriers_or_shared = true;
                validate_shared_type(final_t, ctx.out->types, ctx.diags, s->span, s->var_name);
            }
            if (annotated && s->init_expr) {
                check_assignable(s->init_expr.get(), *annotated, ctx.diags, s->span);
            }
            s->resolved_var_type = final_t;
            mark_type_usage(final_t);
            LocalVar lv;
            lv.name = s->var_name;
            lv.type = final_t;
            lv.is_mutable = (s->kind == Stmt::Kind::Var) || s->is_shared;
            lv.is_shared = s->is_shared;
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
            if (!base || (base->kind != Expr::Kind::Ident && base->kind != Expr::Kind::Path)) {
                ctx.diags.error(s->span, "invalid assignment target");
                if (s->assign_value) check_expr(s->assign_value.get());
                return;
            }
            LocalVar* lv = base->kind == Expr::Kind::Ident ? lookup_var(base->name) : nullptr;
            // A module-level resource: a place in a buffer or shared variable.
            LocalVar global_lv;
            if (!lv) {
                int32_t g = ctx.find_global(fn.module_name, base->path, base->name);
                if (g >= 0) {
                    const auto& gp = ctx.out->globals[static_cast<size_t>(g)];
                    global_lv.name = base->name;
                    global_lv.type = gp.type;
                    global_lv.is_mutable = (gp.role == ParamRole::Shared);
                    global_lv.is_shared = (gp.role == ParamRole::Shared);
                    lv = &global_lv;
                }
            }
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
            if (lv->type.is_block()) {
                // A place in the buffer: the variable naming it is not what changes.
                if (!lv->type.is_buffer() || lv->type.access == AccessMode::Read) {
                    ctx.diags.error(s->span, "'" + base->name + "' is read-only (" + lv->type.to_string() + ")");
                } else if (s->assign_target->kind == Expr::Kind::Ident || s->assign_target->kind == Expr::Kind::Path) {
                    ctx.diags.error(s->span, "assign to an element of buffer '" + base->name + "', not the buffer");
                }
            } else if (!lv->is_mutable) {
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

} // namespace brokernel::detail

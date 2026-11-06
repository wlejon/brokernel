#include "sema_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace brokernel::detail {

bool is_ad_diff_type(const Type& t, const TypeRegistry& types) {
    if (t.is_float()) return true;
    if (t.is_array()) return is_ad_diff_type(t.element_type(), types);
    if (t.is_struct()) {
        if (const StructDef* sdef = types.find_struct(t.struct_id)) {
            for (const auto& f : sdef->fields) {
                if (is_ad_diff_type(f.type, types)) return true;
            }
        }
    }
    return false;
}

namespace {

bool is_diff_type(const Type& t, const TypeRegistry& types) {
    return is_ad_diff_type(t, types);
}

ExprPtr make_float_lit(double v, ScalarKind sk, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::FloatLit;
    e->span = sp;
    e->float_val = v;
    e->literal_suffix = sk;
    e->resolved_type = Type::scalar_type(sk);
    return e;
}

ExprPtr make_int_lit(int64_t v, ScalarKind sk, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::IntLit;
    e->span = sp;
    e->int_val = v;
    e->literal_suffix = sk;
    e->resolved_type = Type::scalar_type(sk);
    return e;
}

ExprPtr make_splat_or_scalar(double v, const Type& t, const SourceSpan& sp) {
    ScalarKind sk = (t.scalar == ScalarKind::F64) ? ScalarKind::F64 : ScalarKind::F32;
    ExprPtr lit = make_float_lit(v, sk, sp);
    if (t.is_scalar()) return lit;
    auto call = std::make_unique<Expr>();
    call->kind = Expr::Kind::Call;
    call->span = sp;
    call->name = "__splat";
    call->resolved_type = t;
    call->args.push_back(std::move(lit));
    return call;
}

ExprPtr make_zero_expr(const Type& t, const TypeRegistry& types, const SourceSpan& sp) {
    if (t.is_float()) return make_splat_or_scalar(0.0, t, sp);
    if (t.is_bool()) {
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::BoolLit;
        e->span = sp;
        e->bool_val = false;
        e->resolved_type = t;
        return e;
    }
    if (t.is_integer()) {
        ExprPtr z = make_int_lit(0, t.scalar, sp);
        if (t.is_scalar()) return z;
        auto call = std::make_unique<Expr>();
        call->kind = Expr::Kind::Call;
        call->span = sp;
        call->name = "__splat";
        call->resolved_type = t;
        call->args.push_back(std::move(z));
        return call;
    }
    if (t.is_array()) {
        auto call = std::make_unique<Expr>();
        call->kind = Expr::Kind::Call;
        call->span = sp;
        call->name = "array";
        call->resolved_type = t;
        Type elem = t.element_type();
        for (uint32_t i = 0; i < t.dimension; ++i) {
            call->args.push_back(make_zero_expr(elem, types, sp));
        }
        return call;
    }
    if (t.is_struct()) {
        auto sl = std::make_unique<Expr>();
        sl->kind = Expr::Kind::StructLit;
        sl->span = sp;
        sl->name = t.struct_name;
        sl->resolved_type = t;
        if (const StructDef* sdef = types.find_struct(t.struct_id)) {
            for (const auto& f : sdef->fields) {
                FieldInit fi;
                fi.span = sp;
                fi.name = f.name;
                fi.value = make_zero_expr(f.type, types, sp);
                sl->fields.push_back(std::move(fi));
            }
        }
        return sl;
    }
    return make_float_lit(0.0, ScalarKind::F32, sp);
}

ExprPtr make_ident(const std::string& name, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Ident;
    e->span = sp;
    e->name = name;
    e->resolved_type = t;
    return e;
}

ExprPtr make_unary(UnaryOp op, ExprPtr lhs, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Unary;
    e->span = sp;
    e->unary_op = op;
    e->lhs = std::move(lhs);
    e->resolved_type = t;
    return e;
}

ExprPtr make_binary(BinaryOp op, ExprPtr lhs, ExprPtr rhs, const Type& t, const SourceSpan& sp) {
    if (lhs->resolved_type != t && lhs->resolved_type.is_scalar() && t.is_vector()) {
        auto sp_call = std::make_unique<Expr>();
        sp_call->kind = Expr::Kind::Call;
        sp_call->span = sp;
        sp_call->name = "__splat";
        sp_call->resolved_type = t;
        sp_call->args.push_back(std::move(lhs));
        lhs = std::move(sp_call);
    }
    if (rhs->resolved_type != t && rhs->resolved_type.is_scalar() && t.is_vector()) {
        auto sp_call = std::make_unique<Expr>();
        sp_call->kind = Expr::Kind::Call;
        sp_call->span = sp;
        sp_call->name = "__splat";
        sp_call->resolved_type = t;
        sp_call->args.push_back(std::move(rhs));
        rhs = std::move(sp_call);
    }
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Binary;
    e->span = sp;
    e->binary_op = op;
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    e->resolved_type = t;
    return e;
}

ExprPtr make_call(const std::string& name, std::vector<ExprPtr> args, const Type& ret_t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Call;
    e->span = sp;
    e->name = name;
    e->args = std::move(args);
    e->resolved_type = ret_t;
    return e;
}

StmtPtr make_let_stmt(const std::string& name, const Type& t, ExprPtr init, const SourceSpan& sp, bool is_var = false) {
    auto s = std::make_unique<Stmt>();
    s->kind = is_var ? Stmt::Kind::Var : Stmt::Kind::Let;
    s->span = sp;
    s->var_name = name;
    s->resolved_var_type = t;
    s->init_expr = std::move(init);
    return s;
}

StmtPtr make_return_stmt(ExprPtr val, const SourceSpan& sp) {
    auto s = std::make_unique<Stmt>();
    s->kind = Stmt::Kind::Return;
    s->span = sp;
    s->expr = std::move(val);
    return s;
}

struct AdSynthesizer {
    SemaContext& ctx;
    std::set<std::string> diff_vars;

    std::string ensure_jvp_function(const std::string& fn_name) {
        std::string jvp_name = "__ad_jvp_" + fn_name;
        if (ctx.out->find_function(jvp_name)) return jvp_name;

        const CheckedFunction* orig_ptr = ctx.out->find_function(fn_name);
        if (!orig_ptr) return fn_name;
        std::vector<StmtPtr> orig_body;
        orig_body.reserve(orig_ptr->body.size());
        for (const auto& s : orig_ptr->body) orig_body.push_back(clone_stmt(s.get()));

        CheckedFunction jvp;
        jvp.span = orig_ptr->span;
        jvp.name = jvp_name;
        jvp.source_name = jvp_name;
        jvp.module_name = orig_ptr->module_name;
        jvp.is_entry_kernel = false;
        jvp.is_pub = false;
        jvp.stage = orig_ptr->stage;
        jvp.required_stage = orig_ptr->required_stage;
        jvp.return_type = orig_ptr->return_type;
        jvp.uses_f64 = orig_ptr->uses_f64;
        jvp.f64_site = orig_ptr->f64_site;
        jvp.uses_grid_coords = orig_ptr->uses_grid_coords;
        jvp.uses_descriptors = orig_ptr->uses_descriptors;
        jvp.uses_textures = orig_ptr->uses_textures;
        jvp.grid_dim = orig_ptr->grid_dim;

        std::set<std::string> saved_vars = diff_vars;
        diff_vars.clear();

        for (const auto& p : orig_ptr->params) {
            jvp.params.push_back(p);
            if (is_diff_type(p.type, ctx.out->types)) {
                CheckedParam dp = p;
                dp.name = "_d_" + p.name;
                jvp.params.push_back(std::move(dp));
                diff_vars.insert(p.name);
            }
        }

        jvp.body = diff_stmt_list(orig_body);
        diff_vars = std::move(saved_vars);

        size_t idx = ctx.out->functions.size();
        ctx.out->functions.push_back(std::move(jvp));
        ctx.fn_instances[jvp_name] = idx;
        return jvp_name;
    }

    ExprPtr diff_lvalue(const Expr* target) {
        if (!target) return nullptr;
        if (target->kind == Expr::Kind::Ident) {
            if (diff_vars.count(target->name)) {
                return make_ident("_d_" + target->name, target->resolved_type, target->span);
            }
            return nullptr;
        }
        if (target->kind == Expr::Kind::Field) {
            ExprPtr base = diff_lvalue(target->lhs.get());
            if (!base) return nullptr;
            auto fld = std::make_unique<Expr>();
            fld->kind = Expr::Kind::Field;
            fld->span = target->span;
            fld->name = target->name;
            fld->lhs = std::move(base);
            fld->resolved_type = target->resolved_type;
            return fld;
        }
        if (target->kind == Expr::Kind::Index) {
            ExprPtr base = diff_lvalue(target->lhs.get());
            if (!base) return nullptr;
            auto idx = std::make_unique<Expr>();
            idx->kind = Expr::Kind::Index;
            idx->span = target->span;
            idx->lhs = std::move(base);
            for (const auto& a : target->args) idx->args.push_back(clone_expr(a.get()));
            idx->resolved_type = target->resolved_type;
            return idx;
        }
        return nullptr;
    }

    ExprPtr diff_expr(const Expr* e) {
        if (!e) return make_float_lit(0.0, ScalarKind::F32, {});
        const Type& t = e->resolved_type;
        const SourceSpan& sp = e->span;
        if (!is_diff_type(t, ctx.out->types)) {
            return make_zero_expr(t, ctx.out->types, sp);
        }
        switch (e->kind) {
            case Expr::Kind::IntLit:
            case Expr::Kind::FloatLit:
            case Expr::Kind::BoolLit:
            case Expr::Kind::StringLit:
                return make_zero_expr(t, ctx.out->types, sp);

            case Expr::Kind::Ident:
                if (diff_vars.count(e->name)) {
                    return make_ident("_d_" + e->name, t, sp);
                }
                return make_zero_expr(t, ctx.out->types, sp);

            case Expr::Kind::Path:
                return make_zero_expr(t, ctx.out->types, sp);

            case Expr::Kind::Unary:
                if (e->unary_op == UnaryOp::Neg) {
                    return make_unary(UnaryOp::Neg, diff_expr(e->lhs.get()), t, sp);
                }
                if (e->unary_op == UnaryOp::Pos) {
                    return diff_expr(e->lhs.get());
                }
                return make_zero_expr(t, ctx.out->types, sp);

            case Expr::Kind::Binary: {
                const Expr* l = e->lhs.get();
                const Expr* r = e->rhs.get();
                switch (e->binary_op) {
                    case BinaryOp::Add:
                        return make_binary(BinaryOp::Add, diff_expr(l), diff_expr(r), t, sp);
                    case BinaryOp::Sub:
                        return make_binary(BinaryOp::Sub, diff_expr(l), diff_expr(r), t, sp);
                    case BinaryOp::Mul: {
                        ExprPtr t1 = make_binary(BinaryOp::Mul, diff_expr(l), clone_expr(r), t, sp);
                        ExprPtr t2 = make_binary(BinaryOp::Mul, clone_expr(l), diff_expr(r), t, sp);
                        return make_binary(BinaryOp::Add, std::move(t1), std::move(t2), t, sp);
                    }
                    case BinaryOp::Div: {
                        ExprPtr num1 = make_binary(BinaryOp::Mul, diff_expr(l), clone_expr(r), t, sp);
                        ExprPtr num2 = make_binary(BinaryOp::Mul, clone_expr(l), diff_expr(r), t, sp);
                        ExprPtr num = make_binary(BinaryOp::Sub, std::move(num1), std::move(num2), t, sp);
                        ExprPtr den = make_binary(BinaryOp::Mul, clone_expr(r), clone_expr(r), t, sp);
                        return make_binary(BinaryOp::Div, std::move(num), std::move(den), t, sp);
                    }
                    default:
                        return make_zero_expr(t, ctx.out->types, sp);
                }
            }

            case Expr::Kind::Select: {
                auto sel = std::make_unique<Expr>();
                sel->kind = Expr::Kind::Select;
                sel->span = sp;
                sel->cond = clone_expr(e->cond.get());
                sel->lhs = diff_expr(e->lhs.get());
                sel->rhs = diff_expr(e->rhs.get());
                sel->resolved_type = t;
                return sel;
            }

            case Expr::Kind::Cast: {
                if (e->lhs && e->lhs->resolved_type.is_float() && t.is_float()) {
                    auto c = std::make_unique<Expr>();
                    c->kind = Expr::Kind::Cast;
                    c->span = sp;
                    c->lhs = diff_expr(e->lhs.get());
                    c->target_type = clone_type_expr(e->target_type.get());
                    c->resolved_type = t;
                    return c;
                }
                return make_zero_expr(t, ctx.out->types, sp);
            }

            case Expr::Kind::Field: {
                auto fld = std::make_unique<Expr>();
                fld->kind = Expr::Kind::Field;
                fld->span = sp;
                fld->name = e->name;
                fld->lhs = diff_expr(e->lhs.get());
                fld->resolved_type = t;
                return fld;
            }

            case Expr::Kind::Index: {
                auto idx = std::make_unique<Expr>();
                idx->kind = Expr::Kind::Index;
                idx->span = sp;
                idx->lhs = diff_expr(e->lhs.get());
                for (const auto& a : e->args) idx->args.push_back(clone_expr(a.get()));
                idx->resolved_type = t;
                return idx;
            }

            case Expr::Kind::StructLit: {
                auto sl = std::make_unique<Expr>();
                sl->kind = Expr::Kind::StructLit;
                sl->span = sp;
                sl->name = e->name;
                sl->path = e->path;
                sl->target_type = clone_type_expr(e->target_type.get());
                sl->resolved_type = t;
                for (const auto& f : e->fields) {
                    FieldInit fi;
                    fi.span = f.span;
                    fi.name = f.name;
                    fi.value = diff_expr(f.value.get());
                    sl->fields.push_back(std::move(fi));
                }
                return sl;
            }

            case Expr::Kind::MethodCall:
                return make_zero_expr(t, ctx.out->types, sp);

            case Expr::Kind::Call:
                return diff_call(e);
        }
        return make_zero_expr(t, ctx.out->types, sp);
    }

    ExprPtr diff_call(const Expr* e) {
        const std::string& n = e->name;
        const Type& t = e->resolved_type;
        const SourceSpan& sp = e->span;

        if (const CheckedFunction* callee = ctx.out->find_function(n)) {
            std::vector<CheckedParam> cparams = callee->params;
            std::string jvp_name = ensure_jvp_function(n);
            std::vector<ExprPtr> jvp_args;
            for (size_t i = 0; i < e->args.size() && i < cparams.size(); ++i) {
                jvp_args.push_back(clone_expr(e->args[i].get()));
                if (is_diff_type(cparams[i].type, ctx.out->types)) {
                    jvp_args.push_back(diff_expr(e->args[i].get()));
                }
            }
            return make_call(jvp_name, std::move(jvp_args), t, sp);
        }

        if ((n == "probe" || n == "__probe") && e->args.size() == 2) {
            return diff_expr(e->args[1].get());
        }
        if (n == "__splat" && e->args.size() == 1) {
            std::vector<ExprPtr> a;
            a.push_back(diff_expr(e->args[0].get()));
            return make_call("__splat", std::move(a), t, sp);
        }
        if (n == "vec" || n == "vec2" || n == "vec3" || n == "vec4" ||
            n == "vec2f" || n == "vec3f" || n == "vec4f" ||
            n == "vec2d" || n == "vec3d" || n == "vec4d" || n == "array") {
            auto c = clone_expr(e);
            for (size_t i = 0; i < e->args.size(); ++i) {
                c->args[i] = diff_expr(e->args[i].get());
            }
            return c;
        }
        if ((n == "f32" || n == "f64") && e->args.size() == 1) {
            if (e->args[0]->resolved_type.is_float()) {
                std::vector<ExprPtr> a;
                a.push_back(diff_expr(e->args[0].get()));
                return make_call(n, std::move(a), t, sp);
            }
            return make_zero_expr(t, ctx.out->types, sp);
        }
        if (n == "floor" || n == "ceil" || n == "round" || n == "trunc" || n == "sign" || n == "step") {
            return make_zero_expr(t, ctx.out->types, sp);
        }
        if (n == "fract" && e->args.size() == 1) {
            return diff_expr(e->args[0].get());
        }

        // 1-arg math functions
        if (e->args.size() == 1) {
            const Expr* x = e->args[0].get();
            ExprPtr dx = diff_expr(x);
            auto call1 = [&](const std::string& fn_name, ExprPtr arg) {
                std::vector<ExprPtr> a;
                a.push_back(std::move(arg));
                return make_call(fn_name, std::move(a), t, sp);
            };
            if (n == "sin") {
                return make_binary(BinaryOp::Mul, call1("cos", clone_expr(x)), std::move(dx), t, sp);
            }
            if (n == "cos") {
                return make_unary(UnaryOp::Neg, make_binary(BinaryOp::Mul, call1("sin", clone_expr(x)), std::move(dx), t, sp), t, sp);
            }
            if (n == "tan") {
                ExprPtr cx = call1("cos", clone_expr(x));
                ExprPtr c2 = make_binary(BinaryOp::Mul, clone_expr(cx.get()), std::move(cx), t, sp);
                return make_binary(BinaryOp::Div, std::move(dx), std::move(c2), t, sp);
            }
            if (n == "asin" || n == "acos") {
                ExprPtr x2 = make_binary(BinaryOp::Mul, clone_expr(x), clone_expr(x), t, sp);
                ExprPtr one_minus = make_binary(BinaryOp::Sub, make_splat_or_scalar(1.0, t, sp), std::move(x2), t, sp);
                ExprPtr d = make_binary(BinaryOp::Div, std::move(dx), call1("sqrt", std::move(one_minus)), t, sp);
                return (n == "acos") ? make_unary(UnaryOp::Neg, std::move(d), t, sp) : std::move(d);
            }
            if (n == "atan") {
                ExprPtr x2 = make_binary(BinaryOp::Mul, clone_expr(x), clone_expr(x), t, sp);
                ExprPtr denom = make_binary(BinaryOp::Add, make_splat_or_scalar(1.0, t, sp), std::move(x2), t, sp);
                return make_binary(BinaryOp::Div, std::move(dx), std::move(denom), t, sp);
            }
            if (n == "sinh") {
                return make_binary(BinaryOp::Mul, call1("cosh", clone_expr(x)), std::move(dx), t, sp);
            }
            if (n == "cosh") {
                return make_binary(BinaryOp::Mul, call1("sinh", clone_expr(x)), std::move(dx), t, sp);
            }
            if (n == "tanh") {
                ExprPtr th = call1("tanh", clone_expr(x));
                ExprPtr th2 = make_binary(BinaryOp::Mul, clone_expr(th.get()), std::move(th), t, sp);
                ExprPtr sech2 = make_binary(BinaryOp::Sub, make_splat_or_scalar(1.0, t, sp), std::move(th2), t, sp);
                return make_binary(BinaryOp::Mul, std::move(sech2), std::move(dx), t, sp);
            }
            if (n == "exp") {
                return make_binary(BinaryOp::Mul, clone_expr(e), std::move(dx), t, sp);
            }
            if (n == "exp2") {
                ExprPtr c = make_splat_or_scalar(0.69314718055994530942, t, sp);
                return make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, clone_expr(e), std::move(c), t, sp), std::move(dx), t, sp);
            }
            if (n == "log") {
                return make_binary(BinaryOp::Div, std::move(dx), clone_expr(x), t, sp);
            }
            if (n == "log2") {
                ExprPtr c = make_splat_or_scalar(1.44269504088896340736, t, sp);
                return make_binary(BinaryOp::Div, make_binary(BinaryOp::Mul, std::move(dx), std::move(c), t, sp), clone_expr(x), t, sp);
            }
            if (n == "sqrt") {
                ExprPtr two_sqrt = make_binary(BinaryOp::Mul, make_splat_or_scalar(2.0, t, sp), clone_expr(e), t, sp);
                return make_binary(BinaryOp::Div, std::move(dx), std::move(two_sqrt), t, sp);
            }
            if (n == "inversesqrt") {
                ExprPtr half = make_splat_or_scalar(-0.5, t, sp);
                ExprPtr num = make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, std::move(half), std::move(dx), t, sp), clone_expr(e), t, sp);
                return make_binary(BinaryOp::Div, std::move(num), clone_expr(x), t, sp);
            }
            if (n == "abs") {
                return make_binary(BinaryOp::Mul, call1("sign", clone_expr(x)), std::move(dx), t, sp);
            }
            if (n == "length") {
                const Type& vt = x->resolved_type;
                std::vector<ExprPtr> dargs;
                dargs.push_back(clone_expr(x));
                dargs.push_back(std::move(dx));
                ExprPtr d = make_call("dot", std::move(dargs), t, sp);
                return make_binary(BinaryOp::Div, std::move(d), clone_expr(e), t, sp);
            }
            if (n == "normalize") {
                Type st = Type::scalar_type(t.scalar);
                std::vector<ExprPtr> vv_args;
                vv_args.push_back(clone_expr(x));
                vv_args.push_back(clone_expr(x));
                ExprPtr vv = make_call("dot", std::move(vv_args), st, sp);
                std::vector<ExprPtr> vdv_args;
                vdv_args.push_back(clone_expr(x));
                vdv_args.push_back(clone_expr(dx.get()));
                ExprPtr vdv = make_call("dot", std::move(vdv_args), st, sp);
                ExprPtr term1 = make_binary(BinaryOp::Mul, std::move(dx), std::move(vv), t, sp);
                ExprPtr term2 = make_binary(BinaryOp::Mul, clone_expr(x), std::move(vdv), t, sp);
                ExprPtr num = make_binary(BinaryOp::Sub, std::move(term1), std::move(term2), t, sp);
                std::vector<ExprPtr> l_args;
                l_args.push_back(clone_expr(x));
                ExprPtr len = make_call("length", std::move(l_args), st, sp);
                ExprPtr len3 = make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, clone_expr(len.get()), clone_expr(len.get()), st, sp), std::move(len), st, sp);
                return make_binary(BinaryOp::Div, std::move(num), std::move(len3), t, sp);
            }
        }

        // 2-arg math functions
        if (e->args.size() == 2) {
            const Expr* a = e->args[0].get();
            const Expr* b = e->args[1].get();
            ExprPtr da = diff_expr(a);
            ExprPtr db = diff_expr(b);
            auto step_call = [&](ExprPtr edge, ExprPtr val) {
                std::vector<ExprPtr> sargs;
                sargs.push_back(std::move(edge));
                sargs.push_back(std::move(val));
                return make_call("step", std::move(sargs), t, sp);
            };
            auto mix_call = [&](ExprPtr v0, ExprPtr v1, ExprPtr w) {
                std::vector<ExprPtr> margs;
                margs.push_back(std::move(v0));
                margs.push_back(std::move(v1));
                margs.push_back(std::move(w));
                return make_call("mix", std::move(margs), t, sp);
            };
            if (n == "atan" || n == "atan2") {
                ExprPtr num = make_binary(BinaryOp::Sub,
                    make_binary(BinaryOp::Mul, std::move(da), clone_expr(b), t, sp),
                    make_binary(BinaryOp::Mul, clone_expr(a), std::move(db), t, sp), t, sp);
                ExprPtr den = make_binary(BinaryOp::Add,
                    make_binary(BinaryOp::Mul, clone_expr(b), clone_expr(b), t, sp),
                    make_binary(BinaryOp::Mul, clone_expr(a), clone_expr(a), t, sp), t, sp);
                return make_binary(BinaryOp::Div, std::move(num), std::move(den), t, sp);
            }
            if (n == "pow") {
                ExprPtr ym1 = make_binary(BinaryOp::Sub, clone_expr(b), make_splat_or_scalar(1.0, t, sp), t, sp);
                std::vector<ExprPtr> pargs;
                pargs.push_back(clone_expr(a));
                pargs.push_back(std::move(ym1));
                ExprPtr p_ym1 = make_call("pow", std::move(pargs), t, sp);
                ExprPtr term1 = make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, clone_expr(b), std::move( p_ym1), t, sp), std::move(da), t, sp);
                if (b->kind == Expr::Kind::FloatLit || b->kind == Expr::Kind::IntLit) {
                    return term1;
                }
                std::vector<ExprPtr> largs;
                largs.push_back(clone_expr(a));
                ExprPtr lna = make_call("log", std::move(largs), t, sp);
                ExprPtr term2 = make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, clone_expr(e), std::move(lna), t, sp), std::move(db), t, sp);
                return make_binary(BinaryOp::Add, std::move(term1), std::move(term2), t, sp);
            }
            if (n == "min") {
                return mix_call(std::move(db), std::move(da), step_call(clone_expr(a), clone_expr(b)));
            }
            if (n == "max") {
                return mix_call(std::move(da), std::move(db), step_call(clone_expr(a), clone_expr(b)));
            }
            if (n == "dot") {
                std::vector<ExprPtr> a1, a2;
                a1.push_back(std::move(da));
                a1.push_back(clone_expr(b));
                a2.push_back(clone_expr(a));
                a2.push_back(std::move(db));
                return make_binary(BinaryOp::Add, make_call("dot", std::move(a1), t, sp), make_call("dot", std::move(a2), t, sp), t, sp);
            }
            if (n == "cross") {
                std::vector<ExprPtr> a1, a2;
                a1.push_back(std::move(da));
                a1.push_back(clone_expr(b));
                a2.push_back(clone_expr(a));
                a2.push_back(std::move(db));
                return make_binary(BinaryOp::Add, make_call("cross", std::move(a1), t, sp), make_call("cross", std::move(a2), t, sp), t, sp);
            }
            if (n == "distance") {
                const Type& vt = a->resolved_type;
                ExprPtr diff_v = make_binary(BinaryOp::Sub, clone_expr(a), clone_expr(b), vt, sp);
                ExprPtr d_diff = make_binary(BinaryOp::Sub, std::move(da), std::move(db), vt, sp);
                std::vector<ExprPtr> dargs;
                dargs.push_back(std::move(diff_v));
                dargs.push_back(std::move(d_diff));
                return make_binary(BinaryOp::Div, make_call("dot", std::move(dargs), t, sp), clone_expr(e), t, sp);
            }
        }

        // 3-arg math functions
        if (e->args.size() == 3) {
            const Expr* a = e->args[0].get();
            const Expr* b = e->args[1].get();
            const Expr* c = e->args[2].get();
            ExprPtr da = diff_expr(a);
            ExprPtr db = diff_expr(b);
            ExprPtr dc = diff_expr(c);
            auto step_call = [&](ExprPtr edge, ExprPtr val) {
                std::vector<ExprPtr> sargs;
                sargs.push_back(std::move(edge));
                sargs.push_back(std::move(val));
                return make_call("step", std::move(sargs), t, sp);
            };
            auto mix_call = [&](ExprPtr v0, ExprPtr v1, ExprPtr w) {
                std::vector<ExprPtr> margs;
                margs.push_back(std::move(v0));
                margs.push_back(std::move(v1));
                margs.push_back(std::move(w));
                return make_call("mix", std::move(margs), t, sp);
            };
            if (n == "clamp") {
                // clamp(x, lo, hi): a=x, b=lo, c=hi
                ExprPtr inner = mix_call(std::move(dc), std::move(da), step_call(clone_expr(a), clone_expr(c)));
                return mix_call(std::move(db), std::move(inner), step_call(clone_expr(b), clone_expr(a)));
            }
            if (n == "mix") {
                // mix(a, b, t): da*(1-t) + db*t + (b-a)*dt
                ExprPtr one_minus_t = make_binary(BinaryOp::Sub, make_splat_or_scalar(1.0, t, sp), clone_expr(c), t, sp);
                ExprPtr t1 = make_binary(BinaryOp::Mul, std::move(da), std::move(one_minus_t), t, sp);
                ExprPtr t2 = make_binary(BinaryOp::Mul, std::move(db), clone_expr(c), t, sp);
                ExprPtr b_minus_a = make_binary(BinaryOp::Sub, clone_expr(b), clone_expr(a), t, sp);
                ExprPtr t3 = make_binary(BinaryOp::Mul, std::move(b_minus_a), std::move(dc), t, sp);
                return make_binary(BinaryOp::Add, make_binary(BinaryOp::Add, std::move(t1), std::move(t2), t, sp), std::move(t3), t, sp);
            }
            if (n == "smoothstep") {
                // smoothstep(e0, e1, x): u = clamp((x - e0)/(e1 - e0), 0, 1); d = 6*u*(1-u)*du
                ExprPtr span_e = make_binary(BinaryOp::Sub, clone_expr(b), clone_expr(a), t, sp);
                ExprPtr raw_u = make_binary(BinaryOp::Div, make_binary(BinaryOp::Sub, clone_expr(c), clone_expr(a), t, sp), clone_expr(span_e.get()), t, sp);
                std::vector<ExprPtr> cargs;
                cargs.push_back(clone_expr(raw_u.get()));
                cargs.push_back(make_splat_or_scalar(0.0, t, sp));
                cargs.push_back(make_splat_or_scalar(1.0, t, sp));
                ExprPtr u = make_call("clamp", std::move(cargs), t, sp);
                ExprPtr d_num = make_binary(BinaryOp::Sub, std::move(dc), std::move(da), t, sp);
                ExprPtr d_span = make_binary(BinaryOp::Sub, std::move(db), diff_expr(a), t, sp);
                ExprPtr du_unclamped = make_binary(BinaryOp::Div,
                    make_binary(BinaryOp::Sub, std::move(d_num), make_binary(BinaryOp::Mul, clone_expr(u.get()), std::move(d_span), t, sp), t, sp),
                    std::move(span_e), t, sp);
                ExprPtr mask = make_binary(BinaryOp::Mul,
                    step_call(make_splat_or_scalar(0.0, t, sp), clone_expr(raw_u.get())),
                    step_call( std::move(raw_u), make_splat_or_scalar(1.0, t, sp)), t, sp);
                ExprPtr du = make_binary(BinaryOp::Mul, std::move(du_unclamped), std::move(mask), t, sp);
                ExprPtr six_u = make_binary(BinaryOp::Mul, make_splat_or_scalar(6.0, t, sp), clone_expr(u.get()), t, sp);
                ExprPtr one_m_u = make_binary(BinaryOp::Sub, make_splat_or_scalar(1.0, t, sp), std::move(u), t, sp);
                return make_binary(BinaryOp::Mul, make_binary(BinaryOp::Mul, std::move(six_u), std::move(one_m_u), t, sp), std::move(du), t, sp);
            }
            if (n == "fma") {
                ExprPtr t1 = make_binary(BinaryOp::Mul, std::move(da), clone_expr(b), t, sp);
                ExprPtr t2 = make_binary(BinaryOp::Mul, clone_expr(a), std::move(db), t, sp);
                return make_binary(BinaryOp::Add, make_binary(BinaryOp::Add, std::move(t1), std::move(t2), t, sp), std::move(dc), t, sp);
            }
        }

        return make_zero_expr(t, ctx.out->types, sp);
    }

    std::vector<StmtPtr> diff_stmt_list(const std::vector<StmtPtr>& stmts) {
        std::vector<StmtPtr> out;
        for (const auto& s : stmts) {
            diff_stmt(s.get(), out);
        }
        return out;
    }

    void diff_stmt(const Stmt* s, std::vector<StmtPtr>& out) {
        if (!s) return;
        const SourceSpan& sp = s->span;
        switch (s->kind) {
            case Stmt::Kind::Block: {
                auto blk = std::make_unique<Stmt>();
                blk->kind = Stmt::Kind::Block;
                blk->span = sp;
                blk->attributes = s->attributes;
                blk->body = diff_stmt_list(s->body);
                out.push_back(std::move(blk));
                break;
            }
            case Stmt::Kind::Let:
            case Stmt::Kind::Var: {
                bool is_var = (s->kind == Stmt::Kind::Var);
                const Type& vt = s->resolved_var_type;
                if (!s->is_shared && is_diff_type(vt, ctx.out->types)) {
                    ExprPtr d_init = s->init_expr ? diff_expr(s->init_expr.get()) : make_zero_expr(vt, ctx.out->types, sp);
                    out.push_back(make_let_stmt("_d_" + s->var_name, vt, std::move(d_init), sp, is_var));
                    diff_vars.insert(s->var_name);
                }
                out.push_back(clone_stmt(s));
                break;
            }
            case Stmt::Kind::Assign: {
                ExprPtr d_target = diff_lvalue(s->assign_target.get());
                if (d_target) {
                    ExprPtr expanded_rhs;
                    if (s->compound_op.has_value()) {
                        expanded_rhs = make_binary(*s->compound_op, clone_expr(s->assign_target.get()),
                                                   clone_expr(s->assign_value.get()), s->assign_target->resolved_type, sp);
                    } else {
                        expanded_rhs = clone_expr(s->assign_value.get());
                    }
                    ExprPtr d_rhs = diff_expr(expanded_rhs.get());
                    auto dassign = std::make_unique<Stmt>();
                    dassign->kind = Stmt::Kind::Assign;
                    dassign->span = sp;
                    dassign->assign_target = std::move(d_target);
                    dassign->assign_value = std::move(d_rhs);
                    out.push_back(std::move(dassign));
                }
                out.push_back(clone_stmt(s));
                break;
            }
            case Stmt::Kind::If: {
                auto st = std::make_unique<Stmt>();
                st->kind = Stmt::Kind::If;
                st->span = sp;
                st->attributes = s->attributes;
                st->cond = clone_expr(s->cond.get());
                st->body = diff_stmt_list(s->body);
                st->else_body = diff_stmt_list(s->else_body);
                out.push_back(std::move(st));
                break;
            }
            case Stmt::Kind::ForRange: {
                auto st = std::make_unique<Stmt>();
                st->kind = Stmt::Kind::ForRange;
                st->span = sp;
                st->attributes = s->attributes;
                st->var_name = s->var_name;
                st->var_type = clone_type_expr(s->var_type.get());
                st->resolved_var_type = s->resolved_var_type;
                st->inclusive_range = s->inclusive_range;
                st->range_start = clone_expr(s->range_start.get());
                st->range_end = clone_expr(s->range_end.get());
                st->range_step = clone_expr(s->range_step.get());
                st->body = diff_stmt_list(s->body);
                out.push_back(std::move(st));
                break;
            }
            case Stmt::Kind::ForC: {
                auto st = std::make_unique<Stmt>();
                st->kind = Stmt::Kind::ForC;
                st->span = sp;
                st->attributes = s->attributes;
                if (s->for_init) {
                    if (s->for_init->kind == Stmt::Kind::Let || s->for_init->kind == Stmt::Kind::Var) {
                        const Type& vt = s->for_init->resolved_var_type;
                        if (is_diff_type(vt, ctx.out->types)) {
                            out.push_back(make_let_stmt("_d_" + s->for_init->var_name, vt,
                                s->for_init->init_expr ? diff_expr(s->for_init->init_expr.get()) : make_zero_expr(vt, ctx.out->types, sp),
                                sp, true));
                            diff_vars.insert(s->for_init->var_name);
                        }
                    }
                    st->for_init = clone_stmt(s->for_init.get());
                }
                st->cond = clone_expr(s->cond.get());
                st->body = diff_stmt_list(s->body);
                if (s->for_step) {
                    diff_stmt(s->for_step.get(), st->body);
                }
                out.push_back(std::move(st));
                break;
            }
            case Stmt::Kind::While: {
                auto st = std::make_unique<Stmt>();
                st->kind = Stmt::Kind::While;
                st->span = sp;
                st->attributes = s->attributes;
                st->cond = clone_expr(s->cond.get());
                st->body = diff_stmt_list(s->body);
                out.push_back(std::move(st));
                break;
            }
            case Stmt::Kind::Return: {
                if (s->expr) {
                    out.push_back(make_return_stmt(diff_expr(s->expr.get()), sp));
                } else {
                    out.push_back(clone_stmt(s));
                }
                break;
            }
            case Stmt::Kind::Break:
            case Stmt::Kind::Continue:
            case Stmt::Kind::Expr:
                out.push_back(clone_stmt(s));
                break;
        }
    }
};

} // namespace

std::string synthesize_grad_wrapper(
    SemaContext& ctx,
    const std::string& inst_name,
    const std::vector<CheckedParam>& base_params,
    const Type& x_type,
    const Type& f_ret,
    const SourceSpan& sp) {
    AdSynthesizer ad{ctx, {}};
    std::string jvp_name = ad.ensure_jvp_function(inst_name);

    std::string wrap_name = "__ad_grad_" + inst_name;
    if (!ctx.out->find_function(wrap_name)) {
        const CheckedFunction* orig = ctx.out->find_function(inst_name);
        CheckedFunction wrap;
        wrap.span = sp;
        wrap.name = wrap_name;
        wrap.source_name = wrap_name;
        wrap.module_name = orig->module_name;
        wrap.is_entry_kernel = false;
        wrap.is_pub = false;
        wrap.stage = orig->stage;
        wrap.required_stage = orig->required_stage;
        wrap.params = base_params;
        wrap.return_type = x_type;
        wrap.uses_f64 = orig->uses_f64;
        wrap.f64_site = orig->f64_site;
        wrap.uses_grid_coords = orig->uses_grid_coords;
        wrap.uses_descriptors = orig->uses_descriptors;
        wrap.uses_textures = orig->uses_textures;
        wrap.grid_dim = orig->grid_dim;

        auto build_jvp_call = [&](ExprPtr dx_expr) {
            std::vector<ExprPtr> jargs;
            jargs.push_back(make_ident(base_params[0].name, x_type, sp));
            jargs.push_back(std::move(dx_expr));
            for (size_t i = 1; i < base_params.size(); ++i) {
                jargs.push_back(make_ident(base_params[i].name, base_params[i].type, sp));
                if (is_diff_type(base_params[i].type, ctx.out->types)) {
                    jargs.push_back(make_zero_expr(base_params[i].type, ctx.out->types, sp));
                }
            }
            return make_call(jvp_name, std::move(jargs), f_ret, sp);
        };

        if (x_type.is_scalar()) {
            wrap.body.push_back(make_return_stmt(build_jvp_call(make_float_lit(1.0, x_type.scalar, sp)), sp));
        } else {
            std::vector<ExprPtr> comps;
            for (uint32_t c = 0; c < x_type.dimension; ++c) {
                std::vector<ExprPtr> basis_args;
                for (uint32_t j = 0; j < x_type.dimension; ++j) {
                    basis_args.push_back(make_float_lit(j == c ? 1.0 : 0.0, x_type.scalar, sp));
                }
                ExprPtr basis = make_call("vec" + std::to_string(x_type.dimension), std::move(basis_args), x_type, sp);
                std::string cname = "_g" + std::to_string(c);
                wrap.body.push_back(make_let_stmt(cname, f_ret, build_jvp_call(std::move(basis)), sp));
                comps.push_back(make_ident(cname, f_ret, sp));
            }
            wrap.body.push_back(make_return_stmt(make_call("vec" + std::to_string(x_type.dimension), std::move(comps), x_type, sp), sp));
        }
        size_t idx = ctx.out->functions.size();
        ctx.out->functions.push_back(std::move(wrap));
        ctx.fn_instances[wrap_name] = idx;
    }
    return wrap_name;
}

} // namespace brokernel::detail

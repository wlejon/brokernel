#include "sema_internal.h"

namespace brokernel {

TypeExprPtr clone_type_expr(const TypeExpr* t) {
    if (!t) return nullptr;
    auto c = std::make_unique<TypeExpr>();
    c->span = t->span;
    c->name = t->name;
    c->args.reserve(t->args.size());
    for (const auto& a : t->args) {
        GenericArg ga;
        ga.span = a.span;
        ga.type_arg = clone_type_expr(a.type_arg.get());
        ga.int_arg = a.int_arg;
        ga.ident_or_const_name = a.ident_or_const_name;
        c->args.push_back(std::move(ga));
    }
    return c;
}

ExprPtr clone_expr(const Expr* e) {
    if (!e) return nullptr;
    auto c = std::make_unique<Expr>();
    c->kind = e->kind;
    c->span = e->span;
    c->resolved_type = e->resolved_type;
    c->int_val = e->int_val;
    c->float_val = e->float_val;
    c->bool_val = e->bool_val;
    c->literal_suffix = e->literal_suffix;
    c->name = e->name;
    c->path = e->path;
    c->global_index = e->global_index;
    c->unary_op = e->unary_op;
    c->binary_op = e->binary_op;
    c->lhs = clone_expr(e->lhs.get());
    c->rhs = clone_expr(e->rhs.get());
    c->cond = clone_expr(e->cond.get());
    for (const auto& ga : e->generic_args) {
        GenericArg copy;
        copy.span = ga.span;
        copy.type_arg = clone_type_expr(ga.type_arg.get());
        copy.int_arg = ga.int_arg;
        copy.ident_or_const_name = ga.ident_or_const_name;
        c->generic_args.push_back(std::move(copy));
    }
    for (const auto& arg : e->args) c->args.push_back(clone_expr(arg.get()));
    for (const auto& f : e->fields) {
        FieldInit fi;
        fi.span = f.span;
        fi.name = f.name;
        fi.value = clone_expr(f.value.get());
        c->fields.push_back(std::move(fi));
    }
    c->target_type = clone_type_expr(e->target_type.get());
    return c;
}

StmtPtr clone_stmt(const Stmt* s) {
    if (!s) return nullptr;
    auto c = std::make_unique<Stmt>();
    c->kind = s->kind;
    c->span = s->span;
    c->attributes = s->attributes;
    c->var_name = s->var_name;
    c->var_type = clone_type_expr(s->var_type.get());
    c->init_expr = clone_expr(s->init_expr.get());
    c->resolved_var_type = s->resolved_var_type;
    c->is_shared = s->is_shared;
    c->assign_target = clone_expr(s->assign_target.get());
    c->assign_value = clone_expr(s->assign_value.get());
    c->compound_op = s->compound_op;
    c->cond = clone_expr(s->cond.get());
    for (const auto& b : s->body) c->body.push_back(clone_stmt(b.get()));
    for (const auto& eb : s->else_body) c->else_body.push_back(clone_stmt(eb.get()));
    c->inclusive_range = s->inclusive_range;
    c->range_start = clone_expr(s->range_start.get());
    c->range_end = clone_expr(s->range_end.get());
    c->range_step = clone_expr(s->range_step.get());
    c->for_init = clone_stmt(s->for_init.get());
    c->for_step = clone_stmt(s->for_step.get());
    c->expr = clone_expr(s->expr.get());
    return c;
}

namespace detail {

std::string synthesize_grad_wrapper(
    SemaContext& ctx,
    const std::string& inst_name,
    const std::vector<CheckedParam>& base_params,
    const Type& x_type,
    const Type& f_ret,
    const SourceSpan& sp);

Type FnCheckState::check_texture_method(Expr* e, const Type& recv_t) {
    const std::string& m = e->name;
    const bool image = recv_t.is_image();
    const bool cube = recv_t.image_cube();
    const bool shadow = recv_t.image_shadow();
    const Type elem = recv_t.element_type();
    const uint32_t n = recv_t.image_coords();
    const Type fcoord = Type::vec(ScalarKind::F32, n);
    const Type icoord = Type::vec(ScalarKind::I32, n);
    const Type space = Type::vec(ScalarKind::F32, recv_t.dimension);
    const Type offset = Type::vec(ScalarKind::I32, recv_t.dimension);
    const Type size = Type::vec(ScalarKind::I32, (cube ? 2 : recv_t.dimension) + (recv_t.image_arrayed() ? 1 : 0));

    auto done = [&](const Type& t) {
        e->resolved_type = t;
        return t;
    };
    const bool through = recv_t.image_separate() && m != "size" && m != "levels" && m != "fetch";
    auto takes = [&](std::initializer_list<Type> params, const char* shape) {
        if (e->args.size() != params.size() + (through ? 1 : 0)) {
            ctx.diags.error(e->span, recv_t.to_string() + "." + m + " expects (" + (through ? "sampler, " : "") +
                                         shape + ")");
            for (auto& a : e->args) check_expr(a.get());
            return;
        }
        size_t i = 0;
        if (through) {
            Expr* s = e->args[i++].get();
            Type st = check_expr(s);
            if (!st.is_sampler() || !st.sampler_only()) {
                ctx.diags.error(s->span, recv_t.to_string() + "." + m + " samples through a 'sampler', got '" +
                                             st.to_string() + "'");
            }
        }
        for (const Type& t : params) {
            Expr* a = e->args[i++].get();
            check_expr(a, t);
            check_assignable(a, t, ctx.diags, a->span);
        }
    };
    auto no_cube = [&] {
        if (cube) ctx.diags.error(e->span, recv_t.to_string() + "." + m + ": a cube has no texel offsets or fetches");
    };

    if (image) {
        if (m == "load") {
            if (!can_Read(recv_t.access)) ctx.diags.error(e->span, "cannot load from write-only image");
            takes({icoord}, "coordinates");
            return done(elem);
        }
        if (m == "store") {
            if (!can_Write(recv_t.access)) ctx.diags.error(e->span, "cannot store to read-only image");
            takes({icoord, elem}, "coordinates, texel");
            return done(Type::void_type());
        }
        if (m == "size") { takes({}, ""); return done(Type::vec(ScalarKind::I32, n)); }
    } else {
        if (m == "size") { takes({Type::i32()}, "lod"); return done(size); }
        if (m == "levels") { takes({}, ""); return done(Type::i32()); }
        if (shadow) {
            if (m == "compare") { takes({fcoord, Type::f32()}, "coordinates, reference"); return done(Type::f32()); }
            if (m == "compare_lod") { takes({fcoord, Type::f32(), Type::f32()}, "coordinates, reference, lod"); return done(Type::f32()); }
            if (m == "gather") { takes({fcoord, Type::f32()}, "coordinates, reference"); return done(Type::vec(ScalarKind::F32, 4)); }
            if (m == "fetch") { no_cube(); takes({icoord, Type::i32()}, "texel coordinates, lod"); return done(elem); }
        } else {
            if (m == "sample") { takes({fcoord}, "coordinates"); return done(elem); }
            if (m == "sample_offset") { no_cube(); takes({fcoord, offset}, "coordinates, offset"); return done(elem); }
            if (m == "sample_lod") { takes({fcoord, Type::f32()}, "coordinates, lod"); return done(elem); }
            if (m == "sample_lod_offset") { no_cube(); takes({fcoord, Type::f32(), offset}, "coordinates, lod, offset"); return done(elem); }
            if (m == "sample_grad") { takes({fcoord, space, space}, "coordinates, dx, dy"); return done(elem); }
            if (m == "sample_grad_offset") { no_cube(); takes({fcoord, space, space, offset}, "coordinates, dx, dy, offset"); return done(elem); }
            if (m == "fetch") { no_cube(); takes({icoord, Type::i32()}, "texel coordinates, lod"); return done(elem); }
            if (m == "gather") { takes({fcoord, Type::i32()}, "coordinates, component"); return done(Type::vec(elem.scalar, 4)); }
        }
    }
    for (auto& a : e->args) check_expr(a.get());
    ctx.diags.error(e->span, "type '" + recv_t.to_string() + "' has no method '" + m + "'");
    return done(Type::void_type());
}

namespace {

ExprPtr q_float_lit(double v, ScalarKind sk, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::FloatLit;
    e->span = sp;
    e->float_val = v;
    e->literal_suffix = sk;
    e->resolved_type = Type::scalar_type(sk);
    return e;
}

ExprPtr q_splat_or_scalar(double v, const Type& t, const SourceSpan& sp) {
    ScalarKind sk = (t.scalar == ScalarKind::F64) ? ScalarKind::F64 : ScalarKind::F32;
    ExprPtr lit = q_float_lit(v, sk, sp);
    if (t.is_scalar()) return lit;
    auto call = std::make_unique<Expr>();
    call->kind = Expr::Kind::Call;
    call->span = sp;
    call->name = "__splat";
    call->resolved_type = t;
    call->args.push_back(std::move(lit));
    return call;
}

ExprPtr q_ident(const std::string& name, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Ident;
    e->span = sp;
    e->name = name;
    e->resolved_type = t;
    return e;
}

ExprPtr q_binary(BinaryOp op, ExprPtr lhs, ExprPtr rhs, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Binary;
    e->span = sp;
    e->binary_op = op;
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    e->resolved_type = t;
    return e;
}

ExprPtr q_call(const std::string& name, std::vector<ExprPtr> args, const Type& ret_t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Call;
    e->span = sp;
    e->name = name;
    e->args = std::move(args);
    e->resolved_type = ret_t;
    return e;
}

StmtPtr q_let_stmt(const std::string& name, const Type& t, ExprPtr init, const SourceSpan& sp) {
    auto s = std::make_unique<Stmt>();
    s->kind = Stmt::Kind::Let;
    s->span = sp;
    s->var_name = name;
    s->resolved_var_type = t;
    s->init_expr = std::move(init);
    return s;
}

StmtPtr q_return_stmt(ExprPtr val, const SourceSpan& sp) {
    auto s = std::make_unique<Stmt>();
    s->kind = Stmt::Kind::Return;
    s->span = sp;
    s->expr = std::move(val);
    return s;
}

} // namespace

Type FnCheckState::check_ad_or_quadrature_call(Expr* e, std::optional<Type> expected) {
    const std::string& n = e->name;
    const SourceSpan& sp = e->span;

    // 1. Named probe: probe("label", val)
    if (n == "probe") {
        if (e->args.size() != 2 || e->args[0]->kind != Expr::Kind::StringLit) {
            ctx.diags.error(sp, "'probe(\"label\", expr)' expects a string literal label and a value expression");
            if (e->args.size() >= 2) return check_expr(e->args[1].get(), expected);
            return Type::f32();
        }
        const std::string& label = e->args[0]->name;
        Type vt = check_expr(e->args[1].get(), expected);
        if (is_unsuffixed_literal(e->args[1].get())) {
            coerce_expr_to_type(e->args[1].get(), expected && expected->is_scalar() ? *expected : Type::f32());
            vt = e->args[1]->resolved_type;
        }
        if (!vt.is_scalar() && !vt.is_vector()) {
            ctx.diags.error(e->args[1]->span, "probe value must be a scalar or vector, got '" + vt.to_string() + "'");
        }
        uint32_t probe_idx = 0;
        bool found_mod = false;
        for (const auto& p : ctx.out->probes) {
            if (p.name == label) {
                probe_idx = p.index;
                found_mod = true;
                break;
            }
        }
        if (!found_mod) {
            probe_idx = static_cast<uint32_t>(ctx.out->probes.size());
            ctx.out->probes.push_back(ProbeInfo{label, vt, probe_idx});
        }
        bool found_fn = false;
        for (const auto& p : fn.probes) {
            if (p.name == label) { found_fn = true; break; }
        }
        if (!found_fn) {
            fn.probes.push_back(ProbeInfo{label, vt, probe_idx});
        }
        e->args[0]->int_val = static_cast<int64_t>(probe_idx);
        e->args[0]->resolved_type = Type::i32();
        e->resolved_type = vt;
        return vt;
    }

    // 2. Higher-order constructs: grad, expect, moments, cell_expect
    size_t min_args = (n == "grad") ? 2 : (n == "cell_expect" ? 4 : 3);
    if (e->args.size() < min_args) {
        ctx.diags.error(sp, "'" + n + "' expects at least " + std::to_string(min_args) + " arguments");
        return Type::f32();
    }
    Expr* fn_arg = e->args[0].get();
    if (fn_arg->kind != Expr::Kind::Ident && fn_arg->kind != Expr::Kind::Path) {
        ctx.diags.error(fn_arg->span, "first argument to '" + n + "' must be a function name");
        return Type::f32();
    }
    std::string callee_mod;
    const FnDecl* fdecl = ctx.find_fn_decl(fn.module_name, fn_arg->path, fn_arg->name, &callee_mod);
    if (!fdecl) {
        ctx.diags.error(fn_arg->span, "unknown function '" + fn_arg->name + "' in '" + n + "'");
        return Type::f32();
    }

    size_t extra_start = (n == "grad") ? 2 : (n == "cell_expect" ? 4 : 3);
    for (size_t i = 1; i < e->args.size(); ++i) {
        check_expr(e->args[i].get());
    }
    if (is_unsuffixed_literal(e->args[1].get())) {
        Type default_t = (expected && expected->is_float()) ? Type::scalar_type(expected->scalar) : Type::f32();
        if (!fdecl->params.empty() && fdecl->generic_params.empty() && fdecl->params[0].type) {
            Type p0 = ctx.resolve_type(fdecl->params[0].type.get(), callee_mod, {});
            if (p0.is_float()) default_t = p0;
        }
        coerce_expr_to_type(e->args[1].get(), default_t);
    }
    Type x_type = e->args[1]->resolved_type;
    if (n == "expect" || n == "moments") {
        coerce_expr_to_type(e->args[2].get(), x_type);
    } else if (n == "cell_expect") {
        coerce_expr_to_type(e->args[2].get(), x_type);
        coerce_expr_to_type(e->args[3].get(), x_type);
    }

    Expr temp_call;
    temp_call.kind = Expr::Kind::Call;
    temp_call.span = sp;
    temp_call.name = fn_arg->name;
    temp_call.path = fn_arg->path;
    const auto& src_gargs = !fn_arg->generic_args.empty() ? fn_arg->generic_args : e->generic_args;
    for (const auto& ga : src_gargs) {
        GenericArg copy;
        copy.span = ga.span;
        copy.type_arg = clone_type_expr(ga.type_arg.get());
        copy.int_arg = ga.int_arg;
        copy.ident_or_const_name = ga.ident_or_const_name;
        temp_call.generic_args.push_back(std::move(copy));
    }
    temp_call.args.push_back(clone_expr(e->args[1].get()));
    for (size_t i = extra_start; i < e->args.size(); ++i) {
        temp_call.args.push_back(clone_expr(e->args[i].get()));
    }
    Type f_ret = check_user_call(&temp_call, *fdecl, callee_mod);
    if (ctx.diags.has_errors() || f_ret.is_void()) {
        return Type::f32();
    }
    e->args[1] = clone_expr(temp_call.args[0].get());
    x_type = e->args[1]->resolved_type;
    for (size_t i = extra_start; i < e->args.size(); ++i) {
        e->args[i] = clone_expr(temp_call.args[1 + (i - extra_start)].get());
    }
    if (n == "expect" || n == "moments") {
        check_assignable(e->args[2].get(), x_type, ctx.diags, e->args[2]->span);
    } else if (n == "cell_expect") {
        check_assignable(e->args[2].get(), x_type, ctx.diags, e->args[2]->span);
        check_assignable(e->args[3].get(), x_type, ctx.diags, e->args[3]->span);
    }

    const std::string inst_name = temp_call.name;
    const CheckedFunction* base_ptr = ctx.out->find_function(inst_name);
    if (!base_ptr) return Type::f32();
    std::vector<CheckedParam> base_params = base_ptr->params;

    if (!x_type.is_float()) {
        ctx.diags.error(e->args[1]->span, "'" + n + "' requires a float or float vector domain, got '" + x_type.to_string() + "'");
        return Type::f32();
    }

    if (n == "grad") {
        if (!f_ret.is_float_scalar() || f_ret.scalar != x_type.scalar) {
            ctx.diags.error(sp, "'grad' requires a scalar-returning function matching the domain scalar type, got '" + f_ret.to_string() + "'");
            return x_type;
        }
        std::string wrap_name = synthesize_grad_wrapper(ctx, inst_name, base_params, x_type, f_ret, sp);
        e->name = wrap_name;
        e->path.clear();
        e->generic_args.clear();
        e->args.erase(e->args.begin());
        e->resolved_type = x_type;
        return x_type;
    }

    // Quadrature synthesis: expect, moments, cell_expect
    if (!f_ret.is_float()) {
        ctx.diags.error(sp, "'" + n + "' requires a float or float-vector returning function, got '" + f_ret.to_string() + "'");
        return f_ret;
    }
    if (n == "moments" && !f_ret.is_float_scalar()) {
        ctx.diags.error(sp, "'moments' requires a scalar float-returning function, got '" + f_ret.to_string() + "'");
        return Type::vec(f_ret.scalar, 2);
    }

    Type out_type = (n == "moments") ? Type::vec(f_ret.scalar, 2) : f_ret;
    std::string quad_name = "__quad_" + n + "_" + inst_name;
    if (!ctx.out->find_function(quad_name)) {
        const CheckedFunction* orig = ctx.out->find_function(inst_name);
        CheckedFunction qfn;
        qfn.span = sp;
        qfn.name = quad_name;
        qfn.source_name = quad_name;
        qfn.module_name = orig->module_name;
        qfn.is_entry_kernel = false;
        qfn.is_pub = false;
        qfn.stage = orig->stage;
        qfn.required_stage = orig->required_stage;
        qfn.return_type = out_type;
        qfn.uses_f64 = orig->uses_f64;
        qfn.f64_site = orig->f64_site;
        qfn.uses_grid_coords = orig->uses_grid_coords;
        qfn.uses_descriptors = orig->uses_descriptors;
        qfn.uses_textures = orig->uses_textures;
        qfn.grid_dim = orig->grid_dim;

        CheckedParam p_center = base_params[0];
        p_center.name = "_q_center";
        qfn.params.push_back(p_center);
        if (n == "cell_expect") {
            CheckedParam p_du = p_center; p_du.name = "_q_du";
            CheckedParam p_dv = p_center; p_dv.name = "_q_dv";
            qfn.params.push_back(p_du);
            qfn.params.push_back(p_dv);
        } else {
            CheckedParam p_sig = p_center; p_sig.name = "_q_sigma";
            qfn.params.push_back(p_sig);
        }
        for (size_t i = 1; i < base_params.size(); ++i) {
            qfn.params.push_back(base_params[i]);
        }

        auto call_inst = [&](ExprPtr pt) {
            std::vector<ExprPtr> cargs;
            cargs.push_back(std::move(pt));
            for (size_t i = 1; i < base_params.size(); ++i) {
                cargs.push_back(q_ident(base_params[i].name, base_params[i].type, sp));
            }
            return q_call(inst_name, std::move(cargs), f_ret, sp);
        };

        if (n == "expect" || n == "moments") {
            ExprPtr step = q_binary(BinaryOp::Mul, q_ident("_q_sigma", x_type, sp),
                                    q_splat_or_scalar(1.7320508075688772935, x_type, sp), x_type, sp);
            qfn.body.push_back(q_let_stmt("_step", x_type, std::move(step), sp));
            qfn.body.push_back(q_let_stmt("_y0", f_ret, call_inst(q_ident("_q_center", x_type, sp)), sp));
            qfn.body.push_back(q_let_stmt("_ym", f_ret, call_inst(q_binary(BinaryOp::Sub, q_ident("_q_center", x_type, sp), q_ident("_step", x_type, sp), x_type, sp)), sp));
            qfn.body.push_back(q_let_stmt("_yp", f_ret, call_inst(q_binary(BinaryOp::Add, q_ident("_q_center", x_type, sp), q_ident("_step", x_type, sp), x_type, sp)), sp));

            ExprPtr w0 = q_splat_or_scalar(2.0 / 3.0, f_ret, sp);
            ExprPtr w1 = q_splat_or_scalar(1.0 / 6.0, f_ret, sp);
            ExprPtr mean_expr = q_binary(BinaryOp::Add,
                q_binary(BinaryOp::Mul, q_ident("_y0", f_ret, sp), std::move(w0), f_ret, sp),
                q_binary(BinaryOp::Mul, q_binary(BinaryOp::Add, q_ident("_ym", f_ret, sp), q_ident("_yp", f_ret, sp), f_ret, sp), std::move(w1), f_ret, sp),
                f_ret, sp);

            if (n == "expect") {
                qfn.body.push_back(q_return_stmt(std::move(mean_expr), sp));
            } else {
                qfn.body.push_back(q_let_stmt("_m", f_ret, std::move(mean_expr), sp));
                qfn.body.push_back(q_let_stmt("_d0", f_ret, q_binary(BinaryOp::Sub, q_ident("_y0", f_ret, sp), q_ident("_m", f_ret, sp), f_ret, sp), sp));
                qfn.body.push_back(q_let_stmt("_dm", f_ret, q_binary(BinaryOp::Sub, q_ident("_ym", f_ret, sp), q_ident("_m", f_ret, sp), f_ret, sp), sp));
                qfn.body.push_back(q_let_stmt("_dp", f_ret, q_binary(BinaryOp::Sub, q_ident("_yp", f_ret, sp), q_ident("_m", f_ret, sp), f_ret, sp), sp));
                ExprPtr sq0 = q_binary(BinaryOp::Mul, q_ident("_d0", f_ret, sp), q_ident("_d0", f_ret, sp), f_ret, sp);
                ExprPtr sqm = q_binary(BinaryOp::Mul, q_ident("_dm", f_ret, sp), q_ident("_dm", f_ret, sp), f_ret, sp);
                ExprPtr sqp = q_binary(BinaryOp::Mul, q_ident("_dp", f_ret, sp), q_ident("_dp", f_ret, sp), f_ret, sp);
                ExprPtr raw_var = q_binary(BinaryOp::Add,
                    q_binary(BinaryOp::Mul, std::move(sq0), q_splat_or_scalar(2.0 / 3.0, f_ret, sp), f_ret, sp),
                    q_binary(BinaryOp::Mul, q_binary(BinaryOp::Add, std::move(sqm), std::move(sqp), f_ret, sp), q_splat_or_scalar(1.0 / 6.0, f_ret, sp), f_ret, sp),
                    f_ret, sp);
                std::vector<ExprPtr> max_args;
                max_args.push_back(q_splat_or_scalar(0.0, f_ret, sp));
                max_args.push_back(std::move(raw_var));
                ExprPtr var_clamped = q_call("max", std::move(max_args), f_ret, sp);
                std::vector<ExprPtr> v2_args;
                v2_args.push_back(q_ident("_m", f_ret, sp));
                v2_args.push_back(std::move(var_clamped));
                qfn.body.push_back(q_return_stmt(q_call("vec2", std::move(v2_args), out_type, sp), sp));
            }
        } else {
            ExprPtr gl = q_splat_or_scalar(0.28867513459481288225, x_type, sp);
            qfn.body.push_back(q_let_stmt("_u", x_type, q_binary(BinaryOp::Mul, q_ident("_q_du", x_type, sp), clone_expr(gl.get()), x_type, sp), sp));
            qfn.body.push_back(q_let_stmt("_v", x_type, q_binary(BinaryOp::Mul, q_ident("_q_dv", x_type, sp), std::move(gl), x_type, sp), sp));
            auto pt_uv = [&](BinaryOp op_u, BinaryOp op_v) {
                ExprPtr cu = q_binary(op_u, q_ident("_q_center", x_type, sp), q_ident("_u", x_type, sp), x_type, sp);
                return q_binary(op_v, std::move(cu), q_ident("_v", x_type, sp), x_type, sp);
            };
            qfn.body.push_back(q_let_stmt("_y00", f_ret, call_inst(pt_uv(BinaryOp::Sub, BinaryOp::Sub)), sp));
            qfn.body.push_back(q_let_stmt("_y10", f_ret, call_inst(pt_uv(BinaryOp::Add, BinaryOp::Sub)), sp));
            qfn.body.push_back(q_let_stmt("_y01", f_ret, call_inst(pt_uv(BinaryOp::Sub, BinaryOp::Add)), sp));
            qfn.body.push_back(q_let_stmt("_y11", f_ret, call_inst(pt_uv(BinaryOp::Add, BinaryOp::Add)), sp));
            ExprPtr sum1 = q_binary(BinaryOp::Add, q_ident("_y00", f_ret, sp), q_ident("_y10", f_ret, sp), f_ret, sp);
            ExprPtr sum2 = q_binary(BinaryOp::Add, q_ident("_y01", f_ret, sp), q_ident("_y11", f_ret, sp), f_ret, sp);
            ExprPtr avg = q_binary(BinaryOp::Mul, q_binary(BinaryOp::Add, std::move(sum1), std::move(sum2), f_ret, sp),
                                   q_splat_or_scalar(0.25, f_ret, sp), f_ret, sp);
            qfn.body.push_back(q_return_stmt(std::move(avg), sp));
        }

        size_t idx = ctx.out->functions.size();
        ctx.out->functions.push_back(std::move(qfn));
        ctx.fn_instances[quad_name] = idx;
    }

    e->name = quad_name;
    e->path.clear();
    e->generic_args.clear();
    e->args.erase(e->args.begin());
    e->resolved_type = out_type;
    return out_type;
}

} // namespace detail
} // namespace brokernel

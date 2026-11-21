// Differentiable `@knob` support: `grad_knob(fn, KNOB, ...args)` and `CompileResult::compile_knob_grads`.
#include "sema_internal.h"
#include "brokernel/brokernel.h"

#include <algorithm>
#include <set>

namespace brokernel::detail {

namespace {

ExprPtr make_knob_lit(double v, ScalarKind sk, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::FloatLit;
    e->span = sp;
    e->float_val = v;
    e->literal_suffix = sk;
    e->resolved_type = Type::scalar_type(sk);
    return e;
}

ExprPtr make_knob_ident(const std::string& name, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Ident;
    e->span = sp;
    e->name = name;
    e->resolved_type = t;
    return e;
}

ExprPtr make_knob_call(const std::string& name, std::vector<ExprPtr> args, const Type& t, const SourceSpan& sp) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Kind::Call;
    e->span = sp;
    e->name = name;
    e->args = std::move(args);
    e->resolved_type = t;
    return e;
}

ExprPtr make_knob_splat_or_zero(double v, const Type& t, const TypeRegistry& types, const SourceSpan& sp) {
    if (t.is_float()) {
        ScalarKind sk = (t.scalar == ScalarKind::F64) ? ScalarKind::F64 : ScalarKind::F32;
        ExprPtr lit = make_knob_lit(v, sk, sp);
        if (t.is_scalar()) return lit;
        std::vector<ExprPtr> a;
        a.push_back(std::move(lit));
        return make_knob_call("__splat", std::move(a), t, sp);
    }
    if (t.is_bool()) {
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::BoolLit;
        e->span = sp;
        e->bool_val = false;
        e->resolved_type = t;
        return e;
    }
    if (t.is_integer()) {
        auto lit = std::make_unique<Expr>();
        lit->kind = Expr::Kind::IntLit;
        lit->span = sp;
        lit->int_val = 0;
        lit->literal_suffix = t.scalar;
        lit->resolved_type = Type::scalar_type(t.scalar);
        if (t.is_scalar()) return lit;
        std::vector<ExprPtr> a;
        a.push_back(std::move(lit));
        return make_knob_call("__splat", std::move(a), t, sp);
    }
    if (t.is_array()) {
        std::vector<ExprPtr> a;
        Type elem = t.element_type();
        for (uint32_t i = 0; i < t.dimension; ++i) a.push_back(make_knob_splat_or_zero(0.0, elem, types, sp));
        return make_knob_call("array", std::move(a), t, sp);
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
                fi.value = make_knob_splat_or_zero(0.0, f.type, types, sp);
                sl->fields.push_back(std::move(fi));
            }
        }
        return sl;
    }
    return make_knob_lit(0.0, ScalarKind::F32, sp);
}

std::string sanitize_ident(std::string s) {
    for (char& c : s) {
        if (c == ':' || c == '/' || c == '\\' || c == '<' || c == '>' || c == ',') c = '_';
    }
    return s;
}

} // namespace

Type FnCheckState::check_grad_knob_call(Expr* e, std::optional<Type>) {
    const SourceSpan& sp = e->span;
    if (e->args.size() < 2) {
        ctx.diags.error(sp, "'grad_knob' expects at least 2 arguments: grad_knob(fn, KNOB, ...args)");
        return Type::f32();
    }
    Expr* fn_arg = e->args[0].get();
    if (!fn_arg || (fn_arg->kind != Expr::Kind::Ident && fn_arg->kind != Expr::Kind::Path)) {
        ctx.diags.error(fn_arg ? fn_arg->span : sp, "first argument to 'grad_knob' must be a function name");
        return Type::f32();
    }
    std::string callee_mod;
    const FnDecl* fdecl = ctx.find_fn_decl(fn.module_name, fn_arg->path, fn_arg->name, &callee_mod);
    if (!fdecl) {
        ctx.diags.error(fn_arg->span, "unknown function '" + fn_arg->name + "' in 'grad_knob'");
        return Type::f32();
    }

    Expr* knob_arg = e->args[1].get();
    if (!knob_arg || (knob_arg->kind != Expr::Kind::Ident && knob_arg->kind != Expr::Kind::Path)) {
        ctx.diags.error(knob_arg ? knob_arg->span : sp, "second argument to 'grad_knob' must be a @knob constant");
        return Type::f32();
    }
    auto kkey = ctx.find_knob_key(fn.module_name, knob_arg->path, knob_arg->name);
    auto kcv = ctx.find_const(fn.module_name, knob_arg->path, knob_arg->name, genv, scopes);
    if (!kkey || !kcv) {
        ctx.diags.error(knob_arg->span, "'" + knob_arg->name + "' in 'grad_knob' is not a @knob constant");
        return Type::f32();
    }
    Type knob_type = kcv->type;
    if (!knob_type.is_float() || (!knob_type.is_scalar() && !knob_type.is_vector())) {
        ctx.diags.error(knob_arg->span, "'grad_knob' requires a float or float-vector @knob, got '" + knob_type.to_string() + "'");
        return Type::f32();
    }
    mark_type_usage(knob_type);

    for (size_t i = 2; i < e->args.size(); ++i) {
        check_expr(e->args[i].get());
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
    for (size_t i = 2; i < e->args.size(); ++i) {
        temp_call.args.push_back(clone_expr(e->args[i].get()));
    }
    Type f_ret = check_user_call(&temp_call, *fdecl, callee_mod);
    if (ctx.diags.has_errors() || f_ret.is_void()) {
        return knob_type;
    }
    if (!f_ret.is_float_scalar() || f_ret.scalar != knob_type.scalar) {
        ctx.diags.error(sp, "'grad_knob' requires a scalar float-returning function matching the @knob scalar type, got '" +
                                f_ret.to_string() + "'");
        return knob_type;
    }
    for (size_t i = 2; i < e->args.size(); ++i) {
        e->args[i] = clone_expr(temp_call.args[i - 2].get());
    }

    const std::string inst_name = temp_call.name;
    const CheckedFunction* base_ptr = ctx.out->find_function(inst_name);
    if (!base_ptr) return knob_type;
    std::vector<CheckedParam> base_params = base_ptr->params;

    const std::string suffix = sanitize_ident(*kkey);
    std::string jvp_name = synthesize_knob_jvp_function(ctx, inst_name, *kkey, knob_type, suffix);
    std::string wrap_name = "__ad_knob_grad_" + inst_name + "_" + suffix;

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
        wrap.return_type = knob_type;
        wrap.uses_f64 = orig->uses_f64 || knob_type.scalar == ScalarKind::F64;
        wrap.f64_site = orig->f64_site ? orig->f64_site : (knob_type.scalar == ScalarKind::F64 ? std::optional<SourceSpan>(sp) : std::nullopt);
        wrap.uses_grid_coords = orig->uses_grid_coords;
        wrap.uses_descriptors = orig->uses_descriptors;
        wrap.uses_textures = orig->uses_textures;
        wrap.grid_dim = orig->grid_dim;

        auto build_jvp_call = [&](ExprPtr dknob_expr) {
            std::vector<ExprPtr> jargs;
            for (const auto& bp : base_params) {
                jargs.push_back(make_knob_ident(bp.name, bp.type, sp));
                if (is_ad_diff_type(bp.type, ctx.out->types)) {
                    jargs.push_back(make_knob_splat_or_zero(0.0, bp.type, ctx.out->types, sp));
                }
            }
            jargs.push_back(std::move(dknob_expr));
            return make_knob_call(jvp_name, std::move(jargs), f_ret, sp);
        };

        auto ret_stmt = [&](ExprPtr val) {
            auto s = std::make_unique<Stmt>();
            s->kind = Stmt::Kind::Return;
            s->span = sp;
            s->expr = std::move(val);
            return s;
        };

        if (knob_type.is_scalar()) {
            wrap.body.push_back(ret_stmt(build_jvp_call(make_knob_lit(1.0, knob_type.scalar, sp))));
        } else {
            std::vector<ExprPtr> comps;
            for (uint32_t c = 0; c < knob_type.dimension; ++c) {
                std::vector<ExprPtr> bargs;
                for (uint32_t j = 0; j < knob_type.dimension; ++j) {
                    bargs.push_back(make_knob_lit(j == c ? 1.0 : 0.0, knob_type.scalar, sp));
                }
                ExprPtr basis = make_knob_call("vec" + std::to_string(knob_type.dimension), std::move(bargs), knob_type, sp);
                std::string cname = "_gk" + std::to_string(c);
                auto let_s = std::make_unique<Stmt>();
                let_s->kind = Stmt::Kind::Let;
                let_s->span = sp;
                let_s->var_name = cname;
                let_s->resolved_var_type = f_ret;
                let_s->init_expr = build_jvp_call(std::move(basis));
                wrap.body.push_back(std::move(let_s));
                comps.push_back(make_knob_ident(cname, f_ret, sp));
            }
            wrap.body.push_back(ret_stmt(make_knob_call("vec" + std::to_string(knob_type.dimension), std::move(comps), knob_type, sp)));
        }
        size_t idx = ctx.out->functions.size();
        ctx.out->functions.push_back(std::move(wrap));
        ctx.fn_instances[wrap_name] = idx;
    }

    e->name = wrap_name;
    e->path.clear();
    e->generic_args.clear();
    e->args.erase(e->args.begin(), e->args.begin() + 2);
    e->resolved_type = knob_type;
    return knob_type;
}

} // namespace brokernel::detail

namespace brokernel {

CompileResult CompileResult::compile_knob_grads(
    std::string_view fn_name,
    const std::vector<std::string>& knob_names) const {
    CompileResult out;
    if (knob_names.empty()) {
        out.diagnostics.error({file_name, 1, 1, 0, 0}, "compile_knob_grads: at least one knob name is required");
        return out;
    }

    // Inspect target function signature via check_module so parameter types match even if not an @kernel.
    Compiler inspect_comp;
    for (const auto& sp : search_paths) inspect_comp.add_search_path(sp);
    for (const auto& [mn, ms] : loaded_modules) inspect_comp.register_module(mn, ms);
    if (module_resolver) inspect_comp.set_module_resolver(module_resolver);
    DiagnosticBag inspect_diags;
    auto ast = parse_source(source, file_name, inspect_diags);
    if (!ast || inspect_diags.has_errors()) {
        out.diagnostics.merge_from(inspect_diags);
        return out;
    }
    std::filesystem::path root_dir = std::filesystem::path(file_name).parent_path();
    auto checked = check_module(*ast, inspect_comp.modules(), inspect_diags, root_dir, {},
                                options.knob_overrides.empty() ? nullptr : &options.knob_overrides,
                                options.runtime_knobs);
    if (!checked || inspect_diags.has_errors()) {
        out.diagnostics.merge_from(inspect_diags);
        return out;
    }
    const CheckedFunction* target_fn = checked->find_function(fn_name);
    if (!target_fn) {
        out.diagnostics.error({file_name, 1, 1, 0, 0}, "compile_knob_grads: function '" + std::string(fn_name) + "' not found");
        return out;
    }

    auto find_knob_info = [&](const std::string& name) -> const ConstInfo* {
        for (const auto& c : checked->exported_constants) {
            if (!c.is_knob) continue;
            if (c.name == name || (c.module + "::" + c.name) == name) return &c;
        }
        return nullptr;
    };

    std::string augmented = source;
    augmented += "\n";
    for (const std::string& kname : knob_names) {
        const ConstInfo* ki = find_knob_info(kname);
        if (!ki) {
            out.diagnostics.error({file_name, 1, 1, 0, 0}, "compile_knob_grads: @knob '" + kname + "' not found");
            return out;
        }
        std::string safe_k = detail::sanitize_ident(kname);
        std::string safe_fn = detail::sanitize_ident(std::string(fn_name));
        if (target_fn->cpu_only) augmented += "@cpu ";
        augmented += "@kernel pub fn grad_" + safe_fn + "_" + safe_k + "(";
        for (size_t i = 0; i < target_fn->params.size(); ++i) {
            if (i > 0) augmented += ", ";
            augmented += target_fn->params[i].name + ": " + target_fn->params[i].type.to_string();
        }
        augmented += ") -> " + ki->type.to_string() + " {\n    return grad_knob(" + std::string(fn_name) + ", " + kname;
        for (const auto& p : target_fn->params) {
            augmented += ", " + p.name;
        }
        augmented += ");\n}\n";
    }

    Compiler synth;
    for (const auto& sp : search_paths) synth.add_search_path(sp);
    for (const auto& [mn, ms] : loaded_modules) synth.register_module(mn, ms);
    if (module_resolver) synth.set_module_resolver(module_resolver);

    CompileOptions copts = options;
    copts.entry_name.clear();
    copts.compile_all_functions = false;
    CompileResult compiled = synth.compile_source(augmented, file_name, copts);
    if (!compiled.ok) return compiled;

    // Filter kernels to only the newly synthesized gradient kernels in knob_names order.
    std::vector<CompiledKernel> grad_kernels;
    for (const std::string& kname : knob_names) {
        std::string wname = "grad_" + detail::sanitize_ident(std::string(fn_name)) + "_" + detail::sanitize_ident(kname);
        if (const CompiledKernel* wk = compiled.find_kernel(wname)) {
            grad_kernels.push_back(*wk);
        }
    }
    compiled.kernels = std::move(grad_kernels);
    return compiled;
}

} // namespace brokernel

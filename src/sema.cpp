#include "sema_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace brokernel {

void CheckedParam::record_stencil_offset(int32_t dx, int32_t dy, int32_t dz) {
    StencilOffset off{dx, dy, dz};
    bool found = false;
    for (const auto& existing : stencil_offsets) {
        if (existing == off) {
            found = true;
            break;
        }
    }
    if (!found) {
        stencil_offsets.push_back(off);
    }
    const int32_t ax = std::abs(dx);
    const int32_t ay = std::abs(dy);
    const int32_t az = std::abs(dz);
    halo[0] = std::max(halo[0], ax);
    halo[1] = std::max(halo[1], ay);
    halo[2] = std::max(halo[2], az);
    inferred_radius = std::max({inferred_radius, ax, ay, az});
}

const CheckedFunction* CheckedModule::find_function(std::string_view name) const noexcept {
    for (const auto& fn : functions) {
        if (fn.name == name || fn.source_name == name) return &fn;
    }
    return nullptr;
}

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
    for (const auto& arg : e->args) {
        c->args.push_back(clone_expr(arg.get()));
    }
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

bool SemaContext::load_and_index_module(
    const ModuleAst& ast,
    const std::string& canonical_name,
    const std::filesystem::path& file_dir) {
    for (const auto& active_mod : active_import_stack) {
        if (active_mod == canonical_name) {
            diags.error(ast.span, "circular import involving '" + canonical_name + "'");
            return false;
        }
    }
    if (modules.find(canonical_name) != modules.end()) return true;

    active_import_stack.push_back(canonical_name);
    ModuleScope scope;
    scope.name = canonical_name;
    scope.ast = &ast;
    modules.emplace(canonical_name, std::move(scope));

    // Process imports first
    for (const auto& imp : ast.imports) {
        std::string target_key = !imp.file_path.empty()
                                     ? imp.file_path
                                     : ModuleRegistry::format_module_path(imp.module_path);
        const ModuleAst* imported = registry.load_module(target_key, imp.span, diags, file_dir);
        if (!imported) continue;

        std::string target_canon = !imported->module_name.empty() ? imported->module_name : target_key;
        std::filesystem::path next_dir = file_dir;
        if (!imported->file_path.empty()) {
            std::filesystem::path p(imported->file_path);
            if (p.has_parent_path()) next_dir = p.parent_path();
        }
        load_and_index_module(*imported, target_canon, next_dir);

        auto& cur_scope = modules[canonical_name];
        std::string local_mod_name = imp.alias;
        if (local_mod_name.empty()) {
            if (!imp.module_path.empty()) {
                local_mod_name = imp.module_path.back();
            } else {
                local_mod_name = target_canon;
            }
        }
        cur_scope.module_aliases[local_mod_name] = target_canon;
        cur_scope.module_aliases[target_canon] = target_canon;
        if (!imp.module_path.empty()) {
            cur_scope.module_aliases[ModuleRegistry::format_module_path(imp.module_path)] = target_canon;
        }

        if (imp.wildcard) {
            cur_scope.wildcard_modules.push_back(target_canon);
        }
        for (const auto& sym : imp.symbols) {
            // Verify symbol exists in imported module
            bool exists = false;
            for (const auto& fn : imported->functions) {
                if (fn.name == sym) { exists = true; break; }
            }
            if (!exists) {
                for (const auto& st : imported->structs) {
                    if (st.name == sym) { exists = true; break; }
                }
            }
            if (!exists) {
                for (const auto& c : imported->consts) {
                    if (c.name == sym) { exists = true; break; }
                }
            }
            if (!exists) {
                diags.error(imp.span, "symbol '" + sym + "' not found in module '" + target_canon + "'");
            } else {
                cur_scope.imported_symbols[sym] = {target_canon, sym};
            }
        }
    }

    active_import_stack.pop_back();
    module_order.push_back(canonical_name);
    return !diags.has_errors();
}

const FnDecl* SemaContext::find_fn_decl(
    const std::string& current_mod,
    const std::vector<std::string>& path,
    const std::string& name,
    std::string* resolved_mod_out) const {
    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return nullptr;
    const auto& scope = mod_it->second;

    if (!path.empty()) {
        // Path like `noise::fbm` or `std::noise::fbm`
        std::vector<std::string> mod_parts(path.begin(), path.end() - 1);
        const std::string& fn_name = path.back();
        std::string mod_key = ModuleRegistry::format_module_path(mod_parts);

        std::string target_mod;
        auto alias_it = scope.module_aliases.find(mod_key);
        if (alias_it != scope.module_aliases.end()) {
            target_mod = alias_it->second;
        } else if (mod_parts.size() == 1) {
            auto single_it = scope.module_aliases.find(mod_parts[0]);
            if (single_it != scope.module_aliases.end()) target_mod = single_it->second;
        }
        if (target_mod.empty()) return nullptr;
        auto tmod_it = modules.find(target_mod);
        if (tmod_it == modules.end() || !tmod_it->second.ast) return nullptr;
        for (const auto& fn : tmod_it->second.ast->functions) {
            if (fn.name == fn_name) {
                if (resolved_mod_out) *resolved_mod_out = target_mod;
                return &fn;
            }
        }
        return nullptr;
    }

    // 1. Current module
    if (scope.ast) {
        for (const auto& fn : scope.ast->functions) {
            if (fn.name == name) {
                if (resolved_mod_out) *resolved_mod_out = current_mod;
                return &fn;
            }
        }
    }
    // 2. Selective imports
    auto sym_it = scope.imported_symbols.find(name);
    if (sym_it != scope.imported_symbols.end()) {
        auto tmod_it = modules.find(sym_it->second.first);
        if (tmod_it != modules.end() && tmod_it->second.ast) {
            for (const auto& fn : tmod_it->second.ast->functions) {
                if (fn.name == sym_it->second.second) {
                    if (resolved_mod_out) *resolved_mod_out = sym_it->second.first;
                    return &fn;
                }
            }
        }
    }
    // 3. Wildcard imports
    for (const auto& wmod : scope.wildcard_modules) {
        auto tmod_it = modules.find(wmod);
        if (tmod_it != modules.end() && tmod_it->second.ast) {
            for (const auto& fn : tmod_it->second.ast->functions) {
                if (fn.name == name) {
                    if (resolved_mod_out) *resolved_mod_out = wmod;
                    return &fn;
                }
            }
        }
    }
    return nullptr;
}

const StructDecl* SemaContext::find_struct_decl(
    const std::string& current_mod,
    const std::string& name,
    std::string* resolved_mod_out) const {
    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return nullptr;
    const auto& scope = mod_it->second;

    // Check if `name` has `::`
    auto pos = name.rfind("::");
    if (pos != std::string::npos) {
        std::string mod_prefix = name.substr(0, pos);
        std::string st_name = name.substr(pos + 2);
        auto alias_it = scope.module_aliases.find(mod_prefix);
        if (alias_it != scope.module_aliases.end()) {
            auto tmod_it = modules.find(alias_it->second);
            if (tmod_it != modules.end() && tmod_it->second.ast) {
                for (const auto& st : tmod_it->second.ast->structs) {
                    if (st.name == st_name) {
                        if (resolved_mod_out) *resolved_mod_out = alias_it->second;
                        return &st;
                    }
                }
            }
        }
        return nullptr;
    }

    if (scope.ast) {
        for (const auto& st : scope.ast->structs) {
            if (st.name == name) {
                if (resolved_mod_out) *resolved_mod_out = current_mod;
                return &st;
            }
        }
    }
    auto sym_it = scope.imported_symbols.find(name);
    if (sym_it != scope.imported_symbols.end()) {
        auto tmod_it = modules.find(sym_it->second.first);
        if (tmod_it != modules.end() && tmod_it->second.ast) {
            for (const auto& st : tmod_it->second.ast->structs) {
                if (st.name == sym_it->second.second) {
                    if (resolved_mod_out) *resolved_mod_out = sym_it->second.first;
                    return &st;
                }
            }
        }
    }
    for (const auto& wmod : scope.wildcard_modules) {
        auto tmod_it = modules.find(wmod);
        if (tmod_it != modules.end() && tmod_it->second.ast) {
            for (const auto& st : tmod_it->second.ast->structs) {
                if (st.name == name) {
                    if (resolved_mod_out) *resolved_mod_out = wmod;
                    return &st;
                }
            }
        }
    }
    return nullptr;
}

std::optional<ConstValue> SemaContext::find_const(
    const std::string& current_mod,
    const std::vector<std::string>& path,
    const std::string& name,
    const GenericEnv& genv,
    const std::vector<std::unordered_map<std::string, LocalVar>>& scopes) const {
    if (path.empty()) {
        // 1. Local scopes (innermost to outermost)
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto vit = it->find(name);
            if (vit != it->end()) {
                if (vit->second.is_const) return vit->second.const_val;
                return std::nullopt;
            }
        }
        // 2. Generic const params
        auto git = genv.const_params.find(name);
        if (git != genv.const_params.end()) return git->second;

        // 3. Module-qualified or unqualified constants
        auto c_it = out->constants.find(current_mod + "::" + name);
        if (c_it != out->constants.end()) return c_it->second;

        auto mod_it = modules.find(current_mod);
        if (mod_it != modules.end()) {
            const auto& mscope = mod_it->second;
            auto sym_it = mscope.imported_symbols.find(name);
            if (sym_it != mscope.imported_symbols.end()) {
                auto cit2 = out->constants.find(sym_it->second.first + "::" + sym_it->second.second);
                if (cit2 != out->constants.end()) return cit2->second;
            }
            for (const auto& wmod : mscope.wildcard_modules) {
                auto cit3 = out->constants.find(wmod + "::" + name);
                if (cit3 != out->constants.end()) return cit3->second;
            }
        }
        auto plain_it = out->constants.find(name);
        if (plain_it != out->constants.end()) return plain_it->second;
        return std::nullopt;
    }

    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return std::nullopt;
    const auto& mscope = mod_it->second;
    std::vector<std::string> mod_parts(path.begin(), path.end() - 1);
    std::string mod_key = ModuleRegistry::format_module_path(mod_parts);
    std::string target_mod;
    auto alias_it = mscope.module_aliases.find(mod_key);
    if (alias_it != mscope.module_aliases.end()) {
        target_mod = alias_it->second;
    } else if (mod_parts.size() == 1) {
        auto single_it = mscope.module_aliases.find(mod_parts[0]);
        if (single_it != mscope.module_aliases.end()) target_mod = single_it->second;
    }
    if (target_mod.empty()) return std::nullopt;
    auto cit = out->constants.find(target_mod + "::" + path.back());
    if (cit != out->constants.end()) return cit->second;
    return std::nullopt;
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
            return cv;
        }
        case Expr::Kind::FloatLit: {
            ConstValue cv;
            cv.type = (e->literal_suffix != ScalarKind::Void) ? Type::scalar_type(e->literal_suffix) : Type::f64();
            cv.float_val = e->float_val;
            cv.int_val = static_cast<int64_t>(e->float_val);
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
            ConstValue res = *sub;
            if (e->unary_op == UnaryOp::Neg) {
                res.int_val = -sub->int_val;
                res.float_val = -sub->float_val;
            } else if (e->unary_op == UnaryOp::Not) {
                res.type = Type::bool_type();
                res.bool_val = !sub->bool_val;
                res.int_val = res.bool_val ? 1 : 0;
            } else if (e->unary_op == UnaryOp::BitNot) {
                res.int_val = ~sub->int_val;
            }
            return res;
        }
        case Expr::Kind::Binary: {
            auto l = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            auto r = eval_const_expr(e->rhs.get(), current_mod, genv, scopes);
            if (!l || !r) return std::nullopt;
            ConstValue res;
            bool is_fp = l->type.is_float() || r->type.is_float();
            res.type = is_fp ? (l->type.is_f64() || r->type.is_f64() ? Type::f64() : Type::f32()) : Type::i32();
            switch (e->binary_op) {
                case BinaryOp::Add:
                    res.int_val = l->int_val + r->int_val;
                    res.float_val = l->float_val + r->float_val;
                    return res;
                case BinaryOp::Sub:
                    res.int_val = l->int_val - r->int_val;
                    res.float_val = l->float_val - r->float_val;
                    return res;
                case BinaryOp::Mul:
                    res.int_val = l->int_val * r->int_val;
                    res.float_val = l->float_val * r->float_val;
                    return res;
                case BinaryOp::Div:
                    if (!is_fp && r->int_val == 0) return std::nullopt;
                    if (!is_fp) res.int_val = l->int_val / r->int_val;
                    res.float_val = l->float_val / r->float_val;
                    return res;
                case BinaryOp::Mod:
                    if (r->int_val == 0) return std::nullopt;
                    res.int_val = l->int_val % r->int_val;
                    return res;
                case BinaryOp::Eq:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val == r->float_val) : (l->int_val == r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::Ne:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val != r->float_val) : (l->int_val != r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::Lt:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val < r->float_val) : (l->int_val < r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::Le:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val <= r->float_val) : (l->int_val <= r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::Gt:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val > r->float_val) : (l->int_val > r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::Ge:
                    res.type = Type::bool_type();
                    res.bool_val = is_fp ? (l->float_val >= r->float_val) : (l->int_val >= r->int_val);
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::LogAnd:
                    res.type = Type::bool_type();
                    res.bool_val = l->bool_val && r->bool_val;
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                case BinaryOp::LogOr:
                    res.type = Type::bool_type();
                    res.bool_val = l->bool_val || r->bool_val;
                    res.int_val = res.bool_val ? 1 : 0;
                    return res;
                default:
                    return std::nullopt;
            }
        }
        case Expr::Kind::Cast: {
            auto sub = eval_const_expr(e->lhs.get(), current_mod, genv, scopes);
            if (!sub) return std::nullopt;
            Type target = resolve_type(e->target_type.get(), current_mod, genv);
            ConstValue res = *sub;
            res.type = target;
            if (target.is_integer()) {
                res.int_val = sub->type.is_float() ? static_cast<int64_t>(sub->float_val) : sub->int_val;
                res.float_val = static_cast<double>(res.int_val);
            } else if (target.is_float()) {
                res.float_val = sub->type.is_float() ? sub->float_val : static_cast<double>(sub->int_val);
                res.int_val = static_cast<int64_t>(res.float_val);
            }
            return res;
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

static std::optional<int64_t> resolve_generic_int_arg(
    const GenericArg& arg,
    const std::string& current_mod,
    const GenericEnv& genv,
    SemaContext& ctx) {
    if (arg.int_arg.has_value()) return *arg.int_arg;
    if (!arg.ident_or_const_name.empty()) {
        auto git = genv.const_params.find(arg.ident_or_const_name);
        if (git != genv.const_params.end()) return git->second.int_val;
        auto cv = ctx.find_const(current_mod, {}, arg.ident_or_const_name, genv, {});
        if (cv && cv->type.is_integer()) return cv->int_val;
    }
    return std::nullopt;
}

Type SemaContext::resolve_type(
    const TypeExpr* texpr,
    const std::string& current_mod,
    const GenericEnv& genv) {
    if (!texpr) return Type::void_type();
    const std::string& n = texpr->name;

    // 1. Check generic type parameters first
    if (texpr->args.empty()) {
        auto git = genv.type_params.find(n);
        if (git != genv.type_params.end()) return git->second;
    }

    // 2. Built-in scalars
    if (n == "void") return Type::void_type();
    if (n == "bool") return Type::bool_type();
    if (n == "i32" || n == "int") return Type::i32();
    if (n == "u32" || n == "uint") return Type::u32();
    if (n == "i64") return Type::i64();
    if (n == "u64") return Type::u64();
    if (n == "f32" || n == "float") return Type::f32();
    if (n == "f64" || n == "double") return Type::f64();

    // 3. Built-in vector shorthands
    if (n == "vec2f") return Type::vec(ScalarKind::F32, 2);
    if (n == "vec3f") return Type::vec(ScalarKind::F32, 3);
    if (n == "vec4f") return Type::vec(ScalarKind::F32, 4);
    if (n == "vec2d") return Type::vec(ScalarKind::F64, 2);
    if (n == "vec3d") return Type::vec(ScalarKind::F64, 3);
    if (n == "vec4d") return Type::vec(ScalarKind::F64, 4);
    if (n == "vec2i" || n == "ivec2") return Type::vec(ScalarKind::I32, 2);
    if (n == "vec3i" || n == "ivec3") return Type::vec(ScalarKind::I32, 3);
    if (n == "vec4i" || n == "ivec4") return Type::vec(ScalarKind::I32, 4);
    if (n == "vec2u" || n == "uvec2") return Type::vec(ScalarKind::U32, 2);
    if (n == "vec3u" || n == "uvec3") return Type::vec(ScalarKind::U32, 3);
    if (n == "vec4u" || n == "uvec4") return Type::vec(ScalarKind::U32, 4);

    if (n == "vec2" || n == "vec3" || n == "vec4") {
        uint32_t lanes = static_cast<uint32_t>(n.back() - '0');
        ScalarKind sk = ScalarKind::F32;
        if (!texpr->args.empty() && texpr->args[0].type_arg) {
            Type elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
            if (!elem.is_scalar()) {
                diags.error(texpr->span, "vector element type must be a scalar");
                return Type::vec(ScalarKind::F32, lanes);
            }
            sk = elem.scalar;
        }
        return Type::vec(sk, lanes);
    }

    if (n == "vec") {
        if (texpr->args.size() != 2 || !texpr->args[0].type_arg) {
            diags.error(texpr->span, "'vec<T, D>' requires a scalar type T and dimension D (1..4)");
            return Type::vec(ScalarKind::F32, 3);
        }
        Type elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        auto dim_opt = resolve_generic_int_arg(texpr->args[1], current_mod, genv, *this);
        if (!dim_opt || *dim_opt < 1 || *dim_opt > 4) {
            diags.error(texpr->span, "'vec<T, D>' dimension D must be a compile-time integer in 1..4");
            return Type::vec(elem.scalar, 3);
        }
        return Type::vec(elem.scalar, static_cast<uint32_t>(*dim_opt));
    }

    // 4. Fixed array: `array<T, N>`
    if (n == "array") {
        if (texpr->args.size() != 2 || !texpr->args[0].type_arg) {
            diags.error(texpr->span, "'array<T, N>' requires element type T and length N");
            return Type::array_of(Type::f32(), 1);
        }
        Type elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        auto len_opt = resolve_generic_int_arg(texpr->args[1], current_mod, genv, *this);
        if (!len_opt || *len_opt <= 0) {
            diags.error(texpr->span, "'array<T, N>' length N must be > 0");
            return Type::array_of(elem, 1);
        }
        return Type::array_of(elem, static_cast<uint32_t>(*len_opt));
    }

    // 5. Execution domain: `grid<D>` or `grid1d`, `grid2d`, `grid3d`
    if (n == "grid" || n == "grid1d" || n == "grid2d" || n == "grid3d") {
        uint32_t dim = 2;
        if (n == "grid1d") dim = 1;
        else if (n == "grid2d") dim = 2;
        else if (n == "grid3d") dim = 3;
        else if (!texpr->args.empty()) {
            auto d_opt = resolve_generic_int_arg(texpr->args[0], current_mod, genv, *this);
            if (!d_opt || *d_opt < 1 || *d_opt > 3) {
                diags.error(texpr->span, "'grid<D>' dimension D must be 1, 2, or 3");
            } else {
                dim = static_cast<uint32_t>(*d_opt);
            }
        }
        return Type::grid(dim);
    }

    // 6. Stencils: `stencil<T, D[, R]>`, `stencil1d<T[, R]>`, `stencil2d<T[, R]>`, `stencil3d<T[, R]>`
    if (n == "stencil" || n == "stencil1d" || n == "stencil2d" || n == "stencil3d") {
        if (texpr->args.empty() || !texpr->args[0].type_arg) {
            diags.error(texpr->span, "'" + n + "' requires an element type parameter");
            return Type::stencil(Type::f32(), 2, -1);
        }
        Type elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        uint32_t dim = 2;
        int32_t radius = -1;
        size_t rad_idx = 1;
        if (n == "stencil") {
            if (texpr->args.size() < 2) {
                diags.error(texpr->span, "'stencil<T, D[, R]>' requires dimension D (1..3)");
                return Type::stencil(elem, 2, -1);
            }
            auto d_opt = resolve_generic_int_arg(texpr->args[1], current_mod, genv, *this);
            if (!d_opt || *d_opt < 1 || *d_opt > 3) {
                diags.error(texpr->span, "stencil dimension D must be 1, 2, or 3");
            } else {
                dim = static_cast<uint32_t>(*d_opt);
            }
            rad_idx = 2;
        } else {
            dim = static_cast<uint32_t>(n[7] - '0');
        }
        if (texpr->args.size() > rad_idx) {
            auto r_opt = resolve_generic_int_arg(texpr->args[rad_idx], current_mod, genv, *this);
            if (!r_opt || *r_opt < -1) {
                diags.error(texpr->span, "stencil radius R must be a non-negative compile-time integer");
            } else {
                radius = static_cast<int32_t>(*r_opt);
            }
        }
        return Type::stencil(elem, dim, radius);
    }

    // 7. Stores: `store<T, D>`, `read_store<T, D>`, `write_store<T, D>`, `rw_store<T, D>`, etc.
    if (n == "store" || n == "read_store" || n == "write_store" || n == "rw_store" ||
        n == "rstore" || n == "wstore" ||
        n == "store1d" || n == "store2d" || n == "store3d" ||
        n == "read_store1d" || n == "read_store2d" || n == "read_store3d" ||
        n == "write_store1d" || n == "write_store2d" || n == "write_store3d") {
        AccessMode mode = AccessMode::ReadWrite;
        if (n.rfind("read_", 0) == 0 || n == "rstore") mode = AccessMode::Read;
        else if (n.rfind("write_", 0) == 0 || n == "wstore") mode = AccessMode::Write;

        if (texpr->args.empty() || !texpr->args[0].type_arg) {
            diags.error(texpr->span, "'" + n + "' requires an element type parameter");
            return Type::store(Type::f32(), 2, mode);
        }
        Type elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        uint32_t dim = 2;
        if (n.size() >= 2 && n.back() == 'd' && (n[n.size() - 2] >= '1' && n[n.size() - 2] <= '3')) {
            dim = static_cast<uint32_t>(n[n.size() - 2] - '0');
        } else if (texpr->args.size() >= 2) {
            auto d_opt = resolve_generic_int_arg(texpr->args[1], current_mod, genv, *this);
            if (!d_opt || *d_opt < 1 || *d_opt > 3) {
                diags.error(texpr->span, "store dimension D must be 1, 2, or 3");
            } else {
                dim = static_cast<uint32_t>(*d_opt);
            }
        } else {
            dim = 1; // `store<T>` defaults to 1D if D omitted
        }
        return Type::store(elem, dim, mode);
    }

    // 8. Images & Samplers
    if (n == "image" || n == "image1d" || n == "image2d" || n == "image3d" ||
        n == "read_image2d" || n == "write_image2d") {
        AccessMode mode = AccessMode::ReadWrite;
        if (n.rfind("read_", 0) == 0) mode = AccessMode::Read;
        else if (n.rfind("write_", 0) == 0) mode = AccessMode::Write;
        Type elem = Type::vec(ScalarKind::F32, 4);
        if (!texpr->args.empty() && texpr->args[0].type_arg) {
            elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        }
        uint32_t dim = 2;
        if (n.find("1d") != std::string::npos) dim = 1;
        else if (n.find("3d") != std::string::npos) dim = 3;
        else if (n == "image" && texpr->args.size() >= 2) {
            auto d_opt = resolve_generic_int_arg(texpr->args[1], current_mod, genv, *this);
            if (d_opt && *d_opt >= 1 && *d_opt <= 3) dim = static_cast<uint32_t>(*d_opt);
        }
        return Type::image(elem, dim, mode);
    }

    if (n == "sampler" || n == "sampler1d" || n == "sampler2d" || n == "sampler3d") {
        Type elem = Type::vec(ScalarKind::F32, 4);
        if (!texpr->args.empty() && texpr->args[0].type_arg) {
            elem = resolve_type(texpr->args[0].type_arg.get(), current_mod, genv);
        }
        uint32_t dim = 2;
        if (n == "sampler1d") dim = 1;
        else if (n == "sampler3d") dim = 3;
        return Type::sampler(elem, dim);
    }

    // 9. User-defined or imported Struct (possibly generic)
    std::string st_mod;
    const StructDecl* sdecl = find_struct_decl(current_mod, n, &st_mod);
    if (sdecl) {
        GenericEnv st_env;
        std::string mangled = sdecl->name;
        if (!sdecl->generic_params.empty()) {
            if (texpr->args.size() > sdecl->generic_params.size()) {
                diags.error(texpr->span, "too many generic arguments for struct '" + sdecl->name + "'");
                return Type::void_type();
            }
            mangled += "<";
            for (size_t i = 0; i < sdecl->generic_params.size(); ++i) {
                if (i > 0) mangled += ",";
                const auto& gp = sdecl->generic_params[i];
                if (gp.is_const) {
                    int64_t val = 0;
                    if (i < texpr->args.size()) {
                        auto vopt = resolve_generic_int_arg(texpr->args[i], current_mod, genv, *this);
                        if (!vopt) {
                            diags.error(texpr->span, "expected compile-time integer for generic parameter '" + gp.name + "'");
                        } else {
                            val = *vopt;
                        }
                    } else if (gp.default_int.has_value()) {
                        val = *gp.default_int;
                    } else {
                        diags.error(texpr->span, "missing generic argument for '" + gp.name + "'");
                    }
                    ConstValue cv;
                    cv.type = Type::i32();
                    cv.int_val = val;
                    st_env.const_params[gp.name] = cv;
                    mangled += std::to_string(val);
                } else {
                    Type targ = Type::f32();
                    if (i < texpr->args.size() && texpr->args[i].type_arg) {
                        targ = resolve_type(texpr->args[i].type_arg.get(), current_mod, genv);
                    } else if (gp.default_type) {
                        targ = resolve_type(gp.default_type.get(), st_mod, st_env);
                    } else {
                        diags.error(texpr->span, "missing generic type argument for '" + gp.name + "'");
                    }
                    st_env.type_params[gp.name] = targ;
                    mangled += targ.to_string();
                }
            }
            mangled += ">";
        }
        if (const StructDef* existing = out->types.find_struct(mangled)) {
            return Type::make_struct(existing->id, existing->name);
        }
        StructDef def;
        def.name = mangled;
        for (const auto& fdecl : sdecl->fields) {
            StructField sf;
            sf.name = fdecl.name;
            sf.type = resolve_type(fdecl.type.get(), st_mod, st_env);
            def.fields.push_back(std::move(sf));
        }
        uint32_t id = out->types.register_struct(std::move(def));
        return Type::make_struct(id, mangled);
    }

    diags.error(texpr->span, "unknown type '" + n + "'");
    return Type::void_type();
}

} // namespace detail

std::unique_ptr<CheckedModule> check_module(
    const ModuleAst& root_ast,
    ModuleRegistry& registry,
    DiagnosticBag& diags,
    const std::filesystem::path& root_dir) {
    detail::SemaContext ctx(registry, diags, root_dir);
    std::string root_name = !root_ast.module_name.empty() ? root_ast.module_name : "root";
    ctx.out->module_name = root_name;

    ctx.load_and_index_module(root_ast, root_name, root_dir);
    if (diags.has_errors()) return std::move(ctx.out);

    // Evaluate module-level constants and non-generic structs in dependency order
    detail::GenericEnv empty_env;
    for (const auto& mod_name : ctx.module_order) {
        const ModuleAst* ast = ctx.modules[mod_name].ast;
        if (!ast) continue;

        for (const auto& const_decl : ast->consts) {
            auto cv_opt = ctx.eval_const_expr(const_decl.value.get(), mod_name, empty_env, {});
            if (!cv_opt) {
                diags.error(const_decl.span, "constant '" + const_decl.name + "' initializer must be a compile-time constant expression");
                continue;
            }
            ConstValue cv = *cv_opt;
            if (const_decl.type) {
                Type expected = ctx.resolve_type(const_decl.type.get(), mod_name, empty_env);
                cv.type = expected;
            }
            ctx.out->constants[mod_name + "::" + const_decl.name] = cv;
            if (mod_name == root_name) {
                ctx.out->constants[const_decl.name] = cv;
            }
        }

        for (const auto& sdecl : ast->structs) {
            if (!sdecl.generic_params.empty()) continue; // Instantiated on demand
            TypeExpr te;
            te.span = sdecl.span;
            te.name = sdecl.name;
            ctx.resolve_type(&te, mod_name, empty_env);
        }
    }

    // Type-check all non-generic functions in the root module (kernels first, plus non-generic functions)
    for (const auto& fn : root_ast.functions) {
        if (!fn.generic_params.empty()) {
            if (fn.is_entry_kernel()) {
                diags.error(fn.span, "entry kernel '" + fn.name + "' cannot have generic parameters");
            }
            continue;
        }
        ctx.instantiate_function(fn, root_name, empty_env, fn.span, fn.is_entry_kernel());
    }

    return std::move(ctx.out);
}

} // namespace brokernel

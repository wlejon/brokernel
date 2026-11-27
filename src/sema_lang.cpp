#include "sema_internal.h"

#include <climits>
#include <cstdint>

namespace brokernel::detail {

const EnumDecl* SemaContext::find_enum_in_module(
    const std::string& mod_name,
    const std::string& sym,
    std::string* resolved_mod_out,
    std::set<std::string>& visited) const {
    if (!visited.insert(mod_name).second) return nullptr;
    auto it = modules.find(mod_name);
    if (it == modules.end() || !it->second.ast) return nullptr;
    const auto& scope = it->second;
    if (auto eit = scope.enum_by_name.find(sym); eit != scope.enum_by_name.end()) {
        if (resolved_mod_out) *resolved_mod_out = mod_name;
        return eit->second;
    }
    auto pit = scope.pub_imported_symbols.find(sym);
    if (pit != scope.pub_imported_symbols.end()) {
        return find_enum_in_module(pit->second.first, pit->second.second, resolved_mod_out, visited);
    }
    for (const auto& wmod : scope.pub_wildcard_modules) {
        if (const EnumDecl* ed = find_enum_in_module(wmod, sym, resolved_mod_out, visited)) return ed;
    }
    return nullptr;
}

const EnumDecl* SemaContext::find_enum_decl(
    const std::string& current_mod,
    const std::string& name,
    std::string* resolved_mod_out) const {
    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return nullptr;
    const auto& scope = mod_it->second;

    auto pos = name.rfind("::");
    if (pos != std::string::npos) {
        std::string mod_prefix = name.substr(0, pos);
        std::string sym_name = name.substr(pos + 2);
        auto alias_it = scope.module_aliases.find(mod_prefix);
        if (alias_it != scope.module_aliases.end()) {
            std::set<std::string> visited;
            return find_enum_in_module(alias_it->second, sym_name, resolved_mod_out, visited);
        }
        return nullptr;
    }

    if (auto eit = scope.enum_by_name.find(name); eit != scope.enum_by_name.end()) {
        if (resolved_mod_out) *resolved_mod_out = current_mod;
        return eit->second;
    }
    auto sym_it = scope.imported_symbols.find(name);
    if (sym_it != scope.imported_symbols.end()) {
        std::set<std::string> visited;
        if (const EnumDecl* ed = find_enum_in_module(sym_it->second.first, sym_it->second.second, resolved_mod_out, visited)) {
            return ed;
        }
    }
    for (const auto& wmod : scope.wildcard_modules) {
        std::set<std::string> visited;
        if (const EnumDecl* ed = find_enum_in_module(wmod, name, resolved_mod_out, visited)) {
            return ed;
        }
    }
    return nullptr;
}

const NewtypeDecl* SemaContext::find_newtype_in_module(
    const std::string& mod_name,
    const std::string& sym,
    std::string* resolved_mod_out,
    std::set<std::string>& visited) const {
    if (!visited.insert(mod_name).second) return nullptr;
    auto it = modules.find(mod_name);
    if (it == modules.end() || !it->second.ast) return nullptr;
    const auto& scope = it->second;
    if (auto nit = scope.newtype_by_name.find(sym); nit != scope.newtype_by_name.end()) {
        if (resolved_mod_out) *resolved_mod_out = mod_name;
        return nit->second;
    }
    auto pit = scope.pub_imported_symbols.find(sym);
    if (pit != scope.pub_imported_symbols.end()) {
        return find_newtype_in_module(pit->second.first, pit->second.second, resolved_mod_out, visited);
    }
    for (const auto& wmod : scope.pub_wildcard_modules) {
        if (const NewtypeDecl* nt = find_newtype_in_module(wmod, sym, resolved_mod_out, visited)) return nt;
    }
    return nullptr;
}

const NewtypeDecl* SemaContext::find_newtype_decl(
    const std::string& current_mod,
    const std::string& name,
    std::string* resolved_mod_out) const {
    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return nullptr;
    const auto& scope = mod_it->second;

    auto pos = name.rfind("::");
    if (pos != std::string::npos) {
        std::string mod_prefix = name.substr(0, pos);
        std::string sym_name = name.substr(pos + 2);
        auto alias_it = scope.module_aliases.find(mod_prefix);
        if (alias_it != scope.module_aliases.end()) {
            std::set<std::string> visited;
            return find_newtype_in_module(alias_it->second, sym_name, resolved_mod_out, visited);
        }
        return nullptr;
    }

    if (auto nit = scope.newtype_by_name.find(name); nit != scope.newtype_by_name.end()) {
        if (resolved_mod_out) *resolved_mod_out = current_mod;
        return nit->second;
    }
    auto sym_it = scope.imported_symbols.find(name);
    if (sym_it != scope.imported_symbols.end()) {
        std::set<std::string> visited;
        if (const NewtypeDecl* nt = find_newtype_in_module(sym_it->second.first, sym_it->second.second, resolved_mod_out, visited)) {
            return nt;
        }
    }
    for (const auto& wmod : scope.wildcard_modules) {
        std::set<std::string> visited;
        if (const NewtypeDecl* nt = find_newtype_in_module(wmod, name, resolved_mod_out, visited)) {
            return nt;
        }
    }
    return nullptr;
}

std::optional<ConstValue> SemaContext::find_enum_variant_const(
    const std::string& current_mod,
    const std::vector<std::string>& path,
    const std::string& name) const {
    if (path.size() < 2) return std::nullopt;
    const std::string& variant_name = path.back();
    const std::string& enum_name = path[path.size() - 2];
    (void)name;
    std::string enum_mod;
    const EnumDecl* ed = nullptr;
    if (path.size() == 2) {
        ed = find_enum_decl(current_mod, enum_name, &enum_mod);
    } else {
        auto mod_it = modules.find(current_mod);
        if (mod_it == modules.end()) return std::nullopt;
        const auto& mscope = mod_it->second;
        std::vector<std::string> mod_parts(path.begin(), path.end() - 2);
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
        std::set<std::string> visited;
        ed = find_enum_in_module(target_mod, enum_name, &enum_mod, visited);
    }
    if (!ed) return std::nullopt;
    auto cit = out->constants.find(enum_mod + "::" + ed->name + "::" + variant_name);
    if (cit != out->constants.end()) return cit->second;
    return std::nullopt;
}

std::optional<Type> SemaContext::resolve_enum_or_newtype_type(
    const TypeExpr* texpr,
    const std::string& current_mod) {
    if (!texpr) return std::nullopt;
    const std::string& n = texpr->name;

    std::string enum_mod;
    if (const EnumDecl* ed = find_enum_decl(current_mod, n, &enum_mod)) {
        if (!texpr->args.empty()) {
            diags.error(texpr->span, "enum '" + ed->name + "' does not take generic arguments");
        }
        std::string mangled = (enum_mod == out->module_name) ? ed->name : (enum_mod + "::" + ed->name);
        if (const EnumDef* existing = out->types.find_enum(mangled)) {
            return Type::make_enum(existing->id, existing->name, existing->underlying);
        }
        ScalarKind sk = ScalarKind::I32;
        if (auto it = enum_info_by_key.find(enum_mod + "::" + ed->name); it != enum_info_by_key.end()) {
            sk = it->second.underlying;
        } else if (ed->underlying && (ed->underlying->name == "u32" || ed->underlying->name == "uint")) {
            sk = ScalarKind::U32;
        }
        uint32_t id = out->types.register_enum(EnumDef{0, mangled, sk});
        return Type::make_enum(id, mangled, sk);
    }

    std::string nt_mod;
    if (const NewtypeDecl* nt = find_newtype_decl(current_mod, n, &nt_mod)) {
        if (!texpr->args.empty()) {
            diags.error(texpr->span, "newtype '" + nt->name + "' does not take generic arguments");
        }
        std::string mangled = (nt_mod == out->module_name) ? nt->name : (nt_mod + "::" + nt->name);
        if (const NewtypeDef* existing = out->types.find_newtype(mangled)) {
            return Type::make_newtype(existing->id, existing->name, existing->underlying);
        }
        for (const auto& active : active_newtype_stack) {
            if (active == mangled) {
                diags.error(texpr->span, "cyclic newtype definition involving '" + nt->name + "'");
                return Type::void_type();
            }
        }
        active_newtype_stack.push_back(mangled);
        GenericEnv empty_env;
        Type underlying = resolve_type(nt->target.get(), nt_mod, empty_env);
        active_newtype_stack.pop_back();
        if (underlying.is_void() || underlying.is_resource()) {
            diags.error(nt->span, "newtype '" + nt->name + "' cannot wrap '" + underlying.to_string() + "'");
            return Type::void_type();
        }
        uint32_t id = out->types.register_newtype(NewtypeDef{0, mangled, underlying});
        return Type::make_newtype(id, mangled, underlying);
    }

    return std::nullopt;
}

void SemaContext::evaluate_module_enums(const std::string& mod_name, const std::string& root_name) {
    auto mod_it = modules.find(mod_name);
    if (mod_it == modules.end() || !mod_it->second.ast) return;
    const ModuleAst* ast = mod_it->second.ast;
    GenericEnv empty_env;

    for (const auto& ed : ast->enums) {
        ScalarKind underlying = ScalarKind::I32;
        if (ed.underlying) {
            const std::string& uname = ed.underlying->name;
            if ((uname == "i32" || uname == "int") && ed.underlying->args.empty()) {
                underlying = ScalarKind::I32;
            } else if ((uname == "u32" || uname == "uint") && ed.underlying->args.empty()) {
                underlying = ScalarKind::U32;
            } else {
                diags.error(ed.underlying->span, "enum '" + ed.name + "' underlying type must be 'i32' or 'u32'");
            }
        }

        std::string mangled = (mod_name == root_name) ? ed.name : (mod_name + "::" + ed.name);
        uint32_t eid = out->types.register_enum(EnumDef{0, mangled, underlying});
        Type enum_type = Type::make_enum(eid, mangled, underlying);

        EnumInfo info;
        info.name = ed.name;
        info.module = mod_name;
        info.underlying = underlying;

        if (ed.variants.empty()) {
            diags.error(ed.span, "enum '" + ed.name + "' must declare at least one variant");
        }

        std::vector<std::unordered_map<std::string, LocalVar>> enum_scopes(1);
        std::set<std::string> seen_variants;
        int64_t next_val = 0;

        for (const auto& v : ed.variants) {
            if (!seen_variants.insert(v.name).second) {
                diags.error(v.span, "duplicate variant '" + v.name + "' in enum '" + ed.name + "'");
                continue;
            }
            int64_t val = next_val;
            if (v.value) {
                auto cv_opt = eval_const_expr(v.value.get(), mod_name, empty_env, enum_scopes);
                if (!cv_opt || !cv_opt->type.is_integer()) {
                    diags.error(v.span, "enum variant '" + v.name + "' value must be a compile-time integer");
                } else {
                    val = cv_opt->int_val;
                }
            }
            if (underlying == ScalarKind::U32) {
                if (val < 0 || val > static_cast<int64_t>(UINT32_MAX)) {
                    diags.error(v.span, "enum variant '" + v.name + "' value " + std::to_string(val) +
                                        " out of range for 'u32'");
                }
            } else {
                if (val < static_cast<int64_t>(INT32_MIN) || val > static_cast<int64_t>(INT32_MAX)) {
                    diags.error(v.span, "enum variant '" + v.name + "' value " + std::to_string(val) +
                                        " out of range for 'i32'");
                }
            }
            next_val = val + 1;

            ConstValue cv;
            cv.type = enum_type;
            cv.int_val = val;
            cv.float_val = static_cast<double>(val);
            cv.bool_val = (val != 0);
            out->constants[mod_name + "::" + ed.name + "::" + v.name] = cv;
            enum_scopes.back()[v.name] = LocalVar{v.name, enum_type, false, true, false, cv, -1};
            info.variants.emplace_back(v.name, val);
        }

        enum_info_by_key[mod_name + "::" + ed.name] = std::move(info);
    }
}

void SemaContext::collect_exported_constants(const std::string& root_name) {
    std::set<std::string> seen_consts;
    std::set<std::string> visited_mods;
    auto add_const_info = [&](const std::string& exported_name, const std::string& mod_name, const ConstDecl& cd) {
        if (!seen_consts.insert(exported_name).second) return;
        auto cit = out->constants.find(mod_name + "::" + cd.name);
        if (cit == out->constants.end()) return;
        out->exported_constants.push_back(ConstInfo{
            exported_name, mod_name, cit->second.type, cit->second, cd.is_knob});
    };
    std::function<void(const std::string&)> collect_mod_exports = [&](const std::string& mname) {
        if (!visited_mods.insert(mname).second) return;
        auto mit = modules.find(mname);
        if (mit == modules.end() || !mit->second.ast) return;
        const auto& mscope = mit->second;
        for (const auto& cd : mscope.ast->consts) {
            if (cd.is_pub || cd.is_knob) {
                add_const_info(cd.name, mname, cd);
            }
        }
        for (const auto& [local_sym, target] : mscope.pub_imported_symbols) {
            auto tmit = modules.find(target.first);
            if (tmit != modules.end()) {
                if (auto cit = tmit->second.const_by_name.find(target.second);
                    cit != tmit->second.const_by_name.end()) {
                    add_const_info(local_sym, target.first, *cit->second);
                }
            }
        }
        for (const auto& wmod : mscope.pub_wildcard_modules) {
            collect_mod_exports(wmod);
        }
    };
    collect_mod_exports(root_name);
    for (const auto& mod_name : module_order) {
        const ModuleAst* ast = modules[mod_name].ast;
        if (!ast) continue;
        for (const auto& cd : ast->consts) {
            if (cd.is_knob) add_const_info(cd.name, mod_name, cd);
        }
    }
}

void SemaContext::collect_exported_enums(const std::string& root_name) {
    std::set<std::string> seen_enums;
    std::set<std::string> visited_mods;
    auto add_enum_info = [&](const std::string& exported_name, const std::string& mod_name, const EnumDecl& ed) {
        if (!seen_enums.insert(exported_name).second) return;
        auto eit = enum_info_by_key.find(mod_name + "::" + ed.name);
        if (eit == enum_info_by_key.end()) return;
        EnumInfo info = eit->second;
        info.name = exported_name;
        out->exported_enums.push_back(std::move(info));
    };
    std::function<void(const std::string&)> collect = [&](const std::string& mname) {
        if (!visited_mods.insert(mname).second) return;
        auto mit = modules.find(mname);
        if (mit == modules.end() || !mit->second.ast) return;
        const auto& mscope = mit->second;
        for (const auto& ed : mscope.ast->enums) {
            if (ed.is_pub) add_enum_info(ed.name, mname, ed);
        }
        for (const auto& [local_sym, target] : mscope.pub_imported_symbols) {
            std::string rmod;
            std::set<std::string> v;
            if (const EnumDecl* ed = find_enum_in_module(target.first, target.second, &rmod, v)) {
                add_enum_info(local_sym, rmod, *ed);
            }
        }
        for (const auto& wmod : mscope.pub_wildcard_modules) {
            collect(wmod);
        }
    };
    collect(root_name);
}

std::optional<ConstValue> SemaContext::eval_sizeof_offsetof_query(
    const Expr* e,
    const std::string& current_mod,
    const GenericEnv& genv,
    DiagnosticBag* report_diags) {
    if (!e || e->generic_args.size() != 1) {
        if (report_diags) {
            report_diags->error(e ? e->span : SourceSpan{}, "'" + (e ? e->name : std::string{}) + "' expects a type argument");
        }
        return std::nullopt;
    }
    const std::string& n = e->name;
    const bool is_sizeof = (n == "sizeof" || n == "std430_sizeof" || n == "std140_sizeof");
    const bool is_offsetof = (n == "offsetof" || n == "std430_offsetof" || n == "std140_offsetof");
    if (!is_sizeof && !is_offsetof) return std::nullopt;
    if (is_sizeof && !e->args.empty()) {
        if (report_diags) report_diags->error(e->span, "'" + n + "' expects 1 type argument");
        return std::nullopt;
    }
    if (is_offsetof && e->args.empty()) {
        if (report_diags) report_diags->error(e->span, "'" + n + "' expects a struct type and field path");
        return std::nullopt;
    }

    Type t = Type::void_type();
    const auto& ga = e->generic_args[0];
    if (ga.type_arg) {
        t = resolve_type(ga.type_arg.get(), current_mod, genv);
    } else if (!ga.ident_or_const_name.empty()) {
        TypeExpr tmp{ga.span, ga.ident_or_const_name, {}};
        t = resolve_type(&tmp, current_mod, genv);
    }
    if (t.is_void()) {
        if (report_diags) report_diags->error(ga.span, "unknown or void type in '" + n + "'");
        return std::nullopt;
    }

    if (is_sizeof) {
        uint32_t sz = 0;
        if (n == "sizeof") {
            sz = out->types.byte_size(t);
        } else {
            const bool std140 = (n == "std140_sizeof");
            sz = out->types.block_layout(t, std140).size;
        }
        return ConstValue::from_i32(static_cast<int32_t>(sz));
    }

    const bool packed = (n == "offsetof");
    const bool std140 = (n == "std140_offsetof");
    uint64_t total_off = 0;
    Type cur_t = t;
    for (const auto& seg_expr : e->args) {
        const std::string& seg = seg_expr->name;
        if (cur_t.is_newtype()) {
            if (seg == "0") {
                cur_t = out->types.unwrap_newtype(cur_t);
                continue;
            }
            if (report_diags) {
                report_diags->error(seg_expr->span, "cannot access field '" + seg + "' directly on newtype '" +
                                                        cur_t.to_string() + "' in '" + n + "' (use '.0' first)");
            }
            return std::nullopt;
        }
        if (!cur_t.is_struct()) {
            if (report_diags) {
                report_diags->error(seg_expr->span, "'" + n + "' expected struct type before field '" + seg +
                                                        "', got '" + cur_t.to_string() + "'");
            }
            return std::nullopt;
        }
        const StructDef* sdef = out->types.find_struct(cur_t.struct_id);
        const StructField* sf = sdef ? sdef->find_field(seg) : nullptr;
        if (!sf) {
            if (report_diags) {
                report_diags->error(seg_expr->span, "struct '" + cur_t.struct_name + "' has no field '" + seg + "'");
            }
            return std::nullopt;
        }
        if (packed) {
            total_off += sf->byte_offset;
        } else {
            uint32_t foff = out->types.block_field_offset(*sdef, seg, std140);
            if (foff == UINT32_MAX) return std::nullopt;
            total_off += foff;
        }
        cur_t = sf->type;
    }
    return ConstValue::from_i32(static_cast<int32_t>(total_off));
}

Type check_newtype_or_enum_cast(FnCheckState& st, Expr* e) {
    Type target = st.ctx.resolve_type(e->target_type.get(), st.fn.module_name, st.genv);
    std::optional<Type> expected_src;
    if (target.is_newtype() && is_unsuffixed_literal(e->lhs.get())) {
        Type u = st.ctx.out->types.unwrap_newtype(target);
        if (u.is_scalar()) expected_src = u;
    }
    Type src = st.check_expr(e->lhs.get(), expected_src);
    if (src.is_newtype() || target.is_newtype()) {
        Type src_u = src.is_newtype() ? st.ctx.out->types.unwrap_newtype(src) : src;
        Type dst_u = target.is_newtype() ? st.ctx.out->types.unwrap_newtype(target) : target;
        bool ok = (src == target);
        if (!ok && src.is_newtype() && !target.is_newtype()) {
            ok = (src_u == target) ||
                 (src_u.is_scalar() && src_u.is_numeric() && target.is_scalar() && (target.is_numeric() || target.is_bool()));
        } else if (!ok && !src.is_newtype() && target.is_newtype()) {
            ok = (src == dst_u) ||
                 (src.is_scalar() && (src.is_numeric() || src.is_bool()) && dst_u.is_scalar() && dst_u.is_numeric());
        }
        if (!ok) {
            st.ctx.diags.error(e->span, "cannot cast '" + src.to_string() + "' to '" + target.to_string() + "'");
        }
    }
    st.mark_type_usage(target);
    e->resolved_type = target;
    return target;
}

Type check_newtype_unary(FnCheckState& st, Expr* e, const Type& sub) {
    if (e->unary_op == UnaryOp::Not) {
        if (!sub.is_bool()) {
            st.ctx.diags.error(e->span, "logical '!' requires a bool operand, got '" + sub.to_string() + "'");
        }
        return e->resolved_type = Type::bool_type();
    }
    if (e->unary_op == UnaryOp::BitNot) {
        if (!sub.is_integer()) {
            st.ctx.diags.error(e->span, "bitwise '~' requires an integer operand, got '" + sub.to_string() + "'");
        }
        return e->resolved_type = sub;
    }
    if (!sub.is_numeric()) {
        st.ctx.diags.error(e->span, "unary operator requires a numeric operand, got '" + sub.to_string() + "'");
    }
    st.mark_type_usage(sub);
    return e->resolved_type = sub;
}

std::optional<Type> check_newtype_or_enum_binary(FnCheckState& st, Expr* e, Type lt, Type rt) {
    if (!lt.is_newtype() && !rt.is_newtype() && !lt.is_enum() && !rt.is_enum()) {
        return std::nullopt;
    }
    if (lt != rt) {
        st.ctx.diags.error(e->span, "binary operator type mismatch: '" + lt.to_string() + "' vs '" + rt.to_string() + "'");
        st.mark_type_usage(lt);
        bool is_cmp = (e->binary_op == BinaryOp::Eq || e->binary_op == BinaryOp::Ne ||
                       e->binary_op == BinaryOp::Lt || e->binary_op == BinaryOp::Le ||
                       e->binary_op == BinaryOp::Gt || e->binary_op == BinaryOp::Ge ||
                       e->binary_op == BinaryOp::LogAnd || e->binary_op == BinaryOp::LogOr);
        return e->resolved_type = is_cmp ? Type::bool_type() : lt;
    }
    const BinaryOp op = e->binary_op;
    const bool is_rel = (op == BinaryOp::Eq || op == BinaryOp::Ne || op == BinaryOp::Lt ||
                         op == BinaryOp::Le || op == BinaryOp::Gt || op == BinaryOp::Ge);
    if (lt.is_enum()) {
        if (!is_rel) {
            st.ctx.diags.error(e->span, "operator not allowed on enum '" + lt.to_string() + "' (cast to underlying integer first)");
        }
        return e->resolved_type = Type::bool_type();
    }
    // Same-newtype binary operator
    if (is_rel) {
        if (!lt.is_scalar() || (!lt.is_numeric() && !lt.is_bool())) {
            st.ctx.diags.error(e->span, "relational operator requires scalar newtype, got '" + lt.to_string() + "'");
        }
        return e->resolved_type = Type::bool_type();
    }
    if (op == BinaryOp::Add || op == BinaryOp::Sub) {
        if (!lt.is_numeric()) {
            st.ctx.diags.error(e->span, "arithmetic operator requires numeric newtype, got '" + lt.to_string() + "'");
        }
        st.mark_type_usage(lt);
        return e->resolved_type = lt;
    }
    st.ctx.diags.error(e->span, "operator not allowed on newtype '" + lt.to_string() + "' (unwrap with '.0' first)");
    st.mark_type_usage(lt);
    return e->resolved_type = lt;
}

std::optional<Type> check_newtype_constructor(FnCheckState& st, Expr* e, const std::string& qname) {
    std::string nt_mod;
    const NewtypeDecl* ndecl = st.ctx.find_newtype_decl(st.fn.module_name, qname, &nt_mod);
    if (!ndecl) return std::nullopt;
    TypeExpr te{e->span, qname, {}};
    Type nt_type = st.ctx.resolve_type(&te, st.fn.module_name, st.genv);
    if (!e->generic_args.empty()) {
        st.ctx.diags.error(e->span, "newtype '" + nt_type.to_string() + "' does not take generic arguments");
    }
    if (e->args.size() != 1) {
        st.ctx.diags.error(e->span, "newtype '" + nt_type.to_string() + "' constructor expects 1 argument");
        for (auto& a : e->args) st.check_expr(a.get());
    } else {
        Type underlying = st.ctx.out->types.unwrap_newtype(nt_type);
        st.check_expr(e->args[0].get(), underlying);
        check_assignable(e->args[0].get(), underlying, st.ctx.diags, e->args[0]->span);
    }
    st.mark_type_usage(nt_type);
    e->resolved_type = nt_type;
    return nt_type;
}

uint32_t resolve_image_format(const CheckedParam& cp, const Attribute* attr, DiagnosticBag& diags) {
    static const struct { const char* name; uint32_t code; bool uint; } kFormats[] = {
        {"rgba32f", 1, false}, {"rgba16f", 2, false},  {"r32f", 3, false},     {"rgba8", 4, false},
        {"rgba8_snorm", 5, false}, {"rg32f", 6, false}, {"rg16f", 7, false},   {"r11g11b10f", 8, false},
        {"r16f", 9, false},    {"rgba16", 10, false},   {"rgb10a2", 11, false}, {"rg16", 12, false},
        {"rg8", 13, false},    {"r16", 14, false},      {"r8", 15, false},     {"rgba32ui", 30, true},
        {"rgba16ui", 31, true}, {"rgba8ui", 32, true},  {"r32ui", 33, true},   {"rg32ui", 35, true},
        {"rg16ui", 36, true},  {"rg8ui", 37, true},     {"r16ui", 38, true},   {"r8ui", 39, true},
    };
    bool uint = cp.type.scalar == ScalarKind::U32;
    if (attr) {
        if (attr->str_args.size() != 1) {
            diags.error(attr->span, "@format takes one format name, e.g. @format(rgba16f)");
            return 0;
        }
        for (const auto& f : kFormats) {
            if (attr->str_args[0] != f.name) continue;
            if (f.uint != uint) {
                diags.error(attr->span, std::string("@format(") + f.name + ") does not hold " +
                                            cp.type.element_type().to_string() + " texels");
                return 0;
            }
            return f.code;
        }
        diags.error(attr->span, "unknown image format '" + attr->str_args[0] + "'");
        return 0;
    }
    switch (cp.type.elem_lanes) {
        case 1: return uint ? 33 : 3;
        case 2: return uint ? 35 : 6;
        case 4: return uint ? 30 : 1;
        default:
            diags.error(cp.span, "no image format holds " + cp.type.element_type().to_string() +
                                     " texels: give one with @format");
            return 0;
    }
}

void SemaContext::resolve_globals() {
    GenericEnv empty_env;
    for (const auto& mod_name : module_order) {
        const ModuleAst* ast = modules[mod_name].ast;
        if (!ast) continue;
        for (const auto& d : ast->bindings) {
            CheckedParam cp;
            cp.span = d.span;
            cp.name = d.name;
            cp.type = resolve_type(d.type.get(), mod_name, empty_env);
            cp.access = cp.type.access;
            if (d.is_shared) {
                cp.role = ParamRole::Shared;
                cp.access = AccessMode::ReadWrite;
                if (!d.attributes.empty()) {
                    diags.error(d.attributes.front().span, "module-level shared variable '" + d.name + "' cannot have attributes");
                }
                validate_shared_type(cp.type, out->types, diags, d.span, d.name);
                out->global_index[mod_name + "::" + d.name] = out->globals.size();
                out->globals.push_back(std::move(cp));
                continue;
            }
            if (cp.type.is_image()) {
                cp.role = ParamRole::Image;
                cp.image_format = resolve_image_format(cp, d.find_attribute("format"), diags);
            } else if (cp.type.is_sampler()) {
                cp.role = ParamRole::Sampler;
            } else if (cp.type.is_uniform()) {
                cp.role = ParamRole::Uniform;
            } else if (cp.type.is_buffer()) {
                cp.role = ParamRole::Buffer;
            } else {
                diags.error(d.span, "module-level '" + d.name + "' is a resource at a descriptor (a texture, "
                                    "sampler, image, uniform or buffer), not '" + cp.type.to_string() + "'");
                continue;
            }
            const Attribute* at = d.find_attribute("binding");
            if (!at || at->int_args.empty() || at->int_args.size() > 2) {
                diags.error(d.span, "module-level '" + d.name + "' needs @binding(set, binding)");
                continue;
            }
            cp.descriptor_set = at->int_args.size() == 2 ? static_cast<uint32_t>(at->int_args[0]) : 0;
            cp.descriptor_binding = static_cast<uint32_t>(at->int_args.back());
            out->global_index[mod_name + "::" + d.name] = out->globals.size();
            out->globals.push_back(std::move(cp));
        }
    }
}

} // namespace brokernel::detail

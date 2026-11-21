#include "sema_internal.h"

namespace brokernel::detail {

Type FnCheckState::check_store_lvalue(Expr* e, bool is_compound) {
    if (!e) return Type::void_type();
    SiteScope at(*this, &e->span);
    if (e->kind == Expr::Kind::Index && e->lhs &&
        (e->lhs->kind == Expr::Kind::Ident || e->lhs->kind == Expr::Kind::Path)) {
        Type recv = check_expr(e->lhs.get());
        if (recv.is_store()) {
            if (!can_Write(recv.access)) {
                ctx.diags.error(e->span, "cannot write to read-only store");
            }
            if (is_compound && !can_Read(recv.access)) {
                ctx.diags.error(e->span, "compound assignment on store requires read-write access");
            }
            if (e->args.size() != recv.dimension) {
                ctx.diags.error(e->span, "store indexing expects " + std::to_string(recv.dimension) + " coordinate(s)");
            }
            for (auto& a : e->args) {
                Type at_t = check_expr(a.get(), Type::i32());
                if (!at_t.is_integer() || !at_t.is_scalar()) {
                    ctx.diags.error(a->span, "store coordinate must be a scalar integer, got '" + at_t.to_string() + "'");
                }
            }
            Type elem = recv.element_type();
            mark_type_usage(elem);
            e->resolved_type = elem;
            return elem;
        }
    }
    if (e->kind == Expr::Kind::Field) {
        Type recv = check_store_lvalue(e->lhs.get(), is_compound);
        const std::string& f = e->name;
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
    if (e->kind == Expr::Kind::Index) {
        Type recv = check_store_lvalue(e->lhs.get(), is_compound);
        if (recv.is_vector() || recv.is_array()) {
            if (e->args.size() != 1) {
                ctx.diags.error(e->span, "indexing expects 1 index expression");
            } else {
                Type at_t = check_expr(e->args[0].get(), Type::i32());
                if (!at_t.is_integer() || !at_t.is_scalar()) {
                    ctx.diags.error(e->args[0]->span, "index must be a scalar integer, got '" + at_t.to_string() + "'");
                }
            }
            Type elem = recv.element_type();
            mark_type_usage(elem);
            e->resolved_type = elem;
            return elem;
        }
        ctx.diags.error(e->span, "type '" + recv.to_string() + "' is not indexable");
        return Type::void_type();
    }
    return check_expr(e);
}

namespace {
std::optional<std::string> find_knob_in_module(
    const std::unordered_map<std::string, ModuleScope>& modules,
    const std::string& mod_name,
    const std::string& sym,
    std::set<std::string>& visited) {
    if (!visited.insert(mod_name).second) return std::nullopt;
    auto mit = modules.find(mod_name);
    if (mit == modules.end() || !mit->second.ast) return std::nullopt;
    const auto& mscope = mit->second;
    for (const auto& cd : mscope.ast->consts) {
        if (cd.name == sym) {
            return cd.is_knob ? std::optional<std::string>(mod_name + "::" + sym) : std::nullopt;
        }
    }
    auto pit = mscope.pub_imported_symbols.find(sym);
    if (pit != mscope.pub_imported_symbols.end()) {
        if (auto k = find_knob_in_module(modules, pit->second.first, pit->second.second, visited)) return k;
    }
    for (const auto& wmod : mscope.pub_wildcard_modules) {
        if (auto k = find_knob_in_module(modules, wmod, sym, visited)) return k;
    }
    return std::nullopt;
}
} // namespace

std::optional<std::string> SemaContext::find_knob_key(
    const std::string& current_mod,
    const std::vector<std::string>& path,
    const std::string& name) const {
    auto mod_it = modules.find(current_mod);
    if (mod_it == modules.end()) return std::nullopt;
    const auto& mscope = mod_it->second;
    if (path.empty()) {
        if (mscope.ast) {
            for (const auto& cd : mscope.ast->consts) {
                if (cd.name == name) {
                    return cd.is_knob ? std::optional<std::string>(current_mod + "::" + name) : std::nullopt;
                }
            }
        }
        auto sym_it = mscope.imported_symbols.find(name);
        if (sym_it != mscope.imported_symbols.end()) {
            std::set<std::string> visited;
            if (auto k = find_knob_in_module(modules, sym_it->second.first, sym_it->second.second, visited)) return k;
        }
        for (const auto& wmod : mscope.wildcard_modules) {
            std::set<std::string> visited;
            if (auto k = find_knob_in_module(modules, wmod, name, visited)) return k;
        }
        return std::nullopt;
    }
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
    std::set<std::string> visited;
    return find_knob_in_module(modules, target_mod, path.back(), visited);
}

void SemaContext::init_knobs_layout() {
    uint32_t offset = 0;
    uint32_t idx = 0;
    for (const auto& mod_name : module_order) {
        auto mit = modules.find(mod_name);
        if (mit == modules.end() || !mit->second.ast) continue;
        for (const auto& cd : mit->second.ast->consts) {
            if (!cd.is_knob) continue;
            std::string key = mod_name + "::" + cd.name;
            auto cit = out->constants.find(key);
            if (cit == out->constants.end()) continue;
            knob_index_by_key[key] = idx++;
            if (runtime_knobs) {
                uint32_t bsz = out->types.byte_size(cit->second.type);
                uint32_t aligned_sz = (bsz + 3u) & ~3u;
                out->runtime_knobs.push_back(RuntimeKnobLayout{
                    cd.name, mod_name, key, cit->second.type, offset, bsz, cit->second});
                offset += aligned_sz;
            }
        }
    }
    out->runtime_knob_bytes = offset;
}

} // namespace brokernel::detail

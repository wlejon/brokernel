#pragma once
// Internal shared structures for brokernel semantic analysis (src/sema.cpp & src/sema_check.cpp).
#include "brokernel/sema.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
namespace brokernel::detail {
struct GenericEnv {
    std::unordered_map<std::string, Type> type_params;
    std::unordered_map<std::string, ConstValue> const_params;
};
struct ModuleScope {
    std::string name;
    const ModuleAst* ast = nullptr;
    std::unordered_map<std::string, std::string> module_aliases; // alias -> canonical module name
    std::unordered_map<std::string, std::pair<std::string, std::string>> imported_symbols; // local -> (mod, sym)
    std::vector<std::string> wildcard_modules;
};
struct LocalVar {
    std::string name;
    Type type;
    bool is_mutable = false;
    bool is_const = false;
    ConstValue const_val{};
    int param_index = -1; // >= 0 if this refers to a CheckedFunction parameter
};
struct SemaContext {
    ModuleRegistry& registry;
    DiagnosticBag& diags;
    std::filesystem::path root_dir;
    std::unique_ptr<CheckedModule> out;
    std::unordered_map<std::string, ModuleScope> modules;
    std::vector<std::string> module_order;
    // Instantiation cache: mangled_name -> index in out->functions
    std::unordered_map<std::string, size_t> fn_instances;
    // Active call stack of mangled function names for recursion detection
    std::vector<std::string> active_fn_stack;
    // Active module stack for circular import detection
    std::vector<std::string> active_import_stack;
    SemaContext(ModuleRegistry& reg, DiagnosticBag& d, std::filesystem::path dir)
        : registry(reg), diags(d), root_dir(std::move(dir)),
          out(std::make_unique<CheckedModule>()) {}
    bool load_and_index_module(const ModuleAst& ast, const std::string& canonical_name,
                               const std::filesystem::path& file_dir);
    const FnDecl* find_fn_decl(const std::string& current_mod,
                               const std::vector<std::string>& path,
                               const std::string& name,
                               std::string* resolved_mod_out) const;
    const StructDecl* find_struct_decl(const std::string& current_mod,
                                       const std::string& name,
                                       std::string* resolved_mod_out) const;
    std::optional<ConstValue> find_const(const std::string& current_mod,
                                         const std::vector<std::string>& path,
                                         const std::string& name,
                                         const GenericEnv& genv,
                                         const std::vector<std::unordered_map<std::string, LocalVar>>& scopes) const;
    Type resolve_type(const TypeExpr* texpr, const std::string& current_mod, const GenericEnv& genv);
    std::optional<ConstValue> eval_const_expr(
        const Expr* e,
        const std::string& current_mod,
        const GenericEnv& genv,
        const std::vector<std::unordered_map<std::string, LocalVar>>& scopes);
    std::optional<int64_t> eval_const_int(
        const Expr* e,
        const std::string& current_mod,
        const GenericEnv& genv,
        const std::vector<std::unordered_map<std::string, LocalVar>>& scopes);
    const CheckedFunction* instantiate_function(
        const FnDecl& decl,
        const std::string& decl_mod,
        const GenericEnv& genv,
        const SourceSpan& call_site,
        bool as_entry_kernel);
};
/// Coerces an expression to `target_type` if permitted (e.g. unsuffixed literals or scalar->vector splat).
bool coerce_expr_to_type(Expr* e, const Type& target_type);
/// Checks if `from` can assign to `to` (updating `expr` if `expr` is an unsuffixed literal).
bool check_assignable(Expr* expr, const Type& to, DiagnosticBag& diags, const SourceSpan& span);
bool is_unsuffixed_literal(const Expr* e);
struct FnCheckState {
    SemaContext& ctx;
    CheckedFunction& fn;
    const GenericEnv& genv;
    std::vector<std::unordered_map<std::string, LocalVar>> scopes;
    int loop_depth = 0;
    void push_scope() { scopes.emplace_back(); }
    void pop_scope() { scopes.pop_back(); }
    LocalVar* lookup_var(const std::string& name) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto vit = it->find(name);
            if (vit != it->end()) return &vit->second;
        }
        return nullptr;
    }
    void mark_type_usage(const Type& t) {
        if (t.scalar == ScalarKind::F64) fn.uses_f64 = true;
        if (t.is_struct()) {
            auto leaves = ctx.out->types.flatten_scalars(t);
            for (auto sk : leaves) {
                if (sk == ScalarKind::F64) fn.uses_f64 = true;
            }
        }
    }
    void ensure_grid_dim(uint32_t min_dim) {
        fn.uses_grid_coords = true;
        if (fn.grid_dim < min_dim) fn.grid_dim = min_dim;
    }
    Type check_expr(Expr* e, std::optional<Type> expected = std::nullopt);
    void check_stmt(Stmt* s);
    void check_block_stmts(std::vector<StmtPtr>& stmts);
    bool validate_stencil_access(CheckedParam& param, const std::vector<ExprPtr>& offset_args, const SourceSpan& span);
    Type check_builtin_call(Expr* e, std::optional<Type> expected);
    Type check_user_call(Expr* e, const FnDecl& callee_decl, const std::string& callee_mod);
    Type check_method_call(Expr* e);
};
} // namespace brokernel::detail

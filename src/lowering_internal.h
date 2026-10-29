#pragma once
// Internal shared structures for brokernel MIR lowering (src/lowering_mir.cpp & src/lowering_expr.cpp).

#include "brokernel/lowering.h"

#include <unordered_map>
#include <vector>

namespace brokernel::detail {

using ValueList = std::vector<brass::Value*>;

brass::Type to_mir_scalar_type(ScalarKind sk);
uint32_t swizzle_index(char c);
bool stmt_has_return(const Stmt* s);

// lowering_math.cpp: the transcendentals as MIR arithmetic, one algorithm for
// the CPU and the GPU. nullptr when `name` is not one of them.
brass::Value* lower_transcendental_1(brass::Builder& b, std::string_view name, brass::Value* x, ScalarKind sk);
brass::Value* lower_transcendental_2(brass::Builder& b, std::string_view name, brass::Value* x, brass::Value* y,
                                     ScalarKind sk);
// isnan / isinf by their bit patterns (i32 0 / 1).
brass::Value* lower_float_class(brass::Builder& b, std::string_view name, brass::Value* x, ScalarKind sk);

struct LoopLoweringCtx {
    brass::BasicBlock* head_bb = nullptr;
    brass::BasicBlock* exit_bb = nullptr;
    int broke_var_id = -1;
    int cont_var_id = -1;
};

struct MirBodyLowerer {
    const CheckedModule& mod;
    const CheckedFunction& root_fn;
    DiagnosticBag& diags;
    brass::codegen::KernelBuilder& kb;
    brass::Builder& b;
    bool is_gpu = true;

    brass::Value* gx = nullptr;
    brass::Value* gy = nullptr;
    brass::Value* gz = nullptr;
    brass::Value* gw = nullptr;
    brass::Value* gh = nullptr;
    brass::Value* gd = nullptr;

    int next_var_id = 1;
    std::unordered_map<int, ValueList> var_values;
    std::vector<std::unordered_map<std::string, int>> scopes;
    std::vector<LoopLoweringCtx> loop_stack;

    int has_returned_var_id = -1;
    int ret_val_var_id = -1;
    Type active_return_type = Type::void_type();

    MirBodyLowerer(const CheckedModule& m, const CheckedFunction& f, DiagnosticBag& d,
                   brass::codegen::KernelBuilder& kbuilder, bool gpu)
        : mod(m), root_fn(f), diags(d), kb(kbuilder), b(kbuilder.builder()), is_gpu(gpu) {}

    void push_scope() { scopes.emplace_back(); }
    void pop_scope() {
        for (const auto& [name, id] : scopes.back()) {
            var_values.erase(id);
        }
        scopes.pop_back();
    }

    int alloc_var(const std::string& name, ValueList initial_leaves);
    int lookup_var_id(const std::string& name) const;
    ValueList make_zero_leaves(const Type& t);
    std::vector<int> sorted_live_var_ids() const;
    std::vector<brass::Value*> pack_var_ids(const std::vector<int>& ids) const;

    brass::Value* clamp_coord(brass::Value* c, brass::Value* extent);
    brass::Value* compute_linear_index(uint32_t dim, brass::Value* x, brass::Value* y, brass::Value* z);
    ValueList load_element_from_buffer(const Type& elem_type, brass::Value* base_ptr, brass::Value* linear_idx);
    void store_element_to_buffer(const Type& elem_type, brass::Value* base_ptr, brass::Value* linear_idx,
                                 const ValueList& vals);

    brass::Value* lower_scalar_cast(brass::Value* v, ScalarKind from_sk, ScalarKind to_sk);
    brass::Value* lower_scalar_binary(BinaryOp op, brass::Value* l, brass::Value* r, ScalarKind sk);
    brass::Value* lower_scalar_math_1(const std::string& name, brass::Value* x, ScalarKind sk);
    brass::Value* lower_scalar_math_2(const std::string& name, brass::Value* a, brass::Value* c, ScalarKind sk);
    brass::Value* lower_scalar_math_3(const std::string& name, brass::Value* x, brass::Value* y, brass::Value* z, ScalarKind sk);

    ValueList lower_expr(const Expr* e);
    void lower_stmt(const Stmt* s);
    void lower_stmt_list(const std::vector<StmtPtr>& stmts);
    ValueList lower_inlined_call(const CheckedFunction& callee, const std::vector<ValueList>& arg_vals);
};

} // namespace brokernel::detail

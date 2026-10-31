#pragma once
// Internal shared structures for brokernel MIR lowering (src/lowering_mir.cpp & src/lowering_expr.cpp).

#include "brokernel/lowering.h"
#include "lowering_ops.h"

#include <functional>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brokernel::detail {

using ValueList = std::vector<brass::Value*>;

brass::Type to_mir_scalar_type(ScalarKind sk);
uint32_t swizzle_index(char c);
bool stmt_has_return(const Stmt* s);

// lowering_math.cpp: the transcendentals as MIR arithmetic, one algorithm for
// the CPU and the GPU. nullptr when `name` is not one of them.
brass::Value* lower_transcendental_1(Ops& b, std::string_view name, brass::Value* x, ScalarKind sk);
brass::Value* lower_transcendental_2(Ops& b, std::string_view name, brass::Value* x, brass::Value* y,
                                     ScalarKind sk);
// isnan / isinf by their bit patterns (a bool).
brass::Value* lower_float_class(Ops& b, std::string_view name, brass::Value* x, ScalarKind sk);

struct LoopLoweringCtx {
    brass::BasicBlock* head_bb = nullptr;
    brass::BasicBlock* exit_bb = nullptr;
    int broke_var_id = -1;
    int cont_var_id = -1;
};

// A vector build's loop-carried values. A loop's head takes its carried
// values' types from the values entering it, before its body is lowered; a
// value the body makes varying, or changes in a loop whose lanes leave at
// different iterations, cannot be carried in a scalar. Such a value is
// recorded here as (loop, variable, leaf) — loops numbered in lowering order,
// which a rebuild repeats — and the build is redone with it widened at that
// loop's entry, until no loop records one.
struct VectorPlan {
    std::set<std::tuple<int, int, size_t>> widen;
    bool redo = false;
};

struct MirBodyLowerer {
    const CheckedModule& mod;
    const CheckedFunction& root_fn;
    DiagnosticBag& diags;
    brass::codegen::KernelBuilder& kb;
    Ops b;
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

    VectorPlan* vplan = nullptr;   // set for a vector build (b.lanes != 0)
    int loop_seq = 0;
    // Every cell this body runs is at least the kernel's stencil halo from
    // the grid's x edges, so a stencil read's x needs no clamp (and in a
    // vector build stays x's stride: one contiguous load).
    bool x_interior = false;

    // Which store accesses the dispatch bounds rather than the access. One
    // at a cell of the grid (a stencil read, read() / write(), or an index
    // of the cell's own coordinates: g.x(), g.y(), g.z() in their places,
    // or g.linear_index() into a 1D store) is in bounds whenever its store
    // covers the grid's first `dim` axes, which the host checks once per
    // dispatch (KernelParamInfo::grid_cover_dim); any other access is
    // bounded (Ops::Bound). Only a compute kernel over a grid, with the
    // GPU's grid guard (no barriers), has its cells' coordinates in the
    // grid: a point kernel's or a barrier kernel's invocations run past it,
    // so their accesses are all bounded but a stencil's clamped reads. Every
    // build of a kernel decides alike, so grid_cover holds for all of them.
    bool cell_coords_in_grid = true;
    std::unordered_map<const brass::Value*, size_t> store_param;   // a store parameter's pointer -> its index
    std::vector<uint32_t> grid_cover;                              // per parameter
    std::unordered_set<const brass::Value*> linear_indices;        // values of g.linear_index()
    // A store parameter's leaves, its pointer recorded.
    void bind_store_param(size_t index, const ValueList& store) { store_param[store[0]] = index; }
    // A texture parameter's placeholder value -> its parameter index.
    std::unordered_map<const brass::Value*, size_t> texture_param;
    // The bound parameter at an index of texture_param / block_param: the
    // root function's parameters, then the module-level resources
    // (CheckedModule::globals) after them.
    const CheckedParam& bound_param(size_t index) const {
        return index < root_fn.params.size() ? root_fn.params[index] : mod.globals[index - root_fn.params.size()];
    }
    // A module-level resource named by `e`: a placeholder recorded as a
    // parameter's is, made at each use (it is a constant).
    ValueList global_value(const Expr* e) {
        const size_t index = root_fn.params.size() + static_cast<size_t>(e->global_index);
        const CheckedParam& p = bound_param(index);
        brass::Value* v = b.build_iconst_i32(static_cast<int32_t>((p.descriptor_set << 16) | p.descriptor_binding));
        (p.type.is_block() ? block_param : texture_param)[v] = index;
        return {v};
    }
    // An image's or a sampler's method (lowering_texture.cpp).
    ValueList lower_texture_call(const Expr* e);

    // Buffers at descriptors (lowering_block.cpp). A block parameter's
    // placeholder value -> its parameter index.
    std::unordered_map<const brass::Value*, size_t> block_param;
    using BlockLayout = TypeRegistry::BlockLayout;
    struct BlockPlace {
        size_t param = 0;
        bool std140 = false;
        bool aligned16 = true;
        bool vec4_base = false;
        brass::Value* offset = nullptr;   // bytes, i32
        Type type;
        std::vector<uint32_t> lanes;      // a swizzle's
    };
    std::unordered_map<int, BlockPlace> block_places;
    BlockLayout block_layout(const Type& t, bool std140) const { return mod.types.block_layout(t, std140); }
    uint32_t block_stride(const Type& elem, bool std140) const { return mod.types.block_stride(elem, std140); }
    uint32_t block_field_offset(const StructDef& sdef, const std::string& field, bool std140) const {
        return mod.types.block_field_offset(sdef, field, std140);
    }
    uint32_t uniform_bytes(const CheckedParam& param) const;
    bool block_place(const Expr* e, BlockPlace& p);
    bool is_block_path(const Expr* e) const;
    void block_words(const Type& t, bool std140, brass::Value* base, std::vector<brass::Value*>& out,
                     std::vector<ScalarKind>& kinds);
    ValueList block_load(const Expr* e, const BlockPlace& p);
    void block_store(const Expr* e, const BlockPlace& p, const ValueList& value);
    std::optional<ValueList> try_block_read(const Expr* e);
    bool try_block_assign(const Stmt* s, ValueList rhs);
    // `store` accessed at a cell, covering `dim` axes: its pointer alone.
    ValueList at_cells(const ValueList& store, uint32_t dim);
    // `store` (of `dim` axes) accessed at (x, y, z): at_cells when those are
    // the cell's coordinates, else the store bounded.
    ValueList store_access(const ValueList& store, uint32_t dim, brass::Value* x, brass::Value* y, brass::Value* z);

    MirBodyLowerer(const CheckedModule& m, const CheckedFunction& f, DiagnosticBag& d,
                   brass::codegen::KernelBuilder& kbuilder, bool gpu)
        : mod(m), root_fn(f), diags(d), kb(kbuilder), b(kbuilder.builder()), is_gpu(gpu),
          cell_coords_in_grid(f.stage == Stage::Compute && f.grid_dim > 0 && !f.uses_barriers_or_shared),
          grid_cover(f.params.size(), 0) {}

    void push_scope() { scopes.emplace_back(); }
    void pop_scope() {
        for (const auto& [name, id] : scopes.back()) {
            var_values.erase(id);
            block_places.erase(id);
        }
        scopes.pop_back();
    }

    int alloc_var(const std::string& name, ValueList initial_leaves);
    int lookup_var_id(const std::string& name) const;
    ValueList make_zero_leaves(const Type& t);
    std::vector<int> sorted_live_var_ids() const;
    std::vector<brass::Value*> pack_var_ids(const std::vector<int>& ids) const;

    brass::Value* clamp_coord(brass::Value* c, brass::Value* extent);
    // s[dx, dy, dz]: the cell that far from this one, clamped into the grid.
    ValueList lower_stencil_read(const Expr* stencil, const Type& st, const std::vector<ExprPtr>& offsets);
    brass::Value* compute_linear_index(uint32_t dim, brass::Value* x, brass::Value* y, brass::Value* z);
    // An element of a store: `store` is its leaves, the pointer and, for a
    // kernel's store parameter, its element count, against which the access
    // is bounded (Ops::Bound). The return store the host sizes is the
    // pointer alone.
    ValueList load_element_from_buffer(const Type& elem_type, const ValueList& store, brass::Value* linear_idx);
    void store_element_to_buffer(const Type& elem_type, const ValueList& store, brass::Value* linear_idx,
                                 const ValueList& vals);

    brass::Value* lower_scalar_cast(brass::Value* v, ScalarKind from_sk, ScalarKind to_sk);
    brass::Value* lower_scalar_binary(BinaryOp op, brass::Value* l, brass::Value* r, ScalarKind sk);
    brass::Value* lower_scalar_math_1(const std::string& name, brass::Value* x, ScalarKind sk);
    // An f32 transcendental as the device's own instruction (render math),
    // null for any other name.
    brass::Value* lower_native_math_1(const std::string& name, brass::Value* x);
    brass::Value* lower_scalar_math_2(const std::string& name, brass::Value* a, brass::Value* c, ScalarKind sk);
    brass::Value* lower_scalar_math_3(const std::string& name, brass::Value* x, brass::Value* y, brass::Value* z, ScalarKind sk);

    ValueList lower_expr(const Expr* e);
    void lower_stmt(const Stmt* s);
    void lower_stmt_list(const std::vector<StmtPtr>& stmts);

    // Two arms of a branch: `then_arm` where cond holds, `else_arm` (may be
    // empty) where it does not; the variables they change merged after.
    // A varying cond runs both arms in line, each under its lanes, and
    // merges by select; a uniform one branches and merges by block params.
    void lower_branch(brass::Value* cond, const std::function<void()>& then_arm,
                      const std::function<void()>& else_arm);
    // A loop: cond_fn at the head (before the break/return flags), body_fn,
    // then step_fn. `counter_var_id` is a for-range's own counter (stepped
    // for every lane, so it stays uniform), else -1. Only the variables
    // `writers` (the body and step) assign are carried round the loop as
    // block params, with the flags; the rest keep their values (and what
    // the lowerer knows of them, like an index's affinity).
    void lower_loop(int broke_var_id, int cont_var_id, int counter_var_id,
                    const std::vector<const Stmt*>& writers,
                    const std::function<brass::Value*()>& cond_fn, const std::function<void()>& body_fn,
                    const std::function<void()>& step_fn);
    ValueList lower_inlined_call(const CheckedFunction& callee, const std::vector<ValueList>& arg_vals,
                                 const std::vector<std::optional<BlockPlace>>& arg_places = {});
};

} // namespace brokernel::detail

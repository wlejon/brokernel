#pragma once
// The lowerer's instruction layer: brass::Builder's methods, lane-aware.
//
// With `lanes` 0 every method is the Builder's own. With `lanes` W (8 or 4)
// the lowerer is building the CPU's vector grid loop, W cells at a time
// (ISPC-style): a value is either uniform (a scalar, the same in every lane)
// or varying (a brass vector of W lanes). An op on uniform operands stays
// scalar; an op with a varying operand broadcasts the uniform ones and emits
// the vector op. A bool is a mask: 0 or all ones (-1) per lane, uniform bools
// too, so a uniform bool broadcasts to a mask. Varying 64-bit values exist
// only at W = 4 (i64x4 / f64x4 beside i32x4 / f32x4); at W = 8 one throws
// NeedNarrowLanes and the grid loop is rebuilt at 4. An op with no vector
// form is computed lane by lane (`per_lane`); one that cannot be (a call, an
// atomic) throws VectorUnsupported and the grid loop stays scalar.
//
// Stores and loads with a varying index go lane by lane; under `mask` an
// inactive lane reads index 0 and writes to a scratch slot, so masked-off
// lanes never fault or write.
//
// An i32 vector known to be base + lane * stride (x + iota, and what adding
// a uniform or multiplying by a constant makes of it) is affine. A 64-bit
// index is not built as a vector: an i32 vector sign-extended is an index
// marker, a uniform i64 base standing for base + stride * off[lane], off the
// lane number for an affine vector (its base and stride folded in) and the
// i32 vector itself otherwise. Add / sub / mul / select keep markers, a load
// or store at one is contiguous (the lane number, stride 1, no mask) or lane
// by lane at scalar addresses, and any other op given one materializes it as
// a vector (which at W = 8 throws NeedNarrowLanes). So a grid's store reads
// and writes at x run at 8 lanes, and so do those at a clamped x.

#include <brass/mir/builder.hpp>

#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brokernel::detail {

struct NeedNarrowLanes : std::runtime_error {
    NeedNarrowLanes() : std::runtime_error("a varying 64-bit value at 8 lanes") {}
};
struct VectorUnsupported : std::runtime_error {
    explicit VectorUnsupported(const std::string& what) : std::runtime_error(what) {}
};

class Ops {
public:
    explicit Ops(brass::Builder& builder) : b_(builder) {}

    brass::Builder& builder() { return b_; }
    operator brass::Builder&() { return b_; }

    // 0 = scalar build; 8 or 4 = the vector grid loop's width.
    uint32_t lanes = 0;
    // The lanes the code being built runs for (a mask), nullptr = all of them.
    brass::Value* mask = nullptr;
    // Where a masked-off lane's store goes: 8 bytes allocated in the entry
    // block by whoever sets `lanes`.
    brass::Value* scratch = nullptr;

    bool vector_mode() const { return lanes != 0; }
    static bool is_varying(const brass::Value* v) { return v && v->type().is_vector(); }
    // The varying type of a scalar type at this width.
    brass::Type vec_type(brass::Type scalar) const;
    // `v` as a varying value (a uniform one broadcast).
    brass::Value* widen(brass::Value* v);
    // `v` widened to `t` when t is a vector and v is not (merges of a uniform
    // and a varying value).
    brass::Value* coerce(brass::Value* v, brass::Type t);
    // A value's merged type: the varying one when either side is.
    static brass::Type merge_type(brass::Value* a, brass::Value* b) {
        return is_varying(a) ? a->type() : b->type();
    }
    // The bool true: 1 in a scalar build, all ones in a vector one.
    brass::Value* true_value() { return b_.build_iconst_i32(vector_mode() ? -1 : 1); }
    // 0, 1, .., W-1 as i32 lanes.
    brass::Value* iota_i32();
    // vany of a mask (an i32, nonzero when any lane is set); a uniform bool as it is.
    brass::Value* any(brass::Value* m);
    // A mask over `t`'s lanes for select: 64-bit lanes take the i32 mask widened.
    brass::Value* mask_for(brass::Value* m, brass::Type t);

    // The index marker for i32 vector `v` sign-extended to i64.
    brass::Value* index_marker(brass::Value* v);
    bool is_index_marker(const brass::Value* v) const { return markers_.count(v) != 0; }

    // Blocks and control flow (always scalar).
    void position_at_end(brass::BasicBlock* bb) { b_.position_at_end(bb); }
    brass::BasicBlock* current_block() const { return b_.current_block(); }
    brass::BasicBlock* append_block(std::string_view name) { return b_.append_block(name); }
    // A block appended to the function, the insertion point left where it was.
    brass::BasicBlock* side_block(std::string_view name);
    brass::Value* add_block_param(brass::BasicBlock* bb, brass::Type t) { return b_.add_block_param(bb, t); }
    brass::Value* add_param(brass::Type t) { return b_.add_param(t); }
    brass::Instruction* build_br(brass::BasicBlock* t) { return b_.build_br(t); }
    brass::Instruction* build_br(brass::BasicBlock* t, const std::vector<brass::Value*>& args) { return b_.build_br(t, args); }
    brass::Instruction* build_br(brass::BasicBlock* t, std::initializer_list<brass::Value*> args) { return b_.build_br(t, args); }
    brass::Instruction* build_br_if(brass::Value* c, brass::BasicBlock* t, brass::BasicBlock* f);
    brass::Instruction* build_ret_void() { return b_.build_ret_void(); }

    // Constants (uniform).
    brass::Value* build_iconst_i32(int32_t v) { return b_.build_iconst_i32(v); }
    brass::Value* build_iconst_i64(int64_t v) { return b_.build_iconst_i64(v); }
    brass::Value* build_fconst_f32(float v) { return b_.build_fconst_f32(v); }
    brass::Value* build_fconst_f64(double v) { return b_.build_fconst_f64(v); }

    // Conversions.
    brass::Value* build_sext_i64(brass::Value* v);
    brass::Value* build_zext_i64(brass::Value* v);
    brass::Value* build_trunc_i32(brass::Value* v);
    brass::Value* build_fptosi_i32(brass::Value* v);
    brass::Value* build_fptosi_i64(brass::Value* v);
    brass::Value* build_fptosi_i32_f32(brass::Value* v);
    brass::Value* build_fptosi_i64_f32(brass::Value* v);
    brass::Value* build_sitofp_f64_i32(brass::Value* v);
    brass::Value* build_sitofp_f64_i64(brass::Value* v);
    brass::Value* build_sitofp_f32_i32(brass::Value* v);
    brass::Value* build_sitofp_f32_i64(brass::Value* v);
    brass::Value* build_fptrunc_f32_f64(brass::Value* v);
    brass::Value* build_fpext_f64_f32(brass::Value* v);
    brass::Value* build_bitcast_i64_f64(brass::Value* v);
    brass::Value* build_bitcast_f64_i64(brass::Value* v);
    brass::Value* build_bitcast_i32_f32(brass::Value* v);
    brass::Value* build_bitcast_f32_i32(brass::Value* v);

    // Arithmetic and logic.
    brass::Value* build_add(brass::Value* a, brass::Value* c);
    brass::Value* build_sub(brass::Value* a, brass::Value* c);
    brass::Value* build_mul(brass::Value* a, brass::Value* c);
    brass::Value* build_fma(brass::Value* a, brass::Value* c, brass::Value* d);
    brass::Value* build_sdiv(brass::Value* a, brass::Value* c);
    brass::Value* build_udiv(brass::Value* a, brass::Value* c);
    brass::Value* build_smod(brass::Value* a, brass::Value* c);
    brass::Value* build_umod(brass::Value* a, brass::Value* c);
    brass::Value* build_neg(brass::Value* v);
    brass::Value* build_not(brass::Value* v);
    brass::Value* build_and(brass::Value* a, brass::Value* c);
    brass::Value* build_or(brass::Value* a, brass::Value* c);
    brass::Value* build_xor(brass::Value* a, brass::Value* c);
    brass::Value* build_shl(brass::Value* a, brass::Value* c);
    brass::Value* build_lshr(brass::Value* a, brass::Value* c);
    brass::Value* build_ashr(brass::Value* a, brass::Value* c);
    brass::Value* build_sqrt(brass::Value* v);
    brass::Value* build_floor(brass::Value* v);
    brass::Value* build_ceil(brass::Value* v);
    brass::Value* build_round(brass::Value* v);
    brass::Value* build_fabs(brass::Value* v);
    brass::Value* build_fmin(brass::Value* a, brass::Value* c);
    brass::Value* build_fmax(brass::Value* a, brass::Value* c);

    // Compares: i32 bools (masks in a vector build).
    brass::Value* build_eq(brass::Value* a, brass::Value* c);
    brass::Value* build_ne(brass::Value* a, brass::Value* c);
    brass::Value* build_slt(brass::Value* a, brass::Value* c);
    brass::Value* build_sle(brass::Value* a, brass::Value* c);
    brass::Value* build_sgt(brass::Value* a, brass::Value* c);
    brass::Value* build_sge(brass::Value* a, brass::Value* c);
    brass::Value* build_ult(brass::Value* a, brass::Value* c);
    brass::Value* build_ule(brass::Value* a, brass::Value* c);
    brass::Value* build_ugt(brass::Value* a, brass::Value* c);
    brass::Value* build_uge(brass::Value* a, brass::Value* c);
    brass::Value* build_select(brass::Value* c, brass::Value* t, brass::Value* f);

    // Memory.
    //
    // A store's bound: the element an access is at (an i64 index, possibly
    // varying or a marker) and the store's element count (a uniform i64).
    // A bounded access outside [0, count) reads zero and writes nothing; an
    // out-of-bounds load reads element 0 (the leaf at `offset` of it) and
    // discards it, so a store the host passes has room for one element even
    // when its count is 0.
    struct Bound {
        brass::Value* index;
        brass::Value* count;
    };
    brass::Value* build_load(brass::Type t, brass::Value* base, int32_t offset);
    brass::Value* build_load_indexed(brass::Type t, brass::Value* base, brass::Value* index, uint8_t scale,
                                     int32_t offset = 0, const Bound* bound = nullptr);
    void build_store_indexed(brass::Type t, brass::Value* base, brass::Value* index, uint8_t scale,
                             int32_t offset, brass::Value* val, const Bound* bound = nullptr);
    void build_store_indexed(brass::Type t, brass::Value* base, brass::Value* index, uint8_t scale,
                             brass::Value* val) {
        build_store_indexed(t, base, index, scale, 0, val);
    }

    // Calls take uniform arguments only.
    brass::Value* build_call(std::string_view callee, brass::Type ret) { return b_.build_call(callee, ret); }
    brass::Value* build_call(std::string_view callee, brass::Type ret, std::initializer_list<brass::Value*> args);
    brass::Value* build_call(std::string_view callee, brass::Type ret, const std::vector<brass::Value*>& args);

private:
    brass::Builder& b_;

    struct Affine {
        brass::Value* base;   // uniform, the vector's element type
        int64_t stride;
    };
    struct Marker {
        int64_t stride;
        brass::Value* off;   // an i32 vector, or nullptr for the lane number
        bool operator==(const Marker& o) const { return stride == o.stride && off == o.off; }
    };
    std::unordered_map<const brass::Value*, Affine> affine_;    // i32 vectors
    std::unordered_map<const brass::Value*, Marker> markers_;   // i64 index markers (each a fresh value)
    // `v` with an index marker materialized as a vector.
    brass::Value* real(brass::Value* v);
    brass::Value* mark(brass::Value* base, Marker m);
    std::optional<Affine> affine_of(brass::Value* v) const;
    // Marker / affine arithmetic, nullptr when the operands are not that.
    brass::Value* affine_bin(brass::Opcode op, brass::Value* a, brass::Value* c);
    // The scalar index of lane i of marker m.
    brass::Value* marker_lane(brass::Value* m, uint32_t i);
    // A load or store at `index` reads or writes the lanes' consecutive elements.
    bool is_contiguous(const brass::Value* index, uint8_t scale, brass::Type t) const;
    // Whether element `index` (a scalar i64) is in [0, count): an i32 bool.
    brass::Value* in_bounds(brass::Value* index, const Bound& bound);
    // For a contiguous access, the uniform test that every lane's element is
    // in bounds; nullptr when the bound's index is not a lane-number marker.
    brass::Value* group_in_bounds(const Bound& bound);
    // The lane-by-lane forms (also the bounded path of a contiguous access
    // whose group runs past its store).
    brass::Value* load_lanes(brass::Type t, brass::Value* base, brass::Value* index, uint8_t scale,
                             int32_t offset, const Bound* bound);
    void store_lanes(brass::Type t, brass::Value* base, brass::Value* index, uint8_t scale, int32_t offset,
                     brass::Value* val, const Bound* bound);

    enum class Cmp { eq, ne, lt, le, gt, ge };
    brass::Value* compare(Cmp c, bool is_unsigned, brass::Value* a, brass::Value* d);
    brass::Value* scalar_compare(Cmp c, bool is_unsigned, brass::Value* a, brass::Value* d);
    // A uniform bool (0 / 1) in the mask encoding (0 / -1).
    brass::Value* bool_to_mask(brass::Value* v) { return b_.build_neg(v); }
    brass::Value* narrow_mask(brass::Value* m);   // 64-bit lane mask -> i32 lanes
    void check_lanes(brass::Type t) const;         // throws NeedNarrowLanes
    // The vector op `op` on both operands widened, or nullptr when neither is varying.
    brass::Value* vec_bin(brass::Opcode op, brass::Value* a, brass::Value* c);
    // `scalar_op` applied lane by lane to operands of which any may be varying.
    template <typename F>
    brass::Value* per_lane(brass::Type res_scalar, std::initializer_list<brass::Value*> operands, F scalar_op);
    brass::Value* lane_of(brass::Value* v, uint32_t i);
    brass::Value* convert(brass::Type to_scalar, brass::Value* v);
    brass::Value* bitcast(brass::Type to_scalar, brass::Value* v);
    brass::Value* safe_divisor(brass::Value* a, brass::Value* d);
};

} // namespace brokernel::detail

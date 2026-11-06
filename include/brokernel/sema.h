#pragma once
// Semantic analysis, type checking, generic monomorphization, and
// grid/stencil/store analysis for brokernel.

#include "brokernel/ast.h"
#include "brokernel/diagnostics.h"
#include "brokernel/module_loader.h"
#include "brokernel/types.h"

#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace brokernel {

using Stage = brass::target::SpirvStage;

/// A 1D, 2D, or 3D integer offset for a stencil access (`s.at(dx, dy, dz)`).
struct StencilOffset {
    int32_t dx = 0;
    int32_t dy = 0;
    int32_t dz = 0;

    bool operator==(const StencilOffset& o) const noexcept {
        return dx == o.dx && dy == o.dy && dz == o.dz;
    }
};

enum class ParamRole : uint8_t {
    Scalar = 0,
    Vector = 1,
    Struct = 2,
    Array = 3,
    Grid = 4,
    Store = 5,
    Stencil = 6,
    Image = 7,
    Sampler = 8,
    GridDim = 9,       // Implicit grid extent (width, height, depth)
    ReturnStore = 10,  // Implicit output store for value-returning GPU dispatches
    StageInput = 11,   // A vertex or fragment stage's input at @location(n)
    Uniform = 12,      // uniform<T> at a descriptor
    Buffer = 13,       // buffer<T> / read_buffer<T> at a descriptor
    Shared = 14,       // shared workgroup variable
};

// A parameter whose value the caller does not pack: the grid (the dispatch's
// domain), a texture or buffer at a descriptor, or a stage input.
constexpr bool is_bound_param(ParamRole r) noexcept {
    return r == ParamRole::Grid || r == ParamRole::Image || r == ParamRole::Sampler || r == ParamRole::StageInput ||
           r == ParamRole::Uniform || r == ParamRole::Buffer || r == ParamRole::Shared;
}

/// Checked parameter metadata for a kernel or helper function.
struct CheckedParam {
    SourceSpan span;
    std::string name;
    Type type;
    ParamRole role = ParamRole::Scalar;
    AccessMode access = AccessMode::Read;
    uint32_t descriptor_set = 0;
    uint32_t descriptor_binding = 0;
    uint32_t image_format = 0;         // An image: its SPIR-V ImageFormat, from @format or its texel
    uint32_t location = 0;             // A stage input's @location
    bool flat = false;                 // A stage input marked @flat (an integer one always is)
    int32_t declared_radius = -1;      // >= 0 when explicitly declared in stencil<T, D, R>
    int32_t inferred_radius = 0;       // Max |offset| seen by the compiler across all axes
    int32_t halo[3] = {0, 0, 0};       // Max |offset| per axis (x, y, z)
    std::vector<StencilOffset> stencil_offsets;

    void record_stencil_offset(int32_t dx, int32_t dy, int32_t dz);
};

/// Compile-time constant value (`const` declaration or `const` generic parameter).
struct ConstValue {
    Type type = Type::i32();
    int64_t int_val = 0;
    double float_val = 0.0;
    bool bool_val = false;
    std::vector<ConstValue> elements = {};

    static ConstValue from_f32(float v) {
        ConstValue c;
        c.type = Type::f32();
        c.float_val = static_cast<double>(v);
        return c;
    }
    static ConstValue from_f64(double v) {
        ConstValue c;
        c.type = Type::f64();
        c.float_val = v;
        return c;
    }
    static ConstValue from_i32(int32_t v) {
        ConstValue c;
        c.type = Type::i32();
        c.int_val = v;
        return c;
    }
    static ConstValue from_u32(uint32_t v) {
        ConstValue c;
        c.type = Type::u32();
        c.int_val = static_cast<int64_t>(v);
        return c;
    }
    static ConstValue from_bool(bool v) {
        ConstValue c;
        c.type = Type::bool_type();
        c.bool_val = v;
        c.int_val = v ? 1 : 0;
        return c;
    }
};

/// Exported constant or `@knob` declaration metadata.
struct ConstInfo {
    std::string name;
    std::string module;
    Type type;
    ConstValue value;
    bool is_knob = false;
};

/// Named probe metadata (`probe("label", expr)`).
struct ProbeInfo {
    std::string name;
    Type type;
    uint32_t index = 0;
};

/// Captured runtime value for a named probe at a single invocation/cell.
struct ProbeSample {
    std::string name;
    Type type;
    float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t lanes = 1;
    bool hit = false;
};

/// A concrete, monomorphized, type-checked function ready for MIR lowering.
struct CheckedFunction {
    SourceSpan span;
    std::string name;                  // Mangled/unique name for generic instantiations
    std::string source_name;           // Original function name in source
    std::string module_name;
    bool is_entry_kernel = false;
    bool is_pub = true;
    Stage stage = Stage::Compute;
    std::optional<Stage> required_stage;
    uint32_t grid_dim = 0;             // 0 = point/raw kernel, 1..3 = 1D/2D/3D grid kernel
    uint32_t local_size[3] = {64, 1, 1};
    bool has_explicit_local_size = false;
    std::vector<CheckedParam> params;
    Type return_type = Type::void_type();
    bool uses_f64 = false;
    // Where the first f64 it reaches is (in its body or a callee's): a GPU
    // kernel that uses f64 is a compile error naming it.
    std::optional<SourceSpan> f64_site;
    bool uses_grid_coords = false;
    bool uses_barriers_or_shared = false;
    bool uses_descriptors = false;     // a texture or buffer parameter at a descriptor
    bool uses_textures = false;        // an image or sampler at a descriptor: GPU only, no CPU build
    // @cpu: a kernel with no GPU build (it may use f64).
    bool cpu_only = false;
    int32_t max_stencil_radius = 0;
    int32_t stencil_halo[3] = {0, 0, 0};
    std::vector<ProbeInfo> probes;
    std::vector<StmtPtr> body;         // Deep-cloned, type-annotated & const-specialized body
};

/// Result of semantic analysis on a root module and its transitive imports.
struct CheckedModule {
    std::string module_name;
    TypeRegistry types;
    std::unordered_map<std::string, ConstValue> constants;
    std::vector<ConstInfo> exported_constants;
    std::vector<ProbeInfo> probes;
    std::vector<std::string> exported_functions;
    bool enable_probes = false;
    bool enable_fragment_probes = false;
    // Module-level resources at descriptors (`@binding(s, b) var name: T;`),
    // each as the bound parameter it stands for, keyed "module::name".
    std::vector<CheckedParam> globals;
    std::unordered_map<std::string, size_t> global_index;
    std::vector<CheckedFunction> functions; // Dependency-ordered (callees before callers, plus kernels)

    const CheckedFunction* find_function(std::string_view name) const noexcept;
};

/// Clones an AST expression or statement (used during generic monomorphization and loop unrolling).
ExprPtr clone_expr(const Expr* e);
StmtPtr clone_stmt(const Stmt* s);
TypeExprPtr clone_type_expr(const TypeExpr* t);

/// Runs semantic analysis, import resolution, type checking, generic monomorphization,
/// and stencil/store validation on `root_ast`.
std::unique_ptr<CheckedModule> check_module(
    const ModuleAst& root_ast,
    ModuleRegistry& registry,
    DiagnosticBag& diags,
    const std::filesystem::path& root_dir = {},
    std::string_view entry_name = {},
    const std::unordered_map<std::string, ConstValue>* knob_overrides = nullptr);

} // namespace brokernel

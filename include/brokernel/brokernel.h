#pragma once
// brokernel: the kernel language for world code, compiled through brass to
// SPIR-V for the GPU and to native code for the CPU. See README.md.

#include "brokernel/ast.h"
#include "brokernel/diagnostics.h"
#include "brokernel/lowering.h"
#include "brokernel/module_loader.h"
#include "brokernel/sema.h"
#include "brokernel/types.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/gpu/vulkan_driver.hpp>
#include <brass/target/spirv_target.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace brokernel {

/// The library's version, "major.minor.patch".
const char* version();

/// 1D, 2D or 3D grid extent for kernel dispatch and point queries.
struct GridShape {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;

    constexpr GridShape() noexcept = default;
    constexpr explicit GridShape(uint32_t w, uint32_t h = 1, uint32_t d = 1) noexcept
        : width(w), height(h), depth(d) {}

    constexpr size_t total_points() const noexcept {
        return static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth);
    }
};

/// Argument passed to a kernel at evaluation or dispatch time (CPU or GPU).
/// Supports scalars, vectors/structs (as flattened scalars or byte spans), and
/// host memory buffers (for `store` and `stencil` parameters).
class KernelArg {
public:
    enum class Kind : uint8_t {
        I32 = 0,
        U32,
        I64,
        U64,
        F32,
        F64,
        Buffer,      // Host pointer + byte size (for store / stencil)
        ScalarsF32,  // Multi-component f32 vector/struct
        ScalarsF64,  // Multi-component f64 vector/struct
        ScalarsI32,  // Multi-component i32 vector/struct
    };

    KernelArg() = default;
    KernelArg(int32_t v) noexcept : kind_(Kind::I32), i64_(v) {}          // NOLINT
    KernelArg(uint32_t v) noexcept : kind_(Kind::U32), u64_(v) {}         // NOLINT
    KernelArg(int64_t v) noexcept : kind_(Kind::I64), i64_(v) {}          // NOLINT
    KernelArg(uint64_t v) noexcept : kind_(Kind::U64), u64_(v) {}         // NOLINT
    KernelArg(float v) noexcept : kind_(Kind::F32), f64_(v) {}            // NOLINT
    KernelArg(double v) noexcept : kind_(Kind::F64), f64_(v) {}           // NOLINT
    KernelArg(bool v) noexcept : kind_(Kind::I32), i64_(v ? 1 : 0) {}     // NOLINT

    static KernelArg buffer(void* data, size_t byte_size) noexcept {
        KernelArg a;
        a.kind_ = Kind::Buffer;
        a.ptr_ = data;
        a.byte_size_ = byte_size;
        return a;
    }

    static KernelArg buffer(const void* data, size_t byte_size) noexcept {
        KernelArg a;
        a.kind_ = Kind::Buffer;
        a.ptr_ = const_cast<void*>(data);
        a.byte_size_ = byte_size;
        a.read_only_ = true;
        return a;
    }

    template <typename T>
    static KernelArg store(std::vector<T>& vec) noexcept {
        return buffer(vec.data(), vec.size() * sizeof(T));
    }

    template <typename T>
    static KernelArg store(const std::vector<T>& vec) noexcept {
        return buffer(vec.data(), vec.size() * sizeof(T));
    }

    template <typename T>
    static KernelArg store(std::span<T> sp) noexcept {
        return buffer(sp.data(), sp.size_bytes());
    }

    template <typename T>
    static KernelArg store(std::span<const T> sp) noexcept {
        return buffer(sp.data(), sp.size_bytes());
    }

    static KernelArg vec2f(float x, float y) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF32;
        a.f32_vals_ = {x, y};
        return a;
    }
    static KernelArg vec3f(float x, float y, float z) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF32;
        a.f32_vals_ = {x, y, z};
        return a;
    }
    static KernelArg vec4f(float x, float y, float z, float w) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF32;
        a.f32_vals_ = {x, y, z, w};
        return a;
    }
    static KernelArg vec2d(double x, double y) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF64;
        a.f64_vals_ = {x, y};
        return a;
    }
    static KernelArg vec3d(double x, double y, double z) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF64;
        a.f64_vals_ = {x, y, z};
        return a;
    }
    static KernelArg vec4d(double x, double y, double z, double w) {
        KernelArg a;
        a.kind_ = Kind::ScalarsF64;
        a.f64_vals_ = {x, y, z, w};
        return a;
    }
    static KernelArg vec2i(int32_t x, int32_t y) {
        KernelArg a;
        a.kind_ = Kind::ScalarsI32;
        a.i32_vals_ = {x, y};
        return a;
    }
    static KernelArg vec3i(int32_t x, int32_t y, int32_t z) {
        KernelArg a;
        a.kind_ = Kind::ScalarsI32;
        a.i32_vals_ = {x, y, z};
        return a;
    }
    static KernelArg vec4i(int32_t x, int32_t y, int32_t z, int32_t w) {
        KernelArg a;
        a.kind_ = Kind::ScalarsI32;
        a.i32_vals_ = {x, y, z, w};
        return a;
    }

    Kind kind() const noexcept { return kind_; }
    int32_t as_i32() const noexcept;
    uint32_t as_u32() const noexcept;
    int64_t as_i64() const noexcept;
    uint64_t as_u64() const noexcept;
    float as_f32() const noexcept;
    double as_f64() const noexcept;

    void* buffer_ptr() const noexcept { return ptr_; }
    size_t buffer_bytes() const noexcept { return byte_size_; }
    bool is_read_only() const noexcept { return read_only_; }

    const std::vector<float>& f32_values() const noexcept { return f32_vals_; }
    const std::vector<double>& f64_values() const noexcept { return f64_vals_; }
    const std::vector<int32_t>& i32_values() const noexcept { return i32_vals_; }

private:
    Kind kind_ = Kind::I32;
    int64_t i64_ = 0;
    uint64_t u64_ = 0;
    double f64_ = 0.0;
    void* ptr_ = nullptr;
    size_t byte_size_ = 0;
    bool read_only_ = false;
    std::vector<float> f32_vals_;
    std::vector<double> f64_vals_;
    std::vector<int32_t> i32_vals_;
};

/// Result of comparing a kernel's CPU build against its GPU (SPIR-V) build.
struct ParityReport {
    bool ran_gpu = false;
    bool ok = false;
    size_t elements_compared = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    uint64_t max_ulp_diff = 0;
    std::string message;
};

/// Options for compiling a .bk source or file.
struct CompileOptions {
    std::string entry_name;              // Empty = compile all kernels (or all functions if compile_all_functions)
    bool compile_all_functions = false;  // When true, also compiles non-@kernel functions as callable entry points
    Stage default_stage = Stage::Compute;
    uint32_t local_size[3] = {64, 1, 1};
    bool local_size_spec_constants = true;
    uint32_t spirv_version = 0x00010500u; // 0x00010500 (SPIR-V 1.5) or 0x00010600 (SPIR-V 1.6)
    bool emit_spirv = true;
    bool emit_spirv_disassembly = true;
    bool emit_ptx = false;
    bool compile_cpu = true;
    bool verify_spirv_equivalence = false;
    int opt_level = 2;
};

/// One compiled kernel with its SPIR-V binary, CPU native builds, and reflection metadata.
class CompiledKernel {
public:
    std::string name;
    Stage stage = Stage::Compute;
    uint32_t grid_dim = 0;               // 0 = point kernel, 1..3 = 1D/2D/3D grid kernel
    uint32_t local_size[3] = {64, 1, 1};
    std::vector<KernelParamInfo> params; // Source parameters
    std::vector<KernelParamInfo> all_mir_params; // Includes implicit GridDim and ReturnStore
    Type return_type = Type::void_type();
    uint32_t return_flat_leaves = 0;
    uint32_t return_byte_size = 0;
    std::vector<ScalarKind> return_leaf_kinds;
    bool uses_f64 = false;
    bool uses_grid_coords = false;
    int32_t max_stencil_radius = 0;
    int32_t stencil_halo[3] = {0, 0, 0};

    brass::target::SpirvKernel spirv;
    std::string spirv_disassembly;
    std::string ptx;
    bool spirv_equivalent = false;

    brass::codegen::KernelFunction cpu_point_fn;
    brass::codegen::KernelFunction cpu_grid_fn;

    bool valid() const noexcept {
        return !spirv.words.empty() || cpu_point_fn.is_valid() || cpu_grid_fn.is_valid();
    }
    bool has_return_value() const noexcept { return !return_type.is_void(); }

    const KernelParamInfo* find_param(std::string_view pname) const noexcept;

    // -----------------------------------------------------------------------
    // CPU Point Evaluation (single continuous point or single grid cell)
    // -----------------------------------------------------------------------

    /// Evaluates a point kernel at `args` and writes the return value into `out_return`.
    /// For a grid kernel (`grid_dim > 0`), evaluates at grid coordinate `(0, 0, 0)` of shape `(1, 1, 1)`.
    bool eval_cpu(std::span<const KernelArg> args, void* out_return = nullptr, std::string* err = nullptr) const;
    bool eval_cpu(std::initializer_list<KernelArg> args, void* out_return = nullptr, std::string* err = nullptr) const {
        return eval_cpu(std::span<const KernelArg>(args.begin(), args.size()), out_return, err);
    }

    /// Evaluates a grid/stencil/store kernel at a single grid cell `(gx, gy, gz)` within `shape` on the CPU.
    bool eval_grid_point_cpu(
        int32_t gx, int32_t gy, int32_t gz,
        GridShape shape,
        std::span<const KernelArg> args,
        void* out_return = nullptr,
        std::string* err = nullptr) const;
    bool eval_grid_point_cpu(
        int32_t gx, int32_t gy, int32_t gz,
        GridShape shape,
        std::initializer_list<KernelArg> args,
        void* out_return = nullptr,
        std::string* err = nullptr) const {
        return eval_grid_point_cpu(gx, gy, gz, shape, std::span<const KernelArg>(args.begin(), args.size()), out_return, err);
    }

    float eval_f32(std::span<const KernelArg> args) const;
    float eval_f32(std::initializer_list<KernelArg> args) const {
        return eval_f32(std::span<const KernelArg>(args.begin(), args.size()));
    }
    double eval_f64(std::span<const KernelArg> args) const;
    double eval_f64(std::initializer_list<KernelArg> args) const {
        return eval_f64(std::span<const KernelArg>(args.begin(), args.size()));
    }
    int32_t eval_i32(std::span<const KernelArg> args) const;
    int32_t eval_i32(std::initializer_list<KernelArg> args) const {
        return eval_i32(std::span<const KernelArg>(args.begin(), args.size()));
    }
    int64_t eval_i64(std::span<const KernelArg> args) const;
    int64_t eval_i64(std::initializer_list<KernelArg> args) const {
        return eval_i64(std::span<const KernelArg>(args.begin(), args.size()));
    }

    float eval_point_2d_f32(int32_t gx, int32_t gy, uint32_t width, uint32_t height,
                            std::span<const KernelArg> args) const;
    float eval_point_2d_f32(int32_t gx, int32_t gy, uint32_t width, uint32_t height,
                            std::initializer_list<KernelArg> args) const {
        return eval_point_2d_f32(gx, gy, width, height, std::span<const KernelArg>(args.begin(), args.size()));
    }
    double eval_point_2d_f64(int32_t gx, int32_t gy, uint32_t width, uint32_t height,
                             std::span<const KernelArg> args) const;
    double eval_point_2d_f64(int32_t gx, int32_t gy, uint32_t width, uint32_t height,
                             std::initializer_list<KernelArg> args) const {
        return eval_point_2d_f64(gx, gy, width, height, std::span<const KernelArg>(args.begin(), args.size()));
    }

    // -----------------------------------------------------------------------
    // CPU Grid Dispatch (runs the kernel over the full 1D/2D/3D grid on CPU)
    // -----------------------------------------------------------------------

    bool dispatch_cpu(
        GridShape shape,
        std::span<const KernelArg> args,
        void* return_store_out = nullptr,
        std::string* err = nullptr) const;
    bool dispatch_cpu(
        GridShape shape,
        std::initializer_list<KernelArg> args,
        void* return_store_out = nullptr,
        std::string* err = nullptr) const {
        return dispatch_cpu(shape, std::span<const KernelArg>(args.begin(), args.size()), return_store_out, err);
    }

    // -----------------------------------------------------------------------
    // GPU (Vulkan) Dispatch & Point Evaluation
    // -----------------------------------------------------------------------

    bool dispatch_gpu(
        GridShape shape,
        std::span<const KernelArg> args,
        void* return_store_out = nullptr,
        std::string* err = nullptr) const;
    bool dispatch_gpu(
        GridShape shape,
        std::initializer_list<KernelArg> args,
        void* return_store_out = nullptr,
        std::string* err = nullptr) const {
        return dispatch_gpu(shape, std::span<const KernelArg>(args.begin(), args.size()), return_store_out, err);
    }

    bool eval_gpu(std::span<const KernelArg> args, void* out_return = nullptr, std::string* err = nullptr) const;
    bool eval_gpu(std::initializer_list<KernelArg> args, void* out_return = nullptr, std::string* err = nullptr) const {
        return eval_gpu(std::span<const KernelArg>(args.begin(), args.size()), out_return, err);
    }
    float eval_gpu_f32(std::span<const KernelArg> args, std::string* err = nullptr) const;
    float eval_gpu_f32(std::initializer_list<KernelArg> args, std::string* err = nullptr) const {
        return eval_gpu_f32(std::span<const KernelArg>(args.begin(), args.size()), err);
    }
    double eval_gpu_f64(std::span<const KernelArg> args, std::string* err = nullptr) const;
    double eval_gpu_f64(std::initializer_list<KernelArg> args, std::string* err = nullptr) const {
        return eval_gpu_f64(std::span<const KernelArg>(args.begin(), args.size()), err);
    }

    // -----------------------------------------------------------------------
    // Target Agreement (CPU vs GPU parity check)
    // -----------------------------------------------------------------------

    ParityReport compare_cpu_gpu(
        GridShape shape,
        std::span<const KernelArg> args,
        double rel_tol = 1e-5,
        double abs_tol = 1e-6) const;
    ParityReport compare_cpu_gpu(
        GridShape shape,
        std::initializer_list<KernelArg> args,
        double rel_tol = 1e-5,
        double abs_tol = 1e-6) const {
        return compare_cpu_gpu(shape, std::span<const KernelArg>(args.begin(), args.size()), rel_tol, abs_tol);
    }

private:
    // The Vulkan pipeline, built on the first GPU dispatch and shared by copies.
    struct GpuCache {
        std::mutex mu;
        std::unique_ptr<brass::gpu::VulkanModule> module;
    };
    std::shared_ptr<GpuCache> gpu_cache_ = std::make_shared<GpuCache>();
};

/// Result of compiling a .bk source unit.
struct CompileResult {
    bool ok = false;
    std::string module_name;
    DiagnosticBag diagnostics;
    TypeRegistry types;
    std::vector<CompiledKernel> kernels;

    const CompiledKernel* find_kernel(std::string_view name) const noexcept;
    const CompiledKernel* default_kernel() const noexcept {
        return kernels.empty() ? nullptr : &kernels.front();
    }
    std::string error_message() const { return diagnostics.format_all(); }
};

/// Stateful compiler holding a ModuleRegistry (standard library + custom modules + search paths).
class Compiler {
public:
    Compiler();

    ModuleRegistry& modules() noexcept { return modules_; }
    const ModuleRegistry& modules() const noexcept { return modules_; }

    void register_module(std::string name, std::string source);
    void add_search_path(std::filesystem::path dir);

    CompileResult compile_source(
        std::string_view source,
        std::string file_name = "kernel.bk",
        const CompileOptions& opts = {});

    CompileResult compile_file(
        const std::filesystem::path& path,
        const CompileOptions& opts = {});

private:
    ModuleRegistry modules_;
};

/// Convenience one-shot compilation functions.
CompileResult compile_source(
    std::string_view source,
    std::string file_name = "kernel.bk",
    const CompileOptions& opts = {});

CompileResult compile_file(
    const std::filesystem::path& path,
    const CompileOptions& opts = {});

} // namespace brokernel

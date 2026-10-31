#pragma once
// Type system for brokernel: scalars (including first-class f64), vectors,
// fixed arrays, structs, grids, stencils, stores, images, and samplers.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brokernel {

enum class ScalarKind : uint8_t {
    Void = 0,
    Bool = 1,
    I32 = 2,
    U32 = 3,
    I64 = 4,
    U64 = 5,
    F32 = 6,
    F64 = 7,
};

enum class AccessMode : uint8_t {
    Read = 0,
    Write = 1,
    ReadWrite = 2,
};

enum class TypeKind : uint8_t {
    Void = 0,
    Scalar = 1,
    Vector = 2,
    Array = 3,
    Struct = 4,
    Grid = 5,
    Stencil = 6,
    Store = 7,
    Image = 8,
    Sampler = 9,
    Uniform = 10,   // a uniform buffer at a descriptor holding one T (std140)
    Buffer = 11,    // a storage buffer at a descriptor, an array of T (std430)
};

/// Value type descriptor.
struct Type {
    TypeKind kind = TypeKind::Void;
    ScalarKind scalar = ScalarKind::Void;
    uint32_t dimension = 0;       // Vector lanes (2..4), Grid/Store/Stencil/Image dim (1..3), or Array length
    int32_t stencil_radius = 0;   // Stencil radius (>= 0, or -1 when inferred from accesses)
    AccessMode access = AccessMode::ReadWrite;
    uint32_t struct_id = UINT32_MAX;
    std::string struct_name;

    // For composite stores/stencils/arrays whose element is a vector or struct:
    TypeKind elem_kind = TypeKind::Scalar;
    uint32_t elem_lanes = 1;
    // Images and samplers: an array of layers, a depth texture read by
    // comparison, a cube.
    static constexpr uint8_t kImageArrayed = 1, kImageShadow = 2, kImageCube = 4;
    uint8_t image_flags = 0;

    static Type void_type() noexcept { return Type{}; }
    static Type scalar_type(ScalarKind s) noexcept {
        if (s == ScalarKind::Void) return void_type();
        Type t;
        t.kind = TypeKind::Scalar;
        t.scalar = s;
        t.dimension = 1;
        return t;
    }
    static Type bool_type() noexcept { return scalar_type(ScalarKind::Bool); }
    static Type i32() noexcept { return scalar_type(ScalarKind::I32); }
    static Type u32() noexcept { return scalar_type(ScalarKind::U32); }
    static Type i64() noexcept { return scalar_type(ScalarKind::I64); }
    static Type u64() noexcept { return scalar_type(ScalarKind::U64); }
    static Type f32() noexcept { return scalar_type(ScalarKind::F32); }
    static Type f64() noexcept { return scalar_type(ScalarKind::F64); }

    static Type vec(ScalarKind s, uint32_t lanes) noexcept {
        if (lanes <= 1) return scalar_type(s);
        Type t;
        t.kind = TypeKind::Vector;
        t.scalar = s;
        t.dimension = lanes;
        return t;
    }

    static Type array_of(const Type& elem, uint32_t length) {
        Type t;
        t.kind = TypeKind::Array;
        t.scalar = elem.scalar;
        t.dimension = length;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.struct_id = elem.struct_id;
        t.struct_name = elem.struct_name;
        return t;
    }

    static Type make_struct(uint32_t id, std::string name) {
        Type t;
        t.kind = TypeKind::Struct;
        t.struct_id = id;
        t.struct_name = std::move(name);
        return t;
    }

    static Type grid(uint32_t dim) noexcept {
        Type t;
        t.kind = TypeKind::Grid;
        t.dimension = dim;
        return t;
    }

    static Type stencil(const Type& elem, uint32_t dim, int32_t radius = -1) {
        Type t;
        t.kind = TypeKind::Stencil;
        t.scalar = elem.scalar;
        t.dimension = dim;
        t.stencil_radius = radius;
        t.access = AccessMode::Read;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.struct_id = elem.struct_id;
        t.struct_name = elem.struct_name;
        return t;
    }

    static Type store(const Type& elem, uint32_t dim, AccessMode mode = AccessMode::ReadWrite) {
        Type t;
        t.kind = TypeKind::Store;
        t.scalar = elem.scalar;
        t.dimension = dim;
        t.access = mode;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.struct_id = elem.struct_id;
        t.struct_name = elem.struct_name;
        return t;
    }

    static Type image(const Type& elem, uint32_t dim, AccessMode mode = AccessMode::ReadWrite, uint8_t flags = 0) {
        Type t;
        t.kind = TypeKind::Image;
        t.scalar = elem.scalar;
        t.dimension = dim;
        t.access = mode;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.image_flags = flags;
        return t;
    }

    // A sampled texture of `dim` dimensions (a cube's are its 3 direction
    // components), `flags` kImage*: its texel `elem`, or f32 for a shadow.
    static Type sampler(const Type& elem, uint32_t dim, uint8_t flags = 0) {
        Type t;
        t.kind = TypeKind::Sampler;
        t.scalar = elem.scalar;
        t.dimension = dim;
        t.access = AccessMode::Read;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.image_flags = flags;
        return t;
    }

    // uniform<T> (kind Uniform) or buffer<T> / read_buffer<T> (kind Buffer).
    static Type block(TypeKind k, const Type& elem, AccessMode mode = AccessMode::Read) {
        Type t;
        t.kind = k;
        t.scalar = elem.scalar;
        t.dimension = elem.is_array() ? elem.dimension : 1;
        t.access = mode;
        t.elem_kind = elem.kind;
        t.elem_lanes = elem.is_vector() ? elem.dimension : 1;
        t.struct_id = elem.struct_id;
        t.struct_name = elem.struct_name;
        return t;
    }

    bool image_arrayed() const noexcept { return (image_flags & kImageArrayed) != 0; }
    bool image_shadow() const noexcept { return (image_flags & kImageShadow) != 0; }
    bool image_cube() const noexcept { return (image_flags & kImageCube) != 0; }
    // An image's or a sampler's coordinate components: its dimensions and,
    // for an array, the layer.
    uint32_t image_coords() const noexcept { return dimension + (image_arrayed() ? 1 : 0); }

    bool is_void() const noexcept { return kind == TypeKind::Void; }
    bool is_scalar() const noexcept { return kind == TypeKind::Scalar; }
    bool is_vector() const noexcept { return kind == TypeKind::Vector; }
    bool is_array() const noexcept { return kind == TypeKind::Array; }
    bool is_struct() const noexcept { return kind == TypeKind::Struct; }
    bool is_grid() const noexcept { return kind == TypeKind::Grid; }
    bool is_stencil() const noexcept { return kind == TypeKind::Stencil; }
    bool is_store() const noexcept { return kind == TypeKind::Store; }
    bool is_image() const noexcept { return kind == TypeKind::Image; }
    bool is_sampler() const noexcept { return kind == TypeKind::Sampler; }
    bool is_uniform() const noexcept { return kind == TypeKind::Uniform; }
    bool is_buffer() const noexcept { return kind == TypeKind::Buffer; }
    bool is_block() const noexcept { return is_uniform() || is_buffer(); }
    bool is_resource() const noexcept {
        return is_grid() || is_stencil() || is_store() || is_image() || is_sampler() || is_block();
    }

    bool is_bool() const noexcept { return is_scalar() && scalar == ScalarKind::Bool; }
    bool is_i32() const noexcept { return is_scalar() && scalar == ScalarKind::I32; }
    bool is_u32() const noexcept { return is_scalar() && scalar == ScalarKind::U32; }
    bool is_i64() const noexcept { return is_scalar() && scalar == ScalarKind::I64; }
    bool is_u64() const noexcept { return is_scalar() && scalar == ScalarKind::U64; }
    bool is_f32() const noexcept { return is_scalar() && scalar == ScalarKind::F32; }
    bool is_f64() const noexcept { return is_scalar() && scalar == ScalarKind::F64; }

    bool is_float_scalar() const noexcept {
        return is_scalar() && (scalar == ScalarKind::F32 || scalar == ScalarKind::F64);
    }
    bool is_float() const noexcept {
        return (is_scalar() || is_vector()) && (scalar == ScalarKind::F32 || scalar == ScalarKind::F64);
    }
    bool is_signed_int() const noexcept {
        return (is_scalar() || is_vector()) && (scalar == ScalarKind::I32 || scalar == ScalarKind::I64);
    }
    bool is_unsigned_int() const noexcept {
        return (is_scalar() || is_vector()) && (scalar == ScalarKind::U32 || scalar == ScalarKind::U64);
    }
    bool is_integer() const noexcept { return is_signed_int() || is_unsigned_int(); }
    bool is_numeric() const noexcept { return is_float() || is_integer(); }
    bool is_64bit() const noexcept {
        return scalar == ScalarKind::F64 || scalar == ScalarKind::I64 || scalar == ScalarKind::U64;
    }

    Type element_type() const {
        if (is_vector()) return scalar_type(scalar);
        if (is_array() || is_store() || is_stencil() || is_image() || is_sampler() || is_block()) {
            if (elem_kind == TypeKind::Struct) return make_struct(struct_id, struct_name);
            if (elem_kind == TypeKind::Vector) return vec(scalar, elem_lanes);
            return scalar_type(scalar);
        }
        return *this;
    }

    uint32_t scalar_size_bytes() const noexcept;
    uint32_t size_in_bytes() const noexcept;
    std::string to_string() const;

    bool operator==(const Type& o) const noexcept;
    bool operator!=(const Type& o) const noexcept { return !(*this == o); }
};

struct StructField {
    std::string name;
    Type type;
    uint32_t byte_offset = 0;
    uint32_t flat_leaf_offset = 0;
    uint32_t flat_leaf_count = 1;
};

struct StructDef {
    uint32_t id = 0;
    std::string name;
    std::vector<StructField> fields;
    uint32_t byte_size = 0;
    uint32_t flat_leaf_count = 0;

    const StructField* find_field(std::string_view fname) const noexcept;
};

class TypeRegistry {
public:
    uint32_t register_struct(StructDef def);
    const StructDef* find_struct(uint32_t id) const noexcept;
    const StructDef* find_struct(std::string_view name) const noexcept;
    const std::vector<StructDef>& structs() const noexcept { return structs_; }

    /// Flattens any scalar, vector, array, or struct type into its primitive ScalarKind leaves.
    std::vector<ScalarKind> flatten_scalars(const Type& t) const;
    uint32_t flat_leaf_count(const Type& t) const;
    uint32_t byte_size(const Type& t) const;

private:
    std::vector<StructDef> structs_;
};

const char* scalar_name(ScalarKind s) noexcept;
bool can_Read(AccessMode m) noexcept;
bool can_Write(AccessMode m) noexcept;

} // namespace brokernel

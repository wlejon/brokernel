#include "brokernel/types.h"

#include <sstream>

namespace brokernel {

const char* scalar_name(ScalarKind s) noexcept {
    switch (s) {
        case ScalarKind::Void: return "void";
        case ScalarKind::Bool: return "bool";
        case ScalarKind::I32:  return "i32";
        case ScalarKind::U32:  return "u32";
        case ScalarKind::I64:  return "i64";
        case ScalarKind::U64:  return "u64";
        case ScalarKind::F32:  return "f32";
        case ScalarKind::F64:  return "f64";
    }
    return "unknown";
}

bool can_Read(AccessMode m) noexcept {
    return m == AccessMode::Read || m == AccessMode::ReadWrite;
}

bool can_Write(AccessMode m) noexcept {
    return m == AccessMode::Write || m == AccessMode::ReadWrite;
}

uint32_t Type::scalar_size_bytes() const noexcept {
    switch (scalar) {
        case ScalarKind::Void: return 0;
        case ScalarKind::Bool:
        case ScalarKind::I32:
        case ScalarKind::U32:
        case ScalarKind::F32:  return 4;
        case ScalarKind::I64:
        case ScalarKind::U64:
        case ScalarKind::F64:  return 8;
    }
    return 0;
}

uint32_t Type::size_in_bytes() const noexcept {
    switch (kind) {
        case TypeKind::Void:   return 0;
        case TypeKind::Scalar: return scalar_size_bytes();
        case TypeKind::Vector: return scalar_size_bytes() * dimension;
        case TypeKind::Array:  return element_type().size_in_bytes() * dimension;
        case TypeKind::Struct: return 0; // Use TypeRegistry::byte_size for structs
        case TypeKind::Grid:   return 0;
        case TypeKind::Stencil:
        case TypeKind::Store:  return 8; // 64-bit buffer device address / pointer
        case TypeKind::Image:
        case TypeKind::Sampler: return 0;
    }
    return 0;
}

std::string Type::to_string() const {
    std::ostringstream oss;
    switch (kind) {
        case TypeKind::Void:
            return "void";
        case TypeKind::Scalar:
            return scalar_name(scalar);
        case TypeKind::Vector:
            oss << "vec" << dimension << "<" << scalar_name(scalar) << ">";
            return oss.str();
        case TypeKind::Array:
            oss << "array<" << element_type().to_string() << ", " << dimension << ">";
            return oss.str();
        case TypeKind::Struct:
            return struct_name.empty() ? "struct" : struct_name;
        case TypeKind::Grid:
            oss << "grid<" << dimension << ">";
            return oss.str();
        case TypeKind::Stencil:
            oss << "stencil<" << element_type().to_string() << ", " << dimension;
            if (stencil_radius >= 0) oss << ", " << stencil_radius;
            oss << ">";
            return oss.str();
        case TypeKind::Store: {
            const char* prefix = "rw_store";
            if (access == AccessMode::Read) prefix = "read_store";
            else if (access == AccessMode::Write) prefix = "write_store";
            oss << prefix << "<" << element_type().to_string() << ", " << dimension << ">";
            return oss.str();
        }
        case TypeKind::Image:
        case TypeKind::Sampler:
            oss << (kind == TypeKind::Image ? "image" : "sampler");
            if (image_cube()) oss << "_cube";
            else oss << dimension << "d";
            if (image_arrayed()) oss << "_array";
            if (image_shadow()) return oss.str() + "_shadow";
            oss << "<" << element_type().to_string() << ">";
            return oss.str();
    }
    return "unknown";
}

bool Type::operator==(const Type& o) const noexcept {
    if (kind != o.kind) return false;
    switch (kind) {
        case TypeKind::Void:
            return true;
        case TypeKind::Scalar:
            return scalar == o.scalar;
        case TypeKind::Vector:
            return scalar == o.scalar && dimension == o.dimension;
        case TypeKind::Array:
            return scalar == o.scalar && dimension == o.dimension &&
                   elem_kind == o.elem_kind && elem_lanes == o.elem_lanes &&
                   struct_id == o.struct_id;
        case TypeKind::Struct:
            return struct_id == o.struct_id && struct_name == o.struct_name;
        case TypeKind::Grid:
            return dimension == o.dimension;
        case TypeKind::Stencil:
            return scalar == o.scalar && dimension == o.dimension &&
                   elem_kind == o.elem_kind && elem_lanes == o.elem_lanes &&
                   struct_id == o.struct_id &&
                   (stencil_radius < 0 || o.stencil_radius < 0 || stencil_radius == o.stencil_radius);
        case TypeKind::Store:
            return scalar == o.scalar && dimension == o.dimension &&
                   access == o.access && elem_kind == o.elem_kind &&
                   elem_lanes == o.elem_lanes && struct_id == o.struct_id;
        case TypeKind::Image:
        case TypeKind::Sampler:
            return scalar == o.scalar && dimension == o.dimension &&
                   elem_kind == o.elem_kind && elem_lanes == o.elem_lanes && image_flags == o.image_flags;
    }
    return false;
}

const StructField* StructDef::find_field(std::string_view fname) const noexcept {
    for (const auto& f : fields) {
        if (f.name == fname) return &f;
    }
    return nullptr;
}

uint32_t TypeRegistry::register_struct(StructDef def) {
    for (const auto& existing : structs_) {
        if (existing.name == def.name) {
            return existing.id;
        }
    }
    uint32_t id = static_cast<uint32_t>(structs_.size());
    def.id = id;
    uint32_t byte_off = 0;
    uint32_t leaf_off = 0;
    for (auto& f : def.fields) {
        uint32_t f_bytes = byte_size(f.type);
        uint32_t f_leaves = flat_leaf_count(f.type);
        f.byte_offset = byte_off;
        f.flat_leaf_offset = leaf_off;
        f.flat_leaf_count = f_leaves;
        byte_off += f_bytes;
        leaf_off += f_leaves;
    }
    def.byte_size = byte_off;
    def.flat_leaf_count = leaf_off;
    structs_.push_back(std::move(def));
    return id;
}

const StructDef* TypeRegistry::find_struct(uint32_t id) const noexcept {
    return id < structs_.size() ? &structs_[id] : nullptr;
}

const StructDef* TypeRegistry::find_struct(std::string_view name) const noexcept {
    for (const auto& s : structs_) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

std::vector<ScalarKind> TypeRegistry::flatten_scalars(const Type& t) const {
    std::vector<ScalarKind> out;
    if (t.is_scalar()) {
        out.push_back(t.scalar);
    } else if (t.is_vector()) {
        out.assign(t.dimension, t.scalar);
    } else if (t.is_array()) {
        auto elem = flatten_scalars(t.element_type());
        out.reserve(elem.size() * t.dimension);
        for (uint32_t i = 0; i < t.dimension; ++i) {
            out.insert(out.end(), elem.begin(), elem.end());
        }
    } else if (t.is_struct()) {
        if (const StructDef* sd = find_struct(t.struct_id)) {
            for (const auto& f : sd->fields) {
                auto sub = flatten_scalars(f.type);
                out.insert(out.end(), sub.begin(), sub.end());
            }
        }
    }
    return out;
}

uint32_t TypeRegistry::flat_leaf_count(const Type& t) const {
    if (t.is_void()) return 0;
    if (t.is_scalar()) return 1;
    if (t.is_vector()) return t.dimension;
    if (t.is_array()) return flat_leaf_count(t.element_type()) * t.dimension;
    if (t.is_struct()) {
        if (const StructDef* sd = find_struct(t.struct_id)) {
            return sd->flat_leaf_count;
        }
        return 0;
    }
    return 1;
}

uint32_t TypeRegistry::byte_size(const Type& t) const {
    if (t.is_struct()) {
        if (const StructDef* sd = find_struct(t.struct_id)) {
            return sd->byte_size;
        }
        return 0;
    }
    if (t.is_array()) {
        return byte_size(t.element_type()) * t.dimension;
    }
    return t.size_in_bytes();
}

} // namespace brokernel

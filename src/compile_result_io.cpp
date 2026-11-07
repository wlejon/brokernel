// A compile's result as bytes and back, for the compile cache: every field
// a compile produces, in a fixed order (compile_cache_internal.h).
#include "compile_cache_internal.h"

#include <algorithm>

namespace brokernel::detail {

namespace {

constexpr int kMaxTypeDepth = 64;

void put(ByteWriter& w, const Type& t) {
    w.u8(static_cast<uint8_t>(t.kind));
    w.u8(static_cast<uint8_t>(t.scalar));
    w.u32(t.dimension);
    w.i32(t.stencil_radius);
    w.u8(static_cast<uint8_t>(t.access));
    w.u32(t.struct_id);
    w.str(t.struct_name);
    w.u8(static_cast<uint8_t>(t.elem_kind));
    w.u32(t.elem_lanes);
    w.u8(t.image_flags);
    w.boolean(t.nested_elem != nullptr);
    if (t.nested_elem) put(w, *t.nested_elem);
}

Type get_type(ByteReader& r, int depth = 0) {
    Type t;
    t.kind = static_cast<TypeKind>(r.u8());
    t.scalar = static_cast<ScalarKind>(r.u8());
    t.dimension = r.u32();
    t.stencil_radius = r.i32();
    t.access = static_cast<AccessMode>(r.u8());
    t.struct_id = r.u32();
    t.struct_name = r.str();
    t.elem_kind = static_cast<TypeKind>(r.u8());
    t.elem_lanes = r.u32();
    t.image_flags = r.u8();
    if (r.boolean()) {
        if (depth >= kMaxTypeDepth) {
            r.fail();
            return t;
        }
        t.nested_elem = std::make_shared<const Type>(get_type(r, depth + 1));
    }
    return t;
}

void put(ByteWriter& w, const ConstValue& c) {
    put(w, c.type);
    w.i64(c.int_val);
    w.f64(c.float_val);
    w.boolean(c.bool_val);
    w.u64(c.elements.size());
    for (const auto& e : c.elements) put(w, e);
}

ConstValue get_const(ByteReader& r, int depth = 0) {
    ConstValue c;
    c.type = get_type(r);
    c.int_val = r.i64();
    c.float_val = r.f64();
    c.bool_val = r.boolean();
    const size_t n = r.count(8);
    if (n && depth >= kMaxTypeDepth) {
        r.fail();
        return c;
    }
    for (size_t i = 0; i < n && r.ok(); ++i) c.elements.push_back(get_const(r, depth + 1));
    return c;
}

void put(ByteWriter& w, const SourcePos& p) {
    w.u32(p.line);
    w.u32(p.column);
    w.u32(p.offset);
}

SourcePos get_pos(ByteReader& r) {
    SourcePos p;
    p.line = r.u32();
    p.column = r.u32();
    p.offset = r.u32();
    return p;
}

void put(ByteWriter& w, const ProbeInfo& p) {
    w.str(p.name);
    put(w, p.type);
    w.u32(p.index);
}

ProbeInfo get_probe(ByteReader& r) {
    ProbeInfo p;
    p.name = r.str();
    p.type = get_type(r);
    p.index = r.u32();
    return p;
}

void put(ByteWriter& w, const KernelParamInfo& p) {
    w.str(p.name);
    put(w, p.type);
    w.u8(static_cast<uint8_t>(p.role));
    w.u8(static_cast<uint8_t>(p.access));
    w.u32(p.dimension);
    w.i32(p.declared_radius);
    w.i32(p.inferred_radius);
    for (int32_t h : p.halo) w.i32(h);
    w.u64(p.stencil_offsets.size());
    for (const auto& o : p.stencil_offsets) {
        w.i32(o.dx);
        w.i32(o.dy);
        w.i32(o.dz);
    }
    w.u32(p.mir_param_index);
    w.u32(p.mir_param_count);
    w.u32(p.element_bytes);
    w.u32(p.grid_cover_dim);
    w.u32(p.push_constant_offset);
    w.u32(p.push_constant_size);
    w.u32(p.descriptor_set);
    w.u32(p.descriptor_binding);
    w.u32(p.image_format);
}

KernelParamInfo get_param(ByteReader& r) {
    KernelParamInfo p;
    p.name = r.str();
    p.type = get_type(r);
    p.role = static_cast<ParamRole>(r.u8());
    p.access = static_cast<AccessMode>(r.u8());
    p.dimension = r.u32();
    p.declared_radius = r.i32();
    p.inferred_radius = r.i32();
    for (int32_t& h : p.halo) h = r.i32();
    const size_t n = r.count(12);
    p.stencil_offsets.resize(n);
    for (auto& o : p.stencil_offsets) {
        o.dx = r.i32();
        o.dy = r.i32();
        o.dz = r.i32();
    }
    p.mir_param_index = r.u32();
    p.mir_param_count = r.u32();
    p.element_bytes = r.u32();
    p.grid_cover_dim = r.u32();
    p.push_constant_offset = r.u32();
    p.push_constant_size = r.u32();
    p.descriptor_set = r.u32();
    p.descriptor_binding = r.u32();
    p.image_format = r.u32();
    return p;
}

template <typename T, typename Put>
void put_list(ByteWriter& w, const std::vector<T>& v, Put each) {
    w.u64(v.size());
    for (const auto& x : v) each(x);
}

template <typename T, typename Get>
void get_list(ByteReader& r, std::vector<T>& v, Get each) {
    const size_t n = r.count();
    v.clear();
    v.reserve(n);
    for (size_t i = 0; i < n && r.ok(); ++i) v.push_back(each());
}

void put_strings(ByteWriter& w, const std::vector<std::string>& v) {
    put_list(w, v, [&](const std::string& s) { w.str(s); });
}

void get_strings(ByteReader& r, std::vector<std::string>& v) {
    get_list(r, v, [&] { return r.str(); });
}

void put(ByteWriter& w, const brass::target::SpirvKernel& s) {
    w.str(s.entry);
    w.u8(static_cast<uint8_t>(s.stage));
    w.u64(s.words.size());
    w.raw(s.words.data(), s.words.size() * sizeof(uint32_t));
    put_list(w, s.params, [&](const brass::target::SpirvParam& p) {
        w.u8(static_cast<uint8_t>(p.type.kind()));
        w.u32(p.offset);
        w.u32(p.size);
    });
    put_list(w, s.descriptors, [&](const brass::target::SpirvDescriptorBinding& d) {
        w.u32(d.set);
        w.u32(d.binding);
        w.u8(static_cast<uint8_t>(d.kind));
        w.u8(static_cast<uint8_t>(d.dim));
        w.boolean(d.arrayed);
        w.boolean(d.depth);
        w.u8(static_cast<uint8_t>(d.scalar));
        w.u32(d.format);
        w.u32(d.bytes);
    });
    w.u32(s.push_constant_bytes);
    w.u32(s.shared_bytes);
    for (uint32_t l : s.local_size) w.u32(l);
    w.boolean(s.local_size_spec_constants);
    put_strings(w, s.capabilities);
    put_strings(w, s.extensions);
}

void get_spirv(ByteReader& r, brass::target::SpirvKernel& s) {
    s.entry = r.str();
    s.stage = static_cast<brass::target::SpirvStage>(r.u8());
    const size_t n = r.count(sizeof(uint32_t));
    s.words.resize(n);
    r.raw(s.words.data(), n * sizeof(uint32_t));
    get_list(r, s.params, [&] {
        brass::target::SpirvParam p;
        p.type = brass::Type(static_cast<brass::TypeKind>(r.u8()));
        p.offset = r.u32();
        p.size = r.u32();
        return p;
    });
    get_list(r, s.descriptors, [&] {
        brass::target::SpirvDescriptorBinding d;
        d.set = r.u32();
        d.binding = r.u32();
        d.kind = static_cast<brass::target::SpirvDescriptorKind>(r.u8());
        d.dim = static_cast<brass::target::SpirvImageDim>(r.u8());
        d.arrayed = r.boolean();
        d.depth = r.boolean();
        d.scalar = static_cast<brass::target::SpirvImageScalar>(r.u8());
        d.format = r.u32();
        d.bytes = r.u32();
        return d;
    });
    s.push_constant_bytes = r.u32();
    s.shared_bytes = r.u32();
    for (uint32_t& l : s.local_size) l = r.u32();
    s.local_size_spec_constants = r.boolean();
    get_strings(r, s.capabilities);
    get_strings(r, s.extensions);
}

void put(ByteWriter& w, const CompiledKernel& k) {
    w.str(k.name);
    w.boolean(k.is_entry_kernel);
    w.boolean(k.is_pub);
    w.u8(static_cast<uint8_t>(k.stage));
    w.u32(k.grid_dim);
    for (uint32_t l : k.local_size) w.u32(l);
    put_list(w, k.params, [&](const KernelParamInfo& p) { put(w, p); });
    put_list(w, k.all_mir_params, [&](const KernelParamInfo& p) { put(w, p); });
    put_list(w, k.block_bindings, [&](const KernelParamInfo& p) { put(w, p); });
    put_list(w, k.probes, [&](const ProbeInfo& p) { put(w, p); });
    put_list(w, k.function_costs, [&](const CompiledKernel::FunctionCost& c) {
        w.str(c.name);
        w.u32(c.self);
        w.u32(c.total);
    });
    w.u64(k.code_hash);
    w.boolean(k.probes_compiled);
    put(w, k.return_type);
    w.u32(k.return_flat_leaves);
    w.u32(k.return_byte_size);
    put_list(w, k.return_leaf_kinds, [&](ScalarKind s) { w.u8(static_cast<uint8_t>(s)); });
    w.boolean(k.uses_f64);
    w.boolean(k.cpu_only);
    w.boolean(k.uses_grid_coords);
    w.boolean(k.uses_barriers_or_shared);
    w.i32(k.max_stencil_radius);
    for (int32_t h : k.stencil_halo) w.i32(h);
    put(w, k.spirv);
    w.str(k.spirv_disassembly);
    w.str(k.ptx);
    w.boolean(k.spirv_equivalent);
    w.boolean(k.has_cpu_build());
}

void get_kernel(ByteReader& r, CompiledKernel& k, uint8_t& had_cpu) {
    k.name = r.str();
    k.is_entry_kernel = r.boolean();
    k.is_pub = r.boolean();
    k.stage = static_cast<Stage>(r.u8());
    k.grid_dim = r.u32();
    for (uint32_t& l : k.local_size) l = r.u32();
    get_list(r, k.params, [&] { return get_param(r); });
    get_list(r, k.all_mir_params, [&] { return get_param(r); });
    get_list(r, k.block_bindings, [&] { return get_param(r); });
    get_list(r, k.probes, [&] { return get_probe(r); });
    get_list(r, k.function_costs, [&] {
        CompiledKernel::FunctionCost c;
        c.name = r.str();
        c.self = r.u32();
        c.total = r.u32();
        return c;
    });
    k.code_hash = r.u64();
    k.probes_compiled = r.boolean();
    k.return_type = get_type(r);
    k.return_flat_leaves = r.u32();
    k.return_byte_size = r.u32();
    get_list(r, k.return_leaf_kinds, [&] { return static_cast<ScalarKind>(r.u8()); });
    k.uses_f64 = r.boolean();
    k.cpu_only = r.boolean();
    k.uses_grid_coords = r.boolean();
    k.uses_barriers_or_shared = r.boolean();
    k.max_stencil_radius = r.i32();
    for (int32_t& h : k.stencil_halo) h = r.i32();
    get_spirv(r, k.spirv);
    k.spirv_disassembly = r.str();
    k.ptx = r.str();
    k.spirv_equivalent = r.boolean();
    had_cpu = r.boolean() ? 1 : 0;
}

void put(ByteWriter& w, const StructDef& s) {
    w.u32(s.id);
    w.str(s.name);
    put_list(w, s.fields, [&](const StructField& f) {
        w.str(f.name);
        put(w, f.type);
        w.u32(f.byte_offset);
        w.u32(f.flat_leaf_offset);
        w.u32(f.flat_leaf_count);
    });
    w.u32(s.byte_size);
    w.u32(s.flat_leaf_count);
}

// The registry the compile had: its structs registered again in their order
// (which lays them out as it did), checked against what was stored.
bool get_types(ByteReader& r, TypeRegistry& types) {
    const size_t n = r.count();
    for (size_t i = 0; i < n && r.ok(); ++i) {
        StructDef s;
        s.id = r.u32();
        s.name = r.str();
        get_list(r, s.fields, [&] {
            StructField f;
            f.name = r.str();
            f.type = get_type(r);
            f.byte_offset = r.u32();
            f.flat_leaf_offset = r.u32();
            f.flat_leaf_count = r.u32();
            return f;
        });
        s.byte_size = r.u32();
        s.flat_leaf_count = r.u32();
        if (!r.ok()) return false;
        const uint32_t id = types.register_struct(s);
        const StructDef* got = types.find_struct(id);
        if (id != s.id || !got || got->byte_size != s.byte_size || got->flat_leaf_count != s.flat_leaf_count ||
            got->fields.size() != s.fields.size()) {
            return false;
        }
        for (size_t f = 0; f < s.fields.size(); ++f) {
            const StructField& a = got->fields[f];
            const StructField& b = s.fields[f];
            if (a.byte_offset != b.byte_offset || a.flat_leaf_offset != b.flat_leaf_offset ||
                a.flat_leaf_count != b.flat_leaf_count) {
                return false;
            }
        }
    }
    return r.ok();
}

} // namespace

void encode_options(const CompileOptions& o, ByteWriter& w) {
    w.str(o.entry_name);
    w.boolean(o.compile_all_functions);
    w.u8(static_cast<uint8_t>(o.default_stage));
    for (uint32_t l : o.local_size) w.u32(l);
    w.boolean(o.local_size_spec_constants);
    w.u32(o.spirv_version);
    w.boolean(o.emit_spirv);
    w.boolean(o.emit_spirv_disassembly);
    w.boolean(o.emit_ptx);
    w.boolean(o.compile_cpu);
    w.boolean(o.verify_spirv_equivalence);
    w.i32(o.opt_level);
    w.boolean(o.enable_probes);
    w.boolean(o.enable_fragment_probes);
    std::vector<const std::pair<const std::string, ConstValue>*> knobs;
    for (const auto& kv : o.knob_overrides) knobs.push_back(&kv);
    std::sort(knobs.begin(), knobs.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
    w.u64(knobs.size());
    for (const auto* kv : knobs) {
        w.str(kv->first);
        put(w, kv->second);
    }
}

void encode_modules(const std::vector<ModuleRecord>& mods, ByteWriter& w) {
    put_list(w, mods, [&](const ModuleRecord& m) {
        w.str(m.name);
        w.str(m.importing_dir);
        w.str(m.key);
        w.str(m.path);
        w.u64(m.source.lo);
        w.u64(m.source.hi);
    });
}

bool decode_modules(ByteReader& r, std::vector<ModuleRecord>& mods) {
    get_list(r, mods, [&] {
        ModuleRecord m;
        m.name = r.str();
        m.importing_dir = r.str();
        m.key = r.str();
        m.path = r.str();
        m.source.lo = r.u64();
        m.source.hi = r.u64();
        return m;
    });
    return r.ok();
}

void encode_result(const CompileResult& res, ByteWriter& w) {
    w.str(res.module_name);
    put_list(w, res.diagnostics.items(), [&](const Diagnostic& d) {
        w.u8(static_cast<uint8_t>(d.severity));
        w.str(d.span.file);
        put(w, d.span.start);
        put(w, d.span.end);
        w.str(d.message);
        w.str(d.line_text);
    });
    put_list(w, res.types.structs(), [&](const StructDef& s) { put(w, s); });
    put_list(w, res.resources, [&](const ResourceInfo& ri) {
        w.str(ri.name);
        w.str(ri.module);
        w.u32(ri.set);
        w.u32(ri.binding);
        put(w, ri.type);
        w.u32(ri.image_format);
    });
    put_list(w, res.constants, [&](const ConstInfo& c) {
        w.str(c.name);
        w.str(c.module);
        put(w, c.type);
        put(w, c.value);
        w.boolean(c.is_knob);
    });
    put_list(w, res.probes, [&](const ProbeInfo& p) { put(w, p); });
    put_strings(w, res.exported_functions);
    put_list(w, res.kernels, [&](const CompiledKernel& k) { put(w, k); });
}

bool decode_result(ByteReader& r, CompileResult& res, std::vector<uint8_t>& had_cpu) {
    res.module_name = r.str();
    const size_t ndiag = r.count();
    for (size_t i = 0; i < ndiag && r.ok(); ++i) {
        const auto sev = static_cast<Severity>(r.u8());
        SourceSpan span;
        span.file = r.str();
        span.start = get_pos(r);
        span.end = get_pos(r);
        std::string msg = r.str();
        std::string line = r.str();
        res.diagnostics.add(sev, std::move(span), std::move(msg), std::move(line));
    }
    if (!get_types(r, res.types)) return false;
    get_list(r, res.resources, [&] {
        ResourceInfo ri;
        ri.name = r.str();
        ri.module = r.str();
        ri.set = r.u32();
        ri.binding = r.u32();
        ri.type = get_type(r);
        ri.image_format = r.u32();
        return ri;
    });
    get_list(r, res.constants, [&] {
        ConstInfo c;
        c.name = r.str();
        c.module = r.str();
        c.type = get_type(r);
        c.value = get_const(r);
        c.is_knob = r.boolean();
        return c;
    });
    get_list(r, res.probes, [&] { return get_probe(r); });
    get_strings(r, res.exported_functions);
    const size_t nk = r.count();
    res.kernels.clear();
    res.kernels.reserve(nk);
    had_cpu.assign(nk, 0);
    for (size_t i = 0; i < nk && r.ok(); ++i) {
        res.kernels.emplace_back();
        get_kernel(r, res.kernels.back(), had_cpu[i]);
    }
    return r.ok() && r.at_end() && !res.diagnostics.has_errors();
}

} // namespace brokernel::detail

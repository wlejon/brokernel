// Image and sampler methods, lowered to brass's texture calls on the
// parameter's descriptor (set, binding). A texture parameter's value is a
// placeholder the lowerer maps back to the parameter, so a texture passed to
// a helper function, inlined, still names its descriptor. Texels come back
// as four lanes (f32, or i32 for a u32 texture) of which the texel type takes
// its first; a size as four i32 lanes.

#include "lowering_internal.h"

namespace brokernel::detail {

namespace {

brass::codegen::KernelBuilder::ImageRef image_ref(const CheckedParam& p) {
    brass::codegen::KernelBuilder::ImageRef r;
    r.set = p.descriptor_set;
    r.binding = p.descriptor_binding;
    const Type& t = p.type;
    if (t.image_cube()) r.dim = brass::target::SpirvImageDim::Cube;
    else if (t.dimension == 1) r.dim = brass::target::SpirvImageDim::D1;
    else if (t.dimension == 3) r.dim = brass::target::SpirvImageDim::D3;
    else r.dim = brass::target::SpirvImageDim::D2;
    r.arrayed = t.image_arrayed();
    r.depth = t.image_shadow();
    r.scalar = t.scalar == ScalarKind::U32 ? brass::target::SpirvImageScalar::U32 : brass::target::SpirvImageScalar::F32;
    r.format = t.is_image() ? p.image_format : 0;
    if (t.image_separate()) r.sampler_set = r.sampler_binding = -2;   // read with no sampler, until given one
    return r;
}

} // namespace

ValueList MirBodyLowerer::lower_texture_call(const Expr* e) {
    brass::Builder& mb = kb.builder();
    const Type& rt = e->lhs->resolved_type;
    const std::string& m = e->name;
    ValueList recv = lower_expr(e->lhs.get());
    auto found = recv.empty() ? texture_param.end() : texture_param.find(recv[0]);
    if (found == texture_param.end()) {
        diags.error(e->span, "a texture must be a kernel parameter (with @binding) to be used");
        return make_zero_leaves(e->resolved_type);
    }
    const CheckedParam& param = root_fn.params[found->second];
    auto img = image_ref(param);
    // A separate texture's filtering methods name their sampler first.
    size_t first = 0;
    if (rt.image_separate() && m != "size" && m != "levels" && m != "fetch") {
        ValueList s = lower_expr(e->args[0].get());
        auto sp = s.empty() ? texture_param.end() : texture_param.find(s[0]);
        if (sp == texture_param.end()) {
            diags.error(e->args[0]->span, "a sampler must be a kernel parameter (with @binding) to be used");
            return make_zero_leaves(e->resolved_type);
        }
        const CheckedParam& sparam = root_fn.params[sp->second];
        img.sampler_set = static_cast<int32_t>(sparam.descriptor_set);
        img.sampler_binding = static_cast<int32_t>(sparam.descriptor_binding);
        first = 1;
    }

    auto arg = [&](size_t i) { return lower_expr(e->args[first + i].get()); };
    auto scalar = [&](size_t i) { return arg(i)[0]; };
    // The first `lanes` lanes of a four-lane result.
    auto lanes_of = [&](brass::Value* v, uint32_t lanes) {
        ValueList out;
        for (uint32_t i = 0; i < lanes; ++i) out.push_back(mb.build_vextract_lane(v, i));
        return out;
    };
    auto texel = [&](brass::Value* v) { return lanes_of(v, rt.element_type().is_vector() ? rt.elem_lanes : 1); };

    if (m == "size") {
        brass::Value* s = rt.is_image() ? kb.image_size(img) : kb.texture_size(img, scalar(0));
        const Type& t = e->resolved_type;
        return lanes_of(s, t.is_vector() ? t.dimension : 1);
    }
    if (m == "levels") return {kb.texture_levels(img)};
    if (m == "load") return texel(kb.image_load(img, arg(0)));
    if (m == "store") {
        ValueList v = arg(1);
        brass::Value* zero = rt.scalar == ScalarKind::U32 ? b.build_iconst_i32(0) : b.build_fconst_f32(0.0f);
        brass::Value* vec = mb.build_vbroadcast(rt.scalar == ScalarKind::U32 ? brass::Type::i32x4() : brass::Type::f32x4(), zero);
        for (uint32_t i = 0; i < v.size() && i < 4; ++i) vec = mb.build_vinsert_lane(vec, v[i], i);
        kb.image_store(img, arg(0), vec);
        return {};
    }
    if (m == "sample") return texel(kb.texture_sample(img, arg(0)));
    if (m == "sample_offset") return texel(kb.texture_sample(img, arg(0), arg(1)));
    if (m == "sample_lod") return texel(kb.texture_sample_lod(img, arg(0), scalar(1)));
    if (m == "sample_lod_offset") return texel(kb.texture_sample_lod(img, arg(0), scalar(1), arg(2)));
    if (m == "sample_grad") return texel(kb.texture_sample_grad(img, arg(0), arg(1), arg(2)));
    if (m == "sample_grad_offset") return texel(kb.texture_sample_grad(img, arg(0), arg(1), arg(2), arg(3)));
    if (m == "fetch") return texel(kb.texture_fetch(img, arg(0), scalar(1)));
    if (m == "gather") return lanes_of(kb.texture_gather(img, arg(0), scalar(1)), 4);
    if (m == "compare") return {kb.texture_compare(img, arg(0), scalar(1))};
    if (m == "compare_lod") return {kb.texture_compare(img, arg(0), scalar(1), scalar(2))};
    diags.error(e->span, "type '" + rt.to_string() + "' has no method '" + m + "'");
    return make_zero_leaves(e->resolved_type);
}

} // namespace brokernel::detail

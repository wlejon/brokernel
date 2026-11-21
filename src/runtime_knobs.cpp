// Zero-recompile runtime `@knob` storage, MIR lowering, and GPU push-constant packing.
#include "compile_internal.h"
#include "lowering_internal.h"

#include <atomic>
#include <cstring>
#include <mutex>

namespace brokernel {

struct RuntimeKnobStorage {
    std::mutex mu;
    std::atomic<uint64_t> generation{1};
    std::vector<uint8_t> bytes;
};

namespace {

void flatten_const_scalars(const ConstValue& cv, const Type& expected, const TypeRegistry& types,
                           std::vector<ConstValue>& out) {
    if (cv.elements.empty()) {
        out.push_back(cv);
        return;
    }
    if (cv.type.is_scalar()) {
        out.push_back(cv);
        return;
    }
    for (const auto& el : cv.elements) {
        flatten_const_scalars(el, expected, types, out);
    }
}

bool pack_knob_value(const ConstValue& input, const RuntimeKnobInfo& info, const TypeRegistry& types, uint8_t* base) {
    ConstValue conv = input;
    // Convert scalar/composite ConstValue leaf-by-leaf to match info.type.
    std::vector<ConstValue> raw_leaves;
    flatten_const_scalars(input, info.type, types, raw_leaves);
    auto kinds = types.flatten_scalars(info.type);
    if (kinds.empty()) return false;
    if (raw_leaves.size() == 1 && kinds.size() > 1) {
        raw_leaves.resize(kinds.size(), raw_leaves[0]);
    }
    if (raw_leaves.size() != kinds.size()) return false;

    uint32_t off = info.byte_offset;
    for (size_t i = 0; i < kinds.size(); ++i) {
        const ConstValue& src = raw_leaves[i];
        const ScalarKind sk = kinds[i];
        double fval = src.type.is_float_scalar() ? src.float_val
                    : src.type.is_bool() ? (src.bool_val ? 1.0 : 0.0)
                                         : static_cast<double>(src.int_val);
        int64_t ival = src.type.is_float_scalar() ? static_cast<int64_t>(src.float_val)
                     : src.type.is_bool() ? (src.bool_val ? 1 : 0)
                                          : src.int_val;
        if (sk == ScalarKind::F32) {
            float v = static_cast<float>(fval);
            std::memcpy(base + off, &v, 4);
            off += 4;
        } else if (sk == ScalarKind::F64) {
            double v = fval;
            std::memcpy(base + off, &v, 8);
            off += 8;
        } else if (sk == ScalarKind::I64) {
            int64_t v = ival;
            std::memcpy(base + off, &v, 8);
            off += 8;
        } else if (sk == ScalarKind::U64) {
            uint64_t v = static_cast<uint64_t>(ival);
            std::memcpy(base + off, &v, 8);
            off += 8;
        } else if (sk == ScalarKind::Bool) {
            int32_t v = (src.type.is_bool() ? src.bool_val : (fval != 0.0)) ? 1 : 0;
            std::memcpy(base + off, &v, 4);
            off += 4;
        } else if (sk == ScalarKind::U32) {
            uint32_t v = static_cast<uint32_t>(ival);
            std::memcpy(base + off, &v, 4);
            off += 4;
        } else {
            int32_t v = static_cast<int32_t>(ival);
            std::memcpy(base + off, &v, 4);
            off += 4;
        }
    }
    return true;
}

} // namespace

void CompiledKernel::init_runtime_knobs() {
    if (runtime_knobs.empty()) {
        runtime_knob_storage.reset();
        return;
    }
    auto storage = std::make_shared<RuntimeKnobStorage>();
    storage->bytes.assign(std::max<uint32_t>(runtime_knob_bytes, 4), 0);
    // Pack defaults with empty TypeRegistry fallback using scalar flattening.
    TypeRegistry empty_reg;
    for (const auto& rk : runtime_knobs) {
        pack_knob_value(rk.default_value, rk, empty_reg, storage->bytes.data());
    }
    runtime_knob_storage = std::move(storage);
}

bool CompiledKernel::set_knobs(const std::unordered_map<std::string, ConstValue>& overrides) const {
    if (runtime_knobs.empty() || !runtime_knob_storage) return false;
    std::lock_guard<std::mutex> lock(runtime_knob_storage->mu);
    std::vector<uint8_t> staged = runtime_knob_storage->bytes;
    TypeRegistry empty_reg;
    for (const auto& [name, val] : overrides) {
        bool matched = false;
        for (const auto& rk : runtime_knobs) {
            if (rk.key == name || rk.name == name) {
                if (!pack_knob_value(val, rk, empty_reg, staged.data())) {
                    return false;
                }
                matched = true;
            }
        }
        if (!matched) return false;
    }
    runtime_knob_storage->bytes.swap(staged);
    runtime_knob_storage->generation.fetch_add(1, std::memory_order_release);
    return true;
}

const void* CompiledKernel::runtime_knob_data() const {
    return runtime_knob_storage ? runtime_knob_storage->bytes.data() : nullptr;
}

uint64_t CompiledKernel::runtime_knob_generation() const {
    return runtime_knob_storage ? runtime_knob_storage->generation.load(std::memory_order_acquire) : 0;
}

namespace detail {

ValueList MirBodyLowerer::lower_runtime_knob(const Expr* e) {
    if (!e || !e->is_runtime_knob) return {};
    if (is_gpu) {
        auto it = gpu_knob_values.find(e->runtime_knob_index);
        if (it != gpu_knob_values.end()) return it->second;
        return make_zero_leaves(e->resolved_type);
    }
    if (!cpu_knobs_ptr || e->runtime_knob_index >= mod.runtime_knobs.size()) {
        return make_zero_leaves(e->resolved_type);
    }
    const RuntimeKnobLayout& rk = mod.runtime_knobs[e->runtime_knob_index];
    auto kinds = mod.types.flatten_scalars(rk.type);
    brass::Builder& rb = kb.builder();
    ValueList out;
    out.reserve(kinds.size());
    uint32_t off = rk.byte_offset;
    for (ScalarKind sk : kinds) {
        brass::Type mt = to_mir_scalar_type(sk);
        brass::Value* v = rb.build_load(mt, cpu_knobs_ptr, static_cast<int32_t>(off));
        if (sk == ScalarKind::Bool && b.vector_mode()) {
            v = rb.build_ne(v, rb.build_iconst_i32(0));
        }
        out.push_back(v);
        off += static_cast<uint32_t>(mt.size_in_bytes());
    }
    return out;
}

void append_gpu_runtime_knob_args(const CompiledKernel& k, std::vector<brass::gpu::VulkanArg>& vk_args) {
    if (k.runtime_knobs.empty() || !k.runtime_knob_storage) return;
    std::lock_guard<std::mutex> lock(k.runtime_knob_storage->mu);
    const uint8_t* base = k.runtime_knob_storage->bytes.data();
    TypeRegistry empty_reg;
    for (const auto& rk : k.runtime_knobs) {
        auto kinds = empty_reg.flatten_scalars(rk.type);
        uint32_t off = rk.byte_offset;
        for (ScalarKind sk : kinds) {
            if (sk == ScalarKind::F32) {
                float v = 0.0f;
                std::memcpy(&v, base + off, 4);
                vk_args.emplace_back(v);
                off += 4;
            } else if (sk == ScalarKind::F64) {
                double v = 0.0;
                std::memcpy(&v, base + off, 8);
                vk_args.emplace_back(v);
                off += 8;
            } else if (sk == ScalarKind::U32) {
                uint32_t v = 0;
                std::memcpy(&v, base + off, 4);
                vk_args.emplace_back(v);
                off += 4;
            } else if (sk == ScalarKind::I64) {
                int64_t v = 0;
                std::memcpy(&v, base + off, 8);
                vk_args.emplace_back(v);
                off += 8;
            } else if (sk == ScalarKind::U64) {
                uint64_t v = 0;
                std::memcpy(&v, base + off, 8);
                vk_args.emplace_back(v);
                off += 8;
            } else {
                int32_t v = 0;
                std::memcpy(&v, base + off, 4);
                vk_args.emplace_back(v);
                off += 4;
            }
        }
    }
}

} // namespace detail
} // namespace brokernel

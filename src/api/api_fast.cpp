#include "api_internal.h"
#include "embed/embed_native.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brokernel::api {

struct FastEvalPlan {
    bool checked = false;
    bool eligible = false;
    void (*fn_ptr)(const uint64_t*, const void*, void*) = nullptr;
    size_t passed_count = 0;
    ScalarKind param_kinds[8]{};
    ScalarKind ret_kind = ScalarKind::Void;
    bool has_return = false;
};

namespace {

struct alignas(8) FastCpuContext {
    int32_t gx, gy, gz;
    int32_t gw, gh, gd;
    int32_t lid_x, lid_y, lid_z;
    int32_t wgid_x, wgid_y, wgid_z;
    int32_t lsz_x, lsz_y, lsz_z;
    int32_t nwg_x, nwg_y, nwg_z;
    void* shared_base;
    void (*barrier_fn)(void*);
    void* barrier_ctx;
    void* probe_buf;
};

void fastLoneBarrier(void*) {}

void initFastPlan(KernelHandle& h) {
    if (h.fast_eval && h.fast_eval->checked) return;
    auto plan = std::make_shared<FastEvalPlan>();
    plan->checked = true;
    const CompiledKernel* k = h.kernel.get();
    if (!k || k->grid_dim != 0 || k->uses_barriers_or_shared || !k->block_bindings.empty() ||
        !k->runtime_knobs.empty() || (k->uses_telemetry && k->telemetry_slots > 0) ||
        k->cpu_shared_bytes() > 0) {
        h.fast_eval = std::move(plan);
        return;
    }
    if (!k->cpu_point_fn().is_valid()) {
        h.fast_eval = std::move(plan);
        return;
    }
    if (k->has_return_value() && !k->return_type.is_scalar()) {
        h.fast_eval = std::move(plan);
        return;
    }
    size_t pc = 0;
    for (const auto& p : k->params) {
        if (!isPassed(p)) continue;
        if (p.role != ParamRole::Scalar || pc >= 8) {
            h.fast_eval = std::move(plan);
            return;
        }
        plan->param_kinds[pc++] = p.type.scalar;
    }
    plan->passed_count = pc;
    plan->has_return = k->has_return_value();
    plan->ret_kind = k->return_type.scalar;
    plan->fn_ptr = k->cpu_point_fn().as<void (*)(const uint64_t*, const void*, void*)>();
    plan->eligible = (plan->fn_ptr != nullptr);
    h.fast_eval = std::move(plan);
}

bool packFastScalar(ScalarKind sk, Value v, uint64_t& out) {
    if (!ev::isNumber(v) && !ev::isBigInt(v) && !ev::isBool(v)) return false;
    switch (sk) {
        case ScalarKind::F32: {
            float f = static_cast<float>(ev::toDouble(v));
            uint32_t u = 0;
            std::memcpy(&u, &f, 4);
            out = u;
            return true;
        }
        case ScalarKind::F64: {
            double d = ev::toDouble(v);
            std::memcpy(&out, &d, 8);
            return true;
        }
        case ScalarKind::U32:
            out = static_cast<uint64_t>(static_cast<uint32_t>(ev::toUint64(v)));
            return true;
        case ScalarKind::I64:
            out = static_cast<uint64_t>(ev::toInt64(v));
            return true;
        case ScalarKind::U64:
            out = ev::toUint64(v);
            return true;
        case ScalarKind::Bool:
            out = ev::toBool(v) ? 1ull : 0ull;
            return true;
        default:
            out = static_cast<uint64_t>(static_cast<uint32_t>(ev::toInt64(v)));
            return true;
    }
}

Value decodeFastScalar(ScalarKind sk, uint64_t raw) {
    switch (sk) {
        case ScalarKind::F32: {
            float f = 0.0f;
            uint32_t u = static_cast<uint32_t>(raw);
            std::memcpy(&f, &u, 4);
            return ev::fromDouble(f);
        }
        case ScalarKind::F64: {
            double d = 0.0;
            std::memcpy(&d, &raw, 8);
            return ev::fromDouble(d);
        }
        case ScalarKind::U32:
            return ev::fromDouble(static_cast<uint32_t>(raw));
        case ScalarKind::I64: {
            int64_t x = 0;
            std::memcpy(&x, &raw, 8);
            return ev::fromDouble(static_cast<double>(x));
        }
        case ScalarKind::U64:
            return ev::fromDouble(static_cast<double>(raw));
        case ScalarKind::Bool:
            return ev::fromBool(static_cast<int32_t>(raw) != 0);
        default:
            return ev::fromDouble(static_cast<int32_t>(raw));
    }
}

const char* bronzeNativeTypeOf(ScalarKind sk) {
    switch (sk) {
        case ScalarKind::Void: return "void";
        case ScalarKind::F32:
        case ScalarKind::F64:  return "f64";
        case ScalarKind::Bool: return "bool";
        case ScalarKind::I32:
        case ScalarKind::U32:  return "i32";
        default: return nullptr;
    }
}

std::mutex g_exportedMu;
std::vector<brass::codegen::KernelFunction> g_exportedTrampolines;

} // namespace

bool tryFastEval(KernelHandle& h, std::span<const Value> args, Value& out) {
    if (!h.kernel || h.kernel->grid_dim != 0 || !h.kernel->cpu_built()) return false;
    initFastPlan(h);
    const FastEvalPlan* plan = h.fast_eval.get();
    if (!plan || !plan->eligible) return false;
    if (args.size() != plan->passed_count) return false;

    uint64_t slots[8]{};
    for (size_t i = 0; i < plan->passed_count; ++i) {
        if (!packFastScalar(plan->param_kinds[i], args[i], slots[i])) return false;
    }

    const int32_t lx = static_cast<int32_t>(std::max(1u, h.kernel->local_size[0]));
    const int32_t ly = static_cast<int32_t>(std::max(1u, h.kernel->local_size[1]));
    const int32_t lz = static_cast<int32_t>(std::max(1u, h.kernel->local_size[2]));
    FastCpuContext ctx{
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        lx, ly, lz, 1, 1, 1,
        nullptr, &fastLoneBarrier, nullptr, nullptr,
    };
    uint64_t ret = 0;
    plan->fn_ptr(slots, &ctx, &ret);
    out = plan->has_return ? decodeFastScalar(plan->ret_kind, ret) : ev::undefined();
    return true;
}

bool exportKernelAsBronzeNative(const CompiledKernel& k, std::string_view jsPath, std::string* err) {
    if (k.grid_dim != 0 || k.uses_barriers_or_shared || !k.block_bindings.empty() ||
        !k.runtime_knobs.empty() || (k.uses_telemetry && k.telemetry_slots > 0) ||
        k.cpu_shared_bytes() > 0) {
        if (err) *err = "exportNative only supports stateless scalar point kernels";
        return false;
    }
    if (!k.cpu_point_fn().is_valid()) {
        if (err) *err = "kernel has no valid CPU build";
        return false;
    }
    if (k.has_return_value() && !k.return_type.is_scalar()) {
        if (err) *err = "exportNative only supports scalar or void return types";
        return false;
    }

    ev::NativeSignature sig;
    const char* retStr = k.has_return_value() ? bronzeNativeTypeOf(k.return_type.scalar) : "void";
    if (!retStr) {
        if (err) *err = "unsupported return type for Bronze native";
        return false;
    }
    sig.returnType = retStr;

    std::vector<ScalarKind> paramKinds;
    std::vector<brass::Type> mirParamTypes;
    for (const auto& p : k.params) {
        if (!isPassed(p)) continue;
        if (p.role != ParamRole::Scalar) {
            if (err) *err = "exportNative parameter '" + p.name + "' must be scalar";
            return false;
        }
        const char* pstr = bronzeNativeTypeOf(p.type.scalar);
        if (!pstr) {
            if (err) *err = "unsupported scalar type for parameter '" + p.name + "'";
            return false;
        }
        sig.paramTypes.emplace_back(pstr);
        paramKinds.push_back(p.type.scalar);
        mirParamTypes.push_back((p.type.scalar == ScalarKind::F32 || p.type.scalar == ScalarKind::F64)
                                    ? brass::Type::f64()
                                    : brass::Type::i32());
    }

    brass::Type mirRetType = brass::Type::void_type();
    if (k.has_return_value()) {
        mirRetType = (k.return_type.scalar == ScalarKind::F32 || k.return_type.scalar == ScalarKind::F64)
                         ? brass::Type::f64()
                         : brass::Type::i32();
    }

    brass::Module mod("brokernel_native_trampoline");
    brass::Function* fn = mod.create_function("trampoline", mirRetType, mirParamTypes);
    brass::Builder b(mod);
    b.set_function(fn);
    brass::BasicBlock* entry = b.append_block("entry");
    std::vector<brass::Value*> args;
    args.reserve(mirParamTypes.size());
    for (const auto& pt : mirParamTypes) args.push_back(b.add_block_param(entry, pt));
    b.position_at_end(entry);

    const size_t slotBytes = std::max<size_t>(1, paramKinds.size()) * 8;
    brass::Value* slots = b.build_alloca(static_cast<uint32_t>(slotBytes), 8);
    brass::Value* ctx = b.build_alloca(sizeof(FastCpuContext), 8);
    brass::Value* retSlot = b.build_alloca(16, 8);
    brass::Value* fnPtrSlot = b.build_alloca(8, 8);

    // Initialize CpuPointContext fields needed by point kernels: gw=1,gh=1,gd=1, lsz=1, barrier_fn=&fastLoneBarrier
    b.build_store(brass::Type::i32(), ctx, 0, b.build_iconst_i32(0));
    b.build_store(brass::Type::i32(), ctx, 4, b.build_iconst_i32(0));
    b.build_store(brass::Type::i32(), ctx, 8, b.build_iconst_i32(0));
    b.build_store(brass::Type::i32(), ctx, 12, b.build_iconst_i32(1));
    b.build_store(brass::Type::i32(), ctx, 16, b.build_iconst_i32(1));
    b.build_store(brass::Type::i32(), ctx, 20, b.build_iconst_i32(1));
    b.build_store(brass::Type::i64(), ctx, 72, b.build_iconst_i64(0));
    b.build_store(brass::Type::i64(), ctx, 80, b.build_iconst_i64(reinterpret_cast<int64_t>(&fastLoneBarrier)));
    b.build_store(brass::Type::i64(), ctx, 88, b.build_iconst_i64(0));
    b.build_store(brass::Type::i64(), ctx, 96, b.build_iconst_i64(0));

    for (size_t i = 0; i < paramKinds.size(); ++i) {
        brass::Value* v = args[i];
        int32_t off = static_cast<int32_t>(i * 8);
        if (paramKinds[i] == ScalarKind::F32) {
            b.build_store(brass::Type::f32(), slots, off, b.build_fptrunc_f32_f64(v));
        } else if (paramKinds[i] == ScalarKind::F64) {
            b.build_store(brass::Type::f64(), slots, off, v);
        } else if (paramKinds[i] == ScalarKind::Bool) {
            b.build_store(brass::Type::i32(), slots, off, b.build_ne(v, b.build_iconst_i32(0)));
        } else {
            b.build_store(brass::Type::i32(), slots, off, v);
        }
    }

    b.build_store(brass::Type::i64(), fnPtrSlot, 0,
                  b.build_iconst_i64(reinterpret_cast<int64_t>(k.cpu_point_fn().entry_point())));
    brass::Value* targetFn = b.build_load(brass::Type::ptr(), fnPtrSlot, 0);
    b.build_call_indirect(targetFn, brass::Type::void_type(), {slots, ctx, retSlot});

    if (!k.has_return_value()) {
        b.build_ret_void();
    } else if (k.return_type.scalar == ScalarKind::F32) {
        brass::Value* f = b.build_load(brass::Type::f32(), retSlot, 0);
        b.build_ret(b.build_fpext_f64_f32(f));
    } else if (k.return_type.scalar == ScalarKind::F64) {
        b.build_ret(b.build_load(brass::Type::f64(), retSlot, 0));
    } else if (k.return_type.scalar == ScalarKind::Bool) {
        brass::Value* iv = b.build_load(brass::Type::i32(), retSlot, 0);
        b.build_ret(b.build_ne(iv, b.build_iconst_i32(0)));
    } else {
        b.build_ret(b.build_load(brass::Type::i32(), retSlot, 0));
    }

    brass::codegen::KernelJit jit;
    brass::codegen::KernelFunction compiled = jit.compile(*fn);
    if (!compiled.is_valid()) {
        if (err) *err = "failed to compile native trampoline";
        return false;
    }
    void* raw = compiled.entry_point();
    {
        std::lock_guard<std::mutex> lk(g_exportedMu);
        g_exportedTrampolines.push_back(std::move(compiled));
    }
    return ev::registerNative(jsPath, raw, sig, err);
}

Value js_kernelExportNative(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.exportNative: invalid receiver");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Kernel.exportNative: requires a JS dotted path string (e.g. 'bro.world.height')");
    }
    ensureCpuBuilt(*h->kernel);
    std::string err;
    if (!exportKernelAsBronzeNative(*h->kernel, ev::toUtf8(args[0]), &err)) {
        return ev::throwError("Kernel.exportNative: " + err);
    }
    return ev::undefined();
}

} // namespace brokernel::api

#include "api_internal.h"
#include "embed/embed_native.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brokernel::api {

struct FastParam {
    bool is_store = false;
    bool writable = false;
    ScalarKind scalar = ScalarKind::Void;
    uint32_t element_bytes = 0;
    uint32_t slot_offset = 0;
};

struct FastEvalPlan {
    bool checked = false;
    bool eligible = false;
    void (*fn_ptr)(const uint64_t*, const void*, void*) = nullptr;
    size_t passed_count = 0;
    size_t total_slots = 0;
    FastParam params[16]{};
    ScalarKind ret_kind = ScalarKind::Void;
    bool has_return = false;
};

namespace {

alignas(32) uint8_t s_emptyStore[256] = {};
std::atomic<uint64_t> g_reloadEpoch{1};

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

bool packFastScalarSlots(ScalarKind sk, Value v, uint64_t* dst) {
    if (!ev::isNumber(v) && !ev::isBigInt(v) && !ev::isBool(v)) return false;
    switch (sk) {
        case ScalarKind::F32: {
            float f = static_cast<float>(ev::toDouble(v));
            uint32_t u = 0;
            std::memcpy(&u, &f, 4);
            dst[0] = u;
            return true;
        }
        case ScalarKind::F64: {
            double d = ev::toDouble(v);
            std::memcpy(&dst[0], &d, 8);
            return true;
        }
        case ScalarKind::DF32: {
            double d = ev::toDouble(v);
            float hi = static_cast<float>(d);
            float lo = static_cast<float>(d - static_cast<double>(hi));
            uint32_t uhi = 0, ulo = 0;
            std::memcpy(&uhi, &hi, 4);
            std::memcpy(&ulo, &lo, 4);
            dst[0] = uhi;
            dst[1] = ulo;
            return true;
        }
        case ScalarKind::U32:
            dst[0] = static_cast<uint64_t>(static_cast<uint32_t>(ev::toUint64(v)));
            return true;
        case ScalarKind::I64:
            dst[0] = static_cast<uint64_t>(ev::toInt64(v));
            return true;
        case ScalarKind::U64:
            dst[0] = ev::toUint64(v);
            return true;
        case ScalarKind::Bool:
            dst[0] = ev::toBool(v) ? 1ull : 0ull;
            return true;
        default:
            dst[0] = static_cast<uint64_t>(static_cast<uint32_t>(ev::toInt64(v)));
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
        case ScalarKind::DF32: {
            float hi = 0.0f, lo = 0.0f;
            uint32_t uhi = static_cast<uint32_t>(raw & 0xFFFFFFFFu);
            uint32_t ulo = static_cast<uint32_t>(raw >> 32);
            std::memcpy(&hi, &uhi, 4);
            std::memcpy(&lo, &ulo, 4);
            return ev::fromDouble(static_cast<double>(hi) + static_cast<double>(lo));
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
        case ScalarKind::F64:
        case ScalarKind::DF32: return "f64";
        case ScalarKind::Bool: return "bool";
        case ScalarKind::I32:
        case ScalarKind::U32:  return "i32";
        default: return nullptr;
    }
}

bool bronzeNativeStoreArrayOf(const TypeRegistry& types,
                              const Type& elem,
                              const char*& outType,
                              uint32_t& outElemsPerItem) {
    auto kinds = types.flatten_scalars(elem);
    if (kinds.empty()) return false;
    auto bucketOf = [](ScalarKind sk, ScalarKind& bkt) -> bool {
        switch (sk) {
            case ScalarKind::F32:
            case ScalarKind::DF32: bkt = ScalarKind::F32; return true;
            case ScalarKind::F64:  bkt = ScalarKind::F64; return true;
            case ScalarKind::I32:
            case ScalarKind::Bool: bkt = ScalarKind::I32; return true;
            case ScalarKind::U32:  bkt = ScalarKind::U32; return true;
            default: return false;
        }
    };
    ScalarKind first{};
    if (!bucketOf(kinds[0], first)) return false;
    for (size_t i = 1; i < kinds.size(); ++i) {
        ScalarKind bkt{};
        if (!bucketOf(kinds[i], bkt) || bkt != first) return false;
    }
    uint32_t unitBytes = 0;
    switch (first) {
        case ScalarKind::F32: outType = "f32[]"; unitBytes = 4; break;
        case ScalarKind::F64: outType = "f64[]"; unitBytes = 8; break;
        case ScalarKind::I32: outType = "i32[]"; unitBytes = 4; break;
        case ScalarKind::U32: outType = "u32[]"; unitBytes = 4; break;
        default: return false;
    }
    const uint32_t elemBytes = types.byte_size(elem);
    if (elemBytes == 0 || elemBytes % unitBytes != 0) return false;
    outElemsPerItem = elemBytes / unitBytes;
    return true;
}

std::mutex g_exportedMu;
std::vector<brass::codegen::KernelFunction> g_exportedTrampolines;

void initFastPlan(KernelHandle& h) {
    if (h.fast_eval && h.fast_eval->checked) return;
    h.fast_eval = h.kernel ? buildFastEvalPlan(*h.kernel) : std::make_shared<FastEvalPlan>();
}

} // namespace

const std::atomic<uint64_t>& reloadEpochCounter() { return g_reloadEpoch; }
uint64_t currentReloadEpoch() { return g_reloadEpoch.load(std::memory_order_acquire); }
void bumpReloadEpoch() { g_reloadEpoch.fetch_add(1, std::memory_order_release); }

std::shared_ptr<FastEvalPlan> buildFastEvalPlan(const CompiledKernel& k) {
    auto plan = std::make_shared<FastEvalPlan>();
    plan->checked = true;
    if (k.uses_grid_coords || k.uses_barriers_or_shared || !k.block_bindings.empty() ||
        !k.runtime_knobs.empty() || (k.uses_telemetry && k.telemetry_slots > 0) ||
        k.cpu_shared_bytes() > 0 || !k.cpu_point_fn().is_valid()) {
        return plan;
    }
    if (k.has_return_value() && !k.return_type.is_scalar()) {
        return plan;
    }
    size_t pc = 0;
    size_t sc = 0;
    for (const auto& p : k.params) {
        if (!isPassed(p)) continue;
        if (pc >= 16) return plan;
        FastParam fp;
        fp.slot_offset = static_cast<uint32_t>(sc);
        if (p.role == ParamRole::Scalar && p.type.is_scalar()) {
            fp.scalar = p.type.scalar;
            sc += (fp.scalar == ScalarKind::DF32) ? 2 : 1;
        } else if (p.role == ParamRole::Store && p.type.dimension <= 1 && p.element_bytes > 0) {
            fp.is_store = true;
            fp.writable = (p.access != AccessMode::Read);
            fp.element_bytes = p.element_bytes;
            sc += 2;
        } else {
            return plan;
        }
        if (sc > 16) return plan;
        plan->params[pc++] = fp;
    }
    plan->passed_count = pc;
    plan->total_slots = sc;
    plan->has_return = k.has_return_value();
    plan->ret_kind = k.return_type.scalar;
    plan->fn_ptr = k.cpu_point_fn().as<void (*)(const uint64_t*, const void*, void*)>();
    plan->eligible = (plan->fn_ptr != nullptr);
    return plan;
}

bool fastPlanIsEligible(const FastEvalPlan& plan) { return plan.eligible; }
size_t fastPlanParamCount(const FastEvalPlan& plan) { return plan.passed_count; }
bool isFastScalarParam(const FastEvalPlan& plan, size_t argIdx) {
    return argIdx < plan.passed_count && !plan.params[argIdx].is_store;
}

bool packFastArgIntoSlots(const FastEvalPlan& plan, size_t argIdx, Value v, uint64_t* slots,
                          std::shared_ptr<GpuBuffer>* outWritableBuf, bool* outIsPinnedBuffer) {
    if (outIsPinnedBuffer) *outIsPinnedBuffer = false;
    if (argIdx >= plan.passed_count) return false;
    const FastParam& fp = plan.params[argIdx];
    if (!fp.is_store) {
        return packFastScalarSlots(fp.scalar, v, slots + fp.slot_offset);
    }
    if (auto* bh = static_cast<BufferHandle*>(g_bufferClass.unwrap(v))) {
        if (!bh->buffer || bh->buffer->residency() == BufferResidency::Gpu) return false;
        std::string err;
        if (!bh->buffer->prepareForCpu(&err)) return false;
        void* ptr = bh->buffer->hostDataForCpu();
        const uint64_t count = fp.element_bytes ? static_cast<uint64_t>(bh->buffer->bytes() / fp.element_bytes) : 0;
        slots[fp.slot_offset] = reinterpret_cast<uint64_t>(count ? ptr : s_emptyStore);
        slots[fp.slot_offset + 1] = count;
        if (outWritableBuf && fp.writable) *outWritableBuf = bh->buffer;
        if (outIsPinnedBuffer) *outIsPinnedBuffer = true;
        return true;
    }
    if (ev::isTypedArray(v)) {
        const auto info = ev::typedArrayInfo(v);
        if (!info.data && info.byteLength > 0) return false;
        const uint64_t count = fp.element_bytes ? static_cast<uint64_t>(info.byteLength / fp.element_bytes) : 0;
        slots[fp.slot_offset] = reinterpret_cast<uint64_t>(count ? info.data : s_emptyStore);
        slots[fp.slot_offset + 1] = count;
        return true;
    }
    return false;
}

Value evalFastPlanSlots(const FastEvalPlan& plan, const CompiledKernel& k, const uint64_t* slots) {
    const int32_t lx = static_cast<int32_t>(std::max(1u, k.local_size[0]));
    const int32_t ly = static_cast<int32_t>(std::max(1u, k.local_size[1]));
    const int32_t lz = static_cast<int32_t>(std::max(1u, k.local_size[2]));
    FastCpuContext ctx{
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        lx, ly, lz, 1, 1, 1,
        nullptr, &fastLoneBarrier, nullptr, nullptr,
    };
    uint64_t ret = 0;
    plan.fn_ptr(slots, &ctx, &ret);
    return plan.has_return ? decodeFastScalar(plan.ret_kind, ret) : ev::undefined();
}

bool tryFastEval(KernelHandle& h, std::span<const Value> args, Value& out) {
    if (!h.kernel || h.kernel->grid_dim != 0 || !h.kernel->cpu_built()) return false;
    initFastPlan(h);
    const FastEvalPlan* plan = h.fast_eval.get();
    if (!plan || !plan->eligible) return false;
    if (args.size() != plan->passed_count) return false;

    uint64_t slots[16]{};
    std::shared_ptr<GpuBuffer> dirtyBufs[16];
    for (size_t i = 0; i < plan->passed_count; ++i) {
        if (!packFastArgIntoSlots(*plan, i, args[i], slots, &dirtyBufs[i])) return false;
    }

    out = evalFastPlanSlots(*plan, *h.kernel, slots);
    for (size_t i = 0; i < plan->passed_count; ++i) {
        if (dirtyBufs[i]) dirtyBufs[i]->markHostWritten();
    }
    return true;
}

bool exportKernelAsBronzeNative(const CompiledKernel& k, std::string_view jsPath, std::string* err) {
    if (k.uses_grid_coords || k.uses_barriers_or_shared || !k.block_bindings.empty() ||
        !k.runtime_knobs.empty() || (k.uses_telemetry && k.telemetry_slots > 0) ||
        k.cpu_shared_bytes() > 0) {
        if (err) *err = "exportNative only supports stateless non-grid kernels";
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

    struct TrampolineParam {
        bool is_store = false;
        ScalarKind scalar = ScalarKind::Void;
        uint32_t elems_per_item = 1;
        uint32_t cpu_slot = 0;
        size_t mir_idx = 0;
    };
    std::vector<TrampolineParam> tparams;
    std::vector<brass::Type> mirParamTypes;
    uint32_t totalCpuSlots = 0;
    for (const auto& p : k.params) {
        if (!isPassed(p)) continue;
        TrampolineParam tp;
        tp.cpu_slot = totalCpuSlots;
        tp.mir_idx = mirParamTypes.size();
        if (p.role == ParamRole::Scalar && p.type.is_scalar()) {
            const char* pstr = bronzeNativeTypeOf(p.type.scalar);
            if (!pstr) {
                if (err) *err = "unsupported scalar type for parameter '" + p.name + "'";
                return false;
            }
            tp.scalar = p.type.scalar;
            sig.paramTypes.emplace_back(pstr);
            mirParamTypes.push_back((p.type.scalar == ScalarKind::F32 ||
                                     p.type.scalar == ScalarKind::F64 ||
                                     p.type.scalar == ScalarKind::DF32)
                                        ? brass::Type::f64()
                                        : brass::Type::i32());
            totalCpuSlots += (p.type.scalar == ScalarKind::DF32) ? 2 : 1;
        } else if (p.role == ParamRole::Store && p.type.dimension <= 1 && k.types) {
            const char* arrType = nullptr;
            uint32_t perItem = 1;
            if (!bronzeNativeStoreArrayOf(*k.types, p.type.element_type(), arrType, perItem)) {
                if (err) *err = "exportNative store parameter '" + p.name + "' cannot map to typed array";
                return false;
            }
            tp.is_store = true;
            tp.elems_per_item = perItem;
            sig.paramTypes.emplace_back(arrType);
            mirParamTypes.push_back(brass::Type::i64()); // ptr
            mirParamTypes.push_back(brass::Type::i32()); // element count
            totalCpuSlots += 2;
        } else {
            if (err) *err = "exportNative parameter '" + p.name + "' has unsupported role or type";
            return false;
        }
        tparams.push_back(tp);
    }

    brass::Type mirRetType = brass::Type::void_type();
    if (k.has_return_value()) {
        mirRetType = (k.return_type.scalar == ScalarKind::F32 ||
                      k.return_type.scalar == ScalarKind::F64 ||
                      k.return_type.scalar == ScalarKind::DF32)
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

    const size_t slotBytes = std::max<size_t>(1, totalCpuSlots) * 8;
    brass::Value* slots = b.build_alloca(static_cast<uint32_t>(slotBytes), 8);
    brass::Value* ctx = b.build_alloca(sizeof(FastCpuContext), 8);
    brass::Value* retSlot = b.build_alloca(16, 8);
    brass::Value* fnPtrSlot = b.build_alloca(8, 8);

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

    for (const auto& tp : tparams) {
        int32_t off = static_cast<int32_t>(tp.cpu_slot * 8);
        if (tp.is_store) {
            brass::Value* ptr64 = args[tp.mir_idx];
            brass::Value* taLen = args[tp.mir_idx + 1];
            brass::Value* count64 = b.build_zext_i64(taLen);
            if (tp.elems_per_item > 1) {
                count64 = b.build_udiv(count64, b.build_iconst_i64(tp.elems_per_item));
            }
            brass::Value* empty64 = b.build_iconst_i64(reinterpret_cast<int64_t>(s_emptyStore));
            brass::Value* nonZero = b.build_ne(count64, b.build_iconst_i64(0));
            brass::Value* safePtr = b.build_select(nonZero, ptr64, empty64);
            b.build_store(brass::Type::i64(), slots, off, safePtr);
            b.build_store(brass::Type::i64(), slots, off + 8, count64);
            continue;
        }
        brass::Value* v = args[tp.mir_idx];
        if (tp.scalar == ScalarKind::F32) {
            b.build_store(brass::Type::f32(), slots, off, b.build_fptrunc_f32_f64(v));
        } else if (tp.scalar == ScalarKind::F64) {
            b.build_store(brass::Type::f64(), slots, off, v);
        } else if (tp.scalar == ScalarKind::DF32) {
            brass::Value* hi = b.build_fptrunc_f32_f64(v);
            brass::Value* lo = b.build_fptrunc_f32_f64(b.build_sub(v, b.build_fpext_f64_f32(hi)));
            b.build_store(brass::Type::f32(), slots, off, hi);
            b.build_store(brass::Type::f32(), slots, off + 8, lo);
        } else if (tp.scalar == ScalarKind::Bool) {
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
    } else if (k.return_type.scalar == ScalarKind::DF32) {
        brass::Value* hi = b.build_load(brass::Type::f32(), retSlot, 0);
        brass::Value* lo = b.build_load(brass::Type::f32(), retSlot, 4);
        b.build_ret(b.build_add(b.build_fpext_f64_f32(hi), b.build_fpext_f64_f32(lo)));
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

    std::shared_ptr<const brass::Module> directMod;
    std::string directEntry;
    std::string mirErr;
    if (!k.checks_compiled && !k.trap_non_finite_compiled && k.probes.empty() &&
        k.ensure_direct_cpu_mir(directMod, directEntry, &mirErr)) {
        ev::NativeMirSpec spec;
        spec.mirModule = std::move(directMod);
        spec.mirFunction = std::move(directEntry);
        spec.epochPtr = &reloadEpochCounter();
        spec.expectedEpoch = currentReloadEpoch();
        spec.maxInlinedInsts = 8192;
        return ev::registerNativeMir(jsPath, raw, sig, spec, err);
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

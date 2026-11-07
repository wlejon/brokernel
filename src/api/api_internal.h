#pragma once

#include "api.h"

#include "brokernel/brokernel.h"
#include "embed/embed_host_class.h"
#include "embed/embed_object_builder.h"
#include "embed/embed_typed_array.h"

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace brokernel::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

extern ev::HostClass g_moduleClass;
extern ev::HostClass g_kernelClass;
extern ev::HostClass g_bufferClass;
extern ev::HostClass g_fnClass;

// A compiled module as its handles share it: bro.kernel.reload() replaces
// `result` and counts `generation` up, and every Kernel and KernelFn taken
// from it follows at its next use.
struct ModuleCell {
    std::shared_ptr<const CompileResult> result;
    uint64_t generation = 0;
};

struct ModuleHandle {
    std::shared_ptr<ModuleCell> cell;
};

// A new module's cell, listed for bro.kernel.reload().
std::shared_ptr<ModuleCell> makeModuleCell(std::shared_ptr<const CompileResult> result);
// This thread's modules still held by a handle.
std::vector<std::shared_ptr<ModuleCell>> liveModuleCells();

struct KernelHandle {
    std::shared_ptr<const CompileResult> module_result;
    std::shared_ptr<const CompiledKernel> kernel;
    std::shared_ptr<GpuProgram> program;
    std::shared_ptr<ModuleCell> owner;   // the module it came from
    uint64_t generation = 0;             // the owner's, as last followed
    std::unordered_map<std::string, ConstValue> knobs;   // its own (Kernel.knobs)
};

// The Kernel behind `self`, moved onto its module's current compile first.
KernelHandle* kernelHandle(Value self);

// Drops `p` on a thread of its own: a compile replaced by a reload, whose
// teardown (millions of small blocks, made on the thread that compiled it)
// takes seconds that a frame must not wait for.
void releaseLater(std::shared_ptr<const void> p);

struct BufferHandle {
    std::shared_ptr<GpuBuffer> buffer;
};

GpuDevice* currentDevice();
std::shared_ptr<GpuBuffer> allocateBuffer(size_t bytes, std::string* err);
Value makeBuffer(std::shared_ptr<GpuBuffer> buffer);

inline bool isPassed(const KernelParamInfo& p) {
    return !is_bound_param(p.role);
}

inline bool isStore(const KernelParamInfo& p) {
    return p.role == ParamRole::Store || p.role == ParamRole::Stencil;
}

inline std::optional<bronze::ElementKind> elementKindOf(ScalarKind sk) {
    switch (sk) {
        case ScalarKind::F32: return ev::elements::Float32;
        case ScalarKind::F64: return ev::elements::Float64;
        case ScalarKind::I32:
        case ScalarKind::Bool: return ev::elements::Int32;
        case ScalarKind::U32: return ev::elements::Uint32;
        case ScalarKind::I64: return ev::elements::BigInt64;
        case ScalarKind::U64: return ev::elements::BigUint64;
        default: return std::nullopt;
    }
}

inline const char* elementKindName(bronze::ElementKind k) {
    if (k == ev::elements::Float32) return "Float32Array";
    if (k == ev::elements::Float64) return "Float64Array";
    if (k == ev::elements::Int32) return "Int32Array";
    if (k == ev::elements::Uint32) return "Uint32Array";
    if (k == ev::elements::BigInt64) return "BigInt64Array";
    if (k == ev::elements::BigUint64) return "BigUint64Array";
    return "typed array";
}

inline size_t scalarBytes(ScalarKind sk) {
    return (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) ? 8 : 4;
}

inline KernelArg scalarArg(ScalarKind sk, Value v) {
    switch (sk) {
        case ScalarKind::F32: return KernelArg(static_cast<float>(ev::toDouble(v)));
        case ScalarKind::F64: return KernelArg(ev::toDouble(v));
        case ScalarKind::U32: return KernelArg(static_cast<uint32_t>(ev::toUint64(v)));
        case ScalarKind::I64: return KernelArg(ev::toInt64(v));
        case ScalarKind::U64: return KernelArg(ev::toUint64(v));
        case ScalarKind::Bool: return KernelArg(ev::toBool(v));
        default: return KernelArg(static_cast<int32_t>(ev::toInt64(v)));
    }
}

class ArgPack {
public:
    ArgPack(const CompiledKernel& k, std::span<const Value> js, const std::string& what, bool gpu = false) {
        size_t passed = 0;
        for (const auto& p : k.params) passed += isPassed(p) ? 1 : 0;
        if (js.size() != passed) {
            ev::throwTypeError(what + ": kernel '" + k.name + "' takes " + std::to_string(passed) +
                               " argument(s), got " + std::to_string(js.size()));
        }
        args_.resize(passed);
        bytes_.resize(passed);
        size_t i = 0;
        for (const auto& p : k.params) {
            if (!isPassed(p)) continue;
            const Value v = js[i];
            if (isStore(p) && gpu) {
                auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(v));
                if (!h) ev::throwTypeError(what + ": '" + p.name + "' is a store and takes a KernelBuffer");
                if (!h->buffer) ev::throwError(what + ": '" + p.name + "' is a destroyed KernelBuffer");
                std::shared_ptr<GpuBuffer> b = h->buffer;
                args_[i] = KernelArg::device(b->address(), b->bytes(), b->capacity());
                buffers_.push_back(std::move(b));
            } else if (isStore(p)) {
                if (!ev::isTypedArray(v)) ev::throwTypeError(what + ": '" + p.name + "' is a store and takes a typed array");
                const Type elem = p.type.element_type();
                const auto want = elem.is_scalar() ? elementKindOf(elem.scalar) : std::nullopt;
                if (want && ev::typedArrayInfo(v).elementKind != *want) {
                    ev::throwTypeError(what + ": store '" + p.name + "' takes a " + elementKindName(*want));
                }
                stores_.push_back(i);
            } else if (p.role == ParamRole::Scalar) {
                if (!ev::isNumber(v) && !ev::isBigInt(v) && !ev::isBool(v)) {
                    ev::throwTypeError(what + ": '" + p.name + "' takes a number");
                }
                args_[i] = scalarArg(p.type.scalar, v);
            } else if (ev::isTypedArray(v)) {
                const auto info = ev::typedArrayInfo(v);
                bytes_[i].assign(info.data, info.data + info.byteLength);
                args_[i] = KernelArg::buffer(static_cast<const void*>(bytes_[i].data()), bytes_[i].size());
            } else if (ev::isObject(v) && p.role != ParamRole::Struct) {
                const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(v, "length")));
                if (n != p.mir_param_count) {
                    ev::throwTypeError(what + ": '" + p.name + "' takes " + std::to_string(p.mir_param_count) +
                                       " number(s)");
                }
                const ScalarKind sk = p.type.scalar;
                const size_t eb = scalarBytes(sk);
                bytes_[i].assign(n * eb, 0);
                for (uint32_t e = 0; e < n; ++e) {
                    const double d = ev::toDouble(ev::getElement(v, e));
                    uint8_t* at = bytes_[i].data() + e * eb;
                    if (sk == ScalarKind::F64) {
                        std::memcpy(at, &d, 8);
                    } else if (sk == ScalarKind::F32) {
                        const float f = static_cast<float>(d);
                        std::memcpy(at, &f, 4);
                    } else if (eb == 8) {
                        const int64_t x = static_cast<int64_t>(d);
                        std::memcpy(at, &x, 8);
                    } else {
                        const int32_t x = static_cast<int32_t>(static_cast<int64_t>(d));
                        std::memcpy(at, &x, 4);
                    }
                }
                args_[i] = KernelArg::buffer(static_cast<const void*>(bytes_[i].data()), bytes_[i].size());
            } else {
                ev::throwTypeError(what + ": '" + p.name + "' takes " +
                                   (p.role == ParamRole::Struct ? "a typed array of its bytes" : "an array of numbers"));
            }
            ++i;
        }
    }

    std::span<const KernelArg> bind(std::span<const Value> js) {
        for (size_t i : stores_) {
            const auto info = ev::typedArrayInfo(js[i]);
            args_[i] = KernelArg::buffer(static_cast<void*>(info.data), info.byteLength);
        }
        return args_;
    }

    std::span<const KernelArg> gpuArgs() const { return args_; }

private:
    std::vector<KernelArg> args_;
    std::vector<std::shared_ptr<GpuBuffer>> buffers_;
    std::vector<size_t> stores_;
    std::vector<std::vector<uint8_t>> bytes_;
};

struct HeldArgs {
    explicit HeldArgs(std::span<const Value> js) {
        held.reserve(js.size());
        for (Value v : js) held.emplace_back(v);
    }
    std::vector<Value> current() const {
        std::vector<Value> out;
        out.reserve(held.size());
        for (const auto& p : held) out.push_back(p.get());
        return out;
    }
    std::vector<ev::Persistent> held;
};

Value decodeLeaves(const CompiledKernel& k, const uint8_t* raw);
GridShape shapeArg(Value v, const CompiledKernel& k, const std::string& what);
// The Error a failed compile throws (its message and `.diagnostics`).
Value makeCompileError(const CompileResult& r);
[[noreturn]] void throwCompileError(const CompileResult& r);
// bro.kernel.compile(source, opts).
Value js_compile(Value self, std::span<const Value> args);
// A KernelModule object over `result`, in the current realm.
Value makeModuleValue(std::shared_ptr<const CompileResult> result);

// --- Compiling off the page thread (api_async.cpp) ---------------------------
//
// runCompileWork runs `work` on a worker thread while this thread waits in
// steps of its wait pump's interval, calling the pump between steps and
// answering the worker's module requests (the JS resolver runs here, never on
// the worker); inline when this thread has no pump, or is itself a worker.
// `work` touches no JS value or realm: it compiles. A JS resolver that throws
// is not asked again (the worker's requests answer nothing), and its exception
// is rethrown here once the worker is done.
void runCompileWork(const std::function<void()>& work);

// `compile()` (a CompileResult) through runCompileWork.
template <typename F>
std::shared_ptr<CompileResult> compileOff(F&& compile) {
    std::shared_ptr<CompileResult> out;
    runCompileWork([&] { out = std::make_shared<CompileResult>(compile()); });
    return out;
}

// On a compile worker: the module source as this thread's owner would resolve
// it (the global tables here, the JS resolver there).
bool onCompileWorker();
std::optional<std::string> workerResolve(std::string_view name);
// On a compile worker, the CPU builds of `r`'s @cpu kernels, made now rather
// than at their first run on the page thread (a cache hit lowers them then).
void prebuildCpuOnlyKernels(const CompileResult& r);

// The module tables (api_modules.cpp). resolveFromTables reads the registered
// and resolved modules (and the host's with `withHostModules`) with no JS;
// resolveWithJs is resolveRegisteredModule on the owning thread, a resolver's
// throw left in `thrown` (`threw`) rather than raised.
bool threadHasJsResolver();
std::optional<std::string> resolveFromTables(std::string_view name, bool withHostModules);
std::optional<std::string> resolveWithJs(std::string_view name, ev::Persistent& thrown, bool& threw);
void registerModulesInto(Compiler& compiler);
void decorateModuleRegistry(ev::ObjectBuilder& b);   // resolve, registerModule, clearModules
void decorateAsync(ev::ObjectBuilder& b);            // compileAsync
void installAsyncRealmHook();

void decorateFnClass(ev::ObjectBuilder& b);
void decorateModuleExtras(ev::ObjectBuilder& b);
void decorateKernelExtras(ev::ObjectBuilder& b);
void decorateBroKernelExtras(ev::ObjectBuilder& b);
void decorateReload(ev::ObjectBuilder& b);   // bro.kernel.reload (api_reload.cpp)
// Registered and resolved modules of these names take the new sources.
void updateRegisteredSources(const std::unordered_map<std::string, std::string>& sources);
void configureCompilerModulesAndOptions(
    std::span<const Value> args,
    Compiler& compiler,
    std::string& file,
    CompileOptions& compile_opts);

} // namespace brokernel::api

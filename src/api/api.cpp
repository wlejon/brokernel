#include "api.h"

#include "brokernel/brokernel.h"
#include "embed/embed_host_class.h"
#include "embed/embed_object_builder.h"
#include "embed/embed_typed_array.h"

#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace brokernel::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

namespace {

ev::HostClass g_moduleClass;
ev::HostClass g_kernelClass;

struct ModuleHandle {
    std::shared_ptr<const CompileResult> result;
};

struct KernelHandle {
    std::shared_ptr<const CompiledKernel> kernel;  // aliases its CompileResult
};

const CompiledKernel* unwrapKernel(Value self) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(self));
    return h ? h->kernel.get() : nullptr;
}

const char* roleName(ParamRole r) {
    switch (r) {
        case ParamRole::Scalar: return "scalar";
        case ParamRole::Vector: return "vector";
        case ParamRole::Struct: return "struct";
        case ParamRole::Array: return "array";
        case ParamRole::Grid: return "grid";
        case ParamRole::Store: return "store";
        case ParamRole::Stencil: return "stencil";
        case ParamRole::Image: return "image";
        case ParamRole::Sampler: return "sampler";
        default: return "other";
    }
}

const char* accessName(AccessMode m) {
    switch (m) {
        case AccessMode::Read: return "read";
        case AccessMode::Write: return "write";
        default: return "read_write";
    }
}

// The parameters a caller passes: all but the grid and the GPU's image and
// sampler bindings.
bool isPassed(const KernelParamInfo& p) {
    return p.role != ParamRole::Grid && p.role != ParamRole::Image && p.role != ParamRole::Sampler;
}

bool isStore(const KernelParamInfo& p) {
    return p.role == ParamRole::Store || p.role == ParamRole::Stencil;
}

// The typed-array element kind of a scalar: what a store of them takes.
std::optional<bronze::ElementKind> elementKindOf(ScalarKind sk) {
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

const char* elementKindName(bronze::ElementKind k) {
    if (k == ev::elements::Float32) return "Float32Array";
    if (k == ev::elements::Float64) return "Float64Array";
    if (k == ev::elements::Int32) return "Int32Array";
    if (k == ev::elements::Uint32) return "Uint32Array";
    if (k == ev::elements::BigInt64) return "BigInt64Array";
    if (k == ev::elements::BigUint64) return "BigUint64Array";
    return "typed array";
}

size_t scalarBytes(ScalarKind sk) {
    return (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) ? 8 : 4;
}

KernelArg scalarArg(ScalarKind sk, Value v) {
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

// One call's arguments as KernelArgs. In two passes because of the
// typed-array pointer contract (embed_typed_array.h): the constructor reads
// every value (numbers, arrays of them, a vector's bytes), then bind() takes
// the stores' pointers last, with nothing allocating between them and the
// dispatch. A bad argument throws, naming the call.
class ArgPack {
public:
    ArgPack(const CompiledKernel& k, std::span<const Value> js, const std::string& what) {
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
            if (isStore(p)) {
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
                // A vector or struct as its bytes, copied now.
                const auto info = ev::typedArrayInfo(v);
                bytes_[i].assign(info.data, info.data + info.byteLength);
                args_[i] = KernelArg::buffer(static_cast<const void*>(bytes_[i].data()), bytes_[i].size());
            } else if (ev::isObject(v) && p.role != ParamRole::Struct) {
                // A vector or array as numbers, laid out as its scalar's bytes.
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

    // The stores' pointers, read from `js` (which the caller keeps current
    // across anything that allocated), immediately before the dispatch.
    std::span<const KernelArg> bind(std::span<const Value> js) {
        for (size_t i : stores_) {
            const auto info = ev::typedArrayInfo(js[i]);
            args_[i] = KernelArg::buffer(static_cast<void*>(info.data), info.byteLength);
        }
        return args_;
    }

private:
    std::vector<KernelArg> args_;
    std::vector<size_t> stores_;
    std::vector<std::vector<uint8_t>> bytes_;
};

// Values that must stay current across allocations: the call's arguments.
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

Value decodeLeaves(const CompiledKernel& k, const uint8_t* raw) {
    auto leaf = [&](size_t i, size_t& off) -> Value {
        const ScalarKind sk = k.return_leaf_kinds[i];
        double d = 0.0;
        switch (sk) {
            case ScalarKind::F64: { double x; std::memcpy(&x, raw + off, 8); d = x; off += 8; break; }
            case ScalarKind::F32: { float x; std::memcpy(&x, raw + off, 4); d = x; off += 4; break; }
            case ScalarKind::I64: { int64_t x; std::memcpy(&x, raw + off, 8); d = static_cast<double>(x); off += 8; break; }
            case ScalarKind::U64: { uint64_t x; std::memcpy(&x, raw + off, 8); d = static_cast<double>(x); off += 8; break; }
            case ScalarKind::U32: { uint32_t x; std::memcpy(&x, raw + off, 4); d = x; off += 4; break; }
            case ScalarKind::Bool: { int32_t x; std::memcpy(&x, raw + off, 4); off += 4; return ev::fromBool(x != 0); }
            default: { int32_t x; std::memcpy(&x, raw + off, 4); d = x; off += 4; break; }
        }
        return ev::fromDouble(d);
    };
    size_t off = 0;
    if (k.return_leaf_kinds.size() == 1) return leaf(0, off);
    ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(k.return_leaf_kinds.size())));
    for (size_t i = 0; i < k.return_leaf_kinds.size(); ++i) {
        ev::Persistent v(leaf(i, off));
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), v.get()));
    }
    return arr.get();
}

// kernel.eval(...args): a point kernel at one point, on the CPU.
Value js_eval(Value self, std::span<const Value> args) {
    const CompiledKernel* k = unwrapKernel(self);
    if (!k) return ev::throwTypeError("Kernel.eval: invalid receiver");
    if (k->grid_dim != 0) return ev::throwTypeError("Kernel.eval: '" + k->name + "' is a grid kernel; use run()");
    if (!k->cpu_point_fn.is_valid()) return ev::throwTypeError("Kernel.eval: '" + k->name + "' has no CPU build");
    ArgPack pack(*k, args, "Kernel.eval");
    alignas(16) uint8_t ret[256] = {};
    std::string err;
    const bool ok = k->eval_cpu(pack.bind(args), k->return_byte_size <= sizeof ret ? ret : nullptr, &err);
    if (!ok) return ev::throwError("Kernel.eval: " + err);
    if (!k->has_return_value() || k->return_byte_size > sizeof ret) return ev::undefined();
    return decodeLeaves(*k, ret);
}

// The grid shape `v` names: a width, or [width, height, depth].
GridShape shapeArg(Value v, const CompiledKernel& k) {
    auto extent = [](Value e) -> uint32_t {
        const double d = ev::toDouble(e);
        if (!(d >= 1.0) || d > 4294967295.0) ev::throwRangeError("Kernel.run: a grid extent is a positive integer");
        return static_cast<uint32_t>(d);
    };
    GridShape s;
    if (ev::isNumber(v)) {
        s.width = extent(v);
    } else if (ev::isObject(v)) {
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(v, "length")));
        if (n < 1 || n > 3) ev::throwTypeError("Kernel.run: the grid is [width, height, depth] (1 to 3 extents)");
        s.width = extent(ev::getElement(v, 0));
        if (n >= 2) s.height = extent(ev::getElement(v, 1));
        if (n >= 3) s.depth = extent(ev::getElement(v, 2));
    } else {
        ev::throwTypeError("Kernel.run: the grid is a width or [width, height, depth]");
    }
    if ((s.height > 1 && k.grid_dim < 2) || (s.depth > 1 && k.grid_dim < 3)) {
        ev::throwTypeError("Kernel.run: '" + k.name + "' is a " + std::to_string(k.grid_dim) + "D grid kernel");
    }
    return s;
}

// kernel.run(grid, ...args): a grid kernel over its grid, on the CPU. Its
// stores are written in place; a kernel that returns a value returns one per
// cell, in a typed array.
Value js_run(Value self, std::span<const Value> args) {
    const CompiledKernel* k = unwrapKernel(self);
    if (!k) return ev::throwTypeError("Kernel.run: invalid receiver");
    if (k->grid_dim == 0) return ev::throwTypeError("Kernel.run: '" + k->name + "' is a point kernel; use eval()");
    if (!k->cpu_grid_fn.is_valid()) return ev::throwTypeError("Kernel.run: '" + k->name + "' has no CPU build");
    if (args.empty()) return ev::throwTypeError("Kernel.run: the first argument is the grid");
    const GridShape shape = shapeArg(args[0], *k);
    const std::span<const Value> kargs = args.subspan(1);
    ArgPack pack(*k, kargs, "Kernel.run");

    // The returns' typed array is allocated before the stores are bound.
    ev::Persistent out;
    if (k->has_return_value()) {
        const auto& kinds = k->return_leaf_kinds;
        bool uniform = !kinds.empty();
        for (ScalarKind sk : kinds) uniform = uniform && sk == kinds[0];
        const auto kind = uniform ? elementKindOf(kinds[0]) : std::nullopt;
        const size_t cells = shape.total_points();
        const size_t count = kind ? cells * kinds.size() : cells * k->return_byte_size;
        if (count > 0xffffffffull) return ev::throwRangeError("Kernel.run: the grid's returns do not fit a typed array");
        HeldArgs held(kargs);
        out.set(ev::createTypedArray(kind ? *kind : ev::elements::Uint8, static_cast<uint32_t>(count)));
        const std::vector<Value> current = held.current();
        std::span<const KernelArg> bound = pack.bind(current);
        std::string err;
        const auto info = ev::typedArrayInfo(out.get());
        if (!k->dispatch_cpu(shape, bound, info.data, &err)) return ev::throwError("Kernel.run: " + err);
        return out.get();
    }
    std::string err;
    if (!k->dispatch_cpu(shape, pack.bind(kargs), nullptr, &err)) return ev::throwError("Kernel.run: " + err);
    return ev::undefined();
}

Value paramInfo(const KernelParamInfo& p) {
    ev::ObjectBuilder b;
    b.set("name", p.name);
    b.set("kind", roleName(p.role));
    b.set("type", p.type.to_string());
    if (isStore(p)) {
        b.set("access", accessName(p.access));
        b.set("dimension", p.dimension);
        // The grid axes the kernel accesses it at, so it must cover them (0: none).
        b.set("cells", p.grid_cover_dim);
    }
    return b.get();
}

void decorateKernel(ev::ObjectBuilder& b) {
    b.def("eval", 0, js_eval);
    b.def("run", 1, js_run);
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        const CompiledKernel* k = unwrapKernel(self);
        return k ? ev::fromUtf8(k->name) : ev::undefined();
    });
    b.accessor("gridDim", [](Value self, std::span<const Value>) -> Value {
        const CompiledKernel* k = unwrapKernel(self);
        return k ? ev::fromDouble(k->grid_dim) : ev::undefined();
    });
    b.accessor("returns", [](Value self, std::span<const Value>) -> Value {
        const CompiledKernel* k = unwrapKernel(self);
        if (!k) return ev::undefined();
        return k->has_return_value() ? ev::fromUtf8(k->return_type.to_string()) : ev::null();
    });
    b.accessor("cpuLanes", [](Value self, std::span<const Value>) -> Value {
        const CompiledKernel* k = unwrapKernel(self);
        return k ? ev::fromDouble(k->cpu_grid_lanes) : ev::undefined();
    });
    b.accessor("params", [](Value self, std::span<const Value>) -> Value {
        const CompiledKernel* k = unwrapKernel(self);
        if (!k) return ev::undefined();
        std::vector<const KernelParamInfo*> passed;
        for (const auto& p : k->params) {
            if (isPassed(p)) passed.push_back(&p);
        }
        ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(passed.size())));
        for (size_t i = 0; i < passed.size(); ++i) {
            ev::Persistent v(paramInfo(*passed[i]));
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), v.get()));
        }
        return arr.get();
    });
}

const CompileResult* unwrapModule(Value self) {
    auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
    return h ? h->result.get() : nullptr;
}

void decorateModule(ev::ObjectBuilder& b) {
    b.def("kernel", 1, [](Value self, std::span<const Value> args) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h) return ev::throwTypeError("KernelModule.kernel: invalid receiver");
        const std::string name = !args.empty() && ev::isString(args[0]) ? ev::toUtf8(args[0]) : "";
        const CompiledKernel* k = h->result->find_kernel(name);
        if (!k) return ev::throwError("KernelModule.kernel: no kernel '" + name + "'");
        auto handle = std::make_unique<KernelHandle>();
        handle->kernel = std::shared_ptr<const CompiledKernel>(h->result, k);
        return g_kernelClass.createInstance(std::move(handle));
    });
    b.accessor("kernels", [](Value self, std::span<const Value>) -> Value {
        const CompileResult* r = unwrapModule(self);
        if (!r) return ev::undefined();
        ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(r->kernels.size())));
        for (size_t i = 0; i < r->kernels.size(); ++i) {
            ev::Persistent s(ev::fromUtf8(r->kernels[i].name));
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), s.get()));
        }
        return arr.get();
    });
}

// A compile error: an Error whose message is the formatted diagnostics and
// whose `diagnostics` lists them ({severity, file, line, column, message}).
[[noreturn]] void throwCompileError(const CompileResult& r) {
    ev::Persistent list(ev::makeArray(static_cast<uint32_t>(r.diagnostics.items().size())));
    uint32_t i = 0;
    for (const Diagnostic& d : r.diagnostics.items()) {
        ev::ObjectBuilder b;
        b.set("severity", d.severity == Severity::Error ? "error" : d.severity == Severity::Warning ? "warning" : "note");
        b.set("file", d.span.file);
        b.set("line", d.span.start.line);
        b.set("column", d.span.start.column);
        b.set("message", d.message);
        ev::Persistent v(b.get());
        list.set(ev::setElement(list.get(), i++, v.get()));
    }
    ev::Persistent err;
    auto ctor = ev::globalValue("Error");
    if (ctor.found && ev::isFunction(ctor.value)) {
        ev::Persistent c(ctor.value);
        ev::Persistent msg(ev::fromUtf8("bro.kernel.compile: " + r.error_message()));
        const Value arg = msg.get();
        auto made = ev::construct(c.get(), std::span<const Value>(&arg, 1));
        if (!made.thrown) err.set(made.value);
    }
    if (!ev::isObject(err.get())) ev::throwError("bro.kernel.compile: " + r.error_message());
    err.set(ev::setProperty(err.get(), "diagnostics", list.get()));
    ev::throwValue(err.get());
}

// bro.kernel.compile(source, { file, modules }): every kernel of `source`,
// compiled for the CPU and the GPU. `modules` maps a module name
// ("world::fields") to its source, for `import`; `file` names the source in
// diagnostics.
Value js_compile(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) return ev::throwTypeError("bro.kernel.compile: the source is a string");
    const std::string source = ev::toUtf8(args[0]);
    std::string file = "kernel.bk";
    Compiler compiler;
    if (args.size() > 1 && ev::isObject(args[1])) {
        ev::Persistent opts(args[1]);
        const Value f = ev::getProperty(opts.get(), "file");
        // A name only: an import is never looked for beside it on disk.
        if (ev::isString(f)) file = std::filesystem::path(ev::toUtf8(f)).filename().string();
        ev::Persistent mods(ev::getProperty(opts.get(), "modules"));
        if (ev::isObject(mods.get())) {
            auto object = ev::globalValue("Object");
            if (!object.found) return ev::throwError("bro.kernel.compile: no Object global");
            ev::Persistent keysFn(ev::getProperty(object.value, "keys"));
            const Value marg = mods.get();
            auto keys = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(&marg, 1));
            if (keys.thrown) return ev::throwValue(keys.value);
            ev::Persistent names(keys.value);
            const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(names.get(), "length")));
            for (uint32_t i = 0; i < n; ++i) {
                const std::string name = ev::toUtf8(ev::getElement(names.get(), i));
                const Value src = ev::getProperty(mods.get(), name);
                if (!ev::isString(src)) return ev::throwTypeError("bro.kernel.compile: module '" + name + "' is not a string");
                compiler.register_module(name, ev::toUtf8(src));
            }
        }
    }
    auto result = std::make_shared<CompileResult>(compiler.compile_source(source, file));
    if (!result->ok) throwCompileError(*result);
    auto handle = std::make_unique<ModuleHandle>();
    handle->result = std::move(result);
    return g_moduleClass.createInstance(std::move(handle));
}

Value ensureBroKernel() {
    ev::Persistent broP;
    auto bro = ev::globalValue("bro");
    if (bro.found && ev::isObject(bro.value)) broP.set(bro.value);
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
    }
    ev::Persistent kernelP(ev::getProperty(broP.get(), "kernel"));
    if (!ev::isObject(kernelP.get())) {
        kernelP.set(ev::createObject());
        broP.set(ev::setProperty(broP.get(), "kernel", kernelP.get()));
    }
    return kernelP.get();
}

} // namespace

void installKernel() {
    g_moduleClass.install("KernelModule", 0, nullptr, decorateModule, /*global=*/false);
    g_kernelClass.install("Kernel", 0, nullptr, decorateKernel, /*global=*/false);
    ev::Persistent kernelObj(ensureBroKernel());
    ev::ObjectBuilder b(kernelObj.get());
    b.set("available", true);
    b.set("version", version());
    b.def("compile", 2, js_compile);
}

std::shared_ptr<const CompiledKernel> kernelOf(Value v) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(v));
    return h ? h->kernel : nullptr;
}

} // namespace brokernel::api

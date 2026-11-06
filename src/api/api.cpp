#include "api_internal.h"

#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace brokernel::api {

ev::HostClass g_moduleClass;
ev::HostClass g_kernelClass;
ev::HostClass g_bufferClass;
ev::HostClass g_fnClass;

namespace {

// The device this thread's realms dispatch on (setGpuDevice).
thread_local std::shared_ptr<GpuDevice> t_device;
// GPU buffer bytes allocated since the last collection, less bytes explicitly
// released by KernelBuffer.destroy().
thread_local size_t t_uncollectedBufferBytes = 0;
constexpr size_t kGcPressureBytes = 64ull * 1024 * 1024;

void collectAndReclaim() {
    t_uncollectedBufferBytes = 0;
    ev::collectGarbage();
    if (t_device) t_device->reclaim();
}

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
        case ParamRole::StageInput: return "stage_input";
        case ParamRole::Uniform: return "uniform";
        case ParamRole::Buffer: return "buffer";
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

// kernel.eval(...args): a point kernel at one point, on the CPU.
Value js_eval(Value self, std::span<const Value> args) {
    const CompiledKernel* k = unwrapKernel(self);
    if (!k) return ev::throwTypeError("Kernel.eval: invalid receiver");
    if (k->grid_dim != 0) return ev::throwTypeError("Kernel.eval: '" + k->name + "' is a grid kernel; use run()");
    if (!k->cpu_point_fn().is_valid()) return ev::throwTypeError("Kernel.eval: '" + k->name + "' has no CPU build" + (k->cpu_error().empty() ? std::string() : ": " + k->cpu_error()));
    ArgPack pack(*k, args, "Kernel.eval");
    alignas(16) uint8_t ret[256] = {};
    std::string err;
    const bool ok = k->eval_cpu(pack.bind(args), k->return_byte_size <= sizeof ret ? ret : nullptr, &err);
    if (!ok) return ev::throwError("Kernel.eval: " + err);
    if (!k->has_return_value() || k->return_byte_size > sizeof ret) return ev::undefined();
    return decodeLeaves(*k, ret);
}

// kernel.run(grid, ...args): a grid kernel over its grid, on the CPU. Its
// stores are written in place; a kernel that returns a value returns one per
// cell, in a typed array.
Value js_run(Value self, std::span<const Value> args) {
    const CompiledKernel* k = unwrapKernel(self);
    if (!k) return ev::throwTypeError("Kernel.run: invalid receiver");
    if (k->grid_dim == 0) return ev::throwTypeError("Kernel.run: '" + k->name + "' is a point kernel; use eval()");
    if (!k->cpu_grid_fn().is_valid()) return ev::throwTypeError("Kernel.run: '" + k->name + "' has no CPU build" + (k->cpu_error().empty() ? std::string() : ": " + k->cpu_error()));
    if (args.empty()) return ev::throwTypeError("Kernel.run: the first argument is the grid");
    const GridShape shape = shapeArg(args[0], *k, "Kernel.run");
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

// kernel.dispatch(grid, ...args): a grid kernel over its grid, on the GPU,
// after every dispatch so far. Its stores are KernelBuffers; a kernel that
// returns a value returns a new KernelBuffer of one value per cell.
Value js_dispatch(Value self, std::span<const Value> args) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(self));
    if (!h) return ev::throwTypeError("Kernel.dispatch: invalid receiver");
    const CompiledKernel& k = *h->kernel;
    if (!t_device || !t_device->available()) {
        return ev::throwError("Kernel.dispatch: this realm has no GPU device (bro.kernel.gpu is false)");
    }
    if (k.grid_dim == 0) return ev::throwTypeError("Kernel.dispatch: '" + k.name + "' is a point kernel; use eval()");
    if (args.empty()) return ev::throwTypeError("Kernel.dispatch: the first argument is the grid");
    const GridShape shape = shapeArg(args[0], k, "Kernel.dispatch");
    ArgPack pack(k, args.subspan(1), "Kernel.dispatch", /*gpu=*/true);

    std::string err;
    if (!h->program) {
        h->program = t_device->createProgram(k, &err);
        if (!h->program) return ev::throwError("Kernel.dispatch: " + err);
    }
    std::shared_ptr<GpuBuffer> returns;
    if (const size_t bytes = k.gpu_return_bytes(shape)) {
        returns = allocateBuffer(bytes, &err);
        if (!returns) return ev::throwError("Kernel.dispatch: " + err);
    }
    GpuLaunch launch;
    if (!k.gpu_launch(shape, pack.gpuArgs(), returns ? returns->address() : 0, launch, &err) ||
        !t_device->dispatch(*h->program, k, launch, &err)) {
        return ev::throwError("Kernel.dispatch: " + err);
    }
    return returns ? makeBuffer(std::move(returns)) : ev::undefined();
}

// A byte offset argument (default 0).
size_t offsetArg(std::span<const Value> args, size_t i, const std::string& what) {
    if (args.size() <= i || ev::isUndefined(args[i])) return 0;
    const double d = ev::toDouble(args[i]);
    if (!(d >= 0.0) || d != static_cast<double>(static_cast<uint64_t>(d))) {
        ev::throwRangeError(what + ": the byte offset is a non-negative integer");
    }
    return static_cast<size_t>(d);
}

GpuBuffer* unwrapBuffer(Value self, const std::string& what) {
    auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
    if (!h) ev::throwTypeError(what + ": invalid receiver");
    if (!h->buffer) ev::throwError(what + ": the KernelBuffer has been destroyed");
    return h->buffer.get();
}

// buffer.write(source, byteOffset = 0): `source`'s bytes (a typed array),
// copied now, written after every dispatch so far.
Value js_bufferWrite(Value self, std::span<const Value> args) {
    GpuBuffer* b = unwrapBuffer(self, "KernelBuffer.write");
    if (args.empty() || !ev::isTypedArray(args[0])) return ev::throwTypeError("KernelBuffer.write: the source is a typed array");
    const size_t offset = offsetArg(args, 1, "KernelBuffer.write");
    const auto info = ev::typedArrayInfo(args[0]);
    if (offset > b->bytes() || info.byteLength > b->bytes() - offset) {
        return ev::throwRangeError("KernelBuffer.write: " + std::to_string(info.byteLength) + " byte(s) at " +
                                   std::to_string(offset) + " do not fit its " + std::to_string(b->bytes()));
    }
    std::string err;
    if (!b->write(offset, info.data, info.byteLength, &err)) return ev::throwError("KernelBuffer.write: " + err);
    return ev::undefined();
}

// buffer.read(target, byteOffset = 0): fills `target` (a typed array) with
// the bytes at byteOffset, after every dispatch so far; returns it.
Value js_bufferRead(Value self, std::span<const Value> args) {
    GpuBuffer* b = unwrapBuffer(self, "KernelBuffer.read");
    if (args.empty() || !ev::isTypedArray(args[0])) return ev::throwTypeError("KernelBuffer.read: the target is a typed array");
    const size_t offset = offsetArg(args, 1, "KernelBuffer.read");
    const auto info = ev::typedArrayInfo(args[0]);
    if (offset > b->bytes() || info.byteLength > b->bytes() - offset) {
        return ev::throwRangeError("KernelBuffer.read: " + std::to_string(info.byteLength) + " byte(s) at " +
                                   std::to_string(offset) + " are outside its " + std::to_string(b->bytes()));
    }
    std::string err;
    if (!b->read(offset, info.data, info.byteLength, &err)) return ev::throwError("KernelBuffer.read: " + err);
    return args[0];
}

// buffer.destroy(): releases the buffer's GPU memory once in-flight work
// finishes; any later use of the buffer throws.
Value js_bufferDestroy(Value self, std::span<const Value>) {
    auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
    if (!h) return ev::throwTypeError("KernelBuffer.destroy: invalid receiver");
    if (h->buffer) {
        const size_t bytes = h->buffer->bytes();
        t_uncollectedBufferBytes = t_uncollectedBufferBytes > bytes ? t_uncollectedBufferBytes - bytes : 0;
        h->buffer->destroy();
        h->buffer.reset();
    }
    return ev::undefined();
}

void decorateBuffer(ev::ObjectBuilder& b) {
    b.def("write", 1, js_bufferWrite);
    b.def("read", 1, js_bufferRead);
    b.def("destroy", 0, js_bufferDestroy);
    b.accessor("byteLength", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
        if (!h) return ev::undefined();
        if (!h->buffer) return ev::throwError("KernelBuffer.byteLength: the KernelBuffer has been destroyed");
        return ev::fromDouble(static_cast<double>(h->buffer->bytes()));
    });
}

// bro.kernel.createBuffer(byteLength | typedArray): GPU memory for a
// dispatch's stores, reading zero, or holding the typed array's bytes.
Value js_createBuffer(Value, std::span<const Value> args) {
    if (!t_device || !t_device->available()) {
        return ev::throwError("bro.kernel.createBuffer: this realm has no GPU device (bro.kernel.gpu is false)");
    }
    const bool fromArray = !args.empty() && ev::isTypedArray(args[0]);
    size_t bytes = 0;
    if (fromArray) {
        bytes = ev::typedArrayInfo(args[0]).byteLength;
    } else if (!args.empty() && ev::isNumber(args[0])) {
        const double d = ev::toDouble(args[0]);
        if (!(d >= 0.0) || d != static_cast<double>(static_cast<uint64_t>(d))) {
            return ev::throwRangeError("bro.kernel.createBuffer: the byte length is a non-negative integer");
        }
        bytes = static_cast<size_t>(d);
    } else {
        return ev::throwTypeError("bro.kernel.createBuffer: takes a byte length or a typed array");
    }
    std::string err;
    std::shared_ptr<GpuBuffer> buffer = allocateBuffer(bytes, &err);
    if (!buffer) return ev::throwError("bro.kernel.createBuffer: " + err);
    if (fromArray && bytes > 0) {
        const auto info = ev::typedArrayInfo(args[0]);
        if (!buffer->write(0, info.data, info.byteLength, &err)) return ev::throwError("bro.kernel.createBuffer: " + err);
    }
    return makeBuffer(std::move(buffer));
}

Value paramInfo(const KernelParamInfo& p) {
    ev::ObjectBuilder b;
    b.set("name", p.name);
    b.set("kind", roleName(p.role));
    b.set("type", p.type.to_string());
    if (isStore(p)) {
        b.set("access", accessName(p.access));
        b.set("dimension", p.dimension);
        b.set("cells", p.grid_cover_dim);
    }
    return b.get();
}

void decorateKernel(ev::ObjectBuilder& b) {
    b.def("eval", 0, js_eval);
    b.def("run", 1, js_run);
    b.def("dispatch", 1, js_dispatch);
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
    decorateKernelExtras(b);
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
        handle->module_result = h->result;
        handle->kernel = std::shared_ptr<const CompiledKernel>(h->result, k);
        if (k->grid_dim > 0 && !k->spirv.words.empty() && t_device && t_device->available()) {
            std::string err;
            handle->program = t_device->createProgram(*k, &err);
        }
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
    decorateModuleExtras(b);
}

Value js_compile(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) return ev::throwTypeError("bro.kernel.compile: the source is a string");
    const std::string source = ev::toUtf8(args[0]);
    std::string file = "kernel.bk";
    Compiler compiler;
    CompileOptions compile_opts;
    configureCompilerModulesAndOptions(args, compiler, file, compile_opts);
    auto result = std::make_shared<CompileResult>(compiler.compile_source(source, file, compile_opts));
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

GpuDevice* currentDevice() {
    return (t_device && t_device->available()) ? t_device.get() : nullptr;
}

std::shared_ptr<GpuBuffer> allocateBuffer(size_t bytes, std::string* err) {
    if (t_uncollectedBufferBytes + bytes >= kGcPressureBytes) collectAndReclaim();
    std::shared_ptr<GpuBuffer> b = t_device->createBuffer(bytes, err);
    if (!b) {
        collectAndReclaim();
        if (err) err->clear();
        b = t_device->createBuffer(bytes, err);
    }
    if (b) t_uncollectedBufferBytes += bytes;
    return b;
}

Value makeBuffer(std::shared_ptr<GpuBuffer> buffer) {
    auto handle = std::make_unique<BufferHandle>();
    handle->buffer = std::move(buffer);
    return g_bufferClass.createInstance(std::move(handle));
}

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

GridShape shapeArg(Value v, const CompiledKernel& k, const std::string& what) {
    auto extent = [&](Value e) -> uint32_t {
        const double d = ev::toDouble(e);
        if (!(d >= 1.0) || d > 4294967295.0) ev::throwRangeError(what + ": a grid extent is a positive integer");
        return static_cast<uint32_t>(d);
    };
    GridShape s;
    if (ev::isNumber(v)) {
        s.width = extent(v);
    } else if (ev::isObject(v)) {
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(v, "length")));
        if (n < 1 || n > 3) ev::throwTypeError(what + ": the grid is [width, height, depth] (1 to 3 extents)");
        s.width = extent(ev::getElement(v, 0));
        if (n >= 2) s.height = extent(ev::getElement(v, 1));
        if (n >= 3) s.depth = extent(ev::getElement(v, 2));
    } else {
        ev::throwTypeError(what + ": the grid is a width or [width, height, depth]");
    }
    if ((s.height > 1 && k.grid_dim < 2) || (s.depth > 1 && k.grid_dim < 3)) {
        ev::throwTypeError(what + ": '" + k.name + "' is a " + std::to_string(k.grid_dim) + "D grid kernel");
    }
    return s;
}

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

void installKernel() {
    ModuleRegistry::set_default_resolver([](std::string_view name) {
        return resolveRegisteredModule(name);
    });
    g_moduleClass.install("KernelModule", 0, nullptr, decorateModule, /*global=*/false);
    g_kernelClass.install("Kernel", 0, nullptr, decorateKernel, /*global=*/false);
    g_bufferClass.install("KernelBuffer", 0, nullptr, decorateBuffer, /*global=*/false);
    g_fnClass.install("KernelFn", 0, nullptr, decorateFnClass, /*global=*/false);
    ev::Persistent kernelObj(ensureBroKernel());
    ev::ObjectBuilder b(kernelObj.get());
    b.set("available", true);
    b.set("version", version());
    b.accessor("gpu", [](Value, std::span<const Value>) -> Value {
        return ev::fromBool(t_device && t_device->available());
    });
    b.def("compile", 2, js_compile);
    if (t_device) b.def("createBuffer", 1, js_createBuffer);
    decorateBroKernelExtras(b);
}

void setGpuDevice(std::shared_ptr<GpuDevice> device) {
    t_device = std::move(device);
    t_uncollectedBufferBytes = 0;
}

std::shared_ptr<const CompiledKernel> kernelOf(Value v) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(v));
    return h ? h->kernel : nullptr;
}

std::shared_ptr<GpuBuffer> bufferOf(Value v) {
    auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(v));
    return h ? h->buffer : nullptr;
}

Value bufferValue(std::shared_ptr<GpuBuffer> buffer) {
    return buffer ? makeBuffer(std::move(buffer)) : ev::undefined();
}

} // namespace brokernel::api

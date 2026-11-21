#include "api_internal.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace brokernel::api {

namespace {

ev::HostClass g_structViewClass;

class HostGpuBuffer final : public GpuBuffer {
public:
    explicit HostGpuBuffer(size_t bytes)
        : bytes_(bytes), data_(std::make_shared<std::vector<uint8_t>>(std::max<size_t>(bytes, 256), 0)) {}

    uint64_t address() const override { return 0; }
    size_t bytes() const override { return bytes_; }
    size_t capacity() const override { return data_ ? data_->size() : 0; }
    BufferResidency residency() const override { return BufferResidency::Host; }
    void* hostDataForCpu() override { return data_ ? data_->data() : nullptr; }
    bool prepareForCpu(std::string* err) override {
        if (!data_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        return true;
    }
    bool write(size_t offset, const void* data, size_t n, std::string* err) override {
        if (!data_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (offset > bytes_ || n > bytes_ - offset) {
            if (err) *err = "write out of bounds";
            return false;
        }
        if (n > 0) std::memcpy(data_->data() + offset, data, n);
        return true;
    }
    bool read(size_t offset, void* out, size_t n, std::string* err) override {
        if (!data_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (offset > bytes_ || n > bytes_ - offset) {
            if (err) *err = "read out of bounds";
            return false;
        }
        if (n > 0) std::memcpy(out, data_->data() + offset, n);
        return true;
    }
    void destroy() override {
        data_.reset();
        bytes_ = 0;
    }

    const std::shared_ptr<std::vector<uint8_t>>& backing() const { return data_; }

private:
    size_t bytes_ = 0;
    std::shared_ptr<std::vector<uint8_t>> data_;
};

class SharedGpuBuffer final : public GpuBuffer {
public:
    SharedGpuBuffer(size_t bytes, std::shared_ptr<GpuBuffer> gpu)
        : bytes_(bytes),
          host_(std::make_shared<std::vector<uint8_t>>(std::max<size_t>(bytes, 256), 0)),
          gpu_(std::move(gpu)) {}

    uint64_t address() const override { return gpu_ ? gpu_->address() : 0; }
    size_t bytes() const override { return bytes_; }
    size_t capacity() const override { return gpu_ ? gpu_->capacity() : 0; }
    BufferResidency residency() const override { return BufferResidency::Shared; }
    void* hostDataForCpu() override { return host_ ? host_->data() : nullptr; }

    bool prepareForCpu(std::string* err) override {
        if (!host_ || !gpu_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (gpuDirty_) {
            if (bytes_ > 0 && !gpu_->read(0, host_->data(), bytes_, err)) return false;
            gpuDirty_ = false;
        }
        return true;
    }
    void markHostWritten() override { hostDirty_ = true; }

    bool prepareForGpu(std::string* err) override {
        if (!host_ || !gpu_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (hostDirty_) {
            if (bytes_ > 0 && !gpu_->write(0, host_->data(), bytes_, err)) return false;
            hostDirty_ = false;
        }
        return true;
    }
    void markGpuWritten() override { gpuDirty_ = true; }

    bool write(size_t offset, const void* data, size_t n, std::string* err) override {
        if (!host_ || !gpu_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (offset > bytes_ || n > bytes_ - offset) {
            if (err) *err = "write out of bounds";
            return false;
        }
        if (gpuDirty_ && !(offset == 0 && n == bytes_)) {
            if (!prepareForCpu(err)) return false;
        }
        if (n > 0) std::memcpy(host_->data() + offset, data, n);
        hostDirty_ = true;
        gpuDirty_ = false;
        return true;
    }

    bool read(size_t offset, void* out, size_t n, std::string* err) override {
        if (!host_ || !gpu_) {
            if (err) *err = "the KernelBuffer has been destroyed";
            return false;
        }
        if (offset > bytes_ || n > bytes_ - offset) {
            if (err) *err = "read out of bounds";
            return false;
        }
        if (gpuDirty_) {
            return gpu_->read(offset, out, n, err);
        }
        if (n > 0) std::memcpy(out, host_->data() + offset, n);
        return true;
    }

    std::shared_ptr<GpuReadback> readLater(size_t offset, size_t n, std::string* err) override {
        if (!prepareForGpu(err)) return nullptr;
        return gpu_->readLater(offset, n, err);
    }

    void destroy() override {
        if (gpu_) gpu_->destroy();
        gpu_.reset();
        host_.reset();
        bytes_ = 0;
    }

    const std::shared_ptr<std::vector<uint8_t>>& backing() const { return host_; }

private:
    size_t bytes_ = 0;
    std::shared_ptr<std::vector<uint8_t>> host_;
    std::shared_ptr<GpuBuffer> gpu_;
    bool hostDirty_ = false;
    bool gpuDirty_ = false;
};

struct ViewField {
    std::string name;
    Type type;
    uint32_t byte_offset = 0;
    uint32_t byte_size = 0;
    uint32_t leaves = 1;
};

struct StructViewHandle {
    std::string struct_name;
    uint32_t byte_stride = 0;
    uint32_t count = 0;
    std::vector<ViewField> fields;
    std::unordered_map<std::string, size_t> field_index;
    std::shared_ptr<GpuBuffer> kernel_buf;
    ev::Persistent typed_array;
};

const ViewField* findViewField(const StructViewHandle& vh, std::string_view name) {
    auto it = vh.field_index.find(std::string(name));
    return it == vh.field_index.end() ? nullptr : &vh.fields[it->second];
}

uint8_t* lockViewBytes(StructViewHandle& vh, bool for_write, size_t& out_bytes, const std::string& what) {
    if (vh.kernel_buf) {
        std::string err;
        if (!vh.kernel_buf->prepareForCpu(&err)) ev::throwError(what + ": " + err);
        (void)for_write;
        out_bytes = vh.kernel_buf->bytes();
        return static_cast<uint8_t*>(vh.kernel_buf->hostDataForCpu());
    }
    const auto info = ev::typedArrayInfo(vh.typed_array.get());
    if (!info.data) ev::throwTypeError(what + ": underlying TypedArray is invalid");
    out_bytes = info.byteLength;
    return info.data;
}

void finishViewWrite(StructViewHandle& vh) {
    if (vh.kernel_buf) vh.kernel_buf->markHostWritten();
}

Value readScalarAt(const uint8_t* p, ScalarKind sk) {
    switch (sk) {
        case ScalarKind::F32: { float v; std::memcpy(&v, p, 4); return ev::fromDouble(v); }
        case ScalarKind::F64: { double v; std::memcpy(&v, p, 8); return ev::fromDouble(v); }
        case ScalarKind::I32: { int32_t v; std::memcpy(&v, p, 4); return ev::fromDouble(v); }
        case ScalarKind::U32: { uint32_t v; std::memcpy(&v, p, 4); return ev::fromDouble(v); }
        case ScalarKind::I64: { int64_t v; std::memcpy(&v, p, 8); return ev::fromDouble(static_cast<double>(v)); }
        case ScalarKind::U64: { uint64_t v; std::memcpy(&v, p, 8); return ev::fromDouble(static_cast<double>(v)); }
        case ScalarKind::Bool: { int32_t v; std::memcpy(&v, p, 4); return ev::fromBool(v != 0); }
        default: return ev::undefined();
    }
}

void writeScalarAt(uint8_t* p, ScalarKind sk, Value val) {
    switch (sk) {
        case ScalarKind::F32: { float v = static_cast<float>(ev::toDouble(val)); std::memcpy(p, &v, 4); break; }
        case ScalarKind::F64: { double v = ev::toDouble(val); std::memcpy(p, &v, 8); break; }
        case ScalarKind::I32: { int32_t v = static_cast<int32_t>(ev::toInt64(val)); std::memcpy(p, &v, 4); break; }
        case ScalarKind::U32: { uint32_t v = static_cast<uint32_t>(ev::toUint64(val)); std::memcpy(p, &v, 4); break; }
        case ScalarKind::I64: { int64_t v = ev::toInt64(val); std::memcpy(p, &v, 8); break; }
        case ScalarKind::U64: { uint64_t v = ev::toUint64(val); std::memcpy(p, &v, 8); break; }
        case ScalarKind::Bool: { int32_t v = ev::toBool(val) ? 1 : 0; std::memcpy(p, &v, 4); break; }
        default: break;
    }
}

Value readFieldValue(const uint8_t* base, const ViewField& f) {
    const uint8_t* p = base + f.byte_offset;
    if (f.type.is_scalar()) return readScalarAt(p, f.type.scalar);
    if (f.type.is_vector()) {
        const uint32_t sb = f.type.scalar_size_bytes();
        ev::Persistent arr(ev::makeArray(f.type.dimension));
        for (uint32_t i = 0; i < f.type.dimension; ++i) {
            ev::Persistent elem(readScalarAt(p + i * sb, f.type.scalar));
            arr.set(ev::setElement(arr.get(), i, elem.get()));
        }
        return arr.get();
    }
    ev::Persistent bytes(ev::createTypedArray(ev::elements::Uint8, f.byte_size));
    if (f.byte_size > 0) ev::fillTypedArray(bytes.get(), std::span<const uint8_t>(p, f.byte_size));
    return bytes.get();
}

void writeFieldValue(uint8_t* base, const ViewField& f, Value val, const std::string& what) {
    uint8_t* p = base + f.byte_offset;
    if (f.type.is_scalar()) {
        writeScalarAt(p, f.type.scalar, val);
        return;
    }
    if (f.type.is_vector()) {
        const uint32_t sb = f.type.scalar_size_bytes();
        if (ev::isTypedArray(val)) {
            const auto info = ev::typedArrayInfo(val);
            if (info.byteLength < f.byte_size) ev::throwRangeError(what + ": typed array too small for vector field");
            std::memcpy(p, info.data, f.byte_size);
            return;
        }
        if (!ev::isObject(val)) ev::throwTypeError(what + ": expected array of numbers for vector field");
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(val, "length")));
        if (n < f.type.dimension) ev::throwRangeError(what + ": vector field needs " + std::to_string(f.type.dimension) + " elements");
        for (uint32_t i = 0; i < f.type.dimension; ++i) {
            writeScalarAt(p + i * sb, f.type.scalar, ev::getElement(val, i));
        }
        return;
    }
    if (ev::isTypedArray(val)) {
        const auto info = ev::typedArrayInfo(val);
        if (info.byteLength < f.byte_size) ev::throwRangeError(what + ": typed array too small");
        std::memcpy(p, info.data, f.byte_size);
        return;
    }
    ev::throwTypeError(what + ": expected typed array for composite field '" + f.name + "'");
}

StructViewHandle* unwrapView(Value self, const std::string& what) {
    auto* h = static_cast<StructViewHandle*>(g_structViewClass.unwrap(self));
    if (!h) ev::throwTypeError(what + ": invalid StructView");
    return h;
}

uint32_t checkIndex(const StructViewHandle& vh, Value v, const std::string& what) {
    const double d = ev::toDouble(v);
    if (!(d >= 0.0) || d != static_cast<double>(static_cast<uint32_t>(d))) {
        ev::throwRangeError(what + ": index must be a non-negative integer");
    }
    const uint32_t idx = static_cast<uint32_t>(d);
    if (idx >= vh.count) ev::throwRangeError(what + ": index " + std::to_string(idx) + " >= count " + std::to_string(vh.count));
    return idx;
}

void decorateStructView(ev::ObjectBuilder& b) {
    b.accessor("byteStride", [](Value self, std::span<const Value>) -> Value {
        return ev::fromDouble(unwrapView(self, "StructView.byteStride")->byte_stride);
    });
    b.accessor("count", [](Value self, std::span<const Value>) -> Value {
        return ev::fromDouble(unwrapView(self, "StructView.count")->count);
    });
    b.accessor("fields", [](Value self, std::span<const Value>) -> Value {
        StructViewHandle* vh = unwrapView(self, "StructView.fields");
        ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(vh->fields.size())));
        for (size_t i = 0; i < vh->fields.size(); ++i) {
            const ViewField& f = vh->fields[i];
            ev::ObjectBuilder fb;
            fb.set("name", f.name);
            fb.set("type", f.type.to_string());
            fb.set("byteOffset", static_cast<double>(f.byte_offset));
            fb.set("byteSize", static_cast<double>(f.byte_size));
            ev::Persistent fv(fb.get());
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), fv.get()));
        }
        return arr.get();
    });
    b.def("get", 2, [](Value self, std::span<const Value> args) -> Value {
        StructViewHandle* vh = unwrapView(self, "StructView.get");
        if (args.size() < 2 || !ev::isString(args[1])) {
            return ev::throwTypeError("StructView.get: requires (index, fieldName)");
        }
        const uint32_t idx = checkIndex(*vh, args[0], "StructView.get");
        const std::string fname = ev::toUtf8(args[1]);
        const ViewField* f = findViewField(*vh, fname);
        if (!f) return ev::throwError("StructView.get: unknown field '" + fname + "'");
        size_t total_bytes = 0;
        const uint8_t* ptr = lockViewBytes(*vh, false, total_bytes, "StructView.get");
        const size_t base_off = static_cast<size_t>(idx) * vh->byte_stride;
        if (base_off + vh->byte_stride > total_bytes) return ev::throwRangeError("StructView.get: buffer too small");
        return readFieldValue(ptr + base_off, *f);
    });
    b.def("set", 3, [](Value self, std::span<const Value> args) -> Value {
        StructViewHandle* vh = unwrapView(self, "StructView.set");
        if (args.size() < 3 || !ev::isString(args[1])) {
            return ev::throwTypeError("StructView.set: requires (index, fieldName, value)");
        }
        const uint32_t idx = checkIndex(*vh, args[0], "StructView.set");
        const std::string fname = ev::toUtf8(args[1]);
        const ViewField* f = findViewField(*vh, fname);
        if (!f) return ev::throwError("StructView.set: unknown field '" + fname + "'");
        size_t total_bytes = 0;
        uint8_t* ptr = lockViewBytes(*vh, true, total_bytes, "StructView.set");
        const size_t base_off = static_cast<size_t>(idx) * vh->byte_stride;
        if (base_off + vh->byte_stride > total_bytes) return ev::throwRangeError("StructView.set: buffer too small");
        writeFieldValue(ptr + base_off, *f, args[2], "StructView.set");
        finishViewWrite(*vh);
        return ev::undefined();
    });
    b.def("getStruct", 1, [](Value self, std::span<const Value> args) -> Value {
        StructViewHandle* vh = unwrapView(self, "StructView.getStruct");
        if (args.empty()) return ev::throwTypeError("StructView.getStruct: requires (index)");
        const uint32_t idx = checkIndex(*vh, args[0], "StructView.getStruct");
        size_t total_bytes = 0;
        const uint8_t* ptr = lockViewBytes(*vh, false, total_bytes, "StructView.getStruct");
        const size_t base_off = static_cast<size_t>(idx) * vh->byte_stride;
        if (base_off + vh->byte_stride > total_bytes) return ev::throwRangeError("StructView.getStruct: buffer too small");
        ev::ObjectBuilder obj;
        for (const ViewField& f : vh->fields) {
            ev::Persistent fv(readFieldValue(ptr + base_off, f));
            obj.set(f.name, fv.get());
        }
        return obj.get();
    });
    b.def("setStruct", 2, [](Value self, std::span<const Value> args) -> Value {
        StructViewHandle* vh = unwrapView(self, "StructView.setStruct");
        if (args.size() < 2 || !ev::isObject(args[1])) {
            return ev::throwTypeError("StructView.setStruct: requires (index, obj)");
        }
        const uint32_t idx = checkIndex(*vh, args[0], "StructView.setStruct");
        ev::Persistent srcObj(args[1]);
        size_t total_bytes = 0;
        uint8_t* ptr = lockViewBytes(*vh, true, total_bytes, "StructView.setStruct");
        const size_t base_off = static_cast<size_t>(idx) * vh->byte_stride;
        if (base_off + vh->byte_stride > total_bytes) return ev::throwRangeError("StructView.setStruct: buffer too small");
        for (const ViewField& f : vh->fields) {
            const Value fv = ev::getProperty(srcObj.get(), f.name);
            if (!ev::isUndefined(fv)) {
                writeFieldValue(ptr + base_off, f, fv, "StructView.setStruct");
            }
        }
        finishViewWrite(*vh);
        return ev::undefined();
    });
}

void releaseSharedVec(void* user, uint8_t*) {
    delete static_cast<std::shared_ptr<std::vector<uint8_t>>*>(user);
}

bronze::ElementKind elementKindForCtor(Value ctor, uint32_t& elem_bytes) {
    elem_bytes = 1;
    if (ev::isUndefined(ctor) || ev::isNull(ctor)) return ev::elements::Uint8;
    const Value nameVal = ev::getProperty(ctor, "name");
    const std::string name = ev::isString(nameVal) ? ev::toUtf8(nameVal) : "";
    if (name == "Float32Array") { elem_bytes = 4; return ev::elements::Float32; }
    if (name == "Float64Array") { elem_bytes = 8; return ev::elements::Float64; }
    if (name == "Int32Array")   { elem_bytes = 4; return ev::elements::Int32; }
    if (name == "Uint32Array")  { elem_bytes = 4; return ev::elements::Uint32; }
    if (name == "Int16Array")   { elem_bytes = 2; return ev::elements::Int16; }
    if (name == "Uint16Array")  { elem_bytes = 2; return ev::elements::Uint16; }
    if (name == "Int8Array")    { elem_bytes = 1; return ev::elements::Int8; }
    if (name == "Uint8Array")   { elem_bytes = 1; return ev::elements::Uint8; }
    if (name == "Uint8ClampedArray") { elem_bytes = 1; return ev::elements::Uint8Clamped; }
    if (name == "BigInt64Array")  { elem_bytes = 8; return ev::elements::BigInt64; }
    if (name == "BigUint64Array") { elem_bytes = 8; return ev::elements::BigUint64; }
    ev::throwTypeError("KernelBuffer.asTypedArray: unsupported TypedArray constructor '" + name + "'");
    return ev::elements::Uint8;
}

const std::shared_ptr<std::vector<uint8_t>>* hostBackingOf(GpuBuffer* b) {
    if (auto* hb = dynamic_cast<HostGpuBuffer*>(b)) return &hb->backing();
    if (auto* sb = dynamic_cast<SharedGpuBuffer*>(b)) return &sb->backing();
    return nullptr;
}

} // namespace

Value js_createBuffer(Value, std::span<const Value> args) {
    BufferResidency res = BufferResidency::Gpu;
    if (args.size() > 1 && ev::isObject(args[1])) {
        const Value rval = ev::getProperty(args[1], "residency");
        if (ev::isString(rval)) {
            const std::string rs = ev::toUtf8(rval);
            if (rs == "gpu") res = BufferResidency::Gpu;
            else if (rs == "host") res = BufferResidency::Host;
            else if (rs == "shared") res = BufferResidency::Shared;
            else return ev::throwTypeError("bro.kernel.createBuffer: residency must be 'gpu', 'host', or 'shared'");
        }
    }
    if (res != BufferResidency::Host && !currentDevice()) {
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
    std::shared_ptr<GpuBuffer> buffer;
    if (res == BufferResidency::Host) {
        buffer = std::make_shared<HostGpuBuffer>(bytes);
    } else if (res == BufferResidency::Shared) {
        auto gpu = allocateBuffer(bytes, &err);
        if (!gpu) return ev::throwError("bro.kernel.createBuffer: " + err);
        buffer = std::make_shared<SharedGpuBuffer>(bytes, std::move(gpu));
    } else {
        buffer = allocateBuffer(bytes, &err);
        if (!buffer) return ev::throwError("bro.kernel.createBuffer: " + err);
    }

    if (fromArray && bytes > 0) {
        const auto info = ev::typedArrayInfo(args[0]);
        if (!buffer->write(0, info.data, info.byteLength, &err)) {
            return ev::throwError("bro.kernel.createBuffer: " + err);
        }
    }
    return makeBuffer(std::move(buffer));
}

void installBufferClasses() {
    g_structViewClass.install("KernelStructView", 0, nullptr, decorateStructView, /*global=*/false);
}

void decorateBufferExtras(ev::ObjectBuilder& b) {
    b.accessor("residency", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
        if (!h || !h->buffer) return ev::undefined();
        switch (h->buffer->residency()) {
            case BufferResidency::Host: return ev::fromUtf8("host");
            case BufferResidency::Shared: return ev::fromUtf8("shared");
            default: return ev::fromUtf8("gpu");
        }
    });
    b.def("syncToGpu", 0, [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
        if (!h) return ev::throwTypeError("KernelBuffer.syncToGpu: invalid receiver");
        if (!h->buffer) return ev::throwError("KernelBuffer.syncToGpu: the KernelBuffer has been destroyed");
        std::string err;
        if (!h->buffer->prepareForGpu(&err)) return ev::throwError("KernelBuffer.syncToGpu: " + err);
        return ev::undefined();
    });
    b.def("syncToHost", 0, [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
        if (!h) return ev::throwTypeError("KernelBuffer.syncToHost: invalid receiver");
        if (!h->buffer) return ev::throwError("KernelBuffer.syncToHost: the KernelBuffer has been destroyed");
        std::string err;
        if (!h->buffer->prepareForCpu(&err)) return ev::throwError("KernelBuffer.syncToHost: " + err);
        return ev::undefined();
    });
    b.def("asTypedArray", 0, [](Value self, std::span<const Value> args) -> Value {
        auto* h = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
        if (!h) return ev::throwTypeError("KernelBuffer.asTypedArray: invalid receiver");
        if (!h->buffer) return ev::throwError("KernelBuffer.asTypedArray: the KernelBuffer has been destroyed");
        std::string err;
        if (!h->buffer->prepareForCpu(&err)) return ev::throwError("KernelBuffer.asTypedArray: " + err);
        const auto* vecPtr = hostBackingOf(h->buffer.get());
        if (!vecPtr || !*vecPtr) return ev::throwError("KernelBuffer.asTypedArray: buffer has no host storage");
        uint32_t eb = 1;
        const bronze::ElementKind ek = elementKindForCtor(args.empty() ? ev::undefined() : args[0], eb);
        const size_t nbytes = h->buffer->bytes();
        auto* keep = new std::shared_ptr<std::vector<uint8_t>>(*vecPtr);
        ev::Persistent arrBuf(ev::createExternalArrayBuffer((*vecPtr)->data(),
                                                            static_cast<uint32_t>(nbytes),
                                                            releaseSharedVec,
                                                            keep));
        return ev::createTypedArrayView(ek, arrBuf.get(), 0, static_cast<uint32_t>(nbytes / eb));
    });
}

Value js_moduleStructView(Value self, std::span<const Value> args) {
    auto* mh = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
    if (!mh || !mh->cell || !mh->cell->result) {
        return ev::throwTypeError("KernelModule.structView: invalid receiver");
    }
    if (args.size() < 2 || !ev::isString(args[0])) {
        return ev::throwTypeError("KernelModule.structView: requires (structName, bufferOrTypedArray, count?)");
    }
    const std::string name = ev::toUtf8(args[0]);
    const TypeRegistry& types = mh->cell->result->types;
    const StructDef* sdef = types.find_struct(name);
    if (!sdef) {
        for (const auto& st : types.structs()) {
            auto pos = st.name.rfind("::");
            if (pos != std::string::npos && st.name.substr(pos + 2) == name) {
                sdef = &st;
                break;
            }
        }
    }
    if (!sdef) return ev::throwError("KernelModule.structView: unknown struct '" + name + "'");

    const uint32_t stride = sdef->byte_size;
    if (stride == 0) return ev::throwError("KernelModule.structView: struct '" + name + "' has zero byte size");

    auto vh = std::make_unique<StructViewHandle>();
    vh->struct_name = name;
    vh->byte_stride = stride;
    size_t byte_len = 0;
    if (auto* bh = static_cast<BufferHandle*>(g_bufferClass.unwrap(args[1]))) {
        if (!bh->buffer) return ev::throwError("KernelModule.structView: KernelBuffer has been destroyed");
        if (bh->buffer->residency() == BufferResidency::Gpu) {
            return ev::throwTypeError("KernelModule.structView: requires a TypedArray or 'host'/'shared' KernelBuffer");
        }
        vh->kernel_buf = bh->buffer;
        byte_len = bh->buffer->bytes();
    } else if (ev::isTypedArray(args[1])) {
        vh->typed_array.set(args[1]);
        byte_len = ev::typedArrayInfo(args[1]).byteLength;
    } else {
        return ev::throwTypeError("KernelModule.structView: second argument must be a TypedArray or KernelBuffer");
    }

    uint32_t max_count = static_cast<uint32_t>(byte_len / stride);
    if (args.size() > 2 && !ev::isUndefined(args[2])) {
        const double d = ev::toDouble(args[2]);
        if (!(d >= 0.0) || d != static_cast<double>(static_cast<uint32_t>(d))) {
            return ev::throwRangeError("KernelModule.structView: count must be a non-negative integer");
        }
        const uint32_t req = static_cast<uint32_t>(d);
        if (req > max_count) {
            return ev::throwRangeError("KernelModule.structView: buffer too small for " + std::to_string(req) + " element(s)");
        }
        vh->count = req;
    } else {
        vh->count = max_count;
    }

    for (size_t i = 0; i < sdef->fields.size(); ++i) {
        const auto& sf = sdef->fields[i];
        ViewField vf;
        vf.name = sf.name;
        vf.type = sf.type;
        vf.byte_offset = sf.byte_offset;
        vf.byte_size = types.byte_size(sf.type);
        vf.leaves = sf.flat_leaf_count;
        vh->field_index[vf.name] = i;
        vh->fields.push_back(std::move(vf));
    }
    return g_structViewClass.createInstance(std::move(vh));
}

} // namespace brokernel::api

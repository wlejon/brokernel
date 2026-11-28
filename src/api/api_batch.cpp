#include "api_internal.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace brokernel::api {

namespace {

struct PendingReadback {
    ev::Realm* realm = nullptr;
    ev::Persistent promise;
    std::shared_ptr<GpuReadback> readback;
    bronze::ElementKind elemKind = ev::elements::Uint8;
    uint32_t elemBytes = 1;
    size_t bytes = 0;
};

thread_local std::vector<PendingReadback> t_pendingReads;
thread_local uint32_t t_batchDepth = 0;

size_t parseOptionalOffset(std::span<const Value> args, size_t idx, size_t defVal, const char* what) {
    if (args.size() <= idx || ev::isUndefined(args[idx])) return defVal;
    const double d = ev::toDouble(args[idx]);
    if (!(d >= 0.0) || d != static_cast<double>(static_cast<size_t>(d))) {
        ev::throwRangeError(std::string(what) + ": byte offset/length must be a non-negative integer");
    }
    return static_cast<size_t>(d);
}

Value resolveReadbackVec(std::shared_ptr<std::vector<uint8_t>> vec,
                           size_t nbytes,
                           bronze::ElementKind ek,
                           uint32_t eb) {
    return makeTypedArrayFromVec(std::move(vec), nbytes, ek, eb);
}

void settleOneReadback(PendingReadback& pr) {
    if (!pr.realm) return;
    ev::RealmScope scope(pr.realm);
    ev::Persistent p(pr.promise.get());
    pr.promise.set(ev::undefined());
    const size_t n = std::min(pr.bytes, pr.readback ? pr.readback->bytes() : 0);
    auto vec = std::make_shared<std::vector<uint8_t>>(std::max<size_t>(n, 1), 0);
    if (n > 0 && pr.readback && pr.readback->data()) {
        std::memcpy(vec->data(), pr.readback->data(), n);
    }
    ev::resolvePromise(p.get(), resolveReadbackVec(std::move(vec), n, pr.elemKind, pr.elemBytes));
}

Value js_bufferReadAsync(Value self, std::span<const Value> args) {
    auto* bh = static_cast<BufferHandle*>(g_bufferClass.unwrap(self));
    if (!bh) return ev::throwTypeError("KernelBuffer.readAsync: invalid receiver");
    if (!bh->buffer) return ev::throwError("KernelBuffer.readAsync: KernelBuffer has been destroyed");
    GpuBuffer& buf = *bh->buffer;

    Value ctor = ev::undefined();
    size_t nextIdx = 0;
    if (!args.empty() && (ev::isUndefined(args[0]) || ev::isNull(args[0]) || ev::isFunction(args[0]))) {
        ctor = args[0];
        nextIdx = 1;
    }

    bronze::ElementKind ek = ev::elements::Uint8;
    uint32_t eb = 1;
    typedArrayKindForCtor(ctor, ek, eb);

    const size_t offset = parseOptionalOffset(args, nextIdx, 0, "KernelBuffer.readAsync");
    if (offset > buf.bytes()) {
        return ev::throwRangeError("KernelBuffer.readAsync: byte offset out of range");
    }
    const size_t rem = buf.bytes() - offset;
    const size_t n = parseOptionalOffset(args, nextIdx + 1, rem, "KernelBuffer.readAsync");
    if (n > rem) {
        return ev::throwRangeError("KernelBuffer.readAsync: read range exceeds buffer length");
    }

    ev::Persistent promise(ev::createPromise());

    if (n == 0) {
        auto vec = std::make_shared<std::vector<uint8_t>>(1, 0);
        ev::resolvePromise(promise.get(), resolveReadbackVec(std::move(vec), 0, ek, eb));
        return promise.get();
    }

    if (buf.residency() == BufferResidency::Host) {
        const auto* backing = hostBackingOf(&buf);
        if (!backing || !*backing || offset + n > (*backing)->size()) {
            ev::rejectPromise(promise.get(), ev::fromUtf8("KernelBuffer.readAsync: invalid host backing"));
            return promise.get();
        }
        auto vec = std::make_shared<std::vector<uint8_t>>(n);
        std::memcpy(vec->data(), (*backing)->data() + offset, n);
        ev::resolvePromise(promise.get(), resolveReadbackVec(std::move(vec), n, ek, eb));
        return promise.get();
    }

    std::string err;
    std::shared_ptr<GpuReadback> rb = buf.readLater(offset, n, &err);
    if (rb) {
        if (rb->ready()) {
            auto vec = std::make_shared<std::vector<uint8_t>>(n);
            if (rb->data()) std::memcpy(vec->data(), rb->data(), n);
            ev::resolvePromise(promise.get(), resolveReadbackVec(std::move(vec), n, ek, eb));
            return promise.get();
        }
        PendingReadback pr;
        pr.realm = ev::currentRealm();
        pr.promise.set(promise.get());
        pr.readback = std::move(rb);
        pr.elemKind = ek;
        pr.elemBytes = eb;
        pr.bytes = n;
        t_pendingReads.push_back(std::move(pr));
        return promise.get();
    }

    auto vec = std::make_shared<std::vector<uint8_t>>(n);
    std::string readErr;
    if (buf.read(offset, vec->data(), n, &readErr)) {
        ev::resolvePromise(promise.get(), resolveReadbackVec(std::move(vec), n, ek, eb));
    } else {
        ev::rejectPromise(promise.get(), ev::fromUtf8(readErr.empty() ? err : readErr));
    }
    return promise.get();
}

Value js_broKernelBatch(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isFunction(args[0])) {
        return ev::throwTypeError("bro.kernel.batch: requires a callback function");
    }
    GpuDevice* dev = currentDevice();
    const bool outer = (t_batchDepth == 0);
    ++t_batchDepth;
    if (outer && dev) dev->beginBatch();

    auto callRes = ev::call(args[0], ev::undefined(), {});

    --t_batchDepth;
    std::string flushErr;
    bool flushOk = true;
    if (outer && dev) flushOk = dev->endBatch(&flushErr);

    if (callRes.thrown) return ev::throwValue(callRes.value);
    if (!flushOk) return ev::throwError("bro.kernel.batch: " + flushErr);
    return callRes.value;
}

} // namespace

void settleAsyncReadbacks(bool waitAll) {
    (void)waitAll;
    if (t_pendingReads.empty()) return;
    for (size_t i = 0; i < t_pendingReads.size();) {
        if (!t_pendingReads[i].readback || t_pendingReads[i].readback->ready()) {
            PendingReadback done = std::move(t_pendingReads[i]);
            t_pendingReads.erase(t_pendingReads.begin() + static_cast<std::ptrdiff_t>(i));
            settleOneReadback(done);
        } else {
            ++i;
        }
    }
}

void decorateBufferBatch(ev::ObjectBuilder& b) {
    b.def("readAsync", 0, js_bufferReadAsync);
}

void decorateBroKernelBatch(ev::ObjectBuilder& b) {
    b.def("batch", 1, js_broKernelBatch);
}

} // namespace brokernel::api

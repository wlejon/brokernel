#pragma once

// brokernel's bronze JavaScript binding: `bro.kernel`.
//
//   const mod = bro.kernel.compile(source, { file, modules });
//   const k = mod.kernel('k_height');
//   k.eval(x, y, z, seed)              // a point kernel at one position, on the CPU
//   k.run([w, h], src, out, 0.5)       // a grid kernel over the grid, on the CPU
//
// A store argument is a typed array of the store's element type (any typed
// array for a vector or struct element, read as raw bytes); every access to
// it stays inside it (the language's bounded stores, README.md). Compiling
// reads nothing from disk: imports resolve against the standard library and
// the `modules` given as source.
//
// The GPU half runs on the host's device (GpuDevice, setGpuDevice):
//
//   const buf = bro.kernel.createBuffer(new Float32Array(field));  // GPU memory
//   k.dispatch([w, h], buf, out, 0.5)  // a grid kernel over the grid, on the GPU
//   out.read(new Float32Array(w * h))  // after every dispatch so far
//
// A store argument of dispatch() is a KernelBuffer, bounded like a typed
// array is on the CPU. kernelOf() / bufferOf() hand the host the compiled
// kernel and the buffer behind a Kernel or KernelBuffer object.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "embed/embed.h"

namespace brokernel {
class CompiledKernel;
struct GpuLaunch;
}

namespace brokernel::api {

/// A buffer in the host's GPU memory: a KernelBuffer's storage.
class GpuBuffer {
public:
    virtual ~GpuBuffer() = default;
    /// Its device address: a store argument of a dispatch.
    virtual uint64_t address() const = 0;
    /// Its bytes (a KernelBuffer's byteLength).
    virtual size_t bytes() const = 0;
    /// The bytes behind address(), at least bytes() (KernelArg::device).
    virtual size_t capacity() const = 0;
    /// Copy `n` bytes of `data` (now) to byte `offset`, after every dispatch
    /// so far.
    virtual bool write(size_t offset, const void* data, size_t n, std::string* err) = 0;
    /// Copy `n` bytes at byte `offset` into `out`, after every dispatch so
    /// far (blocks until they are done).
    virtual bool read(size_t offset, void* out, size_t n, std::string* err) = 0;
};

/// A compiled kernel as the host's pipeline (GpuDevice::createProgram).
class GpuProgram {
public:
    virtual ~GpuProgram() = default;
};

/// The host's GPU device: what bro.kernel's GPU half runs on. Its work runs
/// in the order it is asked for.
class GpuDevice {
public:
    virtual ~GpuDevice() = default;
    /// The device can run work now (`bro.kernel.gpu`; asked at every read,
    /// since a host's device may come up after its realms are installed).
    virtual bool available() const = 0;
    /// A buffer of `bytes` bytes, reading zero.
    virtual std::shared_ptr<GpuBuffer> createBuffer(size_t bytes, std::string* err) = 0;
    /// `kernel`'s SPIR-V as a pipeline; null (and `err`) when the device
    /// cannot run it.
    virtual std::shared_ptr<GpuProgram> createProgram(const CompiledKernel& kernel, std::string* err) = 0;
    /// One dispatch of `program` (made from `kernel`).
    virtual bool dispatch(GpuProgram& program, const CompiledKernel& kernel, const GpuLaunch& launch,
                          std::string* err) = 0;
};

/// The device the current thread's realms dispatch on (null: none). Set it
/// before installKernel: a realm installed without one has
/// `bro.kernel.gpu === false` and no createBuffer(); with one,
/// `bro.kernel.gpu` is its available().
void setGpuDevice(std::shared_ptr<GpuDevice> device);

/// Mounts `bro.kernel` into the current bronze realm.
void installKernel();

/// The compiled kernel a `Kernel` object holds, kept alive with the module
/// it came from; nullptr for any other value.
std::shared_ptr<const CompiledKernel> kernelOf(bronze::Value v);

/// The GPU buffer a `KernelBuffer` object holds; nullptr for any other value.
std::shared_ptr<GpuBuffer> bufferOf(bronze::Value v);

} // namespace brokernel::api

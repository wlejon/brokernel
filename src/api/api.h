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
//
// Compiling off the page thread:
//
//   const mod = await bro.kernel.compileAsync(source, { file, modules });
//
// compileAsync takes compile()'s arguments and returns a Promise of the same
// KernelModule, rejected with the same Error (its `.diagnostics`) where
// compile() would throw. On a thread with a wait pump (setThreadWaitPump) it
// starts the compile on a worker at once, so an app can start all its
// compiles together; the promises settle at the host's settleAsyncCompiles().
// On a thread with no pump it compiles at once and returns a settled promise.
// With a pump, compile() and the other compiles the API makes (withKnobs,
// knobs(), fn(), reload()) also run on a worker while the thread waits,
// pumping. Either way bro.kernel.resolve's function runs on this thread
// alone: a worker asks it for a module the registered, resolved and host
// modules do not hold.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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
    /// Release the buffer's GPU memory once in-flight work finishes.
    virtual void destroy() {}
};

/// Driver-level executable statistics for a compiled pipeline stage.
struct ExecutableStatEntry {
    std::string name;
    double value = 0.0;
};
struct ExecutableStats {
    std::string name;
    std::string description;
    uint32_t subgroupSize = 0;
    std::vector<ExecutableStatEntry> statistics;
};

/// A compiled kernel as the host's pipeline (GpuDevice::createProgram).
class GpuProgram {
public:
    virtual ~GpuProgram() = default;
    virtual std::vector<ExecutableStats> executableStats() { return {}; }
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
    /// Reclaim GPU memory from destroyed buffers whose work has finished.
    virtual void reclaim() {}
    /// `kernel`'s SPIR-V as a pipeline; null (and `err`) when the device
    /// cannot run it. A host may build it on a thread of its own, its first
    /// dispatch waiting for it (and failing with the build's error); the API
    /// makes a grid kernel's program with its handle (KernelModule.kernel).
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

/// A `KernelBuffer` object over a buffer the host made (a view into memory
/// it owns, say), in the current realm; undefined for null.
bronze::Value bufferValue(std::shared_ptr<GpuBuffer> buffer);

/// Resolves a module by name using the current thread/realm's registered
/// modules (`bro.kernel.registerModule`) or JS resolver (`bro.kernel.resolve`).
std::optional<std::string> resolveRegisteredModule(std::string_view name);

/// Registers a built-in host module (e.g. "tiles") that persists across clearModules().
void registerHostModule(std::string_view name, std::string_view source);

/// What the current thread runs while it waits for a compile worker: called
/// every `intervalMs` (and when the worker asks it for a module) until the
/// worker is done. The host's pump lets the OS process its window's messages;
/// it must not run script. With one set, this thread's compiles run on
/// workers (see the header comment); null removes it.
using WaitPump = void (*)();
void setThreadWaitPump(WaitPump pump, int intervalMs);

/// On the thread that called compileAsync, once a frame: answers its
/// in-flight compiles' module requests and settles the finished ones'
/// promises, each in the realm that called compileAsync. With `waitAll` it
/// waits (pumping) until every compile of this thread has finished and is
/// settled, so a promise settles at the first call after compileAsync
/// (deterministic for headless runs). The promises' reactions run at the
/// host's next microtask checkpoint.
void settleAsyncCompiles(bool waitAll);

/// Drops this thread's pending compileAsync jobs (their promises never
/// settle) and joins their workers. A realm destroyed with jobs pending drops
/// its own the same way; their workers are joined here or by a later
/// settleAsyncCompiles.
void shutdownKernelAsync();

} // namespace brokernel::api

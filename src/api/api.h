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
// The GPU half belongs to the host, which dispatches a kernel's SPIR-V on its
// own device: kernelOf() hands it the compiled kernel behind a Kernel object.

#include <memory>

#include "embed/embed.h"

namespace brokernel {
class CompiledKernel;
}

namespace brokernel::api {

/// Mounts `bro.kernel` into the current bronze realm.
void installKernel();

/// The compiled kernel a `Kernel` object holds, kept alive with the module
/// it came from; nullptr for any other value.
std::shared_ptr<const CompiledKernel> kernelOf(bronze::Value v);

} // namespace brokernel::api

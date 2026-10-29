#pragma once
// brokernel: the kernel language for world code, compiled through brass to
// SPIR-V for the GPU and to native code for the CPU. See README.md.

namespace brokernel {

/// The library's version, "major.minor.patch".
const char* version();

}  // namespace brokernel

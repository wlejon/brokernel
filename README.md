# brokernel

The kernel language for world code. A world's systems (a planet's heights,
erosion, climate, its surfaces) are written as kernels in brokernel, and one
kernel source compiles through [brass](https://github.com/wlejon/brass) to
SPIR-V for the GPU and to native code for the CPU. The same function that
fills a terrain tile on the GPU answers a height query for physics, the
camera or placement on the CPU, so there is no second copy to keep in step.

brokernel knows nothing of scenes, planets or bro. It is a compiler library:
source in, diagnostics or compiled kernels out.

## Where it sits

```
world code (an app's .bk files)
      │  brokernel: parse, check, lower
      ▼
brass MIR (KernelBuilder)
      ├── SpirvTarget  → SPIR-V (compute, vertex, fragment) for Vulkan
      └── brass native → x64 / arm64 functions for the CPU
```

- **brass** does the code generation. Its SPIR-V target already lowers MIR
  built with `KernelBuilder` (the same MIR its PTX target takes) to verified
  SPIR-V 1.5/1.6 for the compute, vertex and fragment stages, with f64
  (brass `docs/spirv_backend_design.md`, `docs/ptx_kernel_authoring.md`).
  brokernel is a frontend over it, as bronze (JavaScript) and brasslua (Lua)
  are frontends over brass's CPU backends.
- **bro** links brokernel and exposes general mechanisms only: run a kernel
  over a grid into a store as world-service work, call a kernel's CPU build,
  let a kernel supply a surface's height, normal and material inside the
  engine's own shading. The engine owns scheduling, residency, budgets,
  level-of-detail invariants and observability; it holds no world's logic.
- **Worlds** live in apps (broworkshop): kernel sources plus the JavaScript
  that wires them to the scene. Deleting a world is deleting its folder.

## The language

brokernel is a small, statically typed language in the family of WGSL,
written for one author (a coding agent) and for world kernels rather than
general graphics:

- **Modules.** A kernel imports the functions it uses (noise, filters, a
  world's own fields) by module, not by textual include.
- **Generics.** One noise or filter written once over its dimension and
  scalar type.
- **f64 as a first-class scalar.** Positions on an Earth-radius body need
  it; a kernel says where precision is needed and where f32 suffices.
- **Grids, stencils and stores as language constructs.** A kernel is
  written per point of a grid; neighbour reads are stencils the compiler
  sees; the stores it reads and writes are typed. That is what lets one
  source become both a GPU dispatch and a CPU point query, and lets the
  engine schedule and cache what a kernel makes.
- **The same semantics on both targets.** A kernel's CPU and GPU builds
  agree to within each target's floating-point rounding, and a test
  compares them.

Source files use the `.bk` extension.

## Layout

| Path | What |
|------|------|
| `include/brokernel/brokernel.h` | the public API (`brokernel::version()`) |
| `src/` | the library |
| `tests/` | `test_smoke` (the library links, brass's SPIR-V target is reachable); `no_dialogs.cpp` keeps Windows test executables from opening modal dialogs |
| `cmake/bro_deps.cmake` | the ecosystem's shared dependency resolver (identical in every repo; `scripts/sync-deps.sh` in bro copies it) |

## Building

Dependencies resolve like every sibling: an outer project's target, else
`../brass` beside this checkout, else brass's main fetched at configure.

Windows (Visual Studio generator, one build dir, the config chosen at build
time):

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

Linux / macOS:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release
```

`-DBROKERNEL_BUILD_TESTS=OFF` skips the tests (off by default when brokernel
is a subproject).

## License

MIT (see `LICENSE`).

# brokernel

The kernel language built so a coding agent can write, see inside, and tune
world code fast.

A world's systems (a planet's heights, erosion, climate, its surfaces) are
written as kernels in brokernel, and one kernel source compiles through
[brass](https://github.com/wlejon/brass) to SPIR-V for the GPU and to native
code for the CPU. The same function that fills a terrain tile on the GPU
answers a height query for physics, the camera or placement on the CPU, so
there is no second copy to keep in step.

## Where it sits

```
world code (an app's .bk files)
      │  brokernel: parse, check, AD / quadrature, lower
      ▼
brass MIR (KernelBuilder)
      ├── SpirvTarget  → SPIR-V (compute, vertex, fragment) for Vulkan
      └── brass native → x64 / arm64 functions for the CPU
```

- **brass** does the code generation. Its SPIR-V target lowers MIR built with
  `KernelBuilder` to verified SPIR-V 1.5/1.6 for compute, vertex and fragment
  stages. brokernel is a frontend over it, as bronze (JavaScript) and brasslua
  (Lua) are frontends over brass's CPU backends.
- **bro.kernel** (`src/api/`) exposes the compiler, CPU/GPU kernel execution,
  function harnesses, probes, knobs, and constant/layout reflection to bronze.

## Agent-first language features

brokernel is a statically typed language in the family of WGSL, designed for
rapid world iteration by a coding agent:

- **Named probes (`probe("label", expr)`).** Wrap any scalar or vector
  subexpression in `probe("cu", cu)` inside any function, compute kernel, or
  fragment stage. It returns its argument unchanged and compiles to a zero-cost
  no-op in normal builds. When probed (`kernel.probe(...)`, `fn.probe(...)`,
  `CompiledKernel::probe_cpu` / `probe_gpu`, or fragment stage probing at
  `@binding(4, 23)`), it records the exact intermediate value at the target
  cell or pixel (`{ cu: 0.041, ridge: 0.72, ... }`) without debug packing into
  color outputs.
- **Knobs without source edits (`@knob const`).** Mark a tunable constant
  `@knob const CU_CAP: f32 = 0.06;`. Override any set of knobs at compile time
  or on a compiled module/kernel (`mod.withKnobs({ CU_CAP: 0.04 })`,
  `k.knobs({ CU_CAP: 0.04 }).dispatch(...)`, `CompileResult::with_knobs`) to
  sweep parameters in-process without rewriting `.bk` files. `mod.knobs`
  reflects all declared knobs and their current values.
- **Forward-mode automatic differentiation (`grad`).** `grad(f, x, ...extra)`
  differentiates any function `f` in scope with respect to its first parameter
  `x` (`f32`, `f64`, or `vec2`..`vec4` for scalar-returning `f`), returning
  the exact analytical derivative or gradient vector $\nabla f(x)$. Terrain
  slopes, B-spline derivatives, soil gradients, and flow directions stay in
  sync with the height function automatically.
- **Quadrature and footprint moments (`expect`, `moments`, `cell_expect`).**
  - `expect(f, mean, sigma, ...extra)` evaluates $\mathbb{E}[f(X)]$ for
    $X \sim \mathcal{N}(\text{mean}, \sigma^2)$ via 3-point Gauss-Hermite
    quadrature (for scalar or vector returns).
  - `moments(f, mean, sigma, ...extra)` returns `vec2(E[f(X)], Var[f(X)])`
    from the same 3 quadrature points.
  - `cell_expect(f, center, du, dv, ...extra)` integrates `f(center + s*du + t*dv, ...)`
    over $[-0.5, 0.5]^2$ via $2\times 2$ Gauss-Legendre quadrature for coarse
    LOD cell averages.
- **Tests over functions, not pixels (`mod.fn(name)`).** Call any `pub fn`
  directly from JavaScript without writing a `@kernel` wrapper by hand:
  - `f.eval(...args)` evaluates one point on the CPU.
  - `f.run(count, ...args)` runs `count` invocations on the CPU where each
    argument can be either a broadcast scalar/vector or a per-invocation
    `TypedArray`.
  - `f.dispatch(count, ...args)` runs `count` invocations on the GPU with
    `KernelBuffer`s for per-invocation arguments.
- **Modules by name & re-exports (`pub import` / `pub use`).**
  - `import` and `use` are synonyms (`import a::b::*;`, `use a::b::{x, y};`,
    `from a::b import x, y;`).
  - Prefixing with `pub` (`pub import sub::*;`, `pub use sub::{foo, BAR};`,
    `pub from sub import foo;`) re-exports those functions, structs, constants,
    and descriptor bindings through the module, so a facade module can split
    internals across files without breaking callers.
  - `bro.kernel.resolve(fn)` and `bro.kernel.registerModule(name, source)`
    (and `Compiler::set_module_resolver` in C++) resolve imported modules on
    demand by name (`"world::fields"`), so callers never have to thread a
    `{ modules }` map by hand.
- **Shared declarations with JS (`mod.consts`, `mod.structLayout`).**
  - Every `pub const` (and re-exported `pub const`) in a module is reflected on
    `mod.consts` as a JS number, boolean, or array (`CompileResult::constants`
    in C++).
  - `mod.structLayout(name)` returns `{ name, byteSize, std140Size, std140Align, std430Size, std430Align, fields: [{ name, type, byteOffset, std140Offset, std430Offset }] }`
    so host code and kernels share one source of truth for buffer layouts.

## Core execution model

- **Generics.** Write a noise or filter once over its dimension and scalar type.
- **GPU code is f32; f64 is the CPU's.** A kernel built for the GPU that
  reaches `f64` is a compile error at the `f64` site. Precision far from the
  origin comes from anchors: compute an anchor position in `f64` on the CPU and
  pass `f32` offsets to the GPU. A kernel marked `@cpu` has no GPU build and
  may use `f64` freely (e.g., planetary height queries for physics and camera).
- **Grids, stencils and bounded stores.** A kernel is written per point of a
  grid (`grid<1..3>`); neighbour reads are `stencil<T, D>` the compiler sees;
  `read_store` / `write_store` / `rw_store` carry their element bounds on both
  CPU and GPU so out-of-bounds reads return zero and out-of-bounds writes do
  nothing. Grid linear indices (`g.linear_index()`, `g.total_cells()`) are
  64-bit.
- **Vertex and fragment stages (`@vertex`, `@fragment`).** Stage inputs at
  `@location(n)` (`@flat` supported), outputs via `set_vertex_output`,
  `set_frag_output`, `set_frag_color`, `set_vertex_position`, `set_frag_depth`,
  and builtins `vertex_index()`, `instance_index()`, `frag_coord()`,
  `front_facing()`, `discard_fragment()`.
- **Descriptor resources (`@binding(set, binding)`).** Combined samplers
  (`sampler2d<T>`, `sampler_cube<T>`, shadow samplers), separate `texture*` and
  `sampler`, storage images (`image2d<T>`, `@format(...)`), `uniform<T>`
  (std140), and `buffer<T>` / `read_buffer<T>` (std430). Module-scope
  `@binding(set, binding) pub var camera: uniform<Camera>;` declarations are
  bound automatically by any stage that reaches them.
- **Fast math & CPU vectorization.** GPU builds use GLSL.std.450 fast math; CPU
  builds use FMA and Cephes polynomials (`src/lowering_math.cpp`), vectorizing
  grid loops 8-wide (or 4-wide for 64-bit) on AVX2 hosts (`src/lowering_ops.cpp`).

Source files use the `.bk` extension.

## Compile cache

`Compiler::compile_source` (and everything built on it: `compile_file`,
`with_knobs`, `with_sources`, probed variants, `bro.kernel.compile`) loads a
result from disk when `brokernel::set_compile_cache({dir, max_bytes})` names a
directory (`include/brokernel/compile_cache.h`; off by default, and per
compiler with `Compiler::set_use_compile_cache(false)`).

- **Key:** the compiler build (the binary holding brokernel's code: path,
  size, mtime; or `CompileCacheConfig::build_stamp`), brokernel's version, the
  host's AVX2/FMA, the source, its file name and every `CompileOptions` field
  (knob overrides in any order). An entry also records each module the compile
  read (the import as asked, where it resolved, a hash of its source); a lookup
  resolves each again as the compiler would now, without parsing, and a changed
  or missing module is a miss. Only successful compiles are stored.
- **A hit is the compile:** every `CompileResult` and `CompiledKernel` field
  (SPIR-V, reflection, types, constants, probes, costs, hashes, diagnostics),
  with source, options, modules, search paths and resolver from the compiler.
  A kernel's CPU build is lowered on its first CPU run by compiling the source
  again for that kernel (a `@cpu` kernel's too), and its probed variant when
  first asked for. `CompileResult::cache_status` (`Off` / `Hit` / `Miss`) and
  `cache_note` (why it missed) say which; `compile_cache_stats()` counts them.
- **Store:** one `<key>.bkc` file per entry, written to a temporary file and
  renamed, so threads and processes share a directory; a header with a format
  version and checksum makes a truncated, damaged or foreign entry a miss. A hit
  touches its mtime; a store that takes the directory past `max_bytes` (whole
  4 KiB units) has a background thread delete the least recently used entries
  down to 7/8 of it, and `flush_compile_cache()` waits for that trim. Deleting
  the directory is always safe.

## Layout

| Path | What |
|------|------|
| `include/brokernel/brokernel.h` | public C++ API: `compile_source` / `compile_file` / `Compiler`, `CompileResult` (`with_knobs`, `constants`, `probes`), `CompiledKernel` (CPU/GPU dispatch, point eval, `probe_cpu`, `probe_gpu`, `compare_cpu_gpu`) |
| `include/brokernel/{ast,types,sema,lowering,module_loader,diagnostics}.h` | compiler stages and `ModuleRegistry` |
| `include/brokernel/compile_cache.h`, `src/compile_cache*.cpp`, `src/compile_result_io.cpp` | the on-disk compile cache: configuration and lookup around `compile_source`, the entry files and their trim, a result's byte encoding |
| `src/lexer.cpp`, `src/parser.cpp` | source to AST (`import` / `use`, `pub import`, `@knob`, `probe`) |
| `src/sema*.cpp`, `src/types.cpp` | name resolution, re-exports, generics, type checking, forward-mode AD (`src/sema_ad.cpp`), quadrature (`expect`, `moments`, `cell_expect`) |
| `src/lowering_*.cpp` | AST to brass MIR: GPU SPIR-V entry points, CPU point/grid functions, and CPU/GPU/fragment probes |
| `src/stdlib_*.cpp` | built-in modules `std::math`, `std::noise`, `std::filter`, `std::sphere`, `std::color` |
| `src/api/api.h`, `src/api/api*.cpp` | bronze JS bindings (`bro.kernel`): `compile`, `compileAsync` (off-thread compiles under a host wait pump, `api_async.cpp`), `resolve`, `registerModule`, `KernelModule` (`kernel`, `fn`, `consts`, `knobs`, `withKnobs`, `structLayout`), `Kernel` (`eval`, `run`, `dispatch`, `probe`, `knobs`), `KernelBuffer` |
| `tests/` | unit and parity tests for lexer/parser, sema, modules/generics, SPIR-V stages, CPU/GPU parity, math, FastNoise2 parity, and agent wishlist features (`test_wishlist.cpp`) |
| `bench/bench_hillclimb*.cpp` | parity and speed harness for `std::noise`, 2D stencils, and planetary `f64` height fields against FastNoise2 and `glslc` |

## Building

Windows:

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Linux / macOS:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release --output-on-failure
```

## License

MIT (see `LICENSE`).

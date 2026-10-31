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
- Nothing links brokernel yet; its tests drive it directly.

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
  run the same operations:
  - The GPU build is emitted with brass's `precise_float`, so the driver
    rounds every float add, mul and div as the CPU does. It may not fuse
    them into FMAs or re-associate them.
  - The transcendentals (`sin` through `atan2`, `exp`/`log` and their base-2
    forms, `pow`, the hyperbolics) are lowered as MIR arithmetic
    (`src/lowering_math.cpp`), not called. That avoids both the host's libm
    and the driver's GLSL.std.450, and covers f64, which GLSL.std.450 has no
    transcendentals for.
  - f64 is within a few ulp of libm, and f32 uses Cephes' single-precision
    polynomials.
  - Render math is the exception: a vertex or fragment stage, and a
    kernel marked `@fast`, is emitted without `precise_float`, so the driver
    may fuse and reorder, and its f32 transcendentals are the device's own
    GLSL.std.450 instructions. A `@fast` kernel's CPU build keeps the exact
    polynomials, so its two builds agree only to the device's precision.
  - `compare_cpu_gpu` dispatches both builds and compares what they wrote. It
    fails, with `ran_gpu` false, when there is no Vulkan device.
- **A store access never leaves its store.** A kernel's store takes its
  element count with its address, on both targets:
  - An access at the grid's cells (a stencil read, `read()` / `write()`, or
    an index of the cell's own coordinates: `g.x()`, `g.y()`, `g.z()` in
    their places, or `g.linear_index()` into a 1D store) is bounded by the
    dispatch. The store must hold the grid's cells over its axes
    (`KernelParamInfo::grid_cover_dim`), or the dispatch fails naming it.
  - Any other access is bounded where it is made: outside the store a read
    is zero, a write and an atomic do nothing. In the CPU's vector loop a
    contiguous access tests its whole group once and goes lane by lane only
    when the group runs past the store's end.
  - A point query's cell has to lie in its grid.
- **Grid indices are 64-bit.** `g.linear_index()`, `g.total_cells()` (and
  `grid_linear_index()`, `grid_total_cells()`) are i64, stores take i32 or
  i64 coordinates, and the CPU's grid loop indexes in i64, so a grid may
  pass 2^31 cells.
- **Vertex and fragment stages** (`@vertex`, `@fragment`):
  - Their inputs are parameters at `@location(n)`: f32, i32 or u32, or a
    vector of one. A fragment input is interpolated unless it is marked
    `@flat`; an integer input is always flat.
  - `set_vertex_output(location, value)` and `set_frag_output(location,
    value)` take the same types. `set_frag_color(rgba)` is location 0.
  - Also `set_vertex_position`, `set_frag_depth`, `vertex_index()`,
    `instance_index()`, `frag_coord()` and `front_facing()`.
  - Their other parameters are push constants.
- **Textures are parameters at a descriptor**, `@binding(set, binding)`:
  - Samplers (a combined image and sampler) are `sampler1d`, `sampler2d`,
    `sampler3d`, `sampler_cube`, `sampler2d_array` and `sampler_cube_array`,
    each `<T>` for an f32 or u32 texel or a vector of either (`vec4<f32>` when
    left out). The shadow forms `sampler2d_shadow`, `sampler2d_array_shadow`
    and `sampler_cube_shadow` compare to an f32.
  - A texture sampled through a separate `sampler` takes the same shapes
    named `texture*` (`texture2d<T>`, `texture_cube<T>`,
    `texture2d_array_shadow`, ...). Its filtering methods take the sampler
    first (`albedo.sample(linear, uv)`, `shadow.compare(cmp, uv, ref)`);
    `fetch`, `size` and `levels` take none. One sampler serves any number of
    textures.
  - A sampler's methods:
    - `sample(uv)` at the fragment's implicit LOD, `sample_lod(uv, lod)` and
      `sample_grad(uv, dx, dy)`. Each has an `_offset` form taking constant
      texels, except on a cube.
    - `fetch(texel, lod)`, `gather(uv, component)`, `size(lod)` and `levels()`.
    - On a shadow sampler, `compare(uv, ref)`, `compare_lod(uv, ref, lod)`
      and `gather(uv, ref)`.
  - Coordinates are f32 to filter and i32 for texels; a texture array's layer
    comes after its dimensions.
  - Storage images are `image1d`, `image2d`, `image3d` and `image2d_array`,
    with `load`, `store` and `size`. Their format is `@format(rgba16f)` (and
    the rest of Vulkan's storage formats), or else the one that holds `T`
    whole.
  - A texture passed to a helper function still names its descriptor.
    `CompiledKernel::spirv.descriptors` lists each texture's shape.
- **Buffers at a descriptor** are parameters too, `@binding(set, binding)`:
  - `uniform<T>` holds one T, laid out std140.
  - `buffer<T>` and `read_buffer<T>` hold an array of T, laid out std430.
    Only `buffer<T>` is written.
  - T is a struct, or a 32-bit scalar or vector.
  - An access is a path of fields, indices (dynamic ones too) and swizzles.
    It reads or writes only the words the path names, at the offset its
    layout gives.
  - A kernel that takes a texture or a buffer runs on the GPU alone.
- **A module may declare the resources it reads**, `@binding(set, binding)
  pub var camera: uniform<Camera>;` (any texture, sampler, image or buffer
  type). Any function of the module or of one importing it reads them by
  name: unqualified, through `from scene import camera`, or as
  `scene::camera`. A stage binds only the ones it reaches, and a kernel that
  reaches one runs on the GPU alone. An engine's frame state (its camera,
  lights, shared samplers) is written this way once, and every stage
  composed with it reads that state without taking parameters for it.
- **Type introspection for generics.** `is_f64<T>()` and `sizeof<T>()`
  fold to constants at check time, so a generic body can branch on its
  scalar; `bitcast<T>(x)` reinterprets a value's bits as a type of the same
  width (f32 / i32 / u32, f64 / i64 / u64).
- **The CPU's grid loop is vectorized on AVX2 hosts** (`src/lowering_ops.*`),
  ISPC-style, 8 cells along x at a time, or 4 when a varying value is 64-bit
  (f64 work):
  - A value is uniform (a scalar, the same for every cell) or varying (a
    brass vector). Bools are lane masks.
  - A varying `if` runs both arms under masks and merges by select. A loop
    runs while any lane is still in it. Integer division, 64-bit multiplies
    and gathers go lane by lane.
  - A 64-bit index is a uniform base plus a varying i32 offset, never an
    i64 vector, so 8 lanes hold. An index of the form base + lane × stride
    (a store read or written at `g.x()`) loads and stores contiguously;
    any other goes lane by lane at scalar addresses.
  - A group whose stencil reads all land inside the row (its x at least
    the kernel's x halo from either end) reads them unclamped, so a
    stencil's neighbours are contiguous loads too; the groups at a row's
    ends clamp.
  - A row's last `width % lanes` cells run one at a time.
  - A kernel with a call on varying arguments or an atomic runs every cell
    one at a time, as do non-AVX2 hosts. `CompiledKernel::cpu_grid_lanes`
    says which build it got.
- **CPU builds are for compute kernels.** The CPU runs a kernel as a subgroup
  and workgroup of one:
  - The subgroup operations are the identity there.
  - Vertex and fragment stages have no CPU build. Neither do kernels that
    cooperate through barriers or shared memory, or that take a texture or
    a buffer at a descriptor.
  - Derivatives, `front_facing` and `discard_fragment` (a demote to helper)
    are fragment-only.
  - A kernel's native code is compiled the first time something runs it on
    the CPU (`cpu_point_fn()` / `cpu_grid_fn()`), so a module whose kernels
    only run on the GPU compiles in the time its SPIR-V takes. A failure
    then is in `cpu_error()`. SPIR-V disassembly is only made on request
    (`CompileOptions::emit_spirv_disassembly`).

Source files use the `.bk` extension.

## Layout

| Path | What |
|------|------|
| `include/brokernel/brokernel.h` | the public API: `compile_source` / `compile_file` / `Compiler`, `CompiledKernel` (CPU and GPU dispatch, point evaluation, `compare_cpu_gpu`), `KernelArg`, `GridShape` |
| `include/brokernel/{ast,types,sema,lowering,module_loader,diagnostics}.h` | the compiler's stages |
| `src/lexer.cpp`, `src/parser.cpp` | source to AST |
| `src/sema*.cpp`, `src/types.cpp` | name resolution, generics, type checking, builtin signatures |
| `src/lowering_mir.cpp`, `src/lowering_expr.cpp`, `src/lowering_math.cpp` | AST to brass MIR: the GPU entry point and the CPU point / grid functions |
| `src/stdlib_modules.cpp` | the built-in modules `std::math`, `std::noise`, `std::filter`, `std::sphere`, `std::color`. `std::noise` is FastNoise2's nodes ported (its primes, hashes, gradients and bounds): Value, Perlin, Simplex, SuperSimplex, White, Checkerboard, SineWave, the FBm / Ridged / PingPong fractals, the cellular value / distance / lookup nodes, and the gradient and fractal domain warps, in 2D, 3D and 4D, f32 or f64. In f64 a lattice noise rebases on its own cell, so it holds about an origin millions of units out, as FastNoise2's double origins do |
| `tests/` | lexer/parser, sema, modules and generics, SPIR-V stages, CPU/GPU parity (against host references), the transcendentals against libm on both targets, and `std::noise` against FastNoise2 on both (43 node / dimension cases, to 1e-5; the cellular and warp nodes to 1e-4, FastNoise2 being built relaxed). A GPU test with no Vulkan device exits 77 (skipped). `no_dialogs.cpp` keeps Windows test executables from opening modal dialogs |
| `bench/bench_hillclimb.cpp` | the hill-climb harness: parity and speed of `std::noise` against FastNoise2 (CPU) and against glslc's SPIR-V of the same node (GPU: every f32 node, hand-written GLSL in std::noise's order of operations), five stencil kernels and the planet's rivers (its erosion pass: drainage, water level and stream power over eight neighbours, from broworkshop `1fbd228^:demos/planet/terrain.js`) against C++ and GLSL, and the planet's noise fields (eight FBm-of-Simplex fields in f64 about a double origin) against FastNoise2, in metres, both over a grid and one position per call (`eval_f32` against `GenSingle3D`, the height query physics and the camera make). It prints the geomeans, the GPU's also over the compute-bound rows alone (the cheap nodes run at the memory's rate in either build); `--write-golden` records them in `bench/goldens.windows.json` and `--check-golden` fails on a parity failure or a geomean 10% under the golden's. `-DBROKERNEL_BUILD_BENCH=OFF` skips it |
| `cmake/bro_deps.cmake` | the ecosystem's shared dependency resolver (identical in every repo; `scripts/sync-deps.sh` in bro copies it) |

## Building

Dependencies resolve like every sibling: an outer project's target, else
`../brass` beside this checkout, else brass's main fetched at configure. The
tests and the bench also take FastNoise2 (wlejon's fork, `../FastNoise2` or
fetched) as their oracle; the library itself does not link it.

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

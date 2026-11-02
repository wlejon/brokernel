#include "bench_hillclimb.h"

uint32_t ulp_distance(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return UINT32_MAX;
    int32_t ia = 0, ib = 0;
    std::memcpy(&ia, &a, sizeof(float));
    std::memcpy(&ib, &b, sizeof(float));
    if (ia < 0) ia = int32_t(0x80000000u) - ia;
    if (ib < 0) ib = int32_t(0x80000000u) - ib;
    return static_cast<uint32_t>(std::abs(int64_t(ia) - int64_t(ib)));
}

ErrorStats compare_buffers(const std::vector<float>& got, const std::vector<float>& ref, float tol) {
    ErrorStats s{};
    if (got.size() != ref.size()) return s;
    s.pass = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) {
            s.pass = false;
            s.max_abs = INFINITY;
            s.max_ulp = UINT32_MAX;
            break;
        }
        float err = std::abs(got[i] - ref[i]);
        if (err > s.max_abs) s.max_abs = err;
        if (err > 1e-6f) {
            s.max_ulp = std::max(s.max_ulp, ulp_distance(got[i], ref[i]));
        }
        if (err > tol) s.pass = false;
    }
    return s;
}

std::vector<uint32_t> compile_glsl_compute(const std::string& glsl_src, const std::string& tag) {
    fs::path tmp_dir = fs::temp_directory_path() / "brokernel_hillclimb_spv";
    fs::create_directories(tmp_dir);
    fs::path comp_path = tmp_dir / (tag + ".comp");
    fs::path spv_path = tmp_dir / (tag + ".spv");

    // Cache compiled SPIR-V if source matches
    bool need_compile = !fs::exists(spv_path);
    if (!need_compile && fs::exists(comp_path)) {
        std::ifstream existing(comp_path, std::ios::binary);
        std::ostringstream ss;
        ss << existing.rdbuf();
        if (ss.str() != glsl_src) need_compile = true;
    } else {
        need_compile = true;
    }

    if (need_compile) {
        {
            std::ofstream ofs(comp_path, std::ios::binary);
            ofs << glsl_src;
        }
        std::string glslc = "glslc";
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        if (const char* sdk = std::getenv("VULKAN_SDK")) {
            fs::path p = fs::path(sdk) / "Bin" / "glslc";
            if (fs::exists(p) || fs::exists(p.string() + ".exe")) glslc = p.string();
        }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        std::string cmd = "\"\"" + glslc + "\" --target-env=vulkan1.3 -O -fshader-stage=compute \"" +
                          comp_path.string() + "\" -o \"" + spv_path.string() + "\"\"";
        int rc = std::system(cmd.c_str());
        if (rc != 0 || !fs::exists(spv_path)) {
            std::cerr << "glslc failed for " << tag << "\n";
            return {};
        }
    }

    std::ifstream ifs(spv_path, std::ios::binary | std::ios::ate);
    size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint32_t> words(sz / 4);
    ifs.read(reinterpret_cast<char*>(words.data()), sz);
    return words;
}

// Wrap a glslc-compiled SPIR-V binary into a brass::target::SpirvKernel with the given push-constant layout
brass::gpu::VulkanModule load_glsl_vulkan_module(
    const std::vector<uint32_t>& words,
    const std::vector<brass::Type>& param_types,
    uint32_t lx, uint32_t ly, uint32_t lz) {
    brass::target::SpirvKernel sk;
    sk.entry = "main";
    sk.stage = brass::target::SpirvStage::Compute;
    sk.words = words;
    sk.local_size[0] = lx;
    sk.local_size[1] = ly;
    sk.local_size[2] = lz;
    sk.local_size_spec_constants = false;

    uint32_t offset = 0;
    for (const auto& t : param_types) {
        uint32_t sz = static_cast<uint32_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;
        offset = (offset + sz - 1) & ~(sz - 1);
        sk.params.push_back({t, offset, sz});
        offset += sz;
    }
    sk.push_constant_bytes = offset;

    std::string err;
    auto mod = brass::gpu::VulkanModule::load(sk, &err);
    if (!mod.valid()) {
        std::cerr << "VulkanModule::load failed for GLSL shader: " << err << "\n";
    }
    return mod;
}


std::vector<brass::gpu::VulkanArg> glsl_args(const std::vector<GpuArg>& args) {
    std::vector<brass::gpu::VulkanArg> out;
    for (const auto& a : args) out.push_back(a.arg);
    return out;
}

double time_bk_gpu_ms(
    const CompiledKernel* k,
    GridShape shape,
    const std::vector<GpuArg>& args_without_grid_and_out,
    uint32_t repeat) {
    std::string err;
    auto vk_mod = brass::gpu::VulkanModule::load(k->spirv, &err);
    if (!vk_mod.valid()) {
        std::cerr << "Failed to load bk VulkanModule for " << k->name << ": " << err << "\n";
        return 1e9;
    }

    // The buffers are the kernel's stores, in order.
    std::vector<uint32_t> store_elem_bytes;
    for (const auto& p : k->params) {
        if (p.role == ParamRole::Store || p.role == ParamRole::Stencil) store_elem_bytes.push_back(p.element_bytes);
    }
    std::vector<brass::gpu::VulkanArg> full_args;
    size_t store_i = 0;
    for (const auto& a : args_without_grid_and_out) {
        full_args.push_back(a.arg);
        if (a.buf) {
            const uint32_t eb = store_i < store_elem_bytes.size() ? store_elem_bytes[store_i++] : 4;
            full_args.emplace_back(static_cast<int64_t>(a.buf->size() / std::max(eb, 1u)));
        }
    }
    uint32_t w = std::max(1u, shape.width);
    uint32_t h = std::max(1u, shape.height);
    uint32_t d = std::max(1u, shape.depth);
    if (k->grid_dim >= 1) full_args.emplace_back(static_cast<int32_t>(w));
    if (k->grid_dim >= 2) full_args.emplace_back(static_cast<int32_t>(h));
    if (k->grid_dim >= 3) full_args.emplace_back(static_cast<int32_t>(d));

    brass::gpu::VulkanBuffer ret_buf;
    if (k->has_return_value()) {
        ret_buf = brass::gpu::VulkanBuffer::alloc(shape.total_points() * std::max<size_t>(k->return_byte_size, 4));
        full_args.emplace_back(ret_buf);
    }

    uint32_t lx = std::max(1u, k->local_size[0]);
    uint32_t ly = std::max(1u, k->local_size[1]);
    uint32_t lz = std::max(1u, k->local_size[2]);

    brass::gpu::VulkanDispatch disp;
    disp.grid[0] = (k->grid_dim >= 1) ? (w + lx - 1) / lx : 1;
    disp.grid[1] = (k->grid_dim >= 2) ? (h + ly - 1) / ly : 1;
    disp.grid[2] = (k->grid_dim >= 3) ? (d + lz - 1) / lz : 1;
    disp.block[0] = lx;
    disp.block[1] = ly;
    disp.block[2] = lz;

    // Warmup
    disp.repeat = 1;
    double warm_ms = 0.0;
    disp.gpu_ms = &warm_ms;
    vk_mod.launch(disp, full_args, &err);

    // Timed repeat
    double total_ms = 0.0;
    disp.repeat = repeat;
    disp.gpu_ms = &total_ms;
    vk_mod.launch(disp, full_args, &err);
    return (total_ms > 0.0) ? (total_ms / repeat) : 1e-4;
}

double time_glsl_gpu_ms(
    const brass::gpu::VulkanModule& vk_mod,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t lx, uint32_t ly, uint32_t lz,
    const std::vector<brass::gpu::VulkanArg>& vk_args,
    uint32_t repeat) {
    std::string err;
    brass::gpu::VulkanDispatch disp;
    disp.grid[0] = gx;
    disp.grid[1] = gy;
    disp.grid[2] = gz;
    disp.block[0] = lx;
    disp.block[1] = ly;
    disp.block[2] = lz;

    disp.repeat = 1;
    double warm_ms = 0.0;
    disp.gpu_ms = &warm_ms;
    vk_mod.launch(disp, vk_args, &err);

    double total_ms = 0.0;
    disp.repeat = repeat;
    disp.gpu_ms = &total_ms;
    vk_mod.launch(disp, vk_args, &err);
    return (total_ms > 0.0) ? (total_ms / repeat) : 1e-4;
}

const char* kBkNoiseSuiteSource = R"BK(
    import std::noise::*;

    @kernel @grid(1)
    fn k_value2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_value3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_value4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_perlin2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_perlin3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_perlin4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_simplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_simplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = simplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_supersimplex2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_supersimplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_supersimplex4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = supersimplex4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_white2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white2d<f32>(vec2<f32>(px[i], py[i]), seed);
    }
    @kernel @grid(1)
    fn k_white3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed);
    }
    @kernel @grid(1)
    fn k_white4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = white4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed);
    }

    @kernel @grid(1)
    fn k_fbm2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_fbm3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_fbm4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = fbm_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
    }

    @kernel @grid(1)
    fn k_ridged2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 2>(vec2<f32>(px[i], py[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_ridged3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 4, 2.0, 0.5, 0.25, seed);
    }
    @kernel @grid(1)
    fn k_ridged4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ridged_perlin<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 4, 2.0, 0.5, 0.25, seed);
    }

    @kernel @grid(1)
    fn k_pingpong2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin2d<f32>(vec2<f32>(px[i], py[i]), seed), 2.0);
    }
    @kernel @grid(1)
    fn k_pingpong3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed), 2.0);
    }
    @kernel @grid(1)
    fn k_pingpong4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = ping_pong_fn<f32>(perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed), 2.0);
    }

    @kernel @grid(1)
    fn k_cell_val2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_val3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_val4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_value4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 1.0);
    }

    @kernel @grid(1)
    fn k_cell_dist2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance2d<f32>(vec2<f32>(px[i], py[i]), seed, 0, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_dist3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 0, 0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_dist4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_distance4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 0, 0, 1.0);
    }

    @kernel @grid(1)
    fn k_cell_lookup2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin2d<f32>(vec2<f32>(px[i], py[i]), seed, 1.0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_lookup3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 1.0, 1.0);
    }
    @kernel @grid(1)
    fn k_cell_lookup4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x(); out[i] = cellular_lookup_perlin4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 1.0, 1.0);
    }

    @kernel @grid(1)
    fn k_warp_grad2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient2d<f32>(vec2<f32>(px[i], py[i]), seed, 1.5, 0.5);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_grad3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient3d<f32>(vec3<f32>(px[i], py[i], pz[i]), seed, 1.5, 0.5);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_grad4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_gradient4d<f32>(vec4<f32>(px[i], py[i], pz[i], pw[i]), seed, 1.5, 0.5);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @grid(1)
    fn k_warp_prog2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 2>(vec2<f32>(px[i], py[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_prog3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_prog4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_progressive<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @grid(1)
    fn k_warp_indep2d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 2>(vec2<f32>(px[i], py[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin2d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_indep3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 3>(vec3<f32>(px[i], py[i], pz[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin3d<f32>(wp, seed);
    }
    @kernel @grid(1)
    fn k_warp_indep4d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, pw: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {
        let i = g.x();
        let wp = domain_warp_fractal_independent<f32, 4>(vec4<f32>(px[i], py[i], pz[i], pw[i]), 3, 2.0, 0.5, 0.25, 1.5, 0.5, seed);
        out[i] = perlin4d<f32>(wp, seed);
    }

    @kernel @cpu @grid(1)
    fn k_planet_f64_simplex3d(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>, out: write_store<f32, 1>, origin: f64, seed: i32) {
        let i = g.x();
        let p = vec3<f64>((px[i] as f64) + origin, (py[i] as f64) + origin, (pz[i] as f64) + origin);
        out[i] = simplex3d<f64>(p, seed) as f32;
    }
)BK";

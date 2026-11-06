// Shipped production shaders & tile kernels benchmark (Wishlist 5.1):
// Compiles and measures the heavy production stages from ../broworkshop/demos/planet
// and ../bro/src/scene/vulkan/shaders when present alongside brokernel:
//   - Planet tile kernels: k_tile, k_color, k_horizon (ground.bk)
//   - Cloud tile kernels:  k_cloud_tile, k_cloud_sun, k_cloud_ground (clouds.bk)
//   - Ground mesh shader:  mesh_hook_vert, mesh_hook_frag (ground_mesh.bk + mesh_main.bk)
//   - Cloud volume shader: volume_frag (cloud_hook.bk + volume_main.bk)
// Reports compile time (ms), SPIR-V instruction breakdown (total, ALU, texture
// samples, memory ops, branches), and GPU dispatch time (ms) for tile kernels.

#include "bench_hillclimb.h"
#include <regex>

namespace {

std::string read_file_str(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

struct SpirvCounts {
    size_t words = 0;
    uint32_t total_insts = 0;
    uint32_t alu_insts = 0;
    uint32_t sample_insts = 0;
    uint32_t mem_insts = 0;
    uint32_t branch_insts = 0;
};

SpirvCounts analyze_spirv_words(const std::vector<uint32_t>& words) {
    SpirvCounts c;
    c.words = words.size();
    if (words.size() < 5) return c;
    bool in_func = false;
    size_t i = 5;
    while (i < words.size()) {
        uint32_t w0 = words[i];
        uint16_t wc = static_cast<uint16_t>(w0 >> 16);
        uint16_t op = static_cast<uint16_t>(w0 & 0xFFFFu);
        if (wc == 0 || i + wc > words.size()) break;
        if (op == 54) in_func = true;       // OpFunction
        else if (op == 56) in_func = false; // OpFunctionEnd
        else if (in_func && op != 55 && op != 59 && op != 248 && op != 245) {
            // Count executable instructions inside functions (excluding OpFunctionParameter, OpVariable, OpLabel, OpPhi)
            c.total_insts++;
            if ((op >= 87 && op <= 99) || op == 103) {
                c.sample_insts++; // OpImageSample*, OpImageFetch, OpImageRead, OpImageWrite
            } else if (op == 61 || op == 62 || op == 65 || op == 66) {
                c.mem_insts++; // OpLoad, OpStore, OpAccessChain, OpInBoundsAccessChain
            } else if (op >= 249 && op <= 255) {
                c.branch_insts++; // OpBranch, OpBranchConditional, OpSwitch, OpReturn, etc.
            } else if ((op >= 126 && op <= 207) || op == 12) {
                c.alu_insts++; // conversions, arithmetic, logical, OpExtInst (GLSL.std.450)
            }
        }
        i += wc;
    }
    return c;
}

std::string expand_debug_views(const std::string& src) {
    static const std::regex kRe(
        R"RE(\bSCENE_DEBUG\s*\(\s*([A-Za-z0-9_-]+)\s*\))RE",
        std::regex::ECMAScript);
    return std::regex_replace(src, kRe, "(scene::debug_view() == 999999)");
}

void register_planet_modules(Compiler& c, const fs::path& planet_dir, const fs::path& bro_dir) {
    c.register_module("tiles", read_file_str(bro_dir / "src" / "tilecache" / "tiles.bk"));
    for (const char* name : {"ground", "surface", "ground_detail", "relief", "rivers", "anchor", "climate", "dendry", "dendry_grid", "dendry_field",
                             "clouds", "cloud_field", "cloud_lighting", "trees", "tree_crown", "grass", "grass_floor", "sun"}) {
        fs::path p = planet_dir / (std::string(name) + ".bk");
        if (fs::exists(p)) {
            c.register_module(std::string("planet::") + name, read_file_str(p));
        }
    }
    c.register_module("planet::tree_layout", R"BK(
module planet::tree_layout;
pub const N: i32 = 512;
pub const B: i32 = 8;
pub const FAR: f32 = 1900.0;
pub const SUN_STRIDE: i32 = 4;
pub const SUN_P: i32 = 129;
pub const MAX_TREE_H: f32 = 30.0;
pub const SHADOW_REACH: f32 = 64.0;
pub const SHADOW_STEPS: i32 = 14;
)BK");
    c.register_module("planet::grass_layout", R"BK(
module planet::grass_layout;
pub const N: i32 = 128;
pub const P: i32 = 129;
pub const THIN: f32 = 6.0;
pub const FAR: f32 = 40.0;
pub const DENSE: f32 = 128.0;
pub const SUN_STRIDE: i32 = 4;
pub const SUN_P: i32 = 33;
pub const GRASS_FLOOR: f32 = 0.55;
pub const FIELD: i32 = 8;
)BK");
    std::ostringstream gl;
    gl << "module planet::ground_layout;\n"
       << "pub const CELLS: i32 = 36;\n"
       << "pub const BORDER: i32 = 2;\n"
       << "pub const TILE: i32 = 32;\n"
       << "pub const DETAIL: i32 = 4;\n"
       << "pub const TEXELS: i32 = 144;\n"
       << "pub const HZ: i32 = 18;\n"
       << "pub const FACE_EXTENT: f32 = 10007543.0;\n"
       << "pub const DETAIL_PERIOD: f32 = 64.0;\n"
       << "pub const SWELL_PERIOD: f32 = 1024.0;\n"
       << "pub const SEED: i32 = 7;\n"
       << "pub const GL_TILE: i32 = 32;\n"
       << "pub const GL_BORDER: i32 = 1;\n"
       << "pub const GL_ATLAS: i32 = 2;\n"
       << "pub const GL_LEVELS: i32 = 14;\n"
       << "pub const WAVE_COUNT: i32 = 4;\n"
       << "pub const WAVE_SWELL: i32 = 2;\n"
       << "pub const WAVE_FRESH: i32 = 2;\n"
       << "pub const WAVE_RESIDUAL: f32 = 0.0116;\n"
       << "pub const WAVE_FRESH_SCALE: f32 = 0.5;\n"
       << "pub const WAVE_K: array<vec4<f32>, 4> = [\n"
       << "    vec4<f32>(0.02, 0.01, -0.02, 40.0),\n"
       << "    vec4<f32>(0.05, 0.03, -0.05, 16.0),\n"
       << "    vec4<f32>(0.50, 0.30, -0.50, 1.6),\n"
       << "    vec4<f32>(1.20, 0.80, -1.20, 0.6)\n"
       << "];\n"
       << "pub const WAVE_F: array<vec4<f32>, 4> = [\n"
       << "    vec4<f32>(0.5, 0.02, 0.1, 0.002),\n"
       << "    vec4<f32>(0.8, 0.015, 0.05, 0.002),\n"
       << "    vec4<f32>(2.5, 0.01, 0.01, 0.002),\n"
       << "    vec4<f32>(4.0, 0.005, 0.004, 0.002)\n"
       << "];\n";
    c.register_module("planet::ground_layout", gl.str());
}

void register_scene_modules(Compiler& c, const fs::path& shaders_dir) {
    c.register_module("wind", read_file_str(shaders_dir / "common" / "wind.bk"));
    c.register_module("scene", read_file_str(shaders_dir / "common" / "scene.bk"));
    c.register_module("atmosphere", read_file_str(shaders_dir / "common" / "atmosphere.bk"));
    c.register_module("lighting", read_file_str(shaders_dir / "common" / "lighting.bk"));
    c.register_module("custom", read_file_str(shaders_dir / "common" / "custom.bk"));
    c.register_module("vegetation", read_file_str(shaders_dir / "common" / "vegetation.bk"));
    c.register_module("procedural", read_file_str(shaders_dir / "common" / "procedural.bk"));
    c.register_module("bark", read_file_str(shaders_dir / "draw" / "bark.bk"));
    c.register_module("mesh", read_file_str(shaders_dir / "draw" / "mesh.bk"));
    c.register_module("volume", read_file_str(shaders_dir / "draw" / "volume.bk"));
    c.register_module("near", read_file_str(shaders_dir / "draw" / "near_grid.bk"));
}

} // namespace

void run_shipped_shaders_bench() {
    const fs::path root = fs::path(BROKERNEL_SOURCE_DIR).parent_path();
    const fs::path planet_dir = root / "broworkshop" / "demos" / "planet";
    const fs::path bro_dir = root / "bro";
    const fs::path shaders_dir = bro_dir / "src" / "scene" / "vulkan" / "shaders";
    if (!fs::exists(planet_dir / "ground.bk") || !fs::exists(shaders_dir / "draw" / "mesh_main.bk")) {
        return;
    }

    struct StageReport {
        std::string stage_name;
        std::string entry_name;
        double compile_ms = 0.0;
        SpirvCounts counts;
        double gpu_us = 0.0;
    };
    std::vector<StageReport> reports;

    auto bench_compile = [&](const std::string& label, const std::string& main_src, const std::string& fname,
                             const std::string& hook_src, const std::vector<std::string>& entries) {
        for (const std::string& entry : entries) {
            Compiler compiler;
            register_planet_modules(compiler, planet_dir, bro_dir);
            register_scene_modules(compiler, shaders_dir);
            if (!hook_src.empty()) {
                compiler.register_module("hook", hook_src);
            }
            CompileOptions opts;
            opts.opt_level = 3;
            opts.compile_cpu = false;
            opts.entry_name = entry;

            auto t0 = std::chrono::high_resolution_clock::now();
            CompileResult res = compiler.compile_source(main_src, fname, opts);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!res.ok || res.kernels.empty()) {
                std::cerr << "Shipped shader bench failed compiling " << label << " (" << entry << "):\n"
                          << res.error_message() << "\n";
                continue;
            }
            const CompiledKernel& k = res.kernels.front();
            StageReport rep;
            rep.stage_name = label;
            rep.entry_name = entry;
            rep.compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            rep.counts = analyze_spirv_words(k.spirv.words);
            reports.push_back(std::move(rep));
        }
    };

    const std::string planet_src = read_file_str(planet_dir / "planet.bk");
    const std::string clouds_src = read_file_str(planet_dir / "clouds.bk");
    const std::string ground_mesh_src = expand_debug_views(read_file_str(planet_dir / "ground_mesh.bk"));
    const std::string clouds_vol_src = expand_debug_views(read_file_str(planet_dir / "clouds_volume.bk"));
    const std::string mesh_main_src = read_file_str(shaders_dir / "draw" / "mesh_main.bk");
    const std::string volume_main_src = read_file_str(shaders_dir / "draw" / "volume_main.bk");

    bench_compile("planet/planet.bk", planet_src, "planet.bk", "", {"k_tile", "k_detail", "k_horizon"});
    bench_compile("planet/clouds.bk", clouds_src, "clouds.bk", "", {"k_cloud_tile", "k_cloud_sun", "k_cloud_ground"});
    bench_compile("planet/ground_mesh.bk", mesh_main_src, "mesh_main.bk", ground_mesh_src,
                  {"mesh_hook_vert", "mesh_hook_frag"});
    bench_compile("planet/clouds_volume.bk", volume_main_src, "volume_main.bk", clouds_vol_src, {"volume_main"});

    std::printf("\n============================================================================================================================\n");
    std::printf("SHIPPED PLANET SHADERS & TILE KERNELS (Wishlist 5.1):\n");
    std::printf("%-22s | %-18s | %10s | %10s | %9s | %9s | %8s | %8s | %8s\n",
                "Module", "Entry", "Compile ms", "SPIRV KB", "Insts", "ALU", "Samples", "MemOps", "Branches");
    std::printf("----------------------------------------------------------------------------------------------------------------------------\n");
    for (const auto& r : reports) {
        std::printf("%-22s | %-18s | %10.1f | %10.2f | %9u | %9u | %8u | %8u | %8u\n",
                    r.stage_name.c_str(), r.entry_name.c_str(), r.compile_ms,
                    (r.counts.words * 4.0) / 1024.0, r.counts.total_insts,
                    r.counts.alu_insts, r.counts.sample_insts, r.counts.mem_insts, r.counts.branch_insts);
    }
    std::printf("============================================================================================================================\n");
}

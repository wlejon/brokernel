#include "bench_hillclimb.h"

namespace {
void write_golden_json(const fs::path& path, const std::vector<BenchRow>& rows,
                       int parity_passed, int parity_total,
                       double geomean_cpu, double geomean_gpu, double geomean_gpu_compute) {
    std::ofstream ofs(path);
    ofs << "  \"parity_passed\": " << parity_passed << ",\n";
    ofs << "  \"parity_total\": " << parity_total << ",\n";
    ofs << "  \"geomean_cpu_vs_ref\": " << std::fixed << std::setprecision(4) << geomean_cpu << ",\n";
    ofs << "  \"geomean_gpu_vs_glsl\": " << std::fixed << std::setprecision(4) << geomean_gpu << ",\n";
    ofs << "  \"geomean_gpu_compute_vs_glsl\": " << std::fixed << std::setprecision(4) << geomean_gpu_compute << ",\n";
    ofs << "  \"benchmarks\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        ofs << "    {\n";
        ofs << "      \"tier\": \"" << r.tier << "\",\n";
        ofs << "      \"name\": \"" << r.name << "\",\n";
        ofs << "      \"dim\": " << r.dim << ",\n";
        ofs << "      \"points\": " << r.points << ",\n";
        ofs << "      \"cpu_pass\": " << (r.cpu_pass ? "true" : "false") << ",\n";
        ofs << "      \"gpu_pass\": " << (r.gpu_pass ? "true" : "false") << ",\n";
        ofs << "      \"cpu_max_abs\": " << std::scientific << std::setprecision(3) << r.cpu_max_abs << ",\n";
        ofs << "      \"cpu_max_ulp\": " << r.cpu_max_ulp << ",\n";
        ofs << "      \"gpu_max_abs\": " << std::scientific << std::setprecision(3) << r.gpu_max_abs << ",\n";
        ofs << "      \"gpu_max_ulp\": " << r.gpu_max_ulp << ",\n";
        ofs << "      \"cpu_bk_mpts\": " << std::fixed << std::setprecision(2) << r.cpu_bk_mpts << ",\n";
        ofs << "      \"cpu_ref_mpts\": " << std::fixed << std::setprecision(2) << r.cpu_ref_mpts << ",\n";
        ofs << "      \"cpu_ratio\": " << std::fixed << std::setprecision(4) << r.cpu_ratio << ",\n";
        ofs << "      \"gpu_bk_mpts\": " << std::fixed << std::setprecision(2) << r.gpu_bk_mpts << ",\n";
        ofs << "      \"gpu_glsl_mpts\": " << std::fixed << std::setprecision(2) << r.gpu_glsl_mpts << ",\n";
        ofs << "      \"gpu_ratio\": " << std::fixed << std::setprecision(4) << r.gpu_ratio << "\n";
        ofs << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

// A top-level number of the golden file ("key": value), or a negative when absent.
double read_golden_number(const fs::path& path, const std::string& key) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string text = ss.str();
    size_t at = text.find("\"" + key + "\"");
    if (at == std::string::npos) return -1.0;
    at = text.find(':', at);
    return at == std::string::npos ? -1.0 : std::strtod(text.c_str() + at + 1, nullptr);
}

// One of the planet's FastNoise fields (broworkshop 1fbd228^:demos/planet/terrain.js):
// FractalFBm of Simplex at a feature scale, lacunarity 2.03, times an amplitude.
struct PlanetField {
    float scale;
    int octaves;
    int seed_offset;
    float gain;
    float amp;
};

// WARP, CONTINENT and RELIEF, the four regional relief bands, and TREE.
const PlanetField kPlanetFields[] = {
    {2.2e6f, 4, 7, 0.5f, 1.0f},
    {3.0e6f, 6, 9, 0.55f, 1.0f},
    {60000.0f, 2, 0, 0.5f, 1.0f},
    {500.0f, 2, 31, 0.5f, 30.0f},
    {2000.0f, 2, 32, 0.5f, 120.0f},
    {8000.0f, 2, 33, 0.5f, 330.0f},
    {32000.0f, 2, 34, 0.5f, 520.0f},
    {120.0f, 3, 3, 0.5f, 3.0f},
};

// The fields summed as one FastNoise2 tree: Add(Add(f0, f1), f2)..., a field
// with an amplitude a Multiply by it.
FastNoise::SmartNode<> planet_fields_tree() {
    FastNoise::SmartNode<> sum;
    for (const auto& f : kPlanetFields) {
        auto simplex = FastNoise::New<FastNoise::Simplex>();
        simplex->SetScale(f.scale);
        simplex->SetSeedOffset(f.seed_offset);
        auto fbm = FastNoise::New<FastNoise::FractalFBm>();
        fbm->SetSource(simplex);
        fbm->SetOctaveCount(f.octaves);
        fbm->SetLacunarity(2.03f);
        fbm->SetGain(f.gain);
        FastNoise::SmartNode<> field = fbm;
        if (f.amp != 1.0f) {
            auto mul = FastNoise::New<FastNoise::Multiply>();
            mul->SetLHS(fbm);
            mul->SetRHS(f.amp);
            field = mul;
        }
        if (!sum) {
            sum = field;
        } else {
            auto add = FastNoise::New<FastNoise::Add>();
            add->SetLHS(sum);
            add->SetRHS(field);
            sum = add;
        }
    }
    return sum;
}

// The same fields as one brokernel kernel, in f64 about a double origin. Its
// constants are FastNoise2's floats (1 / scale, 2.03, the gains) widened, so
// both evaluate the same lattice positions.
// The grid kernel k_planet_fields over position stores, and the point kernel
// k_planet_point at one position (what a height query for physics or the
// camera calls), both summing the fields at p in planet_height.
std::string planet_fields_source() {
    std::ostringstream ss;
    ss << std::setprecision(17);
    ss << "import std::noise::*;\n"
          "fn planet_height(p: vec3<f64>, seed: i32) -> f32 {\n";
    bool first = true;
    for (const auto& f : kPlanetFields) {
        ss << (first ? "    var h: f32 = " : "    h = h + ");
        ss << "(fbm_simplex<f64, 3>(p * " << double(1.0f / f.scale) << ", " << f.octaves << ", "
           << double(2.03f) << ", " << double(f.gain) << ", 0.0, seed + " << f.seed_offset << ") as f32)";
        if (f.amp != 1.0f) ss << " * " << std::fixed << std::setprecision(1) << f.amp << std::defaultfloat << std::setprecision(17);
        ss << ";\n";
        first = false;
    }
    ss << "    return h;\n}\n"
          "@kernel @cpu @grid(1)\n"
          "fn k_planet_fields(g: grid<1>, px: read_store<f32, 1>, py: read_store<f32, 1>, pz: read_store<f32, 1>,\n"
          "                   out: write_store<f32, 1>, ox: f64, oy: f64, oz: f64, seed: i32) {\n"
          "    let i = g.x();\n"
          "    out[i] = planet_height(vec3<f64>((px[i] as f64) + ox, (py[i] as f64) + oy, (pz[i] as f64) + oz), seed);\n"
          "}\n"
          "@kernel @cpu\n"
          "fn k_planet_point(x: f64, y: f64, z: f64, seed: i32) -> f32 {\n"
          "    return planet_height(vec3<f64>(x, y, z), seed);\n"
          "}\n";
    return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    bool write_golden = false;
    bool check_golden = false;
    fs::path golden_path = fs::path(BROKERNEL_SOURCE_DIR) / "bench" / "goldens.windows.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--write-golden") write_golden = true;
        else if (arg == "--check-golden") check_golden = true;
        else if (arg == "--golden" && i + 1 < argc) golden_path = argv[++i];
    }

    if (!brass::gpu::vulkan_available()) {
        std::cerr << "bench_hillclimb: Vulkan compute device unavailable\n";
        return 1;
    }

    auto suite_t0 = std::chrono::high_resolution_clock::now();

    CompileOptions copts;
    copts.opt_level = 3;
    auto noise_res = compile_source(kBkNoiseSuiteSource, "noise_hillclimb.bk", copts);
    if (!noise_res.ok) {
        std::cerr << "Noise compile error:\n" << noise_res.error_message() << "\n";
        return 1;
    }

    constexpr uint32_t N = kParityPoints;
    constexpr uint32_t NT = kGpuTimePoints;
    constexpr int kSeed = 1337;
    const GridShape shape1d(N);
    const GridShape shape_time(NT);

    // The positions: the first N are the parity set, all NT the GPU's timing set.
    std::vector<float> px(NT), py(NT), pz(NT), pw(NT);
    for (uint32_t i = 0; i < NT; ++i) {
        float t = static_cast<float>(i & 4095u) * 0.03125f - 32.0f + static_cast<float>(i >> 12) * 0.0009765625f;
        px[i] = t * 1.13f + 0.17f;
        py[i] = std::sin(t * 0.37f) * 14.3f - 2.1f;
        pz[i] = std::cos(t * 0.23f) * 11.7f + 3.4f;
        pw[i] = std::sin(t * 0.19f + 1.1f) * 9.5f - 1.3f;
    }

    // Device buffers for the GPU's timed runs (timestamp queries, inputs uploaded once).
    brass::gpu::VulkanBuffer d_px = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_py = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_pz = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_pw = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_out = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    d_px.upload(px.data(), NT * sizeof(float));
    d_py.upload(py.data(), NT * sizeof(float));
    d_pz.upload(pz.data(), NT * sizeof(float));
    d_pw.upload(pw.data(), NT * sizeof(float));

    // The GLSL baselines: the same FastNoise2 node, hand-written. A row whose
    // node has none here has no GPU ratio.
    std::vector<brass::Type> layout2 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    std::vector<brass::Type> layout3 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    std::vector<brass::Type> layout4 = {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()};
    auto glsl_module = [&](int dim, const std::string& expr, const std::string& tag) {
        auto spv = compile_glsl_compute(make_glsl_noise_shader(dim, expr), tag);
        return load_glsl_vulkan_module(spv, dim == 2 ? layout2 : dim == 3 ? layout3 : layout4);
    };
    auto vmod_val2 = glsl_module(2, "fn_value2d(p, seed)", "glsl_val2");
    auto vmod_val3 = glsl_module(3, "fn_value3d(p, seed)", "glsl_val3");
    auto vmod_val4 = glsl_module(4, "fn_value4d(p, seed)", "glsl_val4");
    auto vmod_perlin2 = glsl_module(2, "fn_perlin2d(p, seed)", "glsl_perlin2");
    auto vmod_perlin3 = glsl_module(3, "fn_perlin3d(p, seed)", "glsl_perlin3");
    auto vmod_perlin4 = glsl_module(4, "fn_perlin4d(p, seed)", "glsl_perlin4");
    auto vmod_fbm2 = glsl_module(2, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm2");
    auto vmod_fbm3 = glsl_module(3, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm3");
    auto vmod_fbm4 = glsl_module(4, "fn_fbm(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_fbm4");
    auto vmod_ridged2 = glsl_module(2, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged2");
    auto vmod_ridged3 = glsl_module(3, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged3");
    auto vmod_ridged4 = glsl_module(4, "fn_ridged(p, seed, 4, 2.0, 0.5, 0.25)", "glsl_ridged4");
    // One module per (node, dimension): expr's %D is the dimension.
    auto glsl_set = [&](const std::string& expr, const std::string& tag) {
        std::vector<brass::gpu::VulkanModule> mods;
        for (int dim = 2; dim <= 4; ++dim) {
            std::string e = expr;
            for (size_t at; (at = e.find("%D")) != std::string::npos;) e.replace(at, 2, std::to_string(dim));
            mods.push_back(glsl_module(dim, e, tag + std::to_string(dim)));
        }
        return mods;
    };
    auto vmod_simplex = glsl_set("fn_simplex%Dd(p, seed)", "glsl_simplex");
    auto vmod_supersimplex = glsl_set("fn_supersimplex%Dd(p, seed)", "glsl_supersimplex");
    auto vmod_white = glsl_set("fn_white%Dd(p, seed)", "glsl_white");
    auto vmod_pingpong = glsl_set("fn_ping_pong(fn_perlin(p, seed), 2.0)", "glsl_pingpong");
    auto vmod_cell_val = glsl_set("fn_cell_value%Dd(p, seed)", "glsl_cell_val");
    auto vmod_cell_dist = glsl_set("fn_cell_distance%Dd(p, seed)", "glsl_cell_dist");
    auto vmod_cell_lookup = glsl_set("fn_cell_lookup%Dd(p, seed)", "glsl_cell_lookup");
    auto vmod_warp_grad = glsl_set("fn_perlin(fn_warp_grad(p, seed, 1.5, 0.5), seed)", "glsl_warp_grad");
    auto vmod_warp_prog = glsl_set("fn_perlin(fn_warp_fractal(p, seed, 3, 2.0, 0.5, 0.25, 1.5, 0.5, true), seed)", "glsl_warp_prog");
    auto vmod_warp_indep = glsl_set("fn_perlin(fn_warp_fractal(p, seed, 3, 2.0, 0.5, 0.25, 1.5, 0.5, false), seed)", "glsl_warp_indep");

    std::vector<float> ref(N), cpu_out(N), gpu_out(N), glsl_out(N);
    std::vector<BenchRow> all_rows;

    auto bench_noise = [&](const std::string& tier, const std::string& name, int dim, const char* kname,
                           FastNoise::SmartNode<> fn, const brass::gpu::VulkanModule* baseline, float tol) {
        BenchRow row;
        row.tier = tier;
        row.name = name;
        row.dim = dim;
        row.points = N;

        const auto* k = noise_res.find_kernel(kname);

        double fn_ms = best_cpu_ms([&] {
            if (dim == 2) fn->GenPositionArray2D(ref.data(), N, px.data(), py.data(), 0.0f, 0.0f, kSeed);
            else if (dim == 3) fn->GenPositionArray3D(ref.data(), N, px.data(), py.data(), pz.data(), 0.0f, 0.0f, 0.0f, kSeed);
            else fn->GenPositionArray4D(ref.data(), N, px.data(), py.data(), pz.data(), pw.data(), 0.0f, 0.0f, 0.0f, 0.0f, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        std::vector<KernelArg> cargs, gargs;
        std::vector<GpuArg> vk_args;
        if (dim == 2) {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_out, kSeed};
        } else if (dim == 3) {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_pz, d_out, kSeed};
        } else {
            cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(cpu_out), KernelArg(kSeed)};
            gargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(pw), KernelArg::store(gpu_out), KernelArg(kSeed)};
            vk_args = {d_px, d_py, d_pz, d_pw, d_out, kSeed};
        }

        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);

        k->dispatch_gpu(shape1d, gargs);

        // The baseline counts only if it computes the node: its first N
        // points against FastNoise2. Not to the kernel's tolerance: the GLSL
        // is not precise, so the driver fuses multiply-adds, and Simplex's
        // small output bound turns that into ~1e-4. A wrong node is off by O(1).
        if (baseline) {
            std::vector<brass::gpu::VulkanArg> bargs = glsl_args(vk_args);
            bargs.emplace_back(static_cast<int32_t>(NT));
            double glsl_ms = time_glsl_gpu_ms(*baseline, NT / 64, 1, 1, 64, 1, 1, bargs);
            d_out.download(glsl_out.data(), N * sizeof(float));
            auto bs = compare_buffers(glsl_out, ref, std::max(tol, 1e-3f));
            if (bs.pass) {
                row.has_gpu_baseline = true;
                row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
            } else {
                std::cerr << name << ": the GLSL baseline does not match FastNoise2 (max err "
                          << bs.max_abs << "), no GPU ratio\n";
            }
        }
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_time, vk_args);
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref, tol);
        auto gs = compare_buffers(gpu_out, ref, tol);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        bool pass = row.cpu_pass && row.gpu_pass;
        row.cpu_ratio = pass ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.gpu_ratio = (pass && row.has_gpu_baseline) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        all_rows.push_back(row);
    };

    // 1. Value (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Value>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "Value_2D", 2, "k_value2d", fn, &vmod_val2, kTol);
        bench_noise("Noise", "Value_3D", 3, "k_value3d", fn, &vmod_val3, kTol);
        bench_noise("Noise", "Value_4D", 4, "k_value4d", fn, &vmod_val4, kTol);
    }
    // 2. Perlin (2D, 3D, 4D)
    auto fn_perlin = FastNoise::New<FastNoise::Perlin>();
    fn_perlin->SetScale(1.0f);
    bench_noise("Noise", "Perlin_2D", 2, "k_perlin2d", fn_perlin, &vmod_perlin2, kTol);
    bench_noise("Noise", "Perlin_3D", 3, "k_perlin3d", fn_perlin, &vmod_perlin3, kTol);
    bench_noise("Noise", "Perlin_4D", 4, "k_perlin4d", fn_perlin, &vmod_perlin4, kTol);

    // 3. Simplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "Simplex_2D", 2, "k_simplex2d", fn, &vmod_simplex[0], kTol);
        bench_noise("Noise", "Simplex_3D", 3, "k_simplex3d", fn, &vmod_simplex[1], kTol);
        bench_noise("Noise", "Simplex_4D", 4, "k_simplex4d", fn, &vmod_simplex[2], kTol);
    }
    // 4. SuperSimplex (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::SuperSimplex>();
        fn->SetScale(1.0f);
        bench_noise("Noise", "SuperSimplex_2D", 2, "k_supersimplex2d", fn, &vmod_supersimplex[0], kTol);
        bench_noise("Noise", "SuperSimplex_3D", 3, "k_supersimplex3d", fn, &vmod_supersimplex[1], kTol);
        bench_noise("Noise", "SuperSimplex_4D", 4, "k_supersimplex4d", fn, &vmod_supersimplex[2], kTol);
    }
    // 5. White (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::White>();
        bench_noise("Noise", "White_2D", 2, "k_white2d", fn, &vmod_white[0], kTol);
        bench_noise("Noise", "White_3D", 3, "k_white3d", fn, &vmod_white[1], kTol);
        bench_noise("Noise", "White_4D", 4, "k_white4d", fn, &vmod_white[2], kTol);
    }
    // 6. FractalFBm (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalFBm>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "FractalFBm_2D", 2, "k_fbm2d", fn, &vmod_fbm2, kTol);
        bench_noise("Noise", "FractalFBm_3D", 3, "k_fbm3d", fn, &vmod_fbm3, kTol);
        bench_noise("Noise", "FractalFBm_4D", 4, "k_fbm4d", fn, &vmod_fbm4, kTol);
    }
    // 7. FractalRidged (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::FractalRidged>();
        fn->SetSource(fn_perlin);
        fn->SetOctaveCount(4);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "FractalRidged_2D", 2, "k_ridged2d", fn, &vmod_ridged2, kTol);
        bench_noise("Noise", "FractalRidged_3D", 3, "k_ridged3d", fn, &vmod_ridged3, kTol);
        bench_noise("Noise", "FractalRidged_4D", 4, "k_ridged4d", fn, &vmod_ridged4, kTol);
    }
    // 8. PingPong (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::PingPong>();
        fn->SetSource(fn_perlin);
        fn->SetPingPongStrength(2.0f);
        bench_noise("Noise", "PingPong_2D", 2, "k_pingpong2d", fn, &vmod_pingpong[0], kTol);
        bench_noise("Noise", "PingPong_3D", 3, "k_pingpong3d", fn, &vmod_pingpong[1], kTol);
        bench_noise("Noise", "PingPong_4D", 4, "k_pingpong4d", fn, &vmod_pingpong[2], kTol);
    }
    // 9. CellularValue (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularValue>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularValue_2D", 2, "k_cell_val2d", fn, &vmod_cell_val[0], kTolRelaxed);
        bench_noise("Noise", "CellularValue_3D", 3, "k_cell_val3d", fn, &vmod_cell_val[1], kTolRelaxed);
        bench_noise("Noise", "CellularValue_4D", 4, "k_cell_val4d", fn, &vmod_cell_val[2], kTolRelaxed);
    }
    // 10. CellularDistance (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularDistance>();
        fn->SetScale(1.0f);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularDistance_2D", 2, "k_cell_dist2d", fn, &vmod_cell_dist[0], kTolRelaxed);
        bench_noise("Noise", "CellularDistance_3D", 3, "k_cell_dist3d", fn, &vmod_cell_dist[1], kTolRelaxed);
        bench_noise("Noise", "CellularDistance_4D", 4, "k_cell_dist4d", fn, &vmod_cell_dist[2], kTolRelaxed);
    }
    // 11. CellularLookup (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::CellularLookup>();
        fn->SetScale(1.0f);
        fn->SetLookup(fn_perlin);
        fn->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
        fn->SetGridJitter(1.0f);
        bench_noise("Noise", "CellularLookup_2D", 2, "k_cell_lookup2d", fn, &vmod_cell_lookup[0], kTolRelaxed);
        bench_noise("Noise", "CellularLookup_3D", 3, "k_cell_lookup3d", fn, &vmod_cell_lookup[1], kTolRelaxed);
        bench_noise("Noise", "CellularLookup_4D", 4, "k_cell_lookup4d", fn, &vmod_cell_lookup[2], kTolRelaxed);
    }
    // 12. DomainWarpGradient (2D, 3D, 4D)
    auto fn_warp_grad = FastNoise::New<FastNoise::DomainWarpGradient>();
    fn_warp_grad->SetSource(fn_perlin);
    fn_warp_grad->SetWarpAmplitude(1.5f);
    fn_warp_grad->SetScale(2.0f);
    bench_noise("Noise", "DomainWarpGrad_2D", 2, "k_warp_grad2d", fn_warp_grad, &vmod_warp_grad[0], kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_3D", 3, "k_warp_grad3d", fn_warp_grad, &vmod_warp_grad[1], kTolRelaxed);
    bench_noise("Noise", "DomainWarpGrad_4D", 4, "k_warp_grad4d", fn_warp_grad, &vmod_warp_grad[2], kTolRelaxed);

    // 13. DomainWarpFractalProgressive (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalProgressive>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpProg_2D", 2, "k_warp_prog2d", fn, &vmod_warp_prog[0], kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_3D", 3, "k_warp_prog3d", fn, &vmod_warp_prog[1], kTolRelaxed);
        bench_noise("Noise", "DomainWarpProg_4D", 4, "k_warp_prog4d", fn, &vmod_warp_prog[2], kTolRelaxed);
    }
    // 14. DomainWarpFractalIndependent (2D, 3D, 4D)
    {
        auto fn = FastNoise::New<FastNoise::DomainWarpFractalIndependent>();
        fn->SetSource(fn_warp_grad);
        fn->SetOctaveCount(3);
        fn->SetLacunarity(2.0f);
        fn->SetGain(0.5f);
        fn->SetWeightedStrength(0.25f);
        bench_noise("Noise", "DomainWarpIndep_2D", 2, "k_warp_indep2d", fn, &vmod_warp_indep[0], kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_3D", 3, "k_warp_indep3d", fn, &vmod_warp_indep[1], kTolRelaxed);
        bench_noise("Noise", "DomainWarpIndep_4D", 4, "k_warp_indep4d", fn, &vmod_warp_indep[2], kTolRelaxed);
    }

    // 15. Planetary double origin (6,371,000 m) Simplex 3D
    {
        BenchRow row;
        row.tier = "Noise";
        row.name = "Simplex_3D_f64_Planet";
        row.dim = 3;
        row.points = N;

        auto fn = FastNoise::New<FastNoise::Simplex>();
        fn->SetScale(1.0f);
        const double origin = 6371000.0;
        std::vector<float> px_zero(N, 0.0f);

        // Reference via GenSingle3D at (double(px[i]) + origin) for exact parity oracle
        for (uint32_t i = 0; i < N; ++i) {
            ref[i] = fn->GenSingle3D(double(px[i]) + origin, double(py[i]) + origin, double(pz[i]) + origin, kSeed);
        }
        // FastNoise2's own speed: GenPositionArray3D about the double origin.
        std::vector<float> fn_bench_out(N);
        double fn_ms = best_cpu_ms([&] {
            fn->GenPositionArray3D(fn_bench_out.data(), N, px.data(), py.data(), pz.data(), origin, origin, origin, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        const auto* k = noise_res.find_kernel("k_planet_f64_simplex3d");
        std::vector<KernelArg> cargs = {KernelArg::store(px), KernelArg::store(py), KernelArg::store(pz), KernelArg::store(cpu_out), KernelArg(origin), KernelArg(kSeed)};

        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);

        // f64 is CPU only (@cpu): no GPU column.
        auto cs = compare_buffers(cpu_out, ref, kTol);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = true;
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        all_rows.push_back(row);
    }

    // Tier 2: Stencil Kernels
    auto stencil_rows = run_stencil_tier();
    for (const auto& r : stencil_rows) all_rows.push_back(r);

    // Capstone: the planet's noise fields (kPlanetFields), summed, over an
    // 8 km patch of its surface: float offsets about a double origin on the
    // sphere, as the planet sampled them. Parity is in metres of height, and
    // FastNoise2 scales the offsets in float, so a wider patch is less exact
    // on its side (16 km out a float's step is a millimetre).
    {
        BenchRow row;
        row.tier = "Capstone";
        row.name = "PlanetFields_3D_f64";
        row.dim = 3;
        row.points = N;

        auto planet_res = compile_source(planet_fields_source(), "planet_fields.bk", copts);
        if (!planet_res.ok) {
            std::cerr << "Planet fields compile error:\n" << planet_res.error_message() << "\n";
            return 1;
        }
        const auto* k = planet_res.find_kernel("k_planet_fields");
        auto tree = planet_fields_tree();

        const double R = 6371000.0;
        const double ox = R * 0.36, oy = R * 0.48, oz = R * 0.8;
        std::vector<float> qx(NT), qy(NT), qz(NT);
        for (uint32_t i = 0; i < NT; ++i) {
            qx[i] = px[i] * 100.0f;
            qy[i] = py[i] * 100.0f;
            qz[i] = pz[i] * 100.0f;
        }
        double fn_ms = best_cpu_ms([&] {
            tree->GenPositionArray3D(ref.data(), N, qx.data(), qy.data(), qz.data(), ox, oy, oz, kSeed);
        });
        row.cpu_ref_mpts = mpts(N, fn_ms);

        std::vector<KernelArg> cargs = {KernelArg::store(qx), KernelArg::store(qy), KernelArg::store(qz), KernelArg::store(cpu_out),
                                        KernelArg(ox), KernelArg(oy), KernelArg(oz), KernelArg(kSeed)};
        double bk_cpu_ms = best_cpu_ms([&] { k->dispatch_cpu(shape1d, cargs); });
        row.cpu_bk_mpts = mpts(N, bk_cpu_ms);

        // Heights of up to ~1000 m (the bands' amplitudes), to a millimetre.
        // f64 is CPU only (@cpu): no GPU column.
        auto cs = compare_buffers(cpu_out, ref, 1e-3f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = true;
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        all_rows.push_back(row);

        // The same fields one position at a time, as physics, the camera and
        // placement ask for a height: k_planet_point's eval_f32 per call
        // against FastNoise2's GenSingle3D on the same tree. Its rate is the
        // call's whole cost, argument packing included.
        BenchRow prow;
        prow.tier = "Point";
        prow.name = "PlanetPoint_3D_f64";
        prow.dim = 3;
        constexpr uint32_t kPoints = 1u << 14;
        prow.points = kPoints;
        const auto* kp = planet_res.find_kernel("k_planet_point");
        std::vector<double> wx(kPoints), wy(kPoints), wz(kPoints);
        for (uint32_t i = 0; i < kPoints; ++i) {
            wx[i] = ox + double(qx[i]);
            wy[i] = oy + double(qy[i]);
            wz[i] = oz + double(qz[i]);
        }
        std::vector<float> pref(kPoints), pbk(kPoints);
        double pfn_ms = best_cpu_ms([&] {
            for (uint32_t i = 0; i < kPoints; ++i) pref[i] = tree->GenSingle3D(wx[i], wy[i], wz[i], kSeed);
        });
        double pbk_ms = best_cpu_ms([&] {
            for (uint32_t i = 0; i < kPoints; ++i) {
                pbk[i] = kp->eval_f32({KernelArg(wx[i]), KernelArg(wy[i]), KernelArg(wz[i]), KernelArg(kSeed)});
            }
        });
        prow.cpu_ref_mpts = mpts(kPoints, pfn_ms);
        prow.cpu_bk_mpts = mpts(kPoints, pbk_ms);
        prow.gpu_bk_mpts = row.gpu_bk_mpts;
        auto ps = compare_buffers(pbk, pref, 1e-3f);
        prow.cpu_pass = ps.pass; prow.cpu_max_abs = ps.max_abs; prow.cpu_max_ulp = ps.max_ulp;
        prow.gpu_pass = row.gpu_pass; prow.gpu_max_abs = row.gpu_max_abs; prow.gpu_max_ulp = row.gpu_max_ulp;
        prow.cpu_ratio = (prow.cpu_pass && prow.gpu_pass) ? (prow.cpu_bk_mpts / prow.cpu_ref_mpts) : 0.0;
        all_rows.push_back(prow);
    }

    auto suite_t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(suite_t1 - suite_t0).count();

    // Print unified table
    std::printf("\n============================================================================================================================\n");
    std::printf("%-9s | %-24s | %-6s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s\n",
                "Tier", "Node / Kernel", "Parity", "CPU MaxErr", "GPU MaxErr", "CPU Mpt/s", "GPU Mpt/s", "CPU/Ref", "GPU/GLSL");
    std::printf("----------------------------------------------------------------------------------------------------------------------------\n");

    int parity_passed = 0;
    int parity_total = static_cast<int>(all_rows.size());
    double log_sum_cpu = 0.0;
    double log_sum_gpu = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;

    for (const auto& r : all_rows) {
        bool both_pass = r.cpu_pass && r.gpu_pass;
        if (both_pass) {
            parity_passed++;
            if (r.cpu_ratio > 0.0) {
                log_sum_cpu += std::log(r.cpu_ratio);
                cpu_count++;
            }
            if (r.gpu_ratio > 0.0) {
                log_sum_gpu += std::log(r.gpu_ratio);
                gpu_count++;
            }
        }
        char gpu_ratio[16];
        if (r.has_gpu_baseline) std::snprintf(gpu_ratio, sizeof gpu_ratio, "%7.2fx", r.gpu_ratio);
        else std::snprintf(gpu_ratio, sizeof gpu_ratio, "%8s", "n/a");
        std::printf("%-9s | %-24s | %-6s | %10.2e | %10.2e | %10.2f | %10.2f | %7.2fx | %s\n",
                    r.tier.c_str(),
                    r.name.c_str(),
                    both_pass ? "PASS" : "FAIL",
                    r.cpu_max_abs,
                    r.gpu_max_abs,
                    r.cpu_bk_mpts,
                    r.gpu_bk_mpts,
                    r.cpu_ratio,
                    gpu_ratio);
    }

    double geomean_cpu = cpu_count > 0 ? std::exp(log_sum_cpu / cpu_count) : 0.0;
    double geomean_gpu = gpu_count > 0 ? std::exp(log_sum_gpu / gpu_count) : 0.0;

    // The GPU rows the code decides: a noise row whose GLSL baseline runs
    // under kComputeBound of White's rate in its dimension (White hashes its
    // inputs and nothing else, so it runs at the memory's rate).
    constexpr double kComputeBound = 0.6;
    double white_mpts[5] = {};
    for (const auto& r : all_rows) {
        if (r.name.rfind("White_", 0) == 0 && r.has_gpu_baseline) white_mpts[r.dim] = r.gpu_glsl_mpts;
    }
    double log_sum_compute = 0.0;
    int compute_count = 0;
    for (const auto& r : all_rows) {
        if (r.tier != "Noise" || r.gpu_ratio <= 0.0 || white_mpts[r.dim] <= 0.0) continue;
        if (r.gpu_glsl_mpts >= kComputeBound * white_mpts[r.dim]) continue;
        log_sum_compute += std::log(r.gpu_ratio);
        compute_count++;
    }
    double geomean_gpu_compute = compute_count > 0 ? std::exp(log_sum_compute / compute_count) : 0.0;
    const auto& capstone = *std::find_if(all_rows.begin(), all_rows.end(),
                                         [](const BenchRow& r) { return r.tier == "Capstone"; });

    std::printf("============================================================================================================================\n");
    std::printf("ROLL-UP SCORES:\n");
    std::printf("  1. Completeness / Parity : %d / %d (%5.1f%%) nodes & stencils pass CPU and GPU parity\n",
                parity_passed, parity_total, 100.0 * parity_passed / std::max(parity_total, 1));
    std::printf("  2. Speed (GeoMean CPU)   : %.3fx relative to FastNoise2 SIMD / C++ Reference\n", geomean_cpu);
    std::printf("  3. Speed (GeoMean GPU)   : %.3fx relative to glslc SPIR-V of the same node (%d rows with one)\n",
                geomean_gpu, gpu_count);
    std::printf("     of which compute-bound: %.3fx (%d noise rows under %.0f%% of White's memory-bound rate)\n",
                geomean_gpu_compute, compute_count, kComputeBound * 100);
    std::printf("  4. Capstone Planet Fields: %s (CPU: %.2f Mpt/s [%.2fx FastNoise2], GPU: %.2f Mpt/s, MaxErr: %.2e m)\n",
                (capstone.cpu_pass && capstone.gpu_pass) ? "PASS" : "FAIL",
                capstone.cpu_bk_mpts, capstone.cpu_ratio, capstone.gpu_bk_mpts,
                std::max(capstone.cpu_max_abs, capstone.gpu_max_abs));
    std::printf("  Total Harness Wall Time  : %.2f s\n", total_sec);
    std::printf("============================================================================================================================\n");

    if (write_golden) {
        write_golden_json(golden_path, all_rows, parity_passed, parity_total, geomean_cpu, geomean_gpu,
                          geomean_gpu_compute);
        std::cout << "Wrote golden ratchet file: " << golden_path.string() << "\n";
    }

    if (check_golden) {
        if (!fs::exists(golden_path)) {
            std::cerr << "Golden file missing: " << golden_path.string() << "\n";
            return 1;
        }
        if (parity_passed != parity_total) {
            std::cerr << "Parity regression: " << parity_passed << " / " << parity_total << "\n";
            return 1;
        }
        // The geomeans may not fall more than kNoise below the golden's: run
        // to run they move a few percent (CPU load, GPU clocks).
        constexpr double kNoise = 0.10;
        bool ok = true;
        auto check = [&](const char* key, double now) {
            double was = read_golden_number(golden_path, key);
            if (was <= 0.0) {
                std::cerr << "Golden file has no " << key << "\n";
                ok = false;
            } else if (now < was * (1.0 - kNoise)) {
                std::cerr << "Speed regression: " << key << " " << now << " < golden " << was << "\n";
                ok = false;
            }
        };
        check("geomean_cpu_vs_ref", geomean_cpu);
        check("geomean_gpu_vs_glsl", geomean_gpu);
        check("geomean_gpu_compute_vs_glsl", geomean_gpu_compute);
        if (!ok) return 1;
        std::cout << "Golden ratchet check PASSED (" << parity_passed << "/" << parity_total
                  << " parity, geomeans within " << int(kNoise * 100) << "% of the golden or above).\n";
    }

    return (parity_passed == parity_total) ? 0 : 1;
}

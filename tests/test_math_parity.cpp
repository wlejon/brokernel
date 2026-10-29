// The transcendentals (lowering_math.cpp) against the host's libm, on the CPU
// build and on the GPU build: each function over a sweep of its domain plus
// its edge values (0, +-inf, NaN, out of domain), in f32 and f64. Exits 77
// (skipped) after the CPU checks when there is no Vulkan device.

#include "brokernel/brokernel.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace brokernel;

namespace {

int g_failures = 0;
bool g_have_gpu = false;

struct Fn {
    const char* name;
    int arity;
    double lo, hi;        // sweep (log-spaced when geometric)
    bool geometric;
    double rel64, rel32;  // relative tolerance against libm
    double abs64, abs32;  // absolute tolerance (results near zero)
    std::function<double(double, double)> ref;
};

double inf() { return std::numeric_limits<double>::infinity(); }
double nan() { return std::numeric_limits<double>::quiet_NaN(); }

std::vector<double> sweep(const Fn& f, bool f64) {
    std::vector<double> xs;
    const int n = 509;
    for (int i = 0; i < n; ++i) {
        double t = i / double(n - 1);
        xs.push_back(f.geometric ? std::exp(std::log(f.lo) + t * (std::log(f.hi) - std::log(f.lo)))
                                 : f.lo + t * (f.hi - f.lo));
    }
    for (double e : {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, inf(), -inf(), nan()}) xs.push_back(e);
    if (!f64) {
        for (double& x : xs) x = static_cast<float>(x);
    }
    return xs;
}

template <typename T>
bool agrees(double got, double want, double rel, double abs) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want)) return got == want;
    // A finite result past the type's range is inf in the type.
    if (std::abs(want) > std::numeric_limits<T>::max()) return std::isinf(got) && (got > 0) == (want > 0);
    double d = std::abs(got - want);
    return d <= abs || d <= rel * std::abs(want);
}

template <typename T>
void check_fn(const Fn& f) {
    const bool f64 = sizeof(T) == 8;
    const char* tn = f64 ? "f64" : "f32";
    std::string src = std::string("@kernel @grid(1)\nfn k(xs: read_store<") + tn + ", 1>, " +
                      (f.arity == 2 ? std::string("ys: read_store<") + tn + ", 1>, " : std::string()) +
                      "out: write_store<" + tn + ", 1>) {\n    out.write(" + f.name + "(xs.read()" +
                      (f.arity == 2 ? ", ys.read()" : "") + "));\n}\n";
    auto res = compile_source(src, std::string(f.name) + ".bk");
    if (!res.ok) {
        std::cerr << f.name << "<" << tn << ">: compile failed\n" << res.error_message() << "\n";
        ++g_failures;
        return;
    }
    const CompiledKernel* k = res.default_kernel();

    std::vector<double> xd = sweep(f, f64);
    std::vector<double> yd;
    if (f.arity == 2) {
        // pow: exponents over [-8, 8]; atan2: x over the same sweep reversed.
        for (size_t i = 0; i < xd.size(); ++i) {
            double y = std::string(f.name) == "pow" ? -8.0 + 16.0 * ((i * 37) % xd.size()) / double(xd.size())
                                                    : xd[xd.size() - 1 - i];
            yd.push_back(f64 ? y : static_cast<float>(y));
        }
    }
    std::vector<T> xs, ys;
    for (double v : xd) xs.push_back(static_cast<T>(v));
    for (double v : yd) ys.push_back(static_cast<T>(v));
    const double rel = f64 ? f.rel64 : f.rel32;
    const double abs = f64 ? f.abs64 : f.abs32;

    auto run = [&](bool gpu, std::vector<T>& out) {
        std::vector<KernelArg> args = {KernelArg::store(xs)};
        if (f.arity == 2) args.push_back(KernelArg::store(ys));
        args.push_back(KernelArg::store(out));
        std::string err;
        bool ok = gpu ? k->dispatch_gpu(GridShape(uint32_t(xs.size())), args, nullptr, &err)
                      : k->dispatch_cpu(GridShape(uint32_t(xs.size())), args, nullptr, &err);
        if (!ok) {
            std::cerr << f.name << "<" << tn << "> " << (gpu ? "GPU" : "CPU") << " dispatch failed: " << err << "\n";
            ++g_failures;
        }
        return ok;
    };
    auto verify = [&](const char* target, const std::vector<T>& out) {
        int bad = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            double x = xs[i], y = f.arity == 2 ? double(ys[i]) : 0.0;
            double want = f.ref(x, y);
            if (!f64) want = static_cast<double>(static_cast<float>(want));
            if (!agrees<T>(out[i], want, rel, abs)) {
                if (bad++ < 4) {
                    std::fprintf(stderr, "%s<%s> %s: x=%.17g y=%.17g got %.17g want %.17g\n", f.name, tn, target, x, y,
                                 double(out[i]), want);
                }
            }
        }
        if (bad) {
            std::fprintf(stderr, "%s<%s> %s: %d of %zu disagree with libm\n", f.name, tn, target, bad, xs.size());
            ++g_failures;
        }
    };

    std::vector<T> cpu(xs.size(), T(-12345));
    if (run(false, cpu)) verify("CPU", cpu);
    if (g_have_gpu) {
        std::vector<T> gpu(xs.size(), T(-12345));
        if (run(true, gpu)) verify("GPU", gpu);
    }
}

} // namespace

int main() {
    g_have_gpu = brass::gpu::vulkan_available();
    const std::vector<Fn> fns = {
        {"sin", 1, -60.0, 60.0, false, 4e-16, 2e-7, 4e-15, 4e-6, [](double x, double) { return std::sin(x); }},
        {"cos", 1, -60.0, 60.0, false, 4e-16, 2e-7, 4e-15, 4e-6, [](double x, double) { return std::cos(x); }},
        {"tan", 1, -1.5, 1.5, false, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::tan(x); }},
        {"asin", 1, -1.0, 1.0, false, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::asin(x); }},
        {"acos", 1, -1.0, 1.0, false, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::acos(x); }},
        {"atan", 1, -40.0, 40.0, false, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::atan(x); }},
        {"sinh", 1, -20.0, 20.0, false, 1e-15, 6e-7, 1e-16, 1e-7, [](double x, double) { return std::sinh(x); }},
        {"cosh", 1, -20.0, 20.0, false, 1e-15, 6e-7, 1e-16, 1e-7, [](double x, double) { return std::cosh(x); }},
        {"tanh", 1, -12.0, 12.0, false, 1e-15, 6e-7, 1e-16, 1e-7, [](double x, double) { return std::tanh(x); }},
        {"exp", 1, -80.0, 80.0, false, 1e-15, 4e-7, 0.0, 0.0, [](double x, double) { return std::exp(x); }},
        {"exp2", 1, -120.0, 120.0, false, 1e-15, 4e-7, 0.0, 0.0, [](double x, double) { return std::exp2(x); }},
        {"log", 1, 1e-37, 1e37, true, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::log(x); }},
        {"log2", 1, 1e-37, 1e37, true, 1e-15, 4e-7, 1e-16, 1e-7, [](double x, double) { return std::log2(x); }},
        {"pow", 2, 1e-3, 1e3, true, 4e-14, 2e-5, 0.0, 0.0,
         // C's pow, except a negative base is NaN for every exponent.
         [](double x, double y) { return y == 0.0 || x == 1.0 ? 1.0 : x < 0.0 ? nan() : std::pow(x, y); }},
        {"atan2", 2, -30.0, 30.0, false, 1e-15, 4e-7, 1e-16, 1e-7,
         [](double y, double x) { return std::atan2(y, x); }},
    };
    for (const Fn& f : fns) {
        check_fn<float>(f);
        check_fn<double>(f);
    }
    if (g_failures) {
        std::cerr << "test_math_parity: " << g_failures << " failure(s)\n";
        return 1;
    }
    if (!g_have_gpu) {
        std::cout << "test_math_parity: CPU passed; no Vulkan device, GPU skipped\n";
        return 77;
    }
    std::cout << "test_math_parity: ALL PASSED (CPU + GPU)\n";
    return 0;
}

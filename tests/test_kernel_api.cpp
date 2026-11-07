// bro.kernel compiling off the calling thread (src/api/api_async.cpp), driven
// from C++ through bronze's embedding API: compile() with a wait pump runs on
// a worker and returns what it returns inline, its module requests answered
// by the JS resolver on this thread; compileAsync() promises settle at
// settleAsyncCompiles; a realm destroyed with a compile pending drops it.
// CPU only.
#include "api.h"
#include "embed/embed.h"
#include "embed/embed_object_builder.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ev = bronze::embed;
using bronze::Value;
using namespace brokernel::api;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

namespace {

std::atomic<int> g_pumps{0};
void countingPump() { ++g_pumps; }

// Enough code that a compile takes a while (tens of milliseconds).
std::string heavySource(int variant) {
    return "import std::noise::*;\n"
           "@kernel @grid(1) fn k(g: grid<1>, px: read_store<f32, 1>, out: write_store<f32, 1>, seed: i32) {\n"
           "    let p = vec2<f32>(px[g.x()], " + std::to_string(variant) + ".0);\n"
           "    out[g.x()] = perlin2d<f32>(p, seed) + cellular_distance2d<f32>(p, seed) + perlin3d<f32>(vec3<f32>(p, 0.5), seed);\n"
           "}\n"
           "@kernel fn f(x: f32) -> f32 { return x * " + std::to_string(variant) + ".0 + 1.0; }\n";
}

Value kernelObj() {
    auto bro = ev::globalValue("bro");
    CHECK(bro.found);
    return ev::getProperty(bro.value, "kernel");
}

ev::CallResult callMethod(Value obj, const char* name, std::vector<Value> args) {
    ev::Persistent o(obj);
    ev::Persistent fn(ev::getProperty(o.get(), name));
    CHECK(ev::isFunction(fn.get()));
    return ev::call(fn.get(), o.get(), args);
}

ev::CallResult compileJs(const char* fnName, const std::string& source, Value opts = ev::undefined()) {
    ev::Persistent src(ev::fromUtf8(source));
    ev::Persistent o(opts);
    ev::Persistent k(kernelObj());
    return callMethod(k.get(), fnName, {src.get(), o.get()});
}

// mod.kernel('f').eval(x)
double evalF(Value mod, double x) {
    ev::Persistent m(mod);
    ev::Persistent name(ev::fromUtf8("f"));
    auto k = callMethod(m.get(), "kernel", {name.get()});
    CHECK(!k.thrown);
    ev::Persistent kp(k.value);
    auto r = callMethod(kp.get(), "eval", {ev::fromDouble(x)});
    CHECK(!r.thrown && ev::isNumber(r.value));
    return ev::toDouble(r.value);
}

std::string messageOf(Value err) {
    const Value m = ev::getProperty(err, "message");
    return ev::isString(m) ? ev::toUtf8(m) : (ev::isString(err) ? ev::toUtf8(err) : std::string());
}

// The outcome of a promise, recorded by its reactions.
struct Outcome {
    int settled = 0;
    bool fulfilled = false;
    ev::Persistent value;
};

void observe(Value promise, Outcome& out) {
    ev::Persistent p(promise);
    ev::Persistent onOk(ev::makeFunction([&out](Value, std::span<const Value> a) -> Value {
        ++out.settled;
        out.fulfilled = true;
        out.value.set(a.empty() ? ev::undefined() : a[0]);
        return ev::undefined();
    }, 1));
    ev::Persistent onErr(ev::makeFunction([&out](Value, std::span<const Value> a) -> Value {
        ++out.settled;
        out.fulfilled = false;
        out.value.set(a.empty() ? ev::undefined() : a[0]);
        return ev::undefined();
    }, 1));
    auto r = callMethod(p.get(), "then", {onOk.get(), onErr.get()});
    CHECK(!r.thrown);
}

void test_sync_with_pump() {
    const std::string src = heavySource(3);
    setThreadWaitPump(nullptr, 0);
    auto inline_ = compileJs("compile", src);
    CHECK(!inline_.thrown);
    ev::Persistent inlineMod(inline_.value);

    setThreadWaitPump(countingPump, 1);
    g_pumps = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto off = compileJs("compile", heavySource(4));   // another source: a real compile, not a cache hit
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(!off.thrown);
    ev::Persistent offMod(off.value);
    std::cout << "  compile with a pump: " << ms << " ms, pump called " << g_pumps.load() << " times\n";
    CHECK(g_pumps > 0);
    CHECK(evalF(inlineMod.get(), 2.0) == 7.0);
    CHECK(evalF(offMod.get(), 2.0) == 9.0);

    // Same source off-thread and inline: the same kernels, the same results.
    auto again = compileJs("compile", src);
    CHECK(!again.thrown);
    ev::Persistent againMod(again.value);
    CHECK(evalF(againMod.get(), -1.5) == evalF(inlineMod.get(), -1.5));
    ev::Persistent kernelsA(ev::getProperty(againMod.get(), "kernels"));
    ev::Persistent kernelsB(ev::getProperty(inlineMod.get(), "kernels"));
    CHECK(ev::toDouble(ev::getProperty(kernelsA.get(), "length")) == ev::toDouble(ev::getProperty(kernelsB.get(), "length")));

    // withKnobs recompiles through the worker too.
    ev::Persistent knobMod(ev::fromUtf8("@knob const G: f32 = 2.0;\n@kernel fn f(x: f32) -> f32 { return x * G; }\n"));
    auto km = compileJs("compile", ev::toUtf8(knobMod.get()));
    CHECK(!km.thrown);
    ev::Persistent kmod(km.value);
    ev::ObjectBuilder knobs;
    knobs.set("G", 5.0);
    ev::Persistent kv(knobs.get());
    g_pumps = 0;
    auto tuned = callMethod(kmod.get(), "withKnobs", {kv.get()});
    CHECK(!tuned.thrown);
    ev::Persistent tunedMod(tuned.value);
    CHECK(evalF(tunedMod.get(), 3.0) == 15.0);

    // reload() compiles its modules on the worker: the kernel follows.
    ev::ObjectBuilder fileOpt;
    fileOpt.set("file", "reloaded.bk");
    ev::Persistent fo(fileOpt.get());
    auto rl = compileJs("compile", "@kernel fn f(x: f32) -> f32 { return x + 1.0; }\n", fo.get());
    CHECK(!rl.thrown);
    ev::Persistent rlMod(rl.value);
    CHECK(evalF(rlMod.get(), 1.0) == 2.0);
    ev::ObjectBuilder sources;
    sources.set("reloaded.bk", "@kernel fn f(x: f32) -> f32 { return x + 100.0; }\n");
    ev::Persistent sv(sources.get());
    ev::Persistent k(kernelObj());
    auto reloaded = callMethod(k.get(), "reload", {sv.get()});
    CHECK(!reloaded.thrown);
    CHECK(evalF(rlMod.get(), 1.0) == 101.0);

    // A compile error throws the same Error off-thread as inline.
    const std::string bad = "@kernel fn f(x: f32) -> f32 { return y; }\n";
    setThreadWaitPump(nullptr, 0);
    auto e1 = compileJs("compile", bad);
    setThreadWaitPump(countingPump, 1);
    auto e2 = compileJs("compile", bad);
    CHECK(e1.thrown && e2.thrown);
    ev::Persistent e1v(e1.value), e2v(e2.value);
    CHECK(messageOf(e1v.get()) == messageOf(e2v.get()));
    CHECK(messageOf(e2v.get()).find("bro.kernel.compile:") == 0);
    ev::Persistent diags(ev::getProperty(e2v.get(), "diagnostics"));
    CHECK(ev::toDouble(ev::getProperty(diags.get(), "length")) >= 1);
    std::cout << "  sync compile with a pump: ok\n";
}

// A @cpu kernel's CPU build begins on a thread of its own as its compile ends:
// the compile does not wait for it, and its first eval waits for it pumped.
void test_cpu_prebuild() {
    auto cpuSource = [](int variant) {
        return "import std::noise::*;\n"
               "@kernel @cpu fn f(x: f32) -> f32 {\n"
               "    let p = vec2<f32>(x, " + std::to_string(variant) + ".0);\n"
               "    return x * " + std::to_string(variant) + ".0 + 1.0 + 0.0 * (perlin2d<f32>(p, 1) + cellular_distance2d<f32>(p, 2)"
               " + perlin3d<f32>(vec3<f32>(p, 0.5), 3));\n"
               "}\n";
    };
    setThreadWaitPump(nullptr, 0);
    auto inline_ = compileJs("compile", cpuSource(5));
    CHECK(!inline_.thrown);
    ev::Persistent inlineMod(inline_.value);
    CHECK(evalF(inlineMod.get(), 2.0) == 11.0);

    setThreadWaitPump(countingPump, 1);
    for (int v = 6; v < 9; ++v) {
        auto off = compileJs("compile", cpuSource(v));
        CHECK(!off.thrown);
        ev::Persistent offMod(off.value);
        CHECK(evalF(offMod.get(), 2.0) == 2.0 * v + 1.0);
        CHECK(evalF(offMod.get(), -1.0) == -1.0 * v + 1.0);
    }
    // compileAsync: the module, then its kernel's first eval.
    Outcome out;
    {
        auto p = compileJs("compileAsync", cpuSource(9));
        CHECK(!p.thrown);
        ev::Persistent pv(p.value);
        observe(pv.get(), out);
    }
    settleAsyncCompiles(true);
    ev::drainMicrotasks();
    CHECK(out.settled == 1 && out.fulfilled);
    CHECK(evalF(out.value.get(), 3.0) == 28.0);

    // cpuReady(): a promise settled once the builds are made, a frame's
    // settle away; for a module whose builds were never begun (compiled with
    // no pump, no worker to begin them) it begins them.
    auto awaitReady = [](Value mod) {
        ev::Persistent m(mod);
        auto p = callMethod(m.get(), "cpuReady", {});
        CHECK(!p.thrown);
        Outcome ready;
        ev::Persistent pv(p.value);
        observe(pv.get(), ready);
        settleAsyncCompiles(true);
        ev::drainMicrotasks();
        CHECK(ready.settled == 1 && ready.fulfilled);
    };
    auto fresh = compileJs("compile", cpuSource(10));
    CHECK(!fresh.thrown);
    ev::Persistent freshMod(fresh.value);
    awaitReady(freshMod.get());
    CHECK(evalF(freshMod.get(), 2.0) == 21.0);
    setThreadWaitPump(nullptr, 0);
    auto unbegun = compileJs("compile", cpuSource(11));
    CHECK(!unbegun.thrown);
    ev::Persistent unbegunMod(unbegun.value);
    setThreadWaitPump(countingPump, 1);
    awaitReady(unbegunMod.get());
    CHECK(evalF(unbegunMod.get(), 2.0) == 23.0);
    awaitReady(unbegunMod.get());   // made already: settled at once
    setThreadWaitPump(nullptr, 0);
    std::cout << "  @cpu builds beside the compile: ok\n";
}

void test_resolver_channel() {
    setThreadWaitPump(countingPump, 1);
    const std::thread::id owner = std::this_thread::get_id();
    std::vector<std::string> asked;
    bool offThread = false;
    ev::Persistent resolver(ev::makeFunction([&](Value, std::span<const Value> a) -> Value {
        const std::string name = ev::toUtf8(a[0]);
        asked.push_back(name);
        offThread = offThread || std::this_thread::get_id() != owner;
        if (name == "chan_mod") return ev::fromUtf8("module chan_mod;\npub fn twice(x: f32) -> f32 { return x * 2.0; }\n");
        if (name == "chan_throw") return ev::throwError("resolver says no");
        return ev::undefined();
    }, 1));
    ev::Persistent k(kernelObj());
    CHECK(!callMethod(k.get(), "resolve", {resolver.get()}).thrown);

    auto r = compileJs("compile", "import chan_mod;\n@kernel fn f(x: f32) -> f32 { return chan_mod::twice(x) + 0.5; }\n");
    if (r.thrown) std::cerr << messageOf(r.value) << "\n";
    CHECK(!r.thrown);
    ev::Persistent mod(r.value);
    CHECK(evalF(mod.get(), 4.0) == 8.5);
    CHECK(!asked.empty() && asked[0] == "chan_mod" && !offThread);

    // A resolver that throws: the compile throws its exception.
    auto t = compileJs("compile", "import chan_throw;\n@kernel fn f(x: f32) -> f32 { return x; }\n");
    CHECK(t.thrown);
    ev::Persistent tv(t.value);
    CHECK(messageOf(tv.get()) == "resolver says no");

    // The same through compileAsync: rejected with it.
    auto p = compileJs("compileAsync", "import chan_throw;\n@kernel fn f(x: f32) -> f32 { return x; }\n");
    CHECK(!p.thrown && ev::isPromise(p.value));
    ev::Persistent pv(p.value);
    Outcome o;
    observe(pv.get(), o);
    settleAsyncCompiles(true);
    ev::drainMicrotasks();
    CHECK(o.settled == 1 && !o.fulfilled && messageOf(o.value.get()) == "resolver says no");

    CHECK(!callMethod(k.get(), "resolve", {ev::null()}).thrown);
    std::cout << "  resolver through the channel: ok\n";
}

void test_async() {
    setThreadWaitPump(countingPump, 1);
    constexpr int kJobs = 4;
    std::vector<ev::Persistent> promises;
    std::vector<Outcome> outcomes(kJobs + 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kJobs; ++i) {
        auto p = compileJs("compileAsync", heavySource(10 + i));
        CHECK(!p.thrown && ev::isPromise(p.value));
        promises.emplace_back(p.value);
    }
    auto bad = compileJs("compileAsync", "@kernel fn f(x: f32) -> f32 { return nope; }\n");
    CHECK(!bad.thrown);
    promises.emplace_back(bad.value);
    const double startMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int i = 0; i <= kJobs; ++i) observe(promises[i].get(), outcomes[i]);
    ev::drainMicrotasks();
    for (const auto& o : outcomes) CHECK(o.settled == 0);   // only settleAsyncCompiles settles them

    settleAsyncCompiles(true);
    const double allMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ev::drainMicrotasks();
    std::cout << "  compileAsync: " << kJobs + 1 << " started in " << startMs << " ms, all settled at " << allMs << " ms\n";
    for (int i = 0; i < kJobs; ++i) {
        CHECK(outcomes[i].settled == 1 && outcomes[i].fulfilled);
        CHECK(evalF(outcomes[i].value.get(), 1.0) == 10.0 + i + 1.0);
    }
    CHECK(outcomes[kJobs].settled == 1 && !outcomes[kJobs].fulfilled);
    ev::Persistent diags(ev::getProperty(outcomes[kJobs].value.get(), "diagnostics"));
    CHECK(ev::toDouble(ev::getProperty(diags.get(), "length")) >= 1);
    CHECK(messageOf(outcomes[kJobs].value.get()).find("bro.kernel.compile:") == 0);

    // settle(false): returns at once; called again until the job is in.
    auto p = compileJs("compileAsync", heavySource(30));
    ev::Persistent pv(p.value);
    Outcome o;
    observe(pv.get(), o);
    int calls = 0;
    while (o.settled == 0) {
        settleAsyncCompiles(false);
        ev::drainMicrotasks();
        ++calls;
        if (o.settled == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CHECK(calls < 10000);
    }
    CHECK(o.fulfilled && calls >= 1);

    // Without a pump: compiled at once, the promise already settled.
    setThreadWaitPump(nullptr, 0);
    auto q = compileJs("compileAsync", heavySource(31));
    ev::Persistent qv(q.value);
    Outcome oq;
    observe(qv.get(), oq);
    ev::drainMicrotasks();
    CHECK(oq.settled == 1 && oq.fulfilled);
    std::cout << "  compileAsync: ok\n";
}

void test_realm_destroy() {
    setThreadWaitPump(countingPump, 1);
    ev::Realm* realm = ev::createRealm();
    Outcome o;
    {
        ev::RealmScope scope(realm);
        installKernel();
        auto p = compileJs("compileAsync", heavySource(40));
        CHECK(!p.thrown);
        ev::Persistent pv(p.value);
        observe(pv.get(), o);
    }
    ev::destroyRealm(realm);
    settleAsyncCompiles(true);   // joins the dropped job's worker, settles nothing
    ev::drainMicrotasks();
    CHECK(o.settled == 0);

    // A pending job at shutdown is dropped and its worker joined.
    auto p = compileJs("compileAsync", heavySource(41));
    ev::Persistent pv(p.value);
    Outcome o2;
    observe(pv.get(), o2);
    shutdownKernelAsync();
    settleAsyncCompiles(true);
    ev::drainMicrotasks();
    CHECK(o2.settled == 0);
    std::cout << "  realm destroyed with a pending compile: ok\n";
}

} // namespace

int main() {
    installKernel();
    test_sync_with_pump();
    test_cpu_prebuild();
    test_resolver_channel();
    test_async();
    test_realm_destroy();
    setThreadWaitPump(nullptr, 0);
    std::cout << "test_kernel_api: all passed\n";
    return 0;
}

// bro.kernel compiling off the calling thread (src/api/api_async.cpp), driven
// from C++ through bronze's embedding API: compile() with a wait pump runs on
// a worker and returns what it returns inline, its module requests answered
// by the JS resolver on this thread; compileAsync() promises settle at
// settleAsyncCompiles; a realm destroyed with a compile pending drops it.
// CPU only.
#include "api.h"
#include "brokernel/brokernel.h"
#include "embed/embed.h"
#include "embed/embed_object_builder.h"
#include "embed/embed_typed_array.h"
#include "eval/eval.h"

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

void test_diag_constants_and_enums() {
    setThreadWaitPump(nullptr, 0);
    ev::Persistent kobj(kernelObj());

    // 1. registerConstants: numbers, booleans, vectors, arrays, [type, val] tuples, and __enum
    ev::ObjectBuilder decls;
    decls.set("SCALE", 2);
    decls.set("FLAG", true);
    ev::ArrayBuilder vecArr(3);
    vecArr.set(0, 1.0);
    vecArr.set(1, 2.5);
    vecArr.set(2, -3.0);
    decls.set("OFFSET", vecArr.get());

    ev::ArrayBuilder tupI32(2);
    tupI32.set(0, "i32");
    tupI32.set(1, 64);
    decls.set("COUNT", tupI32.get());

    ev::ArrayBuilder tupU32(2);
    tupU32.set(0, "u32");
    tupU32.set(1, 7);
    decls.set("MASK", tupU32.get());

    ev::ArrayBuilder vecList(2);
    ev::ArrayBuilder v0(4);
    v0.set(0, 1.0); v0.set(1, 0.0); v0.set(2, 0.0); v0.set(3, 1.0);
    ev::ArrayBuilder v1(4);
    v1.set(0, 0.0); v1.set(1, 1.0); v1.set(2, 0.0); v1.set(3, 1.0);
    vecList.set(0, v0.get());
    vecList.set(1, v1.get());
    ev::ArrayBuilder tupVecArr(2);
    tupVecArr.set(0, "vec4<f32>");
    tupVecArr.set(1, vecList.get());
    decls.set("BASIS", tupVecArr.get());

    ev::ObjectBuilder enumDecl;
    enumDecl.set("__enum", "u32");
    enumDecl.set("Grass", 1);
    enumDecl.set("Rock", 2);
    decls.set("Biome", enumDecl.get());

    ev::Persistent modName(ev::fromUtf8("test::consts"));
    ev::Persistent declsVal(decls.get());
    auto regRes = callMethod(kobj.get(), "registerConstants", {modName.get(), declsVal.get()});
    CHECK(!regRes.thrown && ev::isString(regRes.value) && ev::toUtf8(regRes.value) == "test::consts");

    const std::string src = R"(
        import test::consts::*;
        pub enum Mode : i32 {
            Off = 0,
            On = 3,
        }
        @kernel @grid(2)
        fn k_diag(g: grid<2>, inp: read_store<f32, 2>, out: write_store<f32, 2>) -> f32 {
            let x = inp[g.x(), g.y()];
            check("nonneg", x >= 0.0);
            let k = probe_if("big", x > 5.0, vec2<f32>(x, f32(g.x())));
            let b: u32 = u32(Biome::Rock);
            let res = k.x * SCALE + OFFSET.y + BASIS[1].y + f32(COUNT) * 0.0 + f32(b) * 0.0;
            out[g.x(), g.y()] = res;
            return res;
        }
        @kernel fn k_pt(x: f32) -> f32 {
            check("ok", x > 0.0);
            return 1.0 / x;
        }
    )";

    ev::ObjectBuilder copts;
    copts.set("enableChecks", true);
    copts.set("trapNonFinite", true);
    ev::Persistent coptsVal(copts.get());
    auto compRes = compileJs("compile", src, coptsVal.get());
    if (compRes.thrown) std::cerr << messageOf(compRes.value) << "\n";
    CHECK(!compRes.thrown);
    ev::Persistent mod(compRes.value);

    // 2. mod.enums reflection
    ev::Persistent enumsObj(ev::getProperty(mod.get(), "enums"));
    CHECK(ev::isObject(enumsObj.get()));
    ev::Persistent modeObj(ev::getProperty(enumsObj.get(), "Mode"));
    CHECK(ev::isObject(modeObj.get()));
    CHECK(ev::toDouble(ev::getProperty(modeObj.get(), "Off")) == 0.0);
    CHECK(ev::toDouble(ev::getProperty(modeObj.get(), "On")) == 3.0);

    // 3. kernel.check & kernel.probeIf on 2D grid kernel
    ev::Persistent kName(ev::fromUtf8("k_diag"));
    auto kRes = callMethod(mod.get(), "kernel", {kName.get()});
    CHECK(!kRes.thrown);
    ev::Persistent kern(kRes.value);

    constexpr uint32_t W = 4, H = 3;
    ev::Persistent inp(ev::createTypedArray(ev::elements::Float32, W * H));
    ev::Persistent out(ev::createTypedArray(ev::elements::Float32, W * H));
    {
        auto info = ev::typedArrayInfo(inp.get());
        auto* fp = reinterpret_cast<float*>(info.data);
        for (uint32_t i = 0; i < W * H; ++i) fp[i] = 1.0f;
        fp[1 * W + 2] = 8.0f;
        fp[2 * W + 3] = -2.0f;
    }

    ev::ArrayBuilder grid(2);
    grid.set(0, static_cast<int32_t>(W));
    grid.set(1, static_cast<int32_t>(H));
    ev::Persistent gridVal(grid.get());

    auto chk = callMethod(kern.get(), "check", {gridVal.get(), inp.get(), out.get()});
    CHECK(!chk.thrown && ev::isObject(chk.value));
    ev::Persistent chkObj(chk.value);
    CHECK(!ev::toBool(ev::getProperty(chkObj.get(), "ok")));
    CHECK(ev::toUtf8(ev::getProperty(chkObj.get(), "kind")) == "check");
    CHECK(ev::toUtf8(ev::getProperty(chkObj.get(), "label")) == "nonneg");
    ev::Persistent coord(ev::getProperty(chkObj.get(), "coord"));
    CHECK(ev::toDouble(ev::getElement(coord.get(), 0)) == 3.0);
    CHECK(ev::toDouble(ev::getElement(coord.get(), 1)) == 2.0);
    CHECK(ev::toDouble(ev::getElement(coord.get(), 2)) == 0.0);

    ev::ObjectBuilder pifOpts;
    pifOpts.set("maxSamples", 8);
    ev::Persistent pifOptsVal(pifOpts.get());
    auto pif = callMethod(kern.get(), "probeIf", {gridVal.get(), inp.get(), out.get(), pifOptsVal.get()});
    CHECK(!pif.thrown && ev::isObject(pif.value));
    ev::Persistent pifArr(pif.value);
    CHECK(ev::toDouble(ev::getProperty(pifArr.get(), "length")) == 1.0);
    ev::Persistent s0(ev::getElement(pifArr.get(), 0));
    CHECK(ev::toUtf8(ev::getProperty(s0.get(), "label")) == "big");
    ev::Persistent s0Coord(ev::getProperty(s0.get(), "coord"));
    CHECK(ev::toDouble(ev::getElement(s0Coord.get(), 0)) == 2.0);
    CHECK(ev::toDouble(ev::getElement(s0Coord.get(), 1)) == 1.0);
    ev::Persistent s0Val(ev::getProperty(s0.get(), "value"));
    CHECK(ev::toDouble(ev::getElement(s0Val.get(), 0)) == 8.0);
    CHECK(ev::toDouble(ev::getElement(s0Val.get(), 1)) == 2.0);

    // 4. point kernel check (trapNonFinite on 1.0 / 0.0)
    ev::Persistent ptName(ev::fromUtf8("k_pt"));
    auto ptRes = callMethod(mod.get(), "kernel", {ptName.get()});
    CHECK(!ptRes.thrown);
    ev::Persistent ptKern(ptRes.value);
    auto ptChk = callMethod(ptKern.get(), "check", {ev::fromDouble(0.0)});
    CHECK(!ptChk.thrown && ev::isObject(ptChk.value));
    ev::Persistent ptChkObj(ptChk.value);
    CHECK(!ev::toBool(ev::getProperty(ptChkObj.get(), "ok")));
    CHECK(ev::toUtf8(ev::getProperty(ptChkObj.get(), "kind")) == "check");

    ev::ObjectBuilder noCheckNf;
    noCheckNf.set("trapNonFinite", true);
    ev::Persistent noCheckNfVal(noCheckNf.get());
    auto ptInf = callMethod(ptKern.get(), "check", {ev::fromDouble(0.0), noCheckNfVal.get()});
    CHECK(!ptInf.thrown);

    // 5. compareCpuGpu (works with or without Vulkan, checking fields)
    ev::ObjectBuilder cmpOpts;
    cmpOpts.set("relTol", 1e-4);
    cmpOpts.set("absTol", 1e-5);
    ev::Persistent cmpOptsVal(cmpOpts.get());
    auto cmp = callMethod(kern.get(), "compareCpuGpu", {gridVal.get(), inp.get(), out.get(), cmpOptsVal.get()});
    CHECK(!cmp.thrown && ev::isObject(cmp.value));
    ev::Persistent cmpObj(cmp.value);
    ev::Persistent wc(ev::getProperty(cmpObj.get(), "worstCoord"));
    CHECK(ev::toDouble(ev::getProperty(wc.get(), "length")) == 3.0);
    if (ev::toBool(ev::getProperty(cmpObj.get(), "ranGpu"))) {
        CHECK(ev::toBool(ev::getProperty(cmpObj.get(), "ok")));
    }

    std::cout << "  diag, constants, enums, check, probeIf, compareCpuGpu: ok\n";
}

struct MockReadback final : public GpuReadback {
    std::vector<uint8_t> payload;
    mutable int polls = 0;
    bool ready() const override { return ++polls >= 2; }
    const void* data() const override { return payload.data(); }
    size_t bytes() const override { return payload.size(); }
};

struct MockGpuBuffer final : public GpuBuffer {
    std::vector<uint8_t> data;
    explicit MockGpuBuffer(size_t n) : data(std::max<size_t>(n, 4), 0), len_(n) {}
    size_t len_ = 0;
    uint64_t address() const override { return reinterpret_cast<uint64_t>(data.data()); }
    size_t bytes() const override { return len_; }
    size_t capacity() const override { return data.size(); }
    bool write(size_t off, const void* src, size_t n, std::string* err) override {
        if (off + n > len_) { if (err) *err = "oob"; return false; }
        if (n > 0) std::memcpy(data.data() + off, src, n);
        return true;
    }
    bool read(size_t off, void* dst, size_t n, std::string* err) override {
        if (off + n > len_) { if (err) *err = "oob"; return false; }
        if (n > 0) std::memcpy(dst, data.data() + off, n);
        return true;
    }
    std::shared_ptr<GpuReadback> readLater(size_t off, size_t n, std::string* err) override {
        if (off + n > len_) { if (err) *err = "oob"; return nullptr; }
        auto rb = std::make_shared<MockReadback>();
        rb->payload.assign(data.begin() + static_cast<std::ptrdiff_t>(off),
                           data.begin() + static_cast<std::ptrdiff_t>(off + n));
        return rb;
    }
};

struct MockGpuDevice final : public GpuDevice {
    int begin_calls = 0;
    int end_calls = 0;
    bool available() const override { return true; }
    std::shared_ptr<GpuBuffer> createBuffer(size_t bytes, std::string*) override {
        return std::make_shared<MockGpuBuffer>(bytes);
    }
    std::shared_ptr<GpuProgram> createProgram(const brokernel::CompiledKernel&, std::string* err) override {
        if (err) *err = "mock";
        return nullptr;
    }
    bool dispatch(GpuProgram&, const brokernel::CompiledKernel&, const brokernel::GpuLaunch&, std::string* err) override {
        if (err) *err = "mock";
        return false;
    }
    void beginBatch() override { ++begin_calls; }
    bool endBatch(std::string*) override { ++end_calls; return true; }
};

void test_cross_lang_features() {
    setThreadWaitPump(nullptr, 0);
    auto r1 = bronze::eval::evalScript(R"(
        // Negative parse guard: malformed struct array syntax must fail cleanly without hanging.
        let badThrew = false;
        try {
            bro.kernel.compile('struct BadArr { x: [f32, 2], } @kernel fn k() -> f32 { return 1.0; }');
        } catch (e) {
            badThrew = true;
        }
        if (!badThrew) throw new Error('expected compile error for [f32, 2]');

        bro.kernel.registerModule('cross_helper', 'module cross_helper;\npub fn bias() -> f64 { return 1.0; }');
        const src = [
            "import cross_helper;",
            "struct NestedCfg { offset: vec2<f32>, anchor: df32, gains: [f32; 2], }",
            "struct SampleCell { weight: f32, height: df32, tag: i32, }",
            "@kernel @cpu fn eval_cfg(cfg: NestedCfg, x: f64) -> f64 {",
            "    let d = df32(cfg.offset.x) + cfg.anchor;",
            "    return f64(d) * f64(cfg.gains[0]) + f64(cfg.offset.y) * f64(cfg.gains[1]) + x;",
            "}",
            "@kernel fn df_diff(a: df32) -> df32 {",
            "    return (df32(16384.0) + a) - df32(16384.0);",
            "}",
            "@kernel @cpu fn bump_zero_ret(out: rw_store<i32, 1>, idx: i32, addv: i32) -> f64 {",
            "    out[idx] = out[idx] + addv;",
            "    return 0.0;",
            "}",
            "@kernel @grid(1) fn bump_grid(g: grid<1>, out: rw_store<i32, 1>, addv: i32) {",
            "    out[g.x()] = out[g.x()] + addv;",
            "}",
            "@kernel @cpu fn store_accum(arr: read_store<f32, 1>, scale: f64, n: i32) -> f64 {",
            "    var acc: f64 = 0.0;",
            "    for i in 0..n {",
            "        acc = acc + f64(arr[i]) * scale;",
            "    }",
            "    return acc;",
            "}",
            "pub fn add_mul(a: f32, b: f32, c: f32) -> f32 {",
            "    return (a + b) * c;",
            "}"
        ].join('\n');
        const mod = bro.kernel.compile(src, 'cross_lang_mod');

        // 1. Plain JS object struct param + nested vector + df32 + fixed array
        const cfgOut = mod.kernel('eval_cfg').eval({
            offset: { x: 2.0, y: 3.0 },
            anchor: 16384.0001220703125,
            gains: [2.0, 5.0],
        }, 1.0);
        if (Math.abs(cfgOut - ((2.0 + 16384.0001220703125) * 2.0 + 15.0 + 1.0)) > 1e-6) {
            throw new Error('plain struct eval mismatch: ' + cfgOut);
        }

        // 2. df32 + packDF32 / unpackDF32
        const dfVal = mod.kernel('df_diff').eval(0.062623456789);
        if (Math.abs(dfVal - 0.062623456789) > 1e-10) {
            throw new Error('df_diff mismatch: ' + dfVal);
        }
        const packed = bro.kernel.packDF32(16384.0 + 0.062623456789);
        const unpacked = bro.kernel.unpackDF32(packed);
        if (Math.abs(unpacked[0] - (16384.0 + 0.062623456789)) > 1e-10) {
            throw new Error('packDF32/unpackDF32 mismatch: ' + unpacked[0]);
        }

        // 3. Kernel.bind() & KernelFn.bind(), ensuring +0.0 return never double-runs side effects
        const hBuf = bro.kernel.createBuffer(16, { residency: 'host' });
        const iView = new Int32Array(hBuf.asTypedArray(Int32Array).buffer, 0, 4);
        iView[0] = 0; iView[1] = 10;
        const kBumpFn = mod.fn('bump_zero_ret').bind(hBuf, undefined, 7);
        const zRet = kBumpFn(1);
        if (zRet !== 0.0 || iView[1] !== 17) {
            throw new Error('fn.bind zero-return side-effect check failed: zRet=' + zRet + ' v=' + iView[1]);
        }
        const kBumpGrid = mod.kernel('bump_grid').bind(undefined, hBuf, 3);
        kBumpGrid(2);
        if (iView[0] !== 3 || iView[1] !== 20) {
            throw new Error('kernel.bind grid check failed: v0=' + iView[0] + ' v1=' + iView[1]);
        }
        const fBound = mod.fn('add_mul').bind(undefined, 4.0, undefined);
        if (Math.abs(fBound(2.0, 3.0) - 18.0) > 1e-5) {
            throw new Error('KernelFn.bind mismatch: ' + fBound(2.0, 3.0));
        }

        // Non-grid kernel with 1D read_store parameter must support direct .eval() and .bind() fast path
        const sampleArr = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        const kStoreEval = mod.kernel('store_accum');
        if (Math.abs(kStoreEval.eval(sampleArr, 2.0, 4) - 20.0) > 1e-6) {
            throw new Error('store_accum.eval mismatch');
        }
        const kStoreBound = kStoreEval.bind(sampleArr, undefined, 4);
        if (Math.abs(kStoreBound(3.0) - 30.0) > 1e-6) {
            throw new Error('store_accum.bind mismatch');
        }

        // 4. exportNative with store + scalar + df32, and StructView.exportFieldNative
        mod.kernel('store_accum').exportNative('bro.test.storeAccum');
        mod.kernel('df_diff').exportNative('bro.test.dfDiff');

        const sv = mod.structView('SampleCell', bro.kernel.createBuffer(mod.structLayout('SampleCell').byteSize * 3, { residency: 'host' }));
        sv.setStruct(0, { weight: 1.5, height: 16384.250123456, tag: 101 });
        sv.setStruct(1, { weight: 3.5, height: -42.125, tag: -7 });
        sv.exportFieldNative('height', 'bro.test.cellHeight');
        sv.exportFieldNative('tag', 'bro.test.cellTag');
        sv.exportFieldNative('weight', 'bro.test.cellWeightF32', 'f32[]');
        globalThis.__testView = sv;
    )", bronze::eval::EvalOptions{.filename = "cross_lang_setup.js"});
    if (r1.thrown) {
        std::cerr << "setupJs threw: " << ev::uncaughtReport(r1.value) << "\n";
        CHECK(false);
    }

    // Separate script so Bronze compiler sees registered MIR natives and fuses calls in loops.
    const std::string fusedJs = R"(
        const arr = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        let sum = 0.0;
        for (let i = 0; i < 4; ++i) {
            sum += bro.test.storeAccum(arr, 2.0, i + 1);
        }
        // i=0:2, i=1:6, i=2:12, i=3:20 -> total 40
        if (Math.abs(sum - 40.0) > 1e-6) {
            throw new Error('fused storeAccum loop mismatch: ' + sum);
        }
        const dd = bro.test.dfDiff(0.062623456789);
        if (Math.abs(dd - 0.062623456789) > 1e-10) {
            throw new Error('fused dfDiff mismatch: ' + dd);
        }

        const sv = globalThis.__testView;
        const h0 = bro.test.cellHeight(sv.bytes, 0);
        const t1 = bro.test.cellTag(sv.bytes, 1);
        const hOob = bro.test.cellHeight(sv.bytes, 99);
        const hNeg = bro.test.cellHeight(sv.bytes, -1);
        const f32Bytes = new Float32Array(sv.bytes.buffer, sv.bytes.byteOffset, sv.bytes.byteLength / 4);
        const w1 = bro.test.cellWeightF32(f32Bytes, 1);
        if (Math.abs(h0 - 16384.250123456) > 1e-6 || t1 !== -7 || hOob !== 0 || hNeg !== 0 || Math.abs(w1 - 3.5) > 1e-5) {
            throw new Error('StructView.exportFieldNative mismatch: ' + [h0, t1, hOob, hNeg, w1].join(', '));
        }
        globalThis.__testFusedFn = function() { return bro.test.storeAccum(arr, 2.0, 4); };
        if (Math.abs(globalThis.__testFusedFn() - 20.0) > 1e-6) throw new Error('pre-reload call failed');
    )";

    auto r2 = bronze::eval::evalScript(fusedJs, bronze::eval::EvalOptions{.filename = "cross_lang_fused.js"});
    if (r2.thrown) {
        std::cerr << "fusedJs threw: " << ev::uncaughtReport(r2.value) << "\n";
        CHECK(false);
    }

    // Reload bumps epoch -> fused function falls back cleanly to C trampoline,
    // and test Host readAsync + mocked GpuDevice batch() & readAsync().
    auto mockDev = std::make_shared<MockGpuDevice>();
    setGpuDevice(mockDev);

    const std::string batchAsyncJs = R"(
        // Trigger bro.kernel.reload epoch bump and invoke previously fused function
        const rep = bro.kernel.reload({ cross_helper: 'pub fn bias() -> f64 { return 2.0; }' });
        if (!Array.isArray(rep)) throw new Error('reload did not return array');
        if (Math.abs(globalThis.__testFusedFn() - 20.0) > 1e-6) {
            throw new Error('post-reload fallback failed');
        }

        // Nested bro.kernel.batch
        const bRet = bro.kernel.batch(() => {
            return bro.kernel.batch(() => 123);
        });
        if (bRet !== 123) throw new Error('batch return mismatch: ' + bRet);

        // GPU buffer write + readAsync
        const gbuf = bro.kernel.createBuffer(16, { residency: 'gpu' });
        gbuf.write(new Float32Array([10.5, 20.25, -3.5, 42.0]));
        globalThis.__readAsyncDone = false;
        gbuf.readAsync(Float32Array).then((arr) => {
            if (arr.length !== 4 || arr[0] !== 10.5 || arr[1] !== 20.25 || arr[2] !== -3.5 || arr[3] !== 42.0) {
                throw new Error('readAsync GPU mismatch: ' + Array.from(arr));
            }
            globalThis.__readAsyncDone = true;
        });

        // Host buffer readAsync
        const hb = bro.kernel.createBuffer(8, { residency: 'host' });
        hb.write(new Float32Array([7.0, 9.0]));
        globalThis.__hostReadDone = false;
        hb.readAsync(Float32Array).then((arr) => {
            if (arr.length !== 2 || arr[0] !== 7.0 || arr[1] !== 9.0) {
                throw new Error('readAsync Host mismatch');
            }
            globalThis.__hostReadDone = true;
        });
    )";

    auto r3 = bronze::eval::evalScript(batchAsyncJs, bronze::eval::EvalOptions{.filename = "cross_lang_async.js"});
    if (r3.thrown) {
        std::cerr << "batchAsyncJs threw: " << ev::uncaughtReport(r3.value) << "\n";
        CHECK(false);
    }

    CHECK(mockDev->begin_calls == 1 && mockDev->end_calls == 1);
    for (int i = 0; i < 4; ++i) {
        settleAsyncCompiles(false);
        ev::drainMicrotasks();
    }

    auto gt = ev::globalValue("globalThis");
    CHECK(gt.found);
    CHECK(ev::toBool(ev::getProperty(gt.value, "__readAsyncDone")));
    CHECK(ev::toBool(ev::getProperty(gt.value, "__hostReadDone")));
    setGpuDevice(nullptr);

    std::cout << "  cross-lang (plain struct, df32, bind, exportNative MIR, StructView, batch, readAsync): ok\n";
}

} // namespace

int main() {
    installKernel();
    test_sync_with_pump();
    test_cpu_prebuild();
    test_resolver_channel();
    test_async();
    test_realm_destroy();
    test_diag_constants_and_enums();
    test_cross_lang_features();
    setThreadWaitPump(nullptr, 0);
    std::cout << "test_kernel_api: all passed\n";
    return 0;
}

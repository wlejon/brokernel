// Compiling off the page thread: compile() waiting on a worker while the
// host's wait pump keeps its window answering, and compileAsync() promises
// settled by the host once a frame (api.h).
//
// A worker compiles and touches nothing else: no JS value, no realm, no
// thread-local of the API. Its module requests (the compiler's resolver)
// read the global module tables itself; a name they do not hold, when the
// owning thread has a JS resolver, goes to the owner over its Channel and
// waits for the answer, which the owner gives while it waits or settles.
#include "api_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iterator>
#include <mutex>
#include <thread>

namespace brokernel::api {

namespace {

thread_local WaitPump t_pump = nullptr;
thread_local int t_pumpInterval = 16;

// One per owning thread: the lock and the wakeup it shares with its workers.
// `events` counts what a worker signalled (a request, its end).
struct Wake {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t events = 0;
};
thread_local std::shared_ptr<Wake> t_wake;

std::shared_ptr<Wake> ownerWake() {
    if (!t_wake) t_wake = std::make_shared<Wake>();
    return t_wake;
}

void signal(Wake& w) {   // wake->mu held
    ++w.events;
    w.cv.notify_all();
}

// A worker and the thread it answers to; every field under wake->mu.
struct Channel {
    std::shared_ptr<Wake> wake;
    bool askOwner = false;   // the owner had a JS resolver when the work began
    bool requested = false;
    bool replied = false;
    bool closed = false;     // the owner answers nothing more
    bool done = false;
    std::string name;
    std::optional<std::string> reply;
};

// The channel of the work this thread is doing for its owner (a worker's).
thread_local Channel* t_channel = nullptr;

// The owner's side of one worker.
struct Waiter {
    std::shared_ptr<Channel> ch;
    std::thread thread;
    bool threw = false;          // the JS resolver threw: `thrown`
    ev::Persistent thrown;

    bool done() const {
        std::lock_guard<std::mutex> lock(ch->wake->mu);
        return ch->done;
    }
    void close() const {
        std::lock_guard<std::mutex> lock(ch->wake->mu);
        ch->closed = true;
        signal(*ch->wake);
    }
};

// `work` (which must not throw) on a new worker answering to this thread.
void startWorker(Waiter& w, std::function<void()> work) {
    w.ch = std::make_shared<Channel>();
    w.ch->wake = ownerWake();
    w.ch->askOwner = threadHasJsResolver();
    w.thread = std::thread([ch = w.ch, work = std::move(work)] {
        t_channel = ch.get();
        work();
        t_channel = nullptr;
        std::lock_guard<std::mutex> lock(ch->wake->mu);
        ch->done = true;
        signal(*ch->wake);
    });
}

// Answers the worker's module request, if it is waiting on one, with the
// JS resolver; a throw closes the channel.
void service(Waiter& w) {
    Channel& ch = *w.ch;
    std::string name;
    {
        std::lock_guard<std::mutex> lock(ch.wake->mu);
        if (!ch.requested || ch.replied || ch.closed) return;
        name = ch.name;
    }
    bool threw = false;
    std::optional<std::string> src = resolveWithJs(name, w.thrown, threw);
    std::lock_guard<std::mutex> lock(ch.wake->mu);
    if (threw) {
        w.threw = true;
        ch.closed = true;
    } else {
        ch.reply = std::move(src);
        ch.replied = true;
    }
    signal(*ch.wake);
}

uint64_t wakeEvents() {
    Wake& w = *ownerWake();
    std::lock_guard<std::mutex> lock(w.mu);
    return w.events;
}

// One step of waiting: until a worker signals past `seen` or the pump's
// interval passes; then the pump.
void waitStep(uint64_t& seen) {
    Wake& w = *ownerWake();
    {
        std::unique_lock<std::mutex> lock(w.mu);
        w.cv.wait_for(lock, std::chrono::milliseconds(std::max(1, t_pumpInterval)),
                      [&] { return w.events != seen; });
        seen = w.events;
    }
    if (t_pump) t_pump();
}

Value makeError(const std::string& message) {
    auto ctor = ev::globalValue("Error");
    if (ctor.found && ev::isFunction(ctor.value)) {
        ev::Persistent c(ctor.value);
        ev::Persistent msg(ev::fromUtf8(message));
        const Value arg = msg.get();
        auto made = ev::construct(c.get(), std::span<const Value>(&arg, 1));
        if (!made.thrown) return made.value;
    }
    return ev::fromUtf8(message);
}

// --- compileAsync ------------------------------------------------------------

struct AsyncJob {
    Waiter w;
    ev::Realm* realm = nullptr;    // the realm that called compileAsync
    ev::Persistent promise;
    std::shared_ptr<CompileResult> result;   // the worker's, read once it is done
    std::string failure;
    bool dropped = false;          // its realm went: its result is discarded
};
thread_local std::vector<std::unique_ptr<AsyncJob>> t_jobs;

void dropJob(AsyncJob& j) {
    j.dropped = true;
    j.realm = nullptr;
    j.promise.set(ev::undefined());
    j.w.close();
}

void settleJob(AsyncJob& j) {
    ev::RealmScope scope(j.realm);
    ev::Persistent p(j.promise.get());
    j.promise.set(ev::undefined());
    if (j.w.threw) {
        ev::rejectPromise(p.get(), j.w.thrown.get());
    } else if (!j.result) {
        ev::Persistent e(makeError("bro.kernel.compileAsync: " + (j.failure.empty() ? std::string("compile failed") : j.failure)));
        ev::rejectPromise(p.get(), e.get());
    } else if (!j.result->ok) {
        ev::Persistent e(makeCompileError(*j.result));
        ev::rejectPromise(p.get(), e.get());
    } else {
        ev::Persistent m(makeModuleValue(std::move(j.result)));
        ev::resolvePromise(p.get(), m.get());
    }
}

// compileAsync's arguments read here and the compile started on a worker.
void startJob(std::span<const Value> args, Value promise) {
    if (args.empty() || !ev::isString(args[0])) ev::throwTypeError("bro.kernel.compileAsync: the source is a string");
    std::string source = ev::toUtf8(args[0]);
    std::string file = "kernel.bk";
    auto compiler = std::make_shared<Compiler>();
    CompileOptions opts;
    configureCompilerModulesAndOptions(args, *compiler, file, opts);
    auto job = std::make_unique<AsyncJob>();
    job->realm = ev::currentRealm();
    job->promise.set(promise);
    AsyncJob* raw = job.get();
    startWorker(raw->w, [raw, compiler, source = std::move(source), file, opts] {
        try {
            CompileResult r = compiler->compile_source(source, file, opts);
            prebuildCpuOnlyKernels(r);
            raw->result = std::make_shared<CompileResult>(std::move(r));
        } catch (const std::exception& e) {
            raw->failure = e.what();
        } catch (...) {
            raw->failure = "compile failed";
        }
    });
    t_jobs.push_back(std::move(job));
}

Value js_compileAsync(Value, std::span<const Value> args) {
    HeldArgs held(args);
    ev::Persistent promise(ev::createPromise());
    const std::vector<Value> current = held.current();
    ev::CallResult r;
    if (!t_pump || t_channel) {
        // No pump: compiled now, the promise settled already.
        ev::Persistent fn(ev::makeFunction([](Value self, std::span<const Value> a) { return js_compile(self, a); }, 2));
        r = ev::call(fn.get(), ev::undefined(), current);
        ev::Persistent v(r.value);
        if (r.thrown) ev::rejectPromise(promise.get(), v.get());
        else ev::resolvePromise(promise.get(), v.get());
        return promise.get();
    }
    ev::Persistent starter(ev::makeFunction(
        [&promise](Value, std::span<const Value> a) -> Value {
            startJob(a, promise.get());
            return ev::undefined();
        },
        2));
    r = ev::call(starter.get(), ev::undefined(), current);
    if (r.thrown) {
        ev::Persistent v(r.value);
        ev::rejectPromise(promise.get(), v.get());
    }
    return promise.get();
}

void onRealmDestroyed(ev::Realm* realm) {
    for (auto& j : t_jobs) {
        if (!j->dropped && j->realm == realm) dropJob(*j);
    }
}

} // namespace

bool onCompileWorker() {
    return t_channel != nullptr;
}

std::optional<std::string> workerResolve(std::string_view name) {
    Channel* ch = t_channel;
    if (auto s = resolveFromTables(name, /*withHostModules=*/!ch->askOwner)) return s;
    if (!ch->askOwner) return std::nullopt;
    std::unique_lock<std::mutex> lock(ch->wake->mu);
    if (ch->closed) return std::nullopt;
    ch->name = std::string(name);
    ch->requested = true;
    ch->replied = false;
    signal(*ch->wake);
    ch->wake->cv.wait(lock, [&] { return ch->replied || ch->closed; });
    ch->requested = false;
    if (!ch->replied) return std::nullopt;
    ch->replied = false;
    return std::move(ch->reply);
}

namespace {

// The @cpu kernels' CPU builds made on threads of their own, so the compile
// that gave them is done without waiting for them; a first run waits
// (ensureCpuBuilt). Joined by shutdownKernelAsync.
struct Prebuild {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
};
std::mutex g_prebuildMu;
std::vector<Prebuild> g_prebuilds;

void joinPrebuilds(bool finishedOnly) {
    std::vector<Prebuild> join;
    {
        std::lock_guard<std::mutex> lock(g_prebuildMu);
        auto keep = std::partition(g_prebuilds.begin(), g_prebuilds.end(), [&](const Prebuild& p) {
            return finishedOnly && !p.done->load(std::memory_order_acquire);
        });
        std::move(keep, g_prebuilds.end(), std::back_inserter(join));
        g_prebuilds.erase(keep, g_prebuilds.end());
    }
    for (auto& p : join) p.thread.join();
}

using CpuBuilds = std::vector<std::shared_ptr<CompiledKernel::CpuBuild>>;

// `r`'s @cpu kernels' builds not made yet.
CpuBuilds unbuiltCpuOnly(const CompileResult& r) {
    CpuBuilds builds;
    for (const auto& k : r.kernels) {
        if (k.cpu_only && k.has_cpu_build() && !k.cpu_built()) builds.push_back(k.cpu_);
    }
    return builds;
}

bool allBuilt(const CpuBuilds& builds) {
    return std::all_of(builds.begin(), builds.end(), [](const auto& b) { return b->ready.load(std::memory_order_acquire); });
}

// The builds made on a thread of their own (one a build another thread has
// begun waits for it there).
void startCpuBuilds(CpuBuilds builds);

// --- cpuReady ------------------------------------------------------------------

// A module's cpuReady() promise, settled by settleAsyncCompiles once its
// builds are made.
struct CpuWait {
    ev::Realm* realm = nullptr;
    ev::Persistent promise;
    CpuBuilds builds;
};
thread_local std::vector<std::unique_ptr<CpuWait>> t_cpuWaits;

// The ready ones out of the list, then resolved in their realms (which runs
// JS); with waitAll, waits (pumped) until none is left.
void settleCpuWaits(bool waitAll) {
    uint64_t seen = wakeEvents();
    while (!t_cpuWaits.empty()) {
        std::vector<std::unique_ptr<CpuWait>> ready;
        for (auto it = t_cpuWaits.begin(); it != t_cpuWaits.end();) {
            if (allBuilt((*it)->builds)) {
                ready.push_back(std::move(*it));
                it = t_cpuWaits.erase(it);
            } else {
                ++it;
            }
        }
        for (auto& w : ready) {
            if (!w->realm) continue;
            ev::RealmScope scope(w->realm);
            ev::Persistent p(w->promise.get());
            w->promise.set(ev::undefined());
            ev::resolvePromise(p.get(), ev::undefined());
        }
        if (!waitAll || t_cpuWaits.empty()) break;
        waitStep(seen);
    }
}

} // namespace

// module.cpuReady() -> Promise<undefined>: once the module's @cpu kernels'
// CPU builds are made (begun now if none was), so their first eval or run
// does not wait in a frame.
Value cpuReadyPromise(const CompileResult& r) {
    ev::Persistent promise(ev::createPromise());
    CpuBuilds builds = unbuiltCpuOnly(r);
    if (builds.empty()) {
        ev::resolvePromise(promise.get(), ev::undefined());
        return promise.get();
    }
    if (!t_pump || t_channel) {
        // No pump: made now, the promise settled already.
        for (const auto& b : builds) {
            CompiledKernel k;
            k.cpu_ = b;
            (void)k.cpu_error();
        }
        ev::resolvePromise(promise.get(), ev::undefined());
        return promise.get();
    }
    startCpuBuilds(builds);
    auto w = std::make_unique<CpuWait>();
    w->realm = ev::currentRealm();
    w->promise.set(promise.get());
    w->builds = std::move(builds);
    t_cpuWaits.push_back(std::move(w));
    return promise.get();
}

namespace {

void startCpuBuilds(CpuBuilds builds) {
    joinPrebuilds(/*finishedOnly=*/true);
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread t([builds = std::move(builds), done] {
        // Module requests from the tables alone: no owner answers this thread.
        Channel ch;
        t_channel = &ch;
        for (const auto& b : builds) {
            CompiledKernel k;
            k.cpu_ = b;
            (void)k.cpu_error();
        }
        t_channel = nullptr;
        done->store(true, std::memory_order_release);
    });
    std::lock_guard<std::mutex> lock(g_prebuildMu);
    g_prebuilds.push_back({std::move(t), std::move(done)});
}

} // namespace

void prebuildCpuOnlyKernels(const CompileResult& r) {
    if (!t_channel) return;
    CpuBuilds builds = unbuiltCpuOnly(r);
    if (!builds.empty()) startCpuBuilds(std::move(builds));
}

void ensureCpuBuilt(const CompiledKernel& k) {
    if (k.cpu_built()) return;
    runCompileWork([&k] { (void)k.cpu_error(); });
}

void runCompileWork(const std::function<void()>& work) {
    if (!t_pump || t_channel) {
        work();
        return;
    }
    std::exception_ptr failure;
    Waiter w;
    uint64_t seen = wakeEvents();
    startWorker(w, [&work, &failure] {
        try {
            work();
        } catch (...) {
            failure = std::current_exception();
        }
    });
    for (;;) {
        service(w);
        if (w.done()) break;
        waitStep(seen);
    }
    w.thread.join();
    if (w.threw) ev::throwValue(w.thrown.get());
    if (failure) {
        std::string what = "compile failed";
        try {
            std::rethrow_exception(failure);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        ev::throwError("bro.kernel: " + what);
    }
}

void decorateAsync(ev::ObjectBuilder& b) {
    b.def("compileAsync", 2, js_compileAsync);
}

void installAsyncRealmHook() {
    ev::addRealmDestroyHook(onRealmDestroyed);
    ev::addRealmDestroyHook([](ev::Realm* realm) {
        for (auto& w : t_cpuWaits) {
            if (w->realm != realm) continue;
            w->realm = nullptr;
            w->promise.set(ev::undefined());
        }
    });
}

void setThreadWaitPump(WaitPump pump, int intervalMs) {
    t_pump = pump;
    t_pumpInterval = intervalMs > 0 ? intervalMs : 16;
}

void settleAsyncCompiles(bool waitAll) {
    settleCpuWaits(waitAll);
    if (t_jobs.empty()) return;
    uint64_t seen = wakeEvents();
    for (;;) {
        // By index: the resolver's JS may start more jobs.
        for (size_t i = 0; i < t_jobs.size(); ++i) {
            AsyncJob& j = *t_jobs[i];
            if (j.dropped) continue;
            ev::RealmScope scope(j.realm);
            service(j.w);
        }
        // The finished out of the list before settling (which runs JS).
        std::vector<std::unique_ptr<AsyncJob>> finished;
        for (auto it = t_jobs.begin(); it != t_jobs.end();) {
            if ((*it)->w.done()) {
                finished.push_back(std::move(*it));
                it = t_jobs.erase(it);
            } else {
                ++it;
            }
        }
        for (auto& j : finished) {
            j->w.thread.join();
            if (!j->dropped) settleJob(*j);
        }
        if (!waitAll || t_jobs.empty()) break;
        waitStep(seen);
    }
}

void shutdownKernelAsync() {
    std::vector<std::unique_ptr<AsyncJob>> jobs = std::move(t_jobs);
    t_jobs.clear();
    for (auto& j : jobs) dropJob(*j);
    for (auto& j : jobs) {
        if (j->w.thread.joinable()) j->w.thread.join();
    }
    t_cpuWaits.clear();
    joinPrebuilds(/*finishedOnly=*/false);
}

} // namespace brokernel::api

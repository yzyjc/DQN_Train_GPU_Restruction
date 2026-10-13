#include "../DQN_Main_TUI_No_Terminal/dashboard_workers.h"
#include <cassert>
#include <string>
#include <cstdio>

struct FakeBackend {
    bool live[3] = { true, true, true };
    bool protect_ok = true, exit_ok = true, operation_ok = true;
    std::string calls;
    bool running(int id) { return live[id]; }
    bool protect_trainer_stop() { calls += 'P'; return protect_ok; }
    bool request_trainer_exit() { calls += 'E'; return exit_ok; }
    bool start(int id) { calls += char('0' + id); if (operation_ok) live[id] = true; return operation_ok; }
    bool stop(int id) { calls += char('a' + id); if (operation_ok) live[id] = false; return operation_ok; }
};

int main() {
    for (int id = 0; id < DASH_WORKER_COUNT; ++id) {
        DashboardWorkers state;
        FakeBackend backend;
        assert(dashboard_toggle_worker(state, id, backend));
        assert(backend.calls == (id == 0 ? "PE" : id == 1 ? "b" : "c"));
        assert(state.stopping[id]);
        for (int other = 0; other < 3; ++other) if (other != id) assert(backend.live[other]);
        backend.live[id] = false; // Simulate acknowledgement and process exit.
        backend.calls.clear();
        assert(dashboard_toggle_worker(state, id, backend));
        assert(backend.calls == std::string(1, char('0' + id)));
        assert(!state.stopping[id]);
        assert(backend.live[id]);
        if (id == 1) assert(state.watchdog_enabled);
        if (id == 2) assert(state.guard_enabled);
    }
    DashboardWorkers state;
    FakeBackend backend;
    backend.protect_ok = false;
    assert(!dashboard_toggle_worker(state, DASH_TRAINER, backend));
    assert(backend.calls == "P" && !state.stopping[0]);
    backend.protect_ok = true; backend.exit_ok = false; backend.calls.clear();
    assert(!dashboard_toggle_worker(state, DASH_TRAINER, backend));
    assert(backend.calls == "PE" && !state.stopping[0]);
    backend.exit_ok = true; backend.calls.clear();
    assert(dashboard_toggle_worker(state, DASH_TRAINER, backend));
    assert(dashboard_toggle_worker(state, DASH_TRAINER, backend)); // Failed save can be retried.
    assert(backend.calls == "PEPE" && backend.live[0]);
    backend.operation_ok = false; backend.calls.clear();
    assert(!dashboard_toggle_worker(state, DASH_WATCHDOG, backend));
    assert(state.watchdog_enabled && !state.stopping[1]);
    backend.live[2] = false;
    assert(!dashboard_toggle_worker(state, DASH_LOGGUARD, backend));
    assert(!state.guard_enabled);
    backend.calls.clear();
    assert(!dashboard_toggle_worker(state, -1, backend));
    assert(!dashboard_toggle_worker(state, DASH_WORKER_COUNT, backend));
    assert(backend.calls.empty());
    DashboardWorkerHit hit = { 4, 8, 13 };
    assert(hit.contains(4, 8) && hit.contains(16, 8));
    assert(!hit.contains(17, 8) && !hit.contains(3, 8) && !hit.contains(4, 9));
    puts("Dashboard worker regression checks passed.");
}

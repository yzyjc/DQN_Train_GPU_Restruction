#pragma once

enum DashboardWorker { DASH_TRAINER, DASH_WATCHDOG, DASH_LOGGUARD, DASH_WORKER_COUNT };
struct DashboardWorkers {
    int guard_enabled = 0;
    int watchdog_enabled = 1;
    bool stopping[DASH_WORKER_COUNT] = {};
};
struct DashboardWorkerHit {
    int x = -1, y = -1, width = 0;
    bool contains(int px, int py) const { return py == y && px >= x && px < x + width; }
};

// Policy has no process/file dependencies, so ordering can be tested safely.
template<class Backend>
bool dashboard_toggle_worker(DashboardWorkers& state, int id, Backend& backend) {
    if (id < 0 || id >= DASH_WORKER_COUNT) return false;
    bool running = backend.running(id);
    if (id == DASH_TRAINER) {
        if (running) {
            if (!backend.protect_trainer_stop()) return false;
            if (!backend.request_trainer_exit()) return false;
            state.stopping[id] = true;
            return true;
        }
        if (!backend.start(id)) return false;
    } else if (running) {
        if (!backend.stop(id)) return false;
        if (id == DASH_WATCHDOG) state.watchdog_enabled = 0;
        else state.guard_enabled = 0;
        state.stopping[id] = true;
        return true;
    } else {
        if (!backend.start(id)) return false;
        if (id == DASH_WATCHDOG) state.watchdog_enabled = 1;
        else state.guard_enabled = 1;
    }
    state.stopping[id] = false;
    return true;
}

/* Pure scheduling policy for Neural Probe publication.
   Kept free of GPU/IPC calls so timing and step-count behaviour is testable. */
#pragma once
#include <stdint.h>
#include "neural_probe_protocol.h"

typedef struct ProbePublishSchedule {
    uint64_t pretrain_last_ms;
    uint32_t formal_steps_since_publish;
    int formal_viewer_seen;
} ProbePublishSchedule;

static inline int probe_schedule_pretrain(ProbePublishSchedule* state,
    int viewer_active, uint64_t now_ms) {
    if (!state) return 0;
    if (!viewer_active) {
        state->pretrain_last_ms = 0;
        return 0;
    }
    if (state->pretrain_last_ms != 0 &&
        now_ms - state->pretrain_last_ms < PROBE_PRETRAIN_PUBLISH_MS) return 0;
    state->pretrain_last_ms = now_ms;
    return 1;
}

static inline int probe_schedule_formal_step(ProbePublishSchedule* state,
    int viewer_active) {
    if (!state) return 0;
    if (!viewer_active) {
        state->formal_steps_since_publish = 0;
        state->formal_viewer_seen = 0;
        return 0;
    }
    if (!state->formal_viewer_seen) {
        state->formal_viewer_seen = 1;
        state->formal_steps_since_publish = 0;
        return 1; /* Viewer just connected: publish the first completed step. */
    }
    if (++state->formal_steps_since_publish >= PROBE_FORMAL_PUBLISH_STEPS) {
        state->formal_steps_since_publish = 0;
        return 1;
    }
    return 0;
}

#pragma once
#include <stdio.h>

// Legacy checkpoints lack curriculum state. Replay only events at/before the
// resumed episode; later log entries belong to a discarded future branch.
inline void restore_pretrain_progress(const char* path, int episode, int* target, int* passes) {
    *target = *passes = 0;
    FILE* f = fopen(path, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int ep = 0, value = 0; float success = 0, average = 0;
        if (sscanf(line, "CONSOLIDATE %d %d", &ep, &value) == 2 &&
            ep <= episode && value >= ep) *target = value;
        else if (sscanf(line, "EVAL %d %f %f %d", &ep, &success, &average, &value) == 4 &&
            ep <= episode && value >= 0) *passes = value;
    }
    fclose(f);
}

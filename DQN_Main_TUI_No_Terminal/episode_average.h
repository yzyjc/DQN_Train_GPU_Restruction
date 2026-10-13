#pragma once
#include <stdint.h>
#include <math.h>

// State survives read batches. The reader owns deduplication by byte offset and
// must reset this together with its cursor after replacement/rollback/reload.
struct EpisodeAccumulator {
    int ep = 0;
    uint64_t count = 0;
    double sum = 0;
    bool add(int episode, float value) {
        if (episode <= 0 || !isfinite(value) || fabsf(value) >= 1e30f ||
            (count && episode < ep)) return false;
        if (!count || episode != ep) { ep = episode; count = 0; sum = 0; }
        sum += value; ++count;
        return true;
    }
    float mean() const { return count ? (float)(sum / count) : NAN; }
};

#pragma once
#include "core.h"

// One epsilon=0 policy for live play and examinations. Near ties are sampled,
// not resolved by board index (which lets a zero network accidentally make four).
inline int greedy_legal_action(const GameState* game, const float* q, PRNGState* rng) {
    if (!game || !q || !rng) return -1;
    int candidates[CELL_COUNT], count = 0;
    float best = -INFINITY;
    for (int a = 0; a < CELL_COUNT; ++a) if (!game->board[a]) {
        if (!isfinite(q[a])) return -1;
        best = fmaxf(best, q[a]);
    }
    for (int a = 0; a < CELL_COUNT; ++a)
        if (!game->board[a] && fabsf(q[a] - best) < 1e-6f) candidates[count++] = a;
    return count ? candidates[prng_int(rng, count)] : -1;
}

// Works for both monotonic counters and checkpoint-normalized next slots.
inline int opponent_pool_latest_index(int next, int capacity) {
    if (next < 0 || capacity <= 0) return -1;
    return (next % capacity + capacity - 1) % capacity;
}

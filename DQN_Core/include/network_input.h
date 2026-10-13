#pragma once

// Replay/on-disk states keep absolute colours and the side to move. Only the
// network boundary converts to self/opponent; action coordinates never change.
#ifdef __HIPCC__
__host__ __device__
#endif
inline int network_token_index(const float* state, int cell) {
    const float own = state[25] >= 0.5f ? -state[cell] : state[cell];
    return own > 0.5f ? 2 : (own < -0.5f ? 0 : 1);
}

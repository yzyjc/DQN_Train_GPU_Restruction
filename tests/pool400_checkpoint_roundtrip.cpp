#include "core.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>

int main() {
    const char* path = "pool400_v8_roundtrip_test.ckpt";
    std::unique_ptr<DQNAgent> source(new DQNAgent{});
    std::unique_ptr<DQNAgent> restored(new DQNAgent{});
    init_dqn_ex(source.get(), 0);
    init_dqn_ex(restored.get(), 0);
    g_start_time = time(nullptr);
    g_wins[0] = g_wins[1] = g_wins[2] = 0;

    float input[CELL_COUNT + 1] = {};
    for (int i = 0; i < CELL_COUNT; ++i) input[i] = float((i * 7) % 3 - 1);
    input[CELL_COUNT] = 1.0f;
    float before[CELL_COUNT], after[CELL_COUNT];
    forward_cpu(&source->q_net, input, before);

    int ok = save_checkpoint(source.get(), 1, 123, 2, path);
    int phase = -1, episode = -1, depth = -1;
    if (ok) ok = load_checkpoint(restored.get(), &phase, &episode, &depth, path, 1);
    if (ok) forward_cpu(&restored->q_net, input, after);

    float max_q_error = 0.0f;
    if (ok) {
        for (int i = 0; i < CELL_COUNT; ++i)
            max_q_error = fmaxf(max_q_error, fabsf(before[i] - after[i]));
        ok = phase == 1 && episode == 123 && depth == 2 &&
             restored->checkpoint_version == 11 &&
             memcmp(source->q_net.pool_w, restored->q_net.pool_w,
                    sizeof(source->q_net.pool_w)) == 0 &&
             max_q_error == 0.0f;
    }

    printf("V11 checkpoint roundtrip: %s (phase=%d episode=%d depth=%d version=%d max_q_error=%.9g)\n",
        ok ? "PASS" : "FAIL", phase, episode, depth,
        restored->checkpoint_version, max_q_error);
    remove(path);
    free_dqn(source.get());
    free_dqn(restored.get());
    return ok ? 0 : 1;
}

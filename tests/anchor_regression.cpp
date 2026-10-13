// anchor_regression.cpp — 记忆锚点 Replay 回归
// 验证：锚点混合后 replay_from_buffer 仍正常（不崩溃、返回有限 loss）；
//       锚点 transition 全部通过 validate_experience（由 anchor_push 结构性保证，
//       这里通过调用 store_experience 生成合法经验后跑 replay 验证混合路径）。
#include "core.h"
#include <memory>

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); g_failures++; } } while (0)

int main() {
    g_current_agent = NULL;
    g_learning_rate = 0.0004f;
    g_training_fault = 0;

    DQNAgent* agent = (DQNAgent*)calloc(1, sizeof(DQNAgent));
    CHECK(agent != NULL);
    init_dqn_ex(agent, 0);
    g_current_agent = agent;

    /* 填充 replay_black 到至少 BATCH_SIZE，供 replay_from_buffer 采样。 */
    GameState g; reset_game(&g);
    float state[CELL_COUNT + 1], next_state[CELL_COUNT + 1];
    state_to_input(&g, state);
    for (int i = 0; i < BATCH_SIZE * 2; ++i) {
        GameState ng = g;
        int action = i % CELL_COUNT;
        make_move(&ng, action);
        state_to_input(&ng, next_state);
        store_experience(agent, state, action, 0.0f, next_state, 0, 0);
    }

    /* 跑几次 replay，锚点会以 10% 比例混入采样，验证路径不崩溃、loss 有限。 */
    for (int k = 0; k < 3; ++k) {
        float loss = replay_from_buffer(agent, agent->replay_black,
            agent->replay_black_count, &agent->replay_black_idx);
        CHECK(isfinite(loss));
    }
    printf("anchor replay: 3 batches ok, global_step=%lld\n", agent->global_step);

    free_dqn(agent);
    free(agent);
    g_current_agent = NULL;

    printf("\n%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

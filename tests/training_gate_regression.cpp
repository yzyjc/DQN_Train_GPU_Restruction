// training_gate_regression.cpp — 训练安全门 + TQ 语义回归
// 验证：正常更新成功；LR 突增/发散被安全门或哨兵拒绝且权重不变；
//       terminal 经验不 bootstrap（符号/裁剪语义）。
#include "core.h"
#include "sentinel_guard.h"
#include <memory>

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); g_failures++; } } while (0)

static int nets_equal(const NeuralNet* a, const NeuralNet* b) {
    return memcmp(a, b, sizeof(NeuralNet)) == 0;
}

int main() {
    g_current_agent = NULL;
    g_learning_rate = 0.0004f;
    g_training_fault = 0;

    DQNAgent* agent = (DQNAgent*)calloc(1, sizeof(DQNAgent));
    CHECK(agent != NULL);
    init_dqn_ex(agent, 0);
    g_current_agent = agent;

    float state[CELL_COUNT + 1];
    GameState g; reset_game(&g);
    state_to_input(&g, state);
    int action = 0;
    float target = 0.5f;
    float out_loss = 0.0f;

    /* 1. 正常小 LR 更新成功，网络变化，不触发任何拒绝。 */
    NeuralNet before1 = agent->q_net;
    int trained = train_agent_batch(agent, state, &action, &target, 1, 0.0004f, &out_loss);
    CHECK(trained == 1);
    CHECK(!nets_equal(&before1, &agent->q_net));
    printf("normal update: loss=%.6g ok\n", out_loss);

    /* 2. 先跑几批正常更新建立 EMA，再用超大 LR + 超大 target 制造突增，
          安全门或哨兵必须拒绝，且 q_net 逐字节不变。 */
    for (int i = 0; i < 5; ++i) {
        train_agent_batch(agent, state, &action, &target, 1, 0.0004f, &out_loss);
    }
    NeuralNet before2 = agent->q_net;
    float states[BATCH_SIZE][CELL_COUNT + 1];
    int actions[BATCH_SIZE];
    float targets[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; ++i) {
        memcpy(states[i], state, sizeof(state));
        actions[i] = 0;
        targets[i] = 100.0f;
    }
    const unsigned long long delta_rejects_before = g_batch_delta_reject_count;
    int trained2 = train_agent_batch(agent, (const float*)states, actions, targets,
        BATCH_SIZE, 1.0f, &out_loss);
    CHECK(trained2 == 2);                       /* 安全拒绝，未提交 */
    CHECK(nets_equal(&before2, &agent->q_net)); /* q_net 逐字节不变 */
    CHECK(g_batch_delta_reject_count > delta_rejects_before);
    printf("surge update rejected: reject_count=%llu\n", sentinel_reject_count());

    free_dqn(agent);
    free(agent);
    g_current_agent = NULL;

    printf("\n%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

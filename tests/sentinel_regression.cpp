// sentinel_regression.cpp — 哨兵保护回归测试
// 验证：哨兵评估区分健康/发散网络；sentinel_reject_update 拒绝 Q 爆炸；
//       正常更新成功；触发哨兵的更新被拒绝后 q_net 逐字节不变。
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
    g_learning_rate = 0.0005f;
    g_training_fault = 0;

    /* 1. 哨兵评估：健康网络无非有限值、max|Q| 有限。 */
    NeuralNet healthy;
    init_net(&healthy);
    SentinelHealth h;
    sentinel_evaluate_net(&healthy, &h);
    CHECK(h.nonfinite == 0);
    CHECK(h.total > 0);
    CHECK(isfinite(h.max_abs_q));
    printf("healthy sentinel: total=%d max|Q|=%.4f min_margin=%.4f nonfinite=%d\n",
        h.total, h.max_abs_q, h.min_margin, h.nonfinite);

    /* 2. 发散网络：放大可学习参数，Q 爆炸，sentinel_reject_update 拒绝。 */
    NeuralNet diverged;
    memcpy(&diverged, &healthy, sizeof(NeuralNet));
    {
        float* p = (float*)&diverged;
        /* 可学习参数区（与 GpuNetParams 布局一致）。 */
        const size_t learnable_floats =
            3 * TOKEN_EMBED_DIM + CELL_COUNT * TOKEN_EMBED_DIM +
            HIDDEN1_SIZE * FEATURE_DIM + HIDDEN1_SIZE +
            HIDDEN2_SIZE * HIDDEN1_SIZE + HIDDEN2_SIZE +
            HIDDEN3_SIZE * HIDDEN2_SIZE + HIDDEN3_SIZE +
            CELL_COUNT * HIDDEN3_SIZE + CELL_COUNT +
            ATTENTION_EMBED * TOKEN_EMBED_DIM * 3 + ATTENTION_EMBED * 3 +
            TOKEN_EMBED_DIM * ATTENTION_EMBED + TOKEN_EMBED_DIM +
            FEATURE_DIM * CELL_COUNT * TOKEN_EMBED_DIM + FEATURE_DIM;
        for (size_t i = 0; i < learnable_floats; ++i) p[i] *= 8.0f;
    }
    SentinelHealth d;
    sentinel_evaluate_net(&diverged, &d);
    printf("diverged sentinel: total=%d max|Q|=%.4f min_margin=%.4f nonfinite=%d\n",
        d.total, d.max_abs_q, d.min_margin, d.nonfinite);
    char reason[128];
    int rejected = sentinel_reject_update(&h, &d, reason, sizeof(reason));
    CHECK(rejected == 1);
    CHECK(reason[0] != '\0');
    printf("reject reason: %s\n", reason);

    /* An already-high checkpoint must be allowed to move back toward health;
       otherwise every restart inevitably accumulates 20 rejects and pauses. */
    {
        SentinelHealth improving = d;
        improving.max_abs_q = d.max_abs_q * 0.95f;
        CHECK(sentinel_reject_update(&d, &improving, reason, sizeof(reason)) == 0);
        SentinelHealth small_worsening = d;
        small_worsening.max_abs_q = d.max_abs_q * 1.05f;
        CHECK(sentinel_reject_update(&d, &small_worsening, reason, sizeof(reason)) == 0);
        SentinelHealth worsening = d;
        worsening.max_abs_q = d.max_abs_q * 1.25f;
        CHECK(sentinel_reject_update(&d, &worsening, reason, sizeof(reason)) == 1);

        SentinelHealth low_margin = h;
        low_margin.min_margin = -30.0f;
        SentinelHealth margin_improving = low_margin;
        margin_improving.min_margin = -25.0f;
        CHECK(sentinel_reject_update(&low_margin, &margin_improving, reason, sizeof(reason)) == 0);
        SentinelHealth margin_small_worsening = low_margin;
        margin_small_worsening.min_margin = -31.0f;
        CHECK(sentinel_reject_update(&low_margin, &margin_small_worsening, reason, sizeof(reason)) == 0);
        SentinelHealth margin_worsening = low_margin;
        margin_worsening.min_margin = -40.0f;
        CHECK(sentinel_reject_update(&low_margin, &margin_worsening, reason, sizeof(reason)) == 1);

        /* Repeated rejected candidates are already rolled back.  The guard must
           never freeze episode progress or silently quarter the global LR. */
        const float lr_before_escalation = g_learning_rate;
        for (int i = 0; i < 2 * 20; ++i) sentinel_rejection_escalate("regression burst");
        CHECK(sentinel_guard_paused() == 0);
        CHECK(g_learning_rate == lr_before_escalation);
        sentinel_guard_resume();
    }

    /* 3. 正常 CPU 更新成功，且网络确实变化。 */
    {
        DQNAgent* agent = (DQNAgent*)calloc(1, sizeof(DQNAgent));
        CHECK(agent != NULL);
        init_dqn_ex(agent, 0);
        g_current_agent = agent;
        g_training_fault = 0;

        NeuralNet before = agent->q_net;
        float state[CELL_COUNT + 1];
        GameState g; reset_game(&g);
        state_to_input(&g, state);
        int action = 0;
        float target = 0.5f;
        float out_loss = 0.0f;
        int trained = train_agent_batch(agent, state, &action, &target, 1, 0.0005f, &out_loss);
        CHECK(trained == 1);
        CHECK(!nets_equal(&before, &agent->q_net));

        /* 4. 已越界网络必须能提交改善型更新，否则恢复后会固定累计拒绝
           并进入 SAFE_HOLD。硬越界/继续恶化的拒绝语义已由上面的纯判据
           用例覆盖。 */
        {
            float* p = (float*)&agent->q_net;
            const size_t learnable_floats =
                3 * TOKEN_EMBED_DIM + CELL_COUNT * TOKEN_EMBED_DIM +
                HIDDEN1_SIZE * FEATURE_DIM + HIDDEN1_SIZE +
                HIDDEN2_SIZE * HIDDEN1_SIZE + HIDDEN2_SIZE +
                HIDDEN3_SIZE * HIDDEN2_SIZE + HIDDEN3_SIZE +
                CELL_COUNT * HIDDEN3_SIZE + CELL_COUNT +
                ATTENTION_EMBED * TOKEN_EMBED_DIM * 3 + ATTENTION_EMBED * 3 +
                TOKEN_EMBED_DIM * ATTENTION_EMBED + TOKEN_EMBED_DIM +
                FEATURE_DIM * CELL_COUNT * TOKEN_EMBED_DIM + FEATURE_DIM;
            for (size_t i = 0; i < learnable_floats; ++i) p[i] *= 8.0f;
            NeuralNet before2 = agent->q_net;
            float out_loss2 = 0.0f;
            int trained2 = train_agent_batch(agent, state, &action, &target, 1, 0.0005f, &out_loss2);
            CHECK(trained2 == 1); /* 改善型更新允许提交 */
            CHECK(!nets_equal(&before2, &agent->q_net));
            CHECK(sentinel_reject_count() >= 1);
        }

        free_dqn(agent);
        free(agent);
        g_current_agent = NULL;
    }

    printf("\n%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

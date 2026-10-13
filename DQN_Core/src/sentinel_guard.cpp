// sentinel_guard.cpp — 哨兵集保护实现
// 纯 CPU 前向、无记忆库叠加的"裸网络"健康检查。
#include "sentinel_guard.h"
#include "skill_test.h"
#include "training_log_timeline.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* 健康阈值：
   - 训练 target 被 clamp 到 +/-5，正常 Q 值也应在该量级附近；>20 即视为爆炸。
   - 正常 skill margin 为正值（正确动作 Q 高于错误动作）；<-20 即视为崩塌。 */
#define SENTINEL_Q_ABS_LIMIT   20.0f
#define SENTINEL_MIN_MARGIN   -20.0f
#define SENTINEL_WIN1_COUNT    8
#define SENTINEL_BLOCK1_COUNT  8

static SentinelPuzzle g_sentinels[SENTINEL_MAX_PUZZLES];
static int g_sentinel_count = 0;
static int g_sentinel_inited = 0;
static unsigned long long g_sentinel_reject_count = 0;

void sentinel_guard_init(void) {
    if (g_sentinel_inited) return;
    g_sentinel_inited = 1;
    g_sentinel_count = 0;

    /* Win1 / Block1 固定题库（skill_test 用固定 seed 生成，确定性）。 */
    GameState win1[64], block1[64];
    int win1_answers[64][CELL_COUNT], block1_answers[64][CELL_COUNT];
    int win1_na[64], block1_na[64];
    int win1_count = 0, block1_count = 0;
    generate_win1_fixed(win1, win1_answers, win1_na, &win1_count, 64);
    generate_block1_fixed(block1, block1_answers, block1_na, &block1_count, 64);

    for (int i = 0; i < SENTINEL_WIN1_COUNT && i < win1_count && g_sentinel_count < SENTINEL_MAX_PUZZLES; ++i) {
        SentinelPuzzle* sp = &g_sentinels[g_sentinel_count++];
        sp->puzzle = win1[i];
        sp->num_answers = win1_na[i];
        memcpy(sp->answers, win1_answers[i], sizeof(int) * sp->num_answers);
        sp->is_win1 = 1;
    }
    for (int i = 0; i < SENTINEL_BLOCK1_COUNT && i < block1_count && g_sentinel_count < SENTINEL_MAX_PUZZLES; ++i) {
        SentinelPuzzle* sp = &g_sentinels[g_sentinel_count++];
        sp->puzzle = block1[i];
        sp->num_answers = block1_na[i];
        memcpy(sp->answers, block1_answers[i], sizeof(int) * sp->num_answers);
        sp->is_win1 = 0;
    }

    /* 追加空盘作为正常局面哨兵（is_win1 = -1）。 */
    if (g_sentinel_count < SENTINEL_MAX_PUZZLES) {
        SentinelPuzzle* sp = &g_sentinels[g_sentinel_count++];
        reset_game(&sp->puzzle);
        sp->num_answers = 0;
        sp->is_win1 = -1;
    }

    printf("[SENTINEL] 已初始化 %d 个哨兵（%d Win1 + %d Block1 + 正常局面）\n",
        g_sentinel_count, SENTINEL_WIN1_COUNT, SENTINEL_BLOCK1_COUNT);
}

static int evaluate_one(const NeuralNet* net, const SentinelPuzzle* sp, SentinelHealth* h) {
    float input[CELL_COUNT + 1];
    state_to_input(&sp->puzzle, input);
    float q[CELL_COUNT];
    forward_cpu_only(net, input, q);

    int nonfinite_here = 0;
    float correct = -INFINITY, wrong = -INFINITY;
    for (int a = 0; a < CELL_COUNT; ++a) {
        if (sp->puzzle.board[a]) continue;
        float v = q[a];
        if (!isfinite(v)) { nonfinite_here = 1; continue; }
        if (fabsf(v) > h->max_abs_q) h->max_abs_q = fabsf(v);
        if (sp->is_win1 >= 0) {
            int valid = 0;
            for (int j = 0; j < sp->num_answers; ++j) if (sp->answers[j] == a) valid = 1;
            if (valid) correct = fmaxf(correct, v); else wrong = fmaxf(wrong, v);
        }
    }
    if (nonfinite_here) h->nonfinite++;
    int has_margin = 0;
    if (sp->is_win1 >= 0 && isfinite(correct) && isfinite(wrong)) {
        float m = correct - wrong;
        if (m < h->min_margin) h->min_margin = m;
        h->mean_margin += m;
        /* 选择动作：取 Q 最大的合法位置（与 greedy 一致，这里只统计正确性） */
        int chosen = -1; float best = -INFINITY;
        for (int a = 0; a < CELL_COUNT; ++a) {
            if (sp->puzzle.board[a]) continue;
            if (q[a] > best) { best = q[a]; chosen = a; }
        }
        for (int j = 0; j < sp->num_answers; ++j) if (sp->answers[j] == chosen) h->correct_count++;
        has_margin = 1;
    }
    h->total++;
    return has_margin;
}

/* ===== 仅测试构建可用的确定性注入 =====
   `SENTINEL_TEST_INJECT` **只在测试专用构建**里定义（见
   build_verify/indep_p1_tests/build_inject_lib.ps1）；生产库与正式 Trainer 从不
   定义它，因此不存在运行期开关，也不存在任何生产代码路径能触达。
   用途：把真实训练入口 `train_agent_batch_side` 确定性地驱进哨兵拒绝分支，验证
   回滚行为（权重/EMA/预热/提交计数全部恢复、只有拒绝统计变化）。
   它**不冒充**自然数值触发，也不冒充 GPU 数值验证——注入的是「本次评估结果非有限」
   这一条判据本身，测试必须同时读出 reason 字符串来证明命中的是
   `after->nonfinite > 0`。 */
#ifdef SENTINEL_TEST_INJECT
static int g_sentinel_test_inject_after = 0;

/* 仅测试构建：让**接下来 N 次** `sentinel_evaluate_net` 的结果带 nonfinite>0。
   `train_agent_batch_side` 在一批里会评估两次（提交前的 before 与提交后的 after），
   所以测试注入 2 次：这样 after 一定会走 `after->nonfinite > 0` 这一条判据，
   与注入与否完全解耦（不依赖哪一次先被调用）。 */
extern "C" void sentinel_test_inject_after_nonfinite_n(int n) {
    g_sentinel_test_inject_after = (n > 0) ? n : 0;
}

extern "C" int sentinel_test_inject_remaining(void) {
    return g_sentinel_test_inject_after;
}
#endif

void sentinel_evaluate_net(const NeuralNet* net, SentinelHealth* out) {
    if (!net || !out) return;
    memset(out, 0, sizeof(*out));
    out->min_margin = INFINITY;
    out->mean_margin = 0.0f;
    sentinel_guard_init();
    int margin_count = 0;
    for (int i = 0; i < g_sentinel_count; ++i) {
        margin_count += evaluate_one(net, &g_sentinels[i], out);
    }
    if (out->min_margin == INFINITY) out->min_margin = 0.0f;
    if (margin_count > 0) out->mean_margin /= (float)margin_count;
#ifdef SENTINEL_TEST_INJECT
    if (g_sentinel_test_inject_after > 0) {
        --g_sentinel_test_inject_after;   /* 计数式：用尽后自动解除 */
        if (out->nonfinite < 1) out->nonfinite = 1;
    }
#endif
}

int sentinel_reject_update(const SentinelHealth* before, const SentinelHealth* after,
    char* reason, int reason_size) {
    if (!after || !reason || reason_size <= 0) return 0;
    reason[0] = '\0';
    if (after->nonfinite > 0) {
        snprintf(reason, reason_size, "sentinel non-finite Q count=%d", after->nonfinite);
        g_sentinel_reject_count++;
        return 1;
    }
    if (after->max_abs_q > SENTINEL_Q_ABS_LIMIT) {
        /* The absolute envelope is an alarm threshold, not a wall.  A restored
           model can already sit near/outside it; rejecting every tiny crossing
           then makes correction impossible.  Reject only a material one-batch
           surge.  Slow finite drift is handled by bounded Huber gradients, the
           batch-delta gate and periodic Skill Test, without freezing episodes. */
        const float baseline = before && isfinite(before->max_abs_q) ? before->max_abs_q : 0.0f;
        const float surge = fmaxf(1.0f, baseline * 0.10f);
        if (!before || after->max_abs_q > baseline + surge) {
            snprintf(reason, reason_size, "sentinel max|Q| surged %.6g -> %.6g",
                baseline, after->max_abs_q);
            g_sentinel_reject_count++;
            return 1;
        }
    }
    if (after->min_margin < SENTINEL_MIN_MARGIN) {
        const float baseline = before && isfinite(before->min_margin) ? before->min_margin : 0.0f;
        const float collapse = fmaxf(1.0f, fabsf(baseline) * 0.10f);
        if (!before || after->min_margin < baseline - collapse) {
            snprintf(reason, reason_size, "sentinel min margin collapsed %.6g -> %.6g",
                baseline, after->min_margin);
            g_sentinel_reject_count++;
            return 1;
        }
    }
    return 0;
}

unsigned long long sentinel_reject_count(void) {
    return g_sentinel_reject_count;
}

/* 连续拒绝是诊断信号，不得改写学习率或停止环境局数。
   候选批次已由调用方回滚，继续采样新批次才有机会自我恢复。 */
#define SENTINEL_CONSECUTIVE_LIMIT 20
static int g_sentinel_consecutive = 0;
static volatile LONG g_sentinel_paused = 0;
static char g_sentinel_pause_reason[128] = "";

/* ===== 内存中的最后健康权重快照：已删除 =====
   原先这里有一份共享 `NeuralNet` 健康快照 + 两个访问函数。它们在按方路由下
   语义不正确（详见文件末尾的说明），已随调用面一起移除。 */

void sentinel_rejection_escalate(const char* reason) {
    ++g_sentinel_consecutive;
    if (g_sentinel_consecutive < SENTINEL_CONSECUTIVE_LIMIT) return;
    g_sentinel_consecutive = 0;
    strncpy(g_sentinel_pause_reason, reason ? reason : "repeated sentinel rejection",
        sizeof(g_sentinel_pause_reason) - 1);
    g_sentinel_pause_reason[sizeof(g_sentinel_pause_reason) - 1] = '\0';
    printf("[SENTINEL] 连续拒绝 %d 个危险候选批：%s。均已回滚，继续训练且不改写学习率。\n",
        SENTINEL_CONSECUTIVE_LIMIT, g_sentinel_pause_reason);

    /* NoTerminal 的 stdout 通常不可见；把升级事件持久化，下次可直接取证。 */
    const std::string path = dqn_log_path("sentinel_guard.log");
    FILE* log = fopen(path.c_str(), "a");
    if (log) {
        SYSTEMTIME now;
        GetLocalTime(&now);
        fprintf(log, "%04u-%02u-%02u %02u:%02u:%02u rejected_total=%llu lr=%.9g reason=%s action=rollback_continue\n",
            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
            g_sentinel_reject_count, g_learning_rate, g_sentinel_pause_reason);
        fclose(log);
    }
}

void sentinel_rejection_reset(void) {
    g_sentinel_consecutive = 0;
}

int sentinel_guard_paused(void) {
    return InterlockedCompareExchange(&g_sentinel_paused, 0, 0) != 0;
}

void sentinel_guard_resume(void) {
    g_sentinel_consecutive = 0;
    g_sentinel_pause_reason[0] = '\0';
    InterlockedExchange(&g_sentinel_paused, 0);
    printf("[SENTINEL] 已清除拒绝计数器。\n");
}

const char* sentinel_guard_pause_reason(void) { return g_sentinel_pause_reason; }

/* ===== 已移除：共享 q_net 健康快照（sentinel_set/restore_healthy_q）=====
   这两个函数连同 `sentinel_has_healthy_snapshot` 一起删除，原因不是「暂时没用」，
   而是它们在按方路由下**在语义上无法正确工作**：

   * 存的是共享 `agent->q_net`，而所有按方消费者读的是 `online(side)`，
     还原时只会改共享字段 ⇒ 表面上「回滚了」，实际上没有任何一方的权重被恢复；
   * GPU 四句柄驻留时 `agent->q_net` 还可能落后于该方设备句柄，
     存下来的快照连「某一次真实评测用过的网络」都不保证。

   保留它等于给将来留下一个**静默无效的回滚**。真正的单批回滚由
   `train_agent_batch_side` 内的 `before_net` 副本 + `sentinel_reject_update`
   承担，逐方、逐字节，已有 `tests/sentinel_rollback_regression.cpp` 覆盖。
   若将来看护真需要「最远回滚点」，必须按方存（`online/target` 各一份）并显式
   上传该方句柄——那是一个独立的设计决定，不应靠复活这两个函数来完成。 */

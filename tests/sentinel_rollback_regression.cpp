// sentinel_rollback_regression.cpp — 哨兵回滚专用测试（两层，deepseek-flash）
//
// 用户指令要求哨兵测试分两层：
//
// **第一层（纯判据，不需要训练）**：用明确的 before/after `SentinelHealth` 直接调
// `sentinel_reject_update`，逐条验证 `DQN_Core/src/sentinel_guard.cpp:115-146` 的三项
// 判据，并覆盖两个负例：
//   (a) 越过绝对包络但同批突增不足 → **不**拒绝；
//   (b) 未越过绝对包络（|Q| ≤ 20 / margin ≥ −20）→ **不**拒绝。
// 同时显式证明 `correct_count` 不参与判断。
//
// **第二层（回滚行为，走真实入口）**：把真实训练入口 `train_agent_batch_side` 驱进
// 哨兵拒绝分支，断言
//   * `online` 整结构逐字节不变（CPU 路径是 `*own = *staged` 之前返回，影子不动）；
//   * `delta_ema` / `delta_warmup` / `committed_batches` 保持更新前值
//     （`updates(s) = gate_state_before` 把保护门基线一起回滚）；
//   * 只有 `rejected_batches` 与 `sentinel_consecutive_rejects` 各 +1；
//   * 拒绝原因字符串命中的是 **`after->nonfinite > 0`** 这一条，不是 DELTA 门
//     （`updates(s).rejected_delta_gate` 必须**不变**）。
//
// 注入的边界（不得含糊）：
//   * 注入符号 `sentinel_test_inject_after_nonfinite` **只在定义 SENTINEL_TEST_INJECT
//     的测试专用库里存在**（`build_verify/indep_p1_tests/inject_lib/DQN_Core.lib`，
//     由 `build_p1_libs.ps1` 生成）。生产库、Trainer、正式 EXE 都不定义该宏，
//     链接进去会直接 `LNK2001`，因此**没有生产运行开关**。
//   * 本测试**不冒充自然数值触发**：它不声称「训练自然会触发哨兵」，只声称
//     「拒绝分支一旦命中，回滚语义正确」。自然触发的样本另有说明（见文末）。
//   * 本测试**不冒充 GPU 数值验证**：显式只走 CPU 路径，并在输出里标注。
#include "core.h"
#include "independent_networks.h"
#include "sentinel_guard.h"
#include "skill_test.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace independent_training;

/* 注入符号只在测试专用库里存在；生产库没有它们。 */
extern "C" void sentinel_test_inject_after_nonfinite_n(int n);
extern "C" int sentinel_test_inject_remaining(void);

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); ++g_failures; } } while (0)
#define NEED(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); return 1; } } while (0)

/* ---------------- 第一层：判据与负例 ---------------- */

static SentinelHealth health(float max_abs_q, float min_margin, int nonfinite) {
    SentinelHealth h{};
    h.max_abs_q = max_abs_q;
    h.min_margin = min_margin;
    h.mean_margin = min_margin;
    h.nonfinite = nonfinite;
    h.correct_count = 0;   /* 刻意保持 0：证明它不参与判断 */
    h.total = 17;
    return h;
}

static int layer1() {
    char reason[160];
    printf("=== layer 1: sentinel_reject_update criteria (pure, no training) ===\n");

    /* (1) nonfinite > 0 → 拒绝（与包络无关）。 */
    {
        SentinelHealth b = health(1.0f, 1.0f, 0);
        SentinelHealth a = health(1.0f, 1.0f, 3);
        reason[0] = '\0';
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-1 nonfinite] rc=%d reason=\"%s\"\n", rc, reason);
        CHECK(rc == 1);
        CHECK(strstr(reason, "non-finite") != nullptr);
    }

    /* (2) max|Q| 越过 20 包络 **且** 同批实质突增 → 拒绝。 */
    {
        SentinelHealth b = health(5.0f, 1.0f, 0);
        SentinelHealth a = health(30.0f, 1.0f, 0);   /* 30 > 20，且 30 > 5+max(1,0.5)=6 */
        reason[0] = '\0';
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-2 max|Q| surge] rc=%d reason=\"%s\"\n", rc, reason);
        CHECK(rc == 1);
        CHECK(strstr(reason, "max|Q| surged") != nullptr);
    }

    /* (3) min_margin 跌破 −20 包络 **且** 同批实质塌陷 → 拒绝。 */
    {
        SentinelHealth b = health(5.0f, -5.0f, 0);
        SentinelHealth a = health(5.0f, -40.0f, 0);  /* −40 < −20，且 −40 < −5−max(1,0.5) */
        reason[0] = '\0';
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-3 margin collapse] rc=%d reason=\"%s\"\n", rc, reason);
        CHECK(rc == 1);
        CHECK(strstr(reason, "margin collapsed") != nullptr);
    }

    /* 负例 (a)：越过包络但同批突增不足 → 不拒绝。
       baseline 已越界（25 > 20），after 只涨到 25.5 < 25 + max(1, 2.5) = 27.5。 */
    {
        SentinelHealth b = health(25.0f, 1.0f, 0);
        SentinelHealth a = health(25.5f, 1.0f, 0);
        reason[0] = '\0';
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-neg-a envelope exceeded, small worsening] rc=%d reason=\"%s\" "
               "(baseline=25 envelope=20 allowance=max(1,0.1*25)=2.5 ⇒ 25.5 not enough)\n",
            rc, reason);
        CHECK(rc == 0);
        CHECK(reason[0] == '\0');
    }

    /* 负例 (b)：未越包络（|Q| ≤ 20 且 margin ≥ −20）→ 不拒绝，即使变化很大。 */
    {
        SentinelHealth b = health(1.0f, 1.0f, 0);
        SentinelHealth a = health(19.9f, -19.9f, 0);  /* 大幅变化但都没越包络 */
        reason[0] = '\0';
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-neg-b inside envelope, large change] rc=%d reason=\"%s\" "
               "(19.9 <= 20 and -19.9 >= -20)\n", rc, reason);
        CHECK(rc == 0);
        CHECK(reason[0] == '\0');
    }

    /* 边界值：恰好等于包络（> 20 才越界，== 20 不算）。 */
    {
        SentinelHealth b = health(1.0f, 1.0f, 0);
        SentinelHealth a = health(20.0f, -20.0f, 0);
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-boundary exact envelope] rc=%d (20 is not > 20; -20 is not < -20)\n", rc);
        CHECK(rc == 0);
    }

    /* 改善方向：已越界网络允许往回收（负例的另一种形式）。 */
    {
        SentinelHealth b = health(30.0f, -30.0f, 0);
        SentinelHealth a = health(28.0f, -28.0f, 0);
        const int rc = sentinel_reject_update(&b, &a, reason, sizeof(reason));
        printf("[L1-improving while out of envelope] rc=%d\n", rc);
        CHECK(rc == 0);
    }

    /* correct_count 不参与：把它设成极大/极小都不改变判定。 */
    {
        SentinelHealth b = health(5.0f, 1.0f, 0);
        SentinelHealth a0 = health(30.0f, 1.0f, 0); a0.correct_count = 0;
        SentinelHealth a1 = a0; a1.correct_count = 9999;
        SentinelHealth a2 = a0; a2.correct_count = -12345;
        char r0[160], r1[160], r2[160];
        const int rc0 = sentinel_reject_update(&b, &a0, r0, sizeof(r0));
        const int rc1 = sentinel_reject_update(&b, &a1, r1, sizeof(r1));
        const int rc2 = sentinel_reject_update(&b, &a2, r2, sizeof(r2));
        printf("[L1-correct_count ignored] rc(correct=0)=%d rc(9999)=%d rc(-12345)=%d\n", rc0, rc1, rc2);
        CHECK(rc0 == 1 && rc1 == 1 && rc2 == 1);
    }

    return 0;
}

/* ---------------- 第二层：真实入口的回滚行为 ---------------- */

static std::unique_ptr<DQNAgent> make_agent() {
    auto a = std::unique_ptr<DQNAgent>((DQNAgent*)calloc(1, sizeof(DQNAgent)));
    if (!a) return nullptr;
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    if (!indep_ensure(a.get())) return nullptr;
    a->indep->set_active(1);
    a->indep->set_bound(Side::Black);
    a->indep->seed_both_from(a->q_net, a->q_net);
    a->indep->set_learning_rate(Side::Black, 1e-4f);
    a->indep->set_learning_rate(Side::White, 1e-4f);
    return a;
}

/* 真实两 ply 样本（与 gpu_side_multibatch_parity_regression 同一构造思路）。 */
static int fill_batch(std::vector<Experience>& out, int samples) {
    out.assign(samples, Experience{});
    for (int i = 0; i < samples; ++i) {
        GameState g; memset(&g, 0, sizeof(g));
        g.turn = 0;
        g.board[0] = 1;
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = 2;
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&g, moves);
        if (n <= 0) return 0;
        const int action = moves[i % n];
        GameState mid = g; make_move(&mid, action);
        if (check_winner(&mid) != 0 || is_full(&mid)) return 0;
        int reply_moves[CELL_COUNT];
        const int rn = get_legal_moves(&mid, reply_moves);
        if (rn <= 0) return 0;
        const int reply = reply_moves[(i * 7 + 3) % rn];
        GameState after = mid; make_move(&after, reply);
        if (check_winner(&after) != 0 || is_full(&after)) return 0;
        Experience& e = out[i];
        state_to_input(&g, e.state);
        state_to_input(&after, e.next_state);
        e.action = action;
        e.reward = 0.05f;
        e.done = 0;
        if (!validate_experience(&e)) return 0;
    }
    return 1;
}

static int layer2() {
    printf("=== layer 2: rollback through the real entry train_agent_batch_side (CPU path) ===\n");

    const int COUNT = BATCH_SIZE;
    std::vector<Experience> batch;
    NEED(fill_batch(batch, COUNT));
    float states[BATCH_SIZE][CELL_COUNT + 1];
    int actions[BATCH_SIZE];
    float targets[BATCH_SIZE];
    for (int i = 0; i < COUNT; ++i) {
        memcpy(states[i], batch[i].state, sizeof(states[i]));
        actions[i] = batch[i].action;
        targets[i] = batch[i].reward + 0.25f * sinf((float)i);
        NEED(isfinite(targets[i]));
    }

    /* ---- 控制组：不注入，同一批必须正常提交（证明注入才是分歧来源）---- */
    {
        auto a = make_agent();
        NEED(a != nullptr);
        const NeuralNet before = a->indep->online(Side::Black);
        const SideUpdateState st_before = a->indep->updates(Side::Black);
        float loss = -1.0f;
        const int rc = train_agent_batch_side(a.get(), 0, (const float*)states, actions,
            targets, COUNT, 1e-4f, &loss);
        const SideUpdateState st_after = a->indep->updates(Side::Black);
        printf("[L2-control no injection] rc=%d loss=%.9g committed %llu->%llu rejected %llu->%llu "
               "delta_ema %.9g->%.9g warmup %d->%d sentinel_consec %d->%d delta_gate_rej %llu->%llu\n",
            rc, (double)loss, st_before.committed_batches, st_after.committed_batches,
            st_before.rejected_batches, st_after.rejected_batches,
            st_before.delta_ema, st_after.delta_ema, st_before.delta_warmup, st_after.delta_warmup,
            st_before.sentinel_consecutive_rejects, st_after.sentinel_consecutive_rejects,
            st_before.rejected_delta_gate, st_after.rejected_delta_gate);
        printf("[L2-control] online changed=%d\n",
            (int)(memcmp(&before, &a->indep->online(Side::Black), sizeof(NeuralNet)) != 0));
        NEED(rc == 1);
        NEED(memcmp(&before, &a->indep->online(Side::Black), sizeof(NeuralNet)) != 0);
        NEED(st_after.committed_batches == st_before.committed_batches + 1);
        NEED(st_after.rejected_batches == st_before.rejected_batches);
        NEED(st_after.rejected_delta_gate == st_before.rejected_delta_gate);
        free_dqn(a.get());
        g_current_agent = nullptr;
    }

    /* ---- 注入组：把真实入口驱进哨兵拒绝分支 ---- */
    {
        auto a = make_agent();
        NEED(a != nullptr);

        const NeuralNet before_online = a->indep->online(Side::Black);
        const NeuralNet before_target = a->indep->target(Side::Black);
        const SideUpdateState st_before = a->indep->updates(Side::Black);
        const NeuralNet white_before = a->indep->online(Side::White);
        const SideUpdateState white_state_before = a->indep->updates(Side::White);
        const unsigned long long guard_rejects_before = sentinel_reject_count();

        CHECK(sentinel_test_inject_remaining() == 0);
        /* 一批里评估两次（提交前 before + 提交后 after），注入 2 次使 after 必然命中
           `after->nonfinite > 0`；这与「哪一次先被调用」无关。 */
        sentinel_test_inject_after_nonfinite_n(2);
        CHECK(sentinel_test_inject_remaining() == 2);

        float loss = -1.0f;
        const int rc = train_agent_batch_side(a.get(), 0, (const float*)states, actions,
            targets, COUNT, 1e-4f, &loss);

        const SideUpdateState st_after = a->indep->updates(Side::Black);
        const int online_identical =
            memcmp(&before_online, &a->indep->online(Side::Black), sizeof(NeuralNet)) == 0;
        const int target_identical =
            memcmp(&before_target, &a->indep->target(Side::Black), sizeof(NeuralNet)) == 0;
        const int white_identical =
            memcmp(&white_before, &a->indep->online(Side::White), sizeof(NeuralNet)) == 0;
        printf("[L2-inject] rc=%d loss=%.9g training_fault=%ld\n", rc, (double)loss, (long)g_training_fault);
        printf("[L2-inject] online byte-identical=%d target byte-identical=%d other-side online identical=%d\n",
            online_identical, target_identical, white_identical);
        printf("[L2-inject] rejected_batches %llu->%llu (+%lld) sentinel_consecutive %d->%d (+%d)\n",
            st_before.rejected_batches, st_after.rejected_batches,
            (long long)(st_after.rejected_batches - st_before.rejected_batches),
            st_before.sentinel_consecutive_rejects, st_after.sentinel_consecutive_rejects,
            st_after.sentinel_consecutive_rejects - st_before.sentinel_consecutive_rejects);
        printf("[L2-inject] delta_ema %.9g->%.9g  delta_warmup %d->%d  committed %llu->%llu  "
               "delta_gate_rej %llu->%llu  loss_rej %llu->%llu  target_counter %d->%d\n",
            st_before.delta_ema, st_after.delta_ema, st_before.delta_warmup, st_after.delta_warmup,
            st_before.committed_batches, st_after.committed_batches,
            st_before.rejected_delta_gate, st_after.rejected_delta_gate,
            st_before.rejected_loss_batches, st_after.rejected_loss_batches,
            st_before.target_update_counter, st_after.target_update_counter);
        printf("[L2-inject] sentinel_reject_count %llu->%llu ; other side state unchanged=%d\n",
            guard_rejects_before, sentinel_reject_count(),
            (int)(memcmp(&white_state_before, &a->indep->updates(Side::White),
                sizeof(SideUpdateState)) == 0));
        printf("[L2-inject] injection consumed after one batch: remaining=%d\n",
            sentinel_test_inject_remaining());

        /* 1) 走的是哨兵拒绝（rc=2），不是 DELTA 门。 */
        CHECK(rc == 2);
        CHECK(loss == 0.0f);
        CHECK(g_training_fault == 0);
        /* 2) online 整结构逐字节不变（含 target / 另一方）。 */
        CHECK(online_identical);
        CHECK(target_identical);
        CHECK(white_identical);
        /* 3) 保护门基线一起回滚：EMA / 预热 / 提交计数保持更新前值。 */
        CHECK(st_after.delta_ema == st_before.delta_ema);
        CHECK(st_after.delta_warmup == st_before.delta_warmup);
        CHECK(st_after.committed_batches == st_before.committed_batches);
        CHECK(st_after.rejected_loss_batches == st_before.rejected_loss_batches);
        CHECK(st_after.target_update_counter == st_before.target_update_counter);
        /* 4) 只有拒绝统计变化：rejected_batches 与 sentinel_consecutive_rejects 各 +1，
              且 **DELTA 门拒绝计数不变**（区分哨兵拒绝与 DELTA 门拒绝，两者 rc 同为 2）。 */
        CHECK(st_after.rejected_batches == st_before.rejected_batches + 1);
        CHECK(st_after.sentinel_consecutive_rejects == st_before.sentinel_consecutive_rejects + 1);
        CHECK(st_after.rejected_delta_gate == st_before.rejected_delta_gate);
        CHECK(sentinel_reject_count() == guard_rejects_before + 1);
        /* 5) 另一方完全不受影响。 */
        CHECK(memcmp(&white_state_before, &a->indep->updates(Side::White),
            sizeof(SideUpdateState)) == 0);
        /* 6) 计数用完即自动解除，不会泄漏到后续批次。 */
        CHECK(sentinel_test_inject_remaining() == 0);

        /* 7) 注入解除后同一批必须能正常提交（回滚不是把这一方锁死）。 */
        {
            const SideUpdateState st2_before = a->indep->updates(Side::Black);
            float loss2 = -1.0f;
            const int rc2 = train_agent_batch_side(a.get(), 0, (const float*)states, actions,
                targets, COUNT, 1e-4f, &loss2);
            const SideUpdateState st2_after = a->indep->updates(Side::Black);
            printf("[L2-inject/after-disarm] rc=%d loss=%.9g committed %llu->%llu rejected %llu->%llu\n",
                rc2, (double)loss2, st2_before.committed_batches, st2_after.committed_batches,
                st2_before.rejected_batches, st2_after.rejected_batches);
            CHECK(rc2 == 1);
            CHECK(st2_after.committed_batches == st2_before.committed_batches + 1);
            CHECK(memcmp(&before_online, &a->indep->online(Side::Black), sizeof(NeuralNet)) != 0);
        }

        free_dqn(a.get());
        g_current_agent = nullptr;
    }

    return 0;
}

int main() {
    g_training_fault = 0;
    g_learning_rate = 0.0005f;
    sentinel_guard_init();

    const int rc1 = layer1();
    const int rc2 = layer2();

    printf("\n自然触发说明：本测试不构造自然触发样本；若将来构造，必须先记录实际命中的是\n"
           "nonfinite / max|Q| 突增 / min_margin 塌陷中的哪一条，并用 rejected_delta_gate\n"
           "与 sentinel_consecutive_rejects 区分 DELTA 门拒绝（rc 同为 2）。\n");

    const int total_failures = g_failures;
    printf("\n%s (%d failures)\n", total_failures == 0 ? "PASS" : "FAIL", total_failures);
    return (total_failures == 0 && rc1 == 0 && rc2 == 0) ? 0 : 1;
}

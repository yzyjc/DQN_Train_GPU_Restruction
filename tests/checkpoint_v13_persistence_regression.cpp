// 独立网络存档持久性回归（deepseek-flash）
//
// 为什么单独写一个：`tests/independent_checkpoint_v13_regression.cpp` **从不调用
// `TerminalOutcomeCredit::observe`**，而生产路径每个 ply 都会调用它。于是那份测试
// 漏掉了一个真实缺陷：
//
//   observe()（terminal_outcome_credit.h:20）无条件 `agent->checkpoint_version = 12;`
//   而 save_checkpoint()（checkpoint.cpp:494）只在 `< 12` 时才升级到 13
//   ⇒ 正式训练中 save_version 恒为 12 ⇒ 独立块不写 ⇒ 另一方网络被静默丢弃，
//     恢复时用共享 q_net 播种两方（HANDOFF:78 明文禁止的「复制黑方补白方」）。
//   载入 V13 后走一次 observe 再存，还会把已有 V13 降级成 V12。
//
// 本测试走**真实生产者**，因此它在缺陷存在时会失败，在修复后会通过。
#include "independent_networks.h"
#include "terminal_outcome_credit.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while (0)

namespace {

void perturb(DQNAgent* a, Side side, float delta) {
    a->indep->online(side).b4[0] += delta;
    a->indep->online(side).w4[0][0] += delta * 0.5f;
    a->indep->target(side).b4[2] -= delta * 0.25f;
    a->indep->updates(side).committed_batches += (unsigned long long)(side == Side::Black ? 11 : 22);
    a->indep->set_learning_rate(side, (side == Side::Black) ? 3e-4f : 4e-4f);
}

bool same_net(const NeuralNet& x, const NeuralNet& y) {
    return memcmp(&x, &y, sizeof(NeuralNet)) == 0;
}

/* 文件头前 8 字节：int32 version + uint32 magic（checkpoint.cpp:113 的读法）。 */
int file_version(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int32_t version = -1;
    uint32_t magic = 0;
    const bool ok = fread(&version, sizeof(version), 1, f) == 1 &&
                    fread(&magic, sizeof(magic), 1, f) == 1;
    fclose(f);
    if (!ok || magic != 0x5A4D4E54u) return -1;
    return version;
}

/* 走真实生产经验生产者：一手一手地下，让 observe() 处理 pending 与终局。 */
int drive_real_episode(DQNAgent* agent, int plies, int seed_offset) {
    GameState game;
    memset(&game, 0, sizeof(game));
    game.turn = 0;
    TerminalOutcomeCredit credit;
    int stored = 0;
    for (int i = 0; i < plies; ++i) {
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&game, moves);
        if (n <= 0) break;
        const int action = moves[(i + seed_offset) % n];
        GameState next = game;
        make_move(&next, action);
        const bool done = check_winner(&next) != 0 || is_full(&next);
        const float reward = compute_reward(agent, &game, &next, action, game.turn, 1, 0, 0);
        const int before = agent->replay_black_count + agent->replay_white_count;
        credit.observe(agent, game, next, action, reward, 1, 0);
        stored += (agent->replay_black_count + agent->replay_white_count) - before;
        game = next;
        if (done) break;
    }
    return stored;
}

} // namespace

int main() {
    /* ---- 1. 真实生产者会把 checkpoint_version 拉回 12（这是前提，先取证）---- */
    auto a = std::make_unique<DQNAgent>();
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    NEED(indep_ensure(a.get()) != nullptr);
    a->indep->set_active(1);
    a->indep->set_bound(Side::Black);
    perturb(a.get(), Side::Black, 1e-2f);
    perturb(a.get(), Side::White, -3e-2f);
    NEED(!same_net(a->indep->online(Side::Black), a->indep->online(Side::White)));

    const int stored = drive_real_episode(a.get(), 6, 0);
    printf("[observe] stored=%d experiences; checkpoint_version after real producer = %d\n",
        stored, a->checkpoint_version);
    NEED(stored > 0);
    /* 不变量：按方路由开启时，真实生产者必须把版本标记保持在 13。
       修复前这里是无条件 12，于是 save_version 退化成 12、独立块永不写入。 */
    NEED(a->checkpoint_version == 13);

    /* ---- 2. 保存必须仍然写出 V13 独立块（缺陷存在时这里就是 12）---- */
    const NeuralNet b_online = a->indep->online(Side::Black);
    const NeuralNet b_target = a->indep->target(Side::Black);
    const NeuralNet w_online = a->indep->online(Side::White);
    const NeuralNet w_target = a->indep->target(Side::White);

    NEED(save_checkpoint(a.get(), 1, 4242, 2, "checkpoints/checkpoint_phase1_4242.bin"));
    const int v1 = file_version("checkpoints/checkpoint_phase1_4242.bin");
    printf("[save#1] file version = %d (must be 13), agent->checkpoint_version = %d\n",
        v1, a->checkpoint_version);
    NEED(v1 == 13);
    NEED(a->checkpoint_version == 13);

    /* ---- 3. 精确恢复：两方都必须回来，且互不相同 ---- */
    {
        auto b = std::make_unique<DQNAgent>();
        init_dqn_ex(b.get(), 0);
        g_current_agent = b.get();
        NEED(indep_ensure(b.get()) != nullptr);
        int phase = 0, ep = 0, depth = 0;
        NEED(load_checkpoint(b.get(), &phase, &ep, &depth,
            "checkpoints/checkpoint_phase1_4242.bin", 1));
        NEED(phase == 1 && ep == 4242 && depth == 2);
        NEED(b->checkpoint_version == 13);
        NEED(indep_active(b.get()));
        NEED(same_net(b->indep->online(Side::Black), b_online));
        NEED(same_net(b->indep->target(Side::Black), b_target));
        NEED(same_net(b->indep->online(Side::White), w_online));
        NEED(same_net(b->indep->target(Side::White), w_target));
        NEED(b->indep->online(Side::Black).b4[0] != b->indep->online(Side::White).b4[0]);
        printf("[load] two sides restored byte-identical and mutually distinct; "
            "black.b4[0]=%.9g white.b4[0]=%.9g\n",
            (double)b->indep->online(Side::Black).b4[0],
            (double)b->indep->online(Side::White).b4[0]);

        /* ---- 4. 已恢复的 V13 再走一次真实生产者，再存必须仍是 13（不许降级）---- */
        const int stored2 = drive_real_episode(b.get(), 6, 2);
        NEED(stored2 > 0);
        NEED(b->checkpoint_version == 13);   /* 已恢复的 V13 不得被生产者拉回 12 */
        NEED(save_checkpoint(b.get(), 1, 4243, 2, "checkpoints/checkpoint_phase1_4243.bin"));
        const int v2 = file_version("checkpoints/checkpoint_phase1_4243.bin");
        printf("[save#2 after reload] file version = %d (must stay 13)\n", v2);
        NEED(v2 == 13);
    }

    /* ---- 5. 遗留共享模式必须仍然写 V12（不能因为上面这条改动而误升 13）---- */
    {
        auto c = std::make_unique<DQNAgent>();
        init_dqn_ex(c.get(), 0);
        g_current_agent = c.get();
        NEED(indep_ensure(c.get()) != nullptr);   /* 容器存在但 active==0 */
        NEED(!indep_active(c.get()));
        const int stored3 = drive_real_episode(c.get(), 6, 1);
        NEED(stored3 > 0);
        NEED(c->checkpoint_version == 12);
        NEED(save_checkpoint(c.get(), 1, 4244, 2, "checkpoints/checkpoint_phase1_4244.bin"));
        const int v3 = file_version("checkpoints/checkpoint_phase1_4244.bin");
        printf("[save#3 legacy] file version = %d (must be 12)\n", v3);
        NEED(v3 == 12);
    }

    puts("PASS checkpoint persistence with the real experience producer: independent mode always "
         "writes V13 (both sides persisted), reload restores two distinct sides, a reloaded "
         "archive is never silently downgraded to V12, and legacy shared mode still writes V12");
    return 0;
}

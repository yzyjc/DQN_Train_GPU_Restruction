// 两 ply 契约的守卫回归（deepseek-flash）
//
// 为什么单独写：`tests/independent_side_isolation_regression.cpp` 只证明「合法样本能训练」，
// 没有证明「不合法样本会被拦下」。跨度守卫如果写成空操作（或 observed_plies 判错），
// 那份隔离测试照样全绿。这里把守卫本身钉死：
//
//   1. 独立模式 + 两 ply 样本  -> 正常训练（loss > 0，不闩死）
//   2. 遗留共享模式 + 一 ply 样本 -> 正常训练（历史行为逐位不变，不闩死）
//   3. 独立模式 + 一 ply 样本  -> 必须返回 0 且闩死 g_training_fault
//
// 第 3 条会闩住全局训练故障，所以放在最后。
#include "independent_networks.h"
#include "sentinel_guard.h"
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while (0)

namespace {

const float PLACED[2] = { 1.0f, -1.0f };

/* 一 ply：本方落子之后立刻收尾（对手尚未回应）。 */
void fill_one_ply(std::vector<Experience>& out, int side, int samples) {
    out.assign(samples, Experience{});
    for (int i = 0; i < samples; ++i) {
        GameState g;
        memset(&g, 0, sizeof(g));
        g.turn = side;
        g.board[side == 0 ? 0 : 24] = PLACED[side];
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = -PLACED[side];
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&g, moves);
        if (n <= 0) { out[i].action = -1; continue; }
        const int action = moves[i % n];
        GameState mid = g;
        make_move(&mid, action);
        if (check_winner(&mid) != 0 || is_full(&mid)) { out[i].action = -1; continue; }
        Experience& e = out[i];
        state_to_input(&g, e.state);
        state_to_input(&mid, e.next_state);
        e.action = action;
        e.reward = (side == 0) ? 0.05f : 0.02f;
        e.done = 0;
        if (!validate_experience(&e)) out[i].action = -1;
    }
}

/* 两 ply：本方落子 + 对手真实回应。 */
void fill_two_ply(std::vector<Experience>& out, int side, int samples) {
    out.assign(samples, Experience{});
    for (int i = 0; i < samples; ++i) {
        GameState g;
        memset(&g, 0, sizeof(g));
        g.turn = side;
        g.board[side == 0 ? 0 : 24] = PLACED[side];
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = -PLACED[side];
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&g, moves);
        if (n <= 0) { out[i].action = -1; continue; }
        const int action = moves[i % n];
        GameState mid = g;
        make_move(&mid, action);
        if (check_winner(&mid) != 0 || is_full(&mid)) { out[i].action = -1; continue; }
        int reply_moves[CELL_COUNT];
        const int rn = get_legal_moves(&mid, reply_moves);
        if (rn <= 0) { out[i].action = -1; continue; }
        const int reply = reply_moves[(i * 7 + 3) % rn];
        GameState after = mid;
        make_move(&after, reply);
        if (check_winner(&after) != 0 || is_full(&after)) { out[i].action = -1; continue; }
        Experience& e = out[i];
        state_to_input(&g, e.state);
        state_to_input(&after, e.next_state);
        e.action = action;
        e.reward = (side == 0) ? 0.05f : 0.02f;
        e.done = 0;
        if (!validate_experience(&e)) out[i].action = -1;
    }
}

int put_replay(DQNAgent* agent, std::vector<Experience>& src, int side) {
    if (src.size() > (size_t)REPLAY_CAPACITY) return 0;
    Experience* buf = (side == 0) ? agent->replay_black : agent->replay_white;
    int* idx = (side == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
    int* cnt = (side == 0) ? &agent->replay_black_count : &agent->replay_white_count;
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i].action < 0) return 0;
        buf[i] = src[i];
    }
    *idx = 0;
    *cnt = (int)src.size();
    return 1;
}

std::unique_ptr<DQNAgent> make_agent(int indep_on) {
    auto a = std::make_unique<DQNAgent>();
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    if (!indep_ensure(a.get())) return nullptr;
    if (indep_on) {
        a->indep->set_active(1);
        a->indep->set_bound(Side::Black);
    }
    return a;
}

float train_black(DQNAgent* a) {
    return replay_from_buffer(a, a->replay_black, a->replay_black_count, &a->replay_black_idx);
}

double count_plies(const Experience& e) {
    int a = 0, b = 0;
    for (int i = 0; i < CELL_COUNT; ++i) {
        if (e.state[i] != 0.0f) ++a;
        if (e.next_state[i] != 0.0f) ++b;
    }
    return (double)(b - a);
}

} // namespace

int main() {
    g_learning_rate = 0.0001f;
    sentinel_guard_init();

    /* ---- 1. 独立模式 + 两 ply：必须正常训练 ---- */
    {
        auto a = make_agent(1);
        NEED(a != nullptr);
        std::vector<Experience> two;
        fill_two_ply(two, 0, 2 * BATCH_SIZE);
        NEED(put_replay(a.get(), two, 0));
        NEED(count_plies(a->replay_black[0]) == 2.0);
        const float loss = train_black(a.get());
        printf("[leg1 indep/2ply] loss=%.9g training_fault=%ld\n", (double)loss, (long)g_training_fault);
        NEED(g_training_fault == 0);
        NEED(isfinite(loss) && loss > 0.0f);
    }

    /* ---- 2. 遗留共享模式 + 一 ply：历史行为必须保持（不闩死）---- */
    {
        auto b = make_agent(0);
        NEED(b != nullptr);
        NEED(!indep_active(b.get()));
        std::vector<Experience> one;
        fill_one_ply(one, 0, 2 * BATCH_SIZE);
        NEED(put_replay(b.get(), one, 0));
        NEED(count_plies(b->replay_black[0]) == 1.0);
        const float loss = train_black(b.get());
        printf("[leg2 legacy/1ply] loss=%.9g training_fault=%ld\n", (double)loss, (long)g_training_fault);
        NEED(g_training_fault == 0);
        NEED(isfinite(loss) && loss > 0.0f);
    }

    /* ---- 3. 独立模式 + 一 ply：必须被跨度守卫拦下并闩死（放在最后）---- */
    {
        auto c = make_agent(1);
        NEED(c != nullptr);
        std::vector<Experience> one;
        fill_one_ply(one, 0, 2 * BATCH_SIZE);
        NEED(put_replay(c.get(), one, 0));
        NEED(count_plies(c->replay_black[0]) == 1.0);
        NEED(validate_experience(&c->replay_black[0]));   /* 结构上合法，只是跨度不对 */
        const float loss = train_black(c.get());
        printf("[leg3 indep/1ply] loss=%.9g training_fault=%ld (must be 0 and nonzero)\n",
            (double)loss, (long)g_training_fault);
        NEED(loss == 0.0f);
        NEED(g_training_fault != 0);
    }

    puts("PASS replay span contract: 2-ply trains under independent mode, 1-ply keeps training in "
         "legacy shared mode, and 1-ply is rejected with a latched fault under independent mode");
    return 0;
}

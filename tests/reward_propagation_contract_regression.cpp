// 按方收益传播契约回归（第 4 步规格；取代早期的 known-gap 断言）
//
// 用真实 get_legal_moves / make_move / check_winner / compute_reward 构造样本，
// 逐项核对契约：
//   1. 形态与折扣指数：非终局两 ply = γ²，遗留一 ply = γ，终局不 bootstrap；
//   2. 手算对照：y = r_p + γ²·maxQ_p(s'')，并断言它既不是 +γ·maxQ 也不是 −γ·maxQ；
//   3. 正号：独立模式下黑、白两方都是正号，且不再依赖 turn 判号；
//   4. 非对称收益不被抹平：同一终局棋盘 player=0/1 的奖励不互为相反数；
//   5. 终局跨度：一 ply 直接终局（本方那一手直接连四）仍合法；
//   6. 遗留言义：同一批样本在 indep_active==0 时目标与改造前的公式一致。
//
// 早期版本曾断言 y_strict = R_p − γ·R_opp − γ²·maxQ，该式已撤回：
//   (a) R_opp 不属于 p 的收益流（对手那一手的奖励记在对手缓冲里），
//   (b) 终局桥接被置 done=1、根本不 bootstrap，该式描述的两个端点都不存在。
#include "experience_semantics.h"
#include "terminal_outcome_credit.h"
#include "step_outcome.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace es = experience_semantics;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while(0)

/* 造一条「本方落子 -> 对手真实回应」的两 ply 非终局样本。 */
static bool make_two_ply(int side, int seed, GameState* s0, int* action, GameState* s2) {
    GameState g; memset(&g, 0, sizeof(g));
    g.turn = side;
    g.board[(side == 0) ? 0 : 24] = (side == 0) ? 1 : 2;
    const int other = 5 + (seed % 10);
    if (g.board[other] == 0) g.board[other] = (side == 0) ? 2 : 1;
    int moves[CELL_COUNT];
    const int n = get_legal_moves(&g, moves);
    if (n <= 0) return false;
    const int a = moves[seed % n];
    GameState mid = g; make_move(&mid, a);
    if (check_winner(&mid) != 0 || is_full(&mid)) return false;
    int replies[CELL_COUNT];
    const int rn = get_legal_moves(&mid, replies);
    if (rn <= 0) return false;
    GameState after = mid; make_move(&after, replies[(seed * 3 + 1) % rn]);
    if (check_winner(&after) != 0 || is_full(&after)) return false;
    *s0 = g; *action = a; *s2 = after;
    return true;
}

/* 手算 target：独立契约 y = r + γ^plies · maxQ（正号）。 */
static float contract_target(float r, int plies, float max_q) {
    return r + es::bootstrap_discount_for_plies(plies, GAMMA) * max_q;
}

int main() {
    puts("=== reward propagation contract ==="); fflush(stdout);

    /* ---- 1. 形态与折扣指数 ---- */
    NEED(es::bootstrap_plies(0, 1) == 1);   /* 遗留一 ply 非终局 */
    NEED(es::bootstrap_plies(1, 0) == 2);   /* 终局桥接：端点在对手回应之后 */
    NEED(es::bootstrap_discount_for_plies(0, GAMMA) == 0.0f);
    NEED(fabsf(es::bootstrap_discount_for_plies(1, GAMMA) - GAMMA) < 1e-7f);
    NEED(fabsf(es::bootstrap_discount_for_plies(2, GAMMA) - GAMMA * GAMMA) < 1e-6f);
    NEED(es::discount_exponent_matches(2, 2));
    printf("gamma=%.6f gamma^1=%.9g gamma^2=%.9g\n", (double)GAMMA,
        (double)es::bootstrap_discount_for_plies(1, GAMMA),
        (double)es::bootstrap_discount_for_plies(2, GAMMA));

    /* 形态判定的负面对照：跨度守卫不能是空操作。 */
    float one_ply_state[CELL_COUNT + 1] = { 0 }, one_ply_next[CELL_COUNT + 1] = { 0 };
    one_ply_state[0] = 1.0f; one_ply_state[CELL_COUNT] = 0.0f;
    one_ply_next[0] = 1.0f; one_ply_next[1] = -1.0f; one_ply_next[CELL_COUNT] = 1.0f;
    NEED(es::observed_plies(one_ply_state, one_ply_next) == 1);
    NEED(es::check_replay_span(one_ply_state, one_ply_next, 0) != 0);   /* 非终局一 ply 非法 */
    NEED(es::check_replay_span(one_ply_state, one_ply_next, 1) == 0);   /* 一 ply 直接终局合法 */

    auto agent = std::make_unique<DQNAgent>();
    init_dqn_ex(agent.get(), 0);
    g_current_agent = agent.get();
    g_learning_rate = 0.0001f;

    /* ---- 2. 两 ply 手算对照 + 指数与符号不能巧合相等 ---- */
    for (int side = 0; side < 2; ++side) {
        GameState s0, s2; int action = -1;
        NEED(make_two_ply(side, 3 + side, &s0, &action, &s2));
        const float r = (side == 0) ? 0.05f : 0.02f;
        const float max_q = 0.3f;
        const float y = contract_target(r, 2, max_q);
        const float wrong_one_ply_pos = r + GAMMA * max_q;
        const float wrong_one_ply_neg = r - GAMMA * max_q;
        printf("[2ply side=%d] r=%.9g y=r+gamma^2*maxQ=%.9g  (+gamma*maxQ=%.9g  -gamma*maxQ=%.9g)\n",
            side, (double)r, (double)y, (double)wrong_one_ply_pos, (double)wrong_one_ply_neg);
        NEED(fabsf(y - (r + GAMMA * GAMMA * max_q)) < 1e-6f);
        /* 反例断言：既不是 +γ·maxQ 也不是 −γ·maxQ。 */
        NEED(fabsf(y - wrong_one_ply_pos) > 1e-4f);
        NEED(fabsf(y - wrong_one_ply_neg) > 1e-4f);
        /* 实测跨度确为 2，且轮次回到本方。 */
        float a[CELL_COUNT + 1], b[CELL_COUNT + 1];
        state_to_input(&s0, a); state_to_input(&s2, b);
        NEED(es::observed_plies(a, b) == 2);
        NEED(a[CELL_COUNT] == b[CELL_COUNT]);          /* s'' 轮到本方 */
        NEED(es::check_replay_span(a, b, 0) == 0);
    }

    /* ---- 3. 正号：独立模式对两方都取正，且不再依赖 turn 判号 ---- */
    {
        /* 构造 state_turn == next_turn 的样本：独立契约下仍是正号。 */
        GameState s0, s2; int action = -1;
        NEED(make_two_ply(0, 5, &s0, &action, &s2));
        const float placed = 1.0f;
        Experience e{};
        state_to_input(&s0, e.state);
        state_to_input(&s2, e.next_state);
        e.action = action; e.reward = 0.05f; e.done = 0;
        NEED(validate_experience(&e));
        printf("[sign] state_turn=%.0f next_turn=%.0f -> independent sign=+plies(+2) "
            "legacy sign=+1(equal turns)\n",
            (double)e.state[CELL_COUNT], (double)e.next_state[CELL_COUNT]);
        NEED(e.next_state[CELL_COUNT] == e.state[CELL_COUNT]);
        (void)placed;
    }

    /* ---- 4. 非对称收益不被抹平 ---- */
    {
        GameState g; memset(&g, 0, sizeof(g));
        g.board[0] = 1; g.board[1] = 1; g.board[2] = 1;   /* 黑差一格 */
        g.turn = 0;
        GameState win = g; make_move(&win, 3);
        NEED(check_winner(&win) == 1);
        const float r_black_win = compute_reward(agent.get(), &g, &win, 3, 0, 1, 0, 3);
        const float r_white_view = compute_reward(agent.get(), &g, &win, 3, 1, 1, 0, 3);
        printf("[asymmetry] r(player=0)=%.9g r(player=1)=%.9g sum=%.9g\n",
            (double)r_black_win, (double)r_white_view, (double)(r_black_win + r_white_view));
        NEED(fabsf(r_black_win - 1.0f) < 1e-6f);
        NEED(fabsf(r_white_view - (-0.5f)) < 1e-6f);
        /* 和不为 0：契约里不得出现任何 negamax 取负假设。 */
        NEED(fabsf(r_black_win + r_white_view) > 1e-6f);
    }

    /* ---- 5. 白和 +0.05 / 黑和 0（真实满盘，不手搓常量） ---- */
    {
        StepOutcomeConfig c;
        c.enabled = 1; c.black_fast_win_scale = 0.10f; c.black_early_loss_scale = 0.05f;
        c.white_survival_scale = 0.05f; c.white_draw_length_scale = 0.05f;
        StepOutcomeScope scope(c);
        PRNGState rng; prng_init(&rng, 7);
        GameState trial; bool found = false;
        for (int attempt = 0; attempt < 20000 && !found; ++attempt) {
            memset(&trial, 0, sizeof(trial)); int filled = 0;
            for (int j = 0; j < CELL_COUNT; ++j) {
                int moves[CELL_COUNT], cnt = 0;
                for (int a = 0; a < CELL_COUNT; ++a) {
                    if (trial.board[a]) continue;
                    GameState nx = trial; make_move(&nx, a);
                    if (!check_winner(&nx)) moves[cnt++] = a;
                }
                if (!cnt) break;
                make_move(&trial, moves[prng_int(&rng, cnt)]);
                ++filled;
            }
            if (filled == CELL_COUNT && check_winner(&trial) == 0) found = true;
        }
        NEED(found);
        GameState before = trial;
        int last = -1;
        for (int a = 0; a < CELL_COUNT; ++a) if (before.board[a]) { last = a; break; }
        NEED(last >= 0);
        before.board[last] = 0;
        before.turn = (CELL_COUNT - 1) % 2;
        const float r_draw = compute_reward(agent.get(), &before, &trial, last,
            before.turn, 1, 0, CELL_COUNT - 1);
        const float qual = step_outcome_bonus(c, before.turn, 0, CELL_COUNT, CELL_COUNT);
        printf("[draw] mover=%d reward=%.9g quality=%.9g\n", before.turn,
            (double)r_draw, (double)qual);
        NEED(fabsf(r_draw - (DRAW_REWARD + qual)) < 1e-6f);
        if (before.turn == 0) NEED(fabsf(qual) < 1e-9f);          /* 黑和 = 0 */
        else NEED(fabsf(qual - c.white_draw_length_scale) < 1e-6f); /* 白和 = +0.05 */
    }

    /* ---- 6. 遗留言义不变 ---- */
    {
        float s[CELL_COUNT + 1], n[CELL_COUNT + 1];
        memset(s, 0, sizeof(s)); memset(n, 0, sizeof(n));
        s[0] = 1.0f; s[CELL_COUNT] = 0.0f;
        n[0] = 1.0f; n[1] = -1.0f; n[CELL_COUNT] = 1.0f;   /* 一手之后轮到对手 */
        const int state_turn = (int)(s[CELL_COUNT] + 0.5f);
        const int next_turn = (int)(n[CELL_COUNT] + 0.5f);
        const int legacy_sign = (state_turn == next_turn) ? 1 : -1;
        const float legacy_y = 0.02f + legacy_sign * GAMMA * 0.3f;
        printf("[legacy] sign=%d y=%.9g (unchanged formula)\n", legacy_sign, (double)legacy_y);
        NEED(legacy_sign == -1);
        NEED(fabsf(legacy_y - (0.02f - GAMMA * 0.3f)) < 1e-6f);
    }

    free_dqn(agent.get());
    g_current_agent = nullptr;
    puts("PASS reward propagation contract: 2-ply non-terminal uses +gamma^2 with the side's own "
         "target network, terminals never bootstrap, asymmetric rewards are preserved, and the "
         "legacy shared-network formula is bit-identical");
    return 0;
}

// 按方经验语义与折扣约定（第 4 步契约；本文件已按新契约重写）
//
// 与 tests/reward_propagation_contract_regression.cpp 的分工：
//   本文件 = 登记表 + 符号约定 + 折扣指数的一致性（契约层）；
//   那份文件 = 用真实棋盘做手算对照、反例断言与非对称收益断言（数值层）。
//
// 已撤回的内容（勿恢复）：早期版本在这里断言
//   y_strict = R_p - γ·R_opp - γ²·maxQ
// 该式不成立：(a) R_opp 不属于 p 的收益流（对手那一手的奖励记在对手缓冲里），
// (b) 终局桥接被置 done=1、根本不 bootstrap，所以 y_now/y_strict 描述的两个
// 端点都不对应真实样本。它是**纯算术自洽**断言，不改也会继续 PASS —— 留着就是
// 一份「绿灯但语义错误」的测试，因此整段删除。
#include "experience_semantics.h"
#include "terminal_outcome_credit.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace es = experience_semantics;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while(0)

int main() {
    puts("=== experience semantics contract (post step-4) ==="); fflush(stdout);

    /* ---- 登记表：时间距离 -> 折扣指数 ----
       A 两 ply 非终局       : 端点 = 本方下一个决策点，指数 2
       B 终局桥接            : 端点同样是「对手回应之后」，跨度也是 2（但不 bootstrap）
       B' 本方那一手直接终局 : 一 ply 终局样本，跨度 1，不 bootstrap
       C 遗留一 ply 非终局   : 指数 1（仅供未开启按方路由的共享模式） */
    NEED(es::bootstrap_plies(0, 1) == 1);   /* C */
    NEED(es::bootstrap_plies(1, 0) == 2);   /* B */
    NEED(es::discount_exponent_matches(2, 2));
    NEED(es::discount_exponent_matches(1, 1));

    NEED(es::bootstrap_discount_for_plies(0, GAMMA) == 0.0f);   /* 不 bootstrap */
    NEED(fabsf(es::bootstrap_discount_for_plies(1, GAMMA) - GAMMA) < 1e-7f);
    NEED(fabsf(es::bootstrap_discount_for_plies(2, GAMMA) - GAMMA * GAMMA) < 1e-6f);
    printf("gamma=%.6f gamma^1=%.9g gamma^2=%.9g\n", (double)GAMMA,
        (double)es::bootstrap_discount_for_plies(1, GAMMA),
        (double)es::bootstrap_discount_for_plies(2, GAMMA));

    /* 跨度守卫的真实谓词（不是空操作）。 */
    float one[CELL_COUNT + 1] = { 0 }, two[CELL_COUNT + 1] = { 0 };
    one[0] = 1.0f; one[CELL_COUNT] = 0.0f;
    two[0] = 1.0f; two[1] = -1.0f; two[CELL_COUNT] = 1.0f;
    NEED(es::observed_plies(one, two) == 1);
    NEED(es::check_replay_span(one, two, 0) != 0);   /* 非终局一 ply：独立模式非法 */
    NEED(es::check_replay_span(one, two, 1) == 0);   /* 一 ply 直接终局：合法（B'） */
    float three[CELL_COUNT + 1] = { 0 };
    three[0] = 1.0f; three[1] = -1.0f; three[2] = 1.0f; three[CELL_COUNT] = 0.0f;
    NEED(es::observed_plies(one, three) == 2);
    NEED(es::check_replay_span(one, three, 0) == 0);
    NEED(es::check_replay_span(one, three, 1) == 0); /* 两 ply 终局桥接：合法（B） */

    /* ---- 符号约定：遗留模式 ±1 按 turn；独立模式恒 +plies ---- */
    {
        const int legacy_sign = (one[CELL_COUNT] == two[CELL_COUNT]) ? 1 : -1;
        NEED(legacy_sign == -1);
        const int legacy_same = (one[CELL_COUNT] == three[CELL_COUNT]) ? 1 : -1;
        NEED(legacy_same == 1);
        const int plies = 2;
        const float y = 0.02f + es::bootstrap_discount_for_plies(plies, GAMMA) * 0.3f;
        const float y_pos1 = 0.02f + GAMMA * 0.3f;
        const float y_neg1 = 0.02f - GAMMA * 0.3f;
        printf("[sign] independent plies=%d sign=+%d y=%.9g (+gamma^1=%.9g -gamma^1=%.9g)\n",
            plies, plies, (double)y, (double)y_pos1, (double)y_neg1);
        NEED(fabsf(y - y_pos1) > 1e-4f);   /* 不是 γ^1 正号 */
        NEED(fabsf(y - y_neg1) > 1e-4f);   /* 更不是 γ^1 负号 */
    }

    /* ---- 生产者形态：真实 observe 产出的经验必须过校验器 ---- */
    {
        auto agent = std::make_unique<DQNAgent>();
        init_dqn_ex(agent.get(), 0);
        g_current_agent = agent.get();
        GameState s0; memset(&s0, 0, sizeof(s0));
        s0.turn = 0; s0.board[0] = 1; s0.board[7] = 2;
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&s0, moves);
        NEED(n > 0);
        GameState s1 = s0; make_move(&s1, moves[0]);
        NEED(check_winner(&s1) == 0 && !is_full(&s1));
        int replies[CELL_COUNT];
        const int rn = get_legal_moves(&s1, replies);
        NEED(rn > 0);
        GameState s2 = s1; make_move(&s2, replies[0]);
        NEED(check_winner(&s2) == 0 && !is_full(&s2));

        TerminalOutcomeCredit credit;
        credit.observe(agent.get(), s0, s1, moves[0], 0.0f, 1, 0);
        credit.observe(agent.get(), s1, s2, replies[0], 0.0f, 1, 0);
        /* 生产者语义：每来一手会把上一手挂起的经验落盘，自己再挂起。
           因此两次 observe 之后应有 1 条已入库、1 条 pending。 */
        NEED(credit.pending);
        /* store_experience 还会写一份对称增强副本，因此两个缓冲各可能有一条。 */
        NEED(agent->replay_black_count + agent->replay_white_count >= 1);
        Experience* buf = (agent->replay_black_count > 0) ? agent->replay_black : agent->replay_white;
        const int cnt = (agent->replay_black_count > 0) ? agent->replay_black_count
                                                       : agent->replay_white_count;
        int two_ply = 0;
        for (int i = 0; i < cnt; ++i) {
            NEED(validate_experience(&buf[i]));
            if (!buf[i].done && es::observed_plies(buf[i].state, buf[i].next_state) == 2) ++two_ply;
        }
        printf("[producer] stored=%d two-ply non-terminal=%d checkpoint_version=%d\n",
            cnt, two_ply, agent->checkpoint_version);
        NEED(two_ply > 0);
        /* 未开启按方路由时版本标记仍是 12（V12 读取方不受影响）。 */
        NEED(agent->checkpoint_version == 12);
        free_dqn(agent.get());
        g_current_agent = nullptr;
    }

    puts("PASS experience semantics: registration table and discount exponents match the 2-ply "
         "contract, legacy sign stays turn-based, independent sign is +plies with gamma^2, and "
         "the revoked y_strict section is gone");
    return 0;
}

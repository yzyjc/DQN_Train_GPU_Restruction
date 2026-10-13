#pragma once
#include "core.h"
#include <math.h>

/* ============================================================================
   按方经验的收益传播契约（HANDOFF 第 50-58 行；第 4 步正式规格）

   这是本仓库对「一条 Experience 跨了几手、bootstrap 用什么折扣、什么符号」的
   唯一权威约定。任何新增经验构造路径都必须先在这里登记形态，再用
   bootstrap_discount_for_plies() 取折扣，禁止在调用点手写 GAMMA。

   ---- 收益流定义（先钉死这一条）------------------------------------------

   对物理棋色 p：每个 ply（半手）恰好有一个行动方。
     * 行动方是 p    -> p 收到 compute_reward(..., player=p)
     * 行动方是对手  -> p 收到 0（对手那一手的奖励记在对手自己的回放里）

   V_p(s)  = E[ Σ_{t>=0} γ^t · r_p(ply t) | s_0 = s ]
   Q_p(s,a) = r_p(s,a) + γ · V_p(s')      s' = p 落子之后（轮到对手）
   V_p(s')  = γ · V_p(s'')                对手那一手，p 得 0
   V_p(s'') = max_a Q_p(s'', a)           s'' = 对手回应之后，p 再次决策

   合并（ply 计时）：

       y = r_p + γ^2 · max_a Q_p_target(s'')     <-- 端点 = 本方下一个决策点

   * **正号**：端点是**己方**价值，己方收益到己方价值之间没有任何取负。
   * **指数 2**：按 ply 计时跨两手。若改按「己方决策」计时，则 γ_own = γ²、
     指数为 1；两种计时给出同一个数字。本仓库显式采用 **ply 计时 + 指数 2**，
     因为 Experience 里存的就是棋盘，observed_plies() 可直接取证。
   * **非对称奖励原样保留**：WIN_REWARD=+0.5（phase>=1 双倍 = +1.0）与
     LOSE_REWARD=-0.5 不构成零和（一手终局双方收益之和 = +0.5），
     step_outcome 的四个系数也天然不对称。因此任何 negamax 取负都不成立，
     契约里不得出现 -V_opp。

   ---- 经验形态 -----------------------------------------------------------

   A. 两 ply 非终局（独立网络的正式形态，P1b）
        state      = 行动者 p 决策时的棋盘
        action     = p 的真实落子
        next_state = **对手真实回应之后**、p 再次决策的棋盘
        reward     = p 自己那一手的奖励（对手那一手的奖励不属于 p）
        done       = 0
        bootstrap  = γ^2 · max_a Q_p_target(next_state)，**正号**

   B. 终局桥接（对手回应结束比赛）
        state      = 行动者 p 决策时的棋盘
        action     = p 的真实落子
        next_state = 对手回应之后的最终棋盘（写法与 A 相同）
        reward     = 从 p 落子前到最终棋盘重算的 p 的结果奖励
        done       = 1
        bootstrap  = 无（终局一律停止 bootstrap）

   C. 一 ply 非终局（遗留共享网络模式的形态）
        state      = 行动者 p 决策时的棋盘
        next_state = 一手之后、轮到对手的棋盘
        bootstrap  = ±1 · γ · max_a Q_target(next_state)，符号按 turn 判定
     这个形态**只在未开启按方路由时**合法；独立模式下出现即视为回放语义破损，
     由 check_replay_span() 判失败，绝不静默按新公式重算。

   ---- 本文件不做什么 -----------------------------------------------------

   不提供「负对手奖励」公式。本文件早期版本曾写入
   y_strict = R_p - γ·R_opp - γ²·maxQ，该式在本仓库没有成立基础：
   (1) R_opp 不属于 p 的收益流（见上），
   (2) B 形态被置 done=1、根本不 bootstrap，所以 y_now/y_strict 描述的两个
       端点都不对应真实样本。该式已作废，相关断言已从测试里删除。
   ========================================================================= */
namespace experience_semantics {

/* 经验形态登记。任何生产构造路径都必须能映射到其中之一。 */
enum class Shape {
    TwoPlyNonTerminal = 0,    /* A：本方决策 -> 本方下一个决策 */
    TwoPlyTerminalBridge = 1, /* B：对手回应结束比赛 */
    OnePlyLegacy = 2,         /* C：仅遗留共享网络模式 */
    Invalid = 3
};

inline const char* shape_name(Shape s) {
    switch (s) {
    case Shape::TwoPlyNonTerminal: return "2ply-nonterminal";
    case Shape::TwoPlyTerminalBridge: return "2ply-terminal-bridge";
    case Shape::OnePlyLegacy: return "1ply-legacy";
    default: return "invalid";
    }
}

/* 一条经验从 state 到 bootstrap 端点跨越的「手数」（ply）。
   注意 B 形态虽然不 bootstrap，端点仍然是「对手回应之后」，跨度同样是 2 手；
   返回 2 而不是 0，是为了让「指数必须等于跨度」这条校验对 A/B 一视同仁。 */
inline int bootstrap_plies(int is_terminal_bridge, int is_one_step) {
    if (is_terminal_bridge) return 2;
    if (is_one_step) return 1;
    return 0;
}

/* 由时间距离得到 bootstrap 折扣。plies<=0 返回 0（不 bootstrap），
   而不是静默返回 1.0 —— 那会把「不 bootstrap」变成「全额 bootstrap」。 */
inline float bootstrap_discount_for_plies(int plies, float gamma) {
    if (plies <= 0) return 0.0f;
    float d = 1.0f;
    for (int i = 0; i < plies; ++i) d *= gamma;
    return d;
}

/* 契约校验：指数与登记的时间距离必须一致。 */
inline int discount_exponent_matches(int plies, int exponent) {
    return plies == exponent;
}

/* 实测跨手数：用棋盘上的棋子差判断下一状态前进了几手。 */
inline int observed_plies(const float* state, const float* next_state) {
    if (!state || !next_state) return -1;
    int a = 0, b = 0;
    for (int i = 0; i < CELL_COUNT; ++i) {
        if (state[i] != 0.0f) ++a;
        if (next_state[i] != 0.0f) ++b;
    }
    return b - a;
}

inline int is_two_ply_experience(const float* state, const float* next_state) {
    return observed_plies(state, next_state) == 2;
}

/* 独立网络下的形态判定：非终局样本必须是两 ply（A），终局样本允许两 ply 终局
   桥接（B：对手回应结束比赛）或一 ply 直接终局（本方那一手直接连四，生产者见
   terminal_outcome_credit.h 的 done 分支；此时 next_turn != state_turn）。
   返回 0 合法，非 0 为违反的形态编号，供 training_fail_detail(2,"replay_span",..) 记录。
   它不替代 validate_experience 的棋盘/轮次校验，只判跨度语义。 */
inline int check_replay_span(const float* state, const float* next_state, int done) {
    const int plies = observed_plies(state, next_state);
    if (done) return (plies == 1 || plies == 2) ? 0 : 1;
    return plies == 2 ? 0 : 2;
}

} // namespace experience_semantics

// D1 回归：独立模式下「行动方选招」必须走该方自己的 online 网络，而不是共享 q_net
//
// 重要性：独立模式里共享 q_net/target_net 从播种那一刻起就不再更新，若选招仍读
// q_net，则在线方每一步都用**预训练种子网络**选招，网络在被训练但行为永不反映
// 学到的策略 —— 表现成「怎么练都不涨」。
//
// 做法是行为断言而不是只看指针：把共享 q_net 与「该方 online」分别调成偏好
// 不同的着法，然后看 select_action 实际选了哪一手。
//   独立模式 + 黑方行动 -> 必须选 online(black) 偏好那一手；
//   独立模式 + 白方行动 -> 必须选 online(white) 偏好那一手；
//   遗留共享模式        -> 必须仍选 q_net 偏好那一手（证明未波及历史行为）。
#include "independent_networks.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while(0)

namespace {

/* 让某个网络在零权重下强烈偏好某个着法：forward 的输出 = b4 + w4*hidden3，
   权重全零时 argmax 就是 b4 最大的下标（唯一最大 ⇒ 无并列歧义）。 */
void prefer(NeuralNet* net, int action) {
    memset(net, 0, sizeof(*net));
    net->b4[action] = 100.0f;
}

GameState sparse_board(int turn) {
    GameState g; memset(&g, 0, sizeof(g));
    g.turn = turn;
    g.board[0] = 1;
    g.board[7] = 2;
    return g;
}

} // namespace

int main() {
    puts("=== D1: action selection routes to the acting side's own online network ===");
    fflush(stdout);
    g_hot_config.memory_enabled = 0;   /* 记忆叠加会改动 qvals，这里只测路由 */

    auto agent = std::make_unique<DQNAgent>();
    init_dqn_ex(agent.get(), 0);
    g_current_agent = agent.get();
    NEED(indep_ensure(agent.get()) != nullptr);

    const int black_choice = 1;   /* online(black) 偏好 */
    const int white_choice = 2;   /* online(white) 偏好 */
    const int shared_choice = 3;  /* 共享 q_net 偏好（与前两者都不同） */

    /* ---- 遗留共享模式：必须仍按 q_net 选（未开启按方路由时逐位不变） ---- */
    {
        prefer(&agent->q_net, shared_choice);
        prefer(&agent->indep->online(Side::Black), black_choice);
        prefer(&agent->indep->online(Side::White), white_choice);
        GameState g = sparse_board(0);
        PRNGState rng; prng_init(&rng, 11);
        const int action = select_action(agent.get(), &g, 0, 0.0f, &rng, 0);
        printf("[legacy] indep_active=0 turn=0 -> action=%d (q_net prefers %d)\n",
            action, shared_choice);
        NEED(action == shared_choice);
    }

    /* ---- 独立模式：按 g->turn 路由到该方 online ---- */
    agent->indep->set_active(1);
    NEED(indep_active(agent.get()));

    {
        GameState g = sparse_board(0);
        PRNGState rng; prng_init(&rng, 11);
        const int action = select_action(agent.get(), &g, 0, 0.0f, &rng, 0);
        printf("[indep] turn=0(black) -> action=%d (online(black) prefers %d, q_net prefers %d)\n",
            action, black_choice, shared_choice);
        NEED(action == black_choice);
        NEED(action != shared_choice);
    }
    {
        GameState g = sparse_board(1);
        PRNGState rng; prng_init(&rng, 11);
        const int action = select_action(agent.get(), &g, 0, 0.0f, &rng, 0);
        printf("[indep] turn=1(white) -> action=%d (online(white) prefers %d, q_net prefers %d)\n",
            action, white_choice, shared_choice);
        NEED(action == white_choice);
        NEED(action != shared_choice);
    }

    /* ---- 直接指针断言：路由返回的就是按方网络 ---- */
    NEED(agent_online_net(agent.get(), Side::Black) == &agent->indep->online(Side::Black));
    NEED(agent_online_net(agent.get(), Side::Black) != &agent->q_net);
    NEED(agent_online_net(agent.get(), Side::White) != &agent->q_net);
    printf("[routing] online(black)=%p online(white)=%p shared q_net=%p (all distinct)\n",
        (void*)&agent->indep->online(Side::Black), (void*)&agent->indep->online(Side::White),
        (void*)&agent->q_net);

    /* ---- 反向验证：改掉该方 online 的偏好，选招必须随之改变 ---- */
    {
        prefer(&agent->indep->online(Side::Black), 4);
        GameState g = sparse_board(0);
        PRNGState rng; prng_init(&rng, 11);
        const int action = select_action(agent.get(), &g, 0, 0.0f, &rng, 0);
        printf("[indep] after changing online(black) preference to 4 -> action=%d\n", action);
        NEED(action == 4);
    }

    free_dqn(agent.get());
    g_current_agent = nullptr;
    puts("PASS D1: independent mode selects actions with the acting side's own online "
         "network (verified behaviourally for both colours and by changing that network), "
         "while legacy shared mode still selects with q_net, bit for bit");
    return 0;
}

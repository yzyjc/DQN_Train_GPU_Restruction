// independent_networks.cpp — 黑白独立网络的物理棋色所有权、按方状态与路由
//
// 这个文件刻意很小：它只负责「哪一方」这件事，不含任何训练数学。
// 训练数学仍在 neural_net.cpp / dqn_agent.cpp，由按方状态驱动。
#include "core.h"
#include "independent_networks.h"

namespace independent_training {

int indep_start_from_shared(DQNAgent* agent) {
    if (!agent) return 0;
    if (indep_active(agent)) return 1;
    IndependentNetworks* nets = indep_ensure(agent);
    if (!nets) return 0;
    const int device_active = gpu_net_active();
    if (device_active && gpu_net_download_agent(agent) != 0) return 0;
    nets->seed_both_from(agent->q_net, agent->target_net);
    nets->set_active(1);
    nets->set_bound(Side::Black);
    if (device_active && gpu_net_activate(agent) != 0) {
        training_fail_detail(1, "indep_gpu_activation", -1);
        return 0;
    }
    return 1;
}

IndependentNetworks* indep_ensure(DQNAgent* agent) {
    if (!agent) return nullptr;
    if (agent->indep) return agent->indep;
    agent->indep = new (std::nothrow)
        IndependentNetworks(agent->q_net, agent->target_net);
    return agent->indep;
}

NeuralNet* agent_online_net(DQNAgent* agent, Side side) {
    if (!agent) return nullptr;
    if (!indep_active(agent)) return &agent->q_net;
    return &agent->indep->online(side);
}

NeuralNet* agent_target_net(DQNAgent* agent, Side side) {
    if (!agent) return nullptr;
    if (!indep_active(agent)) return &agent->target_net;
    return &agent->indep->target(side);
}

const NeuralNet* agent_online_net(const DQNAgent* agent, Side side) {
    if (!agent) return nullptr;
    if (!indep_active(agent)) return &agent->q_net;
    return &agent->indep->online(side);
}

/* D2：以该物理棋色出场的冻结对手。路由生效时是**该方自己的 target 网络**，
   因此白方不会再被喂黑方的网络快照（HANDOFF:72 明确禁止），也不需要再新增
   一份 3.4MB 的副本成员（那会触发 cl 的 C1060 编译期堆耗尽）。 */
NeuralNet* agent_opponent_net(DQNAgent* agent, Side side) {
    if (!agent) return nullptr;
    if (!indep_active(agent)) return &agent->opponent_net;
    return &agent->indep->target(side);
}

const NeuralNet* agent_opponent_net(const DQNAgent* agent, Side side) {
    if (!agent) return nullptr;
    if (!indep_active(agent)) return &agent->opponent_net;
    return &agent->indep->target(side);
}

int experience_side(const Experience* buf, const DQNAgent* agent, Side* out_side) {
    if (!agent || !out_side) return 0;
    /* 一个批次必须整体属于同一物理棋色：这里只认两个合法回放缓冲区。
       既不匹配黑也不匹配白的缓冲区是调用方错误，绝不猜测。 */
    if (buf == agent->replay_black) { *out_side = Side::Black; return 1; }
    if (buf == agent->replay_white) { *out_side = Side::White; return 1; }
    return 0;
}

} // namespace independent_training

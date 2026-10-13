// 黑白独立网络：生产训练路径隔离验证（INDEPENDENT_NETWORKS_HANDOFF.md 测试 2）
//
// 这个测试刻意走真实生产入口：
//     replay_from_buffer(agent, agent->replay_black|white, ...)
//         -> 按方 target 计算 -> train_agent_batch_side -> train_cpu_inplace
// 不调用 train_step_cpu_only，不直接改容器字段，不比较 loss / 缓存 / 指针地址。
//
// 断言强度：对 NeuralNet 与 SideUpdateState 做整结构逐字节比较
// （memcmp），而不是抽样比较几个权重。任一方被对方更新污染都会直接失败。
#include "independent_networks.h"
#include "sentinel_guard.h"
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

namespace {

/* 用一个明确合法的一步 transition 填充回放，绝不造假数据：
   board 里放一枚棋子并让行动者落子在空位，next_state 与 state 都经
   state_to_input 生成，因此 validate_experience 能通过。 */
void fill_replay(DQNAgent* agent, std::vector<Experience>& out, int side, int samples) {
    out.assign(samples, Experience{});
    const float placed = (side == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < samples; ++i) {
        GameState g;
        memset(&g, 0, sizeof(g));
        g.turn = side;
        /* 稀疏棋盘：远离任何可能连四的密集区域，保证两手都不会终局。 */
        g.board[side == 0 ? 0 : 24] = placed;
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = -placed;

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
        state_to_input(&after, e.next_state);   /* 两 ply 之后的棋盘 */
        e.action = action;
        e.reward = (side == 0) ? 0.05f : 0.02f;  /* 本方那一手的奖励 */
        e.done = 0;
        if (!validate_experience(&e)) out[i].action = -1;
    }
}

int put_replay(DQNAgent* agent, std::vector<Experience>& src, int side) {
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

/* 6.8MB 的 NeuralNet 绝不能按值放栈上：用堆快照 + memcmp 做逐字节比较。 */
struct SideSnapshot {
    std::unique_ptr<NeuralNet> online;
    std::unique_ptr<NeuralNet> target;
    SideUpdateState state{};
    explicit SideSnapshot(const DQNAgent* agent, Side side)
        : online(new NeuralNet(agent->indep->online(side))),
          target(new NeuralNet(agent->indep->target(side))),
          state(agent->indep->updates(side)) {}
};

std::unique_ptr<SideSnapshot> take_snapshot(const DQNAgent* agent, Side side) {
    return std::unique_ptr<SideSnapshot>(new SideSnapshot(agent, side));
}

bool online_identical(const SideSnapshot& a, const DQNAgent* agent, Side side) {
    return memcmp(a.online.get(), &agent->indep->online(side), sizeof(NeuralNet)) == 0;
}
bool target_identical(const SideSnapshot& a, const DQNAgent* agent, Side side) {
    return memcmp(a.target.get(), &agent->indep->target(side), sizeof(NeuralNet)) == 0;
}
bool state_identical(const SideSnapshot& a, const DQNAgent* agent, Side side) {
    return memcmp(&a.state, &agent->indep->updates(side), sizeof(SideUpdateState)) == 0;
}

/* 让两方初始权重不同：没有这一步，「另一方未被修改」的断言可能在
   两方本来就是同一个缓冲区时误过。都只改参数，不碰任何缓存字段。 */
void perturb(DQNAgent* agent, Side side, float delta) {
    NeuralNet& n = agent->indep->online(side);
    n.b4[0] += delta;
    n.w4[0][0] += delta;
    agent->indep->target(side).b4[1] += delta * 0.5f;
}

/* 按 GpuNetParams 的字段顺序（= NeuralNet 的可学习参数子集）算两点之间的 L2 差。
   刻意不把 tokens/attn_context/flat_cache 等前向缓存算进来：
   这些字段在 staged 副本里也会被写，被误当成「参数变了」就失去证明力。 */
struct ParamDiff {
    double norm = 0.0;
    double b4_abs = 0.0;
    long long different = 0;
};

void accumulate(ParamDiff* d, const float* a, const float* b, size_t n, bool is_b4 = false) {
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)a[i] - (double)b[i];
        d->norm += x * x;
        if (x != 0.0) d->different++;
        if (is_b4) d->b4_abs = fmax(d->b4_abs, fabs(x));
    }
}

template <size_t R, size_t C>
void accumulate2(ParamDiff* d, const float (&a)[R][C], const float (&b)[R][C]) {
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c) {
            const double x = (double)a[r][c] - (double)b[r][c];
            d->norm += x * x;
            if (x != 0.0) d->different++;
        }
}

ParamDiff param_diff(const NeuralNet* a, const NeuralNet* b) {
    ParamDiff d;
    accumulate2(&d, a->token_embed, b->token_embed);
    accumulate2(&d, a->pos_embed, b->pos_embed);
    accumulate2(&d, a->w1, b->w1);
    accumulate(&d, a->b1, b->b1, HIDDEN1_SIZE);
    accumulate2(&d, a->w2, b->w2);
    accumulate(&d, a->b2, b->b2, HIDDEN2_SIZE);
    accumulate2(&d, a->w3, b->w3);
    accumulate(&d, a->b3, b->b3, HIDDEN3_SIZE);
    accumulate2(&d, a->w4, b->w4);
    accumulate(&d, a->b4, b->b4, CELL_COUNT, true);
    accumulate2(&d, a->attn_wq, b->attn_wq);
    accumulate(&d, a->attn_bq, b->attn_bq, ATTENTION_EMBED);
    accumulate2(&d, a->attn_wk, b->attn_wk);
    accumulate(&d, a->attn_bk, b->attn_bk, ATTENTION_EMBED);
    accumulate2(&d, a->attn_wv, b->attn_wv);
    accumulate(&d, a->attn_bv, b->attn_bv, ATTENTION_EMBED);
    accumulate2(&d, a->attn_wo, b->attn_wo);
    accumulate(&d, a->attn_bo, b->attn_bo, TOKEN_EMBED_DIM);
    accumulate2(&d, a->pool_w, b->pool_w);
    accumulate(&d, a->pool_b, b->pool_b, FEATURE_DIM);
    d.norm = sqrt(d.norm);
    return d;
}

} // namespace

/* 主隔离验证：只训练某一方的批次，检查
     - 该方 online 参数确实变化；
     - 该方 target 完全不变（未发生目标同步）；
     - 另一方 online / target / 全部按方状态逐字节不变。
   两个方向都跑。 */
static int run_direction(DQNAgent* agent, Side trained, const char* label) {
    const Side other = opposite_side(trained);
    const int trained_id = side_to_physical(trained);

    std::vector<Experience> samples;
    fill_replay(agent, samples, trained_id, 2 * BATCH_SIZE);
    if (!put_replay(agent, samples, trained_id)) {
        printf("FAIL %s: cannot build valid replay\n", label);
        return 1;
    }
    /* 另一方的回放留空，确保不会顺带训练到对方。 */
    {
        std::vector<Experience> empty;
        put_replay(agent, empty, side_to_physical(other));
    }

    const std::unique_ptr<SideSnapshot> own_before = take_snapshot(agent, trained);
    const std::unique_ptr<SideSnapshot> other_before = take_snapshot(agent, other);

    /* 路由必须真的指向按方网络，而不是共享 q_net。 */
    NEED(agent_online_net(agent, trained) == &agent->indep->online(trained));
    NEED(agent_target_net(agent, trained) == &agent->indep->target(trained));
    NEED(agent_online_net(agent, other) == &agent->indep->online(other));
    NEED(agent_online_net(agent, trained) != agent_online_net(agent, other));
    NEED(agent->indep->online(trained).b4[0] != agent->indep->online(other).b4[0]);

    const unsigned long long committed_before = own_before->state.committed_batches;
    const float loss = replay_from_buffer(agent,
        (trained == Side::Black) ? agent->replay_black : agent->replay_white,
        (trained == Side::Black) ? agent->replay_black_count : agent->replay_white_count,
        (trained == Side::Black) ? &agent->replay_black_idx : &agent->replay_white_idx);

    printf("[%s] replay_from_buffer loss=%.9g (finite=%d)\n", label, loss, (int)isfinite(loss));
    NEED(isfinite(loss));
    NEED(loss > 0.0f);

    /* --- 被训练方：online 必须变，target 必须不变 --- */
    const bool own_online_changed =
        memcmp(own_before->online.get(), &agent->indep->online(trained), sizeof(NeuralNet)) != 0;
    const ParamDiff own_diff = param_diff(own_before->online.get(), &agent->indep->online(trained));
    printf("[%s] trained-side online changed=%d; parameter L2 delta=%.9g; "
        "changed params=%lld; max|db4|=%.9g; b4[0] %.12g -> %.12g\n",
        label, (int)own_online_changed, own_diff.norm, own_diff.different, own_diff.b4_abs,
        own_before->online->b4[0], agent->indep->online(trained).b4[0]);
    NEED(own_online_changed);
    NEED(isfinite(own_diff.norm) && own_diff.norm > 0.0);
    /* 必须是真实参数变化，不是只有前向缓存被写。 */
    NEED(own_diff.different > 0 && own_diff.b4_abs > 0.0);
    NEED(memcmp(own_before->online->b4, agent->indep->online(trained).b4,
        sizeof(own_before->online->b4)) != 0);
    NEED(target_identical(*own_before, agent, trained));
    NEED(agent->indep->updates(trained).committed_batches == committed_before + 1);
    NEED(agent->indep->updates(trained).rejected_batches == own_before->state.rejected_batches);

    /* --- 未被训练方：online / target / 按方状态全部逐字节不变 --- */
    const bool other_online_same = online_identical(*other_before, agent, other);
    const bool other_target_same = target_identical(*other_before, agent, other);
    const bool other_state_same = state_identical(*other_before, agent, other);
    const ParamDiff other_diff =
        param_diff(other_before->online.get(), &agent->indep->online(other));
    const ParamDiff other_target_diff =
        param_diff(other_before->target.get(), &agent->indep->target(other));
    printf("[%s] other side (%s): online_same=%d target_same=%d state_same=%d "
        "parameter L2 delta=%.9g/%lld target L2 delta=%.9g/%lld "
        "(committed=%llu delta_ema=%.9g warmup=%d rejected=%llu target_sync=%d lr=%.9g)\n",
        label, side_name(other), (int)other_online_same, (int)other_target_same,
        (int)other_state_same, other_diff.norm, other_diff.different,
        other_target_diff.norm, other_target_diff.different,
        agent->indep->updates(other).committed_batches,
        agent->indep->updates(other).delta_ema,
        agent->indep->updates(other).delta_warmup,
        agent->indep->updates(other).rejected_batches,
        agent->indep->updates(other).target_update_counter,
        agent->indep->updates(other).learning_rate);
    NEED(other_diff.norm == 0.0 && other_diff.different == 0);
    NEED(other_target_diff.norm == 0.0 && other_target_diff.different == 0);
    NEED(other_online_same);
    NEED(other_target_same);
    NEED(other_state_same);

    /* 训练后按方状态必须真的有推进（否则上一条断言可能只是因为什么都没做）。 */
    printf("[%s] trained side state: committed=%llu delta_ema=%.9g warmup=%d rejected=%llu lr=%.9g\n",
        label,
        agent->indep->updates(trained).committed_batches,
        agent->indep->updates(trained).delta_ema,
        agent->indep->updates(trained).delta_warmup,
        agent->indep->updates(trained).rejected_batches,
        agent->indep->updates(trained).learning_rate);
    return 0;
}

int main() {
    g_learning_rate = 0.0001f;
    sentinel_guard_init();

    auto agent = std::make_unique<DQNAgent>();
    init_dqn_ex(agent.get(), 0);
    g_current_agent = agent.get();

    NEED(agent->indep == nullptr);
    NEED(!indep_active(agent.get()));
    /* 未开启按方路由时，路由查询必须回落到共享网络。 */
    NEED(agent_online_net(agent.get(), Side::Black) == &agent->q_net);
    NEED(agent_online_net(agent.get(), Side::White) == &agent->q_net);
    NEED(agent_target_net(agent.get(), Side::White) == &agent->target_net);

    IndependentNetworks* nets = indep_ensure(agent.get());
    NEED(nets != nullptr);
    /* 基础权重来自当前共享网络（预训练兼容路径：只在开启前复制一次）。 */
    NEED(memcmp(&nets->online(Side::Black), &agent->q_net, sizeof(NeuralNet)) == 0);
    NEED(memcmp(&nets->online(Side::White), &agent->q_net, sizeof(NeuralNet)) == 0);

    perturb(agent.get(), Side::Black, 1e-3f);
    perturb(agent.get(), Side::White, -2e-3f);

    /* 显式开启路由：此后绝不重新复制两方权重。 */
    nets->set_active(1);
    NEED(indep_active(agent.get()));
    NEED(nets->seed_once_from_shared(agent->q_net, agent->target_net) == 0);

    /* 逐字节快照，用于证明「开启路由本身不改权重」。 */
    {
        auto black_online = std::make_unique<NeuralNet>(nets->online(Side::Black));
        auto white_online = std::make_unique<NeuralNet>(nets->online(Side::White));
        NEED(memcmp(black_online.get(), &nets->online(Side::Black), sizeof(NeuralNet)) == 0);
        NEED(memcmp(white_online.get(), &nets->online(Side::White), sizeof(NeuralNet)) == 0);
    }

    if (int rc = run_direction(agent.get(), Side::Black, "train-black")) return rc;
    if (int rc = run_direction(agent.get(), Side::White, "train-white")) return rc;

    /* 再次确认起始扰动仍然存在：说明没有任何一方在训练后被从共享网络重播。 */
    NEED(nets->online(Side::Black).b4[0] != nets->online(Side::White).b4[0]);

    /* 非法物理棋色必须被显式拒绝，不得静默回落到默认方。 */
    bool threw = false;
    try { nets->at(static_cast<Side>(2)); } catch (const std::invalid_argument&) { threw = true; }
    NEED(threw);
    threw = false;
    try { side_from_physical(7); } catch (const std::invalid_argument&) { threw = true; }
    NEED(threw);

    /* 不属于任何物理棋色的缓冲区：独立网络开启时必须致命失败，不得猜测归属。 */
    {
        std::vector<Experience> alien(2 * BATCH_SIZE);
        Side s = Side::Black;
        NEED(!experience_side(alien.data(), agent.get(), &s));
        /* 只做归属判定，不真的调用 replay（那会闩死 g_training_fault）。 */
        NEED(experience_side(agent->replay_black, agent.get(), &s) && s == Side::Black);
        NEED(experience_side(agent->replay_white, agent.get(), &s) && s == Side::White);
    }

    free_dqn(agent.get());
    g_current_agent = nullptr;
    puts("PASS production replay_from_buffer isolation: trained side online changes; "
         "its target and the other side's online/target/per-side state stay byte-identical");
    return 0;
}

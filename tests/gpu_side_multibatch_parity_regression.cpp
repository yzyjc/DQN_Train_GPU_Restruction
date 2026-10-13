// ③ 收尾：每方向连续多批次 + 按方路径 CPU/GPU 一致性（deepseek-flash）
//
// HANDOFF:84 要求「GPU batch=32、多动作、**多连续批次**，CPU/GPU loss 和参数更新
// 比较；检查每项有限性、参数 delta 的 L2 相对误差、实际后端与返回码。双方各自的
// 设备快照证明未被另一方更新。单样本 PASS 不算批量验证。」
//
// 已有 `independent_gpu_side_isolation_regression` 覆盖「按方隔离 + 真实设备回读」，
// 但每个方向只跑 1 个批次。本测试补两块：
//   A. **按方路径 CPU/GPU 一致性**：两个初始状态完全相同的 agent（一个纯 CPU、
//      一个 GPU 绑定方句柄），用同一批 32 条样本训练同一方，比较 loss 与参数
//      相对 L2 误差（判据与 step_outcome_regression 相同）。
//   B. **每方向连续 3 个批次**：每批设备参数都必须真的前进，另一方全程逐字节冻结。
#include "independent_networks.h"
#include "sentinel_guard.h"
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while (0)

namespace {

/* ---- 参数差（只取可学习参数，忽略前向缓存字段）---- */
struct ParamDiff { double norm = 0.0; long long different = 0; };

void accum(ParamDiff* d, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)a[i] - (double)b[i];
        d->norm += x * x;
        if (x != 0.0) d->different++;
    }
}
template <size_t R, size_t C>
void accum2(ParamDiff* d, const float (&a)[R][C], const float (&b)[R][C]) {
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c) {
            const double x = (double)a[r][c] - (double)b[r][c];
            d->norm += x * x;
            if (x != 0.0) d->different++;
        }
}
ParamDiff param_diff(const NeuralNet* a, const NeuralNet* b) {
    ParamDiff d;
    accum2(&d, a->token_embed, b->token_embed);
    accum2(&d, a->pos_embed, b->pos_embed);
    accum2(&d, a->w1, b->w1); accum(&d, a->b1, b->b1, HIDDEN1_SIZE);
    accum2(&d, a->w2, b->w2); accum(&d, a->b2, b->b2, HIDDEN2_SIZE);
    accum2(&d, a->w3, b->w3); accum(&d, a->b3, b->b3, HIDDEN3_SIZE);
    accum2(&d, a->w4, b->w4); accum(&d, a->b4, b->b4, CELL_COUNT);
    accum2(&d, a->attn_wq, b->attn_wq); accum(&d, a->attn_bq, b->attn_bq, ATTENTION_EMBED);
    accum2(&d, a->attn_wk, b->attn_wk); accum(&d, a->attn_bk, b->attn_bk, ATTENTION_EMBED);
    accum2(&d, a->attn_wv, b->attn_wv); accum(&d, a->attn_bv, b->attn_bv, ATTENTION_EMBED);
    accum2(&d, a->attn_wo, b->attn_wo); accum(&d, a->attn_bo, b->attn_bo, TOKEN_EMBED_DIM);
    accum2(&d, a->pool_w, b->pool_w);   accum(&d, a->pool_b, b->pool_b, FEATURE_DIM);
    d.norm = sqrt(d.norm);
    return d;
}
double param_magnitude(const NeuralNet* n) {
    ParamDiff d;
    const NeuralNet zero = NeuralNet();
    (void)zero;
    NeuralNet z; memset(&z, 0, sizeof(z));
    d = param_diff(n, &z);
    return d.norm;
}

/* ---- 真实两 ply 样本（本方落子 + 对手真实回应）---- */
const float PLACED[2] = { 1.0f, -1.0f };

void fill_two_ply(std::vector<Experience>& out, int side, int samples) {
    out.assign(samples, Experience{});
    for (int i = 0; i < samples; ++i) {
        GameState g; memset(&g, 0, sizeof(g));
        g.turn = side;
        g.board[side == 0 ? 0 : 24] = PLACED[side];
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = -PLACED[side];
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&g, moves);
        if (n <= 0) { out[i].action = -1; continue; }
        const int action = moves[i % n];
        GameState mid = g; make_move(&mid, action);
        if (check_winner(&mid) != 0 || is_full(&mid)) { out[i].action = -1; continue; }
        int reply_moves[CELL_COUNT];
        const int rn = get_legal_moves(&mid, reply_moves);
        if (rn <= 0) { out[i].action = -1; continue; }
        const int reply = reply_moves[(i * 7 + 3) % rn];
        GameState after = mid; make_move(&after, reply);
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
    Experience* buf = (side == 0) ? agent->replay_black : agent->replay_white;
    int* idx = (side == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
    int* cnt = (side == 0) ? &agent->replay_black_count : &agent->replay_white_count;
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i].action < 0) return 0;
        buf[i] = src[i];
    }
    *idx = 0; *cnt = (int)src.size();
    return 1;
}

void perturb(DQNAgent* a, Side side, float delta) {
    a->indep->online(side).b4[0] += delta;
    a->indep->online(side).w4[0][0] += delta * 0.5f;
    a->indep->target(side).b4[2] -= delta * 0.25f;
}

std::unique_ptr<DQNAgent> make_agent() {
    auto a = std::make_unique<DQNAgent>();
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    if (!indep_ensure(a.get())) return nullptr;
    a->indep->set_active(1);
    a->indep->set_bound(Side::Black);
    return a;
}

} // namespace

int main() {
    g_learning_rate = 0.0001f;
    sentinel_guard_init();

    /* ================= A. 按方路径 CPU/GPU 一致性 =================
       顺序刻意如此：CPU 那一半必须在 gpu_init 之前跑，因为 gpu_net_active()
       是进程级标志，一旦激活，A 也会走 GPU 路径。 */
    {
        auto gpu_a = make_agent();
        NEED(gpu_a != nullptr);
        perturb(gpu_a.get(), Side::Black, 1e-2f);
        perturb(gpu_a.get(), Side::White, -3e-2f);

        /* 同一批 32 条真实两 ply 样本，两个 agent 用完全相同的输入。 */
        std::vector<Experience> batch;
        fill_two_ply(batch, 0, BATCH_SIZE * 2);
        NEED(batch.size() >= (size_t)BATCH_SIZE);
        float states[BATCH_SIZE][CELL_COUNT + 1];
        int actions[BATCH_SIZE];
        float targets[BATCH_SIZE];
        for (int i = 0; i < BATCH_SIZE; ++i) {
            memcpy(states[i], batch[i].state, sizeof(states[i]));
            actions[i] = batch[i].action;
            targets[i] = batch[i].reward + 0.25f * sinf((float)i);
            NEED(isfinite(targets[i]));
        }

        /* CPU agent：与 GPU agent 完全同源，然后走纯 CPU 训练。 */
        auto cpu_a = make_agent();
        NEED(cpu_a != nullptr);
        cpu_a->indep->online(Side::Black) = gpu_a->indep->online(Side::Black);
        cpu_a->indep->target(Side::Black) = gpu_a->indep->target(Side::Black);
        cpu_a->indep->updates(Side::Black) = gpu_a->indep->updates(Side::Black);
        cpu_a->indep->online(Side::White) = gpu_a->indep->online(Side::White);
        NEED(param_diff(&cpu_a->indep->online(Side::Black),
                        &gpu_a->indep->online(Side::Black)).different == 0);
        NEED(!gpu_net_active());
        const NeuralNet black_before = cpu_a->indep->online(Side::Black);
        float cpu_loss = 0.0f;
        const int cpu_rc = train_agent_batch_side(cpu_a.get(), 0, (const float*)states,
            actions, targets, BATCH_SIZE, 0.0001f, &cpu_loss);
        printf("[A cpu] rc=%d loss=%.9g\n", cpu_rc, (double)cpu_loss);
        NEED(cpu_rc == 1 && isfinite(cpu_loss));
        const NeuralNet black_cpu_after = cpu_a->indep->online(Side::Black);
        NEED(param_diff(&black_before, &black_cpu_after).different > 0);

        /* GPU agent：激活后显式绑定到黑方，用同一批样本训练同一方。 */
        NEED(gpu_init() == 0);
        NEED(gpu_net_activate(gpu_a.get()) == 0);
        NEED(gpu_net_bind_side(gpu_a.get(), 0) == 0);
        float gpu_loss = 0.0f;
        const int gpu_rc = train_agent_batch_side(gpu_a.get(), 0, (const float*)states,
            actions, targets, BATCH_SIZE, 0.0001f, &gpu_loss);
        printf("[A gpu] rc=%d loss=%.9g\n", gpu_rc, (double)gpu_loss);
        NEED(gpu_rc == cpu_rc);
        NEED(isfinite(gpu_loss));
        NEED(gpu_net_download_agent(gpu_a.get()) == 0);
        const NeuralNet black_gpu_after = gpu_a->indep->online(Side::Black);

        const ParamDiff d_after = param_diff(&black_cpu_after, &black_gpu_after);
        const double mag = param_magnitude(&black_cpu_after);
        const double relative = mag > 0.0 ? d_after.norm / mag : d_after.norm;
        const double loss_gap = fabs((double)cpu_loss - (double)gpu_loss);
        printf("[A parity] |dCPU-dGPU| L2=%.9g magnitude=%.9g relative=%.9g "
            "changed=%lld |loss gap|=%.9g (limits relative<0.005, gap<1e-5)\n",
            d_after.norm, mag, relative, d_after.different, loss_gap);
        NEED(isfinite(relative) && relative < 0.005);
        NEED(loss_gap < 1e-5);
        /* 未被训练的一方在两套后端下都必须原样不动。 */
        NEED(param_diff(&gpu_a->indep->online(Side::White),
                        &cpu_a->indep->online(Side::White)).different == 0);

        gpu_net_deactivate();
        gpu_shutdown();
        free_dqn(cpu_a.get());
        free_dqn(gpu_a.get());
        g_current_agent = nullptr;
    }

    /* ================= B. 每方向连续 3 个批次 ================= */
    {
        NEED(gpu_init() == 0);
        auto a = make_agent();
        NEED(a != nullptr);
        perturb(a.get(), Side::Black, 1e-2f);
        perturb(a.get(), Side::White, -3e-2f);
        NEED(gpu_net_activate(a.get()) == 0);
        NEED(gpu_net_bind_side(a.get(), 0) == 0);

        NeuralNet white_online_ref = a->indep->online(Side::White);
        NeuralNet white_target_ref = a->indep->target(Side::White);
        const SideUpdateState white_state_ref = a->indep->updates(Side::White);

        NeuralNet prev_online = a->indep->online(Side::Black);
        for (int b = 0; b < 3; ++b) {
            std::vector<Experience> batch;
            fill_two_ply(batch, 0, 2 * BATCH_SIZE);
            NEED(put_replay(a.get(), batch, 0));
            const float loss = replay_from_buffer(a.get(), a->replay_black,
                a->replay_black_count, &a->replay_black_idx);
            std::unique_ptr<NeuralNet> dev_online(new NeuralNet());
            std::unique_ptr<NeuralNet> dev_target(new NeuralNet());
            NEED(gpu_net_snapshot_side_pair(a.get(), 0, dev_online.get(), dev_target.get()) == 0);
            const ParamDiff dev_delta = param_diff(&prev_online, dev_online.get());
            const ParamDiff shadow_delta = param_diff(&prev_online,
                &a->indep->online(Side::Black));
            printf("[B black batch %d] loss=%.9g training_fault=%ld "
                "DEVICE online delta L2=%.9g/%lld shadow delta L2=%.9g/%lld\n",
                b + 1, (double)loss, (long)g_training_fault,
                dev_delta.norm, dev_delta.different, shadow_delta.norm, shadow_delta.different);
            NEED(g_training_fault == 0);
            NEED(isfinite(loss) && loss > 0.0f);
            NEED(dev_delta.different > 0 && dev_delta.norm > 0.0);
            NEED(shadow_delta.different > 0 && shadow_delta.norm > 0.0);
            /* 绑定方的设备与影子必须一致（设备就是这一方的镜像）。 */
            const ParamDiff mirror = param_diff(dev_online.get(), &a->indep->online(Side::Black));
            NEED(mirror.different == 0);
            /* 另一方全程不许动：影子、target、按方状态。 */
            NEED(param_diff(&white_online_ref, &a->indep->online(Side::White)).different == 0);
            NEED(param_diff(&white_target_ref, &a->indep->target(Side::White)).different == 0);
            NEED(memcmp(&white_state_ref, &a->indep->updates(Side::White),
                sizeof(SideUpdateState)) == 0);
            prev_online = a->indep->online(Side::Black);
        }
        printf("[B black] 3 consecutive batches: device moved each batch, white frozen throughout\n");

        /* 反方向：绑定白方后连续 3 个批次，黑方必须全程冻结。 */
        NEED(gpu_net_bind_side(a.get(), 1) == 0);
        NeuralNet black_online_ref2 = a->indep->online(Side::Black);
        NeuralNet black_target_ref2 = a->indep->target(Side::Black);
        const SideUpdateState black_state_ref2 = a->indep->updates(Side::Black);
        NeuralNet prev_white = a->indep->online(Side::White);
        for (int b = 0; b < 3; ++b) {
            std::vector<Experience> batch;
            fill_two_ply(batch, 1, 2 * BATCH_SIZE);
            NEED(put_replay(a.get(), batch, 1));
            const float loss = replay_from_buffer(a.get(), a->replay_white,
                a->replay_white_count, &a->replay_white_idx);
            std::unique_ptr<NeuralNet> dev_online(new NeuralNet());
            std::unique_ptr<NeuralNet> dev_target(new NeuralNet());
            NEED(gpu_net_snapshot_side_pair(a.get(), 1, dev_online.get(), dev_target.get()) == 0);
            const ParamDiff dev_delta = param_diff(&prev_white, dev_online.get());
            const ParamDiff shadow_delta = param_diff(&prev_white, &a->indep->online(Side::White));
            printf("[B white batch %d] loss=%.9g training_fault=%ld "
                "DEVICE online delta L2=%.9g/%lld shadow delta L2=%.9g/%lld\n",
                b + 1, (double)loss, (long)g_training_fault,
                dev_delta.norm, dev_delta.different, shadow_delta.norm, shadow_delta.different);
            NEED(g_training_fault == 0);
            NEED(isfinite(loss) && loss > 0.0f);
            NEED(dev_delta.different > 0 && dev_delta.norm > 0.0);
            NEED(shadow_delta.different > 0 && shadow_delta.norm > 0.0);
            NEED(param_diff(dev_online.get(), &a->indep->online(Side::White)).different == 0);
            NEED(param_diff(&black_online_ref2, &a->indep->online(Side::Black)).different == 0);
            NEED(param_diff(&black_target_ref2, &a->indep->target(Side::Black)).different == 0);
            NEED(memcmp(&black_state_ref2, &a->indep->updates(Side::Black),
                sizeof(SideUpdateState)) == 0);
            prev_white = a->indep->online(Side::White);
        }
        printf("[B white] 3 consecutive batches: device moved each batch, black frozen throughout\n");

        gpu_net_deactivate();
        gpu_shutdown();
        free_dqn(a.get());
        g_current_agent = nullptr;
    }

    puts("PASS side-path CPU/GPU parity and multi-batch isolation: 3 consecutive batches in EACH "
         "direction keep the device in step with the bound side's shadow while the other side "
         "stays byte-identical (online, target and per-side state), and the side path's CPU/GPU "
         "relative parameter error and loss gap stay inside the required limits");
    return 0;
}

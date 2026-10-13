// GPU 侧按方隔离验证（HANDOFF 测试 3 的按方版本）
//
// 本实现是「单方驻留 + 显式切换」：设备上同一时刻只有被绑定方的参数
// （neural_net.cpp 的 g_gpu_q / g_gpu_target 代表 indep_bound() 那一方）。
// 因此证据链必须是三段式，而不是「绑定 A 时回读 B」：
//
//   L1 设备==影子恒等式：绑定 X 后，设备回读的 online 与 target 都必须逐字节
//      等于 shadow(X)。这证明设备里装的就是 X，没有别的方。
//   L2 另一方影子零变化：online + target 逐字节 + SideUpdateState 全部相同。
//   L3 切回另一方后，设备回读 == 该方在操作前的设备回读（逐字节）——
//      这才是「另一方的设备参数没有被覆盖」的直接证据。
//
// 另外覆盖：按方 target 同步只动指定方；绑定与提交方不一致时必须整体拒绝且
// 任何影子/设备零变化（放在最后，因为会闩住 g_training_fault）。
//
// 所有 6.8MB 级结构都放堆上（栈上会 0xC00000FD），全部比较都是逐字节。
#include "independent_networks.h"
#include "sentinel_guard.h"
#include "gpu_types.h"
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

/* 与 neural_net.cpp 的 pack_params 同布局：只取可学习参数区。 */
void pack_local(const NeuralNet* net, GpuNetParams* p) {
    memcpy(p->token_embed, net->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, net->pos_embed, sizeof(p->pos_embed));
    memcpy(p->w1, net->w1, sizeof(p->w1));
    memcpy(p->b1, net->b1, sizeof(p->b1));
    memcpy(p->w2, net->w2, sizeof(p->w2));
    memcpy(p->b2, net->b2, sizeof(p->b2));
    memcpy(p->w3, net->w3, sizeof(p->w3));
    memcpy(p->b3, net->b3, sizeof(p->b3));
    memcpy(p->w4, net->w4, sizeof(p->w4));
    memcpy(p->b4, net->b4, sizeof(p->b4));
    memcpy(p->attn_wq, net->attn_wq, sizeof(p->attn_wq));
    memcpy(p->attn_bq, net->attn_bq, sizeof(p->attn_bq));
    memcpy(p->attn_wk, net->attn_wk, sizeof(p->attn_wk));
    memcpy(p->attn_bk, net->attn_bk, sizeof(p->attn_bk));
    memcpy(p->attn_wv, net->attn_wv, sizeof(p->attn_wv));
    memcpy(p->attn_bv, net->attn_bv, sizeof(p->attn_bv));
    memcpy(p->attn_wo, net->attn_wo, sizeof(p->attn_wo));
    memcpy(p->attn_bo, net->attn_bo, sizeof(p->attn_bo));
    memcpy(p->pool_w, net->pool_w, sizeof(p->pool_w));
    memcpy(p->pool_b, net->pool_b, sizeof(p->pool_b));
}

GpuNetParams* pack_new(const NeuralNet* net) {
    GpuNetParams* p = new GpuNetParams();
    pack_local(net, p);
    return p;
}

struct ParamDiff { double norm = 0.0; long long different = 0; };

ParamDiff diff_policies(const GpuNetParams& a, const GpuNetParams& b) {
    ParamDiff d;
    const float* fa = (const float*)&a;
    const float* fb = (const float*)&b;
    const size_t n = sizeof(GpuNetParams) / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)fa[i] - (double)fb[i];
        d.norm += x * x;
        if (x != 0.0) d.different++;
    }
    d.norm = sqrt(d.norm);
    return d;
}

ParamDiff diff_shadows(const NeuralNet* a, const NeuralNet* b) {
    std::unique_ptr<GpuNetParams> pa(pack_new(a));
    std::unique_ptr<GpuNetParams> pb(pack_new(b));
    return diff_policies(*pa, *pb);
}

/* 只读取证：回读当前绑定方的 online + target（不写任何影子）。 */
bool snapshot_pair(DQNAgent* agent, Side side, std::unique_ptr<NeuralNet>* online_out,
                   std::unique_ptr<NeuralNet>* target_out) {
    if (!*online_out) online_out->reset(new NeuralNet());
    if (!*target_out) target_out->reset(new NeuralNet());
    return gpu_net_snapshot_side_pair(agent, side_to_physical(side),
        online_out->get(), target_out->get()) == 0;
}

/* L1：绑定 X 后，设备回读的 online/target 必须逐字节等于 X 的影子。 */
bool assert_device_matches_shadow(DQNAgent* agent, Side side, const char* tag) {
    std::unique_ptr<NeuralNet> dev_online, dev_target;
    if (!snapshot_pair(agent, side, &dev_online, &dev_target)) {
        printf("    [%s] side=%s DEVICE read-back FAILED\n", tag, side_name(side));
        fflush(stdout);
        return false;
    }
    const ParamDiff on = diff_shadows(dev_online.get(), &agent->indep->online(side));
    const ParamDiff tg = diff_shadows(dev_target.get(), &agent->indep->target(side));
    printf("    [%s] L1 device==shadow side=%s online L2=%.9g/%lld target L2=%.9g/%lld\n",
        tag, side_name(side), on.norm, on.different, tg.norm, tg.different);
    fflush(stdout);
    return on.different == 0 && tg.different == 0;
}

/* L3：切到 side 后，设备回读必须与 reference 逐字节一致。 */
bool assert_device_matches_ref(DQNAgent* agent, Side side, NeuralNet* ref_online,
                               NeuralNet* ref_target, const char* tag) {
    std::unique_ptr<NeuralNet> dev_online, dev_target;
    if (!snapshot_pair(agent, side, &dev_online, &dev_target)) {
        printf("    [%s] side=%s DEVICE read-back FAILED\n", tag, side_name(side));
        fflush(stdout);
        return false;
    }
    const ParamDiff on = diff_shadows(dev_online.get(), ref_online);
    const ParamDiff tg = diff_shadows(dev_target.get(), ref_target);
    printf("    [%s] L3 device==reference side=%s online L2=%.9g/%lld target L2=%.9g/%lld\n",
        tag, side_name(side), on.norm, on.different, tg.norm, tg.different);
    fflush(stdout);
    return on.different == 0 && tg.different == 0;
}

void fill_replay(std::vector<Experience>& out, int side, int samples) {
    out.assign(samples, Experience{});
    const float placed = (side == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < samples; ++i) {
        GameState g; memset(&g, 0, sizeof(g));
        g.turn = side;
        g.board[(side == 0) ? 0 : 24] = placed;
        const int other = 5 + (i % 10);
        if (g.board[other] == 0) g.board[other] = -placed;
        int moves[CELL_COUNT];
        const int n = get_legal_moves(&g, moves);
        if (n <= 0) { out[i].action = -1; continue; }
        const int action = moves[i % n];
        GameState mid = g;
        make_move(&mid, action);
        if (check_winner(&mid) != 0 || is_full(&mid)) { out[i].action = -1; continue; }
        /* 第 4 步契约：非终局样本必须跨两 ply（本方落子 + 对手真实回应）。 */
        int replies[CELL_COUNT];
        const int rn = get_legal_moves(&mid, replies);
        if (rn <= 0) { out[i].action = -1; continue; }
        GameState after = mid;
        make_move(&after, replies[(i * 7 + 3) % rn]);
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
    if (src.empty()) return 1;
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
    a->indep->online(side).w4[0][0] += delta;
    a->indep->target(side).b4[1] += delta * 0.5f;
}

} // namespace

int main() {
    printf("=== GPU per-side isolation: L1 device==shadow, L2 other shadow frozen, L3 device ref ===\n");
    fflush(stdout);

    /* 运行库取证：打印本进程实际加载的 amdhip64_7.dll 路径（先落盘再继续）。 */
    {
        HMODULE h = GetModuleHandleA("amdhip64_7.dll");
        char path[MAX_PATH] = { 0 };
        if (h) GetModuleFileNameA(h, path, MAX_PATH);
        printf("[MODULE] amdhip64_7.dll loaded from: %s\n", h ? path : "(NOT LOADED)");
        fflush(stdout);
    }

    g_learning_rate = 0.0001f;
    sentinel_guard_init();

    NEED(gpu_init() == 0);
    printf("gpu_init ok\n"); fflush(stdout);

    auto agent = std::make_unique<DQNAgent>();
    init_dqn_ex(agent.get(), 0);
    g_current_agent = agent.get();
    NEED(indep_ensure(agent.get()) != nullptr);
    perturb(agent.get(), Side::Black, 1e-2f);
    perturb(agent.get(), Side::White, -3e-2f);
    agent->indep->set_active(1);
    agent->indep->set_bound(Side::Black);
    NEED(agent->indep->online(Side::Black).b4[0] != agent->indep->online(Side::White).b4[0]);

    NEED(gpu_net_activate(agent.get()) == 0);
    NEED(gpu_net_active());
    NEED(gpu_net_bind_side(agent.get(), 0) == 0);
    printf("gpu_net_activate + bind(black) ok\n"); fflush(stdout);

    /* 基准快照（堆上） */
    auto black_online_ref = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::Black)));
    auto black_target_ref = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::Black)));
    auto white_online_ref = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::White)));
    auto white_target_ref = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::White)));
    const SideUpdateState white_state_ref = agent->indep->updates(Side::White);
    const SideUpdateState black_state_ref = agent->indep->updates(Side::Black);

    /* 步骤 1：绑定黑方 -> L1 */
    NEED(assert_device_matches_shadow(agent.get(), Side::Black, "step1-bind-black"));

    /* 步骤 2：黑批训练 */
    {
        std::vector<Experience> black, empty;
        fill_replay(black, 0, 2 * BATCH_SIZE);
        NEED(put_replay(agent.get(), black, 0));
        put_replay(agent.get(), empty, 1);

        const float loss = replay_from_buffer(agent.get(), agent->replay_black,
            agent->replay_black_count, &agent->replay_black_idx);
        printf("[step2 train-black] loss=%.9g\n", (double)loss); fflush(stdout);
        NEED(isfinite(loss) && loss > 0.0f);

        /* 设备上的黑方 online 必须真的变了。 */
        std::unique_ptr<NeuralNet> dev_online, dev_target;
        NEED(snapshot_pair(agent.get(), Side::Black, &dev_online, &dev_target));
        const ParamDiff dev_delta = diff_shadows(dev_online.get(), black_online_ref.get());
        const ParamDiff shadow_delta =
            diff_shadows(&agent->indep->online(Side::Black), black_online_ref.get());
        printf("[step2 train-black] black online delta: DEVICE L2=%.9g/%lld shadow L2=%.9g/%lld\n",
            dev_delta.norm, dev_delta.different, shadow_delta.norm, shadow_delta.different);
        fflush(stdout);
        NEED(dev_delta.different > 0 && dev_delta.norm > 0.0);

        /* 黑方 target 未被本次训练改写。 */
        NEED(diff_shadows(&agent->indep->target(Side::Black), black_target_ref.get()).different == 0);
        /* L2：白方影子 online/target/状态零变化。 */
        const ParamDiff w_on = diff_shadows(&agent->indep->online(Side::White), white_online_ref.get());
        const ParamDiff w_tg = diff_shadows(&agent->indep->target(Side::White), white_target_ref.get());
        const bool w_state_same =
            memcmp(&white_state_ref, &agent->indep->updates(Side::White), sizeof(SideUpdateState)) == 0;
        printf("[step2 train-black] L2 white shadow: online L2=%.9g/%lld target L2=%.9g/%lld "
            "state_same=%d; committed black=%llu white=%llu\n",
            w_on.norm, w_on.different, w_tg.norm, w_tg.different, (int)w_state_same,
            agent->indep->updates(Side::Black).committed_batches,
            agent->indep->updates(Side::White).committed_batches);
        fflush(stdout);
        NEED(w_on.different == 0 && w_tg.different == 0 && w_state_same);
        NEED(agent->indep->updates(Side::Black).committed_batches == 1);
        NEED(agent->indep->updates(Side::White).committed_batches == 0);
    }

    /* 步骤 3：记录训练后的黑方影子与按方状态（之后黑方不应再变） */
    auto post_black_online = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::Black)));
    auto post_black_target = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::Black)));
    const SideUpdateState post_black_state = agent->indep->updates(Side::Black);
    printf("[step3] post-black-training black state: committed=%llu rejected=%llu delta_ema=%.9g "
        "warmup=%d sync=%d lr=%.9g\n",
        post_black_state.committed_batches, post_black_state.rejected_batches,
        post_black_state.delta_ema, post_black_state.delta_warmup,
        post_black_state.target_update_counter, post_black_state.learning_rate);
    fflush(stdout);
    NEED(post_black_state.committed_batches == 1 && post_black_state.delta_warmup == 1);
    std::unique_ptr<NeuralNet> dev_black_after_train_online(new NeuralNet());
    std::unique_ptr<NeuralNet> dev_black_after_train_target(new NeuralNet());
    NEED(snapshot_pair(agent.get(), Side::Black, &dev_black_after_train_online,
        &dev_black_after_train_target));

    /* 步骤 4：切到白方 -> L1（设备白 == 白影子）+ L2（白影子仍等于训练前快照）
       四句柄驻留说明：`gpu_net_bind_side` 现在只是兼容 shim（不搬运权重）。下面的
       断言检查的是**驻留本身的语义**——白方设备句柄从未被黑方训练碰过。 */
    {
        /* 驻留证据：黑方训练之后，白方的**设备句柄**必须仍等于训练前的白方权重。 */
        std::unique_ptr<NeuralNet> dev_w_on, dev_w_tg;
        NEED(snapshot_pair(agent.get(), Side::White, &dev_w_on, &dev_w_tg));
        const ParamDiff w_dev_on = diff_shadows(dev_w_on.get(), white_online_ref.get());
        const ParamDiff w_dev_tg = diff_shadows(dev_w_tg.get(), white_target_ref.get());
        printf("[step4] resident white DEVICE after black-only training: online L2=%.9g/%lld "
            "target L2=%.9g/%lld\n", w_dev_on.norm, w_dev_on.different, w_dev_tg.norm, w_dev_tg.different);
        fflush(stdout);
        NEED(w_dev_on.different == 0 && w_dev_tg.different == 0);

        NEED(gpu_net_bind_side(agent.get(), 1) == 0);
        NEED(indep_bound(agent.get()) == Side::White);
        printf("[step4 bind-white]\n"); fflush(stdout);
        /* shim 不改任何影子。 */
        NEED(diff_shadows(&agent->indep->online(Side::White), white_online_ref.get()).different == 0);
        NEED(diff_shadows(&agent->indep->target(Side::White), white_target_ref.get()).different == 0);
        NEED(assert_device_matches_shadow(agent.get(), Side::White, "step4-bind-white"));
        /* 黑方影子在切到白方后也没有被改写。 */
        NEED(diff_shadows(&agent->indep->online(Side::Black), post_black_online.get()).different == 0);
        NEED(diff_shadows(&agent->indep->target(Side::Black), post_black_target.get()).different == 0);
    }

    /* 步骤 5：白批训练 -> 设备白 online 变；黑方影子 == post_black（没被覆盖） */
    {
        std::vector<Experience> white, empty;
        fill_replay(white, 1, 2 * BATCH_SIZE);
        NEED(put_replay(agent.get(), white, 1));
        put_replay(agent.get(), empty, 0);

        const float loss = replay_from_buffer(agent.get(), agent->replay_white,
            agent->replay_white_count, &agent->replay_white_idx);
        printf("[step5 train-white] loss=%.9g\n", (double)loss); fflush(stdout);
        NEED(isfinite(loss) && loss > 0.0f);

        std::unique_ptr<NeuralNet> dev_online, dev_target;
        NEED(snapshot_pair(agent.get(), Side::White, &dev_online, &dev_target));
        const ParamDiff dev_delta = diff_shadows(dev_online.get(), white_online_ref.get());
        const ParamDiff shadow_delta =
            diff_shadows(&agent->indep->online(Side::White), white_online_ref.get());
        printf("[step5 train-white] white online delta: DEVICE L2=%.9g/%lld shadow L2=%.9g/%lld\n",
            dev_delta.norm, dev_delta.different, shadow_delta.norm, shadow_delta.different);
        fflush(stdout);
        NEED(dev_delta.different > 0 && dev_delta.norm > 0.0);
        /* 白方 target 未被本次训练改写。 */
        NEED(diff_shadows(&agent->indep->target(Side::White), white_target_ref.get()).different == 0);

        /* L2 关键：黑方影子（online/target/按方状态）与白方训练前完全一致。
           注意参考点必须是「黑方自己训练完之后」的状态快照（post_black_state），
           不是最开始的全零状态——黑方训练合法地推进了自己的计数与 EMA。 */
        const ParamDiff b_on = diff_shadows(&agent->indep->online(Side::Black), post_black_online.get());
        const ParamDiff b_tg = diff_shadows(&agent->indep->target(Side::Black), post_black_target.get());
        const bool b_state_same =
            memcmp(&post_black_state, &agent->indep->updates(Side::Black), sizeof(SideUpdateState)) == 0;
        printf("[step5 train-white] L2 black shadow: online L2=%.9g/%lld target L2=%.9g/%lld "
            "state_same=%d; committed black=%llu white=%llu\n",
            b_on.norm, b_on.different, b_tg.norm, b_tg.different, (int)b_state_same,
            agent->indep->updates(Side::Black).committed_batches,
            agent->indep->updates(Side::White).committed_batches);
        fflush(stdout);
        NEED(b_on.different == 0 && b_tg.different == 0 && b_state_same);
        NEED(agent->indep->updates(Side::White).committed_batches == 1);
    }

    /* 步骤 6：切回黑方 -> L1 + L3（设备黑 == 黑方训练后设备回读） */
    {
        NEED(gpu_net_bind_side(agent.get(), 0) == 0);
        NEED(indep_bound(agent.get()) == Side::Black);
        printf("[step6 bind-black]\n"); fflush(stdout);
        NEED(assert_device_matches_shadow(agent.get(), Side::Black, "step6-bind-black"));
        /* L3：设备上的黑方参数必须仍然等于「白方训练之前」读到的黑方设备参数。 */
        std::unique_ptr<NeuralNet> dev_online, dev_target;
        NEED(snapshot_pair(agent.get(), Side::Black, &dev_online, &dev_target));
        const ParamDiff on = diff_shadows(dev_online.get(), dev_black_after_train_online.get());
        const ParamDiff tg = diff_shadows(dev_target.get(), dev_black_after_train_target.get());
        printf("    [step6] L3 device(black) vs pre-white-training black device: "
            "online L2=%.9g/%lld target L2=%.9g/%lld\n", on.norm, on.different, tg.norm, tg.different);
        fflush(stdout);
        NEED(on.different == 0 && tg.different == 0);
    }

    /* 步骤 6b：按方 target 同步只动指定方 */
    {
        std::unique_ptr<NeuralNet> b_dev_online_before, b_dev_target_before;
        NEED(snapshot_pair(agent.get(), Side::Black, &b_dev_online_before, &b_dev_target_before));
        auto b_online_shadow_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::Black)));
        auto b_target_shadow_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::Black)));
        auto w_online_shadow_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::White)));
        auto w_target_shadow_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::White)));

        NEED(update_target_side(agent.get(), 0, 1) == 0);   /* 黑方 soft */

        const ParamDiff b_target_moved =
            diff_shadows(&agent->indep->target(Side::Black), b_target_shadow_before.get());
        const ParamDiff b_online_same =
            diff_shadows(&agent->indep->online(Side::Black), b_online_shadow_before.get());
        const ParamDiff w_online_same =
            diff_shadows(&agent->indep->online(Side::White), w_online_shadow_before.get());
        const ParamDiff w_target_same =
            diff_shadows(&agent->indep->target(Side::White), w_target_shadow_before.get());
        std::unique_ptr<NeuralNet> b_dev_online_after, b_dev_target_after;
        NEED(snapshot_pair(agent.get(), Side::Black, &b_dev_online_after, &b_dev_target_after));
        const ParamDiff b_dev_online_same =
            diff_shadows(b_dev_online_after.get(), b_dev_online_before.get());
        const ParamDiff b_dev_target_moved =
            diff_shadows(b_dev_target_after.get(), b_dev_target_before.get());
        printf("[step6b target-sync-black] black target moved shadow L2=%.9g/%lld DEVICE L2=%.9g/%lld; "
            "black online same shadow L2=%.9g/%lld DEVICE L2=%.9g/%lld; "
            "white online L2=%.9g/%lld white target L2=%.9g/%lld\n",
            b_target_moved.norm, b_target_moved.different,
            b_dev_target_moved.norm, b_dev_target_moved.different,
            b_online_same.norm, b_online_same.different,
            b_dev_online_same.norm, b_dev_online_same.different,
            w_online_same.norm, w_online_same.different,
            w_target_same.norm, w_target_same.different);
        fflush(stdout);
        NEED(b_target_moved.different > 0);
        NEED(b_dev_target_moved.different > 0);     /* 设备上的 target 也同步更新了 */
        NEED(b_online_same.different == 0);
        NEED(b_dev_online_same.different == 0);
        NEED(w_online_same.different == 0 && w_target_same.different == 0);
        NEED(agent->indep->updates(Side::Black).target_update_counter == 0);
        NEED(agent->indep->updates(Side::White).target_update_counter == 0);
    }

    /* 步骤 7：驻留语义——训练哪一方只由 side 参数决定，与 gpu_net_bind_side 那条
       兼容记录**无关**。旧实现（单方驻留 + 每批切换）在这里返回 0 并闩住
       g_training_fault（`indep_side_not_bound`）；四句柄驻留下这个状态不存在，
       因此要求：白批在「记录方=黑」时**正常提交**，只动白方，黑方零变化。 */
    {
        std::unique_ptr<NeuralNet> b_dev_online_before, b_dev_target_before;
        NEED(snapshot_pair(agent.get(), Side::Black, &b_dev_online_before, &b_dev_target_before));
        auto b_online_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::Black)));
        auto b_target_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::Black)));
        auto w_online_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->online(Side::White)));
        auto w_target_before = std::unique_ptr<NeuralNet>(new NeuralNet(agent->indep->target(Side::White)));

        float* states = (float*)calloc(BATCH_SIZE * (CELL_COUNT + 1), sizeof(float));
        int* actions = (int*)calloc(BATCH_SIZE, sizeof(int));
        float* targets = (float*)calloc(BATCH_SIZE, sizeof(float));
        for (int i = 0; i < BATCH_SIZE; ++i) targets[i] = 0.1f;
        float loss = 0.0f;
        const int rc = train_agent_batch_side(agent.get(), 1 /*white*/, states, actions,
            targets, BATCH_SIZE, 0.0001f, &loss);

        const ParamDiff b_on = diff_shadows(&agent->indep->online(Side::Black), b_online_before.get());
        const ParamDiff b_tg = diff_shadows(&agent->indep->target(Side::Black), b_target_before.get());
        const ParamDiff w_on = diff_shadows(&agent->indep->online(Side::White), w_online_before.get());
        const ParamDiff w_tg = diff_shadows(&agent->indep->target(Side::White), w_target_before.get());
        printf("[step7 residency] train white while the compatibility record says black -> rc=%d "
            "(independent of that record); black online L2=%.9g/%lld black target L2=%.9g/%lld "
            "white online L2=%.9g/%lld white target L2=%.9g/%lld g_training_fault=%ld loss=%.9g\n",
            rc, b_on.norm, b_on.different, b_tg.norm, b_tg.different,
            w_on.norm, w_on.different, w_tg.norm, w_tg.different, (long)g_training_fault, (double)loss);
        fflush(stdout);
        NEED(rc == 1);                                    /* 正常提交，不再被绑定状态拒绝 */
        NEED(isfinite(loss) && loss > 0.0f);
        NEED(b_on.different == 0 && b_tg.different == 0); /* 黑方零变化 */
        NEED(w_on.different > 0);                         /* 白方 online 真的更新了 */
        NEED(w_tg.different == 0);                        /* 白方 target 仍未被这次训练改写 */
        NEED(g_training_fault == 0);
        free(states); free(actions); free(targets);
    }

    gpu_net_deactivate();
    gpu_shutdown();
    free_dqn(agent.get());
    g_current_agent = nullptr;
    puts("PASS GPU per-side isolation (four resident handles): each side's device handles equal "
         "that side's own shadow; training one side leaves the other side's device handles, "
         "shadow, target and per-side state byte-identical in both directions; each side's "
         "device read-back is unchanged by the other side's training; per-side target sync "
         "touches only its own handles; train_agent_batch_side acts on its side argument "
         "independently of the gpu_net_bind_side compatibility record");
    return 0;
}

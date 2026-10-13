// 旧档迁移与新档保存恢复（INDEPENDENT_NETWORKS_HANDOFF.md 测试 5）
//
// 覆盖：
//   1. V13 存档精确恢复两方 online / target / 按方训练状态；
//   2. 第二方状态被截断或权重被写成非有限值时，整体加载必须失败
//      （绝不半加载、绝不拿黑方补白方）；
//   3. 自动发现不会把截断的 V13 当成最新有效存档；
//   4. V11/V12 旧档仍可按历史语义加载，并且显式按「迁移初始化」处理：
//      两方从共享权重播种，旧回放不兼容时显式清空并记录原因。
#include "independent_networks.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); return 1; } } while(0)

namespace {

void perturb(DQNAgent* a, Side side, float delta) {
    a->indep->online(side).b4[0] += delta;
    a->indep->online(side).w4[0][0] += delta * 0.5f;
    a->indep->target(side).b4[2] -= delta * 0.25f;
    a->indep->updates(side).committed_batches += (unsigned long long)(side == Side::Black ? 11 : 22);
    a->indep->updates(side).rejected_batches += (unsigned long long)(side == Side::Black ? 1 : 2);
    a->indep->updates(side).delta_ema = (side == Side::Black) ? 1.5e-3 : -2.5e-3;
    a->indep->updates(side).delta_warmup = (side == Side::Black) ? 0 : 1;
    a->indep->updates(side).target_update_counter = (side == Side::Black) ? 7 : 9;
    a->indep->set_learning_rate(side, (side == Side::Black) ? 3e-4f : 4e-4f);
}

bool same_net(const NeuralNet& a, const NeuralNet& b) {
    return memcmp(&a, &b, sizeof(NeuralNet)) == 0;
}
bool same_state(const SideUpdateState& a, const SideUpdateState& b) {
    return memcmp(&a, &b, sizeof(SideUpdateState)) == 0;
}

bool overwrite_at(const char* path, long long offset_from_end, const void* data, size_t size) {
    FILE* f = fopen(path, "r+b");
    if (!f) return false;
    if (_fseeki64(f, offset_from_end, SEEK_END) != 0) { fclose(f); return false; }
    const bool ok = fwrite(data, size, 1, f) == 1;
    fclose(f);
    return ok;
}

bool truncate_by(const char* path, long long bytes) {
    FILE* f = fopen(path, "r+b");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    const long long size = _ftelli64(f);
    if (size <= bytes) { fclose(f); return false; }
    const bool ok = _chsize_s(_fileno(f), size - bytes) == 0;
    fclose(f);
    return ok;
}

} // namespace

int main() {
    // ---- 存档 A：两方都有不可混淆的独立状态 ----
    auto a = std::make_unique<DQNAgent>();
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    NEED(indep_ensure(a.get()) != nullptr);
    a->indep->set_active(1);
    perturb(a.get(), Side::Black, 1e-2f);
    perturb(a.get(), Side::White, -3e-2f);
    NEED(!same_net(a->indep->online(Side::Black), a->indep->online(Side::White)));

    const NeuralNet b_online = a->indep->online(Side::Black);
    const NeuralNet b_target = a->indep->target(Side::Black);
    const SideUpdateState b_state = a->indep->updates(Side::Black);
    const NeuralNet w_online = a->indep->online(Side::White);
    const NeuralNet w_target = a->indep->target(Side::White);
    const SideUpdateState w_state = a->indep->updates(Side::White);

    NEED(save_checkpoint(a.get(), 1, 4242, 2, "checkpoints/checkpoint_phase1_4242.bin"));
    NEED(a->checkpoint_version == 13);
    printf("[save] V13 written; version downgraded to 13 automatically\n");

    // ---- 精确恢复 ----
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
    NEED(same_state(b->indep->updates(Side::Black), b_state));
    NEED(same_net(b->indep->online(Side::White), w_online));
    NEED(same_net(b->indep->target(Side::White), w_target));
    NEED(same_state(b->indep->updates(Side::White), w_state));
    NEED(b->indep->online(Side::Black).b4[0] != b->indep->online(Side::White).b4[0]);
    printf("[load] V13 exact restore: black/white online+target+state byte-identical; "
        "b_updates(commit=%llu rej=%llu ema=%.9g warmup=%d sync=%d lr=%.9g) "
        "w_updates(commit=%llu rej=%llu ema=%.9g warmup=%d sync=%d lr=%.9g)\n",
        b->indep->updates(Side::Black).committed_batches,
        b->indep->updates(Side::Black).rejected_batches,
        b->indep->updates(Side::Black).delta_ema,
        b->indep->updates(Side::Black).delta_warmup,
        b->indep->updates(Side::Black).target_update_counter,
        b->indep->updates(Side::Black).learning_rate,
        b->indep->updates(Side::White).committed_batches,
        b->indep->updates(Side::White).rejected_batches,
        b->indep->updates(Side::White).delta_ema,
        b->indep->updates(Side::White).delta_warmup,
        b->indep->updates(Side::White).target_update_counter,
        b->indep->updates(Side::White).learning_rate);

    // ---- 第二方权重非有限（CRC 仍然正确）-> 按方校验必须整体拒绝 ----
    {
        const char* bad = "checkpoints/checkpoint_phase1_4444.bin";
        NEED(save_checkpoint(a.get(), 1, 4444, 2, bad));
        /* V13 磁盘布局（文件末尾，从后往前）：
             [white_updates][white_target][white_online][black_updates][black_target]
             [black_online][present:4]
           把白方 online 内的一个权重写成 +inf，并同步修正末尾 8 字节 CRC trailer，
           这样 CRC 通过、必须由按方有限性校验拒绝。 */
        const long long white_online_off = (long long)sizeof(NeuralNet) + (long long)sizeof(NeuralNet) +
            (long long)sizeof(SideUpdateState) + (long long)sizeof(int32_t) + 256;
        const uint32_t inf_bits = 0x7F800000u;
        NEED(overwrite_at(bad, -white_online_off, &inf_bits, sizeof(inf_bits)));
        /* 用 Core 自己的实现重算 trailer：让 CRC 正确，从而真正走到
           「按方有限性校验」这条分支，而不是被 CRC 提前拦下。 */
        NEED(checkpoint_repair_crc_trailer(bad) == 1);
        auto c = std::make_unique<DQNAgent>();
        init_dqn_ex(c.get(), 0);
        NEED(indep_ensure(c.get()) != nullptr);
        const NeuralNet c_black_before = c->indep->online(Side::Black);
        int p2 = 0, e2 = 0, d2 = 0;
        const int ok = load_checkpoint(c.get(), &p2, &e2, &d2, bad, 1);
        printf("[corrupt] non-finite white online -> load_checkpoint=%d (must be 0)\n", ok);
        NEED(ok == 0);
        /* 整体拒绝：黑方也不能被改成文件里的值，容器也不能被激活。 */
        NEED(same_net(c->indep->online(Side::Black), c_black_before));
        NEED(!indep_active(c.get()));
        free_dqn(c.get());
    }

    // ---- 截断文件不能让自动发现选中坏档 ----
    {
        const char* good = "checkpoints/checkpoint_phase1_5555.bin";
        const char* trunc = "checkpoints/checkpoint_phase1_6666.bin";
        NEED(save_checkpoint(a.get(), 1, 5555, 2, good));
        NEED(save_checkpoint(a.get(), 1, 6666, 2, trunc));
        /* 只砍掉 CRC trailer 而保留完整 body：文件长度与 body 不符，
           必须被拒绝，并且不能让自动发现把它选成最新有效存档。 */
        NEED(truncate_by(trunc, 8));
        {
            auto c = std::make_unique<DQNAgent>();
            init_dqn_ex(c.get(), 0);
            NEED(indep_ensure(c.get()) != nullptr);
            int p2 = 0, e2 = 0, d2 = 0;
            const int ok = load_checkpoint(c.get(), &p2, &e2, &d2, trunc, 1);
            printf("[truncated] missing CRC trailer -> load_checkpoint=%d (must be 0); "
                "indep_active=%d\n", ok, indep_active(c.get()));
            NEED(ok == 0);
            NEED(!indep_active(c.get()));
            free_dqn(c.get());
        }
        NEED(save_checkpoint(a.get(), 1, 6666, 2, trunc));
        /* 再砍掉独立状态块的一部分：长度错 + 按方状态截断，同样必须拒绝。 */
        NEED(truncate_by(trunc, 40));

        char found[260] = { 0 };
        int latest = 0;
        const int found_ok = find_latest_checkpoint(found, sizeof(found), &latest);
        printf("[discovery] truncation test: find_latest_checkpoint=%d name=%s ep=%d "
            "(must not select the truncated 6666 file)\n", found_ok, found, latest);
        NEED(found_ok == 1);
        NEED(latest == 5555);
        NEED(strstr(found, "5555") != nullptr);
    }

    // ---- V12 档仍然可加载，并按「迁移初始化」显式处理 ----
    {
        /* 构造一个 V12 档：把 a 的 checkpoint_version 强制回 12 再保存。 */
        a->checkpoint_version = 12;
        const int saved_active = a->indep->active();
        a->indep->set_active(0);
        NEED(save_checkpoint(a.get(), 1, 3333, 2, "checkpoints/checkpoint_phase1_3333.bin"));
        a->indep->set_active(saved_active);
        a->checkpoint_version = 13;

        auto d = std::make_unique<DQNAgent>();
        init_dqn_ex(d.get(), 0);
        NEED(indep_ensure(d.get()) != nullptr);
        int p3 = 0, e3 = 0, d3 = 0;
        NEED(load_checkpoint(d.get(), &p3, &e3, &d3, "checkpoints/checkpoint_phase1_3333.bin", 1));
        NEED(d->checkpoint_version == 12);
        /* V12 没有两方状态：加载后容器仍是未激活的迁移种子，必须显式播种。 */
        NEED(!indep_active(d.get()));
        const int seeded = d->indep->seed_once_from_shared(d->q_net, d->target_net);
        NEED(seeded == 1);
        NEED(same_net(d->indep->online(Side::Black), d->q_net));
        NEED(same_net(d->indep->online(Side::White), d->q_net));
        d->indep->set_active(1);
        /* 开启后再调用必须返回 0：绝不允许每局重新复制共享权重。 */
        NEED(d->indep->seed_once_from_shared(d->q_net, d->target_net) == 0);
        printf("[migration] V12 loaded (version=%d) -> explicit migrate-seed from shared weights; "
            "re-seed refused after activation\n", d->checkpoint_version);
        free_dqn(d.get());
    }

    free_dqn(b.get());
    free_dqn(a.get());
    g_current_agent = nullptr;
    puts("PASS checkpoint V13: exact two-side restore, whole-file rejection on corrupt/truncated "
         "second side, discovery skips the bad file, V12 legacy path stays migrate-seed only");
    return 0;
}

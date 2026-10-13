#pragma once
#include "core.h"
#include <memory>
#include <stdexcept>

/* ============================================================================
   黑白独立网络：物理棋色角色、所有权与按方训练状态

   GPT6-Astra 提供了基础容器（仅权重 + 空状态）。本文件在其之上补齐生产所需
   的语义，是「物理棋色」的唯一权威定义：

     Side::Black == 物理执黑（GameState.turn == 0 / board 值 1）
     Side::White == 物理执白（GameState.turn == 1 / board 值 2）

   **物理棋色与「本局在线方 / 冻结对手方」是两件不同的事。**
   在线方是「本局谁在学」，每局轮换；物理棋色是「棋盘上谁执黑」。
   任何只接受一个 int 的接口都必须显式说明它收的是哪一种，禁止互相传递。

   所有权约定：
     - 每方各自拥有 online / target 两组权重，堆分配，禁止浅拷贝与隐式别名。
     - 每方各自拥有 delta 门 EMA/预热、提交/拒绝计数、目标同步计数与学习率。
       一方的健康统计绝不能限制或推进另一方的更新。
     - 进程级状态（训练故障闩锁、GPU 设备健康）可以共用，但不得当成模型健康统计。

   当前优化器是纯 SGD，没有动量/二阶矩缓冲。将来引入优化器状态时必须按方保存，
   不得退回进程全局单例（见 INDEPENDENT_NETWORKS_HANDOFF.md 第 12 行）。
   ========================================================================= */
namespace independent_training {

enum class Side { Black = 0, White = 1 };

/* 显式命名转换：所有 int 与 Side 之间的跨越只能走这两个函数。 */
inline Side side_from_physical(int physical) {
    if (physical == 0) return Side::Black;
    if (physical == 1) return Side::White;
    throw std::invalid_argument("physical chess side must be 0 (black) or 1 (white)");
}

/* 物理棋色 -> 0/1；越界返回 -1，供必须返回 int 的调用方显式检查。 */
inline int side_to_physical(Side side) {
    if (side == Side::Black) return 0;
    if (side == Side::White) return 1;
    return -1;
}

/* 日志/断言用名字，避免把棋色打成无意义的 0/1。 */
inline const char* side_name(Side side) {
    return side == Side::Black ? "black" : "white";
}

inline Side opposite_side(Side side) {
    return side == Side::Black ? Side::White : Side::Black;
}

/* 按方训练状态。原先这些量都是 neural_net.cpp / sentinel_guard.cpp 里的进程
   全局静态量，黑方一批的范数突增会把白方的 DELTA 门 EMA 拉高，从而限制白方
   提交；学习率回退也会同时改写两方。此处把它们逐项归属到棋色。 */
struct SideUpdateState {
    double delta_ema = 0.0;               /* 整批参数更新范数 EMA（提交侧安全门） */
    int delta_warmup = 2;                 /* 前 2 批只做基线，不做突增判定 */
    unsigned long long committed_batches = 0;
    unsigned long long rejected_batches = 0;      /* 所有安全拒绝的总和 */
    unsigned long long rejected_delta_gate = 0;   /* 仅参数范数突增门 */
    unsigned long long rejected_loss_batches = 0; /* GPU 高 loss 提交前拒绝 */
    int target_update_counter = 0;        /* 该方目标同步相位（HARD 用） */
    float learning_rate = 0.0f;           /* 0 = 尚未初始化，回退到进程学习率 */
    int sentinel_consecutive_rejects = 0; /* 连续哨兵拒绝（诊断用，按方） */
};

struct SideNetworks {
    NeuralNet online{};
    NeuralNet target{};
    SideUpdateState updates{};
};

/* 按方路由开关 + 当前绑定方。
   active == 0（默认）时整个生产路径逐字节保持历史行为：一切仍然作用于
   agent->q_net / target_net。只有生产调用方显式开启后才会改走按方网络。 */
struct SideRouting {
    int active = 0;
    Side bound = Side::Black;   /* GPU 句柄当前代表哪一方 */
};

} // namespace independent_training

/* 容器：双方 online/target + 按方状态。堆分配、禁拷贝。
   构造只迁移权重；构造函数绝不迁移回放语义（旧一步回放不等于新语义经验）。

   刻意放在全局作用域：DQNAgent（C 兼容布局）里必须能前向声明同一个
   IndependentNetworks 类型，命名空间类型无法在那里不完整声明。 */
class IndependentNetworks {
    std::unique_ptr<independent_training::SideNetworks> black_;
    std::unique_ptr<independent_training::SideNetworks> white_;
    independent_training::SideRouting routing_{};

    independent_training::SideNetworks* slot(independent_training::Side side) {
        return side == independent_training::Side::Black ? black_.get() : white_.get();
    }
    const independent_training::SideNetworks* slot(independent_training::Side side) const {
        return side == independent_training::Side::Black ? black_.get() : white_.get();
    }

public:
    using Side = independent_training::Side;
    using SideNetworks = independent_training::SideNetworks;
    using SideUpdateState = independent_training::SideUpdateState;

    /* 显式把当前方绑定到某个物理棋色。析构时恢复进入时的绑定，因此嵌套或
       提前 return 都不会把绑定状态留成半开。 */
    class ScopedSide {
        IndependentNetworks* owner_;
        Side previous_;
        Side side_;
    public:
        ScopedSide(IndependentNetworks* owner, Side side)
            : owner_(owner), previous_(Side::Black), side_(side) {
            if (!owner_) throw std::invalid_argument("ScopedSide requires an owner");
            owner_->at(side);                 /* 越界物理棋色在此立刻抛出 */
            previous_ = owner_->bound();
            owner_->set_bound(side);
        }
        ~ScopedSide() { if (owner_) owner_->set_bound(previous_); }
        ScopedSide(const ScopedSide&) = delete;
        ScopedSide& operator=(const ScopedSide&) = delete;
        Side side() const { return side_; }
    };

    /* 显式把当前方绑定到某个物理棋色，作用域结束后恢复。 */
    ScopedSide bind(Side side) { return ScopedSide(this, side); }

    /* 仅迁移种子权重。不迁移回放、不迁移语义、不声称是共享网络运行的精确续跑。 */
    IndependentNetworks(const NeuralNet& online_seed, const NeuralNet& target_seed)
        : black_(new SideNetworks), white_(new SideNetworks) {
        black_->online = white_->online = online_seed;
        black_->target = white_->target = target_seed;
    }

    IndependentNetworks(const IndependentNetworks&) = delete;
    IndependentNetworks& operator=(const IndependentNetworks&) = delete;

    /* 显式物理棋色访问：黑白双方各一组 online/target/state。 */
    SideNetworks& at(Side side) {
        if (side != Side::Black && side != Side::White)
            throw std::invalid_argument("invalid physical chess side");
        return *slot(side);
    }
    const SideNetworks& at(Side side) const {
        if (side != Side::Black && side != Side::White)
            throw std::invalid_argument("invalid physical chess side");
        return *slot(side);
    }

    NeuralNet& online(Side side) { return at(side).online; }
    NeuralNet& target(Side side) { return at(side).target; }
    SideUpdateState& updates(Side side) { return at(side).updates; }
    const NeuralNet& online(Side side) const { return at(side).online; }
    const NeuralNet& target(Side side) const { return at(side).target; }
    const SideUpdateState& updates(Side side) const { return at(side).updates; }

    /* ---- 路由 ---------------------------------------------------------- */
    int active() const { return routing_.active; }
    Side bound() const { return routing_.bound; }
    void set_active(int on) { routing_.active = on ? 1 : 0; }
    void set_bound(Side side) { at(side); routing_.bound = side; }

    /* 打开按方路由前必须显式播种：由共享预训练/恢复权重克隆出两方。
       之后绝不再自动复制——每局重新复制等于让两方永远相同。 */
    void seed_both_from(const NeuralNet& online_seed, const NeuralNet& target_seed) {
        black_->online = white_->online = online_seed;
        black_->target = white_->target = target_seed;
    }

    /* 只从共享网络补齐尚未开启路由时的初始化权重。
       返回 1 表示执行了播种，0 表示路由已激活（禁止重新复制）。 */
    int seed_once_from_shared(const NeuralNet& shared_online, const NeuralNet& shared_target) {
        if (routing_.active) return 0;
        seed_both_from(shared_online, shared_target);
        return 1;
    }

    /* ---- 按方 Q 路由 --------------------------------------------------- */
    /* 路由开启时返回该方自己的 online；关闭时返回 fallback（= 历史共享 q_net）。 */
    NeuralNet* online_or(NeuralNet* fallback, Side side) {
        return routing_.active ? &online(side) : fallback;
    }
    const NeuralNet* online_or(const NeuralNet* fallback, Side side) const {
        return routing_.active ? &online(side) : fallback;
    }

    /* ---- 按方 DELTA 门 ------------------------------------------------- */
    /* 判断本批是否允许提交。warmup 内的批次只建立基线。
       纯查询，不修改状态；通过后再调用 commit_delta。 */
    int delta_gate_allows(Side side, double delta_norm,
        double surge_ratio, double epsilon, double* out_ema) const {
        const SideUpdateState& u = at(side).updates;
        if (out_ema) *out_ema = u.delta_ema;
        if (u.delta_warmup > 0) return 1;
        if (!(delta_norm == delta_norm) /* NaN */) return 0;
        return delta_norm <= surge_ratio * u.delta_ema + epsilon;
    }

    /* 提交一次 DELTA 门结果。allowed==0 时记录按方拒绝计数且不污染健康 EMA。 */
    void commit_delta(Side side, double delta_norm, int allowed, double ema_alpha) {
        SideUpdateState& u = at(side).updates;
        if (!allowed) { u.rejected_delta_gate++; return; }
        if (u.delta_warmup > 0) {
            --u.delta_warmup;
            u.delta_ema = delta_norm;   /* 用最后一个 warmup 批初始化健康基线 */
            return;
        }
        u.delta_ema += ema_alpha * (delta_norm - u.delta_ema);
    }

    /* 按方学习率：未初始化时用 fallback（进程学习率）补齐并记录。 */
    float learning_rate_or(Side side, float fallback) {
        SideUpdateState& u = at(side).updates;
        if (!(u.learning_rate > 0.0f)) u.learning_rate = fallback;
        return u.learning_rate;
    }
    void set_learning_rate(Side side, float lr) { at(side).updates.learning_rate = lr; }

    /* ---- 按方计数 ------------------------------------------------------ */
    void note_committed(Side side) { at(side).updates.committed_batches++; }
    void note_rejected(Side side) { at(side).updates.rejected_batches++; }
    void note_hard_target_sync(Side side) {
        at(side).updates.target_update_counter = 0;
    }
    int* target_hard_counter(Side side) { return &at(side).updates.target_update_counter; }
};

/* ============================================================================
   面向 DQNAgent 的薄封装。未分配 indep 时一律回落历史共享语义，
   因此长期存在的测试与未迁移调用方行为不变。
   ========================================================================= */
namespace independent_training {

/* calloc/malloc 出来的 DQNAgent 里 indep 是垃圾指针；初始化时先做这一步。 */
inline void indep_reset(DQNAgent* agent) {
    if (agent) agent->indep = nullptr;
}

/* 已分配则直接返回；否则按需分配并保留当前共享权重作为种子。 */
IndependentNetworks* indep_ensure(DQNAgent* agent);
// GPT6-Astra: synchronize the shared GPU seed before creating four resident nets.
// Already-independent agents are never reseeded; replay migration is caller-owned.
int indep_start_from_shared(DQNAgent* agent);

inline void indep_free(DQNAgent* agent) {
    if (!agent) return;
    delete agent->indep;
    agent->indep = nullptr;
}

inline int indep_active(const DQNAgent* agent) {
    return agent && agent->indep && agent->indep->active();
}

inline Side indep_bound(const DQNAgent* agent) {
    return (agent && agent->indep) ? agent->indep->bound() : Side::Black;
}

/* 路由生效时返回该方 online/target，否则返回共享 q_net/target_net。 */
NeuralNet* agent_online_net(DQNAgent* agent, Side side);
NeuralNet* agent_target_net(DQNAgent* agent, Side side);
const NeuralNet* agent_online_net(const DQNAgent* agent, Side side);
/* D2：以该物理棋色出场的冻结对手。
   路由生效时返回**该方自己的 target 网络**：按方存储、随该方的 soft/hard 目标
   同步连续推进，因此双方各有明确且互不混淆的对手来源（HANDOFF:72 要求「不能
   白方继续使用黑方网络快照而未说明」）。
   刻意不新增第三份 3.4MB 值成员：给 SideNetworks 再加一个 NeuralNet 会让 cl
   编译期堆耗尽（实测 error C1060），而 target 本身就是该方的滞后策略副本。
   路由关闭时返回历史共享 opponent_net。 */
NeuralNet* agent_opponent_net(DQNAgent* agent, Side side);
const NeuralNet* agent_opponent_net(const DQNAgent* agent, Side side);

/* 把共享回放/经验交给正确的物理棋色。这是「一个批次属于同一物理棋色」的入口。 */
int experience_side(const Experience* buf, const DQNAgent* agent, Side* out_side);

} // namespace independent_training

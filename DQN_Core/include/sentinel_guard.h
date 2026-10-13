// sentinel_guard.h
// 小型固定哨兵集保护：在每次训练更新提交前，用一组固定的分布外状态
// （Win-in-1 / Block-in-1 及少量正常局面）检查候选网络是否发生 Q 值发散。
// 这是纯 CPU 前向、不含记忆库叠加的"裸网络"健康检查，用于在灾难性遗忘
// 发生的早期（Q 爆炸 / margin 崩塌 / 非有限值）拒绝提交，而不是事后过滤。
#pragma once
#include "core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 哨兵数量上限：8 个 Win1 + 8 个 Block1 + 少量正常局面 */
#define SENTINEL_MAX_PUZZLES 24

typedef struct {
    GameState puzzle;              /* 哨兵局面 */
    int answers[CELL_COUNT];       /* 正确动作集合 */
    int num_answers;
    int is_win1;                   /* 1 = Win-in-1，0 = Block-in-1，-1 = 正常局面 */
} SentinelPuzzle;

/* 一次哨兵评估的健康度量 */
typedef struct {
    float max_abs_q;       /* 所有哨兵状态的最大 |Q| */
    float min_margin;      /* 最小 margin（best-correct Q - best-wrong Q） */
    float mean_margin;     /* 平均 margin */
    int   nonfinite;       /* 非有限 Q 的哨兵数量 */
    int   correct_count;   /* 选择正确的哨兵数量 */
    int   total;           /* 哨兵总数 */
} SentinelHealth;

/* 初始化哨兵集（固定 seed，确定性）。重复调用是幂等的。 */
void sentinel_guard_init(void);

/* 用纯 CPU 前向评估给定网络的哨兵健康（不叠加记忆库、不走 GPU、不改全局状态）。 */
void sentinel_evaluate_net(const NeuralNet* net, SentinelHealth* out);

/* 判定一次更新是否应被拒绝：after 相比 before 出现 Q 爆炸 / margin 崩塌 /
   非有限值。返回 1 表示拒绝，并把原因写入 reason。 */
int sentinel_reject_update(const SentinelHealth* before, const SentinelHealth* after,
    char* reason, int reason_size);

/* 进程级拒绝计数（诊断/测试用）。 */
unsigned long long sentinel_reject_count(void);

/* 连续拒绝升级：异常批次已经由调用方回滚。达到阈值时仅记录诊断；
   不改写学习率、不暂停 Trainer，避免保护器把可恢复的有限 Q 偏差锁死。 */
void sentinel_rejection_escalate(const char* reason);

/* 一次健康更新后清零连续拒绝计数。 */
void sentinel_rejection_reset(void);

/* 保留兼容 API；新保护策略不会自动进入暂停。 */
int sentinel_guard_paused(void);
void sentinel_guard_resume(void);
const char* sentinel_guard_pause_reason(void);

/* ===== 已移除的 API（不要再加回来）=====
   `sentinel_set_healthy_snapshot` / `sentinel_has_healthy_snapshot` /
   `sentinel_restore_healthy_q` 已删除。它们存/还原的是**共享** `q_net`，
   在按方路由下无法回滚任何一方（消费者读的是 `online(side)`），
   在 GPU 四句柄驻留下甚至连「评测真实用过的网络」都不保证。
   逐方、逐字节的回滚由 `train_agent_batch_side` 内的 before 副本承担，
   见 tests/sentinel_rollback_regression.cpp。 */

#ifdef __cplusplus
}
#endif

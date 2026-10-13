/* memory_weight_scheduler.h — Evaluation-Driven Memory Weight Scheduler
 *
 * 根据多深度 Minimax 评估结果动态调节 Pretrain Memory 采样比例。
 * 基础能力下降 → Memory 权重升高；基础能力稳定 → Memory 权重降低。
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MWS_MAX_DEPTHS 4

/* ========== 单深度评估结果 ========== */
typedef struct {
    int depth;
    uint32_t wins, draws, losses;
    float win_rate, draw_rate, loss_rate;
    float score;    /* R_d = win_rate + draw_score * draw_rate */
} DepthEvalResult;

/* ========== 评估快照（一次完整评估的全部结果） ========== */
typedef struct {
    uint64_t episode;
    DepthEvalResult depths[MWS_MAX_DEPTHS];
    int depth_count;
    float weighted_score;     /* Σ(w_d * R_d) / Σ(w_d) */
    float weakest_score;      /* min(R_d) */
    float overall_score;      /* (1-λ)*weighted + λ*weakest */
    float base_score;         /* 0.7*R1 + 0.3*R2 */
    float strategy_score;     /* 高深度加权 */
    float smoothed_base;      /* EMA 平滑后的 base_score */
    float target_weight;      /* W_min + (W_max-W_min)*(1-S̄_base)^γ */
    float current_weight;     /* 平滑后的实际权重 */
    bool regression;          /* emergency boost 触发标志 */
} MemoryWeightSnapshot;

/* ========== 调度器状态 ========== */
typedef struct {
    /* 配置 */
    uint32_t eval_interval;                     /* 评估间隔（局数），默认 1000 */
    int      depths[MWS_MAX_DEPTHS];            /* 评估深度列表 */
    int      depth_count;                       /* 深度数量 */
    float    draw_score;                        /* 和棋得分，默认 0.30 */
    float    depth_weights[MWS_MAX_DEPTHS];     /* 各深度权重，0=自动生成 2^(d-1) */
    float    weakest_lambda;                    /* 短板权重，默认 0.30 */
    float    base_w1, base_w2;                  /* 基础评分权重，默认 0.70, 0.30 */
    float    w_min, w_max, w_gamma;             /* Memory 权重映射参数 */
    float    initial_weight;                    /* 初始权重，默认 0.10 */
    float    ema_beta;                          /* EMA 平滑系数，默认 0.80 */
    float    weight_update_rate;                /* 权重更新率 η，默认 0.25 */
    float    regression_threshold;              /* 退化阈值 Δ，默认 0.10 */
    float    regression_boost;                  /* 退化加成，默认 0.05 */
    bool     enabled;                           /* 总开关 */
    /* 运行时状态 */
    float    smoothed_base;                     /* EMA 平滑后的 base_score */
    float    current_weight;                    /* 当前实际权重 */
    float    prev_base_score;                   /* 上次 base_score（用于退化检测） */
    bool     initialized;                       /* 是否已初始化 */
    uint64_t last_eval_episode;                 /* 上次评估的 episode */
    /* 最近一次快照（供日志/GUI 读取） */
    MemoryWeightSnapshot last_snapshot;
} MemoryWeightScheduler;

/* ========== API ========== */

/* 初始化调度器（设默认参数） */
static inline void mws_init(MemoryWeightScheduler* s) {
    if (!s) return;
    s->eval_interval = 1000;
    s->depth_count = 3;
    s->depths[0] = 1; s->depths[1] = 2; s->depths[2] = 3; s->depths[3] = 0;
    s->draw_score = 0.30f;
    for (int i = 0; i < MWS_MAX_DEPTHS; i++) s->depth_weights[i] = 0; /* 0=自动生成 */
    s->weakest_lambda = 0.30f;
    s->base_w1 = 0.70f;
    s->base_w2 = 0.30f;
    s->w_min = 0.02f;
    s->w_max = 0.25f;
    s->w_gamma = 2.0f;
    s->initial_weight = 0.10f;
    s->ema_beta = 0.80f;
    s->weight_update_rate = 0.25f;
    s->regression_threshold = 0.10f;
    s->regression_boost = 0.05f;
    s->enabled = true;
    s->smoothed_base = 0.0f;
    s->current_weight = s->initial_weight;
    s->prev_base_score = 0.0f;
    s->initialized = false;
    s->last_eval_episode = 0;
    s->last_snapshot = {};
}

/* 设置评估深度 */
static inline void mws_set_depths(MemoryWeightScheduler* s, const int* depths, int count) {
    if (!s || !depths || count <= 0) return;
    s->depth_count = count < MWS_MAX_DEPTHS ? count : MWS_MAX_DEPTHS;
    for (int i = 0; i < s->depth_count; i++) s->depths[i] = depths[i];
}

/* 判断是否应该执行评估 */
static inline bool mws_should_eval(const MemoryWeightScheduler* s, uint64_t episode) {
    if (!s || !s->enabled || s->eval_interval == 0) return false;
    return (episode > 0) && (episode % s->eval_interval == 0);
}

/* 获取当前 pretrain memory 采样比例 */
static inline float mws_get_pretrain_ratio(const MemoryWeightScheduler* s) {
    if (!s || !s->enabled) return 0.0f;
    return s->current_weight;
}

/* ========== 核心更新函数（在 cpp 中实现） ========== */

/* 根据多深度评估结果更新调度器状态 */
void mws_update(MemoryWeightScheduler* s, const DepthEvalResult* results,
                int result_count, uint64_t episode);

/* 写入 metric_memory_eval.bin */
void mws_write_metric(const MemoryWeightScheduler* s, const char* log_dir);

/* 输出日志摘要到 stdout */
void mws_print_summary(const MemoryWeightScheduler* s);

#ifdef __cplusplus
}
#endif

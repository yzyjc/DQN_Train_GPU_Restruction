/* memory_weight_scheduler.cpp — Evaluation-Driven Memory Weight Scheduler
 *
 * 核心逻辑：多深度评估 → Base/Strategy/Overall Score → EMA → 权重映射 → Emergency Boost
 */
#include "memory_weight_scheduler.h"
#include "metric_log_writer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* 获取深度权重（配置值或自动生成 2^(d-1)） */
static float get_depth_weight(const MemoryWeightScheduler* s, int depth, int index) {
    if (index < MWS_MAX_DEPTHS && s->depth_weights[index] > 0)
        return s->depth_weights[index];
    return powf(2.0f, (float)(depth - 1));
}

void mws_update(MemoryWeightScheduler* s, const DepthEvalResult* results,
                int result_count, uint64_t episode) {
    if (!s || !results || result_count <= 0) return;

    MemoryWeightSnapshot snap = {};
    snap.episode = episode;
    snap.depth_count = result_count < MWS_MAX_DEPTHS ? result_count : MWS_MAX_DEPTHS;

    /* Step 1: 计算各深度得分 R_d */
    for (int i = 0; i < snap.depth_count; i++) {
        DepthEvalResult* d = &snap.depths[i];
        *d = results[i];
        d->score = d->win_rate + s->draw_score * d->draw_rate;
    }

    /* Step 2: 加权分数 S_weighted */
    float w_sum = 0, wr_sum = 0;
    for (int i = 0; i < snap.depth_count; i++) {
        float w = get_depth_weight(s, snap.depths[i].depth, i);
        w_sum += w;
        wr_sum += w * snap.depths[i].score;
    }
    snap.weighted_score = (w_sum > 0) ? wr_sum / w_sum : 0;

    /* Step 3: 短板分数 S_weakest */
    snap.weakest_score = snap.depths[0].score;
    for (int i = 1; i < snap.depth_count; i++)
        if (snap.depths[i].score < snap.weakest_score)
            snap.weakest_score = snap.depths[i].score;

    /* Step 4: 综合分数 S_overall */
    snap.overall_score = (1.0f - s->weakest_lambda) * snap.weighted_score
                       + s->weakest_lambda * snap.weakest_score;

    /* Step 5: Base Score（D1 + D2） */
    if (snap.depth_count >= 2) {
        snap.base_score = s->base_w1 * snap.depths[0].score
                        + s->base_w2 * snap.depths[1].score;
    } else {
        snap.base_score = snap.depths[0].score;
    }

    /* Step 6: Strategy Score（D2+ 加权） */
    if (snap.depth_count >= 3) {
        float sw = 0, swr = 0;
        for (int i = 1; i < snap.depth_count; i++) {
            float w = get_depth_weight(s, snap.depths[i].depth, i);
            sw += w; swr += w * snap.depths[i].score;
        }
        snap.strategy_score = (sw > 0) ? swr / sw : 0;
    } else if (snap.depth_count >= 2) {
        snap.strategy_score = snap.depths[1].score;
    } else {
        snap.strategy_score = snap.base_score;
    }

    /* Step 7: EMA 平滑 base_score */
    if (!s->initialized) {
        s->smoothed_base = snap.base_score;
        s->prev_base_score = snap.base_score;
        s->initialized = true;
    } else {
        s->smoothed_base = s->ema_beta * s->smoothed_base
                         + (1.0f - s->ema_beta) * snap.base_score;
    }
    snap.smoothed_base = s->smoothed_base;

    /* Step 8: 目标权重映射 W_target = W_min + (W_max-W_min)*(1-S̄_base)^γ */
    float sb = s->smoothed_base;
    if (sb < 0) sb = 0; if (sb > 1) sb = 1;
    float t = powf(1.0f - sb, s->w_gamma);
    snap.target_weight = s->w_min + (s->w_max - s->w_min) * t;

    /* Step 9: Emergency Boost 检测 */
    snap.regression = false;
    if (s->prev_base_score > 0) {
        float drop = s->prev_base_score - snap.base_score;
        if (drop > s->regression_threshold) {
            snap.regression = true;
            snap.target_weight += s->regression_boost;
            if (snap.target_weight > s->w_max) snap.target_weight = s->w_max;
        }
    }
    s->prev_base_score = snap.base_score;

    /* Step 10: 权重平滑更新 W_current += η*(W_target - W_current) */
    s->current_weight += s->weight_update_rate * (snap.target_weight - s->current_weight);
    if (s->current_weight < s->w_min) s->current_weight = s->w_min;
    if (s->current_weight > s->w_max) s->current_weight = s->w_max;
    snap.current_weight = s->current_weight;

    /* 保存快照 */
    s->last_snapshot = snap;
    s->last_eval_episode = episode;
}

/* 写入 metric_memory_eval.bin */
void mws_write_metric(const MemoryWeightScheduler* s, const char* log_dir) {
    if (!s || !log_dir) return;
    const MemoryWeightSnapshot* snap = &s->last_snapshot;
    char path[512];
    snprintf(path, sizeof(path), "%smetric_memory_eval.bin", log_dir);
    /* field_count=14: d1w,d1s, d2w,d2s, d3w,d3s, base,strategy,overall,smoothed,target,current,regression */
    FILE* f = metric_log_open_append(path, 14);
    if (!f) return;
    struct { int32_t ep, step; float fields[14]; } rec;
    rec.ep = (int32_t)snap->episode;
    rec.step = 0;
    for (int i = 0; i < 14; i++) rec.fields[i] = 0;
    for (int i = 0; i < snap->depth_count && i < 3; i++) {
        rec.fields[i * 2]     = snap->depths[i].win_rate * 100.0f;
        rec.fields[i * 2 + 1] = snap->depths[i].score * 100.0f;
    }
    rec.fields[6]  = snap->base_score * 100.0f;
    rec.fields[7]  = snap->strategy_score * 100.0f;
    rec.fields[8]  = snap->overall_score * 100.0f;
    rec.fields[9]  = snap->smoothed_base * 100.0f;
    rec.fields[10] = snap->target_weight * 100.0f;
    rec.fields[11] = snap->current_weight * 100.0f;
    rec.fields[12] = snap->regression ? 1.0f : 0.0f;
    rec.fields[13] = 0; /* reserved */
    if (!s->enabled) {
        // No assessment/retention sampler: missing skill scores, actual ratio 0.
        for (int i = 0; i < 10; ++i) rec.fields[i] = NAN;
        rec.fields[10] = rec.fields[11] = rec.fields[12] = 0.0f;
    }
    fwrite(&rec, sizeof(rec), 1, f);
    fclose(f);
}

/* 输出日志摘要 */
void mws_print_summary(const MemoryWeightScheduler* s) {
    if (!s) return;
    const MemoryWeightSnapshot* snap = &s->last_snapshot;
    printf("[MEMORY EVALUATION]\n");
    printf("Episode: %llu\n", (unsigned long long)snap->episode);
    for (int i = 0; i < snap->depth_count; i++) {
        printf("Depth %d: Win=%.1f%% Draw=%.1f%% Loss=%.1f%% Score=%.1f%%\n",
            snap->depths[i].depth,
            snap->depths[i].win_rate * 100, snap->depths[i].draw_rate * 100,
            snap->depths[i].loss_rate * 100, snap->depths[i].score * 100);
    }
    printf("Weighted Score: %.1f%%\n", snap->weighted_score * 100);
    printf("Weakest Score: %.1f%%\n", snap->weakest_score * 100);
    printf("Overall Score: %.1f%%\n", snap->overall_score * 100);
    printf("Base Skill Score: %.1f%%\n", snap->base_score * 100);
    printf("Strategy Score: %.1f%%\n", snap->strategy_score * 100);
    printf("Smoothed Base: %.1f%%\n", snap->smoothed_base * 100);
    printf("Pretrain Memory Target: %.1f%%\n", snap->target_weight * 100);
    printf("Pretrain Memory Current: %.1f%%\n", snap->current_weight * 100);
    printf("Regression Emergency: %s\n", snap->regression ? "YES" : "NO");
}

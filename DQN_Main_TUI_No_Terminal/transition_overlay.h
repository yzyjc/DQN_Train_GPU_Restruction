/* transition_overlay.h — 页面切换粒子动画
 * 字符炸开 → 聚拢 → 重组为新页面 */
#pragma once
#include <windows.h>
#include <stdint.h>
#include "virtual_terminal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 动画阶段 */
typedef enum {
    TRANS_IDLE = 0,
    TRANS_EXPLODE,    /* 炸开：飞出屏幕外 */
    TRANS_PAUSE,      /* 停顿：全部粒子在屏幕外 */
    TRANS_ASSEMBLE    /* 重组：飞回目标位置 */
} TransitionPhase;

/* 单个粒子 */
typedef struct {
    float x, y;           /* 当前像素位置 */
    float vx, vy;         /* 速度 */
    float sx, sy, svx, svy; /* segment origin and velocity */
    float ox, oy; /* deterministic offscreen waypoint */
    wchar_t target_glyph;
    int width, target_width;
    float tx, ty;         /* 目标像素位置 */
    uint32_t fg, bg;      /* 当前颜色 */
    uint32_t target_fg, target_bg;  /* 目标颜色 */
    wchar_t glyph;        /* 字符 */
    float opacity;        /* 透明度 0~1 */
    float target_opacity; /* 目标透明度 */
    int alive;            /* 是否存活 */
} TransitionParticle;

/* 过渡状态 */
typedef struct {
    TransitionPhase phase;
    int active;
    TransitionParticle* particles;
    int particle_count;
    int particle_cap;
    ULONGLONG phase_start;    /* 当前阶段开始时间 */
    int cols, rows;       /* 动画区域尺寸 */
    int content_y0, content_y1;  /* 内容区行范围 */
    int duration_out, duration_pause, duration_in;
    int fast_forward;     /* 快速切换标志 */
    int skip_pause;       /* 跳过停顿阶段（Probe 3D 模式） */
} TransitionState;

/* 动画是否正在进行（供外部判断是否跳过正常渲染） */
int trans_is_active(const TransitionState* st);

/* 初始化/销毁 */
void trans_init(TransitionState* st);
void trans_destroy(TransitionState* st);

/* 触发过渡动画：捕获当前屏幕，准备飞向新页面 */
void trans_start(TransitionState* st, int content_y0, int content_y1, int cols, int rows);

/* 设置目标页面内容（在新页面渲染到 back buffer 后调用） */
void trans_set_target(TransitionState* st);

/* 每帧更新：返回 1 表示动画进行中，0 表示已完成 */
int trans_update(TransitionState* st);

/* 渲染粒子层到 D2D（在 vt_present 之后调用） */
void trans_render_overlay(TransitionState* st);

/* 取消动画（窗口缩放、设备丢失时） */
void trans_cancel(TransitionState* st);

#ifdef __cplusplus
}
#endif

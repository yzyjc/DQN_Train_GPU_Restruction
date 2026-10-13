// terminal_window.h - 自有 Win32 窗口 + 渲染器接入
#pragma once
#include <windows.h>
#include <stdint.h>

extern HWND g_hwnd;
extern int g_cell_w, g_cell_h;   /* 每格像素（字体度量） */
extern int g_running;            /* 主循环退出标志（DQN_Main_TUI.cpp 定义） */

/* 鼠标事件全局（WndProc 喂入，主循环读取；阶段 3 启用） */
extern volatile int g_mouse_click_x, g_mouse_click_y, g_mouse_click_ready;
extern volatile int g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta, g_mouse_wheel_ready;
extern volatile int g_mouse_move_x, g_mouse_move_y, g_mouse_move_ready;   /* hover 追踪（十字轴） */
extern volatile int g_display_config_changed;   /* 显卡/显示器变化，主循环重新探测设备 */

HWND win_create(void);
void win_pump_messages(void);
void win_shutdown(void);
void win_register_renderer(void);
void win_force_redraw(void);   /* 模拟窗口大小改变的完整重绘（D2D后缓冲刷新） */

/* D2D 原始绘图（供粒子动画层使用） */
void d2d_begin_raw(void);
void d2d_end_raw(void);
void d2d_fill_rect(float x, float y, float w, float h, uint32_t color);
void d2d_draw_glyph_at(float x, float y, wchar_t ch, uint32_t fg);

void win_set_scene(int key); /* Interruptible page/modal crossfade before present. */

int d2d_raw_available(void);
void d2d_draw_chrome(int y0, int y1);
void d2d_clip_content(float top, float bottom, int push);

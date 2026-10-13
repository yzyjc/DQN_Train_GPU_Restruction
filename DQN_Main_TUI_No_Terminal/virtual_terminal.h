// virtual_terminal.h - 自有字符终端（与 Windows Terminal 完全解耦）
// 提供 cell buffer 双缓冲 + 原语兼容层（set_color/gotoxy/printf），
// 以及键队列（替代 _getch/_kbhit）。渲染由 VTRenderer 实现（GDI/D2D）。
#pragma once
#include <stdint.h>

/* 一个字符格 */
typedef struct {
    wchar_t glyph;   /* L' ' 空白；宽字符存 lead 格，右半格 VT_ATTR_CONT */
    uint32_t fg;     /* 0x00RRGGBB */
    uint32_t bg;     /* 0x00RRGGBB */
    uint8_t  attr;
} VTCell;

#define VT_ATTR_BOLD 1u
#define VT_ATTR_CONT 2u
#define VT_ATTR_ROLL 4u
typedef struct { wchar_t glyph; float offset; } VTDigitLayer;
int vt_digit_layers(int x, int y, VTDigitLayer layers[4]);
void vt_reset_digit_motion(void);
int vt_roll_printf(const char* fmt, ...);

/* 行内 run（相邻同 fg+bg 合并），由渲染器按 run 绘制 */
typedef struct { int x0, len; uint32_t fg, bg; } VTRun;

/* 曲线 overlay（D2D 真实曲线，非字符折线） */
#define VT_CURVE_MAX 512
#define VT_CURVE_SLOTS 192  /* 9 框 × 17 指标，保留额外槽位 */
typedef struct {
    float pts[VT_CURVE_MAX][2];   /* 归一化 x,y：x 0..1，y 0(底)..1(顶) */
    uint8_t break_before[VT_CURVE_MAX]; /* 该点前存在日志缺口：开启新线段，禁止伪连线 */
    int n;
    int x, y, w, h;               /* 图框内绘图区（字符坐标） */
    uint32_t color;
} VTCurve;

/* 十字轴（hover 准线 + 数值小框） */
typedef struct {
    float x_ratio, y_ratio;   /* 鼠标在 plot 区归一化位置（y 0 底..1 顶） */
    int x, y, w, h;           /* plot 区字符坐标 */
    uint32_t color;
#define VT_CROSSHAIR_LINES 20
    wchar_t lines[VT_CROSSHAIR_LINES][48];
    int line_count;
} VTCrosshair;

/* 渲染器接口（GDI / D2D 实现） */
typedef struct {
    void (*begin_frame)(void);
    void (*draw_line)(int y, const VTCell* line, int cols);
    void (*draw_curve)(const VTCurve* c);         /* 可选：真实曲线 */
    void (*draw_crosshair)(const VTCrosshair* ch); /* 可选：十字轴 */
    void (*draw_area)(int x, int y, int w, int h, const VTCell* cells);   /* 面板遮罩区域：黑底 + 文字，精确覆盖曲线 */
    void (*end_frame)(void);
} VTRenderer;

void vt_init(int cols, int rows);
void vt_resize(int cols, int rows);
void vt_clear_screen(void);
const VTCell* vt_get_back_cell(int x, int y);   /* 读取 back buffer 中的 cell（供动画层使用） */
void vt_clear_rect(int x, int y, int w, int h);
void vt_gotoxy(int x, int y);
void vt_get_cursor(int* x, int* y);
void vt_set_color(int fg, int bg);                 /* enum Color 0-15 → Campbell 调色板 */
uint32_t vt_palette_rgb(int index);                /* 取调色板 RGB（enum Color → 0xRRGGBB） */
void vt_set_truecolor_fg(int r, int g, int b);
void vt_set_truecolor_bg(int r, int g, int b);
void vt_reset_color(void);
int  vt_printf(const char* fmt, ...);              /* 兼容 printf 的虚拟终端输出 */
int  vt_char_width(wchar_t c);                     /* 字符显示宽度：盒线 1 格，CJK/全角 2 格（与渲染对齐） */
int  vt_get_width(void);
int  vt_get_height(void);
int  vt_has_dirty(void);
void vt_mark_all_dirty(void);                      /* 全量重画（target 重建等场景） */
int  vt_animation_active(void);
void vt_set_animation_enabled(int enabled);       /* disable for frame-exact video */
void vt_present(void);                             /* diff + 渲染到窗口 */
void vt_set_renderer(const VTRenderer* r);
void vt_curve_clear(void);                         /* 每帧渲染前清曲线 overlay */
void vt_curve_add(const VTCurve* c);               /* 注册一格真实曲线 */
int  vt_curve_count(void);                         /* 已注册曲线数（十字准线求交点用） */
const VTCurve* vt_curve_get(int i);                /* 取第 i 条曲线 */
void vt_crosshair_clear(void);
void vt_crosshair_add(const VTCrosshair* ch);      /* 注册十字轴 */
void vt_mask_clear(void);
void vt_mask_add(int x, int y, int w, int h);      /* 面板遮罩：该区域曲线被重画 cell 覆盖 */

/* 键队列（WM_KEYDOWN/WM_CHAR 喂入；替代 _getch/_kbhit） */
void vt_key_push(int code);
int  vt_key_poll(void);        /* 返回下一个键码，无则 -1 */
void vt_wait_any_key(void);    /* 阻塞直到有任意键（viewer/BAV 用） */

const VTCell* vt_get_front_cell(int x, int y);

// GPT6-Astra: scoped text viewport for independently scrolling panels.
void vt_text_clip(int x, int y, int w, int h);
void vt_text_clip_reset(void);

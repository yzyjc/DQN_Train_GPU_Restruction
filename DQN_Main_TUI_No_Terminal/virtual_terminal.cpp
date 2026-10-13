// virtual_terminal.cpp - 自有字符终端实现
// cell 双缓冲、光标/颜色状态机、UTF-8→wchar 解码、ANSI SGR 解析、
// vt_printf（vsnprintf→逐格写入）、行级 diff、键队列。
#define _CRT_SECURE_NO_WARNINGS
#include "virtual_terminal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <windows.h>
#include <vector>
#include "rolling_digit.h"

// Per-cell targets are independent of dirty flags: rebuilding a page must not
// restart an unchanged transition. Retarget from the last displayed colors.
struct VTColorMotion {
    uint32_t from_fg, from_bg, target_fg, target_bg;
    ULONGLONG started;
};
static std::vector<VTColorMotion> g_motion;
static std::vector<RollingDigit> g_digits;
static bool g_reset_digits = false;
static int g_motion_enabled = 1;
static int g_motion_active = 0;
static int g_first_frame = 1;
static const ULONGLONG kMotionMs = 160;
static uint32_t motion_color(uint32_t a, uint32_t b, float t) {
    uint32_t result = 0;
    for (int shift = 0; shift <= 16; shift += 8) {
        int av = (a >> shift) & 255, bv = (b >> shift) & 255;
        result |= (uint32_t)(av + (bv - av) * t + 0.5f) << shift;
    }
    return result;
}
int vt_animation_active(void) { return g_motion_active; }
void vt_set_animation_enabled(int enabled) {
    g_motion_enabled = enabled != 0;
    vt_mark_all_dirty();
}

static VTCell* g_front = NULL;       /* 已上屏内容 */
static VTCell* g_back = NULL;        /* TUI 正在画的内容 */
static uint8_t* g_line_dirty = NULL; /* 每行 1 bit */
static int g_cols = 0, g_rows = 0;
static int g_cur_x = 0, g_cur_y = 0;
static uint32_t g_cur_fg = 0, g_cur_bg = 0;
static uint8_t g_cur_attr = 0;
static int g_inited = 0;
static VTRenderer g_renderer = { NULL, NULL, NULL };

void vt_reset_digit_motion(void) { g_reset_digits = true; vt_mark_all_dirty(); }
int vt_digit_layers(int x, int y, VTDigitLayer layers[4]) {
    if (x<0 || y<0 || x>=g_cols || y>=g_rows) return 0;
    const auto& m=g_digits[y*g_cols+x];
    if (!m.active) return 0;
    int n=0;
    for(int i=0;i<4;++i) if(m.layers[i].glyph) {
        float offset=m.position(i,GetTickCount64());
        if(fabsf(offset)<1.0f) layers[n++]={m.layers[i].glyph,offset};
    }
    return n;
}

/* Campbell 调色板（Windows Terminal 默认，Win11 观感） */
static const uint32_t g_palette[16] = {
    0x0C0C0C, 0x0037DA, 0x13A10E, 0x3A96DD,
    0xC50F1F, 0x881798, 0xC19C00, 0xCCCCCC,
    0x767676, 0x3B78FF, 0x16C60C, 0x39C5BB,   /* COLOR_CYAN=11：初音色 39C5BB，原青色 61D6D6 用于进度条高光 */
    0xE74856, 0xB4009E, 0xF9F1A5, 0xF2F2F2
};

/* 键队列 */
#define VT_KEYQ_MAX 64
static int g_keyq[VT_KEYQ_MAX];
static int g_keyq_head = 0, g_keyq_tail = 0;

/* 曲线 overlay（真实曲线）：槽位数在 virtual_terminal.h 定义（VT_CURVE_SLOTS=64，6框×9指标） */
static VTCurve g_curves[VT_CURVE_SLOTS];
static int g_curve_n = 0;

void vt_curve_clear(void) { g_curve_n = 0; }
void vt_curve_add(const VTCurve* c) {
    if (!c || g_curve_n >= VT_CURVE_SLOTS) return;
    g_curves[g_curve_n++] = *c;
}
int vt_curve_count(void) { return g_curve_n; }
const VTCurve* vt_curve_get(int i) { return (i >= 0 && i < g_curve_n) ? &g_curves[i] : NULL; }

/* 十字轴 */
#define VT_CROSSHAIR_MAX 8
static VTCrosshair g_crosshairs[VT_CROSSHAIR_MAX];
static int g_crosshair_n = 0;
void vt_crosshair_clear(void) { g_crosshair_n = 0; }
void vt_crosshair_add(const VTCrosshair* ch) {
    if (!ch || g_crosshair_n >= VT_CROSSHAIR_MAX) return;
    g_crosshairs[g_crosshair_n++] = *ch;
}

/* 面板遮罩：present 画完曲线后，对遮罩区域重画 cell，遮住该区域曲线 */
#define VT_MASK_MAX 4
typedef struct { int x, y, w, h; } VTMask;
static VTMask g_masks[VT_MASK_MAX];
static int g_mask_n = 0;
void vt_mask_clear(void) { g_mask_n = 0; }
void vt_mask_add(int x, int y, int w, int h) {
    if (g_mask_n >= VT_MASK_MAX) return;
    g_masks[g_mask_n].x = x; g_masks[g_mask_n].y = y;
    g_masks[g_mask_n].w = w; g_masks[g_mask_n].h = h;
    g_mask_n++;
}

/* 字符宽度：盒线/符号 1 格，CJK/全角 2 格（与现有 draw_border 布局一致） */
int vt_char_width(wchar_t c) {
    if (c < 0x20) return 0;
    if (c < 0x7F) return 1;
    if (c >= 0x1100 && c <= 0x115F) return 2;
    if (c == 0x2329 || c == 0x232A) return 2;
    if (c >= 0x2E80 && c <= 0xA4CF) return 2;
    if (c >= 0xAC00 && c <= 0xD7A3) return 2;
    if (c >= 0xF900 && c <= 0xFAFF) return 2;
    if (c >= 0xFE30 && c <= 0xFE6F) return 2;
    if (c >= 0xFF00 && c <= 0xFF60) return 2;
    if (c >= 0xFFE0 && c <= 0xFFE6) return 2;
    return 1;   /* ─│┌┐└┘━ ● █ ◄ ► · 等一律 1 格 */
}

// Clipping preserves off-screen logical cursor positions; no wrapping across panels.
static bool g_text_clipped = false;
static int g_clip_x, g_clip_y, g_clip_right, g_clip_bottom;
void vt_text_clip(int x,int y,int w,int h) {
    g_text_clipped=true; g_clip_x=x; g_clip_y=y; g_clip_right=x+w; g_clip_bottom=y+h;
}
void vt_text_clip_reset() { g_text_clipped=false; }

static void vt_put_cell(wchar_t cp, int w) {
    if (g_text_clipped && (g_cur_x < g_clip_x || g_cur_x + w > g_clip_right ||
        g_cur_y < g_clip_y || g_cur_y >= g_clip_bottom || g_cur_x < 0 ||
        g_cur_x + w > g_cols || g_cur_y < 0 || g_cur_y >= g_rows)) {
        g_cur_x += w; return;
    }
    if (!g_text_clipped && g_cur_x >= g_cols) { g_cur_x = 0; if (g_cur_y < g_rows - 1) g_cur_y++; }
    if (g_cur_y >= g_rows) return;
    int i = g_cur_y * g_cols + g_cur_x;
    g_back[i].glyph = cp;
    g_back[i].fg = g_cur_fg;
    g_back[i].bg = g_cur_bg;
    g_back[i].attr = g_cur_attr;
    if (w == 2 && g_cur_x + 1 < g_cols) {   /* 宽字符右半 CONT 格 */
        g_back[i + 1].glyph = L'\0';
        g_back[i + 1].fg = g_cur_fg;
        g_back[i + 1].bg = g_cur_bg;
        g_back[i + 1].attr = g_cur_attr | VT_ATTR_CONT;
    }
    g_line_dirty[g_cur_y] = 1;
    g_cur_x += w;
    if (!g_text_clipped && g_cur_x >= g_cols) { g_cur_x = 0; if (g_cur_y < g_rows - 1) g_cur_y++; }
}

/* UTF-8 多字节 → wchar_t（1-3 字节，BMP 内） */
static int utf8_decode(const unsigned char* s, wchar_t* out) {
    unsigned char c = s[0];
    if (c < 0x80) { *out = (wchar_t)c; return 1; }
    if ((c & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        *out = (wchar_t)(((c & 0x1F) << 6) | (s[1] & 0x3F)); return 2;
    }
    if ((c & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *out = (wchar_t)(((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F)); return 3;
    }
    *out = (wchar_t)c; return 1;
}

/* ANSI 解析：只处理本项目出现的 4 类（清屏/归位/?序列/SGR），其余安全跳过 */
static const char* vt_parse_ansi(const char* p) {
    if (p[0] != '\033' || p[1] != '[') return p + 1;
    p += 2;
    if (p[0] == '2' && p[1] == 'J') { vt_clear_screen(); return p + 2; }
    if (p[0] == 'H') { vt_gotoxy(0, 0); return p + 1; }
    if (p[0] == '?') {
        while (*p && !(*p >= 'A' && *p <= 'Z')) p++;
        return *p ? p + 1 : p;
    }
    /* SGR：0 / 1 / 22 / 38;2;r;g;b / 48;2;r;g;b */
    int params[8];
    int np = 0;
    while (*p && np < 8 && *p != 'm') {
        int v = 0;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        params[np++] = v;
        if (*p == ';') p++;
    }
    if (*p == 'm') p++;
    for (int i = 0; i < np; i++) {
        int v = params[i];
        if (v == 0) vt_reset_color();
        else if (v == 1) g_cur_attr |= VT_ATTR_BOLD;
        else if (v == 22) g_cur_attr &= ~VT_ATTR_BOLD;
        else if (v == 38 || v == 48) {
            if (i + 4 < np && params[i + 1] == 2) {
                if (v == 38) vt_set_truecolor_fg(params[i + 2], params[i + 3], params[i + 4]);
                else vt_set_truecolor_bg(params[i + 2], params[i + 3], params[i + 4]);
                i += 4;
            }
        }
    }
    return p;
}

void vt_init(int cols, int rows) {
    if (cols < 1) cols = 1;
    if (rows < 1) rows = 1;
    if ((size_t)cols > SIZE_MAX / (size_t)rows / sizeof(VTCell)) return;
    size_t cell_count = (size_t)cols * (size_t)rows;
    // Allocate before replacing existing buffers, including during resize.
    std::vector<VTColorMotion> new_motion;
    std::vector<RollingDigit> new_digits;
    try { new_motion.resize(cell_count); new_digits.resize(cell_count); } catch (...) { return; }
    VTCell* new_front = (VTCell*)malloc(cell_count * sizeof(VTCell));
    VTCell* new_back = (VTCell*)malloc(cell_count * sizeof(VTCell));
    uint8_t* new_dirty = (uint8_t*)malloc((size_t)rows);
    if (!new_front || !new_back || !new_dirty) {
        free(new_front); free(new_back); free(new_dirty);
        return; /* 保留旧缓冲和旧尺寸，避免 resize 内存不足后空指针崩溃 */
    }
    free(g_front); free(g_back); free(g_line_dirty);
    g_motion.swap(new_motion);
    g_digits.swap(new_digits);
    g_first_frame = 1;
    g_motion_active = 0;
    g_front = new_front;
    g_back = new_back;
    g_line_dirty = new_dirty;
    g_cols = cols; g_rows = rows;
    for (int i = 0; i < cols * rows; i++) {
        g_front[i].glyph = L' '; g_front[i].fg = 0; g_front[i].bg = 0; g_front[i].attr = 0;
        g_back[i].glyph = L' '; g_back[i].fg = 0; g_back[i].bg = 0; g_back[i].attr = 0;
    }
    memset(g_line_dirty, 1, (size_t)rows);
    g_cur_x = 0; g_cur_y = 0;
    vt_reset_color();
    g_inited = 1;
}

void vt_resize(int cols, int rows) {
    if (cols == g_cols && rows == g_rows && g_inited) return;
    vt_init(cols, rows);
}

void vt_clear_screen(void) {
    if (!g_inited) return;
    for (int i = 0; i < g_cols * g_rows; i++) {
        g_back[i].glyph = L' ';
        g_back[i].fg = g_cur_fg;
        g_back[i].bg = g_cur_bg;
        g_back[i].attr = 0;
    }
    memset(g_line_dirty, 1, (size_t)g_rows);
    g_cur_x = 0; g_cur_y = 0;
}

void vt_clear_rect(int x, int y, int w, int h) {
    if (!g_inited) return;
    for (int yy = y; yy < y + h && yy < g_rows; yy++) {
        if (yy < 0) continue;
        for (int xx = x; xx < x + w && xx < g_cols; xx++) {
            if (xx < 0) continue;
            int i = yy * g_cols + xx;
            g_back[i].glyph = L' ';
            g_back[i].fg = g_cur_fg;
            g_back[i].bg = g_cur_bg;
            g_back[i].attr = 0;
        }
        g_line_dirty[yy] = 1;
    }
}

const VTCell* vt_get_back_cell(int x, int y) {
    if (!g_inited || x < 0 || x >= g_cols || y < 0 || y >= g_rows) return NULL;
    return &g_back[y * g_cols + x];
}

void vt_gotoxy(int x, int y) {
    if(g_text_clipped){g_cur_x=x;g_cur_y=y;return;}
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (g_cols > 0 && x >= g_cols) x = g_cols - 1;
    if (g_rows > 0 && y >= g_rows) y = g_rows - 1;
    g_cur_x = x; g_cur_y = y;
}

void vt_set_color(int fg, int bg) {
    g_cur_fg = g_palette[fg & 15];
    g_cur_bg = g_palette[bg & 15];
}
uint32_t vt_palette_rgb(int index) { return g_palette[index & 15]; }
void vt_set_truecolor_fg(int r, int g, int b) {
    g_cur_fg = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}
void vt_set_truecolor_bg(int r, int g, int b) {
    g_cur_bg = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}
void vt_reset_color(void) {
    g_cur_fg = g_palette[7];   /* 亮灰（原 COLOR_WHITE） */
    g_cur_bg = g_palette[0];   /* 黑 */
    g_cur_attr = 0;
}

int vt_printf(const char* fmt, ...) {
    if (!g_inited) return 0;
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) return 0;
    const unsigned char* p = (const unsigned char*)buf;
    while (*p) {
        if (*p == '\033') { p = (const unsigned char*)vt_parse_ansi((const char*)p); continue; }
        if (*p == '\n') { vt_gotoxy(0, g_cur_y < g_rows - 1 ? g_cur_y + 1 : g_cur_y); p++; continue; }
        if (*p == '\r') { vt_gotoxy(0, g_cur_y); p++; continue; }
        if (*p == '\t') { vt_gotoxy(((g_cur_x / 8) + 1) * 8, g_cur_y); p++; continue; }
        wchar_t cp;
        int adv = utf8_decode(p, &cp);
        int w = vt_char_width(cp);
        if (w > 0) vt_put_cell(cp, w);
        p += adv;
    }
    return n;
}

int vt_get_width(void) { return g_cols; }
void vt_get_cursor(int* x, int* y) { *x = g_cur_x; *y = g_cur_y; }
int vt_roll_printf(const char* fmt, ...) {
    char text[4096];va_list ap;va_start(ap,fmt);
    int n=vsnprintf(text,sizeof(text),fmt,ap);va_end(ap);
    if(n<0)return 0;
    uint8_t old=g_cur_attr;g_cur_attr|=VT_ATTR_ROLL;
    vt_printf("%s",text);g_cur_attr=old;
    return n;
}
int vt_get_height(void) { return g_rows; }

int vt_has_dirty(void) {
    if (!g_inited) return 0;
    if (g_motion_active) return 1;
    for (int i = 0; i < g_rows; i++) if (g_line_dirty[i]) return 1;
    return 0;
}

void vt_mark_all_dirty(void) {
    if (g_inited) memset(g_line_dirty, 1, (size_t)g_rows);
}

void vt_present(void) {
    int has_overlay = (g_curve_n > 0 || g_crosshair_n > 0);
    if (!g_inited || !g_renderer.begin_frame) return;
    if (!vt_has_dirty() && !has_overlay) return;   /* 无 cell 变化且无 overlay（曲线/十字轴）时跳过 */
    const ULONGLONG now = GetTickCount64();
    g_motion_active = 0;
    for (int y = 0; y < g_rows; ++y) {
        for (int x = 0; x < g_cols; ++x) {
            const int i = y * g_cols + x;
            VTCell& f = g_front[i];
            const VTCell& b = g_back[i];
            VTColorMotion& m = g_motion[i];
            const bool was_moving = f.fg != m.target_fg || f.bg != m.target_bg;
            if (was_moving) g_line_dirty[y] = 1;
            const bool glyph_changed = f.glyph != b.glyph || f.attr != b.attr;
            bool rolling = g_digits[i].active;
            g_digits[i].update(b.glyph, now, !g_first_frame && !g_reset_digits &&
                g_motion_enabled && (f.attr & VT_ATTR_ROLL) && (b.attr & VT_ATTR_ROLL));
            if (rolling || g_digits[i].active) g_line_dirty[y]=1;
            if (g_digits[i].active) g_motion_active=1;
            if (g_first_frame || !g_motion_enabled) {
                m = {b.fg, b.bg, b.fg, b.bg, now};
                f = b;
            } else {
                // Advance the old target first, even if a live highlight retargets
                // every frame. Otherwise repeated input can freeze the animation.
                ULONGLONG old_elapsed = now - m.started;
                float old_t = old_elapsed >= kMotionMs ? 1.0f : (float)old_elapsed / kMotionMs;
                old_t = 1.0f - (1.0f - old_t) * (1.0f - old_t) * (1.0f - old_t);
                f.fg = motion_color(m.from_fg, m.target_fg, old_t);
                f.bg = motion_color(m.from_bg, m.target_bg, old_t);
                if (m.target_fg != b.fg || m.target_bg != b.bg || glyph_changed) {
                    // Keep new text immediately readable; never queue stale glyphs.
                    uint32_t start_fg = f.fg;
                    if (glyph_changed && f.glyph == L' ' && b.glyph != L' ')
                        start_fg = motion_color(f.bg, b.fg, 0.72f);
                    m = {start_fg, f.bg, b.fg, b.bg, now};
                }
                ULONGLONG elapsed = now - m.started;
                float t = elapsed >= kMotionMs ? 1.0f : (float)elapsed / kMotionMs;
                t = 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
                f.glyph = b.glyph;
                f.attr = b.attr;
                f.fg = motion_color(m.from_fg, m.target_fg, t);
                f.bg = motion_color(m.from_bg, m.target_bg, t);
            }
            if (f.fg != b.fg || f.bg != b.bg) {
                g_line_dirty[y] = 1;
                g_motion_active = 1;
            }
        }
    }
    g_first_frame = 0;
    g_reset_digits = false;
    g_renderer.begin_frame();
    /* 重画 dirty 行及其上下各 1 行（强制，覆盖字形溢出到相邻行导致的上下残影） */
    static uint8_t* to_draw = NULL;
    uint8_t* resized = (uint8_t*)realloc(to_draw, (size_t)g_rows);
    if (!resized) { g_renderer.end_frame(); return; }
    to_draw = resized;
    memset(to_draw, 0, (size_t)g_rows);
    for (int y = 0; y < g_rows; y++) {
        if (g_line_dirty[y]) {
            to_draw[y] = 1;
            if (y > 0) to_draw[y - 1] = 1;
            if (y + 1 < g_rows) to_draw[y + 1] = 1;
        }
    }
    for (int y = 0; y < g_rows; y++) {
        if (!to_draw[y]) continue;
        VTCell* f = &g_front[y * g_cols];
        g_renderer.draw_line(y, f, g_cols);
        g_line_dirty[y] = 0;
    }
    for (int i = 0; i < g_curve_n; i++)   /* 真实曲线 overlay */
        if (g_renderer.draw_curve) g_renderer.draw_curve(&g_curves[i]);
    for (int m = 0; m < g_mask_n; m++) {   /* 面板遮罩：黑底矩形 + 文字，精确覆盖该区域曲线（不动其他区域） */
        VTMask* mk = &g_masks[m];
        if (!g_renderer.draw_area) continue;
        int ax = mk->x, ay = mk->y, aw = mk->w, ah = mk->h;
        if (ax < 0) { aw += ax; ax = 0; }
        if (ay < 0) { ah += ay; ay = 0; }
        if (ax + aw > g_cols) aw = g_cols - ax;
        if (ay + ah > g_rows) ah = g_rows - ay;
        if (aw <= 0 || ah <= 0) continue;
        VTCell* cells = (VTCell*)malloc((size_t)aw * ah * sizeof(VTCell));
        if (cells) {
            for (int yy = 0; yy < ah; yy++)
                for (int xx = 0; xx < aw; xx++)
                    cells[yy * aw + xx] = g_front[(ay + yy) * g_cols + (ax + xx)];
            g_renderer.draw_area(ax, ay, aw, ah, cells);
            free(cells);
        }
    }
    for (int i = 0; i < g_crosshair_n; i++)   /* 十字轴 overlay */
        if (g_renderer.draw_crosshair) g_renderer.draw_crosshair(&g_crosshairs[i]);
    g_renderer.end_frame();
}

void vt_set_renderer(const VTRenderer* r) {
    if (r) g_renderer = *r;
    else memset(&g_renderer, 0, sizeof g_renderer);
}

/* ---------- 键队列 ---------- */
void vt_key_push(int code) {
    int next = (g_keyq_tail + 1) % VT_KEYQ_MAX;
    if (next != g_keyq_head) { g_keyq[g_keyq_tail] = code; g_keyq_tail = next; }
}
int vt_key_poll(void) {
    if (g_keyq_head == g_keyq_tail) return -1;
    int c = g_keyq[g_keyq_head];
    g_keyq_head = (g_keyq_head + 1) % VT_KEYQ_MAX;
    return c;
}
void vt_wait_any_key(void) {
    extern void win_pump_messages(void);
    while (1) {
        if (vt_key_poll() != -1) return;
        win_pump_messages();
        Sleep(10);
    }
}

const VTCell* vt_get_front_cell(int x, int y) {
    if (!g_inited || x < 0 || x >= g_cols || y < 0 || y >= g_rows) return NULL;
    return &g_front[y * g_cols + x];
}

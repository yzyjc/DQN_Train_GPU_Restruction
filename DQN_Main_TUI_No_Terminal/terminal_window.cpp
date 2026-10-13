// terminal_window.cpp - Win32 窗口 + DirectWrite/Direct2D 渲染 + 输入适配
// 渲染：D2D 优先（清晰、fallback 字体），D2D 初始化失败回退 GDI。
// 输入：WM_KEYDOWN/WM_CHAR → 键队列；WM_LBUTTONDOWN/WM_MOUSEWHEEL → 鼠标全局。
#define _CRT_SECURE_NO_WARNINGS
#include "terminal_window.h"
#include "virtual_terminal.h"
#include <stdio.h>
#include <string.h>
#include <map>
#include <vector>
#include <deque>
#include "touch_scroll.h"
#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#include <windowsx.h>
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

HWND g_hwnd = NULL;
int g_cell_w = 10, g_cell_h = 20;
/* g_running 由 DQN_Main_TUI.cpp 定义（terminal_window.h 已 extern 声明） */

volatile int g_mouse_click_x = 0, g_mouse_click_y = 0, g_mouse_click_ready = 0;
volatile int g_mouse_wheel_x = 0, g_mouse_wheel_y = 0, g_mouse_wheel_delta = 0, g_mouse_wheel_ready = 0;
volatile int g_mouse_move_x = 0, g_mouse_move_y = 0, g_mouse_move_ready = 0;
volatile int g_display_config_changed = 0;

static const wchar_t kWinClass[] = L"DQN_TUI_MAIN_WIN";
static const uint32_t kDefBg = 0x0C0C0C;

// GPT6-Astra: reuse the existing wheel dispatcher for all touch scrolling.
static TouchScrollGesture g_touch;
static DWORD g_touch_id = 0;
static bool g_touch_registered = false;
struct TouchWheelEvent { int x, y, delta; };
static std::deque<TouchWheelEvent> g_touch_wheels;
static void cancel_touch() {
    g_touch.cancel();
    g_touch_wheels.clear();
}
static bool promoted_touch_mouse() {
    // Microsoft MI_WP_SIGNATURE plus the touch bit (not pen input).
    return g_touch_registered &&
        (static_cast<ULONG_PTR>(GetMessageExtraInfo()) & 0xffffff80) == 0xff515780;
}
static void touch_update(int x, int y) {
    int ticks = g_touch.update(x, y);
    // Anchor to the initial panel even when the finger crosses its border.
    while (ticks && g_touch_wheels.size() < 64) {
        int sign = ticks > 0 ? 1 : -1;
        g_touch_wheels.push_back({g_touch.start_x / g_cell_w,
            g_touch.start_y / g_cell_h, sign * WHEEL_DELTA});
        ticks -= sign;
    }
}

/* ---------- GDI fallback 渲染器 ---------- */
static HDC g_gdi_hdc = NULL;
static HFONT g_gdi_font = NULL;
static void gdi_begin(void) {
    if (!g_hwnd) return;
    g_gdi_hdc = GetDC(g_hwnd);
    if (g_gdi_hdc && g_gdi_font) SelectObject(g_gdi_hdc, g_gdi_font);
}
static void gdi_line(int y, const VTCell* line, int cols) {
    HDC hdc = g_gdi_hdc;
    if (!hdc) return;
    int ypx = y * g_cell_h;
    RECT r = { 0, ypx, cols * g_cell_w, ypx + g_cell_h };
    HBRUSH def = CreateSolidBrush(RGB(12, 12, 12));
    FillRect(hdc, &r, def);
    DeleteObject(def);
    int x = 0;
    wchar_t wbuf[512];
    while (x < cols) {
        uint32_t fg = line[x].fg, bg = line[x].bg;
        int x0 = x, wn = 0;
        while (x < cols && line[x].fg == fg && line[x].bg == bg) {
            if (!(line[x].attr & VT_ATTR_CONT)) wbuf[wn++] = line[x].glyph;
            x++;
        }
        RECT rr = { x0 * g_cell_w, ypx, x * g_cell_w, ypx + g_cell_h };
        HBRUSH bb = CreateSolidBrush(RGB((bg >> 16) & 255, (bg >> 8) & 255, bg & 255));
        FillRect(hdc, &rr, bb);
        DeleteObject(bb);
        if (wn > 0) {
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, RGB((fg >> 16) & 255, (fg >> 8) & 255, fg & 255));
            TextOutW(hdc, x0 * g_cell_w, ypx, wbuf, wn);
        }
        for (int dx=x0;dx<x;++dx) if(line[dx].attr & VT_ATTR_ROLL) {
            VTDigitLayer digits[4];int n=vt_digit_layers(dx,y,digits);
            if(!n)continue;
            RECT cell={dx*g_cell_w,ypx,(dx+1)*g_cell_w,ypx+g_cell_h};
            SetBkColor(hdc,RGB((bg>>16)&255,(bg>>8)&255,bg&255));
            ExtTextOutW(hdc,cell.left,cell.top,ETO_OPAQUE,&cell,L"",0,NULL);
            for(int d=0;d<n;++d)
                ExtTextOutW(hdc,cell.left,ypx+(int)(digits[d].offset*g_cell_h),ETO_CLIPPED,&cell,&digits[d].glyph,1,NULL);
        }
    }
}
static void gdi_end(void) {
    if (g_gdi_hdc) { ReleaseDC(g_hwnd, g_gdi_hdc); g_gdi_hdc = NULL; }
}
static VTRenderer g_gdi_renderer = { gdi_begin, gdi_line, NULL, NULL, NULL, gdi_end };

/* ---------- DirectWrite/Direct2D 渲染器 ---------- */
static ID2D1Factory* g_d2d = NULL;
static IDWriteFactory* g_dw = NULL;
static ID2D1HwndRenderTarget* g_rt = NULL;
static IDWriteTextFormat* g_fmt = NULL;
static ID2D1Bitmap* g_scene_from = NULL;
static ULONGLONG g_scene_started = 0;
static int g_scene_key = -1;
static bool g_scene_presented = false;
static void scene_discard(void) {
    if (g_scene_from) { g_scene_from->Release(); g_scene_from = NULL; }
}
extern int g_skip_scene_crossfade;
void win_set_scene(int key) {
    if (g_scene_key == key) return;
    bool animate = g_scene_key != -1 && g_scene_presented && !g_skip_scene_crossfade;
    g_scene_key = key;
    vt_reset_digit_motion();
    scene_discard();
    if (!animate || !g_rt || IsIconic(g_hwnd)) return;
    // Capture the already composited frame, including an interrupted fade.
    D2D1_SIZE_U size = g_rt->GetPixelSize();
    D2D1_BITMAP_PROPERTIES props = D2D1::BitmapProperties(g_rt->GetPixelFormat(), 96, 96);
    if (SUCCEEDED(g_rt->CreateBitmap(size, props, &g_scene_from))) {
        if (FAILED(g_scene_from->CopyFromRenderTarget(NULL, g_rt, NULL))) scene_discard();
        else { g_scene_started = GetTickCount64(); vt_mark_all_dirty(); }
    }
}

static std::map<uint32_t, ID2D1SolidColorBrush*> g_brushes;

static bool d2d_init(void) {
    if (g_d2d) return true;
    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &g_d2d))) return false;
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), (IUnknown**)&g_dw))) return false;
    if (FAILED(g_dw->CreateTextFormat(L"Cascadia Mono", NULL, DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 18.0f, L"zh-CN", &g_fmt))) return false;
    g_fmt->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    return true;
}

static void d2d_measure_cell(void) {
    if (!g_dw || !g_fmt) return;
    IDWriteTextLayout* layout = NULL;
    /* 多字符平均 advance（消除首字符/尾随空白误差，保证 DrawText 推进 = cell_w） */
    if (SUCCEEDED(g_dw->CreateTextLayout(L"MMMM", 4, g_fmt, 1000, 1000, &layout))) {
        DWRITE_TEXT_METRICS tm;
        if (SUCCEEDED(layout->GetMetrics(&tm))) {
            g_cell_w = (int)(tm.width / 4.0 + 0.5);
            if (g_cell_w < 8) g_cell_w = 8;
        }
        layout->Release();
    }
    if (SUCCEEDED(g_dw->CreateTextLayout(L"Wg", 2, g_fmt, 1000, 1000, &layout))) {
        DWRITE_TEXT_METRICS tm;
        if (SUCCEEDED(layout->GetMetrics(&tm))) {
            g_cell_h = (int)(tm.height + 0.5);
            if (g_cell_h < 12) g_cell_h = 12;
        }
        layout->Release();
    }
}

static void d2d_ensure_target(void) {
    if (g_rt || !g_d2d || !g_hwnd) return;
    RECT rc; GetClientRect(g_hwnd, &rc);
    g_d2d->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties(g_hwnd, D2D1::SizeU((UINT32)rc.right, (UINT32)rc.bottom), D2D1_PRESENT_OPTIONS_RETAIN_CONTENTS),
        &g_rt);
    if (g_rt) {
        g_rt->SetDpi(96.0f, 96.0f);   /* 强制 DIP = 物理像素：坐标整数即像素整数，杜绝 DPI 缩放模糊 */
        g_rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE);
    }
}

static ID2D1SolidColorBrush* get_brush(uint32_t rgb) {
    std::map<uint32_t, ID2D1SolidColorBrush*>::iterator it = g_brushes.find(rgb);
    if (it != g_brushes.end()) return it->second;
    ID2D1SolidColorBrush* b = NULL;
    if (g_rt) {
        g_rt->CreateSolidColorBrush(D2D1::ColorF(
            (float)((rgb >> 16) & 255) / 255.0f,
            (float)((rgb >> 8) & 255) / 255.0f,
            (float)(rgb & 255) / 255.0f), &b);
        /* 创建失败时不缓存 NULL，否则后续帧永远不会重试该颜色。 */
        if (b) g_brushes[rgb] = b;
    }
    return b;
}

static void d2d_begin(void) {
    if (g_scene_from) vt_mark_all_dirty();
    // Animated true colors must not grow the device brush cache indefinitely.
    // Release only between frames, when no caller retains a borrowed brush.
    if (g_brushes.size() > 1024) {
        for (auto& entry : g_brushes) if (entry.second) entry.second->Release();
        g_brushes.clear();
    }
    if (!d2d_init()) { gdi_begin(); return; }
    if (!g_rt) d2d_ensure_target();
    if (g_rt) g_rt->BeginDraw();   /* BeginDraw 返回 void；错误经 EndDraw 返回 */
}
static void d2d_line(int y, const VTCell* line, int cols) {
    if (!g_rt) return;
    int ypx = y * g_cell_h;
    float top = (float)ypx, bot = (float)(ypx + g_cell_h);
    ID2D1SolidColorBrush* def = get_brush(kDefBg);
    if (def) g_rt->FillRectangle(D2D1::RectF(0, top, (float)(cols * g_cell_w), bot), def);
    /* 背景：同色 run 合并 fill */
    int x = 0;
    while (x < cols) {
        uint32_t fg = line[x].fg, bg = line[x].bg;
        int x0 = x;
        while (x < cols && line[x].fg == fg && line[x].bg == bg) x++;
        if (bg != kDefBg) {
            ID2D1SolidColorBrush* bb = get_brush(bg);
            if (bb) g_rt->FillRectangle(D2D1::RectF((float)(x0 * g_cell_w), top, (float)(x * g_cell_w), bot), bb);
        }
        /* 文本：逐 cell 单字符，绝对格子位（消除 advance 累积偏移导致的边框错位） */
        ID2D1SolidColorBrush* fb = get_brush(fg);
        x = x0;
        while (x < cols && line[x].fg == fg && line[x].bg == bg) {
            if (!(line[x].attr & VT_ATTR_CONT) && line[x].glyph != L'\0' && line[x].glyph != L' ') {
                wchar_t c = line[x].glyph;
                VTDigitLayer digits[4];
                int digit_count = (line[x].attr & VT_ATTR_ROLL) ? vt_digit_layers(x,y,digits) : 0;
                if (digit_count && fb && g_fmt) {
                    g_rt->PushAxisAlignedClip(D2D1::RectF(float(x*g_cell_w),top,float((x+1)*g_cell_w),bot),D2D1_ANTIALIAS_MODE_ALIASED);
                    for(int d=0;d<digit_count;++d) {
                        float shift=digits[d].offset*g_cell_h;
                        g_rt->DrawText(&digits[d].glyph,1,g_fmt,
                            D2D1::RectF(float(x*g_cell_w),top+shift,float((x+1)*g_cell_w),bot+shift),fb,D2D1_DRAW_TEXT_OPTIONS_NONE);
                    }
                    g_rt->PopAxisAlignedClip();
                    ++x;continue;
                }
                if (x + 1 < cols && (line[x + 1].attr & VT_ATTR_CONT)) {   /* CJK/全角 2 格 */
                    if (fb && g_fmt)
                        g_rt->DrawText(&c, 1, g_fmt,
                            D2D1::RectF((float)(x * g_cell_w), top, (float)((x + 2) * g_cell_w), bot),
                            fb, D2D1_DRAW_TEXT_OPTIONS_NONE);
                    x += 2;
                    continue;
                }
                if (fb && g_fmt)
                    g_rt->DrawText(&c, 1, g_fmt,
                        D2D1::RectF((float)(x * g_cell_w), top, (float)((x + 1) * g_cell_w), bot),
                        fb, D2D1_DRAW_TEXT_OPTIONS_NONE);
            }
            x++;
        }
    }
}
/* 外部标志：动画期间跳过场景淡入淡出 */
extern int g_skip_scene_crossfade;

static void d2d_end(void) {
    if (g_rt) {
        if (g_scene_from && !g_skip_scene_crossfade) {
            float t = (float)(GetTickCount64() - g_scene_started) / 200.0f;
            if (t < 1.0f) {
                float opacity = (1.0f - t) * (1.0f - t) * (1.0f - t);
                D2D1_SIZE_F size = g_rt->GetSize();
                // Navigation and shortcuts stay crisp and clickable.
                g_rt->PushAxisAlignedClip(D2D1::RectF(0, 3.0f * g_cell_h,
                    size.width, size.height - 2.0f * g_cell_h), D2D1_ANTIALIAS_MODE_ALIASED);
                g_rt->DrawBitmap(g_scene_from, D2D1::RectF(0, 0, size.width, size.height), opacity);
                g_rt->PopAxisAlignedClip();
                vt_mark_all_dirty();
            } else scene_discard();
        }
        // Cell dimensions truncate the client size. Always erase the remaining
        // right/bottom pixels AFTER overlays so particles cannot leave trails there.
        D2D1_SIZE_F client = g_rt->GetSize();
        float grid_right = (float)(vt_get_width() * g_cell_w);
        float grid_bottom = (float)(vt_get_height() * g_cell_h);
        ID2D1SolidColorBrush* edge = get_brush(kDefBg);
        if (edge) {
            if (grid_right < client.width)
                g_rt->FillRectangle(D2D1::RectF(grid_right, 0, client.width, client.height), edge);
            if (grid_bottom < client.height)
                g_rt->FillRectangle(D2D1::RectF(0, grid_bottom, client.width, client.height), edge);
        }
        HRESULT hr = g_rt->EndDraw();
        g_scene_presented = SUCCEEDED(hr);
        if (hr == D2DERR_RECREATE_TARGET) {
            scene_discard();   /* 窗口句柄变化等导致 target 失效：重建 + 全量重画 */
            for (std::map<uint32_t, ID2D1SolidColorBrush*>::iterator it = g_brushes.begin();
                 it != g_brushes.end(); ++it)
                if (it->second) it->second->Release();
            g_brushes.clear();
            g_rt->Release(); g_rt = NULL;
            vt_mark_all_dirty();
        }
    }
}
static void d2d_draw_curve(const VTCurve* c) {
    if (!g_rt || !c || c->n < 2) return;
    float x0 = (float)(c->x) * g_cell_w;
    float y0 = (float)(c->y) * g_cell_h;
    float pw = (float)(c->w) * g_cell_w;
    float ph = (float)(c->h) * g_cell_h;
    ID2D1PathGeometry* geo = NULL;
    ID2D1GeometrySink* sink = NULL;
    if (FAILED(g_d2d->CreatePathGeometry(&geo))) return;
    if (FAILED(geo->Open(&sink))) { geo->Release(); return; }
    sink->BeginFigure(
        D2D1::Point2F(x0 + c->pts[0][0] * pw, y0 + (1.0f - c->pts[0][1]) * ph),
        D2D1_FIGURE_BEGIN_HOLLOW);
    for (int i = 1; i < c->n; i++) {
        D2D1_POINT_2F pt = D2D1::Point2F(
            x0 + c->pts[i][0] * pw, y0 + (1.0f - c->pts[i][1]) * ph);
        if (c->break_before[i]) {
            sink->EndFigure(D2D1_FIGURE_END_OPEN);
            sink->BeginFigure(pt, D2D1_FIGURE_BEGIN_HOLLOW);
        } else {
            sink->AddLine(pt);
        }
    }
    sink->EndFigure(D2D1_FIGURE_END_OPEN);
    sink->Close();
    sink->Release();
    ID2D1SolidColorBrush* fb = get_brush(c->color);
    if (fb) g_rt->DrawGeometry(geo, fb, 1.5f, NULL);
    geo->Release();
}
/* 曲线在 x 归一化位置处插值 y（返回 0 底..1 顶，越界 -1） */
static float curve_y_at(const VTCurve* c, float xr) {
    if (!c || c->n < 1) return -1.0f;
    if (xr <= c->pts[0][0]) return c->pts[0][1];
    for (int i = 1; i < c->n; i++) {
        if (c->pts[i][0] >= xr) {
            if (c->break_before[i] && xr < c->pts[i][0]) return -1.0f;
            float dx = c->pts[i][0] - c->pts[i - 1][0];
            if (dx < 1e-6f) return c->pts[i][1];
            float t = (xr - c->pts[i - 1][0]) / dx;
            return c->pts[i - 1][1] + t * (c->pts[i][1] - c->pts[i - 1][1]);
        }
    }
    return c->pts[c->n - 1][1];
}
static void d2d_draw_crosshair(const VTCrosshair* ch) {
    if (!g_rt || !ch || ch->w <= 0 || ch->h <= 0) return;
    float x0 = (float)ch->x * g_cell_w;
    float y0 = (float)ch->y * g_cell_h;
    float pw = (float)ch->w * g_cell_w;
    float ph = (float)ch->h * g_cell_h;
    float px = x0 + ch->x_ratio * pw;
    float py = y0 + ch->y_ratio * ph;   /* 小框跟随鼠标屏幕 y（不反转） */
    ID2D1SolidColorBrush* b = get_brush(ch->color);
    if (b) g_rt->DrawLine(D2D1::Point2F(px, y0), D2D1::Point2F(px, y0 + ph), b, 1.0f);   /* 垂直准线 */
    if (ch->line_count <= 0) return;
    /* 数值小框（吸附在指针旁，放不下则反向；宽度按最长文本像素自适应，包住最后一位小数） */
    float bh = (float)ch->line_count * (float)g_cell_h + 8;
    float max_tw = 0;
    if (g_dw && g_fmt) {
        for (int i = 0; i < ch->line_count && i < 8; i++) {
            int l = (int)wcslen(ch->lines[i]);
            if (l <= 0) continue;
            IDWriteTextLayout* tl = NULL;
            if (SUCCEEDED(g_dw->CreateTextLayout(ch->lines[i], (UINT32)l, g_fmt, 10000.0f, 10000.0f, &tl))) {
                DWRITE_TEXT_METRICS tm;
                if (SUCCEEDED(tl->GetMetrics(&tm)) && tm.widthIncludingTrailingWhitespace > max_tw)
                    max_tw = tm.widthIncludingTrailingWhitespace;
                tl->Release();
            }
        }
    }
    float bw = (max_tw > 0) ? (max_tw + 12.0f) : 108.0f;   /* 文本区 = bx+4..bx+bw-4，+8 padding + 余量 */
    float bx = px + 4, by = py - bh / 2;
    if (bx + bw > x0 + pw) bx = px - 4 - bw;
    if (bx < x0) bx = x0;
    if (by < y0) by = y0;
    if (by + bh > y0 + ph) by = y0 + ph - bh;
    ID2D1SolidColorBrush* bb = get_brush(0x202020);
    if (bb) g_rt->FillRectangle(D2D1::RectF(bx, by, bx + bw, by + bh), bb);
    ID2D1SolidColorBrush* wb = get_brush(0xE0E0E0);
    if (wb && g_fmt) {
        for (int i = 0; i < ch->line_count && i < 8; i++) {
            int l = (int)wcslen(ch->lines[i]);
            if (l > 0) g_rt->DrawTextW(ch->lines[i], (UINT32)l, g_fmt,
                D2D1::RectF(bx + 4, by + 3 + (float)i * g_cell_h, bx + bw - 4, by + 3 + (float)(i + 1) * g_cell_h),
                wb, D2D1_DRAW_TEXT_OPTIONS_NONE);
        }
    }
    /* 鼠标准线与各曲线相交处画白点（仅当前 plot 区的曲线；插值求交） */
    ID2D1SolidColorBrush* wp = get_brush(0xF2F2F2);
    if (wp) {
        float r = 1.8f;
        for (int ci = 0; ci < vt_curve_count(); ci++) {
            const VTCurve* cv = vt_curve_get(ci);
            if (!cv || cv->n < 2) continue;
            if (cv->x != ch->x || cv->y != ch->y || cv->w != ch->w || cv->h != ch->h) continue;
            float yr = curve_y_at(cv, ch->x_ratio);
            if (yr < 0.0f || yr > 1.0f) continue;
            float py = y0 + (1.0f - yr) * ph;
            g_rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(px, py), r, r), wp);
        }
    }
}
static void d2d_draw_area(int x, int y, int w, int h, const VTCell* cells) {
    if (!g_rt || w <= 0 || h <= 0) return;
    float lx = (float)(x * g_cell_w), top = (float)(y * g_cell_h);
    float rx = (float)((x + w) * g_cell_w), bot = (float)((y + h) * g_cell_h);
    ID2D1SolidColorBrush* def = get_brush(kDefBg);   /* 黑底覆盖该区域曲线 */
    if (def) g_rt->FillRectangle(D2D1::RectF(lx, top, rx, bot), def);
    for (int yy = 0; yy < h; yy++) {
        for (int xx = 0; xx < w; xx++) {
            const VTCell* c = &cells[yy * w + xx];
            if (c->attr & VT_ATTR_CONT || c->glyph == L'\0' || c->glyph == L' ') continue;
            ID2D1SolidColorBrush* fb = get_brush(c->fg);
            if (!fb || !g_fmt) continue;
            wchar_t gl = c->glyph;
            float cy = (float)((y + yy) * g_cell_h);   /* 每行文字画在对应行 */
            g_rt->DrawText(&gl, 1, g_fmt,
                D2D1::RectF((float)((x + xx) * g_cell_w), cy, (float)((x + xx + 1) * g_cell_w), cy + (float)g_cell_h),
                fb, D2D1_DRAW_TEXT_OPTIONS_NONE);
        }
    }
}
static VTRenderer g_d2d_renderer = { d2d_begin, d2d_line, d2d_draw_curve, d2d_draw_crosshair, d2d_draw_area, d2d_end };

/* ========== D2D 原始绘图 API（供粒子动画层使用） ========== */
int d2d_raw_available(void) { return g_rt != NULL; }
void d2d_begin_raw(void) { scene_discard(); d2d_begin(); }
void d2d_clip_content(float top, float bottom, int push) {
    if (!g_rt) return;
    if (push) g_rt->PushAxisAlignedClip(D2D1::RectF(0,top,g_rt->GetSize().width,bottom),D2D1_ANTIALIAS_MODE_ALIASED);
    else g_rt->PopAxisAlignedClip();
}
void d2d_draw_chrome(int y0, int y1) {
    for (int y=0; y<vt_get_height(); ++y) if (y<y0 || y>=y1) {
        const VTCell* row=vt_get_back_cell(0,y);
        if (row) d2d_line(y,row,vt_get_width());
    }
}
void d2d_end_raw(void) { d2d_end(); }

void d2d_fill_rect(float x, float y, float w, float h, uint32_t color) {
    if (!g_rt) return;
    ID2D1SolidColorBrush* b = get_brush(color);
    if (b) g_rt->FillRectangle(D2D1::RectF(x, y, x + w, y + h), b);
}

void d2d_draw_glyph_at(float x, float y, wchar_t ch, uint32_t fg) {
    if (!g_rt || !g_fmt || ch == L' ' || ch == L'\0') return;
    ID2D1SolidColorBrush* fb = get_brush(fg);
    if (!fb) return;
    g_rt->DrawText(&ch, 1, g_fmt,
        D2D1::RectF(x, y, x + (float)(g_cell_w * (vt_char_width(ch) > 1 ? 2 : 1)), y + (float)g_cell_h),
        fb, D2D1_DRAW_TEXT_OPTIONS_NONE);
}

void win_register_renderer(void) {
    if (d2d_init()) d2d_ensure_target();
    /* D2D factory 成功但 HwndRenderTarget 创建失败时也要回退，否则窗口全黑。 */
    if (g_rt) vt_set_renderer(&g_d2d_renderer);
    else vt_set_renderer(&g_gdi_renderer);
}

/* WM_KEYDOWN 中不产生 WM_CHAR 的键（其余走 WM_CHAR） */
static int map_key(int vk) {
    if (vk == VK_ESCAPE) return 27;
    return 0;
}

/* ---------- 窗口 ---------- */
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd, &ps);
        /* 窗口恢复/解遮挡时 D2D 后缓冲未必仍有效，下一轮必须全量重画。 */
        vt_mark_all_dirty();
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_SIZE: {
        cancel_touch();
        scene_discard();
        g_scene_presented = false;
        int cw = (int)(LOWORD(lp) / (unsigned)g_cell_w);
        int ch = (int)(HIWORD(lp) / (unsigned)g_cell_h);
        if (cw < 40) cw = 40;
        if (ch < 15) ch = 15;
        if (g_rt) {
            g_rt->Resize(D2D1::SizeU(LOWORD(lp), HIWORD(lp)));
            vt_mark_all_dirty();
        }
        vt_resize(cw, ch);
        return 0;
    }
    case WM_DISPLAYCHANGE:
    case WM_DEVICECHANGE:
        g_display_config_changed = 1;
        vt_mark_all_dirty();
        return 0;
    case WM_KEYDOWN: {
        int c = map_key((int)wp);
        if (c) vt_key_push(c);
        return 0;
    }
    case WM_CHAR:
        if ((wp >= 32 && wp < 0x10000) || wp == 8 || wp == 13) vt_key_push((int)wp);   /* 含 Backspace/Enter */
        return 0;
    case WM_TOUCH: {
        UINT count = LOWORD(wp);
        std::vector<TOUCHINPUT> inputs(count);
        HTOUCHINPUT handle = reinterpret_cast<HTOUCHINPUT>(lp);
        if (GetTouchInputInfo(handle, count, inputs.data(), sizeof(TOUCHINPUT))) {
            for (const auto& input : inputs) {
                POINT pt = {TOUCH_COORD_TO_PIXEL(input.x), TOUCH_COORD_TO_PIXEL(input.y)};
                ScreenToClient(hwnd, &pt);
                if ((input.dwFlags & TOUCHEVENTF_DOWN) &&
                    !(input.dwFlags & TOUCHEVENTF_PRIMARY)) {
                    cancel_touch(); // Multi-touch is not a click or a scroll gesture.
                    continue;
                }
                if ((input.dwFlags & TOUCHEVENTF_DOWN) &&
                    (input.dwFlags & TOUCHEVENTF_PRIMARY)) {
                    cancel_touch();
                    g_touch_id = input.dwID;
                    g_touch.begin(pt.x, pt.y, g_cell_h);
                }
                if (!g_touch.active || input.dwID != g_touch_id) continue;
                touch_update(pt.x, pt.y);
                if (input.dwFlags & TOUCHEVENTF_UP) {
                    if (g_touch.end()) {
                        g_mouse_click_x = pt.x / g_cell_w;
                        g_mouse_click_y = pt.y / g_cell_h;
                        g_mouse_click_ready = 1;
                    }
                }
            }
        } else cancel_touch();
        CloseTouchInputHandle(handle);
        return 0;
    }
    case WM_CANCELMODE:
    case WM_KILLFOCUS:
        cancel_touch();
        break;
    case WM_LBUTTONDOWN:
        if (promoted_touch_mouse()) return 0;
        g_mouse_click_x = GET_X_LPARAM(lp) / g_cell_w;
        g_mouse_click_y = GET_Y_LPARAM(lp) / g_cell_h;
        g_mouse_click_ready = 1;
        return 0;
    case WM_MOUSEWHEEL:
        if (promoted_touch_mouse()) return 0;
        /* BUG-053 修复：WM_MOUSEWHEEL lp 是屏幕坐标，需转客户区坐标 */
        { POINT pt = { GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
          ScreenToClient(g_hwnd, &pt);
          g_mouse_wheel_x = pt.x / g_cell_w;
          g_mouse_wheel_y = pt.y / g_cell_h;
        }
        g_mouse_wheel_delta = GET_WHEEL_DELTA_WPARAM(wp);
        g_mouse_wheel_ready = 1;
        return 0;
    case WM_MOUSEMOVE:
        if (promoted_touch_mouse()) return 0;
        g_mouse_move_x = GET_X_LPARAM(lp) / g_cell_w;
        g_mouse_move_y = GET_Y_LPARAM(lp) / g_cell_h;
        g_mouse_move_ready = 1;
        return 0;
    case WM_DPICHANGED:
        /* D2D render target 固定为 96 DPI，字符度量也必须继续使用启动时
           从实际字体测得的像素尺寸。若这里只按显示器 DPI 放大 cell，字体
           并没有同步缩放，切回高 DPI 小屏后网格会变稀，曲线和 Die-shot 被挤掉。 */
        { /* 接受系统建议的窗口矩形，但不改变绘制字体或字符网格的物理像素尺寸。 */
          RECT* rc = (RECT*)lp;
          SetWindowPos(hwnd, NULL, rc->left, rc->top,
                       rc->right - rc->left, rc->bottom - rc->top,
                       SWP_NOZORDER | SWP_NOACTIVATE);
          /* 重设 D2D render target 大小 */
          if (g_rt) {
              RECT cr; GetClientRect(hwnd, &cr);
              g_rt->Resize(D2D1::SizeU(cr.right, cr.bottom));
          }
          /* 重新计算列/行数并通知 vt_resize */
          RECT cr; GetClientRect(hwnd, &cr);
          int cw = cr.right / g_cell_w, ch = cr.bottom / g_cell_h;
          if (cw < 40) cw = 40; if (ch < 15) ch = 15;
          vt_resize(cw, ch);
          vt_mark_all_dirty();
        }
        return 0;
    case WM_GETMINMAXINFO:
        /* 最小窗口尺寸：保证双列图表、状态栏和交互按钮不会互相覆盖。 */
        { MINMAXINFO* mmi = (MINMAXINFO*)lp;
          mmi->ptMinTrackSize.x = (LONG)(120 * g_cell_w);
          mmi->ptMinTrackSize.y = (LONG)(36 * g_cell_h);
        }
        return 0;
    case WM_CLOSE:
        /* 关闭按钮必须直接终止主循环。旧代码只 PostQuitMessage，窗口本身
           没有销毁且 g_running 仍为 1；如果退出消息未被下一轮 Peek 到，
           就会留下 HWND=0/不可见但仍持续刷新的后台 TUI。 */
        g_running = 0;
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        g_running = 0;
        if (g_hwnd == hwnd) g_hwnd = NULL;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

HWND win_create(void) {
    SetConsoleOutputCP(65001);   /* UTF-8 输出，支持 Unicode 方块字符 ░▒▓█ */
    SetConsoleCP(65001);         /* UTF-8 输入 */
    WNDCLASSEXW wc = { sizeof wc };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.hCursor = NULL;
    wc.lpszClassName = kWinClass;
    wc.hIcon = LoadIconW(wc.hInstance, MAKEINTRESOURCEW(101));      /* 大图标：标题栏/Alt+Tab */
    wc.hIconSm = LoadIconW(wc.hInstance, MAKEINTRESOURCEW(101));   /* 小图标：任务栏 */
    /* BUG-066: 检查 RegisterClassExW 返回值 */
    if (!RegisterClassExW(&wc)) {
        /* 类可能已注册（重复调用），继续尝试创建窗口 */
    }

    /* The Probe D3D viewer is a real child HWND.  Excluding child rectangles
       from parent drawing prevents Direct2D/GDI terminal frames from painting
       over a frame that the child has already presented. */
    const DWORD window_style = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN;
    RECT rc = { 0, 0, 1280, 800 };
    AdjustWindowRectEx(&rc, window_style, FALSE, 0);
    g_hwnd = CreateWindowExW(0, kWinClass, L"DQN 训练总控 (GPU 重构)",
        window_style, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return NULL;
    g_touch_registered = RegisterTouchWindow(g_hwnd, 0) != FALSE;

    /* 度量 cell：D2D 优先，失败 GDI */
    if (d2d_init()) {
        d2d_measure_cell();
    } else {
        HDC dc = GetDC(g_hwnd);
        g_gdi_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
        if (g_gdi_font) {
            HFONT old = (HFONT)SelectObject(dc, g_gdi_font);
            TEXTMETRICW tm;
            if (GetTextMetricsW(dc, &tm)) g_cell_h = tm.tmHeight + tm.tmExternalLeading;
            SIZE sz;
            if (GetTextExtentPoint32W(dc, L"M", 1, &sz)) g_cell_w = sz.cx;
            SelectObject(dc, old);
        }
        ReleaseDC(g_hwnd, dc);
    }

    ShowWindow(g_hwnd, SW_SHOW);
    SetFocus(g_hwnd);   /* 键盘焦点：否则收不到 WM_CHAR/WM_KEYDOWN，命令行失效 */
    return g_hwnd;
}

void win_pump_messages(void) {
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            g_running = 0;
            break;   /* WM_QUIT 不是窗口消息，不能再 Translate/Dispatch。 */
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    // Deliver one notch per UI iteration: existing consumers use its sign.
    if (!g_mouse_wheel_ready && !g_touch_wheels.empty()) {
        const auto event = g_touch_wheels.front();
        g_touch_wheels.pop_front();
        g_mouse_wheel_x = event.x; g_mouse_wheel_y = event.y;
        g_mouse_wheel_delta = event.delta; g_mouse_wheel_ready = 1;
    }
}

void win_shutdown(void) {
    scene_discard();
    std::map<uint32_t, ID2D1SolidColorBrush*>::iterator it;
    for (it = g_brushes.begin(); it != g_brushes.end(); ++it) if (it->second) it->second->Release();
    g_brushes.clear();
    if (g_rt) { g_rt->Release(); g_rt = NULL; }
    if (g_fmt) { g_fmt->Release(); g_fmt = NULL; }
    if (g_dw) { g_dw->Release(); g_dw = NULL; }
    if (g_d2d) { g_d2d->Release(); g_d2d = NULL; }
    if (g_gdi_font) { DeleteObject(g_gdi_font); g_gdi_font = NULL; }
    if (g_hwnd) { DestroyWindow(g_hwnd); g_hwnd = NULL; }
}

void win_force_redraw(void) {
    /* 模拟 WM_SIZE 的重绘路径：Resize D2D 后缓冲 + 全行标脏。
       用于 D3D 子窗口隐藏后强制 D2D 刷新被覆盖区域。 */
    if (!g_hwnd || !g_rt) return;
    RECT rc;
    if (!GetClientRect(g_hwnd, &rc)) return;
    D2D1_SIZE_U size = g_rt->GetPixelSize();
    if (size.width != (UINT32)rc.right || size.height != (UINT32)rc.bottom) {
        scene_discard();
        g_scene_presented = false;
        g_rt->Resize(D2D1::SizeU((UINT32)rc.right, (UINT32)rc.bottom));
    }
    vt_mark_all_dirty();
}

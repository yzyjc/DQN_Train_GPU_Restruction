/* Character flight: explode offscreen, pause, then return as the next page.
   Analytic paths use elapsed time, so frame rate does not change the choreography. */
#include "transition_overlay.h"
#include "terminal_window.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

static const int kExplodeMs = 320, kPauseMs = 240, kAssembleMs = 440;
static float det_rand(int seed) {
    uint32_t x = (uint32_t)seed * 2654435761u;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    return (float)(x & 65535) / 65535.0f;
}
void trans_init(TransitionState* st) { if (st) memset(st, 0, sizeof(*st)); }
int trans_is_active(const TransitionState* st) { return st && st->active; }
void trans_cancel(TransitionState* st) {
    if (st) { st->active = 0; st->phase = TRANS_IDLE; vt_mark_all_dirty(); }
}
void trans_destroy(TransitionState* st) {
    if (!st) return;
    free(st->particles); memset(st, 0, sizeof(*st));
}
static bool reserve_particles(TransitionState* st, int count) {
    if (count <= st->particle_cap) return true;
    void* p = realloc(st->particles, (size_t)count * sizeof(TransitionParticle));
    if (!p) { trans_cancel(st); return false; }
    st->particles = (TransitionParticle*)p; st->particle_cap = count;
    return true;
}
static bool visible_cell(const VTCell* c) {
    return c && !(c->attr & VT_ATTR_CONT) &&
        ((c->glyph && c->glyph != L' ') || (c->bg != 0 && c->bg != 0x0C0C0C));
}
static void outer_waypoint(TransitionState* st, TransitionParticle& p, int index) {
    float cx = st->cols * g_cell_w * 0.5f;
    float cy = (st->content_y0 + st->content_y1) * g_cell_h * 0.5f;
    float dx = p.x - cx, dy = p.y - cy;
    float angle = atan2f(dy, dx) + (det_rand(index) - 0.5f) * 0.24f;
    // Circumscribed ellipse: every waypoint lies outside the content rectangle.
    float radius = 1.55f + det_rand(index + 71) * 0.25f;
    p.ox = cx + cosf(angle) * (cx + 2 * g_cell_w) * radius;
    p.oy = cy + sinf(angle) * ((st->content_y1 - st->content_y0) * g_cell_h * 0.5f + 2 * g_cell_h) * radius;
}
void trans_start(TransitionState* st, int y0, int y1, int cols, int rows) {
    if (!st) return;
    bool interrupted = st->active && st->cols == cols && st->rows == rows;
    if (interrupted) trans_update(st);
    interrupted = interrupted && st->active;
    st->cols = cols; st->rows = rows;
    st->content_y0 = (std::max)(0, y0); st->content_y1 = (std::min)(rows, y1);
    if (cols <= 0 || st->content_y1 <= st->content_y0) { trans_cancel(st); return; }
    if (!interrupted) {
        if (!reserve_particles(st, cols * (st->content_y1 - st->content_y0))) return;
        st->particle_count = 0;
        for (int y = st->content_y0; y < st->content_y1; ++y) for (int x = 0; x < cols; ++x) {
            const VTCell* c = vt_get_front_cell(x, y);
            if (!visible_cell(c)) continue;
            TransitionParticle p = {};
            p.x = (float)x * g_cell_w; p.y = (float)y * g_cell_h;
            p.fg = c->fg; p.bg = c->bg; p.glyph = c->glyph;
            p.width = (std::max)(1, vt_char_width(c->glyph));
            p.opacity = 1; p.alive = 1;
            st->particles[st->particle_count++] = p;
        }
    }
    st->fast_forward = interrupted; st->skip_pause = 0;
    st->duration_out = interrupted ? 180 : kExplodeMs;
    st->duration_pause = interrupted ? 80 : kPauseMs;
    st->duration_in = interrupted ? 300 : kAssembleMs;
    for (int i = 0; i < st->particle_count; ++i) {
        auto& p = st->particles[i];
        p.sx = p.x; p.sy = p.y; p.svx = p.vx; p.svy = p.vy;
        p.target_glyph = p.glyph; p.target_width = p.width;
        p.target_fg = p.fg; p.target_bg = p.bg; p.target_opacity = 0;
        outer_waypoint(st, p, i);
    }
    st->phase_start = GetTickCount64(); st->phase = TRANS_EXPLODE; st->active = 1;
}
void trans_set_target(TransitionState* st) {
    if (!st || !st->active) return;
    int old_count = st->particle_count, count = 0;
    for (int y = st->content_y0; y < st->content_y1; ++y) for (int x = 0; x < st->cols; ++x) {
        const VTCell* c = vt_get_back_cell(x, y);
        if (!visible_cell(c)) continue;
        if (!reserve_particles(st, count + 1)) return;
        auto& p = st->particles[count];
        if (count >= old_count) {
            p = {};
            p.x = (float)x * g_cell_w; p.y = (float)y * g_cell_h;
            outer_waypoint(st, p, count);
            p.x = p.sx = p.ox; p.y = p.sy = p.oy;
            p.glyph = c->glyph; p.fg = c->fg; p.bg = c->bg;
            p.width = (std::max)(1, vt_char_width(c->glyph));
            p.alive = 1; // New glyphs wait invisibly outside the viewport.
        }
        p.tx = (float)x * g_cell_w; p.ty = (float)y * g_cell_h;
        p.target_glyph = c->glyph; p.target_width = (std::max)(1, vt_char_width(c->glyph));
        p.target_fg = c->fg; p.target_bg = c->bg; p.target_opacity = 1;
        ++count;
    }
    for (int i = count; i < old_count; ++i) st->particles[i].target_opacity = 0;
    st->particle_count = (std::max)(count, old_count);
}
// Cubic Hermite preserves the velocity when a flight is interrupted.
static void sample_path(float a, float b, float velocity, float t, float seconds, float& pos, float& speed) {
    float t2 = t*t, t3 = t2*t;
    pos = (2*t3-3*t2+1)*a + (t3-2*t2+t)*velocity*seconds + (-2*t3+3*t2)*b;
    speed = ((6*t2-6*t)*a + (3*t2-4*t+1)*velocity*seconds + (-6*t2+6*t)*b) / seconds;
}
int trans_update(TransitionState* st) {
    if (!st || !st->active) return 0;
    ULONGLONG elapsed = GetTickCount64() - st->phase_start;
    int pause = st->skip_pause ? 0 : st->duration_pause;
    int return_start = st->duration_out + pause;
    if (elapsed >= (ULONGLONG)(return_start + st->duration_in)) {
        for (int i = 0; i < st->particle_count; ++i) {
            auto& p = st->particles[i]; p.x=p.tx; p.y=p.ty; p.vx=p.vy=0;
        }
        trans_cancel(st); return 0;
    }
    st->phase = elapsed < (ULONGLONG)st->duration_out ? TRANS_EXPLODE :
        elapsed < (ULONGLONG)return_start ? TRANS_PAUSE : TRANS_ASSEMBLE;
    for (int i = 0; i < st->particle_count; ++i) {
        auto& p = st->particles[i];
        if (st->phase == TRANS_EXPLODE) {
            float t = (float)elapsed / st->duration_out;
            sample_path(p.sx,p.ox,p.svx,t,st->duration_out*.001f,p.x,p.vx);
            sample_path(p.sy,p.oy,p.svy,t,st->duration_out*.001f,p.y,p.vy);
        } else if (st->phase == TRANS_PAUSE) {
            p.x=p.ox; p.y=p.oy; p.vx=p.vy=0;
        } else {
            p.glyph=p.target_glyph; p.width=p.target_width;
            p.fg=p.target_fg; p.bg=p.target_bg; p.opacity=p.target_opacity;
            float delay = det_rand(i + 19) * 0.14f;
            float t = (float)(elapsed-return_start) / st->duration_in;
            t = (std::max)(0.0f, (std::min)(1.0f, (t-delay)/(1-delay)));
            float seconds = st->duration_in*.001f*(1-delay);
            sample_path(p.ox,p.tx,0,t,seconds,p.x,p.vx);
            sample_path(p.oy,p.ty,0,t,seconds,p.y,p.vy);
        }
    }
    return 1;
}
void trans_render_overlay(TransitionState* st) {
    if (!st || !st->active) return;
    if (!d2d_raw_available()) { trans_cancel(st); return; }
    d2d_begin_raw();
    d2d_draw_chrome(st->content_y0, st->content_y1);
    float top=(float)st->content_y0*g_cell_h, bottom=(float)st->content_y1*g_cell_h;
    d2d_clip_content(top, bottom, 1);
    d2d_fill_rect(0,top,(float)st->cols*g_cell_w,bottom-top,0x0C0C0C);
    for (int i=0;i<st->particle_count;++i) {
        auto& p=st->particles[i];
        if (!p.alive || p.opacity <= .01f || p.x+p.width*g_cell_w<0 ||
            p.x>st->cols*g_cell_w || p.y+g_cell_h<top || p.y>bottom) continue;
        if (p.bg && p.bg!=0x0C0C0C) d2d_fill_rect(p.x,p.y,(float)p.width*g_cell_w,(float)g_cell_h,p.bg);
        d2d_draw_glyph_at(p.x,p.y,p.glyph,p.fg);
    }
    d2d_clip_content(top,bottom,0);
    d2d_end_raw();
}

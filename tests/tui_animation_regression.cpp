#include <windows.h>
#include <assert.h>
#include <stdio.h>
static ULONGLONG test_now = 1000;
static ULONGLONG animation_clock() { return test_now; }
#define GetTickCount64 animation_clock
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#undef GetTickCount64
void win_pump_messages(void) {}
static VTCell displayed[16];
static int frames = 0;
static void begin() { ++frames; }
static void line(int y, const VTCell* cells, int cols) {
    if (y == 0) memcpy(displayed, cells, cols * sizeof(VTCell));
}
static void end() {}
static void write_color(int color, const char* text = "A") {
    vt_gotoxy(0, 0);
    vt_set_truecolor_fg(color, color, color);
    vt_printf("%s", text);
}
int main() {
    VTRenderer r = {begin, line, nullptr, nullptr, nullptr, end};
    vt_set_renderer(&r);
    vt_init(8, 3);
    write_color(0);
    vt_present();
    assert(!vt_animation_active());
    write_color(255);
    vt_present();
    assert(displayed[0].fg == 0 && vt_animation_active());
    test_now += 40;
    vt_present();
    unsigned mid = displayed[0].fg;
    assert(mid > 0 && mid < 0xffffff);
    // Reverse before completion: there must be no jump on retarget.
    write_color(0);
    vt_present();
    assert(displayed[0].fg == mid);
    test_now += 40;
    vt_present();
    assert(displayed[0].fg < mid && displayed[0].fg > 0);
    test_now += 200;
    vt_present();
    assert(displayed[0].fg == 0 && !vt_has_dirty());
    int settled = frames;
    vt_present();
    assert(frames == settled);
    // Re-rendering an identical target must not restart the easing clock.
    write_color(255);
    vt_present();
    test_now += 80;
    write_color(255);
    vt_mark_all_dirty();
    vt_present();
    test_now += 80;
    write_color(255);
    vt_present();
    assert(displayed[0].fg == 0xffffff && !vt_animation_active());
    // A continuously moving target must still advance on every frame.
    for (int i = 0; i < 8; ++i) {
        test_now += 16;
        write_color(100 + i);
        vt_present();
    }
    assert(displayed[0].fg < 0xffffff && displayed[0].fg > 0x646464);
    // CJK glyphs and continuation cells update together, with no stale text.
    write_color(128, "中文");
    vt_present();
    assert(displayed[0].glyph == L'中');
    assert(displayed[1].attr & VT_ATTR_CONT);
    vt_set_animation_enabled(0);
    vt_present();
    assert(displayed[0].fg == 0x808080 && !vt_animation_active());
    vt_set_animation_enabled(1);
    write_color(255);
    vt_present();
    vt_resize(6, 2);
    write_color(192);
    vt_present();
    assert(displayed[0].fg == 0xc0c0c0 && !vt_animation_active());
    puts("PASS: retarget continuity, convergence, idle, redraw, CJK, video bypass, resize");
}

#include <windows.h>
#include <assert.h>
#include <stdio.h>
static ULONGLONG test_tick = 10000;
static ULONGLONG animation_clock() { return test_tick; }
#define GetTickCount64 animation_clock
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#include "../DQN_Main_TUI_No_Terminal/transition_overlay.cpp"
#undef GetTickCount64
HWND g_hwnd = NULL;
int g_cell_w=10, g_cell_h=20;
void win_pump_messages() {}
void d2d_begin_raw() {} void d2d_end_raw() {}
void d2d_fill_rect(float,float,float,float,uint32_t) {}
void d2d_draw_glyph_at(float,float,wchar_t,uint32_t) {}
int d2d_raw_available() { return 1; }
void d2d_draw_chrome(int,int) {} void d2d_clip_content(float,float,int) {}
static void begin() {} static void end() {}
static void line(int,const VTCell*,int) {}
static void page(const char* text) {
    vt_clear_screen(); vt_gotoxy(1,3); vt_printf("%s",text);
}
int main() {
    VTRenderer r={begin,line,nullptr,nullptr,nullptr,end};
    vt_set_renderer(&r); vt_set_animation_enabled(0); vt_init(40,20);
    page("中文A"); vt_present();
    TransitionState st={}; trans_start(&st,3,17,40,20);
    assert(st.particle_count==3 && st.particles[0].width==2);
    page("A longer new page"); trans_set_target(&st);
    assert(st.particle_count>3 && st.particles[0].glyph==L'中');
    test_tick+=100; trans_update(&st);
    float x=st.particles[0].x, v=st.particles[0].vx;
    trans_start(&st,3,17,40,20);
    assert(st.fast_forward && st.particles[0].x==x && st.particles[0].svx==v);
    page("B"); trans_set_target(&st);
    assert(st.particle_count>1 && st.particles[1].target_opacity==0);
    test_tick=st.phase_start+st.duration_out;
    trans_update(&st); assert(st.phase==TRANS_PAUSE);
    for(int i=0;i<st.particle_count;i++) {
        auto& p=st.particles[i];
        assert(p.x<0 || p.x>400 || p.y<60 || p.y>340);
    }
    test_tick=st.phase_start+st.duration_out+st.duration_pause+st.duration_in;
    assert(!trans_update(&st));
    assert(st.particles[0].x==10 && st.particles[0].y==60);
    trans_destroy(&st);
    // Samples at the same time are independent of the intermediate frame count.
    page("ABC"); vt_present(); trans_start(&st,3,17,40,20); page("XYZ");trans_set_target(&st);
    ULONGLONG start=st.phase_start;
    for(int n=1;n<=10;n++){test_tick=start+n*10;trans_update(&st);}
    x=st.particles[0].x;
    trans_destroy(&st); page("ABC");vt_present();trans_start(&st,3,17,40,20);page("XYZ");trans_set_target(&st);
    test_tick=st.phase_start+100;trans_update(&st);
    assert(fabsf(x-st.particles[0].x)<.001f);
    trans_cancel(&st); trans_destroy(&st);
    page(""); vt_present();trans_start(&st,3,17,40,20);page("New from empty");trans_set_target(&st);
    assert(st.particle_count>0);
    trans_destroy(&st);
    puts("PASS: CJK, added/removed glyphs, interruption position/velocity, offscreen pause, completion, frame independence, empty source");
}

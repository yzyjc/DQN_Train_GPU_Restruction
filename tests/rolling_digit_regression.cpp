#include <windows.h>
#include <assert.h>
#include <stdio.h>
static ULONGLONG test_now=1000;
static ULONGLONG animation_clock(){return test_now;}
#define GetTickCount64 animation_clock
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#undef GetTickCount64
void win_pump_messages(void){}
static void begin(){}
static void line(int,const VTCell*,int){}
static void end(){}
static void write(const char* text){vt_gotoxy(0,0);vt_roll_printf("%s",text);vt_present();}
int main(){
    RollingDigit digit;digit.reset(L'9');digit.update(L'0',1000,true);
    assert(digit.active&&digit.layers[0].to==-1);
    float old[4];for(int i=0;i<4;i++)old[i]=digit.position(i,1060);
    digit.update(L'1',1060,true);
    for(int i=0;i<2;i++)assert(fabsf(old[i]-digit.position(i,1060))<1e-6f);
    digit.update(L'1',1420,true);assert(!digit.active&&digit.target==L'1');
    digit.update(L'0',1240,true);assert(digit.layers[0].to==1);
    digit.update(L' ',1250,true);assert(!digit.active);
    RollingDigit fast; fast.reset(L'0');
    for (uint64_t tick=2000; tick<3000; tick+=20) {
        wchar_t next=L'0'+((tick-2000)/20+1)%10;
        fast.update(next,tick,true);
        if(fast.active) {
            float a=fast.position(0,tick), b=fast.position(1,tick);
            assert(fabsf(fabsf(a-b)-1.0f)<1e-6f);
            assert(a<=0 && b>=0); // Rapid changes must not reverse the wheel.
        }
    }
    wchar_t latest=fast.target;fast.update(latest,3400,true);
    assert(!fast.active&&fast.target==latest);
    VTRenderer renderer={begin,line,nullptr,nullptr,nullptr,end};vt_set_renderer(&renderer);vt_init(20,3);
    write("19.25");assert(!g_digits[1].active);
    test_now+=200;write("20.24");assert(g_digits[0].active&&g_digits[1].active&&!g_digits[2].active);
    assert(vt_has_dirty());
    test_now+=50;write("21.23");assert(g_digits[1].target==L'1');
    test_now+=400;vt_present();assert(!vt_animation_active());
    write("22.23");assert(g_digits[1].active);
    vt_reset_digit_motion();vt_present();assert(!g_digits[1].active);
    test_now+=200;write("23.23");vt_set_animation_enabled(0);vt_present();assert(!g_digits[1].active);
    vt_set_animation_enabled(1);vt_resize(10,4);write("100");assert(!g_digits[0].active);
    test_now+=200;vt_gotoxy(0,0);vt_printf("200");vt_present();assert(!g_digits[0].active);
    puts("PASS roll direction/carry, interrupt continuity, latest target, punctuation, settle, scene/resize/disable, opt-in scope");
}

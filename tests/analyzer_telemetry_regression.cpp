// GPT6-Astra: real CPU topology and clipped telemetry rendering regression.
#define _CRT_SECURE_NO_WARNINGS
#include "../DQN_Main_TUI_No_Terminal/analyzer_cpu_topology.h"
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#include <cassert>
#include <set>
#include <limits>
void win_pump_messages(void) {}
int main() {
    using namespace analyzer_cpu;
    assert(valid_busy(0,PDH_CSTATUS_VALID_DATA)==0);
    assert(valid_busy(100,PDH_CSTATUS_NEW_DATA)==100);
    assert(valid_busy(115.3,PDH_CSTATUS_VALID_DATA)<0);
    assert(valid_busy(-1,PDH_CSTATUS_VALID_DATA)<0);
    assert(valid_busy(50,PDH_CSTATUS_INVALID_DATA)<0);
    assert(valid_busy(std::numeric_limits<double>::quiet_NaN(),0)<0);
    auto stats=summarize({0,20,40,60,100}); assert(stats.count==5 && stats.mean==44 && stats.p95==100 && stats.peak==100);
    assert(summarize({-1,50}).count==0); assert(summarize({}).count==0);
    auto cores=topology(); assert(!cores.empty());
    std::set<unsigned> ids;
    for(auto& c:cores) for(auto t:c.threads) assert(ids.insert(t.group*64+t.number).second);
    PDH_HQUERY q=nullptr; PDH_HCOUNTER c=nullptr;
    assert(PdhOpenQueryA(nullptr,0,&q)==ERROR_SUCCESS);
    assert(PdhAddEnglishCounterA(q,"\\Processor Information(*)\\% Processor Time",0,&c)==ERROR_SUCCESS);
    PdhCollectQueryData(q); Sleep(300); PdhCollectQueryData(q);
    float maximum=sample(c); assert(maximum>=0 && maximum<=100);
    auto values=snapshot(); assert(values.size()==ids.size());
    for(auto t:values) assert(t.busy>=0 && t.busy<=100 && ids.count(t.group*64+t.number));
    PdhCloseQuery(q); sample(nullptr); assert(snapshot().empty());
    vt_init(20,6);
    vt_text_clip(3,2,5,2);
    vt_gotoxy(3,-1); vt_printf("hidden");
    vt_gotoxy(3,2); vt_printf("ABCDEFGHIJKLM");
    assert(g_back[2*20+3].glyph==L'A'); assert(g_back[2*20+7].glyph==L'E');
    assert(g_back[2*20+8].glyph==L' '); assert(g_back[3*20].glyph==L' ');
    vt_gotoxy(7,3); vt_printf("中"); assert(g_back[3*20+7].glyph==L' ');
    vt_text_clip_reset(); vt_gotoxy(0,0); vt_printf("Z"); assert(g_back[0].glyph==L'Z');
    printf("PASS: %zu cores, %zu logical threads, maximum %.1f%%; clipping bounds/reset\n",cores.size(),ids.size(),maximum);
}

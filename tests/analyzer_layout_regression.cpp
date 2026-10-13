// GPT6-Astra: render the production Analyzer page at three screen sizes.
#define UNICODE
#define _UNICODE
#define wWinMain unused_tui_entry
#include "../DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp"
#undef wWinMain
#undef printf
#include <cassert>
#include <string>
static std::wstring screen() {
    std::wstring s;
    for(int y=0;y<g_console_height;++y) {
        for(int x=0;x<g_console_width;++x) s+=vt_get_back_cell(x,y)->glyph;
        s+=L'\n';
    }
    return s;
}
static bool environment_has(const std::vector<wchar_t>& block,const wchar_t* expected) {
    for(const wchar_t* p=block.data();*p;p+=wcslen(p)+1) if(!_wcsicmp(p,expected)) return true;
    return false;
}
int main() {
    vt_init(85,36); g_active_tab=3;
    for(int width:{85,160,240}) {
        g_console_width=width; g_console_height=width==240?65:36;
        vt_resize(width,g_console_height); vt_clear_screen();
        render_gpu_analyze();
        auto s=screen();
        assert(s.find(L"CAPTURE: off")!=s.npos);
        assert(s.find(L"Modules 8")!=s.npos);
        assert(s.find(L"Copy Report")!=s.npos);
        assert(((s.find(L"CPU / GPU SCHEMATIC")!=s.npos)||(s.find(L"STRIX HALO")!=s.npos))==(width==240));
        if(width==240 && s.find(L"STRIX HALO")!=s.npos) { assert(s.find(L"CORE 15")!=s.npos); assert(s.find(L"WGP 20")!=s.npos); assert(s.find(L"CU 38/39")!=s.npos); }
        if(width<240) assert(g_analyze_scroll_max>0);
        g_analyze_scroll=g_analyze_scroll_max; render_gpu_analyze();
        assert(screen().find(L"STATISTICS")!=std::wstring::npos);
        assert(screen().find(L"CAPABILITIES")==std::wstring::npos);
        assert(analyzer_wheel(10,CONTENT_Y+5,WHEEL_DELTA));
        assert(g_analyze_scroll==(g_analyze_scroll_max>2?g_analyze_scroll_max-2:0));
    }
    vt_clear_screen();g_analyze.inited=1;g_analyze.connected=1;
    g_analyze.gpu_temp=82;g_analyze.latest.gpu_power_w=45;g_analyze.latest.gpu_clock_mhz=1094;
    g_analyze.latest.gpu_memory_clock_mhz=1000;
    sec_realtime(0,0,100,nullptr);
    assert(screen().find(L"1094 MHz")!=std::wstring::npos);
    assert(screen().find(L"Memory Clock")!=std::wstring::npos);
    assert(screen().find(L"1000 MHz")!=std::wstring::npos);
    assert(screen().find(L"45.0 W")!=std::wstring::npos);
    vt_clear_screen();g_analyze.connected=0;g_analyze.gpu_temp=-1;
    sec_realtime(0,0,100,nullptr);
    assert(screen().find(L"1094 MHz")==std::wstring::npos);
    assert(screen().find(L"Memory Clock")==std::wstring::npos);
    assert(screen().find(L"45.0 W")==std::wstring::npos);
    for(auto label:{L"GFX Clock",L"GPU Temperature",L"GPU Power",L"Memory Clock",L"GPU Hotspot",L"Throttle State"}) assert(screen().find(label)==std::wstring::npos);
    handle_mouse_click(5, CONTENT_Y + 2);
    assert(g_gpu_profile_arm_batches==8 && g_gpu_profile_arm_mode==1);
    vt_clear_screen(); render_gpu_analyze();
    assert(screen().find(L"NEXT SESSION: modules, 8 batches")!=std::wstring::npos);
    handle_mouse_click(5, CONTENT_Y + 2);
    assert(g_gpu_profile_arm_batches==0);
    run_palette_command("gpu profile modules 2");
    assert(g_gpu_profile_arm_batches==2);
    vt_clear_screen(); render_gpu_analyze();
    assert(screen().find(L"NEXT SESSION: modules, 2 batches")!=std::wstring::npos);
    run_palette_command("gpu profile off");
    assert(g_gpu_profile_arm_batches==0);
    run_palette_command("gpu profile events 2");
    assert(g_gpu_profile_arm_batches==2 && g_gpu_profile_arm_mode==2);
    vt_clear_screen(); render_gpu_analyze();
    assert(screen().find(L"NEXT SESSION: events, 2 batches")!=std::wstring::npos);
    g_gpu_profile_session_batches=2;
    g_gpu_profile_session_mode=2;
    wcscpy_s(g_gpu_profile_session_dir,L"logs\\capture_test");
    std::vector<wchar_t> child_environment;
    assert(build_gpu_profile_environment(child_environment));
    assert(environment_has(child_environment,L"DQN_GPU_PROFILE_MODE=deep"));
    assert(environment_has(child_environment,L"DQN_GPU_PROFILE_BATCHES=2"));
    assert(environment_has(child_environment,L"DQN_GPU_PROFILE_DIR=logs\\capture_test"));
    const auto report=build_gpu_analyze_report();
    assert(report.find("GPU Analyze report")!=std::string::npos);
    assert(report.find("Inter-operation gaps are not a direct GPU-idle measurement")!=std::string::npos);
    g_gpu_profile_session_batches=0; g_gpu_profile_session_dir[0]=L'\0';
    printf("PASS: 85/160/240 columns, compact/wide, last metric reachable, wheel scroll\n");
}

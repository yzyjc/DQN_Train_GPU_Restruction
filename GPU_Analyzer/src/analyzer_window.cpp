// GPT6-Astra: Win32 UI thread + independent sampling worker.
#include "analyzer_window.h"
#include <windows.h>
#include <process.h>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <shellapi.h>
#include <string>
#include <vector>
#pragma comment(lib,"shell32.lib")
static std::vector<std::string> arguments;
static std::vector<char*> argv_utf8;
extern int analyzer_main(int argc, char* argv[]);
static SRWLOCK sample_lock = SRWLOCK_INIT;
static GpuTelemetrySample latest{};
static unsigned long long sample_count = 0;
static std::atomic<bool> finished{false};
static int result = 0;
static HWND label = nullptr;
void analyzer_window_sample(const GpuTelemetrySample* sample) {
    AcquireSRWLockExclusive(&sample_lock);
    latest = *sample; ++sample_count;
    ReleaseSRWLockExclusive(&sample_lock);
}
static unsigned __stdcall worker(void*) {
    result = analyzer_main((int)arguments.size(), argv_utf8.data());
    finished.store(true);
    return 0;
}
static LRESULT CALLBACK procedure(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        label = CreateWindowW(L"STATIC", L"Initializing GPU telemetry...", WS_CHILD | WS_VISIBLE,
            20,20,580,240,hwnd,nullptr,nullptr,nullptr);
        CreateWindowW(L"BUTTON",L"Hide / 隐藏",WS_CHILD|WS_VISIBLE,20,280,160,36,hwnd,(HMENU)1,nullptr,nullptr);
        CreateWindowW(L"BUTTON",L"Stop / 停止采样",WS_CHILD|WS_VISIBLE,200,280,180,36,hwnd,(HMENU)2,nullptr,nullptr);
        SetTimer(hwnd,1,500,nullptr);
        return 0;
    case WM_TIMER: {
        if (finished.load()) { DestroyWindow(hwnd); return 0; }
        GpuTelemetrySample s; unsigned long long count;
        AcquireSRWLockShared(&sample_lock); s=latest; count=sample_count; ReleaseSRWLockShared(&sample_lock);
        wchar_t temperature[32]=L"N/A",power[32]=L"N/A",clock[32]=L"N/A",memory_clock[32]=L"N/A";
        if(count && s.gpu_temp_c>=0)swprintf_s(temperature,L"%.1f C",s.gpu_temp_c);
        if(count && s.gpu_power_w>=0)swprintf_s(power,L"%.1f W",s.gpu_power_w);
        if(count && s.gpu_clock_mhz>=0)swprintf_s(clock,L"%.0f MHz",s.gpu_clock_mhz);
        if(count && s.gpu_memory_clock_mhz>=0)swprintf_s(memory_clock,L"%.0f MHz",s.gpu_memory_clock_mhz);
        wchar_t text[640];
        swprintf_s(text,L"GPU Analyzer — RUNNING\r\n\r\nSamples: %llu\r\nGPU utilization: %.1f %%\r\nGPU memory: %.0f / %.0f MB\r\nCPU: %.1f %%\r\nSystem memory: %.1f / %.1f GB\r\nADLX: temp %ls | power %ls | core %ls | memory %ls",
            count,s.gpu_util_pct,s.gpu_mem_used_mb,s.gpu_mem_total_mb,s.cpu_usage_pct,s.sys_mem_used_gb,s.sys_mem_total_gb,temperature,power,clock,memory_clock);
        SetWindowTextW(label,text);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wp)==1) ShowWindow(hwnd,SW_HIDE);
        if (LOWORD(wp)==2) { analyzer_request_stop(); SetWindowTextW(label,L"Stopping; flushing telemetry logs..."); }
        return 0;
    case WM_APP+41: // same-session TUI control; no Trainer command routing.
        if(wp==0) analyzer_request_stop();
        else ShowWindow(hwnd,wp==1?SW_SHOWNORMAL:SW_HIDE);
        return 0;
    case WM_CLOSE: ShowWindow(hwnd,SW_HIDE); return 0;
    case WM_DESTROY: KillTimer(hwnd,1); PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd,msg,wp,lp);
}
int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR,int) {
    int count=0;
    LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&count);
    if(!wide) return 5;
    for(int i=0;i<count;++i) {
        int size=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
        std::string arg(size,'\0');
        WideCharToMultiByte(CP_UTF8,0,wide[i],-1,arg.data(),size,nullptr,nullptr);
        arg.pop_back(); arguments.push_back(arg);
    }
    LocalFree(wide);
    for(auto& arg:arguments) argv_utf8.push_back(arg.data());
    argv_utf8.push_back(nullptr);
    bool report=false,hidden=false;
    for(int i=1;i<(int)arguments.size();++i) {
        if(!strcmp(argv_utf8[i],"--report")||!strcmp(argv_utf8[i],"--csv")||!strcmp(argv_utf8[i],"--compare")||!strcmp(argv_utf8[i],"--help")||!strcmp(argv_utf8[i],"-h")) report=true;
        if(!strcmp(argv_utf8[i],"--hidden")) hidden=true;
    }
    // Preserve command-line report/CSV behavior without opening a GUI.
    if(report) {
        if(GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_UNKNOWN && AttachConsole(ATTACH_PARENT_PROCESS)) {
            FILE* output=nullptr; freopen_s(&output,"CONOUT$","w",stdout);
        }
        return analyzer_main((int)arguments.size(), argv_utf8.data());
    }
    HWND existing=FindWindowW(L"DQNGpuAnalyzerWindow",nullptr);
    if(existing) { if(!hidden) PostMessageW(existing,WM_APP+41,1,0); return 2; }
    FILE* log=nullptr;
    freopen_s(&log,"gpu_analyzer.runtime.log","a",stdout);
    setvbuf(stdout,nullptr,_IOLBF,1024);
    WNDCLASSW cls{};cls.lpfnWndProc=procedure;cls.hInstance=instance;
    cls.lpszClassName=L"DQNGpuAnalyzerWindow";cls.hCursor=LoadCursor(nullptr,IDC_ARROW);
    cls.hbrBackground=(HBRUSH)(COLOR_WINDOW+1);
    if(!RegisterClassW(&cls)) return 3;
    HWND hwnd=CreateWindowW(cls.lpszClassName,L"DQN GPU Analyzer",WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX,
        CW_USEDEFAULT,CW_USEDEFAULT,660,380,nullptr,nullptr,instance,nullptr);
    if(!hwnd) return 3;
    ShowWindow(hwnd, SW_HIDE); // Consume startup show-state before explicit TUI controls.
    if(!hidden) ShowWindow(hwnd,SW_SHOW);
    HANDLE thread=(HANDLE)_beginthreadex(nullptr,0,worker,nullptr,0,nullptr);
    if(!thread){DestroyWindow(hwnd);return 4;}
    MSG msg;while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}
    analyzer_request_stop();WaitForSingleObject(thread,INFINITE);CloseHandle(thread);
    return result;
}

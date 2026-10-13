#define UNICODE
#define _UNICODE
#define wWinMain unused_tui_entry
#include "../DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp"
#undef wWinMain
#undef printf
static int failures=0;
#define CHECK(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);++failures;}}while(0)

static LRESULT CALLBACK ui_probe_parent_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    return DefWindowProcW(hwnd, msg, wp, lp);
}

/* Regression for the Probe refresh bug: a normal 3D child must not be
   destroyed by render-time orphan cleanup, while a real orphan still is. */
static void probe_viewer_refresh_lifecycle_test(void) {
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = ui_probe_parent_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"ProbeUIRefreshRegressionParent";
    CHECK(RegisterClassExW(&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS);
    HWND parent = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW,
        0, 0, 640, 480, NULL, NULL, wc.hInstance, NULL);
    CHECK(parent != NULL);
    if (!parent) return;

    g_hwnd = parent;
    CHECK(nv_init(parent, 0, 0, 320, 240) == 0);
    HWND viewer = nv_get_hwnd();
    CHECK(viewer != NULL && IsWindow(viewer));

    g_nv_lifecycle_initialized = 1;
    probe_3d_remove_orphan_window();
    CHECK(IsWindow(viewer)); /* ordinary Probe refresh keeps the live viewer */

    g_nv_lifecycle_initialized = 0;
    probe_3d_remove_orphan_window();
    CHECK(!IsWindow(viewer)); /* stale child from an interrupted lifecycle is removed */

    nv_shutdown();
    g_hwnd = NULL;
    DestroyWindow(parent);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
}

int main(){
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);CreateDirectoryA("logs",nullptr);
    FILE* f=fopen("logs/metric_loss.bin","wb");BRHeader h={BR_MAGIC,3,1,{}};fwrite(&h,sizeof(h),1,f);
    struct Loss{int32_t ep,kind;float v[3];};
    for(int ep=1;ep<=20000;++ep)for(int r=0;r<40;++r){Loss row={ep,r%2,{(float)r,.2f,.3f}};fwrite(&row,sizeof(row),1,f);}
    fclose(f);f=fopen("logs/metric_win.bin","wb");h.field_count=4;fwrite(&h,sizeof(h),1,f);
    struct Win{int32_t ep,kind;float v[4];};
    for(int ep=1000;ep<=20000;ep+=1000)for(int mode=0;mode<3;++mode){Win row={ep,0,{(float)mode,50,25,25}};fwrite(&row,sizeof(row),1,f);}
    fclose(f);f=fopen("logs/metric_eval.bin","wb");h.field_count=7;fwrite(&h,sizeof(h),1,f);
    struct Eval{int32_t ep,kind;float v[7];};
    for(int ep=1000;ep<=20000;ep+=1000)for(int mode=0;mode<3;++mode){Eval er={ep,0,{1.0f,ep<15000?2.0f:3.0f,.5f,(float)mode,50,25,25}};fwrite(&er,sizeof(er),1,f);}
    fclose(f);sample_bin_metrics();
    auto& loss=g_hist[METRIC_LOSS];auto& win=g_hist[METRIC_WIN0];auto& depth=g_hist[METRIC_DEPTH];
    CHECK(loss.n==20000 && loss.ep[0]==1 && loss.ep[loss.n-1]==20000);
    CHECK(win.n==20 && win.ep[0]==1000 && win.ep[win.n-1]==20000);
    CHECK(depth.n==20 && depth.ep[0]==1000 && depth.ep[depth.n-1]==20000);
    CHECK(depth.v[0]==2.0f && depth.v[depth.n-1]==3.0f);
    memset(g_cell_checks[0],0,sizeof(g_cell_checks[0]));
    g_cell_checks[0][METRIC_LOSS]=g_cell_checks[0][METRIC_WIN0]=1;
    float xmin=0,xmax=0;CHECK(cell_global_episode_range(0,&xmin,&xmax));CHECK(xmin==1000&&xmax==20000);
    memset(g_cell_checks[0],0,sizeof(g_cell_checks[0]));
    g_cell_checks[0][METRIC_DEPTH]=1;
    CHECK(cell_global_episode_range(0,&xmin,&xmax));CHECK(xmin==1000&&xmax==20000);
    for(int i=0;i<loss.n;++i)CHECK(loss.v[i]==19.5f && loss.ep[i]==i+1);
    sample_bin_metrics();CHECK(loss.n==20000 && loss.v[19999]==19.5f);
    // Incremental append in the same episode must continue its full accumulator.
    f=fopen("logs/metric_loss.bin","ab");Loss row={20000,0,{40,.2f,.3f}};fwrite(&row,sizeof(row),1,f);fclose(f);
    sample_bin_metrics();CHECK(loss.n==20000 && loss.v[19999]==20);
    // Incomplete tails are not read, and the next complete point remains aligned.
    f=fopen("logs/metric_loss.bin","ab");row={20001,0,{8,.2f,.3f}};fwrite(&row,5,1,f);fclose(f);
    sample_bin_metrics();CHECK(loss.n==20000);
    f=fopen("logs/metric_loss.bin","ab");fwrite((char*)&row+5,sizeof(row)-5,1,f);fclose(f);
    sample_bin_metrics();CHECK(loss.n==20001 && loss.ep[20000]==20001 && loss.v[20000]==8);
    CHECK(select_training_log_directory("logs",true));
    f=fopen("logs/pretrain/metric_loss.bin","wb");h.field_count=3;fwrite(&h,sizeof(h),1,f);
    row={1,0,{.75f,.2f,.3f}};fwrite(&row,sizeof(row),1,f);fclose(f);
    sample_bin_metrics();CHECK(loss.n==1 && loss.ep[0]==1 && loss.v[0]==.75f && win.n==0);
    CHECK(select_training_log_directory("logs",false));sample_bin_metrics();
    CHECK(loss.n==20001 && loss.ep[0]==1 && loss.v[0]==19.5f && win.n==20);
    // Changing the sampling methodology must keep one stable 29-field file.
    auto write_skill_file=[](const char* path,const int* eps,const float* rates,int n){
        FILE* sf=fopen(path,"wb");SkillMetricHeader sh={SKILL_MAGIC,SKILL_METRIC_FIELDS,1,SKILL_SUITE_VERSION,{0,0}};
        fwrite(&sh,sizeof(sh),1,sf);
        for(int i=0;i<n;++i){SkillMetricRow sr{};sr.episode=eps[i];sr.fields[0]=rates[i];sr.fields[1]=4;sr.fields[2]=1;sr.fields[3]=4;sr.fields[4]=4;fwrite(&sr,sizeof(sr),1,sf);}
        fclose(sf);
    };
    {int eps[]={1000,2000,3000};float rates[]={.4f,.8f,.9f};write_skill_file("logs/metric_skill_v4.bin",eps,rates,3);}
    g_skill_last_read=0;read_skill_metrics_cached();
    CHECK(g_skill_display_count==3);
    CHECK(g_skill_display[0].episode==1000 && g_skill_display[1].episode==2000 && g_skill_display[2].episode==3000);
    CHECK(g_skill_display[1].empty_win==.8f);
    CHECK(select_training_log_directory("logs",true));
    f=fopen("logs/pretrain/metric_loss.bin","wb");h.field_count=3;fwrite(&h,sizeof(h),1,f);
    for(int ep=1;ep<=30000;++ep)for(int r=0;r<40;++r){row={ep,r%2,{(float)r,.2f,.3f}};fwrite(&row,sizeof(row),1,f);}
    fclose(f);sample_bin_metrics();CHECK(loss.n==HIST_MAX && loss.ep[0]==5001 && loss.ep[loss.n-1]==30000);
    CHECK(loss.v[0]==19.5f && loss.v[loss.n-1]==19.5f);
    probe_viewer_refresh_lifecycle_test();
    puts("UI: history sampling and Probe 3D refresh lifecycle verified");
    printf("RECHECK2 UI: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}

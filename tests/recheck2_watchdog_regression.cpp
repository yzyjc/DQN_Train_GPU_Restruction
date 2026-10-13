#define _CRT_SECURE_NO_WARNINGS
#define UNICODE
#define _UNICODE
#include <initializer_list>
#define wWinMain unused_watchdog_entry
#include "../Watchdog_NoTerminal/Watchdog_NoTerminal.cpp"
#undef wWinMain
static int failures=0;
#define CHECK(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);++failures;}}while(0)
int main(){
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);InitializeCriticalSection(&g_wterm_lock);
    last_eval_phase=collapse_phase=2;last_known_depth=2;history_count=0;has_good_performance=0;
    for(int i=0;i<PATIENCE;++i)CHECK(!check_collapse(.1f,2,10000+i*1000));
    CHECK(has_good_performance==1);CHECK(!check_collapse(.95f,3,20000));CHECK(has_good_performance==0);
    for(int i=0;i<PATIENCE;++i)CHECK(!check_collapse(.95f,3,21000+i*1000));
    for(int i=0;i<PATIENCE;++i)CHECK(!check_collapse(.1f,3,30000+i*1000));
    CHECK(has_good_performance==1);
    for(int i=0;i<PATIENCE-1;++i)CHECK(!check_collapse(.95f,3,40000+i*1000));
    CHECK(check_collapse(.95f,3,44000)==1); // real regression at same difficulty still detected
    int old_count=history_count;
    for(float v:{NAN,INFINITY,-.1f,1.1f}){
        CHECK(!check_collapse(v,3,50000));CHECK(history_count==old_count);
        FILE* f=fopen(EVAL_FILE,"wb");fprintf(f,"50000 %g checkpoint_phase2_50000.bin 3 2\n",v);fclose(f);
        float loss=123;int ep=456,dep=789;CHECK(!read_latest_eval(&loss,&ep,&dep));CHECK(loss==123&&ep==456&&dep==789);
    }
    FILE* f=fopen(EVAL_FILE,"wb");fputs("50000 0.5 checkpoint_phase2_50000.bin 3 2\n",f);fclose(f);
    float loss=0;int ep=0,dep=0;CHECK(read_latest_eval(&loss,&ep,&dep));CHECK(loss==.5f&&ep==50000&&dep==3);
    f=fopen(TRAINING_FAULT_FILE,"wb");fputs("synthetic fault\n",f);fclose(f);
    CHECK(!start_train_process());
    remove(TRAINING_FAULT_FILE);
    DeleteCriticalSection(&g_wterm_lock);printf("RECHECK2 GUARD: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}

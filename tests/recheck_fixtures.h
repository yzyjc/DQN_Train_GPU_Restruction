#define _CRT_SECURE_NO_WARNINGS
#include "core.h"
#include "gpu_types.h"
#include <memory>
#include <vector>
#include <string>
#include "../DQN_Main_TUI_No_Terminal/dqn_report.cpp"
float g_ep_time=0, g_elapsed=0, g_remaining=0;

// Inject one upload error at the hardware boundary, without editing production.
#define gpu_nn_upload injected_upload
#include "../DQN_Core/src/neural_net.cpp"
#undef gpu_nn_upload
extern "C" int gpu_nn_upload(GpuNetHandle,const void*,size_t);
extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);
static bool fail_upload=false;
extern "C" int injected_upload(GpuNetHandle h,const void* data,size_t bytes) {
    if(fail_upload){fail_upload=false;return -333;}
    return gpu_nn_upload(h,data,bytes);
}
static int exit_requests=0;
static void record_exit(){++exit_requests;}
static int nonfinite(const float* data,size_t count) {
    int n=0; for(size_t i=0;i<count;++i)n+=!isfinite(data[i]); return n;
}
static void hard_failure_probe(DQNAgent* a) {
    if(gpu_net_activate(a)!=0) {puts("SETUP ERROR GPU activation");exit(2);}
    g_current_agent=a;g_trainer_request_exit_cb=record_exit;
    a->replay_black_idx=a->replay_black_count=BATCH_SIZE;
    for(int i=0;i<BATCH_SIZE;++i) {
        Experience& e=a->replay_black[i];e={};
        e.state[0]=e.state[1]=e.state[2]=1;
        memcpy(e.next_state,e.state,sizeof(e.state));e.next_state[3]=1;
        e.action=3;e.reward=1;e.done=1;
    }
    fail_upload=true;
    const int hard_rc=gpu_net_hard_update();
    for(int i=0;i<3;++i) replay_from_buffer(a,a->replay_black,BATCH_SIZE,&a->replay_black_idx);
    printf("HARD_FAILURE rc=%d untrusted=%d exit_requests=%d global_step=%lld (safe expectation: exit requested)\n",
        hard_rc,g_gpu_untrusted,exit_requests,a->global_step);
    gpu_net_deactivate();g_current_agent=nullptr;g_trainer_request_exit_cb=nullptr;
}
static void numeric_probe() {
    auto a=std::make_unique<DQNAgent>(); memset(a.get(),0,sizeof(*a));
    // Finite weights and a finite forward loss, but an overflowing chain-rule product.
    a->q_net.b2[0]=1e-30f;a->q_net.w3[0][0]=1e30f;a->q_net.w4[0][0]=1e30f;
    auto initial=std::make_unique<NeuralNet>(a->q_net);
    float state[26]={},loss=0,target=0;int action=0;
    int cpu_ok=train_agent_batch(a.get(),state,&action,&target,1,.001f,&loss);
    printf("NUMERIC_CPU accepted=%d finite_loss=%d nonfinite_params=%d loss=%g\n",cpu_ok,isfinite(loss),
        nonfinite((float*)&a->q_net,sizeof(NeuralNet)/sizeof(float)),loss);
    a->q_net=*initial;a->target_net=*initial;
    if(gpu_net_activate(a.get())!=0) {puts("SETUP ERROR GPU activation");exit(2);}
    g_current_agent=a.get();
    int gpu_ok=train_agent_batch(a.get(),state,&action,&target,1,.001f,&loss);
    auto params=std::make_unique<GpuNetParams>();
    /* 四句柄驻留改造后句柄改为文件静态的 g_gpu_h[2][2]，测试不再直取内部句柄；
       这里用公开的只读快照读「共享 online」（本 fixture 是非独立模式的 agent）。 */
    int downloaded = -1;
    {
        auto snap = std::make_unique<NeuralNet>();
        downloaded = gpu_net_snapshot_q(snap.get());
        if (downloaded == 0) pack_params(snap.get(), params.get());
    }
    printf("NUMERIC_GPU accepted=%d finite_loss=%d nonfinite_params=%d download_rc=%d loss=%g\n",gpu_ok,isfinite(loss),
        nonfinite((float*)params.get(),sizeof(*params)/sizeof(float)),downloaded,loss);
    gpu_net_deactivate();g_current_agent=nullptr;
}
static void reward_feedback_probe(DQNAgent* a) {
    g_wins[0]=50;g_wins[1]=50;g_wins[2]=0;
    a->reward_episode=5;
    memset(a->reward_stats,0,sizeof(a->reward_stats));
    memset(a->reward_usage,0,sizeof(a->reward_usage));
    a->reward_stats[5]=.16f;
    for(auto& value:a->reward_weights)value=1;
    GameState g={};g.turn=1;g.board[0]=g.board[1]=g.board[2]=2;
    GameState next=g;make_move(&next,3);
    float win=compute_reward(a,&g,&next,3,1,2,5,3);
    adjust_reward_weights(a,2,1);
    float block_after=a->reward_weights[5],four_after=a->reward_weights[3],terminal_stat=a->reward_stats[3];
    for(auto& value:a->reward_weights)value=1;
    a->reward_stats[3]=0;
    adjust_reward_weights(a,2,1);
    printf("REWARD_FEEDBACK white_win=%g terminal_stat=%g block_with_terminal=%g block_without_terminal=%g unused_four_weight=%g\n",
        win,terminal_stat,block_after,a->reward_weights[5],four_after);
}
static void dump_report(const char* name) {
    FILE* f=fopen(name,"wb");if(!f)exit(2);fputs(tui_report_get_text(),f);fclose(f);
}
static void report_probe() {
    FILE* f=fopen("logs/metric_loss.bin","wb");
    BRHeader h={BR_MAGIC,3,BR_VERSION,{}};fwrite(&h,sizeof(h),1,f);
    struct Loss {int32_t ep,kind;float values[3];} rows[]={ {1,0,{.1f,.2f,.3f}}, {2,0,{2.f,.2f,.3f}}, {3,0,{NAN,.2f,.3f}} };
    fwrite(rows,sizeof(rows),1,f);fclose(f);
    tui_report_exec("analyze");dump_report("analyze.txt");
    printf("ANALYZE valid_loss_2_visible=%d placeholder_nan_visible=%d\n",
        strstr(tui_report_get_text(),"2.000000")!=nullptr,
        strstr(tui_report_get_text(),"nan")!=nullptr);
    f=fopen("logs/metric_win.bin","wb");h.field_count=4;fwrite(&h,sizeof(h),1,f);
    for(int ep=1;ep<=200;++ep)for(int mode=0;mode<3;++mode){
        struct{int32_t ep,step;float values[4];}r={ep*1000,mode,{(float)mode,50,10,40}};
        fwrite(&r,sizeof(r),1,f);
    }
    fclose(f);
    EvalRecord results[MAX_EVAL_RECORDS];int n=load_eval_records("logs/eval.dat",results,MAX_EVAL_RECORDS);
    printf("EVAL_HISTORY physical_rows=600 returned=%d latest_returned_ep=%d actual_latest_ep=200000\n",n,n?results[n-1].episode:-1);
    tui_report_exec("eval 200000");dump_report("eval.txt");
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    CreateDirectoryA("logs",nullptr);CreateDirectoryA("checkpoints",nullptr);
    if(gpu_init()!=0)return 2;
    auto a=std::make_unique<DQNAgent>();memset(a.get(),0,sizeof(*a));srand(123);init_dqn_ex(a.get(),0);
    hard_failure_probe(a.get());reward_feedback_probe(a.get());free_dqn(a.get());
    numeric_probe();gpu_shutdown();report_probe();
    puts("PROBES COMPLETE: exit zero means collection succeeded, NOT that all safety properties passed.");
}

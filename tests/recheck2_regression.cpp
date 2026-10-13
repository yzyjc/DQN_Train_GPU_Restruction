#define _CRT_SECURE_NO_WARNINGS
#define UNICODE
#define _UNICODE
#define wWinMain unused_trainer_entry
#include "../DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp"
#undef wWinMain
#define gpu_nn_forward_batch injected_forward2
#define gpu_nn_download injected_download2
#include "../DQN_Core/src/neural_net.cpp"
#undef gpu_nn_forward_batch
#undef gpu_nn_download
#include "../DQN_Main_TUI_No_Terminal/dqn_report.cpp"
extern "C" int gpu_nn_forward_batch(GpuNetHandle,const float*,int,float*);
extern "C" int gpu_nn_download(GpuNetHandle,void*,size_t);
static int fail_forward2=0;
static int fail_download2=0;
extern "C" int injected_forward2(GpuNetHandle h,const float* x,int n,float* y){
    if(fail_forward2>0 && n==1){--fail_forward2;return -333;}
    return gpu_nn_forward_batch(h,x,n,y);
}
extern "C" int injected_download2(GpuNetHandle h,void* x,size_t n){
    if(fail_download2>0){--fail_download2;return -333;}
    return gpu_nn_download(h,x,n);
}
static int failures=0, notifications2=0;
#define CHECK(x) do { if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);++failures;} } while(0)
static void notify2(){++notifications2;}
static void reset_fault(){g_training_fault=0;notifications2=0;g_exit_requested=0;g_shutdown_state=0;}
static void numeric_target(DQNAgent* a){
    memset(&a->q_net,0,sizeof(a->q_net));memset(&a->target_net,0,sizeof(a->target_net));
    a->target_net.b3[0]=1e30f;
    for(int i=0;i<CELL_COUNT;++i)a->target_net.w4[i][0]=1e30f;
    a->replay_black_idx=a->replay_black_count=BATCH_SIZE;
    for(int i=0;i<BATCH_SIZE;++i){auto& e=a->replay_black[i];e={};e.action=0;e.next_state[0]=1;e.next_state[25]=1;}
    reset_fault();g_trainer_request_exit_cb=notify2;
    auto before=std::make_unique<NeuralNet>(a->q_net);
    replay_from_buffer(a,a->replay_black,BATCH_SIZE,&a->replay_black_idx);
    CHECK(g_training_fault==2 && notifications2==1 && a->global_step==0);
    CHECK(memcmp(before.get(),&a->q_net,sizeof(NeuralNet))==0);
    replay_from_buffer(a,a->replay_black,BATCH_SIZE,&a->replay_black_idx);
    CHECK(notifications2==1 && a->global_step==0);
    memset(&a->target_net,0,sizeof(a->target_net));a->replay_black_count=a->replay_black_idx=0;
    reset_fault();float state[26]={},target=NAN,loss=0;int action=0;
    CHECK(!train_agent_batch(a,state,&action,&target,1,.001f,&loss));
    CHECK(g_training_fault==2 && notifications2==1);
    reset_fault();g_trainer_request_exit_cb=nullptr;
    puts("NUMERIC: overflow and invalid target latch one fault; no update committed");
}
static uint32_t test_crc(const unsigned char* p,size_t n){
    uint32_t v=~0u;for(size_t i=0;i<n;++i){v^=p[i];for(int j=0;j<8;++j)v=(v>>1)^(0xEDB88320u & (uint32_t)-(int)(v&1));}return ~v;
}
static void metadata(DQNAgent* a){
    a->reward_weights[5]=NAN;CHECK(!save_checkpoint(a,2,12,2,"checkpoints/reject.bin"));
    a->reward_weights[5]=1;a->concentration_factor=NAN;CHECK(!save_checkpoint(a,2,12,2,"checkpoints/reject.bin"));
    a->concentration_factor=.5f;CHECK(save_checkpoint(a,2,12,2,"checkpoints/good.bin"));
    FILE* f=fopen("checkpoints/good.bin","rb");CHECK(f!=nullptr);if(!f)return;
    _fseeki64(f,0,SEEK_END);size_t size=(size_t)_ftelli64(f);rewind(f);
    std::vector<unsigned char> data(size);CHECK(fread(data.data(),1,size,f)==size);fclose(f);
    const size_t offsets[]={offsetof(CheckpointMeta,reward_weights)+5*sizeof(float),offsetof(CheckpointMeta,concentration_factor),
        offsetof(CheckpointMeta,prev_performance),offsetof(CheckpointMeta,ema_winrate),offsetof(CheckpointMeta,current_shaping_scale),
        sizeof(CheckpointMeta)+offsetof(CheckpointTrainingState,skills)+offsetof(SkillTracker,win1_baseline)};
    int ph=0,ep=0,dep=0;
    for(size_t offset:offsets){
        auto bad=data;float nan=NAN;memcpy(bad.data()+offset,&nan,4);
        uint32_t crc=test_crc(bad.data(),size-8);memcpy(bad.data()+size-4,&crc,4);
        f=fopen("checkpoints/bad.bin","wb");fwrite(bad.data(),1,size,f);fclose(f);
        a->q_net.b4[0]=7;CHECK(!load_checkpoint(a,&ph,&ep,&dep,"checkpoints/bad.bin",1));
        CHECK(a->q_net.b4[0]==7 && a->reward_weights[5]==1 && ph==0 && ep==0);
    }
    auto bad=data;double invalid=NAN;memcpy(bad.data()+offsetof(CheckpointMeta,elapsed_seconds),&invalid,8);
    uint32_t crc=test_crc(bad.data(),size-8);memcpy(bad.data()+size-4,&crc,4);
    f=fopen("checkpoints/bad.bin","wb");fwrite(bad.data(),1,size,f);fclose(f);
    CHECK(!load_checkpoint(a,&ph,&ep,&dep,"checkpoints/bad.bin",1));
    CHECK(load_checkpoint(a,&ph,&ep,&dep,"checkpoints/good.bin",1));CHECK(ph==2&&ep==12&&dep==2);
    puts("METADATA: save rejects NaN; 7 CRC-valid poisoned fields rejected atomically; valid round-trip works");
}
static void inference(DQNAgent* a){
    memset(&a->q_net,0,sizeof(a->q_net));a->q_net.b4[0]=1;a->target_net=a->q_net;
    CHECK(gpu_init()==0);CHECK(gpu_net_activate(a)==0);g_current_agent=a;
    reset_fault();g_trainer_request_exit_cb=notify2;
    auto p=std::make_unique<GpuNetParams>();pack_params(&a->q_net,p.get());p->b4[0]=0;p->b4[1]=2;
    /* 四句柄驻留改造后内部句柄是文件静态的 g_gpu_h[2][2]；本 fixture 是非独立
       agent，改 CPU 影子后用公开上传入口写设备，语义与原来一致。 */
    unpack_params(p.get(),&a->q_net);
    CHECK(gpu_net_upload_sides_pair(a)==0);
    float state[26]={},normal[25],fallback[25];forward_cpu(&a->q_net,state,normal);
    fail_forward2=true;forward_cpu(&a->q_net,state,fallback);
    CHECK(normal[0]==0&&normal[1]==2&&fallback[0]==0&&fallback[1]==2);
    CHECK(g_training_fault==0 && notifications2==0);
    fail_forward2=3;fail_download2=3;forward_cpu(&a->q_net,state,fallback);
    CHECK(g_training_fault==1 && notifications2==1 && isnan(fallback[0]));
    gpu_net_deactivate();gpu_shutdown();g_current_agent=nullptr;g_trainer_request_exit_cb=nullptr;reset_fault();
    puts("INFERENCE: fallback uses live weights; failed download stops, never uses stale shadow");
}
static void resume(DQNAgent* a){
    a->q_net.b4[0]=1;a->opponent_net.b4[0]=9;
    g_ctx={};g_ctx.total_games_played=12;g_exit_requested=1;
    train_dqn(a,100,1,1,.05f,1000,1,-1,1,12,2,nullptr,-1);
    CHECK(a->opponent_net.b4[0]==9 && g_ctx.total_games_played==12);
    reset_fault();g_exit_requested=1;
    train_dqn(a,100,1,1,.05f,1000,0,-1,1,12,2,nullptr,-1);
    CHECK(a->opponent_net.b4[0]==9 && g_ctx.total_games_played==12);
    reset_fault();g_current_agent=nullptr;
    puts("RESUME: cloned and random opponent branches both preserve restored network");
}
static void reports_and_timelines(DQNAgent* a){
    struct Row{int32_t ep,kind;float v[3];};
    Row rows[]={{1,0,{.9f,.2f,.3f}},{10000,0,{.8f,.2f,.3f}},{1,0,{.1f,.2f,.3f}},{2,1,{.2f,.2f,.3f}}};
    FILE* f=fopen("logs/metric_loss.bin","wb");BRHeader h={BR_MAGIC,3,1,{}};fwrite(&h,sizeof(h),1,f);fwrite(rows,sizeof(rows),1,f);fclose(f);
    /* GPT5-Sol: this probe verifies the retained legacy timeline splitter and
       summary wording.  The rebuilt full-history analyzer has its own focused
       regression suite and intentionally uses a different report schema. */
    tui_report_exec("analyze-legacy");CHECK(strstr(tui_report_get_text(),"valid=2 missing=0 invalid=0")!=nullptr);
    CHECK(rollback_training_logs("logs",2,999999));BRRecord read[10];int n=br_read_all("logs/metric_loss.bin",read,10);
    CHECK(n==2 && read[0].fields[0]==.1f);
    f=fopen("logs/metric_epsilon.bin","wb");h.field_count=1;fwrite(&h,sizeof(h),1,f);
    struct Eps{int32_t ep,kind;float value;} eps[]={{1,0,.5f},{2,0,.5f}};fwrite(eps,sizeof(eps),1,f);fclose(f);
    tui_report_exec("analyze-legacy");CHECK(strstr(tui_report_get_text(),"Epsilon: [0.500, 0.500]")!=nullptr);
    CHECK(select_training_log_directory("logs",true));CHECK(open_dqn_logs(a));
    g_ctx.total_games_played=0;log_train_step(a,.75f,.2f,.3f,1);
    float r[7]={},w[7]={1,1,1,1,1,1,1};log_episode_rewards(1,.75f,r,w);CHECK(g_training_fault==0);log_episode_time(1,4);
    CHECK(close_training_logs(a));
    tui_report_exec("analyze-legacy");CHECK(strstr(tui_report_get_text(),"loss=0.750000")!=nullptr);
    CHECK(select_training_log_directory("logs",false));CHECK(open_dqn_logs(a));
    g_ctx.total_games_played=2;log_train_step(a,.25f,.2f,.3f,1);
    log_episode_rewards(3,.25f,r,w);CHECK(g_training_fault==0);log_episode_time(3,2);CHECK(close_training_logs(a));
    CHECK(rollback_training_logs(training_writer_dir().c_str(),3,a->global_step));
    n=br_read_all("logs/pretrain/metric_loss.bin",read,10);CHECK(n==1 && read[0].fields[0]==.75f);
    n=br_read_all("logs/metric_loss.bin",read,10);CHECK(n==3 && read[2].episode==3);
    n=br_read_all("logs/pretrain/metric_episode_time.bin",read,10);CHECK(n==1 && read[0].episode==1);
    n=br_read_all("logs/metric_episode_time.bin",read,10);CHECK(n==1 && read[0].episode==3);
    tui_report_exec("analyze");CHECK(strstr(tui_report_get_text(),"0.750000")==nullptr);
    puts("LOGS: legacy timeline split; both epsilon colours mapped; phase directories/handles/reports isolated");
}
int main(){
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    CreateDirectoryA("logs",nullptr);CreateDirectoryA("checkpoints",nullptr);CreateDirectoryA("checkpoints_temp",nullptr);
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);
    numeric_target(a.get());metadata(a.get());inference(a.get());resume(a.get());reports_and_timelines(a.get());
    free_dqn(a.get());printf("RECHECK2 CORE: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}

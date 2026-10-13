#define _CRT_SECURE_NO_WARNINGS
#include "core.h"
#include "gpu_types.h"
#include "skill_test.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include "training_log_rollback.h"
#include "../DQN_Main_TUI_No_Terminal/episode_average.h"
#include "../DQN_Main_TUI_No_Terminal/dqn_report.cpp"
static int failures=0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)
#include "../DQN_Main_TUI_No_Terminal/bin_reader.h"
float g_ep_time=0, g_elapsed=0, g_remaining=0;

// Compile the current CPU facade unchanged except for one injected GPU failure.
// This probe runs in its own directory, never in x64/Release.
#define gpu_nn_train_step_batch audit_inject_train_batch
#include "../DQN_Core/src/neural_net.cpp"
#undef gpu_nn_train_step_batch
extern "C" int gpu_nn_train_step_batch(GpuNetHandle, const float*, const int*, const float*, int, float, float*);
extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);
static int inject_once = 0, retry_updates = 0;
extern "C" int audit_inject_train_batch(GpuNetHandle h, const float* s,
    const int* a, const float* t, int n, float lr, float* loss) {
    if (inject_once == -100) { inject_once = 0; return -100; }
    if (inject_once && n > 1) { inject_once = 0; return -777; }
    int rc = gpu_nn_train_step_batch(h, s, a, t, n, lr, loss);
    if (!rc && n == 1) ++retry_updates;
    return rc;
}
static uint32_t undo_left(uint32_t y, int n) {
    uint32_t x = y; for (int i=0;i<32;++i) x = y ^ (x << n); return x;
}
static uint32_t undo_right(uint32_t y, int n) {
    uint32_t x = y; for (int i=0;i<32;++i) x = y ^ (x >> n); return x;
}
static void rng_probe() {
    uint32_t seed = undo_left(undo_right(undo_left(0xffffffffu, 5), 17), 13);
    PRNGState p{seed}, q{seed};
    for (int bound : {1,25,32,REPLAY_CAPACITY,INT_MAX}) {
        PRNGState extreme{seed}; CHECK(prng_int(&extreme,bound) < bound);
    }
    PRNGState endpoint{seed}, zero{0};
    CHECK(prng_float(&endpoint)<1.0f && prng_int(&zero,25)<25);
    printf("RNG seed=%u output=%u prng_int(25)=%d (valid range 0..24)\n",
        seed, prng_next(&p), prng_int(&q,25));
    printf("CURRICULUM phase(ep0)=%d epsilon(ep1000)=%.6f PHASE0_END=%d\n",
        get_curriculum_phase(0), compute_epsilon(1000,1000,100000), PHASE0_END);
}
static int exit_callbacks=0;
static void fake_exit() { ++exit_callbacks; }
static void command_probe() {
    g_trainer_request_exit_cb=fake_exit;
    char response[256]; process_ipc_command("SAVE_AND_EXIT\n",response,sizeof(response));
    printf("IPC newline shutdown callback_count=%d response=%s\n",exit_callbacks,response);
    process_ipc_command("SAVE_AND_EXIT",response,sizeof(response));
    CHECK(exit_callbacks == 2);
    printf("IPC plain shutdown callback_count=%d response=%s\n",exit_callbacks,response);
    g_trainer_request_exit_cb=nullptr;
}
struct SmallRec { int32_t ep, step; float value; };
static uint32_t fixture_crc(const unsigned char* data, size_t size) {
    uint32_t crc=0xffffffffu;
    for(size_t i=0;i<size;++i) {
        crc^=data[i];
        for(int bit=0;bit<8;++bit) crc=(crc>>1)^(0xedb88320u & (uint32_t)-(int)(crc&1));
    }
    return crc^0xffffffffu;
}
static void write_checkpoint_fixture(const char* path,const std::vector<unsigned char>& body) {
    FILE* f=fopen(path,"wb"); CHECK(f!=nullptr); if(!f)return;
    uint32_t trailer[]={0x43524335,fixture_crc(body.data(),body.size())};
    CHECK(fwrite(body.data(),1,body.size(),f)==body.size());
    CHECK(fwrite(trailer,sizeof(trailer),1,f)==1); fclose(f);
}
static void reader_probe() {
    const char* path="partial_metric.bin";
    BRHeader h={BR_MAGIC,1,BR_VERSION,{0,0,0}};
    SmallRec r[]={ {10,0,1},{11,0,2},{12,0,3} };
    FILE* f=fopen(path,"wb"); fwrite(&h,sizeof(h),1,f);
    fwrite(r,sizeof(r[0]),1,f); fwrite(&r[1],4,1,f); fclose(f);
    BRFile bf={}; strcpy(bf.path,path); BRRecord out[4]={};
    int initial=br_read_tail(&bf,out,4); long long offset=bf.offset;
    f=fopen(path,"ab"); fwrite((char*)&r[1]+4,sizeof(r[1])-4,1,f);
    fwrite(&r[2],sizeof(r[2]),1,f); fclose(f);
    int got=br_read_incremental(&bf,out,4);
    CHECK(initial==1 && offset==36 && got==2 && out[0].episode==11 && out[0].fields[0]==2);
    CHECK(br_read_incremental(&bf,out,4)==0);
    printf("READER initial=%d offset=%lld (expected36), appended=2 read=%d first_ep=%d value=%g (expected11,2)\n",
        initial,offset,got,got?out[0].episode:-1,got?out[0].fields[0]:-1);
}
static void file_probe(DQNAgent* a) {
    auto before=std::make_unique<NeuralNet>(); *before=a->q_net;
    FILE* f=fopen("short_model.bin","wb"); float bad[4]={123,456,789,101};
    fwrite(bad,sizeof(bad),1,f); fclose(f);
    load_model(a,"short_model.bin");
    CHECK(memcmp(before.get(),&a->q_net,sizeof(*before))==0);
    printf("MODEL failed load changed_live_weights=%d first=%g\n",
        memcmp(before.get(),&a->q_net,sizeof(*before))!=0,a->q_net.token_embed[0][0]);
    a->q_net=*before;
    // Subsequent tests start from the original weights.
    a->pretrain_stable_passes=3; a->pretrain_consolidation_target=10000;
    a->skill_tracker.initialized=1;
    int saved=save_checkpoint(a,0,9000,2,"checkpoints/checkpoint_phase0_9000.bin");
    if(!saved) { CHECK(false); return; }
    f=fopen("checkpoints/checkpoint_phase0_9000.bin","rb+");
    _fseeki64(f,0,SEEK_END); auto bytes=_ftelli64(f);
    _chsize_s(_fileno(f),bytes-8); fclose(f);
    int phase=0,ep=0,depth=2;
    int loaded=load_checkpoint(a,&phase,&ep,&depth,"checkpoints/checkpoint_phase0_9000.bin",1);
    printf("CHECKPOINT V7 missing CRC accepted=%d\n",loaded);
    CHECK(loaded==0);
    save_checkpoint(a,0,9000,2,"checkpoints/checkpoint_phase0_9000.bin");
    f=fopen("checkpoints/checkpoint_phase0_9000.bin","rb+");
    _fseeki64(f,100,SEEK_SET); unsigned char byte=0; fread(&byte,1,1,f);
    byte^=1; _fseeki64(f,100,SEEK_SET); fwrite(&byte,1,1,f); fclose(f);
    a->pretrain_stable_passes=3; a->pretrain_consolidation_target=10000;
    a->skill_tracker.initialized=1;
    loaded=load_checkpoint(a,&phase,&ep,&depth,"checkpoints/checkpoint_phase0_9000.bin",1);
    CHECK(!loaded && a->pretrain_stable_passes==3 && a->pretrain_consolidation_target==10000 && a->skill_tracker.initialized==1);
    printf("CHECKPOINT rejected=%d but stable=%d target=%d tracker=%d (expected3,10000,1)\n",
        !loaded,a->pretrain_stable_passes,a->pretrain_consolidation_target,a->skill_tracker.initialized);
    a->opponent_pool[0].b4[0]=.345f;
    CreateDirectoryA("checkpoints/embedded.bin.pool.tmp",nullptr);
    CHECK(save_checkpoint(a,0,9000,2,"checkpoints/embedded.bin"));
    a->opponent_pool[0].b4[0]=.789f;
    CHECK(load_checkpoint(a,&phase,&ep,&depth,"checkpoints/embedded.bin",1));
    CHECK(a->opponent_pool[0].b4[0]==.345f && a->checkpoint_version==11);
    // Valid legacy V6 still loads with its own pool; neither a missing pool nor
    // a missing checksum may silently fall back to another training timeline.
    const size_t v6_size=sizeof(CheckpointMeta)+sizeof(CheckpointTrainingStateV6)+2*sizeof(Experience)*REPLAY_CAPACITY;
    std::vector<unsigned char> legacy(v6_size);
    f=fopen("checkpoints/embedded.bin","rb");
    CHECK(fread(legacy.data(),1,legacy.size(),f)==legacy.size()); fclose(f);
    int v6=6; memcpy(legacy.data(),&v6,sizeof(v6));
    write_checkpoint_fixture("checkpoints/legacy_v6.bin",legacy);
    CHECK(!load_checkpoint(a,&phase,&ep,&depth,"checkpoints/legacy_v6.bin",1));
    CHECK(save_opponent_pool(a,"checkpoints/legacy_v6.bin.pool"));
    CHECK(load_checkpoint(a,&phase,&ep,&depth,"checkpoints/legacy_v6.bin",1));
    CHECK(a->checkpoint_version==6 && a->opponent_pool[0].b4[0]==.345f);
    f=fopen("checkpoints/legacy_v6.bin","rb+"); _chsize_s(_fileno(f),v6_size); fclose(f);
    CHECK(!load_checkpoint(a,&phase,&ep,&depth,"checkpoints/legacy_v6.bin",1));
    // CRC-correct but invalid replay must fail even after metadata was parsed.
    auto meta=std::make_unique<CheckpointMeta>(); memcpy(meta.get(),legacy.data(),sizeof(*meta));
    meta->replay_black_idx=meta->replay_black_count=1;
    memcpy(legacy.data(),meta.get(),sizeof(*meta));
    Experience broken={}; broken.action=CELL_COUNT; broken.reward=1;
    memcpy(legacy.data()+sizeof(CheckpointMeta)+sizeof(CheckpointTrainingStateV6),&broken,sizeof(broken));
    write_checkpoint_fixture("checkpoints/legacy_v6.bin",legacy);
    auto live=std::make_unique<DQNAgent>(*a);
    const float live_lr=g_learning_rate; const time_t live_time=g_start_time;
    int live_wins[3]; memcpy(live_wins,g_wins,sizeof(live_wins));
    int p_before=phase,ep_before=ep,d_before=depth;
    CHECK(!load_checkpoint(a,&phase,&ep,&depth,"checkpoints/legacy_v6.bin",1));
    CHECK(memcmp(a,live.get(),sizeof(*a))==0 && g_learning_rate==live_lr && g_start_time==live_time);
    CHECK(memcmp(live_wins,g_wins,sizeof(live_wins))==0 && phase==p_before && ep==ep_before && depth==d_before);
    // Even with a valid CRC and no active replay rows, a partially filled ring
    // cannot advertise a write cursor outside its contiguous [0,count) layout.
    meta->replay_black_idx=1; meta->replay_black_count=0;
    memcpy(legacy.data(),meta.get(),sizeof(*meta));
    write_checkpoint_fixture("checkpoints/legacy_v6.bin",legacy);
    CHECK(!load_checkpoint(a,&phase,&ep,&depth,"checkpoints/legacy_v6.bin",1));
    CHECK(memcmp(a,live.get(),sizeof(*a))==0 && phase==p_before && ep==ep_before && depth==d_before);
    printf("CHECKPOINT V6 pool/CRC/invalid-replay transaction checks passed\n");
    CHECK(!save_opponent_pool(a,"checkpoints/embedded.bin.pool"));
    CreateDirectoryA("checkpoints/cannot_save.bin.tmp",nullptr);
    CHECK(!save_checkpoint(a,0,9000,2,"checkpoints/cannot_save.bin"));
    CHECK(GetFileAttributesA("checkpoints/cannot_save.bin")==INVALID_FILE_ATTRIBUTES);
    f=fopen("recovery_override.txt","w");
    fputs("version=1\nlearning_rate=0.0002\n",f); fclose(f);
    CHECK(apply_recovery_override("recovery_override.txt")==1 && g_learning_rate==.0002f);
    CHECK(apply_recovery_override("recovery_override.txt")==0);
}
static void gpu_fallback_probe(DQNAgent* a) {
    if(gpu_init()!=0 || gpu_net_activate(a)!=0) { CHECK(false); return; }
    g_current_agent=a; g_learning_rate=0.001f;
    a->replay_black_idx=a->replay_black_count=BATCH_SIZE;
    for(int i=0;i<BATCH_SIZE;++i) {
        Experience& e=a->replay_black[i]; memset(&e,0,sizeof(e));
        e.state[0]=e.state[1]=e.state[2]=1;
        memcpy(e.next_state,e.state,sizeof(e.state)); e.next_state[3]=1;
        e.action=3; e.reward=1; e.done=1;
    }
    auto before=std::make_unique<GpuNetParams>(); pack_params(&a->q_net,before.get());
    inject_once=1; retry_updates=0;
    float loss=replay_from_buffer(a,a->replay_black,BATCH_SIZE,&a->replay_black_idx);
    gpu_net_download_agent(a);
    auto after=std::make_unique<GpuNetParams>(); pack_params(&a->q_net,after.get());
    float difference=0; float* x=(float*)before.get(); float* y=(float*)after.get();
    for(size_t i=0;i<sizeof(*before)/sizeof(float);++i) difference=fmaxf(difference,fabsf(x[i]-y[i]));
    printf("FALLBACK successful GPU single retries=%d loss=%g final_parameter_change=%g global_step=%lld\n",
        retry_updates,loss,difference,a->global_step);
    CHECK(retry_updates==0 && difference>0 && isfinite(loss) && a->global_step==32);
    auto frozen=std::make_unique<NeuralNet>();
    auto live_before=std::make_unique<NeuralNet>(a->q_net);
    CHECK(gpu_net_snapshot_q(frozen.get())==0);
    CHECK(memcmp(live_before.get(),&a->q_net,sizeof(*live_before))==0);
    float inputs[2][26]={}; int actions[2]={0,0}; float targets[2]={1,NAN}, invalid_loss=0;
    CHECK(gpu_net_train_batch(&inputs[0][0],actions,targets,2,.001f,&invalid_loss)!=0);
    auto unchanged=std::make_unique<NeuralNet>();
    CHECK(gpu_net_snapshot_q(unchanged.get())==0);
    CHECK(memcmp(frozen.get(),unchanged.get(),sizeof(*frozen))==0);
    targets[1]=1;
    g_trainer_request_exit_cb=fake_exit;
    const int exits_at_failure=exit_callbacks;
    inject_once=-100;
    CHECK(gpu_net_train_batch(&inputs[0][0],actions,targets,2,.001f,&invalid_loss)==-100);
    CHECK(exit_callbacks == exits_at_failure + 1);
    CHECK(gpu_net_train_batch(&inputs[0][0],actions,targets,2,.001f,&invalid_loss)==-100);
    CHECK(gpu_net_download_agent(a)!=0 && gpu_net_snapshot_q(unchanged.get())!=0);
    CHECK(!save_model(a,"untrusted_model.bin"));
    CHECK(!save_checkpoint(a,0,32,2,"checkpoints/untrusted_checkpoint.bin"));
    CHECK(GetFileAttributesA("untrusted_model.bin")==INVALID_FILE_ATTRIBUTES);
    CHECK(GetFileAttributesA("checkpoints/untrusted_checkpoint.bin")==INVALID_FILE_ATTRIBUTES);
    const int exits_before=exit_callbacks;
    g_trainer_request_exit_cb=fake_exit;
    CHECK(!train_agent_batch(a,&inputs[0][0],actions,targets,2,.001f,&invalid_loss));
    CHECK(exit_callbacks==exits_before); // fault was already notified at its source, exactly once
    g_trainer_request_exit_cb=nullptr;
    gpu_net_deactivate(); g_current_agent=nullptr; gpu_shutdown();
}
static float huber_loss(NeuralNet* net,const float* state) {
    float q[25]; forward_cpu_only(net,state,q); float e=q[0];
    return fabsf(e)<=.5f?.5f*e*e:.5f*(fabsf(e)-.25f);
}
static void gradient_probe() {
    auto net=std::make_unique<NeuralNet>(); memset(net.get(),0,sizeof(*net));
    net->token_embed[1][0]=1; net->token_embed[2][0]=2;
    net->attn_wq[0][0]=net->attn_wk[0][0]=10;
    net->attn_wv[0][0]=net->attn_wo[0][0]=1;
    net->pool_w[0][0]=net->w1[0][0]=net->w2[0][0]=net->w3[0][0]=net->w4[0][0]=1;
    float state[26]={}; state[0]=state[1]=state[2]=1;
    net->attn_bq[0]=.01f; float plus=huber_loss(net.get(),state);
    net->attn_bq[0]=-.01f; float minus=huber_loss(net.get(),state);
    net->attn_bq[0]=0; g_learning_rate=.001f;
    float expected_loss=huber_loss(net.get(),state);
    auto original=std::make_unique<GpuNetParams>(); pack_params(net.get(),original.get());
    float cpu_loss=train_step(net.get(),state,0,0);
    CHECK(fabsf(cpu_loss-expected_loss)<1e-5f && fabsf(net->attn_bq[0]/g_learning_rate)<1e-6f);
    printf("CLAMP numeric_gradient=%g CPU_backprop_gradient=%g inference_loss=%g CPU_train_loss=%g\n",
        (plus-minus)/.02f, -net->attn_bq[0]/g_learning_rate,expected_loss,cpu_loss);
    if(gpu_init()==0) {
        GpuNetHandle h={};
        if(gpu_nn_create(&h)==0) {
            gpu_nn_upload(h,original.get(),sizeof(*original));
            int action=0; float target=0,gpu_loss=0;
            int rc=gpu_nn_train_step_batch(h,state,&action,&target,1,g_learning_rate,&gpu_loss);
            gpu_nn_download(h,original.get(),sizeof(*original));
            printf("CLAMP GPU rc=%d train_loss=%g GPU_backprop_gradient=%g (correct gradient0)\n",
                rc,gpu_loss,-original->attn_bq[0]/g_learning_rate);
            gpu_nn_destroy(&h);
            CHECK(rc==0 && fabsf(gpu_loss-expected_loss)<1e-5f && fabsf(original->attn_bq[0]/g_learning_rate)<1e-6f);
        }
        gpu_shutdown();
    }
}
static void skill_probe() {
    auto zero=std::make_unique<DQNAgent>(); memset(zero.get(),0,sizeof(*zero));
    SkillTestResult result; run_skill_tests(zero.get(),0,&result,0);
    SkillTracker tracker; skill_tracker_init(&tracker);
    int alert=skill_tracker_update(&tracker,&result);
    CHECK(tracker.win1_state != SKILL_MASTERED);
    tracker.win1_state=SKILL_MASTERED; tracker.win1_baseline=.08f;
    skill_tracker_update(&tracker,&result);
    CHECK(tracker.win1_state!=SKILL_MASTERED);
    SkillTracker unchanged=tracker;
    SkillTestResult invalid=result; invalid.win1_fixed=NAN;
    CHECK(skill_tracker_update(&tracker,&invalid)==1 && memcmp(&tracker,&unchanged,sizeof(tracker))==0);
    printf("SKILL zero network empty_steps=%.1f win1=%.4f tracker_accepts=%d win1_state=%d\n",
        result.empty_avg_steps,result.win1_fixed,!alert,tracker.win1_state);
    GameState puzzles[512]; int answers[512][25],nanswers[512],count=0;
    generate_block1_fixed(puzzles,answers,nanswers,&count,512);
    int unreachable=0;
    for(int i=0;i<count;++i) {
        int black=0,white=0; for(int j=0;j<25;++j) {black+=puzzles[i].board[j]==1;white+=puzzles[i].board[j]==2;}
        if ((puzzles[i].turn==0 && black!=white)||(puzzles[i].turn==1 && black!=white+1)) ++unreachable;
    }
    printf("SKILL block puzzles unreachable by alternating play=%d/%d\n",unreachable,count);
}
static void reward_probe(DQNAgent* agent) {
    g_hot_config.block_win_scale=g_hot_config.advanced_reward_scale=1;
    agent->current_shaping_scale=1;
    GameState g={}; g.board[1]=g.board[2]=g.board[3]=2;
    GameState next=g; make_move(&next,0);
    float reward=compute_reward(agent,&g,&next,0,0,2,0,0);
    CHECK(reward==0);
    printf("REWARD incomplete block bonus=%g remaining_enemy_win=%d\n",reward,find_opponent_winning_move(&next,2));
    memset(agent->reward_usage,0,sizeof(agent->reward_usage));
    memset(agent->reward_stats,0,sizeof(agent->reward_stats));
    g={}; g.board[0]=g.board[1]=g.board[2]=1; next=g; make_move(&next,3);
    int pattern=is_threat_four(&g,0,3,1);
    reward=compute_reward(agent,&g,&next,3,0,3,0,0);
    CHECK(agent->reward_stats[3]==reward && agent->reward_usage[3]==1);
    printf("REWARD four pattern=%d terminal_reward=%g R_Four=%g uses=%d\n",pattern,reward,agent->reward_stats[3],agent->reward_usage[3]);
}
static void params_probe() {
    FILE* f=fopen("params.txt","w"); fputs("learning_rate = nan\n",f); fclose(f);
    load_params();
    CHECK(isfinite(g_learning_rate));
    printf("PARAMS accepts_NaN=%d startup_guard_detects=%d\n", !isfinite(g_learning_rate),
        g_learning_rate<=0 || g_learning_rate>1);
    g_learning_rate=.0005f;
}
static std::string read_file(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f) return {};
    std::string s; char buf[4096]; size_t n;
    while((n=fread(buf,1,sizeof(buf),f))>0) s.append(buf,n);
    fclose(f); return s;
}
static void write_text(const char* path,const char* value) {
    FILE* f=fopen(path,"wb"); CHECK(f!=nullptr);
    if(f){fputs(value,f);fclose(f);}
}
static void rollback_and_report_probe() {
    CreateDirectoryA("rollback",nullptr);
    write_text("rollback/game.dat","0 kept\n1 future\n");
    write_text("rollback/train.dat","0 .1 .2 .3 .4 0\n32 .1 .2 .3 .4 0\n64 .1 .2 .3 .4 0\n");
    write_text("rollback/eval.dat","1 kept\n2 future\n");
    write_text("rollback/reward_episode.dat","1 kept\n2 future\n");
    SkillMetricRecord skill={1}; write_skill_metric("rollback/metric_skill_v4.bin",&skill);
    skill.episode=2; write_skill_metric("rollback/metric_skill_v4.bin",&skill);
    CHECK(rollback_training_logs("rollback",1,64));
    CHECK(read_file("rollback/game.dat")=="0 kept\n");
    CHECK(read_file("rollback/train.dat").find("64 ")==std::string::npos);
    CHECK(read_file("rollback/eval.dat")=="1 kept\n");
    CHECK(read_file("rollback/reward_episode.dat")=="1 kept\n");
    CHECK(read_file("rollback/metric_skill_v4.bin").size()==
        sizeof(SkillMetricHeader) + sizeof(SkillMetricRow));
    write_text("rollback/game.dat","0 kept\n1 future\n");
    write_text("rollback/metric_loss.bin","bad");
    CHECK(!rollback_training_logs("rollback",1,64));
    CHECK(read_file("rollback/game.dat")=="0 kept\n1 future\n");
    FILE* f=fopen("logs/metric_loss.bin","wb");
    MetricLogHeader loss_header=metric_log_header(3);
    fwrite(&loss_header,sizeof(loss_header),1,f);
    for(int i=0;i<10000;++i) {
        MetricLogPrefix prefix={i*32,0};
        float fields[3]={.1f,.2f,.3f};
        fwrite(&prefix,sizeof(prefix),1,f);
        fwrite(fields,sizeof(fields),1,f);
    }
    fclose(f);
    /* Reports intentionally inspect only the bounded recent tail of a long
       log, so query a point that is still inside that tail. */
    tui_report_exec("trainstat 319936");
    CHECK(strstr(tui_report_get_text(),"step 319936:")!=nullptr);
    CHECK(strstr(tui_report_get_text(),"Scanned 8192")!=nullptr);
    EpisodeAccumulator a;
    for(int i=0;i<10;++i) CHECK(a.add(1,1));
    CHECK(a.add(1,10) && fabsf(a.mean()-20.f/11)<1e-6f);
    CHECK(!a.add(1,NAN));
    CHECK(a.add(2,7) && a.mean()==7);
    CHECK(!a.add(1,99) && a.mean()==7);
    a={}; CHECK(a.add(1,9) && a.mean()==9);
}
static void ipc_transport_probe(DQNAgent* a) {
    g_current_agent=a;
    g_trainer_request_exit_cb=fake_exit;
    start_ipc_server();
    HANDLE clients[3]={INVALID_HANDLE_VALUE,INVALID_HANDLE_VALUE,INVALID_HANDLE_VALUE};
    for(auto& client:clients) {
        for(int i=0;i<200 && client==INVALID_HANDLE_VALUE;++i) {
            client=CreateFileA("\\\\.\\pipe\\dqn_control_pipe",GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_EXISTING,0,nullptr);
            if(client==INVALID_HANDLE_VALUE) Sleep(10);
        }
        CHECK(client!=INVALID_HANDLE_VALUE);
        if(client==INVALID_HANDLE_VALUE) continue;
        DWORD mode=PIPE_READMODE_MESSAGE; SetNamedPipeHandleState(client,&mode,nullptr,nullptr);
        DWORD sent=0;
        const char* command="SAVE_AND_EXIT\r\n";
        CHECK(WriteFile(client,command,(DWORD)strlen(command),&sent,nullptr));
        bool acknowledged=false;
        for(int i=0;i<200 && !acknowledged;++i) {
            DWORD available=0;
            if(!PeekNamedPipe(client,nullptr,0,nullptr,&available,nullptr))break;
            if(!available){Sleep(10);continue;}
            char msg[2048]={};DWORD bytes=0;
            if(ReadFile(client,msg,sizeof(msg)-1,&bytes,nullptr))
                acknowledged=strstr(msg,"graceful exit requested")!=nullptr;
        }
        CHECK(acknowledged);
    }
    stop_ipc_server();
    for(auto client:clients)if(client!=INVALID_HANDLE_VALUE)CloseHandle(client);
    g_trainer_request_exit_cb=nullptr;
    char response[256];
    process_ipc_command("SAVE_SNAPSHOT",response,sizeof(response));
    CHECK(strstr(response,"queued")!=nullptr);
    service_snapshot_requests(a);
    CHECK(GetFileAttributesA("current_snapshot.bin")!=INVALID_FILE_ATTRIBUTES);
    g_current_agent=nullptr;
}
int main() {
    CreateDirectoryA("logs",nullptr); CreateDirectoryA("checkpoints",nullptr);
    rng_probe(); command_probe(); reader_probe(); gradient_probe(); skill_probe();
    auto agent=std::make_unique<DQNAgent>(); memset(agent.get(),0,sizeof(*agent));
    srand(123); init_dqn(agent.get()); reward_probe(agent.get()); file_probe(agent.get());
    gpu_fallback_probe(agent.get()); ipc_transport_probe(agent.get());
    free_dqn(agent.get()); params_probe(); rollback_and_report_probe();
    printf("AUDIT REGRESSION: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

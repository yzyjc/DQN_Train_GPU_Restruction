// Actual Core/Trainer functions with only the expensive full evaluation replaced
// for post-processing tests. No production entry point, IPC server or training.
#define UNICODE
#define _UNICODE
#include "core.h"
#include "skill_test.h"
#include "skill_metric_reader.h"
#include "action_policy.h"
#include <memory>
#include <vector>

static bool invalid_search = false;
static int controlled_search(GameState* game, int depth, float eps, PRNGState* rng) {
    return invalid_search ? -1 : minimax_best_move(game, depth, eps, rng);
}
#define minimax_best_move controlled_search
#include "../DQN_Core/src/dqn_agent.cpp"
#undef minimax_best_move

static int tested_depth = -1;
static EvaluationStatus evaluation_status = EVALUATION_COMPLETE;
static EvaluationStatus synthetic_evaluation(DQNAgent*, int, int depth,
    int* bw, int* bd, int* bl, int* ww, int* wd, int* wl,
    float* win, float* draw, float* lose, FILE*, EvaluationCancel = nullptr,
    void* = nullptr, EvaluationProgress = nullptr) {
    tested_depth = depth;
    if (evaluation_status != EVALUATION_COMPLETE) return evaluation_status;
    for (int m = 0; m < 3; ++m) {
        bw[m] = ww[m] = 160; bd[m] = wd[m] = 0; bl[m] = wl[m] = 40;
        win[m] = .8f; draw[m] = 0; lose[m] = .2f;
    }
    return EVALUATION_COMPLETE;
}
#define record_minimax_games_for_episode synthetic_evaluation
#define wWinMain unused_trainer_entry
#include "../DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp"
#undef wWinMain
#undef record_minimax_games_for_episode

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; printf("FAIL %d: %s\n", __LINE__, #x); } } while (0)
static void zero_net(NeuralNet* net) { memset(net, 0, sizeof(*net)); }
static int cancel_after(void* context) { return --*(int*)context <= 0; }
static int progress_calls = 0, progress_last = 0, progress_total = 0;
static void capture_progress(int completed, int total, void*) {
    ++progress_calls; progress_last = completed; progress_total = total;
}

static void test_evaluation(DQNAgent* a) {
    int bw[3]{7,7,7}, bd[3]{}, bl[3]{}, ww[3]{}, wd[3]{}, wl[3]{};
    float w[3]{.123f,.123f,.123f}, d[3]{}, l[3]{};
    a->epsilon = .43f;
    g_training_fault = 1;
    CHECK(record_minimax_games_for_episode(a,1,0,bw,bd,bl,ww,wd,wl,w,d,l,nullptr) == EVALUATION_FAULT);
    CHECK(bw[0] == 7 && w[0] == .123f && a->epsilon == .43f);
    g_training_fault = 0;
    int calls = 1;
    CHECK(record_minimax_games_for_episode(a,1,0,bw,bd,bl,ww,wd,wl,w,d,l,nullptr,cancel_after,&calls) == EVALUATION_CANCELLED);
    calls = 3; // cancel after one valid move: output must still be untouched
    CHECK(record_minimax_games_for_episode(a,1,0,bw,bd,bl,ww,wd,wl,w,d,l,nullptr,cancel_after,&calls) == EVALUATION_CANCELLED);
    CHECK(bw[0] == 7 && w[0] == .123f && a->epsilon == .43f);
    invalid_search = true;
    CHECK(record_minimax_games_for_episode(a,1,0,bw,bd,bl,ww,wd,wl,w,d,l,nullptr) == EVALUATION_FAULT);
    CHECK(g_training_fault && bw[0] == 7 && a->epsilon == .43f);
    invalid_search = false; g_training_fault = 0; g_shutdown_state = 0;
    progress_calls = progress_last = progress_total = 0;
    CHECK(record_minimax_games_for_episode(a,1,0,bw,bd,bl,ww,wd,wl,w,d,l,
        nullptr,nullptr,nullptr,capture_progress)==EVALUATION_COMPLETE);
    CHECK(progress_calls == 1200 && progress_last == 1200 && progress_total == 1200);
    for(int m=0;m<3;++m) {
        CHECK(bw[m]+bd[m]+bl[m]==200 && ww[m]+wd[m]+wl[m]==200);
        CHECK(fabsf(w[m]+d[m]+l[m]-1.0f)<1e-6f);
    }
    CHECK(a->epsilon==.43f);
    GameState g{}; make_move(&g,-1); make_move(&g,CELL_COUNT);
    CHECK(g.turn == 0 && g.board[0] == 0);
    CHECK(!is_legal_move(&g,-1) && !is_legal_move(&g,CELL_COUNT) && is_legal_move(&g,0));
    make_move(&g,0); make_move(&g,0); CHECK(g.turn == 1 && g.board[0] == 1);
    CHECK(!is_legal_move(&g,0));

    g_exit_requested = 1;
    CHECK(evaluation_cancelled(nullptr));
    g_exit_requested = 0;
    puts("PASS TR-001: fault, cancellation, invalid search, board bounds, unchanged outputs");
}

static void test_candidates_and_handoff(DQNAgent* a) {
    zero_net(&a->q_net); copy_net(&a->target_net,&a->q_net);
    g_learning_rate = .0001f; a->global_step = 600;
    CHECK(save_checkpoint(a,0,6000,2,"checkpoints/checkpoint_phase0_6000.bin"));
    CHECK(save_model(a,"bad_candidate.bin"));
    a->q_net.b4[0] = .7f; a->global_step = 12345; g_learning_rate = .0005f;
    a->pretrain_stable_passes = 4; a->pretrain_consolidation_target = 10000;
    auto before = std::make_unique<DQNAgent>(*a);
    const int wins = g_wins[0]; const time_t start = g_start_time;
    for (bool checkpoint : {false,true}) {
        float sr = 0, avg = 0;
        auto candidate = assess_pretrain_candidate(checkpoint ?
            "checkpoints/checkpoint_phase0_6000.bin" : "bad_candidate.bin", checkpoint, &sr, &avg);
        CHECK(!candidate && avg > PRETRAIN_AVG_STEPS_TARGET);
        CHECK(memcmp(a,before.get(),sizeof(*a)) == 0);
    CHECK(g_learning_rate == .0005f && g_wins[0] == wins && g_start_time == start);
    }
    auto net = std::make_unique<NeuralNet>(); zero_net(net.get());
    for (int i = 0; i < CELL_COUNT; ++i) net->b4[i] = (CELL_COUNT-i)*.1f;
    FILE* f = fopen("good_candidate.bin","wb"); fwrite(net.get(),sizeof(*net),1,f); fclose(f);
    float sr = 0, avg = 0;
    auto candidate = assess_pretrain_candidate("good_candidate.bin",false,&sr,&avg);
    CHECK(candidate && sr == 1 && avg == 4);
    CHECK(memcmp(a,before.get(),sizeof(*a)) == 0);
    if (candidate) CHECK(commit_pretrain_candidate(a,candidate.get()));
    CHECK(a->global_step == 12345 && g_learning_rate == .0005f);
    CHECK(a->train_rng.state == before->train_rng.state && a->eval_rng.state == before->eval_rng.state);
    CHECK(memcmp(&a->q_net,net.get(),sizeof(*net)) == 0);
    puts("PASS TR-003: rejected/accepted candidates are isolated; only accepted weights committed");

    float state[CELL_COUNT+1]{}, next[CELL_COUNT+1]{}; next[0]=1;
    pretrain_store(a,state,0,-.01f,next,0);
    a->first_move_buf_idx = 3; a->step_counter = 27;
    CHECK(prepare_formal_task(a));
    CHECK(a->replay_black_count == 0 && a->replay_white_count == 0 && a->first_move_buf_idx == 0 && a->step_counter == 0);
    CHECK(memcmp(&a->q_net,net.get(),sizeof(*net)) == 0 && a->global_step == 12345);
    CHECK(save_checkpoint(a,1,0,2,"checkpoints/checkpoint_phase1_0.bin"));
    pretrain_store(a,state,0,-.01f,next,0);
    int p=0,ep=0,dep=2;
    CHECK(load_checkpoint(a,&p,&ep,&dep,"checkpoints/checkpoint_phase1_0.bin",1));
    CHECK(p==1 && ep==0 && a->replay_black_count==0 && a->replay_white_count==0);
    CHECK(prepare_formal_task(a)); // idempotent restart of phase1_0
    // Legacy full ring: next=3 is the oldest entry. Keep alternating samples
    // in that chronological order, not in their physical array order.
    a->replay_black_count=REPLAY_CAPACITY; a->replay_black_idx=3;
    for(int i=0;i<REPLAY_CAPACITY;++i) {
        a->replay_black[i]={};a->replay_black[i].reward=(float)i;
        a->replay_black[i].next_state[CELL_COUNT]=(i%2)?1.0f:0.0f;
    }
    int removed=discard_solitaire_replay(a->replay_black,&a->replay_black_count,&a->replay_black_idx);
    CHECK(removed==REPLAY_CAPACITY/2 && a->replay_black_count==REPLAY_CAPACITY/2);
    CHECK(a->replay_black[0].reward==3 && a->replay_black[a->replay_black_count-1].reward==1);
    CHECK(discard_solitaire_replay(a->replay_black,&a->replay_black_count,&a->replay_black_idx)==0);
    CHECK(prepare_formal_task(a));
    puts("PASS TR-002: graduation checkpoint/restart contain weights but no solo replay");
}

static void test_policy(DQNAgent* a) {
    zero_net(&a->q_net); a->q_net.b4[3] = 5e-7f;
    GameState g{};g.board[0]=g.board[1]=g.board[2]=1;
    int answer=3, correct=0;PRNGState live_rng,test_rng;
    prng_init(&live_rng,12345);prng_init(&test_rng,12345);
    for(int i=0;i<128;++i) {
        int move=select_action(a,&g,0,0,&live_rng,1);
        auto result=test_puzzle(a,&g,&answer,1,0,&test_rng);
        CHECK(move==result.chosen_action);correct+=result.is_correct;
    }
    CHECK(correct>0 && correct<128);
    zero_net(&a->q_net);
    SkillTestResult one,two;auto train_rng=a->train_rng;auto eval_rng=a->eval_rng;
    run_skill_tests(a,1000,&one,0);run_skill_tests(a,1000,&two,0);
    CHECK(memcmp(&one,&two,sizeof(one))==0);
    CHECK(one.empty_games==64 && one.empty_avg_steps>5 && one.empty_shortest_rate<.99f);
    CHECK(a->train_rng.state==train_rng.state && a->eval_rng.state==eval_rng.state);
    SkillTracker tracker{};tracker.initialized=1;tracker.empty_state=SKILL_MASTERED;
    tracker.empty_baseline=1;tracker.empty_steps_baseline=4;tracker.empty_shortest_baseline=1;
    skill_tracker_update(&tracker,&one);
    CHECK(tracker.initialized==SKILL_SUITE_VERSION && tracker.empty_state!=SKILL_MASTERED);
    puts("PASS TR-004: identical live/test actions, reproducible trials, no training RNG mutation");
}

static void test_metadata(DQNAgent* a) {
    g_ctx={};g_ctx.minimax_depth=2;g_ctx.best_winrate=-1;
    a->ema_winrate=.8f;a->depth_adjust_counter=a->depth_adjust_threshold-1;
    evaluation_status=EVALUATION_CANCELLED;
    evaluate_and_save(a,1,0,100000,1000,nullptr,nullptr,nullptr,-1);
    CHECK(g_ctx.minimax_depth==2 && a->ema_winrate==.8f && g_ctx.best_winrate==-1);
    CHECK(GetFileAttributesA("logs/metric_eval.bin")==INVALID_FILE_ATTRIBUTES);
    evaluation_status=EVALUATION_COMPLETE;
    evaluate_and_save(a,1,0,100000,1000,nullptr,nullptr,nullptr,-1);
    FILE* f=fopen("logs/metric_eval.bin","rb");CHECK(f);
    struct Eval{int ep,step;float fields[7];} row{};uint32_t h[6];
    if(f){fread(h,sizeof(h),1,f);fread(&row,sizeof(row),1,f);fclose(f);}
    CHECK(tested_depth==2 && row.fields[1]==2 && g_ctx.minimax_depth==3);
    CHECK(a->ema_winrate==0 && a->prev_performance==0 && g_ctx.best_winrate==-1);
    f=fopen("latest_eval.txt","rb");CHECK(f);
    int ep=0,depth=0,phase=0;float loss=0;char path[256]{};
    if(f){CHECK(fscanf(f,"%d %f %255s %d %d",&ep,&loss,path,&depth,&phase)==5);fclose(f);}
    CHECK(depth==2);
    CHECK(load_checkpoint(a,&phase,&ep,&depth,"checkpoints/checkpoint_phase1_1.bin",1));
    CHECK(depth==3 && a->ema_winrate==0);
    puts("PASS TR-005: logs use tested depth, resume uses next depth, no cross-depth EMA");
}

static void test_skill_log() {
    SkillMetricRecord rec{1000,1,4,1,.5f,.5f,.5f,.5f};
    CHECK(write_skill_metric("skill_torn.bin",&rec));
    FILE* f=fopen("skill_torn.bin","ab");fputc(0xff,f);fclose(f);
    rec.episode=2000;CHECK(write_skill_metric("skill_torn.bin",&rec));
    SkillDisplayRecord rows[3]{};CHECK(read_skill_metrics_tail("skill_torn.bin",rows,3)==2);
    CHECK(rows[0].episode==1000 && rows[1].episode==2000);
    for(uint32_t suite:{2u,99u}) {
        char path[64];snprintf(path,sizeof(path),"skill_version_%u.bin",suite);
        SkillMetricHeader hdr{SKILL_MAGIC,7,1,suite,{0,0}};
        f=fopen(path,"wb");fwrite(&hdr,sizeof(hdr),1,f);fclose(f);
        const bool written=write_skill_metric(path,&rec)!=0;
        CHECK(suite==2 ? written : !written);
        f=fopen(path,"rb");_fseeki64(f,0,SEEK_END);
        CHECK(_ftelli64(f)==(suite==2 ?
            (int64_t)(sizeof(SkillMetricHeader)+sizeof(SkillMetricRow)) : 24));
        fclose(f);
    }
    f=fopen("short_header.bin","wb");fputc('S',f);fclose(f);
    CHECK(!write_skill_metric("short_header.bin",&rec));
    rec.empty_win=NAN;CHECK(!write_skill_metric("invalid_row.bin",&rec));
    CHECK(GetFileAttributesA("invalid_row.bin")==INVALID_FILE_ATTRIBUTES);
    rec.empty_win=1;
    f=_fsopen("skill_torn.bin","rb+",_SH_DENYWR);CHECK(f);
    CHECK(!write_skill_metric("skill_torn.bin",&rec));if(f)fclose(f);
    CHECK(read_skill_metrics_tail("skill_torn.bin",rows,3)==2 && rows[1].episode==2000);
    CHECK(CreateDirectoryA("skill_rollback",nullptr));
    rec.episode=1000;CHECK(write_skill_metric("skill_rollback/metric_skill_v4.bin",&rec));
    rec.episode=2000;CHECK(write_skill_metric("skill_rollback/metric_skill_v4.bin",&rec));
    CHECK(rollback_training_logs("skill_rollback",1000,0));
    CHECK(read_skill_metrics_tail("skill_rollback/metric_skill_v4.bin",rows,3)==1 && rows[0].episode==1000);
    puts("PASS TR-006: torn append repaired; old/future headers and invalid values refused");
}

int main() {
    setvbuf(stdout,nullptr,_IONBF,0);SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    auto agent=std::make_unique<DQNAgent>();init_dqn_ex(agent.get(),0);
    g_current_agent=agent.get();g_hot_config.memory_enabled=0;g_start_time=time(nullptr);
    test_evaluation(agent.get());test_candidates_and_handoff(agent.get());
    test_policy(agent.get());test_metadata(agent.get());test_skill_log();
    free_dqn(agent.get());g_current_agent=nullptr;
    printf("TRAIN FIX REGRESSION: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}

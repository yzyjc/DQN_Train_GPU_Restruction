// GPT6-Astra: bounded diagnostic, not a production training loop or an efficacy test.
// Run in an isolated directory with a copied checkpoint + memory pair.
#include "core.h"
#include <memory>
#include <cstdlib>
extern "C" int gpu_init();
#define NEED(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); return 3; } } while (0)

static GameState decode(const float* s) {
    GameState g = {}; g.turn = (int)s[CELL_COUNT];
    for (int j=0;j<CELL_COUNT;++j) g.board[j]=s[j]==1 ? 1 : s[j]==-1 ? 2 : 0;
    return g;
}
static void distribution(const char* name, Experience* buf, int count) {
    int terminal=0, winning_opportunity=0, chosen_win=0, invalid=0, losses=0;
    for(int i=0;i<count;++i) {
        const Experience& e=buf[i]; invalid+=!validate_experience(&e);
        terminal+=e.done; losses+=e.done && e.reward<0;
        GameState g=decode(e.state); bool opportunity=false;
        for(int a=0;a<CELL_COUNT;++a) if(!g.board[a]) {
            GameState next=g; make_move(&next,a);
            if(check_winner(&next)==g.turn+1) { opportunity=true; chosen_win+=a==e.action; }
        }
        winning_opportunity+=opportunity;
    }
    printf("REPLAY side=%s count=%d invalid=%d terminal=%d losses=%d win_opportunity=%d chosen_win=%d\n",
        name,count,invalid,terminal,losses,winning_opportunity,chosen_win);
}
static void evaluate(DQNAgent* a, const Experience* samples, int batch) {
    double margin=0, correct=0; int top=0;
    for(int i=0;i<8;++i) {
        float q[CELL_COUNT]; forward_cpu_only(&a->q_net,samples[i].state,q);
        float wrong=-1e30f;
        for(int j=0;j<CELL_COUNT;++j) if(j!=samples[i].action && samples[i].state[j]==0) wrong=fmaxf(wrong,q[j]);
        correct+=q[samples[i].action]; margin+=q[samples[i].action]-wrong;
        top+=q[samples[i].action]>wrong;
    }
    printf("FIT batch=%d correct_q=%.8f margin=%.8f top=%d/8 attempts=%llu commit=%llu reject=%llu fault=%d\n",
        batch,correct/8,margin/8,top,g_replay_attempt_count,g_replay_commit_count,g_replay_reject_count,g_training_fault);
    fflush(stdout);
}
static void layer_delta(const char* name,const float* before,const float* after,size_t count) {
    size_t changed=0;double norm=0,base=0;
    for(size_t i=0;i<count;++i){double d=(double)after[i]-before[i];changed+=d!=0;norm+=d*d;base+=(double)before[i]*before[i];}
    printf("LAYER %s changed=%zu/%zu delta_l2=%.9g relative_l2=%.9g\n",name,changed,count,sqrt(norm),sqrt(norm/fmax(base,1e-30)));
}
int main(int argc,char** argv) {
    NEED(argc>=3);
    auto a=std::make_unique<DQNAgent>(); init_dqn_ex(a.get(),0); g_current_agent=a.get();
    int phase=0,episode=0,depth=0;
    NEED(load_checkpoint(a.get(),&phase,&episode,&depth,argv[1],1));
    printf("BASE phase=%d ep=%d lr=%.9g kb=%d cache=%d\n",phase,episode,g_learning_rate,g_kb_count,g_cache_count);
    distribution("black",a->replay_black,a->replay_black_count);
    distribution("white",a->replay_white,a->replay_white_count);
    Experience samples[32]={};
    for(int i=0;i<32;++i) {
        GameState g={}; int side=(i/4)%2; g.turn=side;
        g.board[0]=g.board[1]=g.board[2]=side+1;
        g.board[5]=g.board[7]=g.board[9]=2-side;
        if(side) g.board[20]=1; // legal black-first counts for white to move
        int action=3;
        for(int r=0;r<i%4;++r) {
            GameState rot={}; rot.turn=side;
            for(int j=0;j<25;++j) rot.board[(j%5)*5+4-j/5]=g.board[j];
            g=rot; action=(action%5)*5+4-action/5;
        }
        NEED(!check_winner(&g)); GameState next=g; make_move(&next,action);
        NEED(check_winner(&next)==side+1);
        state_to_input(&g,samples[i].state); state_to_input(&next,samples[i].next_state);
        samples[i].action=action; samples[i].done=1;
        samples[i].reward=compute_reward(a.get(),&g,&next,action,side,1,episode,6+side);
        NEED(validate_experience(&samples[i]));
    }
    memcpy(a->replay_black,samples,sizeof(samples)); a->replay_black_count=32; a->replay_black_idx=32;
    prng_init(&a->train_rng,20260919); g_ctx.total_games_played=episode;
    const int batches=atoi(argv[2]); NEED(batches>0 && batches<=1000);
    NEED(gpu_init()==0); NEED(gpu_net_activate(a.get())==0); NEED(gpu_net_active());
    auto initial=std::make_unique<NeuralNet>(a->q_net);
    evaluate(a.get(),samples,0);
    for(int b=1;b<=batches;++b) {
        replay_from_buffer(a.get(),a->replay_black,32,&a->replay_black_idx);
        NEED(!g_training_fault && gpu_net_active());
        if(b%10==0 || b==batches) { NEED(gpu_net_download_agent(a.get())==0); evaluate(a.get(),samples,b); }
    }
    printf("COMPLETE batches=%d (production replay including configured anchors; no target update or game rollout)\n",batches);
#define LAYER(member) layer_delta(#member,(const float*)initial->member,(const float*)a->q_net.member,sizeof(initial->member)/sizeof(float))
    LAYER(token_embed);LAYER(pos_embed);LAYER(attn_wq);LAYER(attn_wk);LAYER(attn_wv);LAYER(attn_wo);
    LAYER(pool_w);LAYER(w1);LAYER(w2);LAYER(w3);LAYER(w4);LAYER(b4);
#undef LAYER
    gpu_net_deactivate(); g_current_agent=nullptr; free_dqn(a.get());
    return 0;
}

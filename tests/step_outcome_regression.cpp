// GPT6-Astra: formulas, actual two-sided terminal credit, replay and config contracts.
#include "terminal_outcome_credit.h"
#include "outcome_metrics.h"
#include <memory>
#include <vector>
#define main unused_gpu_reference_main
#include "../DQN_Core/src/test_gpu_train.cpp"
#undef main
float g_ep_time=0,g_elapsed=0,g_remaining=0;
volatile double g_phase_forward_us=0,g_phase_backward_us=0,g_phase_target_us=0;
volatile int g_phase_steps=0,g_side_steps[2]={};
volatile double g_side_forward_us[2]={},g_side_backward_us[2]={};
#define NEED(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(){
    NEED(g_hot_config.step_outcome.enabled==0);
    NEED(g_hot_config.step_outcome.black_fast_win_scale==BLACK_FAST_WIN_SCALE);
    StepOutcomeConfig c;c.enabled=1;NEED(step_outcome_valid(c));
    for(int steps=1;steps<=25;++steps)for(float scale:{.5f,.85f,1.f,1.15f,1.5f}) {
        float bw=scale+step_outcome_bonus(c,0,1,steps,25);
        float bl=-.5f+step_outcome_bonus(c,0,2,steps,25);
        float wl=-.5f+step_outcome_bonus(c,1,1,steps,25);
        NEED(bw>0 && scale>0 && bl<0 && wl<0);
        NEED(step_outcome_bonus(c,1,2,steps,25)==0);
    }
    NEED(!step_outcome_set(c,3,.5f) && !step_outcome_set(c,1,NAN));
    NEED(c.white_draw_length_scale==.05f);
    NEED(!step_outcome_set(c,4,.16f));NEED(isnan(step_outcome_bonus(c,0,1,1,0)));
    NEED(step_outcome_bonus(c,0,1,-1,25)==c.black_fast_win_scale);
    NEED(step_outcome_bonus(c,0,1,26,25)==0);
    StepOutcomeConfig upper=c;
    for(int i=1;i<=4;++i)NEED(step_outcome_set(upper,i,.15f));
    for(int steps=1;steps<=25;++steps){
        const float draw=step_outcome_bonus(upper,1,0,steps,25);
        NEED(draw>0 && draw<.5f);
        NEED(step_outcome_bonus(upper,0,0,steps,25)==0);
        NEED(-.5f+step_outcome_bonus(upper,1,1,steps,25)<0);
        if(steps>1){
            NEED(step_outcome_bonus(c,0,1,steps,25)<step_outcome_bonus(c,0,1,steps-1,25));
            NEED(step_outcome_bonus(c,0,2,steps,25)==0);
            NEED(step_outcome_bonus(c,1,1,steps,25)>step_outcome_bonus(c,1,1,steps-1,25));
        }
    }
    auto agent=std::make_unique<DQNAgent>();init_dqn_ex(agent.get(),0);g_current_agent=agent.get();
    g_learning_rate=.0001f;set_terminal_win_reward_scales(1,1);
    StepOutcomeScope scope(c);
    const int games[2][8]={{0,5,1,7,2,9,3,-1},{20,0,22,1,24,2,10,3}};
    std::vector<Experience> terminal_samples;
    for(int test=0;test<3;++test) {
        GameState g={};TerminalOutcomeCredit credit;
        agent->replay_black_idx=agent->replay_white_idx=0;
        agent->replay_black_count=agent->replay_white_count=0;
        int actions[25],length=0;
        if(test<2){length=test==0?7:8;memcpy(actions,games[test],length*sizeof(int));}
        else {
            // Find a deterministic legal full-board draw, with no earlier winner.
            PRNGState rng;prng_init(&rng,19);bool found=false;
            for(int attempt=0;attempt<10000 && !found;++attempt){
                GameState trial={};length=0;
                for(int j=0;j<25;++j){int moves[25],count=0;
                    for(int a=0;a<25;++a)if(!trial.board[a]){GameState n=trial;make_move(&n,a);if(!check_winner(&n))moves[count++]=a;}
                    if(!count)break;int a=moves[prng_int(&rng,count)];actions[length++]=a;make_move(&trial,a);
                }found=length==25;
            }NEED(found);
        }
        for(int i=0;i<length;++i){GameState next=g;make_move(&next,actions[i]);
            float r=compute_reward(agent.get(),&g,&next,actions[i],g.turn,1,0,i);
            credit.observe(agent.get(),g,next,actions[i],r,1,0);g=next;
        }
        NEED(!credit.pending);
        NEED(agent->replay_black_count+agent->replay_white_count==length*2);
        const int winner=check_winner(&g);NEED(winner==(test<2?test+1:0));
        NEED(log_outcome_metrics(agent.get(),g,test,0,1,c));
        for(int side=0;side<2;++side){Experience* buf=side?agent->replay_white:agent->replay_black;
            int count=side?agent->replay_white_count:agent->replay_black_count;int terminals=0;
            for(int i=0;i<count;++i){NEED(validate_experience(&buf[i]));NEED(buf[i].state[25]==side);
                if(buf[i].done){terminal_samples.push_back(buf[i]);++terminals;float base=winner==0?0:winner==side+1?1.f:-.5f;
                    if(test==2)NEED(fabsf(buf[i].reward-(side ? .05f : 0.f))<1e-6f);
                    NEED(fabsf(buf[i].reward-base-step_outcome_bonus(c,side,winner,length,25))<1e-6f);}
            }NEED(terminals==2);
        }
        if(test==0){Experience bad=agent->replay_white[agent->replay_white_count-2];
            bad.next_state[15]=1;NEED(!validate_experience(&bad));}
    }
    // Real save/load must retain terminal bridges and advertise their new semantics.
    memory_system_init();NEED(save_checkpoint(agent.get(),1,25,2,"outcome.bin"));
    auto restored=std::make_unique<DQNAgent>();init_dqn_ex(restored.get(),0);int phase,ep,depth;
    NEED(load_checkpoint(restored.get(),&phase,&ep,&depth,"outcome.bin",1));
    NEED(restored->checkpoint_version==12);
    NEED(save_checkpoint(agent.get(),1,1000,2,"checkpoints/checkpoint_phase1_1000.bin"));
    char found[260];int latest=0;NEED(find_latest_checkpoint(found,sizeof(found),&latest));NEED(latest==1000);
    for(int i=0;i<restored->replay_white_count;++i)NEED(validate_experience(&restored->replay_white[i]));
    OutcomeMetrics metrics;
    for(int i=0;i<1001;++i)metrics.add(i%3,10+i%10);
    NEED(metrics.counts[5]==1001 && metrics.rolling_counts[5]==1000);
    NEED(metrics.counts[0]==metrics.counts[3] && metrics.counts[1]==metrics.counts[2]);
    OutcomeMetrics rebuilt;NEED(rebuilt.restore("logs/metric_outcome_raw.bin",3));
    NEED(rebuilt.counts[5]==3 && rebuilt.counts[4]==1 && rebuilt.sums[5]==40);
    char response[4096];process_ipc_command("SET black_fast_win_scale 0.12",response,sizeof(response));
    NEED(strstr(response,"queued"));NEED(apply_pending_ipc_config(agent.get()));
    NEED(fabsf(g_hot_config.step_outcome.black_fast_win_scale-.12f)<1e-6f);
    publish_training_status(agent.get());build_status_json(response,sizeof(response));
    NEED(strstr(response,"\"black_fast_win_scale\":0.1200"));
    load_config();NEED(fabsf(g_hot_config.step_outcome.black_fast_win_scale-.12f)<1e-6f);
    process_ipc_command("SET white_draw_length_scale 0.08",response,sizeof(response));
    NEED(strstr(response,"queued") && apply_pending_ipc_config(agent.get()));
    load_config();NEED(g_hot_config.step_outcome.white_draw_length_scale==.08f);
    float states[32][CELL_COUNT+1],targets[32];int actions[32];
    for(int i=0;i<32;++i){const auto& e=terminal_samples[i%terminal_samples.size()];
        memcpy(states[i],e.state,sizeof(states[i]));actions[i]=e.action;targets[i]=e.reward;}
    // Terminal samples have target=reward in the unchanged production Bellman code.
    auto cpu=std::make_unique<DQNAgent>();init_dqn_ex(cpu.get(),0);
    cpu->q_net=agent->q_net;cpu->target_net=agent->target_net;
    auto before=std::make_unique<GpuNetParams[]>(3);auto want=std::make_unique<GpuNetParams[]>(3);
    auto actual=std::make_unique<GpuNetParams>();
    float cpu_loss[3]={};
    for(int batch=0;batch<3;++batch){
        pack(&cpu->q_net,&before[batch]);g_current_agent=cpu.get();
        NEED(train_agent_batch(cpu.get(),&states[0][0],actions,targets,32,.0001f,&cpu_loss[batch])==1);
        pack(&cpu->q_net,&want[batch]);
    }
    NEED(gpu_init()==0);
    g_current_agent=agent.get();NEED(gpu_net_activate(agent.get())==0);
    for(int batch=0;batch<3;++batch){
        float gpu_loss=0;
        NEED(train_agent_batch(agent.get(),&states[0][0],actions,targets,32,.0001f,&gpu_loss)==1);
        NEED(gpu_net_download_agent(agent.get())==0);
        pack(&agent->q_net,actual.get());
        double delta=0,error=0;
        for(size_t j=0;j<sizeof(GpuNetParams)/sizeof(float);++j){
            float w=((float*)&want[batch])[j],v=((float*)actual.get())[j],b=((float*)&before[batch])[j];
            NEED(isfinite(w) && isfinite(v));delta+=(double)(w-b)*(w-b);error+=(double)(w-v)*(w-v);
        }
        const double relative=sqrt(error/fmax(delta,1e-30));
        printf("OUTCOME GPU batch=%d relative_delta_error=%.9g loss_diff=%.9g\n",batch,relative,fabsf(cpu_loss[batch]-gpu_loss));
        NEED(delta>0 && relative<.005 && fabsf(cpu_loss[batch]-gpu_loss)<1e-5f);
    }
    gpu_net_deactivate();
    free_dqn(cpu.get());
    free_dqn(restored.get());free_dqn(agent.get());g_current_agent=nullptr;
    puts("PASS outcome formulas, both losses, white draw, rotation, V12 replay, rolling counts, IPC/config");return 0;
}

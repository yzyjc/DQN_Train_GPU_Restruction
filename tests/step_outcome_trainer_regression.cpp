// GPT6-Astra: bounded real Trainer integration; no production entry point/server.
#define UNICODE
#define _UNICODE
#define wWinMain unused_trainer_entry
#include "../DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp"
#undef wWinMain
#include <memory>
extern "C" int gpu_init();
#define NEED(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(){
    auto agent=std::make_unique<DQNAgent>();init_dqn_ex(agent.get(),0);
    g_current_agent=agent.get();g_learning_rate=.00016f;
    g_hot_config.step_outcome=StepOutcomeConfig{};g_hot_config.step_outcome.enabled=1;
    g_hot_config.memory_enabled=0;g_hot_config.train_repeat_per_step=1;
    agent->epsilon=1;prng_init(&agent->train_rng,20260919);
    NEED(gpu_init()==0 && gpu_net_activate(agent.get())==0);
    for(int ep=0;ep<12;++ep){
        g_ctx.total_games_played=ep;
        play_episode(agent.get(),ep,ep,1,ep%2,0,0,1,TARGET_UPDATE_PHASE1,0,1,12,&g_ctx,0);
        NEED(!g_training_fault && gpu_net_active());
    }
    NEED(gpu_net_download_agent(agent.get())==0);gpu_net_deactivate();
    int negative=0,bridge=0;
    for(int side=0;side<2;++side){Experience* buf=side?agent->replay_white:agent->replay_black;
        int count=side?agent->replay_white_count:agent->replay_black_count;
        for(int i=0;i<count;++i){const auto& e=buf[i];NEED(validate_experience(&e));
            negative+=e.done && e.reward<0;
            bridge+=e.done && e.state[CELL_COUNT]==e.next_state[CELL_COUNT];}
    }
    NEED(negative>0 && bridge==24 && g_replay_commit_count>0);
    OutcomeMetrics metrics;NEED(metrics.restore("logs/metric_outcome_raw.bin",12));NEED(metrics.counts[5]==12);
    printf("PASS real Trainer 12 games: commits=%llu rejects=%llu negative_terminals=%d bridges=%d mean_steps=%.4f\n",
        g_replay_commit_count,g_replay_reject_count,negative,bridge,metrics.sums[5]/12);
    free_dqn(agent.get());g_current_agent=nullptr;return 0;
}

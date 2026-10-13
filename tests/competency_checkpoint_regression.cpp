// Collaboration signature: GPT6-Astra. CPU runtime, real V14 writer/loader.
#include "competency_training.h"
#include <cstdio>
#include <memory>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(independent_training::indep_start_from_shared(a.get()));a->competency.initialized=1;a->competency.stage=3;a->competency.games=7000;a->competency.stage_games=500;
    a->indep->online(independent_training::Side::White).b4[0]+=.02f;
    NEED(competency_refresh_opponents(a.get(),g_hot_config.competency));
    a->competency.skills[1][2].learned=1;a->competency.skills[1][2].baseline=.95f;a->competency.skills[1][2].cooldown_until=8000;
    NEED(save_checkpoint(a.get(),1,7000,2,"checkpoints/checkpoint_phase1_7000.bin"));
    auto b=std::make_unique<DQNAgent>();init_dqn_ex(b.get(),0);g_current_agent=b.get();int p=0,e=0,d=0;
    NEED(load_checkpoint(b.get(),&p,&e,&d,"checkpoints/checkpoint_phase1_7000.bin",1));
    NEED(b->checkpoint_version==14 && b->competency.stage==3 && b->competency.skills[1][2].cooldown_until==8000);
    NEED(!memcmp(&a->competency,&b->competency,sizeof(CompetencyState)));
    NEED(!memcmp(a->competency_opponents,b->competency_opponents,sizeof(CompetencyOpponents)));
    NEED(b->indep->online(independent_training::Side::White).b4[0]==a->indep->online(independent_training::Side::White).b4[0]);
    for(int side=0;side<2;++side)for(int skill=1;skill<=2;++skill){
        GameState train[32],eval[32];int ta[32][CELL_COUNT],ea[32][CELL_COUNT],tn[32],en[32],tc=0,ec=0;
        generate_side_skill_puzzles(side,skill,1234,0,train,ta,tn,&tc,32);
        generate_side_skill_puzzles(side,skill,1234,1,eval,ea,en,&ec,32);NEED(tc>=16 && ec>=16);
        for(int i=0;i<tc;++i){int black=0,white=0;for(int piece:train[i].board){black+=piece==1;white+=piece==2;}
            NEED(train[i].turn==side && black-white==side && check_winner(&train[i])==0 && tn[i]>0);
            for(int j=0;j<ec;++j)NEED(memcmp(train[i].board,eval[j].board,sizeof(train[i].board))!=0);}
    }
    const auto frozen_generation=b->competency.frozen_generation;
    const auto old_lagged=b->competency.lagged_generation;
    b->competency.games+=100;NEED(competency_refresh_opponents(b.get(),g_hot_config.competency));
    NEED(b->competency.frozen_generation==frozen_generation+100 && b->competency.lagged_generation==old_lagged);
    b->competency.games+=400;NEED(competency_refresh_opponents(b.get(),g_hot_config.competency));
    NEED(b->competency.lagged_refresh_game==b->competency.games && b->competency.lagged_generation==frozen_generation+100);
    auto preserved=std::make_unique<NeuralNet>(b->indep->online(independent_training::Side::White));
    const auto saved_state=b->competency;
    NEED(!rename("checkpoints/checkpoint_phase1_7000.bin.mem","checkpoints/withheld.mem"));
    NEED(!load_checkpoint(b.get(),&p,&e,&d,"checkpoints/checkpoint_phase1_7000.bin",1));
    NEED(!memcmp(preserved.get(),&b->indep->online(independent_training::Side::White),sizeof(NeuralNet)));
    NEED(!memcmp(&saved_state,&b->competency,sizeof(CompetencyState)));
    NEED(!rename("checkpoints/withheld.mem","checkpoints/checkpoint_phase1_7000.bin.mem"));
    auto old=std::make_unique<NeuralNet>(b->indep->online(independent_training::Side::Black));
    auto target=std::make_unique<NeuralNet>(b->indep->target(independent_training::Side::Black));
    auto white_target=std::make_unique<NeuralNet>(b->indep->target(independent_training::Side::White));
    auto white_before=std::make_unique<NeuralNet>(b->indep->online(independent_training::Side::White));
    int rc=competency_skill_batch(b.get(),1,0,.0002f);NEED(rc==1);
    NEED(!memcmp(old.get(),&b->indep->online(independent_training::Side::Black),sizeof(NeuralNet)));
    NEED(!memcmp(target.get(),&b->indep->target(independent_training::Side::Black),sizeof(NeuralNet)));
    NEED(!memcmp(white_target.get(),&b->indep->target(independent_training::Side::White),sizeof(NeuralNet)));
    double delta=0;for(int i=0;i<CELL_COUNT;++i){double diff=b->indep->online(independent_training::Side::White).b4[i]-white_before->b4[i];delta+=diff*diff;}NEED(delta>0);
    free_dqn(a.get());free_dqn(b.get());g_current_agent=nullptr;
    puts("PASS V14 two-side frozen snapshots, cooldown exact restore, white rehearsal isolation");return 0;
}

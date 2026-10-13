// Collaboration signature: GPT6-Astra. Exercise the real bounded rehearsal controller.
#include "competency_training.h"
#include <memory>
#include <cstdio>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(independent_training::indep_start_from_shared(a.get()));
    g_hot_config.competency.rehearsal_enabled=1;g_hot_config.competency.eval_interval=1;
    g_hot_config.competency.rehearsal_batches=3;g_hot_config.competency.rehearsal_attempts=6;
    a->competency.initialized=1;a->competency.stage=5;
    auto& skill=a->competency.skills[0][1];skill.learned=1;skill.baseline=1;skill.current=1;skill.regression_streak=1;
    g_learning_rate=.0002f;const float scheduled=g_learning_rate;
    NEED(competency_after_game(a.get(),nullptr));
    printf("rehearsals=%u commits=%u lr=%.9g planned=%.9g\n",skill.rehearsal_count,skill.rehearsal_batches,a->indep->updates(independent_training::Side::Black).learning_rate,scheduled);
    NEED(skill.rehearsal_count==1 && skill.rehearsal_batches==3);
    NEED(a->indep->updates(independent_training::Side::Black).learning_rate==scheduled);
    NEED(g_learning_rate==scheduled);
    int normal=2;for(int attempt=0;attempt<20 && normal==2;++attempt)normal=competency_skill_batch(a.get(),0,1,g_learning_rate);
    printf("normal training rc=%d applied_lr=%.9g\n",normal,a->indep->updates(independent_training::Side::Black).learning_rate);
    NEED(normal==1);
    NEED(a->indep->updates(independent_training::Side::Black).learning_rate==scheduled);
    {CompetencyRehearsalLearningRate scope(a.get(),0,.2f);
        NEED(fabsf(a->indep->updates(independent_training::Side::Black).learning_rate-scheduled*.2f)<1e-9f);
        a->indep->set_learning_rate(independent_training::Side::Black,scope.reduced*.25f);}
    NEED(fabsf(a->indep->updates(independent_training::Side::Black).learning_rate-scheduled*.25f)<1e-9f);
    free_dqn(a.get());g_current_agent=nullptr;puts("PASS real multi-batch rehearsal uses fixed reduced LR; normal planned LR resumes");return 0;
}

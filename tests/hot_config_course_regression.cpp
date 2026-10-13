// Collaboration signature: GPT6-Astra.
#include "independent_networks.h"
#include <memory>
#include <cstdio>
float g_ep_time=0,g_elapsed=0,g_remaining=0;
volatile double g_phase_forward_us=0,g_phase_backward_us=0,g_phase_target_us=0;
volatile int g_phase_steps=0,g_side_steps[2]={};
volatile double g_side_forward_us[2]={},g_side_backward_us[2]={};
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    FILE* f=fopen("train_config.txt","w");NEED(f);fputs("train_repeat=3\n  curriculum_enabled = 1\nrehearsal_batches=17\n",f);fclose(f);
    load_hot_config();NEED(g_hot_config.train_repeat_per_step==3 && g_hot_config.competency.rehearsal_batches==17);NEED(!g_hot_config.competency.enabled);
    NEED(independent_training::indep_start_from_shared(a.get()));load_hot_config();NEED(g_hot_config.competency.enabled);
    f=fopen("train_config.txt","w");NEED(f);fputs("train_repeat = 4\n\tcurriculum_enabled\t=0\nrehearsal_batches = 0\n",f);fclose(f);
    load_hot_config();NEED(g_hot_config.train_repeat_per_step==4 && !g_hot_config.competency.enabled);NEED(g_hot_config.competency.rehearsal_batches==17);
    free_dqn(a.get());g_current_agent=nullptr;puts("PASS hot config compact/spaced syntax, bounds, shared-mode enable guard");return 0;
}

// Collaboration signature: GPT6-Astra. Real resident GPU handles, finite teacher batches only.
#include "competency_training.h"
#define main unused_gpu_reference_main
#include "../DQN_Core/src/test_gpu_train.cpp"
#undef main
#include <memory>
extern "C" void gpu_shutdown(void);
float g_ep_time=0,g_elapsed=0,g_remaining=0;
volatile double g_phase_forward_us=0,g_phase_backward_us=0,g_phase_target_us=0;
volatile int g_phase_steps=0,g_side_steps[2]={};
volatile double g_side_forward_us[2]={},g_side_backward_us[2]={};
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(independent_training::indep_start_from_shared(a.get()));NEED(gpu_init()==0);NEED(gpu_net_activate(a.get())==0);
    a->competency.initialized=1;NEED(competency_refresh_opponents(a.get(),g_hot_config.competency));
    auto before_b=std::make_unique<NeuralNet>(),before_w=std::make_unique<NeuralNet>();
    auto after_b=std::make_unique<NeuralNet>(),after_w=std::make_unique<NeuralNet>();
    auto before=std::make_unique<GpuNetParams[]>(2),after=std::make_unique<GpuNetParams[]>(2);
    for(int side=0;side<2;++side)for(int batch=0;batch<2;++batch){
        NEED(gpu_net_snapshot_online_pair(a.get(),before_b.get(),before_w.get())==0);pack(before_b.get(),&before[0]);pack(before_w.get(),&before[1]);
        auto target=std::make_unique<NeuralNet>(),target_after=std::make_unique<NeuralNet>();
        NEED(gpu_net_snapshot_side_pair(a.get(),side,after_b.get(),target.get())==0);
        NEED(competency_skill_batch(a.get(),side,0,.0002f)==1);
        NEED(gpu_net_snapshot_online_pair(a.get(),after_b.get(),after_w.get())==0);pack(after_b.get(),&after[0]);pack(after_w.get(),&after[1]);
        NEED(!memcmp(&before[1-side],&after[1-side],sizeof(GpuNetParams)));
        NEED(gpu_net_snapshot_side_pair(a.get(),side,before_b.get(),target_after.get())==0);
        auto target_p=std::make_unique<GpuNetParams>(),target_q=std::make_unique<GpuNetParams>();pack(target.get(),target_p.get());pack(target_after.get(),target_q.get());
        NEED(!memcmp(target_p.get(),target_q.get(),sizeof(GpuNetParams)));
        double delta=0;for(size_t i=0;i<sizeof(GpuNetParams)/sizeof(float);++i){float x=((float*)&after[side])[i],b=((float*)&before[side])[i];NEED(isfinite(x));delta+=(double)(x-b)*(x-b);}NEED(delta>0);
        printf("GPU course side=%d batch=%d parameter_delta_L2=%.9g other side byte-identical\n",side,batch,sqrt(delta));
    }
    gpu_net_deactivate();gpu_shutdown();free_dqn(a.get());g_current_agent=nullptr;
    puts("PASS GPU course teacher commits on both colors, opposite resident network and target preserved");return 0;
}

// GPT6-Astra: production shared->four-resident transition, logical binding,
// both-side device inference, V13 restore and idempotent activation.
#include "independent_networks.h"
#include <memory>
#include <cstdio>
#include <cstring>
extern "C" int gpu_init();
using namespace independent_training;
#define NEED(x) do {if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(indep_ensure(a.get()) && gpu_init()==0);
    a->q_net.b4[0]=.123f; a->target_net.b4[0]=.321f;
    NEED(gpu_net_activate(a.get())==0); // phase0/shared legacy checkpoint
    a->q_net.b4[0]=9.f; // stale shadow: promotion must fetch device seed
    NEED(indep_start_from_shared(a.get()));
    auto q=std::make_unique<NeuralNet>();auto t=std::make_unique<NeuralNet>();
    for(int side=0;side<2;++side){
        NEED(gpu_net_snapshot_side_pair(a.get(),side,q.get(),t.get())==0);
        NEED(q->b4[0]==.123f && t->b4[0]==.321f);
    }
    a->indep->online(Side::Black).b4[0]=.456f;
    NEED(gpu_net_upload_side(a.get(),0)==0);
    NEED(indep_start_from_shared(a.get())); // never reseed a restored/live model
    NEED(gpu_net_snapshot_side_pair(a.get(),0,q.get(),t.get())==0 && q->b4[0]==.456f);
    a->indep->online(Side::White).b4[0]=9.f; // binding must not download or upload
    NEED(gpu_net_bind_side(a.get(),1)==0);
    NEED(a->indep->online(Side::White).b4[0]==9.f);
    NEED(gpu_net_snapshot_side_pair(a.get(),1,q.get(),t.get())==0 && q->b4[0]==.123f);
    NEED(gpu_net_download_sides_pair(a.get())==0);
    NEED(save_checkpoint(a.get(),1,3,2,"checkpoints/dual.bin"));
    gpu_net_deactivate();free_dqn(a.get());g_current_agent=nullptr;
    auto restored=std::make_unique<DQNAgent>();init_dqn_ex(restored.get(),0);
    NEED(indep_ensure(restored.get()));int phase,episode,depth;
    NEED(load_checkpoint(restored.get(),&phase,&episode,&depth,"checkpoints/dual.bin",1));
    g_current_agent=restored.get();NEED(gpu_net_activate(restored.get())==0);
    for(int side=0;side<2;++side){
        NEED(gpu_net_snapshot_side_pair(restored.get(),side,q.get(),t.get())==0);
        NEED(q->b4[0]==(side==0?.456f:.123f));
    }
    NEED(!g_training_fault);gpu_net_deactivate();free_dqn(restored.get());g_current_agent=nullptr;
    puts("PASS shared GPU -> four resident; authoritative seeds; zero-transfer bind; V13 four-handle restore");
}

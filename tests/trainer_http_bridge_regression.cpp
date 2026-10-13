// Collaboration signature: GPT6-Astra. Real IPC adapter, no formal Trainer or training loop.
#include "competency_training.h"
#include "../DQN_Main_TUI_No_Terminal/trainer_http_bridge.h"
#include <memory>
#include <cstdio>
float g_ep_time=0,g_elapsed=0,g_remaining=0;
volatile double g_phase_forward_us=0,g_phase_backward_us=0,g_phase_target_us=0;
volatile int g_phase_steps=0,g_side_steps[2]={};
volatile double g_side_forward_us[2]={},g_side_backward_us[2]={};
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);stop_ipc_server();return 1;}}while(0)
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(independent_training::indep_start_from_shared(a.get()));publish_training_status(a.get());start_ipc_server();Sleep(200);
    auto status=tui_http::trainer_command("GET_STATUS");NEED(status.status==200 && status.body.find("\"data\"")!=status.body.npos);
    float before=g_hot_config.learning_rate_override;
    const float planned=g_learning_rate;
    a->indep->set_learning_rate(independent_training::Side::Black,planned);
    a->indep->set_learning_rate(independent_training::Side::White,planned*.5f);
    auto queued=tui_http::trainer_command("SET learning_rate 0.0003");NEED(queued.status==202 && queued.body.find("queued")!=queued.body.npos);
    NEED(g_hot_config.learning_rate_override==before);NEED(apply_pending_ipc_config(a.get()));publish_training_status(a.get());
    auto config=tui_http::trainer_command("GET_CONFIG");NEED(config.status==200 && config.body.find("curriculum_enabled")!=config.body.npos);
    NEED(fabsf(g_hot_config.learning_rate_override-.0003f)<1e-7f);
    NEED(fabsf(a->indep->updates(independent_training::Side::Black).learning_rate-.0003f)<1e-7f);
    NEED(fabsf(a->indep->updates(independent_training::Side::White).learning_rate-.0003f)<1e-7f);
    NEED(tui_http::trainer_command("SET learning_rate -1").status==202);NEED(apply_pending_ipc_config(a.get()));
    NEED(a->indep->updates(independent_training::Side::Black).learning_rate==planned);
    NEED(a->indep->updates(independent_training::Side::White).learning_rate==planned*.5f);
    auto skills=tui_http::trainer_command("GET_COMPETENCY");NEED(skills.status==200 && skills.body.find("\"side\":1")!=skills.body.npos);
    NEED(tui_http::trainer_command("SET regression_tests 1").status==400);
    stop_ipc_server();free_dqn(a.get());g_current_agent=nullptr;
    puts("PASS separate HTTP bridge connection, real Trainer IPC, queued-versus-applied config, two-color query, rejection");return 0;
}

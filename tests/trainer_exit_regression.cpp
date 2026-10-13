// Exercise the actual Trainer functions, but never its entry point, UI, guard,
// or a training session. Only synthetic state and files in a fresh test cwd.
#define UNICODE
#define _UNICODE
#define wWinMain unused_trainer_entry
#include "../DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp"
#undef wWinMain
#include <thread>
#include <atomic>
#include <memory>
static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n",__LINE__,#x); ++failures; } } while(0)

static void exit_branch(DQNAgent* agent, bool pretrain) {
    g_exit_requested = 1; g_training_fault = 0; g_shutdown_state = 0;
    g_ctx = {}; g_ctx.total_games_played = 12;
    const char* path = pretrain ? "checkpoints/checkpoint_phase0_12.bin" : "checkpoints_temp/checkpoint_phase1_12.bin";
    char temporary[256]; snprintf(temporary,sizeof(temporary),"%s.tmp",path);
    // A directory at the exact temporary filename forces the real writer to fail.
    CHECK(CreateDirectoryA(temporary,nullptr));
    std::atomic<bool> saw_failure{false}, timed_out{false};
    std::thread retry([&]() {
        for (int i=0;i<300 && g_shutdown_state!=3;++i) Sleep(10);
        saw_failure = g_shutdown_state==3;
        if (!saw_failure) { timed_out=true; training_fail(1); return; }
        Sleep(100);
        // Only remove the empty fault fixture created above, not a user file.
        if (!RemoveDirectoryA(temporary)) { timed_out=true; training_fail(1); return; }
        char response[256]; process_ipc_command("SAVE_AND_EXIT\r\n",response,sizeof(response));
    });
    if (pretrain) CHECK(train_single_player_pretrain(agent,12)==0);
    else train_dqn(agent,100,1,1,.05f,1000,1,-1,1,12,2,nullptr,-1);
    retry.join();
    CHECK(saw_failure && !timed_out && g_shutdown_state==4 && !g_training_fault);
    CHECK(GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE)!=INVALID_FILE_ATTRIBUTES);
    CHECK(g_ctx.total_games_played==12); // no extra game while exiting or waiting
    CHECK(GetFileAttributesA(path)!=INVALID_FILE_ATTRIBUTES);
    printf("ACTUAL TRAINER %s: failed save paused, IPC retry saved, completed episodes unchanged\n",pretrain?"PHASE0":"FORMAL");
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    CreateDirectoryA("logs",nullptr);CreateDirectoryA("checkpoints",nullptr);CreateDirectoryA("checkpoints_temp",nullptr);
    auto agent=std::make_unique<DQNAgent>(); init_dqn_ex(agent.get(),0);
    g_trainer_request_exit_cb=request_graceful_exit;
    // Signal context changes flags only: no file/log/cache calls. The owner
    // subsequently publishes intentional shutdown and uses the normal save path.
    signal_handler(SIGTERM);
    CHECK(g_exit_requested && g_pending_exit_signal == SIGTERM);
    CHECK(GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE)==INVALID_FILE_ATTRIBUTES);
    CHECK(GetFileAttributesA("dqn_config.txt")==INVALID_FILE_ATTRIBUTES);
    service_signal_exit_request();
    CHECK(g_pending_exit_signal==0 && GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE)!=INVALID_FILE_ATTRIBUTES);
    exit_branch(agent.get(),true);exit_branch(agent.get(),false);
    g_exit_requested=0; g_training_fault=1; g_shutdown_state=0;
    long long previous=agent->global_step;
    train_dqn(agent.get(),100,1,1,.05f,1000,1,-1,1,12,2,nullptr,-1);
    CHECK(g_shutdown_state==5 && agent->global_step==previous);
    CHECK(GetFileAttributesA("checkpoints/checkpoint_phase2_0.bin")==INVALID_FILE_ATTRIBUTES);
    remove(INTENTIONAL_SHUTDOWN_FILE); remove(TRAINING_FAULT_FILE);
    g_exit_requested=0; g_training_fault=2; g_shutdown_state=0;
    request_graceful_exit();
    CHECK(g_exit_requested && g_shutdown_state==5);
    CHECK(GetFileAttributesA(TRAINING_FAULT_FILE)!=INVALID_FILE_ATTRIBUTES);
    CHECK(GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE)==INVALID_FILE_ATTRIBUTES);
    g_current_agent=nullptr;g_trainer_request_exit_cb=nullptr;free_dqn(agent.get());
    printf("TRAINER EXIT REGRESSION: %s (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}

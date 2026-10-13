// Reuse the original fault-injection probe without its non-asserting main.
#define main original_recheck_probe_main
#include "recheck_fixtures.h"
#undef main
#include "shutdown_save.h"
static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static void numeric_assertions() {
    auto a = std::make_unique<DQNAgent>();
    a->q_net.b2[0] = 1e-30f; a->q_net.w3[0][0] = 1e30f; a->q_net.w4[0][0] = 1e30f;
    auto original = std::make_unique<NeuralNet>(a->q_net);
    float states[52] = {}, targets[2] = {1, 0}; int actions[2] = {1, 0}; float loss = 0;
    g_training_fault = 0; g_trainer_request_exit_cb = record_exit; exit_requests = 0;
    CHECK(!train_agent_batch(a.get(), states, actions, targets, 2, .001f, &loss));
    CHECK(memcmp(original.get(), &a->q_net, sizeof(NeuralNet)) == 0);
    CHECK(g_training_fault == 2 && exit_requests == 1);
    CHECK(!train_agent_batch(a.get(), states, actions, targets, 1, .001f, &loss));
    CHECK(exit_requests == 1);
    CHECK(!isfinite(train_step_cpu_only(&a->q_net, states, 0, 0, .001f)));
    CHECK(memcmp(original.get(), &a->q_net, sizeof(NeuralNet)) == 0);
    g_training_fault = 0; exit_requests = 0;
    a->target_net = *original;
    CHECK(gpu_net_activate(a.get()) == 0);
    auto before = std::make_unique<NeuralNet>(); auto after = std::make_unique<NeuralNet>();
    /* 不再直取内部句柄：用公开的只读快照读共享 online（非独立模式 agent）。 */
    CHECK(gpu_net_snapshot_q(before.get()) == 0);
    CHECK(!train_agent_batch(a.get(), states, actions, targets, 2, .001f, &loss));
    CHECK(g_training_fault == 2 && exit_requests == 1);
    CHECK(gpu_net_snapshot_q(after.get()) == 0);
    CHECK(memcmp(before.get(), after.get(), sizeof(*before)) == 0);
    CHECK(nonfinite((float*)after.get(), sizeof(*after)/sizeof(float)) == 0);
    gpu_net_deactivate(); g_training_fault = 0; g_trainer_request_exit_cb = nullptr;

    // A finite high Huber loss is observable, but must not update parameters.
    auto finite = std::make_unique<NeuralNet>();
    memset(finite.get(), 0, sizeof(*finite));
    auto finite_before = std::make_unique<NeuralNet>(*finite);
    float high_loss = train_step_cpu_only(finite.get(), states, 0, -500.0f, .001f);
    CHECK(isfinite(high_loss) && high_loss > 100.0f);
    CHECK(memcmp(finite_before.get(), finite.get(), sizeof(*finite)) == 0);

    auto gpu_finite = std::make_unique<DQNAgent>();
    memset(gpu_finite.get(), 0, sizeof(*gpu_finite));
    g_training_fault = 0;
    CHECK(gpu_net_activate(gpu_finite.get()) == 0);
    float gpu_high_loss = 0.0f;
    float high_target = -500.0f;
    CHECK(train_agent_batch(gpu_finite.get(), states, actions, &high_target, 1, .001f, &gpu_high_loss));
    CHECK(gpu_high_loss == 0.0f && g_training_fault == 0);
    auto gpu_after = std::make_unique<NeuralNet>();
    CHECK(gpu_net_snapshot_q(gpu_after.get()) == 0 && gpu_after->b4[actions[0]] == 0.0f);
    gpu_net_deactivate();
    puts("NUMERIC: rejected CPU/GPU overflowing batches, byte-identical live weights retained");
}
static void shutdown_assertions(DQNAgent* a) {
    // Real checkpoint writer failure: parent does not exist, then repair storage.
    g_training_fault = 0; g_shutdown_state = 0;
    int saves = 0, pumps = 0;
    CHECK(save_before_exit([&]() { ++saves; return save_checkpoint(a, 0, 0, 2, "retry/checkpoint.bin") != 0; }, [&]() {
        CHECK(g_shutdown_state == 3); ++pumps;
        if (pumps == 3) {
            CHECK(saves == 1); // no busy retries or training between requests
            CreateDirectoryA("retry", nullptr);
            InterlockedCompareExchange(&g_shutdown_state, 1, 3);
        }
    }));
    CHECK(saves == 2 && pumps == 3 && g_shutdown_state == 4);
    CHECK(GetFileAttributesA("retry/checkpoint.bin") != INVALID_FILE_ATTRIBUTES);
    char status[4096]; build_status_json(status,sizeof(status));
    CHECK(strstr(status,"\"shutdown_state\":4") != nullptr);
    g_training_fault = 1; saves = 0;
    CHECK(!save_before_exit([&]() { ++saves; return true; }, []() {}));
    CHECK(saves == 0 && g_shutdown_state == 5);
    g_training_fault = 0; g_shutdown_state = 0;
    puts("SHUTDOWN: real write failure retained memory until explicit retry; fatal path never claimed saved");
}
static void report_assertions() {
    report_probe();
    tui_report_exec("analyze");
    CHECK(strstr(tui_report_get_text(),"2.000000") != nullptr);
    /* GPT5-Sol: assert the schema's language-neutral record identity/value so
       the test is not coupled to console code-page rendering of Chinese text. */
    CHECK(strstr(tui_report_get_text(),
        "record_index=2 episode=3 replay_type=0 loss=null") != nullptr);
    CHECK(strstr(tui_report_get_text(),"nan") == nullptr);
    EvalRecord rows[512]; int n = load_eval_records(nullptr, rows, 512);
    CHECK(n == 510 && rows[n-1].episode == 200000 && rows[0].mode == 0);
    n = load_eval_records(nullptr,rows,512,1000);
    CHECK(n == 3 && rows[0].episode == 1000 && rows[2].mode == 2);
    n = load_eval_records(nullptr,rows,512,200000);
    CHECK(n == 3 && rows[2].episode == 200000);
    tui_report_exec("eval 200000");
    CHECK(strstr(tui_report_get_text(),"Episode 200000") != nullptr);
    puts("REPORT: large loss kept, missing excluded; oldest and newest checkpoint modes queryable");
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    CreateDirectoryA("logs",nullptr); CreateDirectoryA("checkpoints",nullptr);
    if (gpu_init() != 0) return 2;
    auto a=std::make_unique<DQNAgent>(); srand(123); init_dqn_ex(a.get(),0);
    hard_failure_probe(a.get());
    CHECK(exit_requests == 1 && a->global_step == 0 && g_training_fault == 1);
    g_training_fault = 0;
    reward_feedback_probe(a.get());
    float weights[7]; memcpy(weights,a->reward_weights,sizeof(weights));
    for (float& w:a->reward_weights) w=1;
    a->reward_stats[3]=100;
    adjust_reward_weights(a.get(),2,1);
    CHECK(memcmp(weights,a->reward_weights,sizeof(weights))==0 && a->reward_weights[3]==1);
    shutdown_assertions(a.get()); free_dqn(a.get());
    numeric_assertions(); gpu_shutdown(); report_assertions();
    printf("RECHECK REGRESSION: %s (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}

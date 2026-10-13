#define _CRT_SECURE_NO_WARNINGS
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include "core.h"
#include "skill_test.h"
#include "training_resume.h"
#include "network_input.h"
#include "skill_metric_reader.h"
#include "skill_metric_writer.h"
#include "neural_probe_schedule.h"
#include <vector>
#include <memory>
#include <assert.h>

// Reuse the project's parameter packing helpers, without its diagnostic main.
#define main old_gpu_gradient_test_main
#include "../DQN_Core/src/test_gpu_train.cpp"
#undef main

extern "C" int gpu_nn_forward(GpuNetHandle, const float*, float*);
static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static void test_probe_publish_schedule() {
    ProbePublishSchedule schedule = {};
    CHECK(!probe_schedule_pretrain(&schedule, 0, 100));
    CHECK(probe_schedule_pretrain(&schedule, 1, 100));
    CHECK(!probe_schedule_pretrain(&schedule, 1, 1099));
    CHECK(probe_schedule_pretrain(&schedule, 1, 1100));
    CHECK(!probe_schedule_pretrain(&schedule, 0, 1200));
    CHECK(probe_schedule_pretrain(&schedule, 1, 1201));

    schedule = {};
    CHECK(probe_schedule_formal_step(&schedule, 1)); /* immediate first frame */
    for (int i = 0; i < PROBE_FORMAL_PUBLISH_STEPS - 1; ++i)
        CHECK(!probe_schedule_formal_step(&schedule, 1));
    CHECK(probe_schedule_formal_step(&schedule, 1));
    CHECK(!probe_schedule_formal_step(&schedule, 0));
    CHECK(probe_schedule_formal_step(&schedule, 1)); /* reconnect is immediate */
    printf("Probe schedule: pretrain=%ums, formal=%u steps\n",
        PROBE_PRETRAIN_PUBLISH_MS, PROBE_FORMAL_PUBLISH_STEPS);
}

static void test_terminal_win_reward_scales() {
    auto agent = std::make_unique<DQNAgent>();
    memset(agent.get(), 0, sizeof(*agent));
    /* This focused test stores only the original and one colour-flipped sample.
       Supply real backing slots because the zeroed test agent does not allocate
       the production replay buffers. */
    Experience black_slot = {}, white_slot = {};
    agent->replay_black = &black_slot;
    agent->replay_white = &white_slot;
    GameState before = {}, after = {};

    before.turn = 0;
    before.board[0] = before.board[1] = before.board[2] = 1;
    after = before;
    after.board[3] = 1;
    after.turn = 1;

    set_terminal_win_reward_scales(0.85f, 1.15f);
    const float black_reward = compute_reward(agent.get(), &before, &after,
        3, 0, 1, 0, 3);
    CHECK(fabsf(black_reward - 0.85f) < 1e-6f);
    float black_state[CELL_COUNT + 1], black_next[CELL_COUNT + 1];
    state_to_input(&before, black_state);
    state_to_input(&after, black_next);
    store_experience(agent.get(), black_state, 3, black_reward,
        black_next, 1, 0);
#if SYMMETRY_ENABLE
    CHECK(fabsf(agent->replay_black[0].reward - 0.85f) < 1e-6f);
    CHECK(fabsf(agent->replay_white[0].reward - 1.15f) < 1e-6f);
#endif

    memset(&before, 0, sizeof(before));
    before.turn = 1;
    before.board[5] = before.board[6] = before.board[7] = 2;
    after = before;
    after.board[8] = 2;
    after.turn = 0;
    const float white_reward = compute_reward(agent.get(), &before, &after,
        8, 1, 1, 0, 3);
    CHECK(fabsf(white_reward - 1.15f) < 1e-6f);

    /* Runtime feedback must not leak into unrelated tests or consumers. */
    set_terminal_win_reward_scales(1.0f, 1.0f);
    printf("Terminal reward scales: black=%g white=%g\n", black_reward, white_reward);
}

static void test_complete_model_package_and_memory_overlay() {
    auto expected = std::make_unique<NeuralNet>();
    auto restored = std::make_unique<NeuralNet>();
    srand(77); init_net(expected.get());
    memory_system_init();
    memset(&g_knowledge_base[0], 0, sizeof(KBEntry));
    int successor[CELL_COUNT] = {};
    successor[12] = 1;
    extract_patches(successor, g_knowledge_base[0].patches);
    for (int p = 0; p < PATCH_COUNT; ++p) g_knowledge_base[0].patch_attention[p] = 1.0f;
    KBEntry& kb = g_knowledge_base[0];
    kb.sample_count = kb.total = kb.wins = 100;
    kb.win_rate = 1.0f; kb.last_queried = 1; kb.timeout_threshold = 1000;
    kb.is_positive = 1; kb.avg_step = 1.0f; kb.black_wins = 100;
    kb.avg_piece_count = 1.0f; kb.perspective = 0; kb.phase_bucket = 0;
    g_kb_count = 1;
    g_hot_config.memory_scale = 0.5f;
    g_hot_config.memory_confidence = 0.3f;

    float scores[CELL_COUNT] = {};
    GameState empty = {}; empty.turn = 0;
    memory_apply_model_scores(&empty, scores);
    CHECK(scores[12] > 0.0f); /* deployed inference really consumed the bank */

    const char* package = "complete_model_fixture.bin";
    CHECK(memory_save_model_package(expected.get(), package, 43210));
    memory_system_init();
    memset(restored.get(), 0, sizeof(NeuralNet));
    int episode = -1;
    CHECK(memory_load_model_package(restored.get(), package, &episode));
    CHECK(episode == 43210 && g_kb_count == 1 && g_cache_count == 0);
    CHECK(memcmp(expected.get(), restored.get(), sizeof(NeuralNet)) == 0);
    memset(scores, 0, sizeof(scores));
    memory_apply_model_scores(&empty, scores);
    CHECK(scores[12] > 0.0f);
    remove(package);
    memory_system_init();
    printf("Complete model package restored matching weights and memory overlay\n");
}

static void test_input_and_gpu() {
    auto net = std::make_unique<NeuralNet>();
    srand(1234); init_net(net.get());
    GameState board = {};
    board.board[0] = board.board[1] = board.board[2] = 1;
    board.board[22] = board.board[23] = board.board[24] = 2;
    float states[8][26] = {}, q0[25], q1[25], gpu[8][25];
    state_to_input(&board, states[0]);
    memcpy(states[1], states[0], sizeof(states[0])); states[1][25] = 1;
    forward_cpu_only(net.get(), states[0], q0);
    forward_cpu_only(net.get(), states[1], q1);
    float difference = 0;
    for (int a = 0; a < 25; ++a) difference = fmaxf(difference, fabsf(q0[a] - q1[a]));
    CHECK(difference > 0);
    // A hand-wired path gives a large, exact side-to-move signal, independent
    // of a randomly initialized network's nearly symmetric outputs.
    auto signal = std::make_unique<NeuralNet>(); memset(signal.get(), 0, sizeof(*signal));
    signal->token_embed[2][0] = 1; signal->token_embed[0][0] = 2;
    signal->pool_w[0][0] = 1;
    signal->w1[0][0] = 10;
    signal->w2[0][0] = signal->w3[0][0] = signal->w4[0][0] = 1;
    float signal0[25], signal1[25];
    // Average pooling: use unequal colour counts for this linear test path.
    float signal_state[26] = {}; signal_state[0] = 1;
    forward_cpu_only(signal.get(), signal_state, signal0);
    signal_state[25] = 1;
    forward_cpu_only(signal.get(), signal_state, signal1);
    CHECK(fabsf(signal0[0] - signal1[0]) > 0.1f);
    for (int a = 0; a < 25; ++a) states[2][a] = -states[0][a];
    states[2][25] = 1;
    forward_cpu_only(net.get(), states[2], q1);
    for (int a = 0; a < 25; ++a) CHECK(q0[a] == q1[a]);
    for (int i = 3; i < 8; ++i) {
        for (int a = 0; a < 25; ++a) states[i][a] = (float)(rand() % 3 - 1);
        states[i][25] = (float)(i % 2);
    }
    CHECK(gpu_init() == 0);
    CHECK(forward_gpu_snapshot(net.get(), &states[0][0], 8, &gpu[0][0]) == 0);
    float parity = 0;
    for (int i = 0; i < 8; ++i) {
        forward_cpu_only(net.get(), states[i], q0);
        for (int a = 0; a < 25; ++a) {
            CHECK(isfinite(gpu[i][a]));
            parity = fmaxf(parity, fabsf(gpu[i][a] - q0[a]));
        }
    }
    CHECK(parity < 1e-5f);
    printf("Turn difference=%g, CPU/GPU forward difference=%g\n", difference, parity);

    GpuNetHandle h = {};
    CHECK(gpu_nn_create(&h) == 0);
    auto params = std::make_unique<GpuNetParams>();
    pack(net.get(), params.get());
    CHECK(gpu_nn_upload(h, params.get(), sizeof(*params)) == 0);
    std::vector<float> poison(2500, NAN), scores(2500);
    void* cache = (char*)h.work + offsetof(GpuWork, scores);
    CHECK(hipMemcpy(cache, poison.data(), poison.size()*sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    CHECK(gpu_nn_forward(h, states[0], q0) == 0);
    CHECK(hipMemcpy(scores.data(), cache, scores.size()*sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    for (int row = 0; row < 100; ++row) {
        float sum = 0;
        for (int j = 0; j < 25; ++j) { CHECK(isfinite(scores[row*25+j])); sum += scores[row*25+j]; }
        CHECK(fabsf(sum - 1) < 1e-5f);
    }
    auto expected = std::make_unique<GpuNetParams>();
    auto initial = std::make_unique<GpuNetParams>(*params);
    g_learning_rate = 0.0004f;
    float gradient_diff = 0;
    for (int i = 0; i < 8; ++i) {
        CHECK(hipMemcpy(cache, poison.data(), poison.size()*sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        float loss_cpu = train_step(net.get(), states[i], 3, 0.6f), loss_gpu = 0;
        CHECK(gpu_nn_train_step(h, states[i], 3, 0.6f, g_learning_rate, &loss_gpu) == 0);
        CHECK(isfinite(loss_gpu) && fabsf(loss_cpu-loss_gpu) < 1e-5f);
        CHECK(gpu_nn_download(h, params.get(), sizeof(*params)) == 0);
        pack(net.get(), expected.get());
        const float* actual = (float*)params.get();
        const float* wanted = (float*)expected.get();
        for (size_t j = 0; j < sizeof(*params)/sizeof(float); ++j) {
            CHECK(isfinite(actual[j]));
            gradient_diff = fmaxf(gradient_diff, fabsf(actual[j]-wanted[j]));
        }
    }
    CHECK(memcmp(initial->attn_wq, params->attn_wq, sizeof(params->attn_wq)) != 0);
    CHECK(memcmp(initial->attn_wk, params->attn_wk, sizeof(params->attn_wk)) != 0);
    CHECK(memcmp(initial->attn_wv, params->attn_wv, sizeof(params->attn_wv)) != 0);
    CHECK(gradient_diff < 2e-6f);
    printf("8-step CPU/GPU parameter difference=%g; poisoned cache overwritten\n", gradient_diff);
    gpu_nn_destroy(&h);
}

static void test_puzzles_and_scores() {
    GameState puzzles[512]; int answers[512][25], na[512], count;
    for (int kind = 0; kind < 2; ++kind) {
        if (kind) generate_block1_fixed(puzzles, answers, na, &count, 512);
        else generate_win1_fixed(puzzles, answers, na, &count, 512);
        CHECK(count > 0);
        for (int i = 0; i < count; ++i) {
            CHECK(!check_winner(&puzzles[i])); CHECK(na[i] > 0);
            for (int j = 0; j < i; ++j)
                CHECK(puzzles[i].turn != puzzles[j].turn ||
                    memcmp(puzzles[i].board, puzzles[j].board, sizeof(puzzles[i].board)) != 0);
            for (int k = 0; k < na[i]; ++k) {
                GameState next = puzzles[i];
                int self = next.turn + 1, opp = 3-self;
                CHECK(next.board[answers[i][k]] == 0);
                next.board[answers[i][k]] = self;
                if (!kind) CHECK(check_winner(&next) == self);
                else if (check_winner(&next) != self) {
                    for (int a = 0; a < 25; ++a) if (!next.board[a]) {
                        GameState reply = next; reply.board[a] = opp;
                        CHECK(check_winner(&reply) != opp);
                    }
                }
            }
        }
        printf("%s fixed questions=%d; terminal/duplicate/unsafe answers=0\n", kind ? "Block" : "Win", count);
    }
    auto agent = std::make_unique<DQNAgent>(); memset(agent.get(), 0, sizeof(*agent));
    // A zero network must not pass by deterministic board-index tie breaking.
    SkillTestResult result;
    run_skill_tests(agent.get(), 0, &result, 0);
    CHECK(result.empty_games == SKILL_EMPTY_TRIALS && result.empty_avg_steps > 5 && result.empty_win_rate > 0);
    CHECK(result.parity_status == 0 && isnan(result.parity_max_diff));
    CHECK(result.seeded_win_rate == 1 && result.seeded_avg_steps >= 4);
    generate_win1_fixed(puzzles, answers, na, &count, 512);
    int right = answers[0][0], wrong = -1;
    for (int a = 0; a < 25; ++a) {
        bool is_answer = false; for (int k = 0; k < na[0]; ++k) is_answer |= a == answers[0][k];
        if (!puzzles[0].board[a] && !is_answer) { wrong = a; break; }
    }
    agent->q_net.b4[right] = 2;
    CHECK(test_puzzle(agent.get(), &puzzles[0], answers[0], na[0], 0).is_correct);
    agent->q_net.b4[wrong] = 3;
    PuzzleResult pr = test_puzzle(agent.get(), &puzzles[0], answers[0], na[0], 0);
    CHECK(!pr.is_correct && pr.margin == -1);
    for (float& value : agent->q_net.b4) value = NAN;
    run_skill_tests(agent.get(), 0, &result, 0);
    CHECK(result.empty_win_rate == 0 && result.seeded_win_rate == 0);
    CHECK(result.empty_avg_steps == 0 && result.seeded_avg_steps == 0);

    SkillTracker tracker; skill_tracker_init(&tracker);
    result = {}; result.suite_version = SKILL_SUITE_VERSION; result.total_tests = 1;
    result.empty_win_rate = 1; result.empty_avg_steps = 4; result.empty_shortest_rate = 1;
    result.win1_fixed = 0.99f;
    CHECK(skill_tracker_update(&tracker, &result) == 0);
    result.empty_avg_steps = 19; result.empty_shortest_rate = 0;
    CHECK(skill_tracker_update(&tracker, &result) == 1);
    CHECK(tracker.empty_steps_baseline == 4);
    result.empty_avg_steps = 4; result.empty_shortest_rate = 1;
    result.win1_fixed = 0.99f;
    for (int i = 0; i < 30; ++i) skill_tracker_update(&tracker, &result);
    CHECK(tracker.win1_baseline == 0.99f && tracker.empty_steps_baseline == 4);
}

static void test_multi_snapshot_skill_window() {
    auto first = std::make_unique<NeuralNet>();
    auto second = std::make_unique<NeuralNet>();
    memset(first.get(), 0, sizeof(*first));
    memcpy(second.get(), first.get(), sizeof(*second));
    const NeuralNet* snapshots[2] = {first.get(), second.get()};

    SkillTestResult empty{};
    int progress = 0, phase = 0;
    CHECK(run_multi_snapshot_empty(snapshots, 2, &empty, &progress, &phase));
    CHECK(empty.empty_games == 2 && progress == 100 && phase == 2);
    CHECK(isfinite(empty.empty_win_rate) && empty.empty_win_rate >= 0.0f && empty.empty_win_rate <= 1.0f);
    CHECK(empty.empty_stat_min <= empty.empty_stat_mean && empty.empty_stat_mean <= empty.empty_stat_max);

    MultiSnapshotResult win{}, block{};
    run_multi_snapshot_win1(snapshots, 2, &win, &progress, &phase);
    run_multi_snapshot_block1(snapshots, 2, &block, &progress, &phase);
    CHECK(win.snapshot_count == 2 && block.snapshot_count == 2);
    CHECK(win.acc_std == 0.0f && block.acc_std == 0.0f); // identical frozen models
    CHECK(win.current_accuracy == win.snapshots[1].accuracy);
    CHECK(block.current_accuracy == block.snapshots[1].accuracy);
    printf("Multi-snapshot window: Empty %.1f%%, Win1 %.1f%%, Block1 %.1f%%\n",
        empty.empty_win_rate * 100.0f, win.acc_mean * 100.0f, block.acc_mean * 100.0f);
}

static void test_checkpoint_and_resume() {
    auto agent = std::make_unique<DQNAgent>(); memset(agent.get(), 0, sizeof(*agent));
    init_dqn(agent.get());
    g_start_time = time(NULL); g_learning_rate = 0.000256f;
    agent->pretrain_consolidation_target = 10000; agent->pretrain_stable_passes = 3;
    skill_tracker_init(&agent->skill_tracker);
    agent->skill_tracker.initialized = 1; agent->skill_tracker.empty_steps_baseline = 4;
    agent->replay_black_count = agent->replay_black_idx = 1;
    Experience& exp0 = agent->replay_black[0];
    memset(&exp0, 0, sizeof(exp0));
    exp0.action = 0; exp0.reward = 0.75f;
    exp0.next_state[0] = 1.0f; exp0.next_state[CELL_COUNT] = 1.0f;
    CHECK(save_checkpoint(agent.get(), 0, 9000, 2, "checkpoints/checkpoint_phase0_9000.bin"));
    agent->pretrain_consolidation_target = agent->pretrain_stable_passes = 0;
    agent->skill_tracker.empty_steps_baseline = 0;
    agent->replay_black[0].reward = 0; g_learning_rate = 0.005f;
    int phase = -1, ep = -1, depth = -1;
    CHECK(load_checkpoint(agent.get(), &phase, &ep, &depth, "checkpoints/checkpoint_phase0_9000.bin", 1));
    CHECK(phase == 0 && ep == 9000 && depth == 2 && agent->checkpoint_version == 11);
    CHECK(agent->pretrain_consolidation_target == 10000 && agent->pretrain_stable_passes == 3);
    CHECK(g_learning_rate == 0.000256f && agent->skill_tracker.empty_steps_baseline == 4);
    CHECK(agent->replay_black[0].reward == 0.75f);
    /* A corrupt newest checkpoint is quarantined and the valid predecessor is
       selected, instead of making the no-terminal Trainer immediately exit. */
    FILE* corrupt = fopen("checkpoints/checkpoint_phase0_10000.bin", "wb");
    int corrupt_header[4] = { 7, 0x5A4D4E54, 0, 10000 };
    CHECK(corrupt != NULL && fwrite(corrupt_header, sizeof(corrupt_header), 1, corrupt) == 1);
    if (corrupt) fclose(corrupt);
    char latest[256] = {}; int latest_ep = -1;
    CHECK(find_latest_checkpoint(latest, sizeof(latest), &latest_ep));
    CHECK(latest_ep == 9000 && strstr(latest, "checkpoint_phase0_9000.bin") != NULL);
    CHECK(GetFileAttributesA("checkpoints/checkpoint_phase0_10000.bin") == INVALID_FILE_ATTRIBUTES);
    WIN32_FIND_DATAA quarantined = {};
    HANDLE quarantine_find = FindFirstFileA(
        "checkpoints_quarantine/checkpoint_phase0_10000.bin.invalid_*", &quarantined);
    CHECK(quarantine_find != INVALID_HANDLE_VALUE);
    if (quarantine_find != INVALID_HANDLE_VALUE) FindClose(quarantine_find);
    // Fixture: V5 body has no V6 extension and no checksum trailer.
    FILE* src = fopen("checkpoints/checkpoint_phase0_9000.bin", "rb");
    FILE* old = fopen("legacy_v5.bin", "wb");
    auto meta = std::make_unique<CheckpointMeta>();
    CHECK(fread(meta.get(), sizeof(*meta), 1, src) == 1); meta->version = 5;
    CHECK(fwrite(meta.get(), sizeof(*meta), 1, old) == 1);
    /* GPT5-Sol: the current V11 body inserts both training state and the
       side-balance controller before replay.  A synthetic V5 fixture must
       omit both blocks or its replay begins at the wrong byte offset. */
    CHECK(_fseeki64(src,
        sizeof(CheckpointTrainingState) + sizeof(SideBalanceController), SEEK_CUR) == 0);
    std::vector<char> replay(sizeof(Experience)*REPLAY_CAPACITY*2);
    CHECK(fread(replay.data(), 1, replay.size(), src) == replay.size());
    CHECK(fwrite(replay.data(), 1, replay.size(), old) == replay.size());
    fclose(src); fclose(old);
    CHECK(load_checkpoint(agent.get(), &phase, &ep, &depth, "legacy_v5.bin", 1));
    CHECK(agent->checkpoint_version == 5 && agent->replay_black[0].reward == 0.75f);
    FILE* log = fopen("pretrain_fixture.dat", "w");
    fprintf(log, "EVAL 8000 1 4 3\nCONSOLIDATE 8000 10000\nEVAL 8500 1 4 4\nCONSOLIDATE 18000 20000\n");
    fclose(log);
    int target = 0, passes = 0;
    restore_pretrain_progress("pretrain_fixture.dat", 9000, &target, &passes);
    CHECK(target == 10000 && passes == 4);
    restore_pretrain_progress("pretrain_fixture.dat", 7500, &target, &passes);
    CHECK(target == 0 && passes == 0);
    free_dqn(agent.get());
    printf("V11 curriculum/LR/baseline/replay roundtrip and V5 compatibility checked\n");
}

static void test_late_and_partial_metric() {
    char path[96]; snprintf(path, sizeof(path), "metric_fixture_%lu.bin", GetTickCount());
    SkillDisplayRecord rows[10];
    CHECK(read_skill_metrics_tail(path, rows, 10) == -1);
    SkillMetricRecord rec = {};
    rec.episode = 0;
    rec.empty_win = 1.0f;
    rec.avg_steps = 4.0f;
    rec.shortest = 1.0f;
    rec.win1_fixed = 0.99f;
    rec.block1_fixed = 0.98f;
    write_skill_metric(path, &rec);
    CHECK(read_skill_metrics_tail(path, rows, 10) == 1);
    CHECK(rows[0].episode == 0 && rows[0].avg_steps == 4);
    rec.episode = 1000; write_skill_metric(path, &rec);
    FILE* f = fopen(path, "ab"); char partial[10] = {}; fwrite(partial, 1, sizeof(partial), f); fclose(f);
    CHECK(read_skill_metrics_tail(path, rows, 10) == 2);
    CHECK(rows[1].episode == 1000 && rows[1].win1_fixed == 0.99f);
    CHECK(read_skill_metrics_tail(path, rows, 1) == 1 && rows[0].episode == 1000);
    printf("Late-created metric and partial trailing record checked\n");
}

static void test_incompatible_skill_metric_is_quarantined() {
    const char* path = "test_metric_skill_incompatible.bin";
    remove(path);
    SkillMetricHeader old{SKILL_MAGIC, 7, 1, SKILL_SUITE_VERSION, {0, 0}};
    FILE* f = fopen(path, "wb");
    CHECK(f != nullptr);
    if (!f) return;
    fwrite(&old, sizeof(old), 1, f);
    fclose(f);

    SkillMetricRow row{};
    row.episode = 32000;
    row.fields[0] = 1.0f;
    row.fields[1] = 4.0f;
    row.fields[2] = 1.0f;
    row.fields[3] = 4.0f;
    row.fields[4] = 21.0f;
    row.fields[5] = 0.82f;
    row.fields[6] = -0.156f;  // margins and Q statistics legitimately cross zero
    row.fields[7] = 0.80f;
    row.fields[8] = -0.68f;
    row.fields[9] = 0.13f;
    row.fields[10] = -1.43f;
    row.fields[11] = 1.12f;
    row.fields[12] = -0.68f;
    row.fields[13] = 1.12f;
    row.fields[14] = 0.705f;
    row.fields[15] = -0.50f;
    row.fields[16] = 0.70f;
    row.fields[23] = 0.05f;
    row.fields[24] = 0.95f;
    row.fields[25] = 1.0f;
    row.fields[26] = 0.10f;
    row.fields[27] = 0.12f;
    row.fields[28] = 150.0f;
    CHECK(skill_metric_row_valid(row));
    CHECK(append_skill_metric(path, row));
    f = fopen(path, "rb");
    SkillMetricHeader now{};
    CHECK(f && fread(&now, sizeof(now), 1, f) == 1 &&
        now.fields == SKILL_METRIC_FIELDS && now.suite == SKILL_SUITE_VERSION);
    if (f) fclose(f);
    SkillDisplayRecord displayed{};
    CHECK(read_skill_metrics_tail(path, &displayed, 1) == 1);
    CHECK(displayed.episode == 32000 && displayed.empty_min_steps == 4.0f &&
        displayed.empty_max_steps == 21.0f && displayed.win1_fixed == 0.82f &&
        displayed.win1_fixed_margin == -0.156f);
    remove(path);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    test_probe_publish_schedule();
    test_terminal_win_reward_scales();
    test_complete_model_package_and_memory_overlay();
    test_input_and_gpu();
    test_puzzles_and_scores();
    test_multi_snapshot_skill_window();
    test_checkpoint_and_resume();
    test_late_and_partial_metric();
    test_incompatible_skill_metric_is_quarantined();
    printf("TRAINING REGRESSION: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

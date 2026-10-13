// skill_eval_side_routing_regression.cpp — 评测路由专用测试（deepseek-flash）
//
// 验收目标（P0 第 1 项）：
//   `run_skill_tests` 内部 `get_q_values()`（DQN_Core/src/skill_test.cpp:419）必须按
//   **题目棋色** `g->turn` 取该方自己的 online 网络，而不是共享 `q_net`，也不是
//   GPU 当前绑定方。CPU 与 GPU 两条路径都要覆盖，并且要专门构造
//   「题目棋色 ≠ 当前 GPU 绑定方」的用例。
//
// 方法：差分法（不改生产代码、不 include .cpp），三个网络**故意不同**：
//   * `online(黑)`：只把 b4[0] 抬高 B_BLACK ⇒ 合法动作 0 永远胜出；
//   * `online(白)`：只把 b4[24] 抬高 B_WHITE ⇒ 合法动作 24 永远胜出；
//   * `q_net`（共享种子）：保持未扰动，与两者都不同。
//   只放大输出层偏置、不碰其它参数，因此 `finite_net/safe_parameters` 不受影响。
//   基线网络 Q 值量级实测约 2e-2（见 build_verify 下的标定输出），0.25 的偏置
//   足以完全支配 argmax ⇒ 各方题面分数就等价于「偏置动作在答案集合里的比例」，
//   从而可以直接判读「评测到底读了哪个网络」。
//
//   判据（每条都要求参考实现与生产实测**逐字段相等**，float 精度）：
//   1) 黑题面（turn=0：empty / 强制先手 / Win1 固定+随机）跟随 online(黑)；
//   2) 白题面（turn=1：Block1 固定/随机，测试内把固定题 turn 改为 1）跟随 online(白)；
//   3) 两者都不跟随共享 q_net（反事实：错用 q_net 的分数必须不同）；
//   4) GPU 路径：同一权重下先绑白方、再绑黑方各跑一次，两次题面分数逐字段相同，
//      且都等于 CPU 参考值 ⇒ 「题目棋色 ≠ 绑定方」时取网依然按题目棋色。
//
// 题面布局实测：`generate_win1_fixed` / `generate_block1_fixed` 都按 player=1 生成
// （skill_test.cpp:229/256 `g.turn = (player == 1) ? 0 : 1`），所以**两套固定题库的
// turn 都是 0**。白题面只能靠 turn=1 的局面进入，因此测试内复制一份 block1 固定题
// 并把 turn 改为 1（只改棋色，不改棋盘）。这只发生在测试进程内。
//
// 说明（不夸大）：题面打分本身走 CPU 前向（`get_q_values` 的 use_gpu 分支目前两条
// 都调 CPU 前向），因此本测试证明的是**取网路由**在 CPU/GPU 两种会话状态下都正确，
// 不是「GPU 数值正确性」验证；GPU 不可用时 GPU 段会在 NEED(gpu_init()==0) 处失败，
// 不会静默降级成 CPU 通过。
#include "core.h"
#include "action_policy.h"
#include "sentinel_guard.h"
#include "independent_networks.h"
#include "skill_test.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

using independent_training::Side;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); ++g_failures; } } while (0)
#define NEED(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); return 1; } } while (0)

/* 刻意放大的输出层偏置：支配 argmax，使三个网络的选择互不相同。 */
static const float B_BLACK = 0.25f;   /* online(黑) 偏好动作 0 */
static const float B_WHITE = 0.25f;   /* online(白) 偏好动作 24 */

extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);

/* ============================================================
   第一部分：独立重算参考值（复刻生产取网规则与 RNG 序列）
   ============================================================ */

/* 用于证明「错法会给出不同分数」的反事实参考值。 */
enum RefMode {
    REF_CORRECT_SIDE = 0,  /* 按题目棋色取该方 online（生产实现） */
    REF_SHARED_QNET = 1,   /* 错误实现：始终用共享 q_net */
    REF_WRONG_SIDE = 2     /* 错误实现：始终用另一方 online */
};

static void ref_q_values_mode(DQNAgent* agent, const GameState* g, float q_out[CELL_COUNT], RefMode mode) {
    float state[CELL_COUNT + 1];
    state_to_input(g, state);
    NeuralNet* net = &agent->q_net;
    if (mode == REF_SHARED_QNET) {
        net = &agent->q_net;
    } else if (agent->indep && agent->indep->active() && g) {
        const int turn = (g->turn == 0 || g->turn == 1) ? g->turn : 0;
        Side s = independent_training::side_from_physical(turn);
        if (mode == REF_WRONG_SIDE) s = independent_training::opposite_side(s);
        NeuralNet* side_net = independent_training::agent_online_net(agent, s);
        if (side_net) net = side_net;
    }
    forward_cpu_only(net, state, q_out);
    memory_apply_model_scores(g, q_out);
}

/* 复刻 skill_test.cpp:68 的 check_win_at（该函数是 static，测试里本地重实现）。 */
static int local_check_win_at(const int board[CELL_COUNT], int pos) {
    const int player = board[pos];
    if (player == 0) return 0;
    const int r = pos / BOARD_SIZE, c = pos % BOARD_SIZE;
    const int dr[] = {0, 1, 1, 1};
    const int dc[] = {1, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int count = 1;
        for (int k = 1; k < WIN_STREAK; k++) {
            const int nr = r + dr[d] * k, nc = c + dc[d] * k;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board[nr * BOARD_SIZE + nc] != player) break;
            count++;
        }
        for (int k = 1; k < WIN_STREAK; k++) {
            const int nr = r - dr[d] * k, nc = c - dc[d] * k;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board[nr * BOARD_SIZE + nc] != player) break;
            count++;
        }
        if (count >= WIN_STREAK) return player;
    }
    return 0;
}

/* 复刻 play_solo（solitaire：落子后**不换手**）。 */
static int ref_play_solo(DQNAgent* agent, GameState g, PRNGState* rng, RefMode mode) {
    const int player = g.turn + 1;
    for (int steps = 1; steps <= CELL_COUNT; ++steps) {
        float q[CELL_COUNT];
        ref_q_values_mode(agent, &g, q, mode);
        const int best = greedy_legal_action(&g, q, rng);
        if (best < 0) return 0;
        g.board[best] = player;
        if (local_check_win_at(g.board, best) == player) return steps;
    }
    return 0;
}

/* 复刻 score_empty_suite（64 快照 × 2 次，RNG 不重置）。 */
static void ref_empty_suite(DQNAgent* agent, float* win_rate, float* avg_steps, RefMode mode) {
    const int SNAPSHOTS = 64, RUNS_PER_SNAP = 2, TOTAL = SNAPSHOTS * RUNS_PER_SNAP;
    GameState empty = {};
    PRNGState rng; prng_init(&rng, 0x534F4C4Fu);
    int total_wins = 0, total_steps = 0;
    for (int s = 0; s < SNAPSHOTS; ++s) {
        for (int r = 0; r < RUNS_PER_SNAP; ++r) {
            const int steps = ref_play_solo(agent, empty, &rng, mode);
            if (steps > 0) { ++total_wins; total_steps += steps; }
        }
    }
    *win_rate = (float)total_wins / TOTAL;
    *avg_steps = total_wins ? (float)total_steps / total_wins : 0.0f;
}

/* 复刻强制先手 25 局（g.board[first] = 1, turn 保持 0）。 */
static void ref_seeded_suite(DQNAgent* agent, float* win_rate, float* avg_steps, RefMode mode) {
    int wins = 0, steps = 0;
    PRNGState rng; prng_init(&rng, 0x46495253u);
    for (int first = 0; first < CELL_COUNT; ++first) {
        GameState g = {};
        g.board[first] = 1;
        const int st = ref_play_solo(agent, g, &rng, mode);
        if (st > 0) { ++wins; steps += st + 1; }  /* seeded_avg_steps 含首子 */
    }
    *win_rate = (float)wins / CELL_COUNT;
    *avg_steps = wins ? (float)steps / wins : 0.0f;
}

/* 复刻「白先分色」套件：白方首着放 25 个位置、turn=1（白方走），与黑先完全对称。
   生产里这段用 seed 0x57484954（"WHIT"）。 */
static void ref_seeded_suite_white(DQNAgent* agent, float* win_rate, float* avg_steps, RefMode mode) {
    int wins = 0, steps = 0;
    PRNGState rng; prng_init(&rng, 0x57484954u);
    for (int first = 0; first < CELL_COUNT; ++first) {
        GameState g = {};
        g.board[first] = 2;   /* 白子 */
        g.turn = 1;           /* 白先 */
        const int st = ref_play_solo(agent, g, &rng, mode);
        if (st > 0) { ++wins; steps += st + 1; }
    }
    *win_rate = (float)wins / CELL_COUNT;
    *avg_steps = wins ? (float)steps / wins : 0.0f;
}

/* 复刻 score_suite（正确数 + margin 和），用于与生产字段精确对照。 */
struct RefSuiteResult { float accuracy = 0; float margin = 0; int n = 0; };

static RefSuiteResult ref_suite_full(DQNAgent* agent, const GameState* puzzles,
    int answers[][CELL_COUNT], const int* na, int count, RefMode mode) {
    RefSuiteResult out;
    out.n = count;
    PRNGState rng; prng_init(&rng, 0x534B494Cu);
    /* 与生产的 score_suite 一样用 float 累加 margin 再除以题数；用 double 会产生
       1e-10 级差异，让「逐字段相等」的判据假失败。 */
    float margin_sum = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        float q[CELL_COUNT];
        ref_q_values_mode(agent, &puzzles[i], q, mode);
        int bad = 0;
        for (int a = 0; a < CELL_COUNT; ++a) if (!puzzles[i].board[a] && !isfinite(q[a])) bad = 1;
        if (bad) continue;  /* 与 test_puzzle 一致：非有限值直接返回且不消耗 RNG */
        float correct_q = -INFINITY, wrong_q = -INFINITY;
        for (int a = 0; a < CELL_COUNT; ++a) {
            if (puzzles[i].board[a]) continue;
            int valid = 0;
            for (int j = 0; j < na[i]; ++j) if (answers[i][j] == a) valid = 1;
            if (valid) correct_q = fmaxf(correct_q, q[a]); else wrong_q = fmaxf(wrong_q, q[a]);
        }
        const int chosen = greedy_legal_action(&puzzles[i], q, &rng);
        if (chosen >= 0) {
            for (int j = 0; j < na[i]; ++j) if (answers[i][j] == chosen) { ++correct; break; }
        }
        if (isfinite(correct_q) && isfinite(wrong_q)) margin_sum += (correct_q - wrong_q);
    }
    out.accuracy = count ? (float)correct / count : 0.0f;
    out.margin = count ? margin_sum / count : 0.0f;
    return out;
}

/* ============================================================
   第二部分：夹具
   ============================================================ */

static std::unique_ptr<DQNAgent> make_agent(NeuralNet* base_out) {
    auto a = std::unique_ptr<DQNAgent>((DQNAgent*)calloc(1, sizeof(DQNAgent)));
    if (!a) return nullptr;
    init_dqn_ex(a.get(), 0);
    g_current_agent = a.get();
    if (!independent_training::indep_ensure(a.get())) return nullptr;
    a->indep->set_active(1);
    a->indep->set_bound(Side::Black);
    /* 只迁移权重：构造时双方都与共享 q_net 同源。 */
    a->indep->seed_both_from(a->q_net, a->q_net);
    if (base_out) *base_out = a->q_net;
    return a;
}

/* 夹具的裁决点：黑偏 0、白偏 24、共享 q_net 偏 12，互不相同。 */
static const int FIX_CELL_BLACK = 0;
static const int FIX_CELL_WHITE = 24;
static const int FIX_CELL_SHARED = 12;

/* 三个网络从同一份 base 出发故意不同：黑偏好 0，白偏好 24，共享 q_net 偏好 12。
   每次都从 base 重算，因此可重复调用（gpu_net_bind_side 会把设备载荷写回旧方影子，
   必须能在每次绑定之后重建同一套夹具）。返回 1 表示空盘上三个 argmax 互不相同。 */
static int apply_differential_fixture_from(DQNAgent* a, const NeuralNet* base_in) {
    /* NeuralNet 单份 ≈3.4 MB：这些中间副本一律堆分配，避免栈溢出
       （测试 EXE 的 /STACK:16777216 也扛不住 5 份同时在栈上）。 */
    std::unique_ptr<NeuralNet> base(new NeuralNet(*base_in));
    std::unique_ptr<NeuralNet> shared(new NeuralNet(*base));
    std::unique_ptr<NeuralNet> black(new NeuralNet(*base));
    std::unique_ptr<NeuralNet> white(new NeuralNet(*base));
    if (!base || !shared || !black || !white) return 0;
    shared->b4[FIX_CELL_SHARED] += 0.5f;
    black->b4[FIX_CELL_BLACK] += B_BLACK;
    white->b4[FIX_CELL_WHITE] += B_WHITE;
    a->indep->online(Side::Black) = *black;
    a->indep->online(Side::White) = *white;
    a->q_net = *shared;

    GameState g = {}; GameState gw = g; gw.turn = 1;
    float qb[CELL_COUNT], qw[CELL_COUNT], qs[CELL_COUNT];
    ref_q_values_mode(a, &g, qb, REF_CORRECT_SIDE);
    ref_q_values_mode(a, &gw, qw, REF_CORRECT_SIDE);
    ref_q_values_mode(a, &g, qs, REF_SHARED_QNET);
    int ab = 0, aw = 0, as = 0;
    for (int k = 0; k < CELL_COUNT; ++k) {
        if (qb[k] > qb[ab]) ab = k;
        if (qw[k] > qw[aw]) aw = k;
        if (qs[k] > qs[as]) as = k;
    }
    return ab == FIX_CELL_BLACK && aw == FIX_CELL_WHITE && as == FIX_CELL_SHARED;
}

struct Metrics {
    float empty_win_rate = 0, empty_avg_steps = 0, empty_shortest_rate = 0;
    float empty_min_steps = 0, empty_max_steps = 0;
    float seeded_win_rate = 0, seeded_avg_steps = 0;
    float win1_fixed = 0, win1_random = 0, block1_fixed = 0, block1_random = 0;
    float win1_fixed_margin = 0, block1_fixed_margin = 0;
    int total_tests = 0, parity_status = 0;
    /* 黑白分色字段 */
    int side_split_valid = 0, black_games = 0, white_games = 0;
    float black_win_rate = 0, black_avg_steps = 0, white_win_rate = 0, white_avg_steps = 0;
    float win1_fixed_black = 0, block1_fixed_black = 0;
};

static Metrics capture(const SkillTestResult& r) {
    Metrics m;
    m.empty_win_rate = r.empty_win_rate;
    m.empty_avg_steps = r.empty_avg_steps;
    m.empty_shortest_rate = r.empty_shortest_rate;
    m.empty_min_steps = r.empty_min_steps;
    m.empty_max_steps = r.empty_max_steps;
    m.seeded_win_rate = r.seeded_win_rate;
    m.seeded_avg_steps = r.seeded_avg_steps;
    m.win1_fixed = r.win1_fixed;
    m.win1_random = r.win1_random;
    m.block1_fixed = r.block1_fixed;
    m.block1_random = r.block1_random;
    m.win1_fixed_margin = r.win1_fixed_margin;
    m.block1_fixed_margin = r.block1_fixed_margin;
    m.total_tests = r.total_tests;
    m.parity_status = r.parity_status;
    m.side_split_valid = r.side_split_valid;
    m.black_games = r.black_games;
    m.white_games = r.white_games;
    m.black_win_rate = r.black_win_rate;
    m.black_avg_steps = r.black_avg_steps;
    m.white_win_rate = r.white_win_rate;
    m.white_avg_steps = r.white_avg_steps;
    m.win1_fixed_black = r.win1_fixed_black;
    m.block1_fixed_black = r.block1_fixed_black;
    return m;
}

static void print_metrics(const char* tag, const Metrics& m) {
    printf("[%s] empty=%.4f/%.4f/%.4f steps[%.1f,%.1f] seeded=%.4f/%.2f "
           "win1f=%.4f m=%.4f win1r=%.4f block1f=%.4f m=%.4f block1r=%.4f total=%d parity=%d\n",
        tag, m.empty_win_rate, m.empty_shortest_rate, m.empty_avg_steps,
        m.empty_min_steps, m.empty_max_steps, m.seeded_win_rate, m.seeded_avg_steps,
        m.win1_fixed, m.win1_fixed_margin, m.win1_random,
        m.block1_fixed, m.block1_fixed_margin, m.block1_random, m.total_tests, m.parity_status);
}

/* 黑题面（turn=0）字段集合：empty / 强制先手 / Win1（固定+随机）。 */
static bool black_suites_equal(const Metrics& a, const Metrics& b) {
    return a.empty_win_rate == b.empty_win_rate &&
           a.empty_avg_steps == b.empty_avg_steps &&
           a.empty_shortest_rate == b.empty_shortest_rate &&
           a.empty_min_steps == b.empty_min_steps &&
           a.empty_max_steps == b.empty_max_steps &&
           a.seeded_win_rate == b.seeded_win_rate &&
           a.seeded_avg_steps == b.seeded_avg_steps &&
           a.win1_fixed == b.win1_fixed &&
           a.win1_fixed_margin == b.win1_fixed_margin &&
           a.win1_random == b.win1_random;
}

/* 白题面（turn=1）字段集合：Block1（固定+随机）。 */
static bool white_suites_equal(const Metrics& a, const Metrics& b) {
    return a.block1_fixed == b.block1_fixed &&
           a.block1_fixed_margin == b.block1_fixed_margin &&
           a.block1_random == b.block1_random;
}

/* 逐字段对照生产实测与参考重算（同一批权重、同一取网规则）。
   注意两套 Block1 题面的用途不同：
     * `block1`（原样，turn=0）对应生产 `result->block1_fixed` 字段——因为固定题库
       按 player=1 生成，字段本身是黑题面；
     * `block1_white`（turn 改为 1）用于单独验证白方路由，生产**没有**对应字段，
       因此只做逐题 `chosen_action` 对照。 */
static bool matches_correct_side(DQNAgent* agent, const GameState* win1, int win1_a[][CELL_COUNT],
    const int* win1_na, int win1_n, const GameState* block1_turn0, const GameState* block1_turn1,
    int b1_a[][CELL_COUNT], const int* b1_na, int b1_n, const Metrics& m) {
    const RefSuiteResult w1 = ref_suite_full(agent, win1, win1_a, win1_na, win1_n, REF_CORRECT_SIDE);
    const RefSuiteResult b0 = ref_suite_full(agent, block1_turn0, b1_a, b1_na, b1_n, REF_CORRECT_SIDE);
    const RefSuiteResult b1 = ref_suite_full(agent, block1_turn1, b1_a, b1_na, b1_n, REF_CORRECT_SIDE);
    printf("[ref] win1(turn0)=%.6f/%.6f block1_field(turn0)=%.6f/%.6f block1_white(turn1)=%.6f/%.6f\n",
        w1.accuracy, w1.margin, b0.accuracy, b0.margin, b1.accuracy, b1.margin);
    return w1.accuracy == m.win1_fixed && w1.margin == m.win1_fixed_margin &&
           b0.accuracy == m.block1_fixed && b0.margin == m.block1_fixed_margin;
}

/* 逐题对照生产 `test_puzzle` 的选择与三种参考取网方式的选择。
   production 与参考各自维护一条**同步消耗**的 RNG 流（同一 seed、同一题序），
   逐题比较等价于比较「最终选择」；这是「评测到底读了哪个网络」的最直接证据。 */
struct DecisionDiff { int n = 0; int mismatch_correct = 0; int mismatch_shared = 0; int mismatch_wrong = 0; };

static DecisionDiff compare_decisions(DQNAgent* agent, const GameState* puzzles,
    int answers[][CELL_COUNT], const int* na, int count) {
    DecisionDiff d;
    d.n = count;
    PRNGState prod_rng; prng_init(&prod_rng, 0x534B494Cu);
    PRNGState ref_rng;  prng_init(&ref_rng, 0x534B494Cu);
    PRNGState shared_rng; prng_init(&shared_rng, 0x534B494Cu);
    PRNGState wrong_rng;  prng_init(&wrong_rng, 0x534B494Cu);
    for (int i = 0; i < count; ++i) {
        const PuzzleResult r = test_puzzle(agent, &puzzles[i], answers[i], na[i], 0, &prod_rng);
        float qc[CELL_COUNT], qs[CELL_COUNT], qw[CELL_COUNT];
        ref_q_values_mode(agent, &puzzles[i], qc, REF_CORRECT_SIDE);
        ref_q_values_mode(agent, &puzzles[i], qs, REF_SHARED_QNET);
        ref_q_values_mode(agent, &puzzles[i], qw, REF_WRONG_SIDE);
        if (greedy_legal_action(&puzzles[i], qc, &ref_rng) != r.chosen_action) ++d.mismatch_correct;
        if (greedy_legal_action(&puzzles[i], qs, &shared_rng) != r.chosen_action) ++d.mismatch_shared;
        if (greedy_legal_action(&puzzles[i], qw, &wrong_rng) != r.chosen_action) ++d.mismatch_wrong;
    }
    return d;
}

int main() {
    g_training_fault = 0;
    g_learning_rate = 0.0005f;
    sentinel_guard_init();

    /* fix_base 是 ~3.4 MB 的 NeuralNet：堆分配，避免主栈上再叠一份。 */
    std::unique_ptr<NeuralNet> fix_base(new NeuralNet());
    auto agent = make_agent(fix_base.get());
    NEED(agent != nullptr);

    /* ---- 题库布局实测与改写 ---- */
    GameState win1[512], block1[512], block1_white[512];
    int win1_a[512][CELL_COUNT], block1_a[512][CELL_COUNT];
    int win1_na[512], block1_na[512];
    int win1_n = 0, block1_n = 0;
    generate_win1_fixed(win1, win1_a, win1_na, &win1_n, 512);
    generate_block1_fixed(block1, block1_a, block1_na, &block1_n, 512);
    NEED(win1_n > 0 && block1_n > 0);
    for (int i = 0; i < block1_n; ++i) { block1_white[i] = block1[i]; block1_white[i].turn = 1; }
    {
        GameState empty = {};
        CHECK(empty.turn == 0);
        int bad_win1 = 0, bad_block1 = 0;
        for (int i = 0; i < win1_n; ++i) if (win1[i].turn != 0) ++bad_win1;
        for (int i = 0; i < block1_n; ++i) if (block1[i].turn != 0) ++bad_block1;
        printf("[turn] win1_fixed n=%d turn!=0: %d ; block1_fixed n=%d turn!=0: %d ; "
               "local white-turn copy n=%d\n", win1_n, bad_win1, block1_n, bad_block1, block1_n);
        NEED(bad_win1 == 0);
        NEED(bad_block1 == 0);
    }

    /* ============ 差分夹具：三个网络故意不同 ============ */
    {
        const int distinct = apply_differential_fixture_from(agent.get(), fix_base.get());
        printf("[fixture] three distinct argmax (black=0 white=24 shared=12): %d\n", distinct);
        CHECK(distinct == 1);
    }
    {
        /* 偏置确实支配了 argmax：动作 0 / 24 在合法位置时必然成为各方的首选。 */
        GameState g = {};
        GameState gw = g; gw.turn = 1;
        float qb[CELL_COUNT], qw[CELL_COUNT], qs[CELL_COUNT];
        ref_q_values_mode(agent.get(), &g, qb, REF_CORRECT_SIDE);   /* turn=0 → 黑 */
        ref_q_values_mode(agent.get(), &gw, qw, REF_CORRECT_SIDE);  /* turn=1 → 白 */
        ref_q_values_mode(agent.get(), &g, qs, REF_SHARED_QNET);
        int argmax_black = 0, argmax_white = 0, argmax_shared = 0;
        for (int a = 0; a < CELL_COUNT; ++a) {
            if (qb[a] > qb[argmax_black]) argmax_black = a;
            if (qw[a] > qw[argmax_white]) argmax_white = a;
            if (qs[a] > qs[argmax_shared]) argmax_shared = a;
        }
        printf("[fixture] empty-board argmax: online(black)=%d online(white)=%d q_net=%d\n",
            argmax_black, argmax_white, argmax_shared);
        CHECK(argmax_black == FIX_CELL_BLACK);
        CHECK(argmax_white == FIX_CELL_WHITE);
        CHECK(argmax_shared == FIX_CELL_SHARED);
        /* 三方逐字节不同。 */
        CHECK(memcmp(&agent->indep->online(Side::Black), &agent->q_net, sizeof(NeuralNet)) != 0);
        CHECK(memcmp(&agent->indep->online(Side::White), &agent->q_net, sizeof(NeuralNet)) != 0);
        CHECK(memcmp(&agent->indep->online(Side::Black),
                     &agent->indep->online(Side::White), sizeof(NeuralNet)) != 0);
    }

    /* ============ CPU 路径 ============ */
    SkillTestResult res{};
    run_skill_tests(agent.get(), 4901, &res, 0);
    Metrics m = capture(res);
    print_metrics("cpu/differential", m);
    NEED(g_training_fault == 0);

    {
        const RefSuiteResult w1c = ref_suite_full(agent.get(), win1, win1_a, win1_na, win1_n, REF_CORRECT_SIDE);
        const RefSuiteResult b1c = ref_suite_full(agent.get(), block1_white, block1_a, block1_na, block1_n, REF_CORRECT_SIDE);
        const RefSuiteResult w1s = ref_suite_full(agent.get(), win1, win1_a, win1_na, win1_n, REF_SHARED_QNET);
        const RefSuiteResult b1s = ref_suite_full(agent.get(), block1_white, block1_a, block1_na, block1_n, REF_SHARED_QNET);
        const RefSuiteResult w1x = ref_suite_full(agent.get(), win1, win1_a, win1_na, win1_n, REF_WRONG_SIDE);
        const RefSuiteResult b1x = ref_suite_full(agent.get(), block1_white, block1_a, block1_na, block1_n, REF_WRONG_SIDE);
        /* 生产 `block1_fixed` 字段是**黑题面**（固定题库 turn=0），白题面路由由
           逐题对照覆盖，所以这里的字段对照用 block1（turn0）+ block1_white（turn1）
           两个参考值，但只与 turn0 的字段值比较。 */
        const RefSuiteResult b0c = ref_suite_full(agent.get(), block1, block1_a, block1_na, block1_n, REF_CORRECT_SIDE);
        printf("[cpu/black-turn win1_fixed n=%d] production=%.6f/%.6f | correct(black)=%.6f/%.6f "
               "| shared q_net=%.6f/%.6f | wrong side(white)=%.6f/%.6f\n",
            win1_n, m.win1_fixed, m.win1_fixed_margin, w1c.accuracy, w1c.margin,
            w1s.accuracy, w1s.margin, w1x.accuracy, w1x.margin);
        printf("[cpu/block1 field (turn=0) n=%d] production=%.6f/%.6f | correct(black)=%.6f/%.6f\n",
            block1_n, m.block1_fixed, m.block1_fixed_margin, b0c.accuracy, b0c.margin);
        printf("[cpu/white-turn block1 copy n=%d] correct(white)=%.6f/%.6f | shared q_net=%.6f/%.6f "
               "| wrong side(black)=%.6f/%.6f (生产无对应字段，仅作路由判别)\n",
            block1_n, b1c.accuracy, b1c.margin, b1s.accuracy, b1s.margin, b1x.accuracy, b1x.margin);

        /* 1) 生产实测 == 按题目棋色取该方 online 的重算值（逐字段）。 */
        CHECK(w1c.accuracy == m.win1_fixed && w1c.margin == m.win1_fixed_margin);
        CHECK(b0c.accuracy == m.block1_fixed && b0c.margin == m.block1_fixed_margin);
        /* 2) 反事实：错用共享 q_net / 错用另一方都会给出**不同**的分数。 */
        CHECK(w1s.accuracy != w1c.accuracy);
        CHECK(w1x.accuracy != w1c.accuracy);
        CHECK(b1s.accuracy != b1c.accuracy);
        CHECK(b1x.accuracy != b1c.accuracy);
    }

    /* 逐题证据：生产的每一次选择都等于「按题目棋色取该方 online」的选择，
       而与共享 q_net / 另一方网络的选择存在大量不一致（判别力证明）。 */
    {
        const DecisionDiff dw = compare_decisions(agent.get(), win1, win1_a, win1_na, win1_n);
        const DecisionDiff db = compare_decisions(agent.get(), block1_white, block1_a, block1_na, block1_n);
        printf("[cpu/per-puzzle black-turn n=%d] mismatch vs correct(black)=%d vs shared_q_net=%d vs wrong side=%d\n",
            dw.n, dw.mismatch_correct, dw.mismatch_shared, dw.mismatch_wrong);
        printf("[cpu/per-puzzle white-turn n=%d] mismatch vs correct(white)=%d vs shared_q_net=%d vs wrong side=%d\n",
            db.n, db.mismatch_correct, db.mismatch_shared, db.mismatch_wrong);
        CHECK(dw.mismatch_correct == 0);
        CHECK(db.mismatch_correct == 0);
        CHECK(dw.mismatch_shared > 0);
        CHECK(db.mismatch_shared > 0);
        CHECK(dw.mismatch_wrong > 0);
        CHECK(db.mismatch_wrong > 0);
    }

    /* 黑白分色字段：套件必须按物理棋色分别给出胜率，且黑先字段沿用原来的
       seeded 语义（对称构造）。这里用与生产同一套权重逐字段对照。 */
    {
        float bw = 0, bs = 0, ww = 0, ws = 0;
        ref_seeded_suite(agent.get(), &bw, &bs, REF_CORRECT_SIDE);
        ref_seeded_suite_white(agent.get(), &ww, &ws, REF_CORRECT_SIDE);
        const RefSuiteResult w1c = ref_suite_full(agent.get(), win1, win1_a, win1_na, win1_n, REF_CORRECT_SIDE);
        const RefSuiteResult b0c = ref_suite_full(agent.get(), block1, block1_a, block1_na, block1_n, REF_CORRECT_SIDE);
        printf("[cpu/side-split] valid=%d black_to_move %.6f/%.6f (ref %.6f/%.6f, n=%d) | "
               "white_to_move %.6f/%.6f (ref %.6f/%.6f, n=%d) | fixed-black win1 %.6f (ref %.6f) "
               "block1 %.6f (ref %.6f)\n",
            m.side_split_valid, m.black_win_rate, m.black_avg_steps, bw, bs, m.black_games,
            m.white_win_rate, m.white_avg_steps, ww, ws, m.white_games,
            m.win1_fixed_black, w1c.accuracy, m.block1_fixed_black, b0c.accuracy);
        CHECK(m.side_split_valid == 1);
        CHECK(m.black_win_rate == bw);
        CHECK(m.black_avg_steps == bs);
        CHECK(m.white_win_rate == ww);
        CHECK(m.white_avg_steps == ws);
        CHECK(m.black_games == CELL_COUNT && m.white_games == CELL_COUNT);
        CHECK(m.black_win_rate == m.seeded_win_rate);
        CHECK(m.win1_fixed_black == m.win1_fixed);
        CHECK(m.block1_fixed_black == m.block1_fixed);
        /* 分色必须真的分得开：两边权重不同 ⇒ 两色胜率/平均步数不应同时相同。 */
        printf("[cpu/side-split/distinct] black %.4f/%.3f vs white %.4f/%.3f (shared q_net ref %.4f/%.3f)\n",
            m.black_win_rate, m.black_avg_steps, m.white_win_rate, m.white_avg_steps, bw, bs);
        {
            float wws = 0, wss = 0, wwx = 0, wsx = 0;
            ref_seeded_suite_white(agent.get(), &wws, &wss, REF_SHARED_QNET);
            ref_seeded_suite_white(agent.get(), &wwx, &wsx, REF_WRONG_SIDE);
            printf("[cpu/side-split/counterfactual] white correct=%.4f/%.3f shared_q_net=%.4f/%.3f "
                   "wrong_side=%.4f/%.3f\n", m.white_win_rate, m.white_avg_steps, wws, wss, wwx, wsx);
            /* 白先 solitaire 在本 fixture 下两种网络都能赢（胜率饱和在 100%），
               所以判据只能是「平均步数不同」——它仍然证明取的是不同的网络；
               不要把它当成质量差异来解读。 */
            CHECK(wws != m.white_win_rate || wss != m.white_avg_steps);
        }
    }

    printf("PASS cpu routing: evaluation follows the puzzle's own colour, not q_net and not the bound side\n");

    /* ============================================================
       第二部分补：激活载荷必须等于**绑定方自己的** indep 权重
       ============================================================ */
    {
        NEED(gpu_init() == 0);
        /* 刻意让共享 q_net/target_net 与绑定方（黑）的 indep 权重**不同代**：
           若激活仍上传共享 q_net，设备载荷就会是这份「外来」权重，下一次
           gpu_net_bind_side 切到另一方时 download_side_locked(旧方) 会把它写进
           旧方的 CPU 影子，静默把一方的权重替换成共享网络的副本。 */
        NEED(apply_differential_fixture_from(agent.get(), fix_base.get()) == 1);
        agent.get()->indep->set_bound(Side::Black);
        agent.get()->target_net = agent.get()->indep->target(Side::White);
        /* q_net 保持在 FIX_CELL_SHARED 那份，与黑方 online 不同代。 */
        NEED(memcmp(&agent.get()->q_net, &agent.get()->indep->online(Side::Black),
            sizeof(NeuralNet)) != 0);

        NEED(gpu_net_activate(agent.get()) == 0);
        {
            std::unique_ptr<NeuralNet> dev_online(new NeuralNet());
            std::unique_ptr<NeuralNet> dev_target(new NeuralNet());
            NEED(gpu_net_snapshot_side_pair(agent.get(), 0, dev_online.get(), dev_target.get()) == 0);
            const int online_is_black =
                memcmp(dev_online.get(), &agent.get()->indep->online(Side::Black), sizeof(NeuralNet)) == 0;
            const int target_is_black =
                memcmp(dev_target.get(), &agent.get()->indep->target(Side::Black), sizeof(NeuralNet)) == 0;
            const int online_is_shared = memcmp(dev_online.get(), &agent.get()->q_net, sizeof(NeuralNet)) == 0;
            printf("[gpu-activate-payload] bound=black : device online == black online : %d ; "
                   "device target == black target : %d ; device online == shared q_net : %d\n",
                online_is_black, target_is_black, online_is_shared);
            CHECK(online_is_black);
            CHECK(target_is_black);
            CHECK(!online_is_shared);
        }
        gpu_net_deactivate();
    }

    /* ============================================================
       第三部分：GPU 路径 —— 绑定方切换不得改变题面取网

       实测坑：`gpu_net_bind_side` 会把绑定方的**设备参数回写**到 CPU 影子
       （`download_side_locked`），绑定因此会抹掉刚施加的差分夹具。所以每次绑定之后
       必须重新施加夹具再跑评测；这一步本身也是「绑定方切换不改题面取网」的前置条件
       （否则三方又变回同一份权重）。
       ============================================================ */
    {
        NEED(gpu_net_activate(agent.get()) == 0);

        /* 绑定方 = 白；题目棋色黑/白混合 ⇒ 必然存在「题目棋色(黑) ≠ 绑定方(白)」。 */
        NEED(gpu_net_bind_side(agent.get(), 1) == 0);
        NEED(apply_differential_fixture_from(agent.get(), fix_base.get()) == 1);
        {
            const DecisionDiff pre = compare_decisions(agent.get(), win1, win1_a, win1_na, win1_n);
            printf("[gpu/bound=white precheck] black-turn per-puzzle mismatch vs correct=%d vs shared=%d\n",
                pre.mismatch_correct, pre.mismatch_shared);
            CHECK(pre.mismatch_correct == 0);
            CHECK(pre.mismatch_shared > 0);
        }
        SkillTestResult rw{};
        run_skill_tests(agent.get(), 4901, &rw, 1);
        Metrics mw = capture(rw);
        print_metrics("gpu/bound=white", mw);
        NEED(g_training_fault == 0);
        NEED(mw.parity_status == 1);

        /* 改绑黑方；题目棋色 = 白 ⇒ 存在「题目棋色(白) ≠ 绑定方(黑)」。 */
        NEED(gpu_net_bind_side(agent.get(), 0) == 0);
        NEED(apply_differential_fixture_from(agent.get(), fix_base.get()) == 1);
        {
            const DecisionDiff pre = compare_decisions(agent.get(), block1_white, block1_a, block1_na, block1_n);
            printf("[gpu/bound=black precheck] white-turn per-puzzle mismatch vs correct=%d vs shared=%d\n",
                pre.mismatch_correct, pre.mismatch_shared);
            CHECK(pre.mismatch_correct == 0);
            CHECK(pre.mismatch_shared > 0);
        }
        SkillTestResult rb{};
        run_skill_tests(agent.get(), 4901, &rb, 1);
        Metrics mb = capture(rb);
        print_metrics("gpu/bound=black", mb);
        NEED(g_training_fault == 0);
        NEED(mb.parity_status == 1);

        printf("[gpu] bound white vs bound black: black-turn identical=%d white-turn identical=%d\n",
            (int)black_suites_equal(mw, mb), (int)white_suites_equal(mw, mb));
        CHECK(black_suites_equal(mw, mb));
        CHECK(white_suites_equal(mw, mb));
        CHECK(mw.total_tests == mb.total_tests);
        printf("[gpu-vs-cpu] bound-white vs cpu: black identical=%d white identical=%d\n",
            (int)black_suites_equal(mw, m), (int)white_suites_equal(mw, m));
        printf("[gpu-vs-cpu] bound-black vs cpu: black identical=%d white identical=%d\n",
            (int)black_suites_equal(mb, m), (int)white_suites_equal(mb, m));
        CHECK(black_suites_equal(mw, m));
        CHECK(white_suites_equal(mw, m));
        CHECK(black_suites_equal(mb, m));
        CHECK(white_suites_equal(mb, m));

        /* GPU 段也必须与「按题目棋色重算」逐字段一致，并在逐题层面与 q_net 反事实区分。 */
        CHECK(matches_correct_side(agent.get(), win1, win1_a, win1_na, win1_n,
            block1, block1_white, block1_a, block1_na, block1_n, mw));
        CHECK(matches_correct_side(agent.get(), win1, win1_a, win1_na, win1_n,
            block1, block1_white, block1_a, block1_na, block1_n, mb));
        {
            const DecisionDiff dw = compare_decisions(agent.get(), win1, win1_a, win1_na, win1_n);
            const DecisionDiff db = compare_decisions(agent.get(), block1_white, block1_a, block1_na, block1_n);
            printf("[gpu-per-puzzle black-turn n=%d] mismatch vs correct=%d vs shared=%d vs wrong side=%d\n",
                dw.n, dw.mismatch_correct, dw.mismatch_shared, dw.mismatch_wrong);
            printf("[gpu-per-puzzle white-turn n=%d] mismatch vs correct=%d vs shared=%d vs wrong side=%d\n",
                db.n, db.mismatch_correct, db.mismatch_shared, db.mismatch_wrong);
            CHECK(dw.mismatch_correct == 0);
            CHECK(db.mismatch_correct == 0);
            CHECK(dw.mismatch_shared > 0);
            CHECK(db.mismatch_shared > 0);
        }

        gpu_net_deactivate();
        gpu_shutdown();
    }

    free_dqn(agent.get());
    agent.reset();
    g_current_agent = nullptr;

    printf("\n%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

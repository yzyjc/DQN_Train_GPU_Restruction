/* skill_test.cpp — 技能回归测试系统
   - Win-in-1 / Block-in-1 题库生成（28 windows × 4 gaps = 112 基础题）
   - 答案验证：valid_answer_set（支持多正确答案）
   - EmptyWin 扰动套件（非确定性重复）
   - Q-margin early warning
   - CPU/GPU parity test
   - 固定回归集 + 随机压力集 */
#include "skill_test.h"
#include "action_policy.h"
#include "skill_metric_writer.h"
#include "stage_progress.h"
#include "independent_networks.h"   /* 按方路由：技能评测必须用题目对应方的网络 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 测试进度（0-100），由 run_skill_tests 各阶段更新，外部可读。
   保留供旧调用方（PG 的旧路径）使用；新调用方改用 StageProgressCb。 */
volatile int g_skill_test_progress = 0;
volatile int g_skill_test_phase = 0;   /* 0=idle, 1=empty, 2=seeded, 3=win1, 4=block1 */

/* 阶段名称映射（与 StageId 一一对应） */
const char* stage_progress_name(int stage) {
    switch (stage) {
    case STG_EVAL:         return "评估";
    case STG_POST_EVAL:    return "评估后处理";
    case STG_GPU_SYNC:     return "GPU 同步";
    case STG_SKILL_EMPTY:  return "技能测试 · EmptyWin";
    case STG_SKILL_SEEDED: return "技能测试 · Seeded";
    case STG_SKILL_WIN1:   return "技能测试 · Win1";
    case STG_SKILL_BLOCK1: return "技能测试 · Block1";
    case STG_MULTI_EMPTY:  return "多快照 · EmptyWin";
    case STG_MULTI_WIN1:   return "多快照 · Win1";
    case STG_MULTI_BLOCK1: return "多快照 · Block1";
    case STG_SAVE:         return "保存检查点";
    default:               return "处理中";
    }
}

const char* stage_progress_unit(int unit) {
    switch (unit) {
    case UNIT_GAMES:    return "局";
    case UNIT_PUZZLES:  return "题";
    case UNIT_SNAPSHOTS:return "快照";
    default:            return "";
    }
}

/* 填充 sp 并回传一次阶段进度（无分配、可高频调用）。 */
static void emit_stage(StageProgressCb cb, void* user, StageProgress* sp,
    int stage, int completed, int total, int unit, int snap_done, int snap_total) {
    if (!cb) return;
    sp->stage = stage;
    sp->completed = completed;
    sp->total = total;
    sp->unit = unit;
    sp->snapshot_completed = snap_done;
    sp->snapshot_total = snap_total;
    sp->indeterminate = 0;
    sp->state = TASK_RUNNING;
    cb(sp, user);
}

/* ========== 内部工具 ========== */

/* 判断 board[pos] 是否形成四连（返回 1=X赢, 2=O赢, 0=无） */
static int check_win_at(const int board[CELL_COUNT], int pos) {
    int player = board[pos];
    if (player == 0) return 0;
    int r = pos / BOARD_SIZE, c = pos % BOARD_SIZE;
    /* 4 个方向：横、竖、正斜、反斜 */
    int dr[] = {0, 1, 1, 1};
    int dc[] = {1, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int count = 1;
        /* 正方向 */
        for (int k = 1; k < WIN_STREAK; k++) {
            int nr = r + dr[d] * k, nc = c + dc[d] * k;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board[nr * BOARD_SIZE + nc] != player) break;
            count++;
        }
        /* 反方向 */
        for (int k = 1; k < WIN_STREAK; k++) {
            int nr = r - dr[d] * k, nc = c - dc[d] * k;
            if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
            if (board[nr * BOARD_SIZE + nc] != player) break;
            count++;
        }
        if (count >= WIN_STREAK) return player;
    }
    return 0;
}

/* 扫描整个棋盘是否有赢家 */
static int check_winner_full(const int board[CELL_COUNT]) {
    for (int i = 0; i < CELL_COUNT; i++) {
        if (board[i] != 0) {
            int w = check_win_at(board, i);
            if (w) return w;
        }
    }
    return 0;
}

/* 收集所有一步四连的位置（player 在空位落子后形成四连） */
static int collect_win_moves(const GameState* g, int player, int out[CELL_COUNT]) {
    int count = 0;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (g->board[i] != 0) continue;
        GameState tmp = *g;
        tmp.board[i] = player;
        if (check_win_at(tmp.board, i) == player) {
            out[count++] = i;
        }
    }
    return count;
}

/* 收集所有能堵住对手一步四连的位置 */
static int collect_block_moves(const GameState* g, int blocker, int out[CELL_COUNT]) {
    int opponent = (blocker == 1) ? 2 : 1;
    int threats[CELL_COUNT];
    if (check_winner_full(g->board) || collect_win_moves(g, opponent, threats) == 0) return 0;
    int count = 0;
    for (int a = 0; a < CELL_COUNT; ++a) {
        if (g->board[a]) continue;
        GameState next = *g;
        next.board[a] = blocker;
        // Winning immediately also prevents the opponent's next turn.
        if (check_win_at(next.board, a) == blocker ||
            collect_win_moves(&next, opponent, threats) == 0) out[count++] = a;
    }
    return count;
}

/* 判断局面是否合法（没有双方同时四连等异常） */
static int is_legal_position(const GameState* g) {
    int x_win = 0, o_win = 0, x_count = 0, o_count = 0;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (g->board[i] == 1) { ++x_count; if (check_win_at(g->board, i)) x_win = 1; }
        if (g->board[i] == 2) { ++o_count; if (check_win_at(g->board, i)) o_win = 1; }
    }
    if (x_win || o_win) return 0;
    return g->turn == 0 ? x_count == o_count : x_count == o_count + 1;
}

static int add_piece_preserving_answer(GameState* g, int piece, int type, int player,
    const int answers[CELL_COUNT], int answer_count, PRNGState* rng) {
    for (int attempt = 0; attempt < CELL_COUNT * 8; ++attempt) {
        const int pos = rng ? prng_int(rng, CELL_COUNT) : attempt % CELL_COUNT;
        if (g->board[pos]) continue;
        g->board[pos] = piece;
        int x_win = 0, o_win = 0;
        for (int i = 0; i < CELL_COUNT; ++i) {
            if (g->board[i] == 1 && check_win_at(g->board, i)) x_win = 1;
            if (g->board[i] == 2 && check_win_at(g->board, i)) o_win = 1;
        }
        int new_answers[CELL_COUNT];
        const int new_count = type == 0 ? collect_win_moves(g, player, new_answers) :
            collect_block_moves(g, player, new_answers);
        if (!x_win && !o_win && new_count == answer_count &&
            memcmp(new_answers, answers, sizeof(int) * (size_t)answer_count) == 0) return 1;
        g->board[pos] = 0;
    }
    return 0;
}

/* Complete the sparse tactical template to a position reachable from normal
   alternating play.  This prevents a high Block1 score on impossible boards
   from being mistaken for real defensive skill. */
static int make_reachable_tactical_position(GameState* g, int type, int player,
    const int answers[CELL_COUNT], int answer_count, PRNGState* rng) {
    int x = 0, o = 0;
    for (int i = 0; i < CELL_COUNT; ++i) { x += g->board[i] == 1; o += g->board[i] == 2; }
    while ((g->turn == 0 && x != o) || (g->turn == 1 && x != o + 1)) {
        int piece;
        if (g->turn == 0) piece = x < o ? 1 : 2;
        else piece = x <= o ? 1 : 2;
        if (!add_piece_preserving_answer(g, piece, type, player, answers, answer_count, rng)) return 0;
        if (piece == 1) ++x; else ++o;
    }
    /* Add 0..2 balanced pairs for a denser held-out distribution. */
    const int pairs = rng ? prng_int(rng, 3) : 0;
    for (int pair = 0; pair < pairs; ++pair) {
        GameState before = *g;
        const int first = g->turn == 0 ? 1 : 2;
        const int second = 3 - first;
        if (!add_piece_preserving_answer(g, first, type, player, answers, answer_count, rng) ||
            !add_piece_preserving_answer(g, second, type, player, answers, answer_count, rng)) {
            *g = before;
            break;
        }
    }
    return is_legal_position(g);
}

/* ========== 题库生成 ========== */

/* 检查一个 4 格窗口是否恰好有 exactly n 个 player 棋子且无对手棋子 */
static int window_has_pieces(const int board[CELL_COUNT], int cells[4], int player, int n) {
    int opponent = (player == 1) ? 2 : 1;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (board[cells[i]] == opponent) return 0;
        if (board[cells[i]] == player) count++;
    }
    return (count == n);
}

/* 生成 Win-in-1 题目：在给定窗口中，player 有 3 颗棋子，1 个空位 */
static int try_generate_win1(const int board_template[CELL_COUNT], int cells[4], int player,
                             GameState* out, int answers[CELL_COUNT], int* num_answers) {
    int gap = -1;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (board_template[cells[i]] == player) count++;
        else if (board_template[cells[i]] == 0) gap = i;
    }
    if (count != 3 || gap < 0) return 0;

    /* 构造题目：清空棋盘，只放这 3 颗棋子 */
    GameState g;
    memset(&g, 0, sizeof(g));
    for (int i = 0; i < 4; i++) {
        if (i != gap) g.board[cells[i]] = player;
    }
    g.turn = (player == 1) ? 0 : 1;

    /* 扫描所有合法答案（可能不止一个四连点） */
    *num_answers = collect_win_moves(&g, player, answers);
    if (*num_answers == 0) return 0;

    *out = g;
    return 1;
}

/* 生成 Block-in-1 题目：对手有 3 颗棋子在窗口中，player 必须堵 */
static int try_generate_block1(const int board_template[CELL_COUNT], int cells[4], int player,
                               GameState* out, int answers[CELL_COUNT], int* num_answers) {
    int opponent = (player == 1) ? 2 : 1;
    int gap = -1;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (board_template[cells[i]] == opponent) count++;
        else if (board_template[cells[i]] == 0) gap = i;
    }
    if (count != 3 || gap < 0) return 0;

    GameState g;
    memset(&g, 0, sizeof(g));
    for (int i = 0; i < 4; i++) {
        if (i != gap) g.board[cells[i]] = opponent;
    }
    g.turn = (player == 1) ? 0 : 1;

    /* 收集所有堵住对手四连的位置 */
    *num_answers = collect_block_moves(&g, player, answers);
    if (*num_answers == 0) return 0;

    *out = g;
    return 1;
}

/* 枚举 5×5 棋盘所有 28 个 winning windows */
static void enumerate_windows(int windows[][4], int* count) {
    *count = 0;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c <= BOARD_SIZE - WIN_STREAK; c++) {
            /* 横向 */
            for (int k = 0; k < WIN_STREAK; k++)
                windows[*count][k] = r * BOARD_SIZE + c + k;
            (*count)++;
        }
    }
    for (int c = 0; c < BOARD_SIZE; c++) {
        for (int r = 0; r <= BOARD_SIZE - WIN_STREAK; r++) {
            /* 纵向 */
            for (int k = 0; k < WIN_STREAK; k++)
                windows[*count][k] = (r + k) * BOARD_SIZE + c;
            (*count)++;
        }
    }
    for (int r = 0; r <= BOARD_SIZE - WIN_STREAK; r++) {
        for (int c = 0; c <= BOARD_SIZE - WIN_STREAK; c++) {
            /* \ 对角线 */
            for (int k = 0; k < WIN_STREAK; k++)
                windows[*count][k] = (r + k) * BOARD_SIZE + c + k;
            (*count)++;
        }
    }
    for (int r = 0; r <= BOARD_SIZE - WIN_STREAK; r++) {
        for (int c = WIN_STREAK - 1; c < BOARD_SIZE; c++) {
            /* / 对角线 */
            for (int k = 0; k < WIN_STREAK; k++)
                windows[*count][k] = (r + k) * BOARD_SIZE + c - k;
            (*count)++;
        }
    }
}

/* 从窗口生成题目的核心函数 */
static int generate_puzzles_from_windows(
    GameState puzzles[], int answers[][CELL_COUNT], int num_answers[],
    int* count, int max, int player, int type /* 0=win1, 1=block1 */,
    const int board_template[CELL_COUNT], int add_interference, PRNGState* rng)
{
    int windows[56][4]; /* 28 windows max, but allocate more for safety */
    int num_windows;
    enumerate_windows(windows, &num_windows);

    for (int w = 0; w < num_windows && *count < max; w++) {
        for (int gap = 0; gap < 4 && *count < max; gap++) {
            /* 在窗口中放棋子，gap 位置留空 */
            int cells[4];
            for (int i = 0; i < 4; i++) cells[i] = windows[w][i];

            /* 构造模板棋盘：3 颗棋子 + gap */
            int template_board[CELL_COUNT];
            memcpy(template_board, board_template, sizeof(int) * CELL_COUNT);

            int target_player = (type == 0) ? player : ((player == 1) ? 2 : 1);
            /* 确保 cells 位置没有冲突 */
            int valid = 1;
            for (int i = 0; i < 4; i++) {
                if (i == gap) {
                    if (template_board[cells[i]] != 0) { valid = 0; break; }
                } else {
                    if (template_board[cells[i]] != 0 && template_board[cells[i]] != target_player) {
                        valid = 0; break;
                    }
                    template_board[cells[i]] = target_player;
                }
            }
            if (!valid) continue;

            /* 验证 gap 位置是空的 */
            if (template_board[cells[gap]] != 0) continue;

            GameState g;
            int ans[CELL_COUNT];
            int na = 0;
            int ok = 0;

            if (type == 0) {
                ok = try_generate_win1(template_board, cells, player, &g, ans, &na);
            } else {
                ok = try_generate_block1(template_board, cells, player, &g, ans, &na);
            }

            if (!ok || na == 0) continue;

            /* 添加合法干扰棋子，并补齐到真实交替落子可达的子数关系。 */
            if (add_interference && rng) {
                if (!make_reachable_tactical_position(&g, type, player, ans, na, rng)) continue;
            }

            /* 最终合法性检查 */
            if (!is_legal_position(&g)) continue;

            int duplicate = 0;
            for (int i = 0; i < *count; ++i)
                if (puzzles[i].turn == g.turn &&
                    memcmp(puzzles[i].board, g.board, sizeof(g.board)) == 0) duplicate = 1;
            if (duplicate) continue;

            puzzles[*count] = g;
            memcpy(answers[*count], ans, sizeof(int) * na);
            num_answers[*count] = na;
            (*count)++;
        }
    }
    return *count;
}

void generate_side_skill_puzzles(int side,int skill,uint32_t seed,int heldout,
    GameState puzzles[],int answers[][CELL_COUNT],int num_answers[],int* count,int max) {
    *count=0;if(side<0 || side>1 || skill<1 || skill>2 || max<=0)return;
    PRNGState rng;prng_init(&rng,seed);int blank[CELL_COUNT]={};
    GameState candidates[128];int ans[128][CELL_COUNT],na[128];
    for(int round=0;round<32 && *count<max;++round){
        int n=0;generate_puzzles_from_windows(candidates,ans,na,&n,128,side+1,skill-1,blank,1,&rng);
        for(int i=0;i<n && *count<max;++i){
            uint32_t hash=2166136261u;
            for(int v:candidates[i].board)hash=(hash^(uint32_t)v)*16777619u;
            hash=(hash^(uint32_t)side)*16777619u;
            if((hash%4==0)!=(heldout!=0))continue;
            bool duplicate=false;for(int j=0;j<*count;++j)if(!memcmp(puzzles[j].board,candidates[i].board,sizeof(candidates[i].board)))duplicate=true;
            if(duplicate)continue;
            puzzles[*count]=candidates[i];num_answers[*count]=na[i];
            memcpy(answers[*count],ans[i],na[i]*sizeof(int));++*count;
        }
    }
}

/* ========== 公共题库生成 API ==========

   固定 seed 的回归集：
   - Win1: player=X(1), 空盘模板, 加干扰
   - Block1: player=X(1), 空盘模板, 加干扰

   随机压力集：
   - 每次用不同 seed 的 rng 生成 */

void generate_win1_fixed(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max) {
    int board[CELL_COUNT] = {0};
    PRNGState rng;
    prng_init(&rng, 12345u); /* 固定 seed */
    *count = 0;
    generate_puzzles_from_windows(puzzles, answers, num_answers, count, max, 1, 0, board, 1, &rng);
}

void generate_block1_fixed(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max) {
    int board[CELL_COUNT] = {0};
    PRNGState rng;
    prng_init(&rng, 12345u);
    *count = 0;
    generate_puzzles_from_windows(puzzles, answers, num_answers, count, max, 1, 1, board, 1, &rng);
}

void generate_win1_random(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max, PRNGState* rng) {
    int board[CELL_COUNT] = {0};
    *count = 0;
    generate_puzzles_from_windows(puzzles, answers, num_answers, count, max, 1, 0, board, 1, rng);
}

void generate_block1_random(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max, PRNGState* rng) {
    int board[CELL_COUNT] = {0};
    *count = 0;
    generate_puzzles_from_windows(puzzles, answers, num_answers, count, max, 1, 1, board, 1, rng);
}

/* ========== 单题测试 ========== */

/* 从 agent 获取 Q 值（使用和实际对局相同的推理路径）
   按方路由开启时必须用**题目对应方**的网络：共享 q_net 从播种那一刻起就不再更新，
   继续读它等于拿预训练种子给当前模型打分，成绩/退化报警/健康快照都会失真。 */
static void get_q_values(DQNAgent* agent, const GameState* g, float q_out[CELL_COUNT], int use_gpu) {
    float state[CELL_COUNT + 1];
    state_to_input(g, state);

    NeuralNet* net = &agent->q_net;
    if (agent->indep && agent->indep->active() && g) {
        const int turn = (g->turn == 0 || g->turn == 1) ? g->turn : 0;
        NeuralNet* side_net = independent_training::agent_online_net(agent,
            independent_training::side_from_physical(turn));
        if (side_net) net = side_net;
    }

    if (use_gpu) {
        /* GPU 推理（与实际对局一致） */
        /* 注意：这里需要 GPU forward 的单样本接口
           如果没有，回退到 CPU */
        forward_cpu(net, state, q_out);
    } else {
        forward_cpu_only(net, state, q_out);
    }
    /* A skill score is a score of the complete model, not of a detached weight
       tensor.  Backend parity below still compares raw CPU/GPU math separately. */
    memory_apply_model_scores(g, q_out);
}

PuzzleResult test_puzzle(DQNAgent* agent, const GameState* puzzle, const int answers[], int num_answers, int use_gpu,
    PRNGState* rng) {
    PuzzleResult result = {};
    result.chosen_action = -1;
    if (!agent || !puzzle || !answers || num_answers < 0 || num_answers > CELL_COUNT) {
        result.margin = NAN; return result;
    }
    result.num_valid = num_answers;
    memcpy(result.valid_answers, answers, sizeof(int) * num_answers);
    float q[CELL_COUNT];
    get_q_values(agent, puzzle, q, use_gpu);
    float best = -INFINITY, second = -INFINITY, correct = -INFINITY, wrong = -INFINITY;
    for (int a = 0; a < CELL_COUNT; ++a) {
        if (puzzle->board[a]) continue;
        if (!isfinite(q[a])) { result.margin = NAN; return result; }
        int valid = 0;
        for (int j = 0; j < num_answers; ++j) if (answers[j] == a) valid = 1;
        if (valid) correct = fmaxf(correct, q[a]); else wrong = fmaxf(wrong, q[a]);
        if (q[a] > best) { second = best; best = q[a]; }
        else second = fmaxf(second, q[a]);
    }
    PRNGState local;
    if (!rng) { prng_init(&local, 0x534B494Cu); rng = &local; }
    result.chosen_action = greedy_legal_action(puzzle, q, rng);
    result.q_chosen = result.chosen_action < 0 ? NAN : q[result.chosen_action];
    result.q_second = second;
    result.margin = isfinite(correct) && isfinite(wrong) ? correct - wrong : 0;
    for (int j = 0; j < num_answers; ++j)
        if (answers[j] == result.chosen_action) result.is_correct = 1;
    return result;
}

// Solitaire, not alternating self-play. A failure remains zero; callers must
// check success before adding preset stones to the total completion length.
static int play_solo(DQNAgent* agent, GameState g, int use_gpu, PRNGState* rng) {
    int player = g.turn + 1;
    for (int steps = 1; steps <= CELL_COUNT; ++steps) {
        float q[CELL_COUNT];
        get_q_values(agent, &g, q, use_gpu);
        int best = greedy_legal_action(&g, q, rng);
        if (best < 0) return 0;
        g.board[best] = player; // deliberately keep side to move unchanged
        if (check_win_at(g.board, best) == player) return steps;
    }
    return 0;
}

static void score_empty_suite(DQNAgent* agent, SkillTestResult* result, int use_gpu,
    StageProgressCb cb, void* user, StageProgress* sp) {
    /* 每个快照内部跑两次并先求平均；对外统计口径始终是64个快照点，
       不是128次底层rollout。 */
    const int SNAPSHOTS = 64;
    const int RUNS_PER_SNAP = 2;
    const int TOTAL = SNAPSHOTS * RUNS_PER_SNAP;  /* 128 */
    GameState empty = {};
    PRNGState rng; prng_init(&rng, 0x534F4C4Fu);

    int total_wins = 0, total_sum = 0, total_shortest = 0;
    float snap_wr[64], snap_avg_steps[64];
    result->empty_min_steps = CELL_COUNT;
    result->empty_max_steps = 0;

    for (int s = 0; s < SNAPSHOTS; s++) {
        int sw = 0, ssum = 0;
        for (int r = 0; r < RUNS_PER_SNAP; r++) {
            int steps = play_solo(agent, empty, use_gpu, &rng);
            if (steps > 0) {
                ++total_wins; ++sw;
                total_sum += steps; ssum += steps;
                total_shortest += (steps <= 5) ? 1 : 0;
                result->empty_min_steps = fminf(result->empty_min_steps, (float)steps);
                result->empty_max_steps = fmaxf(result->empty_max_steps, (float)steps);
            }
            emit_stage(cb, user, sp, STG_SKILL_EMPTY, s * RUNS_PER_SNAP + r + 1,
                TOTAL, UNIT_GAMES, 0, 0);
        }
        snap_wr[s] = (float)sw / RUNS_PER_SNAP;
        snap_avg_steps[s] = sw > 0 ? (float)ssum / sw : 0.0f;
    }

    if (!total_wins) result->empty_min_steps = 0;
    result->empty_games = SNAPSHOTS;
    result->empty_win_rate = (float)total_wins / TOTAL;
    result->empty_wins = (int)lroundf(result->empty_win_rate * SNAPSHOTS);
    result->empty_avg_steps = total_wins ? (float)total_sum / total_wins : 0;
    result->empty_shortest_rate = (float)total_shortest / TOTAL;

    /* 64个快照胜率的统计量 */
    float wr_sum = 0, wr_sq = 0, wr_min = 1.0f, wr_max = 0.0f;
    float st_sum = 0, st_sq = 0;
    int st_cnt = 0;
    for (int s = 0; s < SNAPSHOTS; s++) {
        wr_sum += snap_wr[s];
        wr_sq += snap_wr[s] * snap_wr[s];
        if (snap_wr[s] < wr_min) wr_min = snap_wr[s];
        if (snap_wr[s] > wr_max) wr_max = snap_wr[s];
        if (snap_avg_steps[s] > 0) {
            st_sum += snap_avg_steps[s];
            st_sq += snap_avg_steps[s] * snap_avg_steps[s];
            st_cnt++;
        }
    }
    float wr_mean = wr_sum / SNAPSHOTS;
    float wr_var = wr_sq / SNAPSHOTS - wr_mean * wr_mean;
    result->empty_stat_mean = wr_mean;
    /* Equal two-run groups make win-rate aggregation algebraically identical,
       but publish the explicitly computed snapshot mean as the authoritative value. */
    result->empty_win_rate = wr_mean;
    result->empty_stat_std = sqrtf(wr_var > 0 ? wr_var : 0);
    result->empty_stat_min = wr_min;
    result->empty_stat_max = wr_max;
    if (st_cnt > 0) {
        float st_mean = st_sum / st_cnt;
        float st_var = st_sq / st_cnt - st_mean * st_mean;
        result->empty_steps_mean = st_mean;
        result->empty_steps_std = sqrtf(st_var > 0 ? st_var : 0);
    } else {
        result->empty_steps_mean = 0;
        result->empty_steps_std = 0;
    }
}

void evaluate_side_skills(DQNAgent* agent,SideSkillScores scores[2],int use_gpu){
    for(int side=0;side<2;++side){
        scores[side]=SideSkillScores{};PRNGState rng;prng_init(&rng,0x6576616cu+(unsigned)side);
        int wins=0,total_steps=0;
        for(int i=0;i<64;++i){GameState g={};g.turn=side;
            int steps=play_solo(agent,g,use_gpu,&rng);if(steps>0){++wins;total_steps+=steps;}}
        scores[side].count[0]=64;scores[side].score[0]=(float)wins/64;
        scores[side].empty_steps=wins?(float)total_steps/wins:25.f;
        for(int skill=1;skill<=2;++skill){
            GameState puzzles[128];int answers[128][CELL_COUNT],na[128],count=0,correct=0;
            generate_side_skill_puzzles(side,skill,0x484f4c44u+(unsigned)skill,1,puzzles,answers,na,&count,128);
            for(int i=0;i<count;++i)correct+=test_puzzle(agent,&puzzles[i],answers[i],na[i],use_gpu,&rng).is_correct;
            scores[side].count[skill]=count;scores[side].score[skill]=count?(float)correct/count:0;
        }
    }
}
void test_empty_win_suite(DQNAgent* agent, float* win_rate, float* avg_steps,
                         float* shortest_rate, int use_gpu) {
    SkillTestResult result{};
    score_empty_suite(agent, &result, use_gpu, NULL, NULL, NULL);
    *win_rate = result.empty_win_rate;
    *avg_steps = result.empty_avg_steps;
    *shortest_rate = result.empty_shortest_rate;
}

static void score_suite(DQNAgent* agent, GameState* puzzles, int answers[][CELL_COUNT],
                        int* na, int count, int use_gpu, float* accuracy, float* margin,
                        SkillPuzzleStats* stats,
                        StageProgressCb cb, void* user, StageProgress* sp,
                        int stage, int completed_offset, int total) {
    memset(stats, 0, sizeof(*stats));
    stats->total = count;
    stats->min_q = stats->min_margin = INFINITY;
    stats->max_q = stats->max_margin = -INFINITY;
    float margin_sum = 0;
    PRNGState rng; prng_init(&rng, 0x534B494Cu);
    for (int i = 0; i < count; ++i) {
        PuzzleResult r = test_puzzle(agent, &puzzles[i], answers[i], na[i], use_gpu, &rng);
        stats->correct += r.is_correct;
        if (r.is_correct) stats->q_correct += r.q_chosen; else stats->q_wrong += r.q_chosen;
        stats->min_q = fminf(stats->min_q, r.q_chosen);
        stats->max_q = fmaxf(stats->max_q, r.q_chosen);
        stats->min_margin = fminf(stats->min_margin, r.margin);
        stats->max_margin = fmaxf(stats->max_margin, r.margin);
        margin_sum += r.margin;
        emit_stage(cb, user, sp, stage, completed_offset + i + 1, total, UNIT_PUZZLES, 0, 0);
    }
    if (stats->correct) stats->q_correct /= stats->correct;
    if (count > stats->correct) stats->q_wrong /= count - stats->correct;
    *accuracy = count ? (float)stats->correct / count : 0;
    *margin = count ? margin_sum / count : 0;
    if (!count) stats->min_q = stats->max_q = stats->min_margin = stats->max_margin = 0;
}

void run_skill_tests(DQNAgent* agent, int episode, SkillTestResult* result, int use_gpu,
                     StageProgressCb cb, void* user) {
    StageProgress sp = {};
    g_skill_test_progress = 0; g_skill_test_phase = 1;
    memset(result, 0, sizeof(*result));
    result->episode = episode;
    result->suite_version = SKILL_SUITE_VERSION;
    result->parity_max_diff = result->parity_argmax_match = NAN;
    // CPU checks must use the same weights as resident GPU inference.
    if (use_gpu && agent == g_current_agent && gpu_net_active() &&
        gpu_net_download_agent(agent) != 0) {
        result->parity_status = -1;
        g_skill_test_progress = 100; g_skill_test_phase = 0;
        printf("[SKILL] Cannot freeze current GPU weights; test aborted.\n");
        return;
    }
    score_empty_suite(agent, result, use_gpu, cb, user, &sp);
    g_skill_test_progress = 15; g_skill_test_phase = 2;

    // Robustness is a separate score, never mixed with the pure empty-board baseline.
    int wins = 0, steps = 0;
    PRNGState seeded_rng; prng_init(&seeded_rng, 0x46495253u);
    for (int first = 0; first < CELL_COUNT; ++first) {
        GameState g = {};
        g.board[first] = 1;
        int remaining = play_solo(agent, g, use_gpu, &seeded_rng);
        if (remaining > 0) { ++wins; steps += remaining + 1; }
        emit_stage(cb, user, &sp, STG_SKILL_SEEDED, first + 1, CELL_COUNT, UNIT_GAMES, 0, 0);
    }
    result->seeded_win_rate = (float)wins / CELL_COUNT;
    result->seeded_avg_steps = wins ? (float)steps / wins : 0;
    g_skill_test_progress = 30; g_skill_test_phase = 3;

    /* ===== 白先分色评测 =====
       固定题库全部是黑方走（generate_*_fixed 用 player=1），所以「白方学会了没有」
       只能在白先的对局里测。这里用与黑先完全对称的构造：白方首着放在 25 个位置上，
       然后按题目棋色（turn=1 → 白方 online）solitaire 走完。
       必须与黑先完全对称，否则两色胜率不可比。 */
    {
        int w_wins = 0, w_steps = 0;
        PRNGState wrng; prng_init(&wrng, 0x57484954u);
        for (int first = 0; first < CELL_COUNT; ++first) {
            GameState g = {};
            g.board[first] = 2;      /* 白子 */
            g.turn = 1;              /* 白先 */
            const int remaining = play_solo(agent, g, use_gpu, &wrng);
            if (remaining > 0) { ++w_wins; w_steps += remaining + 1; }
            emit_stage(cb, user, &sp, STG_SKILL_SEEDED, first + 1, CELL_COUNT, UNIT_GAMES, 0, 0);
        }
        result->white_win_rate = (float)w_wins / CELL_COUNT;
        result->white_avg_steps = w_wins ? (float)w_steps / w_wins : 0;
        result->white_games = CELL_COUNT;
        result->black_win_rate = result->seeded_win_rate;
        result->black_avg_steps = result->seeded_avg_steps;
        result->black_games = CELL_COUNT;
        result->side_split_valid = 1;
        printf("[SKILL] side-split (white-to-move first stones): win %.1f%% steps %.2f ; "
               "black-to-move: win %.1f%% steps %.2f\n",
            result->white_win_rate * 100, result->white_avg_steps,
            result->black_win_rate * 100, result->black_avg_steps);
    }

    GameState puzzles[512];
    int answers[512][CELL_COUNT], na[512], count = 0;
    SkillPuzzleStats scratch;
    generate_win1_fixed(puzzles, answers, na, &count, 512);
    int win1_fixed_count = count;
    score_suite(agent, puzzles, answers, na, count, use_gpu,
                &result->win1_fixed, &result->win1_fixed_margin, &result->win1_stats,
                cb, user, &sp, STG_SKILL_WIN1, 0, win1_fixed_count);
    result->total_tests = result->empty_games + CELL_COUNT + count;
    g_skill_test_progress = 50;
    PRNGState rng;
    prng_init(&rng, (uint32_t)episode * 7919u + 1u);
    generate_win1_random(puzzles, answers, na, &count, 512, &rng);
    score_suite(agent, puzzles, answers, na, count, use_gpu,
                &result->win1_random, &result->win1_random_margin, &scratch,
                cb, user, &sp, STG_SKILL_WIN1, win1_fixed_count, win1_fixed_count + count);
    result->total_tests += count;
    g_skill_test_progress = 65; g_skill_test_phase = 4;
    generate_block1_fixed(puzzles, answers, na, &count, 512);
    int block1_fixed_count = count;
    score_suite(agent, puzzles, answers, na, count, use_gpu,
                &result->block1_fixed, &result->block1_fixed_margin, &result->block1_stats,
                cb, user, &sp, STG_SKILL_BLOCK1, 0, block1_fixed_count);
    result->total_tests += count;
    g_skill_test_progress = 80;
    prng_init(&rng, (uint32_t)episode * 104729u + 1u);
    generate_block1_random(puzzles, answers, na, &count, 512, &rng);
    score_suite(agent, puzzles, answers, na, count, use_gpu,
                &result->block1_random, &result->block1_random_margin, &scratch,
                cb, user, &sp, STG_SKILL_BLOCK1, block1_fixed_count, block1_fixed_count + count);
    result->total_tests += count;
    g_skill_test_progress = 95;

    /* 分色字段：固定题库全部是黑方走，所以「固定题正确率」显式标为 black 分色；
       白方分色由上面的白先对局给出。 */
    result->win1_fixed_black = result->win1_fixed;
    result->block1_fixed_black = result->block1_fixed;

    if (use_gpu) {
        /* 按方路由下设备载荷是**当前绑定方**的 online，不是共享 q_net。parity 必须用
           本次评测实际使用的网络（与 get_q_values 的取网规则一致），否则它校验的是
           一个与题面打分无关的权重副本，退化报警的判据会被稀释。 */
        const NeuralNet* parity_net = &agent->q_net;
        if (agent->indep && agent->indep->active()) {
            NeuralNet* side_net = independent_training::agent_online_net(agent,
                independent_training::indep_bound(agent));
            if (side_net) parity_net = side_net;
        }
        float states[8][CELL_COUNT + 1] = {}, cpu[CELL_COUNT], gpu[8][CELL_COUNT];
        for (int i = 0; i < 8; ++i) {
            GameState g = {};
            if (count > 0) g = puzzles[i % count];
            g.turn = i % 2; // parity includes both sides to move
            state_to_input(&g, states[i]);
        }
        if (forward_gpu_snapshot(parity_net, &states[0][0], 8, &gpu[0][0]) == 0) {
            result->parity_status = 1;
            result->parity_max_diff = 0;
            int matches = 0;
            for (int i = 0; i < 8; ++i) {
                forward_cpu_only(parity_net, states[i], cpu);
                int ca = -1, ga = -1;
                for (int a = 0; a < CELL_COUNT; ++a) {
                    if (!isfinite(cpu[a]) || !isfinite(gpu[i][a])) result->parity_status = -1;
                    result->parity_max_diff = fmaxf(result->parity_max_diff, fabsf(cpu[a] - gpu[i][a]));
                    if (states[i][a] != 0) continue;
                    if (ca < 0 || cpu[a] > cpu[ca]) ca = a;
                    if (ga < 0 || gpu[i][a] > gpu[i][ga]) ga = a;
                }
                matches += ca == ga;
            }
            result->parity_argmax_match = matches / 8.0f;
        } else result->parity_status = -1;
    }
    printf("[SKILL v%d] SoloEmpty %.1f%% steps %.2f; forced-first %.1f%% steps %.2f; "
           "Win1 %.1f%% Block1 %.1f%%; parity=%d diff=%g\n",
           SKILL_SUITE_VERSION, result->empty_win_rate * 100, result->empty_avg_steps,
           result->seeded_win_rate * 100, result->seeded_avg_steps,
           result->win1_fixed * 100, result->block1_fixed * 100,
           result->parity_status, result->parity_max_diff);
    g_skill_test_progress = 100; g_skill_test_phase = 0;
}

/* ========== 技能状态追踪 ========== */

void skill_tracker_init(SkillTracker* tracker) {
    memset(tracker, 0, sizeof(SkillTracker));
    // Measuring a baseline is not evidence of mastery.
    tracker->win1_state = SKILL_LEARNING;
    tracker->empty_state = SKILL_LEARNING;
    tracker->block1_state = SKILL_LEARNING;
    tracker->initialized = 0;
}

int skill_tracker_update(SkillTracker* tracker, const SkillTestResult* result) {
    int alert = 0;
    if (!tracker || !result) return 1;
    const float rates[] = { result->win1_fixed, result->block1_fixed,
        result->empty_win_rate, result->empty_shortest_rate };
    for (float rate : rates) if (!isfinite(rate) || rate < 0 || rate > 1) return 1;
    if (result->suite_version != SKILL_SUITE_VERSION || result->total_tests <= 0 ||
        !isfinite(result->empty_avg_steps) || result->empty_avg_steps < 0 ||
        !isfinite(result->win1_fixed_margin) || !isfinite(result->block1_fixed_margin) ||
        result->parity_status < 0 ||
        (result->parity_status == 1 && (!isfinite(result->parity_max_diff) || result->parity_max_diff > 0.001f))) return 1;

    // Old checkpoints may contain an unconditional MASTERED label paired with
    // an 8% baseline. Such a baseline was never evidence of graduation.
    if (tracker->win1_state == SKILL_MASTERED &&
        (!isfinite(tracker->win1_baseline) || tracker->win1_baseline < 0.99f))
        tracker->win1_state = SKILL_LEARNING;
    if (tracker->empty_state == SKILL_MASTERED &&
        (!isfinite(tracker->empty_baseline) || tracker->empty_baseline < 0.99f ||
         !isfinite(tracker->empty_steps_baseline) || tracker->empty_steps_baseline <= 0 ||
         tracker->empty_steps_baseline > 4.2f || !isfinite(tracker->empty_shortest_baseline) ||
         tracker->empty_shortest_baseline < 0.99f))
        tracker->empty_state = SKILL_LEARNING;

    /* 首次运行：记录 baseline */
    // The old flag value 1 identifies strict-argmax baselines. New values carry
    // the suite version without changing the checkpoint's on-disk struct size.
    if (tracker->initialized != SKILL_SUITE_VERSION) {
        skill_tracker_init(tracker);
        tracker->win1_baseline = result->win1_fixed;
        tracker->empty_baseline = result->empty_win_rate;
        tracker->empty_steps_baseline = result->empty_avg_steps;
        tracker->empty_shortest_baseline = result->empty_shortest_rate;
        tracker->block1_baseline = 0.0f; /* 未毕业，无 baseline */
        tracker->initialized = SKILL_SUITE_VERSION;
    }

    if (tracker->win1_state != SKILL_MASTERED && result->win1_fixed >= 0.99f) {
        tracker->win1_state = SKILL_MASTERED;
        tracker->win1_baseline = result->win1_fixed;
    }
    // EmptyWin is explicitly the solo task, not a tactical/generalization certificate.
    if (tracker->empty_state != SKILL_MASTERED && result->empty_win_rate >= 0.99f &&
        result->empty_avg_steps > 0 && result->empty_avg_steps <= 4.2f &&
        result->empty_shortest_rate >= 0.99f) {
        tracker->empty_state = SKILL_MASTERED;
        tracker->empty_baseline = result->empty_win_rate;
        tracker->empty_steps_baseline = result->empty_avg_steps;
        tracker->empty_shortest_baseline = result->empty_shortest_rate;
    }

    /* Win1 回归检查 */
    if (tracker->win1_state == SKILL_MASTERED) {
        float drop = tracker->win1_baseline - result->win1_fixed;
        if (drop > 0.03f) {
            printf("[SKILL] ⚠ Win1 回归: %.1f%% → %.1f%% (下降 %.1fpp)\n",
                   tracker->win1_baseline * 100, result->win1_fixed * 100, drop * 100);
            alert = 1;
        } else if (drop > 0.01f) {
            printf("[SKILL] Win1 轻微退化: %.1f%% → %.1f%%\n",
                   tracker->win1_baseline * 100, result->win1_fixed * 100);
        }
        // Keep the graduation baseline fixed; gradual forgetting must not move the goalposts.
    }

    /* EmptyWin 回归检查 */
    if (tracker->empty_state == SKILL_MASTERED) {
        float drop = tracker->empty_baseline - result->empty_win_rate;
        if (drop > 0.03f) {
            printf("[SKILL] ⚠ EmptyWin 回归: %.1f%% → %.1f%%\n",
                   tracker->empty_baseline * 100, result->empty_win_rate * 100);
            alert = 1;
        }
        if (tracker->empty_steps_baseline > 0 &&
            (result->empty_avg_steps > tracker->empty_steps_baseline + 1.0f ||
             result->empty_shortest_rate < tracker->empty_shortest_baseline - 0.03f)) {
            printf("[SKILL] Solo speed regressed: %.2f -> %.2f steps\n",
                tracker->empty_steps_baseline, result->empty_avg_steps);
            alert = 1;
        }
    }

    /* Block1 毕业检查 */
    if (tracker->block1_state == SKILL_LEARNING && result->block1_fixed >= 0.95f) {
        tracker->block1_state = SKILL_MASTERED;
        tracker->block1_baseline = result->block1_fixed;
        printf("[SKILL] ✓ Block1 毕业! %.1f%% → MASTERED\n", result->block1_fixed * 100);
    }

    /* Block1 回归检查 */
    if (tracker->block1_state == SKILL_MASTERED) {
        float drop = tracker->block1_baseline - result->block1_fixed;
        if (drop > 0.03f) {
            printf("[SKILL] ⚠ Block1 回归: %.1f%% → %.1f%%\n",
                   tracker->block1_baseline * 100, result->block1_fixed * 100);
            alert = 1;
        }
    }

    return alert;
}

/* ========== metric_skill.bin 写入 ==========

   V4格式：24字节 header + N × 124字节记录。
   header 的 field_count 必须与实际写入的29个float一致；否则读取器会按
   旧的36字节步长拆分记录并把一条记录误读成多条。 */

int write_skill_metric(const char* path, const SkillMetricRecord* rec) {
    if (!rec) return 0;
    SkillMetricRow row{};
    row.episode = rec->episode;
    row.step = 0;
    int f = 0;
    /* EmptyWin */
    row.fields[f++] = rec->empty_win;
    row.fields[f++] = rec->avg_steps;
    row.fields[f++] = rec->shortest;
    row.fields[f++] = rec->empty_min_steps;
    row.fields[f++] = rec->empty_max_steps;
    /* Win1 */
    row.fields[f++] = rec->win1_fixed;
    row.fields[f++] = rec->win1_fixed_margin;
    row.fields[f++] = rec->win1_random;
    row.fields[f++] = rec->win1_avg_q_correct;
    row.fields[f++] = rec->win1_avg_q_wrong;
    row.fields[f++] = rec->win1_min_q;
    row.fields[f++] = rec->win1_max_q;
    row.fields[f++] = rec->win1_min_margin;
    row.fields[f++] = rec->win1_max_margin;
    /* Block1 */
    row.fields[f++] = rec->block1_fixed;
    row.fields[f++] = rec->block1_fixed_margin;
    row.fields[f++] = rec->block1_random;
    row.fields[f++] = rec->block1_avg_q_correct;
    row.fields[f++] = rec->block1_avg_q_wrong;
    row.fields[f++] = rec->block1_min_q;
    row.fields[f++] = rec->block1_max_q;
    row.fields[f++] = rec->block1_min_margin;
    row.fields[f++] = rec->block1_max_margin;
    /* 多快照统计 */
    row.fields[f++] = rec->empty_stat_std;
    row.fields[f++] = rec->empty_stat_min;
    row.fields[f++] = rec->empty_stat_max;
    row.fields[f++] = rec->win1_acc_std;
    row.fields[f++] = rec->block1_acc_std;
    row.fields[f++] = (float)rec->mastered_puzzle_count;
    if (append_skill_metric(path, row)) return 1;
    printf("[SKILL LOG] Append failed or incompatible format: %s\n", path ? path : "(null)");
    return 0;
}

/* ========== 多快照测试 ========== */

/* 统计辅助 */
static float calc_mean(const float* v, int n) {
    if (n <= 0) return 0;
    double s = 0; for (int i = 0; i < n; i++) s += v[i];
    return (float)(s / n);
}
static float calc_std(const float* v, int n, float mean) {
    if (n <= 1) return 0;
    double s = 0; for (int i = 0; i < n; i++) s += (v[i] - mean) * (v[i] - mean);
    return sqrtf((float)(s / (n - 1)));
}
static int float_cmp(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}
static float calc_percentile(float* sorted, int n, float pct) {
    if (n <= 0) return 0;
    float idx = pct * (n - 1);
    int lo = (int)idx, hi = lo + 1 < n ? lo + 1 : lo;
    float frac = idx - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

int run_multi_snapshot_empty(const NeuralNet* snapshots[], int snapshot_count,
    SkillTestResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb, void* user) {
    if (!result) return 0;
    memset(result, 0, sizeof(*result));
    if (phase) *phase = 2;
    if (!snapshots || snapshot_count <= 0 || snapshot_count > SKILL_WINDOW_SNAPSHOTS) return 0;

    DQNAgent* temp = (DQNAgent*)calloc(1, sizeof(DQNAgent));
    if (!temp) return 0;
    temp->seed_id = -1;
    PRNGState rng; prng_init(&rng, 0x57494E44u);
    float rates[SKILL_WINDOW_SNAPSHOTS] = {};
    float mean_steps[SKILL_WINDOW_SNAPSHOTS] = {};
    int total_wins = 0, total_steps = 0, total_shortest = 0;
    float min_steps = (float)CELL_COUNT, max_steps = 0.0f;
    StageProgress sp = {};
    const int total_games = snapshot_count * 2;

    for (int s = 0; s < snapshot_count; ++s) {
        if (!snapshots[s]) continue;
        memcpy(&temp->q_net, snapshots[s], sizeof(NeuralNet));
        int wins = 0, steps_sum = 0;
        for (int r = 0; r < 2; ++r) {
            GameState empty = {};
            int steps = play_solo(temp, empty, 0, &rng);
            if (steps > 0) {
                ++wins; ++total_wins;
                steps_sum += steps; total_steps += steps;
                total_shortest += steps <= 5;
                min_steps = fminf(min_steps, (float)steps);
                max_steps = fmaxf(max_steps, (float)steps);
            }
            emit_stage(cb, user, &sp, STG_MULTI_EMPTY, s * 2 + r + 1, total_games,
                UNIT_GAMES, s, snapshot_count);
        }
        rates[s] = wins / 2.0f;
        mean_steps[s] = wins ? (float)steps_sum / wins : 0.0f;
        if (progress) *progress = (s + 1) * 100 / snapshot_count;
    }

    result->empty_games = snapshot_count; /* 对外口径是快照点，不是底层128局 */
    result->empty_win_rate = calc_mean(rates, snapshot_count);
    result->empty_wins = (int)lroundf(result->empty_win_rate * snapshot_count);
    result->empty_avg_steps = total_wins ? (float)total_steps / total_wins : 0.0f;
    result->empty_shortest_rate = (float)total_shortest / (snapshot_count * 2);
    result->empty_min_steps = total_wins ? min_steps : 0.0f;
    result->empty_max_steps = total_wins ? max_steps : 0.0f;
    result->empty_stat_mean = result->empty_win_rate;
    result->empty_stat_std = calc_std(rates, snapshot_count, result->empty_stat_mean);
    result->empty_stat_min = rates[0];
    result->empty_stat_max = rates[0];
    for (int s = 1; s < snapshot_count; ++s) {
        result->empty_stat_min = fminf(result->empty_stat_min, rates[s]);
        result->empty_stat_max = fmaxf(result->empty_stat_max, rates[s]);
    }
    int step_count = 0;
    float valid_steps[SKILL_WINDOW_SNAPSHOTS];
    for (int s = 0; s < snapshot_count; ++s)
        if (mean_steps[s] > 0.0f) valid_steps[step_count++] = mean_steps[s];
    result->empty_steps_mean = calc_mean(valid_steps, step_count);
    result->empty_steps_std = calc_std(valid_steps, step_count, result->empty_steps_mean);
    free(temp);
    return 1;
}

/* 对单个快照跑完整题库，返回准确率和 margin */
static float snapshot_test_puzzles(const NeuralNet* net,
    const GameState* puzzles, const int answers[][CELL_COUNT], const int na[],
    int count, SkillPuzzleStats* stats, uint8_t* correct_out, float* margin_out,
    StageProgressCb cb, void* user, StageProgress* sp, int stage,
    int snapshot_index, int snapshot_total, int total_puzzles) {
    DQNAgent temp;
    memset(&temp, 0, sizeof(temp));
    memcpy(&temp.q_net, net, sizeof(NeuralNet));
    memcpy(&temp.target_net, net, sizeof(NeuralNet));
    temp.seed_id = -1;
    prng_init(&temp.eval_rng, 0x534B494Cu);
    prng_init(&temp.train_rng, 0xDEADBEEF);

    memset(stats, 0, sizeof(*stats));
    stats->min_q = stats->min_margin = INFINITY;
    stats->max_q = stats->max_margin = -INFINITY;
    float margin_sum = 0;
    PRNGState rng; prng_init(&rng, 0x534B494Cu);
    for (int i = 0; i < count; ++i) {
        PuzzleResult r = test_puzzle(&temp, &puzzles[i], answers[i], na[i], 0, &rng);
        if (correct_out) correct_out[i] = (uint8_t)(r.is_correct != 0);
        if (margin_out) margin_out[i] = r.margin;
        stats->correct += r.is_correct;
        margin_sum += r.margin;
        if (r.q_chosen < stats->min_q) stats->min_q = r.q_chosen;
        if (r.q_chosen > stats->max_q) stats->max_q = r.q_chosen;
        if (r.margin < stats->min_margin) stats->min_margin = r.margin;
        if (r.margin > stats->max_margin) stats->max_margin = r.margin;
        emit_stage(cb, user, sp, stage, snapshot_index * count + i + 1, total_puzzles,
            UNIT_PUZZLES, snapshot_index, snapshot_total);
    }
    stats->total = count;
    stats->q_correct = count ? margin_sum / count : 0;
    return count ? (float)stats->correct / count : 0;
}

/* 多快照 Win1/Block1 通用评估 */
static void run_multi_snapshot_puzzles(const NeuralNet* snapshots[], int snapshot_count,
    int is_block, MultiSnapshotResult* result,
    volatile int* progress, volatile int* phase,
    StageProgressCb cb, void* user) {
    memset(result, 0, sizeof(*result));
    result->snapshot_count = snapshot_count;
    if (!snapshots || snapshot_count <= 0 || snapshot_count > SKILL_WINDOW_SNAPSHOTS) return;

    /* 生成固定题库（只生成一次） */
    GameState puzzles[SNAPSHOT_MAX_PUZZLES];
    int answers[SNAPSHOT_MAX_PUZZLES][CELL_COUNT];
    int na[SNAPSHOT_MAX_PUZZLES], count = 0;
    if (is_block)
        generate_block1_fixed(puzzles, answers, na, &count, SNAPSHOT_MAX_PUZZLES);
    else
        generate_win1_fixed(puzzles, answers, na, &count, SNAPSHOT_MAX_PUZZLES);
    result->puzzle_count = count;

    /* 每题的跨快照正确计数 */
    int puzzle_correct[SNAPSHOT_MAX_PUZZLES] = {};
    float puzzle_margin_sum[SNAPSHOT_MAX_PUZZLES] = {};
    StageProgress sp = {};
    const int stage = is_block ? STG_MULTI_BLOCK1 : STG_MULTI_WIN1;
    const int total_puzzles = snapshot_count * count;

    /* 逐快照评估 */
    float accs[SKILL_WINDOW_SNAPSHOTS], margins[SKILL_WINDOW_SNAPSHOTS];
    for (int s = 0; s < snapshot_count; s++) {
        if (progress) *progress = (int)((float)s / snapshot_count * 90);

        uint8_t correct[SNAPSHOT_MAX_PUZZLES] = {};
        float puzzle_margins[SNAPSHOT_MAX_PUZZLES] = {};
        SkillPuzzleStats stats1;
        float acc1 = snapshot_test_puzzles(snapshots[s], puzzles, answers, na, count,
            &stats1, correct, puzzle_margins, cb, user, &sp, stage, s, snapshot_count, total_puzzles);

        /* 记录结果 */
        result->snapshots[s].accuracy = acc1;
        result->snapshots[s].mean_margin = stats1.q_correct;
        result->snapshots[s].positive_margin_rate = count > 0 ?
            (float)(stats1.total - stats1.correct > 0 ? 0 : 0) : 0; /* 简化 */
        result->snapshots[s].determinism_mismatch = 0; /* 固定题库和固定PRNG，单遍即确定 */

        accs[s] = acc1;
        margins[s] = stats1.q_correct;

        for (int i = 0; i < count; ++i) {
            puzzle_correct[i] += correct[i] != 0;
            puzzle_margin_sum[i] += puzzle_margins[i];
        }
    }

    /* 快照级统计 */
    float acc_mean = calc_mean(accs, snapshot_count);
    result->acc_mean = acc_mean;
    result->acc_std = calc_std(accs, snapshot_count, acc_mean);
    result->margin_mean = calc_mean(margins, snapshot_count);
    result->margin_std = calc_std(margins, snapshot_count, result->margin_mean);

    /* 百分位 */
    float sorted[SKILL_WINDOW_SNAPSHOTS];
    memcpy(sorted, accs, snapshot_count * sizeof(float));
    qsort(sorted, snapshot_count, sizeof(float), float_cmp);
    result->acc_min = sorted[0];
    result->acc_max = sorted[snapshot_count - 1];
    result->acc_p10 = calc_percentile(sorted, snapshot_count, 0.10f);
    result->acc_p90 = calc_percentile(sorted, snapshot_count, 0.90f);

    /* 当前模型（最新快照） */
    result->current_accuracy = accs[snapshot_count - 1];
    result->current_margin = margins[snapshot_count - 1];

    /* 题目稳定性 */
    result->mastered_count = 0;
    result->borderline_count = 0;
    result->unknown_count = 0;
    for (int i = 0; i < count; ++i) {
        result->puzzle_stability[i].puzzle_id = i;
        result->puzzle_stability[i].total_snapshots = snapshot_count;
        result->puzzle_stability[i].correct_count = puzzle_correct[i];
        result->puzzle_stability[i].avg_margin = snapshot_count > 0 ?
            puzzle_margin_sum[i] / snapshot_count : 0;
        if (puzzle_correct[i] >= snapshot_count - 1) result->mastered_count++;
        else if (puzzle_correct[i] * 16 >= snapshot_count * 5) result->borderline_count++;
        else result->unknown_count++;
    }

    if (progress) *progress = 100;
}

void run_multi_snapshot_win1(const NeuralNet* snapshots[], int snapshot_count,
    MultiSnapshotResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb, void* user) {
    if (phase) *phase = 3;
    run_multi_snapshot_puzzles(snapshots, snapshot_count, 0, result, progress, phase, cb, user);
}

void run_multi_snapshot_block1(const NeuralNet* snapshots[], int snapshot_count,
    MultiSnapshotResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb, void* user) {
    if (phase) *phase = 4;
    run_multi_snapshot_puzzles(snapshots, snapshot_count, 1, result, progress, phase, cb, user);
}

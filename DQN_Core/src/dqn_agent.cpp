#include "core.h"
#include "competency_training.h"

#include "independent_networks.h"
#include "experience_semantics.h"
#include "training_log_timeline.h"
#include "metric_log_writer.h"
#include "action_policy.h"
#include "skill_test.h"

/* Kept process-local instead of inside DQNAgent: reward feedback is a transient
   training controller and must not silently alter the checkpoint layout. */
static float g_terminal_win_reward_scale[2] = { 1.0f, 1.0f };
/* With the configured terminal and bounded shaping rewards, a legal discounted
   return fits comfortably inside this range.  Bounding the bootstrap target
   prevents a transient target-network spike from recursively amplifying itself. */
static const float DQN_TARGET_ABS_LIMIT = 5.0f;
/* Huber(delta=0.5) above this value requires |Q-target| > 200.25.  With
   targets clamped to +/-5 this is not an ordinary hard sample; it is an
   activation runaway.  Exclude it before backpropagation so one poisoned
   replay item cannot move the shared network. */
static const float DQN_REPLAY_LOSS_REJECT_LIMIT = 100.0f;
static const float DQN_RAW_TQ_REJECT_LIMIT = 50.0f;
static unsigned long long g_replay_loss_reject_count = 0;
/* 实验开关：置 1 时回放采样退化为均匀采样，用于对照 M1 采样偏置（默认 0=生产逻辑）。 */
int g_uniform_replay_sampling = 0;
/* 实验统计：replay_from_buffer 的尝试/提交/拒绝（0=无样本/拒绝/失败）批次计数。 */
unsigned long long g_replay_attempt_count = 0;
unsigned long long g_replay_commit_count = 0;
unsigned long long g_replay_reject_count = 0;

/* ===== TQ 全链路监控 =====
   记录未裁剪的 raw_tq 与 bootstrap 全链路字段。裁剪到 [-5,5] 会隐藏 target_net
   发散（崩溃前训练 loss 长期正常但 raw_tq 已爆炸），因此必须在裁剪前统计并告警。 */
typedef struct {
    double raw_sum, raw_sum2;          /* raw_tq 均值/方差 */
    float  raw_max_abs;                /* raw_tq 最大绝对值 */
    int    count;
    int    clip_count;                 /* 触发裁剪的样本数 */
    int    black_count, white_count;
    double black_raw_abs, white_raw_abs;
    int    terminal_count, nonterminal_count;
    int    neg_bootstrap, pos_bootstrap;
} TqBatch;

static TqBatch g_tq = {0};
static unsigned long long g_tq_flush_count = 0;
static double g_tq_prev_mean = 0.0;      /* 用于"突然跳变"检测 */
static int    g_tq_has_prev = 0;

/* TQ 告警阈值：正常 target 被 clamp 在 ±5，raw_tq 稳定时应在该量级附近。 */
#define TQ_RAW_ABS_ALERT    50.0f     /* raw_tq 绝对值超此视为 target_net 发散 */
#define TQ_CLIP_RATIO_ALERT 0.50f     /* 裁剪比例超 50% 视为 TQ 分布异常 */
#define TQ_FLUSH_INTERVAL   100       /* 每 100 批打印一次统计 */

static void tq_record_sample(float raw_tq, float clipped_tq, int is_black, int is_terminal,
    int bootstrap_sign) {
    g_tq.count++;
    g_tq.raw_sum += raw_tq;
    g_tq.raw_sum2 += (double)raw_tq * raw_tq;
    if (fabsf(raw_tq) > g_tq.raw_max_abs) g_tq.raw_max_abs = fabsf(raw_tq);
    if (raw_tq != clipped_tq) g_tq.clip_count++;
    if (is_black) { g_tq.black_count++; g_tq.black_raw_abs += fabsf(raw_tq); }
    else { g_tq.white_count++; g_tq.white_raw_abs += fabsf(raw_tq); }
    if (is_terminal) g_tq.terminal_count++; else g_tq.nonterminal_count++;
    if (bootstrap_sign < 0) g_tq.neg_bootstrap++; else if (bootstrap_sign > 0) g_tq.pos_bootstrap++;
}

static void tq_flush_batch(void) {
    if (g_tq.count == 0) return;
    ++g_tq_flush_count;

    const double mean = g_tq.raw_sum / g_tq.count;
    const double var = g_tq.count > 1 ?
        (g_tq.raw_sum2 - g_tq.count * mean * mean) / (g_tq.count - 1) : 0.0;
    const double clip_ratio = (double)g_tq.clip_count / g_tq.count;

    int alert = 0;
    if (g_tq.raw_max_abs > TQ_RAW_ABS_ALERT) alert = 1;
    if (clip_ratio > TQ_CLIP_RATIO_ALERT) alert = 1;

    if (g_tq_flush_count % TQ_FLUSH_INTERVAL == 0 || alert) {
        printf("[TQ] n=%d raw mean=%.4g std=%.4g max|raw|=%.4g clip=%.1f%% "
            "B=%d W=%d T=%d/N=%d negB=%d posB=%d%s\n",
            g_tq.count, mean, sqrt(var), g_tq.raw_max_abs, clip_ratio * 100.0,
            g_tq.black_count, g_tq.white_count, g_tq.terminal_count, g_tq.nonterminal_count,
            g_tq.neg_bootstrap, g_tq.pos_bootstrap, alert ? "  [ALERT]" : "");
    }
    if (alert && !g_tq_has_prev) {
        printf("[TQ] ALERT raw_tq 分布异常（max|raw|或裁剪比例超阈值），target_net 可能发散。\n");
    }
    if (g_tq_has_prev && fabsf(mean - g_tq_prev_mean) > TQ_RAW_ABS_ALERT) {
        printf("[TQ] ALERT raw_tq 均值突然跳变: %.4g -> %.4g\n", g_tq_prev_mean, mean);
    }

    g_tq_prev_mean = mean;
    g_tq_has_prev = 1;
    memset(&g_tq, 0, sizeof(g_tq));
}

/* ===== 记忆锚点 Replay：固定战术锚点（Win1/Block1） =====
   从固定题库用真实环境 make_move 生成合法 transition（绝不凭记忆库统计伪造）。
   训练采样时按可配置比例混入，主动维持基础战术，对抗灾难性遗忘。 */
#define ANCHOR_CAPACITY          192     /* X 视角 + O 镜像，容量翻倍 */
#define ANCHOR_SAMPLE_RATIO      0.10f   /* 10% 战术锚点（可配置） */
/* 锚点存原始题面而非冻结 reward：采样时按当前课程阶段动态重算，
   避免 phase1 生成时 Block1 封堵 reward 被冻结为 0（课程推进后不更新）。 */
typedef struct {
    GameState puzzle;   /* 行动者视角题面 */
    int       action;   /* 正确答案动作 */
} AnchorEntry;
static AnchorEntry g_anchor[ANCHOR_CAPACITY];
static int g_anchor_count = 0;
static int g_anchor_inited = 0;

/* 颜色镜像（O 视角等价局面）：左右翻转 + 棋子 1<->2 + turn 0<->1。
   与 store_experience 的 SYMMETRY 镜像语义一致（board.cpp flip_state 的 GameState 版）。 */
static void mirror_puzzle(const GameState* src, GameState* dst) {
    dst->turn = (src->turn == 0) ? 1 : 0;
    for (int i = 0; i < CELL_COUNT; ++i) {
        int c = src->board[24 - i];
        dst->board[i] = (c == 1) ? 2 : (c == 2) ? 1 : 0;
    }
}

static int anchor_push_entry(const GameState* puzzle, int action) {
    if (g_anchor_count >= ANCHOR_CAPACITY) return 0;
    if (!puzzle || action < 0 || action >= CELL_COUNT || puzzle->board[action] != 0) return 0;
    if (puzzle->turn != 0 && puzzle->turn != 1) return 0;
    /* 完整 validate（构造 transition），保证采样时不会触发 validate 失败。 */
    Experience e;
    state_to_input(puzzle, e.state);
    GameState next = *puzzle;
    make_move(&next, action);
    state_to_input(&next, e.next_state);
    e.action = action;
    e.done = check_winner(&next) != 0 || is_full(&next);
    e.reward = 0.0f;   /* 占位：validate 只查 finite 与 done 一致性 */
    if (!validate_experience(&e)) return 0;
    g_anchor[g_anchor_count].puzzle = *puzzle;
    g_anchor[g_anchor_count].action = action;
    ++g_anchor_count;
    return 1;
}

/* 一个题面双写 X 视角 + O 镜像，覆盖双方（普通 replay 有 SYMMETRY 镜像，锚点此前缺失）。 */
static void anchor_push_pair(const GameState* puzzle, int action) {
    anchor_push_entry(puzzle, action);
    GameState mirror;
    mirror_puzzle(puzzle, &mirror);
    anchor_push_entry(&mirror, 24 - action);
}

static void anchor_buffer_init(DQNAgent* agent) {
    if (g_anchor_inited) return;
    g_anchor_inited = 1;
    (void)agent;   /* 锚点只存题面，reward 在采样时按当前课程动态计算 */
    GameState puzzles[64];
    int answers[64][CELL_COUNT];
    int num_answers[64];
    int count = 0;
    generate_win1_fixed(puzzles, answers, num_answers, &count, 32);
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < num_answers[i]; ++j)
            anchor_push_pair(&puzzles[i], answers[i][j]);
    generate_block1_fixed(puzzles, answers, num_answers, &count, 32);
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < num_answers[i]; ++j)
            anchor_push_pair(&puzzles[i], answers[i][j]);
    printf("[ANCHOR] 初始化 %d 条战术锚点（Win1/Block1，含 O 镜像，采样时按当前课程重算奖励）\n", g_anchor_count);
}

void set_terminal_win_reward_scales(float black_scale, float white_scale) {
    if (!isfinite(black_scale) || !isfinite(white_scale)) return;
    /* The Trainer currently uses a tighter range.  This defensive clamp keeps
       accidental callers from changing the Bellman target scale drastically. */
    if (black_scale < 0.5f) black_scale = 0.5f;
    if (black_scale > 1.5f) black_scale = 1.5f;
    if (white_scale < 0.5f) white_scale = 0.5f;
    if (white_scale > 1.5f) white_scale = 1.5f;
    g_terminal_win_reward_scale[0] = black_scale;
    g_terminal_win_reward_scale[1] = white_scale;
}

int get_reward_limit(int type_idx) {
    switch (type_idx) {
    case 0: return MAX_LIVE_TWO_PER_GAME;
    case 1: return MAX_OPEN_THREE_PER_GAME;
    case 2: return MAX_THREAT_THREE_PER_GAME;
    case 3: return MAX_FOUR_THREAT_PER_GAME;
    case 4: return MAX_DOUBLE_THREAT_PER_GAME;
    case 5: return MAX_BLOCK_WIN_PER_GAME;
    case 6: return MAX_BLOCK_OPEN_THREE_PER_GAME;
    default: return 999;
    }
}

int select_action_with_net(DQNAgent* agent, const GameState* g, NeuralNet* net,
    float eps_override, PRNGState* rng) {
    int moves[CELL_COUNT], cnt = get_legal_moves(g, moves);
    if (cnt == 0) return -1;
    if (rng == NULL) rng = &agent->train_rng;

    float eps = (eps_override >= 0) ? eps_override : agent->epsilon;

    float input[CELL_COUNT + 1];
    state_to_input(g, input);
    float qvals[CELL_COUNT];

    // Timeout detection
    LARGE_INTEGER start_time, current_time, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start_time);
    const double TIMEOUT_SEC = 2.0;

    forward_cpu(net, input, qvals);

    if (g_training_fault) return -1;

    QueryPerformanceCounter(&current_time);
    double elapsed = (double)(current_time.QuadPart - start_time.QuadPart) / freq.QuadPart;
    if (elapsed > TIMEOUT_SEC) {
        printf("[WARN] forward_cpu timeout %.3f sec, fallback to random\n", elapsed);
        return moves[prng_int(rng, cnt)];
    }

    // Observe official self-play positions for retention/pruning statistics.
    if (g_hot_config.memory_enabled && agent->seed_id < 0 && g_kb_count > 0) {
        int perspective = g->turn;  // 0=X, 1=O
        GameState normalized;
        normalize_state_to_self(g, &normalized, perspective);
        int piece_count = 0;
        for (int i = 0; i < CELL_COUNT; ++i) piece_count += normalized.board[i] != 0;
        float win_rate = 0.5f, danger = 0.0f;
        memory_query(normalized.board, &win_rate, &danger,
            g_ctx.total_games_played, piece_count, perspective);
    }

    memory_apply_model_scores(g, qvals);

    /* Probe 只发布「当前在线方」的动作分数。未开启按方路由时就是历史共享
       q_net；开启按方路由时必须是当前绑定方自己的 online，否则会把冻结对手
       或另一方的 Q 值误标成整个系统的部署模型输出。 */
    const bool is_online_net =
        (net == &agent->q_net) ||
        (independent_training::indep_active(agent) &&
         net == independent_training::agent_online_net(agent, independent_training::indep_bound(agent)));
    if (is_online_net) {
        /* Probe Q means the action score actually used by the deployed model,
           including its memory component. */
        memcpy(agent->probe_q_values, qvals, sizeof(agent->probe_q_values));
        agent->probe_current_player = g->turn;
        memcpy(agent->probe_last_input, input, sizeof(agent->probe_last_input));
        agent->probe_input_valid = 1;
    }

    if (eps > 0 && prng_float(rng) < eps) {
        return moves[prng_int(rng, cnt)];
    }

    return greedy_legal_action(g, qvals, rng);
}

int select_action(DQNAgent* agent, const GameState* g, int use_opponent,
    float eps_override, PRNGState* rng, int observe_memory) {
    int moves[CELL_COUNT], cnt = get_legal_moves(g, moves);
    if (cnt == 0) return -1;
    if (rng == NULL) rng = &agent->train_rng;

    float eps = (eps_override >= 0) ? eps_override : agent->epsilon;

    float input[CELL_COUNT + 1];
    state_to_input(g, input);
    float qvals[CELL_COUNT];
    /* D1：按方路由。未开启独立网络时保持历史语义（我方用共享 q_net，
       显式冻结对手用 opponent_net）逐位不变。
       独立模式下，共享 q_net 从播种那一刻起就不再更新，继续读它等于拿
       **预训练种子权重**去评估/展示，所以「我方行动」必须按实际行动的
       物理棋色取该方自己的 online 网络（g->turn 就是行动方）。 */
    NeuralNet* net = nullptr;
    if (independent_training::indep_active(agent)) {
        const int turn = (g->turn == 0 || g->turn == 1) ? g->turn : 0;
        net = use_opponent
            /* D2：以该物理棋色出场的**冻结对手**。共享 agent->opponent_net 在
               独立模式下从播种起就不再更新，等于永远对着预训练种子下棋；每方
               各有一份由 Trainer 从该方 online 周期刷新的副本。 */
            ? independent_training::agent_opponent_net(agent,
                independent_training::side_from_physical(turn))
            : independent_training::agent_online_net(agent,
                independent_training::side_from_physical(turn));
    } else {
        net = use_opponent ? &agent->opponent_net : &agent->q_net;
    }
    if (!net) return -1;

    // 超时检测（防卡死）
    LARGE_INTEGER start_time, current_time, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start_time);
    const double TIMEOUT_SEC = 2.0;

    forward_cpu(net, input, qvals);

    if (g_training_fault) return -1;

    QueryPerformanceCounter(&current_time);
    double elapsed = (double)(current_time.QuadPart - start_time.QuadPart) / freq.QuadPart;
    if (elapsed > TIMEOUT_SEC) {
        printf("[WARN] forward_cpu timeout %.3f sec, fallback to random\n", elapsed);
        return moves[prng_int(rng, cnt)];
    }

    // Observe official self-play positions for retention/pruning statistics.
    // IND-04：评测/对战（observe_memory==0）使用只读记忆查询，不更新留存统计。
    if (observe_memory && g_hot_config.memory_enabled && agent->seed_id < 0 && g_kb_count > 0) {
        int perspective = g->turn;  // 0=X, 1=O
        GameState normalized;
        normalize_state_to_self(g, &normalized, perspective);
        int piece_count = 0;
        for (int i = 0; i < CELL_COUNT; ++i) piece_count += normalized.board[i] != 0;
        float win_rate = 0.5f, danger = 0.0f;
        memory_query(normalized.board, &win_rate, &danger,
            g_ctx.total_games_played, piece_count, perspective);
    }

    memory_apply_model_scores(g, qvals);

    if (!use_opponent) {
        memcpy(agent->probe_q_values, qvals, sizeof(agent->probe_q_values));
        agent->probe_current_player = g->turn;
        memcpy(agent->probe_last_input, input, sizeof(agent->probe_last_input));
        agent->probe_input_valid = 1;
    }

    if (eps > 0 && prng_float(rng) < eps) {
        // softmax exploration
        float temp = 0.5f;
        float max_q = qvals[moves[0]];
        for (int i = 1; i < cnt; i++)
            if (qvals[moves[i]] > max_q) max_q = qvals[moves[i]];
        float sum = 0.0f;
        float probs[CELL_COUNT];
        for (int i = 0; i < cnt; i++) {
            probs[i] = expf((qvals[moves[i]] - max_q) / temp);
            sum += probs[i];
        }
        if (sum == 0.0f) return moves[0];
        float r = prng_float(rng) * sum;
        float cum = 0.0f;
        for (int i = 0; i < cnt; i++) {
            cum += probs[i];
            if (cum >= r) return moves[i];
        }
        return moves[cnt - 1];
    }

    return greedy_legal_action(g, qvals, rng);
}

void store_experience(DQNAgent* agent, const float state[], int action, float reward,
    const float next_state[], int done, int player) {
    if (action < 0 || action >= CELL_COUNT) return;
    Experience* buf = (player == 0) ? agent->replay_black : agent->replay_white;
    int* idx = (player == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
    int* cnt = (player == 0) ? &agent->replay_black_count : &agent->replay_white_count;
    Experience* exp = &buf[*idx];
    memcpy(exp->state, state, sizeof(float) * (CELL_COUNT + 1));
    exp->action = action; exp->reward = reward;
    memcpy(exp->next_state, next_state, sizeof(float) * (CELL_COUNT + 1));
    exp->done = done;
    *idx = (*idx + 1) % REPLAY_CAPACITY;
    if (*cnt < REPLAY_CAPACITY) (*cnt)++;

#if SYMMETRY_ENABLE
    {
        float flipped_state[CELL_COUNT + 1];
        float flipped_next[CELL_COUNT + 1];
        int flipped_action = action;
        flip_state(state, flipped_state, &flipped_action);
        flip_state(next_state, flipped_next, NULL);
        int flipped_player = 1 - player;
        const StepOutcomeConfig& outcome = g_episode_outcome_config ?
            *g_episode_outcome_config : g_hot_config.step_outcome;
        /* 独立网络下**必须**用同色旋转，绝不颜色互换。
           原因（可核验）：flip_state 是「180° 旋转 + 棋子符号取反 + turn 取反」，
           也就是颜色互换；颜色互换会把白方视角的样本写进 flipped_player = 1-player
           的缓冲区 = 黑方回放，而 experience_side() 只看缓冲区指针，于是这些样本
           会去更新黑方网络。后果两条：颜色互换版本进了黑方缓冲（HANDOFF:62 要求
           已启用的颜色保持旋转增强继续保留），且白方奖励（如白和 +0.05）被原样
           复制进黑方缓冲，污染非对称收益。
           未开启按方路由时保持历史行为逐位不变。 */
        const bool same_colour_rotation = outcome.enabled || independent_training::indep_active(agent);
        if (same_colour_rotation) {
            // GPT6-Astra: rotation preserves physical roles and legal black-first
            // counts. Colour swapping would alias asymmetric labels after canonicalization.
            for (int j=0;j<CELL_COUNT;++j) {
                flipped_state[CELL_COUNT-1-j]=state[j];
                flipped_next[CELL_COUNT-1-j]=next_state[j];
            }
            flipped_state[CELL_COUNT]=state[CELL_COUNT];
            flipped_next[CELL_COUNT]=next_state[CELL_COUNT];
            flipped_action=CELL_COUNT-1-action;
            flipped_player=player;
        }
        Experience* buf2 = (flipped_player == 0) ? agent->replay_black : agent->replay_white;
        int* idx2 = (flipped_player == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
        int* cnt2 = (flipped_player == 0) ? &agent->replay_black_count : &agent->replay_white_count;
        Experience* exp2 = &buf2[*idx2];
        memcpy(exp2->state, flipped_state, sizeof(float) * (CELL_COUNT + 1));
        exp2->action = flipped_action;
        /* 这是“颜色互换 + 旋转”的当前行动者等价增强：翻转前后奖励都
           表示正在落子的一方是否受益，因此必须保持同号。取反会把每条
           好经验复制成坏经验，直接抵消策略学习。 */
        /* The side-balance controller scales only positive terminal rewards.
           Symmetry swaps colours, so translate that scaled reward to the
           destination side instead of copying a white bonus into black (or
           vice versa).  Non-terminal shaping and draws are unchanged. */
        exp2->reward = reward;
        if (done && reward > 0.0f && player >= 0 && player < 2 &&
            g_terminal_win_reward_scale[player] > 0.0f) {
            exp2->reward = reward / g_terminal_win_reward_scale[player] *
                g_terminal_win_reward_scale[flipped_player];
        }
        memcpy(exp2->next_state, flipped_next, sizeof(float) * (CELL_COUNT + 1));
        exp2->done = done;
        *idx2 = (*idx2 + 1) % REPLAY_CAPACITY;
        if (*cnt2 < REPLAY_CAPACITY) (*cnt2)++;
    }
#endif
}

int validate_experience(const Experience* experience) {
    if (!experience || experience->action < 0 || experience->action >= CELL_COUNT ||
        !isfinite(experience->reward) ||
        (experience->done != 0 && experience->done != 1)) return 0;

    int response_action = -1;
    for (int i = 0; i < CELL_COUNT; ++i) {
        const float before = experience->state[i];
        const float after = experience->next_state[i];
        if (!isfinite(before) || !isfinite(after) ||
            (before != -1.0f && before != 0.0f && before != 1.0f) ||
            (after != -1.0f && after != 0.0f && after != 1.0f)) return 0;
        if (i != experience->action && before != after) {
            if (response_action >= 0 || before != 0) return 0;
            response_action = i;
        }
    }
    const float state_turn = experience->state[CELL_COUNT];
    const float next_turn = experience->next_state[CELL_COUNT];
    if (!isfinite(state_turn) || !isfinite(next_turn) ||
        (state_turn != 0.0f && state_turn != 1.0f) ||
        (next_turn != state_turn && next_turn != 1.0f - state_turn)) return 0;
    if (experience->state[experience->action] != 0.0f) return 0;
    const float placed = state_turn == 0.0f ? 1.0f : -1.0f;
    if (experience->next_state[experience->action] != placed) return 0;

    GameState next = {};
    for (int i = 0; i < CELL_COUNT; ++i)
        next.board[i] = experience->next_state[i] == 1.0f ? 1 :
            (experience->next_state[i] == -1.0f ? 2 : 0);
    next.turn = (int)next_turn;
    const int terminal = check_winner(&next) != 0 || is_full(&next);
    if (response_action >= 0) {
        /* 两 ply 样本：本方落子 + 对手真实回应。两种合法形态：
           (a) 回应结束比赛 -> 终局桥接：terminal && done==1，不 bootstrap；
           (b) 回应未结束   -> 跨到本方下一个决策点：!terminal && done==0。
           两者都必须 next_turn == state_turn（对手回应后重新轮到本方），
           且中间棋盘（去掉回应那一子）不能已经终局。 */
        if (next_turn != state_turn ||
            experience->next_state[response_action] != -placed) return 0;
        GameState intermediate = next;
        intermediate.board[response_action] = 0;
        if (check_winner(&intermediate) || is_full(&intermediate)) return 0;
        intermediate.board[experience->action] = 0;
        if (check_winner(&intermediate)) return 0;
        if (terminal) {
            if (!experience->done) return 0;   /* 终局桥接必须标 done */
        } else {
            if (experience->done) return 0;    /* 非终局不得标 done（不伪造终局） */
        }
    }
    return experience->done == terminal;
}

/* replay_from_buffer 细分计时 (微秒, IPC 推送后重置) */
volatile double g_replay_sample_us = 0, g_replay_forward_us = 0, g_replay_target_us = 0;
volatile double g_replay_train_us = 0, g_replay_sync_us = 0;
volatile int g_replay_count = 0;

float replay_from_buffer(DQNAgent* agent, Experience* buf, int count, int* idx) {
    using namespace independent_training;
    g_replay_attempt_count++;
    if (count < BATCH_SIZE / 2 || !buf || !agent) return 0.0f;

    if (!idx || count < 0 || count > REPLAY_CAPACITY) return 0.0f;
    int write_pos = *idx;

    /* 计时 */
    LARGE_INTEGER pf_rq; QueryPerformanceFrequency(&pf_rq);
    LARGE_INTEGER t_sample_start; QueryPerformanceCounter(&t_sample_start);
    if (write_pos < 0 || write_pos >= REPLAY_CAPACITY) return 0.0f;

    /* 每个批次必须整体属于同一物理棋色。回放缓冲区 -> 棋色只能来自显式映射，
       不猜、不取巧：独立网络开启后指向未知缓冲区的批次直接判失败。 */
    Side side = Side::Black;
    if (!experience_side(buf, agent, &side)) {
        if (indep_active(agent)) {
            printf("[REPLAY] Batch does not belong to any physical side; training stopped.\n");
            training_fail_detail(2, "replay_side", 0);
            return 0.0f;
        }
    }
    const int side_id = side_to_physical(side);
    NeuralNet* const online_net = agent_online_net(agent, side);
    NeuralNet* const target_net = agent_target_net(agent, side);
    if (!online_net || !target_net) { training_fail(2); return 0.0f; }
    const int indep = indep_active(agent);
    const float train_lr = indep ? agent->indep->learning_rate_or(side, g_learning_rate)
                                 : g_learning_rate;

    int batch_size = (count >= BATCH_SIZE) ? BATCH_SIZE : count;

    // 批量收集：只采样，forward 和 target 计算随后批量处理
    float batch_states[BATCH_SIZE][CELL_COUNT + 1];
    float batch_next_states[BATCH_SIZE][CELL_COUNT + 1];
    int batch_actions[BATCH_SIZE];
    float batch_rewards[BATCH_SIZE];
    int batch_dones[BATCH_SIZE];
    int sample_count = 0;

    anchor_buffer_init(agent);
    const StepOutcomeConfig& outcome = g_episode_outcome_config ?
        *g_episode_outcome_config : g_hot_config.step_outcome;
    int eligible_anchors[ANCHOR_CAPACITY], eligible_count=0;
    for(int i=0;i<g_anchor_count;++i) {
        int black=0,white=0;
        for(int j=0;j<CELL_COUNT;++j){black+=g_anchor[i].puzzle.board[j]==1;white+=g_anchor[i].puzzle.board[j]==2;}
        if(!outcome.enabled || black-white==g_anchor[i].puzzle.turn) eligible_anchors[eligible_count++]=i;
    }
    for (int b = 0; b < batch_size; b++) {
        Experience* exp;
        Experience local_anchor;
        /* 记忆锚点：按可配置比例混入固定战术锚点（对抗灾难性遗忘）。
           锚点存原始题面，采样时按当前课程阶段动态重算奖励，
           与 live 封堵奖励保持一致（修复 phase1 冻结 reward≡0）。 */
        if (eligible_count > 0 && prng_float(&agent->train_rng) < ANCHOR_SAMPLE_RATIO) {
            const AnchorEntry* a = &g_anchor[eligible_anchors[prng_int(&agent->train_rng,eligible_count)]];
            state_to_input(&a->puzzle, local_anchor.state);
            GameState next = a->puzzle;
            make_move(&next, a->action);
            const bool anchor_terminal = check_winner(&next) != 0 || is_full(&next);
            if (indep && !anchor_terminal) {
                /* 独立模式的契约要求非终局样本跨两 ply（本方决策 -> 本方下一个决策）。
                   锚点原本是一 ply；这里补一个**真实**的对手回应（从该局面的合法
                   着法里取），把端点挪到本方下一个决策点。锚点自身的 reward 不变
                   （它描述的仍是锚点那一手），因此防守训练信号没有被削弱或删除。
                   未开启按方路由时保持历史一 ply 行为逐位不变。 */
                int reply_moves[CELL_COUNT];
                const int rn = get_legal_moves(&next, reply_moves);
                if (rn > 0) {
                    const int reply = reply_moves[prng_int(&agent->train_rng, rn)];
                    make_move(&next, reply);
                }
            }
            state_to_input(&next, local_anchor.next_state);
            local_anchor.action = a->action;
            local_anchor.done = check_winner(&next) != 0 || is_full(&next);
            const int phase = get_curriculum_phase(g_ctx.total_games_played);
            const int player = a->puzzle.turn;
            /* 重算 reward 不得污染正式 per-episode 统计（reward_stats/reward_usage）。 */
            float saved_stats[7];
            int saved_usage[7];
            memcpy(saved_stats, agent->reward_stats, sizeof(saved_stats));
            memcpy(saved_usage, agent->reward_usage, sizeof(saved_usage));
            local_anchor.reward = compute_reward(agent, &a->puzzle, &next, a->action, player,
                phase, g_ctx.total_games_played, 0);
            memcpy(agent->reward_stats, saved_stats, sizeof(saved_stats));
            memcpy(agent->reward_usage, saved_usage, sizeof(saved_usage));
            exp = &local_anchor;
        } else {
            int r;
            if (g_uniform_replay_sampling) {
                r = prng_int(&agent->train_rng, count);
            } else {
                float p = prng_float(&agent->train_rng);
                if (p < 0.5f) {
                    int start = (count > BATCH_SIZE) ? count - BATCH_SIZE : 0;
                    r = start + prng_int(&agent->train_rng, count - start);
                }
                else if (p < 0.8f) {
                    int end = (count > BATCH_SIZE) ? count - BATCH_SIZE : count;
                    r = prng_int(&agent->train_rng, end);
                }
                else {
                    r = prng_int(&agent->train_rng, count);
                }
            }

            /* r 是从最旧到最新的逻辑下标。缓冲区填满后，idx 指向下一写入位置，
               也就是当前最旧元素；必须转换为物理下标。 */
            int physical = (count == REPLAY_CAPACITY) ?
                (write_pos + r) % REPLAY_CAPACITY : r;
            exp = &buf[physical];
        }
        /* Replay is immutable training evidence.  Never silently rewrite a
           damaged terminal flag: doing so bootstraps from an invalid state and
           makes resume behavior depend on unrelated CRT rand() calls. */
        if (!validate_experience(exp)) {
            printf("[REPLAY] Invalid experience; training stopped.\n");
            training_fail_detail(2, "replay_validation", 0);
            return 0.0f;
        }

        memcpy(batch_states[sample_count], exp->state, sizeof(float) * (CELL_COUNT + 1));
        memcpy(batch_next_states[sample_count], exp->next_state, sizeof(float) * (CELL_COUNT + 1));
        batch_actions[sample_count] = exp->action;
        batch_rewards[sample_count] = exp->reward;
        batch_dones[sample_count] = exp->done;
        sample_count++;
    }

    if (sample_count == 0) return 0.0f;

    /* 采样计时结束 */
    LARGE_INTEGER t_sample_end; QueryPerformanceCounter(&t_sample_end);
    g_replay_sample_us += (double)(t_sample_end.QuadPart - t_sample_start.QuadPart) / (double)pf_rq.QuadPart * 1e6;

    /* Forward 计时开始 */
    LARGE_INTEGER t_fwd_start; QueryPerformanceCounter(&t_fwd_start);

    // 批量 forward（GPU 一次同步/批，省 3*32 次逐样本同步）
    float batch_qvals[BATCH_SIZE][CELL_COUNT];
    float batch_next_target[BATCH_SIZE][CELL_COUNT];
    float batch_next_q[BATCH_SIZE][CELL_COUNT];
    if (gpu_net_active()) {
        /* BUG-071: 检查 GPU forward 返回值，失败时回退 CPU */
        /* 四句柄驻留：三次前向都取**该批次所属方**的句柄（online / target）。 */
        int gpu_ok = 1;
        if (gpu_net_forward_batch_side(side_id, 0, (const float*)batch_states, sample_count, (float*)batch_qvals) != 0) gpu_ok = 0;
        if (gpu_ok && gpu_net_forward_batch_side(side_id, 1, (const float*)batch_next_states, sample_count, (float*)batch_next_target) != 0) gpu_ok = 0;
        if (gpu_ok && gpu_net_forward_batch_side(side_id, 0, (const float*)batch_next_states, sample_count, (float*)batch_next_q) != 0) gpu_ok = 0;
        if (!gpu_ok) {
            /* GPU 驻留训练时 CPU 网络可能落后。只有成功下载最新参数后才能回退。 */
            if (gpu_net_download_agent(agent) != 0) {
                printf("[GPU] Forward and parameter recovery failed; stopping training.\n");
                training_fail(1);
                return 0.0f;
            }
            printf("[GPU] forward batch 失败，已同步参数并回退 CPU\n");
            for (int b = 0; b < sample_count; b++) {
                forward_cpu(online_net, batch_states[b], batch_qvals[b]);
                forward_cpu(target_net, batch_next_states[b], batch_next_target[b]);
                forward_cpu(online_net, batch_next_states[b], batch_next_q[b]);
            }
        }
    } else {
        for (int b = 0; b < sample_count; b++) {
            forward_cpu(online_net, batch_states[b], batch_qvals[b]);
            forward_cpu(target_net, batch_next_states[b], batch_next_target[b]);
            forward_cpu(online_net, batch_next_states[b], batch_next_q[b]);
        }
    }

    /* Forward 计时结束, Target 计时开始 */
    LARGE_INTEGER t_fwd_end; QueryPerformanceCounter(&t_fwd_end);
    g_replay_forward_us += (double)(t_fwd_end.QuadPart - t_fwd_start.QuadPart) / (double)pf_rq.QuadPart * 1e6;
    LARGE_INTEGER t_tgt_start; QueryPerformanceCounter(&t_tgt_start);

    if (g_training_fault) return 0.0f;
    for (int b = 0; b < sample_count; ++b) {
        for (int a = 0; a < CELL_COUNT; ++a) {
            if (!isfinite(batch_qvals[b][a]) || !isfinite(batch_next_q[b][a]) ||
                !isfinite(batch_next_target[b][a])) {
                training_fail_detail(2, "replay_forward_nonfinite", b * CELL_COUNT + a);
                return 0.0f;
            }
        }
    }
    // CPU 算 target（棋盘规则用批量下载的 Q 值）
    float batch_targets[BATCH_SIZE];
    float batch_raw_targets[BATCH_SIZE];
    for (int b = 0; b < sample_count; b++) {
        float target = batch_rewards[b];
        /* bootstrap_sign 口径：
             0        = 终局不 bootstrap
             独立模式 = +plies（正号，端点是自己下一个决策点的己方价值）
             遗留模式 = +1（单人课程，同行动者）/ -1（正式交替，零和目标取负） */
        int bootstrap_sign = 0;
        if (!batch_dones[b]) {
            GameState next_g;
            for (int i = 0; i < CELL_COUNT; i++) {
                if (batch_next_states[b][i] == 1.0f)      next_g.board[i] = 1;
                else if (batch_next_states[b][i] == -1.0f) next_g.board[i] = 2;
                else                                       next_g.board[i] = 0;
            }
            next_g.turn = (int)(batch_next_states[b][CELL_COUNT] + 0.5f);
            int legal_moves[CELL_COUNT];
            int cnt_moves = get_legal_moves(&next_g, legal_moves);
            if (cnt_moves > 0) {
                int best_action = legal_moves[0];
                float best_q = batch_next_q[b][best_action];
                for (int i = 1; i < cnt_moves; i++) {
                    int a = legal_moves[i];
                    if (batch_next_q[b][a] > best_q) {
                        best_q = batch_next_q[b][a];
                        best_action = a;
                    }
                }
                if (indep) {
                    /* 独立网络：端点(next_state)必须是本方下一个决策点，
                       跨度 2 ply，折扣 γ²，**正号**（己方收益 -> 己方价值）。
                       合法着法也就在本方决策点上取 max，类型自洽。
                       形态不符即判失败，绝不按新公式静默重算旧样本。 */
                    const int plies = experience_semantics::observed_plies(
                        batch_states[b], batch_next_states[b]);
                    const int span_rc = experience_semantics::check_replay_span(
                        batch_states[b], batch_next_states[b], 0);
                    if (span_rc != 0 || plies != 2) {
                        printf("[REPLAY] side=%s sample %d spans %d plies; independent-network "
                            "replay requires a 2-ply span (own move + real opponent reply).\n",
                            independent_training::side_name(side), b, plies);
                        training_fail_detail(2, "replay_span", span_rc);
                        return 0.0f;
                    }
                    bootstrap_sign = plies;
                    target += experience_semantics::bootstrap_discount_for_plies(plies, GAMMA) *
                        batch_next_target[b][best_action];
                } else {
                    /* 遗留共享网络模式：逐位保持历史行为。
                       正式对弈会切换行动者，next Q 是对手价值，零和目标取负；
                       单人预训练会把 turn 归零，前后仍为同一行动者，应取正。
                       用状态中的 turn 明确区分，避免修正式训练时破坏预训练。 */
                    int state_turn = (int)(batch_states[b][CELL_COUNT] + 0.5f);
                    int next_turn = (int)(batch_next_states[b][CELL_COUNT] + 0.5f);
                    bootstrap_sign = (state_turn == next_turn) ? 1.0f : -1.0f;
                    target += bootstrap_sign * GAMMA * batch_next_target[b][best_action];
                }
            } else if (indep) {
                /* 独立模式：非终局却没有合法着法是不可能的棋盘，判失败。 */
                training_fail_detail(2, "replay_span", 3);
                return 0.0f;
            }
        }
        if (!isfinite(target)) {
            training_fail_detail(2, "replay_target_nonfinite", b);
            return 0.0f;
        }
        batch_raw_targets[b] = target;
        batch_targets[b] = fmaxf(-DQN_TARGET_ABS_LIMIT,
            fminf(DQN_TARGET_ABS_LIMIT, target));
        /* TQ 监控：记录未裁剪 raw_tq 与裁剪结果，防止 [-5,5] 裁剪掩盖 target 发散。 */
        tq_record_sample(target, batch_targets[b],
            (buf == agent->replay_black), batch_dones[b] != 0, bootstrap_sign);
    }
    tq_flush_batch();

    /* Filter per sample, not per batch.  The forward values above are the
       exact online-network predictions that would enter Huber loss.  Compact
       accepted samples in place; rejected samples never reach CPU or GPU
       backward kernels, while useful neighbours in the same batch survive. */
    int accepted_count = 0;
    int rejected_count = 0;
    float max_rejected_loss = 0.0f;
    for (int b = 0; b < sample_count; ++b) {
        const float error = batch_qvals[b][batch_actions[b]] - batch_targets[b];
        const float abs_error = fabsf(error);
        const float sample_loss = abs_error <= 0.5f
            ? 0.5f * error * error
            : 0.5f * (abs_error - 0.25f);
        if (!isfinite(sample_loss)) {
            training_fail_detail(2, "replay_loss_nonfinite", b);
            return 0.0f;
        }
        const int raw_tq_outlier = fabsf(batch_raw_targets[b]) > DQN_RAW_TQ_REJECT_LIMIT;
        /* A large but finite online Q is a model error, not corrupt evidence:
           keep it so bounded Huber gradients can repair that state. Raw TQ is
           generated by the target network and is safe to quarantine before it
           is hidden by the [-5,5] clamp. */
        if (sample_loss > DQN_REPLAY_LOSS_REJECT_LIMIT || raw_tq_outlier) {
            ++rejected_count;
            if (sample_loss > max_rejected_loss) max_rejected_loss = sample_loss;
            continue;
        }
        if (accepted_count != b) {
            memcpy(batch_states[accepted_count], batch_states[b], sizeof(batch_states[0]));
            memcpy(batch_qvals[accepted_count], batch_qvals[b], sizeof(batch_qvals[0]));
            batch_actions[accepted_count] = batch_actions[b];
            batch_targets[accepted_count] = batch_targets[b];
        }
        ++accepted_count;
    }
    if (rejected_count) {
        g_replay_loss_reject_count += (unsigned long long)rejected_count;
        const unsigned long long n = g_replay_loss_reject_count;
        if (n <= 4 || (n & (n - 1)) == 0) {
            printf("[REPLAY] Skipped %d high-loss sample(s), max=%.6g, total=%llu; no backprop for rejected data.\n",
                rejected_count, max_rejected_loss, n);
        }
    }
    sample_count = accepted_count;
    if (sample_count == 0) {
        agent->probe_last_loss = 0.0f;
        g_replay_reject_count++;
        return 0.0f;
    }

    /* Target 计时结束, Training 计时开始 */
    LARGE_INTEGER t_tgt_end; QueryPerformanceCounter(&t_tgt_end);
    g_replay_target_us += (double)(t_tgt_end.QuadPart - t_tgt_start.QuadPart) / (double)pf_rq.QuadPart * 1e6;
    LARGE_INTEGER t_train_start; QueryPerformanceCounter(&t_train_start);

    float avg_loss = 0.0f;
    /* 按方提交：黑批复用黑方 online/target 与黑方按方状态，白批同理。
       未开启按方路由时 side 被忽略，行为与历史 train_agent_batch 逐字节一致。 */
    const int train_rc = train_agent_batch_side(agent, side_id, (const float*)batch_states,
        batch_actions, batch_targets, sample_count, train_lr, &avg_loss);
    if (train_rc == 0) { g_replay_reject_count++; return 0.0f; }   /* 致命失败 */
    if (train_rc == 2) {                                           /* 安全拒绝：不推进有效训练统计 */
        agent->probe_last_loss = 0.0f;
        g_replay_reject_count++;
        return 0.0f;
    }

    agent->probe_last_loss = avg_loss;

    /* Training 计时结束 */
    LARGE_INTEGER t_train_end; QueryPerformanceCounter(&t_train_end);
    g_replay_train_us += (double)(t_train_end.QuadPart - t_train_start.QuadPart) / (double)pf_rq.QuadPart * 1e6;
    g_replay_count++;

    int replay_type = (buf == agent->replay_black) ? 0 : 1;
    /* Loss is an average over the accepted batch. Log Q/TQ using that same
       population instead of an arbitrary first sample, so the three curves
       describe one coherent batch (Loss remains Huber, not TQ-Q). */
    double q_sum = 0.0, tq_sum = 0.0;
    for (int b = 0; b < sample_count; ++b) {
        q_sum += batch_qvals[b][batch_actions[b]];
        tq_sum += batch_targets[b];
    }
    log_train_step(agent, avg_loss, (float)(q_sum / sample_count),
        (float)(tq_sum / sample_count), replay_type);
    agent->global_step += sample_count;
    g_replay_commit_count++;

    return avg_loss;
}



static void truncate_failed_bin_write(FILE* f, int64_t record_size) {
    if (!f || record_size <= 0) return;
    const int64_t header_size = 24;
    int64_t pos = _ftelli64(f);
    if (pos < header_size) return;
    int64_t aligned = header_size + ((pos - header_size) / record_size) * record_size;
    _chsize_s(_fileno(f), aligned);
    _fseeki64(f, 0, SEEK_END);
    clearerr(f);
}

int open_dqn_logs(DQNAgent* agent) {
    // 打开日志文件并设置大缓冲区，减少磁盘 IO（通信兼日志 → logs\）
    agent->log_game = fopen(dqn_log_path("game.dat").c_str(), "a");
    agent->log_train = fopen(dqn_log_path("train.dat").c_str(), "a");
    if (agent->log_game) {
        setvbuf(agent->log_game, NULL, _IOFBF, 1024 * 1024);  // 1MB 缓冲区
    }
    if (agent->log_train) {
        setvbuf(agent->log_train, NULL, _IOFBF, 1024 * 1024);
    }
    if (!agent->log_game || !agent->log_train)
        printf("警告：无法创建日志文件\n");

    // All metric writers share version checks, tail repair and header creation.
    agent->log_steps_bin = metric_log_open_append(dqn_log_path("metric_steps.bin").c_str(), 4);
    agent->log_game_bin = metric_log_open_append(dqn_log_path("metric_game.bin").c_str(), 1);
    agent->log_loss_bin = metric_log_open_append(dqn_log_path("metric_loss.bin").c_str(), 3);
    agent->log_eps_bin = metric_log_open_append(dqn_log_path("metric_epsilon.bin").c_str(), 1);

    return agent->log_game && agent->log_train && agent->log_steps_bin &&
        agent->log_game_bin && agent->log_loss_bin && agent->log_eps_bin;
}

void init_dqn(DQNAgent* agent) { init_dqn_ex(agent, 1); }

void init_dqn_ex(DQNAgent* agent, int open_logs) {
    /* calloc/malloc 出来的 DQNAgent 里 indep 可能是垃圾指针，先显式清空。
       未调用 indep_ensure 之前，所有生产路径保持历史共享语义。 */
    independent_training::indep_reset(agent);
    agent->competency=CompetencyState{};agent->competency_opponents=nullptr;
    init_net(&agent->q_net);
    copy_net(&agent->target_net, &agent->q_net);
    copy_net(&agent->opponent_net, &agent->q_net);

    // 分配回放缓冲区
    agent->replay_black = (Experience*)malloc(REPLAY_CAPACITY * sizeof(Experience));
    agent->replay_white = (Experience*)malloc(REPLAY_CAPACITY * sizeof(Experience));
    if (!agent->replay_black || !agent->replay_white) {
        printf("[INIT] 回放缓冲区分配失败!\n");
    }
    else {
        memset(agent->replay_black, 0, REPLAY_CAPACITY * sizeof(Experience));
        memset(agent->replay_white, 0, REPLAY_CAPACITY * sizeof(Experience));
    }
    agent->replay_black_idx = 0;
    agent->replay_white_idx = 0;
    agent->replay_black_count = 0;
    agent->replay_white_count = 0;
    agent->epsilon = 1.0f;
    agent->step_counter = 0;
    agent->target_hard_counter = 0;   /* IND-05 */
    agent->concentration_factor = 0.5f;
    agent->prev_performance = 0.0f;
    agent->current_perspective = 0;
    agent->probe_last_loss = 0.0f;
    memset(agent->probe_q_values, 0, sizeof(agent->probe_q_values));
    agent->probe_current_player = 0;
    memset(agent->probe_last_input, 0, sizeof(agent->probe_last_input));
    agent->probe_input_valid = 0;

    if (open_logs) open_dqn_logs(agent);
    agent->global_step = 0;

    for (int i = 0; i < 7; i++) {
        agent->reward_weights[i] = 1.0f;
        agent->reward_usage[i] = 0;
        agent->reward_stats[i] = 0.0f;
    }
    agent->reward_episode = -1;

    // 黑白平衡控制器：全零窗口 + 倍率 1.0（重启默认无偏）
    memset(&agent->side_balance, 0, sizeof(agent->side_balance));
    agent->side_balance.reward_scale[0] = 1.0f;
    agent->side_balance.reward_scale[1] = 1.0f;

    // 对手池（最小化内存占用）
    agent->opponent_pool_size = 1;
    agent->opponent_pool = (NeuralNet*)malloc(agent->opponent_pool_size * sizeof(NeuralNet));
    agent->opponent_pool_count = 0;
    agent->opponent_pool_next = 0;
    if (agent->opponent_pool) {
        copy_net(&agent->opponent_pool[0], &agent->q_net);
        agent->opponent_pool_count = 1;
        agent->opponent_pool_next = 1;
    }

    prng_init(&agent->train_rng, 123456789u);
    prng_init(&agent->eval_rng, 223456789u);
    agent->ema_winrate = 0.0f;
    agent->ema_alpha = 0.85f;
    agent->depth_adjust_counter = 0;
    agent->depth_adjust_threshold = 3;
    agent->first_move_buf_idx = 0;
    memset(agent->first_move_buf, 0, sizeof(agent->first_move_buf));
    agent->current_shaping_scale = 0.0f;
    agent->seed_id = -1;
    /* Fresh agents write the current V10 checkpoint format.  Keeping this at
       a legacy version makes runtime diagnostics report an old format until
       the first reload, even though save_checkpoint() already emits V10. */
    agent->checkpoint_version = 10;
    agent->pretrain_consolidation_target = agent->pretrain_stable_passes = 0;
    memset(&agent->skill_tracker, 0, sizeof(agent->skill_tracker));
}

void free_dqn(DQNAgent* agent) {
    delete agent->competency_opponents;agent->competency_opponents=nullptr;
    if (agent->log_game) fclose(agent->log_game);
    if (agent->log_train) fclose(agent->log_train);
    if (agent->log_steps_bin) fclose(agent->log_steps_bin);
    if (agent->log_game_bin)  fclose(agent->log_game_bin);
    if (agent->log_loss_bin)  fclose(agent->log_loss_bin);
    if (agent->log_eps_bin)   fclose(agent->log_eps_bin);
    if (agent->replay_black) free(agent->replay_black);
    if (agent->replay_white) free(agent->replay_white);
    if (agent->opponent_pool) free(agent->opponent_pool);
    independent_training::indep_free(agent);
}

void soft_update_target(DQNAgent* agent) {
    // GPU 优先：target_net 驻留 GPU 时直接在 GPU 上更新
    if (gpu_net_active()) {
        if (gpu_net_soft_update() == 0) return;
        // An in-place target update can fail after partial modification. Never
        // save that state or apply a second update to the uncertain target.
        printf("[TARGET] GPU target update failed; requesting safe shutdown.\n");
        training_fail(1);
        return;
    }
    float* t = (float*)&agent->target_net;
    float* q = (float*)&agent->q_net;
    int n = sizeof(NeuralNet) / sizeof(float);
    for (int i = 0; i < n; i++)
        t[i] = TAU * q[i] + (1.0f - TAU) * t[i];
    if (gpu_net_active() && gpu_net_upload_agent(agent) != 0)
        printf("[TARGET] CPU 回退 soft update 后上传失败\n");
}

/* 按方 target 同步。物理棋色 side 决定更新哪一方的 target：
   只把该方 online 的参数混入该方 target，另一方的 online/target 一个字节都不动。
   GPU 路径要求会话已绑定到同一方，否则拒绝而不是猜测。
   返回 0 成功，负值失败。 */
int update_target_side(DQNAgent* agent, int side, int soft) {
    using namespace independent_training;
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    if (!agent) return -1;
    if (!indep_active(agent)) {
        /* 未开启按方路由：保持历史单一 target 语义。 */
        if (soft) { soft_update_target(agent); return 0; }
        if (gpu_net_active()) return gpu_net_hard_update();
        copy_net(&agent->target_net, &agent->q_net);
        return 0;
    }
    const int on_gpu = gpu_net_active();
    if (on_gpu) {
        /* 先把该方驻留句柄取回，保证 CPU 影子与设备同代再混合。
           四句柄驻留下按 (side, which) 直取，不需要绑定。 */
        if (gpu_net_download_side(agent, side) != 0) return -3;
    }
    NeuralNet* t = agent_target_net(agent, s);
    const NeuralNet* q = agent_online_net(agent, s);
    if (!t || !q) return -1;
    if (soft) {
        float* tf = (float*)t;
        const float* qf = (const float*)q;
        const int n = (int)(sizeof(NeuralNet) / sizeof(float));
        for (int i = 0; i < n; ++i) tf[i] = TAU * qf[i] + (1.0f - TAU) * tf[i];
    } else {
        copy_net(t, q);
    }
    if (on_gpu) {
        const int rc = soft ? gpu_net_soft_update_side(side) : gpu_net_hard_update_side(side);
        if (rc != 0) return -4;
    }
    if (!soft) agent->indep->note_hard_target_sync(s);
    return 0;
}

void add_to_opponent_pool(DQNAgent* agent) {
    if (!agent->opponent_pool) return;
    int last_idx = opponent_pool_latest_index(agent->opponent_pool_next, agent->opponent_pool_size);
    if (last_idx < 0) return;
    if (agent->opponent_pool_count > 0) {
        if (memcmp(&agent->opponent_pool[last_idx], &agent->target_net,
            sizeof(NeuralNet)) == 0) return;
    }
    int idx = agent->opponent_pool_next % agent->opponent_pool_size;
    copy_net(&agent->opponent_pool[idx], &agent->target_net);
    agent->opponent_pool_next++;
    if (agent->opponent_pool_count < agent->opponent_pool_size)
        agent->opponent_pool_count++;
}

int save_opponent_pool(const DQNAgent* agent, const char* filename) {
    if (!agent || !agent->opponent_pool || agent->opponent_pool_size <= 0 ||
        agent->opponent_pool_count < 0 || agent->opponent_pool_count > agent->opponent_pool_size ||
        agent->opponent_pool_next < 0) return 0;
    char tmp_name[512];
    int n = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (n < 0 || n >= (int)sizeof(tmp_name)) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) { printf("[WARN] 无法创建对手池文件 %s\n", filename); return 0; }
    int meta[3] = { agent->opponent_pool_size, agent->opponent_pool_count,
                    agent->opponent_pool_next };
    int ok = fwrite(meta, sizeof(int), 3, f) == 3 &&
        fwrite(agent->opponent_pool, sizeof(NeuralNet), agent->opponent_pool_count, f) ==
        (size_t)agent->opponent_pool_count && fflush(f) == 0;
    if (ok) ok = (_commit(_fileno(f)) == 0);
    if (fclose(f) != 0) ok = 0;
    if (!ok || !MoveFileExA(tmp_name, filename, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(tmp_name); printf("[WARN] 对手池原子保存失败: %s\n", filename);
        return 0;
    }
    return 1;
}

int load_opponent_pool(DQNAgent* agent, const char* filename) {
    if (!agent || !agent->opponent_pool) return 0;
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    int meta[3];
    if (fread(meta, sizeof(int), 3, f) != 3) { fclose(f); return 0; }
    int pool_size = meta[0], pool_count = meta[1], pool_next = meta[2];
    /* BUG-077: 验证 pool_size/pool_count/pool_next 范围 */
    if (pool_size <= 0 || pool_size != agent->opponent_pool_size ||
        pool_count > pool_size || pool_count < 0 || pool_next < 0) {
        printf("[LOAD] 对手池元数据无效 (size=%d count=%d next=%d)\n", pool_size, pool_count, pool_next);
        fclose(f); return 0;
    }
    _fseeki64(f, 0, SEEK_END);
    int64_t bytes = _ftelli64(f);
    _fseeki64(f, sizeof(meta), SEEK_SET);
    if (bytes != (int64_t)sizeof(meta) + (int64_t)pool_count * sizeof(NeuralNet)) {
        fclose(f); return 0;
    }
    NeuralNet* incoming = (NeuralNet*)calloc((size_t)(pool_count ? pool_count : 1), sizeof(NeuralNet));
    if (!incoming) { fclose(f); return 0; }
    if (fread(incoming, sizeof(NeuralNet), pool_count, f) != (size_t)pool_count) {
        free(incoming); fclose(f); return 0;
    }
    const float* values = (const float*)incoming;
    for (size_t i = 0; i < (size_t)pool_count * sizeof(NeuralNet) / sizeof(float); ++i) {
        if (!isfinite(values[i])) { free(incoming); fclose(f); return 0; }
    }
    memcpy(agent->opponent_pool, incoming, (size_t)pool_count * sizeof(NeuralNet));
    free(incoming);
    agent->opponent_pool_count = pool_count;
    /* pool_next 在运行中是单调计数器，合法值可以超过池容量；加载后规范化为下一写槽。 */
    agent->opponent_pool_next = pool_next % pool_size;
    fclose(f);
    printf("[LOAD] 对手池已从 %s 加载 (count=%d)\n", filename, pool_count);
    return 1;
}

NeuralNet* select_opponent_from_pool(DQNAgent* agent, PRNGState* rng) {
    if (agent->opponent_pool_count == 0) return &agent->opponent_net;
    int idx = prng_int(rng, agent->opponent_pool_count);
    return &agent->opponent_pool[idx];
}

float evaluate_agent_selfplay(DQNAgent* agent, int games) {
    if (!agent || games <= 0) return NAN;
    int wins = 0, draws = 0, losses = 0;
    float orig_eps = agent->epsilon;
    agent->epsilon = 0.0f;
    for (int g = 0; g < games; g++) {
        GameState game; reset_game(&game);
        int completed = 0;
        for (int step = 0; step < CELL_COUNT; ++step) {
            int action = select_action(agent, &game, 0, 0.0f, &agent->eval_rng, 0);
            if (g_training_fault || !is_legal_move(&game, action)) {
                if (!g_training_fault && agent == g_current_agent) training_fail(2);
                agent->epsilon = orig_eps;
                return NAN;
            }
            make_move(&game, action);
            int winner = check_winner(&game);
            if (winner) {
                if (winner == 1) wins++; else losses++;
                completed = 1;
                break;
            }
            if (is_full(&game)) { draws++; completed = 1; break; }
        }
        if (!completed) {
            if (agent == g_current_agent) training_fail(2);
            agent->epsilon = orig_eps;
            return NAN;
        }
    }
    agent->epsilon = orig_eps;
    return (float)wins / games;
}

float compute_reward(DQNAgent* agent, const GameState* game, const GameState* next_game,
    int action, int player, int curriculum_phase, int episode, int step) {
    int player_idx = player + 1;
    int opponent_idx = (player == 0) ? 2 : 1;
    int winner = check_winner(next_game);
    const StepOutcomeConfig& outcome = g_episode_outcome_config ?
        *g_episode_outcome_config : g_hot_config.step_outcome;
    int total_plies=0;
    for(int j=0;j<CELL_COUNT;++j) total_plies+=next_game->board[j]!=0;
    const float quality = curriculum_phase>=1 && (winner || is_full(next_game)) ?
        step_outcome_bonus(outcome,player,winner,total_plies,CELL_COUNT) : 0.f;
    if (winner != 0) {
        float base = (winner == player_idx) ? WIN_REWARD : LOSE_REWARD;
        // 四连终局奖励：完成四连的一方获得额外奖励（优先级高于终局）
        if (curriculum_phase >= 1) {
            if (winner == player_idx) {
                base += WIN_REWARD;  // 赢方双倍奖励
            }
        }
        /* Preserve the physical-side multiplier on the original winning base.
           Terminal opponent-reply credit uses the unchanged loss base. */
        if (winner == player_idx && player >= 0 && player < 2)
            base *= g_terminal_win_reward_scale[player];
        // R-Four records the existing terminal win reward; it is not an extra bonus.
        if (winner == player_idx) {
            agent->reward_stats[3] += base;
            agent->reward_usage[3]++;
        }
        return base + quality;
    }
    if (is_full(next_game)) {
        return DRAW_REWARD + quality;
    }

    // ================================================================
    // 奖励衰减表：每个奖励类型在当前阶段的衰减系数
    // 索引：0=活二, 1=活三, 2=冲三, 3=冲四, 4=双三, 5=封堵获胜, 6=封堵活三/双三
    // ================================================================
    float decay[7] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    switch (curriculum_phase) {
    case 0:
        // Phase 0：无中间奖励
        return 0.0f;

    case 1:
        /* 预训练已经教授“四连是什么”。正式 Phase 1 只保留胜/负/和
           终局反馈，让网络先把胜利模式迁移到真实对抗，不再用活二牵引。 */
        return 0.0f;

    case 2:
        // Phase 2：只加入封堵对手一步必胜
        decay[0] = 0.0f;   // 活二永久停用
        decay[1] = 0.0f;   // 活三禁用
        decay[2] = 0.0f;   // 冲三禁用
        decay[3] = 0.0f;   // 冲四禁用
        decay[4] = 0.0f;   // 双三禁用
        decay[5] = 1.0f;   // 封堵获胜满值
        decay[6] = 0.0f;   // 封堵活三/双三禁用
        break;

    case 3:
        // Phase 3：加入高阶战术奖励
        decay[0] = 0.0f;   // 活二永久停用
        decay[1] = 1.0f;   // 活三满值
        decay[2] = 1.0f;   // 冲三满值
        decay[3] = 1.0f;   // 冲四满值
        decay[4] = 1.0f;   // 双三满值
        decay[5] = 0.7f;   // 封堵获胜保留70%
        decay[6] = 1.0f;   // 封堵活三/双三满值
        break;

    case 4:
    default:
        // Phase 4：保留高阶战术，活二仍不再使用
        decay[0] = 0.0f;
        decay[1] = 1.0f;   // 活三满值
        decay[2] = 1.0f;   // 冲三满值
        decay[3] = 1.0f;   // 冲四满值
        decay[4] = 1.0f;   // 双三满值
        decay[5] = 0.5f;   // 封堵获胜保留50%
        decay[6] = 1.0f;   // 封堵活三/双三满值
        break;
    }

    float reward = 0.0f;
    int r = action / BOARD_SIZE, c = action % BOARD_SIZE;

    // ================================================================
    // 各奖励类型计算（乘以对应衰减系数）
    // ================================================================

    // 封堵获胜检测（Phase 2/3/4）
    if (curriculum_phase >= 2) {
        int opp_win = find_opponent_winning_move(game, opponent_idx);
        if (opp_win >= 0 && find_opponent_winning_move(next_game, opponent_idx) < 0) {
            float bonus = BLOCK_WIN_BONUS_BASE * agent->current_shaping_scale *
                decay[5] * g_hot_config.block_win_scale;
            ACCUMULATE_REWARD(5, bonus);
        }
    }

    // 高阶奖励（Phase 3/4）
    if (curriculum_phase >= 3) {
        if (is_live_three(game, r, c, player_idx)) {
            float bonus = OPEN_THREE_BONUS_BASE * agent->current_shaping_scale *
                decay[1] * g_hot_config.advanced_reward_scale;
            ACCUMULATE_REWARD(1, bonus);
        }
        if (is_threat_three(game, r, c, player_idx)) {
            float bonus = THREAT_THREE_BONUS_BASE * agent->current_shaping_scale *
                decay[2] * g_hot_config.advanced_reward_scale;
            ACCUMULATE_REWARD(2, bonus);
        }
        if (detect_double_threat(game, r, c, player_idx)) {
            float bonus = DOUBLE_THREE_BONUS_BASE * agent->current_shaping_scale *
                decay[4] * g_hot_config.advanced_reward_scale;
            ACCUMULATE_REWARD(4, bonus);
        }
        int opp_double = find_opponent_double_threat(game, opponent_idx);
        int opp_three = find_opponent_open_three(game, opponent_idx);
        if (action == opp_double || action == opp_three) {
            float bonus = BLOCK_OPEN_THREE_BONUS_BASE * agent->current_shaping_scale *
                decay[6] * g_hot_config.advanced_reward_scale;
            ACCUMULATE_REWARD(6, bonus);
        }
    }

    return reward;
}

void adjust_reward_weights(DQNAgent* agent, int win, int last_player) {
    // agent 始终执黑 (player=0)，黑赢 (win==1) 为胜，否则为负
    int is_lost = (win != 0 && win != 1) ? 1 : 0;

    float total_recent = (float)(g_wins[0] + g_wins[1] + g_wins[2]);
    float recent_lose_rate = 0.0f;
    if (total_recent > 0)
        recent_lose_rate = 1.0f - (float)g_wins[0] / total_recent;
    float LOSE_RATE_LOCK = 0.85f;

    int any_decayed = 0;
    if (is_lost) {
        float total = 0.0f;
        // Terminal four-in-a-row is display-only; its reward is not weighted.
        for (int i = 0; i < 7; i++) if (i != 3) total += agent->reward_stats[i];
        if (total > 0.001f && recent_lose_rate < LOSE_RATE_LOCK) {
            for (int i = 0; i < 7; i++) {
                if (i == 3) continue;
                float ratio = agent->reward_stats[i] / total;
                if (ratio > REWARD_WEIGHT_THRESHOLD) {
                    agent->reward_weights[i] *= REWARD_WEIGHT_DECAY;
                    if (agent->reward_weights[i] < REWARD_WEIGHT_MIN)
                        agent->reward_weights[i] = REWARD_WEIGHT_MIN;
                    any_decayed = 1;
                }
            }
        }
    }
    if (!any_decayed || recent_lose_rate >= LOSE_RATE_LOCK) {
        for (int i = 0; i < 7; i++) {
            if (i == 3) continue;
            agent->reward_weights[i] *= REWARD_WEIGHT_RECOVER;
            if (agent->reward_weights[i] > REWARD_WEIGHT_MAX)
                agent->reward_weights[i] = REWARD_WEIGHT_MAX;
        }
        if (recent_lose_rate >= LOSE_RATE_LOCK) {
            for (int i = 0; i < 7; i++) {
                if (i == 3) continue;
                agent->reward_weights[i] *= REWARD_WEIGHT_RECOVER;
                if (agent->reward_weights[i] > REWARD_WEIGHT_MAX)
                    agent->reward_weights[i] = REWARD_WEIGHT_MAX;
            }
            printf("\r[自检] 败率过高 (%.1f%%)，强制回升奖励权重                                        \n", recent_lose_rate * 100);
        }
    }
}

EvaluationStatus record_minimax_games_for_episode(DQNAgent* agent, int episode, int depth,
    int* bw, int* bd, int* bl,
    int* ww, int* wd, int* wl,
    float* avg_win, float* avg_draw, float* avg_lose,
    FILE* log_file, EvaluationCancel cancelled, void* context,
    EvaluationProgress progress) {
    /* Minimax promotion evaluates the complete deployed model: learned network
       plus its knowledge-memory bank. */
    const float epsilons[] = { 0.0f, 0.01f, 0.02f };
    const int total_games = 3 * 2 * 200;
    int counts[3][2][3] = {}; // mode, DQN side, win/draw/loss
    int game_id = 0;
    for (int m = 0; m < 3; m++) {
        float eps = epsilons[m];
        for (int side = 0; side < 2; ++side) {
            for (int g = 0; g < 200; g++) {
                GameState game; reset_game(&game);
                for (int step = 0; step < CELL_COUNT; ++step) {
                    if (g_training_fault) return EVALUATION_FAULT;
                    if (cancelled && cancelled(context)) return EVALUATION_CANCELLED;
                    int turn = game.turn, action = -1;
                    if (turn == side)
                        action = select_action(agent, &game, 0, eps, &agent->eval_rng, 0);
                    else
                        action = minimax_best_move(&game, depth, 0.0f, &agent->eval_rng);
                    // Inference/search can fail or receive a stop request. Never
                    // pass its sentinel to the board, or publish partial scores.
                    if (g_training_fault) return EVALUATION_FAULT;
                    if (cancelled && cancelled(context)) return EVALUATION_CANCELLED;
                    if (action < 0 || action >= CELL_COUNT || game.board[action]) {
                        training_fail(2);
                        return EVALUATION_FAULT;
                    }
                    make_move(&game, action);
                    int winner = check_winner(&game);
                    int done = (winner || is_full(&game));
                    if (log_file)
                        fprintf(log_file, "%d %d %d %d %d %.6f %d\n",
                            episode, game_id, step, turn, action, 0.0f, done);
                    if (done) {
                        ++counts[m][side][winner == 0 ? 1 : (winner == side + 1 ? 0 : 2)];
                        break;
                    }
                }
                game_id++;
                if (progress) progress(game_id, total_games, context);
            }
        }
    }
    if (g_training_fault) return EVALUATION_FAULT;
    if (cancelled && cancelled(context)) return EVALUATION_CANCELLED;
    for (int m = 0; m < 3; ++m) {
        bw[m] = counts[m][0][0]; bd[m] = counts[m][0][1]; bl[m] = counts[m][0][2];
        ww[m] = counts[m][1][0]; wd[m] = counts[m][1][1]; wl[m] = counts[m][1][2];
        avg_win[m] = (float)(bw[m] + ww[m]) / 400.0f;
        avg_draw[m] = (float)(bd[m] + wd[m]) / 400.0f;
        avg_lose[m] = 1.0f - avg_win[m] - avg_draw[m];
    }
    return EVALUATION_COMPLETE;
}

void log_game_step(DQNAgent* agent, int episode, int step, int player,
    int action, float reward, int done) {
    if (!agent->log_game && !agent->log_steps_bin) return;   /* 文本或 bin 任一可用即写入 */

    /* BUG-044: 使用 INIT_ONCE 保证线程安全的一次性初始化 */
    static CRITICAL_SECTION cs_game;
    static INIT_ONCE init_once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&init_once, [](PINIT_ONCE, PVOID, PVOID*) -> BOOL {
        InitializeCriticalSection(&cs_game); return TRUE;
    }, NULL, NULL);

    EnterCriticalSection(&cs_game);
    if (agent->log_game) {
        if (agent->seed_id >= 0) {
            fprintf(agent->log_game, "%d %d %d %d %d %.6f %d\n",
                agent->seed_id, episode, step, player, action, reward, done);
        }
        else {
            fprintf(agent->log_game, "%d %d %d %d %.6f %d\n",
                episode, step, player, action, reward, done);
        }
    }
    // Batch flush every 1000 episodes (not every game)
    // This prevents duplicate logs when interrupted
    // Batch flush every 1000 episodes (not every game)
    // This prevents duplicate logs when interrupted
    /* BUG-031: done 时统一 flush 所有日志（text + bin 共享事务边界） */
    if (done) {
        if (agent->log_game) fflush(agent->log_game);
    }
    /* bin 写入：metric_steps.bin（每步：ep, step, player, action, reward, done = 24 字节）
       BUG-022: 单次 fwrite 写完整记录（连续结构体，避免两次写之间崩溃导致半记录） */
    /* 种群文本日志带 seed_id，而现有 bin schema 没有 seed 字段。把种群
       记录写进主训练 bin 会造成 episode 回跳并污染所有图表/回放。 */
    if (agent->seed_id < 0 && agent->log_steps_bin) {
        struct { int32_t ep; int32_t step; float fields[4]; } rec;
        rec.ep = episode;
        rec.step = step;
        rec.fields[0] = (float)player;
        rec.fields[1] = (float)action;
        rec.fields[2] = reward;
        rec.fields[3] = (float)done;
        if (fwrite(&rec, sizeof(rec), 1, agent->log_steps_bin) != 1) {
            /* 写入失败，按 24 字节文件头之后的记录边界截断。 */
            truncate_failed_bin_write(agent->log_steps_bin, sizeof(rec));
        }
        if (done) fflush(agent->log_steps_bin);
    }
    if (agent->seed_id < 0 && done && agent->log_game_bin) {
        /* BUG-025: 单次 fwrite + 检查返回值 */
        struct { int32_t ep; int32_t step; float len; } rec = { episode, 0, (float)(step + 1) };
        if (fwrite(&rec, sizeof(rec), 1, agent->log_game_bin) != 1) {
            truncate_failed_bin_write(agent->log_game_bin, sizeof(rec));
        }
        fflush(agent->log_game_bin);
    }
    LeaveCriticalSection(&cs_game);
}

// Force flush game log (call on graceful exit)
void log_game_flush(DQNAgent* agent) {
    if (agent->log_game) fflush(agent->log_game);
    if (agent->log_steps_bin) fflush(agent->log_steps_bin);
    if (agent->log_game_bin) fflush(agent->log_game_bin);
    if (agent->log_loss_bin) fflush(agent->log_loss_bin);
    if (agent->log_eps_bin)  fflush(agent->log_eps_bin);
}

void log_train_step(DQNAgent* agent, float loss, float q_value, float target_q,
    int replay_type) {
    if (!agent->log_train && !agent->log_loss_bin) return;

    /* BUG-044: 使用 INIT_ONCE 保证线程安全的一次性初始化 */
    static CRITICAL_SECTION cs_train;
    static INIT_ONCE init_once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&init_once, [](PINIT_ONCE, PVOID, PVOID*) -> BOOL {
        InitializeCriticalSection(&cs_train); return TRUE;
    }, NULL, NULL);

    EnterCriticalSection(&cs_train);
    if (agent->log_train) {
        if (agent->seed_id >= 0) {
            fprintf(agent->log_train, "%d %lld %.6f %.6f %.6f %.6f %d\n",
                agent->seed_id, agent->global_step, (double)loss, (double)q_value,
                (double)target_q, (double)agent->epsilon, replay_type);
        }
        else {
            fprintf(agent->log_train, "%lld %.6f %.6f %.6f %.6f %d\n",
                agent->global_step, (double)loss, (double)q_value, (double)target_q,
                (double)agent->epsilon, replay_type);
        }
        if (agent->global_step % 100 == 0) fflush(agent->log_train);
    }
    /* bin 图表统一使用真实 episode 作为 X；global_step 仍完整保留在 train.dat。
       play_episode 内 g_ctx 是已完成局数，当前正在训练的是下一局。 */
    {
        int64_t chart_ep = (agent->seed_id < 0) ? (int64_t)g_ctx.total_games_played + 1
                                                : agent->global_step;
        if (chart_ep < 1) chart_ep = 1;
        if (chart_ep > INT32_MAX) chart_ep = INT32_MAX;
        int32_t ep32 = (int32_t)chart_ep;
        /* BUG-025: 单次 fwrite + 检查返回值 */
        if (agent->seed_id < 0 && agent->log_loss_bin) {
            struct { int32_t ep; int32_t rt; float fields[3]; } rec = { ep32, replay_type, {loss, q_value, target_q} };
            if (fwrite(&rec, sizeof(rec), 1, agent->log_loss_bin) != 1) {
                truncate_failed_bin_write(agent->log_loss_bin, sizeof(rec));
            }
            if (agent->global_step % 100 == 0) fflush(agent->log_loss_bin);
        }
        if (agent->seed_id < 0 && agent->log_eps_bin) {
            struct { int32_t ep; int32_t step; float eps; } rec = { ep32, 0, (float)agent->epsilon };
            if (fwrite(&rec, sizeof(rec), 1, agent->log_eps_bin) != 1) {
                truncate_failed_bin_write(agent->log_eps_bin, sizeof(rec));
            }
            if (agent->global_step % 100 == 0) fflush(agent->log_eps_bin);
        }
    }
    LeaveCriticalSection(&cs_train);
}

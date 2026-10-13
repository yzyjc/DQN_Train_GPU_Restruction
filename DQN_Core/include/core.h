// core.h
#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "stage_progress.h"
#include "step_outcome.h"
#include "competency_state.h"

// ========================================================================
// 标准库
// ========================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <io.h>      /* _chsize, _fileno, _ftelli64, _fseeki64 */

// ========================================================================
// Windows API
// ========================================================================
#include <windows.h>
#include <conio.h>
#include <sys/stat.h>
#include <process.h>
#include <tlhelp32.h>
#include <mmsystem.h>
#include <xinput.h>

// ========================================================================
// 链接系统库
// ========================================================================
#pragma comment(lib, "User32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "xinput.lib")

// ========================================================================
// 全部宏定义
// ========================================================================
#define BOARD_SIZE              5
#define CELL_COUNT              (BOARD_SIZE * BOARD_SIZE)
#define WIN_STREAK              4

#define HIDDEN1_SIZE            1024
#define HIDDEN2_SIZE            512
#define HIDDEN3_SIZE            256
#define BATCH_SIZE              32
#define REPLAY_CAPACITY         20000
#define GAMMA                   0.95f
#define TARGET_UPDATE_STEP      1
#define EVAL_GAMES              200
#define TRAIN_STEP_EACH         4

#define PHASE1_EPISODES         100000
#define PHASE2_EPISODES         100000
#define PHASE3_EPISODES         100000
#define PHASE4_EPISODES         100000

/* Phase 0 现在是独立的单人预训练，不占正式自对弈的 global episode。
   正式训练从 global_ep=0 直接进入 Phase 1。 */
#define PHASE0_END              0
#define PHASE1_END              (PHASE1_EPISODES)
#define PHASE2_END              (PHASE1_EPISODES + PHASE2_EPISODES)
#define PHASE3_END              (PHASE1_EPISODES + PHASE2_EPISODES + PHASE3_EPISODES)
#define PHASE4_END              (PHASE1_EPISODES + PHASE2_EPISODES + PHASE3_EPISODES + PHASE4_EPISODES)

#define EPS_MAX                 1.0f
#define EPS_MIN                 0.10f
#define EPS_PHASE0              1.0f
#define EPS_PHASE1_START        1.0f
#define EPS_PHASE1_END          0.4f
#define EPS_PHASE2_START        0.4f
#define EPS_PHASE2_END          0.2f
#define EPS_PHASE3_START        0.2f
#define EPS_PHASE3_END          0.1f
#define EPS_PHASE4_START        0.1f
#define EPS_PHASE4_END          0.05f

#define REWARD_SHAPING_PHASE0   0.0f
#define REWARD_SHAPING_PHASE1   0.0f   /* 正式 Phase 1：纯终局奖励，无人工战术引导 */
#define REWARD_SHAPING_PHASE2   0.3f
#define REWARD_SHAPING_PHASE3   1.0f
#define REWARD_SHAPING_PHASE4   1.0f

#define POOL_PROB_PHASE0        0.0f
#define POOL_PROB_PHASE1        0.0f
#define POOL_PROB_PHASE2        0.5f
#define POOL_PROB_PHASE3        1.0f
#define POOL_PROB_PHASE4        1.0f

#define MINIMAX_PROB_PHASE0     0.0f
#define MINIMAX_PROB_PHASE1     0.05f
#define MINIMAX_PROB_PHASE2     0.30f
#define MINIMAX_PROB_PHASE3     0.5f
#define MINIMAX_PROB_PHASE4     1.0f

#define TARGET_UPDATE_HARD      0
#define TARGET_UPDATE_SOFT      1
#define TARGET_UPDATE_PHASE0    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE1    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE2    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE3    TARGET_UPDATE_SOFT
#define TARGET_UPDATE_PHASE4    TARGET_UPDATE_SOFT

#define FIRST_MOVE_ENABLE_PHASE0 0
#define FIRST_MOVE_ENABLE_PHASE1 0
#define FIRST_MOVE_ENABLE_PHASE2 0
#define FIRST_MOVE_ENABLE_PHASE3 0
#define FIRST_MOVE_ENABLE_PHASE4 1

#define TRAIN_ENABLE_PHASE0     0
#define TRAIN_ENABLE_PHASE1     1
#define TRAIN_ENABLE_PHASE2     1
#define TRAIN_ENABLE_PHASE3     1
#define TRAIN_ENABLE_PHASE4     1

#define LR_DECAY_INTERVAL       20000
#define LR_DECAY_FACTOR         0.8f

#define REWARD_WEIGHT_THRESHOLD 0.30f
#define REWARD_WEIGHT_DECAY     0.95f
#define REWARD_WEIGHT_MIN       0.1f
#define REWARD_WEIGHT_RECOVER   1.01f
#define REWARD_WEIGHT_MAX       1.0f

#define WIN_REWARD              0.5f
#define LOSE_REWARD             -0.5f
#define DRAW_REWARD             0.0f

#define BLOCK_WIN_BONUS_BASE         0.08f
#define LIVE_TWO_BONUS_BASE          0.02f
#define OPEN_THREE_BONUS_BASE        0.02f
#define THREAT_THREE_BONUS_BASE      0.01f
#define DOUBLE_THREE_BONUS_BASE      0.03f
#define FOUR_THREAT_BONUS_BASE       0.025f
#define BLOCK_OPEN_THREE_BONUS_BASE  0.03f

#define MAX_LIVE_TWO_PER_GAME         3
#define MAX_OPEN_THREE_PER_GAME       1
#define MAX_THREAT_THREE_PER_GAME     2
#define MAX_FOUR_THREAT_PER_GAME      1
#define MAX_DOUBLE_THREAT_PER_GAME    1
#define MAX_BLOCK_WIN_PER_GAME        1
#define MAX_BLOCK_OPEN_THREE_PER_GAME 2

#define FIRST_MOVE_WIN_BONUS    0.01f
#define FIRST_MOVE_DRAW_BONUS   0.005f

#define SYMMETRY_ENABLE         1
#define WARMUP_EPISODES         5000
#define TAU                     0.003f
#define FIRST_MOVE_BATCH_SIZE   16

#define ATTENTION_HEADS         4
#define ATTENTION_EMBED         32
#define ATTENTION_HEAD_DIM      (ATTENTION_EMBED / ATTENTION_HEADS)
#define TOKEN_EMBED_DIM         16

#define POP_SIZE                8
#define ROUND_EPISODES          1000
#define TOTAL_ROUNDS            4

#define PIPE_NAME               "\\\\.\\pipe\\dqn_control_pipe"
#define MAX_LOG_LINES           8
#define global_minimax_depth    g_ctx.minimax_depth

// 奖励累加宏
#define ACCUMULATE_REWARD(type_idx, bonus) \
    do { \
        if (bonus > 0 && agent->reward_usage[type_idx] < get_reward_limit(type_idx)) { \
            agent->reward_stats[type_idx] += bonus; \
            reward += agent->reward_weights[type_idx] * bonus; \
            agent->reward_usage[type_idx]++; \
        } \
    } while(0)

// ========================================================================
// 记忆系统参数（V5.2 - Patch-based）
// ========================================================================
#define FEATURE_DIM             128
/* Each perspective owns its capacity; retain existing oversized black files. */
#define CACHE_CAPACITY_PER_SIDE 6000
#define KB_CAPACITY_PER_SIDE    1024
#define CACHE_CAPACITY          (2 * CACHE_CAPACITY_PER_SIDE)
#define KB_CAPACITY             (2 * KB_CAPACITY_PER_SIDE)
#define PROMOTE_MIN_TOTAL       50
#define PROMOTE_WIN_RATE        0.60f
#define LOSE_RATE_THRESHOLD     0.60f
#define PROMOTE_INTERVAL        50
#define DEFAULT_TIMEOUT         8000
#define EXTENDED_TIMEOUT        20000
#define PERMANENT_TOTAL         300
#define PERMANENT_DEMOTION      3
#define MAX_EPISODE_STEPS       100

// Patch参数：3x3窗口，5x5棋盘有9个patch
#define PATCH_WINDOW            3
#define PATCH_SIZE              (PATCH_WINDOW * PATCH_WINDOW)  // 9
#define PATCH_COUNT             ((BOARD_SIZE - PATCH_WINDOW + 1) * (BOARD_SIZE - PATCH_WINDOW + 1))  // 9
#define PATCH_SIM_THRESHOLD     0.6f   // 单个patch相似度阈值
#define BOARD_MATCH_THRESHOLD   0.5f   // 整体匹配阈值（至少5/9个patch匹配）

// 特征组成：前 NET_FEATURE_DIM 维为网络特征，后 META_DIM 维为元信息
#define META_DIM                4
#define NET_FEATURE_DIM         (FEATURE_DIM - META_DIM)  // 124

// ========================================================================
// 类型定义
// ========================================================================

/* ============================================================================
   黑白独立网络：仅前置声明 + 不透明指针。
   DQNAgent 定义在 extern "C" 块内，里面不能出现 C++ 命名空间类型的完整定义，
   所以这里只放一个 ::independent_training::IndependentNetworks* 指针，
   真实容器与按方接口见 independent_networks.h。
   未分配（nullptr）时生产路径逐字节保持历史「单一共享 q_net/target_net」语义。
   ========================================================================= */
class IndependentNetworks;

#ifdef __cplusplus
extern "C" {
#endif

    // ---- 棋盘 ----
    typedef struct {
        int board[CELL_COUNT];
        int turn;
    } GameState;

    // ---- 随机数 ----
    typedef struct {
        uint32_t state;
    } PRNGState;

    // ---- 神经网络（V4.0 - 池化层输出128维） ----
    typedef struct {
        float token_embed[3][TOKEN_EMBED_DIM];
        float pos_embed[CELL_COUNT][TOKEN_EMBED_DIM];
        float w1[HIDDEN1_SIZE][FEATURE_DIM];              // FC1: FEATURE_DIM -> HIDDEN1
        float b1[HIDDEN1_SIZE];
        float w2[HIDDEN2_SIZE][HIDDEN1_SIZE];             // FC2: HIDDEN1 -> HIDDEN2
        float b2[HIDDEN2_SIZE];
        float w3[HIDDEN3_SIZE][HIDDEN2_SIZE];             // FC3: HIDDEN2 -> HIDDEN3
        float b3[HIDDEN3_SIZE];
        float w4[CELL_COUNT][HIDDEN3_SIZE];               // FC4: HIDDEN3 -> OUTPUT
        float b4[CELL_COUNT];
        float attn_wq[ATTENTION_EMBED][TOKEN_EMBED_DIM];
        float attn_bq[ATTENTION_EMBED];
        float attn_wk[ATTENTION_EMBED][TOKEN_EMBED_DIM];
        float attn_bk[ATTENTION_EMBED];
        float attn_wv[ATTENTION_EMBED][TOKEN_EMBED_DIM];
        float attn_bv[ATTENTION_EMBED];
        float attn_wo[TOKEN_EMBED_DIM][ATTENTION_EMBED];
        float attn_bo[TOKEN_EMBED_DIM];
        float attn_q[ATTENTION_EMBED];
        float attn_k[ATTENTION_EMBED];
        float attn_v[ATTENTION_EMBED];
        float attn_scores[ATTENTION_HEADS][ATTENTION_HEAD_DIM][ATTENTION_HEAD_DIM];
        float attn_output[TOKEN_EMBED_DIM];
        float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
        float attn_context[CELL_COUNT][TOKEN_EMBED_DIM];

        // 可学习池化层（输出 FEATURE_DIM 维，即128维）
        float pool_w[FEATURE_DIM][CELL_COUNT * TOKEN_EMBED_DIM];
        float pool_b[FEATURE_DIM];
        float flat_cache[CELL_COUNT * TOKEN_EMBED_DIM];
    } NeuralNet;

    // ---- DQN 经验 ----
    typedef struct {
        float state[CELL_COUNT + 1];
        int action;
        float reward;
        float next_state[CELL_COUNT + 1];
        int done;
    } Experience;

/* ---- 黑白平衡控制器（V11 随 checkpoint 持久化，恢复边界标尺非平稳） ----
   一个可训练 q_net 供双方共用。滚动窗口记录在线网络执黑/执白的得分
   （win=1, draw=0.5, loss=0），对终局获胜奖励做小幅、缓慢、保均值的调整。
   完整状态（窗口 + 计数 + 倍率）随 checkpoint 原子提交，保证恢复后连续。 */
#define SIDE_BALANCE_WINDOW 1000
#pragma pack(push, 1)
    typedef struct {
        unsigned char online_side;
        float score;
    } SideBalanceSample;
    typedef struct {
        SideBalanceSample samples[SIDE_BALANCE_WINDOW];
        int write_pos;
        int size;
        int samples_per_side[2];
        float score_sum[2];
        int since_update;
        float reward_scale[2];
    } SideBalanceController;
#pragma pack(pop)

/* 技能状态 */
typedef enum {
    SKILL_LOCKED = 0,    /* 未解锁 */
    SKILL_LEARNING,      /* 学习中（不监控回归） */
    SKILL_MASTERED       /* 已毕业（监控回归阈值） */
} SkillState;
/* 技能状态管理 */
typedef struct {
    SkillState win1_state;
    SkillState empty_state;
    SkillState block1_state;
    float win1_baseline;       /* 首次测量值，作为回归基准 */
    float empty_baseline;
    float block1_baseline;
    float empty_steps_baseline;
    float empty_shortest_baseline;
        int initialized;          /* 0=none, 1=legacy, 2/3=older suites, 4=reachable suite */
} SkillTracker;

    // ---- DQN Agent ----
    typedef struct DQNAgent {
        NeuralNet q_net;
        NeuralNet target_net;
        NeuralNet opponent_net;
        Experience* replay_black;
        Experience* replay_white;
        int replay_black_idx;
        int replay_white_idx;
        int replay_black_count;
        int replay_white_count;
        float epsilon;
        int step_counter;
        int target_hard_counter;   /* IND-05：HARD 目标同步相位，随 checkpoint 持久化 */
        float concentration_factor;
        float prev_performance;
        FILE* log_game;
        FILE* log_train;
        /* bin 格式日志句柄（与文本句柄并存，metric_*.bin） */
        FILE* log_steps_bin;   /* metric_steps.bin: episode, step, player, action */
        FILE* log_game_bin;    /* metric_game.bin:  episode, step=0, game_length */
        FILE* log_loss_bin;    /* metric_loss.bin:  episode, step, loss, qv, tq */
        FILE* log_eps_bin;     /* metric_epsilon.bin: episode, step, epsilon */
        long long global_step;
        float reward_weights[7];
        float reward_stats[7];
        int reward_usage[7];
        int reward_episode;
        int current_perspective;
        /* Probe-only live diagnostics. These are runtime fields, not part of
           any checkpoint disk structure. */
        float probe_last_loss;
        float probe_q_values[CELL_COUNT];
        int probe_current_player;
        float probe_last_input[CELL_COUNT + 1];
        int probe_input_valid;
        NeuralNet* opponent_pool;
        int opponent_pool_size;
        int opponent_pool_count;
        int opponent_pool_next;
        PRNGState train_rng;
        PRNGState eval_rng;
        float ema_winrate;
        float ema_alpha;
        int depth_adjust_counter;
        int depth_adjust_threshold;
        struct {
            float state[CELL_COUNT + 1];
            int action;
            float target;
        } first_move_buf[FIRST_MOVE_BATCH_SIZE];
        int first_move_buf_idx;
        float current_shaping_scale;
        int seed_id;                // 种群训练的 Agent 编号，-1 表示主训练
        int checkpoint_version;
        int pretrain_consolidation_target;
        int pretrain_stable_passes;
        SkillTracker skill_tracker;
        SideBalanceController side_balance;   /* 黑白平衡控制器（V11 持久化） */
        /* 黑白独立网络所有权。nullptr = 未接入，走历史共享 q_net/target_net。
           非空时按方 online/target 与按方训练状态由该容器独占持有。
           指针不参与任何磁盘结构，刻意不写入 checkpoint ABI。 */
        struct IndependentNetworks* indep;
        CompetencyState competency;
        struct CompetencyOpponents* competency_opponents; // optional heap ownership, never a disk pointer
    } DQNAgent;

    // ---- 检查点元数据（磁盘格式） ----
#pragma pack(push, 1)
    typedef struct {
        int version;
        int magic;
        int phase;
        int episode;
        float epsilon;
        int minimax_depth;
        int replay_black_idx;
        int replay_white_idx;
        int replay_black_count;
        int replay_white_count;
        int wins[3];
        double elapsed_seconds;
        float concentration_factor;
        float prev_performance;
        NeuralNet q_net;
        NeuralNet target_net;
        NeuralNet opponent_net;
        float reward_weights[7];
        uint32_t train_rng_state;
        uint32_t eval_rng_state;
        float ema_winrate;
        int depth_adjust_counter;
        int opponent_pool_count;
        int opponent_pool_next;
        float current_shaping_scale;
        long long global_step;   /* BUG-015/068/085: V5 新增，V4 无此字段 */
    } CheckpointMeta;
    // V6 appends this block after the unchanged V5 metadata, before replay.
    // V6~V8 布局（无同步计时）。
    typedef struct {
        int input_encoding; // 2 = side-to-move canonical tokens
        float learning_rate;
        int consolidation_target;
        int stable_passes;
        SkillTracker skills;
    } CheckpointTrainingStateV6;
    // V9 起追加环境步与 HARD 同步相位，随 checkpoint 一起 CRC 校验 + 原子提交，
    // 消除独立 sidecar 因替换失败造成的「新网络 + 旧计时」混代。
    typedef struct {
        int input_encoding; // 2 = side-to-move canonical tokens
        float learning_rate;
        int consolidation_target;
        int stable_passes;
        SkillTracker skills;
        int step_counter;          // IND-05：环境步（训练间隔相位）
        int target_hard_counter;   // IND-05：HARD 目标同步相位
    } CheckpointTrainingState;
#pragma pack(pop)

    /* BUG-085: V4 旧文件的内存布局（有对齐填充），用于整体读取旧 checkpoint */
    typedef struct {
        int version;
        int magic;
        int phase;
        int episode;
        float epsilon;
        int minimax_depth;
        int replay_black_idx;
        int replay_white_idx;
        int replay_black_count;
        int replay_white_count;
        int wins[3];
        /* wins[3] 之后 double 需要 8 字节对齐，编译器插入 4 字节填充 */
        double elapsed_seconds;
        float concentration_factor;
        float prev_performance;
        NeuralNet q_net;
        NeuralNet target_net;
        NeuralNet opponent_net;
        float reward_weights[7];
        uint32_t train_rng_state;
        uint32_t eval_rng_state;
        float ema_winrate;
        int depth_adjust_counter;
        int opponent_pool_count;
        int opponent_pool_next;
        float current_shaping_scale;
        /* V4 没有 global_step */
    } CheckpointMetaV4;

    /* 部分历史 V4 文件在版本号仍为 4 时已经追加了 global_step。
       保留旧编译器自然对齐布局，加载时按实际元数据长度区分两种 V4。 */
    typedef struct {
        int version;
        int magic;
        int phase;
        int episode;
        float epsilon;
        int minimax_depth;
        int replay_black_idx;
        int replay_white_idx;
        int replay_black_count;
        int replay_white_count;
        int wins[3];
        double elapsed_seconds;
        float concentration_factor;
        float prev_performance;
        NeuralNet q_net;
        NeuralNet target_net;
        NeuralNet opponent_net;
        float reward_weights[7];
        uint32_t train_rng_state;
        uint32_t eval_rng_state;
        float ema_winrate;
        int depth_adjust_counter;
        int opponent_pool_count;
        int opponent_pool_next;
        float current_shaping_scale;
        long long global_step;
    } CheckpointMetaV4WithStep;

    // ---- 记忆系统 V5.0 ----
    // 对齐结果结构体
    typedef struct {
        int tx, ty;                     // 平移量
        int rotation;                   // 旋转: 0,1,2,3 (90°倍数)
        float score;                    // 综合得分
        float similarity;               // 重叠区域相似度
        float coverage;                 // 重叠比例
        int overlap_count;              // 重叠位置数
        int overlap_mask[CELL_COUNT];   // 哪些位置参与了重叠 (0/1)
        int transformed_probe[CELL_COUNT]; // 变换后的棋局
    } Alignment;

    // 双缓存条目
    typedef struct {
        int board[CELL_COUNT];  // 棋盘状态 (-1/0/1, 已归一化到X视角)
        int step;
    } CacheEntry;

    // 双缓存
    typedef struct {
        CacheEntry entries[MAX_EPISODE_STEPS];
        int count;
        int is_ready;
    } CacheBuffer;

    // 缓存库条目 (V5.2 Patch-based)
    typedef struct {
        float patches[PATCH_COUNT][PATCH_SIZE];  // 9个3x3 patch，每个9个值
        float patch_attention[PATCH_COUNT];      // 每个patch的注意力权重
        int sample_count;
        int total;
        int wins;
        int losses;
        int last_updated;
        int demotion_count;
        int step_sum;
        int step_count;
        int piece_count_sum;
        int piece_count_count;
        int perspective;
        int phase_bucket;
    } CacheStoreEntry;

    // 正式库条目 (V5.2 Patch-based)
    typedef struct {
        float patches[PATCH_COUNT][PATCH_SIZE];  // 9个3x3 patch，每个9个值
        float patch_attention[PATCH_COUNT];      // 每个patch的注意力权重
        int sample_count;
        int total;
        int wins;
        float win_rate;
        float danger_score;
        int last_queried;
        int query_count;
        int appearance_since_promotion;
        int demotion_count;
        int is_permanent;
        int timeout_threshold;
        int is_positive;
        float avg_step;
        float step_stddev;
        float player_bias;
        int black_wins;
        int white_wins;
        float avg_piece_count;
        int perspective;
        int phase_bucket;
        float urgency_score;
        int draws; // Added in memory V10; legacy KB entries have unknown draws (0).
    } KBEntry;

    // ---- 热配置 ----
    typedef struct {
        int train_repeat_per_step;
        float learning_rate_override;
        float live_two_scale;
        float block_win_scale;
        float advanced_reward_scale;
        float epsilon_override;
        time_t last_load_time;
        int config_version;
        // Memory system settings
        int memory_enabled;         // 0=disabled, 1=enabled
        float memory_scale;         // Q-value adjustment scale (0.0~1.0)
        float memory_confidence;    // Minimum confidence to use memory (0.0~1.0)
        StepOutcomeConfig step_outcome; // GPT6-Astra: opt-in role-specific terminal quality
        CompetencyConfig competency;
    } HotConfig;

    // ---- 训练上下文 ----
    typedef struct TrainingContext {
        int wins[3];
        time_t start_time;
        int minimax_depth;
        float learning_rate;
        float ema_winrate;
        float ema_alpha;
        int depth_adjust_counter;
        int depth_adjust_threshold;
        FILE* eval_log;
        FILE* minimax_log;
        int total_games_played;
        int save_interval;
        float best_winrate;
        int global_episode_offset;
    } TrainingContext;

    // ---- 种群进度 ----
    typedef struct {
        int round;
        int agent_idx;
        int completed;
    } PopulationProgress;

    // ========================================================================
    // 全局变量声明
    // ========================================================================
    extern float g_learning_rate;
    extern float g_ep_time, g_elapsed, g_remaining;   // 训练时间（IPC 推送）
    extern int g_pretrain_active;
    extern int g_pretrain_steps;
    extern float g_pretrain_avg_steps;
    extern float g_pretrain_success_rate;
    extern int g_pretrain_stable_passes;
    /* Checkpoint Minimax evaluation progress, published through Trainer IPC. */
    extern volatile LONG g_eval_active;
    extern volatile LONG g_eval_completed;
    extern volatile LONG g_eval_total;
    extern int g_wins[3];
    extern time_t g_start_time;
    extern TrainingContext g_ctx;
    extern DQNAgent* g_current_agent;
    extern HotConfig g_hot_config;

    extern HANDLE g_pipe;
    extern HANDLE g_pipe_thread;
    extern volatile int g_pipe_running;

    extern CacheStoreEntry g_memory_cache[];
    extern int g_cache_count;
    extern KBEntry g_knowledge_base[];
    extern int g_kb_count;

    extern void (*g_trainer_show_window_cb)(int show);   // 训练程序设置：IPC CONSOLE_SHOW/HIDE 控制自有窗口
    extern void (*g_trainer_request_exit_cb)(void);      // 训练程序设置：IPC SAVE_AND_EXIT 请求优雅保存退出
    extern volatile LONG g_training_fault; // 1: uncertain device state; 2: invalid numeric/data; 3: required I/O
    extern volatile LONG g_shutdown_state; // 0 running, 1 requested, 2 saving, 3 retry needed, 4 saved, 5 fatal
    void training_fail(int reason); // latch once, notify trainer even on early-return paths
    void training_fail_detail(int reason, const char* component, int detail);

    // ========================================================================
    // 目录布局：根目录只放程序+dll+通信文件；日志/存档分文件夹
    // ========================================================================
    #define CKPT_DIR      "checkpoints\\"
    #define CKPT_TEMP_DIR "checkpoints_temp\\"
    #define LOG_DIR       "logs\\"

    // ========================================================================
    // 所有函数前置声明
    // ========================================================================

    // ---- board.c ----
    void reset_game(GameState* g);
    int get_legal_moves(const GameState* g, int moves[]);
    int check_winner(const GameState* g);
    /* Central move invariant used by training, evaluation and UI callers. */
    int is_legal_move(const GameState* g, int action);
    void make_move(GameState* g, int action);
    int is_full(const GameState* g);
    void state_to_input(const GameState* g, float input[]);
    void flip_state(const float* src, float* dst, int* flipped_action);
    int perspective_action(int action, int perspective);
    void normalize_state_to_self(const GameState* src, GameState* dst, int perspective);

    // ---- pattern.c ----
    int is_live_two(const GameState* g, int row, int col, int player);
    int is_live_three(const GameState* g, int row, int col, int player);
    int is_threat_three(const GameState* g, int row, int col, int player);
    int is_threat_four(const GameState* g, int row, int col, int player);
    int detect_double_threat(const GameState* g, int row, int col, int player);
    int find_opponent_winning_move(const GameState* g, int opponent);
    int find_opponent_open_three(const GameState* g, int opponent);
    int find_opponent_double_threat(const GameState* g, int opponent);
    int evaluate_board(const GameState* g, int player);
    int minimax(GameState* g, int depth, int alpha, int beta, int maximizing);
    int minimax_best_move(GameState* g, int depth, float temperature, PRNGState* rng);

    // ---- neural_net.c ----
    void init_net(NeuralNet* net);
    void copy_net(NeuralNet* dst, const NeuralNet* src);
    void forward_cpu(const NeuralNet* net, const float input[], float output[]);
    void forward_cpu_only(const NeuralNet* net, const float input[], float output[]);
    int forward_gpu_snapshot(const NeuralNet* net, const float* states, int count, float* outputs);
    float train_step(NeuralNet* net, const float state[], int action, float target);
    float train_step_cpu_only(NeuralNet* net, const float state[], int action, float target, float learning_rate);
    int train_agent_batch(DQNAgent* agent, const float* states, const int* actions,
        const float* targets, int count, float learning_rate, float* loss);
    /* 按方提交入口：side 是物理棋色（0=黑,1=白）。未开启按方路由时 side 被忽略，
       与 train_agent_batch 完全等价，保证历史调用方行为不变。
       返回 1=已提交，2=安全拒绝（无更新），0=致命失败。 */
    int train_agent_batch_side(DQNAgent* agent, int side, const float* states,
        const int* actions, const float* targets, int count, float learning_rate, float* loss);
    extern int g_uniform_replay_sampling;  /* 实验开关：1=均匀采样对照 */
    extern unsigned long long g_replay_attempt_count, g_replay_commit_count, g_replay_reject_count;
    extern unsigned long long g_batch_delta_reject_count;   /* 参数范数突增门拒绝 */

    // ---- attention.c ----
    void forward_cpu_with_attn(const NeuralNet* net, const float input[],
        float output[], float attn_out[][TOKEN_EMBED_DIM]);
    void extract_feature(const NeuralNet* net, const float attn_out[][TOKEN_EMBED_DIM],
        float* out, int out_dim, const GameState* g);
    void memory_extract_feature(const GameState* g, float* out_feature);

    // ---- dqn_agent.c ----
    void adjust_reward_weights(DQNAgent* agent, int win, int last_player);
    int select_action_with_net(DQNAgent* agent, const GameState* g, NeuralNet* net,
        float eps_override, PRNGState* rng);
    /* observe_memory: 1=正式自博弈观察（更新记忆留存统计），0=评测/对战只读。
       IND-04：评测不得触碰 last_queried/query_count，否则会改变知识淘汰结果。 */
    int select_action(DQNAgent* agent, const GameState* g, int use_opponent,
        float eps_override, PRNGState* rng, int observe_memory);
    void store_experience(DQNAgent* agent, const float state[], int action, float reward,
        const float next_state[], int done, int player);
    float replay_from_buffer(DQNAgent* agent, Experience* buf, int count, int* idx);
    int validate_experience(const Experience* experience);
    void init_dqn(DQNAgent* agent);
    void init_dqn_ex(DQNAgent* agent, int open_logs);
    int open_dqn_logs(DQNAgent* agent);
    void free_dqn(DQNAgent* agent);
    void soft_update_target(DQNAgent* agent);
    /* 按方目标同步：soft=1 时 target <- tau*online + (1-tau)*target，soft=0 时
       target <- online。物理棋色 side 决定只动哪一方；另一方一个字节都不改。
       未开启按方路由时等价于历史单一 target 语义。返回 0 成功，负值失败。 */
    int update_target_side(DQNAgent* agent, int side, int soft);
    void add_to_opponent_pool(DQNAgent* agent);
    int save_opponent_pool(const DQNAgent* agent, const char* filename);
    int load_opponent_pool(DQNAgent* agent, const char* filename);
    NeuralNet* select_opponent_from_pool(DQNAgent* agent, PRNGState* rng);
    float evaluate_agent_selfplay(DQNAgent* agent, int games);
    /* Runtime-only terminal reward balancing.  These factors deliberately live
       outside DQNAgent so changing the controller never changes checkpoint ABI. */
    void set_terminal_win_reward_scales(float black_scale, float white_scale);
    float compute_reward(DQNAgent* agent, const GameState* game, const GameState* next_game,
        int action, int player, int curriculum_phase, int episode, int step);
    typedef enum { EVALUATION_FAULT = -1, EVALUATION_CANCELLED = 0,
        EVALUATION_COMPLETE = 1 } EvaluationStatus;
    typedef int (*EvaluationCancel)(void* context);
    typedef void (*EvaluationProgress)(int completed, int total, void* context);
    // Output arrays are committed only on COMPLETE. Cancellation is owned by
    // the caller; Core must not depend on a particular Trainer's globals.
    EvaluationStatus record_minimax_games_for_episode(DQNAgent* agent, int episode, int depth,
        int* bw, int* bd, int* bl,
        int* ww, int* wd, int* wl,
        float* avg_win, float* avg_draw, float* avg_lose,
        FILE* log_file, EvaluationCancel cancelled = nullptr, void* context = nullptr,
        EvaluationProgress progress = nullptr);
    void log_game_step(DQNAgent* agent, int episode, int step, int player,
        int action, float reward, int done);
    void log_game_flush(DQNAgent* agent);
    void log_train_step(DQNAgent* agent, float loss, float q_value, float target_q,
        int replay_type);
    int get_reward_limit(int type_idx);

    // ---- memory_system.c ----
    void memory_system_init(void);
    void memory_system_free(void);
    void memory_swap_buffers(void);
    CacheBuffer* memory_get_current_buffer(void);
    void memory_process_prev_buffer(int episode, int is_win);
    // outcome: 1=win, 0=loss, -1=draw; cache draws = total-wins-losses.
    void memory_cache_update(const int board[CELL_COUNT], int is_win, int episode, int step,
        int piece_count, int perspective);
    void memory_promote_to_kb(int current_episode);
    int memory_query(const int board[CELL_COUNT], float* out_win_rate, float* out_danger,
        int current_episode, int current_piece_count, int perspective);
    void memory_adjust_action_scores(const GameState* game, float scores[CELL_COUNT],
        float scale, float min_confidence);
    /* Canonical deployed-model inference overlay.  Every action-scoring caller
       (training, evaluation, skill tests and player-facing inference) uses this
       entry point so the learned memory cannot silently disappear by context. */
    void memory_apply_model_scores(const GameState* game, float scores[CELL_COUNT]);
    /* Load the persistent black/white memory set used by a deployed model. */
    int memory_load_deployed(void);
    /* Atomic instantaneous model package: network weights and the exact memory
       generation used with them. */
    int memory_save_model_package(const NeuralNet* net, const char* filename, int episode);
    int memory_load_model_package(NeuralNet* net, const char* filename, int* out_episode);
    // GPT6-Astra: V3 atomically contains both physical sides and their memory.
    int memory_save_model_package_pair(const NeuralNet* black, const NeuralNet* white,
        const char* filename, int episode);
    int memory_load_model_package_pair(NeuralNet* black, NeuralNet* white,
        const char* filename, int* out_episode, int* out_independent);
    void memory_record_appearance(const int board[CELL_COUNT], int current_episode);
    void memory_prune(int current_episode);
    void memory_print_summary(void);
    /* generation is the checkpoint CRC. GPT6-Astra: require_generation also
       requires V11 memory and compares a zero CRC; optional mode accepts legacy
       memory and checks V11 when expected_generation is nonzero. */
    int memory_save_full(const char* filename, int episode, uint32_t generation);
    int memory_load_full(const char* filename, int* out_episode, uint32_t expected_generation,
        bool require_generation = false);
    /* 只读探测：文件是否为 V11 记忆快照且代次匹配（用于两文件保存的中断恢复）。 */
    int memory_snapshot_matches_generation(const char* filename, uint32_t expected_generation);
    /* 只读读出 V11 记忆快照的代次（用于保存前解析残留备份）。 */
    int memory_snapshot_read_generation(const char* filename, uint32_t* out_generation);
    void memory_save_cache(const char* filename);
    void memory_load_cache(const char* filename);
    int memory_save_cache_perspective(const char* filename, int perspective);
    void memory_load_cache_perspective(const char* filename, int perspective);
    int memory_save_kb_perspective(const char* filename, int perspective);
    void memory_load_kb_perspective(const char* filename, int perspective);
    int memory_cache_count_perspective(int perspective);
    int memory_kb_count_perspective(int perspective);
    void memory_save_kb(const char* filename);
    void memory_load_kb(const char* filename);

    // ---- Patch匹配函数 (V5.2) ----
    void extract_patches(const int board[CELL_COUNT], float patches[PATCH_COUNT][PATCH_SIZE]);
    float patch_similarity(const float p1[PATCH_SIZE], const float p2[PATCH_SIZE]);
    float query_match_score(const int probe_board[CELL_COUNT],
        const float proto_board[CELL_COUNT], const float attention[CELL_COUNT]);

    // ---- checkpoint.c ----
    int save_model(const DQNAgent* agent, const char* filename);
    /* 按方导出：side = 0/1 导出该方 online；side < 0 = 历史「当前绑定方」语义。
       两方配对导出写 `<filename>.black.bin` / `<filename>.white.bin`，
       任一件失败即整组返回 0（调用方不得据此宣称导出成功）。 */
    int save_model_side(const DQNAgent* agent, int side, const char* filename);
    int save_model_pair(const DQNAgent* agent, const char* filename);
    int load_model_pair_weights(const char* filename, NeuralNet* black, NeuralNet* white);
    int load_model(DQNAgent* agent, const char* filename);
    /* 读取端按方选方：side = 0/1 装进该方 online+target；side < 0 = 历史共享行为。
       格式按文件大小识别（裸 NeuralNet / OldNeuralNet / V3NeuralNet），不依赖旁路标记。 */
    int load_model_side(DQNAgent* agent, int side, const char* filename);
    /* 优先读 `<filename>.black|white.bin`，缺失时回退到 `<filename>` 并装到该方。 */
    int load_model_for_side(DQNAgent* agent, int side, const char* filename);
    // Read-only extraction for candidate assessment: never restores live/global
    // training counters, replay, LR, or RNG. Output is unchanged on failure.
    int load_checkpoint_model(NeuralNet* out, const char* filename);
    int save_checkpoint(DQNAgent* agent, int phase, int episode, int minimax_depth,
        const char* filename);
    int load_checkpoint(DQNAgent* agent, int* phase, int* episode, int* minimax_depth,
        const char* filename, int load_replay);
    int find_latest_checkpoint(char* filename, size_t size, int* out_episode);
    /* 测试专用：按存档自身内容重算并写回末尾 CRC trailer，用于构造
       「CRC 正确但内容非法」的负向用例。生产路径绝不调用。 */
    int checkpoint_repair_crc_trailer(const char* filename);
    void rollback_log(const char* log_filename, int target_episode);

    // ---- ipc_server.c ----
    void start_ipc_server(void);
    void stop_ipc_server(void);
    void service_snapshot_requests(DQNAgent* agent); // Trainer thread safe point, CPU mode.
    void load_hot_config(void);
    void check_hot_config(void);
    void process_ipc_command(const char* cmd, char* response, int resp_size);
    void build_status_json(char* buf, size_t size);

    // ---- utils.c ----
    void prng_init(PRNGState* prng, uint32_t seed);
    uint32_t prng_next(PRNGState* prng);
    float prng_float(PRNGState* prng);
    int prng_int(PRNGState* prng, int max);
    void load_params(void);
    int apply_recovery_override(const char* path);

    // ---- neural_net.c (GPU integration) ----
    int gpu_net_active(void);
    int gpu_net_activate(const DQNAgent* agent);
    void gpu_net_deactivate(void);
    int gpu_net_download_agent(DQNAgent* agent);
    /* 保存/导出：在同一 safe point 把两方（驻留时是四套）句柄分别回读到各自影子，
       记录「两方同一逻辑时点」。非独立模式退化为共享那一对（历史行为）。 */
    int gpu_net_download_sides_pair(DQNAgent* agent);
    int gpu_net_upload_sides_pair(DQNAgent* agent);
    int gpu_net_snapshot_q(NeuralNet* destination); // Independent, immutable CPU copy.
    int gpu_net_snapshot_online_pair(DQNAgent* agent, NeuralNet* black, NeuralNet* white);
    /* 按方 GPU 参数镜像。side 为物理棋色（0=黑,1=白），非法值返回 -1。
       四套网络同时驻留：按 (side, online|target) 直取该方自己的句柄，不存在
       「设备当前代表哪一方」的状态，因此不需要绑定前置条件。
       按方路由未开启或该方句柄未驻留时全部返回 -1。 */
    int gpu_net_download_side(DQNAgent* agent, int side);
    int gpu_net_upload_side(DQNAgent* agent, int side);
    int gpu_net_snapshot_side(DQNAgent* agent, int side, NeuralNet* destination);
    /* 纯只读取证接口：一次性回读指定方的 online + target 驻留参数。
       不写任何 CPU 影子、不改训练语义，守卫不满足只返回非 0。 */
    int gpu_net_snapshot_side_pair(DQNAgent* agent, int side, NeuralNet* online_out,
        NeuralNet* target_out);
    /* 驻留模式下的兼容 shim：只维护 indep_bound 这条记录并校验该方可用，
       不再切换设备、不再跨方搬运。返回值 0 = 该方可用。 */
    int gpu_net_bind_side(DQNAgent* agent, int side);
    int gpu_net_upload_agent(const DQNAgent* agent);
    /* 按方 soft/hard target 同步：只动该方自己的两个句柄。 */
    int gpu_net_soft_update_side(int side);
    int gpu_net_hard_update_side(int side);
    /* 历史共享接口（非独立模式）：等价于 side<0 的那一对句柄。 */
    int gpu_net_soft_update(void);
    int gpu_net_hard_update(void);
    /* side<0 = 历史共享句柄；否则用该方自己的 online 句柄。 */
    int gpu_net_train_batch_side(int side, const float* states, const int* actions,
        const float* targets, int batch_size, float lr, float* out_loss);
    int gpu_net_train_batch(const float* states, const int* actions, const float* targets,
        int batch_size, float lr, float* out_loss);
    /* side<0 = 历史共享句柄；否则按 (side, use_target) 直取该方句柄。 */
    int gpu_net_forward_batch_side(int side, int use_target, const float* states, int batch_size,
        float* out_qvals);
    int gpu_net_forward_batch(int use_target, const float* states, int batch_size, float* out_qvals);

    /* ========== Neural Probe（神经网络探针） ========== */
    /* Publisher（Trainer 端） */
    int neural_probe_init(void);
    void neural_probe_shutdown(void);
    void neural_probe_publish(DQNAgent* agent, uint64_t episode);
    int neural_probe_enabled(void);
    void neural_probe_set_enabled(int enabled);
    uint64_t neural_probe_snapshot_id(void);

    /* IPC threads only enqueue mutable training configuration.  The trainer
       owner applies it at an episode boundary, so live updates cannot race a
       GPU batch or reward calculation. */
    int apply_pending_ipc_config(DQNAgent* agent);
    void publish_training_status(DQNAgent* agent);
    /* 训练器发布 Minimax 后的阶段进度（技能测试/多快照/保存等），TUI 据此显示。
       整体拷贝快照，跨线程一致；旧训练器不调用时字段保持 0/IDLE。 */
    void publish_training_stage(const StageProgress* sp);
    /* Client（Viewer 端） */
    int probe_client_connect(void);
    void probe_client_disconnect(void);
    int probe_client_is_connected(void);
    int probe_client_update(void);
    uint64_t probe_get_snapshot_id(void);
    /* Client 查询 API */
    struct NeuralProbeHeader;
    struct ProbeLayerDesc;
    struct ProbeGlobalState;
    const struct NeuralProbeHeader* probe_get_header(void);
    const struct ProbeLayerDesc* probe_get_topology(void);
    uint32_t probe_get_layer_count(void);
    const struct ProbeLayerDesc* probe_get_layer(uint32_t index);
    const float* probe_get_weights(uint32_t layer_index);
    const float* probe_get_biases(uint32_t layer_index);
    const float* probe_get_activations(uint32_t layer_index);
    const float* probe_get_gradients(uint32_t layer_index);
    const float* probe_get_input_activations(uint32_t layer_index, uint64_t* out_count);
    const struct ProbeGlobalState* probe_get_global_state(void);
    int probe_get_weight_stats(uint32_t layer_index, float* out_min, float* out_max, float* out_mean, float* out_std);
    int probe_get_bias_stats(uint32_t layer_index, float* out_min, float* out_max, float* out_mean, float* out_std);

    int get_curriculum_phase(int global_ep);
    float compute_epsilon(int ep, int global_ep, int total_eps);
    float compute_shaping_scale(int ep, int global_ep);
    float compute_pool_prob(int global_ep);
    float compute_minimax_prob(int global_ep);
    int compute_train_enabled(int global_ep);
    int compute_target_mode(int global_ep);
    int compute_first_move_enabled(int global_ep);

    void save_population_progress(const PopulationProgress* p);
    int load_population_progress(PopulationProgress* p);
    void scan_population_progress(PopulationProgress* p);
    int find_latest_seed_checkpoint(int agent_idx, char* filename, size_t size);

    // ---- config ----
    void save_config(void);
    void load_config(void);

#ifdef __cplusplus
}
#endif

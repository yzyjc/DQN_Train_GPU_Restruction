#pragma once
#include "core.h"
#include "skill_metric_format.h"
#include "stage_progress.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 阶段进度回调：每完成一局/一题/一快照触发一次，sp 为完整阶段快照。
   传 NULL 表示不采集阶段进度（保持旧的 g_skill_test_progress/phase 行为）。 */
typedef void (*StageProgressCb)(const StageProgress* sp, void* user);

/* ========== 技能回归测试系统 ========== */


/* 单题测试结果 */
typedef struct {
    int puzzle_id;
    int chosen_action;        /* agent 选择的动作 */
    int num_valid;            /* 正确答案数 */
    int valid_answers[CELL_COUNT]; /* 所有正确答案 */
    int is_correct;           /* chosen ∈ valid_answers */
    float q_chosen;           /* 选中位置的 Q 值 */
    float q_second;           /* 第二高 Q 值 */
    float margin;             /* best correct Q - best incorrect Q (signed) */
} PuzzleResult;

typedef struct {
    int correct, total;
    float q_correct, q_wrong, min_q, max_q, min_margin, max_margin;
} SkillPuzzleStats;

/* 技能测试汇总结果 */
typedef struct {
    /* EmptyWin */
    float empty_win_rate;      /* 空盘胜率 (0-1) */
    float empty_avg_steps;     /* 成功时平均步数 */
    float empty_shortest_rate; /* 最短完成率 (≤5步) */
    /* Win-in-1 */
    float win1_fixed;          /* 固定题库正确率 */
    float win1_fixed_margin;   /* 固定题库平均 margin */
    float win1_random;         /* 随机题库正确率 */
    float win1_random_margin;  /* 随机题库平均 margin */
    /* Block-in-1 */
    float block1_fixed;        /* 固定题库正确率 */
    float block1_fixed_margin;
    float block1_random;
    float block1_random_margin;
    /* CPU/GPU parity */
    float parity_max_diff;     /* max |Q_cpu - Q_gpu| */
    float parity_argmax_match; /* argmax 一致率 */
    /* 元数据 */
    int total_tests;
    int episode;
    int suite_version;
    int parity_status;        /* 1=checked, 0=CPU-only/unavailable, -1=invalid values */
    int empty_wins, empty_games;
    float empty_min_steps, empty_max_steps;
    float seeded_win_rate, seeded_avg_steps; /* all 25 forced first moves; includes first stone */
    SkillPuzzleStats win1_stats, block1_stats;
    /* EmptyWin 统计口径：64个快照×2次=128次，64个平均值的分布 */
    float empty_stat_mean;     /* 64个快照平均胜率的均值 */
    float empty_stat_std;      /* 标准差 */
    float empty_stat_min;      /* 最低快照胜率 */
    float empty_stat_max;      /* 最高快照胜率 */
    float empty_steps_mean;    /* 64个快照平均步数的均值 */
    float empty_steps_std;     /* 步数标准差 */
    /* ===== 黑白分色（按物理棋色分别评测，附加字段）=====
       两方独立网络下，「1.0 胜率」如果不分色就无法判断是**哪一方**学会了。
       固定题库的 turn 都是黑方走（generate_win1_fixed / generate_block1_fixed 用
       player=1），所以白方分色只能在**白先**的对局里测：白方开局 25 个首着。
       黑方分色则同时给出「黑先固定题库正确率」和「黑先对局胜率」。
       `side_split_valid == 0` 表示这次评测没有采集分色数据（旧读取方/CPU 路径），
       读取方必须据此判断字段是否可用，不要把 0 当成「胜率为 0」。 */
    int side_split_valid;
    float win1_fixed_black;      /* 黑先固定题目正确率（与 win1_fixed 同源，显式命名） */
    float block1_fixed_black;    /* 黑先 Block1 固定题正确率（与 block1_fixed 同源） */
    float black_win_rate;        /* 黑先（在线方=黑）对局胜率，样本=黑方 25 个首着 */
    float black_avg_steps;       /* 上述胜局平均步数（含首子） */
    float white_win_rate;        /* 白先（在线方=白）对局胜率，样本=白方 25 个首着 */
    float white_avg_steps;
    int black_games, white_games;/* 各自样本数（0 表示该色未采集） */
} SkillTestResult;

/* ========== 多快照测试 ========== */
#define SNAPSHOT_MAX_PUZZLES 512

/* 单快照评估结果 */
typedef struct {
    float accuracy;
    float mean_margin;
    float positive_margin_rate;
    int determinism_mismatch;   /* 两次运行不一致的题数 */
} SnapshotResult;

/* 题目稳定性 */
typedef struct {
    int puzzle_id;
    int total_snapshots;
    int correct_count;          /* 0-16 */
    float avg_margin;
} PuzzleStability;

/* 多快照汇总 */
typedef struct {
    SnapshotResult snapshots[SKILL_WINDOW_SNAPSHOTS];
    int snapshot_count;
    /* 快照级统计 */
    float acc_mean, acc_std, acc_p10, acc_p90, acc_min, acc_max;
    float margin_mean, margin_std;
    /* 题目稳定性 */
    PuzzleStability puzzle_stability[SNAPSHOT_MAX_PUZZLES];
    int puzzle_count;
    int mastered_count;         /* 在除最多一个快照外均答对 */
    int borderline_count;       /* 至少约 31.25% 快照答对 */
    int unknown_count;
    /* 当前模型（最新快照）的单独结果 */
    float current_accuracy;
    float current_margin;
} MultiSnapshotResult;

/* 技能仪表盘条目（持久化到 metric_skill.bin） */
typedef struct {
    int episode;
    /* EmptyWin */
    float empty_win;
    float avg_steps;
    float shortest;
    float empty_min_steps;
    float empty_max_steps;
    /* Win1 */
    float win1_fixed;
    float win1_fixed_margin;
    float win1_random;
    float win1_avg_q_correct;
    float win1_avg_q_wrong;
    float win1_min_q, win1_max_q;
    float win1_min_margin, win1_max_margin;
    /* Block1 */
    float block1_fixed;
    float block1_fixed_margin;
    float block1_random;
    float block1_avg_q_correct;
    float block1_avg_q_wrong;
    float block1_min_q, block1_max_q;
    float block1_min_margin, block1_max_margin;
    /* 多快照统计 */
    float empty_stat_std;
    float empty_stat_min;
    float empty_stat_max;
    float win1_acc_std;
    float block1_acc_std;
    int mastered_puzzle_count;
} SkillMetricRecord;


/* ========== 公共 API ========== */

/* 初始化技能追踪器 */
void skill_tracker_init(SkillTracker* tracker);

/* 运行全部技能测试（ε=0，不写 replay，不更新权重）
   agent: 当前 agent
   episode: 当前局号
   result: 输出结果
   use_gpu: 是否使用 GPU 推理（应与实际对局一致）
   cb/user: 可选阶段进度回调（NULL 表示不采集） */
void run_skill_tests(DQNAgent* agent, int episode, SkillTestResult* result, int use_gpu,
                     StageProgressCb cb = nullptr, void* user = nullptr);

/* 追踪技能状态（自动管理 LOCKED→LEARNING→MASTERED 转换）
   返回: 是否触发回归报警 */
int skill_tracker_update(SkillTracker* tracker, const SkillTestResult* result);

/* 多快照 Win-in-1 测试 */
void run_multi_snapshot_win1(const NeuralNet* snapshots[], int snapshot_count,
    MultiSnapshotResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb = nullptr, void* user = nullptr);

/* 多快照 Block-in-1 测试 */
void run_multi_snapshot_block1(const NeuralNet* snapshots[], int snapshot_count,
    MultiSnapshotResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb = nullptr, void* user = nullptr);

/* 每个历史模型独立跑两次空盘，输出跨快照统计到 result 的 EmptyWin 字段。 */
int run_multi_snapshot_empty(const NeuralNet* snapshots[], int snapshot_count,
    SkillTestResult* result, volatile int* progress, volatile int* phase,
    StageProgressCb cb = nullptr, void* user = nullptr);

/* 写入 metric_skill.bin */
int write_skill_metric(const char* path, const SkillMetricRecord* rec);

/* 生成题库（固定 seed，用于回归集） */
void generate_win1_fixed(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max);
void generate_block1_fixed(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max);

/* 生成随机题库（压力集） */
void generate_win1_random(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max, PRNGState* rng);
void generate_block1_random(GameState puzzles[], int answers[][CELL_COUNT], int num_answers[], int* count, int max, PRNGState* rng);
/* Reachable physical-side puzzles. Global board-hash partition prevents rehearsal/test leakage. */
void generate_side_skill_puzzles(int side,int skill,uint32_t seed,int heldout,
    GameState puzzles[],int answers[][CELL_COUNT],int num_answers[],int* count,int max);
void evaluate_side_skills(DQNAgent* agent,SideSkillScores scores[2],int use_gpu);

/* 单题测试：给定局面和 agent，返回选择结果 */
PuzzleResult test_puzzle(DQNAgent* agent, const GameState* puzzle, const int answers[], int num_answers, int use_gpu,
    PRNGState* rng = nullptr);

/* EmptyWin 测试套件 */
void test_empty_win_suite(DQNAgent* agent, float* win_rate, float* avg_steps, float* shortest_rate, int use_gpu);

#ifdef __cplusplus
}
#endif

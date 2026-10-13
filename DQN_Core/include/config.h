#pragma once

/* ============================================================================
   1. 配置参数（宏定义）
   ============================================================================ */
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

   /* 课程学习阶段边界（全局 episode 计数） */
#define PHASE0_END              20000
#define PHASE1_END              120000
#define PHASE2_END              220000
#define PHASE3_END              320000

/* Epsilon 分段参数（EPS_MAX 和 EPS_MIN 用于种群训练及 stages 数组） */
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

/* Reward Shaping 缩放 */
#define REWARD_SHAPING_PHASE0   0.0f
#define REWARD_SHAPING_PHASE1   0.05f
#define REWARD_SHAPING_PHASE2   0.3f
#define REWARD_SHAPING_PHASE3   1.0f
#define REWARD_SHAPING_PHASE4   1.0f

/* 对手池与 Minimax 概率 */
#define POOL_PROB_PHASE0        0.0f
#define POOL_PROB_PHASE1        0.0f
#define POOL_PROB_PHASE2        0.5f
#define POOL_PROB_PHASE3        1.0f
#define POOL_PROB_PHASE4        1.0f

#define MINIMAX_PROB_PHASE0     0.0f
#define MINIMAX_PROB_PHASE1     0.05f   // 原 0.0f
#define MINIMAX_PROB_PHASE2     0.30f   // 原 0.15f
#define MINIMAX_PROB_PHASE3     0.5f
#define MINIMAX_PROB_PHASE4     1.0f

/* 目标网络更新模式 */
#define TARGET_UPDATE_HARD      0
#define TARGET_UPDATE_SOFT      1
#define TARGET_UPDATE_PHASE0    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE1    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE2    TARGET_UPDATE_HARD
#define TARGET_UPDATE_PHASE3    TARGET_UPDATE_SOFT
#define TARGET_UPDATE_PHASE4    TARGET_UPDATE_SOFT

/* 首步反思、训练启用等 */
#define FIRST_MOVE_ENABLE_PHASE0 0
#define FIRST_MOVE_ENABLE_PHASE1 0
#define FIRST_MOVE_ENABLE_PHASE2 0
#define FIRST_MOVE_ENABLE_PHASE3 0
#define FIRST_MOVE_ENABLE_PHASE4 1

/* 单人四连预训练已经承担基础课程；正式训练从第 0 局就更新网络，
   禁止再用 2 万局纯随机对局填满 Replay 后才开始反向传播。 */
#define TRAIN_ENABLE_PHASE0     1
#define TRAIN_ENABLE_PHASE1     1
#define TRAIN_ENABLE_PHASE2     1
#define TRAIN_ENABLE_PHASE3     1
#define TRAIN_ENABLE_PHASE4     1

/* 学习率衰减 */
#define LR_DECAY_INTERVAL       20000
#define LR_DECAY_FACTOR         0.8f

/* 自适应奖励参数 */
#define REWARD_WEIGHT_THRESHOLD 0.30f
#define REWARD_WEIGHT_DECAY     0.95f
#define REWARD_WEIGHT_MIN       0.1f
#define REWARD_WEIGHT_RECOVER   1.01f
#define REWARD_WEIGHT_MAX       1.0f

/* 终局奖励（归一化） */
#define WIN_REWARD              0.5f
#define LOSE_REWARD             -0.5f
#define DRAW_REWARD             0.0f

/* 中间奖励基础值（乘以 current_shaping_scale） */
#define BLOCK_WIN_BONUS_BASE         0.08f
#define LIVE_TWO_BONUS_BASE          0.02f
#define OPEN_THREE_BONUS_BASE        0.02f
#define THREAT_THREE_BONUS_BASE      0.01f
#define DOUBLE_THREE_BONUS_BASE      0.03f
#define FOUR_THREAT_BONUS_BASE       0.025f
#define BLOCK_OPEN_THREE_BONUS_BASE  0.03f

/* 向后兼容别名（注意：这些宏内部使用了 agent 指针，只能在有 agent 的上下文中使用） */
#define BLOCK_WIN_BONUS         (BLOCK_WIN_BONUS_BASE * agent->current_shaping_scale)
#define LIVE_TWO_BONUS          (LIVE_TWO_BONUS_BASE * agent->current_shaping_scale)
#define OPEN_THREE_BONUS        (OPEN_THREE_BONUS_BASE * agent->current_shaping_scale)
#define THREAT_THREE_BONUS      (THREAT_THREE_BONUS_BASE * agent->current_shaping_scale)
#define DOUBLE_THREE_BONUS      (DOUBLE_THREE_BONUS_BASE * agent->current_shaping_scale)
#define FOUR_THREAT_BONUS       (FOUR_THREAT_BONUS_BASE * agent->current_shaping_scale)
#define BLOCK_OPEN_THREE_BONUS  (BLOCK_OPEN_THREE_BONUS_BASE * agent->current_shaping_scale)

/* 每局各类奖励上限 */
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

/* 多头注意力参数 */
#define ATTENTION_HEADS         4
#define ATTENTION_EMBED         32
#define ATTENTION_HEAD_DIM      (ATTENTION_EMBED / ATTENTION_HEADS) // 8

/* 词向量嵌入维度（每个格子的特征向量维度） */
#define TOKEN_EMBED_DIM        16   // 每个格子用 16 维向量表示

/* 种群训练参数（集中管理） */
#define POP_SIZE                8
#define ROUND_EPISODES          1000
#define TOTAL_ROUNDS            4

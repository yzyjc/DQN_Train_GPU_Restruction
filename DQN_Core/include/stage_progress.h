// stage_progress.h — 共享阶段进度模型
//
// 训练器自动评估（Minimax 后的技能测试 / 多快照 / 保存）与 PG 手动 Skill Test
// 共用同一套阶段进度数据结构和名称/单位映射，保证 TUI 两侧显示口径一致。
//
// 跨线程发布时对 StageProgress 整体赋值/拷贝（结构体小，单次写即可），
// 避免阶段名、计数、总量来自不同时刻。
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* 阶段标识（训练器与 PG 共用） */
typedef enum {
    STG_NONE = 0,
    STG_EVAL,           /* Minimax 评估（已有 g_eval_completed/total 显示） */
    STG_POST_EVAL,      /* 评估后处理（难度调整/最佳模型/指标写入，未知总量） */
    STG_GPU_SYNC,       /* GPU 参数同步（未知总量） */
    STG_SKILL_EMPTY,    /* 技能测试 · EmptyWin */
    STG_SKILL_SEEDED,   /* 技能测试 · Seeded */
    STG_SKILL_WIN1,     /* 技能测试 · Win1（固定+随机题库） */
    STG_SKILL_BLOCK1,   /* 技能测试 · Block1（固定+随机题库） */
    STG_MULTI_EMPTY,    /* 多快照 · EmptyWin */
    STG_MULTI_WIN1,     /* 多快照 · Win1 */
    STG_MULTI_BLOCK1,   /* 多快照 · Block1 */
    STG_SAVE,           /* 保存检查点（未知总量） */
    STG_MAX
} StageId;

/* 计数单位 */
typedef enum {
    UNIT_NONE = 0,       /* 无单位（不定进度/未知总量） */
    UNIT_GAMES = 1,      /* 局 */
    UNIT_PUZZLES = 2,    /* 题 */
    UNIT_SNAPSHOTS = 3   /* 快照 */
} StageUnit;

/* 任务状态（失败/取消/超时必须显示真实状态，不能伪装成 100% 完成） */
typedef enum {
    TASK_IDLE = 0,
    TASK_RUNNING = 1,
    TASK_DONE = 2,
    TASK_FAILED = 3,
    TASK_CANCELLED = 4,
    TASK_TIMEOUT = 5
} TaskState;

/* 阶段进度快照。completed 表示「已完成」数，不是已加载/已开始数。 */
typedef struct {
    int stage;               /* StageId */
    int completed;           /* 已完成数（unit 单位） */
    int total;               /* 总数（unit 单位） */
    int unit;                /* StageUnit */
    int snapshot_completed;  /* 多快照：当前阶段累计完成的快照数 */
    int snapshot_total;      /* 多快照：快照总数 */
    int indeterminate;       /* 1 = 未知总量，使用不定进度 */
    int state;               /* TaskState */
} StageProgress;

/* 阶段名（TUI 显示用）。返回静态只读字符串。 */
const char* stage_progress_name(int stage);

/* 计数单位中文标签。返回静态只读字符串。 */
const char* stage_progress_unit(int unit);

#ifdef __cplusplus
}
#endif

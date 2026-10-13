// playground.h - PLAYGROUND 人机对战快照模型接口
#ifndef PLAYGROUND_H
#define PLAYGROUND_H
#include "core.h"

/* 加载 Trainer 响应 SAVE_SNAPSHOT 写出的完整模型包；旧裸权重仅作兼容回退。
   返回 0 成功，-1 失败。 */
int playground_load_snapshot(void);

/* 候选着法（AI 分析结果） */
typedef struct {
    int move;       /* 0..24 */
    float q;        /* Q 值 */
    float policy;   /* softmax 概率 */
    float win;      /* 胜率估算 0-1 */
} PlayCand;

/* 前向分析：给定棋盘（board[25]: 0空/1黑/2白；turn: 0黑/1白），
   输出全部 Q 值 qvals_out[25]、Top candidates（按 Q 降序前 cand_cap 个）、
   推理耗时 infer_ms。返回候选数；失败 -1。 */
int playground_analyze(const int board[25], int turn,
    float* qvals_out, PlayCand* cands, int cand_cap, int* cand_count, double* infer_ms);

/* 给定棋盘返回 AI 最佳着法 0..24，失败 -1（内部走 analyze） */
int playground_ai_move(const int board[25], int turn);

const char* playground_model_info(void);
int playground_loaded(void);
int playground_param_count(void);
const char* playground_created_at(void);

/* 获取快照模型的 NeuralNet 指针（供技能测试用），未加载返回 NULL */
const NeuralNet* playground_get_net(void);
int playground_copy_net(NeuralNet* out);
int playground_copy_agent(DQNAgent* out);

#endif

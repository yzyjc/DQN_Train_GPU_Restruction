// dqn_replay.h - 对局回放（logreader 移植），供 DQN_Main_TUI.cpp 调用
#ifndef DQN_REPLAY_H
#define DQN_REPLAY_H

#define REPLAY_BOARD_SIZE 5
#define REPLAY_WIN_STREAK 4
#define REPLAY_MAX_STEPS  128

typedef struct {
    int episode;
    int step;
    int player;   /* 0=黑(X) 1=白(O) */
    int action;   /* 0..24 扁平索引 */
    float reward;
    int done;
} GameStep;

/* 解析命令 "g <ep>" 或 "m <ep> <mode> <color> <idx>"，加载棋谱到内部 buffer。
   返回步数；<=0 表示失败（元信息在 tui_replay_meta()）。 */
int tui_replay_load(const char* cmd);
const GameStep* tui_replay_steps(void);
int tui_replay_count(void);
/* 终局胜负：1=黑胜 2=白胜 0=平 -1=未终局 */
int tui_replay_winner(void);
/* 描述性标题，如 "Ep 123 自我对弈" / "Ep 123 Minimax mode0 黑 第5局" */
const char* tui_replay_meta(void);

#endif

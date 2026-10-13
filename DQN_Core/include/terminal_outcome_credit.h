#pragma once
#include "core.h"
#include "independent_networks.h"
// GPT6-Astra: retain one real transition until the opponent replies. A terminal
// reply replaces its pending bootstrap endpoint with the actual final board.
// Each action is inserted once; the winner's own action stays a one-ply sample.
//
// 第 4 步（deepseek-flash）：独立网络的收益传播契约要求 experience 的
// next_state 恒为「对手真实回应之后、本方再次决策」的棋盘（两 ply 跨度）。
// 因此 pending 分支现在**无论对手回应是否结束比赛**都写 next_state；
// 只有回应结束比赛时才改成终局桥接（done=1 + 重算本方结果奖励）。
// 未结束时保持 done=0 与本方那一手原有的 reward：对手那一手的奖励
// 不属于本方的收益流（它记在对手自己的回放缓冲里）。
struct TerminalOutcomeCredit {
    bool pending=false;
    GameState before={};
    Experience experience={};
    void observe(DQNAgent* agent,const GameState& state,const GameState& next,
        int action,float reward,int phase,int episode) {
        const bool done=check_winner(&next)!=0 || is_full(&next);
        /* 版本标记：独立模式下必须保持 13。原先无条件写 12 会让 save_version
           退化成 12，于是独立块永不写入（另一方被静默丢弃），而且会把已载入的
           V13 档在下一个存档点覆写成 V12。这只影响版本标记，不影响回放语义。 */
        const int minimum_version=independent_training::indep_active(agent)?13:12;
        if(agent->checkpoint_version<minimum_version)agent->checkpoint_version=minimum_version;
        if(pending) {
            /* 恒写：端点 = 本方下一个决策点（对手已经真实回应过一手）。
               A 形态（非终局两 ply）与 B 形态（终局桥接）共用同一个 next_state
               定义，因此 validate_experience 的跨度断言对两者一致。 */
            state_to_input(&next,experience.next_state);
            if(done) {
                experience.done=1;
                experience.reward=compute_reward(agent,&before,&next,experience.action,
                    before.turn,phase,episode,0);
            }
            store_experience(agent,experience.state,experience.action,experience.reward,
                experience.next_state,experience.done,before.turn);
            pending=false;
        }
        float input[CELL_COUNT+1],after[CELL_COUNT+1];
        state_to_input(&state,input); state_to_input(&next,after);
        if(done) store_experience(agent,input,action,reward,after,1,state.turn);
        else {
            before=state; pending=true;
            memcpy(experience.state,input,sizeof(input)); memcpy(experience.next_state,after,sizeof(after));
            experience.action=action;experience.reward=reward;experience.done=0;
        }
    }
};

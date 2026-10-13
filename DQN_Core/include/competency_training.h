#pragma once
// Collaboration signature: GPT6-Astra. Optional runtime; targets/replay untouched.
#include "core.h"
#include "independent_networks.h"
#include "skill_test.h"
#include "sentinel_guard.h"
#include <memory>
#include <new>
struct CompetencyOpponents {
    NeuralNet frozen[2];
    NeuralNet lagged[2];
    CompetencyOpponents(){memset(frozen,0,sizeof(frozen));memset(lagged,0,sizeof(lagged));}
};
inline bool competency_refresh_opponents(DQNAgent* a,const CompetencyConfig& c){
    if(!a->indep || !a->indep->active())return false;
    if(!a->competency_opponents)a->competency_opponents=new(std::nothrow) CompetencyOpponents;
    if(!a->competency_opponents)return false;
    auto& s=a->competency;auto& nets=*a->competency_opponents;
    if(s.snapshots_ready && s.games-s.frozen_generation<(uint64_t)c.frozen_interval)return true;
    auto b=std::unique_ptr<NeuralNet>(new(std::nothrow) NeuralNet);auto w=std::unique_ptr<NeuralNet>(new(std::nothrow) NeuralNet);
    if(!b || !w)return false;
    if(gpu_net_active()){if(gpu_net_snapshot_online_pair(a,b.get(),w.get())!=0)return false;}
    else {*b=a->indep->online(independent_training::Side::Black);*w=a->indep->online(independent_training::Side::White);}
    if(!s.snapshots_ready || s.games-s.lagged_refresh_game>=(uint64_t)c.lagged_interval){
        nets.lagged[0]=s.snapshots_ready?nets.frozen[0]:*b;
        nets.lagged[1]=s.snapshots_ready?nets.frozen[1]:*w;
        s.lagged_generation=s.snapshots_ready?s.frozen_generation:s.games;
        s.lagged_refresh_game=s.games;
    }
    nets.frozen[0]=*b;nets.frozen[1]=*w;s.snapshots_ready=1;s.frozen_generation=s.games;return true;
}
struct CompetencyRehearsalLearningRate {
    IndependentNetworks* nets;independent_training::Side side;float normal,reduced;bool valid;
    CompetencyRehearsalLearningRate(DQNAgent* a,int color,float scale):nets(a->indep),side(independent_training::side_from_physical(color)){
        normal=nets->updates(side).learning_rate; if(!(normal>0))normal=g_learning_rate;
        reduced=normal*scale;valid=std::isfinite(normal)&&normal>0&&std::isfinite(reduced)&&reduced>0;
        if(valid)nets->set_learning_rate(side,reduced);
    }
    ~CompetencyRehearsalLearningRate(){if(valid){const float actual=nets->updates(side).learning_rate;
        // Preserve any genuine numerical safety backoff proportionally.
        nets->set_learning_rate(side,actual>0 && actual<reduced?normal*(actual/reduced):normal);}}
    CompetencyRehearsalLearningRate(const CompetencyRehearsalLearningRate&)=delete;
};
inline int competency_skill_batch(DQNAgent* a,int side,int skill,float lr){
    float states[32][CELL_COUNT+1],targets[32];int actions[32];
    if(skill==0){
        const int directions[4][2]={{0,1},{1,0},{1,1},{1,-1}};
        for(int i=0;i<32;++i){
            GameState g={};g.turn=side;
            int d=prng_int(&a->train_rng,4),dr=directions[d][0],dc=directions[d][1];
            int r=prng_int(&a->train_rng,dr?2:5),col=dc<0?3+prng_int(&a->train_rng,2):prng_int(&a->train_rng,dc?2:5);
            int line[4];for(int j=0;j<4;++j)line[j]=(r+j*dr)*5+col+j*dc;
            for(int j=3;j>0;--j){int k=prng_int(&a->train_rng,j+1);int t=line[j];line[j]=line[k];line[k]=t;}
            int prefix=prng_int(&a->train_rng,4);for(int j=0;j<prefix;++j)g.board[line[j]]=side+1;
            actions[i]=line[prefix];targets[i]=WIN_REWARD*powf(GAMMA,(float)(3-prefix));
            state_to_input(&g,states[i]);
        }
    }else{
        GameState puzzles[32];int answers[32][CELL_COUNT],na[32],count=0;
        generate_side_skill_puzzles(side,skill,prng_next(&a->train_rng),0,puzzles,answers,na,&count,32);
        if(count<16)return 2; // insufficient distinct training puzzles: not a commit
        for(int i=0;i<32;++i){int p=i/2;state_to_input(&puzzles[p],states[i]);
            actions[i]=answers[p][prng_int(&a->train_rng,na[p])];targets[i]=WIN_REWARD;
            if(i%2){int wrong[25],n=0;for(int m=0;m<25;++m)if(!puzzles[p].board[m]){
                bool good=false;for(int k=0;k<na[p];++k)if(answers[p][k]==m)good=true;
                if(!good)wrong[n++]=m;
            }if(n){actions[i]=wrong[prng_int(&a->train_rng,n)];targets[i]=LOSE_REWARD;}}
        }
    }
    if(gpu_net_active() && gpu_net_bind_side(a,side)!=0)return 0;
    float loss=0;return train_agent_batch_side(a,side,&states[0][0],actions,targets,32,lr,&loss);
}
inline void competency_write_status(const DQNAgent* a){
    const auto& s=a->competency;const auto& c=g_hot_config.competency;
    FILE* f=fopen(LOG_DIR "competency_status.json.tmp","w");if(!f)return;
    fprintf(f,"{\"schema\":1,\"stage\":%u,\"games\":%llu,\"stage_games\":%llu,\"passes\":%u,\"minimax_ratio\":%.4f,\"config\":{",
        s.stage,(unsigned long long)s.games,(unsigned long long)s.stage_games,s.passes,c.enabled&&s.stage<3?c.minimax_ratio[s.stage]:0);
    for(int i=0;i<30;++i)fprintf(f,"%s\"%s\":%.6f",i?",":"",COMPETENCY_SETTINGS[i].key,competency_value(c,i));
    fprintf(f,"},\"skills\":[");
    for(int side=0;side<2;++side)for(int k=0;k<3;++k){const auto& p=s.skills[side][k];
        fprintf(f,"%s{\"side\":%d,\"skill\":%d,\"score\":%.6f,\"best\":%.6f,\"ema\":%.6f,\"learned\":%u,\"regressions\":%u,\"rehearsals\":%u,\"successes\":%u,\"cooldown_remaining\":%llu}",
            side||k?",":"",side,k,p.current,p.best,p.ema,p.learned,p.regression_count,p.rehearsal_count,p.rehearsal_successes,
            (unsigned long long)(p.cooldown_until>s.games?p.cooldown_until-s.games:0));
    }fprintf(f,"]}\n");bool ok=fflush(f)==0 && _commit(_fileno(f))==0;if(fclose(f)!=0)ok=false;
    if(ok)MoveFileExA(LOG_DIR "competency_status.json.tmp",LOG_DIR "competency_status.json",MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH);
}
inline void competency_event(const DQNAgent* a,const char* event,int side=-1,int skill=-1,float before=0,float after=0,unsigned batches=0){
    FILE* f=fopen(LOG_DIR "competency_events.jsonl","a");if(!f)return;
    fprintf(f,"{\"event\":\"%s\",\"game\":%llu,\"stage\":%u,\"side\":%d,\"skill\":%d,\"before\":%.6f,\"after\":%.6f,\"committed_batches\":%u}\n",
        event,(unsigned long long)a->competency.games,a->competency.stage,side,skill,before,after,batches);fclose(f);
}
inline bool competency_after_game(DQNAgent* a,bool (*stop_requested)()){
    const CompetencyConfig c=g_hot_config.competency;
    if(!c.enabled && !c.rehearsal_enabled)return true;
    auto& s=a->competency;if(!s.initialized && !c.enabled)s.stage=5;
    s.initialized=1;++s.games;++s.stage_games;
    if(!competency_refresh_opponents(a,c))return false;
    if(s.games-s.last_eval<(uint64_t)c.eval_interval){if(s.games%100==0)competency_write_status(a);return true;}
    SideSkillScores scores[2];evaluate_side_skills(a,scores,gpu_net_active());
    if(g_training_fault || !competency_scores_valid(scores)){
        s.last_eval=s.games;s.passes=0;competency_event(a,"evaluation_unavailable");competency_write_status(a);return !g_training_fault;
    }
    if(competency_evaluate(s,c,scores))competency_event(a,"stage_promoted");
    competency_event(a,"evaluated");
    int side,skill;
    if(competency_rehearsal_request(s,c,side,skill)){
        float before=s.skills[side][skill].current;unsigned committed=0;
        // Per-side LR is an override in the shared batch entry, not merely telemetry.
        // Scope the reduction once and restore the normal side policy on every exit.
        CompetencyRehearsalLearningRate learning_rate_scope(a,side,c.rehearsal_lr_scale);
        if(!learning_rate_scope.valid)return false;
        const float rehearsal_lr=learning_rate_scope.reduced;
        competency_event(a,"regression_detected",side,skill,before,before);
        for(int attempt=0;attempt<c.rehearsal_attempts && committed<(unsigned)c.rehearsal_batches;++attempt){
            if(stop_requested && stop_requested())break;
            if(g_training_fault || sentinel_guard_paused())break;
            int rc=competency_skill_batch(a,side,skill,rehearsal_lr);if(rc==0)return false;if(rc==1){++committed;a->global_step+=32;}
            // Deliberately no update_target: other-side and both target weights remain untouched.
        }
        if(stop_requested && stop_requested()){
            competency_finish_rehearsal(s,c,side,skill,before,committed);
            competency_event(a,"rehearsal_interrupted",side,skill,before,before,committed);competency_write_status(a);return true;
        }
        SideSkillScores after[2];evaluate_side_skills(a,after,gpu_net_active());
        if(g_training_fault || !competency_scores_valid(after)){
            competency_finish_rehearsal(s,c,side,skill,before,committed);
            competency_event(a,"retest_unavailable",side,skill,before,before,committed);competency_write_status(a);return !g_training_fault;
        }
        competency_finish_rehearsal(s,c,side,skill,after[side].score[skill],committed);
        competency_event(a,"rehearsal_finished",side,skill,before,after[side].score[skill],committed);
    }
    competency_write_status(a);return true;
}

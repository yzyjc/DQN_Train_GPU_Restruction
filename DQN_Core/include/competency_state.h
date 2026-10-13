#pragma once
// Collaboration signature: GPT6-Astra. No OS, GPU, UI or Bellman dependencies.
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <initializer_list>
struct CompetencyConfig {
    int enabled=0, rehearsal_enabled=0;
    int min_games=5000, eval_interval=500, required_passes=3, stage_cooldown=1000;
    int mastery_passes=2, regression_tests=2, rehearsal_batches=200, rehearsal_attempts=400;
    int rehearsal_cooldown=4000, frozen_interval=100, lagged_interval=500;
    float empty_threshold=.99f, empty_steps=4.2f, mastery_threshold=.90f, regression_delta=.10f;
    float rehearsal_lr_scale=.20f;
    float win1_threshold[6]={0,.90f,.95f,.95f,.95f,.95f};
    float block1_threshold[6]={0,.85f,.90f,.90f,.90f,.90f};
    float minimax_ratio[3]={0,.25f,.50f};
};
struct CompetencySetting { const char* key; float lo,hi; };
inline constexpr CompetencySetting COMPETENCY_SETTINGS[]={
    {"curriculum_enabled",0,1},{"rehearsal_enabled",0,1},
    {"curriculum_min_games",1,1000000},{"curriculum_eval_interval",1,100000},
    {"curriculum_required_passes",1,100},{"curriculum_cooldown",0,1000000},
    {"skill_mastery_passes",1,100},{"regression_tests",2,100},
    {"rehearsal_batches",1,1000},{"rehearsal_attempts",1,10000},
    {"rehearsal_cooldown",1,1000000},{"frozen_interval",1,100000},
    {"lagged_interval",1,1000000},{"empty_threshold",.01f,1},
    {"empty_steps_threshold",4,25},{"skill_mastery_threshold",.01f,1},
    {"regression_delta",.001f,1},{"rehearsal_lr_scale",.001f,1},
    {"stage1_win1",.01f,1},{"stage2_win1",.01f,1},{"stage3_win1",.01f,1},{"stage4_win1",.01f,1},{"stage5_win1",.01f,1},
    {"stage1_block1",.01f,1},{"stage2_block1",.01f,1},{"stage3_block1",.01f,1},{"stage4_block1",.01f,1},{"stage5_block1",.01f,1},
    {"stage1_minimax_ratio",0,1},{"stage2_minimax_ratio",0,1}
};
inline int competency_key(const char* k){for(int i=0;i<30;++i)if(!strcmp(k,COMPETENCY_SETTINGS[i].key))return i;return -1;}
inline float competency_value(const CompetencyConfig& c,int k){
    const int ints[]={c.enabled,c.rehearsal_enabled,c.min_games,c.eval_interval,c.required_passes,c.stage_cooldown,
        c.mastery_passes,c.regression_tests,c.rehearsal_batches,c.rehearsal_attempts,c.rehearsal_cooldown,c.frozen_interval,c.lagged_interval};
    if(k>=0 && k<13)return (float)ints[k];
    const float f[]={c.empty_threshold,c.empty_steps,c.mastery_threshold,c.regression_delta,c.rehearsal_lr_scale};
    if(k>=13 && k<18)return f[k-13];
    if(k>=18 && k<23)return c.win1_threshold[k-17];
    if(k>=23 && k<28)return c.block1_threshold[k-22];
    if(k==28 || k==29)return c.minimax_ratio[k-27];return NAN;
}
inline bool competency_set(CompetencyConfig& c,int k,float v){
    if(k<0 || k>=30 || !std::isfinite(v) || v<COMPETENCY_SETTINGS[k].lo || v>COMPETENCY_SETTINGS[k].hi)return false;
    if(k<13 && v!=std::floor(v))return false;
    int* ints[]={&c.enabled,&c.rehearsal_enabled,&c.min_games,&c.eval_interval,&c.required_passes,&c.stage_cooldown,
        &c.mastery_passes,&c.regression_tests,&c.rehearsal_batches,&c.rehearsal_attempts,&c.rehearsal_cooldown,&c.frozen_interval,&c.lagged_interval};
    if(k<13)*ints[k]=(int)v;
    else if(k<18){float* f[]={&c.empty_threshold,&c.empty_steps,&c.mastery_threshold,&c.regression_delta,&c.rehearsal_lr_scale};*f[k-13]=v;}
    else if(k<23)c.win1_threshold[k-17]=v;
    else if(k<28)c.block1_threshold[k-22]=v;
    else c.minimax_ratio[k-27]=v;
    return true;
}
struct SideSkillScores {
    float score[3]={0,0,0}; // EmptyWin (solo), Win1, Block1
    float empty_steps=0;
    int count[3]={0,0,0};
};
struct SkillProgress {
    float current=0,best=0,ema=0,last=0,baseline=0;
    float before=0,after=0,recovery_delta=0;
    uint64_t cooldown_until=0,last_rehearsal_game=0;
    uint32_t tests=0,learned=0,mastery_streak=0,regression_streak=0,regression_count=0;
    uint32_t rehearsal_count=0,rehearsal_successes=0,rehearsal_batches=0;
};
struct CompetencyState {
    uint32_t schema=1,initialized=0,stage=0,passes=0,snapshots_ready=0;
    uint64_t games=0,stage_games=0,last_eval=0,last_promotion=0,frozen_generation=0,lagged_generation=0,lagged_refresh_game=0;
    SkillProgress skills[2][3];
};
inline bool competency_valid(const CompetencyState& s){
    if(s.schema!=1 || s.initialized>1 || s.stage>5 || s.snapshots_ready>1 || s.last_eval>s.games || s.stage_games>s.games || s.last_promotion>s.games || s.frozen_generation>s.games || s.lagged_generation>s.games || s.lagged_refresh_game>s.games)return false;
    for(const auto& side:s.skills)for(const auto& k:side){
        for(float f:{k.current,k.best,k.ema,k.last,k.baseline,k.before,k.after})if(!std::isfinite(f) || f<0 || f>1)return false;
        if(!std::isfinite(k.recovery_delta) || std::fabs(k.recovery_delta)>1 || k.learned>1 || k.rehearsal_successes>k.rehearsal_count || k.last_rehearsal_game>s.games)return false;
    }return true;
}
inline void competency_observe(SkillProgress& p,float score,const CompetencyConfig& c){
    const float old_reference=p.baseline;
    p.last=p.current;p.current=score;p.best=std::fmax(p.best,score);
    bool regression=p.learned && old_reference-score>=c.regression_delta;
    p.regression_streak=regression?p.regression_streak+1:0;
    if(p.regression_streak==(uint32_t)c.regression_tests)++p.regression_count;
    p.ema=p.tests? .9f*p.ema+.1f*score:score;++p.tests;
    p.mastery_streak=score>=c.mastery_threshold?p.mastery_streak+1:0;
    if(!p.learned && p.mastery_streak>=(uint32_t)c.mastery_passes){p.learned=1;p.baseline=p.ema;}
    if(p.learned && !regression)p.baseline=std::fmax(p.baseline,p.ema);
}
inline bool competency_scores_valid(const SideSkillScores (&scores)[2]){
    for(int side=0;side<2;++side)for(int skill=0;skill<3;++skill){
        const auto& a=scores[side];
        if(a.count[skill]<=0 || !std::isfinite(a.score[skill]) || a.score[skill]<0 || a.score[skill]>1 || !std::isfinite(a.empty_steps) || a.empty_steps<4 || a.empty_steps>25)return false;
    }return true;
}
inline bool competency_evaluate(CompetencyState& s,const CompetencyConfig& c,const SideSkillScores (&scores)[2]){
    s.last_eval=s.games;
    if(!competency_scores_valid(scores)){s.passes=0;return false;}
    bool pass=true;
    for(int side=0;side<2;++side){const auto& a=scores[side];
        for(int skill=0;skill<3;++skill)competency_observe(s.skills[side][skill],a.score[skill],c);
        if(a.score[0]<c.empty_threshold || !std::isfinite(a.empty_steps) || a.empty_steps<4 || a.empty_steps>c.empty_steps)pass=false;
        if(s.stage>0 && (a.score[1]<c.win1_threshold[s.stage] || a.score[2]<c.block1_threshold[s.stage]))pass=false;
    }
    s.last_eval=s.games;s.passes=pass?s.passes+1:0;
    if(!c.enabled || s.stage>=5 || s.stage_games<(uint64_t)c.min_games ||
        s.games-s.last_promotion<(uint64_t)c.stage_cooldown || s.passes<(uint32_t)c.required_passes)return false;
    ++s.stage;s.stage_games=0;s.passes=0;s.last_promotion=s.games;return true;
}
inline bool competency_rehearsal_request(const CompetencyState& s,const CompetencyConfig& c,int& side,int& skill){
    if(!c.rehearsal_enabled)return false;float worst=0;side=skill=-1;
    for(int b=0;b<2;++b)for(int k=0;k<3;++k){const auto& p=s.skills[b][k];
        float drop=p.baseline-p.current;
        if(p.learned && p.regression_streak>=(uint32_t)c.regression_tests && s.games>=p.cooldown_until && drop>worst){worst=drop;side=b;skill=k;}}
    return side>=0;
}
inline void competency_finish_rehearsal(CompetencyState& s,const CompetencyConfig& c,int side,int skill,float after,unsigned batches){
    auto& p=s.skills[side][skill];p.before=p.current;p.after=after;p.recovery_delta=after-p.before;
    p.last_rehearsal_game=s.games;p.cooldown_until=s.games+c.rehearsal_cooldown;
    ++p.rehearsal_count;p.rehearsal_batches=batches;p.regression_streak=0;
    if(std::isfinite(after) && after>=p.baseline-c.regression_delta && after>p.before)++p.rehearsal_successes;
    p.last=p.current;p.current=after;
}

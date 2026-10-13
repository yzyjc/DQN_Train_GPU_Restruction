#pragma once
// Author: GPT6-Astra. Outcome quality is bounded independently of the base reward.
#include <cmath>
#include <cstring>
#include <cstdio>

inline constexpr float BLACK_FAST_WIN_SCALE = .10f;
// Deprecated compatibility field. Kept in the configuration layout for old files.
inline constexpr float BLACK_EARLY_LOSS_SCALE = 0.f;
inline constexpr float WHITE_SURVIVAL_SCALE = .05f;
inline constexpr float WHITE_DRAW_LENGTH_SCALE = .05f; // GPT6-Astra: reward a successful white draw.
struct StepOutcomeConfig {
    int enabled = 0; // old configurations retain their original training semantics
    float black_fast_win_scale = BLACK_FAST_WIN_SCALE;
    float black_early_loss_scale = BLACK_EARLY_LOSS_SCALE;
    float white_survival_scale = WHITE_SURVIVAL_SCALE;
    float white_draw_length_scale = WHITE_DRAW_LENGTH_SCALE;
};
inline constexpr const char* STEP_OUTCOME_KEYS[] = {"step_outcome_enabled",
    "black_fast_win_scale", "black_early_loss_scale", "white_survival_scale", "white_draw_length_scale"};
inline int step_outcome_key(const char* key) {
    for(int i=0;i<5;++i) if(!strcmp(key,STEP_OUTCOME_KEYS[i])) return i;
    return -1;
}
inline float step_outcome_value(const StepOutcomeConfig& c,int key) {
    switch(key) {case 0:return (float)c.enabled;case 1:return c.black_fast_win_scale;
    case 2:return c.black_early_loss_scale;case 3:return c.white_survival_scale;
    case 4:return c.white_draw_length_scale;default:return NAN;}
}
inline bool step_outcome_set(StepOutcomeConfig& c,int key,float value) {
    if(!std::isfinite(value) || key<0 || key>4) return false;
    if(key==0) {if(value!=0 && value!=1) return false;c.enabled=(int)value;return true;}
    // Full-board draws have progress=1, so the draw coefficient is a fixed bonus today.
    if(value<0 || value>.15f) return false;
    if(key==1)c.black_fast_win_scale=value;
    if(key==2)c.black_early_loss_scale=0.f; // old settings are accepted but never applied
    if(key==3)c.white_survival_scale=value;
    if(key==4)c.white_draw_length_scale=value;
    return true;
}
inline bool step_outcome_valid(const StepOutcomeConfig& c) {
    StepOutcomeConfig checked;
    for(int i=0;i<5;++i) if(!step_outcome_set(checked,i,step_outcome_value(c,i)))return false;
    return true;
}
inline float step_outcome_bonus(const StepOutcomeConfig& c,int side,int winner,int steps,int max_steps) {
    if(!step_outcome_valid(c) || side<0 || side>1 || winner<0 || winner>2 || max_steps<=0) return NAN;
    if(!c.enabled)return 0;
    const float p=fmaxf(0.f,fminf(1.f,(float)steps/(float)max_steps));
    if(side==0) return winner==1 ? c.black_fast_win_scale*(1-p) : 0;
    return winner==1 ? c.white_survival_scale*p : winner==0 ? c.white_draw_length_scale*p : 0;
}
// Each formal game snapshots its configuration; parallel workers have separate scopes.
inline thread_local const StepOutcomeConfig* g_episode_outcome_config = nullptr;
struct StepOutcomeScope {
    StepOutcomeConfig config; const StepOutcomeConfig* previous;
    explicit StepOutcomeScope(const StepOutcomeConfig& c):config(c),previous(g_episode_outcome_config) {
        g_episode_outcome_config=&config;
    }
    ~StepOutcomeScope(){g_episode_outcome_config=previous;}
};

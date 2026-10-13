#pragma once
#include "core.h"
#include "metric_log_writer.h"
#include "training_log_timeline.h"
// GPT6-Astra: physical-colour outcome metrics; one observation per completed game.
struct OutcomeMetrics {
    struct Item {int winner=0,steps=0;};
    Item window[1000]={};
    int used=0,pos=0,last_episode=-1;
    double sums[6]={},rolling_sums[6]={};
    unsigned long long counts[6]={},rolling_counts[6]={};
    static bool includes(int k,int winner) {
        return k==5 || (k==4 && winner==0) || ((k==0 || k==3) && winner==1) ||
            ((k==1 || k==2) && winner==2);
    }
    void add(int winner,int steps) {
        if(used==1000) {
            for(int k=0;k<6;++k) if(includes(k,window[pos].winner)) {
                rolling_sums[k]-=window[pos].steps;--rolling_counts[k];
            }
        }
        for(int k=0;k<6;++k) if(includes(k,winner)) {
            sums[k]+=steps;++counts[k];rolling_sums[k]+=steps;++rolling_counts[k];
        }
        window[pos]={winner,steps};pos=(pos+1)%1000;if(used<1000)++used;
    }
    bool restore(const char* path,int before_episode) {
        *this=OutcomeMetrics{};
        FILE* f=fopen(path,"rb");if(!f)return errno==ENOENT;
        MetricLogHeader h={};
        if(fread(&h,sizeof(h),1,f)!=1 || !metric_log_header_valid(h,14)){fclose(f);return false;}
        MetricLogPrefix p;float row[14];
        while(fread(&p,sizeof(p),1,f)==1 && fread(row,sizeof(row),1,f)==1) {
            if(p.episode>=before_episode)break;
            if(!isfinite(row[0]) || !isfinite(row[1]) || row[0]<1 || row[0]>CELL_COUNT ||
                row[1]<0 || row[1]>2 || row[0]!=(int)row[0] || row[1]!=(int)row[1]){fclose(f);return false;}
            add((int)row[1],(int)row[0]);last_episode=p.episode;
        }
        const bool ok=!ferror(f);fclose(f);return ok;
    }
};
inline bool write_outcome_row(const char* name,int episode,const float* row,int count) {
    const std::string path=dqn_log_path(name);
    FILE* f=metric_log_open_append(path.c_str(),count);if(!f)return false;
    const MetricLogPrefix p={episode,0};
    bool ok=fwrite(&p,sizeof(p),1,f)==1 && fwrite(row,sizeof(float),count,f)==(size_t)count && fflush(f)==0;
    if(fclose(f)!=0)ok=false;return ok;
}
inline bool log_outcome_metrics(DQNAgent* agent,const GameState& final,int episode,int online_side,
    int phase,const StepOutcomeConfig& config) {
    static OutcomeMetrics metrics;
    static bool initialized=false;
    static std::string previous_path;
    const std::string path=dqn_log_path("metric_outcome_raw.bin");
    if(!initialized || previous_path!=path || episode<=metrics.last_episode) {
        if(!metrics.restore(path.c_str(),episode))return false;
        previous_path=path;
        initialized=true;
    }
    const int winner=check_winner(&final);
    if(!winner && !is_full(&final))return false;
    int steps=0;for(int j=0;j<CELL_COUNT;++j)steps+=final.board[j]!=0;
    float rewards[2];float saved[7];int usage[7];
    static_assert(sizeof(saved)==sizeof(agent->reward_stats),"reward statistics layout");
    static_assert(sizeof(usage)==sizeof(agent->reward_usage),"reward usage layout");
    memcpy(saved,agent->reward_stats,sizeof(saved));memcpy(usage,agent->reward_usage,sizeof(usage));
    for(int side=0;side<2;++side)rewards[side]=compute_reward(agent,&final,&final,0,side,phase,episode,steps);
    memcpy(agent->reward_stats,saved,sizeof(saved));memcpy(agent->reward_usage,usage,sizeof(usage));
    const float raw[14]={(float)steps,(float)winner,(float)online_side,(float)config.enabled,
        config.black_fast_win_scale,config.black_early_loss_scale,config.white_survival_scale,config.white_draw_length_scale,
        rewards[0],rewards[1],step_outcome_bonus(config,0,winner,steps,CELL_COUNT),
        step_outcome_bonus(config,1,winner,steps,CELL_COUNT),1.f,(float)g_hot_config.config_version};
    if(!write_outcome_row("metric_outcome_raw.bin",episode,raw,14))return false;
    metrics.add(winner,steps);metrics.last_episode=episode;
    float avg[12],counts[12];
    for(int k=0;k<6;++k) {
        counts[k]=(float)metrics.counts[k];counts[k+6]=(float)metrics.rolling_counts[k];
        avg[k]=metrics.counts[k] ? (float)(metrics.sums[k]/metrics.counts[k]):0;
        avg[k+6]=metrics.rolling_counts[k] ? (float)(metrics.rolling_sums[k]/metrics.rolling_counts[k]):0;
    }
    if((episode+1)%100==0) {
        const char* names[6]={"black_win_avg_steps","black_loss_avg_steps","white_win_avg_steps",
            "white_loss_avg_steps","draw_avg_steps","avg_game_steps"};
        for(int k=0;k<6;++k) {
            printf("[OUTCOME] %s=",names[k]);
            if(counts[k])printf("%.4f",avg[k]);else printf("N/A");
            printf(" count=%.0f rolling1000=",counts[k]);
            if(counts[k+6])printf("%.4f",avg[k+6]);else printf("N/A");
            printf(" rolling_count=%.0f\n",counts[k+6]);
        }
    }
    return write_outcome_row("metric_outcome_steps.bin",episode,avg,12) &&
        write_outcome_row("metric_outcome_counts.bin",episode,counts,12);
}

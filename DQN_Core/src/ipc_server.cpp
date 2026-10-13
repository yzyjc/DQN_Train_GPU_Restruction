#include "core.h"
#include "sentinel_guard.h"
#include "independent_networks.h"   /* 按方导出：快照要标明是哪一方的 online */
#include <sddl.h>
#include <memory>
#include <sstream>
#pragma comment(lib, "advapi32.lib")
// ---------- 加载配置文件（热配置） ----------

// config.h（添加）
extern struct TrainingContext g_ctx;
extern struct DQNAgent* g_current_agent;
static ULONGLONG g_hot_config_filetime = 0;

enum PendingConfigBits : uint32_t {
    PENDING_TRAIN_REPEAT = 1u << 0, PENDING_LEARNING_RATE = 1u << 1,
    PENDING_LIVE_TWO = 1u << 2, PENDING_BLOCK_WIN = 1u << 3,
    PENDING_ADVANCED = 1u << 4, PENDING_EPSILON = 1u << 5,
    PENDING_MEMORY_SCALE = 1u << 6, PENDING_MEMORY_CONFIDENCE = 1u << 7,
    PENDING_MEMORY_ENABLED = 1u << 8,
    PENDING_OUTCOME_FIRST = 1u << 9, PENDING_COMPETENCY = 1u << 14
};
struct PendingConfig {
    uint32_t mask;
    int train_repeat;
    int memory_enabled;
    float learning_rate, live_two, block_win, advanced, epsilon;
    float memory_scale, memory_confidence;
    StepOutcomeConfig outcome;
    CompetencyConfig competency;
    uint32_t competency_mask;
};
static PendingConfig g_pending_config = {};
/* 正数热覆盖不能销毁模型按 checkpoint + 衰减计划得到的学习率。
   保存覆盖前的值，使 SET learning_rate -1 能真正恢复，而不只是改标志。 */
static float g_automatic_learning_rate = NAN;
static float g_automatic_side_learning_rate[2]={NAN,NAN};
static void capture_side_control_lr(DQNAgent* a){
    for(int i=0;i<2;++i)g_automatic_side_learning_rate[i]=a && independent_training::indep_active(a)?a->indep->learning_rate_or(independent_training::side_from_physical(i),g_learning_rate):NAN;
}
static void apply_side_control_lr(DQNAgent* a,float value,bool restore){
    if(a && independent_training::indep_active(a))for(int i=0;i<2;++i){
        const float saved=g_automatic_side_learning_rate[i];a->indep->set_learning_rate(independent_training::side_from_physical(i),restore && isfinite(saved) && saved>0?saved:value);}
    if(restore)for(float& v:g_automatic_side_learning_rate)v=NAN;
}
static SRWLOCK g_pending_config_lock = SRWLOCK_INIT;

/* 训练分阶段累计计时 (微秒/步, Trainer 定义, IPC 读取) */
extern volatile double g_phase_forward_us, g_phase_backward_us, g_phase_target_us;
extern volatile int g_phase_steps;
extern volatile double g_side_forward_us[2], g_side_backward_us[2];
extern volatile int g_side_steps[2];
/* replay_from_buffer 细分计时 */
extern volatile double g_replay_sample_us, g_replay_forward_us, g_replay_target_us;
extern volatile double g_replay_train_us, g_replay_sync_us;
extern volatile int g_replay_count;
/* backward_update 分段计时 (extern "C" 统一符号名) */
extern "C" {
extern volatile double g_bwd_fc4_us, g_bwd_fc3_us, g_bwd_fc2_us, g_bwd_fc1_us;
extern volatile double g_bwd_pool_us, g_bwd_attn_us, g_bwd_embed_us;
extern volatile int g_bwd_count;
/* train_step_batch per-sample 开销分段 */
extern volatile double g_ts_setup_us, g_ts_memset_us, g_ts_fwd_us, g_ts_hubler_us, g_ts_bwd_us;
extern volatile double g_ts_validate_us;
extern volatile int g_ts_count;
}

/* IPC publishes one coherent status generation.  Client threads never walk
   the trainer's mutable agent/context directly. */
struct TrainingStatusSnapshot {
    int episode, wins[3], depth, cache_count, kb_count;
    int cache_b, kb_b, cache_w, kb_w;  /* per-perspective counts */
    float epsilon, learning_rate, ep_time, elapsed, remaining;
    HotConfig hot;
    int pretrain_active, pretrain_steps, pretrain_stable;
    float pretrain_avg_steps, pretrain_success;
    int eval_active, eval_completed, eval_total;
    StageProgress stage;    /* Minimax 后的阶段进度（技能测试/多快照/保存） */
    LONG shutdown_state, training_fault;
    /* 训练分阶段计时 (微秒/步) */
    float phase_forward_us, phase_backward_us, phase_target_us;
    int phase_steps;
    /* 按方分阶段计时 (Black=0, White=1) */
    float side_forward_us[2], side_backward_us[2];
    int side_steps[2];
    /* replay_from_buffer 细分计时 */
    float replay_sample_us, replay_forward_us, replay_target_us;
    float replay_train_us, replay_sync_us;
    int replay_count;
    /* backward_update 分段计时 */
    float bwd_fc4_us, bwd_fc3_us, bwd_fc2_us, bwd_fc1_us;
    float bwd_pool_us, bwd_attn_us, bwd_embed_us;
    int bwd_count;
    /* train_step_batch per-sample 开销 */
    float ts_setup_us, ts_memset_us, ts_fwd_us, ts_hubler_us, ts_bwd_us;
    float ts_validate_us;
    int ts_count;
    int sentinel_paused;
    CompetencyState competency;
    int independent_active;
};
static TrainingStatusSnapshot g_status = {};
static SRWLOCK g_status_lock = SRWLOCK_INIT;

/* 训练器发布的阶段进度（跨线程整体拷贝，独立于状态主快照）。 */
static StageProgress g_trainer_stage = {};
static SRWLOCK g_stage_lock = SRWLOCK_INIT;

void publish_training_stage(const StageProgress* sp) {
    if (!sp) return;
    AcquireSRWLockExclusive(&g_stage_lock);
    g_trainer_stage = *sp;
    ReleaseSRWLockExclusive(&g_stage_lock);
}

static StageProgress read_training_stage(void) {
    AcquireSRWLockShared(&g_stage_lock);
    StageProgress copy = g_trainer_stage;
    ReleaseSRWLockShared(&g_stage_lock);
    return copy;
}

void publish_training_status(DQNAgent* agent) {
    TrainingStatusSnapshot next = {};
    next.episode = g_ctx.total_games_played;
    memcpy(next.wins, g_wins, sizeof(next.wins));
    next.depth = g_ctx.minimax_depth;
    next.epsilon = agent ? agent->epsilon : 0.0f;
    next.learning_rate = g_learning_rate;
    next.hot = g_hot_config;
    if(agent){next.competency=agent->competency;next.independent_active=independent_training::indep_active(agent);}
    next.cache_count = g_cache_count; next.kb_count = g_kb_count;
    next.cache_b = memory_cache_count_perspective(0);
    next.kb_b = memory_kb_count_perspective(0);
    next.cache_w = memory_cache_count_perspective(1);
    next.kb_w = memory_kb_count_perspective(1);
    next.ep_time = g_ep_time; next.elapsed = g_elapsed; next.remaining = g_remaining;
    next.pretrain_active = g_pretrain_active; next.pretrain_steps = g_pretrain_steps;
    next.pretrain_avg_steps = g_pretrain_avg_steps;
    next.pretrain_success = g_pretrain_success_rate;
    next.pretrain_stable = g_pretrain_stable_passes;
    next.eval_active = (int)InterlockedCompareExchange(&g_eval_active, 0, 0);
    next.eval_completed = (int)InterlockedCompareExchange(&g_eval_completed, 0, 0);
    next.eval_total = (int)InterlockedCompareExchange(&g_eval_total, 0, 0);
    next.stage = read_training_stage();
    /* 训练分阶段计时 (读取后重置, 避免累计值无限增长) */
    next.phase_forward_us = (float)g_phase_forward_us;
    next.phase_backward_us = (float)g_phase_backward_us;
    next.phase_target_us = (float)g_phase_target_us;
    next.phase_steps = g_phase_steps;
    next.side_forward_us[0] = (float)g_side_forward_us[0];
    next.side_forward_us[1] = (float)g_side_forward_us[1];
    next.side_backward_us[0] = (float)g_side_backward_us[0];
    next.side_backward_us[1] = (float)g_side_backward_us[1];
    next.side_steps[0] = g_side_steps[0];
    next.side_steps[1] = g_side_steps[1];
    next.replay_sample_us = (float)g_replay_sample_us;
    next.replay_forward_us = (float)g_replay_forward_us;
    next.replay_target_us = (float)g_replay_target_us;
    next.replay_train_us = (float)g_replay_train_us;
    next.replay_sync_us = (float)g_replay_sync_us;
    next.replay_count = g_replay_count;
    next.bwd_fc4_us = (float)g_bwd_fc4_us;
    next.bwd_fc3_us = (float)g_bwd_fc3_us;
    next.bwd_fc2_us = (float)g_bwd_fc2_us;
    next.bwd_fc1_us = (float)g_bwd_fc1_us;
    next.bwd_pool_us = (float)g_bwd_pool_us;
    next.bwd_attn_us = (float)g_bwd_attn_us;
    next.bwd_embed_us = (float)g_bwd_embed_us;
    next.bwd_count = g_bwd_count;
    next.ts_setup_us = (float)g_ts_setup_us;
    next.ts_memset_us = (float)g_ts_memset_us;
    next.ts_fwd_us = (float)g_ts_fwd_us;
    next.ts_hubler_us = (float)g_ts_hubler_us;
    next.ts_bwd_us = (float)g_ts_bwd_us;
    next.ts_validate_us = (float)g_ts_validate_us;
    next.ts_count = g_ts_count;
    if (g_phase_steps > 0) {
        g_phase_forward_us = 0; g_phase_backward_us = 0; g_phase_target_us = 0; g_phase_steps = 0;
        g_side_forward_us[0] = 0; g_side_forward_us[1] = 0;
        g_side_backward_us[0] = 0; g_side_backward_us[1] = 0;
        g_side_steps[0] = 0; g_side_steps[1] = 0;
        g_replay_sample_us = 0; g_replay_forward_us = 0; g_replay_target_us = 0;
        g_replay_train_us = 0; g_replay_sync_us = 0; g_replay_count = 0;
        g_bwd_fc4_us = 0; g_bwd_fc3_us = 0; g_bwd_fc2_us = 0; g_bwd_fc1_us = 0;
        g_bwd_pool_us = 0; g_bwd_attn_us = 0; g_bwd_embed_us = 0; g_bwd_count = 0;
        g_ts_setup_us = 0; g_ts_memset_us = 0; g_ts_fwd_us = 0; g_ts_hubler_us = 0; g_ts_bwd_us = 0;
        g_ts_validate_us = 0; g_ts_count = 0;
    }
    next.shutdown_state = InterlockedCompareExchange(&g_shutdown_state, 0, 0);
    next.training_fault = InterlockedCompareExchange(&g_training_fault, 0, 0);
    next.sentinel_paused = sentinel_guard_paused();
    AcquireSRWLockExclusive(&g_status_lock);
    g_status = next;
    ReleaseSRWLockExclusive(&g_status_lock);
}

static TrainingStatusSnapshot read_training_status(void) {
    AcquireSRWLockShared(&g_status_lock);
    TrainingStatusSnapshot copy = g_status;
    ReleaseSRWLockShared(&g_status_lock);
    /* These latches are already atomic and must be visible immediately while
       a save/fatal transition is in progress, even between episode publishes. */
    copy.shutdown_state = InterlockedCompareExchange(&g_shutdown_state, 0, 0);
    copy.training_fault = InterlockedCompareExchange(&g_training_fault, 0, 0);
    copy.sentinel_paused = sentinel_guard_paused();
    return copy;
}

int apply_pending_ipc_config(DQNAgent* agent) {
    AcquireSRWLockExclusive(&g_pending_config_lock);
    PendingConfig pending = g_pending_config;
    g_pending_config.mask = 0;
    g_pending_config.competency_mask=0;
    ReleaseSRWLockExclusive(&g_pending_config_lock);
    if (!pending.mask) return 0;
    for(int i=0;i<30;++i)if(pending.competency_mask&(1u<<i))
        competency_set(g_hot_config.competency,i,competency_value(pending.competency,i));
    for(int i=0;i<5;++i) if(pending.mask & (PENDING_OUTCOME_FIRST << i))
        step_outcome_set(g_hot_config.step_outcome,i,step_outcome_value(pending.outcome,i));
    if (pending.mask & PENDING_TRAIN_REPEAT) g_hot_config.train_repeat_per_step = pending.train_repeat;
    if (pending.mask & PENDING_LEARNING_RATE) {
        const float previous_override = g_hot_config.learning_rate_override;
        if (pending.learning_rate > 0.0f) {
            /* 人工覆盖：从 AUTO 首次进入覆盖时，快照当前自动 LR。 */
            if (previous_override < 0.0f || !isfinite(g_automatic_learning_rate)){
                capture_side_control_lr(agent);
                g_automatic_learning_rate = g_learning_rate;
            }
            g_learning_rate = pending.learning_rate;
            apply_side_control_lr(agent,g_learning_rate,false);
        } else {
            /* 恢复 AUTO（-1）：
               - 仅从「人工覆盖」退出时恢复快照；AUTO→AUTO 幂等，不恢复（否则会撤销
                 训练期间的正常衰减与哨兵降速）。
               - 恢复后立即清空快照，使历史快照失效，避免后续 -1 再次回到旧值。 */
            if (previous_override > 0.0f &&
                isfinite(g_automatic_learning_rate) && g_automatic_learning_rate > 0.0f) {
                g_learning_rate = g_automatic_learning_rate;
                apply_side_control_lr(agent,g_learning_rate,true);
            }
            g_automatic_learning_rate = NAN;
        }
        g_hot_config.learning_rate_override = pending.learning_rate;
    }
    if (pending.mask & PENDING_LIVE_TWO) g_hot_config.live_two_scale = pending.live_two;
    if (pending.mask & PENDING_BLOCK_WIN) g_hot_config.block_win_scale = pending.block_win;
    if (pending.mask & PENDING_ADVANCED) g_hot_config.advanced_reward_scale = pending.advanced;
    if (pending.mask & PENDING_EPSILON) {
        g_hot_config.epsilon_override = pending.epsilon;
        if (agent && pending.epsilon >= 0.0f) agent->epsilon = pending.epsilon;
    }
    if (pending.mask & PENDING_MEMORY_SCALE) g_hot_config.memory_scale = pending.memory_scale;
    if (pending.mask & PENDING_MEMORY_CONFIDENCE) g_hot_config.memory_confidence = pending.memory_confidence;
    if (pending.mask & PENDING_MEMORY_ENABLED) g_hot_config.memory_enabled = pending.memory_enabled;
    g_hot_config.last_load_time = time(NULL);
    g_hot_config.config_version++;
    save_config();
    publish_training_status(agent);
    return 1;
}

static ULONGLONG hot_config_write_time(void) {
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA("train_config.txt", GetFileExInfoStandard, &fad)) return 0;
    ULARGE_INTEGER ft;
    ft.LowPart = fad.ftLastWriteTime.dwLowDateTime;
    ft.HighPart = fad.ftLastWriteTime.dwHighDateTime;
    return ft.QuadPart;
}

void load_hot_config(void) {
    FILE* f = fopen("train_config.txt", "r");
    if (!f) {
        g_hot_config.train_repeat_per_step = 1;
        g_hot_config.learning_rate_override = -1.0f;
        g_hot_config.live_two_scale = 1.0f;
        g_hot_config.block_win_scale = 1.0f;
        g_hot_config.advanced_reward_scale = 1.0f;
        g_hot_config.epsilon_override = -1.0f;
        g_hot_config.last_load_time = time(NULL);
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char key[64]; float val;
        if (sscanf(line, "%63[^=]=%f", key, &val) == 2 && isfinite(val)) {
            char* begin=key;while(*begin==' ' || *begin=='\t')++begin;
            char* end=begin+strlen(begin);while(end>begin && (end[-1]==' ' || end[-1]=='\t'))--end;*end=0;
            if(begin!=key)memmove(key,begin,strlen(begin)+1);
            const int course_key=competency_key(key);
            if(course_key>=0){
                if(course_key<2 && val>0 && g_current_agent && !independent_training::indep_active(g_current_agent)){
                    printf("[HOT] Course flag requires independent networks; ignored: %s\n",key);continue;
                }
                competency_set(g_hot_config.competency,course_key,val);continue;
            }
            if(step_outcome_key(key)>=0) {
                if(!step_outcome_set(g_hot_config.step_outcome,step_outcome_key(key),val))
                    printf("[HOT] Invalid outcome setting ignored: %s\n",key);
                continue;
            }
            if (strcmp(key, "train_repeat") == 0) {
                if (val < 1.0f || val > 20.0f) continue;
                g_hot_config.train_repeat_per_step = (int)val;
                printf("[HOT] 训练频率更新: %d 次/步\n", g_hot_config.train_repeat_per_step);
            }
            else if (strcmp(key, "learning_rate") == 0) {
                if (!((val > 0.0f && val <= 1.0f) || val == -1.0f)) continue;
                const float previous_override = g_hot_config.learning_rate_override;
                if (val > 0.0f) {
                    /* 人工覆盖：从 AUTO 首次进入覆盖时，快照当前自动 LR。 */
                    if (previous_override < 0.0f || !isfinite(g_automatic_learning_rate)){
                        capture_side_control_lr(g_current_agent);
                        g_automatic_learning_rate = g_learning_rate;
                    }
                    g_learning_rate = val;
                    apply_side_control_lr(g_current_agent,g_learning_rate,false);
                    printf("[HOT] 学习率更新: %.6f\n", g_learning_rate);
                } else {
                    /* 恢复 AUTO（-1）：仅从人工覆盖退出时恢复快照；AUTO→AUTO 幂等，
                       恢复后清空快照，避免撤销训练期间衰减/降速。 */
                    if (previous_override > 0.0f &&
                        isfinite(g_automatic_learning_rate) && g_automatic_learning_rate > 0.0f) {
                        g_learning_rate = g_automatic_learning_rate;
                        apply_side_control_lr(g_current_agent,g_learning_rate,true);
                    }
                    g_automatic_learning_rate = NAN;
                    printf("[HOT] 学习率恢复自动计划值: %.8f\n", g_learning_rate);
                }
                g_hot_config.learning_rate_override = val;
            }
            else if (strcmp(key, "live_two_scale") == 0) {
                if (val < 0.0f || val > 10.0f) continue;
                g_hot_config.live_two_scale = val;
                printf("[HOT] 活二缩放更新: %.2f\n", g_hot_config.live_two_scale);
            }
            else if (strcmp(key, "block_win_scale") == 0) {
                if (val < 0.0f || val > 10.0f) continue;
                g_hot_config.block_win_scale = val;
                printf("[HOT] 封堵缩放更新: %.2f\n", g_hot_config.block_win_scale);
            }
            else if (strcmp(key, "advanced_scale") == 0) {
                if (val < 0.0f || val > 10.0f) continue;
                g_hot_config.advanced_reward_scale = val;
                printf("[HOT] 高阶奖励缩放更新: %.2f\n", g_hot_config.advanced_reward_scale);
            }
            else if (strcmp(key, "epsilon") == 0) {
                if (val < -1.0f || val > 1.0f) continue;
                g_hot_config.epsilon_override = val;
                if (val >= 0) {
                    printf("[HOT] Epsilon 覆盖请求: %.4f（将在下一局生效）\n", val);
                }
                else {
                    printf("[HOT] Epsilon 恢复自动调度\n");
                }
            }
        }
    }
    fclose(f);
    g_hot_config_filetime = hot_config_write_time();
    g_hot_config.last_load_time = time(NULL);
    g_hot_config.config_version++;
}

void check_hot_config(void) {
    /* 直接比较完整 FILETIME。旧代码取整到秒后又与 time(NULL) 比较，连续
       点击“应用”或同秒内修改配置时会漏掉更新。 */
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (GetFileAttributesExA("train_config.txt", GetFileExInfoStandard, &fad)) {
        ULARGE_INTEGER ft;
        ft.LowPart = fad.ftLastWriteTime.dwLowDateTime;
        ft.HighPart = fad.ftLastWriteTime.dwHighDateTime;
        if (g_hot_config_filetime == 0 || ft.QuadPart != g_hot_config_filetime) {
            g_hot_config_filetime = ft.QuadPart;
            printf("\n[HOT] 检测到配置文件变化，重新加载...\n");
            load_hot_config();
            if (g_hot_config.epsilon_override >= 0 && g_current_agent) {
                g_current_agent->epsilon = g_hot_config.epsilon_override;
            }
        }
    }
}

static volatile LONG g_snapshot_requested = 0;
static SRWLOCK g_snapshot_file_lock = SRWLOCK_INIT;
void service_snapshot_requests(DQNAgent* agent) {
    if (agent && InterlockedExchange(&g_snapshot_requested, 0)) {
        AcquireSRWLockExclusive(&g_snapshot_file_lock);
        DQNAgent* frozen = (DQNAgent*)calloc(1, sizeof(DQNAgent));
        // GPT6-Astra: capture both online networks under one GPU lock. The
        // package is authoritative; the raw black file is a legacy completion marker.
        auto white = std::make_unique<NeuralNet>();
        const int per_side = independent_training::indep_active(agent);
        int captured = 0;
        if (frozen) {
            if (per_side && gpu_net_active())
                captured = gpu_net_snapshot_online_pair(agent,&frozen->q_net,white.get())==0;
            else if (per_side) {
                frozen->q_net=agent->indep->online(independent_training::Side::Black);
                *white=agent->indep->online(independent_training::Side::White); captured=1;
            } else if (gpu_net_active()) captured=gpu_net_snapshot_q(&frozen->q_net)==0;
            else {frozen->q_net=agent->q_net;captured=1;}
        }
        int ok = captured && (per_side
            ? memory_save_model_package_pair(&frozen->q_net,white.get(),"current_model_package.bin",g_ctx.total_games_played)
            : memory_save_model_package(&frozen->q_net,"current_model_package.bin",g_ctx.total_games_played));
        if(ok) ok=save_model(frozen,"current_snapshot.bin");
        free(frozen);
        if (!ok)
            printf("[SNAPSHOT] Save failed; previous file preserved\n");
        ReleaseSRWLockExclusive(&g_snapshot_file_lock);
    }
}

// ---------- IPC 命令处理 ----------
void process_ipc_command(const char* cmd, char* response, int resp_size) {
    char cmd_copy[512];
    strncpy(cmd_copy, cmd, sizeof(cmd_copy) - 1);
    cmd_copy[sizeof(cmd_copy) - 1] = '\0';

    char* token = strtok(cmd_copy, " \t\r\n");
    if (!token) {
        snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"empty command\"}");
        return;
    }

    if(!strcmp(token,"GET_CONFIG") || !strcmp(token,"GET_COMPETENCY")) {
        const auto s=read_training_status();std::ostringstream out;
        out<<"{\"status\":\"ok\",\"config_version\":"<<s.hot.config_version<<",\"independent_active\":"<<s.independent_active<<",\"data\":{";
        if(!strcmp(token,"GET_CONFIG")) {
            out<<"\"learning_rate\":"<<s.hot.learning_rate_override<<",\"train_repeat\":"<<s.hot.train_repeat_per_step
                <<",\"epsilon\":"<<s.hot.epsilon_override<<",\"memory_enabled\":"<<s.hot.memory_enabled
                <<",\"memory_scale\":"<<s.hot.memory_scale<<",\"memory_confidence\":"<<s.hot.memory_confidence
                <<",\"block_win_scale\":"<<s.hot.block_win_scale<<",\"advanced_scale\":"<<s.hot.advanced_reward_scale;
            for(int i=0;i<5;++i)out<<",\""<<STEP_OUTCOME_KEYS[i]<<"\":"<<step_outcome_value(s.hot.step_outcome,i);
            for(int i=0;i<30;++i)out<<",\""<<COMPETENCY_SETTINGS[i].key<<"\":"<<competency_value(s.hot.competency,i);
        }else{
            out<<"\"stage\":"<<s.competency.stage<<",\"games\":"<<s.competency.games<<",\"passes\":"<<s.competency.passes<<",\"skills\":[";
            for(int b=0;b<2;++b)for(int k=0;k<3;++k){const auto& p=s.competency.skills[b][k];
                out<<(b||k?",":"")<<"{\"side\":"<<b<<",\"skill\":"<<k<<",\"score\":"<<p.current<<",\"best\":"<<p.best
                    <<",\"ema\":"<<p.ema<<",\"learned\":"<<p.learned<<",\"regressions\":"<<p.regression_count
                    <<",\"rehearsals\":"<<p.rehearsal_count<<",\"successes\":"<<p.rehearsal_successes
                    <<",\"cooldown_remaining\":"<<(p.cooldown_until>s.competency.games?p.cooldown_until-s.competency.games:0)<<"}";}
            out<<"]";
        }out<<"}}";const auto text=out.str();
        if(text.size()>=(size_t)resp_size)snprintf(response,resp_size,"{\"status\":\"error\",\"msg\":\"response buffer too small\"}");
        else memcpy(response,text.c_str(),text.size()+1);return;
    }
    if (strcmp(token, "GET_STATUS") == 0) {
        TrainingStatusSnapshot status = read_training_status();
        int total = status.wins[0] + status.wins[1] + status.wins[2];
        float win_rate = total > 0 ? (float)status.wins[0] / total * 100.0f : 0.0f;
        float lose_rate = total > 0 ? (float)status.wins[1] / total * 100.0f : 0.0f;
        float draw_rate = total > 0 ? (float)status.wins[2] / total * 100.0f : 0.0f;
        snprintf(response, resp_size,
            "{\"status\":\"ok\",\"data\":{"
            "\"episode\":%d,"
            "\"phase\":%d,"
            "\"win_rate\":%.2f,"
            "\"lose_rate\":%.2f,"
            "\"draw_rate\":%.2f,"
            "\"depth\":%d,"
            "\"epsilon\":%.4f,"
            "\"learning_rate\":%.6f,"
            "\"train_repeat\":%d,"
            "\"live_two_scale\":%.2f,"
            "\"block_win_scale\":%.2f,"
            "\"advanced_scale\":%.2f,"
            "\"step_outcome_enabled\":%d,\"black_fast_win_scale\":%.4f,\"black_early_loss_scale\":%.4f,\"white_survival_scale\":%.4f,\"white_draw_length_scale\":%.4f,"
            "\"memory_enabled\":%d,"
            "\"memory_scale\":%.2f,"
            "\"memory_confidence\":%.2f,"
            "\"memory_cache\":%d,"
            "\"memory_kb\":%d,"
            "\"memory_cache_b\":%d,\"memory_kb_b\":%d,"
            "\"memory_cache_w\":%d,\"memory_kb_w\":%d,"
            "\"eval_active\":%d,\"eval_completed\":%d,\"eval_total\":%d"
            "}}",
            status.episode,
            get_curriculum_phase(status.episode),
            win_rate,
            lose_rate,
            draw_rate,
            status.depth, status.epsilon, status.learning_rate,
            status.hot.train_repeat_per_step, status.hot.live_two_scale,
            status.hot.block_win_scale, status.hot.advanced_reward_scale,
            status.hot.step_outcome.enabled, status.hot.step_outcome.black_fast_win_scale,
            status.hot.step_outcome.black_early_loss_scale, status.hot.step_outcome.white_survival_scale,
            status.hot.step_outcome.white_draw_length_scale,
            status.hot.memory_enabled, status.hot.memory_scale,
            status.hot.memory_confidence, status.cache_count, status.kb_count,
            status.cache_b, status.kb_b, status.cache_w, status.kb_w,
            status.eval_active, status.eval_completed, status.eval_total
        );
    }
    else if (strcmp(token, "SET") == 0) {
        char* key = strtok(NULL, " \t\r\n");
        char* val_str = strtok(NULL, " \t\r\n");
        if (!key || !val_str) {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"missing key or value\"}");
            return;
        }
        char* endptr = NULL;
        float val = strtof(val_str, &endptr);
        if (endptr == val_str || (endptr && *endptr != '\0') || !isfinite(val)) {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"invalid numeric value\"}");
            return;
        }
        uint32_t bit = 0;
        PendingConfig update = {};
        if(competency_key(key)>=0){
            const int k=competency_key(key);
            if(!competency_set(update.competency,k,val) ||
                ((k==0 || k==1) && val==1 && !read_training_status().independent_active)){
                snprintf(response,resp_size,"{\"status\":\"error\",\"msg\":\"invalid value or independent networks required; enable at startup\"}");return;
            }
            AcquireSRWLockExclusive(&g_pending_config_lock);
            competency_set(g_pending_config.competency,k,val);
            g_pending_config.competency_mask|=1u<<k;g_pending_config.mask|=PENDING_COMPETENCY;
            ReleaseSRWLockExclusive(&g_pending_config_lock);
            snprintf(response,resp_size,"{\"status\":\"ok\",\"msg\":\"%s queued; verify via GET_CONFIG\"}",key);return;
        }
        if(step_outcome_key(key)>=0) {
            const int k=step_outcome_key(key);
            if(!step_outcome_set(update.outcome,k,val)) {
                snprintf(response,resp_size,"{\"status\":\"error\",\"msg\":\"outcome value out of range\"}");return;
            }
            bit=PENDING_OUTCOME_FIRST<<k;
            snprintf(response,resp_size,"{\"status\":\"ok\",\"msg\":\"%s queued\"}",key);
        }
        else if (strcmp(key, "train_repeat") == 0) {
            if (val < 1.0f || val > 20.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"train_repeat out of range\"}"); return;
            }
            bit = PENDING_TRAIN_REPEAT; update.train_repeat = (int)val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"train_repeat=%d queued\"}", (int)val);
        }
        else if (strcmp(key, "learning_rate") == 0) {
            /* -1 = 恢复自动衰减（不覆盖）；正值 = 显式覆盖。 */
            if (!((val > 0.0f && val <= 1.0f) || val == -1.0f)) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"learning_rate out of range\"}"); return;
            }
            bit = PENDING_LEARNING_RATE; update.learning_rate = val;
            if (val == -1.0f)
                snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"learning_rate=auto queued\"}");
            else
                snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"learning_rate=%.6f queued\"}", val);
        }
        else if (strcmp(key, "live_two_scale") == 0) {
            if (val < 0.0f || val > 10.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"live_two_scale out of range\"}"); return;
            }
            bit = PENDING_LIVE_TWO; update.live_two = val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"live_two_scale=%.2f queued\"}", val);
        }
        else if (strcmp(key, "block_win_scale") == 0) {
            if (val < 0.0f || val > 10.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"block_win_scale out of range\"}"); return;
            }
            bit = PENDING_BLOCK_WIN; update.block_win = val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"block_win_scale=%.2f queued\"}", val);
        }
        else if (strcmp(key, "advanced_scale") == 0) {
            if (val < 0.0f || val > 10.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"advanced_scale out of range\"}"); return;
            }
            bit = PENDING_ADVANCED; update.advanced = val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"advanced_scale=%.2f queued\"}", val);
        }
        else if (strcmp(key, "epsilon") == 0) {
            if (val < -1.0f || val > 1.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"epsilon out of range\"}"); return;
            }
            bit = PENDING_EPSILON; update.epsilon = val;
            if (val >= 0) {
                snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"epsilon=%.4f queued\"}", val);
            }
            else {
                // Restore auto scheduling
                snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"epsilon auto restored\"}");
            }
        }
        else if (strcmp(key, "memory_scale") == 0) {
            if (val < 0.0f || val > 10.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"memory_scale out of range\"}"); return;
            }
            bit = PENDING_MEMORY_SCALE; update.memory_scale = val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"memory_scale=%.2f queued\"}", val);
        }
        else if (strcmp(key, "memory_confidence") == 0) {
            if (val < 0.0f || val > 1.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"memory_confidence out of range\"}"); return;
            }
            bit = PENDING_MEMORY_CONFIDENCE; update.memory_confidence = val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"memory_confidence=%.2f queued\"}", val);
        }
        else if (strcmp(key, "memory_enabled") == 0) {
            if (val != 0.0f && val != 1.0f) {
                snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"memory_enabled must be 0 or 1\"}"); return;
            }
            bit = PENDING_MEMORY_ENABLED; update.memory_enabled = (int)val;
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"memory_enabled=%d queued\"}", (int)val);
        }
        else {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"unknown key: %s\"}", key);
        }
        if (bit) {
            AcquireSRWLockExclusive(&g_pending_config_lock);
            for(int i=0;i<5;++i) if(bit & (PENDING_OUTCOME_FIRST<<i))
                step_outcome_set(g_pending_config.outcome,i,step_outcome_value(update.outcome,i));
            if (bit & PENDING_TRAIN_REPEAT) g_pending_config.train_repeat = update.train_repeat;
            if (bit & PENDING_LEARNING_RATE) g_pending_config.learning_rate = update.learning_rate;
            if (bit & PENDING_LIVE_TWO) g_pending_config.live_two = update.live_two;
            if (bit & PENDING_BLOCK_WIN) g_pending_config.block_win = update.block_win;
            if (bit & PENDING_ADVANCED) g_pending_config.advanced = update.advanced;
            if (bit & PENDING_EPSILON) g_pending_config.epsilon = update.epsilon;
            if (bit & PENDING_MEMORY_SCALE) g_pending_config.memory_scale = update.memory_scale;
            if (bit & PENDING_MEMORY_CONFIDENCE) g_pending_config.memory_confidence = update.memory_confidence;
            if (bit & PENDING_MEMORY_ENABLED) g_pending_config.memory_enabled = update.memory_enabled;
            g_pending_config.mask |= bit;
            ReleaseSRWLockExclusive(&g_pending_config_lock);
        }
    }
    else if (strcmp(token, "SAVE_SNAPSHOT") == 0) {
        if (!g_current_agent) {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"no agent\"}");
        } else {
            /* Always serialize on the Trainer safe-point thread.  The memory bank
               is updated there and cannot be captured consistently by IPC. */
            InterlockedExchange(&g_snapshot_requested, 1);
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"complete model snapshot queued\"}");
        }
    }
    else if (strcmp(token, "CONSOLE_HIDE") == 0 || strcmp(token, "CONSOLE_SHOW") == 0) {
        /* TUI 发命令控制训练自有 Win32 窗口显隐（训练程序设置的回调实现） */
        if (g_trainer_show_window_cb)
            g_trainer_show_window_cb((strcmp(token, "CONSOLE_SHOW") == 0) ? 1 : 0);
        snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"%s\"}", token);
    }
    else if (strcmp(token, "SAVE_AND_EXIT") == 0) {
        if (g_trainer_request_exit_cb) {
            g_trainer_request_exit_cb();
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"graceful exit requested\"}");
        } else {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"exit handler unavailable\"}");
        }
    }
    else if (strcmp(token, "RESUME_TRAINING") == 0) {
        sentinel_guard_resume();
        snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"sentinel hold cleared\"}");
    }
    /* Neural Probe 命令 */
    else if (strcmp(token, "PROBE_ENABLE") == 0) {
        if (neural_probe_init() == 0) {
            neural_probe_set_enabled(1);
            snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"probe enabled\"}");
        } else {
            snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"probe init failed\"}");
        }
    }
    else if (strcmp(token, "PROBE_DISABLE") == 0) {
        neural_probe_set_enabled(0);
        snprintf(response, resp_size, "{\"status\":\"ok\",\"msg\":\"probe disabled\"}");
    }
    else if (strcmp(token, "PROBE_STATUS") == 0) {
        snprintf(response, resp_size, "{\"status\":\"ok\",\"enabled\":%d,\"snapshot_id\":%llu}",
            neural_probe_enabled(), (unsigned long long)neural_probe_snapshot_id());
    }
    else {
        snprintf(response, resp_size, "{\"status\":\"error\",\"msg\":\"unknown command: %s\"}", token);
    }
}

// ---------- IPC 多客户端服务器 ----------
static volatile LONG g_pipe_client_count = 0;
static CRITICAL_SECTION g_pipe_command_lock;
static volatile LONG g_pipe_command_lock_ready = 0;

/* 每个已连接客户端独占一个 pipe instance 和工作线程。慢速/卡住的 TUI
   只会断开自己的实例，不再阻塞监听器或其他客户端。 */
static unsigned __stdcall ipc_client_thread(void* arg) {
    HANDLE hPipe = (HANDLE)arg;
    HANDLE write_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    HANDLE read_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!write_event || !read_event) {
        if (write_event) CloseHandle(write_event);
        if (read_event) CloseHandle(read_event);
        DisconnectNamedPipe(hPipe);
        CloseHandle(hPipe);
        InterlockedDecrement(&g_pipe_client_count);
        return 0;
    }

    while (g_pipe_running) {
        char status_json[2048];
        build_status_json(status_json, sizeof(status_json));

        OVERLAPPED wov = {0};
        wov.hEvent = write_event;
        ResetEvent(write_event);
        DWORD bytes_written = 0;
        int write_ok = 1;
        if (!WriteFile(hPipe, status_json, (DWORD)strlen(status_json), &bytes_written, &wov)) {
            if (GetLastError() == ERROR_IO_PENDING) {
                DWORD wr = WaitForSingleObject(write_event, 1000);
                if (wr != WAIT_OBJECT_0 ||
                    !GetOverlappedResult(hPipe, &wov, &bytes_written, FALSE)) {
                    CancelIoEx(hPipe, &wov);
                    GetOverlappedResult(hPipe, &wov, &bytes_written, TRUE);
                    write_ok = 0;
                }
            } else {
                write_ok = 0;
            }
        }
        if (!write_ok) break;

        DWORD bytes_avail = 0;
        if (!PeekNamedPipe(hPipe, NULL, 0, NULL, &bytes_avail, NULL)) break;
        if (bytes_avail > 0) {
            char cmd[256];
            DWORD bytes_read = 0;
            OVERLAPPED rov = {0};
            rov.hEvent = read_event;
            ResetEvent(read_event);
            BOOL read_ok = ReadFile(hPipe, cmd, sizeof(cmd) - 1, &bytes_read, &rov);
            if (!read_ok && GetLastError() == ERROR_IO_PENDING) {
                DWORD rr = WaitForSingleObject(read_event, 200);
                if (rr == WAIT_OBJECT_0)
                    read_ok = GetOverlappedResult(hPipe, &rov, &bytes_read, FALSE);
                else {
                    CancelIoEx(hPipe, &rov);
                    GetOverlappedResult(hPipe, &rov, &bytes_read, TRUE);
                }
            }
            if (read_ok && bytes_read > 0) {
                cmd[bytes_read] = '\0';
                char command_response[8192]; // full config + per-color competency response
                /* 多个控制端可并发读取，但修改训练配置/请求保存必须串行。 */
                EnterCriticalSection(&g_pipe_command_lock);
                process_ipc_command(cmd, command_response, sizeof(command_response));
                command_response[sizeof(command_response) - 1] = 0;
                LeaveCriticalSection(&g_pipe_command_lock);
                OVERLAPPED reply = {};
                reply.hEvent = write_event;
                ResetEvent(write_event);
                DWORD sent = 0;
                if (!WriteFile(hPipe, command_response, (DWORD)strlen(command_response), &sent, &reply)) {
                    if (GetLastError() != ERROR_IO_PENDING) break;
                    if (WaitForSingleObject(write_event, 1000) != WAIT_OBJECT_0 ||
                        !GetOverlappedResult(hPipe, &reply, &sent, FALSE)) {
                        CancelIoEx(hPipe, &reply);
                        GetOverlappedResult(hPipe, &reply, &sent, TRUE);
                        break;
                    }
                }
            }
        }
        Sleep(100);
    }

    CloseHandle(read_event);
    CloseHandle(write_event);
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
    InterlockedDecrement(&g_pipe_client_count);
    return 0;
}

static unsigned __stdcall ipc_server_thread(void* arg) {
    (void)arg;
    const char pipe_name[] = "\\\\.\\pipe\\dqn_control_pipe";

    PSECURITY_DESCRIPTOR pipe_sd = NULL;
    SECURITY_ATTRIBUTES pipe_sa = { sizeof(SECURITY_ATTRIBUTES), NULL, FALSE };
    if (ConvertStringSecurityDescriptorToSecurityDescriptorW(
        L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GRGW;;;AU)(A;;GRGW;;;IU)",
        SDDL_REVISION_1, &pipe_sd, NULL)) {
        pipe_sa.lpSecurityDescriptor = pipe_sd;
    }

    InitializeCriticalSection(&g_pipe_command_lock);
    InterlockedExchange(&g_pipe_command_lock_ready, 1);

    while (g_pipe_running) {
        OVERLAPPED ov = {0};
        ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!ov.hEvent) { Sleep(1000); continue; }

        HANDLE hPipe = CreateNamedPipeA(
            pipe_name,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES, 4096, 4096, 1000,
            pipe_sa.lpSecurityDescriptor ? &pipe_sa : NULL);
        if (hPipe == INVALID_HANDLE_VALUE) {
            CloseHandle(ov.hEvent);
            Sleep(1000);
            continue;
        }

        BOOL connected = ConnectNamedPipe(hPipe, &ov);
        DWORD connect_error = connected ? ERROR_SUCCESS : GetLastError();
        if (!connected && connect_error == ERROR_IO_PENDING) {
            DWORD wait_result = WAIT_TIMEOUT;
            for (int w = 0; w < 20 && g_pipe_running; ++w) {
                wait_result = WaitForSingleObject(ov.hEvent, 100);
                if (wait_result == WAIT_OBJECT_0) break;
            }
            if (wait_result == WAIT_OBJECT_0) {
                DWORD transferred = 0;
                connected = GetOverlappedResult(hPipe, &ov, &transferred, FALSE);
            } else {
                CancelIoEx(hPipe, &ov);
                DWORD ignored = 0;
                GetOverlappedResult(hPipe, &ov, &ignored, TRUE);
            }
        } else if (!connected && connect_error == ERROR_PIPE_CONNECTED) {
            connected = TRUE;
        }
        CloseHandle(ov.hEvent);

        if (!connected || !g_pipe_running) {
            CloseHandle(hPipe);
            continue;
        }

        InterlockedIncrement(&g_pipe_client_count);
        uintptr_t client = _beginthreadex(NULL, 0, ipc_client_thread, hPipe, 0, NULL);
        if (client) {
            CloseHandle((HANDLE)client);  /* 工作线程自行管理 pipe 生命周期 */
        } else {
            InterlockedDecrement(&g_pipe_client_count);
            DisconnectNamedPipe(hPipe);
            CloseHandle(hPipe);
        }
        /* 不等待客户端，立即创建下一实例继续 accept。 */
    }

    while (InterlockedCompareExchange(&g_pipe_client_count, 0, 0) > 0) Sleep(10);
    InterlockedExchange(&g_pipe_command_lock_ready, 0);
    /* 极端情况下某个保存命令可能仍在执行；此时让系统在进程退出时回收
       临界区，不能在客户端线程仍持有它时 DeleteCriticalSection。 */
    if (InterlockedCompareExchange(&g_pipe_client_count, 0, 0) == 0)
        DeleteCriticalSection(&g_pipe_command_lock);
    if (pipe_sd) LocalFree(pipe_sd);
    return 0;
}

void start_ipc_server(void) {
    //printf("[IPC] === start_ipc_server() 被调用 ===\n");
    fflush(stdout);

    publish_training_status(g_current_agent);
    g_pipe_running = 1;
    uintptr_t thread_handle = _beginthreadex(NULL, 0, ipc_server_thread, NULL, 0, NULL);
    g_pipe_thread = thread_handle ? (HANDLE)thread_handle : NULL;

    /* if (g_pipe_thread) {
         printf("[IPC] ✅ 线程创建成功 (句柄: %p)\n", g_pipe_thread);
     }
     else {
         printf("[IPC] ❌ 线程创建失败 (errno: %d)\n", errno);
     }*/
    fflush(stdout);
}

void stop_ipc_server(void) {
    //printf("[IPC] stop_ipc_server() 被调用\n");
    fflush(stdout);

    g_pipe_running = 0;
    if (g_pipe_thread) {
        WaitForSingleObject(g_pipe_thread, INFINITE); // Join before any agent/lock can be freed.
        CloseHandle(g_pipe_thread);
        g_pipe_thread = NULL;
        printf("[IPC] 线程已关闭\n");
    }
    printf("[IPC] 控制服务器已关闭\n");
    fflush(stdout);
}

void build_status_json(char* buf, size_t size) {
    TrainingStatusSnapshot status = read_training_status();
    int total = status.wins[0] + status.wins[1] + status.wins[2];
    float win_rate = total > 0 ? (float)status.wins[0] / total * 100.0f : 0.0f;
    float lose_rate = total > 0 ? (float)status.wins[1] / total * 100.0f : 0.0f;
    float draw_rate = total > 0 ? (float)status.wins[2] / total * 100.0f : 0.0f;

    snprintf(buf, size,
        "{\"ep\":%d,\"ph\":%d,\"wr\":%.2f,\"lr2\":%.2f,\"dr\":%.2f,\"dp\":%d,\"epc\":%.4f,\"lr\":%.6f,"
        "\"train_repeat\":%d,\"block_win_scale\":%.2f,\"advanced_scale\":%.2f,"
        "\"step_outcome_enabled\":%d,\"black_fast_win_scale\":%.4f,\"black_early_loss_scale\":%.4f,\"white_survival_scale\":%.4f,\"white_draw_length_scale\":%.4f,"
        "\"memory_enabled\":%d,\"memory_scale\":%.2f,\"memory_confidence\":%.2f,"
        "\"memory_cache\":%d,\"memory_kb\":%d,"
        "\"memory_cache_b\":%d,\"memory_kb_b\":%d,"
        "\"memory_cache_w\":%d,\"memory_kb_w\":%d,"
        "\"t\":%.3f,\"e\":%.1f,\"r\":%.1f,"
        "\"pretrain\":%d,\"pretrain_steps\":%d,\"pretrain_avg_steps\":%.3f,"
        "\"pretrain_success\":%.4f,\"pretrain_stable\":%d,"
        "\"eval_active\":%d,\"eval_completed\":%d,\"eval_total\":%d,"
        "\"stage_id\":%d,\"stage_completed\":%d,\"stage_total\":%d,\"stage_unit\":%d,"
        "\"stage_snap_done\":%d,\"stage_snap_total\":%d,\"stage_indet\":%d,\"stage_state\":%d,"
        "\"shutdown_state\":%ld,\"training_fault\":%ld,\"sentinel_paused\":%d,"
        "\"fwd_us\":%.1f,\"bwd_us\":%.1f,\"tgt_us\":%.1f,\"ph_steps\":%d,"
        "\"bfwd_us\":%.1f,\"wfwd_us\":%.1f,\"bbwd_us\":%.1f,\"wbwd_us\":%.1f,"
        "\"bsteps\":%d,\"wsteps\":%d,"
        "\"rspl_us\":%.1f,\"rfwd_us\":%.1f,\"rtgt_us\":%.1f,\"rtrn_us\":%.1f,\"rsyn_us\":%.1f,\"rcnt\":%d,"
        "\"fc4_us\":%.1f,\"fc3_us\":%.1f,\"fc2_us\":%.1f,\"fc1_us\":%.1f,"
        "\"pool_us\":%.1f,\"attn_us\":%.1f,\"embed_us\":%.1f,\"bwdcnt\":%d,"
        "\"tssetup\":%.1f,\"tsmemset\":%.1f,\"tsfwd\":%.1f,\"tshubler\":%.1f,"
        "\"tsbwd\":%.1f,\"tsvalidate\":%.1f,\"tscount\":%d}",
        status.episode,
        get_curriculum_phase(status.episode),
        win_rate,
        lose_rate,
        draw_rate,
        status.depth, status.epsilon, status.learning_rate,
        status.hot.train_repeat_per_step, status.hot.block_win_scale,
        status.hot.advanced_reward_scale,
        status.hot.step_outcome.enabled, status.hot.step_outcome.black_fast_win_scale,
        status.hot.step_outcome.black_early_loss_scale, status.hot.step_outcome.white_survival_scale,
        status.hot.step_outcome.white_draw_length_scale, status.hot.memory_enabled,
        status.hot.memory_scale, status.hot.memory_confidence,
        status.cache_count, status.kb_count,
        status.cache_b, status.kb_b, status.cache_w, status.kb_w,
        status.ep_time, status.elapsed, status.remaining,
        status.pretrain_active, status.pretrain_steps, status.pretrain_avg_steps,
        status.pretrain_success, status.pretrain_stable,
        status.eval_active, status.eval_completed, status.eval_total,
        status.stage.stage, status.stage.completed, status.stage.total, status.stage.unit,
        status.stage.snapshot_completed, status.stage.snapshot_total,
        status.stage.indeterminate, status.stage.state,
        status.shutdown_state, status.training_fault, status.sentinel_paused,
        status.phase_forward_us, status.phase_backward_us, status.phase_target_us,
        status.phase_steps,
        status.side_forward_us[0], status.side_forward_us[1],
        status.side_backward_us[0], status.side_backward_us[1],
        status.side_steps[0], status.side_steps[1],
        status.replay_sample_us, status.replay_forward_us, status.replay_target_us,
        status.replay_train_us, status.replay_sync_us, status.replay_count,
        status.bwd_fc4_us, status.bwd_fc3_us, status.bwd_fc2_us, status.bwd_fc1_us,
        status.bwd_pool_us, status.bwd_attn_us, status.bwd_embed_us, status.bwd_count,
        status.ts_setup_us, status.ts_memset_us, status.ts_fwd_us, status.ts_hubler_us,
        status.ts_bwd_us, status.ts_validate_us, status.ts_count
    );
}

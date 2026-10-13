#include "core.h"

// ---------- 全局变量定义 ----------
float g_learning_rate = 0.0005f;
int g_wins[3] = { 0, 0, 0 };
time_t g_start_time = 0;
TrainingContext g_ctx = { 0 };
DQNAgent* g_current_agent = NULL;
HotConfig g_hot_config = { 0 };
int g_pretrain_active = 0;
int g_pretrain_steps = 0;
float g_pretrain_avg_steps = 0.0f;
float g_pretrain_success_rate = 0.0f;
int g_pretrain_stable_passes = 0;
HANDLE g_pipe_thread = NULL;
volatile int g_pipe_running = 1;
void (*g_trainer_show_window_cb)(int show) = NULL;   // 训练程序设置：IPC 控制自有窗口显隐
void (*g_trainer_request_exit_cb)(void) = NULL;      // 训练程序设置：IPC 请求优雅保存退出
volatile LONG g_training_fault = 0;
volatile LONG g_shutdown_state = 0;
/* Evaluation progress is published by Core IPC and therefore owned by Core.
   Keeping the storage in the Trainer translation unit made every Core-only
   diagnostic that used IPC fail to link. */
volatile LONG g_eval_active = 0;
volatile LONG g_eval_completed = 0;
volatile LONG g_eval_total = 0;
void training_fail_detail(int reason, const char* component, int detail) {
    if (InterlockedCompareExchange(&g_training_fault, reason, 0) == 0) {
        const char* source = component ? component : "unspecified";
        printf("[FATAL] Training stopped (reason=%d component=%s detail=%d); last committed checkpoint preserved.\n",
            reason, source, detail);
        /* The trainer window is normally hidden and watchdog restarts it, so
           stdout alone loses the only useful root-cause evidence. */
        CreateDirectoryA(LOG_DIR, NULL);
        FILE* fault = fopen(LOG_DIR "training_fault.log", "a");
        if (fault) {
            SYSTEMTIME now;
            GetLocalTime(&now);
            fprintf(fault,
                "%04u-%02u-%02u %02u:%02u:%02u.%03u pid=%lu reason=%d component=%s detail=%d ep=%d step=%lld winerr=%lu\n",
                now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute,
                now.wSecond, now.wMilliseconds, GetCurrentProcessId(), reason,
                source, detail, g_ctx.total_games_played,
                (long long)(g_current_agent ? g_current_agent->global_step : -1), GetLastError());
            fflush(fault);
            _commit(_fileno(fault));
            fclose(fault);
        }
        if (g_trainer_request_exit_cb) g_trainer_request_exit_cb();
    }
}
void training_fail(int reason) { training_fail_detail(reason, "unspecified", 0); }

// ---------- PRNG 实现（只保留一个，删除 static inline 版本） ----------
void prng_init(PRNGState* prng, uint32_t seed) {
    if (seed == 0) seed = 2463534242u;
    prng->state = seed;
}

uint32_t prng_next(PRNGState* prng) {
    uint32_t x = prng->state;
    if (x == 0) x = 2463534242u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    prng->state = x;
    return x;
}

float prng_float(PRNGState* prng) {
    // Exactly representable 24-bit fraction; never rounds up to 1.0f.
    return (float)(prng_next(prng) >> 8) * (1.0f / 16777216.0f);
}

int prng_int(PRNGState* prng, int max) {
    if (max <= 0) return 0;
    // xorshift32 has the nonzero domain [1, UINT32_MAX]. Rejection removes
    // modulo bias as well as keeping every result strictly below max.
    const uint32_t bound = (uint32_t)max;
    const uint32_t limit = UINT32_MAX - UINT32_MAX % bound;
    uint32_t sample;
    do { sample = prng_next(prng); } while (sample == 0 || sample > limit);
    return (int)((sample - 1) % bound);
}

void load_params(void) {
    FILE* f = fopen("params.txt", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        float val;
        if (sscanf(line, "learning_rate = %f", &val) == 1 &&
            isfinite(val) && val > 0.0f && val <= 1.0f) {
            g_learning_rate = val;
        }
    }
    fclose(f);
    printf("[PARAMS] 已加载参数: lr=%.6f\n", g_learning_rate);
}

int apply_recovery_override(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int version = 0; float rate = 0;
    int ok = fscanf(f, "version=%d\nlearning_rate=%f", &version, &rate) == 2 &&
        version == 1 && isfinite(rate) && rate > 0 && rate <= 1;
    fclose(f);
    if (!ok || remove(path) != 0) return -1;
    g_learning_rate = rate;
    printf("[RECOVERY] Explicit post-checkpoint learning rate: %.8f\n", rate);
    return 1;
}

// ---------- 课程学习参数（去掉 static，让它们可被外部调用） ----------
int get_curriculum_phase(int global_ep) {
    if (global_ep < PHASE0_END) return 0;
    if (global_ep < PHASE1_END) return 1;
    if (global_ep < PHASE2_END) return 2;
    if (global_ep < PHASE3_END) return 3;
    return 4;
}

float compute_epsilon(int ep, int global_ep, int total_eps) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0:
        return EPS_PHASE0;
    case 1: {
        float t = (float)(global_ep - PHASE0_END) / (PHASE1_END - PHASE0_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return EPS_PHASE1_START + t * (EPS_PHASE1_END - EPS_PHASE1_START);
    }
    case 2: {
        float t = (float)(global_ep - PHASE1_END) / (PHASE2_END - PHASE1_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return EPS_PHASE2_START + t * (EPS_PHASE2_END - EPS_PHASE2_START);
    }
    case 3: {
        float t = (float)(global_ep - PHASE2_END) / (PHASE3_END - PHASE2_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return EPS_PHASE3_START + t * (EPS_PHASE3_END - EPS_PHASE3_START);
    }
    default: {
        /* total_eps 是当前阶段的局数（通常 100000），不是全课程终点。
           旧算法用 total_eps-PHASE3_END，结果恒为负数，Phase 4 第一局就
           直接掉到 EPS_PHASE4_END。 */
        float denom = (float)(PHASE4_END - PHASE3_END);
        if (denom <= 0.0f) return EPS_PHASE4_END;
        float t = (float)(global_ep - PHASE3_END) / denom;
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        float val = EPS_PHASE4_START + t * (EPS_PHASE4_END - EPS_PHASE4_START);
        return (val < EPS_PHASE4_END) ? EPS_PHASE4_END : val;
    }
    }
}

float compute_shaping_scale(int ep, int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0:
        return REWARD_SHAPING_PHASE0;
    case 1: {
        float t = (float)(global_ep - PHASE0_END) / (PHASE1_END - PHASE0_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return REWARD_SHAPING_PHASE0 + t * (REWARD_SHAPING_PHASE1 - REWARD_SHAPING_PHASE0);
    }
    case 2: {
        float t = (float)(global_ep - PHASE1_END) / (PHASE2_END - PHASE1_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return REWARD_SHAPING_PHASE1 + t * (REWARD_SHAPING_PHASE2 - REWARD_SHAPING_PHASE1);
    }
    case 3: {
        float t = (float)(global_ep - PHASE2_END) / (PHASE3_END - PHASE2_END);
        t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
        return REWARD_SHAPING_PHASE2 + t * (REWARD_SHAPING_PHASE3 - REWARD_SHAPING_PHASE2);
    }
    default:
        return REWARD_SHAPING_PHASE4;
    }
}

float compute_pool_prob(int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0: return POOL_PROB_PHASE0;
    case 1: return POOL_PROB_PHASE1;
    case 2: return POOL_PROB_PHASE2;
    case 3: return POOL_PROB_PHASE3;
    default: return POOL_PROB_PHASE4;
    }
}

float compute_minimax_prob(int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0: return MINIMAX_PROB_PHASE0;
    case 1: return MINIMAX_PROB_PHASE1;
    case 2: return MINIMAX_PROB_PHASE2;
    case 3: return MINIMAX_PROB_PHASE3;
    default: return MINIMAX_PROB_PHASE4;
    }
}

int compute_train_enabled(int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0: return TRAIN_ENABLE_PHASE0;
    default: return 1;
    }
}

int compute_target_mode(int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0: return TARGET_UPDATE_PHASE0;
    case 1: return TARGET_UPDATE_PHASE1;
    case 2: return TARGET_UPDATE_PHASE2;
    case 3: return TARGET_UPDATE_PHASE3;
    default: return TARGET_UPDATE_PHASE4;
    }
}

int compute_first_move_enabled(int global_ep) {
    int phase = get_curriculum_phase(global_ep);
    switch (phase) {
    case 0: return FIRST_MOVE_ENABLE_PHASE0;
    case 1: return FIRST_MOVE_ENABLE_PHASE1;
    case 2: return FIRST_MOVE_ENABLE_PHASE2;
    case 3: return FIRST_MOVE_ENABLE_PHASE3;
    default: return FIRST_MOVE_ENABLE_PHASE4;
    }
}

// ============================================================================
// Config file save/load
// ============================================================================
#define CONFIG_FILENAME "dqn_config.txt"

void save_config(void) {
    const char* tmp_name = CONFIG_FILENAME ".tmp";
    FILE* f = fopen(tmp_name, "w");
    if (!f) return;

    fprintf(f, "# DQN Training Config\n");
    fprintf(f, "train_repeat=%d\n", g_hot_config.train_repeat_per_step);
    fprintf(f, "learning_rate=%.6f\n", g_hot_config.learning_rate_override);
    fprintf(f, "live_two_scale=%.2f\n", g_hot_config.live_two_scale);
    fprintf(f, "block_win_scale=%.2f\n", g_hot_config.block_win_scale);
    fprintf(f, "advanced_scale=%.2f\n", g_hot_config.advanced_reward_scale);
    fprintf(f, "epsilon_override=%.4f\n", g_hot_config.epsilon_override);
    fprintf(f, "memory_enabled=%d\n", g_hot_config.memory_enabled);
    fprintf(f, "memory_scale=%.2f\n", g_hot_config.memory_scale);
    fprintf(f, "memory_confidence=%.2f\n", g_hot_config.memory_confidence);
    for(int i=0;i<5;++i) fprintf(f,"%s=%.6f\n",STEP_OUTCOME_KEYS[i],
        step_outcome_value(g_hot_config.step_outcome,i));
    for(int i=0;i<30;++i)fprintf(f,"%s=%.6f\n",COMPETENCY_SETTINGS[i].key,competency_value(g_hot_config.competency,i));

    int ok = fflush(f) == 0 && _commit(_fileno(f)) == 0;
    if (fclose(f) != 0) ok = 0;
    if (ok && MoveFileExA(tmp_name, CONFIG_FILENAME,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        printf("[CONFIG] Saved to %s\n", CONFIG_FILENAME);
    } else {
        remove(tmp_name);
        printf("[CONFIG] Save failed; previous config preserved\n");
    }
}

void load_config(void) {
    FILE* f = fopen(CONFIG_FILENAME, "r");
    if (!f) {
        printf("[CONFIG] No config file found, using defaults\n");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        char key[64];
        float val;
        if (sscanf(line, "%63[^=]=%f", key, &val) == 2 && isfinite(val)) {
            if(competency_key(key)>=0){competency_set(g_hot_config.competency,competency_key(key),val);continue;}
            if(step_outcome_key(key)>=0) {
                if(!step_outcome_set(g_hot_config.step_outcome,step_outcome_key(key),val))
                    printf("[CONFIG] Invalid outcome setting ignored: %s\n",key);
                continue;
            }
            if (strcmp(key, "train_repeat") == 0) {
                if (val >= 1.0f && val <= 20.0f)
                    g_hot_config.train_repeat_per_step = (int)val;
            }
            else if (strcmp(key, "learning_rate") == 0) {
                if ((val > 0.0f && val <= 1.0f) || val == -1.0f) {
                    /* 仅持久化覆盖值。启动时不得用它覆盖 g_learning_rate：
                       g_learning_rate 由 checkpoint 恢复 + LR 衰减决定。旧的
                       “每次启动都覆盖”行为会在每次重启把 LR 拉回旧值，抵消
                       LR_DECAY_INTERVAL 衰减，是 EP129000 灾难性遗忘的根因之一。 */
                    g_hot_config.learning_rate_override = val;
                }
            }
            else if (strcmp(key, "live_two_scale") == 0) {
                if (val >= 0.0f && val <= 10.0f) g_hot_config.live_two_scale = val;
            }
            else if (strcmp(key, "block_win_scale") == 0) {
                if (val >= 0.0f && val <= 10.0f) g_hot_config.block_win_scale = val;
            }
            else if (strcmp(key, "advanced_scale") == 0) {
                if (val >= 0.0f && val <= 10.0f) g_hot_config.advanced_reward_scale = val;
            }
            else if (strcmp(key, "epsilon_override") == 0) {
                if (val >= -1.0f && val <= 1.0f) g_hot_config.epsilon_override = val;
            }
            else if (strcmp(key, "memory_enabled") == 0) {
                if (val == 0.0f || val == 1.0f) g_hot_config.memory_enabled = (int)val;
            }
            else if (strcmp(key, "memory_scale") == 0) {
                if (val >= 0.0f && val <= 10.0f) g_hot_config.memory_scale = val;
            }
            else if (strcmp(key, "memory_confidence") == 0) {
                if (val >= 0.0f && val <= 1.0f) g_hot_config.memory_confidence = val;
            }
        }
    }

    fclose(f);
    printf("[CONFIG] Loaded from %s\n", CONFIG_FILENAME);
}

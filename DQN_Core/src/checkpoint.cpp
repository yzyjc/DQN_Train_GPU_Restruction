// checkpoint.c
#include "core.h"
#include "independent_networks.h"
#include "skill_metric_format.h"
#include "competency_training.h"
#include <memory>
#include <new>

static const uint32_t CKPT_CRC_MAGIC = 0x43524335; /* "CRC5" */
/* 当前支持的最高 checkpoint 版本（V11 = side_balance，V12 = 双方终局归因，
   V13 = 黑白独立 online/target + 按方训练状态，V14 = 技能课程与对手快照）。
   保存端、直接加载端、自动发现端必须统一引用此值，避免新格式被误判为坏档。 */
static const int CKPT_MAX_VERSION = 14;
struct CompetencyBlock {
    uint32_t magic=0x34314343u,size=sizeof(CompetencyBlock);
    CompetencyState state;
    CompetencyOpponents opponents;
};
static int64_t checkpoint_body_size(FILE* f, int version);

/* V13 在 opponent pool 之后追加双方独立网络的完整状态（online/target/按方计数）。
   放在文件末尾而不是中间：V11/V12 的既有布局与偏移一个字节都不动，
   旧的 model-only 加载、pool 偏移与 .mem 同代 CRC 逻辑保持原样。
   这里是磁盘布局，字段顺序一旦发布不可再改。

   deepseek-flash：块头带 magic + 结构体长度 + 回放语义版本。
   绝不靠 `version >= 13` 去推断块长——结构体一旦增长，旧文件会被静默按新
   结构体解析成垃圾。读端必须逐项校验，不匹配就整体拒绝加载。 */
#pragma pack(push, 1)
struct IndependentNetworksBlockHeader {
    uint32_t magic;             /* CKPT_INDEP_BLOCK_MAGIC */
    uint32_t struct_size;       /* sizeof(IndependentNetworksState) */
    int32_t  present;           /* 1 = 两方状态有效 */
    int32_t  replay_semantics;  /* 回放语义版本，见 CKPT_REPLAY_SEMANTICS_* */
    uint32_t reserved;          /* 未来扩展，写入必须为 0 */
};
struct IndependentNetworksState {
    IndependentNetworksBlockHeader header;
    NeuralNet black_online;
    NeuralNet black_target;
    independent_training::SideUpdateState black_updates;
    NeuralNet white_online;
    NeuralNet white_target;
    independent_training::SideUpdateState white_updates;
};
#pragma pack(pop)

static const uint32_t CKPT_INDEP_BLOCK_MAGIC = 0x42333149u;   /* "I13B" */
/* 回放语义版本：
   1 = 历史一 ply 非终局样本（第 4 步之前写入的回放）
   2 = 两 ply：本方决策 -> 本方下一个决策（第 4 步起的契约）
   载入时若文件版本 < 当前，回放缓冲显式清空并记录条数，绝不重新标记、
   也绝不按新公式重算旧样本目标（STEP_OUTCOME_SHAPING.md:102）。 */
static const int32_t CKPT_REPLAY_SEMANTICS_PLY1 = 1;
static const int32_t CKPT_REPLAY_SEMANTICS_PLY2 = 2;
static const int32_t CKPT_REPLAY_SEMANTICS_CURRENT = CKPT_REPLAY_SEMANTICS_PLY2;

static size_t independent_state_size(int version) {
    return version >= 13 ? sizeof(IndependentNetworksState) : 0;
}

// GPT6-Astra: only the committed checkpoint identifies the authoritative memory.
static int committed_checkpoint_generation(const char* path, uint32_t* generation);

/* A corrupt checkpoint must not make the window appear to "randomly crash".
   Move it aside without deleting it, then let the scanner try an older file. */
static int quarantine_invalid_checkpoint(const char* path) {
    if (!path || !*path) return 0;
    CreateDirectoryA("checkpoints_quarantine", NULL);
    CreateDirectoryA(LOG_DIR, NULL);
    const char* base = strrchr(path, '\\');
    base = base ? base + 1 : path;
    char destination[MAX_PATH];
    int moved = 0;
    for (int attempt = 0; attempt < 10 && !moved; ++attempt) {
        snprintf(destination, sizeof(destination),
            "checkpoints_quarantine\\%s.invalid_%llu_%lu_%d",
            base, (unsigned long long)GetTickCount64(),
            (unsigned long)GetCurrentProcessId(), attempt);
        moved = MoveFileExA(path, destination, MOVEFILE_WRITE_THROUGH) != FALSE;
    }
    FILE* log = fopen(LOG_DIR "checkpoint_recovery.log", "a");
    if (log) {
        SYSTEMTIME now; GetLocalTime(&now);
        fprintf(log, "%04u-%02u-%02u %02u:%02u:%02u %s: %s%s%s\n",
            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
            moved ? "QUARANTINED" : "QUARANTINE_FAILED", path,
            moved ? " -> " : "", moved ? destination : "");
        fclose(log);
    }
    printf("[CHECKPOINT] Invalid file %s%s%s\n", path,
        moved ? " quarantined to " : " could not be quarantined",
        moved ? destination : "");
    return moved;
}

static uint32_t crc32_stream(FILE* f, int64_t bytes) {
    uint32_t crc = 0xFFFFFFFFu;
    unsigned char buf[64 * 1024];
    while (bytes > 0) {
        size_t want = (bytes > (int64_t)sizeof(buf)) ? sizeof(buf) : (size_t)bytes;
        size_t got = fread(buf, 1, want, f);
        if (got != want) return 0;
        for (size_t i = 0; i < got; i++) {
            crc ^= buf[i];
            for (int b = 0; b < 8; b++)
                crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int)(crc & 1));
        }
        bytes -= (int64_t)got;
    }
    return crc ^ 0xFFFFFFFFu;
}

static int committed_checkpoint_generation(const char* path, uint32_t* generation) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[2] = {};
    bool ok = fread(header, sizeof(header), 1, f) == 1 &&
        header[0] >= 6 && header[0] <= CKPT_MAX_VERSION && header[1] == 0x5A4D4E54;
    const int64_t body = ok ? checkpoint_body_size(f, header[0]) : -1;
    uint32_t trailer[2] = {};
    ok = ok && body >= 0 && _fseeki64(f, 0, SEEK_END) == 0 &&
        _ftelli64(f) == body + 8 && _fseeki64(f, body, SEEK_SET) == 0 &&
        fread(trailer, sizeof(trailer), 1, f) == 1 && trailer[0] == CKPT_CRC_MAGIC;
    if (ok) {
        rewind(f);
        const uint32_t actual = crc32_stream(f, body);
        ok = !ferror(f) && !feof(f) && actual == trailer[1];
    }
    fclose(f);
    if (ok) *generation = trailer[1];
    return ok;
}

static int finite_net(const NeuralNet* net) {
    const float* values = (const float*)net;
    const size_t count = sizeof(NeuralNet) / sizeof(float);
    for (size_t i = 0; i < count; i++) if (!isfinite(values[i])) return 0;
    return 1;
}
static bool valid_replay(const Experience* entries, int count);
static bool valid_competency_block(const CompetencyBlock& b){
    return b.magic==0x34314343u && b.size==sizeof(b) && b.state.initialized==1 && competency_valid(b.state) &&
        finite_net(&b.opponents.frozen[0]) && finite_net(&b.opponents.frozen[1]) &&
        finite_net(&b.opponents.lagged[0]) && finite_net(&b.opponents.lagged[1]);
}

static bool finite_range(double value, double lo, double hi) {
    return isfinite(value) && value >= lo && value <= hi;
}
static bool valid_elapsed(double seconds) {
    // Validate before converting a floating value to time_t (including legacy files).
    return finite_range(seconds, 0, difftime(time(NULL), 0));
}
static bool valid_reward_state(float concentration, float previous, float ema,
    float shaping, const float weights[7], int depth_counter) {
    if (!finite_range(concentration, 0, 1) || !finite_range(previous, 0, 1) ||
        !finite_range(ema, 0, 1) || !isfinite(shaping) || shaping < 0 ||
        depth_counter == INT_MIN) return false;
    for (int i = 0; i < 7; ++i)
        if (!isfinite(weights[i]) || weights[i] < 0) return false;
    return true;
}
static bool valid_skills(const SkillTracker& s) {
    return s.win1_state >= SKILL_LOCKED && s.win1_state <= SKILL_MASTERED &&
        s.empty_state >= SKILL_LOCKED && s.empty_state <= SKILL_MASTERED &&
        s.block1_state >= SKILL_LOCKED && s.block1_state <= SKILL_MASTERED &&
        (s.initialized == 0 || s.initialized == 1 || s.initialized == 2 ||
            s.initialized == 3 || s.initialized == SKILL_SUITE_VERSION) &&
        finite_range(s.win1_baseline, 0, 1) && finite_range(s.empty_baseline, 0, 1) &&
        finite_range(s.block1_baseline, 0, 1) && finite_range(s.empty_shortest_baseline, 0, 1) &&
        finite_range(s.empty_steps_baseline, 0, CELL_COUNT);
}
/* side_balance 状态合法性：计数一致、倍率有限且在夹紧范围内。
   交叉校验活跃窗口内容与 samples_per_side / score_sum 一致，防止越界下标
   或伪造的计数被当作可运行状态恢复。 */
static bool valid_side_balance(const SideBalanceController& sb) {
    if (sb.write_pos < 0 || sb.write_pos >= SIDE_BALANCE_WINDOW ||
        sb.size < 0 || sb.size > SIDE_BALANCE_WINDOW ||
        sb.samples_per_side[0] < 0 || sb.samples_per_side[1] < 0 ||
        sb.since_update < 0 || sb.since_update > SIDE_BALANCE_WINDOW) return false;
    if (!isfinite(sb.score_sum[0]) || !isfinite(sb.score_sum[1]) ||
        !isfinite(sb.reward_scale[0]) || !isfinite(sb.reward_scale[1])) return false;
    /* 控制器把倍率夹在 [0.85,1.15]；放一点余量防跨版本边界。 */
    if (sb.reward_scale[0] < 0.8f || sb.reward_scale[0] > 1.2f ||
        sb.reward_scale[1] < 0.8f || sb.reward_scale[1] > 1.2f) return false;
    /* 活跃样本数（size<WINDOW 时窗口未满，只有 [0,size) 活跃；满则全窗口活跃）。 */
    const int active = (sb.size < SIDE_BALANCE_WINDOW) ? sb.size : SIDE_BALANCE_WINDOW;
    int per_side[2] = {0, 0};
    float sum[2] = {0.0f, 0.0f};
    for (int i = 0; i < active; ++i) {
        const unsigned char side = sb.samples[i].online_side;
        const float score = sb.samples[i].score;
        if (side > 1) return false;                       /* 活跃样本 side 必须 0/1 */
        if (!isfinite(score) || score < 0.0f || score > 1.0f) return false;
        per_side[side]++;
        sum[side] += score;
    }
    if (per_side[0] != sb.samples_per_side[0] || per_side[1] != sb.samples_per_side[1]) return false;
    if (fabsf(sum[0] - sb.score_sum[0]) > 1e-4f || fabsf(sum[1] - sb.score_sum[1]) > 1e-4f) return false;
    return true;
}
static bool valid_agent_metadata(const DQNAgent* a) {
    return a->global_step >= 0 && a->pretrain_consolidation_target >= 0 &&
        a->pretrain_stable_passes >= 0 && valid_skills(a->skill_tracker) &&
        valid_reward_state(a->concentration_factor, a->prev_performance,
            a->ema_winrate, a->current_shaping_scale, a->reward_weights, a->depth_adjust_counter);
}
template<class Meta> static bool valid_meta_extras(const Meta& m) {
    return valid_elapsed(m.elapsed_seconds) && m.wins[0] >= 0 && m.wins[1] >= 0 && m.wins[2] >= 0 &&
        valid_reward_state(m.concentration_factor, m.prev_performance,
            m.ema_winrate, m.current_shaping_scale, m.reward_weights, m.depth_adjust_counter);
}

static int valid_checkpoint_values(int phase, int episode, float epsilon, int depth,
    int black_idx, int white_idx, int black_count, int white_count,
    int pool_count, int pool_next, const NeuralNet* q, const NeuralNet* target,
    const NeuralNet* opponent) {
    if (phase < 0 || phase > 100 || episode < 0 || depth < 0 || depth > CELL_COUNT ||
        !isfinite(epsilon) || epsilon < 0.0f || epsilon > 1.0f ||
        black_idx < 0 || black_idx >= REPLAY_CAPACITY ||
        white_idx < 0 || white_idx >= REPLAY_CAPACITY ||
        black_count < 0 || black_count > REPLAY_CAPACITY ||
        white_count < 0 || white_count > REPLAY_CAPACITY ||
        /* Before the ring fills, samples occupy [0,count) and the next write
           position must be count. Sampling relies on this physical layout. */
        (black_count < REPLAY_CAPACITY && black_idx != black_count) ||
        (white_count < REPLAY_CAPACITY && white_idx != white_count) ||
        pool_count < 0 || pool_count > 1024 || pool_next < 0) return 0;
    return finite_net(q) && finite_net(target) && finite_net(opponent);
}

// ============================================================================
// 旧版 NeuralNet 结构体 V1/V2（用于加载 version < 3 的检查点）
// 注意：旧版没有 pool_w, pool_b, flat_cache 字段
// ============================================================================
typedef struct {
    float token_embed[3][TOKEN_EMBED_DIM];
    float pos_embed[CELL_COUNT][TOKEN_EMBED_DIM];
    float w1[256][TOKEN_EMBED_DIM];
    float b1[256];
    float w2[128][256];
    float b2[128];
    float w3[CELL_COUNT][128];
    float b3[CELL_COUNT];
    float attn_wq[ATTENTION_EMBED][TOKEN_EMBED_DIM];
    float attn_bq[ATTENTION_EMBED];
    float attn_wk[ATTENTION_EMBED][TOKEN_EMBED_DIM];
    float attn_bk[ATTENTION_EMBED];
    float attn_wv[ATTENTION_EMBED][TOKEN_EMBED_DIM];
    float attn_bv[ATTENTION_EMBED];
    float attn_wo[TOKEN_EMBED_DIM][ATTENTION_EMBED];
    float attn_bo[TOKEN_EMBED_DIM];
    float attn_q[ATTENTION_EMBED];
    float attn_k[ATTENTION_EMBED];
    float attn_v[ATTENTION_EMBED];
    float attn_scores[ATTENTION_HEADS][ATTENTION_HEAD_DIM][ATTENTION_HEAD_DIM];
    float attn_output[TOKEN_EMBED_DIM];
    float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
    float attn_context[CELL_COUNT][TOKEN_EMBED_DIM];
    // 没有 pool_w, pool_b, flat_cache
} OldNeuralNet;

// ============================================================================
// V3 NeuralNet 结构体（256/128/25 旧三层，用于加载 version 3 的检查点）
// 所有维度硬编码，不依赖当前宏定义，确保二进制兼容
// ============================================================================
#define V3_TOKEN_EMBED_DIM  16
#define V3_ATTENTION_EMBED  32
#define V3_ATTENTION_HEADS  4
#define V3_ATTENTION_HEAD_DIM  (V3_ATTENTION_EMBED / V3_ATTENTION_HEADS)  // 8
#define V3_FEATURE_DIM      128
#define V3_CELL_COUNT       25

typedef struct {
    float token_embed[3][V3_TOKEN_EMBED_DIM];
    float pos_embed[V3_CELL_COUNT][V3_TOKEN_EMBED_DIM];
    float w1[256][V3_FEATURE_DIM];
    float b1[256];
    float w2[128][256];
    float b2[128];
    float w3[V3_CELL_COUNT][128];
    float b3[V3_CELL_COUNT];
    float attn_wq[V3_ATTENTION_EMBED][V3_TOKEN_EMBED_DIM];
    float attn_bq[V3_ATTENTION_EMBED];
    float attn_wk[V3_ATTENTION_EMBED][V3_TOKEN_EMBED_DIM];
    float attn_bk[V3_ATTENTION_EMBED];
    float attn_wv[V3_ATTENTION_EMBED][V3_TOKEN_EMBED_DIM];
    float attn_bv[V3_ATTENTION_EMBED];
    float attn_wo[V3_TOKEN_EMBED_DIM][V3_ATTENTION_EMBED];
    float attn_bo[V3_TOKEN_EMBED_DIM];
    float attn_q[V3_ATTENTION_EMBED];
    float attn_k[V3_ATTENTION_EMBED];
    float attn_v[V3_ATTENTION_EMBED];
    float attn_scores[V3_ATTENTION_HEADS][V3_ATTENTION_HEAD_DIM][V3_ATTENTION_HEAD_DIM];
    float attn_output[V3_TOKEN_EMBED_DIM];
    float tokens[V3_CELL_COUNT][V3_TOKEN_EMBED_DIM];
    float attn_context[V3_CELL_COUNT][V3_TOKEN_EMBED_DIM];
    float pool_w[V3_FEATURE_DIM][V3_CELL_COUNT * V3_TOKEN_EMBED_DIM];
    float pool_b[V3_FEATURE_DIM];
    float flat_cache[V3_CELL_COUNT * V3_TOKEN_EMBED_DIM];
} V3NeuralNet;

// ============================================================================
// 辅助函数：初始化池化层
// ============================================================================
static void init_pool_layer(NeuralNet* net) {
    float pool_scale = sqrtf(2.0f / (CELL_COUNT * TOKEN_EMBED_DIM + FEATURE_DIM));
    for (int i = 0; i < FEATURE_DIM; i++) {
        net->pool_b[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            net->pool_w[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * pool_scale;
        }
    }
    memset(net->flat_cache, 0, sizeof(net->flat_cache));
}

// ============================================================================
// 辅助函数：初始化新增的 FC3 层（HIDDEN3_SIZE）
// ============================================================================
static void init_fc3_layer(NeuralNet* net) {
    float scale3 = sqrtf(2.0f / (HIDDEN2_SIZE + HIDDEN3_SIZE));
    float scale4 = sqrtf(2.0f / (HIDDEN3_SIZE + CELL_COUNT));
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        net->b3[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            net->w3[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale3;
        }
    }
    for (int i = 0; i < CELL_COUNT; i++) {
        net->b4[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            net->w4[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale4;
        }
    }
}

// ============================================================================
// 将旧版 NeuralNet (V1/V2, 256/128/25) 转换为新版 (V4, 1024/512/256/25)
// ============================================================================
static void convert_old_to_new_net(NeuralNet* dst, const OldNeuralNet* src) {
    // 复制注意力相关字段
    memcpy(dst->token_embed, src->token_embed, sizeof(src->token_embed));
    memcpy(dst->pos_embed, src->pos_embed, sizeof(src->pos_embed));
    memcpy(dst->attn_wq, src->attn_wq, sizeof(src->attn_wq));
    memcpy(dst->attn_bq, src->attn_bq, sizeof(src->attn_bq));
    memcpy(dst->attn_wk, src->attn_wk, sizeof(src->attn_wk));
    memcpy(dst->attn_bk, src->attn_bk, sizeof(src->attn_bk));
    memcpy(dst->attn_wv, src->attn_wv, sizeof(src->attn_wv));
    memcpy(dst->attn_bv, src->attn_bv, sizeof(src->attn_bv));
    memcpy(dst->attn_wo, src->attn_wo, sizeof(src->attn_wo));
    memcpy(dst->attn_bo, src->attn_bo, sizeof(src->attn_bo));
    memcpy(dst->attn_q, src->attn_q, sizeof(src->attn_q));
    memcpy(dst->attn_k, src->attn_k, sizeof(src->attn_k));
    memcpy(dst->attn_v, src->attn_v, sizeof(src->attn_v));
    memcpy(dst->attn_scores, src->attn_scores, sizeof(src->attn_scores));
    memcpy(dst->attn_output, src->attn_output, sizeof(src->attn_output));
    memcpy(dst->tokens, src->tokens, sizeof(src->tokens));
    memcpy(dst->attn_context, src->attn_context, sizeof(src->attn_context));

    // 旧版 FC 权重尺寸不同，重新初始化全部 FC 层
    float scale1 = sqrtf(2.0f / (FEATURE_DIM + HIDDEN1_SIZE));
    float scale2 = sqrtf(2.0f / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        dst->b1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < FEATURE_DIM; j++)
            dst->w1[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale1;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        dst->b2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN1_SIZE; j++)
            dst->w2[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale2;
    }
    init_fc3_layer(dst);
    init_pool_layer(dst);
}

// ============================================================================
// 将 V3 NeuralNet (256/128/25) 转换为 V4 (1024/512/256/25)
// ============================================================================
static void convert_v3_to_v4_net(NeuralNet* dst, const V3NeuralNet* src) {
    // 复制注意力和池化相关字段
    memcpy(dst->token_embed, src->token_embed, sizeof(src->token_embed));
    memcpy(dst->pos_embed, src->pos_embed, sizeof(src->pos_embed));
    memcpy(dst->attn_wq, src->attn_wq, sizeof(src->attn_wq));
    memcpy(dst->attn_bq, src->attn_bq, sizeof(src->attn_bq));
    memcpy(dst->attn_wk, src->attn_wk, sizeof(src->attn_wk));
    memcpy(dst->attn_bk, src->attn_bk, sizeof(src->attn_bk));
    memcpy(dst->attn_wv, src->attn_wv, sizeof(src->attn_wv));
    memcpy(dst->attn_bv, src->attn_bv, sizeof(src->attn_bv));
    memcpy(dst->attn_wo, src->attn_wo, sizeof(src->attn_wo));
    memcpy(dst->attn_bo, src->attn_bo, sizeof(src->attn_bo));
    memcpy(dst->attn_q, src->attn_q, sizeof(src->attn_q));
    memcpy(dst->attn_k, src->attn_k, sizeof(src->attn_k));
    memcpy(dst->attn_v, src->attn_v, sizeof(src->attn_v));
    memcpy(dst->attn_scores, src->attn_scores, sizeof(src->attn_scores));
    memcpy(dst->attn_output, src->attn_output, sizeof(src->attn_output));
    memcpy(dst->tokens, src->tokens, sizeof(src->tokens));
    memcpy(dst->attn_context, src->attn_context, sizeof(src->attn_context));
    memcpy(dst->pool_w, src->pool_w, sizeof(src->pool_w));
    memcpy(dst->pool_b, src->pool_b, sizeof(src->pool_b));
    memcpy(dst->flat_cache, src->flat_cache, sizeof(src->flat_cache));

    // FC 权重尺寸变了，重新初始化
    float scale1 = sqrtf(2.0f / (FEATURE_DIM + HIDDEN1_SIZE));
    float scale2 = sqrtf(2.0f / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        dst->b1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < FEATURE_DIM; j++)
            dst->w1[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale1;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        dst->b2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN1_SIZE; j++)
            dst->w2[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale2;
    }
    init_fc3_layer(dst);
}

/* 取「要写进文件的那张网」。
   * 非独立模式：共享 q_net（GPU 驻留时用共享 online 句柄的快照，历史行为）。
   * 独立模式：显式指定 side 时取该方 online 的驻留快照；side<0 表示
     「当前绑定方」——只为兼容既有调用点（Trainer 导出 lambda、ipc_server、Probe），
     语义与历史一致，不再依赖设备上「是谁」。 */
static const NeuralNet* model_payload(const DQNAgent* agent, int side, std::unique_ptr<NeuralNet>* frozen) {
    const int per_side = independent_training::indep_active(agent);
    const int which = per_side ? (side >= 0 ? side
        : independent_training::side_to_physical(independent_training::indep_bound(agent))) : -1;
    if (agent == g_current_agent && gpu_net_active()) {
        frozen->reset(new (std::nothrow) NeuralNet);
        if (!*frozen) return nullptr;
        int rc;
        if (which < 0) rc = gpu_net_snapshot_q(frozen->get());
        else rc = gpu_net_snapshot_side(const_cast<DQNAgent*>(agent), which, frozen->get());
        if (rc != 0) return nullptr;
        return frozen->get();
    }
    if (which < 0) return &agent->q_net;
    return independent_training::agent_online_net(const_cast<DQNAgent*>(agent),
        independent_training::side_from_physical(which));
}

int save_model(const DQNAgent* agent, const char* filename) {
    return save_model_side(agent, -1, filename);
}

/* side = 0/1 时导出该方 online；side < 0 时导出「当前绑定方」（非独立模式下即共享
   q_net）。两种情形走同一条实现，不再有第三份分支。 */
int save_model_side(const DQNAgent* agent, int side, const char* filename) {
    if (!agent || !filename) return 0;
    std::unique_ptr<NeuralNet> frozen;
    const NeuralNet* model = model_payload(agent, side, &frozen);
    if (!model) return 0;
    if (!finite_net(model)) return 0;
    char tmp_name[512];
    int name_len = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (name_len < 0 || name_len >= (int)sizeof(tmp_name)) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) return 0;
    int ok = fwrite(model, sizeof(NeuralNet), 1, f) == 1;
    if (ok && fflush(f) != 0) ok = 0;
#ifdef _WIN32
    if (ok && _commit(_fileno(f)) != 0) ok = 0;
#endif
    if (fclose(f) != 0) ok = 0;
    if (!ok) { remove(tmp_name); return 0; }
#ifdef _WIN32
    if (!MoveFileExA(tmp_name, filename, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(tmp_name); return 0;
    }
#else
    if (rename(tmp_name, filename) != 0) { remove(tmp_name); return 0; }
#endif
    return 1;
}

// GPT6-Astra: one file / one CRC / one atomic replacement for both sides.
struct ModelPairHeader { uint32_t magic, version, net_bytes, count; };
static const uint32_t MODEL_PAIR_MAGIC = 0x32504E44u; // DNP2
int save_model_pair(const DQNAgent* agent, const char* filename) {
    if (!agent || !filename || !filename[0]) return 0;
    if (!independent_training::indep_active(agent)) return save_model(agent, filename);
    std::unique_ptr<NeuralNet> black(new (std::nothrow) NeuralNet);
    std::unique_ptr<NeuralNet> white(new (std::nothrow) NeuralNet);
    if (!black || !white) return 0;
    if (agent == g_current_agent && gpu_net_active()) {
        if (gpu_net_snapshot_online_pair(const_cast<DQNAgent*>(agent), black.get(), white.get()) != 0) return 0;
    } else {
        *black = agent->indep->online(independent_training::Side::Black);
        *white = agent->indep->online(independent_training::Side::White);
    }
    if (!finite_net(black.get()) || !finite_net(white.get())) return 0;
    char temp[512];
    if (snprintf(temp, sizeof(temp), "%s.tmp", filename) >= (int)sizeof(temp)) return 0;
    FILE* f = fopen(temp, "w+b");
    if (!f) return 0;
    ModelPairHeader h = {MODEL_PAIR_MAGIC, 1, sizeof(NeuralNet), 2};
    bool ok = fwrite(&h, sizeof(h), 1, f)==1 &&
        fwrite(black.get(), sizeof(NeuralNet), 1, f)==1 &&
        fwrite(white.get(), sizeof(NeuralNet), 1, f)==1 && fflush(f)==0;
    if (ok) {
        rewind(f);
        uint32_t crc = crc32_stream(f, sizeof(h)+2*(int64_t)sizeof(NeuralNet));
        ok = !ferror(f) && !feof(f) && _fseeki64(f, 0, SEEK_END)==0 &&
            fwrite(&crc,sizeof(crc),1,f)==1 && fflush(f)==0;
    }
    if (ok) ok = _commit(_fileno(f))==0;
    if (fclose(f)!=0) ok=false;
    if (ok) ok = MoveFileExA(temp,filename,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
    if (!ok) remove(temp);
    return ok;
}

int load_model_pair_weights(const char* filename, NeuralNet* black, NeuralNet* white) {
    if (!filename || !black || !white || black==white) return 0;
    FILE* f=fopen(filename,"rb"); if(!f) return 0;
    ModelPairHeader h={};
    const int64_t body=sizeof(h)+2*(int64_t)sizeof(NeuralNet);
    bool ok=_fseeki64(f,0,SEEK_END)==0 && _ftelli64(f)==body+sizeof(uint32_t);
    rewind(f);
    ok=ok && fread(&h,sizeof(h),1,f)==1 && h.magic==MODEL_PAIR_MAGIC && h.version==1 &&
        h.net_bytes==sizeof(NeuralNet) && h.count==2;
    std::unique_ptr<NeuralNet> b(new (std::nothrow) NeuralNet), w(new (std::nothrow) NeuralNet);
    if(!b || !w){fclose(f);return 0;}
    uint32_t stored=0;
    ok=ok && fread(b.get(),sizeof(NeuralNet),1,f)==1 && fread(w.get(),sizeof(NeuralNet),1,f)==1 &&
        fread(&stored,sizeof(stored),1,f)==1 && finite_net(b.get()) && finite_net(w.get());
    if(ok){rewind(f);ok=crc32_stream(f,body)==stored && !ferror(f) && !feof(f);}
    fclose(f);
    if(ok){*black=*b;*white=*w;}
    return ok;
}

int load_model(DQNAgent* agent, const char* filename) {
    if (!agent || !filename) return 0;
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    _fseeki64(f, 0, SEEK_END);
    const int64_t size = _ftelli64(f);
    rewind(f);
    std::unique_ptr<NeuralNet> candidate(new (std::nothrow) NeuralNet);
    if (candidate) memset(candidate.get(), 0, sizeof(NeuralNet));
    int ok = 0;
    if (candidate && size == sizeof(NeuralNet)) {
        ok = fread(candidate.get(), sizeof(NeuralNet), 1, f) == 1;
    } else if (candidate && size == sizeof(OldNeuralNet)) {
        std::unique_ptr<OldNeuralNet> legacy(new (std::nothrow) OldNeuralNet{});
        if (legacy && fread(legacy.get(), sizeof(OldNeuralNet), 1, f) == 1) {
            convert_old_to_new_net(candidate.get(), legacy.get());
            ok = 1;
        }
    } else if (candidate && size == sizeof(V3NeuralNet)) {
        std::unique_ptr<V3NeuralNet> legacy(new (std::nothrow) V3NeuralNet{});
        if (legacy && fread(legacy.get(), sizeof(V3NeuralNet), 1, f) == 1) {
            convert_v3_to_v4_net(candidate.get(), legacy.get());
            ok = 1;
        }
    }
    fclose(f);
    if (!ok || !finite_net(candidate.get())) {
        printf("[LOAD] Invalid model; current weights preserved: %s\n", filename);
        return 0;
    }
    copy_net(&agent->q_net, candidate.get());
    copy_net(&agent->target_net, candidate.get());
    copy_net(&agent->opponent_net, candidate.get());
    return 1;
}

/* 读取端**按方选方**：side = 0/1 时把该文件装进该方的 online + target（独立模式），
   并按该方句柄上传到设备（驻留时）。side < 0 时就是 load_model 的历史行为（装共享
   q_net/target_net/opponent_net）。
   格式兼容：裸 `NeuralNet`（按**文件大小**识别，与 load_model 同一套判定），
   不依赖任何被旧读取端忽略的旁路标记。 */
int load_model_side(DQNAgent* agent, int side, const char* filename) {
    if (!agent || !filename) return 0;
    if (side < 0) return load_model(agent, filename);
    if (side > 1) return 0;
    if (!independent_training::indep_active(agent)) {
        printf("[LOAD] per-side load requested without independent routing: %s\n", filename);
        return 0;
    }
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    _fseeki64(f, 0, SEEK_END);
    const int64_t size = _ftelli64(f);
    rewind(f);
    std::unique_ptr<NeuralNet> candidate(new (std::nothrow) NeuralNet);
    if (candidate) memset(candidate.get(), 0, sizeof(NeuralNet));
    int ok = 0;
    if (candidate && size == sizeof(NeuralNet)) {
        ok = fread(candidate.get(), sizeof(NeuralNet), 1, f) == 1;
    } else if (candidate && size == sizeof(OldNeuralNet)) {
        std::unique_ptr<OldNeuralNet> legacy(new (std::nothrow) OldNeuralNet{});
        if (legacy && fread(legacy.get(), sizeof(OldNeuralNet), 1, f) == 1) {
            convert_old_to_new_net(candidate.get(), legacy.get());
            ok = 1;
        }
    } else if (candidate && size == sizeof(V3NeuralNet)) {
        std::unique_ptr<V3NeuralNet> legacy(new (std::nothrow) V3NeuralNet{});
        if (legacy && fread(legacy.get(), sizeof(V3NeuralNet), 1, f) == 1) {
            convert_v3_to_v4_net(candidate.get(), legacy.get());
            ok = 1;
        }
    }
    fclose(f);
    if (!ok || !finite_net(candidate.get())) {
        printf("[LOAD] Invalid per-side model; side %d preserved: %s\n", side, filename);
        return 0;
    }
    const independent_training::Side s = independent_training::side_from_physical(side);
    copy_net(independent_training::agent_online_net(agent, s), candidate.get());
    copy_net(independent_training::agent_target_net(agent, s), candidate.get());
    return 1;
}

// GPT6-Astra: never combine legacy sidecars of unknown generation.
int load_model_for_side(DQNAgent* agent, int side, const char* filename) {
    if (!agent || !filename || side<0 || side>1 || !independent_training::indep_active(agent)) return 0;
    FILE* f=fopen(filename,"rb"); if(!f) return 0;
    uint32_t magic=0; const bool read=fread(&magic,sizeof(magic),1,f)==1; fclose(f);
    if(!read) return 0;
    if(magic!=MODEL_PAIR_MAGIC) return load_model_side(agent,side,filename);
    std::unique_ptr<NeuralNet> black(new (std::nothrow) NeuralNet), white(new (std::nothrow) NeuralNet);
    if(!black || !white)return 0;
    if(!load_model_pair_weights(filename,black.get(),white.get())) return 0;
    if(agent==g_current_agent && gpu_net_active()) return 0; // live hot-load requires a separate transaction
    auto s=independent_training::side_from_physical(side);
    const NeuralNet& selected=side==0?*black:*white;
    agent->indep->online(s)=selected; agent->indep->target(s)=selected;
    return 1;
}

// ============================================================================
// 保存检查点（版本 8：400维展平池化、训练状态、回放和对手池单文件提交）
// ============================================================================
int save_checkpoint(DQNAgent* agent, int phase, int episode, int minimax_depth,
    const char* filename) {
    if (!agent || !agent->replay_black || !agent->replay_white) {
        printf("[SAVE] checkpoint 缺少回放缓冲区，取消保存\n");
        return 0;
    }
    // Trainer-thread safe point: do not trust a stale CPU shadow even if a
    // caller forgot to check its own GPU download result.
    /* 四句柄驻留：同一 safe point 内把**两方四个句柄**都回读到各自影子，
       保证存档里两方来自同一逻辑时点且互相一致。非独立模式退化为共享那一对。 */
    if (agent == g_current_agent && gpu_net_active() &&
        independent_training::indep_active(agent)) {
        if (gpu_net_download_sides_pair(agent) != 0) {
            printf("[SAVE] GPU per-side state is unavailable/untrusted; checkpoint not committed.\n");
            return 0;
        }
    } else if (agent == g_current_agent && gpu_net_active() && gpu_net_download_agent(agent) != 0) {
        printf("[SAVE] GPU state is unavailable/untrusted; checkpoint not committed.\n");
        return 0;
    }
    /* 按方路由下 checkpoint_version 必须显式升级到 13：V12 的读取方只会看到
       一对 q_net/target_net，无法还原两方，静默降级保存会让恢复后的隔离失效。
       显式把 agent->checkpoint_version 设为 12 的调用方（迁移/兼容验证）优先。 */
    if (independent_training::indep_active(agent) && agent->checkpoint_version < 12)
        agent->checkpoint_version = 13;
    const int save_version = independent_training::indep_active(agent) && agent->competency.initialized ? 14 : agent->checkpoint_version >= 13 ? 13
        : agent->checkpoint_version >= 12 ? 12
        : (g_hot_config.step_outcome.enabled ? 12 : 11);
    /* V11/V12 读取方（包括只读 model-only 提取与自动发现）仍只理解单一
       q_net/target_net。V13 文件里这两个字段继续写「当前绑定方」的权重，
       新旧读取方都能拿到一个自洽的模型；两方的精确状态在文件末尾的独立块里。 */
    const NeuralNet* canon_online = independent_training::agent_online_net(agent,
        independent_training::indep_bound(agent));
    const NeuralNet* canon_target = independent_training::agent_target_net(agent,
        independent_training::indep_bound(agent));
    if (!canon_online || !canon_target) return 0;
    if (!filename || !valid_agent_metadata(agent) || g_start_time < 0 || g_start_time > time(NULL) ||
        g_wins[0] < 0 || g_wins[1] < 0 || g_wins[2] < 0 ||
        !valid_checkpoint_values(phase, episode, agent->epsilon, minimax_depth,
            agent->replay_black_idx, agent->replay_white_idx, agent->replay_black_count,
            agent->replay_white_count, agent->opponent_pool_count, agent->opponent_pool_next,
            canon_online, canon_target, &agent->opponent_net) ||
        !isfinite(g_learning_rate) || g_learning_rate <= 0 || g_learning_rate > 1 ||
        agent->opponent_pool_size <= 0 || agent->opponent_pool_size > 1024 ||
        !agent->opponent_pool || agent->opponent_pool_count < 0 ||
        agent->opponent_pool_count > agent->opponent_pool_size || agent->opponent_pool_next < 0 ||
        !finite_net(canon_online) || !finite_net(canon_target) ||
        !finite_net(&agent->opponent_net) ||
        !valid_replay(agent->replay_black, agent->replay_black_count) ||
        !valid_replay(agent->replay_white, agent->replay_white_count)) return 0;
    /* V13：两方四套权重与按方状态必须全部有限，否则整体拒绝保存（绝不半保存）。 */
    if (save_version >= 13) {
        if (!independent_training::indep_active(agent)) return 0;
        for (int s = 0; s < 2; ++s) {
            const independent_training::Side side = independent_training::side_from_physical(s);
            if (!finite_net(independent_training::agent_online_net(agent, side)) ||
                !finite_net(independent_training::agent_target_net(agent, side))) return 0;
        }
    }
    for (int i = 0; i < agent->opponent_pool_count; ++i)
        if (!finite_net(&agent->opponent_pool[i])) return 0;
    char tmp_name[512];
    int path_len = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (path_len < 0 || path_len >= (int)sizeof(tmp_name)) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) {
        printf("[SAVE] 无法创建临时文件 %s\n", tmp_name);
        return 0;
    }

    CheckpointMeta* meta = (CheckpointMeta*)calloc(1, sizeof(CheckpointMeta));
    if (!meta) { fclose(f); remove(tmp_name); return 0; }
    meta->version = save_version;
    meta->magic = 0x5A4D4E54;
    meta->phase = phase;
    meta->episode = episode;
    meta->epsilon = agent->epsilon;
    meta->minimax_depth = minimax_depth;
    meta->replay_black_idx = agent->replay_black_idx;
    meta->replay_white_idx = agent->replay_white_idx;
    meta->replay_black_count = agent->replay_black_count;
    meta->replay_white_count = agent->replay_white_count;
    for (int i = 0; i < 3; i++) meta->wins[i] = g_wins[i];
    meta->elapsed_seconds = difftime(time(NULL), g_start_time);
    meta->concentration_factor = agent->concentration_factor;
    meta->prev_performance = agent->prev_performance;
    meta->global_step = agent->global_step;   /* BUG-015: 保存全局训练步数 */
    meta->q_net = *canon_online;
    meta->target_net = *canon_target;
    meta->opponent_net = agent->opponent_net;
    for (int i = 0; i < 7; i++) meta->reward_weights[i] = agent->reward_weights[i];
    meta->train_rng_state = agent->train_rng.state;
    meta->eval_rng_state = agent->eval_rng.state;
    meta->ema_winrate = agent->ema_winrate;
    meta->depth_adjust_counter = agent->depth_adjust_counter;
    meta->opponent_pool_count = agent->opponent_pool_count;
    meta->opponent_pool_next = agent->opponent_pool_next;
    meta->current_shaping_scale = agent->current_shaping_scale;

    long long expected_size = (long long)sizeof(CheckpointMeta);
    size_t written = fwrite(meta, sizeof(CheckpointMeta), 1, f);
    free(meta);
    if (written != 1) {
        printf("[SAVE] 写入元数据失败: %s\n", filename);
        fclose(f); remove(tmp_name); return 0;
    }

    CheckpointTrainingState training = {};
    training.input_encoding = 2;
    training.learning_rate = g_learning_rate;
    training.consolidation_target = agent->pretrain_consolidation_target;
    training.stable_passes = agent->pretrain_stable_passes;
    training.skills = agent->skill_tracker;
    /* IND-05：环境步与 HARD 同步相位随 checkpoint 原子提交 + CRC 校验。 */
    training.step_counter = agent->step_counter;
    training.target_hard_counter = agent->target_hard_counter;
    if (fwrite(&training, sizeof(training), 1, f) != 1) {
        fclose(f); remove(tmp_name); return 0;
    }
    expected_size += sizeof(training);

    /* V11：黑白平衡控制器完整状态（窗口 + 计数 + 倍率）随 checkpoint 原子提交。 */
    if (fwrite(&agent->side_balance, sizeof(agent->side_balance), 1, f) != 1) {
        printf("[SAVE] 写入 side_balance 状态失败: %s\n", filename);
        fclose(f); remove(tmp_name); return 0;
    }
    expected_size += sizeof(agent->side_balance);

    if (agent->replay_black) {
        if (fwrite(agent->replay_black, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
            printf("[SAVE] 写入黑棋回放失败: %s\n", filename);
            fclose(f); remove(tmp_name); return 0;
        }
        expected_size += (long long)sizeof(Experience) * REPLAY_CAPACITY;
    }
    if (agent->replay_white) {
        if (fwrite(agent->replay_white, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
            printf("[SAVE] 写入白棋回放失败: %s\n", filename);
            fclose(f); remove(tmp_name); return 0;
        }
        expected_size += (long long)sizeof(Experience) * REPLAY_CAPACITY;
    }
    // V7 commits the pool in the same CRC-protected file.
    int pool_meta[3] = {agent->opponent_pool_size, agent->opponent_pool_count,
        agent->opponent_pool_next % agent->opponent_pool_size};
    if (fwrite(pool_meta, sizeof(pool_meta), 1, f) != 1 ||
        fwrite(agent->opponent_pool, sizeof(NeuralNet), pool_meta[1], f) != (size_t)pool_meta[1]) {
        fclose(f); remove(tmp_name); return 0;
    }
    expected_size += sizeof(pool_meta) + (int64_t)pool_meta[1] * sizeof(NeuralNet);

    /* V13：双方 online/target 与按方训练状态。放在 pool 之后，因此 V11/V12 的
       全部既有偏移不变。同一 CRC 覆盖这一块，暂存加载失败即整体拒绝。 */
    if (save_version >= 13) {
        std::unique_ptr<IndependentNetworksState> indep(new (std::nothrow) IndependentNetworksState());
        if (!indep) { fclose(f); remove(tmp_name); return 0; }
        indep->header.magic = CKPT_INDEP_BLOCK_MAGIC;
        indep->header.struct_size = (uint32_t)sizeof(*indep);
        indep->header.present = 1;
        indep->header.replay_semantics = CKPT_REPLAY_SEMANTICS_CURRENT;
        indep->header.reserved = 0;
        indep->black_online = *independent_training::agent_online_net(agent, independent_training::Side::Black);
        indep->black_target = *independent_training::agent_target_net(agent, independent_training::Side::Black);
        indep->black_updates = agent->indep->updates(independent_training::Side::Black);
        indep->white_online = *independent_training::agent_online_net(agent, independent_training::Side::White);
        indep->white_target = *independent_training::agent_target_net(agent, independent_training::Side::White);
        indep->white_updates = agent->indep->updates(independent_training::Side::White);
        if (fwrite(indep.get(), sizeof(*indep), 1, f) != 1) {
            printf("[SAVE] 写入黑白独立网络状态失败: %s\n", filename);
            fclose(f); remove(tmp_name); return 0;
        }
        expected_size += (long long)sizeof(*indep);
    }

    if(save_version>=14){
        auto block=std::unique_ptr<CompetencyBlock>(new(std::nothrow) CompetencyBlock);
        if(!block || !agent->competency_opponents){fclose(f);remove(tmp_name);return 0;}
        block->state=agent->competency;block->opponents=*agent->competency_opponents;
        if(!valid_competency_block(*block) || fwrite(block.get(),sizeof(*block),1,f)!=1){fclose(f);remove(tmp_name);return 0;}
        expected_size+=sizeof(*block);
    }

    if (fflush(f) != 0) { fclose(f); remove(tmp_name); return 0; }
#ifdef _WIN32
    if (_commit(_fileno(f)) != 0) { fclose(f); remove(tmp_name); return 0; }
#endif
    if (fclose(f) != 0) { remove(tmp_name); return 0; }

    /* 新 V5 追加可选 CRC trailer；旧 V5 没有 trailer，加载端仍兼容。 */
    FILE* crc_file = fopen(tmp_name, "rb");
    if (!crc_file) { remove(tmp_name); return 0; }
    uint32_t crc = crc32_stream(crc_file, expected_size);
    int crc_read_ok = !ferror(crc_file);
    fclose(crc_file);
    if (!crc_read_ok) { remove(tmp_name); return 0; }
    FILE* trailer_file = fopen(tmp_name, "ab");
    uint32_t trailer[2] = { CKPT_CRC_MAGIC, crc };
    if (!trailer_file || fwrite(trailer, sizeof(uint32_t), 2, trailer_file) != 2 ||
        fflush(trailer_file) != 0) {
        if (trailer_file) fclose(trailer_file);
        remove(tmp_name); return 0;
    }
#ifdef _WIN32
    if (_commit(_fileno(trailer_file)) != 0) {
        fclose(trailer_file); remove(tmp_name); return 0;
    }
#endif
    if (fclose(trailer_file) != 0) { remove(tmp_name); return 0; }
    expected_size += (long long)sizeof(trailer);

    // 校验文件大小
    struct _stat64 st = {};
    if (_stat64(tmp_name, &st) != 0 || st.st_size != expected_size) {
        printf("[SAVE] 文件大小校验失败: 期望 %lld, 实际 %lld (%s)\n",
            expected_size, (long long)st.st_size, filename);
        remove(tmp_name);
        return 0;
    }

    /* ===== checkpoint + .mem 两文件整体提交（可恢复事务） =====
       GPT6-Astra 原子性修复：旧完整恢复点（旧主档 + 旧 .mem）必须在任何一步
       失败或进程中断后仍然存活。
       - 先解析上一次中断遗留的 .mem.bak（恢复旧点或清理陈旧副本），使旧点回到
         干净状态，后续备份才不会覆盖仍需要的旧记忆。
       - 顺序：备份旧 .mem → 写新 .mem（原子）→ 提交主档；失败时回滚 .mem。
       - 进程中断不依赖函数内回滚：加载端按代次匹配从 .mem.bak 恢复。 */
    char mem_name[600];
    char mem_bak[600];
    int mem_name_len = snprintf(mem_name, sizeof(mem_name), "%s.mem", filename);
    int mem_bak_len = snprintf(mem_bak, sizeof(mem_bak), "%s.mem.bak", filename);
    if (mem_name_len < 0 || mem_name_len >= (int)sizeof(mem_name) ||
        mem_bak_len < 0 || mem_bak_len >= (int)sizeof(mem_bak)) {
        remove(tmp_name);
        return 0;
    }

    /* GPT6-Astra: resolve both interruption windows against the committed main
       file. Ambiguous/unreadable pairs remain untouched for explicit recovery. */
    {
        if (GetFileAttributesA(mem_bak) != INVALID_FILE_ATTRIBUTES) {
            uint32_t committed = 0;
            if (!committed_checkpoint_generation(filename, &committed)) {
                printf("[SAVE] Cannot resolve memory backup against committed checkpoint: %s\n", filename);
                remove(tmp_name); return 0;
            }
            if (memory_snapshot_matches_generation(mem_name, committed)) {
                if (remove(mem_bak) != 0) { remove(tmp_name); return 0; }
            } else if (memory_snapshot_matches_generation(mem_bak, committed)) {
                if (!MoveFileExA(mem_bak, mem_name, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                    printf("[SAVE] 恢复残留备份失败: %s (error %lu)\n", mem_bak, GetLastError());
                    remove(tmp_name);
                    return 0;
                }
                printf("[SAVE] 已恢复中断遗留的旧记忆 %s -> %s\n", mem_bak, mem_name);
            } else {
                printf("[SAVE] No complete memory matches committed checkpoint: %s\n", filename);
                remove(tmp_name); return 0;
            }
        }
    }

    /* 1) 备份旧 .mem（若存在），使主档提交失败时可回滚。 */
    int have_backup = 0;
    if (GetFileAttributesA(mem_name) != INVALID_FILE_ATTRIBUTES) {
        if (!MoveFileExA(mem_name, mem_bak, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            printf("[SAVE] 备份旧记忆快照失败: %s (error %lu)\n", mem_name, GetLastError());
            remove(tmp_name);
            return 0;
        }
        have_backup = 1;
    }

    /* 2) 先写新 .mem（memory_save_full 内部 tmp -> 原子 rename）。失败则旧主档
       仍在，且旧 .mem 可回滚，旧恢复点存活。 */
    if (!memory_save_full(mem_name, episode, crc)) {
        printf("[SAVE] 同代记忆快照保存失败: %s，本次保存按失败处理\n", mem_name);
        if (have_backup &&
            !MoveFileExA(mem_bak, mem_name, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            printf("[SAVE] 回滚旧记忆失败（.mem.bak 保留，加载端仍可按代次恢复）: %s\n", mem_bak);
        remove(tmp_name);
        return 0;
    }

    /* 3) 提交主档（最后一步，决定新旧恢复点切换）。 */
#ifdef _WIN32
    if (!MoveFileExA(tmp_name, filename, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        printf("[SAVE] MoveFileEx 失败: %s (error %lu)\n", filename, GetLastError());
        if (have_backup) {
            if (!MoveFileExA(mem_bak, mem_name, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                printf("[SAVE] 回滚旧记忆失败（.mem.bak 保留，加载端仍可按代次恢复）: %s\n", mem_bak);
        } else {
            remove(mem_name);
        }
        remove(tmp_name);
        return 0;
    }
#else
    if (rename(tmp_name, filename) != 0) {
        printf("[SAVE] rename 失败: %s\n", filename);
        if (have_backup) {
            if (rename(mem_bak, mem_name) != 0)
                printf("[SAVE] 回滚旧记忆失败（.mem.bak 保留，加载端仍可按代次恢复）: %s\n", mem_bak);
        } else {
            remove(mem_name);
        }
        remove(tmp_name);
        return 0;
    }
#endif

    /* 4) 两文件均已提交成功，清理备份。 */
    if (have_backup) remove(mem_bak);

    return 1;
}

/* 测试专用：按存档自身的内容重算并写回末尾 CRC trailer。
   只用于构造「CRC 正确但内容非法」的负向用例，生产路径绝不调用。 */
int checkpoint_repair_crc_trailer(const char* filename) {
    if (!filename) return 0;
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    _fseeki64(f, 0, SEEK_END);
    const int64_t total = _ftelli64(f);
    if (total <= 8) { fclose(f); return 0; }
    const int64_t body = total - 8;
    uint32_t trailer[2] = {};
    _fseeki64(f, body, SEEK_SET);
    if (fread(trailer, sizeof(uint32_t), 2, f) != 2 || trailer[0] != CKPT_CRC_MAGIC) {
        fclose(f); return 0;
    }
    rewind(f);
    const uint32_t crc = crc32_stream(f, body);
    fclose(f);
    FILE* w = fopen(filename, "r+b");
    if (!w) return 0;
    const uint32_t fresh[2] = { CKPT_CRC_MAGIC, crc };
    const int ok = _fseeki64(w, body, SEEK_SET) == 0 &&
        fwrite(fresh, sizeof(uint32_t), 2, w) == 2 && fflush(w) == 0;
    fclose(w);
    return ok;
}

static int load_checkpoint_opponent_pool(DQNAgent* agent, const char* checkpoint_name) {
    if (agent->checkpoint_version >= 7) return 1;
    char sidecar[512];
    snprintf(sidecar, sizeof(sidecar), "%s.pool", checkpoint_name);
    if (agent->checkpoint_version == 6)
        return agent->opponent_pool_count == 0 || load_opponent_pool(agent, sidecar);
    if (load_opponent_pool(agent, sidecar)) return 1;
    load_opponent_pool(agent, CKPT_DIR "opponent_pool.bin"); // Legacy only.
    return 1;
}

/* V6~V8 的训练状态块不含同步计时；V9 起含 step_counter / target_hard_counter。 */
static size_t training_state_size(int version) {
    return version >= 9 ? sizeof(CheckpointTrainingState)
                        : sizeof(CheckpointTrainingStateV6);
}

/* V11 起在训练状态块之后追加黑白平衡控制器状态。 */
static size_t side_balance_state_size(int version) {
    return version >= 11 ? sizeof(SideBalanceController) : 0;
}

/* V13 的独立网络状态块位于 opponent pool 之后，大小见上方磁盘布局定义。 */
static int64_t checkpoint_body_size(FILE* f, int version) {
    int64_t size = (int64_t)sizeof(CheckpointMeta) +
        (version >= 6 ? (int64_t)training_state_size(version) : 0) +
        (int64_t)side_balance_state_size(version) +
        (int64_t)sizeof(Experience) * REPLAY_CAPACITY * 2;
    if (version >= 7) {
        int pool[3] = {};
        if (_fseeki64(f, size, SEEK_SET) != 0 || fread(pool, sizeof(pool), 1, f) != 1 ||
            pool[0] <= 0 || pool[0] > 1024 || pool[1] < 0 || pool[1] > pool[0] ||
            pool[2] < 0 || pool[2] >= pool[0]) return -1;
        size += sizeof(pool) + (int64_t)pool[1] * sizeof(NeuralNet);
    }
    return size + (int64_t)independent_state_size(version) + (version>=14?(int64_t)sizeof(CompetencyBlock):0);
}

struct RestoreGlobals { float learning_rate; int wins[3]; time_t start_time; };

// ============================================================================
// 加载检查点（支持版本 <3 → 4, 3 → 4 自动迁移）
// ============================================================================
static int load_checkpoint_staged(DQNAgent* agent, int* phase, int* episode, int* minimax_depth,
    const char* filename, int load_replay, RestoreGlobals& globals, uint32_t* out_crc) {
    char best_file[256];
    const char* file_to_load = filename;
    if (strcmp(filename, "checkpoint.bin") == 0) {
        int latest_ep;
        if (!find_latest_checkpoint(best_file, sizeof(best_file), &latest_ep)) {
            printf("没有找到检查点文件\n");
            return 0;
        }
        file_to_load = best_file;
        printf("找到最新检查点：%s (局数 %d)\n", best_file, latest_ep);
    }

    FILE* f = fopen(file_to_load, "rb");
    if (!f) {
        printf("[LOAD] 无法打开检查点文件 %s\n", file_to_load);
        return 0;
    }

    // ---------- 先读取版本号和魔数（与 CheckpointMeta 字段顺序一致） ----------
    int version, magic;
    if (fread(&version, sizeof(int), 1, f) != 1) {
        printf("[LOAD] 读取版本号失败\n");
        fclose(f);
        return 0;
    }
    if (fread(&magic, sizeof(int), 1, f) != 1) {
        printf("[LOAD] 读取 magic 失败\n");
        fclose(f);
        return 0;
    }

    // 验证魔数
    if (magic != 0x5A4D4E54) {
        printf("[LOAD] 检查点 magic 不匹配 (got 0x%08X, expected 0x5A4D4E54)\n", magic);
        fclose(f);
        return 0;
    }

    printf("\r\n[LOAD] 检查点版本: %d\n", version);
    if (version < 1 || version > CKPT_MAX_VERSION) { fclose(f); return 0; }
    agent->checkpoint_version = version;
    agent->competency=CompetencyState{};
    agent->pretrain_consolidation_target = agent->pretrain_stable_passes = 0;
    memset(&agent->skill_tracker, 0, sizeof(agent->skill_tracker));
    if (version < 6)
        printf("[LOAD] Legacy absolute-colour model: canonical-input semantics changed; old scores are not comparable.\n");

    if (out_crc) *out_crc = 0;   /* 无 CRC 的旧格式保持 0，表示「无同代标识」 */
    if (version >= 5) {
        const int64_t body_size = checkpoint_body_size(f, version);
        if (body_size < 0) { fclose(f); return 0; }
        _fseeki64(f, 0, SEEK_END);
        const int64_t file_size = _ftelli64(f);
        if (file_size == body_size + 8) {
            uint32_t trailer[2];
            _fseeki64(f, body_size, SEEK_SET);
            if (fread(trailer, sizeof(uint32_t), 2, f) != 2 || trailer[0] != CKPT_CRC_MAGIC) {
                printf("[LOAD] V%d checkpoint 校验尾无效\n", version); fclose(f); return 0;
            }
            rewind(f);
            uint32_t actual = crc32_stream(f, body_size);
            if (actual != trailer[1]) {
                printf("[LOAD] V%d checkpoint CRC32 不匹配，文件可能已损坏\n", version); fclose(f); return 0;
            }
            if (out_crc) *out_crc = actual;
            printf("[LOAD] V%d checkpoint CRC32 校验通过\n", version);
        } else if (file_size != body_size || version >= 6) {
            printf("[LOAD] V%d checkpoint 文件长度异常: %lld (required %lld; legacy V5 may omit CRC)\n",
                version, file_size, body_size + 8);
            fclose(f); return 0;
        } else {
            printf("[LOAD] V5 legacy checkpoint（无 CRC）\n");
        }
        rewind(f);
    }

    // ---------- 版本 >= 4：整体读取（BUG-085: 使用正确的结构体避免对齐填充错位） ----------
    if (version >= 4) {
        rewind(f);
        long long global_step = 0;

        if (version >= 5) {
            /* V5: packed 结构体，整体 fread */
            CheckpointMeta* meta = (CheckpointMeta*)calloc(1, sizeof(CheckpointMeta));
            if (!meta) { fclose(f); return 0; }
            if (fread(meta, sizeof(CheckpointMeta), 1, f) != 1) {
                printf("[LOAD] V%d checkpoint 读取失败\n", version);
                free(meta); fclose(f); return 0;
            }
            if (!valid_meta_extras(*meta) || !valid_checkpoint_values(meta->phase, meta->episode, meta->epsilon,
                meta->minimax_depth, meta->replay_black_idx, meta->replay_white_idx,
                meta->replay_black_count, meta->replay_white_count,
                meta->opponent_pool_count, meta->opponent_pool_next,
                &meta->q_net, &meta->target_net, &meta->opponent_net)) {
                printf("[LOAD] V%d checkpoint 字段或网络参数无效\n", version);
                free(meta); fclose(f); return 0;
            }
            if (version >= 6) {
                CheckpointTrainingState training = {};
                if (version >= 9) {
                    if (fread(&training, sizeof(training), 1, f) != 1) {
                        free(meta); fclose(f); return 0;
                    }
                } else {
                    /* V6~V8：读取不含同步计时的旧块，新字段保持 0（等价旧行为）。 */
                    CheckpointTrainingStateV6 old = {};
                    if (fread(&old, sizeof(old), 1, f) != 1) {
                        free(meta); fclose(f); return 0;
                    }
                    training.input_encoding = old.input_encoding;
                    training.learning_rate = old.learning_rate;
                    training.consolidation_target = old.consolidation_target;
                    training.stable_passes = old.stable_passes;
                    training.skills = old.skills;
                }
                if (training.input_encoding != 2 || !isfinite(training.learning_rate) ||
                    training.learning_rate <= 0 || training.learning_rate > 1 ||
                    training.consolidation_target < 0 || training.stable_passes < 0 ||
                    !valid_skills(training.skills)) {
                    free(meta); fclose(f); return 0;
                }
                globals.learning_rate = training.learning_rate;
                agent->pretrain_consolidation_target = training.consolidation_target;
                agent->pretrain_stable_passes = training.stable_passes;
                agent->skill_tracker = training.skills;
                agent->step_counter = training.step_counter;
                agent->target_hard_counter = training.target_hard_counter;
            }
            if (version >= 11) {
                /* V11：读取黑白平衡控制器完整状态（窗口 + 计数 + 倍率）。 */
                SideBalanceController sb = {};
                if (fread(&sb, sizeof(sb), 1, f) != 1) {
                    free(meta); fclose(f); return 0;
                }
                if (!valid_side_balance(sb)) {
                    printf("[LOAD] V%d side_balance 状态无效，使用默认无偏状态\n", version);
                    memset(&agent->side_balance, 0, sizeof(agent->side_balance));
                    agent->side_balance.reward_scale[0] = 1.0f;
                    agent->side_balance.reward_scale[1] = 1.0f;
                } else {
                    agent->side_balance = sb;
                }
            }
            *phase = meta->phase; *episode = meta->episode; *minimax_depth = meta->minimax_depth;
            agent->epsilon = meta->epsilon;
            agent->replay_black_idx = meta->replay_black_idx;
            agent->replay_white_idx = meta->replay_white_idx;
            agent->replay_black_count = meta->replay_black_count;
            agent->replay_white_count = meta->replay_white_count;
            for (int i = 0; i < 3; i++) globals.wins[i] = meta->wins[i];
            globals.start_time = time(NULL) - (time_t)meta->elapsed_seconds;
            agent->concentration_factor = meta->concentration_factor;
            agent->prev_performance = meta->prev_performance;
            agent->q_net = meta->q_net;
            agent->target_net = meta->target_net;
            agent->opponent_net = meta->opponent_net;
            for (int i = 0; i < 7; i++) agent->reward_weights[i] = meta->reward_weights[i];
            agent->train_rng.state = meta->train_rng_state;
            agent->eval_rng.state = meta->eval_rng_state;
            agent->ema_winrate = meta->ema_winrate;
            agent->depth_adjust_counter = meta->depth_adjust_counter;
            agent->current_shaping_scale = meta->current_shaping_scale;
            agent->opponent_pool_count = meta->opponent_pool_count;
            agent->opponent_pool_next = meta->opponent_pool_next;
            global_step = meta->global_step;
            free(meta);
            printf("[LOAD] V%d checkpoint，global_step=%lld\n", version, global_step);
        } else {
            /* V4 曾存在两种磁盘布局：早期无 global_step，后期在未升级版本号时
               已追加 global_step。用文件中元数据的实际长度区分，避免回放区错位。 */
            _fseeki64(f, 0, SEEK_END);
            const int64_t file_size = _ftelli64(f);
            const int64_t replay_bytes =
                (int64_t)sizeof(Experience) * REPLAY_CAPACITY * 2;
            const int64_t meta_bytes = file_size - replay_bytes;
            rewind(f);

            std::unique_ptr<CheckpointMetaV4> v4_storage(new (std::nothrow) CheckpointMetaV4);
            if (!v4_storage) { fclose(f); return 0; }
            CheckpointMetaV4& v4 = *v4_storage;
            memset(&v4, 0, sizeof(v4));
            if (meta_bytes == (int64_t)sizeof(CheckpointMetaV4WithStep)) {
                std::unique_ptr<CheckpointMetaV4WithStep> step_storage(new (std::nothrow) CheckpointMetaV4WithStep);
                if (!step_storage) { fclose(f); return 0; }
                CheckpointMetaV4WithStep& v4_step = *step_storage;
                memset(&v4_step, 0, sizeof(v4_step));
                if (fread(&v4_step, sizeof(v4_step), 1, f) != 1) {
                    printf("[LOAD] V4(global_step) checkpoint 读取失败\n");
                    fclose(f); return 0;
                }
                memcpy(&v4, &v4_step, sizeof(v4));
                global_step = v4_step.global_step;
                printf("[LOAD] V4 checkpoint（含 global_step=%lld）\n", global_step);
            } else if (meta_bytes == (int64_t)sizeof(CheckpointMetaV4)) {
                if (fread(&v4, sizeof(v4), 1, f) != 1) {
                    printf("[LOAD] V4 checkpoint 读取失败\n");
                    fclose(f); return 0;
                }
                global_step = 0;
                printf("[LOAD] V4 checkpoint，global_step=0（旧格式无此字段）\n");
            } else {
                printf("[LOAD] 无法识别 V4 元数据长度: %lld (expected %zu or %zu)\n",
                    meta_bytes, sizeof(CheckpointMetaV4), sizeof(CheckpointMetaV4WithStep));
                fclose(f); return 0;
            }
            if (!valid_meta_extras(v4) || !valid_checkpoint_values(v4.phase, v4.episode, v4.epsilon,
                v4.minimax_depth, v4.replay_black_idx, v4.replay_white_idx,
                v4.replay_black_count, v4.replay_white_count,
                v4.opponent_pool_count, v4.opponent_pool_next,
                &v4.q_net, &v4.target_net, &v4.opponent_net)) {
                printf("[LOAD] V4 checkpoint 字段或网络参数无效\n");
                fclose(f); return 0;
            }
            *phase = v4.phase; *episode = v4.episode; *minimax_depth = v4.minimax_depth;
            agent->epsilon = v4.epsilon;
            agent->replay_black_idx = v4.replay_black_idx;
            agent->replay_white_idx = v4.replay_white_idx;
            agent->replay_black_count = v4.replay_black_count;
            agent->replay_white_count = v4.replay_white_count;
            for (int i = 0; i < 3; i++) globals.wins[i] = v4.wins[i];
            globals.start_time = time(NULL) - (time_t)v4.elapsed_seconds;
            agent->concentration_factor = v4.concentration_factor;
            agent->prev_performance = v4.prev_performance;
            agent->q_net = v4.q_net;
            agent->target_net = v4.target_net;
            agent->opponent_net = v4.opponent_net;
            for (int i = 0; i < 7; i++) agent->reward_weights[i] = v4.reward_weights[i];
            agent->train_rng.state = v4.train_rng_state;
            agent->eval_rng.state = v4.eval_rng_state;
            agent->ema_winrate = v4.ema_winrate;
            agent->depth_adjust_counter = v4.depth_adjust_counter;
            agent->current_shaping_scale = v4.current_shaping_scale;
            agent->opponent_pool_count = v4.opponent_pool_count;
            agent->opponent_pool_next = v4.opponent_pool_next;
        }
        agent->global_step = global_step;

        if (load_replay) {
            if (fread(agent->replay_black, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY ||
                fread(agent->replay_white, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
                fclose(f); return 0;
            }
        } else {
            agent->replay_black_count = agent->replay_white_count = 0;
            agent->replay_black_idx = agent->replay_white_idx = 0;
        }
        if (version >= 7) {
            const int64_t pool_offset = (int64_t)sizeof(CheckpointMeta) +
                (int64_t)training_state_size(version) + (int64_t)side_balance_state_size(version) +
                (int64_t)sizeof(Experience) * REPLAY_CAPACITY * 2;
            int pool[3] = {};
            _fseeki64(f, pool_offset, SEEK_SET);
            if (fread(pool, sizeof(pool), 1, f) != 1 ||
                pool[0] != agent->opponent_pool_size || pool[1] != agent->opponent_pool_count ||
                pool[2] != agent->opponent_pool_next % pool[0] ||
                fread(agent->opponent_pool, sizeof(NeuralNet), pool[1], f) != (size_t)pool[1]) {
                fclose(f); return 0;
            }
            agent->opponent_pool_next = pool[2];
            /* V13：pool 之后是双方独立网络状态。必须整块读到并按方校验，
               任何截断/非法值都让整体加载失败，绝不半加载或拿黑方补白方。 */
            if (version >= 13) {
                std::unique_ptr<IndependentNetworksState> indep(
                    new (std::nothrow) IndependentNetworksState());
                if (!indep) { fclose(f); return 0; }
                const int64_t indep_offset = pool_offset + (int64_t)sizeof(pool) +
                    (int64_t)pool[1] * sizeof(NeuralNet);
                if (_fseeki64(f, indep_offset, SEEK_SET) != 0 ||
                    fread(indep.get(), sizeof(*indep), 1, f) != 1) {
                    printf("[LOAD] V13 独立网络状态块缺失或截断；整体拒绝加载\n");
                    fclose(f); return 0;
                }
                /* 块头必须逐项匹配：magic、结构体长度、present。
                   长度不符说明是不同版本写出的文件，绝不按新结构体强行解析。 */
                if (indep->header.magic != CKPT_INDEP_BLOCK_MAGIC) {
                    printf("[LOAD] V13 独立网络块 magic 不匹配 (0x%08X != 0x%08X)；整体拒绝加载\n",
                        indep->header.magic, CKPT_INDEP_BLOCK_MAGIC);
                    fclose(f); return 0;
                }
                if (indep->header.struct_size != (uint32_t)sizeof(*indep) ||
                    indep->header.reserved != 0) {
                    printf("[LOAD] V13 独立网络块长度不符 (文件 %u, 本程序 %u)；整体拒绝加载\n",
                        indep->header.struct_size, (uint32_t)sizeof(*indep));
                    fclose(f); return 0;
                }
                if (indep->header.present != 1 ||
                    !finite_net(&indep->black_online) || !finite_net(&indep->black_target) ||
                    !finite_net(&indep->white_online) || !finite_net(&indep->white_target)) {
                    printf("[LOAD] V13 独立网络权重非法（截断/非有限）；整体拒绝加载\n");
                    fclose(f); return 0;
                }
                if (!agent->indep) {
                    printf("[LOAD] V13 存档需要独立网络容器；分配失败，整体拒绝加载\n");
                    fclose(f); return 0;
                }
                agent->indep->online(independent_training::Side::Black) = indep->black_online;
                agent->indep->target(independent_training::Side::Black) = indep->black_target;
                agent->indep->updates(independent_training::Side::Black) = indep->black_updates;
                agent->indep->online(independent_training::Side::White) = indep->white_online;
                agent->indep->target(independent_training::Side::White) = indep->white_target;
                agent->indep->updates(independent_training::Side::White) = indep->white_updates;
                agent->indep->set_bound(independent_training::Side::Black);
                agent->indep->set_active(1);
                printf("[LOAD] V13 黑白独立网络已精确恢复（两方 online/target 与按方计数）\n");

                /* 回放语义迁移：语义版本过旧的回放不能按新契约使用，也不能
                   被重新标记。显式清空并记录条数，网络权重完全不受影响。 */
                if (indep->header.replay_semantics < CKPT_REPLAY_SEMANTICS_CURRENT) {
                    const int dropped_black = agent->replay_black_count;
                    const int dropped_white = agent->replay_white_count;
                    printf("[LOAD] 回放语义版本 %d < %d：显式清空不兼容回放 "
                        "(black=%d white=%d 条)，不重新标记、不重算目标\n",
                        indep->header.replay_semantics, CKPT_REPLAY_SEMANTICS_CURRENT,
                        dropped_black, dropped_white);
                    if (agent->replay_black && dropped_black > 0) {
                        memset(agent->replay_black, 0, sizeof(Experience) * REPLAY_CAPACITY);
                        agent->replay_black_count = 0;
                        agent->replay_black_idx = 0;
                    }
                    if (agent->replay_white && dropped_white > 0) {
                        memset(agent->replay_white, 0, sizeof(Experience) * REPLAY_CAPACITY);
                        agent->replay_white_count = 0;
                        agent->replay_white_idx = 0;
                    }
                    FILE* mig = fopen(LOG_DIR "replay_semantics_migration.log", "a");
                    if (mig) {
                        SYSTEMTIME now; GetLocalTime(&now);
                        fprintf(mig, "%04u-%02u-%02u %02u:%02u:%02u source=%s "
                            "from_semantics=%d to_semantics=%d dropped_black=%d dropped_white=%d "
                            "action=cleared_replay_weights_preserved\n",
                            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
                            file_to_load, indep->header.replay_semantics,
                            CKPT_REPLAY_SEMANTICS_CURRENT, dropped_black, dropped_white);
                        fclose(mig);
                    }
                }
            }
        }
        if(version>=14){
            auto block=std::unique_ptr<CompetencyBlock>(new(std::nothrow) CompetencyBlock);
            const int64_t offset=checkpoint_body_size(f,version)-(int64_t)sizeof(CompetencyBlock);
            if(!block || !agent->competency_opponents || _fseeki64(f,offset,SEEK_SET)!=0 ||
                fread(block.get(),sizeof(*block),1,f)!=1 || !valid_competency_block(*block)){fclose(f);return 0;}
            agent->competency=block->state;*agent->competency_opponents=block->opponents;
        }
        fclose(f);
        return load_checkpoint_opponent_pool(agent, file_to_load);
    }

    // ---------- 版本 == 3：V3 格式（256/128/25），需要迁移到 V4 ----------
    if (version == 3) {
        printf("[LOAD] 检测到 V3 检查点，正在迁移到 V4 (1024/512/256/25)...\n");

        // V3 CheckpointMeta 逐字段读取（NeuralNet 尺寸不同）
        int v3_phase, v3_episode, v3_minimax_depth;
        float v3_epsilon;
        int v3_replay_black_idx, v3_replay_white_idx;
        int v3_replay_black_count, v3_replay_white_count;
        int v3_wins[3];
        double v3_elapsed_seconds;
        float v3_concentration_factor, v3_prev_performance;
        std::unique_ptr<V3NeuralNet[]> v3_storage(new (std::nothrow) V3NeuralNet[3]);
        if (!v3_storage) { fclose(f); return 0; }
        V3NeuralNet &v3_q_net = v3_storage[0], &v3_target_net = v3_storage[1], &v3_opponent_net = v3_storage[2];
        float v3_reward_weights[7];
        uint32_t v3_train_rng_state, v3_eval_rng_state;
        float v3_ema_winrate;
        int v3_depth_adjust_counter;
        int v3_opponent_pool_count, v3_opponent_pool_next;
        float v3_current_shaping_scale;

        // 跳过 version + magic（已读取）
        if (fread(&v3_phase, sizeof(int), 1, f) != 1 ||
            fread(&v3_episode, sizeof(int), 1, f) != 1 ||
            fread(&v3_epsilon, sizeof(float), 1, f) != 1 ||
            fread(&v3_minimax_depth, sizeof(int), 1, f) != 1 ||
            fread(&v3_replay_black_idx, sizeof(int), 1, f) != 1 ||
            fread(&v3_replay_white_idx, sizeof(int), 1, f) != 1 ||
            fread(&v3_replay_black_count, sizeof(int), 1, f) != 1 ||
            fread(&v3_replay_white_count, sizeof(int), 1, f) != 1) {
            printf("[LOAD] 读取 V3 元数据失败\n"); fclose(f); return 0;
        }
        if (fread(v3_wins, sizeof(int), 3, f) != 3 ||
            fread(&v3_elapsed_seconds, sizeof(double), 1, f) != 1 ||
            fread(&v3_concentration_factor, sizeof(float), 1, f) != 1 ||
            fread(&v3_prev_performance, sizeof(float), 1, f) != 1) {
            printf("[LOAD] 读取 V3 元数据失败 (2)\n"); fclose(f); return 0;
        }
        if (fread(&v3_q_net, sizeof(V3NeuralNet), 1, f) != 1 ||
            fread(&v3_target_net, sizeof(V3NeuralNet), 1, f) != 1 ||
            fread(&v3_opponent_net, sizeof(V3NeuralNet), 1, f) != 1) {
            printf("[LOAD] 读取 V3 神经网络失败\n"); fclose(f); return 0;
        }
        if (fread(v3_reward_weights, sizeof(float), 7, f) != 7 ||
            fread(&v3_train_rng_state, sizeof(uint32_t), 1, f) != 1 ||
            fread(&v3_eval_rng_state, sizeof(uint32_t), 1, f) != 1 ||
            fread(&v3_ema_winrate, sizeof(float), 1, f) != 1 ||
            fread(&v3_depth_adjust_counter, sizeof(int), 1, f) != 1 ||
            fread(&v3_opponent_pool_count, sizeof(int), 1, f) != 1 ||
            fread(&v3_opponent_pool_next, sizeof(int), 1, f) != 1 ||
            fread(&v3_current_shaping_scale, sizeof(float), 1, f) != 1) {
            printf("[LOAD] 读取 V3 元数据失败 (3)\n"); fclose(f); return 0;
        }

        if (!valid_elapsed(v3_elapsed_seconds)) { fclose(f); return 0; }
        // 填充新版结构体
        *phase = v3_phase;
        *episode = v3_episode;
        *minimax_depth = v3_minimax_depth;
        agent->epsilon = v3_epsilon;
        agent->replay_black_idx = v3_replay_black_idx;
        agent->replay_white_idx = v3_replay_white_idx;
        agent->replay_black_count = v3_replay_black_count;
        agent->replay_white_count = v3_replay_white_count;
        for (int i = 0; i < 3; i++) globals.wins[i] = v3_wins[i];
        globals.start_time = time(NULL) - (time_t)v3_elapsed_seconds;
        agent->concentration_factor = v3_concentration_factor;
        agent->prev_performance = v3_prev_performance;

        convert_v3_to_v4_net(&agent->q_net, &v3_q_net);
        convert_v3_to_v4_net(&agent->target_net, &v3_target_net);
        convert_v3_to_v4_net(&agent->opponent_net, &v3_opponent_net);

        for (int i = 0; i < 7; i++) agent->reward_weights[i] = v3_reward_weights[i];
        agent->train_rng.state = v3_train_rng_state;
        agent->eval_rng.state = v3_eval_rng_state;
        agent->ema_winrate = v3_ema_winrate;
        agent->depth_adjust_counter = v3_depth_adjust_counter;
        agent->opponent_pool_count = v3_opponent_pool_count;
        agent->opponent_pool_next = v3_opponent_pool_next;
        agent->current_shaping_scale = v3_current_shaping_scale;

        printf("[LOAD] V3 -> V4 迁移完成！\n");

        if (load_replay) {
            printf("\r\n[LOAD] 正在读取黑棋回放缓冲区 (%d 条经验)...\n", REPLAY_CAPACITY);
            if (fread(agent->replay_black, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
                printf("[LOAD] 读取黑棋回放缓冲区失败\n"); fclose(f); return 0;
            }
            printf("\r\n[LOAD] 正在读取白棋回放缓冲区 (%d 条经验)...\n", REPLAY_CAPACITY);
            if (fread(agent->replay_white, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
                printf("[LOAD] 读取白棋回放缓冲区失败\n"); fclose(f); return 0;
            }
        }
        else {
            printf("[LOAD] 轻量模式：跳过回放缓冲区加载。\n");
            agent->replay_black_count = 0; agent->replay_white_count = 0;
            agent->replay_black_idx = 0; agent->replay_white_idx = 0;
            fclose(f);
            load_checkpoint_opponent_pool(agent, file_to_load);
            return 1;
        }

        fclose(f);
        load_checkpoint_opponent_pool(agent, file_to_load);
        printf("[LOAD] V3 检查点已加载并迁移 (Episode %d, Phase %d)\n", *episode, *phase);
        return 1;
    }

    // ---------- 版本 < 3：旧版格式，需要迁移 ----------
    if (version < 3) {
        printf("[LOAD] 检测到旧版检查点 (版本 %d)，正在迁移到新版...\n", version);

        // 由于旧版 CheckpointMeta 结构体小，我们逐字段读取
        // 先读取旧版元数据（不含新字段）
        int old_phase, old_episode, old_minimax_depth;
        float old_epsilon;
        int old_replay_black_idx, old_replay_white_idx;
        int old_replay_black_count, old_replay_white_count;
        int old_wins[3];
        double old_elapsed_seconds;
        float old_concentration_factor, old_prev_performance;
        std::unique_ptr<OldNeuralNet[]> old_storage(new (std::nothrow) OldNeuralNet[3]);
        if (!old_storage) { fclose(f); return 0; }
        OldNeuralNet &old_q_net = old_storage[0], &old_target_net = old_storage[1], &old_opponent_net = old_storage[2];
        float old_reward_weights[7];
        uint32_t old_train_rng_state, old_eval_rng_state;
        float old_ema_winrate;
        int old_depth_adjust_counter;
        int old_opponent_pool_count, old_opponent_pool_next;
        float old_current_shaping_scale;

        // 读取旧版字段（顺序必须与旧版 CheckpointMeta 一致）
        if (fread(&old_phase, sizeof(int), 1, f) != 1 ||
            fread(&old_episode, sizeof(int), 1, f) != 1 ||
            fread(&old_epsilon, sizeof(float), 1, f) != 1 ||
            fread(&old_minimax_depth, sizeof(int), 1, f) != 1 ||
            fread(&old_replay_black_idx, sizeof(int), 1, f) != 1 ||
            fread(&old_replay_white_idx, sizeof(int), 1, f) != 1 ||
            fread(&old_replay_black_count, sizeof(int), 1, f) != 1 ||
            fread(&old_replay_white_count, sizeof(int), 1, f) != 1) {
            printf("[LOAD] 读取旧版元数据失败\n");
            fclose(f);
            return 0;
        }

        if (fread(old_wins, sizeof(int), 3, f) != 3 ||
            fread(&old_elapsed_seconds, sizeof(double), 1, f) != 1 ||
            fread(&old_concentration_factor, sizeof(float), 1, f) != 1 ||
            fread(&old_prev_performance, sizeof(float), 1, f) != 1) {
            printf("[LOAD] 读取旧版元数据失败 (2)\n");
            fclose(f);
            return 0;
        }

        // 读取三个旧版神经网络
        if (fread(&old_q_net, sizeof(OldNeuralNet), 1, f) != 1 ||
            fread(&old_target_net, sizeof(OldNeuralNet), 1, f) != 1 ||
            fread(&old_opponent_net, sizeof(OldNeuralNet), 1, f) != 1) {
            printf("[LOAD] 读取旧版神经网络失败\n");
            fclose(f);
            return 0;
        }

        if (fread(old_reward_weights, sizeof(float), 7, f) != 7 ||
            fread(&old_train_rng_state, sizeof(uint32_t), 1, f) != 1 ||
            fread(&old_eval_rng_state, sizeof(uint32_t), 1, f) != 1 ||
            fread(&old_ema_winrate, sizeof(float), 1, f) != 1 ||
            fread(&old_depth_adjust_counter, sizeof(int), 1, f) != 1 ||
            fread(&old_opponent_pool_count, sizeof(int), 1, f) != 1 ||
            fread(&old_opponent_pool_next, sizeof(int), 1, f) != 1 ||
            fread(&old_current_shaping_scale, sizeof(float), 1, f) != 1) {
            printf("[LOAD] 读取旧版元数据失败 (3)\n");
            fclose(f);
            return 0;
        }

        if (!valid_elapsed(old_elapsed_seconds)) { fclose(f); return 0; }
        // ---- 填充新版结构体 ----
        *phase = old_phase;
        *episode = old_episode;
        *minimax_depth = old_minimax_depth;
        agent->epsilon = old_epsilon;
        agent->replay_black_idx = old_replay_black_idx;
        agent->replay_white_idx = old_replay_white_idx;
        agent->replay_black_count = old_replay_black_count;
        agent->replay_white_count = old_replay_white_count;
        for (int i = 0; i < 3; i++) globals.wins[i] = old_wins[i];
        globals.start_time = time(NULL) - (time_t)old_elapsed_seconds;
        agent->concentration_factor = old_concentration_factor;
        agent->prev_performance = old_prev_performance;

        // 转换旧版神经网络到新版
        convert_old_to_new_net(&agent->q_net, &old_q_net);
        convert_old_to_new_net(&agent->target_net, &old_target_net);
        convert_old_to_new_net(&agent->opponent_net, &old_opponent_net);

        for (int i = 0; i < 7; i++) agent->reward_weights[i] = old_reward_weights[i];
        agent->train_rng.state = old_train_rng_state;
        agent->eval_rng.state = old_eval_rng_state;
        agent->ema_winrate = old_ema_winrate;
        agent->depth_adjust_counter = old_depth_adjust_counter;
        agent->current_shaping_scale = old_current_shaping_scale;

        // 对手池计数（池本身需要单独加载）
        agent->opponent_pool_count = old_opponent_pool_count;
        agent->opponent_pool_next = old_opponent_pool_next;

        printf("[LOAD] 旧版检查点迁移完成！\n");

        // 读取回放缓冲区（如果请求）
        if (load_replay) {
            printf("\r\n[LOAD] 正在读取黑棋回放缓冲区 (%d 条经验)...\n", REPLAY_CAPACITY);
            if (fread(agent->replay_black, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
                printf("[LOAD] 读取黑棋回放缓冲区失败\n");
                fclose(f);
                return 0;
            }
            printf("[LOAD] 黑棋回放缓冲区读取完成。\n");

            printf("\r\n[LOAD] 正在读取白棋回放缓冲区 (%d 条经验)...\n", REPLAY_CAPACITY);
            if (fread(agent->replay_white, sizeof(Experience), REPLAY_CAPACITY, f) != REPLAY_CAPACITY) {
                printf("[LOAD] 读取白棋回放缓冲区失败\n");
                fclose(f);
                return 0;
            }
            printf("[LOAD] 白棋回放缓冲区读取完成。\n");
        }
        else {
            printf("[LOAD] 轻量模式：跳过回放缓冲区加载。\n");
            agent->replay_black_count = 0;
            agent->replay_white_count = 0;
            agent->replay_black_idx = 0;
            agent->replay_white_idx = 0;
            fclose(f);
            load_opponent_pool(agent, CKPT_DIR "opponent_pool.bin");
            printf("[LOAD] 轻量模式加载完成。\n");
            return 1;
        }

        fclose(f);
        load_opponent_pool(agent, CKPT_DIR "opponent_pool.bin");
        printf("[LOAD] 旧版检查点已加载并迁移 (Episode %d, Phase %d)\n", *episode, *phase);
        return 1;
    }

    fclose(f);
    return 0;
}

// ============================================================================

static bool valid_replay(const Experience* entries, int count) {
    for (int i = 0; i < count; ++i) {
        if (!validate_experience(&entries[i])) return false;
    }
    return true;
}

int load_checkpoint_model(NeuralNet* out, const char* filename) {
    if (!out || !filename) return 0;
    // Reuse the CRC/layout/metadata parser but keep every side effect local.
    // This is a weight candidate, not a request to roll training back in time.
    auto candidate = std::unique_ptr<DQNAgent>(new (std::nothrow) DQNAgent);
    if (!candidate) return 0;
    memset(candidate.get(), 0, sizeof(DQNAgent));
    auto pool = std::unique_ptr<NeuralNet[]>(new (std::nothrow) NeuralNet[1]);
    if (!pool) return 0;
    memset(pool.get(), 0, sizeof(NeuralNet));
    candidate->opponent_pool = pool.get();
    candidate->opponent_pool_size = 1;
    auto scratch_indep=std::unique_ptr<IndependentNetworks>(new(std::nothrow) IndependentNetworks(candidate->q_net,candidate->target_net));
    auto scratch_course=std::unique_ptr<CompetencyOpponents>(new(std::nothrow) CompetencyOpponents);
    if(!scratch_indep || !scratch_course)return 0;
    candidate->indep=scratch_indep.get();candidate->competency_opponents=scratch_course.get();
    RestoreGlobals globals = {g_learning_rate, {g_wins[0], g_wins[1], g_wins[2]}, g_start_time};
    int phase = 0, episode = 0, depth = 2;
    int ok = load_checkpoint_staged(candidate.get(), &phase, &episode, &depth,
        filename, 0, globals, NULL) && valid_agent_metadata(candidate.get()) &&
        finite_net(&candidate->q_net);
    if (ok) copy_net(out, &candidate->q_net);
    return ok;
}

int load_checkpoint(DQNAgent* agent, int* phase, int* episode, int* minimax_depth,
    const char* filename, int load_replay) {
    if (!agent || !phase || !episode || !minimax_depth || !filename ||
        agent->opponent_pool_size <= 0 || agent->opponent_pool_size > 1024 ||
        !agent->opponent_pool || (load_replay && (!agent->replay_black || !agent->replay_white))) return 0;
    std::unique_ptr<DQNAgent> staged(new (std::nothrow) DQNAgent);
    std::unique_ptr<Experience[]> black(load_replay ? new (std::nothrow) Experience[REPLAY_CAPACITY] : nullptr);
    std::unique_ptr<Experience[]> white(load_replay ? new (std::nothrow) Experience[REPLAY_CAPACITY] : nullptr);
    std::unique_ptr<NeuralNet[]> pool(new (std::nothrow) NeuralNet[agent->opponent_pool_size]);
    if (!staged || !pool || (load_replay && (!black || !white))) return 0;
    *staged = *agent;
    auto staged_indep=std::unique_ptr<IndependentNetworks>(new(std::nothrow) IndependentNetworks(agent->q_net,agent->target_net));
    auto staged_course=std::unique_ptr<CompetencyOpponents>(new(std::nothrow) CompetencyOpponents);
    if(!staged_indep || !staged_course)return 0;
    staged->indep=staged_indep.get();staged->competency_opponents=staged_course.get();
    staged->replay_black = black.get(); staged->replay_white = white.get();
    staged->opponent_pool = pool.get();
    memcpy(pool.get(), agent->opponent_pool, (size_t)agent->opponent_pool_size * sizeof(NeuralNet));
    RestoreGlobals globals = {g_learning_rate, {g_wins[0], g_wins[1], g_wins[2]}, g_start_time};
    int p = *phase, ep = *episode, depth = *minimax_depth;
    uint32_t ckpt_crc = 0;   /* 用于校验关联 .mem 是否与 checkpoint 同代 */
    if (!load_checkpoint_staged(staged.get(), &p, &ep, &depth, filename, load_replay, globals, &ckpt_crc) ||
        !valid_checkpoint_values(p, ep, staged->epsilon, depth,
            staged->replay_black_idx, staged->replay_white_idx,
            staged->replay_black_count, staged->replay_white_count,
            staged->opponent_pool_count, staged->opponent_pool_next,
            &staged->q_net, &staged->target_net, &staged->opponent_net) ||
        staged->opponent_pool_count > agent->opponent_pool_size || !valid_agent_metadata(staged.get()) ||
        globals.wins[0] < 0 || globals.wins[1] < 0 || globals.wins[2] < 0 ||
        !isfinite(globals.learning_rate) || globals.learning_rate <= 0 || globals.learning_rate > 1) return 0;
    for (int i = 0; i < staged->opponent_pool_count; ++i) if (!finite_net(&pool[i])) return 0;
    if (load_replay && (!valid_replay(black.get(), staged->replay_black_count) ||
        !valid_replay(white.get(), staged->replay_white_count))) return 0;
    /* IND 高风险修复：提交 agent 之前先解析关联记忆，保证「新网络 + 匹配记忆」
       才构成可成功续训的恢复点。
       - V10+ checkpoint 保存时必然写入带代次（V11）的 .mem；缺失、不匹配或残留
         旧 V10 文件都说明恢复点不完整，明确拒绝恢复，绝不静默降级。
       - V9- 允许无代号的历史记忆；若已有 V11 代号，仍校验配对。
       GPT6-Astra: 格式要求独立传给读取层，不能由陈旧 .mem 决定降级。 */
    {
        char mem_name[600];
        char mem_bak[600];
        int mem_ep = -1;
        int mem_ok = 1;
        int n = snprintf(mem_name, sizeof(mem_name), "%s.mem", filename);
        int nb = snprintf(mem_bak, sizeof(mem_bak), "%s.mem.bak", filename);
        if (n >= 0 && n < (int)sizeof(mem_name) && nb >= 0 && nb < (int)sizeof(mem_bak)) {
            const int new_format = staged->checkpoint_version >= 10;
            const bool memory_exists = GetFileAttributesA(mem_name) != INVALID_FILE_ATTRIBUTES;
            int loaded = memory_load_full(mem_name, &mem_ep, ckpt_crc, new_format != 0);
            if (!loaded) {
                /* 两文件保存中断恢复：.mem 缺失或代次不匹配时，尝试从 .mem.bak（保存时
                   先备份的旧记忆）恢复。仅在 .mem.bak 代次与 checkpoint CRC 严格匹配时
                   才恢复，绝不放松同代校验。 */
                const bool bak_exists = GetFileAttributesA(mem_bak) != INVALID_FILE_ATTRIBUTES;
                if (bak_exists && memory_snapshot_matches_generation(mem_bak, ckpt_crc) &&
                    MoveFileExA(mem_bak, mem_name, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                    printf("[LOAD] 从备份恢复记忆快照 %s -> %s\n", mem_bak, mem_name);
                    loaded = memory_load_full(mem_name, &mem_ep, ckpt_crc, new_format != 0);
                }
            }
            if (loaded) {
                printf("[LOAD] 恢复同代记忆快照 %s (ep=%d)\n", mem_name, mem_ep);
            } else if (new_format || memory_exists) {
                printf("[LOAD] checkpoint 缺少可用的匹配记忆快照 %s，拒绝恢复\n", mem_name);
                mem_ok = 0;
            }
        } else mem_ok = 0;
        if (!mem_ok) return 0;
    }

    // Commit only after every component has passed; file handles stay owned by the live agent.
    if (load_replay) {
        memcpy(agent->replay_black, black.get(), sizeof(Experience) * REPLAY_CAPACITY);
        memcpy(agent->replay_white, white.get(), sizeof(Experience) * REPLAY_CAPACITY);
    }
    memcpy(agent->opponent_pool, pool.get(), (size_t)agent->opponent_pool_size * sizeof(NeuralNet));
    staged->replay_black = agent->replay_black; staged->replay_white = agent->replay_white;
    staged->opponent_pool = agent->opponent_pool;
    delete agent->indep;delete agent->competency_opponents;
    staged->indep=staged_indep.release();staged->competency_opponents=staged_course.release();
    *agent = *staged;
    g_learning_rate = globals.learning_rate;
    memcpy(g_wins, globals.wins, sizeof(g_wins)); g_start_time = globals.start_time;
    *phase = p; *episode = ep; *minimax_depth = depth;

    return 1;
}

// 查找最新检查点
// ============================================================================
/* 扫描指定目录下所有 checkpoint_phase*_*.bin，更新 best（按时间最新选）
   过滤逻辑：
   - 正常 1000 倍数存档（ep % 1000 == 0, ep > 0, ep < 100000）
   - 中间档：大于最大 1000 倍数 ep，但小于该倍数 + 1000（WM_CLOSE 保存的非 1000 倍数存档）
   - 排除：ep <= 0、ep >= 100000（阶段起始/结束异常档）
   两遍扫描：第一遍找最大 1000 倍数，第二遍选符合规则的最新档 */
static int checkpoint_candidate_valid(const char* path, int filename_phase, int filename_ep) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int version = 0, magic = 0, stored_phase = -1, stored_ep = -1;
    if (fread(&version, sizeof(int), 1, f) != 1 ||
        fread(&magic, sizeof(int), 1, f) != 1 ||
        fread(&stored_phase, sizeof(int), 1, f) != 1 ||
        fread(&stored_ep, sizeof(int), 1, f) != 1 ||
        magic != 0x5A4D4E54 || version < 1 || version > CKPT_MAX_VERSION ||
        stored_phase != filename_phase || stored_ep != filename_ep) {
        fclose(f); return 0;
    }
    if (version >= 5) {
        const int64_t body_size = checkpoint_body_size(f, version);
        if (body_size < 0) { fclose(f); return 0; }
        _fseeki64(f, 0, SEEK_END);
        const int64_t file_size = _ftelli64(f);
        if ((version >= 6 && file_size != body_size + 8) ||
            (file_size != body_size && file_size != body_size + 8)) {
            fclose(f); return 0;
        }
        if (file_size == body_size + 8) {
            uint32_t trailer[2] = { 0, 0 };
            _fseeki64(f, body_size, SEEK_SET);
            if (fread(trailer, sizeof(uint32_t), 2, f) != 2 || trailer[0] != CKPT_CRC_MAGIC) {
                fclose(f); return 0;
            }
            rewind(f);
            if (crc32_stream(f, body_size) != trailer[1]) { fclose(f); return 0; }
        }

        /* CRC proves that the bytes are intact, not that they describe a usable
           training state.  Validate the cheap, critical metadata here so the
           scanner can fall back before startup commits to this candidate. */
        std::unique_ptr<CheckpointMeta> meta(new (std::nothrow) CheckpointMeta());
        if (!meta) { fclose(f); return 0; }
        rewind(f);
        if (fread(meta.get(), sizeof(CheckpointMeta), 1, f) != 1 ||
            !valid_meta_extras(*meta) ||
            !valid_checkpoint_values(meta->phase, meta->episode, meta->epsilon,
                meta->minimax_depth, meta->replay_black_idx, meta->replay_white_idx,
                meta->replay_black_count, meta->replay_white_count,
                meta->opponent_pool_count, meta->opponent_pool_next,
                &meta->q_net, &meta->target_net, &meta->opponent_net)) {
            fclose(f); return 0;
        }
        if (version >= 6) {
            CheckpointTrainingState training = {};
            if (version >= 9) {
                if (fread(&training, sizeof(training), 1, f) != 1) { fclose(f); return 0; }
            } else {
                CheckpointTrainingStateV6 old = {};
                if (fread(&old, sizeof(old), 1, f) != 1) { fclose(f); return 0; }
                training.input_encoding = old.input_encoding;
                training.learning_rate = old.learning_rate;
                training.consolidation_target = old.consolidation_target;
                training.stable_passes = old.stable_passes;
                training.skills = old.skills;
            }
            if (training.input_encoding != 2 || !isfinite(training.learning_rate) ||
                training.learning_rate <= 0 || training.learning_rate > 1 ||
                training.consolidation_target < 0 || training.stable_passes < 0 ||
                !valid_skills(training.skills)) {
                fclose(f); return 0;
            }
        }
        if (version >= 7) {
            const int64_t pool_offset = (int64_t)sizeof(CheckpointMeta) +
                (int64_t)training_state_size(version) + (int64_t)side_balance_state_size(version) +
                (int64_t)sizeof(Experience) * REPLAY_CAPACITY * 2;
            int pool[3] = {};
            _fseeki64(f, pool_offset, SEEK_SET);
            if (fread(pool, sizeof(pool), 1, f) != 1 ||
                pool[0] <= 0 || pool[0] > 1024 || pool[1] < 0 || pool[1] > pool[0] ||
                pool[2] < 0 || pool[2] >= pool[0] ||
                pool[1] != meta->opponent_pool_count ||
                pool[2] != meta->opponent_pool_next % pool[0]) {
                fclose(f); return 0;
            }
            /* V13：自动发现阶段就要拒绝截断的第二方状态，避免把一个坏档
               当成「最新存档」选出来，也避免它被隔离成 quarantine 噪音。 */
            if (version >= 13) {
                std::unique_ptr<IndependentNetworksState> indep(
                    new (std::nothrow) IndependentNetworksState());
                if (!indep) { fclose(f); return 0; }
                const int64_t indep_offset = pool_offset + (int64_t)sizeof(pool) +
                    (int64_t)pool[1] * sizeof(NeuralNet);
                if (_fseeki64(f, indep_offset, SEEK_SET) != 0 ||
                    fread(indep.get(), sizeof(*indep), 1, f) != 1 ||
                    indep->header.magic != CKPT_INDEP_BLOCK_MAGIC ||
                    indep->header.struct_size != (uint32_t)sizeof(*indep) ||
                    indep->header.reserved != 0 ||
                    indep->header.present != 1 ||
                    !finite_net(&indep->black_online) || !finite_net(&indep->black_target) ||
                    !finite_net(&indep->white_online) || !finite_net(&indep->white_target)) {
                    fclose(f); return 0;
                }
                if(version>=14){
                    auto block=std::unique_ptr<CompetencyBlock>(new(std::nothrow) CompetencyBlock);
                    if(!block || fread(block.get(),sizeof(*block),1,f)!=1 || !valid_competency_block(*block)){fclose(f);return 0;}
                }
            }
        }
    }
    fclose(f);
    return 1;
}

static void scan_checkpoint_dir(const char* dir, LONGLONG* best_mtime, char* best,
    int* best_phase, int* best_ep) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "%scheckpoint_phase*_*.bin", dir);
    /* BUG-011 修复：动态分配，避免超过 1024 个检查点被静默忽略 */
    struct { int phase, ep; LONGLONG mt; char name[64]; } *entries = NULL;
    int n = 0, cap = 0;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        int phase, ep;
        if (sscanf(fd.cFileName, "checkpoint_phase%d_%d.bin", &phase, &ep) == 2) {
            if (n >= cap) {
                int next_cap = cap ? cap * 2 : 64;
                void* grown = realloc(entries, (size_t)next_cap * sizeof(*entries));
                if (!grown) break;
                entries = (decltype(entries))grown;
                cap = next_cap;
            }
            entries[n].phase = phase; entries[n].ep = ep;
            entries[n].mt = ((LONGLONG)fd.ftLastWriteTime.dwHighDateTime << 32)
                          | (LONGLONG)fd.ftLastWriteTime.dwLowDateTime;
            snprintf(entries[n].name, 64, "%s", fd.cFileName);
            n++;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    if (n == 0 || !entries) { free(entries); return; }
    /* BUG-001 修复：支持阶段完成档（ep=PHASE?_EPISODES）和起始档（ep=0），
       按 (phase, ep) 倒序选最大进度，再按 mtime 打破平局 */
    /* 每次只校验当前最高进度候选；若它损坏再降级到下一份。避免启动时
       对数百个大 checkpoint 全部重算 CRC。 */
    for (;;) {
        int pick = -1;
        for (int i = 0; i < n; i++) {
            int ep = entries[i].ep;
            if (ep < 0) continue;
            int better_than_global = entries[i].phase > *best_phase ||
                (entries[i].phase == *best_phase && ep > *best_ep) ||
                (entries[i].phase == *best_phase && ep == *best_ep && entries[i].mt > *best_mtime);
            if (!better_than_global) continue;
            if (pick < 0 || entries[i].phase > entries[pick].phase ||
                (entries[i].phase == entries[pick].phase && ep > entries[pick].ep) ||
                (entries[i].phase == entries[pick].phase && ep == entries[pick].ep && entries[i].mt > entries[pick].mt))
                pick = i;
        }
        if (pick < 0) break;
        char candidate[256];
        snprintf(candidate, sizeof(candidate), "%s%s", dir, entries[pick].name);
        if (checkpoint_candidate_valid(candidate, entries[pick].phase, entries[pick].ep)) {
            *best_phase = entries[pick].phase;
            *best_ep = entries[pick].ep;
            *best_mtime = entries[pick].mt;
            snprintf(best, 256, "%s", candidate);
            break;
        }
        quarantine_invalid_checkpoint(candidate);
        entries[pick].phase = -1; /* 损坏或名实不符，继续尝试次新档 */
        entries[pick].ep = -1;
    }
    free(entries);   /* BUG-011: 释放动态分配 */
}

int find_latest_checkpoint(char* filename, size_t size, int* out_episode) {
    /* 同时扫描 checkpoints\ 和 checkpoints_temp\，选时间最新的档
       （temp 里是 WM_CLOSE 保存的中间档，可能比正式档更新） */
    LONGLONG best_mtime = -1;
    char best[256] = "";
    int best_phase = -1;
    int best_ep = -1;
    scan_checkpoint_dir(CKPT_DIR, &best_mtime, best, &best_phase, &best_ep);
    scan_checkpoint_dir(CKPT_TEMP_DIR, &best_mtime, best, &best_phase, &best_ep);
    if (best_mtime >= 0) {
        strncpy(filename, best, size - 1);
        filename[size - 1] = '\0';
        if (out_episode) *out_episode = best_ep;
        return 1;
    }
    if (out_episode) *out_episode = -1;
    return 0;
}

// ============================================================================
// 回滚日志
// ============================================================================
void rollback_log(const char* log_filename, int target_episode) {
    char backup_name[256];
    time_t now = time(NULL);
    snprintf(backup_name, sizeof(backup_name), "%s_backup_%lld", log_filename, (long long)now);
    if (rename(log_filename, backup_name) != 0) {
        printf("[ROLLBACK] 无法重命名 %s，跳过\n", log_filename);
        return;
    }
    FILE* src = fopen(backup_name, "r");
    FILE* dst = fopen(log_filename, "w");
    if (!src || !dst) {
        printf("[ROLLBACK] 无法打开文件\n");
        if (src) fclose(src);
        if (dst) fclose(dst);
        rename(backup_name, log_filename);
        return;
    }
    char line[1024];
    int ep;
    int kept = 0;
    while (fgets(line, sizeof(line), src)) {
        if (sscanf(line, "%d", &ep) == 1) {
            if (ep <= target_episode) {
                fputs(line, dst);
                kept++;
            }
        }
        else {
            fputs(line, dst);
        }
    }
    fclose(src);
    fclose(dst);
    remove(backup_name);
    printf("[ROLLBACK] %s 已回滚到 episode %d，保留 %d 行\n",
        log_filename, target_episode, kept);
}

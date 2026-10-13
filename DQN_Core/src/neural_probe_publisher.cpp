/* neural_probe_publisher.cpp — Trainer 端 Neural Probe 发布器
   创建 Windows Shared Memory，定期将 NeuralNet 快照写入双缓冲区。
   关闭时 perf impact 接近 0（只检查一个 bool）。 */
#include "neural_probe_protocol.h"
#include "core.h"
#include "independent_networks.h"
#include "network_input.h"
#include "probe_diagnostics.h"
#include <memory>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 内部状态 ========== */
static HANDLE g_probe_mapping = NULL;    /* 共享内存文件映射句柄 */
static void*  g_probe_base = NULL;       /* 共享内存基址 */
static int    g_probe_initialized = 0;

static NeuralProbeHeader* g_hdr = NULL;
static ProbeLayerDesc*    g_topology = NULL;
static uint8_t*           g_buf0 = NULL;
static uint8_t*           g_buf1 = NULL;
static SRWLOCK            g_probe_lock = SRWLOCK_INIT;
static const uint32_t     kProbeLayerCount = 11;
static uint64_t           g_probe_state_offset = 0;
static int                g_have_previous_publish = 0;
static ProbeParameterHistory g_parameter_history;

/* ========== 内部函数 ========== */

/* 计算 snapshot buffer 布局，填充 topology 并返回 buffer 大小 */
static uint64_t build_topology_and_calc_size(void) {
    uint64_t off = 0;

    /* 每个描述符只描述一个真实的二维权重张量。不要把多个张量塞进
       weight_count；Matrix/3D 客户端依赖 input_size*output_size 精确定位。 */
    g_topology[0] = { 0, PROBE_LAYER_EMBED, "TokenEmbedding",
        16, 3, 2, {3, 16, 0, 0}, off, 3 * 16,
        0, 0, 0, 0, 0, 0, {0} };
    off += 3 * 16 * 4;

    g_topology[1] = { 1, PROBE_LAYER_EMBED, "PositionEmbedding",
        16, 25, 2, {25, 16, 0, 0}, off, 25 * 16,
        0, 0, 0, 0, 0, 0, {0} };
    off += 25 * 16 * 4;

    g_topology[2] = { 2, PROBE_LAYER_ATTENTION, "Attention_Q",
        16, 32, 2, {32, 16, 0, 0}, off, 32 * 16,
        off + 32*16*4, 32, 0, 0, 0, 0, {0} };
    off += (32 * 16 + 32) * 4;

    g_topology[3] = { 3, PROBE_LAYER_ATTENTION, "Attention_K",
        16, 32, 2, {32, 16, 0, 0}, off, 32 * 16,
        off + 32*16*4, 32, 0, 0, 0, 0, {0} };
    off += (32 * 16 + 32) * 4;

    g_topology[4] = { 4, PROBE_LAYER_ATTENTION, "Attention_V",
        16, 32, 2, {32, 16, 0, 0}, off, 32 * 16,
        off + 32*16*4, 32, 0, 0, 0, 0, {0} };
    off += (32 * 16 + 32) * 4;

    g_topology[5] = { 5, PROBE_LAYER_ATTENTION, "Attention_O",
        32, 16, 2, {16, 32, 0, 0}, off, 16 * 32,
        off + 16*32*4, 16, 0, 0, 0, 0, {0} };
    off += (16 * 32 + 16) * 4;

    /* Layer 6: Pooling（展平 Attention_O[25][16]→flat[400]，全 400 维输入） */
    g_topology[6] = { 6, PROBE_LAYER_POOLING, "Pooling",
        400, 128, 2, {128, 400, 0, 0},
        off, 128 * 400,        /* pool_w[128][400] */
        off + 128*400*4, 128,  /* pool_b[128] */
        0, 0, 0, 0, {0} };
    off += (128*400 + 128) * 4;

    /* Layer 7: FC1 */
    g_topology[7] = { 7, PROBE_LAYER_LINEAR, "FC1",
        128, 1024, 2, {1024, 128, 0, 0},
        off, 1024 * 128,
        off + 1024*128*4, 1024,
        0, 0, 0, 0, {0} };
    off += (1024*128 + 1024) * 4;

    /* Layer 8: FC2 */
    g_topology[8] = { 8, PROBE_LAYER_LINEAR, "FC2",
        1024, 512, 2, {512, 1024, 0, 0},
        off, 512 * 1024,
        off + 512*1024*4, 512,
        0, 0, 0, 0, {0} };
    off += (512*1024 + 512) * 4;

    /* Layer 9: FC3 */
    g_topology[9] = { 9, PROBE_LAYER_LINEAR, "FC3",
        512, 256, 2, {256, 512, 0, 0},
        off, 256 * 512,
        off + 256*512*4, 256,
        0, 0, 0, 0, {0} };
    off += (256*512 + 256) * 4;

    /* Layer 10: FC4 (Output) */
    g_topology[10] = { 10, PROBE_LAYER_OUTPUT, "FC4_Output",
        256, 25, 2, {25, 256, 0, 0},
        off, 25 * 256,
        off + 25*256*4, 25,
        0, 0, 0, 0, {0} };
    off += (25*256 + 25) * 4;

    /* 输出激活、有效梯度、矩阵输入激活分别连续存放。它们不改变参数
       张量的磁盘顺序，因此旧的参数复制代码仍可逐块工作。 */
    for (uint32_t i = 0; i < kProbeLayerCount; ++i) {
        g_topology[i].activation_offset = off;
        g_topology[i].activation_count = g_topology[i].output_size;
        off += g_topology[i].activation_count * sizeof(float);
    }
    for (uint32_t i = 0; i < kProbeLayerCount; ++i) {
        g_topology[i].gradient_offset = off;
        g_topology[i].gradient_count = g_topology[i].weight_count;
        off += g_topology[i].gradient_count * sizeof(float);
    }
    for (uint32_t i = 0; i < kProbeLayerCount; ++i) {
        g_topology[i]._reserved[PROBE_LAYER_INPUT_ACTIVATION_OFFSET] = off;
        g_topology[i]._reserved[PROBE_LAYER_INPUT_ACTIVATION_COUNT] = g_topology[i].input_size;
        off += (uint64_t)g_topology[i].input_size * sizeof(float);
    }

    /* 全局状态区始终位于 buffer 尾部。 */
    g_probe_state_offset = off;
    off += sizeof(ProbeGlobalState);

    return off;
}

/* 将 NeuralNet 的权重复制到 snapshot buffer */
static void copy_net_to_buffer(uint8_t* buf, const NeuralNet* net, const ProbeGlobalState* state) {
    uint64_t off = 0;

    /* Embedding: token_embed[3][16] + pos_embed[25][16] */
    memcpy(buf + off, net->token_embed, sizeof(net->token_embed));
    off += sizeof(net->token_embed);
    memcpy(buf + off, net->pos_embed, sizeof(net->pos_embed));
    off += sizeof(net->pos_embed);

    /* Attention: wq, bq, wk, bk, wv, bv, wo, bo */
    memcpy(buf + off, net->attn_wq, sizeof(net->attn_wq)); off += sizeof(net->attn_wq);
    memcpy(buf + off, net->attn_bq, sizeof(net->attn_bq)); off += sizeof(net->attn_bq);
    memcpy(buf + off, net->attn_wk, sizeof(net->attn_wk)); off += sizeof(net->attn_wk);
    memcpy(buf + off, net->attn_bk, sizeof(net->attn_bk)); off += sizeof(net->attn_bk);
    memcpy(buf + off, net->attn_wv, sizeof(net->attn_wv)); off += sizeof(net->attn_wv);
    memcpy(buf + off, net->attn_bv, sizeof(net->attn_bv)); off += sizeof(net->attn_bv);
    memcpy(buf + off, net->attn_wo, sizeof(net->attn_wo)); off += sizeof(net->attn_wo);
    memcpy(buf + off, net->attn_bo, sizeof(net->attn_bo)); off += sizeof(net->attn_bo);

    /* Pooling（全量 128×400） */
    memcpy(buf + off, net->pool_w, sizeof(net->pool_w)); off += sizeof(net->pool_w);
    memcpy(buf + off, net->pool_b, sizeof(net->pool_b)); off += sizeof(net->pool_b);

    /* FC1 */
    memcpy(buf + off, net->w1, sizeof(net->w1)); off += sizeof(net->w1);
    memcpy(buf + off, net->b1, sizeof(net->b1)); off += sizeof(net->b1);

    /* FC2 */
    memcpy(buf + off, net->w2, sizeof(net->w2)); off += sizeof(net->w2);
    memcpy(buf + off, net->b2, sizeof(net->b2)); off += sizeof(net->b2);

    /* FC3 */
    memcpy(buf + off, net->w3, sizeof(net->w3)); off += sizeof(net->w3);
    memcpy(buf + off, net->b3, sizeof(net->b3)); off += sizeof(net->b3);

    /* FC4 */
    memcpy(buf + off, net->w4, sizeof(net->w4)); off += sizeof(net->w4);
    memcpy(buf + off, net->b4, sizeof(net->b4)); off += sizeof(net->b4);

    /* 全局状态 */
    memcpy(buf + g_probe_state_offset, state, sizeof(ProbeGlobalState));
}

static float* probe_layer_output(uint8_t* buf, uint32_t layer) {
    return (float*)(buf + g_topology[layer].activation_offset);
}

static float* probe_layer_input(uint8_t* buf, uint32_t layer) {
    return (float*)(buf + g_topology[layer]._reserved[PROBE_LAYER_INPUT_ACTIVATION_OFFSET]);
}

/* Re-run one frozen network forward pass for observability only.  This code
   never writes NeuralNet and never feeds values back into training. */
static void write_forward_diagnostics(uint8_t* buf, const NeuralNet* net,
    ProbeGlobalState* state, int input_valid) {
    const uint64_t diag_begin = g_topology[0].activation_offset;
    memset(buf + diag_begin, 0, (size_t)(g_probe_state_offset - diag_begin));
    if (!input_valid) return;

    state->diagnostics_flags |= PROBE_DIAG_FORWARD_VALID;
    const float* input = state->last_input;
    float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
    float token_mean[TOKEN_EMBED_DIM] = {};
    float* token_usage = probe_layer_output(buf, 0);
    float* position_energy = probe_layer_output(buf, 1);
    for (int i = 0; i < CELL_COUNT; ++i) {
        int token = network_token_index(input, i);
        if (token < 0) token = 0;
        if (token > 2) token = 2;
        token_usage[token] += 1.0f / CELL_COUNT;
        double energy = 0.0;
        for (int d = 0; d < TOKEN_EMBED_DIM; ++d) {
            tokens[i][d] = net->token_embed[token][d] + net->pos_embed[i][d];
            token_mean[d] += tokens[i][d] / CELL_COUNT;
            energy += (double)tokens[i][d] * tokens[i][d];
        }
        position_energy[i] = (float)sqrt(energy / TOKEN_EMBED_DIM);
    }
    memcpy(probe_layer_input(buf, 0), token_mean, sizeof(token_mean));
    memcpy(probe_layer_input(buf, 1), token_mean, sizeof(token_mean));

    float q[CELL_COUNT][ATTENTION_EMBED];
    float k[CELL_COUNT][ATTENTION_EMBED];
    float v[CELL_COUNT][ATTENTION_EMBED];
    float* q_mean = probe_layer_output(buf, 2);
    float* k_mean = probe_layer_output(buf, 3);
    float* v_mean = probe_layer_output(buf, 4);
    for (int i = 0; i < CELL_COUNT; ++i) {
        for (int d = 0; d < ATTENTION_EMBED; ++d) {
            q[i][d] = net->attn_bq[d];
            k[i][d] = net->attn_bk[d];
            v[i][d] = net->attn_bv[d];
            for (int j = 0; j < TOKEN_EMBED_DIM; ++j) {
                q[i][d] += net->attn_wq[d][j] * tokens[i][j];
                k[i][d] += net->attn_wk[d][j] * tokens[i][j];
                v[i][d] += net->attn_wv[d][j] * tokens[i][j];
            }
            q_mean[d] += q[i][d] / CELL_COUNT;
            k_mean[d] += k[i][d] / CELL_COUNT;
            v_mean[d] += v[i][d] / CELL_COUNT;
        }
    }
    for (int li = 2; li <= 4; ++li)
        memcpy(probe_layer_input(buf, li), token_mean, sizeof(token_mean));

    float concat[CELL_COUNT][ATTENTION_EMBED] = {};
    for (int h = 0; h < ATTENTION_HEADS; ++h) {
        const int start = h * ATTENTION_HEAD_DIM;
        for (int i = 0; i < CELL_COUNT; ++i) {
            float max_score = -1e30f;
            for (int j = 0; j < CELL_COUNT; ++j) {
                float score = 0.0f;
                for (int d = 0; d < ATTENTION_HEAD_DIM; ++d)
                    score += q[i][start + d] * k[j][start + d];
                score /= sqrtf((float)ATTENTION_HEAD_DIM);
                if (score > 20.0f) score = 20.0f;
                if (score < -20.0f) score = -20.0f;
                state->attention_scores[h][i][j] = score;
                if (score > max_score) max_score = score;
            }
            float sum = 0.0f;
            for (int j = 0; j < CELL_COUNT; ++j) {
                float p = expf(state->attention_scores[h][i][j] - max_score);
                state->attention_scores[h][i][j] = p;
                sum += p;
            }
            if (!(sum > 0.0f) || !isfinite(sum)) sum = (float)CELL_COUNT;
            for (int j = 0; j < CELL_COUNT; ++j)
                state->attention_scores[h][i][j] /= sum;
        }
        for (int i = 0; i < CELL_COUNT; ++i) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; ++d) {
                float value = 0.0f;
                for (int j = 0; j < CELL_COUNT; ++j)
                    value += state->attention_scores[h][i][j] * v[j][start + d];
                concat[i][start + d] = value;
            }
        }
    }

    float concat_mean[ATTENTION_EMBED] = {};
    float attn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    float* attn_mean = probe_layer_output(buf, 5);
    for (int i = 0; i < CELL_COUNT; ++i) {
        for (int d = 0; d < ATTENTION_EMBED; ++d)
            concat_mean[d] += concat[i][d] / CELL_COUNT;
        for (int d = 0; d < TOKEN_EMBED_DIM; ++d) {
            float value = net->attn_bo[d] + tokens[i][d];
            for (int j = 0; j < ATTENTION_EMBED; ++j)
                value += net->attn_wo[d][j] * concat[i][j];
            attn_out[i][d] = value;
            attn_mean[d] += value / CELL_COUNT;
        }
    }
    memcpy(probe_layer_input(buf, 5), concat_mean, sizeof(concat_mean));

    /* 展平 Attention_O[25][16] → flat[400]，与 CPU/GPU 前向一致 */
    float flat_pool[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int c = 0; c < CELL_COUNT; ++c)
        for (int d = 0; d < TOKEN_EMBED_DIM; ++d)
            flat_pool[c * TOKEN_EMBED_DIM + d] = attn_out[c][d];
    float* pool_input = probe_layer_input(buf, 6);
    memcpy(pool_input, flat_pool, sizeof(flat_pool));
    float* pooled_feat = probe_layer_output(buf, 6);
    for (int i = 0; i < FEATURE_DIM; ++i) {
        float value = net->pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; ++j)
            value += net->pool_w[i][j] * flat_pool[j];
        pooled_feat[i] = value > 0.0f ? value : 0.0f;
    }
    memcpy(probe_layer_input(buf, 7), pooled_feat, FEATURE_DIM * sizeof(float));

    float* hidden1 = probe_layer_output(buf, 7);
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
        float value = net->b1[i];
        for (int j = 0; j < FEATURE_DIM; ++j) value += net->w1[i][j] * pooled_feat[j];
        hidden1[i] = value > 0.0f ? value : 0.0f;
    }
    memcpy(probe_layer_input(buf, 8), hidden1, HIDDEN1_SIZE * sizeof(float));

    float* hidden2 = probe_layer_output(buf, 8);
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
        float value = net->b2[i];
        for (int j = 0; j < HIDDEN1_SIZE; ++j) value += net->w2[i][j] * hidden1[j];
        hidden2[i] = value > 0.0f ? value : 0.0f;
    }
    memcpy(probe_layer_input(buf, 9), hidden2, HIDDEN2_SIZE * sizeof(float));

    float* hidden3 = probe_layer_output(buf, 9);
    for (int i = 0; i < HIDDEN3_SIZE; ++i) {
        float value = net->b3[i];
        for (int j = 0; j < HIDDEN2_SIZE; ++j) value += net->w3[i][j] * hidden2[j];
        hidden3[i] = value > 0.0f ? value : 0.0f;
    }
    memcpy(probe_layer_input(buf, 10), hidden3, HIDDEN3_SIZE * sizeof(float));

    float* output = probe_layer_output(buf, 10);
    for (int i = 0; i < CELL_COUNT; ++i) {
        float value = net->b4[i];
        for (int j = 0; j < HIDDEN3_SIZE; ++j) value += net->w4[i][j] * hidden3[j];
        output[i] = value;
    }
}

static void write_effective_gradients(uint8_t* dst, uint32_t side, uint64_t updates,
    ProbeGlobalState* state) {
    auto& baseline = g_parameter_history.at(side);
    const uint8_t* previous = baseline.bytes.empty() ? nullptr : baseline.bytes.data();
    state->gradient_span_steps = updates >= baseline.updates ? updates-baseline.updates : 0;
    const int valid = previous &&
        state->gradient_span_steps > 0 && isfinite(state->learning_rate) &&
        state->learning_rate > 0.0f;
    if (valid) state->diagnostics_flags |= PROBE_DIAG_GRADIENT_VALID;
    for (uint32_t li = 0; li < kProbeLayerCount; ++li) {
        float* gradient = (float*)(dst + g_topology[li].gradient_offset);
        const float* current = (const float*)(dst + g_topology[li].weight_offset);
        const float* old = previous ?
            (const float*)(previous + g_topology[li].weight_offset) : NULL;
        for (uint64_t i = 0; i < g_topology[li].gradient_count; ++i) {
            float value = valid ? ProbeParameterHistory::mean_effective_gradient(
                old[i],current[i],state->learning_rate,state->gradient_span_steps) : 0.0f;
            gradient[i] = isfinite(value) ? value : 0.0f;
        }
    }
    const size_t parameter_bytes=(size_t)g_topology[0].activation_offset;
    try { baseline.bytes.assign(dst,dst+parameter_bytes); baseline.updates=updates; }
    catch(...) { baseline.bytes.clear(); state->diagnostics_flags &= ~PROBE_DIAG_GRADIENT_VALID; }
}

/* ========== 公共 API ========== */

int neural_probe_init(void) {
    AcquireSRWLockExclusive(&g_probe_lock);
    if (g_probe_initialized) { ReleaseSRWLockExclusive(&g_probe_lock); return 0; }

    /* 计算拓扑和大小 */
    uint64_t topo_size = MAX_PROBE_LAYERS * sizeof(ProbeLayerDesc);
    g_topology = (ProbeLayerDesc*)calloc(1, topo_size);
    if (!g_topology) { ReleaseSRWLockExclusive(&g_probe_lock); return -1; }

    uint64_t buf_size = build_topology_and_calc_size();
    uint64_t total_size = sizeof(NeuralProbeHeader) + topo_size + buf_size * 2;

    /* 创建共享内存 */
    g_probe_mapping = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        (DWORD)(total_size >> 32), (DWORD)(total_size & 0xFFFFFFFF),
        NEURAL_PROBE_NAME);
    if (!g_probe_mapping) {
        printf("[PROBE] 创建共享内存失败: %lu\n", GetLastError());
        free(g_topology); g_topology = NULL;
        ReleaseSRWLockExclusive(&g_probe_lock);
        return -1;
    }

    g_probe_base = MapViewOfFile(g_probe_mapping, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)total_size);
    if (!g_probe_base) {
        printf("[PROBE] 映射共享内存失败: %lu\n", GetLastError());
        CloseHandle(g_probe_mapping); g_probe_mapping = NULL;
        free(g_topology); g_topology = NULL;
        ReleaseSRWLockExclusive(&g_probe_lock);
        return -1;
    }

    /* 初始化 header */
    g_hdr = (NeuralProbeHeader*)g_probe_base;
    memset(g_hdr, 0, sizeof(NeuralProbeHeader));
    g_hdr->magic = NEURAL_PROBE_MAGIC;
    g_hdr->version = NEURAL_PROBE_VERSION;
    g_hdr->total_size = total_size;
    g_hdr->layer_count = kProbeLayerCount;
    g_hdr->active_buffer = 0;
    g_hdr->topology_offset = sizeof(NeuralProbeHeader);
    g_hdr->topology_size = topo_size;
    g_hdr->buffer0_offset = sizeof(NeuralProbeHeader) + topo_size;
    g_hdr->buffer1_offset = sizeof(NeuralProbeHeader) + topo_size + buf_size;
    g_hdr->buffer_size = buf_size;
    g_hdr->probe_enabled = 1;
    g_hdr->publish_hz = PROBE_PUBLISH_HZ;
    g_hdr->_reserved[PROBE_RESERVED_PUBLISHER_PID] = GetCurrentProcessId();
    g_hdr->_reserved[PROBE_RESERVED_HEARTBEAT_MS] = GetTickCount64();

    /* 写入 topology */
    uint8_t* topo_dst = (uint8_t*)g_probe_base + g_hdr->topology_offset;
    memcpy(topo_dst, g_topology, topo_size);

    g_buf0 = (uint8_t*)g_probe_base + g_hdr->buffer0_offset;
    g_buf1 = (uint8_t*)g_probe_base + g_hdr->buffer1_offset;

    g_have_previous_publish = 0;
    g_parameter_history.reset();
    g_probe_initialized = 1;
    printf("[PROBE] 共享内存已创建 (%.1f MB, %lu 层)\n",
        total_size / (1024.0 * 1024.0), g_hdr->layer_count);
    ReleaseSRWLockExclusive(&g_probe_lock);
    return 0;
}

void neural_probe_shutdown(void) {
    AcquireSRWLockExclusive(&g_probe_lock);
    if (!g_probe_initialized) { ReleaseSRWLockExclusive(&g_probe_lock); return; }
    if (g_probe_base) { UnmapViewOfFile(g_probe_base); g_probe_base = NULL; }
    if (g_probe_mapping) { CloseHandle(g_probe_mapping); g_probe_mapping = NULL; }
    free(g_topology); g_topology = NULL;
    g_hdr = NULL; g_buf0 = NULL; g_buf1 = NULL;
    g_probe_state_offset = 0;
    g_have_previous_publish = 0;
    g_parameter_history.reset();
    g_probe_initialized = 0;
    printf("[PROBE] 共享内存已关闭\n");
    ReleaseSRWLockExclusive(&g_probe_lock);
}

void neural_probe_publish(DQNAgent* agent, uint64_t episode) {
    if (!agent) return;
    AcquireSRWLockExclusive(&g_probe_lock);
    if (!g_probe_initialized || !g_hdr || !g_hdr->probe_enabled) {
        ReleaseSRWLockExclusive(&g_probe_lock); return;
    }

    NeuralNet* frozen = NULL;
    const NeuralNet* source = &agent->q_net;
    /* 这条快照属于哪一方：0=黑 1=白，共享网络记 PROBE_LAYER_SIDE_SHARED。
       没有这个字段时，TUI 展示的权重无法归因到某一方（两方独立网络下尤其致命）。 */
    uint32_t payload_side = PROBE_LAYER_SIDE_SHARED;
    if (independent_training::indep_active(agent)) {
        payload_side = (uint32_t)independent_training::side_to_physical(
            independent_training::indep_bound(agent));
    }
    if (gpu_net_active()) {
        frozen = (NeuralNet*)malloc(sizeof(NeuralNet));
        if (!frozen) {
            ReleaseSRWLockExclusive(&g_probe_lock);
            return;
        }
        /* 四句柄驻留：按**当前声明方**取该方自己的 online/target 句柄。
           且只用纯只读的 gpu_net_snapshot_side_pair（它不写影子、不改语义）；
           取不到就拒绝发布，绝不用陈旧的 CPU 影子冒充在线快照。 */
        int rc = -1;
        if (independent_training::indep_active(agent)) {
            const independent_training::Side bound = independent_training::indep_bound(agent);
            std::unique_ptr<NeuralNet> tmp_target(new (std::nothrow) NeuralNet());
            rc = tmp_target ? gpu_net_snapshot_side_pair(agent,
                independent_training::side_to_physical(bound), frozen, tmp_target.get()) : -1;
        } else {
            rc = gpu_net_snapshot_q(frozen);
        }
        if (rc != 0) {
            free(frozen);
            ReleaseSRWLockExclusive(&g_probe_lock);
            return; /* Never publish a stale CPU shadow as a live snapshot. */
        }
        source = frozen;
    }

    const LONG64 started = InterlockedIncrement64((volatile LONG64*)&g_hdr->publish_sequence);

    /* 写入 inactive buffer */
    int write_buf = 1 - g_hdr->active_buffer;
    uint8_t* dst = (write_buf == 0) ? g_buf0 : g_buf1;
    const uint8_t* previous = (g_hdr->active_buffer == 0) ? g_buf0 : g_buf1;

    /* 构造全局状态 */
    ProbeGlobalState state;
    memset(&state, 0, sizeof(state));
    state.episode = episode;
    state.train_step = (uint64_t)agent->global_step;
    state.epsilon = agent->epsilon;
    state.loss = agent->probe_last_loss;
    state.learning_rate = g_learning_rate;
    memcpy(state.q_values, agent->probe_q_values, sizeof(state.q_values));
    state.current_player = (uint32_t)(agent->probe_current_player == 1 ? 1 : 0);
    if (agent->probe_input_valid)
        memcpy(state.last_input, agent->probe_last_input, sizeof(state.last_input));

    /* 每条层描述里标出这条快照的棋色，TUI 才能把权重归因到某一方。
       放在 _reserved 里，ProbeLayerDesc 大小不变，旧 Viewer 忽略即可。 */
    if (g_topology) {
        for (uint32_t i = 0; i < kProbeLayerCount; ++i)
            g_topology[i]._reserved[PROBE_LAYER_SIDE_OFFSET] = payload_side;
    }

    /* 诊断前向只读取冻结网络；参数复制后再由前后快照计算有效梯度。 */
    write_forward_diagnostics(dst, source, &state, agent->probe_input_valid);
    copy_net_to_buffer(dst, source, &state);
    uint64_t updates=state.train_step;
    if(independent_training::indep_active(agent)) {
        const auto& u=agent->indep->updates(independent_training::side_from_physical((int)payload_side));
        updates=u.committed_batches;
        if(u.learning_rate>0)state.learning_rate=u.learning_rate;
    }
    write_effective_gradients(dst, payload_side, updates, &state);
    memcpy(dst + g_probe_state_offset, &state, sizeof(state));
    g_have_previous_publish = 1;
    free(frozen);

    /* 更新 header */
    g_hdr->snapshot_id = (uint64_t)(started + 1) / 2;
    g_hdr->episode = state.episode;
    g_hdr->train_step = state.train_step;
    g_hdr->_reserved[PROBE_RESERVED_HEARTBEAT_MS] = GetTickCount64();

    /* 原子切换 active buffer */
    InterlockedExchange((volatile LONG*)&g_hdr->active_buffer, write_buf);
    MemoryBarrier();
    InterlockedIncrement64((volatile LONG64*)&g_hdr->publish_sequence);
    ReleaseSRWLockExclusive(&g_probe_lock);
}

int neural_probe_enabled(void) {
    AcquireSRWLockShared(&g_probe_lock);
    int enabled = 0;
    if (g_probe_initialized && g_hdr && g_hdr->probe_enabled) {
        const uint64_t viewer_tick = g_hdr->_reserved[PROBE_RESERVED_VIEWER_HEARTBEAT_MS];
        const uint64_t now = GetTickCount64();
        enabled = viewer_tick != 0 && now >= viewer_tick &&
            now - viewer_tick <= PROBE_VIEWER_TIMEOUT_MS;
    }
    ReleaseSRWLockShared(&g_probe_lock);
    return enabled;
}

void neural_probe_set_enabled(int enabled) {
    AcquireSRWLockExclusive(&g_probe_lock);
    if (g_hdr) g_hdr->probe_enabled = (uint32_t)enabled;
    ReleaseSRWLockExclusive(&g_probe_lock);
}

uint64_t neural_probe_snapshot_id(void) {
    AcquireSRWLockShared(&g_probe_lock);
    uint64_t id = g_hdr ? g_hdr->snapshot_id : 0;
    ReleaseSRWLockShared(&g_probe_lock);
    return id;
}

#ifdef __cplusplus
}
#endif

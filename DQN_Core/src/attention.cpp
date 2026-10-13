// attention.c
#include "core.h"
#include "network_input.h"

// ============================================================================
// 内部辅助函数（仅本文件使用）
// ============================================================================
static void normalize_feature(float* feat, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += feat[i] * feat[i];
    float norm = sqrtf(sum + 1e-8f);
    for (int i = 0; i < dim; i++) feat[i] /= norm;
}

// ============================================================================
// 原有函数
// ============================================================================
void forward_cpu_with_attn(const NeuralNet* net, const float input[],
    float output[], float attn_out[][TOKEN_EMBED_DIM]) {
    // ========== 1. 构建 Token 序列 ==========
    float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++) {
        int state_idx = network_token_index(input, i);
        if (state_idx < 0) state_idx = 0;
        if (state_idx > 2) state_idx = 2;
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            tokens[i][d] = net->token_embed[state_idx][d] + net->pos_embed[i][d];
        }
    }

    // ========== 2. Q, K, V 投影 ==========
    float q[CELL_COUNT][ATTENTION_EMBED];
    float k[CELL_COUNT][ATTENTION_EMBED];
    float v[CELL_COUNT][ATTENTION_EMBED];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < ATTENTION_EMBED; d++) {
            q[i][d] = net->attn_bq[d];
            k[i][d] = net->attn_bk[d];
            v[i][d] = net->attn_bv[d];
            for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
                q[i][d] += net->attn_wq[d][j] * tokens[i][j];
                k[i][d] += net->attn_wk[d][j] * tokens[i][j];
                v[i][d] += net->attn_wv[d][j] * tokens[i][j];
            }
        }
    }

    // ========== 3. 多头注意力 ==========
    float head_out[ATTENTION_HEADS][CELL_COUNT][ATTENTION_HEAD_DIM];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        float qh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float kh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float vh[CELL_COUNT][ATTENTION_HEAD_DIM];
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                qh[i][d] = q[i][start + d];
                kh[i][d] = k[i][start + d];
                vh[i][d] = v[i][start + d];
            }
        }

        float scores[CELL_COUNT][CELL_COUNT];
        for (int i = 0; i < CELL_COUNT; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < CELL_COUNT; j++) {
                float dot = 0.0f;
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    dot += qh[i][d] * kh[j][d];
                }
                scores[i][j] = dot / sqrtf((float)ATTENTION_HEAD_DIM);
                if (scores[i][j] > 20.0f) scores[i][j] = 20.0f;
                if (scores[i][j] < -20.0f) scores[i][j] = -20.0f;
                if (scores[i][j] > max_val) max_val = scores[i][j];
            }
            float sum = 0.0f;
            for (int j = 0; j < CELL_COUNT; j++) {
                scores[i][j] = expf(scores[i][j] - max_val);
                sum += scores[i][j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < CELL_COUNT; j++) scores[i][j] /= sum;
            }
            else {
                for (int j = 0; j < CELL_COUNT; j++) scores[i][j] = 1.0f / CELL_COUNT;
            }
        }

        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                head_out[h][i][d] = 0.0f;
                for (int j = 0; j < CELL_COUNT; j++) {
                    head_out[h][i][d] += scores[i][j] * vh[j][d];
                }
            }
        }
    }

    // ========== 4. 拼接 + 输出投影 + 残差 ==========
    float concat[CELL_COUNT][ATTENTION_EMBED];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                concat[i][start + d] = head_out[h][i][d];
            }
        }
    }

    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            attn_out[i][d] = net->attn_bo[d];
            for (int j = 0; j < ATTENTION_EMBED; j++) {
                attn_out[i][d] += net->attn_wo[d][j] * concat[i][j];
            }
            attn_out[i][d] += tokens[i][d];
        }
    }

    // ========== 5. 展平 Attention_O[25][16] → flat[400] ==========
    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            flat[i * TOKEN_EMBED_DIM + d] = attn_out[i][d];

    // ========== 6. 可学习池化层（全 400 维输入） ==========
    float pooled_feat[FEATURE_DIM];
    for (int i = 0; i < FEATURE_DIM; i++) {
        pooled_feat[i] = net->pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            pooled_feat[i] += net->pool_w[i][j] * flat[j];
        }
        pooled_feat[i] = (pooled_feat[i] > 0) ? pooled_feat[i] : 0.0f;
    }

    // ========== 7. 全连接层 ==========
    float hidden1[HIDDEN1_SIZE], hidden2[HIDDEN2_SIZE], hidden3[HIDDEN3_SIZE];
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = net->b1[i];
        for (int j = 0; j < FEATURE_DIM; j++) {
            sum += net->w1[i][j] * pooled_feat[j];
        }
        hidden1[i] = (sum > 0) ? sum : 0.0f;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = net->b2[i];
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += net->w2[i][j] * hidden1[j];
        }
        hidden2[i] = (sum > 0) ? sum : 0.0f;
    }
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        float sum = net->b3[i];
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            sum += net->w3[i][j] * hidden2[j];
        }
        hidden3[i] = (sum > 0) ? sum : 0.0f;
    }
    for (int i = 0; i < CELL_COUNT; i++) {
        float sum = net->b4[i];
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            sum += net->w4[i][j] * hidden3[j];
        }
        output[i] = sum;
    }
}

void extract_feature(const NeuralNet* net, const float attn_out[][TOKEN_EMBED_DIM],
    float* out, int out_dim, const GameState* g) {
    // 重新展平 + 池化（全 400 维）
    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            flat[i * TOKEN_EMBED_DIM + d] = attn_out[i][d];

    int dim = (out_dim < FEATURE_DIM) ? out_dim : FEATURE_DIM;
    for (int i = 0; i < dim; i++) {
        out[i] = net->pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            out[i] += net->pool_w[i][j] * flat[j];
        }
        out[i] = (out[i] > 0) ? out[i] : 0.0f;
    }
    for (int i = dim; i < out_dim; i++) out[i] = 0.0f;
    normalize_feature(out, out_dim);
}

// ============================================================================
// 记忆系统专用特征提取（V4.0）
// ============================================================================
void memory_extract_feature(const GameState* g, float* out_feature) {
    if (!g_current_agent) {
        memset(out_feature, 0, FEATURE_DIM * sizeof(float));
        return;
    }

    // 1. 归一化到"自己 vs 对手"视角
    GameState self_game;
    normalize_state_to_self(g, &self_game, g->turn);
    float input[CELL_COUNT + 1];
    state_to_input(&self_game, input);

    // 2. 网络特征（124维，来自池化层）
    float attn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    float qvals[CELL_COUNT];
    float net_feat[NET_FEATURE_DIM];

    forward_cpu_with_attn(&g_current_agent->q_net, input, qvals, attn_out);
    extract_feature(&g_current_agent->q_net, attn_out, net_feat, NET_FEATURE_DIM, &self_game);

    // 3. 填充网络特征到前124维
    for (int i = 0; i < NET_FEATURE_DIM; i++) {
        out_feature[i] = net_feat[i];
    }

    // 4. 元信息（后4维）
    int piece_count = 0;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (self_game.board[i] != 0) piece_count++;
    }
    out_feature[NET_FEATURE_DIM + 0] = piece_count / 25.0f;
    out_feature[NET_FEATURE_DIM + 1] = g->turn / 1.0f;
    out_feature[NET_FEATURE_DIM + 2] = piece_count / 25.0f;
    out_feature[NET_FEATURE_DIM + 3] = 0.0f;

    // 5. 整体归一化
    normalize_feature(out_feature, FEATURE_DIM);
}

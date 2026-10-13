/* test_pool400.cpp — 纯数学验证 Pooling 400 维展平正确性（无 GPU 依赖） */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define CELL_COUNT 25
#define TOKEN_EMBED_DIM 16
#define FEATURE_DIM 128

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

/* Test 1: flat[cell*16+feature] 展平顺序 */
static void test_flatten_order(void) {
    printf("\n=== Test 1: Flatten order ===\n");
    float attn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int c = 0; c < CELL_COUNT; c++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            attn_out[c][d] = (float)(c * 100 + d);

    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            flat[i * TOKEN_EMBED_DIM + d] = attn_out[i][d];

    CHECK(flat[0] == 0.0f, "flat[0]=0 (cell0,feat0)");
    CHECK(flat[15] == 15.0f, "flat[15]=15 (cell0,feat15)");
    CHECK(flat[16] == 100.0f, "flat[16]=100 (cell1,feat0)");
    CHECK(flat[399] == 2415.0f, "flat[399]=2415 (cell24,feat15)");
    CHECK((int)(flat[83]) == 503, "flat[83]=503 (cell5,feat3)");
}

/* Test 2: 单 cell/feature 隔离 */
static void test_single_cell_feature(void) {
    printf("\n=== Test 2: Single cell/feature isolation ===\n");
    /* pool_w 为对角阵，pool_b=0 */
    float pool_w[FEATURE_DIM][CELL_COUNT * TOKEN_EMBED_DIM];
    float pool_b[FEATURE_DIM];
    memset(pool_w, 0, sizeof(pool_w));
    memset(pool_b, 0, sizeof(pool_b));
    for (int i = 0; i < FEATURE_DIM && i < CELL_COUNT * TOKEN_EMBED_DIM; i++)
        pool_w[i][i] = 1.0f;

    /* flat[83] = 7.0 (cell=5, feat=3)，其余为0 */
    float flat[CELL_COUNT * TOKEN_EMBED_DIM] = {};
    flat[5 * TOKEN_EMBED_DIM + 3] = 7.0f;

    float pooled_feat[FEATURE_DIM] = {};
    for (int i = 0; i < FEATURE_DIM; i++) {
        pooled_feat[i] = pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
            pooled_feat[i] += pool_w[i][j] * flat[j];
    }

    CHECK(fabsf(pooled_feat[83] - 7.0f) < 1e-5f, "pooled_feat[83]=7.0 (cell5,feat3)");
    CHECK(fabsf(pooled_feat[0]) < 1e-6f, "pooled_feat[0]=0");
    CHECK(fabsf(pooled_feat[82]) < 1e-6f, "pooled_feat[82]=0");
    CHECK(fabsf(pooled_feat[84]) < 1e-6f, "pooled_feat[84]=0");
    CHECK(fabsf(pooled_feat[127]) < 1e-6f, "pooled_feat[127]=0");
}

/* Test 3: 不同 cell 映射到不同 flat_index */
static void test_different_cells(void) {
    printf("\n=== Test 3: Different cells → different flat indices ===\n");
    CHECK(0 * TOKEN_EMBED_DIM + 0 == 0, "cell0,feat0 → flat[0]");
    CHECK(1 * TOKEN_EMBED_DIM + 0 == 16, "cell1,feat0 → flat[16]");
    CHECK(24 * TOKEN_EMBED_DIM + 0 == 384, "cell24,feat0 → flat[384]");
    CHECK(24 * TOKEN_EMBED_DIM + 15 == 399, "cell24,feat15 → flat[399]");
    CHECK(5 * TOKEN_EMBED_DIM + 3 == 83, "cell5,feat3 → flat[83]");
}

/* Test 4: 反向传播梯度覆盖所有 400 列 */
static void test_gradient_coverage(void) {
    printf("\n=== Test 4: Gradient propagation to all 400 columns ===\n");
    float pool_w[FEATURE_DIM][CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < FEATURE_DIM; i++)
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
            pool_w[i][j] = 0.01f;

    float dL_dpooled_feat[FEATURE_DIM] = {};
    dL_dpooled_feat[0] = 1.0f;

    float dL_dflat[CELL_COUNT * TOKEN_EMBED_DIM] = {};
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
        for (int i = 0; i < FEATURE_DIM; i++)
            dL_dflat[j] += dL_dpooled_feat[i] * pool_w[i][j];

    int nonzero = 0;
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
        if (fabsf(dL_dflat[j]) > 1e-10f) nonzero++;

    CHECK(nonzero == CELL_COUNT * TOKEN_EMBED_DIM, "All 400 columns get non-zero gradient");
    CHECK(fabsf(dL_dflat[0] - 0.01f) < 1e-5f, "dL_dflat[0]=0.01");
    CHECK(fabsf(dL_dflat[16] - 0.01f) < 1e-5f, "dL_dflat[16]=0.01");
    CHECK(fabsf(dL_dflat[399] - 0.01f) < 1e-5f, "dL_dflat[399]=0.01");
}

/* Test 5: 权重更新覆盖所有 400 列 */
static void test_weight_update_coverage(void) {
    printf("\n=== Test 5: Weight update affects all 400 columns ===\n");
    float pool_w[FEATURE_DIM][CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < FEATURE_DIM; i++)
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
            pool_w[i][j] = 1.0f;

    float lr = 0.001f;
    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) flat[j] = 1.0f;
    float grad = 1.0f;

    for (int i = 0; i < FEATURE_DIM; i++)
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
            pool_w[i][j] -= lr * grad * flat[j];

    CHECK(fabsf(pool_w[0][0] - (1.0f - lr)) < 1e-6f, "pool_w[0][0] updated");
    CHECK(fabsf(pool_w[0][16] - (1.0f - lr)) < 1e-6f, "pool_w[0][16] updated");
    CHECK(fabsf(pool_w[0][399] - (1.0f - lr)) < 1e-6f, "pool_w[0][399] updated");
    CHECK(fabsf(pool_w[127][399] - (1.0f - lr)) < 1e-6f, "pool_w[127][399] updated");
}

/* Test 6: Probe index mapping (cell_index, feature_index) */
static void test_probe_index_mapping(void) {
    printf("\n=== Test 6: Probe index mapping ===\n");
    for (int flat_idx = 0; flat_idx < CELL_COUNT * TOKEN_EMBED_DIM; flat_idx++) {
        int cell = flat_idx / TOKEN_EMBED_DIM;
        int feat = flat_idx % TOKEN_EMBED_DIM;
        int reconstructed = cell * TOKEN_EMBED_DIM + feat;
        if (reconstructed != flat_idx) {
            printf("FAIL: flat[%d] → cell=%d feat=%d → reconstructed=%d\n",
                flat_idx, cell, feat, reconstructed);
            g_fail++;
            return;
        }
    }
    CHECK(1, "All 400 flat indices correctly decompose to (cell, feature)");
}

/* Test 7: 反向传播梯度传播回 attn_out（无除法） */
static void test_backward_no_division(void) {
    printf("\n=== Test 7: Backward pass no division ===\n");
    /* 展平层反向：dL_dflat[j] → dL_dattn_out[cell][d] = dL_dflat[cell*16+d] */
    /* 均值池化反向：dL_dattn_out[i][d] = dL_dpooled[d] / CELL_COUNT */
    /* 展平层不应有 /CELL_COUNT */
    float dL_dflat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
        dL_dflat[j] = (float)(j + 1);  /* 非零梯度 */

    float dL_dattn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int c = 0; c < CELL_COUNT; c++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            dL_dattn_out[c][d] = dL_dflat[c * TOKEN_EMBED_DIM + d];

    /* 验证：dL_dattn_out[0][0] = dL_dflat[0] = 1.0（无除法） */
    CHECK(fabsf(dL_dattn_out[0][0] - 1.0f) < 1e-5f,
          "dL_dattn_out[0][0] = dL_dflat[0] (no /CELL_COUNT)");
    CHECK(fabsf(dL_dattn_out[24][15] - 400.0f) < 1e-5f,
          "dL_dattn_out[24][15] = dL_dflat[399] = 400 (no /CELL_COUNT)");
}

/* Test 8: 梯度能传播回多个 attn_out token */
static void test_gradient_to_multiple_tokens(void) {
    printf("\n=== Test 8: Gradient propagates to multiple tokens ===\n");
    float pool_w[FEATURE_DIM][CELL_COUNT * TOKEN_EMBED_DIM];
    memset(pool_w, 0, sizeof(pool_w));
    /* pool_w[0][0] = 1, pool_w[0][16] = 1, pool_w[0][384] = 1 */
    pool_w[0][0] = 1.0f;    /* cell 0, feat 0 */
    pool_w[0][16] = 1.0f;   /* cell 1, feat 0 */
    pool_w[0][384] = 1.0f;  /* cell 24, feat 0 */

    float dL_dpooled_feat[FEATURE_DIM] = {};
    dL_dpooled_feat[0] = 1.0f;

    float dL_dflat[CELL_COUNT * TOKEN_EMBED_DIM] = {};
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++)
        for (int i = 0; i < FEATURE_DIM; i++)
            dL_dflat[j] += dL_dpooled_feat[i] * pool_w[i][j];

    /* 验证：梯度传播回 cell 0, cell 1, cell 24 */
    CHECK(fabsf(dL_dflat[0] - 1.0f) < 1e-5f, "dL_dflat[0]=1 (cell0,feat0)");
    CHECK(fabsf(dL_dflat[16] - 1.0f) < 1e-5f, "dL_dflat[16]=1 (cell1,feat0)");
    CHECK(fabsf(dL_dflat[384] - 1.0f) < 1e-5f, "dL_dflat[384]=1 (cell24,feat0)");
    CHECK(fabsf(dL_dflat[1]) < 1e-6f, "dL_dflat[1]=0 (cell0,feat1, not connected)");
}

int main(void) {
    printf("=== Pooling 400-dim Validation Tests ===\n");
    test_flatten_order();
    test_single_cell_feature();
    test_different_cells();
    test_gradient_coverage();
    test_weight_update_coverage();
    test_probe_index_mapping();
    test_backward_no_division();
    test_gradient_to_multiple_tokens();

    printf("\n=== Summary: %d failures ===\n", g_fail);
    return g_fail;
}

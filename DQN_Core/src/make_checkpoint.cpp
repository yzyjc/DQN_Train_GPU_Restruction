/* make_checkpoint.cpp — 生成 phase=1 episode=1 的空检查点，跳过预训练 */
#include "core.h"
#include "skill_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(void) {
    const char* dir = "logs\\";
    const char* name = "checkpoint_phase1_00001.bin";
    char path[512];
    snprintf(path, sizeof(path), "%s%s", dir, name);

    FILE* f = fopen(path, "wb");
    if (!f) { printf("Cannot create %s\n", path); return 1; }

    /* CheckpointMeta */
    CheckpointMeta* meta = (CheckpointMeta*)calloc(1, sizeof(CheckpointMeta));
    if (!meta) { fclose(f); return 1; }
    meta->version = 10;
    meta->magic = 0x5A4D4E54;
    meta->phase = 1;
    meta->episode = 1;
    meta->epsilon = 1.0f;
    meta->minimax_depth = 2;
    meta->replay_black_idx = 0;
    meta->replay_white_idx = 0;
    meta->replay_black_count = 0;
    meta->replay_white_count = 0;
    meta->wins[0] = meta->wins[1] = meta->wins[2] = 0;
    meta->elapsed_seconds = 0;
    meta->concentration_factor = 0.5f;
    meta->prev_performance = 0;
    meta->global_step = 0;
    meta->ema_winrate = 0;
    meta->depth_adjust_counter = 0;
    meta->opponent_pool_count = 0;
    meta->opponent_pool_next = 0;
    meta->current_shaping_scale = 1.0f;
    /* 初始化神经网络 */
    init_net(&meta->q_net);
    memcpy(&meta->target_net, &meta->q_net, sizeof(NeuralNet));
    memcpy(&meta->opponent_net, &meta->q_net, sizeof(NeuralNet));
    /* reward_weights */
    meta->reward_weights[0] = 1.0f;
    /* RNG */
    meta->train_rng_state = 42;
    meta->eval_rng_state = 137;

    fwrite(meta, sizeof(CheckpointMeta), 1, f);
    free(meta);

    /* CheckpointTrainingState */
    CheckpointTrainingState ts = {};
    ts.input_encoding = 2;
    ts.learning_rate = 0.001f;
    ts.consolidation_target = 0;
    ts.stable_passes = 0;
    skill_tracker_init(&ts.skills);
    fwrite(&ts, sizeof(ts), 1, f);

    /* Replay buffers (empty) */
    Experience empty = {};
    for (int i = 0; i < REPLAY_CAPACITY; i++) fwrite(&empty, sizeof(empty), 1, f);
    for (int i = 0; i < REPLAY_CAPACITY; i++) fwrite(&empty, sizeof(empty), 1, f);

    /* Opponent pool (empty) */
    int pool_meta[3] = {0, 0, 0};
    fwrite(pool_meta, sizeof(pool_meta), 1, f);

    /* CRC32 placeholder */
    uint32_t crc = 0;
    fwrite(&crc, sizeof(crc), 1, f);
    fwrite(&crc, sizeof(crc), 1, f);

    fclose(f);
    printf("Created %s (phase=1, episode=1, version=10)\n", path);
    return 0;
}

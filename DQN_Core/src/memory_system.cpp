// memory_system.cpp - V5.2 Patch-based Memory System
#include "core.h"
#include <limits.h>

// ============================================================================
// Global data
// ============================================================================
static CacheBuffer g_buffers[2];
static int g_active_idx = 0;

CacheStoreEntry g_memory_cache[CACHE_CAPACITY];
int g_cache_count = 0;

KBEntry g_knowledge_base[KB_CAPACITY];
int g_kb_count = 0;

static int g_last_promote_episode = 0;
/* Package reloads can occur while the TUI skill worker is reading memory.
   Action-score queries take the shared side; a complete package replacement
   takes the exclusive side, so one evaluation never observes half a bank. */
static SRWLOCK g_model_memory_lock = SRWLOCK_INIT;

// ============================================================================
// Patch extraction: extract 9 patches (3x3 windows) from a board
// Patch layout for 5x5 board:
//   [0][1][2]
//   [3][4][5]
//   [6][7][8]
// ============================================================================
void extract_patches(const int board[CELL_COUNT], float patches[PATCH_COUNT][PATCH_SIZE]) {
    for (int pr = 0; pr <= BOARD_SIZE - PATCH_WINDOW; pr++) {
        for (int pc = 0; pc <= BOARD_SIZE - PATCH_WINDOW; pc++) {
            int patch_idx = pr * (BOARD_SIZE - PATCH_WINDOW + 1) + pc;
            for (int r = 0; r < PATCH_WINDOW; r++) {
                for (int c = 0; c < PATCH_WINDOW; c++) {
                    int board_idx = (pr + r) * BOARD_SIZE + (pc + c);
                    patches[patch_idx][r * PATCH_WINDOW + c] = (float)board[board_idx];
                }
            }
        }
    }
}

static void extract_patches_float(const float board[CELL_COUNT], float patches[PATCH_COUNT][PATCH_SIZE]) {
    for (int pr = 0; pr <= BOARD_SIZE - PATCH_WINDOW; pr++) {
        for (int pc = 0; pc <= BOARD_SIZE - PATCH_WINDOW; pc++) {
            int patch_idx = pr * (BOARD_SIZE - PATCH_WINDOW + 1) + pc;
            for (int r = 0; r < PATCH_WINDOW; r++) {
                for (int c = 0; c < PATCH_WINDOW; c++) {
                    int board_idx = (pr + r) * BOARD_SIZE + (pc + c);
                    patches[patch_idx][r * PATCH_WINDOW + c] = board[board_idx];
                }
            }
        }
    }
}

// ============================================================================
// Patch similarity: compare two patches (same position)
// Returns 0~1, higher = more similar
// ============================================================================
float patch_similarity(const float p1[PATCH_SIZE], const float p2[PATCH_SIZE]) {
    int match = 0;
    int total = 0;
    for (int i = 0; i < PATCH_SIZE; i++) {
        int s1 = (p1[i] > 0.3f) ? 1 : (p1[i] < -0.3f) ? -1 : 0;
        int s2 = (p2[i] > 0.3f) ? 1 : (p2[i] < -0.3f) ? -1 : 0;
        if (s1 == 0 && s2 == 0) continue;  // Both empty, skip
        total++;
        if (s1 == s2) match++;
    }
    if (total == 0) return 1.0f;  // Both patches are empty
    return (float)match / total;
}

// ============================================================================
// Board-level patch score: compare all 9 patches, count matches
// ============================================================================
static float board_patch_score(const float query_patches[PATCH_COUNT][PATCH_SIZE],
    const float entry_patches[PATCH_COUNT][PATCH_SIZE],
    const float patch_attention[PATCH_COUNT],
    int* out_matching_patches) {
    int matching = 0;
    float weighted_score = 0.0f;
    float total_weight = 0.0f;

    for (int p = 0; p < PATCH_COUNT; p++) {
        float sim = patch_similarity(query_patches[p], entry_patches[p]);
        float att = (patch_attention != NULL) ? patch_attention[p] : 1.0f;

        weighted_score += sim * att;
        total_weight += att;

        if (sim >= PATCH_SIM_THRESHOLD) {
            matching++;
        }
    }

    if (out_matching_patches) *out_matching_patches = matching;
    if (total_weight < 0.001f) return 0.0f;
    return weighted_score / total_weight;
}

// ============================================================================
// Full board match score (used for query)
// ============================================================================
float query_match_score(const int probe_board[CELL_COUNT],
    const float proto_board[CELL_COUNT], const float attention[CELL_COUNT]) {
    float query_patches[PATCH_COUNT][PATCH_SIZE];
    float entry_patches[PATCH_COUNT][PATCH_SIZE];

    extract_patches(probe_board, query_patches);
    extract_patches_float(proto_board, entry_patches);

    // Convert attention to patch_attention (average of positions in each patch)
    float patch_att[PATCH_COUNT];
    if (attention != NULL) {
        for (int p = 0; p < PATCH_COUNT; p++) {
            int pr = p / (BOARD_SIZE - PATCH_WINDOW + 1);
            int pc = p % (BOARD_SIZE - PATCH_WINDOW + 1);
            float sum = 0.0f;
            for (int r = 0; r < PATCH_WINDOW; r++) {
                for (int c = 0; c < PATCH_WINDOW; c++) {
                    sum += attention[(pr + r) * BOARD_SIZE + (pc + c)];
                }
            }
            patch_att[p] = sum / PATCH_SIZE;
        }
    }

    int matching_patches;
    return board_patch_score(query_patches, entry_patches,
        (attention != NULL) ? patch_att : NULL, &matching_patches);
}

// ============================================================================
// Count pieces from int board
// ============================================================================
static int count_pieces_int(const int board[CELL_COUNT]) {
    int count = 0;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (board[i] != 0) count++;
    }
    return count;
}

// ============================================================================
// Fast pre-filter: piece count from float board
// ============================================================================
static int count_pieces_float(const float board[CELL_COUNT]) {
    int count = 0;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (fabsf(board[i]) > 0.3f) count++;
    }
    return count;
}

// ============================================================================
// Find best match in cache (with piece count pre-filter + early exit)
// ============================================================================
static int find_in_cache(const int board[CELL_COUNT], int perspective, float* out_score) {
    float best_score = 0.3f;  // Match threshold
    int best_idx = -1;
    int board_pieces = count_pieces_int(board);

    float query_patches[PATCH_COUNT][PATCH_SIZE];
    extract_patches(board, query_patches);

    for (int i = 0; i < g_cache_count; i++) {
        /* 独立黑白记忆库：只在同视角条目内合并，防止白方样本被并入黑方条目 */
        if (g_memory_cache[i].perspective != perspective) continue;
        // Pre-filter: piece count difference > 4
        const CacheStoreEntry& entry = g_memory_cache[i];
        if (entry.piece_count_count > 0) {
            float entry_pieces = (float)entry.piece_count_sum / entry.piece_count_count;
            if (fabsf(entry_pieces - board_pieces) > 4.0f) continue;
        }

        float score = board_patch_score(query_patches, g_memory_cache[i].patches,
            g_memory_cache[i].patch_attention, NULL);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
            if (best_score > 0.7f) break;  // Good enough
        }
    }

    if (out_score) *out_score = best_score;
    return best_idx;
}

// ============================================================================
// Double buffer
// ============================================================================
void memory_swap_buffers(void) {
    int next = 1 - g_active_idx;
    memset(&g_buffers[next], 0, sizeof(CacheBuffer));
    g_active_idx = next;
}

CacheBuffer* memory_get_current_buffer(void) {
    return &g_buffers[g_active_idx];
}

void memory_process_prev_buffer(int episode, int is_win) {
    int prev = 1 - g_active_idx;
    CacheBuffer* prev_buf = &g_buffers[prev];
    if (!prev_buf->is_ready || prev_buf->count == 0) return;
    prev_buf->is_ready = 0;
    prev_buf->count = 0;
}

// ============================================================================
// Cache update with patch-based matching
// ============================================================================
void memory_cache_update(const int board[CELL_COUNT], int is_win, int episode, int step,
    int piece_count, int perspective) {
    if ((perspective != 0 && perspective != 1) || is_win < -1 || is_win > 1) return;
    // Filter out trivial positions
    if (piece_count < 5 || piece_count > 20) return;  // Skip too early or too late

    int phase_bucket = 0;
    if (piece_count > 10) phase_bucket = 1;
    if (piece_count > 18) phase_bucket = 2;

    float match_score = 0.0f;
    int idx = find_in_cache(board, perspective, &match_score);

    if (idx >= 0) {
        CacheStoreEntry* c = &g_memory_cache[idx];

        // Phase consistency check
        if (abs(c->phase_bucket - phase_bucket) > 1) {
            idx = -1;
        }
    }

    if (idx >= 0) {
        // Merge into existing entry
        CacheStoreEntry* c = &g_memory_cache[idx];

        float new_patches[PATCH_COUNT][PATCH_SIZE];
        extract_patches(board, new_patches);

        float alpha = 1.0f / (c->sample_count + 1);

        // EMA update for each patch + per-patch attention
        for (int p = 0; p < PATCH_COUNT; p++) {
            for (int v = 0; v < PATCH_SIZE; v++) {
                c->patches[p][v] = (1.0f - alpha) * c->patches[p][v] + alpha * new_patches[p][v];
            }
            // Per-patch attention: based on this patch's similarity
            float patch_sim = patch_similarity(new_patches[p], c->patches[p]);
            c->patch_attention[p] += patch_sim;
        }

        c->sample_count++;
        c->total++;
        if (is_win == 1) c->wins++; else if (is_win == 0) c->losses++;
        c->last_updated = episode;
        c->step_sum += step;
        c->step_count++;
        c->piece_count_sum += piece_count;
        c->piece_count_count++;
    }
    else {
        // Create new entry
        int target_idx;
        if (g_cache_count < CACHE_CAPACITY && memory_cache_count_perspective(perspective) < CACHE_CAPACITY_PER_SIDE) {
            target_idx = g_cache_count++;
        }
        else {
            // Evict least valuable entry
            target_idx = -1;
            int min_total = INT_MAX;
            for (int i = 0; i < g_cache_count; i++) {
                if (g_memory_cache[i].perspective == perspective && g_memory_cache[i].total < min_total) {
                    min_total = g_memory_cache[i].total;
                    target_idx = i;
                }
            }
        }

        if (target_idx < 0) return;
        CacheStoreEntry* c = &g_memory_cache[target_idx];
        memset(c, 0, sizeof(CacheStoreEntry));

        // Initialize patches
        extract_patches(board, c->patches);
        for (int p = 0; p < PATCH_COUNT; p++) {
            c->patch_attention[p] = 1.0f;
        }

        c->sample_count = 1;
        c->total = 1;
        c->wins = is_win == 1 ? 1 : 0;
        c->losses = is_win == 0 ? 1 : 0;
        c->last_updated = episode;
        c->step_sum = step;
        c->step_count = 1;
        c->piece_count_sum = piece_count;
        c->piece_count_count = 1;
        c->perspective = perspective;
        c->phase_bucket = phase_bucket;
    }
}

// ============================================================================
// Promote to KB
// ============================================================================
void memory_promote_to_kb(int current_episode) {
    if (current_episode - g_last_promote_episode < PROMOTE_INTERVAL) return;
    g_last_promote_episode = current_episode;

    int side_kb[2] = { memory_kb_count_perspective(0), memory_kb_count_perspective(1) };
    for (int i = g_cache_count - 1; i >= 0; i--) {
        CacheStoreEntry* c = &g_memory_cache[i];
        if (c->total < PROMOTE_MIN_TOTAL) continue;
        float capacity_ratio = (float)side_kb[c->perspective] / KB_CAPACITY_PER_SIDE;
        float dynamic_win_rate = PROMOTE_WIN_RATE + fmaxf(0.0f, capacity_ratio - 0.7f) * 0.3f;

        // Filter out trivial positions
        float avg_piece = (c->piece_count_count > 0) ? (float)c->piece_count_sum / c->piece_count_count : 0.0f;
        if (avg_piece < 5.0f || avg_piece > 20.0f) continue;  // Skip too early or too late

        int draws = c->total - c->wins - c->losses;
        float win_rate = (c->wins + 0.5f * draws) / c->total;
        float lose_rate = (float)c->losses / c->total;

        if (win_rate < dynamic_win_rate && lose_rate < LOSE_RATE_THRESHOLD) continue;
        float bias = fabsf(win_rate - 0.5f);
        if (bias < 0.10f) continue;

        float avg_step = (c->step_count > 0) ? (float)c->step_sum / c->step_count : 15.0f;
        float urgency = 0.3f;
        if (c->phase_bucket == 2 && avg_step < 10) urgency = 0.9f;
        else if (c->phase_bucket == 2) urgency = 0.7f;
        else if (avg_piece > 15) urgency = 0.5f;

        // Find existing similar KB entry
        int exist_idx = -1;
        float best_sim = 0.7f;  // Higher threshold for KB merging
        for (int j = 0; j < g_kb_count; j++) {
            KBEntry* kb = &g_knowledge_base[j];
            if (kb->perspective != c->perspective) continue;
            if (abs(kb->phase_bucket - c->phase_bucket) > 1) continue;

            float score = board_patch_score(c->patches, kb->patches, kb->patch_attention, NULL);
            if (score > best_sim) {
                best_sim = score;
                exist_idx = j;
            }
        }

        if (exist_idx >= 0) {
            // Merge into existing KB entry
            KBEntry* kb = &g_knowledge_base[exist_idx];
            float alpha = 1.0f / (kb->sample_count + 1);
            for (int p = 0; p < PATCH_COUNT; p++) {
                for (int v = 0; v < PATCH_SIZE; v++) {
                    kb->patches[p][v] = (1.0f - alpha) * kb->patches[p][v] + alpha * c->patches[p][v];
                }
                kb->patch_attention[p] += c->patch_attention[p] / c->sample_count;
            }
            kb->sample_count++;
            kb->total += c->total;
            kb->wins += c->wins;
            kb->draws += draws;
            kb->win_rate = (float)kb->wins / kb->total;
            float score = (kb->wins + 0.5f * kb->draws) / kb->total;
            kb->is_positive = (score >= 0.55f) ? 1 : -1;
            kb->danger_score = (1.0f - score) * 5.0f;
            kb->urgency_score = (kb->urgency_score + urgency) / 2.0f;
            if (c->step_count > 0) {
                kb->avg_step = (kb->avg_step * (kb->total - c->total) + avg_step * c->total) / kb->total;
            }
            if (c->piece_count_count > 0) {
                kb->avg_piece_count = (kb->avg_piece_count * (kb->total - c->total) + avg_piece * c->total) / kb->total;
            }
            if (i < g_cache_count - 1) {
                memcpy(&g_memory_cache[i], &g_memory_cache[g_cache_count - 1], sizeof(CacheStoreEntry));
            }
            g_cache_count--;
            continue;
        }

        // Create new KB entry
        if (g_kb_count < KB_CAPACITY && side_kb[c->perspective] < KB_CAPACITY_PER_SIDE) {
            side_kb[c->perspective]++;
            KBEntry* kb = &g_knowledge_base[g_kb_count++];
            memset(kb, 0, sizeof(KBEntry));
            memcpy(kb->patches, c->patches, sizeof(c->patches));
            memcpy(kb->patch_attention, c->patch_attention, sizeof(c->patch_attention));
            kb->sample_count = c->sample_count;
            kb->total = c->total;
            kb->wins = c->wins;
            kb->draws = draws;
            kb->win_rate = (float)c->wins / c->total;
            kb->is_positive = (win_rate >= 0.5f) ? 1 : -1;
            kb->danger_score = (1.0f - win_rate) * 5.0f;
            kb->urgency_score = urgency;
            kb->last_queried = current_episode;
            kb->query_count = 0;
            kb->appearance_since_promotion = 0;
            kb->demotion_count = c->demotion_count;
            kb->avg_step = avg_step;
            kb->avg_piece_count = avg_piece;
            kb->perspective = c->perspective;
            kb->phase_bucket = c->phase_bucket;

            if (kb->demotion_count >= PERMANENT_DEMOTION || (kb->demotion_count >= 2 && kb->total >= PERMANENT_TOTAL)) {
                kb->is_permanent = 1;
                kb->timeout_threshold = 0;
            }
            else if (kb->demotion_count >= 1 || kb->total >= 200) {
                kb->timeout_threshold = EXTENDED_TIMEOUT;
            }
            else {
                kb->is_permanent = 0;
                kb->timeout_threshold = DEFAULT_TIMEOUT;
            }

            if (i < g_cache_count - 1) {
                memcpy(&g_memory_cache[i], &g_memory_cache[g_cache_count - 1], sizeof(CacheStoreEntry));
            }
            g_cache_count--;
        }
    }
}

// ============================================================================
// KB query
// ============================================================================
static int memory_query_impl(const int board[CELL_COUNT], float* out_win_rate, float* out_danger,
    int current_episode, int current_piece_count, int perspective, float* confidence, bool touch) {
    if (confidence) *confidence = 0.0f;
    int phase_bucket = 0;
    if (current_piece_count > 10) phase_bucket = 1;
    if (current_piece_count > 18) phase_bucket = 2;

    int best_idx = -1;
    float best_score = 0.4f;

    float query_patches[PATCH_COUNT][PATCH_SIZE];
    extract_patches(board, query_patches);

    for (int i = 0; i < g_kb_count; i++) {
        KBEntry* kb = &g_knowledge_base[i];
        if (kb->perspective != perspective) continue;

        int phase_diff = abs(kb->phase_bucket - phase_bucket);
        float phase_penalty = 1.0f;
        if (phase_diff == 1) phase_penalty = 0.85f;
        if (phase_diff >= 2) phase_penalty = 0.50f;

        int piece_diff = abs((int)kb->avg_piece_count - current_piece_count);
        float piece_penalty = 1.0f - piece_diff * 0.03f;
        if (piece_penalty < 0.3f) piece_penalty = 0.3f;

        float sim = board_patch_score(query_patches, kb->patches, kb->patch_attention, NULL);
        float score = sim * phase_penalty * piece_penalty;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        KBEntry* kb = &g_knowledge_base[best_idx];
        if (touch) {
            kb->last_queried = current_episode;
            kb->query_count++;
        }
        if (confidence) *confidence = fminf(1.0f, best_score) *
            ((float)kb->total / ((float)kb->total + PROMOTE_MIN_TOTAL));
        *out_win_rate = (kb->wins + 0.5f * kb->draws) / kb->total;
        *out_danger = kb->danger_score;
        return best_idx;
    }

    *out_win_rate = 0.5f;
    *out_danger = 0.0f;
    return -1;
}

int memory_query(const int board[CELL_COUNT], float* win, float* danger,
    int episode, int pieces, int perspective) {
    return memory_query_impl(board, win, danger, episode, pieces, perspective, NULL, true);
}

// Score successor positions in the mover's perspective. Hypothetical queries
// must not keep every inspected memory alive. Rewards and network Q stay raw.
void memory_adjust_action_scores(const GameState* game, float scores[CELL_COUNT],
    float scale, float min_confidence) {
    if (!game || !scores || (game->turn != 0 && game->turn != 1) ||
        !isfinite(scale) || !isfinite(min_confidence) || scale <= 0 ||
        memory_kb_count_perspective(game->turn) == 0) return;
    scale = fminf(scale, 1.0f);
    min_confidence = fmaxf(0.0f, fminf(1.0f, min_confidence));
    int board[CELL_COUNT], pieces = 0;
    for (int i = 0; i < CELL_COUNT; ++i) {
        board[i] = !game->board[i] ? 0 : (game->board[i] == game->turn + 1 ? 1 : -1);
        pieces += board[i] != 0;
    }
    for (int a = 0; a < CELL_COUNT; ++a) if (!board[a] && isfinite(scores[a])) {
        board[a] = 1;
        float win, danger, confidence;
        int match = memory_query_impl(board, &win, &danger, 0, pieces + 1,
            game->turn, &confidence, false);
        board[a] = 0;
        if (match >= 0 && confidence >= min_confidence)
            scores[a] += scale * confidence * (2.0f * win - 1.0f);
    }
}

void memory_record_appearance(const int board[CELL_COUNT], int current_episode) {
    float query_patches[PATCH_COUNT][PATCH_SIZE];
    extract_patches(board, query_patches);

    for (int i = 0; i < g_kb_count; i++) {
        float score = board_patch_score(query_patches, g_knowledge_base[i].patches,
            g_knowledge_base[i].patch_attention, NULL);
        if (score > 0.5f) {
            g_knowledge_base[i].appearance_since_promotion++;
            break;
        }
    }
}

void memory_prune(int current_episode) {
    for (int i = g_kb_count - 1; i >= 0; i--) {
        KBEntry* kb = &g_knowledge_base[i];
        if (kb->is_permanent) continue;

        // Resume/rollback may move the episode clock backwards.
        if (kb->last_queried > current_episode) kb->last_queried = current_episode;
        int no_query_time = current_episode - kb->last_queried;
        if (no_query_time > kb->timeout_threshold && kb->appearance_since_promotion < 10) {
            // Keep the knowledge until its own cache can receive it. Never
            // remove a KB entry merely because demotion has nowhere to go.
            if (g_cache_count >= CACHE_CAPACITY ||
                memory_cache_count_perspective(kb->perspective) >= CACHE_CAPACITY_PER_SIDE)
                continue;
            if (g_cache_count < CACHE_CAPACITY && memory_cache_count_perspective(kb->perspective) < CACHE_CAPACITY_PER_SIDE) {
                CacheStoreEntry* c = &g_memory_cache[g_cache_count++];
                memset(c, 0, sizeof(CacheStoreEntry));
                memcpy(c->patches, kb->patches, sizeof(kb->patches));
                memcpy(c->patch_attention, kb->patch_attention, sizeof(kb->patch_attention));
                c->sample_count = kb->sample_count;
                c->total = kb->total;
                c->wins = kb->wins;
                c->losses = kb->total - kb->wins - kb->draws;
                c->last_updated = current_episode;
                c->demotion_count = kb->demotion_count + 1;
                c->step_sum = (kb->avg_step > 0 && c->total > 0) ? (int)(kb->avg_step * c->total) : 0;
                c->step_count = (c->step_sum > 0) ? c->total : 0;
                c->piece_count_sum = (int)(kb->avg_piece_count * c->total);
                c->piece_count_count = c->total;
                c->perspective = kb->perspective;
                c->phase_bucket = kb->phase_bucket;
            }
            if (i < g_kb_count - 1) {
                memcpy(kb, &g_knowledge_base[g_kb_count - 1], sizeof(KBEntry));
            }
            g_kb_count--;
        }
    }
}

// ============================================================================
// Summary and persistence
// ============================================================================
void memory_print_summary(void) {
    printf("\n========== Memory System Summary (V5.2 Patch-based) ==========\n");
    printf("Storage: 9 patches (3x3 windows) per entry\n");
    printf("Matching: Patch-based structural similarity\n");
    printf("Cache: %d entries\n", g_cache_count);
    printf("KB: %d entries\n", g_kb_count);
    printf("\n--- KB Details ---\n");
    int pos = 0, neg = 0, perm = 0;
    for (int i = 0; i < g_kb_count; i++) {
        KBEntry* kb = &g_knowledge_base[i];
        if (kb->is_positive > 0) pos++; else neg++;
        if (kb->is_permanent) perm++;
        const char* phase_name[] = { "Opening", "Midgame", "Endgame" };
        printf("  #%d: Win %.1f%%, %s, Count %d, Step %.1f, Pieces %.1f, %s, Urgency %.2f\n",
            i + 1,
            kb->win_rate * 100,
            kb->is_positive > 0 ? "Positive" : "Negative",
            kb->total,
            kb->avg_step,
            kb->avg_piece_count,
            phase_name[(kb->phase_bucket >= 0 && kb->phase_bucket < 3) ? kb->phase_bucket : 0],
            kb->urgency_score);
    }
    printf("Positive: %d, Negative: %d, Permanent: %d\n", pos, neg, perm);
    printf("==============================================================\n");
}

// ============================================================================
// Persistence (version 9 - V5.2 Patch-based)
// ============================================================================
static int memory_commit_file(FILE* f, const char* tmp_name, const char* filename, int ok) {
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

int memory_save_full(const char* filename, int episode, uint32_t generation) {
    if (g_cache_count < 0 || g_cache_count > CACHE_CAPACITY ||
        g_kb_count < 0 || g_kb_count > KB_CAPACITY) return 0;
    char tmp_name[512];
    if (snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename) < 0) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) {
        printf("[MEMORY] Cannot save to %s\n", filename);
        return 0;
    }
    uint32_t magic = 0x4D4D535F;
    uint32_t version = 11;   /* v11: 追加 generation（checkpoint CRC 同代标识） */
    int ok = fwrite(&magic, sizeof(uint32_t), 1, f) == 1 &&
        fwrite(&version, sizeof(uint32_t), 1, f) == 1 &&
        fwrite(&episode, sizeof(int), 1, f) == 1 &&
        fwrite(&generation, sizeof(uint32_t), 1, f) == 1 &&
        fwrite(&g_cache_count, sizeof(int), 1, f) == 1 &&
        fwrite(g_memory_cache, sizeof(CacheStoreEntry), g_cache_count, f) == (size_t)g_cache_count &&
        fwrite(&g_kb_count, sizeof(int), 1, f) == 1 &&
        fwrite(g_knowledge_base, sizeof(KBEntry), g_kb_count, f) == (size_t)g_kb_count;
    if (!memory_commit_file(f, tmp_name, filename, ok)) {
        printf("[MEMORY] Failed to save %s\n", filename); return 0;
    }
    printf("[MEMORY] Saved to %s (Ep %d, Cache %d, KB %d, V%d)\n",
        filename, episode, g_cache_count, g_kb_count, version);
    return 1;
}

static int valid_cache_entry(const CacheStoreEntry* c) {
    if (!c || c->sample_count <= 0 || c->total <= 0 || c->wins < 0 || c->losses < 0 ||
        (int64_t)c->wins + c->losses > c->total || c->last_updated < 0 || c->demotion_count < 0 ||
        c->step_sum < 0 || c->step_count <= 0 || c->piece_count_sum < 0 ||
        c->piece_count_count <= 0 || (c->perspective != 0 && c->perspective != 1) ||
        c->phase_bucket < 0 || c->phase_bucket > 2) return 0;
    for (int p = 0; p < PATCH_COUNT; ++p) {
        if (!isfinite(c->patch_attention[p]) || c->patch_attention[p] < 0.0f) return 0;
        for (int v = 0; v < PATCH_SIZE; ++v)
            if (!isfinite(c->patches[p][v]) || c->patches[p][v] < -1.0f ||
                c->patches[p][v] > 1.0f) return 0;
    }
    return 1;
}

static int valid_kb_entry(const KBEntry* k) {
    if (!k || k->sample_count <= 0 || k->total <= 0 || k->wins < 0 || k->wins > k->total || k->draws < 0 || (int64_t)k->wins + k->draws > k->total ||
        k->last_queried < 0 || k->query_count < 0 || k->appearance_since_promotion < 0 ||
        k->demotion_count < 0 || (k->is_permanent != 0 && k->is_permanent != 1) ||
        k->timeout_threshold < 0 || (k->is_positive != -1 && k->is_positive != 1) ||
        k->black_wins < 0 || k->white_wins < 0 ||
        (k->perspective != 0 && k->perspective != 1) || k->phase_bucket < 0 || k->phase_bucket > 2 ||
        !isfinite(k->win_rate) || k->win_rate < 0.0f || k->win_rate > 1.0f ||
        !isfinite(k->danger_score) || !isfinite(k->avg_step) || k->avg_step < 0.0f ||
        k->avg_step > CELL_COUNT || !isfinite(k->step_stddev) || k->step_stddev < 0.0f ||
        !isfinite(k->player_bias) || !isfinite(k->avg_piece_count) ||
        k->avg_piece_count < 0.0f || k->avg_piece_count > CELL_COUNT ||
        !isfinite(k->urgency_score)) return 0;
    for (int p = 0; p < PATCH_COUNT; ++p) {
        if (!isfinite(k->patch_attention[p]) || k->patch_attention[p] < 0.0f) return 0;
        for (int v = 0; v < PATCH_SIZE; ++v)
            if (!isfinite(k->patches[p][v]) || k->patches[p][v] < -1.0f ||
                k->patches[p][v] > 1.0f) return 0;
    }
    return 1;
}

static int valid_memory_arrays(const CacheStoreEntry* cache, int cache_count,
    const KBEntry* kb, int kb_count) {
    int cache_sides[2] = {}, kb_sides[2] = {};
    for (int i = 0; i < cache_count; ++i) {
        if (!valid_cache_entry(&cache[i])) return 0;
        if (++cache_sides[cache[i].perspective] > CACHE_CAPACITY_PER_SIDE) return 0;
    }
    for (int i = 0; i < kb_count; ++i) {
        if (!valid_kb_entry(&kb[i])) return 0;
        if (++kb_sides[kb[i].perspective] > KB_CAPACITY_PER_SIDE) return 0;
    }
    return 1;
}

// V9 records end immediately before draws. Validate length before conversion.
static size_t kb_disk_size(bool legacy) {
    return legacy ? sizeof(KBEntry) - sizeof(int) : sizeof(KBEntry);
}
static bool read_kb_entries(FILE* f, KBEntry* entries, int count, bool legacy) {
    for (int i = 0; i < count; ++i) {
        memset(&entries[i], 0, sizeof(KBEntry));
        if (fread(&entries[i], kb_disk_size(legacy), 1, f) != 1) return false;
    }
    return true;
}

// GPT6-Astra: recovery probes share the complete loader validation without publishing state.
static int memory_load_full_impl(const char* filename, int* out_episode, uint32_t expected_generation,
    bool require_generation, bool publish) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4D4D535F) {
        fclose(f);
        return 0;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        (version != 9 && version != 10 && version != 11)) {
        fclose(f); return 0;
    }
    int episode;
    uint32_t generation = 0;
    if (fread(&episode, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (version == 11) {
        if (fread(&generation, sizeof(uint32_t), 1, f) != 1) { fclose(f); return 0; }
    }
    /* GPT6-Astra: the checkpoint decides whether legacy memory is permitted.
       A stale V10 sidecar cannot downgrade a new checkpoint's requirements. */
    if (require_generation && version != 11) {
        fclose(f);
        printf("[MEMORY] %s 缺少新 checkpoint 必需的记忆代号（memory V%u），拒绝加载\n",
            filename, version);
        return 0;
    }
    if (version == 11 && (require_generation || expected_generation != 0) &&
        generation != expected_generation) {
        fclose(f);
        printf("[MEMORY] %s 与 checkpoint 不同代（gen %u vs %u），拒绝加载\n",
            filename, generation, expected_generation);
        return 0;
    }
    int cache_count, kb_count;
    if (fread(&cache_count, sizeof(int), 1, f) != 1 ||
        cache_count < 0 || cache_count > CACHE_CAPACITY ||
        _fseeki64(f, (int64_t)sizeof(CacheStoreEntry) * cache_count, SEEK_CUR) != 0 ||
        fread(&kb_count, sizeof(int), 1, f) != 1 ||
        kb_count < 0 || kb_count > KB_CAPACITY) {
        fclose(f); return 0;
    }
    /* 文件头：magic(4) + version(4) + episode(4) + [generation(4) v11] + cache_count(4) */
    const int64_t header_size = 4 + 4 + 4 + (version == 11 ? 4 : 0) + 4;
    const int64_t expected = header_size +
        (int64_t)sizeof(CacheStoreEntry) * cache_count + 4 +
        (int64_t)kb_disk_size(version == 9) * kb_count;
    _fseeki64(f, 0, SEEK_END);
    if (_ftelli64(f) != expected) { fclose(f); return 0; }

    const int validated_kb_count = kb_count;
    CacheStoreEntry* cache = cache_count ?
        (CacheStoreEntry*)malloc(sizeof(CacheStoreEntry) * (size_t)cache_count) : NULL;
    KBEntry* kb = validated_kb_count ?
        (KBEntry*)malloc(sizeof(KBEntry) * (size_t)validated_kb_count) : NULL;
    if ((cache_count && !cache) || (kb_count && !kb)) { free(cache); free(kb); fclose(f); return 0; }
    /* 内容也在临时区验证，损坏文件不能部分覆盖在线记忆。 */
    _fseeki64(f, header_size, SEEK_SET);
    int loaded_kb_count = -1;
    if (fread(cache, sizeof(CacheStoreEntry), cache_count, f) != (size_t)cache_count ||
        fread(&loaded_kb_count, sizeof(int), 1, f) != 1 || loaded_kb_count != validated_kb_count ||
        !read_kb_entries(f, kb, validated_kb_count, version == 9) ||
        !valid_memory_arrays(cache, cache_count, kb, validated_kb_count)) {
        free(cache); free(kb); fclose(f); return 0;
    }
    if (!publish) { free(cache); free(kb); fclose(f); return 1; }
    if (cache_count) memcpy(g_memory_cache, cache, sizeof(CacheStoreEntry) * (size_t)cache_count);
    if (validated_kb_count) memcpy(g_knowledge_base, kb, sizeof(KBEntry) * (size_t)validated_kb_count);
    free(cache); free(kb);
    g_cache_count = cache_count;
    g_kb_count = validated_kb_count;
    if (out_episode) *out_episode = episode;

    fclose(f);
    printf("[MEMORY] Loaded from %s (Ep %d, Cache %d, KB %d, V%d)\n",
        filename, episode, g_cache_count, g_kb_count, version);
    return 1;
}

/* 只读探测：文件是否为 V11 记忆快照且代次匹配 checkpoint CRC。无副作用（不载入全局）。 */
int memory_snapshot_matches_generation(const char* filename, uint32_t expected_generation) {
    return memory_load_full_impl(filename, nullptr, expected_generation, true, false);
}

int memory_load_full(const char* filename, int* out_episode, uint32_t expected_generation,
    bool require_generation) {
    return memory_load_full_impl(filename, out_episode, expected_generation, require_generation, true);
}

/* 只读读出 V11 记忆快照的代次（checkpoint CRC）。成功返回 1 并写入 *out_generation。 */
int memory_snapshot_read_generation(const char* filename, uint32_t* out_generation) {
    if (!out_generation) return 0;
    *out_generation = 0;
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4D4D535F ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 11) {
        fclose(f); return 0;
    }
    int episode;
    uint32_t generation = 0;
    if (fread(&episode, sizeof(int), 1, f) != 1 ||
        fread(&generation, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); return 0;
    }
    fclose(f);
    *out_generation = generation;
    return 1;
}

void memory_save_cache(const char* filename) {
    if (g_cache_count < 0 || g_cache_count > CACHE_CAPACITY) return;
    char tmp_name[512];
    if (snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename) < 0) return;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) return;
    int ok = fwrite(&g_cache_count, sizeof(int), 1, f) == 1 &&
        fwrite(g_memory_cache, sizeof(CacheStoreEntry), g_cache_count, f) == (size_t)g_cache_count;
    memory_commit_file(f, tmp_name, filename, ok);
}

void memory_load_cache(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1 || count < 0 || count > CACHE_CAPACITY) { fclose(f); return; }
    _fseeki64(f, 0, SEEK_END);
    if (_ftelli64(f) != 4 + (int64_t)sizeof(CacheStoreEntry) * count) { fclose(f); return; }
    CacheStoreEntry* candidate = count ?
        (CacheStoreEntry*)malloc(sizeof(CacheStoreEntry) * (size_t)count) : NULL;
    if (count && !candidate) { fclose(f); return; }
    _fseeki64(f, 4, SEEK_SET);
    if (fread(candidate, sizeof(CacheStoreEntry), count, f) == (size_t)count &&
        valid_memory_arrays(candidate, count, NULL, 0)) {
        if (count) memcpy(g_memory_cache, candidate, sizeof(CacheStoreEntry) * (size_t)count);
        g_cache_count = count;
    }
    free(candidate);
    fclose(f);
}

/* 按视角单独保存缓存（只保存指定 perspective 的条目） */
int memory_save_cache_perspective(const char* filename, int perspective) {
    int count = 0;
    for (int i = 0; i < g_cache_count; i++)
        if (g_memory_cache[i].perspective == perspective) count++;
    if (perspective != 0 && perspective != 1) return 0;
    char tmp_name[512];
    int length = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (length < 0 || length >= (int)sizeof(tmp_name)) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) return 0;
    int ok = fwrite(&count, sizeof(int), 1, f) == 1;
    for (int i = 0; i < g_cache_count && ok; i++) {
        if (g_memory_cache[i].perspective == perspective)
            ok = fwrite(&g_memory_cache[i], sizeof(CacheStoreEntry), 1, f) == 1;
    }
    return memory_commit_file(f, tmp_name, filename, ok);
}

void memory_apply_model_scores(const GameState* game, float scores[CELL_COUNT]) {
    if (!game || !scores) return;
    /* memory_enabled controls collection/maintenance.  Once a bank exists it is
       part of the deployed model and must participate in every inference path. */
    float scale = g_hot_config.memory_scale;
    float confidence = g_hot_config.memory_confidence;
    /* IND-02 修复：记忆强度 0 是合法的“关闭记忆加分”语义，必须保留。
       只对非有限值、负值等非法输入回退；原 scale<=0 会把 0 误当默认 0.5，
       导致把强度从 0.05 拉到 0 反而放大 10 倍。 */
    if (!isfinite(scale) || scale < 0.0f) scale = 0.5f;
    if (!isfinite(confidence) || confidence < 0.0f || confidence > 1.0f)
        confidence = 0.3f;
    AcquireSRWLockShared(&g_model_memory_lock);
    memory_adjust_action_scores(game, scores, scale, confidence);
    ReleaseSRWLockShared(&g_model_memory_lock);
}

/* 按视角加载缓存（追加到全局缓存）。先整体读入临时区并校验，
   确认文件完整、条目合法后才追加，防止损坏文件注入脏数据。 */
void memory_load_cache_perspective(const char* filename, int perspective) {
    if (perspective != 0 && perspective != 1) return;
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1 || count < 0 || count > CACHE_CAPACITY) { fclose(f); return; }
    _fseeki64(f, 0, SEEK_END);
    if (_ftelli64(f) != 4 + (int64_t)sizeof(CacheStoreEntry) * count) { fclose(f); return; }
    CacheStoreEntry* candidate = count ?
        (CacheStoreEntry*)malloc(sizeof(CacheStoreEntry) * (size_t)count) : NULL;
    if (count && !candidate) { fclose(f); return; }
    _fseeki64(f, 4, SEEK_SET);
    if (fread(candidate, sizeof(CacheStoreEntry), count, f) == (size_t)count &&
        valid_memory_arrays(candidate, count, NULL, 0)) {
        int room = CACHE_CAPACITY_PER_SIDE - memory_cache_count_perspective(perspective);
        int n = count < room ? count : room;
        for (int i = 0; i < n; i++) {
            candidate[i].perspective = perspective;  /* 强制设置视角 */
            g_memory_cache[g_cache_count++] = candidate[i];
        }
    }
    free(candidate);
    fclose(f);
}

void memory_save_kb(const char* filename) {
    if (g_kb_count < 0 || g_kb_count > KB_CAPACITY) return;
    char tmp_name[512];
    if (snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename) < 0) return;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) return;
    int ok = fwrite(&g_kb_count, sizeof(int), 1, f) == 1 &&
        fwrite(g_knowledge_base, sizeof(KBEntry), g_kb_count, f) == (size_t)g_kb_count;
    memory_commit_file(f, tmp_name, filename, ok);
}

void memory_load_kb(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1 || count < 0 || count > KB_CAPACITY) { fclose(f); return; }
    _fseeki64(f, 0, SEEK_END);
    int64_t bytes = _ftelli64(f);
    bool legacy = bytes == 4 + (int64_t)kb_disk_size(true) * count;
    if (!legacy && bytes != 4 + (int64_t)sizeof(KBEntry) * count) { fclose(f); return; }
    KBEntry* candidate = count ? (KBEntry*)malloc(sizeof(KBEntry) * (size_t)count) : NULL;
    if (count && !candidate) { fclose(f); return; }
    _fseeki64(f, 4, SEEK_SET);
    if (read_kb_entries(f, candidate, count, legacy) &&
        valid_memory_arrays(NULL, 0, candidate, count)) {
        if (count) memcpy(g_knowledge_base, candidate, sizeof(KBEntry) * (size_t)count);
        g_kb_count = count;
    }
    free(candidate);
    fclose(f);
}

/* 按视角单独保存 KB */
int memory_save_kb_perspective(const char* filename, int perspective) {
    int count = 0;
    for (int i = 0; i < g_kb_count; i++)
        if (g_knowledge_base[i].perspective == perspective) count++;
    if (perspective != 0 && perspective != 1) return 0;
    char tmp_name[512];
    int length = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (length < 0 || length >= (int)sizeof(tmp_name)) return 0;
    FILE* f = fopen(tmp_name, "wb");
    if (!f) return 0;
    int ok = fwrite(&count, sizeof(int), 1, f) == 1;
    for (int i = 0; i < g_kb_count && ok; i++) {
        if (g_knowledge_base[i].perspective == perspective)
            ok = fwrite(&g_knowledge_base[i], sizeof(KBEntry), 1, f) == 1;
    }
    return memory_commit_file(f, tmp_name, filename, ok);
}

/* 按视角加载 KB（追加到全局 KB）。整体读入临时区校验后追加。 */
void memory_load_kb_perspective(const char* filename, int perspective) {
    if (perspective != 0 && perspective != 1) return;
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1 || count < 0 || count > KB_CAPACITY) { fclose(f); return; }
    _fseeki64(f, 0, SEEK_END);
    int64_t bytes = _ftelli64(f);
    bool legacy = bytes == 4 + (int64_t)kb_disk_size(true) * count;
    if (!legacy && bytes != 4 + (int64_t)sizeof(KBEntry) * count) { fclose(f); return; }
    KBEntry* candidate = count ? (KBEntry*)malloc(sizeof(KBEntry) * (size_t)count) : NULL;
    if (count && !candidate) { fclose(f); return; }
    _fseeki64(f, 4, SEEK_SET);
    if (read_kb_entries(f, candidate, count, legacy) &&
        valid_memory_arrays(NULL, 0, candidate, count)) {
        int room = KB_CAPACITY_PER_SIDE - memory_kb_count_perspective(perspective);
        int n = count < room ? count : room;
        for (int i = 0; i < n; i++) {
            candidate[i].perspective = perspective;
            g_knowledge_base[g_kb_count++] = candidate[i];
        }
    }
    free(candidate);
    fclose(f);
}

int memory_cache_count_perspective(int perspective) {
    int count = 0;
    for (int i = 0; i < g_cache_count; i++)
        if (g_memory_cache[i].perspective == perspective) count++;
    return count;
}

int memory_kb_count_perspective(int perspective) {
    int count = 0;
    for (int i = 0; i < g_kb_count; i++)
        if (g_knowledge_base[i].perspective == perspective) count++;
    return count;
}

void memory_system_init(void) {
    memset(g_memory_cache, 0, sizeof(g_memory_cache));
    g_cache_count = 0;
    memset(g_knowledge_base, 0, sizeof(g_knowledge_base));
    g_kb_count = 0;
    g_last_promote_episode = 0;
    memset(g_buffers, 0, sizeof(g_buffers));
    g_active_idx = 0;
}

void memory_system_free(void) {
    // No dynamic allocation
}

int memory_load_deployed(void) {
    AcquireSRWLockExclusive(&g_model_memory_lock);
    memory_system_init();
    memory_load_cache_perspective("memory_cache_b.dat", 0);
    memory_load_cache_perspective("memory_cache_w.dat", 1);
    memory_load_kb_perspective("memory_kb_b.dat", 0);
    memory_load_kb_perspective("memory_kb_w.dat", 1);
    if (g_cache_count == 0 && g_kb_count == 0) {
        /* One-way compatibility for the last combined-memory generation. */
        memory_load_cache("memory_cache.dat");
        memory_load_kb("memory_kb.dat");
    }
    int loaded = g_cache_count + g_kb_count;
    ReleaseSRWLockExclusive(&g_model_memory_lock);
    return loaded;
}

/* 模型包头部。V1 只含前 6 个字段；V2 追加 memory_scale/memory_confidence。
   IND-06：推理配置随包保存，使 TUI 测试/对战用与 Trainer 一致的策略。 */
struct ModelMemoryPackageHeaderV1 {
    uint32_t magic;
    uint32_t version;
    int32_t episode;
    uint32_t net_bytes;
    int32_t cache_count;
    int32_t kb_count;
};
struct ModelMemoryPackageHeader {
    uint32_t magic;
    uint32_t version;
    int32_t episode;
    uint32_t net_bytes;
    int32_t cache_count;
    int32_t kb_count;
    float memory_scale;
    float memory_confidence;
};

static int finite_package_net(const NeuralNet* net) {
    if (!net || sizeof(NeuralNet) % sizeof(float) != 0) return 0;
    const float* values = (const float*)net;
    for (size_t i = 0; i < sizeof(NeuralNet) / sizeof(float); ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

// GPT6-Astra: V3 includes two networks and a CRC covering the complete package.
static uint32_t package_crc(FILE* f, int64_t bytes) {
    uint32_t crc=0xffffffffu; unsigned char buf[65536];
    while(bytes>0){size_t n=(size_t)(bytes>sizeof(buf)?sizeof(buf):bytes);
        if(fread(buf,1,n,f)!=n)return 0;
        for(size_t i=0;i<n;++i){crc^=buf[i];for(int j=0;j<8;++j)crc=(crc>>1)^(0xedb88320u & (uint32_t)-(int)(crc&1));}
        bytes-=n;
    } return crc^0xffffffffu;
}
static int save_package(const NeuralNet* net, const NeuralNet* white, const char* filename, int episode) {

    if (!filename || episode < 0 || !finite_package_net(net) || (white && !finite_package_net(white))) return 0;
    AcquireSRWLockShared(&g_model_memory_lock);
    if (g_cache_count < 0 || g_cache_count > CACHE_CAPACITY ||
        g_kb_count < 0 || g_kb_count > KB_CAPACITY ||
        !valid_memory_arrays(g_memory_cache, g_cache_count, g_knowledge_base, g_kb_count)) {
        ReleaseSRWLockShared(&g_model_memory_lock);
        return 0;
    }
    char tmp_name[512];
    int length = snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);
    if (length < 0 || length >= (int)sizeof(tmp_name)) {
        ReleaseSRWLockShared(&g_model_memory_lock);
        return 0;
    }
    FILE* f = fopen(tmp_name, "w+b");
    if (!f) { ReleaseSRWLockShared(&g_model_memory_lock); return 0; }
    /* IND-06：推理配置随包保存，TUI 测试/对战据此还原 Trainer 的实际策略。 */
    ModelMemoryPackageHeader hdr = { 0x4B504D44u, white ? 3u : 2u, episode,
        (uint32_t)(sizeof(NeuralNet)*(white?2:1)), g_cache_count, g_kb_count,
        g_hot_config.memory_scale, g_hot_config.memory_confidence };
    int ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1 &&
        fwrite(net, sizeof(NeuralNet), 1, f) == 1 &&
        (!white || fwrite(white, sizeof(NeuralNet), 1, f) == 1) &&
        fwrite(g_memory_cache, sizeof(CacheStoreEntry), g_cache_count, f) == (size_t)g_cache_count &&
        fwrite(g_knowledge_base, sizeof(KBEntry), g_kb_count, f) == (size_t)g_kb_count;
    ReleaseSRWLockShared(&g_model_memory_lock);
    if(ok && white) {
        ok=fflush(f)==0;
        const int64_t body=_ftelli64(f);
        rewind(f); uint32_t crc=package_crc(f,body);
        ok=ok && !ferror(f) && !feof(f) && _fseeki64(f,0,SEEK_END)==0 && fwrite(&crc,sizeof(crc),1,f)==1;
    }
    return memory_commit_file(f, tmp_name, filename, ok);
}

int memory_save_model_package(const NeuralNet* net,const char* filename,int episode) {
    return save_package(net,nullptr,filename,episode);
}
int memory_save_model_package_pair(const NeuralNet* black,const NeuralNet* white,const char* filename,int episode) {
    return white && save_package(black,white,filename,episode);
}
static int load_package(NeuralNet* net,NeuralNet* white,const char* filename,int* out_episode,int* out_independent) {

    if (!net || !filename) return 0;
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    ModelMemoryPackageHeaderV1 hdr1 = {};
    if (fread(&hdr1, sizeof(hdr1), 1, f) != 1 || hdr1.magic != 0x4B504D44u ||
        (hdr1.version != 1 && hdr1.version != 2 && hdr1.version != 3) || hdr1.episode < 0 ||
        hdr1.net_bytes != sizeof(NeuralNet)*(hdr1.version==3?2:1) || (hdr1.version==3 && !white) ||
        hdr1.cache_count < 0 || hdr1.cache_count > CACHE_CAPACITY ||
        hdr1.kb_count < 0 || hdr1.kb_count > KB_CAPACITY) {
        fclose(f); return 0;
    }
    /* IND-06：V2 包携带推理配置，V1 旧包用默认值。 */
    float pkg_scale = 0.5f, pkg_confidence = 0.3f;
    int64_t header_bytes = (int64_t)sizeof(hdr1);
    if (hdr1.version >= 2) {
        if (fread(&pkg_scale, sizeof(float), 1, f) != 1 ||
            fread(&pkg_confidence, sizeof(float), 1, f) != 1) {
            fclose(f); return 0;
        }
        header_bytes += 2 * (int64_t)sizeof(float);
        if (!isfinite(pkg_scale) || pkg_scale < 0.0f) pkg_scale = 0.5f;
        if (!isfinite(pkg_confidence) || pkg_confidence < 0.0f || pkg_confidence > 1.0f)
            pkg_confidence = 0.3f;
    }
    const int64_t expected = header_bytes + (int64_t)hdr1.net_bytes +
        (int64_t)hdr1.cache_count * sizeof(CacheStoreEntry) +
        (int64_t)hdr1.kb_count * sizeof(KBEntry);
    _fseeki64(f, 0, SEEK_END);
    if (_ftelli64(f) != expected+(hdr1.version==3?sizeof(uint32_t):0)) { fclose(f); return 0; }
    if(hdr1.version==3){
        uint32_t stored=0; _fseeki64(f,expected,SEEK_SET);
        if(fread(&stored,sizeof(stored),1,f)!=1){fclose(f);return 0;}
        rewind(f); if(package_crc(f,expected)!=stored || ferror(f) || feof(f)){fclose(f);return 0;}
    }
    _fseeki64(f, header_bytes, SEEK_SET);
    NeuralNet* incoming_net = (NeuralNet*)malloc(sizeof(NeuralNet)*(hdr1.version==3?2:1));
    CacheStoreEntry* incoming_cache = hdr1.cache_count ?
        (CacheStoreEntry*)malloc((size_t)hdr1.cache_count * sizeof(CacheStoreEntry)) : NULL;
    KBEntry* incoming_kb = hdr1.kb_count ?
        (KBEntry*)malloc((size_t)hdr1.kb_count * sizeof(KBEntry)) : NULL;
    int ok = incoming_net && (!hdr1.cache_count || incoming_cache) && (!hdr1.kb_count || incoming_kb) &&
        fread(incoming_net, 1, hdr1.net_bytes, f) == hdr1.net_bytes &&
        fread(incoming_cache, sizeof(CacheStoreEntry), hdr1.cache_count, f) == (size_t)hdr1.cache_count &&
        fread(incoming_kb, sizeof(KBEntry), hdr1.kb_count, f) == (size_t)hdr1.kb_count &&
        finite_package_net(incoming_net) &&
        (hdr1.version!=3 || finite_package_net(incoming_net+1)) &&
        valid_memory_arrays(incoming_cache, hdr1.cache_count, incoming_kb, hdr1.kb_count);
    fclose(f);
    if (!ok) { free(incoming_net); free(incoming_cache); free(incoming_kb); return 0; }
    AcquireSRWLockExclusive(&g_model_memory_lock);
    memcpy(net, incoming_net, sizeof(NeuralNet));
    if(white) memcpy(white,incoming_net+(hdr1.version==3?1:0),sizeof(NeuralNet));
    if (hdr1.cache_count) memcpy(g_memory_cache, incoming_cache,
        (size_t)hdr1.cache_count * sizeof(CacheStoreEntry));
    if (hdr1.kb_count) memcpy(g_knowledge_base, incoming_kb,
        (size_t)hdr1.kb_count * sizeof(KBEntry));
    if (hdr1.cache_count < g_cache_count)
        memset(g_memory_cache + hdr1.cache_count, 0,
            (size_t)(g_cache_count - hdr1.cache_count) * sizeof(CacheStoreEntry));
    if (hdr1.kb_count < g_kb_count)
        memset(g_knowledge_base + hdr1.kb_count, 0,
            (size_t)(g_kb_count - hdr1.kb_count) * sizeof(KBEntry));
    g_cache_count = hdr1.cache_count;
    g_kb_count = hdr1.kb_count;
    ReleaseSRWLockExclusive(&g_model_memory_lock);
    /* IND-06：按包内推理配置恢复记忆强度/置信，使 TUI 本地推理与快照一致。 */
    g_hot_config.memory_scale = pkg_scale;
    g_hot_config.memory_confidence = pkg_confidence;
    free(incoming_net); free(incoming_cache); free(incoming_kb);
    if (out_episode) *out_episode = hdr1.episode;
    if (out_independent) *out_independent = hdr1.version==3;
    return 1;
}

int memory_load_model_package(NeuralNet* net,const char* filename,int* out_episode) {
    return load_package(net,nullptr,filename,out_episode,nullptr);
}
int memory_load_model_package_pair(NeuralNet* black,NeuralNet* white,const char* filename,int* out_episode,int* out_independent) {
    if(!white || black==white)return 0;
    return load_package(black,white,filename,out_episode,out_independent);
}

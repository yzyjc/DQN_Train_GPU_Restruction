#pragma once
#include "core.h"
#include "experience_semantics.h"
#include <algorithm>

// Formal play changes side after EVERY move, including terminal moves. Older
// formal checkpoints can contain same-side solitaire transitions. Remove only
// that incompatible task, preserving the chronological order of legal replay.
inline int discard_solitaire_replay(Experience* buffer, int* count, int* next) {
    if (!buffer || !count || !next || *count <= 0 || *count > REPLAY_CAPACITY ||
        *next < 0 || *next >= REPLAY_CAPACITY) return 0;
    int removed = 0;
    for (int i = 0; i < *count; ++i)
        removed += buffer[i].state[CELL_COUNT] == buffer[i].next_state[CELL_COUNT];
    if (!removed) return 0;
    if (*count == REPLAY_CAPACITY)
        std::rotate(buffer, buffer + *next, buffer + *count);
    int kept = 0;
    for (int i = 0; i < *count; ++i)
        if (buffer[i].state[CELL_COUNT] != buffer[i].next_state[CELL_COUNT])
            buffer[kept++] = buffer[i];
    memset(buffer + kept, 0, sizeof(Experience) * (REPLAY_CAPACITY - kept));
    *count = *next = kept; // at least one removal => kept < capacity
    return removed;
}

/* 独立网络（两 ply 契约）专用：删除语义过时的一 ply 非终局样本。
   绝不能复用 discard_solitaire_replay —— 它的谓词是「state.turn == next.turn」，
   而两 ply 正式样本（本方落子 -> 对手回应 -> 回到本方）同样满足 turn 相等，
   用它会把**合法的新回放整批删掉**（违反 HANDOFF:58「新档恢复必须保留新回放」），
   同时又会把一 ply 正式样本（turn 不相等）一条都不删（随后命中 replay_span 闩死）。
   这里的谓词只看跨度：done == 0 且 observed_plies == 1。
   保留：两 ply 非终局、一 ply/两 ply 终局。返回删除条数。 */
inline int discard_ply1_replay(Experience* buffer, int* count, int* next) {
    if (!buffer || !count || !next || *count <= 0 || *count > REPLAY_CAPACITY ||
        *next < 0 || *next >= REPLAY_CAPACITY) return 0;
    int removed = 0;
    for (int i = 0; i < *count; ++i) {
        const Experience& e = buffer[i];
        if (e.done) continue;                       /* 终局样本保留 */
        if (experience_semantics::observed_plies(e.state, e.next_state) == 1) ++removed;
    }
    if (!removed) return 0;
    if (*count == REPLAY_CAPACITY)
        std::rotate(buffer, buffer + *next, buffer + *count);
    int kept = 0;
    for (int i = 0; i < *count; ++i) {
        const Experience& e = buffer[i];
        const bool obsolete = !e.done &&
            experience_semantics::observed_plies(e.state, e.next_state) == 1;
        if (!obsolete) buffer[kept++] = e;
    }
    memset(buffer + kept, 0, sizeof(Experience) * (REPLAY_CAPACITY - kept));
    *count = *next = kept;
    return removed;
}

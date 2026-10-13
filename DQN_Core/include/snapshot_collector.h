/* snapshot_collector.h — 快照环形缓冲区
 * 每1000局间隔内采集64个NeuralNet快照，用于多快照技能测试 */
#pragma once
#include "core.h"
#include "skill_metric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SNAPSHOT_COUNT SKILL_WINDOW_SNAPSHOTS

typedef struct {
    NeuralNet snapshots[SNAPSHOT_COUNT];
    int head;           /* 下一个写入位置 */
    int count;          /* 已采集数量（最多 SNAPSHOT_COUNT） */
    int interval;       /* 采集间隔（局数），默认 62 */
    uint64_t last_push_episode;
} SnapshotCollector;

/* 初始化采集器 */
static inline void snapshot_collector_init(SnapshotCollector* sc, int interval) {
    if (!sc) return;
    sc->head = 0;
    sc->count = 0;
    sc->interval = interval > 0 ? interval : SKILL_WINDOW_INTERVAL;
    sc->last_push_episode = 0;
}

/* Check before touching the GPU.  The old caller downloaded both resident
   networks every episode and only then let snapshot_collector_push reject
   61/62 copies. */
static inline int snapshot_collector_due(const SnapshotCollector* sc, uint64_t episode) {
    if (!sc) return 0;
    return sc->count == 0 ||
        episode - sc->last_push_episode >= (uint64_t)sc->interval;
}

/* 尝试推入快照（检查间隔是否足够） */
static inline int snapshot_collector_push(SnapshotCollector* sc, const NeuralNet* net, uint64_t episode) {
    if (!sc || !net) return 0;
    if (!snapshot_collector_due(sc, episode)) return 0;
    copy_net(&sc->snapshots[sc->head], net);
    sc->head = (sc->head + 1) % SNAPSHOT_COUNT;
    if (sc->count < SNAPSHOT_COUNT) sc->count++;
    sc->last_push_episode = episode;
    return 1;
}

/* 获取快照数量 */
static inline int snapshot_collector_count(const SnapshotCollector* sc) {
    return sc ? sc->count : 0;
}

/* 获取快照指针（index 0 = 最旧，count-1 = 最新） */
static inline const NeuralNet* snapshot_collector_get(const SnapshotCollector* sc, int index) {
    if (!sc || index < 0 || index >= sc->count) return NULL;
    int oldest = (sc->count < SNAPSHOT_COUNT) ? 0 : sc->head;
    int pos = (oldest + index) % SNAPSHOT_COUNT;
    return &sc->snapshots[pos];
}

#ifdef __cplusplus
}
#endif

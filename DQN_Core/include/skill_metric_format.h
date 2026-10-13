#pragma once
#include <stdint.h>
#include <math.h>

#define SKILL_SUITE_VERSION 4
#define SKILL_METRIC_FILENAME "metric_skill_v4.bin"
#define SKILL_EMPTY_TRIALS 64
#define SKILL_WINDOW_SNAPSHOTS 64
#define SKILL_WINDOW_INTERVAL 15
#define SKILL_METRIC_FIELDS 29
static const uint32_t SKILL_MAGIC = 0x4B494C53;
struct SkillMetricHeader { uint32_t magic, fields, version, suite, reserved[2]; };
struct SkillMetricRow { int32_t episode, step; float fields[SKILL_METRIC_FIELDS]; };
static_assert(sizeof(SkillMetricHeader) == 24, "skill header layout");
inline bool skill_metric_header_valid(const SkillMetricHeader& h) {
    return h.magic == SKILL_MAGIC && (h.fields == 7 || h.fields == 10 || h.fields == 13 || h.fields == 29) &&
        h.version == 1 && (h.suite == 2 || h.suite == 3 || h.suite == SKILL_SUITE_VERSION);
}
inline bool skill_metric_row_valid(const SkillMetricRow& r) {
    if (r.episode < 0 || r.episode > 10000000 || r.step != 0) return false;
    for (int i = 0; i < SKILL_METRIC_FIELDS; ++i)
        if (!isfinite(r.fields[i]) || fabsf(r.fields[i]) >= 1e30f) return false;

    /* V4 inserted EmptyWin min/max steps before the Win1 columns. The old
       generic "first seven values are rates" rule consequently rejected every
       real V4 row (steps > 1 and margins may be negative) before opening the
       file. Validate each bounded field according to the actual 29-field schema. */
    const int rate_fields[] = {0, 2, 5, 7, 14, 16, 24, 25, 26, 27};
    for (int i : rate_fields)
        if (r.fields[i] < 0.0f || r.fields[i] > 1.0f) return false;
    const int step_fields[] = {1, 3, 4};
    for (int i : step_fields)
        if (r.fields[i] < 0.0f || r.fields[i] > 25.0f) return false;
    if (r.fields[23] < 0.0f || r.fields[23] > 1.0f) return false; /* stddev */
    if (r.fields[28] < 0.0f || r.fields[28] > 10000.0f) return false;
    return true;
}

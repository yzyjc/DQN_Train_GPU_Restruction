#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Shared on-disk contract. Do not change layout without a new version/migration.
// Windows logs are little-endian: 24-byte header, 8-byte prefix, then N floats.
inline constexpr uint32_t METRIC_LOG_MAGIC = 0x44514E4C;
inline constexpr uint32_t METRIC_LOG_VERSION = 1; // version 0 remains readable
inline constexpr int METRIC_LOG_MAX_FIELDS = 14;
#pragma pack(push, 1)
struct MetricLogHeader { uint32_t magic, field_count, version, reserved[3]; };
struct MetricLogPrefix { int32_t episode, step; };
#pragma pack(pop)
static_assert(sizeof(MetricLogHeader) == 24 && sizeof(MetricLogPrefix) == 8,
    "Metric log disk layout changed");
static_assert(sizeof(float) == 4, "Metric logs require 32-bit floats");

struct MetricLogSchema {
    const char* name;
    uint32_t fields;
    bool per_episode; // Missing EP can advance by one only for these streams.
    bool zero_based;  // Storage coordinate; UI converts to completed EP once.
};
inline constexpr MetricLogSchema METRIC_LOG_SCHEMAS[] = {
    {"metric_loss.bin", 3, false, false}, // prefix.step is replay type, NOT update index
    {"metric_epsilon.bin", 1, false, false},
    {"metric_epsilon_episode.bin", 1, true, false},
    {"metric_rewards.bin", 8, true, false},
    {"metric_episode_time.bin", 1, true, false},
    {"metric_game.bin", 1, true, true},
    {"metric_steps.bin", 4, false, true},
    {"metric_win.bin", 4, false, false}, // mode, win%, draw%, lose%; 3 modes per EP
    {"metric_eval.bin", 7, false, false}, // phase, depth, epsilon, mode, win%, draw%, lose%
    {"metric_memory_eval.bin", 14, false, false},
    {"metric_outcome_raw.bin", 14, true, true},
    {"metric_outcome_steps.bin", 12, true, true},
    {"metric_outcome_counts.bin", 12, true, true}
};
// Skill-v2 has a different magic/semantic version; its dedicated reader owns it.
inline const MetricLogSchema* metric_log_schema(const char* name) {
    if (!name) return nullptr;
    const char* base = name;
    for (const char* p = name; *p; ++p) if (*p == '/' || *p == '\\') base = p + 1;
    for (const auto& s : METRIC_LOG_SCHEMAS) if (strcmp(base, s.name) == 0) return &s;
    return nullptr;
}
inline constexpr size_t metric_log_record_size(uint32_t fields) {
    return sizeof(MetricLogPrefix) + fields * sizeof(float);
}
inline bool metric_log_header_valid(const MetricLogHeader& h, uint32_t expected = 0) {
    return h.magic == METRIC_LOG_MAGIC && h.version <= METRIC_LOG_VERSION &&
        h.field_count > 0 && h.field_count <= METRIC_LOG_MAX_FIELDS &&
        (!expected || h.field_count == expected);
}
inline MetricLogHeader metric_log_header(uint32_t fields) {
    return {METRIC_LOG_MAGIC, fields, METRIC_LOG_VERSION, {0, 0, 0}};
}

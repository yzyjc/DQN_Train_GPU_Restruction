#pragma once
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "skill_metric_format.h"

typedef struct {
    int episode;
    /* EmptyWin */
    float empty_win, avg_steps;
    float empty_min_steps, empty_max_steps;
    /* Win1 */
    float win1_fixed, win1_fixed_margin;
    float win1_avg_q_correct, win1_avg_q_wrong;
    float win1_min_q, win1_max_q;
    float win1_min_margin, win1_max_margin;
    /* Block1 */
    float block1_fixed, block1_fixed_margin;
    float block1_avg_q_correct, block1_avg_q_wrong;
    float block1_min_q, block1_max_q;
    float block1_min_margin, block1_max_margin;
    /* 多快照统计 */
    float empty_stat_std, empty_stat_min, empty_stat_max;
    float win1_acc_std, block1_acc_std;
} SkillDisplayRecord;

// Return -1 on temporarily unavailable/invalid input. Do not cache absence.
// A live writer may have written only a prefix of the final record.
inline int read_skill_metrics_tail(const char* path, SkillDisplayRecord* out, int capacity) {
    if (!out || capacity <= 0) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    SkillMetricHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || !skill_metric_header_valid(hdr)) { fclose(f); return -1; }
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    int64_t size = _ftelli64(f);
    if (size < 24) { fclose(f); return -1; }
    const int row_size = 8 + hdr.fields * 4;  /* header fields may be 7 or 10 */
    const int64_t total = (size - 24) / row_size;
    int count = total > capacity ? capacity : (int)total;
    if (_fseeki64(f, 24 + (total-count)*row_size, SEEK_SET) != 0) { fclose(f); return -1; }
    int read = 0;
    for (int i = 0; i < count; ++i) {
        SkillMetricRow record{};
        if (fread(&record, row_size, 1, f) != 1) break;
        if (!skill_metric_row_valid(record)) continue;   /* 跳过损坏记录 */
        if (hdr.fields >= 29) {
            /* 新格式 (29字段)：完整映射 */
            out[read] = {record.episode,
                record.fields[0], record.fields[1],    /* empty_win, avg_steps */
                record.fields[3], record.fields[4],    /* empty_min, empty_max */
                record.fields[5], record.fields[6],    /* win1_fixed, win1_margin */
                record.fields[8], record.fields[9],    /* win1_q_correct, win1_q_wrong */
                record.fields[10], record.fields[11],  /* win1_min_q, win1_max_q */
                record.fields[12], record.fields[13],  /* win1_min_margin, win1_max_margin */
                record.fields[14], record.fields[15],  /* block1_fixed, block1_margin */
                record.fields[17], record.fields[18],  /* block1_q_correct, block1_q_wrong */
                record.fields[19], record.fields[20],  /* block1_min_q, block1_max_q */
                record.fields[21], record.fields[22],  /* block1_min_margin, block1_max_margin */
                record.fields[23], record.fields[24], record.fields[25],  /* empty_stat_std/min/max */
                record.fields[26], record.fields[27]}; /* win1/block1_acc_std */
        } else {
            /* 旧格式 (7/10/13字段)：兼容 */
            out[read] = {record.episode,
                record.fields[0], record.fields[1],    /* empty_win, avg_steps */
                0, 0,                                  /* empty_min/max (旧格式无) */
                record.fields[3], 0,                   /* win1_fixed, margin=0 */
                0, 0,                                  /* win1_q (旧格式无) */
                0, 0, 0, 0,                            /* win1_min/max_q/margin */
                record.fields[5], 0,                   /* block1_fixed, margin=0 */
                0, 0,                                  /* block1_q (旧格式无) */
                0, 0, 0, 0,                            /* block1_min/max_q/margin */
                hdr.fields >= 10 ? record.fields[7] : 0.0f,
                hdr.fields >= 10 ? record.fields[8] : 0.0f,
                hdr.fields >= 10 ? record.fields[9] : 0.0f,
                hdr.fields >= 13 ? record.fields[10] : 0.0f,
                hdr.fields >= 13 ? record.fields[11] : 0.0f};
        }
        read++;
    }
    fclose(f);
    return read;
}

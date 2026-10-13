#pragma once
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "metric_log_format.h"

// Walk backwards in bounded blocks, stopping at the requested EP or the most
// recent rewind. Works with duplicate EPs and missing-prefix placeholders.
// Returns -1 on I/O failure; never fabricates a coordinate.
inline int64_t binary_episode_start(FILE* f, int stride, int64_t end,
    int min_episode = INT_MIN, int64_t header = sizeof(MetricLogHeader)) {
    constexpr size_t max_stride = metric_log_record_size(METRIC_LOG_MAX_FIELDS);
    if (!f || stride < (int)sizeof(MetricLogPrefix) || stride > (int)max_stride || end < header) return -1;
    int64_t index = (end - header) / stride;
    int32_t newer = -1;
    unsigned char block[4096 * max_stride];
    while (index > 0) {
        int count = index > 4096 ? 4096 : (int)index;
        int64_t first = index - count;
        if (_fseeki64(f, header + first * stride, SEEK_SET) != 0 ||
            fread(block, stride, count, f) != (size_t)count) return -1;
        for (int i = count - 1; i >= 0; --i) {
            int32_t ep; memcpy(&ep, block + i * stride, sizeof(ep));
            if (ep < 0) continue;
            if ((newer >= 0 && ep > newer) || ep < min_episode)
                return header + (first + i + 1) * stride;
            newer = ep;
        }
        index = first;
    }
    return header;
}

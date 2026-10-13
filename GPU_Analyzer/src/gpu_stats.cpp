/* gpu_stats.cpp — 统计引擎
   计算 mean/median/P5/P95/P99/stddev/min/max。
   输入是未排序的浮点数组, 内部排序后计算。 */

#include "gpu_telemetry_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 比较函数 (qsort 用) */
static int float_cmp(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/* ========== 公共 API ========== */

GpuStats gpu_stats_compute(const float* data, uint32_t count) {
    GpuStats stats = {};
    if (!data || count == 0) return stats;

    /* 拷贝并排序 */
    float* sorted = (float*)malloc(count * sizeof(float));
    if (!sorted) return stats;
    memcpy(sorted, data, count * sizeof(float));
    qsort(sorted, count, sizeof(float), float_cmp);

    stats.count = count;
    stats.min = sorted[0];
    stats.max = sorted[count - 1];

    /* median */
    if (count % 2 == 0) {
        stats.median = (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0f;
    } else {
        stats.median = sorted[count / 2];
    }

    /* percentiles */
    stats.p5 = sorted[(uint32_t)(count * 0.05f)];
    stats.p95 = sorted[(uint32_t)(count * 0.95f)];
    stats.p99 = sorted[(uint32_t)(count * 0.99f)];

    /* mean */
    double sum = 0;
    for (uint32_t i = 0; i < count; i++) {
        sum += sorted[i];
    }
    stats.mean = (float)(sum / count);

    /* stddev */
    double var = 0;
    for (uint32_t i = 0; i < count; i++) {
        double diff = sorted[i] - stats.mean;
        var += diff * diff;
    }
    stats.stddev = (float)sqrt(var / count);

    free(sorted);
    return stats;
}

/* 打印统计结果 */
void gpu_stats_print(const char* label, const GpuStats* stats, const char* unit) {
    if (!stats || stats->count == 0) {
        printf("  %s: 无数据\n", label);
        return;
    }
    printf("  %s (%u samples):\n", label, stats->count);
    printf("    mean=%.2f median=%.2f stddev=%.2f %s\n",
           stats->mean, stats->median, stats->stddev, unit ? unit : "");
    printf("    min=%.2f P5=%.2f P95=%.2f P99=%.2f max=%.2f %s\n",
           stats->min, stats->p5, stats->p95, stats->p99, stats->max, unit ? unit : "");
}

/* 将统计结果写入 JSON buffer, 返回写入的字符数 */
int gpu_stats_to_json(char* buf, size_t buf_size, const GpuStats* stats) {
    if (!buf || !stats) return 0;
    return snprintf(buf, buf_size,
        "{\"mean\":%.4f,\"median\":%.4f,\"p5\":%.4f,\"p95\":%.4f,\"p99\":%.4f,"
        "\"stddev\":%.4f,\"min\":%.4f,\"max\":%.4f,\"samples\":%u}",
        stats->mean, stats->median, stats->p5, stats->p95, stats->p99,
        stats->stddev, stats->min, stats->max, stats->count);
}
/* gpu_report.cpp — GPU 性能报告生成
   读取二进制日志, 计算统计, 生成 JSON/CSV 报告。
   Phase 3 模块。 */

#include "gpu_telemetry_protocol.h"
#include "gpu_telemetry_types.h"
#include "gpu_report_format.h"
#include "metric_log_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <direct.h>

/* ========== 前向声明 ========== */
extern GpuStats gpu_stats_compute(const float* data, uint32_t count);
extern void gpu_stats_print(const char* label, const GpuStats* stats, const char* unit);
extern int gpu_stats_to_json(char* buf, size_t buf_size, const GpuStats* stats);

/* ========== 二进制日志读取 ========== */

typedef struct {
    int32_t episode;
    int32_t step;
    float   fields[14];  /* METRIC_LOG_MAX_FIELDS */
} LogRecord;

/* 读取二进制日志的所有记录 */
static int read_binary_log(const char* path, LogRecord* out, int max_records,
                            uint32_t* out_field_count) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    /* 读 header */
    MetricLogHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic != METRIC_LOG_MAGIC ||
        hdr.field_count == 0 || hdr.field_count > METRIC_LOG_MAX_FIELDS) {
        fclose(f);
        return -1;
    }

    if (out_field_count) *out_field_count = hdr.field_count;

    /* 读记录 */
    size_t record_size = sizeof(MetricLogPrefix) + hdr.field_count * sizeof(float);
    int count = 0;
    MetricLogPrefix prefix;

    while (count < max_records && fread(&prefix, sizeof(prefix), 1, f) == 1) {
        out[count].episode = prefix.episode;
        out[count].step = prefix.step;
        memset(out[count].fields, 0, sizeof(out[count].fields));
        if (fread(out[count].fields, sizeof(float), hdr.field_count, f) != hdr.field_count)
            break;
        count++;
    }

    fclose(f);
    return count;
}

/* ========== JSON 报告生成 ========== */

/* 写 JSON 字符串 (转义特殊字符) */
static void json_string(FILE* f, const char* key, const char* value) {
    fprintf(f, "\"%s\":\"", key);
    for (const char* p = value; *p; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', f);
        fputc(*p, f);
    }
    fprintf(f, "\"");
}

static void json_int(FILE* f, const char* key, long long value) {
    fprintf(f, "\"%s\":%lld", key, value);
}

static void json_float(FILE* f, const char* key, float value) {
    fprintf(f, "\"%s\":%.4f", key, value);
}

static void json_bool(FILE* f, const char* key, int value) {
    fprintf(f, "\"%s\":%s", key, value ? "true" : "false");
}

static void json_stats(FILE* f, const char* key, const GpuStats* stats) {
    fprintf(f, "\"%s\":{", key);
    json_float(f, "mean", stats->mean);
    fprintf(f, ",");
    json_float(f, "median", stats->median);
    fprintf(f, ",");
    json_float(f, "p5", stats->p5);
    fprintf(f, ",");
    json_float(f, "p95", stats->p95);
    fprintf(f, ",");
    json_float(f, "p99", stats->p99);
    fprintf(f, ",");
    json_float(f, "stddev", stats->stddev);
    fprintf(f, ",");
    json_float(f, "min", stats->min);
    fprintf(f, ",");
    json_float(f, "max", stats->max);
    fprintf(f, ",");
    json_int(f, "samples", stats->count);
    fprintf(f, "}");
}

/* 生成时间戳字符串 */
static void format_timestamp(char* buf, size_t buf_size) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    strftime(buf, buf_size, "%Y%m%d_%H%M%S", t);
}

/* 生成 ISO8601 时间戳 */
static void format_iso8601(char* buf, size_t buf_size, uint64_t ms_since_epoch) {
    time_t sec = (time_t)(ms_since_epoch / 1000);
    struct tm* t = localtime(&sec);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", t);
}

/* ========== 公共 API ========== */

/* 从二进制日志生成 JSON 报告 */
int gpu_report_generate_json(const char* log_path, const GpuDeviceInfo* device_info,
                              const char* output_path) {
    /* 读取日志 */
    LogRecord* records = (LogRecord*)malloc(65536 * sizeof(LogRecord));
    if (!records) return -1;

    uint32_t field_count = 0;
    int count = read_binary_log(log_path, records, 65536, &field_count);
    if (count <= 0) {
        printf("[Report] 无法读取日志 %s (记录数=%d)\n", log_path, count);
        free(records);
        return -1;
    }

    printf("[Report] 读取 %d 条记录 (字段数=%u)\n", count, field_count);

    /* 提取各字段 */
    float* gpu_util = (float*)malloc(count * sizeof(float));
    float* gpu_mem = (float*)malloc(count * sizeof(float));
    float* cpu_usage = (float*)malloc(count * sizeof(float));

    if (!gpu_util || !gpu_mem || !cpu_usage) {
        free(records); free(gpu_util); free(gpu_mem); free(cpu_usage);
        return -1;
    }

    for (int i = 0; i < count; i++) {
        gpu_util[i] = records[i].fields[0];
        gpu_mem[i] = records[i].fields[1];
        cpu_usage[i] = records[i].fields[6];
    }

    /* 计算统计 */
    GpuStats util_stats = gpu_stats_compute(gpu_util, count);
    GpuStats mem_stats = gpu_stats_compute(gpu_mem, count);
    GpuStats cpu_stats = gpu_stats_compute(cpu_usage, count);

    /* 生成报告 */
    char timestamp[32];
    format_timestamp(timestamp, sizeof(timestamp));

    char report_path[512];
    if (output_path) {
        strncpy(report_path, output_path, sizeof(report_path) - 1);
    } else {
        snprintf(report_path, sizeof(report_path), "%s_%s.json", GPU_REPORT_PREFIX, timestamp);
    }

    FILE* f = fopen(report_path, "w");
    if (!f) {
        printf("[Report] 无法创建 %s\n", report_path);
        free(records); free(gpu_util); free(gpu_mem); free(cpu_usage);
        return -1;
    }

    /* JSON 输出 */
    fprintf(f, "{\n");

    /* session */
    fprintf(f, "  \"session\":{\n");
    json_int(f, "sample_count", count);
    fprintf(f, ",\n");
    json_int(f, "sample_interval_ms", 1000);
    fprintf(f, ",\n");
    json_string(f, "mode", "NORMAL");
    fprintf(f, "\n  },\n");

    /* device */
    if (device_info) {
        fprintf(f, "  \"device\":{\n");
        json_string(f, "name", device_info->device_name);
        fprintf(f, ",\n");
        json_string(f, "arch", device_info->arch_name);
        fprintf(f, ",\n");
        json_float(f, "vram_mb", (float)(device_info->dedicated_vram_bytes / (1024.0 * 1024.0)));
        fprintf(f, ",\n");
        json_int(f, "compute_units", device_info->compute_units);
        fprintf(f, ",\n");
        json_int(f, "clock_mhz", device_info->clock_mhz);
        fprintf(f, ",\n");
        json_int(f, "memory_clock_mhz", device_info->memory_clock_mhz);
        fprintf(f, ",\n");
        json_int(f, "memory_bus_width", device_info->memory_bus_width);
        fprintf(f, ",\n");
        json_float(f, "theoretical_bandwidth_gb_s", device_info->theoretical_bw_gb_s);
        fprintf(f, "\n  },\n");
    }

    /* gpu_utilization */
    fprintf(f, "  ");
    json_stats(f, "gpu_utilization", &util_stats);
    fprintf(f, ",\n");

    /* vram_usage_mb */
    fprintf(f, "  ");
    json_stats(f, "vram_usage_mb", &mem_stats);
    fprintf(f, ",\n");

    /* cpu_usage_pct */
    fprintf(f, "  ");
    json_stats(f, "cpu_usage_pct", &cpu_stats);
    fprintf(f, ",\n");

    /* training_performance: 从二进制日志推算 (需要 metric_episode_time.bin 等) */
    fprintf(f, "  \"training_performance\":{\n");
    fprintf(f, "    \"note\":\"Run GPU_Analyzer alongside Trainer for live timing;\"\n");
    fprintf(f, "    ,\"details\":\"See TUI GPU Analyze tab for timing breakdown\"\n");
    fprintf(f, "  },\n");

    /* Availability comes from recorded values, never a hardcoded GPU verdict. */
    const char* names[]={"temperature_c","power_w","clock_mhz","memory_clock_mhz","hotspot_temperature_c"};
    fprintf(f,"  \"sensors\":{\n");
    for(int field=0;field<5;++field) {
        double sum=0;int valid=0;float low=0,high=0;
        const uint32_t index=field<3?(uint32_t)(3+field):(field==3?9u:10u);
        for(int i=0;i<count && index<field_count;++i){float v=records[i].fields[index];
            if(!isfinite(v)||v<0)continue;
            if(!valid)low=high=v;else {if(v<low)low=v;if(v>high)high=v;}
            sum+=v;++valid;
        }
        fprintf(f,"    \"%s\":{\"valid_samples\":%d,\"unavailable\":%s",names[field],valid,valid?"false":"true");
        if(valid)fprintf(f,",\"mean\":%.3f,\"min\":%.3f,\"max\":%.3f",sum/valid,low,high);
        fprintf(f,"}%s\n",field==4?"":",");
    }
    fprintf(f,"  },\n  \"raw_counters\":{\"unavailable\":[\"valu\",\"cache_hit\",\"occupancy\"],\"note\":\"No hardware profiler integrated\"}\n");
    fprintf(f, "}\n");
    fclose(f);

    printf("[Report] JSON 报告已生成: %s\n", report_path);

    /* 打印统计摘要 */
    gpu_stats_print("GPU 利用率", &util_stats, "%");
    gpu_stats_print("VRAM 使用", &mem_stats, "MB");
    gpu_stats_print("CPU 使用率", &cpu_stats, "%");

    free(records); free(gpu_util); free(gpu_mem); free(cpu_usage);
    return 0;
}

/* 生成 CSV 时间序列 */
int gpu_report_generate_csv(const char* log_path, const char* output_path) {
    LogRecord* records = (LogRecord*)malloc(65536 * sizeof(LogRecord));
    if (!records) return -1;

    uint32_t field_count = 0;
    int count = read_binary_log(log_path, records, 65536, &field_count);
    if (count <= 0) {
        free(records);
        return -1;
    }

    char timestamp[32];
    format_timestamp(timestamp, sizeof(timestamp));

    char csv_path[512];
    if (output_path) {
        strncpy(csv_path, output_path, sizeof(csv_path) - 1);
    } else {
        snprintf(csv_path, sizeof(csv_path), "%s_%s.csv", GPU_REPORT_PREFIX, timestamp);
    }

    FILE* f = fopen(csv_path, "w");
    if (!f) {
        free(records);
        return -1;
    }

    /* CSV header tracks the recorded schema; preserve compatibility with old 9-field logs. */
    if (field_count >= 11) {
        fprintf(f, "%s\n", GPU_CSV_HEADER);
    } else {
        fprintf(f, "episode,step,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,"
                   "gpu_temp_c,gpu_power_w,gpu_clock_mhz,cpu_usage_pct,"
                   "sys_mem_used_gb,sys_mem_total_gb\n");
    }

    /* 数据行 */
    for (int i = 0; i < count; i++) {
        fprintf(f, "%d,%d", records[i].episode, records[i].step);
        for (uint32_t j = 0; j < field_count && j < 14; j++) {
            fprintf(f, ",%.4f", records[i].fields[j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("[Report] CSV 已生成: %s (%d 条记录)\n", csv_path, count);

    free(records);
    return 0;
}

/* ========== 硬件对比 ========== */

/* 读取 JSON 报告中的统计值 (简化实现, 不用完整 JSON parser) */
static float extract_json_stat(const char* json_path, const char* section, const char* field) {
    FILE* f = fopen(json_path, "r");
    if (!f) return -1.0f;

    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    /* 搜索 "section":{ ... "field":value } */
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":{", section);
    char* start = strstr(buf, pattern);
    if (!start) return -1.0f;

    snprintf(pattern, sizeof(pattern), "\"%s\":", field);
    char* pos = strstr(start, pattern);
    if (!pos) return -1.0f;

    pos += strlen(pattern);
    return (float)atof(pos);
}

/* 生成对比报告 */
int gpu_report_compare(const char* report_a_path, const char* report_b_path,
                        const char* output_path) {
    char timestamp[32];
    format_timestamp(timestamp, sizeof(timestamp));

    char compare_path[512];
    if (output_path) {
        strncpy(compare_path, output_path, sizeof(compare_path) - 1);
    } else {
        snprintf(compare_path, sizeof(compare_path), "comparison_%s.json", timestamp);
    }

    /* 读取两个报告的关键指标 */
    struct {
        const char* section;
        const char* field;
        const char* label;
    } metrics[] = {
        {"gpu_utilization", "mean", "gpu_util_mean"},
        {"gpu_utilization", "median", "gpu_util_median"},
        {"gpu_utilization", "p95", "gpu_util_p95"},
        {"vram_usage_mb", "mean", "vram_mean_mb"},
        {"cpu_usage_pct", "mean", "cpu_mean_pct"},
    };

    FILE* f = fopen(compare_path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"devices\":[\n");
    fprintf(f, "    {\"report\":\"%s\"},\n", report_a_path);
    fprintf(f, "    {\"report\":\"%s\"}\n", report_b_path);
    fprintf(f, "  ],\n");
    fprintf(f, "  \"metrics\":{\n");

    int first = 1;
    for (int i = 0; i < 5; i++) {
        float val_a = extract_json_stat(report_a_path, metrics[i].section, metrics[i].field);
        float val_b = extract_json_stat(report_b_path, metrics[i].section, metrics[i].field);

        if (val_a < 0 && val_b < 0) continue;

        if (!first) fprintf(f, ",\n");
        first = 0;

        fprintf(f, "    \"%s\":{", metrics[i].label);
        fprintf(f, "\"A\":%.4f", val_a);
        fprintf(f, ",\"B\":%.4f", val_b);
        if (val_a > 0) {
            float delta_pct = (val_b - val_a) / val_a * 100.0f;
            fprintf(f, ",\"delta_pct\":%.2f", delta_pct);
        }
        fprintf(f, "}");
    }

    fprintf(f, "\n  }\n");
    fprintf(f, "}\n");
    fclose(f);

    printf("[Report] 对比报告已生成: %s\n", compare_path);
    return 0;
}

/* ========== 命令行入口 ========== */

void gpu_report_print_help(void) {
    printf("GPU 性能报告生成器\n\n");
    printf("用法:\n");
    printf("  GPU_Analyzer.exe --report <log_path> [--output <path>]   生成 JSON 报告\n");
    printf("  GPU_Analyzer.exe --csv <log_path> [--output <path>]     生成 CSV 时间序列\n");
    printf("  GPU_Analyzer.exe --compare <report_a> <report_b>        硬件对比\n");
    printf("\n");
    printf("示例:\n");
    printf("  GPU_Analyzer.exe --report logs/metric_gpu_telemetry_v2.bin\n");
    printf("  GPU_Analyzer.exe --csv logs/metric_gpu_telemetry_v2.bin --output telemetry.csv\n");
    printf("  GPU_Analyzer.exe --compare report_8060S.json report_MI50.json\n");
}

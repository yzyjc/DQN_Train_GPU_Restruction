/* gpu_binary_log.cpp — GPU 遥测二进制日志写入
   复用 metric_log_format.h 的 24-byte header + 8-byte prefix + N-float 模式。
   GPU_Analyzer 是唯一的写者, 不需要锁。 */

#include "gpu_telemetry_protocol.h"
#include "gpu_telemetry_types.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <direct.h>  /* _mkdir */

/* 复用 DQN_Core 的 metric log 格式 */
#include "metric_log_format.h"

/* ========== 内部状态 ========== */

static FILE* g_log_file = NULL;
static char  g_log_path[512] = {0};
static uint32_t g_log_fields = 0;

/* ========== 公共 API ========== */

int gpu_binary_log_open(const char* log_dir, uint32_t field_count) {
    if (g_log_file) return 0;  /* 已打开 */

    /* 确保目录存在 */
    _mkdir(log_dir);

    /* New sensor columns use a versioned file; preserve existing 9-field logs. */
    snprintf(g_log_path, sizeof(g_log_path), "%s\\metric_gpu_telemetry_v2.bin", log_dir);

    g_log_file = fopen(g_log_path, "ab");
    if (!g_log_file) {
        printf("[BinaryLog] 打开 %s 失败\n", g_log_path);
        return -1;
    }

    g_log_fields = field_count;

    /* 检查文件是否为空 (需要写 header) */
    fseek(g_log_file, 0, SEEK_END);
    long size = ftell(g_log_file);

    if (size == 0) {
        /* 写 24-byte header */
        MetricLogHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic = METRIC_LOG_MAGIC;
        hdr.field_count = field_count;
        hdr.version = METRIC_LOG_VERSION;
        fwrite(&hdr, sizeof(hdr), 1, g_log_file);
        fflush(g_log_file);
        printf("[BinaryLog] 创建 %s (fields=%u)\n", g_log_path, field_count);
    } else {
        /* 验证已有 header */
        MetricLogHeader hdr;
        fseek(g_log_file, 0, SEEK_SET);
        if (fread(&hdr, sizeof(hdr), 1, g_log_file) != 1 ||
            hdr.magic != METRIC_LOG_MAGIC ||
            hdr.field_count != field_count) {
            printf("[BinaryLog] 文件 %s 格式不匹配, 将追加新 header\n", g_log_path);
            /* 不清空文件, 追加新 header (reader 会跳过无效记录) */
            fseek(g_log_file, 0, SEEK_END);
        }
        fseek(g_log_file, 0, SEEK_END);
        printf("[BinaryLog] 追加到 %s\n", g_log_path);
    }

    return 0;
}

void gpu_binary_log_close(void) {
    if (g_log_file) {
        fflush(g_log_file);
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

/* 写入一条记录 */
int gpu_binary_log_write(int32_t episode, int32_t step,
                          const float* fields, uint32_t field_count) {
    if (!g_log_file || !fields) return -1;
    if (field_count != g_log_fields) return -1;

    /* 写 prefix (episode, step) */
    MetricLogPrefix prefix;
    prefix.episode = episode;
    prefix.step = step;
    fwrite(&prefix, sizeof(prefix), 1, g_log_file);

    /* 写 fields */
    fwrite(fields, sizeof(float), field_count, g_log_file);

    /* 每条记录刷新一次 (安全, 性能可接受) */
    fflush(g_log_file);
    return 0;
}

/* 批量写入 (减少 fflush 开销) */
int gpu_binary_log_write_batch(int32_t episode, int32_t step,
                                const float* fields, uint32_t field_count,
                                uint32_t count) {
    if (!g_log_file || !fields || count == 0) return -1;
    if (field_count != g_log_fields) return -1;

    for (uint32_t i = 0; i < count; i++) {
        MetricLogPrefix prefix;
        prefix.episode = episode;
        prefix.step = step + (int32_t)i;
        fwrite(&prefix, sizeof(prefix), 1, g_log_file);
        fwrite(&fields[i * field_count], sizeof(float), field_count, g_log_file);
    }

    fflush(g_log_file);
    return 0;
}

/* 获取日志文件路径 */
const char* gpu_binary_log_get_path(void) {
    return g_log_path;
}

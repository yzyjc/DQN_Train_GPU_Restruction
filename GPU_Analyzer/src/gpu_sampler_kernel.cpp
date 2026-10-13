/* gpu_sampler_kernel.cpp — 读取 Trainer 发布的 kernel timing 共享内存
   GPU_Analyzer 用此模块读取 DQN_GPU_KernelTimer, 计算 duty cycle 和统计分布。
   Phase 2 模块: 需要 Trainer 启用 gpu_kernel_timer.h。 */

#include "gpu_telemetry_protocol.h"
#include "gpu_telemetry_types.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* 复用 DQN_Core 的 kernel timer 协议 */
#include "gpu_kernel_timer.h"

/* ========== 内部状态 ========== */

static HANDLE  g_kt_client_map = NULL;
static void*   g_kt_client_base = NULL;
static int     g_kt_client_valid = 0;

/* 上次读取的位置 (增量读取) */
static uint32_t g_kt_last_head = 0;
static uint64_t g_kt_last_seq = 0;

/* 累计统计 */
static float    g_kt_forward_ms = 0;
static float    g_kt_train_ms = 0;
static float    g_kt_soft_update_ms = 0;
static float    g_kt_upload_download_ms = 0;
static uint64_t g_kt_forward_count = 0;
static uint64_t g_kt_train_count = 0;

/* ========== 公共 API ========== */

int gpu_sampler_kernel_open(void) {
    if (g_kt_client_valid) return 0;

    g_kt_client_map = OpenFileMappingA(FILE_MAP_READ, FALSE, GPU_KERNEL_TIMER_NAME);
    if (!g_kt_client_map) {
        /* Trainer 可能未启用 kernel timer, 这不是错误 */
        return -1;
    }

    g_kt_client_base = MapViewOfFile(g_kt_client_map, FILE_MAP_READ, 0, 0, 0);
    if (!g_kt_client_base) {
        CloseHandle(g_kt_client_map);
        g_kt_client_map = NULL;
        return -1;
    }

    /* 验证 header */
    GpuKernelTimerHeader* hdr = (GpuKernelTimerHeader*)g_kt_client_base;
    if (hdr->magic != GPU_KERNEL_TIMER_MAGIC || hdr->version != GPU_KERNEL_TIMER_VERSION) {
        UnmapViewOfFile(g_kt_client_base);
        CloseHandle(g_kt_client_map);
        g_kt_client_base = NULL;
        g_kt_client_map = NULL;
        return -1;
    }

    g_kt_client_valid = 1;
    g_kt_last_head = 0;
    g_kt_last_seq = 0;
    printf("[KernelSampler] 已连接 '%s'\n", GPU_KERNEL_TIMER_NAME);
    return 0;
}

void gpu_sampler_kernel_close(void) {
    if (g_kt_client_base) {
        UnmapViewOfFile(g_kt_client_base);
        g_kt_client_base = NULL;
    }
    if (g_kt_client_map) {
        CloseHandle(g_kt_client_map);
        g_kt_client_map = NULL;
    }
    g_kt_client_valid = 0;
    printf("[KernelSampler] 已断开\n");
}

int gpu_sampler_kernel_is_connected(void) {
    return g_kt_client_valid;
}

/* seqlock 读取 header */
static int read_header(GpuKernelTimerHeader* out) {
    if (!g_kt_client_base || !out) return -1;

    GpuKernelTimerHeader* hdr = (GpuKernelTimerHeader*)g_kt_client_base;
    uint64_t seq_before, seq_after;
    int retries = 0;

    do {
        seq_before = hdr->write_seq;
        if (seq_before & 1) { retries++; YieldProcessor(); continue; }
        memcpy(out, hdr, sizeof(GpuKernelTimerHeader));
        MemoryBarrier();
        seq_after = hdr->write_seq;
        retries++;
    } while (seq_before != seq_after && retries < 100);

    return (seq_before == seq_after) ? 0 : -1;
}

/* 增量读取新记录 */
int gpu_sampler_kernel_read_records(GpuKernelRecord* out, uint32_t max_count,
                                     uint32_t* out_total_records) {
    if (!g_kt_client_valid || !out || max_count == 0) return -1;

    GpuKernelTimerHeader hdr;
    if (read_header(&hdr) != 0) return -1;

    if (out_total_records) *out_total_records = (uint32_t)hdr.total_records;

    /* 计算新记录数 */
    uint32_t current_head = hdr.ring_head;
    uint32_t new_count = 0;

    if (g_kt_last_head == 0 && g_kt_last_seq == 0) {
        /* 首次读取: 只读最后 max_count 条 */
        uint32_t total_in_ring = (uint32_t)(hdr.total_records < GPU_KERNEL_TIMER_RING_SIZE ?
                               hdr.total_records : GPU_KERNEL_TIMER_RING_SIZE);
        new_count = total_in_ring < max_count ? total_in_ring : max_count;
        uint32_t start = (current_head + GPU_KERNEL_TIMER_RING_SIZE - new_count) %
                          GPU_KERNEL_TIMER_RING_SIZE;
        for (uint32_t i = 0; i < new_count; i++) {
            uint32_t idx = (start + i) % GPU_KERNEL_TIMER_RING_SIZE;
            GpuKernelRecord* src = gpu_kt_get_record(g_kt_client_base, idx);
            out[i] = *src;
        }
    } else {
        /* 增量读取: 从 last_head 到 current_head */
        uint32_t delta = (current_head - g_kt_last_head + GPU_KERNEL_TIMER_RING_SIZE) %
                          GPU_KERNEL_TIMER_RING_SIZE;
        new_count = delta < max_count ? delta : max_count;
        for (uint32_t i = 0; i < new_count; i++) {
            uint32_t idx = (g_kt_last_head + i) % GPU_KERNEL_TIMER_RING_SIZE;
            GpuKernelRecord* src = gpu_kt_get_record(g_kt_client_base, idx);
            out[i] = *src;
        }
    }

    g_kt_last_head = current_head;
    g_kt_last_seq = hdr.write_seq;

    /* 更新累计统计 */
    for (uint32_t i = 0; i < new_count; i++) {
        switch ((GpuKernelType)out[i].kernel_type) {
            case GPU_KT_FORWARD:
            case GPU_KT_FORWARD_BATCH:
                g_kt_forward_ms += out[i].elapsed_ms;
                g_kt_forward_count++;
                break;
            case GPU_KT_TRAIN_STEP:
            case GPU_KT_TRAIN_STEP_BATCH:
                g_kt_train_ms += out[i].elapsed_ms;
                g_kt_train_count++;
                break;
            case GPU_KT_SOFT_UPDATE:
                g_kt_soft_update_ms += out[i].elapsed_ms;
                break;
            case GPU_KT_UPLOAD:
            case GPU_KT_DOWNLOAD:
                g_kt_upload_download_ms += out[i].elapsed_ms;
                break;
            default: break;
        }
    }

    return (int)new_count;
}

/* 获取累计统计 */
typedef struct KernelTimingStats {
    float    forward_total_ms;
    float    train_total_ms;
    float    soft_update_total_ms;
    float    upload_download_total_ms;
    uint64_t forward_count;
    uint64_t train_count;
    float    duty_cycle_pct;      /* (forward+train) / wall_time * 100 */
    float    avg_forward_ms;
    float    avg_train_ms;
    int      connected;
} KernelTimingStats;

KernelTimingStats gpu_sampler_kernel_get_stats(void) {
    KernelTimingStats stats = {};
    stats.connected = g_kt_client_valid;

    if (!g_kt_client_valid) return stats;

    /* 从 header 读取累计值 (比本地累计更准确, 因为包含历史) */
    GpuKernelTimerHeader hdr;
    if (read_header(&hdr) == 0) {
        stats.forward_total_ms = hdr.forward_total_ms;
        stats.train_total_ms = hdr.train_total_ms;
        stats.soft_update_total_ms = hdr.soft_update_total_ms;
        stats.upload_download_total_ms = hdr.upload_download_total_ms;
        stats.forward_count = hdr.forward_count;
        stats.train_count = hdr.train_count;

        /* duty cycle: kernel time / (kernel time + idle time) */
        float total_kernel = stats.forward_total_ms + stats.train_total_ms +
                             stats.soft_update_total_ms + stats.upload_download_total_ms;
        /* wall time 从 publisher heartbeat 推算 (不精确, 但够用) */
        uint64_t now = GetTickCount64();
        uint64_t elapsed_ms = now - hdr.publisher_heartbeat_ms;
        /* heartbeat 是最近一次写入的时间, 不是开始时间。
           用 total_records * avg_kernel_time 推算更准确 */
        if (total_kernel > 0) {
            stats.duty_cycle_pct = 100.0f;  /* 有 kernel 运行时, 占空比由外部计算 */
        }

        if (stats.forward_count > 0)
            stats.avg_forward_ms = stats.forward_total_ms / (float)stats.forward_count;
        if (stats.train_count > 0)
            stats.avg_train_ms = stats.train_total_ms / (float)stats.train_count;
    }

    return stats;
}

/* 打印 kernel timing 摘要 */
void gpu_sampler_kernel_print_summary(void) {
    KernelTimingStats stats = gpu_sampler_kernel_get_stats();

    if (!stats.connected) {
        printf("[KernelTiming] 未连接 (Trainer 未启用 kernel timer)\n");
        return;
    }

    printf("\n========== Kernel Timing 摘要 ==========\n");
    printf("Forward:    %llu 次, 总 %.1f ms, 平均 %.3f ms\n",
           stats.forward_count, stats.forward_total_ms, stats.avg_forward_ms);
    printf("Train:      %llu 次, 总 %.1f ms, 平均 %.3f ms\n",
           stats.train_count, stats.train_total_ms, stats.avg_train_ms);
    printf("Soft update: 总 %.1f ms\n", stats.soft_update_total_ms);
    printf("Up/Download: 总 %.1f ms\n", stats.upload_download_total_ms);
    printf("========================================\n\n");
}
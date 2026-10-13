/* gpu_telemetry_client.h — TUI 侧 GPU 遥测客户端
   从 GPU_Analyzer 发布的共享内存读取遥测数据。
   参考 neural_probe_client.cpp 的 seqlock 读取模式。

   注意: 不直接 include gpu_telemetry_protocol.h, 因为 TUI 已有同名
   GpuDeviceInfo 结构体 (含 backend/name 等字段)。这里只定义最小所需类型。 */
#pragma once
#include <windows.h>
#include <stdint.h>
#include <string.h>

/* 共享内存名称和常量 (与 gpu_telemetry_protocol.h 一致) */
#define GPU_TELEM_NAME          "DQN_GPU_Telemetry"
#define GPU_TELEM_MAGIC         0x4754454C
#define GPU_TELEM_VERSION       1
#define GPU_TELEM_SENSOR_EXT_MAGIC 0x53454E53
#define GPU_TELEM_RING_SIZE     4096
#define GPU_TELEM_HEARTBEAT_MS  3000

/* 遥测采样 (64 bytes, 与 GpuTelemetrySample 一致) */
typedef struct GpuTelemSample {
    uint64_t timestamp_ms;
    uint32_t sequence;
    float    gpu_util_pct;
    float    gpu_mem_used_mb;
    float    gpu_mem_total_mb;
    float    gpu_temp_c;
    float    gpu_power_w;
    float    gpu_clock_mhz;
    float    cpu_usage_pct;
    float    sys_mem_used_gb;
    float    sys_mem_total_gb;
    float    gpu_memory_clock_mhz;
    float    gpu_hotspot_temp_c;
    uint32_t _reserved;
} GpuTelemSample;
static_assert(sizeof(GpuTelemSample) == 64, "TUI telemetry sample must remain 64 bytes");

/* 共享内存 header (只保留 TUI 需要的字段) */
typedef struct GpuTelemHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t total_size;
    uint32_t sample_interval_ms;
    uint32_t sample_mode;
    volatile uint64_t write_seq;
    volatile uint32_t ring_head;
    volatile uint32_t sample_count;
    uint64_t publisher_pid;
    uint64_t publisher_heartbeat_ms;
    float    ema_gpu_util;
    float    ema_gpu_mem;
    float    max_gpu_util;
    float    max_gpu_mem;
    uint32_t ring_buffer_offset;    /* ring buffer 在映射中的字节偏移 */
    uint32_t _reserved[7];
} GpuTelemHeader;

/* Ring buffer 偏移: 从 header 的 ring_buffer_offset 字段读取 (GPU_Analyzer 写入) */
static uint32_t g_telem_ring_offset = 0;

/* ========== Kernel Timer 客户端 (读取 Trainer 发布的 DQN_GPU_KernelTimer) ========== */
#define GPU_KT_NAME "DQN_GPU_KernelTimer"
#define GPU_KT_MAGIC 0x4B54494D
#define GPU_KT_RING_SIZE 16384

typedef struct GpuKtHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t total_size;
    volatile uint64_t write_seq;
    volatile uint32_t ring_head;
    volatile uint64_t total_records;
    uint64_t publisher_pid;
    uint64_t publisher_heartbeat_ms;
    float    forward_total_ms;
    float    train_total_ms;
    float    soft_update_total_ms;
    float    upload_download_total_ms;
    uint64_t forward_count;
    uint64_t train_count;
    uint64_t _reserved[4];
} GpuKtHeader;

static HANDLE  g_kt_client_map = NULL;
static void*   g_kt_client_base = NULL;
static int     g_kt_client_valid = 0;

static int gpu_kt_client_open(void) {
    if (g_kt_client_valid) return 0;
    g_kt_client_map = OpenFileMappingA(FILE_MAP_READ, FALSE, GPU_KT_NAME);
    if (!g_kt_client_map) return -1;
    g_kt_client_base = MapViewOfFile(g_kt_client_map, FILE_MAP_READ, 0, 0, 0);
    if (!g_kt_client_base) { CloseHandle(g_kt_client_map); g_kt_client_map = NULL; return -1; }
    GpuKtHeader* hdr = (GpuKtHeader*)g_kt_client_base;
    if (hdr->magic != GPU_KT_MAGIC) {
        UnmapViewOfFile(g_kt_client_base); CloseHandle(g_kt_client_map);
        g_kt_client_base = NULL; g_kt_client_map = NULL; return -1;
    }
    g_kt_client_valid = 1;
    return 0;
}

static void gpu_kt_client_close(void) {
    if (g_kt_client_base) { UnmapViewOfFile(g_kt_client_base); g_kt_client_base = NULL; }
    if (g_kt_client_map) { CloseHandle(g_kt_client_map); g_kt_client_map = NULL; }
    g_kt_client_valid = 0;
}

static int gpu_kt_client_is_alive(void) {
    if (!g_kt_client_valid) return 0;
    GpuKtHeader* hdr = (GpuKtHeader*)g_kt_client_base;
    return (GetTickCount64() - hdr->publisher_heartbeat_ms) < 5000;
}

/* 获取 kernel timing 累计统计 (返回 0=成功) */
static int gpu_kt_client_get_stats(float* fwd_ms, uint64_t* fwd_cnt,
                                     float* train_ms, uint64_t* train_cnt,
                                     float* sync_ms, float* io_ms) {
    if (!g_kt_client_valid) return -1;
    GpuKtHeader* hdr = (GpuKtHeader*)g_kt_client_base;
    uint64_t seq_before, seq_after;
    int retries = 0;
    do {
        seq_before = hdr->write_seq;
        if (seq_before & 1) { retries++; YieldProcessor(); continue; }
        if (fwd_ms) *fwd_ms = hdr->forward_total_ms;
        if (fwd_cnt) *fwd_cnt = hdr->forward_count;
        if (train_ms) *train_ms = hdr->train_total_ms;
        if (train_cnt) *train_cnt = hdr->train_count;
        if (sync_ms) *sync_ms = hdr->soft_update_total_ms;
        if (io_ms) *io_ms = hdr->upload_download_total_ms;
        MemoryBarrier();
        seq_after = hdr->write_seq;
        retries++;
    } while (seq_before != seq_after && retries < 100);
    return (seq_before == seq_after) ? 0 : -1;
}

/* 获取 ring buffer 中第 i 个采样的指针 */
static GpuTelemSample* gpu_telem_get_sample(void* base, uint32_t i) {
    return (GpuTelemSample*)((char*)base + g_telem_ring_offset +
                              sizeof(GpuTelemSample) * (i % GPU_TELEM_RING_SIZE));
}

/* ========== 内部状态 ========== */
static HANDLE  g_telem_client_map = NULL;
static void*   g_telem_client_base = NULL;
static int     g_telem_client_valid = 0;

/* ========== 公共 API ========== */

/* 连接到 GPU_Analyzer 发布的共享内存 */
static int gpu_telem_client_open(void) {
    if (g_telem_client_valid) return 0;

    g_telem_client_map = OpenFileMappingA(FILE_MAP_READ, FALSE, GPU_TELEM_NAME);
    if (!g_telem_client_map) return -1;

    g_telem_client_base = MapViewOfFile(g_telem_client_map, FILE_MAP_READ, 0, 0, 0);
    if (!g_telem_client_base) {
        CloseHandle(g_telem_client_map);
        g_telem_client_map = NULL;
        return -1;
    }

    /* 验证 header */
    GpuTelemHeader* hdr = (GpuTelemHeader*)g_telem_client_base;
    if (hdr->magic != GPU_TELEM_MAGIC || hdr->version != GPU_TELEM_VERSION) {
        UnmapViewOfFile(g_telem_client_base);
        CloseHandle(g_telem_client_map);
        g_telem_client_base = NULL;
        g_telem_client_map = NULL;
        return -1;
    }

    /* 读取 ring buffer 偏移 (GPU_Analyzer 写入, 跨进程兼容) */
    g_telem_ring_offset = hdr->ring_buffer_offset;
    if (g_telem_ring_offset == 0) {
        /* 兼容旧版: 默认 header 之后紧接 ring buffer */
        g_telem_ring_offset = (uint32_t)sizeof(GpuTelemHeader);
    }

    g_telem_client_valid = 1;
    return 0;
}

/* 断开连接 */
static void gpu_telem_client_close(void) {
    if (g_telem_client_base) {
        UnmapViewOfFile(g_telem_client_base);
        g_telem_client_base = NULL;
    }
    if (g_telem_client_map) {
        CloseHandle(g_telem_client_map);
        g_telem_client_map = NULL;
    }
    g_telem_client_valid = 0;
}

/* 检查是否已连接 */
static int gpu_telem_client_is_connected(void) {
    return g_telem_client_valid;
}

/* 检查 publisher 是否存活 (心跳超时) */
static int gpu_telem_client_is_alive(void) {
    if (!g_telem_client_valid) return 0;
    GpuTelemHeader* hdr = (GpuTelemHeader*)g_telem_client_base;
    if (hdr->magic != GPU_TELEM_MAGIC) return 0;
    uint64_t now = GetTickCount64();
    return (now - hdr->publisher_heartbeat_ms) < GPU_TELEM_HEARTBEAT_MS;
}

/* 获取最新采样 (seqlock 读取) */
static int gpu_telem_client_get_latest(GpuTelemSample* out) {
    if (!g_telem_client_valid || !out) return -1;

    GpuTelemHeader* hdr = (GpuTelemHeader*)g_telem_client_base;
    uint64_t seq_before, seq_after;
    int retries = 0;

    do {
        seq_before = hdr->write_seq;
        if (seq_before & 1) { retries++; YieldProcessor(); continue; }

        uint32_t head = hdr->ring_head;
        if (head == 0 && hdr->sample_count == 0) return -1;  /* 无数据 */

        uint32_t latest_idx = (head + GPU_TELEM_RING_SIZE - 1) % GPU_TELEM_RING_SIZE;
        GpuTelemSample* src = gpu_telem_get_sample(g_telem_client_base, latest_idx);
        *out = *src;

        MemoryBarrier();
        seq_after = hdr->write_seq;
        retries++;
    } while (seq_before != seq_after && retries < 100);

    if (seq_before != seq_after) return -1;

    /* Older version-1 publishers leave this former padding zeroed. */
    if (out->_reserved != GPU_TELEM_SENSOR_EXT_MAGIC) {
        out->gpu_memory_clock_mhz = -1.0f;
        out->gpu_hotspot_temp_c = -1.0f;
    }
    return 0;
}

/* 获取统计摘要 */
static int gpu_telem_client_get_stats(float* out_ema_util, float* out_ema_mem,
                                       float* out_peak_util, float* out_peak_mem,
                                       uint32_t* out_sample_count) {
    if (!g_telem_client_valid) return -1;
    GpuTelemHeader* hdr = (GpuTelemHeader*)g_telem_client_base;

    if (out_ema_util) *out_ema_util = hdr->ema_gpu_util;
    if (out_ema_mem) *out_ema_mem = hdr->ema_gpu_mem;
    if (out_peak_util) *out_peak_util = hdr->max_gpu_util;
    if (out_peak_mem) *out_peak_mem = hdr->max_gpu_mem;
    if (out_sample_count) *out_sample_count = hdr->sample_count;
    return 0;
}

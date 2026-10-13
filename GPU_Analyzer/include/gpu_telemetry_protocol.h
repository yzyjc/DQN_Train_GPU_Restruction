/* gpu_telemetry_protocol.h — GPU 遥测共享内存协议
   GPU_Analyzer 发布，TUI 读取。所有 offset 均相对于 shared memory base。
   参考 neural_probe_protocol.h 的 seqlock 模式。 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 常量 ========== */
#define GPU_TELEM_MAGIC         0x4754454C  /* "GTEL" */
#define GPU_TELEM_VERSION       1
#define GPU_TELEM_SENSOR_EXT_MAGIC 0x53454E53 /* "SENS" marks the former padding fields */
#define GPU_TELEM_NAME          "DQN_GPU_Telemetry"
#define GPU_TELEM_RING_SIZE     4096        /* ~68min @ 1Hz, ~6.8min @ 10Hz */
#define GPU_TELEM_HEARTBEAT_MS  3000        /* TUI 判定 publisher 离线的超时 */

/* 采样模式 */
#define GPU_TELEM_MODE_NORMAL       0   /* 1000ms, 长期运行 */
#define GPU_TELEM_MODE_TRACE        1   /* 100ms, 短期分析 */
#define GPU_TELEM_MODE_DEEP_PROFILE 2   /* 100ms + ADLX/WMI */

/* 默认采样间隔 (ms) */
#define GPU_TELEM_INTERVAL_NORMAL_MS    1000
#define GPU_TELEM_INTERVAL_TRACE_MS     100
#define GPU_TELEM_INTERVAL_DEEP_MS      100

/* ========== 设备信息 (静态, 启动时写一次) ========== */
#define GPU_TELEM_DEVICE_NAME_LEN   64
#define GPU_TELEM_ARCH_NAME_LEN     32

typedef struct GpuDeviceInfo {
    char     device_name[GPU_TELEM_DEVICE_NAME_LEN];
    char     arch_name[GPU_TELEM_ARCH_NAME_LEN];
    uint64_t total_vram_bytes;
    uint32_t compute_units;
    uint32_t clock_mhz;            /* GPU 时钟 (MHz) */
    uint32_t memory_clock_mhz;     /* 显存时钟 (MHz) */
    uint32_t memory_bus_width;     /* 显存位宽 (bit) */
    uint32_t l2_cache_size;        /* L2 缓存 (bytes) */
    uint32_t max_shared_per_cu;    /* 每 CU 最大共享内存 (bytes) */
    uint32_t warp_size;
    float    theoretical_bw_gb_s;  /* 理论显存带宽 (GB/s) */
    uint32_t vendor_id;            /* PCI Vendor ID */
    uint32_t device_id;            /* PCI Device ID */
    uint64_t dedicated_vram_bytes; /* DXGI DedicatedVideoMemory */
    uint64_t shared_vram_bytes;    /* DXGI SharedSystemMemory */
    uint32_t _reserved[8];
} GpuDeviceInfo;

/* ========== 遥测采样 (每 64 bytes, 缓存行对齐) ========== */
typedef struct GpuTelemetrySample {
    uint64_t timestamp_ms;      /* GetTickCount64() */
    uint32_t sequence;          /* 单调递增 */
    float    gpu_util_pct;      /* PDH GPU Engine utilization (max of engines) */
    float    gpu_mem_used_mb;   /* hipMemGetInfo: total - free */
    float    gpu_mem_total_mb;  /* hipMemGetInfo: total */
    float    gpu_temp_c;        /* ADLX/WMI, -1 if unavailable */
    float    gpu_power_w;       /* ADLX, -1 if unavailable */
    float    gpu_clock_mhz;    /* ADLX, -1 if unavailable */
    float    cpu_usage_pct;     /* GetSystemTimes */
    float    sys_mem_used_gb;   /* GetPerformanceInfo */
    float    sys_mem_total_gb;
    float    gpu_memory_clock_mhz; /* ADLX, -1 if unavailable; requires _reserved marker */
    float    gpu_hotspot_temp_c;   /* ADLX, -1 if unavailable; requires _reserved marker */
    uint32_t _reserved;            /* GPU_TELEM_SENSOR_EXT_MAGIC for new publishers */
} GpuTelemetrySample;
static_assert(sizeof(GpuTelemetrySample) == 64, "GpuTelemetrySample must be 64 bytes");

/* ========== 共享内存 Header ========== */
typedef struct GpuTelemetryHeader {
    uint32_t magic;                 /* GPU_TELEM_MAGIC */
    uint32_t version;               /* GPU_TELEM_VERSION */
    uint64_t total_size;            /* 整个映射的大小 */
    uint32_t sample_interval_ms;    /* 当前采样间隔 */
    uint32_t sample_mode;           /* GPU_TELEM_MODE_* */
    volatile uint64_t write_seq;    /* seqlock: 奇数=写入中, 偶数=写入完成 */
    volatile uint32_t ring_head;    /* 下一个写入位置 */
    volatile uint32_t sample_count; /* 已写入总数 (不回绕) */
    uint64_t publisher_pid;
    uint64_t publisher_heartbeat_ms; /* GetTickCount64() */
    /* 累计统计 (GPU_Analyzer 维护, TUI 可读) */
    float    ema_gpu_util;          /* GPU 利用率 EMA */
    float    ema_gpu_mem;           /* 显存使用 EMA (MB) */
    float    max_gpu_util;          /* 本次运行最大 GPU 利用率 */
    float    max_gpu_mem;           /* 本次运行最大显存使用 (MB) */
    uint32_t ring_buffer_offset;    /* ring buffer 在映射中的字节偏移 */
    uint32_t _reserved[7];
} GpuTelemetryHeader;

/* 共享内存完整布局:
   [GpuTelemetryHeader]
   [GpuDeviceInfo]
   [GpuTelemetrySample[GPU_TELEM_RING_SIZE]]
*/
#define GPU_TELEM_TOTAL_SIZE \
    (sizeof(GpuTelemetryHeader) + sizeof(GpuDeviceInfo) + \
     sizeof(GpuTelemetrySample) * GPU_TELEM_RING_SIZE)

/* 设备信息在映射中的偏移 */
#define GPU_TELEM_DEVICE_INFO_OFFSET (sizeof(GpuTelemetryHeader))

/* Ring buffer 在映射中的偏移 */
#define GPU_TELEM_RING_OFFSET (sizeof(GpuTelemetryHeader) + sizeof(GpuDeviceInfo))

/* ========== 辅助函数 ========== */

/* 获取 ring buffer 中第 i 个 sample 的指针 */
inline GpuTelemetrySample* gpu_telem_get_sample(void* base, uint32_t i) {
    return (GpuTelemetrySample*)((char*)base + GPU_TELEM_RING_OFFSET +
                                  sizeof(GpuTelemetrySample) * (i % GPU_TELEM_RING_SIZE));
}

/* 获取设备信息指针 */
inline GpuDeviceInfo* gpu_telem_get_device_info(void* base) {
    return (GpuDeviceInfo*)((char*)base + GPU_TELEM_DEVICE_INFO_OFFSET);
}

#ifdef __cplusplus
}
#endif

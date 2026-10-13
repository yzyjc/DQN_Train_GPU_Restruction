/* gpu_telemetry_types.h — GPU 遥测内部类型定义
   GPU_Analyzer 内部使用的数据结构, 不对外暴露。 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 统计结果 ========== */
typedef struct GpuStats {
    float    mean;
    float    median;
    float    p5;
    float    p95;
    float    p99;
    float    stddev;
    float    min;
    float    max;
    uint32_t count;
} GpuStats;

/* ========== EMA 跟踪器 ========== */
typedef struct EmaTracker {
    float value;
    float alpha;
    int   initialized;
} EmaTracker;

inline void ema_init(EmaTracker* ema, float alpha) {
    ema->value = 0.0f;
    ema->alpha = alpha;
    ema->initialized = 0;
}

inline void ema_update(EmaTracker* ema, float new_value) {
    if (!ema->initialized) {
        ema->value = new_value;
        ema->initialized = 1;
    } else {
        ema->value = ema->alpha * new_value + (1.0f - ema->alpha) * ema->value;
    }
}

/* ========== 采样模式 ========== */
typedef enum GpuSampleMode {
    GPU_SAMPLE_NORMAL = 0,
    GPU_SAMPLE_TRACE,
    GPU_SAMPLE_DEEP_PROFILE
} GpuSampleMode;

/* ========== 采样上下文 (GPU_Analyzer 内部状态) ========== */
#define GPU_SAMPLER_MAX_SAMPLES  65536  /* 内存中保留的最大采样数 (用于统计) */

typedef struct GpuSamplerContext {
    /* 采样配置 */
    uint32_t       interval_ms;
    GpuSampleMode  mode;
    volatile long  running;         /* 0=停止, 1=运行 */

    void* thread_handle; // GPT6-Astra: join before releasing sampler resources.

    /* PDH 句柄 */
    void*          pdh_query;       /* PDH_HQUERY */
    void*          pdh_gpu_util;    /* PDH_HCOUNTER */
    void*          pdh_gpu_mem;     /* PDH_HCOUNTER */
    void*          pdh_gpu_mem_total; /* PDH_HCOUNTER */
    int            pdh_valid;

    /* HIP 设备 */
    int            hip_device;
    int            hip_valid;

    /* 统计用采样缓冲 */
    float*         gpu_util_history;
    float*         gpu_mem_history;
    uint32_t       history_count;
    uint32_t       history_capacity;

    /* EMA */
    EmaTracker     ema_gpu_util;
    EmaTracker     ema_gpu_mem;

    /* 峰值 */
    float          peak_gpu_util;
    float          peak_gpu_mem;
} GpuSamplerContext;

/* ========== Kernel 计时类型 (Phase 2) ========== */
#ifndef GPU_KERNEL_TYPE_DEFINED
#define GPU_KERNEL_TYPE_DEFINED
typedef enum GpuKernelType {
    GPU_KT_FORWARD = 0,
    GPU_KT_FORWARD_BATCH,
    GPU_KT_TRAIN_STEP,
    GPU_KT_TRAIN_STEP_BATCH,
    GPU_KT_SOFT_UPDATE,
    GPU_KT_UPLOAD,
    GPU_KT_DOWNLOAD,
    GPU_KT_PHASE_MARKER,
    GPU_KT_COUNT
} GpuKernelType;
#endif

/* kernel 类型名称 (用于报告) */
inline const char* gpu_kernel_type_name(GpuKernelType t) {
    static const char* names[] = {
        "forward", "forward_batch", "train_step", "train_step_batch",
        "soft_update", "upload", "download", "phase_marker"
    };
    if (t >= 0 && t < GPU_KT_COUNT) return names[t];
    return "unknown";
}

/* ========== Benchmark Session 元数据 (Phase 3) ========== */
typedef struct BenchmarkSessionHeader {
    uint32_t magic;             /* 0x42454E43 "BENC" */
    uint32_t version;
    uint64_t session_start_ms;  /* GetTickCount64() */
    uint64_t session_end_ms;
    uint32_t sample_count;      /* telemetry 采样数 */
    uint32_t kernel_record_count;
    uint32_t mode;              /* GpuSampleMode */
    uint32_t _pad;
    /* 设备信息快照 */
    char     device_name[64];
    char     arch_name[32];
    uint64_t total_vram_bytes;
    uint32_t compute_units;
    uint32_t clock_mhz;
    uint32_t memory_clock_mhz;
    uint32_t memory_bus_width;
    uint32_t l2_cache_size;
    uint32_t _reserved[7];
} BenchmarkSessionHeader;

#define BENCHMARK_SESSION_MAGIC   0x42454E43
#define BENCHMARK_SESSION_VERSION 1

#ifdef __cplusplus
}
#endif
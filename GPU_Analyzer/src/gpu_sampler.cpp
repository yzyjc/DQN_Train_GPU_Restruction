/* gpu_sampler.cpp — GPU 遥测采样引擎
   PDH 计数器 + hipMemGetInfo + GetSystemTimes, 后台线程采样。
   参考 TUI 的 init_pdh() / gpu_sample_now() / telemetry_thread()。 */

#include <windows.h>  /* 必须在 pdh.h 之前 */
#include "gpu_telemetry_protocol.h"
#include "gpu_telemetry_types.h"
#include "gpu_sensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <process.h>  /* _beginthreadex */
#include <math.h>
#include <algorithm>
#include <pdh.h>
#include <pdhmsg.h>
#pragma comment(lib, "pdh.lib")

#include <hip/hip_runtime_api.h>  /* host-side API only, cl.exe 可编 */

/* ========== EMA 辅助 (已在 gpu_telemetry_types.h 中 inline 定义) ========== */

/* ========== PDH 初始化 ========== */

static int init_pdh(GpuSamplerContext* ctx) {
    if (ctx->pdh_valid) return 1;

    PDH_STATUS status = PdhOpenQueryA(NULL, 0, (PDH_HQUERY*)&ctx->pdh_query);
    if (status != ERROR_SUCCESS) {
        printf("[Sampler] PdhOpenQuery 失败: 0x%08lX\n", (unsigned long)status);
        return 0;
    }

    /* GPU Engine utilization (取最大引擎值, 不求和) */
    status = PdhAddEnglishCounterA((PDH_HQUERY)ctx->pdh_query,
        "\\GPU Engine(*)\\Utilization Percentage", 0,
        (PDH_HCOUNTER*)&ctx->pdh_gpu_util);
    if (status != ERROR_SUCCESS) {
        printf("[Sampler] PdhAddCounter GPU Engine 失败: 0x%08lX\n", (unsigned long)status);
    }

    /* GPU Adapter Memory - Dedicated Usage */
    status = PdhAddEnglishCounterA((PDH_HQUERY)ctx->pdh_query,
        "\\GPU Adapter Memory(*)\\Dedicated Usage", 0,
        (PDH_HCOUNTER*)&ctx->pdh_gpu_mem);
    if (status != ERROR_SUCCESS) {
        printf("[Sampler] PdhAddCounter GPU Memory 失败: 0x%08lX\n", (unsigned long)status);
    }

    /* GPU Adapter Memory - Total Dedicated Usage (可能在某些系统上不可用) */
    status = PdhAddEnglishCounterA((PDH_HQUERY)ctx->pdh_query,
        "\\GPU Adapter Memory(*)\\Total Dedicated Usage", 0,
        (PDH_HCOUNTER*)&ctx->pdh_gpu_mem_total);
    if (status != ERROR_SUCCESS) {
        printf("[Sampler] PdhAddCounter GPU Memory Total 不可用: 0x%08lX (将使用 HIP hipMemGetInfo)\n",
               (unsigned long)status);
        ctx->pdh_gpu_mem_total = NULL;
    }

    /* 首次采集 (初始化差值基准) */
    PdhCollectQueryData((PDH_HQUERY)ctx->pdh_query);
    Sleep(250);
    PdhCollectQueryData((PDH_HQUERY)ctx->pdh_query);

    ctx->pdh_valid = 1;
    printf("[Sampler] PDH 初始化成功\n");
    return 1;
}

/* ========== PDH 辅助函数 (参考 TUI 的 pdh_max_counter / pdh_sum_counter) ========== */

/* 遍历所有实例, 取最大值 (用于 GPU Engine utilization) */
static double pdh_max_counter(PDH_HCOUNTER counter) {
    DWORD size = 0, count = 0;
    PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, NULL);
    if (size == 0) return 0;
    PDH_FMT_COUNTERVALUE_ITEM_A* items = (PDH_FMT_COUNTERVALUE_ITEM_A*)malloc(size);
    if (!items) return 0;
    double mx = 0;
    if (PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < count; i++)
            if (items[i].FmtValue.doubleValue > mx) mx = items[i].FmtValue.doubleValue;
    }
    free(items);
    return mx;
}

/* 遍历所有实例, 求和 (用于 GPU Memory) */
static double pdh_sum_counter(PDH_HCOUNTER counter) {
    DWORD size = 0, count = 0;
    PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, NULL);
    if (size == 0) return 0;
    PDH_FMT_COUNTERVALUE_ITEM_A* items = (PDH_FMT_COUNTERVALUE_ITEM_A*)malloc(size);
    if (!items) return 0;
    double sum = 0;
    if (PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < count; i++) sum += items[i].FmtValue.doubleValue;
    }
    free(items);
    return sum;
}

/* ========== 单次采样 ========== */

/* GPU Engine 利用率需要两次 PdhCollectQueryData + 中间等待才能算出差值。
   参考 TUI 的 gpu_sample_now(): collect → Sleep(250) → collect → read。 */
#define PDH_SAMPLE_GAP_MS 250

static void sample_once(GpuSamplerContext* ctx, GpuTelemetrySample* sample, void* sensors) {
    memset(sample, 0, sizeof(GpuTelemetrySample));
    sample->timestamp_ms = GetTickCount64();

    /* PDH: 两次采集取差值 */
    if (ctx->pdh_valid) {
        PdhCollectQueryData((PDH_HQUERY)ctx->pdh_query);
        Sleep(PDH_SAMPLE_GAP_MS);
        PdhCollectQueryData((PDH_HQUERY)ctx->pdh_query);
        sample->gpu_util_pct = (float)pdh_max_counter((PDH_HCOUNTER)ctx->pdh_gpu_util);

        /* 显存 */
        double mem_bytes = pdh_sum_counter((PDH_HCOUNTER)ctx->pdh_gpu_mem);
        sample->gpu_mem_used_mb = (float)(mem_bytes / (1024.0 * 1024.0));
        if (ctx->pdh_gpu_mem_total) {
            double mem_total_bytes = pdh_sum_counter((PDH_HCOUNTER)ctx->pdh_gpu_mem_total);
            sample->gpu_mem_total_mb = (float)(mem_total_bytes / (1024.0 * 1024.0));
        }
    }

    /* HIP: hipMemGetInfo (更准确的显存数据) */
    if (ctx->hip_valid) {
        size_t free_mem = 0, total_mem = 0;
        if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
            sample->gpu_mem_total_mb = (float)(total_mem / (1024.0 * 1024.0));
            sample->gpu_mem_used_mb = (float)((total_mem - free_mem) / (1024.0 * 1024.0));
        }
    }

    /* CPU 使用率: GetSystemTimes (首次调用建立基线, 后续计算差值) */
    {
        static ULONGLONG prev_idle = 0, prev_kernel = 0, prev_user = 0;
        static int cpu_inited = 0;
        FILETIME idle_time, kernel_time, user_time;
        if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
            ULONGLONG idle = ((ULONGLONG)idle_time.dwHighDateTime << 32) | idle_time.dwLowDateTime;
            ULONGLONG kernel = ((ULONGLONG)kernel_time.dwHighDateTime << 32) | kernel_time.dwLowDateTime;
            ULONGLONG user = ((ULONGLONG)user_time.dwHighDateTime << 32) | user_time.dwLowDateTime;

            if (!cpu_inited) {
                cpu_inited = 1;
                prev_idle = idle; prev_kernel = kernel; prev_user = user;
            } else {
                ULONGLONG idle_delta = idle - prev_idle;
                ULONGLONG kernel_delta = kernel - prev_kernel;
                ULONGLONG user_delta = user - prev_user;
                ULONGLONG total = kernel_delta + user_delta;
                if (total > 0) {
                    sample->cpu_usage_pct = (float)(100.0 * (1.0 - (double)idle_delta / (double)total));
                }
                prev_idle = idle;
                prev_kernel = kernel;
                prev_user = user;
            }
        }
    }

    /* 系统内存 */
    {
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            sample->sys_mem_total_gb = (float)(mem_status.ullTotalPhys / (1024.0 * 1024.0 * 1024.0));
            sample->sys_mem_used_gb = (float)((mem_status.ullTotalPhys - mem_status.ullAvailPhys) /
                                               (1024.0 * 1024.0 * 1024.0));
        }
    }

    gpu_sensors_sample(sensors, sample);
}

/* ========== 采样线程 ========== */

/* 共享内存写入回调 (由 main 设置) */
static void (*g_sample_write_cb)(const GpuTelemetrySample*) = NULL;

void gpu_sampler_set_write_callback(void (*cb)(const GpuTelemetrySample*)) {
    g_sample_write_cb = cb;
}

static unsigned __stdcall sampler_thread(void* arg) {
    GpuSamplerContext* ctx = (GpuSamplerContext*)arg;
    if (!ctx) return 0;

    printf("[Sampler] 采样线程启动 (间隔 %u ms, 模式 %d)\n",
           ctx->interval_ms, (int)ctx->mode);

    char device_name[256]={};
    hipDeviceProp_t prop={};
    if(ctx->hip_valid && hipSetDevice(ctx->hip_device)==hipSuccess &&
       hipGetDeviceProperties(&prop,ctx->hip_device)==hipSuccess)
        strncpy_s(device_name,prop.name,_TRUNCATE);
    void* sensors=gpu_sensors_create(device_name);
    while (InterlockedCompareExchange(&ctx->running, 0, 0)) {
        GpuTelemetrySample sample;
        sample_once(ctx, &sample, sensors);

        /* 更新统计 */
        if (ctx->history_count < ctx->history_capacity) {
            ctx->gpu_util_history[ctx->history_count] = sample.gpu_util_pct;
            ctx->gpu_mem_history[ctx->history_count] = sample.gpu_mem_used_mb;
        }
        ctx->history_count++;

        /* 更新 EMA */
        ema_update(&ctx->ema_gpu_util, sample.gpu_util_pct);
        ema_update(&ctx->ema_gpu_mem, sample.gpu_mem_used_mb);

        /* 更新峰值 */
        if (sample.gpu_util_pct > ctx->peak_gpu_util)
            ctx->peak_gpu_util = sample.gpu_util_pct;
        if (sample.gpu_mem_used_mb > ctx->peak_gpu_mem)
            ctx->peak_gpu_mem = sample.gpu_mem_used_mb;

        /* 写入共享内存 (真实采样数据) */
        if (g_sample_write_cb) g_sample_write_cb(&sample);

        /* 输出到 stdout (每 10 个采样打印一次, 避免刷屏) */
        if (ctx->history_count % 10 == 1) {
            printf("[Sampler] #%u GPU: %.1f%% (EMA %.1f%%) VRAM: %.0f/%.0f MB CPU: %.1f%%\n",
                   ctx->history_count, sample.gpu_util_pct, ctx->ema_gpu_util.value,
                   sample.gpu_mem_used_mb, sample.gpu_mem_total_mb,
                   sample.cpu_usage_pct);
        }

        /* 减去 sample_once 内部的 PDH_SAMPLE_GAP_MS, 保持总周期 ≈ interval_ms */
        uint32_t remaining = ctx->interval_ms > PDH_SAMPLE_GAP_MS ?
                             ctx->interval_ms - PDH_SAMPLE_GAP_MS : 0;
        Sleep(remaining);
    }

    gpu_sensors_destroy(sensors);
    printf("[Sampler] 采样线程退出\n");
    return 0;
}

/* ========== 公共 API ========== */

GpuSamplerContext* gpu_sampler_create(uint32_t interval_ms, GpuSampleMode mode) {
    GpuSamplerContext* ctx = (GpuSamplerContext*)calloc(1, sizeof(GpuSamplerContext));
    if (!ctx) return NULL;

    ctx->interval_ms = interval_ms;
    ctx->mode = mode;
    ctx->running = 0;

    /* 分配历史缓冲 */
    ctx->history_capacity = GPU_SAMPLER_MAX_SAMPLES;
    ctx->gpu_util_history = (float*)calloc(ctx->history_capacity, sizeof(float));
    ctx->gpu_mem_history = (float*)calloc(ctx->history_capacity, sizeof(float));
    if (!ctx->gpu_util_history || !ctx->gpu_mem_history) {
        free(ctx->gpu_util_history);
        free(ctx->gpu_mem_history);
        free(ctx);
        return NULL;
    }

    /* 初始化 EMA */
    ema_init(&ctx->ema_gpu_util, 0.1f);
    ema_init(&ctx->ema_gpu_mem, 0.1f);

    /* 初始化 PDH */
    init_pdh(ctx);

    /* 初始化 HIP */
    if (hipInit(0) == hipSuccess) {
        int count = 0;
        if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
            if (hipSetDevice(0) == hipSuccess) {
                ctx->hip_valid = 1;
                ctx->hip_device = 0;
                printf("[Sampler] HIP 设备 0 已初始化\n");
            }
        }
    }
    if (!ctx->hip_valid) {
        printf("[Sampler] HIP 不可用, 显存数据将使用 PDH\n");
    }

    return ctx;
}

int gpu_sampler_start(GpuSamplerContext* ctx) {
    if (!ctx || ctx->running) return -1;
    ctx->running = 1;

    HANDLE hThread = (HANDLE)_beginthreadex(NULL, 0, sampler_thread, ctx, 0, NULL);
    if (!hThread) {
        ctx->running = 0;
        printf("[Sampler] 创建采样线程失败\n");
        return -1;
    }
    ctx->thread_handle = hThread;
    return 0;
}

void gpu_sampler_stop(GpuSamplerContext* ctx) {
    if (!ctx) return;
    InterlockedExchange(&ctx->running, 0);
    if (ctx->thread_handle) {
        WaitForSingleObject((HANDLE)ctx->thread_handle, INFINITE);
        CloseHandle((HANDLE)ctx->thread_handle);
        ctx->thread_handle = NULL;
    }
}

void gpu_sampler_destroy(GpuSamplerContext* ctx) {
    if (!ctx) return;
    gpu_sampler_stop(ctx);

    if (ctx->pdh_query) {
        PdhCloseQuery((PDH_HQUERY)ctx->pdh_query);
    }
    if (ctx->hip_valid) {
        hipDeviceReset();
    }

    free(ctx->gpu_util_history);
    free(ctx->gpu_mem_history);
    free(ctx);
}

/* 获取当前统计 */
GpuStats gpu_sampler_get_util_stats(const GpuSamplerContext* ctx) {
    GpuStats stats = {};
    if (!ctx || ctx->history_count == 0) return stats;

    /* 拷贝并排序 */
    uint32_t n = ctx->history_count < ctx->history_capacity ?
                 ctx->history_count : ctx->history_capacity;
    float* sorted = (float*)malloc(n * sizeof(float));
    if (!sorted) return stats;
    memcpy(sorted, ctx->gpu_util_history, n * sizeof(float));

    // GPT6-Astra: O(n log n), replacing quadratic shutdown sorting.
    std::sort(sorted, sorted + n);

    stats.count = n;
    stats.min = sorted[0];
    stats.max = sorted[n - 1];
    stats.median = sorted[n / 2];
    stats.p5 = sorted[(uint32_t)(n * 0.05f)];
    stats.p95 = sorted[(uint32_t)(n * 0.95f)];
    stats.p99 = sorted[(uint32_t)(n * 0.99f)];

    /* mean */
    double sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += sorted[i];
    stats.mean = (float)(sum / n);

    /* stddev */
    double var = 0;
    for (uint32_t i = 0; i < n; i++) {
        double diff = sorted[i] - stats.mean;
        var += diff * diff;
    }
    stats.stddev = (float)sqrt(var / n);

    free(sorted);
    return stats;
}
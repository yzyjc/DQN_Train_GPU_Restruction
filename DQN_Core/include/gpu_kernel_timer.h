/* gpu_kernel_timer.h — GPU 内核计时器
   Trainer 在 gpu_nn_* 函数前后插入 timer_begin/timer_end 调用。
   使用 hipEvent 测量 kernel 时间, 数据写入共享内存 ring buffer,
   GPU_Analyzer 读取并计算 duty cycle / 统计分布。

   纯 C 头文件, .hip 和 .cpp 共用。
   失败时静默跳过 (不阻断训练)。 */
#pragma once
#include <stdint.h>

/* HIP 头文件必须在 extern "C" 之前 include (含 C++ 模板) */
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 共享内存协议 ========== */
#define GPU_KERNEL_TIMER_MAGIC      0x4B54494D  /* "KTIM" */
#define GPU_KERNEL_TIMER_VERSION    1
#define GPU_KERNEL_TIMER_NAME       "DQN_GPU_KernelTimer"
#define GPU_KERNEL_TIMER_RING_SIZE  16384

/* ========== 内核类型 ========== */
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

/* ========== 一条内核计时记录 (24 bytes) ========== */
typedef struct GpuKernelRecord {
    uint64_t timestamp_us;     /* QPC 微秒 */
    uint32_t kernel_type;      /* GpuKernelType */
    uint32_t batch_size;       /* 0 for single */
    float    elapsed_ms;       /* hipEvent ElapsedTime */
    uint32_t phase_episode;    /* 仅 PHASE_MARKER 有效: phase | (episode << 8) */
} GpuKernelRecord;
static_assert(sizeof(GpuKernelRecord) == 24, "GpuKernelRecord must be 24 bytes");

/* ========== 共享内存 Header ========== */
typedef struct GpuKernelTimerHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t total_size;
    volatile uint64_t write_seq;    /* seqlock */
    volatile uint32_t ring_head;
    volatile uint64_t total_records;
    uint64_t publisher_pid;
    uint64_t publisher_heartbeat_ms;
    /* 累计统计 (方便快速读取, 不需要遍历 ring) */
    float    forward_total_ms;
    float    train_total_ms;
    float    soft_update_total_ms;
    float    upload_download_total_ms;
    uint64_t forward_count;
    uint64_t train_count;
    uint64_t _reserved[4];
} GpuKernelTimerHeader;

/* ring buffer 在映射中的偏移 */
#define GPU_KT_RING_OFFSET sizeof(GpuKernelTimerHeader)

/* 总映射大小 */
#define GPU_KT_TOTAL_SIZE \
    (sizeof(GpuKernelTimerHeader) + sizeof(GpuKernelRecord) * GPU_KERNEL_TIMER_RING_SIZE)

/* ========== 辅助: 获取 ring 中第 i 条记录的指针 ========== */
inline GpuKernelRecord* gpu_kt_get_record(void* base, uint32_t i) {
    return (GpuKernelRecord*)((char*)base + GPU_KT_RING_OFFSET +
                               sizeof(GpuKernelRecord) * (i % GPU_KERNEL_TIMER_RING_SIZE));
}

#ifdef __HIP_PLATFORM_AMD__
/* ========== HIP 实现 (.hip 文件中使用) ========== */
#include <windows.h>
#include <stdio.h>

/* 内部状态 (每个进程一份) */
static HANDLE   g_kt_map = NULL;
static void*    g_kt_base = NULL;
static int      g_kt_enabled = 0;

/* hipEvent 对 */
static hipEvent_t g_kt_start = NULL;
static hipEvent_t g_kt_stop = NULL;

/* QPC 频率 */
static double g_kt_qpc_freq_inv = 0.0;

/* 当前正在计时的 kernel 类型 */
static GpuKernelType g_kt_current_type = GPU_KT_FORWARD;
static uint32_t g_kt_current_batch = 0;

/* 初始化: 创建共享内存 + hipEvent */
inline int gpu_timer_init(void) {
    if (g_kt_enabled) return 0;

    /* QPC 频率 */
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) return -1;
    g_kt_qpc_freq_inv = 1000000.0 / (double)freq.QuadPart;

    /* 创建共享内存 */
    g_kt_map = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        0, (DWORD)GPU_KT_TOTAL_SIZE, GPU_KERNEL_TIMER_NAME);
    if (!g_kt_map) {
        printf("[KernelTimer] CreateFileMapping 失败: %lu\n", GetLastError());
        return -1;
    }

    g_kt_base = MapViewOfFile(g_kt_map, FILE_MAP_WRITE, 0, 0, GPU_KT_TOTAL_SIZE);
    if (!g_kt_base) {
        printf("[KernelTimer] MapViewOfFile 失败: %lu\n", GetLastError());
        CloseHandle(g_kt_map);
        g_kt_map = NULL;
        return -1;
    }

    /* 初始化 header */
    GpuKernelTimerHeader* hdr = (GpuKernelTimerHeader*)g_kt_base;
    memset(hdr, 0, sizeof(GpuKernelTimerHeader));
    hdr->magic = GPU_KERNEL_TIMER_MAGIC;
    hdr->version = GPU_KERNEL_TIMER_VERSION;
    hdr->total_size = GPU_KT_TOTAL_SIZE;
    hdr->publisher_pid = GetCurrentProcessId();
    hdr->publisher_heartbeat_ms = GetTickCount64();

    /* 创建 hipEvent */
    if (hipEventCreate(&g_kt_start) != hipSuccess ||
        hipEventCreate(&g_kt_stop) != hipSuccess) {
        printf("[KernelTimer] hipEventCreate 失败, 计时禁用\n");
        UnmapViewOfFile(g_kt_base);
        CloseHandle(g_kt_map);
        g_kt_base = NULL;
        g_kt_map = NULL;
        return -1;
    }

    g_kt_enabled = 1;
    printf("[KernelTimer] 初始化成功 (共享内存 %s, %llu bytes)\n",
           GPU_KERNEL_TIMER_NAME, (unsigned long long)GPU_KT_TOTAL_SIZE);
    return 0;
}

/* 关闭: 释放资源 */
inline void gpu_timer_shutdown(void) {
    if (!g_kt_enabled) return;
    g_kt_enabled = 0;

    if (g_kt_start) { hipEventDestroy(g_kt_start); g_kt_start = NULL; }
    if (g_kt_stop)  { hipEventDestroy(g_kt_stop);  g_kt_stop = NULL; }

    if (g_kt_base) { UnmapViewOfFile(g_kt_base); g_kt_base = NULL; }
    if (g_kt_map)  { CloseHandle(g_kt_map); g_kt_map = NULL; }

    printf("[KernelTimer] 已关闭\n");
}

/* 写入一条记录到 ring buffer (seqlock) */
inline void gpu_kt_write_record(GpuKernelType type, uint32_t batch_size, float elapsed_ms) {
    if (!g_kt_base) return;

    GpuKernelTimerHeader* hdr = (GpuKernelTimerHeader*)g_kt_base;

    /* seqlock 写入 */
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);

    uint32_t head = hdr->ring_head;
    GpuKernelRecord* rec = gpu_kt_get_record(g_kt_base, head);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    rec->timestamp_us = (uint64_t)(now.QuadPart * g_kt_qpc_freq_inv);
    rec->kernel_type = (uint32_t)type;
    rec->batch_size = batch_size;
    rec->elapsed_ms = elapsed_ms;
    rec->phase_episode = 0;

    hdr->ring_head = (head + 1) % GPU_KERNEL_TIMER_RING_SIZE;
    hdr->total_records++;

    /* 更新累计统计 */
    switch (type) {
        case GPU_KT_FORWARD:
        case GPU_KT_FORWARD_BATCH:
            hdr->forward_total_ms += elapsed_ms;
            hdr->forward_count++;
            break;
        case GPU_KT_TRAIN_STEP:
        case GPU_KT_TRAIN_STEP_BATCH:
            hdr->train_total_ms += elapsed_ms;
            hdr->train_count++;
            break;
        case GPU_KT_SOFT_UPDATE:
            hdr->soft_update_total_ms += elapsed_ms;
            break;
        case GPU_KT_UPLOAD:
        case GPU_KT_DOWNLOAD:
            hdr->upload_download_total_ms += elapsed_ms;
            break;
        default: break;
    }

    MemoryBarrier();
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);

    hdr->publisher_heartbeat_ms = GetTickCount64();
}

/* 开始计时 */
inline void gpu_timer_begin(GpuKernelType type, uint32_t batch_size) {
    if (!g_kt_enabled) return;
    g_kt_current_type = type;
    g_kt_current_batch = batch_size;
    hipEventRecord(g_kt_start, 0);
}

/* 结束计时: 记录 elapsed, 写入 ring buffer */
inline void gpu_timer_end(void) {
    if (!g_kt_enabled) return;
    hipEventRecord(g_kt_stop, 0);
    hipEventSynchronize(g_kt_stop);
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, g_kt_start, g_kt_stop);
    gpu_kt_write_record(g_kt_current_type, g_kt_current_batch, elapsed_ms);
}

/* 阶段标记: 记录当前 episode 和 phase */
inline void gpu_timer_mark_phase(uint32_t phase, uint64_t episode) {
    if (!g_kt_base) return;
    /* 复用 write_record, 但把 phase|episode 编码到 phase_episode 字段 */
    GpuKernelTimerHeader* hdr = (GpuKernelTimerHeader*)g_kt_base;
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);

    uint32_t head = hdr->ring_head;
    GpuKernelRecord* rec = gpu_kt_get_record(g_kt_base, head);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    rec->timestamp_us = (uint64_t)(now.QuadPart * g_kt_qpc_freq_inv);
    rec->kernel_type = GPU_KT_PHASE_MARKER;
    rec->batch_size = 0;
    rec->elapsed_ms = 0;
    rec->phase_episode = (uint32_t)(phase | ((uint32_t)(episode & 0xFFFFFF) << 8));

    hdr->ring_head = (head + 1) % GPU_KERNEL_TIMER_RING_SIZE;
    hdr->total_records++;

    MemoryBarrier();
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);
    hdr->publisher_heartbeat_ms = GetTickCount64();
}

#else
/* 非 HIP 编译环境 (.cpp): 提供 no-op 存根 */
inline int  gpu_timer_init(void) { return 0; }
inline void gpu_timer_shutdown(void) {}
inline void gpu_timer_begin(GpuKernelType type, uint32_t batch_size) { (void)type; (void)batch_size; }
inline void gpu_timer_end(void) {}
inline void gpu_timer_mark_phase(uint32_t phase, uint64_t episode) { (void)phase; (void)episode; }
#endif /* __HIP_PLATFORM_AMD__ */

#ifdef __cplusplus
}
#endif
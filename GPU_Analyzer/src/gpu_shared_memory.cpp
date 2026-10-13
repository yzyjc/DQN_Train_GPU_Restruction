/* gpu_shared_memory.cpp — GPU 遥测共享内存发布
   使用 seqlock 协议写入, 参考 neural_probe_publisher.cpp 的模式。
   GPU_Analyzer 是 publisher, TUI 是 client。 */

#include "gpu_telemetry_protocol.h"
#include <stdio.h>
#include <string.h>
#include <windows.h>

/* ========== 内部状态 ========== */

static HANDLE   g_telem_map = NULL;
static void*    g_telem_base = NULL;
static uint32_t g_telem_seq = 0;  /* 本地序号 */

/* ========== 公共 API ========== */

int gpu_shared_memory_init(void) {
    if (g_telem_base) return 0;  /* 已初始化 */

    /* 创建共享内存映射 */
    g_telem_map = CreateFileMappingA(
        INVALID_HANDLE_VALUE,    /* 匿名映射 (pagefile-backed) */
        NULL,                    /* 默认安全属性 */
        PAGE_READWRITE,
        0,                       /* 高位大小 */
        (DWORD)GPU_TELEM_TOTAL_SIZE,  /* 低位大小 */
        GPU_TELEM_NAME           /* 映射名称 */
    );

    if (!g_telem_map) {
        printf("[SharedMemory] CreateFileMapping '%s' 失败: %lu\n",
               GPU_TELEM_NAME, GetLastError());
        return -1;
    }

    g_telem_base = MapViewOfFile(g_telem_map, FILE_MAP_WRITE, 0, 0, GPU_TELEM_TOTAL_SIZE);
    if (!g_telem_base) {
        printf("[SharedMemory] MapViewOfFile 失败: %lu\n", GetLastError());
        CloseHandle(g_telem_map);
        g_telem_map = NULL;
        return -1;
    }

    /* 初始化 header */
    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)g_telem_base;
    memset(hdr, 0, sizeof(GpuTelemetryHeader));
    hdr->magic = GPU_TELEM_MAGIC;
    hdr->version = GPU_TELEM_VERSION;
    hdr->total_size = GPU_TELEM_TOTAL_SIZE;
    hdr->publisher_pid = GetCurrentProcessId();
    hdr->publisher_heartbeat_ms = GetTickCount64();
    hdr->write_seq = 0;
    hdr->ring_head = 0;
    hdr->sample_count = 0;

    /* 设置 ring buffer 偏移 (供 TUI 直接读取, 不依赖 struct 大小计算) */
    hdr->ring_buffer_offset = (uint32_t)GPU_TELEM_RING_OFFSET;

    /* 清零设备信息和 ring buffer */
    memset((char*)g_telem_base + GPU_TELEM_DEVICE_INFO_OFFSET, 0, sizeof(GpuDeviceInfo));
    memset((char*)g_telem_base + GPU_TELEM_RING_OFFSET, 0,
           sizeof(GpuTelemetrySample) * GPU_TELEM_RING_SIZE);

    printf("[SharedMemory] '%s' 创建成功 (%llu bytes)\n",
           GPU_TELEM_NAME, (unsigned long long)GPU_TELEM_TOTAL_SIZE);
    return 0;
}

void gpu_shared_memory_shutdown(void) {
    if (g_telem_base) {
        UnmapViewOfFile(g_telem_base);
        g_telem_base = NULL;
    }
    if (g_telem_map) {
        CloseHandle(g_telem_map);
        g_telem_map = NULL;
    }
    printf("[SharedMemory] 已关闭\n");
}

/* 写入设备信息 (只在启动时调用一次) */
void gpu_shared_memory_write_device_info(const GpuDeviceInfo* info) {
    if (!g_telem_base || !info) return;
    GpuDeviceInfo* dst = gpu_telem_get_device_info(g_telem_base);
    memcpy(dst, info, sizeof(GpuDeviceInfo));
}

/* 设置采样模式和间隔 */
void gpu_shared_memory_set_mode(uint32_t mode, uint32_t interval_ms) {
    if (!g_telem_base) return;
    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)g_telem_base;
    hdr->sample_mode = mode;
    hdr->sample_interval_ms = interval_ms;
}

/* ========== seqlock 写入一个采样 ========== */

void gpu_shared_memory_write_sample(const GpuTelemetrySample* sample) {
    if (!g_telem_base || !sample) return;

    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)g_telem_base;

    /* seqlock 写入:
       1. write_seq++ (奇数 = 写入中)
       2. 写入数据
       3. MemoryBarrier
       4. write_seq++ (偶数 = 写入完成) */

    /* 1. 奇数 = 写入中 */
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);

    /* 2. 写入 ring buffer */
    uint32_t head = hdr->ring_head;
    GpuTelemetrySample* dst = gpu_telem_get_sample(g_telem_base, head);

    GpuTelemetrySample copy = *sample;
    copy.sequence = ++g_telem_seq;
    *dst = copy;

    /* 3. 更新 ring_head 和 sample_count */
    hdr->ring_head = (head + 1) % GPU_TELEM_RING_SIZE;
    hdr->sample_count++;

    /* 更新累计统计 */
    hdr->ema_gpu_util = copy.gpu_util_pct;  /* EMA 由 sampler 维护, 这里直接写 */
    hdr->ema_gpu_mem = copy.gpu_mem_used_mb;
    if (copy.gpu_util_pct > hdr->max_gpu_util)
        hdr->max_gpu_util = copy.gpu_util_pct;
    if (copy.gpu_mem_used_mb > hdr->max_gpu_mem)
        hdr->max_gpu_mem = copy.gpu_mem_used_mb;

    /* 4. MemoryBarrier + 偶数 = 写入完成 */
    MemoryBarrier();
    InterlockedIncrement64((volatile LONG64*)&hdr->write_seq);

    /* 更新心跳 */
    hdr->publisher_heartbeat_ms = GetTickCount64();
}

/* ========== 客户端 API (TUI 读取端) ========== */

/* 打开已存在的共享内存 (只读) */
void* gpu_shared_memory_open_readonly(void) {
    HANDLE map = OpenFileMappingA(FILE_MAP_READ, FALSE, GPU_TELEM_NAME);
    if (!map) return NULL;

    void* base = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(map);
        return NULL;
    }

    /* 验证 header */
    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)base;
    if (hdr->magic != GPU_TELEM_MAGIC || hdr->version != GPU_TELEM_VERSION) {
        UnmapViewOfFile(base);
        CloseHandle(map);
        return NULL;
    }

    /* 注意: 调用者需要保存 map 句柄以便关闭。
       简化实现: 返回 base, 调用者通过 gpu_shared_memory_close_readonly 关闭。 */
    return base;
}

void gpu_shared_memory_close_readonly(void* base) {
    if (base) {
        UnmapViewOfFile(base);
    }
}

/* seqlock 读取最新 N 个采样 */
int gpu_shared_memory_read_samples(void* base, GpuTelemetrySample* out,
                                    uint32_t max_count, uint32_t* out_total) {
    if (!base || !out || max_count == 0) return -1;

    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)base;

    /* seqlock 读取: 重试直到 write_seq 前后一致 */
    uint64_t seq_before, seq_after;
    int retries = 0;
    do {
        seq_before = hdr->write_seq;
        if (seq_before & 1) {
            /* 奇数 = 写入中, 重试 */
            retries++;
            if (retries > 100) return -1;  /* 防止死循环 */
            YieldProcessor();
            continue;
        }

        /* 读取数据 */
        uint32_t head = hdr->ring_head;
        uint32_t total = hdr->sample_count;
        uint32_t count = total < max_count ? total : max_count;

        if (out_total) *out_total = total;

        for (uint32_t i = 0; i < count; i++) {
            uint32_t idx = (head + GPU_TELEM_RING_SIZE - count + i) % GPU_TELEM_RING_SIZE;
            GpuTelemetrySample* src = gpu_telem_get_sample(base, idx);
            out[i] = *src;
        }

        MemoryBarrier();
        seq_after = hdr->write_seq;
        retries++;
    } while (seq_before != seq_after && retries < 100);

    if (seq_before != seq_after) return -1;  /* 读取失败 */

    return (int)(hdr->sample_count < max_count ? hdr->sample_count : max_count);
}

/* 检查 publisher 是否存活 */
int gpu_shared_memory_is_publisher_alive(void* base, uint32_t timeout_ms) {
    if (!base) return 0;
    GpuTelemetryHeader* hdr = (GpuTelemetryHeader*)base;
    if (hdr->magic != GPU_TELEM_MAGIC) return 0;

    uint64_t now = GetTickCount64();
    uint64_t heartbeat = hdr->publisher_heartbeat_ms;
    if (heartbeat == 0) return 0;

    return (now - heartbeat) < timeout_ms;
}
/* gpu_analyzer_main.cpp — GPU 性能监控器入口
   独立进程, 采集 GPU 遥测数据并发布到共享内存。
   不依赖 Trainer, 不修改 Trainer。

   用法:
     GPU_Analyzer.exe                    -- 默认 NORMAL 模式, 1Hz
     GPU_Analyzer.exe --mode trace       -- TRACE 模式, 10Hz
     GPU_Analyzer.exe --mode deep        -- DEEP_PROFILE, 10Hz + ADLX
     GPU_Analyzer.exe --report           -- 生成报告 (Phase 3)
     GPU_Analyzer.exe --duration 3600    -- 运行指定秒数后退出 */

#include "gpu_telemetry_protocol.h"
#include "gpu_telemetry_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <signal.h>
#include <atomic>
#include "analyzer_window.h"

/* ========== 前向声明 (各模块的公共 API) ========== */

/* gpu_detector.cpp */
extern int  gpu_detector_init(GpuDeviceInfo* info);
extern void gpu_detector_print(const GpuDeviceInfo* info);

/* gpu_sampler.cpp */
extern GpuSamplerContext* gpu_sampler_create(uint32_t interval_ms, GpuSampleMode mode);
extern int  gpu_sampler_start(GpuSamplerContext* ctx);
extern void gpu_sampler_stop(GpuSamplerContext* ctx);
extern void gpu_sampler_destroy(GpuSamplerContext* ctx);
extern GpuStats gpu_sampler_get_util_stats(const GpuSamplerContext* ctx);
extern void gpu_sampler_set_write_callback(void (*cb)(const GpuTelemetrySample*));

/* gpu_shared_memory.cpp */
extern int  gpu_shared_memory_init(void);
extern void gpu_shared_memory_shutdown(void);
extern void gpu_shared_memory_write_device_info(const GpuDeviceInfo* info);
extern void gpu_shared_memory_set_mode(uint32_t mode, uint32_t interval_ms);
extern void gpu_shared_memory_write_sample(const GpuTelemetrySample* sample);

/* gpu_binary_log.cpp */
extern int  gpu_binary_log_open(const char* log_dir, uint32_t field_count);
extern void gpu_binary_log_close(void);
extern int  gpu_binary_log_write(int32_t episode, int32_t step,
                                  const float* fields, uint32_t field_count);

/* gpu_stats.cpp */
extern GpuStats gpu_stats_compute(const float* data, uint32_t count);
extern void     gpu_stats_print(const char* label, const GpuStats* stats, const char* unit);

/* gpu_sampler_kernel.cpp */
extern int  gpu_sampler_kernel_open(void);
extern void gpu_sampler_kernel_close(void);
extern int  gpu_sampler_kernel_is_connected(void);
extern void gpu_sampler_kernel_print_summary(void);

/* gpu_report.cpp */
extern int  gpu_report_generate_json(const char* log_path, const GpuDeviceInfo* device_info,
                                      const char* output_path);
extern int  gpu_report_generate_csv(const char* log_path, const char* output_path);
extern int  gpu_report_compare(const char* report_a_path, const char* report_b_path,
                                const char* output_path);
extern void gpu_report_print_help(void);

/* ========== 全局状态 ========== */

static std::atomic<bool> g_running{true};
void analyzer_request_stop() { g_running.store(false); }
static GpuSamplerContext* g_sampler = NULL;

/* ========== 信号处理 ========== */

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    printf("\n[Main] 收到退出信号, 正在停止...\n");
}

/* ========== 命令行解析 ========== */

typedef struct {
    GpuSampleMode mode;
    uint32_t      duration_seconds;  /* 0 = 无限运行 */
    int           enable_log;        /* 是否写二进制日志 */
    const char*   log_dir;
    int           report_mode;       /* --report */
    const char*   report_log_path;   /* --report <path> */
    const char*   report_output;     /* --output <path> */
    int           csv_mode;          /* --csv */
    const char*   csv_log_path;      /* --csv <path> */
    int           compare_mode;      /* --compare */
    const char*   compare_a;         /* --compare <a> <b> */
    const char*   compare_b;
} ProgramConfig;

static void parse_args(int argc, char* argv[], ProgramConfig* cfg) {
    cfg->mode = GPU_SAMPLE_NORMAL;
    cfg->duration_seconds = 0;
    cfg->enable_log = 1;
    cfg->log_dir = "logs";
    cfg->report_mode = 0;
    cfg->report_log_path = NULL;
    cfg->report_output = NULL;
    cfg->csv_mode = 0;
    cfg->csv_log_path = NULL;
    cfg->compare_mode = 0;
    cfg->compare_a = NULL;
    cfg->compare_b = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "trace") == 0) {
                cfg->mode = GPU_SAMPLE_TRACE;
            } else if (strcmp(argv[i], "deep") == 0) {
                cfg->mode = GPU_SAMPLE_DEEP_PROFILE;
            } else {
                cfg->mode = GPU_SAMPLE_NORMAL;
            }
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            cfg->duration_seconds = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-log") == 0) {
            cfg->enable_log = 0;
        } else if (strcmp(argv[i], "--log-dir") == 0 && i + 1 < argc) {
            cfg->log_dir = argv[++i];
        } else if (strcmp(argv[i], "--report") == 0 && i + 1 < argc) {
            cfg->report_mode = 1;
            cfg->report_log_path = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg->csv_mode = 1;
            cfg->csv_log_path = argv[++i];
        } else if (strcmp(argv[i], "--compare") == 0 && i + 2 < argc) {
            cfg->compare_mode = 1;
            cfg->compare_a = argv[++i];
            cfg->compare_b = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg->report_output = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            gpu_report_print_help();
            exit(0);
        }
    }
}

/* ========== 采样循环 ========== */

static int run_sampling_loop(const ProgramConfig* cfg) {
    /* 获取采样间隔 */
    uint32_t interval_ms;
    switch (cfg->mode) {
        case GPU_SAMPLE_TRACE:       interval_ms = GPU_TELEM_INTERVAL_TRACE_MS; break;
        case GPU_SAMPLE_DEEP_PROFILE: interval_ms = GPU_TELEM_INTERVAL_DEEP_MS; break;
        default:                     interval_ms = GPU_TELEM_INTERVAL_NORMAL_MS; break;
    }

    /* 创建采样器 */
    g_sampler = gpu_sampler_create(interval_ms, cfg->mode);
    if (!g_sampler) {
        printf("[Main] 创建采样器失败\n");
        return 1;
    }

    /* 初始化共享内存 */
    if (gpu_shared_memory_init() != 0) {
        printf("[Main] 共享内存初始化失败, 继续运行 (仅日志)\n");
    }

    /* 写入设备信息 */
    GpuDeviceInfo dev_info;
    if (gpu_detector_init(&dev_info) == 0) {
        gpu_detector_print(&dev_info);
        gpu_shared_memory_write_device_info(&dev_info);
    }
    gpu_shared_memory_set_mode(cfg->mode, interval_ms);

    /* 初始化二进制日志 */
    if (cfg->enable_log) {
        /* 11 fields: the original 9 telemetry values plus memory clock and hotspot. */
        gpu_binary_log_open(cfg->log_dir, 11);
    }

    /* 设置采样写入回调 (真实数据直接进共享内存+日志) */
    {
        static int s_enable_log = cfg->enable_log;
        gpu_sampler_set_write_callback([](const GpuTelemetrySample* s) {
            analyzer_window_sample(s);
            gpu_shared_memory_write_sample(s);
            if (s_enable_log) {
                float fields[11] = {
                    s->gpu_util_pct, s->gpu_mem_used_mb, s->gpu_mem_total_mb,
                    s->gpu_temp_c, s->gpu_power_w, s->gpu_clock_mhz,
                    s->cpu_usage_pct, s->sys_mem_used_gb, s->sys_mem_total_gb,
                    s->gpu_memory_clock_mhz, s->gpu_hotspot_temp_c
                };
                static uint32_t log_idx = 0;
                gpu_binary_log_write((int32_t)log_idx++, 0, fields, 11);
            }
        });
    }

    /* 启动采样线程 */
    if (gpu_sampler_start(g_sampler) != 0) {
        printf("[Main] 启动采样线程失败\n");
        gpu_sampler_destroy(g_sampler);
        g_sampler = NULL;
        gpu_binary_log_close();
        return 1;
    }

    /* Phase 2: 尝试连接 kernel timer 共享内存 */
    if (gpu_sampler_kernel_open() == 0) {
        printf("[Main] 已连接 kernel timer (Trainer 已启用)\n");
    } else {
        printf("[Main] kernel timer 不可用 (Trainer 未启用, 跳过)\n");
    }

    /* 主循环: 等待退出信号或超时 */
    printf("[Main] 采样运行中 (模式 %s, 间隔 %u ms)\n",
           cfg->mode == GPU_SAMPLE_NORMAL ? "NORMAL" :
           cfg->mode == GPU_SAMPLE_TRACE ? "TRACE" : "DEEP_PROFILE",
           interval_ms);
    printf("[Main] 使用窗口 Stop 或 TUI gpu stop 停止\n\n");

    uint64_t start_ms = GetTickCount64();

    while (g_running) {
        Sleep(interval_ms);

        /* 检查超时 */
        if (cfg->duration_seconds > 0) {
            uint64_t elapsed = (GetTickCount64() - start_ms) / 1000;
            if (elapsed >= cfg->duration_seconds) {
                printf("\n[Main] 达到运行时限 (%u 秒), 停止\n", cfg->duration_seconds);
                break;
            }
        }
        /* 采样由 sampler 线程的回调自动完成 (共享内存+日志) */
    }

    // Stop and join the writer before reading its statistics.
    gpu_sampler_stop(g_sampler);
    /* 打印最终统计 */
    printf("\n========== 采样统计 ==========\n");
    GpuStats util_stats = gpu_sampler_get_util_stats(g_sampler);
    gpu_stats_print("GPU 利用率", &util_stats, "%");
    printf("总采样数: %u\n", util_stats.count);
    printf("运行时长: %.1f 秒\n", (GetTickCount64() - start_ms) / 1000.0);
    printf("==============================\n");

    /* Phase 2: kernel timing 摘要 */
    gpu_sampler_kernel_print_summary();
    gpu_sampler_kernel_close();

    /* 清理 */
    gpu_sampler_stop(g_sampler);
    gpu_sampler_destroy(g_sampler);
    g_sampler = NULL;

    if (cfg->enable_log) {
        gpu_binary_log_close();
        printf("[Main] 日志已保存: %s\n",
               cfg->log_dir);  /* TODO: 打印完整路径 */
    }
    return 0;
}

/* ========== 主入口 ========== */

int analyzer_main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("  GPU Performance Monitor\n");
    printf("  Windows + AMD ROCm/HIP\n");
    printf("========================================\n\n");

    /* 解析命令行 */
    ProgramConfig cfg;
    parse_args(argc, argv, &cfg);

    /* 注册信号处理 */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (cfg.report_mode) {
        /* Phase 3: JSON 报告生成 */
        GpuDeviceInfo dev_info;
        gpu_detector_init(&dev_info);
        return gpu_report_generate_json(cfg.report_log_path, &dev_info, cfg.report_output);
    }

    if (cfg.csv_mode) {
        /* Phase 3: CSV 时间序列导出 */
        return gpu_report_generate_csv(cfg.csv_log_path, cfg.report_output);
    }

    if (cfg.compare_mode) {
        /* Phase 3: 硬件对比 */
        return gpu_report_compare(cfg.compare_a, cfg.compare_b, cfg.report_output);
    }

    /* 正常采样模式 */
    HANDLE singleton = CreateMutexW(NULL, TRUE, L"Local\\DQN_GPU_Analyzer_Publisher");
    if (!singleton || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (singleton) CloseHandle(singleton);
        return 2;
    }
    int sampling_result = run_sampling_loop(&cfg);
    ReleaseMutex(singleton);
    CloseHandle(singleton);

    /* 清理共享内存 */
    gpu_shared_memory_shutdown();

    printf("[Main] GPU_Analyzer 退出\n");
    return sampling_result;
}

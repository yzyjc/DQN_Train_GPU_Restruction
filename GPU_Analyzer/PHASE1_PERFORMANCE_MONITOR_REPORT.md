# PHASE 1 — GPU Performance Monitor 报告

## 修改的文件

| 文件 | 改动 |
|------|------|
| `GPU_Analyzer/include/gpu_telemetry_protocol.h` | 新增：共享内存协议 (GTEL magic, seqlock, ring buffer 4096×64B) |
| `GPU_Analyzer/include/gpu_telemetry_types.h` | 新增：GpuTelemetrySample, GpuSamplerContext, GpuStats, EmaTracker |
| `GPU_Analyzer/include/gpu_report_format.h` | 新增：CSV header, JSON 结构文档 |
| `GPU_Analyzer/src/gpu_analyzer_main.cpp` | 新增：入口, 命令行, 采样主循环 |
| `GPU_Analyzer/src/gpu_detector.cpp` | 新增：HIP + DXGI 设备探测 |
| `GPU_Analyzer/src/gpu_sampler.cpp` | 新增：PDH + hipMemGetInfo + GetSystemTimes 采样引擎 |
| `GPU_Analyzer/src/gpu_shared_memory.cpp` | 新增：seqlock 共享内存发布 |
| `GPU_Analyzer/src/gpu_binary_log.cpp` | 新增：metric_log_format.h 兼容二进制日志 |
| `GPU_Analyzer/src/gpu_stats.cpp` | 新增：mean/median/P5/P95/P99/stddev |
| `GPU_Analyzer/src/gpu_report.cpp` | 新增：JSON 报告生成 |
| `GPU_Analyzer/src/gpu_sampler_kernel.cpp` | 新增：kernel timer 共享内存读取 (Phase 2) |
| `GPU_Analyzer/GPU_Analyzer.vcxproj` | 新增：MSBuild 工程 |
| `DQN_Core/include/gpu_kernel_timer.h` | 新增：hipEvent 计时器 + DQN_GPU_KernelTimer 共享内存 |
| `DQN_Core/src/gpu_neural_net.hip` | 修改：gpu_nn_* 包裹 timer, backward_update 分段计时 |
| `DQN_Core/src/neural_net.cpp` | 修改：定义 g_bwd_*/g_ts_* 计时变量 |
| `DQN_Core/src/ipc_server.cpp` | 修改：IPC 推送新增 fwd/bwd/tgt/phase/side/replay/bwd_breakdown/ts_overhead |
| `DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp` | 修改：GPU Analyze tab, 自适应双栏布局 |
| `DQN_Main_TUI_No_Terminal/gpu_telemetry_client.h` | 新增：TUI 侧共享内存客户端 |
| `DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp` | 修改：训练分阶段 QPC 计时 |

## 使用的 API

| 指标 | API | 来源 |
|------|-----|------|
| GPU 利用率 | PDH `\\GPU Engine(*)\\Utilization Percentage` | Windows |
| Copy Engine | PDH `\\GPU Engine(*)` 过滤 `engtype_Copy` | Windows |
| VRAM (HIP) | `hipMemGetInfo` | ROCm |
| VRAM (WDDM) | DXGI `DedicatedVideoMemory` / `SharedSystemMemory` | Windows |
| CPU 使用率 | `GetSystemTimes` | Windows |
| CPU Max Core | PDH `\\Processor Information(*)\\% Processor Utility` | Windows |
| 系统内存 | `GlobalMemoryStatusEx` | Windows |
| GPU 温度 | ADL `ADL_Adapter_GetThermalStatus` / `ADL_Overdrive5_Temperature_Get` | AMD |
| GPU 设备属性 | `hipGetDeviceProperties` | ROCm |
| Kernel 计时 | `hipEventRecord` / `hipEventElapsedTime` | ROCm |
| 训练计时 | `QueryPerformanceCounter` | Windows |

## 可以获得的数据

- GPU Compute Engine 利用率 (实时 + EMA + Peak)
- GPU Copy Engine 利用率
- HIP Device Allocation / WDDM Dedicated / WDDM Shared 显存
- CPU 进程利用率 / CPU Max Logical Core / 系统内存
- GPU 温度 (ADL)
- Forward / Backward / Target Sync 分阶段耗时 (ms/step)
- Black / White 分方拆分
- Replay 细分 (Sample / Forward / Target / Train(GPU) / Sync)
- Backward 分层 (FC4/FC3/FC2/FC1/Pool/Attention/Embedding)
- Train Step Overhead (Setup/Memset/Forward/Huber/Backward/Validate)
- Steps/s, ms/step, Episodes/s, s/ep
- 统计分布 (mean/median/P5/P95/P99/stddev/min/max)

## Windows 下无法获得的数据

| 指标 | 原因 |
|------|------|
| GFX Clock / Memory Clock | ADL Overdrive 接口未暴露 |
| GPU Power | ADL 不支持 |
| GPU Hotspot | ADL 不支持 |
| Throttle State | ADL 不支持 |
| VALU / SALU 利用率 | 需要 rocprof (Linux only) |
| CU/WGP Activity | 需要 rocprof (Linux only) |
| Occupancy | 需要 rocprof (Linux only) |
| DRAM 读写带宽 | 需要 rocprof (Linux only) |
| Cache hit/miss | 需要 rocprof (Linux only) |
| Memory Stall | 需要 rocprof (Linux only) |

## 性能开销

- 采样线程: ~0.1% CPU (1Hz NORMAL 模式)
- 共享内存: 256 KB (4096 samples × 64 bytes + header)
- hipEvent 计时: ~1-5 µs / 次 (相比 kernel 0.1-5ms 可忽略)
- QPC 计时: ~0.01 µs / 次
- 二进制日志: ~44 bytes / 记录

## IPC 结构

- 共享内存 `DQN_GPU_Telemetry`: seqlock, 4096×64B ring buffer
- 共享内存 `DQN_GPU_KernelTimer`: seqlock, 16384×24B ring buffer
- 命名管道 `\\.\pipe\dqn_control_pipe`: JSON 状态推送 (~10Hz)

## 数据格式

- 二进制日志: `metric_log_format.h` 兼容 (24B header + 8B prefix + N×4B float)
- JSON 报告: `performance_report_YYYYMMDD_HHMMSS.json`
- CSV 导出: `--csv` 命令行选项

## 已知问题

1. `hipDeviceProp_t.totalGlobalMem` 在 APU 上不准确 (包含共享内存), 应使用 DXGI 值
2. PDH `Total Dedicated Usage` 计数器在某些系统不可用, 回退到 `hipMemGetInfo`
3. GPU Engine PDH 需要两次采集取差值, 首次采样无数据
4. CPU 使用率首次采样为基线, 第二次开始有效
5. ADL 温度在 gfx1151 上可用但接口有限 (无 Hotspot/Power)

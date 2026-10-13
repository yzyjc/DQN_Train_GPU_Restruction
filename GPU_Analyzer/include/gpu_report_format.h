/* gpu_report_format.h — GPU 性能报告格式定义
   报告文件格式、JSON 结构、CSV 列定义。 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 报告文件名模板 ========== */
#define GPU_REPORT_PREFIX       "performance_report"
#define GPU_REPORT_JSON_SUFFIX  ".json"
#define GPU_REPORT_CSV_SUFFIX   ".csv"
#define GPU_REPORT_BIN_SUFFIX   ".bin"

/* ========== CSV 列定义 ========== */
#define GPU_CSV_HEADER \
    "episode,step,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb," \
    "gpu_temp_c,gpu_power_w,gpu_clock_mhz,cpu_usage_pct," \
    "sys_mem_used_gb,sys_mem_total_gb,gpu_memory_clock_mhz,gpu_hotspot_temp_c"

#define GPU_CSV_FORMAT \
    "%d,%d,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.1f,%.1f\n"

/* ========== JSON 报告结构说明 (用于 gpu_report.cpp 的生成逻辑) ==========
 *
 * {
 *   "session": {
 *     "start": "ISO8601",
 *     "end": "ISO8601",
 *     "duration_seconds": N,
 *     "mode": "NORMAL|TRACE|DEEP_PROFILE",
 *     "sample_count": N,
 *     "sample_interval_ms": N
 *   },
 *   "device": {
 *     "name": "...",
 *     "arch": "gfx1151",
 *     "vram_mb": N,
 *     "compute_units": N,
 *     "clock_mhz": N,
 *     "memory_clock_mhz": N,
 *     "memory_bus_width": N,
 *     "theoretical_bandwidth_gb_s": N
 *   },
 *   "gpu_utilization": {
 *     "mean": N, "median": N, "p5": N, "p95": N, "p99": N,
 *     "stddev": N, "min": N, "max": N, "samples": N
 *   },
 *   "vram_usage_mb": { ... same fields ... },
 *   "cpu_usage_pct": { ... same fields ... },
 *   "temperature_c": { ... or "unavailable": true ... },
 *   "power_w": { ... or "unavailable": true ... },
 *   "clock_mhz": { ... or "unavailable": true ... },
 *   "kernel_timing": {  // Phase 2 only
 *     "forward": { "count": N, "total_ms": N, "mean_ms": N, ... },
 *     "train_step": { ... },
 *     "duty_cycle_pct": N,
 *     "inter_kernel_gap_ms": { "mean": N, "median": N, "p95": N }
 *   },
 *   "throughput": {  // Phase 2 only
 *     "episodes_per_hour": N,
 *     "forward_calls_per_second": N,
 *     "train_calls_per_second": N
 *   },
 *   "raw_counters": {
 *     "available": ["gpu_util", "vram", "cpu", "sys_mem"],
 *     "unavailable": ["valu", "cache_hit", "occupancy"],
 *     "notes": { "hotspot_temperature_c": "unavailable when ADLX does not support it" }
 *   }
 * }
 */

/* ========== 硬件对比格式 ==========
 *
 * comparison.json:
 * {
 *   "devices": [
 *     { "name": "...", "arch": "...", "report_file": "report_xxx.json" },
 *     ...
 *   ],
 *   "metrics": {
 *     "gpu_util_mean": { "device1": N, "device2": N, "delta_pct": N },
 *     "forward_mean_ms": { ... },
 *     "train_mean_ms": { ... },
 *     "duty_cycle_pct": { ... },
 *     "episodes_per_hour": { ... }
 *   }
 * }
 */

/* ========== 辅助常量 ========== */
#define GPU_REPORT_TIMESTAMP_BUF_LEN 32  /* "YYYY-MM-DDTHH:MM:SS" + 余量 */
#define GPU_REPORT_MAX_PATH 512

#ifdef __cplusplus
}
#endif

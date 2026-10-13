# PERFORMANCE REPORT — AMD Radeon 8060S (gfx1151)

> 生成时间: 2026-09-26
> 数据来源: GPU_Analyzer + Trainer IPC + TUI GPU Analyze

## SYSTEM

| 项目 | 值 |
|------|-----|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | AMD Radeon(TM) 8060S Graphics |
| Architecture | gfx1151 / RDNA 3.5 |
| Compute Units | 20 |
| GPU Clock | 2900 MHz |
| Memory Clock | 800 MHz |
| Memory Bus | 512 bit |
| Theoretical BW | 102.4 GB/s |
| L2 Cache | 2048 KB |
| Warp Size | 32 |
| WDDM Dedicated | 31.8 GB |
| WDDM Shared | 76.5 GB |
| Driver | AMD 32.0.31035.1003 |
| HIP/ROCm | 7.13.0 (D:\AMD_ROCm) |
| CPU | AMD Ryzen AI Max (Strix Point) |
| RAM | 95.6 GB |

## TRAINING CONFIG

| 项目 | 值 |
|------|-----|
| Network | Token Embed(16) + Attention(4h,32d) + Flatten(400) + FC(128→1024→512→256→25) |
| Batch Size | 32 |
| Replay Capacity | 20000 |
| Train Step Each | 4 |
| Episode | ~33000-34500 (Phase 1) |
| Steps/s | 5.4-7.4 |
| ms/step | 159-223 |

## THROUGHPUT

| 指标 | Mean | P5 | P95 | P99 | Max |
|------|------|-----|------|------|------|
| GPU Utilization | 43.6% | 7.8% | 50.7% | 63.9% | 70.8% |
| Episodes/s | 0.22-0.62 | — | — | — | — |
| s/ep | 2.72-4.60 | — | — | — | — |

## GPU TIMELINE (训练步分解)

```
Total step: 159-223 ms
  ├── Forward        5-8 ms   (3-4%)
  └── Backward     152-216 ms  (96-97%)
```

### Backward 分解

```
Backward 202.97 ms
  ├── backward_update (GPU kernels): 32 × 1.46 = 46.7 ms (23%)
  └── 其余开销: ~156 ms (77%)
       ├── Memset(6x)     17.5 ms  ← 最大单项
       ├── Setup(copy)     0.45 ms
       ├── Forward        0.7 ms
       ├── Huber          0.07 ms
       └── Validate       0.3 ms
```

### Backward 每层耗时 (per call)

| 层 | 耗时 | 占比 |
|----|------|------|
| FC4 | 0.24 ms | 16% |
| FC3 | 0.13 ms | 9% |
| FC2 | 0.18 ms | 12% |
| FC1 | 0.25 ms | 17% |
| Pool | 0.10 ms | 7% |
| **Attention** | **0.47 ms** | **32%** |
| Embedding | 0.09 ms | 6% |

### 按方拆分

| 方 | Forward | Backward | Steps |
|----|---------|----------|-------|
| Black | 3-5 ms | 82-110 ms | 12-24 |
| White | 3-5 ms | 80-110 ms | 12-24 |

## MEMORY / COMPUTE

| 指标 | 值 | 来源 |
|------|-----|------|
| HIP Device Alloc | 155 MB | hipMemGetInfo |
| WDDM Dedicated | 31.8 GB | DXGI |
| WDDM Shared | 76.5 GB | DXGI |
| GPU Copy Engine | 1.2-1.6% | PDH |
| CPU Process | 12-18% | GetSystemTimes |
| CPU Max Logical Core | 56-91% | PDH |
| System RAM | 22-23 / 95.6 GB | GlobalMemoryStatusEx |
| GPU Temperature | N/A | ADL (gfx1151 接口有限) |
| GPU Power | N/A | ADL 不支持 |
| GFX/Memory Clock | N/A | ADL 不支持 |

## BLACK / WHITE BREAKDOWN

两方训练耗时基本对称:
- Forward: Black 3-5 ms, White 3-5 ms
- Backward: Black 82-110 ms, White 80-110 ms
- Steps: 各 12-24 步/采样窗口

## STATISTICS

### GPU Utilization
- Mean: 43.6%, Median: 44.7%
- P5: 7.8%, P95: 50.7%, P99: 63.9%
- Min: 0%, Max: 70.8%
- Stddev: 11.25%
- Samples: 16020

### Training Step
- Mean: ~200 ms/step
- Forward: ~7 ms (3.5%)
- Backward: ~203 ms (96.5%)

## RAW COUNTER AVAILABILITY

| 指标 | 状态 | 来源 |
|------|------|------|
| GPU Compute Engine | ✅ AVAILABLE | PDH |
| GPU Copy Engine | ✅ AVAILABLE | PDH |
| VRAM (HIP) | ✅ AVAILABLE | hipMemGetInfo |
| VRAM (WDDM) | ✅ AVAILABLE | DXGI |
| CPU/RAM | ✅ AVAILABLE | Windows API |
| Temperature | ✅ AVAILABLE | ADL (有限) |
| Forward/Backward timing | ✅ AVAILABLE | QPC instrumentation |
| Per-side breakdown | ✅ AVAILABLE | QPC |
| Replay breakdown | ✅ AVAILABLE | QPC |
| Backward per-layer | ✅ AVAILABLE | QPC |
| Train step overhead | ✅ AVAILABLE | QPC |
| Kernel timing | ✅ AVAILABLE | hipEvent |
| GFX Clock | ❌ UNAVAILABLE | ADL 无此接口 |
| Memory Clock | ❌ UNAVAILABLE | ADL 无此接口 |
| GPU Power | ❌ UNAVAILABLE | ADL 不支持 |
| GPU Hotspot | ❌ UNAVAILABLE | ADL 不支持 |
| Throttle State | ❌ UNAVAILABLE | ADL 不支持 |
| VALU utilization | ❌ N/A (Windows) | 需要 rocprof |
| CU/WGP Activity | ❌ N/A (Windows) | 需要 rocprof |
| Occupancy | ❌ N/A (Windows) | 需要 rocprof |
| DRAM bandwidth | ❌ N/A (Windows) | 需要 rocprof |
| Cache hit rate | ❌ N/A (Windows) | 需要 rocprof |
| Memory stalls | ❌ N/A (Windows) | 需要 rocprof |

## NOTES

1. **Backward 占 96.4%** — 按当前 instrumentation，训练更新阶段几乎完全由 backward 主导
2. **Memset(6x) 是最大单项开销** — 每 batch 32 样本 × 6 次 hipMemset = 192 次，共 17-18 ms
3. **Attention backward 最贵** — 0.47 ms/call，占 backward_update 的 32%
4. **两方对称** — Black/White 的 forward/backward 耗时基本相同
5. **GPU 44% 利用率** — 与 Forward 3% + Backward GPU 工作 + memset 间隙吻合
6. **温度/功耗/时钟不可用** — ADL 在 gfx1151 (RDNA 3.5) 上接口有限，非 RDNA4 问题

## 下一步

1. 优化 Memset(6x): 合并为 1 次调用或用清零 kernel
2. 减少 hipDeviceSynchronize: backward_update 每阶段都 sync，可批量末尾一次
3. Attention 优化: 融合 kernel 减少 launch 开销
4. MI50 对比: 相同 benchmark session 格式
